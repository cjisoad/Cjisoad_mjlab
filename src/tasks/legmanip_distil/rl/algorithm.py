"""Online DAgger over complete recurrent sequences using RSL-RL storage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import random
from typing import Callable

import numpy as np
import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .model import StudentModel


def sequence_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """Sum the 12 action errors per frame and average across all frames/envs."""
  if prediction.shape != target.shape or prediction.shape[-1] != 12:
    raise ValueError("L1 distillation expects matching predictions/labels with 12 actions.")
  return (prediction - target).abs().sum(dim=-1).mean()


class LegManipDistillation(Distillation):
  """Distillation with episode-level DAgger and isolated GRU replay memory.

  ``storage.actions`` records the selected policy input before the environment's
  final action clamp. Supervision uses ``privileged_actions`` (native teacher
  requests); the next observation contains the actually applied common action.
  Terminal auto-resets make post-step action buffers unsuitable for recovery.
  """

  def __init__(
    self,
    student: StudentModel,
    teacher: torch.nn.Module,
    storage: RolloutStorage,
    *,
    num_learning_epochs: int = 8,
    num_mini_batches: int = 2,
    gradient_length: int = 32,
    learning_rate: float = 3.0e-4,
    max_grad_norm: float = 1.0,
    loss_type: str = "l1",
    optimizer: str = "adamw",
    weight_decay: float = 0.01,
    warmup_updates: int = 1000,
    decay_updates: int = 2000,
    teacher_probability: float | None = None,
    device: str = "cpu",
    multi_gpu_cfg: dict | None = None,
    rnd_cfg: None = None,
    symmetry_cfg: None = None,
  ) -> None:
    if storage.training_type != "distillation":
      raise ValueError("LegManipDistillation requires RolloutStorage('distillation').")
    if num_mini_batches < 1 or storage.num_envs < num_mini_batches:
      raise ValueError("num_mini_batches must be positive and no larger than num_envs.")
    if storage.num_envs % num_mini_batches:
      raise ValueError("num_envs must be divisible by num_mini_batches; no environments may be dropped.")
    if gradient_length < 1 or gradient_length != storage.num_transitions_per_env:
      raise ValueError("gradient_length must equal num_steps_per_env for complete-sequence TBPTT.")
    if num_learning_epochs < 1:
      raise ValueError("num_learning_epochs must be positive.")
    if teacher_probability is not None and not 0.0 <= teacher_probability <= 1.0:
      raise ValueError("teacher_probability must be between 0 and 1.")
    if warmup_updates < 0 or decay_updates < 1:
      raise ValueError("warmup_updates must be nonnegative and decay_updates must be positive.")
    if loss_type != "l1" or optimizer.lower() != "adamw":
      raise ValueError("The distillation recipe requires L1 loss and AdamW.")
    if learning_rate < 0 or weight_decay < 0 or max_grad_norm <= 0:
      raise ValueError("Learning rate/weight decay must be nonnegative and max_grad_norm must be positive.")
    if rnd_cfg is not None or symmetry_cfg is not None:
      raise ValueError("RND and symmetry extensions are not supported by FSM distillation.")
    if not getattr(teacher, "loaded", False):
      raise ValueError("All frozen FSM teacher policies must be loaded before distillation.")
    # RSL-RL 5 initializes optimizer, storage, devices and distributed utilities.
    # Its built-in losses are only MSE/Huber, so install the task's L1 below.
    super().__init__(
      student, teacher, storage, num_learning_epochs=num_learning_epochs,
      gradient_length=gradient_length, learning_rate=learning_rate,
      max_grad_norm=max_grad_norm, loss_type="mse", optimizer="adamw",
      device=device, multi_gpu_cfg=multi_gpu_cfg,
    )
    self.loss_fn = sequence_l1_loss
    for group in self.optimizer.param_groups:
      group["weight_decay"] = weight_decay
    self.teacher.requires_grad_(False)
    self.teacher.eval()
    self.teacher_loaded = True
    self.num_mini_batches = num_mini_batches
    self.warmup_updates = warmup_updates
    self.decay_updates = decay_updates
    self.teacher_probability = teacher_probability
    self.teacher_control_mask = torch.zeros(storage.num_envs, dtype=torch.bool, device=device)
    self._pending_control_sample = torch.ones_like(self.teacher_control_mask)
    self.rollout_initial_hidden: torch.Tensor | None = None

  @property
  def beta(self) -> float:
    """Teacher probability for new episodes after ``num_updates`` completions."""
    if self.teacher_probability is not None:
      return self.teacher_probability
    return (1.0 - 1.0 / self.decay_updates) ** max(self.num_updates - self.warmup_updates, 0)

  @torch.no_grad()
  def act(self, obs: TensorDict) -> torch.Tensor:
    if self.storage.step == 0:
      hidden = self.student.get_hidden_state()
      self.rollout_initial_hidden = (
        obs["student"].new_zeros(1, self.storage.num_envs, 256)
        if hidden is None else hidden.detach().clone()
      )
    # Sample one controlling policy per episode; both use the same direct PD.
    pending_ids = self._pending_control_sample.nonzero(as_tuple=False).flatten()
    if pending_ids.numel():
      self.teacher_control_mask[pending_ids] = torch.rand(pending_ids.numel(), device=self.device) < self.beta
      self._pending_control_sample[pending_ids] = False
    # Advance the student on teacher-controlled frames too, and supervise the
    # native teacher request on whichever physical states are actually visited.
    prediction = self.student(obs)
    labels = self.teacher(obs).detach().clone()
    if prediction.shape != labels.shape or labels.shape != (self.storage.num_envs, 12):
      raise ValueError("Student and FSM teacher must both produce [num_envs, 12] actions.")
    actions = torch.where(self.teacher_control_mask[:, None], labels, prediction)
    # Snapshot before env.step: manager buffers may be overwritten in-place.
    self.transition.observations = obs.select("student").clone()
    self.transition.actions = actions.detach()
    self.transition.privileged_actions = labels
    return actions

  @torch.no_grad()
  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> None:
    self.transition.rewards = rewards
    self.transition.dones = dones
    self.storage.add_transition(self.transition)
    self.transition.clear()
    done_mask = dones.reshape(-1).bool()
    self.student.reset(done_mask)
    self.teacher.reset(done_mask)
    self._pending_control_sample |= done_mask

  def update(self) -> dict[str, float]:
    if self.storage.step != self.storage.num_transitions_per_env or self.rollout_initial_hidden is None:
      raise ValueError("A complete rollout is required before a distillation update.")
    # Rollout runs in inference_mode. Clone here, outside that context, to make
    # ordinary tensors autograd can save. Replay never changes live RNN memory.
    initial_hidden = self.rollout_initial_hidden.detach().clone()
    mini_batch_size = self.storage.num_envs // self.num_mini_batches
    loss_sum = 0.0
    for _ in range(self.num_learning_epochs):
      indices = torch.randperm(self.storage.num_envs, device=self.device)
      for ids in indices.split(mini_batch_size):
        values = self.storage.observations["student"][:, ids]
        targets = self.storage.privileged_actions[:, ids]
        dones = self.storage.dones[:, ids, 0].bool()
        prediction, _ = self.student.forward_sequence(values, dones, initial_hidden[:, ids])
        loss = self.loss_fn(prediction, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        loss_sum += loss.detach().item()
    self.storage.clear()
    self.rollout_initial_hidden = None
    self.num_updates += 1
    return {
      "behavior": loss_sum / (self.num_learning_epochs * self.num_mini_batches),
      "teacher_probability": self.beta,
      "teacher_control_fraction": self.teacher_control_mask.float().mean().item(),
    }

  def reset_rollout_state(self) -> None:
    """Start fresh episodes after loading; a checkpoint does not restore physics."""
    self.student.reset()
    self.teacher.reset()
    self.teacher_control_mask.zero_()
    self._pending_control_sample.fill_(True)
    self.rollout_initial_hidden = None
    self.transition.clear()
    self.storage.clear()

  def save(self) -> dict:
    """Save learning state, excluding frozen external teachers and episode state."""
    rng = {
      "torch": torch.get_rng_state(),
      "numpy": np.random.get_state(),
      "python": random.getstate(),
    }
    if torch.cuda.is_initialized():
      rng["cuda"] = torch.cuda.get_rng_state_all()
    return {
      "student_state_dict": self.student.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "completed_updates": self.num_updates,
      "dagger_state": {
        "warmup_updates": self.warmup_updates,
        "decay_updates": self.decay_updates,
        "teacher_probability": self.teacher_probability,
      },
      "rng_state": rng,
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
    options = {"student": True, "optimizer": True, "iteration": True} if load_cfg is None else load_cfg
    if options.get("student", False):
      self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
    if options.get("optimizer", False):
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
      self.learning_rate = self.optimizer.param_groups[0]["lr"]
    restore_iteration = bool(options.get("iteration", False))
    if restore_iteration:
      self.num_updates = int(loaded_dict["completed_updates"])
      schedule = loaded_dict["dagger_state"]
      self.warmup_updates = int(schedule["warmup_updates"])
      self.decay_updates = int(schedule["decay_updates"])
      self.teacher_probability = schedule["teacher_probability"]
      rng = loaded_dict.get("rng_state", {})
      if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
      if "numpy" in rng:
        np.random.set_state(rng["numpy"])
      if "python" in rng:
        random.setstate(rng["python"])
      if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    self.reset_rollout_state()
    return restore_iteration

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env,
    cfg: dict,
    device: str,
    *,
    teacher_factory: Callable | None = None,
  ) -> LegManipDistillation:
    """Use the real FSM by default; allow an injected teacher for CPU checks."""
    if env.num_actions != 12:
      raise ValueError("The legmanip_distil environment must expose 12 actions.")
    if cfg.get("clip_actions") is not None:
      raise ValueError("legmanip_distil requires unclipped deterministic common actions.")
    groups = cfg.get("obs_groups", {"student": ("student",)})
    if set(groups) != {"student"} or tuple(groups["student"]) != ("student",):
      raise ValueError("Distillation uses only the student observation group.")
    student_cfg = cfg.get("student", {})
    student_cfg = asdict(student_cfg) if is_dataclass(student_cfg) else deepcopy(student_cfg)
    student_cfg.pop("class_name", None)
    output_dim = student_cfg.pop("output_dim", 12)
    student = StudentModel(obs.to(device), groups, "student", output_dim=output_dim, **student_cfg)
    teacher_cfg = cfg.get("teacher", {})
    teacher_cfg = asdict(teacher_cfg) if is_dataclass(teacher_cfg) else deepcopy(teacher_cfg)
    teacher_cfg.pop("class_name", None)
    if teacher_factory is None:
      from ..teacher import FSMTeacher

      teacher_factory = FSMTeacher
    teacher = teacher_factory(env.unwrapped, **teacher_cfg)
    storage = RolloutStorage(
      "distillation", env.num_envs, cfg["num_steps_per_env"], obs.select("student"), [12], device,
    )
    algorithm_cfg = cfg["algorithm"]
    algorithm_cfg = asdict(algorithm_cfg) if is_dataclass(algorithm_cfg) else deepcopy(algorithm_cfg)
    algorithm_cfg.pop("class_name", None)
    return LegManipDistillation(
      student, teacher, storage, device=device, multi_gpu_cfg=cfg.get("multi_gpu"), **algorithm_cfg,
    )
