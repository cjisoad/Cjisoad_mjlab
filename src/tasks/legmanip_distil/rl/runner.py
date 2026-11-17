"""RSL-RL runner with completed-update checkpoints and teacher-free exports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
import time

import torch
from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import check_nan

from .algorithm import LegManipDistillation
from .model import save_student


def _teacher_identity(value):
  """Compare content hashes and behavioral settings independently of location."""
  if isinstance(value, dict):
    return {key: _teacher_identity(item) for key, item in value.items() if key != "path"}
  if isinstance(value, (tuple, list)):
    return [_teacher_identity(item) for item in value]
  return value


class LegManipDistillationRunner(DistillationRunner):
  """Use RSL-RL's algorithm factory/logger and preserve FSM command clocks."""

  alg: LegManipDistillation

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    cfg = asdict(train_cfg) if is_dataclass(train_cfg) else deepcopy(train_cfg)
    if cfg.get("save_interval", 0) < 1:
      raise ValueError("save_interval must be positive.")
    super().__init__(env, cfg, log_dir, device)
    self.logger.writer = None

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Train additional updates; never randomize episode/FSM command clocks.

    The base RSL-RL loop stores the last zero-based iteration in ``iter``. This
    loop stores completed updates so warmup, beta, save intervals and resumes
    all use the same counter. Collection and logging retain RSL-RL semantics.
    """
    if not self.alg.teacher_loaded:
      raise ValueError("Frozen FSM teachers must be loaded before learning.")
    if num_learning_iterations < 0:
      raise ValueError("num_learning_iterations must be nonnegative.")
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()
    start_it = self.alg.num_updates
    total_it = start_it + num_learning_iterations
    try:
      while self.alg.num_updates < total_it:
        started = time.perf_counter()
        with torch.inference_mode():
          for _ in range(self.cfg["num_steps_per_env"]):
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            if self.cfg.get("check_for_nan", True):
              check_nan(obs, rewards, dones)
            obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
            self.alg.process_env_step(obs, rewards, dones, extras)
            self.logger.process_env_step(rewards, dones, extras)
          self.alg.compute_returns(obs)
        collected = time.perf_counter()
        losses = self.alg.update()
        learned = time.perf_counter()
        self.current_learning_iteration = self.alg.num_updates
        self.logger.log(
          it=self.current_learning_iteration - 1, start_it=start_it, total_it=total_it,
          collect_time=collected - started, learn_time=learned - collected,
          loss_dict={"behavior": losses["behavior"]}, learning_rate=self.alg.learning_rate,
          action_std=self.alg.get_policy().output_std, rnd_weight=None,
        )
        if self.logger.writer is not None:
          self.logger.writer.add_scalar("DAgger/teacher_probability", losses["teacher_probability"], self.current_learning_iteration)
          self.logger.writer.add_scalar("DAgger/teacher_control_fraction", losses["teacher_control_fraction"], self.current_learning_iteration)
          for name, value in losses.items():
            if name.startswith("supermimic/"):
              self.logger.writer.add_scalar(name, value, self.current_learning_iteration)
          if self.current_learning_iteration % self.cfg["save_interval"] == 0:
            self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))
      if self.logger.writer is not None:
        self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))
    finally:
      if self.logger.writer is not None and self.logger.logger_type == "tensorboard":
        self.logger.writer.flush()
        self.logger.writer.close()
      else:
        self.logger.stop_logging_writer()

  def _metadata(self) -> dict:
    from ..observations import action_contract, observation_contract

    return {
      "version": 1,
      "observation_contract": observation_contract(),
      "action_contract": action_contract(self.env.unwrapped),
      "student_cfg": deepcopy(self.alg.student.model_cfg),
      "training_cfg": deepcopy(self.cfg),
      "teacher": self.alg.teacher.metadata(),
      "completed_updates": self.alg.num_updates,
    }

  def save(self, path: str, infos: dict | None = None) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    details = deepcopy(infos) if infos is not None else {}
    details["legmanip_distil"] = self._metadata()
    checkpoint = self.alg.save()
    checkpoint.update({
      "iter": self.alg.num_updates,
      "infos": details,
      "logger_state": {"tot_timesteps": self.logger.tot_timesteps, "tot_time": self.logger.tot_time},
    })
    torch.save(checkpoint, path_obj)
    save_student(path_obj.with_name("student.pt"), self.alg.student, details["legmanip_distil"])
    self.logger.save_model(str(path_obj), self.alg.num_updates)

  def load(
    self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
  ) -> dict:
    checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
    metadata = checkpoint.get("infos", {}).get("legmanip_distil", {})
    if metadata.get("version") != 1:
      raise ValueError("Unsupported or missing legmanip_distil checkpoint version.")
    current = self._metadata()
    for field in ("observation_contract", "action_contract", "student_cfg"):
      if metadata.get(field) != current[field]:
        raise ValueError(f"Checkpoint {field} does not match this task configuration.")
    if load_cfg is None or load_cfg.get("iteration", False):
      if _teacher_identity(metadata.get("teacher")) != _teacher_identity(current["teacher"]):
        raise ValueError("Checkpoint teacher assets/settings differ; full resume requires the same frozen teachers.")
    if self.alg.load(checkpoint, load_cfg, strict):
      self.current_learning_iteration = self.alg.num_updates
      for key, value in checkpoint.get("logger_state", {}).items():
        if key in ("tot_timesteps", "tot_time"):
          setattr(self.logger, key, value)
    # Physics, manager histories and command clocks are restarted together.
    # The algorithm deliberately does not restore recurrent state/control masks.
    # mjlab sensors can lazily allocate inference tensors during collection;
    # their in-place reset must run in the same mode when reusing a live world.
    with torch.inference_mode():
      self.env.reset()
    return checkpoint["infos"]
