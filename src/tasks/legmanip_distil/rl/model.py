"""Deterministic GRU student and explicit-state sequence replay."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from .config import StudentModelCfg


class StudentModel(RNNModel):
  """RSL-RL GRU with a fixed 48-observation, 12-action contract.

  ``forward`` maintains live rollout memory. ``forward_sequence`` uses explicit
  hidden tensors and leaves that memory untouched, including during training.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str] | tuple[str, ...]] | None = None,
    obs_set: str = "student",
    output_dim: int = 12,
    *,
    input_dim: int = 48,
    hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    rnn_type: str = "gru",
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
  ) -> None:
    groups = obs_groups or {"student": ("student",)}
    if obs_set != "student" or tuple(groups.get(obs_set, ())) != ("student",):
      raise ValueError("The student must use only TensorDict['student'] observations.")
    if input_dim != 48 or output_dim != 12 or obs["student"].ndim != 2 or obs["student"].shape[-1] != 48:
      raise ValueError("The student contract requires 48 observations and 12 actions.")
    if (
      tuple(hidden_dims) != (256, 128)
      or activation != "elu"
      or rnn_type != "gru"
      or rnn_hidden_dim != 256
      or rnn_num_layers != 1
      or obs_normalization
      or distribution_cfg is not None
    ):
      raise ValueError("The student requires GRU(256, 1), MLP(256, 128), ELU, fixed scales and deterministic outputs.")
    super().__init__(
      obs, groups, obs_set, output_dim, hidden_dims=hidden_dims, activation=activation,
      obs_normalization=False, distribution_cfg=None, rnn_type=rnn_type,
      rnn_hidden_dim=rnn_hidden_dim, rnn_num_layers=rnn_num_layers,
    )
    self.model_cfg = asdict(StudentModelCfg())

  @property
  def output_std(self) -> torch.Tensor:
    """RSL-RL's standard logger asks deterministic policies for this statistic."""
    return next(self.parameters()).new_zeros(12)

  def forward_sequence(
    self,
    observations: torch.Tensor | TensorDict,
    dones: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay [time, env, 48] and reset state *after* each terminal frame.

    Contiguous spans without a done share one GRU call. All frames remain in the
    loss, and out-of-place masking cuts gradients across episode boundaries.
    """
    values = observations["student"] if isinstance(observations, TensorDict) else observations
    if values.ndim != 3 or values.shape[-1] != 48 or values.shape[0] == 0:
      raise ValueError("Expected nonempty student observations of shape [time, env, 48].")
    done_mask = dones.squeeze(-1) if dones.ndim == 3 else dones
    if done_mask.shape != values.shape[:2]:
      raise ValueError("Done flags must have shape [time, env].")
    hidden = values.new_zeros(1, values.shape[1], 256) if hidden_state is None else hidden_state
    if hidden.shape != (1, values.shape[1], 256):
      raise ValueError("GRU hidden state must have shape [1, env, 256].")
    # The final endpoint also flushes an incomplete span after the last done.
    endpoints = (done_mask.bool().any(dim=1).nonzero(as_tuple=False).flatten() + 1).tolist()
    if not endpoints or endpoints[-1] != values.shape[0]:
      endpoints.append(values.shape[0])
    outputs = []
    start = 0
    for stop in endpoints:
      latent, hidden = self.rnn.rnn(values[start:stop], hidden)
      outputs.append(self.mlp(latent))
      hidden = hidden * (~done_mask[stop - 1].bool()).to(hidden.dtype)[None, :, None]
      start = stop
    return torch.cat(outputs, dim=0), hidden


def make_student(
  num_envs: int = 1,
  device: str | torch.device = "cpu",
  cfg: StudentModelCfg | dict[str, Any] | None = None,
) -> StudentModel:
  """Construct a student without an environment or teacher policy."""
  if num_envs < 1:
    raise ValueError("num_envs must be positive.")
  model_cfg = asdict(StudentModelCfg())
  if cfg is not None:
    model_cfg.update(asdict(cfg) if is_dataclass(cfg) else dict(cfg))
  model_cfg.pop("class_name", None)
  obs = TensorDict({"student": torch.zeros(num_envs, 48, device=device)}, batch_size=[num_envs])
  return StudentModel(obs, **model_cfg).to(device)


def save_student(path: str | Path, student: StudentModel, metadata: dict) -> None:
  """Write a CPU state-dict artifact that needs no teacher files to load."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "format": "legmanip_distil.student",
    "version": 1,
    "student_cfg": student.model_cfg,
    "student_state_dict": {name: tensor.detach().cpu() for name, tensor in student.state_dict().items()},
    "infos": {"legmanip_distil": metadata},
  }, path)


def load_student(
  path: str | Path,
  device: str | torch.device = "cpu",
  num_envs: int = 1,
) -> tuple[StudentModel, dict]:
  """Load a standalone student (or a task runner checkpoint) for inference.

  Returns the model and its ``infos['legmanip_distil']`` manifest. Episode memory
  starts empty; callers reset done environments after each environment step.
  """
  artifact = torch.load(path, weights_only=False, map_location=device)
  metadata = artifact.get("infos", {}).get("legmanip_distil", {})
  if metadata.get("version") != 1:
    raise ValueError("Unsupported or missing legmanip_distil checkpoint version.")
  if "format" in artifact and artifact["format"] != "legmanip_distil.student":
    raise ValueError("Unsupported student artifact format.")
  if artifact.get("version", 1) != 1:
    raise ValueError("Unsupported student artifact version.")
  cfg = artifact.get("student_cfg", metadata.get("student_cfg"))
  if not isinstance(cfg, dict) or cfg.get("input_dim") != 48 or cfg.get("output_dim") != 12:
    raise ValueError("Student artifact must declare 48 observations and 12 actions.")
  student = make_student(num_envs=num_envs, device=device, cfg=cfg)
  student.load_state_dict(artifact["student_state_dict"], strict=True)
  student.eval()
  return student, metadata
