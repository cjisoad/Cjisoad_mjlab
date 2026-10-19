"""Configuration for online recurrent FSM distillation (RSL-RL 5)."""

from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class StudentModelCfg:
  """Fixed observation scales are applied by the environment before this model."""

  class_name: str = "src.tasks.legmanip_distil.rl.model:StudentModel"
  input_dim: int = 48
  output_dim: int = 12
  hidden_dims: tuple[int, ...] = (256, 128)
  activation: str = "elu"
  obs_normalization: bool = False
  distribution_cfg: dict[str, Any] | None = None
  rnn_type: str = "gru"
  rnn_hidden_dim: int = 256
  rnn_num_layers: int = 1


@dataclass
class LegManipDistillationAlgorithmCfg:
  class_name: str = "src.tasks.legmanip_distil.rl.algorithm:LegManipDistillation"
  num_learning_epochs: int = 8
  num_mini_batches: int = 2
  gradient_length: int = 32
  learning_rate: float = 3.0e-4
  optimizer: str = "adamw"
  weight_decay: float = 0.01
  max_grad_norm: float = 1.0
  loss_type: str = "l1"
  warmup_updates: int = 1000
  decay_updates: int = 2000
  teacher_probability: float | None = None
  """Optional constant probability for smoke tests; None uses the DAgger schedule."""
  rnd_cfg: None = None
  symmetry_cfg: None = None


@dataclass
class LegManipDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "src.tasks.legmanip_distil.rl.runner:LegManipDistillationRunner"
  experiment_name: str = "legmanip_distil"
  num_steps_per_env: int = 32
  max_iterations: int = 60000
  save_interval: int = 500
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  clip_actions: float | None = None
  obs_groups: dict[str, tuple[str, ...]] = field(default_factory=lambda: {"student": ("student",)})
  student: StudentModelCfg = field(default_factory=StudentModelCfg)
  teacher: dict[str, Any] = field(default_factory=dict)
  algorithm: LegManipDistillationAlgorithmCfg = field(default_factory=LegManipDistillationAlgorithmCfg)


def legmanip_distil_runner_cfg() -> LegManipDistillationRunnerCfg:
  return LegManipDistillationRunnerCfg()
