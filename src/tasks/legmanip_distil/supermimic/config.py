"""Fresh four-teacher world-model student recipe."""
from dataclasses import dataclass, field
from ..rl.config import LegManipDistillationRunnerCfg


@dataclass
class SuperMimicCfg:
    horizon: int = 4
    replay_steps: int = 1024
    model_batch_size: int = 128
    model_updates: int = 4
    model_horizon: int = 4
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    imagination_start_update: int = 1000
    imagination_updates: int = 1
    teacher_anchor: float = 1.
    reward_weight: float = .1
    teacher_probability_floor: float = .1
    joint_updates: int = 4
    gradient_interval: int = 100
    detailed_gradient_interval: int = 500
    validation_interval: int = 100
    max_horizon: int = 8
    validation_batch_size: int = 64
    validation_error_scale: float = .5
    reliable_validations: int = 3
    real_critic_weight: float = 1.
    mirror_augmentation: bool = True
    symmetry_weight: float = .05
    command_relabel_fraction: float = .25
    gamma: float = .99
    gae_lambda: float = .95


@dataclass
class ClearMLCfg:
    enabled: bool = False
    offline: bool = False
    project: str = 'Go2/SuperMimic'
    task_name: str = ''
    checkpoint_interval: int = 1000
    output_uri: str | None = None


@dataclass
class SuperMimicRunnerCfg(LegManipDistillationRunnerCfg):
    class_name: str = 'src.tasks.legmanip_distil.supermimic.training:SuperMimicRunner'
    experiment_name: str = 'legmanip_distil_supermimic'
    supermimic: SuperMimicCfg = field(default_factory=SuperMimicCfg)
    clearml: ClearMLCfg = field(default_factory=ClearMLCfg)

    def __post_init__(self):
        self.algorithm.class_name = 'src.tasks.legmanip_distil.supermimic.training:SuperMimicAlgorithm'
