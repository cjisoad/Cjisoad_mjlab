"""Go2 rear-leg standing policy configuration and PPO compatibility adapter."""

from .config import pedipulation_ppo_runner_cfg
from .runner import PedipulationOnPolicyRunner

__all__ = ["PedipulationOnPolicyRunner", "pedipulation_ppo_runner_cfg"]
