"""Go2 rear-leg standing policy configuration and PPO compatibility adapter."""

from .config import stand_ppo_runner_cfg
from .runner import StandOnPolicyRunner

__all__ = ["StandOnPolicyRunner", "stand_ppo_runner_cfg"]
