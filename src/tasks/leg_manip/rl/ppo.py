"""Fused-task PPO with 51D velocity-aware actor observations and symmetry."""

from typing import Any

from rsl_rl.algorithms import PPO

from src.tasks.pedipulation.rl.ppo import PedipulationPPO
from ..constants import ACTOR_OBS_DIM, CRITIC_OBS_DIM
from .symmetry import exact_symmetry_loss


class LegManipPPO(PedipulationPPO):
  """Keep the source update with a task-specific observation mirror."""

  _symmetry_loss = staticmethod(exact_symmetry_loss)

  def __init__(self, *args: Any, sym_coef: float = 1.0, **kwargs: Any) -> None:
    if kwargs.get('symmetry_cfg') is not None:
      raise ValueError('LegManipPPO supplies its own symmetry loss; symmetry_cfg must be None')
    # Skip PedipulationPPO.__init__'s hardcoded 89-dimensional critic check.
    PPO.__init__(self, *args, **kwargs)
    self.sym_coef = sym_coef
    self.validate(self)

  @staticmethod
  def validate(ppo) -> None:
    if ppo.actor.is_recurrent or ppo.critic.is_recurrent:
      raise ValueError('LegManipPPO requires the feed-forward policy')
    if ppo.actor.obs_dim != ACTOR_OBS_DIM or tuple(ppo.actor.obs_groups) != ('actor',):
      raise ValueError(f'LegManipPPO requires one {ACTOR_OBS_DIM}-dimensional actor observation group')
    if ppo.critic.obs_dim != CRITIC_OBS_DIM:
      raise ValueError(f'LegManipPPO requires {CRITIC_OBS_DIM} critic observations')
    if ppo.actor.obs_normalization or ppo.critic.obs_normalization:
      raise ValueError('The fused task does not normalize observations')
    if ppo.actor.distribution.output_dim != 12:
      raise ValueError('LegManipPPO requires 12 canonical joint actions')
