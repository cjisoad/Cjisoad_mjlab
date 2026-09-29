"""Fused-task PPO: PedipulationPPO with the merged critic dimension."""

from typing import Any

from rsl_rl.algorithms import PPO

from src.tasks.pedipulation.rl.ppo import PedipulationPPO


class LegManipPPO(PedipulationPPO):
  """Keep the source symmetry update; relax only the critic width check."""

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
    if ppo.actor.obs_dim != 48 or tuple(ppo.actor.obs_groups) != ('actor',):
      raise ValueError('LegManipPPO requires one 48-dimensional actor observation group')
    if ppo.actor.obs_normalization or ppo.critic.obs_normalization:
      raise ValueError('The fused task does not normalize observations')
    if ppo.actor.distribution.output_dim != 12:
      raise ValueError('LegManipPPO requires 12 canonical joint actions')
