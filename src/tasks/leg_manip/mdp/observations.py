"""Actor ABI (unchanged 48D) and the merged privileged critic observations."""

import torch

from src.tasks.loco_pedipulation.mdp import observations as loco_observations
from src.tasks.loco_pedipulation.mdp.observations import (
  base_velocity,
  gait_phase,
  policy_state,
  shared_policy_state,
)
from src.tasks.pedipulation.mdp.observations import domain_parameters

__all__ = (
  'base_velocity', 'domain_parameters', 'foot_control_state', 'gait_phase',
  'policy_state', 'shared_policy_state',
)


def foot_control_state(env):
  """Privileged internal FR/mode state, intentionally excluded from actor input."""
  term = env.command_manager.get_term('twist')
  base = loco_observations.foot_control_state(env)
  return torch.cat((base, term.group_weight[:, None], term.mode.float()[:, None]), dim=-1)
