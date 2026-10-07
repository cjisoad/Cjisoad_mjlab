"""Teacher-only actor observations shared across teacher configurations."""

import torch

from src.tasks.leg_manip.mdp import observations as leg_observations
from src.tasks.loco_pedipulation.mdp.observations import gait_phase


def loco_policy_state(env, add_noise=True):
  """Append gait phase and quadruped-to-tripod blend to the 51D loco actor."""
  base = leg_observations.policy_state(env, add_noise=add_noise)
  term = env.command_manager.get_term('twist')
  blend = term.blend[:, None]
  action = env.action_manager.get_term('joint_pos')
  action.actor_sample = torch.cat((base, gait_phase(env), blend), dim=-1)
  return action.actor_sample
