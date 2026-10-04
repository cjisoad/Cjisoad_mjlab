"""51D actor with noisy anchor base velocity and matching exact critic velocity."""

import torch

from src.tasks.loco_pedipulation.mdp import observations as loco_observations
from src.tasks.loco_pedipulation.mdp.observations import (
  gait_phase,
)
from src.tasks.pedipulation.mdp.observations import domain_parameters

from ..constants import BASE_VELOCITY_NOISE, BASE_VELOCITY_SCALE, LEGACY_ACTOR_DIM
from .anchor_frame import to_anchor

__all__ = (
  'base_velocity', 'domain_parameters', 'foot_control_state', 'gait_phase',
  'policy_state', 'shared_policy_state',
)


def base_velocity(env):
  """Exact absolute base-origin velocity, rotated into anchor axes and scaled."""
  term = env.command_manager.get_term('twist')
  return to_anchor(term.basis_w, env.scene['robot'].data.root_link_lin_vel_w) * BASE_VELOCITY_SCALE


def policy_state(env, add_noise=True):
  """Keep the original 48 channels; append velocity in every support mode."""
  original = loco_observations.policy_state(env, add_noise=add_noise)
  velocity = base_velocity(env)
  if add_noise:
    velocity = velocity + (torch.rand_like(velocity) * 2 - 1) * (
      BASE_VELOCITY_NOISE * BASE_VELOCITY_SCALE)
  action = env.action_manager.get_term('joint_pos')
  action.actor_sample = torch.cat((original, velocity), dim=-1)
  return action.actor_sample


def shared_policy_state(env):
  """Share the original noise draw; critic receives exact velocity separately."""
  return env.action_manager.get_term('joint_pos').actor_sample[:, :LEGACY_ACTOR_DIM]


def foot_control_state(env):
  """Privileged internal FR/mode state, intentionally excluded from actor input."""
  term = env.command_manager.get_term('twist')
  base = loco_observations.foot_control_state(env)
  return torch.cat((base, term.group_weight[:, None], term.mode.float()[:, None]), dim=-1)
