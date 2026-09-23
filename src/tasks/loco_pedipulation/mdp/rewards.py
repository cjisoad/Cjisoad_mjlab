"""Mode-aware whole-body locomotion and FR tracking rewards."""

import torch

from src.tasks.velocity.mdp.rewards import variable_posture

from .anchor_frame import to_anchor
from .commands import HOLD, LOWER, QUAD


def _term(env):
  return env.command_manager.get_term('twist')


def track_linear_velocity(env, std=.5):
  term = _term(env)
  actual = to_anchor(term.basis_w, env.scene['robot'].data.root_link_lin_vel_w)
  error = (term.command[:, :2] - actual[:, :2]).square().sum(-1) + 2. * actual[:, 2].square()
  return torch.exp(-error / std**2)


def track_angular_velocity(env, std=.70710678):
  term = _term(env)
  actual = to_anchor(term.basis_w, env.scene['robot'].data.root_link_ang_vel_w)
  error = (term.command[:, 2] - actual[:, 2]).square() + .05 * actual[:, :2].square().sum(-1)
  return torch.exp(-error / std**2)


class mode_posture(variable_posture):
  def __call__(self, env, std_standing, std_walking, std_running, asset_cfg,
               command_name, walking_threshold=.1, running_threshold=1.5):
    quad = super().__call__(env, std_standing, std_walking, std_running, asset_cfg,
                           command_name, walking_threshold, running_threshold)
    term = _term(env)
    robot = env.scene[asset_cfg.name]
    error = (robot.data.joint_pos - robot.data.default_joint_pos).square()
    mask = torch.ones_like(error)
    mask[:, term.fr_joint_ids] = 0.
    tripod = torch.exp(-((error / (1.5 * self.std_walking)**2) * mask).sum(-1) / mask.sum(-1))
    return torch.lerp(quad, tripod, term.blend)


def stand_still(env):
  term = _term(env)
  robot = env.scene['robot']
  error = (robot.data.joint_pos - robot.data.default_joint_pos).square()
  error[:, term.fr_joint_ids] *= (1. - term.blend[:, None])
  # Support joints must be free to reposition while the robot balances on three legs.
  return error.sum(-1) * (term.command.norm(dim=-1) < .05) * (1. - term.blend)


def feet_gait(env):
  term = _term(env)
  phase = term.gait_phase[:, None]
  contact = term.contacts
  # Contact and site arrays are explicitly ordered FL, FR, RL, RR.
  quad_stance = ((phase + phase.new_tensor((.5, 0., 0., .5))) % 1.) < .56
  tripod_stance = ((phase + phase.new_tensor((0., 0., 2. / 3., 1. / 3.))) % 1.) < .8
  quad = (quad_stance == contact).float().mean(-1)
  weights = torch.ones_like(contact, dtype=torch.float32)
  weights[:, term.fr_index] = 0.
  tripod = ((tripod_stance == contact).float() * weights).sum(-1) / 3.
  moving = term.command.norm(dim=-1) > .05
  return torch.lerp(quad, tripod, term.blend) * moving


def foot_clearance(env, target_height=.1):
  term = _term(env)
  robot = env.scene['robot']
  height = robot.data.site_pos_w[:, term.site_ids, 2]
  speed = robot.data.site_lin_vel_w[:, term.site_ids, :2].norm(dim=-1)
  cost = (height - target_height).abs() * speed * term.support_weights
  return cost.sum(-1) * (term.command.norm(dim=-1) > .05)


def foot_slip(env):
  term = _term(env)
  velocity = env.scene['robot'].data.site_lin_vel_w[:, term.site_ids, :2]
  return (velocity.square().sum(-1) * term.contacts * term.support_weights).sum(-1)


def soft_landing(env):
  term = _term(env)
  force = env.scene['feet_ground_contact'].data.force[:, term.contact_order].norm(dim=-1)
  return ((force - 150.).clamp_min(0.) * term.support_weights).sum(-1)


def support_contact(env):
  term = _term(env)
  weights = term.support_weights
  return (term.contacts * weights).sum(-1) / weights.sum(-1).clamp_min(1.) * (term.command.norm(dim=-1) < .05)


def fr_position_tracking(env, std=.05):
  term = _term(env)
  error = (term.foot_pos_anchor - term.reference).square().sum(-1)
  return torch.exp(-error / std**2) * term.blend


def fr_ground_contact(env):
  term = _term(env)
  return term.contacts[:, term.fr_index].float() * (term.phase == HOLD)


def base_height(env, std=.06):
  term = _term(env)
  height = env.scene['robot'].data.root_link_pos_w[:, 2]
  return torch.exp(-((height - term.workspace.nominal_height) / std)**2)


def manipulation_fraction(env):
  term = _term(env)
  term.record_outcomes()
  return (term.phase != QUAD).float()


def landing_failed(env):
  term = _term(env)
  return (term.phase == LOWER) & (term.elapsed > term.cfg.reach_time + term.cfg.landing_timeout)
