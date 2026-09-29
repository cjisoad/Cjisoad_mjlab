"""Group-weighted loco/biped reward terms for the fused leg_manip task.

LOCO group terms are the loco_pedipulation functions scaled by ``1 - w``;
BIPED group terms are the Pedipulation functions re-expressed on the ``twist``
command term and scaled by ``w``; shared terms ignore ``w``. The group weight
``w`` is the command term's smoothly ramped ``group_weight``, so switching the
active reward set is continuous.
"""

import torch

from mjlab.managers import SceneEntityCfg

from src.tasks.loco_pedipulation.mdp import rewards as loco
from src.tasks.velocity import mdp as velocity_mdp

from .anchor_frame import to_anchor

ROBOT = SceneEntityCfg('robot')
STAND_GOAL = SceneEntityCfg('robot', site_names=('stand_goal',), preserve_order=True)
FRONT = SceneEntityCfg('robot', site_names=('FL', 'FR'), preserve_order=True)
REAR = SceneEntityCfg('robot', site_names=('RL', 'RR'), preserve_order=True)
RIGHT_FRONT = SceneEntityCfg('robot', site_names=('FR',), preserve_order=True)


def _term(env):
  return env.command_manager.get_term('twist')


def _state(env):
  return env.action_manager.get_term('joint_pos')


def _gate(env):
  return _state(env).height_score > .70


def group_weight(env):
  return _term(env).group_weight


def loco_group(func):
  def weighted(env, **kwargs):
    return func(env, **kwargs) * (1. - group_weight(env))
  return weighted


# Shared terms, active in both groups without weighting.
fr_position_tracking = loco.fr_position_tracking
fr_ground_contact = loco.fr_ground_contact
manipulation_fraction = loco.manipulation_fraction
landing_failed = loco.landing_failed
is_terminated = velocity_mdp.is_terminated
joint_acc_l2 = velocity_mdp.joint_acc_l2
action_rate_l2 = velocity_mdp.action_rate_l2
joint_pos_limits = velocity_mdp.joint_pos_limits


def torques(env):
  return env.scene['robot'].data.actuator_force.abs().sum(-1)


def collision(env):
  return (env.scene['nonfoot_contact'].data.force.norm(dim=-1) > .1).float().sum(-1)


def alive(env):
  return torch.ones(env.num_envs, device=env.device)


def biped_fraction(env):
  return group_weight(env)


# LOCO group: loco_pedipulation terms weighted by 1 - w.
track_linear_velocity = loco_group(loco.track_linear_velocity)
track_angular_velocity = loco_group(loco.track_angular_velocity)
stand_still = loco_group(loco.stand_still)
feet_gait = loco_group(loco.feet_gait)
foot_clearance = loco_group(loco.foot_clearance)
foot_slip = loco_group(loco.foot_slip)
soft_landing = loco_group(loco.soft_landing)
support_contact = loco_group(loco.support_contact)
base_height = loco_group(loco.base_height)
body_orientation_l2 = loco_group(velocity_mdp.body_orientation_l2)
body_ang_vel = loco_group(velocity_mdp.body_angular_velocity_penalty)
angular_momentum = loco_group(velocity_mdp.angular_momentum_penalty)


class mode_posture(loco.mode_posture):
  def __call__(self, env, *args, **kwargs):
    return super().__call__(env, *args, **kwargs) * (1. - group_weight(env))


# BIPED group: Pedipulation terms on the twist term, weighted by w.
def biped_base_height(env):
  score = torch.exp(-torch.abs(env.scene['robot'].data.root_link_pos_w[:, 2] - .52) * 5)
  # The source intentionally uses one batch-wide gate, including its reward order.
  _state(env).height_score = score.mean()
  return score * group_weight(env)


def biped_tracking_lin_vel(env):
  term = _term(env)
  velocity = to_anchor(term.basis_w, env.scene['robot'].data.root_link_lin_vel_w)
  error = (term.command[:, :2] - velocity[:, :2]).square().sum(-1)
  return torch.exp(-error / .25) * _gate(env) * group_weight(env)


def biped_tracking_ang_vel(env):
  term = _term(env)
  velocity = to_anchor(term.basis_w, env.scene['robot'].data.root_link_ang_vel_w)
  error = term.command[:, 2] - velocity[:, 2]
  return torch.exp(-error.square() / .25) * _gate(env) * group_weight(env)


def biped_lin_vel_z(env):
  velocity = to_anchor(_term(env).basis_w, env.scene['robot'].data.root_link_lin_vel_w)
  return torch.exp(-velocity[:, 2].abs() * 10) * group_weight(env)


def biped_ang_vel_xy(env):
  velocity = to_anchor(_term(env).basis_w, env.scene['robot'].data.root_link_ang_vel_w)
  return torch.exp(-velocity[:, :2].norm(dim=-1)) * group_weight(env)


def handstand_orientation(env):
  gravity = env.scene['robot'].data.projected_gravity_b
  target = gravity.new_tensor((-1., 0., 0.))
  return (gravity - target).square().sum(-1) * group_weight(env)


def handstand_feet_on_air(env):
  contacts = env.scene['front_contact'].data.force.norm(dim=-1) > 1.0
  return (~contacts).all(-1).float() * group_weight(env)


def handstand_feet_height_exp(env, asset_cfg=STAND_GOAL):
  heights = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
  return torch.exp(-(heights - .67).abs().sum(-1) * 10) * group_weight(env)


def biped_feet_clearance(env, asset_cfg=REAR):
  heights = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2] - .02
  phase = (env.episode_length_buf * env.step_dt) % 1.6 / 1.6
  target = (2 * torch.pi * phase).sin().abs() * .06
  swing = torch.stack((phase >= .5, phase <= .5), dim=-1)
  return (torch.exp(-(heights - target[:, None]).abs() * 10) * swing).sum(-1) * _gate(env) * group_weight(env)


def biped_contact(env):
  contacts = env.scene['rear_contact'].data.force.norm(dim=-1) > 1.0
  return (contacts.sum(-1) == 1).float() * _gate(env) * group_weight(env)


def biped_ang_xz(env):
  from mjlab.utils.lab_api.math import euler_xyz_from_quat
  roll, _, _ = euler_xyz_from_quat(env.scene['robot'].data.root_link_quat_w)
  roll = torch.where(roll > torch.pi, roll - 2 * torch.pi, roll)
  return roll.abs() * _gate(env) * group_weight(env)


def symmetric_joints(env):
  joints = _state(env).joint_pos.reshape(env.num_envs, 4, 3).clone()
  joints[:, (1, 3), 0] *= -1
  no_target = ~_term(env).enabled
  front_error = (joints[:, 0] - joints[:, 1]).abs().sum(-1)
  rear_error = (joints[:, 2] - joints[:, 3]).abs().sum(-1)
  return (front_error * no_target + rear_error) * _gate(env) * group_weight(env)


def orientation_symmetry(env):
  return env.scene['robot'].data.projected_gravity_b[:, 1].square() * group_weight(env)


def feet_height_symmetry(env, asset_cfg=FRONT):
  heights = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
  enabled = ~_term(env).enabled | ~_gate(env)
  return (heights[:, 0] - heights[:, 1]).abs() * enabled * group_weight(env)


def default_pos_front(env):
  state = _state(env)
  error = (state.joint_pos[:, :6] - state.desired_angles[:6]).abs()
  no_target = ~_term(env).enabled
  return (error[:, :3].sum(-1) + error[:, 3:6].sum(-1) * no_target) * group_weight(env)


def default_pos_rear(env):
  state = _state(env)
  return (state.joint_pos[:, 6:] - state.desired_angles[6:]).abs().sum(-1) * group_weight(env)


def default_hip_pos(env):
  hips = _state(env).joint_pos[:, ::3].abs()
  no_target = ~_term(env).enabled
  return (hips[:, 0] + hips[:, 1] * no_target + hips[:, 2:].sum(-1)) * group_weight(env)


def default_pos_reward_FL(env):
  state = _state(env)
  return (torch.exp(-(state.joint_pos[:, :3] - state.desired_angles[:3]).abs().sum(-1))
          * _gate(env) * group_weight(env))


def default_pos_reward_FR(env):
  state = _state(env)
  no_target = ~_term(env).enabled
  return (torch.exp(-(state.joint_pos[:, 3:6] - state.desired_angles[3:6]).abs().sum(-1))
          * _gate(env) * no_target * group_weight(env))
