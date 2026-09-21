"""Rear-leg stand rewards with gravity-aligned velocity tracking."""

import torch

from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

from .contacts import foot_in_contact


ROBOT = SceneEntityCfg('robot')
STAND_GOAL = SceneEntityCfg('robot', site_names=('stand_goal',), preserve_order=True)
FRONT = SceneEntityCfg('robot', site_names=('FL', 'FR'), preserve_order=True)
REAR = SceneEntityCfg('robot', site_names=('RL', 'RR'), preserve_order=True)


def _state(env):
  return env.action_manager.get_term('joint_pos')


def _gate(env):
  return _state(env).height_score > .70


def _forces(env, sensor):
  return env.scene[sensor].data.force


def base_height(env):
  score = torch.exp(-torch.abs(env.scene['robot'].data.root_link_pos_w[:, 2] - .52) * 5)
  # The source intentionally uses one batch-wide gate, including its reward order.
  _state(env).height_score = score.mean()
  return score


def handstand_orientation(env):
  gravity = env.scene['robot'].data.projected_gravity_b
  target = gravity.new_tensor((-1., 0., 0.))
  return (gravity - target).square().sum(-1)


def handstand_feet_on_air(env):
  return (~foot_in_contact(env, 'front_contact')).all(-1).float()


def handstand_feet_height_exp(env, asset_cfg=STAND_GOAL):
  heights = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
  return torch.exp(-(heights - .67).abs().sum(-1) * 10)


def tracking_lin_vel(env):
  term = env.command_manager.get_term('stand')
  vel = term.frame.to_frame(env.scene['robot'].data.root_link_lin_vel_w)
  cmd = term.command
  error = (cmd[:, :2] - vel[:, :2]).square().sum(-1)
  return torch.exp(-error / .25) * _gate(env)


def tracking_ang_vel(env):
  term = env.command_manager.get_term('stand')
  velocity = term.frame.to_frame(env.scene['robot'].data.root_link_ang_vel_w)
  error = term.command[:, 2] - velocity[:, 2]
  return torch.exp(-error.square() / .25) * _gate(env)


def lin_vel_z(env):
  velocity = env.command_manager.get_term('stand').frame.to_frame(env.scene['robot'].data.root_link_lin_vel_w)
  return torch.exp(-velocity[:, 2].abs() * 10)


def ang_vel_xy(env):
  velocity = env.command_manager.get_term('stand').frame.to_frame(env.scene['robot'].data.root_link_ang_vel_w)
  return torch.exp(-velocity[:, :2].norm(dim=-1))


def ang_xz(env):
  roll, _, _ = euler_xyz_from_quat(env.scene['robot'].data.root_link_quat_w)
  roll = torch.where(roll > torch.pi, roll - 2 * torch.pi, roll)
  return roll.abs() * _gate(env)


def default_pos_front(env):
  state = _state(env)
  return (state.joint_pos[:, :6] - state.desired_angles[:6]).abs().sum(-1)


def default_pos_rear(env):
  state = _state(env)
  return (state.joint_pos[:, 6:] - state.desired_angles[6:]).abs().sum(-1)


def default_pos_reward_FL(env):
  state = _state(env)
  return torch.exp(-(state.joint_pos[:, :3] - state.desired_angles[:3]).abs().sum(-1)) * _gate(env)


def default_pos_reward_FR(env):
  state = _state(env)
  return torch.exp(-(state.joint_pos[:, 3:6] - state.desired_angles[3:6]).abs().sum(-1)) * _gate(env)


def default_hip_pos(env):
  return _state(env).joint_pos[:, ::3].abs().sum(-1)


def symmetric_joints(env):
  joints = _state(env).joint_pos.reshape(env.num_envs, 4, 3).clone()
  joints[:, (1, 3), 0] *= -1
  error = (joints[:, 0] - joints[:, 1]).abs().sum(-1) + (joints[:, 2] - joints[:, 3]).abs().sum(-1)
  return error * _gate(env)


def orientation_symmetry(env):
  return env.scene['robot'].data.projected_gravity_b[:, 1].square()


def feet_height_symmetry(env, asset_cfg=FRONT):
  heights = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
  return (heights[:, 0] - heights[:, 1]).abs()


def feet_clearance(env, asset_cfg=REAR):
  heights = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2] - .02
  phase = (env.episode_length_buf * env.step_dt) % 1.6 / 1.6
  target = (2 * torch.pi * phase).sin().abs() * .06
  swing = torch.stack((phase >= .5, phase <= .5), dim=-1)
  return (torch.exp(-(heights - target[:, None]).abs() * 10) * swing).sum(-1) * _gate(env)


def contact(env):
  contacts = foot_in_contact(env, 'rear_contact')
  return (contacts.sum(-1) == 1).float() * _gate(env)


def collision(env):
  return (_forces(env, 'nonfoot_contact').norm(dim=-1) > .1).float().sum(-1)


def action_rate(env):
  state = _state(env)
  return (state.raw_action - state.previous_action).square().sum(-1)


def dof_acc(env):
  state = _state(env)
  return ((state.joint_vel - state.previous_joint_vel) / env.step_dt).square().sum(-1)


def dof_pos_limits(env):
  state = _state(env)
  limits = env.scene['robot'].data.soft_joint_pos_limits[:, state.joint_ids]
  return ((limits[..., 0] - state.joint_pos).clamp_min(0) + (state.joint_pos - limits[..., 1]).clamp_min(0)).sum(-1)


def torques(env):
  return env.scene['robot'].data.actuator_force.abs().sum(-1)


def alive(env):
  return torch.ones(env.num_envs, device=env.device)
