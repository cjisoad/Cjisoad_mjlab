"""Tracking grace uses physical reset age; physical failures remain immediate."""
import math
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude


def _command(env, name):
  return env.command_manager.get_term(name)


def _active(env, command, grace_s):
  return command.physical_reset_age_steps >= math.ceil(grace_s / env.step_dt - 1e-9)


def tracking_height_failure(env, command_name='motion', threshold=.15, grace_s=.2):
  cmd = _command(env, command_name)
  return _active(env, cmd, grace_s) & ((cmd.anchor_pos_w[:, 2]-cmd.robot_anchor_pos_w[:, 2]).abs() > threshold)


def tracking_orientation_failure(env, command_name='motion', threshold=.8, grace_s=.2):
  cmd = _command(env, command_name)
  return _active(env, cmd, grace_s) & (quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) > threshold)


def tracking_foot_height_failure(env, command_name='motion', threshold=.15, grace_s=.2):
  cmd = _command(env, command_name)
  ids = [cmd.cfg.body_names.index(f'{leg}_foot') for leg in ('FL','FR','RL','RR')]
  error = (cmd.body_pos_w[:, ids, 2]-cmd.robot_body_pos_w[:, ids, 2]).abs().amax(-1)
  return _active(env, cmd, grace_s) & (error > threshold)


def physical_failure(env, command_name='motion'):
  cmd = _command(env, command_name)
  values = (cmd.robot_joint_pos, cmd.robot_joint_vel, cmd.robot_body_pos_w,
            cmd.robot_body_quat_w, cmd.robot_body_lin_vel_w, cmd.robot_body_ang_vel_w)
  failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for value in values:
    failed |= ~torch.isfinite(value).flatten(1).all(-1)
  failed |= (cmd.robot_anchor_quat_w.norm(dim=-1) < 1e-6)
  try:
    sensor = env.scene['base_ground_contact']
  except KeyError:
    sensor = None
  if sensor is not None:
    force = sensor.data.force
    if force is not None:
      failed |= force.norm(dim=-1).flatten(1).amax(-1) > 10.
  return failed
