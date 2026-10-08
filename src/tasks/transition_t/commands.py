"""One-shot BeyondMimic command with physical reset perturbations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp import MotionCommandCfg as FrameworkMotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat, quat_from_euler_xyz, sample_uniform
from src.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from .motion import DEFAULT_MOTION_PATH, JOINT_NAMES, TransitionMotion

TRACKED_BODY_NAMES = ('base_link', *(f'{leg}_{part}' for leg in ('FL', 'FR', 'RL', 'RR')
                                   for part in ('thigh', 'calf', 'foot')))


@dataclass(kw_only=True)
class TransitionCommandCfg(MotionCommandCfg, FrameworkMotionCommandCfg):
  motion_file: str = str(DEFAULT_MOTION_PATH)
  entity_name: str = 'robot'
  anchor_body_name: str = 'base_link'
  body_names: tuple[str, ...] = TRACKED_BODY_NAMES
  resampling_time_range: tuple[float, float] = (1.e9, 1.e9)
  first_frame_probability: float = .25
  joint_position_range: tuple[float, float] = (-.1, .1)

  def build(self, env):
    return TransitionCommand(self, env)


class TransitionCommand(MotionCommand):
  def __init__(self, cfg, env):
    # The parent loader relies on entity indices matching the archive; load by
    # names here, while retaining the parent's metric and adaptive algorithms.
    CommandTerm.__init__(self, cfg, env)
    self.reference = TransitionMotion(cfg.motion_file)
    if not math.isclose(env.step_dt, 1/self.reference.fps, rel_tol=0., abs_tol=1e-9):
      raise ValueError('control period must match the 50 Hz reference')
    for probability in (cfg.first_frame_probability,):
      if not 0. <= probability <= 1.:
        raise ValueError('sampling probabilities must be in [0, 1]')
    if cfg.sampling_mode not in ('adaptive', 'uniform', 'start'):
      raise ValueError('invalid reference sampling_mode')
    if cfg.anchor_body_name != 'base_link':
      raise ValueError('Go2 reset anchor must be base_link')
    self.robot = env.scene[cfg.entity_name]
    if tuple(self.robot.joint_names) != JOINT_NAMES:
      raise ValueError('Robot joint order must be canonical FL/FR/RL/RR')
    if len(set(cfg.body_names)) != len(cfg.body_names):
      raise ValueError('duplicate tracking body names')
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    ids, names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
    if tuple(names) != cfg.body_names:
      raise ValueError('missing tracking body names')
    self.body_indexes = torch.tensor(ids, device=self.device)
    body_ids = [self.reference.body_name_to_index[name] for name in cfg.body_names]
    tensors = {key: torch.tensor(value, dtype=torch.float32, device=self.device)
               for key, value in self.reference.arrays.items()}
    self.motion = SimpleNamespace(**{key: value[:, body_ids] if key.startswith('body_') else value
                                    for key, value in tensors.items()},
                                  time_step_total=self.reference.num_frames)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.reset_age_steps = torch.zeros_like(self.time_steps)
    self._reset_step = torch.full_like(self.time_steps, getattr(env, 'common_step_counter', 0))
    self._last_update_step = self._reset_step.clone()
    self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.reference_rotation = torch.zeros(self.num_envs, 4, device=self.device)
    self.reference_rotation[:, 0] = 1.
    self.reference_translation = torch.zeros(self.num_envs, 3, device=self.device)
    self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
    self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
    self.body_quat_relative_w[..., 0] = 1.
    self.bin_count = int(self.motion.time_step_total // (1/env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
    self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    self.kernel = torch.tensor([cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device)
    self.kernel /= self.kernel.sum()
    for name in ('error_anchor_pos', 'error_anchor_rot', 'error_anchor_lin_vel', 'error_anchor_ang_vel',
                 'error_body_pos', 'error_body_rot', 'error_body_lin_vel', 'error_body_ang_vel',
                 'error_joint_pos', 'error_joint_vel', 'sampling_entropy', 'sampling_top1_prob',
                 'sampling_top1_bin'):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._refresh_relative()

  @property
  def joint_vel(self):
    return torch.where(self.completed[:, None], 0., self.motion.joint_vel[self.time_steps])

  @property
  def body_pos_w(self):
    return (quat_apply(self.reference_rotation[:, None].expand(-1, len(self.cfg.body_names), -1),
                       self.motion.body_pos_w[self.time_steps])
            + self.reference_translation[:, None] + self._env.scene.env_origins[:, None])

  @property
  def body_quat_w(self):
    return quat_mul(self.reference_rotation[:, None].expand(-1, len(self.cfg.body_names), -1),
                    self.motion.body_quat_w[self.time_steps])

  def _world_velocity(self, value):
    rotated = quat_apply(self.reference_rotation[:, None].expand(-1, len(self.cfg.body_names), -1), value[self.time_steps])
    return torch.where(self.completed[:, None, None], 0., rotated)

  @property
  def body_lin_vel_w(self):
    return self._world_velocity(self.motion.body_lin_vel_w)

  @property
  def body_ang_vel_w(self):
    return self._world_velocity(self.motion.body_ang_vel_w)

  @property
  def anchor_pos_w(self):
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self):
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self):
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self):
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  def _resample_command(self, env_ids):
    if len(env_ids) == 0:
      return
    if self.cfg.sampling_mode == 'start':
      self.time_steps[env_ids] = 0
    else:
      if self.cfg.sampling_mode == 'uniform':
        self._uniform_sampling(env_ids)
      else:
        self._adaptive_sampling(env_ids)
      first = torch.rand(len(env_ids), device=self.device) < self.cfg.first_frame_probability
      self.time_steps[env_ids[first]] = 0
    self.completed[env_ids] = False
    self.reset_age_steps[env_ids] = 0
    self._reset_step[env_ids] = getattr(self._env, 'common_step_counter', 0)
    self._last_update_step[env_ids] = self._reset_step[env_ids]
    self.reference_rotation[env_ids] = self.reference_rotation.new_tensor([1., 0., 0., 0.])
    self.reference_translation[env_ids] = 0.
    frames = self.time_steps[env_ids]
    anchor = self.motion_anchor_body_index
    root = torch.cat((self.motion.body_pos_w[frames, anchor], self.motion.body_quat_w[frames, anchor],
                      self.motion.body_lin_vel_w[frames, anchor], self.motion.body_ang_vel_w[frames, anchor]), -1)
    ranges = torch.tensor(
      [self.cfg.pose_range.get(key, (0., 0.)) for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')],
      device=self.device,
    )
    pose_delta = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
    root[:, :3] += pose_delta[:, :3]
    root[:, 3:7] = quat_mul(
      quat_from_euler_xyz(*pose_delta[:, 3:].unbind(-1)), root[:, 3:7]
    )
    velocity_ranges = torch.tensor(
      [self.cfg.velocity_range.get(key, (0., 0.)) for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')],
      device=self.device,
    )
    velocity_delta = sample_uniform(velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=self.device)
    root[:, 7:] += velocity_delta
    joint_pos = self.motion.joint_pos[frames].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(*self.cfg.joint_position_range)
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, limits[:, :, 0], limits[:, :, 1])
    world_root = root
    world_root[:, :3] += self._env.scene.env_origins[env_ids]
    self.robot.write_joint_state_to_sim(joint_pos, self.motion.joint_vel[frames], env_ids=env_ids)
    self.robot.write_root_state_to_sim(world_root, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)
    # Derived body tensors still describe the old episode until sim.forward().
    # Use the root state just written for reset environments' relative targets.
    actual_pos = self.robot_anchor_pos_w.clone()
    actual_quat = self.robot_anchor_quat_w.clone()
    actual_pos[env_ids] = world_root[:, :3]
    actual_quat[env_ids] = world_root[:, 3:7]
    self._refresh_relative(actual_pos, actual_quat)

  def start_from_current_state(self, env_ids):
    """Start the reference without touching physical or actuator state."""
    self.time_steps[env_ids] = 0
    self.completed[env_ids] = False
    reference_quat = self.motion.body_quat_w[0, self.motion_anchor_body_index].expand(len(env_ids), -1)
    rotation = quat_mul(yaw_quat(self.robot_anchor_quat_w[env_ids]), quat_inv(yaw_quat(reference_quat)))
    self.reference_rotation[env_ids] = rotation
    reference_pos = self.motion.body_pos_w[0, self.motion_anchor_body_index].expand(len(env_ids), -1)
    translation = self.robot_anchor_pos_w[env_ids] - self._env.scene.env_origins[env_ids] - quat_apply(rotation, reference_pos)
    translation[:, 2] = 0.
    self.reference_translation[env_ids] = translation
    self.time_left[env_ids] = self.cfg.resampling_time_range[1]
    self._last_update_step[env_ids] = getattr(self._env, 'common_step_counter', 0)
    self._refresh_relative()

  @property
  def physical_reset_age_steps(self):
    if hasattr(self._env, 'common_step_counter'):
      return self._env.common_step_counter - self._reset_step
    return self.reset_age_steps

  def _refresh_relative(self, actual_pos=None, actual_quat=None):
    if actual_pos is None:
      actual_pos = self.robot_anchor_pos_w
    if actual_quat is None:
      actual_quat = self.robot_anchor_quat_w
    anchor = self.anchor_pos_w[:, None].expand(-1, len(self.cfg.body_names), -1)
    target_quat = self.anchor_quat_w[:, None].expand(-1, len(self.cfg.body_names), -1)
    robot_quat = actual_quat[:, None].expand_as(target_quat)
    rotation = yaw_quat(quat_mul(robot_quat, quat_inv(target_quat)))
    origin = actual_pos[:, None].expand_as(anchor).clone()
    origin[..., 2] = anchor[..., 2]
    self.body_quat_relative_w = quat_mul(rotation, self.body_quat_w)
    self.body_pos_relative_w = origin + quat_apply(rotation, self.body_pos_w-anchor)

  def _update_command(self):
    if hasattr(self._env, 'common_step_counter'):
      current_step = self._env.common_step_counter
      advance = current_step > self._last_update_step
      self._last_update_step[:] = current_step
      self.reset_age_steps.copy_(self.physical_reset_age_steps)
    else:
      advance = torch.ones_like(self.completed)
      self.reset_age_steps += 1
    self.completed |= advance & (self.time_steps >= self.motion.time_step_total-1)
    self.time_steps.add_(advance.long()).clamp_(max=self.motion.time_step_total-1)
    self._refresh_relative()
    if self.cfg.sampling_mode == 'adaptive':
      self.bin_failed_count.mul_(1-self.cfg.adaptive_alpha).add_(self._current_bin_failed, alpha=self.cfg.adaptive_alpha)
      self._current_bin_failed.zero_()
