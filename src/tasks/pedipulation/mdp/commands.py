"""Velocity and FR target offsets in the gravity-aligned anchor frame."""

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from .anchor_frame import AnchorFrame
from .foot_workspace import build_foot_workspace


@dataclass(kw_only=True)
class PedipulationCommandCfg(CommandTermCfg):
  lin_vel_x: tuple[float, float] = (-.2, .6)
  lin_vel_y: tuple[float, float] = (0., 0.)
  heading: tuple[float, float] = (-3.14, 3.14)
  zero_probability: float = .20
  xy_zero_probability: float = .10
  heading_stiffness: float = .5
  frame_acquire: float = .15
  frame_release: float = .05
  frame_transition_rate: float = 6.
  foot_target_probability: float = .5
  foot_target_joint_delta: tuple[float, float, float] = (.3, .4, .4)
  foot_target_max_offset: tuple[float, float, float] = (.10, .08, .10)
  foot_target_min_height: float = .10
  foot_target_min_offset: float = .02
  foot_target_std: float = .05

  def build(self, env):
    return PedipulationCommand(self, env)


class PedipulationCommand(CommandTerm):
  def __init__(self, cfg, env):
    if not 0 <= cfg.foot_target_probability <= 1 or not 0 < cfg.foot_target_std < float('inf'):
      raise ValueError('Require target probability in [0, 1] and finite positive target std')
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 6, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self._last_step = 0
    self._anchor_frame = AnchorFrame(env.scene['robot'].data.gravity_vec_w,
      cfg.frame_acquire, cfg.frame_release, cfg.frame_transition_rate)
    self.foot_workspace = build_foot_workspace(tuple(cfg.foot_target_joint_delta),
      tuple(cfg.foot_target_max_offset), cfg.foot_target_min_height, cfg.foot_target_min_offset)
    x, y, z = self.foot_workspace.zero_body
    self._foot_zero_anchor = torch.tensor((-z, y, x), device=self.device, dtype=torch.float32)
    self._foot_offsets = torch.tensor(self.foot_workspace.offsets,
      device=self.device, dtype=torch.float32)

  @property
  def anchor_frame(self):
    self._synchronize()
    return self._anchor_frame

  @property
  def has_foot_target(self):
    return (self.command[:, 3:] != 0).any(-1)

  @property
  def foot_target_pos_anchor(self):
    return self._foot_zero_anchor + self.command[:, 3:]

  @property
  def foot_target_pos_w(self):
    target = self.foot_target_pos_anchor
    offset_w = (self._anchor_frame.basis_w @ target.unsqueeze(-1)).squeeze(-1)
    return self._env.scene['robot'].data.root_link_pos_w + offset_w

  def reset(self, env_ids):
    self._anchor_frame.reset(env_ids)
    return super().reset(env_ids)

  @property
  def command(self):
    self._synchronize()
    return self._command

  def _synchronize(self):
    step = self._env.common_step_counter
    if step != self._last_step:
      period = max(1, int(self.cfg.resampling_time_range[0] / self._env.step_dt))
      episode = self._env.episode_length_buf
      env_ids = ((episode > 0) & (episode % period == 0)).nonzero().flatten()
      self._resample(env_ids)
      self._last_step = step
    # Rewards precede forward(); observations follow it. Refresh orientation on
    # both reads, but advance frame smoothing from one shared previous-step state.
    self._update_command()

  def compute(self, dt):
    # Rewards request commands before mjlab's post-reward command-manager call.
    # Updating lazily preserves source heading feedback and episode-step cadence.
    self._synchronize()

  def _update_metrics(self):
    pass

  def _resample_command(self, env_ids):
    n = len(env_ids)
    for i, bounds in enumerate((self.cfg.lin_vel_x, self.cfg.lin_vel_y)):
      self._command[env_ids, i] = torch.empty(n, device=self.device).uniform_(*bounds)
    self.heading_target[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.heading)
    zero_ids = env_ids[torch.rand(n, device=self.device) < self.cfg.zero_probability]
    self._command[zero_ids, :3] = 0.
    self.heading_target[zero_ids] = 0.
    xy_ids = env_ids[torch.rand(n, device=self.device) > 1. - self.cfg.xy_zero_probability]
    self._command[xy_ids, :2] = 0.
    self._command[env_ids, :2] *= (self._command[env_ids, :2].norm(dim=-1) > .1)[:, None]
    indices = torch.randint(len(self._foot_offsets), (n,), device=self.device)
    active = torch.rand(n, device=self.device) < self.cfg.foot_target_probability
    self._command[env_ids, 3:] = self._foot_offsets[indices] * active[:, None]

  def _update_command(self):
    quat = self._env.scene['robot'].data.root_link_quat_w
    self._anchor_frame.update(quat, self._env.step_dt, self._env.common_step_counter)
    heading = self._anchor_frame.heading
    self._command[:, 2] = (self.cfg.heading_stiffness * wrap_to_pi(self.heading_target - heading)).clamp(-1., 1.)

  def _debug_vis_impl(self, visualizer):
    targets, active = self.foot_target_pos_w, self.has_foot_target
    for index in visualizer.get_env_indices(self.num_envs):
      if active[index]:
        visualizer.add_sphere(targets[index], .025, (0.2, 0.9, 0.3, 1.),
          label=f'fr_target_{index}')
