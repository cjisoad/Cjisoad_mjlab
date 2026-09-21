"""Heading and velocity commands with the source sampling distribution."""

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi


@dataclass(kw_only=True)
class StandCommandCfg(CommandTermCfg):
  lin_vel_x: tuple[float, float] = (-.2, .6)
  lin_vel_y: tuple[float, float] = (0., 0.)
  heading: tuple[float, float] = (-3.14, 3.14)
  zero_probability: float = .20
  xy_zero_probability: float = .10
  heading_stiffness: float = .5

  def build(self, env):
    return StandCommand(self, env)


class StandCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self._last_step = 0

  @property
  def command(self):
    self._synchronize()
    return self._command

  def _synchronize(self):
    step = self._env.common_step_counter
    if step == self._last_step:
      return
    period = int(self.cfg.resampling_time_range[0] / self._env.step_dt)
    env_ids = (self._env.episode_length_buf % period == 0).nonzero().flatten()
    self._resample(env_ids)
    self._update_command()
    self._last_step = step

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
    self._command[zero_ids] = 0.
    self.heading_target[zero_ids] = 0.
    xy_ids = env_ids[torch.rand(n, device=self.device) > 1. - self.cfg.xy_zero_probability]
    self._command[xy_ids, :2] = 0.
    self._command[env_ids, :2] *= (self._command[env_ids, :2].norm(dim=-1) > .1)[:, None]

  def _update_command(self):
    quat = self._env.scene['robot'].data.root_link_quat_w
    forward = quat_apply(quat, quat.new_tensor((1., 0., 0.)).expand(self.num_envs, -1))
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    self._command[:, 2] = (self.cfg.heading_stiffness * wrap_to_pi(self.heading_target - heading)).clamp(-1., 1.)
