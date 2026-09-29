"""Coordinated locomotion and FR commands with loco/biped reward-group switching.

The six-value command ABI and the FR phase machine match loco_pedipulation.
On top of it, the commanded target height ``dz`` drives a hysteresis mode
(``mode``) that selects the active reward group through a smoothly ramped
``group_weight``: above ``tripod_high`` the biped group is requested, below
``biped_low`` the loco group, and inside the 10 cm overlap band the current
mode is held. Mode switches never touch the FR phase machine, so a biped
stand only ever sees REACH and HOLD.
"""

from dataclasses import dataclass
import math

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from src.tasks.loco_pedipulation.mdp.metrics import ManipulationMetrics

from ..constants import BIPED_HIGH, BIPED_LOW, DZ_MIN, STAGE_BIPED, TRIPOD_HIGH
from .anchor_frame import anchor_basis, to_anchor
from .foot_workspace import FOOT_NAMES, FR_JOINTS, LOW, build_foot_workspace

QUAD, PREPARE, REACH, HOLD, LOWER, RECOVER = range(6)


def smoothstep(value):
  value = value.clamp(0., 1.)
  return value**3 * (10. + value * (-15. + 6. * value))


@dataclass(kw_only=True)
class LegManipCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (4., 7.)
  lin_vel_x: tuple[float, float] = (-.5, 1.)
  lin_vel_y: tuple[float, float] = (-.4, .4)
  ang_vel_z: tuple[float, float] = (-.8, .8)
  biped_lin_vel_x: tuple[float, float] = (-.2, .6)
  biped_ang_vel_z: tuple[float, float] = (-.6, .6)
  standing_probability: float = .2
  foot_target_probability: float = .4
  tripod_standing_probability: float = .5
  tripod_velocity_limits: tuple[float, float, float] = (.35, .15, .4)
  biped_velocity_limits: tuple[float, float, float] = (.6, 0., .6)
  tripod_high: float = TRIPOD_HIGH
  biped_low: float = BIPED_LOW
  group_transition_time: float = .8
  easy_max_dz: float = .12
  prepare_time: float = .3
  reach_time: float = .7
  recover_time: float = .3
  landing_timeout: float = 2.
  contact_confirmation: float = .06
  velocity_ramp_time: float = .5
  curriculum_enabled: bool = True
  initial_stage: int = 0
  min_stage_steps: int = 12000
  min_curriculum_episodes: int = 64

  def build(self, env):
    return LegManipCommand(self, env)


class LegManipCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    probabilities = (cfg.standing_probability, cfg.foot_target_probability,
                     cfg.tripod_standing_probability)
    durations = (cfg.prepare_time, cfg.reach_time, cfg.recover_time,
                 cfg.landing_timeout, cfg.contact_confirmation, cfg.velocity_ramp_time,
                 cfg.group_transition_time)
    if (not all(0 <= p <= 1 for p in probabilities)
        or not all(math.isfinite(t) and t > 0 for t in durations)
        or cfg.initial_stage not in range(STAGE_BIPED + 1)):
      raise ValueError('Invalid command probabilities, transition times or curriculum stage')
    for bounds in (cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z,
                   cfg.biped_lin_vel_x, cfg.biped_ang_vel_z):
      if not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
        raise ValueError('Velocity ranges must be finite and ordered')
    if not all(math.isfinite(v) and v >= 0 for v in cfg.tripod_velocity_limits + cfg.biped_velocity_limits):
      raise ValueError('Velocity limits must be nonnegative and finite')
    if not 0 < cfg.biped_low < cfg.tripod_high:
      raise ValueError('Require 0 < biped_low < tripod_high for the overlap band')
    self.robot = env.scene['robot']
    self.site_ids, sites = self.robot.find_sites(FOOT_NAMES, preserve_order=True)
    self.fr_joint_ids, _ = self.robot.find_joints(FR_JOINTS, preserve_order=True)
    _, geoms = self.robot.find_geoms(tuple(f'{name}_foot_collision' for name in FOOT_NAMES))
    # ContactSensor resolves model order, not the order of its pattern tuple.
    self.contact_order = [geoms.index(f'{name}_foot_collision') for name in sites]
    self.fr_index = sites.index('FR')
    self.workspace = build_foot_workspace()
    self.zero = torch.tensor(self.workspace.zero, device=self.device, dtype=torch.float32)
    self.offsets = torch.tensor(self.workspace.offsets, device=self.device, dtype=torch.float32)
    segment = torch.tensor(self.workspace.segment, device=self.device)
    self.low_indices = (segment == LOW).nonzero().flatten()
    easy = self.low_indices[self.offsets[self.low_indices, 2] <= cfg.easy_max_dz]
    self.easy_indices = easy
    if not len(self.easy_indices) or not len(self.low_indices):
      raise ValueError('Workspace has no targets for the initial manipulation curriculum')
    # Public command ABI matches both source tasks: velocity plus FR offset.
    self._command = torch.zeros(self.num_envs, 6, device=self.device)
    self.requested_velocity = torch.zeros(self.num_envs, 3, device=self.device)
    self.requested_offset = torch.zeros(self.num_envs, 3, device=self.device)
    self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.manual = torch.zeros_like(self.enabled)
    self.mode = torch.zeros_like(self.enabled)
    self.group_weight = torch.zeros(self.num_envs, device=self.device)
    self._pending_reset = torch.ones_like(self.enabled)
    self._target_changed = torch.zeros_like(self.enabled)
    self.stage = cfg.initial_stage
    self.stage_started = 0
    self.curriculum_window = torch.zeros(10, device=self.device)
    self.curriculum_episodes = 0
    self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.blend = torch.zeros_like(self.elapsed)
    self._blend_start = torch.zeros_like(self.elapsed)
    self.gait_phase = torch.zeros_like(self.elapsed)
    self.contact_time = torch.zeros_like(self.elapsed)
    self.reference = self.zero.expand(self.num_envs, -1).clone()
    self.start = self.reference.clone()
    self.goal = self.reference.clone()
    # Quad count/error, hold count/position error/contact/velocity error,
    # total steps, failed episodes, landing attempts/successes.
    self.stats = torch.zeros(self.num_envs, 10, device=self.device)
    self.training_metrics = ManipulationMetrics(self.device)
    self._last_recorded_step = -1
    self._gui = None

  @property
  def command(self):
    return self._command

  @property
  def basis_w(self):
    return anchor_basis(self.robot.data.root_link_quat_w, self.robot.data.gravity_vec_w)

  @property
  def foot_pos_anchor(self):
    pos = self.robot.data.site_pos_w[:, self.site_ids[self.fr_index]]
    return to_anchor(self.basis_w, pos - self.robot.data.root_link_pos_w)

  @property
  def reference_w(self):
    return self.robot.data.root_link_pos_w + (self.basis_w @ self.reference[..., None]).squeeze(-1)

  @property
  def contacts(self):
    force = self._env.scene['feet_ground_contact'].data.force[:, self.contact_order]
    return force.norm(dim=-1) > 1.

  @property
  def support_weights(self):
    weights = torch.ones(self.num_envs, 4, device=self.device)
    weights[:, self.fr_index] = 1. - self.blend
    return weights

  def initialize_reference(self):
    ids = self._pending_reset.nonzero().flatten()
    if len(ids):
      self.reference[ids] = self.foot_pos_anchor[ids]
      self.start[ids] = self.reference[ids]
      self.goal[ids] = self.reference[ids]
      self._pending_reset[ids] = False

  def reset(self, env_ids):
    extras = {}
    lower = self.phase[env_ids] == LOWER
    timed_out = lower & (self.elapsed[env_ids] > self.cfg.reach_time + self.cfg.landing_timeout)
    self.training_metrics.landing_event('timeouts', timed_out)
    self.training_metrics.landing_event('interrupted', lower & ~timed_out)
    stats = self.stats[env_ids].sum(0)
    for key, numerator, denominator in (
      ('quad_velocity_error', 1, 0), ('fr_position_error', 3, 2),
      ('fr_contact_fraction', 4, 2), ('tripod_velocity_error', 5, 2),
    ):
      extras[key] = (stats[numerator] / stats[denominator].clamp_min(1)).item()
    self.stats[env_ids] = 0.
    self.phase[env_ids] = QUAD
    self.elapsed[env_ids] = 0.
    self.blend[env_ids] = 0.
    self.contact_time[env_ids] = 0.
    self.gait_phase[env_ids] = 0.
    self._command[env_ids] = 0.
    self.requested_velocity[env_ids] = 0.
    self.requested_offset[env_ids] = 0.
    self.enabled[env_ids] = False
    self.manual[env_ids] = False
    self.mode[env_ids] = False
    self.group_weight[env_ids] = 0.
    self._target_changed[env_ids] = False
    self._pending_reset[env_ids] = True
    super().reset(env_ids)
    return extras

  def _resample_command(self, env_ids):
    ids = env_ids[~self.manual[env_ids]]
    count = len(ids)
    if not count:
      return
    if self.stage < 3:
      bank = self.easy_indices
    elif self.stage < STAGE_BIPED:
      bank = self.low_indices
    else:
      bank = torch.arange(len(self.offsets), device=self.device)
    selected = bank[torch.randint(len(bank), (count,), device=self.device)]
    self.requested_offset[ids] = self.offsets[selected]
    active = ((torch.rand(count, device=self.device) < self.cfg.foot_target_probability)
              & (self.stage > 0))
    above_band = self.requested_offset[ids, 2] > self.cfg.tripod_high

    def uniform(bounds):
      return torch.empty(count, device=self.device).uniform_(*bounds)

    velocity = torch.stack((
      torch.where(above_band, uniform(self.cfg.biped_lin_vel_x), uniform(self.cfg.lin_vel_x)),
      torch.where(above_band, torch.zeros(count, device=self.device), uniform(self.cfg.lin_vel_y)),
      torch.where(above_band, uniform(self.cfg.biped_ang_vel_z), uniform(self.cfg.ang_vel_z)),
    ), dim=-1)
    self.requested_velocity[ids] = velocity
    standing = torch.rand(count, device=self.device) < self.cfg.standing_probability
    standing |= (active & ~above_band & ((self.stage == 1)
      | (torch.rand(count, device=self.device) < self.cfg.tripod_standing_probability)))
    self.requested_velocity[ids[standing]] = 0.
    self._command[ids, 3:] = self.requested_offset[ids] * active[:, None]
    self.enabled[ids] = active
    self._target_changed[ids] = active

  def set_command(self, env_ids, *, velocity=None, target_offset=None):
    """Set a six-value manual command; nonzero FR offset enables manipulation.

    Manual Cartesian targets are kept continuous, matching the keyboard
    interface shared by both source tasks.  Automatic training samples still
    come from the precomputed reachable FK bank.  A literal zero is retained as
    the no-target sentinel.
    """
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    if velocity is not None:
      velocity = torch.as_tensor(velocity, device=self.device, dtype=torch.float32).expand(len(ids), 3)
      if not torch.isfinite(velocity).all():
        raise ValueError('Velocity must be finite')
    if target_offset is not None:
      offsets = torch.as_tensor(target_offset, device=self.device, dtype=torch.float32).expand(len(ids), 3)
      if not torch.isfinite(offsets).all():
        raise ValueError('Foot offsets must be finite')
      active = (offsets != 0).any(-1)
    self.manual[ids] = True
    if velocity is not None:
      self.requested_velocity[ids] = velocity
    if target_offset is not None:
      changed = (offsets != self._command[ids, 3:]).any(-1)
      self.requested_offset[ids] = offsets
      self._command[ids, 3:] = offsets
      self.enabled[ids] = active
      phases = self.phase[ids]
      # During an ongoing reach, move the endpoint continuously instead of
      # restarting the smoothstep trajectory for every keyboard update.  A
      # target change in HOLD still starts one new reach, as in the automatic
      # command path.
      reaching = active & changed & (phases == REACH)
      self.goal[ids[reaching]] = self.zero + offsets[reaching]
      self._target_changed[ids] |= active & changed & (phases == HOLD)

  def release_manual(self, env_ids):
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    self.manual[ids] = False

  def _begin(self, mask, phase, goal=None):
    self.phase[mask] = phase
    self.elapsed[mask] = 0.
    self.start[mask] = self.reference[mask]
    self._blend_start[mask] = self.blend[mask]
    if goal is not None:
      self.goal[mask] = goal[mask]

  def _update_metrics(self):
    pass

  def record_outcomes(self):
    if self._last_recorded_step == self._env.common_step_counter:
      return
    self._last_recorded_step = self._env.common_step_counter
    quad, hold = self.phase == QUAD, self.phase == HOLD
    velocity = to_anchor(self.basis_w, self.robot.data.root_link_lin_vel_w)
    angular = to_anchor(self.basis_w, self.robot.data.root_link_ang_vel_w)
    error = ((velocity[:, :2] - self.command[:, :2]).norm(dim=-1)
             + .25 * (angular[:, 2] - self.command[:, 2]).abs())
    self.stats[:, 0] += quad
    self.stats[:, 1] += error * quad
    self.stats[:, 2] += hold
    self.stats[:, 3] += (self.foot_pos_anchor - self.reference).norm(dim=-1) * hold
    self.stats[:, 4] += self.contacts[:, self.fr_index] * hold
    self.stats[:, 5] += error * hold
    self.stats[:, 6] += 1.
    self.stats[:, 7] += self._env.termination_manager.terminated
    self.training_metrics.record_tracking(hold, self.command, velocity, angular)

  def _update_command(self):
    # The public target sentinel is authoritative; internal phase state is not actor input.
    self.enabled.copy_((self._command[:, 3:] != 0).any(-1))
    self.initialize_reference()
    dt = self._env.step_dt
    self.elapsed += dt
    quad = self.phase == QUAD
    self.reference[quad] = self.foot_pos_anchor[quad]
    self._begin(quad & self.enabled, PREPARE)
    self._begin((self.phase == PREPARE) & ~self.enabled, RECOVER)
    begin_lower = ((self.phase == REACH) | (self.phase == HOLD)) & ~self.enabled
    landing = self.zero.expand(self.num_envs, -1).clone()
    height = (self.robot.data.root_link_pos_w * self.basis_w[:, :, 2]).sum(-1)
    landing[:, 2] = self.workspace.foot_radius - height
    self.stats[:, 8] += begin_lower
    self.training_metrics.landing_event('attempts', begin_lower)
    self._begin(begin_lower, LOWER, landing)
    begin_reach = (((self.phase == PREPARE) & (self.elapsed >= self.cfg.prepare_time))
      | (((self.phase == LOWER) | (self.phase == RECOVER)) & self.enabled)
      | (((self.phase == REACH) | (self.phase == HOLD)) & self.enabled & self._target_changed))
    self.training_metrics.landing_event('interrupted', begin_reach & (self.phase == LOWER))
    self._begin(begin_reach, REACH, self.zero + self.requested_offset)
    self._target_changed.zero_()

    reach, lower = self.phase == REACH, self.phase == LOWER
    progress = smoothstep(self.elapsed / self.cfg.reach_time)
    # The landing height follows the actual base height over the flat plane.
    self.goal[lower, 2] = landing[lower, 2]
    moving = reach | lower
    self.reference[moving] = torch.lerp(self.start, self.goal, progress[:, None])[moving]
    self.blend[reach] = torch.lerp(self._blend_start, torch.ones_like(progress), progress)[reach]
    self._begin(reach & (self.elapsed >= self.cfg.reach_time), HOLD)
    landed = lower & self.contacts[:, self.fr_index]
    self.contact_time = torch.where(landed, self.contact_time + dt, 0.)
    confirmed = lower & (self.elapsed >= self.cfg.reach_time) & (self.contact_time >= self.cfg.contact_confirmation)
    self.stats[:, 9] += confirmed
    self.training_metrics.landing_event('confirmed', confirmed)
    self._begin(confirmed, RECOVER)
    recover = self.phase == RECOVER
    self.blend[recover] = (self._blend_start * (1. - smoothstep(self.elapsed / self.cfg.recover_time)))[recover]
    self._begin(recover & (self.elapsed >= self.cfg.recover_time), QUAD)

    limit = self.command.new_tensor(self.cfg.tripod_velocity_limits).expand(self.num_envs, 3).clone()
    if self.stage == 1:
      limit[:] = 0.
    elif self.stage == 2:
      limit[:] = torch.minimum(limit, limit.new_tensor((.2, .1, .2)))
    limit[self.mode] = self.command.new_tensor(self.cfg.biped_velocity_limits)
    active = self.enabled | (self.phase != QUAD)
    desired = torch.where(active[:, None], self.requested_velocity.clamp(-limit, limit),
                          self.requested_velocity)
    step = dt / self.cfg.velocity_ramp_time
    self._command[:, :3] += (desired - self._command[:, :3]).clamp(-step, step)
    period = .6 + .6 * self.blend
    self.gait_phase = (self.gait_phase + dt / period) % 1.

    # Reward-group hysteresis on commanded target height; phases are untouched.
    dz = self.requested_offset[:, 2]
    above = self.enabled & (dz > self.cfg.tripod_high)
    below = ~self.enabled | (dz < self.cfg.biped_low)
    self.mode.copy_((self.mode | above) & ~below)
    ramp = dt / self.cfg.group_transition_time
    self.group_weight += (self.mode.float() - self.group_weight).clamp(-ramp, ramp)

  def compute(self, dt):
    self._read_gui()
    super().compute(dt)

  def create_gui(self, name, server, get_env_idx):
    from viser import Icon
    with server.gui.add_folder('Leg manip'):
      manual = server.gui.add_checkbox('Manual', initial_value=False)
      velocity = [server.gui.add_slider(label, min=bounds[0], max=bounds[1], step=.01, initial_value=0.)
                  for label, bounds in zip(('vx', 'vy', 'wz'),
                    (self.cfg.lin_vel_x, self.cfg.lin_vel_y, self.cfg.ang_vel_z))]
      offsets = [server.gui.add_slider(label, min=low, max=high, step=.005, initial_value=value)
                 for label, low, high, value in (
                   ('FR dx', -.12, .30, 0.), ('FR dy', -.09, .09, 0.), ('FR dz', .0, BIPED_HIGH, .07))]
      stop = server.gui.add_button('Stop', icon=Icon.PLAYER_STOP)
      @stop.on_click
      def _(_event):
        for slider in velocity:
          slider.value = 0.
        for slider in offsets:
          slider.value = 0.
    self._gui = (manual, velocity, offsets, get_env_idx)
    self._gui_previous = None
    self._gui_env = None

  def _read_gui(self):
    if self._gui is None:
      return
    manual, velocity, offsets, get_env_idx = self._gui
    index = get_env_idx()
    if self._gui_env is not None and (not manual.value or index != self._gui_env):
      self.release_manual([self._gui_env])
      self._gui_previous = None
      self._gui_env = None
    if manual.value:
      values = (index, *(s.value for s in velocity), *(s.value for s in offsets))
      if values != self._gui_previous:
        target = values[4:] if self._gui_previous is None or values[4:] != self._gui_previous[4:] else None
        self.set_command([index], velocity=values[1:4], target_offset=target)
        self._gui_previous = values
      self._gui_env = index

  def _debug_vis_impl(self, visualizer):
    target = self.robot.data.root_link_pos_w + (self.basis_w @ (self.zero + self.requested_offset)[..., None]).squeeze(-1)
    reference = self.reference_w
    for index in visualizer.get_env_indices(self.num_envs):
      if self.enabled[index] or self.phase[index] != QUAD:
        color = (.9, .3, .8, 1.) if self.mode[index] else (.2, .9, .3, 1.)
        visualizer.add_sphere(target[index], .022, color, label=f'fr_goal_{index}')
        visualizer.add_sphere(reference[index], .014, (1., .6, .1, 1.), label=f'fr_reference_{index}')
      origin = self.robot.data.root_link_pos_w[index]
      direction = self.basis_w[index, :, :2] @ self.command[index, :2]
      visualizer.add_arrow(origin, origin + direction * .4, color=(.2, .5, 1., 1.), width=.012)
