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

from .metrics import CourseMetrics, GROUPS
from src.tasks.pedipulation.constants import DESIRED_ANGLES

from ..constants import BIPED_HIGH, BIPED_LOW, DZ_MIN, STAGE_BIPED, TRIPOD_HIGH
from .anchor_frame import anchor_basis, to_anchor
from .foot_workspace import FOOT_NAMES, FR_JOINTS, HIGH, LOW, build_foot_workspace

QUAD, PREPARE, REACH, HOLD, LOWER, RECOVER = range(6)


def smoothstep(value):
  value = value.clamp(0., 1.)
  return value**3 * (10. + value * (-15. + 6. * value))


@dataclass(kw_only=True)
class LegManipCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (6., 9.)
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
  biped_exit_timeout: float = 3.5
  stance_transition_time: float = 1.8
  switch_hold_time: float = 1.
  switch_timeout: float = 5.
  passing_windows: int = 2
  min_group_samples: int = 100
  min_group_episodes: int = 16
  min_switch_attempts: int = 10
  biped_exit_recovery_angle: float = math.radians(50.)
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
  stance_eval_interval: int = 500
  stage0_light_push: bool = False

  def build(self, env):
    return LegManipCommand(self, env)


class LegManipCommand(CommandTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    probabilities = (cfg.standing_probability, cfg.foot_target_probability,
                     cfg.tripod_standing_probability)
    durations = (cfg.prepare_time, cfg.reach_time, cfg.recover_time,
                 cfg.landing_timeout, cfg.contact_confirmation, cfg.velocity_ramp_time,
                 cfg.group_transition_time, cfg.biped_exit_timeout, cfg.stance_transition_time,
                 cfg.switch_hold_time, cfg.switch_timeout)
    if (not all(0 <= p <= 1 for p in probabilities)
        or not all(math.isfinite(t) and t > 0 for t in durations)
        or cfg.initial_stage not in range(STAGE_BIPED + 1)):
      raise ValueError('Invalid command probabilities, transition times or curriculum stage')
    if (cfg.passing_windows < 2 or cfg.min_stage_steps <= 0 or cfg.min_curriculum_episodes <= 0
        or cfg.min_group_samples <= 0 or cfg.min_group_episodes <= 0 or cfg.min_switch_attempts <= 0):
      raise ValueError('Course needs positive windows/sample counts and at least two passing windows')
    if not isinstance(cfg.stance_eval_interval, int) or cfg.stance_eval_interval <= 0:
      raise ValueError('Stance evaluation interval must be a positive integer')
    for bounds in (cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z,
                   cfg.biped_lin_vel_x, cfg.biped_ang_vel_z):
      if not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
        raise ValueError('Velocity ranges must be finite and ordered')
    if not all(math.isfinite(v) and v >= 0 for v in cfg.tripod_velocity_limits + cfg.biped_velocity_limits):
      raise ValueError('Velocity limits must be nonnegative and finite')
    if not 0 < cfg.biped_low < cfg.tripod_high:
      raise ValueError('Require 0 < biped_low < tripod_high for the overlap band')
    if not math.isfinite(cfg.biped_exit_recovery_angle) or not 0 < cfg.biped_exit_recovery_angle < math.pi / 2:
      raise ValueError('Biped exit recovery angle must be finite and between 0 and pi/2')
    self.robot = env.scene['robot']
    self.site_ids, sites = self.robot.find_sites(FOOT_NAMES, preserve_order=True)
    self.fr_joint_ids, _ = self.robot.find_joints(FR_JOINTS, preserve_order=True)
    _, geoms = self.robot.find_geoms(tuple(f'{name}_foot_collision' for name in FOOT_NAMES))
    # ContactSensor resolves model order, not the order of its pattern tuple.
    self.contact_order = [geoms.index(f'{name}_foot_collision') for name in sites]
    self.fr_index = sites.index('FR')
    self.workspace = build_foot_workspace()
    self.nominal_biped_offset = torch.tensor(self.workspace.nominal_biped_offset, device=self.device, dtype=torch.float32)
    self.zero = torch.tensor(self.workspace.zero, device=self.device, dtype=torch.float32)
    self.offsets = torch.tensor(self.workspace.offsets, device=self.device, dtype=torch.float32)
    segment = torch.tensor(self.workspace.segment, device=self.device)
    self.low_indices = (segment == LOW).nonzero().flatten()
    self.high_indices = (segment == HIGH).nonzero().flatten()
    easy = self.low_indices[self.offsets[self.low_indices, 2] <= cfg.easy_max_dz]
    self.easy_indices = easy
    if not len(self.easy_indices) or not len(self.low_indices):
      raise ValueError('Workspace has no targets for the initial manipulation curriculum')
    # Public command ABI matches both source tasks: velocity plus FR offset.
    self._command = torch.zeros(self.num_envs, 6, device=self.device)
    self.requested_velocity = torch.zeros(self.num_envs, 3, device=self.device)
    self.requested_offset = torch.zeros(self.num_envs, 3, device=self.device)
    self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._operation = torch.zeros_like(self.enabled)
    self.manual = torch.zeros_like(self.enabled)
    self._source_manipulating = torch.zeros_like(self.enabled)
    self.mode = torch.zeros_like(self.enabled)
    self.group_weight = torch.zeros(self.num_envs, device=self.device)
    self.biped_exiting = torch.zeros_like(self.enabled)
    self.biped_exit_elapsed = torch.zeros(self.num_envs, device=self.device)
    self._pending_reset = torch.ones_like(self.enabled)
    self._target_changed = torch.zeros_like(self.enabled)
    self.stage = cfg.initial_stage
    self.stage_started = 0
    self.velocity_level = 0
    self.pass_streak = 0
    self.curriculum_window = torch.zeros(4, 3, device=self.device)
    self.switch_window = torch.zeros(2, 2, device=self.device)
    self.curriculum_failures = 0
    self.group_survival = torch.zeros(4, 2, device=self.device)
    self.course_episode_groups = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
    self.last_decision = {}
    self.last_stance_eval_iteration = -1
    self.walk_started_iteration = -1
    self.pending_switch = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
    self.switch_elapsed = torch.zeros(self.num_envs, device=self.device)
    self.endpoint_hold = torch.zeros(self.num_envs, device=self.device)
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
    self.episode_tracking = torch.zeros(self.num_envs, 6, 2, device=self.device)
    self.training_metrics = CourseMetrics(self.device)
    self._last_recorded_step = -1
    self._gui = None

  @property
  def manipulating(self):
    """A nominal raised FR reference remains a stance target, not operation."""
    return self.enabled & (self.manual | self._operation)

  @property
  def trajectory_duration(self):
    return torch.where(self.manual | self._operation,
      torch.full_like(self.elapsed, self.cfg.reach_time),
      torch.full_like(self.elapsed, self.cfg.stance_transition_time))

  def stance_ready(self, *, biped, moving=False, command=None, manipulating=None):
    """Whole-body endpoint criterion; FR accuracy alone cannot certify stance."""
    command = self.command if command is None else command
    manipulating = self.manipulating if manipulating is None else manipulating
    data = self.robot.data
    gravity = data.projected_gravity_b
    height = data.root_link_pos_w[:, 2]
    contacts = self.contacts
    velocity = to_anchor(self.basis_w, data.root_link_lin_vel_w)
    angular = to_anchor(self.basis_w, data.root_link_ang_vel_w)
    tracking = ((velocity[:, :2] - command[:, :2]).norm(dim=-1)
                + .25 * (angular[:, 2] - command[:, 2]).abs())
    quiet = (velocity.norm(dim=-1) < .12) & (angular.norm(dim=-1) < .25)
    motion_ok = (tracking < .15) & (velocity[:, 2].abs() < .10) & (angular[:, :2].norm(dim=-1) < .3)
    motion_ok = motion_ok if moving else quiet
    if biped:
      target = data.joint_pos.new_tensor(DESIRED_ANGLES).expand_as(data.joint_pos)
      pose_error = (data.joint_pos - target).abs()
      pose_error = pose_error.clone()
      pose_error[:, self.fr_joint_ids] *= (~manipulating)[:, None]
      pose = pose_error.mean(-1) < (.4 if moving else .25)
      support = contacts[:, 2:].any(-1) if moving else contacts[:, 2:].all(-1)
      return ((-gravity[:, 0] > math.cos(math.radians(15.)))
        & (height > .46) & (height < .60) & ~contacts[:, :2].any(-1)
        & support & pose & motion_ok)
    pose = (data.joint_pos - data.default_joint_pos).abs().mean(-1) < (.4 if moving else .20)
    support = contacts.sum(-1) >= (2 if moving else 4)
    return ((-gravity[:, 2] > math.cos(math.radians(15.)))
      & ((height - self.workspace.nominal_height).abs() < .05)
      & support & pose & motion_ok)

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
    self.interrupt_switches(env_ids)
    lower = self.phase[env_ids] == LOWER
    timed_out = lower & (self.elapsed[env_ids] > self.trajectory_duration[env_ids] + self.cfg.landing_timeout)
    self.training_metrics.landing_event('timeouts', timed_out)
    self.training_metrics.landing_event('interrupted', lower & ~timed_out)
    stats = self.stats[env_ids].sum(0)
    for key, numerator, denominator in (
      ('fr_position_error', 3, 2), ('fr_contact_fraction', 4, 2),
    ):
      extras[key] = (stats[numerator] / stats[denominator].clamp_min(1)).item()
    tracking = self.episode_tracking[env_ids].sum(0)
    for i, group in enumerate(GROUPS):
      extras['episode_' + group + '_samples'] = float(tracking[i, 0])
      extras['episode_' + group + '_velocity_error'] = float(tracking[i, 1] / tracking[i, 0].clamp_min(1))
    for mode, rows in (('quad', (0, 3)), ('tripod', (1, 4)), ('biped', (2, 5))):
      totals = tracking[list(rows)].sum(0)
      extras[mode + '_velocity_error'] = float(totals[1] / totals[0].clamp_min(1))
    self.episode_tracking[env_ids] = 0.
    self.course_episode_groups[env_ids] = False
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
    self._operation[env_ids] = False
    self.manual[env_ids] = False
    self._source_manipulating[env_ids] = False
    self.mode[env_ids] = False
    self.group_weight[env_ids] = 0.
    self.biped_exiting[env_ids] = False
    self.biped_exit_elapsed[env_ids] = 0.
    self._target_changed[env_ids] = False
    self._pending_reset[env_ids] = True
    super().reset(env_ids)
    return extras

  def _resample_command(self, env_ids):
    ids = env_ids[~self.manual[env_ids]]
    count = len(ids)
    if not count:
      return
    # Every stage includes both stances; switches are stationary teaching examples.
    resetting = self._pending_reset[ids]
    biped = (torch.rand(count, device=self.device) < .5) & ~resetting
    operation = ((torch.rand(count, device=self.device) < self.cfg.foot_target_probability)
                 & (self.stage == STAGE_BIPED) & ~resetting)
    offsets = torch.zeros(count, 3, device=self.device)
    offsets[biped] = self.nominal_biped_offset
    for high, bank in ((False, self.low_indices), (True, self.high_indices)):
      selected = operation & (biped == high)
      n = int(selected.sum())
      if n:
        offsets[selected] = self.offsets[bank[torch.randint(len(bank), (n,), device=self.device)]]
    # Keep the full overlap workspace in an established mode; crossing into a
    # different mode needs an unambiguous target outside the hysteresis band.
    low = operation & ~biped & self.mode[ids] & (offsets[:, 2] >= self.cfg.biped_low)
    if low.any():
      bank = self.low_indices[self.offsets[self.low_indices, 2] < self.cfg.biped_low]
      offsets[low] = self.offsets[bank[torch.randint(len(bank), (int(low.sum()),), device=self.device)]]
    high = operation & biped & ~self.mode[ids] & (offsets[:, 2] <= self.cfg.tripod_high)
    if high.any():
      bank = self.high_indices[self.offsets[self.high_indices, 2] > self.cfg.tripod_high]
      offsets[high] = self.offsets[bank[torch.randint(len(bank), (int(high.sum()),), device=self.device)]]
    def uniform(bounds):
      return torch.empty(count, device=self.device).uniform_(*bounds)
    velocity = torch.stack((
      torch.where(biped, uniform(self.cfg.biped_lin_vel_x), uniform(self.cfg.lin_vel_x)),
      torch.where(biped, torch.zeros(count, device=self.device), uniform(self.cfg.lin_vel_y)),
      torch.where(biped, uniform(self.cfg.biped_ang_vel_z), uniform(self.cfg.ang_vel_z))), dim=-1)
    scale = (0. if self.stage == 0 else ((.25, .5, .75, 1.)[self.velocity_level] if self.stage == 1 else 1.))
    velocity *= scale
    standing = torch.rand(count, device=self.device) < self.cfg.standing_probability
    standing |= biped != self.mode[ids]
    velocity[standing | resetting] = 0.
    changed = (self._command[ids, 3:] != offsets).any(-1)
    self.requested_offset[ids] = offsets
    self.requested_velocity[ids] = velocity
    self._command[ids, 3:] = offsets
    self.enabled[ids] = (offsets != 0).any(-1)
    self._operation[ids] = operation
    self._target_changed[ids] |= self.enabled[ids] & changed

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
    moving = self.command[:, :3].norm(dim=-1) > .05
    context_moving = moving | (self.requested_velocity.norm(dim=-1) > .05)
    context = self.mode.long() + 2 * context_moving.long()
    self.course_episode_groups[torch.arange(self.num_envs, device=self.device), context] = True
    quad_ready = torch.where(moving, self.stance_ready(biped=False, moving=True), self.stance_ready(biped=False))
    biped_ready = torch.where(moving, self.stance_ready(biped=True, moving=True), self.stance_ready(biped=True))
    ready = torch.where(self.mode, biped_ready, quad_ready)
    stable = ((quad & ~self.mode & (self.elapsed >= 1.))
      | (hold & self.mode & (self.group_weight >= .99)))
    valid = stable & ~self._env.termination_manager.terminated
    for i in range(4):
      mask = valid & (self.mode == bool(i % 2)) & (moving == bool(i >= 2))
      self.curriculum_window[i, 0] += mask.sum()
      self.curriculum_window[i, 1] += (ready & mask).sum()
      self.curriculum_window[i, 2] += (error * mask).sum()
    self.training_metrics.record_course(self, velocity, angular, ready)
    self._record_switch_outcomes()

  def interrupt_switches(self, env_ids):
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    for direction in (0, 1):
      self.training_metrics.switch_event(direction, 'interrupted', self.pending_switch[ids] == direction)
    self.pending_switch[ids] = -1
    self.endpoint_hold[ids] = 0.
    self.switch_elapsed[ids] = 0.

  def _record_switch_outcomes(self):
    pending = self.pending_switch >= 0
    self.switch_elapsed += self._env.step_dt * pending
    manager = self._env.termination_manager
    terminal = manager.terminated | getattr(manager, "time_outs", torch.zeros_like(manager.terminated))
    failed = pending & terminal
    timeout = pending & (self.switch_elapsed >= self.cfg.switch_timeout) & ~failed
    endpoint = torch.where(self.pending_switch == 0,
      self.stance_ready(biped=True) & (self.phase == HOLD),
      self.stance_ready(biped=False) & (self.phase == QUAD))
    endpoint &= pending & ~failed & ~timeout
    self.endpoint_hold.copy_(torch.where(endpoint, self.endpoint_hold + self._env.step_dt, 0.))
    succeeded = pending & (self.endpoint_hold + 1e-6 >= self.cfg.switch_hold_time)
    for direction in (0, 1):
      direction_mask = self.pending_switch == direction
      self.switch_window[direction, 1] += (succeeded & direction_mask).sum()
      for name, mask in (('succeeded', succeeded), ('timeouts', timeout), ('interrupted', failed)):
        self.training_metrics.switch_event(direction, name, mask & direction_mask)
    self.pending_switch[succeeded | failed | timeout] = -1
    self.endpoint_hold[~(self.pending_switch >= 0)] = 0.

  def _update_command(self):
    previous_command = self.command[:, :3].clone()
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
    duration = self.trajectory_duration
    progress = smoothstep(self.elapsed / duration)
    # The landing height follows the actual base height over the flat plane.
    self.goal[lower, 2] = landing[lower, 2]
    moving = reach | lower
    self.reference[moving] = torch.lerp(self.start, self.goal, progress[:, None])[moving]
    self.blend[reach] = torch.lerp(self._blend_start, torch.ones_like(progress), progress)[reach]
    self._begin(reach & (self.elapsed >= duration), HOLD)
    landed = lower & self.contacts[:, self.fr_index]
    self.contact_time.copy_(torch.where(landed, self.contact_time + dt, 0.))
    confirmed = lower & (self.elapsed >= duration) & (self.contact_time >= self.cfg.contact_confirmation)
    self.stats[:, 9] += confirmed
    self.training_metrics.landing_event('confirmed', confirmed)
    self._begin(confirmed, RECOVER)
    recover = self.phase == RECOVER
    self.blend[recover] = (self._blend_start * (1. - smoothstep(self.elapsed / self.cfg.recover_time)))[recover]
    self._begin(recover & (self.elapsed >= self.cfg.recover_time), QUAD)

    limit = self.command.new_tensor(self.cfg.tripod_velocity_limits).expand(self.num_envs, 3).clone()
    limit[self.mode] = self.command.new_tensor(self.cfg.biped_velocity_limits)
    active = self.enabled | (self.phase != QUAD)
    desired = torch.where(active[:, None], self.requested_velocity.clamp(-limit, limit),
                          self.requested_velocity)
    step = dt / self.cfg.velocity_ramp_time
    self._command[:, :3] += (desired - self._command[:, :3]).clamp(-step, step)
    period = .6 + .6 * self.blend
    self.gait_phase.copy_((self.gait_phase + dt / period) % 1.)

    # Reward-group hysteresis on commanded target height; phases are untouched.
    dz = self.requested_offset[:, 2]
    above = self.enabled & (dz > self.cfg.tripod_high)
    below = ~self.enabled | (dz < self.cfg.biped_low)
    previous_mode = self.mode.clone()
    self.mode.copy_((self.mode | above) & ~below)
    changed_mode = previous_mode != self.mode
    if changed_mode.any():
      self.interrupt_switches(changed_mode.nonzero().flatten())
      for direction, mask in ((0, changed_mode & self.mode), (1, changed_mode & ~self.mode)):
        source_moving = previous_command.norm(dim=-1) > .05
        static_ready = self.stance_ready(biped=direction == 1, command=previous_command,
          manipulating=self._source_manipulating)
        moving_ready = self.stance_ready(biped=direction == 1, moving=True, command=previous_command,
          manipulating=self._source_manipulating)
        certified = torch.where(source_moving, moving_ready, static_ready)
        self.pending_switch[mask & certified] = direction
        self.switch_window[direction, 0] += mask.sum()
        self.training_metrics.switch_event(direction, 'attempts', mask)
        self.training_metrics.switch_event(direction, 'uncertified', mask & ~certified)
    ramp = dt / self.cfg.group_transition_time
    self.group_weight += (self.mode.float() - self.group_weight).clamp(-ramp, ramp)

    # A target cancellation changes the requested mode before the body can descend.
    # Keep a bounded exit allowance independent of the shorter reward ramp.
    self.biped_exit_elapsed += dt * self.biped_exiting
    begin_exit = previous_mode & ~self.mode
    self.biped_exiting |= begin_exit
    self.biped_exit_elapsed[begin_exit] = 0.
    upright_for_loco = (-self.robot.data.projected_gravity_b[:, 2]
                       >= math.cos(self.cfg.biped_exit_recovery_angle))
    front_support = self.contacts[:, :2].any(dim=-1)
    recovered = (self.group_weight <= 0.) & upright_for_loco & front_support
    finished = self.mode | recovered | (self.biped_exit_elapsed >= self.cfg.biped_exit_timeout)
    self.biped_exiting &= ~finished
    self.biped_exit_elapsed[~self.biped_exiting] = 0.
    self._source_manipulating.copy_(self.manipulating)

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
