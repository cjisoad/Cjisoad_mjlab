"""State-preserving loco → BeyondMimic rise → standing teacher playback."""
from dataclasses import asdict, dataclass, fields
import math

import torch
from mjlab.envs import mdp
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
from src.tasks.teacher_common.commands import LocoPedipulationTeacherCommand, LocoPedipulationTeacherCommandCfg
from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
from src.tasks.loco_pedipulation.mdp.commands import HOLD
from src.tasks.loco_pedipulation.mdp.observations import gait_phase
from src.tasks.pedipulation.mdp.actions import PedipulationPositionAction, PedipulationPositionActionCfg
from src.tasks.pedipulation_bridge_t.experts import (
  FrozenTeacherActor, validate_teacher_contract, teacher_action_to_common,
  common_previous_to_teacher,
)
from .commands import TransitionCommand, TransitionCommandCfg
from .env_cfg import transition_env_cfg
from . import terminations
from .standing import STANDING_HOLD_SECONDS, advance_standing_hold

TRIPOD, TRANSITION, BIPED = 0, 2, 3
STATE_NAMES = {TRIPOD: 'TRIPOD', TRANSITION: 'TRANSITION', BIPED: 'BIPED'}


def standing_conditions(metrics):
  """Contact/posture handoff; global foot accuracy is reported separately."""
  return dict(orientation=metrics['orientation_error'] < .25,
    height=metrics['height_error'] < .05, linear_speed=metrics['linear_speed'] < .3,
    angular_speed=metrics['angular_speed'] < .8, rear_support=metrics['rear_supported'],
    front_clear=metrics['front_clear'], joint_speed=metrics['joint_speed'] < 20.)


def strict_endpoint_candidate(completed, metrics):
  # Same criteria as evaluate_transition_t.py; joint speed is a handoff-only gate.
  return (completed & (metrics['orientation_error'] < .25) & (metrics['height_error'] < .05)
    & (metrics['feet_rms'] < .06) & (metrics['linear_speed'] < .3)
    & (metrics['angular_speed'] < .8) & metrics['rear_supported'] & metrics['front_clear'])


def endpoint_metrics(env):
  motion = env.command_manager.get_term('motion')
  robot = env.scene['robot']
  names = tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR'))
  feet_ids = robot.find_bodies(names, preserve_order=True)[0]
  ref_ids = [motion.cfg.body_names.index(name) for name in names]
  feet = robot.data.body_link_pos_w[:, feet_ids]
  contact = env.scene['feet_ground_contact'].data
  forces = contact.force.reshape(env.num_envs, 4, -1, 3).sum(2).norm(dim=-1)
  found = contact.found.reshape(env.num_envs, 4, -1).any(-1)
  return dict(orientation_error=quat_error_magnitude(motion.anchor_quat_w, motion.robot_anchor_quat_w),
    height_error=(motion.anchor_pos_w[:, 2]-motion.robot_anchor_pos_w[:, 2]).abs(),
    feet_rms=(feet-motion.body_pos_w[:, ref_ids]).square().sum(-1).mean(-1).sqrt(),
    linear_speed=motion.robot_anchor_lin_vel_w.norm(dim=-1),
    angular_speed=motion.robot_anchor_ang_vel_w.norm(dim=-1),
    rear_supported=(found[:, 2:] & (forces[:, 2:] > 5.)).all(-1),
    front_clear=(~found[:, :2]).all(-1) & (feet[:, :2, 2] > .032).all(-1),
    joint_speed=robot.data.joint_vel.norm(dim=-1))


@dataclass(kw_only=True)
class KeyboardMotionCfg(TransitionCommandCfg):
  def build(self, env):
    return KeyboardMotion(self, env)


class KeyboardMotion(TransitionCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids):
    # Physical reset is the default quadruped state, never a reference pose.
    self.active[env_ids] = False
    self.time_steps[env_ids] = 0
    self.completed[env_ids] = False
    self.reference_rotation[env_ids] = self.reference_rotation.new_tensor([1., 0., 0., 0.])
    self.reference_translation[env_ids] = 0.
    self._reset_step[env_ids] = self._env.common_step_counter
    self._last_update_step[env_ids] = self._env.common_step_counter
    self._refresh_relative()

  def _update_command(self):
    self._last_update_step[~self.active] = self._env.common_step_counter
    super()._update_command()


@dataclass(kw_only=True)
class KeyboardTransitionCommandCfg(LocoPedipulationTeacherCommandCfg):
  def build(self, env):
    return KeyboardTransitionCommand(self, env)


class KeyboardTransitionCommand(LocoPedipulationTeacherCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.owner = torch.full((self.num_envs,), TRIPOD, device=self.device, dtype=torch.long)
    self.trigger_height = .33
    self.trigger_elapsed = torch.zeros(self.num_envs, device=self.device)
    self.bridge_elapsed = torch.zeros_like(self.trigger_elapsed)
    self.candidate_elapsed = torch.zeros_like(self.trigger_elapsed)
    self.candidate_bad_samples = torch.zeros_like(self.owner)
    self.strict_elapsed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
    self.strict_held = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.keyboard_command = torch.zeros(self.num_envs, 6, device=self.device)
    self.keyboard_active = torch.zeros_like(self.strict_held)
    self.keyboard_input_enabled = torch.zeros_like(self.strict_held)
    self.keyboard_timed_out = torch.zeros_like(self.strict_held)
    self.reported_failure = torch.zeros_like(self.strict_held)
    self.keyboard_revision = torch.zeros_like(self.owner)
    self.keyboard_events = []
    self._last_advance_step = torch.full_like(self.owner, -1)
    self.session_start_step = self._env.common_step_counter

  def configure_trigger(self, margin):
    if not math.isfinite(margin) or not 0. < margin < .1:
      raise ValueError('Transition margin must be between 0 and 0.10 m')
    self.trigger_height = .35-margin

  def _resample_command(self, env_ids):
    self.owner[env_ids] = TRIPOD
    for value in (self.trigger_elapsed, self.bridge_elapsed, self.candidate_elapsed, self.candidate_bad_samples,
                  self.strict_elapsed, self.keyboard_command, self._command,
                  self.requested_velocity, self.requested_offset):
      value[env_ids] = 0.
    for value in (self.keyboard_active, self.keyboard_input_enabled, self.keyboard_timed_out,
                  self.reported_failure, self.strict_held):
      value[env_ids] = False
    self.manual[env_ids] = True
    self.enabled[env_ids] = False
    self.keyboard_revision[env_ids] += 1
    self._last_advance_step[env_ids] = self._env.common_step_counter

  @property
  def fr_error(self):
    return self.foot_pos_anchor-(self.zero+self.command[:, 3:])

  @property
  def progress(self):
    motion = self._env.command_manager.get_term('motion')
    return motion.time_steps.float()/(motion.motion.time_step_total-1)

  def submit(self, env_ids, commands, target_active, *, input_enabled=True):
    ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
    values = torch.as_tensor(commands, device=self.device, dtype=self.command.dtype).reshape(-1, 6)
    active = torch.as_tensor(target_active, device=self.device, dtype=torch.bool).reshape(-1)
    if len(values) != len(ids) or len(active) != len(ids) or not torch.isfinite(values).all():
      raise ValueError('Require a finite six-value command and target flag per world')
    self.keyboard_input_enabled[ids] = input_enabled
    manual = (self.owner[ids] != TRANSITION) & ~self.keyboard_timed_out[ids]
    ids, values, active = ids[manual], values[manual], active[manual]
    self.keyboard_command[ids] = values
    self.keyboard_active[ids] = active
    source = self.owner[ids] == TRIPOD
    selected = ids[source]
    if len(selected):
      target = values[source, 3:].clone()
      target[:, 0].clamp_(-.30, .30); target[:, 1].clamp_(-.12, .12); target[:, 2].clamp_(.05, .35)
      target = torch.where(active[source, None], target, 0.)
      self.set_command(selected, velocity=values[source, :3], target_offset=target)
    selected = ids[~source]
    if len(selected):
      self._command[selected] = values[~source]
      self.requested_velocity[selected] = values[~source, :3]
      self.requested_offset[selected] = values[~source, 3:]

  def _event(self, ids, previous, state, reason):
    metrics = endpoint_metrics(self._env)
    motion = self._env.command_manager.get_term('motion')
    for i in ids.tolist():
      self.keyboard_events.append(dict(world=i, step=self._env.common_step_counter,
        simulation_seconds=(self._env.common_step_counter-self.session_start_step)*self._env.step_dt,
        previous=previous, state=state, reason=reason, command=self.command[i].cpu().tolist(),
        reference_frame=int(motion.time_steps[i]), fr_error=float(self.fr_error[i].norm()),
        root_position_error=(motion.robot_anchor_pos_w-motion.anchor_pos_w)[i].cpu().tolist(),
        joint_reference_error=float((motion.robot_joint_pos[i]-motion.joint_pos[i]).norm()),
        **{key: value[i].item() for key, value in metrics.items()}))

  def start_transition(self, ids):
    motion = self._env.command_manager.get_term('motion')
    motion.start_from_current_state(ids)
    motion.active[ids] = True
    self.owner[ids] = TRANSITION
    self.bridge_elapsed[ids] = 0.
    self.candidate_elapsed[ids] = 0.
    self.candidate_bad_samples[ids] = 0
    self.strict_elapsed[ids] = 0.
    self.strict_held[ids] = False
    self.keyboard_command[ids, :3] = 0.
    self.keyboard_revision[ids] += 1
    self._event(ids, 'TRIPOD', 'TRANSITION', 'height_and_tracking_hold')

  def advance(self):
    step = self._env.common_step_counter
    advancing = self._last_advance_step != step
    if not advancing.any():
      return
    self._last_advance_step[:] = step
    dt = self._env.step_dt
    motion = self._env.command_manager.get_term('motion')
    source = (self.owner == TRIPOD) & advancing
    transitioning = (self.owner == TRANSITION) & ~self.keyboard_timed_out & advancing
    self.bridge_elapsed += dt*transitioning
    trigger = (source & self.keyboard_active & self.keyboard_input_enabled
      & (self.keyboard_command[:, 5] >= self.trigger_height-1e-6)
      & (self.command[:, 5] >= self.trigger_height-1e-6) & (self.fr_error.norm(dim=-1) < .10))
    self.trigger_elapsed[:] = torch.where(advancing,
      torch.where(trigger, self.trigger_elapsed+dt, 0.), self.trigger_elapsed)
    ids = (trigger & (self.trigger_elapsed >= .1-1e-6)).nonzero().flatten()
    if len(ids):
      self.start_transition(ids)
    metrics = endpoint_metrics(self._env)
    finite = torch.stack([torch.isfinite(value) for value in metrics.values()], -1).all(-1)
    eligible = transitioning & (self.progress >= 1.) & finite & ~terminations.physical_failure(self._env)
    ready = eligible & torch.stack(tuple(standing_conditions(metrics).values()), -1).all(-1)
    duration, bad_samples = advance_standing_hold(self.candidate_elapsed, self.candidate_bad_samples,
      ready, eligible=eligible, active=advancing, dt=dt)
    self.candidate_elapsed.copy_(duration)
    self.candidate_bad_samples.copy_(bad_samples)
    strict = transitioning & strict_endpoint_candidate(self.progress >= 1., metrics)
    self.strict_elapsed[:] = torch.where(advancing,
      torch.where(strict, self.strict_elapsed+dt, 0.), self.strict_elapsed)
    self.strict_held |= self.strict_elapsed >= 1.-1e-9
    ids = (ready & (self.candidate_elapsed >= STANDING_HOLD_SECONDS-1e-6)).nonzero().flatten()
    if len(ids):
      self.owner[ids] = BIPED
      motion.active[ids] = False
      self.keyboard_command[ids] = self.command[ids]
      self.keyboard_command[ids, :3] = 0.
      self.keyboard_active[ids] = True
      self.keyboard_revision[ids] += 1
      self._event(ids, 'TRANSITION', 'BIPED', 'standing_hold_completed')
    ids = ((self.owner == TRANSITION) & ~self.keyboard_timed_out & (self.bridge_elapsed >= 10.-1e-5)).nonzero().flatten()
    if len(ids):
      self.keyboard_timed_out[ids] = True
      self._event(ids, 'TRANSITION', 'TIMED_OUT', 'standing_hold_not_completed')

  def _update_command(self):
    preserved = self.command.clone()
    super()._update_command()
    controlled = self.owner != TRIPOD
    self._command[controlled] = preserved[controlled]
    motion = self._env.command_manager.get_term('motion')
    active = self.owner == TRANSITION
    if active.any():
      reference_basis = anchor_basis(motion.anchor_quat_w, self.robot.data.gravity_vec_w)
      fr = motion.cfg.body_names.index('FR_foot')
      offset = to_anchor(reference_basis, motion.body_pos_w[:, fr]-motion.anchor_pos_w)-self.zero
      self._command[active, :3] = 0.
      self._command[active, 3:] = offset[active]
    self.phase[controlled] = HOLD
    self.blend[controlled] = 1.
    self.advance()

  def status(self, world=0):
    metrics = endpoint_metrics(self._env)
    missing = []
    if self.owner[world] == TRANSITION:
      if self.progress[world] < 1.:
        missing.append('reference')
      missing.extend(name for name, value in standing_conditions(metrics).items() if not value[world])
    return dict(state='TIMED_OUT' if self.keyboard_timed_out[world] else STATE_NAMES[int(self.owner[world])],
      requested_dz=float(self.keyboard_command[world, 5]), executed_dz=float(self.command[world, 5]),
      trigger_height=self.trigger_height, trigger_hold=float(self.trigger_elapsed[world]),
      progress=float(self.progress[world]), standing_hold=float(self.candidate_elapsed[world]),
      standing_bad_samples=int(self.candidate_bad_samples[world]),
      strict_hold=float(self.strict_elapsed[world]), strict_held=bool(self.strict_held[world]),
      fr_error=float(self.fr_error[world].norm()), feet_rms=float(metrics['feet_rms'][world]), missing=missing)


def teacher_observations(env, contract, ids):
  robot = env.scene['robot'].data
  action = env.action_manager.get_term('joint_pos')
  term = env.command_manager.get_term('twist')
  zero = robot.joint_pos.new_tensor(contract['default_angles'])
  previous = common_previous_to_teacher(action.raw_action[ids], action.default_angles[ids], zero)
  command = term.command[ids]
  original = torch.cat((robot.root_link_ang_vel_b[ids]*.25, robot.projected_gravity_b[ids],
    command*command.new_tensor((2., 2., .25, 1., 1., 1.)), robot.joint_pos[ids]-zero,
    robot.joint_vel[ids]*.05, previous), -1)
  gravity = robot.gravity_vec_w
  if gravity.ndim == 2 and gravity.shape[0] == env.num_envs:
    gravity = gravity[ids]
  basis = anchor_basis(robot.root_link_quat_w[ids], gravity)
  obs = torch.cat((original.clamp(-100., 100.), to_anchor(basis, robot.root_link_lin_vel_w[ids])*2.), -1)
  if contract['actor_dim'] == 54:
    obs = torch.cat((obs, gait_phase(env)[ids], term.blend[ids, None]), -1)
  return obs


class AlignedKeyboardTeacher:
  def __init__(self, checkpoint, task, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    self.contract = (saved.get('infos') or {}).get('teacher_contract')
    dim = self.contract.get('actor_dim') if isinstance(self.contract, dict) else None
    validate_teacher_contract(self.contract, task, actor_dim=dim)
    self.actor = FrozenTeacherActor(saved.get('actor_state_dict'), input_dim=dim).to(device)

  @torch.inference_mode()
  def actions(self, env, ids):
    action = env.action_manager.get_term('joint_pos')
    obs = teacher_observations(env, self.contract, ids)
    zero = obs.new_tensor(self.contract['default_angles'])
    return teacher_action_to_common(self.actor(obs), zero, action.default_angles[ids], scale=action.cfg.scale)


@dataclass(kw_only=True)
class KeyboardTransitionActionCfg(PedipulationPositionActionCfg):
  def build(self, env):
    return KeyboardTransitionAction(self, env)


class KeyboardTransitionAction(PedipulationPositionAction):
  source_teacher = None
  biped_teacher = None

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.teacher_delay = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  @torch.inference_mode()
  def process_actions(self, actions):
    term = self._env.command_manager.get_term('twist')
    selected = actions.clamp(-10., 10.).clone()
    self.teacher_delay[:] = term.owner != TRANSITION
    for teacher, state in ((self.source_teacher, TRIPOD), (self.biped_teacher, BIPED)):
      ids = (term.owner == state).nonzero().flatten()
      if len(ids):
        if teacher is None:
          raise RuntimeError('Load both endpoint teachers before stepping')
        selected[ids] = teacher.actions(self._env, ids)
    # Apply translated teacher actions after policy clipping, exactly once.
    super().process_actions(selected)
    # The next transition observation must see the policy that actually acted.
    self._env.action_manager.action.copy_(self.raw_action)

  def apply_actions(self):
    use_previous = self.teacher_delay[:, None] & (self._substep < self.delay_steps)
    current = self._requested_action if self.target_limiter is not None else self.raw_action
    previous = self._previous_requested_action if self.target_limiter is not None else self.previous_action
    action = torch.where(use_previous, previous, current)
    self._write_target(action)
    self._substep += 1


def keyboard_failure(env, kind):
  term = env.command_manager.get_term('twist')
  if kind == 'physical_failure':
    failed = terminations.physical_failure(env)
  else:
    functions = dict(anchor_pos=terminations.tracking_height_failure,
      anchor_ori=terminations.tracking_orientation_failure, ee_body_pos=terminations.tracking_foot_height_failure)
    failed = functions[kind](env) & (term.owner == TRANSITION)
  ids = (failed & ~term.reported_failure).nonzero().flatten()
  if len(ids):
    term._event(ids, 'ACTIVE', 'RESETTING', kind)
    term.reported_failure[ids] = True
  return failed


def keyboard_transition_env_cfg(num_envs=1):
  cfg = transition_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  original_motion = cfg.commands['motion']
  cfg.commands['motion'] = KeyboardMotionCfg(**{
    field.name: getattr(original_motion, field.name) for field in fields(original_motion)})
  source = loco_pedipulation_teacher_env_cfg(play=True).commands['twist']
  cfg.commands['twist'] = KeyboardTransitionCommandCfg(**asdict(source))
  cfg.commands['twist'].resampling_time_range = (1.e9, 1.e9)
  cfg.actions['joint_pos'] = KeyboardTransitionActionCfg(**asdict(cfg.actions['joint_pos']))
  cfg.events = {'reset_scene_to_default': EventTermCfg(func=mdp.reset_scene_to_default, mode='reset')}
  cfg.rewards = {}
  cfg.metrics = {}
  cfg.terminations = {name: TerminationTermCfg(func=keyboard_failure, params={'kind': name})
    for name in ('physical_failure', 'anchor_pos', 'anchor_ori', 'ee_body_pos')}
  cfg.episode_length_s = 1.e9
  return cfg
