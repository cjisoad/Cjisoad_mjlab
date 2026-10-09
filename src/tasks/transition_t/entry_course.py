"""Physical teacher-state initialization with an independent entry curriculum."""
from dataclasses import dataclass, fields
import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
from src.tasks.pedipulation.mdp.target_limiter import target_limiter_contract
from src.tasks.pedipulation.constants import JOINT_NAMES

from .commands import TransitionCommand, TransitionCommandCfg
from .entry_bank import EntryStateBank, EntryCurriculum, LEVELS
from .env_cfg import transition_env_cfg


@dataclass(kw_only=True)
class EntryCommandCfg(TransitionCommandCfg):
  entry_bank_file: str = ''
  entry_split: str = 'train'
  entry_probability: float | None = None
  nominal_probability: float | None = None

  def build(self, env):
    return EntryCommand(self, env)


class EntryCommand(TransitionCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    action = env.cfg.actions['joint_pos']
    limiter = target_limiter_contract(action.target_velocity_limits, action.target_acceleration_limits,
      env.physics_dt)
    if limiter is None:
      raise ValueError('Entry course requires the target limiter')
    joint_ids, _ = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
    self.bank = EntryStateBank(cfg.entry_bank_file, motion_sha256=self.reference.sha256,
      device=self.device, target_limiter_contract=limiter,
      action_zero=self.robot.data.default_joint_pos[0, joint_ids], action_scale=action.scale)
    self.course = EntryCurriculum()
    self.entry_index = torch.full_like(self.time_steps, -1)
    self.entry_source = torch.zeros_like(self.time_steps)
    self.forced_entry_indices = None
    for level in LEVELS:
      for split in ('train', 'holdout'):
        if not len(self.bank.eligible(split=split, max_speed=level['max_speed'], max_fr_error=level['max_fr_error'])):
          raise ValueError(f'Entry bank has no {split} coverage for difficulty {level}')

  def _sample_reference_frames(self, env_ids):
    super()._sample_reference_frames(env_ids)
    settings = self.course.settings
    entry_p = settings['entry_probability'] if self.cfg.entry_probability is None else self.cfg.entry_probability
    nominal_p = settings['nominal_probability'] if self.cfg.nominal_probability is None else self.cfg.nominal_probability
    if not (0 <= entry_p <= 1 and 0 <= nominal_p <= 1-entry_p+1e-9):
      raise ValueError('Invalid entry / nominal mixture probabilities')
    draw = torch.rand(len(env_ids), device=self.device)
    teacher = draw < entry_p
    nominal = (draw >= entry_p) & (draw < entry_p+nominal_p)
    self.entry_source[env_ids] = teacher.long()+2*nominal.long()
    self.entry_index[env_ids] = -1
    self.time_steps[env_ids[teacher | nominal]] = 0
    ids = env_ids[teacher]
    if len(ids):
      selected = self.bank.sample(len(ids), split=self.cfg.entry_split,
        max_speed=settings['max_speed'], max_fr_error=settings['max_fr_error'])
      if self.forced_entry_indices is not None:
        selected = self.forced_entry_indices[ids]
      self.entry_index[ids] = selected

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    ids = env_ids[self.entry_index[env_ids] >= 0]
    if len(ids):
      self.restore_entries(ids, self.entry_index[ids])

  def restore_entries(self, ids, selected):
    """Restore world velocities and common action histories, then align reference."""
    data = self.bank.data
    root = data['root_state'][selected].clone()
    root[:, :3] += self._env.scene.env_origins[ids]
    self.robot.write_joint_state_to_sim(data['joint_pos'][selected], data['joint_vel'][selected], env_ids=ids)
    self.robot.write_root_state_to_sim(root, env_ids=ids)
    self.robot.clear_state(env_ids=ids)
    self.time_steps[ids] = 0
    self.completed[ids] = False
    reference_q = self.motion.body_quat_w[0, self.motion_anchor_body_index].expand(len(ids), -1)
    rotation = quat_mul(yaw_quat(root[:, 3:7]), quat_inv(yaw_quat(reference_q)))
    reference_p = self.motion.body_pos_w[0, self.motion_anchor_body_index].expand(len(ids), -1)
    translation = root[:, :3]-self._env.scene.env_origins[ids]-quat_apply(rotation, reference_p)
    translation[:, 2] = 0.
    self.reference_rotation[ids] = rotation
    self.reference_translation[ids] = translation
    self._last_update_step[ids] = self._env.common_step_counter
    # A bank sample represents a live handoff, not a newly stabilized robot.
    # Do not grant the training reference-reset 0.2 s tracking grace.
    self._reset_step[ids] = self._env.common_step_counter-math.ceil(.2/self._env.step_dt)
    self.reset_age_steps[ids] = math.ceil(.2/self._env.step_dt)
    manager = self._env.action_manager
    action = manager.get_term('joint_pos')
    for name in ('raw_action', 'previous_action', 'previous_joint_vel'):
      getattr(action, name)[ids] = data[name][selected]
    for index, target in enumerate((manager.action, manager.prev_action, manager.prev_prev_action)):
      target[ids] = data['manager_actions'][selected, index]
    action.motor_offsets[ids] = 0.
    action.restore_target_limiter(data['limiter_position'][selected], data['limiter_velocity'][selected],
      env_ids=ids)
    actual_p, actual_q = self.robot_anchor_pos_w.clone(), self.robot_anchor_quat_w.clone()
    actual_p[ids], actual_q[ids] = root[:, :3], root[:, 3:7]
    self._refresh_relative(actual_p, actual_q)


def entry_course_env_cfg(bank_file, *, num_envs=4096, play=False, split='train'):
  cfg = transition_env_cfg(play=play)
  cfg.scene.num_envs = num_envs
  old = cfg.commands['motion']
  cfg.commands['motion'] = EntryCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)},
    entry_bank_file=str(bank_file), entry_split=split)
  # Source selection explicitly determines the first-frame probability.
  cfg.commands['motion'].first_frame_probability = 0.
  if play:
    cfg.commands['motion'].sampling_mode = 'adaptive'
  return cfg
