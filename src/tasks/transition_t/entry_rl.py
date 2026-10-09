"""Warm start and resumable metadata for real-entry PPO fine tuning."""
import hashlib
import math
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch

from .entry_bank import LEVELS, SPEED_EDGES, ERROR_EDGES
from .rl import TransitionOnPolicyRunner


def _recipe_value(value):
  if callable(value):
    return f'{value.__module__}.{value.__qualname__}'
  if is_dataclass(value):
    return {field.name: _recipe_value(getattr(value, field.name)) for field in fields(value)}
  if isinstance(value, dict):
    return {key: _recipe_value(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return tuple(_recipe_value(item) for item in value)
  if isinstance(value, list):
    return [_recipe_value(item) for item in value]
  if isinstance(value, slice):
    return {'slice': [value.start, value.stop, value.step]}
  return value


class EntryOnPolicyRunner(TransitionOnPolicyRunner):
  seed_checkpoint_sha256 = ''

  def _initialization_recipe(self):
    cfg = self.term.cfg
    return {name: getattr(cfg, name) for name in ('entry_split', 'entry_probability',
      'nominal_probability', 'sampling_mode', 'first_frame_probability')}

  def _augmentation_recipe(self):
    cfg = self.env.unwrapped.cfg
    group = cfg.observations['actor']
    return dict(policy='original_transition_randomization_over_nominal_teacher_states',
      events=_recipe_value(cfg.events), actor_corruption=group.enable_corruption,
      actor_noise={name: _recipe_value(term.noise) for name, term in group.terms.items()},
      nominal_pose_range=_recipe_value(cfg.commands['motion'].pose_range),
      nominal_velocity_range=_recipe_value(cfg.commands['motion'].velocity_range),
      nominal_joint_position_range=cfg.commands['motion'].joint_position_range)

  def _entry_recipe(self):
    return dict(version=4, bank_sha256=self.term.bank.sha256,
      teacher_sha256=self.term.bank.metadata['teacher_sha256'], levels=list(LEVELS),
      speed_edges=list(SPEED_EDGES), error_edges=list(ERROR_EDGES),
      split='trajectory_id_modulo_5', sampling='uniform_occupied_bins_then_states',
      evaluation_sampling='uniform_occupied_bins_without_repeated_trajectories_per_pass',
      promotion=dict(min_attempts=64, min_distinct_trajectories=64,
        standing_rate=.8, consecutive_windows=2),
      alignment='reference_first_frame_xy_yaw', tracking_grace_for_entries=0.,
      collection_physics='nominal_no_domain_randomization',
      target_limiter=self._contract().get('target_limiter'),
      initialization=self._initialization_recipe(),
      training_augmentation=self._augmentation_recipe())

  def _verify_bank_physics(self):
    contract = self._contract()
    for key in ('compiled_physics_sha256', 'source_physics_sha256'):
      if self.term.bank.metadata.get(key) != contract[key]:
        raise ValueError(f'Entry bank {key} differs from transition nominal physics')

  def warm_start(self, path):
    path = Path(path).resolve()
    self._verify_bank_physics()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'transition_entry_training' in (saved.get('infos') or {}):
      raise ValueError('Use resume for an entry-course checkpoint')
    info = super().load(str(path), load_cfg={'actor': True, 'critic': True},
      map_location=self.device, allow_legacy_target_limiter=True)
    self.seed_checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    self.current_learning_iteration = 0
    self.term.bin_failed_count.zero_()
    self.term._current_bin_failed.zero_()
    env = self.env.unwrapped
    env.common_step_counter = 0
    env._sim_step_counter = 0
    self.env.reset()
    return info

  def save(self, path, infos=None):
    state = dict(recipe=self._entry_recipe(), course=self.term.course.state_dict(),
      updates_completed=self.current_learning_iteration+1,
      seed_checkpoint_sha256=self.seed_checkpoint_sha256)
    super().save(path, {**(infos or {}), 'transition_entry_training': state})

  def resume(self, path):
    self._verify_bank_physics()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = (saved.get('infos') or {}).get('transition_entry_training')
    if not isinstance(state, dict) or state.get('recipe') != self._entry_recipe():
      raise ValueError('Entry checkpoint recipe / bank differs from current training')
    updates = state.get('updates_completed')
    if type(updates) is not int or updates <= 0:
      raise ValueError('Invalid entry updates_completed')
    self.term.course.load_state_dict(state['course'])
    info = super().load(str(path), map_location=self.device)
    rates = [group['lr'] for group in self.alg.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 or rate != rates[0] for rate in rates):
      raise ValueError('Entry optimizer requires a single positive adaptive learning rate')
    # RSL-RL restores Adam but leaves the adaptive scheduler's scalar at its
    # constructor default. Its next KL update would overwrite the restored lr.
    self.alg.learning_rate = rates[0]
    self.seed_checkpoint_sha256 = state['seed_checkpoint_sha256']
    self.current_learning_iteration = updates
    if hasattr(self, 'env'):
      self.env.unwrapped._sim_step_counter = self.env.unwrapped.common_step_counter*self.env.unwrapped.cfg.decimation
      self.env.reset()
    return info
