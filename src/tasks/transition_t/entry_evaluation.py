"""Independent held-out entry trials for standing handoff eligibility."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from .entry_bank import ERROR_EDGES, LEVELS, SPEED_EDGES
from .standing import (POSITION_RADIUS, STANDING_HOLD_SECONDS, advance_standing_hold, standing_hold_policy,
  endpoint_metrics, standing_conditions)


def json_safe(value):
  if isinstance(value, dict):
    return {key: json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(item) for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


@contextmanager
def isolated_rng(device):
  """Environment seeding changes Python, NumPy, CPU and all CUDA generators."""
  del device
  py_state, np_state = random.getstate(), np.random.get_state()
  hash_seed = os.environ.get('PYTHONHASHSEED')
  devices = list(range(torch.cuda.device_count()))
  try:
    with torch.random.fork_rng(devices=devices):
      yield
  finally:
    random.setstate(py_state)
    np.random.set_state(np_state)
    if hash_seed is None:
      os.environ.pop('PYTHONHASHSEED', None)
    else:
      os.environ['PYTHONHASHSEED'] = hash_seed


class EntryTrials:
  """One attempt per world; all snapshots are taken before automatic reset."""
  def __init__(self, count, dt, *, device, timeout_s=10., position_radius=POSITION_RADIUS):
    if count < 1 or not math.isfinite(dt) or dt <= 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
      raise ValueError('Require positive trial count, control period and timeout')
    self.dt, self.timeout_s = dt, timeout_s
    if not math.isfinite(position_radius) or position_radius <= 0:
      raise ValueError('Position radius must be finite and positive')
    self.position_radius = position_radius
    self.elapsed = torch.zeros(count, dtype=torch.float64, device=device)
    self.standing_duration = torch.zeros_like(self.elapsed)
    self.standing_bad_samples = torch.zeros(count, dtype=torch.long, device=device)
    self.completed = torch.zeros(count, dtype=torch.bool, device=device)
    self.standing_held = torch.zeros_like(self.completed)
    self.finished = torch.zeros_like(self.completed)
    self.reasons = [None] * count
    self.termination_reasons = [[] for _ in range(count)]
    self._metrics = {}
    self.post_samples = torch.zeros_like(self.elapsed)
    self.tracking_samples = torch.zeros_like(self.elapsed)
    self.failed_gates = {name: torch.zeros_like(self.elapsed) for name in ('orientation','height','position')}
    self.post_sums, self.post_maxima, self.tracking_sums = {}, {}, {}

  def record(self, completed, metrics, terminated, truncated, *, termination_reasons=None):
    active = ~self.finished
    self.elapsed += active.to(self.elapsed.dtype) * self.dt
    finite = torch.stack([torch.isfinite(value) for value in metrics.values()], -1).all(-1)
    failed = active & (terminated | ~finite)
    alive = active & ~failed
    self.completed |= completed & alive
    conditions = standing_conditions(metrics, self.position_radius)
    standing = completed & alive & torch.stack(tuple(conditions.values()), -1).all(-1)
    post = completed & alive
    tracking = active & ~completed & finite
    self.post_samples += post
    self.tracking_samples += tracking
    for name, good in conditions.items():
      self.failed_gates[name] += post & ~good
    for name, value in metrics.items():
      if name not in self.post_sums:
        self.post_sums[name] = torch.zeros_like(self.elapsed)
        self.post_maxima[name] = torch.zeros_like(self.elapsed)
        self.tracking_sums[name] = torch.zeros_like(self.elapsed)
      self.post_sums[name] += torch.where(post, value, 0.)
      self.post_maxima[name] = torch.maximum(self.post_maxima[name], torch.where(post, value, 0.))
      self.tracking_sums[name] += torch.where(tracking, value, 0.)
    duration, bad_samples = advance_standing_hold(self.standing_duration, self.standing_bad_samples,
      standing, eligible=completed & alive, active=active, dt=self.dt)
    self.standing_duration.copy_(duration)
    self.standing_bad_samples.copy_(bad_samples)
    self.standing_held |= standing & (self.standing_duration >= STANDING_HOLD_SECONDS-1e-9)
    timed_out = active & (truncated | (self.elapsed >= self.timeout_s-1e-9))
    ending = failed | (active & self.standing_held) | timed_out
    # Own the tensors: the environment can overwrite both state and done buffers.
    for name, value in metrics.items():
      if name not in self._metrics:
        self._metrics[name] = torch.zeros_like(value)
      self._metrics[name][active] = value[active]
    causes = termination_reasons or {}
    for i in ending.nonzero().flatten().tolist():
      names = [name for name, flag in causes.items() if bool(flag[i])]
      self.termination_reasons[i] = names
      if bool(failed[i]):
        if not bool(finite[i]):
          names = [*names, 'nonfinite_metrics']
        self.reasons[i] = '+'.join(names) if names else 'physical_failure'
      elif bool(self.standing_held[i]):
        self.reasons[i] = 'standing_eligible'
      else:
        self.reasons[i] = 'standing_hold_not_completed' if bool(self.completed[i]) else 'reference_not_completed'
    self.finished |= ending

  def records(self):
    values = {name: value.detach().cpu().tolist() for name, value in self._metrics.items()}
    elapsed = self.elapsed.cpu().tolist()
    completed = self.completed.cpu().tolist()
    standing = self.standing_held.cpu().tolist()
    standing_duration = self.standing_duration.cpu().tolist()
    bad_samples = self.standing_bad_samples.cpu().tolist()
    def diagnostic(i):
      n = int(self.post_samples[i]); nt = int(self.tracking_samples[i])
      out = dict(post_reference_samples=n, tracking_samples=nt,
        failed_gate_samples={name: int(value[i]) for name, value in self.failed_gates.items()})
      if n:
        out.update(xy_error_mean=float(self.post_sums['xy_error'][i]/n),
          xy_error_max=float(self.post_maxima['xy_error'][i]))
        for name, label in (('horizontal_speed','horizontal_speed_mean'),
            ('rear_supported','rear_both_support_fraction'), ('rear_any_contact','rear_any_contact_fraction')):
          if name in self.post_sums:
            out[label] = float(self.post_sums[name][i]/n)
      for leg in ('FL','FR','RL','RR'):
        name = f'{leg}_aligned_error'
        if name in self.post_sums:
          if n: out[f'{name}_mean'] = float(self.post_sums[name][i]/n)
          if nt: out[f'tracking_{name}_mean'] = float(self.tracking_sums[name][i]/nt)
      for name in ('knee_contact_fraction', 'knee_force_peak'):
        if name in self.post_sums:
          if n: out[f'{name}_mean'] = float(self.post_sums[name][i]/n)
          if nt: out[f'tracking_{name}_mean'] = float(self.tracking_sums[name][i]/nt)
      return out
    return json_safe([dict(completed=completed[i], standing_held=standing[i],
      time=elapsed[i], reason=self.reasons[i], termination_reasons=list(self.termination_reasons[i]),
      standing_hold_s=standing_duration[i], standing_bad_samples=bad_samples[i],
      endpoint_metrics={name: value[i] for name, value in values.items()}, diagnostics=diagnostic(i))
      for i in range(len(self.reasons))])


def _summary(records):
  count = len(records)
  successes = sum(record['standing_held'] for record in records)
  completed = sum(record['completed'] for record in records)
  return dict(attempts=count, successes=successes,
    full_motion_completions=completed, success_rate=successes/count if count else 0.,
    full_motion_completion_rate=completed/count if count else 0.,
    failure_reasons=dict(Counter(record['reason'] for record in records if not record['standing_held'])))


def _bin_reports(records, key, edges):
  groups = [[] for _ in range(len(edges)+1)]
  bounds = torch.tensor(edges, dtype=torch.float32)
  for record in records:
    index = int(torch.bucketize(torch.tensor(record[key], dtype=torch.float32), bounds))
    groups[index].append(record)
  return [dict(lower_exclusive=edges[i-1] if i else None,
    upper_inclusive=edges[i] if i < len(edges) else None, **_summary(group))
    for i, group in enumerate(groups)]


def _validate_holdout(bank, selected, level, levels=LEVELS):
  eligible = bank.eligible(split='holdout', max_speed=levels[level]['max_speed'],
                           max_fr_error=levels[level]['max_fr_error'])
  if selected.ndim != 1 or selected.dtype != torch.long or not bool(torch.isin(selected, eligible).all()):
    raise ValueError('Forced entry indices must belong to the eligible holdout trajectories')


def _copy_actor_for_eval(actor, device):
  """Deep-copy an RSL-RL actor, excluding its transient Normal distribution."""
  memo = {}
  for module in actor.modules():
    distribution = getattr(module, 'distribution', None)
    cached = getattr(distribution, '_distribution', None)
    if cached is not None:
      memo[id(cached)] = None
  # A stochastic forward caches a Normal with non-leaf tensors. Ignore that
  # transient cache using deepcopy's memo without touching the caller's actor.
  with torch.inference_mode(False), torch.no_grad():
    copied = deepcopy(actor, memo)
  copied = copied.to(device)
  copied.eval()
  return copied


def evaluate_entries(actor, bank_file, *, level: int, device: str, attempts: int = 64, seed: int = 42,
    position_radius: float = POSITION_RADIUS, motion_file=None, curriculum='legacy', initialization='entries'):
  """Evaluate a copy of a trained RSL-RL actor, ending at the standing handoff.

  Success means full reference completion and one continuous second upright
  near the fixed aligned reference endpoint. Stepping is allowed. No standing
  teacher is executed here. Alternate radii require independent full rollouts.
  """
  if curriculum not in ('legacy', 'three-stage') or initialization not in ('entries', 'nominal'):
    raise ValueError('Unknown evaluation curriculum or initialization')
  if curriculum == 'legacy' and initialization != 'entries':
    raise ValueError('Nominal evaluation requires three-stage curriculum')
  if curriculum == 'three-stage':
    from .three_stage import STAGES, three_stage_env_cfg
    levels = STAGES
  else:
    levels = LEVELS
  if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < len(levels):
    raise ValueError('Entry evaluation level must select an existing curriculum level')
  if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
    raise ValueError('Entry evaluation attempts must be a positive integer')
  if not math.isfinite(position_radius) or position_radius <= 0:
    raise ValueError('Position radius must be finite and positive')
  if not Path(bank_file).is_file():
    raise FileNotFoundError(bank_file)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg
  from tensordict import TensorDict
  from .entry_course import entry_course_env_cfg

  with isolated_rng(device):
    policy = _copy_actor_for_eval(actor, device)
    if curriculum == 'three-stage':
      cfg = three_stage_env_cfg(bank_file, stage=level, num_envs=attempts, split='holdout',
        motion_file=motion_file, nominal_evaluation=initialization == 'nominal', evaluation=True)
    else:
      cfg = entry_course_env_cfg(bank_file, num_envs=attempts, play=True, split='holdout',
        motion_file=motion_file)
    cfg.seed = seed
    cfg.episode_length_s = 10.
    cfg.commands['motion'].entry_probability = float(initialization == 'entries')
    cfg.commands['motion'].nominal_probability = float(initialization == 'nominal')
    cfg.commands['motion'].debug_vis = False
    trials = None

    def capture_before_reset(env):
      if trials is not None:
        term = env.command_manager.get_term('motion')
        manager = env.termination_manager
        causes = {name: manager.get_term(name) for name in manager.active_terms if name != 'entry_evaluation_capture'}
        metrics = endpoint_metrics(env)
        if curriculum == 'three-stage':
          force = env.scene['knees_ground_contact'].data.force_history.norm(dim=-1)
          metrics.update(knee_contact_fraction=(force.amax(dim=1) > 5.).float().mean(-1),
            knee_force_peak=force.flatten(1).amax(-1))
        trials.record(term.time_steps >= term.motion.time_step_total-1, metrics,
          manager.terminated, manager.time_outs, termination_reasons=causes)
      return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Append last so physical failure flags are current before recording holds.
    cfg.terminations['entry_evaluation_capture'] = TerminationTermCfg(func=capture_before_reset)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      term = env.command_manager.get_term('motion')
      term.course.level = level
      settings = levels[level]
      selected = None
      if initialization == 'entries':
        selected = term.bank.sample_trials(attempts, split='holdout',
          max_speed=settings['max_speed'], max_fr_error=settings['max_fr_error'])
        _validate_holdout(term.bank, selected, level, levels)
        term.forced_entry_indices = selected.clone()
      obs, _ = env.reset(seed=seed)
      if not bool((term.time_steps == 0).all()):
        raise ValueError('Evaluation must start from the first reference frame')
      if selected is not None:
        _validate_holdout(term.bank, term.entry_index, level, levels)
        if not torch.equal(term.entry_index, selected):
          raise ValueError('Evaluation reset must restore each selected entry')
        bank_rows = {name: term.bank.data[name][selected].detach().cpu().tolist()
                     for name in ('trajectory_id', 'actual_speed', 'fr_error')}
        indices = selected.cpu().tolist()
      else:
        if not bool((term.entry_index == -1).all()):
          raise ValueError('Nominal evaluation must not restore teacher entries')
        bank_rows = dict(trajectory_id=[-1]*attempts,
          actual_speed=term.robot_anchor_lin_vel_w[:, :2].norm(dim=-1).cpu().tolist(),
          fr_error=[0.]*attempts)
        indices = [-1]*attempts
      trials = EntryTrials(attempts, env.step_dt, device=device, position_radius=position_radius)
      with torch.inference_mode():
        for _ in range(math.ceil(10./env.step_dt)):
          actions = policy(TensorDict(obs, batch_size=[attempts]), stochastic_output=False).clamp(-10., 10.)
          actions = torch.where(trials.finished[:, None], 0., actions)
          obs, _, _, _, _ = env.step(actions)
          if bool(trials.finished.all()):
            break
      if not bool(trials.finished.all()):
        raise RuntimeError('Entry evaluation did not capture every attempt within the timeout')
      records = trials.records()
      for i, record in enumerate(records):
        record.update(entry_index=indices[i], **{name: values[i] for name, values in bank_rows.items()})
      report_hold_policy = standing_hold_policy(position_radius)
      return json_safe(dict(evaluation='standing_eligibility', teacher_chain_success_evaluated=False,
        split='holdout' if initialization == 'entries' else 'reference',
        curriculum=curriculum, initialization=initialization, full_reference_from_start=True,
        level=level, seed=seed, timeout_s=10., hold_s=1.,
        standing_hold_policy=report_hold_policy,
        distinct_trajectories=len(set(bank_rows['trajectory_id'])) if selected is not None else 0,
        distinct_states=len(set(indices)) if selected is not None else 0,
        promotion_evidence_sufficient=(attempts >= 64 if selected is None else len(set(bank_rows['trajectory_id'])) >= 64),
        entry_bank_sha256=term.bank.sha256, motion_sha256=term.reference.sha256,
        **_summary(records), by_actual_speed=_bin_reports(records, 'actual_speed', SPEED_EDGES),
        by_fr_error=_bin_reports(records, 'fr_error', ERROR_EDGES), records=records))
    finally:
      env.close()
