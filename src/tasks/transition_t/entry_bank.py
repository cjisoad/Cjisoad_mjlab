"""Validated physical teacher states, trajectory splits and entry difficulty."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


LEVELS = (
  dict(max_speed=.25, max_fr_error=.08, entry_probability=.20, nominal_probability=.40),
  dict(max_speed=.45, max_fr_error=.14, entry_probability=.35, nominal_probability=.30),
  dict(max_speed=.70, max_fr_error=.22, entry_probability=.50, nominal_probability=.25),
  dict(max_speed=1.20, max_fr_error=.45, entry_probability=.60, nominal_probability=.20),
)
SPEED_EDGES = (.1, .25, .4, .6, .8, 1.2)
ERROR_EDGES = (.03, .06, .1, .15, .25, .4)


class EntryStateBank:
  def __init__(self, path, *, motion_sha256, device='cpu', target_limiter_contract=None,
      action_zero=None, action_scale=None):
    self.path = Path(path).resolve()
    self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
    shapes = dict(root_state=(13,), joint_pos=(12,), joint_vel=(12,),
      raw_action=(12,), previous_action=(12,), manager_actions=(3, 12),
      previous_joint_vel=(12,), actual_speed=(), fr_error=(), fr_offset=(3,),
      command=(6,), trajectory_id=())
    with np.load(self.path, allow_pickle=False) as archive:
      self.metadata = json.loads(str(archive['metadata_json'].item()))
      schema = self.metadata.get('schema_version')
      if schema not in (1, 2):
        raise ValueError('Unsupported entry bank schema')
      if schema == 2:
        shapes.update(limiter_position=(12,), limiter_velocity=(12,))
      if target_limiter_contract is not None and (schema != 2
          or self.metadata.get('target_limiter') != target_limiter_contract):
        raise ValueError('Entry bank requires a matching target limiter; recollect the bank')
      if self.metadata.get('motion_sha256') != motion_sha256:
        raise ValueError('Entry bank reference motion differs')
      self.data = {}
      n = len(archive['root_state'])
      if n == 0:
        raise ValueError('Entry bank is empty')
      for name, shape in shapes.items():
        value = archive[name]
        if value.shape != (n, *shape) or not np.isfinite(value).all():
          raise ValueError(f'Entry bank {name}: require shape {(n, *shape)} and finite values')
        if name == 'trajectory_id' and (value.dtype.kind not in 'iu' or (value < 0).any()):
          raise ValueError('trajectory_id must contain nonnegative integers')
        self.data[name] = torch.tensor(value, device=device,
          dtype=torch.long if name == 'trajectory_id' else torch.float32)
    qnorm = self.data['root_state'][:, 3:7].norm(dim=-1)
    if not torch.allclose(qnorm, torch.ones_like(qnorm), atol=1e-4, rtol=0.):
      raise ValueError('Entry root quaternion must be normalized')
    actual = self.data['root_state'][:, 7:9].norm(dim=-1)
    if not torch.allclose(actual, self.data['actual_speed'], atol=1e-4, rtol=1e-4):
      raise ValueError('actual_speed does not match physical root velocity')
    if (self.data['fr_error'] < 0).any():
      raise ValueError('fr_error must be nonnegative')
    if not torch.allclose(self.data['raw_action'], self.data['manager_actions'][:, 0], atol=1e-5, rtol=0.):
      raise ValueError('Manager current action must match actual executed common action')
    if schema == 2:
      contract = self.metadata.get('target_limiter')
      if not isinstance(contract, dict):
        raise ValueError('Entry bank is missing target limiter configuration')
      raw_limits = contract.get('velocity_limits', [])
      acceleration = contract.get('acceleration_limits', [])
      for values in (raw_limits, acceleration):
        if len(values) != 12 or any(value is not None and
            (not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0)
            for value in values):
          raise ValueError('Entry bank has invalid target limiter configuration')
      limits = torch.tensor([float('inf') if value is None else value for value in raw_limits], device=device)
      zero = torch.tensor(self.metadata.get('action_zero', []), device=device)
      scale = self.metadata.get('action_scale')
      if (zero.shape != (12,) or not torch.isfinite(zero).all()
          or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Entry bank has invalid target limiter or action configuration')
      if action_zero is not None and not torch.allclose(zero,
          torch.as_tensor(action_zero, device=device), atol=1e-6, rtol=0.):
        raise ValueError('Entry bank action zero differs from current action configuration')
      if action_scale is not None and (not math.isfinite(action_scale)
          or not math.isclose(float(scale), float(action_scale), rel_tol=0., abs_tol=1e-9)):
        raise ValueError('Entry bank action scale differs from current action configuration')
      if not torch.allclose(self.data['limiter_position'], zero+scale*self.data['raw_action'],
          atol=2e-4, rtol=0.):
        raise ValueError('Entry bank limiter position does not match executed action')
      if (self.data['limiter_velocity'].abs() > limits+1e-5).any():
        raise ValueError('Entry bank limiter velocity exceeds configured velocity limits')
    self.device = device
    self._groups = {}

  def eligible(self, *, split, max_speed, max_fr_error):
    if split not in ('train', 'holdout'):
      raise ValueError('split must be train or holdout')
    if not all(math.isfinite(v) and v > 0 for v in (max_speed, max_fr_error)):
      raise ValueError('Entry difficulty bounds must be finite and positive')
    holdout = self.data['trajectory_id'] % 5 == 0
    mask = holdout if split == 'holdout' else ~holdout
    return (mask & (self.data['actual_speed'] <= max_speed)
      & (self.data['fr_error'] <= max_fr_error)).nonzero().flatten()

  def sample(self, count, *, split, max_speed, max_fr_error):
    key = (split, max_speed, max_fr_error)
    if key not in self._groups:
      ids = self.eligible(split=split, max_speed=max_speed, max_fr_error=max_fr_error)
      if not len(ids):
        raise ValueError(f'No eligible entry states for {key}')
      speed = torch.bucketize(self.data['actual_speed'][ids], ids.new_tensor(SPEED_EDGES, dtype=torch.float32))
      error = torch.bucketize(self.data['fr_error'][ids], ids.new_tensor(ERROR_EDGES, dtype=torch.float32))
      codes = speed*(len(ERROR_EDGES)+1)+error
      order = codes.argsort()
      _, counts = torch.unique_consecutive(codes[order], return_counts=True)
      starts = counts.cumsum(0)-counts
      self._groups[key] = (ids[order], counts, starts)
    ids, counts, starts = self._groups[key]
    bins = torch.randint(len(counts), (count,), device=ids.device)
    offsets = (torch.rand(count, device=ids.device)*counts[bins]).long()
    return ids[starts[bins]+offsets]

  def sample_trials(self, count, *, split, max_speed, max_fr_error):
    """Balance entry bins while taking at most one state per trajectory per pass.

    Small banks can still produce runtime smoke trials, but repeated
    trajectories are reported and cannot satisfy the curriculum evidence gate.
    """
    if type(count) is not int or count < 1:
      raise ValueError('Trial count must be a positive integer')
    ids = self.eligible(split=split, max_speed=max_speed, max_fr_error=max_fr_error)
    if not len(ids):
      raise ValueError('No eligible entry states for evaluation')
    speed = torch.bucketize(self.data['actual_speed'][ids], ids.new_tensor(SPEED_EDGES, dtype=torch.float32))
    error = torch.bucketize(self.data['fr_error'][ids], ids.new_tensor(ERROR_EDGES, dtype=torch.float32))
    codes = speed*(len(ERROR_EDGES)+1)+error
    trajectories = self.data['trajectory_id'][ids]
    available = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
    selected = []
    for _ in range(count):
      if not bool(available.any()):
        available.fill_(True)
      bins = codes[available].unique()
      code = bins[torch.randint(len(bins), (), device=ids.device)]
      rows = (available & (codes == code)).nonzero().flatten()
      row = rows[torch.randint(len(rows), (), device=ids.device)]
      selected.append(ids[row])
      available &= trajectories != trajectories[row]
    return torch.stack(selected)

  def coverage(self):
    result = {}
    for split in ('train', 'holdout'):
      result[split] = []
      for level, cfg in enumerate(LEVELS):
        ids = self.eligible(split=split, max_speed=cfg['max_speed'], max_fr_error=cfg['max_fr_error'])
        result[split].append(dict(level=level, states=len(ids),
          trajectories=len(self.data['trajectory_id'][ids].unique())))
    return result


@dataclass
class EntryCurriculum:
  level: int = 0
  streak: int = 0
  evaluations: int = 0

  @property
  def settings(self):
    return LEVELS[self.level]

  def observe(self, successes, attempts, *, distinct_trajectories):
    if (any(type(v) is not int for v in (successes, attempts, distinct_trajectories))
        or not 0 <= successes <= attempts or not 0 <= distinct_trajectories <= attempts):
      raise ValueError('Invalid held-out success counts')
    self.evaluations += 1
    if attempts < 64 or distinct_trajectories < 64:
      self.streak = 0
      return False
    self.streak = self.streak+1 if successes/attempts >= .8 else 0
    if self.streak >= 2 and self.level < len(LEVELS)-1:
      self.level += 1
      self.streak = 0
      return True
    return False

  def state_dict(self):
    return dict(level=self.level, streak=self.streak, evaluations=self.evaluations)

  def load_state_dict(self, state):
    if (set(state) != {'level', 'streak', 'evaluations'}
        or any(type(v) is not int or v < 0 for v in state.values())
        or state['level'] >= len(LEVELS)):
      raise ValueError('Invalid entry curriculum state')
    self.level, self.streak, self.evaluations = (state[k] for k in ('level', 'streak', 'evaluations'))
