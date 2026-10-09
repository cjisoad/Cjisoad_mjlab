"""Offline path-filtered targets from the same FR geometry used by the keyboard."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .reachability import DEFAULT_CACHE, ReachabilityGrid, analytic_ik, chain_parameters, geometry_fingerprint

TRAINING_BANK_VERSION = 1
DEFAULT_TRAINING_CACHE = DEFAULT_CACHE.with_name(DEFAULT_CACHE.stem+'_train.npz')


def file_fingerprint(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''):
      digest.update(block)
  return digest.hexdigest()


@dataclass(frozen=True)
class TrainingWorkspace:
  zero: np.ndarray
  offsets: np.ndarray
  joint_positions: np.ndarray
  segment: np.ndarray
  nominal_height: float
  foot_radius: float
  metadata: dict


@lru_cache(maxsize=4)
def load_training_workspace(path=DEFAULT_TRAINING_CACHE):
  """Load once per process; neither initialization nor reset performs IK."""
  path = Path(path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(f'Generate the FR training cache first: python scripts/build_teacher_reachability.py --training-only --training-output {path}')
  with np.load(path, allow_pickle=False) as data:
    metadata = json.loads(str(data['metadata']))
    if (metadata['training_bank_version'] != TRAINING_BANK_VERSION
        or metadata['geometry_sha256'] != geometry_fingerprint()):
      raise ValueError('FR training cache does not match this robot; regenerate it')
    offsets = data['offsets'].copy()
    zero = data['zero'].astype(float)
    joints = data['joint_positions'].copy()
  if not len(offsets) or offsets.ndim != 2 or offsets.shape[1] != 3:
    raise ValueError('FR training cache has no valid targets')
  radius = metadata['foot_radius_m']
  return TrainingWorkspace(zero, offsets, joints, np.zeros(len(offsets), dtype=np.int8),
                           float(radius-zero[2]), radius, metadata)


def build_training_cache(geometry_path=DEFAULT_CACHE, output=DEFAULT_TRAINING_CACHE,
                         *, margin=.01, min_lift=.05, path_step=.002,
                         max_joint_step=.15, chunk_size=4096, progress=print):
  """Filter nominal straight reaches/returns offline with sampled geometry/IK.

  The nominal landing offset is zero, so its descent retraces the same segment.
  Live base height, support-leg poses and random dynamics can change the path;
  these offline samples are not a guarantee for every physical rollout.
  """
  import time
  from src.tasks.pedipulation.constants import DEFAULT_ANGLES
  started = time.monotonic()
  if not np.isfinite((margin, min_lift, path_step, max_joint_step)).all() or min(margin, min_lift, path_step, max_joint_step) <= 0 or chunk_size < 1:
    raise ValueError('Training margins, lift, path resolution and chunk size must be positive')
  grid = ReachabilityGrid(geometry_path)
  if margin < grid.margin:
    raise ValueError('Training margin must be at least the keyboard query margin')
  parameters = chain_parameters()
  selected = ((grid.targets[:, 2] >= min_lift-1e-7)
              & (grid.query(grid.targets) >= margin))
  candidates = grid.targets[selected]
  candidate_joints = np.zeros_like(candidates)
  keep = np.zeros(len(candidates), dtype=bool)
  seed = np.asarray(DEFAULT_ANGLES[3:6])
  if grid.query(np.zeros(3)) < margin:
    raise ValueError('Reference foot zero has insufficient training-boundary margin')
  for begin in range(0, len(candidates), chunk_size):
    goal = candidates[begin:begin+chunk_size]
    active = np.ones(len(goal), dtype=bool)
    previous = np.repeat(seed[None], len(goal), axis=0)
    steps = max(1, int(np.ceil(np.linalg.norm(goal, axis=-1).max()/path_step)))
    for fraction in np.linspace(0., 1., steps+1)[1:]:
      ids = np.flatnonzero(active)
      if not len(ids):
        break
      points = goal[ids]*fraction
      in_region = grid.query(points) >= margin
      active[ids[~in_region]] = False
      ids, points = ids[in_region], points[in_region]
      if not len(ids):
        continue
      branches = analytic_ik(points+grid.zero, parameters, grid.metadata['joint_margin_rad'])
      angles = np.stack([branch[0] for branch in branches])
      valid = np.stack([branch[1] for branch in branches])
      jumps = np.max(np.abs(angles-previous[ids][None]), axis=-1)
      jumps[~valid] = np.inf
      choice = np.argmin(jumps, axis=0)
      good = jumps[choice, np.arange(len(ids))] <= max_joint_step
      active[ids[~good]] = False
      previous[ids[good]] = angles[choice[good], np.flatnonzero(good)]
    keep[begin:begin+len(goal)] = active
    candidate_joints[begin:begin+len(goal)] = previous
    if begin % (chunk_size*8) == 0 or begin+len(goal) == len(candidates):
      progress(f'FR training paths {begin+len(goal)}/{len(candidates)}: accepted={int(keep.sum())}, elapsed={time.monotonic()-started:.1f}s', flush=True)
  offsets = candidates[keep]
  witnesses = candidate_joints[keep]
  easy = (np.abs(offsets[:, :2]) <= .045).all(-1) & (offsets[:, 2] <= .085)
  if not len(offsets) or not easy.any():
    raise RuntimeError('Path filtering removed all training targets or the easy curriculum')
  bounds = np.column_stack((offsets.min(0), offsets.max(0)))
  metadata = dict(training_bank_version=TRAINING_BANK_VERSION,
    geometry_sha256=geometry_fingerprint(), geometry_cache_sha256=file_fingerprint(geometry_path),
    geometry_cache_version=grid.metadata['cache_version'], spacing_m=grid.spacing,
    training_margin_m=margin, min_lift_m=min_lift, path_step_m=path_step,
    max_joint_step_rad=max_joint_step, foot_radius_m=grid.metadata['foot_radius_m'],
    candidate_count=len(candidates), target_count=len(offsets), easy_target_count=int(easy.sum()),
    offset_bounds_m=bounds.tolist(), build_seconds=time.monotonic()-started,
    path_reference='nominal shared quadruped zero to target and reverse to nominal landing',
    limitation='Sampled nominal geometric paths with continuous analytic IK branch selection. Collision membership uses cached nominal support-leg geometry; live poses, branch-specific collision clearance and dynamic feasibility are not certified.')
  output = Path(output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix('.tmp.npz')
  np.savez_compressed(temporary, zero=grid.zero, offsets=offsets,
                      joint_positions=witnesses, metadata=json.dumps(metadata, sort_keys=True))
  temporary.replace(output)
  output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
  progress(f'Saved {output}: {len(offsets)} targets, easy={int(easy.sum())}', flush=True)
  return metadata
