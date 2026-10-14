"""Chronological teacher archives and balanced GRU context-window anchors."""
from pathlib import Path
import os
import tempfile

import numpy as np

CATEGORIES = (
    'quadruped_walk', 'quadruped_stand', 'tripod_walk', 'tripod_manipulation',
    'biped_walk', 'biped_manipulation', 'quadruped_to_tripod',
    'tripod_to_quadruped', 'rise', 'down',
)


def classify_frames(owner, phase, blend, command):
    owner, phase, blend, command = map(np.asarray, (owner, phase, blend, command))
    moving = np.linalg.norm(command[..., :3], axis=-1) > 1e-5
    active = np.linalg.norm(command[..., 3:], axis=-1) > 1e-5
    result = np.where(moving, 0, 1).astype(np.int16)
    tripod = (phase == 3) | ((phase == 2) & (blend >= 1.-1e-6))
    result = np.where(tripod, np.where(moving, 2, 3), result)
    lifting = (phase == 1) | ((phase == 2) & (blend < 1.-1e-6)) | ((phase == 0) & active)
    returning = (phase == 4) | (phase == 5) | (tripod & ~active)
    result = np.where(lifting, 6, result)
    result = np.where(returning, 7, result)
    result = np.where(owner == 3, np.where(moving, 4, 5), result)
    result = np.where(owner == 2, 8, result)
    return np.where(owner == 4, 9, result).astype(np.int16)


def exclude_failure_tails(valid, done, tail_frames):
    if tail_frames < 0:
        raise ValueError('Failure tail must be nonnegative')
    result = np.array(valid, dtype=bool, copy=True)
    for env in range(result.shape[1]):
        terminals = np.flatnonzero(done[:, env])
        if len(terminals):
            result[max(0, int(terminals[0])-tail_frames):, env] = False
    return result


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_shard(path, data):
    categories = data['category']
    if categories.ndim != 2 or min(categories.shape) < 1:
        raise ValueError('Shard requires nonempty [time, world] frames')
    steps, worlds = categories.shape
    shapes = {'observations': (steps, worlds, 48), 'actions': (steps, worlds, 12),
        'joint_targets': (steps, worlds, 12), 'commands': (steps, worlds, 6),
        'valid': (steps, worlds), 'done': (steps, worlds), 'episode_starts': (steps, worlds),
        'trajectory_id': (worlds,), 'split': (worlds,)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f'Invalid {key} shape: expected {shape}')
    substeps = data['substep_actions']
    if substeps.ndim != 4 or substeps.shape[:2] != (steps, worlds) or substeps.shape[2] < 1 or substeps.shape[3] != 12:
        raise ValueError('Substep actions must be [time, world, decimation, 12]')
    if np.any((categories < 0) | (categories >= len(CATEGORIES))) or np.any(data['split'] > 1):
        raise ValueError('Invalid category or split')
    for key, value in data.items():
        if value.dtype.hasobject or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
            raise ValueError(f'Nonfinite or object data in {key}')
    atomic_npz(path, data)


class BalancedWindowIndex:
    """Uniform reservoirs of unique windows, balanced by their last frame.

    Split IDs: 0 train, 1 validation. Context may contain other categories.
    Raw shards are never subsampled or reordered.
    """
    def __init__(self, *, train_per_category, validation_per_category, sequence_length, seed):
        if min(train_per_category, validation_per_category, sequence_length) < 1:
            raise ValueError('Quotas and sequence length must be positive')
        self.quotas = np.array([train_per_category, validation_per_category], dtype=np.int64)
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)
        self.seen = np.zeros((2, len(CATEGORIES)), dtype=np.int64)
        self.reservoirs = [[[] for _ in CATEGORIES] for _ in range(2)]
        self.shards = set()

    @property
    def counts(self):
        return np.array([[len(rows) for rows in split] for split in self.reservoirs])

    @property
    def complete(self):
        return bool(np.all(self.counts == self.quotas[:, None]))

    def add_shard(self, shard_id, data):
        if shard_id in self.shards:
            raise ValueError('Shard already indexed')
        self.shards.add(shard_id)
        for env in range(data['category'].shape[1]):
            run = 0
            split = int(data['split'][env])
            for step in range(data['category'].shape[0]):
                if data['episode_starts'][step, env]:
                    run = 0
                if not data['valid'][step, env] or data['done'][step, env]:
                    run = 0
                    continue
                run += 1
                if run < self.sequence_length:
                    continue
                category = int(data['category'][step, env])
                self.seen[split, category] += 1
                rows = self.reservoirs[split][category]
                row = (shard_id, env, step+1-self.sequence_length, step+1,
                    category, split, int(data['trajectory_id'][env]))
                quota = int(self.quotas[split])
                if len(rows) < quota:
                    rows.append(row)
                else:
                    slot = int(self.rng.integers(self.seen[split, category]))
                    if slot < quota:
                        rows[slot] = row

    def arrays(self):
        rows = [row for split in self.reservoirs for category in split for row in category]
        table = np.asarray(rows, dtype=np.int64).reshape(-1, 7)
        return {key: table[:, index] for index, key in enumerate(
            ('shard', 'env', 'start', 'stop', 'category', 'split', 'trajectory_id'))}
