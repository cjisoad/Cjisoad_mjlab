"""Sequence data, anchor losses, and checkpoint helpers for student distillation."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from .observations import observation_contract
from .policy import GRUStudent, GRUStudentConfig

ACTION_DIM = 12
DISTILLATION_VERSION = 1


def _finite(name, value):
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains nonfinite values')


def _check_window(start, stop, data, env):
    if stop <= start or start < 0 or stop > data['observations'].shape[0]:
        raise ValueError('Invalid sequence window')
    if data['observations'].shape[-1] != 48 or data['actions'].shape[-1] != ACTION_DIM:
        raise ValueError('Student window has an incompatible observation/action dimension')
    if not data['valid'][start:stop, env].all() or data['done'][start:stop, env].any():
        raise ValueError('Sequence window contains invalid or terminal frames')
    if data['episode_starts'][start+1:stop, env].any():
        raise ValueError('Sequence window crosses an episode reset')


def _item(observations, actions, category):
    obs = torch.from_numpy(np.ascontiguousarray(observations, dtype=np.float32))
    target = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
    return {'observations': obs, 'actions': target,
            'category': torch.tensor(int(category), dtype=torch.long)}


class WindowDataset(Dataset):
    """Lazy-ish indexed access to recorder shards with contiguous GRU history."""
    def __init__(self, root, *, split=0, max_windows=None):
        self.root = Path(root)
        with np.load(self.root/'windows.npz', allow_pickle=False) as index:
            keys = ('shard', 'env', 'start', 'stop', 'category', 'split', 'trajectory_id')
            values = {key: np.asarray(index[key]) for key in keys}
        if split not in (0, 1):
            raise ValueError('Window split must be 0 (train) or 1 (validation)')
        selected = np.flatnonzero(values['split'] == split)
        if max_windows is not None:
            limit = int(max_windows)
            if limit < 1:
                raise ValueError('max_windows must be positive')
            # Keep a small smoke subset representative of every available
            # category instead of taking the first category block from the
            # recorder's deterministic index order.
            per_category = []
            for category in range(10):
                rows = selected[values['category'][selected] == category]
                per_category.append(rows)
            selected = np.asarray([row for offset in range(limit)
                for rows in per_category
                if offset < len(rows) for row in (rows[offset],)], dtype=np.int64)
            selected = selected[:limit]
        self.rows = {key: value[selected] for key, value in values.items()}
        self._shards = {}
        for shard in np.unique(self.rows['shard']):
            path = self.root/f'shard_{int(shard):05d}.npz'
            if not path.is_file():
                raise FileNotFoundError(path)
            stored = np.load(path, allow_pickle=False)
            self._shards[int(shard)] = {key: stored[key] for key in stored.files}
        self._validate_rows()

    def _validate_rows(self):
        for row in range(len(self)):
            shard = int(self.rows['shard'][row]); env = int(self.rows['env'][row])
            data = self._shards[shard]
            _check_window(int(self.rows['start'][row]), int(self.rows['stop'][row]), data, env)
            if int(data['category'][int(self.rows['stop'][row])-1, env]) != int(self.rows['category'][row]):
                raise ValueError('Window anchor category disagrees with shard')

    def __len__(self):
        return len(self.rows['start'])

    def __getitem__(self, index):
        index = int(index)
        shard = int(self.rows['shard'][index]); env = int(self.rows['env'][index])
        start = int(self.rows['start'][index]); stop = int(self.rows['stop'][index])
        data = self._shards[shard]
        return _item(data['observations'][start:stop, env], data['actions'][start:stop, env],
            self.rows['category'][index])


class ArrayWindowDataset(Dataset):
    """Build contiguous windows from one or more in-memory DAgger trajectories."""
    def __init__(self, data, *, sequence_length):
        self.data = {key: np.asarray(value) for key, value in data.items()}
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError('Sequence length must be positive')
        obs, actions = self.data['observations'], self.data['actions']
        if obs.ndim != 3 or obs.shape[-1] != 48 or actions.shape != (*obs.shape[:2], 12):
            raise ValueError('Replay observations/actions must be [time, world, 48/12]')
        valid = self.data.get('valid', np.ones(obs.shape[:2], bool))
        done = self.data.get('done', np.zeros(obs.shape[:2], bool))
        starts = self.data.get('episode_starts', np.zeros(obs.shape[:2], bool))
        category = self.data.get('category', np.zeros(obs.shape[:2], np.int16))
        if valid.shape != obs.shape[:2] or done.shape != valid.shape or starts.shape != valid.shape:
            raise ValueError('Replay masks must match [time, world]')
        rows = []
        for env in range(obs.shape[1]):
            run = 0
            for step in range(obs.shape[0]):
                if starts[step, env]:
                    run = 0
                if not valid[step, env] or done[step, env]:
                    run = 0
                    continue
                run += 1
                if run >= self.sequence_length:
                    rows.append((env, step+1-self.sequence_length, step+1,
                        int(category[step, env])))
        self.rows = rows
        for key in ('observations', 'actions'):
            _finite(key, self.data[key])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        env, start, stop, category = self.rows[int(index)]
        return _item(self.data['observations'][start:stop, env],
            self.data['actions'][start:stop, env], category)


def anchor_smooth_l1_loss(predicted, target):
    if predicted.ndim != 3 or target.shape != predicted.shape or predicted.shape[-1] != ACTION_DIM:
        raise ValueError('Distillation tensors must have shape [batch, time, 12]')
    return torch.nn.functional.smooth_l1_loss(predicted[:, -1], target[:, -1])


def train_epoch(model, loader, optimizer, *, device='cpu', grad_clip=1.0):
    model.train()
    total = 0.; count = 0
    for batch in loader:
        observations = batch['observations'].to(device)
        target = batch['actions'].to(device)
        predicted, _ = model(observations)
        loss = anchor_smooth_l1_loss(predicted, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        size = observations.shape[0]
        total += float(loss.detach()) * size
        count += size
    return total / count if count else float('nan')


@torch.no_grad()
def evaluate_epoch(model, loader, *, device='cpu'):
    model.eval()
    total = 0.; count = 0
    for batch in loader:
        observations = batch['observations'].to(device)
        target = batch['actions'].to(device)
        predicted, _ = model(observations)
        loss = anchor_smooth_l1_loss(predicted, target)
        size = observations.shape[0]
        total += float(loss) * size
        count += size
    return total / count if count else float('nan')


def _checkpoint_payload(model, optimizer, *, stage, epoch, metrics, seed):
    return {
        'version': 2,
        'config': asdict(model.cfg),
        'state_dict': model.state_dict(),
        'observation_contract': observation_contract(),
        'action_contract': model.action_contract(),
        'distillation_version': DISTILLATION_VERSION,
        'training': {'stage': stage, 'epoch': int(epoch), 'metrics': dict(metrics), 'seed': int(seed)},
        'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
    }


def save_student_checkpoint(path, model, *, optimizer=None, stage, epoch, metrics, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}')
    payload = _checkpoint_payload(model, optimizer, stage=stage, epoch=epoch,
        metrics=metrics, seed=seed)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_student_checkpoint(path, *, device='cpu', optimizer=None):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('version') != 2 or payload.get('observation_contract') != observation_contract():
        raise ValueError('Checkpoint does not match the student observation contract')
    model = GRUStudent(GRUStudentConfig(**payload['config'])).to(device)
    if payload.get('action_contract') != model.action_contract():
        raise ValueError('Checkpoint does not match the student action contract')
    model.load_state_dict(payload['state_dict'], strict=True)
    if optimizer is not None and payload.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    return model, dict(payload.get('training') or {})


def save_metrics(path, metrics):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
