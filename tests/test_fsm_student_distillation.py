"""Unit tests for BC/DAgger sequence training primitives."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.fsm_student import GRUStudent, GRUStudentConfig
from src.tasks.fsm_student.distillation import (
    ArrayWindowDataset, WindowDataset, anchor_smooth_l1_loss,
    load_student_checkpoint, save_student_checkpoint, train_epoch,
)


def make_dataset(root, *, steps=18, worlds=2):
    root.mkdir()
    observations = np.zeros((steps, worlds, 48), np.float32)
    actions = np.zeros((steps, worlds, 12), np.float32)
    observations[:, :, 0] = np.arange(steps)[:, None]
    actions[:, :, 0] = observations[:, :, 0] * .1
    data = dict(
        observations=observations, actions=actions,
        joint_targets=np.zeros((steps, worlds, 12), np.float32),
        substep_actions=np.zeros((steps, worlds, 4, 12), np.float32),
        commands=np.zeros((steps, worlds, 6), np.float32),
        category=np.zeros((steps, worlds), np.int16), valid=np.ones((steps, worlds), bool),
        done=np.zeros((steps, worlds), bool), episode_starts=np.zeros((steps, worlds), bool),
        trajectory_id=np.arange(worlds, dtype=np.int64), split=np.zeros(worlds, np.uint8),
    )
    data['episode_starts'][0] = True
    np.savez_compressed(root/'shard_00000.npz', **data)
    np.savez_compressed(root/'windows.npz', shard=np.zeros(4, np.int64),
        env=np.zeros(4, np.int64), start=np.arange(4, dtype=np.int64),
        stop=np.arange(4, dtype=np.int64)+4, category=np.zeros(4, np.int64),
        split=np.zeros(4, np.int64), trajectory_id=np.zeros(4, np.int64))
    return data


def test_window_dataset_reads_contiguous_history_and_anchor(tmp_path):
    make_dataset(tmp_path/'dataset')
    dataset = WindowDataset(tmp_path/'dataset', split=0)
    assert len(dataset) == 4
    item = dataset[1]
    assert item['observations'].shape == (4, 48)
    assert item['actions'].shape == (4, 12)
    assert item['observations'][-1, 0] == 4
    assert item['category'].item() == 0


def test_anchor_loss_only_supervises_last_frame():
    prediction = torch.zeros(2, 3, 12)
    target = torch.zeros_like(prediction)
    target[:, 0] = 99.
    target[:, -1] = 1.
    assert anchor_smooth_l1_loss(prediction, target).item() == pytest.approx(.5)


def test_training_updates_gru_and_checkpoint_round_trip(tmp_path):
    data = dict(observations=np.random.default_rng(1).normal(size=(12, 1, 48)).astype(np.float32),
        actions=np.random.default_rng(2).normal(size=(12, 1, 12)).astype(np.float32),
        valid=np.ones((12, 1), bool), done=np.zeros((12, 1), bool),
        episode_starts=np.r_[True, np.zeros(11, bool)][:, None],
        category=np.zeros((12, 1), np.int16), trajectory_id=np.array([4]), split=np.array([0], np.uint8))
    dataset = ArrayWindowDataset(data, sequence_length=4)
    model = GRUStudent(GRUStudentConfig(encoder_dims=(16,), hidden_size=16, head_dims=(16,)))
    before = [value.detach().clone() for value in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_epoch(model, torch.utils.data.DataLoader(dataset, batch_size=2), optimizer, device='cpu')
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    path = tmp_path/'student.pt'
    save_student_checkpoint(path, model, optimizer=optimizer, stage='bc', epoch=1,
        metrics={'loss': loss}, seed=3)
    restored, metadata = load_student_checkpoint(path, device='cpu')
    assert metadata['stage'] == 'bc'
    torch.testing.assert_close(model(torch.zeros(1, 4, 48))[0], restored(torch.zeros(1, 4, 48))[0])


def test_short_episodes_produce_padded_windows_without_crossing_reset():
    steps = 7
    data = dict(observations=np.ones((steps, 1, 48), np.float32),
        actions=np.ones((steps, 1, 12), np.float32),
        valid=np.array([1, 1, 0, 1, 1, 1, 0], bool)[:, None],
        done=np.array([0, 0, 1, 0, 0, 0, 1], bool)[:, None],
        episode_starts=np.array([1, 0, 0, 1, 0, 0, 0], bool)[:, None],
        category=np.array([4, 4, 4, 5, 5, 5, 5], np.int16)[:, None])
    data['observations'][:, 0, 0] = np.arange(steps)
    dataset = ArrayWindowDataset(data, sequence_length=32)
    assert len(dataset) == 5
    assert dataset[1]['mask'].sum() == 2
    assert dataset[2]['mask'].sum() == 1
    assert dataset[2]['observations'][0, 0] == 3
    assert dataset[2]['observations'].shape == (32, 48)
    assert dataset[2]['categories'][0] == 5
    assert (dataset[2]['categories'][1:] == -1).all()
    assert not dataset[2]['observations'][1:].any()


def test_bc_adds_supervision_for_episode_prefixes(tmp_path):
    make_dataset(tmp_path/'dataset')
    dataset = WindowDataset(tmp_path/'dataset', split=0, include_prefix=True)
    prefixes = [dataset[i] for i in range(len(dataset)) if dataset[i]['mask'].sum() < 4]
    assert sorted(int(item['mask'].sum()) for item in prefixes) == [1, 2, 3]
    assert all(item['observations'][0, 0] == 0 for item in prefixes)


def test_full_frame_loss_supervises_early_frames_and_ignores_padding():
    from src.tasks.fsm_student.distillation import masked_sequence_loss
    prediction = torch.zeros(1, 4, 12, requires_grad=True)
    target = torch.ones_like(prediction)
    target[:, 2:] = float('nan')
    mask = torch.tensor([[True, True, False, False]])
    loss = masked_sequence_loss(prediction, target, mask)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert prediction.grad[:, :2].abs().sum() > 0
    assert prediction.grad[:, 2:].abs().sum() == 0


def test_full_frame_loss_uses_frame_categories_not_window_anchor():
    from src.tasks.fsm_student.distillation import masked_sequence_loss
    prediction = torch.zeros(1, 4, 12)
    target = torch.zeros_like(prediction)
    target[:, :3] = 1.
    mask = torch.ones(1, 4, dtype=torch.bool)
    categories = torch.tensor([[0, 0, 0, 1]])
    weights = torch.tensor([1/3, 1.])
    assert masked_sequence_loss(prediction, target, mask,
        categories=categories, category_weights=weights).item() == pytest.approx(.25)
    with pytest.raises(ValueError, match='valid'):
        masked_sequence_loss(prediction, target, torch.zeros_like(mask))


def test_warm_start_timing_is_per_environment_after_each_reset():
    from src.tasks.fsm_student.distillation import warm_start_mask
    steps = torch.tensor([0, 49, 50, 70])
    assert warm_start_mask(steps, step_dt=.02, seconds=1.).tolist() == [True, True, False, False]
    steps[2] = 0
    assert warm_start_mask(steps, step_dt=.02, seconds=1.).tolist() == [True, True, True, False]
    assert not warm_start_mask(steps, step_dt=.02, seconds=0.).any()
    with pytest.raises(ValueError):
        warm_start_mask(steps, step_dt=.02, seconds=-1.)
