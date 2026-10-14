"""Balanced sequence recording, episode boundaries, and actual actuator labels."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def recording():
    assert importlib.util.find_spec('src.tasks.fsm_student.recording') is not None, 'recording helpers are missing'
    return importlib.import_module('src.tasks.fsm_student.recording')


def script():
    path = ROOT/'scripts/record_fsm_teacher.py'
    assert path.is_file(), 'FSM teacher recording script is missing'
    spec = importlib.util.spec_from_file_location('record_fsm_teacher', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shard(categories, *, num_envs=2):
    steps = len(categories)
    starts = np.zeros((steps, num_envs), dtype=bool)
    starts[0] = True
    return dict(
        observations=np.zeros((steps, num_envs, 48), dtype=np.float32),
        actions=np.zeros((steps, num_envs, 12), dtype=np.float32),
        joint_targets=np.zeros((steps, num_envs, 12), dtype=np.float32),
        substep_actions=np.zeros((steps, num_envs, 4, 12), dtype=np.float32),
        commands=np.zeros((steps, num_envs, 6), dtype=np.float32),
        category=np.broadcast_to(np.asarray(categories, dtype=np.int16)[:, None], (steps, num_envs)).copy(),
        valid=np.ones((steps, num_envs), dtype=bool),
        done=np.zeros((steps, num_envs), dtype=bool), episode_starts=starts,
        trajectory_id=np.arange(num_envs, dtype=np.int64)+10,
        split=np.arange(num_envs, dtype=np.uint8)%2,
    )


def test_categories_cover_gaits_operations_and_both_directions_of_transitions():
    module = recording()
    owner = np.array([0, 0, 0, 0, 3, 3, 0, 0, 2, 4, 0, 0])
    phase = np.array([0, 0, 3, 2, 3, 3, 1, 4, 3, 3, 0, 3])
    blend = np.array([0., 0., 1., 1., 1., 1., 0., .4, 1., 1., 0., 1.])
    command = np.zeros((12, 6))
    command[[0, 2, 4], 0] = .2
    command[[2, 3, 4, 5, 6, 8, 9, 10], 5] = .12
    actual = module.classify_frames(owner, phase, blend, command)
    names = [module.CATEGORIES[index] for index in actual]
    assert names[:10] == list(module.CATEGORIES)
    assert names[10] == 'quadruped_to_tripod'  # Request before support FSM advances.
    assert names[11] == 'tripod_to_quadruped'  # Return request before LOWER begins.


def test_window_reservoir_balances_each_split_without_discarding_context():
    module = recording()
    data = shard(np.repeat(np.arange(10), 12))
    index = module.BalancedWindowIndex(train_per_category=4, validation_per_category=2,
        sequence_length=4, seed=7)
    index.add_shard(0, data)
    assert index.complete
    windows = index.arrays()
    for split, quota in ((0, 4), (1, 2)):
        np.testing.assert_array_equal(np.bincount(windows['category'][windows['split'] == split], minlength=10),
            np.full(10, quota))
    for row in range(len(windows['start'])):
        start, stop, env_id = [int(windows[key][row]) for key in ('start', 'stop', 'env')]
        assert stop-start == 4
        assert data['valid'][start:stop, env_id].all()
        assert data['category'][stop-1, env_id] == windows['category'][row]
    assert len(data['observations']) == 120  # Raw frames are retained in order.
    assert set(windows['trajectory_id'][windows['split'] == 0]).isdisjoint(
        windows['trajectory_id'][windows['split'] == 1])


def test_windows_never_cross_reset_or_invalid_frame():
    module = recording()
    data = shard([0]*12)
    data['episode_starts'][5, 0] = True
    data['valid'][8, 1] = False
    index = module.BalancedWindowIndex(train_per_category=100, validation_per_category=100,
        sequence_length=4, seed=3)
    index.add_shard(0, data)
    windows = index.arrays()
    for start, stop, env_id in zip(windows['start'], windows['stop'], windows['env']):
        assert data['valid'][start:stop, env_id].all()
        assert not data['episode_starts'][start+1:stop, env_id].any()
    assert not index.complete
    assert np.any(windows['start'] == 5)  # Starting at a real reset is permitted.


def test_failure_tail_excludes_terminal_and_post_reset_frames():
    module = recording()
    valid = np.ones((12, 2), dtype=bool)
    done = np.zeros_like(valid)
    done[7, 0] = True
    actual = module.exclude_failure_tails(valid, done, tail_frames=2)
    assert actual[:5, 0].all()
    assert not actual[5:, 0].any()
    assert actual[:, 1].all()
    assert valid.all()


def test_reservoir_is_reproducible_and_does_not_duplicate_windows():
    module = recording()
    data = shard(np.repeat(np.arange(10), 30))
    outputs = []
    for _ in range(2):
        index = module.BalancedWindowIndex(train_per_category=3, validation_per_category=2,
            sequence_length=4, seed=9)
        index.add_shard(0, data)
        outputs.append(index.arrays())
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
    rows = list(zip(outputs[0]['shard'], outputs[0]['env'], outputs[0]['stop']))
    assert len(rows) == len(set(rows))


def test_save_shard_round_trip_and_reject_nonfinite_training_labels(tmp_path):
    module = recording()
    data = shard([0]*6)
    path = tmp_path/'episode.npz'
    module.save_shard(path, data)
    with np.load(path, allow_pickle=False) as stored:
        for key in data:
            np.testing.assert_array_equal(stored[key], data[key])
    data['actions'][2, 0, 0] = np.nan
    with pytest.raises(ValueError):
        module.save_shard(tmp_path/'bad.npz', data)
    assert not (tmp_path/'bad.npz').exists()


def test_cli_defaults_and_positive_collection_limits():
    module = script()
    args = module.parse_args([])
    assert args.device == 'cuda:0'
    assert args.sequence_length == 32
    assert args.windows_per_category == 2000
    assert args.quad_checkpoint_file.name == 'quad_teacher.pt'
    for option in ('--num-envs', '--windows-per-category', '--sequence-length', '--max-batches'):
        with pytest.raises(SystemExit):
            module.parse_args([option, '0'])


def test_scripted_commands_preserve_frozen_transition_intent_and_return_without_r_bit():
    module = script()
    nominal = np.array([.13, .03, .48])
    scenario = module.ScriptedEpisode('rise', num_envs=2, nominal=nominal,
        seed=7, rise_seconds=3.2, down_seconds=3.7)
    alive = np.ones(2, dtype=bool)
    commands_seen = []
    stages_seen = []
    for _ in range(1600):
        command = scenario.command()
        assert command.shape == (2, 6)
        assert np.isfinite(command).all()
        name = scenario.stage_name
        commands_seen.append(command.copy())
        stages_seen.append(name)
        # Emulate completion at min duration; live runtime verifies actual FSM.
        owners = np.full(2, 3 if name == 'rise' else (0 if name == 'down' else 0))
        phases = np.zeros(2, dtype=int)
        scenario.advance(.02, owners, phases, alive)
        if scenario.finished:
            break
    assert scenario.finished
    rise = [command for name, command in zip(stages_seen, commands_seen) if name == 'rise']
    assert rise
    assert all(np.array_equal(command, rise[0]) for command in rise)
    down = [command for name, command in zip(stages_seen, commands_seen) if name == 'down']
    assert down and (down[0][:, 5] <= .25).all()
    returns = [command for name, command in zip(stages_seen, commands_seen) if name == 'return']
    assert returns and not np.asarray(returns).any()


def test_capture_keeps_applied_action_before_auto_reset():
    module = script()
    action = SimpleNamespace(raw_action=torch.zeros(2, 12), joint_ids=torch.arange(12),
        _entity=SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 12))))
    def write(request):
        # Stand-in for the actuator interface: actual output differs from request.
        action.raw_action.copy_(request*.1)
        action._entity.data.joint_pos_target.copy_(action.raw_action*.25+1.)
    action._write_target = write
    with module.AppliedActionCapture(action) as capture:
        capture.begin()
        for substep in range(4):
            action._write_target(torch.full((2, 12), float(substep+1)))
        action.raw_action.zero_()  # Automatic reset after physics must not erase labels.
        action._entity.data.joint_pos_target.zero_()
        actions, targets, substeps = capture.finish(4)
        torch.testing.assert_close(actions, torch.full((2, 12), .4))
        torch.testing.assert_close(targets, torch.full((2, 12), 1.1))
        assert substeps.shape == (2, 4, 12)
        torch.testing.assert_close(substeps[:, 0], torch.full((2, 12), .1))
    assert action._write_target is write


def test_scenario_selection_targets_the_missing_category():
    module = script()
    counts = np.full((2, 10), 10)
    counts[:, 8] = 0  # Rise has no coverage.
    assert module.choose_scenario(counts, np.array([10, 2]))[0] == 'rise'
    counts[:, 8] = 10
    counts[:, 4] = 0  # Biped walk can be collected from a physical biped reset.
    assert module.choose_scenario(counts, np.array([10, 2]))[0] == 'biped'
