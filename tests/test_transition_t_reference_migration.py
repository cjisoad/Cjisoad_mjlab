"""A new rise reference must propagate everywhere without weakening physics checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.tasks.transition_t.motion import DEFAULT_MOTION_PATH, TransitionMotion


def test_custom_motion_reaches_entry_config_and_both_command_lines():
    from src.tasks.transition_t.entry_course import entry_course_env_cfg
    from scripts.train_transition_entry import parse_args as train_args
    from scripts.evaluate_transition_entry import parse_args as eval_args
    path = Path('/tmp/custom_rise.npz')
    cfg = entry_course_env_cfg('bank.npz', motion_file=path)
    assert cfg.commands['motion'].motion_file == str(path)
    assert train_args(['--entry-bank', 'bank.npz', '--motion-file', str(path),
                       '--allow-reference-migration']).allow_reference_migration
    assert eval_args(['--entry-bank', 'bank.npz', '--checkpoint', 'model.pt',
                      '--motion-file', str(path)]).motion_file == path
    with pytest.raises(SystemExit):
        train_args(['--entry-bank', 'bank.npz', '--resume', '--allow-reference-migration'])


def runner_fixture():
    from src.tasks.transition_t.rl import TransitionOnPolicyRunner
    runner = object.__new__(TransitionOnPolicyRunner)
    contract = dict(version=3, motion_sha256='a'*64, adaptive_bins=5,
                    actor_dim=75, action_scale=.25, compiled_physics_sha256='physics',
                    target_limiter={'velocity_limits': [2.]*12})
    runner._contract = lambda: copy.deepcopy(contract)
    runner.term = SimpleNamespace(bin_failed_count=torch.ones(5), _current_bin_failed=torch.ones(5))
    runner.alg = SimpleNamespace(actor=torch.nn.Linear(1, 1), critic=torch.nn.Linear(1, 1))
    return runner, contract


def test_reference_migration_is_weights_only_and_clears_old_sampling(tmp_path, monkeypatch):
    from mjlab.rl import MjlabOnPolicyRunner
    runner, contract = runner_fixture()
    old = {**contract, 'motion_sha256': 'b'*64, 'adaptive_bins': 8}
    p = tmp_path/'old.pt'
    torch.save({'infos': {'transition_t_contract': old,
                         'transition_t_sampling': {'bin_failed_count': torch.ones(8)}}}, p)
    calls = []
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', lambda *a, **kw: calls.append(kw) or {})
    with pytest.raises(ValueError, match='contract'):
        runner.load(p, load_cfg={'actor': True, 'critic': True})
    with pytest.raises(ValueError, match='weights|optimizer|iteration'):
        runner.load(p, allow_reference_migration=True)
    runner.load(p, load_cfg={'actor': True, 'critic': True}, allow_reference_migration=True)
    assert len(calls) == 1
    assert not runner.term.bin_failed_count.any()
    assert not runner.term._current_bin_failed.any()


def test_reference_migration_rejects_other_contract_changes(tmp_path, monkeypatch):
    from mjlab.rl import MjlabOnPolicyRunner
    runner, contract = runner_fixture()
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', lambda *a, **kw: pytest.fail('weights loaded'))
    for key, value in [('actor_dim', 54), ('action_scale', .5),
                       ('compiled_physics_sha256', 'changed'), ('target_limiter', None)]:
        old = {**contract, 'motion_sha256': 'b'*64, 'adaptive_bins': 8, key: value}
        p = tmp_path/'bad.pt'; torch.save({'infos': {'transition_t_contract': old}}, p)
        with pytest.raises(ValueError, match='contract'):
            runner.load(p, load_cfg={'actor': True}, allow_reference_migration=True)


def test_bank_retarget_recomputes_fr_error_and_preserves_physical_states(tmp_path):
    from src.tasks.transition_t.entry_retarget import retarget_entry_bank
    old = TransitionMotion(DEFAULT_MOTION_PATH)
    arrays = {k: np.array(v) for k, v in old.arrays.items()}
    arrays['fps'] = np.array(50)
    arrays['joint_names'] = np.array(old.joint_names)
    arrays['body_names'] = np.array(old.body_names)
    fr = old.body_name_to_index['FR_foot']; base = old.body_name_to_index['base_link']
    arrays['body_pos_w'][:, fr, 0] += .04
    target = tmp_path/'new.npz'; np.savez_compressed(target, **arrays)
    n = 80
    root = np.tile(np.r_[old.body_pos_w[0, base], old.body_quat_w[0, base], np.zeros(6)], (n, 1))
    data = dict(root_state=root, joint_pos=np.tile(old.joint_pos[0], (n, 1)),
                joint_vel=np.zeros((n, 12)), actual_speed=np.zeros(n), fr_error=np.full(n, .123),
                trajectory_id=np.arange(n), fr_offset=np.zeros((n, 3)), command=np.zeros((n, 6)),
                manager_actions=np.zeros((n, 3, 12)))
    for key in ('raw_action', 'previous_action', 'previous_joint_vel'): data[key] = np.zeros((n, 12))
    source = tmp_path/'bank.npz'
    np.savez_compressed(source, **data, metadata_json=np.array(json.dumps(dict(
        schema_version=1, motion_sha256=old.sha256, teacher_sha256='teacher'))))
    output = tmp_path/'new_bank.npz'
    report = retarget_entry_bank(source, target, output, source_motion_file=DEFAULT_MOTION_PATH)
    with np.load(output) as result:
        for key, value in data.items():
            if key != 'fr_error': np.testing.assert_array_equal(result[key], value)
        np.testing.assert_allclose(result['fr_error'], .04, atol=1e-6)
        meta = json.loads(result['metadata_json'].item())
        assert meta['motion_sha256'] == TransitionMotion(target).sha256
        assert meta['reference_retarget']['source_motion_sha256'] == old.sha256
    assert report['states'] == n
    with pytest.raises(FileExistsError):
        retarget_entry_bank(source, target, output, source_motion_file=DEFAULT_MOTION_PATH)
    with pytest.raises(ValueError, match='source|reference'):
        retarget_entry_bank(source, target, tmp_path/'bad.npz', source_motion_file=target)
