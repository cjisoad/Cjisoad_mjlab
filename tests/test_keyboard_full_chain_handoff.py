"""Real command/observation regressions for bidirectional keyboard handoffs."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import keyboard_transition_fsm as fsm


@pytest.fixture(scope='module')
def chain():
    loaded = fsm.load_full_chain(fsm.parse_args(['--device', 'cpu']))
    try:
        yield loaded
    finally:
        loaded[0].close()


@pytest.fixture
def ready_chain(chain):
    env, wrapped, action, term, rise, down, paths = chain
    term._ensure_motion_references()
    # Isolate each regression even if an earlier test hit the old stale index.
    term.motion_term.time_steps.zero_()
    term._select_reference(term.down_reference)
    wrapped.reset()
    return chain


def finish_rise(chain):
    env, _, _, term, *_ = chain
    term._select_reference(term.rise_reference)
    motion = term.motion_term
    motion.time_steps[:] = term.rise_reference.num_frames - 1
    motion.completed[:] = True
    motion.active[:] = True
    term.owner[:] = fsm.RISE
    env.common_step_counter += 1
    term._update_command()


def test_rise_last_frame_handoff_keeps_observations_readable(ready_chain):
    env, _, action, term, *_ = ready_chain
    before = [value.clone() for value in (env.sim.data.qpos, env.sim.data.qvel,
        action.raw_action, action.previous_action,
        action.target_limiter.position, action.target_limiter.velocity)]
    finish_rise(ready_chain)
    assert int(term.owner[0]) == fsm.BIPED
    assert not term.motion_term.active.any()
    assert term.motion_term.reference is term.rise_reference
    obs = env.observation_manager.compute()
    assert torch.isfinite(obs['actor']).all()
    assert obs['actor'].shape == (1, 75)
    for expected, value in zip(before, (env.sim.data.qpos, env.sim.data.qvel,
        action.raw_action, action.previous_action,
        action.target_limiter.position, action.target_limiter.velocity)):
        torch.testing.assert_close(value.clone(), expected, atol=0, rtol=0)


def test_reference_swap_resets_indices_and_sampling_buffers(ready_chain):
    _, _, _, term, *_ = ready_chain
    term._select_reference(term.rise_reference)
    motion = term.motion_term
    assert motion.bin_failed_count.shape == (motion.bin_count,)
    assert motion._current_bin_failed.shape == (motion.bin_count,)
    motion.time_steps[:] = term.rise_reference.num_frames - 1
    motion.completed[:] = True
    term._select_reference(term.down_reference)
    assert motion.time_steps.tolist() == [0]
    assert not motion.completed.any()
    assert motion.bin_failed_count.shape == (motion.bin_count,)
    assert motion._current_bin_failed.shape == (motion.bin_count,)
    assert torch.isfinite(motion.joint_pos).all()


def test_quadruped_first_key_and_r_use_quadruped_zero(ready_chain):
    _, _, _, term, *_ = ready_chain
    inputs = fsm.FullKeyboardInputs(term)
    assert inputs.controller.command == (0.,) * 6
    inputs.update({'q'}, .1, focused=True, paused=False)
    assert float(term.keyboard_command[0, 5]) == pytest.approx(.05)
    assert term.keyboard_command[0, 3:5].tolist() == [0., 0.]
    inputs.update({'r'}, .1, focused=True, paused=False)
    assert term.keyboard_command[0].tolist() == [0.] * 6
    assert not bool(term.keyboard_active[0])


def test_quad_teacher_idle_observation_requests_four_foot_support(ready_chain):
    env, _, action, term, *_ = ready_chain
    from src.tasks.loco_pedipulation.mdp.commands import QUAD
    from src.tasks.transition_t.keyboard import teacher_observations
    env.common_step_counter += 1
    term._update_command()
    assert int(term.phase[0]) == QUAD
    assert float(term.blend[0]) == 0.
    assert not bool(term.enabled[0])
    obs = teacher_observations(env, action.source_teacher.contract, torch.tensor([0]))
    assert obs.shape == (1, 54)
    assert float(obs[0, -1]) == 0.
    assert term.support_weights.tolist() == [[1.] * 4]


def test_quad_teacher_r_restores_support_gradually(ready_chain, monkeypatch):
    env, _, action, term, *_ = ready_chain
    from src.tasks.loco_pedipulation.mdp.commands import (
        LocoPedipulationCommand, HOLD, QUAD, RETURN)
    from src.tasks.transition_t.keyboard import teacher_observations
    # Exercise the real command FSM with confirmed contact; GPU playback
    # separately verifies that the policy physically puts FR on the floor.
    monkeypatch.setattr(LocoPedipulationCommand, 'contacts', property(
        lambda self: torch.ones(self.num_envs, 4, dtype=torch.bool, device=self.device)))
    inputs = fsm.FullKeyboardInputs(term)
    def tick(keys=()):
        inputs.update(set(keys), env.step_dt, focused=True, paused=False)
        env.common_step_counter += 1
        term._update_command()
    tick({'q'})
    for _ in range(60):
        tick()
    assert int(term.owner[0]) == fsm.QUADRUPED
    assert int(term.phase[0]) == HOLD
    assert float(term.blend[0]) == 1.
    assert bool(term.enabled[0])
    # A second target in HOLD must restart the source teacher's reach.
    tick({'q'})
    assert int(term.phase[0]) != HOLD
    for _ in range(40):
        tick()
    tick({'r'})
    assert int(term.phase[0]) == RETURN
    assert 0. < float(term.blend[0]) <= 1.
    blends = [float(term.blend[0])]
    for _ in range(60):
        tick()
        blends.append(float(term.blend[0]))
    assert all(a + 1e-6 >= b for a, b in zip(blends, blends[1:])), blends
    assert int(term.phase[0]) == QUAD
    assert float(term.blend[0]) == 0.
    assert not bool(term.enabled[0])
    assert term.command.tolist() == [[0.] * 6]
    obs = teacher_observations(env, action.source_teacher.contract, torch.tensor([0]))
    assert float(obs[0, -1]) == 0.


def test_keyboard_foot_tracking_does_not_reset_playback(ready_chain):
    env, _, _, _, *_ = ready_chain
    # Inspect the real playback environment, including inherited terminations.
    assert set(env.cfg.terminations) == {'physical_failure', 'anchor_pos', 'anchor_ori'}
    from src.tasks.transition_down_v2.env_cfg import landing_env_cfg
    assert 'ee_body_pos' in landing_env_cfg().terminations


def test_transition_input_cannot_overwrite_reference_or_pending_target(ready_chain):
    _, _, _, term, *_ = ready_chain
    inputs = fsm.FullKeyboardInputs(term)
    term.submit([0], [[0., 0., 0., .02, .01, .35]], [True])
    inputs.sync()
    term.owner[:] = fsm.RISE
    reference_command = torch.tensor([[0., 0., 0., .10, .03, .42]])
    term._command.copy_(reference_command)
    pending = term.keyboard_command.clone()
    inputs.update({'w', 'e', 'r'}, .1, focused=True, paused=False)
    torch.testing.assert_close(term.command, reference_command)
    torch.testing.assert_close(term.keyboard_command, pending)


def test_biped_handoff_target_uses_shared_zero_and_next_key_is_continuous(ready_chain):
    _, _, _, term, *_ = ready_chain
    from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
    term._select_reference(term.rise_reference)
    motion = term.motion_term
    motion.time_steps[:] = term.rise_reference.num_frames - 1
    fr = motion.cfg.body_names.index('FR_foot')
    offset = to_anchor(anchor_basis(motion.anchor_quat_w, term.robot.data.gravity_vec_w),
        motion.body_pos_w[:, fr] - motion.anchor_pos_w) - term.zero
    finish_rise(ready_chain)
    torch.testing.assert_close(term.keyboard_command[:, 3:], offset)
    torch.testing.assert_close(term.command[:, 3:], offset)
    assert term.keyboard_active.all()
    inputs = fsm.FullKeyboardInputs(term)
    inputs.update({'q'}, .02, focused=True, paused=False)
    expected = offset.clone()
    expected[:, 2] += .005
    torch.testing.assert_close(term.keyboard_command[:, 3:], expected)


def test_reset_after_rise_restores_down_reference_and_quad_target(ready_chain):
    env, wrapped, _, term, *_ = ready_chain
    term._select_reference(term.rise_reference)
    term.motion_term.time_steps[:] = 100
    wrapped.reset()
    assert term.motion_term.reference is term.down_reference
    assert term.motion_term.time_steps.tolist() == [0]
    assert term.keyboard_command.tolist() == [[0.] * 6]
    assert term.command.tolist() == [[0.] * 6]
    assert int(term.owner[0]) == fsm.QUADRUPED
    assert torch.isfinite(env.observation_manager.compute()['actor']).all()


def test_partial_reset_does_not_replace_another_worlds_active_rise():
    env = None
    try:
        env, wrapped, _, term, *_ = fsm.load_full_chain(
            fsm.parse_args(['--device', 'cpu', '--num-envs', '2']))
        term._ensure_motion_references()
        term._select_reference(term.rise_reference)
        motion = term.motion_term
        motion.time_steps[0] = 100
        motion.active[0] = True
        term.owner[0] = fsm.RISE
        pose = env.sim.data.qpos[0].clone()
        env._reset_idx(torch.tensor([1]))
        assert motion.reference is term.rise_reference
        assert int(motion.time_steps[0]) == 100
        assert bool(motion.active[0])
        assert int(term.owner[0]) == fsm.RISE
        torch.testing.assert_close(env.sim.data.qpos[0].clone(), pose, atol=0, rtol=0)
    finally:
        if env is not None:
            env.close()


def test_down_handoff_preserves_endpoint_target_for_next_key(ready_chain):
    env, _, _, term, *_ = ready_chain
    from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
    motion = term.motion_term
    motion.time_steps[:] = term.down_reference.num_frames - 1
    motion.completed[:] = True
    motion.active[:] = True
    term.owner[:] = fsm.DOWN
    fr = motion.cfg.body_names.index('FR_foot')
    offset = to_anchor(anchor_basis(motion.anchor_quat_w, term.robot.data.gravity_vec_w),
        motion.body_pos_w[:, fr] - motion.anchor_pos_w) - term.zero
    env.common_step_counter += 1
    term._update_command()
    assert int(term.owner[0]) == fsm.QUADRUPED
    torch.testing.assert_close(term.keyboard_command[:, 3:], offset)
    inputs = fsm.FullKeyboardInputs(term)
    inputs.update({'q'}, .02, focused=True, paused=False)
    expected = offset.clone()
    expected[:, 2] += .005
    torch.testing.assert_close(term.keyboard_command[:, 3:], expected)


def test_biped_start_first_key_and_r_request_return_without_target_jump():
    env = None
    try:
        env, _, action, term, *_ = fsm.load_full_chain(
            fsm.parse_args(['--device', 'cpu', '--initial-state', 'biped']))
        for teacher in (action.source_teacher, action.biped_teacher):
            torch.testing.assert_close(term.zero, torch.tensor(teacher.contract['foot_zero']))
        inputs = fsm.FullKeyboardInputs(term)
        nominal = term.nominal_biped_offset.clone()
        torch.testing.assert_close(term.keyboard_command[0, 3:], nominal)
        inputs.update({'q'}, .02, focused=True, paused=False)
        expected = nominal.clone()
        expected[2] += .005
        torch.testing.assert_close(term.keyboard_command[0, 3:], expected)
        inputs.update({'r'}, .02, focused=True, paused=False)
        torch.testing.assert_close(term.keyboard_command[0, 3:], expected)
        assert bool(term.return_to_quadruped[0])
        assert int(term.owner[0]) == fsm.BIPED
        # A short press remains pending after key release, until the next
        # command-manager update can start DOWN with matching observations.
        inputs.update(set(), .02, focused=True, paused=False)
        env.common_step_counter += 1
        term._update_command()
        assert int(term.owner[0]) == fsm.DOWN
        assert term.motion_term.reference is term.down_reference
        assert bool(term.motion_term.active[0])
    finally:
        if env is not None:
            env.close()


def test_r_after_rise_returns_to_quad_default_without_physical_reset(ready_chain):
    env, _, action, term, *_ = ready_chain
    finish_rise(ready_chain)
    inputs = fsm.FullKeyboardInputs(term)
    before = [v.clone() for v in (env.sim.data.qpos, env.sim.data.qvel,
        action.raw_action, action.previous_action,
        action.target_limiter.position, action.target_limiter.velocity)]
    target = term.keyboard_command[:, 3:].clone()
    inputs.update({'r'}, .02, focused=True, paused=False)
    torch.testing.assert_close(term.keyboard_command[:, 3:], target)
    env.common_step_counter += 1
    term._update_command()
    assert int(term.owner[0]) == fsm.DOWN
    for expected, value in zip(before, (env.sim.data.qpos, env.sim.data.qvel,
        action.raw_action, action.previous_action,
        action.target_limiter.position, action.target_limiter.velocity)):
        torch.testing.assert_close(value.clone(), expected, atol=0, rtol=0)
    motion = term.motion_term
    motion.time_steps[:] = term.down_reference.num_frames - 1
    motion.completed[:] = True
    env.common_step_counter += 1
    term._update_command()
    assert int(term.owner[0]) == fsm.QUADRUPED
    assert term.keyboard_command.tolist() == [[0.] * 6]
    assert term.command.tolist() == [[0.] * 6]
    assert not bool(term.keyboard_active[0])
    assert not bool(term.return_to_quadruped[0])


@pytest.mark.parametrize('focused, paused', [(False, False), (True, True)])
def test_r_cannot_request_down_when_input_is_inactive(ready_chain, focused, paused):
    _, _, _, term, *_ = ready_chain
    finish_rise(ready_chain)
    inputs = fsm.FullKeyboardInputs(term)
    inputs.update({'r'}, .02, focused=focused, paused=paused)
    assert not bool(term.return_to_quadruped[0])
    assert int(term.owner[0]) == fsm.BIPED


def test_environment_reset_clears_pending_r_return(ready_chain):
    _, wrapped, _, term, *_ = ready_chain
    finish_rise(ready_chain)
    fsm.FullKeyboardInputs(term).update({'r'}, .02, focused=True, paused=False)
    assert bool(term.return_to_quadruped[0])
    wrapped.reset()
    assert not bool(term.return_to_quadruped[0])
