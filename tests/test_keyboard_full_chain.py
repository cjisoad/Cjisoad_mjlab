"""Contracts for the four-owner keyboard teacher chain."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_full_chain_cli_exposes_initial_owner_and_model_defaults():
    import keyboard_transition_fsm as fsm

    args = fsm.parse_args([])
    assert args.initial_state == "quadruped"
    assert args.rise_trigger_height == pytest.approx(0.35)
    assert args.down_trigger_height == pytest.approx(0.25)
    assert args.workspace == "union"
    assert args.quad_checkpoint_file.name == "quad_teacher.pt"
    assert args.biped_checkpoint_file.name == "biped_teacher.pt"
    assert args.rise_checkpoint_file == fsm.ROOT/"mark/up_teacher.pt"
    assert args.down_checkpoint_file == fsm.ROOT/"mark/down_teacher.pt"
    assert args.rise_motion_file.name == "tripod_to_biped_v3_speed125_train.npz"


def test_full_chain_cli_accepts_separate_rise_reference():
    import keyboard_transition_fsm as fsm

    args = fsm.parse_args(['--initial-state', 'biped', '--rise-checkpoint-file', 'model_9999.pt',
        '--rise-motion-file', 'tripod_to_biped_v3_speed125_train.npz'])
    assert args.rise_checkpoint_file.name == 'model_9999.pt'
    assert args.rise_motion_file.name == 'tripod_to_biped_v3_speed125_train.npz'
    assert args.motion_file.name == 'biped_to_tripod_v2.npz'


def test_full_chain_thresholds_are_directional_and_have_a_hold_boundary():
    import keyboard_transition_fsm as fsm

    assert fsm.should_start_rise(0.35, 0.35, 0.10, 0.10)
    assert not fsm.should_start_rise(0.3499, 0.35, 0.0, 0.10)
    assert fsm.should_start_down(0.25, 0.25, 0.10, 0.10)
    assert not fsm.should_start_down(0.2501, 0.25, 0.10)
    assert fsm.should_start_down(0.25, 0.25, 0.1001)


def test_full_chain_state_order_is_explicit():
    import keyboard_transition_fsm as fsm

    assert fsm.STATE_ORDER == (fsm.QUADRUPED, fsm.RISE, fsm.BIPED, fsm.DOWN)
    assert fsm.TEACHER_STATES == (fsm.QUADRUPED, fsm.BIPED)


def test_full_chain_constructs_all_actors_and_a_physical_world():
    import keyboard_transition_fsm as fsm

    args = fsm.parse_args(["--device", "cpu"])
    env = None
    try:
        env, wrapped, action, term, rise, down, paths = fsm.load_full_chain(args)
        assert env.num_envs == 1
        assert int(term.owner[0]) == fsm.QUADRUPED
        robot = env.scene["robot"]
        torch.testing.assert_close(robot.data.joint_pos, robot.data.default_joint_pos, atol=1e-6, rtol=0)
        assert action.source_teacher is not None
        assert action.biped_teacher is not None
        assert callable(rise) and callable(down)
        assert paths["rise"] == fsm.ROOT/"mark/up_teacher.pt"
        assert paths["down"] == fsm.ROOT/"mark/down_teacher.pt"
        obs = env.observation_manager.compute()
        step_obs, _, done, _ = wrapped.step(fsm.FullChainPolicy(term, rise, down)(obs))
        assert step_obs["actor"].shape[-1] == 75
        assert not bool(done[0])
    finally:
        if env is not None:
            env.close()


def test_full_chain_biped_entry_is_physical_and_deterministic():
    import keyboard_transition_fsm as fsm
    import torch

    args = fsm.parse_args(["--device", "cpu", "--initial-state", "biped"])
    env = None
    try:
        env, wrapped, action, term, rise, down, paths = fsm.load_full_chain(args)
        root = env.scene["robot"].data.root_link_pos_w[0]
        assert float(root[2]) > 0.4
        bank = torch.from_numpy(__import__("numpy").load(args.entry_bank)["joint_pos"][16508])
        torch.testing.assert_close(env.scene["robot"].data.joint_pos[0], bank, atol=1e-5, rtol=0)
        assert int(term.owner[0]) == fsm.BIPED
        assert term.status()["requested_dz"] > 0.25
    finally:
        if env is not None:
            env.close()
