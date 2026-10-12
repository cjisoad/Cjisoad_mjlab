"""The rise objective must improve before the existing 5 cm success gate."""
from types import SimpleNamespace

import torch


def make_env(errors, rise=.12):
    errors = torch.tensor(errors, dtype=torch.float32)
    target = torch.tensor([[0., 0., .30], [0., 0., .30 + rise]])
    cmd = SimpleNamespace(
        motion=SimpleNamespace(body_pos_w=target[:, None, :]),
        motion_anchor_body_index=0,
        time_steps=torch.ones(len(errors), dtype=torch.long),
        anchor_pos_w=target[1].expand(len(errors), -1).clone(),
        robot_anchor_pos_w=target[1].expand(len(errors), -1).clone(),
    )
    cmd.robot_anchor_pos_w[:, 2] -= errors
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd)), cmd


def test_height_improvement_before_success_gate_and_overshoot_symmetry():
    from src.tasks.transition_t import three_stage_rewards as rewards
    env, cmd = make_env([.084, .07, .06, 0., -.06])
    assert hasattr(rewards, 'rise_height_reward'), 'continuous height shaping is missing'
    values = rewards.rise_height_reward(env, 'motion')
    assert 0 < values[0] < values[1] < values[2] < values[3]
    torch.testing.assert_close(values[2], values[4])
    cmd.robot_anchor_pos_w[:, :2] += 5.
    torch.testing.assert_close(values, rewards.rise_height_reward(env, 'motion'))


def test_reference_height_smoothly_enables_reward_and_keeps_terminal_active():
    from src.tasks.transition_t import three_stage_rewards as rewards
    assert hasattr(rewards, 'rise_height_reward'), 'continuous height shaping is missing'
    values = []
    for rise in (0., .02, .04, .06, .12):
        env, _ = make_env([0.], rise=rise)
        values.append(rewards.rise_height_reward(env, 'motion').item())
    assert values[0] == 0. and values[1] < 1e-10
    assert abs(values[2] - .5) < 1e-5
    assert abs(values[3] - 1.) < 1e-5 and values[4] == 1.


def test_three_stage_registers_height_without_changing_other_objectives():
    from src.tasks.transition_t.three_stage import three_stage_env_cfg
    for stage in range(3):
        cfg = three_stage_env_cfg('bank.npz', stage=stage, num_envs=1)
        assert 'rise_height' in cfg.rewards, 'height reward must reach the actual environment'
        assert cfg.rewards['rise_height'].weight == .5
        assert cfg.rewards['knee_ground'].weight == -.2
        assert cfg.rewards['standing_hold'].weight == .1
        assert cfg.episode_length_s == 10.
