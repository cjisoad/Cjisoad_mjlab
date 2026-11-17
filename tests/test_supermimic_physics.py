"""Physical contracts for short differentiable Go2 rollouts."""
import math

import mujoco
import numpy as np
import pytest
import torch

from src.tasks.pedipulation.constants import DEFAULT_ANGLES, JOINT_NAMES
from src.tasks.pedipulation.robot import get_stand_spec
from src.tasks.legmanip_distil.supermimic import physics


def state(count=2, dtype=torch.float32):
    value = torch.zeros(count, 37, dtype=dtype)
    value[:, 2] = .42
    value[:, 3] = 1.
    value[:, 13:25] = torch.tensor(DEFAULT_ANGLES, dtype=dtype)
    return value


def test_observation_contract_and_world_velocity_exclusion():
    s = state()
    s[:, 7:10] = torch.tensor((100., 200., 300.))
    s[:, 10:13] = torch.tensor((1., 2., 3.))
    s[:, 25:] = 2.
    cmd = torch.ones(2, 6)
    action = torch.full((2, 12), .4)
    obs = physics.state_observation(s, cmd, action, torch.tensor(DEFAULT_ANGLES))
    assert obs.shape == (2, 48)
    torch.testing.assert_close(obs[:, :3], s[:, 10:13] * .25)
    torch.testing.assert_close(obs[:, 3:6], torch.tensor([[0., 0., -1.]]).expand(2, 3))
    torch.testing.assert_close(obs[:, 6:12], torch.tensor([[2., 2., .25, 1., 1., 1.]]).expand(2, 6))
    torch.testing.assert_close(obs[:, 12:24], torch.zeros(2, 12))
    torch.testing.assert_close(obs[:, 24:36], torch.full((2, 12), .1))
    torch.testing.assert_close(obs[:, 36:], action)


@pytest.mark.parametrize('pitch', [0., -math.pi / 2, -.72])
def test_fk_matches_actual_compiled_body_frames(pitch):
    model = get_stand_spec().compile()
    fk = physics.Go2Kinematics(model).double()
    s = state(1, torch.float64)
    s[:, :3] = torch.tensor((1.2, -.3, .5), dtype=torch.float64)
    s[:, 3:7] = torch.tensor((math.cos(pitch / 2), 0., math.sin(pitch / 2), 0.))
    s[:, 13:25] += torch.linspace(-.1, .1, 12)
    data = mujoco.MjData(model)
    data.qpos[:7] = s[0, :7].numpy()
    for i, name in enumerate(JOINT_NAMES):
        data.qpos[model.joint(name).qposadr[0]] = s[0, 13 + i]
    mujoco.mj_forward(model, data)
    out = fk(s)
    for i, name in enumerate(fk.body_names):
        np.testing.assert_allclose(out['body_pos_w'][0, i].detach().numpy(), data.xpos[model.body(name).id], atol=1e-8)
    feet = np.stack([data.xpos[model.body(f'{leg}_foot').id] for leg in ('FL', 'FR', 'RL', 'RR')])
    np.testing.assert_allclose(out['feet_pos_w'][0].detach().numpy(), feet, atol=1e-8)
    s.requires_grad_()
    fk(s)['feet_pos_w'].square().sum().backward()
    assert torch.isfinite(s.grad).all() and s.grad[:, 13:25].abs().sum() > 0


def test_world_model_uses_all_four_physical_targets_and_preserves_quaternion():
    torch.manual_seed(4)
    model = physics.WorldModel(hidden_dim=32, depth=2)
    s = state()
    targets = s[:, None, 13:25].repeat(1, 4, 1).requires_grad_()
    out = model(s, targets)
    assert out['state'].shape == (2, 37)
    assert out['contact_logits'].shape == (2, 4)
    assert out['collision_logits'].shape == (2, 1)
    torch.testing.assert_close(out['state'][:, 3:7].norm(dim=-1), torch.ones(2))
    out['state'].square().sum().backward()
    assert torch.isfinite(targets.grad).all()
    assert (targets.grad.abs().sum((0, 2)) > 0).all()
    with pytest.raises(ValueError, match='4'):
        model(s, targets[:, 0])


def test_manifold_delta_is_sign_invariant_and_roundtrips():
    s = state()
    n = s.clone()
    angle = .3
    n[:, 3:7] = torch.tensor((math.cos(angle/2), 0., math.sin(angle/2), 0.))
    n[:, :3] += .01
    delta = physics.state_delta(s, n)
    torch.testing.assert_close(physics.integrate_state(s, delta), n, atol=1e-6, rtol=1e-6)
    n[:, 3:7] *= -1
    torch.testing.assert_close(physics.state_delta(s, n), delta)


def test_world_model_loss_has_separate_contact_collision_and_state_gradients():
    model = physics.WorldModel(hidden_dim=32, depth=1)
    s = state()
    targets = s[:, None, 13:25].repeat(1, 4, 1)
    loss = model.loss(s, targets, s, torch.ones(2, 4), torch.zeros(2, 1))
    assert {'total', 'state', 'contact', 'collision'} <= loss.keys()
    loss['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_operation_tracking_is_exactly_gated_and_upright_posture_target_is_continuous():
    fk = physics.Go2Kinematics()
    reward = physics.DifferentiableTaskReward(fk)
    s = state()
    action = torch.zeros(2, 12)
    cmd = torch.zeros(2, 6)
    context = {'op_weight': torch.tensor([0., 1.]), 'desired_upright': torch.zeros(2)}
    terms = reward(s, s, action, action, cmd, context)
    assert terms['fr_tracking'][0] == 0 and terms['fr_tracking'][1] > 0
    cmd[:, 3:] = 1.
    changed = reward(s, s, action, action, cmd, context)
    torch.testing.assert_close(terms['fr_tracking'][0], changed['fr_tracking'][0])
    assert changed['fr_tracking'][1] < terms['fr_tracking'][1]
    upright = s.clone()
    upright[:, 3:7] = torch.tensor((math.sqrt(.5), 0., -math.sqrt(.5), 0.))
    context['desired_upright'][:] = 1.
    vertical = reward(s, upright, action, action, cmd, context)
    horizontal = reward(s, s, action, action, cmd, context)
    assert (vertical['posture'] > horizontal['posture']).all()
    context['desired_upright'][:] = .5
    half = s.clone()
    half[:, 3:7] = torch.tensor((math.cos(math.pi/8), 0., -math.sin(math.pi/8), 0.))
    halfway = reward(s, half, action, action, cmd, context)
    assert (halfway['posture'] > .99).all()


def test_reward_has_finite_action_gradient_through_model_and_fk():
    torch.manual_seed(3)
    model = physics.WorldModel(hidden_dim=32, depth=2)
    reward = physics.DifferentiableTaskReward(physics.Go2Kinematics())
    s = state()
    action = torch.full((2, 12), .1, requires_grad=True)
    targets = (s[:, 13:25] + .25 * action)[:, None].expand(-1, 4, -1)
    out = model(s, targets)
    terms = reward(s, out['state'], action, torch.zeros_like(action), torch.zeros(2, 6),
                   {'op_weight': torch.ones(2), 'desired_upright': torch.zeros(2)},
                   out['contact_logits'].sigmoid(), out['collision_logits'].sigmoid())
    terms['total'].sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0


def test_capture_state_reads_generalized_terminal_coordinates_not_stale_links():
    from types import SimpleNamespace
    qpos = torch.zeros(2, 25)
    qvel = torch.zeros(2, 24)
    qpos[:, 2:9] = state()[:, :7]
    qvel[:, 1:7] = torch.tensor((1., 2., 3., 4., 5., 6.))
    qpos[:, 9:21] = torch.arange(12.)
    qvel[:, 7:19] = torch.arange(12.) + .5
    joint_ids = torch.arange(11, -1, -1)
    data = SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel),
                           joint_pos=qpos[:, 9:21], joint_vel=qvel[:, 7:19],
                           root_link_pos_w=torch.full((2, 3), -999.))
    robot = SimpleNamespace(data=data, indexing=SimpleNamespace(
        free_joint_q_adr=torch.arange(2, 9), free_joint_v_adr=torch.arange(1, 7)))
    env = SimpleNamespace(scene={'robot': robot}, action_manager=SimpleNamespace(
        get_term=lambda _: SimpleNamespace(joint_ids=joint_ids)))
    result = physics.state_from_env(env)
    torch.testing.assert_close(result[:, :7], qpos[:, 2:9])
    torch.testing.assert_close(result[:, 7:13], qvel[:, 1:7])
    torch.testing.assert_close(result[:, 13:25], data.joint_pos[:, joint_ids])
    torch.testing.assert_close(result[:, 25:], data.joint_vel[:, joint_ids])
    qpos[:] = -10.
    assert (result[:, 2] > 0).all(), 'snapshot must own its storage'


def test_world_loss_masks_padding_for_every_head():
    torch.manual_seed(7)
    model = physics.WorldModel(hidden_dim=24, depth=1)
    s = state()
    targets = s[:, None, 13:25].repeat(1, 4, 1)
    contacts = torch.ones(2, 4)
    collision = torch.zeros(2, 1)
    expected = model.loss(s[:1], targets[:1], s[:1], contacts[:1], collision[:1])
    mask = torch.tensor([True, False])
    masked = model.loss(s, targets, s, contacts, collision, valid_mask=mask)
    for key in expected:
        torch.testing.assert_close(expected[key], masked[key])
    empty = model.loss(s, targets, s, contacts, collision, valid_mask=torch.zeros(2, dtype=torch.bool))
    assert all(v == 0 for v in empty.values())

def test_tracking_score_penalizes_stationary_low_speed_and_has_far_target_gradient():
    from src.tasks.legmanip_distil.supermimic.physics import tracking_score
    error=torch.tensor([[0.], [.1], [.3]],requires_grad=True)
    score=tracking_score(error,.1)
    assert score[0] > score[1]+.5
    assert score[1] > score[2]
    score[2].backward()
    assert error.grad[2].abs().item()>.5
