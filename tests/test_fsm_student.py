"""Recurrent student contracts, temporal learning, and independent-world resets."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch


def student():
    assert importlib.util.find_spec('src.tasks.fsm_student') is not None, 'GRU student is missing'
    return importlib.import_module('src.tasks.fsm_student')


def small_model():
    module = student()
    torch.manual_seed(7)
    return module.GRUStudent(module.GRUStudentConfig(
        encoder_dims=(16,), hidden_size=16, head_dims=(16,)))


def test_defaults_use_current_observation_and_recurrent_memory():
    module = student()
    model = module.GRUStudent()
    assert model.cfg.hidden_size == 128
    assert model.cfg.num_layers == 1
    assert module.STUDENT_OBSERVATION_DIM == 48
    action, hidden = model(torch.zeros(3, 48))
    assert action.shape == (3, 12)
    assert hidden.shape == (1, 3, 128)
    assert torch.isfinite(action).all()


def test_default_action_reference_matches_full_chain_quadruped_pose():
    module = student()
    assert module.GRUStudentConfig().default_angles == (
        -.1, .9, -1.8, .1, .9, -1.8, -.1, .9, -1.8, .1, .9, -1.8)


def test_sequence_and_step_inference_are_equivalent():
    model = small_model()
    observations = torch.randn(3, 6, 48)
    expected, expected_hidden = model(observations)
    hidden, steps = None, []
    for observation in observations.unbind(1):
        action, hidden = model(observation, hidden)
        steps.append(action)
    torch.testing.assert_close(torch.stack(steps, 1), expected)
    torch.testing.assert_close(hidden, expected_hidden)


def test_equal_current_observations_can_produce_different_actions_from_history():
    model = small_model()
    observations = torch.zeros(2, 4, 48)
    observations[0, :3, 0] = 1.
    observations[1, :3, 0] = -1.
    actions, _ = model(observations)
    assert not torch.allclose(actions[0, -1], actions[1, -1], atol=1e-6, rtol=0.)


def test_sequence_episode_boundaries_clear_only_the_selected_world():
    model = small_model()
    observations = torch.randn(2, 5, 48)
    starts = torch.zeros(2, 5, dtype=torch.bool)
    starts[0, 2] = True
    actual, actual_hidden = model(observations, episode_starts=starts)
    suffix, suffix_hidden = model(observations[0:1, 2:])
    other, other_hidden = model(observations[1:2])
    torch.testing.assert_close(actual[0:1, 2:], suffix)
    torch.testing.assert_close(actual_hidden[:, 0:1], suffix_hidden)
    torch.testing.assert_close(actual[1:2], other)
    torch.testing.assert_close(actual_hidden[:, 1:2], other_hidden)


def test_episode_start_ignores_hidden_state_from_previous_episode():
    model = small_model()
    observation = torch.randn(2, 48)
    initial = torch.randn(1, 2, 16)
    before = initial.clone()
    actual, hidden = model(observation, initial, episode_starts=torch.ones(2, dtype=torch.bool))
    expected, expected_hidden = model(observation)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(initial, before, atol=0., rtol=0.)


def test_policy_batched_reset_and_reset_mask_match_fresh_inference():
    module = student()
    model = small_model()
    policy = module.GRUStudentPolicy(model, num_envs=2)
    observation = torch.randn(2, 48)
    expected, expected_hidden = model(observation)
    torch.testing.assert_close(policy(observation), expected)
    torch.testing.assert_close(policy.hidden, expected_hidden)
    other = policy.hidden[:, 1].clone()
    policy.reset(torch.tensor([0]))
    assert not policy.hidden[:, 0].any()
    torch.testing.assert_close(policy.hidden[:, 1], other)
    expected, _ = model(observation[0:1])
    actual = policy(observation)
    torch.testing.assert_close(actual[0:1], expected)
    expected, _ = model(observation)
    torch.testing.assert_close(policy(observation, reset_mask=torch.ones(2, dtype=torch.bool)), expected)
    policy.reset()
    assert not policy.hidden.any()
    assert not policy.hidden.requires_grad


def test_sequence_bc_can_learn_history_dependent_actions():
    # The final input is identical. Learning opposite final labels requires memory.
    model = small_model()
    observations = torch.zeros(2, 4, 48)
    observations[0, :3, 0] = 1.
    observations[1, :3, 0] = -1.
    labels = torch.tensor([.4, -.4])[:, None].expand(-1, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    with torch.no_grad():
        initial_loss = (model(observations)[0][:, -1]-labels).square().mean().item()
    for _ in range(80):
        optimizer.zero_grad()
        actions, _ = model(observations)
        loss = (actions[:, -1]-labels).square().mean()
        loss.backward()
        assert model.gru.weight_hh_l0.grad.abs().sum() > 0
        optimizer.step()
    with torch.no_grad():
        final_loss = (model(observations)[0][:, -1]-labels).square().mean().item()
    assert final_loss < .002
    assert final_loss < initial_loss*.02


def test_hidden_state_can_be_detached_between_training_chunks():
    model = small_model()
    first = torch.randn(2, 3, 48, requires_grad=True)
    second = torch.randn(2, 3, 48, requires_grad=True)
    _, hidden = model(first)
    actions, _ = model(second, hidden.detach())
    actions.square().mean().backward()
    assert first.grad is None
    assert second.grad.abs().sum() > 0


def observation_env():
    def forbidden(*args):
        raise AssertionError('Student must not read teacher commands or FSM state')
    ids = torch.arange(11, -1, -1)
    data = SimpleNamespace(root_link_ang_vel_b=torch.ones(2, 3)*4.,
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).expand(2, -1),
        joint_pos=torch.arange(24, dtype=torch.float32).reshape(2, 12),
        joint_vel=torch.arange(24, dtype=torch.float32).reshape(2, 12)*2.)
    action = SimpleNamespace(joint_ids=ids, default_angles=torch.ones(2, 12),
        raw_action=torch.full((2, 12), .4))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)},
        action_manager=SimpleNamespace(get_term=lambda name: action),
        command_manager=SimpleNamespace(get_term=forbidden, get_command=forbidden))
    return env, data, action


def test_observations_use_explicit_user_intent_and_common_joint_order():
    module = student()
    env, robot, action = observation_env()
    command = torch.tensor([[.3, .1, .4, .05, .02, .1], [-.2, 0., -.4, 0., 0., .5]])
    actual = module.student_observations(env, command)
    assert actual.shape == (2, 48)
    torch.testing.assert_close(actual[:, :3], torch.ones(2, 3))
    torch.testing.assert_close(actual[:, 3:6], robot.projected_gravity_b)
    torch.testing.assert_close(actual[:, 6:12], command*torch.tensor([2., 2., .25, 1., 1., 1.]))
    torch.testing.assert_close(actual[:, 12:24], robot.joint_pos[:, action.joint_ids]-1.)
    torch.testing.assert_close(actual[:, 24:36], robot.joint_vel[:, action.joint_ids]*.05)
    torch.testing.assert_close(actual[:, 36:48], action.raw_action)


def test_checkpoint_round_trip_preserves_sequence_actions(tmp_path):
    module = student()
    model = small_model()
    path = tmp_path/'student.pt'
    model.save(path)
    restored = module.GRUStudent.load(path)
    observations = torch.randn(2, 4, 48)
    expected, expected_hidden = model(observations)
    actual, hidden = restored(observations)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    torch.testing.assert_close(hidden, expected_hidden, atol=0., rtol=0.)
    assert restored.cfg == model.cfg


@pytest.mark.parametrize('field,value', [('version', 99), ('observation_contract', {}),
    ('action_contract', {})])
def test_checkpoint_rejects_incompatible_contract(tmp_path, field, value):
    module = student()
    path = tmp_path/'student.pt'
    small_model().save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        module.GRUStudent.load(path)


def test_checkpoint_rejects_legacy_r_request_observation_contract(tmp_path):
    module = student()
    path = tmp_path/'legacy_student.pt'
    small_model().save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['version'] = 1
    contract = checkpoint['observation_contract']
    contract['dimension'] = 49
    contract['fields'] = [field for field in contract['fields'] if field[0] != 'user_return_requested']
    contract['fields'].insert(3, ['user_return_requested', 1, 1.])
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        module.GRUStudent.load(path)


@pytest.mark.parametrize('shape', [(2, 49), (2, 3, 50), (2, 0, 48), (48,)])
def test_model_rejects_invalid_observation_shapes(shape):
    with pytest.raises(ValueError):
        small_model()(torch.zeros(shape))


def test_model_rejects_wrong_hidden_or_episode_mask():
    model = small_model()
    observations = torch.zeros(2, 3, 48)
    with pytest.raises(ValueError):
        model(observations, torch.zeros(1, 3, 16))
    with pytest.raises(ValueError):
        model(observations, episode_starts=torch.zeros(2, 3))
    with pytest.raises(ValueError):
        model(observations, episode_starts=torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize('kwargs', [{'hidden_size': 0}, {'num_layers': 0},
    {'encoder_dims': (0,)}, {'head_dims': (-1,)}])
def test_config_rejects_invalid_network_sizes(kwargs):
    module = student()
    with pytest.raises(ValueError):
        module.GRUStudent(module.GRUStudentConfig(**kwargs))
