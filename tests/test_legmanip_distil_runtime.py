"""Exercise the real student environment and frozen teacher action contract."""
import torch
import pytest


def test_observation_contract_has_only_deployable_inputs():
    from src.tasks.legmanip_distil.observations import observation_contract
    contract = observation_contract()
    assert contract['dimension'] == 48
    assert sum(field[1] for field in contract['fields']) == 48
    assert 'owner' not in str(contract['fields'])
    assert contract['return_command'] == [0.] * 6


@pytest.fixture(scope='module')
def student_world():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.legmanip_distil.env_cfg import legmanip_distil_env_cfg
    cfg = legmanip_distil_env_cfg(play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device='cpu')
    env.reset()
    yield env
    env.close()


def test_student_environment_needs_no_teacher_or_motion(student_world):
    env = student_world
    assert set(env.cfg.commands) == {'intent'}
    assert set(env.cfg.observations) == {'student'}
    assert not hasattr(env, 'distillation_teacher')
    obs, _ = env.reset()
    assert obs['student'].shape == (2, 48)
    intent = env.command_manager.get_term('intent')
    commands = torch.tensor([[.1, 0., 0., .08, 0., .12], [0., 0., 0., 0., 0., 0.]])
    intent.set_command(torch.arange(2), commands)
    action = env.action_manager.get_term('joint_pos')
    requested = torch.full((2, 12), .2)
    obs, _, _, _, _ = env.step(requested)
    torch.testing.assert_close(action.raw_action, requested)
    torch.testing.assert_close(env.scene['robot'].data.joint_pos_target[:, action.joint_ids],
                               action.default_angles + .25 * requested)
    torch.testing.assert_close(obs['student'][:, 36:], requested)
    torch.testing.assert_close(intent.command, commands)
    intent.return_to_quadruped(torch.tensor([0]))
    assert torch.count_nonzero(intent.command[0]) == 0


def test_partial_reset_does_not_reset_other_user_command(student_world):
    env = student_world
    env.reset()
    intent = env.command_manager.get_term('intent')
    command = torch.tensor([[.1, 0., 0., .08, 0., .12]])
    intent.set_command(torch.tensor([1]), command)
    before = env.sim.data.qpos[1].clone()
    env._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(intent.command[1], command[0])
    torch.testing.assert_close(env.sim.data.qpos[1], before)


def test_nonfinite_actions_rejected_before_pd_write(student_world):
    action = student_world.action_manager.get_term('joint_pos')
    before = action.raw_action.clone()
    with pytest.raises(ValueError, match='finite'):
        action.process_actions(torch.full((2, 12), float('nan')))
    torch.testing.assert_close(action.raw_action, before)


def test_physical_failure_checks_full_robot_state(student_world):
    from src.tasks.legmanip_distil.env_cfg import physical_failure
    env = student_world
    env.reset()
    robot = env.scene['robot'].data
    original = robot.root_link_ang_vel_w[0, 0].clone()
    robot.root_link_ang_vel_w[0, 0] = float('nan')
    try:
        assert bool(physical_failure(env)[0])
    finally:
        robot.root_link_ang_vel_w[0, 0] = original


@pytest.fixture(scope='module')
def teacher_world():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.legmanip_distil.env_cfg import legmanip_distil_env_cfg
    from src.tasks.legmanip_distil.teacher import FSMTeacher
    cfg = legmanip_distil_env_cfg()
    cfg.scene.num_envs = 4
    cfg.commands['intent'].scenario = 'quad'
    env = ManagerBasedRlEnv(cfg, device='cpu')
    env.reset()
    teacher = FSMTeacher(env)
    yield env, teacher
    env.close()


def test_rise_down_references_and_partial_resets_are_independent(teacher_world):
    env, _ = teacher_world
    env.reset()
    term = env.command_manager.get_term('twist')
    term.start_transition(torch.tensor([0]), 2)
    term.start_transition(torch.tensor([1]), 4)
    rise, down = term.references[2], term.references[4]
    env.common_step_counter += 1
    term._update_command()
    assert rise.time_steps[0] == 1
    assert down.time_steps[1] == 1
    assert rise.time_steps[1] == 0
    assert down.time_steps[0] == 0
    before = (rise.time_steps[0].clone(), rise.reference_rotation[0].clone(),
              env.sim.data.qpos[0].clone())
    env.reset(env_ids=torch.tensor([1]))
    torch.testing.assert_close(rise.time_steps[0], before[0])
    torch.testing.assert_close(rise.reference_rotation[0], before[1])
    torch.testing.assert_close(env.sim.data.qpos[0], before[2])


def test_teacher_query_is_read_only_and_labels_equal_final_targets(teacher_world):
    env, teacher = teacher_world
    env.reset()
    action = env.action_manager.get_term('joint_pos')
    term = env.command_manager.get_term('twist')
    term.owner[:] = torch.tensor([0, 2, 3, 4])
    term.phase[:] = 3
    term.blend[:] = 1.
    before = [value.clone() for value in (env.sim.data.qpos, env.sim.data.qvel,
              action.raw_action, action.executed_position, action.executed_velocity)]
    rng = torch.get_rng_state().clone()
    labels = teacher(None)
    assert labels.shape == (4, 12) and torch.isfinite(labels).all()
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for value, expected in zip((env.sim.data.qpos, env.sim.data.qvel, action.raw_action,
                               action.executed_position, action.executed_velocity), before):
        torch.testing.assert_close(value.clone(), expected)
    teacher.set_control_mask(torch.ones(4, dtype=torch.bool))
    action.process_actions(labels)
    for _ in range(env.cfg.decimation):
        action.apply_actions()
    torch.testing.assert_close(action.raw_action, labels, atol=1e-6, rtol=0)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_teacher_internal_reference_does_not_change_user_intent(teacher_world):
    env, _ = teacher_world
    env.reset()
    intent = env.command_manager.get_term('intent')
    command = torch.tensor([[0., 0., 0., .1, 0., .36]])
    intent.set_command(torch.tensor([0]), command)
    term = env.command_manager.get_term('twist')
    term.start_transition(torch.tensor([0]), 2)
    env.common_step_counter += 1
    term._update_command()
    torch.testing.assert_close(intent.command[0], command[0])
    assert not torch.equal(term.command[0], command[0])


def test_zero_command_starts_biped_return_without_hold_delay(teacher_world):
    env, _ = teacher_world
    env.reset()
    term = env.command_manager.get_term('twist')
    term.owner[0] = 3
    intent = env.command_manager.get_term('intent')
    intent.return_to_quadruped([0])
    env.common_step_counter += 1
    term._update_command()
    assert term.owner[0] == 4
    assert term.returning[0]
    assert term.references[4].time_steps[0] == 0


def test_tripod_operation_stays_below_rise_trigger(teacher_world):
    from src.tasks.legmanip_distil.intent import STAGES, SCENARIOS
    env, _ = teacher_world
    intent = env.command_manager.get_term('intent')
    intent.scenario[:] = 0
    intent.stage[:] = [name for name, _ in SCENARIOS[0]].index('tripod_operation')
    intent.low_target[:] = intent.low_bank[intent.low_bank[:, 2].argmax()]
    for seconds in torch.linspace(0, 3, 31):
        intent.elapsed[:] = seconds
        intent._write_script_commands(torch.arange(env.num_envs))
        assert (intent.command[:, 5] < .35).all()
