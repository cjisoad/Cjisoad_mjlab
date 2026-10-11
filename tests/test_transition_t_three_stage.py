"""Full-start ability and actual disturbed entries must drive separate gates."""
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch


def evidence(successes=52, attempts=64, source='nominal', distinct=64):
    return dict(successes=successes, attempts=attempts, initialization=source,
                distinct_trajectories=distinct, full_reference_from_start=True)


def test_scratch_has_no_checkpoint_dependency_and_rejects_ambiguous_flags():
    from scripts.train_transition_entry import parse_args
    a = parse_args(['--entry-bank', 'bank.npz', '--from-scratch', '--curriculum', 'three-stage'])
    assert a.checkpoint is None and not a.resume
    for extra in [['--resume'], ['--checkpoint', 'old.pt'], ['--allow-reference-migration']]:
        with pytest.raises(SystemExit):
            parse_args(['--entry-bank', 'bank.npz', '--from-scratch', '--curriculum', 'three-stage', *extra])
    with pytest.raises(SystemExit):
        parse_args(['--entry-bank', 'bank.npz', '--curriculum', 'three-stage'])


def test_three_course_mixes_and_nominal_promotion_require_full_start_evidence():
    from src.tasks.transition_t.three_stage import STAGES, ThreeStageCurriculum
    assert len(STAGES) == 3
    assert [(s['nominal_probability'], s['entry_probability']) for s in STAGES] == [(.5, 0.), (.4, .4), (.2, .6)]
    c = ThreeStageCurriculum()
    assert not c.observe(evidence(51)) and c.streak == 0
    assert not c.observe(evidence(64, source='reference_fragment'))
    assert not c.observe(evidence(52, attempts=63))
    assert not c.observe(evidence(52))
    assert c.observe(evidence(52)) and c.level == 1


def test_real_entry_gate_requires_independent_trajectories_and_nominal_retention():
    from src.tasks.transition_t.three_stage import ThreeStageCurriculum
    c = ThreeStageCurriculum(level=1)
    assert not c.observe(evidence(48, source='entries'), nominal=evidence(51))
    assert not c.observe(evidence(48, source='entries', distinct=63), nominal=evidence(52))
    assert not c.observe(evidence(47, source='entries'), nominal=evidence(52))
    assert not c.observe(evidence(48, source='entries'), nominal=evidence(52))
    assert c.observe(evidence(48, source='entries'), nominal=evidence(52))
    assert c.level == 2
    assert not c.observe(evidence(64, source='entries'), nominal=evidence(64))
    assert not c.accepted
    assert not c.observe(evidence(102, 128, 'entries', 128), nominal=evidence(52))
    assert not c.accepted
    c.observe(evidence(103, 128, 'entries', 128), nominal=evidence(52))
    assert c.accepted
    saved = c.state_dict(); restored = ThreeStageCurriculum(); restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    c.observe(evidence(102, 128, 'entries', 128), nominal=evidence(52))
    assert not c.accepted  # Acceptance describes this policy, not a past one.


def test_stage_cfg_changes_actual_startup_physics_noise_and_keeps_limiter():
    from src.tasks.transition_t.three_stage import three_stage_env_cfg
    cfgs = [three_stage_env_cfg('bank.npz', stage=s, num_envs=8) for s in range(3)]
    assert 'base_com' not in cfgs[0].events and 'foot_friction' not in cfgs[0].events
    assert cfgs[1].events['base_com'].params['ranges'][0] == (-.005, .005)
    assert cfgs[2].events['base_com'].params['ranges'][0] == (-.01, .01)
    assert cfgs[1].events['foot_friction'].params['ranges'] == (.8, 1.1)
    assert cfgs[2].events['foot_friction'].params['ranges'] == (.6, 1.2)
    assert cfgs[0].commands['motion'].joint_position_range == (-.01, .01)
    assert cfgs[0].actions['joint_pos'].target_velocity_limits[3:6] == (2.,)*3
    assert cfgs[2].rewards['knee_ground'].weight == -.2
    nominal = three_stage_env_cfg('bank.npz', stage=2, nominal_evaluation=True)
    assert nominal.events == {} and not nominal.observations['actor'].enable_corruption
    assert nominal.commands['motion'].stage == 2


def test_knee_cost_is_bounded_substep_force_and_excludes_feet():
    from src.tasks.transition_t.three_stage_rewards import knee_ground_cost
    history = torch.zeros(2, 4, 4, 3)
    history[0, 0, 0, 2] = 50.
    history[1, 1, :, 2] = 500.
    env = SimpleNamespace(scene={'knees_ground_contact': SimpleNamespace(data=SimpleNamespace(force_history=history))})
    torch.testing.assert_close(knee_ground_cost(env), torch.tensor([.25, 1.]))


def test_leg_velocity_matching_relaxes_after_completion_but_root_does_not():
    from src.tasks.transition_t.three_stage_rewards import terminal_body_velocity_reward
    names = ('base_link', 'RL_foot')
    cmd = SimpleNamespace(cfg=SimpleNamespace(body_names=names), time_steps=torch.tensor([0, 9]),
                          motion=SimpleNamespace(time_step_total=10),
                          body_lin_vel_w=torch.zeros(2, 2, 3), robot_body_lin_vel_w=torch.zeros(2, 2, 3))
    cmd.robot_body_lin_vel_w[:, 1, 0] = 1.
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: cmd))
    values = terminal_body_velocity_reward(env, 'motion', 1.)
    assert values[1] > values[0]
    cmd.robot_body_lin_vel_w.zero_(); cmd.robot_body_lin_vel_w[:, 0, 0] = 1.
    values = terminal_body_velocity_reward(env, 'motion', 1.)
    torch.testing.assert_close(values[0], values[1])


def test_scratch_cli_three_stage_options_are_explicit():
    from scripts.evaluate_transition_entry import parse_args
    args = parse_args(['--entry-bank','bank.npz','--checkpoint','model.pt',
      '--curriculum','three-stage','--initialization','nominal'])
    assert args.curriculum == 'three-stage' and args.initialization == 'nominal'


def test_resume_rejects_changed_ppo_before_loading_weights(tmp_path):
    from src.tasks.transition_t.three_stage_rl import ThreeStageOnPolicyRunner
    from src.tasks.transition_t.three_stage import ThreeStageCurriculum
    from src.tasks.transition_t.rl import transition_ppo_runner_cfg
    runner = object.__new__(ThreeStageOnPolicyRunner)
    runner.cfg = asdict(transition_ppo_runner_cfg())
    runner.term = SimpleNamespace(cfg=SimpleNamespace(stage=0), course=ThreeStageCurriculum())
    runner._verify_bank_physics = lambda: None
    with patch('src.tasks.transition_t.entry_rl.EntryOnPolicyRunner._entry_recipe', return_value={}):
        recipe = runner._entry_recipe()
        checkpoint = tmp_path/'model.pt'
        torch.save(dict(infos=dict(transition_entry_training=dict(recipe=recipe,
          course=runner.term.course.state_dict(), updates_completed=2,seed_checkpoint_sha256=''))),checkpoint)
        original = deepcopy(runner.cfg)
        for key in ('gamma','num_learning_epochs','num_mini_batches','entropy_coef','desired_kl'):
            runner.cfg = deepcopy(original)
            runner.cfg['algorithm'][key] *= 2
            with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load') as load:
                with pytest.raises(ValueError, match='recipe'):
                    runner.resume(checkpoint)
                load.assert_not_called()
        runner.cfg = deepcopy(original)
        runner.cfg['num_steps_per_env'] += 1
        with pytest.raises(ValueError, match='recipe'):
            runner.resume(checkpoint)


def test_stage_restore_rejects_skips_or_mismatched_course(tmp_path):
    from src.tasks.transition_t.three_stage_rl import ThreeStageOnPolicyRunner
    runner = object.__new__(ThreeStageOnPolicyRunner)
    runner.term = SimpleNamespace(cfg=SimpleNamespace(stage=2))
    runner._verify_bank_physics = lambda: None
    runner._entry_recipe = lambda: dict(stage=2, training_augmentation='full')
    checkpoint = tmp_path/'bad.pt'
    for old_stage, course in ((0,2),(1,1)):
        torch.save(dict(infos=dict(transition_entry_training=dict(recipe=dict(stage=old_stage),
          course=dict(level=course)))),checkpoint)
        with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load') as load:
            with pytest.raises(ValueError, match='immediately promoted'):
                runner.restore_stage_transition(checkpoint)
            load.assert_not_called()
