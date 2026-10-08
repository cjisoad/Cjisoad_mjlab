"""Geometry, command transitions, reward masks and curriculum contracts."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz
from src.assets.robots.unitree_go2.go2_constants import get_spec
from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
from src.tasks.loco_pedipulation.mdp.anchor_frame import anchor_basis, to_anchor
from src.tasks.loco_pedipulation.mdp.commands import (
  HOLD, LOWER, PREPARE, QUAD, REACH, RECOVER,
  LocoPedipulationCommand, LocoPedipulationCommandCfg,
)
from src.tasks.loco_pedipulation.mdp.curriculums import manipulation_curriculum
from src.tasks.loco_pedipulation.mdp.foot_workspace import FOOT_NAMES, FR_JOINTS, build_foot_workspace
from src.tasks.loco_pedipulation.mdp.observations import foot_control_state, policy_state
from src.tasks.loco_pedipulation.mdp import rewards
from src.tasks.loco_pedipulation.mdp.metrics import ManipulationMetrics
from src.tasks.loco_pedipulation.rl.metrics import ManipulationLogger


def make_env(count=4):
  workspace = build_foot_workspace()
  root = torch.tensor((0., 0., workspace.nominal_height), dtype=torch.float32).repeat(count, 1)
  sites = root[:, None, :] + torch.tensor(workspace.zero, dtype=torch.float32).expand(count, 4, 3)
  data = SimpleNamespace(
    root_link_pos_w=root, root_link_quat_w=torch.tensor((1., 0., 0., 0.)).repeat(count, 1),
    gravity_vec_w=torch.tensor((0., 0., -1.)).repeat(count, 1), site_pos_w=sites.clone(),
    root_link_lin_vel_w=torch.zeros(count, 3), root_link_ang_vel_w=torch.zeros(count, 3),
    root_link_ang_vel_b=torch.zeros(count, 3), projected_gravity_b=torch.tensor((0., 0., -1.)).repeat(count, 1),
    joint_pos=torch.zeros(count, 12), joint_vel=torch.zeros(count, 12), default_joint_pos=torch.zeros(count, 12),
    site_lin_vel_w=torch.zeros(count, 4, 3))
  robot = SimpleNamespace(data=data,
    find_sites=lambda names, **kw: (list(range(4)), list(FOOT_NAMES)),
    find_joints=lambda names, **kw: ([3, 4, 5], list(FR_JOINTS)),
    find_geoms=lambda names: (list(range(4)), [f'{name}_foot_collision' for name in FOOT_NAMES]))
  force = torch.zeros(count, 4, 3)
  force[:, :, 2] = 30.
  env = SimpleNamespace(num_envs=count, device='cpu', step_dt=.02, common_step_counter=0,
    scene={'robot': robot, 'feet_ground_contact': SimpleNamespace(data=SimpleNamespace(force=force))},
    termination_manager=SimpleNamespace(terminated=torch.zeros(count, dtype=torch.bool)))
  cfg = LocoPedipulationCommandCfg(initial_stage=3, resampling_time_range=(100., 100.))
  term = LocoPedipulationCommand(cfg, env)
  action = SimpleNamespace(joint_pos=data.joint_pos, joint_vel=data.joint_vel,
    target_ids=torch.arange(12), default_angles=data.default_joint_pos, raw_action=torch.zeros(count, 12), actor_sample=None)
  env.action_manager = SimpleNamespace(get_term=lambda name: action)
  env.command_manager = SimpleNamespace(get_term=lambda name: term, get_command=lambda name: term.command)
  term.reset(torch.arange(count))
  term.set_command(torch.arange(count), velocity=(0., 0., 0.), target_offset=(0., 0., 0.))
  term.initialize_reference()
  return env, term


def advance(env, term, steps):
  for _ in range(steps):
    env.common_step_counter += 1
    term.compute(env.step_dt)


class LocoPedipulationTests(unittest.TestCase):
  def test_command_is_six_dimensional_and_target_zero_controls_activation(self):
    env, term = make_env()
    self.assertEqual(term.command.shape, (4, 6))
    target = (.013, -.011, .063)
    term.set_command([0], velocity=(.2, -.1, .3), target_offset=target)
    advance(env, term, 25)
    torch.testing.assert_close(term.command[0, :3], torch.tensor((.2, -.1, .3)))
    # Manual targets stay continuous instead of being snapped to an FK-bank point.
    torch.testing.assert_close(term.command[0, 3:], torch.tensor(target))
    torch.testing.assert_close(term.requested_offset[0], torch.tensor(target))
    self.assertTrue(term.enabled[0])
    self.assertTrue((term.command[0, 3:] != 0).any())
    term.set_command([0], target_offset=(0., 0., 0.))
    self.assertFalse(term.enabled[0])
    torch.testing.assert_close(term.command[0, 3:], torch.zeros(3))

  def test_actor_observation_matches_pedipulation_48_value_abi(self):
    env, term = make_env()
    robot, action = env.scene['robot'].data, env.action_manager.get_term('joint_pos')
    robot.root_link_ang_vel_b[:] = torch.tensor((1., 2., 3.))
    robot.projected_gravity_b[:] = torch.tensor((.1, .2, -.9))
    action.joint_pos[:] = torch.arange(12.)
    action.joint_vel[:] = torch.arange(12.) + 1
    action.raw_action[:] = torch.arange(12.) - 2
    term.set_command(torch.arange(4), velocity=(.2, -.1, .4), target_offset=(.03, 0., .07))
    actual = policy_state(env, add_noise=False)
    expected = torch.cat((robot.root_link_ang_vel_b * .25, robot.projected_gravity_b,
      term.command * torch.tensor((2., 2., .25, 1., 1., 1.)),
      action.joint_pos - action.default_angles, action.joint_vel * .05, action.raw_action), dim=-1)
    self.assertEqual(actual.shape, (4, 48))
    torch.testing.assert_close(actual, expected)

  def test_policy_ignores_internal_state_and_shares_noise_with_critic(self):
    from src.tasks.loco_pedipulation.mdp.observations import shared_policy_state
    env, term = make_env()
    first = policy_state(env, add_noise=False).clone()
    term.phase[:] = HOLD
    term.blend[:] = 1.
    term.reference[:] += .3
    term.gait_phase[:] = .5
    torch.testing.assert_close(policy_state(env, add_noise=False), first)
    noisy = policy_state(env)
    self.assertIs(shared_policy_state(env), noisy)
    limits = torch.tensor((.05,) * 6 + (0.,) * 6 + (.01,) * 12 + (.075,) * 12 + (0.,) * 12)
    self.assertTrue(((noisy - first).abs() <= limits + 1e-6).all())
    cfg = loco_pedipulation_env_cfg()
    self.assertEqual(tuple(cfg.observations['actor'].terms), ('policy',))
    from src.tasks.loco_pedipulation.rl import loco_pedipulation_ppo_runner_cfg
    self.assertFalse(loco_pedipulation_ppo_runner_cfg().actor.obs_normalization)

  def test_default_training_budget_is_13000_iterations(self):
    from src.tasks.loco_pedipulation.rl import loco_pedipulation_ppo_runner_cfg
    self.assertEqual(loco_pedipulation_ppo_runner_cfg().max_iterations, 13000)

  def test_target_does_not_count_as_motion_in_rewards_or_metrics(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., .1))
    self.assertEqual(float(rewards.feet_gait(env)[0]), 0.)
    self.assertEqual(float(rewards.support_contact(env)[0]), 1.)
    term.phase[0] = HOLD
    term.record_outcomes()
    self.assertEqual(float(term.training_metrics.tracking[0, 0]), 1.)
    self.assertEqual(float(term.training_metrics.tracking[1, 0]), 0.)


  def test_anchor_follows_heading_and_stays_horizontal_under_tilt(self):
    roll, pitch, yaw = (torch.tensor(values) for values in
      ((0., .3, -.4), (0., .5, -.2), (0., 1., -2.)))
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    basis = anchor_basis(quat, torch.tensor((0., 0., -1.)))
    expected_x = torch.stack((yaw.cos(), yaw.sin(), torch.zeros(3)), -1)
    torch.testing.assert_close(basis[:, :, 0], expected_x)
    torch.testing.assert_close(basis[:, :, 2], torch.tensor((0., 0., 1.)).expand(3, -1))
    torch.testing.assert_close(basis.transpose(1, 2) @ basis, torch.eye(3).expand(3, -1, -1))
    torch.testing.assert_close(torch.linalg.det(basis), torch.ones(3))
    torch.testing.assert_close(to_anchor(basis, basis[:, :, 0]), torch.tensor((1., 0., 0.)).expand(3, -1))

  def test_workspace_has_named_fk_witnesses_and_lift_clearance(self):
    bank = build_foot_workspace()
    self.assertGreater(len(bank.offsets), 50)
    model = get_spec().compile()
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in FR_JOINTS]
    for offset, joints in zip(bank.offsets, bank.joint_positions):
      data.qpos[addresses] = joints
      mujoco.mj_forward(model, data)
      actual = data.site_xpos[model.site('FR').id] - data.xpos[model.body('base_link').id]
      np.testing.assert_allclose(actual, bank.zero + offset, atol=1e-7)
    self.assertTrue((bank.offsets[:, 2] >= .05).all())
    self.assertTrue((bank.offsets[:, 2] <= .12).all())
    with self.assertRaises(ValueError):
      build_foot_workspace(lift_range=(.12, .05))

  def test_full_transition_requires_landing_contact_and_is_continuous(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., .07))
    previous = term.reference[0].clone()
    for _ in range(60):
      advance(env, term, 1)
      self.assertLess((term.reference[0] - previous).norm().item(), .015)
      previous = term.reference[0].clone()
    self.assertEqual(term.phase[0], HOLD)
    self.assertEqual(term.blend[0], 1.)
    torch.testing.assert_close(term.reference[0], term.zero + term.requested_offset[0])
    self.assertTrue((term.phase[1:] == QUAD).all())
    term.set_command([0], target_offset=(0., 0., 0.))
    env.scene['feet_ground_contact'].data.force[0, 1] = 0.
    advance(env, term, 45)
    self.assertEqual(term.phase[0], LOWER)
    self.assertFalse(rewards.landing_failed(env)[0])
    env.scene['feet_ground_contact'].data.force[0, 1, 2] = 30.
    advance(env, term, 5)
    self.assertEqual(term.phase[0], RECOVER)
    advance(env, term, 20)
    self.assertEqual(term.phase[0], QUAD)
    self.assertEqual(term.blend[0], 0.)
    self.assertEqual(term.stats[0, 8], 1.)
    self.assertEqual(term.stats[0, 9], 1.)

  def test_cancel_retarget_and_reenable_during_transition(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., .07))
    advance(env, term, 1)
    self.assertEqual(term.phase[0], PREPARE)
    term.set_command([0], target_offset=(0., 0., 0.))
    advance(env, term, 20)
    self.assertEqual(term.phase[0], QUAD)
    term.set_command([0], target_offset=(0., 0., .07))
    advance(env, term, 60)
    before = term.reference[0].clone()
    term.set_command([0], target_offset=(.08, -.05, .12))
    advance(env, term, 1)
    self.assertEqual(term.phase[0], REACH)
    torch.testing.assert_close(term.reference[0], before)
    term.set_command([0], target_offset=(0., 0., 0.))
    advance(env, term, 5)
    self.assertEqual(term.phase[0], LOWER)
    before = term.reference[0].clone()
    term.set_command([0], target_offset=(0., 0., .07))
    advance(env, term, 1)
    self.assertEqual(term.phase[0], REACH)
    torch.testing.assert_close(term.reference[0], before)
    advance(env, term, 40)
    self.assertEqual(term.phase[0], HOLD)

  def test_manual_target_updates_during_reach_keep_one_continuous_transition(self):
    env, term = make_env()
    term.set_command([0], target_offset=(.013, -.011, .063))
    advance(env, term, 20)
    self.assertEqual(term.phase[0], REACH)
    elapsed = term.elapsed[0].clone()
    term.set_command([0], target_offset=(.014, -.010, .064))
    torch.testing.assert_close(term.goal[0], term.zero + torch.tensor((.014, -.010, .064)))
    self.assertFalse(bool(term._target_changed[0]))
    advance(env, term, 1)
    self.assertGreater(term.elapsed[0], elapsed)

  def test_failed_landing_times_out_without_releasing_support(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., .07))
    advance(env, term, 60)
    term.set_command([0], target_offset=(0., 0., 0.))
    env.scene['feet_ground_contact'].data.force[0, 1] = 0.
    advance(env, term, 150)
    self.assertTrue(rewards.landing_failed(env)[0])
    self.assertEqual(term.phase[0], LOWER)
    self.assertEqual(term.blend[0], 1.)

  def test_partial_reset_defers_kinematics_and_preserves_other_environments(self):
    env, term = make_env()
    term.set_command([0, 1], target_offset=(0., 0., .07))
    advance(env, term, 25)
    before = (term.reference[1].clone(), term.phase[1].clone(), term.elapsed[1].clone())
    term.reset(torch.tensor([0]))
    env.scene['robot'].data.site_pos_w[0, 1, 0] += .025
    obs = foot_control_state(env)
    torch.testing.assert_close(term.reference[0], term.foot_pos_anchor[0])
    for actual, expected in zip((term.reference[1], term.phase[1], term.elapsed[1]), before):
      torch.testing.assert_close(actual, expected)
    self.assertEqual(obs.shape, (4, 18))
    self.assertTrue(torch.isfinite(obs).all())
    self.assertEqual(term.phase[0], QUAD)

  def test_sampling_is_independent_and_manual_control_survives_resampling(self):
    env, term = make_env(1024)
    term.manual[:] = False
    term.cfg = replace(term.cfg, standing_probability=.5, tripod_standing_probability=.5)
    term._resample_command(torch.arange(1024))
    stopped = term.requested_velocity.norm(dim=-1) == 0
    for active in (False, True):
      for standing in (False, True):
        self.assertTrue(((term.enabled == active) & (stopped == standing)).any())
    term.set_command([0], velocity=(.12, 0., 0.), target_offset=(0., 0., .07))
    requested = term.requested_offset[0].clone()
    term._resample_command(torch.tensor([0]))
    self.assertTrue(term.enabled[0])
    torch.testing.assert_close(term.requested_offset[0], requested)
    torch.testing.assert_close(term.requested_velocity[0], torch.tensor((.12, 0., 0.)))
    term.release_manual([0])
    self.assertFalse(term.manual[0])

  def test_tripod_speed_limits_and_stationary_curriculum(self):
    env, term = make_env()
    term.set_command([0, 1], velocity=(.8, .4, .8), target_offset=(0., 0., .07))
    advance(env, term, 60)
    torch.testing.assert_close(term.command[0, :3], torch.tensor((.35, .15, .4)))
    term.stage = 1
    advance(env, term, 30)
    torch.testing.assert_close(term.command[:2, :3], torch.zeros(2, 3))

  def test_rewards_exclude_fr_only_in_manipulation_and_gate_contact(self):
    env, term = make_env()
    term.phase[:] = torch.tensor((QUAD, REACH, HOLD, LOWER))
    term.blend[:] = torch.tensor((0., 1., 1., 1.))
    torch.testing.assert_close(rewards.fr_ground_contact(env), torch.tensor((0., 0., 1., 0.)))
    term.reference[:] = term.foot_pos_anchor
    torch.testing.assert_close(rewards.fr_position_tracking(env), term.blend)
    robot = env.scene['robot'].data
    robot.site_lin_vel_w[:, 1, 0] = 1.
    torch.testing.assert_close(rewards.foot_slip(env), torch.tensor((1., 0., 0., 0.)))
    robot.site_lin_vel_w[:, 0, 0] = 1.
    torch.testing.assert_close(rewards.foot_slip(env), torch.tensor((2., 1., 1., 1.)))
    robot.joint_pos[:, 3:6] = 1.
    torch.testing.assert_close(rewards.stand_still(env), torch.tensor((3., 0., 0., 0.)))

  def test_tripod_gait_is_independent_of_fr_contact(self):
    env, term = make_env()
    term.blend[:] = 1.
    term._command[:, 0] = .1
    term.gait_phase[:] = torch.tensor((0., .25, .5, .75))
    first = rewards.feet_gait(env)
    env.scene['feet_ground_contact'].data.force[:, 1] = 0.
    torch.testing.assert_close(rewards.feet_gait(env), first)

  def test_moving_tripod_reward_distinguishes_stalling_from_tracking(self):
    env, term = make_env()
    term.blend[:] = torch.tensor((0., 1., 1., 1.))
    term._command[:, 0] = .2
    env.scene['robot'].data.root_link_lin_vel_w[:, 0] = torch.tensor((0., 0., .1, .2))
    actual = rewards.track_linear_velocity(env)
    torch.testing.assert_close(actual, torch.tensor((.8521438, .3380266, 1.2823608, 2.)))
    self.assertGreater(float(actual[3] - actual[1]), 1.6)
    # Stationary tripod and pure yaw commands keep the original linear reward.
    term._command[:, 0] = 0.
    term._command[1, 2] = .2
    robot = env.scene['robot'].data
    robot.root_link_lin_vel_w[:, 2] = .1
    expected = torch.exp(-(robot.root_link_lin_vel_w[:, :2].square().sum(-1)
                           + 2. * robot.root_link_lin_vel_w[:, 2].square()) / .5**2)
    torch.testing.assert_close(rewards.track_linear_velocity(env), expected)

  def test_moving_reward_blends_and_keeps_vertical_damping(self):
    env, term = make_env()
    term._command[:, 0] = .2
    term.blend[:] = torch.tensor((0., .5, 1., 1.))
    base = rewards.track_linear_velocity(env)
    torch.testing.assert_close(base[1], (base[0] + base[2]) / 2.)
    env.scene['robot'].data.root_link_lin_vel_w[:, 2] = .1
    torch.testing.assert_close(rewards.track_linear_velocity(env) / base,
                               torch.full((4,), np.exp(-2. * .1**2 / .5**2)))

  def test_landing_logging_survives_resets_and_counts_interruptions(self):
    env, term = make_env()
    term.set_command([0, 1, 2], target_offset=(0., 0., .07))
    advance(env, term, 60)
    term.set_command([0, 1, 2], target_offset=(0., 0., 0.))
    env.scene['feet_ground_contact'].data.force[1:3, 1] = 0.
    advance(env, term, 40)
    term.set_command([1], target_offset=(0., 0., .07))
    advance(env, term, 1)
    advance(env, term, 100)
    self.assertTrue(rewards.landing_failed(env)[2])
    term.reset(torch.tensor([0, 1, 2]))
    self.assertNotIn('landing_success', term.reset(torch.tensor([3])))
    torch.testing.assert_close(term.training_metrics.landing,
                               torch.tensor((3., 1., 1., 1.), dtype=torch.float64))
    self.assertEqual(float(term.stats[:, 8:].sum()), 0.)

  def test_logger_weights_samples_and_does_not_insert_missing_ratios(self):
    recorded = {}
    writer = SimpleNamespace(add_scalar=lambda key, value, step: recorded.__setitem__((key, step), value))
    base = SimpleNamespace(writer=writer, log=lambda **kw: None)
    metrics = ManipulationMetrics('cpu')
    logger = ManipulationLogger(base, metrics)
    command = torch.tensor(((0., 0., 0.), (.2, 0., 0.), (.2, 0., 0.)))
    velocity = torch.tensor(((0., 0., 0.), (0., 0., 0.), (.2, 0., 0.)))
    hold = torch.ones(3, dtype=torch.bool)
    angular = torch.zeros(3, 3)
    metrics.record_tracking(hold, command, velocity, angular)
    metrics.record_tracking(torch.tensor((False, True, False)), command, velocity, angular)
    metrics.landing_event('attempts', torch.ones(9, dtype=torch.bool))
    metrics.landing_event('confirmed', torch.tensor((True,)))
    logger.log(it=0)
    prefix = 'Metrics/twist/'
    self.assertAlmostEqual(recorded[(prefix + 'tripod_moving_velocity_error', 0)], .4 / 3, places=6)
    self.assertEqual(recorded[(prefix + 'tripod_moving_samples', 0)], 3)
    self.assertEqual(recorded[(prefix + 'tripod_stationary_samples', 0)], 1)
    self.assertAlmostEqual(recorded[(prefix + 'tripod_moving_actual_vx', 0)], .2 / 3, places=6)
    metrics.landing_event('attempts', torch.tensor((True,)))
    metrics.landing_event('confirmed', torch.tensor((True,)))
    logger.log(it=1)
    self.assertEqual(recorded[(prefix + 'landing_attempts_total', 1)], 10)
    self.assertEqual(recorded[(prefix + 'landing_success', 1)], .2)
    self.assertEqual(recorded[(prefix + 'landing_pending', 1)], 8)
    self.assertNotIn((prefix + 'tripod_moving_velocity_error', 1), recorded)
    restored = ManipulationLogger(base, ManipulationMetrics('cpu'))
    restored.restore_counts(logger.landing_totals)
    restored.log(it=2)
    self.assertEqual(recorded[(prefix + 'landing_interrupted_total', 2)], 8)
    self.assertEqual(recorded[(prefix + 'landing_pending', 2)], 0)

  def test_tracking_is_recorded_once_and_ignores_non_hold_phases(self):
    env, term = make_env()
    term.phase[:] = torch.tensor((HOLD, HOLD, QUAD, LOWER))
    term._command[1:, 0] = .2
    term.record_outcomes()
    term.record_outcomes()
    torch.testing.assert_close(term.training_metrics.tracking[:, 0],
                               torch.ones(2, dtype=torch.float64))

  def test_curriculum_needs_success_not_elapsed_time(self):
    env, term = make_env()
    term.cfg = replace(term.cfg, min_stage_steps=10, min_curriculum_episodes=4)
    term.stage = 0
    env.common_step_counter = 100
    term.stats[:, 0] = 100.
    term.stats[:, 1] = 100.
    term.stats[:, 6] = 100.
    manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(term.stage, 0)
    term.stats[:, 1] = 10.
    manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(term.stage, 1)
    env.common_step_counter += 20
    term.stats[:, 2] = 100.
    term.stats[:, 3] = 3.
    term.stats[:, 4] = 0.
    term.stats[:, 8] = 3.
    term.stats[:, 9] = 3.
    manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(term.stage, 2)

  def test_configuration_and_registration_are_independent(self):
    from mjlab.tasks.registry import list_tasks
    from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
    self.assertIn('loco_pedipulation', list_tasks())
    cfg = loco_pedipulation_env_cfg()
    baseline = unitree_go2_flat_env_cfg()
    self.assertNotIn('fr_position_tracking', baseline.rewards)
    self.assertIn('fr_position_tracking', cfg.rewards)
    self.assertEqual(cfg.commands['twist'].initial_stage, 0)
    self.assertEqual(loco_pedipulation_env_cfg(play=True).commands['twist'].initial_stage, 3)


if __name__ == '__main__':
  unittest.main()
