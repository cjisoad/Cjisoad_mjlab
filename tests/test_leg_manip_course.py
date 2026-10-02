"""Three-stage stance course regression tests using real command/reward terms."""
import math
from types import SimpleNamespace
import unittest
import torch
from tests.test_leg_manip_commands import make_env, advance
from tests.test_leg_manip_rewards import rich_env
from src.tasks.leg_manip.mdp import rewards
from src.tasks.leg_manip.mdp.curriculums import manipulation_curriculum
from src.tasks.pedipulation.constants import DESIRED_ANGLES

class CourseSamplingTests(unittest.TestCase):
  def test_stage_zero_samples_both_fixed_stances_and_no_velocity_or_operation(self):
    torch.manual_seed(4)
    env, term = make_env(count=128, stage=0)
    term.release_manual(torch.arange(128))
    term._resample_command(torch.arange(128))
    high = term.command[:, 5] > term.cfg.tripod_high
    self.assertGreater(int(high.sum()), 20)
    self.assertGreater(int((~high).sum()), 20)
    torch.testing.assert_close(term.requested_velocity, torch.zeros(128, 3))
    torch.testing.assert_close(term.command[~high, 3:], torch.zeros(int((~high).sum()), 3))
    torch.testing.assert_close(term.command[high, 3:], term.nominal_biped_offset.expand(int(high.sum()), 3))
    self.assertFalse(term.manipulating.any())
    torch.testing.assert_close(term.nominal_biped_offset, torch.tensor((.133488, .030602, .478042)), atol=2e-6, rtol=0)
    self.assertGreaterEqual(term.cfg.resampling_time_range[0], 6.)

  def test_stage_one_keeps_nominal_targets_and_samples_walking_in_both_modes(self):
    torch.manual_seed(7)
    env, term = make_env(count=128, stage=1)
    term.release_manual(torch.arange(128))
    term._resample_command(torch.arange(128))
    # Repeated stable-mode samples include walking; mode changes are stationary.
    walked = torch.zeros(2, dtype=torch.bool)
    for _ in range(10):
      advance(env, term, 1)
      term._resample_command(torch.arange(128))
      high = term.command[:, 5] > term.cfg.tripod_high
      moving = term.requested_velocity.norm(dim=-1) > .05
      walked[0] |= bool((moving & ~high).any())
      walked[1] |= bool((moving & high).any())
      torch.testing.assert_close(term.command[high, 3:], term.nominal_biped_offset.expand(int(high.sum()), 3))
      self.assertFalse(term.manipulating.any())
    self.assertTrue(walked.all())

  def test_only_stage_two_samples_random_operation_in_both_modes(self):
    env, term = make_env(count=128, stage=2)
    term.release_manual(torch.arange(128))
    term._resample_command(torch.arange(128))
    high = term.command[:, 5] > term.cfg.tripod_high
    self.assertTrue((term.manipulating & high).any())
    self.assertTrue((term.manipulating & ~high).any())
    self.assertTrue((~term.manipulating & high).any())
    self.assertTrue((~term.manipulating & ~high).any())

class CourseRewardTests(unittest.TestCase):
  def nominal_env(self):
    env, term = rich_env()
    term.manual[:] = False
    term._operation[:] = False
    term.enabled[:] = True
    term.requested_offset[:] = term.nominal_biped_offset
    term._command[:, 3:] = term.nominal_biped_offset
    term.group_weight[:] = 1.
    state = env.action_manager.get_term('joint_pos')
    state.height_score = torch.ones(4)
    state.joint_pos[:] = state.desired_angles
    return env, term

  def test_nominal_front_default_posture_remains_active(self):
    env, term = self.nominal_env()
    torch.testing.assert_close(rewards.default_pos_reward_FR(env), torch.ones(4))
    env.action_manager.get_term('joint_pos').joint_pos[:, 3] += .5
    self.assertTrue((rewards.default_pos_front(env) > 0).all())
    self.assertTrue((rewards.symmetric_joints(env) > 0).all())

  def test_static_biped_contact_requires_two_rear_feet_outside_height_gate(self):
    env, term = self.nominal_env()
    env.action_manager.get_term('joint_pos').height_score[:] = 0.
    env.scene['rear_contact'].data.force[:, :, 2] = 30.
    torch.testing.assert_close(rewards.biped_contact(env), torch.ones(4))
    env.scene['rear_contact'].data.force[:, 0] = 0.
    torch.testing.assert_close(rewards.biped_contact(env), torch.zeros(4))

  def test_zero_speed_never_rewards_swing_clearance(self):
    env, term = self.nominal_env()
    cfg = SimpleNamespace(name='robot', site_ids=[2, 3])
    torch.testing.assert_close(rewards.biped_feet_clearance(env, asset_cfg=cfg), torch.zeros(4))

  def test_nominal_fr_tracking_has_lower_weight_until_operation_stage(self):
    env, term = self.nominal_env()
    term.blend[:] = 1.
    for stage in (0, 1):
      term.stage = stage
      self.assertLess(float(rewards.fr_position_tracking(env)[0]), .5)
    term.stage = 2
    torch.testing.assert_close(rewards.fr_position_tracking(env), torch.ones(4))

  def test_observed_low_tilted_fl_supported_cheat_fails_biped_readiness(self):
    env, term = self.nominal_env()
    data = env.scene['robot'].data
    data.root_link_pos_w[:, 2] = .267
    data.projected_gravity_b[:] = torch.tensor((-math.cos(math.radians(37)), 0., -math.sin(math.radians(37))))
    env.scene['feet_ground_contact'].data.force[:] = 0.
    env.scene['feet_ground_contact'].data.force[:, (0, 2, 3), 2] = 30.
    term.reference[:] = term.foot_pos_anchor + torch.tensor((.01, 0., 0.))
    self.assertFalse(term.stance_ready(biped=True).any())
    data.root_link_pos_w[:, 2] = .52
    data.projected_gravity_b[:] = torch.tensor((-1., 0., 0.))
    data.joint_pos[:] = torch.tensor(DESIRED_ANGLES)
    env.scene['feet_ground_contact'].data.force[:, 0] = 0.
    self.assertTrue(term.stance_ready(biped=True).all())

class CourseCurriculumTests(unittest.TestCase):
  def qualified(self, stage=0):
    env, term = make_env(stage=stage, min_stage_steps=100, min_curriculum_episodes=4)
    env.common_step_counter = 100
    term.group_survival[:] = torch.tensor((20., 1.))
    term.curriculum_window[:] = torch.tensor((200., 190., 10.))
    term.switch_window[:] = torch.tensor((20., 19.))
    term.curriculum_episodes = 4
    return env, term

  def decide(self, env, term):
    return manipulation_curriculum(env, torch.empty(0, dtype=torch.long))

  def test_quad_success_cannot_hide_failed_biped_stance(self):
    env, term = self.qualified()
    term.curriculum_window[1, 1] = 0.
    self.decide(env, term)
    self.assertEqual(term.pass_streak, 0)
    self.assertEqual(term.stage, 0)

  def test_both_switch_directions_must_succeed(self):
    for row in (0, 1):
      env, term = self.qualified()
      term.switch_window[row, 1] = 0.
      self.decide(env, term)
      self.assertEqual(term.pass_streak, 0)

  def test_failed_decision_restarts_full_window_clock(self):
    env, term = self.qualified()
    term.switch_window.zero_()
    self.decide(env, term)
    self.assertEqual(term.stage_started, 100)
    term.group_survival[:] = torch.tensor((20., 1.))
    term.curriculum_window[:] = torch.tensor((200., 190., 10.))
    term.switch_window[:] = torch.tensor((20., 19.))
    term.curriculum_episodes = 4
    env.common_step_counter = 101
    self.decide(env, term)
    self.assertEqual(term.pass_streak, 0)
    self.assertEqual(term.curriculum_episodes, 4)

  def test_two_passing_full_windows_required(self):
    env, term = self.qualified()
    self.decide(env, term)
    self.assertEqual(term.stage, 0)
    self.assertEqual(term.pass_streak, 1)
    term.group_survival[:] = torch.tensor((20., 1.))
    term.curriculum_window[:] = torch.tensor((200., 190., 10.))
    term.switch_window[:] = torch.tensor((20., 19.))
    term.curriculum_episodes = 4
    env.common_step_counter = 200
    self.decide(env, term)
    self.assertEqual(term.stage, 1)
    self.assertEqual(term.pass_streak, 0)

  def test_stage_one_requires_both_moving_groups_and_all_velocity_levels(self):
    env, term = self.qualified(stage=1)
    term.curriculum_window[3, 1] = 0.
    self.decide(env, term)
    self.assertEqual(term.velocity_level, 0)
    self.assertEqual(term.pass_streak, 0)
    for level in range(4):
      for _ in range(2):
        env.common_step_counter += 100
        term.group_survival[:] = torch.tensor((20., 1.))
        term.curriculum_window[:] = torch.tensor((200., 190., 10.))
        term.switch_window[:] = torch.tensor((20., 19.))
        term.curriculum_episodes = 4
        self.decide(env, term)
      self.assertEqual(term.stage, 2 if level == 3 else 1)
      self.assertEqual(term.velocity_level, min(level + 1, 3))

class CourseSwitchTests(unittest.TestCase):
  def biped_env(self):
    env, term = make_env(stage=0)
    # Certify a physical quad source before replacing it with the held biped endpoint.
    term.set_command([0, 1], target_offset=term.nominal_biped_offset)
    advance(env, term, 1)
    data = env.scene['robot'].data
    data.root_link_pos_w[:, 2] = .52
    data.projected_gravity_b[:] = torch.tensor((-1., 0., 0.))
    data.joint_pos[:] = torch.tensor(DESIRED_ANGLES)
    env.scene['feet_ground_contact'].data.force[:, :2] = 0.
    return env, term

  def tick(self, env, term, n):
    for _ in range(n):
      advance(env, term, 1)
      term.record_outcomes()

  def test_endpoint_hold_and_readiness_required_not_phase_timer(self):
    env, term = self.biped_env()
    env.scene['robot'].data.root_link_pos_w[:, 2] = .267
    self.tick(env, term, 100)
    self.assertEqual(float(term.switch_window[0, 0]), 2.)
    self.assertEqual(float(term.switch_window[0, 1]), 0.)
    env.scene['robot'].data.root_link_pos_w[:, 2] = .52
    self.tick(env, term, 49)
    self.assertEqual(float(term.switch_window[0, 1]), 0.)
    self.tick(env, term, 2)
    self.assertEqual(float(term.switch_window[0, 1]), 2.)

  def test_subset_reset_interrupts_only_selected_pending_switch(self):
    env, term = self.biped_env()
    self.tick(env, term, 1)
    term.reset(torch.tensor([0]))
    term.set_command([0], target_offset=(0., 0., 0.))
    self.assertEqual(int(term.pending_switch[0]), -1)
    self.assertEqual(int(term.pending_switch[1]), 0)
    self.assertEqual(float(term.switch_window[0, 1]), 0.)
    self.tick(env, term, 105)
    self.assertEqual(float(term.switch_window[0, 1]), 1.)

  def test_terminal_pending_switch_is_failed_even_at_ready_endpoint(self):
    env, term = self.biped_env()
    self.tick(env, term, 98)
    env.termination_manager.terminated[0] = True
    self.tick(env, term, 3)
    self.assertEqual(int(term.pending_switch[0]), -1)
    self.assertEqual(float(term.switch_window[0, 1]), 1.)

  def test_timeout_and_reversal_fail_pending_attempt(self):
    env, term = self.biped_env()
    env.scene['robot'].data.root_link_pos_w[:, 2] = .267
    self.tick(env, term, 260)
    self.assertEqual(int(term.pending_switch[0]), -1)
    self.assertEqual(float(term.switch_window[0, 1]), 0.)
    term.set_command([0], target_offset=(0., 0., 0.))
    self.tick(env, term, 1)
    self.assertEqual(float(term.switch_window[1, 0]), 1.)
    term.set_command([0], target_offset=term.nominal_biped_offset)
    self.tick(env, term, 1)
    self.assertEqual(int(term.pending_switch[0]), -1)
    self.assertGreater(float(term.training_metrics.switches[0, 4]), 0.)
    self.assertEqual(float(term.switch_window[1, 1]), 0.)

class CourseCheckpointTests(unittest.TestCase):
  def test_versioned_state_roundtrip_and_old_checkpoint_rejection(self):
    from src.tasks.leg_manip.rl.checkpoint import course_state, restore_course_state, validate_course_state
    env, term = make_env(stage=1)
    term.stage_started = 234
    term.pass_streak = 1
    term.velocity_level = 2
    term.curriculum_window[:] = torch.tensor((100., 90., 12.))
    term.switch_window[:] = torch.tensor((11., 10.))
    term.curriculum_episodes = 30
    term.curriculum_failures = 2
    counts = torch.tensor((10., 7., 1., 1.), dtype=torch.float64)
    state = course_state(term, counts)
    other_env, other = make_env(stage=0)
    restore_course_state(other, state)
    self.assertEqual(other.stage, 1)
    self.assertEqual(other.stage_started, 234)
    self.assertEqual(other.pass_streak, 1)
    self.assertEqual(other.velocity_level, 2)
    torch.testing.assert_close(other.curriculum_window, term.curriculum_window)
    torch.testing.assert_close(other.switch_window, term.switch_window)
    self.assertEqual(other.curriculum_failures, 2)
    for legacy in (None, {'stage': 0}, {**state, 'version': -1}, {**state, 'stage': 4}):
      with self.subTest(legacy=legacy), self.assertRaises(ValueError):
        validate_course_state(legacy)

  def test_runner_has_task_local_load_and_logger(self):
    from src.tasks.leg_manip.rl import LegManipOnPolicyRunner
    self.assertIn('__init__', LegManipOnPolicyRunner.__dict__)
    self.assertIn('load', LegManipOnPolicyRunner.__dict__)

class CourseLoggingTests(unittest.TestCase):
  def test_streaming_metrics_and_real_logger_write_six_groups_and_survival_fields(self):
    from src.tasks.leg_manip.rl.metrics import CourseLogger
    env, term = rich_env()
    term.record_outcomes()
    written = {}
    writer = SimpleNamespace(add_scalar=lambda key, value, iteration: written.__setitem__(key, value))
    base = SimpleNamespace(writer=writer, log=lambda **kwargs: None)
    logger = CourseLogger(base, term.training_metrics)
    logger.log(it=3)
    self.assertIn('Metrics/twist/quad_stationary_base_height', written)
    for group in ('quad_stationary', 'tripod_stationary', 'biped_stationary', 'quad_moving', 'tripod_moving', 'biped_moving'):
      self.assertIn('Metrics/twist/' + group + '_samples', written)
    self.assertEqual(float(term.training_metrics.tracking.sum()), 0.)
    term.training_metrics.landing_event('attempts', torch.ones(4, dtype=torch.bool))
    term.training_metrics.switch_event(0, 'attempts', torch.ones(4, dtype=torch.bool))
    logger.log(it=4)
    self.assertEqual(float(logger.landing_totals[0]), 4.)
    self.assertEqual(float(logger.switch_totals[0, 0]), 4.)
    logger.restore_counts(logger.landing_totals.clone(), logger.switch_totals.clone())
    self.assertEqual(float(logger.landing_totals[3]), 4.)
    self.assertEqual(float(logger.switch_totals[0, 3]), 4.)

class CourseStreamingTests(unittest.TestCase):
  def test_per_step_groups_do_not_import_completed_episode_stats(self):
    env, term = make_env(stage=0)
    term.phase[:] = 0
    term.elapsed[:] = 2.
    term.enabled[:] = False
    term.mode[:] = False
    term.record_outcomes()
    self.assertEqual(float(term.curriculum_window[0, 0]), 4.)
    self.assertEqual(float(term.curriculum_window[0, 1]), 4.)
    term.curriculum_window.zero_()
    term.stats[:, :6] = 1e6
    manipulation_curriculum(env, torch.arange(4))
    torch.testing.assert_close(term.curriculum_window, torch.zeros(4, 3))
    self.assertEqual(term.curriculum_episodes, 4)

  def test_episode_timeout_cannot_certify_pending_switch(self):
    test = CourseSwitchTests()
    env, term = test.biped_env()
    test.tick(env, term, 100)
    env.termination_manager.time_outs = torch.tensor((True, False, False, False))
    test.tick(env, term, 1)
    self.assertEqual(float(term.switch_window[0, 1]), 1.)
    self.assertEqual(int(term.pending_switch[0]), -1)

  def test_descent_requires_recovered_endpoint_held_after_phase_completion(self):
    test = CourseSwitchTests()
    env, term = test.biped_env()
    test.tick(env, term, 105)
    term.set_command([0, 1], target_offset=(0., 0., 0.))
    test.tick(env, term, 70)
    self.assertEqual(float(term.switch_window[1, 0]), 2.)
    self.assertEqual(float(term.switch_window[1, 1]), 0.)
    data = env.scene['robot'].data
    data.root_link_pos_w[:, 2] = term.workspace.nominal_height
    data.projected_gravity_b[:] = torch.tensor((0., 0., -1.))
    data.joint_pos[:] = data.default_joint_pos
    env.scene['feet_ground_contact'].data.force[:, :, 2] = 30.
    test.tick(env, term, 64)
    self.assertEqual(float(term.switch_window[1, 1]), 0.)
    test.tick(env, term, 6)
    self.assertEqual(float(term.switch_window[1, 1]), 2.)

class CourseReviewRegressionTests(unittest.TestCase):
  def test_logger_separates_quad_tripod_and_biped_in_both_speed_groups(self):
    from src.tasks.leg_manip.rl.metrics import CourseLogger
    from src.tasks.leg_manip.mdp.metrics import GROUPS
    self.assertEqual(set(GROUPS), {f'{mode}_{speed}' for mode in ('quad', 'tripod', 'biped') for speed in ('stationary', 'moving')})
    env, term = rich_env(count=6)
    term.mode[:] = torch.tensor((False, False, True, False, False, True))
    term.manual[:] = False
    term.enabled[:] = torch.tensor((False, True, True, False, True, True))
    term._operation[:] = torch.tensor((False, True, False, False, True, False))
    term.command[3:, 0] = .1
    term.record_outcomes()
    written = {}
    base = SimpleNamespace(writer=SimpleNamespace(add_scalar=lambda key, value, it: written.__setitem__(key, value)), log=lambda **kw: None)
    CourseLogger(base, term.training_metrics).log(it=1)
    for group in GROUPS:
      self.assertEqual(written['Metrics/twist/' + group + '_samples'], 1.)

  def test_failed_rise_cannot_supply_a_certified_descent_origin(self):
    env, term = make_env(stage=0)
    # Start rise at a valid quad source, but remain physically in quad forever.
    term.set_command([0], target_offset=term.nominal_biped_offset)
    advance(env, term, 1)
    term.record_outcomes()
    term.set_command([0], target_offset=(0., 0., 0.))
    for _ in range(120):
      advance(env, term, 1)
      term.record_outcomes()
    self.assertEqual(float(term.switch_window[0, 1]), 0.)
    self.assertEqual(float(term.switch_window[1, 0]), 1.)
    self.assertEqual(float(term.switch_window[1, 1]), 0.)
    self.assertGreater(float(term.training_metrics.switches[1, 4]), 0.)

  def test_course_decision_retains_failure_diagnostics_after_window_reset(self):
    env, term = CourseCurriculumTests().qualified()
    term.curriculum_window[1, 1] = 0.
    result = manipulation_curriculum(env, torch.empty(0, dtype=torch.long))
    self.assertEqual(float(result['decision_passed']), 0.)
    self.assertEqual(float(result['biped_stance_ok']), 0.)
    self.assertEqual(float(result['quad_stance_ok']), 1.)
    self.assertEqual(float(result['biped_stance_samples']), 200.)
    torch.testing.assert_close(term.curriculum_window, torch.zeros(4, 3))
    again = manipulation_curriculum(env, torch.empty(0, dtype=torch.long))
    self.assertEqual(float(again['biped_stance_samples']), 200.)

  def test_reset_tracking_extras_separate_biped_from_tripod_hold(self):
    env, term = make_env(stage=2)
    term.phase[:] = 3
    term.mode[:] = True
    term.record_outcomes()
    extras = term.reset(torch.arange(4))
    self.assertEqual(extras['biped_stationary_samples'], 4.)
    self.assertEqual(extras['tripod_stationary_samples'], 0.)

class CourseGroupSurvivalTests(unittest.TestCase):
  def test_biped_moving_failures_cannot_be_hidden_by_global_survival(self):
    env, term = CourseCurriculumTests().qualified(stage=1)
    term.group_survival[:] = torch.tensor((20., 0.))
    term.group_survival[3] = torch.tensor((20., 12.))
    term.curriculum_episodes = 100
    term.curriculum_failures = 12
    result = manipulation_curriculum(env, torch.empty(0, dtype=torch.long))
    self.assertEqual(term.pass_streak, 0)
    self.assertEqual(float(result['biped_moving_survival_ok']), 0.)
    self.assertGreater(float(result['overall_survival']), .8)

  def test_terminal_context_counts_failure_before_stable_phase(self):
    env, term = make_env(stage=1)
    term.mode[:] = True
    term.phase[:] = 1
    term.requested_velocity[:, 0] = .1
    env.termination_manager.terminated[:] = True
    term.record_outcomes()
    manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(float(term.group_survival[3, 0]), 4.)
    self.assertEqual(float(term.group_survival[3, 1]), 4.)
    self.assertEqual(float(term.curriculum_window[3, 0]), 0.)

class CourseSourceCertificationTests(unittest.TestCase):
  def test_healthy_moving_quad_and_biped_sources_are_certified_using_previous_velocity(self):
    for biped in (False, True):
      env, term = make_env(stage=1)
      data = env.scene['robot'].data
      if biped:
        data.root_link_pos_w[:, 2] = .52
        data.projected_gravity_b[:] = torch.tensor((-1., 0., 0.))
        data.joint_pos[:] = torch.tensor(DESIRED_ANGLES)
        env.scene['feet_ground_contact'].data.force[:, :2] = 0.
        term.mode[:] = True
        term.group_weight[:] = 1.
        term.phase[:] = 3
        term.enabled[:] = True
        term.requested_offset[:] = term.nominal_biped_offset
        term.command[:, 3:] = term.nominal_biped_offset
      term.command[:, 0] = .2
      data.root_link_lin_vel_w[:, 0] = .2
      self.assertFalse(term.stance_ready(biped=biped).any())
      term.set_command([0], velocity=(0., 0., 0.), target_offset=(0., 0., 0.) if biped else term.nominal_biped_offset)
      advance(env, term, 1)
      self.assertEqual(int(term.pending_switch[0]), 1 if biped else 0)

  def test_automatic_resets_start_quad_dwell_in_every_stage(self):
    from src.tasks.leg_manip.mdp.commands import LegManipCommandCfg
    self.assertEqual(LegManipCommandCfg().resampling_time_range, (6., 9.))
    for stage in (0, 1, 2):
      env, term = make_env(count=128, stage=stage)
      term.reset(torch.arange(128))
      torch.testing.assert_close(term.command, torch.zeros(128, 6))
      self.assertFalse(term.enabled.any())

  def test_checkpoint_rejects_invalid_cumulative_event_counts(self):
    from src.tasks.leg_manip.rl.checkpoint import course_state, validate_course_state
    env, term = make_env(stage=0)
    state = course_state(term, torch.zeros(4), torch.zeros(2, 5))
    for value in ('bad', None, torch.zeros(2, 4), torch.full((2, 5), float('nan')), -torch.ones(2, 5),
                  torch.tensor(((0., 1., 0., 0., 0.), (0., 0., 0., 0., 0.)))):
      with self.subTest(value=value), self.assertRaises(ValueError):
        validate_course_state({**state, 'switch_counts': value})
