"""Mode hysteresis, group-weight ramping and generalized phase machine contracts."""

from types import SimpleNamespace
import unittest

import torch

from src.tasks.leg_manip.constants import BIPED_HIGH, BIPED_LOW, STAGE_BIPED, TRIPOD_HIGH
from src.tasks.leg_manip.mdp.commands import (
  HOLD, LOWER, PREPARE, QUAD, REACH, RECOVER, LegManipCommand, LegManipCommandCfg,
)
from src.tasks.leg_manip.mdp.curriculums import manipulation_curriculum
from src.tasks.leg_manip.mdp.foot_workspace import FOOT_NAMES, FR_JOINTS, build_foot_workspace

IN_BAND = (BIPED_LOW + TRIPOD_HIGH) / 2.
BELOW_BAND = .10
ABOVE_BAND = (TRIPOD_HIGH + BIPED_HIGH) / 2.


def make_env(count=4, stage=3, **cfg_overrides):
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
  cfg = LegManipCommandCfg(initial_stage=stage, resampling_time_range=(100., 100.), **cfg_overrides)
  term = LegManipCommand(cfg, env)
  action = SimpleNamespace(joint_pos=data.joint_pos, joint_vel=data.joint_vel,
    target_ids=torch.arange(12), default_angles=data.default_joint_pos,
    raw_action=torch.zeros(count, 12), actor_sample=None)
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


class ModeHysteresisTests(unittest.TestCase):
  def test_mode_switches_only_outside_the_overlap_band(self):
    env, term = make_env()
    self.assertFalse(term.mode.any())
    term.set_command([0], target_offset=(0., 0., BELOW_BAND))
    advance(env, term, 2)
    self.assertFalse(term.mode[0])
    term.set_command([0], target_offset=(0., 0., IN_BAND))
    advance(env, term, 2)
    self.assertFalse(term.mode[0], 'Entering the band from below must keep LOCO mode')
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 2)
    self.assertTrue(term.mode[0], 'Above tripod_high must request BIPED mode')
    term.set_command([0], target_offset=(0., 0., IN_BAND))
    advance(env, term, 2)
    self.assertTrue(term.mode[0], 'Entering the band from above must keep BIPED mode')
    term.set_command([0], target_offset=(0., 0., BELOW_BAND))
    advance(env, term, 2)
    self.assertFalse(term.mode[0], 'Below biped_low must request LOCO mode')

  def test_zero_target_forces_loco_mode_and_landing(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 60)
    self.assertTrue(term.mode[0])
    self.assertEqual(int(term.phase[0]), HOLD)
    term.set_command([0], target_offset=(0., 0., 0.))
    advance(env, term, 2)
    self.assertFalse(term.mode[0])
    self.assertEqual(int(term.phase[0]), LOWER)

  def test_group_weight_ramps_smoothly_between_groups(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    weights = [float(term.group_weight[0])]
    for _ in range(45):
      advance(env, term, 1)
      weights.append(float(term.group_weight[0]))
    deltas = torch.diff(torch.tensor(weights))
    self.assertTrue((deltas >= -1e-6).all(), 'Ramp toward BIPED must be monotonic')
    self.assertAlmostEqual(weights[-1], 1., places=5)
    self.assertAlmostEqual(weights[1] - weights[0], .02 / .8, places=5)
    term.set_command([0], target_offset=(0., 0., BELOW_BAND))
    advance(env, term, 45)
    self.assertAlmostEqual(float(term.group_weight[0]), 0., places=5)

  def test_reset_clears_mode_and_group_weight(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 50)
    self.assertTrue(term.mode[0])
    term.reset(torch.tensor([0]))
    self.assertFalse(term.mode[0])
    self.assertEqual(float(term.group_weight[0]), 0.)

  def test_mode_and_weight_are_not_actor_input(self):
    env, term = make_env()
    before = term.command.clone()
    term.mode[:] = True
    term.group_weight[:] = .5
    torch.testing.assert_close(term.command, before)


class PhaseMachineTests(unittest.TestCase):
  def test_high_target_uses_standard_lift_sequence(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 1)
    self.assertEqual(int(term.phase[0]), PREPARE)
    advance(env, term, 17)
    self.assertEqual(int(term.phase[0]), REACH)
    advance(env, term, 38)
    self.assertEqual(int(term.phase[0]), HOLD)
    self.assertTrue(term.mode[0])
    self.assertAlmostEqual(float(term.blend[0]), 1., places=5)

  def test_biped_retarget_goes_straight_back_to_reach(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 60)
    self.assertEqual(int(term.phase[0]), HOLD)
    seen = set()
    term.set_command([0], target_offset=(.05, .02, ABOVE_BAND - .1))
    for _ in range(40):
      advance(env, term, 1)
      seen.add(int(term.phase[0]))
    self.assertEqual(seen, {REACH, HOLD}, 'Biped retarget must only pass REACH and HOLD')
    self.assertTrue(term.mode[0])

  def test_full_cycle_quad_tripod_biped_and_back(self):
    env, term = make_env()
    term.set_command([0], target_offset=(0., 0., .1))
    advance(env, term, 60)
    self.assertEqual(int(term.phase[0]), HOLD)
    self.assertFalse(term.mode[0], 'Low target keeps tripod semantics')
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 40)
    self.assertEqual(int(term.phase[0]), HOLD)
    self.assertTrue(term.mode[0])
    term.set_command([0], target_offset=(0., 0., 0.))
    advance(env, term, 1)
    self.assertEqual(int(term.phase[0]), LOWER)
    # Land the foot: LOWER needs reach_time then contact_confirmation.
    advance(env, term, 38)
    self.assertEqual(int(term.phase[0]), RECOVER)
    advance(env, term, 18)
    self.assertEqual(int(term.phase[0]), QUAD)
    self.assertFalse(term.mode[0])


class SamplingTests(unittest.TestCase):
  def test_stage_below_four_never_samples_above_band_targets(self):
    torch.manual_seed(0)
    env, term = make_env(stage=3)
    term.release_manual(torch.arange(4))
    for _ in range(25):
      term._resample_command(torch.arange(4))
      self.assertLessEqual(float(term.requested_offset[:, 2].max()), TRIPOD_HIGH + 1e-6)

  def test_stage_four_samples_both_segments(self):
    torch.manual_seed(0)
    env, term = make_env(stage=STAGE_BIPED)
    term.release_manual(torch.arange(4))
    highest, lowest = 0., 1.
    for _ in range(50):
      term._resample_command(torch.arange(4))
      highest = max(highest, float(term.requested_offset[:, 2].max()))
      lowest = min(lowest, float(term.requested_offset[:, 2].min()))
    self.assertGreater(highest, TRIPOD_HIGH)
    self.assertLess(lowest, BIPED_LOW)

  def test_above_band_targets_use_biped_velocity_ranges(self):
    torch.manual_seed(0)
    env, term = make_env(stage=STAGE_BIPED)
    term.release_manual(torch.arange(4))
    sampled_high = False
    for _ in range(50):
      term._resample_command(torch.arange(4))
      high = term.requested_offset[:, 2] > TRIPOD_HIGH
      sampled_high |= bool(high.any())
      if high.any():
        velocity = term.requested_velocity[high]
        self.assertTrue((velocity[:, 0] >= -.2 - 1e-6).all() and (velocity[:, 0] <= .6 + 1e-6).all())
        torch.testing.assert_close(velocity[:, 1], torch.zeros_like(velocity[:, 1]))
        self.assertTrue((velocity[:, 2].abs() <= .6 + 1e-6).all())
    self.assertTrue(sampled_high)

  def test_biped_mode_clamps_lateral_velocity(self):
    env, term = make_env()
    term.set_command([0], velocity=(.3, .3, .3), target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 40)
    self.assertTrue(term.mode[0])
    self.assertAlmostEqual(float(term.command[0, 1]), 0., places=2)
    self.assertLessEqual(float(term.command[0, 0]), .6 + 1e-6)


class CurriculumTests(unittest.TestCase):
  def qualifying_env(self, stage):
    env, term = make_env(stage=stage, min_stage_steps=100, min_curriculum_episodes=4)
    term.stage_started = 0
    env.common_step_counter = 200
    # One completed episode per env with passing statistics.
    term.stats[:, 0] = 200.   # quad steps
    term.stats[:, 1] = 10.    # quad velocity error sum
    term.stats[:, 2] = 200.   # hold steps
    term.stats[:, 3] = 2.     # hold position error sum
    term.stats[:, 4] = 2.     # hold contact count
    term.stats[:, 5] = 10.    # hold velocity error sum
    term.stats[:, 6] = 400.   # total steps -> completed episode
    term.stats[:, 7] = 0.     # terminations
    term.stats[:, 8] = 20.    # landing attempts
    term.stats[:, 9] = 19.    # landing confirmed
    return env, term

  def test_stage_three_promotes_to_biped_stage_on_success(self):
    env, term = self.qualifying_env(3)
    result = manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(term.stage, STAGE_BIPED)
    self.assertEqual(int(result['stage']), STAGE_BIPED)

  def test_biped_stage_is_terminal(self):
    env, term = self.qualifying_env(STAGE_BIPED)
    manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(term.stage, STAGE_BIPED)

  def test_poor_landing_success_blocks_promotion(self):
    env, term = self.qualifying_env(3)
    term.stats[:, 9] = 5.
    manipulation_curriculum(env, torch.arange(4))
    self.assertEqual(term.stage, 3)


if __name__ == '__main__':
  unittest.main()
