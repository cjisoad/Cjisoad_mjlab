"""Reward-group weighting, height gate and mode-aware termination contracts."""

from types import SimpleNamespace
import unittest

import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz
from tests.test_leg_manip_commands import advance, make_env
from src.tasks.leg_manip.constants import TRIPOD_HIGH
from src.tasks.leg_manip.mdp import rewards
from src.tasks.leg_manip.mdp.terminations import fell_over

ABOVE_BAND = TRIPOD_HIGH + .1


def rich_env(count=4):
  """Mock env extended with the sensors and action fields biped rewards read."""
  env, term = make_env(count)
  data = env.scene['robot'].data
  data.actuator_force = torch.zeros(count, 12)
  data.soft_joint_pos_limits = torch.stack((torch.full((12,), -3.), torch.full((12,), 3.)), dim=-1).repeat(count, 1, 1)
  for name in ('front_contact', 'rear_contact', 'nonfoot_contact', 'base_contact'):
    env.scene[name] = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(count, 2, 3)))
  env.episode_length_buf = torch.zeros(count, dtype=torch.long)
  action = env.action_manager.get_term('joint_pos')
  action.desired_angles = torch.arange(12.) * .1
  action.previous_action = torch.zeros(count, 12)
  action.previous_joint_vel = torch.zeros(count, 12)
  action.height_score = torch.tensor(0.)
  action.joint_ids = torch.arange(12)
  return env, term


class GroupWeightTests(unittest.TestCase):
  def test_loco_group_scales_with_one_minus_weight(self):
    env, term = rich_env()
    value = rewards.base_height(env, std=.06)
    self.assertGreater(float(value[0]), .9, 'Nominal stance should score near one')
    term.group_weight[:] = 1.
    torch.testing.assert_close(rewards.base_height(env, std=.06), torch.zeros(4))
    term.group_weight[:] = .25
    torch.testing.assert_close(rewards.base_height(env, std=.06), value * .75)

  def test_biped_group_scales_with_weight(self):
    env, term = rich_env()
    torch.testing.assert_close(rewards.biped_base_height(env), torch.zeros(4))
    term.group_weight[:] = 1.
    height = float(env.scene['robot'].data.root_link_pos_w[0, 2])
    expected = torch.exp(torch.tensor(-abs(height - .52) * 5.))
    torch.testing.assert_close(rewards.biped_base_height(env), expected.repeat(4))

  def test_shared_terms_ignore_group_weight(self):
    env, term = rich_env()
    for weight in (0., .5, 1.):
      term.group_weight[:] = weight
      torch.testing.assert_close(rewards.alive(env), torch.ones(4))

  def test_biped_base_height_writes_per_environment_height_score_gate(self):
    env, term = rich_env()
    action = env.action_manager.get_term('joint_pos')
    term.group_weight[:] = 1.
    rewards.biped_base_height(env)
    self.assertTrue((action.height_score < .70).all())
    env.scene['robot'].data.root_link_pos_w[:, 2] = .52
    rewards.biped_base_height(env)
    self.assertTrue((action.height_score > .70).all())

  def test_height_gate_is_independent_in_mixed_batch(self):
    env, term = rich_env()
    heights = env.scene['robot'].data.root_link_pos_w[:, 2]
    heights[:] = .322
    heights[0] = .52
    term.group_weight[:] = 1.
    rewards.biped_base_height(env)
    score = env.action_manager.get_term('joint_pos').height_score
    self.assertEqual(tuple(score.shape), (4,))
    before = rewards.biped_tracking_lin_vel(env)
    torch.testing.assert_close(before, torch.tensor((1., 0., 0., 0.)))
    heights[1:] = .52
    rewards.biped_base_height(env)
    torch.testing.assert_close(rewards.biped_tracking_lin_vel(env)[0], before[0])

  def test_biped_velocity_tracking_uses_anchor_frame_and_gate(self):
    env, term = rich_env()
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 50)
    self.assertTrue(term.mode[0])
    self.assertAlmostEqual(float(term.group_weight[0]), 1., places=5)
    env.scene['robot'].data.root_link_pos_w[:, 2] = .52
    rewards.biped_base_height(env)
    term.set_command([0], velocity=(.1, 0., 0.))
    advance(env, term, 30)
    value = rewards.biped_tracking_lin_vel(env)
    self.assertGreater(float(value[0]), 0.)
    self.assertLess(float(value[0]), 1.)
    # Command matches actual velocity: full gated score.
    term.set_command([0], velocity=(0., 0., 0.))
    advance(env, term, 30)
    torch.testing.assert_close(rewards.biped_tracking_lin_vel(env)[0], torch.tensor(1.))
    # Without the height gate the term vanishes.
    action = env.action_manager.get_term('joint_pos')
    action.height_score = torch.tensor(0.)
    torch.testing.assert_close(rewards.biped_tracking_lin_vel(env), torch.zeros(4))

  def test_fr_position_tracking_is_shared_across_modes(self):
    env, term = rich_env()
    term.blend[:] = 1.
    for weight in (0., 1.):
      term.group_weight[:] = weight
      torch.testing.assert_close(rewards.fr_position_tracking(env, std=.05), torch.ones(4),
                                 atol=1e-6, rtol=1e-6)

  def test_fr_default_reward_disabled_while_target_active(self):
    env, term = rich_env()
    term.set_command(torch.arange(4), target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 50)
    self.assertTrue(term.mode.all())
    action = env.action_manager.get_term('joint_pos')
    action.height_score = torch.tensor(1.)
    torch.testing.assert_close(rewards.default_pos_reward_FR(env), torch.zeros(4))
    term.set_command(torch.arange(4), target_offset=(0., 0., 0.))
    advance(env, term, 2)
    self.assertGreater(float(rewards.default_pos_reward_FR(env)[0]), 0.)

  def test_biped_fraction_metric_reports_group_weight(self):
    env, term = rich_env()
    term.group_weight[:] = .3
    torch.testing.assert_close(rewards.biped_fraction(env), torch.full((4,), .3))


class ModeAwareTerminationTests(unittest.TestCase):
  def env_with_quat(self, quat):
    from mjlab.utils.lab_api.math import quat_apply_inverse
    env, term = rich_env()
    count = env.num_envs
    quat = quat.repeat(count, 1)
    env.scene['robot'].data.root_link_quat_w = quat
    env.scene['robot'].data.projected_gravity_b = quat_apply_inverse(
      quat, torch.tensor((0., 0., -1.)).repeat(count, 1))
    return env, term

  def test_loco_mode_terminates_on_tilt(self):
    import math
    env, term = self.env_with_quat(quat_from_euler_xyz(torch.tensor(0.), torch.tensor(math.radians(70.)), torch.tensor(0.)))
    term.mode[:] = False
    self.assertTrue(fell_over(env, limit_angle=math.radians(60.)).all())
    env, term = self.env_with_quat(quat_from_euler_xyz(torch.tensor(0.), torch.tensor(math.radians(10.)), torch.tensor(0.)))
    self.assertFalse(fell_over(env, limit_angle=math.radians(60.)).any())

  def test_biped_mode_allows_full_pitch_but_limits_roll(self):
    import math
    from mjlab.utils.lab_api.math import quat_mul
    env, term = self.env_with_quat(quat_from_euler_xyz(torch.tensor(0.), torch.tensor(-math.pi / 2.), torch.tensor(0.)))
    term.mode[:] = True
    self.assertFalse(fell_over(env, limit_angle=math.radians(60.)).any(),
                     'Full rear-leg stand must survive in biped mode')
    # Sideways tip at the stand: rotate 70 degrees about world forward (body -Z).
    tipped = quat_mul(quat_from_euler_xyz(torch.tensor(math.radians(70.)), torch.tensor(0.), torch.tensor(0.)),
                      quat_from_euler_xyz(torch.tensor(0.), torch.tensor(-math.pi / 2.), torch.tensor(0.)))
    env, term = self.env_with_quat(tipped)
    term.mode[:] = True
    self.assertTrue(fell_over(env, limit_angle=math.radians(60.)).all())

  def upright_exit(self, target=(0., 0., 0.)):
    import math
    env, term = self.env_with_quat(quat_from_euler_xyz(
      torch.tensor(0.), torch.tensor(-math.pi / 2.), torch.tensor(0.)))
    term.set_command([0], target_offset=(0., 0., ABOVE_BAND))
    advance(env, term, 60)
    self.assertTrue(term.mode[0])
    term.set_command([0], target_offset=target)
    advance(env, term, 1)
    self.assertFalse(term.mode[0])
    return env, term

  def test_canceling_upright_biped_allows_descent(self):
    env, term = self.upright_exit()
    self.assertFalse(fell_over(env)[0], 'Cancel must not immediately kill the upright robot')
    self.assertAlmostEqual(float(term.group_weight[0]), .975, places=5)

  def test_low_target_also_allows_biped_descent(self):
    env, term = self.upright_exit(target=(0., 0., .1))
    self.assertFalse(fell_over(env)[0])

  def test_descent_grace_expires_even_if_reward_weight_is_zero(self):
    env, term = self.upright_exit()
    advance(env, term, 45)
    self.assertEqual(float(term.group_weight[0]), 0.)
    self.assertFalse(fell_over(env)[0], 'Descent grace must outlive the reward ramp')
    advance(env, term, int(term.cfg.biped_exit_timeout / env.step_dt))
    self.assertTrue(fell_over(env)[0], 'An upright failed exit must eventually terminate')

  def test_descent_still_terminates_sideways_and_excessive_tilt(self):
    import math
    env, term = self.upright_exit()
    gravity = env.scene['robot'].data.projected_gravity_b
    gravity[0] = torch.tensor((0., 1., 0.))
    self.assertTrue(fell_over(env)[0])
    gravity[0] = torch.tensor((-math.sin(math.radians(120.)), 0., -math.cos(math.radians(120.))))
    self.assertTrue(fell_over(env)[0])

  def test_descent_finishes_after_posture_and_front_contact_recover(self):
    import math
    env, term = self.upright_exit()
    gravity = env.scene['robot'].data.projected_gravity_b
    gravity[0] = torch.tensor((-math.sin(math.radians(40.)), 0., -math.cos(math.radians(40.))))
    force = env.scene['feet_ground_contact'].data.force
    force[0, :2] = 0.
    advance(env, term, 45)
    self.assertTrue(term.biped_exiting[0], 'Safe tilt alone does not confirm front support')
    force[0, 0, 2] = 30.
    advance(env, term, 1)
    self.assertFalse(term.biped_exiting[0])
    gravity[0] = torch.tensor((-1., 0., 0.))
    self.assertTrue(fell_over(env)[0], 'Normal LOCO limits must resume after recovery')


if __name__ == '__main__':
  unittest.main()
