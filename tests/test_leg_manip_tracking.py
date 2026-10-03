"""Command-scaled horizontal tracking across support modes and anchor axes."""

import unittest
import torch

from tests.test_leg_manip_rewards import rich_env
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp import rewards


class HorizontalTrackingTests(unittest.TestCase):
  def make_batch(self, commands, actual, biped=False):
    env, term = rich_env(len(commands))
    term._command[:, :2] = torch.tensor(commands)
    env.scene['robot'].data.root_link_lin_vel_w[:, :2] = torch.tensor(actual)
    term.group_weight.fill_(float(biped))
    env.action_manager.get_term('joint_pos').height_score = torch.ones(len(commands))
    return env, term

  def score(self, env, biped):
    return (rewards.biped_tracking_lin_vel if biped else rewards.track_linear_velocity)(env)

  def test_low_speed_tracking_rewards_progress_in_both_modes(self):
    for biped in (False, True):
      with self.subTest(biped=biped):
        env, _ = self.make_batch([[.12, 0.]] * 3, [[0., 0.], [.06, 0.], [.12, 0.]], biped)
        torch.testing.assert_close(self.score(env, biped), torch.tensor([.1690133, .6411804, 1.]))

  def test_zero_command_drift_is_penalized_in_both_modes(self):
    for biped in (False, True):
      with self.subTest(biped=biped):
        env, _ = self.make_batch([[0., 0.]] * 3, [[0., 0.], [.02, 0.], [.08, 0.]], biped)
        torch.testing.assert_close(self.score(env, biped), torch.tensor([1., .8521438, .0773047]))

  def test_reverse_lateral_and_diagonal_use_command_magnitude(self):
    for biped in (False, True):
      with self.subTest(biped=biped):
        env, _ = self.make_batch([[.12, 0.], [-.12, 0.], [0., -.12], [.072, .096]], [[0., 0.]] * 4, biped)
        torch.testing.assert_close(self.score(env, biped), torch.full((4,), .1690133))

  def test_sigma_floor_and_relative_tolerance_at_higher_speed(self):
    for biped in (False, True):
      with self.subTest(biped=biped):
        env, _ = self.make_batch([[.02, 0.], [.8, 0.]], [[0., 0.], [0., 0.]], biped)
        torch.testing.assert_close(self.score(env, biped), torch.tensor([.8521438, .1690133]))

  def test_tracking_uses_horizontal_anchor_axes_through_rear_up_and_yaw(self):
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    for biped in (False, True):
      with self.subTest(biped=biped):
        env, term = self.make_batch([[.12, 0.]] * 4, [[0., 0.]] * 4, biped)
        data = env.scene['robot'].data
        data.root_link_quat_w = quat_from_euler_xyz(torch.zeros(4),
          torch.tensor([0., -.7853982, -1.5707963, -1.5707963]), torch.tensor([0., .8, 1.1, -1.]))
        actual_anchor = torch.tensor([.06, 0., 0.]).expand(4, 3)
        data.root_link_lin_vel_w[:] = (term.basis_w @ actual_anchor.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(self.score(env, biped), torch.full((4,), .6411804))

  def test_group_weights_and_biped_height_gate_are_preserved(self):
    env, term = self.make_batch([[.12, 0.]] * 4, [[0., 0.]] * 4)
    term.group_weight[:] = torch.tensor([0., .25, .5, 1.])
    env.action_manager.get_term('joint_pos').height_score = torch.tensor([1., 0., 1., 1.])
    torch.testing.assert_close(rewards.track_linear_velocity(env), torch.tensor([.1690133, .1267600, .0845067, 0.]))
    torch.testing.assert_close(rewards.biped_tracking_lin_vel(env), torch.tensor([0., 0., .0845067, .1690133]))

  def test_loco_vertical_damping_and_tripod_kernel_are_preserved(self):
    env, term = self.make_batch([[.12, 0.], [.12, 0.]], [[0., 0.], [0., 0.]])
    env.scene['robot'].data.root_link_lin_vel_w[:, 2] = .1
    term.blend[:] = torch.tensor([0., 1.])
    torch.testing.assert_close(rewards.track_linear_velocity(env), torch.tensor([.1690133, 1.0545848]) * .9231163)

  def test_config_exposes_the_approved_parameters_for_both_modes(self):
    cfg = leg_manip_env_cfg()
    for name in ('track_linear_velocity', 'biped_tracking_lin_vel'):
      self.assertEqual(cfg.rewards[name].params.get('min_std'), .05)
      self.assertEqual(cfg.rewards[name].params.get('speed_ratio'), .75)


if __name__ == '__main__':
  unittest.main()
