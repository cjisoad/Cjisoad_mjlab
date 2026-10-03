"""Quadruped gait must complete stance and swing together; preserve other modes."""

import unittest
import torch

from tests.test_leg_manip_commands import make_env
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp import rewards
from src.tasks.loco_pedipulation.mdp import rewards as loco


def contacts(env, term, values):
  force = env.scene['feet_ground_contact'].data.force
  force.zero_()
  force[:, term.contact_order, 2] = torch.as_tensor(values).float() * 30.


class QuadrupedGaitTests(unittest.TestCase):
  def test_stationary_all_contact_never_scores_even_during_stance_overlap(self):
    env, term = make_env(101)
    term._command[:, 0] = .12
    term.gait_phase[:] = torch.linspace(0., .999, 101)
    contacts(env, term, torch.ones(101, 4))
    torch.testing.assert_close(rewards.feet_gait(env), torch.zeros(101))

  def test_correct_stance_and_swing_both_required(self):
    env, term = make_env(6)
    term._command[:, 0] = .12
    term.gait_phase[:] = torch.tensor([.25, .25, .25, .25, .75, .75])
    contacts(env, term, [[0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0],
                         [0, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]])
    torch.testing.assert_close(rewards.feet_gait(env), torch.tensor([1., 0., 0., 0., 1., 0.]))

  def test_zero_command_and_biped_group_still_disable_quad_gait(self):
    env, term = make_env(4)
    term._command[:, 0] = torch.tensor([.12, 0., .12, .12])
    term.group_weight[:] = torch.tensor([0., 0., .25, 1.])
    term.gait_phase[:] = .25
    contacts(env, term, [[0, 1, 1, 0]] * 4)
    torch.testing.assert_close(rewards.feet_gait(env), torch.tensor([1., 0., .75, 0.]))

  def test_tripod_gait_remains_identical_to_source(self):
    env, term = make_env(32)
    term._command[:, 0] = .12
    term.gait_phase[:] = torch.linspace(0., .99, 32)
    term.blend[:] = 1.
    contact = (torch.arange(128).reshape(32, 4) % 3) != 0
    contacts(env, term, contact)
    torch.testing.assert_close(rewards.feet_gait(env), loco.feet_gait(env))

  def test_tripod_transition_keeps_existing_linear_blend(self):
    env, term = make_env(4)
    term._command[:, 0] = .12
    term.gait_phase[:] = .25
    term.blend[:] = torch.tensor([0., .25, .5, 1.])
    contacts(env, term, torch.ones(4, 4))
    blend = term.blend.clone()
    term.blend[:] = 1.
    tripod = loco.feet_gait(env)
    term.blend[:] = blend
    torch.testing.assert_close(rewards.feet_gait(env), tripod * blend)


class TrackingWeightTests(unittest.TestCase):
  def test_quad_and_biped_horizontal_weights_are_equal(self):
    cfg = leg_manip_env_cfg()
    self.assertEqual(cfg.rewards['track_linear_velocity'].weight, 2.5)
    self.assertEqual(cfg.rewards['biped_tracking_lin_vel'].weight, 2.5)
    self.assertEqual(cfg.rewards['track_angular_velocity'].weight, 1.)

  def test_weighted_tripod_tracking_remains_two_with_existing_vertical_damping(self):
    env, term = make_env(4)
    term._command[:, 0] = .12
    term.blend[:] = torch.tensor([0., .25, .5, 1.])
    env.scene['robot'].data.root_link_lin_vel_w[:, 2] = .1
    cfg = leg_manip_env_cfg().rewards['track_linear_velocity']
    actual = cfg.func(env, **cfg.params) * cfg.weight
    quad = 2.5 * torch.exp(torch.tensor(-.12**2 / .09**2))
    tripod = 2. * torch.exp(torch.tensor(-.12**2 / .15**2))
    expected = torch.lerp(quad.expand(4), tripod.expand(4), term.blend)
    expected *= torch.exp(torch.tensor(-2. * .1**2 / .5**2))
    torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
  unittest.main()
