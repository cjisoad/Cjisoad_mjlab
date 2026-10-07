"""Motion sampling and observation contracts for robust bridge tracking."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.tasks.pedipulation_bridge_t import guidance, observations
from src.tasks.leg_manip.mdp import observations as teacher_observations
from src.tasks.teacher_common.anchor_frame import to_anchor
from test_leg_manip_commands import make_env


def reference_fixture():
  root = np.zeros((4, 13), dtype=np.float32)
  root[:, 2] = (.3, .4, .5, .5)
  root[:, 3] = (1., -1., 1., 1.)
  root[:2, 7:10] = (1., 2., 3.)
  root[:2, 10:13] = (.1, .2, .3)
  joints = np.broadcast_to(np.array((0., 1., 2., 2.))[:, None], (4, 12)).copy()
  velocities = np.zeros((4, 12)); velocities[:2] = 1.
  feet = root[:, None, :3] + np.broadcast_to(np.array((.1, .2, -.2)), (4, 4, 3))
  banks = np.stack((joints[:, 3:6], joints[:, 3:6] * 2. + 10.))
  offsets = np.stack((np.full((4, 3), .1), np.full((4, 3), .2)))
  motion = SimpleNamespace(fps=1., data=dict(time=np.arange(4.), root_state=root,
    joint_pos=joints, joint_vel=velocities, foot_pos_w=feet, contact=np.ones((4, 4)),
    fl_unload=np.arange(4.) / 3., fr_joint_bank=banks, fr_offset_bank=offsets,
    foot_zero=np.array((.01, .02, .03))))
  return guidance.RiseGuidance(motion, 'cpu')


def observation_fixture():
  env, term = make_env(count=2)
  action = env.action_manager.get_term('joint_pos')
  term.course_stage = 0
  term.disturbed = torch.tensor((False, True))
  term.full_start = torch.tensor((True, False))
  term.reference_speed = torch.tensor((1., 1.1))
  term.progress = torch.tensor((.2, .7))
  term.orientation_error_b = torch.tensor((.1, .2, .3)).expand(2, 3)
  term.fr_error = torch.tensor((.01, .02, .03)).expand(2, 3)
  term.joint_ids = torch.arange(12)
  term.site_ids = torch.arange(4)
  term.support_ids = torch.tensor(guidance.SUPPORT_JOINTS)
  term.reference_state = dict(height=torch.tensor((.4, .5)),
    gravity=torch.tensor((0., 0., -1.)).expand(2, 3),
    fl_position=torch.zeros(2, 3), fl_unload=torch.zeros(2),
    joint_pos=torch.full((2, 12), .3), joint_vel=torch.full((2, 12), 2.),
    linear_velocity=torch.tensor((.4, .5, .6)).expand(2, 3),
    angular_velocity=torch.tensor((.7, .8, .9)).expand(2, 3),
    contact=torch.tensor((0., 0., 1., 1.)).expand(2, 4))
  action.dr_observation = torch.arange(34.).expand(2, 34)
  env.scene['nonfoot_ground_touch'] = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(2, 2, 3)))
  return env, term, action


class TrackingReferenceTests(unittest.TestCase):
  def test_complete_reference_exposes_exact_root_and_velocity_state(self):
    ref = reference_fixture()
    sample = ref.sample(torch.tensor((.25,)), speed=torch.tensor((2.,)))
    for key in ('quaternion', 'joint_vel', 'linear_velocity', 'angular_velocity',
                'fl_height', 'root_state', 'fr_offset'):
      self.assertIn(key, sample)
    self.assertAlmostEqual(float(sample['height'][0]), .35, places=6)
    torch.testing.assert_close(sample['root_state'][:, 7:10], torch.tensor(((2., 4., 6.),)))
    torch.testing.assert_close(sample['joint_vel'][:, :3], torch.full((1, 3), 2.))
    torch.testing.assert_close(sample['linear_velocity'], sample['root_state'][:, 7:10])
    torch.testing.assert_close(sample['fl_height'], torch.tensor((.15,)))

  def test_quaternion_interpolation_uses_shortest_hemisphere(self):
    sample = reference_fixture().sample(torch.tensor((.25,)))
    self.assertIn('quaternion', sample)
    torch.testing.assert_close(sample['quaternion'], torch.tensor(((1., 0., 0., 0.),)))
    torch.testing.assert_close(sample['gravity'], torch.tensor(((0., 0., -1.),)))

  def test_selected_fr_bank_controls_joints_velocity_and_offset(self):
    ref = reference_fixture()
    sample = ref.sample(torch.tensor((.5, .5)), speed=torch.tensor((1., 2.)),
      bank_index=torch.tensor((0, 1)))
    torch.testing.assert_close(sample['joint_pos'][:, 3:6], torch.tensor(((1.,) * 3, (12.,) * 3)))
    torch.testing.assert_close(sample['joint_vel'][:, 3:6], torch.tensor(((1.,) * 3, (4.,) * 3)))
    torch.testing.assert_close(sample['fr_offset'], torch.tensor(((.1,) * 3, (.2,) * 3)))
    torch.testing.assert_close(sample['foot_pos_w'][:, 1] - sample['root_state'][:, :3],
      ref.zero + sample['fr_offset'])

  def test_exact_time_sampling_and_completed_endpoint_are_stationary(self):
    ref = reference_fixture()
    exact = ref.sample_time(torch.tensor((1.,)), speed=1.5, bank_index=1)
    progress = ref.sample(torch.tensor((.5,)), speed=1.5, bank_index=1)
    for key in exact: torch.testing.assert_close(exact[key], progress[key])
    held = ref.sample(torch.tensor((1., 2.)), speed=2., bank_index=1)
    torch.testing.assert_close(held['joint_pos'][0], held['joint_pos'][1])
    torch.testing.assert_close(held['joint_vel'], torch.zeros(2, 12))
    torch.testing.assert_close(held['root_state'][:, 7:], torch.zeros(2, 6))

  def test_anchor_reference_velocities_follow_reference_heading(self):
    ref = reference_fixture()
    ref.frames['root_state'][:, 3:7] = torch.tensor((2.**-.5, 0., 0., 2.**-.5))
    sample = ref.sample_time(torch.tensor((.5,)))
    torch.testing.assert_close(sample['linear_velocity'], torch.tensor(((2., -1., 3.),)), atol=1e-6, rtol=1e-6)


class TrackingObservationTests(unittest.TestCase):
  def test_actor84_layout_preserves_public_teacher51(self):
    env, term, action = observation_fixture()
    public = teacher_observations.policy_state(env, add_noise=False).clone()
    actor = observations.policy_state(env, add_noise=False)
    self.assertEqual(actor.shape, (2, 84))
    self.assertEqual((guidance.ACTOR_DIM, guidance.CRITIC_DIM, guidance.GUIDANCE_DIM), (84, 172, 33))
    torch.testing.assert_close(actor[:, :51], public)
    expected = torch.cat((env.scene['robot'].data.root_link_pos_w[:, 2, None], term.fr_error,
      term.progress[:, None], term.orientation_error_b, term.reference_state['height'][:, None],
      term.reference_state['joint_pos'][:, term.support_ids] - action.default_angles[:, term.support_ids],
      term.reference_state['joint_vel'][:, term.support_ids] * .05,
      term.reference_state['linear_velocity'] * 2., term.reference_state['angular_velocity'] * .25), -1)
    torch.testing.assert_close(actor[:, 51:], expected)

  def test_critic172_shares_full_actor_and_exact_physical_state(self):
    env, term, action = observation_fixture()
    data = env.scene['robot'].data
    data.root_link_lin_vel_w[:] = torch.tensor((.2, .3, .4))
    data.root_link_ang_vel_w[:] = torch.tensor((.5, .6, .7))
    env.scene['nonfoot_ground_touch'].data.force[:, 0, 2] = 50.
    actor = observations.policy_state(env, add_noise=False)
    shared = observations.shared_policy_state(env)
    self.assertEqual(shared.shape, (2, 84))
    torch.testing.assert_close(shared, actor)
    privileged = observations.privileged_state(env)
    self.assertEqual(privileged.shape, (2, 88))
    self.assertEqual(torch.cat((shared, privileged), -1).shape, (2, 172))
    torch.testing.assert_close(privileged[:, :3], to_anchor(term.basis_w, data.root_link_lin_vel_w))
    torch.testing.assert_close(privileged[:, 46:47], torch.full((2, 1), .5))
    torch.testing.assert_close(privileged[:, 47:81], action.dr_observation)

  def test_noise_is_enabled_only_for_stage2_disturbed_worlds(self):
    env, term, _ = observation_fixture()
    clean = observations.policy_state(env, add_noise=False).clone()
    term.course_stage = 1
    torch.testing.assert_close(observations.policy_state(env, add_noise=True), clean)
    term.course_stage = 2
    torch.manual_seed(17)
    noisy = observations.policy_state(env, add_noise=True)
    torch.testing.assert_close(noisy[0], clean[0])
    self.assertFalse(torch.equal(noisy[1], clean[1]))
    torch.testing.assert_close(noisy[:, 56:], clean[:, 56:])

  def test_privileged_feet_force_follows_canonical_contact_order(self):
    env, term, _ = observation_fixture()
    term.contact_order = torch.tensor((3, 2, 1, 0))
    force = env.scene['feet_ground_contact'].data.force
    force[:] = torch.arange(12.).reshape(1, 4, 3)
    privileged = observations.privileged_state(env)
    expected = force[:, term.contact_order].flatten(1)/100.
    torch.testing.assert_close(privileged[:, 30:42], expected)


if __name__ == '__main__': unittest.main()
