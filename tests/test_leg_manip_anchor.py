"""Unified anchor frame and band constants contracts for leg_manip."""

import math
import unittest

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz
from src.tasks.loco_pedipulation.mdp.anchor_frame import anchor_basis as loco_anchor_basis
from src.tasks.loco_pedipulation.mdp.anchor_frame import to_anchor as loco_to_anchor


def basis(quat, gravity=None):
  from src.tasks.leg_manip.mdp.anchor_frame import anchor_basis
  count = quat.shape[0]
  if gravity is None:
    gravity = torch.tensor((0., 0., -1.)).repeat(count, 1)
  return anchor_basis(quat, gravity)


def single(euler):
  quat = quat_from_euler_xyz(torch.tensor([euler[0]]), torch.tensor([euler[1]]),
                             torch.tensor([euler[2]]))
  return basis(quat)[0]


class AnchorFrameTests(unittest.TestCase):
  def test_level_pose_matches_loco_anchor_exactly(self):
    quat = quat_from_euler_xyz(torch.zeros(4), torch.zeros(4), torch.tensor((0., .3, -.7, 2.2)))
    gravity = torch.tensor((0., 0., -1.)).repeat(4, 1)
    actual = basis(quat, gravity)
    expected = loco_anchor_basis(quat, gravity)
    torch.testing.assert_close(actual, expected)

  def test_small_pitch_matches_loco_anchor(self):
    quat = quat_from_euler_xyz(torch.zeros(2), torch.tensor((-.1, -.2)), torch.zeros(2))
    gravity = torch.tensor((0., 0., -1.)).repeat(2, 1)
    torch.testing.assert_close(basis(quat, gravity), loco_anchor_basis(quat, gravity))

  def test_full_rear_matches_pedipulation_minus_z_projection(self):
    # Nose-up 90 degrees about body Y: body -Z points horizontally forward (+X world).
    frame = single((0., -math.pi / 2., 0.))
    forward, left, up = frame[:, 0], frame[:, 1], frame[:, 2]
    torch.testing.assert_close(forward, torch.tensor((1., 0., 0.)), atol=1e-5, rtol=0.)
    torch.testing.assert_close(up, torch.tensor((0., 0., 1.)), atol=1e-5, rtol=0.)
    torch.testing.assert_close(left, torch.cross(up, forward, dim=0), atol=1e-5, rtol=0.)

  def test_forward_is_continuous_through_the_rear_up_pitch_sweep(self):
    pitches = torch.linspace(0., -math.pi / 2., 25)
    forwards = []
    for pitch in pitches:
      forwards.append(single((0., float(pitch), 0.))[:, 0])
    forwards = torch.stack(forwards)
    # Every intermediate frame keeps world +X forward and stays unit length.
    torch.testing.assert_close(forwards, torch.tensor((1., 0., 0.)).expand(25, 3),
                               atol=1e-4, rtol=0.)

  def test_frame_follows_yaw_in_both_stances(self):
    for pitch in (0., -math.pi / 2.):
      frame = single((0., pitch, math.pi / 2.))
      torch.testing.assert_close(frame[:, 0], torch.tensor((0., 1., 0.)), atol=1e-4, rtol=0.)

  def test_small_roll_tilts_forward_slightly_but_stays_forward(self):
    frame = single((.2, 0., 0.))
    forward = frame[:, 0]
    self.assertGreater(float(forward[0]), .95)
    self.assertLess(abs(float(forward[1])), .25)
    self.assertAlmostEqual(float(forward.norm()), 1., places=5)

  def test_degenerate_diagonal_falls_back_to_finite_reference(self):
    # Nose-down 45 degrees puts the body (X - Z) diagonal vertical.
    frame = single((0., math.pi / 4., 0.))
    self.assertTrue(torch.isfinite(frame).all())
    self.assertAlmostEqual(float(frame[:, 0].norm()), 1., places=5)

  def test_to_anchor_roundtrip_and_gravity_tilt(self):
    from src.tasks.leg_manip.mdp.anchor_frame import to_anchor
    quat = quat_from_euler_xyz(torch.tensor((.1,)), torch.tensor((-.3,)), torch.tensor((.8,)))
    gravity = torch.tensor(((.05, -.02, -1.),))
    frame = basis(quat, gravity)
    vector = torch.tensor(((.3, -.4, .7),))
    local = to_anchor(frame, vector)
    self.assertEqual(local.shape, (1, 3))
    recovered = (frame @ local[..., None]).squeeze(-1)
    torch.testing.assert_close(recovered, vector)
    # Local Z is always along gravity-up, never along the tilted body axis.
    body_z = quat_apply(quat, torch.tensor((0., 0., 1.)))
    up_world = -gravity / gravity.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(frame[:, :, 2], up_world)
    self.assertLess(float((frame[:, :, 2] * body_z).sum()), .99)


class BandConstantsTests(unittest.TestCase):
  def test_biped_lower_limit_is_ten_centimeters_below_tripod_upper_limit(self):
    from src.tasks.leg_manip import constants
    self.assertAlmostEqual(constants.TRIPOD_HIGH - constants.BIPED_LOW, .10, places=9)
    self.assertLess(constants.DZ_MIN, constants.BIPED_LOW)
    self.assertLess(constants.TRIPOD_HIGH, constants.BIPED_HIGH)

  def test_band_limits_respect_the_measured_workspace(self):
    # FK sweep on the stand spec (2026-09-29): low segment reaches dz .04-.68,
    # high segment covers dz .14-.77 with endpoint clearance.
    from src.tasks.leg_manip import constants
    self.assertLessEqual(constants.TRIPOD_HIGH, .60)
    self.assertGreaterEqual(constants.BIPED_LOW, .145)
    self.assertLessEqual(constants.BIPED_HIGH, .77)


if __name__ == '__main__':
  unittest.main()
