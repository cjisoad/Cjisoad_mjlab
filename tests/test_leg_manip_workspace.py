"""Unified two-segment FR foot workspace contracts for leg_manip."""

import unittest

import mujoco
import numpy as np

from src.tasks.leg_manip.constants import BIPED_HIGH, BIPED_LOW, DZ_MIN, TRIPOD_HIGH
from src.tasks.leg_manip.mdp.foot_workspace import HIGH, LOW, build_foot_workspace
from src.tasks.pedipulation.constants import DEFAULT_ANGLES, DESIRED_ANGLES, JOINT_NAMES
from src.tasks.pedipulation.robot import get_stand_spec


class FootWorkspaceTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.workspace = build_foot_workspace()

  def test_bank_is_nonempty_with_both_segments(self):
    workspace = self.workspace
    self.assertGreater(len(workspace.offsets), 0)
    self.assertEqual(workspace.offsets.shape, workspace.joint_positions.shape)
    self.assertEqual(workspace.offsets.shape[0], workspace.segment.shape[0])
    self.assertGreater((workspace.segment == LOW).sum(), 0)
    self.assertGreater((workspace.segment == HIGH).sum(), 0)

  def test_zero_radius_and_nominal_height_match_stand_spec_fk(self):
    workspace = self.workspace
    np.testing.assert_allclose(workspace.zero, (.17782, -.17260, -.30022), atol=1e-4)
    self.assertAlmostEqual(workspace.foot_radius, .022, places=5)
    self.assertAlmostEqual(workspace.nominal_height, workspace.foot_radius - workspace.zero[2],
                           places=7)

  def test_segment_dz_ranges_respect_band_limits(self):
    dz = self.workspace.offsets[:, 2]
    segment = self.workspace.segment
    self.assertGreaterEqual(dz[segment == LOW].min(), DZ_MIN - 1e-6)
    self.assertLessEqual(dz[segment == LOW].max(), TRIPOD_HIGH + 1e-6)
    self.assertGreaterEqual(dz[segment == HIGH].min(), BIPED_LOW - 1e-6)
    self.assertLessEqual(dz[segment == HIGH].max(), BIPED_HIGH + 1e-6)

  def test_overlap_band_is_populated_from_both_segments(self):
    dz = self.workspace.offsets[:, 2]
    segment = self.workspace.segment
    in_band = (dz >= BIPED_LOW) & (dz <= TRIPOD_HIGH)
    self.assertGreater((in_band & (segment == LOW)).sum(), 0)
    self.assertGreater((in_band & (segment == HIGH)).sum(), 0)

  def test_easy_curriculum_subset_is_nonempty(self):
    offsets = self.workspace.offsets
    segment = self.workspace.segment
    easy = (segment == LOW) & (np.abs(offsets[:, :2]) <= .045).all(-1) & (offsets[:, 2] <= .085)
    self.assertGreater(easy.sum(), 0)

  def test_witnesses_reproduce_targets_under_fk(self):
    workspace = self.workspace
    model = get_stand_spec().compile()
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    site, base = model.site('FR').id, model.body('base_link').id
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    data = mujoco.MjData(model)
    count = len(workspace.offsets)
    checked = 0
    for index in range(0, count, max(1, count // 40)):
      angles = (DEFAULT_ANGLES if workspace.segment[index] == LOW else DESIRED_ANGLES)
      data.qpos[addresses] = angles
      data.qpos[addresses[3:6]] = workspace.joint_positions[index]
      mujoco.mj_kinematics(model, data)
      offset = data.site_xpos[site] - data.xpos[base] - workspace.zero
      if workspace.segment[index] == HIGH:
        offset = rotation @ (offset + workspace.zero) - workspace.zero
      np.testing.assert_allclose(offset, workspace.offsets[index], atol=1e-4)
      checked += 1
    self.assertGreaterEqual(checked, 20)

  def test_builder_is_cached(self):
    self.assertIs(build_foot_workspace(), self.workspace)


if __name__ == '__main__':
  unittest.main()
