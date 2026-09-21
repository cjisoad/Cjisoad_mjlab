"""CPU-only checks of the task-local source Go2 physical model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest

import mujoco
import numpy as np


TASK = Path(__file__).resolve().parents[1] / "src/tasks/stand"


def adapter():
  package = ModuleType("_stand_robot_tests")
  package.__path__ = [str(TASK)]
  sys.modules[package.__name__] = package
  spec = importlib.util.spec_from_file_location(
    package.__name__ + ".robot", TASK / "robot.py"
  )
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class StandRobotTests(unittest.TestCase):
  def setUp(self):
    self.assertTrue((TASK / "robot.py").exists(), "Source physics adapter is missing")
    self.robot = adapter()

  def test_fixed_bodies_preserve_contact_ownership(self):
    model = self.robot.get_stand_spec().compile()
    self.assertEqual(model.nbody, 30)  # World plus all 29 URDF links.
    self.assertEqual(model.njnt, 13)
    self.assertEqual(self.robot.BASE_CONTACT, "base_link")
    self.assertEqual(len(self.robot.NONFOOT_BODIES), 25)
    for leg in ("FL", "FR", "RL", "RR"):
      foot = model.body(leg + "_foot")
      self.assertNotIn(foot.name, self.robot.NONFOOT_BODIES)
      self.assertEqual(model.geom(leg + "_foot_collision").bodyid, foot.id)
      self.assertEqual(model.site(leg).bodyid, foot.id)
      np.testing.assert_allclose(model.site(leg).pos, 0)
      np.testing.assert_allclose(model.geom(leg + "_foot_collision").pos, [-.002, 0, 0])
    self.assertEqual(model.geom("Head_upper_collision").bodyid, model.body("Head_upper").id)
    self.assertEqual(model.geom("base_link_collision").bodyid, model.body("base_link").id)

  def test_source_inertias_and_mass_are_restored(self):
    model = self.robot.get_stand_spec().compile()
    np.testing.assert_allclose(model.body("base_link").mass, [6.921])
    np.testing.assert_allclose(model.body("FL_calf").mass, [.154])
    np.testing.assert_allclose(model.body("FL_foot").mass, [.04])
    np.testing.assert_allclose(model.body("Head_upper").mass, [.001])
    # Only the eight collision-only calflower links require density inference.
    self.assertAlmostEqual(float(model.body_mass.sum()), 15.019, places=5)
    calf = model.body("FL_calf")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, calf.iquat)
    rotation = rotation.reshape(3, 3)
    inertia = rotation @ np.diag(calf.inertia) @ rotation.T
    np.testing.assert_allclose(inertia, [
      [.00108, 3.4e-7, 1.72e-5],
      [3.4e-7, .0011, 8.28e-6],
      [1.72e-5, 8.28e-6, 3.29e-5],
    ], atol=1e-10)

  def test_capsules_self_collision_and_fixed_link_geometry(self):
    model = self.robot.get_stand_spec().compile()
    for leg in ("FL", "FR", "RL", "RR"):
      geom = model.geom(leg + "_calflower1_collision")
      self.assertEqual(geom.type.item(), int(mujoco.mjtGeom.mjGEOM_CAPSULE))
      np.testing.assert_allclose(geom.size[:2], [.0155, .015])
    for gid in range(model.ngeom):
      geom = model.geom(gid)
      if geom.name.endswith("_collision"):
        self.assertEqual(geom.contype.item(), 1)
        self.assertEqual(geom.conaffinity.item(), 1)
        self.assertEqual(geom.condim.item(), 3)
        self.assertEqual(geom.priority.item(), 1)
    self.assertGreater(model.nexclude, 0)
    self.assertEqual(model.ngeom, 60)  # 33 reused visual + 27 source collision geoms.

  def test_pd_limits_joint_order_and_initial_pose(self):
    from mjlab.actuator import IdealPdActuatorCfg
    from mjlab.entity import Entity

    cfg = self.robot.get_stand_robot_cfg()
    self.assertEqual(cfg.init_state.pos, (0., 0., .42))
    for actuator in cfg.articulation.actuators:
      self.assertIsInstance(actuator, IdealPdActuatorCfg)
      self.assertEqual((actuator.stiffness, actuator.damping), (40., 1.))
    model = Entity(cfg).spec.compile()
    self.assertEqual(tuple(model.joint(i).name for i in range(1, 13)), self.robot.JOINT_NAMES)
    self.assertEqual(model.nu, 12)
    for index, name in enumerate(self.robot.JOINT_NAMES):
      joint = model.joint(name)
      limit = 35.55 if "calf" in name else 23.7
      aid = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id).item()
      np.testing.assert_allclose(model.actuator_forcerange[aid], [-limit, limit])
      np.testing.assert_allclose(model.key_qpos[0, 7 + index], self.robot.DEFAULT_ANGLES[index])
    np.testing.assert_allclose(model.joint("RL_thigh_joint").range, [-.5236, 4.5379])
    np.testing.assert_allclose(model.joint("FL_thigh_joint").range, [-1.5708, 3.4907])

  def test_default_target_asset_is_unchanged_and_snapshot_has_provenance(self):
    from src.assets.robots.unitree_go2.go2_constants import get_spec

    before = get_spec().compile()
    self.robot.get_stand_spec().compile()
    after = get_spec().compile()
    self.assertEqual(after.nbody, 14)
    np.testing.assert_array_equal(before.body_mass, after.body_mass)
    self.assertAlmostEqual(float(after.body_mass.sum()), 15.206408)
    physics = self.robot.load_source_physics()
    self.assertEqual(physics["source"], "resources/robots/go2/urdf/go2.urdf")
    self.assertEqual(len(physics["sha256"]), 64)
    self.assertIn("snapshot-from", physics["regenerate"])


if __name__ == "__main__":
  unittest.main()
