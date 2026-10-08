"""Physical reset checks against MuJoCo, independent of the training runtime."""
import unittest

import mujoco
import numpy as np
import torch

from src.tasks.pedipulation.constants import DEFAULT_ANGLES, JOINT_NAMES
from src.tasks.pedipulation.robot import get_stand_spec
from src.tasks.transition_t.initialization import ResetSampler


class ResetSamplerTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    spec = get_stand_spec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .1])
    cls.model = spec.compile()
    cls.sampler = ResetSampler(cls.model, JOINT_NAMES)

  def reference(self, batch=8):
    root = torch.zeros(batch, 13)
    root[:, 2] = .42
    root[:, 3] = 1.
    joints = torch.tensor(DEFAULT_ANGLES).repeat(batch, 1)
    feet = self.sampler.forward(root, joints)
    root[:, 2] -= feet[:, :, 2].min(dim=1).values - .022
    feet = self.sampler.forward(root, joints)
    contacts = feet[:, :, 2] < .024
    return root, joints, feet, contacts

  def test_forward_matches_mujoco_for_four_distinct_legs(self):
    root, joints, _, _ = self.reference(1)
    root[0, 3:7] = torch.tensor([.93, .1, -.15, .3])
    root[0, 3:7] /= root[0, 3:7].norm()
    data = mujoco.MjData(self.model)
    data.qpos[:7] = root[0, :7].numpy()
    for index, name in enumerate(JOINT_NAMES):
      data.qpos[self.model.joint(name).qposadr[0]] = joints[0, index]
    mujoco.mj_forward(self.model, data)
    expected = np.stack([data.site_xpos[self.model.site(leg).id]
                         for leg in ("FL", "FR", "RL", "RR")])
    np.testing.assert_allclose(self.sampler.forward(root, joints)[0], expected, atol=1e-6)
    self.assertEqual(len(np.unique(expected.round(5), axis=0)), 4)

  def test_zero_perturbation_preserves_reference_exactly(self):
    args = self.reference()
    args[0][:, 7:] = .13
    result = self.sampler.sample(*args, {}, {}, (0., 0., 0.), 0.)
    for name, expected in zip(("root_state", "joint_pos", "foot_targets"), args):
      self.assertTrue(torch.equal(result[name], expected), name)
    self.assertFalse(result["fallback"].any())
    self.assertFalse(result["disturbed"].any())

  def test_undisturbed_probability_one_is_exact(self):
    args = self.reference()
    result = self.sampler.sample(*args, {"x": (-.02, .02)}, {"x": (-.2, .2)},
                                 undisturbed_probability=1.)
    self.assertTrue(torch.equal(result["root_state"], args[0]))
    self.assertTrue(torch.equal(result["joint_pos"], args[1]))

  def test_randomization_moves_real_feet_and_respects_limits(self):
    torch.manual_seed(7)
    args = self.reference(12)
    original = [value.clone() for value in args]
    pose = dict(x=(-.02, .02), y=(-.02, .02), z=(-.01, .01),
                roll=(-.08, .08), pitch=(-.08, .08), yaw=(-.15, .15))
    velocity = dict(x=(-.2, .2), y=(-.2, .2), z=(-.1, .1),
                    roll=(-.3, .3), pitch=(-.3, .3), yaw=(-.3, .3))
    result = self.sampler.sample(*args, pose, velocity, undisturbed_probability=0.)
    moved = result["disturbed"] & ~result["fallback"]
    self.assertTrue(moved.any(), "all samples fell back to the reference")
    actual = self.sampler.forward(result["root_state"], result["joint_pos"])
    self.assertLess((actual - result["foot_targets"]).norm(dim=-1).max(), .002)
    self.assertGreater((actual[moved] - args[2][moved]).abs().max(), .001)
    self.assertTrue(torch.equal(result["foot_targets"][:, :, 2][args[3]],
                                args[2][:, :, 2][args[3]]))
    self.assertTrue(torch.all(actual[:, :, 2] >= .020))
    self.assertTrue(torch.all(result["joint_pos"] >= self.sampler.lower))
    self.assertTrue(torch.all(result["joint_pos"] <= self.sampler.upper))
    self.assertGreater(result["root_state"][moved, 7:].abs().max(), .01)
    scale = result["scale"]
    self.assertTrue(torch.all((result["root_state"][:, :3]-args[0][:, :3]).abs()
                              <= .020*scale[:, None]+1e-6))
    self.assertTrue(torch.all(result["root_state"][:, 7:10].abs()
                              <= .2*scale[:, None]+1e-6))
    self.assertTrue(torch.all(result["root_state"][:, 10:13].abs()
                              <= .3*scale[:, None]+1e-6))
    foot_delta = result["foot_targets"]-args[2]
    self.assertTrue(torch.all(foot_delta[:, :, :2].norm(dim=-1)
                              <= (2**.5)*.015*scale[:, None]+1e-6))
    self.assertTrue(torch.all(foot_delta[:, :, 2].abs()
                              <= .01*scale[:, None]+1e-6))
    for value, expected in zip(args, original):
      self.assertTrue(torch.equal(value, expected))

  def test_unreachable_target_falls_back_exactly(self):
    args = self.reference()
    result = self.sampler.sample(*args, {"z": (10., 10.)}, {"x": (1., 1.)},
                                 (0., 0., 0.), 0.)
    self.assertTrue(result["fallback"].all())
    self.assertTrue(torch.equal(result["root_state"], args[0]))
    self.assertTrue(torch.equal(result["joint_pos"], args[1]))
    self.assertTrue(torch.equal(result["foot_targets"], args[2]))
    self.assertTrue((result["scale"] == 0.).all())

  def test_nonfinite_input_is_rejected(self):
    args = self.reference()
    args[0][0, 2] = float("nan")
    with self.assertRaises(ValueError):
      self.sampler.sample(*args, {}, {})

  def test_real_ground_collision_is_rejected(self):
    root, joints, _, contacts = self.reference(1)
    self.assertTrue(self.sampler._collision_free(root[0].numpy(), joints[0].numpy(), contacts[0].numpy()))
    root[0, 2] -= .1
    self.assertFalse(self.sampler._collision_free(root[0].numpy(), joints[0].numpy(), contacts[0].numpy()))

  def test_floor_tolerance_applies_only_to_support_feet(self):
    root, joints, _, contacts = self.reference(1)
    self.sampler._collision_free(root[0].numpy(), joints[0].numpy(), contacts[0].numpy())
    radius = self.model.geom("FL_foot_collision").size[0]
    height = self.sampler.data.geom_xpos[self.model.geom("FL_foot_collision").id, 2]
    root[0, 2] += radius-height-.001
    self.assertTrue(self.sampler._collision_free(root[0].numpy(), joints[0].numpy(), contacts[0].numpy()))
    self.assertFalse(self.sampler._collision_free(root[0].numpy(), joints[0].numpy(), np.zeros(4, dtype=bool)))

  def test_floor_height_uses_model_world_transform(self):
    spec = get_stand_spec()
    floor_body = spec.worldbody.add_body(name="floor_body", pos=[0., 0., .12])
    floor_body.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                        pos=[0., 0., .03], size=[0., 0., .1])
    model = spec.compile()
    sampler = ResetSampler(model, JOINT_NAMES)
    self.assertAlmostEqual(sampler.floor_height, .15)
    root, joints, _, contacts = self.reference(1)
    root[:, 2] += .15
    self.assertTrue(sampler._collision_free(root[0].numpy(), joints[0].numpy(), contacts[0].numpy()))

  def test_subset_local_reset_does_not_mutate_other_rows(self):
    args = self.reference(8)
    snapshots = [value.clone() for value in args]
    result = self.sampler.sample(*(value[2:5] for value in args), {}, {"x": (.1, .1)},
                                 (0., 0., 0.), 0.)
    self.assertEqual(result["root_state"].shape, (3, 13))
    self.assertTrue(result["disturbed"].all())
    torch.testing.assert_close(result["root_state"][:, 7], torch.full((3,), .1))
    for value, before in zip(args, snapshots):
      self.assertTrue(torch.equal(value, before))

  def test_nearby_world_obstacle_is_not_treated_as_supporting_floor(self):
    root, joints, _, contacts = self.reference(1)
    data = mujoco.MjData(self.model)
    data.qpos[:7] = root[0, :7].numpy()
    data.qpos[self.sampler.qpos_addresses] = joints[0].numpy()
    mujoco.mj_forward(self.model, data)
    foot_geom = self.model.geom("FL_foot_collision").id
    obstacle_position = data.geom_xpos[foot_geom].copy()
    obstacle_position[0] += self.model.geom_size[foot_geom, 0] + .005 - .0005
    spec = get_stand_spec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .1])
    spec.worldbody.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=obstacle_position, size=[.005, .005, .005])
    model = spec.compile()
    sampler = ResetSampler(model, JOINT_NAMES)
    self.assertFalse(sampler._collision_free(root[0].numpy(), joints[0].numpy(),
                                             contacts[0].numpy()))

  def test_foot_offsets_follow_reference_heading(self):
    torch.manual_seed(10)
    root, joints, _, contacts = self.reference()
    root[:, 3], root[:, 6] = 2**-.5, 2**-.5
    feet = self.sampler.forward(root, joints)
    result = self.sampler.sample(root, joints, feet, contacts, {}, {}, (.01, 0., 0.), 0.)
    self.assertTrue(result["disturbed"].any())
    delta = result["foot_targets"]-feet
    self.assertLess(delta[:, :, 0].abs().max(), 1e-6)
    self.assertGreater(delta[:, :, 1].abs().max(), .001)


if __name__ == "__main__":
  unittest.main()
