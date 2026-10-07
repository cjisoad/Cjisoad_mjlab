"""Reachable LOW-to-HIGH bridge target pairs and reproducible sampling."""
import unittest

import numpy as np
import torch


class BridgePathTests(unittest.TestCase):
  def _paths(self):
    try:
      from src.tasks.pedipulation_bridge_t.paths import BridgeTargetPaths
    except ModuleNotFoundError:
      self.fail('BridgeTargetPaths must provide audited LOW-to-HIGH target pairs')
    return BridgeTargetPaths(device='cpu')

  def test_paths_use_teacher_banks_and_monotonic_requested_height_bands(self):
    from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace, LOW, HIGH
    paths = self._paths()
    workspace = build_foot_workspace()
    starts, ends = paths.starts.numpy(), paths.ends.numpy()
    self.assertGreaterEqual(len(starts), 4)
    self.assertGreaterEqual(len(np.unique(starts.round(6), axis=0)), 4)
    self.assertGreaterEqual(len(np.unique(ends[:, 2].round(6))), 2)
    self.assertTrue(((starts[:, 2] >= .05) & (starts[:, 2] <= .2)).all())
    self.assertTrue(((ends[:, 2] >= .30) & (ends[:, 2] <= .33)).all())
    for targets, segment in ((starts, LOW),):
      distance = np.linalg.norm(targets[:, None]-workspace.offsets[workspace.segment == segment], axis=-1)
      self.assertLess(distance.min(axis=1).max(), 1e-6)
    curve = starts[:, None]+np.linspace(0., 1., 101)[None, :, None]*(ends-starts)[:, None]
    self.assertTrue((np.diff(curve[:, :, 2], axis=1) > 0).all())
    self.assertFalse(paths.report['dynamic_tracking_validated'])
    self.assertGreaterEqual(len(paths.data['h_start_values']), 13)
    self.assertEqual(paths.report['ground_or_self_penetrations'], [])
    self.assertLess(paths.report['max_ik_error_m'], .004)
    self.assertLess(paths.report['max_static_base_residual'], .005)
    self.assertLess(paths.report['max_static_torque_ratio'], 1.)
    self.assertLess(paths.report.get('max_final_rear_hip_abs_rad', np.inf), .1)
    self.assertGreater(paths.report.get('min_final_knee_spacing_m', 0.), .26)

  def test_sampler_is_reproducible_and_returns_only_audited_pairs(self):
    paths = self._paths()
    a = paths.sample(64, generator=torch.Generator().manual_seed(13))
    b = paths.sample(64, generator=torch.Generator().manual_seed(13))
    for first, second in zip(a, b):
      self.assertEqual(first.shape, (64, 3))
      torch.testing.assert_close(first, second)
    candidates = torch.cat((paths.starts, paths.ends), dim=-1)
    sampled = torch.cat(a, dim=-1)
    self.assertTrue(((sampled[:, None]-candidates).abs().amax(-1).amin(-1) < 1e-6).all())

  def test_pose_witnesses_reconstruct_targets_and_planted_rear_support(self):
    import mujoco
    from src.tasks.pedipulation_transition_t.motion import make_model, MotionReference, NOMINAL_MOTION_PATH
    from src.tasks.pedipulation.constants import DESIRED_ANGLES, JOINT_NAMES
    paths = self._paths()
    model, data = make_model(), None
    data = mujoco.MjData(model)
    reference = MotionReference(NOMINAL_MOTION_PATH)
    fr = model.site('FR').id
    rear = [model.site(name).id for name in ('RL', 'RR')]
    expected_rear = reference.data['foot_pos_w'][0, 2:]
    # Independently forward the saved full-pose witnesses for every path and
    # threshold; use a frame stride to keep this regression fast.
    poses = paths.data['qpos_witnesses'][:, :, ::8].reshape(-1, model.nq)
    targets = paths.data['offset_witnesses'][:, :, ::8].reshape(-1, 3)
    for pose, target in zip(poses, targets):
      data.qpos[:] = pose
      mujoco.mj_forward(model, data)
      np.testing.assert_allclose(data.site_xpos[fr]-data.qpos[:3]-paths.data['zero'], target, atol=.001)
      np.testing.assert_allclose(data.site_xpos[rear], expected_rear, atol=.001)
    # Endpoint perturbations must have real HIGH-teacher FK witnesses even
    # when they are between the original discrete workspace heights.
    addresses = [int(model.joint(n).qposadr[0]) for n in JOINT_NAMES]
    for target, witness in zip(paths.data['ends'], paths.data['endpoint_joint_witnesses']):
      data.qpos[:7] = [0., 0., .5, 2**-.5, 0., -2**-.5, 0.]
      data.qpos[addresses] = DESIRED_ANGLES
      data.qpos[np.array(addresses)[3:6]] = witness
      mujoco.mj_forward(model, data)
      np.testing.assert_allclose(data.site_xpos[fr]-data.qpos[:3]-paths.data['zero'], target, atol=1e-4)


if __name__ == '__main__':
  unittest.main()
