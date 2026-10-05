"""Geometry and coordinate consumers for the teachers' approved body-Y frame."""
import math
import unittest

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz

from src.tasks.teacher_common.commands import anchor_basis
from src.tasks.teacher_common.rl import TeacherOnPolicyRunner
from src.tasks.leg_manip.mdp.anchor_frame import anchor_basis as old_basis
from src.tasks.loco_pedipulation.mdp import rewards as loco_rewards
from src.tasks.teacher_common.rewards import pedipulation_reward
from test_teacher_tasks import TASKS, command_fixture


def quaternion(roll=0., pitch=0., yaw=0., count=4):
  return quat_from_euler_xyz(*(torch.full((count,), float(value)) for value in (roll, pitch, yaw)))


class TeacherAnchorGeometryTests(unittest.TestCase):
  def test_level_quad_roll_does_not_change_heading(self):
    gravity = torch.tensor((0., 0., -1.)).expand(4, 3)
    for roll in (-.7, -.2, .2, .7):
      actual = anchor_basis(quaternion(roll=roll, yaw=.6), gravity)
      expected = torch.tensor((math.cos(.6), math.sin(.6), 0.)).expand(4, 3)
      torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-6, rtol=0.)

  def test_rear_up_sweep_and_nose_down_keep_forward(self):
    gravity = torch.tensor((0., 0., -1.)).expand(4, 3)
    for pitch in (0., -.3, -.8, -math.pi/2., math.pi/4.):
      actual = anchor_basis(quaternion(pitch=pitch, yaw=.6), gravity)
      expected = torch.tensor((math.cos(.6), math.sin(.6), 0.)).expand(4, 3)
      torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-6, rtol=0.)

  def test_full_biped_matches_minus_body_z(self):
    gravity = torch.tensor((0., 0., -1.)).expand(4, 3)
    q = quaternion(roll=.25, pitch=-math.pi/2., yaw=.6)
    expected = quat_apply(q, torch.tensor((0., 0., -1.)).expand(4, 3))
    expected[:, 2] = 0.
    expected = torch.nn.functional.normalize(expected, dim=-1)
    torch.testing.assert_close(anchor_basis(q, gravity)[:, :, 0], expected, atol=1e-6, rtol=0.)

  def test_tilted_gravity_is_orthonormal_and_uses_body_y(self):
    gravity = torch.tensor((.1, -.2, -1.)).expand(4, 3)
    q = quaternion(roll=.25, pitch=-.8, yaw=.6)
    up = -torch.nn.functional.normalize(gravity, dim=-1)
    body_y = quat_apply(q, torch.tensor((0., 1., 0.)).expand(4, 3))
    expected = torch.nn.functional.normalize(torch.cross(body_y, up, dim=-1), dim=-1)
    actual = anchor_basis(q, gravity)
    torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-6, rtol=0.)
    torch.testing.assert_close(actual[:, :, 2], up)
    torch.testing.assert_close(actual.transpose(-1, -2) @ actual, torch.eye(3).expand(4, 3, 3), atol=1e-6, rtol=0.)
    torch.testing.assert_close(torch.linalg.det(actual), torch.ones(4), atol=1e-6, rtol=0.)

  def test_body_y_vertical_has_finite_deterministic_fallback(self):
    q = quaternion(roll=math.pi/2.)
    gravity = torch.tensor((0., 0., -1.)).expand(4, 3)
    actual = anchor_basis(q, gravity)
    self.assertTrue(torch.isfinite(actual).all())
    torch.testing.assert_close(actual[:, :, 0], torch.tensor((1., 0., 0.)).expand(4, 3))
    from src.tasks.teacher_common.anchor_frame import anchor_degenerate
    self.assertTrue(anchor_degenerate(q, gravity).all())


class TeacherAnchorConsumerTests(unittest.TestCase):
  def test_both_command_classes_use_new_frame_and_decode_identical_goals(self):
    goals = []
    for task in TASKS:
      env, term = command_fixture(task, count=4)
      data = env.scene['robot'].data
      data.root_link_quat_w[:] = quaternion(roll=.3, pitch=-.5, yaw=.6)
      term.set_command(torch.arange(4), velocity=(.2, -.03, .1), target_offset=(.04, .01, .30))
      expected = anchor_basis(data.root_link_quat_w, data.gravity_vec_w)
      torch.testing.assert_close(term.basis_w, expected)
      goals.append(term.foot_target_pos_w)
    torch.testing.assert_close(goals[0], goals[1], atol=1e-6, rtol=0.)

  def test_horizontal_tracking_and_foot_reward_use_new_basis(self):
    for task in TASKS:
      env, term = command_fixture(task, count=4)
      data = env.scene['robot'].data
      data.root_link_quat_w[:] = quaternion(roll=.3)
      term.set_command(torch.arange(4), velocity=(.2, 0., .1), target_offset=(.04, .01, .30))
      # Loco commands normally ramp requested velocity; isolate the frame with
      # one fixed, already-ramped command for this reward comparison.
      term._command[:, :3] = torch.tensor((.2, 0., .1))
      data.root_link_lin_vel_w[:] = torch.tensor((.2, 0., 0.))
      action = env.action_manager.get_term('joint_pos')
      action.height_score = torch.tensor(1.)
      if task == 'pedipulation_t':
        torch.testing.assert_close(pedipulation_reward(env, 'tracking_lin_vel'), torch.ones(4))
        data.site_pos_w[:, 1] = term.foot_target_pos_w
        from types import SimpleNamespace
        torch.testing.assert_close(pedipulation_reward(env, 'FR_pos_track',
          asset_cfg=SimpleNamespace(name='robot', site_ids=[1])), torch.ones(4))
      else:
        term.blend.fill_(1.)
        torch.testing.assert_close(loco_rewards.track_linear_velocity(env), torch.full((4,), 2.))
        term.reference[:] = term.zero + term.command[:, 3:]
        data.site_pos_w[:, 1] = term.reference_w
        torch.testing.assert_close(loco_rewards.fr_position_tracking(env), torch.ones(4))

  def test_biped_heading_feedback_does_not_turn_for_pure_quad_roll(self):
    env, term = command_fixture('pedipulation_t', count=4)
    env.scene['robot'].data.root_link_quat_w[:] = quaternion(roll=.3)
    term.manual.zero_()
    term.heading_target.zero_()
    torch.testing.assert_close(term.command[:, 2], torch.zeros(4), atol=1e-6, rtol=0.)

  def test_yaw_angular_reward_is_unchanged_for_fixed_command(self):
    env, term = command_fixture('loco_pedipulation_t', count=4)
    data = env.scene['robot'].data
    data.root_link_quat_w[:] = quaternion(roll=.3, pitch=-.5, yaw=.6)
    data.root_link_ang_vel_w[:] = torch.tensor((.3, -.2, .1))
    term.set_command(torch.arange(4), velocity=(.2, 0., .15))
    new_reward = loco_rewards.track_angular_velocity(env)
    from src.tasks.teacher_common.commands import LocoPedipulationTeacherCommand
    from unittest.mock import PropertyMock, patch
    with patch.object(LocoPedipulationTeacherCommand, 'basis_w', new_callable=PropertyMock,
                      return_value=old_basis(data.root_link_quat_w, data.gravity_vec_w)):
      old_reward = loco_rewards.track_angular_velocity(env)
    torch.testing.assert_close(new_reward, old_reward, atol=1e-6, rtol=0.)

  def test_checkpoint_contract_identifies_new_anchor(self):
    from types import SimpleNamespace
    env, term = command_fixture('loco_pedipulation_t', count=4)
    action = env.action_manager.get_term('joint_pos')
    action.cfg = SimpleNamespace(scale=.25, delay=True)
    action._entity = SimpleNamespace(joint_names=[f'joint_{i}' for i in range(12)])
    action.joint_ids = torch.arange(12)
    wrapped = SimpleNamespace(unwrapped=env, clip_actions=10.)
    stub = SimpleNamespace(env=wrapped, teacher_term=term, teacher_task='loco_pedipulation_t',
                           alg=SimpleNamespace(critic=SimpleNamespace(obs_dim=95)))
    contract = TeacherOnPolicyRunner._contract(stub)
    self.assertEqual(contract['version'], 2)
    self.assertEqual(contract['anchor'], 'teacher_body_y_cross_up_v1')

  def test_old_anchor_checkpoint_is_rejected_before_loading_weights(self):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    stub = SimpleNamespace(_contract=lambda: {'version': 2, 'anchor': 'teacher_body_y_cross_up_v1'})
    with TemporaryDirectory() as directory:
      path = Path(directory) / 'old.pt'
      torch.save({'infos': {'teacher_contract': {'version': 1, 'anchor': 'leg_manip_body_x_minus_z'}}}, path)
      with self.assertRaisesRegex(ValueError, 'Checkpoint is not an aligned teacher'):
        TeacherOnPolicyRunner.load(stub, str(path))

  def test_reflected_pose_and_velocity_keep_existing_mirror_signs(self):
    from src.tasks.teacher_common.anchor_frame import to_anchor
    gravity = torch.tensor((0., 0., -1.)).expand(4, 3)
    world_velocity = torch.tensor((.3, -.2, .1)).expand(4, 3)
    reflect = torch.tensor((1., -1., 1.))
    for pitch in (0., -.5, -math.pi/2.):
      basis = anchor_basis(quaternion(roll=.25, pitch=pitch, yaw=.6), gravity)
      mirrored = anchor_basis(quaternion(roll=-.25, pitch=pitch, yaw=-.6), gravity)
      velocity = to_anchor(basis, world_velocity)
      torch.testing.assert_close(to_anchor(mirrored, world_velocity * reflect),
        velocity * reflect, atol=1e-6, rtol=0.)
      angular = to_anchor(basis, world_velocity)
      torch.testing.assert_close(to_anchor(mirrored, -world_velocity * reflect),
        angular * torch.tensor((-1., 1., -1.)), atol=1e-6, rtol=0.)


if __name__ == '__main__':
  unittest.main()
