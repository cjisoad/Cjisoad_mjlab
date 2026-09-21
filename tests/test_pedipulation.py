"""Gravity-aligned velocity tracking and independent task contracts."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_apply_inverse


def quats(roll, pitch, yaw):
  return quat_from_euler_xyz(*[torch.as_tensor(v, dtype=torch.float32) for v in (roll, pitch, yaw)])


def frame_type():
  path = Path(__file__).resolve().parents[1] / 'src/tasks/pedipulation/mdp/frame.py'
  if not path.exists():
    raise AssertionError('Pedipulation horizontal frame has not been implemented')
  from src.tasks.pedipulation.mdp.frame import HeadingFrame
  return HeadingFrame


class PedipulationFrameTests(unittest.TestCase):
  def make_frame(self, n=1):
    return frame_type()(torch.tensor([[0., 0., -1.]]).repeat(n, 1))

  def settle(self, frame, quat):
    for step in range(40):
      frame.update(quat, .02, step)

  def test_frame_is_horizontal_right_handed_and_follows_minus_body_z(self):
    frame = self.make_frame(3)
    quat = quats([0., .25, -.3], [-1.57, -1.0, -1.2], [0., .8, -1.])
    self.settle(frame, quat)
    basis = frame.basis_w
    torch.testing.assert_close(basis.transpose(1, 2) @ basis, torch.eye(3).repeat(3, 1, 1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(basis), torch.ones(3))
    from mjlab.utils.lab_api.math import quat_apply
    projected = quat_apply(quat, torch.tensor([[0., 0., -1.]]).repeat(3, 1))
    projected[:, 2] = 0
    projected /= projected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(basis[:, :, 0], projected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(basis[:, :, 2], torch.tensor([[0., 0., 1.]]).repeat(3, 1))

  def test_horizontal_velocity_is_independent_of_pitch(self):
    frame = self.make_frame(3)
    self.settle(frame, quats([0., 0., 0.], [-.4, -1., -1.57], [0., 0., 0.]))
    expected = torch.tensor([[.3, -.1, .7]]).repeat(3, 1)
    torch.testing.assert_close(frame.to_frame(expected), expected)
    torch.testing.assert_close(frame.to_frame(torch.tensor([[0., 0., 1.]]).repeat(3, 1))[:, :2], torch.zeros(3, 2))

  def test_yaw_follows_robot_and_velocity_transform_round_trips(self):
    frame = self.make_frame()
    self.settle(frame, quats([0.], [-torch.pi / 2], [torch.pi / 2]))
    torch.testing.assert_close(frame.to_frame(torch.tensor([[0., .3, 0.]])), torch.tensor([[.3, 0., 0.]]), atol=1e-6, rtol=1e-6)
    velocity = torch.tensor([[.2, -.4, .6]])
    world = (frame.basis_w @ velocity.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(frame.to_frame(world), velocity)

  def test_singular_initialization_hysteresis_and_rate_limit(self):
    frame = self.make_frame()
    frame.update(quats([0.], [0.], [.7]), .02, 0)
    torch.testing.assert_close(frame.heading, torch.tensor([.7]))
    self.assertFalse(frame.valid.item())
    frame.update(quats([0.], [-.1], [1.2]), .02, 1)
    torch.testing.assert_close(frame.heading, torch.tensor([.7]))
    frame.update(quats([0.], [-.3], [1.2]), .02, 2)
    self.assertTrue(frame.valid.item())
    self.assertLessEqual(abs(frame.heading.item() - .7), .12001)
    saved = frame.basis_w.clone()
    for _ in range(5):
      frame.update(quats([0.], [-.3], [1.2]), .02, 2)
    torch.testing.assert_close(frame.basis_w, saved)
    frame.update(quats([0.], [-.1], [1.2]), .02, 3)
    self.assertTrue(frame.valid.item())
    previous = frame.basis_w.clone()
    frame.update(quats([0.], [-.01], [-2.]), .02, 4)
    self.assertFalse(frame.valid.item())
    torch.testing.assert_close(frame.basis_w, previous)

  def test_same_step_is_idempotent_but_refreshes_pose_and_partial_reset(self):
    frame = self.make_frame(2)
    self.settle(frame, quats([0., 0.], [-1., -1.], [.5, -.5]))
    pose = quats([0., 0.], [-1., -1.], [.6, -.6])
    frame.update(pose, .02, 40)
    saved = frame.basis_w.clone()
    frame.update(pose, .02, 40)
    torch.testing.assert_close(saved, frame.basis_w)
    frame.update(quats([0., 0.], [-1., -1.], [.65, -.65]), .02, 40)
    torch.testing.assert_close(frame.heading, torch.tensor([.65, -.65]))
    frame.reset(torch.tensor([0]))
    frame.update(quats([0., 0.], [0., -1.], [1., -.65]), .02, 40)
    torch.testing.assert_close(frame.heading, torch.tensor([1., -.65]))
    self.assertEqual(frame.valid.tolist(), [False, True])

  def test_arbitrary_gravity(self):
    frame = frame_type()(torch.tensor([[0., -1., 0.]]))
    self.settle(frame, quats([0.], [0.], [0.]))
    torch.testing.assert_close(frame.basis_w[:, :, 2], torch.tensor([[0., 1., 0.]]))
    torch.testing.assert_close(frame.to_frame(torch.tensor([[0., 2., 0.]])), torch.tensor([[0., 0., 2.]]))

  def test_reacquisition_takes_short_path_across_heading_wrap(self):
    frame = self.make_frame()
    frame.update(quats([0.], [0.], [3.1]), .02, 0)
    frame.update(quats([0.], [-1.], [-3.1]), .02, 1)
    torch.testing.assert_close(frame.heading, torch.tensor([-3.1]))
    self.assertFalse(frame.aligning.item())

  def test_frame_direction_reflects_as_polar_vector(self):
    first, reflected = self.make_frame(), self.make_frame()
    pose = quats([.25], [-1.2], [.8])
    mirror_pose = quats([-.25], [-1.2], [-.8])
    self.settle(first, pose)
    self.settle(reflected, mirror_pose)
    direction = quat_apply_inverse(pose, first.forward)
    mirror_direction = quat_apply_inverse(mirror_pose, reflected.forward)
    torch.testing.assert_close(mirror_direction, direction * torch.tensor([1., -1., 1.]))


class PedipulationTaskTests(unittest.TestCase):
  def test_task_registered_independently(self):
    import src.tasks
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
    self.assertIn('Pedipulation', list_tasks())
    self.assertIn('Unitree-Go2-RearStand', list_tasks())
    cfg = load_env_cfg('Pedipulation')
    self.assertEqual(len(cfg.rewards), 23)
    self.assertEqual(load_rl_cfg('Pedipulation').experiment_name, 'pedipulation')
    self.assertEqual(load_rl_cfg('Unitree-Go2-RearStand').experiment_name, 'go2_stand')
    self.assertIn('pedipulation', cfg.commands['stand'].__class__.__module__)

  def make_env(self):
    frame_type()
    from src.tasks.pedipulation.mdp.commands import PedipulationCommand, PedipulationCommandCfg
    quat = quats([0., 0.], [-.6, -1.3], [0., 0.])
    data = SimpleNamespace(root_link_quat_w=quat,
      gravity_vec_w=torch.tensor([[0., 0., -1.]]).repeat(2, 1),
      root_link_lin_vel_w=torch.tensor([[.3, 0., .2]]).repeat(2, 1),
      root_link_ang_vel_w=torch.tensor([[0., 0., .4]]).repeat(2, 1),
      root_link_ang_vel_b=quat_apply_inverse(quat, torch.tensor([[0., 0., .4]]).repeat(2, 1)),
      projected_gravity_b=quat_apply_inverse(quat, torch.tensor([[0., 0., -1.]]).repeat(2, 1)))
    state = SimpleNamespace(height_score=torch.tensor(.8), joint_pos=torch.zeros(2, 12),
      default_angles=torch.zeros(2, 12), joint_vel=torch.zeros(2, 12), raw_action=torch.zeros(2, 12))
    env = SimpleNamespace(num_envs=2, device='cpu', common_step_counter=1,
      episode_length_buf=torch.ones(2, dtype=torch.long), step_dt=.02,
      scene={'robot': SimpleNamespace(data=data)}, action_manager=SimpleNamespace(get_term=lambda _: state))
    term = PedipulationCommand(PedipulationCommandCfg(resampling_time_range=(10., 10.)), env)
    env.command_manager = SimpleNamespace(get_term=lambda _: term, get_command=lambda _: term.command)
    term._command[:, 0] = .3
    term.heading_target[:] = .8
    return env, term

  def test_rewards_observations_and_heading_share_frame(self):
    env, term = self.make_env()
    from src.tasks.pedipulation.mdp import rewards, observations
    torch.testing.assert_close(rewards.tracking_lin_vel(env), torch.ones(2))
    torch.testing.assert_close(rewards.tracking_ang_vel(env), torch.ones(2))
    torch.testing.assert_close(rewards.lin_vel_z(env), torch.full((2,), torch.exp(torch.tensor(-2.)).item()))
    torch.testing.assert_close(rewards.ang_vel_xy(env), torch.ones(2))
    obs = observations.policy_state(env, add_noise=False)
    self.assertEqual(obs.shape, (2, 45))
    torch.testing.assert_close(obs[:, 33:], env.action_manager.get_term('joint_pos').raw_action)
    torch.testing.assert_close(observations.base_velocity(env), torch.tensor([[.6, 0., .4]]).repeat(2, 1))

  def test_mirror_matches_original_45_dimensional_layout(self):
    frame_type()
    from src.tasks.pedipulation.rl.symmetry import mirror_actor_observations
    from src.tasks.stand.rl.symmetry import mirror_actor_observations as original_mirror
    obs = torch.randn(8, 45)
    reflected = mirror_actor_observations(obs)
    torch.testing.assert_close(reflected, original_mirror(obs))
    torch.testing.assert_close(mirror_actor_observations(reflected), obs)

  def test_command_resampling_once_and_reset_uses_fresh_pose(self):
    env, term = self.make_env()
    env.episode_length_buf[:] = torch.tensor([499, 500])
    first = term.command.clone()
    self.assertEqual(term.command_counter.tolist(), [0, 1])
    term.compute(.02)
    torch.testing.assert_close(term.command, first)
    self.assertEqual(term.command_counter.tolist(), [0, 1])
    # reset() is called before MuJoCo forward: it must not use the old pose.
    term.reset(torch.tensor([0]))
    env.episode_length_buf[0] = 0
    env.scene['robot'].data.root_link_quat_w[0] = quats([0.], [0.], [1.1])[0]
    other = term.command_counter[1].clone()
    torch.testing.assert_close(term.frame.heading[0], torch.tensor(1.1))
    term.compute(.02)
    self.assertEqual(term.command_counter[0].item(), 1)
    torch.testing.assert_close(term.command_counter[1], other)

  def test_mirror_loss_backpropagates_through_45_dimensional_observations(self):
    frame_type()
    from src.tasks.pedipulation.rl.symmetry import exact_symmetry_loss, mirror_actions, mirror_actor_observations
    torch.manual_seed(42)
    actor = torch.nn.Linear(45, 12)
    obs = torch.randn(16, 45)
    actual = exact_symmetry_loss(actor, obs)
    gradients = torch.autograd.grad(actual, tuple(actor.parameters()))
    expected = (actor(obs) - mirror_actions(actor(mirror_actor_observations(obs)))).square().mean()
    expected_gradients = torch.autograd.grad(expected, tuple(actor.parameters()))
    for got, want in zip(gradients, expected_gradients):
      torch.testing.assert_close(got, want)
    self.assertGreater(gradients[0].abs().sum().item(), 0.)


if __name__ == '__main__':
  unittest.main()
