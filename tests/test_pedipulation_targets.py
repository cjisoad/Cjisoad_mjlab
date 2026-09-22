"""FR target command geometry, gating and policy contracts."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import test_pedipulation as existing


class FrontTargetTests(unittest.TestCase):
  def make_env(self):
    env, term = existing.PedipulationTaskTests().make_env()
    env.scene['robot'].data.root_link_pos_w = torch.tensor([[1., 2., .52], [-3., 1., .52]])
    return env, term

  def test_six_commands_and_independent_target_sampling(self):
    env, term = self.make_env()
    self.assertEqual(term.command.shape, (2, 6))
    term.cfg = replace(term.cfg, zero_probability=1., xy_zero_probability=1., foot_target_probability=1.)
    term._resample_command(torch.tensor([0, 1]))
    self.assertTrue((term.command[:, :2] == 0).all())
    self.assertTrue(term.has_foot_target.all())
    self.assertTrue((term.command[:, 3:].norm(dim=-1) >= term.cfg.foot_target_min_offset - 1e-6).all())
    saved = term.command[1].clone()
    term.cfg = replace(term.cfg, foot_target_probability=0.)
    term.reset(torch.tensor([0]))
    self.assertFalse(term.has_foot_target[0])
    torch.testing.assert_close(term.command[0, 3:], torch.zeros(3))
    torch.testing.assert_close(term.command[1], saved)

  def test_zero_point_is_upright_desired_fk_and_does_not_sink_with_tilt(self):
    env, term = self.make_env()
    self.assertTrue(hasattr(term, 'anchor_frame'))
    robot = env.scene['robot'].data
    rotation = torch.tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    zero_anchor = rotation @ torch.as_tensor(term.foot_workspace.zero_body, dtype=torch.float32)
    torch.testing.assert_close(term.foot_target_pos_anchor, zero_anchor.expand(2, -1))
    expected = robot.root_link_pos_w + (term.anchor_frame.basis_w @ zero_anchor[:, None]).squeeze(-1)
    torch.testing.assert_close(term.foot_target_pos_w, expected)
    term._command[:, 3:] = torch.tensor([.02, -.03, .04])
    delta_w = (term.anchor_frame.basis_w @ term.command[:, 3:, None]).squeeze(-1)
    torch.testing.assert_close(term.foot_target_pos_w, expected + delta_w)
    local = term.anchor_frame.to_frame(term.foot_target_pos_w - robot.root_link_pos_w)
    torch.testing.assert_close(local, term.foot_target_pos_anchor)
    for pitch in (0., -1.3, -torch.pi / 2):
      robot.root_link_quat_w[:] = existing.quats([.3], [pitch], [.4])
      env.common_step_counter += 1
      torch.testing.assert_close(term.foot_target_pos_anchor, zero_anchor + term.command[:, 3:])
      self.assertTrue((term.foot_target_pos_anchor[:, 2] >= term.foot_workspace.min_height).all())

  def test_workspace_targets_have_fk_witness_and_clear_torso(self):
    import mujoco
    from src.tasks.pedipulation.robot import get_stand_spec
    from src.tasks.pedipulation.constants import DESIRED_ANGLES, JOINT_NAMES
    env, term = self.make_env()
    self.assertTrue(hasattr(term, 'foot_workspace'))
    bank = term.foot_workspace
    self.assertGreater(len(bank.offsets), 100)
    m = get_stand_spec().compile()
    d = mujoco.MjData(m)
    addresses = [int(m.joint(name).qposadr[0]) for name in JOINT_NAMES]
    d.qpos[addresses] = DESIRED_ANGLES
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    zero = rotation @ bank.zero_body
    targets = bank.offsets + zero
    self.assertTrue((targets[:, 2] >= bank.min_height - 1e-7).all())
    self.assertGreater(bank.min_height, .057)
    for index in np.linspace(0, len(bank.offsets) - 1, 100, dtype=int):
      d.qpos[addresses[3:6]] = bank.joint_positions[index]
      mujoco.mj_kinematics(m, d)
      foot = d.site_xpos[m.site('FR').id] - d.xpos[m.body('base_link').id]
      np.testing.assert_allclose(rotation @ foot, targets[index], atol=1e-7)
      np.testing.assert_array_equal(d.qpos[addresses[6:]], DESIRED_ANGLES[6:])
      joints = [m.joint(name).id for name in JOINT_NAMES[3:6]]
      self.assertTrue((bank.joint_positions[index] > m.jnt_range[joints, 0]).all())
      self.assertTrue((bank.joint_positions[index] < m.jnt_range[joints, 1]).all())

  def reward_env(self, gate):
    state = SimpleNamespace(height_score=torch.tensor(gate),
      joint_pos=torch.zeros(4, 12), desired_angles=torch.zeros(12))
    state.joint_pos[1, 3:6] = .1
    state.joint_pos[2:, 3:6] = 1.
    sites = torch.zeros(4, 2, 3)
    sites[:, 0, 2] = .8
    sites[:, 1, 2] = .6
    target = sites[:, 1].clone()
    target[3, 0] = .05
    term = SimpleNamespace(has_foot_target=torch.tensor([False, False, True, True]),
      foot_target_pos_w=target, cfg=SimpleNamespace(foot_target_std=.05))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(site_pos_w=sites))},
      command_manager=SimpleNamespace(get_term=lambda _: term),
      action_manager=SimpleNamespace(get_term=lambda _: state))
    return env

  def test_reward_switch_and_height_gate(self):
    from src.tasks.pedipulation.mdp import rewards
    asset = SimpleNamespace(name='robot', site_ids=[1])
    env = self.reward_env(.8)
    expected = torch.tensor([1., np.exp(-.3), 1., np.exp(-1.)], dtype=torch.float32)
    actual = rewards.default_pos_reward_FR(env, asset_cfg=asset)
    torch.testing.assert_close(actual, expected)
    env.action_manager.get_term('').height_score.fill_(.7)
    torch.testing.assert_close(rewards.default_pos_reward_FR(env, asset_cfg=asset), torch.zeros(4))

  def test_height_difference_penalty_truth_table(self):
    from src.tasks.pedipulation.mdp.rewards import feet_height_symmetry
    env = self.reward_env(.8)
    asset = SimpleNamespace(name='robot', site_ids=[0, 1])
    torch.testing.assert_close(feet_height_symmetry(env, asset), torch.tensor([.2, .2, 0., 0.]))
    env.action_manager.get_term('').height_score.fill_(.7)
    torch.testing.assert_close(feet_height_symmetry(env, asset), torch.full((4,), .2))

  def test_observation_target_scale_noise_and_action_offsets(self):
    from src.tasks.pedipulation.mdp.observations import policy_state
    env, term = self.make_env()
    self.assertEqual(term.command.shape[-1], 6)
    term._command[:, 3:] = torch.tensor([.02, -.03, .04])
    state = env.action_manager.get_term('joint_pos')
    state.raw_action[:] = torch.arange(12)
    plain = policy_state(env, add_noise=False)
    noisy = policy_state(env, add_noise=True)
    self.assertEqual(plain.shape, (2, 48))
    torch.testing.assert_close(plain[:, 9:12], term.command[:, 3:])
    torch.testing.assert_close(noisy[:, 6:12], plain[:, 6:12])
    torch.testing.assert_close(noisy[:, 36:], state.raw_action)

  def test_active_targets_do_not_impose_left_right_policy_symmetry(self):
    from src.tasks.pedipulation.rl.symmetry import exact_symmetry_loss, mirror_actions, mirror_actor_observations
    actor = torch.nn.Linear(48, 12)
    obs = torch.randn(6, 48)
    obs[:3, 9:12] = 0.
    actual = exact_symmetry_loss(actor, obs)
    idle = obs[:3]
    expected = (actor(idle) - mirror_actions(actor(mirror_actor_observations(idle)))).square().mean()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, tuple(actor.parameters()))
    expected_grad = torch.autograd.grad(expected, tuple(actor.parameters()))
    for got, want in zip(actual_grad, expected_grad):
      torch.testing.assert_close(got, want)
    active_loss = exact_symmetry_loss(actor, obs[3:])
    self.assertEqual(active_loss.item(), 0.)
    active_loss.backward()
    self.assertTrue(all((p.grad == 0).all() for p in actor.parameters()))


if __name__ == '__main__':
  unittest.main()
