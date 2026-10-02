"""Anchor velocity agreement across stances, symmetry and safe warm-start."""

import copy
import math
from types import SimpleNamespace
import unittest

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from src.tasks.leg_manip.mdp.anchor_frame import anchor_basis
from src.tasks.leg_manip.mdp import observations


def make_env(count=128):
  pitch = -torch.linspace(0., math.pi / 2, count)
  yaw = torch.linspace(-math.pi, math.pi, count)
  quat = quat_from_euler_xyz(torch.zeros(count), pitch, yaw)
  frame = anchor_basis(quat, torch.tensor((0., 0., -1.)).expand(count, 3))
  expected = torch.tensor((.2, -.07, .03)).expand(count, 3)
  data = SimpleNamespace(
    root_link_lin_vel_w=(frame @ expected[..., None]).squeeze(-1),
    root_link_ang_vel_b=torch.zeros(count, 3), projected_gravity_b=torch.zeros(count, 3),
    joint_pos=torch.zeros(count, 12), default_joint_pos=torch.zeros(count, 12),
    joint_vel=torch.zeros(count, 12))
  action = SimpleNamespace(raw_action=torch.zeros(count, 12), actor_sample=None)
  term = SimpleNamespace(basis_w=frame, mode=torch.zeros(count, dtype=torch.bool),
                         group_weight=torch.zeros(count))
  command = torch.tensor((.3, .05, .1, 0., 0., 0.)).expand(count, 6)
  env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)},
    action_manager=SimpleNamespace(get_term=lambda _: action),
    command_manager=SimpleNamespace(get_term=lambda _: term, get_command=lambda _: command))
  return env, expected, term


class VelocityObservationTests(unittest.TestCase):
  def test_exact_velocity_has_same_scale_and_axes_in_every_rear_up_pose(self):
    env, expected, _ = make_env()
    actor = observations.policy_state(env, add_noise=False)
    self.assertEqual(actor.shape, (128, 51))
    torch.testing.assert_close(actor[:, 48:51], expected * 2)
    torch.testing.assert_close(observations.base_velocity(env), actor[:, 48:51])
    shared = observations.shared_policy_state(env)
    self.assertEqual(shared.shape, (128, 48))
    torch.testing.assert_close(shared, actor[:, :48])

  def test_actor_noise_is_small_additive_and_critic_remains_exact(self):
    env, expected, _ = make_env(2048)
    actor = observations.policy_state(env, add_noise=True)
    self.assertEqual(actor.shape[-1], 51)
    critic_velocity = observations.base_velocity(env)
    torch.testing.assert_close(critic_velocity, expected * 2)
    error = (actor[:, 48:51] - critic_velocity) / 2
    self.assertLessEqual(float(error.abs().max()), .020001)
    self.assertGreater(float(error.std()), .009)
    torch.testing.assert_close(observations.shared_policy_state(env), actor[:, :48])
    # Noise remains present at zero speed; precise critic does not change.
    env.scene['robot'].data.root_link_lin_vel_w.zero_()
    actor = observations.policy_state(env, add_noise=True)
    self.assertGreater(float(actor[:, 48:51].abs().sum()), 0.)
    torch.testing.assert_close(observations.base_velocity(env), torch.zeros(2048, 3))

  def test_velocity_is_not_gated_by_biped_mode_or_reward_weight(self):
    env, _, term = make_env()
    first = observations.policy_state(env, add_noise=False).clone()
    term.mode.fill_(True)
    term.group_weight.fill_(1.)
    torch.testing.assert_close(observations.policy_state(env, add_noise=False), first)


class VelocitySymmetryTests(unittest.TestCase):
  def test_velocity_reflects_y_and_mirror_is_an_involution(self):
    from src.tasks.leg_manip.rl.symmetry import mirror_actor_observations
    obs = torch.randn(8, 51)
    mirrored = mirror_actor_observations(obs)
    torch.testing.assert_close(mirrored[:, 48:51], obs[:, 48:51] * torch.tensor((1., -1., 1.)))
    torch.testing.assert_close(mirror_actor_observations(mirrored), obs)

  def test_target_free_symmetry_loss_has_gradients_for_51_inputs(self):
    from src.tasks.leg_manip.rl.symmetry import exact_symmetry_loss
    actor = torch.nn.Linear(51, 12)
    obs = torch.randn(8, 51)
    obs[:, 9:12] = 0.
    loss = exact_symmetry_loss(actor, obs)
    loss.backward()
    self.assertGreater(float(actor.weight.grad.abs().sum()), 0.)
    obs[:, 9] = .1
    self.assertEqual(float(exact_symmetry_loss(actor, obs)), 0.)


class CheckpointMigrationTests(unittest.TestCase):
  def make_checkpoint(self):
    return {
      'actor_state_dict': {'mlp.0.weight': torch.randn(12, 48), 'mlp.0.bias': torch.randn(12)},
      'critic_state_dict': {'mlp.0.weight': torch.randn(1, 131), 'mlp.0.bias': torch.randn(1)},
      'optimizer_state_dict': {'state': {0: {'exp_avg': torch.ones(12, 48)}},
                               'param_groups': [{'lr': .0005, 'params': [0]}]},
      'iter': 14999,
      'infos': {'env_state': {'common_step_counter': 360000},
        'loco_pedipulation': {'stage': 2, 'stage_started': 1000,
          'window': torch.ones(10), 'episodes': 63, 'landing_counts': torch.ones(4)}}}

  def test_migration_preserves_predictions_and_resets_incompatible_state(self):
    from src.tasks.leg_manip.rl.checkpoint import migrate_velocity_checkpoint
    old = self.make_checkpoint()
    original = copy.deepcopy(old)
    migrated = migrate_velocity_checkpoint(old)
    a_old, a_new = old['actor_state_dict'], migrated['actor_state_dict']
    c_old, c_new = old['critic_state_dict'], migrated['critic_state_dict']
    obs = torch.randn(8, 48)
    new_obs = torch.cat((obs, torch.randn(8, 3)), dim=-1)
    torch.testing.assert_close(new_obs @ a_new['mlp.0.weight'].T,
                               obs @ a_old['mlp.0.weight'].T)
    critic = torch.randn(8, 131)
    new_critic = critic.clone()
    new_critic[:, 48:51] *= 2
    torch.testing.assert_close(new_critic @ c_new['mlp.0.weight'].T,
                               critic @ c_old['mlp.0.weight'].T)
    self.assertEqual(migrated['optimizer_state_dict']['state'], {})
    self.assertEqual(migrated['optimizer_state_dict']['param_groups'],
                     old['optimizer_state_dict']['param_groups'])
    state = migrated['infos']['loco_pedipulation']
    self.assertEqual(state['stage'], 2)
    self.assertEqual(state['stage_started'], 360000)
    self.assertEqual(state['episodes'], 0)
    self.assertEqual(migrated['iter'], 14999)
    self.assertEqual(float(state['window'].sum()), 0.)
    torch.testing.assert_close(old['actor_state_dict']['mlp.0.weight'],
                               original['actor_state_dict']['mlp.0.weight'])
    torch.testing.assert_close(old['infos']['loco_pedipulation']['window'], torch.ones(10))
    self.assertEqual(migrated['infos']['leg_manip_observations']['actor_dim'], 51)

  def test_migration_rejects_an_already_migrated_or_other_task_checkpoint(self):
    from src.tasks.leg_manip.rl.checkpoint import migrate_velocity_checkpoint
    old = self.make_checkpoint()
    with self.assertRaises(ValueError):
      migrate_velocity_checkpoint(migrate_velocity_checkpoint(old))
    old['critic_state_dict']['mlp.0.weight'] = torch.zeros(1, 95)
    with self.assertRaises(ValueError):
      migrate_velocity_checkpoint(old)


if __name__ == '__main__':
  unittest.main()
