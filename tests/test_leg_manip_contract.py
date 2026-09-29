"""leg_manip env_cfg assembly and real-env integration contracts."""

import unittest

import torch

from src.tasks.leg_manip.constants import STAGE_BIPED
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp.commands import LegManipCommandCfg

SHARED_TERMS = ('fr_position_tracking', 'fr_ground_contact', 'is_terminated', 'joint_acc_l2',
                'action_rate_l2', 'joint_pos_limits', 'torques', 'collision', 'alive')
LOCO_TERMS = ('track_linear_velocity', 'track_angular_velocity', 'pose', 'stand_still',
              'foot_gait', 'foot_clearance', 'foot_slip', 'soft_landing', 'support_contact',
              'base_height', 'body_orientation_l2', 'body_ang_vel', 'angular_momentum')
BIPED_TERMS = ('biped_base_height', 'biped_tracking_lin_vel', 'biped_tracking_ang_vel',
               'biped_lin_vel_z', 'biped_ang_vel_xy', 'handstand_orientation',
               'handstand_feet_on_air', 'handstand_feet_height_exp', 'biped_feet_clearance',
               'biped_contact', 'biped_ang_xz', 'symmetric_joints', 'orientation_symmetry',
               'feet_height_symmetry', 'default_pos_front', 'default_pos_rear',
               'default_hip_pos', 'default_pos_reward_FL', 'default_pos_reward_FR')
GATED_BIPED_TERMS = ('biped_tracking_lin_vel', 'biped_tracking_ang_vel', 'biped_feet_clearance',
                     'biped_contact', 'biped_ang_xz', 'symmetric_joints',
                     'default_pos_reward_FL', 'default_pos_reward_FR')


class EnvCfgAssemblyTests(unittest.TestCase):
  def test_reward_groups_cover_both_source_tasks(self):
    cfg = leg_manip_env_cfg()
    self.assertEqual(tuple(cfg.rewards), SHARED_TERMS + LOCO_TERMS + BIPED_TERMS)
    self.assertEqual(len(cfg.rewards), 41)
    weights = {name: cfg.rewards[name].weight for name in cfg.rewards}
    self.assertEqual(weights['fr_position_tracking'], 2.5)
    self.assertEqual(weights['biped_base_height'], 1.5)
    self.assertEqual(weights['is_terminated'], -200.)
    self.assertEqual(weights['alive'], 1.)
    self.assertEqual(weights['base_height'], .5)
    self.assertEqual(weights['pose'], 1.)
    self.assertEqual(weights['handstand_feet_height_exp'], 5.)

  def test_height_gate_term_precedes_all_gated_biped_terms(self):
    names = tuple(leg_manip_env_cfg().rewards)
    gate_index = names.index('biped_base_height')
    for name in GATED_BIPED_TERMS:
      self.assertLess(gate_index, names.index(name), name)

  def test_actor_abi_is_a_single_48_value_policy_term(self):
    for play in (False, True):
      cfg = leg_manip_env_cfg(play=play)
      self.assertEqual(tuple(cfg.observations['actor'].terms), ('policy',))
      self.assertEqual(cfg.observations['actor'].terms['policy'].params['add_noise'], not play)

  def test_command_term_and_curriculum_wiring(self):
    cfg = leg_manip_env_cfg()
    self.assertIsInstance(cfg.commands['twist'], LegManipCommandCfg)
    self.assertTrue(cfg.commands['twist'].curriculum_enabled)
    self.assertEqual(cfg.commands['twist'].initial_stage, 0)
    self.assertIn('manipulation', cfg.curriculum)
    play = leg_manip_env_cfg(play=True)
    self.assertFalse(play.commands['twist'].curriculum_enabled)
    self.assertEqual(play.commands['twist'].initial_stage, STAGE_BIPED)
    self.assertEqual(play.curriculum, {})
    self.assertGreater(play.episode_length_s, 1e8)

  def test_terminations_cover_both_stances(self):
    names = tuple(leg_manip_env_cfg().terminations)
    for name in ('time_out', 'base_contact', 'illegal_contact', 'fell_over', 'landing_failed'):
      self.assertIn(name, names)

  def test_push_robot_removed_in_play(self):
    self.assertIn('push_robot', leg_manip_env_cfg().events)
    self.assertNotIn('push_robot', leg_manip_env_cfg(play=True).events)


class RealEnvIntegrationTests(unittest.TestCase):
  def test_real_env_builds_and_steps_on_cpu(self):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.leg_manip.mdp.commands import LegManipCommand
    from src.tasks.pedipulation.mdp.actions import PedipulationPositionAction

    cfg = leg_manip_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device='cpu')
    try:
      obs, _ = env.reset()
      self.assertEqual(obs['actor'].shape, (2, 48))
      # 48 shared + 3 lin vel + 34 DR + 4 contact + 2 gait + 20 FR/mode
      # + 4 height + 4 air time + 12 contact forces.
      self.assertEqual(obs['critic'].shape, (2, 131))
      self.assertIsInstance(env.command_manager.get_term('twist'), LegManipCommand)
      self.assertIsInstance(env.action_manager.get_term('joint_pos'), PedipulationPositionAction)
      action = torch.zeros(2, 12)
      for _ in range(5):
        obs, reward, terminated, truncated, _ = env.step(action)
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue(torch.isfinite(obs['actor']).all())
        self.assertTrue(torch.isfinite(obs['critic']).all())
    finally:
      env.close()


if __name__ == '__main__':
  unittest.main()
