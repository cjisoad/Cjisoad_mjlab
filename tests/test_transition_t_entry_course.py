"""Restore real teacher state without replacing it by the reference pose."""
from dataclasses import fields
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


def module():
  assert importlib.util.find_spec('src.tasks.transition_t.entry_course') is not None, 'entry course is missing'
  return importlib.import_module('src.tasks.transition_t.entry_course')


class EntryRestoreTests(unittest.TestCase):
  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, 'env'):
      cls.env.close(); cls.directory.cleanup()

  def world(self):
    cls = type(self)
    if hasattr(cls, 'env'):
      return cls.env
    m = module()
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.transition_t.motion import TransitionMotion, DEFAULT_MOTION_PATH
    reference = TransitionMotion(DEFAULT_MOTION_PATH)
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    from src.tasks.pedipulation.mdp.target_limiter import target_limiter_contract
    config = transition_env_cfg(play=True)
    action_config = config.actions['joint_pos']
    zero = np.array([-.1, .9, -1.8, .1, .9, -1.8, -.1, .9, -1.8, .1, .9, -1.8], np.float32)
    limiter = target_limiter_contract(action_config.target_velocity_limits,
      action_config.target_acceleration_limits, .005)
    arrays = reference.arrays
    anchor = reference.body_name_to_index['base_link']
    root = np.tile(np.concatenate((arrays['body_pos_w'][0, anchor], arrays['body_quat_w'][0, anchor],
      np.array([.1, 0, 0, .05, 0, 0]))), (100, 1)).astype(np.float32)
    root[:, :2] += [.3, -.2]; root[:, 2] += .01
    action = np.full((100, 12), .3, np.float32)
    bank = dict(root_state=root, joint_pos=np.tile(arrays['joint_pos'][0]+.02, (100, 1)),
      joint_vel=np.full((100, 12), .4, np.float32), raw_action=action,
      previous_action=action-.1, manager_actions=np.stack((action, action-.1, action-.2), 1),
      previous_joint_vel=np.full((100, 12), .5, np.float32), actual_speed=np.full(100, .1, np.float32),
      limiter_position=zero+action*.25, limiter_velocity=np.full((100, 12), .6, np.float32),
      fr_error=np.full(100, .02, np.float32), fr_offset=np.zeros((100, 3), np.float32),
      command=np.zeros((100, 6), np.float32), trajectory_id=np.arange(100),
      metadata_json=np.array(json.dumps(dict(schema_version=2, motion_sha256=reference.sha256,
        teacher_sha256='fixture', target_limiter=limiter, action_scale=.25, action_zero=zero.tolist()))))
    cls.directory = tempfile.TemporaryDirectory()
    path = Path(cls.directory.name)/'entries.npz'; np.savez_compressed(path, **bank)
    cfg = m.entry_course_env_cfg(path, num_envs=3, play=True)
    cfg.commands['motion'].entry_probability = 1.
    cfg.commands['motion'].nominal_probability = 0.
    torch.set_num_threads(1)
    cls.env = ManagerBasedRlEnv(cfg, device='cpu')
    return cls.env

  def test_real_state_and_all_action_history_restored_before_observation(self):
    world = self.world()
    obs, _ = world.reset(seed=15)
    motion = world.command_manager.get_term('motion')
    action = world.action_manager.get_term('joint_pos')
    ids = motion.entry_index
    bank = motion.bank.data
    torch.testing.assert_close(world.scene['robot'].data.joint_pos, bank['joint_pos'][ids])
    torch.testing.assert_close(world.scene['robot'].data.joint_vel, bank['joint_vel'][ids])
    torch.testing.assert_close(action.raw_action, bank['raw_action'][ids])
    torch.testing.assert_close(action.previous_action, bank['previous_action'][ids])
    torch.testing.assert_close(world.action_manager.action, bank['manager_actions'][ids, 0])
    torch.testing.assert_close(world.action_manager.prev_action, bank['manager_actions'][ids, 1])
    torch.testing.assert_close(world.action_manager.prev_prev_action, bank['manager_actions'][ids, 2])
    assert obs['actor'].shape == (3, 75) and obs['critic'].shape == (3, 192)
    assert torch.isfinite(obs['actor']).all()
    assert motion.time_steps.tolist() == [0, 0, 0]
    assert (motion.physical_reset_age_steps >= 10).all()
    assert ((motion.robot_anchor_pos_w-motion.anchor_pos_w)[:, 2]-.01).abs().max() < 1e-5
    torch.testing.assert_close(motion.robot_anchor_pos_w[:, :2], motion.anchor_pos_w[:, :2])

  def test_partial_entry_restore_keeps_other_worlds_and_history(self):
    world = self.world(); world.reset(seed=11)
    motion = world.command_manager.get_term('motion')
    action = world.action_manager
    root = world.sim.data.qpos.clone(); history = action.action.clone()
    motion._resample_command(torch.tensor([1]))
    torch.testing.assert_close(world.sim.data.qpos.clone()[[0, 2]], root[[0, 2]])
    torch.testing.assert_close(action.action[[0, 2]], history[[0, 2]])

  def test_original_physics_and_observation_contract_retained(self):
    m = module()
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    world = self.world(); cfg = world.cfg
    original = transition_env_cfg(play=True)
    assert cfg.scene.entities['robot'] == original.scene.entities['robot']
    assert cfg.sim == original.sim
    assert list(cfg.observations['actor'].terms) == list(original.observations['actor'].terms)
    assert cfg.actions == original.actions

  def test_entry_restores_limiter_momentum_before_first_applied_target(self):
    world = self.world(); world.reset(seed=15)
    action = world.action_manager.get_term('joint_pos')
    motion = world.command_manager.get_term('motion')
    ids = motion.entry_index
    torch.testing.assert_close(action.target_limiter.position, motion.bank.data['limiter_position'][ids])
    torch.testing.assert_close(action.target_limiter.velocity, motion.bank.data['limiter_velocity'][ids])
    before = action.target_limiter.velocity.clone()
    world.action_manager.process_action(torch.full((3, 12), -10.))
    world.action_manager.apply_action()
    assert ((action.target_limiter.velocity-before).abs()
      <= action.target_limiter.acceleration_limits*world.physics_dt+1e-6).all()


if __name__ == '__main__':
  unittest.main()
