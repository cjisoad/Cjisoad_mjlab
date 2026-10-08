"""Checkpoint compatibility must fail before any policy weights are loaded."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg


class RunnerContracts(unittest.TestCase):
  def test_compact_beyondmimic_ppo(self):
    cfg = transition_ppo_runner_cfg()
    self.assertEqual(tuple(cfg.actor.hidden_dims), (256, 128, 128))
    self.assertEqual(tuple(cfg.critic.hidden_dims), (256, 128, 128))
    self.assertTrue(cfg.actor.obs_normalization)
    self.assertEqual(cfg.actor.distribution_cfg['init_std'], .5)
    self.assertEqual(cfg.algorithm.learning_rate, 3e-4)
    self.assertEqual(cfg.algorithm.gamma, .99)
    self.assertEqual(cfg.num_steps_per_env, 24)

  def runner(self):
    runner = object.__new__(TransitionOnPolicyRunner)
    runner._contract = lambda: {'motion_sha256': 'expected', 'action_dim': 12}
    runner.alg = SimpleNamespace(actor=torch.nn.Module(), critic=torch.nn.Module())
    runner.term = SimpleNamespace(bin_failed_count=torch.zeros(3), _current_bin_failed=torch.zeros(3))
    return runner

  def test_incompatible_checkpoint_rejected_before_loading_weights(self):
    runner = self.runner()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'model.pt'
      torch.save({'infos': {'transition_t_contract': {'motion_sha256': 'wrong'}}}, path)
      with patch('mjlab.rl.MjlabOnPolicyRunner.load') as parent:
        with self.assertRaisesRegex(ValueError, 'contract'):
          runner.load(str(path))
        parent.assert_not_called()

  def test_sampling_state_roundtrip(self):
    runner = self.runner()
    info = {'transition_t_contract': runner._contract(), 'transition_t_sampling': {
      'bin_failed_count': torch.tensor([.1, .2, .3]),
      '_current_bin_failed': torch.tensor([0., 1., 0.])}}
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'model.pt'
      torch.save({'infos': info}, path)
      with patch('mjlab.rl.MjlabOnPolicyRunner.load', return_value=info):
        runner.load(str(path))
      torch.testing.assert_close(runner.term.bin_failed_count, info['transition_t_sampling']['bin_failed_count'])
      torch.testing.assert_close(runner.term._current_bin_failed, info['transition_t_sampling']['_current_bin_failed'])

  def test_load_handles_normalization_buffers_created_during_rollout(self):
    runner = self.runner()
    with torch.inference_mode():
      runner.alg.actor.register_buffer('statistics', torch.ones(2))
    info = {'transition_t_contract': runner._contract(), 'transition_t_sampling': {
      'bin_failed_count': torch.zeros(3), '_current_bin_failed': torch.zeros(3)}}
    def restore(*args, **kwargs):
      runner.alg.actor.statistics.copy_(torch.zeros(2))
      return info
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / 'model.pt'
      torch.save({'infos': info}, path)
      with patch('mjlab.rl.MjlabOnPolicyRunner.load', side_effect=restore):
        runner.load(str(path))
      self.assertFalse(runner.alg.actor.statistics.is_inference())

  def test_episode_counter_is_never_randomized_by_runner(self):
    runner = self.runner()
    with patch('mjlab.rl.MjlabOnPolicyRunner.learn') as parent:
      runner.learn(1, init_at_random_ep_len=True)
      parent.assert_called_once_with(1, init_at_random_ep_len=False)


if __name__ == '__main__':
  unittest.main()
