"""Holding reward credits good samples and tolerates short support interruptions."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def metrics(count=2):
  return {k: torch.full((count,), v) for k, v in dict(orientation_error=.1,
    height_error=.01, feet_rms=.2, linear_speed=.1, angular_speed=.1,
    tripod_supported=True, fr_clear=True, joint_speed=1.).items()}


class StandingRewardTests(unittest.TestCase):
  def setUp(self):
    self.mod = importlib.import_module('src.tasks.transition_down_t.rewards')
    self.motion = SimpleNamespace(time_steps=torch.full((2,), 350),
      motion=SimpleNamespace(time_step_total=351))
    self.env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0,
      command_manager=SimpleNamespace(get_term=lambda name: self.motion),
      termination_manager=SimpleNamespace(terminated=torch.zeros(2, dtype=torch.bool)))
    self.values = metrics()
    self.failed = torch.zeros(2, dtype=torch.bool)
    self.reward = self.mod.SustainedTripodReward(SimpleNamespace(params={}), self.env)
    self.patches = [patch.object(self.mod, 'endpoint_metrics', side_effect=lambda env: self.values),
      patch.object(self.mod.terminations, 'physical_failure', side_effect=lambda env: self.failed)]
    for p in self.patches:p.start();self.addCleanup(p.stop)

  def step(self):
    self.env.common_step_counter += 1
    return self.reward(self.env)

  def test_no_reward_or_time_before_reference_completion(self):
    self.motion.time_steps[:] = 349
    for _ in range(60):self.assertEqual(float(self.step().sum()), 0.)
    self.assertEqual(float(self.reward.duration.sum()), 0.)
    self.motion.time_steps[:] = 350
    self.assertAlmostEqual(float(self.step()[0]), .216, places=6)

  def test_ramp_and_cap_with_world_foot_drift_allowed(self):
    for _ in range(25):value = self.step()
    self.assertAlmostEqual(float(value[0]), .6, places=6)
    for _ in range(25):value = self.step()
    self.assertAlmostEqual(float(value[0]), 1., places=6)
    for _ in range(100):value = self.step()
    self.assertAlmostEqual(float(value[0]), 1., places=6)

  def test_bad_samples_give_zero_pause_twice_then_reset(self):
    for _ in range(25):self.step()
    self.values['tripod_supported'][0] = False
    for i in (1, 2):
      self.assertEqual(float(self.step()[0]), 0.)
      self.assertAlmostEqual(float(self.reward.duration[0]), .5)
      self.assertEqual(int(self.reward.bad_samples[0]), i)
    self.step()
    self.assertEqual(float(self.reward.duration[0]), 0.)
    self.values['tripod_supported'][0] = True
    self.assertAlmostEqual(float(self.step()[0]), .216, places=6)

  def test_recovery_preserves_time_and_clears_streak(self):
    for _ in range(25):self.step()
    self.values['tripod_supported'][0] = False
    self.step();self.step()
    self.values['tripod_supported'][0] = True
    self.assertAlmostEqual(float(self.step()[0]), .616, places=6)
    self.assertEqual(int(self.reward.bad_samples[0]), 0)

  def test_failure_and_nonfinite_clear_immediately(self):
    for kind in ('physical', 'terminated', 'nonfinite'):
      with self.subTest(kind=kind):
        self.reward.reset();self.failed.zero_();self.env.termination_manager.terminated.zero_()
        self.values=metrics()
        for _ in range(10):self.step()
        if kind=='physical':self.failed[0]=True
        elif kind=='terminated':self.env.termination_manager.terminated[0]=True
        else:self.values['linear_speed'][0]=float('nan')
        self.assertEqual(float(self.step()[0]),0.)
        self.assertEqual(float(self.reward.duration[0]),0.)

  def test_partial_reset_and_same_step_calls_do_not_duplicate_credit(self):
    for _ in range(25):value=self.step()
    before=self.reward.duration.clone()
    torch.testing.assert_close(self.reward(self.env),value)
    torch.testing.assert_close(self.reward.duration,before)
    self.reward.reset(torch.tensor([0]))
    self.assertEqual(float(self.reward(self.env)[0]),0.)
    self.assertAlmostEqual(float(self.reward.duration[1]),.5)
    self.step()
    self.assertAlmostEqual(float(self.reward.duration[0]),.02)
    self.assertAlmostEqual(float(self.reward.duration[1]),.52)


class StandingRewardIntegrationTests(unittest.TestCase):
  def test_configuration_and_real_reward_manager_scaling(self):
    from src.tasks.transition_down_t.env_cfg import down_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.transition_down_t import rewards
    cfg=down_env_cfg(play=True);cfg.scene.num_envs=2
    self.assertEqual(cfg.episode_length_s,10.)
    self.assertEqual(cfg.rewards['standing_hold'].weight,.1)
    torch.set_num_threads(1)
    env=ManagerBasedRlEnv(cfg,device='cpu')
    try:
      env.reset(seed=42)
      motion=env.command_manager.get_term('motion')
      motion.time_steps[:]=motion.motion.time_step_total-1
      with patch.object(rewards,'endpoint_metrics',return_value=metrics()), \
        patch.object(rewards.terminations,'physical_failure',return_value=torch.zeros(2,dtype=torch.bool)):
        for _ in range(50):
          env.common_step_counter+=1
          env.reward_manager.compute(env.step_dt)
        rate=dict(env.reward_manager.get_active_iterable_terms(0))['standing_hold'][0]
        self.assertAlmostEqual(rate,.1,places=6)
        self.assertAlmostEqual(float(env.reward_manager._episode_sums['standing_hold'][0]),.0608,places=6)
        term=env.reward_manager.get_term_cfg('standing_hold').func
        env.reward_manager.reset(torch.tensor([0]))
        self.assertEqual(float(term.duration[0]),0.)
        self.assertAlmostEqual(float(term.duration[1]),1.)
    finally:env.close()


if __name__=='__main__':unittest.main()
