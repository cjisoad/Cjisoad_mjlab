import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


def module():
  return importlib.import_module('src.tasks.transition_t.entry_evaluation')


def good_metrics(count=1):
  return {
    'orientation_error': torch.full((count,), .1),
    'height_error': torch.full((count,), .01),
    'feet_rms': torch.full((count,), .01),
    'linear_speed': torch.full((count,), .1),
    'angular_speed': torch.full((count,), .1),
    'rear_supported': torch.ones(count, dtype=torch.bool),
    'front_clear': torch.ones(count, dtype=torch.bool),
    'joint_speed': torch.ones(count),
  }


def rsl_actor():
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict
  actor = MLPModel(TensorDict({'actor': torch.zeros(2, 75)}, batch_size=[2]),
    {'actor': ['actor']}, 'actor', 12, hidden_dims=(8,), obs_normalization=True,
    distribution_cfg={'class_name': 'rsl_rl.modules.GaussianDistribution', 'init_std': .5})
  # PPO leaves differentiable runtime distribution tensors after an update.
  actor(TensorDict({'actor': torch.ones(2, 75)}, batch_size=[2]), stochastic_output=True)
  return actor


class ScriptedEnv:
  """Minimal simulator boundary that overwrites terminal state during reset."""
  instances = []

  def __init__(self, cfg, device):
    self.cfg, self.device, self.num_envs = cfg, device, cfg.scene.num_envs
    self.step_dt, self.closed, self.step_count = .5, False, 0
    self.term = SimpleNamespace(course=SimpleNamespace(level=0),
      bank=SimpleNamespace(data=dict(trajectory_id=torch.tensor([1, 5, 10]),
        actual_speed=torch.tensor([.1, .2, .15]), fr_error=torch.tensor([.02, .03, .04])),
        sha256='bank-sha', sample_trials=lambda count, **kwargs: torch.tensor([1, 2]),
        eligible=lambda **kwargs: torch.tensor([1, 2])),
      reference=SimpleNamespace(sha256='motion-sha'), forced_entry_indices=None,
      time_steps=torch.zeros(2, dtype=torch.long), motion=SimpleNamespace(time_step_total=3))
    self.command_manager = SimpleNamespace(get_term=lambda name: self.term)
    self.flags = {name: torch.zeros(2, dtype=torch.bool) for name in cfg.terminations}
    self.termination_manager = SimpleNamespace(active_terms=list(cfg.terminations),
      get_term=lambda name: self.flags[name], terminated=torch.zeros(2, dtype=torch.bool),
      time_outs=torch.zeros(2, dtype=torch.bool))
    self.metrics = good_metrics(2)
    self.actions = []
    self.instances.append(self)
    # Mimic simulator construction reseeding the caller's global generators.
    torch.manual_seed(998)
    np.random.seed(998)
    random.seed(998)

  def reset(self, seed):
    assert self.term.course.level == 0
    self.term.entry_index = self.term.forced_entry_indices.clone()
    self.term.time_steps.zero_()
    return {'actor': torch.ones(2, 75)}, {}

  def step(self, actions):
    self.actions.append(actions.clone())
    self.step_count += 1
    self.term.time_steps[:] = 2
    self.flags['physical_failure'][:] = self.step_count == 2
    self.flags['physical_failure'][1] = False
    self.termination_manager.terminated[:] = self.flags['physical_failure']
    self.metrics['feet_rms'][:] = .12
    self.cfg.terminations['entry_evaluation_capture'].func(self)
    # Automatic reset destroys the terminal frame and entry metadata.
    self.term.time_steps[0] = 0
    self.term.entry_index[0] = 0
    self.metrics['orientation_error'][0] = .9
    return {'actor': torch.ones(2, 75)}, torch.zeros(2), self.termination_manager.terminated, \
      self.termination_manager.time_outs, {}

  def close(self):
    self.closed = True


class EntryEvaluationHelpers(unittest.TestCase):
  def test_hold_timer_requires_completion_and_resets_on_third_bad_sample(self):
    m = module()
    trials = m.EntryTrials(1, .02, device='cpu')
    metrics = good_metrics()
    completed = torch.zeros(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    for _ in range(60):
      trials.record(completed, metrics, healthy, healthy)
      self.assertFalse(trials.standing_held[0])
    completed[:] = True
    for _ in range(25):
      trials.record(completed, metrics, healthy, healthy)
      self.assertFalse(trials.standing_held[0])
    metrics['front_clear'][:] = False
    for bad_count in (1, 2):
      trials.record(completed, metrics, healthy, healthy)
      self.assertAlmostEqual(float(trials.standing_duration[0]), .5)
      self.assertEqual(int(trials.standing_bad_samples[0]), bad_count)
      self.assertFalse(trials.standing_held[0])
    trials.record(completed, metrics, healthy, healthy)
    self.assertEqual(float(trials.standing_duration[0]), 0.)
    metrics['front_clear'][:] = True
    for _ in range(49):
      trials.record(completed, metrics, healthy, healthy)
      self.assertFalse(trials.standing_held[0])
    trials.record(completed, metrics, healthy, healthy)
    self.assertTrue(trials.standing_held[0])
    self.assertTrue(trials.finished[0])

  def test_recovery_preserves_credit_and_restarts_bad_sample_streak(self):
    trials = module().EntryTrials(1, .02, device='cpu')
    metrics = good_metrics()
    completed = torch.ones(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    for _ in range(25):
      trials.record(completed, metrics, healthy, healthy)
    for _ in range(2):
      metrics['rear_supported'][:] = False
      trials.record(completed, metrics, healthy, healthy)
    metrics['rear_supported'][:] = True
    trials.record(completed, metrics, healthy, healthy)
    self.assertAlmostEqual(float(trials.standing_duration[0]), .52)
    self.assertEqual(int(trials.standing_bad_samples[0]), 0)
    metrics['rear_supported'][:] = False
    for _ in range(2):
      trials.record(completed, metrics, healthy, healthy)
    self.assertAlmostEqual(float(trials.standing_duration[0]), .52)
    metrics['rear_supported'][:] = True
    for _ in range(23):
      trials.record(completed, metrics, healthy, healthy)
    self.assertFalse(trials.standing_held[0])
    trials.record(completed, metrics, healthy, healthy)
    self.assertTrue(trials.standing_held[0])

  def test_different_failed_conditions_count_as_consecutive_bad_samples(self):
    trials = module().EntryTrials(1, .02, device='cpu')
    completed = torch.ones(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    trials.record(completed, good_metrics(), healthy, healthy)
    for i, condition in enumerate(('rear_supported', 'front_clear', 'rear_supported'), 1):
      metrics = good_metrics(); metrics[condition][:] = False
      trials.record(completed, metrics, healthy, healthy)
      self.assertEqual(int(trials.standing_bad_samples[0]), i)
      self.assertAlmostEqual(float(trials.standing_duration[0]), .02 if i < 3 else 0.)

  def test_tolerated_bad_sample_pauses_standing_without_strict_diagnostic(self):
    trials = module().EntryTrials(1, .02, device='cpu')
    completed = torch.ones(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    metrics = good_metrics()
    trials.record(completed, metrics, healthy, healthy)
    metrics['rear_supported'][:] = False
    trials.record(completed, metrics, healthy, healthy)
    self.assertAlmostEqual(float(trials.standing_duration[0]), .02)
    record = trials.records()[0]
    self.assertNotIn('strict_held', record)
    self.assertNotIn('strict_hold_s', record)

  def test_physical_failure_clears_credit_immediately_during_tolerated_gap(self):
    trials = module().EntryTrials(1, .02, device='cpu')
    completed = torch.ones(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    for _ in range(10):
      trials.record(completed, good_metrics(), healthy, healthy)
    metrics = good_metrics(); metrics['rear_supported'][:] = False
    trials.record(completed, metrics, healthy, healthy)
    trials.record(completed, metrics, ~healthy, healthy,
      termination_reasons={'physical_failure': ~healthy})
    self.assertEqual(float(trials.standing_duration[0]), 0.)
    self.assertFalse(trials.standing_held[0])
    self.assertTrue(trials.finished[0])
    self.assertEqual(trials.records()[0]['reason'], 'physical_failure')

  def test_physical_failure_wins_over_simultaneous_standing_hold(self):
    m = module()
    trials = m.EntryTrials(1, .1, device='cpu')
    completed = torch.ones(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    for _ in range(9):
      trials.record(completed, good_metrics(), healthy, healthy)
    trials.record(completed, good_metrics(), ~healthy, healthy,
                  termination_reasons={'physical_failure': ~healthy})
    result = trials.records()[0]
    self.assertFalse(result['standing_held'])
    self.assertEqual(result['reason'], 'physical_failure')

  def test_timeout_is_failure_and_cannot_earn_success_after_automatic_reset(self):
    m = module()
    trials = m.EntryTrials(2, .1, device='cpu', timeout_s=1.)
    completed = torch.ones(2, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    metrics = good_metrics(2)
    metrics['front_clear'][0] = False
    metrics['feet_rms'][0] = .12
    for _ in range(10):
      trials.record(completed, metrics, healthy, healthy)
    terminal = copy.deepcopy(trials.records())
    self.assertEqual(terminal[0]['reason'], 'standing_hold_not_completed')
    self.assertFalse(terminal[0]['standing_held'])
    self.assertTrue(terminal[1]['standing_held'])
    for _ in range(20):
      trials.record(completed, good_metrics(2), healthy, healthy)
    self.assertEqual(trials.records(), terminal)
    self.assertAlmostEqual(terminal[0]['time'], 1.)

  def test_timeout_occurs_at_500_control_steps(self):
    m = module()
    trials = m.EntryTrials(1, .02, device='cpu')
    metrics = good_metrics()
    metrics['front_clear'][:] = False
    completed = torch.ones(1, dtype=torch.bool)
    healthy = torch.zeros_like(completed)
    for _ in range(499):
      trials.record(completed, metrics, healthy, healthy)
    self.assertFalse(trials.finished[0])
    trials.record(completed, metrics, healthy, healthy)
    self.assertTrue(trials.finished[0])
    self.assertAlmostEqual(trials.records()[0]['time'], 10., places=9)

  def test_terminal_metrics_are_copied_before_reset_and_nonfinite_is_failure(self):
    m = module()
    trials = m.EntryTrials(1, .02, device='cpu')
    metrics = good_metrics()
    metrics['orientation_error'][:] = float('nan')
    flag = torch.ones(1, dtype=torch.bool)
    trials.record(flag, metrics, ~flag, ~flag)
    metrics['orientation_error'].zero_()
    result = trials.records()[0]
    self.assertEqual(result['reason'], 'nonfinite_metrics')
    self.assertIsNone(result['endpoint_metrics']['orientation_error'])
    json.dumps(result, allow_nan=False)

  def test_json_safe_and_rng_isolation(self):
    m = module()
    value = {'bad': float('nan'), 'nested': [float('inf')]}
    self.assertEqual(m.json_safe(value), {'bad': None, 'nested': [None]})
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    py_state, np_state = random.getstate(), np.random.get_state()
    with m.isolated_rng('cpu'):
      torch.rand(20)
      random.random()
      np.random.rand()
    self.assertTrue(torch.equal(torch.rand(4), expected))
    self.assertEqual(random.getstate(), py_state)
    self.assertEqual(np.random.get_state()[0], np_state[0])
    self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
    self.assertEqual(np.random.get_state()[2:], np_state[2:])

  def test_api_is_importable_and_actor_state_is_not_mutated(self):
    m = module()
    self.assertTrue(callable(m.evaluate_entries))
    actor = torch.nn.Linear(2, 2)
    actor.train()
    before = copy.deepcopy(actor.state_dict())
    with self.assertRaises((FileNotFoundError, ValueError, ImportError)):
      m.evaluate_entries(actor, '/does/not/exist.npz', level=0, device='cpu', attempts=1)
    self.assertTrue(actor.training)
    self.assertEqual(before.keys(), actor.state_dict().keys())
    for key in before:
      torch.testing.assert_close(before[key], actor.state_dict()[key])

  def test_api_rejects_invalid_level_and_attempts_before_environment_construction(self):
    m = module()
    for level, attempts in ((-1, 64), (4, 64), (0, 0), (0, 1.5)):
      with self.subTest(level=level, attempts=attempts), self.assertRaises(ValueError):
        m.evaluate_entries(torch.nn.Linear(2, 2), 'bank.npz', level=level,
                           device='cpu', attempts=attempts)

  def test_rollout_captures_before_reset_and_preserves_trained_actor_and_rng(self):
    m = module()
    actor = rsl_actor()
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    runtime_distribution = actor.distribution._distribution
    cfg = SimpleNamespace(seed=None, scene=SimpleNamespace(num_envs=2), episode_length_s=8.,
      commands={'motion': SimpleNamespace(entry_probability=None, nominal_probability=None, debug_vis=True)},
      terminations={'physical_failure': SimpleNamespace()})
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with tempfile.TemporaryDirectory() as directory:
      bank = Path(directory)/'bank.npz'
      bank.touch()
      with patch('mjlab.envs.ManagerBasedRlEnv', ScriptedEnv), \
        patch('src.tasks.transition_t.entry_course.entry_course_env_cfg', return_value=cfg), \
        patch.object(m, 'endpoint_metrics', side_effect=lambda env: env.metrics):
        report = m.evaluate_entries(actor, bank, level=0, device='cpu', attempts=2)
    self.assertEqual(report['attempts'], 2)
    self.assertEqual(report['successes'], 1)
    self.assertNotIn('strict_successes', report)
    self.assertNotIn('strict_success_rate', report)
    self.assertNotIn('strict_endpoint', report['standing_hold_policy'])
    self.assertEqual(report['full_motion_completions'], 2)
    self.assertEqual(report['distinct_trajectories'], 2)
    self.assertFalse(report['promotion_evidence_sufficient'])
    self.assertEqual(report['failure_reasons'], {'physical_failure': 1})
    self.assertEqual(report['records'][0]['entry_index'], 1)
    self.assertEqual(report['records'][0]['trajectory_id'], 5)
    self.assertEqual(report['records'][0]['endpoint_metrics']['orientation_error'], float(torch.tensor(.9)))
    self.assertEqual(report['records'][1]['reason'], 'standing_eligible')
    self.assertEqual(sum(row['attempts'] for row in report['by_actual_speed']), 2)
    self.assertEqual(sum(row['attempts'] for row in report['by_fr_error']), 2)
    for row in report['by_actual_speed'] + report['by_fr_error']:
      self.assertNotIn('strict_successes', row)
      self.assertNotIn('strict_success_rate', row)
    for row in report['records']:
      self.assertNotIn('strict_held', row)
      self.assertNotIn('strict_hold_s', row)
    self.assertEqual(report['evaluation'], 'standing_eligibility')
    self.assertFalse(report['teacher_chain_success_evaluated'])
    self.assertTrue(ScriptedEnv.instances[-1].closed)
    self.assertTrue(actor.training)
    self.assertIs(actor.distribution._distribution, runtime_distribution)
    for key, value in actor.state_dict().items():
      torch.testing.assert_close(value, before[key])
    self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
    self.assertEqual(random.getstate(), py_state)
    self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
    json.dumps(report, allow_nan=False)


if __name__ == '__main__':
  unittest.main()
