"""Bounded training CLI requires an explicit initialization distribution."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class EntryTrainCliTests(unittest.TestCase):
  def script(self):
    path = Path(__file__).resolve().parents[1]/'scripts/train_transition_entry.py'
    assert path.is_file(), 'entry training script is missing'
    spec = importlib.util.spec_from_file_location('train_transition_entry', path)
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    return script

  def test_requires_state_bank_and_uses_new_experiment(self):
    script = self.script()
    with self.assertRaises(SystemExit):
      script.parse_args([])
    args = script.parse_args(['--entry-bank', 'entries.npz'])
    assert args.checkpoint.name == 'model_30000.pt'
    assert args.iterations == 5000
    assert args.eval_attempts >= 64
    assert args.output is None

  def test_rejects_empty_training_and_evaluation(self):
    script = self.script()
    for key in ('--iterations', '--num-envs', '--eval-interval', '--eval-attempts'):
      with self.assertRaises(SystemExit):
        script.parse_args(['--entry-bank', 'entries.npz', key, '0'])

  def test_load_failure_closes_environment_without_masking_original_error(self):
    script = self.script()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      for name in ('entries.npz', 'seed.pt'):
        (path/name).touch()
      args = script.parse_args(['--entry-bank', str(path/'entries.npz'),
        '--checkpoint', str(path/'seed.pt'), '--device', 'cpu', '--output', str(path/'run')])
      env = Mock()
      runner = SimpleNamespace(logger=SimpleNamespace(), warm_start=Mock(side_effect=ValueError('seed contract differs')))
      with patch('mjlab.envs.ManagerBasedRlEnv', return_value=env), \
        patch('mjlab.rl.RslRlVecEnvWrapper', return_value=env), \
        patch('src.tasks.transition_t.entry_rl.EntryOnPolicyRunner', return_value=runner):
        with self.assertRaisesRegex(ValueError, 'seed contract differs'):
          script.train(args)
      env.close.assert_called_once()

  def test_resume_requires_a_target_beyond_completed_updates(self):
    script = self.script()
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      for name in ('entries.npz', 'resume.pt'):
        (path/name).touch()
      args = script.parse_args(['--entry-bank', str(path/'entries.npz'),
        '--checkpoint', str(path/'resume.pt'), '--resume', '--device', 'cpu',
        '--iterations', '2', '--baseline', 'none', '--output', str(path/'run')])
      env = Mock()
      runner = SimpleNamespace(logger=SimpleNamespace(), resume=Mock(), current_learning_iteration=2,
        term=SimpleNamespace(course=SimpleNamespace(state_dict=lambda: {}),
          bank=SimpleNamespace(sha256='bank', coverage=lambda: {})),
        seed_checkpoint_sha256='seed', _entry_recipe=lambda: {})
      with patch('mjlab.envs.ManagerBasedRlEnv', return_value=env), \
        patch('mjlab.rl.RslRlVecEnvWrapper', return_value=env), \
        patch('mjlab.utils.os.dump_yaml'), \
        patch('src.tasks.transition_t.entry_rl.EntryOnPolicyRunner', return_value=runner):
        with self.assertRaisesRegex(ValueError, 'already|completed|iterations'):
          script.train(args)
      self.assertFalse((path/'run/finished.json').exists())
      env.close.assert_called_once()


if __name__ == '__main__':
  unittest.main()
