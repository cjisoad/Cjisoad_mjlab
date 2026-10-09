"""Standalone held-out entry evaluations need no training updates."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/evaluate_transition_entry.py'


class EntryEvaluationCliTests(unittest.TestCase):
  def module(self):
    self.assertTrue(SCRIPT.is_file(), 'entry evaluation CLI is missing')
    spec = importlib.util.spec_from_file_location('evaluate_transition_entry', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

  def test_defaults_cover_all_course_levels(self):
    args = self.module().parse_args(['--entry-bank', 'entries.npz', '--checkpoint', 'model.pt'])
    self.assertEqual(args.level, 'all')
    self.assertEqual(args.attempts, 64)

  def test_old_policy_with_new_limiter_evaluation_requires_explicit_flag(self):
    args = self.module().parse_args(['--entry-bank', 'entries.npz', '--checkpoint', 'old.pt',
      '--allow-legacy-transition-limiter'])
    assert args.allow_legacy_transition_limiter

  def test_invalid_attempts_and_level_are_rejected(self):
    for option, value in (('--attempts', '0'), ('--level', '4')):
      with self.assertRaises(SystemExit):
        self.module().parse_args(['--entry-bank', 'bank.npz', '--checkpoint', 'model.pt', option, value])


if __name__ == '__main__':
  unittest.main()
