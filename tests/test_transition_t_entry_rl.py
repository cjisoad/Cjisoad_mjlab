"""Entry-course resume refuses changes to the initialization distribution."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch


def module():
  assert importlib.util.find_spec('src.tasks.transition_t.entry_rl') is not None, 'entry runner is missing'
  return importlib.import_module('src.tasks.transition_t.entry_rl')


class EntryRunnerTests(unittest.TestCase):
  def runner(self):
    m = module()
    from src.tasks.transition_t.entry_bank import EntryCurriculum
    runner = object.__new__(m.EntryOnPolicyRunner)
    runner.term = SimpleNamespace(bank=SimpleNamespace(sha256='bank', metadata={'teacher_sha256': 'teacher'}),
      course=EntryCurriculum(), bin_failed_count=torch.zeros(3), _current_bin_failed=torch.zeros(3))
    runner.seed_checkpoint_sha256 = 'seed'
    runner.device = 'cpu'
    runner.current_learning_iteration = 10
    runner.alg = SimpleNamespace(learning_rate=3e-4,
      optimizer=SimpleNamespace(param_groups=[{'lr': 1e-5}]))
    runner._contract = lambda: {'motion': 'same'}
    runner._verify_bank_physics = lambda: None
    runner._augmentation_recipe = lambda: {'fixture': 'same'}
    runner._initialization_recipe = lambda: {'fixture': 'same'}
    return runner

  def test_resume_rejects_changed_entry_split_or_mixture_before_weights(self):
    runner = self.runner()
    command = SimpleNamespace(entry_split='train', entry_probability=None,
      nominal_probability=None, sampling_mode='adaptive', first_frame_probability=0.)
    runner.term.cfg = command
    del runner._initialization_recipe
    before = runner._entry_recipe()
    self.assertEqual(before['initialization']['entry_split'], 'train')
    info = dict(recipe=before, course=runner.term.course.state_dict(),
      updates_completed=10, seed_checkpoint_sha256='seed')
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'model.pt'; torch.save({'infos': {'transition_entry_training': info}}, p)
      for field, changed in (('entry_split', 'holdout'), ('entry_probability', 1.),
          ('nominal_probability', .9), ('sampling_mode', 'start'), ('first_frame_probability', .5)):
        previous = getattr(command, field)
        setattr(command, field, changed)
        with self.subTest(field=field), patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load') as parent:
          with self.assertRaisesRegex(ValueError, 'recipe|bank'):
            runner.resume(p)
          parent.assert_not_called()
        setattr(command, field, previous)

  def test_recipe_records_startup_augmentation_and_rejects_modified_friction(self):
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    runner = self.runner()
    cfg = transition_env_cfg()
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))
    del runner._augmentation_recipe
    before = runner._entry_recipe()
    augmentation = before['training_augmentation']
    self.assertTrue(augmentation['actor_corruption'])
    self.assertEqual(augmentation['events']['foot_friction']['params']['ranges'], (.6, 1.2))
    self.assertEqual(augmentation['events']['base_com']['params']['ranges'][0], (-.01, .01))
    self.assertEqual(augmentation['events']['encoder_bias']['params']['bias_range'], (-.01, .01))
    cfg.events['foot_friction'].params['ranges'] = (.4, 1.5)
    self.assertNotEqual(before, runner._entry_recipe())
    info = dict(recipe=before, course=runner.term.course.state_dict(),
      updates_completed=10, seed_checkpoint_sha256='seed')
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'model.pt'; torch.save({'infos': {'transition_entry_training': info}}, p)
      with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load') as parent:
        with self.assertRaisesRegex(ValueError, 'recipe|bank'):
          runner.resume(p)
        parent.assert_not_called()

  def test_course_state_and_distribution_roundtrip(self):
    runner = self.runner()
    runner.term.course.level = 2
    runner.term.course.streak = 1
    captured = []
    with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.save', side_effect=lambda p, i: captured.append(i)):
      runner.save('unused.pt')
    state = captured[0]['transition_entry_training']
    assert state['updates_completed'] == 11
    assert state['recipe']['bank_sha256'] == 'bank'
    assert state['course']['level'] == 2
    resumed = self.runner()
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'model.pt'; torch.save({'infos': captured[0]}, p)
      with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load', return_value=captured[0]):
        resumed.resume(str(p))
    assert resumed.term.course.state_dict() == runner.term.course.state_dict()
    assert resumed.current_learning_iteration == 11

  def test_mismatched_bank_is_rejected_before_weights(self):
    runner = self.runner()
    info = dict(recipe={**runner._entry_recipe(), 'bank_sha256': 'wrong'},
      course=runner.term.course.state_dict(), updates_completed=10, seed_checkpoint_sha256='seed')
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'model.pt'; torch.save({'infos': {'transition_entry_training': info}}, p)
      with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load') as parent:
        with self.assertRaisesRegex(ValueError, 'recipe|bank'):
          runner.resume(str(p))
        parent.assert_not_called()

  def test_resume_restores_adaptive_learning_rate_from_optimizer(self):
    runner = self.runner()
    info = dict(recipe=runner._entry_recipe(), course=runner.term.course.state_dict(),
      updates_completed=10, seed_checkpoint_sha256='seed')
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'model.pt'; torch.save({'infos': {'transition_entry_training': info}}, p)
      with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load', return_value={}):
        runner.resume(p)
    self.assertEqual(runner.alg.learning_rate, runner.alg.optimizer.param_groups[0]['lr'])

  def test_warm_start_retains_actor_critic_but_starts_new_training_clock(self):
    runner = self.runner()
    env = SimpleNamespace(common_step_counter=720024, _sim_step_counter=2880096)
    runner.env = SimpleNamespace(unwrapped=env, reset=lambda: None)
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'seed.pt'; torch.save({'infos': {}}, p)
      with patch('src.tasks.transition_t.rl.TransitionOnPolicyRunner.load', return_value={}) as parent:
        runner.warm_start(p)
      assert parent.call_args.kwargs['load_cfg'] == {'actor': True, 'critic': True}
    assert env.common_step_counter == 0
    assert env._sim_step_counter == 0
    assert runner.current_learning_iteration == 0


if __name__ == '__main__':
  unittest.main()
