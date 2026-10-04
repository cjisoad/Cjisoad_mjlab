"""MLP baseline and MoE task own independent training and environment code."""
import unittest
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls


class TaskSplitTests(unittest.TestCase):
  def test_new_task_uses_four_dense_actor_experts_and_original_mlp_critic(self):
    self.assertIn('moe_leg_manip', list_tasks())
    mlp = load_rl_cfg('leg_manip')
    moe = load_rl_cfg('moe_leg_manip')
    self.assertEqual(mlp.actor.class_name, 'MLPModel')
    self.assertEqual(moe.actor.class_name, 'src.tasks.moe_leg_manip.rl.moe:DenseMoEActor')
    self.assertEqual(moe.actor.num_experts, 4)
    self.assertEqual(moe.actor.top_k, -1)
    self.assertFalse(moe.actor.use_explicit_expert)
    self.assertEqual(moe.critic, mlp.critic)
    self.assertEqual(moe.experiment_name, 'moe_leg_manip')
    self.assertEqual(mlp.experiment_name, 'leg_manip')

  def test_task_code_and_mutable_settings_are_independent(self):
    self.assertIn('moe_leg_manip', list_tasks())
    mlp = load_env_cfg('leg_manip')
    moe = load_env_cfg('moe_leg_manip')
    self.assertEqual(type(mlp.commands['twist']).__module__, 'src.tasks.leg_manip.mdp.commands')
    self.assertEqual(type(moe.commands['twist']).__module__, 'src.tasks.moe_leg_manip.mdp.commands')
    self.assertEqual(load_runner_cls('leg_manip').__module__, 'src.tasks.leg_manip.rl')
    self.assertEqual(load_runner_cls('moe_leg_manip').__module__, 'src.tasks.moe_leg_manip.rl')
    self.assertIsNot(mlp.commands['twist'], moe.commands['twist'])
    original = mlp.commands['twist'].stance_eval_interval
    moe.commands['twist'].stance_eval_interval = 123
    self.assertEqual(load_env_cfg('leg_manip').commands['twist'].stance_eval_interval, original)
    for key in mlp.rewards:
      old_reward, new_reward = mlp.rewards[key], moe.rewards[key]
      self.assertEqual(old_reward.weight, new_reward.weight)
      module = old_reward.func.__module__
      if module.startswith('src.tasks.leg_manip.'):
        self.assertEqual(new_reward.func.__module__, module.replace('src.tasks.leg_manip.', 'src.tasks.moe_leg_manip.'))


if __name__ == '__main__':
  unittest.main()
