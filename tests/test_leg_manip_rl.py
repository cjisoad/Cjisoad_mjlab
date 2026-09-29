"""Runner configuration, fused PPO validation and task registration contracts."""

from types import SimpleNamespace
import unittest


class RunnerCfgTests(unittest.TestCase):
  def test_runner_cfg_matches_fused_task(self):
    from src.tasks.leg_manip.rl import leg_manip_ppo_runner_cfg
    cfg = leg_manip_ppo_runner_cfg()
    self.assertEqual(cfg.experiment_name, 'leg_manip')
    self.assertFalse(cfg.actor.obs_normalization)
    self.assertFalse(cfg.critic.obs_normalization)
    self.assertEqual(tuple(cfg.actor.hidden_dims), (512, 256, 128))
    self.assertEqual(cfg.algorithm.class_name, 'src.tasks.leg_manip.rl.ppo:LegManipPPO')
    self.assertEqual(cfg.algorithm.sym_coef, 1.0)
    self.assertEqual(cfg.max_iterations, 15000)
    self.assertEqual(cfg.clip_actions, 10.)
    self.assertEqual(cfg.logger, 'wandb')
    self.assertFalse(cfg.upload_model)
    self.assertEqual(cfg.obs_groups, {'actor': ('actor',), 'critic': ('critic',)})


class LegManipPpoValidationTests(unittest.TestCase):
  def make_ppo_like(self, actor_obs=48, critic_obs=131, norm=False, output=12):
    from src.tasks.leg_manip.rl.ppo import LegManipPPO
    holder = SimpleNamespace(
      actor=SimpleNamespace(is_recurrent=False, obs_dim=actor_obs, obs_groups=('actor',),
                            obs_normalization=norm,
                            distribution=SimpleNamespace(output_dim=output)),
      critic=SimpleNamespace(is_recurrent=False, obs_dim=critic_obs, obs_normalization=norm))
    LegManipPPO.validate(holder)

  def test_accepts_merged_critic_dimension(self):
    self.make_ppo_like(critic_obs=131)

  def test_rejects_non_48_actor(self):
    with self.assertRaises(ValueError):
      self.make_ppo_like(actor_obs=45)

  def test_rejects_obs_normalization(self):
    with self.assertRaises(ValueError):
      self.make_ppo_like(norm=True)

  def test_rejects_non_12_outputs(self):
    with self.assertRaises(ValueError):
      self.make_ppo_like(output=10)

  def test_subclass_of_pedipulation_ppo(self):
    from src.tasks.leg_manip.rl.ppo import LegManipPPO
    from src.tasks.pedipulation.rl.ppo import PedipulationPPO
    self.assertTrue(issubclass(LegManipPPO, PedipulationPPO))


class RegistrationTests(unittest.TestCase):
  def test_leg_manip_and_source_tasks_registered(self):
    import src.tasks  # noqa: F401 — triggers package auto-import
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
    tasks = list_tasks()
    for task_id in ('leg_manip', 'loco_pedipulation', 'Pedipulation'):
      self.assertIn(task_id, tasks)
    from src.tasks.leg_manip.mdp.commands import LegManipCommandCfg
    from src.tasks.leg_manip.rl import LegManipOnPolicyRunner
    self.assertIs(load_runner_cls('leg_manip'), LegManipOnPolicyRunner)
    self.assertIsInstance(load_env_cfg('leg_manip').commands['twist'], LegManipCommandCfg)

  def test_runner_preserves_curriculum_persistence(self):
    from src.tasks.leg_manip.rl import LegManipOnPolicyRunner
    from src.tasks.loco_pedipulation.rl import LocoPedipulationOnPolicyRunner
    self.assertTrue(issubclass(LegManipOnPolicyRunner, LocoPedipulationOnPolicyRunner))


if __name__ == '__main__':
  unittest.main()
