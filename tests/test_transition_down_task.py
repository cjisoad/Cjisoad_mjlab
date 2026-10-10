from dataclasses import asdict
import unittest
from src.tasks.transition_t.env_cfg import transition_env_cfg
from src.tasks.transition_t.rl import transition_ppo_runner_cfg

class DownTaskTests(unittest.TestCase):
  def test_original_physics_actions_observations_ppo_retained(self):
    from src.tasks.transition_down_t.env_cfg import down_env_cfg
    from src.tasks.transition_down_t.rl import down_ppo_runner_cfg
    from src.tasks.transition_down_t.data import DEFAULT_MOTION_PATH
    original=transition_env_cfg();cfg=down_env_cfg()
    from src.tasks.transition_t.entry_rl import _recipe_value
    self.assertEqual(_recipe_value(cfg.scene),_recipe_value(original.scene))
    self.assertEqual(cfg.sim,original.sim)
    self.assertEqual(cfg.actions,original.actions);self.assertEqual(cfg.events,original.events)
    self.assertEqual(cfg.observations,original.observations);self.assertEqual(cfg.terminations,original.terminations)
    for name in original.rewards:
      if name!='standing_hold':self.assertEqual(cfg.rewards[name],original.rewards[name])
    self.assertEqual(cfg.commands['motion'].motion_file,str(DEFAULT_MOTION_PATH))
    a=asdict(transition_ppo_runner_cfg());b=asdict(down_ppo_runner_cfg())
    for k in ('class_name','experiment_name','save_interval'):a.pop(k);b.pop(k)
    self.assertEqual(a,b)
  def test_independent_registration(self):
    import src.tasks
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.transition_down_t.commands import DownMotionCfg
    self.assertIsInstance(load_env_cfg('Transition-Down-Go2-v0').commands['motion'],DownMotionCfg)
    self.assertNotIsInstance(load_env_cfg('Transition-Go2-v0').commands['motion'],DownMotionCfg)
