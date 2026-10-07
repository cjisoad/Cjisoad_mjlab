import unittest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import src.tasks


class BridgeRegistrationTests(unittest.TestCase):
  def test_bridge_has_training_and_play_configs(self):
    cfg = load_env_cfg('pedipulation_bridge_t')
    play = load_env_cfg('pedipulation_bridge_t', play=True)
    self.assertEqual(cfg.commands['twist'].verify_seconds, 2.5)
    self.assertEqual(cfg.actions['joint_pos'].scale, .25)
    self.assertEqual(cfg.actions['joint_pos'].delay, False)
    self.assertEqual(cfg.commands['twist'].start_height_range, play.commands['twist'].start_height_range)
    self.assertEqual(load_runner_cls('pedipulation_bridge_t').__name__, 'BridgeOnPolicyRunner')
    self.assertEqual(load_rl_cfg('pedipulation_bridge_t').experiment_name, 'pedipulation_bridge_t')


if __name__ == '__main__': unittest.main()
