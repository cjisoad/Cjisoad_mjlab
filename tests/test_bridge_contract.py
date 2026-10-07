import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
from src.tasks.pedipulation_bridge_t.rl import BridgeOnPolicyRunner


def contract_runner(play=False):
  cfg=bridge_env_cfg(play=play)
  runner=BridgeOnPolicyRunner.__new__(BridgeOnPolicyRunner)
  runner.env=SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg,step_dt=cfg.decimation*cfg.sim.mujoco.timestep),
    clip_actions=10.)
  runner.term=SimpleNamespace(cfg=cfg.commands['twist'])
  runner.action=SimpleNamespace(cfg=cfg.actions['joint_pos'],default_angles=torch.zeros(1,12))
  runner.experts=SimpleNamespace(contracts={'quadruped':'same','biped':'same'})
  return runner


class BridgeContractTests(unittest.TestCase):
  def test_reference_and_rewards_version_the_objective(self):
    runner=contract_runner()
    original=runner._contract()
    self.assertEqual(original['version'],2)
    self.assertEqual((original['actor_dim'],original['critic_dim']),(73,117))
    for key in ('reference_speed_range','reference_blend_seconds','fr_tracking_scale',
                'initial_impulse_probability','initial_linear_impulse','handoff_linear_speed'):
      self.assertIn(key,original['command_parameters'])
    self.assertIn('reward_terms',original)
    runner.env.unwrapped.cfg.rewards['orientation'].weight+=1.
    self.assertNotEqual(original,runner._contract())

  def test_play_and_training_contracts_are_compatible(self):
    self.assertEqual(contract_runner()._contract(), contract_runner(play=True)._contract())

  def test_used_inherited_controls_and_physics_are_in_contract(self):
    runner=contract_runner()
    original=runner._contract()
    for key in ('reach_time','velocity_ramp_time'):
      self.assertIn(key,original['command_parameters'])
      before=getattr(runner.term.cfg,key)
      setattr(runner.term.cfg,key,before+.01)
      self.assertNotEqual(original,runner._contract())
      setattr(runner.term.cfg,key,before)
    runner.env.unwrapped.step_dt += .001
    self.assertNotEqual(original,runner._contract())

  def test_failure_sensor_semantics_are_part_of_checkpoint_contract(self):
    runner=contract_runner()
    original=runner._contract()
    sensor=next(s for s in runner.env.unwrapped.cfg.scene.sensors if s.name=='bridge_self_contact')
    sensor.secondary_policy='any'
    self.assertNotEqual(original,runner._contract())

  def test_legacy_bridge_checkpoint_is_rejected_before_weight_loading(self):
    runner=contract_runner()
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
      checkpoint=Path(directory)/'legacy.pt'
      legacy=runner._contract(); legacy['version']=1
      legacy['actor_dim']=51; legacy['critic_dim']=95
      torch.save({'infos':{'bridge_contract':legacy}},checkpoint)
      with self.assertRaisesRegex(ValueError,'contract differs'): runner.load(str(checkpoint))

  def test_load_rejects_protocol_change_before_loading_weights(self):
    runner=contract_runner()
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
      checkpoint=Path(directory)/'contract_only.pt'
      torch.save({'infos':{'bridge_contract':runner._contract()}},checkpoint)
      runner.term.cfg.reach_time += .1
      with self.assertRaisesRegex(ValueError,'contract differs'):
        runner.load(str(checkpoint))


if __name__=='__main__': unittest.main()
