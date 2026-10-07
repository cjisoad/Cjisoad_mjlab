"""Evaluation captures terminal physics and excludes subsequent autoresets."""
import math
import importlib.util
from pathlib import Path
import unittest

import torch

spec=importlib.util.spec_from_file_location('bridge_biped_evaluation',
  Path(__file__).resolve().parents[1]/'scripts/evaluate_bridge_biped.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
FirstOutcomeStats=evaluation.FirstOutcomeStats
summarize_records=evaluation.summarize_records


def snapshot(fr,events):
  n=len(events)
  return dict(fr_error=torch.full((n,3),float(fr)),contacts=torch.tensor([[False,False,True,True]]*n),
    gravity=torch.tensor([[-1.,0.,0.]]*n),height=torch.full((n,),.52),
    gate_conditions={'height':torch.ones(n,dtype=torch.bool),'front_off':torch.ones(n,dtype=torch.bool)},
    physical_failure=torch.zeros(n,dtype=torch.bool),
    events=torch.tensor(events),reasons=torch.tensor([6 if e==1 else 4 if e==-1 else 0 for e in events]))


class BipedEvaluationTests(unittest.TestCase):
  def test_static_baseline_reports_only_actual_standing_verification(self):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
    torch.set_num_threads(1)
    cfg=bridge_env_cfg(play=True); cfg.scene.num_envs=2
    cfg.commands['twist'].debug_vis=False
    env=ManagerBasedRlEnv(cfg,device='cpu')
    try:
      env.reset()
      bank,moving=evaluation.initialize_static_baseline(env)
      self.assertFalse(moving.any(),'Static zero-velocity standing baseline must not label moving trials')
      self.assertNotEqual(int(bank[0]),int(bank[1]))
      term=env.command_manager.get_term('twist')
      torch.testing.assert_close(term.command[:,:3],torch.zeros(2,3))
      torch.testing.assert_close(term.progress,torch.ones(2))
    finally: env.close()

  def test_first_outcome_keeps_terminal_sample_and_ignores_autoreset(self):
    stats=FirstOutcomeStats(2,device='cpu')
    stats.observe(**snapshot(1.,[1,0]),dt=.02)
    stats.observe(**snapshot(3.,[-1,-1]),dt=.02)
    self.assertEqual(stats.samples.tolist(),[1,2])
    self.assertEqual(stats.outcomes.tolist(),[1,-1])
    self.assertEqual(stats.reasons.tolist(),[6,4])
    self.assertFalse(stats.active.any())
    records=stats.records()
    self.assertAlmostEqual(records[0]['fr_rms_error_m'],math.sqrt(3.))
    self.assertAlmostEqual(records[1]['fr_rms_error_m'],math.sqrt(15.))
    self.assertAlmostEqual(records[0]['duration_s'],.02,places=6)
    self.assertEqual(records[0]['contact_fraction_FL_FR_RL_RR'],[0.,0.,1.,1.])

  def test_summary_separates_stationary_and_moving_for_each_end_height(self):
    records=[dict(command_class=kind,h_end_m=height,outcome=event,
      fr_rms_error_m=.04,duration_s=2.5,physical_failure_observed=False,
      reached_validation_horizon=True,state_gate_failure_fraction={'height':0.},reason='tail_success')
      for kind,height,event in (('stationary',.301,1),('moving',.301,-1),('moving',.322,1))]
    groups=summarize_records(records)
    self.assertEqual(len(groups['by_command_class_and_end_height']),3)
    self.assertEqual(groups['by_command_class']['stationary']['success_rate'],1.)
    self.assertEqual(groups['by_command_class']['moving']['success_rate'],.5)


if __name__=='__main__':
  unittest.main()
