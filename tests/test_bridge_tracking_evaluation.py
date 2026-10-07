import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

spec=importlib.util.spec_from_file_location('tracking_evaluation',Path(__file__).resolve().parents[1]/'scripts/evaluate_bridge_tracking.py')
evaluation=importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)


class TrackingEvaluationTests(unittest.TestCase):
  def test_live_classes_have_exact_half_quota_and_real_handoff_flag(self):
    term=SimpleNamespace(course=SimpleNamespace(),num_envs=4,device='cpu',cfg=SimpleNamespace())
    evaluation.configure_case(term,'live_handoff')
    self.assertEqual(term.forced_moving.tolist(),[False,True,False,True])
    self.assertTrue(term.cfg.verify_handoff)
    self.assertEqual(term.forced_live_probability,1.)
    self.assertEqual(term.course.stage,2)

  def test_reference_case_does_not_verify_biped(self):
    term=SimpleNamespace(course=SimpleNamespace(),num_envs=4,device='cpu',cfg=SimpleNamespace())
    evaluation.configure_case(term,'nominal')
    self.assertFalse(term.cfg.verify_handoff)
    self.assertEqual(term.forced_bank,0)
    self.assertEqual(term.forced_live_probability,0.)
    self.assertFalse(term.forced_disturbed)

  def test_evaluation_error_restores_temporary_advance_hook(self):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
    cfg=bridge_env_cfg(play=True); cfg.scene.num_envs=2
    cfg.commands['twist'].debug_vis=False
    raw=ManagerBasedRlEnv(cfg,device='cpu')
    try:
      env=RslRlVecEnvWrapper(raw,clip_actions=10.)
      term=raw.command_manager.get_term('twist'); original=term.advance
      def failing_policy(obs): raise RuntimeError('policy sentinel')
      with self.assertRaisesRegex(RuntimeError,'policy sentinel'):
        evaluation.evaluate(env,failing_policy,1)
      self.assertEqual(term.advance,original)
    finally: raw.close()


if __name__=='__main__': unittest.main()
