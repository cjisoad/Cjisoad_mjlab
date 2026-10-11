import unittest
import torch
from src.tasks.transition_down_v2.chain import live_chain_cfg,sample_biped_commands
from src.tasks.transition_down_v2.gates import GateConfig

class ChainSetup(unittest.TestCase):
    def test_full_horizon_and_owner_tracking(self):
        cfg=live_chain_cfg(GateConfig(),128,42)
        self.assertNotIn('time_out',cfg.terminations)
        for name in ('anchor_pos','anchor_ori','ee_body_pos'):
            self.assertEqual(cfg.terminations[name].func.__name__,'chain_failure')
        self.assertIn('physical_failure',cfg.terminations)
        self.assertIn('landing_safety',cfg.terminations)
    def test_documented_fresh_source_commands(self):
        torch.manual_seed(42)
        c=sample_biped_commands(128,'cpu')
        self.assertEqual(c.shape,(128,6))
        self.assertTrue(torch.equal(c[:64,3:],torch.tensor([.12,0.,.36]).expand(64,-1)))
        self.assertTrue((c[96:,:3].abs().sum(-1)>0).all())
        self.assertGreater(len(c.unique(dim=0)),64)
        self.assertTrue(((c[64:,3:]>=torch.tensor([.07,-.04,.30]))&(c[64:,3:]<=torch.tensor([.17,.04,.40]))).all())
