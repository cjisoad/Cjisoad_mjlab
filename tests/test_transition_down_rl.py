from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch

class DownContracts(unittest.TestCase):
  def test_changed_direction_teacher_alignment_and_reward_rejected_before_weights(self):
    from src.tasks.transition_down_t.rl import DownOnPolicyRunner
    runner=object.__new__(DownOnPolicyRunner)
    contract=dict(task='transition_down_t',direction='biped_to_tripod',alignment='teacher_heading',teachers={'biped':'accepted'},rewards={'hold':.1})
    runner._contract=lambda:contract
    runner.alg=SimpleNamespace(actor=torch.nn.Module(),critic=torch.nn.Module())
    runner.term=SimpleNamespace(bin_failed_count=torch.zeros(3),_current_bin_failed=torch.zeros(3))
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/'model.pt'
      for key,value in [('task','transition_t'),('direction','tripod_to_biped'),('alignment','euler'),('teachers',{}),('rewards',{})]:
        torch.save({'infos':{'transition_t_contract':{**contract,key:value}}},path)
        with patch('mjlab.rl.MjlabOnPolicyRunner.load') as parent:
          with self.assertRaises(ValueError):runner.load(path,load_cfg={'actor':True})
          parent.assert_not_called()
