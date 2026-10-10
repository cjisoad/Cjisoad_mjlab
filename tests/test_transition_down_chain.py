"""Live ownership switches preserve physical and actuator history."""
from types import SimpleNamespace
import unittest
import torch

class ChainSwitchTests(unittest.TestCase):
  def test_two_switches_do_not_write_execution_state(self):
    from src.tasks.transition_down_t.chain import switch_owner, execution_snapshot
    action=SimpleNamespace(raw_action=torch.randn(2,12),previous_action=torch.randn(2,12),
      previous_joint_vel=torch.randn(2,12),_requested_action=torch.randn(2,12),
      _previous_requested_action=torch.randn(2,12),target_limiter=SimpleNamespace(position=torch.randn(2,12),velocity=torch.randn(2,12)))
    manager=SimpleNamespace(get_term=lambda _:action,action=torch.randn(2,12),prev_action=torch.randn(2,12),prev_prev_action=torch.randn(2,12))
    motion=SimpleNamespace(active=torch.zeros(2,dtype=torch.bool),start_from_current_state=lambda _:None)
    twist=SimpleNamespace(owner=torch.tensor([3,3]),_command=torch.zeros(2,6))
    env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.randn(2,19),qvel=torch.randn(2,18))),
      action_manager=manager,command_manager=SimpleNamespace(get_term=lambda n:motion if n=='motion' else twist))
    before=execution_snapshot(env)
    for state in (2,0):
      report=switch_owner(env,torch.tensor([0]),state)
      self.assertTrue(all(value==0. for value in report.values()))
      for key,value in before.items():torch.testing.assert_close(execution_snapshot(env)[key],value,rtol=0,atol=0)
    self.assertEqual(twist.owner.tolist(),[0,3])
    torch.testing.assert_close(twist._command[0],torch.tensor([0.,0.,0.,.12,0.,.33]))

class ChainInitializationTests(unittest.TestCase):
  def test_runner_hidden_reset_precedes_forced_entry_and_command_restore(self):
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[1].joinpath('scripts/evaluate_transition_down_chain.py').read_text()
    tree=ast.parse(source)
    def lines(name):
      return [n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and
        ((isinstance(n.func,ast.Name) and n.func.id==name) or (isinstance(n.func,ast.Attribute) and n.func.attr==name))]
    self.assertLess(lines('RslRlVecEnvWrapper')[0],lines('reset')[0])
    self.assertLess(lines('load')[0],lines('reset')[0])
    self.assertIn("twist._command.copy_(motion.bank.data['command'][selected])",source)
