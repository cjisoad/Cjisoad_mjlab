import unittest

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from src.tasks.pedipulation_bridge_t.rollout import BridgeRolloutStorage


def transition(actions, rewards, dones=None):
  n = len(actions)
  tr = RolloutStorage.Transition()
  tr.observations = TensorDict({'actor': torch.tensor(actions).float().view(n, 1)}, [n])
  tr.actions = torch.tensor(actions).float().view(n, 1)
  tr.rewards = torch.tensor(rewards).float()
  tr.dones = torch.zeros(n) if dones is None else torch.tensor(dones)
  tr.values = torch.zeros(n, 1)
  tr.actions_log_prob = torch.zeros(n)
  tr.distribution_params = (tr.actions.clone(), torch.ones(n, 1))
  return tr


class BridgeRolloutTests(unittest.TestCase):
  def storage(self, n=3, capacity=4):
    return BridgeRolloutStorage('rl', n, capacity,
      TensorDict({'actor': torch.zeros(n, 1)}, [n]), [1])

  def test_experts_are_excluded_and_lengths_pack_independently(self):
    st = self.storage()
    st.collection_mask = torch.tensor([False, True, False])
    st.add_transition(transition([999, 10, 999], [999, 1, 999]))
    st.collection_mask = torch.tensor([True, True, False])
    st.add_transition(transition([20, 11, 999], [2, 3, 999]))
    self.assertEqual(st.lengths.tolist(), [1, 2, 0])
    st.finish(torch.tensor([0, 1]), torch.tensor([0., 0.]), torch.tensor([0, 0]), gamma=.9)
    st.compute_closed_returns(gamma=.9, lam=1., normalize=False)
    self.assertAlmostEqual(st.returns[0, 1, 0].item(), 3.7, places=5)
    batches = list(st.mini_batch_generator(2, 1))
    self.assertEqual(sum(len(b.actions) for b in batches), 3)
    self.assertEqual(sorted(torch.cat([b.actions[:, 0] for b in batches]).tolist()), [10, 11, 20])

  def test_delayed_tail_feedback_lands_on_last_learned_action(self):
    st = self.storage(n=2)
    st.collection_mask = torch.tensor([True, False])
    st.add_transition(transition([1, 999], [1, 999]))
    st.add_transition(transition([2, 999], [2, 999]))
    st.finish(torch.tensor([0, 1]), torch.tensor([-10., 100.]), torch.tensor([2, 0]), gamma=.5)
    self.assertAlmostEqual(st.rewards[1, 0, 0].item(), -.5)
    self.assertEqual(st.dones[1, 0, 0].item(), 1)
    st.compute_closed_returns(gamma=.5, lam=1., normalize=False)
    self.assertAlmostEqual(st.returns[0, 0, 0].item(), .75)
    self.assertEqual(st.valid.sum().item(), 2)

  def test_refuses_unresolved_handoff_and_double_feedback(self):
    st = self.storage(n=1)
    st.collection_mask[:] = True
    st.add_transition(transition([1], [0]))
    with self.assertRaisesRegex(RuntimeError, 'unresolved'):
      st.compute_closed_returns(.99, .95)
    st.finish(torch.tensor([0]), torch.tensor([1.]), torch.tensor([1]), gamma=.99)
    with self.assertRaisesRegex(RuntimeError, 'already'):
      st.finish(torch.tensor([0]), torch.tensor([1.]), torch.tensor([1]), gamma=.99)

  def test_measured_outcome_has_equal_strength_for_different_validation_lengths(self):
    st=self.storage(n=2)
    st.collection_mask[:]=True
    st.add_transition(transition([1,2],[0,0]))
    st.finish(torch.tensor([0,1]),torch.tensor([20.,-20.]),torch.tensor([125,10]),
      gamma=.99,discount_tail=False)
    torch.testing.assert_close(st.rewards[0,:,0],torch.tensor([20.,-20.]))

  def test_clear_resets_collection_and_preserves_capacity(self):
    st = self.storage(n=1, capacity=1)
    st.collection_mask[:] = True
    st.add_transition(transition([1], [0]))
    with self.assertRaises(OverflowError):
      st.add_transition(transition([1], [0]))
    st.clear()
    self.assertEqual(st.lengths.sum().item(), 0)
    self.assertFalse(st.valid.any())
    self.assertFalse(st.collection_mask.any())


if __name__ == '__main__':
  unittest.main()
