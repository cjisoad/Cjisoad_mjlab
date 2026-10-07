import unittest
from dataclasses import asdict
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from src.tasks.pedipulation_bridge_t.rl import BridgePPO, bridge_ppo_runner_cfg
from src.tasks.teacher_common.rl import teacher_ppo_runner_cfg
from src.tasks.pedipulation_bridge_t import rl


class BridgePPOTests(unittest.TestCase):
  def test_expanded_actor_initial_mean_matches_frozen_source(self):
    self.assertTrue(callable(getattr(rl,'initialize_guided_actor',None)))
    source=torch.nn.Sequential(torch.nn.Linear(51,512),torch.nn.ELU(),
      torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),
      torch.nn.ELU(),torch.nn.Linear(128,12))
    guided=torch.nn.Sequential(torch.nn.Linear(84,512),torch.nn.ELU(),
      torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),
      torch.nn.ELU(),torch.nn.Linear(128,12))
    rl.initialize_guided_actor(guided,source)
    original=torch.randn(4,51)
    torch.testing.assert_close(guided(torch.cat((original,torch.randn(4,33)),-1)),
      source(original),rtol=1e-5,atol=1e-6)
    self.assertFalse(guided[0].weight[:,51:].any())

  def test_settings_use_old_tracking_recipe(self):
    old = asdict(teacher_ppo_runner_cfg('loco_pedipulation_t'))
    new = asdict(bridge_ppo_runner_cfg())
    self.assertEqual(old['critic'], new['critic'])
    self.assertEqual(new['actor']['distribution_cfg']['init_std'],.5)
    self.assertEqual(new['algorithm']['learning_rate'],3e-4)
    self.assertEqual(new['algorithm']['entropy_coef'],.005)
    self.assertEqual(new['algorithm']['gamma'],.995)
    self.assertEqual(new['num_steps_per_env'], 24)
    self.assertEqual(new['clip_actions'], 10.)

  def test_closed_packed_episodes_update_real_ppo(self):
    torch.manual_seed(123)
    obs = TensorDict({'actor':torch.randn(3, 51),'critic':torch.randn(3, 95)}, [3])
    groups = {'actor':['actor'],'critic':['critic']}
    actor = MLPModel(obs, groups, 'actor', 12, hidden_dims=[16], activation='elu',
      obs_normalization=False, distribution_cfg={'class_name':'GaussianDistribution',
      'init_std':1., 'std_type':'scalar'})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[16], activation='elu',
      obs_normalization=False)
    alg = BridgePPO(actor, critic, RolloutStorage('rl',3,24,obs,[12]),
      max_bridge_steps=8, num_learning_epochs=2, num_mini_batches=2,
      schedule='fixed', device='cpu')
    before = [p.clone().detach() for p in actor.parameters()]
    with torch.inference_mode():
      for _ in range(3):
        alg.storage.collection_mask[:] = torch.tensor([True, False, True])
        alg.act(obs)
        alg.process_env_step(obs,torch.tensor([1.,999.,2.]),torch.zeros(3),{})
      alg.storage.finish(torch.arange(3),torch.tensor([10.,0.,-10.]),torch.tensor([2,0,1]),gamma=alg.gamma)
      alg.compute_returns(obs)
    self.assertEqual(alg.storage.valid.sum().item(),6)
    loss = alg.update()
    self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in loss.values()))
    self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,actor.parameters())))
    self.assertEqual(alg.storage.lengths.sum().item(),0)


if __name__ == '__main__': unittest.main()
