"""Dense actor MoE behavior and read-only gradient diagnostics."""
import copy
import importlib
import unittest
from dataclasses import asdict
import torch
from tensordict import TensorDict
from tests import test_leg_manip_gradient_diagnostics as diagnostic_tests


class DenseMoETests(unittest.TestCase):
  def module(self):
    try:
      return importlib.import_module('src.tasks.moe_leg_manip.rl.moe')
    except ModuleNotFoundError:
      self.fail('Dense actor MoE integration is missing')

  def actor(self):
    module = self.module()
    obs = TensorDict({'actor': torch.randn(8, 51), 'critic': torch.randn(8, 131)}, batch_size=[8])
    actor = module.DenseMoEActor(obs, {'actor': ['actor'], 'critic': ['critic']}, 'actor', 12,
      hidden_dims=(16, 8), gate_hidden_dims=(8,), distribution_cfg={
        'class_name': 'src.tasks.moe_leg_manip.rl.distribution:StanceGaussianDistribution',
        'init_std': .2, 'std_type': 'scalar'})
    return actor, obs

  def test_dense_learned_four_experts_all_contribute_and_receive_gradients(self):
    actor, obs = self.actor()
    mean = actor(obs)
    network = actor.mlp
    weights = network._last_gate_weights.squeeze(1)
    self.assertFalse(network.use_explicit_expert)
    self.assertFalse(network.is_sparse)
    self.assertEqual(len(network.experts), 4)
    self.assertEqual(mean.shape, (8, 12))
    self.assertTrue((weights > 0).all())
    torch.testing.assert_close(weights.sum(-1), torch.ones(8))
    torch.testing.assert_close(mean, (network._last_component_outputs * weights.unsqueeze(1)).sum(-1))
    mean.square().mean().backward()
    for part in [network.gate, *network.experts]:
      self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None), 0.)
    self.assertEqual(actor.obs_dim, 51)
    self.assertEqual(actor.distribution.std_param.shape, (12,))
    actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.output_std, torch.full((8, 12), .2))

  def test_actor_only_config_preserves_legacy_and_training_hyperparameters(self):
    self.module()
    from src.tasks.moe_leg_manip.rl import leg_manip_ppo_runner_cfg
    from mjlab.tasks.registry import load_rl_cfg, load_env_cfg
    original = asdict(leg_manip_ppo_runner_cfg())
    moe = asdict(load_rl_cfg('moe_leg_manip'))
    self.assertEqual(moe['critic'], original['critic'])
    self.assertEqual(moe['algorithm'], original['algorithm'])
    self.assertEqual(moe['actor']['distribution_cfg'], original['actor']['distribution_cfg'])
    self.assertEqual(moe['actor']['num_experts'], 4)
    self.assertEqual(moe['actor']['top_k'], -1)
    self.assertFalse(moe['actor']['use_explicit_expert'])
    self.assertFalse(moe['resume'])
    self.assertEqual(load_rl_cfg('leg_manip').actor.class_name, 'MLPModel')
    self.assertEqual(load_env_cfg('moe_leg_manip').commands['twist'].initial_stage, 0)

  def ppo(self):
    ppo = diagnostic_tests.GradientDiagnosticsTests().make_ppo()
    actor, _ = self.actor()
    from src.tasks.moe_leg_manip.rl.ppo import LegManipPPO
    ppo = LegManipPPO(actor, ppo.critic, ppo.storage, num_learning_epochs=1, num_mini_batches=1,
      sym_coef=.7, schedule='fixed', desired_kl=None)
    with torch.no_grad():
      flat = ppo.storage.observations.flatten(0, 1)
      mean = actor(flat)
      actor.distribution.update(mean)
      ppo.storage.actions_log_prob[:] = actor.get_output_log_prob(ppo.storage.actions.flatten(0, 1)).reshape(2, 8, 1)
      ppo.storage.distribution_params = tuple(p.reshape(2, 8, -1).detach().clone() for p in actor.output_distribution_params)
    return ppo

  def test_diagnostics_log_routing_and_each_expert_without_changing_caches(self):
    self.module()
    from src.tasks.moe_leg_manip.rl.gradient_diagnostics import GradientConflictDiagnostics
    ppo = self.ppo()
    caches = {n: getattr(ppo.actor.mlp, n) for n in (
      '_last_gate_weights', '_last_unmasked_gate_weights', '_last_component_outputs')}
    distribution = ppo.actor.distribution._distribution
    rng = torch.get_rng_state().clone()
    record = GradientConflictDiagnostics(interval=1, max_samples=3)
    record.groups = list(diagnostic_tests.GradientDiagnosticsTests().labels())
    metrics = record.evaluate(ppo)
    for group in ('quad', 'biped', 'quad_stand', 'biped_stand', 'quad_moving', 'biped_moving'):
      self.assertAlmostEqual(sum(metrics[f'moe_routing/{group}/expert_{i}'] for i in range(4)), 1., places=5)
    for part in ('router', 'expert_0', 'expert_1', 'expert_2', 'expert_3'):
      self.assertEqual(metrics[f'quad_biped/policy/{part}/valid'], 1.)
      self.assertTrue(-1. <= metrics[f'quad_biped/policy/{part}/cosine'] <= 1.)
    for n, v in caches.items(): self.assertIs(getattr(ppo.actor.mlp, n), v)
    self.assertIs(distribution, ppo.actor.distribution._distribution)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

  def test_moe_ppo_update_is_identical_with_diagnostics_enabled(self):
    self.module()
    baseline = self.ppo()
    # Remove transient autograd caches before deepcopy; recompute during update.
    for n in ('_last_gate_weights', '_last_unmasked_gate_weights', '_last_component_outputs'):
      setattr(baseline.actor.mlp, n, torch.empty(0))
    observed = copy.deepcopy(baseline)
    baseline.gradient_diagnostics.interval = 0
    observed.gradient_diagnostics.interval = 1
    observed.gradient_diagnostics.groups = list(diagnostic_tests.GradientDiagnosticsTests().labels())
    rng = torch.get_rng_state().clone()
    first = baseline.update()
    torch.set_rng_state(rng)
    second = observed.update()
    self.assertEqual(first, second)
    for a, b in zip(baseline.actor.parameters(), observed.actor.parameters()):
      torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(baseline.critic.parameters(), observed.critic.parameters()):
      torch.testing.assert_close(a, b, rtol=0, atol=0)

  def test_checkpoint_detection_preserves_legacy_and_supports_moe(self):
    module = self.module()
    actor, obs = self.actor()
    checkpoint = {'actor_state_dict': actor.state_dict()}
    self.assertTrue(module.checkpoint_uses_moe(checkpoint))
    self.assertFalse(module.checkpoint_uses_moe({'actor_state_dict': {'mlp.0.weight': torch.zeros(1)}}))
    reloaded, _ = self.actor()
    reloaded.load_state_dict(actor.state_dict())
    torch.testing.assert_close(actor(obs), reloaded(obs), rtol=0, atol=0)
    exported = actor.as_jit()
    torch.testing.assert_close(exported(obs['actor']), actor(obs))


if __name__ == '__main__': unittest.main()
