"""Gradient diagnostics must observe PPO without changing its updates."""
import copy, importlib, unittest
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


class GradientDiagnosticsTests(unittest.TestCase):
  def module(self):
    try:
      return importlib.import_module('src.tasks.leg_manip.rl.gradient_diagnostics')
    except ModuleNotFoundError:
      self.fail('Missing gradient diagnostics module')

  def make_ppo(self):
    from src.tasks.leg_manip.rl.ppo import LegManipPPO
    obs=TensorDict({'actor':torch.zeros(8,51),'critic':torch.zeros(8,131)},batch_size=[8])
    groups={'actor':['actor'],'critic':['critic']}
    actor=MLPModel(obs,groups,'actor',12,hidden_dims=(8,),distribution_cfg={
      'class_name':'src.tasks.leg_manip.rl.distribution:StanceGaussianDistribution',
      'init_std':.2,'std_type':'scalar'})
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=(8,))
    storage=RolloutStorage('rl',8,2,obs,[12])
    ppo=LegManipPPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=1,
      sym_coef=0.,schedule='fixed',desired_kl=None)
    with torch.no_grad():
      for p in actor.mlp.parameters(): p.zero_()
      storage.observations['actor'][:,:,0]=1.
      storage.actions[:,:4]=.1; storage.actions[:,4:]=-.1
      actor.distribution.update(actor(storage.observations.flatten(0,1)))
      storage.actions_log_prob[:]=actor.get_output_log_prob(storage.actions.flatten(0,1)).reshape(2,8,1)
      storage.advantages.fill_(1.)
      storage.distribution_params=tuple(p.reshape(2,8,-1).detach().clone() for p in actor.output_distribution_params)
      storage.step=2
    return ppo

  def labels(self):
    return torch.tensor([[0,0,2,2,1,1,3,3]]*2)

  def test_observed_opposite_and_aligned_policy_gradients(self):
    module=self.module(); ppo=self.make_ppo()
    record=module.GradientConflictDiagnostics(interval=1,max_samples=64)
    record.groups=list(self.labels())
    metrics=record.evaluate(ppo)
    self.assertAlmostEqual(metrics['quad_biped/policy/cosine'],-1.,places=5)
    self.assertEqual(metrics['quad_biped/policy/negative_fraction'],1.)
    record.groups=list(self.labels())
    ppo.storage.actions.fill_(.1)
    with torch.no_grad():
      ppo.actor.distribution.update(ppo.actor(ppo.storage.observations.flatten(0,1)))
      ppo.storage.actions_log_prob[:]=ppo.actor.get_output_log_prob(ppo.storage.actions.flatten(0,1)).reshape(2,8,1)
    metrics=record.evaluate(ppo)
    self.assertAlmostEqual(metrics['quad_biped/policy/cosine'],1.,places=5)
    self.assertEqual(metrics['quad_biped/policy/negative_fraction'],.5)

  def test_command_mode_labels_exclude_reward_transitions(self):
    module=self.module()
    from src.tasks.leg_manip.mdp.commands import QUAD,HOLD,REACH,LOWER
    term=SimpleNamespace(mode=torch.tensor([False,True,False,True,True,False]),
      phase=torch.tensor([QUAD,HOLD,QUAD,HOLD,REACH,LOWER]),
      group_weight=torch.tensor([0.,1.,0.,1.,.5,.5]),
      command=torch.zeros(6,6))
    term.command[2,0]=.12;term.command[3,2]=.12
    torch.testing.assert_close(module.classify_command_groups(term),torch.tensor([0,1,2,3,-1,-1]))

  def test_diagnostic_preserves_rng_existing_grads_distribution_and_storage(self):
    module=self.module();ppo=self.make_ppo()
    record=module.GradientConflictDiagnostics(interval=1,max_samples=3)
    record.groups=list(self.labels())
    for p in ppo.actor.parameters():p.grad=torch.ones_like(p)
    states={k:v.clone() for k,v in ppo.actor.state_dict().items()}
    grads=[p.grad.clone() for p in ppo.actor.parameters()]
    rng=torch.get_rng_state().clone()
    distribution=ppo.actor.distribution._distribution
    advantages=ppo.storage.advantages.clone()
    record.evaluate(ppo)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    self.assertIs(ppo.actor.distribution._distribution,distribution)
    torch.testing.assert_close(ppo.storage.advantages,advantages,rtol=0,atol=0)
    for k,v in ppo.actor.state_dict().items():torch.testing.assert_close(v,states[k],rtol=0,atol=0)
    for p,g in zip(ppo.actor.parameters(),grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)

  def test_missing_and_zero_gradient_groups_do_not_report_false_cosines(self):
    module=self.module();ppo=self.make_ppo()
    record=module.GradientConflictDiagnostics(interval=1,max_samples=64)
    record.groups=[torch.zeros(8,dtype=torch.long)]*2
    metrics=record.evaluate(ppo)
    self.assertEqual(metrics['quad_biped/policy/valid'],0.)
    self.assertNotIn('quad_biped/policy/cosine',metrics)
    record.groups=list(self.labels());ppo.storage.advantages.zero_()
    metrics=record.evaluate(ppo)
    self.assertEqual(metrics['quad_biped/policy/valid'],0.)
    self.assertNotIn('quad_biped/policy/negative_fraction',metrics)

  def test_ppo_update_is_identical_with_diagnostics_enabled(self):
    self.module();baseline=self.make_ppo();baseline.sym_coef=.7
    with torch.no_grad():
      baseline.actor.mlp[-1].bias.copy_(torch.arange(12)/50.)
    observed=copy.deepcopy(baseline)
    baseline.gradient_diagnostics.interval=0
    observed.gradient_diagnostics.interval=1
    observed.gradient_diagnostics.groups=list(self.labels())
    rng=torch.get_rng_state().clone()
    first=baseline.update()
    torch.set_rng_state(rng)
    second=observed.update()
    self.assertEqual(first,second)
    for a,b in zip(baseline.actor.parameters(),observed.actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(baseline.critic.parameters(),observed.critic.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    self.assertIsNotNone(observed.last_gradient_diagnostics)

  def test_labels_are_captured_before_the_environment_changes_mode(self):
    self.module();ppo=self.make_ppo();ppo.storage.clear()
    from src.tasks.leg_manip.mdp.commands import QUAD,HOLD
    term=SimpleNamespace(mode=torch.zeros(8,dtype=torch.bool),
      phase=torch.full((8,),QUAD),group_weight=torch.zeros(8),command=torch.zeros(8,6))
    ppo.gradient_command_term=term
    with torch.inference_mode():
      ppo.act(ppo.storage.observations[0])
      term.mode.fill_(True);term.phase.fill_(HOLD);term.group_weight.fill_(1.)
    torch.testing.assert_close(ppo.gradient_diagnostics.groups[0],torch.zeros(8,dtype=torch.long))


if __name__=='__main__':unittest.main()
