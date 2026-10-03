"""Stage-zero stance assessment and perturbation regression tests."""
import importlib
import unittest
from types import SimpleNamespace
import torch
from tests.test_leg_manip_commands import make_env
from src.tasks.leg_manip.rl import leg_manip_ppo_runner_cfg
from src.tasks.leg_manip.mdp.curriculums import manipulation_curriculum

class StageZeroTests(unittest.TestCase):
  def module(self, name):
    try:
      return importlib.import_module(name)
    except ModuleNotFoundError:
      self.fail(f'Missing stage-zero implementation: {name}')

  def test_training_window_cannot_promote_stage_zero(self):
    env, term = make_env(stage=0, min_stage_steps=100, min_curriculum_episodes=4)
    env.common_step_counter=100
    for _ in range(2):
      term.curriculum_window[:] = torch.tensor((200.,190.,10.))
      term.group_survival[:] = torch.tensor((20.,1.))
      term.switch_window[:] = torch.tensor((20.,19.))
      term.curriculum_episodes=4
      manipulation_curriculum(env,torch.empty(0,dtype=torch.long))
      env.common_step_counter+=100
    self.assertEqual(term.stage,0,'Only independent stance evaluations may promote stage zero')

  def test_failed_trial_cannot_be_erased_by_reset(self):
    module=self.module('src.tasks.leg_manip.rl.stance_evaluation')
    meter=module.HoldTrials(2,5,'cpu')
    meter.record(torch.tensor([True,True]),torch.tensor([True,False]))
    for _ in range(4): meter.record(torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    result=meter.summary()
    self.assertAlmostEqual(result['hold_fraction'],.5)
    self.assertAlmostEqual(result['survival'],.5)
    self.assertEqual(result['planned_frames'],10)

  def test_stage_zero_needs_two_independent_balanced_evaluations(self):
    module=self.module('src.tasks.leg_manip.rl.stance_evaluation')
    env,term=make_env(stage=0)
    good={name:{'hold_fraction':.75,'survival':.9,'trials':32,'planned_frames':8000} for name in ('quad','biped')}
    module.apply_stance_evaluation(term,good,500)
    self.assertEqual(term.stage,0)
    self.assertEqual(term.pass_streak,1)
    module.apply_stance_evaluation(term,good,500)
    self.assertEqual(term.stage,0,'Duplicate evaluation must not advance streak')
    module.apply_stance_evaluation(term,good,1000)
    self.assertEqual(term.stage,1)
    self.assertEqual(term.pass_streak,0)

  def test_insufficient_biped_trials_and_failed_stance_reset_streak(self):
    module=self.module('src.tasks.leg_manip.rl.stance_evaluation')
    env,term=make_env(stage=0)
    good={name:{'hold_fraction':.75,'survival':.9,'trials':32,'planned_frames':8000} for name in ('quad','biped')}
    module.apply_stance_evaluation(term,good,500)
    good['biped']['trials']=31
    module.apply_stance_evaluation(term,good,1000)
    self.assertEqual(term.pass_streak,0)
    good['biped']['trials']=32;good['biped']['hold_fraction']=.69
    module.apply_stance_evaluation(term,good,1500)
    self.assertEqual(term.stage,0)

  def test_contact_grace_is_bounded_and_motion_is_not_cancelled(self):
    module=self.module('src.tasks.leg_manip.mdp.stance_hold')
    meter=module.StanceHoldHistory(1,.02,'cpu')
    contacts=torch.ones(1,4,dtype=torch.bool)
    for _ in range(10): meter.update(contacts,torch.zeros(1,3),torch.zeros(1,3))
    contacts[0,0]=False
    for _ in range(3): meter.update(contacts,torch.zeros(1,3),torch.zeros(1,3))
    self.assertTrue(meter.support(biped=False).item())
    meter.update(contacts,torch.zeros(1,3),torch.zeros(1,3))
    self.assertFalse(meter.support(biped=False).item())
    for i in range(10): meter.update(torch.ones_like(contacts),torch.tensor([[(-1.)**i,0.,0.]]),torch.zeros(1,3))
    self.assertFalse(meter.motion_ok().item(),'Opposite velocities must not cancel')

  def test_real_push_refreshes_velocity_before_observation(self):
    module=self.module('src.tasks.leg_manip.mdp.events')
    calls=[]
    robot=SimpleNamespace(data=SimpleNamespace(root_link_vel_w=torch.zeros(2,6)),
      write_root_link_velocity_to_sim=lambda velocity,env_ids: calls.append(('write',velocity.clone())))
    term=SimpleNamespace(stage=1)
    env=SimpleNamespace(num_envs=2,device='cpu',scene={'robot':robot},
      command_manager=SimpleNamespace(get_term=lambda name:term),sim=SimpleNamespace(forward=lambda:calls.append(('forward',None))))
    module.push_robot(env,torch.arange(2),{'x':(.3,.3)})
    self.assertEqual([c[0] for c in calls],['write','forward'])
    torch.testing.assert_close(calls[0][1][:,0],torch.full((2,),.3))

  def test_stage_zero_push_disabled_by_default(self):
    module=self.module('src.tasks.leg_manip.mdp.events')
    env,term=make_env(stage=0)
    module.push_robot(env,torch.arange(4),{'x':(.3,.3)})
    torch.testing.assert_close(term.robot.data.root_link_lin_vel_w,torch.zeros(4,3))

  def test_stage_zero_exploration_defaults(self):
    cfg=leg_manip_ppo_runner_cfg()
    self.assertEqual(cfg.actor.distribution_cfg['init_std'],.2)
    self.assertEqual(cfg.algorithm.entropy_coef,.001)

  def test_bounded_distribution_uses_same_std_for_sampling_and_probability(self):
    module=self.module('src.tasks.leg_manip.rl.distribution')
    dist=module.StanceGaussianDistribution(12,init_std=.9)
    mean=torch.zeros(20000,12)
    dist.update(mean)
    sampled=dist.sample()
    self.assertAlmostEqual(float(sampled.std()),.3,delta=.005)
    expected=torch.distributions.Normal(mean,torch.full_like(mean,.3))
    torch.testing.assert_close(dist.log_prob(sampled),expected.log_prob(sampled).sum(-1))
    torch.testing.assert_close(dist.entropy,expected.entropy().sum(-1))

  def test_streaming_and_episode_metric_names_do_not_overlap(self):
    from src.tasks.leg_manip.rl.metrics import CourseLogger
    env,term=make_env(stage=0)
    term.phase[:]=0;term.elapsed[:]=2.;term.record_outcomes()
    written={}
    writer=SimpleNamespace(add_scalar=lambda key,value,it:written.__setitem__(key,value))
    extra=term.reset(torch.tensor([0]))
    def base_log(**kwargs):
      for key,value in extra.items(): writer.add_scalar('Metrics/twist/'+key,value,kwargs['it'])
    CourseLogger(SimpleNamespace(writer=writer,log=base_log),term.training_metrics).log(it=3)
    self.assertIn('Metrics/twist/episode_quad_stationary_samples',written)
    self.assertEqual(written['Metrics/twist/quad_stationary_samples'],4.)

  def test_automatic_landing_has_full_trajectory_timeout(self):
    from src.tasks.leg_manip.mdp import rewards
    env,term=make_env(stage=0)
    term.phase[:]=4;term.manual[:]=False;term._operation[:]=False
    term.elapsed[:]=3.
    self.assertFalse(rewards.landing_failed(env).any())
    term.manual[:]=True
    self.assertTrue(rewards.landing_failed(env).all())

  def test_descent_source_uses_posture_context_before_target_cancel(self):
    from tests.test_leg_manip_commands import advance
    from src.tasks.pedipulation.constants import DESIRED_ANGLES
    env,term=make_env(stage=1)
    term.set_command([0],target_offset=term.nominal_biped_offset)
    advance(env,term,1)
    data=term.robot.data
    data.root_link_pos_w[0,2]=.52;data.projected_gravity_b[0]=torch.tensor((-1.,0.,0.))
    data.joint_pos[0]=torch.tensor(DESIRED_ANGLES)
    data.joint_pos[0,3:6]+=torch.tensor((.8,1.,1.3))
    env.scene['feet_ground_contact'].data.force[0,:2]=0.
    advance(env,term,120)
    self.assertTrue(term.stance_ready(biped=True)[0])
    term.set_command([0],target_offset=(0.,0.,0.))
    advance(env,term,1)
    self.assertEqual(int(term.pending_switch[0]),1,'Canceling FR target must not rewrite the source posture criterion')

  def test_evaluation_interval_must_be_positive(self):
    with self.assertRaises(ValueError): make_env(stage=0,stance_eval_interval=0)

  def test_rng_isolation_covers_every_cuda_generator_even_for_cpu_eval(self):
    from contextlib import nullcontext
    from unittest.mock import patch
    from src.tasks.leg_manip.rl.stance_evaluation import isolated_rng
    # Simulate a multi-GPU host without requiring multiple physical devices.
    with patch('torch.cuda.device_count',return_value=3), patch('torch.random.fork_rng',return_value=nullcontext()) as fork:
      with isolated_rng('cpu'): pass
      fork.assert_called_once_with(devices=[0,1,2])

  def test_switch_buffers_remain_mutable_after_inference_rollout(self):
    from tests.test_leg_manip_commands import advance
    env,term=make_env(stage=0)
    with torch.inference_mode():
      advance(env,term,1)
      term._record_switch_outcomes()
    try:
      term.interrupt_switches(torch.arange(4))
      term.reset(torch.arange(4))
    except RuntimeError as error:
      self.fail(f'Promotion outside inference mode cannot clear switch buffers: {error}')

  def test_checkpoint_migration_preserves_mean_and_resets_training(self):
    module=self.module('src.tasks.leg_manip.rl.checkpoint')
    self.assertTrue(hasattr(module,'migrate_stance_checkpoint'))
    _,term=make_env(stage=0)
    cp={'actor_state_dict':{'mlp.0.weight':torch.randn(4,51),'distribution.std_param':torch.ones(12)},
      'critic_state_dict':{'mlp.0.weight':torch.randn(4,131)},'optimizer_state_dict':{'state':{0:{'step':4}}},
      'iter':14999,'infos':{'leg_manip_course':{'version':1},'env_state':{'common_step_counter':360000}}}
    migrated=module.migrate_stance_checkpoint(cp,term)
    torch.testing.assert_close(migrated['actor_state_dict']['mlp.0.weight'],cp['actor_state_dict']['mlp.0.weight'])
    torch.testing.assert_close(migrated['actor_state_dict']['distribution.std_param'],torch.full((12,),.2))
    self.assertEqual(migrated['optimizer_state_dict']['state'],{})
    self.assertEqual(migrated['iter'],0)
    self.assertEqual(migrated['infos']['env_state']['common_step_counter'],0)
    module.validate_course_state(migrated['infos']['leg_manip_course'])

if __name__=='__main__': unittest.main()
