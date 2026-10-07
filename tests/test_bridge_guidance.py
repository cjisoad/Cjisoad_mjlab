"""Behavioral contract for a bridge whose only task is reference-guided rising."""
import unittest
from types import SimpleNamespace

import torch

from test_bridge_env import fixture, tick, set_matching_feet, stable_biped
from src.tasks.pedipulation_bridge_t.commands import PREFIX, BRIDGE, VERIFY
from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
from src.tasks.pedipulation_bridge_t import rewards


class RiseGuidanceTests(unittest.TestCase):
  def start(self, env, term, ids=None):
    self.assertTrue(callable(getattr(term, 'start_bridge', None)),
      'Boundary handoff must capture actual state for independent guidance')
    term.start_bridge(torch.arange(env.num_envs) if ids is None else ids)

  def test_velocity_commands_are_not_rise_objectives(self):
    cfg=bridge_env_cfg()
    self.assertNotIn('linear_velocity', cfg.rewards)
    self.assertNotIn('angular_velocity', cfg.rewards)
    self.assertNotIn('rear_hip', cfg.rewards)
    for name in ('height','orientation','support_pose','fl_position','rise_progress'):
      self.assertIn(name,cfg.rewards)

  def test_boundary_accepts_recoverable_fr_and_support_deviations(self):
    env,term,_=fixture()
    term.owner[:]=PREFIX; term.prefix_elapsed[:]=10.
    term._update_command()
    env.scene['feet_ground_contact'].data.force[:,0]=0.
    env.scene['robot'].data.site_pos_w[:,1,0]+= .2
    tick(env,term)
    self.assertTrue((term.owner==BRIDGE).all(), 'A healthy boundary state must not wait for perfect source tracking')

  def test_handoff_keeps_actual_velocity_joints_and_action_history(self):
    env,term,action=fixture()
    data=env.scene['robot'].data
    data.root_link_lin_vel_w[:]=torch.tensor((.2,.05,0.))
    data.joint_vel[:]=.3
    action._raw_action[:]=.7; action.previous_action[:]=.6
    before={name:getattr(data,name).clone() for name in
      ('root_link_pos_w','root_link_quat_w','root_link_lin_vel_w','joint_pos','joint_vel')}
    self.start(env,term)
    for name,value in before.items(): torch.testing.assert_close(getattr(data,name),value)
    torch.testing.assert_close(action.raw_action,torch.full((4,12),.7))
    torch.testing.assert_close(action.previous_action,torch.full((4,12),.6))
    torch.testing.assert_close(term.reference_height,data.root_link_pos_w[:,2])
    torch.testing.assert_close(term.reference_gravity,data.projected_gravity_b)

  def test_reference_clock_does_not_depend_on_fr_command(self):
    env,term,_=fixture()
    self.start(env,term)
    term.reference_speed[:]=1.
    term.bridge_elapsed[:]=1.2
    term._command[:,5]=torch.tensor((.1,.2,.3,.4))
    torch.testing.assert_close(term.progress,torch.full((4,),1.2/term.cfg.bridge_rise_seconds))
    torch.testing.assert_close(term.reference_height,term.reference_height[0].expand(4))
    torch.testing.assert_close(term.reference_gravity,term.reference_gravity[0].expand(4,3))

  def test_complete_preparation_and_endpoint_hold(self):
    env,term,_=fixture()
    self.start(env,term); term.reference_speed[:]=1.
    term.bridge_elapsed[:]=.4
    self.assertTrue((term.reference_gravity[:,0].abs()<1e-5).all(),
      'Early reference must include horizontal preparation instead of starting at0.9s')
    term.bridge_elapsed[:]=term.cfg.bridge_rise_seconds
    height,gravity,joints=term.reference_height.clone(),term.reference_gravity.clone(),term.reference_joint_pos.clone()
    term.bridge_elapsed[:]+=1.
    torch.testing.assert_close(term.reference_height,height)
    torch.testing.assert_close(term.reference_gravity,gravity)
    torch.testing.assert_close(term.reference_joint_pos,joints)
    self.assertTrue((term.progress==1.).all())
    self.assertTrue((gravity[:,0]<-.99).all())

  def test_source_velocity_is_retained_but_bridge_and_verification_request_standing(self):
    env,term,_=fixture()
    term.requested_velocity[:]=torch.tensor((.1,0.,.1))
    term.owner[:]=PREFIX; term.prefix_elapsed[:]=2.
    term._update_command()
    self.assertTrue((term.command[:,0]>.05).all())
    self.start(env,term); term._update_command()
    torch.testing.assert_close(term.command[:,:3],torch.zeros(4,3))
    term.owner[:]=VERIFY; term._update_command()
    torch.testing.assert_close(term.command[:,:3],torch.zeros(4,3))

  def test_partial_reset_does_not_rewind_other_references(self):
    env,term,_=fixture()
    self.start(env,term); term.bridge_elapsed[:]=1.
    expected=term.reference_height[1:].clone()
    speed=term.reference_speed[1:].clone()
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.reference_height[1:],expected)
    torch.testing.assert_close(term.reference_speed[1:],speed)

  def test_reference_features_are_visible_in_actor_and_critic(self):
    env,term,action=fixture()
    self.start(env,term); term.reference_speed[:]=1.
    cfg=bridge_env_cfg(play=True)
    actor_fn=cfg.observations['actor'].terms['policy'].func
    critic_fn=cfg.observations['critic'].terms['policy'].func
    first=actor_fn(env,add_noise=False)
    self.assertEqual(first.shape,(4,73))
    shared=critic_fn(env)
    self.assertEqual(shared.shape,(4,70))
    torch.testing.assert_close(shared[:,:48],first[:,:48])
    torch.testing.assert_close(shared[:,48:],first[:,51:])
    term.bridge_elapsed[:]=1.5
    second=actor_fn(env,add_noise=False)
    torch.testing.assert_close(first[:,:51],second[:,:51])
    self.assertFalse(torch.equal(first[:,51:],second[:,51:]))

  def test_fr_tracking_is_more_tolerant_during_rise(self):
    env,term,_=fixture()
    self.start(env,term); term.action_owner[:]=BRIDGE
    term.bridge_elapsed[:]=.5; term._update_command(); set_matching_feet(env,term)
    env.scene['robot'].data.site_pos_w[:,1,0]+=.08
    early=rewards.shaping(env,'fr_tracking')
    term.bridge_elapsed[:]=10.; term._update_command(); set_matching_feet(env,term)
    env.scene['robot'].data.site_pos_w[:,1,0]+=.08
    late=rewards.shaping(env,'fr_tracking')
    self.assertTrue((early>late*2.).all())

  def test_handoff_rejects_excessive_actual_linear_momentum(self):
    env,term,_=fixture()
    stable_biped(env,term)
    env.scene['robot'].data.root_link_lin_vel_w[:]=torch.tensor((2.,0.,0.))
    term.rear_contact_age[:]=0.
    self.assertFalse(term._biped_candidate().any())

  def test_configuration_rejects_invalid_guidance_and_impulses(self):
    for overrides in ({'reference_speed_range':(0.,1.)}, {'reference_blend_seconds':0.},
                      {'initial_impulse_probability':1.1}, {'initial_linear_impulse':(-.1,.1,0.)}):
      with self.subTest(overrides=overrides), self.assertRaises(ValueError): fixture(**overrides)

  def test_initial_impulse_is_once_and_preserves_pose_and_history(self):
    env,term,action=fixture(initial_impulse_probability=1.)
    data=env.scene['robot'].data
    def write_velocity(value,env_ids):
      data.root_link_lin_vel_w[env_ids]=value[:,:3]
      data.root_link_ang_vel_w[env_ids]=value[:,3:]
    env.scene['robot'].write_root_link_velocity_to_sim=write_velocity
    action._raw_action[:]=.4
    self.start(env,term)
    term.initial_impulse[:]=torch.tensor((.08,-.06,0.,.1,0.,-.1))
    pose=data.root_link_pos_w.clone(); joints=data.joint_pos.clone()
    action.process_actions(torch.full((4,12),.5))
    first=data.root_link_lin_vel_w.clone()
    torch.testing.assert_close(first,torch.tensor((.08,-.06,0.)).expand(4,3))
    torch.testing.assert_close(data.root_link_pos_w,pose)
    torch.testing.assert_close(data.joint_pos,joints)
    torch.testing.assert_close(action.previous_action,torch.full((4,12),.4))
    action.process_actions(torch.full((4,12),.5))
    torch.testing.assert_close(data.root_link_lin_vel_w,first)

  def test_pose_rewards_peak_at_reference_and_ignore_fr_joint_angles(self):
    env,term,_=fixture()
    self.start(env,term); term.action_owner[:]=BRIDGE
    term.bridge_elapsed[:]=1.5
    ref=term.reference_state; data=env.scene['robot'].data
    data.root_link_pos_w[:,2]=ref['height']
    data.projected_gravity_b.copy_(ref['gravity'])
    data.joint_pos.copy_(ref['joint_pos'])
    for name in ('height','orientation','support_pose'):
      torch.testing.assert_close(rewards.shaping(env,name),torch.ones(4))
    data.joint_pos[:,3:6]+=1.
    torch.testing.assert_close(rewards.shaping(env,'support_pose'),torch.ones(4))
    data.joint_pos[:,6]+=.5
    self.assertTrue((rewards.shaping(env,'support_pose')<1.).all())

  def test_progress_reward_requires_actual_improvement_and_is_once_per_step(self):
    env,term,_=fixture()
    self.start(env,term)
    term.begin_step()
    data=env.scene['robot'].data
    data.root_link_pos_w[:,2]=.5
    data.projected_gravity_b[:]=torch.tensor((-1.,0.,0.))
    env.common_step_counter+=1; term.advance()
    self.assertTrue((rewards.shaping(env,'rise_progress')>0.).all())
    previous=term.rise_progress_delta.clone()
    term.advance()
    torch.testing.assert_close(term.rise_progress_delta,previous)
    tick(env,term)
    torch.testing.assert_close(rewards.shaping(env,'rise_progress'),torch.zeros(4))

  def test_nonfinite_actual_velocity_fails_before_boundary_handoff(self):
    env,term,_=fixture()
    term.owner[:]=PREFIX; term.prefix_elapsed[:]=10.; term._update_command()
    env.scene['robot'].data.root_link_lin_vel_w[:,0]=float('nan')
    tick(env,term)
    self.assertEqual(term.outcome_event.tolist(),[-2]*4)


if __name__=='__main__': unittest.main()
