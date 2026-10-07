"""Ownership, event attribution, and continuous frozen-teacher handoffs."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.tasks.pedipulation_bridge_t.commands import (
  BridgeCommand, BridgeCommandCfg, WARM, PREFIX, BRIDGE, VERIFY, DONE,
)
from src.tasks.pedipulation_bridge_t.actions import BridgePositionAction, BridgePositionActionCfg
from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
from src.tasks.pedipulation_bridge_t import rewards
from src.tasks.pedipulation_bridge_t.experts import JOINT_NAMES, QUADRUPED_DEFAULT_ANGLES
from test_leg_manip_commands import make_env


class KnownPaths:
  h_start_range=(.25,.28)
  def sample(self,count):
    return torch.tensor((.02,.01,.07)).expand(count,3).clone(),torch.tensor((.04,.02,.32)).expand(count,3).clone()


def fixture(count=4,**overrides):
  env,_=make_env(count=count)
  data=env.scene['robot'].data
  data.default_joint_pos[:]=torch.tensor(QUADRUPED_DEFAULT_ANGLES)
  data.joint_pos.copy_(data.default_joint_pos)
  robot=env.scene['robot']; robot.joint_names=JOINT_NAMES
  robot.find_joints=lambda names,**kwargs:([JOINT_NAMES.index(n) for n in names],list(names))
  robot.set_joint_position_target=lambda values,**kwargs:setattr(robot,'last_target',values.clone())
  env.cfg=SimpleNamespace(decimation=4)
  env.episode_length_buf=torch.zeros(count,dtype=torch.long)
  defaults=dict(initial_stage=2,curriculum_enabled=False,candidate_seconds=.1,bridge_deadline=5.)
  defaults.update(overrides)
  cfg=BridgeCommandCfg(**defaults)
  term=BridgeCommand(cfg,env,paths=KnownPaths())
  env.command_manager=SimpleNamespace(get_term=lambda _:term,get_command=lambda _:term.command)
  action=BridgePositionAction(BridgePositionActionCfg(entity_name='robot'),env)
  env.action_manager=SimpleNamespace(get_term=lambda _:action)
  term.forced_live_probability=1.
  term.forced_disturbed=overrides.get('initial_impulse_probability',0.)==1.
  term.cfg.verify_handoff=True
  term.reset(torch.arange(count))
  return env,term,action


def tick(env,term,steps=1):
  for _ in range(steps):
    term.begin_step()
    env.common_step_counter+=1
    term.advance()
    term._update_command()


def set_matching_feet(env,term):
  env.scene['robot'].data.site_pos_w[:,1]=term.foot_target_pos_w


def stable_biped(env,term):
  data=env.scene['robot'].data
  data.root_link_pos_w[:,2]=.52
  data.root_link_quat_w[:]=torch.tensor((2.**-.5,0.,-2.**-.5,0.))
  data.projected_gravity_b[:]=torch.tensor((-1.,0.,0.))
  env.scene['feet_ground_contact'].data.force[:,:2]=0.
  set_matching_feet(env,term)


class OwnershipTests(unittest.TestCase):
  def test_zero_warmup_then_low_hold_and_fixed_velocity(self):
    env,term,_=fixture()
    torch.testing.assert_close(term.command,torch.zeros(4,6))
    term.warm_duration[:]=1.5; term.hold_duration[:]=1.
    tick(env,term,74)
    self.assertTrue((term.owner==WARM).all())
    tick(env,term,2)
    self.assertTrue((term.owner==PREFIX).all())
    self.assertTrue((term.command[:,1]==0.).all())
    self.assertTrue((term.h_start>=.25).all() and (term.h_start<=.28).all())
    self.assertTrue((term.h_end-term.h_start>=.02-1e-6).all())
    term.prefix_elapsed[:]=term.cfg.reach_time+.5
    term._update_command()
    torch.testing.assert_close(term.command[:,3:],term.low_offset)
    before=term.requested_velocity.clone()
    term.prefix_elapsed+=.2; term._update_command()
    torch.testing.assert_close(term.requested_velocity,before)

  def test_healthy_boundary_starts_bridge_without_perfect_source_support(self):
    env,term,_=fixture()
    term.owner[:]=PREFIX; term.prefix_elapsed[:]=10.
    term._update_command()
    env.scene['feet_ground_contact'].data.force[:,0]=0.
    set_matching_feet(env,term)
    tick(env,term)
    self.assertTrue((term.owner==BRIDGE).all())
    self.assertTrue((term.action_owner==PREFIX).all())

  def test_advance_is_once_per_physics_step(self):
    env,term,_=fixture()
    tick(env,term)
    before=term.attempt_elapsed.clone()
    term.advance(); rewards.bridge_termination(env); term._update_command()
    torch.testing.assert_close(term.attempt_elapsed,before)

  def test_scene_lookup_matches_real_scene_without_membership_protocol(self):
    env,term,_=fixture()
    values=env.scene
    class SceneView:
      env_origins=torch.zeros(4,3)
      def __getitem__(self,key):
        return values[key]
    env.scene=SceneView()
    tick(env,term)
    self.assertFalse(term.outcome_event.any())

  def test_nonfoot_ground_and_self_collision_fail_with_original_owner(self):
    for name in ('nonfoot_ground_touch','bridge_self_contact'):
      with self.subTest(sensor=name):
        env,term,_=fixture()
        term.owner[:]=torch.tensor((WARM,PREFIX,BRIDGE,VERIFY))
        env.scene[name]=SimpleNamespace(data=SimpleNamespace(force=torch.full((4,2,3),30.)))
        tick(env,term)
        self.assertEqual(term.outcome_event.tolist(),[-2,-2,-1,-1])

  def test_bridge_to_verify_preserves_state_history_and_waits_for_gate(self):
    env,term,action=fixture()
    term.owner[:]=BRIDGE; term.bridge_elapsed[:]=3.; term._update_command()
    action._raw_action[:]=.4; action.previous_action[:]=.3
    tick(env,term,10)
    self.assertTrue((term.owner==BRIDGE).all())
    stable_biped(env,term)
    state=env.scene['robot'].data.joint_pos.clone()
    tick(env,term,6)
    self.assertTrue((term.owner==VERIFY).all())
    torch.testing.assert_close(action.raw_action,torch.full((4,12),.4))
    torch.testing.assert_close(action.previous_action,torch.full((4,12),.3))
    torch.testing.assert_close(env.scene['robot'].data.joint_pos,state)
    self.assertTrue((term.handoff_count==1).all())

  def test_failures_keep_controller_attribution_and_events_survive_reset(self):
    env,term,_=fixture()
    term.owner[:]=torch.tensor((WARM,PREFIX,BRIDGE,VERIFY))
    env.scene['robot'].data.root_link_pos_w[:,2]=.05
    tick(env,term)
    self.assertEqual(term.outcome_event.tolist(),[-2,-2,-1,-1])
    self.assertEqual(term.action_owner.tolist(),[WARM,PREFIX,BRIDGE,VERIFY])
    tail_steps=term.tail_steps.clone()
    term.reset(torch.arange(4))
    self.assertEqual(term.outcome_event.tolist(),[-2,-2,-1,-1])
    self.assertEqual(term.action_owner.tolist(),[WARM,PREFIX,BRIDGE,VERIFY])
    torch.testing.assert_close(term.tail_steps,tail_steps)
    self.assertTrue((term.owner==WARM).all())
    term.begin_step()
    self.assertEqual(term.outcome_event.tolist(),[0,0,0,0])

  def test_tail_window_reports_success_and_timeout_is_failure(self):
    env,term,_=fixture()
    term.owner[:]=VERIFY; term.bridge_elapsed[:]=3.; term._update_command()
    stable_biped(env,term)
    for _ in range(130):
      tick(env,term)
      if term.outcome_event.any():
        break
    self.assertEqual(term.outcome_event.tolist(),[1]*4)
    self.assertTrue((term.owner==DONE).all())
    env,term,_=fixture()
    term.owner[:]=BRIDGE; term.bridge_elapsed[:]=5.01
    tick(env,term)
    self.assertEqual(term.outcome_event.tolist(),[-1]*4)

  def test_action_override_happens_after_policy_clip_and_keeps_previous_target(self):
    env,term,action=fixture()
    term.owner[:]=torch.tensor((WARM,PREFIX,BRIDGE,VERIFY))
    action._raw_action[:]=.7; action.motor_offsets[:]=.013
    action.experts=SimpleNamespace(actions=lambda name,command,env_ids:
      torch.full((len(env_ids),12),11.2 if name=='biped' else .2))
    action.process_actions(torch.ones(4,12)*40.)
    self.assertEqual(term.action_owner.tolist(),[WARM,PREFIX,BRIDGE,VERIFY])
    torch.testing.assert_close(action.previous_action,torch.full((4,12),.7))
    torch.testing.assert_close(action.raw_action[2],torch.full((12,),10.))
    torch.testing.assert_close(action.raw_action[3],torch.full((12,),11.2))
    action.delay_steps[:]=0; action.apply_actions()
    torch.testing.assert_close(env.scene['robot'].last_target,
      action.default_angles+.25*action.raw_action+action.motor_offsets)


class ConfigurationTests(unittest.TestCase):
  def test_self_contact_covers_robot_subtree_and_excludes_terrain(self):
    cfg=bridge_env_cfg()
    sensor=next(s for s in cfg.scene.sensors if s.name=='bridge_self_contact')
    self.assertEqual(sensor.secondary.mode,'subtree')
    self.assertEqual(sensor.secondary.pattern,'base_link')
    self.assertEqual(sensor.secondary.entity,'robot')

  def test_handoff_start_height_must_stay_inside_audited_path_range(self):
    for bounds in ((.24,.26),(.27,.29),(.24,.24)):
      with self.subTest(bounds=bounds),self.assertRaisesRegex(ValueError,'audited.*start.*height'):
        fixture(start_height_range=bounds)

  def test_divided_source_durations_must_be_finite_and_positive(self):
    for name in ('reach_time','velocity_ramp_time'):
      for value in (0.,-.1,float('nan')):
        with self.subTest(parameter=name,value=value),self.assertRaisesRegex(ValueError,'bridge durations'):
          fixture(**{name:value})

  def test_source_physics_and_privileged_channels_are_preserved(self):
    from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
    cfg=bridge_env_cfg(); source=loco_pedipulation_teacher_env_cfg()
    self.assertEqual(cfg.scene.entities['robot'],source.scene.entities['robot'])
    self.assertEqual(set(cfg.observations['critic'].terms),{'policy','privileged'})
    self.assertEqual(cfg.observations['actor'].terms['policy'].params,
      source.observations['actor'].terms['policy'].params)
    self.assertEqual(set(cfg.events),{'physics'})
    self.assertEqual(tuple(cfg.terminations),('bridge_outcome',))
    self.assertFalse(cfg.curriculum)
    self.assertIn('support_pose',cfg.rewards)
    self.assertIn('bridge_self_contact',[sensor.name for sensor in cfg.scene.sensors])

  def test_shaping_only_rewards_bridge_owned_steps(self):
    env,term,_=fixture()
    term.action_owner[:]=torch.tensor((WARM,PREFIX,BRIDGE,VERIFY))
    for name in ('fr_tracking','height','orientation','support_pose','fl_height'):
      values=rewards.shaping(env,name)
      self.assertEqual(values[[0,1,3]].tolist(),[0.,0.,0.])
      self.assertTrue(torch.isfinite(values).all())


@unittest.skipUnless(Path('src/assets/motions/go2/bridge_target_paths.npz').is_file(),
                     'offline audited bridge paths are required')
class CpuRuntimeTests(unittest.TestCase):
  def test_actor84_critic172_frozen_actions_and_events_survive_real_autoreset(self):
    from mjlab.envs import ManagerBasedRlEnv
    torch.set_num_threads(1)
    cfg=bridge_env_cfg(play=True)
    cfg.scene.num_envs=3
    cfg.commands['twist'].debug_vis=False
    cfg.commands['twist'].attempt_deadline=.01
    env=ManagerBasedRlEnv(cfg,device='cpu')
    try:
      import mujoco
      model=env.sim.mj_model
      sensors=[i for i in range(model.nsensor) if model.sensor(i).name.startswith('bridge_self_contact_')]
      self.assertGreater(len(sensors),2)
      for i in sensors:
        self.assertEqual(model.sensor_reftype[i],mujoco.mjtObj.mjOBJ_XBODY)
        self.assertEqual(model.body(int(model.sensor_refid[i])).name,'robot/base_link')
      obs,_=env.reset()
      self.assertEqual(obs['actor'].shape,(3,84))
      self.assertEqual(obs['critic'].shape,(3,172))
      term=env.command_manager.get_term('twist')
      action=env.action_manager.get_term('joint_pos')
      self.assertIsNone(action.experts)
      term.owner[:]=torch.tensor((WARM,BRIDGE,VERIFY))
      before_mass=env.sim.model.body_mass.clone()
      obs,reward,terminated,truncated,_=env.step(torch.zeros(3,12))
      self.assertEqual(term.outcome_event.tolist(),[-2,-1,-1])
      self.assertEqual(term.action_owner.tolist(),[WARM,BRIDGE,VERIFY])
      self.assertEqual(term.tail_steps.tolist(),[0,0,1])
      self.assertEqual(term.owner.tolist(),[BRIDGE,BRIDGE,BRIDGE])
      self.assertTrue(terminated.all())
      self.assertFalse(truncated.any())
      self.assertEqual(reward[[0,2]].tolist(),[0.,0.])
      self.assertIsNotNone(action.experts)
      self.assertTrue(all(torch.isfinite(value).all() for value in obs.values()))
      torch.testing.assert_close(env.sim.model.body_mass.clone(),before_mass)
      for actor in action.experts.actors.values():
        self.assertFalse(actor.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in actor.parameters()))
    finally:
      env.close()


if __name__=='__main__':
  unittest.main()
