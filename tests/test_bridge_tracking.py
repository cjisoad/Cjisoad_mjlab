import unittest
from dataclasses import asdict


class TrackingSettingsTests(unittest.TestCase):
  def test_default_is_reference_tracking_with_three_stage_course(self):
    from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
    cfg=bridge_env_cfg()
    command=cfg.commands['twist']
    self.assertEqual(command.initial_stage,0)
    self.assertTrue(command.curriculum_enabled)
    self.assertFalse(command.verify_handoff)
    self.assertEqual(command.bridge_deadline,10.)
    self.assertEqual(command.candidate_seconds,1.)
    self.assertNotIn('rise_progress',cfg.rewards)
    self.assertNotIn('linear_velocity',cfg.rewards)
    self.assertNotIn('angular_velocity',cfg.rewards)
    self.assertIn('fl_height',cfg.rewards)
    self.assertIn('support_loss',cfg.rewards)

  def test_ppo_uses_old_tracking_exploration_and_horizon(self):
    from src.tasks.pedipulation_bridge_t.rl import bridge_ppo_runner_cfg
    cfg=asdict(bridge_ppo_runner_cfg())
    self.assertEqual(cfg['actor']['distribution_cfg']['init_std'],.5)
    self.assertEqual(cfg['algorithm']['learning_rate'],3e-4)
    self.assertEqual(cfg['algorithm']['entropy_coef'],.005)
    self.assertEqual(cfg['algorithm']['gamma'],.995)
    self.assertEqual(cfg['algorithm']['max_bridge_steps'],502)
    self.assertEqual(cfg['max_iterations'],15000)
    self.assertEqual(cfg['logger'],'tensorboard')
    self.assertFalse(cfg['upload_model'])


class TrackingStateTests(unittest.TestCase):
  def test_reference_cache_handles_inference_quaternion_and_refreshes_anchor(self):
    import torch
    from unittest.mock import patch
    from test_bridge_env import fixture
    from src.tasks.teacher_common.anchor_frame import to_anchor
    env,term,_=fixture()
    term.start_bridge(torch.arange(env.num_envs))
    term.bridge_elapsed[:]=1.
    data=env.scene['robot'].data
    with torch.inference_mode():
      data.root_link_quat_w=data.root_link_quat_w.clone()
      with patch.object(term,'_reference_at',wraps=term._reference_at) as sample:
        first=term.reference_state
        data.root_link_quat_w[:]=torch.tensor((2.**-.5,0.,0.,2.**-.5))
        second=term.reference_state
        self.assertEqual(sample.call_count,2)
        self.assertIs(first,second)
        torch.testing.assert_close(second['linear_velocity'],
          to_anchor(term.basis_w,second['root_state'][:,7:10]))

  def test_reference_cache_invalidates_on_same_step_phase_change(self):
    import torch
    from test_bridge_env import fixture
    env,term,_=fixture()
    term.start_bridge(torch.arange(env.num_envs))
    first=term.reference_state
    self.assertIs(term.reference_state,first)
    term.bridge_elapsed[:]=1.5
    second=term.reference_state
    self.assertIsNot(second,first)
    self.assertFalse(torch.equal(second['quaternion'],first['quaternion']))

  def test_course0_restores_nominal_physics_after_randomized_episode(self):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
    cfg=bridge_env_cfg(play=True); cfg.scene.num_envs=2
    cfg.commands['twist'].debug_vis=False
    env=ManagerBasedRlEnv(cfg,device='cpu')
    try:
      term=env.command_manager.get_term('twist'); action=env.action_manager.get_term('joint_pos')
      term.course.stage=2; term.course.level=4; term.forced_disturbed=True
      term.forced_live_probability=0.
      env.reset()
      randomized=env.sim.model.body_mass.clone()
      self.assertTrue(action.motor_offsets.abs().max()>0.)
      self.assertTrue(action.cfg.delay)
      term.course.stage=0; term.forced_disturbed=False
      env.reset()
      bodies=env.scene['robot'].indexing.body_ids
      nominal=env.sim.get_default_field('body_mass')[bodies]
      torch.testing.assert_close(env.sim.model.body_mass[:,bodies],nominal[None].expand(2,-1))
      self.assertFalse(torch.equal(randomized,env.sim.model.body_mass))
      self.assertFalse(action.motor_offsets.any())
      self.assertFalse(action.cfg.delay)
      torch.testing.assert_close(action.kp_multipliers,torch.ones_like(action.kp_multipliers))
    finally: env.close()

  def test_reference_reset_matches_pose_and_velocity_and_keeps_other_world(self):
    import torch
    from test_bridge_env import fixture, BRIDGE
    from mjlab.utils.lab_api.math import quat_apply_inverse
    env,term,action=fixture()
    robot=env.scene['robot']; data=robot.data
    def write_root(root,env_ids):
      data.root_link_pos_w[env_ids]=root[:,:3]; data.root_link_quat_w[env_ids]=root[:,3:7]
      data.root_link_lin_vel_w[env_ids]=root[:,7:10]; data.root_link_ang_vel_w[env_ids]=root[:,10:13]
      data.projected_gravity_b[env_ids]=quat_apply_inverse(root[:,3:7],data.gravity_vec_w[env_ids])
    def write_joint(q,qd,joint_ids,env_ids):
      data.joint_pos[env_ids]=q; data.joint_vel[env_ids]=qd
    robot.write_root_state_to_sim=write_root; robot.write_joint_state_to_sim=write_joint
    term.forced_live_probability=0.; term.forced_start='rise'; term.forced_bank=0
    term.course.stage=0; term.forced_disturbed=False
    before=data.joint_pos[1:].clone()
    term.reset(torch.tensor([0]))
    ref=term.reference_state
    torch.testing.assert_close(data.joint_pos[0],ref['joint_pos'][0])
    torch.testing.assert_close(data.root_link_quat_w[0],ref['quaternion'][0])
    torch.testing.assert_close(data.root_link_pos_w[0,2],ref['height'][0])
    torch.testing.assert_close(data.joint_pos[1:],before)
    self.assertEqual(int(term.owner[0]),BRIDGE)
    self.assertFalse(bool(term.full_start[0]))
    self.assertFalse(action.cfg.delay)

  def test_success_requires_current_hold_and_is_not_a_latched_past_success(self):
    from test_bridge_env import fixture, stable_biped, tick, BRIDGE
    env,term,_=fixture(candidate_seconds=1.,bridge_deadline=10.)
    term.cfg.verify_handoff=False
    term.start_bridge(__import__('torch').arange(env.num_envs))
    term.reference_speed[:]=1.; term.bridge_elapsed[:]=3.
    term._update_command()
    stable_biped(env,term)
    tick(env,term,55)
    self.assertTrue(term.held_success.all())
    self.assertFalse(term.outcome_event.any())
    env.scene['robot'].data.root_link_pos_w[:,2]=.3
    tick(env,term)
    self.assertFalse(term.held_success.any())
    term.bridge_elapsed[:]=10.
    tick(env,term)
    self.assertEqual(term.outcome_event.tolist(),[-1]*4)

  def test_training_hold_never_switches_to_frozen_biped(self):
    import torch
    from test_bridge_env import fixture, stable_biped, tick, BRIDGE
    env,term,_=fixture(candidate_seconds=1.,bridge_deadline=10.)
    term.cfg.verify_handoff=False
    term.start_bridge(torch.arange(env.num_envs)); term.reference_speed[:]=1.
    term.bridge_elapsed[:]=3.; stable_biped(env,term)
    term._update_command(); stable_biped(env,term)
    tick(env,term,55)
    self.assertEqual(term.owner.tolist(),[BRIDGE]*4)
    term.bridge_elapsed[:]=10.
    tick(env,term)
    self.assertEqual(term.outcome_event.tolist(),[1]*4)


if __name__=='__main__': unittest.main()
