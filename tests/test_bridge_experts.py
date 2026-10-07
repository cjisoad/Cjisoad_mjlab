"""Frozen checkpoint ABI and physical-target coordinate invariance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from src.tasks.pedipulation_bridge_t import experts


ROOT = Path(__file__).resolve().parents[1]
QUAD_PATH = ROOT / 'outputs/new_teachers_review_20261006/quadruped/checkpoints/model_9900.pt'
BIPED_PATH = ROOT / 'outputs/new_teachers_review_20261006/biped/checkpoints/model_14600.pt'
HAS_CHECKPOINTS = QUAD_PATH.is_file() and BIPED_PATH.is_file()


def fixture(count=4):
  quad_zero = torch.tensor((-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8))
  data = SimpleNamespace(root_link_ang_vel_b=torch.arange(count*3).reshape(count,3)*.01,
    projected_gravity_b=torch.tensor((0.,0.,-1.)).expand(count,3),
    root_link_quat_w=torch.tensor((1.,0.,0.,0.)).expand(count,4),
    gravity_vec_w=torch.tensor((0.,0.,-1.)).expand(count,3),
    root_link_lin_vel_w=torch.arange(count*3).reshape(count,3)*.02,
    joint_pos=quad_zero.expand(count,12).clone()+.1,
    joint_vel=torch.arange(count*12).reshape(count,12)*.01)
  robot=SimpleNamespace(data=data, joint_names=experts.JOINT_NAMES)
  action=SimpleNamespace(default_angles=quad_zero.expand(count,12).clone(),
    joint_ids=torch.arange(12), raw_action=torch.arange(count*12).reshape(count,12)*.02,
    motor_offsets=torch.ones(count,12)*.013,
    cfg=SimpleNamespace(scale=.25, delay=True, clip_actions=12.))
  action.joint_pos=data.joint_pos; action.joint_vel=data.joint_vel
  command=torch.tensor((.2,-.1,.05,.04,.01,.3)).expand(count,6).clone()
  env=SimpleNamespace(scene={'robot':robot},device='cpu',
    action_manager=SimpleNamespace(get_term=lambda name:action),
    command_manager=SimpleNamespace(get_command=lambda name:command))
  return env, action, command


class CoordinateTests(unittest.TestCase):
  def test_previous_action_preserves_delayed_physical_target(self):
    common_zero=torch.randn(3,12); teacher_zero=torch.randn(12)
    previous=torch.randn(3,12); offsets=torch.randn(3,12)*.015
    translated=experts.common_previous_to_teacher(previous,common_zero,teacher_zero)
    torch.testing.assert_close(teacher_zero+.25*translated+offsets,
                               common_zero+.25*previous+offsets)

  def test_clip_teacher_raw_before_translation_preserves_saturated_target(self):
    common_zero=torch.zeros(2,12); teacher_zero=torch.ones(12)*.3
    raw=torch.full((2,12),40.); offsets=torch.full((2,12),.013)
    common=experts.teacher_action_to_common(raw,teacher_zero,common_zero,common_clip=12.)
    torch.testing.assert_close(common,torch.full((2,12),11.2))
    torch.testing.assert_close(common_zero+.25*common+offsets,
                               teacher_zero+.25*raw.clamp(-10.,10.)+offsets)

  def test_unrepresentable_common_target_is_rejected_instead_of_clamped(self):
    with self.assertRaisesRegex(ValueError,'common.*clip'):
      experts.teacher_action_to_common(torch.full((1,12),10.),torch.ones(12)*.3,
                                      torch.zeros(1,12),common_clip=10.)


@unittest.skipUnless(HAS_CHECKPOINTS,'approved frozen checkpoints are not available')
class FrozenCheckpointTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.checkpoints={name:torch.load(path,map_location='cpu',weights_only=False)
      for name,path in (('quadruped',QUAD_PATH),('biped',BIPED_PATH))}

  def pair(self):
    env,action,command=fixture()
    return experts.FrozenTeacherPair(env,quadruped_checkpoint=QUAD_PATH,
      biped_checkpoint=BIPED_PATH),env,action,command

  def test_default_source_uses_original_model_9900(self):
    preferred=ROOT/'outputs/new_teachers_review_20261006/quadruped/checkpoints/model_9900.pt'
    self.assertEqual(experts.DEFAULT_QUADRUPED_CHECKPOINT,preferred)
    env,_,command=fixture()
    pair=experts.FrozenTeacherPair(env)
    checkpoint=torch.load(preferred,map_location='cpu',weights_only=False)
    expected=experts.FrozenTeacherActor(checkpoint['actor_state_dict'])
    observations=pair.observations('quadruped',command)
    torch.testing.assert_close(pair.actor_actions('quadruped',observations),expected(observations),rtol=0.,atol=0.)

  def test_real_metadata_passes_and_each_abi_error_fails(self):
    for name,checkpoint in self.checkpoints.items():
      task='loco_pedipulation_t' if name=='quadruped' else 'pedipulation_t'
      original=checkpoint['infos']['teacher_contract']
      experts.validate_teacher_contract(original,task)
      mutations={'version':1,'anchor':'leg_manip_body_x_minus_z','actor_dim':48,
        'task':'pedipulation_transition_t','command':'another_zero',
        'foot_zero':[0.,0.,0.], 'joint_names':list(reversed(original['joint_names'])),
        'default_angles':[0.]*12,'action_scale':.5,'clip_actions':100.,
        'control_delay':False,'base_velocity_scale':1.,'critic_dim':1}
      for field,value in mutations.items():
        with self.subTest(teacher=name,field=field):
          broken=deepcopy(original); broken[field]=value
          with self.assertRaisesRegex(ValueError,field):
            experts.validate_teacher_contract(broken,task)

  def test_actor_matches_original_rsl_model_and_cannot_train(self):
    pair,_,_,_=self.pair()
    observations=torch.linspace(-1.,1.,4*51).reshape(4,51)
    for name,checkpoint in self.checkpoints.items():
      with self.subTest(teacher=name):
        original=MLPModel(TensorDict({'actor':observations},batch_size=[4]),
          {'actor':['actor']},'actor',12,hidden_dims=(512,256,128),activation='elu',
          obs_normalization=False,distribution_cfg={
            'class_name':'src.tasks.pedipulation.rl.ppo:PedipulationGaussianDistribution',
            'init_std':1.,'std_type':'scalar'})
        original.load_state_dict(checkpoint['actor_state_dict'])
        expected=original(TensorDict({'actor':observations},batch_size=[4]))
        actual=pair.actor_actions(name,observations.requires_grad_())
        torch.testing.assert_close(actual,expected)
        self.assertFalse(actual.requires_grad)
        self.assertFalse(pair.actors[name].training)
        self.assertTrue(all(not p.requires_grad for p in pair.actors[name].parameters()))
        self.assertFalse(hasattr(pair,'optimizer'))

  def test_actual_state_observations_subset_and_common_targets_match_teachers(self):
    pair,env,action,command=self.pair(); ids=torch.tensor([3,1])
    for name,checkpoint in self.checkpoints.items():
      teacher_zero=torch.tensor(checkpoint['infos']['teacher_contract']['default_angles'])
      obs=pair.observations(name,command,env_ids=ids)
      data=env.scene['robot'].data
      expected=torch.cat((data.root_link_ang_vel_b[ids]*.25,data.projected_gravity_b[ids],
        command[ids]*torch.tensor((2.,2.,.25,1.,1.,1.)),
        data.joint_pos[ids]-teacher_zero,data.joint_vel[ids]*.05,
        action.raw_action[ids]+(action.default_angles[ids]-teacher_zero)/.25,
        data.root_link_lin_vel_w[ids]*2.),-1)
      torch.testing.assert_close(obs,expected)
      common=pair.actions(name,command,env_ids=ids)
      teacher_raw=pair.actor_actions(name,obs).clamp(-10.,10.)
      torch.testing.assert_close(action.default_angles[ids]+.25*common+action.motor_offsets[ids],
        teacher_zero+.25*teacher_raw+action.motor_offsets[ids],atol=1e-6,rtol=1e-6)
      torch.testing.assert_close(pair.actions(name,command[ids],env_ids=ids),common)
      torch.testing.assert_close(pair.actions(name,command,env_ids=ids),common)

  def test_common_joint_order_and_scale_fail_before_queries(self):
    env,action,_=fixture(); env.scene['robot'].joint_names=tuple(reversed(experts.JOINT_NAMES))
    with self.assertRaisesRegex(ValueError,'joint order'):
      experts.FrozenTeacherPair(env,quadruped_checkpoint=QUAD_PATH,biped_checkpoint=BIPED_PATH)
    env,action,_=fixture(); action.cfg.scale=.5
    with self.assertRaisesRegex(ValueError,'scale'):
      experts.FrozenTeacherPair(env,quadruped_checkpoint=QUAD_PATH,biped_checkpoint=BIPED_PATH)

  def test_queries_match_source_teacher_observations_without_mutating_state(self):
    from src.tasks.leg_manip.mdp.observations import policy_state
    pair,env,action,command=self.pair()
    original_action=action.raw_action.clone()
    original_offsets=action.motor_offsets.clone()
    original_position=env.scene['robot'].data.joint_pos.clone()
    for name,checkpoint in self.checkpoints.items():
      expected=pair.observations(name,command)
      teacher_env,teacher_action,_=fixture()
      teacher_data=teacher_env.scene['robot'].data
      teacher_zero=torch.tensor(checkpoint['infos']['teacher_contract']['default_angles'])
      teacher_data.default_joint_pos=teacher_zero.expand(4,12)
      teacher_action.default_angles=teacher_data.default_joint_pos
      teacher_action.raw_action=original_action+(action.default_angles-teacher_zero)/.25
      teacher_env.command_manager.get_term=lambda _:SimpleNamespace(basis_w=torch.eye(3).expand(4,3,3))
      source=policy_state(teacher_env,add_noise=False)
      torch.testing.assert_close(expected,source)
      pair.actions(name)
    torch.testing.assert_close(action.raw_action,original_action)
    torch.testing.assert_close(action.motor_offsets,original_offsets)
    torch.testing.assert_close(env.scene['robot'].data.joint_pos,original_position)

  def test_loading_preserves_rng_and_explicit_robot_action_interface(self):
    env,action,command=fixture()
    before=torch.random.get_rng_state()
    pair=experts.FrozenTeacherPair(robot=env.scene['robot'],action=action,
      quadruped_checkpoint=QUAD_PATH,biped_checkpoint=BIPED_PATH)
    self.assertTrue(torch.equal(torch.random.get_rng_state(),before))
    self.assertEqual(pair.actions('biped',command).shape,(4,12))
    with self.assertRaisesRegex(ValueError,'commands are required'):
      pair.actions('biped')
    pair.actors['biped'].train(True)
    self.assertFalse(pair.actors['biped'].training)

  def test_bad_actor_shapes_and_hidden_normalization_are_rejected(self):
    state=self.checkpoints['quadruped']['actor_state_dict']
    broken=dict(state); broken['mlp.0.weight']=torch.zeros(512,48)
    with self.assertRaisesRegex(ValueError,'architecture'):
      experts.FrozenTeacherActor(broken)
    broken=dict(state); broken['obs_normalizer._mean']=torch.zeros(51)
    with self.assertRaisesRegex(ValueError,'unnormalized'):
      experts.FrozenTeacherActor(broken)


if __name__=='__main__':
  unittest.main()
