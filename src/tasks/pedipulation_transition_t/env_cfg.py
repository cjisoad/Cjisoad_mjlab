"""Independent rear-leg transition teacher with nominal-first physics."""
from dataclasses import dataclass
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields, RecomputeLevel
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig
from src.tasks.pedipulation.robot import get_stand_robot_cfg, NONFOOT_BODIES
from src.tasks.pedipulation.mdp.actions import PedipulationPositionActionCfg, PedipulationPositionAction
from .commands import TransitionCommandCfg
from . import observations, rewards


@dataclass(kw_only=True)
class TransitionPositionActionCfg(PedipulationPositionActionCfg):
  def build(self,env):
    return TransitionPositionAction(self,env)


class TransitionPositionAction(PedipulationPositionAction):
  def process_actions(self,actions):
    super().process_actions(actions.clamp(-10.,10.))
    term=self._env.command_manager.get_term('twist')
    self.delay_steps[~term.disturbed]=0

  def reset(self,env_ids=None):
    super().reset(env_ids)
    self._raw_action[slice(None) if env_ids is None else env_ids]=0.


@requires_model_fields('body_mass','body_inertia','body_ipos','geom_friction',
  'dof_armature','dof_damping','dof_frictionloss',recompute=RecomputeLevel.none)
def declare_transition_physics(env,env_ids):
  """Declare expanded model fields up front; stage zero preserves nominal data."""
  action=env.action_manager.get_term('joint_pos')
  action.dr_observation[:,0]=1.  # Robot contact friction.
  action.dr_observation[:,5:29]=1.  # Nominal kp and kd multipliers.
  armatures=env.sim.get_default_field('dof_armature')[env.scene['robot'].indexing.joint_v_adr]
  action.dr_observation[:,29]=armatures.mean()


def randomize_transition_physics(env,ids):
  """Final-stage episodic DR; undisturbed instances retain nominal physics."""
  term=env.command_manager.get_term('twist'); robot=env.scene['robot']
  action=env.action_manager.get_term('joint_pos'); model=env.sim.model
  active=term.disturbed[ids]; n=len(ids)
  def sample(low,high,width):
    return torch.empty(n,width,device=env.device).uniform_(low,high)
  bodies=robot.indexing.body_ids; geoms=robot.indexing.geom_ids; dofs=robot.indexing.joint_v_adr
  factors=torch.where(active[:,None],sample(.9,1.1,len(bodies)),1.)
  mass=env.sim.get_default_field('body_mass')[bodies]
  model.body_mass[ids[:,None],bodies]=mass[None]*factors
  model.body_inertia[ids[:,None],bodies]=env.sim.get_default_field('body_inertia')[bodies][None]*factors[...,None]
  base_local=robot.find_bodies('base_link')[0][0]; base=bodies[base_local]
  offset=sample(-.015,.015,3)*active[:,None]
  model.body_ipos[ids,base]=env.sim.get_default_field('body_ipos')[base]+offset
  friction=torch.where(active[:,None],sample(.6,1.2,1),1.)
  model.geom_friction[ids[:,None],geoms,0]=friction
  nominal_armature=env.sim.get_default_field('dof_armature')[dofs]
  armature=nominal_armature[None]*torch.where(active[:,None],sample(.8,1.2,len(dofs)),1.)
  model.dof_armature[ids[:,None],dofs]=armature
  damping=sample(0.,.05,1)*active[:,None]; joint_friction=sample(0.,.03,1)*active[:,None]
  model.dof_damping[ids[:,None],dofs]=damping
  model.dof_frictionloss[ids[:,None],dofs]=joint_friction
  kp=torch.where(active[:,None],sample(.9,1.1,12),1.)
  kd=torch.where(active[:,None],sample(.9,1.1,12),1.)
  action.kp_multipliers[ids]=kp; action.kd_multipliers[ids]=kd
  action.motor_offsets[ids]=sample(-.015,.015,12)*active[:,None]
  for actuator in robot.actuators:
    canonical=[list(robot.joint_names).index(name) for name in actuator.target_names]
    ordered=[action.joint_ids.tolist().index(i) for i in canonical]
    actuator.set_gains(ids,40.*kp[:,ordered],kd[:,ordered])
  zeros=friction.new_zeros(n,1)
  added=(model.body_mass[ids,base]-mass[base_local])[:,None]
  action.dr_observation[ids]=torch.cat((friction,added,offset,kp,kd,
    armature.mean(-1,keepdim=True),joint_friction,damping,zeros,zeros),-1)


def push_transition(env,env_ids):
  term=env.command_manager.get_term('twist')
  if term.stage<3 or term.cfg.play: return
  term._ensure_reference()
  elapsed=env.episode_length_buf*env.step_dt
  post=term.disturbed&~term.post_pushed&(term.reference_time>=term.stand_start+1.)&(elapsed>=.8)
  mid=term.disturbed&~term.mid_pushed&(term.reference_time>=term.stand_start*.54)&(term.reference_time<term.stand_start*.84)&(elapsed>=.3)&(term.stage>=4)
  ids=(post|mid).nonzero().flatten()
  if not len(ids): return
  robot=env.scene['robot']; data=robot.data
  velocity=torch.cat((data.root_link_lin_vel_w[ids],data.root_link_ang_vel_w[ids]),-1)
  velocity[:,:2]+=torch.empty(len(ids),2,device=env.device).uniform_(-.2,.2)
  velocity[:,3:]+=torch.empty(len(ids),3,device=env.device).uniform_(-.3,.3)
  robot.write_root_link_velocity_to_sim(velocity,env_ids=ids)
  # Outcomes are computed before step events. A success acquired on the push
  # step must be earned again through a full continuous hold after the impulse.
  term.hold_time[post]=0.
  term.held_success[post]=False
  term.metrics['held_success'][post]=0.
  term.post_pushed[post]=True; term.mid_pushed[mid]=True
  env.sim.forward()


def _sensor(name,bodies,secondary):
  return ContactSensorCfg(name=name,
    primary=ContactMatch(mode='body',pattern=tuple(bodies),entity='robot'),
    secondary=secondary,fields=('force','found'),reduce='netforce',num_slots=1,global_frame=True)


def transition_env_cfg(play=False):
  robot=get_stand_robot_cfg()
  for actuator in robot.articulation.actuators:
    actuator.armature=.02 if 'calf' in actuator.target_names_expr[0] else .01
  ground=ContactMatch(mode='geom',pattern='terrain')
  sensors=(
    _sensor('feet_ground_contact',('FL_foot','FR_foot','RL_foot','RR_foot'),ground),
    _sensor('nonfoot_ground_contact',NONFOOT_BODIES,ground),
    _sensor('base_ground_contact',('base_link','Head_upper','Head_lower'),ground),
    _sensor('self_contact',NONFOOT_BODIES,
      ContactMatch(mode='subtree',pattern='base_link',entity='robot')))
  weights={'orientation':3.,'height':1.5,'fr_position':3.,'support_pose':.5,
    'linear_velocity':.3,'angular_velocity':.3,'fl_height':.5}
  reward_terms={name:RewardTermCfg(func=rewards.tracking_reward,weight=weight,params={'name':name})
    for name,weight in weights.items()}
  penalties={'support_loss':-.7,'front_contact':-.5,'slip':-.15,'excessive_force':-.1,
    'collision':-1.,'action_change':-.03,'torque':-.02,'joint_limits':-.5}
  reward_terms.update({name:RewardTermCfg(func=rewards.penalty,weight=weight,params={'name':name})
    for name,weight in penalties.items()})
  reward_terms['failure']=RewardTermCfg(func=rewards.failure_cost,weight=1.)
  return ManagerBasedRlEnvCfg(seed=1,
    scene=SceneCfg(num_envs=1 if play else 4096,env_spacing=3.,entities={'robot':robot},
      terrain=TerrainEntityCfg(terrain_type='plane',collisions=(CollisionCfg(
        geom_names_expr=('terrain',),friction=(1.,.005,.0001),contype=1,conaffinity=1,condim=3,priority=0),)),
      sensors=sensors),
    actions={'joint_pos':TransitionPositionActionCfg(entity_name='robot',scale=.25,delay=False)},
    commands={'twist':TransitionCommandCfg(resampling_time_range=(100.,100.),play=play,debug_vis=play)},
    observations={
      'actor':ObservationGroupCfg(terms={'policy':ObservationTermCfg(func=observations.policy_state,params={'add_noise':not play})}),
      'critic':ObservationGroupCfg(terms={'policy':ObservationTermCfg(func=observations.shared_policy_state),
        'privileged':ObservationTermCfg(func=observations.privileged_state)})},
    events={'physics':EventTermCfg(func=declare_transition_physics,mode='startup'),
      **({} if play else {'push_robot':EventTermCfg(func=push_transition,mode='step')})},
    rewards=reward_terms,
    terminations={'failure':TerminationTermCfg(func=rewards.failure_termination),
      'time_out':TerminationTermCfg(func=rewards.time_out,time_out=True)},
    metrics={name:MetricsTermCfg(func=rewards.transition_metric,params={'name':name}) for name in
      ('held_success','full_start','orientation_error','height_error','fr_error','course_stage')},
    curriculum={},sim=SimulationCfg(nconmax=96,njmax=512,contact_sensor_maxmatch=1024,
      mujoco=MujocoCfg(timestep=.005,iterations=30,ls_iterations=10)),
    decimation=4,episode_length_s=10.,scale_rewards_by_dt=True,
    viewer=ViewerConfig(entity_name='robot',body_name='base_link',
      origin_type=ViewerConfig.OriginType.ASSET_BODY,distance=2.5,elevation=-10.))
