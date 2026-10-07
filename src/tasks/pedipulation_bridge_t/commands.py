"""Physical handoffs with independent controller ownership and source ABI."""
from dataclasses import dataclass
import math

import torch
from src.tasks.teacher_common.anchor_frame import to_anchor

from src.tasks.teacher_common.commands import (
  LocoPedipulationTeacherCommand, LocoPedipulationTeacherCommandCfg,
)
from src.tasks.loco_pedipulation.mdp.commands import QUAD, HOLD, smoothstep
from src.tasks.pedipulation_transition_t.motion import MotionReference, NOMINAL_MOTION_PATH
from .experts import JOINT_NAMES
from .guidance import RiseGuidance, SUPPORT_JOINTS

WARM, PREFIX, BRIDGE, VERIFY, DONE = range(5)
BRIDGE_CONTRACT = 'reference_guided_robust_rise_v2'
NO_REASON, PHYSICAL_FAILURE, SOURCE_DEADLINE, BRIDGE_DEADLINE, TAIL_INVALID, ATTEMPT_DEADLINE, TAIL_SUCCESS = range(7)
EVENT_REASON_NAMES = ('none','physical_failure','source_deadline','bridge_deadline',
                      'tail_invalid','attempt_deadline','tail_success')


@dataclass(kw_only=True)
class BridgeCommandCfg(LocoPedipulationTeacherCommandCfg):
  resampling_time_range: tuple[float,float] = (1e9,1e9)
  initial_stage: int = 3
  curriculum_enabled: bool = False
  warm_seconds: tuple[float,float] = (1.5,2.)
  low_hold_seconds: tuple[float,float] = (1.,1.5)
  prefix_rise_seconds: float = 2.
  bridge_rise_seconds: float = 2.5
  start_height_range: tuple[float,float] = (.25,.28)
  moving_probability: float = .5
  lin_vel_x: tuple[float,float] = (.05,.15)
  lin_vel_y: tuple[float,float] = (0.,0.)
  ang_vel_z: tuple[float,float] = (-.15,.15)
  source_deadline: float = 10.
  bridge_deadline: float = 5.
  attempt_deadline: float = 15.
  verify_seconds: float = 2.5
  candidate_seconds: float = .1
  recent_rear_seconds: float = .3
  invalid_seconds: float = .2
  reference_speed_range: tuple[float,float] = (.85,1.15)
  reference_blend_seconds: float = .3
  fr_tracking_scale: tuple[float,float] = (.10,.05)
  handoff_fr_error: float = .1
  handoff_linear_speed: float = .5
  verification_velocity: tuple[float,float,float] = (0.,0.,0.)
  initial_impulse_probability: float = .5
  initial_linear_impulse: tuple[float,float,float] = (.10,.10,0.)
  initial_angular_impulse: tuple[float,float,float] = (.20,.20,.20)
  motion_path: str = str(NOMINAL_MOTION_PATH)

  def build(self,env):
    return BridgeCommand(self,env)


class BridgeCommand(LocoPedipulationTeacherCommand):
  """Preserve source command state while scheduling independently guided rising.

  ``owner`` selects the next controller; ``action_owner`` is captured before
  physics and describes the transition that just occurred. Outcome events
  survive automatic resets and remain available to the rollout collector.
  """
  def __init__(self,cfg,env,*,paths=None):
    positive = (cfg.reach_time,cfg.velocity_ramp_time,
      cfg.prefix_rise_seconds,cfg.bridge_rise_seconds,cfg.source_deadline,
      cfg.bridge_deadline,cfg.attempt_deadline,cfg.verify_seconds,cfg.candidate_seconds,
      cfg.recent_rear_seconds,cfg.invalid_seconds,cfg.reference_blend_seconds,
      cfg.handoff_fr_error,cfg.handoff_linear_speed,*cfg.fr_tracking_scale)
    if not all(math.isfinite(x) and x>0 for x in positive):
      raise ValueError('bridge durations must be finite and positive')
    for name,bounds in (('warm_seconds',cfg.warm_seconds),('low_hold_seconds',cfg.low_hold_seconds),
                        ('start_height_range',cfg.start_height_range),
                        ('reference_speed_range',cfg.reference_speed_range)):
      if not all(math.isfinite(x) and x>0 for x in bounds) or bounds[0]>bounds[1]:
        raise ValueError(f'{name} must be positive ordered bounds')
    if cfg.lin_vel_y != (0.,0.) or not 0<=cfg.moving_probability<=1:
      raise ValueError('bridge requires vy=0 and a valid moving_probability')
    if not math.isfinite(cfg.initial_impulse_probability) or not 0<=cfg.initial_impulse_probability<=1:
      raise ValueError('initial impulse probability must be in [0,1]')
    for values in (cfg.initial_linear_impulse,cfg.initial_angular_impulse):
      if len(values)!=3 or not all(math.isfinite(x) and x>=0 for x in values):
        raise ValueError('initial impulse bounds must be three finite nonnegative values')
    if len(cfg.verification_velocity)!=3 or not all(math.isfinite(x) for x in cfg.verification_velocity):
      raise ValueError('verification velocity must have three finite values')
    super().__init__(cfg,env)
    if paths is None:
      from .paths import BridgeTargetPaths
      paths=BridgeTargetPaths(device=self.device)
    audited=getattr(paths,'h_start_range',None)
    if (audited is None or len(audited)!=2 or
        cfg.start_height_range[0]<audited[0] or cfg.start_height_range[1]>audited[1]):
      raise ValueError(f'audited path start height range is {audited!r}; '
                       f'requested {cfg.start_height_range!r} is outside it')
    self.paths=paths
    ids,names=self.robot.find_joints(JOINT_NAMES,preserve_order=True)
    if tuple(names)!=JOINT_NAMES:
      raise ValueError('bridge requires canonical joint order')
    self.joint_ids=torch.tensor(ids,device=self.device)
    self.support_ids=torch.tensor(SUPPORT_JOINTS,device=self.device)
    self.motion=MotionReference(cfg.motion_path)
    self.guidance=RiseGuidance(self.motion,self.device)
    n=self.num_envs
    self.owner=torch.full((n,),WARM,device=self.device,dtype=torch.long)
    self.action_owner=self.owner.clone()
    self.outcome_event=torch.zeros(n,device=self.device,dtype=torch.long)
    self.event_reason=torch.zeros_like(self.outcome_event)
    self.handoff_count=torch.zeros_like(self.outcome_event)
    self.tail_steps=torch.zeros_like(self.outcome_event)
    for name in ('attempt_elapsed','prefix_elapsed','bridge_elapsed','verify_elapsed',
                 'warm_duration','hold_duration','h_start','h_end','candidate_elapsed',
                 'invalid_elapsed'):
      setattr(self,name,torch.zeros(n,device=self.device))
    self.rear_contact_age=torch.full((n,2),1.,device=self.device)
    self.moving=torch.zeros(n,device=self.device,dtype=torch.bool)
    self.reference_speed=torch.ones(n,device=self.device)
    self.reference_started=torch.zeros(n,device=self.device,dtype=torch.bool)
    self.pending_initial_impulse=torch.zeros_like(self.reference_started)
    self.initial_impulse=torch.zeros(n,6,device=self.device)
    self.initial_height=torch.zeros(n,device=self.device)
    self.initial_gravity=torch.zeros(n,3,device=self.device)
    self.initial_joint_pos=torch.zeros(n,12,device=self.device)
    self.initial_fl_position=torch.zeros(n,3,device=self.device)
    self.start_linear_velocity=torch.zeros(n,3,device=self.device)
    self.start_angular_velocity=torch.zeros(n,3,device=self.device)
    self.start_fr_error=torch.zeros(n,device=self.device)
    self.rise_progress_delta=torch.zeros(n,device=self.device)
    self._previous_rise_score=torch.zeros(n,device=self.device)
    self.low_offset=torch.zeros(n,3,device=self.device)
    self.end_offset=torch.zeros_like(self.low_offset)
    self.start_offset=torch.zeros_like(self.low_offset)
    self._last_advance_step=env.common_step_counter
    self.metrics.update({name:torch.zeros(n,device=self.device) for name in
      ('bridge_owner','source_rejected','handoff_success','handoff_failed','handoff_count')})

  @property
  def fr_error(self):
    return self.foot_target_pos_anchor-self.foot_pos_anchor

  @property
  def foot_target_pos_anchor(self):
    return self.zero+self._command[:,3:]

  @property
  def progress(self):
    return (self.bridge_elapsed*self.reference_speed/self.cfg.bridge_rise_seconds).clamp(0.,1.)

  @property
  def reference_state(self):
    values=self.guidance.sample(self.progress)
    blend=smoothstep(self.bridge_elapsed/self.cfg.reference_blend_seconds)
    blend=torch.where(self.reference_started,blend,torch.ones_like(blend))
    for name,initial in (('height',self.initial_height),('gravity',self.initial_gravity),
                        ('joint_pos',self.initial_joint_pos),('fl_position',self.initial_fl_position)):
      alpha=blend if initial.ndim==1 else blend[:,None]
      values[name]=torch.lerp(initial,values[name],alpha)
    values['gravity']=torch.nn.functional.normalize(values['gravity'],dim=-1)
    return values

  @property
  def reference_height(self):
    return self.reference_state['height']

  @property
  def reference_gravity(self):
    return self.reference_state['gravity']

  @property
  def reference_joint_pos(self):
    return self.reference_state['joint_pos']

  @property
  def reference_fl_position(self):
    return self.reference_state['fl_position']

  @property
  def reference_fl_unload(self):
    return self.reference_state['fl_unload']

  def rise_score(self):
    data=self.robot.data
    origins=getattr(self._env.scene,'env_origins',data.root_link_pos_w.new_zeros(self.num_envs,3))
    end_height=self.guidance.frames['height'][-1]
    end_gravity=self.guidance.frames['gravity'][-1]
    height=(data.root_link_pos_w[:,2]-origins[:,2]-end_height)/.10
    gravity=(data.projected_gravity_b-end_gravity)/.5
    return torch.exp(-height.square()-gravity.square().sum(-1))

  def start_bridge(self,ids):
    """Capture reference alignment; preserve actual state and controller history."""
    if not len(ids): return
    data=self.robot.data
    origins=getattr(self._env.scene,'env_origins',data.root_link_pos_w.new_zeros(self.num_envs,3))
    self.owner[ids]=BRIDGE; self.bridge_elapsed[ids]=0.
    self.reference_started[ids]=True
    self.initial_height[ids]=data.root_link_pos_w[ids,2]-origins[ids,2]
    self.initial_gravity[ids]=data.projected_gravity_b[ids]
    self.initial_joint_pos[ids]=data.joint_pos[ids][:,self.joint_ids]
    fl=to_anchor(self.basis_w,data.site_pos_w[:,self.site_ids[0]]-data.root_link_pos_w)
    self.initial_fl_position[ids]=fl[ids]
    self.start_linear_velocity[ids]=to_anchor(self.basis_w,data.root_link_lin_vel_w)[ids]
    self.start_angular_velocity[ids]=data.root_link_ang_vel_b[ids]
    self.start_fr_error[ids]=self.fr_error.norm(dim=-1)[ids]
    self._previous_rise_score[ids]=self.rise_score()[ids]
    active=torch.rand(len(ids),device=self.device)<self.cfg.initial_impulse_probability
    bounds=data.root_link_pos_w.new_tensor((*self.cfg.initial_linear_impulse,*self.cfg.initial_angular_impulse))
    self.initial_impulse[ids]=(torch.rand(len(ids),6,device=self.device)*2.-1.)*bounds*active[:,None]
    self.pending_initial_impulse[ids]=active

  def _resample_command(self,env_ids):
    count=len(env_ids)
    if not count:
      return
    starts,ends=self.paths.sample(count)
    starts=starts.to(device=self.device,dtype=self._command.dtype)
    ends=ends.to(device=self.device,dtype=self._command.dtype)
    if starts.shape!=(count,3) or ends.shape!=(count,3) or not torch.isfinite(starts).all() or not torch.isfinite(ends).all():
      raise ValueError('validated bridge paths must return finite [N,3] endpoints')
    self.low_offset[env_ids]=starts; self.end_offset[env_ids]=ends
    self.h_start[env_ids]=torch.empty(count,device=self.device).uniform_(*self.cfg.start_height_range)
    self.h_end[env_ids]=ends[:,2]
    if ((self.h_end[env_ids]-self.h_start[env_ids])<.02-1e-6).any():
      raise ValueError('validated path must end at least .02m above handoff start height')
    ratio=((self.h_start[env_ids]-starts[:,2])/(ends[:,2]-starts[:,2])).clamp(0.,1.)
    self.start_offset[env_ids]=torch.lerp(starts,ends,ratio[:,None])
    self.warm_duration[env_ids]=torch.empty(count,device=self.device).uniform_(*self.cfg.warm_seconds)
    self.hold_duration[env_ids]=torch.empty(count,device=self.device).uniform_(*self.cfg.low_hold_seconds)
    self.moving[env_ids]=torch.rand(count,device=self.device)<self.cfg.moving_probability
    self.requested_velocity[env_ids]=0.
    for axis,bounds in ((0,self.cfg.lin_vel_x),(2,self.cfg.ang_vel_z)):
      values=torch.empty(count,device=self.device).uniform_(*bounds)
      self.requested_velocity[env_ids,axis]=values*self.moving[env_ids]
    self.requested_offset[env_ids]=starts
    self.owner[env_ids]=WARM
    self.reference_speed[env_ids]=torch.empty(count,device=self.device).uniform_(*self.cfg.reference_speed_range)
    self.reference_started[env_ids]=False; self.pending_initial_impulse[env_ids]=False
    self.rise_progress_delta[env_ids]=0.
    for name in ('attempt_elapsed','prefix_elapsed','bridge_elapsed',
                 'candidate_elapsed','invalid_elapsed'):
      getattr(self,name)[env_ids]=0
    # The collector reads a finished tail after ManagerBasedRlEnv autoresets.
    # Preserve its verification clock until the next action starts that attempt.
    fresh=env_ids[self.outcome_event[env_ids]==0]
    self.verify_elapsed[fresh]=0.; self.tail_steps[fresh]=0
    self.rear_contact_age[env_ids]=1.
    self._command[env_ids]=0.
    self.reference[env_ids]=self.zero
    self.start[env_ids]=self.zero; self.goal[env_ids]=self.zero

  def reset(self,env_ids):
    # Source reset preserves our terminal event, owner snapshot and lifetime
    # handoff count; it resamples the next attempt through the override above.
    return super().reset(env_ids)

  def begin_step(self):
    """Called exactly once by the action term before physics."""
    self.outcome_event.zero_(); self.event_reason.zero_()
    fresh=self.owner==WARM
    self.verify_elapsed[fresh]=0.; self.tail_steps[fresh]=0
    self.action_owner.copy_(self.owner)
    self._previous_rise_score.copy_(self.rise_score())

  def initialize_reference(self):
    self._pending_reset[:]=False

  def _update_command(self):
    self.advance()
    warm=self.owner==WARM
    prefix=self.owner==PREFIX
    remaining=~warm & ~prefix
    reach=smoothstep(self.prefix_elapsed/self.cfg.reach_time)
    start_rise=self.cfg.reach_time+self.hold_duration
    rise=smoothstep((self.prefix_elapsed-start_rise)/self.cfg.prefix_rise_seconds)
    prefix_offset=torch.lerp(self.low_offset,self.start_offset,rise[:,None])
    prefix_offset=torch.where((self.prefix_elapsed<self.cfg.reach_time)[:,None],
                             self.low_offset*reach[:,None],prefix_offset)
    bridge_fraction=smoothstep(self.progress)
    bridge_offset=torch.lerp(self.start_offset,self.end_offset,bridge_fraction[:,None])
    self._command[warm]=0.
    self._command[prefix,3:]=prefix_offset[prefix]
    self._command[remaining,3:]=bridge_offset[remaining]
    self._command[prefix,:3]=self.requested_velocity[prefix]*smoothstep(
      self.prefix_elapsed[prefix]/self.cfg.velocity_ramp_time)[:,None]
    self._command[remaining,:3]=0.
    verifying=(self.owner==VERIFY)|(self.owner==DONE)
    self._command[verifying,:3]=self._command.new_tensor(self.cfg.verification_velocity)
    self.phase[:]=torch.where(warm,QUAD,HOLD)
    self.enabled[:]=~warm
    self.blend[:]=torch.where(warm,torch.zeros_like(reach),reach)
    self.reference.copy_(self.foot_target_pos_anchor)
    self.goal.copy_(self.reference)
    self.elapsed.copy_(torch.where(prefix,self.prefix_elapsed,self.bridge_elapsed))

  def _finish(self,mask,event,reason):
    mask=mask & (self.outcome_event==0) & (self.owner!=DONE)
    self.outcome_event[mask]=event; self.event_reason[mask]=reason
    self.owner[mask]=DONE

  def _physical_failure(self):
    data=self.robot.data
    state=torch.cat((data.root_link_pos_w,data.root_link_quat_w,data.joint_pos,
                     data.joint_vel,data.root_link_ang_vel_b,
                     data.root_link_lin_vel_w,data.root_link_ang_vel_w),-1)
    origins=getattr(self._env.scene,'env_origins',data.root_link_pos_w.new_zeros(self.num_envs,3))
    hard=(~torch.isfinite(state).all(-1)) | ((data.root_link_pos_w[:,2]-origins[:,2])<.12)
    for sensor in ('base_contact','nonfoot_ground_touch','bridge_self_contact'):
      try:
        force=self._env.scene[sensor].data.force
      except KeyError:
        continue
      hard |= force.reshape(self.num_envs,-1,3).norm(dim=-1).amax(-1)>10.
    return hard

  def _biped_candidate(self):
    data=self.robot.data
    origins=getattr(self._env.scene,'env_origins',data.root_link_pos_w.new_zeros(self.num_envs,3))
    gravity=data.projected_gravity_b
    return ((data.root_link_pos_w[:,2]-origins[:,2]>.44) & (gravity[:,0]<-.8) &
      (gravity[:,2].abs()<.35) & (data.root_link_ang_vel_b.norm(dim=-1)<2.) &
      (data.root_link_lin_vel_w.norm(dim=-1)<self.cfg.handoff_linear_speed) &
      ~self.contacts[:,:2].any(-1) &
      (self.rear_contact_age<self.cfg.recent_rear_seconds).all(-1) &
      (self.fr_error.norm(dim=-1)<self.cfg.handoff_fr_error) & (data.joint_vel[:,self.joint_ids].norm(dim=-1)<20.))

  def advance(self):
    """Observe post-physics state once before termination/reward/reset."""
    step=self._env.common_step_counter
    if self._last_advance_step==step:
      return
    self._last_advance_step=step
    dt=self._env.step_dt
    live=self.owner!=DONE
    source=live & ((self.action_owner==WARM)|(self.action_owner==PREFIX))
    bridge=live & (self.action_owner==BRIDGE)
    verify=live & (self.action_owner==VERIFY)
    self.attempt_elapsed+=dt*live
    self.prefix_elapsed+=dt*(self.action_owner==PREFIX)*live
    self.bridge_elapsed+=dt*bridge
    self.verify_elapsed+=dt*verify
    self.tail_steps+=verify.long()
    self.rise_progress_delta[:]=(self.rise_score()-self._previous_rise_score)/dt*bridge
    self.gait_phase[:]=(self.gait_phase+dt*1.5)%1.
    self.rear_contact_age[:]=torch.where(self.contacts[:,2:],0.,self.rear_contact_age+dt)
    failed=self._physical_failure()
    self._finish(source & failed,-2,PHYSICAL_FAILURE)
    self._finish((bridge|verify) & failed,-1,PHYSICAL_FAILURE)
    self._finish(source & (self.attempt_elapsed>self.cfg.source_deadline),-2,SOURCE_DEADLINE)
    self._finish(bridge & (self.bridge_elapsed>self.cfg.bridge_deadline),-1,BRIDGE_DEADLINE)
    self._finish(source & (self.attempt_elapsed>self.cfg.attempt_deadline),-2,ATTEMPT_DEADLINE)
    self._finish((bridge|verify) & (self.attempt_elapsed>self.cfg.attempt_deadline),-1,ATTEMPT_DEADLINE)
    warmed=(self.owner==WARM) & (self.attempt_elapsed>=self.warm_duration)
    self.owner[warmed]=PREFIX
    # The command held during physics must already be at the start height.
    begin=(self.owner==PREFIX) & (self._command[:,5]>=self.h_start-1e-6)
    self.start_bridge(begin.nonzero().flatten())
    candidate=self._biped_candidate()
    ready=(self.owner==BRIDGE) & (self.progress>=1.-1e-6) & candidate
    self.candidate_elapsed[:]=torch.where(ready,self.candidate_elapsed+dt,0.)
    handoff=(self.owner==BRIDGE) & (self.candidate_elapsed>=self.cfg.candidate_seconds-1e-6)
    self.owner[handoff]=VERIFY
    self.verify_elapsed[handoff]=0.; self.tail_steps[handoff]=0
    self.handoff_count+=handoff.long()
    self.invalid_elapsed[:]=torch.where(verify & ~candidate,self.invalid_elapsed+dt,0.)
    self._finish(verify & (self.invalid_elapsed>=self.cfg.invalid_seconds),-1,TAIL_INVALID)
    self._finish(verify & candidate & (self.verify_elapsed>=self.cfg.verify_seconds-1e-5),1,TAIL_SUCCESS)
    self.metrics['bridge_owner'][:]=(self.action_owner==BRIDGE).float()
    self.metrics['source_rejected'][:]=(self.outcome_event==-2).float()
    self.metrics['handoff_success'][:]=(self.outcome_event==1).float()
    self.metrics['handoff_failed'][:]=(self.outcome_event==-1).float()
    self.metrics['handoff_count'][:]=self.handoff_count.float()
