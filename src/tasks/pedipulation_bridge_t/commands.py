"""Physical handoffs with independent controller ownership and source ABI."""
from dataclasses import dataclass
import math

import torch
from src.tasks.teacher_common.anchor_frame import to_anchor
from mjlab.utils.lab_api.math import quat_mul, quat_apply, quat_apply_inverse
from .curriculum import TrackingCourse

from src.tasks.teacher_common.commands import (
  LocoPedipulationTeacherCommand, LocoPedipulationTeacherCommandCfg,
)
from src.tasks.loco_pedipulation.mdp.commands import QUAD, HOLD, smoothstep
from src.tasks.pedipulation_transition_t.motion import MotionReference, NOMINAL_MOTION_PATH
from .experts import JOINT_NAMES
from .guidance import RiseGuidance, SUPPORT_JOINTS

WARM, PREFIX, BRIDGE, VERIFY, DONE = range(5)
BRIDGE_CONTRACT = 'three_course_reference_tracking_v3'
NO_REASON, PHYSICAL_FAILURE, SOURCE_DEADLINE, BRIDGE_DEADLINE, TAIL_INVALID, ATTEMPT_DEADLINE, TAIL_SUCCESS = range(7)
EVENT_REASON_NAMES = ('none','physical_failure','source_deadline','bridge_deadline',
                      'tail_invalid','attempt_deadline','tail_success')


@dataclass(kw_only=True)
class BridgeCommandCfg(LocoPedipulationTeacherCommandCfg):
  resampling_time_range: tuple[float,float] = (1e9,1e9)
  initial_stage: int = 0
  curriculum_enabled: bool = True
  play: bool = False
  verify_handoff: bool = False
  min_stage_steps: int = 1500
  min_curriculum_episodes: int = 128
  success_window: int = 512
  promotion_success: float = .85
  undisturbed_probability: float = .35
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
  bridge_deadline: float = 10.
  attempt_deadline: float = 24.
  verify_seconds: float = 2.5
  candidate_seconds: float = 1.
  recent_rear_seconds: float = .3
  invalid_seconds: float = .2
  reference_speed_range: tuple[float,float] = (.85,1.15)
  reference_blend_seconds: float = .3
  fr_tracking_scale: tuple[float,float] = (.08,.08)
  handoff_fr_error: float = .1
  handoff_linear_speed: float = .5
  verification_velocity: tuple[float,float,float] = (0.,0.,0.)
  initial_impulse_probability: float = .65
  initial_linear_impulse: tuple[float,float,float] = (.12,.12,.12)
  initial_angular_impulse: tuple[float,float,float] = (.18,.18,.18)
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
    self.course=TrackingCourse(stage=cfg.initial_stage,enabled=cfg.curriculum_enabled,
      minimum_episodes=cfg.min_curriculum_episodes,window_size=cfg.success_window,
      minimum_steps=cfg.min_stage_steps,success_threshold=cfg.promotion_success)
    if not .3<=cfg.undisturbed_probability<=1.:
      raise ValueError('At least30% undisturbed samples are required')
    self.forced_start=None; self.forced_bank=None; self.forced_live_probability=None
    self.forced_disturbed=None
    self.forced_moving=None
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
    self.initial_fl_height=torch.zeros(n,device=self.device)
    self.start_linear_velocity=torch.zeros(n,3,device=self.device)
    self.start_angular_velocity=torch.zeros(n,3,device=self.device)
    self.start_fr_error=torch.zeros(n,device=self.device)
    self.rise_progress_delta=torch.zeros(n,device=self.device)
    self._previous_rise_score=torch.zeros(n,device=self.device)
    self.low_offset=torch.zeros(n,3,device=self.device)
    self.end_offset=torch.zeros_like(self.low_offset)
    self.start_offset=torch.zeros_like(self.low_offset)
    self.live_start=torch.zeros(n,device=self.device,dtype=torch.bool)
    self.full_start=torch.zeros_like(self.live_start)
    self.disturbed=torch.zeros_like(self.live_start)
    self.start_time=torch.zeros(n,device=self.device)
    self.bank_index=torch.zeros(n,device=self.device,dtype=torch.long)
    self.yaw=torch.zeros(n,device=self.device)
    self.initial_root_state=torch.zeros(n,13,device=self.device)
    self.initial_root_state[:,3]=1.
    self.held_success=torch.zeros_like(self.live_start)
    self.post_pushed=torch.zeros_like(self.live_start); self.mid_pushed=torch.zeros_like(self.live_start)
    self.first_stand_time=torch.full((n,),float('nan'),device=self.device)
    self.completed_first_stand_time=self.first_stand_time.clone()
    self._reference_cache=None; self._reference_cache_key=None
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
    return ((self.start_time+self.bridge_elapsed*self.reference_speed)/self.guidance.rise_duration).clamp(0.,1.)

  @property
  def course_stage(self):
    return self.course.stage

  def _origins(self):
    return getattr(self._env.scene,'env_origins',self.start_time.new_zeros(self.num_envs,3))

  def _reference_at(self, elapsed):
    time=self.start_time+elapsed*self.reference_speed
    values=self.guidance.sample_time(time,speed=self.reference_speed,bank_index=self.bank_index)
    heading=values['quaternion'].new_zeros(self.num_envs,4)
    heading[:,0]=(self.yaw/2).cos(); heading[:,3]=(self.yaw/2).sin()
    root=values['root_state'].clone()
    root[:,:3]=quat_apply(heading,root[:,:3])+self._origins()
    root[:,3:7]=quat_mul(heading,root[:,3:7])
    # Align horizontal translation at the entry, without rewarding absoluteXY.
    origin0=self.guidance.frames['root_state'][0,:3].expand(self.num_envs,3)
    xyshift=self.initial_root_state[:,:2]-quat_apply(heading,origin0)[:,:2]-self._origins()[:,:2]
    root[:,:2]+=xyshift*self.live_start[:,None]
    alpha=smoothstep(elapsed/self.cfg.reference_blend_seconds)
    alpha=torch.where(self.live_start & self.reference_started,alpha,torch.ones_like(alpha))
    root[:,:3]=torch.lerp(self.initial_root_state[:,:3],root[:,:3],alpha[:,None])
    q0=self.initial_root_state[:,3:7]; q1=root[:,3:7]
    q1=torch.where((q0*q1).sum(-1,keepdim=True)<0.,-q1,q1)
    dot=(q0*q1).sum(-1,keepdim=True).clamp(-1.,1.)
    theta=dot.acos(); sine=theta.sin().clamp_min(1e-7); a=alpha[:,None]
    q=(torch.sin((1.-a)*theta)*q0+torch.sin(a*theta)*q1)/sine
    q=torch.where(dot>.9995,torch.lerp(q0,q1,a),q)
    root[:,3:7]=torch.nn.functional.normalize(q,dim=-1)
    values['root_state']=root; values['quaternion']=root[:,3:7]
    values['height']=root[:,2]-self._origins()[:,2]
    values['gravity']=quat_apply_inverse(root[:,3:7],root.new_tensor((0.,0.,-1.)).expand(self.num_envs,3))
    values['joint_pos']=torch.lerp(self.initial_joint_pos,values['joint_pos'],a)
    values['fl_position']=torch.lerp(self.initial_fl_position,values['fl_position'],a)
    values['fl_height']=torch.lerp(self.initial_fl_height,
      self._origins()[:,2]+values['fl_height'],alpha)
    return values

  @property
  def reference_state(self):
    # One interpolated trajectory per physics state. Version counters also
    # invalidate explicit same-step phase changes and partial reference resets.
    tensors=(self.bridge_elapsed,self.start_time,self.reference_speed,self.bank_index,
      self.yaw,self.live_start,self.reference_started,self.initial_root_state,
      self.initial_joint_pos,self.initial_fl_position,self.initial_fl_height)
    try: key=(self._env.common_step_counter,tuple(t._version for t in tensors))
    except RuntimeError: key=None  # Fully inference-created environments have no versions.
    if key is not None and key==self._reference_cache_key:
      return self._reference_in_anchor(self._reference_cache)
    values=self._reference_at(self.bridge_elapsed)
    # Differentiate the blended trajectory, including its changing blend.
    eps=.001
    future=self._reference_at(self.bridge_elapsed+eps)
    values['joint_vel']=(future['joint_pos']-values['joint_pos'])/eps
    linear=(future['root_state'][:,:3]-values['root_state'][:,:3])/eps
    q=values['quaternion']; inv=torch.cat((q[:,:1],-q[:,1:]),-1)
    delta=quat_mul(future['quaternion'],inv)
    delta=torch.where(delta[:,:1]<0.,-delta,delta)
    v=delta[:,1:]; norm=v.norm(dim=-1,keepdim=True)
    angular=v*(2.*torch.atan2(norm,delta[:,:1])/norm.clamp_min(1e-8))/eps
    values['root_state'][:,7:10]=linear
    values['root_state'][:,10:13]=angular
    self._reference_cache=values; self._reference_cache_key=key
    return self._reference_in_anchor(values)

  def _reference_in_anchor(self, values):
    # EntityData can return an inference-created quaternion temporary. Cache
    # world references separately and refresh the actual anchor on every access.
    basis=self.basis_w
    values['linear_velocity']=to_anchor(basis,values['root_state'][:,7:10])
    values['angular_velocity']=to_anchor(basis,values['root_state'][:,10:13])
    return values

  @property
  def orientation_error_b(self):
    actual=self.robot.data.root_link_quat_w
    conjugate=torch.cat((actual[:,:1],-actual[:,1:]),-1)
    error=quat_mul(conjugate,self.reference_state['quaternion'])
    error=torch.where(error[:,:1]<0.,-error,error)
    v=error[:,1:]; norm=v.norm(dim=-1,keepdim=True)
    return v*(2.*torch.atan2(norm,error[:,:1])/norm.clamp_min(1e-8))

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
    self.initial_root_state[ids]=torch.cat((data.root_link_pos_w[ids],data.root_link_quat_w[ids],
      data.root_link_lin_vel_w[ids],data.root_link_ang_vel_w[ids]),-1)
    heading=self.basis_w[:,:,0]
    self.yaw[ids]=torch.atan2(heading[ids,1],heading[ids,0])
    fl=to_anchor(self.basis_w,data.site_pos_w[:,self.site_ids[0]]-data.root_link_pos_w)
    self.initial_fl_position[ids]=fl[ids]
    self.initial_fl_height[ids]=data.site_pos_w[ids,self.site_ids[0],2]
    self.start_linear_velocity[ids]=to_anchor(self.basis_w,data.root_link_lin_vel_w)[ids]
    self.start_angular_velocity[ids]=data.root_link_ang_vel_b[ids]
    self.start_fr_error[ids]=self.fr_error.norm(dim=-1)[ids]
    self._previous_rise_score[ids]=self.rise_score()[ids]
    active=self.disturbed[ids] & (self.course_stage>=1) & (torch.rand(len(ids),device=self.device)<self.cfg.initial_impulse_probability)
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
    if self.forced_moving is not None: self.moving[env_ids]=self.forced_moving[env_ids]
    self.requested_velocity[env_ids]=0.
    for axis,bounds in ((0,self.cfg.lin_vel_x),(2,self.cfg.ang_vel_z)):
      values=torch.empty(count,device=self.device).uniform_(*bounds)
      self.requested_velocity[env_ids,axis]=values*self.moving[env_ids]
    self.requested_offset[env_ids]=starts
    self.owner[env_ids]=WARM
    probability=self.course.live_probability if self.forced_live_probability is None else self.forced_live_probability
    self.live_start[env_ids]=torch.rand(count,device=self.device)<probability
    self.disturbed[env_ids]=(torch.rand(count,device=self.device)>=self.cfg.undisturbed_probability)&(self.course_stage>=1)&(not self.cfg.play)
    if self.forced_disturbed is not None: self.disturbed[env_ids]=self.forced_disturbed
    self.reference_speed[env_ids]=1.
    if self.course_stage>=1:
      self.reference_speed[env_ids]=torch.empty(count,device=self.device).uniform_(*self.cfg.reference_speed_range)
    self.bank_index[env_ids]=torch.randint(len(self.guidance.fr_offsets),(count,),device=self.device)
    self.bank_index[env_ids[torch.rand(count,device=self.device)<self.cfg.undisturbed_probability]]=0
    if self.forced_bank is not None: self.bank_index[env_ids]=self.forced_bank
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
    self.held_success[env_ids]=False; self.post_pushed[env_ids]=False; self.mid_pushed[env_ids]=False
    self.first_stand_time[env_ids]=float('nan')
    self.start_time[env_ids]=0.
    self.full_start[env_ids]=False
    live_ids=env_ids[self.live_start[env_ids]]
    data=self.robot.data
    if len(live_ids) and hasattr(data,'default_root_state'):
      root=data.default_root_state[live_ids].clone()
      root[:,:3]+=self._origins()[live_ids]; root[:,7:]=0.
      q=data.default_joint_pos[live_ids][:,self.joint_ids]
      self.robot.write_root_state_to_sim(root,env_ids=live_ids)
      self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=self.joint_ids,env_ids=live_ids)
    reference_ids=env_ids[~self.live_start[env_ids]]
    if len(reference_ids):
      self.moving[reference_ids]=False
      sample=torch.rand(len(reference_ids),device=self.device)
      starts=torch.zeros_like(sample)
      rising=(sample>=.5)&(sample<.8)
      starts[rising]=torch.empty(int(rising.sum()),device=self.device).uniform_(.5,self.guidance.rise_duration)
      starts[sample>=.8]=self.guidance.rise_duration+.5
      forced=self.forced_start if self.forced_start is not None else ('full' if self.cfg.play else None)
      if forced=='full': starts[:]=0.
      elif forced=='rise': starts.uniform_(.5,self.guidance.rise_duration)
      elif forced=='endpoint': starts[:]=self.guidance.rise_duration+.5
      elif forced is not None: raise ValueError('Unknown forced reference start')
      self.start_time[reference_ids]=starts
      self.full_start[reference_ids]=starts==0.
      self.yaw[reference_ids]=0. if self.cfg.play else torch.empty(len(reference_ids),device=self.device).uniform_(-torch.pi,torch.pi)
      state=self.guidance.sample_time(starts,speed=self.reference_speed[reference_ids],bank_index=self.bank_index[reference_ids])
      root=state['root_state'].clone()
      yaw=self.yaw[reference_ids]; heading=root.new_zeros(len(reference_ids),4)
      heading[:,0]=(yaw/2).cos(); heading[:,3]=(yaw/2).sin()
      root[:,:3]=quat_apply(heading,root[:,:3])+self._origins()[reference_ids]
      root[:,3:7]=quat_mul(heading,root[:,3:7])
      root[:,7:10]=quat_apply(heading,root[:,7:10]); root[:,10:13]=quat_apply(heading,root[:,10:13])
      self.initial_root_state[reference_ids]=root
      self.initial_joint_pos[reference_ids]=state['joint_pos']
      self.initial_fl_height[reference_ids]=self._origins()[reference_ids,2]+state['fl_height']
      self.robot.write_root_state_to_sim(root,env_ids=reference_ids)
      self.robot.write_joint_state_to_sim(state['joint_pos'],state['joint_vel'],joint_ids=self.joint_ids,env_ids=reference_ids)
      self.owner[reference_ids]=BRIDGE
      self.start_linear_velocity[reference_ids]=root[:,7:10]
      self.start_angular_velocity[reference_ids]=root[:,10:13]
      bounds=root.new_tensor((*self.cfg.initial_linear_impulse,*self.cfg.initial_angular_impulse))
      active=self.disturbed[reference_ids] & (self.course_stage>=1)
      self.initial_impulse[reference_ids]=(torch.rand(len(reference_ids),6,device=self.device)*2.-1.)*bounds*active[:,None]
      self.pending_initial_impulse[reference_ids]=active
    if hasattr(self._env,'sim'):
      from .physics import reset_tracking_physics
      reset_tracking_physics(self._env,env_ids,self.course.randomization_fraction)
    action=self._env.action_manager.get_term('joint_pos')
    action.cfg.delay=self.course_stage>=2
    self._pending_reset[env_ids]=False
    self._command[reference_ids,3:]=state['fr_offset'] if len(reference_ids) else self._command[reference_ids,3:]

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
    # Course1: recovery after standing. Course2 also perturbs mid-rise.
    eligible=self.disturbed & (self.owner==BRIDGE) & (self.course_stage>=1)
    post=eligible & ~self.post_pushed & self.held_success & (self.progress>=1.) & (self.bridge_elapsed>=3.5)
    mid=eligible & ~self.mid_pushed & (self.course_stage>=2) & (self.progress>=.54) & (self.progress<.84)
    ids=(post|mid).nonzero().flatten()
    if len(ids):
      data=self.robot.data
      vel=torch.cat((data.root_link_lin_vel_w[ids],data.root_link_ang_vel_w[ids]),-1)
      fraction=torch.where(mid[ids],self.course.randomization_fraction,1.)[:,None]
      vel[:,:2]+=(torch.rand(len(ids),2,device=self.device)*2.-1.)*.2*fraction
      vel[:,3:]+=(torch.rand(len(ids),3,device=self.device)*2.-1.)*.3*fraction
      self.robot.write_root_link_velocity_to_sim(vel,env_ids=ids)
      self.post_pushed[post]=True; self.mid_pushed[mid]=True
      self.candidate_elapsed[ids]=0.; self.held_success[ids]=False

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
    reference=remaining & ~self.live_start
    self._command[reference,3:]=self._reference_at(self.bridge_elapsed)['fr_offset'][reference]
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
    self.completed_first_stand_time[mask]=self.first_stand_time[mask]

  def _physical_failure(self):
    data=self.robot.data
    state=torch.cat((data.root_link_pos_w,data.root_link_quat_w,data.joint_pos,
                     data.joint_vel,data.root_link_ang_vel_b,
                     data.root_link_lin_vel_w,data.root_link_ang_vel_w),-1)
    origins=getattr(self._env.scene,'env_origins',data.root_link_pos_w.new_zeros(self.num_envs,3))
    hard=(~torch.isfinite(state).all(-1)) | ((data.root_link_pos_w[:,2]-origins[:,2])<.12)
    limits=getattr(data,'soft_joint_pos_limits',None)
    if limits is not None:
      q=data.joint_pos[:,self.joint_ids]
      hard|=((q<limits[:,self.joint_ids,0]-.25)|(q>limits[:,self.joint_ids,1]+.25)).any(-1)
    for sensor in ('base_contact','nonfoot_ground_touch','bridge_self_contact'):
      try:
        force=self._env.scene[sensor].data.force
      except KeyError:
        continue
      hard |= force.reshape(self.num_envs,-1,3).norm(dim=-1).amax(-1)>10.
    return hard

  def biped_conditions(self):
    """Named handoff gates, shared by training and keyboard diagnostics."""
    data=self.robot.data
    origins=getattr(self._env.scene,'env_origins',data.root_link_pos_w.new_zeros(self.num_envs,3))
    gravity=data.projected_gravity_b
    ref=self.reference_state
    orientation=2.*torch.acos((data.root_link_quat_w*ref['quaternion']).sum(-1).abs().clamp(0.,1.))
    return {
      'orientation':orientation<torch.pi/12.,
      'height_error':(data.root_link_pos_w[:,2]-origins[:,2]-ref['height']).abs()<.08,
      'base_height':data.root_link_pos_w[:,2]-origins[:,2]>.44,
      'upright':(gravity[:,0]<-.8)&(gravity[:,2].abs()<.35),
      'angular_speed':data.root_link_ang_vel_b.norm(dim=-1)<2.,
      'linear_speed':data.root_link_lin_vel_w.norm(dim=-1)<self.cfg.handoff_linear_speed,
      'front_clear':~self.contacts[:,:2].any(-1),
      'rear_support':self.contacts[:,2:].any(-1),
      'recent_rears':(self.rear_contact_age<self.cfg.recent_rear_seconds).all(-1),
      'fr_error':self.fr_error.norm(dim=-1)<self.cfg.handoff_fr_error,
      'joint_speed':data.joint_vel[:,self.joint_ids].norm(dim=-1)<20.,
    }

  def _biped_candidate(self):
    return torch.stack(tuple(self.biped_conditions().values()),dim=-1).all(-1)

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
    self.gait_phase[:]=(self.gait_phase+dt*1.5)%1.
    self.rear_contact_age[:]=torch.where(self.contacts[:,2:],0.,self.rear_contact_age+dt)
    failed=self._physical_failure()
    self._finish(source & failed,-2,PHYSICAL_FAILURE)
    self._finish((bridge|verify) & failed,-1,PHYSICAL_FAILURE)
    self._finish(source & (self.attempt_elapsed>self.cfg.source_deadline),-2,SOURCE_DEADLINE)
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
    self.held_success[:]=(self.candidate_elapsed>=self.cfg.candidate_seconds-1e-6)&(self.owner==BRIDGE)
    first=self.held_success & ~torch.isfinite(self.first_stand_time)
    self.first_stand_time[first]=self.bridge_elapsed[first]
    finished_tracking=(self.owner==BRIDGE)&(self.bridge_elapsed>=self.cfg.bridge_deadline-1e-5)
    if not self.cfg.verify_handoff:
      self._finish(finished_tracking & self.held_success,1,TAIL_SUCCESS)
      self._finish(finished_tracking & ~self.held_success,-1,BRIDGE_DEADLINE)
    handoff=(self.owner==BRIDGE) & self.held_success & self.cfg.verify_handoff
    self._finish(finished_tracking & ~self.held_success,-1,BRIDGE_DEADLINE)
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
