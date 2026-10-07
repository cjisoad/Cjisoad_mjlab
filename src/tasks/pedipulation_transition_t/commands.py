"""Fixed-clock transition reference, isolated resets and held-success course."""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul
from src.tasks.pedipulation.constants import JOINT_NAMES
from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
from .motion import FOOT_NAMES, MOTION_PATH, MotionReference, SUPPORT_JOINTS


@dataclass(kw_only=True)
class TransitionCommandCfg(CommandTermCfg):
  motion_path: str = str(MOTION_PATH)
  play: bool = False
  curriculum_enabled: bool = True
  initial_stage: int = 0
  min_stage_steps: int = 1500
  success_window: int = 512
  minimum_full_starts: int = 128
  promotion_success: float = .85
  hold_seconds: float = 1.
  failure_persistence: float = .2
  timing_range: tuple[float,float] = (.85,1.15)
  undisturbed_probability: float = .35

  def build(self, env):
    return TransitionCommand(self,env)


class TransitionCommand(CommandTerm):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    if not 0 <= cfg.initial_stage <= 5 or cfg.minimum_full_starts < 128:
      raise ValueError('Require stage 0..5 and at least 128 full-start episodes')
    if not .3 <= cfg.undisturbed_probability <= 1.:
      raise ValueError('At least 30% of worlds must be entirely undisturbed')
    self.robot=env.scene['robot']; self.motion=MotionReference(cfg.motion_path)
    moving=np.flatnonzero(np.linalg.norm(self.motion.data['joint_vel'],axis=-1)>1e-7)
    derived_start=float(self.motion.data['time'][min(int(moving[-1])+1,len(self.motion.data['time'])-1)]) if len(moving) else 0.
    self.stand_start=float(getattr(self.motion,'stand_start',derived_start))
    self.rise_start=.5
    self.stage=cfg.initial_stage; self.stage_started_step=0
    self.full_history=deque(maxlen=cfg.success_window)
    self.reset_history=deque(maxlen=cfg.success_window)
    self.completed_full_starts=0; self.completed_reference_starts=0
    self.forced_start=None; self.forced_bank=None
    ids,names=self.robot.find_joints(JOINT_NAMES,preserve_order=True)
    if tuple(names)!=JOINT_NAMES: raise ValueError('Transition requires canonical joint ordering')
    self.joint_ids=torch.tensor(ids,device=self.device)
    sites,names=self.robot.find_sites(FOOT_NAMES,preserve_order=True)
    if tuple(names)!=FOOT_NAMES: raise ValueError('Transition requires FL/FR/RL/RR foot order')
    self.site_ids=torch.tensor(sites,device=self.device)
    self.support_ids=torch.tensor(SUPPORT_JOINTS,device=self.device)
    d=self.motion.data
    self.frames={key:torch.as_tensor(d[key],dtype=torch.float32,device=self.device)
      for key in ('root_state','joint_pos','joint_vel','foot_pos_w','contact','fl_unload')}
    # Older references remain usable for nominal debugging. Diversity requires
    # the offline IK/clearance checked bank rather than online target guesses.
    self.zero=torch.as_tensor(d.get('foot_zero',np.zeros(3)),dtype=torch.float32,device=self.device)
    nominal=d['foot_pos_w'][:,1]-d['root_state'][:,:3]-self.zero.cpu().numpy()
    self.fr_offsets=torch.as_tensor(d.get('fr_offset_bank',nominal[None]),dtype=torch.float32,device=self.device)
    self.fr_joints=torch.as_tensor(d.get('fr_joint_bank',d['joint_pos'][None,:,3:6]),dtype=torch.float32,device=self.device)
    nframes=len(d['time']); k=len(self.fr_offsets)
    if self.fr_offsets.shape!=(k,nframes,3) or self.fr_joints.shape!=(k,nframes,3):
      raise ValueError('FR trajectory banks must have matching (K,T,3) shapes')
    if not torch.isfinite(self.fr_offsets).all() or not torch.isfinite(self.fr_joints).all():
      raise ValueError('FR trajectory banks must be finite')
    self.fr_velocities=torch.gradient(self.fr_joints,spacing=(1./self.motion.fps,),dim=(1,))[0]
    self.fr_velocities[:,(d['time']<=self.rise_start)|(d['time']>=self.stand_start)]=0.
    n=self.num_envs
    self.start_time=torch.zeros(n,device=self.device); self.speed=torch.ones(n,device=self.device)
    self.yaw=torch.zeros(n,device=self.device); self.bank_index=torch.zeros(n,dtype=torch.long,device=self.device)
    self.reference_time=torch.zeros(n,device=self.device)
    self.full_start=torch.zeros(n,dtype=torch.bool,device=self.device)
    self.disturbed=torch.zeros_like(self.full_start); self.initialized=torch.zeros_like(self.full_start)
    self.held_success=torch.zeros_like(self.full_start); self.failure_mask=torch.zeros_like(self.full_start)
    self.last_episode_success=torch.zeros_like(self.full_start)
    self.hold_time=torch.zeros(n,device=self.device); self.failure_time=torch.zeros(n,device=self.device)
    self.rear_contact_age=torch.full((n,2),1.,device=self.device)
    self.mid_pushed=torch.zeros_like(self.full_start); self.post_pushed=torch.zeros_like(self.full_start)
    self._outcome_step=torch.full((n,),-1,dtype=torch.long,device=self.device)
    self.ref_root_state=torch.zeros(n,13,device=self.device)
    self.ref_root_state[:,3]=1.
    self.ref_joint_pos=torch.zeros(n,12,device=self.device); self.ref_joint_vel=torch.zeros_like(self.ref_joint_pos)
    self.ref_foot_pos_w=torch.zeros(n,4,3,device=self.device)
    self.ref_contact=torch.zeros(n,4,device=self.device); self.ref_fl_unload=torch.zeros(n,device=self.device)
    self.ref_fr_offset=torch.zeros(n,3,device=self.device)
    self.ref_gravity=torch.zeros(n,3,device=self.device)
    self._command=torch.zeros(n,6,device=self.device)
    self._reference_step=-1
    self.metrics={name:torch.zeros(n,device=self.device) for name in
      ('held_success','full_start','orientation_error','height_error','fr_error','course_stage')}
    self.synchronize_reference()

  @property
  def basis_w(self):
    return anchor_basis(self.robot.data.root_link_quat_w,self.robot.data.gravity_vec_w)

  @property
  def command(self):
    self._ensure_reference(); return self._command

  @property
  def foot_pos_anchor(self):
    return to_anchor(self.basis_w,self.robot.data.site_pos_w[:,self.site_ids[1]]-self.robot.data.root_link_pos_w)

  @property
  def foot_target_pos_anchor(self):
    self._ensure_reference(); return self.zero+self.ref_fr_offset

  @property
  def foot_target_pos_w(self):
    return self.robot.data.root_link_pos_w+(self.basis_w@self.foot_target_pos_anchor[...,None]).squeeze(-1)

  @property
  def fr_error(self):
    return self.foot_target_pos_anchor-self.foot_pos_anchor

  @property
  def progress(self):
    self._ensure_reference(); return (self.reference_time/max(self.stand_start,1e-6)).clamp(0.,1.)

  @property
  def ref_linear_velocity(self):
    self._ensure_reference(); return to_anchor(self.basis_w,self.ref_root_state[:,7:10])

  @property
  def ref_angular_velocity(self):
    self._ensure_reference(); return to_anchor(self.basis_w,self.ref_root_state[:,10:13])

  @property
  def orientation_error_b(self):
    """Reference rotation vector in the current body, including heading error.

    Projected reference gravity alone cannot expose accumulated yaw to Actor84.
    The three-vector preserves its dimensional contract and makes the full
    quaternion tracking objective observable to this privileged teacher.
    """
    self._ensure_reference()
    actual=self.robot.data.root_link_quat_w
    conjugate=torch.cat((actual[:,:1],-actual[:,1:]),-1)
    error=torch.nn.functional.normalize(quat_mul(conjugate,self.ref_root_state[:,3:7]),dim=-1)
    error=torch.where(error[:,:1]<0.,-error,error)
    vector=error[:,1:]; norm=vector.norm(dim=-1,keepdim=True)
    angle=2.*torch.atan2(norm,error[:,:1])
    return vector*(angle/norm.clamp_min(1e-8))

  def _origins(self):
    return getattr(self._env.scene,'env_origins',self.start_time.new_zeros(self.num_envs,3))

  def _sample(self,time,ids):
    position=time.clamp(0.,self.motion.duration)*self.motion.fps
    low=position.long().clamp_max(len(self.frames['joint_pos'])-2); alpha=position-low
    result={}
    for name,array in self.frames.items():
      a=alpha.reshape((-1,)+(1,)*(array.ndim-1))
      result[name]=(1.-a)*array[low]+a*array[low+1]
    bank=self.bank_index[ids]; a=alpha[:,None]
    for name,array in (('offset',self.fr_offsets),('frq',self.fr_joints),('frqd',self.fr_velocities)):
      result[name]=(1.-a)*array[bank,low]+a*array[bank,low+1]
    result['joint_pos'][:,3:6]=result['frq']; result['joint_vel'][:,3:6]=result['frqd']
    root=result['root_state']; root[:,3:7]=torch.nn.functional.normalize(root[:,3:7],dim=-1)
    heading=root.new_zeros(len(ids),4); heading[:,0]=(self.yaw[ids]/2).cos(); heading[:,3]=(self.yaw[ids]/2).sin()
    origin=self._origins()[ids]
    root[:,:3]=quat_apply(heading,root[:,:3])+origin
    root[:,3:7]=quat_mul(heading,root[:,3:7])
    root[:,7:10]=quat_apply(heading,root[:,7:10]); root[:,10:13]=quat_apply(heading,root[:,10:13])
    factors=self.speed[ids]*(time<self.motion.duration)
    root[:,7:]*=factors[:,None]; result['joint_vel']*=factors[:,None]
    feet=result['foot_pos_w']; result['foot_pos_w']=quat_apply(heading[:,None,:].expand(-1,4,-1).reshape(-1,4),feet.reshape(-1,3)).reshape(-1,4,3)+origin[:,None,:]
    # Bank offsets are in the heading-aligned common anchor, not body axes.
    result['foot_pos_w'][:,1]=root[:,:3]+quat_apply(heading,self.zero+result['offset'])
    return result

  def synchronize_reference(self):
    episode=getattr(self._env,'episode_length_buf',self.start_time.new_zeros(self.num_envs))
    time=self.start_time+episode*self._env.step_dt*self.speed
    self.reference_time.copy_(time.clamp_max(self.motion.duration))
    ids=torch.arange(self.num_envs,device=self.device)
    sample=self._sample(time,ids)
    self.ref_root_state.copy_(sample['root_state']); self.ref_joint_pos.copy_(sample['joint_pos'])
    self.ref_joint_vel.copy_(sample['joint_vel']); self.ref_foot_pos_w.copy_(sample['foot_pos_w'])
    self.ref_contact.copy_(sample['contact']); self.ref_fl_unload.copy_(sample['fl_unload'])
    self.ref_fr_offset.copy_(sample['offset'])
    gravity=self.ref_root_state.new_tensor([0.,0.,-1.]).expand(self.num_envs,3)
    self.ref_gravity.copy_(quat_apply_inverse(self.ref_root_state[:,3:7],gravity))
    self._command[:,:2]=to_anchor(self.basis_w,self.ref_root_state[:,7:10])[:,:2]
    self._command[:,2]=to_anchor(self.basis_w,self.ref_root_state[:,10:13])[:,2]
    self._command[:,3:]=self.ref_fr_offset
    self._reference_step=self._env.common_step_counter

  def _ensure_reference(self):
    if self._reference_step!=self._env.common_step_counter:
      self.synchronize_reference()

  def record_episodes(self,success,full_start):
    full=full_start.detach().cpu().bool(); success=success.detach().cpu().bool()
    self.full_history.extend(success[full].tolist()); self.reset_history.extend(success[~full].tolist())
    self.completed_full_starts+=int(full.sum()); self.completed_reference_starts+=int((~full).sum())
    age=self._env.common_step_counter-self.stage_started_step
    if (not self.cfg.play and self.cfg.curriculum_enabled and self.stage<5 and age>=self.cfg.min_stage_steps and
        len(self.full_history)>=self.cfg.minimum_full_starts and
        sum(self.full_history)/len(self.full_history)>=self.cfg.promotion_success):
      self.stage+=1; self.stage_started_step=self._env.common_step_counter
      self.full_history.clear(); self.reset_history.clear()

  def course_state_dict(self):
    return dict(stage=self.stage,stage_started_step=self.stage_started_step,
      stage_elapsed_steps=self._env.common_step_counter-self.stage_started_step,
      full_history=list(self.full_history),reset_history=list(self.reset_history),
      completed_full_starts=self.completed_full_starts,
      completed_reference_starts=self.completed_reference_starts)

  def load_course_state_dict(self,state):
    if not 0<=int(state['stage'])<=5: raise ValueError('Invalid saved course stage')
    for name in ('stage','stage_started_step','completed_full_starts','completed_reference_starts'):
      setattr(self,name,int(state[name]))
    for name in ('full_history','reset_history'):
      history=getattr(self,name); history.clear(); history.extend(bool(v) for v in state[name])
    self.stage_started_step=self._env.common_step_counter-int(state.get('stage_elapsed_steps',0))
    # A restored course applies its noise/delay/physical state on fresh resets.
    if hasattr(self._env,'action_manager'):
      self._env.action_manager.get_term('joint_pos').cfg.delay=self.stage>=5 and not self.cfg.play

  def reset(self,env_ids):
    ids=torch.as_tensor(env_ids,dtype=torch.long,device=self.device)
    old=ids[self.initialized[ids]&(self._env.episode_length_buf[ids]>0)]
    # Keep the completed result available after autoreset clears held_success.
    # Empty explicit resets carry no completed outcome and must not reuse one.
    self.last_episode_success[ids]=False
    if len(old):
      terminated=self.failure_mask
      if hasattr(self._env,'termination_manager'):
        terminated=terminated|self._env.termination_manager.terminated
      success=self.held_success[old]&~terminated[old]
      self.last_episode_success[old]=success
      self.record_episodes(success,self.full_start[old])
    extras={name:float(values[ids].mean()) for name,values in self.metrics.items()}
    extras['full_start_success_rate']=sum(self.full_history)/max(1,len(self.full_history))
    extras['reference_start_success_rate']=sum(self.reset_history)/max(1,len(self.reset_history))
    extras['completed_full_starts']=float(self.completed_full_starts)
    self._resample_command(ids)
    return extras

  def _resample_command(self,ids):
    n=len(ids)
    if not n: return
    sample=torch.rand(n,device=self.device); starts=torch.zeros(n,device=self.device)
    rise=(sample>=.5)&(sample<.8); endpoint=sample>=.8
    starts[rise]=torch.empty(int(rise.sum()),device=self.device).uniform_(self.rise_start,self.stand_start)
    starts[endpoint]=self.stand_start+.5
    forced=self.forced_start if self.forced_start is not None else ('full' if self.cfg.play else None)
    if forced is not None:
      if forced=='full': starts[:]=0.
      elif forced=='rise': starts.uniform_(self.rise_start,self.stand_start)
      elif forced=='endpoint': starts[:]=self.stand_start+.5
      else: starts[:]=float(forced)
    self.start_time[ids]=starts; self.full_start[ids]=starts==0
    self.speed[ids]=1.
    if self.stage>=2 and not self.cfg.play:
      self.speed[ids]=torch.empty(n,device=self.device).uniform_(*self.cfg.timing_range)
    self.bank_index[ids]=0
    if self.stage>=1: self.bank_index[ids]=torch.randint(len(self.fr_offsets),(n,),device=self.device)
    if self.forced_bank is not None: self.bank_index[ids]=int(self.forced_bank)
    self.yaw[ids]=0. if self.cfg.play else torch.empty(n,device=self.device).uniform_(-torch.pi,torch.pi)
    self.disturbed[ids]=(torch.rand(n,device=self.device)>=self.cfg.undisturbed_probability)&(self.stage>=3)&(not self.cfg.play)
    self.hold_time[ids]=0.; self.failure_time[ids]=0.; self.held_success[ids]=False; self.failure_mask[ids]=False
    self.rear_contact_age[ids]=1.
    self.mid_pushed[ids]=False; self.post_pushed[ids]=False; self._outcome_step[ids]=-1
    self.initialized[ids]=True
    self._reference_step=-1
    for values in self.metrics.values(): values[ids]=0.
    self.metrics['full_start'][ids]=self.full_start[ids].float(); self.metrics['course_stage'][ids]=self.stage
    action=self._env.action_manager.get_term('joint_pos')
    action.raw_action[ids]=0.; action.previous_action[ids]=0.; action.previous_joint_vel[ids]=0.; action.delay_steps[ids]=0
    action.cfg.delay=self.stage>=5 and not self.cfg.play
    # Reset managers have not cleared episode_length_buf yet. Sample the actual
    # chosen start directly, rather than reading stale terminal episode time.
    state=self._sample(starts,ids); root=state['root_state']
    birth=self.disturbed[ids]
    root[:,7:10]+=torch.empty(n,3,device=self.device).uniform_(-.12,.12)*birth[:,None]
    root[:,10:13]+=torch.empty(n,3,device=self.device).uniform_(-.18,.18)*birth[:,None]
    self.robot.write_root_state_to_sim(root,env_ids=ids)
    self.robot.write_joint_state_to_sim(state['joint_pos'],state['joint_vel'],joint_ids=self.joint_ids,env_ids=ids)
    if self.stage>=5 and hasattr(self._env,'sim'):
      from .env_cfg import randomize_transition_physics
      randomize_transition_physics(self._env,ids)
      from mjlab.managers.event_manager import RecomputeLevel
      self._env.sim.recompute_constants(RecomputeLevel.set_const)

  def update_outcomes(self):
    from .rewards import state_errors, contact_state
    eligible=(self._outcome_step!=self._env.common_step_counter)&(self._env.episode_length_buf>0)
    if not eligible.any(): return
    self._ensure_reference(); errors=state_errors(self._env); force,contact=contact_state(self._env)
    self.rear_contact_age[eligible]=torch.where(contact[eligible,2:],0.,self.rear_contact_age[eligible]+self._env.step_dt)
    bad=(errors['orientation']>torch.pi/3.)|(errors['height'].abs()>.20)
    self.failure_time[eligible]=torch.where(bad[eligible],self.failure_time[eligible]+self._env.step_dt,0.)
    data=self.robot.data; q=data.joint_pos[:,self.joint_ids]
    limits=data.soft_joint_pos_limits[:,self.joint_ids]
    limit_bad=((q<limits[...,0]-.25)|(q>limits[...,1]+.25)).any(-1)
    finite=torch.isfinite(torch.cat((data.root_link_pos_w,data.root_link_quat_w,
      data.root_link_lin_vel_w,data.root_link_ang_vel_w,q,data.joint_vel[:,self.joint_ids]),-1)).all(-1)
    body=self._env.scene['base_ground_contact'].data
    impact=body.force.reshape(self.num_envs,-1,3).norm(dim=-1).amax(-1)>15.
    fail=(self.failure_time>=self.cfg.failure_persistence-1e-6)|limit_bad|~finite|impact
    self.failure_mask[eligible]=fail[eligible]
    endpoint=self.reference_time>=self.stand_start
    good=(endpoint&(errors['orientation']<torch.pi/12)&(errors['height'].abs()<.05)&
      (errors['fr'].norm(dim=-1)<.05)&~contact[:,:2].any(-1)&contact[:,2:].any(-1)&
      (self.rear_contact_age<=.15).all(-1)&
      (data.site_pos_w[:,self.site_ids[:2],2].amin(-1)>self._origins()[:,2]+.06)&
      (data.root_link_lin_vel_w[:,:2].norm(dim=-1)<.2)&(data.root_link_ang_vel_w.norm(dim=-1)<.5)&~fail)
    self.hold_time[eligible]=torch.where(good[eligible],self.hold_time[eligible]+self._env.step_dt,0.)
    self.held_success[eligible]|=self.hold_time[eligible]>=self.cfg.hold_seconds-1e-6
    self._outcome_step[eligible]=self._env.common_step_counter
    for name,value in (('held_success',self.held_success.float()),('orientation_error',errors['orientation']),
      ('height_error',errors['height'].abs()),('fr_error',errors['fr'].norm(dim=-1))):
      self.metrics[name][eligible]=torch.nan_to_num(value[eligible],nan=10.,posinf=10.,neginf=10.)

  def compute(self,dt): self._ensure_reference()
  def _update_command(self): self._ensure_reference()
  def _update_metrics(self): self.update_outcomes()

  def _debug_vis_impl(self,visualizer):
    for index in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(self.foot_target_pos_w[index],.025,(.2,.9,.3,1.),label=f'FR_transition_{index}')
