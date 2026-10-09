"""Manual tripod → reference rise → biped control in one physical world."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch

from src.tasks.leg_manip.constants import (
  TRIPOD_HIGH, TRIPOD_X_RANGE, TRIPOD_Y_RANGE, DZ_MIN,
  BIPED_LOW, BIPED_HIGH, BIPED_X_RANGE, BIPED_Y_RANGE,
)
from src.tasks.loco_pedipulation.mdp.commands import LocoPedipulationCommand, QUAD, HOLD, smoothstep
from src.tasks.loco_pedipulation.mdp.observations import gait_phase
from src.tasks.pedipulation.mdp.actions import PedipulationPositionAction
from .actions import BridgePositionAction, BridgePositionActionCfg
from .commands import BridgeCommand, BridgeCommandCfg, WARM, BRIDGE, VERIFY, DONE, PHYSICAL_FAILURE
from .experts import FrozenTeacherActor, validate_teacher_contract, teacher_action_to_common

STATE_NAMES = {WARM:'TRIPOD', BRIDGE:'TRANSITION', VERIFY:'BIPED', DONE:'RESETTING'}


class KeyboardTeacher:
  """An explicitly validated teacher, independent of the bridge training pair."""
  def __init__(self, checkpoint, task, device):
    saved=torch.load(Path(checkpoint),map_location='cpu',weights_only=False)
    self.contract=(saved.get('infos') or {}).get('teacher_contract')
    self.input_dim=self.contract.get('actor_dim') if isinstance(self.contract,dict) else None
    validate_teacher_contract(self.contract,task,actor_dim=self.input_dim)
    self.actor=FrozenTeacherActor(saved.get('actor_state_dict'),input_dim=self.input_dim).to(device)
    self.name='quadruped' if task=='loco_pedipulation_t' else 'biped'

  @torch.inference_mode()
  def actions(self, action, term, ids):
    pair=action._ensure_experts()
    obs=pair.observations(self.name,term.command,env_ids=ids)
    if self.input_dim==54:
      obs=torch.cat((obs,gait_phase(action._env)[ids],term.blend[ids,None]),-1)
    zero=obs.new_tensor(self.contract['default_angles'])
    return teacher_action_to_common(self.actor(obs),zero,action.default_angles[ids],scale=action.cfg.scale)


@dataclass(kw_only=True)
class KeyboardBridgeCommandCfg(BridgeCommandCfg):
  def build(self, env):
    return KeyboardBridgeCommand(self,env)


class KeyboardBridgeCommand(BridgeCommand):
  """Use source manipulation phases but own the three-controller schedule."""
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.forced_live_probability=1.
    self.forced_disturbed=False
    self.forced_bank=0
    self.course.enabled=False
    self.trigger_height=.33
    self.trigger_seconds=.10
    self.trigger_elapsed=torch.zeros(self.num_envs,device=self.device)
    self.keyboard_command=torch.zeros(self.num_envs,6,device=self.device)
    self.keyboard_active=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    self.keyboard_input_enabled=torch.zeros_like(self.keyboard_active)
    self.keyboard_timed_out=torch.zeros_like(self.keyboard_active)
    self.keyboard_revision=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
    self.keyboard_events=[]

  def configure_trigger(self,margin):
    if not math.isfinite(margin) or not 0.<margin<TRIPOD_HIGH-BIPED_LOW:
      raise ValueError('Transition margin must be between0 and0.10m')
    self.trigger_height=TRIPOD_HIGH-margin

  def _resample_command(self,env_ids):
    super()._resample_command(env_ids)
    self.owner[env_ids]=WARM
    self.requested_velocity[env_ids]=0.
    self.requested_offset[env_ids]=0.
    self._command[env_ids]=0.
    self.enabled[env_ids]=False
    self.phase[env_ids]=QUAD
    if hasattr(self,'keyboard_revision'):
      self.trigger_elapsed[env_ids]=0.
      self.keyboard_command[env_ids]=0.
      self.keyboard_active[env_ids]=False
      self.keyboard_input_enabled[env_ids]=False
      self.keyboard_timed_out[env_ids]=False
      self.keyboard_revision[env_ids]+=1

  def submit(self,env_ids,commands,target_active,*,input_enabled=True):
    """Receive manual input between steps; transition targets remain owned here."""
    ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long).reshape(-1)
    values=torch.as_tensor(commands,device=self.device,dtype=self.command.dtype).reshape(-1,6)
    active=torch.as_tensor(target_active,device=self.device,dtype=torch.bool).reshape(-1)
    if len(values)!=len(ids) or len(active)!=len(ids) or not torch.isfinite(values).all():
      raise ValueError('Keyboard commands require one finite command and active flag per world')
    self.keyboard_input_enabled[ids]=input_enabled
    manual=(self.owner[ids]==WARM)|(self.owner[ids]==VERIFY)
    ids,values,active=ids[manual],values[manual],active[manual]
    if not len(ids): return
    self.keyboard_command[ids]=values
    self.keyboard_active[ids]=active
    for state,x,y,z in ((WARM,TRIPOD_X_RANGE,TRIPOD_Y_RANGE,(DZ_MIN,TRIPOD_HIGH)),
                        (VERIFY,BIPED_X_RANGE,BIPED_Y_RANGE,(BIPED_LOW,BIPED_HIGH))):
      chosen=self.owner[ids]==state
      selected=ids[chosen]
      if not len(selected): continue
      target=values[chosen,3:]
      bounds=target.new_tensor((x,y,z))
      target=torch.maximum(bounds[:,0],torch.minimum(bounds[:,1],target))
      if state==WARM:
        target=torch.where(active[chosen,None],target,torch.zeros_like(target))
        LocoPedipulationCommand.set_command(self,selected,velocity=values[chosen,:3],target_offset=target)
      else:
        self._command[selected,:3]=values[chosen,:3]
        self._command[selected,3:]=target
        self.requested_velocity[selected]=values[chosen,:3]
        self.requested_offset[selected]=target

  def _event(self,ids,old,new,reason):
    for world in ids.tolist():
      self.keyboard_events.append(dict(world=world,step=self._env.common_step_counter,
        simulation_seconds=self._env.common_step_counter*self._env.step_dt,
        previous=old,state=new,reason=reason,command=self.command[world].cpu().tolist(),
        fr_error=float(self.fr_error[world].norm()),
        linear_speed=float(self.robot.data.root_link_lin_vel_w[world].norm()),
        angular_speed=float(self.robot.data.root_link_ang_vel_w[world].norm())))

  def handoff_conditions(self):
    """Keyboard handoff requires stable standing; FR tracking stays diagnostic."""
    return {name:value for name,value in self.biped_conditions().items()
            if name!='fr_error'}

  def _biped_candidate(self):
    return torch.stack(tuple(self.handoff_conditions().values()),dim=-1).all(-1)

  def advance(self):
    step=self._env.common_step_counter
    if self._last_advance_step==step: return
    self._last_advance_step=step
    dt=self._env.step_dt
    source=self.action_owner==WARM
    transition=(self.action_owner==BRIDGE)&~self.keyboard_timed_out
    self.attempt_elapsed+=dt
    self.bridge_elapsed+=dt*transition
    self.rear_contact_age[:]=torch.where(self.contacts[:,2:],0.,self.rear_contact_age+dt)
    failed=self._physical_failure()
    ids=failed.nonzero().flatten()
    if len(ids):
      self._event(ids,'ACTIVE','RESETTING','physical_failure')
      self._finish(failed,-1,PHYSICAL_FAILURE)
    trigger=(source & ~failed & self.keyboard_active & self.keyboard_input_enabled &
      (self.keyboard_command[:,5]>=self.trigger_height-1e-6)&
      (self.command[:,5]>=self.trigger_height-1e-6)&(self.fr_error.norm(dim=-1)<.10))
    self.trigger_elapsed[:]=torch.where(trigger,self.trigger_elapsed+dt,0.)
    ids=(trigger&(self.trigger_elapsed>=self.trigger_seconds-1e-6)).nonzero().flatten()
    if len(ids):
      self.start_offset[ids]=self.command[ids,3:]
      self.start_time[ids]=0.
      self.reference_speed[ids]=1.
      self.bank_index[ids]=0
      self.start_bridge(ids)
      self.keyboard_command[ids,:3]=0.
      self.keyboard_command[ids,3:]=self.start_offset[ids]
      self.keyboard_revision[ids]+=1
      self._event(ids,'TRIPOD','TRANSITION','height_and_tracking_hold')
    ready=((self.owner==BRIDGE)&~failed&~self.keyboard_timed_out &
      (self.progress>=1.-1e-6)&self._biped_candidate())
    self.candidate_elapsed[:]=torch.where(ready,self.candidate_elapsed+dt,0.)
    self.held_success[:]=ready&(self.candidate_elapsed>=self.cfg.candidate_seconds-1e-6)
    ids=self.held_success.nonzero().flatten()
    if len(ids):
      self.owner[ids]=VERIFY
      self.keyboard_command[ids]=self.command[ids]
      self.keyboard_command[ids,:3]=0.
      self.keyboard_active[ids]=True
      self.requested_offset[ids]=self.command[ids,3:]
      self.requested_velocity[ids]=0.
      self.keyboard_revision[ids]+=1
      self.handoff_count[ids]+=1
      self._event(ids,'TRANSITION','BIPED','standing_hold_completed')
    expired=((self.owner==BRIDGE)&~self.keyboard_timed_out &
      (self.bridge_elapsed>=self.cfg.bridge_deadline-1e-5))
    ids=expired.nonzero().flatten()
    if len(ids):
      self.keyboard_timed_out[ids]=True
      self._event(ids,'TRANSITION','TIMED_OUT','standing_hold_not_completed')

  def _update_command(self):
    self.advance()
    # Retain the actual source PREPARE/REACH/HOLD/LOWER phases and gait signal.
    biped=self.owner==VERIFY
    biped_command=self.command[biped].clone()
    LocoPedipulationCommand._update_command(self)
    transition=self.owner==BRIDGE
    fraction=smoothstep(self.progress)
    target=torch.lerp(self.start_offset,self.end_offset,fraction[:,None])
    self._command[transition,:3]=0.
    self._command[transition,3:]=target[transition]
    self._command[biped]=biped_command
    active=transition|biped
    self.phase[active]=HOLD
    self.enabled[active]=True
    self.blend[active]=1.
    self.reference[active]=self.zero+self.command[active,3:]
    self.goal[active]=self.reference[active]
    self._command[self.owner==DONE]=0.

  def status(self,world=0):
    state='TIMED_OUT' if self.keyboard_timed_out[world] else STATE_NAMES[int(self.owner[world])]
    missing=[]
    if self.owner[world]==BRIDGE:
      if self.progress[world]<1.-1e-6: missing.append('reference')
      missing.extend(name for name,value in self.handoff_conditions().items() if not bool(value[world]))
    return dict(state=state,requested_dz=float(self.keyboard_command[world,5]),
      executed_dz=float(self.command[world,5]),trigger_height=self.trigger_height,
      trigger_hold=float(self.trigger_elapsed[world]),progress=float(self.progress[world]),
      standing_hold=float(self.candidate_elapsed[world]),
      fr_error=float(self.fr_error[world].norm()),missing=missing)


@dataclass(kw_only=True)
class KeyboardBridgeActionCfg(BridgePositionActionCfg):
  def build(self,env):
    return KeyboardBridgeAction(self,env)


class KeyboardBridgeAction(BridgePositionAction):
  source_teacher=None
  biped_teacher=None

  @torch.inference_mode()
  def process_actions(self,actions):
    term=self._env.command_manager.get_term('twist')
    term.begin_step()
    selected=actions.clamp(-10.,10.).clone()
    for teacher,state in ((self.source_teacher,WARM),(self.biped_teacher,VERIFY)):
      ids=(term.action_owner==state).nonzero().flatten()
      if len(ids):
        if teacher is None: raise RuntimeError('Keyboard teachers must be loaded before stepping')
        selected[ids]=teacher.actions(self,term,ids)
    # Teacher rebase happens after policy clipping, preserving physical q_des.
    PedipulationPositionAction.process_actions(self,selected)
    self.delay_steps.zero_()


def keyboard_bridge_env_cfg(num_envs=1):
  from .env_cfg import bridge_env_cfg
  cfg=bridge_env_cfg(play=True)
  cfg.scene.num_envs=num_envs
  cfg.commands['twist']=KeyboardBridgeCommandCfg(**asdict(cfg.commands['twist']))
  # Match the current keyboard source's simultaneous lowering/support recovery.
  cfg.commands['twist'].unified_return=True
  cfg.actions['joint_pos']=KeyboardBridgeActionCfg(**asdict(cfg.actions['joint_pos']))
  for group in cfg.observations.values():
    if group is not None and group.history_length==1: group.history_length=0
  return cfg
