"""BIPED → DOWN → TRIPOD in one world, with measured handoff continuity."""
from dataclasses import asdict, dataclass, fields
import torch
from mjlab.managers import TerminationTermCfg
from src.tasks.transition_t.keyboard import (
  KeyboardTransitionActionCfg, AlignedKeyboardTeacher, TRIPOD, TRANSITION, BIPED)
from src.tasks.transition_t import terminations
from src.tasks.teacher_common.commands import LocoPedipulationTeacherCommand, LocoPedipulationTeacherCommandCfg
from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
from src.tasks.loco_pedipulation.mdp.commands import HOLD
from .commands import DownMotion
from .entry_course import DownEntryCommand, DownEntryCommandCfg, down_entry_env_cfg


def execution_snapshot(env):
  action=env.action_manager.get_term('joint_pos')
  result={name:getattr(env.sim.data,name).clone() for name in ('qpos','qvel')}
  for name in ('action','prev_action','prev_prev_action'):
    result['manager_'+name]=getattr(env.action_manager,name).clone()
  for name in ('raw_action','previous_action','previous_joint_vel','_requested_action','_previous_requested_action'):
    result[name]=getattr(action,name).clone()
  for name in ('position','velocity'):
    result['limiter_'+name]=getattr(action.target_limiter,name).clone()
  return result


def switch_owner(env,ids,state):
  before=execution_snapshot(env)
  motion=env.command_manager.get_term('motion');twist=env.command_manager.get_term('twist')
  if state==TRANSITION:
    motion.start_from_current_state(ids);motion.active[ids]=True
  elif state==TRIPOD:
    motion.active[ids]=False
    twist._command[ids]=twist._command.new_tensor([0.,0.,0.,.12,0.,.33])
  else:raise ValueError('Only the two approved forward ownership switches are supported')
  twist.owner[ids]=state
  after=execution_snapshot(env)
  differences={name:float((value-after[name]).abs().max()) for name,value in before.items()}
  if any(value!=0. for value in differences.values()):
    raise RuntimeError(f'Handoff changed execution state: {differences}')
  return differences


@dataclass(kw_only=True)
class ChainMotionCfg(DownEntryCommandCfg):
  def build(self,env):return ChainMotion(self,env)


class ChainMotion(DownEntryCommand):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.active=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)

  def _resample_command(self,ids):
    super()._resample_command(ids)
    self.active[ids]=False

  def _update_command(self):
    self._last_update_step[~self.active]=self._env.common_step_counter
    super()._update_command()

  start_from_current_state=DownMotion.start_from_current_state


@dataclass(kw_only=True)
class ChainTwistCfg(LocoPedipulationTeacherCommandCfg):
  def build(self,env):return ChainTwist(self,env)


class ChainTwist(LocoPedipulationTeacherCommand):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.owner=torch.full((self.num_envs,),BIPED,dtype=torch.long,device=self.device)

  def _resample_command(self,ids):
    super()._resample_command(ids)
    self.owner[ids]=BIPED
    self.manual[ids]=True
    self.enabled[ids]=False

  def _update_command(self):
    preserved=self._command.clone()
    super()._update_command()
    self._command.copy_(preserved)
    self.phase[:]=HOLD;self.blend[:]=1.


def chain_failure(env,kind):
  if kind=='physical_failure':return terminations.physical_failure(env)
  functions=dict(anchor_pos=terminations.tracking_height_failure,
    anchor_ori=terminations.tracking_orientation_failure,ee_body_pos=terminations.tracking_foot_height_failure)
  return functions[kind](env) & (env.command_manager.get_term('twist').owner==TRANSITION)


def chain_env_cfg(bank_file,*,motion_file,num_envs):
  cfg=down_entry_env_cfg(bank_file,motion_file=motion_file,num_envs=num_envs,play=True,split='holdout')
  old=cfg.commands['motion']
  cfg.commands['motion']=ChainMotionCfg(**{f.name:getattr(old,f.name) for f in fields(old)})
  cfg.commands['motion'].entry_probability=1.;cfg.commands['motion'].nominal_probability=0.
  source=loco_pedipulation_teacher_env_cfg(play=True).commands['twist']
  cfg.commands['twist']=ChainTwistCfg(**asdict(source))
  cfg.commands['twist'].resampling_time_range=(1.e9,1.e9)
  cfg.actions['joint_pos']=KeyboardTransitionActionCfg(**asdict(cfg.actions['joint_pos']))
  cfg.terminations={name:TerminationTermCfg(func=chain_failure,params={'kind':name})
    for name in ('physical_failure','anchor_pos','anchor_ori','ee_body_pos')}
  return cfg
