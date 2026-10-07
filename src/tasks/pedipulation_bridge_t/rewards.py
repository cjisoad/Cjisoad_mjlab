"""Transition shaping only; delayed expert-tail events belong to the collector."""
import torch

from src.tasks.teacher_common.anchor_frame import to_anchor
from .commands import BRIDGE


def _term(env):
  return env.command_manager.get_term('twist')


def _owned(env,values):
  return torch.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)*(_term(env).action_owner==BRIDGE)


def shaping(env,name):
  term=_term(env); data=env.scene['robot'].data
  if name=='fr_tracking':
    ramp=((term.progress-.7)/.3).clamp(0.,1.)
    early,late=term.cfg.fr_tracking_scale
    scale=early+(late-early)*ramp
    result=torch.exp(-term.fr_error.square().sum(-1)/scale.square())
  elif name=='height':
    origins=getattr(env.scene,'env_origins',data.root_link_pos_w.new_zeros(env.num_envs,3))
    height=(data.root_link_pos_w[:,2]-origins[:,2]-term.reference_height)/.08
    result=torch.exp(-height.square())
  elif name=='orientation':
    error=(data.projected_gravity_b-term.reference_gravity)/.35
    result=torch.exp(-error.square().sum(-1))
  elif name=='support_pose':
    error=(data.joint_pos[:,term.joint_ids][:,term.support_ids]-
      term.reference_joint_pos[:,term.support_ids])/.5
    result=torch.exp(-error.square().mean(-1))
  elif name=='fl_position':
    position=to_anchor(term.basis_w,data.site_pos_w[:,term.site_ids[0]]-data.root_link_pos_w)
    error=(position-term.reference_fl_position)/.10
    result=torch.exp(-error.square().mean(-1))
  elif name=='support':
    ref=term.reference_state
    rear_weights=ref['contact'][:,2:]
    recent=((term.rear_contact_age<term.cfg.recent_rear_seconds).float()*rear_weights).sum(-1)
    fl_weight=ref['contact'][:,0]*(1.-ref['fl_unload'])
    result=(recent+term.contacts[:,0].float()*fl_weight)/(rear_weights.sum(-1)+fl_weight).clamp_min(1.)
  elif name=='rise_progress':
    result=term.rise_progress_delta
  else:
    raise ValueError(f'unknown bridge shaping {name!r}')
  return _owned(env,result)


def penalty(env,name):
  term=_term(env); data=env.scene['robot'].data; action=env.action_manager.get_term('joint_pos')
  if name=='action_change':
    result=(action.raw_action-action.previous_action).square().mean(-1)
  elif name=='slip':
    speed=data.site_lin_vel_w[:,term.site_ids,:2].square().sum(-1)
    result=(speed*term.contacts).sum(-1)
  elif name=='front_contact':
    result=term.contacts[:,0].float()*term.reference_fl_unload+term.contacts[:,1].float()
  elif name=='joint_limits':
    q=data.joint_pos[:,term.joint_ids]
    limits=data.soft_joint_pos_limits[:,term.joint_ids]
    result=((limits[...,0]-q).clamp_min(0.)+(q-limits[...,1]).clamp_min(0.)).sum(-1)
  elif name=='torque':
    limits=data.actuator_force.new_tensor([35.55 if i%3==2 else 23.7 for i in range(12)])
    result=(data.actuator_force/limits).square().mean(-1)
  elif name=='collision':
    force=env.scene['nonfoot_contact'].data.force
    result=(force.reshape(env.num_envs,-1,3).norm(dim=-1)>1.).float().sum(-1)
  else:
    raise ValueError(f'unknown bridge penalty {name!r}')
  return _owned(env,result).clamp_max(100.)


def bridge_termination(env):
  term=_term(env); term.advance()
  return term.outcome_event!=0


def bridge_metric(env,name):
  term=_term(env); term.advance()
  return term.metrics[name]
