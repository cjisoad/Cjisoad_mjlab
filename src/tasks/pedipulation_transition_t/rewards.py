"""Fresh transition rewards; all shaping and failure checks are per world."""
import torch
from src.tasks.teacher_common.anchor_frame import to_anchor


def state_errors(env):
  term=env.command_manager.get_term('twist'); term._ensure_reference()
  data=env.scene['robot'].data
  dot=(data.root_link_quat_w*term.ref_root_state[:,3:7]).sum(-1).abs().clamp(0.,1.)
  return dict(orientation=2.*torch.acos(dot),
    height=data.root_link_pos_w[:,2]-term.ref_root_state[:,2],
    fr=term.fr_error,
    support=data.joint_pos[:,term.joint_ids][:,term.support_ids]-term.ref_joint_pos[:,term.support_ids],
    linear=to_anchor(term.basis_w,data.root_link_lin_vel_w)-term.ref_linear_velocity,
    angular=to_anchor(term.basis_w,data.root_link_ang_vel_w)-term.ref_angular_velocity,
    fl_height=data.site_pos_w[:,term.site_ids[0],2]-term.ref_foot_pos_w[:,0,2])


def contact_state(env):
  data=env.scene['feet_ground_contact'].data
  force=data.force.reshape(env.num_envs,4,-1,3).sum(2)
  found=data.found.reshape(env.num_envs,4,-1).any(-1)
  return force,found


def tracking_reward(env,name):
  errors=state_errors(env)
  key,scale={'orientation':('orientation',torch.pi/15.),'height':('height',.05),
    'fr_position':('fr',.04),'support_pose':('support',.30),
    'linear_velocity':('linear',errors['linear'].new_tensor((.30,.30,.20))),
    'angular_velocity':('angular',.8),'fl_height':('fl_height',.05)}[name]
  value=errors[key]/scale
  squared=value.square() if value.ndim==1 else value.square().mean(-1)
  return torch.exp(-torch.nan_to_num(squared,nan=1e6,posinf=1e6,neginf=1e6))


def penalty(env,name):
  term=env.command_manager.get_term('twist'); term._ensure_reference()
  data=env.scene['robot'].data; action=env.action_manager.get_term('joint_pos')
  force,contact=contact_state(env); magnitude=force.norm(dim=-1)
  if name=='support_loss':
    readiness=(magnitude/25.).clamp(0.,1.)
    rear=(1.-readiness[:,2:].amax(-1))
    fl=(1.-readiness[:,0])*(1.-term.ref_fl_unload)
    result=rear+fl
  elif name=='front_contact':
    weights=torch.stack((term.ref_fl_unload,torch.ones_like(term.ref_fl_unload)),-1)
    result=(contact[:,:2].float()*weights).sum(-1)
  elif name=='slip':
    speed=data.site_lin_vel_w[:,term.site_ids,:2].square().sum(-1)
    result=(speed*contact.float()).sum(-1)
  elif name=='excessive_force': result=((magnitude-160.).clamp_min(0.)/160.).square().sum(-1)
  elif name=='collision':
    result=data.root_link_pos_w.new_zeros(env.num_envs)
    for sensor in ('nonfoot_ground_contact','self_contact'):
      found=env.scene[sensor].data.found.reshape(env.num_envs,-1)
      result+=found.any(-1).float()
  elif name=='action_change':
    result=((action.raw_action-action.previous_action)*action.cfg.scale/.05).square().mean(-1)
  elif name=='torque':
    limits=force.new_tensor([35.55 if i%3==2 else 23.7 for i in range(12)])
    result=(data.actuator_force/limits).square().mean(-1)
  elif name=='joint_limits':
    q=data.joint_pos[:,term.joint_ids]; limits=data.soft_joint_pos_limits[:,term.joint_ids]
    result=((limits[...,0]-q).clamp_min(0.)+(q-limits[...,1]).clamp_min(0.)).sum(-1)
  else: raise ValueError(f'Unknown transition penalty {name}')
  return torch.nan_to_num(result,nan=100.,posinf=100.,neginf=100.).clamp_max(100.)


def failure_termination(env):
  term=env.command_manager.get_term('twist'); term.update_outcomes()
  return term.failure_mask


def failure_cost(env):
  """An event cost of -5 after the environment's reward-dt multiplication."""
  return failure_termination(env).float()*(-5./env.step_dt)


def time_out(env):
  return env.episode_length_buf>=env.max_episode_length


def transition_metric(env,name):
  term=env.command_manager.get_term('twist'); term.update_outcomes()
  return term.metrics[name]
