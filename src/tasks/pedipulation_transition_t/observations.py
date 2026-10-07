"""Actor84 and a critic sharing its noise draw plus clean physical state."""
import torch
from src.tasks.teacher_common.anchor_frame import to_anchor
from .rewards import contact_state


def policy_state(env,add_noise=True):
  term=env.command_manager.get_term('twist'); term._ensure_reference()
  data=env.scene['robot'].data; action=env.action_manager.get_term('joint_pos')
  joint_ids=term.joint_ids; origins=term._origins()
  velocity=to_anchor(term.basis_w,data.root_link_lin_vel_w)
  original=torch.cat((data.root_link_ang_vel_b*.25,data.projected_gravity_b,
    term.command*data.joint_pos.new_tensor((2.,2.,.25,1.,1.,1.)),
    data.joint_pos[:,joint_ids]-data.default_joint_pos[:,joint_ids],
    data.joint_vel[:,joint_ids]*.05,action.raw_action,velocity*2.),-1)
  extra=torch.cat(((data.root_link_pos_w[:,2]-origins[:,2])[:,None],term.fr_error,
    term.progress[:,None],term.orientation_error_b,(term.ref_root_state[:,2]-origins[:,2])[:,None],
    term.ref_joint_pos[:,term.support_ids]-data.default_joint_pos[:,joint_ids][:,term.support_ids],
    term.ref_joint_vel[:,term.support_ids]*.05,
    term.ref_linear_velocity*2.,term.ref_angular_velocity*.25),-1)
  obs=torch.cat((original,extra),-1)
  if add_noise and term.stage>=5 and not term.cfg.play:
    amplitude=obs.new_tensor((.05,)*6+(0.,)*6+(.01,)*12+(.075,)*12+(0.,)*12+(.04,)*3+
      (.005,)+(.005,)*3+(0.,)*29)
    noise=(torch.rand_like(obs)*2.-1.)*amplitude
    obs+=noise*term.disturbed[:,None]
  action.actor_sample=torch.nan_to_num(obs,nan=0.,posinf=100.,neginf=-100.).clamp(-100.,100.)
  return action.actor_sample


def shared_policy_state(env):
  return env.action_manager.get_term('joint_pos').actor_sample


def privileged_state(env):
  term=env.command_manager.get_term('twist'); term._ensure_reference()
  data=env.scene['robot'].data; action=env.action_manager.get_term('joint_pos')
  feet=data.site_pos_w[:,term.site_ids]-data.root_link_pos_w[:,None,:]
  basis=term.basis_w[:,None,:,:].expand(-1,4,-1,-1)
  feet=(basis.transpose(-1,-2)@feet[...,None]).squeeze(-1)
  foot_velocities=(basis.transpose(-1,-2)@data.site_lin_vel_w[:,term.site_ids,...,None]).squeeze(-1)
  force,contact=contact_state(env)
  body_force=env.scene['nonfoot_ground_contact'].data.force.reshape(env.num_envs,-1,3).norm(dim=-1).amax(-1,keepdim=True)
  physics=action.dr_observation if hasattr(action,'dr_observation') else feet.new_zeros(env.num_envs,34)
  values=torch.cat((to_anchor(term.basis_w,data.root_link_lin_vel_w),
    to_anchor(term.basis_w,data.root_link_ang_vel_w),feet.flatten(1),foot_velocities.flatten(1),force.flatten(1)/100.,
    contact.float(),body_force/100.,physics,term.ref_contact,
    term.speed[:,None],term.full_start.float()[:,None],term.disturbed.float()[:,None]),-1)
  return torch.nan_to_num(values,nan=0.,posinf=100.,neginf=-100.).clamp(-100.,100.)
