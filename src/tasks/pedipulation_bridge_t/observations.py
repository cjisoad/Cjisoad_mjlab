"""Bridge guidance channels; frozen experts keep their original actor51 ABI."""
import torch
from src.tasks.leg_manip.mdp import observations as teacher_observations


def policy_state(env, add_noise=True):
  original=teacher_observations.policy_state(env,add_noise=add_noise)
  term=env.command_manager.get_term('twist')
  data=env.scene['robot'].data
  action=env.action_manager.get_term('joint_pos')
  ref=term.reference_state
  origins=getattr(env.scene,'env_origins',data.root_link_pos_w.new_zeros(env.num_envs,3))
  height=data.root_link_pos_w[:,2]-origins[:,2]
  if add_noise: height=height+(torch.rand_like(height)*2.-1.)*.01
  joints=ref['joint_pos'][:,term.support_ids]-action.default_angles[:,term.support_ids]
  guidance=torch.cat((term.progress[:,None],height[:,None],ref['height'][:,None],
    ref['gravity'],joints,ref['fl_position'],ref['fl_unload'][:,None],term.fr_error),-1)
  action.actor_sample=torch.cat((original,guidance),-1).clamp(-100.,100.)
  return action.actor_sample


def shared_policy_state(env):
  """Share actor noise; retain the source critic's separate exact velocity."""
  sample=env.action_manager.get_term('joint_pos').actor_sample
  return torch.cat((sample[:,:48],sample[:,51:]),-1)
