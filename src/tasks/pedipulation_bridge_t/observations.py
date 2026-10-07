"""Actor84 and its shared noisy sample plus 88 exact critic channels."""
import torch

from src.tasks.leg_manip.mdp import observations as teacher_observations
from src.tasks.teacher_common.anchor_frame import to_anchor


def _finite(values):
  return torch.nan_to_num(values, nan=0., posinf=100., neginf=-100.).clamp(-100., 100.)


def policy_state(env, add_noise=True):
  term = env.command_manager.get_term('twist')
  data = env.scene['robot'].data
  action = env.action_manager.get_term('joint_pos')
  ref = term.reference_state
  noise_enabled = add_noise and term.course_stage >= 2
  fraction=getattr(getattr(term,'course',None),'randomization_fraction',1.)
  # Query the public teacher ABI without changing the frozen teacher module.
  original = teacher_observations.policy_state(env, add_noise=False)
  if noise_enabled:
    noisy = teacher_observations.policy_state(env, add_noise=True)
    original = original+(noisy-original)*term.disturbed[:,None]*fraction
  origins = getattr(env.scene, 'env_origins', data.root_link_pos_w.new_zeros(env.num_envs, 3))
  height = data.root_link_pos_w[:, 2]-origins[:, 2]
  fr_error = term.fr_error
  if noise_enabled:
    height = height + (torch.rand_like(height)*2.-1.)*.005*term.disturbed*fraction
    fr_error = fr_error + (torch.rand_like(fr_error)*2.-1.)*.005*term.disturbed[:, None]*fraction
  joints = ref['joint_pos'][:, term.support_ids]-action.default_angles[:, term.support_ids]
  guidance = torch.cat((height[:, None], fr_error, term.progress[:, None],
    term.orientation_error_b, ref['height'][:, None], joints,
    ref['joint_vel'][:, term.support_ids]*.05,
    ref['linear_velocity']*2., ref['angular_velocity']*.25), -1)
  action.actor_sample = _finite(torch.cat((original, guidance), -1))
  return action.actor_sample


def shared_policy_state(env):
  """The critic receives the exact 84-channel actor noise draw."""
  return env.action_manager.get_term('joint_pos').actor_sample


def privileged_state(env):
  """Exact physical and reference state in the public anchor coordinates."""
  term = env.command_manager.get_term('twist')
  data = env.scene['robot'].data
  action = env.action_manager.get_term('joint_pos')
  ref = term.reference_state
  basis = term.basis_w
  feet = data.site_pos_w[:, term.site_ids]-data.root_link_pos_w[:, None, :]
  foot_basis = basis[:, None].expand(-1, 4, -1, -1)
  feet = to_anchor(foot_basis, feet)
  foot_velocities = to_anchor(foot_basis, data.site_lin_vel_w[:, term.site_ids])
  force = env.scene['feet_ground_contact'].data.force.reshape(env.num_envs, 4, -1, 3).sum(2)
  force = force[:, getattr(term, 'contact_order', range(4))]
  body_force = env.scene['nonfoot_ground_touch'].data.force.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(-1, keepdim=True)
  values = torch.cat((to_anchor(basis, data.root_link_lin_vel_w),
    to_anchor(basis, data.root_link_ang_vel_w), feet.flatten(1), foot_velocities.flatten(1),
    force.flatten(1)/100., term.contacts.float(), body_force/100., action.dr_observation,
    ref['contact'], term.reference_speed[:, None], term.full_start.float()[:, None],
    term.disturbed.float()[:, None]), -1)
  return _finite(values)
