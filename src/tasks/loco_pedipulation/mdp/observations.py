"""Actor ABI and privileged locomotion/manipulation observations."""

import torch

from .anchor_frame import to_anchor
from .commands import QUAD


def policy_state(env, add_noise=True):
  """Pedipulation-compatible 48-value actor input."""
  robot = env.scene['robot'].data
  action = env.action_manager.get_term('joint_pos')
  command = env.command_manager.get_command('twist')
  obs = torch.cat((robot.root_link_ang_vel_b * .25, robot.projected_gravity_b,
    command * command.new_tensor((2., 2., .25, 1., 1., 1.)),
    robot.joint_pos - robot.default_joint_pos, robot.joint_vel * .05,
    action.raw_action), dim=-1)
  if add_noise:
    amplitude = obs.new_tensor((.05,) * 6 + (0.,) * 6 + (.01,) * 12 + (.075,) * 12 + (0.,) * 12)
    obs = obs + (torch.rand_like(obs) * 2 - 1) * amplitude
  action.actor_sample = obs.clamp(-100., 100.)
  return action.actor_sample


def shared_policy_state(env):
  """Return the exact actor noise draw for asymmetric critic training."""
  return env.action_manager.get_term('joint_pos').actor_sample


def foot_control_state(env):
  """Privileged internal FR state, intentionally excluded from actor input."""
  term = env.command_manager.get_term('twist')
  term.initialize_reference()
  progress = (term.elapsed / term.cfg.reach_time).clamp(0., 1.)
  active = (term.phase != QUAD).float()[:, None]
  return torch.cat((
    term.enabled.float()[:, None], term.blend[:, None],
    torch.nn.functional.one_hot(term.phase, 6).float(), progress[:, None],
    term.requested_offset * term.enabled[:, None],
    (term.reference - term.zero) * active,
    (term.reference - term.foot_pos_anchor) * active,
  ), dim=-1)


def gait_phase(env):
  term = env.command_manager.get_term('twist')
  phase = term.gait_phase * (2. * torch.pi)
  moving = term.command[:, :3].norm(dim=-1) > .05
  return torch.stack((phase.sin(), phase.cos()), dim=-1) * moving[:, None]


def base_velocity(env):
  term = env.command_manager.get_term('twist')
  return to_anchor(term.basis_w, env.scene['robot'].data.root_link_lin_vel_w)
