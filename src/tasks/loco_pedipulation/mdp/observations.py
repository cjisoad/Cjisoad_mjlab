"""Locomotion observations and explicit manipulation transition state."""

import torch

from .anchor_frame import to_anchor
from .commands import QUAD


def foot_control_state(env):
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
  moving = term.command.norm(dim=-1) > .05
  return torch.stack((phase.sin(), phase.cos()), dim=-1) * moving[:, None]


def base_velocity(env):
  term = env.command_manager.get_term('twist')
  return to_anchor(term.basis_w, env.scene['robot'].data.root_link_lin_vel_w)
