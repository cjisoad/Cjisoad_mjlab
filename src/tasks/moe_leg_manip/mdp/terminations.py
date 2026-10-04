"""Mode-aware orientation termination.

In LOCO mode the quadruped/tripod tilt limit applies, matching
loco_pedipulation. In BIPED mode the robot is supposed to pitch to a full
rear-leg stand, so only the lateral lean (sideways fall) is limited; the latch
uses the hysteresis mode flag. A bounded descent allowance handles the physical
delay after BIPED -> LOCO, independently of the continuous reward-group weight.
"""

import math

import torch


def _term(env):
  return env.command_manager.get_term('twist')


def fell_over(env, limit_angle=math.radians(60.), exit_limit_angle=math.radians(110.)):
  gravity = env.scene['robot'].data.projected_gravity_b
  tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.)).abs()
  lateral = gravity[:, 1].abs() > math.sin(limit_angle)
  term = _term(env)
  loco_limit = torch.where(term.biped_exiting, lateral | (tilt > exit_limit_angle),
                          tilt > limit_angle)
  return torch.where(term.mode, lateral, loco_limit)
