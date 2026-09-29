"""Mode-aware orientation termination.

In LOCO mode the quadruped/tripod tilt limit applies, matching
loco_pedipulation. In BIPED mode the robot is supposed to pitch to a full
rear-leg stand, so only the lateral lean (sideways fall) is limited; the latch
uses the hysteresis mode flag, never the continuous group weight.
"""

import math

import torch


def _term(env):
  return env.command_manager.get_term('twist')


def fell_over(env, limit_angle=math.radians(60.)):
  gravity = env.scene['robot'].data.projected_gravity_b
  tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.)).abs()
  lateral = gravity[:, 1].abs() > math.sin(limit_angle)
  return torch.where(_term(env).mode, lateral, tilt > limit_angle)
