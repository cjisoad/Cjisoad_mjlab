"""Teacher anchor: body-left cross gravity-up defines horizontal forward.

The frame matches body +X at level quadruped stance and body -Z at a full
rear-leg stand. Pure roll at level stance does not turn its heading. It is
stateless so matched-state teacher queries do not depend on frame history.
When body +Y is parallel to up, the same deterministic horizontal reference
fallback used by the original frames keeps terminal observations finite.
"""

import torch
from mjlab.utils.lab_api.math import quat_apply

ANCHOR_CONTRACT = 'teacher_body_y_cross_up_v1'
ANCHOR_EPS = 1e-6


def _forward_and_up(quat_w, gravity_w):
  up = -torch.nn.functional.normalize(gravity_w, dim=-1)
  up = up.expand(quat_w.shape[0], 3)
  body_y = quat_apply(quat_w, up.new_tensor((0., 1., 0.)).expand_as(up))
  return torch.cross(body_y, up, dim=-1), up


def anchor_degenerate(quat_w, gravity_w):
  forward, _ = _forward_and_up(quat_w, gravity_w)
  return forward.norm(dim=-1) <= ANCHOR_EPS


def anchor_basis(quat_w, gravity_w):
  forward, up = _forward_and_up(quat_w, gravity_w)
  reference = up.new_tensor((1., 0., 0.)).expand_as(up)
  reference = reference - (reference * up).sum(-1, keepdim=True) * up
  alternate = up.new_tensor((0., 1., 0.)).expand_as(up)
  alternate = alternate - (alternate * up).sum(-1, keepdim=True) * up
  reference = torch.where(reference.norm(dim=-1, keepdim=True) > ANCHOR_EPS,
                          reference, alternate)
  forward = torch.where(forward.norm(dim=-1, keepdim=True) > ANCHOR_EPS,
                        forward, reference)
  forward = torch.nn.functional.normalize(forward, dim=-1)
  return torch.stack((forward, torch.cross(up, forward, dim=-1), up), dim=-1)


def to_anchor(basis, vector_w):
  """Rotate an absolute world vector into anchor axes at the same point."""
  return (basis.transpose(-1, -2) @ vector_w.unsqueeze(-1)).squeeze(-1)


def anchor_degenerate_fraction(env):
  """Per-step fallback indicator, averaged by the environment metric manager."""
  data = env.scene['robot'].data
  return anchor_degenerate(data.root_link_quat_w, data.gravity_vec_w).float()
