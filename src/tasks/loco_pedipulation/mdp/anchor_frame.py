"""Gravity-aligned anchor with horizontal body +X as forward."""

import torch

from mjlab.utils.lab_api.math import quat_apply


def anchor_basis(quat_w, gravity_w):
  up = -torch.nn.functional.normalize(gravity_w, dim=-1)
  up = up.expand(quat_w.shape[0], 3)
  body_x = quat_apply(quat_w, up.new_tensor((1., 0., 0.)).expand_as(up))
  forward = body_x - (body_x * up).sum(-1, keepdim=True) * up
  # Falls terminate before this singularity; keep terminal observations finite.
  reference = up.new_tensor((1., 0., 0.)).expand_as(up)
  reference = reference - (reference * up).sum(-1, keepdim=True) * up
  alternate = up.new_tensor((0., 1., 0.)).expand_as(up)
  alternate = alternate - (alternate * up).sum(-1, keepdim=True) * up
  reference = torch.where(reference.norm(dim=-1, keepdim=True) > 1e-6, reference, alternate)
  forward = torch.where(forward.norm(dim=-1, keepdim=True) > 1e-6, forward, reference)
  forward = torch.nn.functional.normalize(forward, dim=-1)
  return torch.stack((forward, torch.cross(up, forward, dim=-1), up), dim=-1)


def to_anchor(basis, vector_w):
  return (basis.transpose(-1, -2) @ vector_w.unsqueeze(-1)).squeeze(-1)
