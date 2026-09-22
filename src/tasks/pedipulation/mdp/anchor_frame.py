"""Batched gravity-aligned frame following the horizontal body -Z direction."""

import torch

from mjlab.utils.lab_api.math import quat_apply


class AnchorFrame:
  def __init__(self, gravity_w, acquire=.15, release=.05, transition_rate=6.):
    if not 0 < release < acquire < 1 or transition_rate <= 0:
      raise ValueError('Require 0 < release < acquire < 1 and positive transition rate')
    if not torch.isfinite(gravity_w).all() or (gravity_w.norm(dim=-1) < 1e-6).any():
      raise ValueError('Gravity must be finite and nonzero')
    self.up = -torch.nn.functional.normalize(gravity_w, dim=-1)
    self.acquire, self.release = acquire, release
    self.transition_rate = transition_rate
    world_x = self.up.new_tensor((1., 0., 0.)).expand_as(self.up)
    world_y = self.up.new_tensor((0., 1., 0.)).expand_as(self.up)
    reference = self._project(world_x)
    reference = torch.where(reference.norm(dim=-1, keepdim=True) > 1e-6,
                            reference, self._project(world_y))
    self.reference = torch.nn.functional.normalize(reference, dim=-1)
    self.forward = self.reference.clone()
    self.valid = torch.zeros(len(gravity_w), dtype=torch.bool, device=gravity_w.device)
    self.aligning = self.valid.clone()
    self._pending_reset = torch.ones_like(self.valid)
    self._step = None

  def _project(self, vector):
    return vector - (vector * self.up).sum(-1, keepdim=True) * self.up

  def reset(self, env_ids):
    # State written during reset is not forwarded yet. Read its quaternion later.
    self._pending_reset[env_ids] = True

  def update(self, quat_w, dt, step):
    if step != self._step:
      self._previous_forward = self.forward.clone()
      self._previous_valid = self.valid.clone()
      self._previous_aligning = self.aligning.clone()
      self._step = step

    body_x = quat_apply(quat_w, self.up.new_tensor((1., 0., 0.)).expand_as(self.up))
    projected_x = self._project(body_x)
    initial = torch.where(projected_x.norm(dim=-1, keepdim=True) > 1e-6,
                          torch.nn.functional.normalize(projected_x, dim=-1), self.reference)
    reset = self._pending_reset
    self._previous_forward = torch.where(reset[:, None], initial, self._previous_forward)
    self._previous_valid = self._previous_valid & ~reset
    self._previous_aligning = self._previous_aligning & ~reset
    self._pending_reset.zero_()

    minus_z = quat_apply(quat_w, self.up.new_tensor((0., 0., -1.)).expand_as(self.up))
    projection = self._project(minus_z)
    length = projection.norm(dim=-1)
    target = projection / length.clamp_min(1e-6)[:, None]
    self.valid = torch.where(self._previous_valid, length > self.release, length >= self.acquire)
    delta = torch.atan2(
      (torch.cross(self._previous_forward, target, dim=-1) * self.up).sum(-1),
      (self._previous_forward * target).sum(-1),
    )
    transitioning = self._previous_aligning | ~self._previous_valid
    limit = self.transition_rate * dt
    angle = delta.clamp(-limit, limit)
    rotated = (angle.cos()[:, None] * self._previous_forward
      + angle.sin()[:, None] * torch.cross(self.up, self._previous_forward, dim=-1))
    self.aligning = self.valid & transitioning & (delta.abs() > limit)
    candidate = torch.where(self.aligning[:, None], rotated, target)
    self.forward = torch.where(self.valid[:, None], candidate, self._previous_forward)
    self.forward = torch.nn.functional.normalize(self._project(self.forward), dim=-1)

  @property
  def basis_w(self):
    return torch.stack((self.forward, torch.cross(self.up, self.forward, dim=-1), self.up), dim=-1)

  @property
  def heading(self):
    left = torch.cross(self.up, self.reference, dim=-1)
    return torch.atan2((self.forward * left).sum(-1), (self.forward * self.reference).sum(-1))

  def to_frame(self, vector_w):
    """Express an absolute velocity at the same physical point in this frame."""
    return (self.basis_w.transpose(-1, -2) @ vector_w.unsqueeze(-1)).squeeze(-1)
