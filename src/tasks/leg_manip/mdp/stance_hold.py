"""Loose stance holding, separate from the strict endpoint certification."""
import math
import torch
from src.tasks.pedipulation.constants import DESIRED_ANGLES


class StanceHoldHistory:
  def __init__(self, count, dt, device):
    self.dt = dt
    self.air_time = torch.full((count, 4), 1., device=device)
    self.contacts = torch.zeros(count, 4, dtype=torch.bool, device=device)
    self.speeds = torch.zeros(max(1, round(.2 / dt)), count, 2, device=device)
    self.index = 0
    self.samples = 0

  def update(self, contacts, linear, angular):
    self.contacts.copy_(contacts)
    self.air_time.copy_(torch.where(contacts, 0., self.air_time + self.dt))
    self.speeds[self.index, :, 0] = linear.norm(dim=-1)
    self.speeds[self.index, :, 1] = angular.norm(dim=-1)
    self.index = (self.index + 1) % len(self.speeds)
    self.samples = min(self.samples + 1, len(self.speeds))

  def reset(self, ids):
    self.air_time[ids] = 1.
    self.contacts[ids] = False
    # A reset cannot immediately pass motion history with fabricated zeros.
    self.speeds[:, ids] = float('inf')

  def support(self, *, biped):
    recent = self.air_time <= .06 + 1e-6
    if biped:
      return recent[:, 2:].all(-1) & self.contacts[:, 2:].any(-1) & ~self.contacts[:, :2].any(-1)
    return recent.all(-1) & (self.contacts.sum(-1) >= 3)

  def motion_ok(self):
    mean = self.speeds[:self.samples].mean(0) if self.samples else self.speeds[0] + float('inf')
    return (mean[:, 0] < .3) & (mean[:, 1] < 1.)


def stance_hold(term, history, *, biped):
  data = term.robot.data
  height = data.root_link_pos_w[:, 2]
  gravity = data.projected_gravity_b
  if biped:
    # Evaluation uses nominal posture: all twelve joints are counted.
    pose = (data.joint_pos - data.joint_pos.new_tensor(DESIRED_ANGLES)).abs().mean(-1) < .35
    geometry = (-gravity[:, 0] > math.cos(math.radians(25.))) & (height > .44) & (height < .62)
  else:
    pose = (data.joint_pos - data.default_joint_pos).abs().mean(-1) < .25
    geometry = (-gravity[:, 2] > math.cos(math.radians(20.))) & ((height - term.workspace.nominal_height).abs() < .06)
  return geometry & pose & history.support(biped=biped) & history.motion_ok()
