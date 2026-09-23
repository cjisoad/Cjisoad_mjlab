"""Count-based logging buffers, independent of episode and curriculum resets."""

import torch


LANDING_EVENTS = ('attempts', 'confirmed', 'timeouts', 'interrupted')
TRACKING_FIELDS = ('samples', 'velocity_error', 'linear_error', 'yaw_error',
                   'actual_vx', 'command_vx', 'actual_speed', 'command_speed')


class ManipulationMetrics:
  def __init__(self, device):
    self.landing = torch.zeros(len(LANDING_EVENTS), device=device, dtype=torch.float64)
    self.tracking = torch.zeros(2, len(TRACKING_FIELDS), device=device, dtype=torch.float64)

  def landing_event(self, name, mask):
    self.landing[LANDING_EVENTS.index(name)] += mask.sum()

  def record_tracking(self, hold, command, velocity, angular):
    moving = command.norm(dim=-1) > .05
    linear_error = (velocity[:, :2] - command[:, :2]).norm(dim=-1)
    yaw_error = (angular[:, 2] - command[:, 2]).abs()
    values = torch.stack((torch.ones_like(linear_error), linear_error + .25 * yaw_error,
      linear_error, yaw_error, velocity[:, 0], command[:, 0],
      velocity[:, :2].norm(dim=-1), command[:, :2].norm(dim=-1)), dim=-1)
    for index, mask in enumerate((hold & ~moving, hold & moving)):
      self.tracking[index] += (values * mask[:, None]).sum(0)

  def drain(self):
    values = torch.cat((self.landing, self.tracking.flatten()))
    self.landing.zero_()
    self.tracking.zero_()
    return values
