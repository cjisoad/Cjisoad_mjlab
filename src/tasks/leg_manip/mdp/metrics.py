"""Streaming whole-body metrics, independent of episode and promotion resets."""
import torch
from src.tasks.loco_pedipulation.mdp.metrics import ManipulationMetrics, TRACKING_FIELDS
from src.tasks.pedipulation.constants import DESIRED_ANGLES

GROUPS = tuple(f'{mode}_{speed}' for speed in ('stationary', 'moving') for mode in ('quad', 'tripod', 'biped'))
COURSE_FIELDS = (*TRACKING_FIELDS, 'ready_fraction', 'pose_error', 'base_height',
                 'front_contacts', 'rear_contacts', 'height_gate', 'fr_error', 'gravity_angle_deg')
SWITCH_EVENTS = ('attempts', 'succeeded', 'timeouts', 'interrupted', 'uncertified')

class CourseMetrics(ManipulationMetrics):
  def __init__(self, device):
    super().__init__(device)
    self.tracking = torch.zeros(6, len(COURSE_FIELDS), device=device, dtype=torch.float64)
    self.switches = torch.zeros(2, len(SWITCH_EVENTS), device=device, dtype=torch.float64)

  def switch_event(self, direction, name, mask):
    self.switches[direction, SWITCH_EVENTS.index(name)] += mask.sum()

  def record_course(self, term, velocity, angular, ready):
    command, data = term.command, term.robot.data
    moving = command[:, :3].norm(dim=-1) > .05
    linear = (velocity[:, :2] - command[:, :2]).norm(dim=-1)
    yaw = (angular[:, 2] - command[:, 2]).abs()
    desired = data.joint_pos.new_tensor(DESIRED_ANGLES).expand_as(data.joint_pos)
    target = torch.where(term.mode[:, None], desired, data.default_joint_pos)
    pose = (data.joint_pos - target).abs().mean(-1)
    contacts = term.contacts
    height = data.root_link_pos_w[:, 2]
    values = torch.stack((torch.ones_like(linear), linear + .25 * yaw, linear, yaw,
      velocity[:, 0], command[:, 0], velocity[:, :2].norm(dim=-1), command[:, :2].norm(dim=-1),
      ready.float(), pose, height, contacts[:, :2].sum(-1), contacts[:, 2:].sum(-1),
      (torch.exp(-(height - .52).abs() * 5) > .70).float(),
      (term.foot_pos_anchor - term.reference).norm(dim=-1),
      torch.acos(torch.where(term.mode, -data.projected_gravity_b[:, 0], -data.projected_gravity_b[:, 2]).clamp(-1, 1)) * (180. / torch.pi)), dim=-1)
    loco_mode = torch.where(term.manipulating, torch.ones_like(term.phase), torch.zeros_like(term.phase))
    mode = torch.where(term.mode, torch.full_like(term.phase, 2), loco_mode)
    for i in range(6):
      mask = (mode == i % 3) & (moving == bool(i >= 3))
      self.tracking[i] += (values * mask[:, None]).sum(0)
      term.episode_tracking[:, i, 0] += mask
      term.episode_tracking[:, i, 1] += values[:, 1] * mask

  def drain(self):
    values = torch.cat((self.landing, self.switches.flatten(), self.tracking.flatten()))
    self.landing.zero_()
    self.switches.zero_()
    self.tracking.zero_()
    return values
