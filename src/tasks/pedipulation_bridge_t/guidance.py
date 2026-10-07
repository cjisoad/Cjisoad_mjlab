"""Full rise reference in translation/heading invariant task coordinates."""
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor

SUPPORT_JOINTS = (0, 1, 2, 6, 7, 8, 9, 10, 11)
GUIDANCE_DIM = 22
ACTOR_DIM = 51 + GUIDANCE_DIM
CRITIC_DIM = 95 + GUIDANCE_DIM
GUIDANCE_CHANNELS = ('progress', 'height', 'reference_height', 'reference_gravity3',
  'reference_support_joints9', 'reference_fl_position3', 'reference_fl_unload', 'fr_error3')


class RiseGuidance:
  """Read nominal preparation through rise; hold the completed endpoint."""
  def __init__(self, motion, device):
    d = motion.data
    moving = np.flatnonzero(np.linalg.norm(d['joint_vel'], axis=-1) > 1e-7)
    if not len(moving):
      raise ValueError('Rise guidance requires a moving reference')
    self.rise_duration = float(d['time'][min(int(moving[-1])+1, len(d['time'])-1)])
    self.fps = motion.fps
    root = torch.as_tensor(d['root_state'], device=device, dtype=torch.float32)
    gravity = root.new_tensor((0., 0., -1.)).expand(len(root), 3)
    basis = anchor_basis(root[:, 3:7], gravity)
    fl = torch.as_tensor(d['foot_pos_w'][:, 0], device=device, dtype=torch.float32)
    self.frames = dict(height=root[:, 2],
      gravity=quat_apply_inverse(root[:, 3:7], gravity),
      joint_pos=torch.as_tensor(d['joint_pos'], device=device, dtype=torch.float32),
      fl_position=to_anchor(basis, fl-root[:, :3]),
      contact=torch.as_tensor(d['contact'], device=device, dtype=torch.float32),
      fl_unload=torch.as_tensor(d['fl_unload'], device=device, dtype=torch.float32))

  def sample(self, progress):
    position = progress.clamp(0., 1.) * self.rise_duration * self.fps
    low = position.long().clamp(0, len(self.frames['height'])-2)
    alpha = (position-low).clamp(0., 1.)
    result = {}
    for name, frames in self.frames.items():
      blend = alpha.reshape((-1,)+(1,)*(frames.ndim-1))
      result[name] = torch.lerp(frames[low], frames[low+1], blend)
    result['gravity'] = torch.nn.functional.normalize(result['gravity'], dim=-1)
    return result
