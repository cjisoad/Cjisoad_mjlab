"""Full rise reference in translation/heading invariant task coordinates."""
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor

SUPPORT_JOINTS = (0, 1, 2, 6, 7, 8, 9, 10, 11)
GUIDANCE_DIM = 33
ACTOR_DIM = 51 + GUIDANCE_DIM
CRITIC_DIM = ACTOR_DIM + 88
GUIDANCE_CHANNELS = ('height', 'fr_error3', 'progress', 'orientation_error_b3',
  'reference_height', 'reference_support_joints9', 'reference_support_joint_velocities9',
  'reference_linear_velocity3', 'reference_angular_velocity3')


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
    feet = torch.as_tensor(d['foot_pos_w'], device=device, dtype=torch.float32)
    self.frames = dict(root_state=root, quaternion=root[:, 3:7], height=root[:, 2],
      gravity=quat_apply_inverse(root[:, 3:7], gravity),
      joint_pos=torch.as_tensor(d['joint_pos'], device=device, dtype=torch.float32),
      joint_vel=torch.as_tensor(d['joint_vel'], device=device, dtype=torch.float32),
      linear_velocity=to_anchor(basis, root[:, 7:10]),
      angular_velocity=to_anchor(basis, root[:, 10:13]),
      foot_pos_w=feet, fl_height=feet[:, 0, 2],
      fl_position=to_anchor(basis, fl-root[:, :3]),
      contact=torch.as_tensor(d['contact'], device=device, dtype=torch.float32),
      fl_unload=torch.as_tensor(d['fl_unload'], device=device, dtype=torch.float32))
    self.zero = root.new_tensor(d.get('foot_zero', np.zeros(3)))
    nominal = to_anchor(basis, feet[:, 1]-root[:, :3])-self.zero
    self.fr_offsets = root.new_tensor(d.get('fr_offset_bank', nominal.cpu().numpy()[None]))
    self.fr_joints = root.new_tensor(d.get('fr_joint_bank', d['joint_pos'][None, :, 3:6]))
    shape = (len(self.fr_offsets), len(root), 3)
    if self.fr_offsets.shape != shape or self.fr_joints.shape != shape:
      raise ValueError('FR trajectory banks must have matching (K,T,3) shapes')
    if not torch.isfinite(self.fr_offsets).all() or not torch.isfinite(self.fr_joints).all():
      raise ValueError('FR trajectory banks must be finite')
    self.fr_velocities = torch.gradient(self.fr_joints, spacing=(1./self.fps,), dim=(1,))[0]
    self.fr_velocities[:, torch.as_tensor(d['time'] >= self.rise_duration, device=device)] = 0.
    self.frames['fr_offset'] = self.fr_offsets[0]

  def sample(self, progress, speed=None, bank_index=None):
    """Sample normalized rise phase; hold the endpoint with zero velocity."""
    progress = torch.as_tensor(progress, device=self.frames['root_state'].device)
    return self.sample_time(progress*self.rise_duration, speed=speed, bank_index=bank_index)

  def sample_time(self, time, speed=None, bank_index=None):
    """Sample asset state before world yaw/translation alignment.

    Root state velocities use asset world axes. The separate velocity channels
    use the reference anchor; the command aligns them to the live anchor.
    FR bank joint and target samples always use the same index and time.
    """
    root_frames = self.frames['root_state']
    time = torch.as_tensor(time, device=root_frames.device, dtype=root_frames.dtype).reshape(-1)
    speed = torch.as_tensor(1. if speed is None else speed, device=time.device, dtype=time.dtype).expand_as(time)
    if not torch.isfinite(time).all() or not torch.isfinite(speed).all() or (speed < 0.).any():
      raise ValueError('Sample time and nonnegative speed must be finite')
    position = time.clamp(0., self.rise_duration)*self.fps
    low = position.long().clamp(0, len(self.frames['height'])-2)
    alpha = (position-low).clamp(0., 1.)
    result = {}
    for name, frames in self.frames.items():
      blend = alpha.reshape((-1,)+(1,)*(frames.ndim-1))
      result[name] = torch.lerp(frames[low], frames[low+1], blend)
    # Quaternion signs are interchangeable; blend in a common hemisphere.
    q0, q1 = root_frames[low, 3:7], root_frames[low+1, 3:7]
    q1 = torch.where((q0*q1).sum(-1, keepdim=True) < 0., -q1, q1)
    quat = torch.nn.functional.normalize(torch.lerp(q0, q1, alpha[:, None]), dim=-1)
    result['root_state'][:, 3:7] = quat
    result['quaternion'] = quat
    bank = torch.as_tensor(0 if bank_index is None else bank_index, device=time.device, dtype=torch.long).expand_as(low)
    if (bank < 0).any() or (bank >= len(self.fr_offsets)).any():
      raise ValueError('FR bank index is outside the reference bank')
    for name, frames in (('fr_offset', self.fr_offsets), ('frq', self.fr_joints), ('frqd', self.fr_velocities)):
      result[name] = torch.lerp(frames[bank, low], frames[bank, low+1], alpha[:, None])
    result['joint_pos'][:, 3:6] = result.pop('frq')
    result['joint_vel'][:, 3:6] = result.pop('frqd')
    factor = speed*(time < self.rise_duration)
    result['joint_vel'] *= factor[:, None]
    result['root_state'][:, 7:] *= factor[:, None]
    root = result['root_state']
    gravity = root.new_tensor((0., 0., -1.)).expand(len(root), 3)
    basis = anchor_basis(quat, gravity)
    result['gravity'] = quat_apply_inverse(quat, gravity)
    result['linear_velocity'] = to_anchor(basis, root[:, 7:10])
    result['angular_velocity'] = to_anchor(basis, root[:, 10:13])
    result['foot_pos_w'][:, 1] = root[:, :3] + (basis @ (self.zero+result['fr_offset'])[..., None]).squeeze(-1)
    result['fl_position'] = to_anchor(basis, result['foot_pos_w'][:, 0]-root[:, :3])
    return result
