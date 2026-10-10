"""Reuse one-shot tracking with stable teacher heading at the biped entry."""
from dataclasses import dataclass
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
from src.tasks.teacher_common.anchor_frame import anchor_basis
from src.tasks.transition_t.commands import TransitionCommand, TransitionCommandCfg
from .data import DEFAULT_MOTION_PATH, verify_approved_assets


def teacher_heading_quat(quaternion):
  basis = anchor_basis(quaternion, quaternion.new_tensor([0., 0., -1.]))
  angle = torch.atan2(basis[:, 1, 0], basis[:, 0, 0])
  result = torch.zeros_like(quaternion)
  result[:, 0] = torch.cos(angle/2)
  result[:, 3] = torch.sin(angle/2)
  return result


def align_reference(term, ids, actual_pos, actual_quat):
  reference_quat = term.motion.body_quat_w[0, term.motion_anchor_body_index].expand(len(ids), -1)
  rotation = quat_mul(teacher_heading_quat(actual_quat), quat_inv(teacher_heading_quat(reference_quat)))
  reference_pos = term.motion.body_pos_w[0, term.motion_anchor_body_index].expand(len(ids), -1)
  translation = actual_pos-term._env.scene.env_origins[ids]-quat_apply(rotation, reference_pos)
  translation[:, 2] = 0.
  term.reference_rotation[ids] = rotation
  term.reference_translation[ids] = translation


@dataclass(kw_only=True)
class DownMotionCfg(TransitionCommandCfg):
  motion_file: str = str(DEFAULT_MOTION_PATH)

  def build(self, env):
    return DownMotion(self, env)


class DownMotion(TransitionCommand):
  def __init__(self, cfg, env):
    verify_approved_assets(motion_file=cfg.motion_file)
    super().__init__(cfg, env)

  def start_from_current_state(self, env_ids):
    self.time_steps[env_ids] = 0
    self.completed[env_ids] = False
    align_reference(self, env_ids, self.robot_anchor_pos_w[env_ids], self.robot_anchor_quat_w[env_ids])
    self.time_left[env_ids] = self.cfg.resampling_time_range[1]
    self._last_update_step[env_ids] = self._env.common_step_counter
    self._refresh_relative()
