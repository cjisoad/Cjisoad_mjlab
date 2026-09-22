"""MuJoCo FK witnesses for FR targets with the base and other legs fixed."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import mujoco
import numpy as np

from ..constants import DESIRED_ANGLES, JOINT_NAMES
from ..robot import get_stand_spec


@dataclass(frozen=True)
class FootWorkspace:
  zero_body: np.ndarray
  offsets: np.ndarray
  joint_positions: np.ndarray
  min_height: float


@lru_cache(maxsize=8)
def build_foot_workspace(joint_delta=(.3, .4, .4), max_offset=(.10, .08, .10),
                         min_height=.10, min_offset=.02):
  """Filter FK samples in an upright base-origin anchor frame, in meters.

  The grid stays within joint limits and retains a witness FR configuration for
  every target. No inverse kinematics or CPU work occurs during rollout steps.
  """
  if (len(joint_delta) != 3 or len(max_offset) != 3
      or not np.isfinite((*joint_delta, *max_offset, min_height, min_offset)).all()
      or min(joint_delta) <= 0 or min(max_offset) <= 0 or min_offset <= 0):
    raise ValueError('FR workspace requires finite positive ranges and minimum offset')
  model = get_stand_spec().compile()
  data = mujoco.MjData(model)
  addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
  data.qpos[addresses] = DESIRED_ANGLES
  mujoco.mj_kinematics(model, data)
  site, base = model.site('FR').id, model.body('base_link').id
  zero_body = (data.site_xpos[site] - data.xpos[base]).copy()
  # Pitch -pi/2: body +X is up and body -Z is anchor-forward.
  rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
  zero_anchor = rotation @ zero_body
  torso = model.geom('base_link_collision')
  torso_top = float(torso.pos[2] + torso.size[2])
  height_floor = max(min_height, torso_top + .02)
  limits = model.jnt_range[[model.joint(name).id for name in JOINT_NAMES[3:6]]]
  desired = np.array(DESIRED_ANGLES[3:6])
  low = np.maximum(desired - joint_delta, limits[:, 0] + .05)
  high = np.minimum(desired + joint_delta, limits[:, 1] - .05)
  offsets, joints = [], []
  for angles in product(*(np.linspace(lo, hi, 9) for lo, hi in zip(low, high))):
    data.qpos[addresses[3:6]] = angles
    mujoco.mj_kinematics(model, data)
    target = rotation @ (data.site_xpos[site] - data.xpos[base])
    offset = target - zero_anchor
    if (target[2] >= height_floor and np.linalg.norm(offset) >= min_offset
        and (np.abs(offset) <= max_offset).all()):
      offsets.append(offset)
      joints.append(angles)
  if not offsets:
    raise ValueError('No reachable FR targets satisfy the configured workspace')
  return FootWorkspace(zero_body, np.array(offsets), np.array(joints), height_floor)
