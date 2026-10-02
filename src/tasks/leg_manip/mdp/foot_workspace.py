"""Two-segment FR target bank with FK witnesses on the stand spec snapshot.

The low segment holds tripod-capable targets validated from the quadruped
default stance with per-target Cartesian path IK and collision clearance, as in
loco_pedipulation. The high segment holds biped targets validated from the
reared reference stance after rotation into the unified anchor frame, with
endpoint collision clearance replacing the source task's torso-top floor so the
operation band extends below the head end. All offsets are relative to one
shared zero: the quadruped-stance FR site offset from the base.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.tasks.pedipulation.constants import DEFAULT_ANGLES, DESIRED_ANGLES, JOINT_NAMES
from src.tasks.pedipulation.robot import get_stand_spec

from ..constants import BIPED_HIGH, BIPED_LOW, DZ_MIN, TRIPOD_HIGH

LOW, HIGH = 0, 1
FR_JOINTS = JOINT_NAMES[3:6]
FOOT_NAMES = ('FL', 'FR', 'RL', 'RR')
MOVING_GEOMS = ('FR_foot_collision', 'FR_calf_collision',
                'FR_calflower_collision', 'FR_calflower1_collision')
# Body +X is up and body -Z is anchor-forward at the full rear-leg stand.
REARED_ROTATION = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])


@dataclass(frozen=True)
class FootWorkspace:
  zero: np.ndarray
  offsets: np.ndarray
  joint_positions: np.ndarray
  segment: np.ndarray
  nominal_height: float
  foot_radius: float
  nominal_biped_offset: np.ndarray


@lru_cache(maxsize=8)
def build_foot_workspace(lift_range=(DZ_MIN, TRIPOD_HIGH),
                         biped_range=(BIPED_LOW, BIPED_HIGH),
                         max_offset=(.12, .09), biped_offset_x=(-.05, .30),
                         biped_offset_y=(-.10, .10), grid_points=9):
  if (len(lift_range) != 2 or len(biped_range) != 2 or len(max_offset) != 2
      or len(biped_offset_x) != 2 or len(biped_offset_y) != 2):
    raise ValueError('Workspace ranges must be (low, high) pairs')
  values = (*lift_range, *biped_range, *max_offset, *biped_offset_x, *biped_offset_y)
  if not np.isfinite(values).all() or min(max_offset) <= 0 or grid_points < 2:
    raise ValueError('Require finite ranges, positive offset bounds and at least two grid points')
  if not 0 < lift_range[0] < lift_range[1] or not biped_range[0] < biped_range[1]:
    raise ValueError('Segment dz ranges must be positive and ordered')

  model = get_stand_spec().compile()
  data = mujoco.MjData(model)
  addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
  site, base = model.site('FR').id, model.body('base_link').id

  def forward_kinematics(angles):
    data.qpos[addresses] = angles
    mujoco.mj_kinematics(model, data)
    return data.site_xpos[site] - data.xpos[base]

  zero = forward_kinematics(DEFAULT_ANGLES).copy()
  nominal_biped_offset = REARED_ROTATION @ forward_kinematics(DESIRED_ANGLES) - zero
  radius = float(model.geom('FR_foot_collision').size[0])
  nominal_height = radius - zero[2]
  joint_ids = [model.joint(name).id for name in FR_JOINTS]
  limits = model.jnt_range[joint_ids]
  velocity_addresses = model.jnt_dofadr[joint_ids]
  jacobian = np.zeros((3, model.nv))
  moving = [model.geom(name).id for name in MOVING_GEOMS]
  obstacles = [i for i in range(model.ngeom)
               if re.fullmatch(r'(base_link|Head_upper|Head_lower|FL_.*)_collision',
                               model.geom(i).name)]

  def clearance_ok():
    for moved, obstacle in product(moving, obstacles):
      if mujoco.mj_geomDistance(model, data, moved, obstacle, .02, None) < .01:
        return False
    return True

  def clear_path(offset):
    # Solve the actual Cartesian interpolation at startup, never during rollout.
    rest = np.array(DEFAULT_ANGLES[3:6])
    solution = rest.copy()
    for fraction in np.linspace(0., 1., 9):
      target = zero + fraction * offset

      def residual(angles):
        data.qpos[addresses[3:6]] = angles
        mujoco.mj_forward(model, data)
        return data.site_xpos[site] - data.xpos[base] - target

      def jac(angles):
        residual(angles)
        mujoco.mj_jacSite(model, data, jacobian, None, site)
        return jacobian[:, velocity_addresses]

      result = least_squares(residual, solution, jac=jac,
        bounds=(limits[:, 0] + .05, limits[:, 1] - .05), max_nfev=30)
      if np.linalg.norm(result.fun) > 1e-4:
        return False
      solution = result.x
      data.qpos[addresses[3:6]] = solution
      mujoco.mj_forward(model, data)
      if data.site_xpos[site, 2] - data.xpos[base, 2] < zero[2] - .005:
        return False
      if not clearance_ok():
        return False
    return True

  grid = [np.linspace(lo + .05, hi - .05, grid_points) for lo, hi in limits]
  offsets, witnesses, segments = [], [], []
  for candidate in product(*grid):
    angles = list(DEFAULT_ANGLES)
    angles[3:6] = candidate
    offset = forward_kinematics(angles) - zero
    if ((np.abs(offset[:2]) <= max_offset).all()
        and lift_range[0] <= offset[2] <= lift_range[1]
        and clearance_ok() and clear_path(offset)):
      offsets.append(offset.copy())
      witnesses.append(candidate)
      segments.append(LOW)
  low_count = len(offsets)
  for candidate in product(*grid):
    angles = list(DESIRED_ANGLES)
    angles[3:6] = candidate
    offset = REARED_ROTATION @ forward_kinematics(angles) - zero
    if (biped_offset_x[0] <= offset[0] <= biped_offset_x[1]
        and biped_offset_y[0] <= offset[1] <= biped_offset_y[1]
        and biped_range[0] <= offset[2] <= biped_range[1]
        and clearance_ok()):
      offsets.append(offset.copy())
      witnesses.append(candidate)
      segments.append(HIGH)
  if not low_count:
    raise ValueError('No FR targets satisfy the tripod workspace bounds')
  if len(offsets) == low_count:
    raise ValueError('No FR targets satisfy the biped workspace bounds')
  return FootWorkspace(zero, np.asarray(offsets), np.asarray(witnesses),
                       np.asarray(segments, dtype=np.int64), nominal_height, radius, nominal_biped_offset)
