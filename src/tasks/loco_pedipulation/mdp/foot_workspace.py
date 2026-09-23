"""Reference-pose FK targets for the standard quadruped Go2 asset."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg

FR_JOINTS = ('FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint')
FOOT_NAMES = ('FL', 'FR', 'RL', 'RR')


@dataclass(frozen=True)
class FootWorkspace:
  zero: np.ndarray
  offsets: np.ndarray
  joint_positions: np.ndarray
  nominal_height: float
  foot_radius: float


@lru_cache(maxsize=8)
def build_foot_workspace(joint_delta=(.25, .55, .65), max_offset=(.10, .07, .14),
                         lift_range=(.05, .12)):
  if (len(joint_delta) != 3 or len(max_offset) != 3 or len(lift_range) != 2
      or not np.isfinite((*joint_delta, *max_offset, *lift_range)).all()
      or min(joint_delta) <= 0 or min(max_offset) <= 0
      or not 0 < lift_range[0] < lift_range[1] <= max_offset[2]):
    raise ValueError('Require positive finite workspace bounds and ordered lift range')
  cfg = get_go2_robot_cfg()
  model = cfg.spec_fn().compile()
  data = mujoco.MjData(model)
  data.qpos[:3] = cfg.init_state.pos
  data.qpos[3:7] = (1., 0., 0., 0.)
  for index in range(model.njnt):
    joint = model.joint(index)
    for pattern, value in cfg.init_state.joint_pos.items():
      if re.fullmatch(pattern, joint.name):
        data.qpos[joint.qposadr[0]] = value
  mujoco.mj_forward(model, data)
  site, base = model.site('FR').id, model.body('base_link').id
  zero = (data.site_xpos[site] - data.xpos[base]).copy()
  radius = float(model.geom('FR_foot_collision').size[0])
  nominal_height = radius - zero[2]
  joints = [model.joint(name).id for name in FR_JOINTS]
  addresses = model.jnt_qposadr[joints]
  rest = data.qpos[addresses].copy()
  limits = model.jnt_range[joints]
  velocity_addresses = model.jnt_dofadr[joints]
  jacobian = np.zeros((3, model.nv))
  low = np.maximum(rest - joint_delta, limits[:, 0] + .05)
  high = np.minimum(rest + joint_delta, limits[:, 1] - .05)
  calf_geoms = [model.geom(name).id for name in
                ('FR_foot_collision', 'FR_calf1_collision', 'FR_calf2_collision')]
  obstacles = [i for i in range(model.ngeom)
               if re.fullmatch(r'(base.*|FL_.*)_collision', model.geom(i).name)]

  def clear_path(offset):
    # Solve the actual Cartesian interpolation at startup, never during rollout.
    solution = rest.copy()
    for fraction in np.linspace(0., 1., 9):
      target = zero + fraction * offset

      def residual(angles):
        data.qpos[addresses] = angles
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
      data.qpos[addresses] = solution
      mujoco.mj_forward(model, data)
      if data.site_xpos[site, 2] - data.xpos[base, 2] < zero[2] - .005:
        return False
      for moving, obstacle in product(calf_geoms, obstacles):
        if mujoco.mj_geomDistance(model, data, moving, obstacle, .02, None) < .01:
          return False
    return True

  offsets, witnesses = [], []
  for values in product(*(np.linspace(lo, hi, 9) for lo, hi in zip(low, high))):
    angles = np.asarray(values)
    data.qpos[addresses] = angles
    mujoco.mj_forward(model, data)
    offset = data.site_xpos[site] - data.xpos[base] - zero
    if ((np.abs(offset) <= max_offset).all()
        and lift_range[0] <= offset[2] <= lift_range[1] and clear_path(offset)):
      offsets.append(offset.copy())
      witnesses.append(angles.copy())
  if not offsets:
    raise ValueError('No FR targets satisfy the quadruped workspace bounds')
  return FootWorkspace(zero, np.asarray(offsets), np.asarray(witnesses), nominal_height, radius)
