"""Offline compatible LOW-to-HIGH paths in the shared gravity-aligned anchor.

Only a filtered library of Cartesian line segments is exposed. Endpoint FK
witnesses alone do not establish intermediate reachability. Construction checks
the tripod prefix and solves full-pose support/COM IK along the nominal rear-up
reference. These are sampled kinematic/static checks, not dynamic validation.
"""
from functools import lru_cache
import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace, LOW, HIGH, MOVING_GEOMS
from src.tasks.pedipulation.constants import DEFAULT_ANGLES, DESIRED_ANGLES, JOINT_NAMES
from src.tasks.pedipulation_transition_t.motion import (
  MotionReference, NOMINAL_MOTION_PATH, FOOT_NAMES, make_model,
)

PATHS_PATH = NOMINAL_MOTION_PATH.with_name('bridge_target_paths.npz')
PATH_ASSET = PATHS_PATH
AUDIT_PATH = Path('outputs/bridge_target_paths/audit.json')
H_START_VALUES = np.linspace(.25, .28, 13)


def build_bridge_target_paths(count=12, prefix_samples=41, rise_samples=65):
  """Filter teacher-bank pairs and retain full static pose witnesses offline."""
  if count < 4 or prefix_samples < 20 or rise_samples < 30:
    raise ValueError('Require at least four paths and dense prefix/rise samples')
  workspace, reference = build_foot_workspace(), MotionReference(NOMINAL_MOTION_PATH)
  model, data = make_model(), None
  data = mujoco.MjData(model)
  qa = np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
  va = np.array([int(model.joint(n).dofadr[0]) for n in JOINT_NAMES])
  joint_ids = [model.joint(n).id for n in JOINT_NAMES]
  sites = [model.site(n).id for n in FOOT_NAMES]
  base, floor = model.body('base_link').id, model.geom('floor').id
  limits = model.jnt_range[joint_ids]
  bounds = (np.r_[-1., -1., limits[:, 0]+.025], np.r_[1., 1., limits[:, 1]-.025])
  jac = np.zeros((3, model.nv))
  columns = np.r_[0, 1, va]
  mass = float(model.body_mass.sum())
  torque_limits = np.array([35.55 if 'calf' in n else 23.7 for n in JOINT_NAMES])
  low = np.flatnonzero((workspace.segment == LOW) & (workspace.offsets[:, 2] <= .2))
  high = np.flatnonzero((workspace.segment == HIGH) &
    (workspace.offsets[:, 2] >= .30) & (workspace.offsets[:, 2] <= .33))
  # Prefer forward endpoints, which leave space around the folded-leg gap.
  high = high[np.argsort(-workspace.offsets[high, 0])]
  low = low[np.argsort(workspace.offsets[low, 2])]
  selected, rejected = [], []
  moving_geoms = [model.geom(n).id for n in MOVING_GEOMS]
  obstacle_geoms = [i for i in range(model.ngeom) if
    re.fullmatch(r'(base_link|Head_upper|Head_lower|FL_.*)_collision', model.geom(i).name)]

  def teacher_clearance():
    return all(mujoco.mj_geomDistance(model, data, moved, obstacle, .02, None) >= .01
      for moved in moving_geoms for obstacle in obstacle_geoms)

  def endpoint_witness(end, hint):
    data.qpos[:7] = [0., 0., .5, 2**-.5, 0., -2**-.5, 0.]
    data.qpos[qa] = DESIRED_ANGLES

    def residual(q):
      data.qpos[qa[3:6]] = q
      mujoco.mj_forward(model, data)
      return data.site_xpos[sites[1]]-data.qpos[:3]-workspace.zero-end

    def derivative(q):
      residual(q)
      mujoco.mj_jacSite(model, data, jac, None, sites[1])
      return jac[:, va[3:6]]

    result = least_squares(residual, hint, jac=derivative,
      bounds=(limits[3:6, 0]+.025, limits[3:6, 1]-.025), max_nfev=50,
      ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if np.max(np.abs(residual(result.x))) > .0001 or not teacher_clearance():
      return None
    return result.x.copy()

  endpoints = [(int(i), workspace.offsets[i].copy(), workspace.joint_positions[i].copy(), True) for i in high]
  # Retain small continuous endpoint-height variants only after HIGH-stance FK
  # and the entire nominal support-transfer path pass the same audit.
  for i in high[:2]:
    for delta in (-.002, .002, .004, .006):
      end = workspace.offsets[i].copy()
      end[2] += delta
      if .30 <= end[2] <= .33:
        witness = endpoint_witness(end, workspace.joint_positions[i])
        if witness is not None:
          endpoints.append((int(i), end, witness, False))

  def penetration():
    for contact in data.contact:
      names = [model.geom(int(g)).name for g in contact.geom]
      nonfoot_ground = (floor in contact.geom and contact.dist < -.002 and
        not any('_foot_collision' in n for n in names))
      self_contact = floor not in contact.geom and contact.dist < -.005
      fr_contact = any(n.startswith('FR_') for n in names) and contact.dist < -.002
      if nonfoot_ground or self_contact or fr_contact:
        return True
    return False

  def source_prefix(start, end):
    """Check LOW stance, zero-to-start lift, and the full prefix to h_start=.28."""
    waypoint = start+(.28-start[2])/(end[2]-start[2])*(end-start)
    if np.any(np.abs(waypoint[:2]) > [.12, .09]):
      return False
    targets = np.concatenate((np.linspace(np.zeros(3), start, 17),
      np.linspace(start, waypoint, prefix_samples)))
    solution = np.array(DEFAULT_ANGLES[3:6])
    data.qpos[:7] = [0., 0., workspace.nominal_height, 1., 0., 0., 0.]
    data.qpos[qa] = DEFAULT_ANGLES
    for target in targets:
      def residual(q):
        data.qpos[qa[3:6]] = q
        mujoco.mj_forward(model, data)
        return data.site_xpos[sites[1]]-data.qpos[:3]-workspace.zero-target

      def derivative(q):
        residual(q)
        mujoco.mj_jacSite(model, data, jac, None, sites[1])
        return jac[:, va[3:6]]

      result = least_squares(residual, solution, jac=derivative,
        bounds=(limits[3:6, 0]+.025, limits[3:6, 1]-.025), max_nfev=40,
        ftol=1e-10, xtol=1e-10, gtol=1e-10)
      solution = result.x
      if np.max(np.abs(residual(solution))) > .001 or penetration() or not teacher_clearance():
        return False
    return True

  def solve_path(start, end, threshold):
    heights = np.r_[np.linspace(start[2], threshold, prefix_samples),
      np.linspace(threshold, end[2], rise_samples)[1:]]
    progress = np.clip((heights-threshold)/(end[2]-threshold), 0., 1.)
    times = .9+1.6*progress
    frames = reference.sample(times)
    x, witnesses, targets, errors, residuals, ratios, clearance = None, [], [], [], [], [], []
    for i, (height, time) in enumerate(zip(heights, times)):
      root, q = frames['root_state'][i], frames['joint_pos'][i]
      feet = frames['foot_pos_w'][i]
      position = time*reference.fps
      index = min(int(position), len(reference.data['time'])-2)
      alpha = position-index
      force = ((1.-alpha)*reference.data['static_contact_force'][index]+
        alpha*reference.data['static_contact_force'][index+1])
      com_goal = (force[:, 2, None]*feet).sum(axis=0)[:2]/(mass*9.81)
      target = start+(height-start[2])/(end[2]-start[2])*(end-start)
      if x is None:
        x = np.r_[root[:2], q]

      def residual(x):
        data.qpos[:7] = root[:7]
        data.qpos[:2], data.qpos[qa] = x[:2], x[2:]
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        return np.r_[(data.site_xpos[sites][[0, 2, 3]]-feet[[0, 2, 3]]).ravel(),
          data.site_xpos[sites[1]]-data.qpos[:3]-workspace.zero-target,
          data.subtree_com[base, :2]-com_goal, .0003*(x[2:]-q)]

      def derivative(x):
        residual(x)
        rows = []
        for foot in (0, 2, 3, 1):
          mujoco.mj_jacSite(model, data, jac, None, sites[foot])
          row = jac[:, columns].copy()
          if foot == 1:
            row[:, :2] -= np.eye(3)[:, :2]
          rows.append(row)
        mujoco.mj_jacSubtreeCom(model, data, jac, base)
        rows.extend((jac[:2, columns].copy(), np.c_[np.zeros((12, 2)), .0003*np.eye(12)]))
        return np.vstack(rows)

      result = least_squares(residual, x, jac=derivative, bounds=bounds,
        max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10)
      x = result.x
      error = float(np.max(np.abs(residual(x)[:14])))
      generalized = np.zeros(model.nv)
      for foot in (0, 2, 3):
        mujoco.mj_jacSite(model, data, jac, None, sites[foot])
        generalized += jac.T@force[foot]
      tau = data.qfrc_bias-generalized
      static_residual = float(np.linalg.norm(tau[:6])/(mass*9.81))
      ratio = float(np.max(np.abs(tau[va])/torque_limits))
      fr_clearance = float(data.site_xpos[sites[1], 2]-workspace.foot_radius)
      rear_hip_error = float(np.max(np.abs(x[2:][[6, 9]])))
      endpoint_hip_bad = progress[i] >= 1.-1e-9 and rear_hip_error > .1
      if error > .001 or static_residual > .005 or ratio >= 1. or fr_clearance < .02 or penetration() or endpoint_hip_bad:
        return None, dict(height=float(height), threshold=float(threshold),
          ik_error_m=error, static_base_residual=static_residual, torque_ratio=ratio)
      witnesses.append(data.qpos.copy())
      targets.append(target)
      errors.append(error); residuals.append(static_residual); ratios.append(ratio); clearance.append(fr_clearance)
    return dict(qpos=np.array(witnesses), offsets=np.array(targets), reference_time=times,
      max_ik_error_m=max(errors), max_static_base_residual=max(residuals),
      max_static_torque_ratio=max(ratios), min_fr_clearance_m=min(clearance)), None

  for low_index in low:
    start = workspace.offsets[low_index]
    used_heights = {round(s[1][2], 7) for s in selected}
    candidates = sorted(endpoints, key=lambda item:(round(item[1][2], 7) in used_heights, -item[1][0]))
    for high_index, end, endpoint_q, is_bank_endpoint in candidates:
      if not source_prefix(start, end):
        rejected.append(dict(low_index=int(low_index), high_index=int(high_index),
          end_offset=end.tolist(), reason='tripod_prefix'))
        continue
      audited = []
      for threshold in H_START_VALUES:
        result, failure = solve_path(start, end, threshold)
        if failure is not None:
          rejected.append(dict(low_index=int(low_index), high_index=int(high_index),
            end_offset=end.tolist(), reason='full_pose_path', **failure))
          break
        audited.append(result)
      if len(audited) == len(H_START_VALUES):
        selected.append((start.copy(), end.copy(), audited, int(low_index), int(high_index), endpoint_q, is_bank_endpoint))
        break  # Retain different initial tripod targets instead of duplicate starts.
    if len(selected) >= count:
      break
  if len(selected) < 4:
    raise ValueError(f'Only {len(selected)} compatible LOW-to-HIGH paths passed; thresholds were not widened')
  knee_spacings, foot_width_reductions, final_hips = [], [], []
  for candidate in selected:
    for audited in candidate[2]:
      data.qpos[:] = audited['qpos'][-1]
      mujoco.mj_forward(model, data)
      knees = data.xpos[[model.body('RL_calf').id, model.body('RR_calf').id]]
      feet = data.site_xpos[np.array(sites)[[2, 3]]]
      spacing = float(knees[0, 1]-knees[1, 1])
      knee_spacings.append(spacing)
      foot_width_reductions.append(float(feet[0, 1]-feet[1, 1])-spacing)
      final_hips.append(data.qpos[qa[[6, 9]]].copy())
  report = dict(kind='sampled_static_LOW_to_HIGH_path_library',
    paths=len(selected), low_candidates=len(low), high_candidates=len(high),
    h_start_values=H_START_VALUES.tolist(), requested_h_end_range=[.30, .33],
    prefix_samples=prefix_samples, rise_samples=rise_samples,
    zero_to_low_samples=17,
    full_pose_witnesses=len(selected)*len(H_START_VALUES)*(prefix_samples+rise_samples-1),
    max_ik_error_m=max(a['max_ik_error_m'] for s in selected for a in s[2]),
    max_static_base_residual=max(a['max_static_base_residual'] for s in selected for a in s[2]),
    max_static_torque_ratio=max(a['max_static_torque_ratio'] for s in selected for a in s[2]),
    min_fr_clearance_m=min(a['min_fr_clearance_m'] for s in selected for a in s[2]),
    ground_or_self_penetrations=[], dynamic_tracking_validated=False,
    reference_sha256=reference.sha256, rejected_pairs=rejected,
    distinct_endpoint_heights_m=sorted({float(s[1][2]) for s in selected}),
    initial_dz_range_m=[float(min(s[0][2] for s in selected)), float(max(s[0][2] for s in selected))],
    original_HIGH_endpoints=sum(bool(s[6]) for s in selected),
    perturbed_HIGH_FK_endpoints=sum(not bool(s[6]) for s in selected),
    max_final_rear_hip_abs_rad=float(np.max(np.abs(final_hips))),
    max_path_rear_hip_abs_rad=max(float(np.max(np.abs(a['qpos'][:, qa[[6, 9]]]))) for s in selected for a in s[2]),
    min_final_knee_spacing_m=min(knee_spacings), max_final_knee_spacing_m=max(knee_spacings),
    max_final_knee_width_reduction_m=max(foot_width_reductions),
    note='Discrete full-pose IK/static audit of straight XYZ paths; no claim that the Cartesian bounds form a reachable box or that dynamics can track the witnesses.')
  payload = dict(schema_version=np.array(1), zero=workspace.zero,
    starts=np.array([s[0] for s in selected]), ends=np.array([s[1] for s in selected]),
    low_indices=np.array([s[3] for s in selected]), high_indices=np.array([s[4] for s in selected]),
    endpoint_joint_witnesses=np.array([s[5] for s in selected]),
    endpoint_from_discrete_bank=np.array([s[6] for s in selected]),
    h_start_values=H_START_VALUES, report_json=np.array(json.dumps(report)),
    qpos_witnesses=np.array([[a['qpos'] for a in s[2]] for s in selected]),
    offset_witnesses=np.array([[a['offsets'] for a in s[2]] for s in selected]),
    reference_time=np.array([[a['reference_time'] for a in s[2]] for s in selected]))
  return payload, report


@lru_cache(maxsize=4)
def _load_paths(path):
  with np.load(path, allow_pickle=False) as archive:
    data = {k:archive[k] for k in archive.files}
  if int(data['schema_version']) != 1:
    raise ValueError('Unsupported bridge paths schema')
  for key in ('starts', 'ends'):
    if data[key].ndim != 2 or data[key].shape[1] != 3 or not np.isfinite(data[key]).all():
      raise ValueError('Bridge targets require finite (N,3) values')
  if data['starts'].shape != data['ends'].shape or len(data['starts']) < 4:
    raise ValueError('Bridge paths require at least four matched target pairs')
  report = json.loads(str(data['report_json']))
  if report['reference_sha256'] != MotionReference(NOMINAL_MOTION_PATH).sha256:
    raise ValueError('Bridge path library requires regeneration for the current nominal motion')
  return data, report


class BridgeTargetPaths:
  """Load offline witnesses; sample compatible pairs without online IK."""
  def __init__(self, device='cpu', path=PATHS_PATH):
    self.device = torch.device(device)
    self.data, self.report = _load_paths(str(Path(path).resolve()))
    self.starts = torch.as_tensor(self.data['starts'], dtype=torch.float32, device=self.device)
    self.ends = torch.as_tensor(self.data['ends'], dtype=torch.float32, device=self.device)
    self.zero = torch.as_tensor(self.data['zero'], dtype=torch.float32, device=self.device)
    self.h_start_range = (float(self.data['h_start_values'].min()), float(self.data['h_start_values'].max()))

  def sample(self, num_envs, generator=None):
    if not isinstance(num_envs, int) or num_envs < 0:
      raise ValueError('num_envs must be a nonnegative integer')
    if isinstance(generator, np.random.Generator):
      index = torch.as_tensor(generator.integers(len(self.starts), size=num_envs), device=self.device)
    else:
      index = torch.randint(len(self.starts), (num_envs,), device=self.device, generator=generator)
    return self.starts[index], self.ends[index]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=PATHS_PATH)
  parser.add_argument('--report', type=Path, default=AUDIT_PATH)
  parser.add_argument('--count', type=int, default=12)
  args = parser.parse_args()
  payload, report = build_bridge_target_paths(args.count)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix='.npz', delete=False) as stream:
      temporary = Path(stream.name)
      np.savez_compressed(stream, **payload)
    os.replace(temporary, args.output)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)
  args.report.parent.mkdir(parents=True, exist_ok=True)
  args.report.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k != 'rejected_pairs'}, indent=2))


if __name__ == '__main__':
  main()
