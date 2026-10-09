"""Cached nominal FR geometry and continuous keyboard boundary queries.

The signed-distance surface is a sampled approximation. Collision checks use
nominal support-leg poses; it does not certify dynamic balance or live collisions.
IK is used only by the offline generator, never by the runtime query class.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / 'outputs/fr_reachability/go2_fr_v1.npz'


def geometry_fingerprint():
  snapshot = Path(__file__).parents[1] / 'pedipulation/source_physics.json'
  return hashlib.sha256(snapshot.read_bytes()).hexdigest()


def chain_parameters():
  """Read and validate the exact Go2 X/Y/Y three-hinge chain used by this cache."""
  snapshot = json.loads((Path(__file__).parents[1] / 'pedipulation/source_physics.json').read_text())
  joints = {j['name']: j for j in snapshot['joints']}
  names = ('FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint')
  for name, axis in zip(names, ((1, 0, 0), (0, 1, 0), (0, 1, 0))):
    if not np.allclose(joints[name]['axis'], axis) or not np.allclose(joints[name]['origin']['rpy'], 0):
      raise ValueError('Analytic FR IK requires the captured Go2 X/Y/Y chain')
  hip = np.asarray(joints[names[0]]['origin']['xyz'])
  lateral = np.asarray(joints[names[1]]['origin']['xyz'])
  thigh = np.asarray(joints[names[2]]['origin']['xyz'])
  foot = np.asarray(joints['FR_foot_joint']['origin']['xyz'])
  if not (np.allclose(lateral[[0, 2]], 0) and np.allclose(thigh[:2], 0)
          and np.allclose(foot[:2], 0) and thigh[2] < 0 and foot[2] < 0):
    raise ValueError('Analytic FR IK requires lateral hip and straight thigh/calf offsets')
  limits = np.asarray([[joints[n]['limit']['lower'], joints[n]['limit']['upper']] for n in names])
  if not (-np.pi < limits[2, 0] < limits[2, 1] < 0):
    raise ValueError('Captured calf joint must use the negative knee branch')
  return hip, lateral[1], -thigh[2], -foot[2], limits


def analytic_ik(points_body, parameters, joint_margin):
  """Return both sagittal branches and validity masks for Cartesian points."""
  hip, lateral, upper, lower, limits = parameters
  p = np.asarray(points_body) - hip
  x, y, z = p.T
  sagittal_sq = y*y + z*z - lateral*lateral
  cosine = (x*x + sagittal_sq - upper*upper - lower*lower) / (2*upper*lower)
  possible = (sagittal_sq >= 0) & (cosine >= -1) & (cosine <= 1)
  knee = -np.arccos(np.clip(cosine, -1, 1))
  result = []
  for sign in (-1., 1.):
    sagittal_z = sign * np.sqrt(np.maximum(0, sagittal_sq))
    hip_angle = np.arctan2(z, y) - np.arctan2(sagittal_z, lateral)
    hip_angle = (hip_angle + np.pi) % (2*np.pi) - np.pi
    thigh_angle = (np.arctan2(-x, -sagittal_z)
                   - np.arctan2(lower*np.sin(knee), upper+lower*np.cos(knee)))
    # The source thigh limit extends past pi; retain that valid representation.
    thigh_angle = np.where(thigh_angle < limits[1, 0] + joint_margin,
                           thigh_angle + 2*np.pi, thigh_angle)
    angles = np.stack((hip_angle, thigh_angle, knee), axis=-1)
    valid = possible & ((angles >= limits[:, 0]+joint_margin)
                        & (angles <= limits[:, 1]-joint_margin)).all(-1)
    result.append((angles, valid))
  return result


class ReachabilityGrid:
  """Trilinear signed-distance lookup with first-crossing segment clipping."""
  def __init__(self, path=DEFAULT_CACHE):
    self.path = Path(path).expanduser().resolve()
    if not self.path.is_file():
      raise FileNotFoundError(f'Generate the FR boundary first: python scripts/build_teacher_reachability.py --output {self.path}')
    with np.load(self.path, allow_pickle=False) as data:
      self.metadata = json.loads(str(data['metadata']))
      self.lower = data['lower'].astype(float)
      self.spacing = float(data['spacing'])
      self.distance = data['signed_distance'].copy()
      self.zero = data['zero'].astype(float)
      self.targets = data['targets'].copy()
    if self.metadata['cache_version'] != CACHE_VERSION or self.metadata['geometry_sha256'] != geometry_fingerprint():
      raise ValueError('FR reachability cache does not match this robot; regenerate it')
    self.upper = self.lower + (np.asarray(self.distance.shape)-1)*self.spacing
    self.margin = float(self.metadata['query_margin_m'])
    self._tree = None

  def query(self, points):
    p = np.asarray(points, dtype=float)
    scalar = p.ndim == 1
    p = np.atleast_2d(p)
    if p.shape[-1] != 3:
      raise ValueError('FR offsets must have three coordinates')
    in_box = np.isfinite(p).all(-1) & (p >= self.lower).all(-1) & (p <= self.upper).all(-1)
    u = (np.nan_to_num(p)-self.lower)/self.spacing
    index = np.clip(np.floor(u).astype(int), 0, np.asarray(self.distance.shape)-2)
    fraction = np.clip(u-index, 0., 1.)
    value = np.zeros(len(p))
    for a in (0, 1):
      for b in (0, 1):
        for c in (0, 1):
          corner = np.array((a, b, c))
          weight = np.prod(np.where(corner, fraction, 1-fraction), axis=-1)
          ijk = index+corner
          value += weight*self.distance[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    value[~in_box] = -np.inf
    return float(value[0]) if scalar else value

  def contains(self, point):
    return self.query(point) >= self.margin

  def nearest_interior(self, point):
    # Used only to recover a seed after a frame change or an external target.
    if self._tree is None:
      from scipy.spatial import cKDTree
      self._tree = cKDTree(self.targets)
    _, index = self._tree.query(point)
    return self.targets[index].astype(float)

  def clip_segment(self, start, end):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if not np.isfinite(np.r_[start, end]).all():
      raise ValueError('FR target must be finite')
    if not self.contains(start):
      start = self.nearest_interior(start)
    delta = end-start
    count = max(1, int(np.ceil(np.linalg.norm(delta)/(self.spacing*.4))))
    fractions = np.linspace(0., 1., count+1)
    valid = self.query(start+fractions[:, None]*delta) >= self.margin
    invalid = np.flatnonzero(~valid)
    if not len(invalid):
      return end.copy()
    first = int(invalid[0])
    if first == 0:
      return start.copy()
    low, high = fractions[first-1], fractions[first]
    for _ in range(18):
      mid = (low+high)/2
      if self.contains(start+mid*delta):
        low = mid
      else:
        high = mid
    return start+low*delta

  def advance(self, start, end):
    """Accept interior moves exactly, with small validated sliding at a boundary."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    hit = self.clip_segment(start, end)
    remaining = end-hit
    if np.linalg.norm(remaining) < 1e-8:
      return hit
    # A local inward normal lets diagonal/tangential input follow the surface.
    # Keep this correction local; never project across a hole to another lobe.
    h = self.spacing*.2
    offsets = np.eye(3)*h
    gradient = (self.query(hit+offsets)-self.query(hit-offsets))/(2*h)
    norm = np.linalg.norm(gradient)
    if not np.isfinite(norm) or norm < 1e-8:
      return hit
    normal = gradient/norm
    tangent = remaining-min(0., np.dot(remaining, normal))*normal
    if np.linalg.norm(tangent) < 1e-8:
      return hit
    candidate = hit+tangent
    for _ in range(5):
      distance = self.query(candidate)
      if distance >= self.margin+1e-7:
        break
      gradient = (self.query(candidate+offsets)-self.query(candidate-offsets))/(2*h)
      squared = np.dot(gradient, gradient)
      if not np.isfinite(distance) or not np.isfinite(squared) or squared < 1e-8:
        return hit
      candidate += (self.margin+1e-7-distance)*gradient/squared
    # Respect the input speed and certify the sampled chord from the old target.
    length = np.linalg.norm(candidate-start)
    maximum = np.linalg.norm(end-start)
    if length > maximum and length > 0:
      candidate = start+(candidate-start)*maximum/length
    if np.linalg.norm(candidate-hit) > np.linalg.norm(remaining)*1.05:
      return hit
    return self.clip_segment(start, candidate)


class ContinuousFootWorkspace:
  """Keyboard-facing adapter from live anchor offsets to nominal body geometry."""
  mode = 'quadruped'

  def __init__(self, path=DEFAULT_CACHE, min_lift=.05):
    self.grid = ReachabilityGrid(path)
    self.min_lift = min_lift
    self.anchor_to_body = np.eye(3)
    self.ground_min_lift = -np.inf
    bounds = self.grid.metadata['usable_offset_bounds_m']
    self.boxes = (tuple((float(lo), float(hi)) for lo, hi in bounds),)

  def set_frame(self, anchor_to_body, base_height=None):
    matrix = np.asarray(anchor_to_body, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
      raise ValueError('Anchor-to-body rotation must be a finite 3x3 matrix')
    self.anchor_to_body = matrix
    if base_height is not None:
      self.ground_min_lift = self.grid.metadata['foot_radius_m']-base_height-self.grid.zero[2]

  def _to_body(self, offset):
    return self.anchor_to_body @ (self.grid.zero+np.asarray(offset))-self.grid.zero

  def _to_anchor(self, offset):
    return self.anchor_to_body.T @ (self.grid.zero+np.asarray(offset))-self.grid.zero

  def contains(self, offset):
    return (offset[2] >= max(self.min_lift, self.ground_min_lift)-1e-9
            and self.grid.contains(self._to_body(offset)))

  def clip(self, offset):
    return self.project(offset)

  def project(self, offset):
    return self.advance((0., 0., 0.), offset)

  def advance(self, origin, requested):
    requested = np.asarray(requested, dtype=float).copy()
    requested[2] = max(requested[2], self.min_lift, self.ground_min_lift)
    point = self.grid.advance(self._to_body(origin), self._to_body(requested))
    accepted = self._to_anchor(point)
    floor = max(self.min_lift, self.ground_min_lift)
    if accepted[2] < floor and origin[2] >= floor:
      # Boundary sliding must also respect the continuous height half-space.
      # Clip the accepted displacement instead of snapping onto a grid node.
      start = np.asarray(origin, dtype=float)
      fraction = (start[2]-floor)/(start[2]-accepted[2])
      accepted = start+fraction*(accepted-start)
    # A pose change can invalidate the old origin. Never return a below-floor
    # replacement seed or silently pass an invalid result to the command term.
    if accepted[2] < floor-1e-8 or not self.grid.contains(self._to_body(accepted)):
      body_targets = self.grid.targets
      anchor_targets = (body_targets+self.grid.zero) @ self.anchor_to_body-self.grid.zero
      eligible = anchor_targets[:, 2] >= floor
      if not eligible.any():
        raise RuntimeError('Current base pose has no above-ground FR keyboard targets')
      candidates = anchor_targets[eligible]
      accepted = candidates[np.argmin(np.sum((candidates-requested)**2, axis=-1))]
    return tuple(float(x) for x in accepted)


def build_cache(output=DEFAULT_CACHE, *, spacing=.005, joint_margin=.025,
                collision_margin=.005, query_margin=.005, progress=print):
  """Offline Cartesian IK/FK/collision scan, then a connected distance field."""
  import time
  import mujoco
  from scipy.ndimage import distance_transform_edt, label
  from src.tasks.pedipulation.constants import DEFAULT_ANGLES, JOINT_NAMES
  from src.tasks.pedipulation.robot import get_stand_spec

  if (not np.isfinite((spacing, joint_margin, collision_margin, query_margin)).all()
      or spacing <= 0 or joint_margin < 0 or collision_margin < 0 or query_margin < spacing/2):
    raise ValueError('Require positive resolution and margins; query margin >= half resolution')
  started = time.monotonic()
  parameters = chain_parameters()
  hip, lateral, upper_link, lower_link, _ = parameters
  model = get_stand_spec().compile()
  data = mujoco.MjData(model)
  addresses = [int(model.joint(n).qposadr[0]) for n in JOINT_NAMES]
  fr_addresses = addresses[3:6]
  data.qpos[addresses] = DEFAULT_ANGLES
  mujoco.mj_forward(model, data)
  site, base = model.site('FR').id, model.body('base_link').id
  zero = (data.site_xpos[site]-data.xpos[base]).copy()
  radius = float(model.geom('FR_foot_collision').size[0])
  moving = [i for i in range(model.ngeom) if model.geom(i).name.startswith('FR_')
            and 'collision' in model.geom(i).name and '_hip_' not in model.geom(i).name]
  obstacles = [i for i in range(model.ngeom) if 'collision' in model.geom(i).name
               and not model.geom(i).name.startswith('FR_')]
  pairs = np.array([(a, b) for a in moving for b in obstacles], dtype=int)
  pair_radii = model.geom_rbound[pairs[:, 0]]+model.geom_rbound[pairs[:, 1]]+collision_margin
  reach = np.hypot(upper_link+lower_link, lateral)
  lower = np.floor((hip-reach-zero-.02)/spacing)*spacing
  upper = np.ceil((hip+reach-zero+.02)/spacing)*spacing
  # Keyboard operation is above its shared reference plane; retain the zero
  # neighborhood for a connected, continuous entry path from idle.
  lower[2] = max(lower[2], -5*spacing)
  axes = [np.arange(lo, hi+spacing*.5, spacing) for lo, hi in zip(lower, upper)]
  shape = tuple(map(len, axes))
  valid_grid = np.zeros(shape, dtype=bool)
  witnesses = np.full((*shape, 3), np.nan, dtype=np.float32)
  max_fk_error = 0.
  checked, collision_rejected = 0, 0
  yz = np.stack(np.meshgrid(axes[1], axes[2], indexing='ij'), axis=-1).reshape(-1, 2)
  for ix, x in enumerate(axes[0]):
    offsets = np.column_stack((np.full(len(yz), x), yz))
    branches = analytic_ik(offsets+zero, parameters, joint_margin)
    indices = np.flatnonzero(branches[0][1] | branches[1][1])
    for flat in indices:
      iy, iz = np.unravel_index(flat, shape[1:])
      for angles, valid in branches:
        if not valid[flat]:
          continue
        checked += 1
        data.qpos[fr_addresses] = angles[flat]
        mujoco.mj_forward(model, data)
        error = np.linalg.norm(data.site_xpos[site]-data.xpos[base]-zero-offsets[flat])
        max_fk_error = max(max_fk_error, float(error))
        if error > 1e-7:
          raise RuntimeError(f'Analytic IK does not match MuJoCo FK: {error}m')
        delta = data.geom_xpos[pairs[:, 0]]-data.geom_xpos[pairs[:, 1]]
        nearby = np.flatnonzero(np.einsum('ij,ij->i', delta, delta) < pair_radii**2)
        collides = any(mujoco.mj_geomDistance(model, data, *map(int, pairs[j]),
                                             collision_margin+.001, None) < collision_margin
                       for j in nearby)
        if collides:
          collision_rejected += 1
          continue
        valid_grid[ix, iy, iz] = True
        witnesses[ix, iy, iz] = angles[flat]
        break
    if ix % 8 == 0 or ix == shape[0]-1:
      progress(f'FR scan {ix+1}/{shape[0]}: valid={int(valid_grid.sum())}, elapsed={time.monotonic()-started:.1f}s', flush=True)
  components, count = label(valid_grid)
  seed_index = np.rint((np.zeros(3)-lower)/spacing).astype(int)
  component = int(components[tuple(seed_index)])
  if not component:
    raise RuntimeError('Reference stance is outside the scanned collision-free region')
  connected = components == component
  disconnected = int(valid_grid.sum()-connected.sum())
  signed = (distance_transform_edt(connected, sampling=spacing)
            - distance_transform_edt(~connected, sampling=spacing)).astype(np.float32)
  # Grid samples are witnesses, not whole certified cells. Leave a small
  # inward margin before exposing the interpolated approximate surface.
  usable = connected & (signed >= query_margin+spacing*.25)
  ijk = np.argwhere(usable)
  targets = (lower+ijk*spacing).astype(np.float32)
  joints = witnesses[usable]
  operation = targets[:, 2] >= .05
  if not operation.any():
    raise RuntimeError('No usable lifted FR targets')
  bounds = np.column_stack((targets[operation].min(0), targets[operation].max(0)))
  metadata = dict(cache_version=CACHE_VERSION, geometry_sha256=geometry_fingerprint(),
    mujoco_version=mujoco.__version__, spacing_m=spacing, joint_margin_rad=joint_margin,
    collision_margin_m=collision_margin, query_margin_m=query_margin,
    foot_radius_m=radius, grid_shape=shape, checked_ik_witnesses=checked,
    collision_rejected=collision_rejected, components=count,
    disconnected_nodes_removed=disconnected, connected_nodes=int(connected.sum()),
    usable_targets=len(targets), usable_offset_bounds_m=bounds.tolist(),
    max_fk_residual_m=max_fk_error, build_seconds=time.monotonic()-started,
    collision_obstacles=[model.geom(i).name for i in obstacles],
    limitation='Sampled nominal body-frame geometry; fixed support-leg poses; no dynamic balance guarantee or arbitrary-pose collision certification.')
  output = Path(output).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix('.tmp.npz')
  np.savez_compressed(temporary, lower=lower, spacing=spacing, zero=zero,
    signed_distance=signed, targets=targets, joint_positions=joints,
    metadata=json.dumps(metadata, sort_keys=True))
  temporary.replace(output)
  output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
  progress(f'Saved {output}: {len(targets)} usable targets, {metadata["build_seconds"]:.1f}s', flush=True)
  return metadata
