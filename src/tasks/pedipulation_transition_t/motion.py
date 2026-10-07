"""MuJoCo support-constrained reference construction and validated CPU loading.

The asset is a kinematic reference with a quasi-static inverse-dynamics audit.
Neither IK nor the static audit proves that a policy can track it dynamically.
"""
from pathlib import Path
import hashlib

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.tasks.pedipulation.constants import JOINT_NAMES, DEFAULT_ANGLES, DESIRED_ANGLES
from src.tasks.pedipulation.robot import get_stand_spec

MOTION_PATH = Path(__file__).resolve().parents[2]/'assets/motions/go2/tripod_to_biped.npz'
NOMINAL_MOTION_PATH = MOTION_PATH.with_name('tripod_to_biped_nominal.npz')
FOOT_NAMES = ('FL', 'FR', 'RL', 'RR')
SUPPORT_JOINTS = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11])
SCHEMA_VERSION = 1


def smoothstep(x):
  x = np.clip(x, 0., 1.)
  return x*x*x*(10. + x*(-15. + 6.*x))


def make_model():
  spec = get_stand_spec()
  spec.option.timestep = .005
  spec.option.iterations = 30
  spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0., 0., .1], friction=[1., .005, .0001], condim=3)
  for name in JOINT_NAMES:
    spec.joint(name).armature = .02 if 'calf' in name else .01
    spec.add_actuator(name=name.replace('_joint', '_motor'), target=name,
      trntype=mujoco.mjtTrn.mjTRN_JOINT,
      ctrllimited=True, ctrlrange=[-35.55, 35.55] if 'calf' in name else [-23.7, 23.7])
  return spec.compile()


def generate_motion(fps=50, variant='legacy'):
  """Construct planted-foot support references without changing legacy assets.

  The nominal biped variant starts with narrower rear footholds. Keeping the
  old wide footholds while rotating the trunk upright forces approximately
  +/-0.70 rad rear hips. Choosing the footholds before the transfer removes
  that incompatible constraint without prescribing any ground-foot sliding.
  Sagittal angles still satisfy foot height and COM equilibrium, so they are
  close to, rather than exactly equal to, the standing teacher's pose hint.
  """
  if not isinstance(fps, int) or fps < 10:
    raise ValueError('fps must be an integer >= 10')
  if variant not in ('legacy', 'nominal_biped'):
    raise ValueError(f'Unknown reference variant: {variant}')
  model, data = make_model(), None
  data = mujoco.MjData(model)
  qadr = np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
  vadr = np.array([int(model.joint(n).dofadr[0]) for n in JOINT_NAMES])
  jids = [model.joint(n).id for n in JOINT_NAMES]
  sites = [model.site(n).id for n in FOOT_NAMES]
  base = model.body('base_link').id
  radius = float(model.geom('FR_foot_collision').size[0])
  mass = model.body_mass.sum()
  limits = model.jnt_range[jids]
  initial = np.array(DEFAULT_ANGLES)
  # Leave clearance for dynamic height/foot errors during the crouch.
  initial[3:6] = [0., .6, -2.0]
  if variant == 'nominal_biped':
    initial[[6, 9]] = 0.
  data.qpos[:3] = [0., 0., .31]
  data.qpos[3:7] = [1., 0., 0., 0.]
  data.qpos[qadr] = initial
  mujoco.mj_forward(model, data)
  support = data.site_xpos[sites].copy()
  support[:, 2] = radius
  rear_center = support[2:].mean(axis=0)
  root, joints, feet, contact, unload, coms, forces, torques = [], [], [], [], [], [], [], []
  errors, residuals, collisions = [], [], []
  times = np.arange(6*fps + 1, dtype=float)/fps
  # Root xy + the nine support joint angles are solved together. FR is free.
  previous = np.r_[0., .05, initial[SUPPORT_JOINTS]]
  lower = np.r_[-1., -1., limits[SUPPORT_JOINTS, 0] + .025]
  upper = np.r_[1., 1., limits[SUPPORT_JOINTS, 1] - .025]
  jac = np.zeros((3, model.nv))
  for t in times:
    shift = smoothstep((t-.5)/.45)
    rise = smoothstep((t-.9)/1.6)
    pitch = -np.pi/2 * rise
    # A short crouch makes the weight transfer reachable while FL is planted.
    height = .31 - .05*shift + .24*rise
    beta = .4*(1.-shift)
    com_goal = (1.-beta)*rear_center[:2] + beta*support[0, :2]
    grounded = t <= .95 + 1e-9
    pose_hint = (1.-rise)*initial + rise*np.array(DESIRED_ANGLES)
    fr = (1.-rise)*(initial[3:6]+shift*np.array([0., -.4, 0.])) + rise*np.array(DESIRED_ANGLES[3:6])
    swing = smoothstep((t-.95)/1.55)
    fl_end = np.array([rear_center[0]+.303, .142, .5+.17782152])
    fl_goal = (1.-swing)*support[0] + swing*fl_end
    quat = np.array([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])

    def residual(x, regularize=True):
      data.qpos[:3] = [x[0], x[1], height]
      data.qpos[3:7] = quat
      q = pose_hint.copy()
      q[SUPPORT_JOINTS], q[3:6] = x[2:], fr
      data.qpos[qadr] = q
      data.qvel[:] = 0.
      mujoco.mj_forward(model, data)
      foot_error = data.site_xpos[np.array(sites)[[2, 3]]] - support[[2, 3]]
      parts = [foot_error.ravel(), data.subtree_com[base, :2]-com_goal]
      parts.append(data.site_xpos[sites[0]]-fl_goal)
      if regularize:
        parts.append(.0003*(x[2:]-pose_hint[SUPPORT_JOINTS]))
      return np.concatenate(parts)

    solution = least_squares(residual, previous, bounds=(lower, upper),
      ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
    previous = solution.x
    residual(previous)
    q = data.qpos[qadr].copy()
    pos = data.site_xpos[sites].copy()
    errors.append(float(np.max(np.abs(pos[2:]-support[2:]))))
    if grounded:
      errors[-1] = max(errors[-1], float(np.max(np.abs(pos[0]-support[0]))))
    f = np.zeros((4, 3))
    f[0, 2] = beta*mass*9.81
    f[2:, 2] = (1.-beta)*mass*9.81/2
    generalized = np.zeros(model.nv)
    for i in (0, 2, 3):
      mujoco.mj_jacSite(model, data, jac, None, sites[i])
      generalized += jac.T @ f[i]
    inverse = data.qfrc_bias - generalized
    residuals.append(float(np.linalg.norm(inverse[:6])/(mass*9.81)))
    # Only ground collisions matter here; self clearance is also checked below.
    floor = model.geom('floor').id
    for c in data.contact:
      names = [model.geom(int(g)).name for g in c.geom]
      if floor in c.geom and c.dist < -.002 and not any('_foot_collision' in n for n in names):
        collisions.append({'time':float(t), 'geoms':names, 'depth':float(c.dist)})
      elif floor not in c.geom and c.dist < -.005:
        collisions.append({'time':float(t), 'geoms':names, 'depth':float(c.dist)})
    root.append(np.r_[data.qpos[:7], np.zeros(6)])
    joints.append(q)
    feet.append(pos)
    coms.append(data.subtree_com[base].copy())
    forces.append(f)
    torques.append(inverse[vadr])
    contact.append([float(grounded), 0., 1., 1.])
    unload.append(shift)
  root, joints = np.array(root), np.array(joints)
  root[:, 7:10] = np.gradient(root[:, :3], 1./fps, axis=0)
  rotations = Rotation.from_quat(root[:, [4, 5, 6, 3]])
  root[1:-1, 10:13] = (rotations[2:]*rotations[:-2].inv()).as_rotvec()/(2./fps)
  velocities = np.gradient(joints, 1./fps, axis=0)
  still = (times <= .5) | (times >= 2.5)
  root[still, 7:] = 0.
  velocities[still] = 0.
  torque_limits = np.array([35.55 if 'calf' in n else 23.7 for n in JOINT_NAMES])
  payload = dict(schema_version=np.array(SCHEMA_VERSION), fps=np.array(fps), time=times,
    joint_names=np.array(JOINT_NAMES), root_state=root, joint_pos=joints,
    joint_vel=velocities, foot_pos_w=np.array(feet), contact=np.array(contact),
    fl_unload=np.array(unload), com_w=np.array(coms), static_contact_force=np.array(forces),
    static_torque=np.array(torques))
  report = dict(kind='support_constrained_kinematic_reference', fps=fps,
    frames=len(times), duration_s=6., foot_radius=radius,
    max_ik_error_m=max(errors), max_static_base_residual=max(residuals),
    max_static_torque_ratio=float(np.max(np.abs(payload['static_torque'])/torque_limits)),
    min_fr_clearance_m=float(np.min(payload['foot_pos_w'][:, 1, 2])-radius),
    ground_or_self_penetrations=collisions,
    dynamic_tracking_validated=False,
    note='Static audit is not evidence of a dynamically executable joint-only controller.')
  if variant == 'nominal_biped':
    payload['reference_variant'] = np.array(variant)
    report.update(reference_variant=variant,
      rear_foothold_strategy='Narrow initial stance; both rear feet remain planted throughout.',
      rear_foot_width_m=float(feet[-1][2, 1]-feet[-1][3, 1]),
      final_rear_joint_pos_rad=joints[-1, 6:].tolist(),
      final_rear_teacher_pose_error_rad=(joints[-1, 6:]-np.array(DESIRED_ANGLES[6:])).tolist(),
      max_joint_speed_rad_s=float(np.max(np.abs(velocities))))
  return payload, report


def validate_motion(motion, report):
  """Reject physically invalid candidates before replacing a training asset."""
  if not np.isfinite(motion['joint_pos']).all() or not np.isfinite(motion['root_state']).all():
    raise ValueError('Reference values must be finite')
  if report['max_ik_error_m'] > .004:
    raise ValueError('Reference support-foot IK error exceeds 4 mm')
  if report['max_static_base_residual'] > .005:
    raise ValueError('Reference static support equilibrium residual exceeds 0.5% mg')
  if report['max_static_torque_ratio'] >= 1.:
    raise ValueError('Reference static torque exceeds actuator limit')
  if report['min_fr_clearance_m'] < .02:
    raise ValueError('Reference FR ground clearance must be at least 2 cm')
  if report['ground_or_self_penetrations']:
    raise ValueError('Reference has ground/self penetration')
  model=make_model()
  limits=model.jnt_range[[model.joint(n).id for n in JOINT_NAMES]]
  q=motion['joint_pos']
  if np.any(q<limits[:,0]+.02) or np.any(q>limits[:,1]-.02):
    raise ValueError('Reference joint limit margin must be at least .02 rad')
  if 'fr_joint_bank' in motion:
    q=motion['fr_joint_bank']
    if not np.isfinite(q).all() or np.any(q<limits[3:6,0]+.02) or np.any(q>limits[3:6,1]-.02):
      raise ValueError('Reference FR bank violates finite joint limit margin')


def audit_fr_bank(motion):
  """Recompute quasi-static equilibrium, torque and collisions for every variant."""
  model,data=make_model(),None
  data=mujoco.MjData(model)
  qadr=np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
  vadr=np.array([int(model.joint(n).dofadr[0]) for n in JOINT_NAMES])
  sites=[model.site(n).id for n in FOOT_NAMES]
  jac=np.zeros((3,model.nv));mass=model.body_mass.sum()
  limits=np.array([35.55 if 'calf' in n else 23.7 for n in JOINT_NAMES])
  residual,ratio,clearance=0.,0.,float('inf')
  collisions=[]
  floor=model.geom('floor').id
  radius=float(model.geom('FR_foot_collision').size[0])
  for bank_index,bank in enumerate(motion['fr_joint_bank']):
    for i in range(len(motion['time'])):
      data.qpos[:7]=motion['root_state'][i,:7]
      data.qpos[qadr]=motion['joint_pos'][i]
      data.qpos[qadr[3:6]]=bank[i]
      data.qvel[:]=0.
      mujoco.mj_forward(model,data)
      generalized=np.zeros(model.nv)
      for foot in (0,2,3):
        mujoco.mj_jacSite(model,data,jac,None,sites[foot])
        generalized+=jac.T@motion['static_contact_force'][i,foot]
      tau=data.qfrc_bias-generalized
      residual=max(residual,float(np.linalg.norm(tau[:6])/(mass*9.81)))
      ratio=max(ratio,float(np.max(np.abs(tau[vadr])/limits)))
      clearance=min(clearance,float(data.site_xpos[sites[1],2]-radius))
      for c in data.contact:
        names=[model.geom(int(g)).name for g in c.geom]
        nonfoot_ground=(floor in c.geom and c.dist<-.002
          and not any('_foot_collision' in n for n in names))
        self_penetration=floor not in c.geom and c.dist<-.005
        fr_penetration=any(n.startswith('FR_') for n in names) and c.dist<-.002
        if nonfoot_ground or self_penetration or fr_penetration:
          collisions.append(dict(bank_index=bank_index,time=float(motion['time'][i]),
            geoms=names,depth=float(c.dist)))
  return dict(max_bank_static_base_residual=residual,max_bank_static_torque_ratio=ratio,
    min_bank_fr_clearance_m=clearance,bank_ground_or_self_penetrations=collisions)


def build_fr_bank(motion, count=25, seed=7):
  """Offline IK-checked smooth target paths around the nominal FR trajectory.

  Each path is nominal(t) plus a smooth start/end displacement, expressed in
  the existing teacher's anchor. This preserves the necessary crouch/rear-up
  motion while allowing independently varying FR positions.
  """
  if count < 1:
    raise ValueError('FR bank count must be positive')
  model, data = make_model(), None
  data = mujoco.MjData(model)
  qadr = np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
  fr_ids = [model.joint(n).id for n in JOINT_NAMES[3:6]]
  vadr = model.jnt_dofadr[fr_ids]
  limits = model.jnt_range[fr_ids]
  site, radius = model.site('FR').id, float(model.geom('FR_foot_collision').size[0])
  data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
  data.qpos[qadr] = DEFAULT_ANGLES
  mujoco.mj_forward(model, data)
  zero = data.site_xpos[site].copy()
  nominal = motion['foot_pos_w'][:, 1]-motion['root_state'][:, :3]-zero
  bank_q, bank_offset = [motion['joint_pos'][:, 3:6].copy()], [nominal]
  rng, jac = np.random.default_rng(seed), np.zeros((3, model.nv))
  blend = smoothstep((motion['time']-.5)/2.)[:, None]
  attempts = 0
  while len(bank_q) < count and attempts < count*8:
    attempts += 1
    start = rng.uniform([-.025, -.020, 0.], [.025, .020, .040])
    end = rng.uniform([-.040, -.030, -.040], [.040, .030, .040])
    goals = nominal+(1.-blend)*start+blend*end
    solution, path, valid = motion['joint_pos'][0, 3:6].copy(), [], True
    for i in range(len(motion['time'])):
      data.qpos[:7] = motion['root_state'][i, :7]
      data.qpos[qadr] = motion['joint_pos'][i]
      goal = data.qpos[:3]+zero+goals[i]

      def residual(q):
        data.qpos[qadr[3:6]] = q
        mujoco.mj_forward(model, data)
        return data.site_xpos[site]-goal

      def jacobian(q):
        residual(q)
        mujoco.mj_jacSite(model, data, jac, None, site)
        return jac[:, vadr]

      result = least_squares(residual, solution, jac=jacobian,
        bounds=(limits[:, 0]+.025, limits[:, 1]-.025), max_nfev=25,
        ftol=1e-9, xtol=1e-9, gtol=1e-9)
      solution = result.x
      residual(solution)
      if np.linalg.norm(result.fun) > .001 or data.site_xpos[site, 2] < radius+.02:
        valid = False
        break
      for c in data.contact:
        names = [model.geom(int(g)).name for g in c.geom]
        if any(n.startswith('FR_') for n in names) and c.dist < -.002:
          valid = False
          break
      if not valid:
        break
      path.append(solution.copy())
    if valid:
      bank_q.append(np.array(path))
      bank_offset.append(goals)
  if len(bank_q) != count:
    raise ValueError(f'Only {len(bank_q)}/{count} FR paths passed full-pose IK/clearance')
  return dict(fr_joint_bank=np.array(bank_q), fr_offset_bank=np.array(bank_offset), foot_zero=zero)


class MotionReference:
  """Strict NPZ contract and continuous sampling; no pickle or implicit reordering."""
  def __init__(self, path=MOTION_PATH):
    self.path = Path(path)
    self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
    with np.load(self.path, allow_pickle=False) as archive:
      self.data = {k:archive[k] for k in archive.files}
    d = self.data
    if int(d['schema_version']) != SCHEMA_VERSION:
      raise ValueError('Unsupported motion schema')
    if tuple(d['joint_names']) != JOINT_NAMES:
      raise ValueError('Motion joint order must be canonical FL/FR/RL/RR')
    n = len(d['time'])
    expected = {'root_state':(n, 13), 'joint_pos':(n, 12), 'joint_vel':(n, 12),
      'foot_pos_w':(n, 4, 3), 'contact':(n, 4), 'fl_unload':(n,)}
    for key, shape in expected.items():
      if d[key].shape != shape or not np.isfinite(d[key]).all():
        raise ValueError(f'{key} must have shape {shape} and finite values')
    if n < 2 or not np.isfinite(d['time']).all() or not np.all(np.diff(d['time']) > 0):
      raise ValueError('Motion time must be finite and strictly increasing')
    self.fps = float(d['fps'])
    if not np.isfinite(self.fps) or self.fps <= 0 or not np.allclose(np.diff(d['time']), 1./self.fps):
      raise ValueError('Motion fps must match time samples')
    if not np.allclose(np.linalg.norm(d['root_state'][:, 3:7], axis=1), 1., atol=1e-5):
      raise ValueError('Motion quaternions must be normalized')
    if np.any((d['contact'] < 0) | (d['contact'] > 1)) or np.any((d['fl_unload'] < 0) | (d['fl_unload'] > 1)):
      raise ValueError('Motion contact/unload values must be in [0, 1]')
    if 'fr_joint_bank' in d:
      k = len(d['fr_joint_bank'])
      for key, shape in {'fr_joint_bank':(k, n, 3), 'fr_offset_bank':(k, n, 3), 'foot_zero':(3,)}.items():
        if d[key].shape != shape or not np.isfinite(d[key]).all():
          raise ValueError(f'{key} must have shape {shape} and finite values')
    self.duration = float(d['time'][-1])

  def sample(self, time, speed=1.):
    time = np.asarray(time, dtype=float).reshape(-1)
    speed = np.broadcast_to(speed, time.shape)
    if not np.isfinite(time).all() or not np.isfinite(speed).all() or np.any(speed < 0):
      raise ValueError('Sample time and nonnegative speed must be finite')
    position = np.clip(time, 0., self.duration)*self.fps
    low = np.minimum(position.astype(int), len(self.data['time'])-2)
    alpha = position-low
    result = {}
    for key in ('root_state', 'joint_pos', 'joint_vel', 'foot_pos_w', 'contact', 'fl_unload'):
      array = self.data[key]
      blend = alpha.reshape((-1,)+(1,)*(array.ndim-1))
      result[key] = (1.-blend)*array[low]+blend*array[low+1]
    quat = result['root_state'][:, 3:7]
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    factor = speed*(time < self.duration)
    result['joint_vel'] *= factor[:, None]
    result['root_state'][:, 7:] *= factor[:, None]
    return result
