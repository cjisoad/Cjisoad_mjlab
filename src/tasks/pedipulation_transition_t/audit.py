"""Joint-only forward-dynamics audit, distinct from reference pose playback."""
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from src.tasks.pedipulation.constants import JOINT_NAMES
from .motion import make_model, FOOT_NAMES


def replay_motion(motion, duration=None, record_stride=4):
  """Constrained inverse-dynamics controller; only twelve joint torques applied.

  Contact forces are optimization variables, never injected into the simulator.
  MuJoCo must produce the forces from actual ground contact.
  """
  model, data = make_model(), None
  data = mujoco.MjData(model)
  qadr = np.array([int(model.joint(n).qposadr[0]) for n in JOINT_NAMES])
  vadr = np.array([int(model.joint(n).dofadr[0]) for n in JOINT_NAMES])
  sites = np.array([model.site(n).id for n in FOOT_NAMES])
  torque_limits = np.array([35.55 if 'calf' in n else 23.7 for n in JOINT_NAMES])
  data.qpos[:7] = motion['root_state'][0, :7]
  data.qpos[qadr] = motion['joint_pos'][0]
  data.qvel[:] = 0.
  mujoco.mj_forward(model, data)
  dt, fps = model.opt.timestep, float(motion['fps'])
  end = float(motion['time'][-1]) if duration is None else min(duration, float(motion['time'][-1]))
  if end <= 0 or record_stride < 1:
    raise ValueError('Replay duration and record stride must be positive')
  mass_matrix = np.zeros((model.nv, model.nv))
  jac, jacdot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
  errors, height_errors, fr_errors, qpos_frames, solver_failures = [], [], [], [], 0
  max_ratio, failed, held_time = 0., False, 0.
  failure_reason=None
  ground_impacts=[]
  recent_rear=np.zeros(2)
  for step in range(round(end/dt)):
    t = step*dt
    frame = min(int(t*fps), len(motion['time'])-2)
    a = t*fps-frame
    def sample(key):
      return (1.-a)*motion[key][frame]+a*motion[key][frame+1]
    root, qref, qdref = sample('root_state'), sample('joint_pos'), sample('joint_vel')
    quat = root[3:7]/np.linalg.norm(root[3:7])
    ref_rot = Rotation.from_quat(quat[[1, 2, 3, 0]])
    cur_rot = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
    ori_error = (cur_rot.inv()*ref_rot).as_rotvec()
    desired = np.zeros(model.nv)
    desired[:3] = 100.*(root[:3]-data.qpos[:3])+20.*(root[7:10]-data.qvel[:3])
    desired[3:6] = 100.*ori_error+20.*(cur_rot.inv().apply(root[10:13])-data.qvel[3:6])
    desired[vadr] = 100.*(qref-data.qpos[qadr])+20.*(qdref-data.qvel[vadr])
    desired = np.clip(desired, -200., 200.)
    support = [0, 2, 3] if sample('contact')[0] > .5 else [2, 3]
    nf, nv = 3*len(support), model.nv
    jrows, accrows = [], []
    foot_reference = sample('foot_pos_w')
    for foot in support:
      site = sites[foot]
      mujoco.mj_jacSite(model, data, jac, None, site)
      mujoco.mj_jacDot(model, data, jacdot, None, data.site_xpos[site], int(model.site_bodyid[site]))
      position = foot_reference[foot].copy()
      position[2] -= .001
      acc = 200.*(position-data.site_xpos[site])-30.*(jac @ data.qvel)-jacdot @ data.qvel
      jrows.append(jac.copy())
      accrows.append(acc)
    J = np.vstack(jrows)
    mujoco.mj_fullM(model, mass_matrix, data.qM)
    A = np.vstack((np.c_[mass_matrix[:6], -J.T[:6]], np.c_[J, np.zeros((nf, nf))]))
    b = np.r_[-data.qfrc_bias[:6], np.concatenate(accrows)]
    weights = np.r_[np.full(6, 20.), np.ones(nv-6), np.full(nf, .0002)]
    target = np.r_[desired, sample('static_contact_force')[support].ravel()]
    invw = 1./weights
    correction = np.linalg.lstsq((A*invw) @ A.T, b-A @ target, rcond=1e-10)[0]
    solution = target+invw*(A.T @ correction)
    T = np.c_[mass_matrix[vadr], -J.T[vadr]]
    offset = data.qfrc_bias[vadr]
    friction = np.zeros((len(support)*5, nv+nf))
    for i in range(len(support)):
      j, row = nv+3*i, 5*i
      friction[row, j+2] = 1.
      friction[row+1, [j, j+2]] = [-1., .8]
      friction[row+2, [j, j+2]] = [1., .8]
      friction[row+3, [j+1, j+2]] = [-1., .8]
      friction[row+4, [j+1, j+2]] = [1., .8]
    def inequalities(x):
      tau = T @ x+offset
      return np.r_[friction @ x, torque_limits-tau, torque_limits+tau]
    if np.min(inequalities(solution)) < -.001:
      B = np.vstack((friction, -T, T))
      result = minimize(lambda x: .5*np.sum(weights*(x-target)**2), solution,
        jac=lambda x: weights*(x-target), method='SLSQP',
        constraints=[{'type':'eq', 'fun':lambda x:A @ x-b, 'jac':lambda x:A},
                     {'type':'ineq', 'fun':inequalities, 'jac':lambda x:B}],
        options={'maxiter':50, 'ftol':1e-5})
      if result.success:
        solution = result.x
      else:
        solver_failures += 1
    torque = np.clip(T @ solution+offset, -torque_limits, torque_limits)
    # No mocap body, root teleportation, qfrc_applied, or xfrc_applied.
    data.ctrl[:] = torque
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    angle = np.linalg.norm((cur_rot.inv()*ref_rot).as_rotvec())
    errors.append(float(np.degrees(angle)))
    height_errors.append(float(abs(data.qpos[2]-root[2])))
    fr_errors.append(float(np.linalg.norm(data.site_xpos[sites[1]]-foot_reference[1])))
    max_ratio = max(max_ratio, float(np.max(np.abs(torque)/torque_limits)))
    contact=np.zeros(4,dtype=bool);nonfoot_impact=False
    wrench=np.zeros(6)
    for ci,c in enumerate(data.contact):
      names=[model.geom(int(g)).name for g in c.geom]
      if 'floor' not in names:continue
      mujoco.mj_contactForce(model,data,ci,wrench)
      if abs(wrench[0])>1.:
        for fi,name in enumerate(FOOT_NAMES):
          if f'{name}_foot_collision' in names:contact[fi]=True
      if abs(wrench[0])>15. and not any('_foot_collision' in name for name in names):
        nonfoot_impact=True
        ground_impacts.append(dict(time_s=float(t),geoms=names,normal_force_n=float(abs(wrench[0]))))
    recent_rear=np.where(contact[2:],0.,recent_rear+dt)
    endpoint=t>=float(motion['time'][np.flatnonzero(np.linalg.norm(motion['joint_vel'],axis=1)>1e-7)[-1]])
    good=(endpoint and angle<np.radians(15.) and height_errors[-1]<.05 and fr_errors[-1]<.05
      and not contact[:2].any() and contact[2:].any() and (recent_rear<=.15).all()
      and np.linalg.norm(data.qvel[:2])<.2 and np.linalg.norm(data.qvel[3:6])<.5 and not nonfoot_impact)
    held_time=held_time+dt if good else 0.
    if step % record_stride == 0:
      qpos_frames.append(data.qpos.copy())
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or nonfoot_impact or data.qpos[2] < .10 or angle > np.radians(75.):
      failed = True
      failure_reason=('nonfinite_state' if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
        else 'nonfoot_ground_impact' if nonfoot_impact else 'base_height' if data.qpos[2]<.1 else 'reference_orientation')
      break
  return dict(actuator_count=model.nu, no_external_base_forces=True,
    simulated_duration_s=(step+1)*dt, solver_failures=solver_failures,
    max_orientation_error_deg=max(errors), max_height_error_m=max(height_errors),
    max_fr_error_m=max(fr_errors), max_torque_ratio=max_ratio,
    endpoint_hold_seconds=held_time,
    failure_reason=failure_reason,ground_impacts=ground_impacts,
    complete_reference_replayed=bool(not failed and end >= float(motion['time'][-1])),
    dynamic_tracking_validated=bool(not failed and end >= float(motion['time'][-1]) and held_time>=1. and max(errors)<15. and max(height_errors)<.05 and max(fr_errors)<.05),
    qpos_frames=np.array(qpos_frames))
