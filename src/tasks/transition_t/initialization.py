"""Coupled base/foot reset perturbations with batched, bounded Go2 IK.

Kinematics run on the tensor device. Accepted candidates receive a final CPU
MuJoCo collision check using a private MjData, never the live simulator state.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import torch


def _rotation(quaternion):
  q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
  w, x, y, z = q.unbind(-1)
  return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                      2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                      2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def _multiply(a, b):
  aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
  return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                    aw*bv+bw*av+torch.linalg.cross(av, bv)), -1)


class ResetSampler:
  """Sample physically realizable perturbations without altering the reference.

  Joint columns must use FL, FR, RL, RR order, with hip/thigh/calf per leg.
  ``pose_range`` and ``velocity_range`` use x/y/z/roll/pitch/yaw keys and
  (lower, upper) values. Velocities are additive world-frame perturbations.
  Root and foot positions use the model's world coordinates. In a vectorized
  environment, pass local positions with the environment origins removed;
  the caller restores origins when writing the returned root states.
  ``scale`` is zero for unchanged/fallback samples. ``disturbed`` records
  successfully applied perturbations, rather than attempted perturbations.
  """

  def __init__(self, model, joint_names, device="cpu", soft_limit_factor=.98):
    if not 0. < soft_limit_factor <= 1.:
      raise ValueError("soft_limit_factor must be in (0, 1]")
    self.model = model
    self.device = torch.device(device)
    self.data = mujoco.MjData(model)

    def resolve(kind, name):
      count = {mujoco.mjtObj.mjOBJ_BODY: model.nbody,
               mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
               mujoco.mjtObj.mjOBJ_SITE: model.nsite}[kind]
      found = [i for i in range(count)
               if (mujoco.mj_id2name(model, kind, i) or "").split("/")[-1] == name.split("/")[-1]]
      if len(found) != 1:
        raise ValueError(f"Expected exactly one {name}, found {len(found)}")
      return found[0]

    canonical = tuple(f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR")
                      for part in ("hip", "thigh", "calf"))
    if tuple(name.split("/")[-1] for name in joint_names) != canonical:
      raise ValueError("joint_names must use FL/FR/RL/RR hip/thigh/calf order")
    self.joint_ids = [resolve(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
    limits = model.jnt_range[self.joint_ids]
    mid, half = limits.mean(axis=1), (limits[:, 1]-limits[:, 0])*.5*soft_limit_factor
    self.lower = torch.as_tensor(mid-half, device=self.device, dtype=torch.float32)
    self.upper = torch.as_tensor(mid+half, device=self.device, dtype=torch.float32)
    root_body = resolve(mujoco.mjtObj.mjOBJ_BODY, "base_link")
    root_joint = model.body_jntadr[root_body]
    if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError("base_link must have a free joint")
    self.root_qpos = int(model.jnt_qposadr[root_joint])
    self.site_ids = [resolve(mujoco.mjtObj.mjOBJ_SITE, leg) for leg in ("FL", "FR", "RL", "RR")]
    self.foot_bodies = set(int(model.site_bodyid[i]) for i in self.site_ids)
    self.robot_bodies = {root_body}
    for body in range(root_body+1, model.nbody):
      if int(model.body_parentid[body]) in self.robot_bodies:
        self.robot_bodies.add(body)
    self.paths = []
    for site in self.site_ids:
      path, body = [], int(model.site_bodyid[site])
      while body != root_body:
        if body == 0:
          raise ValueError("Foot sites must descend from base_link")
        path.append(body)
        body = int(model.body_parentid[body])
      self.paths.append(path[::-1])
    self.stages = []
    for depth in range(max(map(len, self.paths))):
      positions, quaternions, axes, anchors, indices, references = [], [], [], [], [], []
      for leg, path in enumerate(self.paths):
        body = path[depth] if depth < len(path) else None
        positions.append(model.body_pos[body] if body is not None else np.zeros(3))
        quaternions.append(model.body_quat[body] if body is not None else np.array([1., 0., 0., 0.]))
        joint = int(model.body_jntadr[body]) if body is not None else -1
        if joint >= 0:
          if model.body_jntnum[body] != 1 or joint not in self.joint_ids:
            raise ValueError("Expected one canonical hinge per moving leg body")
          index = self.joint_ids.index(joint)
          if index//3 != leg:
            raise ValueError("Joint belongs to a different foot chain")
          indices.append(index)
          axes.append(model.jnt_axis[joint])
          anchors.append(model.jnt_pos[joint])
          references.append(model.qpos0[model.jnt_qposadr[joint]])
        else:
          indices.append(-1)
          axes.append(np.zeros(3))
          anchors.append(np.zeros(3))
          references.append(0.)
      tensor = lambda values: torch.as_tensor(np.asarray(values), dtype=torch.float32, device=self.device)
      self.stages.append((tensor(positions), _rotation(tensor(quaternions)), tensor(axes),
                          tensor(anchors), indices, tensor(references)))
    self.site_positions = torch.as_tensor(model.site_pos[self.site_ids], dtype=torch.float32, device=self.device)
    mujoco.mj_forward(model, self.data)
    # A static horizontal plane is floor; other world-attached shapes can be
    # obstacles and must never inherit the supporting-foot penetration margin.
    self.floor_geoms = {
      i for i in range(model.ngeom)
      if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
      and model.body_weldid[model.geom_bodyid[i]] == 0
      and (model.geom_contype[i] or model.geom_conaffinity[i])
      and np.allclose(self.data.geom_xmat[i].reshape(3, 3)[:, 2], [0., 0., 1.], atol=1e-6)
    }
    self.floor_height = max((float(self.data.geom_xpos[i, 2]) for i in self.floor_geoms), default=0.)
    self.foot_radii = torch.tensor([
      max((float(model.geom_size[g, 0]) for g in range(model.ngeom)
           if model.geom_bodyid[g] == model.site_bodyid[s]
           and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
           and (model.geom_contype[g] or model.geom_conaffinity[g])), default=.022)
      for s in self.site_ids], device=self.device)

  def _kinematics(self, root, joints, jacobian=False):
    batch = len(root)
    pos = root[:, None, :3].expand(-1, 4, -1)
    rotation = _rotation(root[:, 3:7])[:, None].expand(-1, 4, -1, -1)
    origins, axes_world = {}, {}
    for local_pos, local_rotation, axis, anchor, indices, reference in self.stages:
      pos = pos + (rotation @ local_pos.to(root)[None, :, :, None]).squeeze(-1)
      rotation = rotation @ local_rotation.to(root)[None]
      if max(indices) < 0:
        continue
      axis = axis.to(root)
      anchor = anchor.to(root)
      world_axis = (rotation @ axis[None, :, :, None]).squeeze(-1)
      origin = pos + (rotation @ anchor[None, :, :, None]).squeeze(-1)
      angle = joints[:, [max(i, 0) for i in indices]] - reference.to(root)
      angle = angle * torch.tensor([i >= 0 for i in indices], device=root.device)
      quaternion = torch.cat((torch.cos(angle/2)[..., None],
                              torch.sin(angle/2)[..., None]*axis[None]), -1)
      rotation = rotation @ _rotation(quaternion)
      pos = origin - (rotation @ anchor[None, :, :, None]).squeeze(-1)
      if jacobian:
        for leg, index in enumerate(indices):
          if index >= 0:
            origins[index] = origin[:, leg]
            axes_world[index] = world_axis[:, leg]
    feet = pos + (rotation @ self.site_positions.to(root)[None, :, :, None]).squeeze(-1)
    if not jacobian:
      return feet
    columns = [torch.linalg.cross(axes_world[i], feet[:, i//3]-origins[i]) for i in range(12)]
    return feet, torch.stack(columns, -1).reshape(batch, 3, 4, 3).permute(0, 2, 1, 3)

  def forward(self, root_state, joint_pos):
    """World coordinates of FL, FR, RL, RR foot sites."""
    return self._kinematics(root_state, joint_pos)

  def _solve(self, root, joints, targets):
    q = joints.clone()
    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    for _ in range(32):
      feet, jacobian = self._kinematics(root, q, jacobian=True)
      error = targets-feet
      if bool((error.norm(dim=-1) < .0003).all()):
        break
      transpose = jacobian.transpose(-1, -2)
      update = transpose @ torch.linalg.solve(jacobian @ transpose + .004**2*identity,
                                               error[..., None])
      q = (q + update.reshape(-1, 12).clamp(-.15, .15)).clamp(self.lower.to(q), self.upper.to(q))
    feet = self.forward(root, q)
    valid = ((feet-targets).norm(dim=-1) <= .002).all(-1)
    valid &= (q >= self.lower.to(q)).all(-1) & (q <= self.upper.to(q)).all(-1)
    return q, feet, valid

  def _collision_free(self, root, joints, contacts):
    self.data.qpos[:] = self.model.qpos0
    self.data.qpos[self.root_qpos:self.root_qpos+7] = root[:7]
    self.data.qpos[self.qpos_addresses] = joints
    mujoco.mj_forward(self.model, self.data)
    support = {int(self.model.site_bodyid[self.site_ids[i]]) for i in range(4) if contacts[i]}
    for contact in self.data.contact:
      if contact.dist >= -1e-6:
        continue
      b1, b2 = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
      if b1 not in self.robot_bodies and b2 not in self.robot_bodies:
        continue
      foot_ground = ((b1 in support and contact.geom2 in self.floor_geoms)
                     or (b2 in support and contact.geom1 in self.floor_geoms))
      if foot_ground and contact.dist >= -.002:
        continue
      return False
    return True

  @torch.no_grad()
  def sample(self, root_state, joint_pos, feet_world, contacts, pose_range, velocity_range,
             foot_range=(.015, .015, .01), undisturbed_probability=.25):
    batch = len(root_state)
    for value, shape in ((root_state, (batch, 13)), (joint_pos, (batch, 12)),
                         (feet_world, (batch, 4, 3)), (contacts, (batch, 4))):
      if value.shape != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Expected finite tensor of shape {shape}")
    if not bool((root_state[:, 3:7].norm(dim=-1) > 1e-8).all()):
      raise ValueError("Root quaternion must have nonzero norm")
    if not 0. <= undisturbed_probability <= 1.:
      raise ValueError("undisturbed_probability must be in [0, 1]")
    if len(foot_range) != 3 or any(not math.isfinite(x) or x < 0 for x in foot_range):
      raise ValueError("foot_range must contain three finite nonnegative values")
    keys = ("x", "y", "z", "roll", "pitch", "yaw")

    def draw(ranges):
      if set(ranges)-set(keys):
        raise ValueError("Range keys must be x/y/z/roll/pitch/yaw")
      values = [ranges.get(key, (0., 0.)) for key in keys]
      if any(len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] > v[1] for v in values):
        raise ValueError("Each range must be a finite ordered pair")
      limits = root_state.new_tensor(values)
      return limits[:, 0] + torch.rand(batch, 6, device=root_state.device, dtype=root_state.dtype)*(limits[:, 1]-limits[:, 0])

    pose, velocity = draw(pose_range), draw(velocity_range)
    offsets = (torch.rand_like(feet_world)*2-1)*feet_world.new_tensor(foot_range)
    heading = torch.atan2(_rotation(root_state[:, 3:7])[:, 1, 0], _rotation(root_state[:, 3:7])[:, 0, 0])
    c, s = heading.cos()[:, None], heading.sin()[:, None]
    x, y = offsets[:, :, 0].clone(), offsets[:, :, 1].clone()
    offsets[:, :, 0], offsets[:, :, 1] = c*x-s*y, s*x+c*y
    offsets[:, :, 2] = torch.where(contacts.bool(), 0., offsets[:, :, 2])
    requested = (torch.rand(batch, device=root_state.device) >= undisturbed_probability)
    requested &= (pose.abs().sum(-1)+velocity.abs().sum(-1)+offsets.abs().sum((1, 2))) > 0
    result = dict(root_state=root_state.clone(), joint_pos=joint_pos.clone(),
                  foot_targets=feet_world.clone(), foot_actual=self.forward(root_state, joint_pos),
                  fallback=torch.zeros(batch, dtype=torch.bool, device=root_state.device),
                  scale=root_state.new_zeros(batch), disturbed=torch.zeros_like(requested))
    pending = requested.clone()
    for scale in (1., .5, .25):
      ids = pending.nonzero().flatten()
      if not len(ids):
        break
      root = root_state[ids].clone()
      root[:, :3] += pose[ids, :3]*scale
      euler = pose[ids, 3:]*scale/2
      cr, cp, cy = euler.cos().unbind(-1)
      sr, sp, sy = euler.sin().unbind(-1)
      delta = torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                           cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)
      root[:, 3:7] = _multiply(delta, root[:, 3:7])
      root[:, 7:] += velocity[ids]*scale
      target = feet_world[ids]+offsets[ids]*scale
      target[:, :, 2] = torch.where(contacts[ids].bool(), feet_world[ids, :, 2],
                                    target[:, :, 2].clamp_min(self.floor_height+self.foot_radii.to(target)))
      q, feet, valid = self._solve(root, joint_pos[ids], target)
      root_cpu, q_cpu, contact_cpu = root.cpu().numpy(), q.cpu().numpy(), contacts[ids].cpu().numpy()
      for row in valid.nonzero().flatten().tolist():
        valid[row] = self._collision_free(root_cpu[row], q_cpu[row], contact_cpu[row])
      accepted = ids[valid]
      result["root_state"][accepted] = root[valid]
      result["joint_pos"][accepted] = q[valid]
      result["foot_targets"][accepted] = target[valid]
      result["foot_actual"][accepted] = feet[valid]
      result["scale"][accepted] = scale
      result["disturbed"][accepted] = True
      pending[accepted] = False
    result["fallback"] = pending
    return result
