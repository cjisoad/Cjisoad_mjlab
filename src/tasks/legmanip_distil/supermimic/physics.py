"""Command-free learned dynamics and differentiable source-Go2 task physics.

State ABI (37): root world position, root wxyz quaternion, world linear
velocity, BODY angular velocity, ABI-ordered joint positions/velocities.
Targets are actual physical joint radians [batch, 4 physics substeps, 12],
not normalized requests. Rotation residuals live in the local SO(3) tangent.
"""
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.tasks.pedipulation.constants import JOINT_NAMES, LEGS
from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor

STATE_DIM = 37
DELTA_DIM = 36
TARGET_SUBSTEPS = 4


def quat_normalize(q):
    length = q.norm(dim=-1, keepdim=True)
    identity = torch.cat((torch.ones_like(q[..., :1]), torch.zeros_like(q[..., 1:])), -1)
    return torch.where(length > 1.e-9, q / length.clamp_min(1.e-9), identity)


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def quat_rotate(q, vector):
    uv = torch.cross(q[..., 1:].expand_as(vector), vector, dim=-1)
    return vector + 2. * (q[..., :1] * uv + torch.cross(q[..., 1:].expand_as(vector), uv, dim=-1))


def rotvec_quat(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle * .5), .5 * torch.sinc(angle / (2. * math.pi)) * vector), -1)


def quat_rotvec(q):
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0., -q, q)
    length = q[..., 1:].norm(dim=-1, keepdim=True)
    scale = 2. * torch.atan2(length, q[..., :1]) / length.clamp_min(1.e-8)
    scale = torch.where(length < 1.e-8, torch.full_like(scale, 2.), scale)
    return q[..., 1:] * scale


def state_delta(state, next_state):
    """Physical 36-vector residual with shortest-arc BODY rotation delta."""
    rotation = quat_rotvec(quat_multiply(quat_conjugate(quat_normalize(state[..., 3:7])),
                                        quat_normalize(next_state[..., 3:7])))
    return torch.cat((next_state[..., :3] - state[..., :3], rotation,
                      next_state[..., 7:] - state[..., 7:]), -1)


def integrate_state(state, delta):
    """Apply physical residual; normalize quaternion without Euclidean drift."""
    rotation = quat_normalize(quat_multiply(quat_normalize(state[..., 3:7]), rotvec_quat(delta[..., 3:6])))
    return torch.cat((state[..., :3] + delta[..., :3], rotation, state[..., 7:] + delta[..., 6:]), -1)


def state_from_env(env):
    """Read current generalized coordinates, including the pre-reset terminal.

    Derived cvel/xpos are one integration behind after mj_step; the raw free
    joint has world translational and body rotational generalized velocities.
    """
    robot = env.scene['robot']
    data = robot.data
    action = env.action_manager.get_term('joint_pos')
    indexing = robot.indexing
    sim_data = data.data
    pose = sim_data.qpos[:, indexing.free_joint_q_adr]
    velocity = sim_data.qvel[:, indexing.free_joint_v_adr]
    return torch.cat((pose, velocity[:, :3], velocity[:, 3:6],
                      data.joint_pos[:, action.joint_ids], data.joint_vel[:, action.joint_ids]), -1).clone()


def state_observation(state, command, previous_action, default_angles):
    """Exact noiseless 48D deployment ABI; no hidden FSM/reference inputs."""
    q = quat_normalize(state[..., 3:7])
    down = state.new_tensor((0., 0., -1.)).expand_as(state[..., :3])
    gravity = quat_rotate(quat_conjugate(q), down)
    scale = command.new_tensor((2., 2., .25, 1., 1., 1.))
    return torch.cat((state[..., 10:13] * .25, gravity, command * scale,
                      state[..., 13:25] - default_angles, state[..., 25:37] * .05,
                      previous_action), -1).clamp(-100., 100.)


class WorldModel(nn.Module):
    """Residual MLP with supervised contact and forbidden-collision heads.

    No command/context argument exists: dynamics only depend on physical state
    and the four actually applied targets. Horizontal translation is removed
    from inputs for invariance across tiled training worlds. Fixed unit scales
    make heterogeneous state/delta dimensions comparable before data exists.
    """
    def __init__(self, hidden_dim=512, depth=3, dt=.02):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer('input_scale', torch.tensor(
            [1.] * 7 + [3.] * 3 + [5.] * 3 + [3.] * 12 + [15.] * 12))
        self.register_buffer('delta_scale', torch.tensor(
            [.04] * 3 + [.15] * 3 + [.5] * 3 + [1.] * 3 + [.15] * 12 + [3.] * 12))
        layers = []
        width = STATE_DIM + TARGET_SUBSTEPS * 12
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.ELU()))
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(width, DELTA_DIM)
        self.contact_head = nn.Linear(width, 4)
        self.collision_head = nn.Linear(width, 1)
        nn.init.normal_(self.delta_head.weight, std=.005)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, state, targets):
        if state.shape[-1] != STATE_DIM or targets.shape != (*state.shape[:-1], TARGET_SUBSTEPS, 12):
            raise ValueError('Expected state [...,37] and actual targets [...,4,12] radians')
        normalized = state / self.input_scale
        # The root has no privileged global horizontal origin.
        normalized = torch.cat((torch.zeros_like(normalized[..., :2]), normalized[..., 2:]), -1)
        # q and -q are the same physical rotation.
        quaternion = quat_normalize(normalized[..., 3:7])
        quaternion = torch.where(quaternion[..., :1] < 0., -quaternion, quaternion)
        normalized = torch.cat((normalized[..., :3], quaternion, normalized[..., 7:]), -1)
        hidden = self.trunk(torch.cat((normalized, targets.flatten(-2) / 3.), -1))
        normalized_delta = self.delta_head(hidden)
        return {'state': integrate_state(state, normalized_delta * self.delta_scale),
                'normalized_delta': normalized_delta,
                'contact_logits': self.contact_head(hidden),
                'collision_logits': self.collision_head(hidden)}

    def loss(self, state, targets, next_state, contacts, collision, valid_mask=None):
        valid = torch.ones_like(state[..., 0]) if valid_mask is None else valid_mask.to(state.dtype)
        # Exclude invalid rows before manifold operations. Multiplying a NaN
        # loss/derivative by a zero mask would still poison the optimizer.
        identity_state = torch.zeros_like(state)
        identity_state[..., 3] = 1.
        state = torch.where(valid[..., None].bool(), state, identity_state)
        targets = torch.where(valid[..., None, None].bool(), targets, torch.zeros_like(targets))
        next_state = torch.where(valid[..., None].bool(), next_state, state.detach())
        contacts = torch.where(valid[..., None].bool(), contacts, torch.zeros_like(contacts))
        collision = torch.where(valid[..., None].bool(), collision, torch.zeros_like(collision))
        predicted = self(state, targets)
        target_delta = state_delta(state, next_state) / self.delta_scale
        def masked(value):
            while value.ndim > valid.ndim:
                value = value.mean(-1)
            return (value * valid).sum() / valid.sum().clamp_min(1.)
        state_loss = masked(F.smooth_l1_loss(predicted['normalized_delta'], target_delta, reduction='none'))
        contact = masked(F.binary_cross_entropy_with_logits(predicted['contact_logits'], contacts.to(state.dtype), reduction='none'))
        collision = masked(F.binary_cross_entropy_with_logits(predicted['collision_logits'],
                                                       collision.to(state.dtype).reshape_as(predicted['collision_logits']), reduction='none'))
        return {'total': state_loss + .1 * contact + .1 * collision,
                'state': state_loss, 'contact': contact, 'collision': collision}


class Go2Kinematics(nn.Module):
    """Torch FK extracted from the actual source-URDF MuJoCo robot model.

    Uses link frame transforms (not COM frames) and preserves all fixed links.
    Pass env.sim.mj_model for exact compiled scene geometry or omit to compile
    get_stand_spec(). Namespaced ``robot/`` body/joint names are supported.
    """
    def __init__(self, model=None):
        super().__init__()
        import mujoco
        if model is None:
            from src.tasks.pedipulation.robot import get_stand_spec
            model = get_stand_spec().compile()
        def short(name):
            return name.rsplit('/', 1)[-1] if name else ''
        bodies = {short(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)): i for i in range(1, model.nbody)}
        joints = {short(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)): i for i in range(model.njnt)}
        root = bodies['base_link']
        selected = [root]
        for i in range(root + 1, model.nbody):
            if int(model.body_parentid[i]) in selected:
                selected.append(i)
        self.body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in selected)
        self.body_ids = tuple(selected)
        self.parents = tuple(-1 if i == root else selected.index(int(model.body_parentid[i])) for i in selected)
        self.joint_indices = []
        positions, quats, axes, pivots, references = [], [], [], [], []
        for i in selected:
            positions.append(model.body_pos[i].copy())
            quats.append(model.body_quat[i].copy())
            joint_list = [int(j) for j in range(model.njnt) if model.jnt_bodyid[j] == i and model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
            if len(joint_list) > 1:
                raise ValueError('Go2 FK requires at most one hinge per link')
            if joint_list:
                j = joint_list[0]
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('Go2 articulated joints must be hinges')
                name = short(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
                self.joint_indices.append(JOINT_NAMES.index(name))
                axes.append(model.jnt_axis[j].copy())
                pivots.append(model.jnt_pos[j].copy())
                references.append(float(model.qpos0[model.jnt_qposadr[j]]))
            else:
                self.joint_indices.append(-1)
                axes.append([1., 0., 0.])
                pivots.append([0., 0., 0.])
                references.append(0.)
        import numpy as np
        for key, values in (('local_pos', positions), ('local_quat', quats), ('joint_axis', axes), ('joint_pivot', pivots), ('joint_reference', references)):
            self.register_buffer(key, torch.tensor(np.asarray(values), dtype=torch.float64))
        self.register_buffer('joint_limits', torch.tensor(np.stack([model.jnt_range[joints[name]] for name in JOINT_NAMES]), dtype=torch.float32))
        self.foot_indices = tuple(selected.index(bodies[f'{leg}_foot']) for leg in LEGS)

    def forward(self, state):
        # Constants are cast to match the caller; no implicit double rollouts.
        local_pos, local_quat = self.local_pos.to(state), self.local_quat.to(state)
        axis, pivot, ref = self.joint_axis.to(state), self.joint_pivot.to(state), self.joint_reference.to(state)
        positions, rotations = [], []
        for i, parent in enumerate(self.parents):
            if parent < 0:
                positions.append(state[..., :3])
                rotations.append(quat_normalize(state[..., 3:7]))
                continue
            p = local_pos[i].expand_as(state[..., :3])
            q = local_quat[i].expand_as(state[..., 3:7])
            j = self.joint_indices[i]
            if j >= 0:
                rotation = rotvec_quat(axis[i] * (state[..., 13 + j:14 + j] - ref[i]))
                p = p + quat_rotate(q, pivot[i].expand_as(p) - quat_rotate(rotation, pivot[i].expand_as(p)))
                q = quat_multiply(q, rotation)
            positions.append(positions[parent] + quat_rotate(rotations[parent], p))
            rotations.append(quat_multiply(rotations[parent], q))
        body_pos, body_quat = torch.stack(positions, -2), torch.stack(rotations, -2)
        return {'body_pos_w': body_pos, 'body_quat_w': body_quat,
                'feet_pos_w': body_pos[..., self.foot_indices, :]}


def tracking_score(error, scale):
    """Fine precision bonus plus a non-saturating corrective tail."""
    scaled = error / torch.as_tensor(scale, device=error.device, dtype=error.dtype)
    return torch.exp(-scaled.square().sum(-1)) - .1 * F.smooth_l1_loss(
        scaled, torch.zeros_like(scaled), reduction='none').sum(-1)


class DifferentiableTaskReward(nn.Module):
    """User task reward plus mode-aware support and physical regularization.

    context: desired_upright [N] continuously moves pitch 0 -> -pi/2;
    op_weight [N] gates FR tracking exactly (locomotion is zero); optional
    target_height [N], foot_zero [N,3] (anchor FR zero), gait_phase [N],
    support_weight [N,4]. All context is training-only and never dynamics input.
    Components are unweighted scores/costs; total is their weighted sum.
    """
    def __init__(self, kinematics, dt=.02):
        super().__init__()
        self.kinematics = kinematics
        self.dt = float(dt)
        # Matches the teacher's source-FK anchor FR zero.
        from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace
        self.register_buffer('foot_zero', torch.tensor(build_foot_workspace().zero, dtype=torch.float32))

    def forward(self, state, next_state, action, previous_action, command, context,
                contact_probability=None, collision_probability=None):
        def value(name, default):
            raw = context.get(name, default) if isinstance(context, Mapping) else getattr(context, name, default)
            return torch.as_tensor(raw, device=state.device, dtype=state.dtype)
        upright = value('desired_upright', 0.).reshape(-1).expand(state.shape[0]).clamp(0., 1.)
        operation = value('op_weight', 0.).reshape(-1).expand(state.shape[0]).clamp(0., 1.)
        q = quat_normalize(next_state[:, 3:7])
        gravity_w = state.new_tensor((0., 0., -1.)).expand_as(state[:, :3])
        basis = anchor_basis(q, gravity_w)
        linear = to_anchor(basis, next_state[:, 7:10])
        angular = to_anchor(basis, quat_rotate(q, next_state[:, 10:13]))
        linear_error = linear[:, :2] - command[:, :2]
        angular_error = angular[:, 2:3] - command[:, 2:3]
        linear_tracking = tracking_score(linear_error, (.10 - .05 * upright)[:, None])
        angular_tracking = tracking_score(angular_error, (.30 - .10 * upright)[:, None])
        feet = self.kinematics(next_state)['feet_pos_w']
        previous_feet = self.kinematics(state)['feet_pos_w']
        fr = to_anchor(basis, feet[:, 1] - next_state[:, :3])
        zero = value('foot_zero', self.foot_zero)
        fr_error = fr - zero - command[:, 3:6]
        fr_tracking = operation * tracking_score(fr_error, .05)
        angle = upright * (math.pi / 2.)
        target_gravity = torch.stack((-angle.sin(), torch.zeros_like(angle), -angle.cos()), -1)
        gravity_b = quat_rotate(quat_conjugate(q), gravity_w)
        posture = torch.exp(-(gravity_b - target_gravity).square().sum(-1) / .25)
        target_height = value('target_height', .42 + .10 * upright)
        height = torch.exp(-(next_state[:, 2] - target_height).square() / .08**2)
        rate = (action - previous_action).square().mean(-1)
        limits = self.kinematics.joint_limits.to(state)
        midpoint = limits.mean(-1)
        halfspan = (limits[:, 1] - limits[:, 0]) * .45
        joint_limit = (next_state[:, 13:25].sub(midpoint).abs() - halfspan).clamp_min(0.).square().mean(-1)
        support = value('support_weight', torch.ones_like(feet[:, :, 0]))
        # FR is manipulation-only when engaged; both fronts leave support as
        # upright rises. Rear feet remain support throughout the transition.
        default_support = torch.stack((1. - upright, (1. - upright) * (1. - operation), torch.ones_like(upright), torch.ones_like(upright)), -1)
        support = support * default_support
        contact = (torch.sigmoid((.035 - feet[:, :, 2]) / .01) if contact_probability is None
                   else contact_probability.reshape(-1, 4).clamp(0., 1.))
        velocity = (feet - previous_feet) / self.dt
        slip = (velocity[:, :, :2].square().sum(-1) * contact * support).sum(-1) / support.sum(-1).clamp_min(1.)
        support_contact = (contact * support).sum(-1) / support.sum(-1).clamp_min(1.)
        phase = value('gait_phase', 0.).reshape(-1, 1)
        quad_stance = (((phase + state.new_tensor((.5, 0., 0., .5))) % 1.) < .56).to(state)
        tripod_stance = (((phase + state.new_tensor((0., 0., 2./3., 1./3.))) % 1.) < .8).to(state)
        desired_contact = torch.lerp(quad_stance, tripod_stance, operation[:, None])
        # Upright walking uses alternating rear support, independent of the
        # manipulation arm's state. Front contacts never enter this constraint.
        biped_stance = (((phase + state.new_tensor((0.,0.,0.,.5))) % 1.) < .60).to(state)
        desired_contact = torch.lerp(desired_contact, biped_stance, upright[:, None])
        gait = ((1. - (contact - desired_contact).square()) * support).sum(-1) / support.sum(-1).clamp_min(1.)
        moving = (command[:, :3].norm(dim=-1) > .05).to(state)
        gait = gait * moving + support_contact * (1. - moving)
        collision = (torch.zeros_like(upright) if collision_probability is None else collision_probability.reshape(-1).clamp(0., 1.))
        vertical_velocity = linear[:, 2].square()
        terms = dict(linear_tracking=linear_tracking, angular_tracking=angular_tracking,
                     fr_tracking=fr_tracking, posture=posture, height=height,
                     action_rate=rate, joint_limits=joint_limit, slip=slip,
                     collision=collision, gait=gait, support_contact=support_contact,
                     vertical_velocity=vertical_velocity)
        terms.update(linear_error=linear_error.norm(dim=-1),
                     angular_error=angular_error.abs().squeeze(-1),
                     foot_error=operation * fr_error.norm(dim=-1),
                     operation=operation)
        terms['velocity_objective'] = value('velocity_weight', 1.) * (2.5 * linear_tracking + 1.5 * angular_tracking)
        terms['foot_objective'] = 3. * fr_tracking
        terms['posture_objective'] = 2. * posture + .5 * height
        terms['regularization_objective'] = (.3 * gait - .05 * rate - 2. * joint_limit
                                             - .05 * slip - 2. * collision - .2 * vertical_velocity)
        terms['total'] = sum(terms[k] for k in ('velocity_objective','foot_objective',
                                               'posture_objective','regularization_objective'))
        return terms
