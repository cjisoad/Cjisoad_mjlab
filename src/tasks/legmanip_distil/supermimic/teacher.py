"""Isolated four-teacher context driven by predicted Go2 physical states.

The shadow environment only computes kinematics/sensors. Physics integration is
performed by the differentiable world model, and actor weights are shared.
"""
from copy import copy, deepcopy
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply
from .physics import state_from_env

capture_state = state_from_env


def tensor_state(obj):
    return {name: value.detach().clone() for name, value in vars(obj).items()
            if isinstance(value, torch.Tensor)}


def restore_tensors(obj, state):
    for name, value in state.items():
        target = getattr(obj, name)
        if target.shape != value.shape:
            raise ValueError(f'Context tensor shape changed: {name}')
        target.copy_(value)


class ImaginedTeacher:
    def __init__(self, teacher):
        self.live = teacher
        cfg = deepcopy(teacher.env.cfg)
        # Training callbacks only capture real integration steps.
        cfg.terminations = dict(teacher.env.cfg.terminations)
        cfg.terminations.pop('supermimic_capture', None)
        devices = [torch.device(teacher.env.device).index or 0] if 'cuda' in str(teacher.env.device) else []
        with torch.random.fork_rng(devices=devices):
            self.env = ManagerBasedRlEnv(cfg, device=teacher.env.device)
            self.env.reset()
        self.teacher = copy(teacher)
        self.teacher.env = self.env
        self.teacher.term = self.env.command_manager.get_term('twist')
        self.teacher.action = self.env.action_manager.get_term('joint_pos')
        self.teacher.limiter = copy(teacher.limiter)
        self.teacher._preview_engaged = teacher._preview_engaged.clone()
        self.saved = None

    def _objects(self, env):
        term = env.command_manager.get_term('twist')
        return [env.command_manager.get_term('intent'), term,
                env.action_manager.get_term('joint_pos'), *term.references.values()]

    @torch.no_grad()
    def seed(self):
        source = self.live.env
        self.saved = [tensor_state(obj) for obj in self._objects(source)]
        self.saved_counter = source.common_step_counter
        self.saved_episode_steps = source.episode_length_buf.clone()
        self.reset()

    @torch.no_grad()
    def reset(self):
        if self.saved is None:
            raise RuntimeError('Seed the imagined teacher before querying')
        for obj, values in zip(self._objects(self.env), self.saved):
            restore_tensors(obj, values)
        self.env.common_step_counter = self.saved_counter
        self.env.episode_length_buf.copy_(self.saved_episode_steps)
        self.env.action_manager.action.copy_(self.teacher.action.raw_action)
        self.teacher._preview_engaged.copy_(self.teacher.term.fr_engaged)

    @torch.no_grad()
    def _write_state(self, state):
        robot = self.env.scene['robot']
        root = torch.cat((state[:, :10],
                          quat_apply(state[:, 3:7], state[:, 10:13])), -1)
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(state[:, 13:25], state[:, 25:37])
        robot.clear_state()
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.env.scene.update(dt=0.)
        self.env.sim.sense()

    @torch.no_grad()
    def query(self, state):
        self._write_state(state)
        return self.teacher().detach().clone()

    @property
    def command(self):
        return self.env.command_manager.get_term('intent').command

    @torch.no_grad()
    def context(self):
        return teacher_context(self.teacher)

    @torch.no_grad()
    def set_walking_velocity(self, ids, velocity):
        # The original command setter updates a requested value, then smooths it
        # at the next tick. Hindsight labels need the same immediate command
        # that the imagined student and reward receive at this tick.
        self.command[ids, :3] = velocity
        self.teacher.term.set_command(ids, velocity=velocity)
        self.teacher.term._command[ids, :3] = velocity

    @torch.no_grad()
    def advance(self, state, executed_action):
        action = self.teacher.action
        action.previous_action.copy_(action.raw_action)
        action._raw_action.copy_(executed_action)
        action.last_request.copy_(executed_action)
        target = action.default_angles + action.cfg.scale * executed_action + action.motor_offsets
        # A constant student target has zero target velocity by the final
        # physics substep, matching DistilAction's acceleration-limiter seed.
        action.executed_velocity.zero_()
        action.executed_position.copy_(target)
        self.env.action_manager.action.copy_(executed_action)
        self.teacher.term.fr_engaged.copy_(self.teacher._preview_engaged)
        self._write_state(state)
        self.env.common_step_counter += 1
        self.env.episode_length_buf += 1
        self.env.command_manager.compute(dt=self.env.step_dt)

    def close(self):
        self.env.close()



def teacher_context(teacher):
    term = teacher.term
    upright = (term.owner == 3).float()
    target_height = torch.where(term.owner == 3, .52, term.workspace.nominal_height)
    for owner, reference in term.references.items():
        ids = term.owner == owner
        target_height[ids] = reference.anchor_pos_w[ids, 2]
        from mjlab.utils.lab_api.math import quat_apply_inverse
        gravity = torch.zeros_like(reference.anchor_pos_w)
        gravity[:, 2] = -1.
        upright[ids] = (-quat_apply_inverse(reference.anchor_quat_w, gravity)[ids, 0]).clamp(0., 1.)
    operation = term.blend.clamp(0., 1.).clone()
    intent = teacher.env.command_manager.get_term('intent')
    target = intent.command[:, 3:]
    biped_operation = ((target - intent.nominal).abs().amax(-1) > 1e-6)
    operation = torch.where(term.owner == 3, biped_operation.float(), operation)
    operation[(term.owner == 2) | (term.owner == 4)] = 0.
    return {'desired_upright': upright, 'op_weight': operation, 'target_height': target_height,
            'gait_phase': term.gait_phase.clone(), 'foot_zero': term.zero.clone(),
            'velocity_weight': torch.where((term.owner == 2) | (term.owner == 4), .2, 1.)}
