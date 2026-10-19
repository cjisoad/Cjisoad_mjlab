"""Direct student PD, with a training-only cache for original teacher substeps."""
from dataclasses import dataclass

import torch

from src.tasks.pedipulation.mdp.actions import (
    PedipulationPositionAction, PedipulationPositionActionCfg,
)


@dataclass(kw_only=True)
class DistilActionCfg(PedipulationPositionActionCfg):
    delay: bool = False

    def build(self, env):
        return DistilAction(self, env)


class DistilAction(PedipulationPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.target_limiter is not None or cfg.delay:
            raise ValueError('Student PD cannot depend on a teacher limiter or delay')
        self.teacher_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.teacher_targets = torch.zeros(env.cfg.decimation, self.num_envs, 12, device=self.device)
        self.teacher_requests = torch.zeros_like(self.raw_action)
        self.last_request = torch.zeros_like(self.raw_action)
        self.executed_position = self.default_angles.clone()
        self.executed_velocity = torch.zeros_like(self.raw_action)
        self.teacher_plan_ready = False

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.teacher_mask[ids] = False
        self.last_request[ids] = 0.
        self.executed_position[ids] = self.default_angles[ids]
        self.executed_velocity[ids] = 0.

    def process_actions(self, actions):
        if actions.shape != self.raw_action.shape or not torch.isfinite(actions).all():
            raise ValueError('Actions must be finite [num_envs, 12] values')
        if self.teacher_mask.any() and not self.teacher_plan_ready:
            raise RuntimeError('Teacher control requires a freshly queried substep plan')
        self.previous_action.copy_(self.raw_action)
        self.previous_joint_vel.copy_(self.joint_vel)
        self._raw_action.copy_(actions.clamp(-100., 100.))
        self.last_request.copy_(torch.where(self.teacher_mask[:, None],
                                            self.teacher_requests, self.raw_action))
        if self.teacher_plan_ready:
            # Consume exactly the draw previewed by the frozen teacher. The
            # student path has no delay; this only preserves teacher RNG timing.
            self.delay_steps.random_(0, self._env.cfg.decimation)
        self._substep = 0

    def apply_actions(self):
        target = self.default_angles + self.cfg.scale * self.raw_action + self.motor_offsets
        if self.teacher_mask.any():
            target = torch.where(self.teacher_mask[:, None], self.teacher_targets[self._substep], target)
        velocity = (target - self.executed_position) / self._env.physics_dt
        velocity[:, 3:6].clamp_(-2., 2.)
        self.executed_velocity.copy_(velocity)
        self.executed_position.copy_(target)
        self._raw_action.copy_((target - self.default_angles - self.motor_offsets) / self.cfg.scale)
        self._env.action_manager.action.copy_(self.raw_action)
        self._entity.set_joint_position_target(target, joint_ids=self.joint_ids)
        self._substep += 1
        if self._substep == self._env.cfg.decimation:
            self.teacher_plan_ready = False
