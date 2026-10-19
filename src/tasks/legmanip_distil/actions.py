"""The same direct position-target hold for teacher and student control."""
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
            raise ValueError('Distillation PD cannot use a target limiter or action delay')

    def process_actions(self, actions):
        if actions.shape != self.raw_action.shape or not torch.isfinite(actions).all():
            raise ValueError('Actions must be finite [num_envs, 12] values')
        self.previous_action.copy_(self.raw_action)
        self.previous_joint_vel.copy_(self.joint_vel)
        self._raw_action.copy_(actions.clamp(-100., 100.))
        # Motion teachers read manager.action as their previous applied action.
        self._env.action_manager.action.copy_(self.raw_action)
        self._substep = 0

    def apply_actions(self):
        target = self.default_angles + self.cfg.scale * self.raw_action + self.motor_offsets
        self._entity.set_joint_position_target(target, joint_ids=self.joint_ids)
        self._substep += 1
