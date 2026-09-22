"""Source PD position offset action with a substep switch delay."""

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from src.tasks.pedipulation.constants import JOINT_NAMES, DESIRED_ANGLES


@dataclass(kw_only=True)
class PedipulationPositionActionCfg(ActionTermCfg):
  scale: float = .25
  delay: bool = True

  def build(self, env):
    return PedipulationPositionAction(self, env)


class PedipulationPositionAction(ActionTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    ids, names = self._entity.find_joints(JOINT_NAMES, preserve_order=True)
    if tuple(names) != JOINT_NAMES:
      raise ValueError(f'Unexpected Go2 joint order: {names}')
    self.joint_ids = torch.tensor(ids, device=self.device)
    self.default_angles = self._entity.data.default_joint_pos[:, self.joint_ids].clone()
    self.desired_angles = torch.tensor(DESIRED_ANGLES, device=self.device)
    self._raw_action = torch.zeros(self.num_envs, 12, device=self.device)
    self.previous_action = torch.zeros_like(self._raw_action)
    self.previous_joint_vel = torch.zeros_like(self._raw_action)
    self.motor_offsets = torch.zeros_like(self._raw_action)
    self.kp_multipliers = torch.ones_like(self._raw_action)
    self.kd_multipliers = torch.ones_like(self._raw_action)
    self.dr_observation = torch.zeros(self.num_envs, 34, device=self.device)
    self.actor_sample = torch.zeros(self.num_envs, 48, device=self.device)
    self.height_score = torch.tensor(0., device=self.device)
    self.delay_steps = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
    self._substep = 0

  @property
  def action_dim(self):
    return 12

  @property
  def raw_action(self):
    return self._raw_action

  @property
  def joint_pos(self):
    return self._entity.data.joint_pos[:, self.joint_ids]

  @property
  def joint_vel(self):
    return self._entity.data.joint_vel[:, self.joint_ids]

  def process_actions(self, actions):
    self.previous_action.copy_(self._raw_action)
    self.previous_joint_vel.copy_(self.joint_vel)
    self._raw_action.copy_(actions.clamp(-100., 100.))
    self.delay_steps.random_(0, self._env.cfg.decimation)
    self._substep = 0

  def apply_actions(self):
    if self.cfg.delay:
      action = torch.where(self._substep >= self.delay_steps, self._raw_action, self.previous_action)
    else:
      action = self._raw_action
    target = self.default_angles + self.cfg.scale * action + self.motor_offsets
    self._entity.set_joint_position_target(target, joint_ids=self.joint_ids)
    self._substep += 1

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    # Source retains the terminal action in the next observation and delay buffer.
    self.previous_action[ids] = 0.
    self.previous_joint_vel[ids] = 0.
    self.delay_steps[ids] = 0
