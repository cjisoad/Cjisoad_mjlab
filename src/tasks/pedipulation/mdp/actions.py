"""Source PD position offset action with a substep switch delay."""

from dataclasses import dataclass
import math

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from src.tasks.pedipulation.constants import JOINT_NAMES, DESIRED_ANGLES
from .target_limiter import JointTargetLimiter


@dataclass(kw_only=True)
class PedipulationPositionActionCfg(ActionTermCfg):
  scale: float = .25
  delay: bool = True
  target_velocity_limits: tuple[float | None, ...] | None = None
  target_acceleration_limits: tuple[float | None, ...] | None = None

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
    self.target_limiter = None
    if (cfg.target_velocity_limits is None) != (cfg.target_acceleration_limits is None):
      raise ValueError('Configure both target velocity and acceleration limits, or neither')
    if cfg.target_velocity_limits is not None:
      if not math.isfinite(cfg.scale) or cfg.scale <= 0:
        raise ValueError('Limited position action requires positive finite scale')
      self.target_limiter = JointTargetLimiter(self.joint_pos,
        cfg.target_velocity_limits, cfg.target_acceleration_limits, env.physics_dt)
      self._requested_action = torch.zeros_like(self._raw_action)
      self._previous_requested_action = torch.zeros_like(self._raw_action)
      self._limiter_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

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
    if self.target_limiter is None:
      self._raw_action.copy_(actions.clamp(-100., 100.))
    else:
      if not torch.isfinite(actions).all():
        raise ValueError('Limited position action must be finite')
      self._previous_requested_action.copy_(self._requested_action)
      self._requested_action.copy_(actions.clamp(-100., 100.))
      ids = self._limiter_pending.nonzero().flatten()
      if len(ids):
        # Commands may write the final reset pose after the action manager reset.
        self.restore_target_limiter(self.joint_pos[ids], env_ids=ids)
    self.delay_steps.random_(0, self._env.cfg.decimation)
    self._substep = 0

  def _selected_action(self):
    current = self._raw_action if self.target_limiter is None else self._requested_action
    if self.cfg.delay:
      previous = self.previous_action if self.target_limiter is None else self._previous_requested_action
      return torch.where(self._substep >= self.delay_steps, current, previous)
    return current

  def _write_target(self, action):
    target = self.default_angles + self.cfg.scale * action + self.motor_offsets
    if self.target_limiter is not None:
      target = self.target_limiter.step(target, validate=False)
      self._raw_action.copy_((target-self.default_angles-self.motor_offsets)/self.cfg.scale)
      # The next observation and manager history must describe the target that
      # reached the actuator, rather than the unrestricted policy request.
      manager = self._env.action_manager
      if manager.action.shape != self._raw_action.shape:
        raise RuntimeError('Limited common position action requires a single 12-joint action term')
      manager.action.copy_(self._raw_action)
    self._entity.set_joint_position_target(target, joint_ids=self.joint_ids)

  def apply_actions(self):
    self._write_target(self._selected_action())
    self._substep += 1

  def restore_target_limiter(self, position, velocity=None, *, env_ids=None):
    if self.target_limiter is None:
      raise ValueError('Target limiter is disabled')
    self.target_limiter.reset(position, velocity, env_ids=env_ids)
    self._limiter_pending[slice(None) if env_ids is None else env_ids] = False

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    # Start the new episode with zero action history in observations and delay.
    self._raw_action[ids] = 0.
    self.previous_action[ids] = 0.
    self.previous_joint_vel[ids] = 0.
    self.delay_steps[ids] = 0
    if self.target_limiter is not None:
      self._requested_action[ids] = 0.
      self._previous_requested_action[ids] = 0.
      self._limiter_pending[ids] = True
