"""Batched second-order governor for commanded joint-position trajectories."""
import math

import torch


def target_limiter_contract(velocity_limits, acceleration_limits, dt):
  if velocity_limits is None and acceleration_limits is None:
    return None
  def contract_limits(values):
    return [None if value is None else float(value) for value in values]
  return dict(algorithm='discrete_braking_governor_v1', dt=float(dt),
    velocity_limits=contract_limits(velocity_limits), acceleration_limits=contract_limits(acceleration_limits),
    history='last_applied_target_in_common_action_coordinates')


class JointTargetLimiter:
  def __init__(self, position, velocity_limits, acceleration_limits, dt):
    if position.ndim != 2 or not torch.isfinite(position).all():
      raise ValueError('Limiter position must be a finite world-by-joint tensor')
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError('Limiter dt must be finite and positive')
    self.dt = float(dt)
    limits = []
    for name, values in (('velocity', velocity_limits), ('acceleration', acceleration_limits)):
      if len(values) != position.shape[1] or any(value is not None and
          (not math.isfinite(value) or value <= 0) for value in values):
        raise ValueError(f'Limiter {name} limits must be positive finite values or None per joint')
      value = position.new_tensor([float('inf') if item is None else item for item in values])
      limits.append(value)
    self.velocity_limits, self.acceleration_limits = limits
    self._velocity_limit_contract = tuple(velocity_limits)
    self._acceleration_limit_contract = tuple(acceleration_limits)
    self.position = position.clone()
    self.velocity = torch.zeros_like(position)

  def contract(self):
    return target_limiter_contract(self._velocity_limit_contract,
      self._acceleration_limit_contract, self.dt)

  @torch.no_grad()
  def reset(self, position, velocity=None, *, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    velocity = torch.zeros_like(position) if velocity is None else velocity
    if (position.shape != self.position[ids].shape or velocity.shape != position.shape
        or not torch.isfinite(position).all() or not torch.isfinite(velocity).all()
        or (velocity.abs() > self.velocity_limits+1e-6).any()):
      raise ValueError('Invalid limiter restore position or velocity')
    self.position[ids] = position
    self.velocity[ids] = velocity

  @torch.no_grad()
  def step(self, target, *, validate=True):
    if validate and (target.shape != self.position.shape or not torch.isfinite(target).all()):
      raise ValueError('Limiter target must match state shape and be finite')
    error = target-self.position
    acceleration_bounded = torch.isfinite(self.acceleration_limits)
    dv = torch.where(acceleration_bounded, self.acceleration_limits*self.dt,
      torch.zeros_like(self.acceleration_limits))
    # This envelope reserves distance for both the next step and braking.
    # A reversal may cross the new target: acceleration bounds take priority.
    safe_acceleration = torch.where(acceleration_bounded, self.acceleration_limits,
      torch.zeros_like(self.acceleration_limits))
    speed = torch.sqrt(dv.square()+2*safe_acceleration*error.abs())-dv
    finite_desired = error.sign()*torch.minimum(speed, self.velocity_limits)
    unbounded_desired = error.sign()*torch.minimum(error.abs()/self.dt, self.velocity_limits)
    desired = torch.where(acceleration_bounded, finite_desired, unbounded_desired)
    change = (desired-self.velocity).clamp(-dv, dv)
    velocity = torch.where(acceleration_bounded, self.velocity+change, desired)
    exact = error/self.dt
    can_settle = ((exact.abs() <= self.velocity_limits) & (~acceleration_bounded
      | ((exact.abs() <= dv) & ((exact-self.velocity).abs() <= dv))))
    velocity = torch.where(can_settle, exact, velocity)
    self.velocity.copy_(velocity)
    self.position.add_(velocity*self.dt)
    return self.position
