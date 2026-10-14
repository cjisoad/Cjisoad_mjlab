"""Smooth terminal position reward, allowing in-place balance steps."""
import torch

from .standing import (advance_standing_hold,
  endpoint_metrics, standing_conditions)
from . import terminations

# Keep shaping width independent of the standing acceptance position tolerance.
POSITION_REWARD_STD = .08


class SustainedStandingReward:
  """Per-world good-time credit; manager resets own the episode boundary."""
  def __init__(self, cfg, env):
    del cfg
    self.duration = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    self.bad_samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last_step = torch.full_like(self.bad_samples, -1)
    self.value = torch.zeros(env.num_envs, device=env.device)
    self._env = env

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.duration[ids] = 0.
    self.bad_samples[ids] = 0
    self.value[ids] = 0.
    self.last_step[ids] = self._env.common_step_counter

  def __call__(self, env):
    active = self.last_step != env.common_step_counter
    if not bool(active.any()):
      return self.value
    motion = env.command_manager.get_term('motion')
    metrics = endpoint_metrics(env)
    finite = torch.stack([torch.isfinite(v) for v in metrics.values()], -1).all(-1)
    completed = motion.time_steps >= motion.motion.time_step_total-1
    eligible = (completed & finite & ~env.termination_manager.terminated
      & ~terminations.physical_failure(env))
    good = eligible & torch.stack(tuple(standing_conditions(metrics).values()), -1).all(-1)
    duration, bad = advance_standing_hold(self.duration, self.bad_samples, good,
      eligible=eligible, active=active, dt=env.step_dt)
    self.duration.copy_(duration)
    self.bad_samples.copy_(bad)
    upright = eligible & (metrics['orientation_error'] < .25) & (metrics['height_error'] < .05)
    # No contact gate or timer ramp: balancing steps keep earning credit.
    raw = torch.where(upright, torch.exp(-(metrics['xy_error']/POSITION_REWARD_STD).square()), 0.)
    self.value[active] = raw[active].to(self.value.dtype)
    self.last_step[active] = env.common_step_counter
    return self.value
