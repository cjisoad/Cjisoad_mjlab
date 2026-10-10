"""Small terminal shaping reward for sustained standing after the descent."""
import torch

from .standing import (STANDING_HOLD_SECONDS, advance_standing_hold,
  tripod_endpoint_metrics as endpoint_metrics, tripod_conditions as standing_conditions)
from src.tasks.transition_t import terminations


class SustainedTripodReward:
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
    raw = torch.where(good, .2+.8*(duration/STANDING_HOLD_SECONDS).clamp(max=1.), 0.)
    self.value[active] = raw[active].to(self.value.dtype)
    self.last_step[active] = env.common_step_counter
    return self.value
