"""Write count-weighted task metrics once per PPO iteration."""

import torch
import torch.distributed as dist

from ..mdp.metrics import LANDING_EVENTS, TRACKING_FIELDS


class ManipulationLogger:
  def __init__(self, logger, metrics, distributed=False):
    self.logger = logger
    self.metrics = metrics
    self.distributed = distributed
    self.landing_totals = torch.zeros_like(metrics.landing)

  def __getattr__(self, name):
    return getattr(self.logger, name)

  def restore_counts(self, counts):
    self.landing_totals.copy_(counts)
    # Simulation state is reset on resume, so pending landings are interrupted.
    pending = (self.landing_totals[0] - self.landing_totals[1:].sum()).clamp_min(0.)
    self.landing_totals[3] += pending

  def log(self, *, it, **kwargs):
    values = self.metrics.drain()
    if self.distributed:
      dist.all_reduce(values, op=dist.ReduceOp.SUM)
    self.landing_totals += values[:len(LANDING_EVENTS)]
    if self.logger.writer is not None:
      prefix = 'Metrics/twist/'
      totals = self.landing_totals.tolist()
      for name, count in zip(LANDING_EVENTS, totals):
        self.logger.writer.add_scalar(prefix + f'landing_{name}_total', count, it)
      self.logger.writer.add_scalar(prefix + 'landing_pending', totals[0] - sum(totals[1:]), it)
      if totals[0] > 0:
        self.logger.writer.add_scalar(prefix + 'landing_success', totals[1] / totals[0], it)
      tracking = values[len(LANDING_EVENTS):].reshape(2, len(TRACKING_FIELDS)).tolist()
      for mode, row in zip(('stationary', 'moving'), tracking):
        key = prefix + f'tripod_{mode}_'
        self.logger.writer.add_scalar(key + 'samples', row[0], it)
        if row[0] > 0:
          for field, value in zip(TRACKING_FIELDS[1:], row[1:]):
            self.logger.writer.add_scalar(key + field, value / row[0], it)
    self.logger.log(it=it, **kwargs)
