"""Task-local six-group logging and persistent landing/switch totals."""
import torch
import torch.distributed as dist
from src.tasks.loco_pedipulation.mdp.metrics import LANDING_EVENTS
from ..mdp.metrics import COURSE_FIELDS, GROUPS, SWITCH_EVENTS

class CourseLogger:
  def __init__(self, logger, metrics, distributed=False, iteration_callback=None):
    self.logger, self.metrics, self.distributed = logger, metrics, distributed
    self.landing_totals = torch.zeros_like(metrics.landing)
    self.switch_totals = torch.zeros_like(metrics.switches)
    self.iteration_callback = iteration_callback

  def __getattr__(self, name):
    return getattr(self.logger, name)

  def restore_counts(self, counts, switches=None):
    self.landing_totals.copy_(counts)
    self.landing_totals[3] += (self.landing_totals[0] - self.landing_totals[1:].sum()).clamp_min(0)
    if switches is not None:
      self.switch_totals.copy_(switches)
      self.switch_totals[:, 3] += (self.switch_totals[:, 0] - self.switch_totals[:, 1:].sum(-1)).clamp_min(0)

  def log(self, *, it, **kwargs):
    if self.iteration_callback is not None:
      self.iteration_callback(it)
    values = self.metrics.drain()
    if self.distributed:
      dist.all_reduce(values, op=dist.ReduceOp.SUM)
    self.landing_totals += values[:4]
    switch_end = 4 + self.switch_totals.numel()
    self.switch_totals += values[4:switch_end].reshape_as(self.switch_totals)
    writer = self.logger.writer
    if writer is not None:
      prefix = 'Metrics/twist/'
      for name, count in zip(LANDING_EVENTS, self.landing_totals.tolist()):
        writer.add_scalar(prefix + 'landing_' + name + '_total', count, it)
      pending = self.landing_totals[0] - self.landing_totals[1:].sum()
      writer.add_scalar(prefix + 'landing_pending', float(pending), it)
      if self.landing_totals[0] > 0:
        writer.add_scalar(prefix + 'landing_success', float(self.landing_totals[1] / self.landing_totals[0]), it)
      for direction, row in zip(('rise', 'descent'), self.switch_totals.tolist()):
        for name, count in zip(SWITCH_EVENTS, row):
          writer.add_scalar(prefix + direction + '_' + name + '_total', count, it)
        writer.add_scalar(prefix + direction + '_pending', row[0] - sum(row[1:]), it)
        if row[0] > 0:
          writer.add_scalar(prefix + direction + '_success', row[1] / row[0], it)
      rows = values[switch_end:].reshape(len(GROUPS), len(COURSE_FIELDS)).tolist()
      for group, row in zip(GROUPS, rows):
        writer.add_scalar(prefix + group + '_samples', row[0], it)
        if row[0] > 0:
          for field, value in zip(COURSE_FIELDS[1:], row[1:]):
            writer.add_scalar(prefix + group + '_' + field, value / row[0], it)
    self.logger.log(it=it, **kwargs)
