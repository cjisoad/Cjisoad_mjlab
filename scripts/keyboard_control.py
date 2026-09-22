"""Simulator-independent held-key commands for Pedipulation playback."""
from dataclasses import dataclass, field
import math


@dataclass
class KeyboardController:
  min_z_offset: float = -.10
  linear_acceleration: float = 1.
  yaw_acceleration: float = 2.
  target_rate: float = .04
  _values: list[float] = field(default_factory=lambda: [0.] * 6, init=False)

  @property
  def command(self):
    return tuple(self._values)

  def reset(self):
    self._values[:] = [0.] * 6

  def update(self, keys, dt, *, focused=True, paused=False):
    if not math.isfinite(dt) or dt < 0:
      raise ValueError('Input timestep must be finite and nonnegative')
    # Ignore a long render stall instead of jumping to a distant target.
    dt = min(dt, .1)
    if not focused or paused:
      self._values[:3] = [0., 0., 0.]
      return

    def direction(positive, negative):
      return int(positive in keys) - int(negative in keys)

    for index, sign, rate, low, high in (
        (0, direction('w', 's'), self.linear_acceleration, -.2, .6),
        (2, direction('a', 'd'), self.yaw_acceleration, -1., 1.)):
      previous = self._values[index]
      if sign == 0 or previous * sign < 0:
        previous = 0.
      self._values[index] = max(low, min(high, previous + sign * rate * dt)) if sign else 0.

    if 'r' in keys:
      self._values[3:] = [0., 0., 0.]
      return
    for index, sign, low, high in (
        (3, direction('right', 'left'), -.10, .10),
        (4, direction('up', 'down'), -.08, .08),
        (5, direction('q', 'e'), max(-.10, self.min_z_offset), .10)):
      value = max(low, min(high, self._values[index] + sign * self.target_rate * dt))
      self._values[index] = 0. if abs(value) < 1e-9 else value
