"""Keyboard commands without a graphics context or simulator."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class KeyboardControlTests(unittest.TestCase):
  def controller(self, **kwargs):
    path = Path(__file__).resolve().parents[1] / 'scripts/keyboard_control.py'
    self.assertTrue(path.exists(), 'Keyboard controller is not implemented')
    from keyboard_control import KeyboardController
    return KeyboardController(**kwargs)

  def test_starts_stopped_with_no_target(self):
    self.assertEqual(self.controller().command, (0.,) * 6)

  def test_forward_yaw_ramp_and_release(self):
    c = self.controller()
    c.update({'w', 'a'}, .1)
    self.assertAlmostEqual(c.command[0], .1)
    self.assertAlmostEqual(c.command[2], .2)
    c.update({'w'}, .1)
    self.assertAlmostEqual(c.command[0], .2)
    self.assertEqual(c.command[2], 0.)
    c.update(set(), .01)
    self.assertEqual(c.command[:3], (0., 0., 0.))

  def test_backward_right_and_opposites(self):
    c = self.controller()
    for _ in range(20):
      c.update({'s', 'd'}, .1)
    self.assertEqual(c.command[:3], (-.2, 0., -1.))
    c.update({'w', 's', 'a', 'd'}, .1)
    self.assertEqual(c.command[:3], (0., 0., 0.))

  def test_target_mapping_persistence_and_clear(self):
    for key, axis, sign in [('left', 3, -1), ('right', 3, 1),
                            ('up', 4, 1), ('down', 4, -1),
                            ('q', 5, 1), ('e', 5, -1)]:
      with self.subTest(key=key):
        c = self.controller()
        c.update({key}, .1)
        self.assertAlmostEqual(c.command[axis], sign * .012)
        saved = c.command
        c.update(set(), .1)
        self.assertEqual(c.command, saved)
        c.update({'r'}, .1)
        self.assertEqual(c.command[3:], (0., 0., 0.))

  def test_limits_and_height_floor(self):
    c = self.controller(min_z_offset=-.025)
    for _ in range(100):
      c.update({'w', 'a', 'right', 'up', 'q'}, .1)
    self.assertEqual(c.command, (.6, 0., 1., .10, .08, .10))
    for _ in range(100):
      c.update({'left', 'down', 'e'}, .1)
    self.assertEqual(c.command[3:], (-.10, -.08, -.025))

  def test_unfocused_and_paused_do_not_change_target(self):
    c = self.controller()
    c.update({'w', 'a', 'up'}, .1)
    target = c.command[3:]
    c.update({'w', 'a', 'q'}, .1, focused=False)
    self.assertEqual(c.command[:3], (0., 0., 0.))
    self.assertEqual(c.command[3:], target)
    c.update({'w', 'a', 'q'}, .1, paused=True)
    self.assertEqual(c.command[:3], (0., 0., 0.))
    self.assertEqual(c.command[3:], target)

  def test_reset_and_wall_time_invariance(self):
    a, b = self.controller(), self.controller()
    for _ in range(10):
      a.update({'w', 'q'}, .02)
    for _ in range(20):
      b.update({'w', 'q'}, .01)
    for x, y in zip(a.command, b.command):
      self.assertAlmostEqual(x, y)
    a.reset()
    self.assertEqual(a.command, (0.,) * 6)


if __name__ == '__main__':
  unittest.main()
