"""Keyboard range extension and leg_manip playback wiring contracts."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from src.tasks.leg_manip.constants import TRIPOD_HIGH


class KeyboardRangeTests(unittest.TestCase):
  def test_default_bounds_unchanged(self):
    from keyboard_control import KeyboardController
    c = KeyboardController()
    for _ in range(100):
      c.update({'right', 'q'}, .1)
    self.assertEqual(c.command[3], .10)
    self.assertEqual(c.command[5], .10)

  def test_extended_bounds_reach_above_band_targets(self):
    from keyboard_control import KeyboardController
    c = KeyboardController(dx_high=.30, dz_high=.72)
    for _ in range(200):
      c.update({'right', 'q'}, .1)
    self.assertEqual(c.command[3], .30)
    self.assertEqual(c.command[5], .72)
    self.assertGreater(c.command[5], TRIPOD_HIGH)
    for _ in range(400):
      c.update({'left', 'e'}, .1)
    self.assertEqual(c.command[3], -.10)
    self.assertEqual(c.command[5], -.10)


class LegManipPlaybackTests(unittest.TestCase):
  def test_bridge_is_the_shared_loco_bridge(self):
    from keyboard_leg_manip import KeyboardCommandBridge
    from keyboard_loco_pedipulation import KeyboardCommandBridge as LocoBridge
    self.assertIs(KeyboardCommandBridge, LocoBridge)

  def test_controller_bounds_come_from_the_workspace_bank(self):
    from keyboard_leg_manip import workspace_controller
    from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace
    controller = workspace_controller()
    workspace = build_foot_workspace()
    self.assertGreaterEqual(controller.dz_high, float(workspace.offsets[:, 2].max()))
    self.assertGreaterEqual(controller.dx_high, float(workspace.offsets[:, 0].max()))
    self.assertGreater(controller.target_rate, .12)


if __name__ == '__main__':
  unittest.main()
