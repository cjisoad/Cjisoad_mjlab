"""Keyboard bridge tests for Loco Pedipulation playback."""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class KeyboardLocoBridgeTests(unittest.TestCase):
  def test_bridge_maps_velocity_and_target_and_clears_target(self):
    from keyboard_loco_pedipulation import KeyboardCommandBridge

    calls = []

    class Term:
      def set_command(self, env_ids, *, velocity=None, target_offset=None):
        calls.append((tuple(env_ids), velocity, target_offset))

    controller = SimpleNamespace(command=(.2, 0., -.4, .03, -.02, .07))
    bridge = KeyboardCommandBridge(Term(), num_envs=1)
    bridge.update(controller)
    self.assertEqual(calls[-1], ((0,), (.2, 0., -.4), (.03, -.02, .07)))

    controller.command = (0., 0., 0., 0., 0., 0.)
    bridge.update(controller)
    self.assertEqual(calls[-1], ((0,), (0., 0., 0.), (0., 0., 0.)))

  def test_bridge_does_not_restart_target_transition_every_tick(self):
    from keyboard_loco_pedipulation import KeyboardCommandBridge

    target_calls = []

    class Term:
      def set_command(self, env_ids, *, velocity=None, target_offset=None):
        if target_offset is not None:
          target_calls.append(target_offset)

    controller = SimpleNamespace(command=(.1, 0., 0., .03, 0., .07))
    bridge = KeyboardCommandBridge(Term(), num_envs=1)
    bridge.update(controller)
    bridge.update(controller)
    self.assertEqual(target_calls, [(0.03, 0., 0.07)])


if __name__ == "__main__":
  unittest.main()
