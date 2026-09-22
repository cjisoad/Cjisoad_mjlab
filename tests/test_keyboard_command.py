"""Manual commands remain authoritative across resampling, heading and reset."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class ManualCommandTests(unittest.TestCase):
  def test_manual_command_survives_heading_resampling_and_reset(self):
    path = Path(__file__).resolve().parents[1] / 'scripts/keyboard_play.py'
    self.assertTrue(path.exists(), 'Keyboard playback is not implemented')
    import torch
    from keyboard_control import KeyboardController
    from keyboard_play import manual_command_cfg
    from test_pedipulation import PedipulationTaskTests
    env, previous = PedipulationTaskTests().make_env()
    controller = KeyboardController()
    term = manual_command_cfg(previous.cfg, controller).build(env)
    controller.update({'w', 'd', 'up', 'e'}, .1)
    expected = torch.tensor(controller.command).expand(2, -1)
    for step in (1, 499, 500, 501, 1000):
      env.common_step_counter = step
      env.episode_length_buf[:] = step
      term.heading_target[:] = 2.8
      torch.testing.assert_close(term.command, expected)
      term.compute(.02)
      torch.testing.assert_close(term.command, expected)
    term.reset(torch.tensor([0, 1]))
    torch.testing.assert_close(term.command, expected)
    controller.reset()
    self.assertFalse(term.has_foot_target.any())
    torch.testing.assert_close(term.command, torch.zeros(2, 6))
    self.assertTrue(torch.isfinite(term.anchor_frame.basis_w).all())


if __name__ == '__main__':
  unittest.main()
