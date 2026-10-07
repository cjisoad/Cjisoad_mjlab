"""Teacher workspace selection and continuous absolute target commands."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import keyboard_teacher as player


class TeacherKeyboardTests(unittest.TestCase):
  def workspace(self, mode):
    self.assertTrue(hasattr(player, 'workspace_for_mode'), 'Three workspace modes are missing')
    return player.workspace_for_mode(mode)

  def controller(self, mode, idle=(0., 0., 0.)):
    return player.TeacherKeyboardController(self.workspace(mode), idle_target=idle)

  def test_teacher_ranges_and_union_preserve_the_three_dimensional_shape(self):
    quad, biped, union = [self.workspace(mode) for mode in ('quadruped', 'biped', 'union')]
    self.assertTrue(quad.contains((-.12, -.09, .05)))
    self.assertTrue(biped.contains((.30, .10, .72)))
    self.assertFalse(quad.contains((.2, 0., .15)))
    self.assertFalse(biped.contains((-.1, 0., .5)))
    for point in ((-.1, .08, .1), (.3, .1, .6), (0., 0., .3)):
      self.assertTrue(union.contains(point))
    # These are inside the combined bounding box, outside the actual union.
    for point in ((.2, 0., .1), (-.1, 0., .5), (-.1, .1, .3)):
      self.assertFalse(union.contains(point))
      self.assertTrue(union.contains(union.project(point)))

  def test_each_mode_reaches_its_full_height_band(self):
    for mode, low, high in (('quadruped', .05, .35), ('biped', .25, .72), ('union', .05, .72)):
      with self.subTest(mode=mode):
        controller = self.controller(mode)
        for _ in range(200):
          controller.update({'q'}, .1)
        self.assertAlmostEqual(controller.command[5], high)
        for _ in range(200):
          controller.update({'e'}, .1)
        self.assertAlmostEqual(controller.command[5], low)

  def test_union_crosses_overlap_while_remaining_inside_a_teacher_range(self):
    controller = self.controller('union')
    for _ in range(200):
      controller.update({'left'}, .1)
    self.assertAlmostEqual(controller.command[3], -.12)
    for _ in range(200):
      controller.update({'q'}, .1)
      self.assertTrue(controller.workspace.contains(controller.command[3:]))
    self.assertAlmostEqual(controller.command[3], -.05)
    self.assertAlmostEqual(controller.command[5], .72)

  def test_biped_idle_offset_is_not_added_twice_and_r_restores_idle(self):
    idle = (.13349, .03060, .47804)
    controller = self.controller('biped', idle)
    self.assertEqual(controller.command[3:], idle)
    controller.update({'q'}, .1)
    self.assertAlmostEqual(controller.command[5], idle[2] + .025)
    saved = controller.command[3:]
    controller.update(set(), .1)
    self.assertEqual(controller.command[3:], saved)
    controller.update({'r'}, .1)
    self.assertEqual(controller.command[3:], idle)
    self.assertFalse(controller.target_active)

  def test_cross_mode_idle_is_preserved_until_a_target_key_is_pressed(self):
    idle = (.13349, .03060, .47804)
    controller = self.controller('quadruped', idle)
    controller.update({'w'}, .1)
    self.assertEqual(controller.command[3:], idle)
    controller.update({'q'}, .1)
    self.assertTrue(controller.workspace.contains(controller.command[3:]))

  def test_focus_pause_reset_and_runtime_timestep_validation(self):
    controller = self.controller('union')
    controller.update({'w', 'q'}, .1)
    saved = controller.command[3:]
    for kw in ({'focused': False}, {'paused': True}):
      controller.update({'w', 'e'}, .1, **kw)
      self.assertEqual(controller.command[:3], (0., 0., 0.))
      self.assertEqual(controller.command[3:], saved)
    controller.reset()
    self.assertEqual(controller.command, (0.,) * 6)
    with self.assertRaises(ValueError):
      controller.update({'q'}, float('nan'))

  def test_velocity_limits_follow_the_loaded_teacher_and_release_stops_motion(self):
    controller = player.TeacherKeyboardController(
      self.workspace('union'), vx_range=(-.5, 1.), wz_range=(-.8, .8))
    for _ in range(100):
      controller.update({'w', 'a'}, .1)
    self.assertEqual(controller.command[:3], (1., 0., .8))
    self.assertFalse(controller.target_active)
    for _ in range(100):
      controller.update({'s', 'd'}, .1)
    self.assertEqual(controller.command[:3], (-.5, 0., -.8))
    controller.update(set(), .1)
    self.assertEqual(controller.command[:3], (0., 0., 0.))

  def test_cli_exposes_three_modes_and_accepts_documented_num_envs(self):
    self.assertTrue(hasattr(player, 'parse_args'), 'CLI workspace selection is missing')
    for mode in ('quadruped', 'biped', 'union'):
      args = player.parse_args(['pedipulation_t', '--checkpoint-file', 'model.pt',
                                '--workspace', mode, '--num-envs', '1'])
      self.assertEqual(args.workspace, mode)
      self.assertEqual(args.num_envs, 1)


if __name__ == '__main__':
  unittest.main()
