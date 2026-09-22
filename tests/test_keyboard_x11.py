"""X11 focus polling handles a destroyed window without exiting playback."""
import ctypes as C
import os
from pathlib import Path
import sys
from unittest import SkipTest, TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class X11KeyboardTests(TestCase):
  def test_only_one_error_handler_owner_can_be_active(self):
    if not os.environ.get('DISPLAY'):
      raise SkipTest('X11 display is unavailable')
    from keyboard_x11 import X11Keyboard
    keyboard = X11Keyboard()
    try:
      with self.assertRaisesRegex(RuntimeError, 'already active'):
        X11Keyboard()
    finally:
      keyboard.close()

  def test_stale_focus_window_is_unfocused(self):
    if not os.environ.get('DISPLAY'):
      raise SkipTest('X11 display is unavailable')
    from keyboard_x11 import X11Keyboard
    keyboard = X11Keyboard()
    original = keyboard.lib.XGetInputFocus

    def stale(_display, window, revert):
      C.cast(window, C.POINTER(C.c_ulong))[0] = 0x7FFFFFFE
      C.cast(revert, C.POINTER(C.c_int))[0] = 1
      return 1

    keyboard.lib.XGetInputFocus = stale
    try:
      self.assertFalse(keyboard._owns_focus())
    finally:
      keyboard.lib.XGetInputFocus = original
      keyboard.close()
