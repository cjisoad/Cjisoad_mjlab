"""X11 focus polling handles a destroyed window without exiting playback."""
import ctypes as C
import os
from pathlib import Path
import sys
from unittest import SkipTest, TestCase
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class X11KeyboardTests(TestCase):
  def test_controller_keys_are_consumed_but_space_reaches_window(self):
    if not os.environ.get('DISPLAY'):
      raise SkipTest('X11 display is unavailable')
    from ctypes.util import find_library
    from keyboard_x11 import X11Keyboard
    library = find_library('Xtst')
    if not library:
      raise SkipTest('XTest is unavailable')
    keyboard = X11Keyboard()
    x = keyboard.lib
    signatures = {
      'XDefaultRootWindow': ([C.c_void_p], C.c_ulong),
      'XCreateSimpleWindow': ([C.c_void_p, C.c_ulong, C.c_int, C.c_int,
        C.c_uint, C.c_uint, C.c_uint, C.c_ulong, C.c_ulong], C.c_ulong),
      'XChangeProperty': ([C.c_void_p, C.c_ulong, C.c_ulong, C.c_ulong,
        C.c_int, C.c_int, C.c_void_p, C.c_int], C.c_int),
      'XSelectInput': ([C.c_void_p, C.c_ulong, C.c_long], C.c_int),
      'XMapWindow': ([C.c_void_p, C.c_ulong], C.c_int),
      'XSetInputFocus': ([C.c_void_p, C.c_ulong, C.c_int, C.c_ulong], C.c_int),
      'XDestroyWindow': ([C.c_void_p, C.c_ulong], C.c_int),
    }
    for name, (args, result) in signatures.items():
      getattr(x, name).argtypes, getattr(x, name).restype = args, result
    display = x.XOpenDisplay(None)
    previous, revert = C.c_ulong(), C.c_int()
    x.XGetInputFocus(display, C.byref(previous), C.byref(revert))
    window = x.XCreateSimpleWindow(display, x.XDefaultRootWindow(display),
      0, 0, 100, 100, 0, 0, 0)
    pid = C.c_ulong(os.getpid())
    x.XChangeProperty(display, window, keyboard.pid_atom, 6, 32, 0, C.byref(pid), 1)
    x.XSelectInput(display, window, 1 | 2)
    x.XMapWindow(display, window)
    x.XSync(display, 0)
    time.sleep(.1)
    x.XSetInputFocus(display, window, 1, 0)
    x.XSync(display, 0)
    xtest = C.CDLL(library)
    xtest.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
    xtest.XTestFakeKeyEvent.restype = C.c_int

    def inject(code, pressed):
      xtest.XTestFakeKeyEvent(display, code, pressed, 0)
      x.XSync(display, 0)

    def key_events():
      types = []
      event = (C.c_long * 24)()
      while x.XPending(display):
        x.XNextEvent(display, C.byref(event))
        event_type = C.cast(C.byref(event), C.POINTER(C.c_int))[0]
        if event_type in (2, 3):
          types.append(event_type)
      return types

    code = keyboard.keycodes['w']
    space = x.XKeysymToKeycode(display, x.XStringToKeysym(b'space'))
    try:
      keyboard.read()
      self.assertTrue(keyboard.focused)
      for name, code in keyboard.keycodes.items():
        inject(code, 1)
        self.assertIn(name, keyboard.read())
        self.assertEqual(key_events(), [], f'{name} must not reach native viewer')
        inject(code, 0)
        self.assertNotIn(name, keyboard.read())
        self.assertEqual(key_events(), [])
      inject(space, 1)
      inject(space, 0)
      self.assertEqual(key_events(), [2, 3], 'Space must still reach native viewer')
      x.XSetInputFocus(display, previous, revert, 0)
      x.XSync(display, 0)
      self.assertEqual(keyboard.read(), set())
      self.assertEqual(keyboard._grab_window, 0)
    finally:
      inject(code, 0)
      inject(space, 0)
      keyboard.close()
      x.XSetInputFocus(display, previous, revert, 0)
      x.XDestroyWindow(display, window)
      x.XCloseDisplay(display)

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
