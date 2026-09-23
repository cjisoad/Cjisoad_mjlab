"""Read held control keys only while this process owns the focused X11 window."""
import ctypes as C
from ctypes.util import find_library
import os


class _XErrorEvent(C.Structure):
  _fields_ = [
    ('type', C.c_int), ('display', C.c_void_p), ('resourceid', C.c_ulong),
    ('serial', C.c_ulong), ('error_code', C.c_ubyte),
    ('request_code', C.c_ubyte), ('minor_code', C.c_ubyte),
  ]


_XErrorHandler = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(_XErrorEvent))
_active_keyboard = None


@_XErrorHandler
def _handle_x_error(display, event):
  # X11 reports a destroyed focused window asynchronously. Returning from the
  # handler lets the next focus poll treat that window as unfocused instead of
  # terminating the whole playback process through Xlib's default handler.
  keyboard = _active_keyboard
  if keyboard is not None:
    if display == keyboard.display and event.contents.error_code == 3:
      keyboard._x_error = event.contents.error_code
      return 0
    # XSetErrorHandler is process-wide. Preserve GLFW's and other connections'
    # error handling, and never suppress an unrelated error on our connection.
    if keyboard._old_error_handler:
      return keyboard._old_error_handler(display, event)
  return 0


class X11Keyboard:
  KEYS = {k: k for k in ('w', 'a', 's', 'd', 'q', 'e', 'r')}
  KEYS.update(up='Up', down='Down', left='Left', right='Right')

  def __init__(self):
    global _active_keyboard
    if _active_keyboard is not None:
      raise RuntimeError('An X11 keyboard connection is already active')
    if not os.environ.get('DISPLAY'):
      raise RuntimeError('Keyboard playback requires an X11 DISPLAY')
    self.lib = C.CDLL(find_library('X11') or 'libX11.so.6')
    ptr, ulong, integer = C.c_void_p, C.c_ulong, C.c_int
    signatures = {
      'XOpenDisplay': ([C.c_char_p], ptr),
      'XCloseDisplay': ([ptr], integer),
      'XStringToKeysym': ([C.c_char_p], ulong),
      'XKeysymToKeycode': ([ptr, ulong], C.c_ubyte),
      'XInternAtom': ([ptr, C.c_char_p, integer], ulong),
      'XGetInputFocus': ([ptr, C.POINTER(ulong), C.POINTER(integer)], integer),
      'XQueryKeymap': ([ptr, C.POINTER(C.c_ubyte)], integer),
      'XQueryTree': ([ptr, ulong, C.POINTER(ulong), C.POINTER(ulong),
                      C.POINTER(C.POINTER(ulong)), C.POINTER(C.c_uint)], integer),
      'XGetWindowProperty': ([ptr, ulong, ulong, C.c_long, C.c_long, integer, ulong,
                              C.POINTER(ulong), C.POINTER(integer), C.POINTER(ulong),
                              C.POINTER(ulong), C.POINTER(ptr)], integer),
      'XFree': ([ptr], integer),
      'XSync': ([ptr, integer], integer),
      'XGrabKey': ([ptr, integer, C.c_uint, ulong, integer, integer, integer], integer),
      'XUngrabKey': ([ptr, integer, C.c_uint, ulong], integer),
      'XPending': ([ptr], integer),
      'XNextEvent': ([ptr, ptr], integer),
      'XSetErrorHandler': ([_XErrorHandler], _XErrorHandler),
    }
    for name, (args, result) in signatures.items():
      function = getattr(self.lib, name)
      function.argtypes, function.restype = args, result
    self.display = self.lib.XOpenDisplay(None)
    if not self.display:
      raise RuntimeError('Cannot connect to the X11 display')
    self.pid_atom = self.lib.XInternAtom(self.display, b'_NET_WM_PID', 0)
    self._x_error = 0
    _active_keyboard = self
    self._old_error_handler = self.lib.XSetErrorHandler(_handle_x_error)
    self.keycodes = {name: self.lib.XKeysymToKeycode(
      self.display, self.lib.XStringToKeysym(symbol.encode())) for name, symbol in self.KEYS.items()}
    self.focused = False
    self._focus_window = 0
    self._grab_window = 0

  def _set_grab_window(self, window):
    if window == self._grab_window:
      return
    if self._grab_window:
      for code in set(self.keycodes.values()) - {0}:
        self.lib.XUngrabKey(self.display, code, 1 << 15, self._grab_window)
    self._grab_window = window
    if window:
      # Passive grabs consume controller keys before GLFW/MuJoCo's built-in
      # shortcuts. Polling XQueryKeymap still provides held-key state.
      for code in set(self.keycodes.values()) - {0}:
        self.lib.XGrabKey(self.display, code, 1 << 15, window, 0, 1, 1)
    self.lib.XSync(self.display, 0)

  def _discard_key_events(self):
    event = (C.c_long * 24)()  # XEvent's ABI reserves 24 longs.
    while self.lib.XPending(self.display):
      self.lib.XNextEvent(self.display, C.byref(event))

  def _owns_focus(self):
    self._x_error = 0
    self._focus_window = 0
    window, revert = C.c_ulong(), C.c_int()
    self.lib.XGetInputFocus(self.display, C.byref(window), C.byref(revert))
    # Toolkits may focus a child window while _NET_WM_PID is on its parent.
    for _ in range(16):
      if window.value <= 1:
        return False
      actual_type, fmt = C.c_ulong(), C.c_int()
      count, remaining, data = C.c_ulong(), C.c_ulong(), C.c_void_p()
      status = self.lib.XGetWindowProperty(self.display, window, self.pid_atom,
        0, 1, 0, 6, C.byref(actual_type), C.byref(fmt), C.byref(count),
        C.byref(remaining), C.byref(data))
      self.lib.XSync(self.display, 0)
      if self._x_error:
        self._x_error = 0
        if data:
          self.lib.XFree(data)
        return False
      try:
        if status == 0 and fmt.value == 32 and count.value and data:
          owned = C.cast(data, C.POINTER(C.c_ulong))[0] == os.getpid()
          if owned:
            self._focus_window = window.value
          return owned
      finally:
        if data:
          self.lib.XFree(data)
      root, parent, children, size = C.c_ulong(), C.c_ulong(), C.POINTER(C.c_ulong)(), C.c_uint()
      ok = self.lib.XQueryTree(self.display, window, C.byref(root), C.byref(parent),
                             C.byref(children), C.byref(size))
      self.lib.XSync(self.display, 0)
      if self._x_error:
        self._x_error = 0
        if children:
          self.lib.XFree(children)
        return False
      if children:
        self.lib.XFree(children)
      if not ok:
        return False
      window = parent
    return False

  def read(self):
    self.focused = self._owns_focus()
    self._set_grab_window(self._focus_window if self.focused else 0)
    self._discard_key_events()
    if not self.focused:
      return set()
    bitmap = (C.c_ubyte * 32)()
    self.lib.XQueryKeymap(self.display, bitmap)
    return {name for name, code in self.keycodes.items()
            if code and bitmap[code // 8] & (1 << (code % 8))}

  def close(self):
    if self.display:
      self._set_grab_window(0)
      self.lib.XSync(self.display, 0)
      self.lib.XSetErrorHandler(self._old_error_handler)
      global _active_keyboard
      if _active_keyboard is self:
        _active_keyboard = None
      self.lib.XCloseDisplay(self.display)
      self.display = None
