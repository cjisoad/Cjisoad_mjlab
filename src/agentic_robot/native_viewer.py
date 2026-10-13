"""Third-person desktop display with queued physical button presses."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import math
import threading

import mujoco
import numpy as np

from .observation.controls import BUTTON_NAMES


def configure_camera(camera, option):
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.
    camera.elevation = -25.
    camera.distance = 9.5
    camera.lookat[:] = (4.6, -1., .7)
    # Group 3 contains invisible collision walls, as in the web scene camera.
    option.geomgroup[:] = (1, 1, 1, 0, 1, 1)


class NativeSceneViewer:
    def __init__(self, model, data, *, on_button=None):
        self.source_model = model
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.handle = None
        self.lock = threading.Lock()
        self.snapshot = None
        self.on_button = on_button
        self.publish(data)

    def publish(self, data):
        # No GUI lock or runtime lock is needed here. The action owner already
        # has a current CPU mirror; publishing never reads GPU state again.
        snapshot = {name: getattr(data, name).copy() for name in (
            'qpos', 'qvel', 'ctrl', 'act', 'mocap_pos', 'mocap_quat',
            'qfrc_applied', 'xfrc_applied')}
        snapshot['time'] = float(data.time)
        for name in ('light_active', 'mat_emission', 'geom_rgba'):
            snapshot[name] = getattr(self.source_model, name).copy()
        with self.lock:
            self.snapshot = snapshot

    def open(self):
        self._refresh()
        # launch_passive exposes keyboard callbacks only. Own the GLFW window
        # so single mouse clicks can pick buttons without touching live physics.
        self.handle = _SceneWindow(self)

    def click(self, scene, option, aspect, relx, rely):
        if not (math.isfinite(relx) and math.isfinite(rely) and 0 <= relx <= 1 and 0 <= rely <= 1):
            return None
        point = np.zeros(3)
        geom, flex, skin = (np.array([-1], dtype=np.int32) for _ in range(3))
        body = mujoco.mjv_select(self.model, self.data, option, aspect, relx, rely,
            scene, point, geom, flex, skin)
        if body < 0 or geom[0] < 0:
            return None
        name = self.model.body(int(self.model.geom_bodyid[geom[0]])).name
        button = name.removeprefix('button_') if name else None
        if button not in BUTTON_NAMES:
            return None
        if self.on_button is not None:
            self.on_button(button)
        return button

    def _refresh(self):
        with self.lock:
            snapshot = self.snapshot
        for name, value in snapshot.items():
            if name == 'time':
                self.data.time = value
            elif name in ('light_active', 'mat_emission', 'geom_rgba'):
                getattr(self.model, name)[:] = value
            else:
                getattr(self.data, name)[:] = value
        mujoco.mj_forward(self.model, self.data)

    def tick(self):
        if self.handle is None or not self.handle.is_running():
            return False
        with self.handle.lock():
            self._refresh()
        self.handle.sync(state_only=True)
        return True

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class _SceneWindow:
    """Render the CPU mirror only; GLFW callbacks queue scene input."""
    def __init__(self, view):
        import glfw
        self.glfw = glfw
        self.view = view
        self.window = None
        self.context = None
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        configure_camera(self.cam, self.opt)
        self.scene = mujoco.MjvScene(view.model, maxgeom=10000)
        self._drag_button = None
        self._press_start = None
        self._dragged = False
        self._last_cursor = (0., 0.)
        if not glfw.init():
            raise RuntimeError('GLFW could not initialize; check DISPLAY/OpenGL')
        try:
            self.window = glfw.create_window(1280, 720, 'MuJoCo : Agentic Elevator', None, None)
            if self.window is None:
                raise RuntimeError('GLFW could not create the elevator window')
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = mujoco.MjrContext(view.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
            glfw.set_scroll_callback(self.window, self._on_scroll)
            glfw.set_key_callback(self.window, self._on_key)
        except Exception:
            self.close()
            raise

    def lock(self):
        return nullcontext()

    def _update_scene(self):
        mujoco.mjv_updateScene(self.view.model, self.view.data, self.opt, None,
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

    def _on_mouse_button(self, window, button, action, _mods):
        glfw = self.glfw
        cursor = glfw.get_cursor_pos(window)
        if action == glfw.PRESS:
            self._drag_button = button
            self._last_cursor = cursor
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._press_start = cursor
                self._dragged = False
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT and self._press_start is not None:
                dx, dy = (cursor[i] - self._press_start[i] for i in range(2))
                if not self._dragged and math.hypot(dx, dy) <= 5.:
                    width, height = glfw.get_window_size(window)
                    fb_width, fb_height = glfw.get_framebuffer_size(window)
                    if width > 0 and height > 0 and fb_width > 0 and fb_height > 0:
                        self._update_scene()
                        self.view.click(self.scene, self.opt, fb_width / fb_height,
                            cursor[0] / width, 1. - cursor[1] / height)
                self._press_start = None
            self._drag_button = None

    def _on_cursor_move(self, window, x, y):
        glfw = self.glfw
        dx, dy = x - self._last_cursor[0], y - self._last_cursor[1]
        self._last_cursor = (x, y)
        if self._drag_button is None:
            return
        if self._press_start is not None:
            if math.hypot(x - self._press_start[0], y - self._press_start[1]) > 5.:
                self._dragged = True
            if not self._dragged:
                return
        _, height = glfw.get_window_size(window)
        if height <= 0:
            return
        shift = any(glfw.get_key(window, key) == glfw.PRESS for key in
                    (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT))
        if self._drag_button == glfw.MOUSE_BUTTON_RIGHT:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._drag_button == glfw.MOUSE_BUTTON_LEFT:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.view.model, action, dx / height, dy / height, self.scene, self.cam)

    def _on_scroll(self, _window, _xoffset, yoffset):
        mujoco.mjv_moveCamera(self.view.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0., -.05 * yoffset, self.scene, self.cam)

    def _on_key(self, window, key, _scancode, action, _mods):
        if key == self.glfw.KEY_ESCAPE and action == self.glfw.PRESS:
            self.glfw.set_window_should_close(window, True)

    def sync(self, **_kwargs):
        self.glfw.make_context_current(self.window)
        self._update_scene()
        width, height = self.glfw.get_framebuffer_size(self.window)
        if width > 0 and height > 0:
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, self.context)
            self.glfw.swap_buffers(self.window)
        self.glfw.poll_events()

    def is_running(self):
        return self.window is not None and not self.glfw.window_should_close(self.window)

    def close(self):
        if self.window is not None:
            self.glfw.make_context_current(self.window)
            if self.context is not None:
                self.context.free()
                self.context = None
            self.glfw.destroy_window(self.window)
            self.window = None
