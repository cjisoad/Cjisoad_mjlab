"""GLFW viewer with mouse picking that presses physical scene buttons."""

from __future__ import annotations

import math
import time
from typing import Optional

import glfw
import mujoco
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glPixelStorei,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glViewport,
    GL_QUADS,
    GL_TEXTURE0,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_CLAMP_TO_EDGE,
    glActiveTexture,
    glOrtho,
)

from .controller import ThreeFloorElevatorController


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def pick_button(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: mujoco.MjvScene,
    option: mujoco.MjvOption,
    aspect: float,
    relx: float,
    rely: float,
) -> Optional[str]:
    """Return the logical button under normalized viewport coordinates."""

    selected_point = np.zeros(3, dtype=np.float64)
    geom_id = np.array([-1], dtype=np.int32)
    flex_id = np.array([-1], dtype=np.int32)
    skin_id = np.array([-1], dtype=np.int32)
    body_id = mujoco.mjv_select(
        model,
        data,
        option,
        float(aspect),
        float(relx),
        float(rely),
        scene,
        selected_point,
        geom_id,
        flex_id,
        skin_id,
    )
    if body_id < 0 or geom_id[0] < 0:
        return None

    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0]))
    if not geom_name or not geom_name.startswith("geom_"):
        return None
    return geom_name.removeprefix("geom_")


def _make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (1.4, -0.1, 4.15)
    camera.distance = 14.0
    camera.azimuth = 90.0
    camera.elevation = -12.0
    return camera


def _upload_frame(texture_id: int, image: np.ndarray) -> None:
    height, width, _ = image.shape
    flipped = np.ascontiguousarray(np.flipud(image))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        flipped,
    )


def _draw_texture(texture_id: int, width: int, height: int) -> None:
    glViewport(0, 0, width, height)
    glClearColor(0.025, 0.03, 0.04, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, 1.0)
    glEnd()


def run(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ThreeFloorElevatorController,
) -> None:
    """Run the real-time simulation and interactive MuJoCo-rendered window."""

    if not glfw.init():
        raise RuntimeError("GLFW could not initialize; check the display/OpenGL setup")

    window = None
    renderer = None
    texture_id = 0
    try:
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Three-Floor MuJoCo Elevator", None, None)
        if window is None:
            raise RuntimeError("GLFW could not create a window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        renderer = mujoco.Renderer(model, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
        glfw.make_context_current(window)
        camera = _make_camera()
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        option.geomgroup[3] = 0
        texture_id = int(glGenTextures(1))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        rotating = False
        last_cursor = [0.0, 0.0]

        def on_mouse_button(current_window, button, action, _mods):
            nonlocal rotating
            if button == glfw.MOUSE_BUTTON_RIGHT:
                rotating = action != glfw.RELEASE
                last_cursor[:] = glfw.get_cursor_pos(current_window)
                return
            if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
                return

            cursor_x, cursor_y = glfw.get_cursor_pos(current_window)
            window_width, window_height = glfw.get_window_size(current_window)
            if window_width <= 0 or window_height <= 0:
                return
            renderer.update_scene(data, camera=camera, scene_option=option)
            name = pick_button(
                model,
                data,
                renderer.scene,
                option,
                WINDOW_WIDTH / WINDOW_HEIGHT,
                cursor_x / window_width,
                1.0 - cursor_y / window_height,
            )
            if name is not None:
                controller.mouse_press(name)

        def on_cursor_move(_current_window, cursor_x, cursor_y):
            if rotating:
                dx = cursor_x - last_cursor[0]
                dy = cursor_y - last_cursor[1]
                camera.azimuth = (camera.azimuth + 0.35 * dx) % 360.0
                camera.elevation = max(-80.0, min(10.0, camera.elevation + 0.30 * dy))
            last_cursor[:] = (cursor_x, cursor_y)

        def on_scroll(_current_window, _xoffset, yoffset):
            camera.distance = max(6.0, min(32.0, camera.distance * math.exp(-0.08 * yoffset)))

        glfw.set_mouse_button_callback(window, on_mouse_button)
        glfw.set_cursor_pos_callback(window, on_cursor_move)
        glfw.set_scroll_callback(window, on_scroll)

        timestep = model.opt.timestep
        last_wall_time = time.perf_counter()
        accumulator = 0.0
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            accumulator = min(accumulator + now - last_wall_time, 0.05)
            last_wall_time = now
            while accumulator >= timestep:
                controller.apply_mouse_forces()
                mujoco.mj_step(model, data)
                controller.update()
                accumulator -= timestep

            renderer.update_scene(data, camera=camera, scene_option=option)
            image = renderer.render()
            glfw.make_context_current(window)
            _upload_frame(texture_id, image)
            framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
            _draw_texture(texture_id, framebuffer_width, framebuffer_height)
            glfw.swap_buffers(window)
            glfw.poll_events()
            if accumulator < timestep:
                glfw.wait_events_timeout(0.001)
    finally:
        if window is not None and texture_id:
            glfw.make_context_current(window)
            glDeleteTextures([texture_id])
        if renderer is not None:
            renderer.close()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
