"""The desktop view mirrors state without gaining control of the simulation."""
from contextlib import nullcontext

import mujoco
import numpy as np
import pytest


def test_native_mirror_refreshes_during_action_without_advancing_source():
    from src.agentic_robot.native_viewer import NativeSceneViewer

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <light name="lamp"/><body><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>''')
    source = mujoco.MjData(model)
    view = NativeSceneViewer(model, source)

    class Handle:
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        def lock(self):
            return nullcontext()
        def sync(self, **kwargs):
            self.synced_time = float(view.data.time)
            # Simulate GUI edits: they must never reach the source world.
            view.data.qpos[0] = 99.
        def is_running(self):
            return True

    view.handle = Handle()
    source.time = .42
    source.qpos[0] = 2.
    model.light_active[0] = False
    view.publish(source)
    assert view.tick()
    assert view.handle.synced_time == .42
    assert source.time == .42
    assert source.qpos[0] == 2.
    assert not view.model.light_active[0]
    assert view.model is not model
    # Refresh clears edits made by the GUI before the next render.
    view.handle.sync = lambda **kwargs: None
    view.tick()
    np.testing.assert_array_equal(view.data.qpos, source.qpos)


def test_native_camera_is_third_person_and_hides_collision_walls():
    from src.agentic_robot.native_viewer import configure_camera

    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    configure_camera(camera, option)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert camera.distance > 2.
    assert option.geomgroup[3] == 0


@pytest.mark.parametrize('button', ['light_f1', 'call_up_f1', 'cab_f2'])
def test_native_click_picks_real_button_and_queues_press_without_editing_world(button):
    from src.agentic_robot.native_viewer import NativeSceneViewer
    from test_three_floor_elevator import SCENE
    from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, source)
    controller.reset()
    presses = []
    view = NativeSceneViewer(model, source, on_button=presses.append)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    from src.agentic_robot.native_viewer import configure_camera
    configure_camera(camera, option)
    view._refresh()
    # Look straight at the button's physical face from inside the room/cabin.
    geom = view.data.geom('geom_' + button)
    camera.lookat[:] = geom.xpos
    camera.distance = .5
    camera.elevation = 0.
    camera.azimuth = 0. if button.startswith('cab_') else 90.
    scene = mujoco.MjvScene(view.model, maxgeom=1000)
    mujoco.mjv_updateScene(view.model, view.data, option, None, camera,
        mujoco.mjtCatBit.mjCAT_ALL, scene)
    before = source.qpos.copy()
    assert view.click(scene, option, 16 / 9, .5, .5) == button
    assert presses == [button]
    np.testing.assert_array_equal(source.qpos, before)
    assert source.time == 0.


def test_native_mirror_includes_lamp_emission_and_call_indicator_color():
    from src.agentic_robot.native_viewer import NativeSceneViewer
    from test_three_floor_elevator import SCENE
    from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, source)
    controller.reset()
    view = NativeSceneViewer(model, source)
    controller._handle_button_press('light_f1')
    controller._handle_button_press('call_up_f1')
    view.publish(source)
    view._refresh()
    np.testing.assert_array_equal(view.model.mat_emission, model.mat_emission)
    np.testing.assert_array_equal(view.model.geom_rgba, model.geom_rgba)


def test_native_glfw_single_click_uses_window_coordinates_and_drag_does_not_press(monkeypatch):
    import glfw
    from types import SimpleNamespace
    from src.agentic_robot.native_viewer import _SceneWindow
    window = object.__new__(_SceneWindow)
    window.glfw = glfw
    window.window = object()
    window._drag_button = None
    window._press_start = None
    window._dragged = False
    window._last_cursor = (0., 0.)
    clicks = []
    window.view = SimpleNamespace(click=lambda *args: clicks.append(args))
    window.scene = object()
    window.opt = object()
    window._update_scene = lambda: None
    cursor = [320., 180.]
    monkeypatch.setattr(glfw, 'get_cursor_pos', lambda _: tuple(cursor))
    monkeypatch.setattr(glfw, 'get_window_size', lambda _: (640, 360))
    monkeypatch.setattr(glfw, 'get_framebuffer_size', lambda _: (1280, 720))
    window._on_mouse_button(window.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    window._on_mouse_button(window.window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
    assert clicks == [(window.scene, window.opt, 1280 / 720, .5, .5)]
    window._on_mouse_button(window.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    cursor[0] += 20
    window._on_mouse_button(window.window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
    assert len(clicks) == 1
