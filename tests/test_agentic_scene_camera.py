from types import SimpleNamespace

import numpy as np

from src.agentic_robot.observation.scene_camera import SceneCamera


class Renderer:
    def __init__(self):
        self.calls = 0

    def render(self):
        self.calls += 1
        return np.full((2, 3, 3), self.calls, dtype=np.uint8)


def test_scene_image_reuses_unchanged_frames_and_invalidates_on_state_or_camera():
    camera = SceneCamera.__new__(SceneCamera)
    camera.closed = False
    camera.renderer = Renderer()
    camera._last_signature = None
    camera._last_png = None
    updates = []
    camera._update_scene = lambda data, view: updates.append((data.time, view))
    data = SimpleNamespace(time=0., qpos=np.zeros(2), qvel=np.zeros(2), ctrl=np.zeros(1))

    first = camera.render(data, {})
    assert camera.render(data, {}) is first
    assert camera.renderer.calls == 1

    data.qpos[0] = 1.
    second = camera.render(data, {})
    assert second != first
    assert camera.renderer.calls == 2

    third = camera.render(data, {'azimuth': 30})
    assert third != second
    assert camera.renderer.calls == 3
