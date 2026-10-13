"""Third-person MuJoCo scene rendering with a caller-controlled free camera."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController
from simulate.mujoco.model.three_floor_elevator.web_viewer import encode_png


class SceneCamera:
    def __init__(self, model, *, width=640, height=360):
        self.model = model
        self.width, self.height = width, height
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.geomgroup[:] = (1, 1, 1, 0, 1, 1)
        self.closed = False
        self._last_signature = None
        self._last_png = None

    @staticmethod
    def _number(camera, key, default, low, high):
        value = float(camera.get(key, default))
        if not math.isfinite(value):
            raise ValueError(f'camera {key} must be finite')
        return max(low, min(high, value))

    def _camera_pose(self, camera):
        camera = camera or {}
        return (
            self._number(camera, 'azimuth', 145., -360., 360.) % 360.,
            self._number(camera, 'elevation', -25., -85., 85.),
            self._number(camera, 'distance', 9.5, 2., 40.),
            (
                self._number(camera, 'x', 4.6, -10., 20.),
                self._number(camera, 'y', -1., -12., 12.),
                self._number(camera, 'z', 0.7, -3., 14.),
            ),
        )

    def _signature(self, data, camera):
        return (float(data.time), data.qpos.tobytes(), data.qvel.tobytes(),
                data.ctrl.tobytes(), self._camera_pose(camera))

    def _update_scene(self, data, camera=None):
        if self.closed:
            raise RuntimeError('scene camera is closed')
        azimuth, elevation, distance, lookat = self._camera_pose(camera)
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
        self.camera.lookat[:] = lookat
        self.data.qpos[:] = data.qpos
        self.data.qvel[:] = data.qvel
        self.data.ctrl[:] = data.ctrl
        self.data.qfrc_applied[:] = data.qfrc_applied
        if self.model.na:
            self.data.act[:] = data.act
        if self.model.nmocap:
            self.data.mocap_pos[:] = data.mocap_pos
            self.data.mocap_quat[:] = data.mocap_quat
        self.data.time = data.time
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)

    def render(self, data, camera=None):
        if self.closed:
            raise RuntimeError('scene camera is closed')
        signature = self._signature(data, camera)
        if signature == self._last_signature:
            return self._last_png
        self._update_scene(data, camera)
        self._last_png = encode_png(np.asarray(self.renderer.render(), dtype=np.uint8).copy())
        self._last_signature = signature
        return self._last_png

    def pick_button(self, data, camera, relx, rely):
        if not (math.isfinite(relx) and math.isfinite(rely) and 0 <= relx <= 1 and 0 <= rely <= 1):
            raise ValueError('scene click must be inside the viewport')
        self._update_scene(data, camera)
        point = np.zeros(3, dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        flex_id = np.array([-1], dtype=np.int32)
        skin_id = np.array([-1], dtype=np.int32)
        body_id = mujoco.mjv_select(self.model, self.data, self.option,
            self.width / self.height, relx, rely, self.renderer.scene,
            point, geom_id, flex_id, skin_id)
        if body_id < 0 or geom_id[0] < 0:
            return None
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
            int(self.model.geom_bodyid[int(geom_id[0])]))
        if not body_name or not body_name.startswith('button_'):
            return None
        name = body_name.removeprefix('button_')
        return name if name in ThreeFloorElevatorController.BUTTON_NAMES else None

    def close(self):
        if not self.closed:
            self.renderer.close()
            self.closed = True
