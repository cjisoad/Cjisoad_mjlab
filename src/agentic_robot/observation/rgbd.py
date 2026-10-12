"""Synchronous native MuJoCo RGB and metric camera-Z rendering."""
from __future__ import annotations

from io import BytesIO
import math

import mujoco
import numpy as np

from simulate.mujoco.model.three_floor_elevator.web_viewer import encode_png


class RGBDCapture:
    def __init__(self, model, *, width=640, height=480):
        if width < 16 or height < 16:
            raise ValueError('RGBD dimensions must be at least 16 pixels')
        self.model = model
        self.width, self.height = int(width), int(height)
        self.data = mujoco.MjData(model)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot_ego')
        if self.camera_id < 0:
            raise ValueError("MuJoCo model is missing camera 'robot_ego'")
        self.renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.option)
        self.option.geomgroup[:] = (1, 1, 1, 0, 1, 1)
        self.closed = False

    def capture(self, data):
        if self.closed:
            raise RuntimeError('RGBD renderer is closed')
        # Rendering uses an isolated data snapshot and never advances physics.
        self.data.qpos[:] = data.qpos
        self.data.qvel[:] = data.qvel
        if self.model.nmocap:
            self.data.mocap_pos[:] = data.mocap_pos
            self.data.mocap_quat[:] = data.mocap_quat
        self.data.time = data.time
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera='robot_ego', scene_option=self.option)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data, camera='robot_ego', scene_option=self.option)
            depth_buffer = np.asarray(self.renderer.render(), dtype=np.float32).copy()
        finally:
            self.renderer.disable_depth_rendering()
        # MuJoCo depth rendering returns metric distance from the camera plane,
        # which is optical-axis Z.
        fovy = math.radians(float(self.model.cam_fovy[self.camera_id]))
        f = self.height / (2 * math.tan(fovy / 2))
        cx, cy = (self.width - 1) / 2, (self.height - 1) / 2
        depth = depth_buffer.astype(np.float32)
        mask = np.isfinite(depth) & (depth > 0)
        depth[~mask] = 0
        camera_rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        position = self.data.cam_xpos[self.camera_id].copy()
        # MuJoCo camera stores wxyz; convert basis with the camera-axis flip.
        quat_mj = np.zeros(4)
        mujoco.mju_mat2Quat(quat_mj, camera_rotation.ravel())
        flip_quat = np.array((0., 1., 0., 0.))
        quat_optical = np.zeros(4)
        mujoco.mju_mulQuat(quat_optical, quat_mj, flip_quat)
        metadata = dict(width=self.width, height=self.height, depth_unit='m',
            depth_convention='optical_z', capture_time=float(data.time),
            camera_name='robot_ego', camera_position_w=position.tolist(),
            camera_orientation_wxyz=quat_optical.tolist(),
            camera_axes='x_right,y_down,z_forward',
            intrinsics=[[f, 0., cx], [0., f, cy], [0., 0., 1.]],
            depth_visualization=dict(min_m=.1, max_m=8., colormap='grayscale'))
        raw = BytesIO()
        np.save(raw, depth, allow_pickle=False)
        mask_bytes = np.packbits(mask, axis=None).tobytes()
        visual = (np.clip((depth - .1) / (8. - .1), 0, 1) * 255).astype(np.uint8)
        visual_rgb = np.repeat(visual[:, :, None], 3, axis=2)
        blobs = {
            'rgb': (encode_png(rgb), 'image/png', dict(width=self.width, height=self.height, color_space='RGB')),
            'depth_visualization': (encode_png(visual_rgb), 'image/png', dict(width=self.width, height=self.height)),
            'depth': (raw.getvalue(), 'application/x-npy', dict(width=self.width, height=self.height, dtype='float32', unit='m', convention='optical_z')),
            'depth_mask': (mask_bytes, 'application/octet-stream', dict(width=self.width, height=self.height, dtype='bit-packed', bit_order='little')),
        }
        return metadata, blobs

    def close(self):
        if not self.closed:
            self.renderer.close()
            self.closed = True
