"""MuJoCo EGL renderers must stay on the thread that owns their contexts."""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

from src.agentic_robot.observation.render_worker import RenderWorker


def test_rgbd_and_scene_renderers_share_one_thread_across_callers():
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual>
      <global offwidth="64" offheight="64"/></visual><worldbody>
      <light pos="0 0 2" dir="0 0 -1"/>
      <body><freejoint/><camera name="robot_ego" pos="0 0 .5"/>
      <geom type="sphere" size=".1"/></body>
      <geom type="plane" size="2 2 .1"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    worker = RenderWorker()

    def create_renderers():
        from src.agentic_robot.observation.rgbd import RGBDCapture
        scene = mujoco.Renderer(model, width=64, height=64)
        rgbd = RGBDCapture(model, width=64, height=64)
        return scene, rgbd

    try:
        try:
            scene, rgbd = worker.call(create_renderers)
        except (mujoco.FatalError, RuntimeError) as exc:
            pytest.skip(f'EGL renderer unavailable: {exc}')

        def render(which):
            if which == 'scene':
                scene.update_scene(data)
                return scene.render().shape
            _metadata, blobs = rgbd.capture(data)
            return np.load(__import__('io').BytesIO(blobs['depth'][0]), allow_pickle=False).shape

        with ThreadPoolExecutor(max_workers=2) as callers:
            results = list(callers.map(lambda index: worker.call(render, 'scene' if index % 2 == 0 else 'rgbd'), range(8)))
        assert results == [(64, 64, 3), (64, 64)] * 4
        worker.call(scene.close)
        worker.call(rgbd.close)
    finally:
        worker.close()
