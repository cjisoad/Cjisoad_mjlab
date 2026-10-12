"""EGL RGB/depth captures share one robot-mounted camera state."""
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

from io import BytesIO

import mujoco
import numpy as np
import pytest


def model_fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco><visual>
      <global offwidth="160" offheight="120"/><map znear=".01" zfar="20"/></visual>
      <worldbody><light name="key" pos="0 0 5" dir="0 0 -1" diffuse=".8 .8 .8"/>
      <body name="robot/base_link" pos="0 0 0"><freejoint/>
        <geom type="sphere" pos="1 0 0" size=".02"/>
        <camera name="robot_ego" pos="0 0 .5" quat="1 0 0 0" fovy="70"/>
      </body><geom name="plane" type="plane" pos="0 0 -2.5" size="5 5 .1" rgba=".8 .2 .1 1"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def capture():
    from src.agentic_robot.observation.rgbd import RGBDCapture
    model, data = model_fixture()
    try:
        renderer = RGBDCapture(model, width=160, height=120)
    except (mujoco.FatalError, RuntimeError) as exc:
        pytest.skip(f'EGL renderer unavailable: {exc}')
    try:
        yield model, data, renderer
    finally:
        renderer.close()


def test_rgb_depth_intrinsics_and_camera_pose_share_snapshot(capture):
    model, data, renderer = capture
    data.qpos[:3] = [1, 2, 0]
    mujoco.mj_forward(model, data)
    original = data.qpos.copy()
    metadata, blobs = renderer.capture(data)
    assert set(blobs) == {'rgb', 'depth_visualization', 'depth', 'depth_mask'}
    assert metadata['depth_convention'] == 'optical_z'
    assert metadata['camera_axes'] == 'x_right,y_down,z_forward'
    assert np.asarray(metadata['intrinsics']).shape == (3, 3)
    assert metadata['width'] == 160 and metadata['height'] == 120
    from PIL import Image
    rgb = np.asarray(Image.open(BytesIO(blobs['rgb'][0])))
    assert rgb.shape == (120, 160, 3)
    depth = np.load(BytesIO(blobs['depth'][0]), allow_pickle=False)
    assert depth.shape == (120, 160) and depth.dtype == np.float32
    assert depth[60, 80] == pytest.approx(3.0, abs=.08)
    assert np.isfinite(depth).all()
    assert np.array_equal(data.qpos, original)
    assert metadata['camera_position_w'] == pytest.approx([1., 2., .5])


def test_rgb_tracks_light_change_in_same_model(capture):
    model, data, renderer = capture
    _, on = renderer.capture(data)
    light = model.light('key').id
    model.light_active[light] = 0
    _, off = renderer.capture(data)
    from PIL import Image
    a = np.asarray(Image.open(BytesIO(on['rgb'][0]))).astype(float)
    b = np.asarray(Image.open(BytesIO(off['rgb'][0]))).astype(float)
    assert not np.array_equal(a, b)
