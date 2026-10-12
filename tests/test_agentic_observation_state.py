"""Small physical fixtures for observation contracts (no policy/checkpoint load)."""
import json

import mujoco
import numpy as np
import pytest

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController


def make_model():
    buttons = ''.join(f'''<body name="button_{name}" pos="{i * .15} 3 1">
      <joint name="joint_{name}" type="slide" axis="0 1 0"/>
      <geom type="sphere" size=".04"/><site name="surface_{name}" pos="0 -.05 0"/>
    </body>''' for i, name in enumerate(ThreeFloorElevatorController.BUTTON_NAMES))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><visual>
      <global offwidth="160" offheight="120"/><headlight active="0"/></visual>
      <worldbody><light name="lamp" pos="0 0 2" dir="0 0 -1" ambient="0 0 0" diffuse="1 1 1"/>
      <body name="robot/base_link" pos="0 0 1"><freejoint/>
      <geom type="sphere" size=".05"/><site name="robot/FR" pos=".2 -.1 -.3"/>
      <camera name="robot_ego" pos="0 0 .1" fovy="70"/>
      </body><geom name="plane" type="plane" size="5 5 .01" rgba=".8 .6 .4 1"/>
      <body name="car"><joint name="lift" type="slide" axis="0 0 1"/>
      <geom type="sphere" size=".02"/>{buttons}</body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_anchor_state_and_world_command_roundtrip(tmp_path):
    from src.agentic_robot.observation import ObservationService, world_to_command
    model, data = make_model()
    service = ObservationService(model, tmp_path, fr_zero_anchor=[.2, -.1, -.3], capture_images=False)
    service.reset('episode', data, fsm={'owner': 'QUAD'})
    robot = service.observe(data, fsm={'owner': 'QUAD'})['robot']
    np.testing.assert_allclose(robot['anchor']['R_world_from_anchor'], np.eye(3))
    np.testing.assert_allclose(world_to_command(robot['fr_position_w'], robot['anchor']), 0, atol=1e-15)
    assert robot['anchor']['contract'] == 'teacher_body_y_cross_up_v1'
    assert robot['fsm']['owner'] == 'QUAD'
    # Rear-leg stance must follow body -Z, rather than horizontal body +X.
    data.qpos[3:7] = [np.sqrt(.5), 0, -np.sqrt(.5), 0]
    mujoco.mj_forward(model, data)
    robot = service.observe(data, fsm={})['robot']
    np.testing.assert_allclose(np.array(robot['anchor']['R_world_from_anchor'])[:, 0], [1, 0, 0], atol=1e-12)
    target = [2, 3, 4]
    command = world_to_command(target, robot['anchor'])
    anchor = robot['anchor']
    np.testing.assert_allclose(np.array(anchor['origin_w']) + np.array(anchor['R_world_from_anchor']) @ (command + anchor['fr_zero_anchor']), target)


def test_controls_follow_moving_parent_and_plunger(tmp_path):
    from src.agentic_robot.observation import ObservationService
    model, data = make_model()
    service = ObservationService(model, tmp_path, fr_zero_anchor=[0, 0, 0], capture_images=False)
    service.reset('episode', data, fsm={})
    before = service.observe(data, fsm={})
    data.qpos[model.jnt_qposadr[model.joint('lift').id]] = 2.8
    data.qpos[model.jnt_qposadr[model.joint('joint_cab_f1').id]] = .02
    mujoco.mj_forward(model, data)
    after = service.observe(data, fsm={})
    assert len(after['controls']) == 10
    a = {c['id']: c for c in before['controls']}['cab_f1']
    b = {c['id']: c for c in after['controls']}['cab_f1']
    np.testing.assert_allclose(np.subtract(b['surface_position_w'], a['surface_position_w']), [0, .02, 2.8])
    assert b['target_floor'] == 1 and 'floor' not in b
    assert b['position_source'] == 'sim_ground_truth'
    np.testing.assert_allclose(b['normal_out_w'], [0, -1, 0])
    np.testing.assert_allclose(b['press_direction_w'], [0, 1, 0])
    json.dumps(after, allow_nan=False)


def test_missing_names_reported_together(tmp_path):
    from src.agentic_robot.observation import ObservationService
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    with pytest.raises(ValueError, match='robot/base_link.*robot/FR.*robot_ego.*surface_light_f1'):
        ObservationService(model, tmp_path, fr_zero_anchor=[0, 0, 0])


def test_anchor_degenerate_and_arbitrary_gravity():
    from src.agentic_robot.observation.state import anchor_basis
    # Body +Y points up: use the same fixed +X fallback as the teacher.
    np.testing.assert_allclose(anchor_basis([np.sqrt(.5), np.sqrt(.5), 0, 0], [0, 0, -9.81]), np.eye(3), atol=1e-12)
    basis = anchor_basis([1, 0, 0, 0], [-9.81, 0, 0])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(basis[:, 2], [1, 0, 0])
