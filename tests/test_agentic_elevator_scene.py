"""Physical composition and adapter regressions, without policy checkpoints."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def composed(configure_spec=None):
    from mjlab.scene import Scene
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    from src.agentic_robot.scene import configure_elevator
    cfg = transition_env_cfg(play=True)
    configure_elevator(cfg)
    if configure_spec is not None:
        original = cfg.scene.spec_fn
        def configure(spec):
            original(spec)
            configure_spec(spec)
        cfg.scene.spec_fn = configure
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return cfg, model, data


def test_composition_has_one_robot_scene_surfaces_and_camera():
    cfg, model, data = composed()
    from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController
    assert cfg.scene.terrain is None
    assert cfg.scene.num_envs == 1
    assert not any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert model.body('robot/base_link').id > 0
    assert model.camera('robot_ego').id >= 0
    for name in ThreeFloorElevatorController.BUTTON_NAMES:
        site = model.site(f'surface_{name}').id
        assert np.isfinite(data.site_xpos[site]).all()
    feet = next(s for s in cfg.scene.sensors if s.name == 'feet_ground_contact')
    assert feet.secondary.mode == 'subtree'
    assert feet.secondary.pattern == 'elevator_scene'
    assert model.body('elevator_car').parentid[0] == model.body('elevator_scene').id
    assert model.body('room_f3').parentid[0] == model.body('elevator_scene').id
    assert cfg.rewards == {}
    assert cfg.metrics == {}


class Array:
    """CPU stand-in for indexed Warp/Torch arrays, exercising actual controller."""
    def __init__(self, value):
        self.value = value
    def __getitem__(self, key):
        import torch
        return torch.as_tensor(self.value[key])
    def __setitem__(self, key, value):
        self.value[key] = np.asarray(value)


def native_bridge(model, data):
    fields = ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'qfrc_bias', 'qacc_warmstart')
    arrays = {name: Array(getattr(data, name)[None]) for name in fields}
    arrays['time'] = Array(np.array([data.time]))
    return SimpleNamespace(mj_model=model, data=SimpleNamespace(**arrays))


def test_idle_scene_keeps_robot_state_while_light_lift_and_doors_run():
    _, model, data = composed()
    from src.agentic_robot.elevator_adapter import ElevatorAdapter
    sim = native_bridge(model, data)
    sim.forward = lambda: mujoco.mj_forward(model, data)
    adapter = ElevatorAdapter(sim)
    adapter.reset()
    # Include nonzero velocity/warm-start values to check exact state holding.
    joint = model.joint('robot/floating_base_joint').id
    dof = model.jnt_dofadr[joint]
    data.qvel[dof] = .12
    data.qacc_warmstart[dof] = .34
    # Go2 has a free base (7 positions/6 velocities) and 12 hinge joints;
    # scene button joints can appear before the lift joint in the model.
    robot_qpos = data.qpos[:19].copy()
    robot_qvel = data.qvel[:18].copy()
    robot_warmstart = data.qacc_warmstart[:18].copy()
    adapter.controller.mouse_press('light_f1')
    adapter.controller.mouse_press('call_up_f1')

    def step():
        mujoco.mj_step(model, data)
        sim.data.time.value[0] = data.time

    for _ in range(round(12. / model.opt.timestep)):
        adapter.step_scene(step)
    np.testing.assert_array_equal(data.qpos[:len(robot_qpos)], robot_qpos)
    np.testing.assert_array_equal(data.qvel[:len(robot_qvel)], robot_qvel)
    np.testing.assert_array_equal(data.qacc_warmstart[:len(robot_warmstart)], robot_warmstart)
    assert data.time == pytest.approx(12.)
    assert not model.light_active[adapter.controller.lamp_ids[1]]
    assert adapter.controller.current_floor == 1
    assert not adapter.controller.motion_active
    for adr in adapter.controller.door_qposadr.values():
        assert data.qpos[adr] == pytest.approx(.65, abs=.02)


def test_adapter_reset_preserves_robot_and_writes_only_scene_indices():
    _, model, data = composed()
    from src.agentic_robot.elevator_adapter import ElevatorAdapter
    sim = native_bridge(model, data)
    robot_joint = model.joint('robot/floating_base_joint').id
    adr = model.jnt_qposadr[robot_joint]
    data.qpos[adr:adr + 3] = (3.9, -3.0, .42)
    before = data.qpos[adr:adr + 7].copy()
    data.qfrc_applied[model.jnt_dofadr[robot_joint]] = 3.25
    adapter = ElevatorAdapter(sim)
    adapter.reset()
    np.testing.assert_array_equal(data.qpos[adr:adr + 7], before)
    assert data.qfrc_applied[model.jnt_dofadr[robot_joint]] == 3.25
    assert data.qpos[adapter.controller.lift_qposadr] == pytest.approx(5.6)
    assert data.ctrl[adapter.controller.lift_actuator_id] == pytest.approx(5.6)
    adapter.before_step()
    adapter.after_step()
    assert data.qfrc_applied[model.jnt_dofadr[robot_joint]] == 3.25


def test_surface_tracks_physical_button_travel_and_car_motion():
    _, model, data = composed()
    initial = data.site('surface_call_up_f1').xpos.copy()
    j = model.joint('joint_call_up_f1').id
    direction = data.xaxis[j].copy()
    data.qpos[model.jnt_qposadr[j]] = .02
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site('surface_call_up_f1').xpos, initial + .02 * direction)
    cab = data.site('surface_cab_f1').xpos.copy()
    lift = model.jnt_qposadr[model.joint('lift_z').id]
    data.qpos[lift] -= 2.8
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site('surface_cab_f1').xpos, cab - (0, 0, 2.8))


def test_reset_refreshes_robot_and_camera_pose_before_initial_observation():
    _, model, data = composed()
    from src.agentic_robot.elevator_adapter import ElevatorAdapter
    sim = native_bridge(model, data)
    joint = model.joint('robot/floating_base_joint').id
    adr = model.jnt_qposadr[joint]
    data.qpos[adr:adr + 3] = (3.9, -3., .42)
    adapter = ElevatorAdapter(sim)
    adapter.reset()
    np.testing.assert_allclose(adapter.data.body('robot/base_link').xpos, (3.9, -3., .42))
    # Capture metadata and render pose must originate from the same refreshed world.
    expected = mujoco.MjData(model)
    expected.qpos[:] = data.qpos
    mujoco.mj_forward(model, expected)
    np.testing.assert_allclose(adapter.data.camera('robot_ego').xpos,
        expected.camera('robot_ego').xpos)


def test_first_floor_call_button_is_within_biped_reach_and_keeps_full_travel():
    _, model, data = composed()
    surface = model.site('surface_call_up_f1').id
    joint = model.joint('joint_call_up_f1').id
    initial = data.site_xpos[surface].copy()
    assert initial[2] == pytest.approx(.62, abs=.01)

    data.qpos[model.jnt_qposadr[joint]] = .02
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[surface], initial + .02 * data.xaxis[joint])


@pytest.mark.parametrize('button', [
    'light_f1', 'light_f2', 'light_f3', 'call_up_f1', 'call_up_f2',
    'call_down_f2', 'call_down_f3', 'cab_f1', 'cab_f2', 'cab_f3',
])
def test_composed_adapter_detects_real_foot_contact_on_every_button(button):
    """Foot-sized physical collider; no mouse force or logical press shortcut."""
    floor = int(button[-1])
    height = .62 if button == 'call_up_f1' else .85
    floor_z = 5.6 if button.startswith('cab_') else (floor - 1) * 2.8
    if button.startswith('light_'):
        position, force = (-2.14, 1.65, floor_z + height - .07), (0, 12, 0)
    elif button.startswith('call_'):
        x = {'call_up_f2': 4.1, 'call_down_f2': 4.45}.get(button, 4.25)
        position, force = (x + .06, .85, floor_z + height - .07), (0, 12, 0)
    else:
        y = {1: -.87, 2: -.4, 3: .07}[floor]
        position, force = (6.65, y + .06, floor_z + height - .07), (12, 0, 0)
    def add_foot(spec):
        foot = spec.worldbody.add_body(name='test_contact_foot', pos=position)
        foot.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        foot.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.025,), mass=.5)
    _, model, data = composed(add_foot)
    from src.agentic_robot.elevator_adapter import ElevatorAdapter
    sim = native_bridge(model, data)
    sim.forward = lambda: mujoco.mj_forward(model, data)
    adapter = ElevatorAdapter(sim)
    adapter.reset()
    body = model.body('test_contact_foot').id
    max_travel = 0.
    def step():
        data.xfrc_applied[body, :3] = force
        data.xfrc_applied[body, 2] = -.5 * model.opt.gravity[2]
        mujoco.mj_step(model, data)
        sim.data.time.value[0] = data.time
    for _ in range(200):
        adapter.step_scene(step)
        max_travel = max(max_travel, data.qpos[adapter.controller.button_qposadr[button]])
        if adapter.events:
            break
    assert max_travel >= adapter.controller.PRESS_THRESHOLD
    assert [e['button'] for e in adapter.events] == [button]
    if button.startswith('light_'):
        assert not model.light_active[adapter.controller.lamp_ids[floor]]
    elif floor != 3:
        assert adapter.controller.current_target == floor
        assert adapter.controller.motion_active or adapter.controller.doors_closing
