"""Physical composition and adapter regressions, without policy checkpoints."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def composed():
    from mjlab.scene import Scene
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    from src.agentic_robot.scene import configure_elevator
    cfg = transition_env_cfg(play=True)
    configure_elevator(cfg)
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
    fields = ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'qfrc_bias')
    arrays = {name: Array(getattr(data, name)[None]) for name in fields}
    arrays['time'] = Array(np.array([data.time]))
    return SimpleNamespace(mj_model=model, data=SimpleNamespace(**arrays))


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


def test_first_floor_call_button_is_within_biped_reach_and_keeps_full_travel():
    _, model, data = composed()
    surface = model.site('surface_call_up_f1').id
    joint = model.joint('joint_call_up_f1').id
    initial = data.site_xpos[surface].copy()
    assert initial[2] == pytest.approx(.62, abs=.01)

    data.qpos[model.jnt_qposadr[joint]] = .02
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[surface], initial + .02 * data.xaxis[joint])
