from pathlib import Path

import mujoco
import pytest

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "simulate/mujoco/model/three_floor_elevator/three_floor_elevator.xml"


def _object_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type.name}: {name}"
    return object_id


def test_scene_compiles_and_has_fixed_overview_camera():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    _object_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_overview")
    assert model.nbody > 1
    assert model.nbody < 200
    assert model.nu < 20


def test_scene_exposes_three_floors_lights_and_button_groups():
    model = mujoco.MjModel.from_xml_path(str(SCENE))

    for floor in (1, 2, 3):
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"room_f{floor}")
        _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, f"lamp_f{floor}")
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_light_f{floor}")
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_cab_f{floor}")

    for name in (
        "joint_call_up_f1",
        "joint_call_up_f2",
        "joint_call_down_f2",
        "joint_call_down_f3",
        "lift_z",
    ):
        _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    for side in ("front", "back"):
        _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"car_door_{side}")
        _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_car_door_{side}")

    assert model.nlight >= 3
    assert model.nbody < 200


def test_scene_starts_lights_on_and_lift_at_floor_three_with_car_doors_open():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    lift_joint = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_z")
    lift_qpos = model.jnt_qposadr[lift_joint]
    assert data.qpos[lift_qpos] == pytest.approx(5.6)

    for floor in (1, 2, 3):
        lamp_id = _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, f"lamp_f{floor}")
        assert model.light_active[lamp_id]

    for side in ("front", "back"):
        assert _joint_position(model, data, f"joint_car_door_{side}") == pytest.approx(0.65)


def test_lobbies_connect_room_floors_and_doors_belong_to_car():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    car = model.body("elevator_car").id
    for floor in (1, 2, 3):
        lobby = model.geom(f"lobby_floor_f{floor}").id
        room = model.geom(f"floor_f{floor}").id
        assert data.geom_xpos[lobby, 2] + model.geom_size[lobby, 2] == pytest.approx((floor - 1) * 2.8)
        assert data.geom_xpos[lobby, 0] - model.geom_size[lobby, 0] == pytest.approx(3.0)
        assert data.geom_xpos[room, 0] + model.geom_size[room, 0] == pytest.approx(3.0)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"gate_body_f{floor}") == -1
    for side in ("front", "back"):
        assert model.body(f"car_door_body_{side}").parentid[0] == car


def test_car_has_clear_vertical_travel_past_lobby_edges(scene):
    import numpy as np
    model, data, controller = scene
    car = model.body("elevator_car").id
    car_bodies = {car}
    for body in range(model.nbody):
        if int(model.body_parentid[body]) in car_bodies:
            car_bodies.add(body)
    for height in np.linspace(0, 5.6, 113):
        data.qpos[controller.lift_qposadr] = height
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            on_car = [int(model.geom_bodyid[g]) in car_bodies for g in (contact.geom1, contact.geom2)]
            assert not (any(on_car) and not all(on_car) and contact.dist < 0), (
                height, model.geom(contact.geom1).name, model.geom(contact.geom2).name
            )


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    return model, data, controller


def _set_joint_position(model, data, joint_name, position):
    joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = position


def _joint_position(model, data, joint_name):
    joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return data.qpos[model.jnt_qposadr[joint_id]]


def test_controller_reset_restores_lights_lift_and_car_doors(scene):
    model, data, controller = scene
    assert controller.current_floor == 3
    assert not controller.motion_active
    assert controller.pending_floors == ()
    for side in ("front", "back"):
        assert _joint_position(model, data, f"joint_car_door_{side}") == pytest.approx(0.65)
    for floor in (1, 2, 3):
        lamp_id = _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, f"lamp_f{floor}")
        assert model.light_active[lamp_id]
    assert _joint_position(model, data, "lift_z") == pytest.approx(5.6)


def test_initial_platform_and_open_car_doors_are_physically_aligned(scene):
    model, data, _ = scene
    platform = model.geom("lift_platform").id
    floor = model.geom("floor_f3").id
    platform_top = data.geom_xpos[platform, 2] + model.geom_size[platform, 2]
    floor_top = data.geom_xpos[floor, 2] + model.geom_size[floor, 2]
    assert platform_top == pytest.approx(floor_top, abs=0.001)
    assert data.geom_xpos[model.geom("geom_car_door_front").id, 1] < -1.1
    assert data.geom_xpos[model.geom("geom_car_door_back").id, 1] > 0.7


def test_queued_trips_open_car_doors_before_departing(scene):
    model, data, controller = scene
    controller.request_floor(1)
    controller.request_floor(2)
    reached_first = False
    open_since = None
    for _ in range(20000):
        controller.apply_mouse_forces()
        mujoco.mj_step(model, data)
        controller.update()
        if controller.current_floor == 1:
            reached_first = True
            if min(_joint_position(model, data, f"joint_car_door_{s}") for s in ("front", "back")) > 0.63 and open_since is None:
                open_since = data.time
            if controller.motion_active:
                assert open_since is not None, "queued trip departed without opening the gate"
                assert data.time - open_since >= 1.0
                break
    else:
        pytest.fail("queued trip never departed floor one")
    assert reached_first


@pytest.mark.parametrize("payload_mass", [0.0, 15.0])
def test_middle_and_upward_trips_align_platform_with_payload(payload_mass):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    if payload_mass:
        payload = spec.worldbody.add_body(name="test_payload", pos=(5.9, -0.2, 5.8))
        payload.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        payload.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.2, 0.2, 0.1), mass=payload_mass)
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    for destination in (2, 3):
        controller.request_floor(destination)
        peak_speed = 0.0
        for _ in range(12500):
            controller.apply_mouse_forces()
            mujoco.mj_step(model, data)
            controller.update()
            peak_speed = max(peak_speed, abs(float(data.qvel[controller.lift_dofadr])))
            if controller.current_floor == destination and not controller.motion_active:
                break
        assert controller.current_floor == destination
        assert peak_speed <= 1.6
        mujoco.mj_forward(model, data)
        platform_top = data.geom_xpos[model.geom("lift_platform").id, 2] + 0.09
        assert platform_top == pytest.approx(controller.FLOOR_Z[destination], abs=0.02)
        if payload_mass:
            assert abs(data.body("test_payload").xpos[2] - platform_top - 0.1) < 0.02


def test_closed_landing_has_no_unprotected_rear_gap(scene):
    import numpy as np
    model, data, _ = scene
    # mj_ray skips alpha-zero geoms; expose the collision boundary for probing.
    model.geom_rgba[model.geom("collision_right_back_f1").id, 3] = 1.0
    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([2.8, 1.6, 0.5]),
                             np.array([1., 0., 0.]), None, True, -1, geom_id)
    assert 0.0 < distance < 0.2
    assert model.geom(int(geom_id[0])).name == "collision_right_back_f1"


def test_first_floor_entry_room_gap_has_a_low_collision_barrier():
    spec = mujoco.MjSpec.from_file(str(SCENE))
    probe = spec.worldbody.add_body(name="corner_probe")
    probe.add_joint(name="corner_probe_joint", type=mujoco.mjtJoint.mjJNT_FREE)
    probe.add_geom(name="corner_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                   size=(0.003, 0, 0), mass=0.01)
    model = spec.compile()
    data = mujoco.MjData(model)
    probe_id = model.geom("corner_probe_geom").id
    adr = model.jnt_qposadr[model.joint("corner_probe_joint").id]
    # Seal the entry pad's room-side edge from the room front to the guard.
    # Even a 6 mm sphere must meet a physical barrier throughout this gap.
    for x in (2.98, 2.99, 3.0):
        for y in (-1.95, -1.8, -1.65, -1.59):
            for z in (0.15, 0.55, 1.05):
                data.qpos[adr:adr + 3] = (x, y, z)
                mujoco.mj_forward(model, data)
                assert any(probe_id in (c.geom1, c.geom2) and c.dist < 0
                           for c in data.contact), (x, y, z)

    # The new barrier ends at guard height, without columns up to the ceiling.
    for position in ((3.0, -1.8, 1.5), (3.0, -1.6, 2.3), (3.0, 1.2, 2.3)):
        data.qpos[adr:adr + 3] = position
        mujoco.mj_forward(model, data)
        assert not any(probe_id in (c.geom1, c.geom2) and c.dist < 0
                       for c in data.contact), position

    # The regular outside entrance and room/lobby passage remain traversable.
    spec.geom("corner_probe_geom").size = (0.2, 0, 0)
    model = spec.compile()
    data = mujoco.MjData(model)
    probe_id = model.geom("corner_probe_geom").id
    adr = model.jnt_qposadr[model.joint("corner_probe_joint").id]
    for position in ((3.91, -1.56, 0.4), (3.0, -0.2, 0.4)):
        data.qpos[adr:adr + 3] = position
        mujoco.mj_forward(model, data)
        assert not any(probe_id in (c.geom1, c.geom2) and c.dist < 0
                       for c in data.contact), position


def test_idle_lift_holds_its_floor_against_gravity(scene):
    model, data, controller = scene
    for _ in range(1000):
        controller.apply_mouse_forces()
        mujoco.mj_step(model, data)
        controller.update()

    assert _joint_position(model, data, "lift_z") == pytest.approx(5.6, abs=0.02)


def test_light_button_toggles_its_lamp_once_until_released(scene):
    model, data, controller = scene
    lamp1 = _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, "lamp_f1")
    lamp2 = _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, "lamp_f2")

    _set_joint_position(model, data, "joint_light_f1", 0.020)
    controller.update()
    controller.update()
    assert not model.light_active[lamp1]
    assert model.light_active[lamp2]

    _set_joint_position(model, data, "joint_light_f1", 0.0)
    controller.update()
    data.time = 0.5
    _set_joint_position(model, data, "joint_light_f1", 0.020)
    controller.update()
    assert model.light_active[lamp1]


def test_call_closes_both_car_doors_before_starting_lift(scene):
    model, data, controller = scene
    lift_actuator = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_position")

    _set_joint_position(model, data, "joint_call_up_f1", 0.020)
    controller.update()

    assert controller.current_target == 1
    assert controller.motion_active
    assert controller.doors_closing
    assert data.ctrl[lift_actuator] == pytest.approx(5.6)
    for side in ("front", "back"):
        assert data.ctrl[model.actuator(f"car_door_{side}").id] == pytest.approx(0.0)
        assert _joint_position(model, data, f"joint_car_door_{side}") == pytest.approx(0.65)

    for _ in range(100):
        mujoco.mj_step(model, data)
        controller.update()
        assert data.ctrl[lift_actuator] == pytest.approx(5.6)


def test_floor_requests_are_deduplicated_and_served_in_order(scene):
    _, _, controller = scene
    controller.request_floor(1)
    controller.request_floor(2)
    controller.request_floor(1)
    assert controller.pending_floors == (1, 2)


def test_elevator_carries_closed_doors_then_opens_at_destination(scene):
    model, data, controller = scene
    controller.request_floor(1)

    for _ in range(6500):
        controller.apply_mouse_forces()
        mujoco.mj_step(model, data)
        controller.update()
        if controller.motion_active and not controller.doors_closing:
            for side in ("front", "back"):
                assert _joint_position(model, data, f"joint_car_door_{side}") < 0.02
        if controller.current_floor == 1 and not controller.motion_active:
            break

    assert controller.current_floor == 1
    assert _joint_position(model, data, "lift_z") == pytest.approx(0.0, abs=0.02)
    for _ in range(2500):
        mujoco.mj_step(model, data)
        controller.update()
        if min(_joint_position(model, data, f"joint_car_door_{s}") for s in ("front", "back")) > 0.63:
            break
    mujoco.mj_forward(model, data)
    for side in ("front", "back"):
        assert _joint_position(model, data, f"joint_car_door_{side}") > 0.63
        assert data.geom_xpos[model.geom(f"geom_car_door_{side}").id, 2] == pytest.approx(1.12, abs=0.02)


def _scene_with_test_pusher():
    spec = mujoco.MjSpec.from_file(str(SCENE))
    pusher = spec.worldbody.add_body(name="test_pusher", pos=(-2.20, 1.35, 0.85))
    pusher.add_joint(name="test_pusher_free", type=mujoco.mjtJoint.mjJNT_FREE)
    pusher.add_geom(
        name="test_pusher_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.05,),
        mass=0.5,
        friction=(1.0, 0.005, 0.0001),
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    return model, data, controller


def test_physical_contact_depresses_the_same_light_button():
    model, data, controller = _scene_with_test_pusher()
    lamp_id = _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, "lamp_f1")
    pusher_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "test_pusher")
    max_button_position = 0.0

    for _ in range(1000):
        data.xfrc_applied[pusher_id, 1] = 25.0
        mujoco.mj_step(model, data)
        max_button_position = max(max_button_position, _joint_position(model, data, "joint_light_f1"))
        controller.update()
        if not model.light_active[lamp_id]:
            break

    assert max_button_position > controller.PRESS_THRESHOLD
    assert not model.light_active[lamp_id]


@pytest.mark.parametrize("button", ThreeFloorElevatorController.BUTTON_NAMES)
def test_raised_robot_foot_can_press_lower_off_center_button_face(button):
    """A foot-sized contact at a comfortable reach must move the real plunger."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    floor = int(button[-1])
    height = 0.62 if button == "call_up_f1" else 0.85
    floor_z = 5.6 if button.startswith("cab_") else (floor - 1) * 2.8
    if button.startswith("light_"):
        position, force = (-2.14, 1.65, floor_z + height - 0.07), (0, 12, 0)
    elif button.startswith("call_"):
        x = {"call_up_f2": 4.1, "call_down_f2": 4.45}.get(button, 4.25)
        position, force = (x + 0.06, 0.85, floor_z + height - 0.07), (0, 12, 0)
    else:
        y = {1: -0.87, 2: -0.4, 3: 0.07}[floor]
        position, force = (6.65, y + 0.06, floor_z + height - 0.07), (12, 0, 0)
    pusher = spec.worldbody.add_body(name="raised_foot", pos=position)
    pusher.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    pusher.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.025,), mass=0.5)
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    pressed = []
    handle_press = controller._handle_button_press

    def record_press(name):
        pressed.append(name)
        handle_press(name)

    controller._handle_button_press = record_press
    foot = model.body("raised_foot").id
    for _ in range(200):
        # Represent the robot supporting its foot against gravity while pressing.
        data.xfrc_applied[foot, :3] = force
        data.xfrc_applied[foot, 2] = -0.5 * model.opt.gravity[2]
        mujoco.mj_step(model, data)
        controller.update()
        if pressed:
            break
    assert pressed == [button]


def test_mouse_press_uses_the_same_physical_button_and_event_path(scene):
    model, data, controller = scene
    lamp_id = _object_id(model, mujoco.mjtObj.mjOBJ_LIGHT, "lamp_f1")
    button_qpos = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_light_f1")
    qpos_address = model.jnt_qposadr[button_qpos]
    max_button_position = 0.0
    controller.mouse_press("light_f1")

    for _ in range(100):
        controller.apply_mouse_forces()
        mujoco.mj_step(model, data)
        max_button_position = max(max_button_position, float(data.qpos[qpos_address]))
        controller.update()
        if not model.light_active[lamp_id]:
            break

    assert max_button_position > controller.PRESS_THRESHOLD
    assert not model.light_active[lamp_id]


@pytest.mark.parametrize("button,position,force", [
    ("call_up_f1", (4.25, 0.80, 0.62), (0, 25, 0)),
    ("cab_f1", (6.65, -0.87, 6.45), (25, 0, 0)),
])
def test_robot_side_contact_can_press_call_and_cabin_buttons(button, position, force):
    spec = mujoco.MjSpec.from_file(str(SCENE))
    pusher = spec.worldbody.add_body(name="call_test_pusher", pos=position)
    pusher.add_joint(name="call_test_pusher_free", type=mujoco.mjtJoint.mjJNT_FREE)
    pusher.add_geom(
        name="call_test_pusher_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.05,),
        mass=0.5,
        friction=(1.0, 0.005, 0.0001),
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    pusher_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "call_test_pusher")
    max_button_position = 0.0

    for _ in range(1000):
        data.xfrc_applied[pusher_id, :3] = force
        mujoco.mj_step(model, data)
        max_button_position = max(max_button_position, _joint_position(model, data, "joint_" + button))
        controller.update()
        if controller.current_target == 1 and controller.motion_active:
            break

    assert max_button_position > controller.PRESS_THRESHOLD
    assert controller.current_target == 1
    assert controller.motion_active


def test_mouse_picker_maps_selected_button_geom_to_its_logical_name(scene, monkeypatch):
    from simulate.mujoco.model.three_floor_elevator.viewer import pick_button

    model, data, _ = scene
    scene_stub = object()
    option = object()
    selected_geom = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_cab_f2")

    def select(_model, _data, _option, _aspect, _relx, _rely, _scene, _point, geomid, _flexid, _skinid):
        geomid[0] = selected_geom
        return 1

    monkeypatch.setattr(mujoco, "mjv_select", select)
    assert pick_button(model, data, scene_stub, option, 16 / 9, 0.5, 0.5) == "cab_f2"

    selected_geom = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "mark_cab_f2_a")
    assert pick_button(model, data, scene_stub, option, 16 / 9, 0.5, 0.5) == "cab_f2"

    selected_geom = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_car_door_front")
    assert pick_button(model, data, scene_stub, option, 16 / 9, 0.5, 0.5) is None

    def select_nothing(_model, _data, _option, _aspect, _relx, _rely, _scene, _point, geomid, _flexid, _skinid):
        geomid[0] = -1
        return -1

    monkeypatch.setattr(mujoco, "mjv_select", select_nothing)
    assert pick_button(model, data, scene_stub, option, 16 / 9, 0.95, 0.05) is None


def test_external_entry_button_calls_the_elevator_from_floor_one(scene):
    model, data, controller = scene
    assert model.geom("entry_pad").size[0] >= 0.8
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_entry_call_up_f1") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "entry_call_panel") == -1
