"""Robot-independent interaction logic for the three-floor elevator scene."""

from __future__ import annotations

from collections import deque

import mujoco


class ThreeFloorElevatorController:
    """Read physical button travel and control lamps, lift, and car doors."""

    FLOOR_Z = {1: 0.0, 2: 2.8, 3: 5.6}
    PRESS_THRESHOLD = 0.016
    RELEASE_THRESHOLD = 0.006
    BUTTON_COOLDOWN = 0.15
    MOUSE_PRESS_DURATION = 0.15
    MOUSE_PRESS_FORCE = 15.0
    LIFT_SPEED = 1.5
    LIFT_POSITION_TOLERANCE = 0.02
    LIFT_VELOCITY_TOLERANCE = 0.03
    DOOR_POSITION_TOLERANCE = 0.02
    DOOR_VELOCITY_TOLERANCE = 0.03
    DOOR_CLOSED = 0.0
    DOOR_OPEN = 0.65
    DOOR_DWELL = 2.0

    BUTTON_NAMES = (
        "light_f1",
        "light_f2",
        "light_f3",
        "call_up_f1",
        "call_up_f2",
        "call_down_f2",
        "call_down_f3",
        "cab_f1",
        "cab_f2",
        "cab_f3",
    )

    CALL_INDICATORS = {
        "call_up_f1": "indicator_call_up_f1",
        "call_up_f2": "indicator_call_up_f2",
        "call_down_f2": "indicator_call_down_f2",
        "call_down_f3": "indicator_call_down_f3",
    }

    INDICATOR_ACTIVE_RGBA = (0.12, 0.82, 0.32, 1.0)
    INDICATOR_IDLE_RGBA = (0.12, 0.16, 0.18, 1.0)
    LAMP_EMISSION_ON = 0.22
    LAMP_EMISSION_OFF = 0.0

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self.button_qposadr: dict[str, int] = {}
        self.button_dofadr: dict[str, int] = {}
        self.button_joint_ids: dict[str, int] = {}
        for name in self.BUTTON_NAMES:
            joint_id = self._resolve(mujoco.mjtObj.mjOBJ_JOINT, f"joint_{name}")
            self.button_joint_ids[name] = joint_id
            self.button_qposadr[name] = int(model.jnt_qposadr[joint_id])
            self.button_dofadr[name] = int(model.jnt_dofadr[joint_id])

        self.lift_joint_id = self._resolve(mujoco.mjtObj.mjOBJ_JOINT, "lift_z")
        self.lift_qposadr = int(model.jnt_qposadr[self.lift_joint_id])
        self.lift_dofadr = int(model.jnt_dofadr[self.lift_joint_id])
        self._lift_gravity_compensation = 0.0
        self._last_lift_gravity_compensation = 0.0
        self.lift_actuator_id = self._resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_position")
        self.door_joint_ids = {
            side: self._resolve(mujoco.mjtObj.mjOBJ_JOINT, f"joint_car_door_{side}")
            for side in ("front", "back")
        }
        self.door_qposadr = {
            side: int(model.jnt_qposadr[joint_id])
            for side, joint_id in self.door_joint_ids.items()
        }
        self.door_dofadr = {
            side: int(model.jnt_dofadr[joint_id])
            for side, joint_id in self.door_joint_ids.items()
        }
        self.door_actuator_ids = {
            side: self._resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, f"car_door_{side}")
            for side in self.door_joint_ids
        }
        self.lamp_ids = {
            floor: self._resolve(mujoco.mjtObj.mjOBJ_LIGHT, f"lamp_f{floor}")
            for floor in self.FLOOR_Z
        }
        self.diffuser_material_ids = {}
        for floor in self.FLOOR_Z:
            diffuser_id = self._resolve(mujoco.mjtObj.mjOBJ_GEOM, f"diffuser_f{floor}")
            material_id = int(model.geom_matid[diffuser_id])
            if material_id < 0:
                raise ValueError(f"missing material for diffuser_f{floor}")
            self.diffuser_material_ids[floor] = material_id
        self.indicator_geom_ids = {
            button: self._resolve(mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            for button, geom_name in self.CALL_INDICATORS.items()
        }

        self._latched = {name: False for name in self.BUTTON_NAMES}
        self._last_event_time = {name: float("-inf") for name in self.BUTTON_NAMES}
        self._mouse_force_until: dict[str, float] = {}
        self._pending: deque[int] = deque()
        self.current_floor = 3
        self.current_target = 3
        self.motion_active = False
        self.doors_closing = False
        self._motion_goal = self.FLOOR_Z[3]
        self._door_open_since: float | None = -self.DOOR_DWELL

    def _resolve(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing {object_type.name} named {name!r}")
        return int(object_id)

    @property
    def pending_floors(self) -> tuple[int, ...]:
        """Floors waiting behind the active trip, in service order."""

        return tuple(self._pending)

    def reset(self) -> None:
        """Restore the model's authored start state and controller state."""

        self.data.time = 0.0
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self._last_lift_gravity_compensation = 0.0

        self.data.qpos[self.lift_qposadr] = self.FLOOR_Z[3]
        self.data.ctrl[self.lift_actuator_id] = self.FLOOR_Z[3]
        for side, actuator_id in self.door_actuator_ids.items():
            self.data.qpos[self.door_qposadr[side]] = self.DOOR_OPEN
            self.data.ctrl[actuator_id] = self.DOOR_OPEN
        for floor, lamp_id in self.lamp_ids.items():
            self.model.light_active[lamp_id] = True
            self._set_lamp_emission(floor, True)
        for button, geom_id in self.indicator_geom_ids.items():
            self.model.geom_rgba[geom_id] = self.INDICATOR_IDLE_RGBA

        self._pending.clear()
        self._latched = {name: False for name in self.BUTTON_NAMES}
        self._last_event_time = {name: float("-inf") for name in self.BUTTON_NAMES}
        self._mouse_force_until.clear()
        self.current_floor = 3
        self.current_target = 3
        self.motion_active = False
        self.doors_closing = False
        self._motion_goal = self.FLOOR_Z[3]
        self._door_open_since = -self.DOOR_DWELL
        mujoco.mj_forward(self.model, self.data)
        self._apply_lift_gravity_compensation()
        mujoco.mj_forward(self.model, self.data)

    def request_floor(self, floor: int) -> None:
        """Add a valid floor request once, preserving active and queued trips."""

        if floor not in self.FLOOR_Z:
            raise ValueError(f"invalid floor: {floor}")
        if floor != self.current_target and floor not in self._pending:
            self._pending.append(floor)

    def mouse_press(self, button_name: str) -> None:
        """Apply a short physical press to a named plunger joint."""

        if button_name not in self.button_dofadr:
            raise KeyError(button_name)
        self._mouse_force_until[button_name] = self.data.time + self.MOUSE_PRESS_DURATION

    def apply_mouse_forces(self) -> None:
        """Apply only button generalized forces, preserving other user forces."""

        self._apply_lift_gravity_compensation()
        for name, dof_address in self.button_dofadr.items():
            until = self._mouse_force_until.get(name, float("-inf"))
            self.data.qfrc_applied[dof_address] = (
                self.MOUSE_PRESS_FORCE if self.data.time < until else 0.0
            )

    def update(self) -> None:
        """Read new presses, start queued trips, and advance lift/door targets."""

        self._apply_lift_gravity_compensation()
        self._read_button_edges()
        self._start_next_floor_if_idle()
        self._update_motion_and_doors()

    def _read_button_edges(self) -> None:
        now = float(self.data.time)
        for name, qpos_address in self.button_qposadr.items():
            position = float(self.data.qpos[qpos_address])
            if position <= self.RELEASE_THRESHOLD:
                self._latched[name] = False
                continue
            if position < self.PRESS_THRESHOLD or self._latched[name]:
                continue

            self._latched[name] = True
            if now - self._last_event_time[name] < self.BUTTON_COOLDOWN:
                continue
            self._last_event_time[name] = now
            self._handle_button_press(name)

    def _handle_button_press(self, name: str) -> None:
        if name.startswith("light_f"):
            floor = int(name[-1])
            lamp_id = self.lamp_ids[floor]
            is_active = not bool(self.model.light_active[lamp_id])
            self.model.light_active[lamp_id] = is_active
            self._set_lamp_emission(floor, is_active)
            return

        if name in self.CALL_INDICATORS:
            floor = int(name[-1])
            self.request_floor(floor)
            if floor != self.current_floor:
                indicator_id = self.indicator_geom_ids[name]
                self.model.geom_rgba[indicator_id] = self.INDICATOR_ACTIVE_RGBA
            return

        if name.startswith("cab_"):
            self.request_floor(int(name[-1]))
            return

        raise ValueError(f"unsupported button: {name}")

    def _set_lamp_emission(self, floor: int, active: bool) -> None:
        emission = self.LAMP_EMISSION_ON if active else self.LAMP_EMISSION_OFF
        self.model.mat_emission[self.diffuser_material_ids[floor]] = emission

    def _apply_lift_gravity_compensation(self) -> None:
        """Balance the car's weight while preserving other generalized forces."""

        self._lift_gravity_compensation = float(self.data.qfrc_bias[self.lift_dofadr])
        self.data.qfrc_applied[self.lift_dofadr] -= self._last_lift_gravity_compensation
        self.data.qfrc_applied[self.lift_dofadr] += self._lift_gravity_compensation
        self._last_lift_gravity_compensation = self._lift_gravity_compensation

    def _start_next_floor_if_idle(self) -> None:
        if self.motion_active:
            return

        doors_open = all(
            self.data.qpos[self.door_qposadr[side]] >= self.DOOR_OPEN - self.DOOR_POSITION_TOLERANCE
            and abs(self.data.qvel[self.door_dofadr[side]]) <= self.DOOR_VELOCITY_TOLERANCE
            for side in self.door_joint_ids
        )
        if not doors_open:
            self._door_open_since = None
            return
        if self._door_open_since is None:
            self._door_open_since = float(self.data.time)
        # Wait for fully open doors and a boarding interval between trips.
        if not self._pending or self.data.time - self._door_open_since < self.DOOR_DWELL:
            return

        floor = self._pending.popleft()
        if floor == self.current_floor:
            self.current_target = floor
            for actuator_id in self.door_actuator_ids.values():
                self.data.ctrl[actuator_id] = self.DOOR_OPEN
            self._clear_floor_indicators(floor)
            return

        self.current_target = floor
        self._motion_goal = self.FLOOR_Z[floor]
        self.motion_active = True
        self.doors_closing = True
        for actuator_id in self.door_actuator_ids.values():
            self.data.ctrl[actuator_id] = self.DOOR_CLOSED

    def _update_motion_and_doors(self) -> None:
        lift_position = float(self.data.qpos[self.lift_qposadr])
        lift_velocity = float(self.data.qvel[self.lift_dofadr])

        if not self.motion_active:
            self.data.ctrl[self.lift_actuator_id] = self.FLOOR_Z[self.current_floor]
            return

        if self.doors_closing:
            doors_closed = all(
                self.data.qpos[self.door_qposadr[side]] <= self.DOOR_POSITION_TOLERANCE
                and abs(self.data.qvel[self.door_dofadr[side]]) <= self.DOOR_VELOCITY_TOLERANCE
                for side in self.door_joint_ids
            )
            self.data.ctrl[self.lift_actuator_id] = self.FLOOR_Z[self.current_floor]
            if not doors_closed:
                return
            self.doors_closing = False

        if (
            abs(lift_position - self._motion_goal) <= self.LIFT_POSITION_TOLERANCE
            and abs(lift_velocity) <= self.LIFT_VELOCITY_TOLERANCE
        ):
            self.current_floor = self.current_target
            self.motion_active = False
            self.doors_closing = False
            self.data.ctrl[self.lift_actuator_id] = self._motion_goal
            for actuator_id in self.door_actuator_ids.values():
                self.data.ctrl[actuator_id] = self.DOOR_OPEN
            self._door_open_since = None
            self._clear_floor_indicators(self.current_floor)
            return

        maximum_step = self.LIFT_SPEED * self.model.opt.timestep
        target = float(self.data.ctrl[self.lift_actuator_id])
        delta = self._motion_goal - target
        self.data.ctrl[self.lift_actuator_id] = target + max(-maximum_step, min(maximum_step, delta))

    def _clear_floor_indicators(self, floor: int) -> None:
        for button, geom_id in self.indicator_geom_ids.items():
            if int(button[-1]) == floor:
                self.model.geom_rgba[geom_id] = self.INDICATOR_IDLE_RGBA
