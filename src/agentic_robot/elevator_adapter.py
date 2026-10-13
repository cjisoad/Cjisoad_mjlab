"""Reuse scene logic against authoritative mjlab arrays at every substep."""
from __future__ import annotations

import mujoco
import numpy as np

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController


def cpu_array(value):
    """Copy both Warp tensor views and native test arrays into NumPy."""
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


class ElevatorAdapter:
    """CPU logic mirror; only mjlab performs physical integration.

    Native forward kinematics refresh the controller's bias/geometry mirror.
    It never runs mj_step. Only the scene's indexed controls and forces flow
    back to the authoritative simulation, preserving robot command history.
    """
    def __init__(self, sim):
        self.sim = sim
        self.model = sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.controller = ThreeFloorElevatorController(self.model, self.data)
        c = self.controller
        self.ctrl_ids = np.array([c.lift_actuator_id, *c.door_actuator_ids.values()])
        self.force_ids = np.array([c.lift_dofadr, *c.button_dofadr.values()])
        self.qpos_ids = np.array([c.lift_qposadr, *c.door_qposadr.values(), *c.button_qposadr.values()])
        self.dof_ids = np.array([c.lift_dofadr, *c.door_dofadr.values(), *c.button_dofadr.values()])
        robot_joints = [j for j in range(self.model.njnt)
                        if self.model.body(int(self.model.jnt_bodyid[j])).name.startswith('robot/')]
        qpos_ids, dof_ids = [], []
        for j in robot_joints:
            kind = self.model.jnt_type[j]
            nq = 7 if kind == mujoco.mjtJoint.mjJNT_FREE else 4 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
            nv = 6 if kind == mujoco.mjtJoint.mjJNT_FREE else 3 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
            qpos_ids.extend(range(int(self.model.jnt_qposadr[j]), int(self.model.jnt_qposadr[j]) + nq))
            dof_ids.extend(range(int(self.model.jnt_dofadr[j]), int(self.model.jnt_dofadr[j]) + nv))
        self.robot_qpos_ids = np.asarray(qpos_ids, dtype=np.int64)
        self.robot_dof_ids = np.asarray(dof_ids, dtype=np.int64)
        self.events = []
        handle = c._handle_button_press

        def record_press(name):
            handle(name)
            self.events.append({'type': 'button_press', 'button': name,
                                'sim_time': float(self.data.time)})

        c._handle_button_press = record_press

    def sync(self):
        for name in ('qpos', 'qvel', 'ctrl', 'qfrc_applied'):
            getattr(self.data, name)[:] = cpu_array(getattr(self.sim.data, name)[0])
        self.data.time = float(self.sim.data.time[0])
        mujoco.mj_forward(self.model, self.data)

    def _write(self):
        for name, ids in (('ctrl', self.ctrl_ids), ('qfrc_applied', self.force_ids)):
            target = getattr(self.sim.data, name)
            value = self.data.__getattribute__(name)[ids].copy()
            if hasattr(target, 'device'):
                import torch
                value = torch.as_tensor(value, device=target.device, dtype=target.dtype)
            target[0, ids] = value

    def reset(self):
        # controller.reset clears its mirror, so restore robot and unrelated
        # state before writing ONLY elevator qpos/qvel/ctrl/force indices.
        self.sync()
        saved_qpos, saved_qvel = self.data.qpos.copy(), self.data.qvel.copy()
        saved_forces = self.data.qfrc_applied.copy()
        self.controller.reset()
        saved_qpos[self.qpos_ids] = self.data.qpos[self.qpos_ids]
        saved_qvel[self.dof_ids] = 0.
        self.data.qpos[:], self.data.qvel[:] = saved_qpos, saved_qvel
        saved_forces[self.force_ids] = self.data.qfrc_applied[self.force_ids]
        self.data.qfrc_applied[:] = saved_forces
        for name, ids, value in (('qpos', self.qpos_ids, saved_qpos[self.qpos_ids]),
                                 ('qvel', self.dof_ids, np.zeros(len(self.dof_ids)))):
            target = getattr(self.sim.data, name)
            if hasattr(target, 'device'):
                import torch
                value = torch.as_tensor(value.copy(), device=target.device, dtype=target.dtype)
            target[0, ids] = value
        self.events.clear()
        self._write()
        # controller.reset() ran forward kinematics before restoring the robot's
        # qpos. Refresh derived poses before publishing the initial observation.
        self.sync()

    def before_step(self):
        self.sync()
        self.controller.apply_mouse_forces()
        self._write()

    def after_step(self):
        self.sync()
        self.controller.update()
        self._write()

    def step_scene(self, physics_step):
        """Advance the scene one substep, holding only the robot's state.

        Restore before refreshing contacts/controller state, so observations
        never see the temporary robot integration. Keep solver warm-start
        state too, allowing the next command to resume the held robot state.
        """
        saved = []
        for name, ids in (('qpos', self.robot_qpos_ids),
                          ('qvel', self.robot_dof_ids),
                          ('qacc_warmstart', self.robot_dof_ids)):
            target = getattr(self.sim.data, name)
            value = target[0, ids]
            saved.append((target, ids, value.clone() if hasattr(value, 'clone') else value.copy()))
        self.before_step()
        try:
            physics_step()
        finally:
            for target, ids, value in saved:
                target[0, ids] = value
            self.sim.forward()
        self.after_step()

    def evaluation_state(self):
        c = self.controller
        return {'floor': c.current_floor, 'target': c.current_target,
                'moving': c.motion_active, 'closing': c.doors_closing,
                'lamps': {str(f): bool(self.model.light_active[i]) for f, i in c.lamp_ids.items()}}
