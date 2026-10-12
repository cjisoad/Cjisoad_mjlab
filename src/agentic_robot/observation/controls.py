"""Dynamic contact surfaces for the elevator controller's stable logical IDs."""
import mujoco

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController

BUTTON_NAMES = ThreeFloorElevatorController.BUTTON_NAMES


class ControlRegistry:
    def __init__(self, model):
        self.model = model
        self.entries = []
        for name in BUTTON_NAMES:
            site = model.site(f'surface_{name}').id
            joint = model.joint(f'joint_{name}').id
            body = int(model.site_bodyid[site])
            parent_id = int(model.body_parentid[body])
            parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or 'world'
            semantic = dict(id=name, type='cabin_destination' if name.startswith('cab_') else
                            'light_switch' if name.startswith('light_') else 'hall_call',
                            parent=parent, position_source='sim_ground_truth')
            semantic['target_floor' if name.startswith('cab_') else 'floor'] = int(name[-1])
            self.entries.append((site, joint, semantic))

    def read(self, data):
        return [dict(metadata, surface_position_w=data.site_xpos[site].tolist(),
                     press_direction_w=data.xaxis[joint].tolist(),
                     normal_out_w=(-data.xaxis[joint]).tolist())
                for site, joint, metadata in self.entries]
