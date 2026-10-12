"""Compose the authored elevator with the existing teacher robot physics."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController

SCENE_XML = Path(__file__).resolve().parents[2] / 'simulate/mujoco/model/three_floor_elevator/three_floor_elevator.xml'


def add_elevator(spec: mujoco.MjSpec) -> None:
    """Attach the entire environment under a non-robot contact subtree."""
    elevator = mujoco.MjSpec.from_file(str(SCENE_XML))
    for name in ThreeFloorElevatorController.BUTTON_NAMES:
        body = elevator.body(f'button_{name}')
        geom = elevator.geom(f'geom_{name}')
        joint = elevator.joint(f'joint_{name}')
        # Authored buttons press along local +Y (light) or +X (round).
        # Their touch surface is one half-thickness outward from geom center.
        axis = np.asarray(joint.axis)
        half = float(geom.size[1])  # box Y half-size or cylinder half-length
        surface = np.asarray(geom.pos) - half * axis
        body.add_site(name=f'surface_{name}', pos=surface, size=(.005, 0, 0),
                      rgba=(0, 0, 0, 0), group=5)
    parent = spec.worldbody.add_body(name='elevator_scene')
    spec.attach(elevator, prefix='', frame=parent.add_frame())
    # OpenGL camera looks down local -Z, local +Y is image up.
    # These axes make robot +X forward and +Z image up at level stance.
    rotation = np.array(((0., 0., -1.), (-1., 0., 0.), (0., 1., 0.)))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    spec.body('robot/Head_upper').add_camera(
        name='robot_ego', pos=(.065, 0, .045), quat=quat, fovy=70.)
    spec.visual.global_.offwidth = max(640, spec.visual.global_.offwidth)
    spec.visual.global_.offheight = max(480, spec.visual.global_.offheight)
    spec.visual.quality.shadowsize = 2048
    # Entity keyframes were created before scene joints were attached. Runtime
    # reset uses explicit entity/default and elevator state, not partial keys.
    for key in list(spec.keys):
        spec.delete(key)


def configure_elevator(cfg, *, spawn=(3.9, -3.0, .42), yaw=math.pi / 2) -> None:
    """Configuration callback for load_full_chain, leaving its actors intact."""
    from mjlab.sensor import ContactMatch
    if cfg.scene.num_envs != 1:
        raise ValueError('Agentic elevator supports exactly one physical world')
    cfg.scene.terrain = None
    robot = cfg.scene.entities['robot']
    robot.init_state.pos = tuple(spawn)
    robot.init_state.rot = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))
    old_callback = cfg.scene.spec_fn

    def compose(spec):
        if old_callback is not None:
            old_callback(spec)
        add_elevator(spec)

    cfg.scene.spec_fn = compose
    for sensor in cfg.scene.sensors:
        if sensor.name != 'self_collision' and hasattr(sensor, 'secondary'):
            # All physical scene contacts, including platforms, doors, furniture
            # and buttons, with no robot self-contact counted as support.
            sensor.secondary = ContactMatch(mode='subtree', pattern='elevator_scene')
    cfg.rewards = {}
    cfg.metrics = {}
    cfg.curriculum = {}
    # The agent runtime stops explicitly after a physical failure. Manager
    # auto-reset would otherwise erase the failure state inside env.step().
    cfg.terminations = {}
    cfg.episode_length_s = 1.e9
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048
    cfg.sim.contact_sensor_maxmatch = 4096
