"""Read-only robot state and the existing teacher anchor convention."""
from copy import deepcopy

import mujoco
import numpy as np

ANCHOR_CONTRACT = 'teacher_body_y_cross_up_v1'
ANCHOR_EPS = 1e-6


def anchor_basis(quat_wxyz, gravity_w):
    gravity = np.asarray(gravity_w, dtype=float)
    if not np.isfinite(gravity).all() or np.linalg.norm(gravity) <= 1e-12:
        raise ValueError('gravity must be a finite nonzero vector')
    up = -gravity / max(np.linalg.norm(gravity), 1e-12)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat_wxyz, dtype=float))
    forward = np.cross(rotation.reshape(3, 3)[:, 1], up)
    reference = np.array([1., 0., 0.]) - up[0] * up
    if np.linalg.norm(reference) <= ANCHOR_EPS:
        reference = np.array([0., 1., 0.]) - up[1] * up
    if np.linalg.norm(forward) <= ANCHOR_EPS:
        forward = reference
    forward /= max(np.linalg.norm(forward), 1e-12)
    return np.column_stack((forward, np.cross(up, forward), up))


def world_to_command(target_position_w, anchor):
    """World FR-site target to teacher cmd_p (offset from term.zero)."""
    return (np.asarray(anchor['R_world_from_anchor']).T @
            (np.asarray(target_position_w) - anchor['origin_w']) - anchor['fr_zero_anchor'])


class StateReader:
    def __init__(self, model, body_id, fr_site_id, fr_zero_anchor):
        self.model, self.body_id, self.fr_site_id = model, body_id, fr_site_id
        self.zero = np.asarray(fr_zero_anchor, dtype=float).reshape(3)
        if not np.isfinite(self.zero).all():
            raise ValueError('fr_zero_anchor must be finite')

    def read(self, data, fsm):
        position = data.xpos[self.body_id].tolist()
        quat = data.xquat[self.body_id].tolist()
        return dict(position_w=position, orientation_wxyz=quat,
                    anchor=dict(origin_w=position.copy(), R_world_from_anchor=anchor_basis(quat, self.model.opt.gravity).tolist(),
                                fr_zero_anchor=self.zero.tolist(), contract=ANCHOR_CONTRACT),
                    fr_position_w=data.site_xpos[self.fr_site_id].tolist(), fsm=deepcopy(fsm))
