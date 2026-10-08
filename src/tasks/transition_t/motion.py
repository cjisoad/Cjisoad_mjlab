"""Validated NumPy reference data for the one-shot Go2 rise transition.

The asset is a kinematic reference, not a dynamically validated trajectory.
This module deliberately has no simulator or task-registry dependencies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


DEFAULT_MOTION_PATH = (
    Path(__file__).resolve().parents[3]
    / "outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz"
)
ACCEPTED_MOTION_SHA256 = "1cb970ae8cfb107e25f3aca2e1d48b8cf9a2a145d6fac25c9aa67076c3bdd0b6"
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)
BODY_NAMES = (
    "base_link", "Head_upper", "Head_lower",
    "FL_hip", "FL_thigh", "FL_calf", "FL_calflower", "FL_calflower1", "FL_foot",
    "FR_hip", "FR_thigh", "FR_calf", "FR_calflower", "FR_calflower1", "FR_foot",
    "RL_hip", "RL_thigh", "RL_calf", "RL_calflower", "RL_calflower1", "RL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_calflower", "RR_calflower1", "RR_foot",
    "imu", "radar",
)
ARRAY_SHAPES = {
    "joint_pos": (12,),
    "joint_vel": (12,),
    "body_pos_w": (29, 3),
    "body_quat_w": (29, 4),
    "body_lin_vel_w": (29, 3),
    "body_ang_vel_w": (29, 3),
    "foot_pos_w": (4, 3),
    "contact": (4,),
}
REFERENCE_FIELDS = ("fps", "joint_names", "body_names", *ARRAY_SHAPES)
VELOCITY_FIELDS = frozenset(("joint_vel", "body_lin_vel_w", "body_ang_vel_w"))


def _name_order(values: np.ndarray, expected: tuple[str, ...], field: str) -> list[int]:
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{field} must have one-dimensional string shape")
    names = tuple(values.astype(str).tolist())
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate names in {field}")
    missing = sorted(set(expected) - set(names))
    unexpected = sorted(set(names) - set(expected))
    if missing or unexpected:
        raise ValueError(f"{field}: missing names {missing}; unexpected names {unexpected}")
    return [names.index(name) for name in expected]


class TransitionMotion:
    """Load and sample a 50 Hz reference in canonical joint and body order.

    ``foot_pos_w`` and ``contact`` always use FL/FR/RL/RR order. Body
    quaternions use MuJoCo's wxyz convention. Explicit path overrides must
    satisfy the same schema, but may have another SHA256 and frame count.
    """

    def __init__(self, path: str | Path = DEFAULT_MOTION_PATH):
        self.path = Path(path).expanduser().resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as archive:
            missing = [field for field in REFERENCE_FIELDS if field not in archive.files]
            if missing:
                raise ValueError(f"missing required reference fields: {', '.join(missing)}")
            fields = {field: np.array(archive[field], copy=True) for field in REFERENCE_FIELDS}

        fps = fields["fps"]
        if fps.shape != () or fps.dtype.kind not in "iuf" or not np.isfinite(fps) or float(fps) != 50.0:
            raise ValueError("fps must be a finite scalar equal to 50 Hz")
        self.fps = 50.0
        joint_order = _name_order(fields["joint_names"], JOINT_NAMES, "joint_names")
        body_order = _name_order(fields["body_names"], BODY_NAMES, "body_names")
        joint_pos = fields["joint_pos"]
        if joint_pos.ndim != 2:
            raise ValueError("joint_pos must have shape (frames, 12)")
        self.num_frames = joint_pos.shape[0]
        if self.num_frames == 0:
            raise ValueError("reference must contain at least one frame")

        self.arrays: dict[str, np.ndarray] = {}
        for field, trailing_shape in ARRAY_SHAPES.items():
            value = fields[field]
            expected_shape = (self.num_frames, *trailing_shape)
            if value.shape != expected_shape:
                raise ValueError(f"{field} must have shape {expected_shape}; got {value.shape}")
            if value.dtype.kind not in "iuf":
                raise ValueError(f"{field} must contain numeric values")
            if not np.isfinite(value).all():
                raise ValueError(f"{field} must contain only finite values")
            if field.startswith("joint_"):
                value = value[:, joint_order]
            elif field.startswith("body_"):
                value = value[:, body_order]
            self.arrays[field] = value
            setattr(self, field, value)

        if not np.allclose(np.linalg.norm(self.body_quat_w, axis=-1), 1.0, rtol=0.0, atol=1e-5):
            raise ValueError("body_quat_w must contain valid unit quaternions")
        self.joint_names = JOINT_NAMES
        self.body_names = BODY_NAMES
        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}
        self.body_name_to_index = {name: i for i, name in enumerate(self.body_names)}
        self.duration = (self.num_frames - 1) / self.fps

    def sample(self, time: float | np.ndarray) -> dict[str, np.ndarray]:
        """Interpolate time in seconds, clamp pose, and zero velocities past end.

        Leading dimensions follow ``time``. Contacts are held from the earlier
        frame, while quaternion interpolation follows the shortest sign path.
        The exact final time retains the last frame's recorded velocities;
        strictly later times request a stationary endpoint.
        """
        times = np.asarray(time, dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError("sample time must contain only finite values")
        clamped = np.clip(times, 0.0, self.duration)
        frame = clamped * self.fps
        nearest = np.rint(frame)
        tolerance = 8 * np.finfo(np.float64).eps * np.maximum(1., np.abs(frame))
        frame = np.where(np.abs(frame-nearest) <= tolerance, nearest, frame)
        # Multiplication can round an exact endpoint just below its frame index.
        frame = np.where(clamped >= self.duration, self.num_frames - 1, frame)
        lower = np.minimum(np.floor(frame).astype(np.int64), self.num_frames - 1)
        upper = np.minimum(lower + 1, self.num_frames - 1)
        fraction = frame - lower
        result = {}
        for field, values in self.arrays.items():
            a, b = values[lower], values[upper]
            weight = fraction.reshape(times.shape + (1,) * (values.ndim - 1))
            if field == "contact":
                sampled = a.copy()
            elif field == "body_quat_w":
                b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0.0, -b, b)
                sampled = (1.0 - weight) * a + weight * b
                sampled /= np.linalg.norm(sampled, axis=-1, keepdims=True)
            else:
                sampled = (1.0 - weight) * a + weight * b
            if field in VELOCITY_FIELDS:
                completed = (times > self.duration).reshape(times.shape + (1,) * (values.ndim - 1))
                sampled = np.where(completed, 0.0, sampled)
            result[field] = sampled
        return result
