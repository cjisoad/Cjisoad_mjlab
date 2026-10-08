"""Data and endpoint contract for the transition_t kinematic reference."""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest


ASSET_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz"
)
ASSET_SHA256 = "1cb970ae8cfb107e25f3aca2e1d48b8cf9a2a145d6fac25c9aa67076c3bdd0b6"

CANONICAL_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

CANONICAL_BODY_NAMES = (
    "base_link",
    "Head_upper",
    "Head_lower",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_calflower",
    "FL_calflower1",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_calflower",
    "FR_calflower1",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_calflower",
    "RL_calflower1",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_calflower",
    "RR_calflower1",
    "RR_foot",
    "imu",
    "radar",
)

REFERENCE_FIELDS = (
    "fps",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "foot_pos_w",
    "contact",
)


@lru_cache(maxsize=1)
def _transition_motion_class():
    """Load the standalone module without importing the Isaac task registry."""
    module_path = Path(__file__).resolve().parents[1] / "src/tasks/transition_t/motion.py"
    spec = importlib.util.spec_from_file_location("transition_t_motion_under_test", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load transition motion module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TransitionMotion


def _load_asset_fields() -> dict[str, np.ndarray]:
    with np.load(ASSET_PATH, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _save_variant(path: Path, fields: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **fields)
    return path


def test_accepted_reference_has_expected_identity_and_canonical_named_arrays():
    TransitionMotion = _transition_motion_class()

    assert hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest() == ASSET_SHA256
    motion = TransitionMotion(ASSET_PATH)

    assert motion.sha256 == ASSET_SHA256
    assert motion.fps == 50
    assert motion.num_frames == 351
    assert motion.duration == pytest.approx(7.0)
    assert motion.joint_names == CANONICAL_JOINT_NAMES
    assert motion.body_names == CANONICAL_BODY_NAMES
    assert motion.joint_name_to_index["RR_calf_joint"] == 11
    assert motion.body_name_to_index["FL_foot"] == 8
    assert tuple(motion.arrays) == REFERENCE_FIELDS[3:]

    expected_shapes = {
        "joint_pos": (351, 12),
        "joint_vel": (351, 12),
        "body_pos_w": (351, 29, 3),
        "body_quat_w": (351, 29, 4),
        "body_lin_vel_w": (351, 29, 3),
        "body_ang_vel_w": (351, 29, 3),
        "foot_pos_w": (351, 4, 3),
        "contact": (351, 4),
    }
    for field, shape in expected_shapes.items():
        value = getattr(motion, field)
        assert value.shape == shape
        assert np.isfinite(value).all()


def test_loader_reads_only_approved_fields_from_asset(tmp_path: Path):
    TransitionMotion = _transition_motion_class()

    asset = _load_asset_fields()
    assert "root_state" in asset  # The accepted archive contains unrelated audit data.
    asset["ignored_pickle"] = np.array([object()], dtype=object)
    motion = TransitionMotion(_save_variant(tmp_path / "extra_fields.npz", asset))

    assert set(motion.arrays) == set(REFERENCE_FIELDS[3:])
    assert "root_state" not in motion.arrays


def test_shuffled_joint_and_body_names_restore_canonical_array_order(tmp_path: Path):
    TransitionMotion = _transition_motion_class()

    original = TransitionMotion(ASSET_PATH)
    fields = _load_asset_fields()
    joint_order = np.array([7, 0, 11, 4, 2, 9, 1, 10, 5, 8, 3, 6])
    body_order = np.array([28, 2, 9, 0, 17, 5, 24, 11, 1, 22, 7, 14, 3, 20, 10, 27, 4, 19, 8, 25, 6, 15, 12, 26, 13, 21, 16, 23, 18])
    fields["joint_names"] = fields["joint_names"][joint_order]
    for key in ("joint_pos", "joint_vel"):
        fields[key] = fields[key][:, joint_order]
    fields["body_names"] = fields["body_names"][body_order]
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        fields[key] = fields[key][:, body_order]

    shuffled = TransitionMotion(_save_variant(tmp_path / "shuffled.npz", fields))
    assert shuffled.joint_names == CANONICAL_JOINT_NAMES
    assert shuffled.body_names == CANONICAL_BODY_NAMES
    for key in REFERENCE_FIELDS[3:]:
        np.testing.assert_array_equal(getattr(shuffled, key), getattr(original, key))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_joint", "duplicate.*joint"),
        ("duplicate_body", "duplicate.*body"),
        ("missing_joint", "joint.*missing|missing.*joint"),
        ("missing_body", "body.*missing|missing.*body"),
        ("joint_shape", "joint_pos.*shape|shape.*joint_pos"),
        ("body_shape", "body_pos_w.*shape|shape.*body_pos_w"),
        ("nan", "finite"),
        ("zero_quaternion", "quaternion"),
        ("nonunit_quaternion", "quaternion"),
        ("boolean_quaternion", "body_quat_w.*numeric"),
        ("wrong_fps", "fps|50.*hz"),
        ("missing_field", "missing.*contact"),
        ("nonnumeric", "joint_pos.*numeric"),
        ("empty", "frame"),
        ("name_dimension", "joint_names.*shape"),
    ],
)
def test_loader_rejects_malformed_reference_with_specific_error(
    tmp_path: Path, mutation: str, message: str
):
    TransitionMotion = _transition_motion_class()

    fields = _load_asset_fields()
    if mutation == "duplicate_joint":
        fields["joint_names"][0] = fields["joint_names"][1]
    elif mutation == "duplicate_body":
        fields["body_names"][0] = fields["body_names"][1]
    elif mutation == "missing_joint":
        fields["joint_names"][0] = "unexpected_joint"
    elif mutation == "missing_body":
        fields["body_names"][0] = "unexpected_body"
    elif mutation == "joint_shape":
        fields["joint_pos"] = fields["joint_pos"][:, :-1]
    elif mutation == "body_shape":
        fields["body_pos_w"] = fields["body_pos_w"][:, :-1]
    elif mutation == "nan":
        fields["joint_vel"][0, 0] = np.nan
    elif mutation == "zero_quaternion":
        fields["body_quat_w"][0, 0] = 0.0
    elif mutation == "nonunit_quaternion":
        fields["body_quat_w"][0, 0] *= 2.0
    elif mutation == "boolean_quaternion":
        fields["body_quat_w"] = np.zeros(fields["body_quat_w"].shape, dtype=bool)
        fields["body_quat_w"][..., 0] = True
    elif mutation == "wrong_fps":
        fields["fps"] = np.array(49)
    elif mutation == "missing_field":
        del fields["contact"]
    elif mutation == "nonnumeric":
        fields["joint_pos"] = fields["joint_pos"].astype(str)
    elif mutation == "empty":
        fields["joint_pos"] = fields["joint_pos"][:0]
    elif mutation == "name_dimension":
        fields["joint_names"] = fields["joint_names"][None, :]

    path = _save_variant(tmp_path / f"{mutation}.npz", fields)
    with pytest.raises(ValueError, match=message):
        TransitionMotion(path)


def test_sampling_interpolates_and_holds_final_pose_with_zero_velocities_after_end():
    TransitionMotion = _transition_motion_class()

    motion = TransitionMotion(ASSET_PATH)
    halfway = motion.sample(2.31)
    np.testing.assert_allclose(halfway["joint_pos"], (motion.joint_pos[115] + motion.joint_pos[116]) / 2.0)
    np.testing.assert_allclose(halfway["foot_pos_w"], (motion.foot_pos_w[115] + motion.foot_pos_w[116]) / 2.0)
    np.testing.assert_allclose(np.linalg.norm(halfway["body_quat_w"], axis=-1), 1.0, atol=1e-7)

    end = motion.sample(motion.duration)
    after = motion.sample(motion.duration + 0.5)
    for field in ("joint_pos", "body_pos_w", "body_quat_w", "foot_pos_w", "contact"):
        np.testing.assert_allclose(after[field], getattr(motion, field)[-1])
    for field in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_array_equal(end[field], getattr(motion, field)[-1])
        np.testing.assert_array_equal(after[field], np.zeros_like(after[field]))


def test_sampling_selects_exact_final_frame_for_non_351_frame_reference(tmp_path: Path):
    TransitionMotion = _transition_motion_class()
    fields = _load_asset_fields()
    for field in REFERENCE_FIELDS[3:]:
        fields[field] = fields[field][:30]
    fields["joint_pos"][-1] = fields["joint_pos"][-2] + 5.0
    fields["joint_vel"][-1] = 3.0
    motion = TransitionMotion(_save_variant(tmp_path / "thirty_frames.npz", fields))

    # At this valid 50 Hz endpoint, duration * fps rounds just below frame 29.
    assert motion.duration * motion.fps < motion.num_frames - 1
    sampled = motion.sample(motion.duration)
    np.testing.assert_array_equal(sampled["joint_pos"], motion.joint_pos[-1])
    np.testing.assert_array_equal(sampled["joint_vel"], motion.joint_vel[-1])


def test_sampling_batch_clamps_negative_times_and_rejects_nonfinite_times():
    TransitionMotion = _transition_motion_class()
    motion = TransitionMotion(ASSET_PATH)
    sampled = motion.sample(np.array([[-1.0, 2.31], [7.0, 8.0]]))
    assert sampled["joint_pos"].shape == (2, 2, 12)
    np.testing.assert_array_equal(sampled["joint_pos"][0, 0], motion.joint_pos[0])
    np.testing.assert_array_equal(sampled["joint_pos"][1, 1], motion.joint_pos[-1])
    np.testing.assert_array_equal(sampled["joint_vel"][1, 1], np.zeros(12))
    with pytest.raises(ValueError, match="finite"):
        motion.sample(np.array([np.nan]))


def test_sampling_handles_equivalent_quaternion_signs_and_discrete_contact(tmp_path: Path):
    TransitionMotion = _transition_motion_class()
    fields = _load_asset_fields()
    fields["body_quat_w"][1] = -fields["body_quat_w"][0]
    fields["contact"][0] = 0.0
    fields["contact"][1] = 1.0
    motion = TransitionMotion(_save_variant(tmp_path / "signs.npz", fields))
    sampled = motion.sample(0.01)
    np.testing.assert_allclose(sampled["body_quat_w"], fields["body_quat_w"][0])
    np.testing.assert_array_equal(sampled["contact"], fields["contact"][0])


def test_exact_interior_timestamp_selects_current_contact(tmp_path: Path):
    fields = _load_asset_fields()
    fields['contact'][28] = 0.
    fields['contact'][29] = 1.
    motion = _transition_motion_class()(_save_variant(tmp_path / 'interior.npz', fields))
    np.testing.assert_array_equal(motion.sample(.58)['contact'], fields['contact'][29])
    np.testing.assert_array_equal(motion.sample(.58 - 1e-7)['contact'], fields['contact'][28])
