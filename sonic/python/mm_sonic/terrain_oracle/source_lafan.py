"""Adapter for the audited 30 Hz G1-retargeted LAFAN1 CSV release."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mm_sonic.grail_terrain_source import (
    G1MujocoFK,
    mujoco_to_isaaclab_joints,
)
from mm_sonic.joints import ContractError

from .canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
    TerrainBinding,
)
from .math3d import (
    angular_velocity_world_wxyz,
    finite_difference,
    slerp_wxyz,
    unroll_quaternions_wxyz,
)


LAFAN1_UPSTREAM_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
_SOURCE_FPS = 30
_TARGET_FPS = 50
_FOOT_BODY_INDICES = (18, 19)
_INFERRED_FACING_HORIZON_FRAMES = 24
_CONTACT_PLACEHOLDER_TAG = "contacts-unreconstructed"


def _fail(detail: str) -> ValueError:
    return ValueError(f"LAFAN G1 CSV {detail}")


def classify_lafan_name(name: str) -> tuple[str, ...]:
    """Return conservative action labels derived only from a LAFAN clip name."""

    if type(name) is not str or not name:
        raise _fail("name must be a nonempty string")
    normalized = "".join(character for character in name.lower() if character.isalnum())
    tags = ["lafan"]
    if "obstacle" in normalized:
        tags.append("obstacle")
    elif normalized.startswith("walk"):
        tags.append("walk")
    elif normalized.startswith("run"):
        tags.append("run")
    elif normalized.startswith("sprint"):
        tags.append("sprint")
    elif normalized.startswith("jump"):
        tags.append("jump")
    elif normalized.startswith("fallandgetup"):
        tags.extend(("fall", "get-up"))
    elif normalized.startswith("dance"):
        tags.append("dance")
    elif normalized.startswith("fightandsports"):
        tags.extend(("fight", "sports"))
    elif normalized.startswith("fight"):
        tags.append("fight")
    else:
        tags.append("unclassified")
    return tuple(tags)


def _load_rows(path: Path) -> np.ndarray:
    try:
        rows = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise _fail("must contain only comma-separated numeric rows") from error
    if (
        rows.ndim != 2
        or rows.shape[0] < 2
        or rows.shape[1] != 36
        or not np.isfinite(rows).all()
    ):
        raise _fail("must contain at least two finite 36-column frames")
    quaternion = rows[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(np.abs(norms - 1.0) > 1.0e-4):
        raise _fail("root XYZW quaternions must be normalized")
    return np.ascontiguousarray(rows)


def _resample_rows(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_count = len(rows)
    target_count = ((source_count - 1) * _TARGET_FPS) // _SOURCE_FPS + 1
    source_coordinates = (
        np.arange(target_count, dtype=np.float64)
        * np.float64(_SOURCE_FPS / _TARGET_FPS)
    )
    left = np.floor(source_coordinates).astype(np.int64)
    right = np.minimum(left + 1, source_count - 1)
    fraction = source_coordinates - left

    root_position = (
        (1.0 - fraction[:, None]) * rows[left, :3]
        + fraction[:, None] * rows[right, :3]
    )
    joints = (
        (1.0 - fraction[:, None]) * rows[left, 7:]
        + fraction[:, None] * rows[right, 7:]
    )
    source_root_wxyz = unroll_quaternions_wxyz(rows[:, (6, 3, 4, 5)])
    root_wxyz = slerp_wxyz(
        source_root_wxyz[left], source_root_wxyz[right], fraction
    )
    return (
        np.ascontiguousarray(root_position, dtype=np.float32),
        root_wxyz,
        np.ascontiguousarray(joints, dtype=np.float32),
    )


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _infer_commands(
    root_quaternion_world_wxyz: np.ndarray,
    root_linear_velocity_world: np.ndarray,
    root_angular_velocity_world: np.ndarray,
) -> CommandTrack:
    frame_count = len(root_quaternion_world_wxyz)
    yaw = _yaw_from_wxyz(root_quaternion_world_wxyz)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    velocity = np.asarray(root_linear_velocity_world)[:, :2]
    local_velocity = np.stack(
        (
            cosine * velocity[:, 0] + sine * velocity[:, 1],
            -sine * velocity[:, 0] + cosine * velocity[:, 1],
        ),
        axis=1,
    )
    future = np.minimum(
        np.arange(frame_count) + _INFERRED_FACING_HORIZON_FRAMES,
        frame_count - 1,
    )
    facing_delta = np.arctan2(
        np.sin(yaw[future] - yaw), np.cos(yaw[future] - yaw)
    )
    return CommandTrack(
        observed_travel_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_facing_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_mask=np.zeros(frame_count, dtype=np.bool_),
        inferred_velocity_local_xy=local_velocity,
        inferred_facing_local_xy=np.stack(
            (np.cos(facing_delta), np.sin(facing_delta)), axis=1
        ),
        inferred_yaw_rate_rad_s=np.asarray(root_angular_velocity_world)[:, 2],
    )


def load_lafan_csv(
    path: Path,
    model_path: Path,
    *,
    terrain: TerrainBinding | None,
) -> CanonicalClip:
    """Load one audited G1 CSV and convert it to the reviewed 50 Hz boundary."""

    if terrain is not None and not isinstance(terrain, TerrainBinding):
        raise ContractError("LAFAN terrain must be a TerrainBinding or None")
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_bytes = source_path.read_bytes()
    rows = _load_rows(source_path)
    root_position, root_wxyz, joints_mujoco = _resample_rows(rows)
    root_xyzw = np.ascontiguousarray(root_wxyz[:, (1, 2, 3, 0)])
    forward = G1MujocoFK(model_path).forward(
        root_position, root_xyzw, joints_mujoco
    )
    if tuple(forward.body_names) != ISAACLAB_BODY_NAMES:
        raise _fail("FK result does not use the canonical body order")
    body_quaternion_wxyz = unroll_quaternions_wxyz(
        forward.body_quaternion_world_xyzw[..., (3, 0, 1, 2)]
    )
    joint_position = mujoco_to_isaaclab_joints(joints_mujoco)
    root_linear_velocity = finite_difference(root_position, 50.0)
    root_angular_velocity = angular_velocity_world_wxyz(root_wxyz, 50.0)
    joint_velocity = finite_difference(joint_position, 50.0)
    body_linear_velocity = finite_difference(
        forward.body_position_world, 50.0
    )
    body_angular_velocity = angular_velocity_world_wxyz(
        body_quaternion_wxyz, 50.0
    )
    sole_position = forward.body_position_world[:, _FOOT_BODY_INDICES]
    sole_quaternion = body_quaternion_wxyz[:, _FOOT_BODY_INDICES]
    frame_count = len(root_position)
    action_tags = (
        *classify_lafan_name(source_path.stem),
        _CONTACT_PLACEHOLDER_TAG,
    )
    clip = CanonicalClip(
        clip_id=source_path.stem,
        fps=50.0,
        source=SourceIdentity(
            source_format=(
                "lafan1-retargeted-g1-csv-30hz@"
                + LAFAN1_UPSTREAM_REVISION
            ),
            source_path=str(source_path),
            source_size_bytes=len(source_bytes),
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            source_license_id="CC-BY-NC-ND-4.0",
            coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
            quaternion_convention="wxyz",
            pose_origin=CLEAN_POSE_ORIGIN,
        ),
        joint_names=ISAACLAB_JOINT_NAMES,
        body_names=ISAACLAB_BODY_NAMES,
        root_position_world=root_position,
        root_quaternion_world_wxyz=root_wxyz,
        joint_position=joint_position,
        root_linear_velocity_world=root_linear_velocity,
        root_angular_velocity_world=root_angular_velocity,
        joint_velocity=joint_velocity,
        body_position_world=forward.body_position_world,
        body_quaternion_world_wxyz=body_quaternion_wxyz,
        body_linear_velocity_world=body_linear_velocity,
        body_angular_velocity_world=body_angular_velocity,
        sole_position_world=sole_position,
        sole_quaternion_world_wxyz=sole_quaternion,
        heel_position_world=sole_position,
        toe_position_world=sole_position,
        contact=np.zeros((frame_count, 2), dtype=np.float32),
        contact_confidence=np.zeros((frame_count, 2), dtype=np.float32),
        commands=_infer_commands(
            root_wxyz,
            root_linear_velocity,
            root_angular_velocity,
        ),
        terrain=terrain,
        action_tags=action_tags,
    )
    clip.validate()
    return clip
