"""Adapter from Justin's clean stair Zarr to the canonical clip contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np

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
from .math3d import unroll_quaternions_wxyz


_REQUIRED_ARRAYS = (
    "fps",
    "clip_names",
    "clip_start_idx",
    "clip_end_idx",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_FOOT_BODY_INDICES = (18, 19)
_INFERRED_FACING_HORIZON_FRAMES = 24
_CONTACT_PLACEHOLDER_TAG = "contacts-unreconstructed"
_QUATERNION_NORM_TOLERANCE = 1.0e-4


def _fail(detail: str) -> ContractError:
    return ContractError(f"Justin source {detail}")


def _finite_float32(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iuf" or not np.isfinite(source).all():
        raise _fail(f"{label} must contain finite real values")
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.ascontiguousarray(source, dtype=np.float32)
    if not np.isfinite(output).all():
        raise _fail(f"{label} must be representable as finite float32")
    return output


def _directory_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total_size = 0
    files = sorted(
        (entry for entry in path.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )
    if not files:
        raise _fail("Zarr directory contains no files")
    for entry in files:
        relative = entry.relative_to(path).as_posix()
        payload = entry.read_bytes()
        total_size += len(payload)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
    return total_size, digest.hexdigest()


def _decode_names(value: object) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise _fail("clip_names must be one-dimensional")
    names: list[str] = []
    for item in array:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        name = str(item)
        if not name:
            raise _fail("clip_names must be nonempty strings")
        names.append(name)
    if len(set(names)) != len(names):
        raise _fail("clip_names must be unique")
    return tuple(names)


def _validate_order(root: object) -> str:
    joint_names = root.attrs.get("joint_names")  # type: ignore[attr-defined]
    body_names = root.attrs.get("body_names")  # type: ignore[attr-defined]
    if joint_names is None and body_names is None:
        return "justin-zarr-legacy-isaaclab-v1"
    if joint_names is None or body_names is None:
        raise _fail("joint_names and body_names attrs must be declared together")
    if tuple(str(name) for name in joint_names) != ISAACLAB_JOINT_NAMES:
        raise _fail("joint_names do not equal the canonical IsaacLab order")
    if tuple(str(name) for name in body_names) != ISAACLAB_BODY_NAMES:
        raise _fail("body_names do not equal the canonical IsaacLab order")
    return "justin-zarr-isaaclab-v1"


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
        np.sin(yaw[future] - yaw),
        np.cos(yaw[future] - yaw),
    )
    inferred_facing = np.stack(
        (np.cos(facing_delta), np.sin(facing_delta)), axis=1
    )
    return CommandTrack(
        observed_travel_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_facing_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_mask=np.zeros(frame_count, dtype=np.bool_),
        inferred_velocity_local_xy=local_velocity,
        inferred_facing_local_xy=inferred_facing,
        inferred_yaw_rate_rad_s=np.asarray(root_angular_velocity_world)[:, 2],
    )


def _action_tags(name: str) -> tuple[str, ...]:
    tags = ["terrain", "stairs"]
    if name.startswith("up_"):
        tags.append("up")
    elif name.startswith("down_"):
        tags.append("down")
    tags.append(_CONTACT_PLACEHOLDER_TAG)
    return tuple(tags)


def iter_justin_clips(
    zarr_path: Path, terrain: TerrainBinding
) -> Iterator[CanonicalClip]:
    """Yield validated [start,end) clips from Justin's IsaacLab-order archive."""

    if not isinstance(terrain, TerrainBinding):
        raise _fail("terrain must be a TerrainBinding")
    source_path = Path(zarr_path).resolve()
    if not source_path.is_dir():
        raise _fail(f"Zarr path is not a directory: {source_path}")
    try:
        import zarr
    except ImportError as error:
        raise _fail("requires zarr") from error
    root = zarr.open(str(source_path), mode="r")
    missing = [name for name in _REQUIRED_ARRAYS if name not in root]
    if missing:
        raise _fail(f"is missing required arrays: {', '.join(missing)}")
    if root.attrs.get("quaternion_convention") != "xyzw":
        raise _fail("quaternion_convention must equal xyzw")
    source_format = _validate_order(root)

    fps = np.asarray(root["fps"])
    if (
        fps.dtype.kind not in "iuf"
        or fps.size != 1
        or not np.isfinite(fps).all()
        or float(fps.reshape(-1)[0]) != 50.0
    ):
        raise _fail("fps must contain exactly one value equal to 50")
    names = _decode_names(root["clip_names"][:])
    starts = np.asarray(root["clip_start_idx"])
    ends = np.asarray(root["clip_end_idx"])
    if (
        starts.dtype.kind not in "iu"
        or ends.dtype.kind not in "iu"
        or starts.shape != (len(names),)
        or ends.shape != (len(names),)
    ):
        raise _fail("exclusive clip ranges must be one integer pair per clip")
    starts = starts.astype(np.int64, copy=False)
    ends = ends.astype(np.int64, copy=False)

    joint_position = _finite_float32(root["joint_pos"][:], "joint_pos")
    joint_velocity = _finite_float32(root["joint_vel"][:], "joint_vel")
    body_position = _finite_float32(root["body_pos_w"][:], "body_pos_w")
    body_quaternion_xyzw = _finite_float32(
        root["body_quat_w"][:], "body_quat_w"
    )
    body_linear_velocity = _finite_float32(
        root["body_lin_vel_w"][:], "body_lin_vel_w"
    )
    body_angular_velocity = _finite_float32(
        root["body_ang_vel_w"][:], "body_ang_vel_w"
    )
    frame_count = joint_position.shape[0] if joint_position.ndim == 2 else -1
    expected_shapes = {
        "joint_pos": (frame_count, 29),
        "joint_vel": (frame_count, 29),
        "body_pos_w": (frame_count, 30, 3),
        "body_quat_w": (frame_count, 30, 4),
        "body_lin_vel_w": (frame_count, 30, 3),
        "body_ang_vel_w": (frame_count, 30, 3),
    }
    arrays = {
        "joint_pos": joint_position,
        "joint_vel": joint_velocity,
        "body_pos_w": body_position,
        "body_quat_w": body_quaternion_xyzw,
        "body_lin_vel_w": body_linear_velocity,
        "body_ang_vel_w": body_angular_velocity,
    }
    for label, expected in expected_shapes.items():
        if arrays[label].shape != expected:
            raise _fail(f"{label} must have shape {expected}")
    if (
        not len(names)
        or starts[0] != 0
        or ends[-1] != frame_count
        or np.any(ends <= starts)
        or not np.array_equal(starts[1:], ends[:-1])
    ):
        raise _fail(
            "exclusive clip ranges must be nonempty, contiguous, and cover all frames"
        )
    quaternion_norm = np.linalg.norm(body_quaternion_xyzw, axis=-1)
    if np.any(np.abs(quaternion_norm - 1.0) > _QUATERNION_NORM_TOLERANCE):
        raise _fail("body_quat_w has a quaternion norm error above 1e-4")

    source_size, source_sha256 = _directory_identity(source_path)
    source = SourceIdentity(
        source_format=source_format,
        source_path=str(source_path),
        source_size_bytes=source_size,
        source_sha256=source_sha256,
        source_license_id="UNRECORDED",
        coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
        quaternion_convention="wxyz",
        pose_origin=CLEAN_POSE_ORIGIN,
    )
    for name, start_value, end_value in zip(names, starts, ends, strict=True):
        start = int(start_value)
        end = int(end_value)
        body_quaternion = unroll_quaternions_wxyz(
            body_quaternion_xyzw[start:end, :, (3, 0, 1, 2)]
        )
        root_position = body_position[start:end, 0]
        root_quaternion = body_quaternion[:, 0]
        root_linear_velocity = body_linear_velocity[start:end, 0]
        root_angular_velocity = body_angular_velocity[start:end, 0]
        sole_position = body_position[start:end, _FOOT_BODY_INDICES]
        sole_quaternion = body_quaternion[:, _FOOT_BODY_INDICES]
        clip_frames = end - start
        clip = CanonicalClip(
            clip_id=name,
            fps=50.0,
            source=source,
            joint_names=ISAACLAB_JOINT_NAMES,
            body_names=ISAACLAB_BODY_NAMES,
            root_position_world=root_position,
            root_quaternion_world_wxyz=root_quaternion,
            joint_position=joint_position[start:end],
            root_linear_velocity_world=root_linear_velocity,
            root_angular_velocity_world=root_angular_velocity,
            joint_velocity=joint_velocity[start:end],
            body_position_world=body_position[start:end],
            body_quaternion_world_wxyz=body_quaternion,
            body_linear_velocity_world=body_linear_velocity[start:end],
            body_angular_velocity_world=body_angular_velocity[start:end],
            sole_position_world=sole_position,
            sole_quaternion_world_wxyz=sole_quaternion,
            heel_position_world=sole_position,
            toe_position_world=sole_position,
            contact=np.zeros((clip_frames, 2), dtype=np.float32),
            contact_confidence=np.zeros((clip_frames, 2), dtype=np.float32),
            commands=_infer_commands(
                root_quaternion,
                root_linear_velocity,
                root_angular_velocity,
            ),
            terrain=terrain,
            action_tags=_action_tags(name),
        )
        clip.validate()
        yield clip
