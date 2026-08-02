"""Adapter from native Takara motion folders to the canonical clip contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_motion_data import MotionFolder

from .canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
)


_ROOT_BODY_INDEX = 0
_FOOT_BODY_INDICES = (18, 19)
_INFERRED_FACING_HORIZON_FRAMES = 24
_CONTACT_PLACEHOLDER_TAG = "contacts-unreconstructed"


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


def iter_flat_clips(
    root: Path, *, tags: tuple[str, ...]
) -> Iterator[CanonicalClip]:
    """Yield every native 50 Hz/wxyz clip below ``root`` without reordering."""

    if type(tags) is not tuple or not tags or any(
        type(tag) is not str or not tag for tag in tags
    ):
        raise ContractError("flat adapter tags must be a nonempty tuple of strings")
    folder = MotionFolder.load(root)
    if folder.layout.joint_count != 29 or folder.layout.body_count != 30:
        raise ContractError("flat source must use the canonical 29/30 G1 layout")
    action_tags = (
        tags
        if _CONTACT_PLACEHOLDER_TAG in tags
        else (*tags, _CONTACT_PLACEHOLDER_TAG)
    )
    for native in folder.clips:
        source_path = folder.root / native.relative_path
        source_bytes = source_path.read_bytes()
        source = SourceIdentity(
            source_format="takara-motion-npz-v1",
            source_path=str(source_path),
            source_size_bytes=len(source_bytes),
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            source_license_id="UNRECORDED",
            coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
            quaternion_convention="wxyz",
            pose_origin=CLEAN_POSE_ORIGIN,
        )
        root_position = native.body_position_world[:, _ROOT_BODY_INDEX]
        root_quaternion = native.body_quaternion_world_wxyz[
            :, _ROOT_BODY_INDEX
        ]
        root_linear_velocity = native.body_linear_velocity_world[
            :, _ROOT_BODY_INDEX
        ]
        root_angular_velocity = native.body_angular_velocity_world[
            :, _ROOT_BODY_INDEX
        ]
        sole_position = native.body_position_world[:, _FOOT_BODY_INDICES]
        sole_quaternion = native.body_quaternion_world_wxyz[
            :, _FOOT_BODY_INDICES
        ]
        frame_count = native.frame_count
        clip_id = Path(native.relative_path).parent.as_posix()
        clip = CanonicalClip(
            clip_id=clip_id,
            fps=float(native.fps),
            source=source,
            joint_names=ISAACLAB_JOINT_NAMES,
            body_names=ISAACLAB_BODY_NAMES,
            root_position_world=root_position,
            root_quaternion_world_wxyz=root_quaternion,
            joint_position=native.joint_position,
            root_linear_velocity_world=root_linear_velocity,
            root_angular_velocity_world=root_angular_velocity,
            joint_velocity=native.joint_velocity,
            body_position_world=native.body_position_world,
            body_quaternion_world_wxyz=native.body_quaternion_world_wxyz,
            body_linear_velocity_world=native.body_linear_velocity_world,
            body_angular_velocity_world=native.body_angular_velocity_world,
            sole_position_world=sole_position,
            sole_quaternion_world_wxyz=sole_quaternion,
            heel_position_world=sole_position,
            toe_position_world=sole_position,
            contact=np.zeros((frame_count, 2), dtype=np.float32),
            contact_confidence=np.zeros((frame_count, 2), dtype=np.float32),
            commands=_infer_commands(
                root_quaternion,
                root_linear_velocity,
                root_angular_velocity,
            ),
            terrain=None,
            action_tags=action_tags,
        )
        clip.validate()
        yield clip
