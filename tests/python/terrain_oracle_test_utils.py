"""Small, hand-authored fixtures for terrain-oracle contract tests."""

from __future__ import annotations

import numpy as np

from mm_sonic.terrain_oracle.canonical import (
    CanonicalClip,
    CommandTrack,
    SourceIdentity,
    derive_clip_kinematics,
)


def synthetic_canonical_clip(
    *, frames: int = 6, observed_commands: bool = True
) -> CanonicalClip:
    """Build a valid 50 Hz clip moving at exactly 0.4 m/s along world X."""

    if frames < 2:
        raise ValueError("synthetic canonical clips need at least two frames")
    timeline = np.arange(frames, dtype=np.float32)
    root_position = np.zeros((frames, 3), dtype=np.float32)
    root_position[:, 0] = timeline * np.float32(0.008)
    identity_quaternion = np.zeros((frames, 4), dtype=np.float32)
    identity_quaternion[:, 0] = 1.0
    clip = CanonicalClip(
        clip_id="synthetic-forward",
        fps=50.0,
        source=SourceIdentity(
            source_format="synthetic",
            source_path="synthetic://forward",
            source_sha256="0" * 64,
            coordinate_convention="world-right-handed-z-up",
            quaternion_convention="wxyz",
        ),
        root_position_world=root_position,
        root_quaternion_world_wxyz=identity_quaternion,
        joint_position=np.zeros((frames, 29), dtype=np.float32),
        root_linear_velocity_world=np.zeros((frames, 3), dtype=np.float32),
        root_angular_velocity_world=np.zeros((frames, 3), dtype=np.float32),
        joint_velocity=np.zeros((frames, 29), dtype=np.float32),
        body_position_world=np.zeros((frames, 30, 3), dtype=np.float32),
        body_quaternion_world_wxyz=np.broadcast_to(
            identity_quaternion[:, None, :], (frames, 30, 4)
        ).copy(),
        body_linear_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        body_angular_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        sole_position_world=np.zeros((frames, 2, 3), dtype=np.float32),
        sole_quaternion_world_wxyz=np.broadcast_to(
            identity_quaternion[:, None, :], (frames, 2, 4)
        ).copy(),
        heel_position_world=np.zeros((frames, 2, 3), dtype=np.float32),
        toe_position_world=np.zeros((frames, 2, 3), dtype=np.float32),
        contact=np.zeros((frames, 2), dtype=np.float32),
        contact_confidence=np.ones((frames, 2), dtype=np.float32),
        commands=CommandTrack(
            observed_travel_stick_xy=np.zeros((frames, 2), dtype=np.float32),
            observed_facing_stick_xy=np.zeros((frames, 2), dtype=np.float32),
            observed_mask=np.full(frames, observed_commands, dtype=np.bool_),
            inferred_velocity_local_xy=np.tile(
                np.array([0.4, 0.0], dtype=np.float32), (frames, 1)
            ),
            inferred_facing_local_xy=np.tile(
                np.array([1.0, 0.0], dtype=np.float32), (frames, 1)
            ),
            inferred_yaw_rate_rad_s=np.zeros(frames, dtype=np.float32),
        ),
        terrain=None,
        action_tags=("walk",),
    )
    return derive_clip_kinematics(clip)
