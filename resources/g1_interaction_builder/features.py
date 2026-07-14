from collections.abc import Sequence

import numpy as np

from resources import quat as holden_quat
from resources.g1_terrain_builder.schema import SkeletonSpec

from .schema import (
    FeatureGroup,
    FeatureSet,
    G1_SKELETON,
    InteractionHand,
    InteractionValidationError,
    LabeledInteractionClip,
)


FEATURE_GROUPS = (
    FeatureGroup("pose", 0, 33),
    FeatureGroup("trajectory", 33, 45),
    FeatureGroup("grasp", 45, 57),
    FeatureGroup("root_target", 57, 65),
    FeatureGroup("context", 65, 71),
)
POSE_BONES = (7, 13, 1, 16)
FUTURE_OFFSETS = (8, 17, 25)


def normalize_feature_groups(
    raw: np.ndarray,
    groups: Sequence[FeatureGroup],
) -> FeatureSet:
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise InteractionValidationError(
            "feature_shape", f"expected a matrix, got shape {raw.shape}"
        )
    groups = tuple(groups)
    values = raw.astype(np.float32, copy=True)
    offsets = np.zeros(raw.shape[1], np.float32)
    scales = np.ones(raw.shape[1], np.float32)

    for group in groups:
        group_values = raw[:, group.start:group.stop]
        if not np.isfinite(group_values).all():
            raise InteractionValidationError(
                "non_finite_features",
                f"group {group.name} contains non-finite values",
            )
        group_offsets = np.mean(group_values, axis=0)
        component_std = np.std(group_values, axis=0)
        group_scale = max(float(np.mean(component_std)), 1e-5)
        offsets[group.start:group.stop] = group_offsets
        scales[group.start:group.stop] = group_scale
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values[:, group.start:group.stop] = (
                group_values - group_offsets
            ) / group_scale

    if not (
        np.isfinite(values).all()
        and np.isfinite(offsets).all()
        and np.isfinite(scales).all()
    ):
        raise InteractionValidationError(
            "non_finite_features",
            "normalized feature matrix contains non-finite values",
        )
    return FeatureSet(values, offsets, scales, groups)


def _raw_clip_features(labeled: LabeledInteractionClip) -> np.ndarray:
    clip = labeled.motion
    world_rot, world_pos, world_vel, world_ang = holden_quat.fk_vel(
        clip.rotations,
        clip.positions,
        clip.velocities,
        clip.angular_velocities,
        G1_SKELETON.parents,
    )
    object_vel = clip.object_velocities
    object_ang = clip.object_angular_velocities
    hand_bone = (
        23 if labeled.active_hand == InteractionHand.LEFT else 30
    )
    pose_bones = POSE_BONES + (hand_bone,)
    rows = np.empty((len(clip.positions), 71), np.float32)

    for frame in range(len(clip.positions)):
        root_position = world_pos[frame, 0]
        root_rotation = world_rot[frame, 0]
        inverse_root = holden_quat.inv(root_rotation)
        grasp_offset_world = holden_quat.mul_vec(
            clip.object_rotations[frame],
            labeled.grasp_position_object,
        )
        grasp_position = (
            clip.object_positions[frame] + grasp_offset_world
        )
        grasp_rotation = holden_quat.mul(
            clip.object_rotations[frame],
            labeled.grasp_rotation_object,
        )
        inverse_grasp = holden_quat.inv(grasp_rotation)
        grasp_velocity = object_vel[frame] + np.cross(
            object_ang[frame], grasp_offset_world
        )

        values: list[float] = []
        for bone in pose_bones:
            values.extend(
                holden_quat.mul_vec(
                    inverse_root,
                    world_pos[frame, bone] - root_position,
                )
            )
        for bone in pose_bones:
            values.extend(
                holden_quat.mul_vec(
                    inverse_root, world_vel[frame, bone]
                )
            )
        root_velocity = holden_quat.mul_vec(
            inverse_root, world_vel[frame, 0]
        )
        values.extend(
            (
                root_velocity[0],
                root_velocity[2],
                world_ang[frame, 0, 1],
            )
        )

        for offset in FUTURE_OFFSETS:
            future = min(frame + offset, len(clip.positions) - 1)
            delta = holden_quat.mul_vec(
                inverse_root,
                world_pos[future, 0] - root_position,
            )
            values.extend((delta[0], delta[2]))
        for offset in FUTURE_OFFSETS:
            future = min(frame + offset, len(clip.positions) - 1)
            facing_world = holden_quat.mul_vec(
                world_rot[future, 0],
                np.array([0.0, 0.0, 1.0]),
            )
            facing_local = holden_quat.mul_vec(
                inverse_root, facing_world
            )
            values.extend((facing_local[0], facing_local[2]))

        hand_position = world_pos[frame, hand_bone]
        hand_rotation = world_rot[frame, hand_bone]
        values.extend(
            holden_quat.mul_vec(
                inverse_grasp, hand_position - grasp_position
            )
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            orientation_error = holden_quat.to_scaled_angle_axis(
                holden_quat.mul(inverse_grasp, hand_rotation)
            )
        values.extend(orientation_error)
        values.extend(
            holden_quat.mul_vec(
                inverse_grasp,
                world_vel[frame, hand_bone] - grasp_velocity,
            )
        )
        values.extend(
            holden_quat.mul_vec(
                inverse_grasp,
                world_ang[frame, hand_bone] - object_ang[frame],
            )
        )

        values.extend(
            holden_quat.mul_vec(
                inverse_grasp, root_position - grasp_position
            )
        )
        root_facing_world = holden_quat.mul_vec(
            root_rotation, np.array([0.0, 0.0, 1.0])
        )
        root_facing_grasp = holden_quat.mul_vec(
            inverse_grasp, root_facing_world
        )
        values.extend(
            (root_facing_grasp[0], root_facing_grasp[2])
        )
        values.extend(
            holden_quat.mul_vec(
                inverse_grasp,
                world_vel[frame, 0] - grasp_velocity,
            )
        )

        table_top = clip.table_position[1] + 0.5 * clip.table_size[1]
        values.append(grasp_position[1] - table_top)
        values.extend(
            (
                labeled.approach_direction_object[0],
                labeled.approach_direction_object[2],
            )
        )
        values.extend(clip.object_dimensions)
        if len(values) != 71:
            raise InteractionValidationError(
                "feature_dimension", f"built {len(values)} values"
            )
        rows[frame] = values
    return rows


def build_features(
    clips: Sequence[LabeledInteractionClip],
    skeleton: SkeletonSpec,
) -> FeatureSet:
    if skeleton.signature() != G1_SKELETON.signature():
        raise InteractionValidationError(
            "skeleton_mismatch", skeleton.signature()
        )
    if not clips:
        raise InteractionValidationError(
            "empty_database", "no database clips"
        )
    raw = np.concatenate(
        [_raw_clip_features(clip) for clip in clips], axis=0
    )
    return normalize_feature_groups(raw, FEATURE_GROUPS)
