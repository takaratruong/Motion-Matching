import numpy as np

from resources import quat as holden_quat
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
    forward_local_hierarchy,
    quaternions_zup_to_yup,
    vectors_zup_to_yup,
)
from resources.g1_terrain_builder.resample import (
    resample_quaternions_wxyz,
    resample_vectors,
)
from resources.g1_terrain_builder.schema import SkeletonSpec, SourceClip

from .schema import (
    CanonicalInteractionClip,
    ConversionValidationError,
    G1_SKELETON,
    RawInteractionClip,
)


def finite_difference_vectors(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, np.float64)
    out = np.empty_like(values)
    if len(values) < 2:
        out.fill(0.0)
    else:
        out[0] = (values[1] - values[0]) * fps
        out[-1] = (values[-1] - values[-2]) * fps
        if len(values) > 2:
            out[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return out.astype(np.float32)


def finite_difference_quaternions(
    values: np.ndarray, fps: float
) -> np.ndarray:
    """Differentiate rotations in their containing (spatial) frame."""
    q = holden_quat.unroll(
        holden_quat.normalize(np.asarray(values, np.float64))
    )
    out = np.zeros(q.shape[:-1] + (3,), np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if len(q) > 1:
            out[0] = holden_quat.to_scaled_angle_axis(
                holden_quat.mul(q[1], holden_quat.inv(q[0]))
            ) * fps
            out[-1] = holden_quat.to_scaled_angle_axis(
                holden_quat.mul(q[-1], holden_quat.inv(q[-2]))
            ) * fps
        if len(q) > 2:
            delta = holden_quat.mul(q[2:], holden_quat.inv(q[:-2]))
            out[1:-1] = holden_quat.to_scaled_angle_axis(delta) * (
                0.5 * fps
            )
    return out.astype(np.float32)


def resample_discrete_by_source_frame(
    values: np.ndarray, source_frames: np.ndarray
) -> np.ndarray:
    source_frames = np.asarray(source_frames, np.int64)
    if np.any(source_frames < 0) or np.any(source_frames >= len(values)):
        raise ConversionValidationError(
            "frame_count_mismatch", "resampled source frame is out of range"
        )
    return np.asarray(values)[source_frames].copy()


def size_zup_to_yup(size: np.ndarray) -> np.ndarray:
    size = np.asarray(size, np.float32)
    if size.shape != (3,):
        raise ConversionValidationError(
            "invalid_dimensions", f"expected three dimensions, got {size.shape}"
        )
    return size[[0, 2, 1]].copy()


def convert_interaction(
    raw: RawInteractionClip,
    kinematics: G1Kinematics,
    target_fps: float = 25.0,
) -> tuple[CanonicalInteractionClip, SkeletonSpec, dict[str, float]]:
    if target_fps != 25.0:
        raise ConversionValidationError(
            "fps_mismatch", f"schema v1 requires 25 Hz, got {target_fps}"
        )
    raw.validate()
    model = kinematics.model
    joint_ids = np.flatnonzero(np.asarray(model.jnt_qposadr) >= 7)
    if model.nq != 36 or len(joint_ids) != 29:
        raise ConversionValidationError(
            "skeleton_mismatch",
            f"expected nq=36 and 29 body joints, got nq={model.nq}, "
            f"joints={len(joint_ids)}",
        )
    for joint_id in joint_ids:
        if not bool(model.jnt_limited[joint_id]):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        values = raw.qpos[:, qpos_index]
        if np.any(values < lower - 1e-4) or np.any(values > upper + 1e-4):
            raise ConversionValidationError(
                "joint_limit_violation",
                f"joint {model.joint(int(joint_id)).name} outside "
                f"[{lower}, {upper}]",
            )
    source = SourceClip(
        raw.sequence_id,
        raw.fps,
        raw.qpos,
        raw.source_frames,
        raw.object_id,
    )
    try:
        motion, skeleton, report = convert_source_clip(
            source, kinematics, target_fps=target_fps
        )
    except ValueError as error:
        message = str(error)
        known = (
            ("rotational FK error", "fk_rotation_error"),
            ("exported FK error", "fk_error"),
            ("duration error", "duration_error"),
            ("quaternion", "invalid_quaternion"),
            ("non-finite", "non_finite"),
        )
        for fragment, code in known:
            if fragment in message:
                raise ConversionValidationError(code, message) from error
        raise
    if skeleton.signature() != G1_SKELETON.signature():
        raise ConversionValidationError(
            "skeleton_mismatch", skeleton.signature()
        )

    world_positions, world_rotations = forward_local_hierarchy(
        motion.positions.astype(np.float64),
        motion.rotations.astype(np.float64),
        skeleton.parents,
    )
    object_positions = resample_vectors(
        vectors_zup_to_yup(raw.object_positions), raw.fps, target_fps
    )
    object_rotations = resample_quaternions_wxyz(
        quaternions_zup_to_yup(raw.object_rotations), raw.fps, target_fps
    )
    hand_dof = resample_vectors(raw.hand_dof, raw.fps, target_fps)
    hand_contacts = resample_discrete_by_source_frame(
        raw.hand_contacts, motion.source_frames
    )

    velocities = finite_difference_vectors(motion.positions, target_fps)
    angular_velocities = finite_difference_quaternions(
        motion.rotations, target_fps
    )
    hand_dof_velocities = finite_difference_vectors(hand_dof, target_fps)
    object_velocities = finite_difference_vectors(
        object_positions, target_fps
    )
    object_angular_velocities = finite_difference_quaternions(
        object_rotations, target_fps
    )
    toe_positions = world_positions[:, [7, 13]]
    toe_speeds = np.linalg.norm(
        finite_difference_vectors(toe_positions, target_fps), axis=-1
    )
    toe_floor = np.min(toe_positions[:, :, 1], axis=0, keepdims=True)
    foot_contacts = np.logical_and(
        toe_speeds < 0.15,
        toe_positions[:, :, 1] - toe_floor < 0.06,
    ).astype(np.uint8)

    clip = CanonicalInteractionClip(
        sequence_id=raw.sequence_id,
        object_id=raw.object_id,
        fps=target_fps,
        positions=motion.positions.astype(np.float32),
        velocities=velocities,
        rotations=motion.rotations.astype(np.float32),
        angular_velocities=angular_velocities,
        foot_contacts=foot_contacts,
        hand_contacts=hand_contacts,
        hand_dof=hand_dof.astype(np.float32),
        hand_dof_velocities=hand_dof_velocities,
        hand_positions=world_positions[:, [23, 30]].astype(np.float32),
        hand_rotations=world_rotations[:, [23, 30]].astype(np.float32),
        object_positions=object_positions.astype(np.float32),
        object_rotations=object_rotations.astype(np.float32),
        object_velocities=object_velocities,
        object_angular_velocities=object_angular_velocities,
        table_position=vectors_zup_to_yup(raw.table_position).astype(
            np.float32
        ),
        table_rotation=quaternions_zup_to_yup(raw.table_rotation).astype(
            np.float32
        ),
        table_size=size_zup_to_yup(raw.table_size),
        object_dimensions=raw.object_dimensions.astype(np.float32),
        source_frames=motion.source_frames.astype(np.int32),
    )
    clip.validate()
    report = dict(report)
    report.update(
        {
            "target_fps": target_fps,
            "left_hand_contact_frames": int(hand_contacts[:, 0].sum()),
            "right_hand_contact_frames": int(hand_contacts[:, 1].sum()),
        }
    )
    return clip, skeleton, report
