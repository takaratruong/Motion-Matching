"""Heading-general endpoint alignment for a MotionBricks drop template."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
)
from .resample import shortest_path_slerp


@dataclass(frozen=True)
class MotionBricksDrop:
    qpos: np.ndarray
    joint_position: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    support_mask: np.ndarray

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        frames = len(qpos)
        joints = np.asarray(self.joint_position, dtype=np.float64)
        root = np.asarray(self.root_position_world, dtype=np.float64)
        quaternion = np.asarray(
            self.root_orientation_world_wxyz, dtype=np.float64
        )
        support = np.asarray(self.support_mask)
        if (
            qpos.shape != (frames, 36)
            or joints.shape != (frames, 29)
            or root.shape != (frames, 3)
            or quaternion.shape != (frames, 4)
            or support.shape != (frames, 2)
            or support.dtype != np.bool_
            or frames < 3
            or not all(
                np.isfinite(value).all()
                for value in (qpos, joints, root, quaternion)
            )
            or not bool(support[0].any())
            or not bool(support[-1].all())
        ):
            raise ContractError("MotionBricks drop output is invalid")
        for name, value in (
            ("qpos", qpos),
            ("joint_position", joints),
            ("root_position_world", root),
            ("root_orientation_world_wxyz", quaternion),
            ("support_mask", support),
        ):
            object.__setattr__(self, name, np.ascontiguousarray(value))


def _target_joints(native_joints: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        native_joints[
            ..., np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
        ]
    )


def _native_qpos(
    joints: np.ndarray, root: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    native = np.empty_like(joints)
    native[
        ..., np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
    ] = joints
    return np.ascontiguousarray(
        np.concatenate((root, quaternion, native), axis=-1)
    )


def pad_motionbricks_exact_endpoints(
    qpos: object, *, hold_frames: int
) -> np.ndarray:
    """Add fixed endpoint context for bounded terrain preparation/recovery."""

    motion = np.asarray(qpos, dtype=np.float64)
    if (
        motion.ndim != 2
        or motion.shape[1] != 36
        or len(motion) < 3
        or not np.isfinite(motion).all()
        or type(hold_frames) is not int
        or hold_frames < 1
    ):
        raise ContractError("MotionBricks endpoint padding is invalid")
    return np.ascontiguousarray(
        np.concatenate(
            (
                np.repeat(
                    motion[:1], hold_frames + 1, axis=0
                ),
                motion[1:-1],
                np.repeat(
                    motion[-1:], hold_frames + 1, axis=0
                ),
            ),
            axis=0,
        )
    )


def terrain_clearance_envelope(
    *,
    minimum_sole_clearance_by_frame_foot: object,
    accepted_clearance_m: float,
    maximum_correction_step_m: float,
) -> np.ndarray:
    """Build the smallest endpoint-anchored lift with bounded frame slope."""

    clearance = np.asarray(
        minimum_sole_clearance_by_frame_foot, dtype=np.float64
    )
    if (
        clearance.ndim != 2
        or clearance.shape[1] != 2
        or len(clearance) < 3
        or not np.isfinite(clearance).all()
        or not math.isfinite(float(accepted_clearance_m))
        or accepted_clearance_m >= 0.0
        or not math.isfinite(float(maximum_correction_step_m))
        or maximum_correction_step_m <= 0.0
    ):
        raise ContractError("terrain clearance envelope input is invalid")
    raw = np.maximum(0.0, float(accepted_clearance_m) - clearance)
    endpoint_capacity = (
        np.minimum(
            np.arange(len(raw)),
            np.arange(len(raw) - 1, -1, -1),
        )[:, None]
        * float(maximum_correction_step_m)
    )
    if bool((raw > endpoint_capacity + 1.0e-9).any()):
        raise ContractError(
            "terrain collision is too close to an exact endpoint"
        )
    envelope = raw.copy()
    step = float(maximum_correction_step_m)
    for frame in range(1, len(envelope)):
        envelope[frame] = np.maximum(
            envelope[frame], envelope[frame - 1] - step
        )
    for frame in range(len(envelope) - 2, -1, -1):
        envelope[frame] = np.maximum(
            envelope[frame], envelope[frame + 1] - step
        )
    envelope = np.minimum(envelope, endpoint_capacity)
    envelope[[0, -1]] = 0.0
    return np.ascontiguousarray(envelope)


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    return np.asarray(values, dtype=np.float64) @ rotation.T


def _warp_and_resample(
    generated_qpos: np.ndarray,
    *,
    exact_start_qpos: np.ndarray,
    exact_stop_qpos: np.ndarray,
    source_frames_per_second: float,
    target_frames_per_second: float,
) -> np.ndarray:
    generated = np.asarray(generated_qpos, dtype=np.float64)
    start = np.asarray(exact_start_qpos, dtype=np.float64)
    stop = np.asarray(exact_stop_qpos, dtype=np.float64)
    if (
        generated.ndim != 2
        or generated.shape[1] != 36
        or len(generated) < 2
        or start.shape != (36,)
        or stop.shape != (36,)
        or source_frames_per_second <= 0.0
        or target_frames_per_second <= 0.0
    ):
        raise ContractError("MotionBricks drop warp input is invalid")
    warped = generated.copy()
    yaw_delta = _yaw_from_wxyz(start[3:7]) - _yaw_from_wxyz(
        warped[0, 3:7]
    )
    warped[:, :2] = (
        start[:2]
        + _rotate_xy(warped[:, :2] - warped[0, :2], yaw_delta)
    )
    warped[:, 2] += start[2] - warped[0, 2]
    yaw_quaternion = np.array(
        (
            math.cos(yaw_delta / 2.0),
            0.0,
            0.0,
            math.sin(yaw_delta / 2.0),
        )
    )
    warped[:, 3:7] = np.stack(
        [
            _multiply_wxyz(yaw_quaternion, quaternion)
            for quaternion in warped[:, 3:7]
        ]
    )
    alpha = np.linspace(0.0, 1.0, len(warped))[:, None]
    for columns in (slice(0, 3), slice(7, 36)):
        warped[:, columns] += (
            (1.0 - alpha)
            * (start[columns] - warped[0, columns])
            + alpha * (stop[columns] - warped[-1, columns])
        )
    for frame, fraction in enumerate(alpha[:, 0]):
        quaternion = (
            warped[frame, 3:7]
            + (1.0 - fraction)
            * (start[3:7] - warped[0, 3:7])
            + fraction
            * (stop[3:7] - warped[-1, 3:7])
        )
        warped[frame, 3:7] = quaternion / np.linalg.norm(quaternion)
    output_count = (
        round(
            (len(warped) - 1)
            * target_frames_per_second
            / source_frames_per_second
        )
        + 1
    )
    coordinate = np.linspace(0.0, len(warped) - 1, output_count)
    native = np.arange(len(warped), dtype=np.float64)
    output = np.empty((output_count, 36), dtype=np.float64)
    for column in (*range(3), *range(7, 36)):
        output[:, column] = np.interp(
            coordinate, native, warped[:, column]
        )
    for frame, value in enumerate(coordinate):
        left = min(int(math.floor(value)), len(warped) - 1)
        right = min(left + 1, len(warped) - 1)
        output[frame, 3:7] = shortest_path_slerp(
            warped[left, 3:7],
            warped[right, 3:7],
            float(value - left),
        )
    output[0] = start
    output[-1] = stop
    return np.ascontiguousarray(output)


def solve_motionbricks_landing_qpos(
    *,
    reference_qpos: object,
    target_foot_position_world: object,
    target_heading_world_xy: object,
    kinematics: object,
    retargeter: object,
) -> np.ndarray:
    """Align one released G1 pose to a heading-local double-support landing."""

    reference = np.asarray(reference_qpos, dtype=np.float64)
    targets = np.asarray(
        target_foot_position_world, dtype=np.float64
    )
    heading = np.asarray(target_heading_world_xy, dtype=np.float64)
    if (
        reference.shape != (36,)
        or targets.shape != (2, 3)
        or heading.shape != (2,)
        or not all(
            np.isfinite(value).all()
            for value in (reference, targets, heading)
        )
        or np.linalg.norm(heading) <= 1.0e-6
        or not callable(getattr(kinematics, "foot_positions", None))
        or not callable(getattr(retargeter, "solve_frame", None))
    ):
        raise ContractError("MotionBricks landing input is invalid")
    heading = heading / np.linalg.norm(heading)
    landing_joints = _target_joints(reference[7:])
    landing_root = reference[:3].copy()
    landing_quaternion = reference[3:7].copy()
    desired_yaw = math.atan2(float(heading[1]), float(heading[0]))
    yaw_delta = desired_yaw - _yaw_from_wxyz(landing_quaternion)
    yaw_quaternion = np.array(
        (
            math.cos(yaw_delta / 2.0),
            0.0,
            0.0,
            math.sin(yaw_delta / 2.0),
        )
    )
    landing_quaternion = _multiply_wxyz(
        yaw_quaternion, landing_quaternion
    )
    actual = kinematics.foot_positions(
        landing_joints[None],
        landing_root[None],
        landing_quaternion[None],
    )[0]
    landing_root += (targets - actual).mean(axis=0)
    solved_joints, solved_root = retargeter.solve_frame(
        joint_position=landing_joints,
        root_position_world=landing_root,
        root_orientation_world_wxyz=landing_quaternion,
        solve_feet=np.ones(2, dtype=np.bool_),
        enforce_target_error_feet=np.ones(2, dtype=np.bool_),
        target_foot_position_world=targets,
        level_feet=np.ones(2, dtype=np.bool_),
    )
    return _native_qpos(
        np.asarray(solved_joints),
        np.asarray(solved_root),
        landing_quaternion,
    )


def project_motionbricks_flight_over_terrain(
    *,
    qpos: object,
    sample_surface: object,
    kinematics: object,
    sole_kinematics: object,
    retargeter: object,
    accepted_clearance_m: float = -0.02,
    maximum_correction_step_m: float = 0.008,
    recovery_frames: int = 10,
) -> np.ndarray:
    """Apply a smooth, bounded full-sole terrain projection to a flight."""

    motion = np.asarray(qpos, dtype=np.float64)
    if (
        motion.ndim != 2
        or motion.shape[1] != 36
        or len(motion) < 4
        or not np.isfinite(motion).all()
        or not callable(sample_surface)
        or not callable(getattr(kinematics, "foot_positions", None))
        or not callable(getattr(sole_kinematics, "sole_points", None))
        or not callable(getattr(retargeter, "solve_frame", None))
        or type(recovery_frames) is not int
        or recovery_frames < 2
    ):
        raise ContractError("MotionBricks terrain projection input is invalid")
    joints_original = _target_joints(motion[:, 7:])
    roots_original = motion[:, :3].copy()
    quaternions = motion[:, 3:7].copy()
    sole = np.asarray(
        sole_kinematics.sole_points(
            joints_original, roots_original, quaternions
        ),
        dtype=np.float64,
    )
    if sole.ndim != 4 or sole.shape[:2] != (len(motion), 2):
        raise ContractError("MotionBricks sole geometry is invalid")
    surface = np.asarray(
        sample_surface(sole[..., :2]), dtype=np.float64
    )
    if surface.shape != sole.shape[:-1] or not np.isfinite(
        surface
    ).all():
        raise ContractError("MotionBricks terrain samples are invalid")
    envelope = terrain_clearance_envelope(
        minimum_sole_clearance_by_frame_foot=(
            sole[..., 2] - surface
        ).min(axis=2),
        accepted_clearance_m=accepted_clearance_m,
        maximum_correction_step_m=maximum_correction_step_m,
    )
    corrected_frames = np.flatnonzero(envelope.max(axis=1) > 1.0e-9)
    if not len(corrected_frames):
        return np.ascontiguousarray(motion.copy())

    joints = joints_original.copy()
    roots = roots_original.copy()
    original_feet = np.asarray(
        kinematics.foot_positions(
            joints_original, roots_original, quaternions
        ),
        dtype=np.float64,
    )
    if original_feet.shape != (len(motion), 2, 3):
        raise ContractError("MotionBricks foot geometry is invalid")
    previous_joints = joints[0].copy()
    previous_root = roots[0].copy()
    for frame in range(1, len(motion) - 1):
        solve_feet = envelope[frame] > 1.0e-9
        if not bool(solve_feet.any()):
            previous_joints = joints[frame].copy()
            previous_root = roots[frame].copy()
            continue
        targets = original_feet[frame].copy()
        targets[:, 2] += envelope[frame]
        solved_joints, solved_root = retargeter.solve_frame(
            joint_position=joints_original[frame],
            root_position_world=roots_original[frame],
            root_orientation_world_wxyz=quaternions[frame],
            solve_feet=solve_feet,
            enforce_target_error_feet=solve_feet,
            target_foot_position_world=targets,
            target_foot_position_weights=np.ones(2, dtype=np.float64),
            level_feet=np.zeros(2, dtype=np.bool_),
            initial_joint_position=previous_joints,
            initial_root_position_world=previous_root,
        )
        joints[frame] = solved_joints
        roots[frame] = solved_root
        previous_joints = joints[frame].copy()
        previous_root = roots[frame].copy()

    last_corrected = int(corrected_frames[-1])
    recovery_count = min(
        recovery_frames, len(motion) - 1 - last_corrected
    )
    joint_delta = joints[last_corrected] - joints_original[last_corrected]
    root_delta = roots[last_corrected] - roots_original[last_corrected]
    for offset in range(1, recovery_count + 1):
        frame = last_corrected + offset
        fraction = offset / recovery_count
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        joints[frame] = (
            joints_original[frame] + (1.0 - smooth) * joint_delta
        )
        roots[frame] = (
            roots_original[frame] + (1.0 - smooth) * root_delta
        )

    output = motion.copy()
    output[:, :3] = roots
    native_joints = np.empty_like(joints)
    native_joints[
        ..., np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
    ] = joints
    output[:, 7:] = native_joints
    output[[0, -1]] = motion[[0, -1]]
    return np.ascontiguousarray(output)


def realize_motionbricks_drop(
    *,
    template_qpos: object,
    start_qpos: object,
    target_foot_position_world: object,
    target_heading_world_xy: object,
    kinematics: object,
    retargeter: object,
    sample_surface: object | None = None,
    start_support: object = (True, True),
    source_frames_per_second: float = 30.0,
    target_frames_per_second: float = 50.0,
) -> MotionBricksDrop:
    """Align one released drop prior to arbitrary heading/contact endpoints."""

    template = np.asarray(template_qpos, dtype=np.float64)
    start = np.asarray(start_qpos, dtype=np.float64)
    targets = np.asarray(
        target_foot_position_world, dtype=np.float64
    )
    heading = np.asarray(target_heading_world_xy, dtype=np.float64)
    support = np.asarray(start_support, dtype=np.bool_)
    if (
        template.ndim != 2
        or template.shape[1] != 36
        or len(template) < 10
        or start.shape != (36,)
        or targets.shape != (2, 3)
        or heading.shape != (2,)
        or support.shape != (2,)
        or not bool(support.any())
        or not all(
            np.isfinite(value).all()
            for value in (template, start, targets, heading)
        )
        or np.linalg.norm(heading) <= 1.0e-6
        or not callable(getattr(kinematics, "foot_positions", None))
        or not callable(getattr(retargeter, "solve_frame", None))
    ):
        raise ContractError("MotionBricks drop input is invalid")
    heading /= np.linalg.norm(heading)

    stop = solve_motionbricks_landing_qpos(
        reference_qpos=template[-1],
        target_foot_position_world=targets,
        target_heading_world_xy=heading,
        kinematics=kinematics,
        retargeter=retargeter,
    )
    qpos = _warp_and_resample(
        template[3:-3],
        exact_start_qpos=start,
        exact_stop_qpos=stop,
        source_frames_per_second=source_frames_per_second,
        target_frames_per_second=target_frames_per_second,
    )
    joints = _target_joints(qpos[:, 7:])
    roots = qpos[:, :3].copy()
    quaternions = qpos[:, 3:7].copy()
    if sample_surface is not None:
        if not callable(sample_surface):
            raise ContractError(
                "MotionBricks drop surface sampler is invalid"
            )
        start_feet = kinematics.foot_positions(
            joints[:1], roots[:1], quaternions[:1]
        )[0]
        forward = np.array((heading[0], heading[1], 0.0))
        lateral = np.array((-heading[1], heading[0], 0.0))
        foot_rotation = np.tile(
            np.stack(
                (forward, lateral, np.array((0.0, 0.0, 1.0))),
                axis=1,
            ),
            (2, 1, 1),
        )
        previous_joints = joints[0].copy()
        previous_root = roots[0].copy()
        for frame in range(1, len(qpos) - 1):
            fraction = frame / (len(qpos) - 1)
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            desired = (
                (1.0 - smooth) * start_feet
                + smooth * targets
            )
            current_feet = kinematics.foot_positions(
                joints[frame : frame + 1],
                roots[frame : frame + 1],
                quaternions[frame : frame + 1],
            )[0]
            current_surface = np.asarray(
                sample_surface(current_feet[:, :2]),
                dtype=np.float64,
            )
            if (
                current_surface.shape != (2,)
                or not np.isfinite(current_surface).all()
            ):
                raise ContractError(
                    "MotionBricks drop surface samples are invalid"
                )
            colliding = (
                current_feet[:, 2] - 0.035 - current_surface
            ) < -0.020
            if not bool(colliding.any()):
                previous_joints = joints[frame].copy()
                previous_root = roots[frame].copy()
                continue
            target = current_feet.copy()
            target[colliding, :2] = desired[colliding, :2]
            surface = np.asarray(
                sample_surface(target[:, :2]), dtype=np.float64
            )
            linear_height = (
                (1.0 - smooth) * start_feet[:, 2]
                + smooth * targets[:, 2]
            )
            clearance_arc = 0.06 * math.sin(math.pi * fraction)
            target[colliding, 2] = np.maximum(
                linear_height[colliding],
                surface[colliding] + 0.035 + clearance_arc,
            )
            solved_joints, solved_root = retargeter.solve_frame(
                joint_position=joints[frame],
                root_position_world=roots[frame],
                root_orientation_world_wxyz=quaternions[frame],
                solve_feet=colliding,
                enforce_target_error_feet=colliding,
                target_foot_position_world=target,
                target_foot_rotation_world=foot_rotation,
                level_feet=colliding,
                initial_joint_position=previous_joints,
                initial_root_position_world=previous_root,
            )
            joints[frame] = solved_joints
            roots[frame] = solved_root
            qpos[frame] = _native_qpos(
                joints[frame],
                roots[frame],
                quaternions[frame],
            )
            previous_joints = joints[frame].copy()
            previous_root = roots[frame].copy()
    output_support = np.zeros((len(qpos), 2), dtype=np.bool_)
    output_support[0] = support
    output_support[-1] = True
    return MotionBricksDrop(
        qpos=qpos,
        joint_position=joints,
        root_position_world=roots,
        root_orientation_world_wxyz=quaternions,
        support_mask=output_support,
    )
