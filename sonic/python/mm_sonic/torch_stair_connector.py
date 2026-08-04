"""Offline stair-motion connector geometry and trajectory primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .joints import ContractError, PINNED_TARGET_TO_SOURCE_PERMUTATION


_LEG_JOINT_INDICES = tuple(
    np.asarray(
        tuple(
            target
            for target, source in enumerate(
                PINNED_TARGET_TO_SOURCE_PERMUTATION
            )
            if start <= source < start + 6
        ),
        dtype=np.int64,
    )
    for start in (0, 6)
)


@dataclass(frozen=True)
class PlanarFootAlignment:
    rotation_xy: np.ndarray
    translation_xy: np.ndarray
    yaw_rad: float
    aligned_source_xy: np.ndarray
    residual_xy: np.ndarray
    maximum_residual_m: float


@dataclass(frozen=True)
class StairConnectorBoundary:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_velocity_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    foot_position_world: np.ndarray

    def __post_init__(self) -> None:
        fields = (
            ("joint_position", (29,), "connector joint position"),
            ("joint_velocity", (29,), "connector joint velocity"),
            ("root_position_world", (3,), "connector root position"),
            ("root_velocity_world", (3,), "connector root velocity"),
            (
                "root_orientation_world_wxyz",
                (4,),
                "connector root orientation",
            ),
            ("foot_position_world", (2, 3), "connector foot position"),
        )
        for name, shape, label in fields:
            owned = _finite_array(getattr(self, name), shape, label).copy()
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        if (
            abs(
                float(
                    np.linalg.norm(self.root_orientation_world_wxyz)
                )
                - 1.0
            )
            > 1e-4
        ):
            raise ContractError("connector root orientation must be unit length")


@dataclass(frozen=True)
class DoubleSupportConnector:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    foot_position_world: np.ndarray
    maximum_foot_error_m: float
    maximum_joint_speed_rad_s: float
    outgoing_alignment: PlanarFootAlignment


@dataclass(frozen=True)
class SingleStepConnector:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    foot_position_world: np.ndarray
    target_foot_position_world: np.ndarray
    pivot_foot: int
    maximum_foot_error_m: float
    maximum_joint_speed_rad_s: float
    maximum_joint_acceleration_rad_s2: float


class BoundedFootKinematics:
    """Add explicit joint bounds to any public batch foot-FK implementation."""

    def __init__(
        self,
        foot_kinematics: object,
        *,
        sole_kinematics: object | None = None,
        maximum_function_evaluations: int = 64,
        posture_weight: float = 1e-6,
        foot_tolerance_m: float = 0.005,
        sole_tolerance_m: float = 0.006,
    ) -> None:
        if (
            not callable(getattr(foot_kinematics, "foot_positions", None))
            or (
                sole_kinematics is not None
                and not callable(
                    getattr(sole_kinematics, "sole_points", None)
                )
            )
            or type(maximum_function_evaluations) is not int
            or maximum_function_evaluations < 1
            or not math.isfinite(float(posture_weight))
            or float(posture_weight) <= 0.0
            or not math.isfinite(float(foot_tolerance_m))
            or float(foot_tolerance_m) <= 0.0
            or not math.isfinite(float(sole_tolerance_m))
            or float(sole_tolerance_m) <= 0.0
        ):
            raise ContractError("bounded foot kinematics inputs are invalid")
        self._foot_kinematics = foot_kinematics
        self._sole_kinematics = sole_kinematics
        self._maximum_function_evaluations = maximum_function_evaluations
        self._posture_scale = math.sqrt(float(posture_weight))
        self._foot_tolerance_m = float(foot_tolerance_m)
        self._sole_tolerance_m = float(sole_tolerance_m)

    def foot_positions(
        self,
        joint_positions: object,
        root_positions_world: object,
        root_orientations_world_wxyz: object,
    ) -> np.ndarray:
        return np.asarray(
            self._foot_kinematics.foot_positions(
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ),
            dtype=np.float64,
        )

    def sole_points(
        self,
        joint_positions: object,
        root_positions_world: object,
        root_orientations_world_wxyz: object,
    ) -> np.ndarray:
        if self._sole_kinematics is None:
            raise ContractError("bounded sole kinematics are unavailable")
        return np.asarray(
            self._sole_kinematics.sole_points(
                joint_positions,
                root_positions_world,
                root_orientations_world_wxyz,
            ),
            dtype=np.float64,
        )

    def solve_leg_positions_bounded(
        self,
        joint_position: object,
        root_position_world: object,
        root_orientation_world_wxyz: object,
        solve_feet: object,
        target_foot_position_world: object,
        joint_position_lower: object,
        joint_position_upper: object,
    ) -> np.ndarray:
        joints = _finite_array(
            joint_position,
            (29,),
            "bounded IK joint position",
        )
        root = _finite_array(
            root_position_world,
            (3,),
            "bounded IK root position",
        )
        orientation = _finite_array(
            root_orientation_world_wxyz,
            (4,),
            "bounded IK root orientation",
        )
        feet_mask = np.asarray(solve_feet)
        targets = _finite_array(
            target_foot_position_world,
            (2, 3),
            "bounded IK foot target",
        )
        lower = _finite_array(
            joint_position_lower,
            (29,),
            "bounded IK lower limit",
        )
        upper = _finite_array(
            joint_position_upper,
            (29,),
            "bounded IK upper limit",
        )
        if (
            feet_mask.dtype != np.bool_
            or feet_mask.shape != (2,)
            or not bool(feet_mask.any())
            or abs(float(np.linalg.norm(orientation)) - 1.0) > 1e-4
            or np.any(lower >= upper)
            or np.any(joints < lower)
            or np.any(joints > upper)
        ):
            raise ContractError("bounded foot IK contract is invalid")
        selected_feet = np.flatnonzero(feet_mask)
        selected_joints = np.concatenate(
            [_LEG_JOINT_INDICES[index] for index in selected_feet]
        )
        seed = joints[selected_joints].copy()
        selected_lower = lower[selected_joints]
        selected_upper = upper[selected_joints]

        def candidate(value: np.ndarray) -> np.ndarray:
            output = joints.copy()
            output[selected_joints] = value
            return output

        def residual(value: np.ndarray) -> np.ndarray:
            output = candidate(value)
            feet = self.foot_positions(
                output[None, :],
                root[None, :],
                orientation[None, :],
            )
            if feet.shape != (1, 2, 3) or not np.isfinite(feet).all():
                raise ContractError("bounded foot IK FK output is invalid")
            foot_error = np.concatenate(
                [
                    feet[0, index] - targets[index]
                    for index in selected_feet
                ]
            )
            posture = self._posture_scale * (value - seed)
            return np.concatenate((foot_error, posture))

        try:
            from scipy.optimize import least_squares

            result = least_squares(
                residual,
                seed,
                bounds=(selected_lower, selected_upper),
                max_nfev=self._maximum_function_evaluations,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        except Exception as error:
            raise ContractError("bounded foot IK solve failed") from error
        output = candidate(result.x)
        feet = self.foot_positions(
            output[None, :],
            root[None, :],
            orientation[None, :],
        )[0]
        if any(
            float(np.linalg.norm(feet[index] - targets[index]))
            > self._foot_tolerance_m
            for index in selected_feet
        ):
            raise ContractError("bounded foot IK target is unreachable")
        return np.ascontiguousarray(output)

    def solve_leg_sole_positions_bounded(
        self,
        joint_position: object,
        root_position_world: object,
        root_orientation_world_wxyz: object,
        solve_feet: object,
        target_sole_position_world: object,
        joint_position_lower: object,
        joint_position_upper: object,
    ) -> np.ndarray:
        """Solve selected legs to complete oriented sole-point targets."""

        if self._sole_kinematics is None:
            raise ContractError("bounded sole kinematics are unavailable")
        joints = _finite_array(
            joint_position,
            (29,),
            "bounded sole IK joint position",
        )
        root = _finite_array(
            root_position_world,
            (3,),
            "bounded sole IK root position",
        )
        orientation = _finite_array(
            root_orientation_world_wxyz,
            (4,),
            "bounded sole IK root orientation",
        )
        feet_mask = np.asarray(solve_feet)
        targets = np.asarray(target_sole_position_world, dtype=np.float64)
        lower = _finite_array(
            joint_position_lower,
            (29,),
            "bounded sole IK lower limit",
        )
        upper = _finite_array(
            joint_position_upper,
            (29,),
            "bounded sole IK upper limit",
        )
        if (
            feet_mask.dtype != np.bool_
            or feet_mask.shape != (2,)
            or not bool(feet_mask.any())
            or targets.ndim != 3
            or targets.shape[0] != 2
            or targets.shape[1] < 3
            or targets.shape[2] != 3
            or not np.isfinite(targets).all()
            or abs(float(np.linalg.norm(orientation)) - 1.0) > 1e-4
            or np.any(lower >= upper)
            or np.any(joints < lower)
            or np.any(joints > upper)
        ):
            raise ContractError("bounded sole IK contract is invalid")
        selected_feet = np.flatnonzero(feet_mask)
        selected_joints = np.concatenate(
            [_LEG_JOINT_INDICES[index] for index in selected_feet]
        )
        seed = joints[selected_joints].copy()
        selected_lower = lower[selected_joints]
        selected_upper = upper[selected_joints]

        def candidate(value: np.ndarray) -> np.ndarray:
            output = joints.copy()
            output[selected_joints] = value
            return output

        def residual(value: np.ndarray) -> np.ndarray:
            output = candidate(value)
            soles = self.sole_points(
                output[None, :],
                root[None, :],
                orientation[None, :],
            )
            if (
                soles.shape != (1, *targets.shape)
                or not np.isfinite(soles).all()
            ):
                raise ContractError("bounded sole IK FK output is invalid")
            sole_error = np.concatenate(
                [
                    (soles[0, index] - targets[index]).reshape(-1)
                    for index in selected_feet
                ]
            )
            posture = self._posture_scale * (value - seed)
            return np.concatenate((sole_error, posture))

        try:
            from scipy.optimize import least_squares

            result = least_squares(
                residual,
                seed,
                bounds=(selected_lower, selected_upper),
                max_nfev=self._maximum_function_evaluations,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        except Exception as error:
            raise ContractError("bounded sole IK solve failed") from error
        output = candidate(result.x)
        soles = self.sole_points(
            output[None, :],
            root[None, :],
            orientation[None, :],
        )[0]
        if any(
            float(
                np.linalg.norm(
                    soles[index] - targets[index],
                    axis=1,
                ).max()
            )
            > self._sole_tolerance_m
            for index in selected_feet
        ):
            raise ContractError("bounded sole IK target is unreachable")
        return np.ascontiguousarray(output)


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{label} must have finite shape {shape}")
    return np.ascontiguousarray(array)


def planar_foot_alignment(
    source_foot_xy: object,
    target_foot_xy: object,
) -> PlanarFootAlignment:
    """Return the least-squares proper SE(2) alignment of two labelled feet."""

    source = _finite_array(source_foot_xy, (2, 2), "source foot XY")
    target = _finite_array(target_foot_xy, (2, 2), "target foot XY")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    translation = target_center - source_center @ rotation
    aligned = source @ rotation + translation
    residual = aligned - target
    return PlanarFootAlignment(
        rotation_xy=np.ascontiguousarray(rotation),
        translation_xy=np.ascontiguousarray(translation),
        yaw_rad=math.atan2(float(rotation[0, 1]), float(rotation[0, 0])),
        aligned_source_xy=np.ascontiguousarray(aligned),
        residual_xy=np.ascontiguousarray(residual),
        maximum_residual_m=float(
            np.linalg.norm(residual, axis=1).max()
        ),
    )


def pivot_stance_target(
    foot_position_world: object,
    *,
    pivot_foot: int,
    yaw_delta_rad: float,
) -> np.ndarray:
    """Rotate the other foothold about one planted foot in the horizontal plane."""

    feet = _finite_array(
        foot_position_world,
        (2, 3),
        "pivot stance foot position",
    )
    if (
        type(pivot_foot) is not int
        or pivot_foot not in (0, 1)
        or isinstance(yaw_delta_rad, bool)
        or not isinstance(yaw_delta_rad, (int, float))
        or not math.isfinite(float(yaw_delta_rad))
    ):
        raise ContractError("pivot stance target inputs are invalid")
    swing_foot = 1 - pivot_foot
    cosine = math.cos(float(yaw_delta_rad))
    sine = math.sin(float(yaw_delta_rad))
    rotation = np.asarray(
        ((cosine, sine), (-sine, cosine)),
        dtype=np.float64,
    )
    output = feet.copy()
    relative = feet[swing_foot, :2] - feet[pivot_foot, :2]
    output[swing_foot, :2] = (
        feet[pivot_foot, :2] + relative @ rotation
    )
    return np.ascontiguousarray(output)


def _minimum_jerk_weight(phase: np.ndarray) -> np.ndarray:
    return (
        6.0 * phase**5
        - 15.0 * phase**4
        + 10.0 * phase**3
    )


def single_step_foot_trajectory(
    foot_position_world: object,
    swing_target_world: object,
    *,
    pivot_foot: int,
    frame_count: int,
    swing_clearance_m: float,
) -> np.ndarray:
    """Construct one smooth pivot-held footstep with an endpoint-safe apex."""

    feet = _finite_array(
        foot_position_world,
        (2, 3),
        "single-step foot position",
    )
    target = _finite_array(
        swing_target_world,
        (3,),
        "single-step swing target",
    )
    if (
        type(pivot_foot) is not int
        or pivot_foot not in (0, 1)
        or type(frame_count) is not int
        or frame_count < 3
        or isinstance(swing_clearance_m, bool)
        or not isinstance(swing_clearance_m, (int, float))
        or not math.isfinite(float(swing_clearance_m))
        or float(swing_clearance_m) <= 0.0
    ):
        raise ContractError("single-step foot trajectory inputs are invalid")
    swing_foot = 1 - pivot_foot
    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    smooth = _minimum_jerk_weight(phase)
    output = np.repeat(feet[None, :, :], frame_count, axis=0)
    start = feet[swing_foot]
    output[:, swing_foot] = (
        (1.0 - smooth[:, None]) * start
        + smooth[:, None] * target
    )
    endpoint_height_delta = abs(float(target[2] - start[2]))
    apex_lift = float(swing_clearance_m) + 0.5 * endpoint_height_delta
    # A parabolic bump has non-zero lift velocity at liftoff and touchdown,
    # which creates a large knee-velocity reversal when alternating feet.
    # This sixth-order bump keeps the same apex while its first and second
    # derivatives are zero at both contact boundaries.
    output[:, swing_foot, 2] += (
        64.0
        * apex_lift
        * phase**3
        * (1.0 - phase) ** 3
    )
    output[0, swing_foot] = start
    output[-1, swing_foot] = target
    return np.ascontiguousarray(output)


def single_step_sole_trajectory(
    sole_position_world: object,
    foot_position_world: object,
    foot_target_trajectory_world: object,
    *,
    pivot_foot: int,
    yaw_delta_rad: float,
) -> np.ndarray:
    """Rigidly yaw and translate the swing sole while fixing the pivot sole."""

    soles = np.asarray(sole_position_world, dtype=np.float64)
    feet = _finite_array(
        foot_position_world,
        (2, 3),
        "single-step sole foot position",
    )
    foot_targets = np.asarray(
        foot_target_trajectory_world,
        dtype=np.float64,
    )
    if (
        soles.ndim != 3
        or soles.shape[0] != 2
        or soles.shape[1] < 3
        or soles.shape[2] != 3
        or not np.isfinite(soles).all()
        or foot_targets.ndim != 3
        or foot_targets.shape[1:] != (2, 3)
        or len(foot_targets) < 3
        or not np.isfinite(foot_targets).all()
        or type(pivot_foot) is not int
        or pivot_foot not in (0, 1)
        or isinstance(yaw_delta_rad, bool)
        or not isinstance(yaw_delta_rad, (int, float))
        or not math.isfinite(float(yaw_delta_rad))
    ):
        raise ContractError("single-step sole trajectory inputs are invalid")
    swing_foot = 1 - pivot_foot
    phase = np.linspace(0.0, 1.0, len(foot_targets), dtype=np.float64)
    yaw = float(yaw_delta_rad) * _minimum_jerk_weight(phase)
    offsets = soles[swing_foot] - feet[swing_foot]
    output = np.repeat(soles[None, :, :, :], len(foot_targets), axis=0)
    for frame, angle in enumerate(yaw):
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        rotation = np.asarray(
            ((cosine, sine), (-sine, cosine)),
            dtype=np.float64,
        )
        rotated = offsets.copy()
        rotated[:, :2] = offsets[:, :2] @ rotation
        output[frame, swing_foot] = (
            foot_targets[frame, swing_foot] + rotated
        )
    output[:, pivot_foot] = soles[pivot_foot]
    output[0] = soles
    return np.ascontiguousarray(output)


def cubic_hermite_trajectory(
    start_position: object,
    end_position: object,
    start_velocity: object,
    end_velocity: object,
    *,
    frame_count: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a cubic that exactly preserves endpoint position and velocity."""

    start = np.asarray(start_position, dtype=np.float64)
    end = np.asarray(end_position, dtype=np.float64)
    velocity_start = np.asarray(start_velocity, dtype=np.float64)
    velocity_end = np.asarray(end_velocity, dtype=np.float64)
    if (
        start.ndim != 1
        or start.shape != end.shape
        or start.shape != velocity_start.shape
        or start.shape != velocity_end.shape
        or not all(
            np.isfinite(value).all()
            for value in (start, end, velocity_start, velocity_end)
        )
        or type(frame_count) is not int
        or frame_count < 2
        or isinstance(dt_s, bool)
        or not isinstance(dt_s, (int, float))
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
    ):
        raise ContractError("stair connector Hermite inputs are invalid")
    duration = (frame_count - 1) * float(dt_s)
    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)[:, None]
    phase2 = phase * phase
    phase3 = phase2 * phase
    h00 = 2.0 * phase3 - 3.0 * phase2 + 1.0
    h10 = phase3 - 2.0 * phase2 + phase
    h01 = -2.0 * phase3 + 3.0 * phase2
    h11 = phase3 - phase2
    position = (
        h00 * start
        + h10 * duration * velocity_start
        + h01 * end
        + h11 * duration * velocity_end
    )
    dh00 = (6.0 * phase2 - 6.0 * phase) / duration
    dh10 = 3.0 * phase2 - 4.0 * phase + 1.0
    dh01 = (-6.0 * phase2 + 6.0 * phase) / duration
    dh11 = 3.0 * phase2 - 2.0 * phase
    velocity = (
        dh00 * start
        + dh10 * velocity_start
        + dh01 * end
        + dh11 * velocity_end
    )
    return (
        np.ascontiguousarray(position),
        np.ascontiguousarray(velocity),
    )


def _quaternion_multiply_wxyz(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _interpolate_quaternions(
    start: np.ndarray,
    end: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    terminal = end if float(start @ end) >= 0.0 else -end
    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    weight = 3.0 * phase * phase - 2.0 * phase * phase * phase
    output = (
        (1.0 - weight[:, None]) * start[None, :]
        + weight[:, None] * terminal[None, :]
    )
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return np.ascontiguousarray(output)


def synthesize_double_support_connector(
    incoming: StairConnectorBoundary,
    outgoing: StairConnectorBoundary,
    *,
    kinematics: object,
    frame_count: int,
    dt_s: float,
    maximum_joint_speed_rad_s: float,
    maximum_foot_error_m: float = 0.005,
    joint_position_lower: object | None = None,
    joint_position_upper: object | None = None,
) -> DoubleSupportConnector:
    """Bridge two boundaries while holding both incoming footholds fixed."""

    if (
        not isinstance(incoming, StairConnectorBoundary)
        or not isinstance(outgoing, StairConnectorBoundary)
        or type(frame_count) is not int
        or frame_count < 2
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
        or not math.isfinite(float(maximum_joint_speed_rad_s))
        or float(maximum_joint_speed_rad_s) <= 0.0
        or not math.isfinite(float(maximum_foot_error_m))
        or float(maximum_foot_error_m) <= 0.0
        or not (
            callable(getattr(kinematics, "solve_leg_positions", None))
            or callable(
                getattr(kinematics, "solve_leg_positions_bounded", None)
            )
        )
        or not callable(getattr(kinematics, "foot_positions", None))
    ):
        raise ContractError("double-support connector inputs are invalid")
    if (joint_position_lower is None) != (joint_position_upper is None):
        raise ContractError(
            "double-support connector joint limits must be paired"
        )
    lower = upper = None
    if joint_position_lower is not None:
        lower = _finite_array(
            joint_position_lower,
            (29,),
            "connector joint lower limits",
        )
        upper = _finite_array(
            joint_position_upper,
            (29,),
            "connector joint upper limits",
        )
        if np.any(lower >= upper):
            raise ContractError("double-support connector joint limits are invalid")
        if (
            np.any(incoming.joint_position < lower)
            or np.any(incoming.joint_position > upper)
            or np.any(outgoing.joint_position < lower)
            or np.any(outgoing.joint_position > upper)
        ):
            raise ContractError(
                "double-support connector boundary violates joint limits"
            )
    alignment = planar_foot_alignment(
        outgoing.foot_position_world[:, :2],
        incoming.foot_position_world[:, :2],
    )
    aligned_root = outgoing.root_position_world.copy()
    aligned_root[:2] = (
        aligned_root[:2] @ alignment.rotation_xy
        + alignment.translation_xy
    )
    aligned_root_velocity = outgoing.root_velocity_world.copy()
    aligned_root_velocity[:2] = (
        aligned_root_velocity[:2] @ alignment.rotation_xy
    )
    half_yaw = 0.5 * alignment.yaw_rad
    alignment_quaternion = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        dtype=np.float64,
    )
    aligned_orientation = _quaternion_multiply_wxyz(
        alignment_quaternion,
        outgoing.root_orientation_world_wxyz,
    )
    aligned_orientation /= np.linalg.norm(aligned_orientation)
    joints, _ = cubic_hermite_trajectory(
        incoming.joint_position,
        outgoing.joint_position,
        incoming.joint_velocity,
        outgoing.joint_velocity,
        frame_count=frame_count,
        dt_s=dt_s,
    )
    roots, _ = cubic_hermite_trajectory(
        incoming.root_position_world,
        aligned_root,
        incoming.root_velocity_world,
        aligned_root_velocity,
        frame_count=frame_count,
        dt_s=dt_s,
    )
    orientations = _interpolate_quaternions(
        incoming.root_orientation_world_wxyz,
        aligned_orientation,
        frame_count,
    )
    target_feet = incoming.foot_position_world
    solve_mask = np.ones(2, dtype=np.bool_)
    solved = np.empty_like(joints)
    bounded_solve = getattr(
        kinematics,
        "solve_leg_positions_bounded",
        None,
    )
    for frame in range(frame_count):
        seed = joints[frame]
        if frame > 0:
            seed = solved[frame - 1] + joints[frame] - joints[frame - 1]
        if lower is not None and upper is not None:
            seed = np.clip(seed, lower, upper)
        if callable(bounded_solve) and lower is not None and upper is not None:
            frame_lower = lower
            frame_upper = upper
            if frame > 0:
                maximum_step = (
                    float(maximum_joint_speed_rad_s) * float(dt_s)
                )
                frame_lower = np.maximum(
                    lower,
                    solved[frame - 1] - maximum_step,
                )
                frame_upper = np.minimum(
                    upper,
                    solved[frame - 1] + maximum_step,
                )
                seed = np.clip(seed, frame_lower, frame_upper)
            solved[frame] = bounded_solve(
                seed,
                roots[frame],
                orientations[frame],
                solve_mask,
                target_feet,
                frame_lower,
                frame_upper,
            )
        else:
            solved[frame] = kinematics.solve_leg_positions(
                seed,
                roots[frame],
                orientations[frame],
                solve_mask,
                target_feet,
            )
        if (
            lower is not None
            and upper is not None
            and (
                np.any(solved[frame] < lower - 1e-9)
                or np.any(solved[frame] > upper + 1e-9)
            )
        ):
            raise ContractError(
                "double-support connector IK violates joint limits"
            )
    feet = np.asarray(
        kinematics.foot_positions(solved, roots, orientations),
        dtype=np.float64,
    )
    if feet.shape != (frame_count, 2, 3) or not np.isfinite(feet).all():
        raise ContractError("double-support connector FK output is invalid")
    foot_error = np.linalg.norm(feet - target_feet[None, :, :], axis=2)
    observed_foot_error = float(foot_error.max())
    if observed_foot_error > float(maximum_foot_error_m):
        raise ContractError(
            "double-support connector exceeds the stance-foot error limit"
        )
    velocities = np.gradient(
        solved,
        float(dt_s),
        axis=0,
        edge_order=2 if frame_count >= 3 else 1,
    )
    observed_joint_speed = float(np.abs(velocities).max())
    if observed_joint_speed > float(maximum_joint_speed_rad_s):
        raise ContractError(
            "double-support connector exceeds the joint-speed limit"
        )
    return DoubleSupportConnector(
        joint_position=np.ascontiguousarray(solved),
        joint_velocity=np.ascontiguousarray(velocities),
        root_position_world=np.ascontiguousarray(roots),
        root_orientation_world_wxyz=np.ascontiguousarray(orientations),
        foot_position_world=np.ascontiguousarray(feet),
        maximum_foot_error_m=observed_foot_error,
        maximum_joint_speed_rad_s=observed_joint_speed,
        outgoing_alignment=alignment,
    )


def synthesize_single_step_connector(
    incoming: StairConnectorBoundary,
    *,
    swing_target_world: object,
    pivot_foot: int,
    root_yaw_delta_rad: float,
    kinematics: object,
    frame_count: int,
    dt_s: float,
    swing_clearance_m: float,
    root_height_offset_m: float = 0.0,
    maximum_joint_speed_rad_s: float,
    maximum_joint_acceleration_rad_s2: float = 100.0,
    maximum_foot_error_m: float = 0.005,
    joint_position_lower: object | None = None,
    joint_position_upper: object | None = None,
) -> SingleStepConnector:
    """Synthesize one planted-foot pivot step toward a new contact stance."""

    target = _finite_array(
        swing_target_world,
        (3,),
        "single-step connector swing target",
    )
    if (
        not isinstance(incoming, StairConnectorBoundary)
        or type(pivot_foot) is not int
        or pivot_foot not in (0, 1)
        or isinstance(root_yaw_delta_rad, bool)
        or not isinstance(root_yaw_delta_rad, (int, float))
        or not math.isfinite(float(root_yaw_delta_rad))
        or type(frame_count) is not int
        or frame_count < 3
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
        or not math.isfinite(float(swing_clearance_m))
        or float(swing_clearance_m) <= 0.0
        or isinstance(root_height_offset_m, bool)
        or not isinstance(root_height_offset_m, (int, float))
        or not math.isfinite(float(root_height_offset_m))
        or not math.isfinite(float(maximum_joint_speed_rad_s))
        or float(maximum_joint_speed_rad_s) <= 0.0
        or not math.isfinite(float(maximum_joint_acceleration_rad_s2))
        or float(maximum_joint_acceleration_rad_s2) <= 0.0
        or not math.isfinite(float(maximum_foot_error_m))
        or float(maximum_foot_error_m) <= 0.0
        or not (
            callable(getattr(kinematics, "solve_leg_positions", None))
            or callable(
                getattr(kinematics, "solve_leg_positions_bounded", None)
            )
        )
        or not callable(getattr(kinematics, "foot_positions", None))
    ):
        raise ContractError("single-step connector inputs are invalid")
    if (joint_position_lower is None) != (joint_position_upper is None):
        raise ContractError(
            "single-step connector joint limits must be paired"
        )
    lower = upper = None
    if joint_position_lower is not None:
        lower = _finite_array(
            joint_position_lower,
            (29,),
            "single-step connector joint lower limits",
        )
        upper = _finite_array(
            joint_position_upper,
            (29,),
            "single-step connector joint upper limits",
        )
        if (
            np.any(lower >= upper)
            or np.any(incoming.joint_position < lower)
            or np.any(incoming.joint_position > upper)
        ):
            raise ContractError(
                "single-step connector joint limits are invalid"
            )
    foot_targets = single_step_foot_trajectory(
        incoming.foot_position_world,
        target,
        pivot_foot=pivot_foot,
        frame_count=frame_count,
        swing_clearance_m=swing_clearance_m,
    )
    sole_solver = getattr(
        kinematics,
        "solve_leg_sole_positions_bounded",
        None,
    )
    sole_targets = None
    if callable(sole_solver):
        try:
            initial_soles = np.asarray(
                kinematics.sole_points(
                    incoming.joint_position[None, :],
                    incoming.root_position_world[None, :],
                    incoming.root_orientation_world_wxyz[None, :],
                )[0],
                dtype=np.float64,
            )
        except Exception as error:
            raise ContractError(
                "single-step connector sole FK failed"
            ) from error
        sole_targets = single_step_sole_trajectory(
            initial_soles,
            incoming.foot_position_world,
            foot_targets,
            pivot_foot=pivot_foot,
            yaw_delta_rad=root_yaw_delta_rad,
        )
    terminal_feet = foot_targets[-1]
    start_foot_center = incoming.foot_position_world.mean(axis=0)
    end_foot_center = terminal_feet.mean(axis=0)
    terminal_root = incoming.root_position_world.copy()
    terminal_root += end_foot_center - start_foot_center
    terminal_root[2] += float(root_height_offset_m)
    half_yaw = 0.5 * float(root_yaw_delta_rad)
    yaw_rotation = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        dtype=np.float64,
    )
    terminal_orientation = _quaternion_multiply_wxyz(
        yaw_rotation,
        incoming.root_orientation_world_wxyz,
    )
    terminal_orientation /= np.linalg.norm(terminal_orientation)
    roots, _ = cubic_hermite_trajectory(
        incoming.root_position_world,
        terminal_root,
        incoming.root_velocity_world,
        np.zeros(3, dtype=np.float64),
        frame_count=frame_count,
        dt_s=dt_s,
    )
    orientations = _interpolate_quaternions(
        incoming.root_orientation_world_wxyz,
        terminal_orientation,
        frame_count,
    )
    reference = np.repeat(
        incoming.joint_position[None, :],
        frame_count,
        axis=0,
    )
    solved = np.empty_like(reference)
    bounded_solve = getattr(
        kinematics,
        "solve_leg_positions_bounded",
        None,
    )
    solve_mask = np.ones(2, dtype=np.bool_)
    maximum_step = float(maximum_joint_speed_rad_s) * float(dt_s)
    maximum_second_step = (
        float(maximum_joint_acceleration_rad_s2) * float(dt_s) ** 2
    )
    preceding_joint_position = (
        incoming.joint_position
        - incoming.joint_velocity * float(dt_s)
    )
    for frame in range(frame_count):
        seed = (
            reference[frame]
            if frame == 0
            else solved[frame - 1]
        )
        frame_lower = lower
        frame_upper = upper
        if lower is not None and upper is not None:
            if frame > 0:
                frame_lower = np.maximum(
                    lower,
                    solved[frame - 1] - maximum_step,
                )
                frame_upper = np.minimum(
                    upper,
                    solved[frame - 1] + maximum_step,
                )
                previous_previous = (
                    preceding_joint_position
                    if frame == 1
                    else solved[frame - 2]
                )
                acceleration_center = (
                    2.0 * solved[frame - 1] - previous_previous
                )
                frame_lower = np.maximum(
                    frame_lower,
                    acceleration_center - maximum_second_step,
                )
                frame_upper = np.minimum(
                    frame_upper,
                    acceleration_center + maximum_second_step,
                )
                if np.any(frame_lower >= frame_upper):
                    raise ContractError(
                        "single-step connector dynamic bounds are empty"
                    )
            seed = np.clip(seed, frame_lower, frame_upper)
        if (
            callable(sole_solver)
            and sole_targets is not None
            and frame_lower is not None
        ):
            solved[frame] = sole_solver(
                seed,
                roots[frame],
                orientations[frame],
                solve_mask,
                sole_targets[frame],
                frame_lower,
                frame_upper,
            )
        elif callable(bounded_solve) and frame_lower is not None:
            solved[frame] = bounded_solve(
                seed,
                roots[frame],
                orientations[frame],
                solve_mask,
                foot_targets[frame],
                frame_lower,
                frame_upper,
            )
        else:
            solved[frame] = kinematics.solve_leg_positions(
                seed,
                roots[frame],
                orientations[frame],
                solve_mask,
                foot_targets[frame],
            )
        if (
            lower is not None
            and (
                np.any(solved[frame] < lower - 1e-9)
                or np.any(solved[frame] > upper + 1e-9)
            )
        ):
            raise ContractError(
                "single-step connector IK violates joint limits"
            )
    feet = np.asarray(
        kinematics.foot_positions(solved, roots, orientations),
        dtype=np.float64,
    )
    if feet.shape != (frame_count, 2, 3) or not np.isfinite(feet).all():
        raise ContractError("single-step connector FK output is invalid")
    observed_foot_error = float(
        np.linalg.norm(feet - foot_targets, axis=2).max()
    )
    if observed_foot_error > float(maximum_foot_error_m):
        raise ContractError(
            "single-step connector exceeds the foot error limit"
        )
    velocities = np.gradient(
        solved,
        float(dt_s),
        axis=0,
        edge_order=2 if frame_count >= 3 else 1,
    )
    observed_joint_speed = float(np.abs(velocities).max())
    if observed_joint_speed > float(maximum_joint_speed_rad_s):
        raise ContractError(
            "single-step connector exceeds the joint-speed limit"
        )
    acceleration_path = np.concatenate(
        (preceding_joint_position[None, :], solved),
        axis=0,
    )
    observed_joint_acceleration = float(
        np.abs(np.diff(acceleration_path, n=2, axis=0)).max()
        / float(dt_s) ** 2
    )
    if observed_joint_acceleration > (
        float(maximum_joint_acceleration_rad_s2) + 1e-6
    ):
        raise ContractError(
            "single-step connector exceeds the joint-acceleration limit"
        )
    return SingleStepConnector(
        joint_position=np.ascontiguousarray(solved),
        joint_velocity=np.ascontiguousarray(velocities),
        root_position_world=np.ascontiguousarray(roots),
        root_orientation_world_wxyz=np.ascontiguousarray(orientations),
        foot_position_world=np.ascontiguousarray(feet),
        target_foot_position_world=np.ascontiguousarray(terminal_feet),
        pivot_foot=pivot_foot,
        maximum_foot_error_m=observed_foot_error,
        maximum_joint_speed_rad_s=observed_joint_speed,
        maximum_joint_acceleration_rad_s2=(
            observed_joint_acceleration
        ),
    )
