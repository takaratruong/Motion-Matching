"""Hybrid flat/terrain clean-kinematic motion-matching primitives.

The flat Takara/BONES matcher is translation invariant, while Justin's stair
motions are tied to one fixed staircase.  This module supplies the small
world-alignment and inertialization layer needed to connect those two spaces
without changing joystick commands into global-position commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol

import numpy as np

from .terrain_catalog import (
    FUTURE_OFFSETS,
    JUSTIN_STAIR_ASCENT_YAW_WORLD,
    JUSTIN_STAIR_ORIGIN_WORLD_XY,
)
from .terrain_scene import (
    CausalStairObservationFilter,
    ObservedStair,
    StairIntentLatch,
    StairPlacement,
)
from .terrain_motion import complete_route_ready_frames


def _finite(
    value: object, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and be finite")
    return np.ascontiguousarray(array)


def landing_recovery_velocity_world_xy(
    placement: StairPlacement,
    source_clip_name: str,
    *,
    speed_mps: float = 0.16,
) -> np.ndarray:
    """Point a short flat recovery away from the traversed stair edge."""

    if not isinstance(placement, StairPlacement):
        raise ValueError("placement must be StairPlacement")
    name = str(source_clip_name)
    speed = float(speed_mps)
    if (
        not name
        or not math.isfinite(speed)
        or speed <= 0.0
    ):
        raise ValueError("landing recovery inputs are invalid")
    sign = -1.0 if name.startswith(("down_", "grail_down_")) else 1.0
    yaw = float(placement.ascent_yaw_world)
    return np.asarray(
        (
            sign * speed * math.cos(yaw),
            sign * speed * math.sin(yaw),
        ),
        dtype=np.float64,
    )


def _normalize_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("quaternion must be nonzero")
    return value / norm


def _yaw_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = _normalize_xyzw(np.asarray(quaternion)).reshape(4)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _yaw_quaternion_xyzw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray((0.0, 0.0, math.sin(half), math.cos(half)))


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    lx, ly, lz, lw = np.moveaxis(l, -1, 0)
    rx, ry, rz, rw = np.moveaxis(r, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _quat_inverse_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = _normalize_xyzw(quaternion)
    inverse = value.copy()
    inverse[..., :3] *= -1.0
    return inverse


def _scaled_quaternion_offset(
    offset_xyzw: np.ndarray, fraction: float
) -> np.ndarray:
    value = _normalize_xyzw(offset_xyzw)
    value = np.where(value[..., 3:4] < 0.0, -value, value)
    vector = value[..., :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, value[..., 3:4])
    axis = vector / np.maximum(vector_norm, 1.0e-12)
    scaled = 0.5 * angle * float(fraction)
    result = np.concatenate((axis * np.sin(scaled), np.cos(scaled)), axis=-1)
    near = vector_norm[..., 0] < 1.0e-10
    if np.any(near):
        result[near] = np.asarray((0.0, 0.0, 0.0, 1.0))
    return _normalize_xyzw(result)


def _rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


@dataclass(frozen=True)
class KinematicPose:
    root_position_world: np.ndarray
    root_orientation_world_xyzw: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_orientation_world_xyzw: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        values = {
            "root_position_world": _finite(
                self.root_position_world, (3,), "root_position_world"
            ),
            "root_orientation_world_xyzw": _finite(
                self.root_orientation_world_xyzw,
                (4,),
                "root_orientation_world_xyzw",
            ),
            "joint_position": _finite(
                self.joint_position, (29,), "joint_position"
            ),
            "joint_velocity": _finite(
                self.joint_velocity, (29,), "joint_velocity"
            ),
            "body_position_world": _finite(
                self.body_position_world, (30, 3), "body_position_world"
            ),
            "body_orientation_world_xyzw": _finite(
                self.body_orientation_world_xyzw,
                (30, 4),
                "body_orientation_world_xyzw",
            ),
            "body_linear_velocity_world": _finite(
                self.body_linear_velocity_world,
                (30, 3),
                "body_linear_velocity_world",
            ),
            "body_angular_velocity_world": _finite(
                self.body_angular_velocity_world,
                (30, 3),
                "body_angular_velocity_world",
            ),
        }
        values["root_orientation_world_xyzw"] = np.ascontiguousarray(
            _normalize_xyzw(values["root_orientation_world_xyzw"]),
            dtype=np.float32,
        )
        values["body_orientation_world_xyzw"] = np.ascontiguousarray(
            _normalize_xyzw(values["body_orientation_world_xyzw"]),
            dtype=np.float32,
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def root_yaw_world(self) -> float:
        return _yaw_xyzw(self.root_orientation_world_xyzw)


def _reframe_robot_local_xy(
    values: np.ndarray,
    *,
    source_pose: KinematicPose,
    target_pose: KinematicPose,
    positions: bool,
) -> np.ndarray:
    """Express causal robot-local points/vectors in a later robot frame."""

    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1:] != (2,)
        or not np.isfinite(array).all()
    ):
        raise ValueError("robot-local values must have shape [N,2]")
    source_rotation = _rotation_z(source_pose.root_yaw_world)[:2, :2]
    target_rotation = _rotation_z(target_pose.root_yaw_world)[:2, :2]
    world = array @ source_rotation.T
    if positions:
        world += source_pose.root_position_world[:2]
        world -= target_pose.root_position_world[:2]
    return np.ascontiguousarray(world @ target_rotation, dtype=np.float32)


def _reframe_observed_stair(
    stair: ObservedStair,
    *,
    source_pose: KinematicPose,
    target_pose: KinematicPose,
) -> ObservedStair:
    """Preserve one observed stair placement across an internal pose advance."""

    placement = stair.placement_from_robot(
        root_position_world=source_pose.root_position_world,
        root_yaw_world=source_pose.root_yaw_world,
    )
    target_rotation = _rotation_z(target_pose.root_yaw_world)[:2, :2]
    origin_robot = (
        placement.origin_world_xyz[:2]
        - target_pose.root_position_world[:2]
    ) @ target_rotation
    geometry = stair.geometry
    return ObservedStair(
        type(geometry)(
            geometry.run_m,
            geometry.rise_m,
            geometry.width_m,
            geometry.tread_count,
            np.ascontiguousarray(origin_robot, dtype=np.float32),
            math.remainder(
                placement.ascent_yaw_world - target_pose.root_yaw_world,
                2.0 * math.pi,
            ),
            geometry.confidence,
        ),
        base_height_robot_m=(
            float(placement.origin_world_xyz[2])
            - float(target_pose.root_position_world[2])
        ),
    )


@dataclass(frozen=True)
class FlatWorldTransform:
    """Rigid Z-up transform from the matcher's free flat world to the scene."""

    yaw_offset: float
    translation_world: np.ndarray

    def __post_init__(self) -> None:
        yaw = float(self.yaw_offset)
        if not math.isfinite(yaw):
            raise ValueError("yaw_offset must be finite")
        object.__setattr__(self, "yaw_offset", yaw)
        object.__setattr__(
            self,
            "translation_world",
            _finite(self.translation_world, (3,), "translation_world"),
        )

    @classmethod
    def align(
        cls,
        internal_pose: KinematicPose,
        *,
        root_position_world: np.ndarray,
        root_yaw_world: float,
    ) -> "FlatWorldTransform":
        target = _finite(
            root_position_world, (3,), "root_position_world"
        )
        yaw_offset = math.remainder(
            float(root_yaw_world) - internal_pose.root_yaw_world,
            2.0 * math.pi,
        )
        rotation = _rotation_z(yaw_offset)
        translation = target - rotation @ internal_pose.root_position_world
        return cls(yaw_offset, translation)

    def apply(self, pose: KinematicPose) -> KinematicPose:
        rotation = _rotation_z(self.yaw_offset)
        yaw_quaternion = _yaw_quaternion_xyzw(self.yaw_offset)
        body_quaternion = _quat_multiply_xyzw(
            np.broadcast_to(yaw_quaternion, (30, 4)),
            pose.body_orientation_world_xyzw,
        )
        root_quaternion = _quat_multiply_xyzw(
            yaw_quaternion, pose.root_orientation_world_xyzw
        )
        return KinematicPose(
            root_position_world=(
                rotation @ pose.root_position_world
                + self.translation_world
            ),
            root_orientation_world_xyzw=root_quaternion,
            joint_position=pose.joint_position,
            joint_velocity=pose.joint_velocity,
            body_position_world=(
                pose.body_position_world @ rotation.T
                + self.translation_world
            ),
            body_orientation_world_xyzw=body_quaternion,
            body_linear_velocity_world=(
                pose.body_linear_velocity_world @ rotation.T
            ),
            body_angular_velocity_world=(
                pose.body_angular_velocity_world @ rotation.T
            ),
        )

    def velocity_to_internal(self, velocity_world_xy: np.ndarray) -> np.ndarray:
        value = _finite(
            velocity_world_xy, (2,), "velocity_world_xy"
        ).astype(np.float64)
        rotation = _rotation_z(-self.yaw_offset)[:2, :2]
        return np.ascontiguousarray(rotation @ value, dtype=np.float32)

    def yaw_to_internal(self, yaw_world: float) -> float:
        value = float(yaw_world)
        if not math.isfinite(value):
            raise ValueError("yaw_world must be finite")
        return math.remainder(value - self.yaw_offset, 2.0 * math.pi)


class _FlatMatcher(Protocol):
    def reset(self) -> object:
        raise NotImplementedError

    def step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
    ) -> object:
        raise NotImplementedError


def _tensor_numpy(value: object) -> np.ndarray:
    detached = value.detach()  # type: ignore[attr-defined]
    return np.asarray(detached.cpu().numpy(), dtype=np.float32)


def pose_from_motion_match(result: object) -> KinematicPose:
    """Convert a native Torch match result to renderer-friendly NumPy/xyzw."""

    root_wxyz = _tensor_numpy(
        result.root_orientation_world_wxyz  # type: ignore[attr-defined]
    )
    body_wxyz = _tensor_numpy(
        result.body_orientation_world_wxyz  # type: ignore[attr-defined]
    )
    return KinematicPose(
        root_position_world=_tensor_numpy(
            result.root_position_world  # type: ignore[attr-defined]
        ),
        root_orientation_world_xyzw=root_wxyz[[1, 2, 3, 0]],
        joint_position=_tensor_numpy(
            result.joint_position  # type: ignore[attr-defined]
        ),
        joint_velocity=_tensor_numpy(
            result.joint_velocity  # type: ignore[attr-defined]
        ),
        body_position_world=_tensor_numpy(
            result.body_position_world  # type: ignore[attr-defined]
        ),
        body_orientation_world_xyzw=body_wxyz[:, [1, 2, 3, 0]],
        body_linear_velocity_world=_tensor_numpy(
            result.body_linear_velocity_world  # type: ignore[attr-defined]
        ),
        body_angular_velocity_world=_tensor_numpy(
            result.body_angular_velocity_world  # type: ignore[attr-defined]
        ),
    )


@dataclass(frozen=True)
class PreparedFlatKinematicStep:
    matcher_prepared: object
    result: object
    pose: KinematicPose


class FlatKinematicSource:
    """Run the free flat matcher in a rigidly aligned fixed stair scene."""

    def __init__(self, matcher: _FlatMatcher) -> None:
        self.matcher = matcher
        self.transform: FlatWorldTransform | None = None
        self.last_result: object | None = None

    def reset(
        self,
        *,
        root_position_world: np.ndarray,
        root_yaw_world: float,
    ) -> KinematicPose:
        result = self.matcher.reset()
        internal = pose_from_motion_match(result)
        self.transform = FlatWorldTransform.align(
            internal,
            root_position_world=root_position_world,
            root_yaw_world=root_yaw_world,
        )
        self.last_result = result
        return self.transform.apply(internal)

    def reset_supported(
        self,
        *,
        root_position_world: np.ndarray,
        root_yaw_world: float,
        support_pose_world: KinematicPose,
        support_contact: np.ndarray,
        maximum_vertical_correction_m: float = 0.08,
        maximum_planar_correction_m: float = 0.08,
    ) -> KinematicPose:
        """Phase-match and rebase flat motion to the landing support.

        Equal root height does not imply equal sole pose across two motion
        sources.  When the matcher exposes its clean reset index, select a
        compatible joint/contact/velocity phase instead of its arbitrary
        standing reset.  Then align the contacted ankle-roll bodies in all
        three axes before exit inertialization, so the decaying root offset
        cannot drag a planted foot across or into the landing.
        """

        if not isinstance(support_pose_world, KinematicPose):
            raise ValueError("support_pose_world must be a KinematicPose")
        contact = np.asarray(support_contact, dtype=bool)
        vertical_limit = float(maximum_vertical_correction_m)
        planar_limit = float(maximum_planar_correction_m)
        if (
            contact.shape != (2,)
            or not np.any(contact)
            or not math.isfinite(vertical_limit)
            or vertical_limit <= 0.0
            or not math.isfinite(planar_limit)
            or planar_limit <= 0.0
        ):
            raise ValueError("support reset contact and limit are invalid")
        select_reset = getattr(
            self.matcher, "select_supported_reset_row", None
        )
        reset_to_row = getattr(self.matcher, "reset_to_row", None)
        if callable(select_reset) and callable(reset_to_row):
            world_to_root = _rotation_z(
                -support_pose_world.root_yaw_world
            )
            feet = np.asarray((18, 19), dtype=np.int64)
            feet_root_local = (
                support_pose_world.body_position_world[feet]
                - support_pose_world.root_position_world
            ) @ world_to_root.T
            root_velocity_local_xy = (
                world_to_root
                @ support_pose_world.body_linear_velocity_world[0]
            )[:2]
            reset_row = select_reset(
                joint_position=support_pose_world.joint_position,
                feet_position_root_local=feet_root_local,
                support_contact=contact,
                root_velocity_local_xy=root_velocity_local_xy,
            )
            result = reset_to_row(reset_row)
        else:
            result = self.matcher.reset()
        internal = pose_from_motion_match(result)
        yaw_offset = math.remainder(
            float(root_yaw_world) - internal.root_yaw_world,
            2.0 * math.pi,
        )
        rotation = _rotation_z(yaw_offset)
        root_aligned_translation = (
            _finite(
                root_position_world,
                (3,),
                "root_position_world",
            )
            - rotation @ internal.root_position_world
        )
        feet = np.asarray((18, 19), dtype=np.int64)
        support_translation = np.median(
            support_pose_world.body_position_world[feet[contact]]
            - internal.body_position_world[feet[contact]] @ rotation.T,
            axis=0,
        )
        correction = support_translation - root_aligned_translation
        if abs(float(correction[2])) > vertical_limit:
            raise ValueError(
                "flat landing support-height correction exceeds limit"
            )
        if (
            float(np.linalg.norm(correction[:2]))
            > planar_limit
        ):
            raise ValueError(
                "flat landing support-planar correction exceeds limit"
            )
        self.transform = FlatWorldTransform(
            yaw_offset,
            np.ascontiguousarray(support_translation),
        )
        self.last_result = result
        return self.transform.apply(internal)

    def step(
        self,
        *,
        velocity_world_xy: np.ndarray,
        heading_world_yaw: float,
    ) -> KinematicPose:
        transform = self.transform
        if transform is None:
            raise RuntimeError("flat source must be reset before step")
        internal_velocity = transform.velocity_to_internal(
            velocity_world_xy
        )
        internal_heading = transform.yaw_to_internal(heading_world_yaw)
        result = self.matcher.step(
            (float(internal_velocity[0]), float(internal_velocity[1])),
            internal_heading,
        )
        self.last_result = result
        return transform.apply(pose_from_motion_match(result))

    def prepare_step(
        self,
        *,
        velocity_world_xy: np.ndarray,
        heading_world_yaw: float,
    ) -> PreparedFlatKinematicStep:
        """Select a flat motion without advancing matcher state."""

        transform = self.transform
        prepare = getattr(self.matcher, "prepare_step", None)
        if transform is None:
            raise RuntimeError("flat source must be reset before step")
        if not callable(prepare):
            raise RuntimeError("flat matcher does not support transactions")
        internal_velocity = transform.velocity_to_internal(
            velocity_world_xy
        )
        internal_heading = transform.yaw_to_internal(heading_world_yaw)
        matcher_prepared = prepare(
            (float(internal_velocity[0]), float(internal_velocity[1])),
            internal_heading,
        )
        result = matcher_prepared.result
        pose = transform.apply(pose_from_motion_match(result))
        return PreparedFlatKinematicStep(
            matcher_prepared=matcher_prepared,
            result=result,
            pose=pose,
        )

    def prepare_row(
        self,
        row: int,
        *,
        velocity_world_xy: np.ndarray,
        heading_world_yaw: float,
    ) -> PreparedFlatKinematicStep:
        """Prepare one offline phase-planner-selected flat database row."""

        transform = self.transform
        prepare = getattr(self.matcher, "prepare_row", None)
        if transform is None:
            raise RuntimeError("flat source must be reset before step")
        if not callable(prepare):
            raise RuntimeError(
                "flat matcher does not support explicit phase preparation"
            )
        internal_velocity = transform.velocity_to_internal(
            velocity_world_xy
        )
        internal_heading = transform.yaw_to_internal(heading_world_yaw)
        matcher_prepared = prepare(
            int(row),
            (
                float(internal_velocity[0]),
                float(internal_velocity[1]),
            ),
            internal_heading,
        )
        result = matcher_prepared.result
        return PreparedFlatKinematicStep(
            matcher_prepared=matcher_prepared,
            result=result,
            pose=transform.apply(pose_from_motion_match(result)),
        )

    def select_supported_phase_rows(
        self,
        pose_world: KinematicPose,
        support_contact: np.ndarray,
    ) -> tuple[int, ...]:
        """Pair a terrain support pose with a contiguous flat lead-in."""

        if not isinstance(pose_world, KinematicPose):
            raise ValueError("pose_world must be a KinematicPose")
        contact = np.asarray(support_contact, dtype=bool)
        if contact.shape != (2,) or not np.any(contact):
            raise ValueError(
                "support_contact must contain at least one supported foot"
            )
        select = getattr(
            self.matcher, "select_supported_reset_row", None
        )
        if not callable(select):
            raise RuntimeError(
                "flat matcher does not support support-phase selection"
            )
        world_to_root = _rotation_z(-pose_world.root_yaw_world)
        feet = np.asarray((18, 19), dtype=np.int64)
        feet_root_local = (
            pose_world.body_position_world[feet]
            - pose_world.root_position_world
        ) @ world_to_root.T
        root_velocity_local_xy = (
            world_to_root @ pose_world.body_linear_velocity_world[0]
        )[:2]
        matched_row = int(
            select(
                joint_position=pose_world.joint_position,
                feet_position_root_local=feet_root_local,
                support_contact=contact,
                root_velocity_local_xy=root_velocity_local_xy,
            )
        )
        lead_rows_method = getattr(
            self.matcher, "source_lead_rows", None
        )
        if callable(lead_rows_method):
            rows = tuple(
                int(row)
                for row in lead_rows_method(
                    matched_row, lead_frames=15
                )
            )
            if not rows or rows[-1] != matched_row:
                raise RuntimeError(
                    "flat phase lead-in must end at its matched phase"
                )
            return rows
        lead_method = getattr(self.matcher, "source_lead_row", None)
        lead = (
            int(lead_method(matched_row, lead_frames=15))
            if callable(lead_method)
            else matched_row
        )
        return (lead,) if lead == matched_row else (lead, matched_row)

    def select_supported_phase_row(
        self,
        pose_world: KinematicPose,
        support_contact: np.ndarray,
    ) -> int:
        """Return the first row of the supported flat phase lead-in."""

        return self.select_supported_phase_rows(
            pose_world, support_contact
        )[0]

    def prepare_steps(
        self,
        *,
        velocity_world_xy: np.ndarray,
        heading_world_yaw: float,
        maximum_candidates: int,
    ):
        """Yield ranked flat candidates without advancing matcher state."""

        transform = self.transform
        prepare_many = getattr(self.matcher, "prepare_steps", None)
        if transform is None:
            raise RuntimeError("flat source must be reset before step")
        if not callable(prepare_many):
            yield self.prepare_step(
                velocity_world_xy=velocity_world_xy,
                heading_world_yaw=heading_world_yaw,
            )
            return
        internal_velocity = transform.velocity_to_internal(
            velocity_world_xy
        )
        internal_heading = transform.yaw_to_internal(heading_world_yaw)
        for matcher_prepared in prepare_many(
            (
                float(internal_velocity[0]),
                float(internal_velocity[1]),
            ),
            internal_heading,
            maximum_candidates=maximum_candidates,
        ):
            result = matcher_prepared.result
            yield PreparedFlatKinematicStep(
                matcher_prepared=matcher_prepared,
                result=result,
                pose=transform.apply(pose_from_motion_match(result)),
            )

    def commit_step(
        self, prepared: PreparedFlatKinematicStep
    ) -> KinematicPose:
        """Commit one previously validated flat motion selection."""

        transform = self.transform
        commit = getattr(self.matcher, "commit", None)
        if transform is None:
            raise RuntimeError("flat source must be reset before step")
        if not isinstance(prepared, PreparedFlatKinematicStep):
            raise ValueError("prepared flat step has an invalid type")
        if not callable(commit):
            raise RuntimeError("flat matcher does not support transactions")
        result = commit(prepared.matcher_prepared)
        self.last_result = result
        return transform.apply(pose_from_motion_match(result))

    def prepared_preview_world(
        self,
        prepared: PreparedFlatKinematicStep,
        *,
        frame_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialize a prepared dense root/body prefix in scene coordinates."""

        transform = self.transform
        count = int(frame_count)
        if transform is None:
            raise RuntimeError("flat source must be reset before preview")
        if (
            not isinstance(prepared, PreparedFlatKinematicStep)
            or type(frame_count) is not int
            or count <= 0
        ):
            raise ValueError("prepared preview request is invalid")
        internal_root = _tensor_numpy(
            prepared.result.dense_root_position_window
        )
        internal_body = _tensor_numpy(
            prepared.result.dense_body_position_window
        )
        if (
            internal_root.ndim != 2
            or internal_root.shape[1:] != (3,)
            or internal_body.shape != (len(internal_root), 30, 3)
            or count > len(internal_root)
        ):
            raise ValueError("prepared dense preview has an invalid shape")
        rotation = _rotation_z(transform.yaw_offset)
        roots = (
            internal_root[:count] @ rotation.T
            + transform.translation_world
        )
        bodies = (
            internal_body[:count] @ rotation.T
            + transform.translation_world
        )
        return (
            np.ascontiguousarray(roots, dtype=np.float32),
            np.ascontiguousarray(bodies, dtype=np.float32),
        )

    def future_feet_world(
        self, *, offsets_frames: tuple[int, ...]
    ) -> np.ndarray:
        """Return selected future left/right feet in the rebased scene."""

        transform = self.transform
        result = self.last_result
        offsets = tuple(int(value) for value in offsets_frames)
        if transform is None or result is None or not offsets:
            raise ValueError("flat selected future feet are unavailable")
        internal_body = _tensor_numpy(
            result.dense_body_position_window  # type: ignore[attr-defined]
        )
        if (
            internal_body.ndim != 3
            or internal_body.shape[1:] != (30, 3)
            or any(value < 0 or value >= len(internal_body) for value in offsets)
        ):
            raise ValueError("flat dense body window has an invalid shape")
        internal_feet = internal_body[list(offsets)][:, [18, 19]]
        rotation = _rotation_z(transform.yaw_offset)
        world_feet = (
            internal_feet @ rotation.T + transform.translation_world
        )
        return np.ascontiguousarray(world_feet, dtype=np.float32)


def flat_selected_preview_local(
    source: FlatKinematicSource,
    current_pose: KinematicPose,
    *,
    offsets_frames: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the flat matcher's dense selection in the current root frame."""

    transform = source.transform
    result = source.last_result
    offsets = tuple(int(value) for value in offsets_frames)
    if (
        transform is None
        or result is None
        or not offsets
        or any(value < 0 or value >= 46 for value in offsets)
    ):
        raise ValueError("flat selected preview is unavailable")
    internal_root = _tensor_numpy(
        result.dense_root_position_window  # type: ignore[attr-defined]
    )[list(offsets)]
    rotation = _rotation_z(transform.yaw_offset)
    world_root = (
        internal_root @ rotation.T + transform.translation_world
    )
    current_yaw = current_pose.root_yaw_world
    current_rotation_inverse = _rotation_z(-current_yaw)[:2, :2]
    local_root = (
        world_root[:, :2] - current_pose.root_position_world[:2]
    ) @ current_rotation_inverse.T

    internal_wxyz = _tensor_numpy(
        result.dense_root_orientation_window_wxyz  # type: ignore[attr-defined]
    )[list(offsets)]
    facing = np.empty((len(offsets), 2), dtype=np.float32)
    for index, quaternion_wxyz in enumerate(internal_wxyz):
        internal_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        relative_yaw = (
            _yaw_xyzw(internal_xyzw)
            + transform.yaw_offset
            - current_yaw
        )
        facing[index] = (
            math.cos(relative_yaw),
            math.sin(relative_yaw),
        )
    return (
        np.ascontiguousarray(local_root, dtype=np.float32),
        np.ascontiguousarray(facing),
    )


@dataclass(frozen=True)
class StairEntryMatch:
    row: int
    source_clip: int
    source_frame: int
    cost: float
    root_distance_m: float
    yaw_error_rad: float
    planted_foot_error_m: float
    placement: StairPlacement | None = None
    planar_warp_stair_xy: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class StairHandoffDiagnostics:
    candidate_rows: int
    root_compatible_rows: int
    yaw_compatible_rows: int
    endpoint_compatible_rows: int
    contact_compatible_rows: int
    minimum_root_distance_m: float | None
    minimum_yaw_error_rad: float | None
    minimum_endpoint_error_m: float | None
    minimum_contact_error_m: float | None


def _complete_route_ready_frames(database: object) -> dict[int, int]:
    """Return clips that demonstrably reach a post-transition support pose."""

    return complete_route_ready_frames(database)


class StairExitIndex:
    """Locate the first stable landing stance after each stair traversal."""

    def __init__(self, database: object) -> None:
        self._ready_frame = _complete_route_ready_frames(database)

    def is_ready(self, *, source_clip: int, source_frame: int) -> bool:
        threshold = self._ready_frame.get(int(source_clip))
        return threshold is not None and int(source_frame) >= threshold


class StairHandoffIndex:
    """Find pre-contact Justin approach frames compatible with flat motion."""

    _LOWER_BODY_JOINT_INDICES = np.asarray(
        (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18),
        dtype=np.int64,
    )

    def __init__(
        self,
        database: object,
        archive: object,
        *,
        approach_lookback_frames: int = 35,
        approach_window_frames: int = 60,
        maximum_root_distance_m: float = 0.13,
        maximum_yaw_error_rad: float = math.radians(10.0),
        maximum_planted_foot_error_m: float = 0.04,
        maximum_planar_warp_m: float = 0.025,
        maximum_planar_warp_residual_m: float = 0.010,
        maximum_flight_landing_error_m: float = 0.04,
        maximum_endpoint_error_m: float = 0.30,
        maximum_continuation_joint_step_rad: float | None = None,
        maximum_lower_body_joint_error_rad: float = 0.80,
    ) -> None:
        lookback = int(approach_lookback_frames)
        window = int(approach_window_frames)
        if lookback <= 0 or window <= 0:
            raise ValueError(
                "approach lookback and window frames must be positive"
            )
        self.database = database
        self.archive = archive
        self.maximum_root_distance_m = float(maximum_root_distance_m)
        self.maximum_yaw_error_rad = float(maximum_yaw_error_rad)
        self.maximum_planted_foot_error_m = float(
            maximum_planted_foot_error_m
        )
        self.maximum_planar_warp_m = float(maximum_planar_warp_m)
        self.maximum_planar_warp_residual_m = float(
            maximum_planar_warp_residual_m
        )
        self.maximum_flight_landing_error_m = float(
            maximum_flight_landing_error_m
        )
        self.maximum_endpoint_error_m = float(maximum_endpoint_error_m)
        self.maximum_continuation_joint_step_rad = (
            None
            if maximum_continuation_joint_step_rad is None
            else float(maximum_continuation_joint_step_rad)
        )
        self.maximum_lower_body_joint_error_rad = float(
            maximum_lower_body_joint_error_rad
        )
        for name in (
            "maximum_root_distance_m",
            "maximum_yaw_error_rad",
            "maximum_planted_foot_error_m",
            "maximum_planar_warp_m",
            "maximum_planar_warp_residual_m",
            "maximum_flight_landing_error_m",
            "maximum_endpoint_error_m",
            "maximum_lower_body_joint_error_rad",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.maximum_continuation_joint_step_rad is not None
            and (
                not math.isfinite(
                    self.maximum_continuation_joint_step_rad
                )
                or self.maximum_continuation_joint_step_rad <= 0.0
            )
        ):
            raise ValueError(
                "maximum_continuation_joint_step_rad must be finite and "
                "positive when provided"
            )

        source_clip = np.asarray(database.source_clip)
        source_frame = np.asarray(database.source_frame)
        entry = np.asarray(database.entry, dtype=bool)
        # A partial capture can be valuable as a mid-stair search fragment,
        # but it must never be the initial handoff route: reaching its final
        # frame would strand the robot on the obstacle.  Older synthetic
        # callers predate lifecycle annotations, so preserve their behavior
        # when ``exit`` is absent.
        complete_route_clips = (
            set(_complete_route_ready_frames(database))
            if hasattr(database, "exit")
            else None
        )
        continuation_safe: np.ndarray | None = None
        feature = np.asarray(getattr(database, "feature", np.empty((0, 0))))
        if (
            hasattr(database, "exit")
            and self.maximum_continuation_joint_step_rad is not None
            and feature.ndim == 2
            and feature.shape[0] == len(source_clip)
            and feature.shape[1] >= 29
        ):
            exit_flag = np.asarray(database.exit, dtype=bool)
            continuation_safe = np.zeros(len(source_clip), dtype=bool)
            for clip in sorted(set(int(value) for value in source_clip)):
                clip_rows = np.flatnonzero(source_clip == clip)
                clip_rows = clip_rows[
                    np.argsort(source_frame[clip_rows], kind="stable")
                ]
                next_row: int | None = None
                for row_value in clip_rows[::-1]:
                    row = int(row_value)
                    if bool(exit_flag[row]):
                        safe = True
                    elif next_row is None:
                        safe = False
                    else:
                        sequential = (
                            int(source_frame[next_row])
                            == int(source_frame[row]) + 1
                        )
                        joint_step = float(
                            np.max(
                                np.abs(
                                    feature[next_row, :29]
                                    - feature[row, :29]
                                ),
                                initial=0.0,
                            )
                        )
                        safe = bool(
                            continuation_safe[next_row]
                            and sequential
                            and joint_step
                            <= self.maximum_continuation_joint_step_rad
                        )
                    continuation_safe[row] = safe
                    next_row = row
        candidates: list[int] = []
        for clip in sorted(set(int(value) for value in source_clip)):
            if (
                complete_route_clips is not None
                and clip not in complete_route_clips
            ):
                continue
            rows = np.flatnonzero(source_clip == clip)
            entry_rows = rows[entry[rows]]
            if continuation_safe is not None:
                entry_rows = entry_rows[continuation_safe[entry_rows]]
            if entry_rows.size == 0:
                continue
            first_entry_frame = int(np.min(source_frame[entry_rows]))
            if first_entry_frame < lookback:
                # The catalog's reviewed pre-contact window can legitimately
                # reach frame zero for a short clean approach.  Such clips
                # used to contribute no candidates at all because this index
                # subtracted a second lookback from an already-looked-back
                # annotation.  Lifecycle filtering above has already removed
                # continuous/mid-route captures, so retain the reviewed entry
                # rows for complete routes.
                candidates.extend(int(row) for row in entry_rows)
                continue
            last = first_entry_frame - lookback
            first = max(0, last - window)
            prior = rows[
                (source_frame[rows] >= first)
                & (source_frame[rows] <= last)
            ]
            if continuation_safe is not None:
                prior = prior[continuation_safe[prior]]
            candidates.extend(int(row) for row in prior)
        if not candidates:
            raise ValueError("terrain database has no pre-contact approaches")
        self.rows = tuple(candidates)
        self.last_diagnostics = StairHandoffDiagnostics(
            candidate_rows=len(self.rows),
            root_compatible_rows=0,
            yaw_compatible_rows=0,
            endpoint_compatible_rows=0,
            contact_compatible_rows=0,
            minimum_root_distance_m=None,
            minimum_yaw_error_rad=None,
            minimum_endpoint_error_m=None,
            minimum_contact_error_m=None,
        )

    def _lower_body_joint_error(
        self, pose: KinematicPose, row: int
    ) -> float:
        feature = np.asarray(getattr(self.database, "feature", ()))
        if (
            feature.ndim != 2
            or feature.shape[0] <= int(row)
            or feature.shape[1] < 29
        ):
            return 0.0
        indices = self._LOWER_BODY_JOINT_INDICES
        return float(
            np.max(
                np.abs(
                    pose.joint_position[indices]
                    - feature[int(row), indices]
                ),
                initial=0.0,
            )
        )

    @staticmethod
    def _root_velocity_local(pose: KinematicPose) -> np.ndarray:
        velocity = pose.body_linear_velocity_world[0, :2].astype(
            np.float64
        )
        c, s = math.cos(pose.root_yaw_world), math.sin(pose.root_yaw_world)
        return np.asarray(
            (c * velocity[0] + s * velocity[1],
             -s * velocity[0] + c * velocity[1]),
            dtype=np.float32,
        )

    def approach_correction_robot(
        self,
        pose: KinematicPose,
        future_root_xy: np.ndarray,
        *,
        stair: ObservedStair,
        maximum_correction_mps: float = 0.12,
        maximum_alignment_distance_m: float = 0.12,
        support_contact: np.ndarray | None = None,
    ) -> np.ndarray:
        """Steer a stair approach toward compatible supported feet."""

        roots = np.asarray(future_root_xy, dtype=np.float32)
        database_roots = np.asarray(self.database.future_root_xy)
        maximum = float(maximum_correction_mps)
        maximum_distance = float(maximum_alignment_distance_m)
        query_support: np.ndarray | None = None
        if support_contact is not None:
            candidate_support = np.asarray(support_contact)
            if (
                candidate_support.shape != (2,)
                or candidate_support.dtype.kind != "b"
            ):
                raise ValueError(
                    "support_contact must contain two booleans"
                )
            query_support = np.ascontiguousarray(
                candidate_support, dtype=bool
            )
        if (
            roots.shape != database_roots.shape[1:]
            or not np.isfinite(roots).all()
            or not math.isfinite(maximum)
            or maximum <= 0.0
            or not math.isfinite(maximum_distance)
            or maximum_distance <= 0.0
        ):
            raise ValueError("downstair correction inputs are invalid")
        # Once stair intent is latched, apply the small support-foot
        # alignment while ordinary walking is still able to steer.  Waiting
        # until speed fell below 0.18 m/s meant the terrain-edge safety gate
        # had already stopped translation, so oblique approaches could never
        # remove their final 10--13 cm splice error.  Only suppress this
        # correction for genuinely fast motion where a lateral nudge would be
        # visually and mechanically unsafe.
        if np.linalg.norm(self._root_velocity_local(pose)) > 0.60:
            return np.zeros(2, dtype=np.float64)
        placement = stair.placement_from_robot(
            root_position_world=pose.root_position_world,
            root_yaw_world=pose.root_yaw_world,
        )
        query_feet = placement.world_to_stair_xyz(
            pose.body_position_world[[18, 19]]
        )
        best_delta_stair: np.ndarray | None = None
        best_cost = math.inf
        for row in self.rows:
            if (
                self._lower_body_joint_error(pose, row)
                > self.maximum_lower_body_joint_error_rad
            ):
                continue
            contact = np.asarray(self.database.contact[row], dtype=bool)
            if (
                not np.any(contact)
                or (
                    query_support is not None
                    and not np.array_equal(contact, query_support)
                )
            ):
                continue
            root_distance = float(
                np.linalg.norm(
                    np.asarray(
                        self.database.stair_origin_root_xy[row],
                        dtype=np.float32,
                    )
                    - stair.geometry.origin_robot_xy
                )
            )
            yaw_error = abs(
                math.remainder(
                    float(self.database.stair_ascent_yaw_root[row])
                    - stair.geometry.ascent_yaw_robot,
                    2.0 * math.pi,
                )
            )
            endpoint_error = float(
                np.linalg.norm(
                    roots[-1]
                    - np.asarray(
                        self.database.future_root_xy[row],
                        dtype=np.float32,
                    )[-1]
                )
            )
            if (
                root_distance > self.maximum_root_distance_m
                or yaw_error > self.maximum_yaw_error_rad
                or endpoint_error > self.maximum_endpoint_error_m
            ):
                continue
            source_feet = np.asarray(
                self.database.feet_xyz_stair[row], dtype=np.float32
            )
            delta = np.mean(
                source_feet[contact, :2] - query_feet[contact, :2],
                axis=0,
            )
            individual_delta = (
                source_feet[contact, :2] - query_feet[contact, :2]
            )
            if (
                float(
                    np.max(
                        np.linalg.norm(individual_delta, axis=1),
                        initial=0.0,
                    )
                )
                > maximum_distance
            ):
                continue
            residual = (
                query_feet[contact, :2]
                + delta
                - source_feet[contact, :2]
            )
            cost = float(
                np.mean(residual * residual)
                + 0.1 * np.dot(delta, delta)
                + root_distance * root_distance
                + yaw_error * yaw_error
                + endpoint_error * endpoint_error
            )
            if cost < best_cost:
                best_cost = cost
                best_delta_stair = delta
        if best_delta_stair is None:
            return np.zeros(2, dtype=np.float64)
        rotation_robot_from_stair = _rotation_z(
            stair.geometry.ascent_yaw_robot
        )[:2, :2]
        correction = rotation_robot_from_stair @ best_delta_stair
        magnitude = float(np.linalg.norm(correction))
        if magnitude > maximum:
            correction *= maximum / magnitude
        return np.ascontiguousarray(correction, dtype=np.float64)

    def plan_route(
        self,
        pose: KinematicPose,
        future_root_xy: np.ndarray,
        future_facing_xy: np.ndarray,
        *,
        stair: ObservedStair,
        maximum_root_distance_m: float = 0.35,
        maximum_yaw_error_rad: float = math.radians(25.0),
    ) -> StairEntryMatch | None:
        """Choose a spatial stair snippet before the outgoing gait phase fits.

        This is deliberately not permission to splice.  It ignores current
        joints, contacts, and planted-foot agreement so the flat matcher can
        be guided toward the clean phase paired with a viable route.  The
        ordinary :meth:`select` hard gates remain the only way to enter.
        """

        if not isinstance(pose, KinematicPose):
            raise ValueError("pose must be a KinematicPose")
        if not isinstance(stair, ObservedStair):
            raise ValueError("stair must be an ObservedStair")
        roots = np.asarray(future_root_xy, dtype=np.float32)
        facings = np.asarray(future_facing_xy, dtype=np.float32)
        database_roots = np.asarray(self.database.future_root_xy)
        root_limit = float(maximum_root_distance_m)
        yaw_limit = float(maximum_yaw_error_rad)
        if (
            roots.shape != database_roots.shape[1:]
            or facings.shape != roots.shape
            or not np.isfinite(roots).all()
            or not np.isfinite(facings).all()
            or not math.isfinite(root_limit)
            or root_limit <= 0.0
            or not math.isfinite(yaw_limit)
            or yaw_limit <= 0.0
        ):
            raise ValueError("route-planning inputs are invalid")
        placement = stair.placement_from_robot(
            root_position_world=pose.root_position_world,
            root_yaw_world=pose.root_yaw_world,
        )
        best: StairEntryMatch | None = None
        for row in self.rows:
            root_distance = float(
                np.linalg.norm(
                    np.asarray(
                        self.database.stair_origin_root_xy[row],
                        dtype=np.float32,
                    )
                    - stair.geometry.origin_robot_xy
                )
            )
            yaw_error = abs(
                math.remainder(
                    float(self.database.stair_ascent_yaw_root[row])
                    - stair.geometry.ascent_yaw_robot,
                    2.0 * math.pi,
                )
            )
            if root_distance > root_limit or yaw_error > yaw_limit:
                continue
            candidate_root = np.asarray(
                self.database.future_root_xy[row], dtype=np.float32
            )
            candidate_facing = np.asarray(
                self.database.future_facing_xy[row], dtype=np.float32
            )
            endpoint_error = float(
                np.linalg.norm(roots[-1] - candidate_root[-1])
            )
            if endpoint_error > self.maximum_endpoint_error_m:
                continue
            trajectory_cost = float(
                np.mean((roots - candidate_root) ** 2)
                + np.mean((facings - candidate_facing) ** 2)
            )
            cost = (
                25.0 * root_distance * root_distance
                + 2.0 * yaw_error * yaw_error
                + 8.0 * trajectory_cost
            )
            match = StairEntryMatch(
                row=int(row),
                source_clip=int(self.database.source_clip[row]),
                source_frame=int(self.database.source_frame[row]),
                cost=cost,
                root_distance_m=root_distance,
                yaw_error_rad=yaw_error,
                planted_foot_error_m=0.0,
                placement=placement,
            )
            if best is None or match.cost < best.cost:
                best = match
        return best

    def approach_correction_to_row(
        self,
        *,
        stair: ObservedStair,
        row: int,
        maximum_correction_mps: float = 0.16,
    ) -> np.ndarray:
        """Move the root toward one locked snippet's recorded entry frame."""

        index = int(row)
        maximum = float(maximum_correction_mps)
        if (
            not isinstance(stair, ObservedStair)
            or not 0 <= index < len(self.database.source_clip)
            or not math.isfinite(maximum)
            or maximum <= 0.0
        ):
            raise ValueError("planned approach correction inputs are invalid")
        correction = (
            np.asarray(
                stair.geometry.origin_robot_xy, dtype=np.float64
            )
            - np.asarray(
                self.database.stair_origin_root_xy[index],
                dtype=np.float64,
            )
        )
        magnitude = float(np.linalg.norm(correction))
        if magnitude > maximum:
            correction *= maximum / magnitude
        return np.ascontiguousarray(correction)

    def select(
        self,
        pose: KinematicPose,
        future_root_xy: np.ndarray,
        future_facing_xy: np.ndarray,
        *,
        stair: ObservedStair | None = None,
        future_feet_world: np.ndarray | None = None,
        support_contact: np.ndarray | None = None,
        maximum_planted_foot_error_m_override: float | None = None,
    ) -> StairEntryMatch | None:
        roots = np.asarray(future_root_xy, dtype=np.float32)
        facings = np.asarray(future_facing_xy, dtype=np.float32)
        database_roots = np.asarray(self.database.future_root_xy)
        if (
            roots.shape != database_roots.shape[1:]
            or facings.shape != roots.shape
            or not np.isfinite(roots).all()
            or not np.isfinite(facings).all()
        ):
            raise ValueError(
                "future command arrays must match the terrain horizon"
            )
        query_future_feet_world: np.ndarray | None = None
        if future_feet_world is not None:
            query_future_feet_world = _finite(
                future_feet_world,
                (roots.shape[0], 2, 3),
                "future_feet_world",
            )
        query_support: np.ndarray | None = None
        if support_contact is not None:
            candidate_support = np.asarray(support_contact)
            if (
                candidate_support.shape != (2,)
                or candidate_support.dtype.kind != "b"
            ):
                raise ValueError(
                    "support_contact must contain two booleans"
                )
            query_support = np.ascontiguousarray(
                candidate_support, dtype=bool
            )
        planted_foot_limit = self.maximum_planted_foot_error_m
        if maximum_planted_foot_error_m_override is not None:
            override = float(maximum_planted_foot_error_m_override)
            if (
                not math.isfinite(override)
                or override < self.maximum_planted_foot_error_m
            ):
                raise ValueError(
                    "planted-foot override must be finite and no stricter "
                    "than the configured handoff limit"
                )
            planted_foot_limit = override
        feature = np.concatenate(
            (
                pose.joint_position,
                pose.joint_velocity,
                self._root_velocity_local(pose),
            )
        )
        scale = np.asarray(self.database.feature_scale, dtype=np.float32)
        if feature.shape != scale.shape:
            raise ValueError(
                "flat and terrain pose feature layouts do not agree"
            )
        best: StairEntryMatch | None = None
        placement = (
            None
            if stair is None
            else stair.placement_from_robot(
                root_position_world=pose.root_position_world,
                root_yaw_world=pose.root_yaw_world,
            )
        )
        feet_stair = (
            None
            if placement is None
            else placement.world_to_stair_xyz(
                pose.body_position_world[[18, 19]]
            )
        )
        future_feet_stair = (
            None
            if placement is None or query_future_feet_world is None
            else placement.world_to_stair_xyz(query_future_feet_world)
        )
        root_compatible_rows = 0
        yaw_compatible_rows = 0
        endpoint_compatible_rows = 0
        contact_compatible_rows = 0
        minimum_root_distance = math.inf
        minimum_yaw_error = math.inf
        minimum_endpoint_error = math.inf
        minimum_contact_error = math.inf
        for row in self.rows:
            if (
                self._lower_body_joint_error(pose, row)
                > self.maximum_lower_body_joint_error_rad
            ):
                continue
            clip = int(self.database.source_clip[row])
            frame = int(self.database.source_frame[row])
            if stair is None:
                position, _quaternion, _joints, yaw = (
                    self.archive.root_pose(clip, frame)
                )
                root_distance = float(
                    np.linalg.norm(
                        pose.root_position_world[:2]
                        - np.asarray(position, dtype=np.float32)[:2]
                    )
                )
                yaw_error = abs(
                    math.remainder(
                        pose.root_yaw_world - float(yaw),
                        2.0 * math.pi,
                    )
                )
            else:
                root_distance = float(
                    np.linalg.norm(
                        np.asarray(
                            self.database.stair_origin_root_xy[row],
                            dtype=np.float32,
                        )
                        - stair.geometry.origin_robot_xy
                    )
                )
                yaw_error = abs(
                    math.remainder(
                        float(
                            self.database.stair_ascent_yaw_root[row]
                        )
                        - stair.geometry.ascent_yaw_robot,
                        2.0 * math.pi,
                    )
                )
            endpoint_error = float(
                np.linalg.norm(
                    roots[-1]
                    - np.asarray(self.database.future_root_xy[row])[-1]
                )
            )
            minimum_root_distance = min(
                minimum_root_distance, root_distance
            )
            minimum_yaw_error = min(minimum_yaw_error, yaw_error)
            minimum_endpoint_error = min(
                minimum_endpoint_error, endpoint_error
            )
            if root_distance > self.maximum_root_distance_m:
                continue
            root_compatible_rows += 1
            if yaw_error > self.maximum_yaw_error_rad:
                continue
            yaw_compatible_rows += 1
            if endpoint_error > self.maximum_endpoint_error_m:
                continue
            endpoint_compatible_rows += 1
            contact = np.asarray(
                self.database.contact[row], dtype=bool
            )
            if (
                query_support is not None
                and (
                    (
                        np.any(contact)
                        and np.any(contact & ~query_support)
                    )
                    or (
                        not np.any(contact)
                        and np.any(query_support)
                    )
                )
            ):
                continue
            planted_error = 0.0
            planar_warp = np.zeros(2, dtype=np.float32)
            if np.any(contact):
                if feet_stair is None:
                    index = self.archive.global_index(clip, frame)
                    source_feet = np.asarray(
                        self.archive.body_pos_w[index, [18, 19]],
                        dtype=np.float32,
                    )
                    query_feet = pose.body_position_world[[18, 19]]
                else:
                    source_feet = np.asarray(
                        self.database.feet_xyz_stair[row],
                        dtype=np.float32,
                    )
                    query_feet = feet_stair
                planted_delta = (
                    query_feet[contact, :2]
                    - source_feet[contact, :2]
                )
                planted_error = float(
                    np.max(
                        np.linalg.norm(
                            planted_delta,
                            axis=1,
                        ),
                        initial=0.0,
                    )
                )
                minimum_contact_error = min(
                    minimum_contact_error, planted_error
                )
                planar_warp = np.mean(planted_delta, axis=0)
                warp_residual = float(
                    np.max(
                        np.linalg.norm(
                            planted_delta - planar_warp,
                            axis=1,
                        ),
                        initial=0.0,
                    )
                )
                if (
                    planted_error > planted_foot_limit
                    or float(np.linalg.norm(planar_warp))
                    > self.maximum_planar_warp_m
                    or warp_residual
                    > self.maximum_planar_warp_residual_m
                ):
                    continue
            else:
                # Root/yaw agreement cannot make an airborne splice safe.
                # The selected flat motion must predict the same supported
                # future landing feet as the stair clip.
                if future_feet_stair is None:
                    continue
                landing_contact = np.asarray(
                    self.database.future_contact[row], dtype=bool
                )
                source_future_feet = np.asarray(
                    self.database.future_feet_xyz_stair[row],
                    dtype=np.float32,
                )
                if (
                    landing_contact.shape != (roots.shape[0], 2)
                    or source_future_feet.shape
                    != (roots.shape[0], 2, 3)
                    or not np.any(landing_contact)
                ):
                    continue
                planted_error = float(
                    np.max(
                        np.linalg.norm(
                            future_feet_stair[landing_contact]
                            - source_future_feet[landing_contact],
                            axis=1,
                        ),
                        initial=0.0,
                    )
                )
                minimum_contact_error = min(
                    minimum_contact_error, planted_error
                )
                if planted_error > self.maximum_flight_landing_error_m:
                    continue
            contact_compatible_rows += 1
            feature_cost = float(
                np.mean(
                    (
                        (
                            feature
                            - np.asarray(
                                self.database.feature[row],
                                dtype=np.float32,
                            )
                        )
                        / scale
                    )
                    ** 2
                )
            )
            trajectory_cost = float(
                np.mean(
                    (
                        roots
                        - np.asarray(
                            self.database.future_root_xy[row],
                            dtype=np.float32,
                        )
                    )
                    ** 2
                )
                + np.mean(
                    (
                        facings
                        - np.asarray(
                            self.database.future_facing_xy[row],
                            dtype=np.float32,
                        )
                    )
                    ** 2
                )
            )
            cost = (
                feature_cost
                + 25.0 * root_distance * root_distance
                + 2.0 * yaw_error * yaw_error
                + 20.0 * planted_error * planted_error
                + 8.0 * trajectory_cost
            )
            match = StairEntryMatch(
                row=row,
                source_clip=clip,
                source_frame=frame,
                cost=cost,
                root_distance_m=root_distance,
                yaw_error_rad=yaw_error,
                planted_foot_error_m=planted_error,
                placement=placement,
                planar_warp_stair_xy=(
                    float(planar_warp[0]),
                    float(planar_warp[1]),
                ),
            )
            if best is None or match.cost < best.cost:
                best = match
        self.last_diagnostics = StairHandoffDiagnostics(
            candidate_rows=len(self.rows),
            root_compatible_rows=root_compatible_rows,
            yaw_compatible_rows=yaw_compatible_rows,
            endpoint_compatible_rows=endpoint_compatible_rows,
            contact_compatible_rows=contact_compatible_rows,
            minimum_root_distance_m=(
                None
                if not math.isfinite(minimum_root_distance)
                else minimum_root_distance
            ),
            minimum_yaw_error_rad=(
                None
                if not math.isfinite(minimum_yaw_error)
                else minimum_yaw_error
            ),
            minimum_endpoint_error_m=(
                None
                if not math.isfinite(minimum_endpoint_error)
                else minimum_endpoint_error
            ),
            minimum_contact_error_m=(
                None
                if not math.isfinite(minimum_contact_error)
                else minimum_contact_error
            ),
        )
        return best


def pose_from_stair_row(
    database: object,
    archive: object,
    row: int,
    *,
    placement: StairPlacement | None = None,
) -> KinematicPose:
    """Materialize one Justin row on its observed canonical stair placement."""

    index_value = int(row)
    source_clip = int(database.source_clip[index_value])
    source_frame = int(database.source_frame[index_value])
    index = int(archive.global_index(source_clip, source_frame))
    position, quaternion, joints, source_root_yaw = archive.root_pose(
        source_clip, source_frame
    )
    body_position = np.asarray(
        archive.body_pos_w[index], dtype=np.float32
    )
    body_orientation = np.asarray(
        archive.body_quat_w[index], dtype=np.float32
    )
    archived_body_velocity = getattr(
        archive, "body_lin_vel_w", None
    )
    if archived_body_velocity is not None:
        body_velocity = np.asarray(
            archived_body_velocity[index], dtype=np.float32
        )
    else:
        rate = float(archive.fps)
        body_velocity = np.zeros((30, 3), dtype=np.float32)
        for neighbor_frame, sign in (
            (source_frame + 1, 1.0),
            (source_frame - 1, -1.0),
        ):
            try:
                neighbor = int(
                    archive.global_index(source_clip, neighbor_frame)
                )
            except (IndexError, KeyError, ValueError):
                continue
            body_velocity = (
                np.asarray(
                    archive.body_pos_w[neighbor], dtype=np.float32
                )
                - body_position
            ) * (rate * sign)
            break
    archived_body_angular_velocity = getattr(
        archive, "body_ang_vel_w", None
    )
    body_angular_velocity = (
        np.zeros((30, 3), dtype=np.float32)
        if archived_body_angular_velocity is None
        else np.asarray(
            archived_body_angular_velocity[index], dtype=np.float32
        )
    )
    feature = np.asarray(database.feature[index_value], dtype=np.float32)
    if feature.shape[0] < 58:
        raise ValueError("terrain feature must contain joint pose and velocity")
    pose = KinematicPose(
        root_position_world=np.asarray(position, dtype=np.float32),
        root_orientation_world_xyzw=np.asarray(
            quaternion, dtype=np.float32
        ),
        joint_position=np.asarray(joints, dtype=np.float32),
        joint_velocity=feature[29:58],
        body_position_world=body_position,
        body_orientation_world_xyzw=body_orientation,
        body_linear_velocity_world=body_velocity,
        body_angular_velocity_world=body_angular_velocity,
    )
    if placement is None:
        return pose

    has_catalogued_frame = all(
        hasattr(database, name)
        for name in (
            "stair_origin_root_xy",
            "stair_ascent_yaw_root",
            "root_height_above_stair_base_m",
        )
    )
    if has_catalogued_frame:
        source_origin_root = np.asarray(
            database.stair_origin_root_xy[index_value],
            dtype=np.float64,
        )
        root_rotation = _rotation_z(source_root_yaw)[:2, :2]
        source_origin_xy = (
            np.asarray(position[:2], dtype=np.float64)
            + root_rotation @ source_origin_root
        )
        source_origin = np.asarray(
            (
                source_origin_xy[0],
                source_origin_xy[1],
                float(position[2])
                - float(
                    database.root_height_above_stair_base_m[index_value]
                ),
            ),
            dtype=np.float64,
        )
        source_ascent_yaw = (
            source_root_yaw
            + float(database.stair_ascent_yaw_root[index_value])
        )
    else:
        # Compatibility for small synthetic callers predating the per-row
        # source stair frame. Real catalogs always take the branch above.
        source_origin = np.asarray(
            (
                JUSTIN_STAIR_ORIGIN_WORLD_XY[0],
                JUSTIN_STAIR_ORIGIN_WORLD_XY[1],
                0.0,
            ),
            dtype=np.float64,
        )
        source_ascent_yaw = JUSTIN_STAIR_ASCENT_YAW_WORLD
    source_to_instance_yaw = math.remainder(
        placement.ascent_yaw_world - source_ascent_yaw,
        2.0 * math.pi,
    )
    rotation = _rotation_z(source_to_instance_yaw)

    def transform_points(points: np.ndarray) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64)
        return np.ascontiguousarray(
            (value - source_origin) @ rotation.T
            + placement.origin_world_xyz,
            dtype=np.float32,
        )

    yaw_quaternion = _yaw_quaternion_xyzw(source_to_instance_yaw)
    return KinematicPose(
        root_position_world=transform_points(pose.root_position_world),
        root_orientation_world_xyzw=_quat_multiply_xyzw(
            yaw_quaternion, pose.root_orientation_world_xyzw
        ),
        joint_position=pose.joint_position,
        joint_velocity=pose.joint_velocity,
        body_position_world=transform_points(pose.body_position_world),
        body_orientation_world_xyzw=_quat_multiply_xyzw(
            np.broadcast_to(yaw_quaternion, (30, 4)),
            pose.body_orientation_world_xyzw,
        ),
        body_linear_velocity_world=(
            pose.body_linear_velocity_world @ rotation.T
        ),
        body_angular_velocity_world=(
            pose.body_angular_velocity_world @ rotation.T
        ),
    )


def _translate_pose_world_xy(
    pose: KinematicPose, translation_world_xy: np.ndarray
) -> KinematicPose:
    translation_xy = _finite(
        translation_world_xy, (2,), "translation_world_xy"
    )
    translation = np.asarray(
        (translation_xy[0], translation_xy[1], 0.0),
        dtype=np.float32,
    )
    return KinematicPose(
        root_position_world=pose.root_position_world + translation,
        root_orientation_world_xyzw=pose.root_orientation_world_xyzw,
        joint_position=pose.joint_position,
        joint_velocity=pose.joint_velocity,
        body_position_world=pose.body_position_world + translation,
        body_orientation_world_xyzw=pose.body_orientation_world_xyzw,
        body_linear_velocity_world=pose.body_linear_velocity_world,
        body_angular_velocity_world=pose.body_angular_velocity_world,
    )


class PoseInertializer:
    """Decay one pose discontinuity while subsequent source frames advance."""

    def __init__(
        self,
        prior: KinematicPose,
        target_at_switch: KinematicPose,
        *,
        halflife_s: float = 0.10,
    ) -> None:
        half = float(halflife_s)
        if not math.isfinite(half) or half <= 0.0:
            raise ValueError("halflife_s must be finite and positive")
        self._halflife_s = half
        self._root_position_offset = (
            prior.root_position_world - target_at_switch.root_position_world
        )
        self._joint_position_offset = (
            prior.joint_position - target_at_switch.joint_position
        )
        self._joint_velocity_offset = (
            prior.joint_velocity - target_at_switch.joint_velocity
        )
        self._body_position_offset = (
            prior.body_position_world - target_at_switch.body_position_world
        )
        self._body_velocity_offset = (
            prior.body_linear_velocity_world
            - target_at_switch.body_linear_velocity_world
        )
        self._body_angular_velocity_offset = (
            prior.body_angular_velocity_world
            - target_at_switch.body_angular_velocity_world
        )
        self._root_orientation_offset = _quat_multiply_xyzw(
            prior.root_orientation_world_xyzw,
            _quat_inverse_xyzw(
                target_at_switch.root_orientation_world_xyzw
            ),
        )
        self._body_orientation_offset = _quat_multiply_xyzw(
            prior.body_orientation_world_xyzw,
            _quat_inverse_xyzw(
                target_at_switch.body_orientation_world_xyzw
            ),
        )

    def _fraction(self, elapsed_s: float) -> float:
        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        y = (4.0 * math.log(2.0) / (self._halflife_s + 1.0e-5)) / 2.0
        return math.exp(-y * elapsed) * (1.0 + y * elapsed)

    def apply(
        self, target: KinematicPose, *, elapsed_s: float
    ) -> KinematicPose:
        fraction = self._fraction(elapsed_s)
        root_offset = _scaled_quaternion_offset(
            self._root_orientation_offset, fraction
        )
        body_offset = _scaled_quaternion_offset(
            self._body_orientation_offset, fraction
        )
        return KinematicPose(
            root_position_world=(
                target.root_position_world
                + self._root_position_offset * fraction
            ),
            root_orientation_world_xyzw=_quat_multiply_xyzw(
                root_offset, target.root_orientation_world_xyzw
            ),
            joint_position=(
                target.joint_position
                + self._joint_position_offset * fraction
            ),
            joint_velocity=(
                target.joint_velocity
                + self._joint_velocity_offset * fraction
            ),
            body_position_world=(
                target.body_position_world
                + self._body_position_offset * fraction
            ),
            body_orientation_world_xyzw=_quat_multiply_xyzw(
                body_offset, target.body_orientation_world_xyzw
            ),
            body_linear_velocity_world=(
                target.body_linear_velocity_world
                + self._body_velocity_offset * fraction
            ),
            body_angular_velocity_world=(
                target.body_angular_velocity_world
                + self._body_angular_velocity_offset * fraction
            ),
        )


class HybridMode(str, Enum):
    FLAT = "flat"
    STAIR_ARMED = "stair_armed"
    SAFE_STOP = "safe_stop"
    STAIR = "stair"
    LANDING = "landing"


def flat_preview_uses_nominal_sole_proxy(
    mode: HybridMode,
    *,
    exact_pose_repair: bool,
    post_landing_recovery: bool = False,
) -> bool:
    """Keep the proxy except during exactly gated landing transitions."""

    if not isinstance(mode, HybridMode):
        raise ValueError("mode must be a HybridMode")
    if (
        type(exact_pose_repair) is not bool
        or type(post_landing_recovery) is not bool
    ):
        raise ValueError("preview gate flags must be bool")
    return not (
        (mode is HybridMode.LANDING or post_landing_recovery)
        and exact_pose_repair
    )


@dataclass(frozen=True)
class HybridStepOutcome:
    mode: HybridMode
    pose: KinematicPose
    tick: int
    entered_stairs: bool
    exited_stairs: bool
    source_clip: int
    source_frame: int
    cost: float
    candidate_count: int
    maximum_planted_foot_mismatch_m: float
    reason: str
    terrain_outcome: object | None = None


class HybridKinematicSession:
    """Alternate between free flat MM and robot-observed stair MM."""

    def __init__(
        self,
        database: object,
        archive: object,
        flat_source: FlatKinematicSource,
        *,
        initial_root_position_world: np.ndarray,
        initial_root_yaw_world: float,
        initial_support_pose_world: KinematicPose | None = None,
        initial_support_contact: np.ndarray | None = None,
        entry_index: object | None = None,
        exit_index: object | None = None,
        terrain_session_factory: object | None = None,
        intent_latch: StairIntentLatch | None = None,
        stair_observation_filter: CausalStairObservationFilter | None = None,
        transition_halflife_s: float = 0.10,
        maximum_unblended_joint_step_rad: float = 0.25,
        entry_planar_warp_release_frames: int = 20,
        reentry_cooldown_frames: int = 75,
        landing_confirmation_frames: int = 3,
        allow_stalled_handoff_relaxation: bool = False,
        stalled_handoff_relaxation_frames: int = 8,
        maximum_stalled_handoff_foot_error_m: float = 0.18,
        fps: float = 50.0,
    ) -> None:
        if terrain_session_factory is None:
            from .terrain_interactive import KinematicTerrainSession

            terrain_session_factory = KinematicTerrainSession
        self.database = database
        self.archive = archive
        self.flat_source = flat_source
        self.entry_index = (
            StairHandoffIndex(database, archive)
            if entry_index is None
            else entry_index
        )
        self.exit_index = (
            StairExitIndex(database) if exit_index is None else exit_index
        )
        self._terrain_session_factory = terrain_session_factory
        self.intent_latch = (
            StairIntentLatch() if intent_latch is None else intent_latch
        )
        if not isinstance(self.intent_latch, StairIntentLatch):
            raise ValueError("intent_latch must be StairIntentLatch")
        self.stair_observation_filter = (
            CausalStairObservationFilter()
            if stair_observation_filter is None
            else stair_observation_filter
        )
        if not isinstance(
            self.stair_observation_filter,
            CausalStairObservationFilter,
        ):
            raise ValueError(
                "stair_observation_filter must be a "
                "CausalStairObservationFilter"
            )
        self._initial_root_position = _finite(
            initial_root_position_world,
            (3,),
            "initial_root_position_world",
        )
        self._initial_root_yaw = float(initial_root_yaw_world)
        if not math.isfinite(self._initial_root_yaw):
            raise ValueError("initial_root_yaw_world must be finite")
        if (initial_support_pose_world is None) != (
            initial_support_contact is None
        ):
            raise ValueError(
                "initial support pose and contact must be provided together"
            )
        if initial_support_pose_world is not None:
            if not isinstance(initial_support_pose_world, KinematicPose):
                raise ValueError(
                    "initial_support_pose_world must be a KinematicPose"
                )
            support_contact = np.asarray(
                initial_support_contact, dtype=bool
            )
            if support_contact.shape != (2,) or not np.any(
                support_contact
            ):
                raise ValueError(
                    "initial_support_contact must contain supported feet"
                )
            self._initial_support_pose = initial_support_pose_world
            self._initial_support_contact = support_contact.copy()
        else:
            self._initial_support_pose = None
            self._initial_support_contact = None
        self._transition_halflife_s = float(transition_halflife_s)
        self._maximum_unblended_joint_step_rad = float(
            maximum_unblended_joint_step_rad
        )
        self._entry_planar_warp_release_frames = int(
            entry_planar_warp_release_frames
        )
        self._cooldown_length = int(reentry_cooldown_frames)
        self._landing_confirmation_frames = int(
            landing_confirmation_frames
        )
        self._allow_stalled_handoff_relaxation = (
            allow_stalled_handoff_relaxation
        )
        self._stalled_handoff_relaxation_frames = int(
            stalled_handoff_relaxation_frames
        )
        self._maximum_stalled_handoff_foot_error_m = float(
            maximum_stalled_handoff_foot_error_m
        )
        self._dt = 1.0 / float(fps)
        if (
            not math.isfinite(self._transition_halflife_s)
            or self._transition_halflife_s <= 0.0
            or not math.isfinite(
                self._maximum_unblended_joint_step_rad
            )
            or self._maximum_unblended_joint_step_rad <= 0.0
            or self._cooldown_length < 0
            or type(entry_planar_warp_release_frames) is not int
            or self._entry_planar_warp_release_frames <= 0
            or type(landing_confirmation_frames) is not int
            or self._landing_confirmation_frames <= 0
            or type(allow_stalled_handoff_relaxation) is not bool
            or type(stalled_handoff_relaxation_frames) is not int
            or self._stalled_handoff_relaxation_frames <= 0
            or not math.isfinite(
                self._maximum_stalled_handoff_foot_error_m
            )
            or self._maximum_stalled_handoff_foot_error_m <= 0.0
            or not math.isfinite(self._dt)
            or self._dt <= 0.0
        ):
            raise ValueError("hybrid timing values are invalid")
        self.mode = HybridMode.FLAT
        self.tick = 0
        self.current_pose: KinematicPose | None = None
        self.terrain_session: object | None = None
        self._terrain_raw_pose: KinematicPose | None = None
        self._entry_blend: PoseInertializer | None = None
        self._entry_blend_elapsed_s = 0.0
        self._stair_planar_warp_world_xy = np.zeros(
            2, dtype=np.float32
        )
        self._stair_planar_warp_frame = 0
        self._exit_blend: PoseInertializer | None = None
        self._exit_blend_elapsed_s = 0.0
        self._reentry_cooldown = 0
        self._post_landing_recovery_velocity_world_xy = np.zeros(
            2, dtype=np.float64
        )
        self._stair_placement: StairPlacement | None = None
        self._landing_gate: object | None = None
        self._landing_filter: object | None = None
        self._landing_end_stair: object | None = None
        self._landing_ready_frames = 0
        self._landing_source_clip = -1
        self._landing_source_frame = -1
        self._entry_no_match_safe_frames = 0
        self._planned_entry_row: int | None = None
        self._planned_flat_phase_rows: tuple[int, ...] = ()
        self._planned_phase_cursor = 0
        self._planned_phase_age_frames = 0
        self.relaxed_handoff_attempt_count = 0
        self.relaxed_handoff_entry_count = 0
        self.approach_velocity_correction_robot_xy = np.zeros(
            2, dtype=np.float64
        )

    @property
    def phase_staging_active(self) -> bool:
        return self._planned_entry_row is not None

    @property
    def post_landing_recovery_active(self) -> bool:
        return bool(
            self._reentry_cooldown > 0
            and np.any(
                np.abs(
                    self._post_landing_recovery_velocity_world_xy
                )
                > 0.0
            )
        )

    def _clear_entry_plan(self) -> None:
        self._planned_entry_row = None
        self._planned_flat_phase_rows = ()
        self._planned_phase_cursor = 0
        self._planned_phase_age_frames = 0

    def reset(self) -> KinematicPose:
        if self._initial_support_pose is None:
            pose = self.flat_source.reset(
                root_position_world=self._initial_root_position,
                root_yaw_world=self._initial_root_yaw,
            )
        else:
            reset_supported = getattr(
                self.flat_source, "reset_supported", None
            )
            if not callable(reset_supported):
                raise RuntimeError(
                    "initial support alignment requires reset_supported"
                )
            pose = reset_supported(
                root_position_world=self._initial_root_position,
                root_yaw_world=self._initial_root_yaw,
                support_pose_world=self._initial_support_pose,
                support_contact=self._initial_support_contact,
            )
        self.mode = HybridMode.FLAT
        self.tick = 0
        self.current_pose = pose
        self.terrain_session = None
        self._terrain_raw_pose = None
        self._entry_blend = None
        self._entry_blend_elapsed_s = 0.0
        self._stair_planar_warp_world_xy[:] = 0.0
        self._stair_planar_warp_frame = 0
        self._exit_blend = None
        self._exit_blend_elapsed_s = 0.0
        self._reentry_cooldown = 0
        self._post_landing_recovery_velocity_world_xy[:] = 0.0
        self._stair_placement = None
        self._landing_gate = None
        self._landing_filter = None
        self._landing_end_stair = None
        self._landing_ready_frames = 0
        self._landing_source_clip = -1
        self._landing_source_frame = -1
        self._entry_no_match_safe_frames = 0
        self._clear_entry_plan()
        self.relaxed_handoff_attempt_count = 0
        self.relaxed_handoff_entry_count = 0
        self.approach_velocity_correction_robot_xy[:] = 0.0
        self.intent_latch.reset()
        self.stair_observation_filter.reset()
        return pose

    def step(
        self,
        velocity_world_xy: np.ndarray,
        heading_world_yaw: float,
        future_root_xy: np.ndarray,
        future_facing_xy: np.ndarray,
        *,
        stop_requested: bool = False,
        stair_observation: ObservedStair | None = None,
        flat_path_safe: bool = True,
        flat_preview_validator: object | None = None,
        safe_recovery_velocity_world_xy: np.ndarray | None = None,
        intent_future_root_xy: np.ndarray | None = None,
        terrain_pose_filter: object | None = None,
    ) -> HybridStepOutcome:
        if self.current_pose is None:
            raise RuntimeError("hybrid session must be reset before step")
        intent_path = (
            np.asarray(future_root_xy, dtype=np.float32)
            if intent_future_root_xy is None
            else np.asarray(intent_future_root_xy, dtype=np.float32)
        )
        if (
            intent_path.shape
            != np.asarray(future_root_xy).shape
            or intent_path.ndim != 2
            or intent_path.shape[1:] != (2,)
            or not np.isfinite(intent_path).all()
        ):
            raise ValueError(
                "intent future path must match future_root_xy"
            )
        if (
            flat_preview_validator is not None
            and not callable(flat_preview_validator)
        ):
            raise ValueError("flat preview validator must be callable")
        recovery_velocity = (
            np.zeros(2, dtype=np.float64)
            if safe_recovery_velocity_world_xy is None
            else _finite(
                safe_recovery_velocity_world_xy,
                (2,),
                "safe_recovery_velocity_world_xy",
            )
        )
        if (
            self._reentry_cooldown > 0
            and not stop_requested
            and np.any(
                np.abs(
                    self._post_landing_recovery_velocity_world_xy
                )
                > 0.0
            )
        ):
            recovery_velocity = (
                self._post_landing_recovery_velocity_world_xy.copy()
            )
        if (
            terrain_pose_filter is not None
            and not callable(terrain_pose_filter)
        ):
            raise ValueError("terrain_pose_filter must be callable")
        if self.mode in (
            HybridMode.FLAT,
            HybridMode.STAIR_ARMED,
            HybridMode.SAFE_STOP,
            HybridMode.LANDING,
        ):
            stair_observation = self.stair_observation_filter.update(
                stair_observation,
                root_position_world=self.current_pose.root_position_world,
                root_yaw_world=self.current_pose.root_yaw_world,
            )
        self.tick += 1
        if self.mode in (
            HybridMode.FLAT,
            HybridMode.STAIR_ARMED,
            HybridMode.SAFE_STOP,
            HybridMode.LANDING,
        ):
            # The terrain observation and command knots were measured in the
            # robot frame at the beginning of this tick.  The flat matcher can
            # advance the kinematic pose before the handoff query below, so
            # retain that sensor-frame pose and explicitly re-express all
            # causal inputs in the advanced pose's frame.
            observation_pose = self.current_pose
            flat_advance_safe = bool(flat_path_safe)
            flat_pose_advanced = False

            def prepared_entry_match(
                prepared: object,
            ) -> StairEntryMatch | None:
                """Preview a flat phase against the imminent stair splice.

                Ordinary flat MM ranks candidates by flat pose/trajectory
                quality.  At an armed terrain handoff that ordering can keep
                selecting a perfectly safe but stair-incompatible support
                phase forever.  Re-express the same causal observation and
                command in each prepared pose, then prefer the first ranked
                flat candidate that already admits a real stair snippet.
                """

                if (
                    self.mode
                    not in (HybridMode.STAIR_ARMED, HybridMode.SAFE_STOP)
                    or stair_observation is None
                ):
                    return None
                candidate_pose = getattr(prepared, "pose", None)
                if not isinstance(candidate_pose, KinematicPose):
                    raise ValueError(
                        "prepared flat candidate has no KinematicPose"
                    )
                candidate_stair = _reframe_observed_stair(
                    stair_observation,
                    source_pose=observation_pose,
                    target_pose=candidate_pose,
                )
                candidate_root = _reframe_robot_local_xy(
                    np.asarray(future_root_xy, dtype=np.float32),
                    source_pose=observation_pose,
                    target_pose=candidate_pose,
                    positions=True,
                )
                candidate_facing = _reframe_robot_local_xy(
                    np.asarray(future_facing_xy, dtype=np.float32),
                    source_pose=observation_pose,
                    target_pose=candidate_pose,
                    positions=False,
                )
                entry_kwargs: dict[str, object] = {
                    "stair": candidate_stair,
                    # Supported handoffs use the current feet.  Airborne
                    # candidates remain conservatively ineligible until the
                    # prepared dense-foot preview is available.
                    "future_feet_world": None,
                }
                if terrain_pose_filter is not None:
                    support_method = getattr(
                        terrain_pose_filter, "support_contact", None
                    )
                    if callable(support_method):
                        entry_kwargs["support_contact"] = np.asarray(
                            support_method(candidate_pose), dtype=bool
                        )
                return self.entry_index.select(
                    candidate_pose,
                    candidate_root,
                    candidate_facing,
                    **entry_kwargs,
                )

            def choose_prepared(
                prepared_candidates: object,
            ) -> object | None:
                first_safe: object | None = None
                phase_search = (
                    self.mode
                    in (HybridMode.STAIR_ARMED, HybridMode.SAFE_STOP)
                    and stair_observation is not None
                )
                for candidate in prepared_candidates:
                    if not bool(flat_preview_validator(candidate)):
                        continue
                    if first_safe is None:
                        first_safe = candidate
                    if (
                        phase_search
                        and prepared_entry_match(candidate) is not None
                    ):
                        return candidate
                    if not phase_search:
                        return candidate
                return first_safe

            # The locked offline phase plan handles the hard case in one
            # explicit transaction.  Keep only a tiny online ranked fallback
            # so CPU browser playback remains real-time.
            flat_candidate_limit = 2
            if flat_advance_safe and flat_preview_validator is not None:
                prepare = getattr(self.flat_source, "prepare_step", None)
                prepare_many = getattr(
                    self.flat_source, "prepare_steps", None
                )
                commit = getattr(self.flat_source, "commit_step", None)
                if not callable(prepare) or not callable(commit):
                    raise RuntimeError(
                        "validated flat motion requires transactional source"
                    )
                planned_prepared: object | None = None
                prepare_row = getattr(
                    self.flat_source, "prepare_row", None
                )
                if (
                    self._planned_entry_row is not None
                    and self._planned_phase_cursor
                    < len(self._planned_flat_phase_rows)
                    and callable(prepare_row)
                ):
                    candidate = prepare_row(
                        self._planned_flat_phase_rows[
                            self._planned_phase_cursor
                        ],
                        velocity_world_xy=velocity_world_xy,
                        heading_world_yaw=heading_world_yaw,
                    )
                    if bool(flat_preview_validator(candidate)):
                        planned_prepared = candidate
                    else:
                        self._clear_entry_plan()
                if planned_prepared is not None:
                    prepared_candidates = (planned_prepared,)
                elif callable(prepare_many):
                    prepared_candidates = prepare_many(
                        velocity_world_xy=velocity_world_xy,
                        heading_world_yaw=heading_world_yaw,
                        maximum_candidates=flat_candidate_limit,
                    )
                else:
                    prepared_candidates = (
                        prepare(
                            velocity_world_xy=velocity_world_xy,
                            heading_world_yaw=heading_world_yaw,
                        ),
                    )
                prepared = choose_prepared(prepared_candidates)
                flat_advance_safe = prepared is not None
                if prepared is not None:
                    pose = commit(prepared)
                    flat_pose_advanced = True
                    if prepared is planned_prepared:
                        self._planned_phase_cursor += 1
                else:
                    # Do not freeze the gait in an incompatible swing phase
                    # at the terrain edge.  The flat bank contains explicit
                    # start/stop motion, so give it one transactional,
                    # zero-velocity opportunity to settle the feet while the
                    # exact same sole-clearance validator remains in force.
                    # This can expose a contact-compatible stair splice on a
                    # later frame without ever committing the unsafe
                    # travelling proposal.
                    brake_candidates = (
                            prepare_many(
                                velocity_world_xy=np.zeros(
                                    2, dtype=np.float64
                                ),
                                heading_world_yaw=heading_world_yaw,
                                maximum_candidates=flat_candidate_limit,
                            )
                        if callable(prepare_many)
                        else (
                            prepare(
                                velocity_world_xy=np.zeros(
                                    2, dtype=np.float64
                                ),
                                heading_world_yaw=heading_world_yaw,
                            ),
                        )
                    )
                    brake = choose_prepared(brake_candidates)
                    if brake is not None:
                        pose = commit(brake)
                        flat_pose_advanced = True
                    else:
                        pose = self.current_pose
            elif flat_advance_safe:
                pose = self.flat_source.step(
                    velocity_world_xy=velocity_world_xy,
                    heading_world_yaw=heading_world_yaw,
                )
                flat_pose_advanced = True
            elif flat_preview_validator is not None:
                # Translation toward the detected support discontinuity is
                # unsafe, but freezing the entire gait can strand an oblique
                # approach a few degrees outside the stair catalog.  Permit a
                # zero-travel heading adjustment only when the flat source can
                # preview transactionally and the exact same sole/terrain
                # oracle proves the selected motion stays on the current
                # support surface.
                prepare = getattr(self.flat_source, "prepare_step", None)
                prepare_many = getattr(
                    self.flat_source, "prepare_steps", None
                )
                commit = getattr(self.flat_source, "commit_step", None)
                if not callable(prepare) or not callable(commit):
                    raise RuntimeError(
                        "validated flat motion requires transactional source"
                    )
                planned_prepared = None
                prepare_row = getattr(
                    self.flat_source, "prepare_row", None
                )
                if (
                    self._planned_entry_row is not None
                    and self._planned_phase_cursor
                    < len(self._planned_flat_phase_rows)
                    and callable(prepare_row)
                ):
                    candidate = prepare_row(
                        self._planned_flat_phase_rows[
                            self._planned_phase_cursor
                        ],
                        velocity_world_xy=recovery_velocity,
                        heading_world_yaw=heading_world_yaw,
                    )
                    if bool(flat_preview_validator(candidate)):
                        planned_prepared = candidate
                    else:
                        self._clear_entry_plan()
                if planned_prepared is not None:
                    staged_turn_candidates = (planned_prepared,)
                elif callable(prepare_many):
                    staged_turn_candidates = prepare_many(
                        velocity_world_xy=recovery_velocity,
                        heading_world_yaw=heading_world_yaw,
                        maximum_candidates=flat_candidate_limit,
                    )
                else:
                    staged_turn_candidates = (
                        prepare(
                            velocity_world_xy=recovery_velocity,
                            heading_world_yaw=heading_world_yaw,
                        ),
                    )
                staged_turn = choose_prepared(staged_turn_candidates)
                if staged_turn is not None:
                    pose = commit(staged_turn)
                    flat_pose_advanced = True
                    if staged_turn is planned_prepared:
                        self._planned_phase_cursor += 1
                else:
                    pose = self.current_pose
            else:
                # Holding the last committed pose prevents command-filter
                # momentum from advancing across a support discontinuity.
                pose = self.current_pose
            if flat_pose_advanced and self._exit_blend is not None:
                self._exit_blend_elapsed_s += self._dt
                pose = self._exit_blend.apply(
                    pose, elapsed_s=self._exit_blend_elapsed_s
                )
                if (
                    self._exit_blend_elapsed_s
                    >= 6.0 * self._transition_halflife_s
                ):
                    self._exit_blend = None
            self.current_pose = pose
            if self.mode is HybridMode.LANDING:
                if callable(self._landing_filter):
                    filtered_landing = self._landing_filter(pose)
                    if filtered_landing is None:
                        pose = observation_pose
                    elif not isinstance(filtered_landing, KinematicPose):
                        raise ValueError(
                            "landing pose filter must return "
                            "KinematicPose or None"
                        )
                    else:
                        pose = filtered_landing
                    self.current_pose = pose
                if (
                    not callable(self._landing_gate)
                    or self._stair_placement is None
                ):
                    raise RuntimeError(
                        "landing bridge has no full-sole occupancy gate"
                    )
                landing_ready = self._landing_gate(
                    pose, self._stair_placement
                )
                if type(landing_ready) is not bool:
                    raise ValueError(
                        "landing occupancy gate must return bool"
                    )
                self._landing_ready_frames = (
                    self._landing_ready_frames + 1
                    if landing_ready
                    else 0
                )
                if (
                    self._landing_ready_frames
                    < self._landing_confirmation_frames
                ):
                    self.approach_velocity_correction_robot_xy[:] = 0.0
                    return HybridStepOutcome(
                        HybridMode.LANDING,
                        pose,
                        self.tick,
                        False,
                        False,
                        self._landing_source_clip,
                        self._landing_source_frame,
                        0.0,
                        0,
                        0.0,
                        (
                            "phase-matched flat landing bridge; waiting for "
                            "both complete soles beyond the top tread"
                            if flat_advance_safe
                            else (
                                "landing bridge held by transactional "
                                "terrain safety"
                            )
                        ),
                    )
                source_clip = self._landing_source_clip
                source_frame = self._landing_source_frame
                if callable(self._landing_end_stair):
                    self._landing_end_stair()
                self.mode = HybridMode.FLAT
                self._stair_placement = None
                self._landing_gate = None
                self._landing_filter = None
                self._landing_end_stair = None
                self._landing_ready_frames = 0
                self._landing_source_clip = -1
                self._landing_source_frame = -1
                self.intent_latch.reset()
                self.stair_observation_filter.reset()
                self._reentry_cooldown = self._cooldown_length
                self.approach_velocity_correction_robot_xy[:] = 0.0
                return HybridStepOutcome(
                    HybridMode.FLAT,
                    pose,
                    self.tick,
                    False,
                    True,
                    source_clip,
                    source_frame,
                    0.0,
                    0,
                    0.0,
                    "full-sole stair-to-flat landing handoff",
                )
            if self._reentry_cooldown > 0:
                self.approach_velocity_correction_robot_xy[:] = 0.0
                self._reentry_cooldown -= 1
                if self._reentry_cooldown == 0:
                    self._post_landing_recovery_velocity_world_xy[:] = 0.0
                self.mode = (
                    HybridMode.FLAT
                    if flat_advance_safe or flat_pose_advanced
                    else HybridMode.SAFE_STOP
                )
                self.intent_latch.reset()
                return HybridStepOutcome(
                    self.mode,
                    pose,
                    self.tick,
                    False,
                    False,
                    -1,
                    -1,
                    0.0,
                    0,
                    0.0,
                    (
                        "flat continuation after stair landing"
                        if flat_advance_safe
                        else (
                            "bounded flat recovery away from stair edge"
                            if flat_pose_advanced
                            else (
                                "safe stop at terrain edge during "
                                "re-entry cooldown"
                            )
                        )
                    ),
                )
            stair_for_pose = stair_observation
            future_root_for_pose = np.asarray(
                future_root_xy, dtype=np.float32
            )
            future_facing_for_pose = np.asarray(
                future_facing_xy, dtype=np.float32
            )
            intent_path_for_pose = intent_path
            if stair_for_pose is not None:
                stair_for_pose = _reframe_observed_stair(
                    stair_for_pose,
                    source_pose=observation_pose,
                    target_pose=pose,
                )
                future_root_for_pose = _reframe_robot_local_xy(
                    future_root_for_pose,
                    source_pose=observation_pose,
                    target_pose=pose,
                    positions=True,
                )
                future_facing_for_pose = _reframe_robot_local_xy(
                    future_facing_for_pose,
                    source_pose=observation_pose,
                    target_pose=pose,
                    positions=False,
                )
                intent_path_for_pose = _reframe_robot_local_xy(
                    intent_path_for_pose,
                    source_pose=observation_pose,
                    target_pose=pose,
                    positions=True,
                )
            intent_ready = self.intent_latch.update(
                stair_for_pose, intent_path_for_pose
            )
            if self.intent_latch.aligned_frames == 0:
                self._entry_no_match_safe_frames = 0
                self._clear_entry_plan()
                self.approach_velocity_correction_robot_xy[:] = 0.0
                self.mode = (
                    HybridMode.FLAT
                    if flat_advance_safe
                    else HybridMode.SAFE_STOP
                )
                return HybridStepOutcome(
                    self.mode,
                    pose,
                    self.tick,
                    False,
                    False,
                    -1,
                    -1,
                    0.0,
                    0,
                    0.0,
                    (
                        "free flat motion matching; no terrain-aligned intent"
                        if flat_advance_safe
                        else "safe stop before unsupported terrain edge"
                    ),
                )
            self.mode = HybridMode.STAIR_ARMED
            if not intent_ready:
                if not flat_advance_safe:
                    self.mode = HybridMode.SAFE_STOP
                return HybridStepOutcome(
                    self.mode,
                    pose,
                    self.tick,
                    False,
                    False,
                    -1,
                    -1,
                    0.0,
                    0,
                    0.0,
                    (
                        "stair intent dwell"
                        if flat_advance_safe
                        else "safe stop while stair intent dwells"
                    ),
                )
            if stair_for_pose is None:
                raise RuntimeError(
                    "ready stair intent has no robot-relative observation"
                )
            query_support_contact: np.ndarray | None = None
            if terrain_pose_filter is not None:
                support_method = getattr(
                    terrain_pose_filter, "support_contact", None
                )
                if callable(support_method):
                    query_support_contact = np.asarray(
                        support_method(pose), dtype=bool
                    )
            if self._planned_entry_row is None:
                route_method = getattr(
                    self.entry_index, "plan_route", None
                )
                phase_method = getattr(
                    self.flat_source,
                    "select_supported_phase_row",
                    None,
                )
                phase_rows_method = getattr(
                    self.flat_source,
                    "select_supported_phase_rows",
                    None,
                )
                route = (
                    route_method(
                        pose,
                        future_root_for_pose,
                        future_facing_for_pose,
                        stair=stair_for_pose,
                    )
                    if callable(route_method)
                    else None
                )
                if (
                    isinstance(route, StairEntryMatch)
                    and route.root_distance_m <= 0.18
                    and route.yaw_error_rad
                    <= max(
                        math.radians(12.0),
                        float(
                            getattr(
                                self.entry_index,
                                "maximum_yaw_error_rad",
                                math.radians(10.0),
                            )
                        ),
                    )
                    and (
                        callable(phase_rows_method)
                        or callable(phase_method)
                    )
                ):
                    target_phase = pose_from_stair_row(
                        self.database,
                        self.archive,
                        int(route.row),
                        placement=route.placement,
                    )
                    target_contact = np.asarray(
                        self.database.contact[int(route.row)],
                        dtype=bool,
                    )
                    if np.any(target_contact):
                        phase_rows = (
                            tuple(
                                int(row)
                                for row in phase_rows_method(
                                    target_phase, target_contact
                                )
                            )
                            if callable(phase_rows_method)
                            else (
                                int(
                                    phase_method(
                                        target_phase, target_contact
                                    )
                                ),
                            )
                        )
                        if not phase_rows:
                            raise RuntimeError(
                                "flat phase planner returned no lead-in"
                            )
                        self._planned_entry_row = int(route.row)
                        self._planned_flat_phase_rows = phase_rows
                        self._planned_phase_cursor = 0
                        self._planned_phase_age_frames = 0
            correction_method = getattr(
                self.entry_index, "approach_correction_robot", None
            )
            planned_correction_method = getattr(
                self.entry_index,
                "approach_correction_to_row",
                None,
            )
            if (
                self._planned_entry_row is not None
                and callable(planned_correction_method)
            ):
                self.approach_velocity_correction_robot_xy = np.asarray(
                    planned_correction_method(
                        stair=stair_for_pose,
                        row=self._planned_entry_row,
                    ),
                    dtype=np.float64,
                )
            else:
                self.approach_velocity_correction_robot_xy = (
                    np.asarray(
                        correction_method(
                            pose,
                            future_root_for_pose,
                            stair=stair_for_pose,
                            support_contact=query_support_contact,
                        ),
                        dtype=np.float64,
                    )
                    if callable(correction_method)
                    else np.zeros(2, dtype=np.float64)
                )
            future_feet_method = getattr(
                self.flat_source, "future_feet_world", None
            )
            future_feet_world = (
                future_feet_method(offsets_frames=FUTURE_OFFSETS)
                if callable(future_feet_method)
                else None
            )
            entry_kwargs: dict[str, object] = {
                "stair": stair_for_pose,
                "future_feet_world": future_feet_world,
            }
            if query_support_contact is not None:
                entry_kwargs["support_contact"] = query_support_contact
            relaxed_handoff = (
                self._allow_stalled_handoff_relaxation
                and self._entry_no_match_safe_frames
                >= self._stalled_handoff_relaxation_frames
                and hasattr(
                    self.entry_index,
                    "maximum_planted_foot_error_m",
                )
            )
            if relaxed_handoff:
                configured_limit = float(
                    self.entry_index.maximum_planted_foot_error_m
                )
                entry_kwargs[
                    "maximum_planted_foot_error_m_override"
                ] = max(
                    configured_limit,
                    self._maximum_stalled_handoff_foot_error_m,
                )
                self.relaxed_handoff_attempt_count += 1
            match = self.entry_index.select(
                pose,
                future_root_for_pose,
                future_facing_for_pose,
                **entry_kwargs,
            )
            if match is None:
                if self._planned_entry_row is not None:
                    self._planned_phase_age_frames += 1
                    if (
                        self._planned_phase_cursor
                        >= len(self._planned_flat_phase_rows)
                        or self._planned_phase_age_frames >= 45
                    ):
                        # One lead-in has had enough time to reach and pass
                        # its paired support phase.  Replan from the new
                        # robot-relative state instead of replaying the same
                        # short fragment in a visible loop.
                        self._clear_entry_plan()
                if not flat_advance_safe:
                    self.mode = HybridMode.SAFE_STOP
                    self._entry_no_match_safe_frames += 1
                else:
                    self._entry_no_match_safe_frames = 0
                return HybridStepOutcome(
                    self.mode,
                    pose,
                    self.tick,
                    False,
                    False,
                    -1,
                    -1,
                    0.0,
                    0,
                    0.0,
                    (
                        "stair armed; waiting for contact-compatible approach"
                        if flat_advance_safe
                        else "safe stop; no contact-compatible stair splice"
                    ),
                )
            if relaxed_handoff:
                self.relaxed_handoff_entry_count += 1
            self._entry_no_match_safe_frames = 0
            self._clear_entry_plan()
            self._post_landing_recovery_velocity_world_xy[:] = 0.0
            terrain_session = self._terrain_session_factory(
                self.database, initial_row=int(match.row)
            )
            self.approach_velocity_correction_robot_xy[:] = 0.0
            target = pose_from_stair_row(
                self.database,
                self.archive,
                int(match.row),
                placement=match.placement,
            )
            planar_warp = np.asarray(
                match.planar_warp_stair_xy, dtype=np.float32
            )
            planar_warp_world = np.zeros(2, dtype=np.float32)
            if np.any(np.abs(planar_warp) > 0.0):
                if match.placement is None:
                    planar_warp_world = planar_warp
                else:
                    planar_warp_world = (
                        _rotation_z(match.placement.ascent_yaw_world)[
                            :2, :2
                        ]
                        @ planar_warp
                    )
                target = _translate_pose_world_xy(
                    target, planar_warp_world
                )
            self._stair_planar_warp_world_xy = np.ascontiguousarray(
                planar_warp_world, dtype=np.float32
            )
            self._stair_planar_warp_frame = 0
            self._entry_blend = PoseInertializer(
                pose, target, halflife_s=self._transition_halflife_s
            )
            pose = self._entry_blend.apply(target, elapsed_s=0.0)
            if terrain_pose_filter is not None:
                begin_entry = getattr(
                    terrain_pose_filter, "begin_entry", None
                )
                if callable(begin_entry):
                    begin_entry()
                filtered = terrain_pose_filter(
                    pose,
                    np.asarray(
                        self.database.contact[int(match.row)],
                        dtype=bool,
                    ),
                )
                if filtered is None:
                    self._entry_blend = None
                    self._stair_planar_warp_world_xy[:] = 0.0
                    self._stair_planar_warp_frame = 0
                    self.mode = HybridMode.STAIR_ARMED
                    return HybridStepOutcome(
                        HybridMode.STAIR_ARMED,
                        self.current_pose,
                        self.tick,
                        False,
                        False,
                        -1,
                        -1,
                        0.0,
                        0,
                        0.0,
                        "stair entry rejected by exact terrain pose filter",
                    )
                if not isinstance(filtered, KinematicPose):
                    raise ValueError(
                        "terrain_pose_filter must return KinematicPose or None"
                    )
                pose = filtered
            self.current_pose = pose
            self.terrain_session = terrain_session
            self._terrain_raw_pose = target
            self.mode = HybridMode.STAIR
            self._stair_placement = match.placement
            self._entry_blend_elapsed_s = 0.0
            return HybridStepOutcome(
                HybridMode.STAIR,
                pose,
                self.tick,
                True,
                False,
                int(match.source_clip),
                int(match.source_frame),
                float(match.cost),
                1,
                float(match.planted_foot_error_m),
                "contact-compatible flat-to-stair handoff",
            )

        terrain_session = self.terrain_session
        if terrain_session is None:
            raise RuntimeError("stair mode has no terrain session")
        snapshot = None
        filter_snapshot = None
        filter_restore_method = None
        if terrain_pose_filter is not None:
            snapshot_method = getattr(
                terrain_session, "snapshot_state", None
            )
            restore_method = getattr(
                terrain_session, "restore_state", None
            )
            if not callable(snapshot_method) or not callable(restore_method):
                raise RuntimeError(
                    "filtered terrain playback requires transactional session"
                )
            snapshot = snapshot_method()
            filter_snapshot_method = getattr(
                terrain_pose_filter, "snapshot_state", None
            )
            filter_restore_method = getattr(
                terrain_pose_filter, "restore_state", None
            )
            if callable(filter_snapshot_method) != callable(
                filter_restore_method
            ):
                raise RuntimeError(
                    "stateful terrain pose filter must provide both "
                    "snapshot_state and restore_state"
                )
            if callable(filter_snapshot_method):
                filter_snapshot = filter_snapshot_method()
        terrain_outcome = terrain_session.step(
            future_root_xy,
            future_facing_xy,
            stop_requested=stop_requested,
        )
        target = pose_from_stair_row(
            self.database,
            self.archive,
            int(terrain_outcome.row),
            placement=self._stair_placement,
        )
        proposed_warp_frame = min(
            self._entry_planar_warp_release_frames,
            self._stair_planar_warp_frame + 1,
        )
        warp_fraction = max(
            0.0,
            1.0
            - proposed_warp_frame
            / float(self._entry_planar_warp_release_frames),
        )
        if (
            warp_fraction > 0.0
            and np.any(
                np.abs(self._stair_planar_warp_world_xy) > 0.0
            )
        ):
            target = _translate_pose_world_xy(
                target,
                self._stair_planar_warp_world_xy * warp_fraction,
            )
        proposed_entry_blend = self._entry_blend
        proposed_entry_elapsed_s = self._entry_blend_elapsed_s
        previous_raw_pose = self._terrain_raw_pose
        raw_joint_step = (
            0.0
            if previous_raw_pose is None
            else float(
                np.max(
                    np.abs(
                        target.joint_position
                        - previous_raw_pose.joint_position
                    ),
                    initial=0.0,
                )
            )
        )
        source_discontinuity = (
            raw_joint_step > self._maximum_unblended_joint_step_rad
        )
        proposed_raw_pose = target
        if (
            bool(getattr(terrain_outcome, "switched", False))
            or source_discontinuity
        ):
            # A terrain search result is a new short snippet, even when it
            # skips ahead within the same source clip.  Begin it exactly at
            # the last rendered pose.  The same applies to an authored
            # sequential frame whose joint delta is too large to track.
            proposed_entry_blend = PoseInertializer(
                self.current_pose,
                target,
                halflife_s=self._transition_halflife_s,
            )
            proposed_entry_elapsed_s = 0.0
        elif proposed_entry_blend is not None:
            proposed_entry_elapsed_s += self._dt
        pose = target
        if proposed_entry_blend is not None:
            pose = proposed_entry_blend.apply(
                target, elapsed_s=proposed_entry_elapsed_s
            )
        if terrain_pose_filter is not None:
            filtered = terrain_pose_filter(
                pose,
                np.asarray(
                    self.database.contact[int(terrain_outcome.row)],
                    dtype=bool,
                ),
            )
            if filtered is None:
                assert snapshot is not None
                fallback_accepted = False
                if proposed_entry_blend is not None and pose is not target:
                    # Joint-space inertialization can temporarily put a
                    # support foot through the terrain even when the authored
                    # target pose itself is collision-safe.  Prefer the smooth
                    # pose, but never deadlock source playback on that visual
                    # correction: transactionally validate the raw authored
                    # target and drop the blend only when the exact oracle
                    # accepts it.
                    if callable(filter_restore_method):
                        filter_restore_method(filter_snapshot)
                    raw_filtered = terrain_pose_filter(
                        target,
                        np.asarray(
                            self.database.contact[
                                int(terrain_outcome.row)
                            ],
                            dtype=bool,
                        ),
                    )
                    if raw_filtered is not None:
                        if not isinstance(raw_filtered, KinematicPose):
                            raise ValueError(
                                "terrain_pose_filter must return "
                                "KinematicPose or None"
                            )
                        pose = raw_filtered
                        filtered = raw_filtered
                        proposed_entry_blend = None
                        proposed_entry_elapsed_s = 0.0
                        fallback_accepted = True
                if (
                    not fallback_accepted
                    and bool(getattr(terrain_outcome, "switched", False))
                ):
                    terrain_session.restore_state(snapshot)
                    if callable(filter_restore_method):
                        filter_restore_method(filter_snapshot)
                    step_sequential = getattr(
                        terrain_session, "step_sequential", None
                    )
                    if callable(step_sequential):
                        fallback_outcome = step_sequential(
                            "exact pose filter rejected cross-clip match"
                        )
                        if not bool(
                            getattr(fallback_outcome, "held", False)
                        ):
                            fallback_target = pose_from_stair_row(
                                self.database,
                                self.archive,
                                int(fallback_outcome.row),
                                placement=self._stair_placement,
                            )
                            if (
                                warp_fraction > 0.0
                                and np.any(
                                    np.abs(
                                        self._stair_planar_warp_world_xy
                                    )
                                    > 0.0
                                )
                            ):
                                fallback_target = _translate_pose_world_xy(
                                    fallback_target,
                                    self._stair_planar_warp_world_xy
                                    * warp_fraction,
                                )
                            proposed_raw_pose = fallback_target
                            fallback_pose = fallback_target
                            fallback_blend = self._entry_blend
                            fallback_elapsed_s = (
                                self._entry_blend_elapsed_s
                            )
                            if fallback_blend is not None:
                                fallback_elapsed_s += self._dt
                                fallback_pose = fallback_blend.apply(
                                    fallback_target,
                                    elapsed_s=fallback_elapsed_s,
                                )
                            fallback_filtered = terrain_pose_filter(
                                fallback_pose,
                                np.asarray(
                                    self.database.contact[
                                        int(fallback_outcome.row)
                                    ],
                                    dtype=bool,
                                ),
                            )
                            if fallback_filtered is not None:
                                if not isinstance(
                                    fallback_filtered, KinematicPose
                                ):
                                    raise ValueError(
                                        "terrain_pose_filter must return "
                                        "KinematicPose or None"
                                    )
                                terrain_outcome = fallback_outcome
                                pose = fallback_filtered
                                filtered = fallback_filtered
                                proposed_entry_blend = fallback_blend
                                proposed_entry_elapsed_s = (
                                    fallback_elapsed_s
                                )
                                fallback_accepted = True
                if not fallback_accepted:
                    terrain_session.restore_state(snapshot)
                    if callable(filter_restore_method):
                        filter_restore_method(filter_snapshot)
                    active_row = int(terrain_session.active_row)
                    return HybridStepOutcome(
                        HybridMode.STAIR,
                        self.current_pose,
                        self.tick,
                        False,
                        False,
                        int(self.database.source_clip[active_row]),
                        int(self.database.source_frame[active_row]),
                        float(terrain_outcome.cost),
                        int(terrain_outcome.candidate_count),
                        float(
                            terrain_outcome.maximum_planted_foot_mismatch_m
                        ),
                        (
                            "terrain pose rejected; held and rolled back "
                            "the proposed source frame"
                        ),
                        terrain_outcome,
                    )
            if not isinstance(filtered, KinematicPose):
                raise ValueError(
                    "terrain_pose_filter must return KinematicPose or None"
                )
            pose = filtered
        self._entry_blend = proposed_entry_blend
        self._terrain_raw_pose = proposed_raw_pose
        self._stair_planar_warp_frame = proposed_warp_frame
        if (
            self._stair_planar_warp_frame
            >= self._entry_planar_warp_release_frames
        ):
            self._stair_planar_warp_world_xy[:] = 0.0
        if self._entry_blend is not None:
            self._entry_blend_elapsed_s = proposed_entry_elapsed_s
            if (
                self._entry_blend_elapsed_s
                >= 6.0 * self._transition_halflife_s
            ):
                self._entry_blend = None
        self.current_pose = pose
        ready = self.exit_index.is_ready(
            source_clip=int(terrain_outcome.source_clip),
            source_frame=int(terrain_outcome.source_frame),
        )
        if ready:
            source_name = str(
                self.archive.clip_names[int(terrain_outcome.source_clip)]
            )
            if self._stair_placement is not None:
                self._post_landing_recovery_velocity_world_xy = (
                    landing_recovery_velocity_world_xy(
                        self._stair_placement,
                        source_name,
                    )
                )
            landing_gate = (
                None
                if terrain_pose_filter is None
                else getattr(
                    terrain_pose_filter, "landing_complete", None
                )
            )
            end_stair = (
                None
                if terrain_pose_filter is None
                else getattr(terrain_pose_filter, "end_stair", None)
            )
            supported_reset = getattr(
                self.flat_source, "reset_supported", None
            )
            if callable(supported_reset):
                flat_target = supported_reset(
                    root_position_world=pose.root_position_world,
                    root_yaw_world=pose.root_yaw_world,
                    support_pose_world=pose,
                    support_contact=np.asarray(
                        self.database.contact[
                            int(terrain_outcome.row)
                        ],
                        dtype=bool,
                    ),
                )
            else:
                flat_target = self.flat_source.reset(
                    root_position_world=pose.root_position_world,
                    root_yaw_world=pose.root_yaw_world,
                )
            self._exit_blend = PoseInertializer(
                pose,
                flat_target,
                halflife_s=self._transition_halflife_s,
            )
            self._exit_blend_elapsed_s = 0.0
            self._entry_blend = None
            self.terrain_session = None
            self._terrain_raw_pose = None
            self._stair_planar_warp_world_xy[:] = 0.0
            self._stair_planar_warp_frame = 0
            if (
                callable(landing_gate)
                and self._stair_placement is not None
            ):
                begin_landing = getattr(
                    terrain_pose_filter, "begin_landing", None
                )
                landing_filter = getattr(
                    terrain_pose_filter, "filter_landing", None
                )
                if callable(begin_landing):
                    landing_side = (
                        "ground"
                        if source_name.startswith(
                            ("down_", "grail_down_")
                        )
                        else "top"
                    )
                    begin_landing(pose, side=landing_side)
                initial_ready = landing_gate(
                    pose, self._stair_placement
                )
                if type(initial_ready) is not bool:
                    raise ValueError(
                        "landing occupancy gate must return bool"
                    )
                self.mode = HybridMode.LANDING
                self._landing_gate = landing_gate
                self._landing_filter = (
                    landing_filter
                    if callable(landing_filter)
                    else None
                )
                self._landing_end_stair = (
                    end_stair if callable(end_stair) else None
                )
                self._landing_ready_frames = int(initial_ready)
                self._landing_source_clip = int(
                    terrain_outcome.source_clip
                )
                self._landing_source_frame = int(
                    terrain_outcome.source_frame
                )
                self.approach_velocity_correction_robot_xy[:] = 0.0
                return HybridStepOutcome(
                    HybridMode.LANDING,
                    pose,
                    self.tick,
                    False,
                    False,
                    int(terrain_outcome.source_clip),
                    int(terrain_outcome.source_frame),
                    float(terrain_outcome.cost),
                    int(terrain_outcome.candidate_count),
                    float(
                        terrain_outcome.maximum_planted_foot_mismatch_m
                    ),
                    (
                        "phase-matched flat landing bridge started; "
                        "stair source reached its landing stance"
                    ),
                    terrain_outcome,
                )
            if callable(end_stair):
                end_stair()
            self.mode = HybridMode.FLAT
            self._stair_placement = None
            self.intent_latch.reset()
            self.stair_observation_filter.reset()
            self._reentry_cooldown = self._cooldown_length
            return HybridStepOutcome(
                HybridMode.FLAT,
                pose,
                self.tick,
                False,
                True,
                int(terrain_outcome.source_clip),
                int(terrain_outcome.source_frame),
                float(terrain_outcome.cost),
                int(terrain_outcome.candidate_count),
                float(
                    terrain_outcome.maximum_planted_foot_mismatch_m
                ),
                "double-support stair-to-flat landing handoff",
                terrain_outcome,
            )
        return HybridStepOutcome(
            HybridMode.STAIR,
            pose,
            self.tick,
            False,
            False,
            int(terrain_outcome.source_clip),
            int(terrain_outcome.source_frame),
            float(terrain_outcome.cost),
            int(terrain_outcome.candidate_count),
            float(terrain_outcome.maximum_planted_foot_mismatch_m),
            str(terrain_outcome.reason),
            terrain_outcome,
        )
