"""Global portal playback around exact-safe MotionBricks terrain courses.

The flat controller remains the official MotionBricks generator.  A globally
known stair is represented by a precompiled flat->terrain->flat course which
has already passed complete-G1 mesh collision and mechanical gates.  This
module contains the runtime-neutral pieces: loading that course, deciding when
the character is at its entry portal, smoothly entering playback, and
resampling the last four poses so MotionBricks can continue from the landing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .gear_action import (
    isaaclab_to_mujoco_joint_vector,
    mujoco_to_isaaclab_joint_vector,
)


LOWER_BODY = np.asarray(
    (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18), dtype=np.int64
)


def _yaw_wxyz(value: object) -> float:
    w, x, y, z = (float(component) for component in np.asarray(value))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap(value: float) -> float:
    return math.remainder(float(value), 2.0 * math.pi)


def _slerp_wxyz(left: object, right: object, alpha: float) -> np.ndarray:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    amount = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        result = first + amount * (second - first)
    else:
        angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
        sine = math.sin(angle)
        result = (
            math.sin((1.0 - amount) * angle) / sine * first
            + math.sin(amount * angle) / sine * second
        )
    return result / np.linalg.norm(result)


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class PortalCapture:
    accepted: bool
    position_error_m: float
    yaw_error_rad: float
    travel_alignment: float
    lower_body_rmse_rad: float


@dataclass(frozen=True)
class PortalSelection:
    course_index: int
    capture: PortalCapture
    score: float
    entry_frame_index: int = 0


@dataclass(frozen=True)
class MotionBricksTerrainCourse:
    path: Path
    fps: float
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position_isaaclab: np.ndarray
    seam_indices: tuple[int, ...]

    @classmethod
    def load(cls, path: Path) -> "MotionBricksTerrainCourse":
        resolved = path.expanduser().resolve()
        with np.load(resolved, allow_pickle=False) as arrays:
            fps = float(np.asarray(arrays["fps"]).item())
            root = np.asarray(arrays["root_position_world"], dtype=np.float64)
            quaternion = np.asarray(
                arrays["root_quaternion_world_wxyz"], dtype=np.float64
            )
            joints = np.asarray(arrays["joint_position"], dtype=np.float64)
            seams = tuple(int(value) for value in arrays["seam_indices"])
        if (
            not math.isfinite(fps)
            or fps <= 0.0
            or root.ndim != 2
            or root.shape[1:] != (3,)
            or quaternion.shape != (len(root), 4)
            or joints.shape != (len(root), 29)
            or len(root) < 8
            or len(seams) not in (1, 2)
            or seams != tuple(sorted(seams))
            or not 0 < seams[0]
            or not seams[-1] < len(root)
            or not np.isfinite(root).all()
            or not np.isfinite(quaternion).all()
            or not np.isfinite(joints).all()
        ):
            raise ValueError(f"invalid MotionBricks terrain course: {resolved}")
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        return cls(resolved, fps, root, quaternion, joints, seams)

    @property
    def frame_count(self) -> int:
        return len(self.root_position_world)

    @property
    def start_yaw_world(self) -> float:
        return _yaw_wxyz(self.root_quaternion_world_wxyz[0])

    @property
    def approach_direction_world_xy(self) -> np.ndarray:
        stop = max(1, self.seam_indices[0] - 1)
        direction = (
            self.root_position_world[stop, :2]
            - self.root_position_world[0, :2]
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-6:
            raise ValueError("terrain course has no approach displacement")
        return direction / norm

    def entry_frame_index(self, lead_time_s: float) -> int:
        """Return a late flat-approach frame for just-in-time commitment."""

        if not math.isfinite(lead_time_s) or lead_time_s <= 0.0:
            raise ValueError("portal lead time must be positive and finite")
        lead_frames = max(2, int(round(float(lead_time_s) * self.fps)))
        return max(0, self.seam_indices[0] - lead_frames)

    def travel_direction_world_xy(self, frame_index: int = 0) -> np.ndarray:
        frame = int(frame_index)
        if frame < 0 or frame >= self.seam_indices[0]:
            raise ValueError("portal entry frame must precede the terrain seam")
        direction = (
            self.root_position_world[-1, :2]
            - self.root_position_world[frame, :2]
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-6:
            raise ValueError("terrain course has no entry displacement")
        return direction / norm

    def native_mujoco_qpos(self) -> np.ndarray:
        result = np.empty((self.frame_count, 36), dtype=np.float64)
        result[:, :3] = self.root_position_world
        result[:, 3:7] = self.root_quaternion_world_wxyz
        result[:, 7:] = np.stack(
            [
                isaaclab_to_mujoco_joint_vector(joints)
                for joints in self.joint_position_isaaclab
            ]
        )
        return result

    def portal_capture(
        self,
        current_qpos: object,
        requested_velocity_world_xy: object,
        *,
        maximum_position_error_m: float = 0.20,
        maximum_yaw_error_rad: float = math.radians(45.0),
        minimum_travel_alignment: float = 0.45,
        maximum_lower_body_rmse_rad: float = 0.38,
        frame_index: int = 0,
    ) -> PortalCapture:
        qpos = np.asarray(current_qpos, dtype=np.float64)
        velocity = np.asarray(requested_velocity_world_xy, dtype=np.float64)
        if qpos.shape != (36,) or velocity.shape != (2,):
            raise ValueError("portal capture expects qpos[36] and velocity[2]")
        frame = int(frame_index)
        if frame < 0 or frame >= self.seam_indices[0]:
            raise ValueError("portal capture frame must precede the terrain seam")
        position_error = float(
            np.linalg.norm(qpos[:2] - self.root_position_world[frame, :2])
        )
        reference_yaw = _yaw_wxyz(self.root_quaternion_world_wxyz[frame])
        yaw_error = abs(_wrap(_yaw_wxyz(qpos[3:7]) - reference_yaw))
        speed = float(np.linalg.norm(velocity))
        alignment = (
            -1.0
            if speed < 1.0e-5
            else float(
                np.dot(
                    velocity / speed,
                    self.travel_direction_world_xy(frame),
                )
            )
        )
        current_isaac = mujoco_to_isaaclab_joint_vector(qpos[7:])
        difference = (
            current_isaac[LOWER_BODY]
            - self.joint_position_isaaclab[frame, LOWER_BODY]
        )
        lower_body_rmse = float(np.sqrt(np.mean(np.square(difference))))
        accepted = bool(
            position_error <= float(maximum_position_error_m)
            and yaw_error <= float(maximum_yaw_error_rad)
            and alignment >= float(minimum_travel_alignment)
            and lower_body_rmse <= float(maximum_lower_body_rmse_rad)
        )
        return PortalCapture(
            accepted,
            position_error,
            yaw_error,
            alignment,
            lower_body_rmse,
        )

    def resampled_context_qpos(
        self,
        *,
        at_end: bool,
        target_fps: float = 30.0,
        frame_count: int = 4,
    ) -> np.ndarray:
        if target_fps <= 0.0 or frame_count < 2:
            raise ValueError("context sampling parameters are invalid")
        duration = (self.frame_count - 1) / self.fps
        if at_end:
            times = duration - np.arange(frame_count - 1, -1, -1) / target_fps
        else:
            times = np.arange(frame_count) / target_fps
        times = np.clip(times, 0.0, duration)
        coordinates = times * self.fps
        left = np.floor(coordinates).astype(np.int64)
        right = np.minimum(left + 1, self.frame_count - 1)
        alpha = coordinates - left
        qpos = self.native_mujoco_qpos()
        result = np.empty((frame_count, 36), dtype=np.float64)
        result[:, :3] = (
            (1.0 - alpha[:, None]) * qpos[left, :3]
            + alpha[:, None] * qpos[right, :3]
        )
        result[:, 7:] = (
            (1.0 - alpha[:, None]) * qpos[left, 7:]
            + alpha[:, None] * qpos[right, 7:]
        )
        result[:, 3:7] = np.stack(
            [
                _slerp_wxyz(qpos[a, 3:7], qpos[b, 3:7], value)
                for a, b, value in zip(left, right, alpha, strict=True)
            ]
        )
        return result


def select_terrain_portal(
    courses: tuple[MotionBricksTerrainCourse, ...],
    current_qpos: object,
    requested_velocity_world_xy: object,
    *,
    entry_lead_time_s: float | None = None,
) -> PortalSelection:
    """Choose the best globally known entry portal for the live pose.

    Multiple approach families can lead to the same exact stair traversal.
    Accepted captures always outrank non-captures; within either group the
    normalized position, yaw, pose, and travel errors choose the smoothest
    handoff deterministically.
    """

    if not courses:
        raise ValueError("terrain portal selection needs at least one course")
    ranked: list[tuple[bool, float, int, PortalCapture, int]] = []
    for index, course in enumerate(courses):
        entry_frame = (
            0
            if entry_lead_time_s is None
            else course.entry_frame_index(entry_lead_time_s)
        )
        capture = course.portal_capture(
            current_qpos,
            requested_velocity_world_xy,
            frame_index=entry_frame,
        )
        alignment_error = max(0.0, 0.45 - capture.travel_alignment)
        score = float(
            capture.position_error_m / 0.20
            + capture.yaw_error_rad / math.radians(45.0)
            + capture.lower_body_rmse_rad / 0.38
            + alignment_error / 0.45
        )
        ranked.append((not capture.accepted, score, index, capture, entry_frame))
    _, score, index, capture, entry_frame = min(
        ranked, key=lambda value: value[:3]
    )
    return PortalSelection(index, capture, score, entry_frame)


class TerrainCoursePlayback:
    """One committed course traversal with a short flat-entry residual blend."""

    def __init__(
        self,
        course: MotionBricksTerrainCourse,
        current_qpos: object,
        *,
        blend_frames: int = 18,
        start_frame: int = 0,
    ) -> None:
        current = np.asarray(current_qpos, dtype=np.float64)
        if current.shape != (36,) or not np.isfinite(current).all():
            raise ValueError("course playback needs one finite qpos[36]")
        first = int(start_frame)
        if first < 0 or first >= course.seam_indices[0]:
            raise ValueError("course playback must start before the terrain seam")
        if (
            blend_frames < 2
            or blend_frames >= course.seam_indices[0] - first
        ):
            raise ValueError("course entry blend must finish before the stair seam")
        self.course = course
        self.course_qpos = course.native_mujoco_qpos()
        self.start_qpos = current.copy()
        self.blend_frames = int(blend_frames)
        self.start_frame = first
        self.index = first

    @property
    def done(self) -> bool:
        return self.index >= self.course.frame_count

    def next_qpos(self) -> np.ndarray:
        if self.done:
            return self.course_qpos[-1].copy()
        frame = self.index
        result = self.course_qpos[frame].copy()
        blend_frame = frame - self.start_frame
        if blend_frame < self.blend_frames:
            alpha = _smoothstep(
                blend_frame / float(self.blend_frames - 1)
            )
            # Preserve the authored course displacement while smoothly paying
            # back the small capture residual on the collision-free flat lead.
            residual_weight = 1.0 - alpha
            result[:3] += residual_weight * (
                self.start_qpos[:3]
                - self.course_qpos[self.start_frame, :3]
            )
            result[3:7] = _slerp_wxyz(
                self.start_qpos[3:7], result[3:7], alpha
            )
            result[7:] += residual_weight * (
                self.start_qpos[7:]
                - self.course_qpos[self.start_frame, 7:]
            )
        self.index += 1
        return result
