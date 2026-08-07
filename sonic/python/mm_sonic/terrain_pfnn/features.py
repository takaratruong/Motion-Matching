"""Authoritative 288-input/268-target terrain-PFNN feature seam.

``center_frame=t`` is the recurrent input state. Pose, body state, contacts,
and the next trajectory are sampled at ``t+1``. Root motion and phase advance
are the transitions from ``t`` to ``t+1``, which the runtime integrates after
inference to reach that predicted state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
import re
from typing import Callable, Literal

import numpy as np

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_oracle.math3d import quaternion_multiply_wxyz

from .layout import INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S
from .phase import ContactPhaseTrack
from .sources import PFNNSourceClip
from .splits import (
    SplitName,
    split_identity as sealed_split_identity,
    terrain_identity as canonical_split_identity,
)


TerrainClass = Literal["flat", "ascent", "descent", "transition"]
HeightFunction = Callable[[np.ndarray], np.ndarray]

_FPS = 30.0
_TERRAIN_HALF_WIDTH_M = 0.25
_FLAT_GRADE_DEGREES = 1.0
_STATIONARY_SPEED_M_S = 0.05
_TWO_PI = 2.0 * math.pi
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLAR_MIRROR = np.array((1.0, -1.0, 1.0), dtype=np.float64)
_CONTACT_MIRROR = np.array((2, 3, 0, 1), dtype=np.int64)


def _mirror_permutation(names: tuple[str, ...]) -> np.ndarray:
    indices = {name: index for index, name in enumerate(names)}
    output = np.arange(len(names), dtype=np.int64)
    for index, name in enumerate(names):
        if name.startswith("left_"):
            output[index] = indices["right_" + name[len("left_") :]]
        elif name.startswith("right_"):
            output[index] = indices["left_" + name[len("right_") :]]
    return output


_JOINT_MIRROR = _mirror_permutation(ISAACLAB_JOINT_NAMES)
_BODY_MIRROR = _mirror_permutation(ISAACLAB_BODY_NAMES)
_JOINT_MIRROR_SIGN = np.asarray(
    [-1.0 if "roll" in name or "yaw" in name else 1.0
     for name in ISAACLAB_JOINT_NAMES],
    dtype=np.float64,
)
assert np.array_equal(_JOINT_MIRROR[_JOINT_MIRROR], np.arange(29))
assert np.array_equal(_BODY_MIRROR[_BODY_MIRROR], np.arange(30))
assert np.array_equal(_JOINT_MIRROR_SIGN**2, np.ones(29))


def _readonly_vector(value: object, width: int, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iuf" or source.shape != (width,):
        raise ValueError(f"{label} must be a real vector with shape ({width},)")
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(source, dtype=np.float64).astype(
            np.float32, casting="unsafe", copy=True
        )
    if not np.isfinite(output).all():
        raise ValueError(f"{label} must contain finite binary32 values")
    output = np.ascontiguousarray(output).copy()
    output.flags.writeable = False
    return output


@dataclass(frozen=True)
class PFNNTrainingWindow:
    x: np.ndarray
    y: np.ndarray
    phase: float
    clip_id: str
    split_identity: str
    split: SplitName
    center_frame: int
    motion_sha256: str
    terrain_sha256: str | None
    terrain_class: TerrainClass = "flat"

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _readonly_vector(self.x, 288, "x"))
        object.__setattr__(self, "y", _readonly_vector(self.y, 268, "y"))
        phase = float(self.phase)
        if not math.isfinite(phase) or not 0.0 <= phase < _TWO_PI:
            raise ValueError("phase must be finite in [0, 2*pi)")
        object.__setattr__(self, "phase", phase)
        if type(self.clip_id) is not str or not self.clip_id:
            raise ValueError("clip_id must be a nonempty string")
        if self.split_identity != canonical_split_identity(self.clip_id):
            raise ValueError("split_identity must equal the canonical identity")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if self.split != sealed_split_identity(self.split_identity):
            raise ValueError("split does not match the sealed split")
        if self.terrain_class not in ("flat", "ascent", "descent", "transition"):
            raise ValueError("terrain_class is invalid")
        if type(self.center_frame) is not int or self.center_frame < 0:
            raise ValueError("center_frame must be a nonnegative integer")
        if _SHA256_RE.fullmatch(self.motion_sha256) is None:
            raise ValueError("motion_sha256 must be a lowercase SHA-256 digest")
        if (
            self.terrain_sha256 is not None
            and _SHA256_RE.fullmatch(self.terrain_sha256) is None
        ):
            raise ValueError("terrain_sha256 must be a lowercase SHA-256 digest or None")
        if not np.isin(self.y[OUTPUT_LAYOUT["contact_logit"]], (0.0, 1.0)).all():
            raise ValueError("contact_logit targets must be binary labels")


@dataclass(frozen=True)
class PFNNWindowRejection:
    center_frame: int | None
    reason: str


@dataclass(frozen=True)
class PFNNWindowBuildResult:
    windows: tuple[PFNNTrainingWindow, ...]
    rejections: tuple[PFNNWindowRejection, ...]


class _RejectWindow(ValueError):
    pass


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _world_to_local(vector: np.ndarray, yaw: float) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    output = values.copy()
    output[..., 0] = cosine * values[..., 0] + sine * values[..., 1]
    output[..., 1] = -sine * values[..., 0] + cosine * values[..., 1]
    return output


def _linear_sample(trace: np.ndarray, frame: float) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    if not math.isfinite(frame) or frame < 0.0 or frame > len(values) - 1:
        raise _RejectWindow("trajectory_knot_outside_clip")
    lower = int(math.floor(frame))
    upper = min(lower + 1, len(values) - 1)
    fraction = frame - lower
    return (1.0 - fraction) * values[lower] + fraction * values[upper]


def _query_heights(height_at: HeightFunction, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    try:
        heights = np.asarray(height_at(points), dtype=np.float64)
    except (ArithmeticError, TypeError, ValueError) as error:
        raise _RejectWindow("terrain_ray_missing") from error
    if heights.shape != points.shape[:-1] or not np.isfinite(heights).all():
        raise _RejectWindow("terrain_ray_missing")
    return heights


def _root_planar_velocity(root_position: np.ndarray) -> np.ndarray:
    xy = np.asarray(root_position[:, :2], dtype=np.float64)
    output = np.empty_like(xy)
    output[0] = (xy[1] - xy[0]) * _FPS
    output[-1] = (xy[-1] - xy[-2]) * _FPS
    if len(xy) > 2:
        output[1:-1] = (xy[2:] - xy[:-2]) * (_FPS / 2.0)
    return output


def _trajectory(
    clip: PFNNSourceClip,
    *,
    reference_frame: int,
    root_yaw: np.ndarray,
    facing_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = reference_frame + TRAJECTORY_TIMES_S * _FPS
    positions_world = np.stack(
        [_linear_sample(clip.root_position_world, float(frame)) for frame in frames]
    )
    directions_world = np.stack(
        [_linear_sample(facing_world, float(frame)) for frame in frames]
    )
    direction_norm = np.linalg.norm(directions_world, axis=1)
    if np.any(direction_norm < 1.0e-8) or not np.isfinite(direction_norm).all():
        raise _RejectWindow("invalid_trajectory_facing")
    directions_world /= direction_norm[:, None]
    origin = np.asarray(clip.root_position_world[reference_frame, :2], dtype=np.float64)
    yaw = float(root_yaw[reference_frame])
    position_local = _world_to_local(positions_world[:, :2] - origin, yaw)
    direction_local = _world_to_local(directions_world, yaw)
    velocity_world = _root_planar_velocity(clip.root_position_world)
    sampled_speed = np.asarray(
        [np.linalg.norm(_linear_sample(velocity_world, float(frame))) for frame in frames]
    )
    walking = sampled_speed >= _STATIONARY_SPEED_M_S
    intent = np.column_stack((~walking, walking)).astype(np.float64)
    return position_local, direction_local, intent


def _terrain_track(
    clip: PFNNSourceClip,
    *,
    reference_frame: int,
    position_local: np.ndarray,
    direction_local: np.ndarray,
    root_yaw: np.ndarray,
    height_at: HeightFunction,
) -> np.ndarray:
    yaw = float(root_yaw[reference_frame])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    root_xy = np.asarray(clip.root_position_world[reference_frame, :2], dtype=np.float64)
    center_world = position_local @ rotation.T + root_xy
    direction_world = direction_local @ rotation.T
    left_world = center_world + _TERRAIN_HALF_WIDTH_M * np.column_stack(
        (-direction_world[:, 1], direction_world[:, 0])
    )
    right_world = center_world - _TERRAIN_HALF_WIDTH_M * np.column_stack(
        (-direction_world[:, 1], direction_world[:, 0])
    )
    probes = np.stack((left_world, center_world, right_world), axis=1)
    heights = _query_heights(height_at, probes)
    support = _query_heights(height_at, root_xy[None, :])[0]
    return heights - support


def _local_body_state(
    clip: PFNNSourceClip,
    *,
    frame: int,
    root_yaw: np.ndarray,
    support_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.array(
        (*np.asarray(clip.root_position_world[frame, :2], dtype=np.float64), support_height),
        dtype=np.float64,
    )
    position = _world_to_local(
        np.asarray(clip.body_position_world[frame], dtype=np.float64) - origin,
        float(root_yaw[frame]),
    )
    velocity = _world_to_local(
        np.asarray(clip.body_linear_velocity_world[frame], dtype=np.float64),
        float(root_yaw[frame]),
    )
    return position, velocity


def _root_tilt(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    inverse_yaw = np.array((math.cos(half), 0.0, 0.0, -math.sin(half)))
    residual = quaternion_multiply_wxyz(inverse_yaw, quaternion)
    residual = np.asarray(residual, dtype=np.float64)
    residual /= np.linalg.norm(residual)
    if residual[0] < 0.0:
        residual *= -1.0
    vector_norm = float(np.linalg.norm(residual[1:]))
    if vector_norm < 1.0e-12:
        return np.zeros(2, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(residual[0]))
    return residual[1:3] * (angle / vector_norm)


def _family_grade_degrees(
    clip: PFNNSourceClip, height_at: HeightFunction
) -> float | None:
    try:
        height = _query_heights(height_at, clip.root_position_world[:, :2])
    except _RejectWindow:
        return None
    distance = np.linalg.norm(np.diff(clip.root_position_world[:, :2], axis=0), axis=1)
    usable = distance > 1.0e-8
    if not np.any(usable):
        return 0.0
    grade = np.degrees(np.arctan2(np.abs(np.diff(height)[usable]), distance[usable]))
    return float(np.max(grade))


def _validate_terrain_family(clip: PFNNSourceClip, height_at: HeightFunction) -> None:
    if re.search(r"slope_\d+", clip.terrain_id) is None:
        return
    grade = _family_grade_degrees(clip, height_at)
    if grade is None:
        raise ValueError("terrain_family_grade_unmeasurable")
    if grade < 5.0 - 1.0e-6:
        raise ValueError(
            f"terrain_family_grade_below_5_degrees: measured {grade:.6f}"
        )
    if grade > 20.0 + 1.0e-6:
        raise ValueError(
            f"terrain_family_grade_above_20_degrees: measured {grade:.6f}"
        )


def _terrain_class(
    clip: PFNNSourceClip,
    frame: int,
    height_at: HeightFunction,
) -> TerrainClass:
    indices = np.array((frame - 1, frame, frame + 1), dtype=np.int64)
    points = np.asarray(clip.root_position_world[indices, :2], dtype=np.float64)
    height = _query_heights(height_at, points)
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    signed_grade = np.zeros(2, dtype=np.float64)
    moving = distance > 1.0e-8
    signed_grade[moving] = np.degrees(
        np.arctan2(np.diff(height)[moving], distance[moving])
    )
    state = np.where(
        signed_grade > _FLAT_GRADE_DEGREES,
        1,
        np.where(signed_grade < -_FLAT_GRADE_DEGREES, -1, 0),
    )
    if np.all(state == 0):
        return "flat"
    if np.all(state == 1):
        return "ascent"
    if np.all(state == -1):
        return "descent"
    return "transition"


def _stationary(clip: PFNNSourceClip, frame: int) -> bool:
    segment = np.asarray(clip.root_position_world[frame - 1 : frame + 2, :2])
    speed = np.linalg.norm(np.diff(segment, axis=0), axis=1) * _FPS
    return bool(np.max(speed) < _STATIONARY_SPEED_M_S)


def _validate_track(track: ContactPhaseTrack, frames: int) -> None:
    if not isinstance(track, ContactPhaseTrack):
        raise TypeError("phase_track must be a ContactPhaseTrack")
    expected = {
        "contact": (frames, 4),
        "confidence": (frames, 4),
        "phase": (frames,),
        "phase_advance": (frames,),
        "valid": (frames,),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(track, name))
        if value.shape != shape:
            raise ValueError(f"phase_track.{name} must have shape {shape}")
    if np.asarray(track.valid).dtype != np.dtype(np.bool_):
        raise ValueError("phase_track.valid must have boolean dtype")
    if (
        not np.isfinite(track.confidence).all()
        or not np.isfinite(track.phase).all()
        or not np.isfinite(track.phase_advance).all()
    ):
        raise ValueError("phase_track values must be finite")
    if not np.isin(np.asarray(track.contact), (False, True)).all():
        raise ValueError("phase_track.contact must contain binary labels")
    if np.any(np.asarray(track.phase_advance) < 0.0):
        raise ValueError("phase_track.phase_advance must be nonnegative")
    if np.any(np.asarray(track.phase) < 0.0) or np.any(
        np.asarray(track.phase) >= _TWO_PI
    ):
        raise ValueError("phase_track.phase must be in [0, 2*pi)")


def _pack_window(
    clip: PFNNSourceClip,
    phase_track: ContactPhaseTrack,
    *,
    center_frame: int,
    height_at: HeightFunction,
    root_yaw: np.ndarray,
    facing_world: np.ndarray,
    terrain_class: TerrainClass,
    idle: bool,
) -> PFNNTrainingWindow:
    target_frame = center_frame + 1
    if not idle and not (
        bool(phase_track.valid[center_frame])
        and bool(phase_track.valid[target_frame])
    ):
        raise _RejectWindow("invalid_phase")

    input_position, input_direction, intent = _trajectory(
        clip,
        reference_frame=center_frame,
        root_yaw=root_yaw,
        facing_world=facing_world,
    )
    target_position, target_direction, _ = _trajectory(
        clip,
        reference_frame=target_frame,
        root_yaw=root_yaw,
        facing_world=facing_world,
    )
    terrain = _terrain_track(
        clip,
        reference_frame=center_frame,
        position_local=input_position,
        direction_local=input_direction,
        root_yaw=root_yaw,
        height_at=height_at,
    )
    input_support = _query_heights(
        height_at, clip.root_position_world[center_frame : center_frame + 1, :2]
    )[0]
    target_support = _query_heights(
        height_at, clip.root_position_world[target_frame : target_frame + 1, :2]
    )[0]
    input_body_position, input_body_velocity = _local_body_state(
        clip,
        frame=center_frame,
        root_yaw=root_yaw,
        support_height=float(input_support),
    )
    target_body_position, target_body_velocity = _local_body_state(
        clip,
        frame=target_frame,
        root_yaw=root_yaw,
        support_height=float(target_support),
    )

    x = np.empty(INPUT_LAYOUT.size, dtype=np.float64)
    x[INPUT_LAYOUT["trajectory_position"]] = input_position.reshape(-1)
    x[INPUT_LAYOUT["trajectory_direction"]] = input_direction.reshape(-1)
    x[INPUT_LAYOUT["terrain_height"]] = terrain.reshape(-1)
    x[INPUT_LAYOUT["semantic_intent"]] = intent.reshape(-1)
    x[INPUT_LAYOUT["previous_body_position"]] = input_body_position.reshape(-1)
    x[INPUT_LAYOUT["previous_body_velocity"]] = input_body_velocity.reshape(-1)

    delta_world = (
        np.asarray(clip.root_position_world[target_frame, :2], dtype=np.float64)
        - np.asarray(clip.root_position_world[center_frame, :2], dtype=np.float64)
    )
    root_planar_velocity = (
        _world_to_local(delta_world, float(root_yaw[center_frame])) * _FPS
    )
    root_yaw_velocity = _wrap_angle(
        float(root_yaw[target_frame] - root_yaw[center_frame])
    ) * _FPS
    y = np.empty(OUTPUT_LAYOUT.size, dtype=np.float64)
    y[OUTPUT_LAYOUT["trajectory_position"]] = target_position.reshape(-1)
    y[OUTPUT_LAYOUT["trajectory_direction"]] = target_direction.reshape(-1)
    y[OUTPUT_LAYOUT["body_position"]] = target_body_position.reshape(-1)
    y[OUTPUT_LAYOUT["body_velocity"]] = target_body_velocity.reshape(-1)
    y[OUTPUT_LAYOUT["root_height"]] = (
        float(clip.root_position_world[target_frame, 2]) - target_support
    )
    y[OUTPUT_LAYOUT["root_tilt"]] = _root_tilt(
        clip.root_quaternion_world_wxyz[target_frame], float(root_yaw[target_frame])
    )
    y[OUTPUT_LAYOUT["joint_position"]] = clip.joint_position[target_frame]
    y[OUTPUT_LAYOUT["root_planar_velocity"]] = root_planar_velocity
    y[OUTPUT_LAYOUT["root_yaw_velocity"]] = root_yaw_velocity
    y[OUTPUT_LAYOUT["phase_advance"]] = (
        0.0 if idle else float(phase_track.phase_advance[center_frame])
    )
    # These are binary BCE-with-logits targets, not transformed logits.
    y[OUTPUT_LAYOUT["contact_logit"]] = np.asarray(
        phase_track.contact[target_frame], dtype=np.float64
    )

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise _RejectWindow("nonfinite_packed_value")
    return PFNNTrainingWindow(
        x=x,
        y=y,
        phase=(0.0 if idle else float(phase_track.phase[center_frame])) % _TWO_PI,
        clip_id=clip.clip_id,
        split_identity=canonical_split_identity(clip.clip_id),
        split=sealed_split_identity(clip.clip_id),
        center_frame=center_frame,
        motion_sha256=clip.motion_sha256,
        terrain_sha256=clip.terrain_sha256,
        terrain_class=terrain_class,
    )


def build_clip_windows(
    clip: PFNNSourceClip,
    phase_track: ContactPhaseTrack,
    *,
    height_at: HeightFunction,
) -> tuple[PFNNTrainingWindow, ...]:
    """Build every bounded ``t -> t+1`` window without crossing clip edges.

    Stationary flat centers are deterministically repeated at eight phase bins;
    ordinary centers require valid phase at both recurrent states.
    """

    result = build_clip_windows_with_audit(
        clip, phase_track, height_at=height_at
    )
    if not result.windows:
        rejection_counts = Counter(item.reason for item in result.rejections)
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(rejection_counts.items())
        )
        raise ValueError(f"no accepted training windows: {reasons}")
    return result.windows


def build_clip_windows_with_audit(
    clip: PFNNSourceClip,
    phase_track: ContactPhaseTrack,
    *,
    height_at: HeightFunction,
) -> PFNNWindowBuildResult:
    """Return accepted windows and every rejected center without information loss."""

    if not isinstance(clip, PFNNSourceClip):
        raise TypeError("clip must be a PFNNSourceClip")
    if not callable(height_at):
        raise TypeError("height_at must be callable")
    _validate_track(phase_track, clip.frame_count)
    _validate_terrain_family(clip, height_at)
    root_yaw = _yaw_from_wxyz(clip.root_quaternion_world_wxyz)
    facing_world = np.column_stack((np.cos(root_yaw), np.sin(root_yaw)))

    earliest = int(math.ceil(-float(np.min(TRAJECTORY_TIMES_S)) * _FPS))
    latest = int(
        math.floor(
            (clip.frame_count - 1)
            - float(np.max(TRAJECTORY_TIMES_S)) * _FPS
            - 1.0
        )
    )
    windows: list[PFNNTrainingWindow] = []
    rejections: list[PFNNWindowRejection] = []
    if latest < earliest:
        return PFNNWindowBuildResult(
            windows=(),
            rejections=(
                PFNNWindowRejection(None, "trajectory_knot_outside_clip"),
            ),
        )
    for center_frame in range(earliest, latest + 1):
        try:
            terrain_class = _terrain_class(clip, center_frame, height_at)
            idle = terrain_class == "flat" and _stationary(clip, center_frame)
            base = _pack_window(
                clip,
                phase_track,
                center_frame=center_frame,
                height_at=height_at,
                root_yaw=root_yaw,
                facing_world=facing_world,
                terrain_class=terrain_class,
                idle=idle,
            )
        except _RejectWindow as error:
            rejections.append(PFNNWindowRejection(center_frame, str(error)))
            continue
        if idle:
            for phase_bin in range(8):
                windows.append(
                    replace(base, phase=phase_bin * (_TWO_PI / 8.0))
                )
        else:
            windows.append(base)
    return PFNNWindowBuildResult(tuple(windows), tuple(rejections))


def _mirrored_clip_id(clip_id: str) -> str:
    suffix = "__mirror"
    return clip_id[: -len(suffix)] if clip_id.endswith(suffix) else clip_id + suffix


def mirror_window(window: PFNNTrainingWindow) -> PFNNTrainingWindow:
    """Apply the committed sagittal G1 mirror, including ``phase + pi``."""

    if not isinstance(window, PFNNTrainingWindow):
        raise TypeError("window must be a PFNNTrainingWindow")
    x = np.array(window.x, dtype=np.float64, copy=True)
    y = np.array(window.y, dtype=np.float64, copy=True)

    for layout, vector in (
        (INPUT_LAYOUT, x),
        (OUTPUT_LAYOUT, y),
    ):
        for field in ("trajectory_position", "trajectory_direction"):
            values = vector[layout[field]].reshape(12, 2)
            values[:, 1] *= -1.0
    terrain = x[INPUT_LAYOUT["terrain_height"]].reshape(12, 3)
    terrain[:] = terrain[:, ::-1].copy()
    for field in ("previous_body_position", "previous_body_velocity"):
        values = x[INPUT_LAYOUT[field]].reshape(30, 3)
        values[:] = values[_BODY_MIRROR] * _POLAR_MIRROR
    for field in ("body_position", "body_velocity"):
        values = y[OUTPUT_LAYOUT[field]].reshape(30, 3)
        values[:] = values[_BODY_MIRROR] * _POLAR_MIRROR
    root_tilt = y[OUTPUT_LAYOUT["root_tilt"]]
    root_tilt[0] *= -1.0
    joint = y[OUTPUT_LAYOUT["joint_position"]]
    joint[:] = joint[_JOINT_MIRROR] * _JOINT_MIRROR_SIGN
    y[OUTPUT_LAYOUT["root_planar_velocity"]][1] *= -1.0
    y[OUTPUT_LAYOUT["root_yaw_velocity"]] *= -1.0
    contact = y[OUTPUT_LAYOUT["contact_logit"]]
    contact[:] = contact[_CONTACT_MIRROR]

    return PFNNTrainingWindow(
        x=x,
        y=y,
        phase=(window.phase + math.pi) % _TWO_PI,
        clip_id=_mirrored_clip_id(window.clip_id),
        split_identity=window.split_identity,
        split=window.split,
        center_frame=window.center_frame,
        motion_sha256=window.motion_sha256,
        terrain_sha256=window.terrain_sha256,
        terrain_class=window.terrain_class,
    )


__all__ = [
    "PFNNTrainingWindow",
    "PFNNWindowBuildResult",
    "PFNNWindowRejection",
    "TerrainClass",
    "build_clip_windows",
    "build_clip_windows_with_audit",
    "mirror_window",
]
