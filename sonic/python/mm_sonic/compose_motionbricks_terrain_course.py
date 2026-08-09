"""Compose MotionBricks flat motion with one exact-safe stair traversal.

The input flat motions are native 30 Hz MotionBricks G1 qpos recordings from
``generate_motionbricks_terrain_transitions``.  This stage resamples them to
the 50 Hz terrain convention, selects compatible gait phases, applies only an
SE(2) registration, and joins approach→stair→exit with the same contact-aware
inertialization and complete-G1 mesh audit as the stair block compositor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_coherent_block_plan import (
    _mechanically_accepted,
    _repair_exact_mesh_clearance,
    _retarget_composed_seams_to_safe_footfalls,
)
from .compose_privileged_stair_route import (
    MotionSegment,
    _maximum_steps,
    _save_motion,
    concatenate_segments,
    endpoint_seam_metrics,
)
from .grail_terrain_source import mujoco_to_isaaclab_joints
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stair_geometry_warp import _archive_terrain_index
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


TARGET_FPS = 50.0
LOWER_BODY = np.asarray(
    (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18), dtype=np.int64
)


@dataclass(frozen=True)
class _PhaseCandidate:
    index: int
    score: float
    position_correction_m: float
    yaw_correction_rad: float
    source_planar_speed_mps: float
    root_speed_error_mps: float
    lower_body_rmse_rad: float
    maximum_lower_body_error_rad: float
    maximum_sole_position_gap_m: float
    support_foot: str
    support_foot_liftoff_frame: int
    swing_foot_liftoff_frame: int
    support_translation_world_xyz: tuple[float, float, float]


def _yaw_wxyz(value: np.ndarray) -> float:
    w, x, y, z = (float(component) for component in value)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap(value: float) -> float:
    return math.remainder(float(value), 2.0 * math.pi)


def _rotation_xy(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))


def _multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(l, -1, 0)
    rw, rx, ry, rz = np.moveaxis(r, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _sole_centres(
    segment: MotionSegment, adapter: _G1FootfallAdapter
) -> np.ndarray:
    """Return the two world-space sole centres for every motion frame."""

    return np.asarray(
        [
            [
                np.mean(value, axis=0)
                for value in adapter.sole_positions_for_pose(
                    root_position=np.asarray(
                        segment.root_position_world[frame], dtype=np.float64
                    ),
                    root_quaternion_wxyz=np.asarray(
                        segment.root_quaternion_world_wxyz[frame],
                        dtype=np.float64,
                    ),
                    joints=np.asarray(
                        segment.joint_position[frame], dtype=np.float64
                    ),
                )
            ]
            for frame in range(len(segment.root_position_world))
        ],
        dtype=np.float64,
    )


def _support_foot_after(
    centres: np.ndarray,
    *,
    start_frame: int,
    scan_frames: int,
) -> tuple[int, tuple[int, int]]:
    """Select the foot which stays planted longer after a handoff.

    Motion matching should register the support foot, not the average of a
    soon-to-swing foot and a planted foot.  The former is free to absorb a
    stance mismatch during its next swing; moving the latter creates exactly
    the visible slide/penetration that foot locking is meant to prevent.
    """

    start = int(start_frame)
    stop = min(len(centres), start + max(3, int(scan_frames)))
    values = np.asarray(centres[start:stop], dtype=np.float64)
    if len(values) < 3:
        return 0, (len(values), len(values))
    baseline_count = min(6, len(values))
    baseline_z = np.median(values[:baseline_count, :, 2], axis=0)
    liftoffs: list[int] = []
    for foot in range(2):
        liftoff = len(values)
        for frame in range(1, max(1, len(values) - 2)):
            if np.all(
                values[frame : frame + 3, foot, 2]
                - baseline_z[foot]
                > 0.008
            ):
                liftoff = frame
                break
        liftoffs.append(liftoff)
    return int(np.argmax(liftoffs)), (liftoffs[0], liftoffs[1])


def _resample_motionbricks(path: Path, *, label: str) -> MotionSegment:
    with np.load(path, allow_pickle=False) as arrays:
        source_fps = float(np.asarray(arrays["fps"]).item())
        root = np.asarray(arrays["root_position_world"], dtype=np.float64)
        quaternion_wxyz = np.asarray(
            arrays["root_quaternion_world_wxyz"], dtype=np.float64
        )
        joints_mujoco = np.asarray(
            arrays["joint_position_mjcf"], dtype=np.float64
        )
    if (
        root.ndim != 2
        or root.shape[1:] != (3,)
        or quaternion_wxyz.shape != (len(root), 4)
        or joints_mujoco.shape != (len(root), 29)
        or len(root) < 3
        or not np.isfinite(root).all()
        or not np.isfinite(quaternion_wxyz).all()
        or not np.isfinite(joints_mujoco).all()
        or not math.isfinite(source_fps)
        or source_fps <= 0.0
    ):
        raise ValueError(f"invalid MotionBricks recording: {path}")
    duration = (len(root) - 1) / source_fps
    count = int(math.floor(duration * TARGET_FPS + 1.0e-8)) + 1
    source_time = np.arange(len(root), dtype=np.float64) / source_fps
    target_time = np.arange(count, dtype=np.float64) / TARGET_FPS
    target_time = np.minimum(target_time, source_time[-1])
    output_root = np.stack(
        [np.interp(target_time, source_time, root[:, axis]) for axis in range(3)],
        axis=1,
    )
    output_joints_mujoco = np.stack(
        [
            np.interp(target_time, source_time, joints_mujoco[:, axis])
            for axis in range(29)
        ],
        axis=1,
    )
    normalized = quaternion_wxyz / np.linalg.norm(
        quaternion_wxyz, axis=1, keepdims=True
    )
    for index in range(1, len(normalized)):
        if float(np.dot(normalized[index - 1], normalized[index])) < 0.0:
            normalized[index] *= -1.0
    slerp = Slerp(
        source_time,
        Rotation.from_quat(normalized[:, (1, 2, 3, 0)]),
    )
    output_xyzw = slerp(target_time).as_quat()
    output_wxyz = output_xyzw[:, (3, 0, 1, 2)]
    output_joints = mujoco_to_isaaclab_joints(output_joints_mujoco)
    provenance = tuple(
        FrameProvenance(-10, index, label) for index in range(count)
    )
    return MotionSegment(
        label=label,
        root_position_world=np.asarray(output_root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(output_wxyz, dtype=np.float32),
        joint_position=np.asarray(output_joints, dtype=np.float32),
        provenance=provenance,
    )


def _load_stair_motion(path: Path) -> MotionSegment:
    with np.load(path, allow_pickle=False) as arrays:
        root = np.asarray(arrays["root_position_world"], dtype=np.float32)
        quaternion = np.asarray(
            arrays["root_quaternion_world_wxyz"], dtype=np.float32
        )
        joints = np.asarray(arrays["joint_position"], dtype=np.float32)
        if all(
            field in arrays
            for field in (
                "source_archive_clip_index",
                "source_frame",
                "source_clip_id",
            )
        ):
            clips = np.asarray(arrays["source_archive_clip_index"])
            frames = np.asarray(arrays["source_frame"])
            ids = np.asarray(arrays["source_clip_id"])
            provenance = tuple(
                FrameProvenance(int(clip), int(frame), str(clip_id))
                for clip, frame, clip_id in zip(
                    clips, frames, ids, strict=True
                )
            )
        else:
            provenance = tuple(
                FrameProvenance(-20, index, "exact_safe_stair")
                for index in range(len(root))
            )
    return MotionSegment(
        "exact_safe_stair", root, quaternion, joints, provenance
    )


def _phase_candidates(
    source: MotionSegment,
    target: MotionSegment,
    *,
    source_is_approach: bool,
    maximum_position_correction_m: float,
    maximum_yaw_correction_rad: float,
    candidate_limit: int,
    sole_adapter: _G1FootfallAdapter | None = None,
    minimum_planar_speed_mps: float = 0.0,
    minimum_approach_displacement_m: float = 0.0,
    planar_registration_is_free: bool = False,
) -> tuple[_PhaseCandidate, ...]:
    source_root = np.asarray(source.root_position_world, dtype=np.float64)
    source_quaternion = np.asarray(
        source.root_quaternion_world_wxyz, dtype=np.float64
    )
    source_joints = np.asarray(source.joint_position, dtype=np.float64)
    target_root = np.asarray(target.root_position_world, dtype=np.float64)
    target_quaternion = np.asarray(
        target.root_quaternion_world_wxyz, dtype=np.float64
    )
    target_joints = np.asarray(target.joint_position, dtype=np.float64)
    if source_is_approach:
        target_index = 0
        indices = range(max(2, int(0.65 * TARGET_FPS)), len(source_root))
    else:
        target_index = len(target_root) - 1
        first = max(2, int(0.25 * TARGET_FPS))
        stop = min(len(source_root) - 2, int(2.5 * TARGET_FPS))
        indices = range(first, max(first + 1, stop))
    target_yaw = _yaw_wxyz(target_quaternion[target_index])
    target_velocity = (
        (target_root[1] - target_root[0]) * TARGET_FPS
        if target_index == 0
        else (target_root[-1] - target_root[-2]) * TARGET_FPS
    )
    source_sole_centres = (
        None if sole_adapter is None else _sole_centres(source, sole_adapter)
    )
    target_sole_centres = (
        None if sole_adapter is None else _sole_centres(target, sole_adapter)
    )
    target_endpoint_soles = (
        None
        if sole_adapter is None
        else sole_adapter.sole_positions_for_pose(
            root_position=target_root[target_index],
            root_quaternion_wxyz=target_quaternion[target_index],
            joints=target_joints[target_index],
        )
    )
    rows: list[_PhaseCandidate] = []
    for index in indices:
        if source_is_approach and float(
            np.linalg.norm(source_root[index, :2] - source_root[0, :2])
        ) < float(minimum_approach_displacement_m):
            continue
        position_correction = float(
            np.linalg.norm(source_root[index] - target_root[target_index])
        )
        source_yaw = _yaw_wxyz(source_quaternion[index])
        yaw_correction = abs(_wrap(target_yaw - source_yaw))
        # A reusable flat template has no authored world frame.  When its
        # planar registration is free, both the XY shift *and* global yaw are
        # gauge freedoms; only the pose, velocity relative to the route, and
        # support-foot agreement distinguish phases.  Retaining the yaw gate
        # here accidentally made an otherwise identical stair fail merely
        # because its USD happened to be rotated in the global scene.
        if not planar_registration_is_free and (
            (
                source_is_approach
                and position_correction > float(maximum_position_correction_m)
            )
            or yaw_correction > float(maximum_yaw_correction_rad)
        ):
            continue
        yaw_offset = _wrap(target_yaw - source_yaw)
        source_velocity = (
            source_root[index] - source_root[index - 1]
        ) * TARGET_FPS
        source_velocity[:2] = _rotation_xy(yaw_offset) @ source_velocity[:2]
        source_planar_speed = float(np.linalg.norm(source_velocity[:2]))
        if source_planar_speed < float(minimum_planar_speed_mps):
            continue
        speed_error = float(np.linalg.norm(source_velocity - target_velocity))
        difference = (
            source_joints[index, LOWER_BODY]
            - target_joints[target_index, LOWER_BODY]
        )
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        maximum = float(np.max(np.abs(difference)))
        maximum_sole_gap = 0.0
        support_foot = 0
        liftoffs = (0, 0)
        support_translation = np.zeros(3, dtype=np.float64)
        if (
            sole_adapter is not None
            and source_sole_centres is not None
            and target_sole_centres is not None
        ):
            source_quaternion_aligned = _multiply_wxyz(
                _yaw_quaternion_wxyz(yaw_offset),
                source_quaternion[index],
            )
            source_soles = sole_adapter.sole_positions_for_pose(
                root_position=target_root[target_index],
                root_quaternion_wxyz=source_quaternion_aligned,
                joints=source_joints[index],
            )
            source_centres = np.asarray(
                [np.mean(source_soles[foot], axis=0) for foot in range(2)]
            )
            target_centres = np.asarray(
                target_sole_centres[target_index], dtype=np.float64
            )
            incoming_centres = (
                target_sole_centres
                if source_is_approach
                else source_sole_centres
            )
            incoming_start = 0 if source_is_approach else index
            support_foot, liftoffs = _support_foot_after(
                incoming_centres,
                start_frame=incoming_start,
                scan_frames=int(round(3.0 * TARGET_FPS)),
            )
            support_translation = (
                target_centres[support_foot]
                - source_centres[support_foot]
            )
            # Centre alignment alone is unsafe when the two ankle pitches
            # differ: the mean can agree while one toe/heel collision sphere
            # remains below the authored safe sole.  Use the smallest rigid
            # vertical lift that places every corresponding support-sole
            # point at or above its exact-safe target.  This retains the
            # source foot orientation and never warps individual frames.
            if target_endpoint_soles is not None:
                support_translation[2] = float(
                    np.max(
                        np.asarray(
                            target_endpoint_soles[support_foot],
                            dtype=np.float64,
                        )[:, 2]
                        - np.asarray(
                            source_soles[support_foot], dtype=np.float64
                        )[:, 2]
                    )
                )
            maximum_sole_gap = float(
                np.max(
                    np.linalg.norm(
                        source_centres
                        + support_translation[None, :]
                        - target_centres,
                        axis=1,
                    )
                )
            )
        score = float(
            (
                position_correction / 0.15
                if source_is_approach and not planar_registration_is_free
                else 0.0
            )
            + (
                0.0
                if planar_registration_is_free
                else yaw_correction / math.radians(12.0)
            )
            + speed_error / 0.40
            + rmse / 0.25
            + maximum / 0.70
            # The support sole is exact after registration.  The remaining
            # gap belongs to the early-lifting swing foot and is paid back in
            # flight, so it should influence ranking without dominating it.
            + maximum_sole_gap / 0.10
            + float(np.linalg.norm(support_translation)) / 0.12
        )
        swing_foot = 1 - support_foot
        rows.append(
            _PhaseCandidate(
                index=index,
                score=score,
                position_correction_m=position_correction,
                yaw_correction_rad=yaw_correction,
                source_planar_speed_mps=source_planar_speed,
                root_speed_error_mps=speed_error,
                lower_body_rmse_rad=rmse,
                maximum_lower_body_error_rad=maximum,
                maximum_sole_position_gap_m=maximum_sole_gap,
                support_foot=("left" if support_foot == 0 else "right"),
                support_foot_liftoff_frame=int(liftoffs[support_foot]),
                swing_foot_liftoff_frame=int(liftoffs[swing_foot]),
                support_translation_world_xyz=tuple(
                    float(value) for value in support_translation
                ),
            )
        )
    rows.sort(key=lambda value: (value.score, value.index))
    return tuple(rows[: int(candidate_limit)])


def _slice(segment: MotionSegment, start: int, stop: int) -> MotionSegment:
    return MotionSegment(
        label=f"{segment.label}[{start}:{stop}]",
        root_position_world=segment.root_position_world[start:stop],
        root_quaternion_world_wxyz=(
            segment.root_quaternion_world_wxyz[start:stop]
        ),
        joint_position=segment.joint_position[start:stop],
        provenance=segment.provenance[start:stop],
    )


def _trim_stationary_stair_margins(
    segment: MotionSegment,
    *,
    sole_adapter: _G1FootfallAdapter,
) -> tuple[MotionSegment, dict[str, object]]:
    """Retain the coherent traversal while removing authored idle margins."""

    frame_count = len(segment.root_position_world)
    sole_centres = np.asarray(
        [
            [
                np.mean(value, axis=0)
                for value in sole_adapter.sole_positions_for_pose(
                    root_position=np.asarray(
                        segment.root_position_world[frame], dtype=np.float64
                    ),
                    root_quaternion_wxyz=np.asarray(
                        segment.root_quaternion_world_wxyz[frame],
                        dtype=np.float64,
                    ),
                    joints=np.asarray(
                        segment.joint_position[frame], dtype=np.float64
                    ),
                )
            ]
            for frame in range(frame_count)
        ],
        dtype=np.float64,
    )
    foot_speed = np.zeros((frame_count, 2), dtype=np.float64)
    foot_speed[1:] = (
        np.linalg.norm(np.diff(sole_centres, axis=0), axis=2) * TARGET_FPS
    )
    moving = np.max(foot_speed, axis=1) > 0.08
    # Exact contact cleanup can create a two-frame correction at frame zero;
    # it is not the beginning of the authored traversal.
    moving[: max(3, int(round(0.25 * TARGET_FPS)))] = False
    sustained = np.convolve(
        moving.astype(np.int32), np.ones(5, dtype=np.int32), mode="same"
    ) >= 3
    indices = np.flatnonzero(sustained)
    if len(indices) == 0:
        return segment, {
            "applied": False,
            "original_frames": frame_count,
            "start_frame": 0,
            "stop_frame_exclusive": frame_count,
            "retained_frames": frame_count,
        }
    first_motion = int(indices[0])
    last_motion = int(indices[-1])
    start = max(0, first_motion - int(round(0.20 * TARGET_FPS)))
    stop = min(
        frame_count,
        last_motion + int(round(0.24 * TARGET_FPS)) + 1,
    )
    if stop - start < int(round(1.0 * TARGET_FPS)):
        raise ValueError("stationary-margin trim leaves no stair traversal")
    return _slice(segment, start, stop), {
        "applied": bool(start > 0 or stop < frame_count),
        "original_frames": frame_count,
        "first_sustained_foot_motion_frame": first_motion,
        "last_sustained_foot_motion_frame": last_motion,
        "start_frame": start,
        "stop_frame_exclusive": stop,
        "retained_frames": stop - start,
    }


def _align_sample(
    segment: MotionSegment,
    *,
    sample_index: int,
    target_position: np.ndarray,
    target_quaternion_wxyz: np.ndarray,
    support_translation_world_xyz: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> MotionSegment:
    root = np.asarray(segment.root_position_world, dtype=np.float64)
    quaternion = np.asarray(
        segment.root_quaternion_world_wxyz, dtype=np.float64
    )
    source_yaw = _yaw_wxyz(quaternion[int(sample_index)])
    target_yaw = _yaw_wxyz(np.asarray(target_quaternion_wxyz))
    yaw_offset = _wrap(target_yaw - source_yaw)
    rotation = _rotation_xy(yaw_offset)
    output_root = root.copy()
    output_root[:, :2] = (
        (root[:, :2] - root[int(sample_index), :2]) @ rotation.T
        + np.asarray(target_position, dtype=np.float64)[:2]
    )
    output_root[:, 2] += float(target_position[2] - root[int(sample_index), 2])
    output_root += np.asarray(
        support_translation_world_xyz, dtype=np.float64
    )[None, :]
    yaw_quaternion = _yaw_quaternion_wxyz(yaw_offset)
    output_quaternion = _multiply_wxyz(
        np.broadcast_to(yaw_quaternion, quaternion.shape), quaternion
    )
    output_quaternion /= np.linalg.norm(
        output_quaternion, axis=1, keepdims=True
    )
    return MotionSegment(
        segment.label,
        np.asarray(output_root, dtype=np.float32),
        np.asarray(output_quaternion, dtype=np.float32),
        segment.joint_position,
        segment.provenance,
    )


def _candidate_row(value: _PhaseCandidate) -> dict[str, object]:
    return {
        "index": value.index,
        "score": value.score,
        "position_correction_m": value.position_correction_m,
        "yaw_correction_rad": value.yaw_correction_rad,
        "source_planar_speed_mps": value.source_planar_speed_mps,
        "root_speed_error_mps": value.root_speed_error_mps,
        "lower_body_rmse_rad": value.lower_body_rmse_rad,
        "maximum_lower_body_error_rad": value.maximum_lower_body_error_rad,
        "maximum_sole_position_gap_m": value.maximum_sole_position_gap_m,
        "support_foot": value.support_foot,
        "support_foot_liftoff_frame": value.support_foot_liftoff_frame,
        "swing_foot_liftoff_frame": value.swing_foot_liftoff_frame,
        "support_translation_world_xyz": list(
            value.support_translation_world_xyz
        ),
    }


def _filter_exact_safe_phase_segments(
    source: MotionSegment,
    target: MotionSegment,
    candidates: tuple[_PhaseCandidate, ...],
    *,
    source_is_approach: bool,
    desired_count: int,
    stairs_archive: Path,
    target_clip_index: int | None,
    target_mesh: TerrainMeshIndex | None,
    model_path: Path,
) -> tuple[tuple[_PhaseCandidate, ...], list[dict[str, object]]]:
    """Keep phases whose unblended flat segment is terrain-compatible.

    Pose/velocity similarity cannot distinguish MotionBricks' startup buffer
    from a settled gait.  Since this is the globally privileged compiler, use
    the known exact mesh to reject a flat continuation with a body collision.
    Foot-only overlap up to the existing 25 mm clearance-repair budget remains
    eligible: an upper-platform approach can initially overlap the landing by
    10--15 mm even though the later contact retarget and smooth clearance pass
    resolves it cleanly.  This is only a candidate prefilter.  Every completed
    composition still passes the unchanged 5 mm foot / zero-body final audit.
    """

    target_index = 0 if source_is_approach else -1
    accepted: list[_PhaseCandidate] = []
    receipts: list[dict[str, object]] = []
    for candidate in candidates:
        aligned = _align_sample(
            source,
            sample_index=candidate.index,
            target_position=target.root_position_world[target_index],
            target_quaternion_wxyz=(
                target.root_quaternion_world_wxyz[target_index]
            ),
            support_translation_world_xyz=(
                candidate.support_translation_world_xyz
            ),
        )
        selected = (
            _slice(aligned, 0, candidate.index + 1)
            if source_is_approach
            else _slice(
                aligned,
                candidate.index,
                len(aligned.root_position_world),
            )
        )
        motion = StitchedMotion(
            fps=TARGET_FPS,
            root_position_world=selected.root_position_world,
            root_quaternion_world_wxyz=(
                selected.root_quaternion_world_wxyz
            ),
            joint_position=selected.joint_position,
            provenance=selected.provenance,
            seam_indices=(),
        )
        audit = audit_stair_motion_collisions(
            motion,
            archive_path=stairs_archive,
            target_clip_index=target_clip_index,
            target_mesh=target_mesh,
            model_path=model_path,
            maximum_foot_penetration_m=0.008,
            maximum_forbidden_body_penetration_m=1.0e-6,
        )
        accepted_without_repair = bool(audit.accepted)
        eligible_for_final_repair = bool(
            audit.maximum_forbidden_body_penetration_m <= 1.0e-6
            and audit.maximum_foot_penetration_m <= 0.025
        )
        receipts.append(
            {
                "phase": _candidate_row(candidate),
                "accepted": eligible_for_final_repair,
                "accepted_without_repair": accepted_without_repair,
                "eligible_for_final_repair": eligible_for_final_repair,
                "maximum_foot_penetration_m": float(
                    audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    audit.maximum_forbidden_body_penetration_m
                ),
            }
        )
        if eligible_for_final_repair:
            accepted.append(candidate)
            if len(accepted) >= int(desired_count):
                break
    return tuple(accepted), receipts


def compose(
    *,
    approach_raw: Path,
    exit_raw: Path,
    stair_motion: Path,
    stairs_archive: Path,
    target_clip_index: int | None,
    model_path: Path,
    output_dir: Path,
    phase_candidate_limit: int,
    maximum_candidate_combinations: int,
    render: bool,
    forced_approach_phase_index: int | None = None,
    forced_exit_phase_index: int | None = None,
    maximum_seam_foot_error_m: float = 0.018,
    target_mesh: TerrainMeshIndex | None = None,
    include_exit: bool = True,
) -> dict[str, object]:
    approach = _resample_motionbricks(approach_raw, label="motionbricks_approach")
    exit_motion = (
        _resample_motionbricks(exit_raw, label="motionbricks_exit")
        if include_exit
        else None
    )
    if not include_exit and forced_exit_phase_index is not None:
        raise ValueError("a terminal entry-only course has no exit phase")
    stair = _load_stair_motion(stair_motion)
    import zarr

    archive = zarr.open_group(str(stairs_archive), mode="r")
    if target_mesh is None:
        if target_clip_index is None:
            raise ValueError("target_mesh or target_clip_index is required")
        target_mesh = _archive_terrain_index(archive, int(target_clip_index))
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=0.95,
        target_tolerance_m=2.5e-4,
        maximum_iterations=96,
        damping=0.006,
        posture_weight=2.0e-5,
    )
    stair, stair_trim = _trim_stationary_stair_margins(
        stair, sole_adapter=adapter
    )
    provisional_candidate_limit = max(64, int(phase_candidate_limit) * 8)
    approach_candidates = _phase_candidates(
        approach,
        stair,
        source_is_approach=True,
        maximum_position_correction_m=0.55,
        maximum_yaw_correction_rad=math.radians(32.0),
        candidate_limit=(
            len(approach.root_position_world)
            if forced_approach_phase_index is not None
            else provisional_candidate_limit
        ),
        sole_adapter=adapter,
        # A reset/startup pose can match a quiet authored stair entry very
        # cheaply while contributing only a few centimetres of trajectory.
        # That turns an intended oblique entry family into a near-stationary
        # portal stub.  Select an already-moving gait phase so the retained
        # course carries the authored MotionBricks approach into the seam.
        minimum_planar_speed_mps=0.10,
        minimum_approach_displacement_m=0.35,
        planar_registration_is_free=True,
    )
    # The exit recording is intentionally rooted at the stair landing before
    # its best gait phase is known.  Translation is arbitrary there; phase
    # selection therefore treats the complete planar registration as free;
    # its score still includes velocity and lower-body compatibility in the
    # stair route frame.
    exit_candidates = (
        _phase_candidates(
            exit_motion,
            stair,
            source_is_approach=False,
            maximum_position_correction_m=2.0,
            maximum_yaw_correction_rad=math.radians(32.0),
            candidate_limit=(
                len(exit_motion.root_position_world)
                if forced_exit_phase_index is not None
                else provisional_candidate_limit
            ),
            sole_adapter=adapter,
            # MotionBricks has a command-generation buffer after reset.  An
            # idle phase can look deceptively compatible with the stationary
            # landing pose, but it leaves a support leg pinned when locomotion
            # begins. Select an active gait phase for a smooth continuation.
            minimum_planar_speed_mps=0.10,
            planar_registration_is_free=True,
        )
        if exit_motion is not None
        else ()
    )
    if forced_approach_phase_index is not None:
        approach_candidates = tuple(
            value
            for value in approach_candidates
            if value.index == int(forced_approach_phase_index)
        )
    if forced_exit_phase_index is not None:
        exit_candidates = tuple(
            value
            for value in exit_candidates
            if value.index == int(forced_exit_phase_index)
        )
    approach_segment_audits: list[dict[str, object]] = []
    exit_segment_audits: list[dict[str, object]] = []
    if forced_approach_phase_index is None:
        approach_candidates, approach_segment_audits = (
            _filter_exact_safe_phase_segments(
                approach,
                stair,
                approach_candidates,
                source_is_approach=True,
                desired_count=phase_candidate_limit,
                stairs_archive=stairs_archive,
                target_clip_index=target_clip_index,
                target_mesh=target_mesh,
                model_path=model_path,
            )
        )
    if exit_motion is not None and forced_exit_phase_index is None:
        exit_candidates, exit_segment_audits = (
            _filter_exact_safe_phase_segments(
                exit_motion,
                stair,
                exit_candidates,
                source_is_approach=False,
                desired_count=phase_candidate_limit,
                stairs_archive=stairs_archive,
                target_clip_index=target_clip_index,
                target_mesh=target_mesh,
                model_path=model_path,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema": "motionbricks_global_terrain_course_v1",
        "approach_raw": str(approach_raw.resolve()),
        "exit_raw": str(exit_raw.resolve()),
        "stair_motion": str(stair_motion.resolve()),
        "target_clip_index": (
            None if target_clip_index is None else int(target_clip_index)
        ),
        "forced_approach_phase_index": forced_approach_phase_index,
        "forced_exit_phase_index": forced_exit_phase_index,
        "maximum_seam_foot_error_m": float(maximum_seam_foot_error_m),
        "include_exit": bool(include_exit),
        "stair_trim": stair_trim,
        "approach_phase_candidates": [
            _candidate_row(value) for value in approach_candidates
        ],
        "exit_phase_candidates": [
            _candidate_row(value) for value in exit_candidates
        ],
        "approach_segment_audits": approach_segment_audits,
        "exit_segment_audits": exit_segment_audits,
        "trials": [],
        "status": "phase_coverage_failed",
    }
    if not approach_candidates or (include_exit and not exit_candidates):
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary

    ranked_pairs = sorted(
        (
            (
                approach_row.score
                + (0.0 if exit_row is None else exit_row.score),
                approach_row,
                exit_row,
            )
            for approach_row in approach_candidates
            for exit_row in (exit_candidates if include_exit else (None,))
        ),
        key=lambda value: (
            value[0],
            value[1].index,
            -1 if value[2] is None else value[2].index,
        ),
    )[: int(maximum_candidate_combinations)]
    selected = None
    best_rejected = None
    trials: list[dict[str, object]] = []
    halflives = (0.06, 0.08, 0.10, 0.12, 0.16, 0.20)
    for pair_index, (_, approach_row, exit_row) in enumerate(ranked_pairs):
        aligned_approach = _align_sample(
            approach,
            sample_index=approach_row.index,
            target_position=stair.root_position_world[0],
            target_quaternion_wxyz=stair.root_quaternion_world_wxyz[0],
            support_translation_world_xyz=(
                approach_row.support_translation_world_xyz
            ),
        )
        approach_segment = _slice(aligned_approach, 0, approach_row.index + 1)
        exit_segment = None
        if exit_row is not None:
            assert exit_motion is not None
            aligned_exit = _align_sample(
                exit_motion,
                sample_index=exit_row.index,
                target_position=stair.root_position_world[-1],
                target_quaternion_wxyz=stair.root_quaternion_world_wxyz[-1],
                support_translation_world_xyz=(
                    exit_row.support_translation_world_xyz
                ),
            )
            exit_segment = _slice(
                aligned_exit,
                exit_row.index,
                len(aligned_exit.root_position_world),
            )
        segments = [approach_segment, stair]
        if exit_segment is not None:
            segments.append(exit_segment)
        seam_metrics = [
            endpoint_seam_metrics(left, right)
            for left, right in zip(segments, segments[1:])
        ]
        for halflife in halflives:
            trial: dict[str, object] = {
                "pair_index": pair_index,
                "approach_phase": _candidate_row(approach_row),
                "exit_phase": (
                    None if exit_row is None else _candidate_row(exit_row)
                ),
                "inertialization_halflife_s": halflife,
                "raw_seams": seam_metrics,
            }
            # Prefer ordinary source-contact release, which gives the best
            # foot lock when an incoming flat phase begins in stance.  A phase
            # can also begin partway through a swing, where the stance detector
            # has no past context and may hold one foot indefinitely.  In that
            # case retry only the landing-to-flat seam with the deterministic
            # two-foot stagger used by the matcher.  This is a gait-phase
            # fallback, not a looser physical contract: both variants pass the
            # same mechanics and exact-mesh audits below.
            seam_mode_options = (
                (
                    ("source_contact_release", "source_contact_release"),
                    ("source_contact_release", "staggered"),
                )
                if include_exit
                else (("source_contact_release",),)
            )
            retarget_errors: list[str] = []
            chosen_seam_modes = None
            composed = None
            receipts = None
            for seam_modes in seam_mode_options:
                candidate_motion = concatenate_segments(
                    segments, fps=TARGET_FPS, halflife_s=halflife
                )
                try:
                    candidate_motion, candidate_receipts = (
                        _retarget_composed_seams_to_safe_footfalls(
                            candidate_motion,
                            segments,
                            adapter=adapter,
                            bridge_frames=36,
                            swing_clearance_m=0.035,
                            # IK convergence guard only.  The authoritative
                            # limits are 5 mm foot penetration, zero forbidden
                            # body penetration, and the mechanics gate below.
                            maximum_foot_error_m=float(
                                maximum_seam_foot_error_m
                            ),
                            seam_modes=seam_modes,
                            temporal_warm_start=True,
                        )
                    )
                except ValueError as error:
                    retarget_errors.append(str(error))
                    continue
                composed = candidate_motion
                receipts = candidate_receipts
                chosen_seam_modes = seam_modes
                break
            if composed is None or receipts is None:
                trial["status"] = "contact_retarget_rejected"
                trial["reason"] = retarget_errors[-1]
                trial["retarget_errors"] = retarget_errors
                trials.append(trial)
                continue
            trial["seam_modes"] = list(chosen_seam_modes)
            trial["retarget_fallback_errors"] = retarget_errors
            mechanics = _maximum_steps(composed)
            trial["contact_aware_seams"] = receipts
            trial["mechanics"] = mechanics
            if not _mechanically_accepted(mechanics):
                trial["status"] = "mechanics_rejected"
                trials.append(trial)
                continue
            audit = audit_stair_motion_collisions(
                composed,
                archive_path=stairs_archive,
                target_clip_index=target_clip_index,
                target_mesh=target_mesh,
                model_path=model_path,
                maximum_foot_penetration_m=0.005,
                maximum_forbidden_body_penetration_m=1.0e-6,
            )
            pre_repair_audit = audit
            clearance_repair = 0.0
            if not audit.accepted:
                composed, audit, clearance_repair = (
                    _repair_exact_mesh_clearance(
                        composed,
                        audit,
                        archive_path=stairs_archive,
                        target_clip_index=target_clip_index,
                        target_mesh=target_mesh,
                        model_path=model_path,
                        maximum_total_lift_m=0.025,
                    )
                )
                if clearance_repair > 0.0:
                    mechanics = _maximum_steps(composed)
                    trial["mechanics_after_clearance_repair"] = mechanics
                    if not _mechanically_accepted(mechanics):
                        trial["clearance_repair_maximum_m"] = (
                            clearance_repair
                        )
                        trial["status"] = (
                            "clearance_repair_mechanics_rejected"
                        )
                        trials.append(trial)
                        continue
            audit_row = {
                "accepted": bool(audit.accepted),
                "maximum_foot_penetration_m": float(
                    audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    audit.maximum_forbidden_body_penetration_m
                ),
                "maximum_foot_penetration_frame": int(
                    np.argmax(audit.per_frame_max_foot_penetration_m)
                ),
                "foot_threshold_exceedance_frame_indices": list(
                    audit.foot_threshold_exceedance_frame_indices
                ),
                "forbidden_body_threshold_exceedance_frame_indices": list(
                    audit.forbidden_body_threshold_exceedance_frame_indices
                ),
            }
            trial["pre_clearance_repair_full_body_audit"] = {
                "accepted": bool(pre_repair_audit.accepted),
                "maximum_foot_penetration_m": float(
                    pre_repair_audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    pre_repair_audit.maximum_forbidden_body_penetration_m
                ),
            }
            trial["clearance_repair_maximum_m"] = clearance_repair
            trial["full_body_audit"] = audit_row
            trial["status"] = "accepted" if audit.accepted else "audit_rejected"
            trials.append(trial)
            if not audit.accepted:
                rejection_severity = float(
                    audit.maximum_foot_penetration_m / 0.005
                    + audit.maximum_forbidden_body_penetration_m / 1.0e-6
                )
                rejected_candidate = (
                    rejection_severity,
                    composed,
                    trial,
                    approach_segment,
                    exit_segment,
                )
                if (
                    best_rejected is None
                    or rejected_candidate[0] < best_rejected[0]
                ):
                    best_rejected = rejected_candidate
                continue
            quality = float(
                approach_row.score
                + (0.0 if exit_row is None else exit_row.score)
                + mechanics["maximum_seam_root_acceleration_m_s2"] / 20.0
                + mechanics["maximum_seam_joint_acceleration_rad_s2"] / 180.0
                + audit.maximum_foot_penetration_m / 0.005
                + clearance_repair / 0.005
                + max(
                    float(
                        receipt.get("maximum_foot_target_error_m", 0.0)
                    )
                    for receipt in receipts
                )
                / 0.010
            )
            candidate = (
                quality,
                composed,
                trial,
                approach_segment,
                exit_segment,
            )
            if selected is None or candidate[0] < selected[0]:
                selected = candidate
        # An accepted first phase pair has already explored the full range of
        # smooth blend times.  Keep one extra pair available only when it fails.
        if selected is not None:
            break

    summary["trials"] = trials
    if selected is not None:
        quality, motion, selected_trial, approach_segment, exit_segment = selected
        _save_motion(output_dir / "motion.npz", motion)
        _save_motion(
            output_dir / "selected_motionbricks_approach.npz",
            StitchedMotion(
                fps=TARGET_FPS,
                root_position_world=approach_segment.root_position_world,
                root_quaternion_world_wxyz=(
                    approach_segment.root_quaternion_world_wxyz
                ),
                joint_position=approach_segment.joint_position,
                provenance=approach_segment.provenance,
                seam_indices=(),
            ),
        )
        if exit_segment is not None:
            _save_motion(
                output_dir / "selected_motionbricks_exit.npz",
                StitchedMotion(
                    fps=TARGET_FPS,
                    root_position_world=exit_segment.root_position_world,
                    root_quaternion_world_wxyz=(
                        exit_segment.root_quaternion_world_wxyz
                    ),
                    joint_position=exit_segment.joint_position,
                    provenance=exit_segment.provenance,
                    seam_indices=(),
                ),
            )
        _save_motion(
            output_dir / "selected_stair_traversal.npz",
            StitchedMotion(
                fps=TARGET_FPS,
                root_position_world=stair.root_position_world,
                root_quaternion_world_wxyz=(
                    stair.root_quaternion_world_wxyz
                ),
                joint_position=stair.joint_position,
                provenance=stair.provenance,
                seam_indices=(),
            ),
        )
        summary["status"] = "accepted"
        summary["quality"] = quality
        summary["selected_trial"] = selected_trial
        artifacts = {
            "motion": "motion.npz",
            "approach": "selected_motionbricks_approach.npz",
            "stair": "selected_stair_traversal.npz",
        }
        if exit_segment is not None:
            artifacts["exit"] = "selected_motionbricks_exit.npz"
        summary["artifacts"] = artifacts
        if render:
            summary["render"] = render_stitched_motion(
                motion,
                archive_path=stairs_archive,
                target_clip_index=target_clip_index,
                target_mesh=target_mesh,
                model_path=model_path,
                output_path=output_dir / "rollout.mp4",
            )
    elif best_rejected is not None:
        _, motion, rejected_trial, _, _ = best_rejected
        _save_motion(output_dir / "best_rejected_motion.npz", motion)
        summary["best_rejected_trial"] = rejected_trial
        summary["artifacts"] = {
            "best_rejected_motion": "best_rejected_motion.npz"
        }
        if render:
            summary["best_rejected_render"] = render_stitched_motion(
                motion,
                archive_path=stairs_archive,
                target_clip_index=target_clip_index,
                target_mesh=target_mesh,
                model_path=model_path,
                output_path=output_dir / "best_rejected_rollout.mp4",
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-raw", type=Path, required=True)
    parser.add_argument("--exit-raw", type=Path, required=True)
    parser.add_argument("--stair-motion", type=Path, required=True)
    parser.add_argument("--stairs-archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--target-clip-index", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-candidate-limit", type=int, default=3)
    parser.add_argument("--maximum-candidate-combinations", type=int, default=4)
    parser.add_argument("--forced-approach-phase-index", type=int)
    parser.add_argument("--forced-exit-phase-index", type=int)
    parser.add_argument("--maximum-seam-foot-error-m", type=float, default=0.018)
    parser.add_argument(
        "--omit-exit",
        action="store_true",
        help="end at the authored traversal and let the runtime synthesize the exit",
    )
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args()
    result = compose(
        approach_raw=arguments.approach_raw,
        exit_raw=arguments.exit_raw,
        stair_motion=arguments.stair_motion,
        stairs_archive=arguments.stairs_archive,
        target_clip_index=arguments.target_clip_index,
        model_path=arguments.model_path,
        output_dir=arguments.output_dir,
        phase_candidate_limit=arguments.phase_candidate_limit,
        maximum_candidate_combinations=arguments.maximum_candidate_combinations,
        render=not arguments.no_render,
        forced_approach_phase_index=arguments.forced_approach_phase_index,
        forced_exit_phase_index=arguments.forced_exit_phase_index,
        maximum_seam_foot_error_m=arguments.maximum_seam_foot_error_m,
        include_exit=not arguments.omit_exit,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
