"""Retarget genuine MotionBricks/BONES lateral gaits onto exact stair footholds.

The ordinary stair directional warp starts from a forward stair climb.  Merely
turning that pelvis by ninety degrees produces a twisted pose, not a lateral
stair gait.  This pilot instead starts from clips whose measured travel is
approximately perpendicular to body facing, aligns that travel with an exact
stairs500 route, locks every stance sole to a real tread, clears swing feet
over the risers, and admits only complete-mesh collision-free results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import zarr

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_coherent_block_plan import _smooth_upper_clearance_envelope
from .build_stairs500_omnidirectional_pilots import (
    DEFAULT_ARCHIVE,
    _fill_short_stance_gaps,
    _remove_short_stance_runs,
    _repair_exact_mesh_clearance,
    _save_motion,
    _stance_runs_maximum_drift,
    _terrain_height,
    warp_motion,
)
from .terrain_oracle.canonical import CanonicalTerrainMesh
from .terrain_oracle.contact import CanonicalMeshQuery
from .terrain_oracle.math3d import (
    RigidTransform,
    quaternion_multiply_wxyz,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_fragment_reconstruction import (
    _blend_anchored_sole_offsets,
    _stance_spans,
)
from .terrain_oracle.stair_geometry_warp import (
    _archive_terrain_index,
    motion_conditioned_stair_support_route,
)
from .terrain_oracle.stair_foothold_anchors import FootholdAnchorConfig
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_BONES_ROOT = Path(
    "/move/data/terrain-aware/motion-matching/"
    "takara_bones_balanced_twostick_v1"
)
DEFAULT_SOURCES = (
    Path(
        "/move/data/terrain-aware/motion-matching/"
        "motionbricks_flat_balanced_v2_ramp25/recordings/"
        "base__right_strafe.npz"
    ),
    Path(
        "/move/data/terrain-aware/motion-matching/"
        "motionbricks_flat_balanced_v2_ramp25/recordings/"
        "base__left_strafe.npz"
    ),
    DEFAULT_BONES_ROOT / "walk_sideway_right_loop_001__A026" / "motion.npz",
    DEFAULT_BONES_ROOT
    / "walk_sideway_right_loop_001__A026_M"
    / "motion.npz",
)
DEFAULT_OUTPUT = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "terrain_maneuver_stairs500_v1/bones_side_on_pilots_v1"
)
MAXIMUM_SIDE_ON_JOINT_CORRECTION_RAD = 1.30
MAXIMUM_ARM_VELOCITY_RAD_S = 2.5
MAXIMUM_WRIST_VELOCITY_RAD_S = 1.25
MAXIMUM_ARM_EXCURSION_RAD = 0.60
MAXIMUM_WRIST_EXCURSION_RAD = 0.25
MAXIMUM_ARM_RMS_VELOCITY_RAD_S = 0.75
MAXIMUM_WRIST_RMS_VELOCITY_RAD_S = 0.30
MAXIMUM_ARM_P99_ACCELERATION_RAD_S2 = 25.0
MAXIMUM_WRIST_P99_ACCELERATION_RAD_S2 = 15.0


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_wxyz(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    w, x, y, z = values.T
    return np.unwrap(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _robust_travel_direction(root_xy: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(root_xy, dtype=np.float64)
    window = max(3, len(values) // 20)
    displacement = np.median(values[-window:], axis=0) - np.median(
        values[:window], axis=0
    )
    distance = float(np.linalg.norm(displacement))
    if distance < 0.75:
        raise ValueError("lateral source has less than 0.75 m net travel")
    direction = displacement / distance
    return direction, math.atan2(float(direction[1]), float(direction[0]))


def _smooth_discontinuous_upper_body_joints(
    joints: np.ndarray,
    joint_names: Sequence[str],
    *,
    maximum_step_rad: float = 0.15,
    sigma_frames: float = 1.5,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Remove isolated MotionBricks arm/wrist branch switches."""

    values = np.asarray(joints, dtype=np.float64).copy()
    if values.shape[1] != len(joint_names):
        raise ValueError("joint names do not match lateral source columns")
    radius = max(1, int(math.ceil(3.0 * float(sigma_frames))))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(coordinates / float(sigma_frames)))
    kernel /= np.sum(kernel)
    smoothed: list[str] = []
    for column, name in enumerate(joint_names):
        if any(part in name for part in ("hip", "knee", "ankle")):
            continue
        if float(np.max(np.abs(np.diff(values[:, column])))) <= float(
            maximum_step_rad
        ):
            continue
        padded = np.pad(values[:, column], (radius, radius), mode="edge")
        values[:, column] = np.convolve(padded, kernel, mode="valid")
        smoothed.append(str(name))
    return values, tuple(smoothed)


def _upper_limb_motion_metrics(
    joints: np.ndarray,
    joint_names: Sequence[str],
    *,
    fps: float,
) -> dict[str, float]:
    values = np.asarray(joints, dtype=np.float64)
    arm = np.asarray(
        [
            index
            for index, name in enumerate(joint_names)
            if any(part in name for part in ("shoulder", "elbow", "wrist"))
        ],
        dtype=np.int64,
    )
    wrist = np.asarray(
        [index for index, name in enumerate(joint_names) if "wrist" in name],
        dtype=np.int64,
    )
    if not len(arm):
        raise ValueError("joint contract contains no upper-limb joints")

    def family(indices: np.ndarray, prefix: str) -> dict[str, float]:
        subset = values[:, indices]
        velocity = np.gradient(subset, axis=0) * float(fps)
        acceleration = np.diff(subset, n=2, axis=0) * float(fps) ** 2
        per_joint_rms_velocity = np.sqrt(np.mean(np.square(velocity), axis=0))
        return {
            f"maximum_{prefix}_excursion_rad": float(
                np.max(np.ptp(subset, axis=0))
            ),
            f"maximum_{prefix}_velocity_rad_s": float(
                np.max(np.abs(velocity))
            ),
            f"maximum_joint_rms_{prefix}_velocity_rad_s": float(
                np.max(per_joint_rms_velocity)
            ),
            f"p99_{prefix}_acceleration_rad_s2": (
                0.0
                if not acceleration.size
                else float(np.quantile(np.abs(acceleration), 0.99))
            ),
        }

    return {**family(arm, "arm"), **family(wrist, "wrist")}


def _attenuate_upper_limb_motion(
    joints: np.ndarray,
    joint_names: Sequence[str],
    *,
    fps: float,
    arm_swing_scale: float,
    wrist_swing_scale: float,
    smoothing_sigma_frames: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep modest counter-swing while removing high-energy hand motion."""

    arm_scale = float(arm_swing_scale)
    wrist_scale = float(wrist_swing_scale)
    sigma = float(smoothing_sigma_frames)
    if (
        not 0.0 <= arm_scale <= 1.0
        or not 0.0 <= wrist_scale <= 1.0
        or not math.isfinite(sigma)
        or sigma < 0.0
    ):
        raise ValueError("upper-limb attenuation parameters are invalid")
    values = np.asarray(joints, dtype=np.float64).copy()
    before = _upper_limb_motion_metrics(values, joint_names, fps=fps)
    radius = max(0, int(math.ceil(3.0 * sigma)))
    if radius:
        coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * np.square(coordinate / sigma))
        kernel /= np.sum(kernel)
    else:
        kernel = np.ones(1, dtype=np.float64)
    affected: list[str] = []
    for column, name in enumerate(joint_names):
        if not any(part in name for part in ("shoulder", "elbow", "wrist")):
            continue
        gain = wrist_scale if "wrist" in name else arm_scale
        track = values[:, column]
        if radius:
            track = np.convolve(
                np.pad(track, (radius, radius), mode="edge"),
                kernel,
                mode="valid",
            )
        centre = float(np.median(track))
        values[:, column] = centre + gain * (track - centre)
        affected.append(str(name))
    after = _upper_limb_motion_metrics(values, joint_names, fps=fps)
    return values, {
        "arm_swing_scale": arm_scale,
        "wrist_swing_scale": wrist_scale,
        "smoothing_sigma_frames": sigma,
        "affected_joint_names": affected,
        "before": before,
        "after": after,
    }


def _select_route_length_segment(
    root_xy: np.ndarray,
    root_quaternion_wxyz: np.ndarray,
    *,
    required_distance_m: float,
    minimum_frames: int = 120,
    minimum_start_frame: int = 0,
    preferred_start_frames: Sequence[int] | None = None,
) -> tuple[int, int, float]:
    """Pick a low-yaw-variation, monotonic, near-90-degree source span."""

    values = np.asarray(root_xy, dtype=np.float64)
    quaternions = np.asarray(root_quaternion_wxyz, dtype=np.float64)
    direction, travel_yaw = _robust_travel_direction(values)
    progress = (values - values[0]) @ direction
    yaw = _yaw_wxyz(quaternions)
    required = float(required_distance_m)
    if required <= 0.5:
        raise ValueError("target route is too short for a lateral stair pilot")
    candidates: list[tuple[tuple[float, ...], int, int, float]] = []
    first_start = max(0, int(minimum_start_frame))
    regular_starts = tuple(
        range(
            first_start,
            max(first_start + 1, len(values) - minimum_frames),
            5,
        )
    )
    preferred_starts = tuple(
        sorted(
            {
                int(value)
                for value in (preferred_start_frames or ())
                if first_start <= int(value) < len(values) - minimum_frames
            }
        )
    )
    starts = preferred_starts if preferred_starts else regular_starts
    for start in starts:
        relative_progress = progress - progress[start]
        reachable = np.flatnonzero(
            (np.arange(len(values)) >= start + minimum_frames)
            & (relative_progress >= required)
        )
        if not len(reachable):
            continue
        stop = int(reachable[0]) + 1
        local_progress = relative_progress[start:stop]
        backwards = float(np.sum(np.maximum(0.0, -np.diff(local_progress))))
        local_yaw = yaw[start:stop]
        facing_delta = _wrap_angle(travel_yaw - float(np.median(local_yaw)))
        side_error = abs(abs(math.degrees(facing_delta)) - 90.0)
        yaw_span = math.degrees(float(np.ptp(local_yaw)))
        rank = (
            side_error,
            8.0 * backwards,
            0.25 * yaw_span,
            float(start),
        )
        candidates.append((rank, start, stop, facing_delta))
    if not candidates and preferred_starts:
        return _select_route_length_segment(
            values,
            quaternions,
            required_distance_m=required,
            minimum_frames=minimum_frames,
            minimum_start_frame=first_start,
            preferred_start_frames=None,
        )
    if not candidates:
        raise ValueError("lateral source cannot cover the requested stair route")
    _rank, start, stop, facing_delta = min(candidates, key=lambda item: item[0])
    return start, stop, facing_delta


def _flat_support_mesh(
    root_xy: np.ndarray, *, height_m: float
) -> TerrainMeshIndex:
    centre = np.mean(np.asarray(root_xy, dtype=np.float64), axis=0)
    half_extent = 10.0
    vertices = np.asarray(
        (
            (centre[0] - half_extent, centre[1] - half_extent, height_m),
            (centre[0] + half_extent, centre[1] - half_extent, height_m),
            (centre[0] + half_extent, centre[1] + half_extent, height_m),
            (centre[0] - half_extent, centre[1] + half_extent, height_m),
        ),
        dtype=np.float32,
    )
    mesh = CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        valid_faces=np.ones(2, dtype=np.bool_),
        source_asset_sha256="0" * 64,
    )
    return TerrainMeshIndex(
        mesh,
        RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        ),
    )


def _load_aligned_lateral_source(
    source_path: Path,
    *,
    adapter: _G1FootfallAdapter,
    joint_names: Sequence[str],
    target_start_xy: np.ndarray,
    target_direction_xy: np.ndarray,
    target_distance_m: float,
    target_base_height_m: float,
    exit_margin_m: float,
    forced_start_frame: int | None,
    arm_swing_scale: float,
    wrist_swing_scale: float,
    arm_smoothing_sigma_frames: float,
) -> tuple[StitchedMotion, TerrainMeshIndex, dict[str, object]]:
    minimum_start_time_s = 0.0
    with np.load(source_path, allow_pickle=False) as payload:
        if {"fps", "body_pos_w", "body_quat_w", "joint_pos"}.issubset(
            payload.files
        ):
            fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
            root = np.asarray(payload["body_pos_w"][:, 0], dtype=np.float64)
            quaternion = np.asarray(
                payload["body_quat_w"][:, 0], dtype=np.float64
            )
            joints = np.asarray(payload["joint_pos"], dtype=np.float64)
        elif {
            "fps",
            "root_position_world",
            "root_quaternion_world_wxyz",
            "joint_position_mjcf",
        }.issubset(payload.files):
            fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
            root = np.asarray(
                payload["root_position_world"], dtype=np.float64
            )
            quaternion = np.asarray(
                payload["root_quaternion_world_wxyz"], dtype=np.float64
            )
            mjcf_joints = np.asarray(
                payload["joint_position_mjcf"], dtype=np.float64
            )
            mujoco = adapter._mujoco
            ordered = sorted(
                (
                    int(adapter.model.jnt_qposadr[joint_id]),
                    str(
                        mujoco.mj_id2name(
                            adapter.model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            joint_id,
                        )
                    ),
                )
                for joint_id in range(int(adapter.model.njnt))
                if int(adapter.model.jnt_type[joint_id])
                != int(mujoco.mjtJoint.mjJNT_FREE)
            )
            source_names = [name for _address, name in ordered]
            source_indices = {name: index for index, name in enumerate(source_names)}
            missing = [name for name in joint_names if name not in source_indices]
            if missing:
                raise ValueError(
                    "MotionBricks source is missing joints: "
                    + ", ".join(missing)
                )
            joints = mjcf_joints[
                :, [source_indices[name] for name in joint_names]
            ]
            if {
                "recorded_start_hold_frames",
                "recorded_start_ramp_frames",
            }.issubset(payload.files):
                minimum_start_time_s = (
                    float(np.asarray(payload["recorded_start_hold_frames"]).item())
                    + float(
                        np.asarray(payload["recorded_start_ramp_frames"]).item()
                    )
                ) / fps
        else:
            raise ValueError(
                "lateral source is neither a BONES nor MotionBricks recording"
            )
    if not math.isfinite(fps) or not 20.0 <= fps <= 120.0:
        raise ValueError("lateral stair source has an invalid frame rate")
    if not (len(root) == len(quaternion) == len(joints)):
        raise ValueError("BONES lateral source arrays have inconsistent lengths")
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("BONES lateral source has a zero quaternion")
    quaternion /= norms
    source_fps = fps
    if abs(fps - 50.0) > 1.0e-6:
        duration = (len(root) - 1) / fps
        frame_count = int(math.floor(duration * 50.0 + 1.0e-8)) + 1
        source_time = np.arange(len(root), dtype=np.float64) / fps
        target_time = np.minimum(
            np.arange(frame_count, dtype=np.float64) / 50.0,
            source_time[-1],
        )
        root = np.stack(
            [
                np.interp(target_time, source_time, root[:, axis])
                for axis in range(3)
            ],
            axis=1,
        )
        joints = np.stack(
            [
                np.interp(target_time, source_time, joints[:, axis])
                for axis in range(joints.shape[1])
            ],
            axis=1,
        )
        continuous_quaternion = quaternion.copy()
        for index in range(1, len(continuous_quaternion)):
            if (
                float(
                    np.dot(
                        continuous_quaternion[index - 1],
                        continuous_quaternion[index],
                    )
                )
                < 0.0
            ):
                continuous_quaternion[index] *= -1.0
        interpolator = Slerp(
            source_time,
            Rotation.from_quat(continuous_quaternion[:, (1, 2, 3, 0)]),
        )
        quaternion_xyzw = interpolator(target_time).as_quat()
        quaternion = quaternion_xyzw[:, (3, 0, 1, 2)]
        fps = 50.0
    joints, smoothed_joint_names = _smooth_discontinuous_upper_body_joints(
        joints, joint_names
    )
    joints, upper_limb_attenuation = _attenuate_upper_limb_motion(
        joints,
        joint_names,
        fps=fps,
        arm_swing_scale=arm_swing_scale,
        wrist_swing_scale=wrist_swing_scale,
        smoothing_sigma_frames=arm_smoothing_sigma_frames,
    )

    source_direction, source_travel_yaw = _robust_travel_direction(root[:, :2])
    target_direction = np.asarray(target_direction_xy, dtype=np.float64)
    target_direction /= np.linalg.norm(target_direction)
    target_yaw = math.atan2(float(target_direction[1]), float(target_direction[0]))
    required_distance = float(target_distance_m + exit_margin_m)
    # Keep the final 0.4 s of MotionBricks' smooth command ramp.  Beginning
    # exactly after the ramp happened to cut into the middle of the first
    # lateral swing, leaving only twelve 50-Hz frames to clear the riser.
    # The late ramp is already at near-steady speed and gives us the complete
    # lead-foot approach without inventing any kinematics.
    ramp_end_frame = int(math.ceil(minimum_start_time_s * fps))
    minimum_start_frame = max(0, ramp_end_frame - int(round(0.4 * fps)))
    late_ramp_starts = (
        (int(forced_start_frame),)
        if forced_start_frame is not None
        else tuple(range(minimum_start_frame, ramp_end_frame, 5))
    )
    start, stop, facing_delta = _select_route_length_segment(
        root[:, :2],
        quaternion,
        required_distance_m=required_distance,
        minimum_start_frame=minimum_start_frame,
        preferred_start_frames=late_ramp_starts,
    )
    root = root[start:stop].copy()
    quaternion = quaternion[start:stop].copy()
    joints = joints[start:stop].copy()

    rotation_yaw = target_yaw - source_travel_yaw
    cosine = math.cos(rotation_yaw)
    sine = math.sin(rotation_yaw)
    rotation_xy = np.asarray(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    root_xy = (
        (root[:, :2] - root[0, :2]) @ rotation_xy.T
        + np.asarray(target_start_xy, dtype=np.float64)
    )
    rotation_quaternion = np.asarray(
        (
            math.cos(0.5 * rotation_yaw),
            0.0,
            0.0,
            math.sin(0.5 * rotation_yaw),
        ),
        dtype=np.float64,
    )
    quaternion = np.asarray(
        [
            quaternion_multiply_wxyz(rotation_quaternion, value)
            for value in quaternion
        ],
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)

    # BONES locomotion is authored on z=0.  Preserve its pelvis bob and move
    # only the flat support plane to the target route's first level.
    root[:, 2] += float(target_base_height_m)
    root[:, :2] = root_xy
    source_id = (
        source_path.parent.name
        if source_path.name == "motion.npz"
        else source_path.stem
    )
    provenance = tuple(
        FrameProvenance(-1, frame, source_id)
        for frame in range(start, stop)
    )
    motion = StitchedMotion(
        fps=fps,
        root_position_world=np.asarray(root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternion, dtype=np.float32),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=provenance,
        seam_indices=(),
    )
    realized_progress = (root_xy - root_xy[0]) @ target_direction
    metadata = {
        "source_motion": str(source_path),
        "source_id": source_id,
        "source_fps": source_fps,
        "output_fps": fps,
        "smoothed_discontinuous_joint_names": list(smoothed_joint_names),
        "upper_limb_attenuation": upper_limb_attenuation,
        "source_start_frame": start,
        "source_stop_frame_exclusive": stop,
        "source_travel_yaw_rad": source_travel_yaw,
        "target_travel_yaw_rad": target_yaw,
        "body_to_travel_angle_deg": math.degrees(facing_delta),
        "realized_source_progress_m": float(np.ptp(realized_progress)),
    }
    return (
        motion,
        _flat_support_mesh(root_xy, height_m=target_base_height_m),
        metadata,
    )


def _mechanically_accepted(diagnostics: object) -> bool:
    return bool(
        diagnostics.maximum_joint_correction_rad
        <= MAXIMUM_SIDE_ON_JOINT_CORRECTION_RAD + 1.0e-6
        and diagnostics.maximum_stance_sole_target_error_m <= 0.015
        and diagnostics.maximum_swing_sole_target_error_m <= 0.050
        and diagnostics.maximum_stance_run_drift_m <= 0.012
        and diagnostics.minimum_stance_support_point_count >= 2
        and diagnostics.maximum_root_translation_step_m <= 0.035
        and diagnostics.maximum_root_rotation_step_rad <= 0.080
        and diagnostics.maximum_joint_step_rad <= 0.20
        and diagnostics.maximum_root_acceleration_m_s2 <= 30.0
    )


def _repairable_warp_accepted(
    diagnostics: object,
    extras: dict[str, np.ndarray],
) -> tuple[bool, dict[str, float | int]]:
    """Allow a mostly valid warp to reach the strict sole-refit pipeline.

    ``warp_motion`` reports its metrics before the later pelvis-reach solve,
    exact sole refit, retiming, collision repair, and no-hover audit.  A single
    takeoff/landing boundary can therefore make its maximum stance error or
    support count fail even when the rest of the gait is well conditioned.
    This is only an entrance gate for those repairs; final admission below
    keeps the original hard per-frame collision, support, drift, and motion
    limits.
    """

    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    error = np.asarray(
        extras.get("per_frame_sole_target_error_by_foot_m"),
        dtype=np.float64,
    )
    support = np.asarray(
        extras.get("target_stance_support_point_count"),
        dtype=np.int64,
    )
    if (
        stance.ndim != 2
        or stance.shape[1] != 2
        or error.shape != stance.shape
        or support.shape != stance.shape
        or not np.isfinite(error).all()
        or not np.any(stance)
    ):
        return False, {"invalid_repair_arrays": 1}
    stance_error = error[stance]
    swing_error = error[~stance]
    unsupported_stance_count = int(np.count_nonzero(stance & (support < 2)))
    metrics: dict[str, float | int] = {
        "stance_error_p95_m": float(np.quantile(stance_error, 0.95)),
        "stance_error_maximum_m": float(np.max(stance_error)),
        "swing_error_p95_m": (
            0.0 if not len(swing_error) else float(np.quantile(swing_error, 0.95))
        ),
        "swing_error_maximum_m": (
            0.0 if not len(swing_error) else float(np.max(swing_error))
        ),
        "unsupported_stance_frame_foot_count": unsupported_stance_count,
        "stance_frame_foot_count": int(np.count_nonzero(stance)),
    }
    accepted = bool(
        diagnostics.maximum_joint_correction_rad
        <= MAXIMUM_SIDE_ON_JOINT_CORRECTION_RAD + 1.0e-6
        and diagnostics.maximum_joint_step_rad <= 0.20 + 1.0e-9
        and diagnostics.maximum_root_translation_step_m <= 0.050
        and diagnostics.maximum_root_rotation_step_rad <= 0.080
        and diagnostics.maximum_root_acceleration_m_s2 <= 75.0
        and float(metrics["stance_error_p95_m"]) <= 0.030
        and float(metrics["stance_error_maximum_m"]) <= 0.140
        and float(metrics["swing_error_p95_m"]) <= 0.060
        and float(metrics["swing_error_maximum_m"]) <= 0.280
        and unsupported_stance_count <= max(
            1, int(math.ceil(0.005 * np.count_nonzero(stance)))
        )
    )
    return accepted, metrics


def _rotation_between_normals(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(target, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if sine <= 1.0e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        raise ValueError("sole and terrain normals are antiparallel")
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _full_sole_stance_mask(
    stance: object,
    *,
    boundary_frames: int = 2,
) -> np.ndarray:
    """Separate rigid plants from valid heel/toe contact transitions.

    Contact labels include the first touchdown and final takeoff frames.  A
    natural gait can still be rolling through the heel or toe there, so
    forcing all four sole probes onto one tread at those frames creates an
    artificial ankle snap and can make an otherwise reachable plant appear
    impossible.  The interior remains a hard whole-sole plant; short runs
    retain at least one interior frame.
    """

    authored = np.asarray(stance, dtype=bool)
    if authored.ndim != 2 or authored.shape[1] != 2:
        raise ValueError("stance mask must have shape [frames, 2]")
    if int(boundary_frames) < 0:
        raise ValueError("boundary frame count must be nonnegative")
    core = np.zeros_like(authored)
    for span in _stance_spans(authored):
        length = int(span.stop_frame) - int(span.start_frame)
        trim = min(int(boundary_frames), max(0, (length - 1) // 2))
        core[
            int(span.start_frame) + trim : int(span.stop_frame) - trim,
            int(span.foot_index),
        ] = True
    return core


def _clamp_bracketed_swing_overshoot(
    sole_targets_world: object,
    full_sole_stance: object,
) -> tuple[np.ndarray, dict[str, float]]:
    """Keep each swing between its preceding and following footholds.

    Blending two planted-foot offsets with an independently moving source
    swing can make the target pass beyond its next tread and then reverse into
    contact.  On stairs that small planar overshoot intersects the following
    riser even when both endpoint footholds are valid.  Preserve the source
    height and sole articulation, but project its planar centre monotonically
    along the segment joining the two actual footholds.
    """

    targets = np.asarray(sole_targets_world, dtype=np.float64).copy()
    stance = np.asarray(full_sole_stance, dtype=bool)
    if targets.ndim != 4 or targets.shape[:2] != stance.shape:
        raise ValueError("swing clamp requires sole targets and stance [T,2]")
    maximum_correction = 0.0
    corrected_frame_foot_count = 0
    for foot in range(2):
        spans = sorted(
            (span for span in _stance_spans(stance) if span.foot_index == foot),
            key=lambda span: span.start_frame,
        )
        for previous, following in zip(spans, spans[1:]):
            start = int(previous.stop_frame)
            stop = int(following.start_frame)
            if stop <= start:
                continue
            previous_centre = np.mean(
                targets[int(previous.stop_frame) - 1, foot], axis=0
            )
            following_centre = np.mean(
                targets[int(following.start_frame), foot], axis=0
            )
            delta = following_centre[:2] - previous_centre[:2]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-6:
                continue
            direction = delta / distance
            centres = np.mean(targets[start:stop, foot], axis=1)
            raw = (centres[:, :2] - previous_centre[:2]) @ direction
            monotone = np.maximum.accumulate(np.clip(raw, 0.0, distance))
            correction = (monotone - raw)[:, None] * direction[None]
            magnitude = np.linalg.norm(correction, axis=1)
            active = magnitude > 1.0e-8
            targets[start:stop, foot, :, :2] += correction[:, None, :]
            maximum_correction = max(
                maximum_correction,
                float(np.max(magnitude)) if len(magnitude) else 0.0,
            )
            corrected_frame_foot_count += int(np.count_nonzero(active))
    return targets, {
        "maximum_planar_swing_overshoot_correction_m": maximum_correction,
        "corrected_swing_frame_foot_count": float(corrected_frame_foot_count),
    }


def _conform_stance_sole_targets_to_mesh(
    extras: dict[str, np.ndarray],
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: TerrainMeshIndex,
    ground_fallback_height_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Align each planted rigid sole with its exact support surface."""

    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    targets = np.asarray(
        extras.get("target_sole_points_world"), dtype=np.float64
    )
    if (
        stance.ndim != 2
        or stance.shape[1] != 2
        or targets.ndim != 4
        or targets.shape[:2] != stance.shape
    ):
        raise ValueError("sole conformance requires stance and full sole targets")
    radii = adapter.sole_sphere_radii()
    query = CanonicalMeshQuery(
        target_mesh.mesh, target_mesh.world_from_terrain
    )
    ray_z = float(np.max(target_mesh.vertices_world[:, 2]) + 1.0)
    full_sole_stance = _full_sole_stance_mask(stance)
    anchored = targets.copy()
    assigned = np.zeros(stance.shape, dtype=bool)
    rotations: list[float] = []
    translations: list[float] = []
    ground_fallback_runs = 0
    partially_supported_runs = 0
    for span in _stance_spans(full_sole_stance):
        foot = int(span.foot_index)
        frame = (int(span.start_frame) + int(span.stop_frame) - 1) // 2
        points = np.asarray(targets[frame, foot], dtype=np.float64)
        centre = np.mean(points, axis=0)
        _u, _singular, basis = np.linalg.svd(points - centre)
        sole_normal = np.asarray(basis[-1], dtype=np.float64)
        if sole_normal[2] < 0.0:
            sole_normal *= -1.0
        surface = query.query(
            np.asarray(((centre[0], centre[1], ray_z),), dtype=np.float64)
        )
        centre_has_surface = bool(
            np.isfinite(surface.downward_ray_distance_m[0])
            and int(surface.downward_ray_face_index[0]) >= 0
        )
        if centre_has_surface:
            terrain_normal = np.asarray(
                surface.downward_ray_normal_world[0], dtype=np.float64
            )
        else:
            # The registered stair mesh is finite and deliberately excludes the
            # surrounding flat runway.  A stance whose centre has no downward
            # hit is therefore on that runway, not on the nearest stair riser.
            terrain_normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            ground_fallback_runs += 1
        if terrain_normal[2] < 0.0:
            terrain_normal *= -1.0
        rotation = _rotation_between_normals(sole_normal, terrain_normal)
        conformed = (points - centre) @ rotation.T + centre
        heights = np.asarray(
            [
                _terrain_height(target_mesh, point[:2], ray_z)
                for point in conformed
            ],
            dtype=np.float64,
        )
        supported = np.isfinite(heights)
        support_count = int(np.count_nonzero(supported))
        if not centre_has_surface:
            heights[:] = float(ground_fallback_height_m)
            supported[:] = True
        elif support_count < len(heights):
            # Do not invent collision support outside the exact mesh.  Missing
            # edge probes are ignored here and remain visible to the later
            # >=3-probe/core-hover admission gate.
            partially_supported_runs += 1
        if not np.any(supported):
            raise ValueError("stance centre hit terrain but no sole probe did")
        vertical_shift = float(
            np.max(
                heights[supported]
                + np.asarray(radii[foot])[supported]
                - conformed[supported, 2]
            )
        )
        conformed[:, 2] += vertical_shift
        anchored[span.start_frame : span.stop_frame, foot] = conformed
        assigned[span.start_frame : span.stop_frame, foot] = True
        rotations.append(
            math.degrees(
                math.acos(
                    float(np.clip(np.dot(sole_normal, terrain_normal), -1.0, 1.0))
                )
            )
        )
        translations.append(abs(vertical_shift))
    if not np.any(assigned):
        raise ValueError("sole conformance found no stance run")
    updated = dict(extras)
    updated["pre_conformance_target_sole_points_world"] = np.asarray(
        targets, dtype=np.float32
    )
    blended = _blend_anchored_sole_offsets(targets, anchored, assigned)
    updated["target_sole_points_world"] = np.asarray(blended, dtype=np.float32)
    updated["stance_sole_surface_conformance_mask"] = assigned
    updated["full_sole_stance_mask"] = full_sole_stance
    updated["partial_contact_stance_mask"] = stance & ~full_sole_stance
    return updated, {
        "stance_run_count": float(len(rotations)),
        "ground_fallback_run_count": float(ground_fallback_runs),
        "partially_supported_run_count": float(partially_supported_runs),
        "maximum_surface_alignment_deg": float(max(rotations)),
        "maximum_vertical_translation_m": float(max(translations)),
    }


def _lateral_stance_level_schedule(
    spans: Sequence[object],
    *,
    leading_foot_index: int,
    level_count: int,
) -> dict[tuple[int, int, int], int]:
    """Assign lead-up/catch-up support phases to successive stair treads."""

    leading_foot = int(leading_foot_index)
    if leading_foot not in (0, 1):
        raise ValueError("leading foot index must be zero or one")
    if int(level_count) < 2:
        raise ValueError("lateral stair schedule needs at least two levels")
    chronological = sorted(
        spans,
        key=lambda value: (
            value.start_frame,
            value.foot_index,
            value.stop_frame,
        ),
    )
    current_level = 0
    final_level = int(level_count) - 1
    result: dict[tuple[int, int, int], int] = {}
    latest_level_by_foot = [-1, -1]
    for span in chronological:
        # A cropped lateral gait can begin with either foot's stance span.
        # Seed both feet on the base tread before allowing the lead foot to
        # advance.  Advancing the first lead-foot span merely because another
        # span had already been seen skipped its base support, leaving no real
        # lead-foot swing from level 0 to level 1 and causing the later route
        # offset solve to jump/glide at the first stair.
        if (
            span.foot_index == leading_foot
            and latest_level_by_foot[leading_foot] == current_level
            and latest_level_by_foot[1 - leading_foot] == current_level
            and current_level < final_level
        ):
            current_level += 1
        key = (span.foot_index, span.start_frame, span.stop_frame)
        result[key] = current_level
        latest_level_by_foot[span.foot_index] = current_level
    if latest_level_by_foot != [final_level, final_level]:
        raise ValueError(
            "lateral source segment does not finish with both feet on the top tread"
        )
    return result


def _support_conditioned_route_offsets(
    source: StitchedMotion,
    *,
    route: object,
    adapter: _G1FootfallAdapter,
    smoothing_frames: float,
    leading_foot_index: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int, int], int],
    int,
]:
    """Fit lateral footfalls to ordered treads and scaffold pelvis height."""

    start = np.asarray(route.start_xy, dtype=np.float64)
    direction = np.asarray(route.end_xy, dtype=np.float64) - start
    direction /= np.linalg.norm(direction)
    roots = np.asarray(source.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        source.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(source.joint_position, dtype=np.float64)
    radii = adapter.sole_sphere_radii()
    support_points = np.empty(
        (len(roots), 2, len(radii[0]), 3), dtype=np.float64
    )
    centres = np.empty((len(roots), 2, 3), dtype=np.float64)
    support_count = np.zeros((len(roots), 2), dtype=np.int16)
    base_height = float(route.levels[0].height_m)
    for frame in range(len(roots)):
        frame_feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        for foot in range(2):
            centres[frame, foot] = np.mean(frame_feet[foot], axis=0)
            support = frame_feet[foot].copy()
            support[:, 2] -= radii[foot]
            support_points[frame, foot] = support
            support_count[frame, foot] = int(
                np.count_nonzero(
                    np.abs(support[:, 2] - base_height) <= 0.020
                )
            )
    speed = np.linalg.norm(
        np.gradient(centres, axis=0) * float(source.fps), axis=2
    )
    stance = _remove_short_stance_runs(
        _fill_short_stance_gaps(
            (support_count >= 2) & (speed <= 0.25),
            maximum_gap_frames=4,
        ),
        minimum_run_frames=4,
    )
    spans = sorted(
        _stance_spans(stance),
        key=lambda value: (
            value.start_frame + value.stop_frame,
            value.foot_index,
        ),
    )
    if not spans:
        raise ValueError("lateral source contains no planted stance phase")
    target_level_by_span = _lateral_stance_level_schedule(
        spans,
        leading_foot_index=leading_foot_index,
        level_count=len(route.levels),
    )
    final_level = len(route.levels) - 1
    top_lead = min(
        (
            span
            for span in spans
            if span.foot_index == int(leading_foot_index)
            and target_level_by_span[
                (span.foot_index, span.start_frame, span.stop_frame)
            ]
            == final_level
        ),
        key=lambda span: span.start_frame,
    )
    top_catch = min(
        (
            span
            for span in spans
            if span.foot_index != int(leading_foot_index)
            and span.start_frame >= top_lead.start_frame
            and target_level_by_span[
                (span.foot_index, span.start_frame, span.stop_frame)
            ]
            == final_level
        ),
        key=lambda span: span.start_frame,
    )
    next_lead = [
        span
        for span in spans
        if span.foot_index == int(leading_foot_index)
        and span.start_frame > top_catch.start_frame
    ]
    suggested_stop_frame = (
        min(span.start_frame for span in next_lead)
        if next_lead
        else len(roots)
    )

    progress_shift_by_span: dict[tuple[int, int, int], float] = {}
    spans_by_level: dict[int, list[object]] = {
        index: [] for index in range(len(route.levels))
    }
    for span in spans:
        frame = (span.start_frame + span.stop_frame - 1) // 2
        coordinate = float((centres[frame, span.foot_index, :2] - start) @ direction)
        # A lateral stair shuffle has a directional lead foot: that foot moves
        # up one tread, then the trailing foot catches onto the same tread.
        # Nearest-tread assignment instead makes the two feet alternately race
        # ahead, which demands an impossible pelvis jump at each hand-off.
        key = (span.foot_index, span.start_frame, span.stop_frame)
        level_index = target_level_by_span[key]
        level = route.levels[level_index]
        footprint_progress = (
            support_points[frame, span.foot_index, :, :2] - start
        ) @ direction
        edge_margin = max(
            0.08, 0.5 * float(np.ptp(footprint_progress)) + 0.03
        )
        level_start = float(level.route_start_distance_m)
        level_stop = float(level.route_stop_distance_m)
        if level_index == 0:
            # The base level often includes a long approach apron.  Plant
            # beside the first riser instead of at that apron's midpoint.
            # A lateral lead swing also needs enough horizontal run-up for
            # the shin and complete foot hull to clear the vertical face.
            desired_coordinate = level_stop - max(edge_margin, 0.18)
        elif level_index == len(route.levels) - 1:
            # Likewise, enter the top tread near its front edge rather than
            # compressing the entire gait to the end of a long platform.
            desired_coordinate = level_start + edge_margin
        else:
            desired_coordinate = 0.5 * (level_start + level_stop)
        desired_coordinate = float(
            np.clip(desired_coordinate, level_start, level_stop)
        )
        progress_shift = desired_coordinate - coordinate
        progress_shift_by_span[key] = progress_shift
        spans_by_level[level_index].append(span)

    missing_levels = [
        level for level, values in spans_by_level.items() if not values
    ]
    if missing_levels:
        raise ValueError(
            "lateral stance schedule has no support phase for tread levels "
            + ", ".join(str(value) for value in missing_levels)
        )
    level_progress = np.asarray(
        [
            np.median(
                [
                    progress_shift_by_span[
                        (span.foot_index, span.start_frame, span.stop_frame)
                    ]
                    for span in spans_by_level[level]
                ]
            )
            for level in range(len(route.levels))
        ],
        dtype=np.float64,
    )
    level_height = np.asarray(
        [float(level.height_m) - base_height for level in route.levels],
        dtype=np.float64,
    )
    height_offset = np.full(len(roots), level_height[0], dtype=np.float64)
    progress_offset = np.full(
        len(roots), level_progress[0], dtype=np.float64
    )
    trailing_foot = 1 - int(leading_foot_index)
    ramp_frames = max(0, int(round(float(smoothing_frames))))
    for level_index in range(1, len(route.levels)):
        lead_candidates = [
            span
            for span in spans_by_level[level_index]
            if span.foot_index == int(leading_foot_index)
        ]
        if not lead_candidates:
            raise ValueError(
                f"tread level {level_index} has no leading-foot support"
            )
        lead = min(lead_candidates, key=lambda span: span.start_frame)
        catch_candidates = [
            span
            for span in spans_by_level[level_index]
            if span.foot_index == trailing_foot
            and span.start_frame >= lead.start_frame
        ]
        if not catch_candidates:
            raise ValueError(
                f"tread level {level_index} has no trailing-foot catch support"
            )
        catch = min(catch_candidates, key=lambda span: span.start_frame)
        previous_lead_candidates = [
            span
            for span in spans_by_level[level_index - 1]
            if span.foot_index == int(leading_foot_index)
            and span.stop_frame <= lead.start_frame
        ]
        if not previous_lead_candidates:
            raise ValueError(
                f"tread level {level_index} has no preceding lead-foot support"
            )
        previous_lead = max(
            previous_lead_candidates, key=lambda span: span.stop_frame
        )

        # Apply the inter-tread displacement while the lead foot is actually
        # in swing.  The earlier implementation began this ramp at lead-foot
        # touchdown and finished near the trailing-foot catch.  That asks an
        # already planted lead leg to absorb the difference between the flat
        # gait's stride and the target tread spacing.  It happened to work for
        # two near-matched stair geometries, but produced 10+ cm stance errors
        # and root impulses on longer or shorter treads.  Ending at touchdown
        # makes the root and landing sole arrive at the new tread together.
        swing_start = max(0, int(previous_lead.stop_frame) - 1)
        swing_stop = int(lead.start_frame)
        if swing_stop <= swing_start:
            raise ValueError(
                f"tread level {level_index} has no lead-foot swing interval"
            )
        previous_height = level_height[level_index - 1]
        previous_progress = level_progress[level_index - 1]

        # Horizontal stride correction must be complete when the lead sole
        # lands.  Distribute it across the whole lead-foot swing.
        progress_start = swing_stop if ramp_frames == 0 else swing_start
        progress_coordinate = np.arange(
            progress_start, swing_stop + 1, dtype=np.float64
        )
        if swing_stop == progress_start:
            progress_smooth = np.ones_like(progress_coordinate)
        else:
            progress_linear = (progress_coordinate - progress_start) / float(
                swing_stop - progress_start
            )
            progress_smooth = progress_linear * progress_linear * (
                3.0 - 2.0 * progress_linear
            )
        progress_offset[progress_start : swing_stop + 1] = (
            (1.0 - progress_smooth) * previous_progress
            + progress_smooth * level_progress[level_index]
        )
        progress_offset[swing_stop + 1 :] = level_progress[level_index]

        # Pelvis height follows the weight transfer from the newly planted
        # lead sole to the trailing-foot catch.  Raising it during the lead
        # swing made the old support leg overextend on double-support frames.
        height_stop = max(lead.start_frame + 1, catch.start_frame)
        height_start = (
            lead.start_frame
            if ramp_frames == 0
            else max(lead.start_frame, height_stop - ramp_frames)
        )
        height_coordinate = np.arange(
            height_start, height_stop + 1, dtype=np.float64
        )
        height_linear = (height_coordinate - height_start) / float(
            height_stop - height_start
        )
        height_smooth = height_linear * height_linear * (
            3.0 - 2.0 * height_linear
        )
        height_offset[height_start : height_stop + 1] = (
            (1.0 - height_smooth) * previous_height
            + height_smooth * level_height[level_index]
        )
        height_offset[height_stop + 1 :] = level_height[level_index]
    return (
        height_offset,
        progress_offset,
        target_level_by_span,
        suggested_stop_frame,
    )


def _truncate_motion(motion: StitchedMotion, stop_frame: int) -> StitchedMotion:
    stop = int(stop_frame)
    if stop < 2 or stop > len(motion.root_position_world):
        raise ValueError("lateral motion trim lies outside the source")
    return StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(motion.root_position_world[:stop]),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz[:stop]
        ),
        joint_position=np.asarray(motion.joint_position[:stop]),
        provenance=motion.provenance[:stop],
        seam_indices=tuple(
            index for index in motion.seam_indices if index < stop
        ),
    )


def _retime_motion_and_extras(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    temporal_scale: float,
) -> tuple[StitchedMotion, dict[str, np.ndarray]]:
    """Slow an accepted motion without changing its spatial trajectory."""

    scale = float(temporal_scale)
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError("lateral temporal scale must be finite and at least one")
    if scale == 1.0:
        return motion, extras
    old_count = len(motion.root_position_world)
    new_count = int(round((old_count - 1) * scale)) + 1
    old_coordinate = np.arange(old_count, dtype=np.float64)
    new_coordinate = np.linspace(
        0.0, float(old_count - 1), new_count, dtype=np.float64
    )

    def interpolate(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != old_count:
            return array.copy()
        if array.dtype.kind not in "fc":
            nearest = np.clip(
                np.rint(new_coordinate).astype(np.int64), 0, old_count - 1
            )
            return array[nearest]
        flat = array.reshape(old_count, -1)
        result = np.stack(
            [
                np.interp(new_coordinate, old_coordinate, flat[:, column])
                for column in range(flat.shape[1])
            ],
            axis=1,
        ).reshape((new_count,) + array.shape[1:])
        return np.asarray(result, dtype=array.dtype)

    roots = interpolate(np.asarray(motion.root_position_world))
    joints = interpolate(np.asarray(motion.joint_position))
    quaternion_xyzw = Slerp(
        old_coordinate,
        Rotation.from_quat(
            np.asarray(motion.root_quaternion_world_wxyz)[:, (1, 2, 3, 0)]
        ),
    )(new_coordinate).as_quat()
    quaternions = quaternion_xyzw[:, (3, 0, 1, 2)]
    nearest = np.clip(
        np.rint(new_coordinate).astype(np.int64), 0, old_count - 1
    )
    retimed = StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(roots, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternions, dtype=np.float32),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=tuple(motion.provenance[index] for index in nearest),
        seam_indices=tuple(
            int(round(index * scale)) for index in motion.seam_indices
        ),
    )
    retimed_extras = {
        key: interpolate(value) for key, value in extras.items()
    }
    if "authored_stance_mask" in extras:
        original_stance = np.asarray(extras["authored_stance_mask"], dtype=bool)
        lower = np.floor(new_coordinate).astype(np.int64)
        upper = np.ceil(new_coordinate).astype(np.int64)
        retimed_extras["authored_stance_mask"] = (
            original_stance[lower] & original_stance[upper]
        )
    return retimed, retimed_extras


def _retimed_motion_metrics(motion: StitchedMotion) -> dict[str, float]:
    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    dot = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    acceleration = np.diff(roots, n=2, axis=0) * float(motion.fps) ** 2
    return {
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(joints, axis=0)))),
        "maximum_root_translation_step_m": float(
            np.max(np.linalg.norm(np.diff(roots, axis=0), axis=1))
        ),
        "maximum_root_rotation_step_rad": float(
            np.max(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
        ),
        "maximum_root_acceleration_m_s2": float(
            np.max(np.linalg.norm(acceleration, axis=1))
        ),
    }


def _adaptively_subdivide_motion_steps(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    target_joint_step_rad: float = 0.14,
) -> tuple[StitchedMotion, dict[str, np.ndarray], dict[str, float]]:
    """Insert time only around large IK changes instead of slowing the clip."""

    old_count = len(motion.root_position_world)
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    if old_count < 2:
        return motion, extras, {"added_frame_count": 0.0}
    interval_step = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    subdivisions = np.maximum(
        1,
        np.ceil(interval_step / float(target_joint_step_rad)).astype(np.int64),
    )
    if int(np.max(subdivisions)) == 1:
        return motion, extras, {
            "added_frame_count": 0.0,
            "maximum_interval_subdivision": 1.0,
        }
    coordinates: list[float] = []
    for frame, count in enumerate(subdivisions):
        coordinates.extend(
            float(frame) + float(value) / float(count)
            for value in range(int(count))
        )
    coordinates.append(float(old_count - 1))
    new_coordinate = np.asarray(coordinates, dtype=np.float64)
    old_coordinate = np.arange(old_count, dtype=np.float64)

    def interpolate(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != old_count:
            return array.copy()
        if array.dtype.kind not in "fc":
            nearest = np.clip(
                np.rint(new_coordinate).astype(np.int64), 0, old_count - 1
            )
            return array[nearest]
        flat = array.reshape(old_count, -1)
        return np.asarray(
            np.stack(
                [
                    np.interp(new_coordinate, old_coordinate, flat[:, column])
                    for column in range(flat.shape[1])
                ],
                axis=1,
            ).reshape((len(new_coordinate),) + array.shape[1:]),
            dtype=array.dtype,
        )

    quaternion_xyzw = Slerp(
        old_coordinate,
        Rotation.from_quat(
            np.asarray(motion.root_quaternion_world_wxyz)[:, (1, 2, 3, 0)]
        ),
    )(new_coordinate).as_quat()
    nearest = np.clip(
        np.rint(new_coordinate).astype(np.int64), 0, old_count - 1
    )
    subdivided = StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(
            interpolate(np.asarray(motion.root_position_world)),
            dtype=np.float32,
        ),
        root_quaternion_world_wxyz=np.asarray(
            quaternion_xyzw[:, (3, 0, 1, 2)], dtype=np.float32
        ),
        joint_position=np.asarray(interpolate(joints), dtype=np.float32),
        provenance=tuple(motion.provenance[index] for index in nearest),
        seam_indices=tuple(
            int(np.argmin(np.abs(new_coordinate - float(index))))
            for index in motion.seam_indices
        ),
    )
    subdivided_extras = {
        key: interpolate(value) for key, value in extras.items()
    }
    if "authored_stance_mask" in extras:
        original_stance = np.asarray(extras["authored_stance_mask"], dtype=bool)
        lower = np.floor(new_coordinate).astype(np.int64)
        upper = np.ceil(new_coordinate).astype(np.int64)
        subdivided_extras["authored_stance_mask"] = (
            original_stance[lower] & original_stance[upper]
        )
    return subdivided, subdivided_extras, {
        "added_frame_count": float(len(new_coordinate) - old_count),
        "maximum_interval_subdivision": float(np.max(subdivisions)),
        "pre_subdivision_maximum_joint_step_rad": float(
            np.max(interval_step)
        ),
    }


def _refit_motion_to_sole_targets(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    adapter: _G1FootfallAdapter,
    stance_only: bool = False,
) -> tuple[StitchedMotion, dict[str, np.ndarray], dict[str, float]]:
    """Replant both soles after a pelvis edit or temporal interpolation.

    Collision clearance is sometimes obtained by moving the pelvis a few
    millimetres.  Moving only the free root also moves every planted foot and
    creates a visually subtle hover.  The warp already stores the exact four
    sole-sphere targets, so solve the legs back to those world-space targets
    while retaining the edited pelvis trajectory.
    """

    targets = np.asarray(extras.get("target_sole_points_world")).copy()
    frame_count = len(motion.root_position_world)
    if targets.ndim != 4 or targets.shape[:2] != (frame_count, 2):
        raise ValueError(
            "sole-preserving refit requires target_sole_points_world [T,2,P,3]"
        )
    if targets.shape[-1] != 3 or not np.isfinite(targets).all():
        raise ValueError("sole targets must be finite three-dimensional points")
    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    has_stance_schedule = stance.shape == (frame_count, 2)
    constraint_stance = np.asarray(
        extras.get("full_sole_stance_mask", stance), dtype=bool
    )
    if has_stance_schedule and constraint_stance.shape != stance.shape:
        raise ValueError("full-sole stance mask must match authored stance")
    if stance_only and not has_stance_schedule:
        raise ValueError("stance-only refit requires authored_stance_mask [T,2]")

    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    authored = np.asarray(motion.joint_position, dtype=np.float64)
    joints = authored.copy()
    errors_by_foot = np.zeros((frame_count, 2), dtype=np.float64)
    centres = np.zeros((frame_count, 2, 3), dtype=np.float64)
    correction = np.zeros(frame_count, dtype=np.float64)

    for frame in range(frame_count):
        authored_feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=authored[frame],
        )
        if stance_only:
            # Terrain correspondence is a hard constraint only while a foot
            # is planted.  Chasing a warped swing target can force an IK branch
            # jump even though that free leg's authored gait pose was already
            # smooth.  Preserve the actual swing geometry here; collision
            # repair remains responsible for lifting a swing that clips.
            for foot in range(2):
                if not constraint_stance[frame, foot]:
                    targets[frame, foot] = authored_feet[foot]
        authored_error = max(
            float(
                np.max(
                    np.linalg.norm(
                        authored_feet[foot] - targets[frame, foot], axis=1
                    )
                )
            )
            for foot in range(2)
        )
        # The input motion was already fitted once.  Keep it as a candidate so
        # a local IK branch switch cannot replace a modest swing-target error
        # with a discontinuous, saturated leg pose.  The solved candidate is
        # still selected whenever it materially improves the sole target.
        candidates = [
            (authored[frame].copy(), 0.0, authored_error),
            adapter.adapt_to_targets(
                root_position=roots[frame],
                root_quaternion_wxyz=quaternions[frame],
                authored_joints=authored[frame],
                sole_targets_world=targets[frame],
                initial_joints=(joints[frame - 1] if frame else None),
            )
        ]
        if frame:
            candidates.append(
                adapter.adapt_to_targets(
                    root_position=roots[frame],
                    root_quaternion_wxyz=quaternions[frame],
                    authored_joints=authored[frame],
                    sole_targets_world=targets[frame],
                    initial_joints=None,
                )
            )

        def _score_geometrically(
            candidate: tuple[np.ndarray, float, float],
        ) -> tuple[np.ndarray, float, float]:
            candidate_joints, candidate_correction, _candidate_error = candidate
            candidate_feet = adapter.sole_positions_for_pose(
                root_position=roots[frame],
                root_quaternion_wxyz=quaternions[frame],
                joints=candidate_joints,
            )
            geometric_error = max(
                float(
                    np.max(
                        np.linalg.norm(
                            candidate_feet[foot] - targets[frame, foot], axis=1
                        )
                    )
                )
                for foot in range(2)
            )
            return candidate_joints, candidate_correction, geometric_error

        # Most planted frames are already solved by the authored/continuation
        # starts.  Score those first and reserve the relatively expensive
        # future-pose starts for the genuinely unresolved near-singular frames.
        geometrically_scored = [
            _score_geometrically(candidate) for candidate in candidates
        ]
        if (
            has_stance_schedule
            and np.any(constraint_stance[frame])
            and min(float(value[2]) for value in geometrically_scored) > 0.008
            and frame + 1 < frame_count
        ):
            # A planted leg can sit at a near-singular straight-knee pose
            # where both the authored and previous-frame starts give the IK
            # zero useful descent direction.  Nearby future gait poses remain
            # on the same motion branch and provide a bounded continuation
            # seed.  Evaluate these only for genuinely bad stance frames so
            # the common path stays inexpensive.
            future_frames = sorted(
                {
                    min(frame_count - 1, frame + offset)
                    for offset in (1, 4, 8)
                }
            )
            for future in future_frames:
                geometrically_scored.append(
                    _score_geometrically(
                        adapter.adapt_to_targets(
                            root_position=roots[frame],
                            root_quaternion_wxyz=quaternions[frame],
                            authored_joints=authored[frame],
                            sole_targets_world=targets[frame],
                            initial_joints=authored[future],
                        )
                    )
                )
        # The adapter's scalar objective includes regularization and is not
        # guaranteed to equal the complete four-probe error used by the final
        # no-hover audit.  Rank every candidate using that exact geometric
        # error so a low internal objective cannot win with a lifted sole edge.
        candidates = geometrically_scored
        best_error = min(float(value[2]) for value in candidates)
        near_best = [
            value for value in candidates if float(value[2]) <= best_error + 0.001
        ]
        selected = min(
            near_best,
            key=lambda value: (
                (
                    float(np.max(np.abs(value[0] - joints[frame - 1])))
                    if frame
                    else float(np.max(np.abs(value[0] - authored[frame])))
                ),
                float(value[2]),
                float(value[1]),
            ),
        )
        joints[frame] = np.asarray(selected[0], dtype=np.float64)
        correction[frame] = float(
            np.max(np.abs(joints[frame] - authored[frame]))
        )
        feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        for foot in range(2):
            centres[frame, foot] = np.mean(feet[foot], axis=0)
            errors_by_foot[frame, foot] = float(
                np.max(np.linalg.norm(feet[foot] - targets[frame, foot], axis=1))
            )

    refitted = StitchedMotion(
        fps=motion.fps,
        root_position_world=motion.root_position_world,
        root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=motion.provenance,
        seam_indices=motion.seam_indices,
    )
    updated = dict(extras)
    updated["target_sole_points_world"] = np.asarray(
        targets, dtype=np.float32
    )
    updated["target_sole_center_world"] = np.asarray(
        np.mean(targets, axis=2), dtype=np.float32
    )
    updated["adapted_sole_center_world"] = np.asarray(centres, dtype=np.float32)
    updated["per_frame_sole_target_error_by_foot_m"] = np.asarray(
        errors_by_foot, dtype=np.float32
    )
    updated["per_frame_sole_target_error_m"] = np.asarray(
        np.max(errors_by_foot, axis=1), dtype=np.float32
    )
    updated["clearance_refit_joint_delta_rad"] = np.asarray(
        correction, dtype=np.float32
    )
    metrics = {
        "maximum_sole_target_error_m": float(np.max(errors_by_foot)),
        "maximum_joint_delta_rad": float(np.max(correction)),
        "maximum_joint_step_rad": float(
            np.max(np.abs(np.diff(joints, axis=0))) if frame_count > 1 else 0.0
        ),
    }
    return refitted, updated, metrics


def _audit_stance_contact_support(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: TerrainMeshIndex,
    ground_fallback_height_m: float,
    support_tolerance_m: float = 0.010,
    maximum_stance_target_error_m: float = 0.012,
    maximum_stance_drift_m: float = 0.012,
    maximum_core_stance_probe_hover_m: float = 0.012,
) -> dict[str, object]:
    """Reject collision-free clips whose authored stance feet are hovering."""

    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    targets = np.asarray(extras.get("target_sole_points_world"), dtype=np.float64)
    frame_count = len(motion.root_position_world)
    if stance.shape != (frame_count, 2):
        raise ValueError("contact audit requires authored_stance_mask [T,2]")
    if targets.ndim != 4 or targets.shape[:2] != (frame_count, 2):
        raise ValueError("contact audit requires full per-frame sole targets")
    if not np.any(stance):
        raise ValueError("contact audit found no authored stance frames")
    core_stance = _full_sole_stance_mask(stance)
    if not np.any(core_stance):
        raise ValueError("contact audit found no core stance frames")

    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    ray_z = float(np.max(target_mesh.vertices_world[:, 2]) + 1.0)
    support_count = np.zeros((frame_count, 2), dtype=np.int16)
    centres = np.zeros((frame_count, 2, 3), dtype=np.float64)
    target_error = np.zeros((frame_count, 2), dtype=np.float64)
    closest_surface_distance: list[float] = []
    all_probe_hover: list[float] = []
    core_probe_hover: list[float] = []

    for frame in range(frame_count):
        feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        support = adapter.sole_support_points_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        for foot in range(2):
            centres[frame, foot] = np.mean(feet[foot], axis=0)
            target_error[frame, foot] = float(
                np.max(np.linalg.norm(feet[foot] - targets[frame, foot], axis=1))
            )
            gaps = []
            for point in support[foot]:
                height = _terrain_height(target_mesh, point[:2], ray_z)
                if not math.isfinite(height):
                    height = float(ground_fallback_height_m)
                gaps.append(float(point[2]) - float(height))
            gaps_array = np.asarray(gaps, dtype=np.float64)
            support_count[frame, foot] = int(
                np.count_nonzero(np.abs(gaps_array) <= float(support_tolerance_m))
            )
            if stance[frame, foot]:
                closest_surface_distance.append(
                    float(np.min(np.abs(gaps_array)))
                )
                all_probe_hover.append(float(np.max(gaps_array)))
            if core_stance[frame, foot]:
                core_probe_hover.append(float(np.max(gaps_array)))

    minimum_support = int(np.min(support_count[stance]))
    maximum_closest_distance = float(max(closest_surface_distance))
    maximum_target_error = float(np.max(target_error[stance]))
    maximum_drift = float(_stance_runs_maximum_drift(centres, stance))
    target_score = np.where(stance, target_error, -math.inf)
    worst_target_flat = int(np.argmax(target_score))
    worst_target_frame, worst_target_foot = np.unravel_index(
        worst_target_flat, target_score.shape
    )
    worst_drift = -1.0
    worst_drift_frame = 0
    worst_drift_span: tuple[int, int, int] | None = None
    for span in _stance_spans(stance):
        values = centres[
            span.start_frame : span.stop_frame,
            span.foot_index,
            :2,
        ]
        distance = np.linalg.norm(values - values[0], axis=1)
        local = int(np.argmax(distance))
        if float(distance[local]) > worst_drift:
            worst_drift = float(distance[local])
            worst_drift_frame = span.start_frame + local
            worst_drift_span = (
                span.foot_index,
                span.start_frame,
                span.stop_frame,
            )
    minimum_core_support = int(np.min(support_count[core_stance]))
    maximum_core_hover = float(max(core_probe_hover))
    extras["target_stance_support_point_count"] = support_count
    extras["adapted_sole_center_world"] = np.asarray(centres, dtype=np.float32)
    extras["per_frame_sole_target_error_by_foot_m"] = np.asarray(
        target_error, dtype=np.float32
    )
    extras["per_frame_sole_target_error_m"] = np.asarray(
        np.max(target_error, axis=1), dtype=np.float32
    )
    accepted = bool(
        minimum_support >= 2
        and maximum_closest_distance <= float(support_tolerance_m)
        and maximum_target_error <= float(maximum_stance_target_error_m)
        and maximum_drift <= float(maximum_stance_drift_m)
        and minimum_core_support >= 3
        and maximum_core_hover <= float(maximum_core_stance_probe_hover_m)
    )
    return {
        "accepted": accepted,
        "support_tolerance_m": float(support_tolerance_m),
        "minimum_stance_support_point_count": minimum_support,
        "maximum_closest_stance_surface_distance_m": maximum_closest_distance,
        "maximum_stance_sole_target_error_m": maximum_target_error,
        "maximum_stance_target_error_frame": int(worst_target_frame),
        "maximum_stance_target_error_foot": int(worst_target_foot),
        "maximum_stance_run_drift_m": maximum_drift,
        "maximum_stance_drift_frame": int(worst_drift_frame),
        "maximum_stance_drift_span": (
            list(worst_drift_span) if worst_drift_span is not None else None
        ),
        "maximum_any_stance_probe_hover_m": float(max(all_probe_hover)),
        "minimum_core_stance_support_point_count": minimum_core_support,
        "maximum_core_stance_probe_hover_m": maximum_core_hover,
        "maximum_core_stance_probe_hover_threshold_m": float(
            maximum_core_stance_probe_hover_m
        ),
    }


def _repair_swing_foot_clearance(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    collision: object,
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: TerrainMeshIndex,
    model_path: Path,
    joint_names: Sequence[str],
    maximum_total_lift_m: float = 0.060,
) -> tuple[
    StitchedMotion,
    dict[str, np.ndarray],
    object,
    dict[str, float],
]:
    """Lift only an offending swing trajectory over a riser.

    The exact collision audit reports aggregate foot penetration.  During a
    single-support interval the non-stance foot is the only valid target to
    move: lifting the pelvis or planted sole creates visible hovering.  A
    smooth upper envelope is accumulated over each swing run and the legs are
    refitted to the modified targets after every audit iteration.
    """

    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    targets = np.asarray(
        extras.get("target_sole_points_world"), dtype=np.float64
    ).copy()
    frame_count = len(motion.root_position_world)
    if stance.shape != (frame_count, 2) or targets.shape[:2] != (
        frame_count,
        2,
    ):
        raise ValueError("swing repair requires stance and full sole targets")
    total = np.zeros((frame_count, 2), dtype=np.float64)
    repaired = motion
    updated = dict(extras)
    current = collision
    previous_penetration = float(current.maximum_foot_penetration_m)
    iterations = 0
    maximum_per_foot_required = np.zeros(2, dtype=np.float64)

    for iteration in range(4):
        if bool(current.accepted):
            break
        if float(current.maximum_forbidden_body_penetration_m) > 1.0e-6:
            break
        penetration = np.asarray(
            current.per_frame_max_foot_penetration_m, dtype=np.float64
        )
        required = np.maximum(penetration - 0.0025, 0.0) + (
            penetration > 0.0025
        ) * 0.001
        # The exact audit historically reports one aggregate foot value.  Do
        # not apply that value to both free legs: estimate which lower-foot
        # geometry is actually below the surface, then assign any aggregate
        # residual to only the responsible swing foot.
        roots = np.asarray(repaired.root_position_world, dtype=np.float64)
        quaternions = np.asarray(
            repaired.root_quaternion_world_wxyz, dtype=np.float64
        )
        joints = np.asarray(repaired.joint_position, dtype=np.float64)
        ray_z = float(np.max(target_mesh.vertices_world[:, 2]) + 1.0)
        per_foot_penetration = np.zeros((frame_count, 2), dtype=np.float64)
        for frame in range(frame_count):
            support, envelope = (
                adapter.sole_support_and_collision_envelope_points_for_pose(
                root_position=roots[frame],
                root_quaternion_wxyz=quaternions[frame],
                joints=joints[frame],
                )
            )
            for foot in range(2):
                points = np.concatenate((support[foot], envelope[foot]), axis=0)
                values = []
                for point in points:
                    height = _terrain_height(target_mesh, point[:2], ray_z)
                    if math.isfinite(height):
                        values.append(max(0.0, float(height) - float(point[2])))
                if values:
                    per_foot_penetration[frame, foot] = max(values)
        per_foot_required = np.maximum(
            per_foot_penetration - 0.0025, 0.0
        ) + (per_foot_penetration > 0.0025) * 0.001
        for frame in range(frame_count):
            free = np.flatnonzero(~stance[frame])
            if not len(free) or required[frame] <= 0.0:
                continue
            responsible = int(
                free[
                    np.argmax(per_foot_penetration[frame, free])
                ]
            )
            per_foot_required[frame, responsible] = max(
                per_foot_required[frame, responsible], required[frame]
            )
        maximum_per_foot_required = np.maximum(
            maximum_per_foot_required,
            np.max(per_foot_required, axis=0),
        )
        increment = np.zeros_like(total)
        for foot in range(2):
            swing = ~stance[:, foot]
            seed = np.where(swing, per_foot_required[:, foot], 0.0)
            if not np.any(seed > 0.0):
                continue
            # Use one smooth clearance arc per flight phase.  Masking a
            # generic upper envelope to zero at stance creates exactly the
            # one-frame knee snap this repair is meant to prevent.  The local
            # smoothstep window reaches full height after eight frames and
            # returns toward the next plant; open clip ends are not forced to
            # an artificial zero-height contact.
            start = 0
            while start < frame_count:
                if not swing[start]:
                    start += 1
                    continue
                stop = start + 1
                while stop < frame_count and swing[stop]:
                    stop += 1
                length = stop - start
                coordinate = np.arange(1, length + 1, dtype=np.float64)
                ramp = 8.0
                entry = np.ones(length, dtype=np.float64)
                exit = np.ones(length, dtype=np.float64)
                if start > 0:
                    entry = np.clip(coordinate / (ramp + 1.0), 0.0, 1.0)
                    entry = entry * entry * (3.0 - 2.0 * entry)
                if stop < frame_count:
                    reverse = np.arange(
                        length, 0, -1, dtype=np.float64
                    )
                    exit = np.clip(reverse / (ramp + 1.0), 0.0, 1.0)
                    exit = exit * exit * (3.0 - 2.0 * exit)
                window = np.minimum(entry, exit)
                # A newly released foot can already intersect terrain that
                # rises immediately beyond its planted patch.  Forcing the
                # clearance arc to near zero on that first free frame makes a
                # perfectly repairable 2--3 cm lift look like a 0.7 m arc and
                # aborts the complete repair.  Allow half-height at the free
                # side of the contact boundary; the later adaptive temporal
                # subdivision supplies the intermediate poses and the final
                # joint-step/contact gates still reject a visible snap.
                window = np.maximum(window, 0.50)
                local_required = seed[start:stop]
                amplitude = float(
                    np.max(local_required / np.maximum(window, 1.0e-6))
                )
                increment[start:stop, foot] = amplitude * window
                start = stop
        if not np.any(increment > 0.0):
            break
        if float(np.max(total + increment)) > float(maximum_total_lift_m):
            break
        total += increment
        targets[:, :, :, 2] += increment[:, :, None]
        updated["target_sole_points_world"] = np.asarray(
            targets, dtype=np.float32
        )
        updated["swing_clearance_repair_m"] = np.asarray(
            total, dtype=np.float32
        )
        repaired, updated, _refit = _refit_motion_to_sole_targets(
            repaired,
            updated,
            adapter=adapter,
        )
        current = audit_stair_motion_collisions(
            repaired,
            model_path=model_path,
            target_mesh=target_mesh,
            joint_names=joint_names,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=0.0,
        )
        iterations = iteration + 1
        new_penetration = float(current.maximum_foot_penetration_m)
        if (
            not bool(current.accepted)
            and new_penetration >= previous_penetration - 1.0e-4
        ):
            break
        previous_penetration = new_penetration

    diagnostics = {
        "iterations": float(iterations),
        "maximum_target_lift_m": float(np.max(total)),
        "maximum_target_lift_step_m": float(
            np.max(np.abs(np.diff(total, axis=0)))
            if frame_count > 1
            else 0.0
        ),
        "maximum_required_lift_by_foot_m": [
            float(value) for value in maximum_per_foot_required
        ],
        "maximum_remaining_foot_penetration_m": float(
            current.maximum_foot_penetration_m
        ),
    }
    return repaired, updated, current, diagnostics


def _repair_swing_planar_overshoot(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    collision: object,
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: TerrainMeshIndex,
    model_path: Path,
    joint_names: Sequence[str],
) -> tuple[
    StitchedMotion,
    dict[str, np.ndarray],
    object,
    dict[str, float],
]:
    """Advance a late swing to its next foothold only when it hits a riser."""

    full_sole_stance = np.asarray(
        extras.get("full_sole_stance_mask"), dtype=bool
    )
    targets = np.asarray(
        extras.get("target_sole_points_world"), dtype=np.float64
    )
    if full_sole_stance.shape != targets.shape[:2]:
        return motion, extras, collision, {
            "maximum_planar_swing_overshoot_correction_m": 0.0,
            "corrected_swing_frame_foot_count": 0.0,
            "reverted_no_improvement": 1.0,
        }
    corrected, diagnostics = _clamp_bracketed_swing_overshoot(
        targets, full_sole_stance
    )
    if diagnostics["maximum_planar_swing_overshoot_correction_m"] <= 1.0e-8:
        return motion, extras, collision, {
            **diagnostics,
            "reverted_no_improvement": 1.0,
        }
    candidate_extras = dict(extras)
    candidate_extras["target_sole_points_world"] = np.asarray(
        corrected, dtype=np.float32
    )
    candidate, candidate_extras, refit = _refit_motion_to_sole_targets(
        motion,
        candidate_extras,
        adapter=adapter,
    )
    candidate_collision = audit_stair_motion_collisions(
        candidate,
        model_path=model_path,
        target_mesh=target_mesh,
        joint_names=joint_names,
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=0.0,
    )
    improved = bool(
        candidate_collision.maximum_forbidden_body_penetration_m
        <= collision.maximum_forbidden_body_penetration_m + 1.0e-9
        and candidate_collision.maximum_foot_penetration_m
        < collision.maximum_foot_penetration_m - 1.0e-4
    )
    report = {
        **diagnostics,
        "maximum_refit_error_m": float(refit["maximum_sole_target_error_m"]),
        "maximum_refit_joint_step_rad": float(refit["maximum_joint_step_rad"]),
        "pre_repair_foot_penetration_m": float(
            collision.maximum_foot_penetration_m
        ),
        "candidate_foot_penetration_m": float(
            candidate_collision.maximum_foot_penetration_m
        ),
        "reverted_no_improvement": float(not improved),
    }
    if not improved:
        return motion, extras, collision, report
    return candidate, candidate_extras, candidate_collision, report


def _repair_stance_foot_mesh_clearance(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    collision: object,
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: TerrainMeshIndex,
    model_path: Path,
    joint_names: Sequence[str],
    maximum_target_lift_m: float = 0.006,
) -> tuple[
    StitchedMotion,
    dict[str, np.ndarray],
    object,
    dict[str, float],
]:
    """Clear the rendered foot hull with a millimetric contact offset."""

    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    targets = np.asarray(
        extras.get("target_sole_points_world"), dtype=np.float64
    )
    penetration = np.asarray(
        collision.per_frame_max_foot_penetration_m, dtype=np.float64
    )
    anchored = targets.copy()
    assigned = np.zeros(stance.shape, dtype=bool)
    maximum_lift = 0.0
    for span in _stance_spans(stance):
        required = float(
            np.max(
                np.maximum(
                    penetration[span.start_frame : span.stop_frame] - 0.0025,
                    0.0,
                )
            )
            + 0.0005
        )
        if required <= 0.0005:
            continue
        if required > float(maximum_target_lift_m):
            continue
        anchored[span.start_frame : span.stop_frame, span.foot_index, :, 2] += (
            required
        )
        assigned[span.start_frame : span.stop_frame, span.foot_index] = True
        maximum_lift = max(maximum_lift, required)
    if not np.any(assigned):
        return motion, extras, collision, {
            "maximum_target_lift_m": 0.0,
            "maximum_remaining_foot_penetration_m": float(
                collision.maximum_foot_penetration_m
            ),
        }

    candidate_extras = dict(extras)
    candidate_extras["target_sole_points_world"] = np.asarray(
        _blend_anchored_sole_offsets(targets, anchored, assigned),
        dtype=np.float32,
    )
    candidate_extras["stance_mesh_clearance_target_mask"] = assigned
    candidate, candidate_extras, _refit = _refit_motion_to_sole_targets(
        motion,
        candidate_extras,
        adapter=adapter,
    )
    candidate_collision = audit_stair_motion_collisions(
        candidate,
        model_path=model_path,
        target_mesh=target_mesh,
        joint_names=joint_names,
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=0.0,
    )
    if (
        float(candidate_collision.maximum_foot_penetration_m)
        >= float(collision.maximum_foot_penetration_m) - 1.0e-4
    ):
        return motion, extras, collision, {
            "maximum_target_lift_m": float(maximum_lift),
            "maximum_remaining_foot_penetration_m": float(
                collision.maximum_foot_penetration_m
            ),
            "reverted_no_improvement": 1.0,
        }
    return candidate, candidate_extras, candidate_collision, {
        "maximum_target_lift_m": float(maximum_lift),
        "maximum_remaining_foot_penetration_m": float(
            candidate_collision.maximum_foot_penetration_m
        ),
        "reverted_no_improvement": 0.0,
    }


def _raise_pelvis_to_reach_sole_targets(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    adapter: _G1FootfallAdapter,
    maximum_lift_m: float = 0.120,
    lift_step_m: float = 0.010,
    stance_tolerance_m: float = 0.003,
    swing_tolerance_m: float = 0.035,
) -> tuple[StitchedMotion, dict[str, np.ndarray], dict[str, float]]:
    """Find a smooth pelvis-height envelope that makes sole IK reachable."""

    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    targets = np.asarray(
        extras.get("target_sole_points_world"), dtype=np.float64
    )
    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    frame_count = len(roots)
    if targets.ndim != 4 or targets.shape[:2] != (frame_count, 2):
        raise ValueError("pelvis reach solve requires full sole targets")
    if stance.shape != (frame_count, 2):
        raise ValueError("pelvis reach solve requires a stance mask")
    constraint_stance = np.asarray(
        extras.get("full_sole_stance_mask", stance), dtype=bool
    )
    if constraint_stance.shape != stance.shape:
        raise ValueError("full-sole stance mask must match authored stance")

    maximum_lift = float(maximum_lift_m)
    candidates = np.arange(
        0.0,
        maximum_lift + 0.5 * float(lift_step_m),
        float(lift_step_m),
        dtype=np.float64,
    )
    feasible = np.ones((frame_count, len(candidates)), dtype=bool)
    for frame in range(frame_count):
        if not np.any(constraint_stance[frame]):
            continue
        feasible[frame] = False
        candidate_errors: list[tuple[float, float, float]] = []
        reach_targets = targets[frame].copy()
        authored_feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        for foot in range(2):
            if not constraint_stance[frame, foot]:
                reach_targets[foot] = authored_feet[foot]
        for state, lift in enumerate(candidates):
            root = roots[frame].copy()
            root[2] += float(lift)
            adapted, _correction, _error = adapter.adapt_to_targets(
                root_position=root,
                root_quaternion_wxyz=quaternions[frame],
                authored_joints=joints[frame],
                sole_targets_world=reach_targets,
                initial_joints=None,
            )
            feet = adapter.sole_positions_for_pose(
                root_position=root,
                root_quaternion_wxyz=quaternions[frame],
                joints=adapted,
            )
            errors = np.asarray(
                [
                    np.max(
                        np.linalg.norm(
                            feet[foot] - reach_targets[foot], axis=1
                        )
                    )
                    for foot in range(2)
                ],
                dtype=np.float64,
            )
            stance_error = float(np.max(errors[constraint_stance[frame]]))
            swing_error = float(
                np.max(errors[~constraint_stance[frame]])
                if np.any(~constraint_stance[frame])
                else 0.0
            )
            candidate_errors.append((float(lift), stance_error, swing_error))
            feasible[frame, state] = bool(
                stance_error <= float(stance_tolerance_m)
                and swing_error <= float(swing_tolerance_m)
            )
        if not np.any(feasible[frame]):
            best_lift, best_stance_error, best_swing_error = min(
                candidate_errors,
                key=lambda value: (
                    max(
                        value[1] / float(stance_tolerance_m),
                        value[2] / float(swing_tolerance_m),
                    ),
                    value[0],
                ),
            )
            raise ValueError(
                "side-on sole targets have no feasible pelvis height: "
                f"frame={frame} "
                f"stance={stance[frame].astype(int).tolist()} "
                "full_sole_stance="
                f"{constraint_stance[frame].astype(int).tolist()} "
                f"best_lift_m={best_lift:.3f} "
                f"stance_error_m={best_stance_error:.6f} "
                f"swing_error_m={best_swing_error:.6f}"
            )

    # Select one globally smooth path through the per-frame feasible sets.
    # A one-sided upper envelope is invalid here: raising the pelvis for the
    # next tread can overextend the opposite leg during double support.
    state_count = len(candidates)
    cost = np.full((frame_count, state_count), math.inf, dtype=np.float64)
    back = np.full((frame_count, state_count), -1, dtype=np.int16)
    height_units = candidates / float(lift_step_m)
    cost[0, feasible[0]] = 0.02 * np.square(height_units[feasible[0]])
    for frame in range(1, frame_count):
        previous_states = np.flatnonzero(np.isfinite(cost[frame - 1]))
        for state in np.flatnonzero(feasible[frame]):
            transition = np.square(
                height_units[state] - height_units[previous_states]
            )
            values = cost[frame - 1, previous_states] + transition
            best_index = int(np.argmin(values))
            prior = int(previous_states[best_index])
            cost[frame, state] = (
                float(values[best_index])
                + 0.02 * float(height_units[state] ** 2)
            )
            back[frame, state] = prior
    final_state = int(np.argmin(cost[-1]))
    if not math.isfinite(float(cost[-1, final_state])):
        raise ValueError("no continuous pelvis-height path crosses the clip")
    selected_states = np.zeros(frame_count, dtype=np.int16)
    selected_states[-1] = final_state
    for frame in range(frame_count - 1, 0, -1):
        selected_states[frame - 1] = back[frame, selected_states[frame]]
    lift = candidates[selected_states]
    raised_roots = roots.copy()
    raised_roots[:, 2] += lift
    raised = StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(raised_roots, dtype=np.float32),
        root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
        joint_position=motion.joint_position,
        provenance=motion.provenance,
        seam_indices=motion.seam_indices,
    )
    updated = dict(extras)
    updated["sole_reach_root_lift_m"] = np.asarray(lift, dtype=np.float32)
    updated["sole_reach_feasible_minimum_lift_m"] = np.asarray(
        [candidates[np.flatnonzero(row)[0]] for row in feasible],
        dtype=np.float32,
    )
    updated["sole_reach_feasible_maximum_lift_m"] = np.asarray(
        [candidates[np.flatnonzero(row)[-1]] for row in feasible],
        dtype=np.float32,
    )
    raised, updated, refit = _refit_motion_to_sole_targets(
        raised,
        updated,
        adapter=adapter,
    )
    return raised, updated, {
        "maximum_selected_lift_m": float(np.max(lift)),
        "maximum_lift_step_m": float(
            np.max(np.abs(np.diff(lift))) if frame_count > 1 else 0.0
        ),
        **{f"refit_{key}": float(value) for key, value in refit.items()},
    }


def build(arguments: argparse.Namespace) -> dict[str, object]:
    archive_path = arguments.archive.expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        arguments.model.expanduser().resolve(),
        joint_names,
        maximum_joint_correction_rad=(
            MAXIMUM_SIDE_ON_JOINT_CORRECTION_RAD
        ),
        target_tolerance_m=1.0e-4,
        maximum_iterations=96,
        damping=0.004,
        posture_weight=1.0e-5,
    )
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = tuple(
        Path(value).expanduser().resolve() for value in arguments.source_motion
    )
    rows: list[dict[str, object]] = []
    for clip_index in arguments.clip_index:
        clip = int(clip_index)
        traversal = str(archive["clip_traversal"][clip])
        if traversal not in {"up", "down"}:
            raise ValueError(f"clip {clip} is not a stair traversal")
        target_mesh = _archive_terrain_index(archive, clip)
        target_route = motion_conditioned_stair_support_route(archive, clip)
        target_start = np.asarray(target_route.start_xy, dtype=np.float64)
        target_direction = np.asarray(
            target_route.end_xy, dtype=np.float64
        ) - target_start
        target_distance = float(np.linalg.norm(target_direction))
        target_direction /= target_distance
        target_start_height = float(target_route.levels[0].height_m)
        target_ground_height = float(
            min(level.height_m for level in target_route.levels)
        )
        for source_path in source_paths:
            source_id = (
                source_path.parent.name
                if source_path.name == "motion.npz"
                else source_path.stem
            )
            label = f"clip_{clip:03d}_{source_id}_side_on"
            destination = output / label
            destination.mkdir(parents=True, exist_ok=True)
            row: dict[str, object] = {
                "label": label,
                "clip_index": clip,
                "clip_name": str(archive["clip_names"][clip]),
                "traversal": traversal,
                "mode": "bones_side_on",
                "source_kind": "motionbricks_lateral",
                "source_motion": str(source_path),
                "status": "rejected",
            }
            try:
                source, source_mesh, source_metadata = _load_aligned_lateral_source(
                    source_path,
                    adapter=adapter,
                    joint_names=joint_names,
                    target_start_xy=target_start,
                    target_direction_xy=target_direction,
                    target_distance_m=target_distance,
                    target_base_height_m=target_start_height,
                    exit_margin_m=arguments.exit_margin_m,
                    forced_start_frame=arguments.source_start_frame,
                    arm_swing_scale=arguments.arm_swing_scale,
                    wrist_swing_scale=arguments.wrist_swing_scale,
                    arm_smoothing_sigma_frames=(
                        arguments.arm_smoothing_sigma_frames
                    ),
                )
                row["source_alignment"] = source_metadata
                (
                    root_height_offset,
                    root_progress_offset,
                    target_level_by_span,
                    suggested_stop_frame,
                ) = _support_conditioned_route_offsets(
                    source,
                    route=target_route,
                    adapter=adapter,
                    smoothing_frames=arguments.pelvis_height_smoothing_frames,
                    leading_foot_index=(
                        1
                        if float(source_metadata["body_to_travel_angle_deg"])
                        < 0.0
                        else 0
                    ),
                )
                if suggested_stop_frame < len(source.root_position_world):
                    source = _truncate_motion(source, suggested_stop_frame)
                    source_metadata["trimmed_output_frame_count"] = len(
                        source.root_position_world
                    )
                    (
                        root_height_offset,
                        root_progress_offset,
                        target_level_by_span,
                        suggested_stop_frame,
                    ) = _support_conditioned_route_offsets(
                        source,
                        route=target_route,
                        adapter=adapter,
                        smoothing_frames=(
                            arguments.pelvis_height_smoothing_frames
                        ),
                        leading_foot_index=(
                            1
                            if float(
                                source_metadata["body_to_travel_angle_deg"]
                            )
                            < 0.0
                            else 0
                        ),
                    )
                if suggested_stop_frame != len(source.root_position_world):
                    raise ValueError("lateral top-exit trim did not converge")
                motion, diagnostics, extras = warp_motion(
                    source,
                    adapter=adapter,
                    target_mesh=target_mesh,
                    target_route=target_route,
                    direction_xy=target_direction,
                    active_start_m=0.0,
                    active_stop_m=target_distance,
                    mode="straight",
                    maximum_amplitude_m=0.25,
                    source_mesh=source_mesh,
                    minimum_stance_run_frames=4,
                    maximum_stance_gap_frames=4,
                    terrain_clear_swings=True,
                    minimum_swing_clearance_m=arguments.swing_clearance_m,
                    swing_adjustment_taper_frames=8,
                    root_height_offset_m=root_height_offset,
                    root_progress_offset_m=root_progress_offset,
                    root_anchor_smoothing_frames=(
                        arguments.root_anchor_smoothing_frames
                    ),
                    maximum_foothold_level_step=1,
                    ik_multistart=True,
                    target_level_by_stance_span=target_level_by_span,
                    maximum_ik_continuity_step_rad=0.18,
                    root_anchor_vertical=False,
                    root_anchor_vertical_limit_m=(
                        arguments.root_anchor_vertical_limit_m
                    ),
                    ik_continuity_acceptable_error_m=0.012,
                    ground_fallback_height_m=target_ground_height,
                    post_selection_maximum_joint_step_rad=0.20,
                    foothold_anchor_config=FootholdAnchorConfig(
                        max_longitudinal_adjustment_m=0.28,
                        max_lateral_adjustment_m=0.0,
                        max_yaw_adjustment_rad=0.0,
                        max_vertical_adjustment_m=(
                            float(
                                max(level.height_m for level in target_route.levels)
                                - min(
                                    level.height_m for level in target_route.levels
                                )
                            )
                            + 0.10
                        ),
                        longitudinal_samples=29,
                        lateral_samples=1,
                        yaw_samples=1,
                        lateral_seed_offsets_m=(0.0,),
                        route_lateral_offset_m=0.0,
                        route_alignment_weight=2.0,
                        minimum_support_points=2,
                    ),
                )
                row["warp"] = asdict(diagnostics)
                mechanical = _mechanically_accepted(diagnostics)
                repairable, repairable_metrics = _repairable_warp_accepted(
                    diagnostics,
                    extras,
                )
                row["mechanical_accepted"] = mechanical
                row["repairable_warp_accepted"] = repairable
                row["repairable_warp_metrics"] = repairable_metrics
                if not (mechanical or repairable):
                    debug_path = destination / "rejected_motion_debug.npz"
                    _save_motion(
                        debug_path,
                        motion,
                        mode="bones_side_on_rejected_debug",
                        extras=extras,
                    )
                    row["rejected_motion_debug"] = str(debug_path.resolve())
                    raise ValueError("retargeted lateral gait failed mechanical gate")
                extras, surface_conformance = (
                    _conform_stance_sole_targets_to_mesh(
                        extras,
                        adapter=adapter,
                        target_mesh=target_mesh,
                        ground_fallback_height_m=target_ground_height,
                    )
                )
                row["stance_sole_surface_conformance"] = surface_conformance
                motion, extras, pelvis_reach = (
                    _raise_pelvis_to_reach_sole_targets(
                        motion,
                        extras,
                        adapter=adapter,
                    )
                )
                row["sole_reach_pelvis_adjustment"] = pelvis_reach
                collision = audit_stair_motion_collisions(
                    motion,
                    model_path=arguments.model,
                    target_mesh=target_mesh,
                    joint_names=joint_names,
                    maximum_foot_penetration_m=0.005,
                    maximum_forbidden_body_penetration_m=0.0,
                )
                row["pre_clearance_collision_audit"] = collision.to_dict()
                clearance_lift = 0.0
                if (
                    not collision.accepted
                    and collision.maximum_forbidden_body_penetration_m
                    <= 1.0e-6
                    and collision.maximum_foot_penetration_m <= 0.025
                ):
                    motion, extras, collision, planar_swing_repair = (
                        _repair_swing_planar_overshoot(
                            motion,
                            extras,
                            collision,
                            adapter=adapter,
                            target_mesh=target_mesh,
                            model_path=arguments.model,
                            joint_names=joint_names,
                        )
                    )
                    row["swing_planar_overshoot_repair"] = planar_swing_repair
                if (
                    not collision.accepted
                    and collision.maximum_forbidden_body_penetration_m
                    <= 1.0e-6
                    and collision.maximum_foot_penetration_m <= 0.015
                ):
                    motion, extras, collision, stance_mesh_repair = (
                        _repair_stance_foot_mesh_clearance(
                            motion,
                            extras,
                            collision,
                            adapter=adapter,
                            target_mesh=target_mesh,
                            model_path=arguments.model,
                            joint_names=joint_names,
                        )
                    )
                    row["stance_mesh_clearance_repair"] = stance_mesh_repair
                if (
                    not collision.accepted
                    and collision.maximum_forbidden_body_penetration_m
                    <= 1.0e-6
                    and collision.maximum_foot_penetration_m <= 0.025
                ):
                    motion, extras, collision, swing_repair = (
                        _repair_swing_foot_clearance(
                            motion,
                            extras,
                            collision,
                            adapter=adapter,
                            target_mesh=target_mesh,
                            model_path=arguments.model,
                            joint_names=joint_names,
                        )
                    )
                    row["swing_clearance_repair"] = swing_repair
                if (
                    not collision.accepted
                    and collision.maximum_forbidden_body_penetration_m
                    <= 1.0e-6
                    and collision.maximum_foot_penetration_m <= 0.015
                ):
                    motion, extras, collision, post_swing_stance_repair = (
                        _repair_stance_foot_mesh_clearance(
                            motion,
                            extras,
                            collision,
                            adapter=adapter,
                            target_mesh=target_mesh,
                            model_path=arguments.model,
                            joint_names=joint_names,
                        )
                    )
                    row["post_swing_stance_mesh_clearance_repair"] = (
                        post_swing_stance_repair
                    )
                if (
                    not collision.accepted
                    and collision.maximum_forbidden_body_penetration_m
                    > 1.0e-6
                    and collision.maximum_forbidden_body_penetration_m <= 0.015
                    and collision.maximum_foot_penetration_m <= 0.025
                ):
                    original = motion
                    clearance_refit: dict[str, float] | None = None
                    for repair_budget_m in (0.035, 0.006):
                        motion, collision, _repair_lift = _repair_exact_mesh_clearance(
                            motion,
                            collision,
                            archive_path=archive_path,
                            target_clip_index=clip,
                            target_mesh=target_mesh,
                            model_path=arguments.model,
                            maximum_total_lift_m=repair_budget_m,
                        )
                        motion, extras, clearance_refit = (
                            _refit_motion_to_sole_targets(
                                motion,
                                extras,
                                adapter=adapter,
                            )
                        )
                        collision = audit_stair_motion_collisions(
                            motion,
                            model_path=arguments.model,
                            target_mesh=target_mesh,
                            joint_names=joint_names,
                            maximum_foot_penetration_m=0.005,
                            maximum_forbidden_body_penetration_m=0.0,
                        )
                        if collision.accepted:
                            break
                    clearance_lift = float(
                        np.max(
                            motion.root_position_world[:, 2]
                            - original.root_position_world[:, 2]
                        )
                    )
                    extras["clearance_root_lift_m"] = np.asarray(
                        motion.root_position_world[:, 2]
                        - original.root_position_world[:, 2],
                        dtype=np.float32,
                    )
                    row["clearance_refit"] = clearance_refit
                row["clearance_repair_maximum_m"] = float(clearance_lift)
                row["collision_audit"] = collision.to_dict()
                if not collision.accepted:
                    debug_path = destination / "rejected_motion_debug.npz"
                    _save_motion(
                        debug_path,
                        motion,
                        mode="bones_side_on_collision_rejected_debug",
                        extras=extras,
                    )
                    row["rejected_motion_debug"] = str(debug_path.resolve())
                    raise ValueError("retargeted lateral gait failed collision gate")
                motion, extras = _retime_motion_and_extras(
                    motion,
                    extras,
                    temporal_scale=arguments.temporal_scale,
                )
                row["temporal_scale"] = float(arguments.temporal_scale)
                motion, extras, retime_refit = _refit_motion_to_sole_targets(
                    motion,
                    extras,
                    adapter=adapter,
                )
                row["retimed_sole_refit"] = retime_refit
                adaptive_subdivision: list[dict[str, object]] = []
                for _subdivision_pass in range(2):
                    current_metrics = _retimed_motion_metrics(motion)
                    if current_metrics["maximum_joint_step_rad"] <= 0.18:
                        break
                    motion, extras, subdivision = (
                        _adaptively_subdivide_motion_steps(
                            motion,
                            extras,
                            target_joint_step_rad=0.14,
                        )
                    )
                    motion, extras, subdivision_refit = (
                        _refit_motion_to_sole_targets(
                            motion,
                            extras,
                            adapter=adapter,
                        )
                    )
                    adaptive_subdivision.append(
                        {
                            **subdivision,
                            "refit": subdivision_refit,
                        }
                    )
                row["adaptive_step_subdivision"] = adaptive_subdivision
                row["retimed_motion_metrics"] = _retimed_motion_metrics(motion)
                collision = audit_stair_motion_collisions(
                    motion,
                    model_path=arguments.model,
                    target_mesh=target_mesh,
                    joint_names=joint_names,
                    maximum_foot_penetration_m=0.005,
                    maximum_forbidden_body_penetration_m=0.0,
                )
                row["retimed_collision_audit"] = collision.to_dict()
                row["collision_audit"] = collision.to_dict()
                if arguments.temporal_scale > 1.0:
                    retimed_clearance_lift = 0.0
                    if (
                        not collision.accepted
                        and collision.maximum_forbidden_body_penetration_m
                        <= 1.0e-6
                        and collision.maximum_foot_penetration_m <= 0.015
                    ):
                        motion, extras, collision, retimed_stance_repair = (
                            _repair_stance_foot_mesh_clearance(
                                motion,
                                extras,
                                collision,
                                adapter=adapter,
                                target_mesh=target_mesh,
                                model_path=arguments.model,
                                joint_names=joint_names,
                            )
                        )
                        row["retimed_stance_mesh_clearance_repair"] = (
                            retimed_stance_repair
                        )
                    if (
                        not collision.accepted
                        and collision.maximum_forbidden_body_penetration_m
                        <= 1.0e-6
                        and collision.maximum_foot_penetration_m <= 0.015
                    ):
                        motion, extras, collision, retimed_swing_repair = (
                            _repair_swing_foot_clearance(
                                motion,
                                extras,
                                collision,
                                adapter=adapter,
                                target_mesh=target_mesh,
                                model_path=arguments.model,
                                joint_names=joint_names,
                                maximum_total_lift_m=0.030,
                            )
                        )
                        row["retimed_swing_clearance_repair"] = (
                            retimed_swing_repair
                        )
                    if (
                        not collision.accepted
                        and collision.maximum_forbidden_body_penetration_m
                        <= 1.0e-6
                        and collision.maximum_foot_penetration_m <= 0.015
                    ):
                        motion, extras, collision, retimed_post_swing_stance = (
                            _repair_stance_foot_mesh_clearance(
                                motion,
                                extras,
                                collision,
                                adapter=adapter,
                                target_mesh=target_mesh,
                                model_path=arguments.model,
                                joint_names=joint_names,
                            )
                        )
                        row["retimed_post_swing_stance_mesh_repair"] = (
                            retimed_post_swing_stance
                        )
                    if (
                        not collision.accepted
                        and collision.maximum_forbidden_body_penetration_m
                        > 1.0e-6
                        and collision.maximum_forbidden_body_penetration_m
                        <= 0.005
                        and collision.maximum_foot_penetration_m <= 0.015
                    ):
                        original_retimed = motion
                        retimed_clearance_refit: dict[str, float] | None = None
                        for repair_budget_m in (0.012, 0.004):
                            motion, collision, _repair_lift = (
                                _repair_exact_mesh_clearance(
                                    motion,
                                    collision,
                                    archive_path=archive_path,
                                    target_clip_index=clip,
                                    target_mesh=target_mesh,
                                    model_path=arguments.model,
                                    maximum_total_lift_m=repair_budget_m,
                                )
                            )
                            motion, extras, retimed_clearance_refit = (
                                _refit_motion_to_sole_targets(
                                    motion,
                                    extras,
                                    adapter=adapter,
                                )
                            )
                            collision = audit_stair_motion_collisions(
                                motion,
                                model_path=arguments.model,
                                target_mesh=target_mesh,
                                joint_names=joint_names,
                                maximum_foot_penetration_m=0.005,
                                maximum_forbidden_body_penetration_m=0.0,
                            )
                            if collision.accepted:
                                break
                        retimed_delta = np.asarray(
                            motion.root_position_world[:, 2]
                            - original_retimed.root_position_world[:, 2],
                            dtype=np.float32,
                        )
                        retimed_clearance_lift = float(np.max(retimed_delta))
                        extras["retimed_clearance_root_lift_m"] = retimed_delta
                        if "clearance_root_lift_m" in extras:
                            extras["clearance_root_lift_m"] = np.asarray(
                                extras["clearance_root_lift_m"] + retimed_delta,
                                dtype=np.float32,
                            )
                        row["retimed_clearance_refit"] = (
                            retimed_clearance_refit
                        )
                    row["retimed_clearance_repair_maximum_m"] = float(
                        retimed_clearance_lift
                    )
                    row["retimed_motion_metrics"] = _retimed_motion_metrics(
                        motion
                    )
                    row["retimed_collision_audit"] = collision.to_dict()
                    row["collision_audit"] = collision.to_dict()
                    if not collision.accepted:
                        debug_path = destination / "rejected_motion_debug.npz"
                        _save_motion(
                            debug_path,
                            motion,
                            mode="bones_side_on_retimed_collision_rejected_debug",
                            extras=extras,
                        )
                        row["rejected_motion_debug"] = str(
                            debug_path.resolve()
                        )
                        raise ValueError(
                            "retimed lateral gait failed collision gate"
                        )
                elif not collision.accepted:
                    raise ValueError(
                        "sole-refitted lateral gait failed collision gate"
                    )
                final_metrics = _retimed_motion_metrics(motion)
                row["retimed_motion_metrics"] = final_metrics
                if (
                    final_metrics["maximum_joint_step_rad"] > 0.20
                    or final_metrics["maximum_root_translation_step_m"] > 0.035
                    or final_metrics["maximum_root_rotation_step_rad"] > 0.080
                    or final_metrics["maximum_root_acceleration_m_s2"] > 30.0
                ):
                    debug_path = destination / "rejected_motion_debug.npz"
                    _save_motion(
                        debug_path,
                        motion,
                        mode="bones_side_on_motion_step_rejected_debug",
                        extras=extras,
                    )
                    row["rejected_motion_debug"] = str(debug_path.resolve())
                    raise ValueError(
                        "sole-preserving repair failed final motion-step gate"
                    )
                final_upper_limb = _upper_limb_motion_metrics(
                    motion.joint_position,
                    joint_names,
                    fps=motion.fps,
                )
                row["final_upper_limb_metrics"] = final_upper_limb
                upper_limb_accepted = bool(
                    final_upper_limb["maximum_arm_excursion_rad"]
                    <= MAXIMUM_ARM_EXCURSION_RAD
                    and final_upper_limb["maximum_wrist_excursion_rad"]
                    <= MAXIMUM_WRIST_EXCURSION_RAD
                    and final_upper_limb["maximum_arm_velocity_rad_s"]
                    <= MAXIMUM_ARM_VELOCITY_RAD_S
                    and final_upper_limb["maximum_wrist_velocity_rad_s"]
                    <= MAXIMUM_WRIST_VELOCITY_RAD_S
                    and final_upper_limb[
                        "maximum_joint_rms_arm_velocity_rad_s"
                    ]
                    <= MAXIMUM_ARM_RMS_VELOCITY_RAD_S
                    and final_upper_limb[
                        "maximum_joint_rms_wrist_velocity_rad_s"
                    ]
                    <= MAXIMUM_WRIST_RMS_VELOCITY_RAD_S
                    and final_upper_limb["p99_arm_acceleration_rad_s2"]
                    <= MAXIMUM_ARM_P99_ACCELERATION_RAD_S2
                    and final_upper_limb["p99_wrist_acceleration_rad_s2"]
                    <= MAXIMUM_WRIST_P99_ACCELERATION_RAD_S2
                )
                row["upper_limb_accepted"] = upper_limb_accepted
                if not upper_limb_accepted:
                    debug_path = destination / "rejected_motion_debug.npz"
                    _save_motion(
                        debug_path,
                        motion,
                        mode="bones_side_on_arm_energy_rejected_debug",
                        extras=extras,
                    )
                    row["rejected_motion_debug"] = str(debug_path.resolve())
                    raise ValueError(
                        "lateral gait failed final upper-limb energy gate"
                    )
                contact_audit = _audit_stance_contact_support(
                    motion,
                    extras,
                    adapter=adapter,
                    target_mesh=target_mesh,
                    ground_fallback_height_m=target_ground_height,
                )
                row["stance_contact_audit"] = contact_audit
                if not bool(contact_audit["accepted"]):
                    debug_path = destination / "rejected_motion_debug.npz"
                    _save_motion(
                        debug_path,
                        motion,
                        mode="bones_side_on_hover_rejected_debug",
                        extras=extras,
                    )
                    row["rejected_motion_debug"] = str(debug_path.resolve())
                    raise ValueError(
                        "lateral gait failed final no-hover stance-contact gate"
                    )
                extras["bones_source_frame"] = np.asarray(
                    [value.source_frame for value in motion.provenance],
                    dtype=np.int64,
                )
                extras["body_to_travel_angle_deg"] = np.full(
                    len(motion.root_position_world),
                    float(source_metadata["body_to_travel_angle_deg"]),
                    dtype=np.float32,
                )
                _save_motion(
                    destination / "motion.npz",
                    motion,
                    mode="bones_side_on",
                    extras=extras,
                )
                row["motion"] = str((destination / "motion.npz").resolve())
                row["status"] = "pending_dense_visual_review"
            except (ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            (destination / "report.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            rows.append(row)
            print(
                f"BONES_SIDE_ON label={label} status={row['status']}"
                + ("" if "error" not in row else f" error={row['error']}"),
                flush=True,
            )
    result = {
        "schema": "bones-side-on-stair-pilots/v1",
        "archive": str(archive_path),
        "pilot_count": len(rows),
        "accepted_count": sum(
            row["status"] == "pending_dense_visual_review" for row in rows
        ),
        "pilots": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-index", type=int, action="append", required=True)
    parser.add_argument(
        "--source-motion",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--exit-margin-m", type=float, default=0.75)
    parser.add_argument("--swing-clearance-m", type=float, default=0.10)
    parser.add_argument("--temporal-scale", type=float, default=1.35)
    parser.add_argument("--source-start-frame", type=int)
    parser.add_argument(
        "--pelvis-height-smoothing-frames", type=float, default=6.0
    )
    parser.add_argument(
        "--root-anchor-smoothing-frames", type=float, default=1.0
    )
    parser.add_argument(
        "--root-anchor-vertical-limit-m", type=float, default=0.0
    )
    parser.add_argument("--arm-swing-scale", type=float, default=0.35)
    parser.add_argument("--wrist-swing-scale", type=float, default=0.10)
    parser.add_argument(
        "--arm-smoothing-sigma-frames", type=float, default=2.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.source_motion:
        arguments.source_motion = list(DEFAULT_SOURCES)
    if arguments.exit_margin_m < 0.0:
        parser.error("--exit-margin-m must be nonnegative")
    if arguments.swing_clearance_m <= 0.0:
        parser.error("--swing-clearance-m must be positive")
    if arguments.temporal_scale < 1.0:
        parser.error("--temporal-scale must be at least one")
    if (
        arguments.source_start_frame is not None
        and arguments.source_start_frame < 0
    ):
        parser.error("--source-start-frame must be nonnegative")
    if arguments.pelvis_height_smoothing_frames < 0.0:
        parser.error("--pelvis-height-smoothing-frames must be nonnegative")
    if arguments.root_anchor_smoothing_frames <= 0.0:
        parser.error("--root-anchor-smoothing-frames must be positive")
    if arguments.root_anchor_vertical_limit_m < 0.0:
        parser.error("--root-anchor-vertical-limit-m must be nonnegative")
    if not 0.0 <= arguments.arm_swing_scale <= 1.0:
        parser.error("--arm-swing-scale must lie in [0,1]")
    if not 0.0 <= arguments.wrist_swing_scale <= 1.0:
        parser.error("--wrist-swing-scale must lie in [0,1]")
    if arguments.arm_smoothing_sigma_frames < 0.0:
        parser.error("--arm-smoothing-sigma-frames must be nonnegative")
    result = build(arguments)
    print(
        json.dumps(
            {
                "pilot_count": result["pilot_count"],
                "accepted_count": result["accepted_count"],
                "manifest": str(
                    arguments.output.expanduser().resolve() / "manifest.json"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
