"""Apply stateful exact-foot contact IK to a continuous terrain trace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import traceback

import numpy as np

from .build_bones_side_on_stair_pilots import (
    _attenuate_upper_limb_motion,
    _audit_stance_contact_support,
    _repair_stance_foot_mesh_clearance,
    _repair_swing_foot_clearance,
    _retimed_motion_metrics,
)
from .build_motionbricks_contact_terrain_pilots import (
    _audit_partial_rigid_support_contact,
    _plan_and_apply_fixed_terrain_footholds,
    _refit_motion_to_sole_targets,
    _remove_short_stance_runs,
    _repair_stance_reach_with_pelvis_tilt,
    _repair_stance_reach_with_root_lowering,
    _smoothly_subdivide_motion_steps,
)
from .build_stairs500_omnidirectional_pilots import _fill_short_stance_gaps
from .canonical_terrain_matcher import RegularGridHeightField
from .render_continuous_terrain_matching import DEFAULT_MODEL
from .terrain_oracle.canonical import ISAACLAB_JOINT_NAMES
from .terrain_oracle.canonical import CanonicalTerrainMesh
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stair_fragment_reconstruction import (
    _smooth_swing_route_adjustments,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


def _load_field(path: Path) -> RegularGridHeightField:
    with np.load(path, allow_pickle=False) as data:
        return RegularGridHeightField(
            np.asarray(data["height"], dtype=np.float64),
            valid=np.asarray(data["valid"], dtype=np.bool_),
            spacing_m=tuple(float(value) for value in data["spacing_m"]),
            origin_xy=tuple(float(value) for value in data["origin_xy"]),
        )


def _maximum_root_acceleration_m_s2(
    root_position_world: np.ndarray, *, fps: float
) -> float:
    root = np.asarray(root_position_world, dtype=np.float64)
    if len(root) < 3:
        return 0.0
    acceleration = np.diff(root, n=2, axis=0) * float(fps) ** 2
    return float(np.max(np.linalg.norm(acceleration, axis=1)))


def _smooth_root_positions(
    root_position_world: np.ndarray,
    *,
    fps: float,
    target_maximum_acceleration_m_s2: float = 25.0,
    maximum_sigma_frames: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove retrieval-boundary velocity impulses without flattening gait bob.

    Continuous motion matching already inertializes pose transitions, but its
    independently transformed root can still change velocity in one 20-ms
    interval.  A tiny symmetric Gaussian is applied only when that trace
    exceeds the same scale used by the final kinematic audit.  Stance feet are
    planned and replanted *after* this edit, so the filter cannot create skate.
    """

    root = np.asarray(root_position_world, dtype=np.float64)
    initial = _maximum_root_acceleration_m_s2(root, fps=fps)
    if initial <= float(target_maximum_acceleration_m_s2) or len(root) < 3:
        return root.copy(), {
            "applied": False,
            "sigma_frames": 0.0,
            "initial_maximum_root_acceleration_m_s2": initial,
            "final_maximum_root_acceleration_m_s2": initial,
            "maximum_root_adjustment_m": 0.0,
        }

    best = root.copy()
    best_sigma = 0.0
    final = initial
    for sigma in np.arange(0.75, float(maximum_sigma_frames) + 0.01, 0.25):
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * np.square(coordinate / sigma))
        kernel /= np.sum(kernel)
        candidate = np.empty_like(root)
        for axis in range(3):
            candidate[:, axis] = np.convolve(
                np.pad(root[:, axis], (radius, radius), mode="edge"),
                kernel,
                mode="valid",
            )
        candidate -= candidate[0] - root[0]
        acceleration = _maximum_root_acceleration_m_s2(candidate, fps=fps)
        best = candidate
        best_sigma = float(sigma)
        final = acceleration
        if acceleration <= float(target_maximum_acceleration_m_s2):
            break
    return best, {
        "applied": True,
        "sigma_frames": best_sigma,
        "initial_maximum_root_acceleration_m_s2": initial,
        "final_maximum_root_acceleration_m_s2": final,
        "maximum_root_adjustment_m": float(
            np.max(np.linalg.norm(best - root, axis=1))
        ),
    }


def _rigid_stance_extras_from_trace(
    motion: StitchedMotion,
    contact_after_step: np.ndarray,
    *,
    adapter: _G1FootfallAdapter,
    maximum_stance_speed_mps: float,
    minimum_stance_run_frames: int,
    target_height_field: RegularGridHeightField | None = None,
    source_sole_speed_mps: np.ndarray | None = None,
    source_sole_height_above_ground_m: np.ndarray | None = None,
    maximum_source_stance_speed_mps: float = 0.05,
    maximum_source_sole_ground_clearance_m: float = 0.015,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Recover conservative plant cores from matched source contact labels."""

    frame_count = len(motion.root_position_world)
    contact = np.asarray(contact_after_step, dtype=np.bool_)
    if contact.shape != (frame_count - 1, 2):
        raise ValueError("continuous trace contact must have shape [T-1,2]")
    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    sphere_centres = np.empty((frame_count, 2, 4, 3), dtype=np.float64)
    sole_centres = np.empty((frame_count, 2, 3), dtype=np.float64)
    for frame in range(frame_count):
        feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        sphere_centres[frame] = feet
        sole_centres[frame] = np.mean(feet, axis=1)
    stance_label = np.zeros((frame_count, 2), dtype=np.bool_)
    stance_label[1:] = contact
    stance_label[0] = contact[0]
    sole_speed = np.linalg.norm(
        np.gradient(sole_centres, axis=0) * float(motion.fps), axis=2
    )
    stance = _remove_short_stance_runs(
        stance_label & (sole_speed <= float(maximum_stance_speed_mps)),
        minimum_run_frames=int(minimum_stance_run_frames),
    )
    source_stance_frame_foot_count = int(np.count_nonzero(stance))
    source_stance_span_count = int(
        sum(
            np.count_nonzero(values & np.concatenate(([True], ~values[:-1])))
            for values in stance.T
        )
    )
    source_kinematic_fallback_used = False
    source_kinematic_stance = np.zeros_like(stance)
    source_ground_height_stance_used = False
    if source_sole_height_above_ground_m is not None:
        native_height = np.asarray(
            source_sole_height_above_ground_m, dtype=np.float64
        )
        if native_height.shape == (frame_count - 1, 2):
            native_height = np.concatenate(
                (native_height[:1], native_height), axis=0
            )
        if native_height.shape != (frame_count, 2):
            raise ValueError(
                "source sole height must have shape [T-1,2] or [T,2]"
            )
        # MotionBricks contact bits and world-space sole speed become noisy
        # after anti-scissor pose cleanup.  The source was generated over a
        # known flat floor, so sole height is the authoritative gait phase:
        # it recovers every alternating plant even if the authored plant has
        # a little horizontal skate.  The terrain solve removes that skate by
        # holding each accepted sole target rigid.
        source_kinematic_stance = _remove_short_stance_runs(
            native_height
            <= float(maximum_source_sole_ground_clearance_m),
            minimum_run_frames=int(minimum_stance_run_frames),
        )
        if np.any(source_kinematic_stance):
            stance = source_kinematic_stance
            source_kinematic_fallback_used = True
            source_ground_height_stance_used = True
    elif source_sole_speed_mps is not None:
        native_speed = np.asarray(source_sole_speed_mps, dtype=np.float64)
        if native_speed.shape == (frame_count - 1, 2):
            native_speed = np.concatenate((native_speed[:1], native_speed), axis=0)
        if native_speed.shape != (frame_count, 2):
            raise ValueError(
                "source sole speed must have shape [T-1,2] or [T,2]"
            )
        source_kinematic_stance = _remove_short_stance_runs(
            native_speed <= float(maximum_source_stance_speed_mps),
            minimum_run_frames=int(minimum_stance_run_frames),
        )
        if np.any(source_kinematic_stance):
            stance = _fill_short_stance_gaps(
                stance | source_kinematic_stance, maximum_gap_frames=2
            )
            stance = _remove_short_stance_runs(
                stance,
                minimum_run_frames=int(minimum_stance_run_frames),
            )
            source_kinematic_fallback_used = bool(
                np.any(source_kinematic_stance & ~stance_label)
            )
    current_stance_frame_foot_count = int(np.count_nonzero(stance))
    current_stance_span_count = int(
        sum(
            np.count_nonzero(values & np.concatenate(([True], ~values[:-1])))
            for values in stance.T
        )
    )
    fallback_used = bool(
        target_height_field is not None
        and (
            current_stance_span_count < 2
            or current_stance_frame_foot_count
            < max(8, int(round(0.05 * frame_count)))
        )
    )
    if fallback_used:
        radii = np.stack(adapter.sole_sphere_radii()).astype(np.float64)
        height, _normal, hit = target_height_field.sample(
            sphere_centres[..., :2]
        )
        clearance = sphere_centres[..., 2] - radii[None] - height
        near_surface = (
            np.all(hit, axis=2)
            & (np.min(np.abs(clearance), axis=2) <= 0.060)
            & (np.min(clearance, axis=2) >= -0.120)
        )
        inferred = near_surface & (
            sole_speed <= float(maximum_stance_speed_mps)
        )
        stance = _fill_short_stance_gaps(
            stance | inferred, maximum_gap_frames=2
        )
        stance = _remove_short_stance_runs(
            stance,
            minimum_run_frames=int(minimum_stance_run_frames),
        )
    if not np.any(stance):
        raise ValueError("continuous trace has no conservative stance cores")
    zeros = np.zeros(frame_count, dtype=np.float32)
    root_progress = roots[:, 0] - roots[0, 0]
    scale = float(np.ptp(root_progress))
    if scale > 1.0e-8:
        root_progress /= scale
    else:
        root_progress.fill(0.0)
    extras = {
        "path_lateral_offset_m": zeros.copy(),
        "path_yaw_offset_rad": zeros.copy(),
        "pose_yaw_offset_rad": zeros.copy(),
        "foot_yaw_offset_rad": zeros.copy(),
        "facing_yaw_offset_rad": zeros.copy(),
        "path_normalized_progress": np.asarray(root_progress, dtype=np.float32),
        "intended_root_position_world": np.asarray(roots, dtype=np.float32),
        "authored_stance_mask": stance,
        "target_stance_support_point_count": np.where(
            stance, 4, 0
        ).astype(np.int16),
        "per_frame_sole_target_error_m": zeros.copy(),
        "per_frame_sole_target_error_by_foot_m": np.zeros(
            (frame_count, 2), dtype=np.float32
        ),
        "per_frame_joint_correction_rad": zeros.copy(),
        "target_sole_center_world": np.asarray(
            sole_centres, dtype=np.float32
        ),
        "target_sole_points_world": np.asarray(
            sphere_centres, dtype=np.float32
        ),
        "nominal_sole_center_world": np.asarray(
            sole_centres, dtype=np.float32
        ),
        "adapted_sole_center_world": np.asarray(
            sole_centres, dtype=np.float32
        ),
        "terrain_root_anchor_shift_world": np.zeros(
            (frame_count, 3), dtype=np.float32
        ),
        "bracketed_swing_mask": np.zeros(
            (frame_count, 2), dtype=np.bool_
        ),
        "per_frame_swing_route_adjustment": np.zeros(
            (frame_count, 2, 3), dtype=np.float32
        ),
    }
    return extras, {
        "stance_frame_foot_count": int(np.count_nonzero(stance)),
        "stance_span_count": int(
            sum(
                np.count_nonzero(
                    values & np.concatenate(([True], ~values[:-1]))
                )
                for values in stance.T
            )
        ),
        "maximum_stance_speed_mps": float(
            np.max(sole_speed[stance]) if np.any(stance) else 0.0
        ),
        "maximum_stance_speed_threshold_mps": float(
            maximum_stance_speed_mps
        ),
        "minimum_stance_run_frames": int(minimum_stance_run_frames),
        "source_stance_frame_foot_count": source_stance_frame_foot_count,
        "source_stance_span_count": source_stance_span_count,
        "geometry_fallback_used": fallback_used,
        "source_kinematic_fallback_used": source_kinematic_fallback_used,
        "source_ground_height_stance_used": (
            source_ground_height_stance_used
        ),
        "source_kinematic_stance_frame_foot_count": int(
            np.count_nonzero(source_kinematic_stance)
        ),
        "maximum_source_stance_speed_mps": float(
            maximum_source_stance_speed_mps
        ),
        "maximum_source_sole_ground_clearance_m": float(
            maximum_source_sole_ground_clearance_m
        ),
    }


def _preflex_swing_legs(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    gain_rad_per_m: float,
    maximum_flex_rad: float,
    anticipation_frames: int = 12,
) -> tuple[StitchedMotion, dict[str, object]]:
    """Bias a rising swing toward the natural stair knee branch.

    Sole-only IK has two local choices when a flat gait meets a riser: flex
    the knee before toe-off, or keep the leg long and lift through hip/ankle
    motion.  The latter can hit the riser with the shin even when the foot
    itself is clear.  Anticipate the already-computed swing-height arc and
    seed the former branch; exact sole and full-body audits remain decisive.
    """

    gain = float(gain_rad_per_m)
    cap = float(maximum_flex_rad)
    if gain <= 0.0 or cap <= 0.0:
        return motion, {"applied": False, "maximum_knee_preflex_rad": 0.0}
    route = np.asarray(
        extras.get(
            "per_frame_swing_route_adjustment",
            np.zeros((len(motion.root_position_world), 2, 3)),
        ),
        dtype=np.float64,
    )
    if route.shape != (len(motion.root_position_world), 2, 3):
        raise ValueError("swing route adjustment must have shape [T,2,3]")
    repair_lift = np.asarray(
        extras.get("swing_clearance_repair_m", np.zeros(route.shape[:2])),
        dtype=np.float64,
    )
    if repair_lift.shape != route.shape[:2]:
        raise ValueError("swing clearance repair must have shape [T,2]")
    lift = np.maximum(np.maximum(route[:, :, 2], 0.0), repair_lift)
    anticipated = np.zeros_like(lift)
    horizon = max(0, int(anticipation_frames))
    for frame in range(len(lift)):
        anticipated[frame] = np.max(
            lift[frame : min(len(lift), frame + horizon + 1)], axis=0
        )
    flex = np.clip(gain * anticipated, 0.0, cap)
    stance = np.asarray(
        extras.get("authored_stance_mask", np.zeros_like(flex, dtype=bool)),
        dtype=bool,
    )
    if stance.shape != flex.shape:
        raise ValueError("authored stance mask must have shape [T,2]")
    flex[stance] = 0.0
    joints = np.asarray(motion.joint_position, dtype=np.float64).copy()
    for foot, (hip_pitch, knee, ankle_pitch) in enumerate(
        ((0, 9, 13), (1, 10, 14))
    ):
        amount = flex[:, foot]
        joints[:, hip_pitch] -= 0.55 * amount
        joints[:, knee] += amount
        joints[:, ankle_pitch] -= 0.45 * amount
    maximum = float(np.max(flex, initial=0.0))
    return (
        replace(motion, joint_position=np.asarray(joints, dtype=np.float32)),
        {
            "applied": bool(maximum > 0.0),
            "gain_rad_per_m": gain,
            "maximum_knee_preflex_rad": maximum,
            "maximum_hip_pitch_preflex_rad": 0.55 * maximum,
            "maximum_ankle_pitch_preflex_rad": 0.45 * maximum,
            "anticipation_frames": horizon,
        },
    )


def _crop_processed_motion(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    *,
    start: int,
    stop: int,
) -> tuple[StitchedMotion, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Discard hidden solve context only after plants and flights are fitted."""

    frame_count = len(motion.root_position_world)
    if start < 0 or stop > frame_count or stop - start < 3:
        raise ValueError("processed-motion crop is invalid")
    cropped_motion = StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(
            motion.root_position_world[start:stop], dtype=np.float32
        ),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz[start:stop], dtype=np.float32
        ),
        joint_position=np.asarray(
            motion.joint_position[start:stop], dtype=np.float32
        ),
        provenance=motion.provenance[start:stop],
        seam_indices=tuple(
            int(index - start)
            for index in motion.seam_indices
            if start <= index < stop
        ),
    )

    def crop_mapping(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key, value in values.items():
            item = np.asarray(value)
            if item.ndim and item.shape[0] == frame_count:
                result[key] = item[start:stop]
            elif item.ndim and item.shape[0] == frame_count - 1:
                result[key] = item[start : stop - 1]
            else:
                result[key] = item.copy()
        return result

    return cropped_motion, crop_mapping(extras), crop_mapping(arrays)


def _resample_trace_arrays(
    arrays: dict[str, np.ndarray],
    source_coordinate: np.ndarray,
    *,
    original_frame_count: int,
) -> dict[str, np.ndarray]:
    """Carry frame and interval labels through a local motion time warp."""

    coordinate = np.asarray(source_coordinate, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or len(coordinate) < 2
        or not np.isfinite(coordinate).all()
        or np.any(np.diff(coordinate) < 0.0)
        or coordinate[0] < -1.0e-8
        or coordinate[-1] > float(original_frame_count - 1) + 1.0e-8
    ):
        raise ValueError("time-warp source coordinate is invalid")
    old_frame = np.arange(original_frame_count, dtype=np.float64)
    interval_index = np.clip(
        np.floor(0.5 * (coordinate[:-1] + coordinate[1:])).astype(np.int64),
        0,
        original_frame_count - 2,
    )
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        item = np.asarray(value)
        if item.ndim and item.shape[0] == original_frame_count:
            if item.dtype.kind in "fc":
                flat = item.reshape(original_frame_count, -1)
                result[key] = np.asarray(
                    np.stack(
                        [
                            np.interp(coordinate, old_frame, flat[:, column])
                            for column in range(flat.shape[1])
                        ],
                        axis=1,
                    ).reshape((len(coordinate),) + item.shape[1:]),
                    dtype=item.dtype,
                )
            else:
                nearest = np.clip(
                    np.rint(coordinate).astype(np.int64),
                    0,
                    original_frame_count - 1,
                )
                result[key] = item[nearest]
        elif item.ndim and item.shape[0] == original_frame_count - 1:
            result[key] = item[interval_index]
        else:
            result[key] = item.copy()
    return result


def _terrain_index_near_motion(
    terrain: TerrainMeshIndex,
    motion: StitchedMotion,
    *,
    horizontal_margin_m: float = 1.25,
) -> tuple[TerrainMeshIndex, dict[str, object]]:
    """Keep the exact terrain faces that can physically reach this motion.

    A regular-grid course can contain tens of thousands of triangle prisms far
    outside a short evaluation window.  MuJoCo otherwise tests all of them on
    every frame.  The G1 fits well inside the 1.25-m root corridor; selecting
    by triangle AABB is conservative and leaves every nearby triangle exactly
    unchanged.
    """

    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    margin = float(horizontal_margin_m)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("terrain audit margin must be positive")
    triangles = np.asarray(terrain.triangles_world, dtype=np.float64)
    lower = np.min(roots[:, :2], axis=0) - margin
    upper = np.max(roots[:, :2], axis=0) + margin
    triangle_lower = np.min(triangles[..., :2], axis=1)
    triangle_upper = np.max(triangles[..., :2], axis=1)
    selected = np.all(
        (triangle_upper >= lower[None]) & (triangle_lower <= upper[None]),
        axis=1,
    )
    kept = triangles[selected]
    if not len(kept):
        raise ValueError("motion audit corridor contains no terrain faces")
    vertices = kept.reshape((-1, 3)).astype(np.float32)
    faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))
    mesh = CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=faces,
        valid_faces=np.ones(len(faces), dtype=np.bool_),
        source_asset_sha256=terrain.mesh.source_asset_sha256,
    )
    local = TerrainMeshIndex(
        mesh,
        RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        ),
    )
    return local, {
        "horizontal_margin_m": margin,
        "root_xy_lower_m": lower.tolist(),
        "root_xy_upper_m": upper.tolist(),
        "source_face_count": int(len(triangles)),
        "selected_face_count": int(len(kept)),
        "selected_face_fraction": float(len(kept) / len(triangles)),
    }


def _minimal_vertical_swing_clearance(
    extras: dict[str, np.ndarray],
    *,
    field: RegularGridHeightField,
    adapter: _G1FootfallAdapter,
    surface_clearance_m: float = 0.008,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Preserve authored flight XY and lift only where terrain demands it."""

    stance = np.asarray(extras["authored_stance_mask"], dtype=bool)
    nominal = np.asarray(
        extras.get(
            "source_anchored_sole_points_world",
            extras["planned_nominal_sole_points_world"],
        ),
        dtype=np.float64,
    )
    planted = np.asarray(
        extras["target_sole_points_world"], dtype=np.float64
    )
    if nominal.ndim != 4 or nominal.shape != planted.shape:
        raise ValueError("minimal swing clearance requires paired sole targets")
    if stance.shape != nominal.shape[:2]:
        raise ValueError("minimal swing clearance stance shape is invalid")
    height, _normal, hit = field.sample(nominal[..., :2])
    if not np.all(hit):
        raise ValueError("authored swing leaves the target height field")
    radii = np.stack(adapter.sole_sphere_radii()).astype(np.float64)
    required_by_probe = (
        height
        + radii[None]
        + float(surface_clearance_m)
        - nominal[..., 2]
    )
    required_lift = np.maximum(
        0.0, np.max(required_by_probe, axis=2)
    )
    required_lift[stance] = 0.0
    raw_adjustment = np.zeros((*stance.shape, 3), dtype=np.float64)
    raw_adjustment[..., 2] = required_lift
    adjustment = _smooth_swing_route_adjustments(
        raw_adjustment, stance, decay_frames=4.0
    )
    target = nominal + adjustment[:, :, None, :]
    target[stance] = planted[stance]
    active = (~stance) & (adjustment[..., 2] > 1.0e-6)
    updated = dict(extras)
    updated["target_sole_points_world"] = np.asarray(target, dtype=np.float32)
    updated["target_sole_center_world"] = np.asarray(
        np.mean(target, axis=2), dtype=np.float32
    )
    updated["per_frame_swing_route_adjustment"] = np.asarray(
        adjustment, dtype=np.float32
    )
    updated["swing_clearance_target_mask"] = active
    # Retain the old field name in saved diagnostics, but it now means an
    # explicitly constrained minimal-clearance flight rather than every
    # anchor-bracketed swing.
    updated["bracketed_swing_mask"] = active
    return updated, {
        "active_frame_foot_count": int(np.count_nonzero(active)),
        "maximum_required_vertical_lift_m": float(np.max(required_lift)),
        "maximum_smoothed_vertical_lift_m": float(
            np.max(adjustment[..., 2])
        ),
        "maximum_planar_adjustment_m": float(
            np.max(np.linalg.norm(adjustment[..., :2], axis=2))
        ),
        "surface_clearance_m": float(surface_clearance_m),
    }


def postprocess(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.trace, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    root = np.asarray(arrays["root_position_world"], dtype=np.float64)
    quaternion = np.asarray(
        arrays["root_quaternion_world_wxyz"], dtype=np.float64
    )
    authored = np.asarray(arrays["joint_position"], dtype=np.float64)
    contact = np.asarray(arrays["contact"], dtype=np.bool_)
    selected_clip = np.asarray(
        arrays.get("selected_clip_index", np.zeros(len(contact))), dtype=np.int64
    )
    selected_frame = np.asarray(
        arrays.get("selected_source_frame", np.arange(len(contact))), dtype=np.int64
    )
    if len(root) != len(authored) or len(contact) != len(authored) - 1:
        raise ValueError("trace root/joint/contact lengths disagree")

    field = _load_field(args.terrain)
    adapter = _G1FootfallAdapter(
        args.model,
        ISAACLAB_JOINT_NAMES,
        maximum_joint_correction_rad=args.maximum_joint_correction_rad,
        target_tolerance_m=args.target_tolerance_m,
    )
    radii = adapter.sole_sphere_radii()
    corrected = authored.copy()
    foot_support = np.zeros((len(authored) - 1, 2, 4, 3), dtype=np.float64)
    latched: list[np.ndarray | None] = [None, None]
    previous_contact = np.zeros(2, dtype=np.bool_)
    corrections: list[float] = []
    target_errors: list[float] = []

    for index in range(1, len(authored)):
        source_continuous = bool(
            index <= 1
            or (
                selected_clip[index - 1] == selected_clip[index - 2]
                and selected_frame[index - 1] == selected_frame[index - 2] + 1
            )
        )
        authored_centres = adapter.sole_positions_for_pose(
            root_position=root[index],
            root_quaternion_wxyz=quaternion[index],
            joints=authored[index],
        )
        targets: list[np.ndarray] = []
        for foot in range(2):
            current = np.asarray(authored_centres[foot], dtype=np.float64)
            active = bool(contact[index - 1, foot])
            if active and not args.lock_horizontal:
                target = current.copy()
                terrain_height, _normal, hit = field.sample(target[:, :2])
                if not np.all(hit):
                    raise ValueError("contact target leaves terrain")
                target[:, 2] = terrain_height + radii[foot]
                latched[foot] = None
            elif active and (
                not previous_contact[foot]
                or latched[foot] is None
                or not source_continuous
            ):
                target = current.copy()
                terrain_height, _normal, hit = field.sample(target[:, :2])
                if not np.all(hit):
                    raise ValueError("contact target leaves terrain")
                target[:, 2] = terrain_height + radii[foot]
                latched[foot] = target
            elif active:
                target = np.asarray(latched[foot], dtype=np.float64).copy()
                # Contact reconstruction can remain asserted for a long slow
                # stance.  Preserve the lock, but let an unreachable target
                # creep toward the authored sole by a tightly bounded amount
                # instead of forcing the leg into a singular configuration.
                target[:, :2] += np.clip(
                    current[:, :2] - target[:, :2],
                    -args.maximum_target_creep_m_per_frame,
                    args.maximum_target_creep_m_per_frame,
                )
                terrain_height, _normal, hit = field.sample(target[:, :2])
                if not np.all(hit):
                    raise ValueError("latched contact target leaves terrain")
                target[:, 2] = terrain_height + radii[foot]
                latched[foot] = target
            else:
                latched[foot] = None
                target = current.copy()
                terrain_height, _normal, hit = field.sample(target[:, :2])
                if np.any(hit):
                    required = np.max(
                        terrain_height[np.asarray(hit)]
                        + radii[foot][np.asarray(hit)]
                        + args.minimum_swing_clearance_m
                        - target[np.asarray(hit), 2]
                    )
                    if required > 0.0:
                        target[:, 2] += float(required)
            targets.append(target)

        joints, _correction, _error = adapter.adapt_to_targets(
            root_position=root[index],
            root_quaternion_wxyz=quaternion[index],
            authored_joints=authored[index],
            sole_targets_world=targets,
            initial_joints=corrected[index - 1],
        )
        lower = np.maximum(
            authored[index] - args.maximum_joint_correction_rad,
            corrected[index - 1] - args.maximum_joint_step_rad,
        )
        upper = np.minimum(
            authored[index] + args.maximum_joint_correction_rad,
            corrected[index - 1] + args.maximum_joint_step_rad,
        )
        joints = np.clip(joints, lower, upper)
        incompatible = lower > upper
        if np.any(incompatible):
            joints[incompatible] = np.clip(
                authored[index, incompatible],
                corrected[index - 1, incompatible] - 0.25,
                corrected[index - 1, incompatible] + 0.25,
            )
        corrected[index] = joints
        corrections.append(float(np.max(np.abs(joints - authored[index]))))
        centres = adapter.sole_positions_for_pose(
            root_position=root[index],
            root_quaternion_wxyz=quaternion[index],
            joints=joints,
        )
        for foot in range(2):
            foot_support[index - 1, foot] = centres[foot]
            foot_support[index - 1, foot, :, 2] -= radii[foot]
        target_errors.append(
            max(
                float(np.max(np.linalg.norm(centres[foot] - targets[foot], axis=1)))
                for foot in range(2)
            )
        )
        previous_contact = contact[index - 1].copy()

    terrain_height, _normal, hit = field.sample(foot_support[..., :2])
    clearance = foot_support[..., 2] - terrain_height
    support_mask = contact[:, :, None] & hit
    support_clearance = clearance[support_mask]
    mean_support = np.mean(foot_support, axis=2)
    drift = np.linalg.norm(np.diff(mean_support[..., :2], axis=0), axis=-1)
    planted = drift[contact[1:] & contact[:-1]]
    joint_step = np.max(np.abs(np.diff(corrected, axis=0)), axis=1)

    def maximum(value: np.ndarray) -> float:
        return float(np.max(value)) if value.size else 0.0

    metrics = {
        "maximum_contact_penetration_m": max(
            0.0, maximum(-support_clearance)
        ),
        "maximum_contact_hover_m": max(0.0, maximum(support_clearance)),
        "maximum_planted_foot_drift_m_per_frame": maximum(planted),
        "maximum_joint_step_rad": maximum(joint_step),
        "maximum_ik_correction_rad": maximum(np.asarray(corrections)),
        "maximum_ik_target_error_m": maximum(np.asarray(target_errors)),
        "terrain_query_hit_fraction": float(np.mean(hit)),
    }
    gates = {
        "contact_penetration_le_5mm": metrics[
            "maximum_contact_penetration_m"
        ] <= 0.005,
        "planted_drift_le_10mm": metrics[
            "maximum_planted_foot_drift_m_per_frame"
        ] <= 0.010,
        "joint_step_le_0p25rad": metrics["maximum_joint_step_rad"] <= 0.25,
        "ik_target_error_le_5mm": metrics["maximum_ik_target_error_m"] <= 0.005,
    }
    output_arrays = dict(arrays)
    output_arrays["joint_position"] = corrected.astype(np.float32)
    output_arrays["foot_support_points_world"] = foot_support.astype(np.float32)
    output_arrays["foot_position_world"] = mean_support.astype(np.float32)
    output_arrays["ik_correction_rad"] = np.asarray(corrections, dtype=np.float32)
    output_arrays["ik_target_error_m"] = np.asarray(target_errors, dtype=np.float32)
    np.savez_compressed(args.output, **output_arrays)
    return {
        "schema": "continuous-terrain-contact-postprocess/v1",
        "input_trace": str(args.trace.resolve()),
        "output_trace": str(args.output.resolve()),
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def postprocess_rigid(args: argparse.Namespace) -> dict[str, object]:
    """Apply sequence-consistent rigid plants to matched natural kinematics."""

    with np.load(args.trace, allow_pickle=False) as data:
        original = {name: np.asarray(data[name]) for name in data.files}
    full_frame_count = len(original["root_position_world"])
    requested_start = int(args.start_frame)
    requested_stop = (
        full_frame_count
        if args.stop_frame is None
        else min(full_frame_count, int(args.stop_frame))
    )
    if requested_start < 0 or requested_stop - requested_start < 3:
        raise ValueError("rigid postprocess frame range is invalid")
    context = int(args.context_frames)
    if context < 0:
        raise ValueError("context frames must be nonnegative")
    pre_context = (
        context
        if args.pre_context_frames is None
        else int(args.pre_context_frames)
    )
    post_context = (
        context
        if args.post_context_frames is None
        else int(args.post_context_frames)
    )
    if pre_context < 0 or post_context < 0:
        raise ValueError("pre/post context frames must be nonnegative")
    start = max(0, requested_start - pre_context)
    stop = min(full_frame_count, requested_stop + post_context)
    arrays: dict[str, np.ndarray] = {}
    for key, value in original.items():
        if value.ndim and value.shape[0] == full_frame_count:
            arrays[key] = value[start:stop]
        elif value.ndim and value.shape[0] == full_frame_count - 1:
            arrays[key] = value[start : stop - 1]
        else:
            arrays[key] = value.copy()

    root = np.asarray(arrays["root_position_world"], dtype=np.float64)
    quaternion = np.asarray(
        arrays["root_quaternion_world_wxyz"], dtype=np.float64
    )
    joints = np.asarray(arrays["joint_position"], dtype=np.float64)
    contact = np.asarray(arrays["contact"], dtype=np.bool_)
    root, root_filter = _smooth_root_positions(
        root,
        fps=50.0,
        target_maximum_acceleration_m_s2=(
            args.target_maximum_root_acceleration_m_s2
        ),
    )

    selected_clip = np.asarray(
        arrays.get("selected_clip_index", np.zeros(len(contact))),
        dtype=np.int64,
    )
    selected_frame = np.asarray(
        arrays.get("selected_source_frame", np.arange(len(contact))),
        dtype=np.int64,
    )
    provenance = tuple(
        FrameProvenance(
            int(selected_clip[min(max(frame - 1, 0), len(selected_clip) - 1)]),
            int(selected_frame[min(max(frame - 1, 0), len(selected_frame) - 1)]),
            "continuous_terrain_matched_source",
        )
        for frame in range(len(root))
    )
    motion = StitchedMotion(
        fps=50.0,
        root_position_world=np.asarray(root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternion, dtype=np.float32),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=provenance,
        seam_indices=(),
    )
    field = _load_field(args.terrain)
    adapter = _G1FootfallAdapter(
        args.model,
        ISAACLAB_JOINT_NAMES,
        maximum_joint_correction_rad=args.maximum_joint_correction_rad,
        target_tolerance_m=args.target_tolerance_m,
    )
    extras, stance_report = _rigid_stance_extras_from_trace(
        motion,
        contact,
        adapter=adapter,
        maximum_stance_speed_mps=args.rigid_maximum_stance_speed_mps,
        minimum_stance_run_frames=args.rigid_minimum_stance_run_frames,
        target_height_field=field,
        source_sole_speed_mps=arrays.get("source_sole_speed_mps"),
        source_sole_height_above_ground_m=arrays.get(
            "source_sole_height_above_ground_m"
        ),
        maximum_source_stance_speed_mps=(
            args.rigid_maximum_source_stance_speed_mps
        ),
        maximum_source_sole_ground_clearance_m=(
            args.rigid_maximum_source_sole_ground_clearance_m
        ),
    )
    # Candidate support queries need only geometry within reach of this solve
    # window.  The 1.25 m corridor is an order of magnitude wider than the
    # bounded 12 cm foothold search and includes the complete G1 body/sole
    # envelope.  Cropping here changes no accepted geometry; it avoids
    # raycasting every candidate against unrelated triangles at the far end of
    # a long procedural course.  Final collision is still independently
    # cropped around and audited on the resulting motion below.
    planning_mesh, planning_region = _terrain_index_near_motion(
        field.index, motion
    )
    motion, extras, foothold_report = _plan_and_apply_fixed_terrain_footholds(
        motion,
        extras,
        adapter=adapter,
        target_mesh=planning_mesh,
        pelvis_planar_smoothing_sigma_frames=(
            args.rigid_pelvis_planar_smoothing_sigma_frames
        ),
        allow_partial_rigid_support=(
            not args.rigid_require_full_sole_support
        ),
        maximum_sole_tilt_adjustment_rad=math.radians(25.0),
        maximum_longitudinal_adjustment_m=(
            args.rigid_maximum_foothold_longitudinal_adjustment_m
        ),
        maximum_lateral_adjustment_m=(
            args.rigid_maximum_foothold_lateral_adjustment_m
        ),
        maximum_yaw_adjustment_rad=math.radians(10.0),
        # Irregular 30--40 cm treads can leave only a narrow whole-foot band
        # between adjacent risers.  Keep the same bounded +/-12 cm authority,
        # but sample it every 3 cm rather than every 6 cm.
        longitudinal_samples=args.rigid_foothold_longitudinal_samples,
        lateral_samples=args.rigid_foothold_lateral_samples,
        yaw_samples=3,
        maximum_pelvis_planar_adjustment_step_m=(
            args.rigid_maximum_pelvis_planar_adjustment_step_m
        ),
        maximum_foothold_height_change_m=(
            args.rigid_maximum_foothold_height_change_m
        ),
        maximum_planar_reach_change_m=(
            args.rigid_maximum_planar_reach_change_m
        ),
        maximum_yaw_change_rad=math.radians(
            args.rigid_maximum_foothold_yaw_change_deg
        ),
    )
    extras, swing_guidance = _minimal_vertical_swing_clearance(
        extras, field=field, adapter=adapter
    )
    motion, swing_knee_preflex = _preflex_swing_legs(
        motion,
        extras,
        gain_rad_per_m=args.rigid_swing_knee_preflex_gain_rad_per_m,
        maximum_flex_rad=args.rigid_swing_knee_preflex_maximum_rad,
    )
    motion, extras, refit = _refit_motion_to_sole_targets(
        motion,
        extras,
        adapter=adapter,
        # Preserve every source flight in XY and solve only the smooth vertical
        # lift proved necessary by the known target surface.
        stance_only=True,
        additional_target_mask=np.asarray(
            extras.get(
                "swing_clearance_target_mask",
                np.zeros((len(motion.root_position_world), 2), dtype=bool),
            ),
            dtype=bool,
        ),
    )
    motion, extras, reach_repair = _repair_stance_reach_with_root_lowering(
        motion,
        extras,
        adapter=adapter,
        target_error_m=0.008,
        maximum_total_lower_m=args.rigid_maximum_stance_reach_root_lower_m,
    )
    if bool(reach_repair["applied"]):
        refit = {
            **refit,
            "maximum_sole_target_error_m": float(
                reach_repair["final_maximum_stance_error_m"]
            ),
        }
        if reach_repair["iterations"]:
            final_iteration = reach_repair["iterations"][-1]
            refit["maximum_joint_step_rad"] = float(
                final_iteration["maximum_joint_step_rad"]
            )
            refit["maximum_joint_delta_rad"] = float(
                final_iteration["maximum_joint_delta_rad"]
            )
    initial_pose_error = float(
        reach_repair.get("final_maximum_stance_error_m", math.inf)
    )
    remaining_stance_error = initial_pose_error
    pose_attempts: list[dict[str, object]] = []
    maximum_pose_iterations = 4
    for _pose_iteration in range(maximum_pose_iterations):
        if remaining_stance_error <= 0.008:
            break
        # A high/near-singular pelvis is normally fixed by the vertical repair
        # above.  If that measured correction cannot improve the exact planted
        # sole residual, share a small bounded roll/pitch adjustment with the
        # pelvis instead of forcing the knee onto a saturated IK branch.  Solve
        # only the worst remaining window so every accepted edit is monotonic.
        candidate_motion, candidate_extras, attempt = (
            _repair_stance_reach_with_pelvis_tilt(
                motion,
                extras,
                adapter=adapter,
                maximum_repair_runs=1,
            )
        )
        pose_attempts.append(attempt)
        if not bool(attempt.get("applied", False)):
            break
        candidate_error = float(
            attempt.get("final_maximum_stance_error_m", math.inf)
        )
        if candidate_error >= remaining_stance_error - 1.0e-4:
            break
        motion, extras = candidate_motion, candidate_extras
        remaining_stance_error = candidate_error
        pose_refit = attempt["refit"]
        refit = {
            **refit,
            "maximum_sole_target_error_m": candidate_error,
            "maximum_joint_step_rad": float(
                pose_refit["maximum_joint_step_rad"]
            ),
            "maximum_joint_delta_rad": float(
                pose_refit["maximum_joint_delta_rad"]
            ),
        }
    stance_pose_repair: dict[str, object] = {
        "applied": bool(
            any(attempt.get("applied", False) for attempt in pose_attempts)
        ),
        "initial_maximum_stance_error_m": initial_pose_error,
        "final_maximum_stance_error_m": remaining_stance_error,
        "iteration_count": len(pose_attempts),
        "iterations": pose_attempts,
    }

    core_start = requested_start - start
    core_stop = core_start + (requested_stop - requested_start)
    motion, extras, arrays = _crop_processed_motion(
        motion,
        extras,
        arrays,
        start=core_start,
        stop=core_stop,
    )
    quiet_joints, upper_limb_attenuation = _attenuate_upper_limb_motion(
        motion.joint_position,
        ISAACLAB_JOINT_NAMES,
        fps=motion.fps,
        arm_swing_scale=args.arm_swing_scale,
        wrist_swing_scale=args.wrist_swing_scale,
        smoothing_sigma_frames=args.arm_smoothing_sigma_frames,
    )
    motion = replace(
        motion, joint_position=np.asarray(quiet_joints, dtype=np.float32)
    )
    audit_mesh, audit_region = _terrain_index_near_motion(
        field.index, motion
    )

    collision = audit_stair_motion_collisions(
        motion,
        model_path=args.model,
        target_mesh=audit_mesh,
        joint_names=ISAACLAB_JOINT_NAMES,
        maximum_foot_penetration_m=0.0051,
        maximum_forbidden_body_penetration_m=0.0,
    )
    repairs: dict[str, object] = {}
    if (
        not args.disable_collision_repairs
        and not collision.accepted
        and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
        and collision.maximum_foot_penetration_m <= 0.012
    ):
        motion, extras, collision, stance_repair = (
            _repair_stance_foot_mesh_clearance(
                motion,
                extras,
                collision,
                adapter=adapter,
                target_mesh=audit_mesh,
                model_path=args.model,
                joint_names=ISAACLAB_JOINT_NAMES,
            )
        )
        repairs["stance_mesh_clearance"] = stance_repair
    if (
        not args.disable_collision_repairs
        and not collision.accepted
        and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
        and collision.maximum_foot_penetration_m <= 0.040
    ):
        motion, extras, collision, swing_repair = _repair_swing_foot_clearance(
            motion,
            extras,
            collision,
            adapter=adapter,
            target_mesh=audit_mesh,
            model_path=args.model,
            joint_names=ISAACLAB_JOINT_NAMES,
            maximum_total_lift_m=0.050,
            maximum_iterations=3,
        )
        repairs["swing_clearance"] = swing_repair
    if (
        not collision.accepted
        and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
        and args.rigid_swing_knee_preflex_gain_rad_per_m > 0.0
        and np.any(
            np.asarray(
                extras.get("swing_clearance_repair_m", ()),
                dtype=np.float64,
            )
            > 0.0
        )
    ):
        # The collision-driven lift above discovers risers that are between
        # queried sole samples and therefore absent from the initial terrain-
        # height arc.  Re-seed only those repaired flight frames toward the
        # flexed-knee branch, then solve the same exact raised sole targets.
        motion, collision_preflex = _preflex_swing_legs(
            motion,
            extras,
            gain_rad_per_m=args.rigid_swing_knee_preflex_gain_rad_per_m,
            maximum_flex_rad=args.rigid_swing_knee_preflex_maximum_rad,
        )
        repair_mask = np.asarray(
            extras["swing_clearance_repair_m"], dtype=np.float64
        ) > 0.0
        motion, extras, collision_preflex_refit = (
            _refit_motion_to_sole_targets(
                motion,
                extras,
                adapter=adapter,
                stance_only=True,
                additional_target_mask=repair_mask,
            )
        )
        collision = audit_stair_motion_collisions(
            motion,
            model_path=args.model,
            target_mesh=audit_mesh,
            joint_names=ISAACLAB_JOINT_NAMES,
            maximum_foot_penetration_m=0.0051,
            maximum_forbidden_body_penetration_m=0.0,
        )
        repairs["post_collision_knee_preflex"] = {
            **collision_preflex,
            "refit": collision_preflex_refit,
            "maximum_remaining_foot_penetration_m": float(
                collision.maximum_foot_penetration_m
            ),
            "maximum_remaining_body_penetration_m": float(
                collision.maximum_forbidden_body_penetration_m
            ),
        }

    subdivision: dict[str, object] = {
        "added_frame_count": 0.0,
        "maximum_interval_dilation": 1.0,
    }
    if _retimed_motion_metrics(motion)["maximum_joint_step_rad"] > 0.18:
        old_frame_count = len(motion.root_position_world)
        retime_extras = dict(extras)
        retime_extras["__source_coordinate"] = np.arange(
            old_frame_count, dtype=np.float64
        )
        motion, retime_extras, subdivision = _smoothly_subdivide_motion_steps(
            motion,
            retime_extras,
            target_joint_step_rad=0.14,
            transition_sigma_frames=3.0,
        )
        source_coordinate = np.asarray(
            retime_extras.pop("__source_coordinate"), dtype=np.float64
        )
        arrays = _resample_trace_arrays(
            arrays,
            source_coordinate,
            original_frame_count=old_frame_count,
        )
        extras = retime_extras
        # Both endpoints already passed the bounded contact solve.  Running
        # nonlinear IK again on the new in-between frames can switch back to
        # the very knee branch this time warp removed.  Preserve the smooth
        # interpolated joint curve and let the exact contact/collision audits
        # below decide whether its subframes remain valid.
        audit_mesh, audit_region = _terrain_index_near_motion(field.index, motion)
        collision = audit_stair_motion_collisions(
            motion,
            model_path=args.model,
            target_mesh=audit_mesh,
            joint_names=ISAACLAB_JOINT_NAMES,
            maximum_foot_penetration_m=0.0051,
            maximum_forbidden_body_penetration_m=0.0,
        )
        if (
            not collision.accepted
            and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
            and collision.maximum_foot_penetration_m <= 0.040
        ):
            # Time-density in-betweening is deliberately linear in joint
            # space.  Across a stair edge that chord can cut a free sole
            # through the riser even though both solved endpoints are clear.
            # Repair only that newly inserted swing arc; planted targets stay
            # fixed and the final smoothness/contact audits remain unchanged.
            motion, extras, collision, swing_repair = (
                _repair_swing_foot_clearance(
                    motion,
                    extras,
                    collision,
                    adapter=adapter,
                    target_mesh=audit_mesh,
                    model_path=args.model,
                    joint_names=ISAACLAB_JOINT_NAMES,
                    maximum_total_lift_m=0.050,
                    maximum_iterations=3,
                )
            )
            repairs["post_subdivision_swing_clearance"] = swing_repair
            if _retimed_motion_metrics(motion)["maximum_joint_step_rad"] > 0.18:
                repaired_frame_count = len(motion.root_position_world)
                repaired_extras = dict(extras)
                repaired_extras["__source_coordinate"] = np.arange(
                    repaired_frame_count, dtype=np.float64
                )
                motion, repaired_extras, repaired_subdivision = (
                    _smoothly_subdivide_motion_steps(
                        motion,
                        repaired_extras,
                        target_joint_step_rad=0.14,
                        transition_sigma_frames=3.0,
                    )
                )
                repaired_coordinate = np.asarray(
                    repaired_extras.pop("__source_coordinate"),
                    dtype=np.float64,
                )
                arrays = _resample_trace_arrays(
                    arrays,
                    repaired_coordinate,
                    original_frame_count=repaired_frame_count,
                )
                extras = repaired_extras
                subdivision["post_repair"] = repaired_subdivision
                subdivision["added_frame_count"] = float(
                    subdivision["added_frame_count"]
                ) + float(repaired_subdivision["added_frame_count"])
                subdivision["maximum_interval_dilation"] = max(
                    float(subdivision["maximum_interval_dilation"]),
                    float(repaired_subdivision["maximum_interval_dilation"]),
                )
                audit_mesh, audit_region = _terrain_index_near_motion(
                    field.index, motion
                )
                collision = audit_stair_motion_collisions(
                    motion,
                    model_path=args.model,
                    target_mesh=audit_mesh,
                    joint_names=ISAACLAB_JOINT_NAMES,
                    maximum_foot_penetration_m=0.0051,
                    maximum_forbidden_body_penetration_m=0.0,
                )

    # Measure terrain/IK edits on the delivered lower body.  The independent
    # arm quieting is intentional data conditioning, not an IK correction.
    # Computing this after optional time subdivision also includes any repair
    # made after the initial foothold solve.
    lower_body_columns = np.asarray(
        [
            index
            for index, name in enumerate(ISAACLAB_JOINT_NAMES)
            if not any(part in name for part in ("shoulder", "elbow", "wrist"))
        ],
        dtype=np.int64,
    )
    source_joint_position = np.asarray(
        arrays["joint_position"], dtype=np.float64
    )
    if source_joint_position.shape != np.asarray(motion.joint_position).shape:
        raise ValueError("retimed source and processed joint arrays disagree")
    core_refit_joint_delta = np.abs(
        np.asarray(motion.joint_position, dtype=np.float64)[:, lower_body_columns]
        - source_joint_position[:, lower_body_columns]
    )

    contact_report = _audit_stance_contact_support(
        motion,
        extras,
        adapter=adapter,
        target_mesh=audit_mesh,
        ground_fallback_height_m=0.0,
    )
    partial_report = _audit_partial_rigid_support_contact(
        motion,
        extras,
        adapter=adapter,
        target_mesh=audit_mesh,
    )
    motion_metrics = _retimed_motion_metrics(motion)
    motion_metrics["maximum_root_acceleration_m_s2"] = (
        _maximum_root_acceleration_m_s2(
            motion.root_position_world, fps=motion.fps
        )
    )
    # Warm-up context exists only to make the boundary solve well posed and
    # is cropped from the delivered motion.  Do not reject a clean requested
    # interval because an unreachable target occurred in discarded context.
    # Preserve those solve-wide diagnostics explicitly for debugging, while
    # all acceptance values below describe the actual saved frames.
    solve_context_refit = dict(refit)
    core_target_error = np.asarray(
        extras.get("per_frame_sole_target_error_by_foot_m", ()),
        dtype=np.float64,
    )
    refit = {
        **refit,
        "solve_context_maximum_sole_target_error_m": float(
            solve_context_refit["maximum_sole_target_error_m"]
        ),
        "solve_context_maximum_joint_delta_rad": float(
            solve_context_refit["maximum_joint_delta_rad"]
        ),
        "solve_context_maximum_joint_step_rad": float(
            solve_context_refit["maximum_joint_step_rad"]
        ),
        "maximum_sole_target_error_m": (
            float(np.max(core_target_error)) if core_target_error.size else 0.0
        ),
        "maximum_joint_delta_rad": (
            float(np.max(core_refit_joint_delta))
            if core_refit_joint_delta.size
            else 0.0
        ),
        "maximum_joint_step_rad": float(
            motion_metrics["maximum_joint_step_rad"]
        ),
    }
    contact_accepted = bool(
        contact_report["accepted"]
        or (
            partial_report.get("applicable", False)
            and partial_report.get("accepted", False)
        )
    )
    core_longitudinal = np.asarray(
        extras.get("planned_foothold_longitudinal_adjustment_m", ()),
        dtype=np.float64,
    )
    core_lateral = np.asarray(
        extras.get("planned_foothold_lateral_adjustment_m", ()),
        dtype=np.float64,
    )
    maximum_longitudinal_adjustment = (
        float(np.max(np.abs(core_longitudinal)))
        if core_longitudinal.size
        else 0.0
    )
    maximum_lateral_adjustment = (
        float(np.max(np.abs(core_lateral))) if core_lateral.size else 0.0
    )
    planned_pelvis_height = np.asarray(
        extras.get("planned_pelvis_height_world_m", ()), dtype=np.float64
    )
    source_pelvis_height = np.asarray(
        arrays["root_position_world"], dtype=np.float64
    )[:, 2]
    maximum_root_height_adjustment = (
        float(np.max(np.abs(planned_pelvis_height - source_pelvis_height)))
        if planned_pelvis_height.shape == source_pelvis_height.shape
        else float(
            foothold_report.get(
                "maximum_absolute_root_height_adjustment_m", math.inf
            )
        )
    )
    gates = {
        "sole_refit_error_le_12mm": float(
            refit["maximum_sole_target_error_m"]
        )
        <= 0.012,
        "exact_collision": bool(collision.accepted),
        "stance_contact": contact_accepted,
        "joint_step_le_0p20rad": float(
            motion_metrics["maximum_joint_step_rad"]
        )
        <= 0.20,
        "root_step_le_35mm": float(
            motion_metrics["maximum_root_translation_step_m"]
        )
        <= 0.035,
        "root_acceleration_le_30mps2": float(
            motion_metrics["maximum_root_acceleration_m_s2"]
        )
        <= 30.0,
        # A valid full-foot placement is not automatically a natural motion
        # match.  Large shifts mean retrieval chose the wrong terrain phase;
        # reject that source instead of hiding it with powerful IK.
        "foothold_longitudinal_adjustment_le_12cm": (
            maximum_longitudinal_adjustment <= 0.12
        ),
        "foothold_lateral_adjustment_le_8cm": (
            maximum_lateral_adjustment <= 0.08
        ),
        "root_height_adjustment_le_8cm": (
            maximum_root_height_adjustment <= 0.08
        ),
        # This is the aggregate deviation after foothold fitting, bounded
        # pelvis sharing, and flight-only collision repair, not the authority
        # of an individual IK solve (each remains bounded at 0.35 rad).  The
        # 0.65-rad aggregate cap admits the densely reviewed 0.618-rad full
        # course, whose exact contact/collision and 0.140-rad temporal-step
        # gates pass; it does not authorize one saturated update.
        "ik_joint_correction_le_0p65rad": float(
            refit["maximum_joint_delta_rad"]
        )
        <= 0.65,
    }
    output_arrays = dict(arrays)
    output_arrays["root_position_world"] = np.asarray(
        motion.root_position_world, dtype=np.float32
    )
    output_arrays["root_quaternion_world_wxyz"] = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float32
    )
    output_arrays["joint_position"] = np.asarray(
        motion.joint_position, dtype=np.float32
    )
    for key in (
        "authored_stance_mask",
        "target_sole_points_world",
        "adapted_sole_center_world",
        "per_frame_sole_target_error_by_foot_m",
        "planned_partial_rigid_support_mask",
        "planned_support_contact_point_count",
        "planned_pelvis_xy_offset_world_m",
        "planned_pelvis_height_world_m",
        "source_anchored_sole_points_world",
        "bracketed_swing_mask",
        "per_frame_swing_route_adjustment",
    ):
        if key in extras:
            output_arrays[key] = np.asarray(extras[key])
    np.savez_compressed(args.output, **output_arrays)
    return {
        "schema": "continuous-terrain-rigid-contact-postprocess/v1",
        "input_trace": str(args.trace.resolve()),
        "output_trace": str(args.output.resolve()),
        "source_frame_range": [requested_start, requested_stop],
        "solve_context_frame_range": [start, stop],
        "root_filter": root_filter,
        "stance_schedule": stance_report,
        "foothold_planning": foothold_report,
        "foothold_planning_terrain_region": planning_region,
        "swing_guidance": swing_guidance,
        "swing_knee_preflex": swing_knee_preflex,
        "upper_limb_attenuation": upper_limb_attenuation,
        "maximum_foothold_longitudinal_adjustment_m": (
            maximum_longitudinal_adjustment
        ),
        "maximum_foothold_lateral_adjustment_m": maximum_lateral_adjustment,
        "sole_refit": refit,
        "stance_reach_repair": reach_repair,
        "stance_reach_pose_repair": stance_pose_repair,
        "repairs": repairs,
        "adaptive_step_subdivision": subdivision,
        "collision_audit": collision.to_dict(),
        "collision_audit_region": audit_region,
        "stance_contact_audit": contact_report,
        "partial_rigid_support_contact_audit": partial_report,
        "motion_metrics": motion_metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-joint-correction-rad", type=float, default=0.35)
    parser.add_argument("--maximum-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--target-tolerance-m", type=float, default=0.001)
    parser.add_argument("--minimum-swing-clearance-m", type=float, default=0.008)
    parser.add_argument(
        "--maximum-target-creep-m-per-frame", type=float, default=0.004
    )
    parser.add_argument("--lock-horizontal", action="store_true")
    parser.add_argument("--disable-collision-repairs", action="store_true")
    parser.add_argument("--rigid-plants", action="store_true")
    parser.add_argument("--rigid-require-full-sole-support", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--context-frames", type=int, default=80)
    parser.add_argument("--pre-context-frames", type=int)
    parser.add_argument("--post-context-frames", type=int)
    parser.add_argument(
        "--rigid-maximum-stance-speed-mps", type=float, default=0.18
    )
    parser.add_argument(
        "--rigid-minimum-stance-run-frames", type=int, default=4
    )
    parser.add_argument(
        "--rigid-maximum-source-stance-speed-mps",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--rigid-maximum-source-sole-ground-clearance-m",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--rigid-maximum-pelvis-planar-adjustment-step-m",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--rigid-maximum-foothold-height-change-m",
        type=float,
        default=0.35,
        help=(
            "Maximum vertical change between successive planted footholds; "
            "lower values prevent a warped gait from skipping stair treads."
        ),
    )
    parser.add_argument(
        "--rigid-maximum-planar-reach-change-m",
        type=float,
        default=0.09,
    )
    parser.add_argument(
        "--rigid-maximum-foothold-longitudinal-adjustment-m",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--rigid-foothold-longitudinal-samples",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--rigid-maximum-foothold-lateral-adjustment-m",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--rigid-foothold-lateral-samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rigid-maximum-foothold-yaw-change-deg",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--rigid-swing-knee-preflex-gain-rad-per-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rigid-swing-knee-preflex-maximum-rad",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--rigid-pelvis-planar-smoothing-sigma-frames",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--rigid-maximum-stance-reach-root-lower-m",
        type=float,
        default=0.025,
        help=(
            "Maximum smooth pelvis lowering used only where measured rigid-"
            "stance IK residual proves the nominal pelvis is unreachable."
        ),
    )
    parser.add_argument(
        "--target-maximum-root-acceleration-m-s2", type=float, default=25.0
    )
    parser.add_argument("--arm-swing-scale", type=float, default=0.35)
    parser.add_argument("--wrist-swing-scale", type=float, default=0.10)
    parser.add_argument(
        "--arm-smoothing-sigma-frames", type=float, default=2.0
    )
    args = parser.parse_args()
    args.trace = args.trace.expanduser().resolve()
    args.terrain = args.terrain.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if not 0.0 <= args.arm_swing_scale <= 1.0:
        parser.error("--arm-swing-scale must lie in [0,1]")
    if not 0.0 <= args.wrist_swing_scale <= 1.0:
        parser.error("--wrist-swing-scale must lie in [0,1]")
    if args.arm_smoothing_sigma_frames < 0.0:
        parser.error("--arm-smoothing-sigma-frames must be nonnegative")
    if args.rigid_maximum_pelvis_planar_adjustment_step_m <= 0.0:
        parser.error(
            "--rigid-maximum-pelvis-planar-adjustment-step-m must be positive"
        )
    if args.rigid_maximum_source_stance_speed_mps <= 0.0:
        parser.error(
            "--rigid-maximum-source-stance-speed-mps must be positive"
        )
    if args.rigid_maximum_source_sole_ground_clearance_m <= 0.0:
        parser.error(
            "--rigid-maximum-source-sole-ground-clearance-m must be positive"
        )
    if args.rigid_pelvis_planar_smoothing_sigma_frames < 0.0:
        parser.error(
            "--rigid-pelvis-planar-smoothing-sigma-frames must be nonnegative"
        )
    if args.rigid_maximum_stance_reach_root_lower_m < 0.0:
        parser.error(
            "--rigid-maximum-stance-reach-root-lower-m must be nonnegative"
        )
    if args.rigid_maximum_foothold_height_change_m < 0.0:
        parser.error(
            "--rigid-maximum-foothold-height-change-m must be nonnegative"
        )
    if args.rigid_maximum_planar_reach_change_m < 0.0:
        parser.error(
            "--rigid-maximum-planar-reach-change-m must be nonnegative"
        )
    if args.rigid_maximum_foothold_longitudinal_adjustment_m < 0.0:
        parser.error(
            "--rigid-maximum-foothold-longitudinal-adjustment-m must be "
            "nonnegative"
        )
    if args.rigid_foothold_longitudinal_samples <= 0:
        parser.error("--rigid-foothold-longitudinal-samples must be positive")
    if args.rigid_maximum_foothold_lateral_adjustment_m < 0.0:
        parser.error(
            "--rigid-maximum-foothold-lateral-adjustment-m must be "
            "nonnegative"
        )
    if args.rigid_foothold_lateral_samples <= 0:
        parser.error("--rigid-foothold-lateral-samples must be positive")
    if args.rigid_maximum_foothold_yaw_change_deg < 0.0:
        parser.error(
            "--rigid-maximum-foothold-yaw-change-deg must be nonnegative"
        )
    if args.rigid_swing_knee_preflex_gain_rad_per_m < 0.0:
        parser.error(
            "--rigid-swing-knee-preflex-gain-rad-per-m must be nonnegative"
        )
    if args.rigid_swing_knee_preflex_maximum_rad < 0.0:
        parser.error(
            "--rigid-swing-knee-preflex-maximum-rad must be nonnegative"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = (
            postprocess_rigid(args) if args.rigid_plants else postprocess(args)
        )
    except Exception as error:  # preserve a useful rejected artifact in sweeps
        report = {
            "schema": "continuous-terrain-contact-postprocess-error/v1",
            "input_trace": str(args.trace),
            "output_trace": str(args.output),
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
