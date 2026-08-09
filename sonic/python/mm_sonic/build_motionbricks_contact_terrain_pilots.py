"""Retarget quiet-arm MotionBricks gaits onto exact procedural rough terrain.

Unlike the earlier root-clearance diagnostic, this builder locks each authored
stance sole to the target mesh, solves the legs against those full rigid-sole
targets, repairs only the offending swing foot, and rejects hovering after a
second exact contact audit.  The input gait and the terrain are stored as a
paired kinematic asset with robot-local velocity and facing labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from .build_bones_side_on_stair_pilots import (
    MAXIMUM_ARM_P99_ACCELERATION_RAD_S2,
    MAXIMUM_ARM_EXCURSION_RAD,
    MAXIMUM_ARM_RMS_VELOCITY_RAD_S,
    MAXIMUM_ARM_VELOCITY_RAD_S,
    MAXIMUM_WRIST_P99_ACCELERATION_RAD_S2,
    MAXIMUM_WRIST_EXCURSION_RAD,
    MAXIMUM_WRIST_RMS_VELOCITY_RAD_S,
    MAXIMUM_WRIST_VELOCITY_RAD_S,
    _adaptively_subdivide_motion_steps,
    _attenuate_upper_limb_motion,
    _audit_stance_contact_support,
    _blend_anchored_sole_offsets,
    _conform_stance_sole_targets_to_mesh,
    _mechanically_accepted,
    _refit_motion_to_sole_targets,
    _repair_stance_foot_mesh_clearance,
    _repair_swing_foot_clearance,
    _repairable_warp_accepted,
    _retime_motion_and_extras,
    _retimed_motion_metrics,
    _upper_limb_motion_metrics,
)
from .build_c490_cowarped_directional import write_terrain_usd
from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .canonical_terrain_matcher import RegularGridHeightField
from .compose_coherent_block_plan import _smooth_upper_clearance_envelope
from .build_rolling_terrain_pilots import (
    DEFAULT_SOURCE,
    PROFILES,
    _source_motion,
    build_profile,
)
from .build_stairs500_omnidirectional_pilots import (
    _fill_short_stance_gaps,
    _remove_short_stance_runs,
    _save_motion,
    _terrain_height,
)
from .terrain_oracle.math3d import quaternion_multiply_wxyz
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_foothold_anchors import NominalFootSolePose
from .terrain_oracle.stair_fragment_reconstruction import (
    _replace_bracketed_swing_trajectories,
    _sole_yaw,
    _stance_spans,
    _smooth_swing_route_adjustments,
)
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import StitchedMotion
from .terrain_foothold_planner import (
    FootholdIntent,
    TerrainFootholdPlannerConfig,
    plan_terrain_footholds,
)


DEFAULT_OUTPUT = Path(
    "/move/data/terrain-aware/sonic-rollouts/"
    "motionbricks_contact_rough_v1"
)


def _align_motion_corridor(
    motion: StitchedMotion,
) -> tuple[StitchedMotion, dict[str, object]]:
    """Rotate the route's main corridor onto +X.

    A net-displacement axis is the least surprising choice for a traversal.
    Stop/reverse/toggle clips can travel several metres while ending near the
    start, however, so their net chord is not a useful corridor.  In that case
    use the principal spatial axis and orient it along the first substantial
    excursion.  The same rigid yaw is applied to root pose and translation,
    preserving every travel-vs-facing command.
    """

    root = np.asarray(motion.root_position_world, dtype=np.float64)
    quaternion = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    window = max(3, len(root) // 20)
    displacement = np.median(root[-window:, :2], axis=0) - np.median(
        root[:window, :2], axis=0
    )
    distance = float(np.linalg.norm(displacement))
    if distance >= 1.25:
        corridor_axis = displacement / distance
        alignment_basis = "net_displacement"
    else:
        centred_xy = root[:, :2] - np.mean(root[:, :2], axis=0)
        covariance = centred_xy.T @ centred_xy
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        corridor_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = root[:, :2] @ corridor_axis
        path_span = float(np.ptp(projection))
        if path_span < 1.25:
            raise ValueError(
                "rough-terrain source spans less than 1.25 m along its main corridor"
            )
        start_xy = np.median(root[:window, :2], axis=0)
        excursion = root[:, :2] - start_xy
        excursion_norm = np.linalg.norm(excursion, axis=1)
        substantial = np.flatnonzero(excursion_norm >= min(0.25, 0.20 * path_span))
        direction_frame = (
            int(substantial[0]) if len(substantial) else int(np.argmax(excursion_norm))
        )
        if float(excursion[direction_frame] @ corridor_axis) < 0.0:
            corridor_axis = -corridor_axis
        alignment_basis = "principal_spatial_axis"
    source_yaw = math.atan2(float(corridor_axis[1]), float(corridor_axis[0]))
    rotation_yaw = -source_yaw
    cosine = math.cos(rotation_yaw)
    sine = math.sin(rotation_yaw)
    rotation_xy = np.asarray(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    aligned_root = root.copy()
    aligned_root[:, :2] = (root[:, :2] - root[0, :2]) @ rotation_xy.T
    yaw_quaternion = np.asarray(
        (
            math.cos(0.5 * rotation_yaw),
            0.0,
            0.0,
            math.sin(0.5 * rotation_yaw),
        ),
        dtype=np.float64,
    )
    aligned_quaternion = np.asarray(
        [
            quaternion_multiply_wxyz(yaw_quaternion, value)
            for value in quaternion
        ],
        dtype=np.float64,
    )
    aligned_quaternion /= np.linalg.norm(
        aligned_quaternion, axis=1, keepdims=True
    )
    aligned = StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(aligned_root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            aligned_quaternion, dtype=np.float32
        ),
        joint_position=np.asarray(motion.joint_position, dtype=np.float32),
        provenance=motion.provenance,
        seam_indices=motion.seam_indices,
    )
    return aligned, {
        "alignment_basis": alignment_basis,
        "source_travel_yaw_rad": source_yaw,
        "alignment_yaw_rad": rotation_yaw,
        "net_travel_distance_m": distance,
        "corridor_span_m": float(np.ptp(aligned_root[:, 0])),
        "aligned_x_range_m": [
            float(np.min(aligned_root[:, 0])),
            float(np.max(aligned_root[:, 0])),
        ],
        "aligned_y_range_m": [
            float(np.min(aligned_root[:, 1])),
            float(np.max(aligned_root[:, 1])),
        ],
    }


def _profile_for_motion(
    profile: str,
    motion: StitchedMotion,
    *,
    spacing_m: float,
) -> tuple[object, dict[str, object]]:
    root = np.asarray(motion.root_position_world, dtype=np.float64)
    x_min = float(np.min(root[:, 0]) - 0.75)
    x_max = float(np.max(root[:, 0]) + 0.75)
    y_min = float(np.min(root[:, 1]) - 0.85)
    y_max = float(np.max(root[:, 1]) + 0.85)
    active_start = float(np.min(root[:, 0]) + 0.30)
    active_stop = float(np.max(root[:, 0]) - 0.30)
    if active_stop - active_start < 1.0:
        raise ValueError("aligned source is too short for a rough passage")
    return build_profile(
        profile,
        spacing_m=spacing_m,
        x_range_m=(x_min, x_max),
        y_range_m=(y_min, y_max),
        active_range_m=(active_start, active_stop),
    )


def _root_height_scaffold(field: object, root_xy: np.ndarray) -> np.ndarray:
    height, _normal, hit = field.sample(np.asarray(root_xy, dtype=np.float64))
    if not np.all(hit) or not np.isfinite(height).all():
        raise ValueError("motion root leaves the procedural terrain")
    values = np.asarray(height, dtype=np.float64)
    # A short symmetric filter removes height-map sampling chatter without
    # erasing the real long-wave vertical route.  Exact sole targets below,
    # not this scaffold, define contact.
    kernel = np.asarray((1, 2, 3, 4, 3, 2, 1), dtype=np.float64)
    kernel /= np.sum(kernel)
    smooth = np.convolve(np.pad(values, (3, 3), mode="edge"), kernel, mode="valid")
    return smooth - float(values[0])


def _paired_flat_support_mesh(field: object) -> object:
    """Build the exact flat-topology counterpart of a target height field."""

    flat = RegularGridHeightField(
        np.zeros_like(field.height),
        spacing_m=tuple(float(value) for value in field.spacing_m),
        origin_xy=tuple(float(value) for value in field.origin_xy),
    ).index
    if not np.array_equal(flat.mesh.faces, field.index.mesh.faces):
        raise ValueError("flat and target terrain topology do not match")
    return flat


def _terrain_clear_transplanted_swings(
    nominal_targets: np.ndarray,
    anchored_targets: np.ndarray,
    stance: np.ndarray,
    sphere_radii: Sequence[np.ndarray],
    *,
    target_mesh: object,
    minimum_arc_clearance_m: float = 0.025,
    surface_clearance_m: float = 0.006,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transplant the learned flight arc between planned fixed footholds.

    A free foot cannot keep its original flat-world target after its takeoff
    and landing plants have moved onto rough terrain.  Conversely, replacing
    the whole flight with a synthetic parabola destroys the learned gait.  We
    retain the authored residual above the takeoff-to-landing interpolation,
    carry the planned anchor offsets through the complete flight, and add only
    the vertical clearance required by the exact target height field.
    """

    targets, bracketed = _replace_bracketed_swing_trajectories(
        nominal_targets,
        anchored_targets,
        stance,
        minimum_clearance_m=float(minimum_arc_clearance_m),
        maximum_planar_deviation_m=0.12,
    )
    stance_mask = np.asarray(stance, dtype=bool)
    required = np.zeros((len(targets), 2, 3), dtype=np.float64)
    ray_z = float(np.max(target_mesh.vertices_world[:, 2]) + 1.0)
    for frame in range(len(targets)):
        for foot in range(2):
            if stance_mask[frame, foot]:
                continue
            radii = np.asarray(sphere_radii[foot], dtype=np.float64)
            lift = 0.0
            for probe, point in enumerate(targets[frame, foot]):
                height = _terrain_height(target_mesh, point[:2], ray_z)
                if math.isfinite(height):
                    lift = max(
                        lift,
                        float(height)
                        + float(radii[probe])
                        + float(surface_clearance_m)
                        - float(point[2]),
                    )
            required[frame, foot, 2] = max(0.0, lift)
    smoothed = _smooth_swing_route_adjustments(
        required,
        stance_mask,
        decay_frames=4.0,
    )
    targets = np.asarray(targets, dtype=np.float64).copy()
    targets += smoothed[:, :, None, :]
    return targets, bracketed, smoothed


def _final_upper_limb_accepted(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["maximum_arm_excursion_rad"] <= MAXIMUM_ARM_EXCURSION_RAD
        and metrics["maximum_wrist_excursion_rad"]
        <= MAXIMUM_WRIST_EXCURSION_RAD
        and metrics["maximum_arm_velocity_rad_s"] <= MAXIMUM_ARM_VELOCITY_RAD_S
        and metrics["maximum_wrist_velocity_rad_s"]
        <= MAXIMUM_WRIST_VELOCITY_RAD_S
        and metrics["maximum_joint_rms_arm_velocity_rad_s"]
        <= MAXIMUM_ARM_RMS_VELOCITY_RAD_S
        and metrics["maximum_joint_rms_wrist_velocity_rad_s"]
        <= MAXIMUM_WRIST_RMS_VELOCITY_RAD_S
        and metrics["p99_arm_acceleration_rad_s2"]
        <= MAXIMUM_ARM_P99_ACCELERATION_RAD_S2
        and metrics["p99_wrist_acceleration_rad_s2"]
        <= MAXIMUM_WRIST_P99_ACCELERATION_RAD_S2
    )


def _repair_forbidden_body_with_root_lift(
    motion: StitchedMotion,
    extras: dict[str, np.ndarray],
    collision: object,
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: object,
    model_path: Path,
    joint_names: Sequence[str],
    maximum_total_lift_m: float = 0.025,
) -> tuple[StitchedMotion, dict[str, np.ndarray], object, dict[str, float]]:
    """Clear a shallow body graze, then replant only rigid stance cores."""

    current_motion = motion
    current_extras = extras
    current_collision = collision
    total = np.zeros(len(motion.root_position_world), dtype=np.float64)
    maximum_refit_error = 0.0
    for _iteration in range(4):
        body = np.asarray(
            current_collision.per_frame_max_forbidden_body_penetration_m,
            dtype=np.float64,
        )
        if float(np.max(body)) <= 1.0e-6:
            break
        required = np.zeros_like(body)
        active = body > 1.0e-6
        required[active] = body[active] + 0.001
        lift = _smooth_upper_clearance_envelope(
            required, radius_frames=16
        )
        remaining = float(maximum_total_lift_m) - float(np.max(total))
        if remaining <= 0.0:
            break
        if float(np.max(lift)) > remaining:
            lift *= remaining / float(np.max(lift))
        roots = np.asarray(
            current_motion.root_position_world, dtype=np.float64
        ).copy()
        roots[:, 2] += lift
        candidate = StitchedMotion(
            fps=current_motion.fps,
            root_position_world=np.asarray(roots, dtype=np.float32),
            root_quaternion_world_wxyz=(
                current_motion.root_quaternion_world_wxyz
            ),
            joint_position=current_motion.joint_position,
            provenance=current_motion.provenance,
            seam_indices=current_motion.seam_indices,
        )
        candidate, candidate_extras, refit = _refit_motion_to_sole_targets(
            candidate,
            current_extras,
            adapter=adapter,
            # A pelvis lift should carry the authored swing with the body.
            # Only planted soles remain fixed in world space; otherwise this
            # repair quietly reintroduces the same kick/shuffle constraint as
            # the initial all-foot refit.
            stance_only=True,
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
            candidate_collision.maximum_forbidden_body_penetration_m
            >= current_collision.maximum_forbidden_body_penetration_m - 1.0e-6
        ):
            break
        total += lift
        current_motion = candidate
        current_extras = candidate_extras
        current_collision = candidate_collision
        maximum_refit_error = max(
            maximum_refit_error,
            float(refit["maximum_sole_target_error_m"]),
        )
    updated = dict(current_extras)
    updated["body_clearance_root_lift_m"] = np.asarray(
        total, dtype=np.float32
    )
    return current_motion, updated, current_collision, {
        "maximum_root_lift_m": float(np.max(total)),
        "maximum_refit_error_m": maximum_refit_error,
        "maximum_remaining_body_penetration_m": float(
            current_collision.maximum_forbidden_body_penetration_m
        ),
    }


def _save_terrain(field: object, destination: Path) -> tuple[Path, Path]:
    npz = destination / "terrain.npz"
    np.savez_compressed(
        npz,
        height=np.asarray(field.height, dtype=np.float32),
        valid=np.asarray(field.valid, dtype=np.bool_),
        spacing_m=np.asarray(field.spacing_m, dtype=np.float32),
        origin_xy=np.asarray(field.origin_xy, dtype=np.float32),
    )
    usd = write_terrain_usd(field.index, destination / "terrain.usda")
    return npz.resolve(), usd.resolve()


def _source_stance_schedule_and_extras(
    source: StitchedMotion,
    *,
    adapter: _G1FootfallAdapter,
    source_mesh: object,
    minimum_stance_run_frames: int = 5,
    maximum_stance_gap_frames: int = 1,
    maximum_stance_speed_mps: float = 0.05,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Detect authored plants without running a discarded terrain IK pass.

    MotionBricks contacts retain a small amount of sole drift, but anchoring
    the broader near-contact phase is worse: a 10 cm/s threshold classified
    toe-off and heel-strike as planted and more than doubled the stance frames
    in a natural forward walk.  Five cm/s keeps only the true low-speed plant
    cores.  The guidance blend carries those fixed targets smoothly into and
    out of contact while swing clearance remains free to preserve the authored
    gait.  Requiring a 100 ms core and closing at most one 20 ms dropout also
    prevents a two-frame toe-off interval from being swallowed into a plant.
    """

    roots = np.asarray(source.root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        source.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(source.joint_position, dtype=np.float64)
    frame_count = len(roots)
    sphere_centres = np.empty((frame_count, 2, 4, 3), dtype=np.float64)
    sole_centres = np.empty((frame_count, 2, 3), dtype=np.float64)
    support_count = np.zeros((frame_count, 2), dtype=np.int16)
    radii = adapter.sole_sphere_radii()
    ray_z = float(np.max(source_mesh.vertices_world[:, 2]) + 1.0)
    for frame in range(frame_count):
        feet = adapter.sole_positions_for_pose(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints[frame],
        )
        sphere_centres[frame] = feet
        sole_centres[frame] = np.asarray(
            [np.mean(value, axis=0) for value in feet]
        )
        for foot in range(2):
            support = feet[foot].copy()
            support[:, 2] -= radii[foot]
            support_count[frame, foot] = sum(
                abs(
                    float(point[2])
                    - _terrain_height(source_mesh, point[:2], ray_z)
                )
                <= 0.020
                for point in support
            )
    sole_speed = np.linalg.norm(
        np.gradient(sole_centres, axis=0) * float(source.fps), axis=2
    )
    stance = _remove_short_stance_runs(
        _fill_short_stance_gaps(
            (support_count >= 2)
            & (sole_speed <= float(maximum_stance_speed_mps)),
            maximum_gap_frames=maximum_stance_gap_frames,
        ),
        minimum_run_frames=minimum_stance_run_frames,
    )
    if not np.any(stance):
        raise ValueError("source gait has no detectable stance frames")
    progress = roots[:, 0] - float(np.min(roots[:, 0]))
    progress_range = float(np.ptp(progress))
    if progress_range > 1.0e-9:
        progress /= progress_range
    else:
        progress.fill(0.0)
    zeros = np.zeros(frame_count, dtype=np.float32)
    extras = {
        "path_lateral_offset_m": zeros.copy(),
        "path_yaw_offset_rad": zeros.copy(),
        "pose_yaw_offset_rad": zeros.copy(),
        "foot_yaw_offset_rad": zeros.copy(),
        "facing_yaw_offset_rad": zeros.copy(),
        "path_normalized_progress": np.asarray(progress, dtype=np.float32),
        "intended_root_position_world": np.asarray(roots, dtype=np.float32),
        "authored_stance_mask": np.asarray(stance, dtype=np.bool_),
        "target_stance_support_point_count": support_count,
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
    diagnostics = {
        "frame_count": frame_count,
        "minimum_stance_run_frames": int(minimum_stance_run_frames),
        "maximum_stance_gap_frames": int(maximum_stance_gap_frames),
        "maximum_stance_speed_threshold_mps": float(maximum_stance_speed_mps),
        "stance_frame_foot_count": int(np.count_nonzero(stance)),
        "stance_span_count": len(_stance_spans(stance)),
        "minimum_source_support_point_count": int(
            np.min(support_count[stance])
        ),
        "maximum_detected_stance_speed_mps": float(
            np.max(sole_speed[stance])
        ),
    }
    return extras, diagnostics


def _plan_and_apply_fixed_terrain_footholds(
    source: StitchedMotion,
    extras: dict[str, np.ndarray],
    *,
    adapter: _G1FootfallAdapter,
    target_mesh: object,
) -> tuple[StitchedMotion, dict[str, np.ndarray], dict[str, object]]:
    """Replace drifting paired-warp contacts with fixed rigid footholds.

    ``warp_motion`` remains useful for robust authored-stance detection and
    shared provenance arrays.  Its continuous per-frame height correspondence
    is not a valid planted-foot target on fixed arbitrary terrain, however: it
    lets a stance slide through the surface.  This stage plans one whole-sole
    pose per stance run, holds it for that complete run, and restores the clean
    source gait as the swing-leg/IK initial condition.
    """

    stance = np.asarray(extras.get("authored_stance_mask"), dtype=bool)
    frame_count = len(source.root_position_world)
    if stance.shape != (frame_count, 2):
        raise ValueError("fixed-terrain planning requires stance mask [T,2]")
    spans = _stance_spans(stance)
    if not spans:
        raise ValueError("fixed-terrain planning found no stance windows")

    source_root = np.asarray(source.root_position_world, dtype=np.float64)
    source_quaternion = np.asarray(
        source.root_quaternion_world_wxyz, dtype=np.float64
    )
    source_joints = np.asarray(source.joint_position, dtype=np.float64)
    nominal_sphere_centres: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    intents: list[FootholdIntent] = []
    for span in spans:
        frame = (span.start_frame + span.stop_frame - 1) // 2
        pose = {
            "root_position": source_root[frame],
            "root_quaternion_wxyz": source_quaternion[frame],
            "joints": source_joints[frame],
        }
        sphere_centres = adapter.sole_positions_for_pose(**pose)
        support_points = adapter.sole_support_points_for_pose(**pose)
        envelopes = adapter.foot_collision_envelope_points_for_pose(**pose)
        foot = span.foot_index
        nominal_sphere_centres[
            (span.foot_index, span.start_frame, span.stop_frame)
        ] = sphere_centres
        intents.append(
            FootholdIntent(
                span=span,
                nominal_pose=NominalFootSolePose(
                    sole_center_world=np.mean(sphere_centres[foot], axis=0),
                    sole_support_points_world=support_points[foot],
                    yaw_rad=_sole_yaw(support_points[foot]),
                    collision_envelope_points_world=envelopes[foot],
                ),
            )
        )

    result = plan_terrain_footholds(
        target_mesh,
        intents,
        source_root[:, 2],
        config=TerrainFootholdPlannerConfig(
            # A rigid G1 sole is about as long as the old +/-12 cm search.
            # On a tread boundary that range can contain no placement whose
            # complete footprint clears the riser.  Search one additional
            # half-foot in either direction at the same 6 cm resolution;
            # downstream whole-leg IK, planted-foot drift, and exact body
            # collision audits still reject an unreachable re-placement.
            maximum_longitudinal_adjustment_m=0.24,
            maximum_lateral_adjustment_m=0.12,
            maximum_yaw_adjustment_rad=math.radians(15.0),
            longitudinal_samples=9,
            maximum_surface_slope_rad=math.radians(25.0),
            maximum_pelvis_height_step_m=0.020,
            maximum_planar_reach_change_m=0.090,
            maximum_yaw_change_rad=math.radians(7.5),
            maximum_pelvis_planar_adjustment_step_m=0.015,
            # Start/stop MotionBricks sequences can contain two separated
            # same-foot replants without an intervening opposite-foot plant.
            # This relaxes only schedule ordering, never support geometry.
            require_alternating_feet=False,
        ),
    )
    planning_report = asdict(result.diagnostics)
    if result.plan is None:
        raise ValueError(
            "fixed-terrain foothold planning failed: "
            f"{result.diagnostics.reason_code}: {result.diagnostics.message}"
        )

    targets = np.asarray(extras.get("target_sole_points_world"), dtype=np.float64)
    if targets.ndim != 4 or targets.shape[:2] != (frame_count, 2):
        raise ValueError("fixed-terrain planning requires sole targets [T,2,P,3]")
    targets = targets.copy()
    assigned = np.zeros((frame_count, 2), dtype=bool)
    surface_normal = np.zeros((frame_count, 2, 3), dtype=np.float64)
    longitudinal = np.zeros((frame_count, 2), dtype=np.float64)
    lateral = np.zeros((frame_count, 2), dtype=np.float64)
    yaw = np.zeros((frame_count, 2), dtype=np.float64)
    for foothold in result.plan.footholds:
        span = foothold.intent.span
        foot = span.foot_index
        sphere_centres = nominal_sphere_centres[
            (span.foot_index, span.start_frame, span.stop_frame)
        ]
        planted_centres = foothold.transform_points(sphere_centres[foot])
        targets[span.start_frame : span.stop_frame, foot] = planted_centres
        assigned[span.start_frame : span.stop_frame, foot] = True
        surface_normal[span.start_frame : span.stop_frame, foot] = (
            foothold.surface_normal_world
        )
        diagnostics = foothold.diagnostics
        longitudinal[span.start_frame : span.stop_frame, foot] = float(
            diagnostics.longitudinal_adjustment_m or 0.0
        )
        lateral[span.start_frame : span.stop_frame, foot] = float(
            diagnostics.lateral_adjustment_m or 0.0
        )
        yaw[span.start_frame : span.stop_frame, foot] = float(
            diagnostics.yaw_adjustment_rad or 0.0
        )
    if not np.array_equal(assigned, stance):
        raise ValueError("foothold plan did not assign every authored stance")

    corridor = result.plan.pelvis_height_corridor
    planned_root = source_root.copy()
    planned_root[:, 2] = corridor.selected_height_world_m
    # The planner selects footholds jointly in world XY and returns the exact
    # rate-feasible pelvis correction used to make that sequence reachable.
    # Reconstructing this independently here was the source of one-frame route
    # jumps when adjacent treads preferred different local minima.
    root_xy_offset = np.asarray(
        result.plan.pelvis_planar_offset_world_m, dtype=np.float64
    )
    if root_xy_offset.shape != (frame_count, 2):
        raise ValueError("foothold plan returned an invalid pelvis XY path")
    planned_root[:, :2] += root_xy_offset
    planned_motion = StitchedMotion(
        fps=source.fps,
        root_position_world=np.asarray(planned_root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            source.root_quaternion_world_wxyz, dtype=np.float32
        ),
        joint_position=np.asarray(source.joint_position, dtype=np.float32),
        provenance=source.provenance,
        seam_indices=source.seam_indices,
    )
    # Construct the target motion from the clean gait under the planned root,
    # then blend it into each fixed foothold before touchdown and out again
    # after liftoff.  This is the missing animation-graph transition: a swing
    # must not receive a clearance lift and then snap to an unrelated planted
    # pose in one frame.
    planned_nominal_targets = np.empty_like(targets)
    for frame in range(frame_count):
        feet = adapter.sole_positions_for_pose(
            root_position=planned_root[frame],
            root_quaternion_wxyz=source_quaternion[frame],
            joints=source_joints[frame],
        )
        planned_nominal_targets[frame] = feet
    blended = _blend_anchored_sole_offsets(
        planned_nominal_targets,
        targets,
        assigned,
    )
    targets, bracketed_swing, swing_adjustment = (
        _terrain_clear_transplanted_swings(
            planned_nominal_targets,
            blended,
            stance,
            adapter.sole_sphere_radii(),
            target_mesh=target_mesh,
        )
    )
    # Every transplanted target is continuous with the fixed takeoff and
    # landing plants.  Unlike the old contact-only fade, the complete flight
    # now receives the same coherent anchor change.
    guidance = np.ones_like(stance, dtype=np.float64)
    updated = dict(extras)
    updated["target_sole_points_world"] = np.asarray(targets, dtype=np.float32)
    updated["target_sole_center_world"] = np.asarray(
        np.mean(targets, axis=2), dtype=np.float32
    )
    updated["planned_foothold_assigned"] = assigned
    updated["planned_foothold_guidance_weight"] = np.asarray(
        guidance, dtype=np.float32
    )
    updated["planned_nominal_sole_points_world"] = np.asarray(
        planned_nominal_targets, dtype=np.float32
    )
    updated["bracketed_swing_mask"] = np.asarray(
        bracketed_swing, dtype=np.bool_
    )
    updated["per_frame_swing_route_adjustment"] = np.asarray(
        swing_adjustment, dtype=np.float32
    )
    updated["planned_foothold_surface_normal_world"] = np.asarray(
        surface_normal, dtype=np.float32
    )
    updated["planned_foothold_longitudinal_adjustment_m"] = np.asarray(
        longitudinal, dtype=np.float32
    )
    updated["planned_foothold_lateral_adjustment_m"] = np.asarray(
        lateral, dtype=np.float32
    )
    updated["planned_foothold_yaw_adjustment_rad"] = np.asarray(
        yaw, dtype=np.float32
    )
    updated["planned_pelvis_height_minimum_world_m"] = np.asarray(
        corridor.minimum_height_world_m, dtype=np.float32
    )
    updated["planned_pelvis_height_maximum_world_m"] = np.asarray(
        corridor.maximum_height_world_m, dtype=np.float32
    )
    updated["planned_pelvis_height_world_m"] = np.asarray(
        corridor.selected_height_world_m, dtype=np.float32
    )
    updated["planned_pelvis_xy_offset_world_m"] = np.asarray(
        root_xy_offset, dtype=np.float32
    )
    updated["intended_root_position_world"] = np.asarray(
        planned_root, dtype=np.float32
    )
    updated["terrain_root_anchor_shift_world"] = np.asarray(
        planned_root - source_root, dtype=np.float32
    )
    planning_report["stance_span_count"] = len(spans)
    planning_report["maximum_absolute_root_height_adjustment_m"] = float(
        np.max(np.abs(planned_root[:, 2] - source_root[:, 2]))
    )
    planning_report["maximum_horizontal_root_adjustment_m"] = float(
        np.max(np.linalg.norm(root_xy_offset, axis=1))
    )
    planning_report["maximum_horizontal_root_adjustment_step_m"] = float(
        np.max(np.linalg.norm(np.diff(root_xy_offset, axis=0), axis=1))
        if frame_count > 1
        else 0.0
    )
    return planned_motion, updated, planning_report


def _build_one(
    source_path: Path,
    profile: str,
    *,
    destination: Path,
    adapter: _G1FootfallAdapter,
    joint_names: Sequence[str],
    model_path: Path,
    spacing_m: float,
    temporal_scale: float,
    arm_swing_scale: float,
    wrist_swing_scale: float,
    arm_smoothing_sigma_frames: float,
    minimum_stance_run_frames: int,
    maximum_stance_gap_frames: int,
    maximum_stance_speed_mps: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "motionbricks-contact-rough-pilot/v1",
        "label": f"{source_path.stem}__{profile}",
        "profile": profile,
        "source_motion": str(source_path),
        "status": "rejected",
    }
    destination.mkdir(parents=True, exist_ok=True)
    try:
        source = _source_motion(source_path)
        quiet_joints, arm_attenuation = _attenuate_upper_limb_motion(
            source.joint_position,
            joint_names,
            fps=source.fps,
            arm_swing_scale=arm_swing_scale,
            wrist_swing_scale=wrist_swing_scale,
            smoothing_sigma_frames=arm_smoothing_sigma_frames,
        )
        source = StitchedMotion(
            fps=source.fps,
            root_position_world=source.root_position_world,
            root_quaternion_world_wxyz=source.root_quaternion_world_wxyz,
            joint_position=np.asarray(quiet_joints, dtype=np.float32),
            provenance=source.provenance,
            seam_indices=source.seam_indices,
        )
        source, alignment = _align_motion_corridor(source)
        row["source_alignment"] = alignment
        row["upper_limb_attenuation"] = arm_attenuation
        field, terrain_metadata = _profile_for_motion(
            profile, source, spacing_m=spacing_m
        )
        row["terrain"] = terrain_metadata
        terrain_npz, terrain_usd = _save_terrain(field, destination)
        row["terrain_npz"] = str(terrain_npz)
        row["terrain_usd"] = str(terrain_usd)
        # Stance detection needs only a flat mesh with the exact target grid
        # extent.  No expensive terrain IK is run until discrete full-foot
        # footholds and a feasible pelvis corridor have been selected.
        source_mesh = _paired_flat_support_mesh(field)
        extras, stance_schedule = _source_stance_schedule_and_extras(
            source,
            adapter=adapter,
            source_mesh=source_mesh,
            minimum_stance_run_frames=minimum_stance_run_frames,
            maximum_stance_gap_frames=maximum_stance_gap_frames,
            maximum_stance_speed_mps=maximum_stance_speed_mps,
        )
        row["source_stance_schedule"] = stance_schedule
        row["continuous_paired_warp_skipped"] = True
        motion, extras, foothold_planning = (
            _plan_and_apply_fixed_terrain_footholds(
                source,
                extras,
                adapter=adapter,
                target_mesh=field.index,
            )
        )
        row["foothold_planning"] = foothold_planning
        row["stance_sole_surface_conformance"] = {
            "skipped": True,
            "reason": "rigid full-foot footholds were planned directly on the exact mesh",
        }
        # The planned pelvis corridor and fixed sole targets replace both the
        # old per-frame terrain correspondence and its root-height scaffold.
        # The clean source gait is now the IK seed, so a rejected paired warp
        # above is diagnostic rather than an admission gate.
        row["sole_reach_pelvis_adjustment"] = {
            "skipped": False,
            "reason": "planned by fixed-terrain pelvis-height corridor",
        }
        motion, extras, refit = _refit_motion_to_sole_targets(
            motion,
            extras,
            adapter=adapter,
            # The swing target is now the learned source arc transplanted
            # between its planned takeoff and landing footholds, with an exact
            # height-field clearance envelope.  It is therefore meaningful to
            # fit both feet here; the obsolete flat-world swing target that
            # caused the kick/shuffle artifact is no longer present.
        )
        row["sole_refit"] = refit
        if (
            refit["maximum_sole_target_error_m"] > 0.005
            or refit["maximum_joint_step_rad"] > 0.20
        ):
            debug_path = destination / "rejected_sole_refit_debug.npz"
            _save_motion(
                debug_path,
                motion,
                mode="motionbricks_contact_rough_sole_refit_debug",
                extras=extras,
            )
            row["rejected_motion_debug"] = str(debug_path.resolve())
        if (
            refit["maximum_sole_target_error_m"] > 0.12
            or refit["maximum_joint_step_rad"] > 0.50
        ):
            raise ValueError("rough-terrain motion failed sole-refit gate")
        collision = audit_stair_motion_collisions(
            motion,
            model_path=model_path,
            target_mesh=field.index,
            joint_names=joint_names,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=0.0,
        )
        row["pre_repair_collision_audit"] = collision.to_dict()
        if (
            collision.maximum_forbidden_body_penetration_m > 1.0e-6
            and collision.maximum_forbidden_body_penetration_m <= 0.024
        ):
            motion, extras, collision, body_repair = (
                _repair_forbidden_body_with_root_lift(
                    motion,
                    extras,
                    collision,
                    adapter=adapter,
                    target_mesh=field.index,
                    model_path=model_path,
                    joint_names=joint_names,
                )
            )
            row["body_clearance_repair"] = body_repair
        if (
            not collision.accepted
            and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
            and collision.maximum_foot_penetration_m <= 0.015
        ):
            motion, extras, collision, stance_repair = (
                _repair_stance_foot_mesh_clearance(
                    motion,
                    extras,
                    collision,
                    adapter=adapter,
                    target_mesh=field.index,
                    model_path=model_path,
                    joint_names=joint_names,
                )
            )
            row["stance_mesh_clearance_repair"] = stance_repair
        if (
            not collision.accepted
            and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
            and collision.maximum_foot_penetration_m <= 0.050
        ):
            motion, extras, collision, swing_repair = _repair_swing_foot_clearance(
                motion,
                extras,
                collision,
                adapter=adapter,
                target_mesh=field.index,
                model_path=model_path,
                joint_names=joint_names,
                maximum_total_lift_m=0.070,
            )
            row["swing_clearance_repair"] = swing_repair
        if (
            not collision.accepted
            and collision.maximum_forbidden_body_penetration_m <= 1.0e-6
            and collision.maximum_foot_penetration_m <= 0.015
        ):
            motion, extras, collision, stance_repair = (
                _repair_stance_foot_mesh_clearance(
                    motion,
                    extras,
                    collision,
                    adapter=adapter,
                    target_mesh=field.index,
                    model_path=model_path,
                    joint_names=joint_names,
                )
            )
            row["post_swing_stance_mesh_clearance_repair"] = stance_repair
        if not collision.accepted:
            debug_path = destination / "rejected_collision_debug.npz"
            _save_motion(
                debug_path,
                motion,
                mode="motionbricks_contact_rough_collision_debug",
                extras=extras,
            )
            row["rejected_motion_debug"] = str(debug_path.resolve())
            raise ValueError("rough-terrain motion failed exact collision gate")
        motion, extras = _retime_motion_and_extras(
            motion, extras, temporal_scale=temporal_scale
        )
        motion, extras, retime_refit = _refit_motion_to_sole_targets(
            motion, extras, adapter=adapter, stance_only=True
        )
        row["retimed_sole_refit"] = retime_refit
        subdivisions: list[dict[str, object]] = []
        for _pass in range(3):
            if _retimed_motion_metrics(motion)["maximum_joint_step_rad"] <= 0.18:
                break
            motion, extras, subdivision = _adaptively_subdivide_motion_steps(
                motion, extras, target_joint_step_rad=0.14
            )
            motion, extras, subdivision_refit = _refit_motion_to_sole_targets(
                motion, extras, adapter=adapter, stance_only=True
            )
            subdivisions.append({**subdivision, "refit": subdivision_refit})
        row["adaptive_step_subdivision"] = subdivisions
        collision = audit_stair_motion_collisions(
            motion,
            model_path=model_path,
            target_mesh=field.index,
            joint_names=joint_names,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=0.0,
        )
        row["collision_audit"] = collision.to_dict()
        if not collision.accepted:
            raise ValueError("retimed rough-terrain motion failed collision gate")
        contact = _audit_stance_contact_support(
            motion,
            extras,
            adapter=adapter,
            target_mesh=field.index,
            ground_fallback_height_m=0.0,
        )
        row["stance_contact_audit"] = contact
        motion_metrics = _retimed_motion_metrics(motion)
        row["motion_metrics"] = motion_metrics
        if (
            motion_metrics["maximum_joint_step_rad"] > 0.20
            or motion_metrics["maximum_root_translation_step_m"] > 0.035
            or motion_metrics["maximum_root_rotation_step_rad"] > 0.080
            or motion_metrics["maximum_root_acceleration_m_s2"] > 30.0
        ):
            debug_path = destination / "rejected_smoothness_debug.npz"
            _save_motion(
                debug_path,
                motion,
                mode="motionbricks_contact_rough_smoothness_debug",
                extras=extras,
            )
            row["rejected_smoothness_debug"] = str(debug_path.resolve())
            raise ValueError("rough-terrain motion failed final smoothness gate")
        upper_limb = _upper_limb_motion_metrics(
            motion.joint_position, joint_names, fps=motion.fps
        )
        row["final_upper_limb_metrics"] = upper_limb
        row["upper_limb_accepted"] = _final_upper_limb_accepted(upper_limb)
        if not row["upper_limb_accepted"]:
            raise ValueError("rough-terrain motion failed upper-limb energy gate")
        if not bool(contact["accepted"]):
            debug_path = destination / "rejected_final_contact_debug.npz"
            _save_motion(
                debug_path,
                motion,
                mode="motionbricks_contact_rough_final_contact_debug",
                extras=extras,
            )
            row["rejected_final_contact_debug"] = str(debug_path.resolve())
            raise ValueError("rough-terrain motion failed no-hover contact gate")
        motion_path = destination / "motion.npz"
        _save_motion(
            motion_path,
            motion,
            mode="motionbricks_contact_rough",
            extras=extras,
        )
        row["motion"] = str(motion_path.resolve())
        row["frame_count"] = len(motion.root_position_world)
        row["duration_s"] = float(len(motion.root_position_world) / motion.fps)
        row["status"] = "pending_dense_visual_review"
    except (ValueError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    (destination / "report.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def build(arguments: argparse.Namespace) -> dict[str, object]:
    archive = zarr.open_group(str(arguments.motion_archive.resolve()), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        arguments.model_path.resolve(),
        joint_names,
        maximum_joint_correction_rad=1.30,
        target_tolerance_m=1.0e-4,
        maximum_iterations=96,
        damping=0.004,
        posture_weight=1.0e-5,
    )
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = tuple(arguments.source) or (DEFAULT_SOURCE,)
    profiles = tuple(arguments.profile) or PROFILES
    rows: list[dict[str, object]] = []
    for source in sources:
        source_path = source.expanduser().resolve()
        for profile in profiles:
            destination = output / f"{source_path.stem}__{profile}"
            row = _build_one(
                source_path,
                profile,
                destination=destination,
                adapter=adapter,
                joint_names=joint_names,
                model_path=arguments.model_path.resolve(),
                spacing_m=arguments.terrain_spacing_m,
                temporal_scale=arguments.temporal_scale,
                arm_swing_scale=arguments.arm_swing_scale,
                wrist_swing_scale=arguments.wrist_swing_scale,
                arm_smoothing_sigma_frames=arguments.arm_smoothing_sigma_frames,
                minimum_stance_run_frames=arguments.minimum_stance_run_frames,
                maximum_stance_gap_frames=arguments.maximum_stance_gap_frames,
                maximum_stance_speed_mps=arguments.maximum_stance_speed_mps,
            )
            rows.append(row)
            print(
                f"ROUGH_CONTACT label={row['label']} status={row['status']}"
                + ("" if "error" not in row else f" error={row['error']}"),
                flush=True,
            )
    result = {
        "schema": "motionbricks-contact-rough-manifest/v1",
        "pilot_count": len(rows),
        "pending_visual_review_count": sum(
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
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--profile", choices=PROFILES, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--motion-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--terrain-spacing-m", type=float, default=0.04)
    parser.add_argument("--temporal-scale", type=float, default=1.0)
    parser.add_argument("--arm-swing-scale", type=float, default=0.35)
    parser.add_argument("--wrist-swing-scale", type=float, default=0.10)
    parser.add_argument("--arm-smoothing-sigma-frames", type=float, default=2.0)
    parser.add_argument("--minimum-stance-run-frames", type=int, default=5)
    parser.add_argument("--maximum-stance-gap-frames", type=int, default=1)
    parser.add_argument("--maximum-stance-speed-mps", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.terrain_spacing_m <= 0.0:
        parser.error("--terrain-spacing-m must be positive")
    if arguments.temporal_scale < 1.0:
        parser.error("--temporal-scale must be at least one")
    if not 0.0 <= arguments.arm_swing_scale <= 1.0:
        parser.error("--arm-swing-scale must lie in [0,1]")
    if not 0.0 <= arguments.wrist_swing_scale <= 1.0:
        parser.error("--wrist-swing-scale must lie in [0,1]")
    if arguments.arm_smoothing_sigma_frames < 0.0:
        parser.error("--arm-smoothing-sigma-frames must be nonnegative")
    if arguments.minimum_stance_run_frames < 1:
        parser.error("--minimum-stance-run-frames must be positive")
    if arguments.maximum_stance_gap_frames < 0:
        parser.error("--maximum-stance-gap-frames must be nonnegative")
    if arguments.maximum_stance_speed_mps <= 0.0:
        parser.error("--maximum-stance-speed-mps must be positive")
    result = build(arguments)
    print(
        json.dumps(
            {
                "pilot_count": result["pilot_count"],
                "pending_visual_review_count": result[
                    "pending_visual_review_count"
                ],
                "manifest": str(arguments.output.expanduser().resolve() / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
