"""Select phase-compatible coherent stair blocks and compose a full route.

The coverage sweep deliberately answers only whether each target span has a
mechanically feasible source-contiguous motion.  This second stage keeps
several exact-mesh-safe realizations of every selected span, ranks their gait
phase at the shared landing, and only accepts a complete route after the same
full-G1 collision and temporal-continuity gates used by the privileged
composer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_privileged_stair_route import (
    MotionSegment,
    _maximum_steps,
    _save_motion,
    concatenate_segments,
    endpoint_seam_cost,
    endpoint_seam_metrics,
)
from .evaluate_coherent_block_coverage import (
    DEFAULT_FRAGMENT_BANK,
    _SourceBlock,
    _load_source_blocks,
    _reason_group,
)
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.stair_geometry_warp import (
    DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    StairGeometryWarpResult,
    _archive_terrain_index,
    motion_conditioned_stair_support_route,
    warp_archive_clip_to_stair_geometry,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    StairMotionCollisionAuditResult,
    audit_stair_motion_collisions,
)
from .terrain_oracle.stair_support_route import (
    StairSupportRoute,
    VisibleTread,
)
from .terrain_oracle.stitch import StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


@dataclass(frozen=True)
class _AcceptedCandidate:
    block: _SourceBlock
    result: StairGeometryWarpResult
    motion: StitchedMotion
    geometry_cost: float
    clearance_repair_maximum_m: float
    body_clearance_repair_maximum_m: float
    body_clearance_vertical_lift_maximum_m: float
    body_clearance_foot_error_m: float
    root_acceleration_repair_maximum_m: float
    root_acceleration_repair_foot_error_m: float
    pre_repair_audit: StairMotionCollisionAuditResult
    standalone_audit: StairMotionCollisionAuditResult

    @property
    def segment(self) -> MotionSegment:
        motion = self.motion
        return MotionSegment(
            label=(
                f"clip{self.block.archive_clip_index:03d}_"
                f"{self.block.source_start_frame:04d}_"
                f"{self.block.source_stop_frame:04d}"
            ),
            root_position_world=motion.root_position_world,
            root_quaternion_world_wxyz=(
                motion.root_quaternion_world_wxyz
            ),
            joint_position=motion.joint_position,
            provenance=motion.provenance,
        )


def _candidate_quality(candidate: _AcceptedCandidate) -> float:
    result = candidate.result
    return float(
        candidate.geometry_cost
        + result.maximum_joint_correction_rad
        + 10.0 * result.maximum_foot_target_error_m
        + 5.0 * result.maximum_triangle_sphere_penetration_m
        + 2.0 * result.maximum_root_clearance_lift_m
        + 2.0 * result.maximum_foothold_progress_shift_m
        + 0.10 * result.maximum_foothold_yaw_adjustment_rad
        + 10.0 * candidate.clearance_repair_maximum_m
        + 2.0 * candidate.body_clearance_repair_maximum_m
        + 2.0 * candidate.body_clearance_vertical_lift_maximum_m
        + 10.0 * candidate.body_clearance_foot_error_m
        + 10.0 * candidate.root_acceleration_repair_maximum_m
        + 10.0 * candidate.root_acceleration_repair_foot_error_m
    )


def _chain_cost(chain: tuple[_AcceptedCandidate, ...]) -> float:
    return float(
        sum(_candidate_quality(value) for value in chain)
        + sum(
            endpoint_seam_cost(source.segment, target.segment)
            for source, target in zip(chain, chain[1:])
        )
    )


def _slice_segment(
    segment: MotionSegment,
    *,
    start: int = 0,
    stop: int | None = None,
) -> MotionSegment:
    end = len(segment.root_position_world) if stop is None else int(stop)
    begin = int(start)
    return MotionSegment(
        label=f"{segment.label}[{begin}:{end}]",
        root_position_world=segment.root_position_world[begin:end],
        root_quaternion_world_wxyz=(
            segment.root_quaternion_world_wxyz[begin:end]
        ),
        joint_position=segment.joint_position[begin:end],
        provenance=segment.provenance[begin:end],
    )


def _phase_trim_pair(
    source: MotionSegment,
    target: MotionSegment,
    *,
    maximum_trim_frames: int = 60,
) -> tuple[MotionSegment, MotionSegment, dict[str, object]]:
    """Choose compatible samples inside the shared landing context.

    Block ranges deliberately include approach/exit context.  Treating their
    outermost samples as a mandatory seam can join opposite gait phases even
    though both blocks contain a clean matching pose on the same landing.  A
    game-style matcher searches those context windows and then inertializes the
    best actual samples; it does not force whole recorded ranges end-to-start.
    """

    source_count = len(source.root_position_world)
    target_count = len(target.root_position_world)
    source_z = np.asarray(source.root_position_world[:, 2], dtype=np.float64)
    target_z = np.asarray(target.root_position_world[:, 2], dtype=np.float64)
    source_plateau = float(np.median(source_z[-min(15, source_count) :]))
    target_plateau = float(np.median(target_z[: min(15, target_count)]))
    source_stops = range(
        max(2, source_count - int(maximum_trim_frames)),
        source_count + 1,
    )
    target_starts = range(
        0,
        min(int(maximum_trim_frames), target_count - 2) + 1,
    )
    best = None
    for source_stop in source_stops:
        if abs(float(source_z[source_stop - 1]) - source_plateau) > 0.08:
            continue
        source_slice = _slice_segment(source, stop=source_stop)
        for target_start in target_starts:
            if abs(float(target_z[target_start]) - target_plateau) > 0.08:
                continue
            target_slice = _slice_segment(target, start=target_start)
            metrics = endpoint_seam_metrics(source_slice, target_slice)
            cost = endpoint_seam_cost(source_slice, target_slice)
            # Prefer retaining context when two candidate samples are nearly
            # equivalent; the seam metrics remain overwhelmingly dominant.
            cost += 0.001 * (
                source_count - source_stop + target_start
            )
            candidate = (
                float(cost),
                source_stop,
                target_start,
                source_slice,
                target_slice,
                metrics,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        metrics = endpoint_seam_metrics(source, target)
        return source, target, {
            "source_trimmed_tail_frames": 0,
            "target_trimmed_head_frames": 0,
            "metrics": metrics,
        }
    _, source_stop, target_start, source_slice, target_slice, metrics = best
    return source_slice, target_slice, {
        "source_trimmed_tail_frames": source_count - int(source_stop),
        "target_trimmed_head_frames": int(target_start),
        "metrics": metrics,
    }


def _phase_aligned_chain_segments(
    chain: tuple[_AcceptedCandidate, ...],
) -> tuple[list[MotionSegment], list[dict[str, object]]]:
    segments = [candidate.segment for candidate in chain]
    trims: list[dict[str, object]] = []
    for index in range(len(segments) - 1):
        source, target, receipt = _phase_trim_pair(
            segments[index], segments[index + 1]
        )
        segments[index] = source
        segments[index + 1] = target
        trims.append(receipt)
    return segments, trims


def _quintic_blend(value: float) -> float:
    coordinate = float(np.clip(value, 0.0, 1.0))
    return float(
        coordinate**3
        * (10.0 - 15.0 * coordinate + 6.0 * coordinate**2)
    )


def _cosine_blend(value: float) -> float:
    coordinate = float(np.clip(value, 0.0, 1.0))
    return float(0.5 - 0.5 * math.cos(math.pi * coordinate))


def _retarget_composed_seams_to_safe_footfalls(
    motion: StitchedMotion,
    segments: list[MotionSegment],
    *,
    adapter: _G1FootfallAdapter,
    bridge_frames: int = 36,
    swing_clearance_m: float = 0.035,
    maximum_foot_error_m: float = 0.010,
    seam_modes: tuple[str, ...] | None = None,
    temporal_warm_start: bool = False,
) -> tuple[StitchedMotion, list[dict[str, object]]]:
    """Preserve exact-safe feet while pose inertialization changes clips.

    Joint-space inertialization is appropriate for the torso and arms, but it
    is not a valid contact model: carrying a 6 cm pelvis/leg offset into a new
    stair phrase can pull both authored feet through a tread.  At each shared
    landing, move one foot and then the other from the outgoing anchors onto
    the already-audited incoming footfall trajectory.  The legs are solved by
    bounded IK against the complete sole-sphere targets; the generic blend is
    retained for the rest of the character.

    This is still purely kinematic and globally privileged.  It uses only the
    selected motions and their known terrain-safe footfalls, never target-clip
    robot state at runtime.
    """

    if len(segments) != len(motion.seam_indices) + 1:
        raise ValueError("motion seams do not match the selected segments")
    roots = np.asarray(motion.root_position_world, dtype=np.float64).copy()
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints = np.asarray(motion.joint_position, dtype=np.float64).copy()
    receipts: list[dict[str, object]] = []
    maximum_error = 0.0
    maximum_correction = 0.0

    modes = (
        tuple("staggered" for _ in motion.seam_indices)
        if seam_modes is None
        else tuple(str(value) for value in seam_modes)
    )
    if len(modes) != len(motion.seam_indices) or any(
        value not in ("staggered", "source_contact_release")
        for value in modes
    ):
        raise ValueError("contact-aware seam modes are invalid")

    for seam_index, incoming, seam_mode in zip(
        motion.seam_indices, segments[1:], modes, strict=True
    ):
        seam = int(seam_index)
        incoming_count = min(
            len(incoming.root_position_world) - 1,
            len(roots) - seam,
        )
        if seam < 1 or incoming_count < 2:
            raise ValueError("contact-aware seam has insufficient context")
        transition_count = min(int(bridge_frames), incoming_count)
        safe_target_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def safe_targets_at(frame: int) -> tuple[np.ndarray, np.ndarray]:
            cached = safe_target_cache.get(int(frame))
            if cached is not None:
                return cached
            value = adapter.sole_positions_for_pose(
                root_position=np.asarray(
                    incoming.root_position_world[frame], dtype=np.float64
                ),
                root_quaternion_wxyz=np.asarray(
                    incoming.root_quaternion_world_wxyz[frame],
                    dtype=np.float64,
                ),
                joints=np.asarray(
                    incoming.joint_position[frame], dtype=np.float64
                ),
            )
            safe_target_cache[int(frame)] = value
            return value

        anchors = adapter.sole_positions_for_pose(
            root_position=roots[seam - 1],
            root_quaternion_wxyz=quaternions[seam - 1],
            joints=joints[seam - 1],
        )
        endpoint_targets = safe_targets_at(transition_count)
        endpoint_displacements = tuple(
            float(
                np.linalg.norm(
                    np.mean(endpoint_targets[foot], axis=0)
                    - np.mean(anchors[foot], axis=0)
                )
            )
            for foot in range(2)
        )
        selection_frame = max(1, transition_count // 2)
        selection_targets = safe_targets_at(selection_frame)
        selection_displacements = tuple(
            float(
                np.linalg.norm(
                    np.mean(selection_targets[foot], axis=0)
                    - np.mean(anchors[foot], axis=0)
                )
            )
            for foot in range(2)
        )
        release_windows: tuple[tuple[int, int], tuple[int, int]] | None = None
        if seam_mode == "source_contact_release":
            # Release each outgoing contact only when that same foot begins
            # its authored incoming swing.  This is the usual game-animation
            # foot-locking rule: a planted foot never slides merely because a
            # pose blend is decaying, and the correction is paid back while
            # the foot is airborne.
            scan_stop = min(incoming_count, int(round(3.0 * motion.fps)))
            scan_frames = range(0, scan_stop + 1)
            centres = np.asarray(
                [
                    [
                        np.mean(safe_targets_at(frame)[foot], axis=0)
                        for foot in range(2)
                    ]
                    for frame in scan_frames
                ],
                dtype=np.float64,
            )
            windows: list[tuple[int, int]] = []
            liftoffs: list[int] = []
            baseline_count = min(6, len(centres))
            for foot in range(2):
                baseline_z = float(
                    np.median(centres[:baseline_count, foot, 2])
                )
                lift = scan_stop + 1
                for frame in range(1, max(1, scan_stop - 1)):
                    if np.all(
                        centres[frame : frame + 3, foot, 2]
                        - baseline_z
                        > 0.008
                    ):
                        lift = frame
                        break
                liftoffs.append(lift)
                windows.append(
                    (
                        # Start unloading shortly before geometric liftoff.
                        # A longer lead prevents the moving pelvis from
                        # over-extending a still-locked leg, while the
                        # quintic has near-zero weight during early stance.
                        max(1, lift - 14),
                        min(incoming_count, lift + 6),
                    )
                )
            release_windows = (windows[0], windows[1])
            first_foot = int(np.argmin(liftoffs))
        else:
            # Identify the first swing halfway through the bridge.  Looking
            # at the endpoint can accidentally select the second foot once it
            # has already begun the following step, especially on descents.
            first_foot = int(np.argmax(selection_displacements))
        for offset in range(incoming_count):
            incoming_frame = offset + 1
            output_frame = seam + offset
            safe_targets = safe_targets_at(incoming_frame)
            progress = min(
                1.0, float(incoming_frame) / float(transition_count)
            )
            targets: list[np.ndarray] = []
            for foot in range(2):
                if release_windows is not None:
                    release_start, release_stop = release_windows[foot]
                    denominator = max(1, release_stop - release_start)
                    phase_coordinate = (
                        float(incoming_frame - release_start)
                        / float(denominator)
                    )
                    phase = _quintic_blend(phase_coordinate)
                    targets.append(
                        (1.0 - phase) * anchors[foot]
                        + phase * safe_targets[foot]
                    )
                    continue
                if foot == first_foot:
                    phase_coordinate = progress / 0.65
                else:
                    phase_coordinate = 2.0 * progress - 1.0
                phase = _quintic_blend(phase_coordinate)
                vertical_delta = float(
                    np.mean(safe_targets[foot], axis=0)[2]
                    - np.mean(anchors[foot], axis=0)[2]
                )
                target = (
                    (1.0 - phase) * anchors[foot]
                    + phase * safe_targets[foot]
                )
                if vertical_delta > 0.03:
                    # A very small horizontal delay lets the sole clear the
                    # vertical face before advancing, without holding it so
                    # far behind the moving pelvis that IK becomes singular.
                    horizontal_phase = _quintic_blend(
                        (phase_coordinate - 0.03) / 0.97
                    )
                    target[:, :2] = (
                        (1.0 - horizontal_phase) * anchors[foot][:, :2]
                        + horizontal_phase * safe_targets[foot][:, :2]
                    )
                elif vertical_delta < -0.03:
                    # Descending is the converse: carry the foot beyond the
                    # tread edge before lowering it.  A linear diagonal path
                    # clips the nosing, while a large symmetric arc is both
                    # unnatural and often unreachable from the lowering hip.
                    vertical_phase = _quintic_blend(
                        (phase_coordinate - 0.15) / 0.85
                    )
                    target[:, 2] = (
                        (1.0 - vertical_phase) * anchors[foot][:, 2]
                        + vertical_phase * safe_targets[foot][:, 2]
                    )
                horizontal_shift = float(
                    np.linalg.norm(
                        np.mean(safe_targets[foot], axis=0)[:2]
                        - np.mean(anchors[foot], axis=0)[:2]
                    )
                )
                if phase > 0.0 and phase < 1.0 and horizontal_shift > 0.012:
                    target = target.copy()
                    # A fixed low arc intersects a riser when the destination
                    # foothold is itself one tread higher.  Lift above the
                    # higher endpoint, as a standard game foot trajectory
                    # would, rather than linearly cutting through the step.
                    clearance_amplitude = (
                        float(swing_clearance_m)
                        + 0.5 * max(vertical_delta, 0.0)
                    )
                    target[:, 2] += (
                        clearance_amplitude
                        * math.sin(math.pi * phase)
                    )
                targets.append(target)
            adapted, correction, error = adapter.adapt_to_targets(
                root_position=roots[output_frame],
                root_quaternion_wxyz=quaternions[output_frame],
                authored_joints=joints[output_frame],
                sole_targets_world=targets,
                initial_joints=(
                    joints[output_frame - 1]
                    if temporal_warm_start
                    else None
                ),
            )
            joints[output_frame] = adapted
            maximum_error = max(maximum_error, float(error))
            maximum_correction = max(maximum_correction, float(correction))
            if float(error) > float(maximum_foot_error_m):
                raise ValueError(
                    "contact-aware seam misses sole target: "
                    f"{float(error):.6f} m; seam={seam}; "
                    f"output_frame={output_frame}; incoming={incoming.label}"
                )
        receipts.append(
            {
                "seam_index": seam,
                "transition_frames": transition_count,
                "retargeted_frames": incoming_count,
                "first_moving_foot": (
                    "left" if first_foot == 0 else "right"
                ),
                "mode": seam_mode,
                "release_windows": (
                    None
                    if release_windows is None
                    else [list(value) for value in release_windows]
                ),
                "endpoint_foot_displacements_m": list(
                    endpoint_displacements
                ),
                "selection_foot_displacements_m": list(
                    selection_displacements
                ),
            }
        )

    return (
        StitchedMotion(
            fps=motion.fps,
            root_position_world=np.asarray(roots, dtype=np.float32),
            root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
            joint_position=np.asarray(joints, dtype=np.float32),
            provenance=motion.provenance,
            seam_indices=motion.seam_indices,
        ),
        [
            *receipts,
            {
                "maximum_joint_correction_rad": maximum_correction,
                "maximum_foot_target_error_m": maximum_error,
            },
        ],
    )


def _audit_row(audit: StairMotionCollisionAuditResult) -> dict[str, object]:
    return {
        "accepted": bool(audit.accepted),
        "maximum_foot_penetration_m": float(
            audit.maximum_foot_penetration_m
        ),
        "maximum_forbidden_body_penetration_m": float(
            audit.maximum_forbidden_body_penetration_m
        ),
    }


def _mechanically_accepted(mechanics: dict[str, float]) -> bool:
    return bool(
        mechanics["maximum_root_translation_step_m"] <= 0.06
        and mechanics["maximum_root_rotation_step_rad"] <= 0.35
        and mechanics["maximum_joint_step_rad"] <= 0.25
        and mechanics["maximum_root_acceleration_m_s2"] <= 40.0
    )


def _route_from_coverage(
    coverage: dict[str, object],
) -> StairSupportRoute | None:
    payload = coverage.get("target_route")
    if not isinstance(payload, dict):
        return None
    rows = payload.get("levels")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("coverage target route is incomplete")
    levels = tuple(
        VisibleTread(
            height_m=float(row["height_m"]),
            route_start_distance_m=float(row["route_start_distance_m"]),
            route_stop_distance_m=float(row["route_stop_distance_m"]),
            route_start_xy=np.asarray(row["route_start_xy"], dtype=np.float32),
            route_stop_xy=np.asarray(row["route_stop_xy"], dtype=np.float32),
            visible_in_mesh=True,
            left_foothold_center_xy=None,
            right_foothold_center_xy=None,
        )
        for row in rows
    )
    return StairSupportRoute(
        start_xy=np.asarray(payload["start_xy"], dtype=np.float32),
        end_xy=np.asarray(payload["end_xy"], dtype=np.float32),
        levels=levels,
    )


def _smooth_upper_clearance_envelope(
    required_lift_m: np.ndarray,
    *,
    radius_frames: int = 12,
) -> np.ndarray:
    """Smooth a nonnegative lift without ever undercutting the requirement."""

    required = np.maximum(
        np.asarray(required_lift_m, dtype=np.float64), 0.0
    )
    radius = int(radius_frames)
    if required.ndim != 1 or radius < 1:
        raise ValueError("clearance envelope inputs are invalid")
    padded = np.pad(required, (radius, radius), mode="edge")
    dilated = np.asarray(
        [
            np.max(padded[index : index + 2 * radius + 1])
            for index in range(len(required))
        ],
        dtype=np.float64,
    )
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(1.0, radius / 3.0)
    kernel = np.exp(-0.5 * np.square(coordinate / sigma))
    kernel /= np.sum(kernel)
    smooth = np.convolve(
        np.pad(dilated, (radius, radius), mode="edge"),
        kernel,
        mode="valid",
    )
    # Dilation and smoothing have equal support, so this projection should
    # only correct floating-point edge cases while retaining C1-like ramps.
    return np.maximum(smooth, required)


def _repair_exact_mesh_clearance(
    motion: StitchedMotion,
    audit: StairMotionCollisionAuditResult,
    *,
    archive_path: Path,
    target_clip_index: int | None,
    target_mesh: TerrainMeshIndex | None = None,
    model_path: Path,
    maximum_total_lift_m: float = 0.025,
) -> tuple[StitchedMotion, StairMotionCollisionAuditResult, float]:
    """Apply the smallest smooth root-Z envelope needed by exact G1 geoms."""

    repaired = motion
    current = audit
    total = np.zeros(len(motion.root_position_world), dtype=np.float64)
    for _iteration in range(5):
        if current.accepted:
            break
        foot = np.asarray(
            current.per_frame_max_foot_penetration_m, dtype=np.float64
        )
        body = np.asarray(
            current.per_frame_max_forbidden_body_penetration_m,
            dtype=np.float64,
        )
        # The complete rendered foot is targeted to 2.5 mm rather than merely
        # the 5 mm hard gate.  Forbidden-body contact remains a zero-tolerance
        # constraint.  A 0.5 mm margin covers contact-solver roundoff.
        required = np.maximum(foot - 0.0025, body) + 0.0005
        required[(foot <= 0.0025) & (body <= 0.0)] = 0.0
        lift = _smooth_upper_clearance_envelope(required)
        if float(np.max(total + lift)) > float(maximum_total_lift_m):
            break
        total += lift
        roots = np.asarray(
            repaired.root_position_world, dtype=np.float64
        ).copy()
        roots[:, 2] += lift
        repaired = StitchedMotion(
            fps=repaired.fps,
            root_position_world=np.asarray(roots, dtype=np.float32),
            root_quaternion_world_wxyz=(
                repaired.root_quaternion_world_wxyz
            ),
            joint_position=repaired.joint_position,
            provenance=repaired.provenance,
            seam_indices=repaired.seam_indices,
        )
        current = audit_stair_motion_collisions(
            repaired,
            archive_path=archive_path,
            target_clip_index=target_clip_index,
            target_mesh=target_mesh,
            model_path=model_path,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=1.0e-6,
        )
    return repaired, current, float(np.max(total))


def _repair_root_acceleration_spikes(
    motion: StitchedMotion,
    audit: StairMotionCollisionAuditResult,
    *,
    archive_path: Path,
    target_clip_index: int | None,
    target_mesh: TerrainMeshIndex | None = None,
    model_path: Path,
    target_acceleration_m_s2: float = 36.0,
    maximum_root_adjustment_m: float = 0.012,
) -> tuple[StitchedMotion, StairMotionCollisionAuditResult, float, float]:
    """Project isolated root kinks while retaining both world-space soles.

    Geometry warping can introduce a millimetre-scale kink where two constant
    stance transforms meet.  Rejecting an otherwise exact-safe phrase for a
    one-frame 2 mm pelvis artifact wastes good data.  Project the worst second
    difference onto a conservative acceleration ball, then use leg IK to keep
    the original exact sole targets.  The full mesh audit remains final.
    """

    import zarr

    if not audit.accepted or len(motion.root_position_world) < 3:
        return motion, audit, 0.0, 0.0
    original_roots = np.asarray(
        motion.root_position_world, dtype=np.float64
    )
    roots = original_roots.copy()
    fps_squared = float(motion.fps) ** 2
    target = float(target_acceleration_m_s2)
    for _iteration in range(64):
        second_difference = np.diff(roots, n=2, axis=0)
        acceleration = (
            np.linalg.norm(second_difference, axis=1) * fps_squared
        )
        worst = int(np.argmax(acceleration))
        maximum = float(acceleration[worst])
        if maximum <= target + 1.0e-6:
            break
        desired = second_difference[worst] * (target / maximum)
        roots[worst + 1] += 0.5 * (
            second_difference[worst] - desired
        )
        adjustment = np.linalg.norm(roots - original_roots, axis=1)
        if float(np.max(adjustment)) > float(maximum_root_adjustment_m):
            return motion, audit, 0.0, 0.0
    adjustment = np.linalg.norm(roots - original_roots, axis=1)
    maximum_adjustment = float(np.max(adjustment))
    if maximum_adjustment <= 1.0e-8:
        return motion, audit, 0.0, 0.0

    archive = zarr.open_group(str(archive_path), mode="r")
    adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=0.35,
        target_tolerance_m=1.0e-4,
        maximum_iterations=96,
        damping=0.004,
        posture_weight=1.0e-5,
    )
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    original_joints = np.asarray(motion.joint_position, dtype=np.float64)
    joints = original_joints.copy()
    maximum_foot_error = 0.0
    modified = np.flatnonzero(adjustment > 1.0e-8)
    for frame in modified:
        foot_targets = adapter.sole_positions_for_pose(
            root_position=original_roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=original_joints[frame],
        )
        adapted, _, foot_error = adapter.adapt_to_targets(
            root_position=roots[frame],
            root_quaternion_wxyz=quaternions[frame],
            authored_joints=original_joints[frame],
            sole_targets_world=foot_targets,
        )
        joints[frame] = adapted
        maximum_foot_error = max(maximum_foot_error, float(foot_error))
    if maximum_foot_error > 0.002:
        return motion, audit, 0.0, 0.0
    candidate = StitchedMotion(
        fps=motion.fps,
        root_position_world=np.asarray(roots, dtype=np.float32),
        root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=motion.provenance,
        seam_indices=motion.seam_indices,
    )
    candidate_audit = audit_stair_motion_collisions(
        candidate,
        archive_path=archive_path,
        target_clip_index=target_clip_index,
        target_mesh=target_mesh,
        model_path=model_path,
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=1.0e-6,
    )
    if not candidate_audit.accepted:
        return motion, audit, 0.0, 0.0
    if (
        _maximum_steps(candidate)["maximum_root_acceleration_m_s2"]
        >= _maximum_steps(motion)["maximum_root_acceleration_m_s2"]
    ):
        return motion, audit, 0.0, 0.0
    return (
        candidate,
        candidate_audit,
        maximum_adjustment,
        maximum_foot_error,
    )


def _repair_exact_mesh_body_clearance(
    motion: StitchedMotion,
    audit: StairMotionCollisionAuditResult,
    *,
    archive_path: Path,
    target_clip_index: int | None,
    target_mesh: TerrainMeshIndex | None = None,
    model_path: Path,
    maximum_total_setback_m: float = 0.12,
) -> tuple[
    StitchedMotion,
    StairMotionCollisionAuditResult,
    float,
    float,
    float,
]:
    """Retarget the legs after a smooth pelvis move away from a riser.

    Short high treads can be valid footholds even when an authored swing knee
    follows a shallower source stair and clips the next riser.  Raising the
    whole character produces the floating failure mode seen in the viewer.
    For a globally known mesh, the game-style repair is instead to move the
    pelvis away from the obstacle while keeping both world-space foot targets
    fixed with bounded leg IK.  The exact full-G1 collision audit closes the
    loop; no target motion sample participates.
    """

    import zarr

    if (
        audit.maximum_forbidden_body_penetration_m <= 1.0e-6
        or len(motion.root_position_world) < 3
    ):
        return motion, audit, 0.0, 0.0, 0.0
    roots_original = np.asarray(
        motion.root_position_world, dtype=np.float64
    )
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    joints_original = np.asarray(motion.joint_position, dtype=np.float64)
    travel = roots_original[-1, :2] - roots_original[0, :2]
    travel_norm = float(np.linalg.norm(travel))
    if travel_norm < 0.05:
        return motion, audit, 0.0, 0.0, 0.0
    direction = travel / travel_norm
    # Ascents set the pelvis back from the next (higher) riser; descents move
    # it forward, away from the preceding higher riser.
    progress_sign = (
        -1.0
        if float(roots_original[-1, 2] - roots_original[0, 2]) >= 0.0
        else 1.0
    )
    archive = zarr.open_group(str(archive_path), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    adapter = _G1FootfallAdapter(
        model_path,
        joint_names,
        maximum_joint_correction_rad=0.65,
        target_tolerance_m=1.0e-4,
        maximum_iterations=128,
        damping=0.004,
        posture_weight=1.0e-5,
    )
    foot_targets = tuple(
        adapter.sole_positions_for_pose(
            root_position=roots_original[frame],
            root_quaternion_wxyz=quaternions[frame],
            joints=joints_original[frame],
        )
        for frame in range(len(roots_original))
    )
    total = np.zeros(len(roots_original), dtype=np.float64)
    repaired = motion
    current = audit
    maximum_foot_error = 0.0
    for _iteration in range(6):
        body = np.asarray(
            current.per_frame_max_forbidden_body_penetration_m,
            dtype=np.float64,
        )
        if float(np.max(body)) <= 1.0e-6:
            break
        increment_required = np.zeros_like(body)
        active = body > 1.0e-6
        increment_required[active] = 1.10 * body[active] + 0.004
        increment = _smooth_upper_clearance_envelope(
            increment_required,
            radius_frames=24,
        )
        # Follow the feasible IK manifold in short continuation steps.  A
        # single 10 cm pelvis jump can be reachable but fail a local solve;
        # three smooth 3--4 cm updates converge reliably to the same pose.
        if float(np.max(increment)) > 0.025:
            increment *= 0.025 / float(np.max(increment))
        proposed = total + increment
        if float(np.max(proposed)) > float(maximum_total_setback_m):
            scale = (
                float(maximum_total_setback_m) - float(np.max(total))
            ) / max(float(np.max(increment)), 1.0e-9)
            if scale <= 0.0:
                break
            increment *= min(1.0, scale)
            proposed = total + increment
        roots = roots_original.copy()
        roots[:, :2] += (
            progress_sign * proposed[:, None] * direction[None]
        )
        joints = joints_original.copy()
        authored_for_ik = np.asarray(
            repaired.joint_position, dtype=np.float64
        )
        iteration_foot_error = 0.0
        for frame in range(len(roots)):
            adapted, _, foot_error = adapter.adapt_to_targets(
                root_position=roots[frame],
                root_quaternion_wxyz=quaternions[frame],
                authored_joints=authored_for_ik[frame],
                sole_targets_world=foot_targets[frame],
            )
            joints[frame] = adapted
            iteration_foot_error = max(
                iteration_foot_error, float(foot_error)
            )
        # A repair that cannot preserve the authored foothold is not a valid
        # alternative to collision.  Exact contact is still checked below.
        if iteration_foot_error > 0.010:
            break
        candidate = StitchedMotion(
            fps=motion.fps,
            root_position_world=np.asarray(roots, dtype=np.float32),
            root_quaternion_world_wxyz=(
                motion.root_quaternion_world_wxyz
            ),
            joint_position=np.asarray(joints, dtype=np.float32),
            provenance=motion.provenance,
            seam_indices=motion.seam_indices,
        )
        candidate_audit = audit_stair_motion_collisions(
            candidate,
            archive_path=archive_path,
            target_clip_index=target_clip_index,
            target_mesh=target_mesh,
            model_path=model_path,
            maximum_foot_penetration_m=0.005,
            maximum_forbidden_body_penetration_m=1.0e-6,
        )
        if (
            candidate_audit.maximum_forbidden_body_penetration_m
            >= current.maximum_forbidden_body_penetration_m - 1.0e-6
        ):
            break
        total = proposed
        repaired = candidate
        current = candidate_audit
        maximum_foot_error = max(
            maximum_foot_error, iteration_foot_error
        )

    # A horizontal setback cannot always clear both the leading and trailing
    # risers on a short, high tread.  Finish by moving the pelvis vertically
    # while retaining the already-repaired world-space feet with the same leg
    # IK.  This changes body posture rather than lifting the planted feet and
    # therefore avoids the hovering failure of a whole-character root-Z lift.
    vertical_total = np.zeros(len(roots_original), dtype=np.float64)
    if current.maximum_forbidden_body_penetration_m > 1.0e-6:
        vertical_base_roots = np.asarray(
            repaired.root_position_world, dtype=np.float64
        )
        vertical_base_joints = np.asarray(
            repaired.joint_position, dtype=np.float64
        )
        vertical_foot_targets = tuple(
            adapter.sole_positions_for_pose(
                root_position=vertical_base_roots[frame],
                root_quaternion_wxyz=quaternions[frame],
                joints=vertical_base_joints[frame],
            )
            for frame in range(len(vertical_base_roots))
        )
        for _iteration in range(20):
            body = np.asarray(
                current.per_frame_max_forbidden_body_penetration_m,
                dtype=np.float64,
            )
            if float(np.max(body)) <= 1.0e-6:
                break
            increment_required = np.zeros_like(body)
            active = body > 1.0e-6
            increment_required[active] = 1.10 * body[active] + 0.003
            increment = _smooth_upper_clearance_envelope(
                increment_required,
                radius_frames=24,
            )
            if float(np.max(increment)) > 0.003:
                increment *= 0.003 / float(np.max(increment))
            proposed = vertical_total + increment
            maximum_vertical_lift_m = 0.06
            if float(np.max(proposed)) > maximum_vertical_lift_m:
                scale = (
                    maximum_vertical_lift_m
                    - float(np.max(vertical_total))
                ) / max(float(np.max(increment)), 1.0e-9)
                if scale <= 0.0:
                    break
                increment *= min(1.0, scale)
                proposed = vertical_total + increment
            roots = vertical_base_roots.copy()
            roots[:, 2] += proposed
            joints = vertical_base_joints.copy()
            authored_for_ik = np.asarray(
                repaired.joint_position, dtype=np.float64
            )
            iteration_foot_error = 0.0
            for frame in range(len(roots)):
                adapted, _, foot_error = adapter.adapt_to_targets(
                    root_position=roots[frame],
                    root_quaternion_wxyz=quaternions[frame],
                    authored_joints=authored_for_ik[frame],
                    sole_targets_world=vertical_foot_targets[frame],
                )
                joints[frame] = adapted
                iteration_foot_error = max(
                    iteration_foot_error, float(foot_error)
                )
            if iteration_foot_error > 0.010:
                print(
                    json.dumps(
                        {
                            "event": "vertical_body_repair",
                            "status": "ik_rejected",
                            "iteration": _iteration,
                            "maximum_lift_m": float(np.max(proposed)),
                            "maximum_foot_error_m": iteration_foot_error,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
            candidate = StitchedMotion(
                fps=motion.fps,
                root_position_world=np.asarray(roots, dtype=np.float32),
                root_quaternion_world_wxyz=(
                    motion.root_quaternion_world_wxyz
                ),
                joint_position=np.asarray(joints, dtype=np.float32),
                provenance=motion.provenance,
                seam_indices=motion.seam_indices,
            )
            candidate_audit = audit_stair_motion_collisions(
                candidate,
                archive_path=archive_path,
                target_clip_index=target_clip_index,
                target_mesh=target_mesh,
                model_path=model_path,
                maximum_foot_penetration_m=0.005,
                maximum_forbidden_body_penetration_m=1.0e-6,
            )
            print(
                json.dumps(
                    {
                        "event": "vertical_body_repair",
                        "status": "audited",
                        "iteration": _iteration,
                        "maximum_lift_m": float(np.max(proposed)),
                        "maximum_foot_error_m": iteration_foot_error,
                        "body_penetration_before_m": (
                            current.maximum_forbidden_body_penetration_m
                        ),
                        "body_penetration_after_m": (
                            candidate_audit.maximum_forbidden_body_penetration_m
                        ),
                        "foot_penetration_after_m": (
                            candidate_audit.maximum_foot_penetration_m
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                candidate_audit.maximum_forbidden_body_penetration_m
                >= current.maximum_forbidden_body_penetration_m - 1.0e-6
            ):
                break
            vertical_total = proposed
            repaired = candidate
            current = candidate_audit
            maximum_foot_error = max(
                maximum_foot_error, iteration_foot_error
            )
    return (
        repaired,
        current,
        float(np.max(total)),
        float(np.max(vertical_total)),
        float(maximum_foot_error),
    )


def compose(
    *,
    coverage_summary: Path,
    archive_path: Path,
    model_path: Path,
    fragment_bank: Path,
    route_catalog: Path,
    candidate_limit_per_span: int,
    accepted_candidates_per_span: int,
    maximum_candidate_chains: int,
    output_dir: Path,
    render: bool,
    target_mesh: TerrainMeshIndex | None = None,
    target_name: str | None = None,
    candidate_start_ranks: tuple[int, ...] = (),
) -> dict[str, object]:
    import zarr

    coverage = json.loads(coverage_summary.read_text())
    if not bool(coverage.get("supported")):
        raise ValueError("coverage summary has no coherent block plan")
    plan_rows = tuple(coverage["plan"])
    if not plan_rows:
        raise ValueError("coverage summary has an empty block plan")
    target_payload = coverage.get("target_clip_index")
    target = None if target_payload is None else int(target_payload)
    traversal = str(coverage["traversal"])
    archive = zarr.open_group(str(archive_path), mode="r")
    if target_mesh is None:
        if target is None:
            raise ValueError(
                "generic coherent composition requires an explicit target mesh"
            )
        target_mesh = _archive_terrain_index(archive, target)
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    route = _route_from_coverage(coverage)
    if route is None:
        if target is None:
            raise ValueError("coverage summary does not contain a target route")
        route = motion_conditioned_stair_support_route(archive, target)
    direction = np.asarray(route.end_xy, dtype=np.float64) - np.asarray(
        route.start_xy, dtype=np.float64
    )
    direction /= np.linalg.norm(direction)
    centres = np.asarray(
        [
            0.5
            * (
                float(level.route_start_distance_m)
                + float(level.route_stop_distance_m)
            )
            for level in route.levels
        ],
        dtype=np.float64,
    )
    heights = np.asarray(
        [float(level.height_m) for level in route.levels], dtype=np.float64
    )
    target_runs = np.diff(centres)
    target_rises = np.diff(heights)
    maximum_transitions = max(int(row["transition_count"]) for row in plan_rows)
    excluded_sources = {
        int(value)
        for value in coverage.get("excluded_source_clip_indices", ())
    }
    if bool(coverage.get("exclude_target_source", True)) and target is not None:
        excluded_sources.add(target)
    source_blocks = tuple(
        block
        for block in _load_source_blocks(
            fragment_bank,
            route_catalog,
            maximum_transitions=maximum_transitions,
        )
        if block.traversal == traversal
        and block.archive_clip_index not in excluded_sources
    )
    terrain_line_route = coverage.get("target_route_kind") in {
        "global_scene_line",
        "arbitrary_mesh_line",
    }

    attempt_rows: list[dict[str, object]] = []
    pools: list[tuple[_AcceptedCandidate, ...]] = []
    route_start = np.asarray(route.start_xy, dtype=np.float64)
    for span_index, row in enumerate(plan_rows):
        start_level = int(row["target_start_level"])
        transitions = int(row["transition_count"])
        stop_level = start_level + transitions
        wanted_run = target_runs[start_level:stop_level]
        wanted_rise = target_rises[start_level:stop_level]
        ranked = [
            (
                float(
                    np.mean(np.abs(block.transition_run_m - wanted_run))
                    + 2.0
                    * np.mean(
                        np.abs(block.transition_rise_m - wanted_rise)
                    )
                ),
                block,
            )
            for block in source_blocks
            if block.transition_count == transitions
        ]
        planned_source = (
            int(row["source_clip_index"]),
            int(row["source_range"][0]),
            int(row["source_range"][1]),
        )
        ranked.sort(
            key=lambda item: (
                (
                    item[1].archive_clip_index,
                    item[1].source_start_frame,
                    item[1].source_stop_frame,
                )
                != planned_source,
                item[0],
                item[1].archive_clip_index,
                item[1].source_start_frame,
            )
        )
        target_start_xy = route_start + centres[start_level] * direction
        target_stop_xy = route_start + centres[stop_level] * direction
        accepted: list[_AcceptedCandidate] = []
        candidate_start = (
            int(candidate_start_ranks[span_index])
            if span_index < len(candidate_start_ranks)
            else 0
        )
        for candidate_rank, (geometry_cost, block) in enumerate(
            ranked[
                candidate_start : candidate_start
                + int(candidate_limit_per_span)
            ],
            start=candidate_start,
        ):
            base = {
                "span_index": span_index,
                "target_start_level": start_level,
                "target_stop_level": stop_level,
                "candidate_rank": candidate_rank,
                "source_clip_index": block.archive_clip_index,
                "source_range": [
                    block.source_start_frame,
                    block.source_stop_frame,
                ],
                "geometry_cost": geometry_cost,
            }
            try:
                warped = warp_archive_clip_to_stair_geometry(
                    archive_path,
                    source_clip_index=block.archive_clip_index,
                    source_start_frame=block.source_start_frame,
                    source_stop_frame=block.source_stop_frame,
                    target_clip_index=target,
                    target_mesh=target_mesh,
                    target_route_start_xy=target_start_xy,
                    target_route_end_xy=target_stop_xy,
                    model_path=model_path,
                    maximum_joint_correction_rad=0.50,
                    maximum_foot_target_error_m=0.003,
                    # Match the coverage-stage proxy tolerance.  The complete
                    # MuJoCo G1/mesh audit below still enforces the 5 mm final
                    # foot limit, so recomposition cannot reject a planned
                    # block merely because two proxy stages used 5 vs 6 mm.
                    maximum_sole_penetration_m=0.006,
                    maximum_root_clearance_lift_m=0.025,
                    maximum_foothold_progress_shift_m=(
                        0.08
                        if terrain_line_route
                        else 0.0
                    ),
                    foothold_edge_clearance_margin_m=(
                        0.02
                        if terrain_line_route
                        else 0.0
                    ),
                    maximum_foothold_yaw_adjustment_rad=(
                        math.radians(50.0)
                        if terrain_line_route
                        else 0.0
                    ),
                    foothold_yaw_search_step_rad=math.radians(10.0),
                    minimum_stance_support_points=1,
                )
            except ValueError as error:
                attempt = {
                    **base,
                    "status": "warp_rejected",
                    "reason_group": _reason_group(str(error)),
                    "reason": str(error),
                }
                attempt_rows.append(attempt)
                print(
                    json.dumps(
                        {
                            "event": "candidate",
                            "span_index": span_index,
                            "candidate_rank": candidate_rank,
                            "source_clip_index": block.archive_clip_index,
                            "status": attempt["status"],
                            "reason_group": attempt["reason_group"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            # Geometry warping owns a temporary MuJoCo IK model.  Reclaim it
            # before compiling/caching the exact face-wise collision scene;
            # otherwise large stair meshes can briefly hold both allocations.
            gc.collect()
            pre_repair = audit_stair_motion_collisions(
                warped.motion,
                archive_path=archive_path,
                target_clip_index=target,
                target_mesh=target_mesh,
                model_path=model_path,
                maximum_foot_penetration_m=0.005,
                maximum_forbidden_body_penetration_m=1.0e-6,
            )
            body_motion = warped.motion
            body_audit = pre_repair
            body_clearance_repair = 0.0
            body_clearance_vertical_lift = 0.0
            body_clearance_foot_error = 0.0
            if terrain_line_route:
                (
                    body_motion,
                    body_audit,
                    body_clearance_repair,
                    body_clearance_vertical_lift,
                    body_clearance_foot_error,
                ) = _repair_exact_mesh_body_clearance(
                    body_motion,
                    body_audit,
                    archive_path=archive_path,
                    target_clip_index=target,
                    target_mesh=target_mesh,
                    model_path=model_path,
                )
            repaired_motion, standalone, clearance_repair = (
                _repair_exact_mesh_clearance(
                    body_motion,
                    body_audit,
                    archive_path=archive_path,
                    target_clip_index=target,
                    target_mesh=target_mesh,
                    model_path=model_path,
                )
            )
            repaired_mechanics = _maximum_steps(repaired_motion)
            root_acceleration_repair = 0.0
            root_acceleration_foot_error = 0.0
            if (
                standalone.accepted
                and repaired_mechanics["maximum_root_acceleration_m_s2"]
                > 40.0
            ):
                (
                    repaired_motion,
                    standalone,
                    root_acceleration_repair,
                    root_acceleration_foot_error,
                ) = _repair_root_acceleration_spikes(
                    repaired_motion,
                    standalone,
                    archive_path=archive_path,
                    target_clip_index=target,
                    target_mesh=target_mesh,
                    model_path=model_path,
                )
                repaired_mechanics = _maximum_steps(repaired_motion)
            attempt = {
                **base,
                "status": (
                    "accepted" if standalone.accepted else "exact_audit_rejected"
                ),
                    "pre_repair_full_body_audit": _audit_row(pre_repair),
                    "body_clearance_repair_maximum_m": (
                        body_clearance_repair
                    ),
                    "body_clearance_vertical_lift_maximum_m": (
                        body_clearance_vertical_lift
                    ),
                    "body_clearance_foot_error_m": (
                        body_clearance_foot_error
                    ),
                    "body_repaired_full_body_audit": _audit_row(
                        body_audit
                    ),
                    "clearance_repair_maximum_m": clearance_repair,
                    "root_acceleration_repair_maximum_m": (
                        root_acceleration_repair
                    ),
                    "root_acceleration_repair_foot_error_m": (
                        root_acceleration_foot_error
                    ),
                    "standalone_full_body_audit": _audit_row(standalone),
                    "repaired_mechanics": repaired_mechanics,
                    "maximum_joint_correction_rad": float(
                        warped.maximum_joint_correction_rad
                    ),
                    "maximum_foothold_progress_shift_m": float(
                        warped.maximum_foothold_progress_shift_m
                    ),
                    "maximum_foothold_yaw_adjustment_rad": float(
                        warped.maximum_foothold_yaw_adjustment_rad
                    ),
                }
            attempt_rows.append(attempt)
            print(
                json.dumps(
                    {
                        "event": "candidate",
                        "span_index": span_index,
                        "candidate_rank": candidate_rank,
                        "source_clip_index": block.archive_clip_index,
                        "status": attempt["status"],
                        "exact_accepted": bool(standalone.accepted),
                        "mechanics_accepted": bool(
                            _mechanically_accepted(repaired_mechanics)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not standalone.accepted or not _mechanically_accepted(
                repaired_mechanics
            ):
                continue
            accepted.append(
                _AcceptedCandidate(
                    block,
                    warped,
                    repaired_motion,
                    geometry_cost,
                    clearance_repair,
                    body_clearance_repair,
                    body_clearance_vertical_lift,
                    body_clearance_foot_error,
                    root_acceleration_repair,
                    root_acceleration_foot_error,
                    pre_repair,
                    standalone,
                )
            )
            if len(accepted) >= int(accepted_candidates_per_span):
                break
        if not accepted:
            pools.append(())
            break
        accepted.sort(key=_candidate_quality)
        pools.append(tuple(accepted))

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the exact-safe constituents even if every attempted seam fails.
    # These are useful visual/debug artifacts and avoid another expensive
    # geometry-warp pass merely to inspect the two sides of a rejected join.
    for span_index, pool in enumerate(pools):
        for candidate_index, candidate in enumerate(pool):
            _save_motion(
                output_dir
                / (
                    f"accepted_span_{span_index:02d}_"
                    f"candidate_{candidate_index:02d}.npz"
                ),
                candidate.motion,
            )
    trial_rows: list[dict[str, object]] = []
    selected = None
    if len(pools) == len(plan_rows) and all(pools):
        contact_adapter = _G1FootfallAdapter(
            model_path,
            tuple(str(value) for value in archive["joint_names"][:]),
            maximum_joint_correction_rad=0.95,
            target_tolerance_m=2.5e-4,
            maximum_iterations=96,
            damping=0.006,
            posture_weight=2.0e-5,
        )
        chains = sorted(
            (tuple(values) for values in product(*pools)),
            key=_chain_cost,
        )[: int(maximum_candidate_chains)]
        halflives = (0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25)
        for chain_index, chain in enumerate(chains):
            segments, phase_trims = _phase_aligned_chain_segments(chain)
            seam_rows = [
                endpoint_seam_metrics(source, target)
                for source, target in zip(segments, segments[1:])
            ]
            trial_halflives = (None,) if len(chain) == 1 else halflives
            for halflife in trial_halflives:
                if len(chain) == 1:
                    motion = chain[0].motion
                else:
                    motion = concatenate_segments(
                        segments,
                        fps=50.0,
                        halflife_s=float(halflife),
                    )
                    try:
                        motion, contact_receipts = (
                            _retarget_composed_seams_to_safe_footfalls(
                                motion,
                                segments,
                                adapter=contact_adapter,
                            )
                        )
                    except ValueError as error:
                        trial_rows.append(
                            {
                                "chain_index": chain_index,
                                "chain_cost": _chain_cost(chain),
                                "source_clip_indices": [
                                    value.block.archive_clip_index
                                    for value in chain
                                ],
                                "inertialization_halflife_s": halflife,
                                "raw_seams": seam_rows,
                                "phase_context_trims": phase_trims,
                                "status": "contact_retarget_rejected",
                                "reason": str(error),
                            }
                        )
                        continue
                mechanics = _maximum_steps(motion)
                trial: dict[str, object] = {
                    "chain_index": chain_index,
                    "chain_cost": _chain_cost(chain),
                    "source_clip_indices": [
                        value.block.archive_clip_index for value in chain
                    ],
                    "inertialization_halflife_s": halflife,
                    "raw_seams": seam_rows,
                    "phase_context_trims": phase_trims,
                    "mechanics": mechanics,
                }
                if len(chain) > 1:
                    trial["contact_aware_seams"] = contact_receipts
                if not _mechanically_accepted(mechanics):
                    trial["status"] = "mechanics_rejected"
                    trial_rows.append(trial)
                    continue
                audit = audit_stair_motion_collisions(
                    motion,
                    archive_path=archive_path,
                    target_clip_index=target,
                    target_mesh=target_mesh,
                    model_path=model_path,
                    maximum_foot_penetration_m=0.005,
                    maximum_forbidden_body_penetration_m=1.0e-6,
                )
                trial["full_body_audit"] = _audit_row(audit)
                trial["status"] = (
                    "accepted" if audit.accepted else "exact_audit_rejected"
                )
                trial_rows.append(trial)
                if audit.accepted:
                    quality = float(
                        _chain_cost(chain)
                        + mechanics["maximum_seam_root_acceleration_m_s2"] / 20.0
                        + mechanics["maximum_seam_joint_acceleration_rad_s2"] / 180.0
                        + audit.maximum_foot_penetration_m / 0.005
                    )
                    candidate = (quality, motion, audit, chain, halflife)
                    if selected is None or quality < selected[0]:
                        selected = candidate

    artifacts: dict[str, str] = {}
    selected_rows: list[dict[str, object]] = []
    if selected is not None:
        _, motion, audit, chain, halflife = selected
        _save_motion(output_dir / "motion.npz", motion)
        artifacts["motion"] = "motion.npz"
        for index, candidate in enumerate(chain):
            block_motion = candidate.motion
            _save_motion(output_dir / f"selected_block_{index:02d}.npz", block_motion)
            selected_rows.append(
                {
                    "span_index": index,
                    "source_clip_index": candidate.block.archive_clip_index,
                    "source_range": [
                        candidate.block.source_start_frame,
                        candidate.block.source_stop_frame,
                    ],
                    "geometry_cost": candidate.geometry_cost,
                    "clearance_repair_maximum_m": (
                        candidate.clearance_repair_maximum_m
                    ),
                    "body_clearance_repair_maximum_m": (
                        candidate.body_clearance_repair_maximum_m
                    ),
                    "body_clearance_vertical_lift_maximum_m": (
                        candidate.body_clearance_vertical_lift_maximum_m
                    ),
                    "body_clearance_foot_error_m": (
                        candidate.body_clearance_foot_error_m
                    ),
                    "root_acceleration_repair_maximum_m": (
                        candidate.root_acceleration_repair_maximum_m
                    ),
                    "root_acceleration_repair_foot_error_m": (
                        candidate.root_acceleration_repair_foot_error_m
                    ),
                    "maximum_foothold_progress_shift_m": float(
                        candidate.result.maximum_foothold_progress_shift_m
                    ),
                    "maximum_foothold_yaw_adjustment_rad": float(
                        candidate.result.maximum_foothold_yaw_adjustment_rad
                    ),
                    "pre_repair_full_body_audit": _audit_row(
                        candidate.pre_repair_audit
                    ),
                    "standalone_full_body_audit": _audit_row(
                        candidate.standalone_audit
                    ),
                }
            )
        if render:
            media = render_stitched_motion(
                motion,
                archive_path=archive_path,
                target_clip_index=target,
                target_mesh=target_mesh,
                model_path=model_path,
                output_path=output_dir / "g1_kinematic_50fps.mp4",
            )
            artifacts["video"] = Path(str(media["video"])).name
        status = "accepted"
        final_audit = _audit_row(audit)
        mechanics = _maximum_steps(motion)
        selected_halflife = halflife
    else:
        status = "no_exact_safe_composition"
        final_audit = None
        mechanics = None
        selected_halflife = None

    report = {
        "schema": "coherent-stair-block-composition/v1",
        "status": status,
        "target_clip_index": target,
        "target_name": (
            str(archive["clip_names"][target])
            if target is not None
            else str(target_name or coverage.get("target_name", "arbitrary-terrain"))
        ),
        "traversal": traversal,
        "coverage_summary": str(coverage_summary.resolve()),
        "plan_block_count": len(plan_rows),
        "candidate_pool_sizes": [len(pool) for pool in pools],
        "candidate_limit_per_span": int(candidate_limit_per_span),
        "candidate_start_ranks": [
            int(value) for value in candidate_start_ranks
        ],
        "accepted_candidates_per_span": int(accepted_candidates_per_span),
        "selected_inertialization_halflife_s": selected_halflife,
        "selected_blocks": selected_rows,
        "mechanics": mechanics,
        "full_body_audit": final_audit,
        "candidate_attempts": attempt_rows,
        "composition_trials": trial_rows,
        "artifacts": artifacts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--fragment-bank", type=Path, default=DEFAULT_FRAGMENT_BANK)
    parser.add_argument(
        "--route-catalog",
        type=Path,
        default=DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    )
    parser.add_argument("--candidate-limit-per-span", type=int, default=32)
    parser.add_argument(
        "--candidate-start-ranks",
        type=str,
        default="",
        help="Comma-separated starting rank for each planned span.",
    )
    parser.add_argument("--accepted-candidates-per-span", type=int, default=4)
    parser.add_argument("--maximum-candidate-chains", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    report = compose(
        coverage_summary=args.coverage_summary.expanduser().resolve(),
        archive_path=args.archive.expanduser().resolve(),
        model_path=args.model.expanduser().resolve(),
        fragment_bank=args.fragment_bank.expanduser().resolve(),
        route_catalog=args.route_catalog.expanduser().resolve(),
        candidate_limit_per_span=args.candidate_limit_per_span,
        accepted_candidates_per_span=args.accepted_candidates_per_span,
        maximum_candidate_chains=args.maximum_candidate_chains,
        output_dir=args.output_dir.expanduser().resolve(),
        render=not args.no_render,
        candidate_start_ranks=tuple(
            int(value)
            for value in args.candidate_start_ranks.split(",")
            if value.strip()
        ),
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"candidate_attempts", "composition_trials"}
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
