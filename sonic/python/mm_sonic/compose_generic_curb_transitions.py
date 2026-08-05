"""Compose an arbitrary alternating curb course from coherent transfer windows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .compose_coherent_block_plan import (
    _phase_trim_pair,
    _repair_exact_mesh_clearance,
    _retarget_composed_seams_to_safe_footfalls,
)
from .compose_privileged_stair_route import (
    MotionSegment,
    _maximum_steps,
    _save_motion,
    concatenate_segments,
    endpoint_seam_cost,
    endpoint_seam_metrics,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_oracle.stair_geometry_warp import warp_archive_clip_to_stair_geometry
from .terrain_oracle.stair_motion_collision_audit import audit_stair_motion_collisions
from .terrain_oracle.stair_support_route import StairSupportRoute, VisibleTread
from .terrain_oracle.stitch import StitchedMotion
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


@dataclass(frozen=True)
class _Window:
    clip_index: int
    start_frame: int
    stop_frame: int
    support_start_xy: np.ndarray
    support_end_xy: np.ndarray
    start_height_m: float
    end_height_m: float
    run_m: float

    @property
    def rise_m(self) -> float:
        return self.end_height_m - self.start_height_m


@dataclass(frozen=True)
class _Candidate:
    transition_index: int
    window: _Window
    motion: StitchedMotion
    quality: float
    audit: object
    mechanics: dict[str, float]

    @property
    def segment(self) -> MotionSegment:
        return MotionSegment(
            label=(
                f"curb_t{self.transition_index:02d}_clip{self.window.clip_index:03d}_"
                f"{self.window.start_frame:04d}_{self.window.stop_frame:04d}"
            ),
            root_position_world=self.motion.root_position_world,
            root_quaternion_world_wxyz=self.motion.root_quaternion_world_wxyz,
            joint_position=self.motion.joint_position,
            provenance=self.motion.provenance,
        )


def _curb_mechanically_accepted(mechanics: dict[str, float]) -> bool:
    """Keep the hard discontinuity gates while allowing energetic curb steps."""

    return bool(
        mechanics["maximum_root_translation_step_m"] <= 0.06
        and mechanics["maximum_root_rotation_step_rad"] <= 0.35
        and mechanics["maximum_joint_step_rad"] <= 0.25
        and mechanics["maximum_root_acceleration_m_s2"] <= 50.0
    )


def _transition_route(
    start_xy: object,
    end_xy: object,
    start_height_m: float,
    end_height_m: float,
) -> StairSupportRoute:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    run = float(np.linalg.norm(end - start))
    if run <= 0.05:
        raise ValueError("curb transition has insufficient run")
    middle = 0.5 * (start + end)
    levels = (
        VisibleTread(
            height_m=float(start_height_m),
            route_start_distance_m=0.0,
            route_stop_distance_m=0.5 * run,
            route_start_xy=np.asarray(start, dtype=np.float32),
            route_stop_xy=np.asarray(middle, dtype=np.float32),
            visible_in_mesh=True,
            left_foothold_center_xy=None,
            right_foothold_center_xy=None,
        ),
        VisibleTread(
            height_m=float(end_height_m),
            route_start_distance_m=0.5 * run,
            route_stop_distance_m=run,
            route_start_xy=np.asarray(middle, dtype=np.float32),
            route_stop_xy=np.asarray(end, dtype=np.float32),
            visible_in_mesh=True,
            left_foothold_center_xy=None,
            right_foothold_center_xy=None,
        ),
    )
    return StairSupportRoute(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        levels=levels,
    )


def _source_windows(
    archive: object,
    *,
    route_catalog: Path,
    excluded: set[int],
    context_frames: int = 0,
) -> tuple[_Window, ...]:
    with np.load(route_catalog.expanduser().resolve(), allow_pickle=False) as data:
        counts = np.asarray(data["level_count"], dtype=np.int16)
        widths = np.asarray(data["level_width_m"], dtype=np.float64)
        heights = np.asarray(data["level_height_m"], dtype=np.float64)
    windows: list[_Window] = []
    for clip in range(len(archive["clip_names"])):
        count = int(counts[clip])
        if (
            clip in excluded
            or str(archive["clip_family"][clip]) != "c490_curb"
            or str(archive["clip_traversal"][clip]) != "up"
            or count < 2
        ):
            continue
        start = int(archive["clip_start_idx"][clip])
        stop = int(archive["clip_end_idx"][clip])
        root = np.asarray(archive["body_pos_w"][start:stop, 0, :2], dtype=np.float64)
        route_vector = root[-1] - root[0]
        route_length = float(np.linalg.norm(route_vector))
        if route_length <= 0.10:
            continue
        direction = route_vector / route_length
        progress = (root - root[0]) @ direction
        level_width = widths[clip, :count]
        level_height = heights[clip, :count]
        centres = np.cumsum(level_width) - 0.5 * level_width
        centre_frames = np.asarray(
            [int(np.argmin(np.abs(progress - value))) for value in centres],
            dtype=np.int64,
        )
        for transition in range(count - 1):
            first = int(centre_frames[transition])
            last = int(centre_frames[transition + 1])
            if last - first < 4:
                continue
            begin = max(0, first - int(context_frames))
            end = min(len(root), last + int(context_frames) + 1)
            support_start = root[first]
            support_end = root[last]
            run = float(np.linalg.norm(support_end - support_start))
            if run <= 0.05:
                continue
            windows.append(
                _Window(
                    clip_index=clip,
                    start_frame=begin,
                    stop_frame=end,
                    support_start_xy=np.asarray(support_start, dtype=np.float32),
                    support_end_xy=np.asarray(support_end, dtype=np.float32),
                    start_height_m=float(level_height[transition]),
                    end_height_m=float(level_height[transition + 1]),
                    run_m=run,
                )
            )
    return tuple(windows)


def _target_transition_routes(route: StairSupportRoute) -> tuple[StairSupportRoute, ...]:
    direction = np.asarray(route.end_xy, dtype=np.float64) - np.asarray(
        route.start_xy, dtype=np.float64
    )
    direction /= np.linalg.norm(direction)
    centres = np.asarray(
        [
            0.5 * (level.route_start_distance_m + level.route_stop_distance_m)
            for level in route.levels
        ],
        dtype=np.float64,
    )
    start = np.asarray(route.start_xy, dtype=np.float64)
    return tuple(
        _transition_route(
            start + centres[index] * direction,
            start + centres[index + 1] * direction,
            route.levels[index].height_m,
            route.levels[index + 1].height_m,
        )
        for index in range(len(route.levels) - 1)
    )


def _has_clear_boundary_context(
    window: _Window,
    *,
    windows: tuple[_Window, ...],
    clip_length: int,
    before_frames: int,
    after_frames: int,
) -> bool:
    """Return whether context stays on the same support level.

    C490 clips may contain a sequence of curbs.  Blindly extending a transfer
    can therefore include the next obstacle rather than flat gait.  Adjacent
    transfer windows overlap by one frame at their shared level centre, so a
    neighboring window entering the requested context interval is sufficient
    to reject it before expensive IK.
    """

    before = int(before_frames)
    after = int(after_frames)
    if window.start_frame < before or window.stop_frame + after > int(clip_length):
        return False
    begin = window.start_frame - before
    end = window.stop_frame + after
    for other in windows:
        if other.clip_index != window.clip_index:
            continue
        if (
            other.start_frame == window.start_frame
            and other.stop_frame == window.stop_frame
        ):
            continue
        if (
            before > 0
            and other.start_frame < window.start_frame
            and other.stop_frame > begin
        ):
            return False
        if (
            after > 0
            and other.start_frame > window.start_frame
            and other.start_frame < end
        ):
            return False
    return True


def _beam_chains(
    pools: list[list[_Candidate]], *, beam_width: int = 8
) -> list[tuple[_Candidate, ...]]:
    beam: list[tuple[float, tuple[_Candidate, ...]]] = [
        (candidate.quality, (candidate,)) for candidate in pools[0]
    ]
    beam.sort(key=lambda value: value[0])
    beam = beam[:beam_width]
    for pool in pools[1:]:
        expanded = []
        for score, chain in beam:
            for candidate in pool:
                expanded.append(
                    (
                        score
                        + candidate.quality
                        + endpoint_seam_cost(chain[-1].segment, candidate.segment),
                        chain + (candidate,),
                    )
                )
        expanded.sort(key=lambda value: value[0])
        beam = expanded[:beam_width]
    return [chain for _, chain in beam]


def compose_transition_fallback(
    *,
    archive_path: Path,
    route_catalog: Path,
    model_path: Path,
    target_mesh: TerrainMeshIndex,
    target_route: StairSupportRoute,
    output_dir: Path,
    excluded_source_clip_indices: tuple[int, ...],
    candidate_limit_per_transition: int = 12,
    accepted_candidates_per_transition: int = 3,
    expanded_transition_indices: tuple[int, ...] = (),
    boundary_context_frames: int = 0,
) -> dict[str, object]:
    import zarr

    archive_source = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(archive_source), mode="r")
    excluded = {int(value) for value in excluded_source_clip_indices}
    windows = _source_windows(
        archive, route_catalog=route_catalog, excluded=excluded
    )
    target_routes = _target_transition_routes(target_route)
    context = int(boundary_context_frames)
    if context < 0:
        raise ValueError("boundary_context_frames must be non-negative")
    attempts: list[dict[str, object]] = []
    pools: list[list[_Candidate]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_attempts: list[dict[str, object]] = []
    expanded = {int(value) for value in expanded_transition_indices}
    prior_summary_path = output_dir / "summary.json"
    if prior_summary_path.is_file():
        try:
            prior = json.loads(prior_summary_path.read_text())
            if isinstance(prior.get("attempts"), list):
                prior_attempts = [
                    dict(row)
                    for row in prior["attempts"]
                    if isinstance(row, dict)
                ]
        except (OSError, ValueError):
            prior_attempts = []
    for transition_index, target in enumerate(target_routes):
        target_run = float(np.linalg.norm(target.end_xy - target.start_xy))
        target_rise = float(
            target.levels[1].height_m - target.levels[0].height_m
        )
        cached_rows = [
            row
            for row in prior_attempts
            if int(row.get("transition_index", -1)) == transition_index
            and row.get("status") == "accepted"
        ]
        cached_files = sorted(
            output_dir.glob(
                f"transition_{transition_index:02d}_candidate_*.npz"
            )
        )
        cached_pool: list[_Candidate] = []
        for cached_index, cached_file in enumerate(cached_files):
            motion = load_stitched_motion_npz(cached_file)
            if cached_index < len(cached_rows):
                row = cached_rows[cached_index]
            else:
                source_clip = int(motion.provenance[0].archive_clip_index)
                source_frames = [
                    int(value.source_frame) for value in motion.provenance
                ]
                row = {
                    "transition_index": transition_index,
                    "source_clip_index": source_clip,
                    "source_frame_range": [
                        min(source_frames),
                        max(source_frames) + 1,
                    ],
                    "source_rise_m": target_rise,
                    "source_run_m": target_run,
                    "geometry_match_score": 0.05 * cached_index,
                    "clearance_repair_maximum_m": 0.0,
                    "mechanics": _maximum_steps(motion),
                    "status": "accepted",
                }
            source_range = row.get("source_frame_range", (0, 1))
            source_run = float(row.get("source_run_m", target_run))
            source_rise = float(row.get("source_rise_m", target_rise))
            window = _Window(
                clip_index=int(row["source_clip_index"]),
                start_frame=int(source_range[0]),
                stop_frame=int(source_range[1]),
                support_start_xy=np.zeros(2, dtype=np.float32),
                support_end_xy=np.asarray((source_run, 0.0), dtype=np.float32),
                start_height_m=0.0,
                end_height_m=source_rise,
                run_m=source_run,
            )
            mechanics = dict(row.get("mechanics", {}))
            cached_pool.append(
                _Candidate(
                    transition_index=transition_index,
                    window=window,
                    motion=motion,
                    quality=float(
                        row.get("geometry_match_score", 0.0)
                        + 10.0 * row.get("clearance_repair_maximum_m", 0.0)
                    ),
                    audit=None,
                    mechanics=mechanics,
                )
            )
            reused = dict(row)
            reused["cache_reused"] = True
            attempts.append(reused)
            if len(cached_pool) >= int(accepted_candidates_per_transition):
                break
        if cached_pool and (
            transition_index not in expanded
            or len(cached_pool) >= int(accepted_candidates_per_transition)
        ):
            pools.append(cached_pool)
            continue
        before_context = context if transition_index == 0 else 0
        after_context = (
            context if transition_index == len(target_routes) - 1 else 0
        )
        ranked = sorted(
            (
                (
                    abs(window.run_m - target_run)
                    + 2.0 * abs(window.rise_m - target_rise),
                    window,
                )
                for window in windows
                if np.sign(window.rise_m) == np.sign(target_rise)
                and _has_clear_boundary_context(
                    window,
                    windows=windows,
                    clip_length=int(
                        archive["clip_end_idx"][window.clip_index]
                        - archive["clip_start_idx"][window.clip_index]
                    ),
                    before_frames=before_context,
                    after_frames=after_context,
                )
            ),
            key=lambda value: (value[0], value[1].clip_index, value[1].start_frame),
        )
        pool: list[_Candidate] = list(cached_pool)
        cached_keys = {
            (
                candidate.window.clip_index,
                candidate.window.start_frame,
                candidate.window.stop_frame,
            )
            for candidate in cached_pool
        }
        for rank, (geometry_cost, core_window) in enumerate(
            ranked[: int(candidate_limit_per_transition)]
        ):
            clip_length = int(
                archive["clip_end_idx"][core_window.clip_index]
                - archive["clip_start_idx"][core_window.clip_index]
            )
            window = _Window(
                clip_index=core_window.clip_index,
                start_frame=(
                    max(0, core_window.start_frame - context)
                    if transition_index == 0
                    else core_window.start_frame
                ),
                stop_frame=(
                    min(clip_length, core_window.stop_frame + context)
                    if transition_index == len(target_routes) - 1
                    else core_window.stop_frame
                ),
                support_start_xy=core_window.support_start_xy,
                support_end_xy=core_window.support_end_xy,
                start_height_m=core_window.start_height_m,
                end_height_m=core_window.end_height_m,
                run_m=core_window.run_m,
            )
            if (
                window.clip_index,
                window.start_frame,
                window.stop_frame,
            ) in cached_keys:
                continue
            row: dict[str, object] = {
                "transition_index": transition_index,
                "rank": rank,
                "source_clip_index": window.clip_index,
                "source_frame_range": [window.start_frame, window.stop_frame],
                "source_rise_m": window.rise_m,
                "target_rise_m": target_rise,
                "source_run_m": window.run_m,
                "target_run_m": target_run,
                "geometry_match_score": geometry_cost,
            }
            try:
                source_route = _transition_route(
                    window.support_start_xy,
                    window.support_end_xy,
                    window.start_height_m,
                    window.end_height_m,
                )
                warped = warp_archive_clip_to_stair_geometry(
                    archive_source,
                    source_clip_index=window.clip_index,
                    source_start_frame=window.start_frame,
                    source_stop_frame=window.stop_frame,
                    target_clip_index=None,
                    target_mesh=target_mesh,
                    target_route_start_xy=target.start_xy,
                    target_route_end_xy=target.end_xy,
                    source_support_route=source_route,
                    target_support_route=target,
                    model_path=model_path,
                    maximum_joint_correction_rad=0.50,
                    # This is an IK-fit residual, not a terrain-safety gate.
                    # A few otherwise useful C490 transfers retain up to 4 cm
                    # of endpoint residual when their rise/run is warped.  The
                    # complete rendered G1 is audited against the exact mesh
                    # immediately below, so rejecting them here only removes
                    # valid source coverage before the real safety check.
                    maximum_foot_target_error_m=0.04,
                    maximum_sole_penetration_m=0.006,
                    maximum_root_clearance_lift_m=0.04,
                    maximum_foothold_progress_shift_m=0.08,
                    foothold_edge_clearance_margin_m=0.02,
                    maximum_foothold_yaw_adjustment_rad=np.deg2rad(50.0),
                    foothold_yaw_search_step_rad=np.deg2rad(10.0),
                    support_contact_tolerance_m=0.025,
                    # A boundary context includes toe-off and landing frames.
                    # The original 0.25 m/s cutoff mislabels a descending
                    # swing foot as planted (the failing case was moving
                    # vertically at 0.23 m/s).  Only genuinely quiet feet are
                    # constrained as stance while the complete mesh audit
                    # below remains unchanged.
                    support_maximum_sole_speed_m_s=(
                        0.05 if context > 0 else 0.25
                    ),
                )
                audit = audit_stair_motion_collisions(
                    warped.motion,
                    archive_path=archive_source,
                    target_mesh=target_mesh,
                    model_path=model_path,
                    maximum_foot_penetration_m=0.005,
                    maximum_forbidden_body_penetration_m=0.0,
                )
                repaired, final_audit, clearance = _repair_exact_mesh_clearance(
                    warped.motion,
                    audit,
                    archive_path=archive_source,
                    target_clip_index=None,
                    target_mesh=target_mesh,
                    model_path=model_path,
                )
                mechanics = _maximum_steps(repaired)
                accepted = bool(
                    final_audit.accepted
                    and _curb_mechanically_accepted(mechanics)
                )
                row.update(
                    {
                        "status": "accepted" if accepted else "rejected",
                        "clearance_repair_maximum_m": float(clearance),
                        "maximum_foot_penetration_m": float(
                            final_audit.maximum_foot_penetration_m
                        ),
                        "maximum_forbidden_body_penetration_m": float(
                            final_audit.maximum_forbidden_body_penetration_m
                        ),
                        "mechanics": mechanics,
                    }
                )
                if accepted:
                    quality = float(
                        geometry_cost
                        + warped.maximum_joint_correction_rad
                        + 10.0 * clearance
                    )
                    pool.append(
                        _Candidate(
                            transition_index,
                            window,
                            repaired,
                            quality,
                            final_audit,
                            mechanics,
                        )
                    )
                    _save_motion(
                        output_dir
                        / f"transition_{transition_index:02d}_candidate_{len(pool)-1:02d}.npz",
                        repaired,
                    )
                    if len(pool) >= int(accepted_candidates_per_transition):
                        attempts.append(row)
                        break
            except (RuntimeError, ValueError) as error:
                row.update({"status": "rejected", "reason": str(error)})
            attempts.append(row)
        pools.append(pool)
        if not pool:
            summary = {
                "schema": "generic-curb-transition-composition/v1",
                "status": "rejected",
                "failed_transition_index": transition_index,
                "transition_count": len(target_routes),
                "boundary_context_frames": context,
                "attempts": attempts,
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            return summary

    adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=0.95,
        target_tolerance_m=2.5e-4,
        maximum_iterations=96,
        damping=0.006,
        posture_weight=2.0e-5,
    )
    trials: list[dict[str, object]] = []
    best = None
    for chain_index, chain in enumerate(_beam_chains(pools, beam_width=8)):
        segments = [candidate.segment for candidate in chain]
        trims = []
        for index in range(len(segments) - 1):
            left, right, receipt = _phase_trim_pair(segments[index], segments[index + 1])
            segments[index] = left
            segments[index + 1] = right
            trims.append(receipt)
        raw_seams = [
            endpoint_seam_metrics(left, right)
            for left, right in zip(segments, segments[1:])
        ]
        for halflife in (0.04, 0.08, 0.12, 0.18):
            try:
                motion = concatenate_segments(segments, halflife_s=halflife)
                motion, contact_receipts = _retarget_composed_seams_to_safe_footfalls(
                    motion,
                    segments,
                    adapter=adapter,
                    maximum_foot_error_m=0.025,
                )
                audit = audit_stair_motion_collisions(
                    motion,
                    archive_path=archive_source,
                    target_mesh=target_mesh,
                    model_path=model_path,
                    maximum_foot_penetration_m=0.005,
                    maximum_forbidden_body_penetration_m=0.0,
                )
                motion, audit, clearance = _repair_exact_mesh_clearance(
                    motion,
                    audit,
                    archive_path=archive_source,
                    target_clip_index=None,
                    target_mesh=target_mesh,
                    model_path=model_path,
                )
                mechanics = _maximum_steps(motion)
                accepted = bool(
                    audit.accepted and _curb_mechanically_accepted(mechanics)
                )
                trial = {
                    "chain_index": chain_index,
                    "source_clip_indices": [c.window.clip_index for c in chain],
                    "source_frame_ranges": [
                        [c.window.start_frame, c.window.stop_frame] for c in chain
                    ],
                    "inertialization_halflife_s": halflife,
                    "phase_context_trims": trims,
                    "raw_seams": raw_seams,
                    "contact_aware_seams": contact_receipts,
                    "clearance_repair_maximum_m": float(clearance),
                    "maximum_foot_penetration_m": float(audit.maximum_foot_penetration_m),
                    "maximum_forbidden_body_penetration_m": float(
                        audit.maximum_forbidden_body_penetration_m
                    ),
                    "mechanics": mechanics,
                    "status": "accepted" if accepted else "rejected",
                }
                trials.append(trial)
                if accepted:
                    score = float(
                        mechanics["maximum_joint_step_rad"]
                        + 0.01 * mechanics["maximum_root_acceleration_m_s2"]
                        + 10.0 * audit.maximum_foot_penetration_m
                    )
                    if best is None or score < best[0]:
                        best = (score, motion, audit, trial)
            except (RuntimeError, ValueError) as error:
                trials.append(
                    {
                        "chain_index": chain_index,
                        "inertialization_halflife_s": halflife,
                        "status": "rejected",
                        "reason": str(error),
                    }
                )
        if best is not None:
            break

    summary: dict[str, object] = {
        "schema": "generic-curb-transition-composition/v1",
        "status": "accepted" if best is not None else "rejected",
        "transition_count": len(target_routes),
        "boundary_context_frames": context,
        "source_window_count": len(windows),
        "pool_sizes": [len(pool) for pool in pools],
        "expanded_transition_indices": sorted(expanded),
        "attempts": attempts,
        "trials": trials,
    }
    if best is not None:
        score, motion, audit, selected = best
        _save_motion(output_dir / "motion.npz", motion)
        summary.update(
            {
                "quality": score,
                "selected": selected,
                "full_body_audit": audit.to_dict(),
                "motion": str((output_dir / "motion.npz").resolve()),
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


__all__ = ("compose_transition_fallback",)
