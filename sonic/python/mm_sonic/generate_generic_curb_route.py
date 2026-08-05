"""Warp one coherent clean C490 curb course onto an arbitrary exact mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_c490_curb_assets import DEFAULT_ROUTE_CATALOG
from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_coherent_block_plan import _repair_exact_mesh_clearance
from .compose_generic_curb_transitions import compose_transition_fallback
from .compose_privileged_stair_route import _maximum_steps, _save_motion
from .generate_generic_stair_route import load_target_mesh
from .terrain_oracle.stair_geometry_warp import (
    _archive_terrain_index,
    _level_boundaries,
    warp_archive_clip_to_stair_geometry,
)
from .terrain_oracle.stair_motion_collision_audit import audit_stair_motion_collisions
from .terrain_oracle.stair_support_route import sample_stair_support_route


def _geometry_cost(
    target_widths: np.ndarray,
    target_heights: np.ndarray,
    source_widths: np.ndarray,
    source_heights: np.ndarray,
) -> float:
    target_rises = np.diff(target_heights)
    source_rises = np.diff(source_heights)
    return float(
        np.mean(np.abs(source_widths - target_widths))
        + 2.0 * np.mean(np.abs(source_rises - target_rises))
    )


def generate(
    *,
    terrain_usd: Path,
    terrain_position: tuple[float, float, float],
    terrain_quaternion_wxyz: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    output_dir: Path,
    archive_path: Path = DEFAULT_ARCHIVE,
    route_catalog: Path = DEFAULT_ROUTE_CATALOG,
    model_path: Path = DEFAULT_G1_MJCF,
    excluded_source_clip_indices: tuple[int, ...] = (),
    candidate_limit: int = 12,
    accepted_candidates_per_transition: int = 3,
    expanded_transition_indices: tuple[int, ...] = (),
    transition_only: bool = False,
    boundary_context_frames: int = 0,
) -> dict[str, object]:
    import zarr

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target_mesh = load_target_mesh(
        terrain_usd,
        position_world=terrain_position,
        quaternion_world_from_usd_wxyz=terrain_quaternion_wxyz,
    )
    target_route = sample_stair_support_route(
        target_mesh,
        start_xy,
        end_xy,
        0.0,
        0.10,
        0.055,
        sample_spacing_m=0.01,
    )
    target_count = len(target_route.levels)
    if target_count < 2:
        raise ValueError("generic curb generation needs at least one support transition")
    target_widths = np.diff(_level_boundaries(target_route))
    target_heights = np.asarray(
        [level.height_m for level in target_route.levels], dtype=np.float64
    )

    archive_source = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(archive_source), mode="r")
    with np.load(route_catalog.expanduser().resolve(), allow_pickle=False) as data:
        counts = np.asarray(data["level_count"], dtype=np.int16)
        widths = np.asarray(data["level_width_m"], dtype=np.float64)
        heights = np.asarray(data["level_height_m"], dtype=np.float64)
    excluded = {int(value) for value in excluded_source_clip_indices}
    candidates: list[tuple[float, int]] = []
    for source in range(len(archive["clip_names"])):
        if (
            source in excluded
            or str(archive["clip_family"][source]) != "c490_curb"
            or str(archive["clip_traversal"][source]) != "up"
            or int(counts[source]) != target_count
        ):
            continue
        candidates.append(
            (
                _geometry_cost(
                    target_widths,
                    target_heights,
                    widths[source, :target_count],
                    heights[source, :target_count],
                ),
                source,
            )
        )
    candidates.sort(key=lambda value: (value[0], value[1]))

    attempts: list[dict[str, object]] = []
    accepted: list[tuple[float, object, object, float, dict[str, object]]] = []
    whole_candidates = () if transition_only else candidates[: int(candidate_limit)]
    for rank, (geometry_cost, source) in enumerate(whole_candidates):
        clip_start = int(archive["clip_start_idx"][source])
        clip_stop = int(archive["clip_end_idx"][source])
        root = np.asarray(archive["body_pos_w"][clip_start:clip_stop, 0])
        source_mesh = _archive_terrain_index(archive, source)
        source_route = sample_stair_support_route(
            source_mesh,
            root[0, :2],
            root[-1, :2],
            0.0,
            0.10,
            0.055,
            sample_spacing_m=0.01,
        )
        attempt: dict[str, object] = {
            "rank": rank,
            "source_clip_index": source,
            "source_clip_name": str(archive["clip_names"][source]),
            "source_frame_range": [0, clip_stop - clip_start],
            "source_support_level_count": len(source_route.levels),
            "geometry_match_score": geometry_cost,
        }
        try:
            warped = warp_archive_clip_to_stair_geometry(
                archive_source,
                source_clip_index=source,
                source_start_frame=0,
                source_stop_frame=clip_stop - clip_start,
                target_clip_index=None,
                target_mesh=target_mesh,
                target_route_start_xy=start_xy,
                target_route_end_xy=end_xy,
                source_support_route=source_route,
                target_support_route=target_route,
                model_path=model_path,
                maximum_joint_correction_rad=0.50,
                maximum_foot_target_error_m=0.003,
                maximum_sole_penetration_m=0.006,
                maximum_root_clearance_lift_m=0.025,
                maximum_foothold_progress_shift_m=0.08,
                foothold_edge_clearance_margin_m=0.02,
                maximum_foothold_yaw_adjustment_rad=np.deg2rad(50.0),
                foothold_yaw_search_step_rad=np.deg2rad(10.0),
                support_contact_tolerance_m=0.025,
            )
            audit = audit_stair_motion_collisions(
                warped.motion,
                archive_path=archive_source,
                target_mesh=target_mesh,
                model_path=model_path,
                maximum_foot_penetration_m=0.005,
                maximum_forbidden_body_penetration_m=0.0,
            )
            repaired, final_audit, clearance_lift = _repair_exact_mesh_clearance(
                warped.motion,
                audit,
                archive_path=archive_source,
                target_clip_index=None,
                target_mesh=target_mesh,
                model_path=model_path,
            )
            mechanics = _maximum_steps(repaired)
            mechanically_safe = bool(
                mechanics["maximum_root_translation_step_m"] <= 0.06
                and mechanics["maximum_root_rotation_step_rad"] <= 0.35
                and mechanics["maximum_joint_step_rad"] <= 0.25
                and mechanics["maximum_root_acceleration_m_s2"] <= 40.0
            )
            attempt.update(
                {
                    "warp": {
                        "maximum_joint_correction_rad": float(
                            warped.maximum_joint_correction_rad
                        ),
                        "maximum_foot_target_error_m": float(
                            warped.maximum_foot_target_error_m
                        ),
                        "maximum_root_clearance_lift_m": float(
                            warped.maximum_root_clearance_lift_m
                        ),
                    },
                    "clearance_repair_maximum_m": float(clearance_lift),
                    "full_body_audit": final_audit.to_dict(),
                    "mechanics": mechanics,
                    "status": (
                        "accepted" if final_audit.accepted and mechanically_safe else "rejected"
                    ),
                }
            )
            if final_audit.accepted and mechanically_safe:
                quality = float(
                    geometry_cost
                    + 2.0 * warped.maximum_joint_correction_rad
                    + 10.0 * clearance_lift
                )
                accepted.append((quality, repaired, final_audit, clearance_lift, attempt))
        except (RuntimeError, ValueError) as error:
            attempt.update({"status": "rejected", "reason": str(error)})
        attempts.append(attempt)

    base_summary = {
        "schema": "generic-curb-route/v1",
        "terrain_usd": str(terrain_usd.expanduser().resolve()),
        "target_clip_index": None,
        "target_support_level_count": target_count,
        "target_level_width_m": target_widths.tolist(),
        "target_level_height_m": target_heights.tolist(),
        "excluded_source_clip_indices": sorted(excluded),
        "attempts": attempts,
    }
    if not accepted:
        fallback = compose_transition_fallback(
            archive_path=archive_source,
            route_catalog=route_catalog,
            model_path=model_path,
            target_mesh=target_mesh,
            target_route=target_route,
            output_dir=destination,
            excluded_source_clip_indices=excluded_source_clip_indices,
            accepted_candidates_per_transition=(
                accepted_candidates_per_transition
            ),
            expanded_transition_indices=expanded_transition_indices,
            boundary_context_frames=boundary_context_frames,
        )
        summary = {
            **base_summary,
            **fallback,
            "schema": "generic-curb-route/v1",
            "method": (
                "contact_aware_curb_transition_composition"
                if fallback["status"] == "accepted"
                else None
            ),
            "whole_course_attempts": attempts,
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary

    accepted.sort(key=lambda value: value[0])
    quality, motion, audit, clearance_lift, selected = accepted[0]
    _save_motion(destination / "motion.npz", motion)
    summary = {
        **base_summary,
        "status": "accepted",
        "method": "whole_coherent_curb_course_warp",
        "selected": selected,
        "quality": float(quality),
        "clearance_repair_maximum_m": float(clearance_lift),
        "full_body_audit": audit.to_dict(),
        "motion": str((destination / "motion.npz").resolve()),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-usd", type=Path, required=True)
    parser.add_argument("--terrain-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument("--terrain-quaternion-wxyz", type=float, nargs=4, default=(1, 0, 0, 0))
    parser.add_argument("--start-xy", type=float, nargs=2, required=True)
    parser.add_argument("--end-xy", type=float, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--route-catalog", type=Path, default=DEFAULT_ROUTE_CATALOG)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--exclude-source-clip-index", type=int, action="append", default=[])
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument(
        "--accepted-candidates-per-transition", type=int, default=3
    )
    parser.add_argument(
        "--expand-transition-index", type=int, action="append", default=[]
    )
    parser.add_argument("--transition-only", action="store_true")
    parser.add_argument("--boundary-context-frames", type=int, default=0)
    arguments = parser.parse_args(argv)
    result = generate(
        terrain_usd=arguments.terrain_usd,
        terrain_position=tuple(arguments.terrain_position),
        terrain_quaternion_wxyz=tuple(arguments.terrain_quaternion_wxyz),
        start_xy=tuple(arguments.start_xy),
        end_xy=tuple(arguments.end_xy),
        output_dir=arguments.output_dir,
        archive_path=arguments.archive,
        route_catalog=arguments.route_catalog,
        model_path=arguments.model,
        excluded_source_clip_indices=tuple(arguments.exclude_source_clip_index),
        candidate_limit=arguments.candidate_limit,
        accepted_candidates_per_transition=(
            arguments.accepted_candidates_per_transition
        ),
        expanded_transition_indices=tuple(arguments.expand_transition_index),
        transition_only=arguments.transition_only,
        boundary_context_frames=arguments.boundary_context_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
