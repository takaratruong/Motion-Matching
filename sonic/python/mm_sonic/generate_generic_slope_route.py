"""Warp a clean C490 ramp traversal onto an arbitrary exact-mesh slope."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .build_c490_slope_window_catalog import (
    DEFAULT_ARCHIVE,
    DEFAULT_OUTPUT as DEFAULT_WINDOW_CATALOG,
)
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_coherent_block_plan import _repair_exact_mesh_clearance
from .compose_privileged_stair_route import _maximum_steps, _save_motion
from .generate_generic_stair_route import load_target_mesh
from .terrain_oracle.continuous_slope import continuous_profile_support_route
from .terrain_oracle.stair_geometry_warp import (
    _archive_terrain_index,
    warp_archive_clip_to_stair_geometry,
)
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.terrain_route_profile import sample_terrain_route_profile


def _match_score(row: dict[str, object], target: object) -> float:
    target_rise = abs(float(target.endpoint_height_delta_m))
    target_run = float(target.distance_m[-1])
    target_slope = float(target.maximum_continuous_slope_rad)
    source_rise = float(row["height_delta_m"])
    source_run = float(row["route_length_m"])
    source_slope = float(row["maximum_continuous_slope_rad"])
    return float(
        abs(math.log((source_rise + 0.02) / (target_rise + 0.02)))
        + 0.45 * abs(math.log((source_run + 0.10) / (target_run + 0.10)))
        + 0.50 * abs(source_slope - target_slope)
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
    window_catalog: Path = DEFAULT_WINDOW_CATALOG,
    model_path: Path = DEFAULT_G1_MJCF,
    excluded_source_clip_indices: tuple[int, ...] = (),
    candidate_limit: int = 12,
    profile_level_count: int = 24,
) -> dict[str, object]:
    import zarr

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target_mesh = load_target_mesh(
        terrain_usd,
        position_world=terrain_position,
        quaternion_world_from_usd_wxyz=terrain_quaternion_wxyz,
    )
    target_profile = sample_terrain_route_profile(
        target_mesh, start_xy, end_xy, sample_spacing_m=0.01
    )
    delta = float(target_profile.endpoint_height_delta_m)
    if target_profile.kind != "slope" or abs(delta) < 0.025:
        raise ValueError(
            "generic slope generation needs a monotonic continuous-ramp route; "
            f"got kind={target_profile.kind}, delta={delta:.4f} m"
        )
    traversal = "up" if delta > 0.0 else "down"
    target_route = continuous_profile_support_route(
        target_profile, level_count=profile_level_count
    )
    archive_source = archive_path.expanduser().resolve()
    archive = zarr.open_group(str(archive_source), mode="r")
    catalog = json.loads(window_catalog.expanduser().resolve().read_text())
    excluded = {int(value) for value in excluded_source_clip_indices}
    candidates = [
        row
        for row in catalog["windows"]
        if row["traversal"] == traversal
        and int(row["clip_index"]) not in excluded
    ]
    candidates.sort(
        key=lambda row: (
            _match_score(row, target_profile),
            int(row["clip_index"]),
        )
    )
    attempts: list[dict[str, object]] = []
    accepted: list[tuple[float, object, object, float, dict[str, object]]] = []
    for rank, row in enumerate(candidates[: int(candidate_limit)]):
        clip_index = int(row["clip_index"])
        source_mesh = _archive_terrain_index(archive, clip_index)
        source_profile = sample_terrain_route_profile(
            source_mesh,
            row["route_start_xy"],
            row["route_end_xy"],
            sample_spacing_m=0.01,
        )
        source_route = continuous_profile_support_route(
            source_profile, level_count=profile_level_count
        )
        attempt: dict[str, object] = {
            "rank": rank,
            "source_clip_index": clip_index,
            "source_clip_name": str(row["clip_name"]),
            "source_frame_range": [
                int(row["start_frame"]),
                int(row["stop_frame"]),
            ],
            "geometry_match_score": _match_score(row, target_profile),
        }
        try:
            warped = warp_archive_clip_to_stair_geometry(
                archive_source,
                source_clip_index=clip_index,
                source_start_frame=int(row["start_frame"]),
                source_stop_frame=int(row["stop_frame"]),
                target_clip_index=None,
                target_mesh=target_mesh,
                target_route_start_xy=start_xy,
                target_route_end_xy=end_xy,
                source_support_route=source_route,
                target_support_route=target_route,
                model_path=model_path,
                maximum_joint_correction_rad=0.45,
                # This is an IK residual prefilter; exact complete-G1 mesh
                # collision and mechanics are checked immediately afterward.
                maximum_foot_target_error_m=0.05,
                maximum_sole_penetration_m=0.003,
                maximum_root_clearance_lift_m=0.025,
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
            repaired, final_audit, clearance_lift = (
                _repair_exact_mesh_clearance(
                    warped.motion,
                    audit,
                    archive_path=archive_source,
                    target_clip_index=None,
                    target_mesh=target_mesh,
                    model_path=model_path,
                )
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
                    "clearance_repair_maximum_m": clearance_lift,
                    "full_body_audit": final_audit.to_dict(),
                    "mechanics": mechanics,
                    "status": (
                        "accepted"
                        if final_audit.accepted and mechanically_safe
                        else "rejected"
                    ),
                }
            )
            if final_audit.accepted and mechanically_safe:
                quality = (
                    float(attempt["geometry_match_score"])
                    + 2.0 * float(warped.maximum_joint_correction_rad)
                    + 10.0 * float(clearance_lift)
                )
                accepted.append(
                    (quality, repaired, final_audit, clearance_lift, attempt)
                )
        except (RuntimeError, ValueError) as error:
            attempt.update({"status": "rejected", "reason": str(error)})
        attempts.append(attempt)

    if not accepted:
        summary = {
            "schema": "generic-slope-route/v1",
            "status": "rejected",
            "terrain_usd": str(terrain_usd.expanduser().resolve()),
            "traversal": traversal,
            "target_profile": {
                "kind": target_profile.kind,
                "route_length_m": float(target_profile.distance_m[-1]),
                "height_delta_m": delta,
                "maximum_continuous_slope_rad": float(
                    target_profile.maximum_continuous_slope_rad
                ),
            },
            "excluded_source_clip_indices": sorted(excluded),
            "attempts": attempts,
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary

    accepted.sort(key=lambda value: value[0])
    quality, motion, audit, clearance_lift, selected = accepted[0]
    _save_motion(destination / "motion.npz", motion)
    summary = {
        "schema": "generic-slope-route/v1",
        "status": "accepted",
        "method": "continuous_profile_coherent_warp",
        "terrain_usd": str(terrain_usd.expanduser().resolve()),
        "target_clip_index": None,
        "traversal": traversal,
        "target_profile": {
            "kind": target_profile.kind,
            "route_length_m": float(target_profile.distance_m[-1]),
            "height_delta_m": delta,
            "maximum_continuous_slope_rad": float(
                target_profile.maximum_continuous_slope_rad
            ),
        },
        "excluded_source_clip_indices": sorted(excluded),
        "selected": selected,
        "quality": float(quality),
        "clearance_repair_maximum_m": float(clearance_lift),
        "full_body_audit": audit.to_dict(),
        "motion": str((destination / "motion.npz").resolve()),
        "attempts": attempts,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-usd", type=Path, required=True)
    parser.add_argument("--terrain-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument(
        "--terrain-quaternion-wxyz", type=float, nargs=4, default=(1, 0, 0, 0)
    )
    parser.add_argument("--start-xy", type=float, nargs=2, required=True)
    parser.add_argument("--end-xy", type=float, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--window-catalog", type=Path, default=DEFAULT_WINDOW_CATALOG)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--exclude-source-clip-index", type=int, action="append", default=[])
    parser.add_argument("--candidate-limit", type=int, default=12)
    arguments = parser.parse_args(argv)
    result = generate(
        terrain_usd=arguments.terrain_usd,
        terrain_position=tuple(arguments.terrain_position),
        terrain_quaternion_wxyz=tuple(arguments.terrain_quaternion_wxyz),
        start_xy=tuple(arguments.start_xy),
        end_xy=tuple(arguments.end_xy),
        output_dir=arguments.output_dir,
        archive_path=arguments.archive,
        window_catalog=arguments.window_catalog,
        model_path=arguments.model,
        excluded_source_clip_indices=tuple(arguments.exclude_source_clip_index),
        candidate_limit=arguments.candidate_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
