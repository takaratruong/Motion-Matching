"""Dispatch a globally known terrain route to the matching kinematic backend.

The target API is scene-agnostic: it receives an exact terrain USD, its world
transform, and a commanded world-space route.  Target motion identity is never
part of the request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_c490_curb_assets import DEFAULT_ROUTE_CATALOG as CURB_ROUTE_CATALOG
from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as C490_ARCHIVE
from .generate_generic_curb_route import generate as generate_curb
from .generate_generic_slope_route import generate as generate_slope
from .generate_generic_stair_route import generate as generate_stepped
from .generate_generic_stair_route import load_target_mesh
from .terrain_oracle.terrain_route_segments import segment_terrain_route_profile
from .terrain_oracle.terrain_route_profile import sample_terrain_route_profile


CURB_BOUNDARY_CONTEXT_FRAMES = 40


def _profile_dict(profile: object) -> dict[str, object]:
    height = np.asarray(profile.height_m, dtype=np.float64)
    jump = np.diff(height)
    jump = jump[np.abs(jump) > 0.03]
    signs = np.sign(jump)
    reversals = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) else 0
    return {
        "kind": str(profile.kind),
        "route_length_m": float(profile.distance_m[-1]),
        "height_range_m": float(profile.height_range_m),
        "endpoint_height_delta_m": float(profile.endpoint_height_delta_m),
        "jump_count": int(profile.jump_count),
        "jump_direction_reversal_count": reversals,
        "continuous_slope_fraction": float(profile.continuous_slope_fraction),
        "maximum_continuous_slope_rad": float(
            profile.maximum_continuous_slope_rad
        ),
    }


def generate(
    *,
    terrain_usd: Path,
    terrain_position: tuple[float, float, float],
    terrain_quaternion_wxyz: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    output_dir: Path,
) -> dict[str, object]:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mesh = load_target_mesh(
        terrain_usd,
        position_world=terrain_position,
        quaternion_world_from_usd_wxyz=terrain_quaternion_wxyz,
    )
    profile = sample_terrain_route_profile(mesh, start_xy, end_xy)
    profile_summary = _profile_dict(profile)
    kind = str(profile.kind)

    if kind == "flat":
        result: dict[str, object] = {
            "status": "accepted",
            "backend": "live_motionbricks_flat",
            "motion": None,
            "note": "flat route remains under the live MotionBricks controller",
        }
    elif kind == "slope":
        result = generate_slope(
            terrain_usd=terrain_usd,
            terrain_position=terrain_position,
            terrain_quaternion_wxyz=terrain_quaternion_wxyz,
            start_xy=start_xy,
            end_xy=end_xy,
            output_dir=destination / "slope",
        )
        result = {**result, "backend": "continuous_slope_warp"}
    elif kind in {"curb", "stairs"}:
        # Multiple alternating jumps describe independent curb/step obstacles,
        # whereas a monotonic support sequence is a staircase.
        obstacle_course = (
            kind == "curb"
            or int(profile_summary["jump_direction_reversal_count"]) > 0
        )
        if obstacle_course:
            result = generate_curb(
                terrain_usd=terrain_usd,
                terrain_position=terrain_position,
                terrain_quaternion_wxyz=terrain_quaternion_wxyz,
                start_xy=start_xy,
                end_xy=end_xy,
                output_dir=destination / "stepped",
                archive_path=C490_ARCHIVE,
                route_catalog=CURB_ROUTE_CATALOG,
                boundary_context_frames=CURB_BOUNDARY_CONTEXT_FRAMES,
            )
        else:
            result = generate_stepped(
                terrain_usd=terrain_usd,
                terrain_position=terrain_position,
                terrain_quaternion_wxyz=terrain_quaternion_wxyz,
                start_xy=start_xy,
                end_xy=end_xy,
                output_dir=destination / "stepped",
            )
        result = {
            **result,
            "backend": (
                "curb_step_course" if obstacle_course else "staircase"
            ),
        }
    else:
        event_results: list[dict[str, object]] = []
        for index, event in enumerate(segment_terrain_route_profile(profile)):
            event_output = destination / f"event_{index:02d}_{event.kind}"
            if event.kind == "slope":
                event_result = generate_slope(
                    terrain_usd=terrain_usd,
                    terrain_position=terrain_position,
                    terrain_quaternion_wxyz=terrain_quaternion_wxyz,
                    start_xy=tuple(float(v) for v in event.start_xy),
                    end_xy=tuple(float(v) for v in event.end_xy),
                    output_dir=event_output,
                )
                event_backend = "continuous_slope_warp"
            elif event.kind == "stepped":
                event_profile = sample_terrain_route_profile(
                    mesh, event.start_xy, event.end_xy
                )
                event_summary = _profile_dict(event_profile)
                obstacle_course = (
                    event_profile.kind == "curb"
                    or int(event_summary["jump_direction_reversal_count"]) > 0
                )
                if obstacle_course:
                    event_result = generate_curb(
                        terrain_usd=terrain_usd,
                        terrain_position=terrain_position,
                        terrain_quaternion_wxyz=terrain_quaternion_wxyz,
                        start_xy=tuple(float(v) for v in event.start_xy),
                        end_xy=tuple(float(v) for v in event.end_xy),
                        output_dir=event_output,
                        archive_path=C490_ARCHIVE,
                        route_catalog=CURB_ROUTE_CATALOG,
                        boundary_context_frames=CURB_BOUNDARY_CONTEXT_FRAMES,
                    )
                else:
                    event_result = generate_stepped(
                        terrain_usd=terrain_usd,
                        terrain_position=terrain_position,
                        terrain_quaternion_wxyz=terrain_quaternion_wxyz,
                        start_xy=tuple(float(v) for v in event.start_xy),
                        end_xy=tuple(float(v) for v in event.end_xy),
                        output_dir=event_output,
                    )
                event_backend = (
                    "curb_step_course" if obstacle_course else "staircase"
                )
            else:
                event_result = {
                    "status": "rejected",
                    "reason": "overlapping slope and stepped event",
                    "motion": None,
                }
                event_backend = "mixed_event"
            event_results.append(
                {
                    "event_index": index,
                    "kind": event.kind,
                    "backend": event_backend,
                    "start_distance_m": float(event.start_distance_m),
                    "end_distance_m": float(event.end_distance_m),
                    "start_xy": [float(v) for v in event.start_xy],
                    "end_xy": [float(v) for v in event.end_xy],
                    "result": event_result,
                }
            )
        accepted = bool(event_results) and all(
            event["result"]["status"] == "accepted" for event in event_results
        )
        result = {
            "status": "accepted" if accepted else "rejected",
            "backend": "live_motionbricks_with_terrain_portals",
            "motion": None,
            "events": event_results,
        }

    summary = {
        "schema": "generic-terrain-route/v1",
        "terrain_usd": str(terrain_usd.expanduser().resolve()),
        "terrain_position_world": list(terrain_position),
        "terrain_quaternion_world_from_usd_wxyz": list(
            terrain_quaternion_wxyz
        ),
        "start_xy": list(start_xy),
        "end_xy": list(end_xy),
        "profile": profile_summary,
        "result": result,
        "status": result["status"],
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
    arguments = parser.parse_args(argv)
    summary = generate(
        terrain_usd=arguments.terrain_usd,
        terrain_position=tuple(arguments.terrain_position),
        terrain_quaternion_wxyz=tuple(arguments.terrain_quaternion_wxyz),
        start_xy=tuple(arguments.start_xy),
        end_xy=tuple(arguments.end_xy),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
