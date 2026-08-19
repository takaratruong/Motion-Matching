"""Cover a target staircase with longest safe source-contiguous sub-traversals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .stair_mesh_profile import measure_archive_stair_profile
from .terrain_oracle.stair_geometry_warp import (
    DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    _archive_terrain_index,
    motion_conditioned_stair_support_route,
    warp_archive_clip_to_stair_geometry,
)
from .terrain_oracle.stair_support_route import (
    StairSupportRoute,
    sample_stair_support_route,
)
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_FRAGMENT_BANK = Path(
    "/move/data/terrain-aware/motion-matching/"
    "stairs500-fragments-all500-exact-v1/fragments.jsonl"
)


@dataclass(frozen=True)
class _SourceBlock:
    archive_clip_index: int
    traversal: str
    source_start_frame: int
    source_stop_frame: int
    transition_count: int
    transition_run_m: np.ndarray
    transition_rise_m: np.ndarray


def _load_source_blocks(
    fragment_bank: Path,
    route_catalog: Path,
    *,
    maximum_transitions: int,
) -> tuple[_SourceBlock, ...]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    with fragment_bank.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("start_tread_index") is None
                or row.get("end_tread_index") is None
            ):
                continue
            groups[int(row["archive_clip_index"])].append(row)

    with np.load(route_catalog, allow_pickle=False) as data:
        counts = np.asarray(data["level_count"], dtype=np.int16)
        widths = np.asarray(data["level_width_m"], dtype=np.float64)
        heights = np.asarray(data["level_height_m"], dtype=np.float64)

    blocks: list[_SourceBlock] = []
    seen: set[tuple[int, int, int, int]] = set()
    for clip_index, records in groups.items():
        count = int(counts[clip_index])
        if count < 2:
            continue
        records.sort(key=lambda row: tuple(int(v) for v in row["context_range"]))
        tread_values = [
            int(row[key])
            for row in records
            for key in ("start_tread_index", "end_tread_index")
        ]
        minimum_tread = min(tread_values)
        maximum_tread = max(tread_values)
        traversal = str(records[0]["source_direction"])

        def route_index(tread: int) -> int:
            return (
                tread - minimum_tread
                if traversal == "up"
                else maximum_tread - tread
            )

        level_widths = widths[clip_index, :count]
        level_heights = heights[clip_index, :count]
        level_centres = np.cumsum(level_widths) - 0.5 * level_widths
        runs = np.diff(level_centres)
        rises = np.diff(level_heights)
        for first in range(len(records)):
            for last in range(first, len(records)):
                start_level = route_index(
                    int(records[first]["start_tread_index"])
                )
                stop_level = route_index(
                    int(records[last]["end_tread_index"])
                )
                transitions = stop_level - start_level
                if (
                    transitions < 1
                    or transitions > int(maximum_transitions)
                    or start_level < 0
                    or stop_level >= count
                ):
                    continue
                start_frame = int(records[first]["context_range"][0])
                stop_frame = int(records[last]["context_range"][1])
                key = (clip_index, start_frame, stop_frame, transitions)
                if key in seen or stop_frame - start_frame < 3:
                    continue
                seen.add(key)
                blocks.append(
                    _SourceBlock(
                        archive_clip_index=clip_index,
                        traversal=traversal,
                        source_start_frame=start_frame,
                        source_stop_frame=stop_frame,
                        transition_count=transitions,
                        transition_run_m=runs[start_level:stop_level].copy(),
                        transition_rise_m=rises[start_level:stop_level].copy(),
                    )
                )
    return tuple(blocks)


def _reason_group(message: str) -> str:
    for label in (
        "same support-level count",
        "excessive root clearance lift",
        "exceeds joint-correction bound",
        "misses foot target",
        "exceeds vertical sole-penetration bound",
        "intersects a target mesh edge or riser",
        "unsupported stance foot",
    ):
        if label in message:
            return label
    return message.split(":", 1)[0]


def global_scene_stair_support_route(
    archive: object,
    target_clip_index: int,
    *,
    approach_yaw_offset_rad: float = 0.0,
    lateral_offset_m: float = 0.0,
) -> StairSupportRoute:
    """Ray-cast the complete commanded stair line, independent of its clip.

    Motion-conditioned routes answer what one recording happened to traverse.
    The privileged ceiling instead knows the full mesh and asks what support
    sequence a forward joystick command encounters.  Ascents start on the
    ground before the first riser and finish on the measured top landing;
    descents start on the top tread and include the ground beyond the last
    riser.
    """

    target = int(target_clip_index)
    traversal = str(archive["clip_traversal"][target])
    profile = measure_archive_stair_profile(archive, target)
    origin = np.asarray(
        archive["terrain_position_env"][target], dtype=np.float64
    )
    yaw = float(archive["travel_yaw_rad"][target])
    authored_direction = np.asarray(
        (math.cos(yaw), math.sin(yaw)), dtype=np.float64
    )
    if traversal == "up":
        start_distance = -0.35
        stop_distance = max(0.05, float(profile.run_m) - 0.05)
    elif traversal == "down":
        start_distance = 0.05
        stop_distance = float(profile.run_m) + 0.35
    else:
        raise ValueError("global stair route requires up/down traversal")
    authored_start = origin[:2] + start_distance * authored_direction
    authored_stop = origin[:2] + stop_distance * authored_direction
    pivot = 0.5 * (authored_start + authored_stop)
    route_length = float(np.linalg.norm(authored_stop - authored_start))
    commanded_yaw = yaw + float(approach_yaw_offset_rad)
    direction = np.asarray(
        (math.cos(commanded_yaw), math.sin(commanded_yaw)), dtype=np.float64
    )
    lateral = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    route_start = (
        pivot
        - 0.5 * route_length * direction
        + float(lateral_offset_m) * lateral
    )
    route_stop = (
        pivot
        + 0.5 * route_length * direction
        + float(lateral_offset_m) * lateral
    )
    return sample_stair_support_route(
        _archive_terrain_index(archive, target),
        route_start,
        route_stop,
        float(origin[2]),
        0.10,
        0.055,
        sample_spacing_m=0.01,
    )


def _route_payload(route: StairSupportRoute) -> dict[str, object]:
    return {
        "start_xy": [float(value) for value in route.start_xy],
        "end_xy": [float(value) for value in route.end_xy],
        "levels": [
            {
                "height_m": float(level.height_m),
                "route_start_distance_m": float(
                    level.route_start_distance_m
                ),
                "route_stop_distance_m": float(
                    level.route_stop_distance_m
                ),
                "route_start_xy": [
                    float(value) for value in level.route_start_xy
                ],
                "route_stop_xy": [
                    float(value) for value in level.route_stop_xy
                ],
            }
            for level in route.levels
        ],
    }


def evaluate(
    *,
    archive_path: Path,
    model_path: Path,
    fragment_bank: Path,
    route_catalog: Path,
    target_clip_index: int | None,
    maximum_block_transitions: int,
    candidate_limit_per_span: int,
    output_dir: Path,
    target_route: StairSupportRoute | None = None,
    target_mesh: TerrainMeshIndex | None = None,
    target_name: str | None = None,
    traversal: str | None = None,
    excluded_source_clip_indices: tuple[int, ...] = (),
    exclude_target_source: bool = True,
    forced_transition_counts: tuple[int, ...] = (),
) -> dict[str, object]:
    import zarr

    archive = zarr.open_group(str(archive_path), mode="r")
    target = (
        None if target_clip_index is None else int(target_clip_index)
    )
    if target_route is None:
        if target is None:
            raise ValueError(
                "generic terrain coverage requires an explicit target route"
            )
        route = motion_conditioned_stair_support_route(archive, target)
    else:
        route = target_route
    if target_mesh is None:
        if target is None:
            raise ValueError(
                "generic terrain coverage requires an explicit target mesh"
            )
        target_mesh = _archive_terrain_index(archive, target)
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    if traversal is None:
        if target is not None:
            traversal = str(archive["clip_traversal"][target])
        else:
            height_delta = float(
                route.levels[-1].height_m - route.levels[0].height_m
            )
            if abs(height_delta) <= 0.04:
                raise ValueError(
                    "generic stair route must have a monotonic height change"
                )
            traversal = "up" if height_delta > 0.0 else "down"
    traversal = str(traversal)
    if traversal not in {"up", "down"}:
        raise ValueError("stair traversal must be up or down")
    level_count = len(route.levels)
    transition_count = level_count - 1
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
    excluded_sources = {
        int(value) for value in excluded_source_clip_indices
    }
    if exclude_target_source and target is not None:
        excluded_sources.add(target)
    blocks = tuple(
        block
        for block in _load_source_blocks(
            fragment_bank,
            route_catalog,
            maximum_transitions=maximum_block_transitions,
        )
        if block.traversal == traversal
        and block.archive_clip_index not in excluded_sources
    )

    attempts: list[dict[str, object]] = []
    accepted_by_span: dict[tuple[int, int], tuple[_SourceBlock, object]] = {}

    def accepted_span(start_level: int, transitions: int):
        key = (start_level, transitions)
        if key in accepted_by_span:
            return accepted_by_span[key]
        stop_level = start_level + transitions
        wanted_run = target_runs[start_level:stop_level]
        wanted_rise = target_rises[start_level:stop_level]
        candidates = [
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
            for block in blocks
            if block.transition_count == transitions
        ]
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].archive_clip_index,
                item[1].source_start_frame,
            )
        )
        route_start = np.asarray(route.start_xy, dtype=np.float64)
        target_start_xy = route_start + centres[start_level] * direction
        target_stop_xy = route_start + centres[stop_level] * direction
        for geometry_cost, block in candidates[: int(candidate_limit_per_span)]:
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
                    maximum_sole_penetration_m=0.006,
                    maximum_root_clearance_lift_m=0.025,
                    maximum_foothold_progress_shift_m=(
                        0.08 if target_route is not None else 0.0
                    ),
                    foothold_edge_clearance_margin_m=(
                        0.02 if target_route is not None else 0.0
                    ),
                    maximum_foothold_yaw_adjustment_rad=(
                        math.radians(50.0)
                        if target_route is not None
                        else 0.0
                    ),
                    foothold_yaw_search_step_rad=math.radians(10.0),
                    minimum_stance_support_points=1,
                )
            except ValueError as error:
                attempt = {
                    "target_start_level": start_level,
                    "transition_count": transitions,
                    "source_clip_index": block.archive_clip_index,
                    "source_range": [
                        block.source_start_frame,
                        block.source_stop_frame,
                    ],
                    "accepted": False,
                    "geometry_cost": geometry_cost,
                    "reason_group": _reason_group(str(error)),
                    "reason": str(error),
                }
                attempts.append(attempt)
                print(
                    json.dumps(
                        {
                            "event": "coverage_candidate",
                            "target_start_level": start_level,
                            "transition_count": transitions,
                            "source_clip_index": block.archive_clip_index,
                            "accepted": False,
                            "reason_group": attempt["reason_group"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            attempts.append(
                {
                    "target_start_level": start_level,
                    "transition_count": transitions,
                    "source_clip_index": block.archive_clip_index,
                    "source_range": [
                        block.source_start_frame,
                        block.source_stop_frame,
                    ],
                    "accepted": True,
                    "geometry_cost": geometry_cost,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "coverage_candidate",
                        "target_start_level": start_level,
                        "transition_count": transitions,
                        "source_clip_index": block.archive_clip_index,
                        "accepted": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            accepted_by_span[key] = (block, warped)
            return accepted_by_span[key]
        return None

    memo: dict[int, list[tuple[int, int, _SourceBlock, object]] | None] = {}

    def solve(level: int):
        if level == transition_count:
            return []
        if level in memo:
            return memo[level]
        maximum = min(
            int(maximum_block_transitions), transition_count - level
        )
        for transitions in range(maximum, 0, -1):
            accepted = accepted_span(level, transitions)
            if accepted is None:
                continue
            tail = solve(level + transitions)
            if tail is not None:
                block, warped = accepted
                memo[level] = [
                    (level, transitions, block, warped), *tail
                ]
                return memo[level]
        memo[level] = None
        return None

    if forced_transition_counts:
        if (
            any(int(value) < 1 for value in forced_transition_counts)
            or sum(int(value) for value in forced_transition_counts)
            != transition_count
        ):
            raise ValueError(
                "forced transition plan must be positive and cover the route"
            )
        forced_plan: list[tuple[int, int, _SourceBlock, object]] = []
        level = 0
        for transitions_value in forced_transition_counts:
            transitions = int(transitions_value)
            accepted = accepted_span(level, transitions)
            if accepted is None:
                forced_plan = []
                break
            block, warped = accepted
            forced_plan.append((level, transitions, block, warped))
            level += transitions
        plan = forced_plan if len(forced_plan) == len(forced_transition_counts) else None
    else:
        plan = solve(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, object]] = []
    if plan is not None:
        for index, (level, transitions, block, warped) in enumerate(plan):
            motion_path = output_dir / f"block_{index:02d}_motion.npz"
            np.savez_compressed(
                motion_path,
                fps=np.asarray(50.0, dtype=np.float32),
                root_position_world=warped.motion.root_position_world,
                root_quaternion_world_wxyz=(
                    warped.motion.root_quaternion_world_wxyz
                ),
                joint_position=warped.motion.joint_position,
                seam_indices=np.asarray(
                    warped.motion.seam_indices, dtype=np.int64
                ),
                source_archive_clip_index=np.asarray(
                    [
                        value.archive_clip_index
                        for value in warped.motion.provenance
                    ],
                    dtype=np.int64,
                ),
                source_frame=np.asarray(
                    [value.source_frame for value in warped.motion.provenance],
                    dtype=np.int64,
                ),
                source_clip_id=np.asarray(
                    [str(value.clip_id) for value in warped.motion.provenance],
                    dtype=np.str_,
                ),
            )
            plan_rows.append(
                {
                    "target_start_level": level,
                    "target_stop_level": level + transitions,
                    "transition_count": transitions,
                    "source_clip_index": block.archive_clip_index,
                    "source_range": [
                        block.source_start_frame,
                        block.source_stop_frame,
                    ],
                    "motion_path": str(motion_path.resolve()),
                    "maximum_joint_correction_rad": float(
                        warped.maximum_joint_correction_rad
                    ),
                    "maximum_foot_target_error_m": float(
                        warped.maximum_foot_target_error_m
                    ),
                    "maximum_triangle_sphere_penetration_m": float(
                        warped.maximum_triangle_sphere_penetration_m
                    ),
                    "maximum_root_clearance_lift_m": float(
                        warped.maximum_root_clearance_lift_m
                    ),
                    "maximum_foothold_progress_shift_m": float(
                        warped.maximum_foothold_progress_shift_m
                    ),
                    "maximum_foothold_yaw_adjustment_rad": float(
                        warped.maximum_foothold_yaw_adjustment_rad
                    ),
                }
            )
    report = {
        "schema": "coherent_stair_block_coverage/v1",
        "target_clip_index": target,
        "target_name": (
            str(archive["clip_names"][target])
            if target is not None
            else str(target_name or "arbitrary-terrain")
        ),
        "traversal": traversal,
        "target_level_count": level_count,
        "target_transition_count": transition_count,
        "target_route_kind": (
            "motion_conditioned"
            if target_route is None
            else (
                "global_scene_line"
                if target is not None
                else "arbitrary_mesh_line"
            )
        ),
        "target_route": _route_payload(route),
        "exclude_target_source": bool(exclude_target_source),
        "excluded_source_clip_indices": sorted(excluded_sources),
        "source_block_count": len(blocks),
        "maximum_block_transitions": int(maximum_block_transitions),
        "candidate_limit_per_span": int(candidate_limit_per_span),
        "forced_transition_counts": [
            int(value) for value in forced_transition_counts
        ],
        "supported": plan is not None,
        "plan_block_count": None if plan is None else len(plan),
        "plan_seam_count": None if plan is None else max(0, len(plan) - 1),
        "plan": plan_rows,
        "attempt_count": len(attempts),
        "rejection_counts": dict(
            Counter(
                str(row["reason_group"])
                for row in attempts
                if not bool(row["accepted"])
            )
        ),
        "attempts": attempts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--fragment-bank", type=Path, default=DEFAULT_FRAGMENT_BANK)
    parser.add_argument(
        "--route-catalog",
        type=Path,
        default=DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    )
    parser.add_argument("--target-clip-index", type=int, required=True)
    parser.add_argument("--maximum-block-transitions", type=int, default=4)
    parser.add_argument("--candidate-limit-per-span", type=int, default=12)
    parser.add_argument(
        "--forced-transition-counts",
        type=str,
        default="",
        help="Optional comma-separated block lengths that must cover the route.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-scene-route", action="store_true")
    parser.add_argument("--global-route-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--global-route-lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--include-target-source", action="store_true")
    args = parser.parse_args()
    import zarr

    archive = zarr.open_group(
        str(args.archive.expanduser().resolve()), mode="r"
    )
    target_route = (
        global_scene_stair_support_route(
            archive,
            args.target_clip_index,
            approach_yaw_offset_rad=math.radians(
                float(args.global_route_yaw_offset_deg)
            ),
            lateral_offset_m=float(args.global_route_lateral_offset_m),
        )
        if args.global_scene_route
        else None
    )
    report = evaluate(
        archive_path=args.archive.expanduser().resolve(),
        model_path=args.model.expanduser().resolve(),
        fragment_bank=args.fragment_bank.expanduser().resolve(),
        route_catalog=args.route_catalog.expanduser().resolve(),
        target_clip_index=args.target_clip_index,
        maximum_block_transitions=args.maximum_block_transitions,
        candidate_limit_per_span=args.candidate_limit_per_span,
        output_dir=args.output_dir.expanduser().resolve(),
        target_route=target_route,
        exclude_target_source=not args.include_target_source,
        forced_transition_counts=tuple(
            int(value)
            for value in args.forced_transition_counts.split(",")
            if value.strip()
        ),
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "attempts"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(report["supported"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
