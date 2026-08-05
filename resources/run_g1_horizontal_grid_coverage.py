#!/usr/bin/env python3
"""Evaluate one unchanged rigid-motion search over horizontal path lanes."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_path_motion_grid import (
    HorizontalGridLane,
    build_grid_playlist,
    classify_lane,
    horizontal_grid_lanes,
)
from mm_sonic.torch_path_motion_placement import (
    PathContactSignature,
    PlacementRejected,
    RawContactEvent,
    RawMotionWindow,
    contact_signature_cost,
    extract_raw_motion_windows,
    place_raw_window_on_path,
)
from resources.run_g1_grail_contact_inventory import _sample_height_grid
from resources.run_g1_path_motion_placement import (
    _descriptor_transform,
    _matcher_archive,
    _minimal_dataset,
    _placement_shortlist_rows,
    _queries,
    _retain_event_count_diversity,
    _row_to_window,
    _source_profiles,
    _target_grid,
    _uncovered_intervals,
    _window_to_row,
    _yaw_wxyz,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--target-scene", required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--start-x", type=float, default=-1.2795985755)
    parser.add_argument("--stop-x", type=float, default=1.2590216406)
    parser.add_argument("--minimum-y", type=float, default=-1.0)
    parser.add_argument("--maximum-y", type=float, default=1.0)
    parser.add_argument("--spacing-m", type=float, default=0.20)
    parser.add_argument("--step-width-m", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--coarse-results", type=int, default=2000)
    parser.add_argument("--placement-shortlist", type=int, default=800)
    parser.add_argument("--hold-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _grid_lanes_from_args(args: argparse.Namespace) -> tuple[
    HorizontalGridLane, ...
]:
    return horizontal_grid_lanes(
        start_x=float(args.start_x),
        stop_x=float(args.stop_x),
        minimum_y=float(args.minimum_y),
        maximum_y=float(args.maximum_y),
        spacing_m=float(args.spacing_m),
    )


def _score_windows_for_lanes(
    windows: object,
    lane_queries: object,
    *,
    path_length_m: float,
) -> tuple[tuple[dict[str, object], ...], ...]:
    if (
        not isinstance(windows, (list, tuple))
        or any(not isinstance(item, RawMotionWindow) for item in windows)
        or not isinstance(lane_queries, (list, tuple))
        or not lane_queries
        or any(
            not isinstance(queries, (list, tuple))
            or not queries
            or any(
                not isinstance(query, PathContactSignature)
                for query in queries
            )
            for queries in lane_queries
        )
        or not math.isfinite(float(path_length_m))
        or path_length_m <= 0.0
    ):
        raise ContractError("horizontal grid scoring input is invalid")
    output = []
    for queries in lane_queries:
        rows = []
        by_count: dict[int, tuple[PathContactSignature, ...]] = {}
        for query in queries:
            by_count.setdefault(len(query.foot_order), ())
            by_count[len(query.foot_order)] += (query,)
        for window in windows:
            compatible = by_count.get(len(window.events), ())
            if not compatible:
                continue
            cost = min(
                contact_signature_cost(query, window)
                for query in compatible
            )
            if math.isfinite(cost):
                rows.append(_window_to_row(window, cost))
        output.append(
            _retain_event_count_diversity(
                rows,
                per_count=2,
                target_progress_m=path_length_m,
            )
            if rows
            else ()
        )
    return tuple(output)


def _scan_grid_one(
    arguments: tuple[
        str,
        dict[str, object],
        tuple[tuple[PathContactSignature, ...], ...],
        float,
    ]
) -> tuple[tuple[dict[str, object], ...], ...]:
    root_text, descriptor, lane_queries, path_length = arguments
    try:
        profiles = _source_profiles(Path(root_text), descriptor)
        counts = tuple(
            len(query.foot_order)
            for queries in lane_queries
            for query in queries
        )
        windows = extract_raw_motion_windows(
            source_clip=str(descriptor["logical_name"]),
            root_position_world=profiles["root"],
            root_orientation_world_wxyz=profiles["quaternion"],
            foot_position_world=profiles["feet"],
            support_mask=profiles["support"],
            foot_surface_height_m=profiles["surface"],
            minimum_events=min(counts),
            maximum_events=max(counts),
        )
        return _score_windows_for_lanes(
            windows, lane_queries, path_length_m=path_length
        )
    except (ContractError, KeyError, OSError, ValueError):
        return tuple(() for _ in lane_queries)


def _trim_lane_pool(
    rows: object, *, path_length_m: float, maximum_results: int
) -> tuple[dict[str, object], ...]:
    return _placement_shortlist_rows(
        rows,
        path_length_m=path_length_m,
        maximum_results=maximum_results,
    )


def _target_alignment(
    root: Path,
    target: dict[str, object],
) -> tuple[np.ndarray, float]:
    rotation, translation, terrain_yaw = _descriptor_transform(target)
    with np.load(
        root / target["relative_motion_path"], allow_pickle=False
    ) as archive:
        target_root = np.asarray(
            archive["body_pos_w"][0, 0], dtype=np.float64
        )
        target_quaternion = np.asarray(
            archive["body_quat_w"][0, 0], dtype=np.float64
        )
    return (
        target_root[:2] @ rotation.T + translation,
        _yaw_wxyz(target_quaternion) + terrain_yaw,
    )


def _certify_lane(
    *,
    lane: HorizontalGridLane,
    rows: object,
    root: Path,
    by_name: dict[str, dict[str, object]],
    sole_kinematics: MujocoG1SoleKinematics,
    sample_surface,
    path_length_m: float,
    placement_shortlist: int,
    target_root_scene_xy: np.ndarray,
    alignment_yaw: float,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray] | None,
    dict[str, object],
]:
    shortlist = _placement_shortlist_rows(
        rows,
        path_length_m=path_length_m,
        maximum_results=placement_shortlist,
    )
    heading = np.array((1.0, 0.0), dtype=np.float64)
    path_start = np.asarray(lane.start_scene_xy, dtype=np.float64)
    profiles_cache: dict[str, dict[str, np.ndarray]] = {}
    soles_cache: dict[str, np.ndarray] = {}
    accepted = []
    rejected: Counter[str] = Counter()
    for row in shortlist:
        source = str(row["source_clip"])
        descriptor = by_name[source]
        try:
            profiles = profiles_cache.get(source)
            if profiles is None:
                profiles = _source_profiles(root, descriptor)
                profiles_cache[source] = profiles
            soles = soles_cache.get(source)
            if soles is None:
                soles = sole_kinematics.sole_points(
                    profiles["joint"],
                    profiles["root"],
                    profiles["quaternion"],
                )
                soles_cache[source] = soles
            window = _row_to_window(dict(row))
            start, stop = window.start_frame, window.stop_frame
            shifted = RawMotionWindow(
                source_clip=window.source_clip,
                start_frame=0,
                stop_frame=stop - start,
                events=tuple(
                    RawContactEvent(
                        event.frame - start,
                        event.foot,
                        event.position_world_xy,
                        event.surface_height_m,
                    )
                    for event in window.events
                ),
                root_start_world_xy=window.root_start_world_xy,
                forward_progress_m=window.forward_progress_m,
                heading_error_rad=window.heading_error_rad,
            )
            placed = place_raw_window_on_path(
                window=shifted,
                joint_position=profiles["joint"][start:stop],
                root_position_world=profiles["root"][start:stop],
                root_orientation_world_wxyz=profiles["quaternion"][start:stop],
                foot_position_world=profiles["feet"][start:stop],
                sole_position_world=soles[start:stop],
                support_mask=profiles["support"][start:stop],
                path_start_scene_xy=path_start,
                path_heading_scene_xy=heading,
                sample_surface=sample_surface,
            )
            if not np.array_equal(
                placed.joint_position, profiles["joint"][start:stop]
            ):
                raise ContractError("grid placement edited source joints")
            uncovered = max(
                0.0, path_length_m - placed.metrics.covered_stop_m
            )
            accepted.append(
                (
                    uncovered,
                    float(row["cost"]),
                    source,
                    start,
                    placed,
                    dict(row),
                )
            )
        except PlacementRejected as error:
            rejected[error.reason] += 1
        except (ContractError, KeyError, OSError, ValueError):
            rejected["invalid-source"] += 1
    lane_query = {
        "lane_id": lane.lane_id,
        "lane_index": lane.index,
        "center_y_m": lane.center_y_m,
        "path_start_scene_xy": list(lane.start_scene_xy),
        "path_stop_scene_xy": list(lane.stop_scene_xy),
        "path_length_m": path_length_m,
    }
    if not accepted:
        coverage = {
            "schema": "g1-path-motion-coverage/v1",
            "path_length_m": path_length_m,
            "covered_intervals_m": [],
            "uncovered_intervals_m": [[0.0, path_length_m]],
            "fully_covered": False,
        }
        evidence = {
            "schema": "g1-path-motion-placement/v1",
            "query": lane_query,
            "selected": None,
            "certified_count": 0,
            "rejected_by_reason": dict(sorted(rejected.items())),
        }
        summary = {
            **lane_query,
            "classification": "infeasible",
            "covered_length_m": 0.0,
            "selected": None,
            "rejected_by_reason": dict(sorted(rejected.items())),
        }
        return summary, None, {"placements": evidence, "coverage": coverage}
    accepted.sort(key=lambda item: item[:4])
    _, _, _, source_start, best, best_row = accepted[0]
    covered = (
        (
            best.metrics.covered_start_m,
            min(path_length_m, best.metrics.covered_stop_m),
        ),
    )
    gaps = _uncovered_intervals(
        path_length_m=path_length_m,
        covered_intervals=covered,
    )
    classification = classify_lane(
        path_length_m=path_length_m,
        covered_intervals=covered,
        has_certified_placement=True,
    )
    connector = _matcher_archive(
        best,
        target_root_scene_xy=target_root_scene_xy,
        alignment_yaw=alignment_yaw,
    )
    selected = {
        **best_row,
        "source_start_frame": source_start,
        "yaw_scene_rad": best.yaw_scene_rad,
        "translation_scene_xyz": list(best.translation_scene_xyz),
        "metrics": asdict(best.metrics),
        "joint_positions_edited": False,
    }
    evidence = {
        "schema": "g1-path-motion-placement/v1",
        "query": lane_query,
        "selected": selected,
        "certified_count": len(accepted),
        "rejected_by_reason": dict(sorted(rejected.items())),
    }
    coverage = {
        "schema": "g1-path-motion-coverage/v1",
        "path_length_m": path_length_m,
        "covered_intervals_m": [list(item) for item in covered],
        "uncovered_intervals_m": [list(item) for item in gaps],
        "fully_covered": not gaps,
    }
    covered_length = sum(stop - start for start, stop in covered)
    summary = {
        **lane_query,
        "classification": classification,
        "covered_length_m": covered_length,
        "selected": selected,
        "rejected_by_reason": dict(sorted(rejected.items())),
    }
    return summary, connector, {"placements": evidence, "coverage": coverage}


def _write_lane_artifacts(
    output: Path,
    lane: HorizontalGridLane,
    connector: dict[str, np.ndarray] | None,
    reports: dict[str, object],
) -> None:
    lane_root = output / "lanes" / lane.lane_id
    lane_root.mkdir(parents=True, exist_ok=True)
    for name in ("placements", "coverage"):
        (lane_root / f"{name}.json").write_text(
            json.dumps(reports[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if connector is not None:
        np.savez_compressed(lane_root / "traversal.npz", **connector)


def _path_polyline_matcher(
    *,
    lane: HorizontalGridLane,
    sample_surface,
    target_root_scene_xy: np.ndarray,
    alignment_yaw: float,
    sample_count: int = 33,
) -> np.ndarray:
    if (
        not callable(sample_surface)
        or np.asarray(target_root_scene_xy).shape != (2,)
        or not np.isfinite(target_root_scene_xy).all()
        or not math.isfinite(float(alignment_yaw))
        or type(sample_count) is not int
        or sample_count < 2
    ):
        raise ContractError("grid path polyline input is invalid")
    scene_xy = np.column_stack(
        (
            np.linspace(
                lane.start_scene_xy[0],
                lane.stop_scene_xy[0],
                sample_count,
            ),
            np.full(sample_count, lane.center_y_m),
        )
    )
    height = np.asarray(sample_surface(scene_xy), dtype=np.float64)
    if height.shape != (sample_count,) or not np.isfinite(height).all():
        raise ContractError("grid path polyline terrain is invalid")
    relative = scene_xy - np.asarray(
        target_root_scene_xy, dtype=np.float64
    )
    cosine, sine = math.cos(alignment_yaw), math.sin(alignment_yaw)
    matcher_xy = np.stack(
        (
            cosine * relative[:, 0] + sine * relative[:, 1],
            -sine * relative[:, 0] + cosine * relative[:, 1],
        ),
        axis=-1,
    )
    return np.column_stack((matcher_xy, height + 0.03))


def main() -> int:
    args = _parser().parse_args()
    lanes = _grid_lanes_from_args(args)
    if (
        args.workers < 1
        or args.coarse_results < 2
        or args.placement_shortlist < 2
        or args.hold_frames < 0
        or not math.isfinite(float(args.step_width_m))
        or args.step_width_m <= 0.0
    ):
        raise ContractError("horizontal grid options are invalid")
    path_length = float(args.stop_x - args.start_x)
    root = args.source_dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    targets = [
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("logical_name") == args.target_scene
        or descriptor.get("relative_motion_path") == args.target_scene
    ]
    if len(targets) != 1 or targets[0].get("kind") != "terrain":
        raise ContractError("grid target scene must resolve exactly once")
    target = targets[0]
    origin, cell, height = _target_grid(root, target)

    def sample_numpy(points):
        return _sample_height_grid(
            origin_xy=origin,
            cell_size_m=cell,
            height_z=height,
            points_xy=np.asarray(points, dtype=np.float64),
        )

    def sample_torch(points):
        values = sample_numpy(points.detach().cpu().numpy())
        return torch.tensor(
            values, dtype=points.dtype, device=points.device
        )

    lane_queries = tuple(
        _queries(
            path_start=np.asarray(lane.start_scene_xy, dtype=np.float64),
            path_heading=np.array((1.0, 0.0), dtype=np.float64),
            path_length=path_length,
            step_width_m=float(args.step_width_m),
            sample_surface=sample_torch,
        )
        for lane in lanes
    )
    descriptors = [
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("kind") == "terrain"
        and descriptor.get("terrain", {}).get("kind") == "heightgrid"
    ]
    descriptors.sort(key=lambda item: item["logical_name"])
    jobs = (
        (str(root), descriptor, lane_queries, path_length)
        for descriptor in descriptors
    )
    pools: list[list[dict[str, object]]] = [[] for _ in lanes]
    returned_rows = [0 for _ in lanes]
    pool_limit = max(
        int(args.coarse_results), 4 * int(args.placement_shortlist)
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for clip_index, lane_rows in enumerate(
            executor.map(_scan_grid_one, jobs, chunksize=8), start=1
        ):
            for lane_index, rows in enumerate(lane_rows):
                pools[lane_index].extend(rows)
                returned_rows[lane_index] += len(rows)
            if clip_index % 256 == 0:
                for lane_index in range(len(lanes)):
                    if len(pools[lane_index]) > pool_limit:
                        pools[lane_index] = list(
                            _trim_lane_pool(
                                pools[lane_index],
                                path_length_m=path_length,
                                maximum_results=pool_limit,
                            )
                        )
    for lane_index in range(len(lanes)):
        if len(pools[lane_index]) > pool_limit:
            pools[lane_index] = list(
                _trim_lane_pool(
                    pools[lane_index],
                    path_length_m=path_length,
                    maximum_results=pool_limit,
                )
            )
    by_name = {
        descriptor["logical_name"]: descriptor for descriptor in descriptors
    }
    target_root_scene_xy, alignment_yaw = _target_alignment(root, target)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    args.output.mkdir(parents=True, exist_ok=True)
    lane_summaries = []
    playlist_inputs = []
    for lane_index, lane in enumerate(lanes):
        summary, connector, reports = _certify_lane(
            lane=lane,
            rows=pools[lane_index],
            root=root,
            by_name=by_name,
            sole_kinematics=sole_kinematics,
            sample_surface=sample_numpy,
            path_length_m=path_length,
            placement_shortlist=int(args.placement_shortlist),
            target_root_scene_xy=target_root_scene_xy,
            alignment_yaw=alignment_yaw,
        )
        summary["coarse_rows_returned"] = returned_rows[lane_index]
        summary["coarse_pool_retained"] = len(pools[lane_index])
        summary["path_polyline_matcher_xyz"] = _path_polyline_matcher(
            lane=lane,
            sample_surface=sample_numpy,
            target_root_scene_xy=target_root_scene_xy,
            alignment_yaw=alignment_yaw,
        ).tolist()
        lane_summaries.append(summary)
        _write_lane_artifacts(
            args.output, lane, connector, reports
        )
        if connector is not None:
            playlist_inputs.append(
                (lane, str(summary["classification"]), connector)
            )
        print(
            json.dumps(
                {
                    "lane": lane.lane_id,
                    "classification": summary["classification"],
                    "covered_length_m": summary["covered_length_m"],
                    "selected": (
                        None
                        if summary["selected"] is None
                        else summary["selected"]["source_clip"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    counts = Counter(
        str(item["classification"]) for item in lane_summaries
    )
    requested_length = path_length * len(lanes)
    covered_length = sum(
        float(item["covered_length_m"]) for item in lane_summaries
    )
    grid_summary = {
        "schema": "g1-horizontal-grid-coverage/v1",
        "target_scene": target["logical_name"],
        "grid": {
            "start_x": float(args.start_x),
            "stop_x": float(args.stop_x),
            "minimum_y": float(args.minimum_y),
            "maximum_y": float(args.maximum_y),
            "spacing_m": float(args.spacing_m),
            "lane_count": len(lanes),
            "step_width_m": float(args.step_width_m),
        },
        "corpus": {
            "scanned_clip_count": len(descriptors),
            "source_identity_override": False,
        },
        "classification_counts": {
            name: int(counts.get(name, 0))
            for name in ("full", "partial", "infeasible")
        },
        "requested_length_m": requested_length,
        "covered_length_m": covered_length,
        "covered_fraction": covered_length / requested_length,
        "lanes": lane_summaries,
    }
    (args.output / "grid-summary.json").write_text(
        json.dumps(grid_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if playlist_inputs:
        arrays, metadata = build_grid_playlist(
            tuple(playlist_inputs), hold_frames=int(args.hold_frames)
        )
        np.savez_compressed(args.output / "grid-playlist.npz", **arrays)
        (args.output / "grid-playlist.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _minimal_dataset(
        source_root=root,
        manifest=manifest,
        target=target,
        output=args.output / "dataset",
    )
    print(
        json.dumps(
            {
                "classification_counts": grid_summary[
                    "classification_counts"
                ],
                "covered_fraction": grid_summary["covered_fraction"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
