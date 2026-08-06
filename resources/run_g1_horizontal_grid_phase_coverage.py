#!/usr/bin/env python3
"""Evaluate mount, interior, and dismount coverage for horizontal lanes."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    nominal_footprint_path,
)
from mm_sonic.torch_path_motion_grid import (
    HorizontalGridLane,
    horizontal_grid_lanes,
)
from mm_sonic.torch_path_motion_placement import (
    PathContactSignature,
    RawMotionWindow,
    extract_raw_motion_windows,
    path_contact_signature,
)
from mm_sonic.torch_path_terrain_phases import (
    TerrainPathPhase,
    segment_path_surface,
)
from resources.run_g1_grail_contact_inventory import _sample_height_grid
from resources.run_g1_horizontal_grid_coverage import (
    _certify_lane,
    _target_alignment,
)
from resources.run_g1_path_motion_placement import (
    _SPEEDS_MPS,
    _STRIDES_M,
    _minimal_dataset,
    _placement_shortlist_rows,
    _retain_event_count_diversity,
    _source_profiles,
    _target_grid,
    _window_to_row,
)


_PHASE_ORDER = {"mount": 0, "interior": 1, "dismount": 2}


def _stable_mount_window(
    window: RawMotionWindow,
    source_support: np.ndarray,
    *,
    maximum_unsupported_run_frames: int = 4,
    minimum_terminal_double_support_frames: int = 8,
) -> bool:
    support = np.asarray(source_support, dtype=np.bool_)
    if (
        not isinstance(window, RawMotionWindow)
        or support.ndim != 2
        or support.shape[1] != 2
        or not 0
        <= window.start_frame
        < window.stop_frame
        <= len(support)
    ):
        raise ContractError("mount window support inputs are invalid")
    selected = support[window.start_frame : window.stop_frame]
    maximum_run = 0
    current_run = 0
    for value in ~selected.any(axis=1):
        current_run = current_run + 1 if value else 0
        maximum_run = max(maximum_run, current_run)
    terminal_double_support = 0
    for value in selected.all(axis=1)[::-1]:
        if not value:
            break
        terminal_double_support += 1
    return (
        maximum_run <= maximum_unsupported_run_frames
        and terminal_double_support >= minimum_terminal_double_support_frames
    )


def _semantic_dismount_window(
    window: RawMotionWindow,
    source_support: np.ndarray,
    source_surface: np.ndarray,
    *,
    minimum_initial_supported_frames: int = 8,
    minimum_drop_height_m: float = 0.12,
    maximum_landing_height_difference_m: float = 0.04,
    maximum_unsupported_run_frames: int = 20,
    minimum_terminal_supported_frames: int = 8,
) -> bool:
    support = np.asarray(source_support, dtype=np.bool_)
    surface = np.asarray(source_surface, dtype=np.float64)
    if (
        not isinstance(window, RawMotionWindow)
        or support.ndim != 2
        or support.shape[1] != 2
        or surface.shape != support.shape
        or not np.isfinite(surface).all()
        or not 0
        <= window.start_frame
        < window.stop_frame
        <= len(support)
        or any(
            event.frame < window.start_frame
            or event.frame >= window.stop_frame
            for event in window.events
        )
    ):
        raise ContractError("dismount window support inputs are invalid")
    selected = support[window.start_frame : window.stop_frame]
    supported = selected.any(axis=1)
    initial_supported = 0
    for value in supported:
        if not value:
            break
        initial_supported += 1
    terminal_supported = 0
    for value in supported[::-1]:
        if not value:
            break
        terminal_supported += 1
    maximum_run = 0
    current_run = 0
    for value in ~supported:
        current_run = current_run + 1 if value else 0
        maximum_run = max(maximum_run, current_run)
    initial_mask = support[window.start_frame]
    if not initial_mask.any():
        return False
    initial_height = float(
        surface[window.start_frame, initial_mask].mean()
    )
    landing_height = min(
        float(event.surface_height_m) for event in window.events
    )
    landing_feet = {
        int(event.foot)
        for event in window.events
        if abs(float(event.surface_height_m) - landing_height)
        <= maximum_landing_height_difference_m
    }
    return (
        initial_supported >= minimum_initial_supported_frames
        and initial_height - landing_height >= minimum_drop_height_m
        and landing_feet == {0, 1}
        and maximum_run <= maximum_unsupported_run_frames
        and terminal_supported >= minimum_terminal_supported_frames
    )


def _windows_for_phase_kind(
    windows: tuple[RawMotionWindow, ...],
    phase_kind: str,
    source_support: np.ndarray,
    *,
    source_surface: np.ndarray | None = None,
) -> tuple[RawMotionWindow, ...]:
    if (
        not isinstance(windows, tuple)
        or any(not isinstance(window, RawMotionWindow) for window in windows)
        or phase_kind not in _PHASE_ORDER
    ):
        raise ContractError("phase window routing inputs are invalid")
    if phase_kind == "interior":
        return windows
    if phase_kind == "dismount":
        if source_surface is None:
            raise ContractError("dismount surface profile is unavailable")
        return tuple(
            window
            for window in windows
            if _semantic_dismount_window(
                window, source_support, source_surface
            )
        )
    return tuple(
        window
        for window in windows
        if _stable_mount_window(window, source_support)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _rounded_signature(
    signature: PathContactSignature,
) -> PathContactSignature:
    return PathContactSignature(
        foot_order=signature.foot_order,
        forward_m=tuple(round(value, 6) for value in signature.forward_m),
        lateral_m=tuple(round(value, 6) for value in signature.lateral_m),
        contact_frame=signature.contact_frame,
        height_pattern_m=tuple(
            round(value, 6) for value in signature.height_pattern_m
        ),
    )


def _phase_queries(
    *,
    phase: TerrainPathPhase,
    step_width_m: float,
    sample_surface,
) -> tuple[PathContactSignature, ...]:
    if (
        not isinstance(phase, TerrainPathPhase)
        or not callable(sample_surface)
        or not math.isfinite(float(step_width_m))
        or step_width_m <= 0.0
    ):
        raise ContractError("phase query inputs are invalid")
    start = np.asarray(phase.start_scene_xy, dtype=np.float64)
    stop = np.asarray(phase.stop_scene_xy, dtype=np.float64)
    displacement = stop - start
    length = float(np.linalg.norm(displacement))
    heading = displacement / length
    lateral = np.array((-heading[1], heading[0]), dtype=np.float64)

    def sample_torch(points: torch.Tensor) -> torch.Tensor:
        try:
            values = np.asarray(
                sample_surface(points.detach().cpu().numpy()),
                dtype=np.float64,
            )
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("phase query terrain sampling failed") from error
        return torch.as_tensor(
            values, dtype=points.dtype, device=points.device
        )

    output = []
    for first_foot in (0, 1):
        offsets = np.zeros((2, 2), dtype=np.float64)
        offsets[:, 1] = (0.5 * step_width_m, -0.5 * step_width_m)
        offsets[first_foot, 0] = -0.01
        start_feet = (
            start[None, :]
            + offsets[:, :1] * heading[None, :]
            + offsets[:, 1:] * lateral[None, :]
        )
        for stride in _STRIDES_M:
            for speed in _SPEEDS_MPS:
                path = nominal_footprint_path(
                    ConstantHeadingRequest(
                        start_foot_scene_xy=torch.tensor(
                            start_feet, dtype=torch.float64
                        ),
                        start_support=torch.tensor((True, True)),
                        heading_scene_xy=torch.tensor(
                            heading, dtype=torch.float64
                        ),
                        distance_m=length,
                        speed_mps=speed,
                        stride_m=stride,
                        step_width_m=step_width_m,
                        frames_per_second=50.0,
                    )
                )
                if len(path.footprints) < 2:
                    continue
                output.append(
                    _rounded_signature(
                        path_contact_signature(
                            path=path, sample_surface=sample_torch
                        )
                    )
                )
    unique = {
        (
            query.foot_order,
            query.forward_m,
            query.lateral_m,
            query.contact_frame,
            query.height_pattern_m,
        ): query
        for query in output
    }
    if not unique:
        raise ContractError("phase query has no contact patterns")
    return tuple(unique[key] for key in sorted(unique))


def _classify_lane_phases(records: object) -> str:
    if (
        not isinstance(records, (list, tuple))
        or not records
        or any(
            not isinstance(record, dict)
            or record.get("kind") not in _PHASE_ORDER
            or record.get("classification")
            not in ("full", "partial", "infeasible")
            for record in records
        )
        or len({str(record["kind"]) for record in records}) != len(records)
    ):
        raise ContractError("lane phase records are invalid")
    states = tuple(str(record["classification"]) for record in records)
    if all(state == "full" for state in states):
        return "full"
    if any(state != "infeasible" for state in states):
        return "partial"
    return "infeasible"


def _ordered_phase_records(records: object) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(records, (list, tuple))
        or any(
            not isinstance(record, dict)
            or type(record.get("lane_index")) is not int
            or record.get("kind") not in _PHASE_ORDER
            for record in records
        )
    ):
        raise ContractError("phase report records are invalid")
    return tuple(
        sorted(
            records,
            key=lambda item: (
                int(item["lane_index"]),
                _PHASE_ORDER[str(item["kind"])],
            ),
        )
    )


def _batched_signature_costs(
    queries: tuple[PathContactSignature, ...],
    windows: tuple[RawMotionWindow, ...],
) -> np.ndarray:
    if (
        not isinstance(queries, tuple)
        or not queries
        or any(not isinstance(query, PathContactSignature) for query in queries)
        or not isinstance(windows, tuple)
        or not windows
        or any(not isinstance(window, RawMotionWindow) for window in windows)
    ):
        raise ContractError("batched phase scoring inputs are invalid")
    count = len(queries[0].foot_order)
    if (
        any(len(query.foot_order) != count for query in queries)
        or any(len(window.events) != count for window in windows)
    ):
        raise ContractError("batched phase event counts differ")

    event_xy = np.asarray(
        [
            [event.position_world_xy for event in window.events]
            for window in windows
        ],
        dtype=np.float64,
    )
    displacement = event_xy[:, -1] - event_xy[:, 0]
    norm = np.linalg.norm(displacement, axis=1)
    if np.any(norm <= 1.0e-6):
        raise ContractError("batched phase window has zero displacement")
    forward_axis = displacement / norm[:, None]
    lateral_axis = np.column_stack(
        (-forward_axis[:, 1], forward_axis[:, 0])
    )
    root_start = np.asarray(
        [window.root_start_world_xy for window in windows],
        dtype=np.float64,
    )
    relative = event_xy - root_start[:, None, :]
    source_forward = np.einsum("nci,ni->nc", relative, forward_axis)
    source_lateral = np.einsum("nci,ni->nc", relative, lateral_axis)
    source_height = np.asarray(
        [
            [event.surface_height_m for event in window.events]
            for window in windows
        ],
        dtype=np.float64,
    )
    source_height -= source_height[:, :1]
    source_time = np.asarray(
        [
            [event.frame for event in window.events]
            for window in windows
        ],
        dtype=np.float64,
    )
    source_time -= source_time[:, :1]
    source_foot = np.asarray(
        [[event.foot for event in window.events] for window in windows],
        dtype=np.int64,
    )

    query_forward = np.asarray(
        [query.forward_m for query in queries], dtype=np.float64
    )
    query_forward -= query_forward[:, :1]
    query_lateral = np.asarray(
        [query.lateral_m for query in queries], dtype=np.float64
    )
    query_height = np.asarray(
        [query.height_pattern_m for query in queries], dtype=np.float64
    )
    query_time = np.asarray(
        [query.contact_frame for query in queries], dtype=np.float64
    )
    query_time -= query_time[:, :1]
    query_foot = np.asarray(
        [query.foot_order for query in queries], dtype=np.int64
    )
    progress = np.asarray(
        [window.forward_progress_m for window in windows], dtype=np.float64
    )
    heading = np.asarray(
        [window.heading_error_rad for window in windows], dtype=np.float64
    )

    cost = (
        20.0
        * np.sum(
            source_foot[:, None, :] != query_foot[None, :, :], axis=2
        )
        + 60.0
        * np.mean(
            np.square(
                source_height[:, None, :] - query_height[None, :, :]
            ),
            axis=2,
        )
        + 2.0
        * np.mean(
            np.square(
                source_forward[:, None, :] - query_forward[None, :, :]
            ),
            axis=2,
        )
        + 2.0
        * np.mean(
            np.square(
                source_lateral[:, None, :] - query_lateral[None, :, :]
            ),
            axis=2,
        )
        + 0.002
        * np.mean(
            np.square(source_time[:, None, :] - query_time[None, :, :]),
            axis=2,
        )
        + 2.0
        * np.square(progress[:, None] - query_forward[None, :, -1])
        + 2.0 * np.square(heading[:, None])
    )
    return np.min(cost, axis=1)


def _score_windows_for_tasks(
    windows: tuple[RawMotionWindow, ...],
    task_queries: tuple[tuple[PathContactSignature, ...], ...],
    task_lengths: tuple[float, ...],
    *,
    task_kinds: tuple[str, ...] | None = None,
    source_support: np.ndarray | None = None,
    source_surface: np.ndarray | None = None,
) -> tuple[tuple[dict[str, object], ...], ...]:
    if (
        not isinstance(windows, tuple)
        or any(not isinstance(window, RawMotionWindow) for window in windows)
        or not isinstance(task_queries, tuple)
        or not task_queries
        or len(task_queries) != len(task_lengths)
        or (
            task_kinds is not None
            and (
                len(task_kinds) != len(task_queries)
                or any(kind not in _PHASE_ORDER for kind in task_kinds)
                or source_support is None
                or ("dismount" in task_kinds and source_surface is None)
            )
        )
        or any(
            not isinstance(queries, tuple)
            or not queries
            or any(
                not isinstance(query, PathContactSignature)
                for query in queries
            )
            for queries in task_queries
        )
        or any(
            not math.isfinite(float(length)) or length <= 0.0
            for length in task_lengths
        )
    ):
        raise ContractError("phase scoring inputs are invalid")
    windows_by_kind = {"all": windows}
    if task_kinds is not None and "mount" in task_kinds:
        windows_by_kind["mount"] = _windows_for_phase_kind(
            windows, "mount", source_support
        )
    if task_kinds is not None and "dismount" in task_kinds:
        windows_by_kind["dismount"] = _windows_for_phase_kind(
            windows,
            "dismount",
            source_support,
            source_surface=source_surface,
        )
    windows_by_count = {}
    for kind, selected_windows in windows_by_kind.items():
        grouped: dict[int, tuple[RawMotionWindow, ...]] = {}
        for window in selected_windows:
            grouped.setdefault(len(window.events), ())
            grouped[len(window.events)] += (window,)
        windows_by_count[kind] = grouped
    output = []
    for task_index, (queries, path_length) in enumerate(
        zip(task_queries, task_lengths)
    ):
        kind = (
            "all" if task_kinds is None else task_kinds[task_index]
        )
        if kind == "interior":
            kind = "all"
        by_count: dict[int, tuple[PathContactSignature, ...]] = {}
        for query in queries:
            by_count.setdefault(len(query.foot_order), ())
            by_count[len(query.foot_order)] += (query,)
        rows = []
        for count, compatible in by_count.items():
            matching_windows = windows_by_count[kind].get(count, ())
            if not matching_windows:
                continue
            costs = _batched_signature_costs(
                compatible, matching_windows
            )
            rows.extend(
                _window_to_row(window, float(cost))
                for window, cost in zip(matching_windows, costs)
                if math.isfinite(float(cost))
            )
        output.append(
            _retain_event_count_diversity(
                rows, per_count=2, target_progress_m=path_length
            )
            if rows
            else ()
        )
    return tuple(output)


def _scan_one(arguments):
    root_text, descriptor, task_queries, task_lengths, task_kinds = arguments
    try:
        profiles = _source_profiles(Path(root_text), descriptor)
        counts = tuple(
            len(query.foot_order)
            for queries in task_queries
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
        return _score_windows_for_tasks(
            windows,
            task_queries,
            task_lengths,
            task_kinds=task_kinds,
            source_support=profiles["support"],
            source_surface=profiles["surface"],
        )
    except (ContractError, KeyError, OSError, ValueError):
        return tuple(() for _ in task_queries)


def _write_phase_artifacts(
    *,
    output: Path,
    lane_id: str,
    kind: str,
    connector: dict[str, np.ndarray] | None,
    reports: dict[str, object],
) -> None:
    root = output / "lanes" / lane_id / kind
    root.mkdir(parents=True, exist_ok=True)
    for name in ("placements", "coverage"):
        (root / f"{name}.json").write_text(
            json.dumps(reports[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if connector is not None:
        np.savez_compressed(root / "traversal.npz", **connector)


def main() -> int:
    args = _parser().parse_args()
    values = (
        args.start_x,
        args.stop_x,
        args.minimum_y,
        args.maximum_y,
        args.spacing_m,
        args.step_width_m,
    )
    if (
        not all(math.isfinite(float(value)) for value in values)
        or args.workers < 1
        or args.coarse_results < 2
        or args.placement_shortlist < 2
    ):
        raise ContractError("phase grid options are invalid")
    lanes = horizontal_grid_lanes(
        start_x=float(args.start_x),
        stop_x=float(args.stop_x),
        minimum_y=float(args.minimum_y),
        maximum_y=float(args.maximum_y),
        spacing_m=float(args.spacing_m),
    )
    root = args.source_dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    targets = [
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("logical_name") == args.target_scene
        or descriptor.get("relative_motion_path") == args.target_scene
    ]
    if len(targets) != 1 or targets[0].get("kind") != "terrain":
        raise ContractError("phase grid target scene must resolve exactly once")
    target = targets[0]
    origin, cell, height = _target_grid(root, target)

    def sample_numpy(points):
        return _sample_height_grid(
            origin_xy=origin,
            cell_size_m=cell,
            height_z=height,
            points_xy=np.asarray(points, dtype=np.float64),
        )

    tasks: list[tuple[HorizontalGridLane, TerrainPathPhase]] = []
    for lane in lanes:
        phases = segment_path_surface(
            path_start_scene_xy=lane.start_scene_xy,
            path_stop_scene_xy=lane.stop_scene_xy,
            step_width_m=float(args.step_width_m),
            sample_surface=sample_numpy,
        )
        tasks.extend((lane, phase) for phase in phases)
    task_queries = tuple(
        _phase_queries(
            phase=phase,
            step_width_m=float(args.step_width_m),
            sample_surface=sample_numpy,
        )
        for _, phase in tasks
    )
    task_lengths = tuple(
        float(phase.stop_m - phase.start_m) for _, phase in tasks
    )
    task_kinds = tuple(phase.kind for _, phase in tasks)
    descriptors = [
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("kind") == "terrain"
        and descriptor.get("terrain", {}).get("kind") == "heightgrid"
    ]
    descriptors.sort(key=lambda item: item["logical_name"])
    jobs = (
        (str(root), descriptor, task_queries, task_lengths, task_kinds)
        for descriptor in descriptors
    )
    pools: list[list[dict[str, object]]] = [[] for _ in tasks]
    returned_rows = [0 for _ in tasks]
    pool_limit = max(
        int(args.coarse_results), 4 * int(args.placement_shortlist)
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for clip_index, task_rows in enumerate(
            executor.map(_scan_one, jobs, chunksize=8), start=1
        ):
            for task_index, rows in enumerate(task_rows):
                pools[task_index].extend(rows)
                returned_rows[task_index] += len(rows)
            if clip_index % 256 == 0:
                for task_index, length in enumerate(task_lengths):
                    if len(pools[task_index]) > pool_limit:
                        pools[task_index] = list(
                            _placement_shortlist_rows(
                                pools[task_index],
                                path_length_m=length,
                                maximum_results=pool_limit,
                            )
                        )
    for task_index, length in enumerate(task_lengths):
        if len(pools[task_index]) > pool_limit:
            pools[task_index] = list(
                _placement_shortlist_rows(
                    pools[task_index],
                    path_length_m=length,
                    maximum_results=pool_limit,
                )
            )

    by_name = {
        descriptor["logical_name"]: descriptor for descriptor in descriptors
    }
    target_root_scene_xy, alignment_yaw = _target_alignment(root, target)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    args.output.mkdir(parents=True, exist_ok=True)
    phase_records = []
    for task_index, ((lane, phase), length) in enumerate(
        zip(tasks, task_lengths)
    ):
        phase_lane = HorizontalGridLane(
            index=task_index,
            lane_id=f"{lane.lane_id}-{phase.kind}",
            center_y_m=lane.center_y_m,
            start_scene_xy=phase.start_scene_xy,
            stop_scene_xy=phase.stop_scene_xy,
        )
        summary, connector, reports = _certify_lane(
            lane=phase_lane,
            rows=pools[task_index],
            root=root,
            by_name=by_name,
            sole_kinematics=sole_kinematics,
            sample_surface=sample_numpy,
            path_length_m=length,
            placement_shortlist=int(args.placement_shortlist),
            target_root_scene_xy=target_root_scene_xy,
            alignment_yaw=alignment_yaw,
        )
        record = {
            "lane_id": lane.lane_id,
            "lane_index": lane.index,
            "center_y_m": lane.center_y_m,
            "kind": phase.kind,
            "start_m": phase.start_m,
            "stop_m": phase.stop_m,
            "path_length_m": length,
            "start_scene_xy": list(phase.start_scene_xy),
            "stop_scene_xy": list(phase.stop_scene_xy),
            "mean_support_height_m": phase.mean_support_height_m,
            "normalized_support_height_m": list(
                phase.normalized_support_height_m
            ),
            "classification": summary["classification"],
            "covered_length_m": summary["covered_length_m"],
            "selected": summary["selected"],
            "rejected_by_reason": summary["rejected_by_reason"],
            "coarse_rows_returned": returned_rows[task_index],
            "coarse_pool_retained": len(pools[task_index]),
        }
        phase_records.append(record)
        _write_phase_artifacts(
            output=args.output,
            lane_id=lane.lane_id,
            kind=phase.kind,
            connector=connector,
            reports=reports,
        )
        print(
            json.dumps(
                {
                    "lane": lane.lane_id,
                    "phase": phase.kind,
                    "classification": record["classification"],
                    "selected": (
                        None
                        if record["selected"] is None
                        else record["selected"]["source_clip"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    phase_records = list(_ordered_phase_records(tuple(phase_records)))
    lane_records = []
    for lane in lanes:
        records = [
            record
            for record in phase_records
            if record["lane_index"] == lane.index
        ]
        lane_records.append(
            {
                "lane_id": lane.lane_id,
                "lane_index": lane.index,
                "center_y_m": lane.center_y_m,
                "classification": _classify_lane_phases(tuple(records)),
                "phases": records,
            }
        )
    lane_counts = Counter(
        str(record["classification"]) for record in lane_records
    )
    phase_counts = Counter(
        str(record["classification"]) for record in phase_records
    )
    summary = {
        "schema": "g1-horizontal-grid-phase-coverage/v1",
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
        "lane_classification_counts": {
            name: int(lane_counts.get(name, 0))
            for name in ("full", "partial", "infeasible")
        },
        "phase_classification_counts": {
            name: int(phase_counts.get(name, 0))
            for name in ("full", "partial", "infeasible")
        },
        "lanes": lane_records,
    }
    (args.output / "phase-grid-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
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
                "lane_classification_counts": summary[
                    "lane_classification_counts"
                ],
                "phase_classification_counts": summary[
                    "phase_classification_counts"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
