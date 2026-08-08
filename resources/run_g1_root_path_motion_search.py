#!/usr/bin/env python3
"""Search a complete terrain-valid G1 motion chain along one root path."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_heading_footprint_realizer import blend_action_boundary
from mm_sonic.torch_path_motion_placement import (
    PlacementRejected,
    RawContactEvent,
    RawMotionWindow,
    place_raw_window_on_path,
)
from mm_sonic.torch_path_terrain_phases import TerrainPathPhase
from mm_sonic.torch_ordered_terrain_levels import (
    OrderedLevelRejected,
    OrderedTerrainLevelContract,
    derive_ordered_terrain_levels,
    match_ordered_touchdown_levels,
)
from mm_sonic.torch_root_path_motion_graph import (
    OrderedLevelMotionCandidate,
    RootPathMotionCandidate,
    RootPathMotionTransition,
    root_path_search_anchors,
    select_ordered_level_motion_chain,
    select_root_path_motion_chain,
)
from resources.run_g1_grail_contact_inventory import _sample_height_grid
from resources.run_g1_horizontal_grid_phase_coverage import (
    _phase_queries,
    _scan_one,
)
from resources.run_g1_horizontal_grid_coverage import _target_alignment
from resources.run_g1_path_motion_placement import (
    _matcher_archive,
    _minimal_dataset,
    _placement_shortlist_rows,
    _row_to_window,
    _source_profiles,
    _target_grid,
)


_G1_LEG_JOINT_INDICES_BY_FOOT = (
    np.asarray((0, 3, 6, 9, 13, 17), dtype=np.int64),
    np.asarray((1, 4, 7, 10, 14, 18), dtype=np.int64),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--target-scene", required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--start-x", type=float, default=-1.2795985755)
    parser.add_argument("--stop-x", type=float, default=1.2590216406)
    parser.add_argument("--path-y", type=float)
    parser.add_argument(
        "--path-start", type=float, nargs=2, metavar=("X", "Y")
    )
    parser.add_argument(
        "--path-stop", type=float, nargs=2, metavar=("X", "Y")
    )
    parser.add_argument("--segment-length-m", type=float, default=0.90)
    parser.add_argument("--search-stride-m", type=float, default=0.20)
    parser.add_argument("--step-width-m", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--coarse-results", type=int, default=2400)
    parser.add_argument("--placement-shortlist", type=int, default=600)
    parser.add_argument("--candidates-per-anchor", type=int, default=24)
    parser.add_argument("--blend-frames", type=int, default=4)
    parser.add_argument("--ordered-contact-levels", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve_path(
    args: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    explicit = args.path_start is not None or args.path_stop is not None
    if explicit:
        if (
            args.path_start is None
            or args.path_stop is None
            or args.path_y is not None
        ):
            raise ContractError("root path specification is ambiguous")
        start = np.asarray(args.path_start, dtype=np.float64)
        stop = np.asarray(args.path_stop, dtype=np.float64)
    else:
        if args.path_y is None:
            raise ContractError("root path specification is missing")
        start = np.array((args.start_x, args.path_y), dtype=np.float64)
        stop = np.array((args.stop_x, args.path_y), dtype=np.float64)
    displacement = stop - start
    length = float(np.linalg.norm(displacement))
    if (
        start.shape != (2,)
        or stop.shape != (2,)
        or not np.isfinite(start).all()
        or not np.isfinite(stop).all()
        or not math.isfinite(length)
        or length <= 0.0
    ):
        raise ContractError("root path endpoints are invalid")
    return start, stop, displacement / length, length


def _scanner_job(
    root: object,
    descriptor: dict[str, object],
    queries: tuple[object, ...],
    lengths: tuple[float, ...],
    phases: tuple[TerrainPathPhase, ...],
) -> tuple[object, ...]:
    return (
        str(root),
        descriptor,
        queries,
        lengths,
        tuple(phase.kind for phase in phases),
    )


def _minimum_search_window_length(segment_length_m: float) -> float:
    if (
        isinstance(segment_length_m, bool)
        or not isinstance(segment_length_m, (int, float))
        or not math.isfinite(float(segment_length_m))
        or segment_length_m <= 0.0
    ):
        raise ContractError("segment length is invalid")
    return min(0.65, float(segment_length_m))


def _search_window_reaches_path_end(
    *, anchor_m: float, length_m: float, path_length_m: float
) -> bool:
    values = (anchor_m, length_m, path_length_m)
    if (
        any(not math.isfinite(float(value)) for value in values)
        or anchor_m < 0.0
        or length_m <= 0.0
        or path_length_m <= 0.0
        or anchor_m >= path_length_m
    ):
        raise ContractError("search window interval is invalid")
    return anchor_m + length_m >= path_length_m - 1.0e-6


def _requires_terminal_settle(
    *, anchor_m: float, length_m: float, path_length_m: float
) -> bool:
    """Apply the terminal support invariant to every terrain mode."""

    return _search_window_reaches_path_end(
        anchor_m=anchor_m,
        length_m=length_m,
        path_length_m=path_length_m,
    )


def _ordered_phase_kind(
    contract: OrderedTerrainLevelContract,
    start_m: float,
    stop_m: float,
) -> str:
    if (
        not isinstance(contract, OrderedTerrainLevelContract)
        or not math.isfinite(float(start_m))
        or not math.isfinite(float(stop_m))
        or start_m < 0.0
        or stop_m <= start_m
        or stop_m > contract.path_length_m + 1.0e-6
    ):
        raise ContractError("ordered phase interval is invalid")
    crossed = tuple(
        level.index
        for level in contract.levels[1:]
        if start_m < level.start_m <= stop_m
    )
    if crossed == (contract.levels[-1].index,):
        return (
            "dismount"
            if contract.levels[-1].height_m
            < contract.levels[-2].height_m
            else "mount"
        )
    if crossed == (1,):
        return "mount"
    return "interior"


def _query_phase(
    *,
    path_start: np.ndarray,
    heading: np.ndarray,
    start_m: float,
    length_m: float,
    step_width_m: float,
    sample_surface,
    kind: str = "interior",
) -> TerrainPathPhase:
    start = path_start + start_m * heading
    stop = start + length_m * heading
    lateral = np.array((-heading[1], heading[0]), dtype=np.float64)
    feet = start[None, :] + np.stack(
        (0.5 * step_width_m * lateral, -0.5 * step_width_m * lateral)
    )
    support_height = np.asarray(sample_surface(feet), dtype=np.float64)
    mean = float(support_height.mean())
    return TerrainPathPhase(
        kind=kind,
        start_m=start_m,
        stop_m=start_m + length_m,
        start_scene_xy=tuple(start),
        stop_scene_xy=tuple(stop),
        mean_support_height_m=mean,
        normalized_support_height_m=tuple(support_height - mean),
    )


def _shifted_window(window: RawMotionWindow) -> RawMotionWindow:
    start, stop = window.start_frame, window.stop_frame
    return RawMotionWindow(
        source_clip=window.source_clip,
        start_frame=0,
        stop_frame=stop - start,
        events=tuple(
            RawContactEvent(
                frame=event.frame - start,
                foot=event.foot,
                position_world_xy=event.position_world_xy,
                surface_height_m=event.surface_height_m,
            )
            for event in window.events
        ),
        root_start_world_xy=window.root_start_world_xy,
        forward_progress_m=window.forward_progress_m,
        heading_error_rad=window.heading_error_rad,
    )


def _extend_window_to_terminal_double_support(
    window: RawMotionWindow,
    support_mask: object,
    *,
    minimum_frames: int = 4,
    maximum_extension_frames: int = 30,
) -> RawMotionWindow:
    """Extend a dismount window through the first stable two-foot settle."""

    support = np.asarray(support_mask)
    if (
        not isinstance(window, RawMotionWindow)
        or support.ndim != 2
        or support.shape[1] != 2
        or support.dtype != np.bool_
        or isinstance(minimum_frames, bool)
        or not isinstance(minimum_frames, int)
        or minimum_frames < 1
        or isinstance(maximum_extension_frames, bool)
        or not isinstance(maximum_extension_frames, int)
        or maximum_extension_frames < 0
        or window.stop_frame > len(support)
    ):
        raise ContractError("terminal double-support inputs are invalid")
    maximum_stop = min(
        len(support), window.stop_frame + maximum_extension_frames
    )
    for stop in range(window.stop_frame, maximum_stop + 1):
        start = stop - minimum_frames
        if (
            start >= window.start_frame
            and bool(support[start:stop].all())
        ):
            return RawMotionWindow(
                source_clip=window.source_clip,
                start_frame=window.start_frame,
                stop_frame=stop,
                events=window.events,
                root_start_world_xy=window.root_start_world_xy,
                forward_progress_m=window.forward_progress_m,
                heading_error_rad=window.heading_error_rad,
            )
    raise PlacementRejected("terminal-double-support")


def _extend_window_through_stable_liftoff(
    window: RawMotionWindow,
    support_mask: object,
    *,
    settle_frames: int = 4,
    maximum_extension_frames: int = 20,
) -> RawMotionWindow:
    """Keep a raw double-support endpoint through its natural foot release."""

    support = np.asarray(support_mask)
    if (
        not isinstance(window, RawMotionWindow)
        or support.ndim != 2
        or support.shape[1] != 2
        or support.dtype != np.bool_
        or window.stop_frame < 1
        or window.stop_frame > len(support)
        or type(settle_frames) is not int
        or settle_frames < 1
        or type(maximum_extension_frames) is not int
        or maximum_extension_frames < settle_frames
    ):
        raise ContractError("stable liftoff extension inputs are invalid")
    if not bool(support[window.stop_frame - 1].all()):
        return window
    maximum_stop = min(
        len(support), window.stop_frame + maximum_extension_frames
    )
    for start in range(window.stop_frame, maximum_stop - settle_frames + 1):
        phase = support[start : start + settle_frames]
        if (
            bool(phase.any(axis=1).all())
            and bool(np.all(phase == phase[:1]))
            and not bool(phase[0].all())
        ):
            return RawMotionWindow(
                source_clip=window.source_clip,
                start_frame=window.start_frame,
                stop_frame=start + settle_frames,
                events=window.events,
                root_start_world_xy=window.root_start_world_xy,
                forward_progress_m=window.forward_progress_m,
                heading_error_rad=window.heading_error_rad,
            )
    return window


def _extend_window_through_next_touchdown(
    window: RawMotionWindow,
    support_mask: object,
    foot_position_world: object,
    surface_height: object,
    *,
    settle_frames: int = 4,
    maximum_extension_frames: int = 100,
    touchdown_count: int = 1,
) -> RawMotionWindow:
    """Extend double support through the next certified raw touchdown."""

    support = np.asarray(support_mask)
    feet = np.asarray(foot_position_world, dtype=np.float64)
    surface = np.asarray(surface_height, dtype=np.float64)
    if (
        not isinstance(window, RawMotionWindow)
        or support.ndim != 2
        or support.shape[1] != 2
        or support.dtype != np.bool_
        or feet.shape != (len(support), 2, 3)
        or surface.shape != (len(support), 2)
        or not np.isfinite(feet).all()
        or not np.isfinite(surface).all()
        or window.stop_frame < 1
        or window.stop_frame > len(support)
        or type(settle_frames) is not int
        or settle_frames < 1
        or type(maximum_extension_frames) is not int
        or maximum_extension_frames < settle_frames
        or type(touchdown_count) is not int
        or touchdown_count < 1
    ):
        raise ContractError("next touchdown extension inputs are invalid")
    if not bool(support[window.stop_frame - 1].all()):
        return window
    maximum_stop = min(
        len(support), window.stop_frame + maximum_extension_frames
    )
    events = list(window.events)
    last_stop = None
    found = 0
    for frame in range(window.stop_frame, maximum_stop - settle_frames + 1):
        touchdown = support[frame] & ~support[frame - 1]
        stable = support[frame : frame + settle_frames, touchdown]
        if not bool(touchdown.any()) or not bool(stable.all()):
            continue
        for foot in np.flatnonzero(touchdown):
            events.append(
                RawContactEvent(
                    frame=int(frame),
                    foot=int(foot),
                    position_world_xy=tuple(feet[frame, foot, :2]),
                    surface_height_m=float(surface[frame, foot]),
                )
            )
            found += 1
        last_stop = frame + settle_frames
        if found < touchdown_count:
            continue
        phase_support = support[frame]
        while (
            last_stop < maximum_stop
            and bool(np.array_equal(support[last_stop], phase_support))
        ):
            last_stop += 1
        return RawMotionWindow(
            source_clip=window.source_clip,
            start_frame=window.start_frame,
            stop_frame=last_stop,
            events=tuple(sorted(events)),
            root_start_world_xy=window.root_start_world_xy,
            forward_progress_m=window.forward_progress_m,
            heading_error_rad=window.heading_error_rad,
        )
    if last_stop is not None:
        return RawMotionWindow(
            source_clip=window.source_clip,
            start_frame=window.start_frame,
            stop_frame=last_stop,
            events=tuple(sorted(events)),
            root_start_world_xy=window.root_start_world_xy,
            forward_progress_m=window.forward_progress_m,
            heading_error_rad=window.heading_error_rad,
        )
    return _extend_window_through_stable_liftoff(
        window,
        support,
        settle_frames=settle_frames,
        maximum_extension_frames=min(maximum_extension_frames, 20),
    )


def _placed_touchdowns(
    *,
    window: RawMotionWindow,
    yaw_scene_rad: float,
    translation_scene_xyz: object,
    path_start_scene_xy: object,
    path_heading_scene_xy: object,
    sample_surface,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    translation = np.asarray(translation_scene_xyz, dtype=np.float64)
    path_start = np.asarray(path_start_scene_xy, dtype=np.float64)
    heading = np.asarray(path_heading_scene_xy, dtype=np.float64)
    if (
        not isinstance(window, RawMotionWindow)
        or not math.isfinite(float(yaw_scene_rad))
        or translation.shape != (3,)
        or path_start.shape != (2,)
        or heading.shape != (2,)
        or not np.isfinite(translation).all()
        or not np.isfinite(path_start).all()
        or not np.isfinite(heading).all()
        or not callable(sample_surface)
    ):
        raise ContractError("placed touchdown inputs are invalid")
    norm = float(np.linalg.norm(heading))
    if norm <= 1.0e-6:
        raise ContractError("placed touchdown heading is zero")
    heading = heading / norm
    cosine = math.cos(float(yaw_scene_rad))
    sine = math.sin(float(yaw_scene_rad))
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    source_xy = np.asarray(
        [event.position_world_xy for event in window.events],
        dtype=np.float64,
    )
    scene_xy = source_xy @ rotation.T + translation[:2]
    progress = (scene_xy - path_start[None, :]) @ heading
    height = np.asarray(sample_surface(scene_xy), dtype=np.float64)
    if height.shape != progress.shape or not np.isfinite(height).all():
        raise ContractError("placed touchdown terrain samples are invalid")
    return (
        progress,
        height,
        tuple(event.foot for event in window.events),
        tuple(event.frame for event in window.events),
    )


def _prepare_candidate_window(
    window: RawMotionWindow,
    support_mask: object,
    foot_position_world: object,
    surface_height: object,
    *,
    require_terminal_double_support: bool,
) -> RawMotionWindow:
    del foot_position_world, surface_height
    if type(require_terminal_double_support) is not bool:
        raise ContractError("candidate terminal-support flag is invalid")
    if not require_terminal_double_support:
        return window
    return _extend_window_to_terminal_double_support(window, support_mask)


def _qualify_candidates(
    *,
    anchor_m: float,
    rows: list[dict[str, object]],
    path_start: np.ndarray,
    heading: np.ndarray,
    root: Path,
    by_name: dict[str, dict[str, object]],
    profiles_cache: dict[str, dict[str, np.ndarray]],
    soles_cache: dict[str, np.ndarray],
    sole_kinematics: MujocoG1SoleKinematics,
    sample_surface,
    target_root_scene_xy: np.ndarray,
    alignment_yaw: float,
    shortlist_count: int,
    candidate_count: int,
    segment_length_m: float,
    ordered_contract: OrderedTerrainLevelContract | None = None,
    require_terminal_double_support: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not rows:
        return [], {}
    shortlist = _placement_shortlist_rows(
        rows,
        path_length_m=float(segment_length_m),
        maximum_results=shortlist_count,
    )
    accepted = []
    rejected: dict[str, int] = {}
    anchor_xy = path_start + anchor_m * heading
    for row in shortlist:
        source = str(row["source_clip"])
        try:
            profiles = profiles_cache.get(source)
            if profiles is None:
                profiles = _source_profiles(root, by_name[source])
                profiles_cache[source] = profiles
            soles = soles_cache.get(source)
            if soles is None:
                soles = sole_kinematics.sole_points(
                    profiles["joint"],
                    profiles["root"],
                    profiles["quaternion"],
                )
                soles_cache[source] = soles
            window = _row_to_window(row)
            window = _prepare_candidate_window(
                window,
                profiles["support"],
                profiles["feet"],
                profiles["surface"],
                require_terminal_double_support=(
                    require_terminal_double_support
                ),
            )
            start, stop = window.start_frame, window.stop_frame
            shifted_window = _shifted_window(window)
            placed = place_raw_window_on_path(
                window=shifted_window,
                joint_position=profiles["joint"][start:stop],
                root_position_world=profiles["root"][start:stop],
                root_orientation_world_wxyz=profiles["quaternion"][start:stop],
                foot_position_world=profiles["feet"][start:stop],
                sole_position_world=soles[start:stop],
                support_mask=profiles["support"][start:stop],
                path_start_scene_xy=anchor_xy,
                path_heading_scene_xy=heading,
                sample_surface=sample_surface,
                maximum_lateral_error_m=(
                    0.18 if require_terminal_double_support else 0.15
                ),
            )
            level_indices = None
            touchdown_feet = None
            touchdown_frames = None
            if ordered_contract is not None:
                (
                    touchdown_progress,
                    touchdown_height,
                    touchdown_feet,
                    touchdown_frames,
                ) = _placed_touchdowns(
                    window=shifted_window,
                    yaw_scene_rad=placed.yaw_scene_rad,
                    translation_scene_xyz=placed.translation_scene_xyz,
                    path_start_scene_xy=path_start,
                    path_heading_scene_xy=heading,
                    sample_surface=sample_surface,
                )
                level_indices = match_ordered_touchdown_levels(
                    ordered_contract,
                    touchdown_progress_m=touchdown_progress,
                    touchdown_height_m=touchdown_height,
                    touchdown_foot=touchdown_feet,
                    progress_tolerance_m=0.20,
                    require_complete=False,
                )
            candidate_id = (
                f"{source}:{start}:{stop}:"
                f"{int(round(anchor_m * 1000.0))}"
            )
            interval_start = anchor_m + placed.metrics.covered_start_m
            interval_stop = anchor_m + placed.metrics.covered_stop_m
            placed_soles = sole_kinematics.sole_points(
                placed.joint_position,
                placed.root_position_scene,
                placed.root_orientation_scene_wxyz,
            )
            feet_scene = np.mean(placed_soles, axis=2)
            naturalness_penalty = _candidate_naturalness_penalty(
                support=np.asarray(
                    profiles["support"][start:stop], dtype=np.bool_
                ),
                feet_scene=feet_scene,
                heading_scene_xy=heading,
            )
            placement_cost = (
                float(row["cost"])
                + 2.0 * placed.metrics.maximum_lateral_error_m
                + 5.0 * placed.metrics.maximum_stance_error_m
                + max(0.0, -placed.metrics.minimum_sole_clearance_m)
                + naturalness_penalty
            )
            accepted.append(
                {
                    "candidate_id": candidate_id,
                    "anchor_m": anchor_m,
                    "interval_start_m": interval_start,
                    "interval_stop_m": interval_stop,
                    "placement_cost": placement_cost,
                    "row": dict(row),
                    "metrics": asdict(placed.metrics),
                    "naturalness_penalty": naturalness_penalty,
                    "feet_scene": feet_scene,
                    "scene": {
                        "joint_position": np.asarray(
                            placed.joint_position, dtype=np.float64
                        ),
                        "root_position_world": np.asarray(
                            placed.root_position_scene, dtype=np.float64
                        ),
                        "root_orientation_world_wxyz": np.asarray(
                            placed.root_orientation_scene_wxyz,
                            dtype=np.float64,
                        ),
                    },
                    "matcher": _matcher_archive(
                        placed,
                        target_root_scene_xy=target_root_scene_xy,
                        alignment_yaw=alignment_yaw,
                    ),
                    "support": np.asarray(
                        profiles["support"][start:stop], dtype=np.bool_
                    ),
                    "path_progress_m": (
                        np.asarray(
                            placed.root_position_scene[:, :2],
                            dtype=np.float64,
                        )
                        - path_start[None, :]
                    )
                    @ heading,
                    "ordered_levels": level_indices,
                    "touchdown_feet": touchdown_feet,
                    "touchdown_frames": touchdown_frames,
                }
            )
        except PlacementRejected as error:
            rejected[error.reason] = rejected.get(error.reason, 0) + 1
        except OrderedLevelRejected as error:
            rejected[error.reason] = rejected.get(error.reason, 0) + 1
        except (ContractError, KeyError, OSError, ValueError):
            rejected["invalid-source"] = rejected.get("invalid-source", 0) + 1
    accepted_by_id: dict[str, dict[str, object]] = {}
    for item in accepted:
        candidate_id = str(item["candidate_id"])
        previous = accepted_by_id.get(candidate_id)
        if (
            previous is None
            or float(item["placement_cost"])
            < float(previous["placement_cost"])
        ):
            accepted_by_id[candidate_id] = item
    accepted = list(accepted_by_id.values())
    accepted.sort(
        key=lambda item: (
            item["placement_cost"],
            -item["interval_stop_m"],
            item["candidate_id"],
        )
    )
    per_progress_limit = max(2, int(math.ceil(candidate_count / 12.0)))
    retained = []
    bucket_counts: dict[int, int] = {}
    retained_ids: set[str] = set()
    for item in accepted:
        progress = float(item["interval_stop_m"]) - float(
            item["interval_start_m"]
        )
        bucket = int(round(progress / 0.10))
        if bucket_counts.get(bucket, 0) >= per_progress_limit:
            continue
        retained.append(item)
        retained_ids.add(str(item["candidate_id"]))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(retained) >= candidate_count:
            break
    if len(retained) < candidate_count:
        for item in accepted:
            if str(item["candidate_id"]) in retained_ids:
                continue
            retained.append(item)
            if len(retained) >= candidate_count:
                break
    retained.sort(
        key=lambda item: (
            item["placement_cost"],
            -item["interval_stop_m"],
            item["candidate_id"],
        )
    )
    return retained, rejected


def _candidate_naturalness_penalty(
    *,
    support: object,
    feet_scene: object,
    heading_scene_xy: object,
) -> float:
    """Penalize crossed support and excessive double-support dwell."""

    support_array = np.asarray(support)
    feet = np.asarray(feet_scene, dtype=np.float64)
    heading = np.asarray(heading_scene_xy, dtype=np.float64)
    frame_count = len(support_array)
    if (
        support_array.shape != (frame_count, 2)
        or support_array.dtype != np.bool_
        or feet.shape != (frame_count, 2, 3)
        or heading.shape != (2,)
        or frame_count < 2
        or not np.isfinite(feet).all()
        or not np.isfinite(heading).all()
        or float(np.linalg.norm(heading)) <= 0.0
    ):
        raise ContractError("candidate naturalness input is invalid")
    forward = heading / np.linalg.norm(heading)
    lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
    separation = (feet[:, 0, :2] - feet[:, 1, :2]) @ lateral
    observed = support_array.any(axis=1)
    crossing = max(
        0.0,
        -float(separation[observed].min()) if bool(observed.any()) else 0.0,
    )
    double = support_array.all(axis=1)
    padded = np.pad(double.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    longest_double = max(
        (int(stop - start) for start, stop in zip(starts, stops)),
        default=0,
    )
    dwell = max(0, longest_double - 20)
    return 25.0 * crossing + 0.01 * dwell


def _transition_candidates(
    first: dict[str, object],
    second: dict[str, object],
) -> tuple[tuple[float, int, int], ...]:
    first_row = first["row"]
    second_row = second["row"]
    same_source_window = (
        first_row["source_clip"] == second_row["source_clip"]
        and first_row["start_frame"] == second_row["start_frame"]
        and first_row["stop_frame"] == second_row["stop_frame"]
    )
    same_source_clip = (
        first_row["source_clip"] == second_row["source_clip"]
    )
    if (
        float(second["interval_stop_m"])
        <= float(first["interval_stop_m"]) + 1.0e-6
        or float(second["interval_start_m"])
        > float(first["interval_stop_m"]) + 0.15
    ):
        return ()
    first_arrays = first["scene"]
    second_arrays = second["scene"]
    first_support = np.asarray(first["support"])
    second_support = np.asarray(second["support"])
    first_root = np.asarray(first_arrays["root_position_world"])
    second_root = np.asarray(second_arrays["root_position_world"])
    first_joint = np.asarray(first_arrays["joint_position"])
    second_joint = np.asarray(second_arrays["joint_position"])
    first_velocity = np.gradient(first_joint, axis=0)
    second_velocity = np.gradient(second_joint, axis=0)
    overlap_start = max(
        float(first["interval_start_m"]),
        float(second["interval_start_m"]),
    )
    overlap_stop = min(
        float(first["interval_stop_m"]),
        float(second["interval_stop_m"]),
    )
    first_progress = np.asarray(first["path_progress_m"], dtype=np.float64)
    second_progress = np.asarray(
        second["path_progress_m"], dtype=np.float64
    )
    first_indices = np.flatnonzero(
        first_support.any(axis=1)
        & (first_progress >= overlap_start - 0.05)
        & (first_progress <= overlap_stop + 0.05)
    )
    second_indices = np.flatnonzero(
        second_support.any(axis=1)
        & (second_progress >= overlap_start - 0.05)
        & (second_progress <= overlap_stop + 0.05)
    )
    second_levels = second.get("ordered_levels")
    second_touchdown_frames = second.get("touchdown_frames")
    if (
        isinstance(second_levels, tuple)
        and isinstance(second_touchdown_frames, tuple)
        and len(second_levels) >= 2
        and len(second_levels) == len(second_touchdown_frames)
        and second_levels[-1] == second_levels[-2]
        and bool(np.asarray(second["support"], dtype=np.bool_)[-1].all())
    ):
        # A repeated terminal level is a two-contact semantic unit.  Cutting
        # after its penultimate touchdown silently turns a settled exit into
        # a one-foot landing.
        second_indices = second_indices[
            second_indices <= second_touchdown_frames[-2]
        ]
    if not len(first_indices) or not len(second_indices):
        return ()
    candidates = []
    for first_frame in first_indices:
        root_gap = np.linalg.norm(
            first_root[first_frame][None, :] - second_root[second_indices],
            axis=1,
        )
        joint_gap = np.linalg.norm(
            first_joint[first_frame][None, :]
            - second_joint[second_indices],
            axis=1,
        )
        velocity_gap = np.linalg.norm(
            first_velocity[first_frame][None, :]
            - second_velocity[second_indices],
            axis=1,
        )
        support_invention = np.any(
            second_support[second_indices]
            & ~first_support[first_frame][None, :],
            axis=1,
        )
        valid = (
            (root_gap <= 0.20)
            & (joint_gap <= 4.0)
            & (~support_invention)
        )
        if same_source_window:
            valid &= second_indices >= first_frame
        for local_index in np.flatnonzero(valid):
            second_frame = int(second_indices[local_index])
            score = (
                10.0 * float(root_gap[local_index])
                + 0.10 * float(joint_gap[local_index])
                + 0.04 * float(velocity_gap[local_index])
                + 0.5
                * abs(
                    float(first_progress[first_frame])
                    - float(second_progress[second_frame])
                )
            )
            score = (
                max(0.0, score - 0.05)
                if same_source_clip
                else score + 0.05
            )
            candidates.append((score, int(first_frame), second_frame))
    if not candidates:
        return ()
    return tuple(item for item in sorted(candidates) if item[0] <= 2.00)


def _best_transition(
    first: dict[str, object],
    second: dict[str, object],
) -> tuple[float, int, int] | None:
    candidates = _transition_candidates(first, second)
    return candidates[0] if candidates else None


def _best_certified_transition(
    first: dict[str, object],
    second: dict[str, object],
    *,
    certify,
    maximum_attempts: int = 4,
) -> tuple[tuple[float, int, int], dict[str, float]] | None:
    if not callable(certify) or type(maximum_attempts) is not int or maximum_attempts < 1:
        raise ContractError("certified transition search options are invalid")
    for candidate in _transition_candidates(first, second)[:maximum_attempts]:
        certification = certify(candidate[1], candidate[2])
        if certification is not None:
            return candidate, certification
    return None


def _contact_preserving_boundary_blend(
    *,
    joint_position: object,
    root_position_world: object,
    root_orientation_world_wxyz: object,
    previous_joint_position: object,
    previous_root_position_world: object,
    previous_root_orientation_world_wxyz: object,
    support_mask: object,
    blend_frames: int,
    sole_kinematics: MujocoG1SoleKinematics,
    blend_mode: str = "contact",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend poses while retaining the incoming terrain-valid stance soles."""

    incoming_joints = np.asarray(joint_position, dtype=np.float64)
    incoming_roots = np.asarray(root_position_world, dtype=np.float64)
    incoming_quaternions = np.asarray(
        root_orientation_world_wxyz, dtype=np.float64
    )
    support = np.asarray(support_mask)
    if (
        support.shape != (len(incoming_joints), 2)
        or support.dtype != np.bool_
        or not bool(support[:blend_frames].any())
        or blend_mode not in ("contact", "pose")
    ):
        raise ContractError("contact-preserving blend support is invalid")
    joints, roots, quaternions = blend_action_boundary(
        joint_position=incoming_joints,
        root_position_world=incoming_roots,
        root_orientation_world_wxyz=incoming_quaternions,
        previous_joint_position=previous_joint_position,
        previous_root_position_world=previous_root_position_world,
        previous_root_orientation_world_wxyz=(
            previous_root_orientation_world_wxyz
        ),
        blend_frames=blend_frames,
    )
    if blend_mode == "pose":
        return (
            np.ascontiguousarray(joints),
            np.ascontiguousarray(roots),
            np.ascontiguousarray(quaternions),
        )
    # The incoming clip has already passed full sole-terrain certification.
    # Keep its initial swing leg geometry so pose interpolation cannot drive
    # that foot through terrain.  Blend the stance leg, then restore its
    # contact below with the root projection.
    initial_support = support[0]
    for foot in range(2):
        if bool(initial_support[foot]):
            continue
        indices = _G1_LEG_JOINT_INDICES_BY_FOOT[foot]
        joints[:blend_frames, indices] = incoming_joints[
            :blend_frames, indices
        ]
    target_soles = sole_kinematics.sole_points(
        incoming_joints, incoming_roots, incoming_quaternions
    )
    blended_soles = sole_kinematics.sole_points(
        joints, roots, quaternions
    )
    correction = np.zeros((blend_frames, 3), dtype=np.float64)
    known = np.flatnonzero(support[:blend_frames].any(axis=1))
    for frame in known:
        supported = support[frame]
        correction[frame] = np.mean(
            target_soles[frame, supported]
            - blended_soles[frame, supported],
            axis=(0, 1),
        )
    correction[: known[0]] = correction[known[0]]
    correction[known[-1] + 1 :] = correction[known[-1]]
    for left, right in zip(known[:-1], known[1:]):
        if right == left + 1:
            continue
        phase = np.linspace(0.0, 1.0, right - left + 1)
        smooth = phase * phase * (3.0 - 2.0 * phase)
        correction[left : right + 1] = (
            correction[left][None] * (1.0 - smooth[:, None])
            + correction[right][None] * smooth[:, None]
        )
    roots[:blend_frames] += correction
    return (
        np.ascontiguousarray(joints),
        np.ascontiguousarray(roots),
        np.ascontiguousarray(quaternions),
    )


def _compose_chain(
    chain: tuple[dict[str, object], ...],
    transition_details: dict[
        tuple[str, str], tuple[float, int, int, str]
    ],
    *,
    blend_frames: int,
    sole_kinematics: MujocoG1SoleKinematics,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, object]]]:
    incoming = [0 for _ in chain]
    outgoing = [
        len(item["scene"]["joint_position"]) for item in chain
    ]
    transitions = []
    blend_modes = []
    for index in range(len(chain) - 1):
        first = chain[index]
        second = chain[index + 1]
        score, first_frame, second_frame, blend_mode = transition_details[
            (first["candidate_id"], second["candidate_id"])
        ]
        outgoing[index] = first_frame + 1
        incoming[index + 1] = second_frame
        transitions.append(
            {
                "first_candidate_id": first["candidate_id"],
                "second_candidate_id": second["candidate_id"],
                "first_source_frame": first_frame,
                "second_source_frame": second_frame,
                "cost": score,
                "blend_mode": blend_mode,
            }
        )
        blend_modes.append(blend_mode)
    segments = []
    supports = []
    for index, item in enumerate(chain):
        start, stop = incoming[index], outgoing[index]
        if stop - start < blend_frames:
            raise ContractError("selected root-path source interval is empty")
        segments.append(
            {
                name: np.array(value[start:stop], copy=True)
                for name, value in item["scene"].items()
            }
        )
        supports.append(np.array(item["support"][start:stop], copy=True))
    composed = segments[0]
    composed_support = supports[0]
    for segment, support, blend_mode in zip(
        segments[1:], supports[1:], blend_modes
    ):
        joints, roots, quaternion = _contact_preserving_boundary_blend(
            joint_position=segment["joint_position"],
            root_position_world=segment["root_position_world"],
            root_orientation_world_wxyz=segment[
                "root_orientation_world_wxyz"
            ],
            previous_joint_position=composed["joint_position"][-1],
            previous_root_position_world=composed["root_position_world"][-1],
            previous_root_orientation_world_wxyz=composed[
                "root_orientation_world_wxyz"
            ][-1],
            support_mask=support,
            blend_frames=blend_frames,
            sole_kinematics=sole_kinematics,
            blend_mode=blend_mode,
        )
        segment = {
            "joint_position": joints,
            "root_position_world": roots,
            "root_orientation_world_wxyz": quaternion,
        }
        composed = {
            name: np.ascontiguousarray(
                np.concatenate((composed[name], segment[name]), axis=0)
            )
            for name in composed
        }
        composed_support = np.ascontiguousarray(
            np.concatenate((composed_support, support), axis=0)
        )
    return composed, composed_support, transitions


def _certify_transition(
    *,
    first: dict[str, object],
    second: dict[str, object],
    first_frame: int,
    second_frame: int,
    blend_frames: int,
    sole_kinematics: MujocoG1SoleKinematics,
    sample_surface,
    blend_mode: str = "contact",
) -> dict[str, float] | None:
    first_arrays = first["scene"]
    second_arrays = second["scene"]
    stop = second_frame + blend_frames
    if stop > len(second_arrays["joint_position"]):
        return None
    support = np.asarray(second["support"])[second_frame:stop]
    joints, roots, quaternion = _contact_preserving_boundary_blend(
        joint_position=second_arrays["joint_position"][second_frame:stop],
        root_position_world=second_arrays["root_position_world"][
            second_frame:stop
        ],
        root_orientation_world_wxyz=second_arrays[
            "root_orientation_world_wxyz"
        ][second_frame:stop],
        previous_joint_position=first_arrays["joint_position"][first_frame],
        previous_root_position_world=first_arrays["root_position_world"][
            first_frame
        ],
        previous_root_orientation_world_wxyz=first_arrays[
            "root_orientation_world_wxyz"
        ][first_frame],
        support_mask=support,
        blend_frames=blend_frames,
        sole_kinematics=sole_kinematics,
        blend_mode=blend_mode,
    )
    soles = sole_kinematics.sole_points(joints, roots, quaternion)
    surface = np.asarray(sample_surface(soles[..., :2]), dtype=np.float64)
    clearance = soles[..., 2] - surface
    if clearance.shape[2] < 3:
        raise ContractError("transition sole footprint is undersampled")
    support_error = np.partition(
        np.abs(clearance), 2, axis=2
    )[:, :, 2][support]
    minimum_clearance = float(clearance.min())
    maximum_supported_error = (
        float(support_error.max())
        if len(support_error)
        else 0.0
    )
    boundary_joints = np.concatenate(
        (
            first_arrays["joint_position"][first_frame : first_frame + 1],
            joints,
        ),
        axis=0,
    )
    boundary_roots = np.concatenate(
        (
            first_arrays["root_position_world"][first_frame : first_frame + 1],
            roots,
        ),
        axis=0,
    )
    maximum_joint_step = float(
        np.abs(np.diff(boundary_joints, axis=0)).max()
    )
    maximum_root_step = float(
        np.linalg.norm(np.diff(boundary_roots, axis=0), axis=1).max()
    )
    if (
        minimum_clearance < -0.03
        or maximum_supported_error > 0.03
        or maximum_joint_step > 0.25
        or maximum_root_step > 0.05
    ):
        return None
    return {
        "minimum_sole_clearance_m": minimum_clearance,
        "maximum_supported_sole_error_m": maximum_supported_error,
        "maximum_joint_step_rad": maximum_joint_step,
        "maximum_root_step_m": maximum_root_step,
    }


def _certify_transition_with_modes(
    *,
    first: dict[str, object],
    second: dict[str, object],
    first_frame: int,
    second_frame: int,
    blend_frames: int,
    sole_kinematics: MujocoG1SoleKinematics,
    sample_surface,
) -> dict[str, object] | None:
    for blend_mode in ("contact", "pose"):
        metrics = _certify_transition(
            first=first,
            second=second,
            first_frame=first_frame,
            second_frame=second_frame,
            blend_frames=blend_frames,
            sole_kinematics=sole_kinematics,
            sample_surface=sample_surface,
            blend_mode=blend_mode,
        )
        if metrics is not None:
            return {**metrics, "blend_mode": blend_mode}
    return None


def _certify_composite(
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    *,
    sole_kinematics: MujocoG1SoleKinematics,
    sample_surface,
) -> dict[str, object]:
    soles = sole_kinematics.sole_points(
        arrays["joint_position"],
        arrays["root_position_world"],
        arrays["root_orientation_world_wxyz"],
    )
    surface = np.asarray(sample_surface(soles[..., :2]), dtype=np.float64)
    clearance = soles[..., 2] - surface
    if clearance.shape[2] < 3:
        raise ContractError("composite sole footprint is undersampled")
    stance_values = np.partition(
        np.abs(clearance), 2, axis=2
    )[:, :, 2][support]
    return {
        "frame_count": len(arrays["joint_position"]),
        "minimum_sole_clearance_m": float(clearance.min()),
        "maximum_supported_sole_error_m": (
            float(stance_values.max()) if len(stance_values) else math.inf
        ),
        "flight_frame_count": int((~support.any(axis=1)).sum()),
        "maximum_joint_step_rad": float(
            np.abs(np.diff(arrays["joint_position"], axis=0)).max()
        ),
        "maximum_root_step_m": float(
            np.linalg.norm(
                np.diff(arrays["root_position_world"], axis=0), axis=1
            ).max()
        ),
    }


def _certified_traversal_archive(
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    *,
    target_root_scene_xy: np.ndarray,
    alignment_yaw: float,
) -> dict[str, np.ndarray]:
    """Export the certified pose stream with its exact support contract."""

    frame_count = len(np.asarray(arrays["joint_position"]))
    owned_support = np.asarray(support)
    if (
        owned_support.shape != (frame_count, 2)
        or owned_support.dtype != np.bool_
    ):
        raise ContractError("composed support mask is invalid")
    archive = _matcher_archive(
        SimpleNamespace(
            joint_position=arrays["joint_position"],
            root_position_scene=arrays["root_position_world"],
            root_orientation_scene_wxyz=arrays[
                "root_orientation_world_wxyz"
            ],
        ),
        target_root_scene_xy=target_root_scene_xy,
        alignment_yaw=alignment_yaw,
    )
    archive["source_support_mask"] = np.array(
        owned_support, dtype=np.bool_, copy=True
    )
    return archive


def main() -> int:
    args = _parser().parse_args()
    path_start, path_stop, heading, path_length = _resolve_path(args)
    if (
        path_length <= args.segment_length_m
        or args.workers < 1
        or args.coarse_results < 2
        or args.placement_shortlist < 2
        or args.candidates_per_anchor < 1
        or not 2 <= args.blend_frames <= 20
    ):
        raise ContractError("root-path search options are invalid")
    root = args.source_dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    targets = [
        item
        for item in manifest["clips"]
        if item.get("logical_name") == args.target_scene
        or item.get("relative_motion_path") == args.target_scene
    ]
    if len(targets) != 1 or targets[0].get("kind") != "terrain":
        raise ContractError("root-path target scene must resolve exactly once")
    target = targets[0]
    origin, cell, height = _target_grid(root, target)

    def sample_surface(points):
        return _sample_height_grid(
            origin_xy=origin,
            cell_size_m=cell,
            height_z=height,
            points_xy=np.asarray(points, dtype=np.float64),
        )

    ordered_contract = (
        derive_ordered_terrain_levels(
            path_start_scene_xy=path_start,
            path_stop_scene_xy=path_stop,
            sample_surface=sample_surface,
        )
        if args.ordered_contact_levels
        else None
    )
    anchors = root_path_search_anchors(
        path_length_m=path_length,
        stride_m=float(args.search_stride_m),
        minimum_window_length_m=_minimum_search_window_length(
            float(args.segment_length_m)
        ),
    )
    lengths = tuple(
        min(float(args.segment_length_m), path_length - anchor)
        for anchor in anchors
    )
    phases = tuple(
        _query_phase(
            path_start=path_start,
            heading=heading,
            start_m=anchor,
            length_m=length,
            step_width_m=float(args.step_width_m),
            sample_surface=sample_surface,
            kind=(
                "interior"
                if ordered_contract is None
                else _ordered_phase_kind(
                    ordered_contract, anchor, anchor + length
                )
            ),
        )
        for anchor, length in zip(anchors, lengths)
    )
    queries = tuple(
        _phase_queries(
            phase=phase,
            step_width_m=float(args.step_width_m),
            sample_surface=sample_surface,
        )
        for phase in phases
    )
    descriptors = sorted(
        (
            item
            for item in manifest["clips"]
            if item.get("kind") == "terrain"
            and item.get("terrain", {}).get("kind") == "heightgrid"
        ),
        key=lambda item: item["logical_name"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if ordered_contract is not None:
        (args.output / "required-levels.json").write_text(
            json.dumps(
                {
                    "schema": "g1-ordered-terrain-level-contract/v1",
                    "path_start_scene_xy": list(
                        ordered_contract.path_start_scene_xy
                    ),
                    "path_heading_scene_xy": list(
                        ordered_contract.path_heading_scene_xy
                    ),
                    "path_length_m": ordered_contract.path_length_m,
                    "levels": [
                        asdict(level) for level in ordered_contract.levels
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    pool_cache = args.output / "coarse-pools.json"
    cache_contract = {
        "source_dataset": str(root),
        "target_scene": str(target["logical_name"]),
        "path_start_scene_xy": path_start.tolist(),
        "path_stop_scene_xy": path_stop.tolist(),
        "segment_length_m": float(args.segment_length_m),
        "search_stride_m": float(args.search_stride_m),
        "step_width_m": float(args.step_width_m),
        "coarse_results": int(args.coarse_results),
        "placement_shortlist": int(args.placement_shortlist),
        "ordered_contact_levels": bool(args.ordered_contact_levels),
        "phase_kinds": [phase.kind for phase in phases],
    }
    if pool_cache.exists():
        cached = json.loads(pool_cache.read_text("utf-8"))
        if (
            cached.get("contract") != cache_contract
            or cached.get("anchors") != list(anchors)
            or cached.get("lengths") != list(lengths)
            or len(cached.get("pools", [])) != len(anchors)
        ):
            raise ContractError("root-path coarse pool cache is incompatible")
        pools = cached["pools"]
        print(
            json.dumps({"loaded_coarse_pool_cache": str(pool_cache)}),
            flush=True,
        )
    else:
        jobs = (
            _scanner_job(root, descriptor, queries, lengths, phases)
            for descriptor in descriptors
        )
        pools = [[] for _ in anchors]
        pool_limit = max(
            int(args.coarse_results), 4 * int(args.placement_shortlist)
        )
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for clip_index, rows_by_anchor in enumerate(
                executor.map(_scan_one, jobs, chunksize=8), start=1
            ):
                for index, rows in enumerate(rows_by_anchor):
                    pools[index].extend(rows)
                if clip_index % 256 == 0:
                    for index in range(len(pools)):
                        if len(pools[index]) > pool_limit:
                            pools[index] = list(
                                _placement_shortlist_rows(
                                    pools[index],
                                    path_length_m=lengths[index],
                                    maximum_results=pool_limit,
                                )
                            )
                if clip_index % 1024 == 0:
                    print(
                        json.dumps(
                            {
                                "scanned_clips": clip_index,
                                "total_clips": len(descriptors),
                            }
                        ),
                        flush=True,
                    )
        pool_cache.write_text(
            json.dumps(
                {
                    "contract": cache_contract,
                    "anchors": list(anchors),
                    "lengths": list(lengths),
                    "pools": pools,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    by_name = {item["logical_name"]: item for item in descriptors}
    target_root_scene_xy, alignment_yaw = _target_alignment(root, target)
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    profiles_cache: dict[str, dict[str, np.ndarray]] = {}
    soles_cache: dict[str, np.ndarray] = {}
    candidates = []
    anchor_reports = []
    for anchor, length, phase, rows in zip(
        anchors, lengths, phases, pools
    ):
        qualified, rejected = _qualify_candidates(
            anchor_m=anchor,
            rows=rows,
            path_start=path_start,
            heading=heading,
            root=root,
            by_name=by_name,
            profiles_cache=profiles_cache,
            soles_cache=soles_cache,
            sole_kinematics=sole_kinematics,
            sample_surface=sample_surface,
            target_root_scene_xy=target_root_scene_xy,
            alignment_yaw=alignment_yaw,
            shortlist_count=int(args.placement_shortlist),
            candidate_count=int(args.candidates_per_anchor),
            segment_length_m=length,
            ordered_contract=ordered_contract,
            require_terminal_double_support=(
                _requires_terminal_settle(
                    anchor_m=anchor,
                    length_m=length,
                    path_length_m=path_length,
                )
            ),
        )
        candidates.extend(qualified)
        anchor_reports.append(
            {
                "anchor_m": anchor,
                "coarse_pool_count": len(rows),
                "qualified_count": len(qualified),
                "rejected_by_reason": rejected,
            }
        )
        print(json.dumps(anchor_reports[-1], sort_keys=True), flush=True)

    graph_candidates = tuple(
        RootPathMotionCandidate(
            candidate_id=item["candidate_id"],
            covered_start_m=max(0.0, float(item["interval_start_m"])),
            covered_stop_m=float(item["interval_stop_m"]),
            placement_cost=float(item["placement_cost"]),
            frame_count=len(item["scene"]["joint_position"]),
        )
        for item in candidates
    )
    transition_details = {}
    transition_certification = {}
    graph_transitions = []
    for first in candidates:
        for second in candidates:
            certified = _best_certified_transition(
                first,
                second,
                certify=lambda first_frame, second_frame: (
                    _certify_transition_with_modes(
                        first=first,
                        second=second,
                        first_frame=first_frame,
                        second_frame=second_frame,
                        blend_frames=int(args.blend_frames),
                        sole_kinematics=sole_kinematics,
                        sample_surface=sample_surface,
                    )
                ),
            )
            if certified is None:
                continue
            result, certification = certified
            key = (first["candidate_id"], second["candidate_id"])
            transition_details[key] = (
                *result,
                str(certification["blend_mode"]),
            )
            transition_certification[key] = certification
            graph_transitions.append(
                RootPathMotionTransition(
                    key[0],
                    key[1],
                    result[0]
                    + float(certification["maximum_joint_step_rad"])
                    + 4.0 * float(certification["maximum_root_step_m"]),
                    result[1],
                    result[2],
                )
            )
    diagnostic = {
        "schema": "g1-root-path-motion-search-diagnostic/v1",
        "anchors": anchor_reports,
        "candidate_count": len(candidates),
        "transition_count": len(graph_transitions),
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "anchor_m": item["anchor_m"],
                "interval_m": [
                    item["interval_start_m"],
                    item["interval_stop_m"],
                ],
                "placement_cost": item["placement_cost"],
                "source_clip": item["row"]["source_clip"],
                "source_frames": [
                    item["row"]["start_frame"],
                    item["row"]["stop_frame"],
                ],
                "ordered_levels": (
                    None
                    if item["ordered_levels"] is None
                    else list(item["ordered_levels"])
                ),
                "touchdown_feet": (
                    None
                    if item["touchdown_feet"] is None
                    else list(item["touchdown_feet"])
                ),
                "touchdown_frames": (
                    None
                    if item["touchdown_frames"] is None
                    else list(item["touchdown_frames"])
                ),
                "terminal_double_support": bool(
                    np.asarray(item["support"], dtype=np.bool_)[-1].all()
                ),
            }
            for item in candidates
        ],
        "transitions": [
            {
                "first_candidate_id": item.first_candidate_id,
                "second_candidate_id": item.second_candidate_id,
                "first_frame": item.first_frame,
                "second_frame": item.second_frame,
                "transition_cost": item.transition_cost,
            }
            for item in graph_transitions
        ],
    }
    (args.output / "diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if ordered_contract is None:
        selected = select_root_path_motion_chain(
            candidates=graph_candidates,
            transitions=tuple(graph_transitions),
            path_length_m=path_length,
            maximum_coverage_gap_m=0.15,
            minimum_segment_frames=int(args.blend_frames),
            fragmentation_cost=0.05,
        )
    else:
        graph_by_id = {
            candidate.candidate_id: candidate
            for candidate in graph_candidates
        }
        ordered_candidates = tuple(
            OrderedLevelMotionCandidate(
                candidate=graph_by_id[str(item["candidate_id"])],
                level_indices=tuple(item["ordered_levels"]),
                touchdown_feet=tuple(item["touchdown_feet"]),
                touchdown_frames=tuple(item["touchdown_frames"]),
                terminal_double_support=bool(
                    np.asarray(item["support"], dtype=np.bool_)[-1].all()
                ),
            )
            for item in candidates
        )
        selected = select_ordered_level_motion_chain(
            candidates=ordered_candidates,
            transitions=tuple(graph_transitions),
            path_length_m=path_length,
            required_level_count=len(ordered_contract.levels),
            maximum_coverage_gap_m=0.15,
            minimum_segment_frames=int(args.blend_frames),
            fragmentation_cost=0.05,
        )
    by_id = {item["candidate_id"]: item for item in candidates}
    chain = tuple(by_id[item] for item in selected.candidate_ids)
    arrays, support, transition_report = _compose_chain(
        chain,
        transition_details,
        blend_frames=int(args.blend_frames),
        sole_kinematics=sole_kinematics,
    )
    metrics = _certify_composite(
        arrays,
        support,
        sole_kinematics=sole_kinematics,
        sample_surface=sample_surface,
    )
    if (
        metrics["minimum_sole_clearance_m"] < -0.03
        or metrics["maximum_supported_sole_error_m"] > 0.03
    ):
        raise ContractError(
            "selected root-path chain failed final terrain certification"
        )
    matcher = _certified_traversal_archive(
        arrays,
        support,
        target_root_scene_xy=target_root_scene_xy,
        alignment_yaw=alignment_yaw,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "traversal.npz", **matcher)
    report = {
        "schema": "g1-root-path-motion-search/v1",
        "target_scene": target["logical_name"],
        "path": {
            "start_scene_xy": path_start.tolist(),
            "stop_scene_xy": path_stop.tolist(),
            "length_m": path_length,
        },
        "required_levels": (
            None
            if ordered_contract is None
            else [asdict(level) for level in ordered_contract.levels]
        ),
        "anchors": anchor_reports,
        "candidate_count": len(candidates),
        "transition_count": len(graph_transitions),
        "selected_candidate_ids": list(selected.candidate_ids),
        "selected_total_cost": selected.total_cost,
        "selected_sources": [
            {
                "candidate_id": item["candidate_id"],
                "source_clip": item["row"]["source_clip"],
                "source_start_frame": item["row"]["start_frame"],
                "source_stop_frame": item["row"]["stop_frame"],
                "anchor_m": item["anchor_m"],
                "interval_m": [
                    item["interval_start_m"],
                    item["interval_stop_m"],
                ],
                "placement_metrics": item["metrics"],
                "ordered_levels": (
                    None
                    if item["ordered_levels"] is None
                    else list(item["ordered_levels"])
                ),
                "touchdown_feet": (
                    None
                    if item["touchdown_feet"] is None
                    else list(item["touchdown_feet"])
                ),
            }
            for item in chain
        ],
        "transitions": transition_report,
        "transition_certification": [
            {
                "first_candidate_id": first,
                "second_candidate_id": second,
                **transition_certification[(first, second)],
            }
            for first, second in zip(
                selected.candidate_ids[:-1],
                selected.candidate_ids[1:],
            )
        ],
        "metrics": metrics,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _minimal_dataset(
        source_root=root,
        manifest=manifest,
        target=target,
        output=args.output / "dataset",
    )
    print(json.dumps(report["metrics"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
