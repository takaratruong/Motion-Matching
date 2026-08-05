#!/usr/bin/env python3
"""Search all raw GRAIL motions for rigid coverage of one directed path."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import (
    ANKLE_ORIGIN_SOLE_M,
    STANCE_CLEARANCE_TOLERANCE_M,
    STANCE_VERTICAL_SPEED_MAX_MPS,
)
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    nominal_footprint_path,
)
from mm_sonic.torch_path_motion_placement import (
    PathContactSignature,
    PlacementRejected,
    RawContactEvent,
    RawMotionWindow,
    contact_signature_cost,
    extract_raw_motion_windows,
    path_contact_signature,
    place_raw_window_on_path,
)

from resources.run_g1_grail_contact_inventory import _sample_height_grid


_FEET = (18, 19)
_STRIDES_M = (0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52)
_SPEEDS_MPS = (0.35, 0.50, 0.65)


def _progress_coverage_error(progress_m: float, path_length_m: float) -> float:
    progress = float(progress_m)
    length = float(path_length_m)
    if (
        not math.isfinite(progress)
        or not math.isfinite(length)
        or progress <= 0.0
        or length <= 0.0
    ):
        raise ContractError("path placement progress is invalid")
    return max(0.0, length - progress) + 0.1 * max(
        0.0, progress - length
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--target-scene", required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument(
        "--path-start", type=float, nargs=2, metavar=("X", "Y"), required=True
    )
    parser.add_argument(
        "--path-stop", type=float, nargs=2, metavar=("X", "Y"), required=True
    )
    parser.add_argument("--step-width-m", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--coarse-results", type=int, default=400)
    parser.add_argument("--placement-shortlist", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _rank_coarse_rows(
    rows: object, *, maximum_results: int
) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(rows, (list, tuple))
        or type(maximum_results) is not int
        or maximum_results < 1
        or any(
            not isinstance(row, dict)
            or not math.isfinite(float(row.get("cost", math.nan)))
            or not isinstance(row.get("source_clip"), str)
            or type(row.get("start_frame")) is not int
            for row in rows
        )
    ):
        raise ContractError("path placement coarse rows are invalid")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                float(row["cost"]),
                str(row["source_clip"]),
                int(row["start_frame"]),
                int(row.get("stop_frame", 0)),
            ),
        )[:maximum_results]
    )


def _retain_event_count_diversity(
    rows: object, *, per_count: int, target_progress_m: float | None = None
) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(rows, (list, tuple))
        or type(per_count) is not int
        or per_count < 1
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("events"), (list, tuple))
            or len(row["events"]) < 2
            for row in rows
        )
    ):
        raise ContractError("path placement diverse rows are invalid")
    if target_progress_m is not None and (
        not math.isfinite(float(target_progress_m))
        or target_progress_m <= 0.0
        or any(
            not math.isfinite(float(row.get("forward_progress_m", math.nan)))
            for row in rows
        )
    ):
        raise ContractError("path placement target progress is invalid")
    by_count: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_count.setdefault(len(row["events"]), []).append(row)
    retained = []
    for count in sorted(by_count):
        group = by_count[count]
        selected = list(
            _rank_coarse_rows(group, maximum_results=per_count)
        )
        if target_progress_m is not None:
            selected.extend(
                sorted(
                    group,
                    key=lambda row: (
                        abs(
                            _progress_coverage_error(
                                float(row["forward_progress_m"]),
                                target_progress_m,
                            )
                        ),
                        float(row["cost"]),
                        str(row["source_clip"]),
                        int(row["start_frame"]),
                        int(row.get("stop_frame", 0)),
                    ),
                )[:per_count]
            )
        unique = {
            (
                str(row["source_clip"]),
                int(row["start_frame"]),
                int(row.get("stop_frame", 0)),
                len(row["events"]),
            ): row
            for row in selected
        }
        retained.extend(unique.values())
    return _rank_coarse_rows(
        retained, maximum_results=max(1, len(retained))
    )


def _placement_shortlist_rows(
    rows: object, *, path_length_m: float, maximum_results: int
) -> tuple[dict[str, object], ...]:
    ranked = _rank_coarse_rows(rows, maximum_results=max(1, len(rows)))
    if (
        not math.isfinite(float(path_length_m))
        or path_length_m <= 0.0
        or type(maximum_results) is not int
        or maximum_results < 2
        or any(
            not math.isfinite(float(row.get("forward_progress_m", math.nan)))
            for row in ranked
        )
    ):
        raise ContractError("path placement shortlist input is invalid")
    coverage_ranked = sorted(
        ranked,
        key=lambda row: (
            _progress_coverage_error(
                float(row["forward_progress_m"]), path_length_m
            ),
            float(row["cost"]),
            str(row["source_clip"]),
            int(row["start_frame"]),
            int(row.get("stop_frame", 0)),
        ),
    )
    blended_ranked = sorted(
        ranked,
        key=lambda row: (
            float(row["cost"])
            + 10.0
            * _progress_coverage_error(
                float(row["forward_progress_m"]), path_length_m
            ),
            float(row["cost"]),
            str(row["source_clip"]),
            int(row["start_frame"]),
            int(row.get("stop_frame", 0)),
        ),
    )
    first = (maximum_results + 2) // 3
    second = (maximum_results + 1) // 3
    third = maximum_results - first - second
    selected = (
        list(ranked[:first])
        + coverage_ranked[:second]
        + blended_ranked[:third]
    )
    unique = {
        (
            str(row["source_clip"]),
            int(row["start_frame"]),
            int(row.get("stop_frame", 0)),
        ): row
        for row in selected
    }
    for ordering in (blended_ranked, ranked, coverage_ranked):
        for row in ordering:
            key = (
                str(row["source_clip"]),
                int(row["start_frame"]),
                int(row.get("stop_frame", 0)),
            )
            unique.setdefault(key, row)
            if len(unique) >= maximum_results:
                break
        if len(unique) >= maximum_results:
            break
    return tuple(
        sorted(
            list(unique.values())[:maximum_results],
            key=lambda row: (
                float(row["cost"]),
                str(row["source_clip"]),
                int(row["start_frame"]),
                int(row.get("stop_frame", 0)),
            ),
        )
    )


def _uncovered_intervals(
    *,
    path_length_m: float,
    covered_intervals: object,
) -> tuple[tuple[float, float], ...]:
    intervals = tuple(
        (float(start), float(stop)) for start, stop in covered_intervals
    )
    if (
        not math.isfinite(float(path_length_m))
        or path_length_m <= 0.0
        or any(
            not math.isfinite(start)
            or not math.isfinite(stop)
            or not 0.0 <= start < stop <= path_length_m + 1.0e-6
            for start, stop in intervals
        )
    ):
        raise ContractError("path placement coverage input is invalid")
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1] + 1.0e-6:
            merged.append([start, min(stop, path_length_m)])
        else:
            merged[-1][1] = max(merged[-1][1], min(stop, path_length_m))
    output = []
    cursor = 0.0
    for start, stop in merged:
        if start > cursor + 1.0e-6:
            output.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < path_length_m - 1.0e-6:
        output.append((cursor, float(path_length_m)))
    return tuple(output)


def _descriptor_transform(
    descriptor: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, float]:
    values = descriptor["terrain"]["motion_to_terrain_xy_yaw"]
    tx, ty, yaw = (float(value) for value in values)
    rotation = np.array(
        (
            (math.cos(yaw), -math.sin(yaw)),
            (math.sin(yaw), math.cos(yaw)),
        ),
        dtype=np.float64,
    )
    return rotation, np.array((tx, ty), dtype=np.float64), yaw


def _source_profiles(
    root: Path, descriptor: dict[str, object]
) -> dict[str, np.ndarray]:
    with np.load(
        root / descriptor["relative_motion_path"], allow_pickle=False
    ) as archive:
        output = {
            "joint": np.asarray(archive["joint_pos"], dtype=np.float64),
            "root": np.asarray(archive["body_pos_w"][:, 0], dtype=np.float64),
            "quaternion": np.asarray(
                archive["body_quat_w"][:, 0], dtype=np.float64
            ),
            "feet": np.asarray(
                archive["body_pos_w"][:, _FEET], dtype=np.float64
            ),
            "foot_velocity": np.asarray(
                archive["body_lin_vel_w"][:, _FEET], dtype=np.float64
            ),
        }
    with np.load(
        root / descriptor["terrain"]["path"], allow_pickle=False
    ) as archive:
        origin = np.asarray(archive["origin_xy"], dtype=np.float64)
        cell = float(np.asarray(archive["cell_size_m"]).reshape(-1)[0])
        height = np.asarray(archive["height_z"], dtype=np.float64)
    rotation, translation, _ = _descriptor_transform(descriptor)
    feet_scene = output["feet"][..., :2] @ rotation.T + translation
    surface = _sample_height_grid(
        origin_xy=origin,
        cell_size_m=cell,
        height_z=height,
        points_xy=feet_scene,
    )
    support = (
        np.abs(
            output["feet"][..., 2]
            - surface
            - float(ANKLE_ORIGIN_SOLE_M)
        )
        <= float(STANCE_CLEARANCE_TOLERANCE_M)
    ) & (
        np.abs(output["foot_velocity"][..., 2])
        <= float(STANCE_VERTICAL_SPEED_MAX_MPS)
    )
    output["surface"] = surface
    output["support"] = np.asarray(support, dtype=np.bool_)
    return output


def _window_to_row(
    window: RawMotionWindow, cost: float
) -> dict[str, object]:
    return {
        "cost": float(cost),
        "source_clip": window.source_clip,
        "start_frame": window.start_frame,
        "stop_frame": window.stop_frame,
        "root_start_world_xy": list(window.root_start_world_xy),
        "forward_progress_m": window.forward_progress_m,
        "heading_error_rad": window.heading_error_rad,
        "events": [asdict(event) for event in window.events],
    }


def _row_to_window(row: dict[str, object]) -> RawMotionWindow:
    return RawMotionWindow(
        source_clip=str(row["source_clip"]),
        start_frame=int(row["start_frame"]),
        stop_frame=int(row["stop_frame"]),
        events=tuple(RawContactEvent(**event) for event in row["events"]),
        root_start_world_xy=tuple(row["root_start_world_xy"]),
        forward_progress_m=float(row["forward_progress_m"]),
        heading_error_rad=float(row["heading_error_rad"]),
    )


def _scan_one(
    arguments: tuple[
        str,
        dict[str, object],
        tuple[PathContactSignature, ...],
        float,
    ]
) -> tuple[dict[str, object], ...]:
    root_text, descriptor, queries, path_length = arguments
    try:
        profiles = _source_profiles(Path(root_text), descriptor)
        counts = tuple(len(query.foot_order) for query in queries)
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
        rows = []
        for window in windows:
            compatible = tuple(
                query
                for query in queries
                if len(query.foot_order) == len(window.events)
            )
            if not compatible:
                continue
            cost = min(
                contact_signature_cost(query, window) for query in compatible
            )
            if math.isfinite(cost):
                rows.append(_window_to_row(window, cost))
        return (
            _retain_event_count_diversity(
                rows, per_count=2, target_progress_m=path_length
            )
            if rows
            else ()
        )
    except (ContractError, KeyError, OSError, ValueError):
        return ()


def _target_grid(
    root: Path, descriptor: dict[str, object]
) -> tuple[np.ndarray, float, np.ndarray]:
    with np.load(
        root / descriptor["terrain"]["path"], allow_pickle=False
    ) as archive:
        return (
            np.asarray(archive["origin_xy"], dtype=np.float64),
            float(np.asarray(archive["cell_size_m"]).reshape(-1)[0]),
            np.asarray(archive["height_z"], dtype=np.float64),
        )


def _queries(
    *,
    path_start: np.ndarray,
    path_heading: np.ndarray,
    path_length: float,
    step_width_m: float,
    sample_surface,
) -> tuple[PathContactSignature, ...]:
    lateral = np.array((-path_heading[1], path_heading[0]))
    output = []
    for first_foot in (0, 1):
        offsets = np.zeros((2, 2), dtype=np.float32)
        offsets[:, 1] = (0.5 * step_width_m, -0.5 * step_width_m)
        offsets[first_foot, 0] = -0.01
        start_feet = (
            path_start[None, :]
            + offsets[:, :1] * path_heading[None, :]
            + offsets[:, 1:] * lateral[None, :]
        )
        for stride in _STRIDES_M:
            if math.floor(path_length / stride) < 4:
                continue
            for speed in _SPEEDS_MPS:
                path = nominal_footprint_path(
                    ConstantHeadingRequest(
                        start_foot_scene_xy=torch.tensor(
                            start_feet, dtype=torch.float32
                        ),
                        start_support=torch.tensor((True, True)),
                        heading_scene_xy=torch.tensor(
                            path_heading, dtype=torch.float32
                        ),
                        distance_m=path_length,
                        speed_mps=speed,
                        stride_m=stride,
                        step_width_m=step_width_m,
                        frames_per_second=50.0,
                    )
                )
                output.append(
                    path_contact_signature(
                        path=path, sample_surface=sample_surface
                    )
                )
    return tuple(output)


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _matcher_archive(
    placed, *, target_root_scene_xy: np.ndarray, alignment_yaw: float
) -> dict[str, np.ndarray]:
    relative = placed.root_position_scene[:, :2] - target_root_scene_xy
    cosine, sine = math.cos(alignment_yaw), math.sin(alignment_yaw)
    matcher_xy = np.stack(
        (
            cosine * relative[:, 0] + sine * relative[:, 1],
            -sine * relative[:, 0] + cosine * relative[:, 1],
        ),
        axis=-1,
    )
    roots = placed.root_position_scene.copy()
    roots[:, :2] = matcher_xy
    inverse = np.array(
        (
            math.cos(alignment_yaw / 2.0),
            0.0,
            0.0,
            -math.sin(alignment_yaw / 2.0),
        )
    )
    quaternion = _quaternion_multiply(
        inverse, placed.root_orientation_scene_wxyz
    )
    return {
        "joint_position": np.asarray(placed.joint_position),
        "root_position_world": roots,
        "root_orientation_world_wxyz": quaternion,
    }


def _minimal_dataset(
    *,
    source_root: Path,
    manifest: dict[str, object],
    target: dict[str, object],
    output: Path,
) -> None:
    flat = next(
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("relative_motion_path") == "flat/motion.npz"
    )
    if output.exists():
        shutil.rmtree(output)
    for descriptor in (flat, target):
        paths = [descriptor["relative_motion_path"]]
        if descriptor["terrain"].get("kind") == "heightgrid":
            paths.append(descriptor["terrain"]["path"])
        for relative in paths:
            source = source_root / relative
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
    subset = {
        key: manifest[key]
        for key in ("schema", "layout", "output_fps", "g1_xml_sha256")
    }
    subset.update(
        {
            "clips": [flat, target],
            "accepted_clips": [flat, target],
            "rejected_candidates": [],
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    path_start = np.asarray(args.path_start, dtype=np.float64)
    path_stop = np.asarray(args.path_stop, dtype=np.float64)
    displacement = path_stop - path_start
    path_length = float(np.linalg.norm(displacement))
    if (
        path_length <= 0.40
        or args.step_width_m <= 0.0
        or args.workers < 1
        or args.coarse_results < 1
        or args.placement_shortlist < 1
    ):
        raise ContractError("path motion placement options are invalid")
    heading = displacement / path_length
    root = args.source_dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    matches = [
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("logical_name") == args.target_scene
        or descriptor.get("relative_motion_path") == args.target_scene
    ]
    if len(matches) != 1 or matches[0].get("kind") != "terrain":
        raise ContractError("target path scene must resolve exactly once")
    target = matches[0]
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
        return torch.tensor(values, dtype=points.dtype, device=points.device)

    queries = _queries(
        path_start=path_start,
        path_heading=heading,
        path_length=path_length,
        step_width_m=float(args.step_width_m),
        sample_surface=sample_torch,
    )
    descriptors = [
        descriptor
        for descriptor in manifest["clips"]
        if descriptor.get("kind") == "terrain"
        and descriptor.get("terrain", {}).get("kind") == "heightgrid"
    ]
    descriptors.sort(key=lambda item: item["logical_name"])
    jobs = (
        (str(root), descriptor, queries, path_length)
        for descriptor in descriptors
    )
    coarse = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for rows in executor.map(_scan_one, jobs, chunksize=8):
            coarse.extend(rows)
    ranked = _rank_coarse_rows(
        coarse, maximum_results=args.coarse_results
    )
    shortlist = _placement_shortlist_rows(
        coarse,
        path_length_m=path_length,
        maximum_results=args.placement_shortlist,
    )
    by_name = {
        descriptor["logical_name"]: descriptor for descriptor in descriptors
    }
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    accepted = []
    rejected: Counter[str] = Counter()
    for row in shortlist:
        descriptor = by_name[str(row["source_clip"])]
        try:
            profiles = _source_profiles(root, descriptor)
            window = _row_to_window(dict(row))
            start, stop = window.start_frame, window.stop_frame
            soles = sole_kinematics.sole_points(
                profiles["joint"][start:stop],
                profiles["root"][start:stop],
                profiles["quaternion"][start:stop],
            )
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
                sole_position_world=soles,
                support_mask=profiles["support"][start:stop],
                path_start_scene_xy=path_start,
                path_heading_scene_xy=heading,
                sample_surface=sample_numpy,
            )
            if not np.array_equal(
                placed.joint_position, profiles["joint"][start:stop]
            ):
                raise ContractError("path placement edited source joints")
            uncovered = max(
                0.0, path_length - placed.metrics.covered_stop_m
            )
            accepted.append(
                (
                    uncovered,
                    float(row["cost"]),
                    str(row["source_clip"]),
                    start,
                    placed,
                    dict(row),
                )
            )
        except PlacementRejected as error:
            rejected[error.reason] += 1
        except (ContractError, KeyError, OSError, ValueError):
            rejected["invalid-source"] += 1
    if not accepted:
        raise ContractError(
            "no certified raw path placement: "
            + json.dumps(dict(sorted(rejected.items())))
        )
    accepted.sort(key=lambda item: item[:4])
    _, _, _, source_start, best, best_row = accepted[0]
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
    target_root_scene = target_root[:2] @ rotation.T + translation
    alignment_yaw = _yaw_wxyz(target_quaternion) + terrain_yaw
    archive = _matcher_archive(
        best,
        target_root_scene_xy=target_root_scene,
        alignment_yaw=alignment_yaw,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "traversal.npz", **archive)
    gaps = _uncovered_intervals(
        path_length_m=path_length,
        covered_intervals=(
            (
                best.metrics.covered_start_m,
                min(path_length, best.metrics.covered_stop_m),
            ),
        ),
    )
    evidence = {
        "schema": "g1-path-motion-placement/v1",
        "query": {
            "target_scene": target["logical_name"],
            "path_start_scene_xy": path_start.tolist(),
            "path_stop_scene_xy": path_stop.tolist(),
            "path_length_m": path_length,
            "step_width_m": float(args.step_width_m),
            "query_signature_count": len(queries),
        },
        "corpus": {
            "scanned_clip_count": len(descriptors),
            "coarse_window_count": len(coarse),
            "coarse_retained_count": len(ranked),
        },
        "selected": {
            **best_row,
            "source_start_frame": source_start,
            "yaw_scene_rad": best.yaw_scene_rad,
            "translation_scene_xyz": list(best.translation_scene_xyz),
            "metrics": asdict(best.metrics),
            "joint_positions_edited": False,
        },
        "certified_count": len(accepted),
        "rejected_by_reason": dict(sorted(rejected.items())),
    }
    coverage = {
        "schema": "g1-path-motion-coverage/v1",
        "path_length_m": path_length,
        "covered_intervals_m": [
            [
                best.metrics.covered_start_m,
                min(path_length, best.metrics.covered_stop_m),
            ]
        ],
        "uncovered_intervals_m": [list(item) for item in gaps],
        "fully_covered": not gaps,
    }
    (args.output / "placements.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n",
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
                "selected": best_row["source_clip"],
                "frames": [best_row["start_frame"], best_row["stop_frame"]],
                "metrics": asdict(best.metrics),
                "uncovered": coverage["uncovered_intervals_m"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
