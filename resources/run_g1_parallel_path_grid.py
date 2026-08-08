#!/usr/bin/env python3
"""Run the same root-path motion search over a heading-local path grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_path_motion_grid import (
    ParallelPath,
    StaircasePathContract,
    build_staircase_grid_playlist,
    classify_staircase_path,
    parallel_path_grid,
    staircase_parallel_path_grid,
)
from resources.run_g1_grail_contact_inventory import _sample_height_grid
from resources.run_g1_path_motion_placement import _target_grid


def child_environment(
    *, existing: dict[str, str], repo_root: Path
) -> dict[str, str]:
    output = dict(existing)
    inherited = output.get("PYTHONPATH", "")
    entries = [str(repo_root)]
    if inherited:
        entries.append(inherited)
    output["PYTHONPATH"] = os.pathsep.join(entries)
    return output


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser()
    output.add_argument("--source-dataset", type=Path, required=True)
    output.add_argument("--target-scene", required=True)
    output.add_argument("--g1-xml", type=Path, required=True)
    output.add_argument(
        "--center-start", type=float, nargs=2, required=True,
        metavar=("X", "Y"),
    )
    output.add_argument("--heading-degrees", type=float, required=True)
    output.add_argument("--path-length-m", type=float, required=True)
    output.add_argument(
        "--minimum-lateral-offset-m", type=float
    )
    output.add_argument(
        "--maximum-lateral-offset-m", type=float
    )
    output.add_argument("--spacing-m", type=float, required=True)
    output.add_argument("--segment-length-m", type=float, default=0.90)
    output.add_argument("--search-stride-m", type=float, default=0.20)
    output.add_argument("--step-width-m", type=float, default=0.20)
    output.add_argument("--workers-per-path", type=int, default=16)
    output.add_argument("--concurrent-paths", type=int, default=2)
    output.add_argument("--coarse-results", type=int, default=2400)
    output.add_argument("--placement-shortlist", type=int, default=600)
    output.add_argument("--candidates-per-anchor", type=int, default=24)
    output.add_argument("--blend-frames", type=int, default=4)
    output.add_argument("--ordered-contact-levels", action="store_true")
    output.add_argument(
        "--validation-config",
        type=Path,
        default=Path(
            "sonic/configs/experiments/"
            "torch_grail_raw_horizontal_preview.json"
        ),
    )
    output.add_argument(
        "--maximum-unsupported-frames", type=int, default=50
    )
    output.add_argument("--output", type=Path, required=True)
    return output


def searchable_contracts(
    contracts: tuple[object, ...],
) -> tuple[object, ...]:
    if not isinstance(contracts, tuple):
        raise ContractError("staircase path contracts are invalid")
    return tuple(
        item
        for item in contracts
        if getattr(item, "classification", None)
        == "staircase_intersecting"
    )


def elevated_grid_cell_centers(
    origin: object,
    cell: float,
    height: object,
    *,
    threshold_m: float = 0.05,
) -> np.ndarray:
    origin_array = np.asarray(origin, dtype=np.float64)
    height_array = np.asarray(height, dtype=np.float64)
    if (
        origin_array.shape != (2,)
        or height_array.ndim != 2
        or min(height_array.shape) < 2
        or not np.isfinite(origin_array).all()
        or not np.isfinite(height_array).all()
        or not math.isfinite(float(cell))
        or float(cell) <= 0.0
        or not math.isfinite(float(threshold_m))
        or float(threshold_m) <= 0.0
    ):
        raise ContractError("target staircase grid is invalid")
    ground = float(np.min(height_array))
    rows, columns = np.nonzero(
        height_array > ground + float(threshold_m)
    )
    if not len(rows):
        raise ContractError("target terrain has no elevated staircase cells")
    return np.column_stack(
        (
            origin_array[0] + (columns + 0.5) * float(cell),
            origin_array[1] + (rows + 0.5) * float(cell),
        )
    )


def classify_search_failure(
    *, log_text: str, diagnostic: object
) -> str:
    if not isinstance(log_text, str):
        raise ContractError("root search failure evidence is invalid")
    phase = (
        diagnostic.get("first_missing_phase")
        if isinstance(diagnostic, dict)
        else None
    )
    if phase in ("mount", "interior", "dismount"):
        return {
            "mount": "no_mount_motion",
            "interior": "no_elevated_traversal_motion",
            "dismount": "no_dismount_motion",
        }[phase]
    lowered = log_text.lower()
    if "height" in lowered and "corpus" in lowered:
        return "height_out_of_corpus_range"
    if "footprint" in lowered:
        return "no_terrain_footprint"
    return "no_contact_compatible_chain"


def route_contract_record(
    contract: StaircasePathContract,
) -> dict[str, object]:
    if not isinstance(contract, StaircasePathContract):
        raise ContractError("route contract is invalid")
    return {
        "path_id": contract.path.path_id,
        "classification": contract.classification,
        "start_scene_xy": list(contract.path.start_scene_xy),
        "stop_scene_xy": list(contract.path.stop_scene_xy),
        "ordered_surface_heights_m": list(
            contract.ordered_surface_heights_m
        ),
        "elevated_intervals_m": [
            list(interval) for interval in contract.elevated_intervals_m
        ],
    }


def root_search_command(
    *,
    python: Path,
    runner: Path,
    source_dataset: Path,
    target_scene: str,
    g1_xml: Path,
    start_scene_xy: tuple[float, float],
    stop_scene_xy: tuple[float, float],
    output: Path,
    segment_length_m: float,
    search_stride_m: float,
    step_width_m: float,
    workers: int,
    coarse_results: int,
    placement_shortlist: int,
    candidates_per_anchor: int,
    blend_frames: int,
    ordered_contact_levels: bool,
) -> list[str]:
    command = [
        str(python),
        "-B",
        str(runner),
        "--source-dataset", str(source_dataset),
        "--target-scene", target_scene,
        "--g1-xml", str(g1_xml),
        "--path-start", *(str(float(value)) for value in start_scene_xy),
        "--path-stop", *(str(float(value)) for value in stop_scene_xy),
        "--segment-length-m", str(float(segment_length_m)),
        "--search-stride-m", str(float(search_stride_m)),
        "--step-width-m", str(float(step_width_m)),
        "--workers", str(int(workers)),
        "--coarse-results", str(int(coarse_results)),
        "--placement-shortlist", str(int(placement_shortlist)),
        "--candidates-per-anchor", str(int(candidates_per_anchor)),
        "--blend-frames", str(int(blend_frames)),
        "--output", str(output),
    ]
    if ordered_contact_levels:
        command.append("--ordered-contact-levels")
    return command


def independent_validation_command(
    *,
    python: Path,
    runner: Path,
    traversal: Path,
    target_dataset: Path,
    config: Path,
    g1_xml: Path,
    route_contract: Path,
    expected_heading_degrees: float,
    output: Path,
    maximum_unsupported_frames: int,
) -> list[str]:
    if maximum_unsupported_frames < 0:
        raise ContractError("independent validation allowance is invalid")
    return [
        str(python),
        "-B",
        str(runner),
        "--input", str(traversal),
        "--target-dataset", str(target_dataset),
        "--config", str(config),
        "--g1-xml", str(g1_xml),
        "--route-contract", str(route_contract),
        "--expected-heading-degrees", str(
            float(expected_heading_degrees)
        ),
        "--maximum-unsupported-frames", str(
            int(maximum_unsupported_frames)
        ),
        "--output", str(output),
    ]


def grid_summary(
    *, geometry: dict[str, object], results: tuple[dict[str, object], ...]
) -> dict[str, object]:
    if (
        not isinstance(geometry, dict)
        or not isinstance(results, tuple)
        or any(not isinstance(item, dict) for item in results)
    ):
        raise ContractError("parallel path grid summary is invalid")
    statuses = [str(item.get("status", "")) for item in results]
    if any(
        status not in ("excluded", "validated", "failed")
        for status in statuses
    ):
        raise ContractError("parallel path grid result status is invalid")
    return {
        "schema": "g1-staircase-parallel-path-grid/v2",
        "geometry": dict(geometry),
        "requested_path_count": len(results),
        "staircase_path_count": (
            statuses.count("validated") + statuses.count("failed")
        ),
        "excluded_path_count": statuses.count("excluded"),
        "validated_path_count": statuses.count("validated"),
        "failed_path_count": statuses.count("failed"),
        "results": list(results),
    }


def merge_grid_results(
    *,
    paths: tuple[ParallelPath, ...],
    result_groups: tuple[tuple[dict[str, object], ...], ...],
) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(paths, tuple)
        or not paths
        or not isinstance(result_groups, tuple)
        or any(not isinstance(group, tuple) for group in result_groups)
    ):
        raise ContractError("parallel path result merge is invalid")
    by_index: dict[int, dict[str, object]] = {}
    for group in result_groups:
        for item in group:
            if not isinstance(item, dict):
                raise ContractError("parallel path result merge is invalid")
            try:
                offset = float(item["lateral_offset_m"])
            except (KeyError, TypeError, ValueError) as error:
                raise ContractError(
                    "parallel path result offset is invalid"
                ) from error
            matches = [
                path
                for path in paths
                if abs(path.lateral_offset_m - offset) <= 1.0e-9
            ]
            if len(matches) != 1 or matches[0].index in by_index:
                raise ContractError(
                    "parallel path result is duplicate or out of grid"
                )
            path = matches[0]
            by_index[path.index] = {
                **item,
                "index": path.index,
                "path_id": path.path_id,
                "lateral_offset_m": path.lateral_offset_m,
                "start_scene_xy": list(path.start_scene_xy),
                "stop_scene_xy": list(path.stop_scene_xy),
            }
    if set(by_index) != set(range(len(paths))):
        raise ContractError("parallel path result merge omitted a path")
    return tuple(by_index[index] for index in range(len(paths)))


def playlist_entries(
    *,
    paths: tuple[ParallelPath, ...],
    results: tuple[dict[str, object], ...],
) -> tuple[tuple[ParallelPath, dict[str, np.ndarray], dict[str, object]], ...]:
    if (
        not isinstance(paths, tuple)
        or not paths
        or not isinstance(results, tuple)
        or any(not isinstance(item, dict) for item in results)
    ):
        raise ContractError("staircase playlist results are invalid")
    by_id = {path.path_id: path for path in paths}
    output = []
    for result in results:
        if result.get("status") != "validated":
            continue
        path = by_id.get(str(result.get("path_id", "")))
        if (
            path is None
            or result.get("independently_validated") is not True
        ):
            raise ContractError(
                "validated staircase result lacks independent admission"
            )
        traversal = Path(str(result.get("traversal", "")))
        with np.load(traversal, allow_pickle=False) as archive:
            connector = {
                name: np.asarray(archive[name], dtype=np.float64).copy()
                for name in (
                    "joint_position",
                    "root_position_world",
                    "root_orientation_world_wxyz",
                )
            }
        admission = {
            "classification": "staircase_intersecting",
            "independently_validated": True,
            "ordered_surface_heights_m": list(
                result["ordered_surface_heights_m"]
            ),
            "elevated_intervals_m": [
                list(interval)
                for interval in result["elevated_intervals_m"]
            ],
            "quality": dict(result["quality"]),
        }
        output.append((path, connector, admission))
    return tuple(output)


def _geometry_record(
    args: argparse.Namespace,
    contracts: tuple[StaircasePathContract, ...],
) -> dict[str, object]:
    heading_rad = math.radians(float(args.heading_degrees))
    paths = tuple(item.path for item in contracts)
    return {
        "center_start_scene_xy": [float(value) for value in args.center_start],
        "heading_degrees": float(args.heading_degrees),
        "heading_scene_xy": [math.cos(heading_rad), math.sin(heading_rad)],
        "path_length_m": float(args.path_length_m),
        "minimum_lateral_offset_m": min(
            path.lateral_offset_m for path in paths
        ),
        "maximum_lateral_offset_m": max(
            path.lateral_offset_m for path in paths
        ),
        "spacing_m": float(args.spacing_m),
        "paths": [
            {
                "index": item.path.index,
                "path_id": item.path.path_id,
                "lateral_offset_m": item.path.lateral_offset_m,
                "start_scene_xy": list(item.path.start_scene_xy),
                "stop_scene_xy": list(item.path.stop_scene_xy),
                "classification": item.classification,
                "ordered_surface_heights_m": list(
                    item.ordered_surface_heights_m
                ),
                "elevated_intervals_m": [
                    list(interval) for interval in item.elevated_intervals_m
                ],
            }
            for item in contracts
        ],
    }


def _target_height_grid(
    *, source_dataset: Path, target_scene: str
) -> tuple[np.ndarray, float, np.ndarray]:
    root = source_dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    targets = [
        item
        for item in manifest["clips"]
        if item.get("logical_name") == target_scene
        or item.get("relative_motion_path") == target_scene
    ]
    if len(targets) != 1 or targets[0].get("kind") != "terrain":
        raise ContractError(
            "parallel grid target scene must resolve exactly once"
        )
    return _target_grid(root, targets[0])


def _terrain_contracts(
    *, args: argparse.Namespace
) -> tuple[StaircasePathContract, ...]:
    origin, cell, height = _target_height_grid(
        source_dataset=args.source_dataset,
        target_scene=args.target_scene,
    )

    def sample_surface(points):
        return _sample_height_grid(
            origin_xy=origin,
            cell_size_m=cell,
            height_z=height,
            points_xy=np.asarray(points, dtype=np.float64),
        )

    manual = (
        args.minimum_lateral_offset_m is not None,
        args.maximum_lateral_offset_m is not None,
    )
    if manual[0] != manual[1]:
        raise ContractError(
            "manual parallel grid bounds must be supplied together"
        )
    heading_rad = math.radians(float(args.heading_degrees))
    heading = (math.cos(heading_rad), math.sin(heading_rad))
    paths = (
        parallel_path_grid(
            center_start_scene_xy=tuple(args.center_start),
            heading_scene_xy=heading,
            path_length_m=float(args.path_length_m),
            minimum_lateral_offset_m=float(
                args.minimum_lateral_offset_m
            ),
            maximum_lateral_offset_m=float(
                args.maximum_lateral_offset_m
            ),
            spacing_m=float(args.spacing_m),
        )
        if all(manual)
        else staircase_parallel_path_grid(
            center_start_scene_xy=tuple(args.center_start),
            heading_scene_xy=heading,
            path_length_m=float(args.path_length_m),
            elevated_scene_xy=elevated_grid_cell_centers(
                origin, cell, height
            ),
            spacing_m=float(args.spacing_m),
        )
    )
    return tuple(
        classify_staircase_path(
            path=path,
            sample_surface=sample_surface,
        )
        for path in paths
    )


def _run_path(
    *,
    contract: StaircasePathContract,
    args: argparse.Namespace,
    runner: Path,
) -> dict[str, object]:
    path = contract.path
    path_output = args.output / path.path_id
    path_output.mkdir(parents=True, exist_ok=True)
    route_contract_path = path_output / "route-contract.json"
    route_contract_path.write_text(
        json.dumps(
            route_contract_record(contract), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    command = root_search_command(
        python=Path(sys.executable),
        runner=runner,
        source_dataset=args.source_dataset,
        target_scene=args.target_scene,
        g1_xml=args.g1_xml,
        start_scene_xy=path.start_scene_xy,
        stop_scene_xy=path.stop_scene_xy,
        output=path_output,
        segment_length_m=args.segment_length_m,
        search_stride_m=args.search_stride_m,
        step_width_m=args.step_width_m,
        workers=args.workers_per_path,
        coarse_results=args.coarse_results,
        placement_shortlist=args.placement_shortlist,
        candidates_per_anchor=args.candidates_per_anchor,
        blend_frames=args.blend_frames,
        ordered_contact_levels=True,
    )
    log_path = path_output / "search.log"
    environment = child_environment(
        existing=dict(os.environ),
        repo_root=Path(__file__).resolve().parents[1],
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    search_mode = "ordered-terrain-levels"
    report_path = path_output / "report.json"
    traversal_path = path_output / "traversal.npz"
    common = {
        "index": path.index,
        "path_id": path.path_id,
        "lateral_offset_m": path.lateral_offset_m,
        "start_scene_xy": list(path.start_scene_xy),
        "stop_scene_xy": list(path.stop_scene_xy),
        "output": str(path_output),
        "returncode": int(completed.returncode),
        "search_mode": search_mode,
    }
    if (
        completed.returncode == 0
        and report_path.is_file()
        and traversal_path.is_file()
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation_path = path_output / "independent-validation.json"
        validation_log_path = path_output / "independent-validation.log"
        validation_command = independent_validation_command(
            python=Path(sys.executable),
            runner=(
                Path(__file__).resolve().parent
                / "run_g1_validate_traversal.py"
            ),
            traversal=traversal_path,
            target_dataset=path_output / "dataset",
            config=args.validation_config,
            g1_xml=args.g1_xml,
            route_contract=route_contract_path,
            expected_heading_degrees=float(args.heading_degrees),
            output=validation_path,
            maximum_unsupported_frames=int(
                args.maximum_unsupported_frames
            ),
        )
        with validation_log_path.open("w", encoding="utf-8") as log:
            validation = subprocess.run(
                validation_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=environment,
            )
        quality = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.is_file()
            else {}
        )
        if validation.returncode == 0 and quality.get("validated") is True:
            return {
                **common,
                "status": "validated",
                "traversal": str(traversal_path),
                "metrics": report.get("metrics", {}),
                "quality": quality,
                "independently_validated": True,
                "independent_validation": str(validation_path),
                "selected_candidate_ids": report.get(
                    "selected_candidate_ids", []
                ),
            }
        return {
            **common,
            "status": "failed",
            "failure_code": "validation_failed",
            "independently_validated": False,
            "independent_validation": (
                str(validation_path) if validation_path.is_file() else None
            ),
            "independent_validation_log": str(validation_log_path),
            "quality": quality,
        }
    diagnostic_path = path_output / "diagnostic.json"
    diagnostic = (
        json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if diagnostic_path.is_file()
        else None
    )
    failure_code = classify_search_failure(
        log_text=log_path.read_text(encoding="utf-8"),
        diagnostic=diagnostic,
    )
    return {
        **common,
        "status": "failed",
        "failure_code": failure_code,
        "diagnostic": (
            str(diagnostic_path) if diagnostic_path.is_file() else None
        ),
        "log": str(log_path),
    }


def main() -> int:
    args = parser().parse_args()
    if args.concurrent_paths < 1 or args.workers_per_path < 1:
        raise ContractError("parallel path grid worker counts are invalid")
    contracts = _terrain_contracts(args=args)
    paths = tuple(item.path for item in contracts)
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = _geometry_record(args, contracts)
    (args.output / "grid-contract.json").write_text(
        json.dumps(geometry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner = Path(__file__).resolve().parent / "run_g1_root_path_motion_search.py"
    by_index: dict[int, dict[str, object]] = {
        item.path.index: {
            "index": item.path.index,
            "path_id": item.path.path_id,
            "lateral_offset_m": item.path.lateral_offset_m,
            "start_scene_xy": list(item.path.start_scene_xy),
            "stop_scene_xy": list(item.path.stop_scene_xy),
            "status": "excluded",
            "failure_code": item.classification,
            "ordered_surface_heights_m": list(
                item.ordered_surface_heights_m
            ),
            "elevated_intervals_m": [
                list(interval) for interval in item.elevated_intervals_m
            ],
        }
        for item in contracts
        if item.classification != "staircase_intersecting"
    }
    selected = searchable_contracts(contracts)
    with ThreadPoolExecutor(max_workers=args.concurrent_paths) as executor:
        pending = {
            executor.submit(
                _run_path, contract=item, args=args, runner=runner
            ): item
            for item in selected
        }
        for future in as_completed(pending):
            item = pending[future]
            result = future.result()
            result["ordered_surface_heights_m"] = list(
                item.ordered_surface_heights_m
            )
            result["elevated_intervals_m"] = [
                list(interval) for interval in item.elevated_intervals_m
            ]
            by_index[item.path.index] = result
            print(json.dumps(result, sort_keys=True), flush=True)
    results = tuple(by_index[index] for index in range(len(paths)))
    summary = grid_summary(geometry=geometry, results=results)
    (args.output / "grid-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    entries = playlist_entries(paths=paths, results=results)
    if entries:
        arrays, playlist = build_staircase_grid_playlist(entries)
        validated_by_id = {
            str(item["path_id"]): item
            for item in results
            if item["status"] == "validated"
        }
        for segment in playlist["segments"]:
            result = validated_by_id[str(segment["path_id"])]
            segment["source"] = str(result["traversal"])
            segment["independent_validation"] = str(
                result["independent_validation"]
            )
        np.savez_compressed(args.output / "grid-playlist.npz", **arrays)
        (args.output / "grid-playlist.json").write_text(
            json.dumps(playlist, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if summary["validated_path_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
