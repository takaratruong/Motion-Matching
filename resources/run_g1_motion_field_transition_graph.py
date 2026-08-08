#!/usr/bin/env python3
"""Register freshly validated routes as directed object-motion-field edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from mm_sonic.joints import ContractError
from mm_sonic.torch_object_motion_field import (
    MotionFieldIntersection,
    MotionFieldLine,
    assign_endpoint_to_heading_line,
)
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def _line_record(line: MotionFieldLine) -> dict[str, object]:
    return {
        "family_index": line.family_index,
        "lane_index": line.lane_index,
        "line_id": line.line_id,
        "heading_degrees": line.heading_degrees,
        "heading_scene_xy": line.heading_scene_xy,
        "lateral_offset_m": line.lateral_offset_m,
        "start_scene_xy": line.start_scene_xy,
        "stop_scene_xy": line.stop_scene_xy,
    }


def _assignment_record(assignment) -> dict[str, object]:
    return {
        "line_id": assignment.line_id,
        "lane_index": assignment.lane_index,
        "heading_degrees": assignment.heading_degrees,
        "projected_scene_xy": assignment.projected_scene_xy,
        "progress_m": assignment.progress_m,
        "lateral_distance_m": assignment.lateral_distance_m,
    }


def load_motion_field_geometry(
    path: Path,
) -> tuple[tuple[MotionFieldLine, ...], tuple[MotionFieldIntersection, ...]]:
    """Load the authoritative raster geometry without regenerating its lanes."""

    try:
        document = json.loads(path.read_text("utf-8"))
        lines = tuple(
            MotionFieldLine(
                family_index=item["family_index"],
                lane_index=item["lane_index"],
                line_id=item["line_id"],
                heading_degrees=item["heading_degrees"],
                heading_scene_xy=tuple(item["heading_scene_xy"]),
                lateral_offset_m=item["lateral_offset_m"],
                start_scene_xy=tuple(item["start_scene_xy"]),
                stop_scene_xy=tuple(item["stop_scene_xy"]),
            )
            for item in document["lines"]
        )
        intersections = tuple(
            MotionFieldIntersection(
                first_line_id=item["first_line_id"],
                second_line_id=item["second_line_id"],
                first_heading_degrees=item["first_heading_degrees"],
                second_heading_degrees=item["second_heading_degrees"],
                scene_xy=tuple(item["scene_xy"]),
                first_progress_m=item["first_progress_m"],
                second_progress_m=item["second_progress_m"],
            )
            for item in document["intersections"]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ContractError("motion field geometry is invalid") from error
    if not lines:
        raise ContractError("motion field geometry is invalid")
    return lines, intersections


def load_edge_record(
    *,
    specification: dict[str, object],
    matcher_to_scene_xy: Callable[[object], object],
    expected_query_scene: str | None = None,
) -> dict[str, object]:
    """Load one route only when its validation hashes the same exact artifact."""

    try:
        artifact = Path(str(specification["artifact"])).resolve()
        validation_path = Path(str(specification["validation"])).resolve()
        validation = json.loads(validation_path.read_text("utf-8"))
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if validation.get("validated") is not True:
            raise ContractError("edge validation did not pass")
        if Path(str(validation["input"])).resolve() != artifact:
            raise ContractError("edge validation input mismatch")
        if validation.get("artifact_sha256") != digest:
            raise ContractError("edge validation hash mismatch")
        if (
            expected_query_scene is not None
            and validation.get("query_scene") != expected_query_scene
        ):
            raise ContractError("edge validation terrain scene mismatch")
        with np.load(artifact, allow_pickle=False) as archive:
            roots = np.asarray(archive["root_position_world"], dtype=np.float64)
        if roots.ndim != 2 or roots.shape[1] != 3 or len(roots) < 2:
            raise ContractError("edge route roots are invalid")
        scene = np.asarray(matcher_to_scene_xy(roots[[0, -1], :2]), dtype=np.float64)
        if scene.shape != (2, 2) or not np.isfinite(scene).all():
            raise ContractError("edge scene endpoints are invalid")
        from_heading = float(specification["from_heading_degrees"])
        to_heading = float(specification["to_heading_degrees"])
        observed = float(specification["observed_heading_change_degrees"])
        edge_id = str(specification["edge_id"])
        provenance = specification.get("provenance", [])
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ContractError("edge specification is invalid") from error
    if (
        not edge_id
        or not all(math.isfinite(value) for value in (from_heading, to_heading, observed))
        or not isinstance(provenance, list)
    ):
        raise ContractError("edge specification is invalid")
    return {
        "edge_id": edge_id,
        "from_heading_degrees": from_heading,
        "to_heading_degrees": to_heading,
        "observed_heading_change_degrees": observed,
        "start_scene_xy": scene[0],
        "end_scene_xy": scene[1],
        "artifact": str(artifact),
        "artifact_sha256": digest,
        "validation": str(validation_path),
        "frame_count": int(validation["frame_count"]),
        "metrics": dict(validation["metrics"]),
        "provenance": provenance,
    }


def build_transition_graph(
    *,
    lines: Sequence[MotionFieldLine],
    intersections: Sequence[MotionFieldIntersection],
    edge_records: Sequence[dict[str, object]],
    maximum_lane_error_m: float,
) -> dict[str, object]:
    """Assign validated route endpoints to lanes and their exact crossing."""

    owned_lines = tuple(lines)
    owned_intersections = tuple(intersections)
    records = tuple(edge_records)
    if (
        not owned_lines
        or not math.isfinite(float(maximum_lane_error_m))
        or maximum_lane_error_m <= 0.0
        or len({str(item.get("edge_id", "")) for item in records}) != len(records)
    ):
        raise ContractError("transition graph input is invalid")
    output = []
    for record in records:
        start = assign_endpoint_to_heading_line(
            lines=owned_lines,
            heading_degrees=float(record["from_heading_degrees"]),
            endpoint_scene_xy=record["start_scene_xy"],
        )
        end = assign_endpoint_to_heading_line(
            lines=owned_lines,
            heading_degrees=float(record["to_heading_degrees"]),
            endpoint_scene_xy=record["end_scene_xy"],
        )
        if max(start.lateral_distance_m, end.lateral_distance_m) > maximum_lane_error_m:
            raise ContractError(f"transition edge lane error exceeds gate: {record['edge_id']}")
        crossing = next(
            (
                item
                for item in owned_intersections
                if {item.first_line_id, item.second_line_id}
                == {start.line_id, end.line_id}
            ),
            None,
        )
        if crossing is None:
            raise ContractError(f"transition edge has no field intersection: {record['edge_id']}")
        edge = dict(record)
        edge["start_scene_xy"] = [float(value) for value in record["start_scene_xy"]]
        edge["end_scene_xy"] = [float(value) for value in record["end_scene_xy"]]
        edge["from_line_id"] = start.line_id
        edge["to_line_id"] = end.line_id
        edge["from_assignment"] = _assignment_record(start)
        edge["to_assignment"] = _assignment_record(end)
        edge["intersection"] = {
            "status": "associated",
            "scene_xy": crossing.scene_xy,
            "route_start_distance_m": float(
                np.linalg.norm(np.asarray(record["start_scene_xy"]) - crossing.scene_xy)
            ),
            "route_end_distance_m": float(
                np.linalg.norm(np.asarray(record["end_scene_xy"]) - crossing.scene_xy)
            ),
        }
        output.append(edge)
    occupied = sorted(
        {line_id for item in output for line_id in (item["from_line_id"], item["to_line_id"])}
    )
    return {
        "schema": "g1-object-motion-field-transition-graph/v1",
        "edge_count": len(output),
        "occupied_line_count": len(occupied),
        "occupied_line_ids": occupied,
        "maximum_lane_error_m": maximum_lane_error_m,
        "edges": output,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--geometry", type=Path, required=True)
    result.add_argument("--edge-specifications", type=Path, required=True)
    result.add_argument("--target-dataset", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--maximum-lane-error-m", type=float, default=0.10)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    lines, intersections = load_motion_field_geometry(args.geometry)
    geometry_document = json.loads(args.geometry.read_text("utf-8"))
    expected_query_scene = (
        "terrain/grail/"
        + geometry_document["geometry"]["target_scene"]
        + "/motion.npz"
    )
    try:
        specifications = json.loads(args.edge_specifications.read_text("utf-8"))["edges"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ContractError("edge specifications are invalid") from error
    resolved = resolve_stair_config(
        args.target_dataset,
        load_experiment_config(args.config),
        device="cpu",
    )

    def matcher_to_scene(points: object) -> np.ndarray:
        tensor = torch.as_tensor(points, dtype=torch.float32)
        return (
            resolved.measurement_extension.alignment.matcher_to_scene_xy(tensor)
            .cpu()
            .numpy()
        )

    records = tuple(
        load_edge_record(
            specification=specification,
            matcher_to_scene_xy=matcher_to_scene,
            expected_query_scene=expected_query_scene,
        )
        for specification in specifications
    )
    graph = build_transition_graph(
        lines=lines,
        intersections=intersections,
        edge_records=records,
        maximum_lane_error_m=args.maximum_lane_error_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({"output": str(args.output), "edge_count": graph["edge_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
