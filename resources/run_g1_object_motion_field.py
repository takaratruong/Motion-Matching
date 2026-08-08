#!/usr/bin/env python3
"""Build a complete object-local multi-heading motion-field manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from mm_sonic.joints import ContractError
from mm_sonic.torch_object_motion_field import (
    MotionFieldLine,
    motion_field_intersections,
    rasterized_motion_field,
)

from resources.run_g1_parallel_path_grid import (
    _target_height_grid,
    elevated_grid_cell_centers,
)


def _wrapped_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _line_record(line: MotionFieldLine) -> dict[str, object]:
    return {
        "line_id": line.line_id,
        "family_index": line.family_index,
        "lane_index": line.lane_index,
        "heading_degrees": line.heading_degrees,
        "heading_scene_xy": list(line.heading_scene_xy),
        "lateral_offset_m": line.lateral_offset_m,
        "start_scene_xy": list(line.start_scene_xy),
        "stop_scene_xy": list(line.stop_scene_xy),
    }


def build_motion_field_manifest(
    *,
    lines: Sequence[MotionFieldLine],
    route_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Attach admitted route artifacts without hiding empty field cells."""

    owned = tuple(lines)
    records = tuple(route_records)
    if (
        not owned
        or any(not isinstance(line, MotionFieldLine) for line in owned)
        or len({line.line_id for line in owned}) != len(owned)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ContractError("motion field manifest input is invalid")
    by_line = {line.line_id: line for line in owned}
    bound: dict[str, dict[str, object]] = {}
    for record in records:
        line_id = str(record.get("line_id", ""))
        if line_id not in by_line:
            raise ContractError("motion field route references unknown line")
        if line_id in bound:
            raise ContractError("motion field line has duplicate routes")
        bound[line_id] = record

    output_lines = []
    admitted = []
    for line in owned:
        item = _line_record(line)
        record = bound.get(line.line_id)
        if record is None:
            item.update(status="missing", reason="no_route_motion")
            output_lines.append(item)
            continue
        try:
            heading = float(record["heading_degrees"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("motion field route record is invalid") from error
        if not math.isfinite(heading):
            raise ContractError("motion field route record is invalid")
        if abs(_wrapped_degrees(heading - line.heading_degrees)) > 1.0e-3:
            reason = "heading_mismatch"
        elif record.get("classification") != "staircase_intersecting":
            reason = "not_staircase_intersecting"
        elif record.get("independently_validated") is not True:
            reason = "independent_validation_failed"
        elif not isinstance(record.get("artifact"), str) or not record["artifact"]:
            reason = "missing_artifact"
        else:
            reason = None
        item["route"] = dict(record)
        if reason is None:
            item["status"] = "admitted"
            admitted.append(line.line_id)
        else:
            item.update(status="rejected", reason=reason)
        output_lines.append(item)

    intersections = motion_field_intersections(owned)
    return {
        "schema": "g1-object-motion-field/v1",
        "line_count": len(owned),
        "admitted_line_count": len(admitted),
        "missing_line_count": sum(
            item["status"] == "missing" for item in output_lines
        ),
        "rejected_line_count": sum(
            item["status"] == "rejected" for item in output_lines
        ),
        "admitted_line_ids": admitted,
        "lines": output_lines,
        "intersection_count": len(intersections),
        "intersections": [
            {
                "first_line_id": item.first_line_id,
                "second_line_id": item.second_line_id,
                "first_heading_degrees": item.first_heading_degrees,
                "second_heading_degrees": item.second_heading_degrees,
                "scene_xy": list(item.scene_xy),
                "first_progress_m": item.first_progress_m,
                "second_progress_m": item.second_progress_m,
            }
            for item in intersections
        ],
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser()
    output.add_argument("--source-dataset", type=Path, required=True)
    output.add_argument("--target-scene", required=True)
    output.add_argument("--heading-degrees", type=float, nargs="+", required=True)
    output.add_argument("--spacing-m", type=float, default=0.20)
    output.add_argument("--approach-margin-m", type=float, default=0.50)
    output.add_argument("--exit-margin-m", type=float, default=0.50)
    output.add_argument("--route-records", type=Path)
    output.add_argument("--output", type=Path, required=True)
    return output


def main() -> int:
    args = parser().parse_args()
    origin, cell, height = _target_height_grid(
        source_dataset=args.source_dataset, target_scene=args.target_scene
    )
    elevated = elevated_grid_cell_centers(origin, cell, height)
    lines = rasterized_motion_field(
        elevated_scene_xy=elevated,
        heading_degrees=tuple(args.heading_degrees),
        spacing_m=float(args.spacing_m),
        approach_margin_m=float(args.approach_margin_m),
        exit_margin_m=float(args.exit_margin_m),
    )
    records = ()
    if args.route_records is not None:
        try:
            document = json.loads(args.route_records.read_text("utf-8"))
            records = tuple(document["routes"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ContractError("motion field route records are invalid") from error
    manifest = build_motion_field_manifest(lines=lines, route_records=records)
    manifest["geometry"] = {
        "target_scene": args.target_scene,
        "spacing_m": float(args.spacing_m),
        "heading_degrees": list(args.heading_degrees),
        "approach_margin_m": float(args.approach_margin_m),
        "exit_margin_m": float(args.exit_margin_m),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "motion-field.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "line_count": manifest["line_count"],
        "admitted_line_count": manifest["admitted_line_count"],
        "intersection_count": manifest["intersection_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
