"""Strict semantic parity gate for the frozen G1 visual-runtime oracle."""

from __future__ import annotations

import argparse
import csv
import io
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


EXPECTED_ROW_COUNT = 64
COST_ULP_LIMIT = 8
COST_COLUMNS = (
    "incumbent_cost",
    "selected_cost",
    "selected_terrain_error",
    "continuation_cost",
)

RUNTIME_COLUMNS = (
    "frame",
    "fixed_dt",
    "scene_id",
    "mode",
    "route",
    "query_bits_hex",
    "query_database_frame",
    "query_range",
    "selected_database_frame",
    "database_frame",
    "range",
    "source_range",
    "searched",
    "transitioned",
    "incumbent_cost",
    "selected_cost",
    "selected_terrain_error",
    "effective_terrain_weight",
    "terrain0",
    "terrain1",
    "terrain2",
    "terrain3",
    "terrain_point0_x",
    "terrain_point0_y",
    "terrain_point0_z",
    "terrain_point1_x",
    "terrain_point1_y",
    "terrain_point1_z",
    "terrain_point2_x",
    "terrain_point2_y",
    "terrain_point2_z",
    "terrain_point3_x",
    "terrain_point3_y",
    "terrain_point3_z",
    "raw_selected_hips_y",
    "inertialized_hips_y",
    "rendered_hips_y",
    "hips_inertial_offset_y",
    "runtime_root_surface_height",
    "runtime_left_toe_surface_height",
    "runtime_right_toe_surface_height",
    "raw_selected_hips_clearance",
    "raw_selected_left_toe_clearance",
    "raw_selected_right_toe_clearance",
    "raw_selected_min_clearance",
    "inertialized_hips_clearance",
    "inertialized_left_toe_clearance",
    "inertialized_right_toe_clearance",
    "inertialized_min_clearance",
    "rendered_hips_clearance",
    "rendered_left_toe_clearance",
    "rendered_right_toe_clearance",
    "rendered_min_clearance",
    "adjustment_xz",
    "adjustment_y",
    "clamp_xz",
    "clamp_y",
    "matching_enabled",
    "adjustment_enabled",
    "clamping_enabled",
    "support_retargeting_enabled",
    "ik_enabled",
    "source_name",
    "source_terrain",
    "source_index",
    "continuation_cost",
    "source_root_height",
    "source_left_toe_height",
    "source_right_toe_height",
    "runtime_support_root_height",
    "runtime_support_left_toe_height",
    "runtime_support_right_toe_height",
    "support_root_delta",
    "support_left_toe_delta",
    "support_right_toe_delta",
    "support_height",
    "support_velocity",
    "support_source",
    "airborne_frames",
    "left_contact",
    "right_contact",
    "support_retargeted_hips_y",
    "ik_adjusted_hips_y",
    "simulation_x",
    "simulation_z",
    "walkability_class",
    "blocked",
    "blocked_reason",
    "blocked_distance",
    "blocked_point_x",
    "blocked_point_z",
    "commanded_speed",
    "applied_speed",
    "route_waypoint",
    "route_complete",
    "route_target_height",
    "scene_generation",
    "scene_frame",
    "scene_reset_count",
    "scene_switch_failed",
    "motion_pack_load_count",
    "model_load_count",
    "model_unload_count",
    "live_model_count",
)

_CANONICAL_HEADER = (",".join(RUNTIME_COLUMNS) + "\n").encode("ascii")
_COST_INDICES = {name: RUNTIME_COLUMNS.index(name) for name in COST_COLUMNS}


class RuntimeParityError(ValueError):
    """The candidate runtime log violates the strict parity contract."""


@dataclass(frozen=True)
class RuntimeParitySummary:
    row_count: int
    exact_column_count: int
    cost_max_ulps: dict[str, int]


def _read_runtime_log(path: str | Path, label: str) -> list[list[str]]:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise RuntimeParityError(
            f"{label} runtime log cannot be read: {source}: {error}"
        ) from error

    header_end = payload.find(b"\n")
    raw_header = payload[: header_end + 1] if header_end >= 0 else payload
    if raw_header != _CANONICAL_HEADER:
        raise RuntimeParityError(
            f"{label} header does not match the canonical schema"
        )

    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeParityError(
            f"{label} runtime log is not ASCII CSV: {source}"
        ) from error

    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise RuntimeParityError(
            f"{label} runtime log is malformed CSV: {error}"
        ) from error

    if not rows or tuple(rows[0]) != RUNTIME_COLUMNS:
        raise RuntimeParityError(
            f"{label} header does not match the canonical schema"
        )

    data_rows = rows[1:]
    expected_fields = len(RUNTIME_COLUMNS)
    for row_number, row in enumerate(data_rows, start=1):
        if len(row) != expected_fields:
            raise RuntimeParityError(
                f"{label} row {row_number} must contain "
                f"{expected_fields} fields, found {len(row)}"
            )

    if len(data_rows) != EXPECTED_ROW_COUNT:
        raise RuntimeParityError(
            f"{label} must contain exactly {EXPECTED_ROW_COUNT} data rows, "
            f"found {len(data_rows)}"
        )
    return data_rows


def _binary32_bits(
    text: str,
    label: str,
    row_number: int,
    column: str,
) -> int:
    try:
        value = float(text)
    except ValueError as error:
        raise RuntimeParityError(
            f"{label} row {row_number} {column} must be finite binary32, "
            f"found {text!r}"
        ) from error
    if not math.isfinite(value):
        raise RuntimeParityError(
            f"{label} row {row_number} {column} must be finite binary32, "
            f"found {text!r}"
        )
    try:
        payload = struct.pack(">f", value)
    except OverflowError as error:
        raise RuntimeParityError(
            f"{label} row {row_number} {column} must be finite binary32, "
            f"found {text!r}"
        ) from error
    rounded = struct.unpack(">f", payload)[0]
    if not math.isfinite(rounded):
        raise RuntimeParityError(
            f"{label} row {row_number} {column} must be finite binary32, "
            f"found {text!r}"
        )
    return struct.unpack(">I", payload)[0]


def _ordered_binary32(bits: int) -> int:
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return bits | 0x80000000


def _cost_ulp_distance(
    expected: str,
    actual: str,
    row_number: int,
    column: str,
) -> int:
    expected_bits = _binary32_bits(
        expected,
        "expected",
        row_number,
        column,
    )
    actual_bits = _binary32_bits(
        actual,
        "actual",
        row_number,
        column,
    )
    return abs(
        _ordered_binary32(expected_bits) - _ordered_binary32(actual_bits)
    )


def compare_runtime_logs(
    expected_path: str | Path,
    actual_path: str | Path,
) -> RuntimeParitySummary:
    """Validate strict behavioral parity and return observed cost maxima."""

    expected_rows = _read_runtime_log(expected_path, "expected")
    actual_rows = _read_runtime_log(actual_path, "actual")
    maxima = {name: 0 for name in COST_COLUMNS}

    for row_number, (expected, actual) in enumerate(
        zip(expected_rows, actual_rows),
        start=1,
    ):
        for column_index, column in enumerate(RUNTIME_COLUMNS):
            if column in _COST_INDICES:
                distance = _cost_ulp_distance(
                    expected[column_index],
                    actual[column_index],
                    row_number,
                    column,
                )
                maxima[column] = max(maxima[column], distance)
                if distance > COST_ULP_LIMIT:
                    raise RuntimeParityError(
                        f"row {row_number} {column} differs by {distance} ULP; "
                        f"limit {COST_ULP_LIMIT}"
                    )
            elif expected[column_index] != actual[column_index]:
                raise RuntimeParityError(
                    f"row {row_number} {column} exact mismatch: "
                    f"expected {expected[column_index]!r}, "
                    f"actual {actual[column_index]!r}"
                )

    return RuntimeParitySummary(
        row_count=EXPECTED_ROW_COUNT,
        exact_column_count=len(RUNTIME_COLUMNS) - len(COST_COLUMNS),
        cost_max_ulps=maxima,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a 64-frame G1 runtime log against its frozen visual "
            "oracle under the strict semantic parity contract."
        )
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = compare_runtime_logs(args.expected, args.actual)
    except RuntimeParityError as error:
        print(f"runtime parity failed: {error}", file=sys.stderr)
        return 1

    print(
        f"runtime parity passed: {summary.row_count} rows; "
        f"{summary.exact_column_count} non-cost columns exact"
    )
    for column in COST_COLUMNS:
        print(
            f"{column}: max {summary.cost_max_ulps[column]} ULP "
            f"(limit {COST_ULP_LIMIT})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
