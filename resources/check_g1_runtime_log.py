#!/usr/bin/env python3
import argparse
import csv
import math
import struct


CSV_COLUMNS = (
    "frame", "fixed_dt", "scene_id", "mode", "route",
    "query_bits_hex",
    "query_database_frame", "query_range", "selected_database_frame",
    "database_frame", "range", "source_range", "searched", "transitioned",
    "incumbent_cost",
    "selected_cost", "selected_terrain_error", "effective_terrain_weight",
    "terrain0", "terrain1", "terrain2", "terrain3",
    "terrain_point0_x", "terrain_point0_y", "terrain_point0_z",
    "terrain_point1_x", "terrain_point1_y", "terrain_point1_z",
    "terrain_point2_x", "terrain_point2_y", "terrain_point2_z",
    "terrain_point3_x", "terrain_point3_y", "terrain_point3_z",
    "raw_selected_hips_y", "inertialized_hips_y", "rendered_hips_y",
    "hips_inertial_offset_y", "runtime_root_surface_height",
    "runtime_left_toe_surface_height", "runtime_right_toe_surface_height",
    "raw_selected_hips_clearance", "raw_selected_left_toe_clearance",
    "raw_selected_right_toe_clearance", "raw_selected_min_clearance",
    "inertialized_hips_clearance", "inertialized_left_toe_clearance",
    "inertialized_right_toe_clearance", "inertialized_min_clearance",
    "rendered_hips_clearance", "rendered_left_toe_clearance",
    "rendered_right_toe_clearance", "rendered_min_clearance",
    "adjustment_xz", "adjustment_y", "clamp_xz", "clamp_y",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled", "ik_enabled",
)
REQUIRED_COLUMNS = set(CSV_COLUMNS)
TEXT_COLUMNS = {"scene_id", "mode", "route", "query_bits_hex"}
INTEGER_COLUMNS = {
    "frame", "query_database_frame", "query_range", "selected_database_frame",
    "database_frame", "range", "source_range", "searched", "transitioned",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled", "ik_enabled",
}
FLAG_COLUMNS = {
    "searched", "transitioned", "matching_enabled", "adjustment_enabled",
    "clamping_enabled", "support_retargeting_enabled", "ik_enabled",
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate CSV column")
        if fields[:len(CSV_COLUMNS)] != list(CSV_COLUMNS):
            raise ValueError("immutable CSV prefix was renamed or reordered")
        missing = sorted(REQUIRED_COLUMNS - set(fields))
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")
        rows = []
        for index, row in enumerate(reader):
            if None in row:
                raise ValueError(f"row {index}: CSV data is wider than header")
            rows.append(row)
        return rows


def _finite(row, name, index):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {index}: invalid {name}") from error
    if not math.isfinite(value):
        raise ValueError(f"row {index}: non-finite {name}")
    return value


def _integer(row, name, index):
    value = _finite(row, name, index)
    if value != int(value):
        raise ValueError(f"row {index}: non-integer {name}")
    integer = int(value)
    if integer < -(2 ** 31) or integer > 2 ** 31 - 1:
        raise ValueError(f"row {index}: {name} is outside signed int32")
    return integer


def _float32_bits(value, index, label):
    try:
        return int.from_bytes(struct.pack(">f", value), "big")
    except (OverflowError, struct.error) as error:
        raise ValueError(f"row {index}: {label} is not float32") from error


def _float32(row, name, index):
    value = _finite(row, name, index)
    _float32_bits(value, index, name)
    return value


def _float32_ulp_distance(left, right, index):
    left_bits = _float32_bits(left, index, "cost")
    right_bits = _float32_bits(right, index, "cost")
    return abs(left_bits - right_bits)


def _check_query_snapshot(row, index):
    snapshot = row["query_bits_hex"]
    values = []
    for dimension in range(31):
        bits = snapshot[dimension * 8:(dimension + 1) * 8]
        value = struct.unpack(">f", bytes.fromhex(bits))[0]
        if not math.isfinite(value):
            raise ValueError(
                f"row {index}: non-finite query dimension {dimension}")
        values.append(value)
    for sample in range(4):
        terrain = _float32(row, f"terrain{sample}", index)
        terrain_bits = struct.pack(">f", terrain).hex()
        query_bits = snapshot[(27 + sample) * 8:(28 + sample) * 8]
        if query_bits != terrain_bits:
            raise ValueError(
                f"row {index}: terrain query bits disagree at sample {sample}")
    return values


def check_substride(rows, minimum_period=13):
    values = [_integer(row, "database_frame", i) for i, row in enumerate(rows)]
    for period in range(1, minimum_period):
        width = 3 * period
        for start in range(0, len(values) - width + 1):
            a = values[start:start + period]
            if a == values[start + period:start + 2 * period] == \
                    values[start + 2 * period:start + 3 * period]:
                raise ValueError(
                    f"row {start}: repeated sub-stride period {period}")


def check_rows(rows):
    if not rows:
        raise ValueError("runtime log is empty")
    previous = None
    previous_range = None
    for index, row in enumerate(rows):
        frame = _integer(row, "frame", index)
        query_frame = _integer(row, "query_database_frame", index)
        query_range = _integer(row, "query_range", index)
        selected_frame = _integer(row, "selected_database_frame", index)
        current = _integer(row, "database_frame", index)
        current_range = _integer(row, "range", index)
        source_range = _integer(row, "source_range", index)
        transitioned = _integer(row, "transitioned", index)
        searched = _integer(row, "searched", index)
        for name, value in (
                ("query_database_frame", query_frame),
                ("selected_database_frame", selected_frame),
                ("database_frame", current),
                ("query_range", query_range),
                ("range", current_range),
                ("source_range", source_range)):
            if value < 0:
                raise ValueError(f"row {index}: {name} must be nonnegative")
        fixed_dt = _float32(row, "fixed_dt", index)
        if fixed_dt <= 0.0:
            raise ValueError(f"row {index}: fixed_dt must be positive")
        terrain_weight = _float32(
            row, "effective_terrain_weight", index)
        if not 0.0 <= terrain_weight <= 10.0:
            raise ValueError(
                f"row {index}: effective_terrain_weight must be in [0, 10]")
        for name in ("adjustment_xz", "clamp_xz"):
            if _float32(row, name, index) < 0.0:
                raise ValueError(f"row {index}: {name} must be nonnegative")
        if frame != index:
            raise ValueError(f"row {index}: frame sequence is {frame}")
        if transitioned not in (0, 1) or searched not in (0, 1):
            raise ValueError(f"row {index}: flags must be 0 or 1")
        if previous is not None and query_frame != previous:
            raise ValueError(
                f"row {index}: query frame {query_frame} does not match "
                f"prior pose frame {previous}")
        if previous_range is not None and query_range != previous_range:
            raise ValueError(
                f"row {index}: query range {query_range} does not match "
                f"prior pose range {previous_range}")
        if transitioned and not searched:
            raise ValueError(f"row {index}: transition without search")
        if transitioned and selected_frame == query_frame:
            raise ValueError(
                f"row {index}: transitioned with unchanged selected frame")
        if not transitioned and selected_frame != query_frame:
            raise ValueError(
                f"row {index}: selected frame changed without transition")
        if current not in (selected_frame, selected_frame + 1):
            raise ValueError(
                f"row {index}: post-advance frame is inconsistent with selected frame")
        if current_range != source_range:
            raise ValueError(
                f"row {index}: pose range differs from selected source range")
        if not transitioned and query_range != source_range:
            raise ValueError(
                f"row {index}: source range changed without transition")
        incumbent = _finite(row, "incumbent_cost", index)
        selected = _finite(row, "selected_cost", index)
        selected_terrain_error = _finite(
            row, "selected_terrain_error", index)
        for value in (incumbent, selected, selected_terrain_error):
            _float32_bits(value, index, "cost")
        for name, value in (
                ("incumbent_cost", incumbent),
                ("selected_cost", selected),
                ("selected_terrain_error", selected_terrain_error)):
            if value < 0.0:
                raise ValueError(f"row {index}: negative {name}")
        # Search and incumbent costs are independently normalized and
        # materialized as float32, so near-ties can round a few ULPs apart.
        if transitioned and not selected < incumbent:
            cost_ulps = _float32_ulp_distance(selected, incumbent, index)
            if cost_ulps > 4:
                raise ValueError(
                    f"row {index}: transition did not beat incumbent cost "
                    f"and differs by {cost_ulps} float32 ULPs")
        if not transitioned and not searched and selected != incumbent:
            raise ValueError(
                f"row {index}: unsearched no-transition selected cost "
                "differs from incumbent cost")
        if not transitioned and searched:
            cost_ulps = _float32_ulp_distance(selected, incumbent, index)
            if cost_ulps > 4:
                raise ValueError(
                    f"row {index}: searched no-transition selected cost "
                    f"differs by {cost_ulps} float32 ULPs")
        for name in CSV_COLUMNS:
            if name in TEXT_COLUMNS:
                if not row.get(name):
                    raise ValueError(f"row {index}: empty {name}")
                if name == "query_bits_hex" and (
                        len(row[name]) != 31 * 8 or
                        any(character not in "0123456789abcdef"
                            for character in row[name])):
                    raise ValueError(
                        f"row {index}: query_bits_hex is not 31 float bit patterns")
            elif name in INTEGER_COLUMNS:
                _integer(row, name, index)
            else:
                _float32(row, name, index)
        for name in FLAG_COLUMNS:
            if _integer(row, name, index) not in (0, 1):
                raise ValueError(f"row {index}: {name} must be 0 or 1")
        _check_query_snapshot(row, index)
        if previous is not None and not transitioned and current != previous + 1:
            raise ValueError(
                f"row {index}: nonsequential advance {previous}->{current}")
        if (previous_range is not None and not transitioned and
                current_range != previous_range):
            raise ValueError(
                f"row {index}: range change without transition")
        previous = current
        previous_range = current_range
    check_substride(rows)
    return {
        "frames": len(rows),
        "transitions": sum(int(row["transitioned"]) for row in rows),
    }


def compare_control(treatment, control):
    check_rows(treatment)
    check_rows(control)
    if len(treatment) != len(control):
        raise ValueError("control and treatment lengths differ")
    metadata = ("frame", "fixed_dt", "scene_id", "mode", "route")
    for index, (treatment_row, control_row) in enumerate(
            zip(treatment, control)):
        if any(treatment_row[name] != control_row[name] for name in metadata):
            raise ValueError(
                f"row {index}: control and treatment script metadata differ")
        if _finite(treatment_row, "effective_terrain_weight", index) != 4.0:
            raise ValueError(f"row {index}: treatment weight must be 4")
        if _finite(control_row, "effective_terrain_weight", index) != 0.0:
            raise ValueError(f"row {index}: control weight must be 0")

    def active_errors(rows, label):
        errors = [
            _finite(row, "selected_terrain_error", index)
            for index, row in enumerate(rows)
            if max(abs(_finite(row, f"terrain{sample}", index))
                   for sample in range(4)) > 0.05
        ]
        if not errors:
            raise ValueError(f"{label} terrain query never became active")
        return errors

    treatment_errors = active_errors(treatment, "treatment")
    control_errors = active_errors(control, "control")
    treatment_error = sum(treatment_errors) / len(treatment_errors)
    control_error = sum(control_errors) / len(control_errors)
    if not treatment_error < control_error:
        raise ValueError(
            "terrain treatment did not improve error: "
            f"{treatment_error} >= {control_error}")
    return treatment_error, control_error


def check_gate_a_contract(rows, expected_frames=375):
    summary = check_rows(rows)
    if len(rows) != expected_frames:
        raise ValueError(
            f"Gate A requires exactly {expected_frames} rows, got {len(rows)}")
    expected_dt = struct.pack(">f", 0.04)
    expected_text = {
        "scene_id": "grail-curb-default",
        "mode": "terrain",
        "route": "curb-forward",
    }
    expected_flags = {
        "matching_enabled": (1, "matching"),
        "adjustment_enabled": (1, "adjustment"),
        "clamping_enabled": (1, "clamping"),
        "support_retargeting_enabled": (0, "support retargeting"),
        "ik_enabled": (0, "IK"),
    }
    for index, row in enumerate(rows):
        fixed_dt = _finite(row, "fixed_dt", index)
        if struct.pack(">f", fixed_dt) != expected_dt:
            raise ValueError(f"row {index}: Gate A fixed_dt must be float32 0.04")
        for name, expected in expected_text.items():
            if row[name] != expected:
                raise ValueError(
                    f"row {index}: Gate A {name.replace('_id', '')} must be {expected}")
        if _finite(row, "effective_terrain_weight", index) != 4.0:
            raise ValueError(f"row {index}: Gate A weight must be 4")
        for name, (expected, label) in expected_flags.items():
            if _integer(row, name, index) != expected:
                raise ValueError(
                    f"row {index}: Gate A {label} must be {expected}")
    return summary


def diagnose_gate_a(rows):
    check_rows(rows)
    positive = next((
        index for index, row in enumerate(rows)
        if max(float(row[f"terrain{sample}"]) for sample in range(4)) > 1e-6
    ), None)
    if positive is None:
        raise ValueError("terrain query never became positive")
    first_penetration = None
    classification = "no-penetration"
    for index, row in enumerate(rows):
        raw = float(row["raw_selected_min_clearance"]) < 0.0
        blended = (
            float(row["inertialized_min_clearance"]) < 0.0
            or float(row["rendered_min_clearance"]) < 0.0)
        if raw or blended:
            first_penetration = index
            classification = "raw-selected" if raw else "blended-rendered"
            break
    diagnostic = rows[first_penetration if first_penetration is not None else positive]
    return {
        "first_positive_query_frame": int(rows[positive]["frame"]),
        "first_penetration_frame": (
            None if first_penetration is None
            else int(rows[first_penetration]["frame"])),
        "penetration_class": classification,
        "selected_range": int(diagnostic["range"]),
        "hips_inertial_offset_y": float(diagnostic["hips_inertial_offset_y"]),
        "adjustment_xz": float(diagnostic["adjustment_xz"]),
        "adjustment_y": float(diagnostic["adjustment_y"]),
        "clamp_xz": float(diagnostic["clamp_xz"]),
        "clamp_y": float(diagnostic["clamp_y"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--compare-control")
    parser.add_argument("--gate-a", action="store_true")
    args = parser.parse_args(argv)
    rows = read_rows(args.log)
    summary = check_rows(rows)
    if args.compare_control:
        treatment, control = compare_control(
            rows, read_rows(args.compare_control))
        print(
            "VALID terrain-comparison "
            f"treatment={treatment:.9g} control={control:.9g}")
    if args.gate_a:
        check_gate_a_contract(rows)
        report = diagnose_gate_a(rows)
        print(
            "VALID gate-a "
            f"first_positive={report['first_positive_query_frame']} "
            f"first_penetration={report['first_penetration_frame']} "
            f"classification={report['penetration_class']} "
            f"range={report['selected_range']} "
            f"hips_offset_y={report['hips_inertial_offset_y']:.9g} "
            f"adjust_y={report['adjustment_y']:.9g} "
            f"clamp_y={report['clamp_y']:.9g}")
    print(
        f"VALID runtime-log frames={summary['frames']} "
        f"transitions={summary['transitions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
