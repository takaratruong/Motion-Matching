#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import struct


GATE_A_COLUMNS = (
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
RUNTIME_SUFFIX = (
    "source_name", "source_terrain", "source_index", "continuation_cost",
    "source_root_height", "source_left_toe_height", "source_right_toe_height",
    "runtime_support_root_height", "runtime_support_left_toe_height",
    "runtime_support_right_toe_height", "support_root_delta",
    "support_left_toe_delta", "support_right_toe_delta", "support_height",
    "support_velocity", "support_source", "airborne_frames", "left_contact",
    "right_contact", "support_retargeted_hips_y", "ik_adjusted_hips_y",
    "simulation_x", "simulation_z", "walkability_class", "blocked",
    "blocked_reason", "blocked_distance", "blocked_point_x",
    "blocked_point_z", "commanded_speed", "applied_speed", "route_waypoint",
    "route_complete", "route_target_height", "scene_generation",
    "scene_frame", "scene_reset_count", "scene_switch_failed",
    "motion_pack_load_count", "model_load_count", "model_unload_count",
    "live_model_count",
)
RUNTIME_COLUMNS = list(GATE_A_COLUMNS) + list(RUNTIME_SUFFIX)

# Gate A callers keep their historical name and exact immutable tuple. Runtime
# readers accept the append-only suffix without weakening that prerequisite.
CSV_COLUMNS = GATE_A_COLUMNS
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
RUNTIME_TEXT_COLUMNS = {
    "source_name", "source_terrain", "support_source", "blocked_reason",
}
RUNTIME_INTEGER_COLUMNS = {
    "source_index", "airborne_frames", "left_contact", "right_contact",
    "walkability_class", "blocked", "route_waypoint", "route_complete",
    "scene_generation", "scene_frame", "scene_reset_count",
    "scene_switch_failed", "motion_pack_load_count", "model_load_count",
    "model_unload_count", "live_model_count",
}
RUNTIME_FLAG_COLUMNS = {
    "left_contact", "right_contact", "blocked", "route_complete",
    "scene_switch_failed",
}
RUNTIME_NONNEGATIVE_INTEGER_COLUMNS = RUNTIME_INTEGER_COLUMNS - {
    "left_contact", "right_contact", "blocked", "route_complete",
    "scene_switch_failed",
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


def _runtime_schema(rows):
    has_runtime = [
        any(name in row for name in RUNTIME_SUFFIX)
        for row in rows
    ]
    if any(has_runtime) and not all(has_runtime):
        raise ValueError("runtime suffix is present on only some rows")
    if not any(has_runtime):
        return False
    for index, row in enumerate(rows):
        missing = [name for name in RUNTIME_SUFFIX if name not in row]
        if missing:
            raise ValueError(
                f"row {index}: incomplete runtime suffix: {missing}")
    return True


def _require_runtime_header(rows):
    expected = tuple(RUNTIME_COLUMNS)
    for index, row in enumerate(rows):
        if tuple(row.keys()) != expected:
            raise ValueError(
                f"row {index}: runtime gate requires exact runtime header")


def _safe_runtime_text(row, name, index):
    value = row.get(name, "")
    if not value:
        raise ValueError(f"row {index}: empty {name}")
    if not value.isprintable() or any(character in value for character in ",\r\n"):
        raise ValueError(f"row {index}: unsafe {name}")
    return value


def _generation_groups(rows):
    if not rows or "scene_generation" not in rows[0]:
        return [rows]
    groups = []
    start = 0
    previous = _integer(rows[0], "scene_generation", 0)
    for index, row in enumerate(rows[1:], 1):
        generation = _integer(row, "scene_generation", index)
        if generation != previous:
            groups.append(rows[start:index])
            start = index
            previous = generation
    groups.append(rows[start:])
    return groups


def _check_substride_by_generation(rows):
    for group in _generation_groups(rows):
        check_substride(group)


def _longest_run(indices):
    best = []
    current = []
    for index in indices:
        if current and index != current[-1] + 1:
            if len(current) > len(best):
                best = current
            current = []
        current.append(index)
    if len(current) > len(best):
        best = current
    return best


def _support_alignment_error(row):
    support_height = float(row["support_height"])
    errors = []
    if int(row["left_contact"]):
        errors.append(abs(
            float(row["source_left_toe_height"]) + support_height -
            float(row["runtime_support_left_toe_height"])))
    if int(row["right_contact"]):
        errors.append(abs(
            float(row["source_right_toe_height"]) + support_height -
            float(row["runtime_support_right_toe_height"])))
    if errors:
        return max(errors)
    return abs(
        float(row["source_root_height"]) + support_height -
        float(row["runtime_support_root_height"]))


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
    runtime = _runtime_schema(rows)
    previous = None
    previous_range = None
    previous_generation = None
    previous_scene_frame = None
    previous_reset_count = None
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
        if runtime and struct.pack(">f", fixed_dt) != struct.pack(">f", 0.04):
            raise ValueError(f"row {index}: fixed_dt must be float32 0.04")
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
        generation_changed = False
        if runtime:
            generation = _integer(row, "scene_generation", index)
            scene_frame = _integer(row, "scene_frame", index)
            reset_count = _integer(row, "scene_reset_count", index)
            if index == 0:
                if scene_frame != 0:
                    raise ValueError(f"row {index}: initial scene_frame must be 0")
            elif generation == previous_generation:
                if scene_frame != previous_scene_frame + 1:
                    raise ValueError(
                        f"row {index}: scene_frame did not increment")
                if reset_count != previous_reset_count:
                    raise ValueError(
                        f"row {index}: scene_reset_count changed without reset")
            else:
                generation_changed = True
                if generation != previous_generation + 1:
                    raise ValueError(
                        f"row {index}: scene_generation must increment by one")
                if scene_frame != 0:
                    raise ValueError(
                        f"row {index}: scene_frame must be 0 after reset")
                if reset_count != previous_reset_count + 1:
                    raise ValueError(
                        f"row {index}: scene_reset_count must increment by one")
        if previous is not None and not generation_changed and query_frame != previous:
            raise ValueError(
                f"row {index}: query frame {query_frame} does not match "
                f"prior pose frame {previous}")
        if (previous_range is not None and not generation_changed and
                query_range != previous_range):
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
                route_may_be_empty = (
                    name == "route" and runtime and row.get("mode") != "route")
                if not row.get(name) and not route_may_be_empty:
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
        if runtime:
            for name in RUNTIME_SUFFIX:
                if name in RUNTIME_TEXT_COLUMNS:
                    _safe_runtime_text(row, name, index)
                elif name in RUNTIME_INTEGER_COLUMNS:
                    _integer(row, name, index)
                else:
                    _finite(row, name, index)
            _safe_runtime_text(row, "scene_id", index)
            if row.get("route"):
                _safe_runtime_text(row, "route", index)
            elif row.get("mode") == "route":
                raise ValueError(f"row {index}: empty route in route mode")
            for name in RUNTIME_FLAG_COLUMNS:
                if _integer(row, name, index) not in (0, 1):
                    raise ValueError(f"row {index}: {name} must be 0 or 1")
            for name in RUNTIME_NONNEGATIVE_INTEGER_COLUMNS:
                if _integer(row, name, index) < 0:
                    raise ValueError(f"row {index}: {name} must be nonnegative")
            if _integer(row, "walkability_class", index) not in (0, 1, 2):
                raise ValueError(
                    f"row {index}: walkability_class must be 0, 1, or 2")
            if _integer(row, "ik_enabled", index) != 0:
                raise ValueError(f"row {index}: IK must be disabled")
            for name in ("adjustment_y", "clamp_y"):
                if _finite(row, name, index) != 0.0:
                    raise ValueError(f"row {index}: {name} must be zero")
            if _integer(row, "motion_pack_load_count", index) != 1:
                raise ValueError(
                    f"row {index}: motion pack load count must be 1")
            if _integer(row, "live_model_count", index) != 1:
                raise ValueError(f"row {index}: live model count must be 1")
        _check_query_snapshot(row, index)
        if (previous is not None and not generation_changed and
                not transitioned and current != previous + 1):
            raise ValueError(
                f"row {index}: nonsequential advance {previous}->{current}")
        if (previous_range is not None and not generation_changed and
                not transitioned and
                current_range != previous_range):
            raise ValueError(
                f"row {index}: range change without transition")
        previous = current
        previous_range = current_range
        if runtime:
            previous_generation = generation
            previous_scene_frame = scene_frame
            previous_reset_count = reset_count
    _check_substride_by_generation(rows)
    return {
        "frames": len(rows),
        "transitions": sum(int(row["transitioned"]) for row in rows),
    }


def _check_route_gate_contract(rows, gate_name):
    _require_runtime_header(rows)
    summary = check_rows(rows)
    if any(row["mode"] != "route" for row in rows):
        raise ValueError(f"{gate_name} requires route mode")
    routes = {(row["scene_id"], row["route"]) for row in rows}
    if len(routes) != 1:
        raise ValueError(f"{gate_name} requires one scene and route")
    if any(_integer(row, "matching_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError(f"{gate_name} matching must remain enabled")
    if any(_integer(row, "support_retargeting_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError(f"{gate_name} support retargeting must remain enabled")
    if any(_integer(row, "ik_enabled", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError(f"{gate_name} IK must remain disabled")
    return summary, next(iter(routes))


def check_mixed_multilevel(rows):
    _require_runtime_header(rows)
    check_rows(rows)
    if {(row["scene_id"], row["route"]) for row in rows} != {
            ("mixed-multilevel", "full-course")}:
        raise ValueError(
            "mixed multilevel checker requires mixed-multilevel/full-course")

    samples = []
    for index, row in enumerate(rows):
        x = _finite(row, "simulation_x", index)
        z = _finite(row, "simulation_z", index)
        height = _finite(row, "runtime_support_root_height", index)
        samples.append((index, x, z, height))

    elevated_start = (0.0, 3.20)
    elevated_end = (0.0, 6.20)

    def endpoint_crossings(endpoint):
        return [
            index for index, x, z, _ in samples
            if math.hypot(x - endpoint[0], z - endpoint[1]) <= 0.05
        ]

    start_crossings = endpoint_crossings(elevated_start)
    end_crossings = endpoint_crossings(elevated_end)
    if (not start_crossings or not end_crossings or
            not any(start < end
                    for start in start_crossings for end in end_crossings)):
        raise ValueError(
            "mixed endpoint crossing did not reach start then end")

    elevated_samples = [
        sample for sample in samples
        if 3.20 - 0.05 <= sample[2] <= 6.20 + 0.05 and
        abs(sample[3] - 0.32) <= 0.02
    ]
    if not elevated_samples:
        raise ValueError("mixed elevated support span has no samples")
    elevated_span = max(sample[2] for sample in elevated_samples) - min(
        sample[2] for sample in elevated_samples)
    if elevated_span < 2.95:
        raise ValueError(
            "mixed elevated support span is shorter than 2.95 m")

    valid_elevated = []
    for index, _, z, height in samples:
        row = rows[index]
        if not 3.20 <= z <= 6.20 or abs(height - 0.32) > 0.02:
            continue
        if (_integer(row, "matching_enabled", index) == 1 and
                _support_alignment_error(row) <= 0.02):
            valid_elevated.append(index)
    elevated_run = _longest_run(valid_elevated)
    if len(elevated_run) < 50:
        raise ValueError(
            "mixed elevated matching/alignment did not persist for 50 frames")

    plateau_definitions = (
        (6.20, 6.80, 0.40),
        (6.80, 7.40, 0.28),
        (7.40, 8.00, 0.32),
    )
    plateau_samples = []
    plateau_medians = []
    for start, end, expected in plateau_definitions:
        region = [sample for sample in samples if start < sample[2] < end]
        if not region:
            raise ValueError("mixed block plateaus have missing interior samples")
        plateau_samples.append(region)
        median = statistics.median(sample[3] for sample in region)
        plateau_medians.append(median)
        if abs(median - expected) > 0.02:
            raise ValueError(
                "mixed block plateaus do not match 0.40/0.28/0.32 m")
    if not (max(sample[0] for sample in plateau_samples[0]) <
            min(sample[0] for sample in plateau_samples[1]) and
            max(sample[0] for sample in plateau_samples[1]) <
            min(sample[0] for sample in plateau_samples[2])):
        raise ValueError("mixed block plateaus are reordered")

    ramp = [sample for sample in samples if 8.00 <= sample[2] <= 9.8148]
    if not ramp:
        raise ValueError("mixed return ramp has no samples")
    ramp_heights = [sample[3] for sample in ramp]
    if (ramp_heights[0] < 0.27 or ramp_heights[-1] > 0.05 or
            max(ramp_heights) < 0.27 or min(ramp_heights) > 0.05):
        raise ValueError("mixed return ramp did not span elevated to base")
    if any(right - left > 0.03
           for left, right in zip(ramp_heights, ramp_heights[1:])):
        raise ValueError("mixed return ramp rose by more than 0.03 m")
    if any(_integer(rows[index], "matching_enabled", index) != 1
           for index, _, _, _ in ramp):
        raise ValueError("mixed return ramp lost matching")

    base_indices = [
        index for index, _, z, height in samples
        if z > 9.8148 and abs(height) <= 0.02
    ]
    base_run = _longest_run(base_indices)
    if len(base_run) < 25 or base_run[0] <= ramp[-1][0]:
        raise ValueError("mixed course never returned to base for 25 frames")

    return {
        "elevated_span_m": elevated_span,
        "elevated_matching_frames": len(elevated_run),
        "plateau_1_m": plateau_medians[0],
        "plateau_2_m": plateau_medians[1],
        "plateau_3_m": plateau_medians[2],
        "return_ramp_drop_m": max(ramp_heights) - min(ramp_heights),
        "base_frames": len(base_run),
    }


def check_gate_c(rows):
    summary, route = _check_route_gate_contract(rows, "Gate C")
    if len(rows) < 20:
        raise ValueError("Gate C requires at least 20 baseline rows")
    baseline = statistics.median(
        _finite(row, "runtime_support_root_height", index)
        for index, row in enumerate(rows[:20]))
    activation = next((
        index for index, row in enumerate(rows)
        if max(abs(_finite(row, f"terrain{sample}", index))
               for sample in range(4)) > 0.02
    ), None)
    rise = next((
        index for index, row in enumerate(rows)
        if _finite(row, "runtime_support_root_height", index) > baseline + 0.04
    ), None)
    if activation is None or rise is None or activation >= rise:
        raise ValueError(
            "Gate C terrain activation must precede root support rise")
    if not any(
            _integer(row, "source_index", index) > 0 and
            row["source_terrain"].lower() != "flat"
            for index, row in enumerate(rows[activation:], activation)):
        raise ValueError(
            "Gate C terrain source did not activate after terrain query")

    maximum_rendered_step = 0.0
    for index in range(1, len(rows)):
        step = abs(
            _finite(rows[index], "rendered_hips_y", index) -
            _finite(rows[index - 1], "rendered_hips_y", index - 1))
        maximum_rendered_step = max(maximum_rendered_step, step)
        if step > 0.05:
            raise ValueError(
                f"row {index}: rendered Hips step exceeds 0.05 m")
    for index, row in enumerate(rows):
        support_y = _finite(row, "support_retargeted_hips_y", index)
        rendered_y = _finite(row, "rendered_hips_y", index)
        ik_y = _finite(row, "ik_adjusted_hips_y", index)
        if max(support_y, rendered_y, ik_y) - min(
                support_y, rendered_y, ik_y) > 1e-6:
            raise ValueError(f"row {index}: Hips stage agreement exceeded 1e-6")

    landing_indices = []
    landing_errors = {}
    for index, row in enumerate(rows):
        height_error = abs(
            _finite(row, "runtime_support_root_height", index) -
            _finite(row, "route_target_height", index))
        support_error = _support_alignment_error(row)
        if height_error <= 0.02 and support_error <= 0.02:
            landing_indices.append(index)
            landing_errors[index] = support_error
    landing = _longest_run(landing_indices)
    if len(landing) < 50:
        raise ValueError("Gate C landing block is shorter than 50 frames")
    support_errors = [landing_errors[index] for index in landing]
    if not any(
            abs(_finite(rows[index], "runtime_support_root_height", index) -
                baseline) <= 0.02
            for index in range(landing[-1] + 1, len(rows))):
        raise ValueError("Gate C did not return to baseline after landing block")

    report = {
        **summary,
        "landing_frames": len(landing),
        "maximum_support_error": max(support_errors),
        "maximum_rendered_hips_step": maximum_rendered_step,
    }
    if route == ("mixed-multilevel", "full-course"):
        report.update(check_mixed_multilevel(rows))
    return report


def check_gate_d(rows):
    summary, route = _check_route_gate_contract(rows, "Gate D")
    if route[0] != "blocked-course" or route[1] not in {
            "wall-safe-stop", "ramp-safe-stop"}:
        raise ValueError("Gate D requires a blocked-course safe-stop route")
    if any(_integer(row, "walkability_class", index) == 0
           for index, row in enumerate(rows)):
        raise ValueError("Gate D entered blocked walkability footprint")
    blocked = [
        index for index, row in enumerate(rows)
        if _integer(row, "blocked", index) == 1
    ]
    if not blocked:
        raise ValueError("Gate D never reported blocked")
    stopped_blocked = [
        index for index in blocked
        if _finite(rows[index], "applied_speed", index) <= 1e-4
    ]
    if not stopped_blocked:
        raise ValueError("Gate D never stopped while blocked")
    first_stopped = stopped_blocked[0]
    stopped_tail = [index for index in blocked if index >= first_stopped]
    clearances = [
        _finite(rows[index], "blocked_distance", index)
        for index in stopped_tail
    ]
    minimum_clearance = min(clearances)
    if minimum_clearance < 0.02 - 1e-4:
        raise ValueError(
            "Gate D stopped clearance fell below 0.02 m")
    stopped_run = []
    for index in range(first_stopped, len(rows)):
        if not (
                _finite(rows[index], "applied_speed", index) <= 1e-4 and
                (_finite(rows[index], "commanded_speed", index) > 1e-4 or
                 _integer(rows[index], "route_complete", index) == 1)):
            break
        stopped_run.append(index)
    if len(stopped_run) < 25:
        raise ValueError(
            "Gate D did not hold 25 consecutive stopped frames")

    first_blocked = blocked[0]
    pre_block = rows[max(0, first_blocked - 20):first_blocked]
    if len(pre_block) != 20:
        raise ValueError(
            "Gate D requires 20 pre-block support baseline rows")
    pre_support = max(
        float(row["source_root_height"]) + float(row["support_height"])
        for row in pre_block)
    blocked_support = max(
        _finite(row, "source_root_height", index) +
        _finite(row, "support_height", index)
        for index, row in enumerate(rows[first_blocked:], first_blocked))
    blocked_support_rise = max(0.0, blocked_support - pre_support)
    if blocked_support_rise > 0.02:
        raise ValueError("Gate D blocked support rise exceeds 0.02 m")
    check_substride(rows[first_stopped:])
    return {
        **summary,
        "blocked_frames": len(blocked),
        "stopped_frames": len(stopped_run),
        "minimum_clearance": minimum_clearance,
        "maximum_blocked_support_rise": blocked_support_rise,
    }


def compare_control(treatment, control):
    treatment_runtime = _runtime_schema(treatment)
    control_runtime = _runtime_schema(control)
    if treatment_runtime != control_runtime:
        raise ValueError("control and treatment runtime schemas differ")
    if treatment_runtime:
        _require_runtime_header(treatment)
        _require_runtime_header(control)
    check_rows(treatment)
    check_rows(control)
    if len(treatment) != len(control):
        raise ValueError("control and treatment lengths differ")
    metadata = ["frame", "fixed_dt", "scene_id", "mode", "route"]
    if treatment_runtime:
        metadata.extend((
            "scene_generation", "scene_frame", "route_waypoint",
            "route_complete", "commanded_speed",
        ))
    for index, (treatment_row, control_row) in enumerate(
            zip(treatment, control)):
        if any(treatment_row[name] != control_row[name] for name in metadata):
            raise ValueError(
                f"row {index}: control and treatment scripted input "
                "(script metadata) differ")
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
    _check_substride_by_generation(treatment)
    _check_substride_by_generation(control)
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


def _print_gate_report(label, report):
    fields = []
    for name in sorted(report):
        value = report[name]
        rendered = f"{value:.9g}" if isinstance(value, float) else str(value)
        fields.append(f"{name}={rendered}")
    print(f"VALID {label} " + " ".join(fields))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--compare-control")
    parser.add_argument("--gate-a", action="store_true")
    runtime_gate = parser.add_mutually_exclusive_group()
    runtime_gate.add_argument("--gate-c", action="store_true")
    runtime_gate.add_argument("--gate-d", action="store_true")
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
    if args.gate_c:
        _print_gate_report("gate-c", check_gate_c(rows))
    if args.gate_d:
        _print_gate_report("gate-d", check_gate_d(rows))
    print(
        f"VALID runtime-log frames={summary['frames']} "
        f"transitions={summary['transitions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
