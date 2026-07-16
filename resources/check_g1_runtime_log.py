#!/usr/bin/env python3
import argparse
import csv
import hashlib
import math
import os
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
LEGACY_RUNTIME_SUFFIX = (
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


def _columns(text):
    return tuple(name.strip() for name in text.split(",") if name.strip())


IK_SUFFIX = _columns("""
ik_applied,ik_safe_stop_requested,ik_stop_reason,max_ik_correction,
actual_simulation_speed,ik_candidate_rejected,ik_candidate_clearance_status,
left_candidate_toe_clearance,left_candidate_foot_clearance,
right_candidate_toe_clearance,right_candidate_foot_clearance,
ik_candidate_minimum_clearance,
left_recorded_contact,right_recorded_contact,left_locked,right_locked,
left_observed_lock_drift,right_observed_lock_drift,
left_lock_drift,right_lock_drift,
left_sole_normal_alignment,right_sole_normal_alignment,
left_contact_residual,right_contact_residual,
left_target_height,right_target_height,
left_target_normal_x,left_target_normal_y,left_target_normal_z,
right_target_normal_x,right_target_normal_y,right_target_normal_z,
left_swing_candidates_evaluated,left_swing_selected_index,
left_swing_selected_lift_bits,left_swing_materialized_command_y_bits,
left_swing_actual_sphere_center_bits_hex,
left_swing_selected_clearance_status,
left_swing_selected_controller_constraints_passed,
left_swing_selected_clearance_certified,
left_swing_lower_margin,left_swing_witness_upper_margin,
left_swing_selected_work_point_queries,left_swing_selected_work_cells_visited,
left_swing_selected_work_primitive_triangle_pairs,
left_swing_selected_work_face_patches,left_swing_selected_work_candidate_tests,
left_swing_selected_work_subdivision_nodes,
left_swing_total_work_point_queries,left_swing_total_work_cells_visited,
left_swing_total_work_primitive_triangle_pairs,
left_swing_total_work_face_patches,left_swing_total_work_candidate_tests,
left_swing_total_work_subdivision_nodes,
right_swing_candidates_evaluated,right_swing_selected_index,
right_swing_selected_lift_bits,right_swing_materialized_command_y_bits,
right_swing_actual_sphere_center_bits_hex,
right_swing_selected_clearance_status,
right_swing_selected_controller_constraints_passed,
right_swing_selected_clearance_certified,
right_swing_lower_margin,right_swing_witness_upper_margin,
right_swing_selected_work_point_queries,right_swing_selected_work_cells_visited,
right_swing_selected_work_primitive_triangle_pairs,
right_swing_selected_work_face_patches,right_swing_selected_work_candidate_tests,
right_swing_selected_work_subdivision_nodes,
right_swing_total_work_point_queries,right_swing_total_work_cells_visited,
right_swing_total_work_primitive_triangle_pairs,
right_swing_total_work_face_patches,right_swing_total_work_candidate_tests,
right_swing_total_work_subdivision_nodes,
left_reachable,right_reachable,left_knee_clearance,left_ankle_clearance,
left_toe_clearance,left_foot_clearance,left_shin_clearance,
left_thigh_clearance,right_knee_clearance,right_ankle_clearance,
right_toe_clearance,right_foot_clearance,right_shin_clearance,
right_thigh_clearance,ik_hips_clearance,ik_minimum_clearance
""")

DIRECTIONAL_SUFFIX = _columns("""
requested_velocity_x,requested_velocity_y,requested_velocity_z,
applied_velocity_x,applied_velocity_y,applied_velocity_z,
desired_heading_bits_hex,predicted_heading_bits_hex,
simulation_heading_error_deg,rendered_heading_error_deg,
footprint_status,footprint_blocked,footprint_blocked_reason,
footprint_root_height,
left_footprint_min_height,left_footprint_max_height,
right_footprint_min_height,right_footprint_max_height,
left_maximum_root_split,right_maximum_root_split,
left_footprint_multilevel,right_footprint_multilevel,
left_landing_expected,left_landing_patch_ready,left_landing_sample,
left_landing_surface_status,left_landing_walkability_class,
left_predicted_landing_center_x,left_predicted_landing_center_y,
left_predicted_landing_center_z,left_predicted_landing_height,
left_predicted_landing_normal_x,left_predicted_landing_normal_y,
left_predicted_landing_normal_z,left_landing_patch_maximum_residual,
right_landing_expected,right_landing_patch_ready,right_landing_sample,
right_landing_surface_status,right_landing_walkability_class,
right_predicted_landing_center_x,right_predicted_landing_center_y,
right_predicted_landing_center_z,right_predicted_landing_height,
right_predicted_landing_normal_x,right_predicted_landing_normal_y,
right_predicted_landing_normal_z,right_landing_patch_maximum_residual,
footprint_sweeps,footprint_surface_queries,footprint_node_visits,
frame_rejected,frame_rejection_stage,ik_safe_stop_latched,
rejected_attempted_footprint_available,rejected_attempted_ik_available,
rejected_stop_reason,rejected_attempted_pose_available,rejected_pose_status,
rejected_pose_minimum_clearance,
rejected_left_landing_expected,rejected_left_landing_patch_ready,
rejected_left_landing_sample,rejected_left_landing_center_x,
rejected_left_landing_center_y,rejected_left_landing_center_z,
rejected_left_landing_surface_status,rejected_left_landing_surface_height,
rejected_left_landing_surface_normal_x,
rejected_left_landing_surface_normal_y,
rejected_left_landing_surface_normal_z,
rejected_left_landing_walkability_class,
rejected_left_landing_patch_maximum_residual,
rejected_left_target_x,rejected_left_target_y,rejected_left_target_z,
rejected_left_target_normal_x,rejected_left_target_normal_y,
rejected_left_target_normal_z,rejected_left_reachable,
rejected_left_correction_limited,rejected_left_selected_clearance_status,
rejected_left_selected_lower_margin,rejected_left_selected_witness_upper,
rejected_right_landing_expected,rejected_right_landing_patch_ready,
rejected_right_landing_sample,rejected_right_landing_center_x,
rejected_right_landing_center_y,rejected_right_landing_center_z,
rejected_right_landing_surface_status,rejected_right_landing_surface_height,
rejected_right_landing_surface_normal_x,
rejected_right_landing_surface_normal_y,
rejected_right_landing_surface_normal_z,
rejected_right_landing_walkability_class,
rejected_right_landing_patch_maximum_residual,
rejected_right_target_x,rejected_right_target_y,rejected_right_target_z,
rejected_right_target_normal_x,rejected_right_target_normal_y,
rejected_right_target_normal_z,rejected_right_reachable,
rejected_right_correction_limited,rejected_right_selected_clearance_status,
rejected_right_selected_lower_margin,rejected_right_selected_witness_upper,
accepted_state_digest_hex
""")

RUNTIME_SUFFIX = LEGACY_RUNTIME_SUFFIX + IK_SUFFIX + DIRECTIONAL_SUFFIX
RUNTIME_COLUMNS = list(GATE_A_COLUMNS) + list(RUNTIME_SUFFIX)
LEGACY_RUNTIME_COLUMNS = tuple(GATE_A_COLUMNS) + LEGACY_RUNTIME_SUFFIX
LOCKED_SCENE_IDS = (
    "grail-curb-default",
    "grail-curb-low",
    "grail-curb-medium",
    "grail-curb-high",
    "stairs-shallow",
    "stairs-standard",
    "stairs-unseen-variable",
    "ramp-05-up-down",
    "ramp-10-up-down",
    "ramp-15-stress",
    "cross-slope-05",
    "cross-slope-10",
    "mixed-multilevel",
    "blocked-course",
)
HEADING_COMPONENTS = {
    "forward": (1.0, 0.0, 0.0, 0.0),
    "backward": (0.0, 0.0, 1.0, 0.0),
    "positive-x": (0.707106769, 0.0, 0.707106769, 0.0),
    "negative-x": (0.707106769, 0.0, -0.707106769, 0.0),
    "diagonal-positive-x": (0.923879504, 0.0, 0.382683426, 0.0),
    "diagonal-negative-x": (0.923879504, 0.0, -0.382683426, 0.0),
}
HEADING_BITS = {
    name: "".join(struct.pack(">f", component).hex()
                  for component in components)
    for name, components in HEADING_COMPONENTS.items()
}
FLAT_ORACLE_CASES = {
    "flat-flat-positive-z__forward__forward.csv":
        ("stairs-shallow", "flat-positive-z", "forward", "forward"),
    "flat-flat-positive-z__backward__backward.csv":
        ("stairs-shallow", "flat-positive-z", "backward", "backward"),
    "flat-flat-positive-z__positive-x__left.csv":
        ("stairs-shallow", "flat-positive-z", "positive-x", "left"),
    "flat-flat-positive-z__negative-x__right.csv":
        ("stairs-shallow", "flat-positive-z", "negative-x", "right"),
    "flat-flat-positive-x__positive-x__forward.csv":
        ("stairs-shallow", "flat-positive-x", "positive-x", "forward"),
    "flat-flat-positive-x__negative-x__backward.csv":
        ("stairs-shallow", "flat-positive-x", "negative-x", "backward"),
    "flat-flat-positive-x__forward__right.csv":
        ("stairs-shallow", "flat-positive-x", "forward", "right"),
    "flat-flat-positive-x__backward__left.csv":
        ("stairs-shallow", "flat-positive-x", "backward", "left"),
}
FORWARD_ORACLE_CASES = {
    "forward-stairs-shallow__ascent-landing-descent.csv":
        ("stairs-shallow", "ascent-landing-descent"),
    "forward-stairs-standard__ascent-landing-descent.csv":
        ("stairs-standard", "ascent-landing-descent"),
    "forward-grail-curb-low__curb-forward.csv":
        ("grail-curb-low", "curb-forward"),
    "forward-ramp-05-up-down__up-landing-down.csv":
        ("ramp-05-up-down", "up-landing-down"),
    "forward-ramp-10-up-down__up-landing-down.csv":
        ("ramp-10-up-down", "up-landing-down"),
}
ORACLE_RECORD_NAMES = (
    "controller", *FLAT_ORACLE_CASES, *FORWARD_ORACLE_CASES)
CLEARANCE_STATUSES = {
    "ok", "outside-domain", "budget-exceeded", "uncertified",
    "invalid-input", "invalid-field", "arithmetic-failure",
}
FOOTPRINT_STATUSES = {
    "ok", "outside-domain", "budget-exceeded", "invalid-input",
    "invalid-field", "arithmetic-failure",
}
SURFACE_STATUSES = {"valid", "outside", "invalid"}
IK_STOP_REASONS = {
    "none", "footprint-blocked", "footprint-outside-domain",
    "footprint-budget-exceeded", "landing-patch-unavailable",
    "target-unreachable", "no-swing-candidate",
    "pose-clearance-rejected",
}
REJECTION_STAGES = {
    "none", "footprint", "landing-patch", "ik-candidate",
    "pose-certificate",
}
WALKABILITY_REASONS = {
    "clear", "blocked-cell", "out-of-bounds", "nonfinite",
}
LANDING_PATCH_RESIDUAL_LIMIT = struct.unpack(
    ">f", bytes.fromhex("3ba3d70a"))[0]

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
    "ik_stop_reason", "ik_candidate_clearance_status",
    "left_swing_actual_sphere_center_bits_hex",
    "left_swing_selected_clearance_status",
    "right_swing_actual_sphere_center_bits_hex",
    "right_swing_selected_clearance_status",
    "desired_heading_bits_hex", "predicted_heading_bits_hex",
    "footprint_status", "footprint_blocked_reason",
    "left_landing_surface_status", "right_landing_surface_status",
    "frame_rejection_stage", "rejected_stop_reason",
    "rejected_pose_status",
    "rejected_left_landing_surface_status",
    "rejected_left_selected_clearance_status",
    "rejected_right_landing_surface_status",
    "rejected_right_selected_clearance_status",
    "accepted_state_digest_hex",
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
    "ik_applied", "ik_safe_stop_requested", "ik_candidate_rejected",
    "left_recorded_contact", "right_recorded_contact",
    "left_locked", "right_locked",
    "left_swing_selected_controller_constraints_passed",
    "left_swing_selected_clearance_certified",
    "right_swing_selected_controller_constraints_passed",
    "right_swing_selected_clearance_certified",
    "left_reachable", "right_reachable", "footprint_blocked",
    "left_footprint_multilevel", "right_footprint_multilevel",
    "left_landing_expected", "left_landing_patch_ready",
    "right_landing_expected", "right_landing_patch_ready",
    "frame_rejected", "ik_safe_stop_latched",
    "rejected_attempted_footprint_available",
    "rejected_attempted_ik_available",
    "rejected_attempted_pose_available",
    "rejected_left_landing_expected",
    "rejected_left_landing_patch_ready", "rejected_left_reachable",
    "rejected_left_correction_limited",
    "rejected_right_landing_expected",
    "rejected_right_landing_patch_ready", "rejected_right_reachable",
    "rejected_right_correction_limited",
}
RUNTIME_UINT32_COLUMNS = {
    "left_swing_candidates_evaluated", "left_swing_selected_index",
    "left_swing_selected_lift_bits",
    "left_swing_materialized_command_y_bits",
    "left_swing_selected_work_point_queries",
    "left_swing_selected_work_cells_visited",
    "left_swing_selected_work_primitive_triangle_pairs",
    "left_swing_selected_work_face_patches",
    "left_swing_selected_work_candidate_tests",
    "left_swing_selected_work_subdivision_nodes",
    "left_swing_total_work_point_queries",
    "left_swing_total_work_cells_visited",
    "left_swing_total_work_primitive_triangle_pairs",
    "left_swing_total_work_face_patches",
    "left_swing_total_work_candidate_tests",
    "left_swing_total_work_subdivision_nodes",
    "right_swing_candidates_evaluated", "right_swing_selected_index",
    "right_swing_selected_lift_bits",
    "right_swing_materialized_command_y_bits",
    "right_swing_selected_work_point_queries",
    "right_swing_selected_work_cells_visited",
    "right_swing_selected_work_primitive_triangle_pairs",
    "right_swing_selected_work_face_patches",
    "right_swing_selected_work_candidate_tests",
    "right_swing_selected_work_subdivision_nodes",
    "right_swing_total_work_point_queries",
    "right_swing_total_work_cells_visited",
    "right_swing_total_work_primitive_triangle_pairs",
    "right_swing_total_work_face_patches",
    "right_swing_total_work_candidate_tests",
    "right_swing_total_work_subdivision_nodes",
    "left_landing_sample", "right_landing_sample",
    "footprint_sweeps", "footprint_surface_queries",
    "footprint_node_visits", "rejected_left_landing_sample",
    "rejected_right_landing_sample",
}
RUNTIME_INTEGER_COLUMNS |= RUNTIME_FLAG_COLUMNS | {
    "left_landing_walkability_class", "right_landing_walkability_class",
    "rejected_left_landing_walkability_class",
    "rejected_right_landing_walkability_class",
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


def _verify_oracle_sha256(path):
    absolute = os.path.abspath(path)
    directory = os.path.dirname(absolute)
    basename = os.path.basename(absolute)
    if basename not in ORACLE_RECORD_NAMES:
        raise ValueError("baseline is not a locked Task-3 oracle name")
    record_path = os.path.join(directory, "SHA256SUMS")
    try:
        with open(record_path, "r", encoding="ascii", newline="") as stream:
            lines = stream.read().splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read Task-3 oracle SHA256 record") from error
    observed = {}
    for line in lines:
        if (len(line) < 67 or line[64:66] != "  " or
                any(character not in "0123456789abcdef" for character in line[:64])):
            raise ValueError("malformed Task-3 oracle SHA256 record")
        name = line[66:]
        if (not name or name in observed or os.path.basename(name) != name or
                name in {".", ".."}):
            raise ValueError("malformed or duplicate Task-3 oracle record entry")
        observed[name] = line[:64]
    if set(observed) != set(ORACLE_RECORD_NAMES):
        raise ValueError("Task-3 oracle record does not contain the exact corpus")
    for name in ORACLE_RECORD_NAMES:
        payload_path = os.path.join(directory, name)
        try:
            with open(payload_path, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise ValueError(
                f"cannot read Task-3 oracle payload {name}") from error
        if digest != observed[name]:
            raise ValueError(f"Task-3 oracle SHA256 mismatch for {name}")
    return absolute


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


def _uint32(row, name, index):
    value = _finite(row, name, index)
    if value != int(value):
        raise ValueError(f"row {index}: non-integer {name}")
    integer = int(value)
    if integer < 0 or integer > 2 ** 32 - 1:
        raise ValueError(f"row {index}: {name} is outside uint32")
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
    has_legacy = [
        any(name in row for name in LEGACY_RUNTIME_SUFFIX)
        for row in rows]
    has_full = [
        any(name in row for name in IK_SUFFIX + DIRECTIONAL_SUFFIX)
        for row in rows]
    if ((any(has_legacy) and not all(has_legacy)) or
            (any(has_full) and not all(has_full))):
        raise ValueError("runtime suffix is present on only some rows")
    if not any(has_legacy) and not any(has_full):
        return None
    expected = RUNTIME_SUFFIX if any(has_full) else LEGACY_RUNTIME_SUFFIX
    for index, row in enumerate(rows):
        missing = [name for name in expected if name not in row]
        if missing:
            raise ValueError(
                f"row {index}: incomplete runtime suffix: {missing}")
    return "full" if any(has_full) else "legacy"


def _require_runtime_header(rows):
    expected = {tuple(RUNTIME_COLUMNS), LEGACY_RUNTIME_COLUMNS}
    for index, row in enumerate(rows):
        if tuple(row.keys()) not in expected:
            raise ValueError(
                f"row {index}: runtime gate requires exact runtime header")


def _require_full_runtime_header(rows):
    expected = tuple(RUNTIME_COLUMNS)
    for index, row in enumerate(rows):
        if tuple(row.keys()) != expected:
            raise ValueError(
                f"row {index}: Gate L requires exact full runtime header")


def _safe_runtime_text(row, name, index):
    value = row.get(name, "")
    if not value:
        raise ValueError(f"row {index}: empty {name}")
    if not value.isprintable() or any(character in value for character in ",\r\n"):
        raise ValueError(f"row {index}: unsafe {name}")
    return value


def _hex_float_words(row, name, index, count):
    value = _safe_runtime_text(row, name, index)
    if (len(value) != count * 8 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(
            f"row {index}: {name} is not {count} lowercase float words")
    output = []
    for word in range(count):
        component = struct.unpack(
            ">f", bytes.fromhex(value[word * 8:(word + 1) * 8]))[0]
        if not math.isfinite(component):
            raise ValueError(f"row {index}: non-finite {name} word {word}")
        output.append(component)
    return tuple(output)


def _hex_digest(row, name, index):
    value = _safe_runtime_text(row, name, index)
    if (len(value) != 16 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(
            f"row {index}: {name} is not one lowercase uint64 word")
    return value


def _require_enum(row, name, index, values):
    value = _safe_runtime_text(row, name, index)
    if value not in values:
        raise ValueError(f"row {index}: {name} has unknown value {value}")
    return value


def _check_landing_schema(row, index, prefix):
    expected = _integer(row, f"{prefix}_landing_expected", index)
    ready = _integer(row, f"{prefix}_landing_patch_ready", index)
    sample = _uint32(row, f"{prefix}_landing_sample", index)
    status = _require_enum(
        row, f"{prefix}_landing_surface_status", index, SURFACE_STATUSES)
    walkability = _integer(
        row, f"{prefix}_landing_walkability_class", index)
    residual = _finite(
        row, f"{prefix}_landing_patch_maximum_residual", index)
    if residual < 0.0:
        raise ValueError(
            f"row {index}: {prefix} landing residual must be nonnegative")
    if not expected:
        canonical_scalars = (
            f"{prefix}_predicted_landing_center_x",
            f"{prefix}_predicted_landing_center_y",
            f"{prefix}_predicted_landing_center_z",
            f"{prefix}_predicted_landing_height",
            f"{prefix}_predicted_landing_normal_x",
            f"{prefix}_predicted_landing_normal_y",
            f"{prefix}_predicted_landing_normal_z",
        )
        if (ready or sample != 2 ** 32 - 1 or status != "invalid" or
                walkability != 0 or residual != 0.0 or
                any(_finite(row, name, index) != 0.0
                    for name in canonical_scalars)):
            raise ValueError(
                f"row {index}: absent {prefix} landing is not canonical")
    else:
        if (sample == 0 or sample >= 4 or status != "valid" or
                walkability not in {1, 2}):
            raise ValueError(
                f"row {index}: expected {prefix} landing is malformed")
        if _integer(row, f"{prefix}_recorded_contact", index) != 0:
            raise ValueError(
                f"row {index}: expected {prefix} landing is not a swing foot")
        if ready and (
                status != "valid" or walkability != 1 or
                residual > LANDING_PATCH_RESIDUAL_LIMIT):
            raise ValueError(
                f"row {index}: ready {prefix} landing patch is inconsistent")


def _check_rejected_landing_schema(row, index, foot):
    prefix = f"rejected_{foot}"
    expected = _integer(row, f"{prefix}_landing_expected", index)
    ready = _integer(row, f"{prefix}_landing_patch_ready", index)
    sample = _uint32(row, f"{prefix}_landing_sample", index)
    status = _require_enum(
        row, f"{prefix}_landing_surface_status", index, SURFACE_STATUSES)
    walkability = _integer(
        row, f"{prefix}_landing_walkability_class", index)
    residual = _finite(
        row, f"{prefix}_landing_patch_maximum_residual", index)
    if residual < 0.0:
        raise ValueError(
            f"row {index}: rejected {foot} landing residual is negative")
    if not expected:
        canonical_scalars = (
            f"{prefix}_landing_center_x",
            f"{prefix}_landing_center_y",
            f"{prefix}_landing_center_z",
            f"{prefix}_landing_surface_height",
            f"{prefix}_landing_surface_normal_x",
            f"{prefix}_landing_surface_normal_y",
            f"{prefix}_landing_surface_normal_z",
        )
        if (ready or sample != 2 ** 32 - 1 or status != "invalid" or
                walkability != 0 or residual != 0.0 or
                any(_finite(row, name, index) != 0.0
                    for name in canonical_scalars)):
            raise ValueError(
                f"row {index}: absent rejected {foot} landing is not canonical")
    else:
        if (sample == 0 or sample >= 4 or status != "valid" or
                walkability not in {1, 2}):
            raise ValueError(
                f"row {index}: rejected {foot} landing is malformed")
        if _integer(row, f"{foot}_recorded_contact", index) != 0:
            raise ValueError(
                f"row {index}: rejected {foot} landing is not a swing foot")
        if ready and (
                status != "valid" or walkability != 1 or
                residual > LANDING_PATCH_RESIDUAL_LIMIT):
            raise ValueError(
                f"row {index}: rejected {foot} ready patch is inconsistent")


def _check_invalid_footprint_is_canonical(row, index):
    scalar_names = (
        "footprint_root_height",
        "left_footprint_min_height", "left_footprint_max_height",
        "right_footprint_min_height", "right_footprint_max_height",
        "left_maximum_root_split", "right_maximum_root_split",
    )
    flag_names = (
        "footprint_blocked", "left_footprint_multilevel",
        "right_footprint_multilevel",
    )
    counter_names = (
        "footprint_sweeps", "footprint_surface_queries",
        "footprint_node_visits",
    )
    if (row["footprint_blocked_reason"] != "clear" or
            any(_finite(row, name, index) != 0.0 for name in scalar_names) or
            any(_integer(row, name, index) != 0 for name in flag_names) or
            any(_uint32(row, name, index) != 0 for name in counter_names) or
            any(_integer(row, f"{foot}_landing_expected", index) != 0
                for foot in ("left", "right"))):
        raise ValueError(
            f"row {index}: invalid-input footprint output is not canonical")


def _check_accepted_status_schema(row, index):
    if _integer(row, "ik_safe_stop_requested", index) != 0:
        raise ValueError(
            f"row {index}: accepted IK state requested a safe-stop")
    if row["ik_stop_reason"] != "none":
        raise ValueError(
            f"row {index}: accepted IK stop reason is not none")
    if _integer(row, "ik_candidate_rejected", index) != 0:
        raise ValueError(
            f"row {index}: accepted IK candidate is marked rejected")

    footprint_status = row["footprint_status"]
    footprint_blocked = _integer(row, "footprint_blocked", index)
    if footprint_blocked:
        raise ValueError(f"row {index}: accepted footprint is blocked")
    if row["footprint_blocked_reason"] != "clear":
        raise ValueError(
            f"row {index}: footprint blocked reason disagrees with clear state")
    if footprint_status == "invalid-input":
        _check_invalid_footprint_is_canonical(row, index)
    elif footprint_status != "ok":
        raise ValueError(
            f"row {index}: accepted footprint status is not Ok or canonical")

    if _integer(row, "ik_applied", index) == 1:
        if footprint_status != "ok":
            raise ValueError(
                f"row {index}: applied IK lacks an Ok footprint")
        if row["ik_candidate_clearance_status"] != "ok":
            raise ValueError(
                f"row {index}: accepted IK candidate clearance status is not Ok")
        for foot in ("left", "right"):
            if (_integer(row, f"{foot}_landing_expected", index) == 1 and
                    _integer(
                        row, f"{foot}_landing_patch_ready", index) != 1):
                raise ValueError(
                    f"row {index}: accepted {foot} landing patch is not ready")


def _check_disabled_ik_is_canonical(row, index):
    text_defaults = {
        "ik_stop_reason": "none",
        "ik_candidate_clearance_status": "invalid-input",
        "left_swing_actual_sphere_center_bits_hex": "00000000" * 12,
        "left_swing_selected_clearance_status": "invalid-input",
        "right_swing_actual_sphere_center_bits_hex": "00000000" * 12,
        "right_swing_selected_clearance_status": "invalid-input",
    }
    uint_defaults = {
        "left_swing_selected_index": 2 ** 32 - 1,
        "right_swing_selected_index": 2 ** 32 - 1,
    }
    for name, expected in text_defaults.items():
        if row[name] != expected:
            raise ValueError(
                f"row {index}: disabled IK output is not canonical")
    for name, expected in uint_defaults.items():
        if _uint32(row, name, index) != expected:
            raise ValueError(
                f"row {index}: disabled IK output is not canonical")
    exceptions = set(text_defaults) | set(uint_defaults)
    for name in IK_SUFFIX:
        if name in exceptions:
            continue
        if _finite(row, name, index) != 0.0:
            raise ValueError(
                f"row {index}: disabled IK output is not canonical")


def _rejected_foot_is_canonical(row, index, foot):
    prefix = f"rejected_{foot}"
    if (_integer(row, f"{prefix}_landing_expected", index) != 0 or
            _integer(row, f"{prefix}_landing_patch_ready", index) != 0 or
            _uint32(row, f"{prefix}_landing_sample", index) != 2 ** 32 - 1 or
            row[f"{prefix}_landing_surface_status"] != "invalid" or
            _integer(row, f"{prefix}_landing_walkability_class", index) != 0 or
            _integer(row, f"{prefix}_reachable", index) != 0 or
            _integer(row, f"{prefix}_correction_limited", index) != 0 or
            row[f"{prefix}_selected_clearance_status"] != "invalid-input"):
        return False
    scalar_suffixes = (
        "landing_center_x", "landing_center_y", "landing_center_z",
        "landing_surface_height", "landing_surface_normal_x",
        "landing_surface_normal_y", "landing_surface_normal_z",
        "landing_patch_maximum_residual", "target_x", "target_y",
        "target_z", "target_normal_x", "target_normal_y",
        "target_normal_z", "selected_lower_margin",
        "selected_witness_upper",
    )
    return all(_finite(row, f"{prefix}_{suffix}", index) == 0.0
               for suffix in scalar_suffixes)


def _check_rejection_schema(row, index, previous_digest):
    rejected = _integer(row, "frame_rejected", index)
    stage = _require_enum(
        row, "frame_rejection_stage", index, REJECTION_STAGES)
    latched = _integer(row, "ik_safe_stop_latched", index)
    footprint_available = _integer(
        row, "rejected_attempted_footprint_available", index)
    ik_available = _integer(
        row, "rejected_attempted_ik_available", index)
    pose_available = _integer(
        row, "rejected_attempted_pose_available", index)
    reason = _require_enum(
        row, "rejected_stop_reason", index, IK_STOP_REASONS)
    pose_status = _require_enum(
        row, "rejected_pose_status", index, CLEARANCE_STATUSES)
    digest = _hex_digest(row, "accepted_state_digest_hex", index)
    if latched != rejected:
        raise ValueError(
            f"row {index}: rejection and safe-stop latch disagree")
    if not rejected:
        if (stage != "none" or footprint_available or ik_available or
                pose_available or reason != "none" or
                pose_status != "invalid-input" or
                not all(_rejected_foot_is_canonical(row, index, foot)
                        for foot in ("left", "right"))):
            raise ValueError(
                f"row {index}: accepted row has noncanonical rejection fields")
        return digest
    if previous_digest is None or digest != previous_digest:
        raise ValueError(
            f"row {index}: finite rejection changed accepted-state digest")
    if (not ik_available and
            not all(_rejected_foot_is_canonical(row, index, foot)
                    for foot in ("left", "right"))):
        raise ValueError(
            f"row {index}: rejection without attempted IK has noncanonical "
            "foot fields")
    if stage == "footprint":
        if ik_available or pose_available or pose_status != "invalid-input":
            raise ValueError(
                f"row {index}: footprint rejection availability is inconsistent")
        valid = (
            (reason == "footprint-blocked" and footprint_available) or
            (reason in {"footprint-outside-domain",
                        "footprint-budget-exceeded"} and
             not footprint_available))
        if not valid:
            raise ValueError(
                f"row {index}: footprint rejection reason is inconsistent")
    elif stage == "landing-patch":
        if (reason != "landing-patch-unavailable" or
                not footprint_available or not ik_available or
                pose_available or pose_status != "invalid-input"):
            raise ValueError(
                f"row {index}: landing-patch rejection availability is inconsistent")
        if not any(
                _integer(row,
                         f"rejected_{foot}_landing_expected", index) == 1 and
                _integer(row,
                         f"rejected_{foot}_landing_patch_ready", index) == 0
                for foot in ("left", "right")):
            raise ValueError(
                f"row {index}: landing-patch rejection lacks unready patch evidence")
    elif stage == "ik-candidate":
        if (reason not in {"target-unreachable", "no-swing-candidate"} or
                not footprint_available or not ik_available or
                pose_available or pose_status != "invalid-input"):
            raise ValueError(
                f"row {index}: IK-candidate rejection availability is inconsistent")
    elif stage == "pose-certificate":
        if (reason != "pose-clearance-rejected" or
                not footprint_available or ik_available):
            raise ValueError(
                f"row {index}: pose-certificate rejection availability is inconsistent")
        if pose_available:
            if pose_status != "ok":
                raise ValueError(
                    f"row {index}: available rejected pose must be Ok")
            if _finite(
                    row, "rejected_pose_minimum_clearance", index) >= -.005:
                raise ValueError(
                    f"row {index}: available rejected pose does not violate "
                    "a physical threshold")
        elif pose_status not in {
                "outside-domain", "budget-exceeded", "uncertified"}:
            raise ValueError(
                f"row {index}: unavailable rejected pose status is inconsistent")
    return digest


def _check_swing_selection_schema(row, index, foot):
    evaluated = _uint32(
        row, f"{foot}_swing_candidates_evaluated", index)
    selected = _uint32(row, f"{foot}_swing_selected_index", index)
    no_candidate = 2 ** 32 - 1
    if evaluated > 41:
        raise ValueError(
            f"row {index}: {foot} swing evaluated more than 41 candidates")
    if selected != no_candidate:
        if evaluated != 41 or selected >= 41:
            raise ValueError(
                f"row {index}: {foot} selected swing index is inconsistent")
        return
    if evaluated not in (0, 41):
        raise ValueError(
            f"row {index}: {foot} no-candidate evaluation count is invalid")
    uint_defaults = (
        "selected_lift_bits", "materialized_command_y_bits",
        "selected_work_point_queries", "selected_work_cells_visited",
        "selected_work_primitive_triangle_pairs",
        "selected_work_face_patches", "selected_work_candidate_tests",
        "selected_work_subdivision_nodes",
    )
    canonical = all(
        _uint32(row, f"{foot}_swing_{suffix}", index) == 0
        for suffix in uint_defaults)
    canonical = canonical and (
        row[f"{foot}_swing_actual_sphere_center_bits_hex"] ==
        "00000000" * 12 and
        row[f"{foot}_swing_selected_clearance_status"] == "invalid-input" and
        _integer(
            row,
            f"{foot}_swing_selected_controller_constraints_passed",
            index) == 0 and
        _integer(
            row, f"{foot}_swing_selected_clearance_certified", index) == 0 and
        _finite(row, f"{foot}_swing_lower_margin", index) == 0.0 and
        _finite(row, f"{foot}_swing_witness_upper_margin", index) == 0.0)
    if not canonical:
        raise ValueError(
            f"row {index}: {foot} no-candidate selection is not canonical")


def _check_full_runtime_suffix(row, index, previous_digest):
    _hex_float_words(row, "desired_heading_bits_hex", index, 4)
    _hex_float_words(row, "predicted_heading_bits_hex", index, 16)
    _hex_float_words(
        row, "left_swing_actual_sphere_center_bits_hex", index, 12)
    _hex_float_words(
        row, "right_swing_actual_sphere_center_bits_hex", index, 12)
    _require_enum(row, "ik_stop_reason", index, IK_STOP_REASONS)
    _require_enum(
        row, "ik_candidate_clearance_status", index, CLEARANCE_STATUSES)
    _require_enum(
        row, "left_swing_selected_clearance_status", index,
        CLEARANCE_STATUSES)
    _require_enum(
        row, "right_swing_selected_clearance_status", index,
        CLEARANCE_STATUSES)
    _check_swing_selection_schema(row, index, "left")
    _check_swing_selection_schema(row, index, "right")
    _require_enum(row, "footprint_status", index, FOOTPRINT_STATUSES)
    _require_enum(
        row, "footprint_blocked_reason", index, WALKABILITY_REASONS)
    _check_landing_schema(row, index, "left")
    _check_landing_schema(row, index, "right")
    _require_enum(
        row, "rejected_left_selected_clearance_status", index,
        CLEARANCE_STATUSES)
    _require_enum(
        row, "rejected_right_selected_clearance_status", index,
        CLEARANCE_STATUSES)
    _check_rejected_landing_schema(row, index, "left")
    _check_rejected_landing_schema(row, index, "right")
    _check_accepted_status_schema(row, index)
    for name in (
            "max_ik_correction", "actual_simulation_speed",
            "left_observed_lock_drift", "right_observed_lock_drift",
            "left_lock_drift", "right_lock_drift",
            "left_contact_residual", "right_contact_residual",
            "left_maximum_root_split", "right_maximum_root_split"):
        if _finite(row, name, index) < 0.0:
            raise ValueError(f"row {index}: {name} must be nonnegative")
    for foot in ("left", "right"):
        minimum = _finite(row, f"{foot}_footprint_min_height", index)
        maximum = _finite(row, f"{foot}_footprint_max_height", index)
        if minimum > maximum:
            raise ValueError(
                f"row {index}: {foot} footprint height envelope is reversed")
    for name in ("simulation_heading_error_deg", "rendered_heading_error_deg"):
        if _finite(row, name, index) < 0.0:
            raise ValueError(f"row {index}: {name} must be nonnegative")
    return _check_rejection_schema(row, index, previous_digest)


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
        if not group or "frame_rejected" not in group[0]:
            check_substride(group)
            continue
        accepted = []
        for row in group:
            if int(row["frame_rejected"]):
                if accepted:
                    check_substride(accepted)
                    accepted = []
            else:
                accepted.append(row)
        if accepted:
            check_substride(accepted)


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


def check_rows(rows, *, allow_ik=False):
    if not rows:
        raise ValueError("runtime log is empty")
    runtime_schema = _runtime_schema(rows)
    runtime = runtime_schema is not None
    full_runtime = runtime_schema == "full"
    active_runtime_suffix = (
        RUNTIME_SUFFIX if full_runtime else LEGACY_RUNTIME_SUFFIX)
    previous = None
    previous_range = None
    previous_generation = None
    previous_scene_frame = None
    previous_reset_count = None
    previous_accepted_digest = None
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
            for name in active_runtime_suffix:
                if name in RUNTIME_TEXT_COLUMNS:
                    _safe_runtime_text(row, name, index)
                elif name in RUNTIME_UINT32_COLUMNS:
                    _uint32(row, name, index)
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
                if name not in row:
                    continue
                if _integer(row, name, index) not in (0, 1):
                    raise ValueError(f"row {index}: {name} must be 0 or 1")
            for name in RUNTIME_NONNEGATIVE_INTEGER_COLUMNS:
                if name not in row:
                    continue
                if _integer(row, name, index) < 0:
                    raise ValueError(f"row {index}: {name} must be nonnegative")
            if _integer(row, "walkability_class", index) not in (0, 1, 2):
                raise ValueError(
                    f"row {index}: walkability_class must be 0, 1, or 2")
            if (not allow_ik and
                    _integer(row, "ik_enabled", index) != 0):
                raise ValueError(f"row {index}: IK must be disabled")
            for name in ("adjustment_y", "clamp_y"):
                if _finite(row, name, index) != 0.0:
                    raise ValueError(f"row {index}: {name} must be zero")
            if _integer(row, "motion_pack_load_count", index) != 1:
                raise ValueError(
                    f"row {index}: motion pack load count must be 1")
            if _integer(row, "live_model_count", index) != 1:
                raise ValueError(f"row {index}: live model count must be 1")
            if full_runtime:
                previous_accepted_digest = _check_full_runtime_suffix(
                    row, index, previous_accepted_digest)
        _check_query_snapshot(row, index)
        row_rejected = (
            full_runtime and _integer(row, "frame_rejected", index) == 1)
        if (previous is not None and not generation_changed and
                not transitioned and not row_rejected and
                current != previous + 1):
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


def _check_legacy_oracle_rows(rows):
    if not rows:
        raise ValueError("Task-3 oracle log is empty")
    for index, row in enumerate(rows):
        if tuple(row.keys()) != LEGACY_RUNTIME_COLUMNS:
            raise ValueError(
                f"row {index}: Task-3 oracle requires exact legacy header")
    # Reuse the mature immutable-prefix parser without pretending the
    # append-only Task-7 suffix was present in the older capture.
    prefix_rows = [
        {name: row[name] for name in GATE_A_COLUMNS}
        for row in rows
    ]
    summary = check_rows(prefix_rows)
    for index, row in enumerate(rows):
        fixed_dt = _finite(row, "fixed_dt", index)
        if struct.pack(">f", fixed_dt) != struct.pack(">f", .04):
            raise ValueError(
                f"row {index}: Task-3 oracle fixed_dt must be float32 0.04")
        for name in LEGACY_RUNTIME_SUFFIX:
            if name in RUNTIME_TEXT_COLUMNS:
                _safe_runtime_text(row, name, index)
            elif name in RUNTIME_INTEGER_COLUMNS:
                _integer(row, name, index)
            else:
                _finite(row, name, index)
        for name in (
                "left_contact", "right_contact", "blocked", "route_complete",
                "scene_switch_failed"):
            if _integer(row, name, index) not in (0, 1):
                raise ValueError(f"row {index}: {name} must be 0 or 1")
        if _integer(row, "scene_frame", index) != index:
            raise ValueError(
                f"row {index}: Task-3 oracle scene_frame is not contiguous")
        if _integer(row, "motion_pack_load_count", index) != 1:
            raise ValueError(
                f"row {index}: Task-3 oracle motion pack load count must be 1")
    return summary


GATE_L_IK_OFF_INVARIANTS = (
    "frame", "fixed_dt", "scene_id", "mode", "route",
    "query_bits_hex", "query_database_frame", "query_range",
    "selected_database_frame", "database_frame", "range", "source_range",
    "searched", "transitioned", "incumbent_cost", "selected_cost",
    "selected_terrain_error", "effective_terrain_weight",
    "terrain0", "terrain1", "terrain2", "terrain3",
    "terrain_point0_x", "terrain_point0_y", "terrain_point0_z",
    "terrain_point1_x", "terrain_point1_y", "terrain_point1_z",
    "terrain_point2_x", "terrain_point2_y", "terrain_point2_z",
    "terrain_point3_x", "terrain_point3_y", "terrain_point3_z",
    "runtime_root_surface_height", "runtime_left_toe_surface_height",
    "runtime_right_toe_surface_height",
    "source_name", "source_terrain", "source_index", "continuation_cost",
    "source_root_height", "source_left_toe_height", "source_right_toe_height",
    "runtime_support_root_height", "runtime_support_left_toe_height",
    "runtime_support_right_toe_height", "support_root_delta",
    "support_left_toe_delta", "support_right_toe_delta",
    "support_height", "support_velocity",
    "support_source", "airborne_frames", "left_contact", "right_contact",
    "support_retargeted_hips_y",
    "simulation_x", "simulation_z", "walkability_class", "blocked",
    "blocked_reason", "blocked_distance", "blocked_point_x",
    "blocked_point_z", "commanded_speed", "applied_speed",
    "route_waypoint", "route_complete", "route_target_height",
    "scene_generation", "scene_frame", "scene_reset_count",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled",
    "requested_velocity_x", "requested_velocity_y", "requested_velocity_z",
    "applied_velocity_x", "applied_velocity_y", "applied_velocity_z",
    "desired_heading_bits_hex", "predicted_heading_bits_hex",
)

GATE_L_ORACLE_INVARIANTS = (
    "frame", "fixed_dt", "scene_id", "mode", "route",
    "query_bits_hex", "query_database_frame", "query_range",
    "selected_database_frame", "database_frame", "range", "source_range",
    "searched", "transitioned", "incumbent_cost", "selected_cost",
    "selected_terrain_error", "effective_terrain_weight",
    "terrain0", "terrain1", "terrain2", "terrain3",
    "terrain_point0_x", "terrain_point0_y", "terrain_point0_z",
    "terrain_point1_x", "terrain_point1_y", "terrain_point1_z",
    "terrain_point2_x", "terrain_point2_y", "terrain_point2_z",
    "terrain_point3_x", "terrain_point3_y", "terrain_point3_z",
    "runtime_root_surface_height", "runtime_left_toe_surface_height",
    "runtime_right_toe_surface_height",
    "source_name", "source_terrain", "source_index", "continuation_cost",
    "source_root_height", "source_left_toe_height",
    "source_right_toe_height", "runtime_support_root_height",
    "runtime_support_left_toe_height", "runtime_support_right_toe_height",
    "support_root_delta", "support_left_toe_delta",
    "support_right_toe_delta", "support_height", "support_velocity",
    "support_source", "airborne_frames", "left_contact", "right_contact",
    "support_retargeted_hips_y",
    "simulation_x", "simulation_z", "walkability_class", "blocked",
    "blocked_reason", "blocked_distance", "blocked_point_x",
    "blocked_point_z", "commanded_speed", "applied_speed",
    "route_waypoint", "route_complete", "route_target_height",
    "scene_generation", "scene_frame", "scene_reset_count",
    "scene_switch_failed", "motion_pack_load_count",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled", "ik_enabled",
)


def _compare_gate_l_ik_off(treatment, control, stop_before=None):
    _require_full_runtime_header(control)
    if len(treatment) != len(control):
        raise ValueError("Gate L IK-off invariant lengths differ")
    stop = len(treatment) if stop_before is None else stop_before
    for index, (left, right) in enumerate(zip(treatment[:stop], control[:stop])):
        for name in GATE_L_IK_OFF_INVARIANTS:
            if left[name] != right[name]:
                raise ValueError(
                    f"row {index}: Gate L IK-off invariant {name} differs")
    check_rows(control)
    for index, row in enumerate(control):
        _check_disabled_ik_is_canonical(row, index)
    if any(_integer(row, "ik_enabled", index) != 0
           for index, row in enumerate(control)):
        raise ValueError("Gate L control must keep IK disabled")
    if any(_integer(row, "ik_applied", index) != 0
           for index, row in enumerate(control)):
        raise ValueError("Gate L IK-off control unexpectedly applied IK")
    if any(_integer(row, name, index) != 0
           for index, row in enumerate(control)
           for name in (
               "footprint_blocked", "ik_safe_stop_requested",
               "ik_safe_stop_latched", "frame_rejected")):
        raise ValueError("Gate L IK-off control contains a safe-stop")
    if any(row["footprint_status"] != "invalid-input" for row in control):
        raise ValueError(
            "Gate L IK-off control footprint status is not canonical")
    return stop


def _gate_l_heading(rows, expected_heading):
    if expected_heading not in HEADING_BITS:
        raise ValueError("Gate L expected heading is invalid")
    desired = HEADING_BITS[expected_heading]
    predicted = desired * 4
    for index, row in enumerate(rows):
        if row["desired_heading_bits_hex"] != desired:
            raise ValueError(
                f"row {index}: Gate L desired heading bits changed")
        if row["predicted_heading_bits_hex"] != predicted:
            raise ValueError(
                f"row {index}: Gate L predicted heading bits changed")
    return desired


def _nearest_rank_percentile(values, fraction):
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _gate_l_surface_split(rows, require_multilevel):
    maximum = 0.0
    qualifying = []
    for index, row in enumerate(rows):
        root = _finite(row, "runtime_root_surface_height", index)
        ik_enabled = _integer(row, "ik_enabled", index) == 1
        for foot in ("left", "right"):
            contact_name = (
                f"{foot}_recorded_contact" if ik_enabled
                else f"{foot}_contact")
            if _integer(row, contact_name, index) != 1:
                continue
            surface = _finite(
                row, f"runtime_{foot}_toe_surface_height", index)
            split = abs(surface - root)
            maximum = max(maximum, split)
            if split >= .04:
                qualifying.append((index, foot))
    if require_multilevel:
        if not qualifying:
            raise ValueError(
                "Gate L requires a 0.04 m multilevel surface split")
        for split_index, foot in qualifying:
            if not any(
                    _integer(rows[index],
                             f"{foot}_footprint_multilevel", index) == 1
                    for index in range(split_index + 1)):
                raise ValueError(
                    "Gate L requires a same-or-earlier footprint "
                    "multilevel report")
    return maximum


def _gate_l_support_peak(rows):
    return max(abs(_finite(row, "support_velocity", index))
               for index, row in enumerate(rows))


def _gate_l_validate_forward_reference(rows, expected_cell):
    _gate_l_normal_contract(rows, "forward", expected_cell)
    _gate_l_physical_report(rows)
    return _gate_l_support_peak(rows)


def _gate_l_normal_contract(rows, expected_heading, expected_cell=None):
    _require_full_runtime_header(rows)
    if any(row["mode"] != "route" for row in rows):
        raise ValueError("Gate L requires route mode")
    cells = {(row["scene_id"], row["route"]) for row in rows}
    if len(cells) != 1:
        raise ValueError("Gate L requires one scene and route")
    cell = next(iter(cells))
    if expected_cell is not None and cell != expected_cell:
        raise ValueError(
            f"Gate L requires {expected_cell[0]}/{expected_cell[1]}")
    generations = {
        _integer(row, "scene_generation", index)
        for index, row in enumerate(rows)
    }
    if len(generations) != 1:
        raise ValueError("Gate L requires one generation")
    summary = check_rows(rows, allow_ik=True)
    if any(_integer(row, "matching_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L matching must remain enabled")
    if any(_integer(row, "support_retargeting_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L support retargeting must remain enabled")
    if any(_integer(row, "ik_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L requires IK enabled")
    if any(_integer(row, "ik_applied", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L requires IK applied on every accepted frame")
    if any(_integer(row, "walkability_class", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L requires class-1 traversal")
    if _integer(rows[-1], "route_complete", len(rows) - 1) != 1:
        raise ValueError("Gate L route did not complete")
    if any(
            _integer(row, name, index) != 0
            for index, row in enumerate(rows)
            for name in (
                "ik_safe_stop_requested", "ik_safe_stop_latched",
                "footprint_blocked", "blocked")):
        raise ValueError("Gate L forbids any footprint or IK safe-stop")
    if any(_integer(row, "frame_rejected", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError("Gate L normal routes forbid finite rejection")
    if any(row["footprint_status"] != "ok" for row in rows):
        raise ValueError("Gate L requires an Ok footprint observation")
    if any(_integer(row, "ik_candidate_rejected", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError("Gate L accepted state reports an IK rejection")
    _gate_l_heading(rows, expected_heading)
    return summary, cell


def _gate_l_physical_report(rows):
    minimum_rendered = math.inf
    minimum_candidate = math.inf
    minimum_certified = math.inf
    maximum_correction = 0.0
    maximum_hips_step = 0.0
    for index, row in enumerate(rows):
        minimum_rendered = min(
            minimum_rendered,
            _finite(row, "rendered_min_clearance", index))
        minimum_certified = min(
            minimum_certified,
            _finite(row, "ik_minimum_clearance", index))
        minimum_candidate = min(
            minimum_candidate,
            _finite(row, "ik_candidate_minimum_clearance", index))
        maximum_correction = max(
            maximum_correction,
            _finite(row, "max_ik_correction", index))
        for foot in ("left", "right"):
            if _integer(row, f"{foot}_recorded_contact", index) == 1:
                planted_clearances = (
                    _finite(row, f"{foot}_candidate_toe_clearance", index),
                    _finite(row, f"{foot}_candidate_foot_clearance", index),
                    _finite(row, f"{foot}_toe_clearance", index),
                    _finite(row, f"{foot}_foot_clearance", index),
                )
                if min(planted_clearances) < -.005:
                    raise ValueError(
                        f"row {index}: Gate L planted toe/foot clearance "
                        "fell below -0.005 m")
        if index:
            step = abs(
                _finite(row, "rendered_hips_y", index) -
                _finite(rows[index - 1], "rendered_hips_y", index - 1))
            maximum_hips_step = max(maximum_hips_step, step)
    if (minimum_rendered < -.01 or minimum_candidate < -.01 or
            minimum_certified < -.01):
        raise ValueError("Gate L physical clearance fell below -0.01 m")
    if maximum_correction > .35:
        raise ValueError("Gate L IK correction exceeded 0.35 rad")
    if maximum_hips_step > .05:
        raise ValueError("Gate L rendered Hips step exceeded 0.05 m")
    return {
        "minimum_rendered_clearance": minimum_rendered,
        "minimum_candidate_clearance": minimum_candidate,
        "minimum_certified_clearance": minimum_certified,
        "maximum_ik_correction": maximum_correction,
        "maximum_rendered_hips_step": maximum_hips_step,
    }


def check_gate_l(
        rows, *, expected_end_x, expected_end_z, expected_heading,
        require_multilevel=False, compare_ik_off=None, compare_forward=None,
        compare_forward_baseline=None, safety_only=False):
    summary, cell = _gate_l_normal_contract(rows, expected_heading)
    try:
        end_x = float(expected_end_x)
        end_z = float(expected_end_z)
    except (TypeError, ValueError) as error:
        raise ValueError("Gate L endpoint must be finite") from error
    if not math.isfinite(end_x) or not math.isfinite(end_z):
        raise ValueError("Gate L endpoint must be finite")
    final_x = _finite(rows[-1], "simulation_x", len(rows) - 1)
    final_z = _finite(rows[-1], "simulation_z", len(rows) - 1)
    endpoint_error = math.hypot(final_x - end_x, final_z - end_z)
    if endpoint_error > .25:
        raise ValueError(
            f"Gate L endpoint error {endpoint_error} exceeds 0.25 m")

    physical = _gate_l_physical_report(rows)

    errors = [
        _finite(row, "rendered_heading_error_deg", index)
        for index, row in enumerate(rows)
        if index >= 50
    ]
    if not errors:
        raise ValueError("Gate L requires heading-error rows after frame 50")
    median_error = statistics.median(errors)
    p95_error = _nearest_rank_percentile(errors, .95)
    maximum_error = max(errors)
    if not safety_only:
        if median_error > 10.0:
            raise ValueError("Gate L median rendered heading error exceeds 10 deg")
        if p95_error > 20.0:
            raise ValueError("Gate L 95th rendered heading error exceeds 20 deg")
        if maximum_error > 35.0:
            raise ValueError("Gate L maximum rendered heading error exceeds 35 deg")

    maximum_split = _gate_l_surface_split(rows, require_multilevel)
    support_peak = _gate_l_support_peak(rows)
    if compare_forward is None:
        support_limit = (
            1.5 if cell ==
            ("mixed-multilevel", "tangent-level-boundary") else math.inf)
    else:
        forward_peak = _gate_l_validate_forward_reference(
            compare_forward, cell)
        support_limit = max(forward_peak + .25, 1.25 * forward_peak)
    if support_peak > support_limit:
        raise ValueError(
            f"Gate L support velocity {support_peak} exceeds {support_limit}")

    paired_frames = 0
    if compare_ik_off is not None:
        paired_frames = _compare_gate_l_ik_off(rows, compare_ik_off)
    forward_baseline_frames = 0
    if compare_forward_baseline is not None:
        if compare_ik_off is None:
            raise ValueError(
                "Gate L forward baseline requires a paired IK-off log")
        baseline_report = check_gate_l_forward_baseline(
            compare_ik_off, compare_forward_baseline)
        forward_baseline_frames = baseline_report["frames"]
    return {
        **summary,
        "scene": cell[0],
        "route": cell[1],
        "heading": expected_heading,
        "endpoint_error_m": endpoint_error,
        **physical,
        "median_rendered_heading_error_deg": median_error,
        "p95_rendered_heading_error_deg": p95_error,
        "maximum_rendered_heading_error_deg": maximum_error,
        "maximum_surface_split_m": maximum_split,
        "maximum_support_velocity": support_peak,
        "support_velocity_limit": support_limit,
        "paired_ik_off_frames": paired_frames,
        "forward_baseline_frames": forward_baseline_frames,
    }


def _gate_l_role_coverage(rows):
    coverage = {"left": set(), "right": set()}
    current_role = None
    current_length = 0

    def commit():
        if current_role is not None and current_length >= 3:
            coverage["left"].add(current_role[0])
            coverage["right"].add(current_role[1])

    for index, row in enumerate(rows):
        role = None
        if (_integer(row, "left_recorded_contact", index) == 1 and
                _integer(row, "right_recorded_contact", index) == 1):
            left = _finite(row, "left_target_height", index)
            right = _finite(row, "right_target_height", index)
            if abs(left - right) >= .04:
                role = ("uphill", "downhill") if left > right else (
                    "downhill", "uphill")
        if role == current_role and role is not None:
            current_length += 1
        else:
            commit()
            current_role = role
            current_length = 1 if role is not None else 0
    commit()
    return coverage


def check_gate_l2_pair(positive_x_rows, negative_x_rows):
    cell = ("stairs-standard", "ascent-landing-descent")
    positive_summary, _ = _gate_l_normal_contract(
        positive_x_rows, "positive-x", cell)
    negative_summary, _ = _gate_l_normal_contract(
        negative_x_rows, "negative-x", cell)
    positive_physical = _gate_l_physical_report(positive_x_rows)
    negative_physical = _gate_l_physical_report(negative_x_rows)
    positive_split = _gate_l_surface_split(positive_x_rows, True)
    negative_split = _gate_l_surface_split(negative_x_rows, True)
    maximum_support_velocity = max(
        _gate_l_support_peak(positive_x_rows),
        _gate_l_support_peak(negative_x_rows))
    if maximum_support_velocity > 1.5:
        raise ValueError(
            "Gate L2 support velocity exceeds the 1.50 m/s tangential cap")
    coverage = {"left": set(), "right": set()}
    for rows in (positive_x_rows, negative_x_rows):
        observed = _gate_l_role_coverage(rows)
        for foot in coverage:
            coverage[foot].update(observed[foot])
    expected = {"uphill", "downhill"}
    if coverage != {"left": expected, "right": expected}:
        raise ValueError(
            "Gate L2 leg-role coverage must include both legs as uphill "
            "and downhill for three consecutive rows")
    return {
        "positive_frames": positive_summary["frames"],
        "negative_frames": negative_summary["frames"],
        "role_coverage": "left:downhill+uphill,right:downhill+uphill",
        "minimum_rendered_clearance": min(
            positive_physical["minimum_rendered_clearance"],
            negative_physical["minimum_rendered_clearance"]),
        "minimum_certified_clearance": min(
            positive_physical["minimum_certified_clearance"],
            negative_physical["minimum_certified_clearance"]),
        "maximum_surface_split_m": max(positive_split, negative_split),
        "maximum_support_velocity": maximum_support_velocity,
    }


def _gate_l_exit_contract(rows, ik_enabled):
    _require_full_runtime_header(rows)
    cell = ("stairs-standard", "landing-side-exit-stress")
    if any(row["mode"] != "route" for row in rows):
        raise ValueError("Gate L2 exit stress requires route mode")
    if {(row["scene_id"], row["route"]) for row in rows} != {cell}:
        raise ValueError(
            "Gate L2 exit stress requires "
            "stairs-standard/landing-side-exit-stress")
    if len({_integer(row, "scene_generation", index)
            for index, row in enumerate(rows)}) != 1:
        raise ValueError("Gate L2 exit stress requires one generation")
    summary = check_rows(rows, allow_ik=bool(ik_enabled))
    if any(_integer(row, "ik_enabled", index) != ik_enabled
           for index, row in enumerate(rows)):
        raise ValueError(
            f"Gate L2 exit stress requires IK enabled={ik_enabled}")
    if any(_integer(row, "matching_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L2 exit stress matching must remain enabled")
    if any(_integer(row, "support_retargeting_enabled", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError(
            "Gate L2 exit stress support retargeting must remain enabled")
    _gate_l_heading(rows, "positive-x")
    if any(_integer(row, "walkability_class", index) != 1
           for index, row in enumerate(rows)):
        raise ValueError("Gate L2 exit stress requires class-1 traversal")
    if any(_integer(row, "ik_applied", index) != ik_enabled
           for index, row in enumerate(rows)):
        raise ValueError(
            f"Gate L2 exit stress requires IK applied={ik_enabled}")
    if any(_integer(row, name, index) != 0
           for index, row in enumerate(rows)
           for name in (
               "footprint_blocked", "blocked", "ik_safe_stop_requested")):
        raise ValueError(
            "Gate L2 exit stress accepted state contains a footprint block")
    expected_footprint_status = "ok" if ik_enabled else "invalid-input"
    if any(row["footprint_status"] != expected_footprint_status
           for row in rows):
        raise ValueError(
            "Gate L2 exit stress footprint status is inconsistent with IK")
    if not ik_enabled:
        if any(_integer(row, name, index) != 0
               for index, row in enumerate(rows)
               for name in ("frame_rejected", "ik_safe_stop_latched")):
            raise ValueError(
                "Gate L2 exit stress IK-off control may not safe-stop")
        if _integer(rows[-1], "route_complete", len(rows) - 1) != 1:
            raise ValueError(
                "Gate L2 exit stress IK-off control did not complete")
    return summary


def _rejected_ik_target_is_valid(row, index, foot):
    prefix = f"rejected_{foot}"
    normal = tuple(
        _finite(row, f"{prefix}_target_normal_{axis}", index)
        for axis in "xyz")
    length = math.sqrt(sum(component * component for component in normal))
    return abs(length - 1.0) <= 1e-5 and normal[1] > 0.0


def _rejected_ik_foot_is_unprocessed(row, index, foot):
    prefix = f"rejected_{foot}"
    return (
        _rejected_ik_target_is_valid(row, index, foot) and
        _integer(row, f"{prefix}_reachable", index) == 1 and
        _integer(row, f"{prefix}_correction_limited", index) == 0 and
        row[f"{prefix}_selected_clearance_status"] == "invalid-input" and
        _finite(row, f"{prefix}_selected_lower_margin", index) == 0.0 and
        _finite(row, f"{prefix}_selected_witness_upper", index) == 0.0)


def _rejected_ik_foot_is_successful(row, index, foot):
    prefix = f"rejected_{foot}"
    if (not _rejected_ik_target_is_valid(row, index, foot) or
            _integer(row, f"{prefix}_reachable", index) != 1 or
            _integer(row, f"{prefix}_correction_limited", index) != 0):
        return False
    recorded_contact = _integer(
        row, f"{foot}_recorded_contact", index) == 1
    status = row[f"{prefix}_selected_clearance_status"]
    lower = _finite(row, f"{prefix}_selected_lower_margin", index)
    upper = _finite(row, f"{prefix}_selected_witness_upper", index)
    if recorded_contact:
        return status == "invalid-input" and lower == 0.0 and upper == 0.0
    return (
        status == "ok" and lower >= 0.0 and upper >= lower and
        upper - lower <= 1.0e-6)


def _rejected_ik_foot_proves_failure(row, index, foot, reason):
    prefix = f"rejected_{foot}"
    if not _rejected_ik_target_is_valid(row, index, foot):
        return False
    recorded_contact = _integer(
        row, f"{foot}_recorded_contact", index) == 1
    reachable = _integer(row, f"{prefix}_reachable", index) == 1
    correction_limited = _integer(
        row, f"{prefix}_correction_limited", index) == 1
    status = row[f"{prefix}_selected_clearance_status"]
    lower = _finite(row, f"{prefix}_selected_lower_margin", index)
    upper = _finite(row, f"{prefix}_selected_witness_upper", index)
    if reason == "target-unreachable":
        return (
            recorded_contact and not reachable and
            status == "invalid-input" and lower == 0.0 and upper == 0.0)
    return (
        reason == "no-swing-candidate" and not recorded_contact and
        reachable and not correction_limited and
        status == "invalid-input" and lower == 0.0 and upper == 0.0)


def _rejected_ik_sequence_proves_failure(row, index, reason):
    left_failed = _rejected_ik_foot_proves_failure(
        row, index, "left", reason)
    right_failed = _rejected_ik_foot_proves_failure(
        row, index, "right", reason)
    return (
        (left_failed and
         _rejected_ik_foot_is_unprocessed(row, index, "right")) or
        (_rejected_ik_foot_is_successful(row, index, "left") and
         right_failed))


def _gate_l_rejected_attempt_is_outside_contract(row, index):
    reason = row["rejected_stop_reason"]
    stage = row["frame_rejection_stage"]
    if reason in {"target-unreachable", "no-swing-candidate"}:
        return (
            stage == "ik-candidate" and
            _rejected_ik_sequence_proves_failure(row, index, reason))
    if reason == "pose-clearance-rejected":
        return stage == "pose-certificate"
    if reason != "landing-patch-unavailable" or stage != "landing-patch":
        return False
    for foot in ("left", "right"):
        expected = _integer(
            row, f"rejected_{foot}_landing_expected", index) == 1
        ready = _integer(
            row, f"rejected_{foot}_landing_patch_ready", index) == 1
        residual = _finite(
            row, f"rejected_{foot}_landing_patch_maximum_residual", index)
        status = row[f"rejected_{foot}_selected_clearance_status"]
        if status == "invalid-input" and (
                expected and not ready and
                residual > LANDING_PATCH_RESIDUAL_LIMIT):
            return True
    return False


def check_gate_l2_exit_stress(rows, compare_ik_off):
    summary = _gate_l_exit_contract(rows, 1)
    _gate_l_exit_contract(compare_ik_off, 0)
    physical = _gate_l_physical_report(rows)
    rejected = [
        index for index, row in enumerate(rows)
        if _integer(row, "frame_rejected", index) == 1
    ]
    if not rejected:
        if _integer(rows[-1], "route_complete", len(rows) - 1) != 1:
            raise ValueError(
                "Gate L2 exit stress neither completed nor safe-stopped")
        if any(_integer(row, name, index) != 0
               for index, row in enumerate(rows)
               for name in ("ik_safe_stop_requested", "ik_safe_stop_latched",
                            "footprint_blocked", "blocked")):
            raise ValueError(
                "Gate L2 exit stress completed branch contains a safe-stop")
        _compare_gate_l_ik_off(rows, compare_ik_off)
        return {**summary, **physical, "branch": "complete",
                "rejected_frames": 0}

    first = rejected[0]
    if first == 0 or rejected != list(range(first, len(rows))):
        raise ValueError(
            "Gate L2 exit stress safe-stop must be a rejected tail")
    if any(_integer(row, "route_complete", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError(
            "Gate L2 exit stress safe-stop may not claim completion")
    baseline = rows[first - 1]
    for index in rejected:
        row = rows[index]
        if row["rejected_stop_reason"] not in {
                "landing-patch-unavailable", "target-unreachable",
                "no-swing-candidate", "pose-clearance-rejected"}:
            raise ValueError(
                f"row {index}: Gate L2 exit stress stop reason is not allowed")
        if not _gate_l_rejected_attempt_is_outside_contract(row, index):
            raise ValueError(
                f"row {index}: Gate L2 rejected landing is not proven "
                "outside the certified contract")
        for name in (
                "left_recorded_contact", "right_recorded_contact",
                "left_target_height", "right_target_height"):
            if row[name] != baseline[name]:
                label = "committed target" if "target" in name else "contact"
                raise ValueError(
                    f"row {index}: Gate L2 safe-stop changed accepted {label}")
    paired = _compare_gate_l_ik_off(rows, compare_ik_off, stop_before=first)
    return {
        **summary, **physical,
        "branch": "safe-stop",
        "rejected_frames": len(rejected),
        "paired_ik_off_frames": paired,
    }


def _compare_oracle_invariants(rows, baseline, label):
    if len(rows) != len(baseline):
        raise ValueError(f"Gate L {label} lengths differ")
    for index, (current, old) in enumerate(zip(rows, baseline)):
        for name in GATE_L_ORACLE_INVARIANTS:
            if current[name] != old[name]:
                raise ValueError(
                    f"row {index}: Gate L {label} invariant {name} differs")


def _maximum_hips_step(rows):
    return max((
        abs(_finite(rows[index], "rendered_hips_y", index) -
            _finite(rows[index - 1], "rendered_hips_y", index - 1))
        for index in range(1, len(rows))
    ), default=0.0)


def _contact_level_bits(rows):
    levels = set()
    for index, row in enumerate(rows):
        for foot in ("left", "right"):
            if _integer(row, f"{foot}_contact", index) == 1:
                value = _finite(
                    row, f"runtime_{foot}_toe_surface_height", index)
                levels.add((foot, struct.pack(">f", value)))
    return levels


def check_gate_l_forward_baseline(rows, baseline_path):
    absolute = _verify_oracle_sha256(baseline_path)
    basename = os.path.basename(absolute)
    if basename not in FORWARD_ORACLE_CASES:
        raise ValueError("Gate L forward baseline is not a locked forward oracle")
    baseline = read_rows(absolute)
    _check_legacy_oracle_rows(baseline)
    _require_full_runtime_header(rows)
    summary = check_rows(rows)
    if len(rows) != 800 or len(baseline) != 800:
        raise ValueError("Gate L forward oracle requires exactly 800 rows")
    cell = FORWARD_ORACLE_CASES[basename]
    if {(row["scene_id"], row["route"]) for row in rows} != {cell} or \
            {(row["scene_id"], row["route"]) for row in baseline} != {cell}:
        raise ValueError("Gate L forward oracle scene/route identity differs")
    if any(row["mode"] != "route" for row in rows + baseline):
        raise ValueError("Gate L forward oracle requires route mode")
    _gate_l_heading(rows, "forward")
    if any(_integer(row, "ik_enabled", index) != 0 or
           _integer(row, "ik_applied", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError("Gate L forward oracle comparison requires IK disabled")
    for index, row in enumerate(rows):
        _check_disabled_ik_is_canonical(row, index)
        if row["footprint_status"] != "invalid-input":
            raise ValueError(
                f"row {index}: Gate L forward IK-off footprint status "
                "is not canonical")
        _check_invalid_footprint_is_canonical(row, index)
    _compare_oracle_invariants(rows, baseline, "forward oracle")
    if _integer(baseline[-1], "route_complete", 799) != 1:
        raise ValueError("Gate L forward oracle baseline did not complete")
    if _integer(rows[-1], "route_complete", 799) != 1:
        raise ValueError("Gate L forward run regressed route completion")
    baseline_clearance = min(
        _finite(row, "rendered_min_clearance", index)
        for index, row in enumerate(baseline))
    current_clearance = min(
        _finite(row, "rendered_min_clearance", index)
        for index, row in enumerate(rows))
    if current_clearance < baseline_clearance:
        raise ValueError("Gate L forward rendered clearance regressed")
    baseline_levels = _contact_level_bits(baseline)
    current_levels = _contact_level_bits(rows)
    if not baseline_levels.issubset(current_levels):
        raise ValueError("Gate L forward contact-level coverage regressed")
    baseline_hips_step = _maximum_hips_step(baseline)
    current_hips_step = _maximum_hips_step(rows)
    if current_hips_step > baseline_hips_step + 1e-9:
        raise ValueError("Gate L forward Hips step regressed")
    return {
        **summary,
        "baseline_minimum_rendered_clearance": baseline_clearance,
        "minimum_rendered_clearance": current_clearance,
        "baseline_maximum_hips_step": baseline_hips_step,
        "maximum_hips_step": current_hips_step,
        "contact_levels": len(current_levels),
    }


def _relative_travel_direction(rows, expected_heading):
    start_x = _finite(rows[0], "simulation_x", 0)
    start_z = _finite(rows[0], "simulation_z", 0)
    final_index = len(rows) - 1
    world_x = _finite(rows[-1], "simulation_x", final_index) - start_x
    world_z = _finite(rows[-1], "simulation_z", final_index) - start_z
    distance = math.hypot(world_x, world_z)
    if distance < .25:
        raise ValueError("Gate L4 relative travel distance is below 0.25 m")
    w, _, y, _ = HEADING_COMPONENTS[expected_heading]
    yaw = math.atan2(2.0 * w * y, 1.0 - 2.0 * y * y)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local_x = cosine * world_x - sine * world_z
    local_z = sine * world_x + cosine * world_z
    if abs(local_z) >= abs(local_x):
        direction = "forward" if local_z >= 0.0 else "backward"
        dominant, secondary = abs(local_z), abs(local_x)
    else:
        direction = "right" if local_x >= 0.0 else "left"
        dominant, secondary = abs(local_x), abs(local_z)
    if dominant < 2.0 * secondary:
        raise ValueError("Gate L4 relative travel direction is ambiguous")
    return direction, distance


def check_gate_l4(
        rows, baseline_path, *, expected_heading,
        expected_relative_direction):
    absolute = _verify_oracle_sha256(baseline_path)
    basename = os.path.basename(absolute)
    if basename not in FLAT_ORACLE_CASES:
        raise ValueError("Gate L4 baseline is not a locked flat oracle")
    scene, route, oracle_heading, oracle_relative = FLAT_ORACLE_CASES[basename]
    if (expected_heading != oracle_heading or
            expected_relative_direction != oracle_relative):
        raise ValueError(
            "Gate L4 expected heading/relative direction does not match "
            "the locked oracle pairing")
    baseline = read_rows(absolute)
    _check_legacy_oracle_rows(baseline)
    _require_full_runtime_header(rows)
    summary = check_rows(rows, allow_ik=True)
    if len(rows) != 100 or len(baseline) != 100:
        raise ValueError("Gate L4 requires exactly 100 rows")
    cell = (scene, route)
    if {(row["scene_id"], row["route"]) for row in rows} != {cell} or \
            {(row["scene_id"], row["route"]) for row in baseline} != {cell}:
        raise ValueError("Gate L4 flat oracle scene/route identity differs")
    if any(row["mode"] != "route" for row in rows + baseline):
        raise ValueError("Gate L4 requires route mode")
    if len({_integer(row, "scene_generation", index)
            for index, row in enumerate(rows)}) != 1:
        raise ValueError("Gate L4 requires one generation")
    _gate_l_heading(rows, expected_heading)
    if any(_integer(row, "ik_enabled", index) != 0 or
           _integer(row, "ik_applied", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError("Gate L4 requires IK disabled")
    for index, row in enumerate(rows):
        _check_disabled_ik_is_canonical(row, index)
    if any(_integer(row, name, index) != 0
           for index, row in enumerate(rows)
           for name in (
               "footprint_blocked", "blocked", "ik_safe_stop_requested",
               "ik_safe_stop_latched", "frame_rejected")):
        raise ValueError("Gate L4 forbids any footprint block/stop")
    if any(row["footprint_status"] != "invalid-input" for row in rows):
        raise ValueError("Gate L4 IK-off footprint status is not canonical")
    if _integer(rows[-1], "route_complete", 99) != 1:
        raise ValueError("Gate L4 route did not complete")
    maximum_split = _gate_l_surface_split(rows, False)
    if maximum_split >= .04:
        raise ValueError("Gate L4 physical surface split reached 0.04 m")
    _compare_oracle_invariants(rows, baseline, "flat oracle")
    relative, distance = _relative_travel_direction(rows, expected_heading)
    if relative != expected_relative_direction:
        raise ValueError(
            f"Gate L4 relative direction {relative} does not match "
            f"{expected_relative_direction}")
    return {
        **summary,
        "scene": scene,
        "route": route,
        "heading": expected_heading,
        "relative_direction": relative,
        "travel_distance_m": distance,
        "maximum_surface_split_m": maximum_split,
    }


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
    blocked_distances = [
        _finite(rows[index], "blocked_distance", index)
        for index in stopped_tail
    ]
    minimum_blocked_distance = min(blocked_distances)
    if minimum_blocked_distance < 0.02 - 1e-4:
        raise ValueError(
            "Gate D stopped distance fell below 0.02 m")
    minimum_rendered_clearance = min(
        _finite(rows[index], "rendered_min_clearance", index)
        for index in stopped_tail)
    # Pre-footprint runtime logs do not carry the accepted IK certificate.
    # They remain readable for the immutable earlier gates, but the append-only
    # schema supplies this independent physical value for new logs.
    minimum_certified_clearance = min(
        _finite(rows[index], "ik_minimum_clearance", index)
        if "ik_minimum_clearance" in rows[index]
        else _finite(rows[index], "rendered_min_clearance", index)
        for index in stopped_tail)
    if (minimum_rendered_clearance < -0.01 or
            minimum_certified_clearance < -0.01):
        raise ValueError(
            "Gate D physical clearance fell below -0.01 m")
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
        "minimum_blocked_distance": minimum_blocked_distance,
        "minimum_rendered_clearance": minimum_rendered_clearance,
        "minimum_certified_clearance": minimum_certified_clearance,
        "maximum_blocked_support_rise": blocked_support_rise,
    }


def _check_model_counters(rows):
    previous_loads = None
    previous_unloads = None
    for index, row in enumerate(rows):
        loads = _integer(row, "model_load_count", index)
        unloads = _integer(row, "model_unload_count", index)
        live = _integer(row, "live_model_count", index)
        if live != loads - unloads:
            raise ValueError(
                f"row {index}: live model count mismatch: "
                "live_model_count must equal "
                "model_load_count minus model_unload_count")
        if previous_loads is not None and loads < previous_loads:
            raise ValueError(
                f"row {index}: model_load_count must be nondecreasing")
        if previous_unloads is not None and unloads < previous_unloads:
            raise ValueError(
                f"row {index}: model_unload_count must be nondecreasing")
        previous_loads = loads
        previous_unloads = unloads


def check_gate_f(rows, expected_scene_ids):
    summary = check_rows(rows)
    _require_runtime_header(rows)
    _check_model_counters(rows)

    if isinstance(expected_scene_ids, (str, bytes)):
        raise ValueError("Gate F expected scene ID list must be nonempty")
    try:
        expected = tuple(expected_scene_ids)
    except TypeError as error:
        raise ValueError(
            "Gate F expected scene ID list must be nonempty") from error
    if not expected or any(
            not isinstance(scene_id, str) or not scene_id
            for scene_id in expected):
        raise ValueError("Gate F expected scene IDs must be nonempty")
    if len(expected) != len(set(expected)):
        raise ValueError("Gate F expected scene IDs must be unique")
    if any(row["mode"] != "scene-cycle" for row in rows):
        raise ValueError("Gate F requires scene-cycle mode throughout")
    if any(_integer(row, "scene_switch_failed", index) != 0
           for index, row in enumerate(rows)):
        raise ValueError("Gate F forbids any switch-failure pulse")

    groups = _generation_groups(rows)
    dwell = len(groups[0])
    if dwell <= 0 or any(len(group) != dwell for group in groups):
        raise ValueError("Gate F generations must have one positive dwell")

    observed = []
    start = 0
    for generation, group in enumerate(groups):
        first = group[0]
        if _integer(first, "scene_generation", start) != generation:
            raise ValueError(
                "Gate F generations must start at generation 0 and be contiguous")
        if _integer(first, "scene_frame", start) != 0:
            raise ValueError(
                f"Gate F generation {generation} must start at scene_frame 0")
        scene_id = first["scene_id"]
        if any(row["scene_id"] != scene_id for row in group):
            raise ValueError(
                f"Gate F generation {generation} must contain one constant scene")
        observed.append(scene_id)
        first_row_requirements = (
            ("scene_reset_count", generation + 1),
            ("live_model_count", 1),
            ("route_waypoint", 0),
            ("blocked", 0),
        )
        for name, required in first_row_requirements:
            if _integer(first, name, start) != required:
                raise ValueError(
                    f"Gate F generation {generation} first-row {name} "
                    f"must be {required}")
        if _integer(first, "airborne_frames", start) > 1:
            raise ValueError(
                f"Gate F generation {generation} first-row "
                "airborne_frames must be at most 1")
        for offset, row in enumerate(group):
            index = start + offset
            for name, required in (
                    ("model_load_count", generation + 1),
                    ("model_unload_count", generation)):
                if _integer(row, name, index) != required:
                    raise ValueError(
                        f"Gate F generation {generation} row {index} "
                        f"{name} must remain {required}")
        start += len(group)

    if len(groups) % len(expected) != 0:
        raise ValueError("Gate F requires complete ordered cycles")
    complete_cycles = len(groups) // len(expected)
    if complete_cycles < 2:
        raise ValueError("Gate F requires at least two complete cycles")
    if tuple(observed) != expected * complete_cycles:
        raise ValueError(
            "Gate F generation scene IDs must form complete ordered cycles")

    final_index = len(rows) - 1
    return {
        "frames": summary["frames"],
        "generations": len(groups),
        "complete_cycles": complete_cycles,
        "motion_pack_loads": _integer(
            rows[0], "motion_pack_load_count", 0),
        "model_loads": _integer(
            rows[-1], "model_load_count", final_index),
        "model_unloads_before_final_cleanup": _integer(
            rows[-1], "model_unload_count", final_index),
    }


def check_failed_switch(rows):
    _require_runtime_header(rows)
    if not rows:
        check_rows(rows)
    _check_model_counters(rows)
    pulses = [
        index for index, row in enumerate(rows)
        if _integer(row, "scene_switch_failed", index) == 1
    ]
    if not pulses:
        check_rows(rows)
        raise ValueError("switch-failure checker requires a failure pulse")

    for index in pulses:
        if index == 0:
            raise ValueError(
                "switch-failure pulse requires a preceding row")
        row = rows[index]
        previous = rows[index - 1]
        if row["scene_id"] != previous["scene_id"]:
            raise ValueError(
                f"row {index}: failed switch must preserve scene")
        generation = _integer(row, "scene_generation", index)
        if generation != _integer(
                previous, "scene_generation", index - 1):
            raise ValueError(
                f"row {index}: failed switch must preserve generation")
        if _integer(row, "scene_frame", index) != _integer(
                previous, "scene_frame", index - 1) + 1:
            raise ValueError(
                f"row {index}: failed switch must continue scene frame")
        if _integer(row, "scene_reset_count", index) != _integer(
                previous, "scene_reset_count", index - 1):
            raise ValueError(
                f"row {index}: failed switch must preserve reset count")
        if _integer(row, "motion_pack_load_count", index) != 1:
            raise ValueError(
                f"row {index}: failed switch must preserve one motion pack")
        if _integer(row, "live_model_count", index) != 1:
            raise ValueError(
                f"row {index}: failed switch must preserve one live model")

        preserved_rows = 0
        successful_generation_began = False
        for later_index in range(index + 1, min(len(rows), index + 11)):
            later = rows[later_index]
            if _integer(later, "scene_generation", later_index) != generation:
                successful_generation_began = True
                break
            if later["scene_id"] != row["scene_id"]:
                raise ValueError(
                    f"row {later_index}: ten-row preservation window "
                    "must preserve scene")
            preserved_rows += 1
        if preserved_rows < 10 and not successful_generation_began:
            raise ValueError(
                f"row {index}: failed switch requires ten subsequent "
                "preserved rows or a later successful generation")

    check_rows(rows)
    first = pulses[0]
    return {
        "switch_failures": len(pulses),
        "preserved_scene": rows[first]["scene_id"],
        "preserved_generation": _integer(
            rows[first], "scene_generation", first),
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
    runtime_gate.add_argument("--gate-f", action="store_true")
    runtime_gate.add_argument("--gate-l", action="store_true")
    runtime_gate.add_argument("--gate-l-safety-only", action="store_true")
    runtime_gate.add_argument("--gate-l2-pair")
    runtime_gate.add_argument("--gate-l2-exit-stress", action="store_true")
    parser.add_argument("--compare-ik-off")
    parser.add_argument("--compare-forward")
    parser.add_argument("--compare-forward-baseline")
    parser.add_argument("--compare-flat-baseline")
    parser.add_argument("--expected-end-x", type=float)
    parser.add_argument("--expected-end-z", type=float)
    parser.add_argument("--expected-heading", choices=tuple(HEADING_BITS))
    parser.add_argument(
        "--expected-relative-direction",
        choices=("forward", "backward", "left", "right"))
    parser.add_argument("--require-multilevel", action="store_true")
    parser.add_argument("--expected-scenes")
    parser.add_argument("--expect-switch-failure", action="store_true")
    args = parser.parse_args(argv)
    gate_l_mode = args.gate_l or args.gate_l_safety_only
    if args.expect_switch_failure and any((
            args.gate_a, args.gate_c, args.gate_d, args.gate_f,
            gate_l_mode, args.gate_l2_pair, args.gate_l2_exit_stress)):
        parser.error(
            "--expect-switch-failure may not combine with gate flags")
    if gate_l_mode and args.compare_flat_baseline is None:
        if args.expected_end_x is None:
            parser.error("--gate-l requires --expected-end-x")
        if args.expected_end_z is None:
            parser.error("--gate-l requires --expected-end-z")
        if args.expected_heading is None:
            parser.error("--gate-l requires --expected-heading")
        if args.compare_ik_off is None:
            parser.error("--gate-l requires --compare-ik-off")
        if args.expected_relative_direction is not None:
            parser.error(
                "--expected-relative-direction requires --compare-flat-baseline")
        if (args.compare_forward_baseline is not None and
                args.compare_ik_off is None):
            parser.error(
                "--compare-forward-baseline requires --compare-ik-off")
    if args.compare_flat_baseline is not None:
        if not args.gate_l or args.gate_l_safety_only:
            parser.error("--compare-flat-baseline requires --gate-l")
        if args.expected_heading is None:
            parser.error(
                "--compare-flat-baseline requires --expected-heading")
        if args.expected_relative_direction is None:
            parser.error(
                "--compare-flat-baseline requires "
                "--expected-relative-direction")
        if any((
                args.expected_end_x is not None,
                args.expected_end_z is not None,
                args.require_multilevel,
                args.compare_ik_off is not None,
                args.compare_forward is not None,
                args.compare_forward_baseline is not None)):
            parser.error(
                "--compare-flat-baseline may not combine with endpoint, "
                "multilevel, or paired comparisons")
    if args.gate_l2_pair is not None and any((
            args.expected_end_x is not None, args.expected_end_z is not None,
            args.expected_heading is not None, args.require_multilevel,
            args.compare_ik_off is not None, args.compare_forward is not None,
            args.compare_forward_baseline is not None,
            args.compare_flat_baseline is not None,
            args.expected_relative_direction is not None)):
        parser.error("--gate-l2-pair may not combine with Gate-L options")
    if args.gate_l2_exit_stress:
        if any((
                args.expected_end_x is not None, args.expected_end_z is not None,
                args.expected_heading is not None, args.require_multilevel,
                args.compare_forward is not None,
                args.compare_forward_baseline is not None,
                args.compare_flat_baseline is not None,
                args.expected_relative_direction is not None)):
            parser.error(
                "--gate-l2-exit-stress may not combine with endpoint or "
                "baseline options")
        if args.compare_ik_off is None:
            parser.error("--gate-l2-exit-stress requires --compare-ik-off")
    if not gate_l_mode and not args.gate_l2_exit_stress and any((
            args.compare_ik_off, args.compare_forward,
            args.compare_forward_baseline, args.compare_flat_baseline,
            args.expected_end_x is not None, args.expected_end_z is not None,
            args.expected_heading is not None,
            args.expected_relative_direction is not None,
            args.require_multilevel)):
        parser.error("Gate-L options require a Gate-L mode")
    expected_scene_ids = None
    if args.gate_f:
        if args.expected_scenes is None:
            parser.error("--gate-f requires --expected-scenes")
        expected_scene_ids = tuple(args.expected_scenes.split(","))
        if expected_scene_ids != LOCKED_SCENE_IDS:
            parser.error(
                "--expected-scenes must match the locked 14-scene catalog")
    elif args.expected_scenes is not None:
        parser.error("--expected-scenes requires --gate-f")
    rows = read_rows(args.log)
    summary = check_rows(
        rows, allow_ik=bool(
            gate_l_mode or args.gate_l2_pair or args.gate_l2_exit_stress))
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
    if args.gate_f:
        _print_gate_report(
            "gate-f", check_gate_f(rows, expected_scene_ids))
    if gate_l_mode:
        if args.compare_flat_baseline:
            report = check_gate_l4(
                rows, args.compare_flat_baseline,
                expected_heading=args.expected_heading,
                expected_relative_direction=args.expected_relative_direction)
            _print_gate_report("gate-l4", report)
        else:
            report = check_gate_l(
                rows,
                expected_end_x=args.expected_end_x,
                expected_end_z=args.expected_end_z,
                expected_heading=args.expected_heading,
                require_multilevel=args.require_multilevel,
                compare_ik_off=(
                    read_rows(args.compare_ik_off)
                    if args.compare_ik_off else None),
                compare_forward=(
                    read_rows(args.compare_forward)
                    if args.compare_forward else None),
                compare_forward_baseline=args.compare_forward_baseline,
                safety_only=args.gate_l_safety_only,
            )
            _print_gate_report(
                "gate-l-safety" if args.gate_l_safety_only else "gate-l",
                report)
    if args.gate_l2_pair:
        _print_gate_report(
            "gate-l2-pair",
            check_gate_l2_pair(rows, read_rows(args.gate_l2_pair)))
    if args.gate_l2_exit_stress:
        _print_gate_report(
            "gate-l2-exit-stress",
            check_gate_l2_exit_stress(
                rows, read_rows(args.compare_ik_off)))
    if args.expect_switch_failure:
        _print_gate_report("switch-failure", check_failed_switch(rows))
    print(
        f"VALID runtime-log frames={summary['frames']} "
        f"transitions={summary['transitions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
