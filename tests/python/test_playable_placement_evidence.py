import copy
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
import unittest
import zlib

from tests.python import test_playable_interaction_evidence as pickup


EXPECTED_STATES = (
    "Locomotion",
    "Preflight",
    "Align",
    "PickupReplay",
    "Hold",
    "Carry",
    "PlacePreflight",
    "PlaceAlign",
    "PlaceReplay",
    "PlaceRelease",
    "Locomotion",
)
CONTROL_RATE_HZ = 25
MIN_WALK_TICKS = 25
MIN_WALK_DISPLACEMENT_M = 2.00
INITIAL_PICKUP_DISTANCE_M = 2.80
STANDOFF_MIN_M = 0.35
STANDOFF_MAX_M = 0.45
SETTLE_TICKS = 5
SETTLE_MAX_SPEED_MPS = 0.10
REACH_WAYPOINT_POSITION_ERROR_LIMIT_M = 0.15
REACH_WAYPOINT_YAW_ERROR_LIMIT_DEGREES = 20.0
STATIONARY_CANDIDATE_COUNT = 186
STATIONARY_CANDIDATES_BY_RANGE = {
    0: frozenset(range(0, 93)),
    1: frozenset(range(118, 211)),
}
BRAKE_SIMULATION_SPEED_LIMIT_MPS = 0.05
INTERACTION_CLEARANCE_EPSILON_M = 0.001
MIN_CARRY_STAGING_TICKS = 25
MAX_CARRY_STAGING_TICKS = 150
MAX_WALK_TICKS = 250
MAX_PICKUP_TO_CARRY_TICKS = 375
MAX_PLACE_TICKS = 250
MAX_EVIDENCE_RECORDS = 900
MIN_PLACEMENT_HANDOFF_FRAMES = 18
MAX_PLACEMENT_HANDOFF_FRAMES = 75
PLACEMENT_SETTLED_FRAMES = 3
PICK_ENTRY_MAXIMUM_COST = 9.0
PLACE_ROOT_ERROR_LIMIT_M = 0.25
PLACE_YAW_ERROR_LIMIT_DEGREES = 25.0
FINAL_POSITION_ERROR_LIMIT_M = 0.02
FINAL_ORIENTATION_ERROR_LIMIT_DEGREES = 10.0
ACTUAL_SUPPORT_GAP_MIN_M = -0.005
ACTUAL_SUPPORT_GAP_MAX_M = 0.020
# Conservative fixed-six serialization bounds over the corresponding
# recomputations: yaw 1.6272e-4 degrees, speed 3.5856e-5 m/s, and final
# three-dimensional distance 2.2321e-6 m.
YAW_RECOMPUTATION_TOLERANCE_DEGREES = 2.0e-4
DISPLAYED_ROOT_SPEED_RECOMPUTATION_TOLERANCE_MPS = 4.0e-5
IK_DISPLAY_DEGREES_TOLERANCE = 3.0e-6
FINAL_POSITION_RECOMPUTATION_TOLERANCE_M = 2.5e-6

EXPECTED_IK = {
    "ik_maximum_request_position_m": 0.12,
    "ik_maximum_request_orientation_degrees": 25.0,
    "ik_accepted_position_m": 0.04,
    "ik_accepted_orientation_degrees": 15.0,
    "ik_damping": 0.05,
    "ik_finite_difference_radians": 0.001,
    "ik_orientation_scale_m_per_radian": 0.25,
    "ik_maximum_step_radians": 0.10,
    "ik_maximum_iterations": 8,
}


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _yaw_radians(rotation: list[float]) -> float:
    facing = pickup._quaternion_rotate(rotation, [0.0, 0.0, 1.0])
    return math.atan2(facing[0], facing[2])


def _yaw_error_degrees(left: list[float], right: list[float]) -> float:
    difference = _yaw_radians(left) - _yaw_radians(right)
    return abs(math.degrees(math.atan2(math.sin(difference), math.cos(difference))))


def _planar_distance(left, right):
    return math.hypot(left[0] - right[0], left[2] - right[2])


def _hand_sided_lateral(row):
    approach = pickup._quaternion_rotate(
        row["interaction_object_rotation"],
        row["interaction_approach_direction_object"],
    )
    length = math.hypot(approach[0], approach[2])
    right = [approach[2] / length, 0.0, -approach[0] / length]
    return [row["interaction_hand_side"] * value for value in right]


def _expected_interaction_waypoint(record: dict) -> tuple[list[float], float]:
    approach = pickup._quaternion_rotate(
        record["interaction_object_rotation"],
        record["interaction_approach_direction_object"],
    )
    approach_length = math.hypot(approach[0], approach[2])
    if approach_length <= 1.0e-5:
        raise _error("interaction approach direction has no planar component")
    approach = [
        approach[0] / approach_length,
        0.0,
        approach[2] / approach_length,
    ]
    unsigned_lateral = [approach[2], 0.0, -approach[0]]
    lateral_object = pickup._quaternion_rotate(
        pickup._quaternion_inverse(record["interaction_object_rotation"]),
        unsigned_lateral,
    )
    support = 0.5 * sum(
        abs(axis) * dimension
        for axis, dimension in zip(
            lateral_object, record["interaction_object_dimensions"]
        )
    )
    chord = (
        support
        + record["interaction_clearance_radius_m"]
        + INTERACTION_CLEARANCE_EPSILON_M
    )
    reach = record["reach_waypoint_position"]
    object_position = record["interaction_object_position"]
    radius_x = reach[0] - object_position[0]
    radius_z = reach[2] - object_position[2]
    distance = math.hypot(radius_x, radius_z)
    if not STANDOFF_MIN_M <= distance <= STANDOFF_MAX_M:
        raise _error("interaction Reach standoff is outside [0.35, 0.45] m")
    if (
        not math.isfinite(chord)
        or chord <= 0.0
        or chord > REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
        or chord > 2.0 * distance
    ):
        raise _error("interaction arc chord is invalid")
    theta = 2.0 * math.asin(chord / (2.0 * distance))

    def candidate(angle):
        cosine, sine = math.cos(angle), math.sin(angle)
        return [
            object_position[0] + cosine * radius_x + sine * radius_z,
            reach[1],
            object_position[2] - sine * radius_x + cosine * radius_z,
        ]

    plus, minus = candidate(+theta), candidate(-theta)
    hand_lateral = [
        record["interaction_hand_side"] * value
        for value in unsigned_lateral
    ]
    plus_score = sum(
        (value - origin) * axis
        for value, origin, axis in zip(plus, reach, hand_lateral)
    )
    minus_score = sum(
        (value - origin) * axis
        for value, origin, axis in zip(minus, reach, hand_lateral)
    )
    if plus_score > minus_score:
        expected = plus
    elif minus_score > plus_score:
        expected = minus
    else:
        expected = plus if record["interaction_hand_side"] == 1 else minus
    if (
        not math.isclose(
            _planar_distance(expected, reach), chord, abs_tol=2.0e-9
        )
        or not math.isclose(
            _planar_distance(expected, object_position),
            distance,
            abs_tol=2.0e-9,
        )
        or expected[1] != reach[1]
    ):
        raise _error("independently derived interaction arc violates an invariant")
    return expected, chord


def _expected_ik_fingerprint(values: dict) -> int:
    state = 14695981039346656037

    def byte(value: int) -> None:
        nonlocal state
        state ^= value & 0xFF
        state = (state * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    def u32(value: int) -> None:
        for shift in range(0, 32, 8):
            byte(value >> shift)

    def scalar(value: float) -> None:
        bits = struct.unpack("<I", struct.pack("<f", _float32(value)))[0]
        if (bits & 0x7FFFFFFF) == 0:
            bits = 0
        u32(bits)

    byte(0xD3)
    u32(0x494B4346)
    byte(0xD3)
    u32(0x5008)
    scalar(values["ik_maximum_request_position_m"])
    scalar(math.radians(values["ik_maximum_request_orientation_degrees"]))
    scalar(values["ik_accepted_position_m"])
    scalar(math.radians(values["ik_accepted_orientation_degrees"]))
    scalar(values["ik_damping"])
    scalar(values["ik_finite_difference_radians"])
    scalar(values["ik_orientation_scale_m_per_radian"])
    scalar(values["ik_maximum_step_radians"])
    u32(values["ik_maximum_iterations"] & 0xFFFFFFFF)
    return state or 0x9E3779B97F4A7C15


EXPECTED_IK_FINGERPRINT = _expected_ik_fingerprint(EXPECTED_IK)

FIELD_TYPES = (
    ("render_frame", "int"),
    ("runtime_tick", "int"),
    ("scheduler_phase", "int"),
    ("state", "str"),
    ("result", "str"),
    ("reason", "str"),
    ("object_state", "str"),
    ("attached", "bool"),
    ("owns_pose", "bool"),
    ("action", "str"),
    ("demo_phase", "str"),
    ("phase_counter", "int"),
    ("pickup_distance_m", "float"),
    ("walking_origin_displacement_m", "float"),
    ("displayed_root_speed_mps", "float"),
    ("left_stick_command", "vec3"),
    ("left_stick_magnitude", "float"),
    ("root_position", "vec3"),
    ("object_position", "vec3"),
    ("object_world_rotation", "quat"),
    ("locomotion_provider_kind", "str"),
    ("canonical_snapshot_used", "bool"),
    ("root_relocation_applied", "bool"),
    ("simulation_root_initialized_from_reach", "bool"),
    ("displayed_root_initialized_from_reach", "bool"),
    ("reach_waypoint_position", "vec3"),
    ("reach_waypoint_rotation", "quat"),
    ("interaction_waypoint_position", "vec3"),
    ("interaction_waypoint_rotation", "quat"),
    ("interaction_lateral_offset_m", "float"),
    ("interaction_object_position", "vec3"),
    ("interaction_object_dimensions", "vec3"),
    ("interaction_object_rotation", "quat"),
    ("interaction_approach_direction_object", "vec3"),
    ("interaction_clearance_radius_m", "float"),
    ("interaction_hand_side", "int"),
    ("reach_waypoint_position_error_m", "float"),
    ("reach_waypoint_yaw_error_degrees", "float"),
    ("reach_brake_latched", "bool"),
    ("stationary_candidate_count", "int"),
    ("stationary_constraint_active", "bool"),
    ("stationary_search_calls", "int"),
    ("stationary_transition_count", "int"),
    ("stationary_selected_frame", "int"),
    ("stationary_selected_range", "int"),
    ("brake_simulation_speed_mps", "float"),
    ("pick_resolver_maximum_m", "float"),
    ("pick_resolver_live_flat", "bool"),
    ("destination_handle_id", "int"),
    ("destination_generation", "int"),
    ("destination_affordance_id", "int"),
    ("destination_generation_changed_before_place", "bool"),
    ("placement_surface_resolver_calls", "int"),
    ("runtime_preview_calls", "int"),
    ("runtime_preview_source_state", "str"),
    ("free_selector_calls", "int"),
    ("external_match_input_calls", "int"),
    ("preview_mutation_count", "int"),
    ("far_selection_id", "int"),
    ("current_selection_id", "int"),
    ("submitted_selection_id", "int"),
    ("candidate_certified", "bool"),
    ("place_mode", "str"),
    ("place_source", "str"),
    ("place_source_id", "int"),
    ("preview_accepted", "bool"),
    ("preview_ready", "bool"),
    ("preview_ik_fingerprint", "int"),
    ("ik_maximum_request_position_m", "float"),
    ("ik_maximum_request_orientation_degrees", "float"),
    ("ik_accepted_position_m", "float"),
    ("ik_accepted_orientation_degrees", "float"),
    ("ik_damping", "float"),
    ("ik_finite_difference_radians", "float"),
    ("ik_orientation_scale_m_per_radian", "float"),
    ("ik_maximum_step_radians", "float"),
    ("ik_maximum_iterations", "int"),
    ("staging_root_position", "vec3"),
    ("staging_root_rotation", "quat"),
    ("preview_root_error_m", "float"),
    ("preview_yaw_error_degrees", "float"),
    ("carry_staging_tick", "int"),
    ("selected_staging_root", "bool"),
    ("direct_root_write", "bool"),
    ("place_phase", "str"),
    ("place_goal_position", "vec3"),
    ("place_goal_rotation", "quat"),
    ("place_position_error_m", "float"),
    ("place_orientation_error_degrees", "float"),
    ("requested_fit_accepted", "bool"),
    ("requested_support_gap_m", "float"),
    ("requested_lowest_corner_m", "float"),
    ("requested_highest_corner_m", "float"),
    ("requested_footprint_valid", "bool"),
    ("requested_overhead_valid", "bool"),
    ("actual_fit_accepted", "bool"),
    ("actual_support_gap_m", "float"),
    ("actual_lowest_corner_m", "float"),
    ("actual_highest_corner_m", "float"),
    ("actual_footprint_valid", "bool"),
    ("actual_bound_corners_valid", "bool"),
    ("actual_overhead_valid", "bool"),
    ("support_sweep_clear", "bool"),
    ("attachment_transition_count", "int"),
    ("attached_to_free_transition_count", "int"),
    ("destination_support_committed", "bool"),
    ("grasp_evidence_valid", "bool"),
    ("active_hand_joint", "int"),
    ("hand_constraint_weight", "float"),
    ("hand_constraint_validated", "bool"),
    ("hand_constraint_applied", "bool"),
    ("hand_constraint_reachable", "bool"),
    ("hand_constraint_used_clavicle", "bool"),
    ("hand_constraint_reach_shortfall_m", "float"),
    ("hand_constraint_calibration_rotation", "quat"),
    ("calibrated_hand_world_rotation", "quat"),
    ("hand_in_object_position", "vec3"),
    ("hand_in_object_rotation", "quat"),
    ("grasp_world_position", "vec3"),
    ("grasp_world_rotation", "quat"),
    ("joint_world_positions", "vec3_array"),
    ("joint_world_rotations", "quat_array"),
    ("pick_preview_epoch", "int"),
    ("pick_preview_consumed_epoch", "int"),
    ("pick_preview_snapshot_fingerprint", "int"),
    ("pick_preview_snapshot_source", "str"),
    ("pick_preview_epoch_count", "int"),
    ("runtime_pick_preview_calls", "int"),
    ("pick_preview_mutation_count", "int"),
    ("pick_preview_elapsed_ticks", "int"),
    ("pick_preview_pending_failure", "str"),
    ("pick_preview_failure_recorded", "bool"),
    ("pick_entry_selected_slot", "str"),
    ("pick_entry_selection_frozen", "bool"),
    ("pick_entry_freeze_tick", "int"),
    ("pick_interact_submission_count", "int"),
    ("pick_reservation_transition_count", "int"),
    ("pick_plus_identity", "str"),
    ("pick_plus_evaluation_order", "int"),
    ("pick_plus_waypoint_position", "vec3"),
    ("pick_plus_waypoint_rotation", "quat"),
    ("pick_plus_prospective_world_x", "float"),
    ("pick_plus_prospective_world_z", "float"),
    ("pick_plus_prospective_world_yaw_radians", "float"),
    ("pick_plus_clearance_chord_m", "float"),
    ("pick_plus_preserved_standoff_m", "float"),
    ("pick_plus_hand_score", "float"),
    ("pick_plus_path_feasible", "bool"),
    ("pick_plus_match_ready", "bool"),
    ("pick_plus_path_reason", "str"),
    ("pick_plus_match_reason", "str"),
    ("pick_plus_feasible_entry_frame", "int"),
    ("pick_plus_contact_frame", "int"),
    ("pick_plus_cost_available", "bool"),
    ("pick_plus_total_cost", "float"),
    ("pick_plus_eligible", "bool"),
    ("pick_plus_selected", "bool"),
    ("pick_minus_identity", "str"),
    ("pick_minus_evaluation_order", "int"),
    ("pick_minus_waypoint_position", "vec3"),
    ("pick_minus_waypoint_rotation", "quat"),
    ("pick_minus_prospective_world_x", "float"),
    ("pick_minus_prospective_world_z", "float"),
    ("pick_minus_prospective_world_yaw_radians", "float"),
    ("pick_minus_clearance_chord_m", "float"),
    ("pick_minus_preserved_standoff_m", "float"),
    ("pick_minus_hand_score", "float"),
    ("pick_minus_path_feasible", "bool"),
    ("pick_minus_match_ready", "bool"),
    ("pick_minus_path_reason", "str"),
    ("pick_minus_match_reason", "str"),
    ("pick_minus_feasible_entry_frame", "int"),
    ("pick_minus_contact_frame", "int"),
    ("pick_minus_cost_available", "bool"),
    ("pick_minus_total_cost", "float"),
    ("pick_minus_eligible", "bool"),
    ("pick_minus_selected", "bool"),
)
EXPECTED_KEYS = tuple(name for name, _ in FIELD_TYPES)

STATES = set(EXPECTED_STATES) | {"Disabled"}
RESULTS = {
    "None", "Accepted", "Succeeded", "Rejected", "Cancelled", "Failed",
    "Reset",
}
REASONS = {
    "None", "PackUnavailable", "TargetUnavailable", "TargetChanged",
    "OutOfRange", "NoCandidate", "PoorMatch", "BlockedPath",
    "CorrectionLimit", "Cancelled", "ContactPosition",
    "ContactOrientation", "JointLimit", "LostContact", "ClipEnded",
    "Reset", "SurfaceUnavailable", "SurfaceChanged",
    "PlacementOutOfBounds", "ReleasePosition", "ReleaseOrientation",
}
OBJECT_STATES = {"Free", "Targeted", "Attached", "Held"}
ACTIONS = {"none", "approach", "interact", "latch_destination", "stage", "place"}
DEMO_PHASES = {
    "approach", "settle", "interact", "pickup", "carry_preview",
    "carry_staging", "place", "handoff",
}
PLACE_PHASES = {"none", "align", "lower", "release", "retract", "finished"}
PLACE_MODES = {"none", "recorded_place", "reversed_pickup"}


class PlacementEvidenceValidationError(ValueError):
    pass


def _error(message: str) -> PlacementEvidenceValidationError:
    return PlacementEvidenceValidationError(message)


def _object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _error(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str):
    raise _error(f"non-finite JSON number {value}")


def _finite_number(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise _error(f"{label} must be a finite number")
    return float(value)


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _error(f"{label} must be an integer >= {minimum}")
    return value


def _vector(value, label: str, count: int) -> None:
    if type(value) is not list or len(value) != count:
        raise _error(f"{label} must contain exactly {count} numbers")
    for component in value:
        _finite_number(component, label)


def _quaternion(value, label: str) -> None:
    _vector(value, label, 4)
    norm_error = abs(math.sqrt(sum(component * component for component in value)) - 1.0)
    if norm_error > pickup.JOINT_QUATERNION_NORM_TOLERANCE:
        raise _error(f"{label} quaternion norm error exceeds 0.001")


def _expected_pick_entry_slots(record: dict) -> dict[str, dict]:
    """Reconstruct the immutable ordered arc pair without controller code."""
    approach = pickup._quaternion_rotate(
        record["interaction_object_rotation"],
        record["interaction_approach_direction_object"],
    )
    planar_length = math.hypot(approach[0], approach[2])
    if not math.isfinite(planar_length) or planar_length <= 1.0e-5:
        raise _error("pick-entry approach direction has no planar component")
    approach = [
        approach[0] / planar_length,
        0.0,
        approach[2] / planar_length,
    ]
    right_of_approach = [approach[2], 0.0, -approach[0]]
    lateral_object = pickup._quaternion_rotate(
        pickup._quaternion_inverse(record["interaction_object_rotation"]),
        right_of_approach,
    )
    support = 0.5 * sum(
        abs(axis) * dimension
        for axis, dimension in zip(
            lateral_object, record["interaction_object_dimensions"]
        )
    )
    chord = (
        support
        + record["interaction_clearance_radius_m"]
        + INTERACTION_CLEARANCE_EPSILON_M
    )
    reach = record["reach_waypoint_position"]
    target = record["interaction_object_position"]
    radial_x = reach[0] - target[0]
    radial_z = reach[2] - target[2]
    standoff = math.hypot(radial_x, radial_z)
    if (
        not math.isfinite(chord)
        or not math.isfinite(standoff)
        or not STANDOFF_MIN_M <= standoff <= STANDOFF_MAX_M
        or chord <= 0.0
        or chord > REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
        or chord > 2.0 * standoff
    ):
        raise _error("pick-entry chord or preserved standoff is invalid")
    arc = 2.0 * math.asin(chord / (2.0 * standoff))
    hand_lateral = [
        record["interaction_hand_side"] * value
        for value in right_of_approach
    ]
    yaw = _yaw_radians(record["reach_waypoint_rotation"])
    slots = {}
    for order, (prefix, identity, sign) in enumerate((
        ("pick_plus", "Plus", 1.0),
        ("pick_minus", "Minus", -1.0),
    )):
        cosine = math.cos(sign * arc)
        sine = math.sin(sign * arc)
        position = [
            target[0] + cosine * radial_x + sine * radial_z,
            reach[1],
            target[2] - sine * radial_x + cosine * radial_z,
        ]
        score = sum(
            (component - origin) * axis
            for component, origin, axis in zip(
                position, reach, hand_lateral
            )
        )
        if (
            any(not math.isfinite(value) for value in position)
            or not math.isfinite(score)
            or abs(_planar_distance(position, reach) - chord) > 2.0e-9
            or abs(_planar_distance(position, target) - standoff) > 2.0e-9
            or position[1] != reach[1]
        ):
            raise _error("independently reconstructed pick-entry slot is invalid")
        slots[prefix] = {
            "identity": identity,
            "evaluation_order": order,
            "waypoint_position": position,
            "waypoint_rotation": copy.deepcopy(
                record["reach_waypoint_rotation"]
            ),
            "prospective_world_x": position[0],
            "prospective_world_z": position[2],
            "prospective_world_yaw_radians": yaw,
            "clearance_chord_m": chord,
            "preserved_standoff_m": standoff,
            "hand_score": score,
        }
    return slots


def _pick_entry_eligible(record: dict, prefix: str) -> bool:
    return (
        record[f"{prefix}_path_feasible"]
        and record[f"{prefix}_match_ready"]
    )


def _expected_pick_entry_winner(record: dict) -> str | None:
    eligible = [
        prefix
        for prefix in ("pick_plus", "pick_minus")
        if _pick_entry_eligible(record, prefix)
    ]
    if not eligible:
        return None
    if len(eligible) == 1:
        return eligible[0]
    reconstructed = _expected_pick_entry_slots(record)
    plus_score = reconstructed["pick_plus"]["hand_score"]
    minus_score = reconstructed["pick_minus"]["hand_score"]
    if plus_score > minus_score:
        return "pick_plus"
    if minus_score > plus_score:
        return "pick_minus"
    return (
        "pick_plus"
        if record["interaction_hand_side"] == 1
        else "pick_minus"
    )


def _render_value(value, kind: str) -> str:
    if kind == "str":
        return json.dumps(value, ensure_ascii=True)
    if kind == "bool":
        return "true" if value else "false"
    if kind == "int":
        return str(value)
    if kind == "float":
        return f"{value:.6f}"
    if kind in {"vec3", "quat"}:
        return "[" + ",".join(f"{component:.6f}" for component in value) + "]"
    if kind in {"vec3_array", "quat_array"}:
        return "[" + ",".join(
            "[" + ",".join(f"{component:.6f}" for component in row) + "]"
            for row in value
        ) + "]"
    raise AssertionError(kind)


def _record_line(record: dict) -> str:
    return "{" + ",".join(
        json.dumps(name) + ":" + _render_value(record[name], kind)
        for name, kind in FIELD_TYPES
    ) + "}\n"


def _validate_record_types(record: dict) -> None:
    if type(record) is not dict or tuple(record) != EXPECTED_KEYS:
        raise _error("record fields/order differ from the required schema")
    for name, kind in FIELD_TYPES:
        value = record[name]
        if kind == "int":
            minimum = -1 if name in {
                "active_hand_joint", "carry_staging_tick",
                "interaction_hand_side",
                "pick_preview_epoch", "pick_preview_consumed_epoch",
                "pick_entry_freeze_tick",
                "stationary_selected_frame",
                "stationary_selected_range",
                "pick_plus_feasible_entry_frame",
                "pick_plus_contact_frame",
                "pick_minus_feasible_entry_frame",
                "pick_minus_contact_frame",
            } else 0
            _integer(value, name, minimum)
        elif kind == "float":
            _finite_number(value, name)
        elif kind == "bool":
            if type(value) is not bool:
                raise _error(f"{name} must be a JSON boolean")
        elif kind == "str":
            if type(value) is not str:
                raise _error(f"{name} must be a JSON string")
        elif kind == "vec3":
            _vector(value, name, 3)
        elif kind == "quat":
            _quaternion(value, name)
        elif kind == "vec3_array":
            if type(value) is not list or len(value) != len(pickup.FLAT_JOINT_NAMES):
                raise _error(f"{name} must contain exactly 23 joints")
            for index, row in enumerate(value):
                _vector(row, f"{name}[{index}]", 3)
        elif kind == "quat_array":
            if type(value) is not list or len(value) != len(pickup.FLAT_JOINT_NAMES):
                raise _error(f"{name} must contain exactly 23 joints")
            for index, row in enumerate(value):
                _quaternion(row, f"{name}[{index}]")

    if record["state"] not in STATES:
        raise _error("state has an invalid value")
    if record["result"] not in RESULTS:
        raise _error("result has an invalid value")
    if record["reason"] not in REASONS:
        raise _error("reason has an invalid value")
    if record["object_state"] not in OBJECT_STATES:
        raise _error("object_state has an invalid value")
    if record["action"] not in ACTIONS:
        raise _error("action has an invalid value")
    if record["demo_phase"] not in DEMO_PHASES:
        raise _error("demo_phase has an invalid value")
    if record["place_phase"] not in PLACE_PHASES:
        raise _error("place_phase has an invalid value")
    if record["place_mode"] not in PLACE_MODES:
        raise _error("place_mode has an invalid value")
    if record["runtime_preview_source_state"] not in {"none", "Carry"}:
        raise _error("runtime_preview_source_state must be none or Carry")
    if record["pick_preview_snapshot_source"] not in {"unset", "live_flat"}:
        raise _error("pick_preview_snapshot_source must be unset or live_flat")
    if record["pick_preview_pending_failure"] not in {
        "none", "no_path", "deadline"
    }:
        raise _error("pick_preview_pending_failure has an invalid value")
    if record["pick_entry_selected_slot"] not in {"none", "Plus", "Minus"}:
        raise _error("pick_entry_selected_slot has an invalid value")
    for prefix in ("pick_plus", "pick_minus"):
        if record[f"{prefix}_path_reason"] not in REASONS:
            raise _error(f"{prefix}_path_reason has an invalid value")
        if record[f"{prefix}_match_reason"] not in REASONS:
            raise _error(f"{prefix}_match_reason has an invalid value")
    if record["active_hand_joint"] not in {-1, 18, 22}:
        raise _error("active_hand_joint must be -1, 18, or 22")
    weight = record["hand_constraint_weight"]
    if not 0.0 <= weight <= 1.0:
        raise _error("hand_constraint_weight must be in [0, 1]")
    if record["hand_constraint_reach_shortfall_m"] < 0.0:
        raise _error("hand constraint reach shortfall must be nonnegative")
    if record["preview_root_error_m"] < 0.0 or record["preview_yaw_error_degrees"] < 0.0:
        raise _error("preview errors must be nonnegative")
    if record["brake_simulation_speed_mps"] < 0.0:
        raise _error("brake simulation speed must be nonnegative")

    grasp_valid = record["grasp_evidence_valid"]
    if grasp_valid != (record["active_hand_joint"] in {18, 22}):
        raise _error("grasp evidence and active hand identity disagree")
    attached_owned = record["attached"] and record["owns_pose"]
    if attached_owned and not grasp_valid:
        raise _error("attached owned pose requires final-FK grasp evidence")
    if grasp_valid:
        expected_position = [
            object_component + local_component
            for object_component, local_component in zip(
                record["object_position"],
                pickup._quaternion_rotate(
                    record["object_world_rotation"],
                    record["hand_in_object_position"],
                ),
            )
        ]
        expected_rotation = pickup._quaternion_multiply(
            record["object_world_rotation"],
            record["hand_in_object_rotation"],
        )
        if math.dist(expected_position, record["grasp_world_position"]) > pickup.GRASP_COMPOSITION_TOLERANCE:
            raise _error("grasp_world must equal object_world * hand_in_object")
        if pickup._quaternion_sign_distance(
            expected_rotation, record["grasp_world_rotation"]
        ) > pickup.GRASP_COMPOSITION_TOLERANCE:
            raise _error("grasp rotation must equal object_world * hand_in_object")
        if weight > 0.0 and not all(
            record[name]
            for name in (
                "hand_constraint_validated",
                "hand_constraint_applied",
                "hand_constraint_reachable",
            )
        ):
            raise _error("positive-weight grasp must be validated, applied, and reachable")
        if record["attached"] and math.isclose(weight, 1.0):
            hand = record["active_hand_joint"]
            if math.dist(
                record["joint_world_positions"][hand],
                record["grasp_world_position"],
            ) > pickup.HAND_POSITION_LIMIT_M:
                raise _error("attached full-weight hand position exceeds 0.01 m")
            calibrated = pickup._quaternion_multiply(
                record["joint_world_rotations"][hand],
                pickup._quaternion_inverse(
                    record["hand_constraint_calibration_rotation"]
                ),
            )
            if pickup._quaternion_sign_distance(
                calibrated, record["calibrated_hand_world_rotation"]
            ) > pickup.GRASP_COMPOSITION_TOLERANCE:
                raise _error("calibrated final-FK hand rotation is inconsistent")
            orientation_error = pickup._joint_rotation_step_degrees(
                record["calibrated_hand_world_rotation"],
                record["grasp_world_rotation"],
            )
            if orientation_error > pickup.HAND_CALIBRATED_ORIENTATION_LIMIT_DEGREES:
                raise _error("attached full-weight hand orientation exceeds 2 degrees")


def load_evidence(path: Path) -> list[dict]:
    path = Path(path)
    try:
        if not path.is_file():
            raise _error(f"evidence log is not a regular file: {path}")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _error(f"cannot read evidence log {path}: {error}") from error
    if not text or not text.endswith("\n"):
        raise _error("evidence log must be nonempty and newline terminated")

    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            record = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, PlacementEvidenceValidationError) as error:
            raise _error(f"invalid evidence line {line_number}: {error}") from error
        if type(record) is not dict or tuple(record) != EXPECTED_KEYS:
            raise _error(f"evidence line {line_number} fields/order must be {EXPECTED_KEYS}")
        _validate_record_types(record)
        if line + "\n" != _record_line(record):
            raise _error(f"evidence line {line_number} is not compact fixed-six-decimal JSON")
        records.append(record)
    return records


def _collapsed_states(records: list[dict]) -> tuple[str, ...]:
    collapsed = []
    for record in records:
        if not collapsed or collapsed[-1] != record["state"]:
            collapsed.append(record["state"])
    return tuple(collapsed)


def _validate_joint_continuity(records: list[dict]) -> None:
    for index, record in enumerate(records[1:], 1):
        previous = records[index - 1]
        authority_seam = (
            record["state"] != previous["state"]
            or record["owns_pose"] != previous["owns_pose"]
            or record["attached"] != previous["attached"]
        )
        for joint, joint_name in enumerate(pickup.FLAT_JOINT_NAMES):
            step = math.dist(
                previous["joint_world_positions"][joint],
                record["joint_world_positions"][joint],
            )
            if authority_seam and step > pickup.AUTHORITY_SEAM_TRANSLATION_LIMIT_M:
                raise _error(
                    f"joint {joint} ({joint_name}) authority seam translation exceeds 0.20 m"
                )
            if not authority_seam:
                if joint_name in pickup.DISTAL_LOWER_LIMB_JOINT_NAMES:
                    if step * CONTROL_RATE_HZ > pickup.DISTAL_LOWER_LIMB_TRANSLATION_SPEED_LIMIT_MPS:
                        raise _error(
                            f"joint {joint} ({joint_name}) translation speed exceeds 12 m/s"
                        )
                elif step > pickup.JOINT_TRANSLATION_LIMIT_M:
                    raise _error(
                        f"joint {joint} ({joint_name}) translation exceeds 0.20 m"
                    )
            rotation = pickup._joint_rotation_step_degrees(
                previous["joint_world_rotations"][joint],
                record["joint_world_rotations"][joint],
            )
            if rotation > pickup.JOINT_ROTATION_LIMIT_DEGREES:
                raise _error(
                    f"joint {joint} ({joint_name}) rotation exceeds 60 degrees"
                )


PICK_ENTRY_GEOMETRY_SUFFIXES = (
    "identity",
    "evaluation_order",
    "waypoint_position",
    "waypoint_rotation",
    "prospective_world_x",
    "prospective_world_z",
    "prospective_world_yaw_radians",
    "clearance_chord_m",
    "preserved_standoff_m",
    "hand_score",
)
PICK_ENTRY_OUTCOME_SUFFIXES = (
    "path_feasible",
    "match_ready",
    "path_reason",
    "match_reason",
    "feasible_entry_frame",
    "contact_frame",
    "cost_available",
    "total_cost",
    "eligible",
)


def _pick_epoch_signature(record: dict) -> tuple:
    return (
        record["pick_preview_snapshot_fingerprint"],
        record["pick_preview_snapshot_source"],
        record["pick_preview_epoch_count"],
        record["runtime_pick_preview_calls"],
        record["pick_preview_elapsed_ticks"],
        *(
            record[f"{prefix}_{suffix}"]
            for prefix in ("pick_plus", "pick_minus")
            for suffix in PICK_ENTRY_OUTCOME_SUFFIXES
        ),
    )


def _validate_pick_entry_evidence(
    records: list[dict], *, require_current_pack: bool
) -> bool:
    geometry_tolerance = 3.0e-5
    first = records[0]
    if first["pick_preview_epoch"] != -1:
        raise _error(
            "the first evidence row must be an exact pre-preview sentinel prefix"
        )
    previous = None
    first_frozen_index = None
    terminal_failure_index = None
    seen_interacts = 0
    seen_reservation = False
    first_preflight_index = None
    first_align_index = None

    for index, record in enumerate(records):
        expected_slots = _expected_pick_entry_slots(record)
        for prefix, expected in expected_slots.items():
            if record[f"{prefix}_identity"] != expected["identity"]:
                raise _error("pick-entry slot identity/order is not Plus then Minus")
            if record[f"{prefix}_evaluation_order"] != expected["evaluation_order"]:
                raise _error("pick-entry slot identity/order is not Plus then Minus")
            if math.dist(
                record[f"{prefix}_waypoint_position"],
                expected["waypoint_position"],
            ) > geometry_tolerance:
                raise _error(f"{prefix} signed arc waypoint geometry is corrupt")
            if pickup._quaternion_sign_distance(
                record[f"{prefix}_waypoint_rotation"],
                expected["waypoint_rotation"],
            ) > 2.0e-6:
                raise _error(f"{prefix} waypoint rotation differs from Reach")
            for suffix in (
                "prospective_world_x",
                "prospective_world_z",
                "prospective_world_yaw_radians",
                "clearance_chord_m",
                "preserved_standoff_m",
                "hand_score",
            ):
                if not math.isclose(
                    record[f"{prefix}_{suffix}"],
                    expected[suffix],
                    abs_tol=geometry_tolerance,
                ):
                    label = "hand score" if suffix == "hand_score" else suffix
                    raise _error(f"{prefix} {label} is inconsistent with arc geometry")
            if not math.isclose(
                record[f"{prefix}_prospective_world_x"],
                record[f"{prefix}_waypoint_position"][0],
                abs_tol=2.0e-6,
            ) or not math.isclose(
                record[f"{prefix}_prospective_world_z"],
                record[f"{prefix}_waypoint_position"][2],
                abs_tol=2.0e-6,
            ):
                raise _error(f"{prefix} prospective root does not match its waypoint")

        if index and any(
            record[f"{prefix}_{suffix}"] != first[f"{prefix}_{suffix}"]
            for prefix in ("pick_plus", "pick_minus")
            for suffix in PICK_ENTRY_GEOMETRY_SUFFIXES
        ):
            raise _error("immutable pick-entry pair geometry changed between rows")

        epoch = record["pick_preview_epoch"]
        consumed = record["pick_preview_consumed_epoch"]
        count = record["pick_preview_epoch_count"]
        calls = record["runtime_pick_preview_calls"]
        elapsed = record["pick_preview_elapsed_ticks"]
        fingerprint = record["pick_preview_snapshot_fingerprint"]
        if record["pick_preview_mutation_count"] != 0:
            raise _error("pickup preview mutation count must remain zero")
        if elapsed > MAX_WALK_TICKS:
            raise _error("pickup preview elapsed ticks exceed the 250 bound")
        if epoch < 0:
            if (
                epoch != -1
                or consumed != -1
                or fingerprint != 0
                or record["pick_preview_snapshot_source"] != "unset"
                or count != 0
                or calls != 0
                or elapsed != 0
            ):
                raise _error("pre-preview fields must use their exact sentinels")
            for prefix in ("pick_plus", "pick_minus"):
                sentinel = (
                    not record[f"{prefix}_path_feasible"]
                    and not record[f"{prefix}_match_ready"]
                    and record[f"{prefix}_path_reason"] == "NoCandidate"
                    and record[f"{prefix}_match_reason"] == "NoCandidate"
                    and record[f"{prefix}_feasible_entry_frame"] == -1
                    and record[f"{prefix}_contact_frame"] == -1
                    and not record[f"{prefix}_cost_available"]
                    and record[f"{prefix}_total_cost"] == 0.0
                    and not record[f"{prefix}_eligible"]
                    and not record[f"{prefix}_selected"]
                )
                if not sentinel:
                    raise _error("pre-preview slot outcomes must use exact sentinels")
        else:
            if (
                count <= 0
                or epoch != count - 1
                or calls != 2 * count
                or elapsed != count
            ):
                raise _error(
                    "each pickup preview epoch requires exactly one complete two-call pair"
                )
            if fingerprint == 0:
                raise _error("pickup preview epoch fingerprint must be nonzero")
            if record["pick_preview_snapshot_source"] != "live_flat":
                raise _error("pickup preview epoch must use one live_flat snapshot")
            if consumed < -1 or consumed > epoch:
                raise _error("pickup preview consumed epoch is outside the committed range")
            for prefix in ("pick_plus", "pick_minus"):
                path = record[f"{prefix}_path_feasible"]
                ready = record[f"{prefix}_match_ready"]
                path_reason = record[f"{prefix}_path_reason"]
                match_reason = record[f"{prefix}_match_reason"]
                entry = record[f"{prefix}_feasible_entry_frame"]
                contact = record[f"{prefix}_contact_frame"]
                if path != (path_reason == "None"):
                    raise _error(f"{prefix} path reason disagrees with path feasibility")
                if ready != (match_reason == "None"):
                    raise _error(f"{prefix} match reason disagrees with match readiness")
                if ready and not path:
                    raise _error(f"{prefix} cannot be match-ready without a feasible path")
                if path:
                    if entry < 0 or contact < 0 or entry >= contact:
                        raise _error(
                            f"{prefix} hard-feasible metadata requires ordered entry/contact frames"
                        )
                elif entry != -1 or contact != -1:
                    raise _error(
                        f"{prefix} infeasible path must not publish feasible frames"
                    )
                if not path and match_reason != path_reason:
                    raise _error(
                        f"{prefix} infeasible path/match reasons must preserve priority"
                    )
                if match_reason == "PoorMatch" and (not path or ready):
                    raise _error("PoorMatch may defer only a path-feasible slot")
                expected_available = ready or match_reason == "PoorMatch"
                if record[f"{prefix}_cost_available"] != expected_available:
                    raise _error(
                        f"{prefix} cost availability must be guarded by ready or PoorMatch"
                    )
                if expected_available and record[f"{prefix}_total_cost"] < 0.0:
                    raise _error(f"{prefix} available total cost must be nonnegative")
                eligible = _pick_entry_eligible(record, prefix)
                if record[f"{prefix}_eligible"] != eligible:
                    raise _error(f"{prefix} recorded eligibility is not independently derived")

        if previous is not None:
            previous_count = previous["pick_preview_epoch_count"]
            count_delta = count - previous_count
            if count_delta not in (0, 1):
                raise _error("pickup preview epoch count must advance one callback at a time")
            if count_delta == 0:
                if epoch != previous["pick_preview_epoch"]:
                    raise _error("pickup preview epoch changed without a complete callback")
                if fingerprint != previous["pick_preview_snapshot_fingerprint"]:
                    raise _error("pickup preview fingerprint changed within one epoch")
                if _pick_epoch_signature(record) != _pick_epoch_signature(previous):
                    raise _error("pickup preview outcomes changed outside a complete pair")
            else:
                if epoch != previous["pick_preview_epoch"] + 1:
                    raise _error("pickup preview epoch sequence is not contiguous")
                if calls != previous["runtime_pick_preview_calls"] + 2:
                    raise _error("pickup preview callback did not make exactly two calls")
                if elapsed != previous["pick_preview_elapsed_ticks"] + 1:
                    raise _error("pickup preview elapsed ticks stalled across a callback")
                if consumed > previous["pick_preview_epoch"]:
                    raise _error(
                        "steering consumed the current epoch in its provider callback tick"
                    )
            previous_consumed = previous["pick_preview_consumed_epoch"]
            if consumed < previous_consumed or consumed - previous_consumed > 1:
                raise _error("pickup preview consumed epoch must be monotonic and contiguous")
            if consumed > previous_consumed and consumed != previous["pick_preview_epoch"]:
                raise _error("pickup preview handoff must consume the preceding callback epoch")

            previous_pair_waiting = (
                previous["pick_preview_epoch"] >= 0
                and previous["pick_preview_consumed_epoch"]
                    < previous["pick_preview_epoch"]
                and not previous["pick_entry_selection_frozen"]
                and previous["pick_preview_pending_failure"] == "none"
            )
            if previous_pair_waiting:
                previous_epoch = previous["pick_preview_epoch"]
                if (
                    consumed != previous_epoch
                    and record["pick_preview_pending_failure"] == "none"
                ):
                    raise _error(
                        "the committed pickup preview pair must be consumed "
                        "immediately next native tick"
                    )
                previous_any_path = any(
                    previous[f"{prefix}_path_feasible"]
                    for prefix in ("pick_plus", "pick_minus")
                )
                previous_winner = _expected_pick_entry_winner(previous)
                current_failure = record["pick_preview_pending_failure"]
                current_frozen = record["pick_entry_selection_frozen"]
                if not previous_any_path:
                    if (
                        count_delta != 0
                        or current_failure != "no_path"
                        or current_frozen
                    ):
                        raise _error(
                            "a both-infeasible pair requires an immediate no_path "
                            "terminal decision with no retry"
                        )
                elif previous_winner is not None:
                    if (
                        count_delta != 0
                        or current_failure != "none"
                        or not current_frozen
                    ):
                        raise _error(
                            "an eligible pair must freeze immediately with no retry"
                        )
                elif previous["pick_preview_elapsed_ticks"] >= MAX_WALK_TICKS:
                    if (
                        count_delta != 0
                        or current_failure != "deadline"
                        or current_frozen
                    ):
                        raise _error(
                            "the 250-tick deferred pair requires an immediate "
                            "deadline terminal decision"
                        )
                elif (
                    count_delta != 1
                    or current_failure != "none"
                    or current_frozen
                ):
                    raise _error(
                        "a deferred pair must be consumed and replaced by exactly "
                        "one callback immediately next native tick; it may not stall"
                    )

        frozen = record["pick_entry_selection_frozen"]
        selected_slot = record["pick_entry_selected_slot"]
        freeze_tick = record["pick_entry_freeze_tick"]
        selected_flags = (
            record["pick_plus_selected"], record["pick_minus_selected"]
        )
        if not frozen:
            if first_frozen_index is not None:
                raise _error(
                    "pick-entry selection cannot unfreeze; the frozen boundary "
                    "must be monotonic"
                )
            if selected_slot != "none" or freeze_tick != -1 or any(selected_flags):
                raise _error("pick-entry selection fields changed before the freeze boundary")
            if epoch >= 0 and (
                record["left_stick_magnitude"] > 1.0e-4
                or record["reach_brake_latched"]
                or record["action"] == "interact"
            ):
                raise _error("slot navigation or braking occurred before selection freeze")
        else:
            if first_frozen_index is None:
                first_frozen_index = index
                if consumed != epoch:
                    raise _error("selection froze before consuming the committed pair")
                if freeze_tick != record["runtime_tick"]:
                    raise _error("pick-entry freeze tick is not the first frozen runtime tick")
            elif previous is not None:
                if not previous["pick_entry_selection_frozen"]:
                    pass
                elif (
                    selected_slot != previous["pick_entry_selected_slot"]
                    or freeze_tick != previous["pick_entry_freeze_tick"]
                    or selected_flags != (
                        previous["pick_plus_selected"],
                        previous["pick_minus_selected"],
                    )
                ):
                    raise _error("pick-entry frozen selection changed or fell back")
                if count != previous["pick_preview_epoch_count"]:
                    raise _error("post-freeze pickup preview or retry is forbidden")
            expected_winner = _expected_pick_entry_winner(record)
            expected_slot = {
                "pick_plus": "Plus", "pick_minus": "Minus"
            }.get(expected_winner)
            if expected_slot is None or selected_slot != expected_slot:
                raise _error("recorded pick-entry winner differs from independent ranking")
            if selected_flags != (
                expected_winner == "pick_plus",
                expected_winner == "pick_minus",
            ):
                raise _error("pick-entry selected booleans disagree with the winner")
            selected_prefix = expected_winner
            if math.dist(
                record["interaction_waypoint_position"],
                record[f"{selected_prefix}_waypoint_position"],
            ) > 2.0e-6 or pickup._quaternion_sign_distance(
                record["interaction_waypoint_rotation"],
                record[f"{selected_prefix}_waypoint_rotation"],
            ) > 2.0e-6:
                raise _error("frozen interaction waypoint is not the selected slot")

        if record["action"] == "interact":
            seen_interacts += 1
            if not frozen or first_frozen_index is None or index <= first_frozen_index:
                raise _error("Interact must follow the frozen pick-entry selection")
            if record["state"] != "Preflight":
                raise _error("Interact must publish Preflight")
        if record["pick_interact_submission_count"] != seen_interacts:
            raise _error("pick Interact submission counter disagrees with observed edges")
        if seen_interacts > 1:
            raise _error("pick Interact may be submitted exactly once")
        if record["pick_resolver_live_flat"] != (seen_interacts == 1):
            raise _error("exactly one post-freeze pick resolver must use live flat")

        if record["state"] == "Preflight":
            if first_preflight_index is not None:
                raise _error(
                    "pick submission must publish exactly one pending Preflight"
                )
            if not (
                seen_interacts == 1
                and record["action"] == "interact"
                and record["result"] == "None"
                and record["reason"] == "None"
                and record["object_state"] == "Free"
                and not record["attached"]
                and not record["owns_pose"]
                and record["pick_reservation_transition_count"] == 0
            ):
                raise _error(
                    "pick submission tick must be pending None/Free "
                    "Preflight, unattached, unowned, and unreserved"
                )
            first_preflight_index = index
        if record["state"] == "Align":
            if not (
                seen_interacts == 1
                and record["action"] == "none"
                and record["result"] == "Accepted"
                and record["reason"] == "None"
                and record["object_state"] == "Targeted"
                and not record["attached"]
                and record["owns_pose"]
            ):
                raise _error(
                    "normal pick Align must remain Accepted, Targeted, "
                    "unattached, and owned"
                )
            if first_align_index is None:
                first_align_index = index
                if (
                    first_preflight_index is None
                    or first_preflight_index + 1 != index
                ):
                    raise _error(
                        "accepted owned Align must follow pending Preflight "
                        "on the immediately next native tick"
                    )
                seen_reservation = True
        expected_reservations = 1 if seen_reservation else 0
        if record["pick_reservation_transition_count"] != expected_reservations:
            raise _error("pick reservation transition counter has invalid timing")
        if epoch >= 0 and not frozen and record["pick_reservation_transition_count"]:
            raise _error("pickup preview must not mutate reservation ownership")

        pending_failure = record["pick_preview_pending_failure"]
        failure_recorded = record["pick_preview_failure_recorded"]
        if pending_failure == "none":
            if failure_recorded:
                raise _error("failure-record flag cannot precede a terminal failure")
        else:
            terminal_failure_index = index
            if index != len(records) - 1:
                raise _error("no evidence row may follow the true terminal failure marker")
            if not failure_recorded:
                raise _error("terminal pickup preview failure was not flush-recorded")
            if index == 0 or epoch < 0 or consumed != epoch:
                raise _error("pickup preview failure appeared before its complete pair")
            if (
                record["left_stick_magnitude"] > 1.0e-9
                or record["action"] != "none"
            ):
                raise _error("terminal pickup preview failure row must have zero input")
            if frozen:
                raise _error("pickup preview failure cannot follow a frozen winner")
            if count != previous["pick_preview_epoch_count"] or (
                _pick_epoch_signature(record) != _pick_epoch_signature(previous)
            ):
                raise _error("terminal failure must retain the last complete pair")
            winner = _expected_pick_entry_winner(record)
            if pending_failure == "no_path":
                if any(
                    record[f"{prefix}_path_feasible"]
                    for prefix in ("pick_plus", "pick_minus")
                ):
                    raise _error("no-path failure requires both paths to be infeasible")
            elif (
                elapsed != MAX_WALK_TICKS
                or winner is not None
                or not any(
                    record[f"{prefix}_path_feasible"]
                    for prefix in ("pick_plus", "pick_minus")
                )
            ):
                raise _error(
                    "deadline requires 250 complete deferred preview epochs "
                    "with at least one feasible path"
                )
        previous = record

    if terminal_failure_index is not None:
        if require_current_pack:
            raise _error(
                "current pack requires a successful ready Plus selection"
            )
        return True
    if first_frozen_index is None:
        raise _error("successful evidence never froze an independently valid winner")
    if seen_interacts != 1:
        raise _error("frozen pick-entry selection requires one later Interact edge")
    if first_preflight_index is None or first_align_index is None:
        raise _error(
            "pending pick Preflight must reach immediate Align and reservation"
        )

    if require_current_pack:
        current = records[first_frozen_index]
        if not (
            current["pick_minus_identity"] == "Minus"
            and not current["pick_minus_path_feasible"]
            and current["pick_minus_path_reason"] == "BlockedPath"
            and not current["pick_minus_selected"]
            and current["pick_plus_identity"] == "Plus"
            and current["pick_plus_path_feasible"]
            and current["pick_plus_match_ready"]
            and current["pick_plus_path_reason"] == "None"
            and current["pick_plus_match_reason"] == "None"
            and current["pick_plus_feasible_entry_frame"] == 114
            and current["pick_plus_contact_frame"] == 139
            and current["pick_plus_cost_available"]
            and 0.0 <= current["pick_plus_total_cost"] <= PICK_ENTRY_MAXIMUM_COST
            and current["pick_plus_selected"]
            and current["pick_entry_selected_slot"] == "Plus"
        ):
            raise _error("current pack must select ready Plus 114/139 over blocked Minus")
    return False


def _validate_authoritative_baseline(record: dict) -> None:
    if not (
        record["render_frame"] == 0
        and record["runtime_tick"] == 1
        and record["state"] == "Locomotion"
        and record["result"] == "None"
        and record["reason"] == "None"
        and record["object_state"] == "Free"
        and not record["attached"]
        and not record["owns_pose"]
        and record["action"] == "none"
        and record["demo_phase"] == "settle"
        and record["phase_counter"] == 0
        and record["left_stick_command"] == [0.0, 0.0, 0.0]
        and record["left_stick_magnitude"] == 0.0
        and record["walking_origin_displacement_m"] == 0.0
        and record["displayed_root_speed_mps"] == 0.0
        and not record["reach_brake_latched"]
        and record["brake_simulation_speed_mps"] == 0.0
        and not record["stationary_constraint_active"]
        and record["stationary_search_calls"] == 0
        and record["stationary_transition_count"] == 0
        and record["stationary_selected_frame"] == -1
        and record["stationary_selected_range"] == -1
        and record["runtime_preview_calls"] == 0
        and record["runtime_preview_source_state"] == "none"
        and not record["preview_accepted"]
        and not record["preview_ready"]
        and record["preview_ik_fingerprint"] == 0
        and record["pick_preview_epoch"] == -1
        and record["pick_preview_consumed_epoch"] == -1
        and record["pick_preview_epoch_count"] == 0
        and record["runtime_pick_preview_calls"] == 0
        and record["pick_interact_submission_count"] == 0
        and record["pick_reservation_transition_count"] == 0
    ):
        raise _error(
            "row 0 baseline must be the exact tick-1 no-input, no-preview, "
            "unreserved Locomotion/None/Free settle sentinel"
        )


def validate_evidence(
    records: list[dict], *, require_current_pack: bool = False
) -> None:
    if not records:
        raise _error("placement evidence has no records")
    if len(records) > MAX_EVIDENCE_RECORDS:
        raise _error(f"placement evidence must contain at most {MAX_EVIDENCE_RECORDS} records")
    _validate_record_types(records[0])
    _validate_authoritative_baseline(records[0])
    first_attachment = next(
        (index for index, record in enumerate(records) if record["attached"]),
        len(records),
    )
    observed_attachment_transitions = 0
    observed_release_transitions = 0
    stationary_interact_seen = False
    first_brake_speed = None
    for index, record in enumerate(records):
        _validate_record_types(record)
        if record["render_frame"] != index:
            raise _error("render_frame must start at zero and be sequential")
        if record["scheduler_phase"] != 0:
            raise _error("scheduler phase must remain zero at exact 25 Hz")
        if index and record["runtime_tick"] != records[index - 1]["runtime_tick"] + 1:
            raise _error("runtime_tick must advance exactly once per 25 Hz record")
        if index and record["attached"] != records[index - 1]["attached"]:
            observed_attachment_transitions += 1
            if records[index - 1]["attached"] and not record["attached"]:
                observed_release_transitions += 1
        if (
            record["attachment_transition_count"]
                != observed_attachment_transitions
            or record["attached_to_free_transition_count"]
                != observed_release_transitions
        ):
            raise _error(
                "attachment transition counters do not match observed edges"
            )
        if record["locomotion_provider_kind"] != "live_flat":
            raise _error("placement evidence requires the live_flat provider throughout")
        if record["canonical_snapshot_used"]:
            raise _error("placement evidence must never use a canonical snapshot")
        if record["root_relocation_applied"]:
            raise _error("placement evidence must never relocate the root")
        if record["simulation_root_initialized_from_reach"] or record["displayed_root_initialized_from_reach"]:
            raise _error("the Reach row may be a waypoint only, never a root initializer")
        if record["direct_root_write"]:
            raise _error("placement auto-demo must not directly write either root")
        if record["placement_surface_resolver_calls"] != 0:
            raise _error("placement auto-demo must not call the placement surface resolver")
        if record["free_selector_calls"] != 0:
            raise _error("placement auto-demo must not call a free place selector")
        if record["external_match_input_calls"] != 0:
            raise _error("placement auto-demo must not construct PlaceMatchInput")
        if record["preview_mutation_count"] != 0:
            raise _error("place preview must not mutate runtime or diagnostics")

        if record["stationary_candidate_count"] != STATIONARY_CANDIDATE_COUNT:
            raise _error(
                "authoritative placement evidence requires exactly 186 "
                "stationary candidates"
            )
        expected_stationary_constraint = (
            record["reach_brake_latched"]
            and not stationary_interact_seen
            and (
                record["state"] == "Locomotion"
                or record["action"] == "interact"
            )
        )
        if (
            record["stationary_constraint_active"]
            != expected_stationary_constraint
        ):
            raise _error(
                "stationary constraint must be exactly brake-latched "
                "pre-Interact Locomotion"
            )
        calls = record["stationary_search_calls"]
        transitions = record["stationary_transition_count"]
        selected_frame = record["stationary_selected_frame"]
        selected_range = record["stationary_selected_range"]
        if transitions > calls:
            raise _error(
                "stationary transition count exceeds constrained-search calls"
            )
        if calls == 0:
            if transitions != 0 or selected_frame != -1 or selected_range != -1:
                raise _error(
                    "stationary selection fields must remain -1 before the "
                    "first constrained search"
                )
        else:
            candidates = STATIONARY_CANDIDATES_BY_RANGE.get(selected_range)
            if candidates is None or selected_frame not in candidates:
                raise _error(
                    "stationary selected frame is not a derived candidate "
                    "in its selected range"
                )
        previous = records[index - 1] if index else None
        previous_calls = (
            0 if previous is None else previous["stationary_search_calls"]
        )
        previous_transitions = (
            0
            if previous is None
            else previous["stationary_transition_count"]
        )
        if calls < previous_calls or transitions < previous_transitions:
            raise _error("stationary counters must be monotonic")
        if not record["stationary_constraint_active"] and (
            calls != previous_calls or transitions != previous_transitions
        ):
            raise _error(
                "only a stationary-constrained pre-Interact search may "
                "change stationary counters"
            )

        first_brake_row = record["reach_brake_latched"] and (
            previous is None or not previous["reach_brake_latched"]
        )
        if not record["reach_brake_latched"]:
            if record["brake_simulation_speed_mps"] != 0.0:
                raise _error(
                    "brake simulation speed changed before the latch"
                )
        else:
            if first_brake_speed is None:
                first_brake_speed = record["brake_simulation_speed_mps"]
            if not math.isclose(
                record["brake_simulation_speed_mps"],
                first_brake_speed,
                abs_tol=1.0e-9,
            ):
                raise _error("brake simulation speed latch must be one-way")
            if (
                record["brake_simulation_speed_mps"]
                > BRAKE_SIMULATION_SPEED_LIMIT_MPS
            ):
                raise _error("brake simulation speed exceeds 0.05 m/s")
        if first_brake_row:
            previous_calls = 0 if previous is None else previous[
                "stationary_search_calls"
            ]
            previous_transitions = 0 if previous is None else previous[
                "stationary_transition_count"
            ]
            if (
                not record["stationary_constraint_active"]
                or calls <= previous_calls
                or transitions <= previous_transitions
            ):
                raise _error(
                    "the first brake-latched row must already contain a "
                    "constrained search and stationary transition"
                )
        if record["action"] == "interact":
            if transitions < 1:
                raise _error(
                    "Interact requires at least one prior stationary transition"
                )
            stationary_interact_seen = True
        if index < first_attachment and (
            math.dist(
                record["interaction_object_position"],
                record["object_position"],
            ) > 2.0e-6
            or pickup._quaternion_sign_distance(
                record["interaction_object_rotation"],
                record["object_world_rotation"],
            ) > 2.0e-6
        ):
            raise _error(
                "interaction object provenance must match the observed "
                "pre-attachment object pose"
            )

        expected_distance = math.hypot(
            record["root_position"][0] - record["object_position"][0],
            record["root_position"][2] - record["object_position"][2],
        )
        if not math.isclose(record["pickup_distance_m"], expected_distance, abs_tol=2.0e-6):
            raise _error("pickup_distance_m does not match displayed root and object")
        expected_left_stick_magnitude = math.sqrt(
            sum(component * component for component in record["left_stick_command"])
        )
        if not math.isclose(
            record["left_stick_magnitude"],
            expected_left_stick_magnitude,
            abs_tol=2.0e-6,
        ):
            raise _error("left-stick input magnitude does not match the command")
        if index and (
            math.dist(
                record["reach_waypoint_position"],
                records[0]["reach_waypoint_position"],
            ) > 2.0e-6
            or pickup._quaternion_sign_distance(
                record["reach_waypoint_rotation"],
                records[0]["reach_waypoint_rotation"],
            ) > 2.0e-6
        ):
            raise _error("Reach navigation waypoint must remain stable")
        if record["interaction_hand_side"] not in {-1, 1}:
            raise _error("interaction hand side must be -1 or 1")
        if any(
            dimension <= 0.0
            for dimension in record["interaction_object_dimensions"]
        ):
            raise _error("interaction object dimensions must be positive")
        if record["interaction_clearance_radius_m"] < 0.0:
            raise _error("interaction clearance radius must be nonnegative")
        approach_length = math.sqrt(sum(
            component * component
            for component in record["interaction_approach_direction_object"]
        ))
        if (
            abs(approach_length - 1.0) > 2.0e-5
            or abs(record["interaction_approach_direction_object"][1]) >
                2.0e-5
        ):
            raise _error(
                "interaction approach direction must be unit and object-horizontal"
            )
        if index and (
            math.dist(
                record["interaction_object_position"],
                records[0]["interaction_object_position"],
            ) > 2.0e-6
            or math.dist(
                record["interaction_object_dimensions"],
                records[0]["interaction_object_dimensions"],
            ) > 2.0e-6
            or pickup._quaternion_sign_distance(
                record["interaction_object_rotation"],
                records[0]["interaction_object_rotation"],
            ) > 2.0e-6
            or math.dist(
                record["interaction_approach_direction_object"],
                records[0]["interaction_approach_direction_object"],
            ) > 2.0e-6
            or not math.isclose(
                record["interaction_clearance_radius_m"],
                records[0]["interaction_clearance_radius_m"],
                abs_tol=2.0e-6,
            )
            or record["interaction_hand_side"] !=
                records[0]["interaction_hand_side"]
        ):
            raise _error(
                "interaction waypoint inputs must remain stable"
            )
        expected_slots = _expected_pick_entry_slots(record)
        expected_lateral_offset = expected_slots["pick_plus"][
            "clearance_chord_m"
        ]
        if expected_lateral_offset > REACH_WAYPOINT_POSITION_ERROR_LIMIT_M:
            raise _error(
                "interaction lateral offset exceeds the 0.15 m Reach neighborhood"
            )
        if not math.isclose(
            record["interaction_lateral_offset_m"],
            expected_lateral_offset,
            abs_tol=2.0e-6,
        ):
            raise _error(
                "interaction arc chord does not match oriented object support"
            )
        if record["active_hand_joint"] in {18, 22}:
            expected_hand_side = (
                1 if record["active_hand_joint"] == 22 else -1
            )
            if record["interaction_hand_side"] != expected_hand_side:
                raise _error(
                    "interaction hand side disagrees with the active hand"
                )
        expected_waypoint_position_error = math.hypot(
            record["root_position"][0] - record["reach_waypoint_position"][0],
            record["root_position"][2] - record["reach_waypoint_position"][2],
        )
        if not math.isclose(
            record["reach_waypoint_position_error_m"],
            expected_waypoint_position_error,
            abs_tol=2.0e-6,
        ):
            raise _error("waypoint position error does not match live root")
        expected_waypoint_yaw_error = _yaw_error_degrees(
            record["joint_world_rotations"][0],
            record["reach_waypoint_rotation"],
        )
        if not math.isclose(
            record["reach_waypoint_yaw_error_degrees"],
            expected_waypoint_yaw_error,
            rel_tol=0.0,
            abs_tol=YAW_RECOMPUTATION_TOLERANCE_DEGREES,
        ):
            raise _error("waypoint yaw error does not match live root")
        origin = records[0]["root_position"]
        expected_walk = math.hypot(
            record["root_position"][0] - origin[0],
            record["root_position"][2] - origin[2],
        )
        if not math.isclose(
            record["walking_origin_displacement_m"], expected_walk, abs_tol=2.0e-6
        ):
            raise _error("walking_origin_displacement_m does not match displayed-root motion")
        if index:
            expected_speed = math.hypot(
                record["root_position"][0] - records[index - 1]["root_position"][0],
                record["root_position"][2] - records[index - 1]["root_position"][2],
            ) * CONTROL_RATE_HZ
            if not math.isclose(
                record["displayed_root_speed_mps"],
                expected_speed,
                rel_tol=0.0,
                abs_tol=DISPLAYED_ROOT_SPEED_RECOMPUTATION_TOLERANCE_MPS,
            ):
                raise _error("displayed_root_speed_mps does not match exact 25 Hz motion")

    pre_release_owned_states = {
        "Carry", "PlacePreflight", "PlaceAlign", "PlaceReplay"
    }
    for row in records:
        if row["state"] not in pre_release_owned_states:
            continue
        if not (
            row["object_state"] == "Held"
            and row["attached"]
            and row["owns_pose"]
            and row["grasp_evidence_valid"]
            and math.isclose(row["hand_constraint_weight"], 1.0, abs_tol=1.0e-9)
            and row["hand_constraint_validated"]
            and row["hand_constraint_applied"]
            and row["hand_constraint_reachable"]
        ):
            raise _error(
                "Carry and pre-release Place rows require a Held, attached, "
                "owned, validated full-weight grasp"
            )

    _validate_joint_continuity(records)
    if _validate_pick_entry_evidence(
        records, require_current_pack=require_current_pack
    ):
        return
    if _collapsed_states(records) != EXPECTED_STATES:
        raise _error("collapsed states do not match walk, pickup, carry, place, handoff")
    handoff = [row for row in records if row["demo_phase"] == "handoff"]
    if (
        len(handoff) < MIN_PLACEMENT_HANDOFF_FRAMES
        or len(handoff) > MAX_PLACEMENT_HANDOFF_FRAMES
        or handoff != records[-len(handoff):]
        or [row["phase_counter"] for row in handoff]
        != list(range(1, len(handoff) + 1))
        or any(
            row["state"] != "Locomotion"
            or row["action"] != "none"
            or math.hypot(
                row["left_stick_command"][0], row["left_stick_command"][2]
            ) > 1.0e-9
            for row in handoff
        )
    ):
        raise _error(
            "evidence must end with eighteen to seventy-five numbered "
            "no-input "
            "Locomotion handoff rows"
        )

    interact_indices = [i for i, row in enumerate(records) if row["action"] == "interact"]
    if len(interact_indices) != 1:
        raise _error("Interact must occur exactly once")
    interact_index = interact_indices[0]
    place_indices = [i for i, row in enumerate(records) if row["action"] == "place"]
    if len(place_indices) != 1:
        raise _error("Place must occur exactly once")
    place_index = place_indices[0]
    if any(row["result"] in {"Rejected", "Cancelled", "Failed", "Reset"} for row in records[interact_index:]):
        raise _error("placement evidence contains an unsuccessful runtime result")

    if not records[0]["pickup_distance_m"] > INITIAL_PICKUP_DISTANCE_M:
        raise _error("initial pickup distance must be strictly greater than 2.80 m")
    approach_count = 0
    for row in records[1:]:
        if row["demo_phase"] != "approach":
            break
        approach_count += 1
    if approach_count < MIN_WALK_TICKS:
        raise _error("evidence requires at least 25 consecutive initial approach rows")
    if approach_count > MAX_WALK_TICKS:
        raise _error("approach exceeded its 250-tick bound")
    initial_approach = records[1 : approach_count + 1]
    if (
        [row["phase_counter"] for row in initial_approach]
            != list(range(1, approach_count + 1))
        or any(
            row["action"] != "approach"
            or row["state"] != "Locomotion"
            or row["owns_pose"]
            or row["attached"]
            or row["object_state"] != "Free"
            or row["left_stick_magnitude"] <= 1.0e-4
            for row in initial_approach
        )
    ):
        raise _error(
            "initial approach rows must be consecutive, numbered from one, "
            "nonzero-input, and non-owned Locomotion/Free"
        )
    walking_prefix = records[:interact_index]
    if any(
        row["state"] != "Locomotion"
        or row["owns_pose"]
        or row["attached"]
        or row["object_state"] != "Free"
        for row in walking_prefix
    ):
        raise _error("walking prefix must be non-owned unattached Locomotion")
    if max(row["walking_origin_displacement_m"] for row in walking_prefix) < MIN_WALK_DISPLACEMENT_M:
        raise _error("displayed-root walk displacement must be at least 2.00 m")
    if not records[interact_index - 1]["pickup_distance_m"] < records[0]["pickup_distance_m"]:
        raise _error("walking must make net pickup-distance progress")
    if any(math.hypot(*[row["left_stick_command"][axis] for axis in (0, 2)]) <= 1.0e-4 for row in initial_approach):
        raise _error("every approach row must use ordinary nonzero left-stick input")
    brake_indices = [
        index for index, row in enumerate(records) if row["reach_brake_latched"]
    ]
    if (
        not brake_indices
        or brake_indices[0] < approach_count + 1
        or any(
            row["reach_brake_latched"] != (index >= brake_indices[0])
            for index, row in enumerate(records)
        )
    ):
        raise _error(
            "the observed Reach brake latch must transition once after approach"
        )
    if any(
        row["left_stick_magnitude"] > 1.0e-4
        for row in records[brake_indices[0]:interact_index + 1]
    ):
        raise _error(
            "Reach braking and settle must remain no-input through Interact"
        )

    if interact_index < SETTLE_TICKS:
        raise _error("Interact has no settled prefix")
    settled = records[interact_index - SETTLE_TICKS:interact_index]
    if [row["phase_counter"] for row in settled] != list(range(1, SETTLE_TICKS + 1)):
        raise _error("Interact must follow five consecutive numbered settle rows")
    if any(
        row["demo_phase"] != "settle"
        or not STANDOFF_MIN_M <= row["pickup_distance_m"] <= STANDOFF_MAX_M
        or row["displayed_root_speed_mps"] > SETTLE_MAX_SPEED_MPS
        for row in settled
    ):
        raise _error("settle rows must remain in the standoff band below 0.10 m/s")
    interact = records[interact_index]
    aligned_rows = settled + [interact]
    if any(
        row["reach_waypoint_position_error_m"] >
            REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
        for row in aligned_rows
    ):
        raise _error(
            "Reach waypoint position error must be at most 0.15 m through Interact"
        )
    if any(
        math.hypot(
            row["root_position"][0] -
                row["interaction_waypoint_position"][0],
            row["root_position"][2] -
                row["interaction_waypoint_position"][2],
        ) > REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
        for row in aligned_rows
    ):
        raise _error(
            "selected interaction-waypoint position error must be at most "
            "0.15 m through Interact"
        )
    if any(
        row["reach_waypoint_yaw_error_degrees"] >
            REACH_WAYPOINT_YAW_ERROR_LIMIT_DEGREES
        for row in aligned_rows
    ):
        raise _error(
            "Reach waypoint yaw error must be at most 20 degrees through Interact"
        )
    if not math.isclose(interact["pick_resolver_maximum_m"], 1.0, abs_tol=1.0e-9):
        raise _error("Interact must retain the exact 1.00 m pickup resolver")
    if not interact["pick_resolver_live_flat"]:
        raise _error("Interact must resolve from the current live flat pose")

    destination_identity = (
        records[0]["destination_handle_id"],
        records[0]["destination_generation"],
        records[0]["destination_affordance_id"],
    )
    if 0 in destination_identity:
        raise _error("authored destination identity must be nonzero")
    if any(
        (row["destination_handle_id"], row["destination_generation"], row["destination_affordance_id"])
        != destination_identity
        for row in records
    ):
        raise _error("retained authored destination identity must remain stable")
    if any(row["destination_generation_changed_before_place"] for row in records[:place_index + 1]):
        raise _error("destination generation changed before Place")

    previous_preview_calls = 0
    for row in records:
        calls = row["runtime_preview_calls"]
        if calls < previous_preview_calls or calls - previous_preview_calls > 1:
            raise _error("runtime preview call count must be monotonic with at most one call per tick")
        called = calls != previous_preview_calls
        if called != (row["runtime_preview_source_state"] == "Carry"):
            raise _error("every runtime preview call must originate from valid Carry")
        previous_preview_calls = calls

    carry_indices = [i for i, row in enumerate(records) if row["state"] == "Carry"]
    if carry_indices[-1] - carry_indices[0] + 1 != len(carry_indices):
        raise _error("Carry rows must be consecutive")
    preview_indices = [
        i for i in carry_indices
        if records[i]["demo_phase"] == "carry_preview"
    ]
    if len(preview_indices) != 1:
        raise _error("initial Carry preview must occur exactly once")
    first_preview_index = preview_indices[0]
    first_preview = records[first_preview_index]
    if any(
        row["demo_phase"] != "pickup"
        or row["preview_accepted"]
        or row["runtime_preview_source_state"] != "none"
        for row in records[carry_indices[0]:first_preview_index]
    ):
        raise _error("Carry prelude must remain ordinary pickup handoff")
    staging_indices = [
        i for i in carry_indices
        if records[i]["demo_phase"] == "carry_staging"
    ]
    if not MIN_CARRY_STAGING_TICKS <= len(staging_indices) <= MAX_CARRY_STAGING_TICKS:
        raise _error("ordinary Carry staging must last from 25 through 150 ticks")
    if staging_indices[-1] - staging_indices[0] + 1 != len(staging_indices):
        raise _error("Carry staging rows must be consecutive")
    if staging_indices[0] != first_preview_index + 1:
        raise _error("ordinary Carry staging must immediately follow the far preview")
    staging = [records[i] for i in staging_indices]
    if [row["carry_staging_tick"] for row in staging] != list(range(len(staging))):
        raise _error("Carry staging ticks must be numbered consecutively from zero")
    if any(
        row["action"] != "stage"
        or row["runtime_preview_source_state"] != "Carry"
        or not row["selected_staging_root"]
        for row in staging
    ):
        raise _error(
            "every staging Carry row must use an ordinary live selected preview"
        )
    if not (
        first_preview["action"] == "latch_destination"
        and first_preview["preview_accepted"]
        and first_preview["candidate_certified"]
        and first_preview["selected_staging_root"]
        and not first_preview["preview_ready"]
        and first_preview["far_selection_id"] != 0
        and first_preview["current_selection_id"] == first_preview["far_selection_id"]
        and first_preview["runtime_preview_source_state"] == "Carry"
        and first_preview["carry_staging_tick"] == -1
    ):
        raise _error(
            "initial Carry preview must be accepted, selected, far, not ready, "
            "and excluded from the staging count"
        )
    if first_preview["preview_root_error_m"] <= PLACE_ROOT_ERROR_LIMIT_M and first_preview["preview_yaw_error_degrees"] <= PLACE_YAW_ERROR_LIMIT_DEGREES:
        raise _error("initial far preview must not be intrinsically ready")
    fingerprint = first_preview["preview_ik_fingerprint"]
    if fingerprint != EXPECTED_IK_FINGERPRINT:
        raise _error(
            "preview IK fingerprint does not match the canonical runtime IK hash"
        )
    for row in [first_preview] + staging + [records[place_index]]:
        if not row["preview_accepted"] or not row["candidate_certified"]:
            raise _error("Carry and Place-edge previews must remain accepted and certified")
        if row["preview_ik_fingerprint"] != fingerprint:
            raise _error("IK fingerprint changed between far and staged previews")
        for name, expected in EXPECTED_IK.items():
            matches = (
                math.isclose(
                    row[name],
                    expected,
                    rel_tol=0.0,
                    abs_tol=IK_DISPLAY_DEGREES_TOLERANCE,
                )
                if name == "ik_maximum_request_orientation_degrees"
                else row[name] == expected
            )
            if not matches:
                raise _error(f"preview IK field {name} differs from runtime configuration")
    if any(row["preview_ready"] for row in staging):
        raise _error("staging Carry cannot be ready before the PlacePreflight edge")
    root_displacement = math.hypot(
        records[place_index]["root_position"][0] - first_preview["root_position"][0],
        records[place_index]["root_position"][2] - first_preview["root_position"][2],
    )
    object_displacement = math.hypot(
        records[place_index]["object_position"][0] - first_preview["object_position"][0],
        records[place_index]["object_position"][2] - first_preview["object_position"][2],
    )
    if root_displacement <= 0.20 or object_displacement <= 0.20:
        raise _error("Carry root and object displacement must both exceed 0.20 m")
    if any(
        math.hypot(row["left_stick_command"][0], row["left_stick_command"][2]) <= 1.0e-4
        for row in staging
    ):
        raise _error("Carry staging must use ordinary preview-guided left-stick input")
    ready = records[place_index]
    if not (
        ready["preview_ready"]
        and ready["selected_staging_root"]
        and ready["preview_root_error_m"] <= PLACE_ROOT_ERROR_LIMIT_M
        and ready["preview_yaw_error_degrees"] <= PLACE_YAW_ERROR_LIMIT_DEGREES
    ):
        raise _error("runtime Place edge requires the exact ready live preview")
    place = records[place_index]
    if place["state"] != "PlacePreflight":
        raise _error("Place action must publish PlacePreflight")
    if not (
        place["preview_ready"]
        and place["selected_staging_root"]
        and place["current_selection_id"] == place["submitted_selection_id"]
        and place["submitted_selection_id"] != place["far_selection_id"]
    ):
        raise _error("Place must submit only the current distinct ready live selection ID")
    if place_index - first_preview_index > MAX_PLACE_TICKS:
        raise _error("placement exceeded its 250-tick bound")
    first_carry_index = carry_indices[0]
    if first_carry_index - interact_index > MAX_PICKUP_TO_CARRY_TICKS:
        raise _error("pickup-to-Carry exceeded its 375-tick bound")

    release_edges = [
        i for i in range(1, len(records))
        if records[i - 1]["attached"] and not records[i]["attached"]
    ]
    if len(release_edges) != 1 or records[release_edges[0]]["state"] != "PlaceRelease":
        raise _error("attachment must release exactly once in PlaceRelease")
    if records[-1]["attachment_transition_count"] != 2 or records[-1]["attached_to_free_transition_count"] != 1:
        raise _error("attachment transition counters must prove one attach and one release")

    final = records[-1]
    if not (
        final["state"] == "Locomotion"
        and final["result"] == "Succeeded"
        and final["reason"] == "None"
        and final["object_state"] == "Free"
        and not final["attached"]
        and not final["owns_pose"]
    ):
        raise _error("final state must be Locomotion/Succeeded/None/Free/unattached")
    if final["place_mode"] not in {"recorded_place", "reversed_pickup"}:
        raise _error("place mode must be recorded_place or reversed_pickup")
    if final["place_source"] == "none" or final["place_source_id"] == 0:
        raise _error("final placement source identity is missing")
    recomputed_position_error = math.dist(
        final["object_position"], final["place_goal_position"]
    )
    if not math.isclose(
        final["place_position_error_m"],
        recomputed_position_error,
        rel_tol=0.0,
        abs_tol=FINAL_POSITION_RECOMPUTATION_TOLERANCE_M,
    ):
        raise _error("final place position error does not match object and goal")
    recomputed_orientation_error = pickup._joint_rotation_step_degrees(
        final["object_world_rotation"], final["place_goal_rotation"]
    )
    if not math.isclose(
        final["place_orientation_error_degrees"],
        recomputed_orientation_error,
        abs_tol=2.0e-5,
    ):
        raise _error("final place orientation error does not match object and goal")
    if recomputed_position_error > FINAL_POSITION_ERROR_LIMIT_M:
        raise _error("final position error exceeds 0.02 m")
    if recomputed_orientation_error > FINAL_ORIENTATION_ERROR_LIMIT_DEGREES:
        raise _error("final orientation error exceeds 10 degrees")
    if not final["requested_fit_accepted"] or not final["requested_footprint_valid"] or not final["requested_overhead_valid"]:
        raise _error("requested placement fit must be accepted with footprint and overhead")
    if not final["actual_fit_accepted"]:
        raise _error("actual placement fit must be accepted")
    if not ACTUAL_SUPPORT_GAP_MIN_M <= final["actual_support_gap_m"] <= ACTUAL_SUPPORT_GAP_MAX_M:
        raise _error("actual support gap is outside [-0.005, +0.020] m")
    if not all(
        final[name]
        for name in (
            "actual_footprint_valid",
            "actual_bound_corners_valid",
            "actual_overhead_valid",
            "support_sweep_clear",
            "destination_support_committed",
        )
    ):
        raise _error("actual bounds, overhead, sweep, and destination support must be valid")


def validate_screenshot(path: Path) -> None:
    try:
        pickup.validate_screenshot(path)
    except pickup.EvidenceValidationError as error:
        raise _error(str(error)) from error
    data = Path(path).read_bytes()
    offset = 8
    header = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        offset += 12 + length
        if kind == b"IEND":
            break
    if header is None:
        raise _error("screenshot PNG has no decodable header")
    width, height, depth, color_type, compression, filtering, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if (
        depth != 8
        or channels is None
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise _error("screenshot PNG format is unsupported for visual proof")
    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    if len(raw) != height * (stride + 1):
        raise _error("screenshot PNG scanline size is inconsistent")

    decoded = bytearray(height * stride)

    def paeth(left: int, above: int, upper_left: int) -> int:
        estimate = left + above - upper_left
        left_error = abs(estimate - left)
        above_error = abs(estimate - above)
        upper_left_error = abs(estimate - upper_left)
        if left_error <= above_error and left_error <= upper_left_error:
            return left
        if above_error <= upper_left_error:
            return above
        return upper_left

    source = 0
    for row in range(height):
        filter_kind = raw[source]
        source += 1
        row_begin = row * stride
        previous_begin = (row - 1) * stride
        for column in range(stride):
            encoded = raw[source]
            source += 1
            left = decoded[row_begin + column - channels] if column >= channels else 0
            above = decoded[previous_begin + column] if row else 0
            upper_left = (
                decoded[previous_begin + column - channels]
                if row and column >= channels
                else 0
            )
            if filter_kind == 0:
                value = encoded
            elif filter_kind == 1:
                value = encoded + left
            elif filter_kind == 2:
                value = encoded + above
            elif filter_kind == 3:
                value = encoded + ((left + above) // 2)
            elif filter_kind == 4:
                value = encoded + paeth(left, above, upper_left)
            else:
                raise _error("screenshot PNG uses an invalid row filter")
            decoded[row_begin + column] = value & 0xFF

    first = tuple(decoded[: min(channels, 3)])
    different_pixels = 0
    minimum_channel = 255
    maximum_channel = 0
    for index in range(0, len(decoded), channels):
        pixel = tuple(decoded[index:index + min(channels, 3)])
        if pixel != first:
            different_pixels += 1
        minimum_channel = min(minimum_channel, *pixel)
        maximum_channel = max(maximum_channel, *pixel)
    if different_pixels < 100 or maximum_channel - minimum_channel < 16:
        raise _error("screenshot is visually blank and cannot prove placement")


def _write_nonblank_png(path: Path) -> None:
    header = struct.pack(">IIBBBBB", 1280, 720, 8, 2, 0, 0, 0)
    rows = bytearray()
    for y in range(720):
        rows.append(0)
        for x in range(1280):
            rows.extend((220, 60, 40) if 80 <= x < 180 and 80 <= y < 180 else (0, 0, 0))
    image = zlib.compress(bytes(rows))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + pickup._png_chunk(b"IHDR", header)
        + pickup._png_chunk(b"tEXt", b"placement-evidence\x00" + b"x" * 10_500)
        + pickup._png_chunk(b"IDAT", image)
        + pickup._png_chunk(b"IEND", b"")
    )
    Path(path).write_bytes(data)


def _identity_quaternion():
    return [1.0, 0.0, 0.0, 0.0]


def _base_record(root, object_position) -> dict:
    joint_positions = [
        [root[0], 0.05 * joint, root[2]]
        for joint in range(len(pickup.FLAT_JOINT_NAMES))
    ]
    joint_positions[22] = copy.deepcopy(object_position)
    record = {
        "render_frame": 0,
        "runtime_tick": 1,
        "scheduler_phase": 0,
        "state": "Locomotion",
        "result": "None",
        "reason": "None",
        "object_state": "Free",
        "attached": False,
        "owns_pose": False,
        "action": "none",
        "demo_phase": "pickup",
        "phase_counter": 0,
        "pickup_distance_m": math.hypot(root[0] - object_position[0], root[2] - object_position[2]),
        "walking_origin_displacement_m": 0.0,
        "displayed_root_speed_mps": 0.0,
        "left_stick_command": [0.0, 0.0, 0.0],
        "left_stick_magnitude": 0.0,
        "root_position": copy.deepcopy(root),
        "object_position": copy.deepcopy(object_position),
        "object_world_rotation": _identity_quaternion(),
        "locomotion_provider_kind": "live_flat",
        "canonical_snapshot_used": False,
        "root_relocation_applied": False,
        "simulation_root_initialized_from_reach": False,
        "displayed_root_initialized_from_reach": False,
        "reach_waypoint_position": [0.0, 0.0, 2.60],
        "reach_waypoint_rotation": _identity_quaternion(),
        "interaction_waypoint_position": [
            0.0904093558401867, 0.0, 2.61035125
        ],
        "interaction_waypoint_rotation": _identity_quaternion(),
        "interaction_lateral_offset_m": 0.091,
        "interaction_object_position": [0.0, 1.10, 3.0],
        "interaction_object_dimensions": [0.10, 0.20, 0.14],
        "interaction_object_rotation": _identity_quaternion(),
        "interaction_approach_direction_object": [0.0, 0.0, 1.0],
        "interaction_clearance_radius_m": 0.04,
        "interaction_hand_side": 1,
        "reach_waypoint_position_error_m": math.hypot(
            root[0], root[2] - 2.60
        ),
        "reach_waypoint_yaw_error_degrees": 0.0,
        "reach_brake_latched": False,
        "stationary_candidate_count": STATIONARY_CANDIDATE_COUNT,
        "stationary_constraint_active": False,
        "stationary_search_calls": 0,
        "stationary_transition_count": 0,
        "stationary_selected_frame": -1,
        "stationary_selected_range": -1,
        "brake_simulation_speed_mps": 0.0,
        "pick_resolver_maximum_m": 1.0,
        "pick_resolver_live_flat": False,
        "destination_handle_id": 2,
        "destination_generation": 1,
        "destination_affordance_id": 1,
        "destination_generation_changed_before_place": False,
        "placement_surface_resolver_calls": 0,
        "runtime_preview_calls": 0,
        "runtime_preview_source_state": "none",
        "free_selector_calls": 0,
        "external_match_input_calls": 0,
        "preview_mutation_count": 0,
        "far_selection_id": 0,
        "current_selection_id": 0,
        "submitted_selection_id": 0,
        "candidate_certified": False,
        "place_mode": "none",
        "place_source": "none",
        "place_source_id": 0,
        "preview_accepted": False,
        "preview_ready": False,
        "preview_ik_fingerprint": 0,
        **EXPECTED_IK,
        "staging_root_position": [0.0, 0.0, 0.0],
        "staging_root_rotation": _identity_quaternion(),
        "preview_root_error_m": 0.0,
        "preview_yaw_error_degrees": 0.0,
        "carry_staging_tick": -1,
        "selected_staging_root": False,
        "direct_root_write": False,
        "place_phase": "none",
        "place_goal_position": [0.0, 0.75, 4.20],
        "place_goal_rotation": _identity_quaternion(),
        "place_position_error_m": 0.0,
        "place_orientation_error_degrees": 0.0,
        "requested_fit_accepted": False,
        "requested_support_gap_m": 0.0,
        "requested_lowest_corner_m": 0.0,
        "requested_highest_corner_m": 0.0,
        "requested_footprint_valid": False,
        "requested_overhead_valid": False,
        "actual_fit_accepted": False,
        "actual_support_gap_m": 0.0,
        "actual_lowest_corner_m": 0.0,
        "actual_highest_corner_m": 0.0,
        "actual_footprint_valid": False,
        "actual_bound_corners_valid": False,
        "actual_overhead_valid": False,
        "support_sweep_clear": False,
        "attachment_transition_count": 0,
        "attached_to_free_transition_count": 0,
        "destination_support_committed": False,
        "grasp_evidence_valid": False,
        "active_hand_joint": -1,
        "hand_constraint_weight": 0.0,
        "hand_constraint_validated": False,
        "hand_constraint_applied": False,
        "hand_constraint_reachable": False,
        "hand_constraint_used_clavicle": False,
        "hand_constraint_reach_shortfall_m": 0.0,
        "hand_constraint_calibration_rotation": _identity_quaternion(),
        "calibrated_hand_world_rotation": _identity_quaternion(),
        "hand_in_object_position": [0.0, 0.0, 0.0],
        "hand_in_object_rotation": _identity_quaternion(),
        "grasp_world_position": [0.0, 0.0, 0.0],
        "grasp_world_rotation": _identity_quaternion(),
        "joint_world_positions": joint_positions,
        "joint_world_rotations": [
            _identity_quaternion() for _ in pickup.FLAT_JOINT_NAMES
        ],
        "pick_preview_epoch": -1,
        "pick_preview_consumed_epoch": -1,
        "pick_preview_snapshot_fingerprint": 0,
        "pick_preview_snapshot_source": "unset",
        "pick_preview_epoch_count": 0,
        "runtime_pick_preview_calls": 0,
        "pick_preview_mutation_count": 0,
        "pick_preview_elapsed_ticks": 0,
        "pick_preview_pending_failure": "none",
        "pick_preview_failure_recorded": False,
        "pick_entry_selected_slot": "none",
        "pick_entry_selection_frozen": False,
        "pick_entry_freeze_tick": -1,
        "pick_interact_submission_count": 0,
        "pick_reservation_transition_count": 0,
        "pick_plus_identity": "Plus",
        "pick_plus_evaluation_order": 0,
        "pick_plus_waypoint_position": [0.0, 0.0, 0.0],
        "pick_plus_waypoint_rotation": _identity_quaternion(),
        "pick_plus_prospective_world_x": 0.0,
        "pick_plus_prospective_world_z": 0.0,
        "pick_plus_prospective_world_yaw_radians": 0.0,
        "pick_plus_clearance_chord_m": 0.0,
        "pick_plus_preserved_standoff_m": 0.0,
        "pick_plus_hand_score": 0.0,
        "pick_plus_path_feasible": False,
        "pick_plus_match_ready": False,
        "pick_plus_path_reason": "NoCandidate",
        "pick_plus_match_reason": "NoCandidate",
        "pick_plus_feasible_entry_frame": -1,
        "pick_plus_contact_frame": -1,
        "pick_plus_cost_available": False,
        "pick_plus_total_cost": 0.0,
        "pick_plus_eligible": False,
        "pick_plus_selected": False,
        "pick_minus_identity": "Minus",
        "pick_minus_evaluation_order": 1,
        "pick_minus_waypoint_position": [0.0, 0.0, 0.0],
        "pick_minus_waypoint_rotation": _identity_quaternion(),
        "pick_minus_prospective_world_x": 0.0,
        "pick_minus_prospective_world_z": 0.0,
        "pick_minus_prospective_world_yaw_radians": 0.0,
        "pick_minus_clearance_chord_m": 0.0,
        "pick_minus_preserved_standoff_m": 0.0,
        "pick_minus_hand_score": 0.0,
        "pick_minus_path_feasible": False,
        "pick_minus_match_ready": False,
        "pick_minus_path_reason": "NoCandidate",
        "pick_minus_match_reason": "NoCandidate",
        "pick_minus_feasible_entry_frame": -1,
        "pick_minus_contact_frame": -1,
        "pick_minus_cost_available": False,
        "pick_minus_total_cost": 0.0,
        "pick_minus_eligible": False,
        "pick_minus_selected": False,
    }
    for prefix, expected in _expected_pick_entry_slots(record).items():
        for suffix, value in expected.items():
            record[f"{prefix}_{suffix}"] = copy.deepcopy(value)
    return record


def _pick_outcome(
    *,
    path_feasible: bool,
    match_ready: bool,
    path_reason: str,
    match_reason: str,
    entry_frame: int,
    contact_frame: int,
    cost_available: bool,
    total_cost: float,
) -> dict:
    return {
        "path_feasible": path_feasible,
        "match_ready": match_ready,
        "path_reason": path_reason,
        "match_reason": match_reason,
        "feasible_entry_frame": entry_frame,
        "contact_frame": contact_frame,
        "cost_available": cost_available,
        "total_cost": total_cost,
        "eligible": path_feasible and match_ready,
    }


PICK_PLUS_POOR = _pick_outcome(
    path_feasible=True,
    match_ready=False,
    path_reason="None",
    match_reason="PoorMatch",
    entry_frame=114,
    contact_frame=139,
    cost_available=True,
    total_cost=10.25,
)
PICK_PLUS_READY = _pick_outcome(
    path_feasible=True,
    match_ready=True,
    path_reason="None",
    match_reason="None",
    entry_frame=114,
    contact_frame=139,
    cost_available=True,
    total_cost=1.25,
)
PICK_MINUS_BLOCKED = _pick_outcome(
    path_feasible=False,
    match_ready=False,
    path_reason="BlockedPath",
    match_reason="BlockedPath",
    entry_frame=-1,
    contact_frame=-1,
    cost_available=False,
    total_cost=0.0,
)


def _publish_pick_epoch(
    row: dict,
    *,
    epoch: int,
    consumed_epoch: int,
    fingerprint: int,
    plus: dict,
    minus: dict,
) -> None:
    row.update(
        pick_preview_epoch=epoch,
        pick_preview_consumed_epoch=consumed_epoch,
        pick_preview_snapshot_fingerprint=fingerprint,
        pick_preview_snapshot_source="live_flat",
        pick_preview_epoch_count=epoch + 1,
        runtime_pick_preview_calls=2 * (epoch + 1),
        pick_preview_mutation_count=0,
        pick_preview_elapsed_ticks=epoch + 1,
    )
    for prefix, outcome in (("pick_plus", plus), ("pick_minus", minus)):
        for suffix, value in outcome.items():
            row[f"{prefix}_{suffix}"] = copy.deepcopy(value)


def _migrate_pick_entry_success(records: list[dict]) -> None:
    stages = {
        row["_pick_preview_stage"]: index
        for index, row in enumerate(records)
        if "_pick_preview_stage" in row
    }
    if set(stages) != {"poor", "ready", "navigate", "brake"}:
        raise AssertionError("success fixture requires the complete preview prelude")
    poor_index = stages["poor"]
    ready_index = stages["ready"]
    freeze_index = stages["navigate"]
    if (ready_index, freeze_index, stages["brake"]) != (
        poor_index + 1,
        poor_index + 2,
        poor_index + 3,
    ):
        raise AssertionError("preview prelude must be contiguous")
    _publish_pick_epoch(
        records[poor_index],
        epoch=0,
        consumed_epoch=-1,
        fingerprint=0xA11CE,
        plus=PICK_PLUS_POOR,
        minus=PICK_MINUS_BLOCKED,
    )
    _publish_pick_epoch(
        records[ready_index],
        epoch=1,
        consumed_epoch=0,
        fingerprint=0xB22DF,
        plus=PICK_PLUS_READY,
        minus=PICK_MINUS_BLOCKED,
    )
    freeze_tick = freeze_index + 1
    for row in records[freeze_index:]:
        _publish_pick_epoch(
            row,
            epoch=1,
            consumed_epoch=1,
            fingerprint=0xB22DF,
            plus=PICK_PLUS_READY,
            minus=PICK_MINUS_BLOCKED,
        )
        row.update(
            pick_entry_selected_slot="Plus",
            pick_entry_selection_frozen=True,
            pick_entry_freeze_tick=freeze_tick,
            pick_plus_selected=True,
            pick_minus_selected=False,
        )
        row["interaction_waypoint_position"] = copy.deepcopy(
            row["pick_plus_waypoint_position"]
        )
        row["interaction_waypoint_rotation"] = copy.deepcopy(
            row["pick_plus_waypoint_rotation"]
        )
    interact_index = next(
        index
        for index, row in enumerate(records)
        if row["action"] == "interact"
    )
    for row in records[interact_index:]:
        row["pick_interact_submission_count"] = 1
        row["pick_resolver_live_flat"] = True
    first_align = next(
        index for index, row in enumerate(records) if row["state"] == "Align"
    )
    for row in records[first_align:]:
        row["pick_reservation_transition_count"] = 1
    for row in records:
        row.pop("_pick_preview_stage", None)


def _valid_pick_entry_failure_records(kind: str) -> list[dict]:
    if kind not in {"no_path", "deadline"}:
        raise AssertionError(kind)
    count = 3 if kind == "no_path" else MAX_WALK_TICKS + 2
    template = _base_record([0.0, 0.0, 0.0], [0.0, 1.10, 3.0])
    template.update(action="none", demo_phase="settle", phase_counter=0)
    records = [copy.deepcopy(template) for _ in range(count)]
    for index, row in enumerate(records[1:], 1):
        row.update(demo_phase="approach", phase_counter=index)
    if kind == "no_path":
        _publish_pick_epoch(
            records[1],
            epoch=0,
            consumed_epoch=-1,
            fingerprint=0xC3301,
            plus=PICK_MINUS_BLOCKED,
            minus=PICK_MINUS_BLOCKED,
        )
        _publish_pick_epoch(
            records[2],
            epoch=0,
            consumed_epoch=0,
            fingerprint=0xC3301,
            plus=PICK_MINUS_BLOCKED,
            minus=PICK_MINUS_BLOCKED,
        )
        records[2].update(
            pick_preview_pending_failure="no_path",
            pick_preview_failure_recorded=True,
        )
    else:
        for index in range(1, MAX_WALK_TICKS + 1):
            _publish_pick_epoch(
                records[index],
                epoch=index - 1,
                consumed_epoch=index - 2,
                fingerprint=0xD0000 + index,
                plus=PICK_PLUS_POOR,
                minus=PICK_MINUS_BLOCKED,
            )
        _publish_pick_epoch(
            records[-1],
            epoch=MAX_WALK_TICKS - 1,
            consumed_epoch=MAX_WALK_TICKS - 1,
            fingerprint=0xD0000 + MAX_WALK_TICKS,
            plus=PICK_PLUS_POOR,
            minus=PICK_MINUS_BLOCKED,
        )
        records[-1].update(
            pick_preview_pending_failure="deadline",
            pick_preview_failure_recorded=True,
        )
    _reclock_and_measure(records)
    return records


def _valid_records() -> list[dict]:
    records = []
    object_position = [0.0, 1.10, 3.0]
    baseline = _base_record([0.0, 0.0, 0.0], object_position)
    baseline.update(
        action="none",
        demo_phase="settle",
        phase_counter=0,
        left_stick_command=[0.0, 0.0, 0.0],
    )
    records.append(baseline)
    approach_ticks = 106
    for tick in range(approach_ticks):
        z = 2.60 * tick / (approach_ticks - 1)
        row = _base_record([0.0, 0.0, z], object_position)
        row.update(
            action="approach",
            demo_phase="approach",
            phase_counter=tick + 1,
            left_stick_command=[0.0, 0.0, 1.0],
        )
        records.append(row)

    for stage in ("poor", "ready", "navigate", "brake"):
        row = _base_record([0.0, 0.0, 2.60], object_position)
        row.update(demo_phase="settle", phase_counter=0)
        row["_pick_preview_stage"] = stage
        if stage == "navigate":
            row.update(
                action="approach",
                demo_phase="approach",
                left_stick_command=[1.0, 0.0, 0.0],
            )
        records.append(row)

    for tick in range(SETTLE_TICKS):
        row = _base_record([0.0, 0.0, 2.60], object_position)
        row.update(demo_phase="settle", phase_counter=tick + 1)
        records.append(row)

    def append_states(states):
        for state, count in states:
            for _ in range(count):
                row = _base_record([0.0, 0.0, 2.60], object_position)
                row["state"] = state
                row["demo_phase"] = "interact" if not records[-1].get("_interacted", False) else "pickup"
                if state == "Align":
                    row.update(result="Accepted", object_state="Targeted", owns_pose=True)
                elif state == "PickupReplay":
                    row.update(result="Accepted", object_state="Attached", attached=True, owns_pose=True)
                elif state == "Hold":
                    row.update(result="Accepted", object_state="Held", attached=True, owns_pose=True)
                if row["attached"]:
                    _enable_grasp(row)
                records.append(row)
                records[-1]["_interacted"] = True

    interact_index = len(records)
    append_states((("Preflight", 1), ("Align", 3), ("PickupReplay", 5), ("Hold", 5)))
    records[interact_index]["action"] = "interact"
    records[interact_index]["demo_phase"] = "interact"

    prelude = _base_record([0.0, 0.0, 2.60], object_position)
    prelude.update(
        state="Carry",
        result="Succeeded",
        object_state="Held",
        attached=True,
        owns_pose=True,
        demo_phase="pickup",
    )
    _enable_grasp(prelude)
    records.append(prelude)

    far_selection = 1000
    far = _base_record([0.0, 0.0, 2.60], object_position)
    far.update(
        state="Carry",
        result="Succeeded",
        object_state="Held",
        attached=True,
        owns_pose=True,
        action="latch_destination",
        demo_phase="carry_preview",
        phase_counter=1,
        runtime_preview_calls=1,
        runtime_preview_source_state="Carry",
        far_selection_id=far_selection,
        current_selection_id=far_selection,
        candidate_certified=True,
        place_mode="reversed_pickup",
        place_source="pickup_clip_0",
        place_source_id=1,
        preview_accepted=True,
        preview_ready=False,
        preview_ik_fingerprint=EXPECTED_IK_FINGERPRINT,
        staging_root_position=[0.0, 0.0, 3.50],
        preview_root_error_m=0.90,
        preview_yaw_error_degrees=30.0,
        selected_staging_root=True,
        requested_fit_accepted=True,
        requested_footprint_valid=True,
        requested_overhead_valid=True,
        requested_highest_corner_m=0.20,
    )
    _enable_grasp(far)
    records.append(far)

    carry_count = 30
    for tick in range(carry_count):
        z = 2.60 + 0.03 * (tick + 1)
        carried_object = [0.0, 1.10, 3.0 + 0.03 * (tick + 1)]
        row = _base_record([0.0, 0.0, z], carried_object)
        current_selection = far_selection + tick + 1
        error = max(0.26, 0.90 - 0.020 * tick)
        row.update(
            state="Carry",
            result="Succeeded",
            object_state="Held",
            attached=True,
            owns_pose=True,
            action="stage",
            demo_phase="carry_staging",
            phase_counter=tick + 1,
            left_stick_command=[0.0, 0.0, 0.75],
            runtime_preview_calls=tick + 2,
            runtime_preview_source_state="Carry",
            far_selection_id=far_selection,
            current_selection_id=current_selection,
            candidate_certified=True,
            place_mode="reversed_pickup",
            place_source="pickup_clip_0",
            place_source_id=1,
            preview_accepted=True,
            preview_ready=False,
            preview_ik_fingerprint=EXPECTED_IK_FINGERPRINT,
            staging_root_position=[0.0, 0.0, 3.50],
            preview_root_error_m=error,
            preview_yaw_error_degrees=30.0,
            carry_staging_tick=tick,
            selected_staging_root=True,
            requested_fit_accepted=True,
            requested_footprint_valid=True,
            requested_overhead_valid=True,
            requested_highest_corner_m=0.20,
        )
        _enable_grasp(row)
        records.append(row)

    last_staging = records[-1]
    ready_selection = last_staging["current_selection_id"] + 1
    place_root = copy.deepcopy(last_staging["root_position"])
    place_object = copy.deepcopy(last_staging["object_position"])
    for state, count in (
        ("PlacePreflight", 1),
        ("PlaceAlign", 3),
        ("PlaceReplay", 20),
    ):
        for tick in range(count):
            progress = 0.0 if state != "PlaceReplay" else (tick + 1) / count
            object_now = [
                place_object[0],
                1.10 + progress * (0.75 - 1.10),
                place_object[2] + progress * (4.20 - place_object[2]),
            ]
            row = _base_record(place_root, object_now)
            row.update(
                state=state,
                result="Accepted",
                object_state="Held",
                attached=True,
                owns_pose=True,
                action="place" if state == "PlacePreflight" else "none",
                demo_phase="place",
                phase_counter=tick + 1,
                runtime_preview_calls=carry_count + 2,
                runtime_preview_source_state="Carry" if state == "PlacePreflight" else "none",
                far_selection_id=far_selection,
                current_selection_id=ready_selection,
                submitted_selection_id=ready_selection if state == "PlacePreflight" else 0,
                candidate_certified=True,
                place_mode="reversed_pickup",
                place_source="pickup_clip_0",
                place_source_id=1,
                preview_accepted=True,
                preview_ready=True,
                preview_ik_fingerprint=EXPECTED_IK_FINGERPRINT,
                staging_root_position=[0.0, 0.0, 3.50],
                preview_root_error_m=0.20,
                preview_yaw_error_degrees=5.0,
                selected_staging_root=True,
                place_phase="align" if state != "PlaceReplay" else ("lower" if tick < count - 1 else "release"),
                requested_fit_accepted=True,
                requested_footprint_valid=True,
                requested_overhead_valid=True,
                requested_highest_corner_m=0.20,
            )
            _enable_grasp(row)
            records.append(row)

    for tick in range(5):
        row = _base_record(place_root, [0.0, 0.75, 4.20])
        row.update(
            state="PlaceRelease",
            result="Succeeded",
            object_state="Free",
            owns_pose=True,
            demo_phase="place",
            phase_counter=tick + 1,
            runtime_preview_calls=carry_count + 2,
            far_selection_id=far_selection,
            current_selection_id=ready_selection,
            candidate_certified=True,
            place_mode="reversed_pickup",
            place_source="pickup_clip_0",
            place_source_id=1,
            preview_accepted=True,
            preview_ready=True,
            preview_ik_fingerprint=EXPECTED_IK_FINGERPRINT,
            staging_root_position=[0.0, 0.0, 3.50],
            preview_root_error_m=0.20,
            preview_yaw_error_degrees=5.0,
            selected_staging_root=True,
            place_phase="retract" if tick < 4 else "finished",
            requested_fit_accepted=True,
            requested_footprint_valid=True,
            requested_overhead_valid=True,
            requested_highest_corner_m=0.20,
            actual_fit_accepted=True,
            actual_highest_corner_m=0.20,
            actual_footprint_valid=True,
            actual_bound_corners_valid=True,
            actual_overhead_valid=True,
            support_sweep_clear=True,
            destination_support_committed=True,
        )
        records.append(row)
    final_release = copy.deepcopy(records[-1])
    for tick in range(MIN_PLACEMENT_HANDOFF_FRAMES):
        final = copy.deepcopy(final_release)
        final.update(
            state="Locomotion",
            owns_pose=False,
            demo_phase="handoff",
            phase_counter=tick + 1,
            place_phase="finished",
        )
        records.append(final)

    brake_index = next(
        index
        for index, row in enumerate(records)
        if row.get("_pick_preview_stage") == "brake"
    )
    for index, row in enumerate(records):
        row.pop("_interacted", None)
        row["reach_brake_latched"] = index >= brake_index
        row["stationary_candidate_count"] = STATIONARY_CANDIDATE_COUNT
        row["brake_simulation_speed_mps"] = (
            0.04 if index >= brake_index else 0.0
        )
        if index < brake_index:
            row.update(
                stationary_constraint_active=False,
                stationary_search_calls=0,
                stationary_transition_count=0,
                stationary_selected_frame=-1,
                stationary_selected_range=-1,
            )
        else:
            constrained = index <= interact_index
            calls = 1 + min(1, (index - brake_index) // 3)
            selected_range = 0 if calls == 1 else 1
            selected_frame = 0 if calls == 1 else 118
            row.update(
                stationary_constraint_active=constrained,
                stationary_search_calls=calls,
                stationary_transition_count=calls,
                stationary_selected_frame=selected_frame,
                stationary_selected_range=selected_range,
            )
    _migrate_pick_entry_success(records)
    _reclock_and_measure(records)
    return records


def _valid_minus_selected_records() -> list[dict]:
    records = _valid_records()
    for row in records:
        if row["pick_preview_epoch"] == 0:
            _publish_pick_epoch(
                row,
                epoch=0,
                consumed_epoch=row["pick_preview_consumed_epoch"],
                fingerprint=row["pick_preview_snapshot_fingerprint"],
                plus=PICK_MINUS_BLOCKED,
                minus=PICK_PLUS_POOR,
            )
        elif row["pick_preview_epoch"] == 1:
            _publish_pick_epoch(
                row,
                epoch=1,
                consumed_epoch=row["pick_preview_consumed_epoch"],
                fingerprint=row["pick_preview_snapshot_fingerprint"],
                plus=PICK_MINUS_BLOCKED,
                minus=PICK_PLUS_READY,
            )
        if row["pick_entry_selection_frozen"]:
            row.update(
                pick_entry_selected_slot="Minus",
                pick_plus_selected=False,
                pick_minus_selected=True,
            )
            row["interaction_waypoint_position"] = copy.deepcopy(
                row["pick_minus_waypoint_position"]
            )
            row["interaction_waypoint_rotation"] = copy.deepcopy(
                row["pick_minus_waypoint_rotation"]
            )
    return records


def _valid_ready_at_bound_records() -> list[dict]:
    records = _valid_records()
    poor_index = next(
        index for index, row in enumerate(records)
        if row["pick_preview_epoch"] == 0
    )
    ready_index = next(
        index for index, row in enumerate(records)
        if row["pick_preview_epoch"] == 1
        and not row["pick_entry_selection_frozen"]
    )
    template = records[poor_index]
    extras = []
    for epoch in range(1, MAX_WALK_TICKS - 1):
        row = copy.deepcopy(template)
        _publish_pick_epoch(
            row,
            epoch=epoch,
            consumed_epoch=epoch - 1,
            fingerprint=0xE0000 + epoch,
            plus=PICK_PLUS_POOR,
            minus=PICK_MINUS_BLOCKED,
        )
        extras.append(row)
    records[ready_index:ready_index] = extras
    ready_index += len(extras)
    _publish_pick_epoch(
        records[ready_index],
        epoch=MAX_WALK_TICKS - 1,
        consumed_epoch=MAX_WALK_TICKS - 2,
        fingerprint=0xE0000 + MAX_WALK_TICKS,
        plus=PICK_PLUS_READY,
        minus=PICK_MINUS_BLOCKED,
    )
    freeze_index = ready_index + 1
    freeze_tick = freeze_index + 1
    for row in records[freeze_index:]:
        _publish_pick_epoch(
            row,
            epoch=MAX_WALK_TICKS - 1,
            consumed_epoch=MAX_WALK_TICKS - 1,
            fingerprint=0xE0000 + MAX_WALK_TICKS,
            plus=PICK_PLUS_READY,
            minus=PICK_MINUS_BLOCKED,
        )
        row["pick_entry_freeze_tick"] = freeze_tick
    _reclock_and_measure(records)
    return records


def _valid_tied_pick_entry_records() -> list[dict]:
    records = _valid_records()
    for row in records:
        row["interaction_approach_direction_object"] = [1.0, 0.0, 0.0]
        expected_slots = _expected_pick_entry_slots(row)
        for prefix, expected in expected_slots.items():
            for suffix, value in expected.items():
                row[f"{prefix}_{suffix}"] = copy.deepcopy(value)
        row["interaction_lateral_offset_m"] = expected_slots["pick_plus"][
            "clearance_chord_m"
        ]
        if row["pick_preview_epoch"] == 0:
            _publish_pick_epoch(
                row,
                epoch=0,
                consumed_epoch=row["pick_preview_consumed_epoch"],
                fingerprint=row["pick_preview_snapshot_fingerprint"],
                plus=PICK_PLUS_POOR,
                minus=PICK_PLUS_POOR,
            )
        elif row["pick_preview_epoch"] == 1:
            _publish_pick_epoch(
                row,
                epoch=1,
                consumed_epoch=row["pick_preview_consumed_epoch"],
                fingerprint=row["pick_preview_snapshot_fingerprint"],
                plus=PICK_PLUS_READY,
                minus=PICK_PLUS_READY,
            )
        if row["pick_entry_selection_frozen"]:
            row["interaction_waypoint_position"] = copy.deepcopy(
                row["pick_plus_waypoint_position"]
            )
            row["interaction_waypoint_rotation"] = copy.deepcopy(
                row["pick_plus_waypoint_rotation"]
            )
    return records


def _refresh_pick_entry_geometry(records: list[dict]) -> None:
    for row in records:
        expected_slots = _expected_pick_entry_slots(row)
        for prefix, expected in expected_slots.items():
            for suffix, value in expected.items():
                row[f"{prefix}_{suffix}"] = copy.deepcopy(value)
        row["interaction_lateral_offset_m"] = expected_slots["pick_plus"][
            "clearance_chord_m"
        ]
        if row["pick_entry_selection_frozen"]:
            prefix = {
                "Plus": "pick_plus", "Minus": "pick_minus"
            }[row["pick_entry_selected_slot"]]
            row["interaction_waypoint_position"] = copy.deepcopy(
                row[f"{prefix}_waypoint_position"]
            )
            row["interaction_waypoint_rotation"] = copy.deepcopy(
                row[f"{prefix}_waypoint_rotation"]
            )


def _enable_grasp(row: dict) -> None:
    row.update(
        grasp_evidence_valid=True,
        active_hand_joint=22,
        hand_constraint_weight=1.0,
        hand_constraint_validated=True,
        hand_constraint_applied=True,
        hand_constraint_reachable=True,
        grasp_world_position=copy.deepcopy(row["object_position"]),
        grasp_world_rotation=_identity_quaternion(),
    )
    row["joint_world_positions"][22] = copy.deepcopy(row["object_position"])


def _reclock_and_measure(records: list[dict]) -> None:
    origin = records[0]["root_position"]
    attachment_transitions = 0
    release_transitions = 0
    for index, row in enumerate(records):
        row["render_frame"] = index
        row["runtime_tick"] = index + 1
        row["scheduler_phase"] = 0
        row["pickup_distance_m"] = math.hypot(
            row["root_position"][0] - row["object_position"][0],
            row["root_position"][2] - row["object_position"][2],
        )
        row["walking_origin_displacement_m"] = math.hypot(
            row["root_position"][0] - origin[0],
            row["root_position"][2] - origin[2],
        )
        row["reach_waypoint_position_error_m"] = math.hypot(
            row["root_position"][0] - row["reach_waypoint_position"][0],
            row["root_position"][2] - row["reach_waypoint_position"][2],
        )
        row["reach_waypoint_yaw_error_degrees"] = _yaw_error_degrees(
            row["joint_world_rotations"][0], row["reach_waypoint_rotation"]
        )
        row["left_stick_magnitude"] = math.sqrt(
            sum(component * component for component in row["left_stick_command"])
        )
        row["displayed_root_speed_mps"] = 0.0 if index == 0 else CONTROL_RATE_HZ * math.hypot(
            row["root_position"][0] - records[index - 1]["root_position"][0],
            row["root_position"][2] - records[index - 1]["root_position"][2],
        )
        if index and row["attached"] != records[index - 1]["attached"]:
            attachment_transitions += 1
            if records[index - 1]["attached"] and not row["attached"]:
                release_transitions += 1
        row["attachment_transition_count"] = attachment_transitions
        row["attached_to_free_transition_count"] = release_transitions


class PlacementEvidenceValidatorUnitTests(unittest.TestCase):
    def test_valid_full_synthetic_sequence_passes(self):
        validate_evidence(_valid_records())

    def test_minimum_placement_handoff_is_eighteen_numbered_no_input_rows(self):
        records = _valid_records()
        validate_evidence(records)

        handoff = [row for row in records if row["demo_phase"] == "handoff"]
        self.assertEqual(len(handoff), MIN_PLACEMENT_HANDOFF_FRAMES)
        self.assertEqual(
            [row["phase_counter"] for row in handoff],
            list(range(1, MIN_PLACEMENT_HANDOFF_FRAMES + 1)),
        )
        for row in handoff:
            self.assertEqual(row["state"], "Locomotion")
            self.assertEqual(row["action"], "none")
            self.assertEqual(row["left_stick_command"], [0.0, 0.0, 0.0])
        self.assertEqual(
            [row["phase_counter"] for row in handoff[-PLACEMENT_SETTLED_FRAMES:]],
            [16, 17, 18],
        )

    def test_behavior_gated_handoff_may_extend_while_remaining_input_free(self):
        records = _valid_records()
        final_handoff = copy.deepcopy(records[-1])
        for phase_counter in range(MIN_PLACEMENT_HANDOFF_FRAMES + 1, 23):
            row = copy.deepcopy(final_handoff)
            row["phase_counter"] = phase_counter
            records.append(row)
        _reclock_and_measure(records)

        validate_evidence(records)

        handoff = [row for row in records if row["demo_phase"] == "handoff"]
        self.assertEqual(len(handoff), 22)
        self.assertTrue(all(row["action"] == "none" for row in handoff))
        self.assertTrue(
            all(row["left_stick_command"] == [0.0, 0.0, 0.0] for row in handoff)
        )

    def test_handoff_length_accepts_seventy_five_and_rejects_seventy_six(self):
        records = _valid_records()
        final_handoff = copy.deepcopy(records[-1])
        for phase_counter in range(
            MIN_PLACEMENT_HANDOFF_FRAMES + 1,
            MAX_PLACEMENT_HANDOFF_FRAMES + 1,
        ):
            row = copy.deepcopy(final_handoff)
            row["phase_counter"] = phase_counter
            records.append(row)
        _reclock_and_measure(records)
        validate_evidence(records)

        too_long = copy.deepcopy(records)
        row = copy.deepcopy(too_long[-1])
        row["phase_counter"] = MAX_PLACEMENT_HANDOFF_FRAMES + 1
        too_long.append(row)
        _reclock_and_measure(too_long)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "eighteen to seventy-five numbered no-input Locomotion handoff",
        ):
            validate_evidence(too_long)

    def test_fixed_six_decimal_yaw_recomputation_tolerance_is_bounded(self):
        records = _valid_records()
        reach_rotation = [0.997487, 0.0, 0.070853, 0.0]
        for row in records:
            row["reach_waypoint_rotation"] = copy.deepcopy(reach_rotation)
        _refresh_pick_entry_geometry(records)
        row = records[10]
        row["joint_world_rotations"][0] = [
            -0.996103, -0.0, -0.088201, -0.0
        ]
        _reclock_and_measure(records)
        row["reach_waypoint_yaw_error_degrees"] = 1.994191
        mismatch = abs(
            row["reach_waypoint_yaw_error_degrees"]
            - _yaw_error_degrees(
                row["joint_world_rotations"][0],
                row["reach_waypoint_rotation"],
            )
        )
        self.assertGreater(mismatch, 2.0e-5)
        self.assertLessEqual(mismatch, 2.0e-4)
        validate_evidence(records)

        corrupt = copy.deepcopy(records)
        corrupt[10]["reach_waypoint_yaw_error_degrees"] += 5.0e-4
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "waypoint yaw error does not match",
        ):
            validate_evidence(corrupt)

    def test_authoritative_row_zero_baseline_and_initial_approach_are_exact(self):
        records = _valid_records()
        validate_evidence(records)
        baseline = records[0]
        self.assertEqual(
            (
                baseline["runtime_tick"],
                baseline["state"],
                baseline["result"],
                baseline["reason"],
                baseline["object_state"],
                baseline["action"],
                baseline["demo_phase"],
                baseline["phase_counter"],
            ),
            (1, "Locomotion", "None", "None", "Free", "none", "settle", 0),
        )
        self.assertEqual(records[1]["phase_counter"], 1)
        self.assertTrue(all(
            row["action"] == "approach"
            and row["demo_phase"] == "approach"
            and row["phase_counter"] == index
            and row["left_stick_magnitude"] > 1.0e-4
            for index, row in enumerate(
                records[1 : MIN_WALK_TICKS + 1], 1
            )
        ))

        missing = copy.deepcopy(records[1:])
        _reclock_and_measure(missing)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "row 0 baseline"
        ):
            validate_evidence(missing)

        corrupt = copy.deepcopy(records)
        corrupt[0]["stationary_search_calls"] = 1
        corrupt[0]["stationary_selected_frame"] = 0
        corrupt[0]["stationary_selected_range"] = 0
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "row 0 baseline"
        ):
            validate_evidence(corrupt)

        extra = copy.deepcopy(records)
        extra.insert(1, copy.deepcopy(extra[0]))
        for row in extra[2:]:
            if row["pick_entry_selection_frozen"]:
                row["pick_entry_freeze_tick"] += 1
        _reclock_and_measure(extra)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "initial approach|25 consecutive",
        ):
            validate_evidence(extra)

        short = copy.deepcopy(records)
        short[MIN_WALK_TICKS]["demo_phase"] = "settle"
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "25 consecutive"
        ):
            validate_evidence(short)

    def test_interact_submission_tick_is_pending_then_immediate_owned_align(self):
        records = _valid_records()
        validate_evidence(records)
        interact_index = next(
            index
            for index, row in enumerate(records)
            if row["action"] == "interact"
        )
        pending = records[interact_index]
        align = records[interact_index + 1]
        self.assertEqual(
            (
                pending["state"], pending["result"], pending["reason"],
                pending["object_state"], pending["attached"],
                pending["owns_pose"],
                pending["pick_reservation_transition_count"],
            ),
            ("Preflight", "None", "None", "Free", False, False, 0),
        )
        self.assertEqual(
            (
                align["state"], align["result"], align["reason"],
                align["object_state"], align["attached"],
                align["owns_pose"],
                align["pick_reservation_transition_count"],
            ),
            ("Align", "Accepted", "None", "Targeted", False, True, 1),
        )

        stale = _valid_records()
        pending = next(row for row in stale if row["action"] == "interact")
        pending.update(result="Accepted", object_state="Targeted")
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "pending.*Preflight"
        ):
            validate_evidence(stale)

        duplicate = _valid_records()
        index = next(
            i for i, row in enumerate(duplicate)
            if row["action"] == "interact"
        )
        duplicate_row = copy.deepcopy(duplicate[index])
        duplicate_row["action"] = "none"
        duplicate_row["stationary_constraint_active"] = False
        duplicate.insert(index + 1, duplicate_row)
        _reclock_and_measure(duplicate)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "exactly one.*Preflight|immediately next",
        ):
            validate_evidence(duplicate)

        gap = _valid_records()
        index = next(
            i for i, row in enumerate(gap) if row["action"] == "interact"
        )
        gap_row = copy.deepcopy(gap[index])
        gap_row.update(
            state="Locomotion",
            action="none",
            demo_phase="pickup",
            phase_counter=0,
            stationary_constraint_active=False,
        )
        gap.insert(index + 1, gap_row)
        _reclock_and_measure(gap)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "immediately next"
        ):
            validate_evidence(gap)

    def test_ik_display_quantization_keeps_fingerprint_and_raw_fields_strict(self):
        records = _valid_records()
        preview = next(
            row for row in records if row["demo_phase"] == "carry_preview"
        )
        preview["ik_maximum_request_orientation_degrees"] = 24.999998
        validate_evidence(records)

        corrupt = _valid_records()
        preview = next(
            row for row in corrupt if row["demo_phase"] == "carry_preview"
        )
        preview["ik_maximum_request_orientation_degrees"] = 24.999996
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "IK field.*differs"
        ):
            validate_evidence(corrupt)

        wrong_fingerprint = _valid_records()
        preview = next(
            row for row in wrong_fingerprint
            if row["demo_phase"] == "carry_preview"
        )
        preview["preview_ik_fingerprint"] += 1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "fingerprint"
        ):
            validate_evidence(wrong_fingerprint)

        changed_raw = _valid_records()
        preview = next(
            row for row in changed_raw if row["demo_phase"] == "carry_preview"
        )
        preview["ik_damping"] += 1.0e-7
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "IK field.*differs"
        ):
            validate_evidence(changed_raw)

    def test_fixed_six_decimal_speed_and_final_distance_bounds_are_tight(self):
        speed = _valid_records()
        speed[10]["displayed_root_speed_mps"] += 3.8e-5
        validate_evidence(speed)

        corrupt_speed = _valid_records()
        corrupt_speed[10]["displayed_root_speed_mps"] += 5.0e-5
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "displayed_root_speed_mps does not match",
        ):
            validate_evidence(corrupt_speed)

        distance = _valid_records()
        distance[-1]["place_position_error_m"] = 2.4e-6
        validate_evidence(distance)

        corrupt_distance = _valid_records()
        corrupt_distance[-1]["place_position_error_m"] = 5.0e-6
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "position error does not match",
        ):
            validate_evidence(corrupt_distance)

    def test_interact_tick_can_add_stationary_search_then_constraint_deactivates(self):
        records = _valid_records()
        interact_index = next(
            index
            for index, row in enumerate(records)
            if row["action"] == "interact"
        )
        previous = records[interact_index - 1]
        interact_calls = previous["stationary_search_calls"] + 1
        interact_transitions = previous["stationary_transition_count"]
        selected_frame = previous["stationary_selected_frame"]
        selected_range = previous["stationary_selected_range"]

        records[interact_index].update(
            stationary_constraint_active=True,
            stationary_search_calls=interact_calls,
            stationary_transition_count=interact_transitions,
            stationary_selected_frame=selected_frame,
            stationary_selected_range=selected_range,
        )
        for row in records[interact_index + 1 :]:
            row.update(
                stationary_constraint_active=False,
                stationary_search_calls=interact_calls,
                stationary_transition_count=interact_transitions,
                stationary_selected_frame=selected_frame,
                stationary_selected_range=selected_range,
            )

        try:
            validate_evidence(records)
        except PlacementEvidenceValidationError as error:
            self.fail(
                "a due Interact-tick stationary search was rejected: "
                f"{error}"
            )

        self.assertEqual(records[interact_index]["state"], "Preflight")
        self.assertTrue(
            records[interact_index]["stationary_constraint_active"]
        )
        self.assertGreater(
            records[interact_index]["stationary_search_calls"],
            previous["stationary_search_calls"],
        )
        self.assertNotEqual(records[interact_index + 1]["state"], "Locomotion")
        self.assertEqual(records[-1]["state"], "Locomotion")
        for row in records[interact_index + 1 :]:
            self.assertFalse(row["stationary_constraint_active"])
            self.assertEqual(row["stationary_search_calls"], interact_calls)
            self.assertEqual(
                row["stationary_transition_count"], interact_transitions
            )

    def test_stationary_counters_start_from_authoritative_zero_baseline(self):
        records = _valid_records()
        for row in records:
            row.update(
                stationary_search_calls=row["stationary_search_calls"] + 1,
                stationary_transition_count=
                    row["stationary_transition_count"] + 1,
                stationary_selected_frame=0,
                stationary_selected_range=0,
            )

        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "row 0 baseline|only a stationary-constrained pre-Interact "
            "search may change stationary counters",
        ):
            validate_evidence(records)

    def test_stationary_arrival_evidence_contract_is_strict(self):
        records = _valid_records()
        brake_index = next(
            index
            for index, row in enumerate(records)
            if row["reach_brake_latched"]
        )

        mutations = (
            (
                "candidate count",
                lambda rows: rows[0].update(stationary_candidate_count=185),
                "exactly 186 stationary candidates",
            ),
            (
                "first search",
                lambda rows: rows[brake_index].update(
                    stationary_search_calls=0,
                    stationary_transition_count=0,
                    stationary_selected_frame=-1,
                    stationary_selected_range=-1,
                ),
                "first brake-latched row",
            ),
            (
                "candidate membership",
                lambda rows: rows[brake_index].update(
                    stationary_selected_frame=93
                ),
                "not a derived candidate",
            ),
            (
                "range membership",
                lambda rows: rows[brake_index].update(
                    stationary_selected_range=1
                ),
                "not a derived candidate",
            ),
            (
                "transition bound",
                lambda rows: rows[brake_index].update(
                    stationary_transition_count=2
                ),
                "exceeds constrained-search calls",
            ),
            (
                "constraint continuity",
                lambda rows: rows[brake_index + 1].update(
                    stationary_constraint_active=False
                ),
                "stationary constraint must be exactly",
            ),
            (
                "brake speed",
                lambda rows: rows[brake_index].update(
                    brake_simulation_speed_mps=0.050001
                ),
                "brake simulation speed exceeds",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(records)
                mutate(changed)
                with self.assertRaisesRegex(
                    PlacementEvidenceValidationError, message
                ):
                    validate_evidence(changed)

    def test_pick_entry_evidence_reconstructs_both_slots_and_winner(self):
        records = _valid_records()
        validate_evidence(records)
        self.assertTrue(all(tuple(row) == EXPECTED_KEYS for row in records))
        self.assertEqual(records[0]["pick_preview_epoch"], -1)
        self.assertEqual(records[0]["pick_preview_snapshot_source"], "unset")
        frozen = next(
            row for row in records if row["pick_entry_selection_frozen"]
        )
        expected = _expected_pick_entry_slots(frozen)
        self.assertEqual(
            [
                frozen["pick_plus_identity"],
                frozen["pick_minus_identity"],
            ],
            ["Plus", "Minus"],
        )
        self.assertEqual(
            [
                frozen["pick_plus_evaluation_order"],
                frozen["pick_minus_evaluation_order"],
            ],
            [0, 1],
        )
        for prefix in ("pick_plus", "pick_minus"):
            self.assertLess(
                math.dist(
                    frozen[f"{prefix}_waypoint_position"],
                    expected[prefix]["waypoint_position"],
                ),
                2.0e-9,
            )
        self.assertGreater(
            frozen["pick_minus_hand_score"],
            frozen["pick_plus_hand_score"],
            "the higher hand score is deliberately path-infeasible",
        )
        self.assertEqual(_expected_pick_entry_winner(frozen), "pick_plus")
        self.assertEqual(frozen["pick_entry_selected_slot"], "Plus")
        self.assertTrue(frozen["pick_plus_selected"])
        self.assertFalse(frozen["pick_minus_selected"])

        missing = copy.deepcopy(records)
        missing[0].pop("pick_plus_identity")
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "fields/order"
        ):
            validate_evidence(missing)
        wrong_integer = copy.deepcopy(records)
        wrong_integer[0]["pick_preview_epoch"] = True
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "must be an integer"
        ):
            validate_evidence(wrong_integer)
        nonfinite = copy.deepcopy(records)
        nonfinite[0]["pick_plus_hand_score"] = math.inf
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "finite number"
        ):
            validate_evidence(nonfinite)

    def test_pick_entry_evidence_preserves_transient_poor_match_epochs(self):
        records = _valid_records()
        validate_evidence(records)
        poor = next(row for row in records if row["pick_preview_epoch"] == 0)
        ready = next(row for row in records if row["pick_preview_epoch"] == 1)
        self.assertTrue(poor["pick_plus_path_feasible"])
        self.assertFalse(poor["pick_plus_match_ready"])
        self.assertEqual(poor["pick_plus_match_reason"], "PoorMatch")
        self.assertEqual(
            (
                poor["pick_plus_feasible_entry_frame"],
                poor["pick_plus_contact_frame"],
            ),
            (114, 139),
        )
        self.assertTrue(poor["pick_plus_cost_available"])
        self.assertFalse(poor["pick_plus_eligible"])
        self.assertEqual(poor["pick_preview_consumed_epoch"], -1)
        self.assertEqual(ready["pick_preview_consumed_epoch"], 0)
        self.assertNotEqual(
            poor["pick_preview_snapshot_fingerprint"],
            ready["pick_preview_snapshot_fingerprint"],
        )

        mixed_priority = _valid_records()
        mixed = next(
            row for row in mixed_priority if row["pick_preview_epoch"] == 0
        )
        mixed["pick_plus_match_reason"] = "OutOfRange"
        mixed["pick_plus_cost_available"] = False
        validate_evidence(mixed_priority)
        self.assertGreater(mixed["pick_plus_total_cost"], 0.0)
        self.assertFalse(mixed["pick_plus_cost_available"])

    def test_pick_entry_evidence_rejects_slot_geometry_order_and_score_corruption(self):
        def set_component(row, field, component, value):
            row[field][component] = value

        cases = (
            (
                "identity",
                lambda row: row.__setitem__("pick_plus_identity", "Minus"),
                "identity/order",
            ),
            (
                "evaluation order",
                lambda row: row.__setitem__("pick_minus_evaluation_order", 0),
                "identity/order",
            ),
            (
                "signed arc",
                lambda row: set_component(
                    row,
                    "pick_plus_waypoint_position",
                    0,
                    -row["pick_plus_waypoint_position"][0],
                ),
                "signed arc",
            ),
            (
                "chord",
                lambda row: row.__setitem__(
                    "pick_plus_clearance_chord_m",
                    row["pick_plus_clearance_chord_m"] + 0.01,
                ),
                "clearance_chord_m",
            ),
            (
                "standoff",
                lambda row: row.__setitem__(
                    "pick_minus_preserved_standoff_m",
                    row["pick_minus_preserved_standoff_m"] + 0.01,
                ),
                "preserved_standoff_m",
            ),
            (
                "rotation",
                lambda row: row.__setitem__(
                    "pick_plus_waypoint_rotation",
                    [math.cos(0.005), 0.0, math.sin(0.005), 0.0],
                ),
                "rotation",
            ),
            (
                "height",
                lambda row: set_component(
                    row,
                    "pick_minus_waypoint_position",
                    1,
                    row["pick_minus_waypoint_position"][1] + 0.01,
                ),
                "signed arc",
            ),
            (
                "prospective root",
                lambda row: row.__setitem__(
                    "pick_plus_prospective_world_x",
                    row["pick_plus_prospective_world_x"] + 0.01,
                ),
                "prospective_world_x",
            ),
            (
                "hand score",
                lambda row: row.__setitem__(
                    "pick_minus_hand_score",
                    row["pick_minus_hand_score"] + 0.01,
                ),
                "hand score",
            ),
        )
        for label, mutate, message in cases:
            records = _valid_records()
            mutate(records[0])
            with self.subTest(label=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(records)

    def test_pick_entry_evidence_rejects_outcome_eligibility_and_selection_corruption(self):
        outcome_cases = (
            ("path reason", "pick_plus_path_reason", "BlockedPath", "path reason"),
            ("match ready", "pick_plus_match_ready", False, "match reason"),
            (
                "entry frame",
                "pick_plus_feasible_entry_frame",
                -1,
                "entry/contact",
            ),
            (
                "cost guard",
                "pick_plus_cost_available",
                False,
                "cost availability",
            ),
            ("eligibility", "pick_plus_eligible", False, "eligibility"),
        )
        for label, field, value, message in outcome_cases:
            records = _valid_records()
            ready_index = next(
                index
                for index, row in enumerate(records)
                if row["pick_preview_epoch"] == 1
                and not row["pick_entry_selection_frozen"]
            )
            records[ready_index][field] = value
            with self.subTest(label=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(records)

        records = _valid_records()
        frozen_index = next(
            index
            for index, row in enumerate(records)
            if row["pick_entry_selection_frozen"]
        )
        self.assertGreater(
            records[frozen_index]["pick_minus_hand_score"],
            records[frozen_index]["pick_plus_hand_score"],
        )
        self.assertFalse(records[frozen_index]["pick_minus_path_feasible"])
        records[frozen_index]["pick_entry_selected_slot"] = "Minus"
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "winner"
        ):
            validate_evidence(records)

        tied = _valid_tied_pick_entry_records()
        validate_evidence(tied)
        tied_frozen = next(
            index
            for index, row in enumerate(tied)
            if row["pick_entry_selection_frozen"]
        )
        self.assertEqual(
            tied[tied_frozen]["pick_plus_hand_score"],
            tied[tied_frozen]["pick_minus_hand_score"],
        )
        self.assertEqual(_expected_pick_entry_winner(tied[tied_frozen]), "pick_plus")
        tied[tied_frozen]["pick_entry_selected_slot"] = "Minus"
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "winner"
        ):
            validate_evidence(tied)

        near_tolerance_tie = _valid_tied_pick_entry_records()
        for row in near_tolerance_tie:
            row["pick_plus_hand_score"] -= 2.0e-5
            if row["pick_entry_selection_frozen"]:
                row.update(
                    pick_entry_selected_slot="Minus",
                    pick_plus_selected=False,
                    pick_minus_selected=True,
                )
                row["interaction_waypoint_position"] = copy.deepcopy(
                    row["pick_minus_waypoint_position"]
                )
                row["interaction_waypoint_rotation"] = copy.deepcopy(
                    row["pick_minus_waypoint_rotation"]
                )
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "independent.*winner|ranking"
        ):
            validate_evidence(near_tolerance_tie)

        changed = _valid_records()
        frozen_index = next(
            index
            for index, row in enumerate(changed)
            if row["pick_entry_selection_frozen"]
        )
        changed[frozen_index + 1]["pick_entry_selected_slot"] = "Minus"
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "changed|fell back"
        ):
            validate_evidence(changed)

        unfreeze = _valid_records()
        first_frozen = next(
            index
            for index, row in enumerate(unfreeze)
            if row["pick_entry_selection_frozen"]
        )
        neutral = copy.deepcopy(unfreeze[first_frozen])
        neutral.update(
            left_stick_command=[0.0, 0.0, 0.0],
            left_stick_magnitude=0.0,
            action="none",
            demo_phase="settle",
            phase_counter=0,
        )
        unfreeze.insert(first_frozen + 1, neutral)
        _reclock_and_measure(unfreeze)
        validate_evidence(unfreeze)
        unfreeze[first_frozen + 1].update(
            pick_entry_selected_slot="none",
            pick_entry_selection_frozen=False,
            pick_entry_freeze_tick=-1,
            pick_plus_selected=False,
            pick_minus_selected=False,
        )
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "unfreeze|monotonic"
        ):
            validate_evidence(unfreeze)

    def test_pick_entry_evidence_rejects_snapshot_mutation_retry_and_timing_corruption(self):
        cases = (
            (
                "zero fingerprint",
                lambda rows: rows[next(
                    i for i, row in enumerate(rows)
                    if row["pick_preview_epoch"] == 0
                )].__setitem__("pick_preview_snapshot_fingerprint", 0),
                "fingerprint.*nonzero",
            ),
            (
                "changed epoch fingerprint",
                lambda rows: rows[next(
                    i for i, row in enumerate(rows)
                    if row["pick_entry_selection_frozen"]
                ) + 1].__setitem__(
                    "pick_preview_snapshot_fingerprint", 0xFEED
                ),
                "fingerprint changed within",
            ),
            (
                "preview mutation",
                lambda rows: rows[0].__setitem__(
                    "pick_preview_mutation_count", 1
                ),
                "mutation count",
            ),
            (
                "partial pair",
                lambda rows: rows[next(
                    i for i, row in enumerate(rows)
                    if row["pick_preview_epoch"] == 0
                )].__setitem__("runtime_pick_preview_calls", 1),
                "two-call pair",
            ),
            (
                "callback count",
                lambda rows: rows[next(
                    i for i, row in enumerate(rows)
                    if row["pick_preview_epoch"] == 1
                    and not row["pick_entry_selection_frozen"]
                )].__setitem__("pick_preview_epoch_count", 3),
                "two-call pair",
            ),
            (
                "stalled elapsed",
                lambda rows: rows[next(
                    i for i, row in enumerate(rows)
                    if row["pick_preview_epoch"] == 1
                    and not row["pick_entry_selection_frozen"]
                )].__setitem__("pick_preview_elapsed_ticks", 1),
                "two-call pair|elapsed",
            ),
        )
        for label, mutate, message in cases:
            records = _valid_records()
            mutate(records)
            with self.subTest(label=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(records)

        over_bound = _valid_pick_entry_failure_records("deadline")
        over_bound[-1]["pick_preview_elapsed_ticks"] = MAX_WALK_TICKS + 1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "250 bound"
        ):
            validate_evidence(over_bound)

        post_freeze = _valid_records()
        index = next(
            i for i, row in enumerate(post_freeze)
            if row["pick_entry_selection_frozen"]
        ) + 1
        post_freeze[index].update(
            pick_preview_epoch=2,
            pick_preview_snapshot_fingerprint=0xF00D,
            pick_preview_epoch_count=3,
            runtime_pick_preview_calls=6,
            pick_preview_elapsed_ticks=3,
        )
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "post-freeze.*preview|retry"
        ):
            validate_evidence(post_freeze)

        multiple_interact = _valid_records()
        first_interact = next(
            i for i, row in enumerate(multiple_interact)
            if row["action"] == "interact"
        )
        multiple_interact[first_interact + 1]["action"] = "interact"
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "Interact"
        ):
            validate_evidence(multiple_interact)

        preview_reservation = _valid_records()
        waiting = next(
            row for row in preview_reservation
            if row["pick_preview_epoch"] == 0
        )
        waiting["pick_reservation_transition_count"] = 1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "reservation"
        ):
            validate_evidence(preview_reservation)

        interact_state = _valid_records()
        interact = next(
            row for row in interact_state if row["action"] == "interact"
        )
        interact["state"] = "Locomotion"
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "Interact.*Preflight"
        ):
            validate_evidence(interact_state)

        for state, corrupt_object_state in (
            ("Preflight", "Targeted"),
            ("Align", "Free"),
        ):
            lifecycle = _valid_records()
            row = next(row for row in lifecycle if row["state"] == state)
            row["object_state"] = corrupt_object_state
            with self.subTest(state=state), self.assertRaisesRegex(
                PlacementEvidenceValidationError,
                "Preflight|Align|Targeted|reservation",
            ):
                validate_evidence(lifecycle)

        for field in (
            "attachment_transition_count",
            "attached_to_free_transition_count",
        ):
            protected = _valid_records()
            waiting = next(
                row for row in protected if row["pick_preview_epoch"] == 0
            )
            waiting[field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError, "attachment.*counter"
            ):
                validate_evidence(protected)

        frozen_from_start = _valid_records()
        for row in frozen_from_start:
            _publish_pick_epoch(
                row,
                epoch=1,
                consumed_epoch=1,
                fingerprint=0xB22DF,
                plus=PICK_PLUS_READY,
                minus=PICK_MINUS_BLOCKED,
            )
            row.update(
                pick_entry_selected_slot="Plus",
                pick_entry_selection_frozen=True,
                pick_entry_freeze_tick=1,
                pick_plus_selected=True,
                pick_minus_selected=False,
            )
            row["interaction_waypoint_position"] = copy.deepcopy(
                row["pick_plus_waypoint_position"]
            )
            row["interaction_waypoint_rotation"] = copy.deepcopy(
                row["pick_plus_waypoint_rotation"]
            )
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "row 0 baseline|pre-preview.*first|sentinel.*prefix",
        ):
            validate_evidence(frozen_from_start)

        after_failure = _valid_pick_entry_failure_records("no_path")
        trailing = copy.deepcopy(after_failure[-1])
        trailing.update(
            pick_preview_pending_failure="none",
            pick_preview_failure_recorded=False,
        )
        after_failure.append(trailing)
        _reclock_and_measure(after_failure)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "terminal failure marker"
        ):
            validate_evidence(after_failure)

    def test_pick_entry_evidence_requires_callback_handoff_and_bounded_elapsed(self):
        ready_at_bound = _valid_ready_at_bound_records()
        validate_evidence(ready_at_bound)
        frozen = next(
            row for row in ready_at_bound if row["pick_entry_selection_frozen"]
        )
        self.assertEqual(frozen["pick_preview_elapsed_ticks"], MAX_WALK_TICKS)
        self.assertEqual(frozen["pick_entry_selected_slot"], "Plus")
        deferred = [
            row for row in ready_at_bound
            if row["pick_preview_epoch"] >= 0
            and not row["pick_entry_selection_frozen"]
        ]
        self.assertTrue(any(
            row["displayed_root_speed_mps"] == 0.0
            and row["left_stick_magnitude"] == 0.0
            for row in deferred
        ))
        self.assertEqual(
            [row["pick_preview_elapsed_ticks"] for row in deferred[-3:]],
            [MAX_WALK_TICKS - 2, MAX_WALK_TICKS - 1, MAX_WALK_TICKS],
        )

        for kind in ("no_path", "deadline"):
            failure = _valid_pick_entry_failure_records(kind)
            validate_evidence(failure)
            with self.subTest(kind=kind):
                self.assertTrue(failure[-1]["pick_preview_failure_recorded"])
                self.assertEqual(failure[-1]["pick_preview_pending_failure"], kind)
                self.assertEqual(failure[-1]["left_stick_magnitude"], 0.0)

        same_tick = _valid_records()
        ready_index = next(
            i for i, row in enumerate(same_tick)
            if row["pick_preview_epoch"] == 1
            and not row["pick_entry_selection_frozen"]
        )
        same_tick[ready_index]["pick_preview_consumed_epoch"] = 1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "current epoch|callback"
        ):
            validate_evidence(same_tick)

        navigation_before_freeze = _valid_records()
        waiting = next(
            row for row in navigation_before_freeze
            if row["pick_preview_epoch"] == 0
        )
        waiting["left_stick_command"] = [0.1, 0.0, 0.0]
        waiting["left_stick_magnitude"] = 0.1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "before selection freeze"
        ):
            validate_evidence(navigation_before_freeze)

        failure_before_pair = _valid_pick_entry_failure_records("no_path")
        failure_before_pair[-1]["pick_preview_consumed_epoch"] = -1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "before its complete pair"
        ):
            validate_evidence(failure_before_pair)

        unrecorded = _valid_pick_entry_failure_records("deadline")
        unrecorded[-1]["pick_preview_failure_recorded"] = False
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "not flush-recorded"
        ):
            validate_evidence(unrecorded)

        changed_terminal_pair = _valid_pick_entry_failure_records("no_path")
        changed_terminal_pair[-1]["pick_minus_total_cost"] = 0.5
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "complete pair|retain"
        ):
            validate_evidence(changed_terminal_pair)

        for epoch, label in ((0, "deferred"), (1, "ready")):
            stalled = _valid_records()
            index = next(
                i for i, row in enumerate(stalled)
                if row["pick_preview_epoch"] == epoch
                and not row["pick_entry_selection_frozen"]
            )
            stalled.insert(index + 1, copy.deepcopy(stalled[index]))
            for row in stalled:
                if row["pick_entry_selection_frozen"]:
                    row["pick_entry_freeze_tick"] += 1
            _reclock_and_measure(stalled)
            with self.subTest(stalled=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError,
                "immediately next native tick|stalled",
            ):
                validate_evidence(stalled)

        blocked_deadline = _valid_pick_entry_failure_records("deadline")
        for row in blocked_deadline:
            if row["pick_preview_epoch"] >= 0:
                _publish_pick_epoch(
                    row,
                    epoch=row["pick_preview_epoch"],
                    consumed_epoch=row["pick_preview_consumed_epoch"],
                    fingerprint=row["pick_preview_snapshot_fingerprint"],
                    plus=PICK_MINUS_BLOCKED,
                    minus=PICK_MINUS_BLOCKED,
                )
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "no.path|deadline.*feasible"
        ):
            validate_evidence(blocked_deadline)

        blocked_retry = _valid_pick_entry_failure_records("no_path")
        retry = copy.deepcopy(blocked_retry[1])
        _publish_pick_epoch(
            retry,
            epoch=1,
            consumed_epoch=0,
            fingerprint=0xABC02,
            plus=PICK_MINUS_BLOCKED,
            minus=PICK_MINUS_BLOCKED,
        )
        blocked_retry.insert(2, retry)
        _publish_pick_epoch(
            blocked_retry[-1],
            epoch=1,
            consumed_epoch=1,
            fingerprint=0xABC02,
            plus=PICK_MINUS_BLOCKED,
            minus=PICK_MINUS_BLOCKED,
        )
        _reclock_and_measure(blocked_retry)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "no.path.*immediate|retry"
        ):
            validate_evidence(blocked_retry)

        eligible_retry = _valid_records()
        ready_index = next(
            i for i, row in enumerate(eligible_retry)
            if row["pick_preview_epoch"] == 1
            and not row["pick_entry_selection_frozen"]
        )
        retry = copy.deepcopy(eligible_retry[ready_index])
        _publish_pick_epoch(
            retry,
            epoch=2,
            consumed_epoch=1,
            fingerprint=0xABC03,
            plus=PICK_PLUS_READY,
            minus=PICK_MINUS_BLOCKED,
        )
        eligible_retry.insert(ready_index + 1, retry)
        for row in eligible_retry[ready_index + 2:]:
            _publish_pick_epoch(
                row,
                epoch=2,
                consumed_epoch=2,
                fingerprint=0xABC03,
                plus=PICK_PLUS_READY,
                minus=PICK_MINUS_BLOCKED,
            )
            if row["pick_entry_selection_frozen"]:
                row["pick_entry_freeze_tick"] += 1
        _reclock_and_measure(eligible_retry)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "eligible.*freeze|retry"
        ):
            validate_evidence(eligible_retry)

        for kind in ("no_path", "deadline"):
            delayed = _valid_pick_entry_failure_records(kind)
            ordinary = copy.deepcopy(delayed[-1])
            ordinary.update(
                pick_preview_pending_failure="none",
                pick_preview_failure_recorded=False,
            )
            delayed.insert(len(delayed) - 1, ordinary)
            _reclock_and_measure(delayed)
            with self.subTest(delayed=kind), self.assertRaisesRegex(
                PlacementEvidenceValidationError,
                "immediate|decisive|terminal",
            ):
                validate_evidence(delayed)

    def test_pick_entry_evidence_requires_current_pack_plus_selection(self):
        records = _valid_records()
        validate_evidence(records, require_current_pack=True)

        for kind in ("no_path", "deadline"):
            failure = _valid_pick_entry_failure_records(kind)
            validate_evidence(failure)
            with self.subTest(failure=kind), self.assertRaisesRegex(
                PlacementEvidenceValidationError,
                "current pack.*successful|ready Plus",
            ):
                validate_evidence(failure, require_current_pack=True)

        minus = _valid_minus_selected_records()
        validate_evidence(minus)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "current pack"
        ):
            validate_evidence(minus, require_current_pack=True)

        cases = (
            ("entry frame", "pick_plus_feasible_entry_frame", 115),
            ("contact frame", "pick_plus_contact_frame", 140),
            ("maximum cost", "pick_plus_total_cost", PICK_ENTRY_MAXIMUM_COST + 0.01),
        )
        for label, field, value in cases:
            changed = _valid_records()
            for row in changed:
                if row["pick_preview_epoch"] == 1:
                    row[field] = value
            validate_evidence(changed)
            with self.subTest(label=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError, "current pack"
            ):
                validate_evidence(changed, require_current_pack=True)

        changed_reason = _valid_records()
        for row in changed_reason:
            if row["pick_preview_epoch"] == 1:
                row["pick_minus_path_reason"] = "OutOfRange"
                row["pick_minus_match_reason"] = "OutOfRange"
        validate_evidence(changed_reason)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "current pack"
        ):
            validate_evidence(changed_reason, require_current_pack=True)

    def test_initial_walk_contract_rejects_every_shortcut(self):
        mutations = {
            "owned": (
                3,
                "owns_pose",
                True,
                "walking prefix|initial approach",
            ),
            "attached": (
                3,
                "attached",
                True,
                "walking prefix|initial approach|attachment transition counters",
            ),
            "canonical": (3, "canonical_snapshot_used", True, "canonical snapshot"),
            "relocation": (3, "root_relocation_applied", True, "relocate"),
            "simulation init": (0, "simulation_root_initialized_from_reach", True, "root initializer"),
            "display init": (0, "displayed_root_initialized_from_reach", True, "root initializer"),
            "provider": (3, "locomotion_provider_kind", "canonical", "live_flat"),
        }
        for label, (index, field, value, message) in mutations.items():
            with self.subTest(label=label):
                records = _valid_records()
                records[index][field] = value
                with self.assertRaisesRegex(PlacementEvidenceValidationError, message):
                    validate_evidence(records)

        records = _valid_records()
        records[0]["root_position"][2] = 0.20
        for joint in range(22):
            records[0]["joint_world_positions"][joint][2] = 0.20
        _reclock_and_measure(records)
        with self.assertRaisesRegex(PlacementEvidenceValidationError, "strictly greater"):
            validate_evidence(records)

        records = _valid_records()
        records[24]["demo_phase"] = "settle"
        with self.assertRaisesRegex(PlacementEvidenceValidationError, "25 consecutive"):
            validate_evidence(records)

        records = _valid_records()
        interact = next(i for i, row in enumerate(records) if row["action"] == "interact")
        for row in records[:interact]:
            row["walking_origin_displacement_m"] *= 0.50
        with self.assertRaisesRegex(PlacementEvidenceValidationError, "does not match|2.00"):
            validate_evidence(records)

    def test_settle_and_interact_require_live_exact_resolver(self):
        cases = (
            (-1, "phase_counter", 4, "five consecutive"),
            (-1, "pickup_distance_m", 0.46, "does not match|standoff"),
            (-1, "displayed_root_speed_mps", 0.100001, "does not match|standoff"),
            (0, "pick_resolver_maximum_m", 0.99, "1.00"),
            (0, "pick_resolver_live_flat", False, "live flat"),
        )
        for offset, field, value, message in cases:
            records = _valid_records()
            interact = next(i for i, row in enumerate(records) if row["action"] == "interact")
            index = interact + offset
            records[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(records)

        records = _valid_records()
        interact = next(
            i for i, row in enumerate(records) if row["action"] == "interact"
        )
        records[interact - 1]["left_stick_command"] = [0.1, 0.0, 0.0]
        records[interact - 1]["left_stick_magnitude"] = 0.1
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "no-input"
        ):
            validate_evidence(records)

    def test_interact_requires_live_reach_alignment_funnel(self):
        records = _valid_records()
        for row in records:
            row["reach_waypoint_position"][0] = 0.151
            row["reach_waypoint_position_error_m"] = math.hypot(
                row["root_position"][0] - 0.151,
                row["root_position"][2] - row["reach_waypoint_position"][2],
            )
        _refresh_pick_entry_geometry(records)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "Reach waypoint position"
        ):
            validate_evidence(records)

        records = _valid_records()
        angle = math.radians(20.001)
        waypoint_rotation = [
            math.cos(0.5 * angle), 0.0, math.sin(0.5 * angle), 0.0
        ]
        for row in records:
            row["reach_waypoint_rotation"] = copy.deepcopy(waypoint_rotation)
            row["reach_waypoint_yaw_error_degrees"] = 20.001
        _refresh_pick_entry_geometry(records)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "Reach waypoint yaw"
        ):
            validate_evidence(records)

        records = _valid_records()
        interact = next(
            i for i, row in enumerate(records) if row["action"] == "interact"
        )
        records[interact - 1]["reach_waypoint_position_error_m"] = 0.01
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "waypoint position error does not match"
        ):
            validate_evidence(records)

    def test_settle_and_interact_require_selected_waypoint_position_bound(self):
        records = _valid_records()
        interact_index = next(
            i for i, row in enumerate(records) if row["action"] == "interact"
        )
        first_settle_index = interact_index - SETTLE_TICKS
        for row in records[first_settle_index - 1:interact_index + 1]:
            translation_x = 0.10 - row["root_position"][0]
            row["root_position"][0] += translation_x
            for joint_position in row["joint_world_positions"]:
                joint_position[0] += translation_x
        _reclock_and_measure(records)

        aligned_rows = records[first_settle_index:interact_index + 1]
        self.assertTrue(all(
            row["reach_waypoint_position_error_m"] <=
                REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
            for row in aligned_rows
        ))
        self.assertTrue(all(
            STANDOFF_MIN_M <= row["pickup_distance_m"] <= STANDOFF_MAX_M
            for row in aligned_rows
        ))
        self.assertTrue(all(
            row["displayed_root_speed_mps"] <= SETTLE_MAX_SPEED_MPS
            for row in aligned_rows
        ))
        self.assertTrue(all(
            _planar_distance(
                row["root_position"], row["interaction_waypoint_position"]
            ) > REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
            for row in aligned_rows
        ))
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "selected interaction-waypoint position error must be at most "
            "0.15 m through Interact",
        ):
            validate_evidence(records)

    def test_interaction_waypoint_preserves_chord_standoff_and_hand_side(self):
        records = _valid_records()
        validate_evidence(records)
        row = records[0]
        self.assertTrue(math.isclose(
            _planar_distance(
                row["interaction_waypoint_position"],
                row["reach_waypoint_position"],
            ),
            row["interaction_lateral_offset_m"],
            abs_tol=2.0e-6,
        ))
        self.assertTrue(math.isclose(
            _planar_distance(
                row["interaction_waypoint_position"],
                row["interaction_object_position"],
            ),
            _planar_distance(
                row["reach_waypoint_position"],
                row["interaction_object_position"],
            ),
            abs_tol=2.0e-6,
        ))
        hand_lateral = _hand_sided_lateral(row)
        chord = [
            value - origin
            for value, origin in zip(
                row["interaction_waypoint_position"],
                row["reach_waypoint_position"],
            )
        ]
        self.assertGreater(
            sum(value * axis for value, axis in zip(chord, hand_lateral)),
            0.0,
        )

    def test_interaction_waypoint_rejects_straight_translation(self):
        records = _valid_records()
        row = records[0]
        lateral = _hand_sided_lateral(row)
        translated = [
            value + row["interaction_lateral_offset_m"] * axis
            for value, axis in zip(row["reach_waypoint_position"], lateral)
        ]
        for record in records:
            record["interaction_waypoint_position"] = translated.copy()
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "standoff|arc|selected slot"
        ):
            validate_evidence(records)

    def test_interaction_waypoint_recomputes_rotation_and_hand_side(self):
        records = _valid_records()
        frozen = next(
            row for row in records if row["pick_entry_selection_frozen"]
        )
        frozen["interaction_waypoint_position"][0] += 0.01
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "interaction waypoint",
        ):
            validate_evidence(records)

        yaw = math.radians(63.0)
        yaw_rotation = [
            math.cos(0.5 * yaw), 0.0, math.sin(0.5 * yaw), 0.0
        ]

        def rotate_world_y(position):
            cosine, sine = math.cos(yaw), math.sin(yaw)
            return [
                cosine * position[0] + sine * position[2],
                position[1],
                -sine * position[0] + cosine * position[2],
            ]

        base = copy.deepcopy(_valid_records()[0])
        base_waypoint, _ = _expected_interaction_waypoint(base)
        right = copy.deepcopy(base)
        right["reach_waypoint_position"] = rotate_world_y(
            right["reach_waypoint_position"]
        )
        right["interaction_object_position"] = rotate_world_y(
            right["interaction_object_position"]
        )
        right["interaction_object_rotation"] = pickup._quaternion_multiply(
            yaw_rotation, right["interaction_object_rotation"]
        )
        right["interaction_hand_side"] = 1
        right_waypoint, right_offset = _expected_interaction_waypoint(right)
        self.assertLess(
            math.dist(right_waypoint, rotate_world_y(base_waypoint)),
            2.0e-9,
        )

        left = copy.deepcopy(right)
        left["interaction_hand_side"] = -1
        left_waypoint, left_offset = _expected_interaction_waypoint(left)
        self.assertAlmostEqual(right_offset, 0.091, places=6)
        self.assertAlmostEqual(left_offset, right_offset, places=12)
        for waypoint in (right_waypoint, left_waypoint):
            self.assertAlmostEqual(
                _planar_distance(waypoint, right["reach_waypoint_position"]),
                right_offset,
                places=6,
            )
            self.assertAlmostEqual(
                _planar_distance(waypoint, right["interaction_object_position"]),
                _planar_distance(
                    right["reach_waypoint_position"],
                    right["interaction_object_position"],
                ),
                places=6,
            )
        self.assertGreater(
            sum(
                (value - origin) * axis
                for value, origin, axis in zip(
                    right_waypoint,
                    right["reach_waypoint_position"],
                    _hand_sided_lateral(right),
                )
            ),
            0.0,
        )
        self.assertGreater(
            sum(
                (value - origin) * axis
                for value, origin, axis in zip(
                    left_waypoint,
                    left["reach_waypoint_position"],
                    _hand_sided_lateral(left),
                )
            ),
            0.0,
        )
        self.assertGreater(
            _planar_distance(right_waypoint, left_waypoint), 0.17
        )

    def test_interaction_waypoint_exact_score_tie_uses_hand_fallback(self):
        right = copy.deepcopy(_valid_records()[0])
        right["interaction_approach_direction_object"] = [1.0, 0.0, 0.0]
        right["interaction_hand_side"] = 1
        right_waypoint, chord = _expected_interaction_waypoint(right)

        reach = right["reach_waypoint_position"]
        object_position = right["interaction_object_position"]
        radius_x = reach[0] - object_position[0]
        radius_z = reach[2] - object_position[2]
        standoff = math.hypot(radius_x, radius_z)
        theta = 2.0 * math.asin(chord / (2.0 * standoff))

        def candidate(angle):
            cosine, sine = math.cos(angle), math.sin(angle)
            return [
                object_position[0] + cosine * radius_x + sine * radius_z,
                reach[1],
                object_position[2] - sine * radius_x + cosine * radius_z,
            ]

        plus = candidate(+theta)
        minus = candidate(-theta)
        left = copy.deepcopy(right)
        left["interaction_hand_side"] = -1
        left_waypoint, left_chord = _expected_interaction_waypoint(left)
        self.assertAlmostEqual(left_chord, chord, places=12)
        self.assertLess(math.dist(right_waypoint, plus), 2.0e-12)
        self.assertLess(math.dist(left_waypoint, minus), 2.0e-12)

    def test_interaction_waypoint_rejects_cpp_subthreshold_planar_approach(self):
        row = copy.deepcopy(_valid_records()[0])
        angle = math.acos(5.0e-6)
        row["interaction_object_rotation"] = [
            math.cos(0.5 * angle), math.sin(0.5 * angle), 0.0, 0.0
        ]
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "planar"
        ):
            _expected_interaction_waypoint(row)

    def test_interaction_waypoint_rotation_and_object_provenance_are_bound(self):
        records = _valid_records()
        yaw = math.radians(10.0)
        for row in records:
            row["interaction_waypoint_rotation"] = [
                math.cos(0.5 * yaw), 0.0, math.sin(0.5 * yaw), 0.0
            ]
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "waypoint rotation|Reach rotation|selected slot",
        ):
            validate_evidence(records)

        records = _valid_records()
        for row in records:
            row["interaction_object_position"][0] += 0.01
            waypoint, chord = _expected_interaction_waypoint(row)
            row["interaction_waypoint_position"] = waypoint
            row["interaction_lateral_offset_m"] = chord
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "observed|provenance"
        ):
            validate_evidence(records)

        records = _valid_records()
        object_yaw = math.radians(20.0)
        for row in records:
            row["interaction_object_rotation"] = [
                math.cos(0.5 * object_yaw),
                0.0,
                math.sin(0.5 * object_yaw),
                0.0,
            ]
            waypoint, chord = _expected_interaction_waypoint(row)
            row["interaction_waypoint_position"] = waypoint
            row["interaction_lateral_offset_m"] = chord
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "observed|provenance"
        ):
            validate_evidence(records)

    def test_interaction_waypoint_rejects_corrupted_arc_inputs(self):
        def mutate_all(records, mutation):
            for row in records:
                mutation(row)

        cases = (
            (
                "object_position",
                lambda row: row["interaction_object_position"].__setitem__(
                    0, row["interaction_object_position"][0] + 0.01
                ),
            ),
            (
                "chord",
                lambda row: row.__setitem__(
                    "interaction_lateral_offset_m",
                    row["interaction_lateral_offset_m"] + 0.005,
                ),
            ),
            (
                "dimensions",
                lambda row: row["interaction_object_dimensions"].__setitem__(
                    0, row["interaction_object_dimensions"][0] + 0.02
                ),
            ),
            (
                "rotation",
                lambda row: row.__setitem__(
                    "interaction_object_rotation",
                    [math.cos(0.1), 0.0, 0.0, math.sin(0.1)],
                ),
            ),
            (
                "approach",
                lambda row: row.__setitem__(
                    "interaction_approach_direction_object",
                    [math.sin(0.2), 0.0, math.cos(0.2)],
                ),
            ),
            (
                "clearance",
                lambda row: row.__setitem__(
                    "interaction_clearance_radius_m",
                    row["interaction_clearance_radius_m"] + 0.01,
                ),
            ),
            (
                "hand",
                lambda row: row.__setitem__("interaction_hand_side", -1),
            ),
        )
        for label, mutation in cases:
            records = _valid_records()
            mutate_all(records, mutation)
            with self.subTest(label=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError,
                "interaction|arc|standoff|lateral",
            ):
                validate_evidence(records)

        records = _valid_records()
        for row in records:
            if row["pick_entry_selection_frozen"]:
                row["interaction_waypoint_position"] = copy.deepcopy(
                    row["pick_minus_waypoint_position"]
                )
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "interaction|arc"
        ):
            validate_evidence(records)

    def test_fixed_clock_and_destination_policy_cannot_be_faked(self):
        cases = (
            (10, "runtime_tick", 99, "runtime_tick"),
            (10, "scheduler_phase", 1, "scheduler phase"),
            (10, "destination_generation", 2, "destination identity"),
            (10, "destination_generation_changed_before_place", True, "generation changed"),
            (10, "placement_surface_resolver_calls", 1, "surface resolver"),
            (10, "free_selector_calls", 1, "free place selector"),
            (10, "external_match_input_calls", 1, "PlaceMatchInput"),
            (10, "preview_mutation_count", 1, "must not mutate"),
        )
        for index, field, value, message in cases:
            records = _valid_records()
            records[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(records)

    def test_preview_and_ik_identity_are_runtime_owned(self):
        records = _valid_records()
        carry = next(
            i for i, row in enumerate(records)
            if row["state"] == "Carry" and row["demo_phase"] == "carry_preview"
        )
        cases = (
            (carry, "preview_accepted", False, "initial Carry preview"),
            (carry, "preview_ready", True, "initial Carry preview|far preview"),
            (carry, "selected_staging_root", False, "initial Carry preview"),
            (carry, "far_selection_id", 0, "initial Carry preview"),
            (carry, "preview_ik_fingerprint", 0, "fingerprint"),
            (carry + 1, "preview_ik_fingerprint", 7, "fingerprint changed"),
            (carry + 1, "ik_maximum_request_position_m", 0.11, "differs"),
            (carry + 1, "runtime_preview_source_state", "none", "preview call"),
        )
        for index, field, value, message in cases:
            mutated = _valid_records()
            mutated[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(mutated)

    def test_carry_staging_requires_ordinary_motion_and_a_new_ready_id(self):
        records = _valid_records()
        carry_indices = [
            i for i, row in enumerate(records)
            if row["state"] == "Carry" and row["demo_phase"] == "carry_staging"
        ]
        place_index = next(
            i for i, row in enumerate(records) if row["action"] == "place"
        )
        cases = (
            (carry_indices[5], "carry_staging_tick", 9, "numbered"),
            (carry_indices[5], "left_stick_command", [0.0, 0.0, 0.0], "left-stick"),
            (carry_indices[5], "direct_root_write", True, "directly write"),
            (place_index, "preview_ready", False, "ready live preview"),
            (place_index, "preview_root_error_m", 0.250001, "ready live preview"),
            (place_index, "preview_yaw_error_degrees", 25.000001, "ready live preview"),
        )
        for index, field, value, message in cases:
            mutated = _valid_records()
            mutated[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(mutated)

        for field, value in (
            ("hand_constraint_weight", 0.0),
            ("object_state", "Free"),
            ("owns_pose", False),
        ):
            mutated = _valid_records()
            mutated[carry_indices[5]][field] = value
            with self.subTest(owned_field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError,
                "Held, attached, owned, validated full-weight grasp",
            ):
                validate_evidence(mutated)

        mutated = _valid_records()
        far = next(
            i for i, row in enumerate(mutated)
            if row["demo_phase"] == "carry_preview"
        )
        mutated[far]["carry_staging_tick"] = 0
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "excluded from the staging count"
        ):
            validate_evidence(mutated)

        mutated = _valid_records()
        last_carry = max(
            i for i, row in enumerate(mutated) if row["state"] == "Carry"
        )
        place = next(i for i, row in enumerate(mutated) if row["action"] == "place")
        mutated[last_carry]["preview_ready"] = True
        mutated[place]["preview_ready"] = False
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError, "cannot be ready before"
        ):
            validate_evidence(mutated)

        mutated = _valid_records()
        place = next(i for i, row in enumerate(mutated) if row["action"] == "place")
        mutated[place]["submitted_selection_id"] = mutated[place]["far_selection_id"]
        with self.assertRaisesRegex(PlacementEvidenceValidationError, "current distinct"):
            validate_evidence(mutated)

    def test_place_release_and_final_support_are_strict(self):
        records = _valid_records()
        place = next(i for i, row in enumerate(records) if row["action"] == "place")
        final = len(records) - 1
        cases = (
            (place, "action", "none", "Place must occur exactly once"),
            (final, "place_position_error_m", 0.020001, "position error"),
            (final, "place_orientation_error_degrees", 10.000001, "orientation error"),
            (final, "actual_support_gap_m", 0.020001, "support gap"),
            (final, "actual_footprint_valid", False, "actual bounds"),
            (final, "actual_bound_corners_valid", False, "actual bounds"),
            (final, "actual_overhead_valid", False, "actual bounds"),
            (final, "support_sweep_clear", False, "actual bounds"),
            (final, "destination_support_committed", False, "destination support"),
            (final, "place_mode", "none", "place mode"),
            (final, "state", "Carry", "collapsed states|final state|full-weight grasp"),
        )
        for index, field, value, message in cases:
            mutated = _valid_records()
            mutated[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(mutated)

        mutated = _valid_records()
        release = next(i for i, row in enumerate(mutated) if row["state"] == "PlaceRelease")
        mutated[release - 1]["attached"] = False
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "release exactly once|attached owned|full-weight grasp|"
            "attachment transition counters",
        ):
            validate_evidence(mutated)

        mutated = _valid_records()
        mutated[-1]["object_position"][0] += 0.001
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "position error does not match",
        ):
            validate_evidence(mutated)

        mutated = _valid_records()
        del mutated[-2]
        _reclock_and_measure(mutated)
        with self.assertRaisesRegex(
            PlacementEvidenceValidationError,
            "eighteen to seventy-five numbered no-input Locomotion handoff",
        ):
            validate_evidence(mutated)

    def test_existing_fk_limits_remain_binding(self):
        cases = (
            ("distal", 4, [0.480001, 0.0, 0.0], "12 m/s"),
            ("authority", 14, [0.200001, 0.0, 0.0], "authority seam"),
        )
        for label, joint, delta, message in cases:
            records = _valid_records()
            index = next(i for i, row in enumerate(records) if row["state"] == "Preflight")
            if label == "distal":
                index = 5
            previous = records[index - 1]["joint_world_positions"][joint]
            records[index]["joint_world_positions"][joint] = [
                previous[axis] + delta[axis] for axis in range(3)
            ]
            with self.subTest(label=label), self.assertRaisesRegex(
                PlacementEvidenceValidationError, message
            ):
                validate_evidence(records)

        records = _valid_records()
        records[5]["joint_world_rotations"][14] = [
            math.cos(math.radians(60.001) / 2.0),
            math.sin(math.radians(60.001) / 2.0),
            0.0,
            0.0,
        ]
        with self.assertRaisesRegex(PlacementEvidenceValidationError, "60 degrees"):
            validate_evidence(records)

    def test_jsonl_and_png_are_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "placement.jsonl"
            screenshot = root / "placement.png"
            records = _valid_records()
            log.write_text("".join(_record_line(row) for row in records), encoding="utf-8")
            _write_nonblank_png(screenshot)
            loaded = load_evidence(log)
            validate_evidence(loaded)
            validate_screenshot(screenshot)

            blank = root / "blank.png"
            pickup._write_png(blank)
            with self.assertRaisesRegex(
                PlacementEvidenceValidationError, "visually blank"
            ):
                validate_screenshot(blank)


class Task8PlacementPolicyTests(unittest.TestCase):
    @staticmethod
    def _between(source: str, begin: str, end: str) -> str:
        start = source.index(begin)
        stop = source.index(end, start + len(begin))
        return source[start:stop]

    @staticmethod
    def _lambda_assignment(source: str, declaration: str) -> str:
        start = source.index(declaration)
        stop = source.index("\n    };", start) + len("\n    };")
        return source[start:stop]

    @staticmethod
    def _braced_scope_from(source: str, start: int) -> str:
        opening = source.index("{", start)
        depth = 0
        for index in range(opening, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start:index + 1]
        raise AssertionError(f"unclosed C++ scope at byte {start}")

    @classmethod
    def _named_braced_scope(cls, source: str, name: str) -> str:
        for match in re.finditer(rf"\b{re.escape(name)}\b", source):
            opening = source.find("{", match.end())
            semicolon = source.find(";", match.end())
            if opening >= 0 and (semicolon < 0 or opening < semicolon):
                return cls._braced_scope_from(source, match.start())
        raise AssertionError(f"C++ definition for {name!r} is missing")

    @staticmethod
    def _split_cpp_arguments(arguments: str) -> list[str]:
        result = []
        start = 0
        depths = {"(": 0, "[": 0, "{": 0, "<": 0}
        closing = {")": "(", "]": "[", "}": "{", ">": "<"}
        for index, character in enumerate(arguments):
            if character in depths:
                depths[character] += 1
            elif character in closing and depths[closing[character]] > 0:
                depths[closing[character]] -= 1
            elif character == "," and all(depth == 0 for depth in depths.values()):
                result.append(arguments[start:index].strip())
                start = index + 1
        tail = arguments[start:].strip()
        if tail:
            result.append(tail)
        return result

    @classmethod
    def _call_arguments(cls, source: str, call_name: str) -> list[str]:
        try:
            start = source.index(call_name)
            opening = source.index("(", start + len(call_name))
        except ValueError as error:
            raise AssertionError(
                f"C++ call to {call_name!r} is missing"
            ) from error
        depth = 0
        for index in range(opening, len(source)):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
                if depth == 0:
                    return cls._split_cpp_arguments(
                        source[opening + 1:index]
                    )
        raise AssertionError(f"unterminated C++ call to {call_name!r}")

    @classmethod
    def _frozen_preview_helper(cls, source: str) -> str:
        candidates = []
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", source):
            name = match.group(1)
            opening = source.find("{", match.end())
            semicolon = source.find(";", match.end())
            if opening < 0 or (semicolon >= 0 and semicolon < opening):
                continue
            signature = source[match.start():opening]
            if (
                "SmartPickupPreviewCallback" in signature
                and "PickEntryRoot" in signature
            ):
                candidates.append((name, match.start()))
        definitions = {(name, offset) for name, offset in candidates}
        if len(definitions) != 1:
            raise AssertionError(
                "interaction_smart_pickup_controller.cpp must define exactly "
                "one narrow helper that accepts both a frozen PickEntryRoot "
                "and SmartPickupPreviewCallback"
            )
        _, start = next(iter(definitions))
        return cls._braced_scope_from(source, start)

    def _assert_single_root_preview_delegation(
        self,
        helper: str,
        callback_expression: str,
    ) -> None:
        opening = helper.index("{")
        signature = helper[:opening]
        body = helper[opening + 1:]
        parameter_patterns = (
            (
                "live snapshot",
                r"(?:const\s+)?(?:interaction::)?LocomotionSnapshot"
                r"(?:\s+const)?\s*&\s*([A-Za-z_]\w*)",
            ),
            (
                "frozen prospective root",
                r"(?:const\s+)?(?:interaction::)?PickEntryRoot"
                r"(?:\s+const)?\s*&?\s*([A-Za-z_]\w*)",
            ),
            (
                "target handle",
                r"(?:const\s+)?(?:interaction::)?TargetHandle"
                r"(?:\s+const)?\s*&?\s*([A-Za-z_]\w*)",
            ),
            (
                "affordance ID",
                r"(?:const\s+)?(?:std::)?uint32_t(?:\s+const)?\s*&?\s*"
                r"([A-Za-z_]\w*)",
            ),
        )
        parameter_names = []
        for label, pattern in parameter_patterns:
            parameter = re.search(pattern, signature)
            self.assertIsNotNone(
                parameter,
                f"single-preview helper must accept the exact {label}",
            )
            if parameter is not None:
                parameter_names.append(parameter.group(1))
        calls = list(re.finditer(callback_expression, body))
        self.assertEqual(
            len(calls),
            1,
            "single-preview helper must contain exactly one direct preview "
            "delegation",
        )
        call = calls[0]
        call_open = call.end() - 1
        depth = 0
        call_close = None
        for index in range(call_open, len(body)):
            if body[index] == "(":
                depth += 1
            elif body[index] == ")":
                depth -= 1
                if depth == 0:
                    call_close = index
                    break
        self.assertIsNotNone(call_close, "preview delegation call is unterminated")
        if call_close is None:
            return
        arguments = self._split_cpp_arguments(body[call_open + 1:call_close])
        self.assertEqual(
            arguments,
            parameter_names,
            "preview delegation must forward the exact live snapshot, frozen "
            "root, target handle, and affordance ID in native order",
        )
        statement_start = max(
            body.rfind(";", 0, call.start()),
            body.rfind("{", 0, call.start()),
            body.rfind("}", 0, call.start()),
        ) + 1
        self.assertEqual(
            body[statement_start:call.start()].strip(),
            "return",
            "the native preview result must be returned directly",
        )
        for forbidden in (
            "PickEntrySlots",
            "std::array",
            "slots.ordered",
            "make_pick_entry_slots",
            "choose_pick_entry_slot",
            "preview_pick_entry_slots",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, helper)
        self.assertNotRegex(helper, r"\b(?:for|while)\s*\(")

    @staticmethod
    def _make_target(source: str, name: str) -> str:
        lines = source.splitlines()
        header = f"{name}:"
        start = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith(header)
            ),
            None,
        )
        if start is None:
            raise AssertionError(f"Makefile target {name!r} is missing")
        target_lines = [lines[start]]
        for line in lines[start + 1:]:
            if line and not line.startswith(("\t", " ")):
                break
            target_lines.append(line)
        return "\n".join(target_lines)

    def test_headless_probe_consumes_shared_controller_scene_unchanged(self):
        source = Path("interaction_place_probe.cpp").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("make_headless_target(", source)
        self.assertNotIn("make_headless_destination_surface(", source)

    def test_placement_tail_is_longer_but_pickup_drain_remains_seven(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertRegex(
            controller,
            r"constexpr\s+uint32_t\s+kAutodemoResetPresentationFrames\s*=\s*7U;",
        )
        self.assertRegex(
            controller,
            r"constexpr\s+uint32_t\s+"
            r"kPlacementAutodemoReturnFrames\s*=\s*15U;",
        )
        self.assertRegex(
            controller,
            r"constexpr\s+uint32_t\s+"
            r"kPlacementAutodemoSettledFrames\s*=\s*3U;",
        )
        self.assertIn(
            "kPlacementAutodemoReturnFrames +\n"
            "    kPlacementAutodemoSettledFrames;",
            controller,
        )
        self.assertRegex(
            controller,
            r"constexpr\s+uint32_t\s+"
            r"kPlacementAutodemoMaximumHandoffFrames\s*=\s*75U;",
        )
        self.assertIn(
            "uint32_t post_release_settled_frames_recorded = 0U;",
            controller,
        )
        self.assertIn(
            "!interaction_frame_state.overrides_locomotion_pose",
            controller,
        )
        self.assertRegex(
            controller,
            r"post_release_settled_frames_recorded\s*>=\s*"
            r"kPlacementAutodemoSettledFrames",
        )

        placement_input = self._between(
            controller,
            "        else if (placement_autodemo_enabled)",
            "        // Get if strafe is desired",
        )
        handoff_guard = placement_input.index(
            "if (placement_autodemo_state.handoff_frames_remaining == 0U)"
        )
        for zeroed_input in (
            "gamepadstick_left = vec3();",
            "gamepadstick_right = vec3();",
            "interaction_edges = {};",
            "placement_autodemo_state.left_stick_command = vec3();",
        ):
            with self.subTest(zeroed_input=zeroed_input):
                self.assertLess(
                    placement_input.index(zeroed_input),
                    handoff_guard,
                    "every placement tail tick must remain input-free",
                )

        handoff_recording = self._between(
            controller,
            "                const bool final_locomotion =",
            "                const bool publish_pick_preview_failure =",
        )
        self.assertIn(
            "++placement_autodemo_state.handoff_frames_recorded;",
            handoff_recording,
            "placement's extended tail must remain logged evidence",
        )
        self.assertIn(
            "!interaction_frame_state.overrides_locomotion_pose",
            handoff_recording,
        )
        self.assertIn(
            "++placement_autodemo_state\n"
            "                              .post_release_settled_frames_recorded;",
            handoff_recording,
        )
        self.assertIn(
            "post_release_settled_frames_recorded = 0U;",
            handoff_recording,
            "an overriding frame must reset the consecutive settled streak",
        )
        self.assertIn(
            "placement_presentation_ready ? 0U : 1U;",
            handoff_recording,
            "an unfinished tail must keep the next frame input-suppressed",
        )
        self.assertEqual(
            controller.count("post_release_settled_frames_recorded >="),
            2,
            "both tail readiness and publication must require the streak",
        )

    def test_certification_failure_reports_all_preview_conjuncts(self):
        source = Path("controller.cpp").read_text(encoding="utf-8")
        certification = self._between(
            source,
            "                    if (!preview.accepted ||",
            "                    placement_autodemo_state.current_selection_id =",
        )
        for token in (
            "accepted=",
            "reason=",
            "source_id=",
            "selection_id=",
            "ik_fingerprint=",
            "preview_ik_exact=",
            "candidate_ik_exact=",
        ):
            with self.subTest(token=token):
                self.assertIn(token, certification)

    @classmethod
    def _assert_pick_entry_cpp_source_contract(cls, helper: str) -> None:
        make_slot = cls._between(
            helper,
            "    const auto make_slot =",
            "    const PickEntrySlots slots",
        )
        radial = cls._between(
            make_slot,
            "        const vec3 radial =",
            "        const vec3 position(",
        )
        normalized_radial = " ".join(radial.split())
        expected_radial = (
            "const vec3 radial = quat_mul_vec3( quat_from_angle_axis( "
            "sign * arc, vec3(0.0F, 1.0F, 0.0F)), reach_radius);"
        )
        if normalized_radial != expected_radial:
            raise AssertionError(
                "make_slot must rotate by sign * arc around positive world Y"
            )
        geometry = cls._between(
            make_slot,
            "        const vec3 radial =",
            "        const float hand_score =",
        ) + cls._between(
            make_slot,
            "        return PickEntrySlot{",
            "            hand_score};",
        )
        if "hand" in geometry.lower():
            raise AssertionError(
                "make_slot geometry must be independent of the active hand"
            )

    @staticmethod
    def _pick_entry_fixture() -> dict:
        object_yaw = 0.61
        reach_yaw = -0.37
        object_position = [1.20, 0.79, -0.60]
        radius = 0.405
        radial_angle = -0.47
        return {
            "reach_position": [
                object_position[0] + radius * math.sin(radial_angle),
                0.11,
                object_position[2] + radius * math.cos(radial_angle),
            ],
            "reach_rotation": [
                math.cos(0.5 * reach_yaw),
                0.0,
                math.sin(0.5 * reach_yaw),
                0.0,
            ],
            "object_position": object_position,
            "object_rotation": [
                math.cos(0.5 * object_yaw),
                0.0,
                math.sin(0.5 * object_yaw),
                0.0,
            ],
            "object_dimensions": [0.20, 0.16, 0.28],
            "approach_direction_object": [0.0, 0.0, 1.0],
            "clearance_radius_m": 0.025,
            "hand_side": 1,
        }

    @staticmethod
    def _construct_pick_entry_slots(fixture: dict) -> dict:
        vectors = (
            fixture["reach_position"],
            fixture["reach_rotation"],
            fixture["object_position"],
            fixture["object_rotation"],
            fixture["object_dimensions"],
            fixture["approach_direction_object"],
        )
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("non-finite transform or geometry")
        if not math.isfinite(fixture["clearance_radius_m"]):
            raise ValueError("non-finite clearance")
        for name in ("reach_rotation", "object_rotation"):
            norm = math.sqrt(sum(value * value for value in fixture[name]))
            if abs(norm - 1.0) > 1.0e-3:
                raise ValueError("invalid quaternion norm")
        if (
            any(value <= 0.0 for value in fixture["object_dimensions"])
            or fixture["clearance_radius_m"] < 0.0
            or fixture["hand_side"] not in (-1, 1)
        ):
            raise ValueError("invalid authored geometry")
        approach_object = fixture["approach_direction_object"]
        approach_length = math.sqrt(sum(value * value for value in approach_object))
        if (
            abs(approach_length - 1.0) > 2.0e-5
            or abs(approach_object[1]) > 2.0e-5
        ):
            raise ValueError("invalid approach")
        approach = pickup._quaternion_rotate(
            fixture["object_rotation"], approach_object
        )
        planar_length = math.hypot(approach[0], approach[2])
        if planar_length <= 1.0e-5:
            raise ValueError("no planar approach")
        approach = [approach[0] / planar_length, 0.0, approach[2] / planar_length]
        right = [approach[2], 0.0, -approach[0]]
        lateral_object = pickup._quaternion_rotate(
            pickup._quaternion_inverse(fixture["object_rotation"]), right
        )
        support = 0.5 * sum(
            abs(axis) * dimension
            for axis, dimension in zip(
                lateral_object, fixture["object_dimensions"]
            )
        )
        chord = (
            support
            + fixture["clearance_radius_m"]
            + INTERACTION_CLEARANCE_EPSILON_M
        )
        reach = fixture["reach_position"]
        object_position = fixture["object_position"]
        radial = [
            reach[0] - object_position[0],
            0.0,
            reach[2] - object_position[2],
        ]
        standoff = math.hypot(radial[0], radial[2])
        if (
            not math.isfinite(chord)
            or not math.isfinite(standoff)
            or not STANDOFF_MIN_M <= standoff <= STANDOFF_MAX_M
            or chord <= 0.0
            or chord > REACH_WAYPOINT_POSITION_ERROR_LIMIT_M
            or chord > 2.0 * standoff
        ):
            raise ValueError("invalid chord or standoff")
        arc = 2.0 * math.asin(chord / (2.0 * standoff))
        hand_lateral = [fixture["hand_side"] * value for value in right]
        slots = []
        for identity, sign in (("Plus", 1.0), ("Minus", -1.0)):
            cosine = math.cos(sign * arc)
            sine = math.sin(sign * arc)
            position = [
                object_position[0] + cosine * radial[0] + sine * radial[2],
                reach[1],
                object_position[2] - sine * radial[0] + cosine * radial[2],
            ]
            score = sum(
                (value - origin) * axis
                for value, origin, axis in zip(position, reach, hand_lateral)
            )
            candidate_radius = _planar_distance(position, object_position)
            candidate_chord = _planar_distance(position, reach)
            if (
                any(not math.isfinite(value) for value in position)
                or not math.isfinite(score)
                or abs(candidate_radius - standoff) > 2.0e-5
                or abs(candidate_chord - chord) > 2.0e-5
                or not STANDOFF_MIN_M <= candidate_radius <= STANDOFF_MAX_M
                or position[1] != reach[1]
            ):
                raise ValueError("derived slot invariant failed")
            slots.append(
                {
                    "identity": identity,
                    "position": position,
                    "rotation": fixture["reach_rotation"].copy(),
                    "yaw": _yaw_radians(fixture["reach_rotation"]),
                    "prospective_root": {
                        "world_x": position[0],
                        "world_z": position[2],
                        "world_yaw_radians": _yaw_radians(
                            fixture["reach_rotation"]
                        ),
                    },
                    "hand_score": score,
                }
            )
        return {"ordered": slots, "chord": chord, "standoff": standoff}

    @staticmethod
    def _pick_entry_preview(
        *, path_feasible: bool, match_ready: bool, label: str
    ) -> dict:
        return {
            "path_feasible": path_feasible,
            "match_ready": match_ready,
            "label": label,
            "prospective_root": {
                "world_x": 1.0 if label.startswith("Plus") else -1.0,
                "world_z": 2.0,
                "world_yaw_radians": 0.25,
            },
        }

    @staticmethod
    def _choose_pick_entry_slot(
        hand_scores: tuple[float, float],
        previews: tuple[dict, dict],
        active_hand: str,
    ) -> int | None:
        eligible = [
            index
            for index, preview in enumerate(previews)
            if preview["path_feasible"] and preview["match_ready"]
        ]
        if not eligible:
            return None
        if len(eligible) == 1:
            return eligible[0]
        if hand_scores[0] > hand_scores[1]:
            return 0
        if hand_scores[1] > hand_scores[0]:
            return 1
        return 0 if active_hand == "Right" else 1

    @classmethod
    def _run_pick_entry_policy(
        cls,
        epochs: list[dict],
        *,
        hand_scores: tuple[float, float] = (2.0, 1.0),
        active_hand: str = "Right",
        extra_ticks: int = 3,
    ) -> dict:
        state = {
            "pick_preview_epoch": -1,
            "pick_preview_consumed_epoch": -1,
            "pick_preview_epoch_count": 0,
            "pick_preview_elapsed_ticks": 0,
            "stored_pair": None,
            "frozen": None,
            "pending_failure": None,
            "failure_rows": 0,
            "failure_visible_tick": None,
            "interact_submissions": 0,
            "provider_error": None,
            "preview_mutation_count": 0,
        }
        preview_calls = []
        provider_returns = []
        timeline = []
        next_epoch = 0
        tick_count = len(epochs) + extra_ticks + 2
        for tick in range(tick_count):
            row = {
                "tick": tick,
                "consumed_epoch": None,
                "stick_zero": True,
                "navigated": False,
                "interact": False,
            }

            # Input/steering can consume only the complete prior callback epoch.
            if (
                state["frozen"] is None
                and state["pending_failure"] is None
                and state["pick_preview_epoch"]
                > state["pick_preview_consumed_epoch"]
            ):
                state["pick_preview_consumed_epoch"] = state[
                    "pick_preview_epoch"
                ]
                row["consumed_epoch"] = state["pick_preview_consumed_epoch"]
                pair = state["stored_pair"]
                path_exists = any(
                    preview["path_feasible"] for preview in pair["previews"]
                )
                if not path_exists:
                    state["pending_failure"] = "NoPath"
                else:
                    selected = cls._choose_pick_entry_slot(
                        hand_scores,
                        tuple(pair["previews"]),
                        active_hand,
                    )
                    # Eligibility wins at the exact bound.
                    if selected is not None:
                        state["frozen"] = {
                            "index": selected,
                            "identity": ("Plus", "Minus")[selected],
                            "waypoint": copy.deepcopy(
                                pair["waypoints"][selected]
                            ),
                            "prospective_root": copy.deepcopy(
                                pair["previews"][selected]["prospective_root"]
                            ),
                            # Selection happens before this native scheduler
                            # update; evidence observes the incremented tick.
                            "freeze_tick": tick + 1,
                        }
                    elif state["pick_preview_elapsed_ticks"] >= MAX_WALK_TICKS:
                        state["pending_failure"] = "Deadline"

            if state["pending_failure"] is not None:
                if state["failure_rows"] == 0:
                    # The one row is constructed as terminal before visibility.
                    state["failure_rows"] = 1
                    state["failure_visible_tick"] = tick + 1
            elif state["frozen"] is not None:
                row["stick_zero"] = False
                row["navigated"] = True
                if state["interact_submissions"] == 0:
                    state["interact_submissions"] = 1
                    row["interact"] = True

            # The ordinary provider callback atomically commits Plus then Minus.
            if (
                state["frozen"] is None
                and state["pending_failure"] is None
                and next_epoch < len(epochs)
            ):
                epoch = epochs[next_epoch]
                pair = {
                    "fingerprint": epoch["fingerprint"],
                    "previews": copy.deepcopy(epoch["previews"]),
                    "waypoints": copy.deepcopy(epoch["waypoints"]),
                }
                for index, identity in enumerate(("Plus", "Minus")):
                    preview_calls.append(
                        (tick, identity, epoch["fingerprint"])
                    )
                    if epoch.get("mutates_on_call") == index:
                        state["preview_mutation_count"] += 1
                        state["provider_error"] = "mutation"
                        break
                    if epoch.get("throws_on_call") == index:
                        state["provider_error"] = "throw"
                        break
                if state["provider_error"] is not None:
                    timeline.append(row)
                    break
                state["stored_pair"] = pair
                state["pick_preview_epoch"] = state[
                    "pick_preview_epoch_count"
                ]
                state["pick_preview_epoch_count"] += 1
                state["pick_preview_elapsed_ticks"] += 1
                provider_returns.append((tick, epoch["fingerprint"]))
                next_epoch += 1
            timeline.append(row)
        return {
            "state": state,
            "preview_calls": preview_calls,
            "provider_returns": provider_returns,
            "timeline": timeline,
            "epochs_consumed_from_script": next_epoch,
        }

    @staticmethod
    def _preview_epoch(
        fingerprint: int,
        plus: dict,
        minus: dict,
    ) -> dict:
        return {
            "fingerprint": fingerprint,
            "previews": [plus, minus],
            "waypoints": [
                {"identity": "Plus", "position": [1.0, 0.0, 2.0]},
                {"identity": "Minus", "position": [-1.0, 0.0, 2.0]},
            ],
        }

    def test_pick_entry_selector_filters_before_hand_score_and_ties(self):
        planner = Path("interaction_pick_approach.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "std::optional<size_t> choose_pick_entry_slot(",
            planner,
        )
        helper = self._between(
            planner,
            "std::optional<size_t> choose_pick_entry_slot(",
            "}  // namespace interaction",
        )
        normalized_helper = " ".join(helper.split())
        required_in_order = (
            "permitted[i] && previews[i].path_feasible && "
            "previews[i].match_ready",
            "if (count == 0U)",
            "if (count == 1U)",
            "slots.ordered[0].hand_score > slots.ordered[1].hand_score",
            "slots.ordered[1].hand_score > slots.ordered[0].hand_score",
            "active_hand == Hand::Right ? 0U : 1U",
        )
        positions = [normalized_helper.index(text) for text in required_in_order]
        self.assertEqual(positions, sorted(positions))
        for forbidden in (
            "total_cost",
            "entry_frame",
            "contact_frame",
            "path_reason",
            "match_reason",
            "world_x",
            "world_z",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, helper)

        for plus_path in (False, True):
            for plus_ready in (False, True):
                for minus_path in (False, True):
                    for minus_ready in (False, True):
                        previews = (
                            self._pick_entry_preview(
                                path_feasible=plus_path,
                                match_ready=plus_ready,
                                label="Plus",
                            ),
                            self._pick_entry_preview(
                                path_feasible=minus_path,
                                match_ready=minus_ready,
                                label="Minus",
                            ),
                        )
                        eligible = [
                            index
                            for index, preview in enumerate(previews)
                            if preview["path_feasible"]
                            and preview["match_ready"]
                        ]
                        expected = (
                            None
                            if not eligible
                            else eligible[0]
                            if len(eligible) == 1
                            else 0
                        )
                        with self.subTest(previews=previews):
                            self.assertEqual(
                                self._choose_pick_entry_slot(
                                    (9.0, -50.0), previews, "Right"
                                ),
                                expected,
                            )
        both_ready = (
            self._pick_entry_preview(
                path_feasible=True, match_ready=True, label="Plus"
            ),
            self._pick_entry_preview(
                path_feasible=True, match_ready=True, label="Minus"
            ),
        )
        self.assertEqual(
            self._choose_pick_entry_slot((1.0, 2.0), both_ready, "Right"), 1
        )
        tied = (_float32(0.25), _float32(0.25))
        self.assertEqual(struct.pack("<f", tied[0]), struct.pack("<f", tied[1]))
        self.assertEqual(
            self._choose_pick_entry_slot(tied, both_ready, "Right"), 0
        )
        self.assertEqual(
            self._choose_pick_entry_slot(tied, both_ready, "Left"), 1
        )

    def test_legacy_placement_preview_uses_one_live_snapshot_in_fixed_order(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn("// Placement pickup preview provider begins.", controller)
        self.assertIn("// Placement pickup preview provider ends.", controller)
        provider = self._between(
            controller,
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        )
        helper = self._lambda_assignment(
            controller,
            "    auto preview_pick_entry_slots =",
        )
        normalized_helper = " ".join(helper.split())
        for required in (
            "std::array<interaction::PickEntryPreview, 2> previews{}",
            "for (size_t index = 0; index < previews.size(); ++index)",
            "previews[index] = interaction_runtime.preview_pick( snapshot, "
            "slots.ordered[index].prospective_root, target, affordance_id);",
            "return previews;",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_helper)
        self.assertEqual(
            helper.count("interaction_runtime.preview_pick("),
            1,
            "the legacy indexed call must execute exactly once for each of the "
            "ordered Plus-then-Minus slots",
        )
        normalized_provider = " ".join(provider.split())
        self.assertEqual(
            provider.count("preview_pick_entry_slots("),
            1,
            "legacy placement must make one fixed-two helper invocation per "
            "preview epoch; the removed manual path must not leave a dead call",
        )
        required_in_order = (
            "const interaction::LocomotionSnapshot& snapshot = live_flat_snapshot",
            "const uint64_t fingerprint = live_flat_snapshot_fingerprint",
            "capture_placement_pick_preview_observation(",
            "previews = preview_pick_entry_slots( snapshot,",
            "pick_entry_slots,",
            "pickup preview mutated protected state",
            "pick_entry_previews = previews",
            "pick_preview_snapshot_fingerprint = fingerprint",
            "runtime_pick_preview_calls += 2U",
            "++placement_autodemo_state.pick_preview_elapsed_ticks",
        )
        positions = [normalized_provider.index(text) for text in required_in_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("catch (...)", provider)
        self.assertLess(
            provider.index("catch (...)"),
            provider.index("pick_entry_previews = previews"),
        )
        self.assertNotIn("locomotion_snapshot_fingerprint(", provider)

        poor = self._pick_entry_preview(
            path_feasible=True, match_ready=False, label="Plus PoorMatch"
        )
        blocked = self._pick_entry_preview(
            path_feasible=False, match_ready=False, label="Minus BlockedPath"
        )
        ready = self._pick_entry_preview(
            path_feasible=True, match_ready=True, label="Plus ready"
        )
        result = self._run_pick_entry_policy(
            [
                self._preview_epoch(0xA1, poor, blocked),
                self._preview_epoch(0xB2, ready, blocked),
            ]
        )
        self.assertEqual(
            result["preview_calls"],
            [
                (0, "Plus", 0xA1),
                (0, "Minus", 0xA1),
                (1, "Plus", 0xB2),
                (1, "Minus", 0xB2),
            ],
        )
        first_consume = result["timeline"][1]
        self.assertEqual(first_consume["consumed_epoch"], 0)
        self.assertTrue(first_consume["stick_zero"])
        self.assertFalse(first_consume["navigated"])
        self.assertFalse(first_consume["interact"])
        self.assertEqual(result["state"]["frozen"]["identity"], "Plus")

        for failure_key, expected_error, expected_mutations in (
            ("mutates_on_call", "mutation", 1),
            ("throws_on_call", "throw", 0),
        ):
            for failed_call in (0, 1):
                failing_epoch = self._preview_epoch(0xC3, ready, blocked)
                failing_epoch[failure_key] = failed_call
                failed = self._run_pick_entry_policy([failing_epoch])
                with self.subTest(
                    failure_key=failure_key, failed_call=failed_call
                ):
                    self.assertEqual(
                        failed["state"]["provider_error"], expected_error
                    )
                    self.assertEqual(
                        failed["state"]["preview_mutation_count"],
                        expected_mutations,
                    )
                    self.assertEqual(failed["state"]["pick_preview_epoch"], -1)
                    self.assertEqual(
                        failed["state"]["pick_preview_epoch_count"], 0
                    )
                    self.assertEqual(
                        failed["state"]["pick_preview_elapsed_ticks"], 0
                    )
                    self.assertIsNone(failed["state"]["stored_pair"])
                    self.assertEqual(failed["provider_returns"], [])
                    self.assertEqual(failed["epochs_consumed_from_script"], 0)
                    self.assertTrue(failed["timeline"][-1]["stick_zero"])
                    self.assertFalse(failed["timeline"][-1]["interact"])

    def test_shared_smart_pickup_coordinator_invokes_frozen_preview_once(self):
        source_path = Path("interaction_smart_pickup_controller.cpp")
        if not source_path.is_file():
            self.fail(
                "shared Smart Pickup coordinator source is missing its "
                "frozen-preview helper"
            )
        source = source_path.read_text(encoding="utf-8")
        helper = self._frozen_preview_helper(source)
        signature = helper[:helper.index("{")]
        callback_parameter = re.search(
            r"(?:const\s+)?(?:interaction::)?SmartPickupPreviewCallback"
            r"(?:\s+const)?\s*&\s*([A-Za-z_]\w*)",
            signature,
        )
        self.assertIsNotNone(
            callback_parameter,
            "the shared frozen-preview helper must accept the native preview "
            "callback by const reference",
        )
        self._assert_single_root_preview_delegation(
            helper,
            rf"\b{re.escape(callback_parameter.group(1))}\s*\(",
        )

    def test_native_smart_pickup_preview_callback_is_one_direct_delegation(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        callback_name = "preview_manual_smart_pickup"
        if callback_name not in controller:
            self.fail(
                "controller must supply the shared coordinator a narrow "
                "preview_manual_smart_pickup runtime callback"
            )
        callback = self._named_braced_scope(controller, callback_name)
        self._assert_single_root_preview_delegation(
            callback,
            r"interaction_runtime\.preview_pick\s*\(",
        )
        observation = self._between(
            controller,
            "// Manual pick-assist observation begins.",
            "// Manual pick-assist observation ends.",
        )
        post_arguments = self._call_arguments(
            observation,
            "manual_smart_pickup_controller.post_step",
        )
        self.assertEqual(
            post_arguments.count(callback_name),
            1,
            "the native preview callback must be passed exactly once to the "
            "manual Smart Pickup post-step, not left as a dead delegate",
        )

    def test_pick_entry_preview_runs_inside_provider_and_consumes_prior_epoch(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn("// Placement pickup preview consume begins.", controller)
        for marker in (
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, controller)
        scripted = self._between(
            controller,
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
        )
        provider = self._between(
            controller,
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        )
        scheduler = self._between(
            controller,
            "const interaction::RuntimeOutput& interaction_output =",
            "const uint64_t placement_preview_call_delta =",
        )
        self.assertIn(
            "pick_preview_epoch >\n"
            "                        placement_autodemo_state\n"
            "                            .pick_preview_consumed_epoch",
            scripted,
        )
        self.assertIn("pick_preview_consumed_epoch =", scripted)
        self.assertIn("interaction_scheduler.tick(", scheduler)
        self.assertIn(provider, scheduler)
        self.assertLess(
            controller.index("// Placement pickup preview consume begins."),
            controller.index("interaction_scheduler.tick("),
        )
        self.assertLess(
            provider.index("pick_entry_previews = previews"),
            provider.index("return snapshot;"),
        )
        self.assertNotIn("LocomotionSnapshot*", controller)
        self.assertNotIn("LocomotionSnapshot& pick_preview", controller)

        ready = self._pick_entry_preview(
            path_feasible=True, match_ready=True, label="Plus ready"
        )
        blocked = self._pick_entry_preview(
            path_feasible=False, match_ready=False, label="Minus blocked"
        )
        result = self._run_pick_entry_policy(
            [self._preview_epoch(0x1234, ready, blocked)]
        )
        self.assertIsNone(result["timeline"][0]["consumed_epoch"])
        self.assertEqual(result["timeline"][1]["consumed_epoch"], 0)
        self.assertEqual(result["provider_returns"], [(0, 0x1234)])
        self.assertEqual(
            [call[2] for call in result["preview_calls"]],
            [0x1234, 0x1234],
        )

    def test_pick_entry_freeze_tick_matches_post_scheduler_evidence_tick(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        consume_begin = controller.index(
            "// Placement pickup preview consume begins."
        )
        scheduler_tick = controller.index(
            "interaction_scheduler.tick(", consume_begin
        )
        runtime_increment = controller.index(
            "++autodemo_state.runtime_tick;", scheduler_tick
        )
        evidence_write = controller.index(
            "write_placement_autodemo_record(", runtime_increment
        )
        self.assertLess(consume_begin, scheduler_tick)
        self.assertLess(scheduler_tick, runtime_increment)
        self.assertLess(runtime_increment, evidence_write)

        consume = self._between(
            controller,
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
        )
        normalized_consume = " ".join(consume.split())
        current_assignment = (
            "pick_entry_freeze_tick = autodemo_state.runtime_tick;"
        )
        next_assignment = (
            "pick_entry_freeze_tick = autodemo_state.runtime_tick + 1U;"
        )
        self.assertIn(next_assignment, normalized_consume)
        self.assertNotIn(current_assignment, normalized_consume)

        pre_scheduler_runtime_tick = 41
        recorded_freeze_tick = pre_scheduler_runtime_tick + (
            1 if next_assignment in normalized_consume else 0
        )
        post_scheduler_evidence_tick = pre_scheduler_runtime_tick + 1
        self.assertEqual(
            recorded_freeze_tick,
            post_scheduler_evidence_tick,
            "selection consumes the prior epoch before the scheduler, but its "
            "first evidence row is written after the one-tick scheduler "
            "advance",
        )

    def test_pick_entry_preview_deadline_advances_while_walk_ticks_pause(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn("// Placement pickup preview consume begins.", controller)
        state = self._between(
            controller,
            "struct ControllerPlacementAutodemoState",
            "float autodemo_planar_distance",
        )
        for marker in (
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, controller)
        consume = self._between(
            controller,
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
        )
        provider = self._between(
            controller,
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        )
        normalized_consume = " ".join(consume.split())
        self.assertIn("uint32_t pick_preview_elapsed_ticks = 0U;", state)
        self.assertIn(
            "++placement_autodemo_state.pick_preview_elapsed_ticks", provider
        )
        self.assertIn(
            ".pick_preview_elapsed_ticks >= "
            "kPlacementAutodemoMaximumWalkTicks",
            normalized_consume,
        )
        self.assertLess(
            normalized_consume.index("interaction::choose_pick_entry_slot("),
            normalized_consume.index("kPlacementAutodemoMaximumWalkTicks"),
        )
        self.assertNotIn("walk_ticks", consume)
        self.assertNotIn("dt", provider)
        self.assertEqual(controller.count("interaction_scheduler.tick("), 1)

        not_ready = self._pick_entry_preview(
            path_feasible=True, match_ready=False, label="Plus PoorMatch"
        )
        blocked = self._pick_entry_preview(
            path_feasible=False, match_ready=False, label="Minus blocked"
        )
        ready = self._pick_entry_preview(
            path_feasible=True, match_ready=True, label="Plus ready"
        )
        deferred = [
            self._preview_epoch(index + 1, not_ready, blocked)
            for index in range(MAX_WALK_TICKS - 1)
        ]
        ready_at_bound = self._run_pick_entry_policy(
            deferred + [self._preview_epoch(MAX_WALK_TICKS, ready, blocked)]
        )
        self.assertEqual(
            ready_at_bound["state"]["pick_preview_elapsed_ticks"],
            MAX_WALK_TICKS,
        )
        self.assertEqual(ready_at_bound["state"]["frozen"]["identity"], "Plus")
        self.assertIsNone(ready_at_bound["state"]["pending_failure"])

        deadline = self._run_pick_entry_policy(
            deferred
            + [self._preview_epoch(MAX_WALK_TICKS, not_ready, blocked)]
        )
        self.assertEqual(deadline["state"]["pending_failure"], "Deadline")
        self.assertEqual(deadline["state"]["failure_rows"], 1)
        deadline_consume = next(
            row
            for row in deadline["timeline"]
            if row["consumed_epoch"] == MAX_WALK_TICKS - 1
        )
        self.assertTrue(deadline_consume["stick_zero"])
        self.assertFalse(deadline_consume["interact"])
        self.assertEqual(
            deadline["state"]["failure_visible_tick"],
            deadline_consume["tick"] + 1,
        )

    def test_pick_entry_selection_defers_freezes_once_and_never_retries(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn("// Placement pickup preview consume begins.", controller)
        state = self._between(
            controller,
            "struct ControllerPlacementAutodemoState",
            "float autodemo_planar_distance",
        )
        for required in (
            "std::optional<size_t> frozen_pick_entry_index{};",
            "std::optional<uint64_t> pick_entry_freeze_tick{};",
            "uint64_t pick_interact_submission_count = 0U;",
            "PlacementPickPreviewFailure pending_pick_preview_failure =",
            "bool pick_preview_failure_recorded = false;",
        ):
            with self.subTest(required=required):
                self.assertIn(required, state)
        for marker in (
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, controller)
        consume = self._between(
            controller,
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
        )
        provider = self._between(
            controller,
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        )
        normalized_provider = " ".join(provider.split())
        terminal_recording = self._between(
            controller,
            "const bool publish_pick_preview_failure =",
            "placement_autodemo_state.previous_displayed_root =",
        )
        normalized_terminal = " ".join(terminal_recording.split())
        self.assertIn("frozen_pick_entry_index = selected", consume)
        self.assertIn(
            "pick_entry_freeze_tick =\n"
            "                                autodemo_state.runtime_tick + 1U",
            consume,
        )
        self.assertIn("pending_pick_preview_failure", normalized_provider)
        self.assertIn(
            "!placement_autodemo_state .frozen_pick_entry_index.has_value()",
            normalized_provider,
        )
        self.assertIn("pick_interact_submission_count == 0U", consume)
        self.assertIn("++placement_autodemo_state.pick_interact_submission_count", consume)
        self.assertNotIn("retry", consume.lower())
        self.assertIn("const float interaction_position_error_m =", consume)
        self.assertLess(
            consume.index(".interaction_waypoint ="),
            consume.index("const float interaction_position_error_m ="),
            "the freeze tick must measure the newly selected waypoint, not the default",
        )
        terminal_steps = (
            "terminal_record_state = placement_autodemo_state",
            "terminal_record_state->pick_preview_failure_recorded = true",
            "write_placement_autodemo_record(",
            "autodemo_state.log.flush()",
            "if (!autodemo_state.log)",
            "placement_autodemo_state .pick_preview_failure_recorded = true",
        )
        terminal_positions = [
            normalized_terminal.index(step) for step in terminal_steps
        ]
        self.assertEqual(terminal_positions, sorted(terminal_positions))
        self.assertIn("record_state,", terminal_recording)

        poor = self._pick_entry_preview(
            path_feasible=True, match_ready=False, label="Plus PoorMatch"
        )
        blocked = self._pick_entry_preview(
            path_feasible=False, match_ready=False, label="Minus BlockedPath"
        )
        plus_ready = self._pick_entry_preview(
            path_feasible=True, match_ready=True, label="Plus ready"
        )
        minus_ready = self._pick_entry_preview(
            path_feasible=True, match_ready=True, label="Minus better later"
        )
        result = self._run_pick_entry_policy(
            [
                self._preview_epoch(1, poor, blocked),
                self._preview_epoch(2, plus_ready, blocked),
                self._preview_epoch(3, plus_ready, minus_ready),
            ],
            hand_scores=(1.0, 100.0),
            extra_ticks=8,
        )
        frozen = copy.deepcopy(result["state"]["frozen"])
        self.assertEqual(frozen["identity"], "Plus")
        self.assertEqual(frozen["index"], 0)
        self.assertEqual(frozen["freeze_tick"], 3)
        self.assertEqual(frozen["waypoint"]["identity"], "Plus")
        self.assertEqual(frozen["prospective_root"], plus_ready["prospective_root"])
        self.assertEqual(result["state"]["interact_submissions"], 1)
        self.assertEqual(result["epochs_consumed_from_script"], 2)
        self.assertEqual(len(result["preview_calls"]), 4)
        self.assertEqual(result["state"]["frozen"], frozen)

        no_path = self._run_pick_entry_policy(
            [self._preview_epoch(9, blocked, blocked)], extra_ticks=4
        )
        self.assertEqual(no_path["state"]["pending_failure"], "NoPath")
        self.assertEqual(no_path["state"]["failure_rows"], 1)
        self.assertEqual(len(no_path["preview_calls"]), 2)
        self.assertTrue(
            all(
                row["stick_zero"] and not row["interact"]
                for row in no_path["timeline"]
            )
        )
        no_path_consume = next(
            row for row in no_path["timeline"] if row["consumed_epoch"] == 0
        )
        self.assertEqual(
            no_path["state"]["failure_visible_tick"],
            no_path_consume["tick"] + 1,
        )

    def test_pick_entry_selection_scope_is_runtime_owned_and_live_only(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn("// Placement pickup preview consume begins.", controller)
        for marker in (
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, controller)
        provider = self._between(
            controller,
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
        )
        consume = self._between(
            controller,
            "// Placement pickup preview consume begins.",
            "// Placement pickup preview consume ends.",
        )
        normalized_provider = " ".join(provider.split())
        self.assertIn("preview_pick_entry_slots(", provider)
        self.assertIn(
            "live_flat_snapshot_fingerprint",
            normalized_provider,
        )
        preview_helper = self._lambda_assignment(
            controller,
            "    auto preview_pick_entry_slots =",
        )
        self.assertIn("interaction_runtime.preview_pick(", preview_helper)
        self.assertIn("capture_placement_pick_preview_observation(", provider)
        self.assertIn("return snapshot;", provider)
        for forbidden in (
            "QueryInput",
            "MatchInput",
            "InteractionRequest",
            "canonical",
            "Features",
            "MatchConfig",
            "select_whole_clip",
            "interaction_runtime.update(",
            "interaction_registry.reserve(",
            "interaction_registry.attach(",
            "interaction_registry.hold(",
            "interaction_registry.release(",
            "interaction_registry.reset(",
            "interaction_surface_registry.upsert(",
            "simulation_position =",
            "simulation_rotation =",
            "bone_positions(0) =",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, provider + consume)
        self.assertEqual(controller.count("interaction_runtime.update("), 1)
        self.assertEqual(
            preview_helper.count("interaction_runtime.preview_pick("), 1
        )
        reservation_observation = self._between(
            controller,
            "const interaction::InteractionTarget* observed_pick_target =",
            "const bool pickup_candidate_state =",
        )
        for required in (
            "runtime_state == interaction::RuntimeState::Align",
            "observed_pick_target->state == interaction::ObjectState::Targeted",
            "observed_pick_target->owner_request + 1U ==\n"
            "                        interaction_next_request_id",
            "pick_reservation_transition_count == 0U",
            "++placement_autodemo_state\n"
            "                          .pick_reservation_transition_count",
        ):
            with self.subTest(reservation_requirement=required):
                self.assertIn(required, reservation_observation)
        self.assertNotIn(
            "runtime_state == interaction::RuntimeState::Preflight",
            reservation_observation,
        )

    def test_pick_entry_constructs_both_ordered_exact_arc_slots(self):
        planner_header = Path("interaction_pick_approach.h").read_text(
            encoding="utf-8"
        )
        planner = Path("interaction_pick_approach.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "enum class PickEntrySlotIdentity : uint8_t { Plus, Minus };",
            planner_header,
        )
        helper = self._between(
            planner,
            "PickEntrySlots make_pick_entry_slots(",
            "std::optional<size_t> choose_pick_entry_slot(",
        )
        for required in (
            "std::array<PickEntrySlot, 2>",
            "PickEntrySlotIdentity::Plus, +1.0F",
            "PickEntrySlotIdentity::Minus, -1.0F",
            "stable_reach_waypoint.position.y",
            "stable_reach_waypoint.rotation",
            "yaw_radians(stable_reach_waypoint.rotation)",
            "clearance_chord_m",
            "standoff_m",
        ):
            with self.subTest(required=required):
                self.assertIn(required, helper)

        self._assert_pick_entry_cpp_source_contract(helper)
        exact_rotation = "sign * arc, vec3(0.0F, 1.0F, 0.0F)"
        mutations = {
            "reversed_arc_sign":
                "-sign * arc, vec3(0.0F, 1.0F, 0.0F)",
            "negative_world_y_axis":
                "sign * arc, vec3(0.0F, -1.0F, 0.0F)",
            "hand_dependent_arc":
                "hand_sign * sign * arc, vec3(0.0F, 1.0F, 0.0F)",
        }
        for name, replacement in mutations.items():
            mutated = helper.replace(exact_rotation, replacement, 1)
            self.assertNotEqual(mutated, helper)
            with self.subTest(mutation=name), self.assertRaises(AssertionError):
                self._assert_pick_entry_cpp_source_contract(mutated)

        fixture = self._pick_entry_fixture()
        result = self._construct_pick_entry_slots(fixture)
        self.assertEqual(
            [slot["identity"] for slot in result["ordered"]],
            ["Plus", "Minus"],
        )
        reach = fixture["reach_position"]
        origin = fixture["object_position"]
        radius = [reach[0] - origin[0], reach[2] - origin[2]]
        for expected_identity, expected_sign, slot in zip(
            ("Plus", "Minus"), (1.0, -1.0), result["ordered"]
        ):
            self.assertEqual(slot["identity"], expected_identity)
            self.assertAlmostEqual(
                _planar_distance(slot["position"], origin),
                result["standoff"],
                places=12,
            )
            self.assertAlmostEqual(
                _planar_distance(slot["position"], reach),
                result["chord"],
                places=12,
            )
            candidate = [
                slot["position"][0] - origin[0],
                slot["position"][2] - origin[2],
            ]
            signed_angle = math.atan2(
                radius[1] * candidate[0] - radius[0] * candidate[1],
                radius[0] * candidate[0] + radius[1] * candidate[1],
            )
            self.assertGreater(expected_sign * signed_angle, 0.0)
            self.assertEqual(slot["position"][1], reach[1])
            self.assertEqual(slot["rotation"], fixture["reach_rotation"])
            self.assertEqual(slot["yaw"], _yaw_radians(fixture["reach_rotation"]))
            self.assertEqual(
                slot["prospective_root"]["world_x"], slot["position"][0]
            )
            self.assertEqual(
                slot["prospective_root"]["world_z"], slot["position"][2]
            )
            self.assertEqual(
                slot["prospective_root"]["world_yaw_radians"], slot["yaw"]
            )
            self.assertTrue(math.isfinite(slot["hand_score"]))

    def test_pick_entry_slots_are_yaw_equivariant_and_hand_scores_only_rank(self):
        planner = Path("interaction_pick_approach.cpp").read_text(
            encoding="utf-8"
        )
        helper = self._between(
            planner,
            "PickEntrySlots make_pick_entry_slots(",
            "std::optional<size_t> choose_pick_entry_slot(",
        )
        self.assertNotIn("plus_score > minus_score", helper)
        self.assertNotIn("minus_score > plus_score", helper)
        self.assertLess(
            helper.index("PickEntrySlotIdentity::Plus, +1.0F"),
            helper.index("PickEntrySlotIdentity::Minus, -1.0F"),
        )

        fixture = self._pick_entry_fixture()
        original = self._construct_pick_entry_slots(fixture)
        opposite_hand_fixture = copy.deepcopy(fixture)
        opposite_hand_fixture["hand_side"] = -fixture["hand_side"]
        opposite_hand = self._construct_pick_entry_slots(opposite_hand_fixture)
        for same_hand, flipped_hand in zip(
            original["ordered"], opposite_hand["ordered"]
        ):
            self.assertEqual(same_hand["identity"], flipped_hand["identity"])
            self.assertEqual(same_hand["position"], flipped_hand["position"])
            self.assertEqual(same_hand["rotation"], flipped_hand["rotation"])
            self.assertEqual(
                same_hand["prospective_root"],
                flipped_hand["prospective_root"],
            )
            self.assertAlmostEqual(
                same_hand["hand_score"],
                -flipped_hand["hand_score"],
                places=12,
            )
        world_yaw = 0.83
        world_rotation = [
            math.cos(0.5 * world_yaw),
            0.0,
            math.sin(0.5 * world_yaw),
            0.0,
        ]
        translation = [1.70, -0.23, 0.91]
        transformed = copy.deepcopy(fixture)
        for field in ("reach_position", "object_position"):
            rotated = pickup._quaternion_rotate(world_rotation, fixture[field])
            transformed[field] = [
                rotated[index] + translation[index] for index in range(3)
            ]
        for field in ("reach_rotation", "object_rotation"):
            transformed[field] = pickup._quaternion_multiply(
                world_rotation, fixture[field]
            )
        equivariant = self._construct_pick_entry_slots(transformed)
        for before, after in zip(original["ordered"], equivariant["ordered"]):
            self.assertEqual(before["identity"], after["identity"])
            expected_position = pickup._quaternion_rotate(
                world_rotation, before["position"]
            )
            expected_position = [
                expected_position[index] + translation[index]
                for index in range(3)
            ]
            for actual, expected in zip(after["position"], expected_position):
                self.assertAlmostEqual(actual, expected, places=12)
            expected_rotation = pickup._quaternion_multiply(
                world_rotation, before["rotation"]
            )
            self.assertLessEqual(
                pickup._quaternion_sign_distance(
                    after["rotation"], expected_rotation
                ),
                1.0e-12,
            )
            self.assertAlmostEqual(
                after["prospective_root"]["world_x"],
                after["position"][0],
                places=12,
            )
            self.assertAlmostEqual(
                after["prospective_root"]["world_z"],
                after["position"][2],
                places=12,
            )
            self.assertAlmostEqual(
                after["prospective_root"]["world_yaw_radians"],
                _yaw_radians(expected_rotation),
                places=12,
            )

        tie = self._pick_entry_fixture()
        approach = pickup._quaternion_rotate(
            tie["object_rotation"], tie["approach_direction_object"]
        )
        right = [approach[2], 0.0, -approach[0]]
        distance = 0.405
        tie["reach_position"] = [
            tie["object_position"][0] + distance * right[0],
            tie["reach_position"][1],
            tie["object_position"][2] + distance * right[2],
        ]
        tied = self._construct_pick_entry_slots(tie)
        self.assertAlmostEqual(
            tied["ordered"][0]["hand_score"],
            tied["ordered"][1]["hand_score"],
            places=12,
        )
        self.assertEqual(
            [slot["identity"] for slot in tied["ordered"]],
            ["Plus", "Minus"],
        )

    def test_pick_entry_slot_construction_fails_closed_for_each_invalid_input(self):
        planner = Path("interaction_pick_approach.cpp").read_text(
            encoding="utf-8"
        )
        helper = self._between(
            planner,
            "PickEntrySlots make_pick_entry_slots(",
            "std::optional<size_t> choose_pick_entry_slot(",
        )
        for required in (
            "clearance_chord_m > 2.0F * standoff_m",
            "kStandoffMinimumM",
            "kStandoffMaximumM",
            "quat_length(target.object_world.rotation)",
            "quat_length(reach_waypoint.rotation)",
            "pick-entry slots arc invariant failed",
        ):
            with self.subTest(required=required):
                self.assertIn(required, helper)
        self.assertNotIn("clampf(", helper)
        self.assertNotIn("std::clamp(", helper)

        invalid = []

        def row(name, mutate):
            fixture = self._pick_entry_fixture()
            mutate(fixture)
            invalid.append((name, fixture))

        row("excessive_chord", lambda fixture: fixture.update(clearance_radius_m=0.30))
        row(
            "chord_greater_than_diameter",
            lambda fixture: fixture.update(clearance_radius_m=0.80),
        )
        nominal = self._construct_pick_entry_slots(self._pick_entry_fixture())
        nominal_clearance = self._pick_entry_fixture()["clearance_radius_m"]
        support = (
            nominal["chord"]
            - nominal_clearance
            - INTERACTION_CLEARANCE_EPSILON_M
        )
        diameter_case_chord = (
            support + 0.80 + INTERACTION_CLEARANCE_EPSILON_M
        )
        self.assertGreater(diameter_case_chord, 2.0 * nominal["standoff"])
        row(
            "zero_standoff",
            lambda fixture: fixture.update(
                reach_position=fixture["object_position"].copy()
            ),
        )
        row(
            "below_standoff_band",
            lambda fixture: fixture.update(
                reach_position=[
                    fixture["object_position"][0] + 0.34,
                    fixture["reach_position"][1],
                    fixture["object_position"][2],
                ]
            ),
        )
        row(
            "above_standoff_band",
            lambda fixture: fixture.update(
                reach_position=[
                    fixture["object_position"][0] + 0.46,
                    fixture["reach_position"][1],
                    fixture["object_position"][2],
                ]
            ),
        )
        row("nonfinite_waypoint", lambda fixture: fixture["reach_position"].__setitem__(0, math.inf))
        row("nonfinite_target", lambda fixture: fixture["object_position"].__setitem__(2, math.nan))
        row("nonfinite_chord", lambda fixture: fixture.update(clearance_radius_m=math.inf))
        row("invalid_waypoint_quaternion", lambda fixture: fixture.update(reach_rotation=[0.5, 0.0, 0.0, 0.0]))
        row("invalid_target_quaternion", lambda fixture: fixture.update(object_rotation=[0.0, 0.0, 0.0, 0.0]))
        row("nonfinite_approach", lambda fixture: fixture["approach_direction_object"].__setitem__(0, math.nan))
        for name, fixture in invalid:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._construct_pick_entry_slots(fixture)

    def test_placement_mode_is_separate_and_remains_native_25_hz(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        for required in (
            "MM_INTERACTION_PLACE_AUTODEMO",
            "enum class AutodemoMode",
            "AutodemoMode::Pickup",
            "AutodemoMode::Placement",
            "const bool pickup_autodemo_enabled",
            "const bool placement_autodemo_enabled",
            "constexpr uint64_t kPlacementAutodemoMaximumEvidenceFrames = 900U;",
            "constexpr uint32_t kPlacementAutodemoMinimumWalkTicks = 25U;",
            "constexpr uint32_t kPlacementAutodemoMinimumCarryTicks = 25U;",
            "SetTargetFPS(25);",
        ):
            with self.subTest(required=required):
                self.assertIn(required, controller)
        self.assertNotIn("SetTargetFPS(60)", controller)
        self.assertNotIn("/ 6", controller)

    def test_reach_helper_returns_only_a_root_transform(self):
        planner = Path("interaction_pick_approach.cpp").read_text(
            encoding="utf-8"
        )
        helper = self._between(
            planner,
            "Transform make_pick_reach_waypoint(",
            "const char* pick_entry_slot_name(",
        )
        self.assertIn("database.positions", helper)
        self.assertIn("database.rotations", helper)
        self.assertIn("g1_skeleton::Simulation", helper)
        self.assertNotIn("interaction::Pose", helper)
        self.assertNotIn("LocomotionSnapshot", helper)
        self.assertNotIn("make_autodemo_canonical_entry", helper)

    def test_pick_entry_slots_are_affordance_derived_and_root_only(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        planner = Path("interaction_pick_approach.cpp").read_text(
            encoding="utf-8"
        )
        helper = self._between(
            planner,
            "PickEntrySlots make_pick_entry_slots(",
            "std::optional<size_t> choose_pick_entry_slot(",
        )
        for required in (
            "approach_direction_object",
            "target.object_world.rotation",
            "target.object_dimensions",
            "quat_inv(target.object_world.rotation)",
            "Hand::Right",
            "Hand::Left",
            "cross(vec3(0.0F, 1.0F, 0.0F), approach_world)",
            "kInteractionClearanceEpsilonM",
            "kReachPositionErrorM",
            "clearance_chord_m",
            "target.object_world.position",
            "std::asin",
            "sign * arc",
            "PickEntrySlotIdentity::Plus, +1.0F",
            "PickEntrySlotIdentity::Minus, -1.0F",
            "PickEntryRoot",
            "yaw_radians(stable_reach_waypoint.rotation)",
            "const vec3 hand_lateral",
            "chord_error_m",
            "standoff_error_m",
            "height_error_m",
            "rotation_copy_error",
        ):
            with self.subTest(required=required):
                self.assertIn(required, helper)
        for forbidden in (
            "interaction::Pose",
            "LocomotionSnapshot",
            "make_autodemo_canonical_entry",
            "preview_pick(",
            "plus_score",
            "minus_score",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, helper)
        near_unit_rotation = [0.9995, 0.0, 0.0, 0.0]
        self.assertLessEqual(
            abs(math.sqrt(sum(value * value for value in near_unit_rotation)) - 1.0),
            1.0e-3,
        )
        self.assertGreater(
            2.0 * math.acos(near_unit_rotation[0] ** 2),
            2.0e-5,
        )
        self.assertNotIn("quat_angle_between(", helper)

        setup = self._between(
            controller,
            "// Placement auto-demo setup begins.",
            "// Placement auto-demo setup ends.",
        )
        scripted = self._between(
            controller,
            "// Placement auto-demo input begins.",
            "// Placement auto-demo input ends.",
        )
        self.assertIn(
            "interaction::make_pick_entry_slots(", setup
        )
        self.assertIn(
            "interaction::PickEntrySlots pick_entry_slots{};", controller
        )
        self.assertIn(".pick_entry_slots =", setup)
        self.assertIn("interaction::choose_pick_entry_slot(", scripted)
        self.assertIn("{true, true}", scripted)
        self.assertIn("frozen_pick_entry_index = selected", scripted)
        self.assertIn(".ordered[*selected].waypoint", scripted)
        for spelling in (
            ".pick_entry_slots.ordered[0]",
            ".pick_entry_slots.ordered.front()",
            ".pick_entry_slots.ordered.at(0)",
            "std::get<0>(placement_autodemo_state.pick_entry_slots.ordered)",
        ):
            with self.subTest(scripted_pair_consumption=spelling):
                self.assertNotIn(spelling, scripted)
        self.assertNotIn("make_placement_autodemo_interaction_waypoint(", controller)
        self.assertNotIn("= 0.10F;", helper)

    def test_placement_setup_and_provider_cannot_use_pickup_canonical_state(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        setup = self._between(
            controller,
            "// Placement auto-demo setup begins.",
            "// Placement auto-demo setup ends.",
        )
        provider = self._between(
            controller,
            "const bool use_autodemo_canonical_snapshot =",
            "// Advance locomotion and interaction synchronously",
        )
        self.assertIn("interaction::make_pick_reach_waypoint(", setup)
        for forbidden in (
            "initialize_autodemo_canonical_world",
            "use_autodemo_canonical_snapshot",
            "autodemo_canonical_entry->snapshot",
            "make_autodemo_canonical_entry",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, setup)
        self.assertIn("pickup_autodemo_enabled &&", provider)
        self.assertNotIn("placement_autodemo_enabled &&", provider)
        self.assertIn(
            "if (pickup_autodemo_enabled)\n"
            "    {\n"
            "        initialize_autodemo_canonical_world();",
            controller,
        )

    def test_placement_input_uses_arrival_sticks_and_live_pose(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        scripted = self._between(
            controller,
            "// Placement auto-demo input begins.",
            "// Placement auto-demo input ends.",
        )
        normalized_scripted = " ".join(scripted.split()).replace(" .", ".")
        for required in (
            "controller_world_navigation_stick(",
            "placement_autodemo_state.reach_waypoint",
            "placement_autodemo_state.reach_entry_point",
            "placement_autodemo_state.reach_entry_reached",
            "placement_autodemo_state.reach_braking",
            "interaction::arrival_navigation_stick(",
            "interaction::arrival_facing_stick(",
            "interaction::arrival_ready(",
            "simulation_velocity",
            "kPlacementAutodemoAlignmentPositionErrorM",
            "kPlacementAutodemoReachYawErrorRadians",
            "controller_place_staging_stick(",
            "kPlacementAutodemoArrivalConfig.minimum_standoff_m",
            "kPlacementAutodemoArrivalConfig.maximum_standoff_m",
            "kPlacementAutodemoSettleTicks",
            "gamepadstick_left =",
            "gamepadstick_right =",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_scripted)
        for forbidden in (
            "simulation_position =",
            "simulation_rotation =",
            "bone_positions(0) =",
            "autodemo_canonical_entry",
            "LocomotionSnapshot",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, scripted)
        self.assertNotIn(
            "kPlacementAutodemoReachBrakingDistanceM", controller
        )

    def test_placement_destination_and_preview_are_runtime_owned(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        resolver = self._between(
            controller,
            "auto resolve_autodemo_place_target =",
            "auto resolve_manual_place_target =",
        )
        for required in (
            "interaction_destination_surface_handle",
            "interaction_destination_affordance_id",
            "interaction_next_request_id++",
        ):
            with self.subTest(required=required):
                self.assertIn(required, resolver)
        for forbidden in (
            "resolve_single_surface(",
            "preview_place_motion(",
            "select_place_motion(",
            "PlaceMatchInput",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, resolver)
        self.assertEqual(controller.count("resolve_single_surface("), 1)
        self.assertEqual(controller.count("interaction_runtime.preview_place("), 1)
        self.assertNotIn("preview_place_motion(", controller)
        self.assertNotIn("PlaceMatchInput", controller)

    def test_placement_evidence_uses_observed_instrumentation(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        writer = self._between(
            controller,
            "void write_placement_autodemo_record(",
            "void validate_autodemo_screenshot(",
        )
        for required in (
            "state.last_provider_was_live_flat",
            "state.canonical_snapshot_provider_calls",
            "state.root_relocation_calls",
            "state.simulation_root_initialization_calls",
            "state.displayed_root_initialization_calls",
            "state.pick_resolver_used_live_flat",
            "state.destination_generation_changed_before_place",
            "state.placement_surface_resolver_calls",
            "state.runtime_preview_calls",
            "state.free_selector_calls",
            "state.external_match_input_calls",
            "state.preview_mutation_count",
            "state.direct_root_write_calls",
            "state.final_support_sweep_clear",
            "state.reach_braking",
            'left_stick_magnitude',
            'reach_waypoint_position_error_m',
            'reach_waypoint_yaw_error_degrees',
            'reach_brake_latched',
            'interaction_waypoint_position',
            'interaction_waypoint_rotation',
            'interaction_lateral_offset_m',
            'interaction_object_position',
            'interaction_object_dimensions',
            'interaction_object_rotation',
            'interaction_approach_direction_object',
            'interaction_clearance_radius_m',
            'interaction_hand_side',
        ):
            with self.subTest(required=required):
                self.assertIn(required, writer)
        for forbidden in (
            'locomotion_provider_kind\\\":\\\"live_flat',
            'canonical_snapshot_used\\\":false',
            'root_relocation_applied\\\":false',
            'placement_surface_resolver_calls\\\":0',
            'preview_mutation_count\\\":0',
            'direct_root_write\\\":false',
            'reach_brake_latched\\\":true',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, writer)
        for required in (
            "placement_preview_calls_before_tick",
            "placement_preview_call_delta",
            "++placement_autodemo_state.runtime_preview_calls",
            "interaction_runtime.diagnostics()",
            "placement auto-demo preview mutated runtime or",
            "placement_autodemo_state.final_support_sweep_clear =",
            ".support_sweep_clear;",
            "handoff_frames_recorded",
            "TakeScreenshot(screenshot_temporary.c_str());",
            "write_placement_autodemo_record(",
        ):
            with self.subTest(observation=required):
                self.assertIn(required, controller)

    def test_runtime_publishes_full_weight_for_owned_place_only(self):
        runtime = Path("interaction_runtime.cpp").read_text(encoding="utf-8")
        test_source = Path("tests/cpp/test_interaction_runtime.cpp").read_text(
            encoding="utf-8"
        )
        for required in (
            "diagnostics_.result != ResultCode::Failed",
            "diagnostics_.attached",
            "state_ == RuntimeState::PlacePreflight",
            "state_ == RuntimeState::PlaceAlign",
            "state_ == RuntimeState::PlaceReplay",
            "diagnostics_.hand_constraint_weight = 1.0F;",
        ):
            with self.subTest(required=required):
                self.assertIn(required, runtime)
        self.assertIn(
            "test_place_hand_constraint_weight_is_full_until_release",
            test_source,
        )

    def test_both_writers_share_the_positive_weight_grasp_publish_guard(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        signature = "bool autodemo_hand_constraint_evidence_valid("
        self.assertIn(
            signature,
            controller,
            "pickup and placement evidence must share one final-FK grasp guard",
        )
        guard = self._between(
            controller,
            signature,
            "\n}\n\n",
        )
        for required in (
            "autodemo_is_finite(capture.hand_constraint_weight)",
            "capture.hand_constraint_weight >= 0.0F",
            "capture.hand_constraint_weight <= 1.0F",
            "capture.hand_constraint_weight == 0.0F",
            "capture.hand_constraint_validated",
            "capture.hand_constraint_result.applied",
            "capture.hand_constraint_result.reachable",
        ):
            with self.subTest(required=required):
                self.assertIn(required, guard)
        self.assertNotIn(
            "!capture.grasp_evidence_valid",
            guard,
            "positive runtime weight must require a same-frame solve even "
            "before the evidence schema exposes selected-grasp fields",
        )

        pickup_writer = self._between(
            controller,
            "void write_autodemo_record(",
            "void write_placement_autodemo_record(",
        )
        placement_writer = self._between(
            controller,
            "void write_placement_autodemo_record(",
            "void validate_autodemo_screenshot(",
        )
        for name, writer in (
            ("pickup", pickup_writer),
            ("placement", placement_writer),
        ):
            with self.subTest(writer=name):
                self.assertIn(
                    "!autodemo_hand_constraint_evidence_valid(capture)",
                    writer,
                    "invalid positive-weight final-FK grasp evidence must be "
                    "rejected before either writer publishes it",
                )

    def test_owned_grasp_guard_reports_every_conjunct_without_weakening_predicate(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        guard = self._between(
            controller,
            "                const bool owned_pre_release =",
            "                if (placement_autodemo_state.interact_pulsed &&",
        )
        predicate = self._between(
            guard,
            "                if (owned_pre_release &&",
            "                {",
        )
        self.assertEqual(
            " ".join(predicate.split()),
            "if (owned_pre_release && "
            "(interaction_output.diagnostics.object_state != "
            "interaction::ObjectState::Held || "
            "!interaction_output.diagnostics.attached || "
            "!interaction_output.owns_pose || "
            "!capture.grasp_evidence_valid || "
            "capture.hand_constraint_weight != 1.0F || "
            "!capture.hand_constraint_validated || "
            "!capture.hand_constraint_result.applied || "
            "!capture.hand_constraint_result.reachable))",
        )
        for label in (
            "state=",
            "object_state=",
            "attached=",
            "owns_pose=",
            "grasp_valid=",
            "weight=",
            "validated=",
            "applied=",
            "reachable=",
            "reach_shortfall=",
            "target=",
            "affordance=",
            "hand=",
        ):
            with self.subTest(label=label):
                self.assertIn(f'" {label}"', guard)

    def test_makefile_adds_an_isolated_safe_placement_gate(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        for required in (
            "PLACEMENT_EVIDENCE_DIR ?= playable-evidence/placement",
            "PLACEMENT_LOG_PATH ?= $(PLACEMENT_EVIDENCE_DIR)/placement.jsonl",
            "PLACEMENT_SCREENSHOT_PATH ?= $(PLACEMENT_EVIDENCE_DIR)/placement.png",
            "PLACEMENT_FEATURES_OUTPUT ?= $(PLACEMENT_EVIDENCE_DIR)/locomotion-features.bin",
            "gate-playable-placement:",
            "MM_INTERACTION_PLACE_AUTODEMO=1",
            "timeout --signal=TERM --kill-after=5s 60s ./controller",
            "PLACEMENT_LOG=\"$(PLACEMENT_LOG_PATH)\"",
            "tests.python.test_playable_placement_evidence -v",
        ):
            with self.subTest(required=required):
                self.assertIn(required, makefile)
        lines = makefile.splitlines()
        start = next(
            index for index, line in enumerate(lines)
            if line.startswith("gate-playable-placement:")
        )
        target_lines = [lines[start]]
        for line in lines[start + 1:]:
            if line and not line.startswith(("\t", " ")):
                break
            target_lines.append(line)
        target = "\n".join(target_lines)
        self.assertIn("xdpyinfo", target)
        self.assertIn("sha256sum resources/features.bin", target)
        self.assertNotIn("interaction_query_probe", target)

    def test_makefile_builds_exact_controller_release_parity_binaries(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "CONTROLLER_RELEASE_PARITY_FLAGS := "
            "-O3 -DNDEBUG -ffast-math -march=native",
            makefile,
        )
        self.assertIn(
            "CONTROLLER_SCENE_RELEASE_FAST_MATH_TEST := "
            "$(CPP_TEST_DIR)/test_controller_scene_release_fast_math",
            makefile,
        )
        self.assertIn(
            "RELEASE_INTERACTION_PLACE_PROBE := "
            "$(CPP_TEST_DIR)/interaction_place_probe_release",
            makefile,
        )

        canary = self._make_target(
            makefile, "$(CONTROLLER_SCENE_RELEASE_FAST_MATH_TEST)"
        )
        self.assertIn(
            "tests/cpp/test_controller_scene_release_fast_math.cpp", canary
        )
        self.assertIn("$(CONTROLLER_RELEASE_PARITY_FLAGS)", canary)

        release_probe = self._make_target(
            makefile, "$(RELEASE_INTERACTION_PLACE_PROBE)"
        )
        self.assertIn("interaction_place_probe.cpp", release_probe)
        self.assertIn("$(CONTROLLER_RELEASE_PARITY_FLAGS)", release_probe)

        safe_suite = self._make_target(makefile, "test-interaction-safe")
        self.assertIn(
            "test-controller-scene-release-fast-math", safe_suite
        )

    def test_headless_gate_validates_normal_and_release_probes_on_same_pack(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        target = self._make_target(makefile, "gate-place-headless")
        self.assertIn(
            './interaction_place_probe "$(INTERACTION_DEMO_PACK)" --json',
            target,
        )
        self.assertIn(
            '"$(RELEASE_INTERACTION_PLACE_PROBE)" '
            '"$(INTERACTION_DEMO_PACK)" --json',
            target,
        )
        self.assertEqual(
            target.count(
                'INTERACTION_PLACE_PROBE_PACK="$(INTERACTION_DEMO_PACK)"'
            ),
            2,
        )
        self.assertEqual(
            target.count("tests.python.test_place_probe -v"),
            2,
        )

    def test_playable_gate_builds_then_certifies_exact_pack_sequentially(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        target = self._make_target(makefile, "gate-playable-placement")
        self.assertEqual(target.splitlines()[0], "gate-playable-placement:")
        ordered = (
            "$(MAKE) demo-interaction-pack",
            "$(MAKE) gate-place-headless",
            "$(MAKE) bootstrap-raylib",
            "$(MAKE) controller",
        )
        recipe = [
            line.strip()
            for line in target.splitlines()[1:]
            if line.startswith("\t") and line.strip()
        ]
        self.assertEqual(tuple(recipe[:4]), ordered)
        positions = [target.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(target.count("$(MAKE) demo-interaction-pack"), 1)
        self.assertEqual(target.count("$(MAKE) gate-place-headless"), 1)
        after_certification = target[positions[1] + len(ordered[1]):]
        self.assertNotIn("build_g1_interaction_database", after_certification)

    def test_playable_gate_rechecks_both_interaction_pack_hashes(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        target = self._make_target(makefile, "gate-playable-placement")
        controller = target.index(
            "timeout --signal=TERM --kill-after=5s 60s ./controller"
        )
        for stem, filename in (
            ("interaction_database", "interaction_database.bin"),
            ("interaction_features", "interaction_features.bin"),
        ):
            command = (
                'sha256sum "$(INTERACTION_DEMO_PACK)/' + filename + '"'
            )
            with self.subTest(filename=filename):
                self.assertEqual(target.count(command), 2)
                before = target.index(
                    f'{stem}_before="$$(sha256sum '
                    f'"$(INTERACTION_DEMO_PACK)/{filename}")"'
                )
                after = target.index(
                    f'{stem}_after="$$(sha256sum '
                    f'"$(INTERACTION_DEMO_PACK)/{filename}")"'
                )
                comparison = target.index(
                    f'if test "$${stem}_before" != "$${stem}_after"; then'
                )
                self.assertLess(before, controller)
                self.assertLess(controller, after)
                self.assertLess(after, comparison)
                self.assertIn(
                    f"ERROR $(INTERACTION_DEMO_PACK)/{filename} changed "
                    "during placement gate",
                    target,
                )

    def test_readme_documents_manual_and_graphical_placement_scope(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        for required in (
            "make gate-place-headless",
            "make gate-playable-placement",
            "playable-evidence/placement/placement.jsonl",
            "playable-evidence/placement/placement.png",
            "reversed_pickup",
            "MM_INTERACTION_PLACE_AUTODEMO=1",
            "navigation waypoint",
            "canonical relocation",
            "live default spawn",
            "flat-ground",
            "manual 1.00 m surface",
            "retained authored destination",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)


class Task14ArrivalControllerSourceTests(unittest.TestCase):
    @staticmethod
    def _between(source: str, begin: str, end: str) -> str:
        start = source.index(begin)
        stop = source.index(end, start + len(begin))
        return source[start:stop]

    def setUp(self):
        self.controller = Path("controller.cpp").read_text(encoding="utf-8")

    def test_stationary_candidates_are_derived_once_and_placement_fails_closed(self):
        for include in (
            '#include "interaction_arrival.h"',
            '#include "locomotion_controller_update.h"',
            '#include "stationary_motion_matching.h"',
        ):
            with self.subTest(include=include):
                self.assertIn(include, self.controller)
        self.assertEqual(
            self.controller.count(
                "stationary_motion_matching::derive_candidates("
            ),
            1,
        )
        feature_setup = self._between(
            self.controller,
            "    database_build_matching_features(",
            "    // Interaction data is a separate fixed-25 pack.",
        )
        self.assertIn(
            "stationary_motion_matching::derive_candidates(db)",
            feature_setup,
        )
        self.assertLess(
            feature_setup.index("database_build_matching_features("),
            feature_setup.index(
                "stationary_motion_matching::derive_candidates(db)"
            ),
        )
        self.assertIn(
            "placement auto-demo requires stationary locomotion candidates",
            self.controller,
        )

    def test_frozen_arrival_uses_paired_sticks_strafe_and_strict_latch(self):
        scripted = self._between(
            self.controller,
            "// Placement auto-demo input begins.",
            "// Placement auto-demo input ends.",
        )
        unfrozen = self._between(
            scripted,
            "// Placement unfrozen entry steering begins.",
            "// Placement unfrozen entry steering ends.",
        )
        frozen = self._between(
            scripted,
            "// Placement frozen arrival steering begins.",
            "// Placement frozen arrival steering ends.",
        )
        latch = self._between(
            scripted,
            "// Placement arrival latch begins.",
            "// Placement arrival latch ends.",
        )
        normalized_latch = " ".join(latch.split())
        self.assertIn("controller_world_navigation_stick(", unfrozen)
        self.assertNotIn("arrival_navigation_stick(", unfrozen)
        for required in (
            "interaction::arrival_navigation_stick(",
            "interaction::arrival_facing_stick(",
            "interaction_waypoint.position",
            "interaction_waypoint.rotation",
            "placement_arrival_facing_override = true",
        ):
            with self.subTest(required=required):
                self.assertIn(required, frozen)
        for required in (
            "interaction::arrival_ready(",
            "interaction_position_error_m",
            "planar_simulation_speed_mps",
            "interaction_yaw_error_radians",
            "pickup_distance_m",
            "reach_braking = true",
            "brake_simulation_speed_mps = planar_simulation_speed_mps",
            "placement_brake_latched_this_tick = true",
            "placement_arrival_facing_override = false",
            "gamepadstick_left = vec3()",
            "gamepadstick_right = vec3()",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_latch)
        self.assertIn("gamepadstick_right = vec3();", scripted)
        self.assertNotIn("kPlacementAutodemoReachBrakingDistanceM", self.controller)

        strafe_to_rotation = self._between(
            self.controller,
            "        bool desired_strafe = raw_desired_strafe;",
            "        // Check if we should force a search because input changed quickly",
        )
        self.assertIn(
            "if (placement_arrival_facing_override)", strafe_to_rotation
        )
        self.assertIn("desired_strafe = true;", strafe_to_rotation)
        self.assertLess(
            strafe_to_rotation.index("desired_strafe = true;"),
            strafe_to_rotation.index("desired_rotation_update("),
        )

    def test_brake_tick_and_recurring_searches_are_stationary_constrained(self):
        search = self._between(
            self.controller,
            "// Placement stationary search begins.",
            "// Placement stationary search ends.",
        )
        for required in (
            "placement_autodemo_enabled",
            "placement_autodemo_state.reach_braking",
            "!placement_autodemo_state.interact_pulsed",
            "interaction::RuntimeState::Locomotion",
            "placement_brake_latched_this_tick",
            "stationary_motion_matching::search(",
            "stationary_search_calls",
            "stationary_transition_count",
            "stationary_selected_frame",
            "stationary_selected_range",
            "database_search(",
            "inertialize_pose_transition(",
        ):
            with self.subTest(required=required):
                self.assertIn(required, search)
        constrained_call = search.index(
            "stationary_motion_matching::search("
        )
        self.assertLess(
            constrained_call,
            search.index("inertialize_pose_transition(", constrained_call),
        )
        self.assertIn(
            "if (best_index != frame_index ||\n"
            "                    (placement_autodemo_state"
            ".stationary_constraint_active &&\n"
            "                     placement_autodemo_state\n"
            "                         .stationary_transition_count == 0U) ||\n"
            "                    manual_pick_stationary_constraint_latched_this_tick)",
            search,
        )
        placement_input = self._between(
            self.controller,
            "// Placement auto-demo input begins.",
            "// Placement auto-demo input ends.",
        )
        for forbidden in (
            "simulation_position =",
            "bone_positions(0) =",
            "joint_world_positions[",
            "joint_world_rotations[",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, placement_input + search)

    def test_pickup_interact_tick_remains_stationary_constrained(self):
        scripted = self._between(
            self.controller,
            "// Placement auto-demo input begins.",
            "// Placement auto-demo input ends.",
        )
        search = self._between(
            self.controller,
            "// Placement stationary search begins.",
            "// Placement stationary search ends.",
        )
        self.assertIn(
            "bool placement_pick_interact_submitted_this_tick = false;",
            self.controller,
        )
        self.assertEqual(
            self.controller.count(
                "placement_pick_interact_submitted_this_tick = true;"
            ),
            1,
        )
        pickup_submission = scripted.index(
            "placement_pick_interact_submitted_this_tick = true;"
        )
        pickup_counter = scripted.index(
            "++placement_autodemo_state.pick_interact_submission_count;"
        )
        destination_submission = scripted.index(
            "!placement_autodemo_state.destination_latched"
        )
        self.assertLess(pickup_submission, pickup_counter)
        self.assertLess(pickup_counter, destination_submission)

        constraint_assignment = self._between(
            search,
            "placement_autodemo_state.stationary_constraint_active =",
            ";",
        )
        normalized_constraint = " ".join(constraint_assignment.split())
        for required in (
            "placement_autodemo_enabled",
            "placement_autodemo_state.reach_braking",
            "interaction_scheduler.cached_output().diagnostics.state == "
            "interaction::RuntimeState::Locomotion",
            "(!placement_autodemo_state.interact_pulsed || "
            "placement_pick_interact_submitted_this_tick)",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_constraint)

    def test_evidence_and_bounded_diagnostic_publish_observed_stationary_state(self):
        writer = self._between(
            self.controller,
            "void write_placement_autodemo_record(",
            "void validate_autodemo_screenshot(",
        )
        for required in (
            "state.stationary_candidate_count",
            "state.stationary_constraint_active",
            "state.stationary_search_calls",
            "state.stationary_transition_count",
            "state.stationary_selected_frame",
            "state.stationary_selected_range",
            "state.brake_simulation_speed_mps",
            'stationary_candidate_count',
            'stationary_constraint_active',
            'stationary_search_calls',
            'stationary_transition_count',
            'stationary_selected_frame',
            'stationary_selected_range',
            'brake_simulation_speed_mps',
        ):
            with self.subTest(required=required):
                self.assertIn(required, writer)
        diagnostic = self._between(
            self.controller,
            "// Placement post-entry diagnostic begins.",
            "// Placement post-entry diagnostic ends.",
        )
        for required in (
            "250U",
            "interaction_error_m=",
            "standoff_m=",
            "displayed_speed_mps=",
            "simulation_speed_mps=",
            "yaw_error_degrees=",
            "brake_latched=",
            "stationary_search_calls=",
            "stationary_transition_count=",
            "stationary_selected_frame=",
            "stationary_selected_range=",
        ):
            with self.subTest(required=required):
                self.assertIn(required, diagnostic)

    def test_shared_update_module_is_controller_linked_and_database_free(self):
        header = Path("locomotion_controller_update.h").read_text(
            encoding="utf-8"
        )
        source = Path("locomotion_controller_update.cpp").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('#include "database.h"', header + source)
        for definition in (
            "vec3 desired_velocity_update(",
            "quat desired_rotation_update(",
            "vec3 simulation_collide_obstacles(",
            "void simulation_positions_update(",
            "void simulation_rotations_update(",
        ):
            with self.subTest(definition=definition):
                self.assertEqual(source.count(definition), 1)
                self.assertNotIn(definition, self.controller)


@unittest.skipUnless(
    os.environ.get("PLACEMENT_LOG") and os.environ.get("PLACEMENT_SCREENSHOT"),
    "placement evidence paths not set",
)
class PlayablePlacementEvidenceTests(unittest.TestCase):
    def test_real_playable_placement_evidence(self):
        records = load_evidence(Path(os.environ["PLACEMENT_LOG"]))
        validate_evidence(records, require_current_pack=True)
        validate_screenshot(Path(os.environ["PLACEMENT_SCREENSHOT"]))
        self.assertEqual(records[-1]["place_mode"], "reversed_pickup")


if __name__ == "__main__":
    unittest.main()
