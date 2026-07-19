import copy
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest
import zlib


EXPECTED_KEYS = (
    "render_frame",
    "runtime_tick",
    "scheduler_phase",
    "state",
    "result",
    "reason",
    "object_state",
    "attached",
    "owns_pose",
    "carry_mode",
    "carry_command_frame",
    "root_position",
    "object_position",
    "grasp_evidence_valid",
    "active_hand_joint",
    "hand_constraint_weight",
    "hand_constraint_validated",
    "hand_constraint_applied",
    "hand_constraint_reachable",
    "hand_constraint_used_clavicle",
    "hand_constraint_reach_shortfall_m",
    "hand_constraint_calibration_rotation",
    "calibrated_hand_world_rotation",
    "object_world_rotation",
    "hand_in_object_position",
    "hand_in_object_rotation",
    "grasp_world_position",
    "grasp_world_rotation",
    "root_displacement_m",
    "joint_world_positions",
    "joint_world_rotations",
    "action",
)


class EvidenceValidationError(ValueError):
    pass


STATES = {
    "Disabled",
    "Locomotion",
    "Preflight",
    "Align",
    "PickupReplay",
    "Hold",
    "Carry",
}
RESULTS = {
    "None",
    "Accepted",
    "Succeeded",
    "Rejected",
    "Cancelled",
    "Failed",
    "Reset",
}
REASONS = {
    "None",
    "PackUnavailable",
    "TargetUnavailable",
    "TargetChanged",
    "OutOfRange",
    "NoCandidate",
    "PoorMatch",
    "BlockedPath",
    "CorrectionLimit",
    "Cancelled",
    "ContactPosition",
    "ContactOrientation",
    "JointLimit",
    "LostContact",
    "ClipEnded",
    "Reset",
}
OBJECT_STATES = {"Free", "Targeted", "Attached", "Held"}
CARRY_MODES = {"none", "recorded", "layered"}
ACTIONS = {"none", "interact", "forward", "reset"}
EXPECTED_STATE_ORDER = (
    "Locomotion",
    "Preflight",
    "Align",
    "PickupReplay",
    "Hold",
    "Carry",
    "Locomotion",
)
CONTROL_RATE_HZ = 25
INTERACT_FRAME = 13
CARRY_COMMAND_COUNT = 63
FINAL_CARRY_COMMAND = CARRY_COMMAND_COUNT - 1
CARRY_DEADLINE_FRAMES = 375
MAX_EVIDENCE_RECORDS = 500
RESET_PRESENTATION_FRAMES = 7
FLAT_JOINT_NAMES = (
    "Entity",
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
)
JOINT_TRANSLATION_LIMIT_M = 0.20
DISTAL_LOWER_LIMB_JOINT_NAMES = frozenset(
    ("LeftFoot", "LeftToe", "RightFoot", "RightToe")
)
DISTAL_LOWER_LIMB_TRANSLATION_SPEED_LIMIT_MPS = 12.0
AUTHORITY_SEAM_TRANSLATION_LIMIT_M = 0.20
HOLD_TO_LAYERED_CARRY_AXIAL_TRANSLATION_LIMIT_M = 0.05
HOLD_TO_LAYERED_CARRY_AXIAL_JOINT_NAMES = frozenset(
    ("Spine2", "Neck", "Head", "LeftShoulder", "RightShoulder")
)
JOINT_ROTATION_LIMIT_DEGREES = 60.0
JOINT_QUATERNION_NORM_TOLERANCE = 1.0e-3
GRASP_COMPOSITION_TOLERANCE = 1.0e-5
GRASP_EVIDENCE_STATES = ("PickupReplay", "Hold", "Carry")
HAND_POSITION_LIMIT_M = 0.01
HAND_CALIBRATED_ORIENTATION_LIMIT_DEGREES = 2.0
# Canonical flat-autodemo morphology guard; recorded/high-carry motion is exempt.
LAYERED_INACTIVE_HAND_ELEVATION_LIMIT_M = 0.10


def _error(message: str) -> EvidenceValidationError:
    return EvidenceValidationError(message)


def _object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise _error(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str):
    raise _error(f"non-finite JSON number {value}")


def _require_integer(record: dict, name: str, minimum: int, maximum=None) -> None:
    value = record[name]
    if type(value) is not int or value < minimum:
        raise _error(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise _error(f"{name} must be <= {maximum}")


def _require_enum(record: dict, name: str, allowed: set[str]) -> None:
    value = record[name]
    if type(value) is not str or value not in allowed:
        raise _error(f"{name} has invalid value {value!r}")


def _require_position(record: dict, name: str) -> None:
    value = record[name]
    if type(value) is not list or len(value) != 3:
        raise _error(f"{name} must contain exactly three numbers")
    for component in value:
        if type(component) not in (int, float) or not math.isfinite(component):
            raise _error(f"{name} must contain only finite numbers")


def _require_quaternion(record: dict, name: str) -> None:
    value = record[name]
    if type(value) is not list or len(value) != 4:
        raise _error(f"{name} must contain exactly four numbers")
    if any(
        type(component) not in (int, float) or not math.isfinite(component)
        for component in value
    ):
        raise _error(f"{name} must contain only finite numbers")
    norm = math.sqrt(sum(component * component for component in value))
    norm_error = abs(norm - 1.0)
    if norm_error > JOINT_QUATERNION_NORM_TOLERANCE:
        raise _error(
            f"{name} quaternion norm error {norm_error:.6f} exceeds max "
            f"{JOINT_QUATERNION_NORM_TOLERANCE:.6f}"
        )


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]
    norm = math.sqrt(sum(component * component for component in result))
    return [component / norm for component in result]


def _quaternion_inverse(rotation: list[float]) -> list[float]:
    norm_squared = sum(component * component for component in rotation)
    return [
        rotation[0] / norm_squared,
        -rotation[1] / norm_squared,
        -rotation[2] / norm_squared,
        -rotation[3] / norm_squared,
    ]


def _quaternion_rotate(rotation: list[float], value: list[float]) -> list[float]:
    _, x, y, z = rotation
    vector = (x, y, z)

    def cross(left, right):
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    twice_cross = tuple(2.0 * component for component in cross(vector, value))
    nested_cross = cross(vector, twice_cross)
    return [
        value[index]
        + rotation[0] * twice_cross[index]
        + nested_cross[index]
        for index in range(3)
    ]


def _quaternion_sign_distance(left: list[float], right: list[float]) -> float:
    same = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    opposite = math.sqrt(sum((a + b) ** 2 for a, b in zip(left, right)))
    return min(same, opposite)


def _require_joint_world_arrays(record: dict) -> None:
    frame = record["render_frame"]
    specifications = (
        ("joint_world_positions", 3),
        ("joint_world_rotations", 4),
    )
    for field, component_count in specifications:
        values = record[field]
        if type(values) is not list or len(values) != len(FLAT_JOINT_NAMES):
            raise _error(
                f"frame {frame} {field} must contain exactly "
                f"{len(FLAT_JOINT_NAMES)} joints"
            )
        for joint, components in enumerate(values):
            joint_name = FLAT_JOINT_NAMES[joint]
            if type(components) is not list or len(components) != component_count:
                raise _error(
                    f"frame {frame} joint {joint} ({joint_name}) {field} "
                    f"must contain exactly {component_count} components"
                )
            if any(
                type(component) not in (int, float)
                or not math.isfinite(component)
                for component in components
            ):
                raise _error(
                    f"frame {frame} joint {joint} ({joint_name}) {field} "
                    "must contain only finite numbers"
                )

    for joint, quaternion in enumerate(record["joint_world_rotations"]):
        norm = math.sqrt(sum(component * component for component in quaternion))
        norm_error = abs(norm - 1.0)
        if norm_error > JOINT_QUATERNION_NORM_TOLERANCE:
            raise _error(
                f"frame {frame} joint {joint} ({FLAT_JOINT_NAMES[joint]}) "
                f"quaternion norm error {norm_error:.6f} exceeds max "
                f"{JOINT_QUATERNION_NORM_TOLERANCE:.6f}"
            )


def _joint_rotation_step_degrees(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    dot = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    sign_invariant_dot = min(1.0, max(0.0, abs(dot)))
    return math.degrees(2.0 * math.acos(sign_invariant_dot))


def _validate_record_types(record: dict) -> None:
    _require_integer(record, "render_frame", 0)
    _require_integer(record, "runtime_tick", 0)
    _require_integer(record, "scheduler_phase", 0, 0)
    _require_enum(record, "state", STATES)
    _require_enum(record, "result", RESULTS)
    _require_enum(record, "reason", REASONS)
    _require_enum(record, "object_state", OBJECT_STATES)
    for name in ("attached", "owns_pose"):
        if type(record[name]) is not bool:
            raise _error(f"{name} must be a JSON boolean")
    _require_enum(record, "carry_mode", CARRY_MODES)
    _require_integer(record, "carry_command_frame", -1, FINAL_CARRY_COMMAND)
    _require_position(record, "root_position")
    _require_position(record, "object_position")
    if type(record["grasp_evidence_valid"]) is not bool:
        raise _error("grasp_evidence_valid must be a JSON boolean")
    active_hand_joint = record["active_hand_joint"]
    if type(active_hand_joint) is not int or active_hand_joint not in (-1, 18, 22):
        raise _error("active_hand_joint must be exactly -1, 18, or 22")
    if record["grasp_evidence_valid"] != (active_hand_joint in (18, 22)):
        raise _error(
            "active_hand_joint must be -1 exactly when grasp evidence is invalid"
        )
    weight = record["hand_constraint_weight"]
    if (
        type(weight) not in (int, float)
        or not math.isfinite(weight)
        or weight < 0.0
        or weight > 1.0
    ):
        raise _error("hand_constraint_weight must be finite and in [0, 1]")
    for name in (
        "hand_constraint_validated",
        "hand_constraint_applied",
        "hand_constraint_reachable",
        "hand_constraint_used_clavicle",
    ):
        if type(record[name]) is not bool:
            raise _error(f"{name} must be a JSON boolean")
    reach_shortfall = record["hand_constraint_reach_shortfall_m"]
    if (
        type(reach_shortfall) not in (int, float)
        or not math.isfinite(reach_shortfall)
        or reach_shortfall < 0.0
    ):
        raise _error(
            "hand_constraint_reach_shortfall_m must be finite and nonnegative"
        )
    _require_quaternion(record, "hand_constraint_calibration_rotation")
    _require_quaternion(record, "calibrated_hand_world_rotation")
    _require_quaternion(record, "object_world_rotation")
    _require_position(record, "hand_in_object_position")
    _require_quaternion(record, "hand_in_object_rotation")
    _require_position(record, "grasp_world_position")
    _require_quaternion(record, "grasp_world_rotation")
    if (
        record["state"] in GRASP_EVIDENCE_STATES
        and not record["grasp_evidence_valid"]
    ):
        raise _error(
            f'{record["state"]} requires valid selected-grasp evidence'
        )
    _require_joint_world_arrays(record)
    if record["grasp_evidence_valid"]:
        rotated_local = _quaternion_rotate(
            record["object_world_rotation"],
            record["hand_in_object_position"],
        )
        expected_position = [
            object_component + local_component
            for object_component, local_component in zip(
                record["object_position"], rotated_local
            )
        ]
        expected_rotation = _quaternion_multiply(
            record["object_world_rotation"],
            record["hand_in_object_rotation"],
        )
        if (
            math.dist(expected_position, record["grasp_world_position"])
            > GRASP_COMPOSITION_TOLERANCE
            or _quaternion_sign_distance(
                expected_rotation, record["grasp_world_rotation"]
            )
            > GRASP_COMPOSITION_TOLERANCE
        ):
            raise _error(
                "grasp_world must equal object_world * hand_in_object"
            )
        if record["hand_constraint_validated"]:
            expected_calibrated_rotation = _quaternion_multiply(
                record["joint_world_rotations"][active_hand_joint],
                _quaternion_inverse(
                    record["hand_constraint_calibration_rotation"]
                ),
            )
            if _quaternion_sign_distance(
                expected_calibrated_rotation,
                record["calibrated_hand_world_rotation"],
            ) > GRASP_COMPOSITION_TOLERANCE:
                raise _error(
                    "calibrated_hand_world_rotation must equal the final "
                    "rendered hand rotation * inverse(epoch calibration)"
                )
        if weight > 0.0:
            for name in (
                "hand_constraint_validated",
                "hand_constraint_applied",
                "hand_constraint_reachable",
            ):
                if not record[name]:
                    raise _error(f"positive-weight grasp requires {name}")
        if record["attached"] and weight == 1.0:
            position_error_m = math.dist(
                record["joint_world_positions"][active_hand_joint],
                record["grasp_world_position"],
            )
            if position_error_m > HAND_POSITION_LIMIT_M:
                raise _error(
                    f"attached full-weight hand position {position_error_m:.6f} m "
                    f"exceeds max {HAND_POSITION_LIMIT_M:.6f} m"
                )
            orientation_error_degrees = _joint_rotation_step_degrees(
                record["calibrated_hand_world_rotation"],
                record["grasp_world_rotation"],
            )
            if (
                orientation_error_degrees
                > HAND_CALIBRATED_ORIENTATION_LIMIT_DEGREES
            ):
                raise _error(
                    "attached full-weight calibrated hand orientation "
                    f"{orientation_error_degrees:.6f} degrees exceeds max "
                    f"{HAND_CALIBRATED_ORIENTATION_LIMIT_DEGREES:.6f} degrees"
                )
    if record["hand_constraint_applied"] and (
        not record["hand_constraint_validated"] or not weight > 0.0
    ):
        raise _error(
            "hand_constraint_applied requires validated positive weight"
        )
    if record["hand_constraint_reachable"] and not record["hand_constraint_applied"]:
        raise _error("hand_constraint_reachable requires hand_constraint_applied")
    if (
        record["hand_constraint_used_clavicle"]
        and not record["hand_constraint_applied"]
    ):
        raise _error(
            "hand_constraint_used_clavicle requires hand_constraint_applied"
        )
    displacement = record["root_displacement_m"]
    if type(displacement) not in (int, float) or not math.isfinite(displacement):
        raise _error("root_displacement_m must be a finite number")
    if displacement < 0.0:
        raise _error("root_displacement_m must be nonnegative")
    _require_enum(record, "action", ACTIONS)


def load_evidence(path: Path) -> list[dict]:
    path = Path(path)
    try:
        if not path.is_file():
            raise _error(f"evidence log is not a regular file: {path}")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _error(f"cannot read evidence log {path}: {error}") from error
    if not text:
        raise _error("evidence log is empty")
    if not text.endswith("\n"):
        raise _error("evidence log must end with a newline")

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise _error(f"blank evidence line {line_number}")
        try:
            record = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, EvidenceValidationError) as error:
            raise _error(f"invalid evidence line {line_number}: {error}") from error
        if type(record) is not dict:
            raise _error(f"evidence line {line_number} must be a JSON object")
        if tuple(record) != EXPECTED_KEYS:
            raise _error(
                f"evidence line {line_number} fields/order must be {EXPECTED_KEYS}"
            )
        _validate_record_types(record)
        if line + "\n" != _record_line(record):
            raise _error(
                f"evidence line {line_number} is not compact fixed-six-decimal JSON"
            )
        records.append(record)
    return records


def _collapsed_states(records: list[dict]) -> tuple[str, ...]:
    collapsed = []
    for record in records:
        state = record["state"]
        if not collapsed or collapsed[-1] != state:
            collapsed.append(state)
    return tuple(collapsed)


def validate_evidence(records: list[dict]) -> None:
    if not records:
        raise _error("evidence has no render records")
    if len(records) > MAX_EVIDENCE_RECORDS:
        raise _error(
            f"evidence must contain at most {MAX_EVIDENCE_RECORDS} render records"
        )
    for index, record in enumerate(records):
        _require_joint_world_arrays(record)
        if record["render_frame"] != index:
            raise _error("render_frame must start at zero and be sequential")
        if index == 0:
            continue
        previous = records[index - 1]
        authority_seam = (
            record["state"] != previous["state"]
            or record["owns_pose"] != previous["owns_pose"]
            or record["attached"] != previous["attached"]
        )
        hold_to_layered_carry = (
            previous["state"] == "Hold"
            and record["state"] == "Carry"
            and record["carry_mode"] == "layered"
        )
        for joint, joint_name in enumerate(FLAT_JOINT_NAMES):
            translation_step_m = math.dist(
                previous["joint_world_positions"][joint],
                record["joint_world_positions"][joint],
            )
            if (
                hold_to_layered_carry
                and joint_name in HOLD_TO_LAYERED_CARRY_AXIAL_JOINT_NAMES
                and translation_step_m
                > HOLD_TO_LAYERED_CARRY_AXIAL_TRANSLATION_LIMIT_M
            ):
                raise _error(
                    f"joint {joint} ({joint_name}) frame {index - 1}->{index} "
                    f"Hold-to-layered Carry translation "
                    f"{translation_step_m:.6f} m exceeds max "
                    f"{HOLD_TO_LAYERED_CARRY_AXIAL_TRANSLATION_LIMIT_M:.6f} m"
                )
            if (
                authority_seam
                and translation_step_m > AUTHORITY_SEAM_TRANSLATION_LIMIT_M
            ):
                raise _error(
                    f"joint {joint} ({joint_name}) frame {index - 1}->{index} "
                    f"translation {translation_step_m:.6f} m across authority "
                    f"seam exceeds max "
                    f"{AUTHORITY_SEAM_TRANSLATION_LIMIT_M:.6f} m"
                )
            if not authority_seam:
                if joint_name in DISTAL_LOWER_LIMB_JOINT_NAMES:
                    translation_speed_mps = translation_step_m * CONTROL_RATE_HZ
                    if (
                        translation_speed_mps
                        > DISTAL_LOWER_LIMB_TRANSLATION_SPEED_LIMIT_MPS
                    ):
                        raise _error(
                            f"joint {joint} ({joint_name}) frame "
                            f"{index - 1}->{index} translation speed "
                            f"{translation_speed_mps:.6f} m/s exceeds max "
                            f"{DISTAL_LOWER_LIMB_TRANSLATION_SPEED_LIMIT_MPS:.6f} "
                            "m/s"
                        )
                elif translation_step_m > JOINT_TRANSLATION_LIMIT_M:
                    raise _error(
                        f"joint {joint} ({joint_name}) frame {index - 1}->{index} "
                        f"translation {translation_step_m:.6f} m exceeds max "
                        f"{JOINT_TRANSLATION_LIMIT_M:.6f} m"
                    )
            rotation_step_degrees = _joint_rotation_step_degrees(
                previous["joint_world_rotations"][joint],
                record["joint_world_rotations"][joint],
            )
            if rotation_step_degrees > JOINT_ROTATION_LIMIT_DEGREES:
                raise _error(
                    f"joint {joint} ({joint_name}) frame {index - 1}->{index} "
                    f"rotation {rotation_step_degrees:.6f} degrees exceeds max "
                    f"{JOINT_ROTATION_LIMIT_DEGREES:.6f} degrees"
                )
        if record["runtime_tick"] != previous["runtime_tick"] + 1:
            raise _error("runtime_tick must advance exactly once per 25 Hz frame")
        if record["scheduler_phase"] != 0:
            raise _error("scheduler_phase must remain zero at synchronous 25 Hz")
        if not previous["owns_pose"] and not record["owns_pose"]:
            root_step_m = math.dist(
                previous["root_position"], record["root_position"]
            )
            if root_step_m > 0.20:
                raise _error(
                    "adjacent non-owned root step must not exceed 0.20 m"
                )

    interact = [record for record in records if record["action"] == "interact"]
    if len(interact) != 1 or interact[0]["render_frame"] != INTERACT_FRAME:
        raise _error(
            f"Interact must be pulsed exactly once on render frame {INTERACT_FRAME}"
        )
    post_interact = records[interact[0]["render_frame"]:]
    if any(
        record["state"] == "Disabled"
        or record["result"] in {"Rejected", "Cancelled", "Failed"}
        for record in post_interact
    ):
        raise _error("evidence contains an unsuccessful runtime output after Interact")

    reset = [record for record in records if record["action"] == "reset"]
    if len(reset) != 1 or reset[0]["render_frame"] != len(records) - 1:
        raise _error("Reset action must appear exactly once on the final record")

    if _collapsed_states(records) != EXPECTED_STATE_ORDER:
        raise _error("collapsed runtime states do not match the playable sequence")

    forward_indices = [
        index for index, record in enumerate(records) if record["action"] == "forward"
    ]
    forward = [records[index] for index in forward_indices]
    if len(forward) != CARRY_COMMAND_COUNT:
        raise _error(
            f"evidence must contain exactly {CARRY_COMMAND_COUNT} forward records"
        )
    if [record["carry_command_frame"] for record in forward] != list(
        range(CARRY_COMMAND_COUNT)
    ):
        raise _error(
            f"forward records must be numbered exactly 0 through "
            f"{FINAL_CARRY_COMMAND}"
        )
    if any(record["state"] != "Carry" or not record["attached"] for record in forward):
        raise _error("every forward record must be attached Carry")
    if any(
        record["carry_command_frame"] != -1
        for record in records
        if record["action"] != "forward"
    ):
        raise _error("non-forward records must use carry_command_frame -1")

    carry = [record for record in records if record["state"] == "Carry"]
    if not carry or not any(record["attached"] for record in records):
        raise _error("evidence must include attachment and Carry")
    if any(
        record["object_state"] != "Held"
        or not record["attached"]
        or not record["owns_pose"]
        for record in carry
    ):
        raise _error("all Carry records must be Held, attached, and own the pose")
    for record in carry:
        if record["carry_mode"] not in {"recorded", "layered"}:
            raise _error(
                f"Carry frame {record['render_frame']} carry_mode must be "
                "recorded or layered"
            )
        if record["carry_mode"] != "layered":
            continue
        if record["active_hand_joint"] == 22:
            inactive_arm_joint, inactive_hand_joint = 16, 18
        elif record["active_hand_joint"] == 18:
            inactive_arm_joint, inactive_hand_joint = 20, 22
        else:
            continue
        elevation_m = (
            record["joint_world_positions"][inactive_hand_joint][1]
            - record["joint_world_positions"][inactive_arm_joint][1]
        )
        if elevation_m > LAYERED_INACTIVE_HAND_ELEVATION_LIMIT_M + 1.0e-9:
            raise _error(
                f"layered inactive hand frame {record['render_frame']} elevation "
                f"{elevation_m:.6f} m exceeds max "
                f"{LAYERED_INACTIVE_HAND_ELEVATION_LIMIT_M:.6f} m"
            )
    first_carry_index = records.index(carry[0])
    if first_carry_index > CARRY_DEADLINE_FRAMES:
        raise _error(
            f"Carry must be observed by evidence frame {CARRY_DEADLINE_FRAMES}"
        )
    if forward_indices[0] != first_carry_index + 1:
        raise _error("forward command 0 must immediately follow first Carry")
    if forward_indices != list(
        range(forward_indices[0], forward_indices[0] + CARRY_COMMAND_COUNT)
    ):
        raise _error(
            f"forward commands 0 through {FINAL_CARRY_COMMAND} must be consecutive"
        )
    final_command_index = forward_indices[FINAL_CARRY_COMMAND]
    if final_command_index + 1 >= len(records):
        raise _error(
            f"Reset must immediately follow command {FINAL_CARRY_COMMAND}"
        )

    origin = carry[0]["root_position"]
    for index, record in enumerate(records):
        if index < first_carry_index:
            expected_displacement = 0.0
        else:
            root = record["root_position"]
            expected_displacement = math.hypot(
                root[0] - origin[0], root[2] - origin[2]
            )
        if not math.isclose(
            record["root_displacement_m"],
            expected_displacement,
            rel_tol=0.0,
            abs_tol=2.0e-6,
        ):
            raise _error("root_displacement_m does not match displayed-root motion")
    if carry[-1]["root_displacement_m"] <= 0.20:
        raise _error("final Carry displacement must exceed 0.20 m")
    first_object = carry[0]["object_position"]
    final_object = carry[-1]["object_position"]
    object_displacement = math.hypot(
        final_object[0] - first_object[0],
        final_object[2] - first_object[2],
    )
    if object_displacement <= 0.20:
        raise _error(
            "final Carry object horizontal displacement must exceed 0.20 m"
        )
    if (
        carry[-1]["action"] != "forward"
        or carry[-1]["carry_command_frame"] != FINAL_CARRY_COMMAND
    ):
        raise _error(
            f"command {FINAL_CARRY_COMMAND} must be the final Carry record"
        )

    final = records[-1]
    if records.index(carry[-1]) != len(records) - 2:
        raise _error("Reset must immediately follow the final Carry record")
    if not (
        final["state"] == "Locomotion"
        and final["action"] == "reset"
        and final["result"] == "Reset"
        and final["reason"] == "Reset"
        and final["object_state"] == "Free"
        and final["attached"] is False
    ):
        raise _error("final evidence record must be successful Reset to Locomotion")


def summarize_grasp_alignment(records: list[dict]) -> list[dict]:
    summaries = []
    for state in GRASP_EVIDENCE_STATES:
        state_records = [
            record
            for record in records
            if record["state"] == state and record["grasp_evidence_valid"]
        ]
        if not state_records:
            continue

        active_joints = {record["active_hand_joint"] for record in state_records}
        if len(active_joints) != 1:
            raise _error(f"{state} changes active_hand_joint within the state")
        active_hand_joint = next(iter(active_joints))

        position_errors = []
        orientation_errors = []
        calibrated_orientation_errors = []
        for record in state_records:
            hand_position = record["joint_world_positions"][active_hand_joint]
            hand_rotation = record["joint_world_rotations"][active_hand_joint]
            position_errors.append(
                math.dist(hand_position, record["grasp_world_position"])
            )
            orientation_errors.append(
                _joint_rotation_step_degrees(
                    hand_rotation, record["grasp_world_rotation"]
                )
            )
            calibrated_orientation_errors.append(
                _joint_rotation_step_degrees(
                    record["calibrated_hand_world_rotation"],
                    record["grasp_world_rotation"],
                )
            )

        position_max_index = max(
            range(len(position_errors)), key=position_errors.__getitem__
        )
        orientation_max_index = max(
            range(len(orientation_errors)), key=orientation_errors.__getitem__
        )
        calibrated_orientation_max_index = max(
            range(len(calibrated_orientation_errors)),
            key=calibrated_orientation_errors.__getitem__,
        )
        summaries.append(
            {
                "state": state,
                "count": len(state_records),
                "position_mean_m": sum(position_errors) / len(position_errors),
                "position_max_m": position_errors[position_max_index],
                "position_max_frame": state_records[position_max_index][
                    "render_frame"
                ],
                "orientation_mean_degrees": (
                    sum(orientation_errors) / len(orientation_errors)
                ),
                "orientation_max_degrees": orientation_errors[
                    orientation_max_index
                ],
                "orientation_max_frame": state_records[
                    orientation_max_index
                ]["render_frame"],
                "calibrated_orientation_mean_degrees": (
                    sum(calibrated_orientation_errors)
                    / len(calibrated_orientation_errors)
                ),
                "calibrated_orientation_max_degrees": (
                    calibrated_orientation_errors[
                        calibrated_orientation_max_index
                    ]
                ),
                "calibrated_orientation_max_frame": state_records[
                    calibrated_orientation_max_index
                ]["render_frame"],
                "active_hand_joint_name": FLAT_JOINT_NAMES[active_hand_joint],
            }
        )
    return summaries


def validate_screenshot(path: Path) -> None:
    path = Path(path)
    try:
        if not path.is_file():
            raise _error(f"screenshot is not a regular file: {path}")
        if path.stat().st_size <= 10_000:
            raise _error("screenshot must be larger than 10,000 bytes")
        data = path.read_bytes()
    except OSError as error:
        raise _error(f"cannot read screenshot {path}: {error}") from error
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _error("screenshot is not a PNG")

    offset = 8
    chunks = []
    idat = bytearray()
    while offset < len(data):
        if offset + 12 > len(data):
            raise _error("screenshot contains a truncated PNG chunk")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        end = offset + 12 + length
        if end > len(data):
            raise _error("screenshot contains a truncated PNG payload")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length:end])[0]
        actual_crc = zlib.crc32(kind)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise _error("screenshot PNG checksum mismatch")
        chunks.append((kind, payload))
        if kind == b"IDAT":
            idat.extend(payload)
        offset = end
        if kind == b"IEND":
            break
    if offset != len(data):
        raise _error("screenshot PNG has trailing bytes")
    if not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise _error("screenshot PNG has no valid IHDR")
    width, height = struct.unpack(">II", chunks[0][1][:8])
    if (width, height) != (1280, 720):
        raise _error("screenshot dimensions must be exactly 1280x720")
    if not idat or chunks[-1][0] != b"IEND":
        raise _error("screenshot PNG must contain IDAT and terminal IEND")
    try:
        zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise _error("screenshot PNG has invalid image data") from error


def _record_line(record: dict) -> str:
    def string(name: str) -> str:
        return json.dumps(record[name], ensure_ascii=True)

    def boolean(name: str) -> str:
        return "true" if record[name] else "false"

    def position(name: str) -> str:
        return "[" + ",".join(f"{value:.6f}" for value in record[name]) + "]"

    def joint_vectors(name: str) -> str:
        return "[" + ",".join(
            "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
            for vector in record[name]
        ) + "]"

    return (
        f'{{"render_frame":{record["render_frame"]},'
        f'"runtime_tick":{record["runtime_tick"]},'
        f'"scheduler_phase":{record["scheduler_phase"]},'
        f'"state":{string("state")},'
        f'"result":{string("result")},'
        f'"reason":{string("reason")},'
        f'"object_state":{string("object_state")},'
        f'"attached":{boolean("attached")},'
        f'"owns_pose":{boolean("owns_pose")},'
        f'"carry_mode":{string("carry_mode")},'
        f'"carry_command_frame":{record["carry_command_frame"]},'
        f'"root_position":{position("root_position")},'
        f'"object_position":{position("object_position")},'
        f'"grasp_evidence_valid":{boolean("grasp_evidence_valid")},'
        f'"active_hand_joint":{record["active_hand_joint"]},'
        f'"hand_constraint_weight":{record["hand_constraint_weight"]:.6f},'
        f'"hand_constraint_validated":{boolean("hand_constraint_validated")},'
        f'"hand_constraint_applied":{boolean("hand_constraint_applied")},'
        f'"hand_constraint_reachable":{boolean("hand_constraint_reachable")},'
        f'"hand_constraint_used_clavicle":{boolean("hand_constraint_used_clavicle")},'
        f'"hand_constraint_reach_shortfall_m":'
        f'{record["hand_constraint_reach_shortfall_m"]:.6f},'
        f'"hand_constraint_calibration_rotation":'
        f'{position("hand_constraint_calibration_rotation")},'
        f'"calibrated_hand_world_rotation":'
        f'{position("calibrated_hand_world_rotation")},'
        f'"object_world_rotation":{position("object_world_rotation")},'
        f'"hand_in_object_position":{position("hand_in_object_position")},'
        f'"hand_in_object_rotation":{position("hand_in_object_rotation")},'
        f'"grasp_world_position":{position("grasp_world_position")},'
        f'"grasp_world_rotation":{position("grasp_world_rotation")},'
        f'"root_displacement_m":{record["root_displacement_m"]:.6f},'
        f'"joint_world_positions":{joint_vectors("joint_world_positions")},'
        f'"joint_world_rotations":{joint_vectors("joint_world_rotations")},'
        f'"action":{string("action")}}}\n'
    )


def _reclock_records(records: list[dict]) -> None:
    for render_frame, record in enumerate(records):
        record["render_frame"] = render_frame
        record["runtime_tick"] = render_frame + 1
        record["scheduler_phase"] = 0


def _valid_records() -> list[dict]:
    states = (
        ["Locomotion"] * 14
        + ["Preflight"] * 2
        + ["Align"] * 3
        + ["PickupReplay"] * 2
        + ["Hold"] * 3
        + ["Carry"] * (CARRY_COMMAND_COUNT + 1)
        + ["Locomotion"]
    )
    records = []
    for render_frame, state in enumerate(states):
        attached = state in {"PickupReplay", "Hold", "Carry"}
        owns_pose = state in {"Align", "PickupReplay", "Hold", "Carry"}
        if state in {"Preflight", "Align"}:
            object_state = "Targeted"
        elif state == "PickupReplay":
            object_state = "Attached"
        elif state in {"Hold", "Carry"}:
            object_state = "Held"
        else:
            object_state = "Free"

        action = "none"
        carry_command_frame = -1
        root_x = 0.0
        if render_frame == INTERACT_FRAME:
            action = "interact"
        if state == "Carry" and render_frame > 24:
            carry_command_frame = render_frame - 25
            action = "forward"
            root_x = 0.005 * (carry_command_frame + 1)
        if render_frame == len(states) - 1:
            root_x = 0.30

        result = "None"
        reason = "None"
        if state in {"Align", "PickupReplay", "Hold"}:
            result = "Accepted"
        elif state == "Carry":
            result = "Succeeded"

        object_position = [
            0.0 if render_frame == len(states) - 1 else root_x,
            0.75 if render_frame == len(states) - 1 else 1.10,
            3.0 if render_frame == len(states) - 1 else 2.0,
        ]
        grasp_evidence_valid = state in {"PickupReplay", "Hold", "Carry"}
        record = {
            "render_frame": render_frame,
            "runtime_tick": render_frame + 1,
            "scheduler_phase": 0,
            "state": state,
            "result": result,
            "reason": reason,
            "object_state": object_state,
            "attached": attached,
            "owns_pose": owns_pose,
            "carry_mode": "layered" if state == "Carry" else "none",
            "carry_command_frame": carry_command_frame,
            "root_position": [root_x, 0.0, 2.0],
            "object_position": object_position,
            "grasp_evidence_valid": grasp_evidence_valid,
            "active_hand_joint": 22 if grasp_evidence_valid else -1,
            "hand_constraint_weight": 1.0 if grasp_evidence_valid else 0.0,
            "hand_constraint_validated": grasp_evidence_valid,
            "hand_constraint_applied": grasp_evidence_valid,
            "hand_constraint_reachable": grasp_evidence_valid,
            "hand_constraint_used_clavicle": False,
            "hand_constraint_reach_shortfall_m": 0.0,
            "hand_constraint_calibration_rotation": [1.0, 0.0, 0.0, 0.0],
            "calibrated_hand_world_rotation": [1.0, 0.0, 0.0, 0.0],
            "object_world_rotation": [1.0, 0.0, 0.0, 0.0],
            "hand_in_object_position": [0.0, 0.0, 0.0],
            "hand_in_object_rotation": [1.0, 0.0, 0.0, 0.0],
            "grasp_world_position": (
                copy.deepcopy(object_position)
                if grasp_evidence_valid else [0.0, 0.0, 0.0]
            ),
            "grasp_world_rotation": [1.0, 0.0, 0.0, 0.0],
            "root_displacement_m": root_x,
            "joint_world_positions": [
                [root_x, 0.05 * joint, 2.0]
                for joint in range(len(FLAT_JOINT_NAMES))
            ],
            "joint_world_rotations": [
                [1.0, 0.0, 0.0, 0.0] for _ in FLAT_JOINT_NAMES
            ],
            "action": action,
        }
        records.append(record)

    records[-1].update(
        result="Reset",
        reason="Reset",
        action="reset",
        object_state="Free",
        attached=False,
        owns_pose=False,
        carry_mode="none",
        carry_command_frame=-1,
    )
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(_record_line(record) for record in records), encoding="utf-8")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _write_png(path: Path, *, padding: int = 10_500) -> None:
    header = struct.pack(">IIBBBBB", 1280, 720, 8, 2, 0, 0, 0)
    scanline = b"\x00" + b"\x00\x00\x00" * 1280
    image = zlib.compress(scanline * 720)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"tEXt", b"evidence\x00" + b"x" * padding)
        + _png_chunk(b"IDAT", image)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


class EvidenceValidatorUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "pickup.jsonl"
        self.screenshot = self.root / "pickup.png"

    def test_anatomical_translation_limits_accept_exact_boundaries_and_reject_more(self):
        frame = 1
        conservative_joints = (0, 1, 14, 21)
        for joint in conservative_joints:
            with self.subTest(
                boundary="exact-conservative",
                joint=FLAT_JOINT_NAMES[joint],
            ):
                records = _valid_records()
                records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                    records[frame - 1]["joint_world_positions"][joint]
                )
                records[frame]["joint_world_positions"][joint][0] += 0.20
                validate_evidence(records)

            with self.subTest(
                boundary="over-conservative",
                joint=FLAT_JOINT_NAMES[joint],
            ):
                records = _valid_records()
                records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                    records[frame - 1]["joint_world_positions"][joint]
                )
                records[frame]["joint_world_positions"][joint][0] += 0.200001
                with self.assertRaisesRegex(
                    EvidenceValidationError,
                    rf"joint {joint} \({FLAT_JOINT_NAMES[joint]}\).*"
                    rf"frame {frame - 1}->{frame}.*0\.200001.*0\.200000",
                ):
                    validate_evidence(records)

        distal_lower_limb_joints = (4, 5, 8, 9)
        for joint in distal_lower_limb_joints:
            with self.subTest(
                boundary="exact-distal-speed",
                joint=FLAT_JOINT_NAMES[joint],
            ):
                records = _valid_records()
                records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                    records[frame - 1]["joint_world_positions"][joint]
                )
                records[frame]["joint_world_positions"][joint][0] += (
                    12.0 / CONTROL_RATE_HZ
                )
                validate_evidence(records)

            with self.subTest(
                boundary="over-distal-speed",
                joint=FLAT_JOINT_NAMES[joint],
            ):
                records = _valid_records()
                records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                    records[frame - 1]["joint_world_positions"][joint]
                )
                records[frame]["joint_world_positions"][joint][0] += (
                    12.000001 / CONTROL_RATE_HZ
                )
                with self.assertRaisesRegex(
                    EvidenceValidationError,
                    rf"joint {joint} \({FLAT_JOINT_NAMES[joint]}\).*"
                    rf"frame {frame - 1}->{frame}.*"
                    r"12\.000001 m/s.*12\.000000 m/s",
                ):
                    validate_evidence(records)

        _write_records(self.log, _valid_records())
        _write_png(self.screenshot)
        validate_evidence(load_evidence(self.log))
        validate_screenshot(self.screenshot)

    def test_grasp_fields_have_exact_fixed_order(self):
        self.assertEqual(
            EXPECTED_KEYS,
            (
                "render_frame", "runtime_tick", "scheduler_phase", "state",
                "result", "reason", "object_state", "attached", "owns_pose",
                "carry_mode", "carry_command_frame", "root_position",
                "object_position", "grasp_evidence_valid",
                "active_hand_joint", "hand_constraint_weight",
                "hand_constraint_validated", "hand_constraint_applied",
                "hand_constraint_reachable",
                "hand_constraint_used_clavicle",
                "hand_constraint_reach_shortfall_m",
                "hand_constraint_calibration_rotation",
                "calibrated_hand_world_rotation", "object_world_rotation",
                "hand_in_object_position", "hand_in_object_rotation",
                "grasp_world_position", "grasp_world_rotation",
                "root_displacement_m", "joint_world_positions",
                "joint_world_rotations", "action",
            ),
        )

    def test_active_hand_joint_accepts_only_invalid_left_or_right(self):
        record = copy.deepcopy(_valid_records()[36])
        for valid, joint in ((False, -1), (True, 18), (True, 22)):
            with self.subTest(valid=valid, joint=joint):
                candidate = copy.deepcopy(record)
                candidate["grasp_evidence_valid"] = valid
                candidate["active_hand_joint"] = joint
                if not valid:
                    candidate["state"] = "Align"
                    candidate["hand_constraint_weight"] = 0.0
                    candidate["hand_constraint_validated"] = False
                    candidate["hand_constraint_applied"] = False
                    candidate["hand_constraint_reachable"] = False
                else:
                    candidate["joint_world_positions"][joint] = copy.deepcopy(
                        candidate["grasp_world_position"]
                    )
                _validate_record_types(candidate)

        for joint in (True, -2, 0, 17, 19, 23):
            with self.subTest(invalid_joint=joint):
                candidate = copy.deepcopy(record)
                candidate["active_hand_joint"] = joint
                with self.assertRaisesRegex(
                    EvidenceValidationError, "active_hand_joint"
                ):
                    _validate_record_types(candidate)

        for valid, joint in ((False, 22), (True, -1)):
            with self.subTest(invalid_pair=(valid, joint)):
                candidate = copy.deepcopy(record)
                candidate["grasp_evidence_valid"] = valid
                candidate["active_hand_joint"] = joint
                with self.assertRaisesRegex(
                    EvidenceValidationError, "active_hand_joint"
                ):
                    _validate_record_types(candidate)

        for invalid_validity in (0, 1):
            with self.subTest(invalid_validity=invalid_validity):
                candidate = copy.deepcopy(record)
                candidate["grasp_evidence_valid"] = invalid_validity
                with self.assertRaisesRegex(
                    EvidenceValidationError, "grasp_evidence_valid"
                ):
                    _validate_record_types(candidate)

    def test_owned_interaction_states_require_valid_grasp_evidence(self):
        for state in ("PickupReplay", "Hold", "Carry"):
            with self.subTest(state=state):
                record = next(
                    copy.deepcopy(item)
                    for item in _valid_records()
                    if item["state"] == state
                )
                record["grasp_evidence_valid"] = False
                record["active_hand_joint"] = -1
                with self.assertRaisesRegex(
                    EvidenceValidationError,
                    "requires valid selected-grasp evidence",
                ):
                    _validate_record_types(record)

    def test_grasp_vectors_are_finite_and_quaternions_are_unit(self):
        record = copy.deepcopy(_valid_records()[36])
        for field in (
            "hand_in_object_position",
            "grasp_world_position",
        ):
            with self.subTest(field=field, failure="nonfinite"):
                candidate = copy.deepcopy(record)
                candidate[field][1] = math.inf
                with self.assertRaisesRegex(EvidenceValidationError, field):
                    _validate_record_types(candidate)

        for field in (
            "object_world_rotation",
            "hand_in_object_rotation",
            "grasp_world_rotation",
            "hand_constraint_calibration_rotation",
            "calibrated_hand_world_rotation",
        ):
            with self.subTest(field=field, failure="nonfinite"):
                candidate = copy.deepcopy(record)
                candidate[field][2] = math.nan
                with self.assertRaisesRegex(EvidenceValidationError, field):
                    _validate_record_types(candidate)
            with self.subTest(field=field, failure="nonunit"):
                candidate = copy.deepcopy(record)
                candidate[field] = [0.0, 0.0, 0.0, 0.0]
                with self.assertRaisesRegex(
                    EvidenceValidationError, rf"{field}.*quaternion norm"
                ):
                    _validate_record_types(candidate)

    def test_hand_constraint_fields_have_strict_types_and_finite_ranges(self):
        record = copy.deepcopy(_valid_records()[36])
        for field in (
            "hand_constraint_validated",
            "hand_constraint_applied",
            "hand_constraint_reachable",
            "hand_constraint_used_clavicle",
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(record)
                candidate[field] = 1
                with self.assertRaisesRegex(EvidenceValidationError, field):
                    _validate_record_types(candidate)

        for invalid_weight in (True, math.nan, math.inf, -0.000001, 1.000001):
            with self.subTest(invalid_weight=invalid_weight):
                candidate = copy.deepcopy(record)
                candidate["hand_constraint_weight"] = invalid_weight
                with self.assertRaisesRegex(
                    EvidenceValidationError, "hand_constraint_weight"
                ):
                    _validate_record_types(candidate)

        for invalid_shortfall in (True, math.nan, math.inf, -0.000001):
            with self.subTest(invalid_shortfall=invalid_shortfall):
                candidate = copy.deepcopy(record)
                candidate["hand_constraint_reach_shortfall_m"] = invalid_shortfall
                with self.assertRaisesRegex(
                    EvidenceValidationError,
                    "hand_constraint_reach_shortfall_m",
                ):
                    _validate_record_types(candidate)

    def test_calibrated_hand_rotation_matches_final_fk_raw_hand_and_epoch_calibration(self):
        record = copy.deepcopy(_valid_records()[36])
        record["hand_constraint_weight"] = 0.5
        half_sqrt_two = math.sqrt(0.5)
        semantic_angle = math.radians(30.0)
        semantic_rotation = [
            math.cos(0.5 * semantic_angle),
            math.sin(0.5 * semantic_angle),
            0.0,
            0.0,
        ]
        calibration_rotation = [half_sqrt_two, 0.0, 0.0, half_sqrt_two]
        raw_hand_rotation = _quaternion_multiply(
            semantic_rotation, calibration_rotation
        )
        joint = record["active_hand_joint"]
        record["joint_world_rotations"][joint] = raw_hand_rotation
        record["hand_constraint_calibration_rotation"] = calibration_rotation
        record["calibrated_hand_world_rotation"] = [-v for v in semantic_rotation]
        _validate_record_types(record)

        record["calibrated_hand_world_rotation"] = [1.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            EvidenceValidationError,
            "calibrated_hand_world_rotation.*final rendered hand",
        ):
            _validate_record_types(record)

    def test_positive_weight_grasp_requires_validated_applied_reachable_solution(self):
        record = copy.deepcopy(_valid_records()[36])
        record["hand_constraint_weight"] = 0.5
        for field in (
            "hand_constraint_validated",
            "hand_constraint_applied",
            "hand_constraint_reachable",
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(record)
                candidate[field] = False
                with self.assertRaisesRegex(EvidenceValidationError, field):
                    _validate_record_types(candidate)

        zero_weight = copy.deepcopy(record)
        zero_weight["hand_constraint_weight"] = 0.0
        zero_weight["hand_constraint_validated"] = False
        zero_weight["hand_constraint_applied"] = False
        zero_weight["hand_constraint_reachable"] = False
        _validate_record_types(zero_weight)

    def test_unreachable_positive_weight_is_an_explicit_gate_failure(self):
        record = copy.deepcopy(_valid_records()[36])
        record["hand_constraint_weight"] = 1.0
        record["hand_constraint_reachable"] = False
        record["hand_constraint_reach_shortfall_m"] = 0.000001
        with self.assertRaisesRegex(
            EvidenceValidationError,
            "hand_constraint_reachable",
        ):
            _validate_record_types(record)

    def test_attached_full_weight_position_limit_accepts_exact_boundary_only(self):
        record = copy.deepcopy(_valid_records()[36])
        joint = record["active_hand_joint"]
        grasp = record["grasp_world_position"]
        record["joint_world_positions"][joint] = [
            grasp[0] + HAND_POSITION_LIMIT_M,
            grasp[1],
            grasp[2],
        ]
        _validate_record_types(record)

        record["joint_world_positions"][joint][0] = (
            grasp[0] + HAND_POSITION_LIMIT_M + 0.000001
        )
        with self.assertRaisesRegex(
            EvidenceValidationError,
            r"hand position.*0\.010001.*0\.010000",
        ):
            _validate_record_types(record)

    def test_attached_full_weight_calibrated_orientation_limit_accepts_exact_boundary_only(self):
        record = copy.deepcopy(_valid_records()[36])
        joint = record["active_hand_joint"]

        def set_error_degrees(error_degrees):
            angle = math.radians(error_degrees)
            rotation = [
                math.cos(0.5 * angle),
                math.sin(0.5 * angle),
                0.0,
                0.0,
            ]
            record["joint_world_rotations"][joint] = rotation
            record["calibrated_hand_world_rotation"] = rotation

        set_error_degrees(HAND_CALIBRATED_ORIENTATION_LIMIT_DEGREES)
        _validate_record_types(record)

        set_error_degrees(
            HAND_CALIBRATED_ORIENTATION_LIMIT_DEGREES + 0.000001
        )
        with self.assertRaisesRegex(
            EvidenceValidationError,
            r"calibrated hand orientation.*2\.000001.*2\.000000",
        ):
            _validate_record_types(record)

    def test_grasp_world_is_exact_object_hand_composition_up_to_quaternion_sign(self):
        record = copy.deepcopy(_valid_records()[36])
        record["hand_constraint_weight"] = 0.0
        record["hand_constraint_validated"] = False
        record["hand_constraint_applied"] = False
        record["hand_constraint_reachable"] = False
        half_sqrt_two = math.sqrt(0.5)
        record.update(
            object_position=[1.0, 2.0, 3.0],
            object_world_rotation=[half_sqrt_two, 0.0, 0.0, half_sqrt_two],
            hand_in_object_position=[1.0, 0.0, 0.0],
            hand_in_object_rotation=[half_sqrt_two, half_sqrt_two, 0.0, 0.0],
            grasp_world_position=[1.0, 3.0, 3.0],
            grasp_world_rotation=[-0.5, -0.5, -0.5, -0.5],
        )
        _validate_record_types(record)

        for field, component in (
            ("grasp_world_position", 0),
            ("grasp_world_rotation", 1),
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(record)
                candidate[field][component] += 0.01
                if field.endswith("rotation"):
                    norm = math.sqrt(sum(value * value for value in candidate[field]))
                    candidate[field] = [value / norm for value in candidate[field]]
                with self.assertRaisesRegex(
                    EvidenceValidationError, "object_world.*hand_in_object"
                ):
                    _validate_record_types(candidate)

    def test_grasp_alignment_summary_is_deterministic_by_state(self):
        summary_function = globals().get("summarize_grasp_alignment")
        self.assertIsNotNone(
            summary_function,
            "grasp alignment summary function is missing",
        )
        if summary_function is None:
            return

        records = []
        angle = math.radians(60.0)
        calibration_angle = math.radians(40.0)
        calibrated_angle = math.radians(20.0)
        for state_index, state in enumerate(("PickupReplay", "Hold", "Carry")):
            for sample, distance in enumerate((1.0, 3.0)):
                record = copy.deepcopy(_valid_records()[36])
                record["render_frame"] = 10 * state_index + sample
                record["state"] = state
                record["active_hand_joint"] = 18 if state == "Hold" else 22
                joint = record["active_hand_joint"]
                record["grasp_world_position"] = [0.0, 0.0, 0.0]
                record["joint_world_positions"][joint] = [distance, 0.0, 0.0]
                record["grasp_world_rotation"] = [1.0, 0.0, 0.0, 0.0]
                record["joint_world_rotations"][joint] = (
                    [-1.0, 0.0, 0.0, 0.0]
                    if sample == 0 else
                    [math.cos(0.5 * angle), math.sin(0.5 * angle), 0.0, 0.0]
                )
                record["hand_constraint_calibration_rotation"] = (
                    [1.0, 0.0, 0.0, 0.0]
                    if sample == 0 else
                    [
                        math.cos(0.5 * calibration_angle),
                        math.sin(0.5 * calibration_angle),
                        0.0,
                        0.0,
                    ]
                )
                record["calibrated_hand_world_rotation"] = (
                    [-1.0, 0.0, 0.0, 0.0]
                    if sample == 0 else
                    [
                        math.cos(0.5 * calibrated_angle),
                        math.sin(0.5 * calibrated_angle),
                        0.0,
                        0.0,
                    ]
                )
                records.append(record)

        summaries = summary_function(records)
        self.assertEqual(
            [summary["state"] for summary in summaries],
            ["PickupReplay", "Hold", "Carry"],
        )
        for state_index, summary in enumerate(summaries):
            self.assertEqual(summary["count"], 2)
            self.assertAlmostEqual(summary["position_mean_m"], 2.0)
            self.assertAlmostEqual(summary["position_max_m"], 3.0)
            self.assertEqual(summary["position_max_frame"], 10 * state_index + 1)
            self.assertAlmostEqual(summary["orientation_mean_degrees"], 30.0)
            self.assertAlmostEqual(summary["orientation_max_degrees"], 60.0)
            self.assertEqual(
                summary["orientation_max_frame"], 10 * state_index + 1
            )
            self.assertAlmostEqual(
                summary["calibrated_orientation_mean_degrees"], 10.0
            )
            self.assertAlmostEqual(
                summary["calibrated_orientation_max_degrees"], 20.0
            )
            self.assertEqual(
                summary["calibrated_orientation_max_frame"],
                10 * state_index + 1,
            )
            expected_joint = "LeftHand" if summary["state"] == "Hold" else "RightHand"
            self.assertEqual(summary["active_hand_joint_name"], expected_joint)

    def test_synthetic_fixture_advances_runtime_once_per_25_hz_frame(self):
        records = _valid_records()
        for previous, record in zip(records, records[1:]):
            self.assertEqual(record["runtime_tick"], previous["runtime_tick"] + 1)
            self.assertEqual(record["scheduler_phase"], 0)

    def test_skipped_or_duplicate_25_hz_runtime_ticks_are_rejected(self):
        for runtime_tick in (1, 4):
            with self.subTest(runtime_tick=runtime_tick):
                records = _valid_records()
                records[2]["runtime_tick"] = runtime_tick
                _write_records(self.log, records)
                with self.assertRaisesRegex(
                    EvidenceValidationError, "advance exactly once"
                ):
                    validate_evidence(load_evidence(self.log))

    def test_nonzero_scheduler_phase_is_rejected(self):
        records = _valid_records()
        records[2]["scheduler_phase"] = 1
        with self.assertRaisesRegex(EvidenceValidationError, "scheduler_phase"):
            validate_evidence(records)

    def test_unsuccessful_runtime_outputs_after_interact_are_rejected(self):
        failure_frame = INTERACT_FRAME + 1
        for field, value in (
            ("result", "Rejected"),
            ("result", "Cancelled"),
            ("result", "Failed"),
            ("state", "Disabled"),
        ):
            with self.subTest(field=field, value=value):
                records = _valid_records()
                records[failure_frame][field] = value
                _write_records(self.log, records)
                with self.assertRaises(EvidenceValidationError):
                    validate_evidence(load_evidence(self.log))

    def test_every_carry_row_is_held_and_owns_its_pose(self):
        for field, value in (("object_state", "Free"), ("owns_pose", False)):
            with self.subTest(field=field):
                records = _valid_records()
                for record in records:
                    if record["state"] == "Carry":
                        record[field] = value
                _write_records(self.log, records)
                with self.assertRaises(EvidenceValidationError):
                    validate_evidence(load_evidence(self.log))

    def test_first_carry_none_cannot_bypass_layered_head_seam_gate(self):
        records = _valid_records()
        first_carry = next(
            index
            for index, record in enumerate(records)
            if record["state"] == "Carry"
        )
        records[first_carry]["carry_mode"] = "none"
        records[first_carry]["joint_world_positions"][14] = copy.deepcopy(
            records[first_carry - 1]["joint_world_positions"][14]
        )
        records[first_carry]["joint_world_positions"][14][0] += 0.19

        with self.assertRaisesRegex(
            EvidenceValidationError,
            r"Carry frame 24.*carry_mode.*recorded or layered",
        ):
            validate_evidence(records)

    def test_layered_inactive_hand_elevation_accepts_exact_boundary_only(self):
        for active_hand_joint in (18, 22):
            with self.subTest(active_hand_joint=active_hand_joint, boundary="exact"):
                records = _valid_records()
                if active_hand_joint == 18:
                    for record in records:
                        record["joint_world_positions"][18][1] = 1.10
                        if record["grasp_evidence_valid"]:
                            record["active_hand_joint"] = 18
                            record["joint_world_positions"][18] = copy.deepcopy(
                                record["grasp_world_position"]
                            )
                inactive_arm_joint, inactive_hand_joint = (
                    (20, 22) if active_hand_joint == 18 else (16, 18)
                )
                for record in records:
                    if record["state"] == "Carry":
                        arm_y = record["joint_world_positions"][inactive_arm_joint][1]
                        record["joint_world_positions"][inactive_hand_joint][1] = (
                            arm_y + LAYERED_INACTIVE_HAND_ELEVATION_LIMIT_M
                        )
                validate_evidence(records)

            with self.subTest(active_hand_joint=active_hand_joint, boundary="epsilon"):
                violating = copy.deepcopy(records)
                carry = next(
                    record for record in violating if record["state"] == "Carry"
                )
                carry["joint_world_positions"][inactive_hand_joint][1] += 0.000001
                with self.assertRaisesRegex(
                    EvidenceValidationError, "layered inactive hand"
                ):
                    validate_evidence(violating)

    def test_recorded_carry_excludes_inactive_hand_elevation_cap(self):
        records = _valid_records()
        for record in records:
            if record["state"] == "Carry":
                record["carry_mode"] = "recorded"
            record["joint_world_positions"][18][1] = (
                record["joint_world_positions"][16][1] + 0.50
            )
        validate_evidence(records)

    def test_carry_must_be_observed_by_evidence_frame_375(self):
        records = _valid_records()
        first_carry = next(
            index for index, record in enumerate(records) if record["state"] == "Carry"
        )
        delayed_carry_frame = CARRY_DEADLINE_FRAMES + 2
        hold = records[first_carry - 1]
        records[first_carry:first_carry] = [
            copy.deepcopy(hold) for _ in range(delayed_carry_frame - first_carry)
        ]
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(
            EvidenceValidationError, f"frame {CARRY_DEADLINE_FRAMES}"
        ):
            validate_evidence(load_evidence(self.log))

    def test_evidence_is_bounded_to_at_most_500_records(self):
        records = _valid_records()
        first_forward = next(
            index
            for index, record in enumerate(records)
            if record["carry_command_frame"] == 0
        )
        waiting_carry = records[first_forward - 1]
        records[first_forward:first_forward] = [
            copy.deepcopy(waiting_carry)
            for _ in range(MAX_EVIDENCE_RECORDS + 1 - len(records))
        ]
        _reclock_records(records)
        self.assertEqual(len(records), MAX_EVIDENCE_RECORDS + 1)
        _write_records(self.log, records)
        with self.assertRaisesRegex(
            EvidenceValidationError, f"at most {MAX_EVIDENCE_RECORDS}"
        ):
            validate_evidence(load_evidence(self.log))

    def test_reset_action_appears_only_on_the_final_record(self):
        records = _valid_records()
        records[10]["action"] = "reset"
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "Reset action"):
            validate_evidence(load_evidence(self.log))

    def test_interact_action_appears_only_once_on_frame_13(self):
        records = _valid_records()
        records[10]["action"] = "interact"
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "Interact"):
            validate_evidence(load_evidence(self.log))

    def test_forward_command_zero_immediately_follows_first_carry(self):
        records = _valid_records()
        first_carry = next(
            index for index, record in enumerate(records) if record["state"] == "Carry"
        )
        records[first_carry + 1:first_carry + 1] = [
            copy.deepcopy(records[first_carry]) for _ in range(12)
        ]
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "immediately follow"):
            validate_evidence(load_evidence(self.log))

    def test_forward_commands_zero_through_62_are_consecutive(self):
        records = _valid_records()
        command_10 = next(
            index
            for index, record in enumerate(records)
            if record["carry_command_frame"] == 10
        )
        wait = copy.deepcopy(records[command_10])
        wait.update(action="none", carry_command_frame=-1)
        records[command_10 + 1:command_10 + 1] = [
            copy.deepcopy(wait) for _ in range(12)
        ]
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(
            EvidenceValidationError,
            f"0 through {FINAL_CARRY_COMMAND}",
        ):
            validate_evidence(load_evidence(self.log))

    def test_no_wait_may_precede_final_command_62(self):
        records = _valid_records()
        command_before_final = next(
            index
            for index, record in enumerate(records)
            if record["carry_command_frame"] == FINAL_CARRY_COMMAND - 1
        )
        wait = copy.deepcopy(records[command_before_final])
        wait.update(action="none", carry_command_frame=-1)
        records.insert(command_before_final + 1, wait)
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "must be consecutive"):
            validate_evidence(load_evidence(self.log))

    def test_reset_immediately_follows_command_62(self):
        records = _valid_records()
        wait = copy.deepcopy(records[-1])
        wait.update(action="none", result="None", reason="None")
        records.insert(len(records) - 1, wait)
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "immediately follow"):
            validate_evidence(load_evidence(self.log))

    def test_carry_mode_may_fall_back_before_the_final_carry_record(self):
        records = _valid_records()
        for record in records:
            if record["state"] == "Carry" and record["carry_command_frame"] < 25:
                record["carry_mode"] = "recorded"
        _write_records(self.log, records)
        validate_evidence(load_evidence(self.log))

    def test_malformed_json_is_rejected(self):
        self.log.write_text("{not json}\n", encoding="utf-8")
        with self.assertRaises(EvidenceValidationError):
            load_evidence(self.log)

    def test_missing_and_extra_fields_are_rejected(self):
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                record = copy.deepcopy(_valid_records()[0])
                if mode == "missing":
                    record.pop("action")
                else:
                    record["unexpected"] = 1
                self.log.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaises(EvidenceValidationError):
                    load_evidence(self.log)

    def test_nonfinite_and_wrong_typed_values_are_rejected(self):
        records = _valid_records()
        records[0]["root_position"][0] = math.nan
        self.log.write_text(
            _record_line(records[0]).replace("nan", "NaN"),
            encoding="utf-8",
        )
        with self.assertRaises(EvidenceValidationError):
            load_evidence(self.log)

        record = copy.deepcopy(_valid_records()[0])
        record["render_frame"] = True
        self.log.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(EvidenceValidationError):
            load_evidence(self.log)

    def test_nonsequential_render_frames_are_rejected(self):
        records = _valid_records()
        records[12]["render_frame"] = 13
        _write_records(self.log, records)
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(load_evidence(self.log))

    def test_adjacent_nonowned_root_jump_over_point_two_is_rejected(self):
        records = _valid_records()
        self.assertFalse(records[INTERACT_FRAME - 1]["owns_pose"])
        self.assertFalse(records[INTERACT_FRAME]["owns_pose"])
        self.assertEqual(records[INTERACT_FRAME]["action"], "interact")
        records[INTERACT_FRAME]["root_position"][0] = 2.433498
        _write_records(self.log, records)
        with self.assertRaisesRegex(
            EvidenceValidationError,
            "adjacent non-owned root step must not exceed 0.20 m",
        ):
            validate_evidence(load_evidence(self.log))

    def test_joint_world_arrays_require_exactly_23_joints(self):
        for field in ("joint_world_positions", "joint_world_rotations"):
            for length in (22, 24):
                with self.subTest(field=field, length=length):
                    records = _valid_records()
                    records[0][field] = records[0][field][:length]
                    if length == 24:
                        records[0][field].append(copy.deepcopy(records[0][field][-1]))
                    with self.assertRaisesRegex(
                        EvidenceValidationError,
                        rf"{field}.*exactly 23",
                    ):
                        validate_evidence(records)

    def test_nonfinite_joint_world_component_is_rejected_with_joint_name(self):
        records = _valid_records()
        records[0]["joint_world_positions"][14][1] = math.nan
        with self.assertRaisesRegex(
            EvidenceValidationError,
            r"frame 0 joint 14 \(Head\).*finite",
        ):
            validate_evidence(records)

    def test_zero_joint_world_quaternion_is_rejected_with_joint_name(self):
        records = _valid_records()
        records[0]["joint_world_rotations"][13] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            EvidenceValidationError,
            r"frame 0 joint 13 \(Neck\).*quaternion norm",
        ):
            validate_evidence(records)

    def test_point_200001_joint_jump_is_rejected_at_every_authority_seam(self):
        records = _valid_records()
        seams = [
            (
                f"state-{records[frame - 1]['state']}-to-{record['state']}",
                frame,
                {},
            )
            for frame, record in enumerate(records[1:], start=1)
            if record["state"] != records[frame - 1]["state"]
        ]
        seams.extend(
            (
                ("owns-pose-only", 1, {"owns_pose": True}),
                ("attached-only", 1, {"attached": True}),
            )
        )
        for seam, frame, mutations in seams:
            for joint in (5, 21):
                with self.subTest(
                    seam=seam,
                    joint=FLAT_JOINT_NAMES[joint],
                ):
                    records = _valid_records()
                    records[frame].update(mutations)
                    records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                        records[frame - 1]["joint_world_positions"][joint]
                    )
                    records[frame]["joint_world_positions"][joint][0] += 0.200001
                    with self.assertRaisesRegex(
                        EvidenceValidationError,
                        rf"joint {joint} \({FLAT_JOINT_NAMES[joint]}\).*"
                        rf"frame {frame - 1}->{frame}.*"
                        r"0\.200001.*0\.200000",
                    ):
                        validate_evidence(records)

    def test_hold_to_layered_carry_axial_seam_has_tighter_translation_bound(self):
        template = _valid_records()
        frame = next(
            index
            for index, record in enumerate(template)
            if record["state"] == "Carry"
            and record["carry_mode"] == "layered"
            and template[index - 1]["state"] == "Hold"
        )
        for joint in (12, 13, 14, 15, 19):
            with self.subTest(
                boundary="exact",
                joint=FLAT_JOINT_NAMES[joint],
            ):
                records = _valid_records()
                records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                    records[frame - 1]["joint_world_positions"][joint]
                )
                records[frame]["joint_world_positions"][joint][0] += 0.05
                validate_evidence(records)

            with self.subTest(
                boundary="epsilon-over",
                joint=FLAT_JOINT_NAMES[joint],
            ):
                records = _valid_records()
                records[frame]["joint_world_positions"][joint] = copy.deepcopy(
                    records[frame - 1]["joint_world_positions"][joint]
                )
                records[frame]["joint_world_positions"][joint][0] += 0.050001
                with self.assertRaisesRegex(
                    EvidenceValidationError,
                    rf"joint {joint} \({FLAT_JOINT_NAMES[joint]}\).*"
                    rf"frame {frame - 1}->{frame}.*Hold-to-layered Carry.*"
                    r"0\.050001.*0\.050000",
                ):
                    validate_evidence(records)

    def test_60_001_degree_joint_jump_is_rejected_at_reset(self):
        records = _valid_records()
        frame = len(records) - 1
        angle = math.radians(60.001)
        records[frame]["joint_world_rotations"][13] = [
            math.cos(0.5 * angle),
            math.sin(0.5 * angle),
            0.0,
            0.0,
        ]
        with self.assertRaisesRegex(
            EvidenceValidationError,
            rf"joint 13 \(Neck\).*frame {frame - 1}->{frame}.*"
            r"60\.001000.*60\.000000",
        ):
            validate_evidence(records)

    def test_illegal_state_order_is_rejected(self):
        records = _valid_records()
        records[34]["state"] = "PickupReplay"
        _write_records(self.log, records)
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(load_evidence(self.log))

    def test_missing_attachment_is_rejected(self):
        records = _valid_records()
        for record in records:
            record["attached"] = False
        _write_records(self.log, records)
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(load_evidence(self.log))

    def test_forward_count_must_be_exactly_63(self):
        records = _valid_records()
        records[80]["action"] = "none"
        records[80]["carry_command_frame"] = -1
        _write_records(self.log, records)
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(load_evidence(self.log))

    def test_final_carry_displacement_must_exceed_point_two_metres(self):
        records = _valid_records()
        for record in records:
            command = record["carry_command_frame"]
            if command >= 0:
                displacement = 0.20 * (command + 1) / CARRY_COMMAND_COUNT
                record["root_position"][0] = displacement
                record["object_position"][0] = displacement
                record["grasp_world_position"][0] = displacement
                record["root_displacement_m"] = displacement
        _write_records(self.log, records)
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(load_evidence(self.log))

    def test_final_carry_object_must_move_with_the_character(self):
        records = _valid_records()
        first_carry_object = next(
            copy.deepcopy(record["object_position"])
            for record in records
            if record["state"] == "Carry"
        )
        for record in records:
            if record["state"] == "Carry":
                record["object_position"] = copy.deepcopy(first_carry_object)
                record["grasp_world_position"] = copy.deepcopy(
                    first_carry_object
                )
                record["joint_world_positions"][
                    record["active_hand_joint"]
                ] = copy.deepcopy(first_carry_object)
        records[-1]["joint_world_positions"][22] = copy.deepcopy(
            first_carry_object
        )
        _write_records(self.log, records)
        with self.assertRaisesRegex(
            EvidenceValidationError,
            "final Carry object horizontal displacement must exceed 0.20 m",
        ):
            validate_evidence(load_evidence(self.log))

    def test_missing_reset_is_rejected(self):
        records = _valid_records()
        records[-1].update(action="none", result="None", reason="None")
        _write_records(self.log, records)
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(load_evidence(self.log))

    def test_screenshot_must_be_larger_than_10000_bytes(self):
        self.screenshot.write_bytes(b"x" * 10_000)
        with self.assertRaises(EvidenceValidationError):
            validate_screenshot(self.screenshot)


class Task12PolicyTests(unittest.TestCase):
    longMessage = False

    @staticmethod
    def _make_rule_block(makefile: str, target: str) -> str:
        lines = makefile.splitlines()
        start = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith(f"{target}:")
            ),
            None,
        )
        if start is None:
            raise AssertionError(f"Makefile has no {target} rule")
        stop = start + 1
        while stop < len(lines):
            line = lines[stop]
            if line and not line[0].isspace() and ":" in line:
                break
            stop += 1
        return "\n".join(lines[start:stop])

    @classmethod
    def _make_prerequisites(cls, makefile: str, target: str) -> set[str]:
        block = cls._make_rule_block(makefile, target)
        header = []
        for line in block.splitlines():
            if line.startswith("\t"):
                break
            header.append(line.rstrip("\\").strip())
        return set(" ".join(header).split(":", 1)[1].split())

    @staticmethod
    def _source_between(source: str, start: str, stop: str) -> str:
        try:
            begin = source.index(start)
            end = source.index(stop, begin)
        except ValueError as error:
            raise AssertionError(
                f"controller source block markers are missing: {start!r}, {stop!r}"
            ) from error
        return source[begin:end]

    @staticmethod
    def _cpp_function(source: str, signature: str) -> str:
        try:
            start = source.index(signature)
            opening = source.index("{", start)
        except ValueError as error:
            raise AssertionError(
                f"controller function is missing: {signature}"
            ) from error
        depth = 0
        for index in range(opening, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start:index + 1]
        raise AssertionError(f"controller function is unterminated: {signature}")

    @staticmethod
    def _cpp_call_end(source: str, call_start: int) -> int:
        try:
            opening = source.index("(", call_start)
        except ValueError as error:
            raise AssertionError("C++ call has no opening parenthesis") from error
        depth = 0
        quote = None
        escaped = False
        line_comment = False
        block_comment = False
        index = opening
        while index < len(source):
            character = source[index]
            following = source[index + 1] if index + 1 < len(source) else ""
            if line_comment:
                if character == "\n":
                    line_comment = False
                index += 1
                continue
            if block_comment:
                if character == "*" and following == "/":
                    block_comment = False
                    index += 2
                else:
                    index += 1
                continue
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                index += 1
                continue
            if character == "/" and following == "/":
                line_comment = True
                index += 2
                continue
            if character == "/" and following == "*":
                block_comment = True
                index += 2
                continue
            if character in ('"', "'"):
                quote = character
                index += 1
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    semicolon = index + 1
                    while (
                        semicolon < len(source)
                        and source[semicolon].isspace()
                    ):
                        semicolon += 1
                    if semicolon >= len(source) or source[semicolon] != ";":
                        raise AssertionError(
                            "C++ call is not terminated by a semicolon"
                        )
                    return semicolon + 1
            index += 1
        raise AssertionError("C++ call has unterminated parentheses")

    @staticmethod
    def _first_if_scope(source: str) -> tuple[str, str]:
        match = re.search(r"\bif\s*\(", source)
        if match is None:
            raise AssertionError("source block has no enclosing if guard")
        condition_open = source.index("(", match.start())
        parenthesis_depth = 0
        condition_close = None
        for index in range(condition_open, len(source)):
            if source[index] == "(":
                parenthesis_depth += 1
            elif source[index] == ")":
                parenthesis_depth -= 1
                if parenthesis_depth == 0:
                    condition_close = index
                    break
        if condition_close is None:
            raise AssertionError("if guard has an unterminated condition")
        scope_open = condition_close + 1
        while scope_open < len(source) and source[scope_open].isspace():
            scope_open += 1
        if scope_open >= len(source) or source[scope_open] != "{":
            raise AssertionError("auto-demo exclusion guard must use a braced scope")
        brace_depth = 0
        for index in range(scope_open, len(source)):
            if source[index] == "{":
                brace_depth += 1
            elif source[index] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return (
                        source[condition_open + 1:condition_close],
                        source[match.start():index + 1],
                    )
        raise AssertionError("if guard has an unterminated braced scope")

    @staticmethod
    def _strip_outer_cpp_parentheses(expression: str) -> str:
        expression = expression.strip()
        while expression.startswith("(") and expression.endswith(")"):
            depth = 0
            enclosing_close = None
            for index, character in enumerate(expression):
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth < 0:
                        raise AssertionError(
                            "boolean guard has unbalanced parentheses"
                        )
                    if depth == 0:
                        enclosing_close = index
                        break
            if enclosing_close != len(expression) - 1:
                break
            expression = expression[1:-1].strip()
        return expression

    @classmethod
    def _split_top_level_cpp_boolean(
        cls,
        expression: str,
        operator: str,
    ) -> list[str]:
        expression = cls._strip_outer_cpp_parentheses(expression)
        parts = []
        start = 0
        depth = 0
        index = 0
        while index < len(expression):
            character = expression[index]
            if character == "(":
                depth += 1
                index += 1
                continue
            if character == ")":
                depth -= 1
                if depth < 0:
                    raise AssertionError(
                        "boolean guard has unbalanced parentheses"
                    )
                index += 1
                continue
            if depth == 0 and expression.startswith(operator, index):
                part = expression[start:index].strip()
                if not part:
                    raise AssertionError(
                        f"boolean guard has an empty {operator!r} operand"
                    )
                parts.append(part)
                index += len(operator)
                start = index
                continue
            index += 1
        if depth != 0:
            raise AssertionError("boolean guard has unbalanced parentheses")
        tail = expression[start:].strip()
        if not tail:
            raise AssertionError(
                f"boolean guard has an empty {operator!r} operand"
            )
        parts.append(tail)
        return parts

    @classmethod
    def _guard_excludes_legacy_autodemos(
        cls,
        condition: str,
        legacy_aliases: tuple[str, ...] = (),
    ) -> bool:
        compact = re.sub(r"\s+", "", condition)
        if len(cls._split_top_level_cpp_boolean(compact, "||")) != 1:
            return False
        conjuncts = cls._split_top_level_cpp_boolean(compact, "&&")
        atoms = {
            cls._strip_outer_cpp_parentheses(conjunct)
            for conjunct in conjuncts
        }
        if "!autodemo_configuration.has_value()" in atoms:
            return True
        if any(f"!{alias}" in atoms for alias in legacy_aliases):
            return True
        if {
            "!pickup_autodemo_enabled",
            "!placement_autodemo_enabled",
        }.issubset(atoms):
            return True
        return any(
            negated_pair in atoms
            for negated_pair in (
                "!(pickup_autodemo_enabled||placement_autodemo_enabled)",
                "!(placement_autodemo_enabled||pickup_autodemo_enabled)",
            )
        )

    def test_safe_query_probe_build_output_is_ignored(self):
        completed = subprocess.run(
            [
                "git",
                "check-ignore",
                "-q",
                "build/task12/interaction_query_probe_safe",
            ],
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "Task 12 safe build output must be ignored",
        )

    def test_gate_uses_only_safe_query_probe_and_safe_suite(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "build/task12/interaction_query_probe_safe",
            makefile,
            "Makefile must name the safe query-probe output",
        )
        safe_prerequisites = self._make_prerequisites(
            makefile, "test-python-interaction-safe"
        )
        self.assertIn(
            "$(SAFE_INTERACTION_QUERY_PROBE)",
            safe_prerequisites,
            "safe Python suite must depend on the safe query probe",
        )
        gate_prerequisites = self._make_prerequisites(
            makefile, "gate-playable-interaction"
        )
        self.assertIn(
            "test-interaction-safe",
            gate_prerequisites,
            "playable gate must depend on the safe interaction suite",
        )
        self.assertTrue(
            {"test-interaction", "test-python", "interaction_query_probe"}.isdisjoint(
                gate_prerequisites
            ),
            "playable gate must not depend on an unsafe aggregate or root probe",
        )
        safe_suite = self._make_rule_block(
            makefile, "test-python-interaction-safe"
        )
        self.assertIn(
            "MM_INTERACTION_QUERY_PROBE",
            safe_suite,
            "safe Python suite must override the parity probe path",
        )
        gate = self._make_rule_block(makefile, "gate-playable-interaction")
        self.assertNotRegex(
            gate,
            r"(?<![-\w])test-(?:python|interaction)(?![-\w])",
            "playable gate recipe must not invoke an unsafe aggregate",
        )
        self.assertNotRegex(
            gate,
            r"(?:^|[\s\"'])\./interaction_query_probe(?:$|[\s\"'])",
            "playable gate recipe must not invoke the repository-root query probe",
        )

    def test_gate_serializes_bootstrap_and_controller_before_evidence(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        prerequisites = self._make_prerequisites(
            makefile, "gate-playable-interaction"
        )
        self.assertTrue(
            {"bootstrap-raylib", "controller"}.isdisjoint(prerequisites),
            "bootstrap and controller must not race as sibling prerequisites",
        )
        self.assertTrue(
            {
                "test-interaction-safe",
                "demo-interaction-pack",
                "interaction_probe",
                "interaction_runtime_probe",
            }.issubset(prerequisites),
            "gate must retain its safe suite, pack, and probe prerequisites",
        )

        gate = self._make_rule_block(makefile, "gate-playable-interaction")
        recipe = [
            line.strip()
            for line in gate.splitlines()
            if line.startswith("\t")
        ]
        self.assertGreaterEqual(len(recipe), 3)
        self.assertEqual(
            recipe[:3],
            [
                "$(MAKE) bootstrap-raylib",
                "$(MAKE) controller",
                'mkdir -p "$(PLAYABLE_EVIDENCE_DIR)"',
            ],
            "recursive bootstrap and controller builds must run sequentially "
            "before evidence setup, including under make -j",
        )

    def test_safe_query_probe_path_cannot_be_redirected_by_make_cli(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        assignment = (
            "override SAFE_INTERACTION_QUERY_PROBE := "
            "build/task12/interaction_query_probe_safe"
        )
        self.assertEqual(
            makefile.splitlines().count(assignment),
            1,
            "safe query-probe output must have one exact override assignment",
        )

    def test_safe_python_suite_passes_g1_xml_to_discovery(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        safe_suite = self._make_rule_block(
            makefile, "test-python-interaction-safe"
        )
        self.assertIn(
            'G1_XML="$(G1_XML)"',
            safe_suite,
            "safe Python discovery must receive the configured G1_XML path",
        )

    def test_readme_documents_safe_probe_and_evidence_path_preconditions(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        for variable in (
            "MM_INTERACTION_QUERY_PROBE",
            "PLAYABLE_LOG",
            "PLAYABLE_SCREENSHOT",
        ):
            with self.subTest(variable=variable):
                self.assertRegex(
                    readme,
                    rf"(?<![A-Z0-9_]){re.escape(variable)}(?![A-Z0-9_])",
                    f"README must document {variable}",
                )
        lowered = readme.lower()
        for pattern, requirement in (
            (r"non[- ]?empty", "nonempty evidence paths"),
            (r"\bdistinct\b", "distinct evidence paths"),
            (r"existing parent director(?:y|ies)", "existing parent directories"),
        ):
            with self.subTest(requirement=requirement):
                self.assertRegex(
                    lowered,
                    pattern,
                    f"README must document {requirement}",
                )

    def test_controller_retains_fixed_clock_and_autodemo_boundaries(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "SetTargetFPS(25);", controller, "controller must use fixed 25 Hz"
        )
        self.assertNotIn("SetTargetFPS(60);", controller)
        self.assertIn(
            "ControllerInteractionScheduler",
            controller,
            "controller must retain the Task 11 scheduler",
        )
        self.assertIn(
            "interaction_scheduler.tick(",
            controller,
            "controller must tick through the Task 11 scheduler",
        )
        self.assertIn(
            "MM_INTERACTION_AUTODEMO",
            controller,
            "controller must expose the deterministic autodemo seam",
        )
        self.assertIn(
            "MM_FEATURES_OUTPUT",
            controller,
            "controller must isolate its feature-cache output",
        )
        self.assertNotIn(
            "GetFrameTime(", controller, "controller must not use wall-clock dt"
        )
        self.assertNotIn(
            "runtime_accumulator",
            controller,
            "controller must not reintroduce accumulator scheduling",
        )
        self.assertNotIn(
            "const float interaction_scene_alpha =",
            controller,
            "25 Hz scene publication must not add render interpolation lag",
        )

    def test_keyboard_f_x_edges_remain_at_the_input_boundary(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        raw_input = self._source_between(
            controller,
            "        // Press edges and runtime updates share the fixed 25 Hz controller",
            "        if (pickup_autodemo_enabled)",
        )
        for key, edge in (
            ("KEY_F", "interact_pressed"),
            ("KEY_X", "cancel_pressed"),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    controller.count(f"IsKeyPressed({key})"),
                    1,
                    f"{key} must be sampled exactly once at the input boundary",
                )
                self.assertIn(f"IsKeyPressed({key})", raw_input)
                self.assertNotIn(
                    f"IsKeyPressed({key})",
                    self._source_between(
                        controller,
                        "        // Manual pick-assist input begins.",
                        "        // Manual pick-assist input ends.",
                    ),
                    f"manual Smart Pickup must consume {edge}, not poll {key}",
                )
        self.assertRegex(
            raw_input,
            r"ControllerInteractionEdges\s+interaction_edges\s*\{\s*"
            r"IsKeyPressed\(KEY_F\)[\s\S]*?IsKeyPressed\(KEY_X\)",
            "keyboard F/X must remain input-level interact/cancel edges",
        )

    def test_manual_smart_pickup_activation_uses_only_the_25_hz_clock(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        timing = Path("locomotion_timing.h").read_text(encoding="utf-8")
        adapter = Path("interaction_controller_adapter.h").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            timing,
            r"kStepSeconds\s*=\s*1\.0F\s*/\s*25\.0F\s*;",
            "the shared locomotion/interaction tick must remain exactly 1/25 s",
        )
        self.assertRegex(
            adapter,
            r"kControllerStepSeconds\s*=\s*"
            r"locomotion_timing::kStepSeconds\s*;",
        )
        self.assertEqual(
            controller.count(
                "const float dt = interaction::kControllerStepSeconds;"
            ),
            1,
            "the controller must have one authoritative 25 Hz update dt",
        )
        self.assertIn("SetTargetFPS(25);", controller)
        self.assertNotRegex(
            controller,
            r"(?:GetFrameTime\s*\(|1\.0[Ff]?\s*/\s*60\.0[Ff]?|"
            r"0\.0*166(?:6|7))",
            "manual activation must not introduce a render-rate controller step",
        )

    def test_manual_smart_pickup_uses_baked_scene_and_shared_one_slot_coordinator(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn(
            '#include "interaction_smart_pickup_scene.h"',
            controller,
            "controller must consume the baked Task 5 scene",
        )
        self.assertIn(
            '#include "interaction_smart_pickup_controller.h"',
            controller,
            "controller must consume the shared two-phase coordinator",
        )
        self.assertEqual(
            controller.count("interaction::make_smart_pickup_demo_target()"),
            1,
            "ordinary manual Smart Pickup must register the baked Task 5 target",
        )
        self.assertRegex(
            controller,
            r"interaction::SmartPickupController\s+"
            r"manual_smart_pickup_controller\b",
            "manual mode must use the shared raylib-free coordinator",
        )

        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        self.assertEqual(
            manual_input.count("manual_smart_pickup_controller.pre_step("),
            1,
            "F/X intent must cross the shared coordinator pre-step exactly once",
        )

        manual_post_step = self._source_between(
            controller,
            "        // Manual pick-assist observation begins.",
            "        // Manual pick-assist observation ends.",
        )
        self.assertEqual(
            manual_post_step.count("manual_smart_pickup_controller.post_step("),
            1,
            "the authoritative live-flat snapshot must cross post-step once",
        )

    def test_manual_smart_pickup_branch_has_no_legacy_slot_synthesis_or_pose_writes(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        for forbidden_state in (
            r"\binteraction::ControllerPickAssist\b",
            r"\bmanual_pick_assist(?:_|\b)",
            r"\bmanual_pick_(?:reach_waypoint|entry_slots|common_entry|"
            r"final_preview_certified_this_tick)\b",
        ):
            with self.subTest(forbidden_state=forbidden_state):
                self.assertNotRegex(
                    controller,
                    forbidden_state,
                    "controller.cpp must not duplicate SmartPickupController "
                    "assist, activation, slot, or preview state",
                )
        manual_source = "\n".join(
            (
                self._source_between(
                    controller,
                    "        // Manual pick-assist input begins.",
                    "        // Manual pick-assist input ends.",
                ),
                self._source_between(
                    controller,
                    "        // Manual pick-assist observation begins.",
                    "        // Manual pick-assist observation ends.",
                ),
                self._source_between(
                    controller,
                    "                    // Manual pick-assist submission begins.",
                    "                    // Manual pick-assist submission ends.",
                ),
            )
        )
        for forbidden in (
            "choose_pick_entry_slot(",
            "make_pick_reach_waypoint(",
            "make_pick_entry_slots(",
            "preview_pick_entry_slots(",
            "manual_pick_assist.begin(",
            "manual_pick_assist.observe(",
            "manual_pick_assist.take_submission(",
            "start.root_world =",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden,
                    manual_source,
                    "controller manual code must delegate frozen-slot work to "
                    "SmartPickupController",
                )
        for forbidden_write in (
            r"\bsimulation_(?:position|rotation)\s*=",
            r"\bbone_(?:positions|velocities|rotations|angular_velocities)"
            r"\s*\([^)]*\)\s*=",
            r"\b(?:locomotion_pose|displayed_pose)\."
            r"(?:positions|velocities|rotations|angular_velocities)"
            r"\s*\[[^]]+\]\s*=",
        ):
            with self.subTest(forbidden_write=forbidden_write):
                self.assertNotRegex(
                    manual_source,
                    forbidden_write,
                    "manual activation may steer through input but may not "
                    "write root, pose, or joint authority",
                )

    def test_manual_smart_pickup_requires_explicit_pack_except_legacy_fixtures(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        pack_selection = self._source_between(
            controller,
            "    const char* interaction_pack_environment =",
            "    interaction::InteractionRuntime interaction_runtime =",
        )
        legacy_match = re.search(
            r"const\s+bool\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"pickup_autodemo_enabled\s*\|\|\s*"
            r"placement_autodemo_enabled\s*;",
            controller,
        )
        self.assertIsNotNone(
            legacy_match,
            "only the two legacy auto-demo modes may use the diagnostic fixture",
        )
        legacy_name = legacy_match.group("name") if legacy_match else ""
        compact_selection = re.sub(r"\s+", "", pack_selection)
        explicit_match = re.search(
            r"const\s+bool\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"interaction_pack_environment\s*!=\s*nullptr\s*&&\s*"
            r"interaction_pack_environment\[0\]\s*!=\s*'\\0'\s*;",
            pack_selection,
        )
        self.assertIsNotNone(
            explicit_match,
            "pack selection must name the nonempty explicit environment case",
        )
        explicit_name = explicit_match.group("name") if explicit_match else ""
        self.assertRegex(
            compact_selection,
            rf"if\(!{re.escape(explicit_name)}&&"
            rf"!{re.escape(legacy_name)}\)\{{?throwstd::runtime_error\(",
            "missing packs must throw exactly for nonlegacy manual/acceptance mode",
        )
        self.assertEqual(
            pack_selection.count("./resources/g1_interaction"),
            1,
            "the diagnostic-pack path must remain a single legacy fixture fallback",
        )
        self.assertRegex(
            compact_selection,
            rf"{re.escape(explicit_name)}\?"
            r"std::filesystem::path\(interaction_pack_environment\):"
            r"std::filesystem::path\(\"\./resources/g1_interaction\"\)",
            "the fallback arm must be reachable only after the explicit/nonlegacy "
            "throw guard",
        )

    def test_manual_smart_pickup_pack_shape_is_exact_and_fail_closed(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        runtime_initialization = self._source_between(
            controller,
            "    interaction::InteractionRuntime interaction_runtime =",
            "    std::optional<AutodemoCanonicalEntry> autodemo_canonical_entry;",
        )
        pack_selection = self._source_between(
            controller,
            "    const char* interaction_pack_environment =",
            "    interaction::InteractionRuntime interaction_runtime =",
        )
        legacy_match = re.search(
            r"const\s+bool\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"pickup_autodemo_enabled\s*\|\|\s*"
            r"placement_autodemo_enabled\s*;",
            controller,
        )
        self.assertIsNotNone(legacy_match)
        legacy_name = legacy_match.group("name") if legacy_match else ""
        full_pack_guard = self._source_between(
            runtime_initialization,
            "            // Manual Smart Pickup full-pack guard begins.",
            "            // Manual Smart Pickup full-pack guard ends.",
        )
        compact_runtime = re.sub(r"\s+", "", full_pack_guard)
        self.assertRegex(
            compact_runtime,
            rf"^//ManualSmartPickupfull-packguardbegins\."
            rf"if\(!{re.escape(legacy_name)}\)\{{",
            "the exact full-pack guard must be scoped to nonlegacy modes",
        )
        outer_guard = f"if(!{legacy_name}){{"
        inner_guard_source = compact_runtime.split(outer_guard, 1)[1]
        shape_guard_match = re.search(
            r"if\((?P<condition>[^{}]+)\)\{"
            r"throwstd::runtime_error\(",
            inner_guard_source,
        )
        self.assertIsNotNone(
            shape_guard_match,
            "invalid full-pack shape must directly guard a startup exception",
        )
        shape_condition = (
            shape_guard_match.group("condition") if shape_guard_match else ""
        )
        for required in (
            "interaction_database->fps_numerator!=25U",
            "interaction_database->fps_denominator!=1U",
            "interaction_database->clip_count!=2045U",
            "interaction_database->frame_count!=511250U",
            "interaction_features->frame_count!="
            "interaction_database->frame_count",
        ):
            with self.subTest(required=required):
                self.assertIn(
                    required,
                    shape_condition,
                    "ordinary manual/acceptance startup must reject a pack "
                    "whose full-corpus shape is not exact",
                )
        self.assertEqual(
            shape_condition.count("||"),
            4,
            "all five exact shape mismatches must feed the same rejection guard",
        )
        self.assertIn("throwstd::runtime_error(", compact_runtime)
        self.assertNotIn("InteractionRuntime::disabled(", full_pack_guard)

        compact_initialization = re.sub(r"\s+", "", runtime_initialization)
        self.assertRegex(
            compact_initialization,
            rf"interaction::InteractionTargetdemo_target="
            rf"{re.escape(legacy_name)}\?"
            r"interaction::make_controller_demo_target\(\*interaction_database\):"
            r"interaction::make_smart_pickup_demo_target\(\);",
            "legacy auto-demo must retain its clip-derived fixture while "
            "nonlegacy manual mode uses only the baked Smart Pickup target",
        )
        catch_boundaries = (
            (
                "        catch (const interaction::FormatError& error)",
                "        catch (const std::exception& error)",
            ),
            (
                "        catch (const std::exception& error)",
                "        catch (...)",
            ),
            ("        catch (...)", "    }();"),
        )
        for start, stop in catch_boundaries:
            with self.subTest(catch_boundary=start):
                catch_source = self._source_between(
                    runtime_initialization, start, stop
                )
                compact_catch = re.sub(r"\s+", "", catch_source)
                self.assertRegex(
                    compact_catch,
                    rf"if\(!{re.escape(legacy_name)}\)\{{?throw;",
                    "each nonlegacy initialization failure must rethrow",
                )
                self.assertIn("InteractionRuntime::disabled(", catch_source)

    def test_controller_authors_one_destination_and_retains_direct_identity(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        adapter = Path("interaction_controller_adapter.cpp").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            controller.count("interaction::PlacementSurface destination_surface"),
            1,
            "scene construction must author exactly one destination surface",
        )
        for required in (
            "interaction::make_controller_demo_destination_surface(",
            "interaction::PlacementSurfaceRegistry interaction_surface_registry;",
            "interaction::PlaceMotionLibrary interaction_place_library{};",
            "interaction_destination_surface_handle =\n"
            "                interaction_surface_registry.upsert(destination_surface);",
            "interaction_destination_affordance_id =\n"
            "                destination_surface.affordances.front().id;",
            "interaction_surface_registry,\n"
            "                interaction_place_library,",
        ):
            with self.subTest(required=required):
                self.assertIn(required, controller)
        self.assertNotIn(
            "interaction_place_library.recorded.push_back",
            controller,
            "manual demo must force the reviewed ReversedPickup fallback",
        )
        self.assertIn("destination_table.position.z += 1.20F;", adapter)
        self.assertIn(
            "destination.support_volume_world = destination_table;", adapter
        )
        self.assertIn(
            "destination.support_volume_size = source_target.table_size;", adapter
        )
        self.assertIn("destination.overhead_clearance_m = 2.00F;", adapter)
        self.assertIn("projected_support_world", adapter)
        self.assertIn(
            "CertifiedControllerSceneSource certified_controller_scene_source(",
            adapter,
        )
        self.assertIn("inverse(source.rest_object_world)", adapter)
        self.assertIn("source.contact_hand_world", adapter)
        destination_constructor = self._cpp_function(
            adapter,
            "PlacementSurface make_controller_demo_destination_surface(",
        )
        self.assertIn(
            "certify_controller_destination_fit(", destination_constructor
        )
        self.assertIn("source_target.object_bounds", destination_constructor)
        self.assertNotIn("stable_source_object", destination_constructor)

    def test_manual_place_resolver_is_the_only_nearby_surface_lookup(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertEqual(
            controller.count("resolve_single_surface("),
            1,
            "only manual F may invoke the nearby-surface resolver",
        )
        resolver = self._source_between(
            controller,
            "    auto resolve_manual_place_target =",
            "    auto make_flat_controller_pose =",
        )
        self.assertIn("resolve_single_surface(", resolver)
        self.assertIn("1.00F", resolver)
        self.assertIn("ControllerPlaceTarget", resolver)
        self.assertNotIn(
            "interaction_destination_surface_handle", resolver,
            "manual nearby lookup must not replace stable scripted identity",
        )
        for required in (
            "interaction_destination_surface_handle",
            "interaction_destination_affordance_id",
        ):
            with self.subTest(required=required):
                self.assertGreaterEqual(
                    controller.count(required),
                    2,
                    "scene must retain the exact authored destination pair",
                )

    def test_place_preview_callback_is_narrow_and_runtime_owned(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        adapter = Path("interaction_controller_adapter.h").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            adapter,
            r"using\s+PlacePreviewResolver\s*=\s*std::function<\s*"
            r"PlaceStagingPreview\(\s*SurfaceHandle,\s*uint32_t\s*\)\s*>;",
            "scheduler preview callback must accept only handle and affordance",
        )
        callback = re.search(
            r"\[&interaction_runtime\]\(\s*"
            r"interaction::SurfaceHandle\s+surface,\s*"
            r"uint32_t\s+affordance_id\s*\)\s*"
            r"\{(?P<body>.*?)\n\s*\}",
            controller,
            re.DOTALL,
        )
        self.assertIsNotNone(
            callback,
            "controller must capture only runtime in the narrow preview lambda",
        )
        body = callback.group("body") if callback else ""
        self.assertRegex(
            body,
            r"return\s+interaction_runtime\.preview_place\(\s*"
            r"surface,\s*affordance_id\s*\)\s*;",
        )
        for forbidden in (
            "Pose", "object", "candidate", "library", "timing",
            "match", "IK", "config", "PlaceMatchInput",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)
        self.assertEqual(controller.count("interaction_runtime.preview_place("), 1)
        for forbidden in (
            "preview_place_motion(",
            "select_place_motion(",
            "PlaceMatchInput",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, controller)

    def test_scheduler_submits_only_newly_recomputed_ready_place_preview(self):
        adapter = Path("interaction_controller_adapter.cpp").read_text(
            encoding="utf-8"
        )
        for required in (
            "cached_output_.diagnostics.state == RuntimeState::Locomotion",
            "cached_output_.diagnostics.state == RuntimeState::Carry",
            "place_target_resolver(input.locomotion)",
            "place_preview_resolver(\n"
            "                latched_place_->surface,\n"
            "                latched_place_->affordance_id)",
            "preview.root_error_m <= kPlaceStagingMaximumRootErrorM",
            "preview.yaw_error_radians <=\n"
            "                kPlaceStagingMaximumYawErrorRadians",
            "preview.candidate.selection_id",
            "input.place_request = PlaceRequest{",
        ):
            with self.subTest(required=required):
                self.assertIn(required, adapter)
        self.assertIn("pending_cancel_", adapter)
        self.assertIn("latched_place_.reset();", adapter)
        self.assertNotIn("resolve_single_target(", adapter)
        self.assertNotIn("resolve_single_surface(", adapter)

    def test_place_staging_uses_camera_left_stick_without_root_writes(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        staging = self._cpp_function(
            controller, "vec3 controller_place_staging_stick("
        )
        for required in (
            "preview.staging_root_world.position - current_root.position",
            "preview.root_error_m",
            "preview.yaw_error_radians",
            "camera_control_basis",
            "quat_inv_mul_vec3(camera_control_basis, world_command)",
        ):
            with self.subTest(required=required):
                self.assertIn(required, staging)
        for forbidden in (
            "simulation_position =",
            "simulation_rotation =",
            "bone_positions(",
            "displayed_pose",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, staging)

        stick_assignment = controller.index(
            "gamepadstick_left = controller_place_staging_stick("
        )
        velocity_update = controller.index(
            "vec3 desired_velocity_curr = desired_velocity_update("
        )
        self.assertLess(
            stick_assignment,
            velocity_update,
            "place staging must enter through ordinary Carry input",
        )
        self.assertNotRegex(
            controller,
            r"gamepadstick_left\s*=\s*0\.25F\s*\*\s*"
            r"controller_place_staging_stick\(",
            "placement auto-demo must use the full staging input so the "
            "25 Hz controller can reach a ready preview within 150 ticks",
        )
        steering_gate = self._source_between(
            controller,
            "        if (!autodemo_configuration.has_value() &&",
            "        // Get if strafe is desired",
        )
        self.assertIn(
            "interaction_scheduler.cached_output().diagnostics.state ==\n"
            "                interaction::RuntimeState::Carry",
            steering_gate,
            "stale placement previews must not override Locomotion input",
        )
        self.assertNotRegex(
            controller,
            r"(?:simulation_position|bone_positions\(0\))\s*=\s*"
            r"[^;]*staging_root_world",
            "staging may never write simulation or displayed root",
        )

    def test_controller_keeps_pick_and_release_registry_policies_separate(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        self.assertEqual(
            controller.count("interaction_registry.resolve_single_target("),
            1,
            "only the retained legacy pickup auto-demo may use proximity "
            "target selection",
        )
        self.assertNotIn("resolve_single_target(", manual_input)
        self.assertRegex(
            manual_input,
            r"interaction_registry\.find\(\s*"
            r"interaction_scene_target_handle\s*\)",
            "manual Smart Pickup must address the baked scene target exactly",
        )
        self.assertNotIn("interaction_registry.reset(", controller)
        self.assertNotIn("interaction_registry.replace_pose(", controller)
        self.assertIn("interaction_registry.find_by_id(", controller)
        self.assertIn(
            "Interaction: F smart pickup/place  WASD/X cancel before attach  R reset",
            Path("interaction_debug_draw.h").read_text(encoding="utf-8"),
        )

    def test_manual_pick_assist_owns_locomotion_f_not_legacy_resolver(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertIn('#include "interaction_smart_pickup_controller.h"', controller)
        self.assertNotRegex(controller, r"\binteraction::ControllerPickAssist\b")

        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        normalized_input = " ".join(manual_input.split())
        for required in (
            "interaction::SmartPickupPreStepInput manual_smart_pickup_pre_input{}",
            "manual_smart_pickup_pre_input.runtime_state = cached_interaction_state",
            "manual_smart_pickup_pre_input.interact_pressed = interaction_edges.interact_pressed",
            "manual_smart_pickup_pre_input.cancel_pressed = interaction_edges.cancel_pressed || interaction_edges.reset_pressed",
            "manual_smart_pickup_pre_input.left_stick = gamepadstick_left",
            "manual_smart_pickup_pre_input.right_stick = gamepadstick_right",
            "manual_smart_pickup_controller.pre_step( manual_smart_pickup_pre_input)",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_input)
        self.assertRegex(
            normalized_input,
            r"if \(manual_smart_pickup_pre_step\.interact_consumed\) \{? "
            r"interaction_edges\.interact_pressed = false;",
            "the coordinator result must consume raw F before the scheduler",
        )
        self.assertRegex(
            normalized_input,
            r"if \(manual_smart_pickup_pre_step\.cancel_consumed\) \{? "
            r"interaction_edges\.cancel_pressed = false;",
            "the coordinator result must consume raw X before the scheduler",
        )
        self.assertLess(
            controller.index("        // Manual pick-assist input ends."),
            controller.index("vec3 desired_velocity_curr = desired_velocity_update("),
        )

        resolver = self._source_between(
            controller,
            "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
            "                    -> std::optional<interaction::PickRequest>",
            "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
            "                    -> std::optional<interaction::ControllerPlaceTarget>",
        )
        legacy_lookup = resolver.index(
            "interaction_registry.resolve_single_target("
        )
        auto_only_guard = resolver.index(
            "if (!autodemo_configuration.has_value())"
        )
        self.assertLess(auto_only_guard, legacy_lookup)
        self.assertIn("return std::nullopt;", resolver[auto_only_guard:legacy_lookup])

    def test_manual_smart_pickup_activation_captures_exact_scene_target_only(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        normalized_input = " ".join(manual_input.split())
        for required in (
            "const interaction::InteractionTarget* manual_smart_pickup_target = interaction_registry.find( interaction_scene_target_handle)",
            "manual_smart_pickup_pre_input.selected_target = manual_smart_pickup_target",
            "manual_smart_pickup_pre_input.selected_affordance_id = manual_smart_pickup_target->affordances.front().id",
            "manual_smart_pickup_controller.pre_step( manual_smart_pickup_pre_input)",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_input)
        ordered = (
            "interaction_registry.find(",
            "manual_smart_pickup_pre_input.selected_target =",
            "manual_smart_pickup_pre_input.selected_affordance_id =",
            "manual_smart_pickup_controller.pre_step(",
        )
        positions = [manual_input.find(marker) for marker in ordered]
        self.assertTrue(all(position >= 0 for position in positions))
        if all(position >= 0 for position in positions):
            self.assertEqual(
                positions,
                sorted(positions),
                "activation must resolve current identity/pose before planning and begin",
            )
        for forbidden in (
            "1.45F",
            "maximum_assisted_path_m",
            "resolve_single_target(",
            "make_pick_reach_waypoint(",
            "make_pick_entry_slots(",
            "PickAssistStart",
            "root_world =",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, manual_input)
        self.assertNotIn("interaction_authored_target", manual_input)

    def test_manual_pick_assist_applies_prior_output_without_hijacking_camera(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertNotIn("manual_pick_assist_output", controller)
        raw_input = self._source_between(
            controller,
            "        // Get gamepad stick states",
            "        // Press edges and runtime updates share",
        )
        self.assertIn(
            "const vec3 raw_gamepadstick_right = gamepadstick_right;",
            raw_input,
        )
        self.assertIn(
            "const bool raw_desired_strafe = desired_strafe_update();",
            raw_input,
        )

        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        normalized_input = " ".join(manual_input.split())
        for required in (
            "gamepadstick_left = manual_smart_pickup_pre_step.left_stick",
            "gamepadstick_right = manual_smart_pickup_pre_step.right_stick",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_input)
        self.assertLess(
            controller.index("        // Manual pick-assist input ends."),
            controller.index("vec3 desired_velocity_curr = desired_velocity_update("),
        )
        prior_output = self._source_between(
            controller,
            "        // Manual pick-assist prior output begins.",
            "        // Manual pick-assist prior output ends.",
        )
        normalized_prior = " ".join(prior_output.split())
        self.assertIn(
            "if (manual_smart_pickup_pre_step.force_strafe)",
            normalized_prior,
        )
        self.assertIn("desired_strafe = true", normalized_prior)
        self.assertNotIn("manual_pick_assist_activation_tick", controller)

        camera_update = self._source_between(
            controller,
            "        orbit_camera_update(",
            "        // Render",
        )
        self.assertIn("raw_gamepadstick_right", camera_update)
        self.assertIn("raw_desired_strafe", camera_update)
        self.assertNotIn("manual_smart_pickup_pre_step", camera_update)

    def test_controller_materializes_one_post_step_live_flat_snapshot(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        seam = self._source_between(
            controller,
            "        const interaction::RuntimeState cached_interaction_state =",
            "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
            "                    -> std::optional<interaction::PickRequest>",
        )
        self.assertEqual(
            seam.count("interaction::LocomotionSnapshot live_flat_snapshot"),
            1,
        )
        for required in (
            "live_flat_snapshot.pose = locomotion_pose;",
            "live_flat_snapshot.future_root_positions[index] =",
            "live_flat_snapshot.future_root_rotations[index] =",
            "const interaction::LocomotionSnapshot& snapshot =\n"
            "                        live_flat_snapshot;",
            "return autodemo_canonical_entry->snapshot;",
            "return snapshot;",
            "// Placement pickup preview provider begins.",
            "// Placement pickup preview provider ends.",
            "manual_smart_pickup_post_input.live_flat_snapshot =\n"
            "            live_flat_snapshot;",
            "manual_smart_pickup_controller.post_step(",
        ):
            with self.subTest(required=required):
                self.assertIn(required, seam)
        snapshot = seam.index(
            "interaction::LocomotionSnapshot live_flat_snapshot"
        )
        smart_post = seam.find("manual_smart_pickup_controller.post_step(")
        scheduler = seam.index("interaction_scheduler.tick(")
        provider_alias = seam.index(
            "const interaction::LocomotionSnapshot& snapshot ="
        )
        self.assertLess(snapshot, scheduler)
        self.assertGreaterEqual(
            smart_post,
            0,
            "the one authoritative snapshot must reach Smart Pickup post-step",
        )
        if smart_post >= 0:
            self.assertLess(snapshot, smart_post)
            self.assertLess(smart_post, scheduler)
        self.assertLess(scheduler, provider_alias)
        self.assertNotIn("interaction::LocomotionSnapshot snapshot;", seam)

    def test_manual_pick_post_step_uses_live_snapshot_fingerprint_and_request_id(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        observation = self._source_between(
            controller,
            "        // Manual pick-assist observation begins.",
            "        // Manual pick-assist observation ends.",
        )
        normalized_observation = " ".join(observation.split())
        for required in (
            "locomotion_snapshot_fingerprint(live_flat_snapshot)",
            "manual_smart_pickup_post_input.live_flat_snapshot = live_flat_snapshot",
            "manual_smart_pickup_post_input.runtime_state = cached_interaction_state",
            "manual_smart_pickup_post_input.next_request_id = interaction_next_request_id",
            "const interaction::InteractionTarget* manual_smart_pickup_current_target = interaction_registry.find( interaction_scene_target_handle)",
            "manual_smart_pickup_post_input.current_target = manual_smart_pickup_current_target",
            "for (int obstacle_index = 0; obstacle_index < obstacles_positions.size; ++obstacle_index)",
            "manual_smart_pickup_post_input.obstacle_centers.push_back( obstacles_positions(obstacle_index))",
            "manual_smart_pickup_post_input.obstacle_sizes.push_back( obstacles_scales(obstacle_index))",
            "manual_smart_pickup_post_input.simulation_velocity = simulation_velocity",
            "manual_smart_pickup_post_input.displayed_planar_speed_mps =",
            "manual_smart_pickup_post_input.camera_azimuth = camera_azimuth",
            "manual_smart_pickup_controller.post_step(",
            "interaction_runtime.preview_pick(",
            "manual_smart_pickup_post_step.snapshot_fingerprint",
        ):
            with self.subTest(observation_required=required):
                self.assertIn(required, normalized_observation)
        self.assertRegex(
            normalized_observation,
            r"if\s*\(\s*manual_smart_pickup_post_step\.snapshot_fingerprint\s*"
            r"!=\s*live_flat_snapshot_fingerprint\s*\)\s*\{?\s*"
            r"throw\s+std::(?:logic_error|runtime_error)\s*\(",
            "snapshot identity must be enforced, not merely compared",
        )
        for forbidden in (
            "preview_pick_entry_slots(",
            "make_pick_reach_waypoint(",
            "make_pick_entry_slots(",
            "choose_pick_entry_slot(",
        ):
            with self.subTest(observation_forbidden=forbidden):
                self.assertNotIn(forbidden, observation)
        self.assertLess(
            controller.index("        // Manual pick-assist observation ends."),
            controller.index("interaction_scheduler.tick("),
        )

    def test_manual_smart_pickup_pre_and_post_are_outside_both_legacy_autodemos(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        legacy_aliases = re.findall(
            r"const\s+bool\s+([A-Za-z_]\w*)\s*=\s*(?:"
            r"pickup_autodemo_enabled\s*\|\|\s*placement_autodemo_enabled|"
            r"placement_autodemo_enabled\s*\|\|\s*pickup_autodemo_enabled)\s*;",
            controller,
        )
        for phase, begin, end, coordinator_call in (
            (
                "pre-step",
                "        // Manual pick-assist input begins.",
                "        // Manual pick-assist input ends.",
                "manual_smart_pickup_controller.pre_step(",
            ),
            (
                "post-step",
                "        // Manual pick-assist observation begins.",
                "        // Manual pick-assist observation ends.",
                "manual_smart_pickup_controller.post_step(",
            ),
        ):
            with self.subTest(phase=phase):
                block = self._source_between(controller, begin, end)
                condition, guarded_scope = self._first_if_scope(block)
                self.assertEqual(
                    block.count(coordinator_call),
                    1,
                    f"manual Smart Pickup {phase} must cross the coordinator once",
                )
                self.assertIn(
                    coordinator_call,
                    guarded_scope,
                    f"synthetic legacy auto-demo F must not enter Smart Pickup {phase}",
                )
                self.assertTrue(
                    self._guard_excludes_legacy_autodemos(
                        condition,
                        tuple(legacy_aliases),
                    ),
                    f"manual Smart Pickup {phase} guard must exclude both "
                    "pickup and placement legacy auto-demo modes",
                )

    def test_legacy_autodemo_guard_parser_rejects_or_bypasses(self):
        aliases = ("legacy_fixture_mode",)
        accepted = (
            "!autodemo_configuration.has_value()",
            "((!autodemo_configuration.has_value())) && debug_enabled",
            "!pickup_autodemo_enabled && !placement_autodemo_enabled",
            "!(pickup_autodemo_enabled || placement_autodemo_enabled)",
            "ready && (!legacy_fixture_mode) && diagnostics_enabled",
        )
        rejected = (
            "!autodemo_configuration.has_value() || true",
            "(!legacy_fixture_mode) || debug_enabled",
            "ready && (!legacy_fixture_mode || debug_enabled)",
            "!pickup_autodemo_enabled || !placement_autodemo_enabled",
            "debug_enabled",
        )
        for condition in accepted:
            with self.subTest(accepted=condition):
                self.assertTrue(
                    self._guard_excludes_legacy_autodemos(
                        condition,
                        aliases,
                    )
                )
        for condition in rejected:
            with self.subTest(rejected=condition):
                self.assertFalse(
                    self._guard_excludes_legacy_autodemos(
                        condition,
                        aliases,
                    )
                )

    def test_manual_pick_activation_observes_post_step_and_caches_next_tick_output(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        observation = self._source_between(
            controller,
            "        // Manual pick-assist observation begins.",
            "        // Manual pick-assist observation ends.",
        )
        normalized_input = " ".join(manual_input.split())
        normalized_observation = " ".join(observation.split())
        for required in (
            "manual_smart_pickup_controller.pre_step(",
            "gamepadstick_left = manual_smart_pickup_pre_step.left_stick",
            "gamepadstick_right = manual_smart_pickup_pre_step.right_stick",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_input)
        for required in (
            "manual_smart_pickup_post_input.live_flat_snapshot = live_flat_snapshot",
            "manual_smart_pickup_controller.post_step(",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_observation)
        fingerprint_guard = re.search(
            r"if\s*\(\s*manual_smart_pickup_post_step\.snapshot_fingerprint\s*"
            r"!=\s*live_flat_snapshot_fingerprint\s*\)\s*\{?\s*"
            r"throw\s+std::(?:logic_error|runtime_error)\s*\(",
            normalized_observation,
        )
        self.assertIsNotNone(
            fingerprint_guard,
            "the post-step fingerprint mismatch must throw before publication",
        )
        stationary_search = self._source_between(
            controller,
            "        // Placement stationary search begins.",
            "        // Placement stationary search ends.",
        )
        self.assertRegex(
            " ".join(stationary_search.split()),
            r"const\s+bool\s+manual_pick_stationary_constraint_active\s*=\s*"
            r"[^;]*manual_smart_pickup_post_step\.assist_output\."
            r"stationary_constraint[^;]*;",
            "the returned assist output must drive the following ordinary tick's "
            "stationary search",
        )
        smart_pickup_header_path = Path("interaction_smart_pickup_controller.h")
        smart_pickup_header = (
            smart_pickup_header_path.read_text(encoding="utf-8")
            if smart_pickup_header_path.is_file()
            else ""
        )
        for forbidden in ("assist_observed", "activation_began"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden,
                    controller + smart_pickup_header,
                    "SmartPickupPostStepResult has no self-attested lifecycle flags",
                )
        self.assertNotIn("manual_pick_assist_activation_tick", controller)
        self.assertNotIn("gamepadstick_left =", observation)
        self.assertNotIn("gamepadstick_right =", observation)

        activation_span = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist observation ends.",
        )
        for once in (
            "manual_smart_pickup_controller.pre_step(",
            "vec3 desired_velocity_curr = desired_velocity_update(",
            "        simulation_positions_update(",
            "        simulation_rotations_update(",
            "interaction::LocomotionSnapshot live_flat_snapshot",
            "manual_smart_pickup_controller.post_step(",
        ):
            with self.subTest(exactly_once=once):
                self.assertEqual(
                    activation_span.count(once),
                    1,
                    "manual activation must reuse exactly one ordinary 25 Hz "
                    "controller update and one live-flat bridge",
                )

        pre_step = controller.index("manual_smart_pickup_controller.pre_step(")
        velocity_update = controller.index(
            "vec3 desired_velocity_curr = desired_velocity_update("
        )
        simulation_update = controller.index(
            "        simulation_positions_update(", velocity_update
        )
        snapshot = controller.index(
            "interaction::LocomotionSnapshot live_flat_snapshot", simulation_update
        )
        post_step = controller.index(
            "manual_smart_pickup_controller.post_step(", snapshot
        )
        fingerprint_use = controller.index(
            "manual_smart_pickup_post_step.snapshot_fingerprint", post_step
        )
        scheduler = controller.index("interaction_scheduler.tick(", post_step)
        self.assertLess(
            pre_step,
            velocity_update,
        )
        self.assertEqual(
            [
                pre_step,
                velocity_update,
                simulation_update,
                snapshot,
                post_step,
                fingerprint_use,
                scheduler,
            ],
            sorted(
                [
                    pre_step,
                    velocity_update,
                    simulation_update,
                    snapshot,
                    post_step,
                    fingerprint_use,
                    scheduler,
                ]
            ),
            "activation must bracket exactly the existing ordinary locomotion "
            "step before post-step observation, fingerprint verification, and "
            "scheduler publication",
        )

    def test_manual_pick_assist_submits_exactly_one_latched_request(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        observation = self._source_between(
            controller,
            "        // Manual pick-assist observation begins.",
            "        // Manual pick-assist observation ends.",
        )
        normalized_observation = " ".join(observation.split())
        for required in (
            "if (manual_smart_pickup_post_step.pick_request.has_value())",
            "interaction_edges.interact_pressed = true",
            "manual_smart_pickup_request = manual_smart_pickup_post_step.pick_request",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_observation)
        self.assertEqual(
            observation.count(
                "manual_smart_pickup_request = "
                "manual_smart_pickup_post_step.pick_request"
            ),
            1,
            "post-step must latch the already-numbered optional exactly once",
        )
        self.assertIsNone(
            re.search(
                r"(?:\+\+\s*interaction_next_request_id|"
                r"interaction_next_request_id\s*\+\+|"
                r"interaction_next_request_id\s*\+=)",
                observation,
            ),
            "post-step publication must not consume a request ID before the "
            "scheduler accepts the request",
        )

        request_guard_source = observation[
            observation.index(
                "if (manual_smart_pickup_post_step.pick_request.has_value())"
            ):
        ]
        request_condition, request_scope = self._first_if_scope(
            request_guard_source
        )
        self.assertEqual(
            "".join(request_condition.split()),
            "manual_smart_pickup_post_step.pick_request.has_value()",
        )
        for required in (
            "interaction_edges.interact_pressed = true",
            "manual_smart_pickup_request = "
            "manual_smart_pickup_post_step.pick_request",
        ):
            with self.subTest(latch_guard_required=required):
                self.assertIn(required, request_scope)

        resolver = self._source_between(
            controller,
            "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
            "                    -> std::optional<interaction::PickRequest>",
            "                [&](const interaction::LocomotionSnapshot& snapshot)\n"
            "                    -> std::optional<interaction::ControllerPlaceTarget>",
        )
        submission = self._source_between(
            resolver,
            "                    // Manual pick-assist submission begins.",
            "                    // Manual pick-assist submission ends.",
        )
        submission_condition, submission_scope = self._first_if_scope(
            submission
        )
        self.assertEqual(
            "".join(submission_condition.split()),
            "manual_smart_pickup_request.has_value()",
            "the manual latch must be the scheduler resolver's first guard",
        )
        exchange_matches = re.findall(
            r"std::exchange\(\s*manual_smart_pickup_request\s*,\s*"
            r"std::nullopt\s*\)",
            submission_scope,
        )
        self.assertEqual(
            len(exchange_matches),
            1,
            "the accepted scheduler path must exchange the latch exactly once",
        )
        local_match = re.search(
            r"(?:(?:const\s+)?std::optional<interaction::PickRequest>|"
            r"(?:const\s+)?auto)\s+([A-Za-z_]\w*)\s*=\s*"
            r"std::exchange\(\s*manual_smart_pickup_request\s*,\s*"
            r"std::nullopt\s*\)\s*;",
            submission_scope,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            local_match,
            "the exchanged request must be retained locally for ID validation",
        )
        submission_local = local_match.group(1)
        compact_scope = "".join(submission_scope.split())
        exchange = compact_scope.index(
            f"{submission_local}=std::exchange("
            "manual_smart_pickup_request,std::nullopt);"
        )
        mismatch_guard = compact_scope.index(
            f"if({submission_local}->request_id!="
            "interaction_next_request_id)",
            exchange,
        )
        mismatch_throw = re.search(
            r"throwstd::(?:logic_error|runtime_error)\(",
            compact_scope[mismatch_guard:],
        )
        self.assertIsNotNone(
            mismatch_throw,
            "request ID mismatch must fail in release builds",
        )
        mismatch_throw_position = mismatch_guard + mismatch_throw.start()
        increment = compact_scope.index(
            "++interaction_next_request_id;", mismatch_throw_position
        )
        returned = compact_scope.index(
            f"return{submission_local};", increment
        )
        self.assertEqual(
            len(
                re.findall(
                    r"(?:\+\+interaction_next_request_id|"
                    r"interaction_next_request_id\+\+|"
                    r"interaction_next_request_id\+=)",
                    compact_scope,
                )
            ),
            1,
            "the accepted scheduler path must consume exactly one request ID",
        )
        self.assertEqual(
            [exchange, mismatch_guard, mismatch_throw_position, increment, returned],
            sorted(
                [
                    exchange,
                    mismatch_guard,
                    mismatch_throw_position,
                    increment,
                    returned,
                ]
            ),
            "the resolver must exchange, validate the pre-increment ID, "
            "increment once, then return that same request",
        )
        self.assertLess(
            resolver.index("// Manual pick-assist submission begins."),
            resolver.index("if (!autodemo_configuration.has_value())"),
        )

        scheduler_call = controller.index("interaction_scheduler.tick(")
        scheduler_return = self._cpp_call_end(controller, scheduler_call)
        post_scheduler_boundary = controller.index(
            "        const uint64_t placement_preview_call_delta =",
            scheduler_return,
        )
        latch_reset = controller.index(
            "manual_smart_pickup_request.reset();", scheduler_return
        )
        self.assertEqual(
            controller[scheduler_return:latch_reset].strip(),
            "",
            "the caller must discard any scheduler-skipped latch immediately "
            "after tick returns",
        )
        self.assertLess(latch_reset, post_scheduler_boundary)
        self.assertNotIn(
            "manual_smart_pickup_request.reset();",
            controller[
                controller.index("        // Manual pick-assist observation ends."):
                scheduler_return
            ],
            "clearing before scheduler.tick would discard accepted requests",
        )
        self.assertNotIn("take_submission(", controller)
        self.assertNotIn("manual_pick_assist_synthetic_interact", controller)

    def test_manual_pick_reset_reaches_scheduler_and_clears_stale_output(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        edge_capture = self._source_between(
            controller,
            "        interaction::ControllerInteractionEdges interaction_edges{",
            "        const interaction::RuntimeState cached_interaction_state =",
        )
        self.assertIn("IsKeyPressed(KEY_R)", edge_capture)

        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        self.assertNotIn(
            "interaction_edges.reset_pressed = false",
            manual_input,
            "manual Smart Pickup must leave R available to scheduler priority",
        )
        self.assertIn(
            "manual_smart_pickup_pre_input.cancel_pressed = "
            "interaction_edges.cancel_pressed || "
            "interaction_edges.reset_pressed",
            " ".join(manual_input.split()),
            "R must cancel an active coordinator attempt without consuming the "
            "scheduler reset edge",
        )

        stationary_search = self._source_between(
            controller,
            "        // Placement stationary search begins.",
            "        // Placement stationary search ends.",
        )
        stationary_match = re.search(
            r"const\s+bool\s+manual_pick_stationary_constraint_active\s*=\s*"
            r"(?P<expression>.*?);",
            stationary_search,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            stationary_match,
            "manual stationary constraint expression is missing",
        )
        stationary_expression = " ".join(
            stationary_match.group("expression").split()
        )
        self.assertIn(
            "!interaction_edges.reset_pressed",
            stationary_expression,
            "R must suppress same-tick stationary interaction matching",
        )

        scheduler_call = controller.index("interaction_scheduler.tick(")
        compact_scheduler_prefix = "".join(
            controller[scheduler_call:scheduler_call + 160].split()
        )
        self.assertTrue(
            compact_scheduler_prefix.startswith(
                "interaction_scheduler.tick(interaction_edges,"
            ),
            "the unconsumed R edge must reach the scheduler",
        )
        scheduler_return = self._cpp_call_end(controller, scheduler_call)
        post_scheduler_boundary = controller.index(
            "        const uint64_t placement_preview_call_delta =",
            scheduler_return,
        )
        post_scheduler = controller[
            scheduler_return:post_scheduler_boundary
        ]
        self.assertEqual(
            post_scheduler.count("manual_smart_pickup_request.reset();"),
            1,
            "the caller must discard a request skipped by reset priority",
        )
        reset_guard_start = post_scheduler.index(
            "if (interaction_edges.reset_pressed)"
        )
        reset_condition, reset_scope = self._first_if_scope(
            post_scheduler[reset_guard_start:]
        )
        self.assertEqual(
            "".join(reset_condition.split()),
            "interaction_edges.reset_pressed",
        )
        self.assertIn(
            "manual_smart_pickup_post_step = {};",
            reset_scope,
            "R must clear persistent assist output after scheduler priority",
        )
        self.assertLess(
            post_scheduler.index("manual_smart_pickup_request.reset();"),
            reset_guard_start,
        )

    def test_manual_braking_uses_combined_stationary_search_only(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        search = self._source_between(
            controller,
            "        // Placement stationary search begins.",
            "        // Placement stationary search ends.",
        )
        normalized_search = " ".join(search.split())
        for required in (
            "const bool manual_pick_stationary_constraint_active =",
            "manual_smart_pickup_post_step.assist_output.stationary_constraint",
            "interaction::RuntimeState::Locomotion",
            "const bool manual_pick_stationary_constraint_latched_this_tick =",
            "manual_pick_stationary_constraint_active &&",
            "!manual_pick_stationary_constraint_was_active",
            "const bool interaction_stationary_constraint_active =",
            "placement_autodemo_state.stationary_constraint_active ||",
            "manual_pick_stationary_constraint_active",
            "manual_pick_stationary_constraint_latched_this_tick;",
            "!interaction_stationary_constraint_active",
            "if (interaction_stationary_constraint_active)",
            "stationary_motion_matching::search(",
            "best_index != frame_index ||",
            "manual_pick_stationary_constraint_latched_this_tick",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_search)

        constrained = self._source_between(
            search,
            "                if (interaction_stationary_constraint_active)",
            "                else\n                {\n                    database_search(",
        )
        placement_counters = self._source_between(
            constrained,
            "                    // Placement stationary evidence counters begin.",
            "                    // Placement stationary evidence counters end.",
        )
        for counter in (
            "stationary_search_calls",
            "stationary_selected_frame",
            "stationary_selected_range",
        ):
            with self.subTest(counter=counter):
                self.assertIn(counter, placement_counters)
                self.assertEqual(constrained.count(counter), placement_counters.count(counter))

    def test_manual_stationary_diagnostics_are_attempt_local_and_guarded(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        declaration = self._source_between(
            controller,
            "struct ManualPickStationaryDiagnostics",
            "};",
        )
        for required in (
            "uint64_t search_count = 0U;",
            "uint64_t transition_count = 0U;",
            "int selected_frame = -1;",
            "float selected_cost = 0.0F;",
        ):
            with self.subTest(required=required):
                self.assertIn(required, declaration)

        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        observation = self._source_between(
            controller,
            "        // Manual pick-assist observation begins.",
            "        // Manual pick-assist observation ends.",
        )
        self.assertEqual(
            manual_input.count("manual_pick_stationary_diagnostics = {};"),
            1,
            "one pre-step consume decision must clear the prior attempt exactly "
            "once, including simultaneous X/F",
        )
        self.assertNotIn(
            "manual_pick_stationary_diagnostics = {};",
            observation,
            "post-step observation must not infer that activation began",
        )
        normalized_input = " ".join(manual_input.split())
        reset_statement = "manual_pick_stationary_diagnostics = {};"
        pre_step_declaration = (
            "interaction::SmartPickupPreStepResult "
            "manual_smart_pickup_pre_step{};"
        )
        self.assertEqual(
            normalized_input.count(pre_step_declaration),
            1,
            "the per-tick result must have one declaration in the surrounding "
            "manual scope so prior output can consume it later",
        )
        legacy_aliases = re.findall(
            r"const\s+bool\s+([A-Za-z_]\w*)\s*=\s*(?:"
            r"pickup_autodemo_enabled\s*\|\|\s*placement_autodemo_enabled|"
            r"placement_autodemo_enabled\s*\|\|\s*pickup_autodemo_enabled)\s*;",
            controller,
        )
        pre_step_condition, pre_step_guarded_scope = self._first_if_scope(
            manual_input
        )
        self.assertTrue(
            self._guard_excludes_legacy_autodemos(
                pre_step_condition,
                tuple(legacy_aliases),
            ),
            "the manual pre-step assignment must be causally guarded from both "
            "legacy auto-demo modes",
        )
        normalized_pre_step_guard = " ".join(pre_step_guarded_scope.split())
        self.assertNotIn(
            pre_step_declaration,
            normalized_pre_step_guard,
            "the result must be declared outside the nonlegacy guard so the "
            "same per-tick value remains available to prior-output steering",
        )
        self.assertLess(
            normalized_input.index(pre_step_declaration),
            normalized_input.index(normalized_pre_step_guard),
            "the shared per-tick result must be declared before the nonlegacy "
            "manual block",
        )
        state_capture = (
            "const interaction::PickAssistState "
            "manual_smart_pickup_state_before_pre_step = "
            "manual_smart_pickup_controller.diagnostics().state;"
        )
        pre_step_assignment = (
            "manual_smart_pickup_pre_step = "
            "manual_smart_pickup_controller.pre_step("
        )
        self.assertEqual(
            normalized_input.count(pre_step_assignment),
            1,
            "there must be exactly one assignment from the real coordinator",
        )
        self.assertEqual(
            normalized_pre_step_guard.count(pre_step_assignment),
            1,
            "an assignment outside the parsed nonlegacy guard must not satisfy "
            "the manual pre-step contract",
        )
        self.assertIn(
            f"{state_capture} {pre_step_assignment}",
            normalized_pre_step_guard,
            "the caller must snapshot the real assist diagnostics immediately "
            "before the guarded pre_step call",
        )
        pre_step_call = normalized_input.index(
            "manual_smart_pickup_controller.pre_step("
        )
        reset = normalized_input.index(reset_statement)
        self.assertLess(
            pre_step_call,
            reset,
            "attempt-local diagnostics may reset only after the coordinator "
            "returns its real consume result",
        )
        new_attempt_definition = self._source_between(
            manual_input,
            "const bool manual_smart_pickup_new_attempt =",
            ";",
        )
        new_attempt_expression = new_attempt_definition.split("=", 1)[1]
        new_attempt_conjuncts = self._split_top_level_cpp_boolean(
            re.sub(r"\s+", "", new_attempt_expression),
            "&&",
        )
        self.assertEqual(
            len(new_attempt_conjuncts),
            2,
            "new-attempt gating must combine consumed F with one terminal-state "
            "predicate",
        )
        self.assertIn(
            "manual_smart_pickup_pre_step.interact_consumed",
            new_attempt_conjuncts,
        )
        terminal_clauses = [
            conjunct
            for conjunct in new_attempt_conjuncts
            if conjunct != "manual_smart_pickup_pre_step.interact_consumed"
        ]
        self.assertEqual(
            len(terminal_clauses),
            1,
            "new-attempt gating must have exactly one terminal-state clause",
        )
        terminal_clause = terminal_clauses[0]
        terminal_states = {
            self._strip_outer_cpp_parentheses(disjunct)
            for disjunct in self._split_top_level_cpp_boolean(
                terminal_clause,
                "||",
            )
        }
        self.assertEqual(
            terminal_states,
            {
                "manual_smart_pickup_state_before_pre_step=="
                "interaction::PickAssistState::Idle",
                "manual_smart_pickup_state_before_pre_step=="
                "interaction::PickAssistState::Submitted",
                "manual_smart_pickup_state_before_pre_step=="
                "interaction::PickAssistState::Failed",
            },
            "repeated F in an active assist state must not start a new diagnostics "
            "epoch",
        )
        self.assertRegex(
            normalized_input,
            r"if\s*\(\s*"
            r"manual_smart_pickup_pre_step\.cancel_consumed\s*\|\|\s*"
            r"manual_smart_pickup_pre_step\.manual_override_consumed\s*"
            r"\|\|\s*manual_smart_pickup_new_attempt\s*\)\s*\{?\s*"
            r"manual_pick_stationary_diagnostics\s*=\s*\{\};",
            "cancel, manual override, or a diagnostic-derived new attempt must "
            "enter one shared reset branch",
        )
        for active_state in (
            "PickAssistState::SlotApproach",
            "PickAssistState::Settling",
            "PickAssistState::FinalPreview",
            "PickAssistState::ReadyToSubmit",
        ):
            with self.subTest(repeated_f_state=active_state):
                self.assertNotIn(active_state, new_attempt_definition)
        for forbidden in ("assist_observed", "activation_began"):
            with self.subTest(observation_forbidden=forbidden):
                self.assertNotIn(forbidden, observation)

        search_update = self._source_between(
            controller,
            "                    // Manual stationary diagnostics search begins.",
            "                    // Manual stationary diagnostics search ends.",
        )
        for required in (
            "if (manual_pick_stationary_constraint_active)",
            "++manual_pick_stationary_diagnostics.search_count;",
        ):
            with self.subTest(required=required):
                self.assertIn(required, search_update)
        self.assertRegex(
            search_update,
            r"manual_pick_stationary_diagnostics\.selected_frame\s*=\s*"
            r"best_index;",
        )
        self.assertRegex(
            search_update,
            r"manual_pick_stationary_diagnostics\.selected_cost\s*=\s*"
            r"best_cost;",
        )
        self.assertNotIn("placement_autodemo_state", search_update)

        transition_update = self._source_between(
            controller,
            "                    // Manual stationary diagnostics transition begins.",
            "                    // Manual stationary diagnostics transition ends.",
        )
        self.assertIn(
            "if (manual_pick_stationary_constraint_active)", transition_update
        )
        self.assertIn(
            "++manual_pick_stationary_diagnostics.transition_count;",
            transition_update,
        )
        self.assertNotIn("placement_autodemo_state", transition_update)
        transition_branch = self._source_between(
            controller,
            "                if (best_index != frame_index ||",
            "                    frame_index = best_index;",
        )
        self.assertLess(
            transition_branch.index("inertialize_pose_transition("),
            transition_branch.index(
                "// Manual stationary diagnostics transition begins."
            ),
        )

    def test_manual_pick_cancel_releases_same_tick_and_carry_controls_pass_through(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        input_routing = self._source_between(
            controller,
            "        // Get gamepad stick states",
            "        if (pickup_autodemo_enabled)",
        )
        normalized_routing = " ".join(input_routing.split())
        for required in (
            "const vec3 raw_gamepadstick_left = gamepadstick_left;",
            "const interaction::RuntimeOutput& cached_interaction_output = interaction_scheduler.cached_output();",
            "const interaction::RuntimeState cached_interaction_state = cached_interaction_output.diagnostics.state;",
            "const bool committed_pickup_manual_override = manual_smart_pickup_override_pressed && !cached_interaction_output.diagnostics.attached && (cached_interaction_state == interaction::RuntimeState::Align || cached_interaction_state == interaction::RuntimeState::PickupReplay);",
            "if (committed_pickup_manual_override) { interaction_edges.cancel_pressed = true; gamepadstick_left = raw_gamepadstick_left; }",
            "if (cached_interaction_output.suppress_steering && !committed_pickup_manual_override) { gamepadstick_left = vec3(); }",
        ):
            with self.subTest(routing_required=required):
                self.assertIn(required, normalized_routing)
        committed_cancel = normalized_routing.find(
            "if (committed_pickup_manual_override)"
        )
        cached_suppression = normalized_routing.find(
            "if (cached_interaction_output.suppress_steering &&"
        )
        self.assertTrue(
            committed_cancel >= 0
            and cached_suppression >= 0
            and committed_cancel < cached_suppression,
            "movement cancellation must restore the raw stick before the "
            "cached suppression branch",
        )

        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        normalized_input = " ".join(manual_input.split())
        for required in (
            "manual_smart_pickup_pre_input.cancel_pressed = interaction_edges.cancel_pressed || interaction_edges.reset_pressed",
            "manual_smart_pickup_controller.pre_step(",
            "if (manual_smart_pickup_pre_step.cancel_consumed)",
            "interaction_edges.cancel_pressed = false",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_input)
        self.assertNotIn("manual_pick_assist.cancel(", controller)
        self.assertNotIn("manual_pick_assist_cancelled_this_tick", controller)

        prior_output = self._source_between(
            controller,
            "        // Manual pick-assist prior output begins.",
            "        // Manual pick-assist prior output ends.",
        )
        self.assertNotIn("cancel", prior_output)

        observation = self._source_between(
            controller,
            "        // Manual pick-assist observation begins.",
            "        // Manual pick-assist observation ends.",
        )
        self.assertNotIn("cancel", observation)
        carry = self._source_between(
            controller,
            "        if (!autodemo_configuration.has_value() &&",
            "        // Get if strafe is desired",
        )
        self.assertIn("interaction::RuntimeState::Carry", carry)
        self.assertIn("controller_place_staging_stick(", carry)

    def test_manual_pick_cancel_dominates_simultaneous_f(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        manual_input = self._source_between(
            controller,
            "        // Manual pick-assist input begins.",
            "        // Manual pick-assist input ends.",
        )
        normalized_input = " ".join(manual_input.split())
        required_in_order = (
            "manual_smart_pickup_pre_input.interact_pressed = interaction_edges.interact_pressed",
            "manual_smart_pickup_pre_input.cancel_pressed = interaction_edges.cancel_pressed || interaction_edges.reset_pressed",
            "manual_smart_pickup_controller.pre_step(",
            "if (manual_smart_pickup_pre_step.cancel_consumed)",
            "interaction_edges.cancel_pressed = false",
            "if (manual_smart_pickup_pre_step.interact_consumed)",
            "interaction_edges.interact_pressed = false",
        )
        positions = [normalized_input.find(marker) for marker in required_in_order]
        self.assertTrue(
            all(position >= 0 for position in positions),
            "simultaneous X/F coordinator seam is incomplete",
        )
        if all(position >= 0 for position in positions):
            self.assertEqual(
                positions,
                sorted(positions),
                "the coordinator must receive simultaneous X/F before the caller "
                "consumes either returned edge",
            )
        for forbidden in (
            "manual_pick_assist.cancel(",
            "manual_pick_assist_cancelled_this_tick",
            "!manual_pick_assist_cancelled_this_tick",
        ):
            self.assertNotIn(forbidden, manual_input)

    def test_manual_pick_assist_draws_acceptance_diagnostics_without_pose_writes(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertNotIn("manual_pick_object_distance_at_begin_m", controller)
        self.assertNotIn("manual_pick_common_entry", controller)

        route = self._source_between(
            controller,
            "        // Manual pick-assist route rendering begins.",
            "        // Manual pick-assist route rendering ends.",
        )
        for required in (
            "manual_smart_pickup_controller.diagnostics()",
            "manual_pick_diagnostics.slot_selection.selected_index",
            "manual_pick_diagnostics.slot_selection.ordered[",
            ".root_world.position",
            "DrawLine3D(",
            "bone_positions(0)",
            "interaction::PickAssistState::ReadyToSubmit",
            "? GREEN",
            "DrawSphereWires(",
        ):
            with self.subTest(required=required):
                self.assertIn(required, route)

        diagnostics = self._source_between(
            controller,
            "        // Manual pick-assist diagnostics begins.",
            "        // Manual pick-assist diagnostics ends.",
        )
        for required in (
            '"SMART PICKUP REACH - WASD or X cancels"',
            '"SMART PICKUP ATTACHED - finishing recorded lift"',
            '"CARRY - WASD moves, F places, R resets"',
            '"PICKUP FAILED: %s - reposition and press F"',
            "interaction_output.diagnostics.attached",
            "interaction::RuntimeState::Align",
            "interaction::RuntimeState::PickupReplay",
            "interaction::RuntimeState::Hold",
            "interaction::RuntimeState::Carry",
            "interaction::RuntimeState::Locomotion",
            "interaction::ResultCode::Rejected",
            "interaction::ResultCode::Failed",
            "interaction::pick_assist_reason_name(",
            "manual_pick_diagnostics.reason",
            "interaction::debug_draw::reason_name(",
            "interaction_output.diagnostics.reason",
            '"assist=%s reason=%s slot=%d settle=%u/%u route=%.3fm "',
            "interaction::pick_assist_state_name(",
            "interaction::pick_assist_reason_name(",
            "manual_smart_pickup_controller.diagnostics()",
            "manual_pick_diagnostics.object_origin_distance_m",
            "manual_pick_diagnostics.object_bounds_center_distance_m",
            '"error=%.3fm yaw=%.2fdeg speed=%.3fm/s"',
            "diagnostics.yaw_error_radians * 180.0F / PIf",
            '"final=%d fp_equal=%d roots_finite=%d root_equal=%d"',
            '"path=%d/%s match=%d/%s"',
            '"entry=%d contact=%d cost=%.3f"',
            "manual_pick_diagnostics.final_preview.available",
            "manual_pick_diagnostics.final_preview.fingerprint_equal",
            "manual_pick_diagnostics.final_preview.path_feasible",
            "manual_pick_diagnostics.final_preview.path_reason",
            "manual_pick_diagnostics.final_preview.match_ready",
            "manual_pick_diagnostics.final_preview.match_reason",
            "manual_pick_diagnostics.final_preview.feasible_entry_frame",
            "manual_pick_diagnostics.final_preview.contact_frame",
            "manual_pick_diagnostics.final_preview.total_cost",
            "interaction::debug_draw::reason_name(",
            '"stationary searches=%llu transitions=%llu frame=%d cost=%.3f"',
            "manual_pick_stationary_diagnostics.search_count",
            "manual_pick_stationary_diagnostics.transition_count",
            "manual_pick_stationary_diagnostics.selected_frame",
            "manual_pick_stationary_diagnostics.selected_cost",
        ):
            with self.subTest(required=required):
                self.assertIn(required, diagnostics)
        self.assertNotIn(
            "interaction::PickAssistState::Submitted",
            diagnostics,
            "submission alone must not render pickup-success state",
        )
        self.assertRegex(
            diagnostics,
            r"manual_pick_diagnostics\.final_preview\s*"
            r"\.all_preview_roots_finite",
        )
        self.assertRegex(
            diagnostics,
            r"manual_pick_diagnostics\.final_preview\s*"
            r"\.prospective_root_equal",
        )
        self.assertLess(
            controller.index("interaction::debug_draw::draw_interaction_text("),
            controller.index("// Manual pick-assist diagnostics begins."),
        )
        for forbidden in (
            "simulation_position =",
            "simulation_rotation =",
            "bone_positions(0) =",
            "bone_rotations(0) =",
            "interaction_registry.replace_pose(",
            "interaction_registry.reset(",
            "make_pick_reach_waypoint(",
            "make_pick_entry_slots(",
            "choose_pick_entry_slot(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, route + diagnostics)

    def test_live_flat_pick_oracle_sources_are_explicit_prerequisites(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        for target in (
            "$(LIVE_FLAT_PICK_ENTRY_ORACLE_TEST)",
            "$(LIVE_FLAT_PICK_ENTRY_ORACLE_RELEASE_TEST)",
        ):
            with self.subTest(target=target):
                prerequisites = self._source_between(
                    makefile,
                    f"{target}:",
                    "\n\t",
                )
                self.assertIn(
                    "$(LIVE_FLAT_PICK_ASSIST_SOURCES)",
                    prerequisites,
                )

    def test_real_pack_oracle_covers_manual_assist_route_and_one_shot_handoff(self):
        oracle = Path("tests/cpp/test_live_flat_pick_entry_oracle.cpp").read_text(
            encoding="utf-8"
        )
        makefile = Path("Makefile").read_text(encoding="utf-8")
        for include in (
            '#include "interaction_smart_pickup_controller.h"',
            '#include "interaction_smart_pickup_scene.h"',
            '#include "interaction_pick_approach.h"',
            '#include "locomotion_controller_update.h"',
        ):
            with self.subTest(include=include):
                self.assertIn(include, oracle)
        witness = self._cpp_function(
            oracle, "void run_manual_pick_assist_oracle("
        )
        for required in (
            "interaction::make_smart_pickup_demo_target()",
            "target->affordances.front().interaction_slots.size() == 3U",
            "OracleCountingSmartPickupBackend backend{}",
            "interaction::SmartPickupController smart_pickup_controller(backend)",
            "activation_input.interact_pressed = true",
            "smart_pickup_controller.pre_step(activation_input)",
            "advance_ordinary_locomotion(activation_pre_step)",
            "desired_velocity_update(",
            "desired_rotation_update(",
            "simulation_positions_update(",
            "simulation_rotations_update(",
            "maximum_tick_displacement_m",
            "locomotion_snapshot_fingerprint(live_flat_snapshot)",
            "materialized_snapshot_count ==\n"
            "                bridge_calls_before_activation + 1U",
            "backend.begin_calls == begins_before_activation + 1U",
            "backend.observe_calls == observations_before_activation + 1U",
            "smart_pickup_controller.post_step(",
            "rejecting_preview",
            "preview_callback_calls == 0U",
            "!post_step_result.pick_request.has_value()",
            "backend.take_submission_calls == 0U",
            "interaction::ControllerInteractionScheduler scheduler",
            "provider_calls == provider_calls_before_activation + 1U",
            "runtime_calls == runtime_calls_before_activation + 1U",
            "publications_before_activation + 1U",
            "post_step_result.assist_output.override_steering",
            "advance_ordinary_locomotion(assisted_pre_step)",
            "ordinary_calls_before_assisted_tick + 1U",
            "bridge_calls_before_assisted_tick + 1U",
            "observations_before_assisted_post + 1U",
            "provider_calls == provider_calls_before_assisted_tick + 1U",
            "runtime_calls == runtime_calls_before_assisted_tick + 1U",
            "publications_before_assisted_tick + 1U",
            "cancel_input.cancel_pressed = true",
            "smart_pickup_controller.pre_step(cancel_input)",
            "backend.cancel_calls == 1U",
            "ordinary_locomotion_step_calls == 2U",
            "materialized_snapshot_count == 2U",
            "provider_calls == 2U && runtime_calls == 2U",
            "scheduler_publication_count == 2U",
            "backend.begin_calls == 1U && backend.observe_calls == 2U",
            "resolver_calls == 0U && preview_callback_calls == 0U",
            "backend.take_submission_calls == 0U",
        ):
            with self.subTest(required=required):
                self.assertIn(required, witness)
        for forbidden in (
            "1.45F",
            "resolve_single_target(",
            "make_pick_reach_waypoint(",
            "make_pick_entry_slots(",
            "choose_pick_entry_slot(",
            "ControllerPickAssist",
            "assist.begin(",
            "assist.observe(",
            "assist.take_submission(",
            "observation.displayed_root = {\n        common_entry",
            "observation.displayed_root = frozen_slot.waypoint",
            "const interaction::LocomotionSnapshot settled_snapshot",
            "preview_pick(",
            "settled_observation_count",
            "stationary_constraint_edges",
            "stationary_output_applied_ticks",
            "provider_calls_before_certification",
            "resolver_calls_before_certification",
            "certified_provider_snapshot_fingerprint",
            "submission_count",
            "++next_request_id",
            "final_preview",
            "RuntimeState::Preflight",
            "RuntimeState::Align",
            "Reason::TargetUnavailable",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, witness)
        self.assertIn(
            "run_manual_pick_assist_oracle(\n"
            "            flat_database, interaction_database, interaction_features, bridge);",
            oracle,
        )
        self.assertIn("interaction_smart_pickup_controller.cpp", makefile)
        for source in (
            "interaction_pick_assist.cpp",
            "interaction_arrival.cpp",
            "locomotion_controller_update.cpp",
        ):
            with self.subTest(source=source):
                self.assertGreaterEqual(makefile.count(source), 3)
        self.assertIn(
            "locomotion_controller_update.h", makefile
        )

    def test_readme_documents_manual_destination_and_reversed_pickup(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        for required in (
            "F` (gamepad right-face-left) to request pickup or placement",
            "1.00 m",
            "1.20 m",
            "ReversedPickup",
            "25 Hz",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)

    def test_readme_documents_manual_pick_assist_contract(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        manual = self._source_between(
            readme,
            "For manual play, build the controller and point it at the generated pack:",
            "The manual demo intentionally supplies no recorded place clips",
        )
        normalized_readme = " ".join(manual.split())
        for required in (
            "MM_INTERACTION_PACK",
            "25/1",
            "2,045 clips",
            "511,250 frames",
            "baked Smart Pickup target",
            "post-step live-flat snapshot",
            "assisted route remains at most 1.00 m",
            "ordinary flat-ground locomotion",
            "one frozen authored-slot preview",
            "five consecutive settled 25 Hz ticks",
            "exactly one pickup request",
            "`X` cancels any pre-submission assist",
            "Carry keeps the existing `F` placement controls",
        ):
            with self.subTest(required=required):
                self.assertIn(required, normalized_readme)
        for forbidden in (
            "1.45 m",
            "both live entry-slot previews",
            "object_distance_at_begin",
            "MM_INTERACTION_PACK=resources/g1_interaction",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, normalized_readme)

    def test_controller_uses_exact_23_pose_bridge_and_flat_toe_indices(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        adapter = Path("interaction_controller_adapter.h").read_text(
            encoding="utf-8"
        )
        adapter_impl = Path("interaction_controller_adapter.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "struct FlatControllerPose",
            adapter,
            "adapter must expose the fixed-size ordinary controller pose",
        )
        self.assertIn(
            "expand_flat_controller_pose(",
            controller,
            "controller must explicitly expand the flat pose into G1 semantics",
        )
        self.assertIn(
            "collapse_interaction_pose(",
            adapter_impl,
            "frame handoff must collapse owned G1 poses to flat semantics",
        )
        self.assertNotIn(
            "collapse_interaction_pose(",
            controller,
            "controller must consume the already-retargeted flat handoff pose",
        )
        self.assertRegex(
            controller,
            r"if\s*\(\s*db\.nbones\(\)\s*!=\s*"
            r"static_cast<int>\(\s*interaction::kFlatControllerBoneCount\s*\)\s*\)",
            "ordinary database must fail fast unless it has exactly 23 bones",
        )
        self.assertIn(
            "db.bone_parents(static_cast<int>(bone)) !=\n"
            "                interaction::kFlatControllerParents[bone]",
            controller,
            "ordinary database must fail fast unless its 23-parent tree is exact",
        )
        self.assertIn(
            "db.bone_parents.size ==\n"
            "        static_cast<int>(interaction::kFlatControllerBoneCount)",
            controller,
            "parent-tree guard must validate its length before indexing",
        )
        self.assertIn(
            "ordinary database parent tree does not match flat controller",
            controller,
            "parent mismatch must have a specific fail-fast diagnostic",
        )
        self.assertNotIn(
            "for (int bone = 0; bone < g1_skeleton::BoneCount; ++bone)",
            controller,
            "no 31-bone loop may index ordinary controller arrays",
        )

        contacts = self._source_between(
            controller,
            "    // Contact and Foot Locking data",
            "    array1d<bool> contact_states",
        )
        self.assertRegex(
            contacts,
            r"contact_bones\(0\)\s*=\s*"
            r"static_cast<int>\(interaction::kFlatControllerLeftToe\)",
            "controller must lock the actual flat left toe (index 5)",
        )
        self.assertRegex(
            contacts,
            r"contact_bones\(1\)\s*=\s*"
            r"static_cast<int>\(interaction::kFlatControllerRightToe\)",
            "controller must lock the actual flat right toe (index 9)",
        )

        self.assertNotIn(
            "latest_owned_interaction_pose",
            controller,
            "flat/G1 calibration must never follow a dynamic owned pose",
        )
        canonical_init = controller.index(
            "        initialize_autodemo_canonical_world();"
        )
        flat_reference_capture = controller.index(
            "const interaction::FlatControllerPose\n"
            "        interaction_flat_reference_pose = [&]()"
        )
        self.assertLess(
            canonical_init,
            flat_reference_capture,
            "the flat reference must be captured after canonical pickup init",
        )
        self.assertEqual(
            controller.count("interaction_flat_reference_pose ="),
            1,
            "the calibrated flat reference must be frozen exactly once",
        )
        fixed_pair = controller[flat_reference_capture:]
        self.assertIn("reference.velocities.fill(vec3());", fixed_pair)
        self.assertIn("reference.angular_velocities.fill(vec3());", fixed_pair)
        for channel in (
            "positions", "velocities", "rotations", "angular_velocities"
        ):
            with self.subTest(root_channel=channel):
                self.assertIn(
                    f"interaction_reference_pose.{channel}[interaction_root] =\n"
                    f"        interaction_flat_reference_pose.{channel}[0];",
                    fixed_pair,
                    "the true-G1 reference root must be scene-aligned to flat",
                )
        self.assertRegex(
            controller,
            r"expand_flat_controller_pose\(\s*flat_locomotion_pose,\s*"
            r"interaction_reference_pose,\s*interaction_flat_reference_pose\s*\)",
            "ordinary live snapshots must use the frozen reference pair",
        )
        self.assertRegex(
            controller,
            r"expand_flat_controller_pose\(\s*interaction_frame_state\.pose,\s*"
            r"interaction_reference_pose,\s*interaction_flat_reference_pose\s*\)",
            "debug reconstruction must use the same frozen reference pair",
        )
        self.assertIn(
            "if (use_autodemo_canonical_snapshot)",
            controller,
            "canonical pickup must retain its exact snapshot exception",
        )
        self.assertIn(
            "return autodemo_canonical_entry->snapshot;",
            controller,
            "canonical pickup must bypass ordinary live reconstruction",
        )

    def test_flat_joint_names_match_character_h_order(self):
        self.assertEqual(
            FLAT_JOINT_NAMES,
            (
                "Entity", "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot",
                "LeftToe", "RightUpLeg", "RightLeg", "RightFoot",
                "RightToe", "Spine", "Spine1", "Spine2", "Neck", "Head",
                "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            ),
        )

    def test_autodemo_captures_selected_grasp_and_rendered_hand_after_final_fk_before_draw(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        final_fk = controller.index("        forward_kinematics_full(")
        capture = controller.index("capture_autodemo_evidence(", final_fk)
        draw = controller.index("        BeginDrawing();", final_fk)
        self.assertLess(
            final_fk,
            capture,
            "autodemo evidence must be sampled after final foot-IK FK",
        )
        self.assertLess(
            capture,
            draw,
            "autodemo evidence must be sampled before drawing",
        )
        self.assertEqual(
            controller[final_fk:draw].count("capture_autodemo_evidence("),
            1,
            "autodemo must capture one immutable final evidence sample",
        )
        capture_call = self._source_between(
            controller,
            "            autodemo_evidence_capture = capture_autodemo_evidence(",
            "        }\n        \n        // Update camera",
        )
        self.assertIn(
            "interaction_frame_state",
            capture_call,
            "post-final-FK capture must receive the validated handoff state",
        )

        capture_function = self._cpp_function(
            controller,
            "AutodemoEvidenceCapture capture_autodemo_evidence(",
        )
        self.assertIn("interaction::kFlatControllerBoneCount", capture_function)
        self.assertIn("global_bone_positions.size", capture_function)
        self.assertIn("global_bone_rotations.size", capture_function)
        self.assertIn("quat_normalize(global_bone_rotations", capture_function)
        self.assertIn(
            "const interaction::ControllerInteractionFrameState& frame_state",
            capture_function,
            "capture must consume the exact handoff validation and calibration",
        )
        self.assertIn(
            "runtime_output.diagnostics.target == scene_target->handle",
            capture_function,
            "grasp capture must agree with the exact selected target handle",
        )
        self.assertIn(
            "affordance.id == runtime_output.diagnostics.affordance_id",
            capture_function,
            "grasp capture must resolve the exact selected affordance",
        )
        self.assertIn(
            "affordance.hand == runtime_output.diagnostics.hand",
            capture_function,
            "grasp capture must agree with the selected runtime hand",
        )
        self.assertIn(
            "interaction::compose(capture.object_world, capture.hand_in_object)",
            capture_function,
            "grasp capture must derive the world transform from scene authority",
        )
        self.assertRegex(
            capture_function,
            r"capture\.calibrated_hand_world_rotation\s*=\s*"
            r"quat_normalize\(quat_mul\(\s*"
            r"capture\.joint_rotations\[.*?active_hand_joint.*?\],\s*"
            r"quat_inv\(capture\.hand_constraint_calibration_rotation\)\s*"
            r"\)\);",
            "semantic hand rotation must be derived from post-final-FK raw hand rotation",
        )
        self.assertNotIn(
            "frame_state.pose.rotations",
            capture_function,
            "pre-foot-IK handoff rotations must not become calibrated evidence",
        )

        writer = self._cpp_function(controller, "void write_autodemo_record(")
        self.assertIn(
            "const AutodemoEvidenceCapture& capture,",
            writer,
            "JSON writer must consume the immutable post-final-FK capture",
        )
        self.assertNotIn("global_bone_positions", writer)
        self.assertNotIn("global_bone_rotations", writer)
        self.assertNotIn("interaction_scene_state", writer)
        self.assertNotIn("interaction_scene_target", writer)
        self.assertIn(
            "const vec3 position = capture.joint_positions[joint];",
            writer,
        )
        self.assertIn(
            "const quat rotation = capture.joint_rotations[joint];",
            writer,
        )
        for member in (
            "capture.grasp_evidence_valid",
            "capture.active_hand_joint",
            "capture.hand_constraint_weight",
            "capture.hand_constraint_validated",
            "capture.hand_constraint_result.applied",
            "capture.hand_constraint_result.reachable",
            "capture.hand_constraint_result.used_clavicle",
            "capture.hand_constraint_result.reach_shortfall_m",
            "capture.hand_constraint_calibration_rotation",
            "capture.calibrated_hand_world_rotation",
            "capture.object_world.rotation",
            "capture.hand_in_object.position",
            "capture.hand_in_object.rotation",
            "capture.grasp_world.position",
            "capture.grasp_world.rotation",
        ):
            self.assertIn(
                member,
                writer,
                f"JSON writer must serialize {member} from the capture",
            )
        self.assertIn('\\"joint_world_positions\\":[', writer)
        self.assertIn('\\"joint_world_rotations\\":[', writer)

        writer_call = self._source_between(
            controller,
            "                write_autodemo_record(",
            "                if (!autodemo_state.carry_origin_captured",
        )
        self.assertEqual(
            writer_call.count("autodemo_evidence_capture.value()"),
            1,
            "JSON writer must receive exactly the immutable final capture",
        )

    def test_autodemo_drains_seven_unlogged_frames_before_exit(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        self.assertRegex(
            controller,
            r"constexpr\s+uint32_t\s+"
            rf"kAutodemoResetPresentationFrames\s*=\s*"
            rf"{RESET_PRESENTATION_FRAMES}U;",
            "autodemo Reset must remain visible for the 0.25 s release",
        )

        state = self._source_between(
            controller,
            "struct ControllerAutodemoState",
            "float autodemo_planar_distance",
        )
        self.assertIn(
            "uint32_t reset_presentation_frames_remaining = 0U;",
            state,
            "autodemo state must own the bounded presentation drain",
        )

        reset_completion = self._source_between(
            controller,
            "                if (autodemo_action == AutodemoAction::Reset)\n"
            "                {\n"
            "                    if (autodemo_state.collapsed_states.size()",
            "                else\n"
            "                {\n"
            "                    ++autodemo_state.render_frame;",
        )
        self.assertIn(
            "autodemo_state.reset_pending = false;",
            reset_completion,
            "the successful Reset edge must not be pulsed during the drain",
        )
        self.assertIn(
            "autodemo_state.reset_presentation_frames_remaining =\n"
            "                        kAutodemoResetPresentationFrames;",
            reset_completion,
            "the logged Reset must start the exact 7-frame drain",
        )
        for forbidden in (
            "publish_autodemo_evidence(",
            "autodemo_state.complete = true;",
            "autodemo_state.exit_requested = true;",
            "++autodemo_state.render_frame;",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden,
                    reset_completion,
                    "Reset recording must not publish, exit, or append a row",
                )

        post_draw = self._source_between(
            controller,
            "        EndDrawing();",
            "            ++autodemo_state.warmup_render_ticks;",
        )
        required_order = (
            "if (autodemo_state.reset_presentation_frames_remaining > 0U)",
            "--autodemo_state.reset_presentation_frames_remaining;",
            "if (autodemo_state.reset_presentation_frames_remaining == 0U)",
            "publish_autodemo_evidence(*autodemo_configuration);",
            "autodemo_state.complete = true;",
            "autodemo_state.exit_requested = true;",
            "return;",
        )
        positions = []
        for marker in required_order:
            self.assertIn(marker, post_draw)
            positions.append(post_draw.index(marker))
        self.assertEqual(
            positions,
            sorted(positions),
            "drain countdown, publication, and exit must remain ordered",
        )

    def test_autodemo_reset_drain_is_input_free_and_precedes_logging(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        autodemo_input = self._source_between(
            controller,
            "            // Auto evidence is deterministic in the absence of external",
            "        // Get if strafe is desired",
        )
        drain_input = re.search(
            r"if\s*\(autodemo_state\.reset_presentation_frames_remaining\s*"
            r">\s*0U\)\s*\{(?P<body>.*?)\n\s*\}",
            autodemo_input,
            re.DOTALL,
        )
        self.assertIsNotNone(
            drain_input,
            "autodemo input handling must have an explicit Reset drain branch",
        )
        drain_input_body = drain_input.group("body") if drain_input else ""
        self.assertRegex(
            drain_input_body,
            r"\binteraction_edges\s*=\s*\{\s*\}\s*;",
            "Reset drain must clear physical F/X/R edges after sampling",
        )
        self.assertNotIn(
            "gamepadstick_right",
            drain_input_body,
            "Reset drain may not disable live camera input",
        )
        left_stick_zero = autodemo_input.index("gamepadstick_left = vec3();")
        drain_guard = autodemo_input.index(
            "if (autodemo_state.reset_presentation_frames_remaining > 0U)"
        )
        scripted_guard = autodemo_input.index(
            "if (autodemo_state.evidence_started &&"
        )
        self.assertLess(
            left_stick_zero,
            drain_guard,
            "autodemo locomotion motion must already be zero before the drain",
        )
        self.assertLess(
            drain_guard,
            scripted_guard,
            "physical edges must be cleared before scripted edge selection",
        )
        self.assertRegex(
            autodemo_input,
            r"if\s*\(autodemo_state\.evidence_started\s*&&\s*"
            r"autodemo_state\.reset_presentation_frames_remaining\s*"
            r"==\s*0U\)",
            "drain renders must not issue Interact, Forward, or Reset input",
        )

        sampled_edges = controller.index(
            "interaction::ControllerInteractionEdges interaction_edges{"
        )
        cleared_edges = controller.index("interaction_edges = {};", sampled_edges)
        scheduler_tick = controller.index(
            "interaction_scheduler.tick(", cleared_edges
        )
        self.assertLess(
            sampled_edges,
            cleared_edges,
            "Reset drain clear must apply to the already-sampled physical edges",
        )
        self.assertLess(
            cleared_edges,
            scheduler_tick,
            "cleared Reset drain edges must reach the fixed-25 scheduler",
        )

        end_drawing = controller.index("        EndDrawing();")
        drain = controller.index(
            "if (autodemo_state.reset_presentation_frames_remaining > 0U)",
            end_drawing,
        )
        warmup = controller.index(
            "++autodemo_state.warmup_render_ticks;", drain
        )
        progression = controller.index(
            "validate_autodemo_state_progression(", warmup
        )
        writer = controller.index("write_autodemo_record(", progression)
        self.assertLess(end_drawing, drain)
        self.assertLess(drain, warmup)
        self.assertLess(warmup, progression)
        self.assertLess(progression, writer)
        drain_block = controller[drain:warmup]
        self.assertIn("return;", drain_block)
        self.assertNotIn("write_autodemo_record(", drain_block)
        self.assertNotIn("++autodemo_state.render_frame", drain_block)
        self.assertNotIn("WaitTime(", drain_block)
        self.assertNotRegex(drain_block, r"\b(?:sleep|usleep)\s*\(")

    def test_canonical_world_is_initialized_once_before_log_and_update(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        initializer_marker = (
            "    auto initialize_autodemo_canonical_world = [&]()"
        )
        call_marker = "    initialize_autodemo_canonical_world();"
        self.assertIn(initializer_marker, controller)
        self.assertIn(call_marker, controller)
        initializer = self._source_between(
            controller,
            initializer_marker,
            call_marker,
        )
        call_index = controller.index(call_marker)
        self.assertLess(
            call_index,
            controller.index("autodemo_state.log.imbue", call_index),
            "canonical world initialization must precede autodemo logging",
        )
        self.assertLess(
            call_index,
            controller.index("    auto update_func = [&]()", call_index),
            "canonical world initialization must precede the update loop",
        )
        self.assertEqual(
            initializer.count("inertialize_root_adjust("),
            1,
            "canonical world initialization must adjust the rendered root once",
        )
        for forbidden in (
            "curr_bone_positions(0) =",
            "curr_bone_rotations(0) =",
            "trns_bone_positions(0) =",
            "trns_bone_rotations(0) =",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden,
                    initializer,
                    "canonical world initialization must not rewrite animation space",
                )
        for required in (
            "bone_velocities(0) = canonical_pose.velocities[root];",
            "bone_angular_velocities(0) =",
            "simulation_position = canonical_pose.positions[root];",
            "desired_velocity_change_curr = vec3();",
            "trajectory_positions(0) = simulation_position;",
            "future_root_positions[index]",
            "adjusted_bone_positions(0) = bone_positions(0);",
            "reset_controller_contacts();",
        ):
            with self.subTest(required=required):
                self.assertIn(
                    required,
                    initializer,
                    "canonical initializer is missing required world/root state",
                )
        self.assertNotIn(
            "canonical_move_applied",
            controller,
            "canonical placement must not be deferred into the render loop",
        )
        update_loop = self._source_between(
            controller,
            "    auto update_func = [&]()",
            "    std::function<void()> u{update_func};",
        )
        self.assertNotIn(
            "const interaction::Pose& canonical_pose =",
            update_loop,
            "update loop must not physically place the canonical root",
        )
        self.assertIn(
            "return autodemo_canonical_entry->snapshot;",
            update_loop,
            "canonical snapshot must remain private to the scheduler provider",
        )

    def test_canonical_snapshot_stays_in_scheduler_provider(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        seam = self._source_between(
            controller,
            "        interaction::Pose locomotion_pose =",
            "        const interaction::Pose interaction_debug_pose =",
        )
        provider = self._source_between(
            seam,
            "interaction_scheduler.tick(",
            "[&](const interaction::LocomotionSnapshot& snapshot)",
        )
        self.assertIn(
            "return autodemo_canonical_entry->snapshot;",
            provider,
            "canonical snapshot must remain available to the provider through Preflight",
        )
        self.assertNotIn(
            "locomotion_pose = autodemo_canonical_entry->snapshot.pose;",
            seam,
            "ordinary locomotion_pose must reach the Task 11 handoff unchanged",
        )
        self.assertRegex(
            seam,
            r"interaction_frame_handoff\.apply\(\s*flat_locomotion_pose,",
            "frame handoff must receive ordinary flat locomotion pose",
        )

    def test_scene_authoritative_grasp_precedes_final_pose_publication(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        scene_apply = controller.index("interaction_scene_handoff.apply(")
        affordance_lookup = controller.index(
            "interaction_registry.find_affordance(", scene_apply
        )
        constraint_apply = controller.index(
            "interaction_frame_handoff.apply(", affordance_lookup
        )
        pose_publication = controller.index(
            "interaction_frame_state.pose.positions[bone]", constraint_apply
        )
        self.assertLess(scene_apply, affordance_lookup)
        self.assertLess(affordance_lookup, constraint_apply)
        self.assertLess(constraint_apply, pose_publication)
        constraint_block = controller[affordance_lookup:constraint_apply]
        self.assertRegex(
            constraint_block,
            r"find_affordance\(\s*"
            r"interaction_output\.diagnostics\.target,\s*"
            r"interaction_output\.diagnostics\.affordance_id\)",
            "constraint must use the runtime's exact target and affordance",
        )
        self.assertRegex(
            constraint_block,
            r"constraint\.grasp_world\s*=\s*interaction::compose\(\s*"
            r"interaction_scene_state\.object_world,\s*"
            r"selected_affordance->hand_in_object\)",
            "constraint must compose the scene-authoritative semantic grasp",
        )
        self.assertNotIn("interaction_registry.replace_pose", constraint_block)
        self.assertNotIn("interaction_registry.reset", constraint_block)
        self.assertNotIn("interaction_registry.upsert", constraint_block)

    def test_selected_grasp_constraint_covers_targeted_reach_but_not_free(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        constraint_block = self._source_between(
            controller,
            "        const interaction::InteractionTarget* selected_constraint_target =",
            "        interaction_frame_state = interaction_frame_handoff.apply(",
        )
        predicate = self._source_between(
            constraint_block,
            "        if (selected_constraint_target != nullptr &&",
            "        {\n            interaction::ControllerInteractionHandConstraint constraint;",
        )
        self.assertEqual(
            " ".join(predicate.split()),
            "if (selected_constraint_target != nullptr && "
            "interaction_scene_target == selected_constraint_target && "
            "selected_affordance != nullptr && "
            "(selected_constraint_target->state == "
            "interaction::ObjectState::Targeted || "
            "selected_constraint_target->state == "
            "interaction::ObjectState::Attached || "
            "selected_constraint_target->state == "
            "interaction::ObjectState::Held) && "
            "selected_affordance->hand == "
            "interaction_output.diagnostics.hand)",
            "the selected grasp gate must admit exactly the Targeted/Attached/Held "
            "union and reject Free before constructing a final-rig constraint",
        )

    def test_both_evidence_writers_use_the_exact_positive_weight_rejection_guard(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        helper = self._cpp_function(
            controller,
            "bool autodemo_hand_constraint_evidence_valid(",
        )
        helper_predicate = self._source_between(
            helper,
            "    return autodemo_is_finite(capture.hand_constraint_weight)",
            ";\n}",
        )
        self.assertEqual(
            " ".join(helper_predicate.split()),
            "return autodemo_is_finite(capture.hand_constraint_weight) && "
            "capture.hand_constraint_weight >= 0.0F && "
            "capture.hand_constraint_weight <= 1.0F && "
            "(capture.hand_constraint_weight == 0.0F || "
            "(capture.hand_constraint_validated && "
            "capture.hand_constraint_result.applied && "
            "capture.hand_constraint_result.reachable))",
            "zero weight must be the only no-solve exception; every positive "
            "weight must require the same-frame validated/applied/reachable conjunction",
        )

        helper_call = "!autodemo_hand_constraint_evidence_valid(capture)"
        for name, signature in (
            ("pickup", "void write_autodemo_record("),
            ("placement", "void write_placement_autodemo_record("),
        ):
            with self.subTest(writer=name):
                writer = self._cpp_function(controller, signature)
                rejection_predicate = self._source_between(
                    writer,
                    "    if (scheduler_phase != 0 ||",
                    "    {\n        throw std::runtime_error",
                )
                normalized = " ".join(rejection_predicate.split())
                self.assertEqual(
                    rejection_predicate.count(helper_call),
                    1,
                    "each writer's primary pre-publication rejection predicate "
                    "must invoke the shared final-FK guard exactly once",
                )
                self.assertIn(
                    f"|| {helper_call} ||",
                    normalized,
                    "invalid hand-constraint evidence must be a negated top-level "
                    "OR rejection term, not an unrelated or permissive helper call",
                )
                self.assertEqual(
                    writer.count(helper_call),
                    1,
                    "the writer must not satisfy the source contract with an "
                    "irrelevant duplicate helper call",
                )
                guard_call = writer.index(helper_call)
                rejection = writer.index("throw std::runtime_error", guard_call)
                publication = writer.index("output <<", rejection)
                self.assertLess(guard_call, rejection)
                self.assertLess(
                    rejection,
                    publication,
                    "the rejecting guard must execute before JSON publication",
                )

    def test_canonical_row_invariant_is_limited_to_locomotion_prefix(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        canonical_entry = self._cpp_function(
            controller,
            "AutodemoCanonicalEntry make_autodemo_canonical_entry(",
        )
        comparison = self._source_between(
            canonical_entry,
            "    float maximum_error = 0.0F;",
            "    return entry;",
        )
        self.assertIn(
            "constexpr size_t kAutodemoLocomotionFeatureStop = 45U;",
            comparison,
            "canonical equality must stop after the stable pose/trajectory prefix",
        )
        self.assertRegex(
            comparison,
            r"for \(size_t dimension = 0;\s*"
            r"dimension < kAutodemoLocomotionFeatureStop;\s*"
            r"\+\+dimension\)",
            "canonical equality must compare exactly the locomotion prefix",
        )
        self.assertNotIn(
            "dimension < query.size()",
            comparison,
            "drifting target-dependent features must not be treated as invariant",
        )
        self.assertIn(
            "!autodemo_is_finite(query[dimension])",
            comparison,
            "canonical locomotion query values must remain finite-checked",
        )
        self.assertIn(
            "!autodemo_is_finite(expected)",
            comparison,
            "source locomotion row values must remain finite-checked",
        )
        self.assertIn(
            "autodemo canonical locomotion query differs from clip-0 Reach row",
            comparison,
            "canonical mismatch diagnostics must describe the compared prefix",
        )

    def test_feature_cache_call_site_uses_checked_serializer(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        save_site = self._source_between(
            controller,
            "    database_build_matching_features(",
            "    // Interaction data is a separate fixed-25 pack.",
        )
        with self.subTest(boundary="legacy serializer"):
            self.assertNotIn(
                "database_save_matching_features(",
                save_site,
                "controller must not use the unchecked legacy serializer",
            )
        with self.subTest(boundary="checked serializer"):
            self.assertIn(
                "save_matching_features_checked(",
                save_site,
                "controller must call its checked feature-cache serializer",
            )

    def test_checked_feature_cache_serializer_checks_io_lifecycle(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        serializer = self._cpp_function(
            controller, "void save_matching_features_checked("
        )
        for operation, pattern in (
            ("open", r"(?:std::)?fopen\s*\(|std::ofstream"),
            ("write", r"(?:std::)?fwrite\s*\(|\.write\s*\("),
            ("flush", r"(?:std::)?fflush\s*\(|\.flush\s*\("),
            ("close", r"(?:std::)?fclose\s*\(|\.close\s*\("),
        ):
            with self.subTest(operation=operation):
                self.assertRegex(
                    serializer,
                    pattern,
                    f"checked feature-cache serializer must {operation}",
                )
        self.assertIn(
            "throw std::runtime_error",
            serializer,
            "feature-cache I/O failures must become controller diagnostics",
        )

    def test_publication_preserves_backups_when_rollback_fails(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        publication = self._cpp_function(
            controller, "void publish_autodemo_evidence("
        )
        catch = publication.index("catch (...)")
        rollback_failure = publication.index(
            "if (!log_restored || !screenshot_restored)", catch
        )
        cleanup = publication.index("cleanup_autodemo_temporaries", catch)
        self.assertLess(
            rollback_failure,
            cleanup,
            "rollback failure must be reported before cleanup can delete a backup",
        )

    def test_success_publication_uses_checked_cleanup(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        publication = self._cpp_function(
            controller, "void publish_autodemo_evidence("
        )
        self.assertIn(
            "cleanup_autodemo_temporaries_checked(configuration);",
            publication,
            "successful publication must check temporary cleanup",
        )
        cleanup = self._cpp_function(
            controller, "void cleanup_autodemo_temporaries_checked("
        )
        self.assertIn(
            "autodemo_remove_checked(",
            cleanup,
            "successful cleanup must use checked removals",
        )

    def test_post_publish_backup_cleanup_is_nonfatal(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        publication = self._cpp_function(
            controller, "void publish_autodemo_evidence("
        )
        temporary_cleanup = self._cpp_function(
            controller, "void cleanup_autodemo_temporaries_checked("
        )
        with self.subTest(boundary="checked temporary cleanup"):
            self.assertNotIn(
                "log_backup",
                temporary_cleanup,
                "checked post-publication cleanup must not delete backups",
            )
            self.assertNotIn(
                "screenshot_backup",
                temporary_cleanup,
                "checked post-publication cleanup must not delete backups",
            )
        with self.subTest(boundary="best-effort backup call"):
            self.assertIn(
                "cleanup_autodemo_backups_best_effort(configuration);",
                publication,
                "published finals must use nonfatal best-effort backup cleanup",
            )

        signature = "void cleanup_autodemo_backups_best_effort("
        if signature in controller:
            backup_cleanup = self._cpp_function(controller, signature)
            self.assertIn("noexcept", backup_cleanup)
            self.assertIn("log_backup", backup_cleanup)
            self.assertIn("screenshot_backup", backup_cleanup)
            self.assertNotIn(
                "autodemo_remove_checked(",
                backup_cleanup,
                "backup cleanup must not throw after final publication",
            )
            self.assertNotIn(
                "throw ",
                backup_cleanup,
                "backup cleanup must preserve undeletable backups without failing",
            )

    def test_autodemo_sigterm_uses_async_safe_flag_and_loop_polling(self):
        controller = Path("controller.cpp").read_text(encoding="utf-8")
        support = self._source_between(
            controller,
            "enum class AutodemoAction",
            "float autodemo_planar_distance",
        )
        flag = re.search(
            r"volatile\s+(?:std::)?sig_atomic_t\s+([A-Za-z_]\w*)",
            support,
        )
        handler = re.search(
            r"void\s+([A-Za-z_]\w*(?:sigterm|signal)[A-Za-z_]\w*)"
            r"\s*\(\s*int(?:\s+[A-Za-z_]\w*)?\s*\)\s*(?:noexcept)?",
            support,
            re.IGNORECASE,
        )
        with self.subTest(boundary="async-safe flag"):
            self.assertIsNotNone(
                flag,
                "autodemo SIGTERM support must use volatile sig_atomic_t",
            )
        with self.subTest(boundary="signal handler"):
            self.assertIsNotNone(
                handler,
                "autodemo must define a SIGTERM flag-only handler",
            )

        setup = self._source_between(
            controller,
            "        autodemo_configuration = parse_autodemo_environment();",
            "    // Init Window",
        )
        with self.subTest(boundary="handler installation"):
            self.assertRegex(
                setup,
                r"(?:(?:std::)?signal|sigaction)\s*\(\s*SIGTERM",
                "autodemo setup must install its SIGTERM handler",
            )

        loop = self._source_between(
            controller,
            "#else\n    while (!autodemo_state.exit_requested)",
            "#endif\n\n    int exit_code",
        )
        if flag is not None:
            with self.subTest(boundary="loop polling"):
                self.assertIn(
                    flag.group(1),
                    loop,
                    "controller loop must poll the async-safe SIGTERM flag",
                )
        if handler is not None:
            handler_source = self._cpp_function(
                support, f"void {handler.group(1)}("
            )
            with self.subTest(boundary="handler safety"):
                self.assertNotRegex(
                    handler_source,
                    r"\b(?:fprintf|remove|cleanup|throw|new|delete)\b",
                    "SIGTERM handler may only set the async-safe flag",
                )

    def test_controller_command_is_externally_bounded(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        self.assertIn(
            "timeout --signal=TERM --kill-after=5s 45s",
            makefile,
            "controller command must have the required external timeout",
        )
        self.assertIn(
            "xdpyinfo", makefile, "playable gate must preflight the display"
        )
        self.assertIn(
            "MM_FEATURES_OUTPUT",
            makefile,
            "playable gate must redirect the feature cache",
        )


@unittest.skipUnless(
    os.environ.get("PLAYABLE_LOG") and os.environ.get("PLAYABLE_SCREENSHOT"),
    "playable evidence paths not set",
)
class PlayableInteractionEvidenceTests(unittest.TestCase):
    def test_real_playable_evidence(self):
        records = load_evidence(Path(os.environ["PLAYABLE_LOG"]))
        validate_evidence(records)
        validate_screenshot(Path(os.environ["PLAYABLE_SCREENSHOT"]))

    def test_real_playable_evidence_preserves_bounded_25_hz_contract(self):
        records = load_evidence(Path(os.environ["PLAYABLE_LOG"]))
        self.assertLessEqual(len(records), MAX_EVIDENCE_RECORDS)
        self.assertEqual(
            [record["render_frame"] for record in records],
            list(range(len(records))),
        )
        self.assertEqual(
            [record["runtime_tick"] for record in records],
            list(range(records[0]["runtime_tick"], records[0]["runtime_tick"] + len(records))),
        )
        self.assertTrue(all(record["scheduler_phase"] == 0 for record in records))
        self.assertEqual(records[-1]["render_frame"], len(records) - 1)
        self.assertEqual(records[-1]["state"], "Locomotion")
        self.assertEqual(records[-1]["action"], "reset")
        self.assertEqual(records[-1]["result"], "Reset")
        self.assertEqual(records[-1]["reason"], "Reset")
        self.assertFalse(records[-1]["attached"])


if __name__ == "__main__":
    unittest.main()
