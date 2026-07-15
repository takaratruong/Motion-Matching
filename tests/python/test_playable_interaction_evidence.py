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
    "root_displacement_m",
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
RUNTIME_CACHED_FIELDS = (
    "state",
    "result",
    "reason",
    "object_state",
    "attached",
    "owns_pose",
    "carry_mode",
)


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


def _validate_record_types(record: dict) -> None:
    _require_integer(record, "render_frame", 0)
    _require_integer(record, "runtime_tick", 0)
    _require_integer(record, "scheduler_phase", 0, 59)
    _require_enum(record, "state", STATES)
    _require_enum(record, "result", RESULTS)
    _require_enum(record, "reason", REASONS)
    _require_enum(record, "object_state", OBJECT_STATES)
    for name in ("attached", "owns_pose"):
        if type(record[name]) is not bool:
            raise _error(f"{name} must be a JSON boolean")
    _require_enum(record, "carry_mode", CARRY_MODES)
    _require_integer(record, "carry_command_frame", -1, 149)
    _require_position(record, "root_position")
    _require_position(record, "object_position")
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
    if len(records) > 1200:
        raise _error("evidence must contain at most 1200 render records")
    for index, record in enumerate(records):
        if record["render_frame"] != index:
            raise _error("render_frame must start at zero and be sequential")
        if index == 0:
            continue
        previous = records[index - 1]
        runtime_step = record["runtime_tick"] - previous["runtime_tick"]
        if runtime_step not in (0, 1):
            raise _error("runtime_tick must be nondecreasing by at most one")
        phase = previous["scheduler_phase"] + 25
        expected_runtime_step = 0
        if phase >= 60:
            phase -= 60
            expected_runtime_step = 1
        if record["scheduler_phase"] != phase:
            raise _error("scheduler_phase does not follow the exact 25-of-60 clock")
        if runtime_step != expected_runtime_step:
            raise _error("runtime_tick does not follow the exact 25-of-60 clock")
        if runtime_step == 0:
            changed = [
                field
                for field in RUNTIME_CACHED_FIELDS
                if record[field] != previous[field]
            ]
            if changed:
                raise _error(
                    "cached runtime output changed on a non-due runtime tick: "
                    + ", ".join(changed)
                )
        if not previous["owns_pose"] and not record["owns_pose"]:
            root_step_m = math.dist(
                previous["root_position"], record["root_position"]
            )
            if root_step_m > 0.20:
                raise _error(
                    "adjacent non-owned root step must not exceed 0.20 m"
                )

    interact = [record for record in records if record["action"] == "interact"]
    if len(interact) != 1 or interact[0]["render_frame"] != 30:
        raise _error("Interact must be pulsed exactly once on render frame 30")
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
    if len(forward) != 150:
        raise _error("evidence must contain exactly 150 forward records")
    if [record["carry_command_frame"] for record in forward] != list(range(150)):
        raise _error("forward records must be numbered exactly 0 through 149")
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
    if carry[-1]["carry_mode"] not in {"recorded", "layered"}:
        raise _error("final Carry mode must be recorded or layered")

    first_carry_index = records.index(carry[0])
    if first_carry_index > 900:
        raise _error("Carry must be observed by evidence frame 900")
    if forward_indices[0] != first_carry_index + 1:
        raise _error("forward command 0 must immediately follow first Carry")
    if forward_indices[:149] != list(
        range(forward_indices[0], forward_indices[0] + 149)
    ):
        raise _error("forward commands 0 through 148 must be consecutive")
    final_command_index = forward_indices[149]
    for wait_index in range(forward_indices[148] + 1, final_command_index):
        wait = records[wait_index]
        if wait["state"] != "Carry" or wait["action"] != "none":
            raise _error("only Carry none waits may precede command 149")
        if wait["scheduler_phase"] + 25 >= 60:
            raise _error("command 149 exceeded the first scheduler-alignment frame")
    if records[final_command_index]["scheduler_phase"] + 25 < 60:
        raise _error("command 149 must immediately precede a scheduler-due render")
    if (
        final_command_index + 1 >= len(records)
        or records[final_command_index + 1]["runtime_tick"]
        != records[final_command_index]["runtime_tick"] + 1
    ):
        raise _error("Reset must follow command 149 on the scheduler-due render")

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
    if carry[-1]["action"] != "forward" or carry[-1]["carry_command_frame"] != 149:
        raise _error("the 150th command must be the final Carry record")

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
        f'"root_displacement_m":{record["root_displacement_m"]:.6f},'
        f'"action":{string("action")}}}\n'
    )


def _reclock_records(records: list[dict]) -> None:
    scheduler_phase = 15
    runtime_tick = 1
    for render_frame, record in enumerate(records):
        if render_frame != 0:
            scheduler_phase += 25
            if scheduler_phase >= 60:
                scheduler_phase -= 60
                runtime_tick += 1
        record["render_frame"] = render_frame
        record["runtime_tick"] = runtime_tick
        record["scheduler_phase"] = scheduler_phase


def _valid_records() -> list[dict]:
    states = (
        ["Locomotion"] * 31
        + ["Preflight"] * 2
        + ["Align"] * 3
        + ["PickupReplay"] * 2
        + ["Hold"] * 3
        + ["Carry"] * 151
        + ["Locomotion"]
    )
    records = []
    scheduler_phase = 15
    runtime_tick = 1
    for render_frame, state in enumerate(states):
        if render_frame != 0:
            scheduler_phase += 25
            if scheduler_phase >= 60:
                scheduler_phase -= 60
                runtime_tick += 1

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
        if render_frame == 30:
            action = "interact"
        if state == "Carry" and render_frame > 41:
            carry_command_frame = render_frame - 42
            action = "forward"
            root_x = 0.002 * (carry_command_frame + 1)
        if render_frame == len(states) - 1:
            root_x = 0.30

        result = "None"
        reason = "None"
        if state in {"Align", "PickupReplay", "Hold"}:
            result = "Accepted"
        elif state == "Carry":
            result = "Succeeded"

        record = {
            "render_frame": render_frame,
            "runtime_tick": runtime_tick,
            "scheduler_phase": scheduler_phase,
            "state": state,
            "result": result,
            "reason": reason,
            "object_state": object_state,
            "attached": attached,
            "owns_pose": owns_pose,
            "carry_mode": "layered" if state == "Carry" else "none",
            "carry_command_frame": carry_command_frame,
            "root_position": [root_x, 0.0, 2.0],
            "object_position": [
                0.0 if render_frame == len(states) - 1 else root_x,
                0.75 if render_frame == len(states) - 1 else 0.95,
                3.0,
            ],
            "root_displacement_m": root_x,
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

    def test_valid_synthetic_evidence_is_accepted(self):
        _write_records(self.log, _valid_records())
        _write_png(self.screenshot)
        records = load_evidence(self.log)
        validate_evidence(records)
        validate_screenshot(self.screenshot)

    def test_synthetic_fixture_changes_runtime_only_on_due_ticks(self):
        records = _valid_records()
        for previous, record in zip(records, records[1:]):
            if record["runtime_tick"] == previous["runtime_tick"]:
                self.assertEqual(
                    tuple(record[field] for field in RUNTIME_CACHED_FIELDS),
                    tuple(previous[field] for field in RUNTIME_CACHED_FIELDS),
                    f"cached runtime output changed on frame {record['render_frame']}",
                )

    def test_non_due_cached_runtime_changes_are_rejected(self):
        mutations = {
            "state": (32, "Align"),
            "result": (1, "Accepted"),
            "reason": (1, "OutOfRange"),
            "object_state": (1, "Targeted"),
            "attached": (1, True),
            "owns_pose": (1, True),
            "carry_mode": (1, "layered"),
        }
        for field, (frame, value) in mutations.items():
            with self.subTest(field=field):
                records = _valid_records()
                records[frame][field] = value
                _write_records(self.log, records)
                with self.assertRaisesRegex(
                    EvidenceValidationError, "non-due runtime tick"
                ):
                    validate_evidence(load_evidence(self.log))

    def test_unsuccessful_runtime_outputs_after_interact_are_rejected(self):
        failure_frame = 31
        failure_tick = _valid_records()[failure_frame]["runtime_tick"]
        for field, value in (
            ("result", "Rejected"),
            ("result", "Cancelled"),
            ("result", "Failed"),
            ("state", "Disabled"),
        ):
            with self.subTest(field=field, value=value):
                records = _valid_records()
                for record in records:
                    if record["runtime_tick"] == failure_tick:
                        record[field] = value
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

    def test_carry_must_be_observed_by_evidence_frame_900(self):
        records = _valid_records()
        first_carry = next(
            index for index, record in enumerate(records) if record["state"] == "Carry"
        )
        delayed_carry_frame = 902
        hold = records[first_carry - 1]
        records[first_carry:first_carry] = [
            copy.deepcopy(hold) for _ in range(delayed_carry_frame - first_carry)
        ]
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "frame 900"):
            validate_evidence(load_evidence(self.log))

    def test_evidence_is_bounded_to_at_most_1200_records(self):
        records = _valid_records()
        first_forward = next(
            index
            for index, record in enumerate(records)
            if record["carry_command_frame"] == 0
        )
        waiting_carry = records[first_forward - 1]
        records[first_forward:first_forward] = [
            copy.deepcopy(waiting_carry) for _ in range(1201 - len(records))
        ]
        _reclock_records(records)
        self.assertEqual(len(records), 1201)
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "at most 1200"):
            validate_evidence(load_evidence(self.log))

    def test_reset_action_appears_only_on_the_final_record(self):
        records = _valid_records()
        records[10]["action"] = "reset"
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "Reset action"):
            validate_evidence(load_evidence(self.log))

    def test_interact_action_appears_only_once_on_frame_30(self):
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

    def test_forward_commands_zero_through_148_are_consecutive(self):
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
        with self.assertRaisesRegex(EvidenceValidationError, "0 through 148"):
            validate_evidence(load_evidence(self.log))

    def test_only_minimal_scheduler_alignment_wait_precedes_command_149(self):
        records = _valid_records()
        command_148 = next(
            index
            for index, record in enumerate(records)
            if record["carry_command_frame"] == 148
        )
        wait = copy.deepcopy(records[command_148])
        wait.update(action="none", carry_command_frame=-1)
        records[command_148 + 1:command_148 + 1] = [
            copy.deepcopy(wait) for _ in range(12)
        ]
        _reclock_records(records)
        _write_records(self.log, records)
        with self.assertRaisesRegex(EvidenceValidationError, "scheduler-alignment"):
            validate_evidence(load_evidence(self.log))

    def test_one_required_scheduler_alignment_wait_before_command_149_is_valid(self):
        records = _valid_records()
        first_carry = next(
            index for index, record in enumerate(records) if record["state"] == "Carry"
        )
        hold = records[first_carry - 1]
        phase_zero_delay = next(
            delay
            for delay in range(12)
            if (15 + 25 * (first_carry + delay)) % 60 == 0
        )
        records[first_carry:first_carry] = [
            copy.deepcopy(hold) for _ in range(phase_zero_delay)
        ]
        _reclock_records(records)
        command_148 = next(
            index
            for index, record in enumerate(records)
            if record["carry_command_frame"] == 148
        )
        wait = copy.deepcopy(records[command_148])
        wait.update(action="none", carry_command_frame=-1)
        records.insert(command_148 + 1, wait)
        _reclock_records(records)
        _write_records(self.log, records)
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
        self.assertFalse(records[29]["owns_pose"])
        self.assertFalse(records[30]["owns_pose"])
        self.assertEqual(records[30]["action"], "interact")
        records[30]["root_position"][0] = 2.433498
        _write_records(self.log, records)
        with self.assertRaisesRegex(
            EvidenceValidationError,
            "adjacent non-owned root step must not exceed 0.20 m",
        ):
            validate_evidence(load_evidence(self.log))

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

    def test_forward_count_must_be_exactly_150(self):
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
                displacement = 0.20 * (command + 1) / 150.0
                record["root_position"][0] = displacement
                record["object_position"][0] = displacement
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
            "SetTargetFPS(60);", controller, "controller must retain fixed 60 Hz"
        )
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

        self.assertIn(
            "std::optional<interaction::Pose> latest_owned_interaction_pose;",
            controller,
            "controller must cache the exact prior owned G1 handoff pose",
        )
        self.assertIn(
            "latest_owned_interaction_pose = interaction_output.pose;",
            controller,
            "owned frames must cache the raw G1 runtime reference",
        )
        self.assertNotIn(
            "latest_owned_interaction_pose = interaction_debug_pose;",
            controller,
            "reconstructed debug poses must never become runtime authority",
        )
        self.assertIn(
            "latest_owned_interaction_pose.reset();",
            controller,
            "release must clear the prior owned reference",
        )

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
            "        if (interaction_frame_state.overrides_locomotion_pose)",
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


if __name__ == "__main__":
    unittest.main()
