import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import unittest


_REPOSITORY = Path(__file__).resolve().parents[2]
_TASK4_REPORT = (
    _REPOSITORY / "build" / "smart-pickup" / "beer10-slot-candidates.json"
)
_FLAT_DATABASE = _REPOSITORY / "resources" / "database.bin"

_EXPECTED_DATABASE_SHA256 = (
    "4d3b65f73e9a207988aaaebded36b988f811ec068988c7e829732701d9d2da1b"
)
_EXPECTED_FEATURES_SHA256 = (
    "3b492ca7e5ed12aade5750ff925c689f4acf56f28edc31a4a5f341e434adf145"
)
_EXPECTED_CLIP_COUNT = 2045
_EXPECTED_FRAME_COUNT = 511250
_TARGET_SEQUENCE_ID = "pickup_table__beer_10__001"

_PREVIEW_AUTHORITY = {
    "candidate_subset_authority": False,
    "mode": "global_unrestricted",
    "runtime_clip_allowlist_applied": False,
}


def _f32_hex(value):
    bits = struct.unpack("!I", struct.pack("!f", float(value)))[0]
    return f"0x{bits:08x}"


_TARGET_SCALARS = {
    "approach_direction_object_x": -0.997760296,
    "approach_direction_object_y": 0.0,
    "approach_direction_object_z": 0.0668911785,
    "clearance_radius_m": 0.04,
    "grasp_position_object_x": 0.0930671170,
    "grasp_position_object_y": -0.119263843,
    "grasp_position_object_z": 0.0375832170,
    "grasp_rotation_object_w": 0.308746904,
    "grasp_rotation_object_x": 0.0609171167,
    "grasp_rotation_object_y": -0.155041456,
    "grasp_rotation_object_z": 0.936443567,
    "object_dimensions_x": 0.0645366386,
    "object_dimensions_y": 0.0645366609,
    "object_dimensions_z": 0.240097240,
    "object_position_x": 0.00394439697,
    "object_position_y": 0.503655553,
    "object_position_z": 2.77000808716,
    "object_rotation_w": -0.0669774629,
    "object_rotation_x": 0.670088462,
    "object_rotation_y": 0.0799996098,
    "object_rotation_z": -0.734911910,
    "table_position_x": 0.0,
    "table_position_y": 0.360757500,
    "table_position_z": 3.0,
    "table_rotation_w": 1.0,
    "table_rotation_x": 0.0,
    "table_rotation_y": 0.0,
    "table_rotation_z": 0.0,
    "table_size_x": 2.0,
    "table_size_y": 0.0399999991,
    "table_size_z": 0.600000024,
}

_EXPECTED_TARGET = {
    "contact_local_frame": 126,
    "entry_local_frame": 101,
    "scalars_f32_hex": {
        name: _f32_hex(value) for name, value in _TARGET_SCALARS.items()
    },
    "sequence_id": _TARGET_SEQUENCE_ID,
}

_F32_HEX = re.compile(r"0x[0-9a-f]{8}\Z")
_CANDIDATE_KEYS = {
    "candidate_ordinal",
    "contact_local_frame",
    "entry_local_frame",
    "match_ready",
    "matcher_provenance",
    "path_feasible",
    "preview_authority",
    "prospective_root_x_object_f32_hex",
    "prospective_root_yaw_object_f32_hex",
    "prospective_root_z_object_f32_hex",
    "record_type",
    "sequence_id",
    "source_clip_ordinal",
}
_MATCHER_KEYS = {
    "clip_ordinal",
    "contact_local_frame",
    "entry_local_frame",
    "sequence_id",
}
_SLOT_KEYS = {
    "contact_local_frame",
    "entry_local_frame",
    "prospective_root_x_object_f32_hex",
    "prospective_root_yaw_object_f32_hex",
    "prospective_root_z_object_f32_hex",
    "sequence_id",
    "slot_id",
}
_TARGET_KEYS = {
    "contact_local_frame",
    "entry_local_frame",
    "scalars_f32_hex",
    "sequence_id",
}
_SELECTED_KEYS = {
    "record_type",
    "slots",
    "target",
    "target_registry_registration_count",
}
_SCENE_KEYS = {"record_type", "slots", "target"}


def _fail(message):
    raise AssertionError(message)


def _require(condition, message):
    if not condition:
        _fail(message)


def _require_exact_keys(value, expected, label):
    _require(type(value) is dict, f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        _fail(
            f"{label} keys differ: missing={missing}, "
            f"unexpected={unexpected}"
        )


def _require_int(value, label, minimum=0):
    _require(type(value) is int, f"{label} must be an integer")
    _require(value >= minimum, f"{label} must be at least {minimum}")


def _require_string(value, label):
    _require(
        type(value) is str and bool(value),
        f"{label} must be a nonempty string",
    )


def _require_f32_hex(value, label):
    _require(
        type(value) is str and _F32_HEX.fullmatch(value) is not None,
        f"{label} must be canonical lowercase float32 hexadecimal bits",
    )


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON number {value!r}")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json_object(text, label):
    _require(type(text) is str and bool(text.strip()), f"{label} is empty")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        _fail(f"{label} is not one deterministic JSON object: {error}")
    _require(type(value) is dict, f"{label} must contain one JSON object")
    return value


def _parse_preview_jsonl(text):
    _require(type(text) is str and bool(text), "preview JSONL is empty")
    lines = text.splitlines()
    _require(lines, "preview JSONL has no records")
    for index, line in enumerate(lines):
        _require(bool(line.strip()), f"preview JSONL line {index} is blank")
    records = [
        _parse_json_object(line, f"preview JSONL line {index}")
        for index, line in enumerate(lines)
    ]
    selected_positions = [
        index
        for index, record in enumerate(records)
        if record.get("record_type") == "selected_slots"
    ]
    _require(
        selected_positions == [len(records) - 1],
        "preview JSONL must end with exactly one selected_slots record",
    )
    candidates = records[:-1]
    _require(candidates, "preview JSONL must contain candidate records")
    _require(
        all(record.get("record_type") == "candidate_preview" for record in candidates),
        "every preview record before selected_slots must be candidate_preview",
    )
    return candidates, records[-1]


def _parse_scene_json(text):
    record = _parse_json_object(text, "scene probe output")
    _require(
        record.get("record_type") == "compiled_scene",
        "scene probe must emit one compiled_scene record",
    )
    return record


def _canonical_json_bytes(value):
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        _fail(f"value is not canonical JSON: {error}")
    return text.encode("ascii")


def _load_json(path, label):
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        return _parse_json_object(path.read_text(encoding="utf-8"), label)
    except OSError as error:
        _fail(f"could not read {label} {path}: {error}")


def _sha256(path, label):
    _require(path.is_file(), f"{label} is missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(f"could not read {label} {path}: {error}")
    return digest.hexdigest()


def _manifest_clips(manifest):
    clips = manifest.get("clips")
    _require(type(clips) is list, "manifest clips must be an array")
    sequence_to_ordinal = {}
    for ordinal, clip in enumerate(clips):
        _require(type(clip) is dict, f"manifest clip {ordinal} must be an object")
        sequence_id = clip.get("sequence_id")
        _require_string(sequence_id, f"manifest clip {ordinal} sequence_id")
        _require(
            sequence_id not in sequence_to_ordinal,
            f"manifest sequence_id is duplicated: {sequence_id}",
        )
        start = clip.get("range_start")
        stop = clip.get("range_stop")
        _require_int(start, f"manifest clip {ordinal} range_start")
        _require_int(stop, f"manifest clip {ordinal} range_stop", minimum=1)
        _require(start < stop, f"manifest clip {ordinal} range must be nonempty")
        sequence_to_ordinal[sequence_id] = ordinal
    return clips, sequence_to_ordinal


def _report_candidates(report):
    candidates = report.get("retained_candidates")
    _require(type(candidates) is list, "Task4 retained_candidates must be an array")
    _require(len(candidates) >= 3, "Task4 report must retain at least three candidates")
    dataset_id = report.get("dataset_id")
    schema_version = report.get("schema_version")
    _require_string(dataset_id, "Task4 dataset_id")
    _require_int(schema_version, "Task4 schema_version", minimum=1)
    stable_keys = []
    provenance = set()
    for index, candidate in enumerate(candidates):
        _require(type(candidate) is dict, f"Task4 candidate {index} must be an object")
        sequence_id = candidate.get("sequence_id")
        entry = candidate.get("entry_local_frame")
        contact = candidate.get("contact_local_frame")
        active_hand = candidate.get("active_hand")
        _require_string(sequence_id, f"Task4 candidate {index} sequence_id")
        _require_int(entry, f"Task4 candidate {index} entry_local_frame")
        _require_int(contact, f"Task4 candidate {index} contact_local_frame")
        _require(entry < contact, f"Task4 candidate {index} event order is invalid")
        _require_int(active_hand, f"Task4 candidate {index} active_hand")
        _require(
            candidate.get("static_path_feasible") is True,
            f"Task4 candidate {index} is not statically path-feasible",
        )
        expected_key = [
            dataset_id,
            schema_version,
            sequence_id,
            entry,
            active_hand,
        ]
        _require(
            candidate.get("stable_key") == expected_key,
            f"Task4 candidate {index} stable_key differs from provenance",
        )
        for field in (
            "root_x_object_m",
            "root_z_object_m",
            "root_yaw_object_radians",
        ):
            value = candidate.get(field)
            _require(
                type(value) in (int, float) and math.isfinite(value),
                f"Task4 candidate {index} {field} must be finite",
            )
        key = (sequence_id, entry, contact)
        _require(key not in provenance, f"Task4 provenance is duplicated: {key}")
        provenance.add(key)
        stable_keys.append(expected_key)
    _require(
        stable_keys == sorted(stable_keys),
        "Task4 retained candidates must remain in stable-key dedupe order",
    )
    return candidates


def _validate_matcher_provenance(value, clips, label):
    _require_exact_keys(value, _MATCHER_KEYS, label)
    ordinal = value["clip_ordinal"]
    _require_int(ordinal, f"{label}.clip_ordinal")
    _require(ordinal < len(clips), f"{label}.clip_ordinal is outside manifest")
    clip = clips[ordinal]
    sequence_id = value["sequence_id"]
    _require_string(sequence_id, f"{label}.sequence_id")
    _require(
        sequence_id == clip["sequence_id"],
        f"{label}.clip_ordinal does not join to sequence_id",
    )
    entry = value["entry_local_frame"]
    contact = value["contact_local_frame"]
    _require_int(entry, f"{label}.entry_local_frame")
    _require_int(contact, f"{label}.contact_local_frame")
    clip_length = clip["range_stop"] - clip["range_start"]
    _require(
        0 <= entry < contact < clip_length,
        f"{label} local event frames are outside the manifest clip",
    )


def _validate_preview_candidates(records, manifest, report):
    clips, sequence_to_ordinal = _manifest_clips(manifest)
    expected = _report_candidates(report)
    _require(
        len(records) == len(expected),
        "preview must emit exactly one row per retained Task4 candidate",
    )
    ready = []
    seen_provenance = set()
    for ordinal, (record, source) in enumerate(zip(records, expected)):
        label = f"candidate_preview[{ordinal}]"
        _require_exact_keys(record, _CANDIDATE_KEYS, label)
        _require(record["record_type"] == "candidate_preview", f"{label} type differs")
        _require_int(record["candidate_ordinal"], f"{label}.candidate_ordinal")
        _require(
            record["candidate_ordinal"] == ordinal,
            f"{label}.candidate_ordinal must preserve report order",
        )
        sequence_id = record["sequence_id"]
        _require_string(sequence_id, f"{label}.sequence_id")
        source_ordinal = record["source_clip_ordinal"]
        _require_int(source_ordinal, f"{label}.source_clip_ordinal")
        _require(
            source_ordinal < len(clips),
            f"{label}.source_clip_ordinal is outside manifest",
        )
        _require(
            source_ordinal == sequence_to_ordinal.get(sequence_id),
            f"{label}.source_clip_ordinal does not join to sequence_id",
        )
        _require(
            sequence_id == source["sequence_id"],
            f"{label}.sequence_id differs from Task4 report",
        )
        for field in ("entry_local_frame", "contact_local_frame"):
            _require_int(record[field], f"{label}.{field}")
            _require(
                record[field] == source[field],
                f"{label}.{field} differs from Task4 report",
            )
        source_clip = clips[source_ordinal]
        source_clip_length = (
            source_clip["range_stop"] - source_clip["range_start"]
        )
        _require(
            0
            <= record["entry_local_frame"]
            < record["contact_local_frame"]
            < source_clip_length,
            f"{label} local event frames are outside the source manifest clip",
        )
        root_fields = (
            ("prospective_root_x_object_f32_hex", "root_x_object_m"),
            ("prospective_root_z_object_f32_hex", "root_z_object_m"),
            (
                "prospective_root_yaw_object_f32_hex",
                "root_yaw_object_radians",
            ),
        )
        for output_field, source_field in root_fields:
            _require_f32_hex(record[output_field], f"{label}.{output_field}")
            _require(
                record[output_field] == _f32_hex(source[source_field]),
                f"{label}.{output_field} differs from Task4 float32 authorship",
            )
        _require(
            record["preview_authority"] == _PREVIEW_AUTHORITY,
            f"{label} must prove global unrestricted preview authority",
        )
        _require(
            type(record["path_feasible"]) is bool,
            f"{label}.path_feasible must be boolean",
        )
        _require(
            type(record["match_ready"]) is bool,
            f"{label}.match_ready must be boolean",
        )
        _require(
            not record["match_ready"] or record["path_feasible"],
            f"{label}.match_ready requires path_feasible",
        )
        if record["match_ready"]:
            _validate_matcher_provenance(
                record["matcher_provenance"],
                clips,
                f"{label}.matcher_provenance",
            )
        else:
            _require(
                record["matcher_provenance"] is None,
                f"{label}.matcher_provenance must be null when not ready",
            )
        provenance = (
            sequence_id,
            record["entry_local_frame"],
            record["contact_local_frame"],
        )
        _require(
            provenance not in seen_provenance,
            f"preview source provenance is duplicated: {provenance}",
        )
        seen_provenance.add(provenance)
        if record["path_feasible"] and record["match_ready"]:
            ready.append(record)
    _require(
        len(ready) >= 3,
        "fewer than three retained candidates are runtime-ready and path-feasible",
    )
    return ready[:3]


def _slot_from_preview(record, slot_id):
    return {
        "contact_local_frame": record["contact_local_frame"],
        "entry_local_frame": record["entry_local_frame"],
        "prospective_root_x_object_f32_hex": record[
            "prospective_root_x_object_f32_hex"
        ],
        "prospective_root_yaw_object_f32_hex": record[
            "prospective_root_yaw_object_f32_hex"
        ],
        "prospective_root_z_object_f32_hex": record[
            "prospective_root_z_object_f32_hex"
        ],
        "sequence_id": record["sequence_id"],
        "slot_id": slot_id,
    }


def _validate_target(target, label):
    _require_exact_keys(target, _TARGET_KEYS, label)
    _require_string(target["sequence_id"], f"{label}.sequence_id")
    _require_int(target["entry_local_frame"], f"{label}.entry_local_frame")
    _require_int(target["contact_local_frame"], f"{label}.contact_local_frame")
    scalars = target["scalars_f32_hex"]
    _require(
        type(scalars) is dict,
        f"{label}.scalars_f32_hex must be an object",
    )
    _require(
        set(scalars) == set(_TARGET_SCALARS),
        f"{label}.scalars_f32_hex fields differ from frozen target",
    )
    for name, value in scalars.items():
        _require_f32_hex(value, f"{label}.scalars_f32_hex.{name}")
    _require(target == _EXPECTED_TARGET, f"{label} differs from frozen target bits")


def _validate_slots(slots, label):
    _require(type(slots) is list, f"{label} must be an array")
    _require(len(slots) == 3, f"{label} must contain exactly three slots")
    identifiers = []
    provenance = []
    for index, slot in enumerate(slots):
        slot_label = f"{label}[{index}]"
        _require_exact_keys(slot, _SLOT_KEYS, slot_label)
        _require_int(slot["slot_id"], f"{slot_label}.slot_id", minimum=1)
        _require_string(slot["sequence_id"], f"{slot_label}.sequence_id")
        _require_int(slot["entry_local_frame"], f"{slot_label}.entry_local_frame")
        _require_int(slot["contact_local_frame"], f"{slot_label}.contact_local_frame")
        _require(
            slot["entry_local_frame"] < slot["contact_local_frame"],
            f"{slot_label} event order is invalid",
        )
        for field in (
            "prospective_root_x_object_f32_hex",
            "prospective_root_z_object_f32_hex",
            "prospective_root_yaw_object_f32_hex",
        ):
            _require_f32_hex(slot[field], f"{slot_label}.{field}")
        identifiers.append(slot["slot_id"])
        provenance.append(
            (
                slot["sequence_id"],
                slot["entry_local_frame"],
                slot["contact_local_frame"],
            )
        )
    _require(identifiers == [1, 2, 3], f"{label} IDs must be exactly [1, 2, 3]")
    _require(len(set(identifiers)) == 3, f"{label} IDs must be unique")
    _require(len(set(provenance)) == 3, f"{label} provenance must be unique")


def _authored_structure(record):
    return {"slots": record["slots"], "target": record["target"]}


def _validate_gate_outputs(candidate_records, selected, scene, manifest, report):
    selected_candidates = _validate_preview_candidates(
        candidate_records, manifest, report
    )
    expected_slots = [
        _slot_from_preview(record, slot_id)
        for slot_id, record in enumerate(selected_candidates, start=1)
    ]

    _require_exact_keys(selected, _SELECTED_KEYS, "selected_slots")
    _require(
        selected["record_type"] == "selected_slots",
        "preview final record must be selected_slots",
    )
    _require_int(
        selected["target_registry_registration_count"],
        "selected_slots.target_registry_registration_count",
        minimum=1,
    )
    _require(
        selected["target_registry_registration_count"] == 1,
        "preview must perform exactly one TargetRegistry registration",
    )
    _validate_target(selected["target"], "selected_slots.target")
    _validate_slots(selected["slots"], "selected_slots.slots")
    _require(
        selected["slots"] == expected_slots,
        "selected_slots must be the first three ready rows in Task4 report order",
    )

    _require_exact_keys(scene, _SCENE_KEYS, "compiled_scene")
    _require(
        scene["record_type"] == "compiled_scene",
        "scene record must be compiled_scene",
    )
    _validate_target(scene["target"], "compiled_scene.target")
    _validate_slots(scene["slots"], "compiled_scene.slots")
    _require(
        _canonical_json_bytes(_authored_structure(selected))
        == _canonical_json_bytes(_authored_structure(scene)),
        "preview selected_slots and compiled scene structures differ",
    )
    return expected_slots


def _environment_path(name):
    value = os.environ.get(name)
    _require(
        type(value) is str and bool(value.strip()),
        f"required environment variable {name} is not set",
    )
    path = Path(value)
    if not path.is_absolute():
        path = _REPOSITORY / path
    return path


def _preflight_full_pack(full_pack):
    manifest = _load_json(full_pack / "manifest.json", "full-pack manifest")
    validation = _load_json(
        full_pack / "validation_report.json", "full-pack validation report"
    )
    report = _load_json(_TASK4_REPORT, "Task4 slot candidate report")
    clips, _ = _manifest_clips(manifest)
    _require(
        manifest.get("diagnostic_limit") is None,
        "full-pack manifest diagnostic_limit must be null",
    )
    _require(
        len(clips) == _EXPECTED_CLIP_COUNT,
        f"full-pack manifest must contain {_EXPECTED_CLIP_COUNT} clips",
    )
    _require(
        validation.get("included_frames") == _EXPECTED_FRAME_COUNT,
        f"full-pack validation must report {_EXPECTED_FRAME_COUNT} frames",
    )
    database_hash = _sha256(
        full_pack / "interaction_database.bin", "full-pack database"
    )
    features_hash = _sha256(
        full_pack / "interaction_features.bin", "full-pack features"
    )
    _require(
        database_hash == _EXPECTED_DATABASE_SHA256,
        "full-pack database SHA-256 differs from the reviewed corpus",
    )
    _require(
        features_hash == _EXPECTED_FEATURES_SHA256,
        "full-pack features SHA-256 differs from the reviewed corpus",
    )
    _report_candidates(report)
    _require(
        report.get("target", {}).get("sequence_id") == _TARGET_SEQUENCE_ID,
        "Task4 report target sequence differs from frozen beer target",
    )
    _require(
        report["target"].get("entry_local_frame")
        == _EXPECTED_TARGET["entry_local_frame"],
        "Task4 target entry provenance differs",
    )
    _require(
        report["target"].get("contact_local_frame")
        == _EXPECTED_TARGET["contact_local_frame"],
        "Task4 target contact provenance differs",
    )
    return manifest, report


def _run_probe(arguments, label):
    try:
        completed = subprocess.run(
            [str(argument) for argument in arguments],
            cwd=_REPOSITORY,
            check=False,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _fail(f"{label} could not run deterministically: {error}")
    _require(
        completed.returncode == 0,
        f"{label} exited {completed.returncode}: {completed.stderr}",
    )
    _require(bool(completed.stdout.strip()), f"{label} emitted no JSON evidence")
    return completed.stdout


def _synthetic_fixture():
    dataset_id = "synthetic/smart-pickup"
    sequences = [
        "pickup_table__alpha__001",
        "pickup_table__bravo__001",
        "pickup_table__charlie__001",
        "pickup_table__delta__001",
        "pickup_table__echo__001",
    ]
    clips = [
        {
            "range_start": index * 250,
            "range_stop": (index + 1) * 250,
            "sequence_id": sequence_id,
        }
        for index, sequence_id in enumerate(sequences)
    ]
    manifest = {"clips": clips}
    roots = [
        (-0.40, -0.20, 1.00),
        (-0.35, -0.10, 1.20),
        (-0.30, 0.00, 1.40),
        (-0.25, 0.10, 1.60),
    ]
    report_candidates = []
    preview_candidates = []
    readiness = (True, False, True, True)
    matcher_ordinals = (4, None, 1, 2)
    for ordinal, (root, ready, matcher_ordinal) in enumerate(
        zip(roots, readiness, matcher_ordinals)
    ):
        sequence_id = sequences[ordinal]
        entry = 40 + ordinal
        contact = 65 + ordinal
        report_candidates.append(
            {
                "active_hand": 1,
                "contact_local_frame": contact,
                "entry_local_frame": entry,
                "root_x_object_m": root[0],
                "root_yaw_object_radians": root[2],
                "root_z_object_m": root[1],
                "sequence_id": sequence_id,
                "stable_key": [dataset_id, 1, sequence_id, entry, 1],
                "static_path_feasible": True,
            }
        )
        matcher = None
        if matcher_ordinal is not None:
            matcher = {
                "clip_ordinal": matcher_ordinal,
                "contact_local_frame": 90 + ordinal,
                "entry_local_frame": 70 + ordinal,
                "sequence_id": sequences[matcher_ordinal],
            }
        preview_candidates.append(
            {
                "candidate_ordinal": ordinal,
                "contact_local_frame": contact,
                "entry_local_frame": entry,
                "match_ready": ready,
                "matcher_provenance": matcher,
                "path_feasible": True,
                "preview_authority": copy.deepcopy(_PREVIEW_AUTHORITY),
                "prospective_root_x_object_f32_hex": _f32_hex(root[0]),
                "prospective_root_yaw_object_f32_hex": _f32_hex(root[2]),
                "prospective_root_z_object_f32_hex": _f32_hex(root[1]),
                "record_type": "candidate_preview",
                "sequence_id": sequence_id,
                "source_clip_ordinal": ordinal,
            }
        )
    report = {
        "dataset_id": dataset_id,
        "retained_candidates": report_candidates,
        "schema_version": 1,
        "target": {
            "contact_local_frame": _EXPECTED_TARGET["contact_local_frame"],
            "entry_local_frame": _EXPECTED_TARGET["entry_local_frame"],
            "sequence_id": _TARGET_SEQUENCE_ID,
        },
    }
    selected_candidates = [
        candidate
        for candidate in preview_candidates
        if candidate["path_feasible"] and candidate["match_ready"]
    ][:3]
    slots = [
        _slot_from_preview(candidate, slot_id)
        for slot_id, candidate in enumerate(selected_candidates, start=1)
    ]
    selected = {
        "record_type": "selected_slots",
        "slots": copy.deepcopy(slots),
        "target": copy.deepcopy(_EXPECTED_TARGET),
        "target_registry_registration_count": 1,
    }
    scene = {
        "record_type": "compiled_scene",
        "slots": copy.deepcopy(slots),
        "target": copy.deepcopy(_EXPECTED_TARGET),
    }
    return manifest, report, preview_candidates, selected, scene


def _canonical_json(value):
    return _canonical_json_bytes(value).decode("ascii")


def _mutated_hex(value):
    return f"0x{int(value, 16) ^ 1:08x}"


class SmartPickupContractTests(unittest.TestCase):
    def assert_contract_rejected(
        self, candidate_records, selected, scene, manifest, report
    ):
        with self.assertRaises(AssertionError):
            _validate_gate_outputs(
                candidate_records, selected, scene, manifest, report
            )

    def test_parses_and_accepts_canonical_synthetic_probe_records(self):
        manifest, report, candidates, selected, scene = _synthetic_fixture()
        preview_text = "\n".join(
            _canonical_json(record) for record in [*candidates, selected]
        ) + "\n"
        scene_text = _canonical_json(scene) + "\n"

        parsed_candidates, parsed_selected = _parse_preview_jsonl(preview_text)
        parsed_scene = _parse_scene_json(scene_text)
        expected_slots = _validate_gate_outputs(
            parsed_candidates,
            parsed_selected,
            parsed_scene,
            manifest,
            report,
        )

        self.assertEqual([slot["slot_id"] for slot in expected_slots], [1, 2, 3])
        self.assertEqual(
            [slot["sequence_id"] for slot in expected_slots],
            [
                "pickup_table__alpha__001",
                "pickup_table__charlie__001",
                "pickup_table__delta__001",
            ],
        )

    def test_parser_requires_one_final_summary_and_one_scene_record(self):
        _, _, candidates, selected, scene = _synthetic_fixture()
        candidate_lines = [_canonical_json(record) for record in candidates]
        selected_line = _canonical_json(selected)

        with self.assertRaises(AssertionError):
            _parse_preview_jsonl("\n".join(candidate_lines) + "\n")
        with self.assertRaises(AssertionError):
            _parse_preview_jsonl(
                "\n".join([*candidate_lines, selected_line, selected_line])
                + "\n"
            )
        with self.assertRaises(AssertionError):
            _parse_scene_json(
                _canonical_json(scene) + "\n" + _canonical_json(scene) + "\n"
            )

    def test_rejects_each_baked_slot_field_and_target_scalar_mutation(self):
        manifest, report, candidates, selected, scene = _synthetic_fixture()
        for field in sorted(_SLOT_KEYS):
            with self.subTest(slot_field=field):
                mutated_scene = copy.deepcopy(scene)
                value = mutated_scene["slots"][1][field]
                if field.endswith("_f32_hex"):
                    value = _mutated_hex(value)
                elif field == "sequence_id":
                    value += "__mutated"
                else:
                    value += 1
                mutated_scene["slots"][1][field] = value
                self.assert_contract_rejected(
                    candidates, selected, mutated_scene, manifest, report
                )

        for scalar in sorted(_TARGET_SCALARS):
            with self.subTest(target_scalar=scalar):
                mutated_scene = copy.deepcopy(scene)
                current = mutated_scene["target"]["scalars_f32_hex"][scalar]
                mutated_scene["target"]["scalars_f32_hex"][scalar] = (
                    _mutated_hex(current)
                )
                self.assert_contract_rejected(
                    candidates, selected, mutated_scene, manifest, report
                )

    def test_rejects_order_cardinality_identity_and_authority_mutations(self):
        manifest, report, candidates, selected, scene = _synthetic_fixture()

        order = copy.deepcopy(scene)
        order["slots"][0], order["slots"][1] = (
            order["slots"][1],
            order["slots"][0],
        )
        self.assert_contract_rejected(candidates, selected, order, manifest, report)

        added = copy.deepcopy(scene)
        added_slot = copy.deepcopy(added["slots"][-1])
        added_slot["slot_id"] = 4
        added["slots"].append(added_slot)
        self.assert_contract_rejected(candidates, selected, added, manifest, report)

        dropped = copy.deepcopy(scene)
        dropped["slots"].pop()
        self.assert_contract_rejected(candidates, selected, dropped, manifest, report)

        duplicate_id = copy.deepcopy(scene)
        duplicate_id["slots"][1]["slot_id"] = duplicate_id["slots"][0]["slot_id"]
        self.assert_contract_rejected(
            candidates, selected, duplicate_id, manifest, report
        )

        duplicate_provenance = copy.deepcopy(scene)
        for field in (
            "sequence_id",
            "entry_local_frame",
            "contact_local_frame",
        ):
            duplicate_provenance["slots"][1][field] = (
                duplicate_provenance["slots"][0][field]
            )
        self.assert_contract_rejected(
            candidates, selected, duplicate_provenance, manifest, report
        )

        for registration_count in (0, 2):
            with self.subTest(registration_count=registration_count):
                registrations = copy.deepcopy(selected)
                registrations["target_registry_registration_count"] = (
                    registration_count
                )
                self.assert_contract_rejected(
                    candidates, registrations, scene, manifest, report
                )

        authority_mutations = {
            "candidate_subset_authority": True,
            "mode": "candidate_subset",
            "runtime_clip_allowlist_applied": True,
        }
        for field, value in authority_mutations.items():
            with self.subTest(preview_authority=field):
                authority = copy.deepcopy(candidates)
                authority[0]["preview_authority"][field] = value
                self.assert_contract_rejected(
                    authority, selected, scene, manifest, report
                )

        for name, candidate_rows in (
            ("added", [*copy.deepcopy(candidates), copy.deepcopy(candidates[-1])]),
            ("dropped", copy.deepcopy(candidates[:-1])),
        ):
            with self.subTest(candidate_row=name):
                self.assert_contract_rejected(
                    candidate_rows, selected, scene, manifest, report
                )


class SmartPickupFullPackGateTests(unittest.TestCase):
    def test_full_pack_preview_and_compiled_scene_are_identical(self):
        full_pack = _environment_path("SMART_PICKUP_FULL_PACK")
        preview_probe = _environment_path("SMART_PICKUP_PREVIEW_PROBE")
        scene_probe = _environment_path("SMART_PICKUP_SCENE_PROBE")

        manifest, report = _preflight_full_pack(full_pack)

        unavailable = []
        for label, path in (
            ("preview probe", preview_probe),
            ("scene probe", scene_probe),
        ):
            if not path.is_file() or not os.access(path, os.X_OK):
                unavailable.append(f"{label}: {path}")
        _require(
            not unavailable,
            "required smart-pickup probe executables are unavailable: "
            + ", ".join(unavailable),
        )
        _require(
            _FLAT_DATABASE.is_file(),
            f"ordinary flat runtime database is missing: {_FLAT_DATABASE}",
        )

        preview_text = _run_probe(
            [preview_probe, _FLAT_DATABASE, full_pack, _TASK4_REPORT],
            "smart-pickup preview probe",
        )
        scene_text = _run_probe(
            [scene_probe, "--json"], "smart-pickup scene probe"
        )
        candidate_records, selected = _parse_preview_jsonl(preview_text)
        scene = _parse_scene_json(scene_text)

        _validate_gate_outputs(
            candidate_records, selected, scene, manifest, report
        )


if __name__ == "__main__":
    unittest.main()
