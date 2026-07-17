"""Pure SONIC delivery, tracking, contact, threshold, and verdict metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .artifacts import RunBundle, verify_run_inventory
from .coordinator import (
    LOGGER_JOINT_PERMUTATION,
    DeliveryAudit,
    DeliveryAuditEvidence,
    parse_official_target_row,
)
from .external import (
    CURRENT_FRAME_ADVANCEMENT_SHA256,
    VerifiedExternal,
    VerifiedGearCheckout,
)
from .joints import ContractError
from .timeline import CanonicalTargetBuffer
from .zmq_v1 import DecodedPoseV1, decode_pose_v1


PINNED_GEAR_COMMIT = "60de0df7ffedeef415fe58d435e92cc5b01ba3d9"
PINNED_CURRENT_FRAME_ADVANCEMENT_SHA256 = CURRENT_FRAME_ADVANCEMENT_SHA256
FORBIDDEN_CONTACT_GROUPS = frozenset(("pelvis", "knees", "torso", "hands"))
REGISTERED_TERRAIN_SCENE_IDS = (
    "grail-curb-low",
    "ramp-10-up-down",
    "stairs-shallow",
)
TARGET_RADIUS_M = 0.25
DURATION_MULTIPLIER = 1.25
MINIMUM_PELVIS_LOCAL_HEIGHT_M = 0.45
MINIMUM_PELVIS_UP_DOT = 0.5
REFERENCE_PENETRATION_LIMIT_M = 0.005
KNOWN_GOOD_METRIC_MULTIPLIER = 1.5
KNOWN_GOOD_ZERO_EPSILON = 1.0e-8
STAGE_A_GATE_NAMES = (
    "external_identity",
    "joint_projection_round_trip",
    "basis_and_scene_alignment",
    "flat_mm_kinematic_replay",
    "known_good_file_dynamic",
    "known_good_stream_delivery",
    "known_good_stream_dynamic",
)
STAGE_A_IDENTITY_KEYS = frozenset(
    (
        "policy",
        "encoder",
        "observation_config",
        "external_commit",
        "official_model_xml",
        "generated_flat_scene",
        "initial_qpos",
        "mm_reference_buffer",
        "known_good_reference_buffer",
    )
)
_STAGE_A_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiments"
    / "stage_a.json"
)
_STAGE_A_EVIDENCE_KEYS = frozenset(
    (
        "schema",
        "registry_sha256",
        "command",
        "mode",
        "argv",
        "invocation_cwd",
        "environment",
        "command_status",
        "stage_a_status",
        "gates",
        "identity",
        "identity_sha256",
        "metrics",
        "outputs",
    )
)
_STAGE_A_GATE_KEYS = frozenset(
    (
        "name",
        "status",
        "reason",
        "identity",
        "evidence_hashes",
        "metrics",
        "outputs",
    )
)
_STAGE_A_ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
)
_STAGE_A_REQUIRED_OPTIONS = frozenset(
    (
        "mode",
        "gear-checkout",
        "policy",
        "observation-config",
        "source-mjcf",
        "terrain-dir",
        "output-root",
    )
)
_STAGE_A_OPTION_NAMES = _STAGE_A_REQUIRED_OPTIONS | {"encoder"}
_STAGE_A_EXTERNAL_OPTIONS = MappingProxyType(
    {
        "gear-checkout": "gear_checkout",
        "policy": "policy",
        "observation-config": "observation_config",
        "source-mjcf": "source_mjcf",
        "terrain-dir": "terrain_dir",
        "encoder": "encoder",
    }
)
_STAGE_A_PRIMARY_OUTPUTS = MappingProxyType(
    {name: f"gates/{name}.json" for name in STAGE_A_GATE_NAMES}
)
_STAGE_A_GATE_HASH_KEYS = MappingProxyType(
    {
        name: frozenset(
            (
                f"{name}_sha256",
                *(
                    ("delivery_audit_sha256",)
                    if name == "known_good_stream_delivery"
                    else ()
                ),
            )
        )
        for name in STAGE_A_GATE_NAMES
    }
)
_STAGE_A_DELIVERY_AUDIT_SUMMARY = MappingProxyType(
    {
        "readiness_publications": 1,
        "logical_publications": 22,
        "padding_publications": 1,
        "receipt_fence_publications": 1,
        "consumer_markers": 25,
    }
)


def _stage_a_json_bytes(value: object, label: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} is not finite JSON data") from error


def stage_a_identity_sha256(identity: Mapping[str, str]) -> str:
    """Hash the exact complete Stage A experiment identity."""

    if not isinstance(identity, Mapping) or set(identity) != STAGE_A_IDENTITY_KEYS:
        raise ContractError("Stage A identity has invalid keys")
    copied = dict(identity)
    for name, value in copied.items():
        if type(name) is not str or type(value) is not str or not value:
            raise ContractError("Stage A identity values must be nonempty strings")
        if name == "external_commit":
            if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
                raise ContractError("Stage A external commit is invalid")
        elif len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ContractError(f"Stage A identity {name} is not a SHA-256")
    return hashlib.sha256(_stage_a_json_bytes(copied, "Stage A identity")).hexdigest()


def _stage_a_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"duplicate Stage A evidence key: {key}")
        output[key] = value
    return output


def _stage_a_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stage_a_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContractError(f"{label} must be one regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ContractError(f"{label} changed while being read")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise ContractError(f"{label} changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _stage_a_output_sha256(bundle: Path, relative: object) -> str:
    if type(relative) is not str:
        raise ContractError("Stage A output path is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != relative
        or relative in ("manifest.json", "inventory.json")
    ):
        raise ContractError("Stage A output is not confined")
    return hashlib.sha256(
        _stage_a_regular_bytes(
            bundle.joinpath(*pure.parts), f"Stage A output {relative}"
        )
    ).hexdigest()


def _stage_a_invocation(argv: object) -> Mapping[str, str]:
    if (
        type(argv) is not list
        or len(argv) < 2
        or argv[0:2] != ["mm_sonic.cli", "stage-a"]
        or any(type(argument) is not str or not argument for argument in argv)
    ):
        raise ContractError("Stage A argv is invalid")
    options: dict[str, str] = {}
    index = 2
    while index < len(argv):
        argument = argv[index]
        if not argument.startswith("--") or argument == "--":
            raise ContractError("Stage A argv contains a positional argument")
        raw_option = argument[2:]
        if "=" in raw_option:
            option, value = raw_option.split("=", 1)
            index += 1
        else:
            option = raw_option
            if index + 1 >= len(argv):
                raise ContractError("Stage A argv option is incomplete")
            value = argv[index + 1]
            if value.startswith("--"):
                raise ContractError("Stage A argv option has an empty value")
            index += 2
        if option not in _STAGE_A_OPTION_NAMES:
            raise ContractError(f"Stage A argv option is unknown: --{option}")
        if not value:
            raise ContractError(f"Stage A argv option is empty: --{option}")
        if option in options:
            raise ContractError(f"Stage A argv option is duplicated: --{option}")
        options[option] = value
    if set(options) - {"encoder"} != _STAGE_A_REQUIRED_OPTIONS:
        raise ContractError("Stage A argv is missing a required option")
    if options["mode"] != "known-good-stream":
        raise ContractError("Stage A argv is not known-good-stream")
    return MappingProxyType(options)


def _stage_a_invocation_cwd(value: object) -> Path:
    if type(value) is not str or not value:
        raise ContractError("Stage A invocation cwd is invalid")
    if value.startswith("~"):
        raise ContractError("Stage A invocation cwd uses user expansion")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ContractError("Stage A invocation cwd is not absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractError("Stage A invocation cwd cannot be resolved") from error
    if value != str(resolved) or not resolved.is_dir():
        raise ContractError("Stage A invocation cwd is not canonical")
    return resolved


def _stage_a_resolved_path(
    value: object,
    label: str,
    *,
    invocation_cwd: Path | None,
) -> Path:
    if type(value) is not str or not value:
        raise ContractError(f"{label} path is invalid")
    if value.startswith("~"):
        raise ContractError(f"{label} path uses user expansion")
    candidate = Path(value)
    if not candidate.is_absolute():
        if invocation_cwd is None:
            raise ContractError(f"{label} path is not absolute")
        candidate = invocation_cwd / candidate
    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractError(f"{label} path cannot be resolved") from error


def validate_stage_a_prerequisite(
    evidence_path: str | os.PathLike[str],
    *,
    expected_identity: Mapping[str, str] | None,
) -> Mapping[str, object]:
    """Authenticate a complete Stage A pass before any Stage B/C launch."""

    candidate = Path(evidence_path).expanduser()
    if not verify_run_inventory(candidate.parent):
        raise ContractError("Stage A evidence inventory did not authenticate")
    try:
        observed = candidate.lstat()
    except OSError as error:
        raise ContractError("Stage A evidence is unavailable") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ContractError("Stage A evidence must be a regular non-symlink file")
    if observed.st_nlink != 1:
        raise ContractError("Stage A evidence must not be a hard-link alias")
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractError("Stage A evidence cannot be resolved") from error
    if path.name != "stage-a-evidence.json" or path.parent != candidate.parent.resolve():
        raise ContractError("Stage A evidence path is not canonical")
    try:
        raw = _stage_a_regular_bytes(path, "Stage A evidence")
        document = json.loads(
            raw.decode("ascii"), object_pairs_hook=_stage_a_no_duplicates
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("Stage A evidence is not canonical JSON") from error
    if type(document) is not dict or raw != _stage_a_json_bytes(
        document, "Stage A evidence"
    ):
        raise ContractError("Stage A evidence is not canonical JSON")
    if document.get("schema") != "mm-sonic-stage-a-evidence/v1":
        raise ContractError("Stage A evidence schema is invalid")
    if set(document) != _STAGE_A_EVIDENCE_KEYS:
        raise ContractError("Stage A evidence has invalid keys")
    registry_raw = _stage_a_regular_bytes(
        _STAGE_A_REGISTRY_PATH, "current Stage A registry"
    )
    registry_sha256 = hashlib.sha256(registry_raw).hexdigest()
    if document.get("registry_sha256") != registry_sha256:
        raise ContractError("Stage A evidence registry is stale")
    if document.get("command") != "stage-a":
        raise ContractError("Stage A evidence command is invalid")
    if document.get("mode") != "known-good-stream":
        raise ContractError("Stage A evidence mode must be known-good-stream")
    argv = document.get("argv")
    invocation = _stage_a_invocation(argv)
    if invocation["mode"] != document["mode"]:
        raise ContractError("Stage A evidence argv changed")
    invocation_cwd = _stage_a_invocation_cwd(
        document.get("invocation_cwd")
    )
    output_root = _stage_a_resolved_path(
        invocation["output-root"],
        "Stage A argv output-root",
        invocation_cwd=invocation_cwd,
    )
    bundle = path.parent
    if bundle.parent.name != "stage-a" or output_root != bundle.parent.parent:
        raise ContractError("Stage A argv output-root does not contain this bundle")
    environment = document.get("environment")
    if (
        type(environment) is not dict
        or set(environment) != {"allowlist", "values"}
        or environment.get("allowlist") != list(_STAGE_A_ENVIRONMENT_ALLOWLIST)
        or type(environment.get("values")) is not dict
        or any(
            name not in _STAGE_A_ENVIRONMENT_ALLOWLIST
            or type(value) is not str
            for name, value in environment["values"].items()
        )
    ):
        raise ContractError("Stage A evidence environment is invalid")
    gates = document.get("gates")
    if type(gates) is not list or len(gates) != len(STAGE_A_GATE_NAMES):
        raise ContractError("Stage A gate evidence is incomplete")
    reconstructed_identity: dict[str, str] = {}
    reconstructed_metrics: dict[str, object] = {}
    reconstructed_outputs: dict[str, str] = {}
    for index, name in enumerate(STAGE_A_GATE_NAMES):
        gate = gates[index]
        if (
            type(gate) is not dict
            or set(gate) != _STAGE_A_GATE_KEYS
            or gate.get("name") != name
        ):
            raise ContractError("Stage A gate order changed")
        if gate.get("status") != "pass":
            if name.startswith("known_good_stream"):
                raise ContractError(
                    "Stage A stream delivery and dynamic gates must pass"
                )
            raise ContractError(f"Stage A gate did not pass: {name}")
        if gate.get("reason") is not None:
            raise ContractError("passing Stage A gate has a reason")
        gate_identity = gate.get("identity")
        evidence_hashes = gate.get("evidence_hashes")
        gate_metrics = gate.get("metrics")
        gate_outputs = gate.get("outputs")
        if (
            type(gate_identity) is not dict
            or type(evidence_hashes) is not dict
            or not evidence_hashes
            or type(gate_metrics) is not dict
            or type(gate_outputs) is not dict
            or not gate_outputs
        ):
            raise ContractError("Stage A passing gate evidence is incomplete")
        if any(
            type(hash_name) is not str
            or not hash_name
            or not _stage_a_sha256(digest)
            for hash_name, digest in evidence_hashes.items()
        ):
            raise ContractError("Stage A gate evidence hashes are invalid")
        if set(evidence_hashes) != _STAGE_A_GATE_HASH_KEYS[name]:
            raise ContractError("Stage A gate evidence hash keys changed")
        for identity_name, identity_value in gate_identity.items():
            if identity_name not in STAGE_A_IDENTITY_KEYS:
                raise ContractError("Stage A gate identity key is invalid")
            current = reconstructed_identity.get(identity_name)
            if current is not None and current != identity_value:
                raise ContractError("Stage A gate identities conflict")
            reconstructed_identity[identity_name] = identity_value
        if gate_metrics:
            reconstructed_metrics[name] = gate_metrics
        for relative, digest in gate_outputs.items():
            if not _stage_a_sha256(digest):
                raise ContractError("Stage A gate output digest is invalid")
            if relative in reconstructed_outputs:
                raise ContractError("Stage A gate output is duplicated")
            if _stage_a_output_sha256(path.parent, relative) != digest:
                raise ContractError("Stage A gate output digest changed")
            reconstructed_outputs[relative] = digest
        primary_relative = _STAGE_A_PRIMARY_OUTPUTS[name]
        primary_digest = gate_outputs.get(primary_relative)
        primary_hash_name = f"{name}_sha256"
        if (
            not _stage_a_sha256(primary_digest)
            or evidence_hashes.get(primary_hash_name) != primary_digest
        ):
            raise ContractError("Stage A gate primary evidence hash changed")
        if name == "known_good_stream_delivery":
            primary_raw = _stage_a_regular_bytes(
                path.parent / primary_relative,
                "Stage A stream delivery primary output",
            )
            try:
                primary_payload = json.loads(
                    primary_raw.decode("ascii"),
                    object_pairs_hook=_stage_a_no_duplicates,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ContractError(
                    "Stage A stream delivery primary output is invalid"
                ) from error
            if (
                type(primary_payload) is not dict
                or primary_raw
                != _stage_a_json_bytes(
                    primary_payload, "Stage A stream delivery primary output"
                )
            ):
                raise ContractError(
                    "Stage A stream delivery primary output is not canonical"
                )
            delivery_audit = primary_payload.get("delivery_audit")
            if type(delivery_audit) is not dict:
                raise ContractError("Stage A stream delivery audit is missing")
            audit_core = dict(delivery_audit)
            embedded_digest = audit_core.pop("evidence_sha256", None)
            if not _stage_a_sha256(embedded_digest):
                raise ContractError("Stage A stream delivery audit hash is invalid")
            for summary_name, summary_value in (
                _STAGE_A_DELIVERY_AUDIT_SUMMARY.items()
            ):
                observed_summary = audit_core.pop(summary_name, None)
                if (
                    type(observed_summary) is not int
                    or observed_summary != summary_value
                ):
                    raise ContractError(
                        "Stage A stream delivery audit summary changed"
                    )
            computed_audit_digest = hashlib.sha256(
                _stage_a_json_bytes(
                    audit_core, "Stage A stream delivery audit core"
                )
            ).hexdigest()
            if (
                embedded_digest != computed_audit_digest
                or evidence_hashes.get("delivery_audit_sha256")
                != computed_audit_digest
            ):
                raise ContractError("Stage A stream delivery audit hash changed")
    if (
        document.get("command_status") != "pass"
        or document.get("stage_a_status") != "pass"
    ):
        raise ContractError("Stage A evidence is not a complete pass")
    identity = document.get("identity")
    if type(identity) is not dict or identity != reconstructed_identity:
        raise ContractError("Stage A identity is missing")
    identity_hash = stage_a_identity_sha256(identity)
    if document.get("identity_sha256") != identity_hash:
        raise ContractError("Stage A identity digest is invalid")
    if expected_identity is not None:
        expected_hash = stage_a_identity_sha256(expected_identity)
        if expected_hash != identity_hash or dict(expected_identity) != dict(identity):
            raise ContractError("Stage A identity does not match this experiment")
    if document.get("metrics") != reconstructed_metrics:
        raise ContractError("Stage A top-level metrics do not match its gates")
    if document.get("outputs") != reconstructed_outputs:
        raise ContractError("Stage A top-level outputs do not match its gates")

    manifest_path = path.parent / "manifest.json"
    try:
        manifest_raw = _stage_a_regular_bytes(
            manifest_path, "Stage A terminal manifest"
        )
        manifest = json.loads(
            manifest_raw.decode("ascii"), object_pairs_hook=_stage_a_no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("Stage A terminal manifest is invalid") from error
    if type(manifest) is not dict:
        raise ContractError("Stage A terminal manifest is invalid")
    if manifest.get("status") != "complete" or manifest.get("outcome") != {
        "stage_a_status": "pass",
        "status": "pass",
    }:
        raise ContractError("Stage A terminal manifest is not an exact pass")
    processes = manifest.get("processes")
    if (
        type(processes) is not list
        or not processes
        or type(processes[0]) is not dict
        or set(processes[0])
        != {"name", "argv", "invocation_cwd", "environment"}
        or processes[0]
        != {
            "name": "mm_sonic.cli",
            "argv": argv,
            "invocation_cwd": str(invocation_cwd),
            "environment": environment,
        }
    ):
        raise ContractError("Stage A terminal invocation does not match evidence")
    external = manifest.get("external")
    if type(external) is not dict:
        raise ContractError("Stage A terminal external inputs are invalid")
    for option_name, manifest_name in _STAGE_A_EXTERNAL_OPTIONS.items():
        argument_value = invocation.get(option_name)
        manifest_value = external.get(manifest_name)
        if option_name == "encoder" and argument_value is None:
            if manifest_value is not None:
                raise ContractError("Stage A argv encoder changed")
            continue
        if argument_value is None or manifest_value is None:
            raise ContractError(f"Stage A argv {option_name} changed")
        if _stage_a_resolved_path(
            argument_value,
            f"Stage A argv {option_name}",
            invocation_cwd=invocation_cwd,
        ) != _stage_a_resolved_path(
            manifest_value,
            f"Stage A manifest external.{manifest_name}",
            invocation_cwd=None,
        ):
            raise ContractError(f"Stage A argv {option_name} changed")
    return MappingProxyType(document)


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be finite")
    return float(value)


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        output = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be numeric") from error
    if output.shape != shape or not np.all(np.isfinite(output)):
        raise ContractError(f"{label} must have shape {shape} with finite values")
    return output


def _frame_indices(value: object, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.shape[0] <= 0
        or raw.dtype.kind not in "iu"
        or raw.dtype.kind == "b"
    ):
        raise ContractError(f"{label} frame indices must be a nonempty integer vector")
    indices = raw.astype(np.int64, copy=True)
    if np.any(indices < 0) or not np.all(np.diff(indices) == 1):
        raise ContractError(f"{label} frame indices must be unique and contiguous")
    return indices


@dataclass(frozen=True)
class TrackingMetrics:
    joint_position_rmse_rad: float
    pelvis_orientation_rms_rad: float
    joint_tracking_trace_rad: tuple[float, ...]
    pelvis_tracking_trace_rad: tuple[float, ...]


def tracking_metrics(
    target_frame_index: object,
    target_joint_position: object,
    target_pelvis_quat_wxyz: object,
    actual_frame_index: object,
    actual_joint_position: object,
    actual_pelvis_quat_wxyz: object,
) -> TrackingMetrics:
    """Score exact aligned 50 Hz rows without averaging per-joint RMSEs."""

    target_indices = _frame_indices(target_frame_index, "target")
    actual_indices = _frame_indices(actual_frame_index, "actual")
    if not np.array_equal(target_indices, actual_indices):
        raise ContractError("target and actual frame coverage must be exactly aligned")
    count = int(target_indices.shape[0])
    target_joint = _finite_array(
        target_joint_position, (count, 29), "target joint position"
    )
    actual_joint = _finite_array(
        actual_joint_position, (count, 29), "actual joint position"
    )
    target_quat = _finite_array(
        target_pelvis_quat_wxyz, (count, 4), "target pelvis quaternion"
    )
    actual_quat = _finite_array(
        actual_pelvis_quat_wxyz, (count, 4), "actual pelvis quaternion"
    )
    for label, quaternions in (
        ("target", target_quat),
        ("actual", actual_quat),
    ):
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(np.abs(norms - 1.0) > 1.0e-6):
            raise ContractError(f"{label} pelvis values must be unit quaternions")

    joint_error = actual_joint - target_joint
    joint_trace = np.sqrt(np.mean(np.square(joint_error), axis=1))
    joint_rmse = math.sqrt(float(np.mean(np.square(joint_error))))
    dots = np.sum(target_quat * actual_quat, axis=1)
    angles = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    pelvis_rms = math.sqrt(float(np.mean(np.square(angles))))
    return TrackingMetrics(
        joint_position_rmse_rad=joint_rmse,
        pelvis_orientation_rms_rad=pelvis_rms,
        joint_tracking_trace_rad=tuple(float(value) for value in joint_trace),
        pelvis_tracking_trace_rad=tuple(float(value) for value in angles),
    )


def minimum_local_pelvis_height(
    pelvis_position_mujoco_xyz: object,
    local_terrain_height_m: object,
) -> float:
    positions = np.asarray(pelvis_position_mujoco_xyz, dtype=np.float64)
    terrain = np.asarray(local_terrain_height_m, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] <= 0
        or positions.shape[1] != 3
        or terrain.shape != (positions.shape[0],)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(terrain))
    ):
        raise ContractError(
            "local pelvis height requires aligned finite xyz/terrain rows"
        )
    return float(np.min(positions[:, 2] - terrain))


def pelvis_up_dots(pelvis_quat_wxyz: object) -> np.ndarray:
    quaternions = np.asarray(pelvis_quat_wxyz, dtype=np.float64)
    if (
        quaternions.ndim != 2
        or quaternions.shape[0] <= 0
        or quaternions.shape[1] != 4
        or not np.all(np.isfinite(quaternions))
    ):
        raise ContractError("pelvis up-dot requires finite wxyz quaternion rows")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(np.abs(norms - 1.0) > 1.0e-6):
        raise ContractError("pelvis up-dot requires unit quaternions")
    _w, x, y, _z = np.moveaxis(quaternions, -1, 0)
    return 1.0 - 2.0 * (np.square(x) + np.square(y))


def _contact_groups(groups: Iterable[str]) -> tuple[str, ...]:
    values = tuple(groups)
    if any(type(value) is not str or not value for value in values):
        raise ContractError("contact groups must be nonempty strings")
    return values


def has_forbidden_contact(groups: Iterable[str]) -> bool:
    return any(group in FORBIDDEN_CONTACT_GROUPS for group in _contact_groups(groups))


def within_target_radius(distance_m: float) -> bool:
    distance = _finite(distance_m, "target distance")
    return distance >= 0.0 and distance <= TARGET_RADIUS_M


def within_duration(reached_time_s: float, nominal_duration_s: float) -> bool:
    reached = _finite(reached_time_s, "target time")
    nominal = _finite(nominal_duration_s, "nominal duration")
    return reached >= 0.0 and nominal > 0.0 and reached <= DURATION_MULTIPLIER * nominal


def has_reference_penetration(
    contact_groups: Iterable[str], penetration_depth_m: Iterable[float]
) -> bool:
    groups = _contact_groups(contact_groups)
    depths = tuple(_finite(value, "penetration depth") for value in penetration_depth_m)
    if len(groups) != len(depths) or any(value < 0.0 for value in depths):
        raise ContractError("penetration rows must align and be nonnegative")
    return any(
        group in FORBIDDEN_CONTACT_GROUPS
        and depth > REFERENCE_PENETRATION_LIMIT_M
        for group, depth in zip(groups, depths, strict=True)
    )


def within_known_good_ratio(candidate: float, known_good: float) -> bool:
    candidate_value = _finite(candidate, "candidate metric")
    known_good_value = _finite(known_good, "known-good metric")
    if candidate_value < 0.0 or known_good_value < 0.0:
        raise ContractError("tracking metrics must be nonnegative")
    if known_good_value <= KNOWN_GOOD_ZERO_EPSILON:
        return candidate_value <= KNOWN_GOOD_ZERO_EPSILON
    return candidate_value <= KNOWN_GOOD_METRIC_MULTIPLIER * known_good_value


def nearest_rank_summary(samples: Iterable[int | float]) -> dict[str, int | float]:
    values = tuple(samples)
    if not values:
        raise ContractError("timing summary requires at least one sample")
    for value in values:
        _finite(value, "timing sample")
        if float(value) < 0.0:
            raise ContractError("timing samples must be nonnegative")
    ordered = sorted(values)

    def nearest(percentile: float):
        rank = max(1, int(math.ceil(percentile * len(ordered))))
        return ordered[rank - 1]

    return {"p50": nearest(0.50), "p95": nearest(0.95), "p99": nearest(0.99)}


@dataclass(frozen=True)
class SecondaryMetrics:
    swing_foot_scuff_count: int
    minimum_foot_clearance_m: float
    horizontal_path_drift_m: float
    joint_tracking_trace_rad: tuple[float, ...]
    pelvis_tracking_trace_rad: tuple[float, ...]
    contact_impulses_ns: tuple[float, ...]
    policy_execution_timing_ns: Mapping[str, int | float] | None

    def __post_init__(self) -> None:
        if (
            type(self.swing_foot_scuff_count) is not int
            or self.swing_foot_scuff_count < 0
        ):
            raise ContractError("swing-foot scuff count must be nonnegative")
        for label, value in (
            ("minimum foot clearance", self.minimum_foot_clearance_m),
            ("horizontal path drift", self.horizontal_path_drift_m),
        ):
            _finite(value, label)
        if self.horizontal_path_drift_m < 0.0:
            raise ContractError("horizontal path drift must be nonnegative")
        for label in ("joint_tracking_trace_rad", "pelvis_tracking_trace_rad"):
            values = tuple(getattr(self, label))
            if not values:
                raise ContractError(f"{label} must not be empty")
            for value in values:
                if _finite(value, label) < 0.0:
                    raise ContractError(f"{label} must be nonnegative")
            object.__setattr__(self, label, values)
        impulses = tuple(self.contact_impulses_ns)
        for value in impulses:
            if _finite(value, "contact_impulses_ns") < 0.0:
                raise ContractError("contact_impulses_ns must be nonnegative")
        object.__setattr__(self, "contact_impulses_ns", impulses)
        if self.policy_execution_timing_ns is not None:
            timing = dict(self.policy_execution_timing_ns)
            if set(timing) != {"p50", "p95", "p99"}:
                raise ContractError("policy timing must contain p50/p95/p99")
            for value in timing.values():
                if _finite(value, "policy timing") < 0.0:
                    raise ContractError("policy timing must be nonnegative")
            if not (timing["p50"] <= timing["p95"] <= timing["p99"]):
                raise ContractError("policy timing quantiles must be ordered")
            object.__setattr__(
                self, "policy_execution_timing_ns", MappingProxyType(timing)
            )


@dataclass(frozen=True)
class DynamicTrialVerdict:
    integration_pass: bool
    dynamic_pass: bool
    checks: Mapping[str, bool]
    secondary: SecondaryMetrics


def evaluate_terrain_trial(
    *,
    integration_pass: bool,
    exact_frame_coverage: bool,
    target_distance_m: float,
    reached_time_s: float,
    nominal_duration_s: float,
    minimum_pelvis_local_height_m: float,
    minimum_pelvis_up_dot: float,
    contact_groups: Iterable[str],
    secondary: SecondaryMetrics,
) -> DynamicTrialVerdict:
    if type(integration_pass) is not bool or type(exact_frame_coverage) is not bool:
        raise ContractError("trial integration/coverage gates must be booleans")
    if not isinstance(secondary, SecondaryMetrics):
        raise ContractError("trial requires complete secondary metrics")
    height = _finite(minimum_pelvis_local_height_m, "minimum local pelvis height")
    up_dot = _finite(minimum_pelvis_up_dot, "minimum pelvis up-dot")
    checks = {
        "integration": integration_pass,
        "exact_frame_coverage": exact_frame_coverage,
        "target_radius": within_target_radius(target_distance_m),
        "duration": within_duration(reached_time_s, nominal_duration_s),
        "pelvis_local_height": height >= MINIMUM_PELVIS_LOCAL_HEIGHT_M,
        "pelvis_up_dot": up_dot >= MINIMUM_PELVIS_UP_DOT,
        "forbidden_contacts": not has_forbidden_contact(contact_groups),
    }
    return DynamicTrialVerdict(
        integration_pass=integration_pass,
        dynamic_pass=all(checks.values()),
        checks=MappingProxyType(checks),
        secondary=secondary,
    )


def evaluate_dynamic_trial(
    *,
    integration_pass: bool,
    exact_frame_coverage: bool,
    target_distance_m: float,
    reached_time_s: float,
    nominal_duration_s: float,
    minimum_pelvis_local_height_m: float,
    minimum_pelvis_up_dot: float,
    contact_groups: Iterable[str],
    secondary: SecondaryMetrics,
) -> DynamicTrialVerdict:
    """Backward-compatible name for the Stage C terrain trial evaluator."""

    return evaluate_terrain_trial(
        integration_pass=integration_pass,
        exact_frame_coverage=exact_frame_coverage,
        target_distance_m=target_distance_m,
        reached_time_s=reached_time_s,
        nominal_duration_s=nominal_duration_s,
        minimum_pelvis_local_height_m=minimum_pelvis_local_height_m,
        minimum_pelvis_up_dot=minimum_pelvis_up_dot,
        contact_groups=contact_groups,
        secondary=secondary,
    )


def evaluate_flat_trial(
    *,
    integration_pass: bool,
    exact_command_coverage: bool,
    exact_frame_coverage: bool,
    exact_control_duration: bool,
    exact_safety_log_coverage: bool,
    minimum_pelvis_local_height_m: float,
    minimum_pelvis_up_dot: float,
    contact_groups: Iterable[str],
    joint_position_rmse_rad: float,
    known_good_joint_position_rmse_rad: float,
    pelvis_orientation_rms_rad: float,
    known_good_pelvis_orientation_rms_rad: float,
    secondary: SecondaryMetrics,
) -> DynamicTrialVerdict:
    """Evaluate every registered Stage B safety, coverage, and tracking gate."""

    booleans = (
        integration_pass,
        exact_command_coverage,
        exact_frame_coverage,
        exact_control_duration,
        exact_safety_log_coverage,
    )
    if any(type(value) is not bool for value in booleans):
        raise ContractError("flat trial integration/coverage gates must be booleans")
    if not isinstance(secondary, SecondaryMetrics):
        raise ContractError("flat trial requires complete secondary metrics")
    height = _finite(minimum_pelvis_local_height_m, "minimum local pelvis height")
    up_dot = _finite(minimum_pelvis_up_dot, "minimum pelvis up-dot")
    checks = {
        "integration": integration_pass,
        "exact_command_coverage": exact_command_coverage,
        "exact_frame_coverage": exact_frame_coverage,
        "exact_control_duration": exact_control_duration,
        "exact_safety_log_coverage": exact_safety_log_coverage,
        "pelvis_local_height": height >= MINIMUM_PELVIS_LOCAL_HEIGHT_M,
        "pelvis_up_dot": up_dot >= MINIMUM_PELVIS_UP_DOT,
        "forbidden_contacts": not has_forbidden_contact(contact_groups),
        "joint_tracking_ratio": within_known_good_ratio(
            joint_position_rmse_rad,
            known_good_joint_position_rmse_rad,
        ),
        "pelvis_tracking_ratio": within_known_good_ratio(
            pelvis_orientation_rms_rad,
            known_good_pelvis_orientation_rms_rad,
        ),
    }
    return DynamicTrialVerdict(
        integration_pass=integration_pass,
        dynamic_pass=all(checks.values()),
        checks=MappingProxyType(checks),
        secondary=secondary,
    )


@dataclass(frozen=True)
class SceneHypothesisVerdict:
    aware_successes: int
    blind_successes: int
    pass_margin: int
    composition_feasible: bool
    awareness_effect: str
    hypothesis_supported: bool


@dataclass(frozen=True)
class OverallHypothesisVerdict:
    scene_verdicts: Mapping[str, SceneHypothesisVerdict]
    composition_feasible: bool
    awareness_effect: str
    hypothesis_supported: bool


def aggregate_scene_hypothesis(
    aware_successes: int, blind_successes: int
) -> SceneHypothesisVerdict:
    for label, value in (
        ("aware successes", aware_successes),
        ("blind successes", blind_successes),
    ):
        if type(value) is not int or value < 0 or value > 10:
            raise ContractError(f"{label} must be an integer in 0..10")
    margin = aware_successes - blind_successes
    aware_pass = aware_successes >= 8
    blind_pass = blind_successes >= 8
    if aware_pass and blind_pass:
        effect = "inconclusive"
    elif aware_pass and margin >= 3:
        effect = "supported"
    else:
        effect = "not_supported"
    return SceneHypothesisVerdict(
        aware_successes=aware_successes,
        blind_successes=blind_successes,
        pass_margin=margin,
        composition_feasible=aware_pass,
        awareness_effect=effect,
        hypothesis_supported=effect == "supported",
    )


def aggregate_overall_hypothesis(
    scene_verdicts: Mapping[str, SceneHypothesisVerdict],
) -> OverallHypothesisVerdict:
    """Require a verdict for every registered class; never average classes."""

    if not isinstance(scene_verdicts, Mapping):
        raise ContractError("overall hypothesis requires three registered classes")
    copied = dict(scene_verdicts)
    if set(copied) != set(REGISTERED_TERRAIN_SCENE_IDS) or any(
        not isinstance(value, SceneHypothesisVerdict) for value in copied.values()
    ):
        raise ContractError("overall hypothesis requires three registered classes")
    ordered = {
        scene_id: copied[scene_id] for scene_id in REGISTERED_TERRAIN_SCENE_IDS
    }
    composition = all(value.composition_feasible for value in ordered.values())
    if all(value.hypothesis_supported for value in ordered.values()):
        effect = "supported"
    elif any(value.awareness_effect == "not_supported" for value in ordered.values()):
        effect = "not_supported"
    else:
        effect = "inconclusive"
    return OverallHypothesisVerdict(
        scene_verdicts=MappingProxyType(ordered),
        composition_feasible=composition,
        awareness_effect=effect,
        hypothesis_supported=effect == "supported",
    )


def attribute_failure(
    *,
    stage: str,
    integration_pass: bool,
    known_good_file_pass: bool,
    known_good_stream_pass: bool,
    kinematic_pass: bool,
    dynamic_pass: bool,
) -> str | None:
    booleans = (
        integration_pass,
        known_good_file_pass,
        known_good_stream_pass,
        kinematic_pass,
        dynamic_pass,
    )
    if type(stage) is not str or stage not in {"A", "B", "C"}:
        raise ContractError("failure attribution stage must be A, B, or C")
    if any(type(value) is not bool for value in booleans):
        raise ContractError("failure attribution gates must be booleans")
    if not integration_pass:
        return "integration"
    if not known_good_file_pass:
        return "gear_setup_model_simulator_or_harness"
    if not known_good_stream_pass:
        return "bridge_protocol"
    if stage == "B":
        if not kinematic_pass:
            return "joint_or_coordinate_conversion"
        if not dynamic_pass:
            return "reference_distribution_or_sonic_tracking"
    if stage == "C":
        if not kinematic_pass:
            return "motion_matching_reference"
        if not dynamic_pass:
            return "low_level_tracking_contact_domain_mismatch"
    if stage == "A" and not dynamic_pass:
        return "gear_setup_model_simulator_or_harness"
    return None


def validate_trial_verdict_semantics(document: Mapping[str, object]) -> None:
    """Reject verdict-field combinations that contradict available evidence."""

    if not isinstance(document, Mapping):
        raise ContractError("trial verdict must be a mapping")
    try:
        stage = document["stage"]
        integration_pass = document["integration_pass"]
        kinematic_pass = document["kinematic_pass"]
        dynamic_pass = document["dynamic_pass"]
        failure_layer = document["failure_layer"]
    except KeyError as error:
        raise ContractError(
            f"trial verdict is missing semantic field: {error.args[0]}"
        ) from error
    if stage not in {"A", "B", "C"}:
        raise ContractError("trial verdict stage must be A, B, or C")
    if any(
        type(value) is not bool
        for value in (integration_pass, kinematic_pass, dynamic_pass)
    ):
        raise ContractError("trial verdict pass fields must be booleans")
    if failure_layer is not None and type(failure_layer) is not str:
        raise ContractError("trial verdict failure layer must be a string or null")

    early_layers = {
        "gear_setup_model_simulator_or_harness",
        "bridge_protocol",
    }
    stage_layers = {
        "A": early_layers,
        "B": early_layers
        | {
            "joint_or_coordinate_conversion",
            "reference_distribution_or_sonic_tracking",
        },
        "C": early_layers
        | {
            "motion_matching_reference",
            "low_level_tracking_contact_domain_mismatch",
        },
    }
    if not integration_pass:
        if dynamic_pass or failure_layer != "integration":
            raise ContractError(
                "terminal integration failure requires integration attribution"
            )
        return
    if failure_layer == "integration":
        raise ContractError(
            "integration attribution requires integration_pass=false"
        )

    if (
        stage == "A"
        and dynamic_pass
        and not kinematic_pass
        and (failure_layer is None or failure_layer in early_layers)
    ):
        return
    if stage == "A":
        should_succeed = dynamic_pass
        local_layer = None
    elif not kinematic_pass:
        should_succeed = False
        local_layer = (
            "joint_or_coordinate_conversion"
            if stage == "B"
            else "motion_matching_reference"
        )
    elif not dynamic_pass:
        should_succeed = False
        local_layer = (
            "reference_distribution_or_sonic_tracking"
            if stage == "B"
            else "low_level_tracking_contact_domain_mismatch"
        )
    else:
        should_succeed = True
        local_layer = None
    if should_succeed:
        if failure_layer is not None:
            raise ContractError("passing trial verdict requires null failure layer")
        return
    if failure_layer is None:
        raise ContractError("failed trial verdict requires failure attribution")
    if failure_layer not in stage_layers[stage]:
        raise ContractError("trial verdict failure attribution contradicts its stage")
    if failure_layer not in early_layers and failure_layer != local_layer:
        raise ContractError("trial verdict failure attribution contradicts pass fields")


def _validate_class_verdict_semantics(
    record: Mapping[str, object],
) -> tuple[str, SceneHypothesisVerdict]:
    required = {
        "scene_id",
        "aware_successes",
        "blind_successes",
        "scored_trials_per_condition",
        "aware_integration_failures",
        "blind_integration_failures",
        "kinematic_defects",
        "pass_margin",
        "composition_verdict",
        "awareness_verdict",
    }
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise ContractError("hypothesis class verdict is incomplete")
    scene_id = record["scene_id"]
    if scene_id not in REGISTERED_TERRAIN_SCENE_IDS:
        raise ContractError("hypothesis class scene is not registered")
    scored = record["scored_trials_per_condition"]
    if type(scored) is not int or scored != 10:
        raise ContractError("hypothesis class scored trial count must equal 10")
    aware = record["aware_successes"]
    blind = record["blind_successes"]
    integration_failures = (
        record["aware_integration_failures"],
        record["blind_integration_failures"],
    )
    try:
        aggregate = aggregate_scene_hypothesis(aware, blind)
    except ContractError as error:
        raise ContractError("hypothesis class success counts are invalid") from error
    if any(
        type(value) is not int or value < 0 or value > scored
        for value in integration_failures
    ) or aware + integration_failures[0] > scored or blind + integration_failures[1] > scored:
        raise ContractError(
            "hypothesis class counts exceed scored trials per condition"
        )
    defects = record["kinematic_defects"]
    if type(defects) is not int or defects < 0 or defects > 2:
        raise ContractError("hypothesis class kinematic defect count is invalid")
    if record["pass_margin"] != aggregate.pass_margin:
        raise ContractError("hypothesis class pass_margin contradicts success counts")
    expected_composition = (
        "supported" if aggregate.composition_feasible else "not_supported"
    )
    if record["composition_verdict"] != expected_composition:
        raise ContractError(
            "hypothesis class composition verdict contradicts success counts"
        )
    if record["awareness_verdict"] != aggregate.awareness_effect:
        raise ContractError(
            "hypothesis class awareness verdict contradicts success counts"
        )
    return scene_id, aggregate


def validate_hypothesis_verdict_semantics(
    document: Mapping[str, object],
) -> None:
    """Recompute every arithmetic, per-class, and overall hypothesis field."""

    if not isinstance(document, Mapping):
        raise ContractError("hypothesis verdict must be a mapping")
    status = document.get("status")
    classes = document.get("terrain_classes")
    overall_claim = document.get("overall_hypothesis")
    if status not in {"complete", "incomplete", "not_run"}:
        raise ContractError("hypothesis verdict status is invalid")
    if type(classes) not in (list, tuple):
        raise ContractError("hypothesis terrain_classes must be an array")
    if status == "not_run":
        if classes or overall_claim is not None:
            raise ContractError(
                "not_run hypothesis cannot contain class or overall results"
            )
        return
    aggregates: dict[str, SceneHypothesisVerdict] = {}
    for record in classes:
        scene_id, aggregate = _validate_class_verdict_semantics(record)
        if scene_id in aggregates:
            raise ContractError("hypothesis terrain class is duplicated")
        aggregates[scene_id] = aggregate
    if status == "incomplete":
        if overall_claim is not None:
            raise ContractError("incomplete hypothesis requires null overall result")
        return
    if set(aggregates) != set(REGISTERED_TERRAIN_SCENE_IDS):
        raise ContractError("complete hypothesis requires three registered classes")
    expected = aggregate_overall_hypothesis(aggregates).awareness_effect
    if overall_claim != expected:
        raise ContractError(
            "overall hypothesis contradicts the three class verdicts"
        )


def _canonical_bytes(buffer: CanonicalTargetBuffer) -> bytes:
    if not isinstance(buffer, CanonicalTargetBuffer):
        raise ContractError("delivery audit requires a canonical target buffer")
    expected_indices = np.arange(buffer.count, dtype=np.int64)
    if not np.array_equal(buffer.frame_index, expected_indices):
        raise ContractError("canonical delivery frame indices must be exactly 0..N-1")
    return b"".join(
        (
            np.asarray(buffer.joint_position, dtype="<f4").tobytes(order="C"),
            np.asarray(buffer.joint_velocity, dtype="<f4").tobytes(order="C"),
            np.asarray(buffer.body_quat_w, dtype="<f4").tobytes(order="C"),
            np.asarray(buffer.frame_index, dtype="<i8").tobytes(order="C"),
        )
    )


def _canonical_json_bytes(value: object, label: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ContractError(
            f"delivery transcript {label} is not canonical JSON"
        ) from error


class _EvidenceTranscript:
    """Length-delimited, domain-separated digest of every audited input."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.add("domain", b"mm-sonic-delivery-evidence/v1")

    def add(self, label: str, payload: bytes) -> None:
        encoded_label = label.encode("ascii")
        self._digest.update(b"\x01")
        self._digest.update(len(encoded_label).to_bytes(4, "big"))
        self._digest.update(encoded_label)
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def add_json(self, label: str, value: object) -> None:
        self.add(label, _canonical_json_bytes(value, label))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _official_row(buffer: CanonicalTargetBuffer, index: int) -> bytes:
    permutation = np.asarray(LOGGER_JOINT_PERMUTATION, dtype=np.int64)
    values = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            buffer.body_quat_w[index],
            buffer.joint_position[index, permutation],
        )
    ).astype(np.float32, copy=False)
    return (
        ",".join(format(float(value), ".6g") for value in values) + ",\n"
    ).encode("ascii")


def _relative_parts(relative: object) -> tuple[str, ...]:
    if type(relative) is not str or not relative:
        raise ContractError("delivery archive path must be nonempty")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ContractError("delivery archive path is not confined")
    return path.parts


def _read_regular_confined(run: RunBundle, relative: object) -> bytes:
    if not isinstance(run, RunBundle):
        raise ContractError("production delivery audit requires a RunBundle")
    parts = _relative_parts(relative)
    try:
        run._require_path_identity()
        retained = getattr(run, "_directory_fd", -1)
        if type(retained) is not int or retained < 0:
            raise ContractError("delivery run descriptor is unavailable")
        root_fd = os.dup(retained)
    except OSError as error:
        raise ContractError("cannot retain delivery run evidence") from error
    descriptor = root_fd
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            if descriptor != root_fd:
                os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ContractError("delivery archive must be one regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise ContractError("cannot read confined delivery evidence") from error
    finally:
        if descriptor != root_fd:
            os.close(descriptor)
        os.close(root_fd)


def _read_durable_readiness(
    evidence: DeliveryAuditEvidence,
    transcript: _EvidenceTranscript,
) -> Mapping[str, object]:
    if not isinstance(evidence.run, RunBundle):
        raise ContractError("production delivery audit requires a RunBundle")
    raw = _read_regular_confined(evidence.run, "readiness.jsonl")
    transcript.add("readiness.durable_jsonl", raw)
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or raw.count(b"\n") != 1:
        raise ContractError("delivery audit requires one durable readiness record")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ContractError("durable readiness record contains duplicate keys")
            output[key] = value
        return output

    try:
        envelope = json.loads(
            raw[:-1],
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"invalid durable readiness constant: {value}")
            ),
        )
        canonical = (
            json.dumps(
                envelope,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (UnicodeDecodeError, json.JSONDecodeError, UnicodeEncodeError) as error:
        raise ContractError("durable readiness record is invalid JSON") from error
    if (
        type(envelope) is not dict
        or set(envelope) != {"kind", "sequence", "recorded_utc", "record"}
        or envelope["kind"] != "readiness"
        or type(envelope["sequence"]) is not int
        or envelope["sequence"] <= 0
        or type(envelope["recorded_utc"]) is not str
        or not envelope["recorded_utc"]
        or type(envelope["record"]) is not dict
        or canonical != raw
    ):
        raise ContractError("durable readiness record has an invalid envelope")
    return envelope["record"]


def _read_publication(
    evidence: DeliveryAuditEvidence,
    publication: Mapping[str, object],
    phase: str,
    transcript: _EvidenceTranscript,
    transcript_label: str,
    archive_paths: set[str],
) -> DecodedPoseV1:
    expected_keys = {
        "first_frame_index",
        "last_frame_index",
        "phase",
        "attempt",
        "message_path",
        "digest_path",
        "sha256",
        "size",
        "local_send_completed",
    }
    record = dict(publication)
    transcript.add_json(f"{transcript_label}.metadata", record)
    if set(record) != expected_keys or record["local_send_completed"] is not True:
        raise ContractError("delivery publication metadata is incomplete")
    if record["phase"] != phase:
        raise ContractError("delivery publication phase changed")
    if phase == "timeline" and record["attempt"] is not None:
        raise ContractError("timeline delivery publication has a readiness attempt")
    if phase == "readiness" and (
        type(record["attempt"]) is not int or record["attempt"] <= 0
    ):
        raise ContractError("readiness delivery publication has an invalid attempt")
    if (
        type(record["first_frame_index"]) is not int
        or type(record["last_frame_index"]) is not int
        or record["first_frame_index"] < 0
        or record["last_frame_index"] < record["first_frame_index"]
    ):
        raise ContractError("delivery publication frame range is invalid")
    stem = (
        f"{record['first_frame_index']:06d}-{record['last_frame_index']:06d}"
    )
    if phase == "readiness":
        leaf_stem = f"attempt-{record['attempt']:06d}__{stem}"
        directory = "transmitted/readiness"
    else:
        leaf_stem = stem
        directory = "transmitted"
    expected_message_path = f"{directory}/{leaf_stem}.bin"
    expected_digest_path = f"{directory}/{leaf_stem}.sha256"
    if (
        record["message_path"] != expected_message_path
        or record["digest_path"] != expected_digest_path
    ):
        raise ContractError("delivery publication archive path changed")
    for path in (expected_message_path, expected_digest_path):
        if path in archive_paths:
            raise ContractError("delivery publication archive path is duplicated")
        archive_paths.add(path)
    if not isinstance(evidence.run, RunBundle):
        raise ContractError("production delivery audit requires a RunBundle")
    message = _read_regular_confined(evidence.run, record["message_path"])
    transcript.add(f"{transcript_label}.message", message)
    digest = hashlib.sha256(message).hexdigest()
    if (
        type(record["sha256"]) is not str
        or record["sha256"] != digest
        or type(record["size"]) is not int
        or record["size"] != len(message)
    ):
        raise ContractError("archived delivery digest or size changed")
    sidecar = _read_regular_confined(evidence.run, record["digest_path"])
    transcript.add(f"{transcript_label}.sidecar", sidecar)
    expected_sidecar = f"{digest}  {leaf_stem}.bin\n".encode("ascii")
    if sidecar != expected_sidecar:
        raise ContractError("archived delivery digest sidecar changed")
    decoded = decode_pose_v1(message, baseline_mode=True)
    if (
        type(record["first_frame_index"]) is not int
        or type(record["last_frame_index"]) is not int
        or record["first_frame_index"] != int(decoded.frame_index[0])
        or record["last_frame_index"] != int(decoded.frame_index[-1])
    ):
        raise ContractError("archived delivery frame range changed")
    return decoded


def _decoded_equals_canonical(
    decoded: DecodedPoseV1,
    canonical: CanonicalTargetBuffer,
    indices: np.ndarray,
) -> bool:
    if not np.array_equal(decoded.frame_index, indices):
        return False
    if np.any(indices < 0) or np.any(indices >= canonical.count):
        return False
    comparisons = (
        (decoded.joint_position, canonical.joint_position[indices], "<f4"),
        (decoded.joint_velocity, canonical.joint_velocity[indices], "<f4"),
        (decoded.body_quat_w, canonical.body_quat_w[indices], "<f4"),
    )
    return all(
        np.asarray(actual, dtype=dtype).tobytes(order="C")
        == np.asarray(expected, dtype=dtype).tobytes(order="C")
        for actual, expected, dtype in comparisons
    )


class ProductionDeliveryAuditor:
    """Recompute terminal delivery evidence from authenticated immutable inputs."""

    def __init__(
        self, verified_gear: VerifiedGearCheckout | VerifiedExternal
    ) -> None:
        if not isinstance(verified_gear, (VerifiedGearCheckout, VerifiedExternal)):
            raise ContractError("delivery auditor requires verified GEAR inputs")
        if verified_gear.gear_commit != PINNED_GEAR_COMMIT:
            raise ContractError("delivery auditor GEAR commit is not pinned")
        digest = verified_gear.hashes.get("gear:current_frame_advancement_source")
        if digest != PINNED_CURRENT_FRAME_ADVANCEMENT_SHA256:
            raise ContractError(
                "delivery auditor requires the pinned CurrentFrameAdvancement "
                "source hash"
            )
        self._current_frame_advancement_sha256 = digest
        self._gear_commit = verified_gear.gear_commit

    def audit(self, evidence: DeliveryAuditEvidence) -> DeliveryAudit:
        if not isinstance(evidence, DeliveryAuditEvidence):
            raise ContractError("delivery auditor requires immutable evidence")
        canonical = evidence.canonical_buffer
        canonical_bytes = _canonical_bytes(canonical)
        transcript = _EvidenceTranscript()
        transcript.add_json(
            "auditor.identity",
            {
                "gear_commit": self._gear_commit,
                "current_frame_advancement_sha256": (
                    self._current_frame_advancement_sha256
                ),
            },
        )
        transcript.add("session_id", evidence.session_id.encode("utf-8"))
        transcript.add_json(
            "canonical.layout",
            {
                "count": canonical.count,
                "joint_position_shape": list(canonical.joint_position.shape),
                "joint_velocity_shape": list(canonical.joint_velocity.shape),
                "body_quat_w_shape": list(canonical.body_quat_w.shape),
                "frame_index_shape": list(canonical.frame_index.shape),
                "float_dtype": "little-endian-f32",
                "index_dtype": "little-endian-i64",
            },
        )
        transcript.add("canonical.bytes", canonical_bytes)
        transcript.add("official_log_slice", evidence.official_log_slice)

        expected_readiness = {
            "session_id": evidence.readiness.session_id,
            "official_log_path": evidence.readiness.official_log_path,
            "log_identity": [
                evidence.readiness.log_device,
                evidence.readiness.log_inode,
            ],
            "readiness_log_start_offset": evidence.readiness.readiness_log_start_offset,
            "scoring_log_offset": evidence.readiness.scoring_log_offset,
            "readiness_attempts": evidence.readiness.readiness_attempts,
            "expected_row_sha256": evidence.readiness.expected_row_sha256,
        }
        transcript.add_json("readiness.evidence", expected_readiness)
        try:
            durable_readiness = dict(_read_durable_readiness(evidence, transcript))
            readiness_exact = (
                durable_readiness == expected_readiness
                and evidence.readiness.session_id == evidence.session_id
                and evidence.readiness.expected_row_sha256
                == hashlib.sha256(_official_row(canonical, 0)).hexdigest()
                and evidence.readiness.scoring_log_offset
                >= evidence.readiness.readiness_log_start_offset
            )
        except ContractError:
            readiness_exact = False

        transmitted_exact = readiness_exact
        decoded_indices: list[int] = []
        archive_paths: set[str] = set()
        transcript.add_json(
            "publication.counts",
            {
                "readiness": len(evidence.readiness_publications),
                "timeline": len(evidence.timeline_publications),
            },
        )
        if (
            len(evidence.readiness_publications)
            != evidence.readiness.readiness_attempts
        ):
            transmitted_exact = False
        for expected_attempt, publication in enumerate(
            evidence.readiness_publications, start=1
        ):
            try:
                decoded = _read_publication(
                    evidence,
                    publication,
                    "readiness",
                    transcript,
                    f"publication.readiness.{expected_attempt:06d}",
                    archive_paths,
                )
                if publication["attempt"] != expected_attempt:
                    transmitted_exact = False
                indices = np.array([0], dtype=np.int64)
                if not _decoded_equals_canonical(decoded, canonical, indices):
                    transmitted_exact = False
            except (ContractError, KeyError, IndexError, TypeError, ValueError):
                transmitted_exact = False
        decoded_indices.append(0)
        for publication_index, publication in enumerate(
            evidence.timeline_publications, start=1
        ):
            try:
                decoded = _read_publication(
                    evidence,
                    publication,
                    "timeline",
                    transcript,
                    f"publication.timeline.{publication_index:06d}",
                    archive_paths,
                )
                indices = np.asarray(decoded.frame_index, dtype=np.int64)
                decoded_indices.extend(int(value) for value in indices)
                if not _decoded_equals_canonical(decoded, canonical, indices):
                    transmitted_exact = False
            except (ContractError, KeyError, IndexError, TypeError, ValueError):
                transmitted_exact = False
        if decoded_indices != list(range(canonical.count)):
            transmitted_exact = False

        official_exact = readiness_exact
        official_rows = evidence.official_log_slice.splitlines(keepends=True)
        observed_rows = 1 + len(official_rows)
        if (
            not evidence.official_log_slice.endswith(b"\n")
            and evidence.official_log_slice
        ):
            official_exact = False
        if len(official_rows) != canonical.count - 1:
            official_exact = False
        for cursor, row in enumerate(official_rows, start=1):
            try:
                parsed = parse_official_target_row(row)
            except ContractError:
                official_exact = False
                continue
            if cursor >= canonical.count or parsed != _official_row(canonical, cursor):
                official_exact = False

        return DeliveryAudit(
            expected_rows=canonical.count,
            observed_rows=observed_rows,
            transmitted_indices_exact=transmitted_exact,
            official_rows_exact=official_exact,
            evidence_sha256=transcript.hexdigest(),
        )
