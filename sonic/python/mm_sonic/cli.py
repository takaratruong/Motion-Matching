"""Non-interactive Stage A orchestration for MM-to-SONIC integration."""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Callable, IO, Mapping, Protocol, Sequence
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from .artifacts import RunBundle
from .commands import CommandSample, command_script_bytes, flat_command_script
from .coordinator import (
    LOGGER_JOINT_PERMUTATION,
    SessionConfig,
    SourceValidator,
)
from .external import (
    ExternalInputError,
    ExternalInputs,
    KNOWN_GOOD_REFERENCE_FILES,
    locked_gear_capability_paths,
    verify_external,
)
from .joints import (
    ContractError,
    SOURCE_JOINT_ORDER,
    contract_json_bytes,
    generate_joint_contract,
    load_joint_contract,
    reorder_source_to_target,
)
from .metrics import (
    SecondaryMetrics,
    STAGE_A_GATE_NAMES,
    STAGE_A_IDENTITY_KEYS,
    evaluate_flat_trial,
    stage_a_identity_sha256,
    tracking_metrics,
    validate_stage_a_prerequisite,
)
from .process import (
    GatedSimulatorClient,
    GearProcess,
    MMChunkClient,
    ProcessError,
    _RemoteMMError,
    _gear_process_argv,
    _linux_group_states,
)
from .reference import ReferenceDiagnostics, write_reference_bundle
from .scene import (
    FLAT_SCENE_ID,
    HOLDEN_COORDINATE_SIGNATURE,
    HOLDEN_TO_MUJOCO_MATRIX,
    MUJOCO_COORDINATE_SIGNATURE,
    RegisteredScene,
    register_scene,
    replay_kinematic_reference,
    verify_mm_scene_identity,
)
from .schema import (
    JointFeasibilityIdentity,
    parse_joint_feasibility_identity,
)
from .stage_b import (
    STAGE_B_CONTROL_LEAD_ROWS,
    STAGE_B_FRAME_COUNT,
    STAGE_B_REFERENCE_DURATION_S,
    build_stream_transport_plan,
    canonicalize_sonic_transport,
    load_flat_simulator_safety,
    validate_stage_b_coverage,
)
from .timeline import CanonicalTargetBuffer, TargetTimeline
from .transform import (
    holden_to_mujoco_quaternions,
    holden_to_mujoco_vectors,
    mujoco_to_holden_quaternions,
    mujoco_to_holden_vectors,
)
from .zmq_v1 import PosePublisher, decode_pose_v1


EXIT_PASS = 0
EXIT_CONFIGURATION = 2
EXIT_SCIENTIFIC = 3
EXIT_NOT_RUN = 4

_SONIC_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _SONIC_ROOT.parent
_LOCK_PATH = _SONIC_ROOT / "configs/gear_sonic.lock.json"
_JOINT_CONTRACT_PATH = _SONIC_ROOT / "configs/g1_joint_contract.json"
_SCENE_REGISTRY_PATH = _SONIC_ROOT / "configs/scene_registry.json"
_STAGE_A_REGISTRY_PATH = _SONIC_ROOT / "configs/experiments/stage_a.json"
_STAGE_B_REGISTRY_PATH = _SONIC_ROOT / "configs/experiments/stage_b.json"
_GEAR_TARGET_ORDER_RELATIVE = Path("gear_sonic/envs/manager_env/robots/g1.py")
_GEAR_MODEL_RELATIVE = Path("gear_sonic_deploy/g1/scene_29dof_with_hand.xml")
_GEAR_ROBOT_RELATIVE = Path("gear_sonic_deploy/g1/g1_29dof_with_hand.xml")
_GEAR_MESHES_RELATIVE = Path("gear_sonic_deploy/g1/meshes")
_GEAR_BINARY_RELATIVE = Path(
    "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
)
_GEAR_MODEL_SHA256 = (
    "f8538904eb47cada1bfb2dcdc157099092aa63df4307d7e077b651b16bfb6c74"
)
_GEAR_ROBOT_SHA256 = (
    "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"
)
_FLAT_ROUTE_ID = "flat-12s"
_KNOWN_GOOD_LEAF = "known_good"
_DYNAMIC_RATE_HZ = 50
_JOINT_CERTIFICATION_SAMPLES = 260
_JOINT_CERTIFICATION_SEED = 20260715
_JOINT_CERTIFICATION_TOLERANCE = 1.0e-4
_SAFE_ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
)
_GATE_STATUS = frozenset(
    ("pass", "integration_failure", "scientific_failure", "not_run")
)
_ARTIFACT_HASH_KEYS = frozenset(
    (
        "policy",
        "encoder",
        "observation_config",
        "model",
        "source_mjcf",
        "motion",
        "terrain",
        "joint_map",
        "scene",
    )
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_LOCAL_ZMQ_ENDPOINT = re.compile(
    r"^tcp://(?P<host>127\.0\.0\.1):(?P<port>[1-9][0-9]{0,4})$"
)
_REMOTE_FINITE_NUMBER = (
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
_REMOTE_JOINT_LIMIT_MESSAGE = re.compile(
    rf"\Ajoint (?P<joint>[a-z][a-z0-9_]*_joint) "
    rf"position (?P<position>{_REMOTE_FINITE_NUMBER}) is outside range "
    rf"\[(?P<lower>{_REMOTE_FINITE_NUMBER}), "
    rf"(?P<upper>{_REMOTE_FINITE_NUMBER})\]\Z"
)
_REMOTE_NO_SAFE_CANDIDATE_MESSAGE = (
    "no joint-limit-safe database candidate"
)
_REMOTE_NO_INERTIALIZED_SAFE_CANDIDATE_MESSAGE = (
    "no inertialized-joint-safe database candidate"
)
_COLD_GEAR_STARTUP_TIMEOUT_S = 600.0
_LOW_STATE_BOOTSTRAP_TIMEOUT_S = 15.0
_SCORING_TARGET_TIMEOUT_S = 30.0
_KNOWN_GOOD_SOURCE_FRAME_COUNT = 455
_KNOWN_GOOD_SOURCE_BODY_COUNT = 14
_KNOWN_GOOD_PROJECTED_FRAME_COUNT = 441
_KNOWN_GOOD_OMITTED_TAIL_COUNT = 14
_STREAM_CHUNK_FRAME_COUNT = 20
_STREAM_PADDING_START = 441
_STREAM_PADDING_STOP = 487
_STREAM_RECEIPT_FENCE_INDEX = 487
_POST_ENABLE_LEFT_LINE = "Delta heading left: 0.1 rad"
_POST_ENABLE_RIGHT_LINE = "Delta heading right: 0 rad"
_POST_ENABLE_FENCE_SEMANTICS = (
    "post-enable-reset-tail-complete-with-net-zero-heading"
)


def _is_remote_registered_joint_limit(error: BaseException) -> bool:
    if not isinstance(error, _RemoteMMError) or error.code != "generation_failed":
        return False
    # The authenticated server renders this message from its loaded contract;
    # retain only the fixed registered name and self-consistent numeric meaning.
    match = _REMOTE_JOINT_LIMIT_MESSAGE.fullmatch(error.message)
    if match is None or match.group("joint") not in SOURCE_JOINT_ORDER:
        return False
    try:
        position = float(match.group("position"))
        lower = float(match.group("lower"))
        upper = float(match.group("upper"))
    except ValueError:
        return False
    return (
        math.isfinite(position)
        and math.isfinite(lower)
        and math.isfinite(upper)
        and lower < upper
        and (position < lower or position > upper)
    )


def _is_remote_no_safe_candidate(error: BaseException) -> bool:
    return (
        isinstance(error, _RemoteMMError)
        and error.code == "generation_failed"
        and error.message == _REMOTE_NO_SAFE_CANDIDATE_MESSAGE
    )


def _is_remote_no_inertialized_safe_candidate(
    error: BaseException,
) -> bool:
    return (
        isinstance(error, _RemoteMMError)
        and error.code == "generation_failed"
        and error.message == _REMOTE_NO_INERTIALIZED_SAFE_CANDIDATE_MESSAGE
    )


def _joint_feasibility_from_hello(
    hello: object,
) -> JointFeasibilityIdentity:
    if not isinstance(hello, Mapping):
        raise ContractError("MM hello identity must be an object")
    return parse_joint_feasibility_identity(hello.get("joint_feasibility"))


def _joint_feasibility_payload(
    identity: JointFeasibilityIdentity,
) -> dict[str, object]:
    if not isinstance(identity, JointFeasibilityIdentity):
        raise ContractError("joint feasibility identity has the wrong type")
    return {
        "schema": identity.schema,
        "frame_count": identity.frame_count,
        "raw_safe_count": identity.raw_safe_count,
        "raw_unsafe_count": identity.raw_unsafe_count,
        "search_safe_count": identity.search_safe_count,
        "joint_limit_violation_count": list(
            identity.joint_limit_violation_count
        ),
        "mask_sha256": identity.mask_sha256,
    }


_PRELOAD_CONSUMER_TRANSCRIPT_PATH = (
    "dynamic/stream/preload-consumer-transcript.json"
)
_PRELOAD_CONSUMER_TRANSCRIPT_SCHEMA = (
    "mm-sonic-preload-consumer-transcript/v1"
)
_STREAM_PROCESSING_START_LINE = (
    "[ZMQEndpointInterface] *** Starting ZMQ processing ***"
)
_STREAM_PROCESSING_END_LINE = (
    "[ZMQEndpointInterface] *** End of ZMQ decoding processing ***"
)

_GEAR_COMMON_LOG_HEADER = (
    "index",
    "time_ms",
    "time_realtime_ms",
    "time_monotonic_ms",
    "ros_timestamp",
)
_GEAR_Q_HEADER = _GEAR_COMMON_LOG_HEADER + tuple(
    f"q_{index}" for index in range(29)
)
_GEAR_BASE_QUAT_HEADER = _GEAR_COMMON_LOG_HEADER + (
    "base_qw",
    "base_qx",
    "base_qy",
    "base_qz",
)
# Independent inverse copied from the authenticated pinned policy source.  Do
# not derive this from LOGGER_JOINT_PERMUTATION: the parser is an independent
# check on that target-logger boundary.
_GEAR_LOGGER_TO_TARGET = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3,
    9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
    24, 18, 25, 19, 26, 20, 27, 21, 28,
)

_KNOWN_GOOD_PROJECTION_RULE = MappingProxyType(
    {
        "schema": "mm-sonic-known-good-projection/v1",
        "source_frame_count": _KNOWN_GOOD_SOURCE_FRAME_COUNT,
        "source_body_count": _KNOWN_GOOD_SOURCE_BODY_COUNT,
        "selected_frame_range": [0, _KNOWN_GOOD_PROJECTED_FRAME_COUNT - 1],
        "selected_frame_count": _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
        "selected_body_indexes": [0],
        "omitted_tail_range": [
            _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
            _KNOWN_GOOD_SOURCE_FRAME_COUNT - 1,
        ],
        "omitted_tail_count": _KNOWN_GOOD_OMITTED_TAIL_COUNT,
        "root_body_position": "positive-zero-f32",
        "root_body_linear_velocity": "positive-zero-f32",
        "root_body_angular_velocity": "positive-zero-f32",
        "projected_csv_numeric_encoding": (
            "binary32-promoted-to-binary64-exact-decimal17"
        ),
        "control_clock": "wall-clock-50hz-asynchronous-to-simulator",
        "stream_readiness_frame_range": [0, 0],
        "stream_readiness_publication_count": 1,
        "stream_logical_frame_range": [1, 440],
        "stream_logical_frame_count": 440,
        "stream_logical_publication_count": 22,
        "stream_chunk_frame_count": _STREAM_CHUNK_FRAME_COUNT,
        "stream_chunk_publication_count": 22,
        "stream_padding_frame_range": [
            _STREAM_PADDING_START,
            _STREAM_PADDING_STOP - 1,
        ],
        "stream_padding_frame_count": (
            _STREAM_PADDING_STOP - _STREAM_PADDING_START
        ),
        "stream_padding_publication_count": 1,
        "stream_receipt_fence_frame_range": [
            _STREAM_RECEIPT_FENCE_INDEX,
            _STREAM_RECEIPT_FENCE_INDEX,
        ],
        "stream_receipt_fence_publication_count": 1,
        "stream_post_enable_fence_key_sequence": "qe",
        "stream_post_enable_fence_left_line": _POST_ENABLE_LEFT_LINE,
        "stream_post_enable_fence_right_line": _POST_ENABLE_RIGHT_LINE,
        "stream_post_enable_fence_semantics": _POST_ENABLE_FENCE_SEMANTICS,
        "stream_stdout_observation_offsets": (
            "raw-utf8-stdout-byte-offsets-after-archive-flush"
        ),
        "stream_consumer_transcript_path": _PRELOAD_CONSUMER_TRANSCRIPT_PATH,
        "stream_consumer_transcript_schema": (
            _PRELOAD_CONSUMER_TRANSCRIPT_SCHEMA
        ),
        "stream_consumer_transcript_binding": (
            "frozen-prefix-with-independent-exact-event-byte-ranges"
        ),
        "stream_consumer_event_ranges": [
            "start",
            "processing",
            "merged",
            "end",
        ],
        "stream_consumer_intervening_bytes": "allowed-and-sha256-bound",
        "stream_consumer_merged_to_end_diagnostics": (
            "arbitrary-and-sha256-bound"
        ),
        "stream_consumer_final_tail": (
            "final-end-authenticated-and-post-end-bytes-sha256-bound"
        ),
        "stream_consumer_publication_boundaries": (
            "after-prior-end-and-no-later-than-current-start"
        ),
        "consumer_processing_start_marker": (
            _STREAM_PROCESSING_START_LINE
        ),
        "consumer_causal_fence": (
            "publication-N+1-authenticated-start-to-end-event-fences-publication-N"
        ),
        "target_coverage_row_count": _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
        "gear_state_row_count": _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
        "state_target_pairing": "same-control-tick-positional",
    }
)


class _CLIConfigurationError(ValueError):
    pass


class CapabilityUnavailable(RuntimeError):
    """A required external runtime capability is absent."""


class ScientificGateFailure(RuntimeError):
    """A gate ran safely and produced a scientific failure."""


def _parse_local_zmq_endpoint(endpoint: str) -> tuple[str, int]:
    if type(endpoint) is not str:
        raise ContractError("resolved ZMQ endpoint must be a string")
    matched = _LOCAL_ZMQ_ENDPOINT.fullmatch(endpoint)
    if matched is None:
        raise ContractError(
            "resolved ZMQ endpoint must be an explicit 127.0.0.1 TCP port"
        )
    port = int(matched.group("port"))
    if port > 65535:
        raise ContractError("resolved ZMQ endpoint port is outside 1..65535")
    return matched.group("host"), port


def _stream_gear_command(
    base: Sequence[str], endpoint: str
) -> tuple[str, ...]:
    if not base or any(type(value) is not str or not value for value in base):
        raise ContractError("GEAR stream base command is invalid")
    host, port = _parse_local_zmq_endpoint(endpoint)
    return (*base, "--zmq-host", host, "--zmq-port", str(port))


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIConfigurationError(message)


@dataclass(frozen=True)
class StageARequest:
    command: str
    mode: str | None
    argv: tuple[str, ...]
    namespace: argparse.Namespace
    environment: Mapping[str, str]
    invocation_cwd: Path = field(
        default_factory=lambda: Path.cwd().resolve(strict=True)
    )


@dataclass(frozen=True)
class StageAContext:
    gear_commit: str
    gear_dirty: bool
    external_hashes: Mapping[str, str]
    motion_matching_commit: str
    motion_matching_dirty: bool
    artifact_hashes: Mapping[str, str | None]
    known_good_reference: Path
    processes: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class GateResult:
    status: str
    reason: str | None = None
    identity: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    evidence_hashes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    metrics: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    outputs: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    scene_registration: Mapping[str, object] | None = None
    artifact_hashes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    processes: tuple[Mapping[str, object], ...] = ()


class StageAOperations(Protocol):
    """Slow/external Stage A seam; parsing and evidence stay in this module."""

    def authenticate(
        self, request: StageARequest, inputs: ExternalInputs
    ) -> StageAContext: ...

    def run_gate(
        self,
        name: str,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult: ...


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExternalInputError(f"input is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
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
            raise ExternalInputError(f"input changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


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
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be finite JSON data") from error


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _safe_environment(source: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            name: source[name]
            for name in _SAFE_ENVIRONMENT_ALLOWLIST
            if name in source
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="python -m mm_sonic.cli")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--gear-checkout", required=True)
        command.add_argument("--policy", required=True)
        command.add_argument("--observation-config", required=True)
        command.add_argument("--encoder")
        command.add_argument("--source-mjcf", required=True)
        command.add_argument("--terrain-dir", required=True)
        command.add_argument("--output-root", required=True)

    preflight = subcommands.add_parser("preflight")
    common(preflight)
    stage_a = subcommands.add_parser("stage-a")
    stage_a.add_argument(
        "--mode",
        required=True,
        choices=("mm-reference", "known-good-file", "known-good-stream"),
    )
    common(stage_a)
    stage_b = subcommands.add_parser("stage-b")
    stage_b.add_argument("--stage-a-evidence", required=True)
    common(stage_b)
    return parser


def _run_id(command: str, mode: str | None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = command if mode is None else mode
    return f"{label}-{now}-{uuid.uuid4().hex[:8]}"


def _target_gate_count(request: StageARequest) -> int:
    if request.command == "preflight":
        return 3
    return {
        "mm-reference": 4,
        "known-good-file": 5,
        "known-good-stream": 7,
    }[request.mode]


def _gate_record(name: str, result: GateResult) -> dict[str, object]:
    return {
        "name": name,
        "status": result.status,
        "reason": result.reason,
        "identity": dict(result.identity),
        "evidence_hashes": dict(result.evidence_hashes),
        "metrics": dict(result.metrics),
        "outputs": dict(result.outputs),
    }


def _pending_gate(name: str, status: str = "pending", reason: str | None = None):
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "identity": {},
        "evidence_hashes": {},
        "metrics": {},
        "outputs": {},
    }


def _validate_gate_result(
    result: GateResult, bundle: RunBundle
) -> GateResult:
    if not isinstance(result, GateResult):
        raise ContractError("Stage A operation did not return a GateResult")
    if result.status not in _GATE_STATUS:
        raise ContractError("Stage A gate status is invalid")
    if result.status == "pass" and result.reason is not None:
        raise ContractError("passing Stage A gate cannot have a reason")
    if result.status != "pass" and (
        type(result.reason) is not str or not result.reason
    ):
        raise ContractError("non-passing Stage A gate requires a reason")
    if result.status == "not_run" and result.metrics:
        raise ContractError("not_run Stage A gate cannot contain results")
    for name, value in result.identity.items():
        if name not in STAGE_A_IDENTITY_KEYS:
            raise ContractError(f"unknown Stage A identity key: {name}")
        if type(value) is not str or not value:
            raise ContractError("Stage A identity values must be nonempty strings")
    for name, digest in result.evidence_hashes.items():
        if type(name) is not str or not name or not _is_sha256(digest):
            raise ContractError("Stage A evidence hashes are invalid")
    for name, digest in result.artifact_hashes.items():
        if name not in _ARTIFACT_HASH_KEYS or not _is_sha256(digest):
            raise ContractError("Stage A artifact hash update is invalid")
    if type(result.processes) is not tuple:
        raise ContractError("Stage A process records must be a tuple")
    for process in result.processes:
        if not isinstance(process, Mapping):
            raise ContractError("Stage A process record must be a mapping")
        name = process.get("name")
        argv = process.get("argv")
        if (
            type(name) is not str
            or not name
            or type(argv) not in (list, tuple)
            or not argv
            or any(type(argument) is not str or not argument for argument in argv)
        ):
            raise ContractError("Stage A process record is invalid")
        executable_digest = process.get("executable_sha256")
        if executable_digest is not None and not _is_sha256(executable_digest):
            raise ContractError("Stage A process executable hash is invalid")
        module_digest = process.get("module_sha256")
        if module_digest is not None and not _is_sha256(module_digest):
            raise ContractError("Stage A process module hash is invalid")
    for relative, expected in result.outputs.items():
        if type(relative) is not str or not _is_sha256(expected):
            raise ContractError("Stage A output declaration is invalid")
        parts = PurePosixPath(relative).parts
        if (
            not parts
            or PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise ContractError("Stage A output is not confined")
        path = bundle.path.joinpath(*parts)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ContractError(f"Stage A output is missing: {relative}") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ContractError(f"Stage A output is unsafe: {relative}")
        if _sha256_file(path) != expected:
            raise ContractError(f"Stage A output hash changed: {relative}")
    return result


def _merge_identity(
    current: dict[str, str], additions: Mapping[str, str]
) -> None:
    for name, value in additions.items():
        if name in current and current[name] != value:
            raise ContractError(
                f"Stage A identity mismatch for {name}: file and stream differ"
            )
    current.update(additions)


def _command_script_manifest(request: StageARequest) -> Mapping[str, str]:
    if request.command == "preflight":
        identity = b"mm-sonic-stage-a/preflight/no-trial\n"
        return MappingProxyType(
            {
                "id": "stage-a-preflight-no-trial",
                "sha256": _sha256_bytes(identity),
            }
        )
    script = command_script_bytes(
        scene_id=FLAT_SCENE_ID,
        route_id=_FLAT_ROUTE_ID,
        commands=flat_command_script(),
    )
    return MappingProxyType(
        {"id": _FLAT_ROUTE_ID, "sha256": _sha256_bytes(script)}
    )


def _context_manifest(
    request: StageARequest,
    inputs: ExternalInputs,
    context: StageAContext,
    scene_registration: Mapping[str, object] | None,
    joint_feasibility: Mapping[str, object] | None,
    artifact_hashes: Mapping[str, str | None],
    processes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    safe_values = dict(request.environment)
    cli_process = {
        "name": "mm_sonic.cli",
        "argv": ["mm_sonic.cli", *request.argv],
        "invocation_cwd": str(request.invocation_cwd),
        "environment": {
            "allowlist": list(_SAFE_ENVIRONMENT_ALLOWLIST),
            "values": safe_values,
        },
    }
    external = {
        "gear_checkout": str(inputs.gear_checkout),
        "gear_commit": context.gear_commit,
        "gear_dirty": context.gear_dirty,
        "policy": str(inputs.policy),
        "observation_config": str(inputs.observation_config),
        "encoder": None if inputs.encoder is None else str(inputs.encoder),
        "terrain_dir": str(inputs.terrain_dir),
        "source_mjcf": str(inputs.source_mjcf),
        "hashes": dict(context.external_hashes),
    }
    manifest: dict[str, object] = {
        "external": external,
        "repositories": {
            "motion_matching": {
                "commit": context.motion_matching_commit,
                "dirty": context.motion_matching_dirty,
            },
            "gear_sonic": {
                "commit": context.gear_commit,
                "dirty": context.gear_dirty,
            },
        },
        "artifact_hashes": dict(artifact_hashes),
        "command_script": dict(_command_script_manifest(request)),
        "perturbation": {
            "id": "nominal",
            "lateral_offset_m": 0.0,
            "yaw_offset_rad": 0.0,
        },
        "coordinate_transform": {
            "source": HOLDEN_COORDINATE_SIGNATURE,
            "target": MUJOCO_COORDINATE_SIGNATURE,
            "matrix": HOLDEN_TO_MUJOCO_MATRIX.tolist(),
        },
        "processes": [cli_process, *(dict(value) for value in processes)],
    }
    if scene_registration is not None:
        manifest["scene_registration"] = dict(scene_registration)
    if joint_feasibility is not None:
        manifest["joint_feasibility"] = dict(joint_feasibility)
    return manifest


def _command_status_exit(status: str) -> int:
    return {
        "pass": EXIT_PASS,
        "integration_failure": EXIT_CONFIGURATION,
        "scientific_failure": EXIT_SCIENTIFIC,
        "not_run": EXIT_NOT_RUN,
    }[status]


def _execute(
    request: StageARequest,
    inputs: ExternalInputs,
    operations: StageAOperations,
    stdout: IO[str],
    bundle: RunBundle,
) -> int:
    context = operations.authenticate(request, inputs)
    if not isinstance(context, StageAContext):
        raise ContractError("Stage A authentication returned an invalid context")
    registry_bytes = _STAGE_A_REGISTRY_PATH.read_bytes()
    registry_sha256 = _sha256_bytes(registry_bytes)
    target_count = _target_gate_count(request)
    records = [_pending_gate(name) for name in STAGE_A_GATE_NAMES]
    identity: dict[str, str] = {}
    accumulated_metrics: dict[str, object] = {}
    all_outputs: dict[str, str] = {}
    artifact_hashes = dict(context.artifact_hashes)
    processes = [dict(value) for value in context.processes]
    scene_registration: Mapping[str, object] | None = None
    joint_feasibility: Mapping[str, object] | None = None
    command_status = "pass"
    failure_reason: str | None = None

    for index, name in enumerate(STAGE_A_GATE_NAMES[:target_count]):
        try:
            raw_result = operations.run_gate(
                name, request, context, bundle
            )
        except CapabilityUnavailable as error:
            raw_result = GateResult(status="not_run", reason=str(error))
        except ScientificGateFailure as error:
            raw_result = GateResult(
                status="scientific_failure", reason=str(error)
            )
        except BaseException as error:
            raw_result = GateResult(
                status="integration_failure",
                reason=f"{type(error).__name__}: {error}",
            )
        try:
            result = _validate_gate_result(raw_result, bundle)
            if result.status == "pass":
                _merge_identity(identity, result.identity)
                for artifact_name, artifact_digest in result.artifact_hashes.items():
                    current = artifact_hashes.get(artifact_name)
                    if current is not None and current != artifact_digest:
                        raise ContractError(
                            "Stage A artifact identity changed for "
                            f"{artifact_name}"
                        )
                    artifact_hashes[artifact_name] = artifact_digest
                if name == "basis_and_scene_alignment":
                    try:
                        feasibility_value = result.metrics[
                            "joint_feasibility"
                        ]
                    except KeyError as error:
                        raise ContractError(
                            "basis gate lacks joint feasibility evidence"
                        ) from error
                    joint_feasibility = _joint_feasibility_payload(
                        parse_joint_feasibility_identity(feasibility_value)
                    )
            processes.extend(dict(value) for value in result.processes)
        except ContractError as error:
            result = GateResult(
                status="integration_failure", reason=str(error)
            )
        records[index] = _gate_record(name, result)
        all_outputs.update(result.outputs)
        if result.status == "pass":
            if result.metrics:
                accumulated_metrics[name] = dict(result.metrics)
            if result.scene_registration is not None:
                if (
                    scene_registration is not None
                    and dict(scene_registration)
                    != dict(result.scene_registration)
                ):
                    result = GateResult(
                        status="integration_failure",
                        reason="Stage A scene registration identity changed",
                    )
                    records[index] = _gate_record(name, result)
                else:
                    scene_registration = result.scene_registration
        if result.status != "pass":
            command_status = result.status
            failure_reason = result.reason
            blocked_reason = f"blocked by {name}: {result.reason}"
            for blocked in range(index + 1, target_count):
                records[blocked] = _pending_gate(
                    STAGE_A_GATE_NAMES[blocked], "not_run", blocked_reason
                )
            break

    if command_status == "pass" and target_count == len(STAGE_A_GATE_NAMES):
        if set(identity) != STAGE_A_IDENTITY_KEYS:
            command_status = "integration_failure"
            failure_reason = "complete Stage A evidence has an incomplete identity"
        else:
            stage_a_status = "pass"
    elif command_status == "pass":
        stage_a_status = "incomplete"
    elif command_status == "not_run":
        stage_a_status = "not_run"
    else:
        stage_a_status = "failed"
    if command_status != "pass":
        stage_a_status = "not_run" if command_status == "not_run" else "failed"

    identity_digest = (
        stage_a_identity_sha256(identity)
        if set(identity) == STAGE_A_IDENTITY_KEYS
        else None
    )
    if command_status == "not_run":
        accumulated_metrics = {}
        for record in records:
            record["metrics"] = {}
    evidence = {
        "schema": "mm-sonic-stage-a-evidence/v1",
        "registry_sha256": registry_sha256,
        "command": request.command,
        "mode": request.mode,
        "argv": ["mm_sonic.cli", *request.argv],
        "invocation_cwd": str(request.invocation_cwd),
        "environment": {
            "allowlist": list(_SAFE_ENVIRONMENT_ALLOWLIST),
            "values": dict(request.environment),
        },
        "command_status": command_status,
        "stage_a_status": stage_a_status,
        "gates": records,
        "identity": identity,
        "identity_sha256": identity_digest,
        "metrics": accumulated_metrics,
        "outputs": all_outputs,
    }
    bundle.write_bytes(
        "stage-a-evidence.json",
        _canonical_json_bytes(evidence, "Stage A evidence"),
    )
    bundle.update_manifest(
        _context_manifest(
            request,
            inputs,
            context,
            scene_registration,
            joint_feasibility,
            artifact_hashes,
            processes,
        )
    )
    terminal_status = {
        "pass": "complete",
        "integration_failure": "failed",
        "scientific_failure": "failed",
        "not_run": "not_run",
    }[command_status]
    outcome: dict[str, object] = {
        "status": command_status,
        "stage_a_status": stage_a_status,
    }
    if failure_reason is not None:
        outcome["reason"] = failure_reason
    bundle.finalize(terminal_status, outcome=outcome)
    stdout.write(
        json.dumps(
            {
                "status": command_status,
                "stage_a_status": stage_a_status,
                "evidence": str(bundle.path / "stage-a-evidence.json"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return _command_status_exit(command_status)


_STAGE_B_GATE_NAMES = (
    "external_identity",
    "joint_projection_round_trip",
    "basis_and_scene_alignment",
    "flat_mm_kinematic_replay",
    "stage_b_dynamic",
)
_STAGE_B_CURRENT_IDENTITY_KEYS = STAGE_A_IDENTITY_KEYS - {
    "known_good_reference_buffer"
}


def _stage_b_failure_layer(name: str, status: str) -> str:
    if name == "flat_mm_kinematic_replay" and status == "scientific_failure":
        return "reference"
    if name == "stage_b_dynamic":
        return "delivery"
    if name == "external_identity":
        return "external_identity"
    return "conversion"


def _stage_b_known_good_metrics(
    prerequisite: Mapping[str, object],
) -> Mapping[str, object]:
    metrics = prerequisite.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ContractError("Stage A prerequisite metrics are unavailable")
    for name in (
        "known_good_stream_dynamic",
        "known_good_file_dynamic",
    ):
        candidate = metrics.get(name)
        if isinstance(candidate, Mapping):
            joint = candidate.get("joint_position_rmse_rad")
            pelvis = candidate.get("pelvis_orientation_rms_rad")
            if type(joint) in (int, float) and type(pelvis) in (int, float):
                return candidate
    raise ContractError("Stage A known-good tracking metrics are unavailable")


def _stage_b_json_object(raw: bytes, label: str) -> Mapping[str, object]:
    def no_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for name, value in pairs:
            if name in output:
                raise ContractError(f"{label} contains a duplicate key")
            output[name] = value
        return output

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} is not an object")
    return value


def _stage_b_prerequisite_scene_registration(
    evidence_path: Path,
) -> Mapping[str, object]:
    snapshot = _stable_regular_file_snapshot(
        evidence_path.parent / "manifest.json",
        "Stage A prerequisite manifest",
    )
    if snapshot is None:
        raise ContractError("Stage A prerequisite manifest is unavailable")
    manifest = _stage_b_json_object(
        snapshot[2], "Stage A prerequisite manifest"
    )
    registration = manifest.get("scene_registration")
    if not isinstance(registration, Mapping):
        raise ContractError(
            "Stage A prerequisite scene registration is unavailable"
        )
    return registration


def _stage_b_scene_semantics(
    registration: Mapping[str, object],
    *,
    raw_scene_sha256: object,
    bundle_root: Path,
) -> Mapping[str, object]:
    """Bind generated XML after normalizing only its run-local include path."""

    if not isinstance(registration, Mapping):
        raise ContractError("Stage B scene registration is unavailable")
    copied = dict(registration)
    terrain_value = copied.pop("terrain_geoms", None)
    terrain_geoms: tuple[int, ...] | None = None
    if terrain_value is not None:
        if (
            type(terrain_value) is not list
            or not terrain_value
            or any(type(value) is not int or value < 0 for value in terrain_value)
            or len(set(terrain_value)) != len(terrain_value)
        ):
            raise ContractError("Stage B scene terrain geom IDs are invalid")
        terrain_geoms = tuple(terrain_value)
    output_hashes = copied.get("output_hashes")
    if not isinstance(output_hashes, Mapping):
        raise ContractError("Stage B scene output hashes are unavailable")
    outputs = dict(output_hashes)
    gear_scene = outputs.pop("gear_scene_xml", None)
    registration_hash = outputs.pop("scene_registration", None)
    robot_include = outputs.get("robot_include")
    if (
        not _is_sha256(raw_scene_sha256)
        or gear_scene != raw_scene_sha256
        or not _is_sha256(registration_hash)
        or not _is_sha256(robot_include)
        or any(not _is_sha256(value) for value in outputs.values())
    ):
        raise ContractError("Stage B scene artifact hashes are invalid")
    copied["output_hashes"] = dict(sorted(outputs.items()))

    root = Path(bundle_root)
    scene_path = root / "scene/gear_scene.xml"
    robot_path = root / "scene/gear_robot.xml"
    scene_snapshot = _stable_regular_file_snapshot(
        scene_path, "Stage B generated scene XML"
    )
    robot_snapshot = _stable_regular_file_snapshot(
        robot_path, "Stage B generated robot include"
    )
    assert scene_snapshot is not None and robot_snapshot is not None
    if (
        _sha256_bytes(scene_snapshot[2]) != gear_scene
        or _sha256_bytes(robot_snapshot[2]) != robot_include
    ):
        raise ContractError("Stage B generated scene artifacts changed")
    try:
        xml_root = ET.fromstring(scene_snapshot[2])
    except ET.ParseError as error:
        raise ContractError("Stage B generated scene XML is invalid") from error
    includes = tuple(xml_root.iter("include"))
    if (
        xml_root.tag != "mujoco"
        or len(includes) != 1
        or set(includes[0].attrib) != {"file"}
    ):
        raise ContractError("Stage B generated scene include contract changed")
    include_value = includes[0].attrib["file"]
    if not include_value or include_value.startswith("~"):
        raise ContractError("Stage B generated scene include path is invalid")
    try:
        resolved_include = Path(include_value).resolve(strict=True)
        expected_include = robot_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractError(
            "Stage B generated scene include path is unavailable"
        ) from error
    if resolved_include != expected_include:
        raise ContractError(
            "Stage B generated scene include leaves the authenticated bundle"
        )
    includes[0].set("file", "__RUN_ROOT__/scene/gear_robot.xml")
    canonical_xml = ET.tostring(
        xml_root,
        encoding="utf-8",
        xml_declaration=False,
        short_empty_elements=True,
    )
    return MappingProxyType(
        {
            "registration": copied,
            "canonical_scene_xml_sha256": _sha256_bytes(canonical_xml),
            "terrain_geoms": terrain_geoms,
        }
    )


def _stage_b_verdict(
    metrics: Mapping[str, object],
    prerequisite: Mapping[str, object],
):
    coverage = metrics.get("coverage")
    tracking = metrics.get("tracking")
    safety = metrics.get("safety")
    secondary_value = metrics.get("secondary")
    if not all(
        isinstance(value, Mapping)
        for value in (coverage, tracking, safety, secondary_value)
    ):
        raise ContractError("Stage B dynamic metrics are incomplete")
    assert isinstance(coverage, Mapping)
    assert isinstance(tracking, Mapping)
    assert isinstance(safety, Mapping)
    assert isinstance(secondary_value, Mapping)
    secondary = SecondaryMetrics(
        swing_foot_scuff_count=secondary_value.get("swing_foot_scuff_count"),
        minimum_foot_clearance_m=secondary_value.get(
            "minimum_foot_clearance_m"
        ),
        horizontal_path_drift_m=secondary_value.get(
            "horizontal_path_drift_m"
        ),
        joint_tracking_trace_rad=tuple(
            tracking.get("joint_tracking_trace_rad", ())
        ),
        pelvis_tracking_trace_rad=tuple(
            tracking.get("pelvis_tracking_trace_rad", ())
        ),
        contact_impulses_ns=tuple(
            secondary_value.get("contact_impulses_ns", ())
        ),
        policy_execution_timing_ns=secondary_value.get(
            "policy_execution_timing_ns"
        ),
    )
    known_good = _stage_b_known_good_metrics(prerequisite)
    return evaluate_flat_trial(
        integration_pass=True,
        exact_command_coverage=coverage.get("exact_command_coverage"),
        exact_frame_coverage=coverage.get("exact_frame_coverage"),
        exact_control_duration=coverage.get("exact_control_duration"),
        exact_safety_log_coverage=safety.get("exact_log_coverage"),
        minimum_pelvis_local_height_m=safety.get(
            "minimum_pelvis_local_height_m"
        ),
        minimum_pelvis_up_dot=safety.get("minimum_pelvis_up_dot"),
        contact_groups=tuple(safety.get("forbidden_contact_groups", ())),
        joint_position_rmse_rad=tracking.get("joint_position_rmse_rad"),
        known_good_joint_position_rmse_rad=known_good.get(
            "joint_position_rmse_rad"
        ),
        pelvis_orientation_rms_rad=tracking.get(
            "pelvis_orientation_rms_rad"
        ),
        known_good_pelvis_orientation_rms_rad=known_good.get(
            "pelvis_orientation_rms_rad"
        ),
        secondary=secondary,
    )


def _execute_stage_b(
    request: StageARequest,
    inputs: ExternalInputs,
    operations: StageAOperations,
    stdout: IO[str],
    bundle: RunBundle,
) -> int:
    """Authenticate Stage A, replay the full reference, then run flat SONIC."""

    prerequisite_path = Path(request.namespace.stage_a_evidence)
    prerequisite = validate_stage_a_prerequisite(
        prerequisite_path,
        expected_identity=None,
    )
    prerequisite_path = prerequisite_path.resolve(strict=True)
    context = operations.authenticate(request, inputs)
    if not isinstance(context, StageAContext):
        raise ContractError("Stage B authentication returned an invalid context")

    records = [_pending_gate(name) for name in _STAGE_B_GATE_NAMES]
    identity: dict[str, str] = {}
    accumulated_metrics: dict[str, object] = {}
    all_outputs: dict[str, str] = {}
    artifact_hashes = dict(context.artifact_hashes)
    processes = [dict(value) for value in context.processes]
    scene_registration: Mapping[str, object] | None = None
    joint_feasibility: Mapping[str, object] | None = None
    command_status = "pass"
    failure_reason: str | None = None
    failure_layer: str | None = None
    integration_pass = False
    kinematic_pass = False
    dynamic_pass = False
    scene_semantic_sha256: str | None = None

    for index, name in enumerate(_STAGE_B_GATE_NAMES[:4]):
        try:
            raw_result = operations.run_gate(name, request, context, bundle)
        except CapabilityUnavailable as error:
            raw_result = GateResult(status="not_run", reason=str(error))
        except ScientificGateFailure as error:
            raw_result = GateResult(
                status="scientific_failure", reason=str(error)
            )
        except BaseException as error:
            raw_result = GateResult(
                status="integration_failure",
                reason=f"{type(error).__name__}: {error}",
            )
        try:
            result = _validate_gate_result(raw_result, bundle)
            if result.status == "pass":
                _merge_identity(identity, result.identity)
                for artifact_name, artifact_digest in result.artifact_hashes.items():
                    current = artifact_hashes.get(artifact_name)
                    if current is not None and current != artifact_digest:
                        raise ContractError(
                            "Stage B artifact identity changed for "
                            f"{artifact_name}"
                        )
                    artifact_hashes[artifact_name] = artifact_digest
                if name == "basis_and_scene_alignment":
                    feasibility_value = result.metrics.get("joint_feasibility")
                    if feasibility_value is None:
                        raise ContractError(
                            "basis gate lacks joint feasibility evidence"
                        )
                    joint_feasibility = _joint_feasibility_payload(
                        parse_joint_feasibility_identity(feasibility_value)
                    )
            processes.extend(dict(value) for value in result.processes)
        except ContractError as error:
            result = GateResult(
                status="integration_failure", reason=str(error)
            )
        records[index] = _gate_record(name, result)
        all_outputs.update(result.outputs)
        if result.status == "pass":
            if result.metrics:
                accumulated_metrics[name] = dict(result.metrics)
            if result.scene_registration is not None:
                if (
                    scene_registration is not None
                    and dict(scene_registration)
                    != dict(result.scene_registration)
                ):
                    result = GateResult(
                        status="integration_failure",
                        reason="Stage B scene registration identity changed",
                    )
                    records[index] = _gate_record(name, result)
                else:
                    scene_registration = result.scene_registration
        if result.status != "pass":
            command_status = result.status
            failure_reason = result.reason
            failure_layer = _stage_b_failure_layer(name, result.status)
            break

    if command_status == "pass":
        kinematic_pass = True
        prerequisite_identity = prerequisite.get("identity")
        identity_mismatch = (
            not isinstance(prerequisite_identity, Mapping)
            or set(identity) != _STAGE_B_CURRENT_IDENTITY_KEYS
            or any(
                name != "generated_flat_scene"
                and prerequisite_identity.get(name) != value
                for name, value in identity.items()
            )
        )
        try:
            if not isinstance(prerequisite_identity, Mapping):
                raise ContractError("Stage A prerequisite identity is unavailable")
            current_scene_semantics = _stage_b_scene_semantics(
                scene_registration,
                raw_scene_sha256=identity.get("generated_flat_scene"),
                bundle_root=bundle.path,
            )
            prerequisite_scene_semantics = _stage_b_scene_semantics(
                _stage_b_prerequisite_scene_registration(prerequisite_path),
                raw_scene_sha256=prerequisite_identity.get(
                    "generated_flat_scene"
                ),
                bundle_root=prerequisite_path.parent,
            )
            current_terrain = current_scene_semantics.get("terrain_geoms")
            prerequisite_terrain = prerequisite_scene_semantics.get(
                "terrain_geoms"
            )
            if (
                type(current_terrain) is not tuple
                or not current_terrain
                or current_scene_semantics.get("registration")
                != prerequisite_scene_semantics.get("registration")
                or current_scene_semantics.get("canonical_scene_xml_sha256")
                != prerequisite_scene_semantics.get(
                    "canonical_scene_xml_sha256"
                )
                or (
                    prerequisite_terrain is not None
                    and prerequisite_terrain != current_terrain
                )
            ):
                identity_mismatch = True
            else:
                scene_semantic_sha256 = _sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "registration": current_scene_semantics[
                                "registration"
                            ],
                            "canonical_scene_xml_sha256": (
                                current_scene_semantics[
                                    "canonical_scene_xml_sha256"
                                ]
                            ),
                            "terrain_geoms": list(current_terrain),
                        },
                        "Stage B cross-bound scene identity",
                    )
                )
        except ContractError:
            identity_mismatch = True
        if identity_mismatch:
            command_status = "integration_failure"
            failure_reason = "Stage B identity does not match Stage A prerequisite"
            failure_layer = "prerequisite"

    if command_status == "pass":
        index = 4
        name = _STAGE_B_GATE_NAMES[index]
        try:
            raw_result = operations.run_gate(name, request, context, bundle)
        except CapabilityUnavailable as error:
            raw_result = GateResult(status="not_run", reason=str(error))
        except ScientificGateFailure as error:
            raw_result = GateResult(
                status="scientific_failure", reason=str(error)
            )
        except BaseException as error:
            raw_result = GateResult(
                status="integration_failure",
                reason=f"{type(error).__name__}: {error}",
            )
        try:
            result = _validate_gate_result(raw_result, bundle)
            processes.extend(dict(value) for value in result.processes)
        except ContractError as error:
            result = GateResult(
                status="integration_failure", reason=str(error)
            )
        records[index] = _gate_record(name, result)
        all_outputs.update(result.outputs)
        if result.status == "pass":
            integration_pass = True
            accumulated_metrics[name] = dict(result.metrics)
            try:
                verdict = _stage_b_verdict(result.metrics, prerequisite)
            except ContractError as error:
                command_status = "integration_failure"
                integration_pass = False
                failure_reason = str(error)
                failure_layer = "delivery"
            else:
                dynamic_pass = verdict.dynamic_pass
                accumulated_metrics["checks"] = dict(verdict.checks)
                if not dynamic_pass:
                    command_status = "scientific_failure"
                    failure_reason = "Stage B dynamic scientific gates failed"
                    checks = verdict.checks
                    if not (
                        checks["exact_command_coverage"]
                        and checks["exact_frame_coverage"]
                        and checks["exact_control_duration"]
                    ):
                        failure_layer = "delivery"
                    elif not (
                        checks["pelvis_local_height"]
                        and checks["pelvis_up_dot"]
                        and checks["forbidden_contacts"]
                        and checks["exact_safety_log_coverage"]
                    ):
                        failure_layer = "safety"
                    else:
                        failure_layer = "sonic_tracking"
        else:
            command_status = result.status
            failure_reason = result.reason
            failure_layer = _stage_b_failure_layer(name, result.status)

    if command_status != "pass":
        first_pending = next(
            (
                index
                for index, record in enumerate(records)
                if record["status"] == "pending"
            ),
            len(records),
        )
        for index in range(first_pending, len(records)):
            records[index] = _pending_gate(
                _STAGE_B_GATE_NAMES[index],
                "not_run",
                f"blocked by {failure_layer}: {failure_reason}",
            )

    registry_sha256 = _sha256_file(_STAGE_B_REGISTRY_PATH)
    prerequisite_sha256 = _sha256_file(prerequisite_path)
    dynamic_metrics = accumulated_metrics.get("stage_b_dynamic", {})
    timings = (
        dict(dynamic_metrics.get("timings", {}))
        if isinstance(dynamic_metrics, Mapping)
        and isinstance(dynamic_metrics.get("timings"), Mapping)
        else {}
    )
    evidence_hashes: dict[str, str] = {
        "registry_sha256": registry_sha256,
        "stage_a_evidence_sha256": prerequisite_sha256,
    }
    if scene_semantic_sha256 is not None:
        evidence_hashes["scene_semantic_sha256"] = scene_semantic_sha256
    for record in records:
        hashes = record.get("evidence_hashes")
        if isinstance(hashes, Mapping):
            evidence_hashes.update(
                (str(name), str(value)) for name, value in hashes.items()
            )
    evidence = {
        "schema": "mm-sonic-trial-verdict/v1",
        "stage": "B",
        "scene_id": FLAT_SCENE_ID,
        "terrain_weight": 0.0,
        "expected_frames": 601,
        "expected_sim_time_s": 12.0,
        "integration_pass": integration_pass,
        "kinematic_pass": kinematic_pass,
        "dynamic_pass": dynamic_pass,
        "failure_layer": failure_layer,
        "command_status": command_status,
        "reason": failure_reason,
        "gates": records,
        "identity": identity,
        "metrics": accumulated_metrics,
        "timings": timings,
        "evidence_hashes": evidence_hashes,
        "outputs": all_outputs,
    }
    bundle.write_bytes(
        "stage-b-evidence.json",
        _canonical_json_bytes(evidence, "Stage B evidence"),
    )
    bundle.update_manifest(
        _context_manifest(
            request,
            inputs,
            context,
            scene_registration,
            joint_feasibility,
            artifact_hashes,
            processes,
        )
    )
    stage_b_status = (
        "pass"
        if command_status == "pass"
        else "not_run" if command_status == "not_run" else "failed"
    )
    outcome: dict[str, object] = {
        "status": command_status,
        "stage_b_status": stage_b_status,
    }
    if failure_reason is not None:
        outcome["reason"] = failure_reason
    bundle.finalize(
        "complete" if command_status == "pass" else stage_b_status,
        outcome=outcome,
    )
    stdout.write(
        json.dumps(
            {
                "status": command_status,
                "stage_b_status": stage_b_status,
                "evidence": str(bundle.path / "stage-b-evidence.json"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return _command_status_exit(command_status)


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        env=dict(_safe_environment(os.environ)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExternalInputError(
            f"cannot authenticate motion-matching repository: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def is_git_lfs_pointer(path: str | os.PathLike[str]) -> bool:
    """Return whether one regular input is an unresolved Git-LFS pointer."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError:
        return False
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > 1024
    ):
        return False
    try:
        contents = candidate.read_bytes()
    except OSError:
        return False
    lines = contents.splitlines()
    return (
        len(lines) == 3
        and lines[0] == b"version https://git-lfs.github.com/spec/v1"
        and lines[1].startswith(b"oid sha256:")
        and lines[2].startswith(b"size ")
    )


def _walk_capability_files(
    root: Path,
    *,
    label: str,
) -> tuple[Path, ...]:
    """Enumerate every regular capability file without following symlinks."""

    discovered: list[Path] = []

    def failed(error: OSError) -> None:
        raise CapabilityUnavailable(f"cannot scan {label}: {error}")

    for directory, directories, files in os.walk(
        root,
        topdown=True,
        onerror=failed,
        followlinks=False,
    ):
        directories[:] = sorted(directories)
        base = Path(directory)
        for name in sorted(files):
            candidate = base / name
            try:
                metadata = candidate.lstat()
            except OSError as error:
                raise CapabilityUnavailable(
                    f"cannot inspect {label} file: {candidate}"
                ) from error
            if stat.S_ISREG(metadata.st_mode):
                discovered.append(candidate)
    return tuple(discovered)


def _find_unresolved_git_lfs_capabilities(
    inputs: ExternalInputs,
) -> tuple[Path, ...]:
    """Scan only external payloads required by the Stage A runtime."""

    candidates = [
        inputs.policy,
        inputs.observation_config,
        inputs.source_mjcf,
    ]
    if inputs.encoder is not None:
        candidates.append(inputs.encoder)
    candidates.extend(
        locked_gear_capability_paths(inputs.gear_checkout, _LOCK_PATH)
    )
    candidates.extend(
        (
            inputs.gear_checkout / _GEAR_MODEL_RELATIVE,
            inputs.gear_checkout / _GEAR_ROBOT_RELATIVE,
            inputs.gear_checkout / _GEAR_BINARY_RELATIVE,
        )
    )

    scan_errors: list[CapabilityUnavailable] = []
    for root, label in (
        (
            inputs.gear_checkout / _GEAR_MESHES_RELATIVE,
            "official GEAR mesh assets",
        ),
        (inputs.terrain_dir, "terrain input"),
    ):
        try:
            candidates.extend(_walk_capability_files(root, label=label))
        except CapabilityUnavailable as error:
            scan_errors.append(error)
    pointers = tuple(
        path for path in dict.fromkeys(candidates) if is_git_lfs_pointer(path)
    )
    if pointers:
        return pointers
    if scan_errors:
        raise scan_errors[0]
    return ()


def _child_environment(
    request: StageARequest,
    **explicit: str,
) -> dict[str, str]:
    environment = dict(request.environment)
    environment.update(explicit)
    return environment


def _environment_record(environment: Mapping[str, str]) -> dict[str, object]:
    return {
        "allowlist": list(environment),
        "values": dict(environment),
    }


def _process_record(
    name: str,
    argv: Sequence[str],
    *,
    executable: Path,
    environment: Mapping[str, str],
    outputs: Sequence[str] = (),
    module: Path | None = None,
) -> Mapping[str, object]:
    record: dict[str, object] = {
        "name": name,
        "argv": list(argv),
        "executable_sha256": _sha256_file(executable),
        "environment": _environment_record(environment),
        "outputs": list(outputs),
    }
    if module is not None:
        record["module_sha256"] = _sha256_file(module)
    return MappingProxyType(record)


def _snapshot_outputs(bundle: RunBundle) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(bundle.path.rglob("*")):
        relative = path.relative_to(bundle.path).as_posix()
        if relative == "manifest.json":
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"run output became a symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ContractError(f"run output became hard linked: {relative}")
            output[relative] = _sha256_file(path)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ContractError(f"run output has an unsafe type: {relative}")
    return output


def _new_outputs(
    bundle: RunBundle,
    before: Mapping[str, str],
) -> Mapping[str, str]:
    after = _snapshot_outputs(bundle)
    changed = {
        name: digest
        for name, digest in after.items()
        if before.get(name) != digest
    }
    return MappingProxyType(changed)


def _canonical_target_sha256(buffer: CanonicalTargetBuffer) -> str:
    digest = hashlib.sha256()
    for value, dtype in (
        (buffer.joint_position, "<f4"),
        (buffer.joint_velocity, "<f4"),
        (buffer.body_quat_w, "<f4"),
        (buffer.frame_index, "<i8"),
    ):
        digest.update(np.asarray(value, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _buffers_bit_equal(
    left: CanonicalTargetBuffer,
    right: CanonicalTargetBuffer,
) -> bool:
    return all(
        np.asarray(getattr(left, name), dtype=dtype).tobytes(order="C")
        == np.asarray(getattr(right, name), dtype=dtype).tobytes(order="C")
        for name, dtype in (
            ("joint_position", "<f4"),
            ("joint_velocity", "<f4"),
            ("body_quat_w", "<f4"),
            ("frame_index", "<i8"),
        )
    )


@dataclass(frozen=True)
class _GearRuntimeInputs:
    policy: Path
    observation_config: Path
    encoder: Path | None


@dataclass(frozen=True)
class _KnownGoodProjection:
    reference_base: Path
    reference_leaf: Path
    canonical: CanonicalTargetBuffer
    body_position: np.ndarray
    evidence: Mapping[str, object]


def _projection_rule_sha256() -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            dict(_KNOWN_GOOD_PROJECTION_RULE),
            "known-good projection rule",
        )
    )


def _csv_f32_bytes(values: np.ndarray, headers: Sequence[str]) -> bytes:
    array = np.asarray(values, dtype="<f4")
    if (
        array.ndim != 2
        or array.shape[0] <= 0
        or array.shape[1] != len(headers)
        or not np.all(np.isfinite(array))
    ):
        raise ContractError("projected CSV values are invalid")
    rows = [",".join(headers)]
    rows.extend(
        ",".join(format(float(value), ".17g") for value in row)
        for row in array
    )
    return ("\n".join(rows) + "\n").encode("ascii")


def _known_good_projection_info() -> bytes:
    return (
        "Motion Information: known_good\n"
        "==================================================\n\n"
        "joint_pos:\n  Shape: (441, 29)\n  Dtype: float32\n\n"
        "joint_vel:\n  Shape: (441, 29)\n  Dtype: float32\n\n"
        "body_pos_w:\n  Shape: (441, 1, 3)\n  Dtype: float32\n\n"
        "body_quat_w:\n  Shape: (441, 1, 4)\n  Dtype: float32\n\n"
        "body_lin_vel_w:\n  Shape: (441, 1, 3)\n  Dtype: float32\n\n"
        "body_ang_vel_w:\n  Shape: (441, 1, 3)\n  Dtype: float32\n\n"
        "_body_indexes:\n  Shape: (1,)\n  Dtype: int64\n  Sample: [0]\n\n"
        "time_step_total:\n  Shape: ()\n  Dtype: int64\n  Sample: [441]\n"
    ).encode("ascii")


def _known_good_projection_metadata() -> bytes:
    return (
        "Metadata for: known_good\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[0]\n\n"
        "Total timesteps: 441\n\n"
        "Data arrays summary:\n"
        "  joint_pos: (441, 29) (float32)\n"
        "  joint_vel: (441, 29) (float32)\n"
        "  body_pos_w: (441, 1, 3) (float32)\n"
        "  body_quat_w: (441, 1, 4) (float32)\n"
        "  body_lin_vel_w: (441, 1, 3) (float32)\n"
        "  body_ang_vel_w: (441, 1, 3) (float32)\n"
        "  _body_indexes: (1,) (int64)\n"
        "  time_step_total: () (int64)\n"
    ).encode("ascii")


def _dynamic_launch_identity(
    inputs: ExternalInputs,
    runtime: _GearRuntimeInputs,
    scene: RegisteredScene | object,
    initial_qpos: np.ndarray,
    canonical: CanonicalTargetBuffer,
    body_position: np.ndarray,
    projection: _KnownGoodProjection | None = None,
) -> Mapping[str, object]:
    if not isinstance(inputs, ExternalInputs) or not isinstance(
        runtime, _GearRuntimeInputs
    ):
        raise ContractError("dynamic launch parity requires authenticated inputs")
    input_pairs = (
        ("policy", inputs.policy, runtime.policy),
        (
            "observation_config",
            inputs.observation_config,
            runtime.observation_config,
        ),
    )
    fields: dict[str, str] = {}
    for name, external, staged in input_pairs:
        external_digest = _sha256_file(external)
        staged_digest = _sha256_file(staged)
        if external_digest != staged_digest:
            raise ContractError(
                f"dynamic launch parity changed for {name}"
            )
        fields[name] = external_digest
    if inputs.encoder is None:
        if runtime.encoder is not None:
            raise ContractError("dynamic launch parity changed for absent encoder")
        fields["encoder"] = _sha256_bytes(b"mm-sonic-encoder/absent\n")
    else:
        if runtime.encoder is None:
            raise ContractError("dynamic launch parity lost the staged encoder")
        external_encoder = _sha256_file(inputs.encoder)
        staged_encoder = _sha256_file(runtime.encoder)
        if external_encoder != staged_encoder:
            raise ContractError("dynamic launch parity changed for encoder")
        fields["encoder"] = external_encoder
    official_model = inputs.gear_checkout / _GEAR_MODEL_RELATIVE
    generated_scene = getattr(scene, "gear_scene_xml", None)
    if not isinstance(generated_scene, Path):
        raise ContractError("dynamic launch parity lacks the generated scene")
    fields.update(
        {
            "official_model": _sha256_file(official_model),
            "generated_scene": _sha256_file(generated_scene),
            "initial_qpos": _sha256_bytes(
                np.asarray(initial_qpos, dtype="<f8").tobytes(order="C")
            ),
            "canonical_reference": _canonical_target_sha256(canonical),
            "reference_body_position": _require_streamable_body_position(
                body_position
            ),
        }
    )
    if projection is not None:
        if not isinstance(projection, _KnownGoodProjection):
            raise ContractError("dynamic launch parity projection is invalid")
        source = projection.evidence.get("source")
        if not isinstance(source, Mapping):
            raise ContractError("dynamic launch parity lacks projection source")
        source_hashes = source.get("sha256")
        if not isinstance(source_hashes, Mapping):
            raise ContractError("dynamic launch parity lacks projection hashes")
        fields.update(
            {
                "known_good_source_set": _sha256_bytes(
                    _canonical_json_bytes(
                        dict(source_hashes),
                        "known-good source hash set",
                    )
                ),
                "known_good_projection_rule": str(
                    projection.evidence.get("rule_sha256")
                ),
                "known_good_projection": str(
                    projection.evidence.get("projection_sha256")
                ),
            }
        )
    payload = _canonical_json_bytes(fields, "dynamic launch parity fields")
    return MappingProxyType(
        {
            "fields": MappingProxyType(dict(sorted(fields.items()))),
            "sha256": _sha256_bytes(payload),
        }
    )


def _require_launch_parity(
    file_identity: Mapping[str, object],
    stream_identity: Mapping[str, object],
) -> Mapping[str, object]:
    file_copy = {
        "fields": dict(file_identity.get("fields", {})),
        "sha256": file_identity.get("sha256"),
    }
    stream_copy = {
        "fields": dict(stream_identity.get("fields", {})),
        "sha256": stream_identity.get("sha256"),
    }
    if file_copy != stream_copy:
        raise ContractError(
            "file-to-stream dynamic launch parity changed before launch"
        )
    return MappingProxyType(
        {
            "file_sha256": file_copy["sha256"],
            "stream_sha256": stream_copy["sha256"],
            "equal": True,
        }
    )


def _official_target_row(
    canonical: CanonicalTargetBuffer,
    index: int,
    body_position: np.ndarray | None = None,
) -> bytes:
    if not isinstance(canonical, CanonicalTargetBuffer):
        raise ContractError("official target audit requires a canonical buffer")
    if type(index) is not int or index < 0 or index >= canonical.count:
        raise ContractError("official target audit frame index is invalid")
    position = (
        np.zeros(3, dtype=np.float32)
        if body_position is None
        else np.asarray(body_position[index], dtype=np.float32)
    )
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ContractError("official target audit body position is invalid")
    permutation = np.asarray(LOGGER_JOINT_PERMUTATION, dtype=np.int64)
    values = np.concatenate(
        (
            position,
            canonical.body_quat_w[index],
            canonical.joint_position[index, permutation],
        )
    ).astype(np.float32, copy=False)
    return (
        ",".join(format(float(value), ".6g") for value in values) + ",\n"
    ).encode("ascii")


@dataclass(frozen=True)
class _WaitForControlEvidence:
    target_device: int
    target_inode: int
    target_rows: int
    q_rows: int
    base_rows: int
    target_sha256: str


@dataclass(frozen=True)
class _StreamPreloadEvidence:
    post_enable_fence: Mapping[str, object]
    readiness_publication: Mapping[str, object]
    logical_publications: tuple[Mapping[str, object], ...]
    padding_publication: Mapping[str, object]
    receipt_fence_publication: Mapping[str, object]
    consumer_markers: tuple[Mapping[str, object], ...]
    causal_fences: tuple[str, ...]


@dataclass(frozen=True)
class _TargetCoverageResult:
    target_rows: int
    control_drive_steps: int
    control_drive_duration_s: float
    simulator_steps: int
    simulator_duration_s: float
    state_rows: int
    contact_rows: int
    wall_duration_s: float
    target_device: int
    target_inode: int
    target_sha256: str
    retained_reader: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _GearControlRows:
    indices: np.ndarray
    common_prefixes: tuple[tuple[str, ...], ...]
    time_monotonic_ms: np.ndarray
    joint_position_target_order: np.ndarray
    base_quat_w: np.ndarray


def _process_group_exists(pgid: int) -> bool:
    """Return whether a Linux process group still has any procfs members."""

    return type(pgid) is int and pgid > 0 and bool(_linux_group_states(pgid))


def _stable_regular_file_snapshot(
    path: Path,
    label: str,
    *,
    allow_missing: bool = False,
) -> tuple[int, int, bytes] | None:
    """Read one regular single-link file without accepting a racing identity."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    for _attempt in range(8):
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            if allow_missing:
                return None
            raise ContractError(f"{label} is unavailable")
        except OSError as error:
            raise ContractError(f"cannot open {label}") from error
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
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before == identity_after and len(raw) == after.st_size:
            return after.st_dev, after.st_ino, raw
        time.sleep(0)
    raise ContractError(f"{label} changed while being read")


def _zero_state_log_rows(path: Path, header: tuple[str, ...], label: str) -> int:
    snapshot = _stable_regular_file_snapshot(path, label, allow_missing=True)
    if snapshot is None or not snapshot[2]:
        return 0
    try:
        text = snapshot[2].decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError(f"{label} is not ASCII CSV") from error
    if not text.endswith("\n"):
        raise ContractError(f"{label} ended with a partial row")
    rows = list(csv.reader(text.splitlines(), strict=True))
    if not rows or tuple(rows[0]) != header:
        raise ContractError(f"{label} header changed")
    return len(rows) - 1


def _audit_wait_for_control_epoch(
    gear: object,
    target_motion_logfile: Path,
    logs_dir: Path,
) -> _WaitForControlEvidence:
    """Prove the cold process is prepared in WAIT with zero policy rows."""

    if (
        not bool(getattr(gear, "wait_for_control_ready", False))
        or bool(getattr(gear, "control_active", False))
        or not bool(getattr(gear, "input_prepared", False))
    ):
        raise ContractError(
            "zero-row audit requires a prepared authenticated WAIT_FOR_CONTROL"
        )
    gear.require_alive()
    target = _stable_regular_file_snapshot(
        Path(target_motion_logfile), "official target log"
    )
    assert target is not None
    target_raw = target[2]
    if target_raw and not target_raw.endswith(b"\n"):
        raise ContractError("official target log ended with a partial WAIT row")
    target_rows = len(target_raw.splitlines())
    q_rows = _zero_state_log_rows(
        Path(logs_dir) / "q.csv", _GEAR_Q_HEADER, "GEAR q log"
    )
    base_rows = _zero_state_log_rows(
        Path(logs_dir) / "base_quat.csv",
        _GEAR_BASE_QUAT_HEADER,
        "GEAR base quaternion log",
    )
    if target_rows or q_rows or base_rows:
        raise ContractError(
            "authenticated WAIT_FOR_CONTROL emitted policy/state/target rows"
        )
    return _WaitForControlEvidence(
        target_device=target[0],
        target_inode=target[1],
        target_rows=target_rows,
        q_rows=q_rows,
        base_rows=base_rows,
        target_sha256=_sha256_bytes(target_raw),
    )


def _preload_buffer(
    canonical: CanonicalTargetBuffer, indices: np.ndarray
) -> CanonicalTargetBuffer:
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0):
        raise ContractError("preload transport indices are invalid")
    source = np.minimum(indices, canonical.count - 1)
    return CanonicalTargetBuffer(
        joint_position=canonical.joint_position[source],
        joint_velocity=canonical.joint_velocity[source],
        body_quat_w=canonical.body_quat_w[source],
        frame_index=np.asarray(indices, dtype="<i8"),
    )


def _preload_known_good_stream(
    canonical: CanonicalTargetBuffer,
    publisher: object,
    gear: object,
    post_enable_fence: Mapping[str, object],
) -> _StreamPreloadEvidence:
    """Publish the complete logical sequence plus padding/fence while in WAIT."""

    plan = build_stream_transport_plan(canonical)
    if (
        not bool(getattr(gear, "wait_for_control_ready", False))
        or bool(getattr(gear, "control_active", False))
        or not bool(getattr(gear, "input_prepared", False))
        or bool(gear.group_is_stopped())
    ):
        raise ContractError("known-good preload requires a running prepared WAIT epoch")
    gear.require_alive()
    if not isinstance(post_enable_fence, Mapping):
        raise ContractError("known-good preload requires post-enable fence evidence")

    readiness_buffer = plan.readiness
    logical_buffers = list(plan.logical)
    padding = plan.padding
    receipt_fence = plan.receipt_fence

    consumer_markers: list[Mapping[str, object]] = []

    def publish_one(
        buffer: CanonicalTargetBuffer,
        *,
        phase: str,
        attempt: int | None = None,
    ) -> Mapping[str, object]:
        boundary = gear.publication_boundary()
        prepared = publisher.prepare(buffer, phase=phase, attempt=attempt)
        sent = publisher.send_prepared(prepared)
        record = MappingProxyType(dict(sent))
        if not bool(record.get("local_send_completed", False)):
            raise ContractError("preload publication did not complete its local send")
        marker = gear.wait_for_stream_processing(
            boundary,
            frame_count=buffer.count,
            global_start=int(buffer.frame_index[0]),
            merged_count=int(buffer.frame_index[-1]) + 1,
        )
        consumer_markers.append(MappingProxyType(dict(marker)))
        return record

    readiness_publication = publish_one(
        readiness_buffer, phase="readiness", attempt=1
    )
    logical_publications = tuple(
        publish_one(buffer, phase="logical") for buffer in logical_buffers
    )
    padding_publication = publish_one(padding, phase="padding")
    receipt_fence_publication = publish_one(
        receipt_fence, phase="receipt_fence"
    )

    starts = [
        int(buffer.frame_index[0])
        for buffer in (readiness_buffer, *logical_buffers, padding, receipt_fence)
    ]
    fences: list[str] = []
    for index in range(len(starts) - 1):
        if index == len(logical_buffers):
            fences.append(
                f"padding-start-fences-logical-{canonical.count - 1}"
            )
        elif index == len(logical_buffers) + 1:
            fences.append("fence-start-fences-padding")
        else:
            fences.append(
                f"publication-start-{starts[index + 1]}-fences-{starts[index]}"
            )
    return _StreamPreloadEvidence(
        post_enable_fence=MappingProxyType(dict(post_enable_fence)),
        readiness_publication=readiness_publication,
        logical_publications=logical_publications,
        padding_publication=padding_publication,
        receipt_fence_publication=receipt_fence_publication,
        consumer_markers=tuple(consumer_markers),
        causal_fences=tuple(fences),
    )


def _archived_preload_buffer(
    bundle: RunBundle,
    record: Mapping[str, object],
    *,
    phase: str,
    attempt: int | None,
) -> tuple[CanonicalTargetBuffer, bytes]:
    if (
        not isinstance(record, Mapping)
        or record.get("phase") != phase
        or record.get("attempt") != attempt
        or not bool(record.get("local_send_completed", False))
    ):
        raise ContractError(
            f"preload {phase} transcript publication metadata changed"
        )
    message_relative = record.get("message_path")
    digest_relative = record.get("digest_path")
    digest = record.get("sha256")
    if not all(
        type(value) is str and value
        for value in (message_relative, digest_relative, digest)
    ):
        raise ContractError("preload transcript archive metadata is incomplete")
    message = _stable_regular_file_snapshot(
        bundle.path / str(message_relative), "preload message archive"
    )
    digest_file = _stable_regular_file_snapshot(
        bundle.path / str(digest_relative), "preload digest archive"
    )
    assert message is not None and digest_file is not None
    observed_digest = _sha256_bytes(message[2])
    leaf = Path(str(message_relative)).name
    if (
        observed_digest != digest
        or digest_file[2] != f"{digest}  {leaf}\n".encode("ascii")
    ):
        raise ContractError("preload transcript archive digest changed")
    return decode_pose_v1(message[2]).buffer, message[2]


def _buffers_bit_equal_exact(
    actual: CanonicalTargetBuffer, expected: CanonicalTargetBuffer
) -> bool:
    return all(
        left.shape == right.shape
        and np.asarray(left, dtype=dtype).tobytes(order="C")
        == np.asarray(right, dtype=dtype).tobytes(order="C")
        for left, right, dtype in (
            (actual.joint_position, expected.joint_position, "<f4"),
            (actual.joint_velocity, expected.joint_velocity, "<f4"),
            (actual.body_quat_w, expected.body_quat_w, "<f4"),
            (actual.frame_index, expected.frame_index, "<i8"),
        )
    )


def _audit_known_good_stream_preload(
    canonical: CanonicalTargetBuffer,
    bundle: RunBundle,
    evidence: _StreamPreloadEvidence,
) -> Mapping[str, object]:
    """Re-read and decode every archived logical and transport publication."""

    if not isinstance(evidence, _StreamPreloadEvidence):
        raise ContractError("preload transcript evidence has the wrong type")
    plan = build_stream_transport_plan(canonical)
    expected_marker_count = plan.publication_count
    expected_fence_count = expected_marker_count - 1
    if (
        len(evidence.logical_publications) != len(plan.logical)
        or len(evidence.consumer_markers) != expected_marker_count
        or len(evidence.causal_fences) != expected_fence_count
    ):
        raise ContractError("preload transport transcript is incomplete")
    expected_readiness = plan.readiness
    expected_logical = list(plan.logical)
    expected_padding = plan.padding
    expected_receipt_fence = plan.receipt_fence
    observed_readiness, readiness_payload = _archived_preload_buffer(
        bundle,
        evidence.readiness_publication,
        phase="readiness",
        attempt=1,
    )
    if not _buffers_bit_equal_exact(observed_readiness, expected_readiness):
        raise ContractError("preload readiness transcript changed")
    logical_payloads: list[bytes] = []
    for record, expected in zip(
        evidence.logical_publications, expected_logical, strict=True
    ):
        observed, payload = _archived_preload_buffer(
            bundle, record, phase="logical", attempt=None
        )
        if not _buffers_bit_equal_exact(observed, expected):
            raise ContractError("preload logical transcript changed")
        logical_payloads.append(payload)
    observed_padding, padding_payload = _archived_preload_buffer(
        bundle,
        evidence.padding_publication,
        phase="padding",
        attempt=None,
    )
    if not _buffers_bit_equal_exact(observed_padding, expected_padding):
        raise ContractError("preload padding transport transcript changed")
    observed_fence, receipt_fence_payload = _archived_preload_buffer(
        bundle,
        evidence.receipt_fence_publication,
        phase="receipt_fence",
        attempt=None,
    )
    if not _buffers_bit_equal_exact(observed_fence, expected_receipt_fence):
        raise ContractError("preload receipt-fence transport transcript changed")
    expected_markers = [
        expected_readiness,
        *expected_logical,
        expected_padding,
        expected_receipt_fence,
    ]
    stdout_snapshot = _stable_regular_file_snapshot(
        bundle.path / "dynamic/stream/gear.stdout",
        "GEAR stream stdout archive",
    )
    assert stdout_snapshot is not None
    stdout_prefix = stdout_snapshot[2]
    try:
        stdout_prefix.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("GEAR stream stdout archive is not UTF-8") from error

    fence = evidence.post_enable_fence
    fence_keys = frozenset(
        (
            "boundary",
            "end_offset",
            "key_sequence",
            "left_line",
            "right_line",
            "semantics",
        )
    )
    if not isinstance(fence, Mapping) or set(fence) != fence_keys:
        raise ContractError("post-enable stream fence evidence changed")
    fence_boundary = fence.get("boundary")
    fence_end = fence.get("end_offset")
    enabled_line = b"ZMQ STREAMING MODE: ENABLED\n"
    left_line = f"{_POST_ENABLE_LEFT_LINE}\n".encode("ascii")
    right_line = f"{_POST_ENABLE_RIGHT_LINE}\n".encode("ascii")
    if (
        type(fence_boundary) is not int
        or fence_boundary < len(enabled_line)
        or type(fence_end) is not int
        or fence_end <= fence_boundary
        or fence_end > len(stdout_prefix)
        or fence.get("key_sequence") != "qe"
        or fence.get("left_line") != _POST_ENABLE_LEFT_LINE
        or fence.get("right_line") != _POST_ENABLE_RIGHT_LINE
        or fence.get("semantics") != _POST_ENABLE_FENCE_SEMANTICS
        or stdout_prefix[fence_boundary - len(enabled_line) : fence_boundary]
        != enabled_line
    ):
        raise ContractError("post-enable stream fence evidence changed")
    fence_lines = stdout_prefix[fence_boundary:fence_end].splitlines(
        keepends=True
    )
    if (
        fence_lines.count(left_line) != 1
        or fence_lines.count(right_line) != 1
        or fence_lines.index(left_line) >= fence_lines.index(right_line)
        or fence_lines[-1] != right_line
    ):
        raise ContractError("post-enable stream fence transcript changed")

    marker_keys = frozenset(
        (
            "boundary",
            "end_offset",
            "frame_count",
            "global_start",
            "merged_count",
            "start_line",
            "processing_line",
            "merged_line",
            "end_line",
            "start_range",
            "processing_range",
            "merged_range",
            "end_range",
        )
    )
    previous_end = fence_end
    validated_markers: list[dict[str, object]] = []

    def exact_event_range(value: object) -> tuple[int, int] | None:
        if (
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not int
            or type(value[1]) is not int
            or value[0] < 0
            or value[1] <= value[0]
            or value[1] > len(stdout_prefix)
        ):
            return None
        return value

    for index, (marker, expected) in enumerate(zip(
        evidence.consumer_markers, expected_markers, strict=True
    )):
        if not isinstance(marker, Mapping):
            raise ContractError("preload consumer marker has the wrong type")
        processing = (
            f"[StreamedMotionMerger] Processing {expected.count} frames, "
            f"incoming_frame_start={int(expected.frame_index[0])}, frame_step=1"
        )
        merged_count = int(expected.frame_index[-1]) + 1
        copied = merged_count - expected.count
        merged = (
            f"[StreamedMotionMerger] Merged motion: {merged_count} frames "
            f"(copied: {copied} + incoming: {expected.count})"
        )
        boundary = marker.get("boundary")
        end_offset = marker.get("end_offset")
        start_range = exact_event_range(marker.get("start_range"))
        processing_range = exact_event_range(marker.get("processing_range"))
        merged_range = exact_event_range(marker.get("merged_range"))
        end_range = exact_event_range(marker.get("end_range"))
        if (
            set(marker) != marker_keys
            or type(boundary) is not int
            or boundary < previous_end
            or type(end_offset) is not int
            or start_range is None
            or processing_range is None
            or merged_range is None
            or end_range is None
            or start_range[0] < boundary
            or start_range[1] > processing_range[0]
            or processing_range[1] > merged_range[0]
            or merged_range[1] > end_range[0]
            or end_offset != end_range[1]
            or marker.get("frame_count") != expected.count
            or marker.get("global_start") != int(expected.frame_index[0])
            or marker.get("merged_count") != merged_count
            or marker.get("start_line") != _STREAM_PROCESSING_START_LINE
            or marker.get("processing_line") != processing
            or marker.get("merged_line") != merged
            or marker.get("end_line") != _STREAM_PROCESSING_END_LINE
        ):
            raise ContractError(
                "preload consumer marker or causal boundary transcript changed"
            )
        assert start_range is not None
        assert processing_range is not None
        assert merged_range is not None
        assert end_range is not None
        expected_events = (
            (start_range, f"{_STREAM_PROCESSING_START_LINE}\n".encode("ascii")),
            (processing_range, f"{processing}\n".encode("ascii")),
            (merged_range, f"{merged}\n".encode("ascii")),
            (end_range, f"{_STREAM_PROCESSING_END_LINE}\n".encode("ascii")),
        )
        if any(
            stdout_prefix[start:end] != expected_line
            for (start, end), expected_line in expected_events
        ):
            raise ContractError("preload consumer stdout event line changed")
        validated_markers.append(dict(marker))
        previous_end = end_range[1]
    if not stdout_prefix.endswith(b"\n"):
        raise ContractError("GEAR stream stdout prefix ends with a partial line")
    start_bytes = f"{_STREAM_PROCESSING_START_LINE}\n".encode("ascii")
    end_bytes = f"{_STREAM_PROCESSING_END_LINE}\n".encode("ascii")
    if (
        stdout_prefix.count(start_bytes) != len(expected_markers)
        or stdout_prefix.count(end_bytes) != len(expected_markers)
    ):
        raise ContractError("GEAR stream stdout has extra consumer boundaries")

    first_boundary = int(validated_markers[0]["boundary"])
    transcript_triplets: list[dict[str, object]] = []
    for index, marker in enumerate(validated_markers):
        boundary = int(marker["boundary"])
        start_start, start_end = marker["start_range"]
        processing_start, processing_end = marker["processing_range"]
        merged_start, merged_end = marker["merged_range"]
        end_start, end_end = marker["end_range"]
        post_end = (
            int(validated_markers[index + 1]["boundary"])
            if index + 1 < len(validated_markers)
            else len(stdout_prefix)
        )
        contiguous_start = boundary
        contiguous_end = end_end
        transcript_triplets.append(
            {
                "index": index,
                **marker,
                "event_sha256": {
                    "start": _sha256_bytes(stdout_prefix[start_start:start_end]),
                    "processing": _sha256_bytes(
                        stdout_prefix[processing_start:processing_end]
                    ),
                    "merged": _sha256_bytes(stdout_prefix[merged_start:merged_end]),
                    "end": _sha256_bytes(stdout_prefix[end_start:end_end]),
                },
                "intervening_ranges": {
                    "publication_to_start": [boundary, start_start],
                    "start_to_processing": [start_end, processing_start],
                    "processing_to_merged": [processing_end, merged_start],
                    "merged_to_end": [merged_end, end_start],
                    "end_to_next_publication_or_prefix": [end_end, post_end],
                },
                "intervening_sha256": {
                    "publication_to_start": _sha256_bytes(
                        stdout_prefix[boundary:start_start]
                    ),
                    "start_to_processing": _sha256_bytes(
                        stdout_prefix[start_end:processing_start]
                    ),
                    "processing_to_merged": _sha256_bytes(
                        stdout_prefix[processing_end:merged_start]
                    ),
                    "merged_to_end": _sha256_bytes(
                        stdout_prefix[merged_end:end_start]
                    ),
                    "end_to_next_publication_or_prefix": _sha256_bytes(
                        stdout_prefix[end_end:post_end]
                    ),
                },
                "range_sha256": _sha256_bytes(
                    stdout_prefix[contiguous_start:contiguous_end]
                ),
            }
        )
    starts = [int(buffer.frame_index[0]) for buffer in expected_markers]
    expected_fences: list[str] = []
    for index in range(len(starts) - 1):
        if index == len(expected_logical):
            expected_fences.append(
                f"padding-start-fences-logical-{canonical.count - 1}"
            )
        elif index == len(expected_logical) + 1:
            expected_fences.append("fence-start-fences-padding")
        else:
            expected_fences.append(
                f"publication-start-{starts[index + 1]}-fences-{starts[index]}"
            )
    if evidence.causal_fences != tuple(expected_fences):
        raise ContractError("preload transport causal fences changed")
    transcript = {
        "schema": _PRELOAD_CONSUMER_TRANSCRIPT_SCHEMA,
        "stdout_path": "dynamic/stream/gear.stdout",
        "stdout_prefix_length": len(stdout_prefix),
        "stdout_prefix_sha256": _sha256_bytes(stdout_prefix),
        "post_enable_fence": dict(fence),
        "post_enable_to_first_publication_range": [fence_end, first_boundary],
        "post_enable_to_first_publication_sha256": _sha256_bytes(
            stdout_prefix[fence_end:first_boundary]
        ),
        "consumer_triplets": transcript_triplets,
        "causal_fences": list(evidence.causal_fences),
    }
    transcript_bytes = _canonical_json_bytes(
        transcript, "preload consumer transcript"
    )
    bundle.write_bytes(_PRELOAD_CONSUMER_TRANSCRIPT_PATH, transcript_bytes)
    return MappingProxyType(
        {
            "readiness_exact": True,
            "logical_exact": True,
            "padding_exact": True,
            "receipt_fence_exact": True,
            "readiness_frames": expected_readiness.count,
            "logical_frames": sum(buffer.count for buffer in expected_logical),
            "padding_frames": expected_padding.count,
            "receipt_fence_frames": expected_receipt_fence.count,
            "consumer_marker_count": len(evidence.consumer_markers),
            "causal_fence_count": len(evidence.causal_fences),
            "post_enable_fence": dict(fence),
            "stdout_prefix_length": len(stdout_prefix),
            "stdout_prefix_sha256": _sha256_bytes(stdout_prefix),
            "consumer_transcript_path": _PRELOAD_CONSUMER_TRANSCRIPT_PATH,
            "consumer_transcript_sha256": _sha256_bytes(transcript_bytes),
            "readiness_sha256": _sha256_bytes(readiness_payload),
            "logical_sha256": _sha256_bytes(b"".join(logical_payloads)),
            "padding_sha256": _sha256_bytes(padding_payload),
            "receipt_fence_sha256": _sha256_bytes(receipt_fence_payload),
            "consumer_markers_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    [dict(marker) for marker in evidence.consumer_markers],
                    "preload consumer marker transcript",
                )
            ),
            "causal_fences_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    list(evidence.causal_fences),
                    "preload causal fence transcript",
                )
            ),
        }
    )


class _AuthoritativeTargetReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            self._descriptor = os.open(self.path, flags)
            metadata = os.fstat(self._descriptor)
        except OSError as error:
            raise ContractError("cannot open official target log") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(self._descriptor)
            raise ContractError("official target log must be one regular file")
        self.device = metadata.st_dev
        self.inode = metadata.st_ino

    def read(self) -> bytes:
        try:
            path_metadata = self.path.lstat()
        except OSError as error:
            raise ContractError("official target log identity changed") from error
        if (
            stat.S_ISLNK(path_metadata.st_mode)
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (self.device, self.inode)
        ):
            raise ContractError("official target log identity changed")
        for _attempt in range(8):
            before = os.fstat(self._descriptor)
            os.lseek(self._descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(self._descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = os.fstat(self._descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                and len(raw) == after.st_size
            ):
                return raw
            time.sleep(0)
        raise ContractError("official target log changed while being read")

    def close(self) -> None:
        descriptor = getattr(self, "_descriptor", None)
        if descriptor is None:
            return
        self._descriptor = None
        os.close(descriptor)


def _validate_target_prefix(
    raw: bytes,
    expected_rows: Sequence[bytes],
) -> int:
    if raw and not raw.endswith(b"\n"):
        raise ContractError("official target log ended with a partial row")
    rows = raw.splitlines(keepends=True)
    if len(rows) > len(expected_rows):
        raise ContractError(
            "official target log overshoot exceeded "
            f"{len(expected_rows)} rows"
        )
    for index, row in enumerate(rows):
        if row != expected_rows[index]:
            raise ContractError(
                "official target log is not the exact canonical prefix"
            )
    return len(rows)


def _drive_authoritative_target_coverage(
    gear: object,
    simulator: object,
    target_motion_logfile: Path,
    canonical: CanonicalTargetBuffer,
    *,
    body_position: np.ndarray | None = None,
    increment_steps: int = 1,
    maximum_wall_seconds: float = _LOW_STATE_BOOTSTRAP_TIMEOUT_S,
    retain_reader: bool = False,
    required_control_duration_s: float | None = None,
    control_lead_rows: int = 0,
) -> _TargetCoverageResult:
    """Advance physics until GEAR logs the complete registered target buffer."""

    if (
        not isinstance(canonical, CanonicalTargetBuffer)
        or canonical.count
        not in (_KNOWN_GOOD_PROJECTED_FRAME_COUNT, STAGE_B_FRAME_COUNT)
        or type(increment_steps) is not int
        or increment_steps <= 0
        or type(maximum_wall_seconds) not in (int, float)
        or not math.isfinite(float(maximum_wall_seconds))
        or maximum_wall_seconds <= 0.0
        or type(retain_reader) is not bool
        or type(control_lead_rows) is not int
        or control_lead_rows < 0
        or (required_control_duration_s is None and control_lead_rows != 0)
        or (
            required_control_duration_s is not None
            and (
                type(required_control_duration_s) not in (int, float)
                or not math.isfinite(float(required_control_duration_s))
                or float(required_control_duration_s) <= 0.0
            )
        )
    ):
        raise ValueError("authoritative target coverage bounds are invalid")
    if (
        not bool(getattr(gear, "control_active", False))
        or bool(gear.group_is_stopped())
        or not bool(gear.group_is_resumed())
    ):
        raise ContractError("target coverage requires one running CONTROL epoch")
    gear.require_alive()
    expected_rows = tuple(
        _official_target_row(canonical, index, body_position)
        for index in range(canonical.count)
    )
    initial_snapshot = simulator.snapshot()
    required_snapshot = {"steps", "sim_time_s", "state_rows", "contact_rows"}
    if not isinstance(initial_snapshot, Mapping) or set(initial_snapshot) != required_snapshot:
        raise ContractError("initial simulator boundary snapshot is invalid")
    initial_steps = initial_snapshot["steps"]
    initial_duration = initial_snapshot["sim_time_s"]
    initial_state_rows = initial_snapshot["state_rows"]
    initial_contact_rows = initial_snapshot["contact_rows"]
    if (
        type(initial_steps) is not int
        or initial_steps < 0
        or type(initial_state_rows) is not int
        or initial_state_rows < 0
        or type(initial_contact_rows) is not int
        or initial_contact_rows < 0
        or type(initial_duration) not in (int, float)
        or not math.isfinite(float(initial_duration))
        or initial_duration < 0.0
    ):
        raise ContractError("initial simulator boundary counters are invalid")
    required_control_steps: int | None = None
    if required_control_duration_s is not None:
        sim_dt = getattr(simulator, "sim_dt", None)
        if (
            type(sim_dt) not in (int, float)
            or not math.isfinite(float(sim_dt))
            or float(sim_dt) <= 0.0
        ):
            raise ContractError("required control duration needs a finite simulator dt")
        required_control_steps = round(
            float(required_control_duration_s) / float(sim_dt)
        )
        if (
            required_control_steps <= 0
            or abs(
                required_control_steps * float(sim_dt)
                - float(required_control_duration_s)
            )
            > 1.0e-12
        ):
            raise ContractError(
                "required control duration is not an exact simulator-step count"
            )
    reader = _AuthoritativeTargetReader(Path(target_motion_logfile))
    completed = False
    driven_steps = 0
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + float(maximum_wall_seconds)
    try:
        while True:
            gear.require_alive()
            raw = reader.read()
            terminal_pre_stopped = False
            if (
                required_control_steps is not None
                and (not raw or raw.endswith(b"\n"))
                and raw.count(b"\n") >= canonical.count
            ):
                gear.stop_group()
                raw = reader.read()
                terminal_pre_stopped = True
            count = _validate_target_prefix(raw, expected_rows)
            if required_control_steps is not None:
                desired_steps = (
                    required_control_steps
                    if count == canonical.count
                    else min(
                        required_control_steps,
                        max(
                            1,
                            math.ceil(
                                (count + control_lead_rows)
                                * required_control_steps
                                / (canonical.count - 1)
                            ),
                        ),
                    )
                )
                advance_steps = desired_steps - driven_steps
                if advance_steps < 0:
                    raise ContractError(
                        "active CONTROL physics exceeded its row-paced duration"
                    )
                if advance_steps:
                    advance = simulator.advance(advance_steps)
                    if (
                        not hasattr(advance, "steps")
                        or advance.steps != advance_steps
                    ):
                        raise ContractError(
                            "simulator did not advance the exact row-paced interval"
                        )
                    driven_steps += advance_steps
                if count != canonical.count:
                    if time.monotonic() >= deadline:
                        raise ProcessError(
                            "timed out waiting for exact authoritative target coverage"
                        )
                    if advance_steps == 0:
                        time.sleep(
                            min(0.001, max(0.0, deadline - time.monotonic()))
                        )
                    continue
            if count == canonical.count:
                if required_control_steps is not None:
                    remaining = required_control_steps - driven_steps
                    if remaining < 0:
                        raise ContractError(
                            "active CONTROL physics exceeded the required duration"
                        )
                    if remaining:
                        raise ContractError(
                            "terminal target row arrived before exact CONTROL physics"
                        )
                    if not terminal_pre_stopped:
                        raise ContractError(
                            "exact CONTROL coverage lacked its pre-audit stop fence"
                        )
                if not bool(gear.group_is_stopped()):
                    gear.stop_group()
                stopped = reader.read()
                if stopped != raw:
                    raise ContractError(
                        "official target log changed at the post-stop boundary"
                    )
                stable = reader.read()
                if stable != stopped:
                    raise ContractError(
                        "official target log changed after the post-stop audit"
                    )
                snapshot = simulator.snapshot()
                if (
                    not isinstance(snapshot, Mapping)
                    or set(snapshot) != required_snapshot
                ):
                    raise ContractError("simulator boundary snapshot is invalid")
                steps = snapshot["steps"]
                duration = snapshot["sim_time_s"]
                state_rows = snapshot["state_rows"]
                contact_rows = snapshot["contact_rows"]
                if (
                    type(steps) is not int
                    or steps < initial_steps
                    or type(state_rows) is not int
                    or state_rows < initial_state_rows
                    or type(contact_rows) is not int
                    or contact_rows < initial_contact_rows
                    or type(duration) not in (int, float)
                    or not math.isfinite(float(duration))
                    or duration < initial_duration
                ):
                    raise ContractError("simulator boundary counters are invalid")
                control_steps = steps - initial_steps
                control_duration = float(duration) - float(initial_duration)
                if required_control_steps is not None and (
                    control_steps != required_control_steps
                    or driven_steps != required_control_steps
                    or abs(
                        control_duration - float(required_control_duration_s)
                    )
                    > 1.0e-9
                ):
                    raise ContractError(
                        "active CONTROL physics duration is not exact"
                    )
                result = _TargetCoverageResult(
                    target_rows=count,
                    control_drive_steps=control_steps,
                    control_drive_duration_s=control_duration,
                    simulator_steps=steps,
                    simulator_duration_s=float(duration),
                    state_rows=state_rows,
                    contact_rows=contact_rows,
                    wall_duration_s=(time.monotonic_ns() - started_ns) / 1.0e9,
                    target_device=reader.device,
                    target_inode=reader.inode,
                    target_sha256=_sha256_bytes(stable),
                    retained_reader=reader if retain_reader else None,
                )
                completed = True
                return result
            if time.monotonic() >= deadline:
                raise ProcessError(
                    "timed out waiting for exact authoritative target coverage"
                )
            advance_steps = increment_steps
            advance = simulator.advance(advance_steps)
            if (
                not hasattr(advance, "steps")
                or advance.steps != advance_steps
            ):
                raise ContractError("simulator did not advance the exact increment")
            driven_steps += advance_steps
    except BaseException as error:
        try:
            if not bool(gear.group_is_stopped()):
                gear.stop_group()
        except BaseException as cleanup_error:
            cleanup_error.__cause__ = error
            raise cleanup_error
        raise
    finally:
        if not (completed and retain_reader):
            reader.close()


def _reset_and_prime_scored_epoch(
    gear: object,
    simulator: object,
    *,
    scene_xml: Path,
    initial_qpos: np.ndarray,
    log_dir: Path,
) -> Mapping[str, object]:
    """Reset registered physics and publish one fresh LowState while GEAR stops."""

    if (
        not bool(getattr(gear, "wait_for_control_ready", False))
        or not bool(getattr(gear, "input_prepared", False))
        or bool(getattr(gear, "control_active", False))
        or not bool(gear.group_is_stopped())
    ):
        raise ContractError(
            "scored simulator reset requires a stopped authenticated WAIT epoch"
        )
    reset = simulator.reset(
        scene_xml=scene_xml,
        initial_qpos=initial_qpos,
        lateral_offset_m=0.0,
        yaw_offset_rad=0.0,
        log_dir=log_dir,
    )
    prime = simulator.advance(1)
    if getattr(prime, "steps", None) != 1:
        raise ContractError("fresh LowState prime did not advance one exact step")
    return MappingProxyType(
        {
            "reset": dict(reset),
            "prime": {
                "steps": prime.steps,
                "sim_time_start_s": prime.sim_time_start_s,
                "sim_time_end_s": prime.sim_time_end_s,
                "state_rows": prime.state_rows,
                "contact_rows": prime.contact_rows,
            },
        }
    )


def _parse_exact_gear_csv(
    path: Path,
    expected_header: tuple[str, ...],
    expected_count: int,
    label: str,
) -> tuple[tuple[tuple[str, ...], ...], np.ndarray]:
    snapshot = _stable_regular_file_snapshot(path, label)
    assert snapshot is not None
    raw = snapshot[2]
    if not raw.endswith(b"\n"):
        raise ContractError(f"{label} ended with a partial row")
    try:
        text = raw.decode("ascii")
        rows = list(csv.reader(text.splitlines(), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ContractError(f"{label} is invalid ASCII CSV") from error
    if not rows or tuple(rows[0]) != expected_header:
        raise ContractError(f"{label} exact pinned header changed")
    data = rows[1:]
    if len(data) != expected_count:
        raise ContractError(f"{label} row cardinality is not {expected_count}")
    width = len(expected_header)
    prefixes: list[tuple[str, ...]] = []
    values: list[list[float]] = []
    for index, row in enumerate(data):
        if len(row) != width or row[0] != str(index):
            raise ContractError(f"{label} indices are not exact 0..{expected_count - 1}")
        try:
            numeric = [float(value) for value in row]
        except ValueError as error:
            raise ContractError(f"{label} contains a nonnumeric value") from error
        if not np.all(np.isfinite(np.asarray(numeric, dtype=np.float64))):
            raise ContractError(f"{label} contains a non-finite value")
        prefixes.append(tuple(row[:5]))
        values.append(numeric[5:])
    return tuple(prefixes), np.asarray(values, dtype=np.float64)


def _parse_gear_control_logs(
    logs_dir: Path, expected_count: int
) -> _GearControlRows:
    """Parse exact same-tick q/base rows from the official GEAR CSV sinks."""

    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected GEAR control row count must be positive")
    q_prefixes, q_raw = _parse_exact_gear_csv(
        Path(logs_dir) / "q.csv", _GEAR_Q_HEADER, expected_count, "GEAR q log"
    )
    base_prefixes, base = _parse_exact_gear_csv(
        Path(logs_dir) / "base_quat.csv",
        _GEAR_BASE_QUAT_HEADER,
        expected_count,
        "GEAR base quaternion log",
    )
    if q_prefixes != base_prefixes:
        raise ContractError("GEAR q/base complete five-column prefixes differ")
    if q_raw.shape != (expected_count, 29) or base.shape != (expected_count, 4):
        raise ContractError("GEAR q/base value widths changed")
    monotonic = np.asarray(
        [float(prefix[3]) for prefix in q_prefixes], dtype=np.float64
    )
    if expected_count > 1 and not np.all(np.diff(monotonic) > 0.0):
        raise ContractError("GEAR monotonic timestamps are not strictly increasing")
    norms = np.linalg.norm(base, axis=1)
    if not np.all(np.isfinite(norms)) or not np.allclose(
        norms, 1.0, rtol=0.0, atol=1.0e-5
    ):
        raise ContractError("GEAR base quaternion rows are not normalized")
    target_order = np.asarray(
        q_raw[:, np.asarray(_GEAR_LOGGER_TO_TARGET, dtype=np.int64)],
        dtype="<f4",
    )
    base = np.asarray(base, dtype="<f4")
    indices = np.arange(expected_count, dtype=np.int64)
    for array in (indices, monotonic, target_order, base):
        array.setflags(write=False)
    return _GearControlRows(
        indices=indices,
        common_prefixes=q_prefixes,
        time_monotonic_ms=monotonic,
        joint_position_target_order=target_order,
        base_quat_w=base,
    )


def _tracking_from_gear_control_rows(
    canonical: CanonicalTargetBuffer,
    rows: _GearControlRows,
) -> dict[str, object]:
    """Score exact positional same-CONTROL-tick GEAR state/target pairs."""

    if (
        not isinstance(canonical, CanonicalTargetBuffer)
        or not isinstance(rows, _GearControlRows)
        or canonical.count != rows.indices.size
        or not np.array_equal(canonical.frame_index, rows.indices)
    ):
        raise ContractError("GEAR control rows do not align with canonical targets")
    metrics = tracking_metrics(
        canonical.frame_index,
        canonical.joint_position,
        canonical.body_quat_w,
        rows.indices,
        rows.joint_position_target_order,
        rows.base_quat_w,
    )
    periods = np.diff(rows.time_monotonic_ms)
    period_stats = {
        "count": int(periods.size),
        "minimum": float(np.min(periods)) if periods.size else None,
        "maximum": float(np.max(periods)) if periods.size else None,
        "mean": float(np.mean(periods)) if periods.size else None,
        "rms": (
            float(np.sqrt(np.mean(np.square(periods))))
            if periods.size
            else None
        ),
    }
    return {
        "frame_count": canonical.count,
        "joint_position_rmse_rad": metrics.joint_position_rmse_rad,
        "pelvis_orientation_rms_rad": metrics.pelvis_orientation_rms_rad,
        "joint_tracking_trace_rad": list(metrics.joint_tracking_trace_rad),
        "pelvis_tracking_trace_rad": list(metrics.pelvis_tracking_trace_rad),
        "pairing": "same-control-tick-positional",
        "tick_period_ms": period_stats,
    }


def _timeout_evidence_bytes(
    error: subprocess.TimeoutExpired,
) -> tuple[bytes, bytes]:
    """Normalize TimeoutExpired partial output for immutable evidence files."""

    def normalize(value: object) -> bytes:
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if type(value) is str:
            return value.encode("utf-8", errors="replace")
        raise ContractError("timeout partial output has an invalid type")

    stdout = getattr(error, "stdout", None)
    if stdout is None:
        stdout = getattr(error, "output", None)
    return normalize(stdout), normalize(getattr(error, "stderr", None))


@dataclass(frozen=True)
class _KnownGoodExecution:
    bootstrap_steps: int
    bootstrap: Mapping[str, object]
    wait_maintenance_steps: int
    wait_maintenance: Mapping[str, object]
    wait_epoch: _WaitForControlEvidence
    prime: Mapping[str, object]
    coverage: _TargetCoverageResult
    target_audit: Mapping[str, object]
    metrics: Mapping[str, object]
    preload: _StreamPreloadEvidence | None
    preload_audit: Mapping[str, object] | None


def _audit_closed_target_boundary(
    coverage: _TargetCoverageResult,
    canonical: CanonicalTargetBuffer,
    *,
    body_position: np.ndarray | None,
) -> Mapping[str, object]:
    """Audit the retained descriptor and path after stopped-group termination."""

    reader = coverage.retained_reader
    if not isinstance(reader, _AuthoritativeTargetReader):
        raise ContractError("closed target audit requires the retained target reader")
    try:
        raw = reader.read()
        expected = tuple(
            _official_target_row(canonical, index, body_position)
            for index in range(canonical.count)
        )
        observed = _validate_target_prefix(raw, expected)
        if (
            observed != canonical.count
            or _sha256_bytes(raw) != coverage.target_sha256
        ):
            raise ContractError(
                "official target log changed after stopped-boundary termination"
            )
        stable = reader.read()
        if stable != raw:
            raise ContractError("official target log changed during final-close audit")
        return MappingProxyType(
            {
                "expected_rows": canonical.count,
                "observed_rows": observed,
                "first_frame": 0,
                "last_frame": canonical.count - 1,
                "device": coverage.target_device,
                "inode": coverage.target_inode,
                "exact": True,
                "sha256": coverage.target_sha256,
                "post_stop_stable": True,
                "final_close_stable": True,
            }
        )
    finally:
        reader.close()


def _execute_known_good_scoring_epoch(
    *,
    mode: str,
    gear: GearProcess,
    simulator: GatedSimulatorClient,
    scene: RegisteredScene,
    initial_qpos: np.ndarray,
    canonical: CanonicalTargetBuffer,
    body_position: np.ndarray,
    bundle: RunBundle,
    bootstrap_cancellation: threading.Event,
    publisher: PosePublisher | None = None,
    required_control_duration_s: float | None = None,
    control_lead_rows: int = 0,
) -> _KnownGoodExecution:
    """Run the identical cold WAIT -> scored CONTROL lifecycle for file/stream."""

    if mode not in ("file", "stream") or (mode == "stream") != (
        publisher is not None
    ):
        raise ValueError("known-good scoring mode/publisher combination is invalid")
    bootstrap: dict[str, object] = {}
    bootstrap_steps = 0
    wait_maintenance: dict[str, object] = {}
    wait_maintenance_steps = 0
    wait_epoch: _WaitForControlEvidence | None = None
    prime: Mapping[str, object] | None = None
    coverage: _TargetCoverageResult | None = None
    target_audit: Mapping[str, object] | None = None
    metrics: Mapping[str, object] | None = None
    preload: _StreamPreloadEvidence | None = None
    preload_audit: Mapping[str, object] | None = None
    scoring_complete = False
    failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    try:
        simulator.hello()
        simulator.reset(
            scene_xml=scene.gear_scene_xml,
            initial_qpos=initial_qpos,
            lateral_offset_m=0.0,
            yaw_offset_rad=0.0,
            log_dir=bundle.path / f"dynamic/{mode}/bootstrap-sim-logs",
        )
        bootstrap_steps = _drive_simulator_until(
            gear.start_to_wait_for_control,
            simulator,
            label=f"{mode}-wait-for-control",
            ready_for_bootstrap=lambda: gear.startup_markers_ready,
            evidence=bootstrap,
            cancellation=bootstrap_cancellation,
        )
        def prepare_input_while_publishing_low_state() -> None:
            nonlocal preload
            if mode == "file":
                gear.prepare_loaded_motion_for_scoring()
                return
            assert publisher is not None
            post_enable_fence = gear.enable_stream_for_preload()
            preload = _preload_known_good_stream(
                canonical,
                publisher,
                gear,
                post_enable_fence,
            )

        wait_maintenance_steps = _drive_simulator_until(
            prepare_input_while_publishing_low_state,
            simulator,
            label=f"{mode}-input-preparation",
            evidence=wait_maintenance,
            cancellation=bootstrap_cancellation,
        )
        if mode == "stream":
            assert preload is not None
            preload_audit = _audit_known_good_stream_preload(
                canonical, bundle, preload
            )
        wait_epoch = _audit_wait_for_control_epoch(
            gear, gear.target_motion_logfile, gear.logs_dir
        )
        gear.stop_group()
        prime = _reset_and_prime_scored_epoch(
            gear,
            simulator,
            scene_xml=scene.gear_scene_xml,
            initial_qpos=initial_qpos,
            log_dir=bundle.path / f"dynamic/{mode}/scored-sim-logs",
        )
        gear.continue_group()
        gear.activate_control()
        coverage = _drive_authoritative_target_coverage(
            gear,
            simulator,
            gear.target_motion_logfile,
            canonical,
            body_position=body_position,
            increment_steps=1,
            maximum_wall_seconds=_SCORING_TARGET_TIMEOUT_S,
            retain_reader=True,
            required_control_duration_s=required_control_duration_s,
            control_lead_rows=control_lead_rows,
        )
        prime_record = prime.get("prime")
        if not isinstance(prime_record, Mapping):
            raise ContractError("fresh LowState prime evidence is unavailable")
        prime_steps = prime_record.get("steps")
        prime_end = prime_record.get("sim_time_end_s")
        prime_state_rows = prime_record.get("state_rows")
        prime_contact_rows = prime_record.get("contact_rows")
        if (
            type(prime_steps) is not int
            or prime_steps != 1
            or type(prime_end) not in (int, float)
            or not math.isfinite(float(prime_end))
            or type(prime_state_rows) is not int
            or prime_state_rows < 0
            or type(prime_contact_rows) is not int
            or prime_contact_rows < 0
            or coverage.simulator_steps
            != prime_steps + coverage.control_drive_steps
            or abs(
                coverage.simulator_duration_s
                - (float(prime_end) + coverage.control_drive_duration_s)
            )
            > 1.0e-9
            or coverage.state_rows < prime_state_rows
            or coverage.contact_rows < prime_contact_rows
        ):
            raise ContractError("scored epoch cadence counters are inconsistent")
        gear.terminate_stopped_at_scoring_boundary()
        target_audit = _audit_closed_target_boundary(
            coverage, canonical, body_position=body_position
        )
        rows = _parse_gear_control_logs(gear.logs_dir, canonical.count)
        metrics = MappingProxyType(
            _tracking_from_gear_control_rows(canonical, rows)
        )
        scoring_complete = True
    except BaseException as error:
        failure = error
    finally:
        if coverage is not None and coverage.retained_reader is not None:
            # The normal path closes this in _audit_closed_target_boundary.
            try:
                coverage.retained_reader.close()
            except BaseException as error:
                if cleanup_failure is None:
                    cleanup_failure = error
        for resource in (gear, simulator, publisher):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                if cleanup_failure is None:
                    cleanup_failure = error
        try:
            pgid = gear.pgid
        except ProcessError:
            pgid = None
        if pgid is not None and _process_group_exists(pgid):
            cleanup_failure = ProcessError(
                f"GEAR process group {pgid} survived known-good cleanup"
            )
    if cleanup_failure is not None:
        if failure is not None:
            cleanup_failure.__cause__ = failure
        raise cleanup_failure
    if failure is not None:
        raise failure
    if not scoring_complete:
        raise AssertionError("known-good scoring exited without result or failure")
    assert (
        wait_epoch is not None
        and prime is not None
        and coverage is not None
        and target_audit is not None
        and metrics is not None
    )
    return _KnownGoodExecution(
        bootstrap_steps=bootstrap_steps,
        bootstrap=MappingProxyType(dict(bootstrap)),
        wait_maintenance_steps=wait_maintenance_steps,
        wait_maintenance=MappingProxyType(dict(wait_maintenance)),
        wait_epoch=wait_epoch,
        prime=prime,
        coverage=coverage,
        target_audit=target_audit,
        metrics=metrics,
        preload=preload,
        preload_audit=preload_audit,
    )


def _dynamic_process_outputs(
    outputs: Mapping[str, str], mode: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if mode not in ("file", "stream"):
        raise ContractError("dynamic process output mode is invalid")
    base = f"dynamic/{mode}/"
    gear_files = {
        f"{base}target.csv",
        f"{base}gear.stdout",
        f"{base}gear.stderr",
    }
    simulator_files = {
        f"{base}simulator.stdout",
        f"{base}simulator.stderr",
    }
    gear = tuple(
        name
        for name in sorted(outputs)
        if name in gear_files
        or name.startswith(f"{base}gear-logs/")
        or (
            name.startswith(f"runtime-inputs/{mode}/")
            and name.endswith(".trt")
        )
    )
    simulator = tuple(
        name
        for name in sorted(outputs)
        if name in simulator_files
        or name.startswith(f"{base}bootstrap-sim-logs/")
        or name.startswith(f"{base}scored-sim-logs/")
    )
    return gear, simulator


def _validate_gear_linkage(output: str) -> tuple[str, str]:
    if type(output) is not str or not output.strip():
        raise ContractError("GEAR runtime linkage probe returned no output")
    lowered = output.lower()
    if "not found" in lowered:
        raise ContractError("GEAR runtime linkage has an unresolved library")
    found = []
    if "libcudart" in lowered:
        found.append("libcudart")
    if "libnvinfer" in lowered:
        found.append("libnvinfer")
    if "libcudart" not in found:
        raise ContractError("GEAR runtime linkage lacks the CUDA runtime")
    if "libnvinfer" not in found:
        raise ContractError("GEAR runtime linkage lacks TensorRT")
    return tuple(found)  # type: ignore[return-value]


def _cuda_device_count(loader=ctypes.CDLL) -> int:
    try:
        driver = loader("libcuda.so.1")
    except OSError as error:
        raise ContractError("CUDA driver library is unavailable") from error
    try:
        initialize = driver.cuInit
        count_devices = driver.cuDeviceGetCount
        initialize.argtypes = [ctypes.c_uint]
        initialize.restype = ctypes.c_int
        count_devices.argtypes = [ctypes.POINTER(ctypes.c_int)]
        count_devices.restype = ctypes.c_int
        initialized = int(initialize(0))
        count = ctypes.c_int(0)
        counted = int(count_devices(ctypes.byref(count)))
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError("CUDA driver probe API is unavailable") from error
    if initialized != 0 or counted != 0:
        raise ContractError(
            "CUDA driver initialization/device enumeration failed: "
            f"cuInit={initialized}, cuDeviceGetCount={counted}"
        )
    if count.value <= 0:
        raise ContractError("CUDA driver reports no available device")
    return int(count.value)


def _read_numeric_csv(path: Path, *, columns: int | None) -> np.ndarray:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError(f"cannot decode official reference CSV: {path.name}") from error
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0]:
        raise ContractError(f"official reference CSV is empty: {path.name}")
    rows: list[list[float]] = []
    width: int | None = columns
    for row_index, line in enumerate(lines[1:]):
        tokens = line.split(",")
        if tokens and tokens[-1] == "":
            tokens.pop()
        if not tokens or any(token.strip() == "" for token in tokens):
            raise ContractError(
                f"official reference CSV row {row_index} is malformed: {path.name}"
            )
        if width is None:
            width = len(tokens)
        if len(tokens) != width:
            raise ContractError(
                f"official reference CSV width changed: {path.name}"
            )
        try:
            values = [float(token) for token in tokens]
        except ValueError as error:
            raise ContractError(
                f"official reference CSV contains an invalid number: {path.name}"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise ContractError(
                f"official reference CSV contains a non-finite value: {path.name}"
            )
        rows.append(values)
    assert width is not None
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(rows, dtype=np.float32)
    if output.ndim != 2 or output.shape[1] != width or not np.all(np.isfinite(output)):
        raise ContractError(
            f"official reference CSV exceeds finite float32: {path.name}"
        )
    return output


def _decode_known_good_reference(reference: Path) -> CanonicalTargetBuffer:
    joint_position = _read_numeric_csv(reference / "joint_pos.csv", columns=29)
    joint_velocity = _read_numeric_csv(reference / "joint_vel.csv", columns=29)
    body_quaternions = _read_numeric_csv(
        reference / "body_quat.csv", columns=None
    )
    count = int(joint_position.shape[0])
    if (
        joint_velocity.shape != (count, 29)
        or body_quaternions.shape[0] != count
        or body_quaternions.shape[1] != 4
    ):
        raise ContractError(
            "official known-good reference must use root-only body quaternions"
        )
    if count < 21 or (count - 1) % 20 != 0:
        raise ContractError(
            "official known-good reference must be frame zero plus 20-frame chunks"
        )
    return CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=body_quaternions,
        frame_index=np.arange(count, dtype=np.int64),
    )


def _decode_known_good_body_position(
    reference: Path, expected_count: int
) -> np.ndarray:
    body_positions = _read_numeric_csv(reference / "body_pos.csv", columns=None)
    if (
        body_positions.shape[0] != expected_count
        or body_positions.shape[1] != 3
    ):
        raise ContractError(
            "official known-good reference must use root-only body positions"
        )
    return np.ascontiguousarray(body_positions, dtype=np.float32)


def _require_streamable_body_position(body_position: np.ndarray) -> str:
    values = np.asarray(body_position)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.dtype != np.dtype("<f4")
        or not values.flags.c_contiguous
        or not np.all(np.isfinite(values))
    ):
        raise ContractError("known-good root body positions are invalid")
    encoded = values.tobytes(order="C")
    if encoded != bytes(len(encoded)):
        raise ContractError(
            "known-good root body positions are not bit-exact stream zeros"
        )
    return _sha256_bytes(encoded)


def _require_positive_zero_f32(values: np.ndarray, label: str) -> str:
    array = np.asarray(values)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or array.dtype != np.dtype("<f4")
        or not array.flags.c_contiguous
        or not np.all(np.isfinite(array))
    ):
        raise ContractError(f"known-good {label} values are invalid")
    encoded = array.tobytes(order="C")
    if encoded != bytes(len(encoded)):
        raise ContractError(
            f"known-good {label} values are not bit-exact positive zeros"
        )
    return _sha256_bytes(encoded)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_inverse(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, np.float64).copy()
    output[..., 1:] *= -1.0
    return output


def _quat_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = _quat_multiply(_quat_inverse(left), right)
    delta = np.where(delta[..., :1] < 0.0, -delta, delta)
    return 2.0 * np.arctan2(
        np.linalg.norm(delta[..., 1:], axis=-1),
        np.abs(delta[..., 0]),
    )


def _projection_pose_from_source_fk(
    mujoco_module: object,
    model: object,
    data: object,
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data.qpos[:] = qpos
    mujoco_module.mj_forward(model, data)
    if int(model.nbody) != 31:
        raise ContractError("joint certification requires the registered 31-body G1")
    body_ids = np.arange(1, 31, dtype=np.int64)
    global_positions = np.zeros((31, 3), np.float32)
    global_rotations = np.zeros((31, 4), np.float32)
    global_rotations[:, 0] = 1.0
    global_positions[1:] = mujoco_to_holden_vectors(data.xpos[body_ids])
    global_rotations[1:] = mujoco_to_holden_quaternions(data.xquat[body_ids])
    local_rotations = np.zeros((31, 4), np.float32)
    local_rotations[:, 0] = 1.0
    for body_id in body_ids:
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0:
            local_rotations[body_id] = global_rotations[body_id]
        else:
            local_rotations[body_id] = _quat_multiply(
                _quat_inverse(global_rotations[parent_id]),
                global_rotations[body_id],
            )
    local_velocities = np.zeros((31, 3), np.float32)
    return (
        local_rotations,
        local_velocities,
        global_positions,
        global_rotations,
    )


def _certify_joint_projection(
    source_mjcf: Path,
    projection_cli: Path,
    environment: Mapping[str, str],
    *,
    before_invoke: Callable[[], None] | None = None,
) -> tuple[dict[str, object], bytes, bytes]:
    try:
        import mujoco
    except ImportError as error:
        raise CapabilityUnavailable(
            "mujoco is unavailable for joint round-trip certification"
        ) from error
    contract = load_joint_contract(_JOINT_CONTRACT_PATH)
    try:
        model = mujoco.MjModel.from_xml_path(str(source_mjcf))
    except (OSError, ValueError) as error:
        raise ContractError(f"source MJCF does not load for certification: {error}") from error
    data = mujoco.MjData(model)
    base = np.asarray(model.qpos0, np.float64).copy()
    addresses = np.asarray(
        [row.qpos_address for row in contract.rows], dtype=np.int64
    )
    if (
        addresses.shape != (29,)
        or np.any(addresses < 0)
        or np.any(addresses >= int(model.nq))
        or len(set(int(value) for value in addresses)) != 29
    ):
        raise ContractError("joint contract qpos addresses are invalid")
    lower = np.asarray([row.lower for row in contract.rows], np.float64)
    upper = np.asarray([row.upper for row in contract.rows], np.float64)
    values = [
        np.zeros(29, np.float64),
        lower + 1.0e-5,
        upper - 1.0e-5,
        0.5 * (lower + upper),
    ]
    rng = np.random.default_rng(_JOINT_CERTIFICATION_SEED)
    values.extend(
        rng.uniform(
            lower + 1.0e-5,
            upper - 1.0e-5,
            size=(256, 29),
        )
    )
    poses: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    qpos_rows: list[np.ndarray] = []
    for row in values:
        qpos = base.copy()
        qpos[addresses] = row
        qpos_rows.append(qpos)
        poses.append(_projection_pose_from_source_fk(mujoco, model, data, qpos))
    if len(poses) != _JOINT_CERTIFICATION_SAMPLES:
        raise ContractError("joint certification sample registry changed")
    tokens = [str(len(poses))]
    for pose in poses:
        for array in pose:
            tokens.extend(format(float(value), ".17g") for value in array.flat)
    standard_input = (" ".join(tokens) + "\n").encode("ascii")
    if before_invoke is not None:
        before_invoke()
    completed = subprocess.run(
        (str(projection_cli), str(_JOINT_CONTRACT_PATH)),
        input=standard_input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(environment),
        cwd=_REPOSITORY_ROOT,
        check=False,
        timeout=90,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(
            "joint projection CLI failed with code "
            f"{completed.returncode}: {detail}"
        )
    output = np.fromstring(
        completed.stdout.decode("ascii", errors="strict"), sep=" "
    )
    values_per_pose = 29 + 29 + 29 + 3 + 4
    expected_size = 1 + len(poses) * values_per_pose
    if (
        output.size != expected_size
        or not np.all(np.isfinite(output))
        or int(output[0]) != len(poses)
    ):
        raise ContractError("joint projection CLI returned an invalid result shape")
    projected = output[1:].reshape(len(poses), values_per_pose)
    source_position = projected[:, :29]
    source_velocity = projected[:, 29:58]
    off_axis = projected[:, 58:87]
    pelvis_position = projected[:, 87:90]
    pelvis_quaternion = projected[:, 90:94]
    expected_source = np.stack(qpos_rows)[:, addresses]
    expected_positions = np.stack([pose[2][1] for pose in poses])
    expected_quaternions = np.stack([pose[3][1] for pose in poses])
    maximum_joint_error = float(
        np.max(np.abs(source_position - expected_source))
    )
    maximum_velocity_error = float(np.max(np.abs(source_velocity)))
    maximum_off_axis = float(np.max(np.abs(off_axis)))
    maximum_pelvis_position = float(
        np.max(np.abs(pelvis_position - expected_positions))
    )
    maximum_pelvis_orientation = float(
        np.max(_quat_distance(pelvis_quaternion, expected_quaternions))
    )
    expected_target = reorder_source_to_target(expected_source, contract)
    projected_target = reorder_source_to_target(source_position, contract)
    maximum_target_error = float(
        np.max(np.abs(projected_target - expected_target))
    )
    checks = {
        "maximum_joint_angle_error_rad": maximum_joint_error,
        "maximum_joint_velocity_error_rad_s": maximum_velocity_error,
        "maximum_off_axis_residual_rad": maximum_off_axis,
        "maximum_target_reorder_error_rad": maximum_target_error,
        "maximum_pelvis_position_error_m": maximum_pelvis_position,
        "maximum_pelvis_orientation_error_rad": maximum_pelvis_orientation,
    }
    if any(
        value > _JOINT_CERTIFICATION_TOLERANCE for value in checks.values()
    ):
        raise ContractError(
            "joint projection round-trip exceeded the registered 1e-4 tolerance"
        )
    return (
        {
            "sample_count": len(poses),
            "random_seed": _JOINT_CERTIFICATION_SEED,
            "input_sha256": _sha256_bytes(standard_input),
            **checks,
        },
        completed.stdout,
        completed.stderr,
    )


def _registered_scene_manifest(scene: RegisteredScene) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "scene_id": scene.scene_id,
            "route_id": scene.route_id,
            "source_kind": scene.source_kind,
            "source_hashes": dict(scene.source_hashes),
            "coordinate_source": scene.coordinate_source,
            "coordinate_target": scene.coordinate_target,
            "transform_matrix": scene.transform_matrix.tolist(),
            "output_hashes": dict(scene.output_hashes),
            "allowed_foot_geoms": list(scene.allowed_foot_geoms),
            "terrain_geoms": list(scene.terrain_geoms),
            "forbidden_geom_groups": {
                name: list(values)
                for name, values in scene.forbidden_geom_groups.items()
            },
        }
    )


@dataclass(frozen=True)
class _KnownGoodInitial:
    session_id: str

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise ContractError("known-good initial session ID is invalid")


@dataclass(frozen=True)
class _KnownGoodSource:
    session_id: str
    candidate_id: str
    predecessor_id: str | None
    timestamps_s: tuple[float, ...]


@dataclass(frozen=True)
class _KnownGoodTarget:
    accepted_chunk_id: str
    source_candidate_id: str
    buffer: CanonicalTargetBuffer
    hashes: Mapping[str, str]

    @property
    def frame_index(self) -> np.ndarray:
        return self.buffer.frame_index


@dataclass(frozen=True)
class _KnownGoodPrepared:
    target: _KnownGoodTarget
    source_candidate_id: str


class _KnownGoodValidator:
    def validate_initial(self, raw: object) -> object:
        return raw

    def validate_source(self, raw: object) -> _KnownGoodSource:
        if not isinstance(raw, _KnownGoodSource):
            raise ContractError("known-good stream source token is invalid")
        return raw


class _KnownGoodMM:
    """Coordinator transaction peer for one already-authenticated CSV buffer."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.outstanding_candidate_id: str | None = None
        self.active_candidate_id: str | None = None
        self.closed = False

    def require_alive(self) -> None:
        if self.closed:
            raise ProcessError("known-good transaction peer is closed")

    def reset(self, _config: object, *, session_id: str) -> _KnownGoodInitial:
        self.require_alive()
        self.session_id = session_id
        self.outstanding_candidate_id = None
        self.active_candidate_id = None
        return _KnownGoodInitial(session_id=session_id)

    def generate(
        self,
        _command: CommandSample,
        *,
        session_id: str,
        candidate_id: str,
        predecessor_id: str | None,
        source_intervals: int,
    ) -> _KnownGoodSource:
        self.require_alive()
        if (
            session_id != self.session_id
            or predecessor_id != self.active_candidate_id
            or self.outstanding_candidate_id is not None
            or source_intervals != 10
        ):
            raise ContractError("known-good transaction identity changed")
        self.outstanding_candidate_id = candidate_id
        return _KnownGoodSource(
            session_id=session_id,
            candidate_id=candidate_id,
            predecessor_id=predecessor_id,
            timestamps_s=tuple(index / 25.0 for index in range(11)),
        )

    def commit(self, candidate_id: str) -> None:
        if candidate_id != self.outstanding_candidate_id:
            raise ContractError("known-good commit candidate changed")
        self.outstanding_candidate_id = None
        self.active_candidate_id = candidate_id

    def abort(self, candidate_id: str) -> None:
        if candidate_id != self.outstanding_candidate_id:
            raise ContractError("known-good abort candidate changed")
        self.outstanding_candidate_id = None

    def close(self) -> None:
        self.closed = True


class _KnownGoodTimeline:
    def __init__(self, canonical: CanonicalTargetBuffer) -> None:
        if canonical.count < 21 or (canonical.count - 1) % 20 != 0:
            raise ContractError("known-good timeline has incomplete 20-frame chunks")
        self._full = canonical
        self._canonical = CanonicalTargetBuffer(
            joint_position=canonical.joint_position[:1],
            joint_velocity=canonical.joint_velocity[:1],
            body_quat_w=canonical.body_quat_w[:1],
            frame_index=canonical.frame_index[:1],
        )
        self._next_frame = 1
        self._last_candidate: str | None = None
        self._pending: _KnownGoodPrepared | None = None

    @property
    def initial_buffer(self) -> CanonicalTargetBuffer:
        return self._canonical

    @property
    def canonical_buffer(self) -> CanonicalTargetBuffer:
        return self._canonical

    @property
    def last_accepted_candidate_id(self) -> str | None:
        return self._last_candidate

    @property
    def pending_candidate_id(self) -> str | None:
        return None if self._pending is None else self._pending.source_candidate_id

    def prepare(self, source: _KnownGoodSource) -> _KnownGoodPrepared:
        if self._pending is not None:
            raise ContractError("known-good target already has a pending chunk")
        if source.predecessor_id != self._last_candidate:
            raise ContractError("known-good predecessor identity changed")
        stop = self._next_frame + 20
        if stop > self._full.count:
            raise ContractError("known-good target exhausted before a full chunk")
        buffer = CanonicalTargetBuffer(
            joint_position=self._full.joint_position[self._next_frame:stop],
            joint_velocity=self._full.joint_velocity[self._next_frame:stop],
            body_quat_w=self._full.body_quat_w[self._next_frame:stop],
            frame_index=self._full.frame_index[self._next_frame:stop],
        )
        digest = _canonical_target_sha256(buffer)
        target = _KnownGoodTarget(
            accepted_chunk_id=(
                f"{source.session_id}:known-good:{self._next_frame:020d}-"
                f"{stop - 1:020d}"
            ),
            source_candidate_id=source.candidate_id,
            buffer=buffer,
            hashes=MappingProxyType(
                {
                    "canonical_target_sha256": digest,
                    "diagnostic_sha256": digest,
                }
            ),
        )
        self._pending = _KnownGoodPrepared(
            target=target,
            source_candidate_id=source.candidate_id,
        )
        return self._pending

    def commit(self, prepared: _KnownGoodPrepared) -> _KnownGoodTarget:
        if prepared is not self._pending:
            raise ContractError("known-good commit object changed")
        stop = int(prepared.target.buffer.frame_index[-1]) + 1
        self._canonical = CanonicalTargetBuffer(
            joint_position=self._full.joint_position[:stop],
            joint_velocity=self._full.joint_velocity[:stop],
            body_quat_w=self._full.body_quat_w[:stop],
            frame_index=self._full.frame_index[:stop],
        )
        self._next_frame = stop
        self._last_candidate = prepared.source_candidate_id
        self._pending = None
        return prepared.target

    def abort(self, candidate_id: str) -> None:
        if self._pending is None or self._pending.source_candidate_id != candidate_id:
            raise ContractError("known-good abort candidate changed")
        self._pending = None


def _drive_simulator_until(
    operation,
    simulator: GatedSimulatorClient,
    *,
    label: str,
    ready_for_bootstrap=None,
    cold_start_maximum_seconds: float = _COLD_GEAR_STARTUP_TIMEOUT_S,
    maximum_seconds: float = _LOW_STATE_BOOTSTRAP_TIMEOUT_S,
    evidence: dict[str, object] | None = None,
    cancellation: threading.Event | None = None,
) -> int:
    """Wait out cold loading, then advance bounded non-scored bootstrap physics."""

    if (
        type(cold_start_maximum_seconds) not in (int, float)
        or not math.isfinite(float(cold_start_maximum_seconds))
        or cold_start_maximum_seconds <= 0.0
        or type(maximum_seconds) not in (int, float)
        or not math.isfinite(float(maximum_seconds))
        or maximum_seconds <= 0.0
    ):
        raise ValueError("bootstrap time bounds must be positive and finite")

    failures: list[BaseException] = []
    cold_started_ns = time.monotonic_ns()

    def invoke() -> None:
        try:
            operation()
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(
        target=invoke,
        name=f"gear-{label}-bootstrap",
        daemon=False,
    )
    worker.start()

    def cancel_and_join() -> None:
        if cancellation is not None:
            cancellation.set()
        worker.join()

    cold_deadline = time.monotonic() + cold_start_maximum_seconds
    try:
        while worker.is_alive() and ready_for_bootstrap is not None:
            if bool(ready_for_bootstrap()):
                break
            if time.monotonic() >= cold_deadline:
                cancel_and_join()
                raise ProcessError(
                    f"timed out during GEAR {label} cold startup"
                )
            worker.join(
                timeout=min(
                    0.05, max(0.0, cold_deadline - time.monotonic())
                )
            )
    except BaseException:
        if worker.is_alive():
            cancel_and_join()
        raise
    if not worker.is_alive():
        if failures:
            raise failures[0]
        if evidence is not None:
            cold_finished_ns = time.monotonic_ns()
            evidence.update(
                {
                    "cold_start_bound_s": float(cold_start_maximum_seconds),
                    "cold_start_elapsed_s": (
                        cold_finished_ns - cold_started_ns
                    )
                    / 1.0e9,
                    "low_state_bound_s": float(maximum_seconds),
                    "low_state_elapsed_s": 0.0,
                    "simulator_steps": 0,
                }
            )
        return 0

    cold_finished_ns = time.monotonic_ns()
    deadline = time.monotonic() + maximum_seconds
    bootstrap_started_ns = time.monotonic_ns()
    advanced = 0
    maximum_steps = max(1, int(math.ceil(maximum_seconds / simulator.sim_dt)))
    try:
        while (
            worker.is_alive()
            and time.monotonic() < deadline
            and advanced < maximum_steps
        ):
            batch = min(10, maximum_steps - advanced)
            simulator.advance(batch)
            advanced += batch
            worker.join(timeout=0.0)
    except BaseException:
        cancel_and_join()
        raise
    worker.join(timeout=0.05)
    if worker.is_alive():
        cancel_and_join()
        raise ProcessError(f"timed out during GEAR {label} bootstrap")
    if failures:
        raise failures[0]
    if evidence is not None:
        bootstrap_finished_ns = time.monotonic_ns()
        evidence.update(
            {
                "cold_start_bound_s": float(cold_start_maximum_seconds),
                "cold_start_elapsed_s": (
                    cold_finished_ns - cold_started_ns
                )
                / 1.0e9,
                "low_state_bound_s": float(maximum_seconds),
                "low_state_elapsed_s": (
                    bootstrap_finished_ns - bootstrap_started_ns
                )
                / 1.0e9,
                "simulator_steps": advanced,
            }
        )
    return advanced


def _exact_steps(duration_s: float, sim_dt_s: float) -> int:
    ratio = duration_s / sim_dt_s
    steps = round(ratio)
    if steps <= 0 or abs(ratio - steps) > 1.0e-12:
        raise ContractError("registered duration is not an exact simulator step count")
    return int(steps)


def _model_initial_state(
    scene: RegisteredScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import mujoco
    except ImportError as error:
        raise CapabilityUnavailable("mujoco is unavailable") from error
    try:
        model = mujoco.MjModel.from_xml_path(str(scene.gear_scene_xml))
    except (OSError, ValueError) as error:
        raise ContractError(f"registered flat scene does not reload: {error}") from error
    addresses: list[int] = []
    from .joints import TARGET_JOINT_ORDER

    for name in TARGET_JOINT_ORDER:
        try:
            joint = model.joint(name)
        except KeyError as error:
            raise ContractError(f"registered model is missing joint {name}") from error
        addresses.append(int(model.jnt_qposadr[int(joint.id)]))
    if len(set(addresses)) != 29:
        raise ContractError("registered model joint qpos addresses are not unique")
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    try:
        pelvis_quaternion = np.asarray(data.body("pelvis").xquat, np.float64).copy()
    except KeyError as error:
        raise ContractError("registered model has no pelvis body") from error
    return (
        np.asarray(model.qpos0, np.float64).copy(),
        np.asarray(addresses, np.int64),
        pelvis_quaternion,
    )


class DefaultStageAOperations:
    """Production adapter over the pinned authenticated CPU/external seams."""

    def __init__(self) -> None:
        self._runtime: dict[int, dict[str, object]] = {}

    def _state(self, context: StageAContext) -> dict[str, object]:
        try:
            return self._runtime[id(context)]
        except KeyError as error:
            raise ContractError("Stage A runtime context is unavailable") from error

    def _inputs(self, context: StageAContext) -> ExternalInputs:
        verified = self._state(context).get("verified_external")
        inputs = getattr(verified, "inputs", None)
        if not isinstance(inputs, ExternalInputs):
            raise ContractError("verified external inputs are unavailable")
        return inputs

    def authenticate(
        self, request: StageARequest, inputs: ExternalInputs
    ) -> StageAContext:
        pointers = _find_unresolved_git_lfs_capabilities(inputs)
        if pointers:
            raise CapabilityUnavailable(
                "unresolved Git-LFS capability: "
                + ", ".join(str(path) for path in pointers[:4])
            )
        verified = verify_external(inputs, _LOCK_PATH, scored=True)
        official_scene = inputs.gear_checkout / _GEAR_MODEL_RELATIVE
        official_robot = inputs.gear_checkout / _GEAR_ROBOT_RELATIVE
        official_scene_sha256 = _sha256_file(official_scene)
        official_robot_sha256 = _sha256_file(official_robot)
        if official_scene_sha256 != _GEAR_MODEL_SHA256:
            raise ExternalInputError("pinned official GEAR scene XML hash changed")
        if official_robot_sha256 != _GEAR_ROBOT_SHA256:
            raise ExternalInputError("pinned official GEAR robot XML hash changed")
        repository_commit = _git_value(
            _REPOSITORY_ROOT, "rev-parse", "--verify", "HEAD"
        )
        repository_dirty = bool(
            _git_value(
                _REPOSITORY_ROOT,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
        )
        artifact_hashes: dict[str, str | None] = {
            "policy": verified.hashes["policy"],
            "encoder": verified.hashes.get("encoder"),
            "observation_config": verified.hashes["observation_config"],
            "model": official_scene_sha256,
            "source_mjcf": verified.hashes["source_mjcf"],
            "motion": _sha256_bytes(
                b"".join(
                    bytes.fromhex(value)
                    for key, value in verified.hashes.items()
                    if key.startswith("gear:known_good_reference/")
                )
            ),
            "terrain": verified.hashes["terrain_dir"],
            "joint_map": _sha256_file(_JOINT_CONTRACT_PATH),
            "scene": None,
        }
        external_hashes = dict(verified.hashes)
        external_hashes.update(
            {
                "gear:official_scene_xml": official_scene_sha256,
                "gear:official_robot_xml": official_robot_sha256,
            }
        )
        context = StageAContext(
            gear_commit=verified.gear_commit,
            gear_dirty=False,
            external_hashes=MappingProxyType(dict(sorted(external_hashes.items()))),
            motion_matching_commit=repository_commit,
            motion_matching_dirty=repository_dirty,
            artifact_hashes=MappingProxyType(artifact_hashes),
            known_good_reference=verified.known_good_reference,
            processes=(),
        )
        self._runtime[id(context)] = {
            "verified_external": verified,
            "official_scene": official_scene,
            "official_robot": official_robot,
        }
        return context

    def run_gate(
        self,
        name: str,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult:
        state = self._state(context)
        attempt = {
            "name": name,
            "before": _snapshot_outputs(bundle),
            "processes": [],
        }
        state["active_gate_attempt"] = attempt
        handler = getattr(self, f"_{name}")
        try:
            return handler(request, context, bundle)
        except BaseException as error:
            if isinstance(error, CapabilityUnavailable):
                status = "not_run"
            elif isinstance(error, ScientificGateFailure):
                status = "scientific_failure"
            else:
                status = "integration_failure"
            outputs = _new_outputs(bundle, attempt["before"])
            processes = [dict(value) for value in attempt["processes"]]
            if name in (
                "known_good_file_dynamic",
                "known_good_stream_delivery",
                "stage_b_dynamic",
            ):
                mode = (
                    "file"
                    if name == "known_good_file_dynamic"
                    else "stream"
                )
                gear_outputs, simulator_outputs = _dynamic_process_outputs(
                    _snapshot_outputs(bundle), mode
                )
                for process in processes:
                    process_name = str(process.get("name", ""))
                    if process_name.startswith("g1_deploy_onnx_ref"):
                        process["outputs"] = list(gear_outputs)
                    elif process_name.startswith("gated-simulator"):
                        process["outputs"] = list(simulator_outputs)
            return GateResult(
                status=status,
                reason=f"{type(error).__name__}: {error}",
                outputs=outputs,
                processes=tuple(
                    MappingProxyType(process) for process in processes
                ),
            )
        finally:
            state.pop("active_gate_attempt", None)

    def _register_process_attempt(
        self,
        context: StageAContext,
        process: Mapping[str, object],
    ) -> None:
        attempt = self._state(context).get("active_gate_attempt")
        if not isinstance(attempt, dict):
            raise ContractError("process attempt lacks an active Stage A gate")
        processes = attempt.get("processes")
        if not isinstance(processes, list):
            raise ContractError("process attempt registry is invalid")
        processes.append(MappingProxyType(dict(process)))

    def _external_identity(
        self, request: StageARequest, context: StageAContext, bundle: RunBundle
    ) -> GateResult:
        encoder = context.artifact_hashes["encoder"]
        identity = {
            "policy": str(context.artifact_hashes["policy"]),
            "encoder": (
                str(encoder)
                if encoder is not None
                else _sha256_bytes(b"mm-sonic-encoder/absent\n")
            ),
            "observation_config": str(
                context.artifact_hashes["observation_config"]
            ),
            "external_commit": context.gear_commit,
        }
        payload = {
            "gear_commit": context.gear_commit,
            "external_hashes": dict(context.external_hashes),
        }
        data = _canonical_json_bytes(payload, "external identity")
        relative = "gates/external_identity.json"
        bundle.write_bytes(relative, data)
        digest = _sha256_bytes(data)
        return GateResult(
            status="pass",
            identity=MappingProxyType(identity),
            evidence_hashes=MappingProxyType(
                {"external_identity_sha256": digest}
            ),
            outputs=MappingProxyType({relative: digest}),
        )

    def _joint_projection_round_trip(
        self, request: StageARequest, context: StageAContext, bundle: RunBundle
    ) -> GateResult:
        before = _snapshot_outputs(bundle)
        inputs = self._inputs(context)
        source = inputs.source_mjcf
        target_order = inputs.gear_checkout / _GEAR_TARGET_ORDER_RELATIVE
        generated = generate_joint_contract(source, target_order)
        generated_bytes = contract_json_bytes(generated)
        committed = _JOINT_CONTRACT_PATH.read_bytes()
        if generated_bytes != committed:
            raise ContractError(
                "generated G1 joint contract differs from the registered contract"
            )
        projection = _SONIC_ROOT / "build/g1_project_pose_cli"
        if not projection.is_file() or not os.access(projection, os.X_OK):
            raise CapabilityUnavailable(
                "registered G1 projection CLI is unavailable"
            )
        environment = _child_environment(request)
        stdout_relative = "gates/joint_projection.stdout"
        stderr_relative = "gates/joint_projection.stderr"
        process = _process_record(
            "g1_project_pose_cli",
            (str(projection), str(_JOINT_CONTRACT_PATH)),
            executable=projection,
            environment=environment,
            outputs=(stdout_relative, stderr_relative),
        )
        try:
            certification, standard_output, standard_error = (
                _certify_joint_projection(
                    source,
                    projection,
                    environment,
                    before_invoke=lambda: self._register_process_attempt(
                        context, process
                    ),
                )
            )
        except subprocess.TimeoutExpired as error:
            standard_output, standard_error = _timeout_evidence_bytes(error)
            bundle.write_bytes(stdout_relative, standard_output)
            bundle.write_bytes(stderr_relative, standard_error)
            raise ContractError(
                "joint projection certification timed out"
            ) from error
        bundle.write_bytes(stdout_relative, standard_output)
        bundle.write_bytes(stderr_relative, standard_error)
        relative = "gates/joint_projection_round_trip.json"
        payload = _canonical_json_bytes(
            {
                "joint_contract_sha256": _sha256_bytes(committed),
                "projection_cli_sha256": _sha256_file(projection),
                "source_mjcf_sha256": generated.source_mjcf_sha256,
                "target_order_source_sha256": generated.target_order_source_sha256,
                "certification": certification,
            },
            "joint projection gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        return GateResult(
            status="pass",
            evidence_hashes=MappingProxyType(
                {"joint_projection_round_trip_sha256": digest}
            ),
            outputs=_new_outputs(bundle, before),
            processes=(process,),
        )

    def _basis_and_scene_alignment(
        self, request: StageARequest, context: StageAContext, bundle: RunBundle
    ) -> GateResult:
        before = _snapshot_outputs(bundle)
        axes = np.eye(3, dtype=np.float64)
        expected = np.array(
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            dtype=np.float32,
        )
        if not np.array_equal(holden_to_mujoco_vectors(axes), expected):
            raise ContractError("Holden/MuJoCo basis fixture changed")
        converted_quat = holden_to_mujoco_quaternions(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        )
        if not np.array_equal(
            converted_quat,
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        ):
            raise ContractError("root quaternion basis fixture changed")

        inputs = self._inputs(context)
        pointers = _find_unresolved_git_lfs_capabilities(inputs)
        if pointers:
            raise CapabilityUnavailable(
                "unresolved Git-LFS capability: "
                + ", ".join(str(path) for path in pointers[:4])
            )
        gear_binary = inputs.gear_checkout / _GEAR_BINARY_RELATIVE
        if not gear_binary.is_file() or not os.access(gear_binary, os.X_OK):
            raise CapabilityUnavailable(
                "official g1_deploy_onnx_ref executable is unavailable"
            )
        mm_server = _SONIC_ROOT / "build/mm_chunk_server"
        if not mm_server.is_file() or not os.access(mm_server, os.X_OK):
            raise CapabilityUnavailable("registered MM chunk server is unavailable")

        environment = _child_environment(request)
        ldd_path = shutil.which("ldd", path=environment.get("PATH"))
        if ldd_path is None:
            raise CapabilityUnavailable("ldd is unavailable for GEAR linkage probe")
        ldd_executable = Path(ldd_path).resolve(strict=True)
        linkage_argv = (str(ldd_executable), str(gear_binary))
        linkage_stdout = "preflight/gear-linkage.stdout"
        linkage_stderr = "preflight/gear-linkage.stderr"
        linkage_process = _process_record(
            "gear-cuda-tensorrt-linkage-preflight",
            linkage_argv,
            executable=ldd_executable,
            environment=environment,
            outputs=(linkage_stdout, linkage_stderr),
        )
        self._register_process_attempt(context, linkage_process)
        try:
            linkage_check = subprocess.run(
                linkage_argv,
                cwd=_REPOSITORY_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
        except subprocess.TimeoutExpired as error:
            standard_output, standard_error = _timeout_evidence_bytes(error)
            bundle.write_bytes(linkage_stdout, standard_output)
            bundle.write_bytes(linkage_stderr, standard_error)
            raise CapabilityUnavailable("GEAR linkage probe timed out") from error
        bundle.write_bytes(linkage_stdout, linkage_check.stdout)
        bundle.write_bytes(linkage_stderr, linkage_check.stderr)
        runtime_error: str | None = None
        linked_libraries: tuple[str, str] | tuple[()] = ()
        cuda_devices = 0
        if linkage_check.returncode != 0:
            runtime_error = (
                "GEAR linkage probe exited "
                f"{linkage_check.returncode}"
            )
        else:
            try:
                linked_libraries = _validate_gear_linkage(
                    (linkage_check.stdout + linkage_check.stderr).decode(
                        "utf-8", errors="replace"
                    )
                )
                cuda_devices = _cuda_device_count()
            except ContractError as error:
                runtime_error = str(error)
        capability_relative = "preflight/gear-runtime-capability.json"
        capability_payload = _canonical_json_bytes(
            {
                "cuda_device_count": cuda_devices,
                "linked_libraries": list(linked_libraries),
                "passed": runtime_error is None,
                "reason": runtime_error,
            },
            "GEAR runtime capability",
        )
        bundle.write_bytes(capability_relative, capability_payload)
        if runtime_error is not None:
            return GateResult(
                status="not_run",
                reason="GEAR CUDA/TensorRT capability is unavailable: "
                + runtime_error,
                outputs=_new_outputs(bundle, before),
                processes=(linkage_process,),
            )
        import_argv = (
            sys.executable,
            "-u",
            "-B",
            "-m",
            "mm_sonic.gated_sim",
            "--gear-checkout",
            str(inputs.gear_checkout),
            "--import-preflight",
        )
        import_stdout = "preflight/gated-sim-import.stdout"
        import_stderr = "preflight/gated-sim-import.stderr"
        import_process = _process_record(
            "gated-simulator-import-preflight",
            import_argv,
            executable=Path(sys.executable).resolve(strict=True),
            environment=environment,
            outputs=(import_stdout, import_stderr),
            module=Path(__file__).with_name("gated_sim.py"),
        )
        self._register_process_attempt(context, import_process)
        try:
            import_check = subprocess.run(
                import_argv,
                cwd=_REPOSITORY_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=90,
            )
        except subprocess.TimeoutExpired as error:
            standard_output, standard_error = _timeout_evidence_bytes(error)
            bundle.write_bytes(import_stdout, standard_output)
            bundle.write_bytes(import_stderr, standard_error)
            raise CapabilityUnavailable(
                "gated simulator import preflight timed out"
            ) from error
        bundle.write_bytes(import_stdout, import_check.stdout)
        bundle.write_bytes(import_stderr, import_check.stderr)
        if import_check.returncode != 0:
            reason = import_check.stderr.decode(
                "utf-8", errors="replace"
            ).strip()
            return GateResult(
                status="not_run",
                reason=(
                    "gated simulator import capability is unavailable"
                    + (f": {reason}" if reason else "")
                ),
                outputs=_new_outputs(bundle, before),
                processes=(linkage_process, import_process),
            )

        mm_environment = _child_environment(
            request,
            SONIC_TERRAIN_DIR=str(inputs.terrain_dir),
        )
        mm_stdout = bundle.path / "preflight/mm.stdout"
        mm_stderr = bundle.path / "preflight/mm.stderr"
        session_id = f"stage-a-preflight-{uuid.uuid4().hex}"
        mm_process = _process_record(
            "mm_chunk_server-preflight",
            (str(mm_server),),
            executable=mm_server,
            environment=mm_environment,
            outputs=(
                mm_stdout.relative_to(bundle.path).as_posix(),
                mm_stderr.relative_to(bundle.path).as_posix(),
            ),
        )
        self._register_process_attempt(context, mm_process)
        with MMChunkClient(
            run_root=bundle.path,
            command=(str(mm_server),),
            stdout_archive=mm_stdout,
            stderr_archive=mm_stderr,
            env=mm_environment,
            cwd=_REPOSITORY_ROOT,
        ) as client:
            hello = client.hello()
            joint_feasibility = _joint_feasibility_from_hello(hello)
            reset = client.reset(
                SessionConfig(
                    scene_id=FLAT_SCENE_ID,
                    route_id=_FLAT_ROUTE_ID,
                    terrain_weight=0.0,
                ),
                session_id=session_id,
            )
        scene = register_scene(
            FLAT_SCENE_ID,
            _FLAT_ROUTE_ID,
            registry_path=_SCENE_REGISTRY_PATH,
            terrain_dir=inputs.terrain_dir,
            official_scene_xml=inputs.gear_checkout / _GEAR_MODEL_RELATIVE,
            output_dir=bundle.path / "scene",
            expected_official_scene_sha256=_GEAR_MODEL_SHA256,
            expected_robot_sha256=_GEAR_ROBOT_SHA256,
            mm_hello_identity=hello,
            mm_scene_identity=reset["scene"],
        )
        initial_qpos, joint_addresses, initial_pelvis = _model_initial_state(scene)
        initial_qpos_sha256 = _sha256_bytes(
            np.asarray(initial_qpos, dtype="<f8").tobytes(order="C")
        )
        scene_registration = _registered_scene_manifest(scene)
        state = self._state(context)
        state.update(
            {
                "scene": scene,
                "scene_registration": scene_registration,
                "initial_qpos": initial_qpos,
                "joint_addresses": joint_addresses,
                "initial_pelvis_quaternion": initial_pelvis,
                "mm_hello": hello,
                "mm_scene": reset["scene"],
                "joint_feasibility": joint_feasibility,
            }
        )
        relative = "gates/basis_and_scene_alignment.json"
        payload = _canonical_json_bytes(
            {
                "basis_fixture": HOLDEN_TO_MUJOCO_MATRIX.tolist(),
                "gear_runtime_capability": {
                    "cuda_device_count": cuda_devices,
                    "linked_libraries": list(linked_libraries),
                },
                "official_model_xml_sha256": _GEAR_MODEL_SHA256,
                "generated_flat_scene_sha256": scene.output_hashes[
                    "gear_scene_xml"
                ],
                "initial_qpos_sha256": initial_qpos_sha256,
                "joint_feasibility": _joint_feasibility_payload(
                    joint_feasibility
                ),
                "scene_registration_sha256": scene.output_hashes[
                    "scene_registration"
                ],
            },
            "basis and scene alignment gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        return GateResult(
            status="pass",
            identity=MappingProxyType(
                {
                    "official_model_xml": _GEAR_MODEL_SHA256,
                    "generated_flat_scene": scene.output_hashes[
                        "gear_scene_xml"
                    ],
                    "initial_qpos": initial_qpos_sha256,
                }
            ),
            evidence_hashes=MappingProxyType(
                {"basis_and_scene_alignment_sha256": digest}
            ),
            metrics=MappingProxyType(
                {
                    "joint_feasibility": _joint_feasibility_payload(
                        joint_feasibility
                    )
                }
            ),
            outputs=_new_outputs(bundle, before),
            scene_registration=scene_registration,
            artifact_hashes=MappingProxyType(
                {
                    "model": _GEAR_MODEL_SHA256,
                    "scene": scene.output_hashes["gear_scene_xml"],
                }
            ),
            processes=(linkage_process, import_process, mm_process),
        )

    def _flat_mm_kinematic_replay(
        self,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult:
        before = _snapshot_outputs(bundle)
        state = self._state(context)
        scene = state.get("scene")
        if not isinstance(scene, RegisteredScene):
            raise ContractError("flat MM replay requires the registered scene")
        expected_joint_feasibility = state.get("joint_feasibility")
        if not isinstance(
            expected_joint_feasibility, JointFeasibilityIdentity
        ):
            raise ContractError(
                "flat MM replay requires preflight joint feasibility identity"
            )
        mm_server = _SONIC_ROOT / "build/mm_chunk_server"
        if not mm_server.is_file() or not os.access(mm_server, os.X_OK):
            raise CapabilityUnavailable("registered MM chunk server is unavailable")
        contract = load_joint_contract(_JOINT_CONTRACT_PATH)
        validator = SourceValidator(contract)
        commands = flat_command_script()
        if len(commands) != 30:
            raise ContractError("flat command script must contain exactly 30 chunks")
        script = command_script_bytes(
            scene_id=FLAT_SCENE_ID,
            route_id=_FLAT_ROUTE_ID,
            commands=commands,
        )
        script_relative = "reference/flat-command-script.json"
        bundle.write_bytes(script_relative, script)
        environment = _child_environment(
            request,
            SONIC_TERRAIN_DIR=str(self._inputs(context).terrain_dir),
        )
        mm_stdout = bundle.path / "reference/mm.stdout"
        mm_stderr = bundle.path / "reference/mm.stderr"
        session_id = f"stage-a-mm-reference-{uuid.uuid4().hex}"
        physical_positions: list[np.ndarray] = []
        virtual_positions: list[np.ndarray] = []
        virtual_quaternions: list[np.ndarray] = []
        scientific_reason: str | None = None
        failed_chunk: int | None = None
        failure_boundary: str | None = None
        timeline: TargetTimeline | None = None
        process = _process_record(
            "mm_chunk_server-flat-reference",
            (str(mm_server),),
            executable=mm_server,
            environment=environment,
            outputs=(
                mm_stdout.relative_to(bundle.path).as_posix(),
                mm_stderr.relative_to(bundle.path).as_posix(),
            ),
        )
        self._register_process_attempt(context, process)
        with MMChunkClient(
            run_root=bundle.path,
            command=(str(mm_server),),
            stdout_archive=mm_stdout,
            stderr_archive=mm_stderr,
            env=environment,
            cwd=_REPOSITORY_ROOT,
        ) as client:
            hello = client.hello()
            run_joint_feasibility = _joint_feasibility_from_hello(hello)
            if run_joint_feasibility != expected_joint_feasibility:
                raise ContractError(
                    "MM joint feasibility identity changed between "
                    "preflight and reference run"
                )
            reset = client.reset(
                SessionConfig(
                    scene_id=FLAT_SCENE_ID,
                    route_id=_FLAT_ROUTE_ID,
                    terrain_weight=0.0,
                ),
                session_id=session_id,
            )
            verify_mm_scene_identity(scene, hello, reset["scene"])
            initial = validator.validate_initial(reset)
            try:
                timeline = TargetTimeline(initial, contract)
            except ContractError as error:
                if "joint limit" not in str(error).lower():
                    raise
                scientific_reason = (
                    "flat MM initial boundary violated the registered joint "
                    f"limits before chunk 0: {error}"
                )
                failed_chunk = 0
                failure_boundary = "initial_boundary"
            if timeline is not None:
                physical_positions.append(
                    holden_to_mujoco_vectors(
                        initial.physical_pelvis_position_holden
                    )
                )
                virtual_positions.append(
                    holden_to_mujoco_vectors(
                        initial.virtual_root_position_holden
                    )
                )
                virtual_quaternions.append(
                    holden_to_mujoco_quaternions(
                        initial.virtual_root_orientation_holden
                    )
                )
                for command in commands:
                    candidate_id = (
                        f"{session_id}:candidate:{command.chunk_index:06d}"
                    )
                    predecessor_id = timeline.last_accepted_candidate_id
                    try:
                        raw = client.generate(
                            command,
                            session_id=session_id,
                            candidate_id=candidate_id,
                            predecessor_id=predecessor_id,
                            source_intervals=10,
                        )
                        source = validator.validate_source(raw)
                        prepared = timeline.prepare(source)
                    except BaseException as error:
                        if client.outstanding_candidate_id == candidate_id:
                            client.abort(candidate_id)
                        no_safe_candidate = _is_remote_no_safe_candidate(
                            error
                        )
                        no_inertialized_safe_candidate = (
                            _is_remote_no_inertialized_safe_candidate(error)
                        )
                        if (
                            isinstance(error, ContractError)
                            and "joint limit" in str(error).lower()
                        ) or _is_remote_registered_joint_limit(
                            error
                        ) or no_safe_candidate or no_inertialized_safe_candidate:
                            if no_inertialized_safe_candidate:
                                scientific_reason = (
                                    "flat MM reference found no authenticated "
                                    "inertialized-joint-safe database candidate at "
                                    f"chunk {command.chunk_index}: {error}"
                                )
                            elif no_safe_candidate:
                                scientific_reason = (
                                    "flat MM reference found no authenticated "
                                    "joint-limit-safe database candidate at chunk "
                                    f"{command.chunk_index}: {error}"
                                )
                            else:
                                scientific_reason = (
                                    "flat MM reference violated the registered joint "
                                    f"limits at chunk {command.chunk_index}: {error}"
                                )
                            failed_chunk = command.chunk_index
                            failure_boundary = "source_chunk"
                            break
                        raise
                    client.commit(candidate_id)
                    target = timeline.commit(prepared)
                    physical_positions.extend(target.physical_pelvis_position)
                    virtual_positions.extend(target.virtual_root_position)
                    virtual_quaternions.extend(target.virtual_root_quat_w)
        canonical = None if timeline is None else timeline.canonical_buffer
        if scientific_reason is not None:
            relative = "gates/flat_mm_kinematic_replay.json"
            payload = _canonical_json_bytes(
                {
                    "status": "scientific_failure",
                    "reason": scientific_reason,
                    "failed_chunk": failed_chunk,
                    "failure_boundary": failure_boundary,
                    "initial_boundary_accepted": timeline is not None,
                    "completed_chunks": (
                        0 if canonical is None else (canonical.count - 1) // 20
                    ),
                    "partial_frame_count": (
                        0 if canonical is None else canonical.count
                    ),
                    "required_chunks": len(commands),
                    "required_frame_count": 601,
                },
                "flat MM scientific failure",
            )
            bundle.write_bytes(relative, payload)
            digest = _sha256_bytes(payload)
            return GateResult(
                status="scientific_failure",
                reason=scientific_reason,
                evidence_hashes=MappingProxyType(
                    {"flat_mm_kinematic_replay_sha256": digest}
                ),
                outputs=_new_outputs(bundle, before),
                processes=(process,),
            )
        assert canonical is not None
        if canonical.count != 601:
            raise ContractError(
                f"full flat MM reference must contain 601 frames, got {canonical.count}"
            )
        diagnostics = ReferenceDiagnostics(
            physical_pelvis_position=np.asarray(
                physical_positions, dtype=np.float32
            ),
            virtual_root_position=np.asarray(
                virtual_positions, dtype=np.float32
            ),
            virtual_root_quat_w=np.asarray(
                virtual_quaternions, dtype=np.float32
            ),
        )
        written = write_reference_bundle(
            bundle,
            canonical,
            diagnostics,
            source_sha256=_sha256_bytes(script),
            scene_id=FLAT_SCENE_ID,
            route_id=_FLAT_ROUTE_ID,
        )
        replay = replay_kinematic_reference(
            scene,
            tuple(
                row.target_name
                for row in sorted(contract.rows, key=lambda row: row.target_index)
            ),
            canonical.joint_position,
            diagnostics.physical_pelvis_position,
            canonical.body_quat_w,
        )
        replay_reason = (
            "flat MM kinematic replay has forbidden penetration above 5 mm"
            if replay.forbidden_penetration
            else None
        )
        state.update(
            {
                "mm_reference": canonical,
                "mm_diagnostics": diagnostics,
            }
        )
        relative = "gates/flat_mm_kinematic_replay.json"
        payload = _canonical_json_bytes(
            {
                "status": (
                    "scientific_failure" if replay_reason is not None else "pass"
                ),
                "reason": replay_reason,
                "command_chunks": len(commands),
                "frame_count": canonical.count,
                "duration_s": 12.0,
                "canonical_target_sha256": written.canonical_target_sha256,
                "diagnostic_sha256": written.diagnostic_sha256,
                "allowed_foot_contacts": replay.allowed_foot_contacts,
                "maximum_forbidden_penetration_m": (
                    replay.maximum_forbidden_penetration_m
                ),
                "forbidden_penetration": replay.forbidden_penetration,
            },
            "flat MM kinematic replay gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        if replay_reason is not None:
            return GateResult(
                status="scientific_failure",
                reason=replay_reason,
                evidence_hashes=MappingProxyType(
                    {"flat_mm_kinematic_replay_sha256": digest}
                ),
                outputs=_new_outputs(bundle, before),
                processes=(process,),
            )
        return GateResult(
            status="pass",
            identity=MappingProxyType(
                {"mm_reference_buffer": written.canonical_target_sha256}
            ),
            evidence_hashes=MappingProxyType(
                {"flat_mm_kinematic_replay_sha256": digest}
            ),
            outputs=_new_outputs(bundle, before),
            processes=(process,),
        )

    def _materialize_known_good_projection(
        self,
        context: StageAContext,
        bundle: RunBundle,
    ) -> _KnownGoodProjection:
        base = bundle.path / "known-good/reference-base"
        leaf = base / _KNOWN_GOOD_LEAF
        source_hashes: dict[str, str] = {}
        for name in KNOWN_GOOD_REFERENCE_FILES:
            source = context.known_good_reference / name
            key = f"gear:known_good_reference/{name}"
            try:
                expected = context.external_hashes[key]
            except KeyError as error:
                raise ContractError(
                    f"authenticated known-good hash is unavailable: {name}"
                ) from error
            if _sha256_file(source) != expected:
                raise ExternalInputError(
                    f"authenticated known-good file changed: {name}"
                )
            source_hashes[key] = expected

        try:
            info = (context.known_good_reference / "info.txt").read_text("utf-8")
            metadata = (context.known_good_reference / "metadata.txt").read_text(
                "utf-8"
            )
        except (OSError, UnicodeDecodeError) as error:
            raise ExternalInputError(
                "cannot read authenticated known-good metadata"
            ) from error
        required_info = (
            "Shape: (455, 29)",
            "Shape: (455, 14, 3)",
            "Shape: (455, 14, 4)",
            "Shape: (14,)",
            "Sample: [455]",
        )
        if any(token not in info for token in required_info):
            raise ContractError(
                "official known-good info does not declare the registered 455x14 layout"
            )
        if (
            "Total timesteps: 455" not in metadata
            or "[ 0  4 10 18  5 11 19  9 16 22 28 17 23 29]"
            not in metadata
        ):
            raise ContractError(
                "official known-good metadata does not declare the registered bodies"
            )

        source = context.known_good_reference
        joint_position = _read_numeric_csv(source / "joint_pos.csv", columns=29)
        joint_velocity = _read_numeric_csv(source / "joint_vel.csv", columns=29)
        body_position = _read_numeric_csv(source / "body_pos.csv", columns=42)
        body_quaternion = _read_numeric_csv(source / "body_quat.csv", columns=56)
        body_linear_velocity = _read_numeric_csv(
            source / "body_lin_vel.csv", columns=42
        )
        body_angular_velocity = _read_numeric_csv(
            source / "body_ang_vel.csv", columns=42
        )
        arrays = (
            joint_position,
            joint_velocity,
            body_position,
            body_quaternion,
            body_linear_velocity,
            body_angular_velocity,
        )
        if any(array.shape[0] != _KNOWN_GOOD_SOURCE_FRAME_COUNT for array in arrays):
            raise ContractError(
                "official known-good CSVs must contain exactly 455 frames"
            )
        for name in KNOWN_GOOD_REFERENCE_FILES:
            key = f"gear:known_good_reference/{name}"
            if _sha256_file(source / name) != source_hashes[key]:
                raise ExternalInputError(
                    f"authenticated known-good file changed while decoding: {name}"
                )

        selected = slice(0, _KNOWN_GOOD_PROJECTED_FRAME_COUNT)
        projected_joint_position = np.ascontiguousarray(
            joint_position[selected], dtype="<f4"
        )
        projected_joint_velocity = np.ascontiguousarray(
            joint_velocity[selected], dtype="<f4"
        )
        projected_body_position = np.zeros(
            (_KNOWN_GOOD_PROJECTED_FRAME_COUNT, 3), dtype="<f4"
        )
        projected_body_quaternion = np.ascontiguousarray(
            body_quaternion[selected].reshape(
                _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
                _KNOWN_GOOD_SOURCE_BODY_COUNT,
                4,
            )[:, 0],
            dtype="<f4",
        )
        projected_body_linear_velocity = np.zeros(
            (_KNOWN_GOOD_PROJECTED_FRAME_COUNT, 3), dtype="<f4"
        )
        projected_body_angular_velocity = np.zeros(
            (_KNOWN_GOOD_PROJECTED_FRAME_COUNT, 3), dtype="<f4"
        )
        projected_files = {
            "joint_pos.csv": _csv_f32_bytes(
                projected_joint_position,
                tuple(f"joint_{index}" for index in range(29)),
            ),
            "joint_vel.csv": _csv_f32_bytes(
                projected_joint_velocity,
                tuple(f"joint_vel_{index}" for index in range(29)),
            ),
            "body_pos.csv": _csv_f32_bytes(
                projected_body_position,
                ("body_0_x", "body_0_y", "body_0_z"),
            ),
            "body_quat.csv": _csv_f32_bytes(
                projected_body_quaternion,
                ("body_0_w", "body_0_x", "body_0_y", "body_0_z"),
            ),
            "body_lin_vel.csv": _csv_f32_bytes(
                projected_body_linear_velocity,
                ("body_0_vel_x", "body_0_vel_y", "body_0_vel_z"),
            ),
            "body_ang_vel.csv": _csv_f32_bytes(
                projected_body_angular_velocity,
                ("body_0_angvel_x", "body_0_angvel_y", "body_0_angvel_z"),
            ),
            "info.txt": _known_good_projection_info(),
            "metadata.txt": _known_good_projection_metadata(),
        }
        projected_hashes: dict[str, str] = {}
        for name in KNOWN_GOOD_REFERENCE_FILES:
            contents = projected_files[name]
            relative = f"known-good/reference-base/{_KNOWN_GOOD_LEAF}/{name}"
            written = bundle.write_bytes(relative, contents)
            digest = _sha256_bytes(contents)
            if _sha256_file(written) != digest:
                raise ContractError(f"projected known-good file changed: {name}")
            projected_hashes[name] = digest
        actual_directories = sorted(
            path.name for path in base.iterdir() if path.is_dir()
        )
        if actual_directories != [_KNOWN_GOOD_LEAF]:
            raise ContractError("run-local known-good base is not single-motion")
        canonical = _decode_known_good_reference(leaf)
        projected_position = _decode_known_good_body_position(
            leaf, canonical.count
        )
        rule_sha256 = _projection_rule_sha256()
        evidence: dict[str, object] = {
            "schema": "mm-sonic-known-good-projection-evidence/v1",
            "source": {
                "frame_count": _KNOWN_GOOD_SOURCE_FRAME_COUNT,
                "body_count": _KNOWN_GOOD_SOURCE_BODY_COUNT,
                "sha256": dict(sorted(source_hashes.items())),
            },
            "selection": {
                "frame_range": [0, _KNOWN_GOOD_PROJECTED_FRAME_COUNT - 1],
                "frame_count": _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
                "body_indexes": [0],
                "omitted_tail_range": [
                    _KNOWN_GOOD_PROJECTED_FRAME_COUNT,
                    _KNOWN_GOOD_SOURCE_FRAME_COUNT - 1,
                ],
                "omitted_tail_count": _KNOWN_GOOD_OMITTED_TAIL_COUNT,
            },
            "rule": dict(_KNOWN_GOOD_PROJECTION_RULE),
            "rule_sha256": rule_sha256,
            "projected_files_sha256": dict(sorted(projected_hashes.items())),
            "canonical_target_sha256": _canonical_target_sha256(canonical),
            "body_position_sha256": _require_streamable_body_position(
                projected_position
            ),
            "body_linear_velocity_sha256": _require_positive_zero_f32(
                projected_body_linear_velocity,
                "root body linear velocity",
            ),
            "body_angular_velocity_sha256": _require_positive_zero_f32(
                projected_body_angular_velocity,
                "root body angular velocity",
            ),
        }
        projection_sha256 = _sha256_bytes(
            _canonical_json_bytes(evidence, "known-good projection evidence")
        )
        evidence["projection_sha256"] = projection_sha256
        evidence_bytes = _canonical_json_bytes(
            evidence, "known-good projection evidence"
        )
        bundle.write_bytes("known-good/projection.json", evidence_bytes)
        return _KnownGoodProjection(
            reference_base=base,
            reference_leaf=leaf,
            canonical=canonical,
            body_position=projected_position,
            evidence=MappingProxyType(evidence),
        )

    def _gear_command(
        self,
        request: StageARequest,
        context: StageAContext,
        reference_base: Path,
        runtime: _GearRuntimeInputs,
    ) -> tuple[str, ...]:
        del request
        inputs = self._inputs(context)
        binary = inputs.gear_checkout / _GEAR_BINARY_RELATIVE
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise CapabilityUnavailable(
                "official g1_deploy_onnx_ref executable is unavailable"
            )
        argv = [
            str(binary),
            "lo",
            str(runtime.policy),
            str(reference_base),
            "--obs-config",
            str(runtime.observation_config),
        ]
        if runtime.encoder is not None:
            argv.extend(("--encoder-file", str(runtime.encoder)))
        return tuple(argv)

    def _stage_gear_runtime_inputs(
        self,
        context: StageAContext,
        bundle: RunBundle,
        mode: str,
    ) -> _GearRuntimeInputs:
        if mode not in ("file", "stream"):
            raise ContractError("GEAR runtime staging mode is invalid")
        inputs = self._inputs(context)

        def copy_one(
            source: Path,
            expected: str,
            relative: str,
            label: str,
        ) -> Path:
            if _sha256_file(source) != expected:
                raise ExternalInputError(
                    f"authenticated {label} changed before runtime staging"
                )
            try:
                contents = source.read_bytes()
            except OSError as error:
                raise ExternalInputError(
                    f"cannot stage authenticated {label}"
                ) from error
            if _sha256_bytes(contents) != expected:
                raise ExternalInputError(
                    f"authenticated {label} changed while staging"
                )
            staged = bundle.write_bytes(relative, contents)
            if _sha256_file(staged) != expected:
                raise ContractError(f"staged {label} hash changed")
            return staged.resolve(strict=True)

        policy = copy_one(
            inputs.policy,
            context.external_hashes["policy"],
            f"runtime-inputs/{mode}/policy.onnx",
            "policy",
        )
        observation = copy_one(
            inputs.observation_config,
            context.external_hashes["observation_config"],
            f"runtime-inputs/{mode}/observation_config.yaml",
            "observation config",
        )
        encoder = None
        if inputs.encoder is not None:
            expected_encoder = context.external_hashes.get("encoder")
            if expected_encoder is None:
                raise ContractError("authenticated encoder hash is unavailable")
            encoder = copy_one(
                inputs.encoder,
                expected_encoder,
                f"runtime-inputs/{mode}/encoder.onnx",
                "encoder",
            )
        return _GearRuntimeInputs(
            policy=policy,
            observation_config=observation,
            encoder=encoder,
        )

    def _known_good_file_dynamic(
        self,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult:
        before = _snapshot_outputs(bundle)
        state = self._state(context)
        scene = state.get("scene")
        initial_qpos = state.get("initial_qpos")
        joint_addresses = state.get("joint_addresses")
        initial_pelvis = state.get("initial_pelvis_quaternion")
        if (
            not isinstance(scene, RegisteredScene)
            or not isinstance(initial_qpos, np.ndarray)
            or not isinstance(joint_addresses, np.ndarray)
            or not isinstance(initial_pelvis, np.ndarray)
        ):
            raise ContractError("known-good file run requires scene preflight state")
        projection = self._materialize_known_good_projection(context, bundle)
        reference_base = projection.reference_base
        canonical = projection.canonical
        body_position = projection.body_position
        reference_sha256 = _canonical_target_sha256(canonical)
        environment = _child_environment(request)
        inputs = self._inputs(context)
        runtime_inputs = self._stage_gear_runtime_inputs(context, bundle, "file")
        file_launch_identity = _dynamic_launch_identity(
            inputs,
            runtime_inputs,
            scene,
            initial_qpos,
            canonical,
            body_position,
            projection,
        )
        command = self._gear_command(
            request,
            context,
            reference_base,
            runtime_inputs,
        )
        bootstrap_cancellation = threading.Event()
        file_target_log = bundle.path / "dynamic/file/target.csv"
        file_logs_dir = bundle.path / "dynamic/file/gear-logs"
        file_gear_argv = _gear_process_argv(
            command,
            launch_profile="loaded_motion",
            target_motion_logfile=file_target_log,
            logs_dir=file_logs_dir,
        )
        self._register_process_attempt(
            context,
            _process_record(
                "g1_deploy_onnx_ref-known-good-file",
                file_gear_argv,
                executable=Path(command[0]),
                environment=environment,
                outputs=(
                    "dynamic/file/target.csv",
                    "dynamic/file/gear.stdout",
                    "dynamic/file/gear.stderr",
                    "dynamic/file/gear-logs/",
                    "runtime-inputs/file/",
                ),
            ),
        )
        gear = GearProcess(
            run_root=bundle.path,
            command=command,
            target_motion_logfile=file_target_log,
            logs_dir=file_logs_dir,
            stdout_archive=bundle.path / "dynamic/file/gear.stdout",
            stderr_archive=bundle.path / "dynamic/file/gear.stderr",
            launch_profile="loaded_motion",
            readiness_timeout_s=_COLD_GEAR_STARTUP_TIMEOUT_S,
            startup_markers=(
                "✓ Motion data loaded successfully!",
                (
                    f"Started with motion: {_KNOWN_GOOD_LEAF} "
                    "(paused at frame 0)"
                ),
                "Initialized keyboard input interface (default)",
            ),
            cancelled=bootstrap_cancellation.is_set,
            env=environment,
            cwd=inputs.gear_checkout / "gear_sonic_deploy",
        )
        try:
            simulator_command = (
                sys.executable,
                "-u",
                "-B",
                "-m",
                "mm_sonic.gated_sim",
                "--gear-checkout",
                str(inputs.gear_checkout),
                "--run-root",
                str(bundle.path),
            )
            self._register_process_attempt(
                context,
                _process_record(
                    "gated-simulator-known-good-file",
                    simulator_command,
                    executable=Path(sys.executable).resolve(strict=True),
                    environment=environment,
                    outputs=(
                        "dynamic/file/simulator.stdout",
                        "dynamic/file/simulator.stderr",
                        "dynamic/file/bootstrap-sim-logs/",
                        "dynamic/file/scored-sim-logs/",
                    ),
                    module=Path(__file__).with_name("gated_sim.py"),
                ),
            )
            simulator = GatedSimulatorClient(
                run_root=bundle.path,
                gear_checkout=inputs.gear_checkout,
                stdout_archive=bundle.path / "dynamic/file/simulator.stdout",
                stderr_archive=bundle.path / "dynamic/file/simulator.stderr",
                env=environment,
                cwd=_REPOSITORY_ROOT,
            )
        except BaseException:
            gear.close()
            raise
        execution = _execute_known_good_scoring_epoch(
            mode="file",
            gear=gear,
            simulator=simulator,
            scene=scene,
            initial_qpos=initial_qpos,
            canonical=canonical,
            body_position=body_position,
            bundle=bundle,
            bootstrap_cancellation=bootstrap_cancellation,
        )
        metrics = dict(execution.metrics)
        state.update(
            {
                "known_good_reference": canonical,
                "known_good_body_position": body_position,
                "known_good_reference_base": reference_base,
                "known_good_projection": projection,
                "known_good_file_metrics": metrics,
                "known_good_file_launch_identity": file_launch_identity,
            }
        )
        relative = "gates/known_good_file_dynamic.json"
        payload = _canonical_json_bytes(
            {
                "reference_buffer_sha256": reference_sha256,
                "reference_frame_count": canonical.count,
                "single_motion_name": _KNOWN_GOOD_LEAF,
                "projection": dict(projection.evidence),
                "launch_parity": {
                    "fields": dict(file_launch_identity["fields"]),
                    "sha256": file_launch_identity["sha256"],
                },
                "bootstrap": {
                    "scored": False,
                    "steps": execution.bootstrap_steps,
                    "phases": dict(execution.bootstrap),
                },
                "wait_low_state_maintenance": {
                    "scored": False,
                    "steps": execution.wait_maintenance_steps,
                    "phase": dict(execution.wait_maintenance),
                },
                "wait_for_control": {
                    "target_rows": execution.wait_epoch.target_rows,
                    "q_rows": execution.wait_epoch.q_rows,
                    "base_rows": execution.wait_epoch.base_rows,
                    "target_device": execution.wait_epoch.target_device,
                    "target_inode": execution.wait_epoch.target_inode,
                    "target_sha256": execution.wait_epoch.target_sha256,
                },
                "fresh_low_state_prime": dict(execution.prime),
                "scored_physics": {
                    "prime_steps": 1,
                    "control_drive_steps": (
                        execution.coverage.control_drive_steps
                    ),
                    "total_scored_epoch_steps": (
                        execution.coverage.simulator_steps
                    ),
                    "control_drive_duration_s": (
                        execution.coverage.control_drive_duration_s
                    ),
                    "total_scored_epoch_duration_s": (
                        execution.coverage.simulator_duration_s
                    ),
                    "state_rows": execution.coverage.state_rows,
                    "contact_rows": execution.coverage.contact_rows,
                    "wall_duration_s": execution.coverage.wall_duration_s,
                    "prime_scope": "pre-CONTROL-fresh-LowState",
                    "control_drive_scope": "CONTROL-active-cadence-only",
                    "log_row_scope": "total-scored-epoch-including-prime",
                },
                "target_log_audit": dict(execution.target_audit),
                "state_target_pairing": {
                    "alignment": "same-control-tick-positional",
                    "source_order": (
                        "GatherRobotStateToLogger -> policy -> target row -> "
                        "CurrentFrameAdvancement"
                    ),
                    "source_sha256": context.external_hashes[
                        "gear:current_frame_advancement_source"
                    ],
                },
                "metrics": metrics,
            },
            "known-good file dynamic gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        current_outputs = _new_outputs(bundle, before)
        gear_outputs, simulator_outputs = _dynamic_process_outputs(
            _snapshot_outputs(bundle), "file"
        )
        process_records = (
            _process_record(
                "g1_deploy_onnx_ref-known-good-file",
                gear.argv,
                executable=Path(command[0]),
                environment=environment,
                outputs=gear_outputs,
            ),
            _process_record(
                "gated-simulator-known-good-file",
                simulator.command,
                executable=Path(sys.executable).resolve(strict=True),
                environment=environment,
                outputs=simulator_outputs,
                module=Path(__file__).with_name("gated_sim.py"),
            ),
        )
        return GateResult(
            status="pass",
            identity=MappingProxyType(
                {"known_good_reference_buffer": reference_sha256}
            ),
            evidence_hashes=MappingProxyType(
                {"known_good_file_dynamic_sha256": digest}
            ),
            metrics=MappingProxyType(metrics),
            outputs=current_outputs,
            processes=process_records,
        )

    def _known_good_stream_delivery(
        self,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult:
        before = _snapshot_outputs(bundle)
        state = self._state(context)
        scene = state.get("scene")
        initial_qpos = state.get("initial_qpos")
        joint_addresses = state.get("joint_addresses")
        initial_pelvis = state.get("initial_pelvis_quaternion")
        reference_base = state.get("known_good_reference_base")
        projection = state.get("known_good_projection")
        file_canonical = state.get("known_good_reference")
        file_body_position = state.get("known_good_body_position")
        file_launch_identity = state.get("known_good_file_launch_identity")
        if (
            not isinstance(scene, RegisteredScene)
            or not isinstance(initial_qpos, np.ndarray)
            or not isinstance(joint_addresses, np.ndarray)
            or not isinstance(initial_pelvis, np.ndarray)
            or not isinstance(reference_base, Path)
            or not isinstance(projection, _KnownGoodProjection)
            or not isinstance(file_canonical, CanonicalTargetBuffer)
            or not isinstance(file_body_position, np.ndarray)
            or not isinstance(file_launch_identity, Mapping)
        ):
            raise ContractError(
                "known-good stream run requires the completed file prerequisite"
            )
        stream_canonical = _decode_known_good_reference(
            projection.reference_leaf
        )
        stream_body_position = _decode_known_good_body_position(
            projection.reference_leaf,
            stream_canonical.count,
        )
        if reference_base != projection.reference_base:
            raise ContractError(
                "known-good projection base changed before stream launch"
            )
        projected_hashes = projection.evidence.get("projected_files_sha256")
        if not isinstance(projected_hashes, Mapping):
            raise ContractError("known-good projected file hashes are unavailable")
        projection_source = projection.evidence.get("source")
        source_hashes = (
            projection_source.get("sha256")
            if isinstance(projection_source, Mapping)
            else None
        )
        if not isinstance(source_hashes, Mapping):
            raise ContractError("known-good projection source hashes are unavailable")
        for name in KNOWN_GOOD_REFERENCE_FILES:
            if _sha256_file(projection.reference_leaf / name) != projected_hashes.get(
                name
            ):
                raise ContractError(
                    f"known-good projected file changed before stream launch: {name}"
                )
            source_key = f"gear:known_good_reference/{name}"
            if (
                _sha256_file(context.known_good_reference / name)
                != source_hashes.get(source_key)
            ):
                raise ExternalInputError(
                    f"authenticated known-good source changed before stream launch: {name}"
                )
        projection_evidence = _canonical_json_bytes(
            dict(projection.evidence), "known-good projection evidence"
        )
        if (
            _sha256_file(bundle.path / "known-good/projection.json")
            != _sha256_bytes(projection_evidence)
        ):
            raise ContractError(
                "known-good projection evidence changed before stream launch"
            )
        if not _buffers_bit_equal(stream_canonical, file_canonical):
            raise ContractError(
                "known-good decoded value mismatch prevents stream launch"
            )
        if (
            stream_body_position.dtype != file_body_position.dtype
            or stream_body_position.shape != file_body_position.shape
            or stream_body_position.tobytes(order="C")
            != file_body_position.tobytes(order="C")
        ):
            raise ContractError(
                "known-good body position mismatch prevents stream launch"
            )
        reference_sha256 = _canonical_target_sha256(stream_canonical)
        if reference_sha256 != _canonical_target_sha256(file_canonical):
            raise ContractError(
                "known-good reference identity changed before stream launch"
            )
        if reference_sha256 != projection.evidence.get(
            "canonical_target_sha256"
        ):
            raise ContractError(
                "known-good projection identity changed before stream launch"
            )
        environment = _child_environment(request)
        inputs = self._inputs(context)
        runtime_inputs = self._stage_gear_runtime_inputs(
            context, bundle, "stream"
        )
        stream_launch_identity = _dynamic_launch_identity(
            inputs,
            runtime_inputs,
            scene,
            initial_qpos,
            stream_canonical,
            stream_body_position,
            projection,
        )
        launch_parity = _require_launch_parity(
            file_launch_identity,
            stream_launch_identity,
        )
        base_command = self._gear_command(
            request,
            context,
            reference_base,
            runtime_inputs,
        )
        publisher = PosePublisher("tcp://127.0.0.1:*", bundle=bundle)
        try:
            endpoint_host, endpoint_port = _parse_local_zmq_endpoint(
                publisher.endpoint
            )
            command = _stream_gear_command(base_command, publisher.endpoint)
            bootstrap_cancellation = threading.Event()
            stream_target_log = bundle.path / "dynamic/stream/target.csv"
            stream_logs_dir = bundle.path / "dynamic/stream/gear-logs"
            stream_gear_argv = _gear_process_argv(
                command,
                launch_profile="zmq_stream",
                target_motion_logfile=stream_target_log,
                logs_dir=stream_logs_dir,
            )
            self._register_process_attempt(
                context,
                _process_record(
                    "g1_deploy_onnx_ref-known-good-stream",
                    stream_gear_argv,
                    executable=Path(command[0]),
                    environment=environment,
                    outputs=(
                        "dynamic/stream/target.csv",
                        "dynamic/stream/gear.stdout",
                        "dynamic/stream/gear.stderr",
                        "dynamic/stream/gear-logs/",
                        "runtime-inputs/stream/",
                    ),
                ),
            )
            gear = GearProcess(
                run_root=bundle.path,
                command=command,
                target_motion_logfile=stream_target_log,
                logs_dir=stream_logs_dir,
                stdout_archive=bundle.path / "dynamic/stream/gear.stdout",
                stderr_archive=bundle.path / "dynamic/stream/gear.stderr",
                launch_profile="zmq_stream",
                readiness_timeout_s=_COLD_GEAR_STARTUP_TIMEOUT_S,
                cancelled=bootstrap_cancellation.is_set,
                env=environment,
                cwd=inputs.gear_checkout / "gear_sonic_deploy",
            )
        except BaseException:
            publisher.close()
            raise
        try:
            simulator_command = (
                sys.executable,
                "-u",
                "-B",
                "-m",
                "mm_sonic.gated_sim",
                "--gear-checkout",
                str(inputs.gear_checkout),
                "--run-root",
                str(bundle.path),
            )
            self._register_process_attempt(
                context,
                _process_record(
                    "gated-simulator-known-good-stream",
                    simulator_command,
                    executable=Path(sys.executable).resolve(strict=True),
                    environment=environment,
                    outputs=(
                        "dynamic/stream/simulator.stdout",
                        "dynamic/stream/simulator.stderr",
                        "dynamic/stream/bootstrap-sim-logs/",
                        "dynamic/stream/scored-sim-logs/",
                    ),
                    module=Path(__file__).with_name("gated_sim.py"),
                ),
            )
            simulator = GatedSimulatorClient(
                run_root=bundle.path,
                gear_checkout=inputs.gear_checkout,
                stdout_archive=bundle.path / "dynamic/stream/simulator.stdout",
                stderr_archive=bundle.path / "dynamic/stream/simulator.stderr",
                env=environment,
                cwd=_REPOSITORY_ROOT,
            )
        except BaseException:
            try:
                gear.close()
            finally:
                publisher.close()
            raise
        execution = _execute_known_good_scoring_epoch(
            mode="stream",
            gear=gear,
            simulator=simulator,
            scene=scene,
            initial_qpos=initial_qpos,
            canonical=stream_canonical,
            body_position=stream_body_position,
            bundle=bundle,
            bootstrap_cancellation=bootstrap_cancellation,
            publisher=publisher,
        )
        metrics = dict(execution.metrics)
        audit = execution.preload_audit
        if audit is None:
            raise AssertionError("stream execution completed without preload audit")
        audit_sha256 = _sha256_bytes(
            _canonical_json_bytes(dict(audit), "stream preload audit")
        )
        stored_audit = MappingProxyType(
            {**dict(audit), "evidence_sha256": audit_sha256}
        )
        state.update(
            {
                "known_good_stream_metrics": metrics,
                "known_good_stream_audit": stored_audit,
            }
        )
        relative = "gates/known_good_stream_delivery.json"
        payload = _canonical_json_bytes(
            {
                "reference_buffer_sha256": reference_sha256,
                "reference_frame_count": stream_canonical.count,
                "projection": {
                    "rule_sha256": projection.evidence["rule_sha256"],
                    "projection_sha256": projection.evidence[
                        "projection_sha256"
                    ],
                    "canonical_target_sha256": projection.evidence[
                        "canonical_target_sha256"
                    ],
                },
                "endpoint": {
                    "bind": "tcp://127.0.0.1:*",
                    "resolved": publisher.endpoint,
                    "host": endpoint_host,
                    "port": endpoint_port,
                },
                "launch_parity": dict(launch_parity),
                "bootstrap": {
                    "scored": False,
                    "steps": execution.bootstrap_steps,
                    "phases": dict(execution.bootstrap),
                },
                "wait_low_state_maintenance": {
                    "scored": False,
                    "steps": execution.wait_maintenance_steps,
                    "phase": dict(execution.wait_maintenance),
                },
                "wait_for_control": {
                    "target_rows": execution.wait_epoch.target_rows,
                    "q_rows": execution.wait_epoch.q_rows,
                    "base_rows": execution.wait_epoch.base_rows,
                    "target_device": execution.wait_epoch.target_device,
                    "target_inode": execution.wait_epoch.target_inode,
                    "target_sha256": execution.wait_epoch.target_sha256,
                },
                "fresh_low_state_prime": dict(execution.prime),
                "scored_physics": {
                    "prime_steps": 1,
                    "control_drive_steps": (
                        execution.coverage.control_drive_steps
                    ),
                    "total_scored_epoch_steps": (
                        execution.coverage.simulator_steps
                    ),
                    "control_drive_duration_s": (
                        execution.coverage.control_drive_duration_s
                    ),
                    "total_scored_epoch_duration_s": (
                        execution.coverage.simulator_duration_s
                    ),
                    "state_rows": execution.coverage.state_rows,
                    "contact_rows": execution.coverage.contact_rows,
                    "wall_duration_s": execution.coverage.wall_duration_s,
                    "prime_scope": "pre-CONTROL-fresh-LowState",
                    "control_drive_scope": "CONTROL-active-cadence-only",
                    "log_row_scope": "total-scored-epoch-including-prime",
                },
                "delivery_audit": {
                    **dict(stored_audit),
                    "readiness_publications": 1,
                    "logical_publications": 22,
                    "padding_publications": 1,
                    "receipt_fence_publications": 1,
                    "consumer_markers": 25,
                },
                "target_log_audit": dict(execution.target_audit),
                "state_target_pairing": {
                    "alignment": "same-control-tick-positional",
                    "source_order": (
                        "GatherRobotStateToLogger -> policy -> target row -> "
                        "CurrentFrameAdvancement"
                    ),
                    "source_sha256": context.external_hashes[
                        "gear:current_frame_advancement_source"
                    ],
                },
                "metrics": metrics,
            },
            "known-good stream delivery gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        current_outputs = _new_outputs(bundle, before)
        gear_outputs, simulator_outputs = _dynamic_process_outputs(
            _snapshot_outputs(bundle), "stream"
        )
        process_records = (
            _process_record(
                "g1_deploy_onnx_ref-known-good-stream",
                gear.argv,
                executable=Path(command[0]),
                environment=environment,
                outputs=gear_outputs,
            ),
            _process_record(
                "gated-simulator-known-good-stream",
                simulator.command,
                executable=Path(sys.executable).resolve(strict=True),
                environment=environment,
                outputs=simulator_outputs,
                module=Path(__file__).with_name("gated_sim.py"),
            ),
        )
        return GateResult(
            status="pass",
            identity=MappingProxyType(
                {"known_good_reference_buffer": reference_sha256}
            ),
            evidence_hashes=MappingProxyType(
                {
                    "known_good_stream_delivery_sha256": digest,
                    "delivery_audit_sha256": audit_sha256,
                }
            ),
            outputs=current_outputs,
            processes=process_records,
        )

    def _known_good_stream_dynamic(
        self,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult:
        del request
        before = _snapshot_outputs(bundle)
        state = self._state(context)
        metrics = state.get("known_good_stream_metrics")
        audit = state.get("known_good_stream_audit")
        if not isinstance(metrics, Mapping) or not isinstance(audit, Mapping):
            raise ContractError(
                "known-good stream dynamic metrics lack delivery evidence"
            )
        relative = "gates/known_good_stream_dynamic.json"
        payload = _canonical_json_bytes(
            {
                "delivery_evidence_sha256": audit["evidence_sha256"],
                "metrics": dict(metrics),
            },
            "known-good stream dynamic gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        return GateResult(
            status="pass",
            evidence_hashes=MappingProxyType(
                {"known_good_stream_dynamic_sha256": digest}
            ),
            metrics=MappingProxyType(dict(metrics)),
            outputs=_new_outputs(bundle, before),
        )

    def _stage_b_dynamic(
        self,
        request: StageARequest,
        context: StageAContext,
        bundle: RunBundle,
    ) -> GateResult:
        """Stream the complete flat MM reference through the pinned SONIC path."""

        before = _snapshot_outputs(bundle)
        state = self._state(context)
        scene = state.get("scene")
        initial_qpos = state.get("initial_qpos")
        source_canonical = state.get("mm_reference")
        diagnostics = state.get("mm_diagnostics")
        if (
            not isinstance(scene, RegisteredScene)
            or not isinstance(initial_qpos, np.ndarray)
            or not isinstance(source_canonical, CanonicalTargetBuffer)
            or not isinstance(diagnostics, ReferenceDiagnostics)
            or source_canonical.count != 601
            or diagnostics.count != source_canonical.count
        ):
            raise ContractError(
                "Stage B dynamic requires the complete 601-frame kinematic reference"
            )
        transport = canonicalize_sonic_transport(source_canonical)
        canonical = transport.buffer

        projection = self._materialize_known_good_projection(context, bundle)
        environment = _child_environment(request)
        inputs = self._inputs(context)
        runtime_inputs = self._stage_gear_runtime_inputs(
            context, bundle, "stream"
        )
        base_command = self._gear_command(
            request,
            context,
            projection.reference_base,
            runtime_inputs,
        )
        body_position = np.zeros((canonical.count, 3), dtype="<f4")
        publisher = PosePublisher("tcp://127.0.0.1:*", bundle=bundle)
        try:
            endpoint_host, endpoint_port = _parse_local_zmq_endpoint(
                publisher.endpoint
            )
            command = _stream_gear_command(base_command, publisher.endpoint)
            bootstrap_cancellation = threading.Event()
            target_log = bundle.path / "dynamic/stream/target.csv"
            logs_dir = bundle.path / "dynamic/stream/gear-logs"
            gear_argv = _gear_process_argv(
                command,
                launch_profile="zmq_stream",
                target_motion_logfile=target_log,
                logs_dir=logs_dir,
            )
            self._register_process_attempt(
                context,
                _process_record(
                    "g1_deploy_onnx_ref-stage-b",
                    gear_argv,
                    executable=Path(command[0]),
                    environment=environment,
                    outputs=(
                        "dynamic/stream/target.csv",
                        "dynamic/stream/gear.stdout",
                        "dynamic/stream/gear.stderr",
                        "dynamic/stream/gear-logs/",
                        "runtime-inputs/stream/",
                    ),
                ),
            )
            gear = GearProcess(
                run_root=bundle.path,
                command=command,
                target_motion_logfile=target_log,
                logs_dir=logs_dir,
                stdout_archive=bundle.path / "dynamic/stream/gear.stdout",
                stderr_archive=bundle.path / "dynamic/stream/gear.stderr",
                launch_profile="zmq_stream",
                readiness_timeout_s=_COLD_GEAR_STARTUP_TIMEOUT_S,
                cancelled=bootstrap_cancellation.is_set,
                env=environment,
                cwd=inputs.gear_checkout / "gear_sonic_deploy",
            )
        except BaseException:
            publisher.close()
            raise
        try:
            simulator_command = (
                sys.executable,
                "-u",
                "-B",
                "-m",
                "mm_sonic.gated_sim",
                "--gear-checkout",
                str(inputs.gear_checkout),
                "--run-root",
                str(bundle.path),
                "--unpaced-physics",
            )
            self._register_process_attempt(
                context,
                _process_record(
                    "gated-simulator-stage-b",
                    simulator_command,
                    executable=Path(sys.executable).resolve(strict=True),
                    environment=environment,
                    outputs=(
                        "dynamic/stream/simulator.stdout",
                        "dynamic/stream/simulator.stderr",
                        "dynamic/stream/bootstrap-sim-logs/",
                        "dynamic/stream/scored-sim-logs/",
                    ),
                    module=Path(__file__).with_name("gated_sim.py"),
                ),
            )
            simulator = GatedSimulatorClient(
                run_root=bundle.path,
                gear_checkout=inputs.gear_checkout,
                unpaced_physics=True,
                stdout_archive=bundle.path / "dynamic/stream/simulator.stdout",
                stderr_archive=bundle.path / "dynamic/stream/simulator.stderr",
                env=environment,
                cwd=_REPOSITORY_ROOT,
            )
        except BaseException:
            try:
                gear.close()
            finally:
                publisher.close()
            raise

        execution = _execute_known_good_scoring_epoch(
            mode="stream",
            gear=gear,
            simulator=simulator,
            scene=scene,
            initial_qpos=initial_qpos,
            canonical=canonical,
            body_position=body_position,
            bundle=bundle,
            bootstrap_cancellation=bootstrap_cancellation,
            publisher=publisher,
            required_control_duration_s=STAGE_B_REFERENCE_DURATION_S,
            control_lead_rows=STAGE_B_CONTROL_LEAD_ROWS,
        )
        coverage = validate_stage_b_coverage(
            commands=flat_command_script(),
            canonical=canonical,
            accepted_command_indices=tuple(range(30)),
            observed_target_rows=execution.coverage.target_rows,
            control_drive_steps=execution.coverage.control_drive_steps,
            control_drive_duration_s=(
                execution.coverage.control_drive_duration_s
            ),
            sim_dt_s=simulator.sim_dt,
            control_lead_rows=STAGE_B_CONTROL_LEAD_ROWS,
        )
        tracking = dict(execution.metrics)
        if (
            tracking.get("frame_count") != canonical.count
            or len(tracking.get("joint_tracking_trace_rad", ()))
            != canonical.count
            or len(tracking.get("pelvis_tracking_trace_rad", ()))
            != canonical.count
        ):
            raise ContractError("Stage B tracking evidence lacks all 601 frames")
        preload_audit = execution.preload_audit
        if not isinstance(preload_audit, Mapping):
            raise ContractError("Stage B delivery lacks the stream preload audit")
        safety = load_flat_simulator_safety(
            bundle.path / "dynamic/stream/scored-sim-logs",
            terrain_geoms=scene.terrain_geoms,
            allowed_foot_geoms=scene.allowed_foot_geoms,
            forbidden_geom_groups=scene.forbidden_geom_groups,
            sim_dt_s=simulator.sim_dt,
            expected_final_pelvis_xy=tuple(
                float(value)
                for value in diagnostics.physical_pelvis_position[-1, :2]
            ),
            expected_simulator_steps=execution.coverage.simulator_steps,
            expected_state_rows=execution.coverage.state_rows,
            expected_contact_rows=execution.coverage.contact_rows,
        )
        coverage_payload = {
            "command_count": coverage.command_count,
            "accepted_command_indices": list(range(30)),
            "frame_count": coverage.frame_count,
            "observed_target_rows": execution.coverage.target_rows,
            "reference_duration_s": coverage.reference_duration_s,
            "exact_command_coverage": coverage.exact_command_coverage,
            "exact_frame_coverage": coverage.exact_frame_coverage,
            "exact_control_duration": coverage.exact_control_duration,
            "sim_dt_s": coverage.sim_dt_s,
            "control_lead_rows": coverage.control_lead_rows,
            "control_drive_steps": execution.coverage.control_drive_steps,
            "control_drive_duration_s": (
                execution.coverage.control_drive_duration_s
            ),
            "total_scored_epoch_steps": execution.coverage.simulator_steps,
            "total_scored_epoch_duration_s": (
                execution.coverage.simulator_duration_s
            ),
        }
        safety_payload = {
            "state_rows": safety.state_rows,
            "contact_rows": safety.contact_rows,
            "expected_state_rows": execution.coverage.state_rows,
            "expected_contact_rows": execution.coverage.contact_rows,
            "expected_simulator_steps": execution.coverage.simulator_steps,
            "exact_log_coverage": True,
            "terrain_geoms": list(scene.terrain_geoms),
            "minimum_pelvis_local_height_m": (
                safety.minimum_pelvis_local_height_m
            ),
            "minimum_pelvis_up_dot": safety.minimum_pelvis_up_dot,
            "forbidden_contact_groups": list(
                safety.forbidden_contact_groups
            ),
        }
        secondary_payload = {
            "swing_foot_scuff_count": safety.swing_foot_scuff_count,
            "minimum_foot_clearance_m": safety.minimum_foot_clearance_m,
            "horizontal_path_drift_m": safety.horizontal_path_drift_m,
            "contact_impulses_ns": list(safety.contact_impulses_ns),
            "policy_execution_timing_ns": None,
        }
        timing_payload = {
            "wall_duration_s": execution.coverage.wall_duration_s,
            "control_drive_duration_s": (
                execution.coverage.control_drive_duration_s
            ),
            "total_scored_epoch_duration_s": (
                execution.coverage.simulator_duration_s
            ),
        }
        metrics = {
            "coverage": coverage_payload,
            "tracking": tracking,
            "safety": safety_payload,
            "secondary": secondary_payload,
            "timings": timing_payload,
            "delivery_audit": dict(preload_audit),
            "transport_normalization": {
                "rule": "float32-subnormal-to-positive-zero",
                "subnormal_count": transport.subnormal_count,
                "maximum_subnormal_magnitude": (
                    transport.maximum_subnormal_magnitude
                ),
                "source_reference_sha256": _canonical_target_sha256(
                    source_canonical
                ),
                "transport_reference_sha256": _canonical_target_sha256(
                    canonical
                ),
            },
        }
        relative = "gates/stage_b_dynamic.json"
        payload = _canonical_json_bytes(
            {
                "source_reference_buffer_sha256": _canonical_target_sha256(
                    source_canonical
                ),
                "reference_buffer_sha256": _canonical_target_sha256(canonical),
                "reference_frame_count": canonical.count,
                "endpoint": {
                    "bind": "tcp://127.0.0.1:*",
                    "resolved": publisher.endpoint,
                    "host": endpoint_host,
                    "port": endpoint_port,
                },
                "bootstrap_reference_projection_sha256": (
                    projection.evidence.get("projection_sha256")
                ),
                "bootstrap": {
                    "scored": False,
                    "steps": execution.bootstrap_steps,
                    "phases": dict(execution.bootstrap),
                },
                "wait_low_state_maintenance": {
                    "scored": False,
                    "steps": execution.wait_maintenance_steps,
                    "phase": dict(execution.wait_maintenance),
                },
                "wait_for_control": {
                    "target_rows": execution.wait_epoch.target_rows,
                    "q_rows": execution.wait_epoch.q_rows,
                    "base_rows": execution.wait_epoch.base_rows,
                    "target_device": execution.wait_epoch.target_device,
                    "target_inode": execution.wait_epoch.target_inode,
                    "target_sha256": execution.wait_epoch.target_sha256,
                },
                "fresh_low_state_prime": dict(execution.prime),
                "target_log_audit": dict(execution.target_audit),
                "metrics": metrics,
            },
            "Stage B dynamic gate",
        )
        bundle.write_bytes(relative, payload)
        digest = _sha256_bytes(payload)
        current_outputs = _new_outputs(bundle, before)
        gear_outputs, simulator_outputs = _dynamic_process_outputs(
            _snapshot_outputs(bundle), "stream"
        )
        process_records = (
            _process_record(
                "g1_deploy_onnx_ref-stage-b",
                gear.argv,
                executable=Path(command[0]),
                environment=environment,
                outputs=gear_outputs,
            ),
            _process_record(
                "gated-simulator-stage-b",
                simulator.command,
                executable=Path(sys.executable).resolve(strict=True),
                environment=environment,
                outputs=simulator_outputs,
                module=Path(__file__).with_name("gated_sim.py"),
            ),
        )
        return GateResult(
            status="pass",
            evidence_hashes=MappingProxyType(
                {"stage_b_dynamic_sha256": digest}
            ),
            metrics=MappingProxyType(metrics),
            outputs=current_outputs,
            processes=process_records,
        )


def _raw_output_and_capability_check(
    namespace: argparse.Namespace,
) -> tuple[Path, str | None]:
    """Check output confinement before diagnosing explicit path capability."""

    raw_output = namespace.output_root
    if raw_output.startswith("~"):
        raise _CLIConfigurationError(
            f"output_root path cannot use user expansion: {raw_output}"
        )
    try:
        output_root = Path(raw_output).resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise _CLIConfigurationError(
            f"invalid output_root: {raw_output}"
        ) from error
    if output_root.exists() and not output_root.is_dir():
        raise _CLIConfigurationError(
            f"output_root must be a directory: {output_root}"
        )

    expected = (
        ("gear_checkout", True),
        ("policy", False),
        ("observation_config", False),
        ("encoder", False),
        ("source_mjcf", False),
        ("terrain_dir", True),
    )
    if namespace.command == "stage-b":
        expected = (*expected, ("stage_a_evidence", False))
    diagnoses: list[str] = []
    for name, directory in expected:
        raw = getattr(namespace, name, None)
        if raw is None and name == "encoder":
            continue
        label = name.replace("_", "-")
        if raw.startswith("~"):
            raise _CLIConfigurationError(
                f"--{label} path cannot use user expansion: {raw}"
            )
        try:
            claimed = Path(raw).resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise _CLIConfigurationError(
                f"invalid --{label} path: {raw}"
            ) from error
        if directory and (
            output_root == claimed or output_root.is_relative_to(claimed)
        ):
            raise _CLIConfigurationError(
                f"output_root {output_root} is nested under input {claimed}"
            )
        if not directory and (
            output_root == claimed or output_root.is_relative_to(claimed)
        ):
            raise _CLIConfigurationError(
                f"output_root {output_root} is nested under input {claimed}"
            )
        try:
            resolved = Path(raw).resolve(strict=True)
        except (OSError, RuntimeError):
            diagnoses.append(f"--{label} does not exist: {raw}")
            continue
        try:
            metadata = resolved.stat()
        except OSError:
            diagnoses.append(f"--{label} cannot be inspected: {raw}")
            continue
        if stat.S_ISDIR(metadata.st_mode):
            if output_root == resolved or output_root.is_relative_to(resolved):
                raise _CLIConfigurationError(
                    f"output_root {output_root} is nested under input {resolved}"
                )
        elif output_root == resolved:
            raise _CLIConfigurationError(
                f"output_root {output_root} duplicates input {resolved}"
            )
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(metadata.st_mode):
            kind = "directory" if directory else "regular file"
            diagnoses.append(f"--{label} must be a {kind}: {raw}")
    return output_root, "; ".join(diagnoses) if diagnoses else None


def _raw_external_manifest(namespace: argparse.Namespace) -> dict[str, object]:
    return {
        "gear_checkout": str(namespace.gear_checkout),
        "gear_commit": None,
        "gear_dirty": None,
        "policy": str(namespace.policy),
        "observation_config": str(namespace.observation_config),
        "encoder": (
            None if namespace.encoder is None else str(namespace.encoder)
        ),
        "terrain_dir": str(namespace.terrain_dir),
        "source_mjcf": str(namespace.source_mjcf),
        "hashes": {
            "gear_checkout": None,
            "policy": None,
            "observation_config": None,
            "encoder": None,
            "terrain_dir": None,
            "source_mjcf": None,
        },
    }


def _finalize_pre_execution(
    request: StageARequest,
    bundle: RunBundle,
    *,
    command_status: str,
    reason: str,
    stdout: IO[str],
) -> int:
    """Seal a result-free diagnosis before external authentication exists."""

    if command_status not in ("not_run", "integration_failure"):
        raise ContractError("pre-execution status is invalid")
    invocation = {
        "argv": ["mm_sonic.cli", *request.argv],
        "invocation_cwd": str(request.invocation_cwd),
        "environment": {
            "allowlist": list(_SAFE_ENVIRONMENT_ALLOWLIST),
            "values": dict(request.environment),
        },
        "external": _raw_external_manifest(request.namespace),
    }
    invocation_sha256 = _sha256_bytes(
        _canonical_json_bytes(invocation, "raw invocation")
    )
    diagnosis = {
        "schema": "mm-sonic-pre-execution-diagnosis/v1",
        "status": command_status,
        "reason": reason,
        "invocation_sha256": invocation_sha256,
    }
    diagnosis_bytes = _canonical_json_bytes(
        diagnosis, "pre-execution diagnosis"
    )
    diagnosis_sha256 = _sha256_bytes(diagnosis_bytes)
    diagnosis_path = "pre-execution-diagnosis.json"
    bundle.write_bytes(diagnosis_path, diagnosis_bytes)

    target_count = _target_gate_count(request)
    records = [_pending_gate(name) for name in STAGE_A_GATE_NAMES]
    first_status = (
        "not_run" if command_status == "not_run" else "integration_failure"
    )
    records[0] = _pending_gate(
        STAGE_A_GATE_NAMES[0], first_status, reason
    )
    for index in range(1, target_count):
        records[index] = _pending_gate(
            STAGE_A_GATE_NAMES[index],
            "not_run",
            f"blocked before authentication: {reason}",
        )
    registry_sha256 = _sha256_file(_STAGE_A_REGISTRY_PATH)
    evidence = {
        "schema": "mm-sonic-stage-a-evidence/v1",
        "registry_sha256": registry_sha256,
        "command": request.command,
        "mode": request.mode,
        "argv": ["mm_sonic.cli", *request.argv],
        "invocation_cwd": str(request.invocation_cwd),
        "environment": {
            "allowlist": list(_SAFE_ENVIRONMENT_ALLOWLIST),
            "values": dict(request.environment),
        },
        "command_status": command_status,
        "stage_a_status": (
            "not_run" if command_status == "not_run" else "failed"
        ),
        "gates": records,
        "identity": {},
        "identity_sha256": None,
        "metrics": {},
        "outputs": {diagnosis_path: diagnosis_sha256},
        "invocation_sha256": invocation_sha256,
        "diagnosis_sha256": diagnosis_sha256,
    }
    bundle.write_bytes(
        "stage-a-evidence.json",
        _canonical_json_bytes(evidence, "Stage A evidence"),
    )
    bundle.update_manifest(
        {
            "external": _raw_external_manifest(request.namespace),
            "repositories": {
                "motion_matching": {"commit": None, "dirty": None},
                "gear_sonic": {"commit": None, "dirty": None},
            },
            "artifact_hashes": {
                name: None for name in sorted(_ARTIFACT_HASH_KEYS)
            },
            "command_script": dict(_command_script_manifest(request)),
            "perturbation": {
                "id": "nominal",
                "lateral_offset_m": 0.0,
                "yaw_offset_rad": 0.0,
            },
            "coordinate_transform": {
                "source": HOLDEN_COORDINATE_SIGNATURE,
                "target": MUJOCO_COORDINATE_SIGNATURE,
                "matrix": HOLDEN_TO_MUJOCO_MATRIX.tolist(),
            },
            "processes": [
                {
                    "name": "mm_sonic.cli",
                    "argv": ["mm_sonic.cli", *request.argv],
                    "invocation_cwd": str(request.invocation_cwd),
                    "environment": {
                        "allowlist": list(_SAFE_ENVIRONMENT_ALLOWLIST),
                        "values": dict(request.environment),
                    },
                }
            ],
        }
    )
    bundle.finalize(
        "not_run",
        outcome={
            "status": command_status,
            "stage_a_status": evidence["stage_a_status"],
            "reason": reason,
            "invocation_sha256": invocation_sha256,
            "diagnosis_sha256": diagnosis_sha256,
        },
    )
    stdout.write(
        json.dumps(
            {
                "status": command_status,
                "stage_a_status": evidence["stage_a_status"],
                "evidence": str(bundle.path / "stage-a-evidence.json"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return _command_status_exit(command_status)


def _finalize_stage_b_pre_execution(
    request: StageARequest,
    bundle: RunBundle,
    *,
    command_status: str,
    reason: str,
    stdout: IO[str],
) -> int:
    """Seal a fail-closed Stage B verdict before authentication can begin."""

    if command_status not in ("not_run", "integration_failure"):
        raise ContractError("Stage B pre-execution status is invalid")
    diagnosis = {
        "schema": "mm-sonic-pre-execution-diagnosis/v1",
        "status": command_status,
        "reason": reason,
        "argv": ["mm_sonic.cli", *request.argv],
        "invocation_cwd": str(request.invocation_cwd),
    }
    diagnosis_bytes = _canonical_json_bytes(
        diagnosis, "Stage B pre-execution diagnosis"
    )
    diagnosis_sha256 = _sha256_bytes(diagnosis_bytes)
    diagnosis_path = "pre-execution-diagnosis.json"
    bundle.write_bytes(diagnosis_path, diagnosis_bytes)
    registry_sha256 = _sha256_file(_STAGE_B_REGISTRY_PATH)
    stage_b_status = (
        "not_run" if command_status == "not_run" else "failed"
    )
    evidence = {
        "schema": "mm-sonic-trial-verdict/v1",
        "stage": "B",
        "scene_id": FLAT_SCENE_ID,
        "terrain_weight": 0.0,
        "expected_frames": 601,
        "expected_sim_time_s": 12.0,
        "integration_pass": False,
        "kinematic_pass": False,
        "dynamic_pass": False,
        "failure_layer": "prerequisite",
        "command_status": command_status,
        "reason": reason,
        "gates": [
            _pending_gate(
                name,
                "not_run",
                f"blocked before authentication: {reason}",
            )
            for name in _STAGE_B_GATE_NAMES
        ],
        "identity": {},
        "metrics": {},
        "timings": {},
        "evidence_hashes": {
            "registry_sha256": registry_sha256,
            "diagnosis_sha256": diagnosis_sha256,
        },
        "outputs": {diagnosis_path: diagnosis_sha256},
    }
    bundle.write_bytes(
        "stage-b-evidence.json",
        _canonical_json_bytes(evidence, "Stage B evidence"),
    )
    bundle.update_manifest(
        {
            "external": _raw_external_manifest(request.namespace),
            "repositories": {
                "motion_matching": {"commit": None, "dirty": None},
                "gear_sonic": {"commit": None, "dirty": None},
            },
            "artifact_hashes": {
                name: None for name in sorted(_ARTIFACT_HASH_KEYS)
            },
            "command_script": dict(_command_script_manifest(request)),
            "perturbation": {
                "id": "nominal",
                "lateral_offset_m": 0.0,
                "yaw_offset_rad": 0.0,
            },
            "coordinate_transform": {
                "source": HOLDEN_COORDINATE_SIGNATURE,
                "target": MUJOCO_COORDINATE_SIGNATURE,
                "matrix": HOLDEN_TO_MUJOCO_MATRIX.tolist(),
            },
            "processes": [
                {
                    "name": "mm_sonic.cli",
                    "argv": ["mm_sonic.cli", *request.argv],
                    "invocation_cwd": str(request.invocation_cwd),
                    "environment": {
                        "allowlist": list(_SAFE_ENVIRONMENT_ALLOWLIST),
                        "values": dict(request.environment),
                    },
                }
            ],
        }
    )
    bundle.finalize(
        stage_b_status,
        outcome={
            "status": command_status,
            "stage_b_status": stage_b_status,
            "reason": reason,
            "diagnosis_sha256": diagnosis_sha256,
        },
    )
    stdout.write(
        json.dumps(
            {
                "status": command_status,
                "stage_b_status": stage_b_status,
                "evidence": str(bundle.path / "stage-b-evidence.json"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return _command_status_exit(command_status)


def _finalize_command_pre_execution(
    request: StageARequest,
    bundle: RunBundle,
    *,
    command_status: str,
    reason: str,
    stdout: IO[str],
) -> int:
    if request.command == "stage-b":
        return _finalize_stage_b_pre_execution(
            request,
            bundle,
            command_status=command_status,
            reason=reason,
            stdout=stdout,
        )
    return _finalize_pre_execution(
        request,
        bundle,
        command_status=command_status,
        reason=reason,
        stdout=stdout,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    operations: StageAOperations | None = None,
    environ: Mapping[str, str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Parse and execute one non-interactive Stage A command."""

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    source_environment = os.environ if environ is None else environ
    try:
        try:
            invocation_cwd = Path.cwd().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise _CLIConfigurationError(
                "invocation cwd cannot be resolved"
            ) from error
        if not invocation_cwd.is_absolute() or not invocation_cwd.is_dir():
            raise _CLIConfigurationError(
                "invocation cwd must be a canonical absolute directory"
            )
        namespace = _parser().parse_args(arguments)
        safe_environment = _safe_environment(source_environment)
        request = StageARequest(
            command=namespace.command,
            mode=getattr(namespace, "mode", None),
            argv=arguments,
            namespace=namespace,
            environment=safe_environment,
            invocation_cwd=invocation_cwd,
        )
        output_root, capability_diagnosis = _raw_output_and_capability_check(
            namespace
        )
        experiment = "stage-b" if request.command == "stage-b" else "stage-a"
        bundle = RunBundle.create(
            output_root,
            experiment,
            _run_id(request.command, request.mode),
        )
        if capability_diagnosis is not None:
            return _finalize_command_pre_execution(
                request,
                bundle,
                command_status="not_run",
                reason=capability_diagnosis,
                stdout=output,
            )
        try:
            inputs = ExternalInputs.from_cli(namespace)
        except ExternalInputError as error:
            return _finalize_command_pre_execution(
                request,
                bundle,
                command_status="not_run",
                reason=str(error),
                stdout=output,
            )
        selected_operations = (
            DefaultStageAOperations() if operations is None else operations
        )
        try:
            if request.command == "stage-b":
                return _execute_stage_b(
                    request,
                    inputs,
                    selected_operations,
                    output,
                    bundle,
                )
            return _execute(
                request,
                inputs,
                selected_operations,
                output,
                bundle,
            )
        except CapabilityUnavailable as error:
            return _finalize_command_pre_execution(
                request,
                bundle,
                command_status="not_run",
                reason=str(error),
                stdout=output,
            )
        except (ExternalInputError, ContractError) as error:
            if (
                bundle.status != "running"
                or bundle.output_exists(
                    "stage-b-evidence.json"
                    if request.command == "stage-b"
                    else "stage-a-evidence.json"
                )
            ):
                errors.write(f"configuration/integration error: {error}\n")
                return EXIT_CONFIGURATION
            code = _finalize_command_pre_execution(
                request,
                bundle,
                command_status="integration_failure",
                reason=f"{type(error).__name__}: {error}",
                stdout=output,
            )
            errors.write(f"configuration/integration error: {error}\n")
            return code
    except (_CLIConfigurationError, ExternalInputError, ContractError) as error:
        errors.write(f"configuration/integration error: {error}\n")
        return EXIT_CONFIGURATION
    except CapabilityUnavailable as error:
        # Default authentication performs only safe read-only work. Capability
        # checks normally occur after RunBundle creation and are preserved as
        # immutable not_run evidence; this guard covers an injected adapter
        # that reports absence too early to create trustworthy evidence.
        errors.write(f"external capability unavailable: {error}\n")
        return EXIT_NOT_RUN


if __name__ == "__main__":
    raise SystemExit(main())
