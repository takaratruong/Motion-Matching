"""Pure offline command artifact codec and auditor for manual SONIC runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Sequence

import numpy as np

from .commands import CommandSample, flat_command_script
from .coordinator import SessionConfig, SourceValidator
from .hands import (
    Dex3HandTargets,
    HandTrackingReport,
    hand_joint_ranges,
    hand_qpos_addresses,
    hand_targets_record,
    measure_hand_tracking,
    parse_hand_targets_record,
)
from .joints import ContractError, load_joint_contract
from .process import MMChunkClient
from .replay_video import load_state_stream
from .timeline import CanonicalTargetBuffer, TargetTimeline
from .zmq_v1 import decode_pose_v1


MANUAL_COMMAND_SCHEMA = "mm-sonic-manual-command/v3"
MANUAL_MAPPER_VERSION = "holden-control/v1"
_MANUAL_INPUT_SOURCES = ("x11", "terminal")
_MANUAL_MODES = ("script", "interactive")
_TARGET_RATE_HZ = 50.0
_HEADING_TOLERANCE_RAD = 1.0e-6
_VELOCITY_TOLERANCE_MPS = 1.0e-9
_NPZ_ARRAYS = {"joint_position", "joint_velocity", "body_quat_w", "frame_index"}
_MANUAL_SUMMARY_SCHEMA = "mm-sonic-manual-demo/v3"
_PRELOAD_CHUNKS = 4
_OPERATOR_CHUNKS = 30
_FRAMES_PER_CHUNK = 20
_MIN_ROOT_HEIGHT_M = 0.65
_MIN_PELVIS_UP_DOT = 0.90
_MIN_PATH_DISTANCE_M = 1.0
_MIN_YAW_CHANGE_RAD = math.radians(20.0)
_MAX_FINAL_STOP_DISPLACEMENT_M = 0.25
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SONIC_ROOT = _REPOSITORY_ROOT / "sonic"
_DEFAULT_TERRAIN = Path(
    "/home/ubuntu/projects/motion-matching/resources/g1_terrain"
)

# The official simulator emits a fall diagnostic on stderr when the humanoid
# topples; its presence in an accepted run is a fail-closed condition.
GEAR_FALL_MARKER = "Warning: Robot has fallen"


@dataclass(frozen=True)
class ManualCommandArtifact:
    """One immutable, ordered record of a manual run's committed commands."""

    schema: str
    mode: str
    preload_chunks: int
    commands: tuple[CommandSample, ...]
    hand_targets: Dex3HandTargets


@dataclass(frozen=True)
class MMCommandReplay:
    """Independent MM qualification plus command-conditioned pose replay."""

    baseline_canonical_parity: bool
    command_buffer: CanonicalTargetBuffer

    def __post_init__(self) -> None:
        if type(self.baseline_canonical_parity) is not bool:
            raise ContractError("MM baseline parity must be a boolean")
        if not isinstance(self.command_buffer, CanonicalTargetBuffer):
            raise ContractError("MM command replay requires a canonical buffer")


def _validated_commands(
    commands: Sequence[CommandSample],
) -> tuple[CommandSample, ...]:
    values = tuple(commands)
    if not values or any(
        not isinstance(value, CommandSample) for value in values
    ):
        raise ContractError(
            "manual command artifact requires immutable CommandSample values"
        )
    if tuple(value.chunk_index for value in values) != tuple(range(len(values))):
        raise ContractError(
            "manual command artifact chunk indices must be exactly 0..N-1"
        )
    return values


def _validated_mode(mode: object) -> str:
    if type(mode) is not str or mode not in _MANUAL_MODES:
        raise ContractError("manual command artifact mode must be script or interactive")
    return mode


def _validated_preload(preload_chunks: object, chunk_count: int) -> int:
    if type(preload_chunks) is not int or preload_chunks < 0:
        raise ContractError(
            "manual command artifact preload_chunks must be a nonnegative integer"
        )
    if preload_chunks > chunk_count:
        raise ContractError(
            "manual command artifact preload_chunks exceeds the command count"
        )
    return preload_chunks


def manual_command_artifact_bytes(
    *,
    mode: str,
    preload_chunks: int,
    commands: Sequence[CommandSample],
    hand_targets: Dex3HandTargets,
) -> bytes:
    values = _validated_commands(commands)
    validated_mode = _validated_mode(mode)
    validated_preload = _validated_preload(preload_chunks, len(values))
    document = {
        "schema": MANUAL_COMMAND_SCHEMA,
        "mode": validated_mode,
        "preload_chunks": validated_preload,
        "chunk_count": len(values),
        "hand_control": hand_targets_record(hand_targets),
        "commands": [
            {
                "chunk_index": command.chunk_index,
                "origin": (
                    "preload"
                    if command.chunk_index < validated_preload
                    else "operator"
                ),
                "requested_velocity_mujoco": list(
                    command.requested_velocity_mujoco
                ),
                "desired_heading_mujoco_wxyz": list(
                    command.desired_heading_mujoco_wxyz
                ),
            }
            for command in values
        ],
    }
    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError(
            "manual command artifact cannot be serialized canonically"
        ) from error


def scan_for_fall_marker(log: bytes | bytearray | memoryview) -> bool:
    """Return whether the GEAR fall diagnostic marker appears in a log."""

    if not isinstance(log, (bytes, bytearray, memoryview)):
        raise ContractError("fall marker scan requires bytes")
    return GEAR_FALL_MARKER.encode("ascii") in bytes(log)


def load_canonical_target_npz(
    data: bytes | bytearray | memoryview,
) -> CanonicalTargetBuffer:
    """Decode a supplied canonical_target.npz into its canonical buffer."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("canonical target npz must be bytes")
    try:
        with np.load(io.BytesIO(bytes(data)), allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_ARRAYS:
                raise ContractError("canonical target npz has unexpected arrays")
            return CanonicalTargetBuffer(
                joint_position=archive["joint_position"],
                joint_velocity=archive["joint_velocity"],
                body_quat_w=archive["body_quat_w"],
                frame_index=archive["frame_index"],
            )
    except ContractError:
        raise
    except (OSError, ValueError) as error:
        raise ContractError("canonical target npz is invalid") from error


def authenticate_pose_stream(
    frames: Sequence[bytes | bytearray | memoryview],
    *,
    expected_first_frame: int,
    hand_targets: Dex3HandTargets | None = None,
) -> tuple[CanonicalTargetBuffer, bool]:
    """Decode archived ZMQ v1 frames into one contiguous canonical buffer.

    When ``hand_targets`` is supplied, every frame must carry both seven-value
    hand fields; the returned flag reports whether their bits matched exactly.
    Missing hand fields fail closed structurally.
    """

    if type(expected_first_frame) is not int or expected_first_frame < 0:
        raise ContractError("expected_first_frame must be a nonnegative integer")
    values = tuple(frames)
    if not values:
        raise ContractError("pose stream must contain at least one frame")
    joint_position: list[np.ndarray] = []
    joint_velocity: list[np.ndarray] = []
    body_quat_w: list[np.ndarray] = []
    frame_index: list[np.ndarray] = []
    expected_next = expected_first_frame
    hand_transport_exact = hand_targets is not None
    expected_left = None if hand_targets is None else hand_targets.left_f32.view(np.uint32)
    expected_right = None if hand_targets is None else hand_targets.right_f32.view(np.uint32)
    for message in values:
        decoded = decode_pose_v1(message, baseline_mode=True)
        if int(decoded.frame_index[0]) != expected_next:
            raise ContractError(
                "pose stream frame indices are not contiguous"
            )
        if hand_targets is not None:
            if decoded.left_hand_joints is None or decoded.right_hand_joints is None:
                raise ContractError(
                    "archived pose stream is missing explicit hand fields"
                )
            left_bits = np.ascontiguousarray(
                decoded.left_hand_joints, dtype="<f4"
            ).view(np.uint32)
            right_bits = np.ascontiguousarray(
                decoded.right_hand_joints, dtype="<f4"
            ).view(np.uint32)
            if not (
                left_bits.shape == expected_left.shape
                and np.array_equal(left_bits, expected_left)
                and right_bits.shape == expected_right.shape
                and np.array_equal(right_bits, expected_right)
            ):
                hand_transport_exact = False
        joint_position.append(decoded.joint_position)
        joint_velocity.append(decoded.joint_velocity)
        body_quat_w.append(decoded.body_quat_w)
        frame_index.append(decoded.frame_index)
        expected_next = int(decoded.frame_index[-1]) + 1
    buffer = CanonicalTargetBuffer(
        joint_position=np.concatenate(joint_position, axis=0),
        joint_velocity=np.concatenate(joint_velocity, axis=0),
        body_quat_w=np.concatenate(body_quat_w, axis=0),
        frame_index=np.concatenate(frame_index),
    )
    return buffer, hand_transport_exact


@dataclass(frozen=True)
class StateMetrics:
    """Scalar summaries of one completed manual run's MuJoCo state stream."""

    minimum_root_height_m: float
    minimum_pelvis_up_dot: float
    path_distance_m: float
    yaw_change_rad: float
    final_stop_displacement_m: float


def _yaw_from_quaternion_wxyz(quaternion: np.ndarray) -> float:
    w = float(quaternion[0])
    x = float(quaternion[1])
    y = float(quaternion[2])
    z = float(quaternion[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def state_metrics(
    states: Sequence[object],
    *,
    final_stop_seconds: float,
) -> StateMetrics:
    values = tuple(states)
    if not values:
        raise ContractError("state metrics require a nonempty state stream")
    if (
        type(final_stop_seconds) not in (int, float)
        or not math.isfinite(float(final_stop_seconds))
        or float(final_stop_seconds) <= 0.0
    ):
        raise ContractError("final_stop_seconds must be positive and finite")

    heights: list[float] = []
    up_dots: list[float] = []
    positions: list[tuple[float, float]] = []
    yaws: list[float] = []
    times: list[float] = []
    for state in values:
        qpos = np.asarray(getattr(state, "qpos"), dtype=np.float64)
        if qpos.shape[0] < 7:
            raise ContractError("state qpos must contain a free root joint")
        heights.append(float(qpos[2]))
        quaternion = qpos[3:7]
        up_dots.append(1.0 - 2.0 * (float(quaternion[1]) ** 2 + float(quaternion[2]) ** 2))
        positions.append((float(qpos[0]), float(qpos[1])))
        yaws.append(_yaw_from_quaternion_wxyz(quaternion))
        times.append(float(getattr(state, "sim_time_s")))

    path_distance = 0.0
    for (x0, y0), (x1, y1) in zip(positions, positions[1:]):
        path_distance += math.hypot(x1 - x0, y1 - y0)

    signed_yaw_change = 0.0
    for previous, current in zip(yaws, yaws[1:]):
        signed_yaw_change += math.remainder(
            current - previous, 2.0 * math.pi
        )

    final_start_time = times[-1] - float(final_stop_seconds)
    window = [
        position
        for position, timestamp in zip(positions, times)
        if timestamp >= final_start_time - 1.0e-9
    ]
    if not window:
        window = [positions[-1]]
    final_displacement = math.hypot(
        window[-1][0] - window[0][0], window[-1][1] - window[0][1]
    )

    return StateMetrics(
        minimum_root_height_m=min(heights),
        minimum_pelvis_up_dot=min(up_dots),
        path_distance_m=path_distance,
        yaw_change_rad=abs(signed_yaw_change),
        final_stop_displacement_m=final_displacement,
    )


@dataclass(frozen=True)
class CommandPhaseReport:
    """Ordered forward / heading-change / final-stand phase presence flags."""

    has_forward: bool
    has_heading_change: bool
    has_final_stand: bool
    phases_ordered: bool


def _is_moving(command: CommandSample) -> bool:
    vx, vy, _vz = command.requested_velocity_mujoco
    return math.hypot(vx, vy) > _VELOCITY_TOLERANCE_MPS


def _is_forward(command: CommandSample) -> bool:
    return command.requested_velocity_mujoco[0] > _VELOCITY_TOLERANCE_MPS


def command_phase_report(commands: Sequence[CommandSample]) -> CommandPhaseReport:
    values = _validated_commands(commands)
    yaws = [_yaw_from_quaternion_wxyz(np.asarray(command.desired_heading_mujoco_wxyz))
            for command in values]

    forward_index: int | None = None
    for index, command in enumerate(values):
        if _is_forward(command):
            forward_index = index
            break

    heading_index: int | None = None
    for index in range(1, len(values)):
        if abs(math.remainder(yaws[index] - yaws[index - 1], 2.0 * math.pi)) > _HEADING_TOLERANCE_RAD:
            heading_index = index
            break

    final_stand_start: int | None = None
    if not _is_moving(values[-1]):
        final_stand_start = len(values) - 1
        while final_stand_start > 0 and not _is_moving(values[final_stand_start - 1]):
            final_stand_start -= 1

    has_forward = forward_index is not None
    has_heading_change = heading_index is not None
    has_final_stand = (
        final_stand_start is not None
        and len(values) - final_stand_start >= 5
    )
    phases_ordered = (
        has_forward
        and has_heading_change
        and has_final_stand
        and forward_index < heading_index  # type: ignore[operator]
        and heading_index <= final_stand_start  # type: ignore[operator]
    )
    return CommandPhaseReport(
        has_forward=has_forward,
        has_heading_change=has_heading_change,
        has_final_stand=has_final_stand,
        phases_ordered=phases_ordered,
    )


_STATE_ROW_COUNT = 600
_CANONICAL_FRAME_COUNT = _STATE_ROW_COUNT + 1
_FINAL_STOP_SECONDS = 2.0


def _validate_state_cadence(states: Sequence[object]) -> tuple[object, ...]:
    values = tuple(states)
    if len(values) != _STATE_ROW_COUNT:
        raise ContractError(
            f"manual run must contain exactly {_STATE_ROW_COUNT} state rows"
        )
    for index, state in enumerate(values):
        step = getattr(state, "step")
        sim_time = getattr(state, "sim_time_s")
        if type(step) is not int or step != 4 * (index + 1):
            raise ContractError("manual run state step sequence is not contiguous")
        expected_time = 0.02 * (index + 1)
        if (
            not isinstance(sim_time, (int, float))
            or isinstance(sim_time, bool)
            or not math.isfinite(float(sim_time))
            or not math.isclose(
                float(sim_time), expected_time, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ContractError("manual run state time sequence is not contiguous")
    return values


def _canonical_bits_equal(
    left: CanonicalTargetBuffer, right: CanonicalTargetBuffer
) -> bool:
    if left.count != right.count:
        return False
    for name in ("joint_position", "joint_velocity", "body_quat_w"):
        actual = np.ascontiguousarray(getattr(left, name), dtype="<f4").view(np.uint32)
        expected = np.ascontiguousarray(getattr(right, name), dtype="<f4").view(
            np.uint32
        )
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return False
    return np.array_equal(left.frame_index, right.frame_index)


def _generate_mm_buffer(
    commands: Sequence[CommandSample],
    *,
    run_root: Path,
    session_id: str,
    mm_server: Path,
    joint_contract: Path,
) -> CanonicalTargetBuffer:
    """Generate and commit one deterministic command sequence in isolation."""

    values = _validated_commands(commands)
    run_root.mkdir(parents=True)
    environment = dict(os.environ)
    python_root = str(_SONIC_ROOT / "python")
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_root if not prior else f"{python_root}{os.pathsep}{prior}"
    )
    environment["SONIC_TERRAIN_DIR"] = str(_DEFAULT_TERRAIN)
    contract = load_joint_contract(joint_contract)
    validator = SourceValidator(contract)
    with MMChunkClient(
        run_root=run_root,
        command=(str(mm_server),),
        stdout_archive=run_root / "mm.stdout",
        stderr_archive=run_root / "mm.stderr",
        env=environment,
        cwd=_REPOSITORY_ROOT,
    ) as client:
        client.hello()
        reset = client.reset(
            SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
            session_id=session_id,
        )
        timeline = TargetTimeline(validator.validate_initial(reset), contract)
        for command in values:
            candidate = f"{session_id}:candidate:{command.chunk_index:06d}"
            raw = client.generate(
                command,
                session_id=session_id,
                candidate_id=candidate,
                predecessor_id=timeline.last_accepted_candidate_id,
                source_intervals=10,
            )
            prepared = timeline.prepare(validator.validate_source(raw))
            client.commit(candidate)
            timeline.commit(prepared)
        return timeline.canonical_buffer


def qualify_and_replay_manual_commands(
    commands: Sequence[CommandSample],
    *,
    canonical_npz: bytes | bytearray | memoryview,
    mm_server: str | Path | None = None,
    joint_contract: str | Path | None = None,
) -> MMCommandReplay:
    """Qualify MM on the fixed route, then replay the recorded commands."""

    expected_baseline = load_canonical_target_npz(canonical_npz)
    if expected_baseline.count != _CANONICAL_FRAME_COUNT:
        raise ContractError("canonical target must contain frames 0..600")
    server = Path(
        _SONIC_ROOT / "build/mm_chunk_server"
        if mm_server is None
        else mm_server
    ).expanduser().resolve(strict=True)
    contract_path = Path(
        _SONIC_ROOT / "configs/g1_joint_contract.json"
        if joint_contract is None
        else joint_contract
    ).expanduser().resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="mm-sonic-manual-replay-") as temporary:
        root = Path(temporary)
        baseline = _generate_mm_buffer(
            flat_command_script(),
            run_root=root / "baseline",
            session_id="manual-audit-baseline",
            mm_server=server,
            joint_contract=contract_path,
        )
        replay = _generate_mm_buffer(
            commands,
            run_root=root / "recorded",
            session_id="manual-audit-recorded",
            mm_server=server,
            joint_contract=contract_path,
        )
    return MMCommandReplay(
        baseline_canonical_parity=_canonical_bits_equal(
            baseline, expected_baseline
        ),
        command_buffer=replay,
    )


@dataclass(frozen=True)
class ManualEvidence:
    """Explicit metrics and fail-closed pass booleans for a manual run."""

    state_rows_valid: bool
    pose_stream_valid: bool
    baseline_canonical_parity: bool
    command_replay_parity: bool
    no_fall_marker: bool
    pose_frame_count: int
    root_height_pass: bool
    upright_pass: bool
    path_pass: bool
    yaw_pass: bool
    final_stop_pass: bool
    hand_transport_exact: bool
    actuator_routing_pass: bool
    hand_tracking: HandTrackingReport
    hand_targets: Dex3HandTargets
    command_phases: CommandPhaseReport
    metrics: StateMetrics
    command_artifact: ManualCommandArtifact

    @property
    def passed(self) -> bool:
        return (
            self.state_rows_valid
            and self.pose_stream_valid
            and self.baseline_canonical_parity
            and self.command_replay_parity
            and self.no_fall_marker
            and self.command_phases.phases_ordered
            and self.root_height_pass
            and self.upright_pass
            and self.path_pass
            and self.yaw_pass
            and self.final_stop_pass
            and self.hand_transport_exact
            and self.actuator_routing_pass
            and self.hand_tracking.passed
        )


def audit_manual_run(
    *,
    states: Sequence[object],
    pose_frames: Sequence[bytes | bytearray | memoryview],
    mm_replay: MMCommandReplay,
    command_artifact: bytes | bytearray | memoryview,
    gear_log: bytes | bytearray | memoryview,
    hand_targets: Dex3HandTargets,
    hand_qpos_addresses: Sequence[int],
    hand_joint_ranges: Sequence[tuple[float, float]],
    actuator_routing_pass: bool,
) -> ManualEvidence:
    """Offline audit of one completed 12-second manual SONIC run.

    Structural forgeries (row count, cadence, stream contiguity, malformed
    archives, missing hand fields) raise :class:`ContractError`. Content
    boundaries (known-good MM qualification, command-conditioned replay parity,
    absence of the fall marker, ordered command phases, exact hand transport,
    actuator routing, settled hand tracking) are reported as pass booleans.
    """

    validated_states = _validate_state_cadence(states)
    artifact = parse_manual_command_artifact(command_artifact)
    if artifact.preload_chunks != 4 or len(artifact.commands) != 34:
        raise ContractError(
            "12-second manual run requires 4 preload and 30 operator commands"
        )
    if not isinstance(hand_targets, Dex3HandTargets):
        raise ContractError("manual run requires resolved Dex3 hand targets")
    if hand_targets_record(hand_targets) != hand_targets_record(artifact.hand_targets):
        raise ContractError("manual run hand targets disagree with the command artifact")
    if type(actuator_routing_pass) is not bool:
        raise ContractError("actuator routing pass must be a boolean")
    stream_buffer, hand_transport_exact = authenticate_pose_stream(
        pose_frames, expected_first_frame=0, hand_targets=hand_targets
    )
    expected_stream_count = 1 + 20 * len(artifact.commands)
    if stream_buffer.count != expected_stream_count:
        raise ContractError(
            "manual run pose stream must contain frames 0.."
            f"{expected_stream_count - 1}"
        )
    if not isinstance(mm_replay, MMCommandReplay):
        raise ContractError("manual run requires independent MM command replay")
    if mm_replay.command_buffer.count != expected_stream_count:
        raise ContractError(
            "MM command replay must contain the complete manual pose stream"
        )
    if not isinstance(gear_log, (bytes, bytearray, memoryview)):
        raise ContractError("gear log must be bytes")

    command_replay_parity = _canonical_bits_equal(
        stream_buffer, mm_replay.command_buffer
    )
    no_fall_marker = not scan_for_fall_marker(gear_log)
    phases = command_phase_report(artifact.commands)
    metrics = state_metrics(
        validated_states, final_stop_seconds=_FINAL_STOP_SECONDS
    )
    hand_tracking = measure_hand_tracking(
        validated_states,
        qpos_addresses=hand_qpos_addresses,
        joint_ranges=hand_joint_ranges,
        targets=hand_targets,
        final_seconds=1.0,
    )
    return ManualEvidence(
        state_rows_valid=True,
        pose_stream_valid=True,
        baseline_canonical_parity=mm_replay.baseline_canonical_parity,
        command_replay_parity=command_replay_parity,
        no_fall_marker=no_fall_marker,
        pose_frame_count=stream_buffer.count,
        root_height_pass=(
            metrics.minimum_root_height_m >= _MIN_ROOT_HEIGHT_M
        ),
        upright_pass=(
            metrics.minimum_pelvis_up_dot >= _MIN_PELVIS_UP_DOT
        ),
        path_pass=metrics.path_distance_m >= _MIN_PATH_DISTANCE_M,
        yaw_pass=metrics.yaw_change_rad >= _MIN_YAW_CHANGE_RAD,
        final_stop_pass=(
            metrics.final_stop_displacement_m
            <= _MAX_FINAL_STOP_DISPLACEMENT_M
        ),
        hand_transport_exact=hand_transport_exact,
        actuator_routing_pass=actuator_routing_pass,
        hand_tracking=hand_tracking,
        hand_targets=hand_targets,
        command_phases=phases,
        metrics=metrics,
        command_artifact=artifact,
    )


def _require_keys(value: object, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ContractError(f"{label} keys do not match the registered contract")
    return value


def _command_from_entry(
    entry: object, index: int, *, expected_origin: str
) -> CommandSample:
    mapping = _require_keys(
        entry,
        {
            "chunk_index",
            "origin",
            "requested_velocity_mujoco",
            "desired_heading_mujoco_wxyz",
        },
        f"manual command artifact commands[{index}]",
    )
    if mapping["origin"] != expected_origin:
        raise ContractError(
            f"manual command artifact commands[{index}] origin is invalid"
        )
    velocity = mapping["requested_velocity_mujoco"]
    heading = mapping["desired_heading_mujoco_wxyz"]
    if type(velocity) is not list or type(heading) is not list:
        raise ContractError(
            f"manual command artifact commands[{index}] vectors must be arrays"
        )
    try:
        return CommandSample(
            chunk_index=mapping["chunk_index"],
            requested_velocity_mujoco=tuple(velocity),
            desired_heading_mujoco_wxyz=tuple(heading),
        )
    except ContractError:
        raise
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"manual command artifact commands[{index}] is invalid"
        ) from error


def parse_manual_command_artifact(
    data: bytes | bytearray | memoryview,
) -> ManualCommandArtifact:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("manual command artifact must be bytes")
    try:
        document = json.loads(bytes(data).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("manual command artifact is not valid ASCII JSON") from error
    root = _require_keys(
        document,
        {"schema", "mode", "preload_chunks", "chunk_count", "hand_control", "commands"},
        "manual command artifact",
    )
    if root["schema"] != MANUAL_COMMAND_SCHEMA:
        raise ContractError("manual command artifact has an unsupported schema")
    mode = _validated_mode(root["mode"])
    hand_targets = parse_hand_targets_record(root["hand_control"])
    raw_commands = root["commands"]
    if type(raw_commands) is not list:
        raise ContractError("manual command artifact commands must be an array")
    if root["chunk_count"] != len(raw_commands):
        raise ContractError(
            "manual command artifact chunk_count does not match the commands"
        )
    preload_chunks = _validated_preload(
        root["preload_chunks"], len(raw_commands)
    )
    commands = _validated_commands(
        [
            _command_from_entry(
                entry,
                index,
                expected_origin=(
                    "preload" if index < preload_chunks else "operator"
                ),
            )
            for index, entry in enumerate(raw_commands)
        ]
    )
    return ManualCommandArtifact(
        schema=MANUAL_COMMAND_SCHEMA,
        mode=mode,
        preload_chunks=preload_chunks,
        commands=commands,
        hand_targets=hand_targets,
    )


def _read_regular(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ContractError(f"{label} is missing or unreadable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum_bytes
    ):
        raise ContractError(f"{label} must be one bounded regular file")
    try:
        data = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ContractError(f"{label} is unreadable") from error
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if len(data) != before.st_size or identity(before) != identity(after):
        raise ContractError(f"{label} changed while it was read")
    return data


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("ascii"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ContractError(f"{label} contains nonfinite value {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid ASCII JSON") from error
    if type(value) is not dict:
        raise ContractError(f"{label} must contain one JSON object")
    return value


_SCENE_CONTROL_KEYS = {
    "gear_scene_sha256",
    "gear_robot_sha256",
    "actuator_joint_order_sha256",
}
_SHA256_PATTERN = "0123456789abcdef"


def _validate_scene_control(value: object) -> dict[str, str]:
    mapping = _require_keys(value, _SCENE_CONTROL_KEYS, "manual summary scene control")
    for key in _SCENE_CONTROL_KEYS:
        digest = mapping[key]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in _SHA256_PATTERN for character in digest)
        ):
            raise ContractError(
                f"manual summary scene control {key} must be a lowercase SHA-256"
            )
    return mapping


_ENVIRONMENT_CONTROL_KEYS = {
    "scene_id",
    "route_id",
    "terrain_weight",
    "input_source",
    "mapper_version",
    "camera_sequence",
    "mm_hello_identity",
    "mm_scene_identity",
    "source_hashes",
    "output_hashes",
    "coordinate_source",
    "coordinate_target",
    "transform_matrix",
    "source_bounds_holden",
    "transformed_bounds_mujoco",
    "initial_boundary_sha256",
    "initial_qpos_sha256",
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_PATTERN for character in value)
    )


def _validate_digest(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ContractError(f"environment control {label} must be a lowercase SHA-256")
    return value


def _validate_nonempty_str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"environment control {label} must be a nonempty string")
    return value


def _validate_finite_float(value: object, label: str) -> float:
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"environment control {label} must be a finite number")
    return float(value)


def _validate_finite_matrix(
    value: object, rows: int, cols: int, label: str
) -> list[list[float]]:
    if type(value) is not list or len(value) != rows:
        raise ContractError(f"environment control {label} must be a {rows}x{cols} array")
    result: list[list[float]] = []
    for row in value:
        if type(row) is not list or len(row) != cols:
            raise ContractError(
                f"environment control {label} must be a {rows}x{cols} array"
            )
        result.append([_validate_finite_float(entry, label) for entry in row])
    return result


def _validate_hash_mapping(value: object, label: str) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ContractError(f"environment control {label} must be a nonempty object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if type(key) is not str or not key:
            raise ContractError(f"environment control {label} keys must be strings")
        result[key] = _validate_digest(digest, f"{label}.{key}")
    return result


def _validate_identity_mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or not value:
        raise ContractError(f"environment control {label} must be a nonempty object")
    for key in value:
        if type(key) is not str or not key:
            raise ContractError(f"environment control {label} keys must be strings")
    return dict(value)


def _validate_environment_control(mapping: dict) -> dict[str, object]:
    """Validate every bound field of one environment_control object."""

    validated: dict[str, object] = {
        "scene_id": _validate_nonempty_str(mapping["scene_id"], "scene_id"),
        "route_id": _validate_nonempty_str(mapping["route_id"], "route_id"),
        "terrain_weight": _validate_finite_float(
            mapping["terrain_weight"], "terrain_weight"
        ),
        "mapper_version": _validate_nonempty_str(
            mapping["mapper_version"], "mapper_version"
        ),
        "coordinate_source": _validate_nonempty_str(
            mapping["coordinate_source"], "coordinate_source"
        ),
        "coordinate_target": _validate_nonempty_str(
            mapping["coordinate_target"], "coordinate_target"
        ),
        "transform_matrix": _validate_finite_matrix(
            mapping["transform_matrix"], 3, 3, "transform_matrix"
        ),
        "source_bounds_holden": _validate_finite_matrix(
            mapping["source_bounds_holden"], 2, 3, "source_bounds_holden"
        ),
        "transformed_bounds_mujoco": _validate_finite_matrix(
            mapping["transformed_bounds_mujoco"], 2, 3, "transformed_bounds_mujoco"
        ),
        "mm_hello_identity": _validate_identity_mapping(
            mapping["mm_hello_identity"], "mm_hello_identity"
        ),
        "mm_scene_identity": _validate_identity_mapping(
            mapping["mm_scene_identity"], "mm_scene_identity"
        ),
        "source_hashes": _validate_hash_mapping(
            mapping["source_hashes"], "source_hashes"
        ),
        "output_hashes": _validate_hash_mapping(
            mapping["output_hashes"], "output_hashes"
        ),
        "initial_boundary_sha256": _validate_digest(
            mapping["initial_boundary_sha256"], "initial_boundary_sha256"
        ),
        "initial_qpos_sha256": _validate_digest(
            mapping["initial_qpos_sha256"], "initial_qpos_sha256"
        ),
    }
    if mapping["input_source"] not in _MANUAL_INPUT_SOURCES:
        raise ContractError("environment control input_source is unsupported")
    validated["input_source"] = mapping["input_source"]
    camera_sequence = mapping["camera_sequence"]
    if type(camera_sequence) is not int or camera_sequence < 0:
        raise ContractError(
            "environment control camera_sequence must be a nonnegative integer"
        )
    validated["camera_sequence"] = camera_sequence
    return validated


def environment_control_record(
    *,
    scene_id: str,
    route_id: str,
    terrain_weight: float,
    input_source: str,
    mapper_version: str,
    camera_sequence: int,
    mm_hello_identity: dict,
    mm_scene_identity: dict,
    source_hashes: dict,
    output_hashes: dict,
    coordinate_source: str,
    coordinate_target: str,
    transform_matrix: object,
    source_bounds_holden: object,
    transformed_bounds_mujoco: object,
    initial_boundary_sha256: str,
    initial_qpos_sha256: str,
) -> dict[str, object]:
    """Build one validated ``environment_control`` object for a v4 terrain run."""

    return _validate_environment_control(
        {
            "scene_id": scene_id,
            "route_id": route_id,
            "terrain_weight": terrain_weight,
            "input_source": input_source,
            "mapper_version": mapper_version,
            "camera_sequence": camera_sequence,
            "mm_hello_identity": mm_hello_identity,
            "mm_scene_identity": mm_scene_identity,
            "source_hashes": source_hashes,
            "output_hashes": output_hashes,
            "coordinate_source": coordinate_source,
            "coordinate_target": coordinate_target,
            "transform_matrix": transform_matrix,
            "source_bounds_holden": source_bounds_holden,
            "transformed_bounds_mujoco": transformed_bounds_mujoco,
            "initial_boundary_sha256": initial_boundary_sha256,
            "initial_qpos_sha256": initial_qpos_sha256,
        }
    )


def parse_environment_control(value: object) -> dict[str, object]:
    """Strictly parse one ``environment_control`` object, rejecting tampering."""

    mapping = _require_keys(value, _ENVIRONMENT_CONTROL_KEYS, "environment control")
    return _validate_environment_control(mapping)


_MANUAL_SUMMARY_V4_SCHEMA = "mm-sonic-manual-demo/v4"
_MANUAL_SUMMARY_V4_KEYS = {
    "schema",
    "mode",
    "run_root",
    "preload_chunks",
    "generated_chunks",
    "lookahead_seconds",
    "command_artifact",
    "hand_control",
    "environment_control",
    "snapshot",
}
_SNAPSHOT_KEYS = {"contact_rows", "sim_time_s", "state_rows", "steps"}


def _validate_snapshot(value: object) -> dict[str, object]:
    snapshot = _require_keys(value, _SNAPSHOT_KEYS, "manual summary snapshot")
    for key in ("contact_rows", "state_rows", "steps"):
        if type(snapshot[key]) is not int or snapshot[key] < 0:
            raise ContractError(
                f"manual summary snapshot {key} must be a nonnegative integer"
            )
    sim_time = snapshot["sim_time_s"]
    if (
        type(sim_time) not in (int, float)
        or isinstance(sim_time, bool)
        or not math.isfinite(float(sim_time))
    ):
        raise ContractError("manual summary snapshot sim_time_s must be finite")
    return dict(snapshot)


def manual_summary_v4_bytes(
    *,
    mode: str,
    run_root: str,
    preload_chunks: int,
    generated_chunks: int,
    lookahead_seconds: float,
    command_bytes: bytes,
    hand_targets: Dex3HandTargets,
    environment_control: dict,
    snapshot: dict,
) -> bytes:
    """Serialize one strict ``mm-sonic-manual-demo/v4`` terrain summary."""

    validated_mode = _validated_mode(mode)
    if type(run_root) is not str or not run_root:
        raise ContractError("manual summary run_root must be a nonempty string")
    if type(preload_chunks) is not int or preload_chunks < 0:
        raise ContractError("manual summary preload_chunks must be nonnegative")
    if type(generated_chunks) is not int or generated_chunks < preload_chunks:
        raise ContractError(
            "manual summary generated_chunks must be at least preload_chunks"
        )
    if (
        type(lookahead_seconds) not in (int, float)
        or isinstance(lookahead_seconds, bool)
        or not math.isfinite(float(lookahead_seconds))
    ):
        raise ContractError("manual summary lookahead_seconds must be finite")
    if not isinstance(command_bytes, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary command_bytes must be bytes")
    document = {
        "schema": _MANUAL_SUMMARY_V4_SCHEMA,
        "mode": validated_mode,
        "run_root": run_root,
        "preload_chunks": preload_chunks,
        "generated_chunks": generated_chunks,
        "lookahead_seconds": float(lookahead_seconds),
        "command_artifact": {
            "path": "manual-commands.json",
            "sha256": hashlib.sha256(bytes(command_bytes)).hexdigest(),
        },
        "hand_control": hand_targets_record(hand_targets),
        "environment_control": _validate_environment_control(
            dict(_require_keys(
                environment_control,
                _ENVIRONMENT_CONTROL_KEYS,
                "environment control",
            ))
        ),
        "snapshot": _validate_snapshot(snapshot),
    }
    try:
        return (
            json.dumps(document, sort_keys=True, indent=2) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("manual summary v4 cannot be serialized") from error


def parse_manual_summary_v4(
    data: bytes | bytearray | memoryview,
) -> dict[str, object]:
    """Strictly parse one ``mm-sonic-manual-demo/v4`` terrain summary."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary must be bytes")
    document = _json_object(bytes(data), label="manual summary")
    root = _require_keys(document, _MANUAL_SUMMARY_V4_KEYS, "manual summary")
    if root["schema"] != _MANUAL_SUMMARY_V4_SCHEMA:
        raise ContractError("manual summary has an unsupported schema")
    mode = _validated_mode(root["mode"])
    if type(root["run_root"]) is not str or not root["run_root"]:
        raise ContractError("manual summary run_root must be a nonempty string")
    if type(root["preload_chunks"]) is not int or root["preload_chunks"] < 0:
        raise ContractError("manual summary preload_chunks must be nonnegative")
    if (
        type(root["generated_chunks"]) is not int
        or root["generated_chunks"] < root["preload_chunks"]
    ):
        raise ContractError("manual summary generated_chunks is invalid")
    lookahead = root["lookahead_seconds"]
    if (
        type(lookahead) not in (int, float)
        or isinstance(lookahead, bool)
        or not math.isfinite(float(lookahead))
    ):
        raise ContractError("manual summary lookahead_seconds must be finite")
    command = _require_keys(
        root["command_artifact"], {"path", "sha256"}, "manual summary command artifact"
    )
    if command["path"] != "manual-commands.json" or not _is_sha256(
        command["sha256"]
    ):
        raise ContractError("manual summary command artifact identity is invalid")
    hand_targets = parse_hand_targets_record(root["hand_control"])
    environment_control = parse_environment_control(root["environment_control"])
    snapshot = _validate_snapshot(root["snapshot"])
    return {
        "schema": _MANUAL_SUMMARY_V4_SCHEMA,
        "mode": mode,
        "run_root": root["run_root"],
        "preload_chunks": root["preload_chunks"],
        "generated_chunks": root["generated_chunks"],
        "lookahead_seconds": float(lookahead),
        "command_artifact": {
            "path": command["path"],
            "sha256": command["sha256"],
        },
        "hand_control": hand_targets_record(hand_targets),
        "environment_control": environment_control,
        "snapshot": snapshot,
    }


_MANUAL_SUMMARY_V5_SCHEMA = "mm-sonic-manual-demo/v5"
_MANUAL_SUMMARY_V5_KEYS = _MANUAL_SUMMARY_V4_KEYS | {"movement_model"}
_MOVEMENT_MODEL_KEYS = {
    "profile",
    "acceleration_mps2",
    "deceleration_mps2",
    "directional_acceleration",
    "turn_strength",
}


def validate_movement_model_record(value: object) -> dict[str, object]:
    """Validate one server-authored movement_model object exactly."""

    record = _require_keys(value, _MOVEMENT_MODEL_KEYS, "movement_model")
    if record["profile"] not in ("raw", "holden-v1"):
        raise ContractError("movement_model profile must be raw or holden-v1")
    if record["acceleration_mps2"] != 1.5 or record["deceleration_mps2"] != 2.0:
        raise ContractError("movement_model parameters are not the fixed values")
    if (
        record["directional_acceleration"] is not False
        or record["turn_strength"] is not False
    ):
        raise ContractError(
            "movement_model must disable directional acceleration and turn strength"
        )
    return {
        "profile": record["profile"],
        "acceleration_mps2": 1.5,
        "deceleration_mps2": 2.0,
        "directional_acceleration": False,
        "turn_strength": False,
    }


def manual_summary_v5_bytes(
    *,
    mode: str,
    run_root: str,
    preload_chunks: int,
    generated_chunks: int,
    lookahead_seconds: float,
    command_bytes: bytes,
    hand_targets: Dex3HandTargets,
    environment_control: dict,
    snapshot: dict,
    movement_model: object,
) -> bytes:
    """Serialize a ``mm-sonic-manual-demo/v5`` summary (v4 plus movement model)."""

    base = manual_summary_v4_bytes(
        mode=mode,
        run_root=run_root,
        preload_chunks=preload_chunks,
        generated_chunks=generated_chunks,
        lookahead_seconds=lookahead_seconds,
        command_bytes=command_bytes,
        hand_targets=hand_targets,
        environment_control=environment_control,
        snapshot=snapshot,
    )
    document = json.loads(base)
    document["schema"] = _MANUAL_SUMMARY_V5_SCHEMA
    document["movement_model"] = validate_movement_model_record(movement_model)
    try:
        return (
            json.dumps(document, sort_keys=True, indent=2) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("manual summary v5 cannot be serialized") from error


def parse_manual_summary_v5(
    data: bytes | bytearray | memoryview,
) -> dict[str, object]:
    """Strictly parse one ``mm-sonic-manual-demo/v5`` summary."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary must be bytes")
    document = _json_object(bytes(data), label="manual summary")
    root = _require_keys(document, _MANUAL_SUMMARY_V5_KEYS, "manual summary")
    if root["schema"] != _MANUAL_SUMMARY_V5_SCHEMA:
        raise ContractError("manual summary has an unsupported schema")
    # Reuse the v4 body validation by parsing a v4 view of the same fields.
    v4_view = {key: root[key] for key in _MANUAL_SUMMARY_V4_KEYS}
    v4_view["schema"] = _MANUAL_SUMMARY_V4_SCHEMA
    parsed = parse_manual_summary_v4(
        (json.dumps(v4_view, sort_keys=True) + "\n").encode("ascii")
    )
    parsed["schema"] = _MANUAL_SUMMARY_V5_SCHEMA
    parsed["movement_model"] = validate_movement_model_record(
        root["movement_model"]
    )
    return parsed


_MANUAL_FLAT_SUMMARY_V6_SCHEMA = "mm-sonic-manual-demo/v6"
_MANUAL_FLAT_SUMMARY_V6_KEYS = {
    "schema",
    "mode",
    "run_root",
    "preload_chunks",
    "generated_chunks",
    "lookahead_seconds",
    "command_artifact",
    "hand_control",
    "scene_control",
    "snapshot",
    "movement_model",
}


def manual_flat_summary_v6_bytes(
    *,
    mode: str,
    run_root: str,
    preload_chunks: int,
    generated_chunks: int,
    lookahead_seconds: float,
    command_bytes: bytes,
    hand_targets: Dex3HandTargets,
    scene_control: dict,
    snapshot: dict,
    movement_model: object,
) -> bytes:
    """Serialize a flat-scene summary with server-authored model identity."""

    validated_mode = _validated_mode(mode)
    if type(run_root) is not str or not run_root:
        raise ContractError("manual summary run_root must be a nonempty string")
    if type(preload_chunks) is not int or preload_chunks < 0:
        raise ContractError("manual summary preload_chunks must be nonnegative")
    if type(generated_chunks) is not int or generated_chunks < preload_chunks:
        raise ContractError(
            "manual summary generated_chunks must be at least preload_chunks"
        )
    if (
        type(lookahead_seconds) not in (int, float)
        or isinstance(lookahead_seconds, bool)
        or not math.isfinite(float(lookahead_seconds))
    ):
        raise ContractError("manual summary lookahead_seconds must be finite")
    if not isinstance(command_bytes, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary command_bytes must be bytes")
    document = {
        "schema": _MANUAL_FLAT_SUMMARY_V6_SCHEMA,
        "mode": validated_mode,
        "run_root": run_root,
        "preload_chunks": preload_chunks,
        "generated_chunks": generated_chunks,
        "lookahead_seconds": float(lookahead_seconds),
        "command_artifact": {
            "path": "manual-commands.json",
            "sha256": hashlib.sha256(bytes(command_bytes)).hexdigest(),
        },
        "hand_control": hand_targets_record(hand_targets),
        "scene_control": _validate_scene_control(scene_control),
        "snapshot": _validate_snapshot(snapshot),
        "movement_model": validate_movement_model_record(movement_model),
    }
    try:
        return (
            json.dumps(document, sort_keys=True, indent=2) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("manual flat summary v6 cannot be serialized") from error


def parse_manual_flat_summary_v6(
    data: bytes | bytearray | memoryview,
) -> dict[str, object]:
    """Strictly parse one flat-scene ``mm-sonic-manual-demo/v6`` summary."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary must be bytes")
    document = _json_object(bytes(data), label="manual summary")
    root = _require_keys(
        document, _MANUAL_FLAT_SUMMARY_V6_KEYS, "manual summary"
    )
    if root["schema"] != _MANUAL_FLAT_SUMMARY_V6_SCHEMA:
        raise ContractError("manual summary has an unsupported schema")
    mode = _validated_mode(root["mode"])
    if type(root["run_root"]) is not str or not root["run_root"]:
        raise ContractError("manual summary run_root must be a nonempty string")
    if type(root["preload_chunks"]) is not int or root["preload_chunks"] < 0:
        raise ContractError("manual summary preload_chunks must be nonnegative")
    if (
        type(root["generated_chunks"]) is not int
        or root["generated_chunks"] < root["preload_chunks"]
    ):
        raise ContractError("manual summary generated_chunks is invalid")
    lookahead = root["lookahead_seconds"]
    if (
        type(lookahead) not in (int, float)
        or isinstance(lookahead, bool)
        or not math.isfinite(float(lookahead))
    ):
        raise ContractError("manual summary lookahead_seconds must be finite")
    command = _require_keys(
        root["command_artifact"],
        {"path", "sha256"},
        "manual summary command artifact",
    )
    if command["path"] != "manual-commands.json" or not _is_sha256(
        command["sha256"]
    ):
        raise ContractError("manual summary command artifact identity is invalid")
    hand_targets = parse_hand_targets_record(root["hand_control"])
    return {
        "schema": _MANUAL_FLAT_SUMMARY_V6_SCHEMA,
        "mode": mode,
        "run_root": root["run_root"],
        "preload_chunks": root["preload_chunks"],
        "generated_chunks": root["generated_chunks"],
        "lookahead_seconds": float(lookahead),
        "command_artifact": dict(command),
        "hand_control": hand_targets_record(hand_targets),
        "scene_control": _validate_scene_control(root["scene_control"]),
        "snapshot": _validate_snapshot(root["snapshot"]),
        "movement_model": validate_movement_model_record(
            root["movement_model"]
        ),
    }


_TURN_MOVEMENT_MODEL_KEYS = _MOVEMENT_MODEL_KEYS | {"max_yaw_rate_deg_s"}


def validate_turn_movement_model_record(value: object) -> dict[str, object]:
    """Validate one server-authored ``holden-turn-v1`` movement_model object."""

    record = _require_keys(value, _TURN_MOVEMENT_MODEL_KEYS, "movement_model")
    if record["profile"] != "holden-turn-v1":
        raise ContractError("movement_model profile must be holden-turn-v1")
    if record["acceleration_mps2"] != 1.5 or record["deceleration_mps2"] != 2.0:
        raise ContractError("movement_model parameters are not the fixed values")
    if (
        record["directional_acceleration"] is not False
        or record["turn_strength"] is not False
    ):
        raise ContractError(
            "movement_model must disable directional acceleration and turn strength"
        )
    max_yaw = record["max_yaw_rate_deg_s"]
    if type(max_yaw) is bool or type(max_yaw) not in (int, float):
        raise ContractError("movement_model max_yaw_rate_deg_s must be a number")
    if not math.isfinite(float(max_yaw)) or float(max_yaw) != 120.0:
        raise ContractError("movement_model max_yaw_rate_deg_s must equal 120.0")
    return {
        "profile": "holden-turn-v1",
        "acceleration_mps2": 1.5,
        "deceleration_mps2": 2.0,
        "directional_acceleration": False,
        "turn_strength": False,
        "max_yaw_rate_deg_s": 120.0,
    }


_MANUAL_SUMMARY_V7_SCHEMA = "mm-sonic-manual-demo/v7"
_MANUAL_SUMMARY_V7_KEYS = _MANUAL_SUMMARY_V5_KEYS


def manual_summary_v7_bytes(
    *,
    mode: str,
    run_root: str,
    preload_chunks: int,
    generated_chunks: int,
    lookahead_seconds: float,
    command_bytes: bytes,
    hand_targets: Dex3HandTargets,
    environment_control: dict,
    snapshot: dict,
    movement_model: object,
) -> bytes:
    """Serialize a terrain ``mm-sonic-manual-demo/v7`` turn-profile summary."""

    base = manual_summary_v4_bytes(
        mode=mode,
        run_root=run_root,
        preload_chunks=preload_chunks,
        generated_chunks=generated_chunks,
        lookahead_seconds=lookahead_seconds,
        command_bytes=command_bytes,
        hand_targets=hand_targets,
        environment_control=environment_control,
        snapshot=snapshot,
    )
    document = json.loads(base)
    document["schema"] = _MANUAL_SUMMARY_V7_SCHEMA
    document["movement_model"] = validate_turn_movement_model_record(
        movement_model
    )
    try:
        return (
            json.dumps(document, sort_keys=True, indent=2) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("manual summary v7 cannot be serialized") from error


def parse_manual_summary_v7(
    data: bytes | bytearray | memoryview,
) -> dict[str, object]:
    """Strictly parse one terrain ``mm-sonic-manual-demo/v7`` summary."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary must be bytes")
    document = _json_object(bytes(data), label="manual summary")
    root = _require_keys(document, _MANUAL_SUMMARY_V7_KEYS, "manual summary")
    if root["schema"] != _MANUAL_SUMMARY_V7_SCHEMA:
        raise ContractError("manual summary has an unsupported schema")
    v4_view = {key: root[key] for key in _MANUAL_SUMMARY_V4_KEYS}
    v4_view["schema"] = _MANUAL_SUMMARY_V4_SCHEMA
    parsed = parse_manual_summary_v4(
        (json.dumps(v4_view, sort_keys=True) + "\n").encode("ascii")
    )
    parsed["schema"] = _MANUAL_SUMMARY_V7_SCHEMA
    parsed["movement_model"] = validate_turn_movement_model_record(
        root["movement_model"]
    )
    return parsed


_MANUAL_FLAT_SUMMARY_V8_SCHEMA = "mm-sonic-manual-demo/v8"
_MANUAL_FLAT_SUMMARY_V8_KEYS = _MANUAL_FLAT_SUMMARY_V6_KEYS


def manual_flat_summary_v8_bytes(
    *,
    mode: str,
    run_root: str,
    preload_chunks: int,
    generated_chunks: int,
    lookahead_seconds: float,
    command_bytes: bytes,
    hand_targets: Dex3HandTargets,
    scene_control: dict,
    snapshot: dict,
    movement_model: object,
) -> bytes:
    """Serialize a flat-scene ``mm-sonic-manual-demo/v8`` turn-profile summary."""

    base = manual_flat_summary_v6_bytes(
        mode=mode,
        run_root=run_root,
        preload_chunks=preload_chunks,
        generated_chunks=generated_chunks,
        lookahead_seconds=lookahead_seconds,
        command_bytes=command_bytes,
        hand_targets=hand_targets,
        scene_control=scene_control,
        snapshot=snapshot,
        movement_model={
            "profile": "raw",
            "acceleration_mps2": 1.5,
            "deceleration_mps2": 2.0,
            "directional_acceleration": False,
            "turn_strength": False,
        },
    )
    document = json.loads(base)
    document["schema"] = _MANUAL_FLAT_SUMMARY_V8_SCHEMA
    document["movement_model"] = validate_turn_movement_model_record(
        movement_model
    )
    try:
        return (
            json.dumps(document, sort_keys=True, indent=2) + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("manual flat summary v8 cannot be serialized") from error


def parse_manual_flat_summary_v8(
    data: bytes | bytearray | memoryview,
) -> dict[str, object]:
    """Strictly parse one flat-scene ``mm-sonic-manual-demo/v8`` summary."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContractError("manual summary must be bytes")
    document = _json_object(bytes(data), label="manual summary")
    root = _require_keys(
        document, _MANUAL_FLAT_SUMMARY_V8_KEYS, "manual summary"
    )
    if root["schema"] != _MANUAL_FLAT_SUMMARY_V8_SCHEMA:
        raise ContractError("manual summary has an unsupported schema")
    mode = _validated_mode(root["mode"])
    if type(root["run_root"]) is not str or not root["run_root"]:
        raise ContractError("manual summary run_root must be a nonempty string")
    if type(root["preload_chunks"]) is not int or root["preload_chunks"] < 0:
        raise ContractError("manual summary preload_chunks must be nonnegative")
    if (
        type(root["generated_chunks"]) is not int
        or root["generated_chunks"] < root["preload_chunks"]
    ):
        raise ContractError("manual summary generated_chunks is invalid")
    lookahead = root["lookahead_seconds"]
    if (
        type(lookahead) not in (int, float)
        or isinstance(lookahead, bool)
        or not math.isfinite(float(lookahead))
    ):
        raise ContractError("manual summary lookahead_seconds must be finite")
    command = _require_keys(
        root["command_artifact"],
        {"path", "sha256"},
        "manual summary command artifact",
    )
    if command["path"] != "manual-commands.json" or not _is_sha256(
        command["sha256"]
    ):
        raise ContractError("manual summary command artifact identity is invalid")
    hand_targets = parse_hand_targets_record(root["hand_control"])
    return {
        "schema": _MANUAL_FLAT_SUMMARY_V8_SCHEMA,
        "mode": mode,
        "run_root": root["run_root"],
        "preload_chunks": root["preload_chunks"],
        "generated_chunks": root["generated_chunks"],
        "lookahead_seconds": float(lookahead),
        "command_artifact": dict(command),
        "hand_control": hand_targets_record(hand_targets),
        "scene_control": _validate_scene_control(root["scene_control"]),
        "snapshot": _validate_snapshot(root["snapshot"]),
        "movement_model": validate_turn_movement_model_record(
            root["movement_model"]
        ),
    }


def _validate_summary(
    summary: dict[str, object],
    *,
    run_root: Path,
    artifact: ManualCommandArtifact,
    command_bytes: bytes,
) -> tuple[Dex3HandTargets, dict[str, str]]:
    if summary.get("schema") != _MANUAL_SUMMARY_SCHEMA:
        raise ContractError("manual summary has an unsupported schema")
    root = _require_keys(
        summary,
        {
            "schema",
            "mode",
            "run_root",
            "preload_chunks",
            "generated_chunks",
            "lookahead_seconds",
            "command_artifact",
            "hand_control",
            "scene_control",
            "snapshot",
        },
        "manual summary",
    )
    if (
        root["schema"] != _MANUAL_SUMMARY_SCHEMA
        or root["mode"] != artifact.mode
        or root["run_root"] != str(run_root)
        or root["preload_chunks"] != _PRELOAD_CHUNKS
        or root["preload_chunks"] != artifact.preload_chunks
        or root["generated_chunks"]
        != _PRELOAD_CHUNKS + _OPERATOR_CHUNKS
        or root["generated_chunks"] != len(artifact.commands)
        or root["lookahead_seconds"] != 1.6
    ):
        raise ContractError("manual summary identity does not match the run")
    hand_targets = parse_hand_targets_record(root["hand_control"])
    if hand_targets_record(hand_targets) != hand_targets_record(artifact.hand_targets):
        raise ContractError("manual summary hand control disagrees with the commands")
    scene_control = _validate_scene_control(root["scene_control"])
    command = _require_keys(
        root["command_artifact"],
        {"path", "sha256"},
        "manual summary command artifact",
    )
    digest = hashlib.sha256(command_bytes).hexdigest()
    if command != {"path": "manual-commands.json", "sha256": digest}:
        raise ContractError("manual summary command artifact identity changed")
    snapshot = _require_keys(
        root["snapshot"],
        {"contact_rows", "sim_time_s", "state_rows", "steps"},
        "manual summary snapshot",
    )
    sim_time = snapshot["sim_time_s"]
    if (
        snapshot["contact_rows"] != 2401
        or snapshot["state_rows"] != _STATE_ROW_COUNT
        or snapshot["steps"] != 2401
        or type(sim_time) not in (int, float)
        or not math.isfinite(float(sim_time))
        or not math.isclose(
            float(sim_time), 12.005, rel_tol=0.0, abs_tol=1.0e-9
        )
    ):
        raise ContractError("manual summary snapshot is not a 12-second run")
    return hand_targets, scene_control


def _expected_archive_paths(
    run_root: Path, *, command_count: int
) -> tuple[Path, ...]:
    paths = [
        run_root
        / "transmitted/readiness/attempt-000001__000000-000000.bin"
    ]
    for chunk in range(command_count):
        first = 1 + chunk * _FRAMES_PER_CHUNK
        leaf = f"{first:06d}-{first + _FRAMES_PER_CHUNK - 1:06d}.bin"
        paths.append(
            run_root / "transmitted/logical" / leaf
            if chunk < _PRELOAD_CHUNKS
            else run_root / "transmitted" / leaf
        )
    return tuple(paths)


def _load_archived_pose_messages(
    run_root: Path, *, command_count: int
) -> tuple[bytes, ...]:
    transmitted = run_root / "transmitted"
    try:
        metadata = transmitted.lstat()
    except OSError as error:
        raise ContractError("manual run transmission archive is missing") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("manual run transmission archive must be a directory")
    expected_bins = _expected_archive_paths(
        run_root, command_count=command_count
    )
    expected_files = {
        path.relative_to(run_root).as_posix()
        for binary in expected_bins
        for path in (binary, binary.with_suffix(".sha256"))
    }
    actual_files: set[str] = set()
    try:
        entries = tuple(transmitted.rglob("*"))
    except OSError as error:
        raise ContractError("manual run transmission archive is unreadable") from error
    for entry in entries:
        try:
            observed = entry.lstat()
        except OSError as error:
            raise ContractError("manual run transmission archive changed") from error
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise ContractError("manual run transmission archive has an alias")
        actual_files.add(entry.relative_to(run_root).as_posix())
    if actual_files != expected_files:
        raise ContractError("manual run transmission archive layout changed")

    messages: list[bytes] = []
    for index, binary in enumerate(expected_bins):
        message = _read_regular(
            binary,
            label=f"pose archive {index}",
            maximum_bytes=1024 * 1024,
        )
        sidecar = _read_regular(
            binary.with_suffix(".sha256"),
            label=f"pose archive digest {index}",
            maximum_bytes=256,
        )
        expected = (
            f"{hashlib.sha256(message).hexdigest()}  {binary.name}\n"
        ).encode("ascii")
        if sidecar != expected:
            raise ContractError(f"pose archive digest {index} changed")
        messages.append(message)
    return tuple(messages)


def _authenticate_run_local_scene(
    run_root: Path, scene_control: dict[str, str]
) -> tuple[tuple[int, ...], tuple[tuple[float, float], ...], bool]:
    """Verify the run-local scene bytes and loaded actuator routing.

    Returns the 14 hand qpos addresses, their joint ranges, and whether the
    loaded actuator routing satisfies the BaseSimulator slot invariant.
    """

    import xml.etree.ElementTree as ET

    from .scene import (
        SceneError,
        normalize_run_local_actuators,
        verify_loaded_actuator_routing,
    )

    scene_path = run_root / "scene/gear_scene.xml"
    robot_path = run_root / "scene/gear_robot.xml"
    scene_bytes = _read_regular(
        scene_path, label="run-local scene XML", maximum_bytes=16 * 1024 * 1024
    )
    robot_bytes = _read_regular(
        robot_path, label="run-local robot XML", maximum_bytes=16 * 1024 * 1024
    )
    if hashlib.sha256(scene_bytes).hexdigest() != scene_control["gear_scene_sha256"]:
        raise ContractError("run-local scene SHA-256 disagrees with scene control")
    if hashlib.sha256(robot_bytes).hexdigest() != scene_control["gear_robot_sha256"]:
        raise ContractError("run-local robot SHA-256 disagrees with scene control")
    try:
        _, _, actuator_sha = normalize_run_local_actuators(
            robot_bytes, label="run-local robot XML"
        )
    except SceneError as error:
        raise ContractError(f"run-local robot actuator order is invalid: {error}") from error
    if actuator_sha != scene_control["actuator_joint_order_sha256"]:
        raise ContractError(
            "run-local actuator joint order disagrees with scene control"
        )
    scene_root = ET.fromstring(scene_bytes)
    includes = scene_root.findall("include")
    if len(includes) != 1 or includes[0].attrib.get("file") != str(robot_path):
        raise ContractError(
            "run-local scene include must point to its own gear_robot.xml"
        )

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_path))
    except (ImportError, ValueError, OSError) as error:
        raise ContractError(f"run-local scene does not load: {error}") from error
    try:
        verify_loaded_actuator_routing(model)
        actuator_routing_pass = True
    except SceneError:
        actuator_routing_pass = False
    addresses = hand_qpos_addresses(model)
    ranges = hand_joint_ranges(model)
    return addresses, ranges, actuator_routing_pass


def audit_manual_bundle(
    run_root: str | Path,
    canonical_target: str | Path,
    *,
    replay: MMCommandReplay | None = None,
) -> ManualEvidence:
    """Authenticate a run and replay its commands through an isolated MM server."""

    root = Path(run_root).expanduser()
    try:
        root_metadata = root.lstat()
        root = root.resolve(strict=True)
    except OSError as error:
        raise ContractError("manual run root is missing") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ContractError("manual run root must be a real directory")

    command_bytes = _read_regular(
        root / "manual-commands.json",
        label="manual command artifact",
        maximum_bytes=1024 * 1024,
    )
    artifact = parse_manual_command_artifact(command_bytes)
    summary = _json_object(
        _read_regular(
            root / "manual-summary.json",
            label="manual summary",
            maximum_bytes=1024 * 1024,
        ),
        label="manual summary",
    )
    hand_targets, scene_control = _validate_summary(
        summary,
        run_root=root,
        artifact=artifact,
        command_bytes=command_bytes,
    )
    hand_addresses, hand_ranges, actuator_routing_pass = _authenticate_run_local_scene(
        root, scene_control
    )

    state_path = root / "scored-sim-logs/state.jsonl"
    _read_regular(
        state_path,
        label="scored state stream",
        maximum_bytes=64 * 1024 * 1024,
    )
    states = load_state_stream(
        state_path,
        expected_count=_STATE_ROW_COUNT,
        nq=50,
        nv=49,
    )
    pose_frames = _load_archived_pose_messages(
        root, command_count=len(artifact.commands)
    )
    simulator_log = _read_regular(
        root / "simulator.stderr",
        label="simulator stderr",
        maximum_bytes=16 * 1024 * 1024,
    )
    canonical_bytes = _read_regular(
        Path(canonical_target).expanduser(),
        label="canonical target",
        maximum_bytes=64 * 1024 * 1024,
    )
    if replay is None:
        replay = qualify_and_replay_manual_commands(
            artifact.commands,
            canonical_npz=canonical_bytes,
        )
    return audit_manual_run(
        states=states,
        pose_frames=pose_frames,
        mm_replay=replay,
        command_artifact=command_bytes,
        gear_log=simulator_log,
        hand_targets=hand_targets,
        hand_qpos_addresses=hand_addresses,
        hand_joint_ranges=hand_ranges,
        actuator_routing_pass=actuator_routing_pass,
    )


def manual_evidence_document(evidence: ManualEvidence) -> dict[str, object]:
    if not isinstance(evidence, ManualEvidence):
        raise ContractError("manual evidence document requires ManualEvidence")
    return {
        "schema": "mm-sonic-manual-evidence/v3",
        "passed": evidence.passed,
        "mode": evidence.command_artifact.mode,
        "preload_chunks": evidence.command_artifact.preload_chunks,
        "command_count": len(evidence.command_artifact.commands),
        "pose_frame_count": evidence.pose_frame_count,
        "hand_control": hand_targets_record(evidence.hand_targets),
        "checks": {
            "state_rows_valid": evidence.state_rows_valid,
            "pose_stream_valid": evidence.pose_stream_valid,
            "mm_server_matches_fixed_canonical": (
                evidence.baseline_canonical_parity
            ),
            "recorded_commands_replay_bit_exact": (
                evidence.command_replay_parity
            ),
            "no_fall_marker": evidence.no_fall_marker,
            "command_phases_ordered": evidence.command_phases.phases_ordered,
            "root_height": evidence.root_height_pass,
            "upright": evidence.upright_pass,
            "path_distance": evidence.path_pass,
            "yaw_change": evidence.yaw_pass,
            "final_stop": evidence.final_stop_pass,
            "hand_transport_exact": evidence.hand_transport_exact,
            "actuator_routing": evidence.actuator_routing_pass,
            "hand_tracking": evidence.hand_tracking.passed,
        },
        "command_phases": {
            "has_forward": evidence.command_phases.has_forward,
            "has_heading_change": evidence.command_phases.has_heading_change,
            "has_final_stand": evidence.command_phases.has_final_stand,
        },
        "hand_tracking": {
            "median_absolute_error_rad": list(
                evidence.hand_tracking.median_absolute_error_rad
            ),
            "median_position_rad": list(
                evidence.hand_tracking.median_position_rad
            ),
            "unexpected_limit_joints": list(
                evidence.hand_tracking.unexpected_limit_joints
            ),
        },
        "metrics": {
            "minimum_root_height_m": evidence.metrics.minimum_root_height_m,
            "minimum_pelvis_up_dot": evidence.metrics.minimum_pelvis_up_dot,
            "path_distance_m": evidence.metrics.path_distance_m,
            "yaw_change_rad": evidence.metrics.yaw_change_rad,
            "final_stop_displacement_m": (
                evidence.metrics.final_stop_displacement_m
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--canonical-target", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    document = manual_evidence_document(
        audit_manual_bundle(args.run_root, args.canonical_target)
    )
    encoded = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("ascii")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    print(encoded.decode("ascii"), end="")
    return 0 if document["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
