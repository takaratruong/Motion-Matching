"""Non-scored rolling Motion Matching -> SONIC operator demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable

import numpy as np

from .artifacts import RunBundle
from .cli import (
    _drive_simulator_until,
    _reset_and_prime_scored_epoch,
    _stream_gear_command,
)
import xml.etree.ElementTree as ET

from .commands import CommandSample, flat_command_script
from .coordinator import SessionConfig, SourceValidator
from .hands import Dex3HandTargets, NEUTRAL_HAND_TARGETS, hand_targets_record
from .holden_control import HoldenControlMapper, MappedControlState
from .joints import ContractError, load_joint_contract
from .manual_evidence import (
    environment_control_record,
    manual_command_artifact_bytes,
    manual_flat_summary_v6_bytes,
    manual_flat_summary_v8_bytes,
    manual_summary_v5_bytes,
    manual_summary_v7_bytes,
)
from .operator import OperatorLimits, OperatorSampler
from .operator_x11 import ContinuousControlLoop, X11KeyStateProvider
from .responsive_scheduler import ResponsiveScheduler
from .responsive_wiring import (
    ManualChunkCommitter,
    StateLogRootReader,
    build_boundary_trace,
    trace_record,
)
from .scene import (
    HOLDEN_COORDINATE_SIGNATURE,
    MUJOCO_COORDINATE_SIGNATURE,
    normalize_run_local_actuators,
    register_scene,
    verify_loaded_actuator_routing,
)
from .scene_runtime import (
    GEAR_ROBOT_RELATIVE,
    GEAR_ROBOT_SHA256,
    GEAR_SCENE_RELATIVE,
    GEAR_SCENE_SHA256,
    SCENE_REGISTRY_PATH,
    initial_physics_state,
)
from .operator_terminal import TerminalInputReader, TerminalKeyBuffer
from .process import (
    ChildProcessDied,
    GatedSimulatorClient,
    GearProcess,
    MMChunkClient,
    ProcessProtocolError,
    SimulationPolicyGate,
)
from .timeline import TargetTimeline
from .transform import holden_to_mujoco_quaternions
from .zmq_v1 import PosePublisher

MANUAL_MAPPER_VERSION = "holden-control/v1"


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SONIC_ROOT = _REPOSITORY_ROOT / "sonic"
_DEFAULT_SOURCE_RUN = Path(
    "/home/ubuntu/mm-flat-walk-stage-b-r16-hold-resume.tym9Fc/stage-b/"
    "stage-b-20260718T005901613783Z-8b63d477"
)
_DEFAULT_GEAR = Path("/tmp/groot-wbc-plan-inspect")
_DEFAULT_RUNTIME = Path(
    "/home/ubuntu/.local/share/motion-matching-deps/gear-sonic/"
    "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
)
_DEFAULT_TERRAIN = Path(
    "/home/ubuntu/projects/motion-matching/resources/g1_terrain"
)
_PRELOAD_CHUNKS = 4
_CHUNK_DURATION_S = 0.4
_SOURCE_RATE_HZ = 25
_SUPPORTED_SOURCE_INTERVALS = (5, 10)


def _wait_for_x11_target(
    provider: X11KeyStateProvider,
    control_loop: ContinuousControlLoop,
    *,
    timeout_s: float = 2.0,
) -> None:
    """Wait until a sampled focused viewer already has passive key grabs."""

    deadline = time.monotonic() + float(timeout_s)
    sequence = 1
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not control_loop.wait_for_sequence(
            sequence, timeout_s=remaining
        ):
            break
        if provider.target_bound:
            return
        sequence += 1
    raise ContractError(
        "continuous X11 control did not acquire the focused MuJoCo viewer"
    )


def _horizon_seconds(source_intervals: int) -> float:
    """Return the exact matched horizon duration for a source-interval count."""

    return source_intervals / _SOURCE_RATE_HZ


def _print_terminal_event(event: str) -> None:
    print(event, flush=True)


def _validated_preload_chunks(value: object) -> int:
    """Return an exact preload depth in 1..4, rejecting everything else."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"preload chunks must be an int in 1..4, got {value!r}"
        )
    if not 1 <= value <= 4:
        raise ContractError(
            f"preload chunks must be an int in 1..4, got {value!r}"
        )
    return value


def _responsive_preload_chunks(default_preload_chunks: int) -> int:
    """Responsive mode commits exactly one matched-horizon prefix.

    The opt-in responsive path drops lookahead to a single irrevocable chunk
    regardless of the resolved interactive default; the default two-chunk path
    never calls this.
    """

    _validated_preload_chunks(default_preload_chunks)
    return 1


def _resolve_responsive_source_intervals(namespace: object) -> int:
    """Return the validated matched source-interval horizon for this run.

    The default 10-interval 0.4s horizon is always allowed.  The opt-in
    5-interval 0.2s horizon is valid only for the interactive X11 responsive
    path; requesting it anywhere else is rejected before any run bundle is
    materialized.
    """

    source_intervals = getattr(namespace, "responsive_source_intervals", 10)
    if (
        type(source_intervals) is not int
        or source_intervals not in _SUPPORTED_SOURCE_INTERVALS
    ):
        raise ContractError(
            "--responsive-source-intervals must be one of "
            + " or ".join(str(count) for count in _SUPPORTED_SOURCE_INTERVALS)
        )
    if source_intervals == 10:
        return 10
    responsive = bool(getattr(namespace, "responsive", False))
    if (
        not responsive
        or getattr(namespace, "mode", None) != "interactive"
        or getattr(namespace, "input_source", None) != "x11"
    ):
        raise ContractError(
            "a responsive 5 source-interval 0.2s horizon is valid only for "
            "--responsive interactive X11 mode"
        )
    return source_intervals


class CommandRecorder:
    """Collect every committed preload/operator command in exact chunk order."""

    def __init__(
        self,
        *,
        mode: str,
        preload_chunks: int,
        hand_targets: Dex3HandTargets,
    ) -> None:
        self._mode = mode
        self._preload_chunks = preload_chunks
        self._hand_targets = hand_targets
        self._commands: list[CommandSample] = []

    def record(self, command: CommandSample) -> None:
        if not isinstance(command, CommandSample):
            raise ContractError("recorded command must be a CommandSample")
        if command.chunk_index != len(self._commands):
            raise ContractError(
                f"recorded command chunk {command.chunk_index} is out of order; "
                f"expected {len(self._commands)}"
            )
        self._commands.append(command)

    def artifact_bytes(self) -> bytes:
        return manual_command_artifact_bytes(
            mode=self._mode,
            preload_chunks=self._preload_chunks,
            commands=self._commands,
            hand_targets=self._hand_targets,
        )


def _utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"manual-{stamp}-{os.getpid()}"


def _environment(terrain_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    python_root = str(_SONIC_ROOT / "python")
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        python_root if not prior else f"{python_root}{os.pathsep}{prior}"
    )
    environment["SONIC_TERRAIN_DIR"] = str(terrain_dir)
    return environment


def _copy_scene(bundle: RunBundle, source_run: Path) -> tuple[Path, dict[str, str]]:
    """Materialize a genuinely run-local, actuator-normalized scene.

    Reads the source overlay and robot, normalizes only the copied robot's
    actuator element order, rewrites the copied overlay's sole include to the
    new local robot, loads the result to verify BaseSimulator routing, and
    never mutates the supplied source run.
    """

    source_scene = (source_run / "scene/gear_scene.xml").read_bytes()
    source_robot = (source_run / "scene/gear_robot.xml").read_bytes()
    robot_bytes, _, actuator_sha = normalize_run_local_actuators(
        source_robot, label="run-local robot XML"
    )
    bundle.write_bytes("scene/gear_robot.xml", robot_bytes)
    robot_path = bundle.path / "scene/gear_robot.xml"

    scene_root = ET.fromstring(source_scene)
    includes = scene_root.findall("include")
    if len(includes) != 1 or set(includes[0].attrib) != {"file"}:
        raise ContractError("source scene must contain exactly one robot include")
    includes[0].attrib["file"] = str(robot_path)
    scene_bytes = ET.tostring(
        scene_root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    ) + b"\n"
    bundle.write_bytes("scene/gear_scene.xml", scene_bytes)
    scene_path = bundle.path / "scene/gear_scene.xml"

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    verify_loaded_actuator_routing(model)

    scene_control = {
        "gear_scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
        "gear_robot_sha256": hashlib.sha256(robot_bytes).hexdigest(),
        "actuator_joint_order_sha256": actuator_sha,
    }
    return scene_path, scene_control


def _initial_qpos(scene_xml: Path) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    return np.asarray(model.qpos0, dtype=np.float64).copy()


def _stand_command(index: int, heading: CommandSample | None = None) -> CommandSample:
    quaternion = (
        (1.0, 0.0, 0.0, 0.0)
        if heading is None
        else heading.desired_heading_mujoco_wxyz
    )
    return CommandSample(index, (0.0, 0.0, 0.0), quaternion)


@dataclass(frozen=True)
class DemoDependencies:
    """The two authenticated environment builders used by startup.

    Production binds the complete registry, model, contract, and closed-hand
    arguments in these callables. Tests can replace only the external builders
    while exercising the real ordering transaction.
    """

    register_scene: Callable[..., object]
    build_initial_state: Callable[[object, object], object]


@dataclass(frozen=True)
class StartupTransaction:
    hello: object
    reset: object
    initial_boundary: object
    scene: object
    initial_state: object


def _require_supported_movement_model(hello: object, movement_model: str) -> None:
    """Fail closed before reset if the server cannot honor the profile."""

    supported = None
    if type(hello) is dict:
        supported = hello.get("supported_movement_models")
    if (
        type(supported) is not list
        or not supported
        or any(type(name) is not str for name in supported)
        or len(set(supported)) != len(supported)
    ):
        raise ValueError(
            "MM hello must advertise a unique list of movement models"
        )
    if movement_model not in supported:
        raise ValueError(
            f"MM server does not support movement model {movement_model!r}"
        )


def _run_startup_transaction(
    dependencies: DemoDependencies,
    *,
    mm: object,
    simulator: object,
    validator: object,
    scene_id: str,
    route_id: str,
    terrain_weight: float,
    session_id: str,
    log_dir: Path,
    movement_model: str = "raw",
) -> StartupTransaction:
    """Run the fail-closed MM -> scene -> initial-state -> simulator order."""

    hello = mm.hello()
    _require_supported_movement_model(hello, movement_model)
    reset = mm.reset(
        SessionConfig(scene_id, route_id, terrain_weight, movement_model),
        session_id=session_id,
    )
    scene_identity = reset["scene"]
    scene = dependencies.register_scene(
        scene_id,
        route_id,
        mm_hello_identity=hello,
        mm_scene_identity=scene_identity,
    )
    initial = validator.validate_initial(reset)
    initial_state = dependencies.build_initial_state(scene, initial)
    simulator.hello()
    simulator.reset(
        scene_xml=scene.gear_scene_xml,
        initial_qpos=initial_state.qpos,
        lateral_offset_m=0.0,
        yaw_offset_rad=0.0,
        log_dir=log_dir,
        elastic_band_enabled=True,
    )
    return StartupTransaction(hello, reset, initial, scene, initial_state)


def _activate_scored_control(
    gear: object,
    simulator: object,
    *,
    before_control: Callable[[], None] | None = None,
) -> SimulationPolicyGate:
    """Prepare one policy action before any scored physics release."""

    if before_control is not None:
        before_control()
    gear.continue_group()
    gear.activate_control()
    ready = gear.wait_for_first_policy_action()
    print(
        "SONIC first action ready: "
        f"index={ready['index']} policy_time={ready['time_ms']:.3f}ms",
        flush=True,
    )
    received = gear.wait_for_received_policy_command(simulator)
    print(
        "SONIC policy command received: "
        f"index={received['index']} "
        f"policy_time={received['time_ms']:.3f}ms",
        flush=True,
    )
    # Fence only GEAR's policy/reference workers. DDS LowState handling and the
    # command writer stay live throughout arbitrarily long MM requests.
    gate = SimulationPolicyGate(
        gear,
        simulator,
        pause_strategy="control-channel",
    )
    gate.pause()
    return gate


@dataclass(frozen=True)
class CameraDeliveryState:
    """Fail-closed state for the optional, synchronized camera channel."""

    enabled: bool = True
    last_sequence: int | None = None
    disabled_reported: bool = False


def _initial_camera_delivery_state(onscreen: bool) -> CameraDeliveryState:
    if type(onscreen) is not bool:
        raise ContractError("onscreen must be a boolean")
    return CameraDeliveryState(enabled=onscreen)


@dataclass(frozen=True)
class X11BoundaryResult:
    command: CommandSample | None
    mapped: MappedControlState
    camera_state: CameraDeliveryState
    advance: object | None


def _heading_yaw_rad(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion_wxyz
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _heading_frame_offset_yaw_rad(
    physical_mujoco_wxyz: tuple[float, float, float, float],
    virtual_mujoco_wxyz: tuple[float, float, float, float],
) -> float:
    """Return the authenticated physical-minus-virtual root yaw offset."""

    return math.remainder(
        _heading_yaw_rad(physical_mujoco_wxyz)
        - _heading_yaw_rad(virtual_mujoco_wxyz),
        2.0 * math.pi,
    )


def _deliver_camera(
    *,
    simulator: object,
    mapped: MappedControlState,
    camera_state: CameraDeliveryState,
    event_sink: Callable[[str], None],
    camera_disabled_prefix: str,
) -> CameraDeliveryState:
    """Deliver one synchronized camera update with fail-closed ack semantics.

    Only an exact-object/echo error (message prefixed ``camera data``) proves
    the JSONL response was fully consumed; it disables the camera once. Any
    other ``ProcessProtocolError`` is fatal because the request/response stream
    may no longer be synchronized.
    """

    next_camera_state = camera_state
    camera = mapped.camera
    if camera_state.enabled and camera.sequence != camera_state.last_sequence:
        try:
            simulator.set_camera(
                camera.sequence,
                math.degrees(camera.azimuth_rad),
                -math.degrees(camera.altitude_rad),
                camera.distance_m,
            )
        except ProcessProtocolError as error:
            if not str(error).startswith("camera data"):
                raise
            if not camera_state.disabled_reported:
                event_sink(f"{camera_disabled_prefix}: {error}")
            next_camera_state = CameraDeliveryState(
                enabled=False,
                last_sequence=camera_state.last_sequence,
                disabled_reported=True,
            )
        else:
            next_camera_state = CameraDeliveryState(
                enabled=True,
                last_sequence=camera.sequence,
                disabled_reported=camera_state.disabled_reported,
            )
    return next_camera_state


def _consume_x11_boundary(
    *,
    control_loop: object,
    simulator: object,
    gate: object,
    generate_and_publish: Callable[..., None],
    chunk_index: int,
    steps_per_chunk: int,
    preload_chunks: int,
    camera_state: CameraDeliveryState,
    event_sink: Callable[[str], None],
    control_prefix: str,
    camera_disabled_prefix: str,
    prefix_duration_s: float = _CHUNK_DURATION_S,
) -> X11BoundaryResult:
    """Atomically forward one mapped control boundary and synchronized camera."""

    command, mapped = control_loop.mailbox.sample(chunk_index)
    if command is None:
        return X11BoundaryResult(None, mapped, camera_state, None)

    next_camera_state = _deliver_camera(
        simulator=simulator,
        mapped=mapped,
        camera_state=camera_state,
        event_sink=event_sink,
        camera_disabled_prefix=camera_disabled_prefix,
    )

    frame_end = generate_and_publish(command, wait=False)
    advance = gate.release_steps(
        steps_per_chunk, expected_stream_frame_end=frame_end
    )
    velocity = command.requested_velocity_mujoco
    heading = _heading_yaw_rad(command.desired_heading_mujoco_wxyz)
    event_sink(
        f"{control_prefix}{command.chunk_index:06d} "
        f"vx={velocity[0]:+.3f} vy={velocity[1]:+.3f} "
        f"heading={heading:+.3f} strafe={int(mapped.strafe)} "
        f"walk={int(mapped.walk_blend >= 0.5)} "
        f"presents_in={preload_chunks * prefix_duration_s:.3f}s"
    )
    return X11BoundaryResult(command, mapped, next_camera_state, advance)


def _write_responsive_evidence(
    bundle: object,
    responsive: bool,
    traces: list[object],
    *,
    committed_prefixes: int | None = None,
    source_intervals: int = 10,
    movement_model: object = None,
) -> dict | None:
    """Write append-only responsive trace JSONL and an honest evidence summary.

    Returns ``None`` on the default (non-responsive) path so nothing is written.
    Each committed prefix contributes one JSONL line; the summary tallies how
    many boundaries carried a real observed root measurement versus how many had
    it unavailable, never fabricating a zero for the unavailable ones.
    """

    if not responsive:
        return None
    if source_intervals not in _SUPPORTED_SOURCE_INTERVALS:
        raise ContractError(
            "responsive evidence source_intervals must be one of "
            + " or ".join(str(count) for count in _SUPPORTED_SOURCE_INTERVALS)
        )
    if committed_prefixes is None:
        committed_prefixes = len(traces)
    if (
        type(committed_prefixes) is not int
        or committed_prefixes < len(traces)
    ):
        raise ContractError(
            "responsive committed_prefixes must cover all presented traces"
        )
    records = [trace_record(trace) for trace in traces]
    lines = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    bundle.write_text("responsive-boundary-trace.jsonl", lines)
    available = sum(1 for record in records if record["observed_root_available"])
    evidence = {
        "schema": "mm-sonic-responsive-evidence/v1",
        "trace_path": "responsive-boundary-trace.jsonl",
        "committed_prefixes": committed_prefixes,
        "presented_prefixes": len(records),
        "queued_prefixes_unobserved": committed_prefixes - len(records),
        "observed_root_available": available,
        "observed_root_unavailable": len(records) - available,
        "source_intervals": source_intervals,
        # Stage R1 honest floor: one irrevocable matched-horizon prefix plus MM
        # generation time. wait=False publication returns the socket send, not a
        # GEAR stream-processing acknowledgement (WAIT-phase only). The lookahead
        # is the true dynamic horizon = source_intervals / 25 s.
        "lookahead_seconds": _horizon_seconds(source_intervals),
        # The server-authored selected profile and fixed parameters, recorded
        # verbatim as honest evidence of the reference model that was applied.
        "movement_model": movement_model,
    }
    bundle.write_text(
        "responsive-evidence.json",
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
    )
    return evidence


def _run_responsive_x11_loop(
    *,
    control_loop: object,
    committer: object,
    simulator: object,
    chunks: int,
    camera_state: CameraDeliveryState,
    event_sink: Callable[[str], None],
    trace_sink: Callable[[object], None],
    session_id: str,
    camera_disabled_prefix: str,
) -> CameraDeliveryState:
    """Drive the opt-in Stage-R1 one-prefix responsive loop.

    Each iteration delivers the synchronized camera at sample time (via the
    scheduler's ``on_sample`` callback, preserving the consumed-response and
    camera-disable semantics) and commits exactly one prefix through the
    ``ManualChunkCommitter``.  A terminate snapshot (X) returns ``None`` from the
    scheduler and breaks the loop without generation, publication, or release.
    An honest ``BoundaryTrace`` is emitted per committed prefix; assembling it
    never alters command execution.
    """

    camera_box: list[CameraDeliveryState] = [camera_state]

    def on_sample(_snapshot: object, mapped: object) -> None:
        # Camera delivery is synchronous and identical to the default path.
        camera_box[0] = _deliver_camera(
            simulator=simulator,
            mapped=mapped,
            camera_state=camera_box[0],
            event_sink=event_sink,
            camera_disabled_prefix=camera_disabled_prefix,
        )

    scheduler = ResponsiveScheduler(
        committer, control_loop.mailbox, on_sample=on_sample
    )
    pending_prefix: object | None = None
    for _consumed_chunk in range(chunks):
        prefix = scheduler.run_one_prefix(committer.next_chunk)
        if prefix is None:
            # Terminate (X): no generation, publication, or physics release.
            break
        if pending_prefix is not None:
            # Exactly one prefix is irrevocably queued. The release performed
            # by this transaction physically presents the prior transaction's
            # prefix, so bind its observation to that prior input/prefix.
            trace = build_boundary_trace(
                pending_prefix,
                pending_prefix.accepted,
                release=prefix.accepted,
                session_id=session_id,
            )
            trace_sink(trace)
        pending_prefix = prefix
    return camera_box[0]


def run_demo(namespace: argparse.Namespace) -> Path:
    responsive = bool(getattr(namespace, "responsive", False))
    if responsive and (
        namespace.mode != "interactive" or namespace.input_source != "x11"
    ):
        raise ContractError(
            "--responsive is valid only for interactive X11 mode"
        )
    # Validate the matched source-interval horizon before any run bundle: the
    # opt-in 5-interval 0.2s prefix is valid only under --responsive X11.
    source_intervals = _resolve_responsive_source_intervals(namespace)
    prefix_duration_s = _horizon_seconds(source_intervals)
    # The opt-in responsive path commits exactly one matched-horizon prefix; the
    # default two-chunk interactive path and all other modes are untouched.
    preload_chunks = (
        _responsive_preload_chunks(namespace.preload_chunks)
        if responsive
        else _validated_preload_chunks(namespace.preload_chunks)
    )
    output_root = Path(namespace.output_root).expanduser().resolve()
    source_run = Path(namespace.source_run).expanduser().resolve(strict=True)
    gear_checkout = Path(namespace.gear_checkout).expanduser().resolve(strict=True)
    runtime = Path(namespace.runtime).expanduser().resolve(strict=True)
    terrain_dir = Path(namespace.terrain_dir).expanduser().resolve(strict=True)
    bundle = RunBundle.create(output_root, "manual-sonic", _utc_run_id())
    environment = _environment(terrain_dir)

    contract = load_joint_contract(_SONIC_ROOT / "configs/g1_joint_contract.json")
    validator = SourceValidator(contract)
    publisher = PosePublisher(
        "tcp://127.0.0.1:*",
        bundle=bundle,
        default_hand_targets=NEUTRAL_HAND_TARGETS,
    )
    mm_server = Path(namespace.mm_server).expanduser().resolve(strict=True)
    mm = MMChunkClient(
        run_root=bundle.path,
        command=(str(mm_server),),
        stdout_archive=bundle.path / "mm.stdout",
        stderr_archive=bundle.path / "mm.stderr",
        env=environment,
        cwd=_REPOSITORY_ROOT,
    )
    base_command = (
        str(
            gear_checkout
            / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
        ),
        "lo",
        str(runtime / "model_decoder.onnx"),
        str(source_run / "known-good/reference-base"),
        "--obs-config",
        str(runtime / "observation_config.yaml"),
        "--encoder-file",
        str(runtime / "model_encoder.onnx"),
    )
    cancellation = threading.Event()
    cancelled = cancellation.is_set
    gear = GearProcess(
        run_root=bundle.path,
        command=_stream_gear_command(base_command, publisher.endpoint),
        target_motion_logfile=bundle.path / "target.csv",
        logs_dir=bundle.path / "gear-logs",
        stdout_archive=bundle.path / "gear.stdout",
        stderr_archive=bundle.path / "gear.stderr",
        launch_profile="zmq_stream",
        simulation_control_gate=True,
        readiness_timeout_s=120.0,
        cancelled=cancelled,
        env=environment,
        cwd=gear_checkout / "gear_sonic_deploy",
    )
    simulator = GatedSimulatorClient(
        run_root=bundle.path,
        gear_checkout=gear_checkout,
        # A rolling operator loop resumes GEAR immediately before each chunk.
        # Keep MuJoCo at wall-clock pace so the policy/control threads can
        # consume LowState and the newly appended reference before physics
        # advances past them.  The fully preloaded evidence runner can safely
        # use unpaced physics; this live producer cannot.
        unpaced_physics=False,
        onscreen=namespace.onscreen,
        # Diagnostic-only: preserve the first fallen pose and visible terrain in
        # manual visible runs. This is valid only when onscreen is enabled.
        freeze_on_fall=namespace.onscreen,
        stdout_archive=bundle.path / "simulator.stdout",
        stderr_archive=bundle.path / "simulator.stderr",
        cancelled=cancelled,
        env=environment,
        cwd=_REPOSITORY_ROOT,
    )

    gate: SimulationPolicyGate | None = None
    timeline: TargetTimeline | None = None
    session_id = f"manual-{os.getpid()}"
    next_chunk = 0
    responsive_traces: list[object] = []
    recorder = CommandRecorder(
        mode=namespace.mode,
        preload_chunks=preload_chunks,
        hand_targets=NEUTRAL_HAND_TARGETS,
    )

    def publish(buffer, *, phase: str, attempt: int | None = None, wait: bool) -> None:
        boundary = gear.publication_boundary() if wait else None
        publisher.send(buffer, phase=phase, attempt=attempt)
        if boundary is not None:
            gear.wait_for_stream_processing(
                boundary,
                frame_count=buffer.count,
                global_start=int(buffer.frame_index[0]),
                merged_count=int(buffer.frame_index[-1]) + 1,
            )

    def generate_and_publish(command: CommandSample, *, wait: bool) -> int:
        nonlocal next_chunk
        assert timeline is not None
        if command.chunk_index != next_chunk:
            raise RuntimeError(
                f"command index {command.chunk_index} != next chunk {next_chunk}"
            )
        candidate = f"{session_id}:candidate:{next_chunk:06d}"
        raw = mm.generate(
            command,
            session_id=session_id,
            candidate_id=candidate,
            predecessor_id=timeline.last_accepted_candidate_id,
            source_intervals=source_intervals,
        )
        prepared = timeline.prepare(validator.validate_source(raw))
        try:
            publish(prepared.target.buffer, phase="logical" if wait else "timeline", wait=wait)
        except BaseException:
            timeline.abort(candidate)
            mm.abort(candidate)
            raise
        mm.commit(candidate)
        timeline.commit(prepared)
        recorder.record(command)
        next_chunk += 1
        return int(prepared.target.buffer.frame_index[-1])

    try:
        hello = mm.hello()
        _require_supported_movement_model(hello, namespace.movement_model)
        reset = mm.reset(
            SessionConfig(
                namespace.scene_id,
                namespace.route_id,
                namespace.terrain_weight,
                namespace.movement_model,
            ),
            session_id=session_id,
        )
        scene = register_scene(
            namespace.scene_id,
            namespace.route_id,
            registry_path=SCENE_REGISTRY_PATH,
            terrain_dir=terrain_dir,
            official_scene_xml=gear_checkout / GEAR_SCENE_RELATIVE,
            output_dir=bundle.path / "scene",
            expected_official_scene_sha256=GEAR_SCENE_SHA256,
            expected_robot_sha256=GEAR_ROBOT_SHA256,
            mm_hello_identity=hello,
            mm_scene_identity=reset["scene"],
        )
        initial = validator.validate_initial(reset)
        initial_state = initial_physics_state(
            scene,
            initial,
            contract,
            NEUTRAL_HAND_TARGETS,
        )
        scene_xml = scene.gear_scene_xml
        initial_qpos = initial_state.qpos
        timeline = TargetTimeline(initial, contract)
        simulator.hello()
        simulator.reset(
            scene_xml=scene_xml,
            initial_qpos=initial_qpos,
            lateral_offset_m=0.0,
            yaw_offset_rad=0.0,
            log_dir=bundle.path / "bootstrap-sim-logs",
            elastic_band_enabled=True,
        )
        print("Starting GEAR and priming the rolling reference...", flush=True)
        _drive_simulator_until(
            gear.start_to_wait_for_control,
            simulator,
            label="manual-wait-for-control",
            ready_for_bootstrap=lambda: gear.startup_markers_ready,
            cancellation=cancellation,
        )

        def prepare_stream() -> None:
            gear.enable_stream_for_preload()
            assert timeline is not None
            publish(
                timeline.initial_buffer,
                phase="readiness",
                attempt=1,
                wait=True,
            )
            for index in range(preload_chunks):
                generate_and_publish(_stand_command(index), wait=True)

        _drive_simulator_until(
            prepare_stream,
            simulator,
            label="manual-stream-preload",
            cancellation=cancellation,
        )
        gear.stop_group()
        _reset_and_prime_scored_epoch(
            gear,
            simulator,
            scene_xml=scene_xml,
            initial_qpos=initial_qpos,
            log_dir=bundle.path / "scored-sim-logs",
        )
        steps_per_chunk = round(prefix_duration_s / simulator.sim_dt)

        camera_state = _initial_camera_delivery_state(namespace.onscreen)
        last_command: CommandSample | None = None
        if namespace.mode == "script":
            gate = _activate_scored_control(gear, simulator)
            script = flat_command_script()
            for consumed_chunk in range(namespace.chunks):
                command = (
                    script[next_chunk]
                    if next_chunk < len(script)
                    else _stand_command(next_chunk, last_command)
                )
                frame_end = generate_and_publish(command, wait=False)
                last_command = command
                advance = gate.release_steps(
                    steps_per_chunk, expected_stream_frame_end=frame_end
                )
                print(
                    f"boundary {consumed_chunk + 1:03d}: "
                    f"sim={advance.sim_time_end_s:.2f}s queued={next_chunk - 1:03d} "
                    f"vx={command.requested_velocity_mujoco[0]:+.2f}",
                    flush=True,
                )
        elif namespace.input_source == "x11":
            initial_heading = _heading_yaw_rad(
                tuple(float(value) for value in initial_qpos[3:7])
            )
            initial_virtual_quaternion = holden_to_mujoco_quaternions(
                initial.virtual_root_orientation_holden
            )
            heading_frame_offset = _heading_frame_offset_yaw_rad(
                tuple(float(value) for value in initial_qpos[3:7]),
                tuple(float(value) for value in initial_virtual_quaternion),
            )
            provider = X11KeyStateProvider()
            mapper = HoldenControlMapper(
                initial_heading_yaw_rad=initial_heading,
                heading_frame_offset_yaw_rad=heading_frame_offset,
            )
            try:
                with ContinuousControlLoop(
                    provider,
                    mapper,
                    event_sink=_print_terminal_event,
                    cancel_event=cancellation,
                ) as control_loop:
                    def wait_for_initial_input() -> None:
                        _wait_for_x11_target(provider, control_loop)
                        print(
                            "LIVE X11: W/A/S/D move, Shift walk, "
                            "Ctrl+arrows strafe/face, arrows orbit camera, "
                            "Q/E zoom, Space stand, X exit. "
                            f"Commands have "
                            f"{preload_chunks * prefix_duration_s:.1f}s "
                            "lookahead latency.",
                            flush=True,
                        )

                    gate = _activate_scored_control(
                        gear,
                        simulator,
                        before_control=wait_for_initial_input,
                    )
                    if responsive:
                        # Opt-in Stage-R1 one-prefix responsive path. The single
                        # preloaded chunk already committed index 0, so control
                        # resumes at next_chunk. The committer presents
                        # manual_demo's transaction as run_one_chunk; physics is
                        # released only after publication and both commits.  The
                        # released physics horizon matches the configured MM
                        # source-interval prefix exactly (source_intervals / 25s
                        # of simulation), keeping generation, publication, and
                        # release on one horizon.
                        committer = ManualChunkCommitter(
                            mm=mm,
                            validator=validator,
                            timeline=timeline,
                            publish=publish,
                            gate=gate,
                            session_id=session_id,
                            steps_per_chunk=steps_per_chunk,
                            source_intervals=source_intervals,
                            sim_dt_s=simulator.sim_dt,
                            recorder=recorder,
                            state_log_reader=StateLogRootReader(
                                bundle.path / "scored-sim-logs" / "state.jsonl"
                            ),
                            next_chunk=next_chunk,
                        )
                        camera_state = _run_responsive_x11_loop(
                            control_loop=control_loop,
                            committer=committer,
                            simulator=simulator,
                            chunks=namespace.chunks,
                            camera_state=camera_state,
                            event_sink=_print_terminal_event,
                            trace_sink=responsive_traces.append,
                            session_id=session_id,
                            camera_disabled_prefix="CAMERA DISABLED",
                        )
                        next_chunk = committer.next_chunk
                    else:
                        for _consumed_chunk in range(namespace.chunks):
                            result = _consume_x11_boundary(
                                control_loop=control_loop,
                                simulator=simulator,
                                gate=gate,
                                generate_and_publish=generate_and_publish,
                                chunk_index=next_chunk,
                                steps_per_chunk=steps_per_chunk,
                                preload_chunks=preload_chunks,
                                camera_state=camera_state,
                                event_sink=_print_terminal_event,
                                control_prefix="CONTROL chunk=",
                                camera_disabled_prefix="CAMERA DISABLED",
                                prefix_duration_s=prefix_duration_s,
                            )
                            camera_state = result.camera_state
                            if result.command is None:
                                break
                            last_command = result.command
            finally:
                provider.close()
        else:
            print(
                "TERMINAL COMPATIBILITY: coarse non-parity W/S/A/D/Q/E "
                "controls; Space stands and X exits.",
                flush=True,
            )
            sampler = OperatorSampler(
                OperatorLimits(
                    forward_mps=0.5,
                    backward_mps=0.5,
                    lateral_mps=0.5,
                )
            )
            key_buffer = TerminalKeyBuffer()
            reader = TerminalInputReader(
                sys.stdin.fileno(),
                key_buffer,
                event_sink=_print_terminal_event,
            )
            with reader:
                gate = _activate_scored_control(gear, simulator)
                for consumed_chunk in range(namespace.chunks):
                    sampler.update(key_buffer.sample())
                    command = sampler.sample_boundary(next_chunk)
                    if command is None:
                        break
                    frame_end = generate_and_publish(command, wait=False)
                    last_command = command
                    advance = gate.release_steps(
                        steps_per_chunk, expected_stream_frame_end=frame_end
                    )
                    print(
                        f"boundary {consumed_chunk + 1:03d}: "
                        f"sim={advance.sim_time_end_s:.2f}s "
                        f"queued={next_chunk - 1:03d} "
                        f"vx={command.requested_velocity_mujoco[0]:+.2f} "
                        f"vy={command.requested_velocity_mujoco[1]:+.2f}",
                        flush=True,
                    )
        # Artifact generation can exceed GEAR's 500 ms LowState watchdog after
        # the final physics step. Use its official stop path while LowState is
        # still fresh, then serialize evidence with GEAR already reaped.
        gate.finish_policy()
        command_artifact = recorder.artifact_bytes()
        responsive_evidence = _write_responsive_evidence(
            bundle,
            responsive,
            responsive_traces,
            committed_prefixes=(
                next_chunk - preload_chunks if responsive else None
            ),
            source_intervals=source_intervals,
            movement_model=reset["movement_model"],
        )
        snapshot = simulator.snapshot()
        if namespace.scene_id == "sonic-flat-baseline":
            registration = json.loads(
                (scene_xml.parent / "scene_registration.json").read_bytes()
            )
            scene_control = {
                "gear_scene_sha256": scene.output_hashes["gear_scene_xml"],
                "gear_robot_sha256": scene.output_hashes["robot_include"],
                "actuator_joint_order_sha256": registration[
                    "actuator_joint_order_sha256"
                ],
            }
            is_turn_profile = namespace.movement_model == "holden-turn-v1"
            flat_summary_builder = (
                manual_flat_summary_v8_bytes
                if is_turn_profile
                else manual_flat_summary_v6_bytes
            )
            expected_flat_schema = (
                "mm-sonic-manual-demo/v8"
                if is_turn_profile
                else "mm-sonic-manual-demo/v6"
            )
            summary_bytes = flat_summary_builder(
                mode=namespace.mode,
                run_root=str(bundle.path),
                preload_chunks=preload_chunks,
                generated_chunks=next_chunk,
                lookahead_seconds=preload_chunks * prefix_duration_s,
                command_bytes=command_artifact,
                hand_targets=NEUTRAL_HAND_TARGETS,
                scene_control=scene_control,
                snapshot=snapshot,
                movement_model=reset["movement_model"],
            )
            summary = json.loads(summary_bytes)
            if summary.get("schema") != expected_flat_schema:
                raise ContractError("flat manual summary schema changed")
        else:
            environment_control = environment_control_record(
                scene_id=scene.scene_id,
                route_id=scene.route_id,
                terrain_weight=namespace.terrain_weight,
                input_source=namespace.input_source,
                mapper_version=MANUAL_MAPPER_VERSION,
                camera_sequence=(
                    0
                    if camera_state.last_sequence is None
                    else camera_state.last_sequence
                ),
                mm_hello_identity=dict(hello),
                mm_scene_identity=dict(reset["scene"]),
                source_hashes=dict(scene.source_hashes),
                output_hashes=dict(scene.output_hashes),
                coordinate_source=scene.coordinate_source,
                coordinate_target=scene.coordinate_target,
                transform_matrix=scene.transform_matrix.tolist(),
                source_bounds_holden=scene.source_bounds_holden.tolist(),
                transformed_bounds_mujoco=(
                    scene.transformed_bounds_mujoco.tolist()
                ),
                initial_boundary_sha256=(
                    initial_state.initial_boundary_sha256
                ),
                initial_qpos_sha256=initial_state.qpos_sha256,
            )
            is_turn_profile = namespace.movement_model == "holden-turn-v1"
            terrain_summary_builder = (
                manual_summary_v7_bytes
                if is_turn_profile
                else manual_summary_v5_bytes
            )
            expected_terrain_schema = (
                "mm-sonic-manual-demo/v7"
                if is_turn_profile
                else "mm-sonic-manual-demo/v5"
            )
            summary_bytes = terrain_summary_builder(
                mode=namespace.mode,
                run_root=str(bundle.path),
                preload_chunks=preload_chunks,
                generated_chunks=next_chunk,
                lookahead_seconds=preload_chunks * prefix_duration_s,
                command_bytes=command_artifact,
                hand_targets=NEUTRAL_HAND_TARGETS,
                environment_control=environment_control,
                snapshot=snapshot,
                movement_model=reset["movement_model"],
            )
            summary = json.loads(summary_bytes)
            if summary.get("schema") != expected_terrain_schema:
                raise ContractError("terrain manual summary schema changed")
        bundle.write_bytes("manual-commands.json", command_artifact)
        bundle.write_bytes("manual-summary.json", summary_bytes)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if responsive_evidence is not None:
            print(json.dumps(responsive_evidence, sort_keys=True), flush=True)
        return bundle.path
    finally:
        cancellation.set()
        if gate is not None:
            gate.close()
        else:
            try:
                simulator.close()
            finally:
                gear.close()
        publisher.close()
        mm.close()


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _resolve_mode_defaults(namespace: argparse.Namespace) -> None:
    """Fill mode-dependent defaults left unset on the command line.

    Interactive runs default to the authenticated ``grail-curb-default`` terrain
    scene, the ``curb-forward`` route, terrain weight ``4.0``, continuous X11
    input, and two 0.4-second preload chunks. The explicit
    ``sonic-flat-baseline`` / ``flat-12s`` / ``0.0`` / four-chunk configuration
    remains the backward-compatible flat diagnostic path.
    """

    interactive = namespace.mode == "interactive"
    if namespace.scene_id is None:
        namespace.scene_id = (
            "grail-curb-default" if interactive else "sonic-flat-baseline"
        )
    if namespace.route_id is None:
        namespace.route_id = "curb-forward" if interactive else "flat-12s"
    if namespace.terrain_weight is None:
        namespace.terrain_weight = 4.0 if interactive else 0.0
    if namespace.input_source is None:
        namespace.input_source = "x11"
    if namespace.preload_chunks is None:
        namespace.preload_chunks = 2 if interactive else _PRELOAD_CHUNKS


class _ManualArgumentParser(argparse.ArgumentParser):
    """Argument parser that resolves mode-dependent interactive defaults."""

    def parse_known_args(self, args=None, namespace=None):  # type: ignore[override]
        parsed, extras = super().parse_known_args(args, namespace)
        _resolve_mode_defaults(parsed)
        return parsed, extras


def _parser() -> argparse.ArgumentParser:
    parser = _ManualArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("script", "interactive"), default="script")
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--preload-chunks", type=int, default=None)
    parser.add_argument("--onscreen", action="store_true")
    parser.add_argument("--responsive", action="store_true")
    parser.add_argument(
        "--responsive-source-intervals",
        type=int,
        choices=(5, 10),
        default=10,
    )
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--route-id", default=None)
    parser.add_argument("--terrain-weight", type=float, default=None)
    parser.add_argument(
        "--movement-model",
        choices=("raw", "holden-v1", "holden-turn-v1"),
        default="raw",
    )
    parser.add_argument(
        "--input-source", choices=("x11", "terminal"), default=None
    )
    parser.add_argument("--output-root", default="/home/ubuntu/mm-sonic-manual-runs")
    parser.add_argument("--source-run", default=str(_DEFAULT_SOURCE_RUN))
    parser.add_argument("--gear-checkout", default=str(_DEFAULT_GEAR))
    parser.add_argument("--runtime", default=str(_DEFAULT_RUNTIME))
    parser.add_argument("--terrain-dir", default=str(_DEFAULT_TERRAIN))
    parser.add_argument(
        "--mm-server", default=str(_SONIC_ROOT / "build/mm_chunk_server")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    if namespace.chunks <= 0:
        raise SystemExit("--chunks must be positive")
    run_demo(namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
