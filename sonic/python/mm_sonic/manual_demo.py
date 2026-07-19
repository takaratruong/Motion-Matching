"""Non-scored rolling Motion Matching -> SONIC operator demonstration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

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
from .joints import ContractError, load_joint_contract
from .manual_evidence import manual_command_artifact_bytes
from .operator import OperatorLimits, OperatorSampler
from .scene import normalize_run_local_actuators, verify_loaded_actuator_routing
from .operator_terminal import TerminalInputReader, TerminalKeyBuffer
from .process import (
    GatedSimulatorClient,
    GearProcess,
    MMChunkClient,
    SimulationPolicyGate,
)
from .timeline import TargetTimeline
from .zmq_v1 import PosePublisher


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


def run_demo(namespace: argparse.Namespace) -> Path:
    preload_chunks = _validated_preload_chunks(namespace.preload_chunks)
    output_root = Path(namespace.output_root).expanduser().resolve()
    source_run = Path(namespace.source_run).expanduser().resolve(strict=True)
    gear_checkout = Path(namespace.gear_checkout).expanduser().resolve(strict=True)
    runtime = Path(namespace.runtime).expanduser().resolve(strict=True)
    terrain_dir = Path(namespace.terrain_dir).expanduser().resolve(strict=True)
    bundle = RunBundle.create(output_root, "manual-sonic", _utc_run_id())
    scene_xml, scene_control = _copy_scene(bundle, source_run)
    initial_qpos = _initial_qpos(scene_xml)
    environment = _environment(terrain_dir)

    contract = load_joint_contract(_SONIC_ROOT / "configs/g1_joint_contract.json")
    validator = SourceValidator(contract)
    publisher = PosePublisher(
        "tcp://127.0.0.1:*",
        bundle=bundle,
        default_hand_targets=NEUTRAL_HAND_TARGETS,
    )
    mm = MMChunkClient(
        run_root=bundle.path,
        command=(str(_SONIC_ROOT / "build/mm_chunk_server"),),
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
    gear = GearProcess(
        run_root=bundle.path,
        command=_stream_gear_command(base_command, publisher.endpoint),
        target_motion_logfile=bundle.path / "target.csv",
        logs_dir=bundle.path / "gear-logs",
        stdout_archive=bundle.path / "gear.stdout",
        stderr_archive=bundle.path / "gear.stderr",
        launch_profile="zmq_stream",
        readiness_timeout_s=120.0,
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
        stdout_archive=bundle.path / "simulator.stdout",
        stderr_archive=bundle.path / "simulator.stderr",
        env=environment,
        cwd=_REPOSITORY_ROOT,
    )

    gate: SimulationPolicyGate | None = None
    timeline: TargetTimeline | None = None
    session_id = f"manual-{os.getpid()}"
    next_chunk = 0
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

    def generate_and_publish(command: CommandSample, *, wait: bool) -> None:
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
            source_intervals=10,
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

    cancellation = threading.Event()
    try:
        mm.hello()
        reset = mm.reset(
            SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
            session_id=session_id,
        )
        timeline = TargetTimeline(validator.validate_initial(reset), contract)
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
        gear.continue_group()
        gear.activate_control()
        gate = SimulationPolicyGate(gear, simulator)
        gate.pause()
        steps_per_chunk = round(_CHUNK_DURATION_S / simulator.sim_dt)

        script = flat_command_script()
        sampler = OperatorSampler(
            OperatorLimits(
                forward_mps=0.5,
                backward_mps=0.5,
                lateral_mps=0.5,
            )
        )
        key_buffer = TerminalKeyBuffer()
        reader = (
            TerminalInputReader(
                sys.stdin.fileno(),
                key_buffer,
                event_sink=_print_terminal_event,
            )
            if namespace.mode == "interactive"
            else None
        )
        context = reader if reader is not None else _NullContext()
        last_command: CommandSample | None = None
        with context:
            if namespace.mode == "interactive":
                print(
                    "LIVE: W forward, space stand, Q/E turn, X exit. "
                    f"Commands have ~{preload_chunks * _CHUNK_DURATION_S:.1f} s "
                    "lookahead latency.",
                    flush=True,
                )
            for consumed_chunk in range(namespace.chunks):
                if namespace.mode == "script":
                    command = (
                        script[next_chunk]
                        if next_chunk < len(script)
                        else _stand_command(next_chunk, last_command)
                    )
                else:
                    sampler.update(key_buffer.sample())
                    command = sampler.sample_boundary(next_chunk)
                    if command is None:
                        break
                generate_and_publish(command, wait=False)
                last_command = command
                advance = gate.release_steps(steps_per_chunk)
                print(
                    f"boundary {consumed_chunk + 1:03d}: "
                    f"sim={advance.sim_time_end_s:.2f}s queued={next_chunk - 1:03d} "
                    f"vx={command.requested_velocity_mujoco[0]:+.2f}",
                    flush=True,
                )
        command_artifact = recorder.artifact_bytes()
        summary = {
            "schema": "mm-sonic-manual-demo/v3",
            "mode": namespace.mode,
            "run_root": str(bundle.path),
            "preload_chunks": preload_chunks,
            "generated_chunks": next_chunk,
            "lookahead_seconds": preload_chunks * _CHUNK_DURATION_S,
            "command_artifact": {
                "path": "manual-commands.json",
                "sha256": hashlib.sha256(command_artifact).hexdigest(),
            },
            "hand_control": hand_targets_record(NEUTRAL_HAND_TARGETS),
            "scene_control": scene_control,
            "snapshot": simulator.snapshot(),
        }
        bundle.write_bytes("manual-commands.json", command_artifact)
        bundle.write_bytes(
            "manual-summary.json",
            (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode(),
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("script", "interactive"), default="script")
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--preload-chunks", type=int, default=_PRELOAD_CHUNKS)
    parser.add_argument("--onscreen", action="store_true")
    parser.add_argument("--output-root", default="/home/ubuntu/mm-sonic-manual-runs")
    parser.add_argument("--source-run", default=str(_DEFAULT_SOURCE_RUN))
    parser.add_argument("--gear-checkout", default=str(_DEFAULT_GEAR))
    parser.add_argument("--runtime", default=str(_DEFAULT_RUNTIME))
    parser.add_argument("--terrain-dir", default=str(_DEFAULT_TERRAIN))
    return parser


def main(argv: list[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    if namespace.chunks <= 0:
        raise SystemExit("--chunks must be positive")
    run_demo(namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
