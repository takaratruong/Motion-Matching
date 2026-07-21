from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import math
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mm_sonic.commands import CommandSample
from mm_sonic.hands import NEUTRAL_HAND_TARGETS, hand_targets_record
from mm_sonic.holden_control import (
    CameraState,
    HoldenControlMapper,
    MappedControlState,
)
from mm_sonic.joints import ContractError
from mm_sonic.manual_demo import (
    CameraDeliveryState,
    CommandRecorder,
    DemoDependencies,
    _consume_x11_boundary,
    _heading_frame_offset_yaw_rad,
    _initial_camera_delivery_state,
    _parser,
    _print_terminal_event,
    _run_startup_transaction,
    _validated_preload_chunks,
    main,
)
from mm_sonic.manual_evidence import parse_manual_command_artifact
from mm_sonic.process import ProcessProtocolError


def _stand(index: int) -> CommandSample:
    return CommandSample(index, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _forward(index: int) -> CommandSample:
    return CommandSample(index, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


class CommandRecorderTests(unittest.TestCase):
    def test_onscreen_is_explicit_and_opt_in(self) -> None:
        self.assertFalse(_parser().parse_args([]).onscreen)
        self.assertTrue(_parser().parse_args(["--onscreen"]).onscreen)

    def test_preload_chunks_defaults_to_four(self) -> None:
        self.assertEqual(_parser().parse_args([]).preload_chunks, 4)

    def test_responsive_flag_defaults_false_and_is_opt_in(self) -> None:
        self.assertFalse(_parser().parse_args([]).responsive)
        self.assertTrue(_parser().parse_args(["--responsive"]).responsive)

    def test_responsive_does_not_change_interactive_two_chunk_default(self) -> None:
        args = _parser().parse_args(["--mode", "interactive"])
        # Defaults untouched: two-chunk interactive path, flag off.
        self.assertEqual(args.preload_chunks, 2)
        self.assertFalse(args.responsive)

    def test_responsive_preload_chunks_forces_exactly_one(self) -> None:
        from mm_sonic.manual_demo import _responsive_preload_chunks

        # Regardless of the resolved default, responsive uses one 0.4s prefix.
        self.assertEqual(_responsive_preload_chunks(2), 1)
        self.assertEqual(_responsive_preload_chunks(4), 1)

    def test_terrain_and_x11_are_interactive_defaults(self) -> None:
        args = _parser().parse_args(["--mode", "interactive", "--onscreen"])
        self.assertEqual(args.scene_id, "grail-curb-default")
        self.assertEqual(args.route_id, "curb-forward")
        self.assertEqual(args.terrain_weight, 4.0)
        self.assertEqual(args.input_source, "x11")
        self.assertEqual(args.preload_chunks, 2)

    def test_flat_mode_requires_registered_flat_identity(self) -> None:
        args = _parser().parse_args([
            "--scene-id", "sonic-flat-baseline",
            "--route-id", "flat-12s",
            "--terrain-weight", "0.0",
        ])
        self.assertEqual(args.scene_id, "sonic-flat-baseline")

    def test_preload_chunks_accepts_explicit_one_and_two(self) -> None:
        self.assertEqual(
            _parser().parse_args(["--preload-chunks", "1"]).preload_chunks, 1
        )
        self.assertEqual(
            _parser().parse_args(["--preload-chunks", "2"]).preload_chunks, 2
        )

    def test_records_committed_commands_in_chunk_order(self) -> None:
        recorder = CommandRecorder(
            mode="script",
            preload_chunks=2,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        recorder.record(_stand(0))
        recorder.record(_stand(1))
        recorder.record(_forward(2))

        parsed = parse_manual_command_artifact(recorder.artifact_bytes())

        self.assertEqual(parsed.mode, "script")
        self.assertEqual(parsed.preload_chunks, 2)
        self.assertEqual(
            [command.chunk_index for command in parsed.commands], [0, 1, 2]
        )

    def test_rejects_out_of_order_commit(self) -> None:
        recorder = CommandRecorder(
            mode="script",
            preload_chunks=2,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        recorder.record(_stand(0))

        with self.assertRaisesRegex(ContractError, "chunk"):
            recorder.record(_forward(2))

    def test_records_neutral_hand_control_in_v3_command_artifact(self) -> None:
        recorder = CommandRecorder(
            mode="script",
            preload_chunks=1,
            hand_targets=NEUTRAL_HAND_TARGETS,
        )
        recorder.record(_stand(0))
        recorder.record(_forward(1))

        document = json.loads(recorder.artifact_bytes())

        self.assertEqual(document["schema"], "mm-sonic-manual-command/v3")
        self.assertEqual(
            document["hand_control"], hand_targets_record(NEUTRAL_HAND_TARGETS)
        )


class _FakeMM:
    def __init__(self, log: list[str], scene_id: str, route_id: str) -> None:
        self._log = log
        self._scene_id = scene_id
        self._route_id = route_id
        self.reset_config = None

    def hello(self) -> dict:
        self._log.append("mm.hello")
        return {"coordinate_signature": "holden", "id": "hello"}

    def reset(self, config: object, *, session_id: str) -> dict:
        self._log.append("mm.reset")
        self.reset_config = config
        return {
            "scene": {
                "scene_id": self._scene_id,
                "route_id": self._route_id,
            },
            "initial_boundary": {"session_id": session_id},
        }


class _FakeValidator:
    def validate_initial(self, raw: object) -> object:
        return ("initial-boundary", raw)


class _FakeSimulator:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.reset_kwargs = None

    def hello(self) -> dict:
        self._log.append("simulator.hello")
        return {"protocol": "gated-sim/v1"}

    def reset(self, **kwargs: object) -> dict:
        self._log.append("simulator.reset")
        self.reset_kwargs = kwargs
        return {"nq": 50, "sim_dt_s": 0.005, "sim_time_s": 0.0}


class _FakeScene:
    def __init__(self) -> None:
        self.gear_scene_xml = Path("/runs/manual/scene/gear_scene.xml")


class _FakeInitialState:
    def __init__(self) -> None:
        self.qpos = [0.0] * 50
        self.qpos_sha256 = "a" * 64
        self.initial_boundary_sha256 = "b" * 64


class StartupTransactionTests(unittest.TestCase):
    def _run(self, scene_id: str, route_id: str, weight: float):
        log: list[str] = []
        mm = _FakeMM(log, scene_id, route_id)
        simulator = _FakeSimulator(log)
        register_calls: list[dict] = []
        initial_calls: list[tuple] = []
        scene = _FakeScene()
        state = _FakeInitialState()

        def register_scene(sid, rid, *, mm_hello_identity, mm_scene_identity):
            log.append("register_scene")
            register_calls.append(
                {
                    "scene_id": sid,
                    "route_id": rid,
                    "hello": mm_hello_identity,
                    "scene": mm_scene_identity,
                }
            )
            return scene

        def build_initial_state(registered, initial_boundary):
            log.append("initial_physics_state")
            initial_calls.append((registered, initial_boundary))
            return state

        deps = DemoDependencies(
            register_scene=register_scene,
            build_initial_state=build_initial_state,
        )
        result = _run_startup_transaction(
            deps,
            mm=mm,
            simulator=simulator,
            validator=_FakeValidator(),
            scene_id=scene_id,
            route_id=route_id,
            terrain_weight=weight,
            session_id="manual-1",
            log_dir=Path("/runs/manual/bootstrap-sim-logs"),
        )
        return log, register_calls, initial_calls, scene, state, mm, simulator, result

    def test_startup_calls_occur_in_registered_order(self) -> None:
        log, *_ = self._run("grail-curb-default", "curb-forward", 4.0)
        self.assertEqual(
            log,
            [
                "mm.hello",
                "mm.reset",
                "register_scene",
                "initial_physics_state",
                "simulator.hello",
                "simulator.reset",
            ],
        )

    def test_register_scene_receives_exact_hello_and_reset_identities(self) -> None:
        _, register_calls, _, _, _, mm, _, _ = self._run(
            "grail-curb-default", "curb-forward", 4.0
        )
        self.assertEqual(register_calls[0]["scene_id"], "grail-curb-default")
        self.assertEqual(register_calls[0]["route_id"], "curb-forward")
        self.assertEqual(register_calls[0]["hello"], {"coordinate_signature": "holden", "id": "hello"})
        self.assertEqual(
            register_calls[0]["scene"],
            {"scene_id": "grail-curb-default", "route_id": "curb-forward"},
        )

    def test_simulator_reset_uses_registered_xml_and_initial_qpos(self) -> None:
        *_, scene, state, _mm, simulator, _ = self._run(
            "grail-curb-default", "curb-forward", 4.0
        )
        self.assertEqual(simulator.reset_kwargs["scene_xml"], scene.gear_scene_xml)
        self.assertIs(simulator.reset_kwargs["initial_qpos"], state.qpos)

    def test_initial_state_built_from_validated_boundary(self) -> None:
        _, _, initial_calls, scene, _, _, _, _ = self._run(
            "grail-curb-default", "curb-forward", 4.0
        )
        registered, initial_boundary = initial_calls[0]
        self.assertIs(registered, scene)
        self.assertEqual(initial_boundary[0], "initial-boundary")

    def test_reset_uses_cli_selected_session_config(self) -> None:
        *_, mm, _simulator, _result = self._run(
            "sonic-flat-baseline", "flat-12s", 0.0
        )
        self.assertEqual(mm.reset_config.scene_id, "sonic-flat-baseline")
        self.assertEqual(mm.reset_config.route_id, "flat-12s")
        self.assertEqual(mm.reset_config.terrain_weight, 0.0)


class _BoundaryMailbox:
    def __init__(self, command: CommandSample, mapped: MappedControlState) -> None:
        self.command = command
        self.mapped = mapped
        self.sampled: list[int] = []

    def sample(self, chunk_index: int):
        self.sampled.append(chunk_index)
        return self.command, self.mapped


class _BoundaryControlLoop:
    def __init__(self, command: CommandSample, mapped: MappedControlState) -> None:
        self.mailbox = _BoundaryMailbox(command, mapped)


class _BoundarySimulator:
    def __init__(self, error: ProcessProtocolError | None = None) -> None:
        self.error = error
        self.camera_calls: list[tuple[int, float, float, float]] = []

    def set_camera(
        self,
        sequence: int,
        azimuth_deg: float,
        elevation_deg: float,
        distance_m: float,
    ) -> dict:
        self.camera_calls.append(
            (sequence, azimuth_deg, elevation_deg, distance_m)
        )
        if self.error is not None:
            raise self.error
        return {
            "sequence": sequence,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "distance_m": distance_m,
        }


class _BoundaryGate:
    def __init__(self) -> None:
        self.releases: list[int] = []

    def release_steps(self, steps: int) -> object:
        self.releases.append(steps)
        return object()


def _mapped_boundary() -> tuple[CommandSample, MappedControlState]:
    yaw = 0.6
    heading = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    command = CommandSample(12, (0.25, -0.4, 0.0), heading)
    mapped = MappedControlState(
        velocity_mujoco=(0.25, -0.4, 0.0),
        desired_heading_mujoco_wxyz=heading,
        camera=CameraState(
            sequence=7,
            azimuth_rad=math.radians(30.0),
            altitude_rad=math.radians(20.0),
            distance_m=4.5,
        ),
        strafe=True,
        walk_blend=1.0,
        stand=False,
        terminate=False,
    )
    return command, mapped


class X11BoundaryIntegrationTests(unittest.TestCase):
    def test_heading_frame_offset_uses_physical_minus_virtual_yaw(self) -> None:
        physical = (
            math.cos(-math.pi / 4.0),
            0.0,
            0.0,
            math.sin(-math.pi / 4.0),
        )

        offset = _heading_frame_offset_yaw_rad(
            physical,
            (1.0, 0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(offset, -math.pi / 2.0)

    def test_headless_interactive_run_disables_only_camera_delivery(self) -> None:
        headless = _initial_camera_delivery_state(False)
        visible = _initial_camera_delivery_state(True)

        self.assertFalse(headless.enabled)
        self.assertTrue(visible.enabled)
        self.assertIsNone(headless.last_sequence)
        self.assertIsNone(visible.last_sequence)

    def test_one_boundary_forwards_lateral_heading_and_matching_camera(self) -> None:
        command, mapped = _mapped_boundary()
        control_loop = _BoundaryControlLoop(command, mapped)
        simulator = _BoundarySimulator()
        gate = _BoundaryGate()
        published: list[tuple[CommandSample, bool]] = []
        events: list[str] = []

        result = _consume_x11_boundary(
            control_loop=control_loop,
            simulator=simulator,
            gate=gate,
            generate_and_publish=lambda value, *, wait: published.append(
                (value, wait)
            ),
            chunk_index=12,
            steps_per_chunk=80,
            preload_chunks=2,
            camera_state=CameraDeliveryState(),
            event_sink=events.append,
            control_prefix="CONTROL chunk=",
            camera_disabled_prefix="CAMERA DISABLED",
        )

        self.assertIs(result.command, command)
        self.assertEqual(published, [(command, False)])
        self.assertEqual(command.requested_velocity_mujoco, (0.25, -0.4, 0.0))
        self.assertEqual(
            command.desired_heading_mujoco_wxyz,
            mapped.desired_heading_mujoco_wxyz,
        )
        self.assertEqual(gate.releases, [80])
        self.assertEqual(control_loop.mailbox.sampled, [12])
        self.assertEqual(len(simulator.camera_calls), 1)
        sequence, azimuth, elevation, distance = simulator.camera_calls[0]
        self.assertEqual(sequence, 7)
        self.assertAlmostEqual(azimuth, 30.0)
        self.assertAlmostEqual(elevation, -20.0)
        self.assertAlmostEqual(distance, 4.5)
        self.assertEqual(result.camera_state.last_sequence, 7)
        self.assertEqual(
            events,
            [
                "CONTROL chunk=000012 vx=+0.250 vy=-0.400 heading=+0.600 "
                "strafe=1 walk=1 presents_in=0.800s"
            ],
        )

    def test_invalid_consumed_camera_ack_disables_only_camera_once(self) -> None:
        command, mapped = _mapped_boundary()
        simulator = _BoundarySimulator(
            ProcessProtocolError(
                "camera data.sequence must echo the requested sequence"
            )
        )
        events: list[str] = []

        first = _consume_x11_boundary(
            control_loop=_BoundaryControlLoop(command, mapped),
            simulator=simulator,
            gate=_BoundaryGate(),
            generate_and_publish=lambda _value, *, wait: None,
            chunk_index=12,
            steps_per_chunk=80,
            preload_chunks=2,
            camera_state=CameraDeliveryState(),
            event_sink=events.append,
            control_prefix="CONTROL chunk=",
            camera_disabled_prefix="CAMERA DISABLED",
        )
        second_command = CommandSample(
            13,
            command.requested_velocity_mujoco,
            command.desired_heading_mujoco_wxyz,
        )
        second = _consume_x11_boundary(
            control_loop=_BoundaryControlLoop(second_command, mapped),
            simulator=simulator,
            gate=_BoundaryGate(),
            generate_and_publish=lambda _value, *, wait: None,
            chunk_index=13,
            steps_per_chunk=80,
            preload_chunks=2,
            camera_state=first.camera_state,
            event_sink=events.append,
            control_prefix="CONTROL chunk=",
            camera_disabled_prefix="CAMERA DISABLED",
        )

        self.assertFalse(first.camera_state.enabled)
        self.assertFalse(second.camera_state.enabled)
        self.assertEqual(len(simulator.camera_calls), 1)
        self.assertEqual(
            sum(event.startswith("CAMERA DISABLED") for event in events), 1
        )
        self.assertEqual(
            sum(event.startswith("CONTROL chunk=") for event in events), 2
        )

    def test_unconsumed_or_unsynchronized_camera_protocol_error_is_fatal(self) -> None:
        command, mapped = _mapped_boundary()
        simulator = _BoundarySimulator(
            ProcessProtocolError("simulator response is invalid JSON")
        )
        with self.assertRaisesRegex(ProcessProtocolError, "invalid JSON"):
            _consume_x11_boundary(
                control_loop=_BoundaryControlLoop(command, mapped),
                simulator=simulator,
                gate=_BoundaryGate(),
                generate_and_publish=lambda _value, *, wait: None,
                chunk_index=12,
                steps_per_chunk=80,
                preload_chunks=2,
                camera_state=CameraDeliveryState(),
                event_sink=lambda _event: None,
                control_prefix="CONTROL chunk=",
                camera_disabled_prefix="CAMERA DISABLED",
            )


class _ResponsiveMailbox:
    """A sample_intent mailbox serving scripted IntentSnapshots."""

    def __init__(self, snapshots, mapped) -> None:
        self._snapshots = list(snapshots)
        self._mapped = mapped
        self.sampled: list[int] = []
        self.current_revision = 1

    def sample_intent(self, chunk_index: int):
        self.sampled.append(chunk_index)
        snapshot = self._snapshots.pop(0)
        self.current_revision = snapshot.revision
        return snapshot, self._mapped


class _ResponsiveControlLoop:
    def __init__(self, mailbox) -> None:
        self.mailbox = mailbox


class ResponsiveX11LoopTests(unittest.TestCase):
    def _mapped(self):
        _command, mapped = _mapped_boundary()
        return mapped

    def test_responsive_loop_commits_one_prefix_and_delivers_camera(self) -> None:
        from mm_sonic.manual_demo import _run_responsive_x11_loop
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_wiring import ManualChunkCommitter

        command = CommandSample(0, (0.25, -0.4, 0.0), (1.0, 0.0, 0.0, 0.0))
        snapshot = IntentSnapshot(revision=1, observed_ns=5, command=command)
        mapped = self._mapped()
        mailbox = _ResponsiveMailbox([snapshot], mapped)
        control_loop = _ResponsiveControlLoop(mailbox)
        simulator = _BoundarySimulator()

        log: list[str] = []

        class _Timeline:
            last_accepted_candidate_id = None

            def prepare(self, checked):
                import numpy as np

                class _T:
                    virtual_root_position = np.zeros((20, 3), dtype=np.float32)
                    buffer = object()

                class _P:
                    target = _T()

                return _P()

            def commit(self, prepared):
                log.append("timeline.commit")

            def abort(self, candidate_id):
                log.append("timeline.abort")

        class _MM:
            def generate(self, command, **kwargs):
                log.append("mm.generate")
                return object()

            def commit(self, candidate_id):
                log.append("mm.commit")

            def abort(self, candidate_id):
                log.append("mm.abort")

        class _Validator:
            def validate_source(self, raw):
                return object()

        class _Gate:
            def release_steps(self, steps):
                log.append("release")
                return object()

        class _Recorder:
            def record(self, command):
                log.append("record")

        published: list = []
        committer = ManualChunkCommitter(
            mm=_MM(),
            validator=_Validator(),
            timeline=_Timeline(),
            publish=lambda buffer, *, phase, wait: published.append((phase, wait)),
            gate=_Gate(),
            session_id="session",
            steps_per_chunk=20,
            recorder=_Recorder(),
        )

        traces: list = []
        events: list[str] = []
        camera_state = CameraDeliveryState()

        final_state = _run_responsive_x11_loop(
            control_loop=control_loop,
            committer=committer,
            simulator=simulator,
            chunks=1,
            camera_state=camera_state,
            event_sink=events.append,
            trace_sink=traces.append,
            session_id="session",
            camera_disabled_prefix="CAMERA DISABLED",
        )

        # One prefix committed and physics released once.
        self.assertEqual(log.count("release"), 1)
        self.assertEqual(published, [("timeline", False)])
        # Camera delivered synchronously via on_sample.
        self.assertEqual(len(simulator.camera_calls), 1)
        # One honest boundary trace emitted (observed root unavailable -> None).
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].input_transition_id, "session:rev:000001")
        self.assertIsNone(traces[0].observed_mujoco_root_displacement)
        self.assertIsInstance(final_state, CameraDeliveryState)

    def test_responsive_loop_terminate_breaks_without_committing(self) -> None:
        from mm_sonic.manual_demo import _run_responsive_x11_loop
        from mm_sonic.operator_x11 import IntentSnapshot

        snapshot = IntentSnapshot(revision=1, observed_ns=5, command=None)
        mapped = self._mapped()
        mailbox = _ResponsiveMailbox([snapshot], mapped)
        control_loop = _ResponsiveControlLoop(mailbox)
        simulator = _BoundarySimulator()

        class _NeverCommitter:
            next_chunk = 0

            def run_one_chunk(self, command, *, command_is_current=None):
                raise AssertionError("terminate must not commit")

        traces: list = []
        final_state = _run_responsive_x11_loop(
            control_loop=control_loop,
            committer=_NeverCommitter(),
            simulator=simulator,
            chunks=3,
            camera_state=CameraDeliveryState(),
            event_sink=lambda _e: None,
            trace_sink=traces.append,
            session_id="session",
            camera_disabled_prefix="CAMERA DISABLED",
        )

        # X terminates: no trace, no camera required, loop broke.
        self.assertEqual(traces, [])
        self.assertIsInstance(final_state, CameraDeliveryState)


class _FakeBundle:
    def __init__(self, path) -> None:
        self.path = path
        self.written: dict = {}

    def write_text(self, relative, text):
        from pathlib import Path

        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.written[str(relative)] = text
        return target


class WriteResponsiveEvidenceTests(unittest.TestCase):
    def _trace_record(self, observed):
        from mm_sonic.responsive_wiring import build_boundary_trace, trace_record
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_scheduler import ScheduledPrefix
        from mm_sonic.responsive_wiring import AcceptedChunk

        snapshot = IntentSnapshot(revision=3, observed_ns=10, command=_forward(0))
        prefix = ScheduledPrefix(snapshot=snapshot, sampled_ns=20, accepted=None)
        accepted = AcceptedChunk(
            presented_prefix_id="session:candidate:000000",
            mm_started_ns=30,
            mm_completed_ns=40,
            publication_sent_ns=50,
            committed_ns=60,
            physics_release_requested_ns=70,
            simulation_advance_completed_ns=80,
            generated_virtual_root_displacement_mujoco=(0.25, 0.0, 0.0),
            observed_mujoco_root_displacement=observed,
            advance=object(),
        )
        return build_boundary_trace(prefix, accepted, session_id="session")

    def test_non_responsive_writes_no_evidence(self):
        from mm_sonic.manual_demo import _write_responsive_evidence

        with TemporaryDirectory() as tmp:
            bundle = _FakeBundle(Path(tmp))
            evidence = _write_responsive_evidence(bundle, False, [])
            self.assertIsNone(evidence)
            self.assertEqual(bundle.written, {})

    def test_responsive_writes_jsonl_and_honest_summary(self):
        from mm_sonic.manual_demo import _write_responsive_evidence

        with TemporaryDirectory() as tmp:
            bundle = _FakeBundle(Path(tmp))
            traces = [
                self._trace_record((0.24, 0.0, 0.0)),
                self._trace_record(None),
            ]
            evidence = _write_responsive_evidence(bundle, True, traces)

            self.assertEqual(evidence["committed_prefixes"], 2)
            # Honest availability tally: one real observed root, one unavailable.
            self.assertEqual(evidence["observed_root_available"], 1)
            self.assertEqual(evidence["observed_root_unavailable"], 1)
            lines = (
                bundle.path / "responsive-boundary-trace.jsonl"
            ).read_text().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["input_transition_id"], "session:rev:000003")
            self.assertTrue(first["observed_root_available"])
            second = json.loads(lines[1])
            self.assertFalse(second["observed_root_available"])
            self.assertIsNone(second["observed_mujoco_root_displacement"])


class TerminalEventPrintingTests(unittest.TestCase):
    def test_prints_one_complete_event_line(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            _print_terminal_event("KEY W -> forward")
        self.assertEqual(output.getvalue(), "KEY W -> forward\n")


class ValidatedPreloadChunksTests(unittest.TestCase):
    def test_accepts_each_value_from_one_through_four(self) -> None:
        for value in (1, 2, 3, 4):
            self.assertEqual(_validated_preload_chunks(value), value)

    def test_rejects_boolean(self) -> None:
        with self.assertRaisesRegex(ContractError, "preload"):
            _validated_preload_chunks(True)

    def test_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ContractError, "preload"):
            _validated_preload_chunks(0)

    def test_rejects_negative(self) -> None:
        with self.assertRaisesRegex(ContractError, "preload"):
            _validated_preload_chunks(-1)

    def test_rejects_above_four(self) -> None:
        with self.assertRaisesRegex(ContractError, "preload"):
            _validated_preload_chunks(5)

    def test_rejects_non_integer(self) -> None:
        with self.assertRaisesRegex(ContractError, "preload"):
            _validated_preload_chunks(2.0)


class MainPreloadValidationTests(unittest.TestCase):
    def test_invalid_preload_chunks_rejected_before_any_run_bundle(self) -> None:
        with TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ContractError, "preload"):
                main(
                    [
                        "--preload-chunks",
                        "5",
                        "--output-root",
                        output_root,
                    ]
                )
            self.assertEqual(list(Path(output_root).iterdir()), [])

    def test_responsive_rejected_in_script_mode_before_any_run_bundle(self) -> None:
        # The --responsive validation boundary lives in the real run_demo
        # entrypoint: it is valid only for interactive X11 and must reject
        # every other mode before any run bundle is materialized. Script mode
        # is the default, so main(["--responsive"]) drives that branch.
        with TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ContractError, "interactive X11"):
                main(
                    [
                        "--responsive",
                        "--mode",
                        "script",
                        "--output-root",
                        output_root,
                    ]
                )
            self.assertEqual(list(Path(output_root).iterdir()), [])

    def test_responsive_rejected_in_interactive_terminal_before_any_run_bundle(
        self,
    ) -> None:
        # Interactive mode alone is not enough: the terminal input source must
        # also be rejected by the same entrypoint guard before any run bundle.
        with TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ContractError, "interactive X11"):
                main(
                    [
                        "--responsive",
                        "--mode",
                        "interactive",
                        "--input-source",
                        "terminal",
                        "--output-root",
                        output_root,
                    ]
                )
            self.assertEqual(list(Path(output_root).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
