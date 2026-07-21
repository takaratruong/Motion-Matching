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
    _activate_scored_control,
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
from mm_sonic.process import ProcessError, ProcessProtocolError, SimulationPolicyGate


def _stand(index: int) -> CommandSample:
    return CommandSample(index, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _forward(index: int) -> CommandSample:
    return CommandSample(index, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


class _StopAfterSimulator(Exception):
    pass


class SimulatorFreezeWiringTests(unittest.TestCase):
    def _drive_run_demo(self, extra_argv):
        from unittest.mock import patch

        from mm_sonic import manual_demo

        captured = {}

        class _FakePublisher:
            endpoint = "tcp://127.0.0.1:5555"

            def __init__(self, *a, **k):
                pass

        def _fake_simulator(**kwargs):
            captured.update(kwargs)
            raise _StopAfterSimulator

        def _fake_gear(*_args, **kwargs):
            captured["gear_cancelled"] = kwargs.get("cancelled")
            return object()

        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            source_run = root / "source"
            gear_checkout = root / "gear"
            runtime = root / "runtime"
            terrain_dir = root / "terrain"
            for directory in (source_run, gear_checkout, runtime, terrain_dir):
                directory.mkdir()
            mm_server = root / "mm_chunk_server"
            mm_server.write_text("#!/bin/sh\n", encoding="utf-8")
            argv = [
                "--output-root",
                str(root / "runs"),
                "--source-run",
                str(source_run),
                "--gear-checkout",
                str(gear_checkout),
                "--runtime",
                str(runtime),
                "--terrain-dir",
                str(terrain_dir),
                "--mm-server",
                str(mm_server),
                *extra_argv,
            ]
            namespace = manual_demo._parser().parse_args(argv)
            with (
                patch.object(manual_demo, "GatedSimulatorClient", _fake_simulator),
                patch.object(
                    manual_demo, "MMChunkClient", lambda *a, **k: object()
                ),
                patch.object(
                    manual_demo, "GearProcess", _fake_gear
                ),
                patch.object(manual_demo, "PosePublisher", _FakePublisher),
                patch.object(
                    manual_demo, "load_joint_contract", lambda *a, **k: object()
                ),
                patch.object(
                    manual_demo, "SourceValidator", lambda *a, **k: object()
                ),
            ):
                with self.assertRaises(_StopAfterSimulator):
                    manual_demo.run_demo(namespace)
        return captured

    def test_freeze_on_fall_matches_onscreen_when_visible(self) -> None:
        captured = self._drive_run_demo(["--onscreen"])
        self.assertIs(captured["onscreen"], True)
        self.assertIs(captured["freeze_on_fall"], True)

    def test_freeze_on_fall_disabled_when_headless(self) -> None:
        captured = self._drive_run_demo([])
        self.assertIs(captured["onscreen"], False)
        self.assertIs(captured["freeze_on_fall"], False)

    def test_startup_processes_share_live_operator_cancellation(self) -> None:
        captured = self._drive_run_demo([])
        self.assertTrue(callable(captured["gear_cancelled"]))
        self.assertIs(captured["gear_cancelled"], captured["cancelled"])
        self.assertIs(captured["cancelled"](), False)


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

    def test_responsive_source_intervals_defaults_to_ten(self) -> None:
        self.assertEqual(
            _parser().parse_args([]).responsive_source_intervals, 10
        )

    def test_responsive_source_intervals_accepts_only_five_and_ten(self) -> None:
        self.assertEqual(
            _parser().parse_args(
                ["--responsive-source-intervals", "5"]
            ).responsive_source_intervals,
            5,
        )
        self.assertEqual(
            _parser().parse_args(
                ["--responsive-source-intervals", "10"]
            ).responsive_source_intervals,
            10,
        )
        with self.assertRaises(SystemExit):
            _parser().parse_args(["--responsive-source-intervals", "7"])

    def test_movement_model_defaults_to_raw(self) -> None:
        self.assertEqual(_parser().parse_args([]).movement_model, "raw")

    def test_movement_model_accepts_holden_v1(self) -> None:
        self.assertEqual(
            _parser().parse_args(["--movement-model", "holden-v1"]).movement_model,
            "holden-v1",
        )

    def test_movement_model_rejects_other_values(self) -> None:
        with self.assertRaises(SystemExit):
            _parser().parse_args(["--movement-model", "other"])

    def test_mm_server_defaults_to_committed_build_path(self) -> None:
        default = _parser().parse_args([]).mm_server
        self.assertTrue(default.endswith("sonic/build/mm_chunk_server"))

    def test_mm_server_override_is_honored(self) -> None:
        args = _parser().parse_args(["--mm-server", "/tmp/x/mm_chunk_server"])
        self.assertEqual(args.mm_server, "/tmp/x/mm_chunk_server")

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
        return {
            "coordinate_signature": "holden",
            "id": "hello",
            "supported_movement_models": ["raw", "holden-v1"],
        }

    def reset(self, config: object, *, session_id: str) -> dict:
        self._log.append("mm.reset")
        self.reset_config = config
        return {
            "scene": {
                "scene_id": self._scene_id,
                "route_id": self._route_id,
            },
            "movement_model": {
                "profile": getattr(config, "movement_model", "raw"),
                "acceleration_mps2": 1.5,
                "deceleration_mps2": 2.0,
                "directional_acceleration": False,
                "turn_strength": False,
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
    def _run(self, scene_id: str, route_id: str, weight: float,
             movement_model: str = "raw"):
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
            movement_model=movement_model,
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
        self.assertEqual(register_calls[0]["hello"], {
            "coordinate_signature": "holden",
            "id": "hello",
            "supported_movement_models": ["raw", "holden-v1"],
        })
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
        self.assertEqual(mm.reset_config.movement_model, "raw")

    def test_reset_session_config_carries_selected_movement_model(self) -> None:
        *_, mm, _simulator, _result = self._run(
            "sonic-flat-baseline", "flat-12s", 0.0, movement_model="holden-v1"
        )
        self.assertEqual(mm.reset_config.movement_model, "holden-v1")

    def test_hello_without_selected_profile_fails_before_reset(self) -> None:
        log: list[str] = []

        class _NoCapabilityMM(_FakeMM):
            def hello(self) -> dict:
                self._log.append("mm.hello")
                return {"coordinate_signature": "holden", "id": "hello"}

        mm = _NoCapabilityMM(log, "sonic-flat-baseline", "flat-12s")
        simulator = _FakeSimulator(log)
        deps = DemoDependencies(
            register_scene=lambda *a, **k: self.fail("scene registered"),
            build_initial_state=lambda *a, **k: self.fail("state built"),
        )
        with self.assertRaisesRegex(ValueError, "movement model"):
            _run_startup_transaction(
                deps,
                mm=mm,
                simulator=simulator,
                validator=_FakeValidator(),
                scene_id="sonic-flat-baseline",
                route_id="flat-12s",
                terrain_weight=0.0,
                session_id="manual-1",
                log_dir=Path("/runs/manual/bootstrap-sim-logs"),
                movement_model="holden-v1",
            )
        self.assertEqual(log, ["mm.hello"])


class ScoredControlStartupTests(unittest.TestCase):
    class _Gear:
        def __init__(self, calls, *, barrier_error=None, receipt_error=None):
            self.calls = calls
            self.barrier_error = barrier_error
            self.receipt_error = receipt_error
            self.stopped = True

        def continue_group(self):
            self.calls.append("gear.continue_group")
            self.stopped = False

        def activate_control(self):
            self.calls.append("gear.activate_control")

        def wait_for_first_policy_action(self):
            self.calls.append("gear.wait_for_first_policy_action")
            if self.barrier_error is not None:
                raise self.barrier_error
            return {
                "index": 1,
                "time_ms": 20.0,
                "time_monotonic_ms": 22.0,
                "action": (0.25,) * 29,
            }

        def wait_for_received_policy_command(self, simulator):
            self.calls.append("gear.wait_for_received_policy_command")
            if self.receipt_error is not None:
                raise self.receipt_error
            return {
                "index": 2,
                "time_ms": 40.0,
                "time_monotonic_ms": 42.0,
                "action": (-0.5,) * 29,
                "q_target": tuple(index / 10.0 for index in range(29)),
            }

        def stop_group(self):
            self.calls.append("gear.stop_group")
            self.stopped = True

        def group_is_stopped(self):
            return self.stopped

        def group_is_resumed(self):
            return not self.stopped

        @staticmethod
        def require_alive():
            return None

    class _Simulator:
        sim_dt = 0.005

        def __init__(self, calls):
            self.calls = calls
            self.advance_calls = 0

        def require_alive(self):
            self.calls.append("simulator.require_alive")

        def advance(self, _steps):
            self.advance_calls += 1
            raise AssertionError("startup must not advance physics")

    def test_action_readiness_precedes_gate_pause(self):
        calls = []
        gear = self._Gear(calls)
        simulator = self._Simulator(calls)

        output = StringIO()
        with redirect_stdout(output):
            gate = _activate_scored_control(gear, simulator)

        self.assertIsInstance(gate, SimulationPolicyGate)
        self.assertEqual(
            calls,
            [
                "gear.continue_group",
                "gear.activate_control",
                "gear.wait_for_first_policy_action",
                "gear.wait_for_received_policy_command",
                "gear.stop_group",
                "simulator.require_alive",
            ],
        )
        self.assertTrue(gate.is_paused)
        self.assertEqual(simulator.advance_calls, 0)
        self.assertEqual(
            output.getvalue(),
            "SONIC first action ready: index=1 policy_time=20.000ms\n"
            "SONIC policy command received: index=2 policy_time=40.000ms\n",
        )

    def test_barrier_failure_does_not_pause_gate_or_advance_physics(self):
        calls = []
        gear = self._Gear(calls, barrier_error=ProcessError("no action"))
        simulator = self._Simulator(calls)

        with self.assertRaisesRegex(ProcessError, "no action"):
            _activate_scored_control(gear, simulator)

        self.assertEqual(
            calls,
            [
                "gear.continue_group",
                "gear.activate_control",
                "gear.wait_for_first_policy_action",
            ],
        )
        self.assertEqual(simulator.advance_calls, 0)

    def test_receiver_failure_does_not_pause_gate_or_advance_physics(self):
        calls = []
        gear = self._Gear(
            calls,
            receipt_error=ProcessError("no received policy command"),
        )
        simulator = self._Simulator(calls)

        with self.assertRaisesRegex(ProcessError, "no received"):
            _activate_scored_control(gear, simulator)

        self.assertEqual(
            calls,
            [
                "gear.continue_group",
                "gear.activate_control",
                "gear.wait_for_first_policy_action",
                "gear.wait_for_received_policy_command",
            ],
        )
        self.assertEqual(simulator.advance_calls, 0)


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

    def test_short_prefix_reports_matched_point_four_second_two_prefix_lookahead(
        self,
    ) -> None:
        command, mapped = _mapped_boundary()
        events: list[str] = []

        _consume_x11_boundary(
            control_loop=_BoundaryControlLoop(command, mapped),
            simulator=_BoundarySimulator(),
            gate=_BoundaryGate(),
            generate_and_publish=lambda value, *, wait: None,
            chunk_index=12,
            steps_per_chunk=10,
            preload_chunks=2,
            camera_state=CameraDeliveryState(),
            event_sink=events.append,
            control_prefix="CONTROL chunk=",
            camera_disabled_prefix="CAMERA DISABLED",
            prefix_duration_s=0.2,
        )

        self.assertIn("presents_in=0.400s", events[-1])

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

    def test_responsive_loop_delays_trace_until_prefix_is_presented(self) -> None:
        from mm_sonic.manual_demo import _run_responsive_x11_loop
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_wiring import ManualChunkCommitter

        command0 = CommandSample(0, (0.25, -0.4, 0.0), (1.0, 0.0, 0.0, 0.0))
        command1 = CommandSample(1, (-0.25, 0.4, 0.0), (1.0, 0.0, 0.0, 0.0))
        snapshots = [
            IntentSnapshot(revision=1, observed_ns=5, command=command0),
            IntentSnapshot(revision=2, observed_ns=6, command=command1),
        ]
        mapped = self._mapped()
        mailbox = _ResponsiveMailbox(snapshots, mapped)
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
                import numpy as np

                class _Checked:
                    command = {
                        "applied_velocity_holden": np.zeros(
                            (10, 3), dtype=np.float32
                        ),
                        "applied_heading_holden_wxyz": np.tile(
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                            (10, 1),
                        ),
                    }

                return _Checked()

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
            chunks=2,
            camera_state=camera_state,
            event_sink=events.append,
            trace_sink=traces.append,
            session_id="session",
            camera_disabled_prefix="CAMERA DISABLED",
        )

        # Two prefixes committed/released; only the first has reached physics.
        self.assertEqual(log.count("release"), 2)
        self.assertEqual(
            published,
            [("timeline", False), ("timeline", False)],
        )
        # Camera delivered synchronously via on_sample.
        # The second sample carries the same camera sequence, so delivery is
        # correctly de-duplicated by the existing camera semantics.
        self.assertEqual(len(simulator.camera_calls), 1)
        # Candidate 0 is traced using the second release. Candidate 1 remains
        # queued at loop exit and must not receive candidate 0's observation.
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].input_transition_id, "session:rev:000001")
        self.assertEqual(traces[0].presented_prefix_id, "session:candidate:000000")
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
            applied_velocity_mujoco_first=(0.1, -0.2, 0.0),
            applied_velocity_mujoco_last=(-0.3, -0.4, 0.0),
            applied_heading_mujoco_wxyz_first=(1.0, 0.0, 0.0, 0.0),
            applied_heading_mujoco_wxyz_last=(0.99912283, 0.0, 0.0, -0.04187565),
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

    def test_evidence_reports_dynamic_five_interval_horizon(self):
        from mm_sonic.manual_demo import _write_responsive_evidence

        with TemporaryDirectory() as tmp:
            bundle = _FakeBundle(Path(tmp))
            evidence = _write_responsive_evidence(
                bundle,
                True,
                [self._trace_record((0.12, 0.0, 0.0))],
                committed_prefixes=1,
                source_intervals=5,
            )
            self.assertEqual(evidence["source_intervals"], 5)
            self.assertEqual(evidence["lookahead_seconds"], 0.2)

    def test_evidence_reports_default_ten_interval_horizon(self):
        from mm_sonic.manual_demo import _write_responsive_evidence

        with TemporaryDirectory() as tmp:
            bundle = _FakeBundle(Path(tmp))
            evidence = _write_responsive_evidence(
                bundle,
                True,
                [self._trace_record((0.12, 0.0, 0.0))],
                committed_prefixes=1,
            )
            self.assertEqual(evidence["source_intervals"], 10)
            self.assertEqual(evidence["lookahead_seconds"], 0.4)

    def test_responsive_writes_jsonl_and_honest_summary(self):
        from mm_sonic.manual_demo import _write_responsive_evidence

        with TemporaryDirectory() as tmp:
            bundle = _FakeBundle(Path(tmp))
            traces = [
                self._trace_record((0.24, 0.0, 0.0)),
                self._trace_record(None),
            ]
            evidence = _write_responsive_evidence(
                bundle,
                True,
                traces,
                committed_prefixes=3,
            )

            self.assertEqual(evidence["committed_prefixes"], 3)
            self.assertEqual(evidence["presented_prefixes"], 2)
            self.assertEqual(evidence["queued_prefixes_unobserved"], 1)
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


class ResponsiveSourceIntervalValidationTests(unittest.TestCase):
    def test_five_intervals_rejected_in_script_mode_before_any_run_bundle(
        self,
    ) -> None:
        # A responsive 0.2s horizon is valid only for interactive X11.  Asking
        # for five source intervals without --responsive must fail before any
        # run bundle is materialized.
        with TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ContractError, "responsive"):
                main(
                    [
                        "--responsive-source-intervals",
                        "5",
                        "--mode",
                        "script",
                        "--output-root",
                        output_root,
                    ]
                )
            self.assertEqual(list(Path(output_root).iterdir()), [])

    def test_five_intervals_rejected_in_interactive_terminal_before_run_bundle(
        self,
    ) -> None:
        with TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(ContractError, "responsive"):
                main(
                    [
                        "--responsive-source-intervals",
                        "5",
                        "--mode",
                        "interactive",
                        "--input-source",
                        "terminal",
                        "--output-root",
                        output_root,
                    ]
                )
            self.assertEqual(list(Path(output_root).iterdir()), [])

    def test_default_ten_intervals_never_trip_the_responsive_guard(self) -> None:
        from mm_sonic.manual_demo import _resolve_responsive_source_intervals

        # Default 10 is always allowed, even on the non-responsive path.
        args = _parser().parse_args(["--mode", "script"])
        self.assertEqual(_resolve_responsive_source_intervals(args), 10)

    def test_five_intervals_resolve_only_under_responsive_x11(self) -> None:
        from mm_sonic.manual_demo import _resolve_responsive_source_intervals

        args = _parser().parse_args(
            [
                "--responsive",
                "--responsive-source-intervals",
                "5",
                "--mode",
                "interactive",
                "--input-source",
                "x11",
            ]
        )
        self.assertEqual(_resolve_responsive_source_intervals(args), 5)


if __name__ == "__main__":
    unittest.main()
