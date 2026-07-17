from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mm_sonic import cli as cli_module
from mm_sonic.artifacts import RunBundle
from mm_sonic.joints import ContractError
from mm_sonic.process import AdvanceResult, GearProcess, ProcessError
from mm_sonic.timeline import CanonicalTargetBuffer
from mm_sonic.zmq_v1 import encode_pose_v1


# Independent literals copied from the authenticated pinned GEAR source.  These
# deliberately do not import the production constants they are meant to pin.
LOGGER_PERMUTATION = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10,
    14, 18, 2, 5, 8, 11, 15, 19, 21, 23,
    25, 27, 12, 16, 20, 22, 24, 26, 28,
)
LOGGER_INVERSE = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3,
    9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
    24, 18, 25, 19, 26, 20, 27, 21, 28,
)
COMMON_HEADER = (
    "index",
    "time_ms",
    "time_realtime_ms",
    "time_monotonic_ms",
    "ros_timestamp",
)
Q_HEADER = COMMON_HEADER + tuple(f"q_{index}" for index in range(29))
BASE_HEADER = COMMON_HEADER + ("base_qw", "base_qx", "base_qy", "base_qz")


def canonical(count: int = 441) -> CanonicalTargetBuffer:
    frame = np.arange(count, dtype=np.float32)[:, None]
    joint = np.arange(29, dtype=np.float32)[None, :]
    positions = np.ascontiguousarray(frame / 100.0 + joint / 1000.0, dtype="<f4")
    velocities = np.zeros((count, 29), dtype="<f4")
    quaternions = np.zeros((count, 4), dtype="<f4")
    quaternions[:, 0] = 1.0
    return CanonicalTargetBuffer(
        joint_position=positions,
        joint_velocity=velocities,
        body_quat_w=quaternions,
        frame_index=np.arange(count, dtype="<i8"),
    )


class TemporaryCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def script(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def process(
        self,
        script: Path,
        *,
        profile: str,
        stem: str,
        env: dict[str, str],
    ) -> GearProcess:
        return GearProcess(
            run_root=self.root,
            command=[sys.executable, "-u", str(script)],
            target_motion_logfile=self.root / f"{stem}.target.csv",
            logs_dir=self.root / f"{stem}.gear-logs",
            stdout_archive=self.root / f"{stem}.stdout",
            stderr_archive=self.root / f"{stem}.stderr",
            launch_profile=profile,
            readiness_timeout_s=0.5,
            readiness_poll_s=0.005,
            signal_poll_s=0.005,
            stop_grace_s=0.05,
            term_grace_s=0.05,
            kill_grace_s=0.2,
            env=env,
        )


class GearWaitLifecycleTests(TemporaryCase):
    def test_stream_preloads_in_wait_then_control_activation_never_stops_on_marker(
        self,
    ) -> None:
        keys = self.root / "stream.keys"
        target = self.root / "stream.target.csv"
        child = self.script(
            "stream_wait.py",
            r'''
            import os
            from pathlib import Path
            import termios

            key_path = Path(os.environ["KEY_PATH"])
            Path(os.environ["TARGET_PATH"]).write_bytes(b"")
            print("Initialized ZMQ endpoint interface", flush=True)
            print("Init Done", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)

            key = os.read(0, 1)
            key_path.write_bytes(key)
            print("ZMQ STREAMING MODE: ENABLED", flush=True)
            keys = os.read(0, 2)
            key_path.write_bytes(key_path.read_bytes() + keys)
            assert keys == b"qe"
            print("Delta heading left: 0.1 rad", flush=True)
            print("Delta heading right: 0 rad", flush=True)
            key = os.read(0, 1)
            key_path.write_bytes(key_path.read_bytes() + key)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.process(
            child,
            profile="zmq_stream",
            stem="stream",
            env=dict(os.environ, KEY_PATH=str(keys), TARGET_PATH=str(target)),
        )
        try:
            gear.start_to_wait_for_control()
            self.assertTrue(gear.wait_for_control_ready)
            self.assertFalse(gear.control_active)
            self.assertFalse(keys.exists())

            gear.enable_stream_for_preload()
            self.assertEqual(keys.read_bytes(), b"\nqe")
            self.assertTrue(gear.wait_for_control_ready)
            self.assertFalse(gear.control_active)
            wait_evidence = cli_module._audit_wait_for_control_epoch(
                gear, target, gear.logs_dir
            )
            self.assertEqual(wait_evidence.target_rows, 0)
            self.assertEqual(wait_evidence.q_rows, 0)
            self.assertEqual(wait_evidence.base_rows, 0)

            gear.stop_group()
            stop_count = gear.signal_history.count(signal.SIGSTOP)
            gear.continue_group()
            gear.activate_control()
            self.assertEqual(keys.read_bytes(), b"\nqe]")
            self.assertTrue(gear.control_active)
            self.assertTrue(gear.group_is_resumed())
            self.assertEqual(gear.signal_history.count(signal.SIGSTOP), stop_count)
        finally:
            gear.close()
        self.assertFalse(cli_module._process_group_exists(gear.pgid))

    def test_loaded_motion_is_reset_and_armed_while_still_in_wait(self) -> None:
        keys = self.root / "loaded.keys"
        target = self.root / "loaded.target.csv"
        child = self.script(
            "loaded_wait.py",
            r'''
            import os
            from pathlib import Path
            import termios

            key_path = Path(os.environ["KEY_PATH"])
            Path(os.environ["TARGET_PATH"]).write_bytes(b"")
            print("\u2713 Motion data loaded successfully!", flush=True)
            print("Started with motion: known_good.csv (paused at frame 0)", flush=True)
            print("Initialized keyboard input interface (default)", flush=True)
            print("Init Done", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)

            observed = b""
            observed += os.read(0, 1)
            key_path.write_bytes(observed)
            print("Reset motion 0 to frame 0 (paused)", flush=True)
            observed += os.read(0, 1)
            key_path.write_bytes(observed)
            print("Playing motion 0 from frame 0 to end (441 total frames)", flush=True)
            observed += os.read(0, 1)
            key_path.write_bytes(observed)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            while os.read(0, 1).lower() != b"o":
                pass
            ''',
        )
        gear = self.process(
            child,
            profile="loaded_motion",
            stem="loaded",
            env=dict(os.environ, KEY_PATH=str(keys), TARGET_PATH=str(target)),
        )
        try:
            gear.start_to_wait_for_control()
            gear.prepare_loaded_motion_for_scoring()
            self.assertEqual(keys.read_bytes(), b"rt")
            self.assertTrue(gear.wait_for_control_ready)
            self.assertFalse(gear.control_active)
            cli_module._audit_wait_for_control_epoch(gear, target, gear.logs_dir)
            gear.stop_group()
            gear.continue_group()
            gear.activate_control()
            self.assertEqual(keys.read_bytes(), b"rt]")
            self.assertTrue(gear.group_is_resumed())
        finally:
            gear.close()

    def test_scoring_boundary_termination_never_resumes_stopped_group(self) -> None:
        target = self.root / "boundary.target.csv"
        child = self.script(
            "stopped_boundary.py",
            r'''
            import os
            from pathlib import Path
            import signal
            import termios

            target = Path(os.environ["TARGET_PATH"])
            target.write_bytes(b"")
            print("Initialized ZMQ endpoint interface", flush=True)
            print("Init Done", flush=True)
            attrs = termios.tcgetattr(0)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(0, termios.TCSANOW, attrs)
            os.read(0, 1)
            print("ZMQ STREAMING MODE: ENABLED", flush=True)
            assert os.read(0, 2) == b"qe"
            print("Delta heading left: 0.1 rad", flush=True)
            print("Delta heading right: 0 rad", flush=True)
            os.read(0, 1)

            def hidden_tick(_signal, _frame):
                with target.open("ab") as output:
                    output.write(b"row-442\n")

            signal.signal(signal.SIGCONT, hidden_tick)
            print(
                "[Control] DEBUG: operator_state.start=true, "
                "transitioning to CONTROL state",
                flush=True,
            )
            while True:
                signal.pause()
            ''',
        )
        gear = self.process(
            child,
            profile="zmq_stream",
            stem="boundary",
            env=dict(os.environ, TARGET_PATH=str(target)),
        )
        gear.start_to_wait_for_control()
        gear.enable_stream_for_preload()
        gear.stop_group()
        gear.continue_group()
        gear.activate_control()
        gear.stop_group()
        history_start = len(gear.signal_history)
        gear.terminate_stopped_at_scoring_boundary()
        self.assertEqual(target.read_bytes(), b"")
        self.assertEqual(
            gear.signal_history[history_start:], [signal.SIGKILL]
        )
        self.assertFalse(cli_module._process_group_exists(gear.pgid))


class StreamPreloadTests(TemporaryCase):
    class Gear:
        wait_for_control_ready = True
        control_active = False
        input_prepared = True

        def __init__(
            self,
            bundle: RunBundle,
            *,
            fail_marker: int | None = None,
        ) -> None:
            self.publications: list[CanonicalTargetBuffer] = []
            self.markers: list[tuple[int, int, int, int]] = []
            self.fail_marker = fail_marker
            enabled = b"ZMQ STREAMING MODE: ENABLED\n"
            left = b"Delta heading left: 0.1 rad\n"
            right = b"Delta heading right: 0 rad\n"
            self.stdout_path = bundle.write_bytes(
                "dynamic/stream/gear.stdout", enabled + left + right
            )
            self.post_enable_fence = MappingProxyType(
                {
                    "boundary": len(enabled),
                    "end_offset": len(enabled + left + right),
                    "key_sequence": "qe",
                    "left_line": "Delta heading left: 0.1 rad",
                    "right_line": "Delta heading right: 0 rad",
                    "semantics": (
                        "post-enable-reset-tail-complete-with-net-zero-heading"
                    ),
                }
            )

        def require_alive(self) -> None:
            pass

        def group_is_stopped(self) -> bool:
            return False

        def publication_boundary(self) -> int:
            return len(self.stdout_path.read_bytes())

        def on_send(self, buffer: CanonicalTargetBuffer) -> None:
            self.publications.append(buffer)

        def wait_for_stream_processing(
            self,
            boundary: int,
            *,
            frame_count: int,
            global_start: int,
            merged_count: int,
        ) -> MappingProxyType:
            marker_number = len(self.markers) + 1
            if marker_number == self.fail_marker:
                raise ProcessError("missing authenticated consumer marker")
            self.assert_boundary(boundary)
            self.markers.append(
                (boundary, frame_count, global_start, merged_count)
            )
            processing = (
                f"[StreamedMotionMerger] Processing {frame_count} frames, "
                f"incoming_frame_start={global_start}, frame_step=1"
            )
            copied = merged_count - frame_count
            merged = (
                f"[StreamedMotionMerger] Merged motion: {merged_count} frames "
                f"(copied: {copied} + incoming: {frame_count})"
            )
            end_line = (
                "[ZMQEndpointInterface] "
                "*** End of ZMQ decoding processing ***"
            )
            start_bytes = (
                "[ZMQEndpointInterface] *** Starting ZMQ processing ***\n"
            ).encode("ascii")
            before_processing = (
                "[ZMQEndpointInterface] Protocol version: 1\n"
                + (
                    "[ZMQEndpointInterface] Protocol version 1 established\n"
                    if marker_number == 1
                    else ""
                )
            ).encode("ascii")
            processing_bytes = f"{processing}\n".encode("ascii")
            merged_bytes = f"{merged}\n".encode("ascii")
            merged_to_end = (
                "[ZMQEndpointInterface] active_protocol_version_=1\n"
                "[ZMQEndpointInterface] result.motion->GetEncodeMode()=0\n"
                "[ZMQEndpointInterface] motion name: streamed\n"
                "[ZMQEndpointInterface] Merged streamed data\n"
            ).encode("ascii")
            end_bytes = f"{end_line}\n".encode("ascii")
            start_range = (boundary, boundary + len(start_bytes))
            processing_range = (
                start_range[1] + len(before_processing),
                start_range[1] + len(before_processing) + len(processing_bytes),
            )
            merged_range = (
                processing_range[1],
                processing_range[1] + len(merged_bytes),
            )
            end_range = (
                merged_range[1] + len(merged_to_end),
                merged_range[1] + len(merged_to_end) + len(end_bytes),
            )
            with self.stdout_path.open("ab") as output:
                output.write(
                    start_bytes
                    + before_processing
                    + processing_bytes
                    + merged_bytes
                    + merged_to_end
                    + end_bytes
                )
                output.flush()
            return MappingProxyType(
                {
                    "boundary": boundary,
                    "end_offset": end_range[1],
                    "frame_count": frame_count,
                    "global_start": global_start,
                    "merged_count": merged_count,
                    "start_line": (
                        "[ZMQEndpointInterface] *** Starting ZMQ processing ***"
                    ),
                    "processing_line": processing,
                    "merged_line": merged,
                    "end_line": end_line,
                    "start_range": start_range,
                    "processing_range": processing_range,
                    "merged_range": merged_range,
                    "end_range": end_range,
                }
            )

        def assert_boundary(self, boundary: int) -> None:
            if boundary != len(self.stdout_path.read_bytes()):
                raise AssertionError("publication boundary changed")

    class Publisher:
        def __init__(self, bundle: RunBundle, gear: "StreamPreloadTests.Gear") -> None:
            self.bundle = bundle
            self.gear = gear

        def prepare(self, buffer, *, phase, attempt=None):
            message = encode_pose_v1(buffer)
            archive = self.bundle.archive_transmission(
                message,
                first_frame_index=int(buffer.frame_index[0]),
                last_frame_index=int(buffer.frame_index[-1]),
                phase=phase,
                attempt=attempt,
            )
            return SimpleNamespace(buffer=buffer, archive=archive)

        def send_prepared(self, prepared):
            self.gear.on_send(prepared.buffer)
            result = dict(prepared.archive)
            result["local_send_completed"] = True
            return MappingProxyType(result)

    def test_exact_wait_preload_has_logical_padding_and_receipt_fence_transcripts(
        self,
    ) -> None:
        expected = canonical()
        bundle = RunBundle.create(self.root / "runs", "preload", "run")
        gear = self.Gear(bundle)
        try:
            evidence = cli_module._preload_known_good_stream(
                expected,
                self.Publisher(bundle, gear),
                gear,
                gear.post_enable_fence,
            )
            self.assertEqual(evidence.readiness_publication["phase"], "readiness")
            self.assertEqual(len(evidence.logical_publications), 22)
            self.assertEqual(
                {record["phase"] for record in evidence.logical_publications},
                {"logical"},
            )
            self.assertEqual(evidence.padding_publication["phase"], "padding")
            self.assertEqual(
                evidence.receipt_fence_publication["phase"], "receipt_fence"
            )
            self.assertTrue(
                all(
                    str(record["message_path"]).startswith(
                        "transmitted/logical/"
                    )
                    for record in evidence.logical_publications
                )
            )
            self.assertTrue(
                str(evidence.padding_publication["message_path"]).startswith(
                    "transmitted/padding/"
                )
            )
            self.assertTrue(
                str(evidence.receipt_fence_publication["message_path"]).startswith(
                    "transmitted/receipt_fence/"
                )
            )
            logical = [
                index
                for buffer in gear.publications[:23]
                for index in buffer.frame_index.tolist()
            ]
            self.assertEqual(logical, list(range(441)))

            padding = gear.publications[23]
            self.assertEqual(padding.frame_index.tolist(), list(range(441, 487)))
            np.testing.assert_array_equal(
                padding.joint_position,
                np.repeat(expected.joint_position[-1:], 46, axis=0),
            )
            fence = gear.publications[24]
            self.assertEqual(fence.frame_index.tolist(), [487])
            np.testing.assert_array_equal(
                fence.joint_position, expected.joint_position[-1:]
            )
            self.assertEqual(len(evidence.consumer_markers), 25)
            self.assertEqual(
                [marker[1:] for marker in gear.markers[-2:]],
                [(46, 441, 487), (1, 487, 488)],
            )
            self.assertEqual(
                evidence.causal_fences[-2:],
                ("padding-start-fences-logical-440", "fence-start-fences-padding"),
            )
            audit = cli_module._audit_known_good_stream_preload(
                expected, bundle, evidence
            )
            self.assertTrue(audit["readiness_exact"])
            self.assertTrue(audit["logical_exact"])
            self.assertTrue(audit["padding_exact"])
            self.assertTrue(audit["receipt_fence_exact"])
            self.assertEqual(audit["readiness_frames"], 1)
            self.assertEqual(audit["logical_frames"], 440)
            self.assertEqual(audit["padding_frames"], 46)
            self.assertEqual(audit["receipt_fence_frames"], 1)
            self.assertEqual(audit["consumer_marker_count"], 25)
            self.assertEqual(audit["causal_fence_count"], 24)
            self.assertRegex(audit["consumer_markers_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(audit["causal_fences_sha256"], r"^[0-9a-f]{64}$")
        finally:
            bundle.__del__()

    def test_missing_consumer_marker_fails_preload(self) -> None:
        bundle = RunBundle.create(self.root / "runs", "marker", "run")
        gear = self.Gear(bundle, fail_marker=24)
        try:
            with self.assertRaisesRegex(ProcessError, "consumer marker"):
                cli_module._preload_known_good_stream(
                    canonical(),
                    self.Publisher(bundle, gear),
                    gear,
                    gear.post_enable_fence,
                )
        finally:
            bundle.__del__()

    def test_auditor_rejects_missing_padding_or_receipt_fence(self) -> None:
        expected = canonical()
        bundle = RunBundle.create(self.root / "runs", "audit", "run")
        gear = self.Gear(bundle)
        try:
            evidence = cli_module._preload_known_good_stream(
                expected,
                self.Publisher(bundle, gear),
                gear,
                gear.post_enable_fence,
            )
            for label, field, phase in (
                ("padding", "padding_publication", "padding"),
                ("fence", "receipt_fence_publication", "receipt_fence"),
            ):
                with self.subTest(label=label):
                    damaged = replace(evidence, **{field: MappingProxyType({})})
                    with self.assertRaisesRegex(ContractError, phase):
                        cli_module._audit_known_good_stream_preload(
                            expected, bundle, damaged
                        )
        finally:
            bundle.__del__()

    def test_auditor_rejects_mutated_consumer_line_or_causal_boundary(self) -> None:
        expected = canonical()
        bundle = RunBundle.create(self.root / "runs", "marker-audit", "run")
        gear = self.Gear(bundle)
        try:
            evidence = cli_module._preload_known_good_stream(
                expected,
                self.Publisher(bundle, gear),
                gear,
                gear.post_enable_fence,
            )
            first = dict(evidence.consumer_markers[0])
            first["merged_line"] = "forged"
            damaged_lines = replace(
                evidence,
                consumer_markers=(
                    MappingProxyType(first),
                    *evidence.consumer_markers[1:],
                ),
            )
            with self.assertRaisesRegex(ContractError, "consumer marker"):
                cli_module._audit_known_good_stream_preload(
                    expected, bundle, damaged_lines
                )

            second = dict(evidence.consumer_markers[1])
            second["boundary"] = 0
            damaged_boundary = replace(
                evidence,
                consumer_markers=(
                    evidence.consumer_markers[0],
                    MappingProxyType(second),
                    *evidence.consumer_markers[2:],
                ),
            )
            with self.assertRaisesRegex(ContractError, "causal"):
                cli_module._audit_known_good_stream_preload(
                    expected, bundle, damaged_boundary
                )
        finally:
            bundle.__del__()


class TargetCoverageTests(TemporaryCase):
    class Gear:
        wait_for_control_ready = True
        input_prepared = True
        control_active = True

        def __init__(self, target: Path, hidden_row: bytes | None = None) -> None:
            self.target = target
            self.hidden_row = hidden_row
            self.stopped = False
            self.stop_calls = 0

        def require_alive(self) -> None:
            pass

        def group_is_stopped(self) -> bool:
            return self.stopped

        def group_is_resumed(self) -> bool:
            return not self.stopped

        def stop_group(self) -> None:
            self.stop_calls += 1
            self.stopped = True
            if self.hidden_row is not None:
                with self.target.open("ab") as output:
                    output.write(self.hidden_row)

    class Simulator:
        sim_dt = 0.002

        def __init__(self, operation) -> None:
            self.operation = operation
            self.steps = 0
            self.state_rows = 0
            self.contact_rows = 0

        def advance(self, steps: int) -> AdvanceResult:
            start = self.steps * self.sim_dt
            self.steps += steps
            self.state_rows += steps // 17
            self.contact_rows += steps
            self.operation()
            return AdvanceResult(
                steps=steps,
                sim_time_start_s=start,
                sim_time_end_s=self.steps * self.sim_dt,
                state_rows=steps // 17,
                contact_rows=steps,
            )

        def snapshot(self):
            return {
                "steps": self.steps,
                "sim_time_s": self.steps * self.sim_dt,
                "state_rows": self.state_rows,
                "contact_rows": self.contact_rows,
            }

    def rows(self, expected: CanonicalTargetBuffer) -> list[bytes]:
        return [
            cli_module._official_target_row(expected, index)
            for index in range(expected.count)
        ]

    def test_variable_wall_clock_controller_stops_on_rows_not_step_ratio(self) -> None:
        expected = canonical()
        rows = self.rows(expected)
        target = self.root / "variable.target.csv"
        target.write_bytes(b"")
        schedule = iter([19, 21] * 11 + [1])
        cursor = 0

        def emit() -> None:
            nonlocal cursor
            count = next(schedule)
            with target.open("ab") as output:
                output.write(b"".join(rows[cursor : cursor + count]))
            cursor += count

        gear = self.Gear(target)
        result = cli_module._drive_authoritative_target_coverage(
            gear,
            self.Simulator(emit),
            target,
            expected,
            increment_steps=200,
            maximum_wall_seconds=1.0,
        )
        self.assertEqual(result.target_rows, 441)
        self.assertEqual(result.simulator_steps, 4600)
        self.assertAlmostEqual(result.simulator_duration_s, 9.2)
        self.assertEqual(result.contact_rows, 4600)
        self.assertGreater(result.wall_duration_s, 0.0)
        self.assertTrue(gear.stopped)

    def test_coverage_counters_exclude_the_stopped_wait_low_state_prime(self) -> None:
        expected = canonical()
        rows = self.rows(expected)
        target = self.root / "primed.target.csv"
        target.write_bytes(b"")

        def emit() -> None:
            target.write_bytes(b"".join(rows))

        simulator = self.Simulator(emit)
        simulator.steps = 1
        simulator.contact_rows = 1
        result = cli_module._drive_authoritative_target_coverage(
            self.Gear(target),
            simulator,
            target,
            expected,
            maximum_wall_seconds=0.2,
        )
        self.assertEqual(result.control_drive_steps, 1)
        self.assertAlmostEqual(result.control_drive_duration_s, 0.002)
        self.assertEqual(result.simulator_steps, 2)
        self.assertAlmostEqual(result.simulator_duration_s, 0.004)
        self.assertEqual(result.contact_rows, 2)

    def test_target_overshoot_is_rejected_and_stopped(self) -> None:
        expected = canonical()
        rows = self.rows(expected)
        target = self.root / "overshoot.target.csv"
        target.write_bytes(b"")

        def emit() -> None:
            target.write_bytes(b"".join(rows + [rows[-1]]))

        gear = self.Gear(target)
        with self.assertRaisesRegex(ContractError, "overshoot"):
            cli_module._drive_authoritative_target_coverage(
                gear,
                self.Simulator(emit),
                target,
                expected,
                maximum_wall_seconds=0.2,
            )
        self.assertGreaterEqual(gear.stop_calls, 1)

    def test_partial_target_line_is_rejected_and_stopped(self) -> None:
        expected = canonical()
        target = self.root / "partial.target.csv"
        target.write_bytes(b"")

        def emit() -> None:
            target.write_bytes(b"1,2,3")

        gear = self.Gear(target)
        with self.assertRaisesRegex(ContractError, "partial"):
            cli_module._drive_authoritative_target_coverage(
                gear,
                self.Simulator(emit),
                target,
                expected,
                maximum_wall_seconds=0.2,
            )
        self.assertGreaterEqual(gear.stop_calls, 1)

    def test_timeout_is_rejected_and_stopped(self) -> None:
        expected = canonical()
        target = self.root / "timeout.target.csv"
        target.write_bytes(b"")

        def emit() -> None:
            time.sleep(0.002)

        gear = self.Gear(target)
        with self.assertRaisesRegex(ProcessError, "timed out"):
            cli_module._drive_authoritative_target_coverage(
                gear,
                self.Simulator(emit),
                target,
                expected,
                maximum_wall_seconds=0.005,
            )
        self.assertGreaterEqual(gear.stop_calls, 1)

    def test_mismatch_duplicate_and_omission_are_rejected(self) -> None:
        expected = canonical()
        rows = self.rows(expected)
        variants = {
            "mismatch": [rows[1]],
            "duplicate": [rows[0], rows[0]],
            "omission": [rows[0], rows[2]],
        }
        for label, emitted in variants.items():
            with self.subTest(label=label):
                target = self.root / f"{label}.target.csv"
                target.write_bytes(b"")
                calls = 0

                def emit() -> None:
                    nonlocal calls
                    if calls == 0:
                        target.write_bytes(b"".join(emitted))
                    calls += 1

                gear = self.Gear(target)
                with self.assertRaisesRegex(ContractError, "canonical"):
                    cli_module._drive_authoritative_target_coverage(
                        gear,
                        self.Simulator(emit),
                        target,
                        expected,
                        maximum_wall_seconds=0.2,
                    )
                self.assertGreaterEqual(gear.stop_calls, 1)

    def test_inode_change_is_rejected_and_stopped(self) -> None:
        expected = canonical()
        target = self.root / "inode.target.csv"
        target.write_bytes(b"")

        def emit() -> None:
            target.unlink()
            target.write_bytes(cli_module._official_target_row(expected, 0))

        gear = self.Gear(target)
        with self.assertRaisesRegex(ContractError, "identity"):
            cli_module._drive_authoritative_target_coverage(
                gear,
                self.Simulator(emit),
                target,
                expected,
                maximum_wall_seconds=0.2,
            )
        self.assertGreaterEqual(gear.stop_calls, 1)

    def test_hidden_post_stop_tick_is_rejected(self) -> None:
        expected = canonical()
        rows = self.rows(expected)
        target = self.root / "hidden.target.csv"
        target.write_bytes(b"")

        def emit() -> None:
            target.write_bytes(b"".join(rows))

        gear = self.Gear(target, hidden_row=rows[-1])
        with self.assertRaisesRegex(ContractError, "post-stop"):
            cli_module._drive_authoritative_target_coverage(
                gear,
                self.Simulator(emit),
                target,
                expected,
                maximum_wall_seconds=0.2,
            )
        self.assertTrue(gear.stopped)


class ScoredEpochTests(TemporaryCase):
    class Gear:
        def __init__(self, root: Path, mode: str) -> None:
            self.mode = mode
            self.target_motion_logfile = root / f"{mode}.target.csv"
            self.logs_dir = root / f"{mode}.logs"
            self.wait_for_control_ready = False
            self.input_prepared = False
            self.control_active = False
            self.stopped = False
            self.closed = False
            self.events: list[str] = []
            self.pgid = 12345

        @property
        def startup_markers_ready(self) -> bool:
            return self.wait_for_control_ready

        def start_to_wait_for_control(self) -> None:
            self.events.append("start-wait")
            self.wait_for_control_ready = True

        def prepare_loaded_motion_for_scoring(self) -> None:
            self.events.append("prepare-file")
            self.input_prepared = True

        def enable_stream_for_preload(self) -> None:
            self.events.append("enable-stream")
            self.input_prepared = True

        def stop_group(self) -> None:
            self.events.append("stop-wait")
            self.stopped = True

        def continue_group(self) -> None:
            self.events.append("resume-once")
            self.stopped = False

        def activate_control(self) -> None:
            if self.stopped or not self.input_prepared:
                raise AssertionError("CONTROL activated outside prepared resume")
            self.events.append("activate-control")
            self.control_active = True

        def group_is_stopped(self) -> bool:
            return self.stopped

        def terminate_stopped_at_scoring_boundary(self) -> None:
            if not self.stopped:
                raise AssertionError("scoring boundary was not stopped")
            self.events.append("terminate-stopped")
            self.closed = True

        def close(self) -> None:
            self.events.append("close")
            self.closed = True

    class Simulator:
        sim_dt = 0.002

        def __init__(self) -> None:
            self.events: list[str] = []
            self.closed = False

        def hello(self):
            self.events.append("hello")
            return {}

        def reset(self, **_kwargs):
            self.events.append("reset")
            return {"nq": 36, "sim_dt_s": self.sim_dt, "sim_time_s": 0.0}

        def advance(self, steps: int) -> AdvanceResult:
            self.events.append(f"advance-{steps}")
            return AdvanceResult(steps, 0.0, steps * self.sim_dt, steps, steps)

        def close(self) -> None:
            self.events.append("close")
            self.closed = True

    class Publisher:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def test_blocked_file_and_stream_preparation_cancel_and_cleanup_promptly(self) -> None:
        original_drive = cli_module._drive_simulator_until

        def bounded_drive(operation, simulator, **kwargs):
            return original_drive(
                operation,
                simulator,
                maximum_seconds=0.02,
                cold_start_maximum_seconds=0.02,
                **kwargs,
            )

        for mode in ("file", "stream"):
            with self.subTest(mode=mode):
                cancellation = threading.Event()

                class BlockingGear(self.Gear):
                    def prepare_loaded_motion_for_scoring(inner_self) -> None:
                        inner_self.events.append("prepare-file-blocked")
                        cancellation.wait(0.3)

                    def enable_stream_for_preload(inner_self):
                        inner_self.events.append("enable-stream-blocked")
                        cancellation.wait(0.3)
                        return MappingProxyType({})

                gear = BlockingGear(self.root, mode)
                simulator = self.Simulator()
                publisher = self.Publisher() if mode == "stream" else None
                started = time.monotonic()
                with (
                    patch.object(
                        cli_module,
                        "_drive_simulator_until",
                        side_effect=bounded_drive,
                    ),
                    patch.object(
                        cli_module, "_process_group_exists", return_value=False
                    ),
                ):
                    with self.assertRaisesRegex(ProcessError, "input-preparation"):
                        cli_module._execute_known_good_scoring_epoch(
                            mode=mode,
                            gear=gear,
                            simulator=simulator,
                            scene=SimpleNamespace(
                                gear_scene_xml=self.root / "scene.xml"
                            ),
                            initial_qpos=np.zeros(36),
                            canonical=canonical(),
                            body_position=np.zeros((441, 3)),
                            bundle=SimpleNamespace(path=self.root),
                            bootstrap_cancellation=cancellation,
                            publisher=publisher,
                        )
                elapsed = time.monotonic() - started
                self.assertTrue(cancellation.is_set())
                self.assertLess(elapsed, 0.2)
                self.assertTrue(gear.closed)
                self.assertTrue(simulator.closed)
                if publisher is not None:
                    self.assertTrue(publisher.closed)

    def test_file_and_stream_share_one_cold_wait_to_control_wiring(self) -> None:
        expected = canonical()
        wait = cli_module._WaitForControlEvidence(1, 2, 0, 0, 0, "0" * 64)
        preload = object()
        coverage_reader = SimpleNamespace(close=lambda: None)
        driven_phases: list[str] = []

        def bootstrap(operation, *_args, **kwargs):
            driven_phases.append(kwargs["label"])
            operation()
            return 7

        def coverage(gear, *_args, **_kwargs):
            gear.stopped = True
            gear.events.append("stop-score")
            return cli_module._TargetCoverageResult(
                target_rows=441,
                control_drive_steps=2200,
                control_drive_duration_s=4.4,
                simulator_steps=2201,
                simulator_duration_s=4.402,
                state_rows=220,
                contact_rows=2201,
                wall_duration_s=8.9,
                target_device=1,
                target_inode=2,
                target_sha256="1" * 64,
                retained_reader=coverage_reader,
            )

        def target_audit(result, *_args, **_kwargs):
            result.retained_reader.close()
            return MappingProxyType({"exact": True, "observed_rows": 441})

        executions = []
        with (
            patch.object(cli_module, "_drive_simulator_until", side_effect=bootstrap),
            patch.object(cli_module, "_audit_wait_for_control_epoch", return_value=wait),
            patch.object(cli_module, "_preload_known_good_stream", return_value=preload),
            patch.object(
                cli_module,
                "_audit_known_good_stream_preload",
                return_value=MappingProxyType({"readiness_exact": True}),
            ),
            patch.object(
                cli_module,
                "_drive_authoritative_target_coverage",
                side_effect=coverage,
            ),
            patch.object(
                cli_module, "_audit_closed_target_boundary", side_effect=target_audit
            ),
            patch.object(cli_module, "_parse_gear_control_logs", return_value=object()),
            patch.object(
                cli_module,
                "_tracking_from_gear_control_rows",
                return_value={"frame_count": 441, "pairing": "same-control-tick-positional"},
            ),
            patch.object(cli_module, "_process_group_exists", return_value=False),
        ):
            for mode in ("file", "stream"):
                gear = self.Gear(self.root, mode)
                simulator = self.Simulator()
                publisher = self.Publisher() if mode == "stream" else None
                execution = cli_module._execute_known_good_scoring_epoch(
                    mode=mode,
                    gear=gear,
                    simulator=simulator,
                    scene=SimpleNamespace(gear_scene_xml=self.root / "scene.xml"),
                    initial_qpos=np.zeros(36),
                    canonical=expected,
                    body_position=np.zeros((441, 3)),
                    bundle=SimpleNamespace(path=self.root),
                    bootstrap_cancellation=SimpleNamespace(),
                    publisher=publisher,
                )
                executions.append(execution)
                preparation = "prepare-file" if mode == "file" else "enable-stream"
                self.assertEqual(
                    gear.events[:7],
                    [
                        "start-wait",
                        preparation,
                        "stop-wait",
                        "resume-once",
                        "activate-control",
                        "stop-score",
                        "terminate-stopped",
                    ],
                )
                self.assertTrue(gear.closed)
                self.assertTrue(simulator.closed)
                if publisher is not None:
                    self.assertTrue(publisher.closed)
                self.assertEqual(execution.bootstrap_steps, 7)
                self.assertEqual(execution.wait_maintenance_steps, 7)
        self.assertEqual(
            driven_phases,
            [
                "file-wait-for-control",
                "file-input-preparation",
                "stream-wait-for-control",
                "stream-input-preparation",
            ],
        )
        self.assertEqual(executions[0].metrics, executions[1].metrics)
        self.assertIsNone(executions[0].preload)
        self.assertIs(executions[1].preload, preload)

    def test_scored_epoch_failure_closes_every_started_resource(self) -> None:
        gear = self.Gear(self.root, "stream")
        simulator = self.Simulator()
        publisher = self.Publisher()

        def bootstrap(operation, *_args, **_kwargs):
            operation()
            return 1

        with (
            patch.object(cli_module, "_drive_simulator_until", side_effect=bootstrap),
            patch.object(
                cli_module,
                "_audit_wait_for_control_epoch",
                return_value=cli_module._WaitForControlEvidence(
                    1, 2, 0, 0, 0, "0" * 64
                ),
            ),
            patch.object(cli_module, "_preload_known_good_stream", return_value=object()),
            patch.object(
                cli_module,
                "_audit_known_good_stream_preload",
                return_value=MappingProxyType({"readiness_exact": True}),
            ),
            patch.object(
                cli_module,
                "_drive_authoritative_target_coverage",
                side_effect=ProcessError("synthetic scored coverage failure"),
            ),
            patch.object(cli_module, "_process_group_exists", return_value=False),
        ):
            with self.assertRaisesRegex(ProcessError, "coverage failure"):
                cli_module._execute_known_good_scoring_epoch(
                    mode="stream",
                    gear=gear,
                    simulator=simulator,
                    scene=SimpleNamespace(gear_scene_xml=self.root / "scene.xml"),
                    initial_qpos=np.zeros(36),
                    canonical=canonical(),
                    body_position=np.zeros((441, 3)),
                    bundle=SimpleNamespace(path=self.root),
                    bootstrap_cancellation=SimpleNamespace(),
                    publisher=publisher,
                )
        self.assertTrue(gear.closed)
        self.assertTrue(simulator.closed)
        self.assertTrue(publisher.closed)

    def test_mujoco_only_reset_cannot_replace_stopped_authenticated_wait_epoch(
        self,
    ) -> None:
        class Gear:
            wait_for_control_ready = True
            input_prepared = True
            control_active = False

            def group_is_stopped(self) -> bool:
                return False

        class Simulator:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []

            def reset(self, **kwargs):
                self.calls.append(("reset", kwargs))
                return {"nq": 36, "sim_dt_s": 0.002, "sim_time_s": 0.0}

            def advance(self, steps):
                self.calls.append(("advance", steps))
                return AdvanceResult(steps, 0.0, steps * 0.002, 0, steps)

        simulator = Simulator()
        with self.assertRaisesRegex(ContractError, "stopped.*WAIT"):
            cli_module._reset_and_prime_scored_epoch(
                Gear(),
                simulator,
                scene_xml=self.root / "scene.xml",
                initial_qpos=np.zeros(36),
                log_dir=self.root / "scored",
            )
        self.assertEqual(simulator.calls, [])


def write_gear_logs(
    root: Path,
    expected: CanonicalTargetBuffer,
    *,
    q_header: tuple[str, ...] = Q_HEADER,
    base_header: tuple[str, ...] = BASE_HEADER,
    q_rows: int | None = None,
    base_rows: int | None = None,
    q_index_override: dict[int, int] | None = None,
    q_monotonic_override: dict[int, float] | None = None,
    base_prefix_override: dict[int, tuple[str, ...]] | None = None,
    q_value_override: dict[tuple[int, int], str] | None = None,
    base_value_override: dict[tuple[int, int], str] | None = None,
) -> None:
    root.mkdir()
    q_count = expected.count if q_rows is None else q_rows
    base_count = expected.count if base_rows is None else base_rows
    raw_joint = expected.joint_position[:, np.asarray(LOGGER_PERMUTATION)]

    q_lines = [",".join(q_header)]
    for index in range(q_count):
        source = min(index, expected.count - 1)
        row_index = (q_index_override or {}).get(index, index)
        monotonic = (q_monotonic_override or {}).get(index, 2000.0 + index * 20.0)
        prefix = (
            str(row_index),
            str(index * 20.0),
            str(1000.0 + index * 20.0),
            str(monotonic),
            "0.0",
        )
        values = [format(float(value), ".9g") for value in raw_joint[source]]
        for (row, column), value in (q_value_override or {}).items():
            if row == index:
                values[column] = value
        q_lines.append(",".join((*prefix, *values)))
    (root / "q.csv").write_text("\n".join(q_lines) + "\n", encoding="ascii")

    base_lines = [",".join(base_header)]
    for index in range(base_count):
        source = min(index, expected.count - 1)
        prefix = (base_prefix_override or {}).get(
            index,
            (
                str(index),
                str(index * 20.0),
                str(1000.0 + index * 20.0),
                str(2000.0 + index * 20.0),
                "0.0",
            ),
        )
        values = [
            format(float(value), ".9g")
            for value in expected.body_quat_w[source]
        ]
        for (row, column), value in (base_value_override or {}).items():
            if row == index:
                values[column] = value
        base_lines.append(",".join((*prefix, *values)))
    (root / "base_quat.csv").write_text(
        "\n".join(base_lines) + "\n", encoding="ascii"
    )


class GearControlLogTests(TemporaryCase):
    def test_exact_pinned_headers_inverse_permutation_and_gear_metrics(self) -> None:
        expected = canonical(3)
        logs = self.root / "good"
        write_gear_logs(logs, expected)
        parsed = cli_module._parse_gear_control_logs(logs, expected.count)
        self.assertEqual(tuple(cli_module._GEAR_Q_HEADER), Q_HEADER)
        self.assertEqual(tuple(cli_module._GEAR_BASE_QUAT_HEADER), BASE_HEADER)
        self.assertEqual(tuple(cli_module._GEAR_LOGGER_TO_TARGET), LOGGER_INVERSE)
        np.testing.assert_array_equal(
            parsed.joint_position_target_order, expected.joint_position
        )
        np.testing.assert_array_equal(parsed.base_quat_w, expected.body_quat_w)
        metrics = cli_module._tracking_from_gear_control_rows(expected, parsed)
        self.assertEqual(metrics["frame_count"], 3)
        self.assertEqual(metrics["joint_position_rmse_rad"], 0.0)
        self.assertEqual(metrics["pelvis_orientation_rms_rad"], 0.0)
        self.assertEqual(metrics["pairing"], "same-control-tick-positional")
        self.assertEqual(metrics["tick_period_ms"]["count"], 2)

    def test_every_q_base_cardinality_prefix_header_and_value_mismatch_fails(
        self,
    ) -> None:
        expected = canonical(3)
        cases = {
            "q-header": {"q_header": Q_HEADER[:-1] + ("wrong",)},
            "base-header": {
                "base_header": BASE_HEADER[:-1] + ("wrong",)
            },
            "q-cardinality": {"q_rows": 2},
            "base-cardinality": {"base_rows": 2},
            "q-index": {"q_index_override": {1: 0}},
            "monotonic": {"q_monotonic_override": {1: 2000.0}},
            "prefix": {
                "base_prefix_override": {
                    1: ("1", "20.0", "1020.0", "2021.0", "0.0")
                }
            },
            "q-nan": {"q_value_override": {(1, 0): "nan"}},
            "base-nan": {"base_value_override": {(1, 0): "nan"}},
            "quaternion-norm": {"base_value_override": {(1, 0): "2.0"}},
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label):
                logs = self.root / label
                write_gear_logs(logs, expected, **kwargs)
                with self.assertRaises(ContractError):
                    cli_module._parse_gear_control_logs(logs, expected.count)


class TimeoutEvidenceTests(TemporaryCase):
    def test_timeout_partial_bytes_are_preserved_for_declared_evidence(self) -> None:
        timeout = subprocess.TimeoutExpired(
            ("synthetic",),
            1.0,
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )
        stdout, stderr = cli_module._timeout_evidence_bytes(timeout)
        self.assertEqual(stdout, b"partial stdout\n")
        self.assertEqual(stderr, b"partial stderr\n")


if __name__ == "__main__":
    unittest.main()
