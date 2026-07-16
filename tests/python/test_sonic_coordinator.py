from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import sys
from unittest import mock

import numpy as np

from mm_sonic.coordinator import (
    LOGGER_JOINT_PERMUTATION,
    AcceptedChunk,
    CommandSample,
    Coordinator,
    CoordinatorState,
    DeliveryAudit,
    DeliveryAuditEvidence,
    IntegrationFailure,
    SessionConfig,
    StateTransitionError,
    expected_official_target_row,
    parse_official_target_row,
)
from mm_sonic.joints import ContractError
from mm_sonic.process import (
    AdvanceResult,
    ChildProcessDied,
    MMChunkClient,
    ProcessError,
    ProcessProtocolError,
)
from mm_sonic.timeline import CanonicalTargetBuffer


_PERMUTATION = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10,
    14, 18, 2, 5, 8, 11, 15, 19, 21, 23,
    25, 27, 12, 16, 20, 22, 24, 26, 28,
)
_TIMING_KEYS = (
    "mm_generation",
    "projection_validation",
    "resampling",
    "artifact_enqueue",
    "publication",
    "simulation_advance",
)


def make_buffer(first: int, count: int) -> CanonicalTargetBuffer:
    position = (
        np.arange(first * 29, (first + count) * 29, dtype=np.float32)
        .reshape(count, 29)
        / np.float32(32.0)
    )
    velocity = -position.copy(order="C")
    quaternion = np.zeros((count, 4), dtype=np.float32)
    quaternion[:, 0] = 1.0
    return CanonicalTargetBuffer(
        joint_position=position,
        joint_velocity=velocity,
        body_quat_w=quaternion,
        frame_index=np.arange(first, first + count, dtype=np.int64),
    )


def official_row_bytes(buffer: CanonicalTargetBuffer, *, mismatch: bool = False) -> bytes:
    values = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            buffer.body_quat_w[0],
            buffer.joint_position[0, np.asarray(_PERMUTATION)],
        )
    ).astype(np.float32)
    if mismatch:
        values[0] = np.float32(1.0)
    # The pinned logger promotes streamed f32 values to double and uses
    # ostream's defaultfloat precision (six significant digits).
    encoded = ",".join(format(float(value), ".6g") for value in values)
    return (encoded + ",\n").encode("ascii")


class InjectedFailure(RuntimeError):
    def __init__(self, site: str):
        super().__init__(f"injected failure at {site}")
        self.failure_site = site


class FakeClock:
    def __init__(self) -> None:
        self.monotonic = 10_000
        self.wall = 1_000_000

    def monotonic_ns(self) -> int:
        self.monotonic += 17
        return self.monotonic

    def wall_time_ns(self) -> int:
        self.wall += 101
        return self.wall


class FakeMM:
    def __init__(self, events: list[str], fail_site: str | None = None) -> None:
        self.events = events
        self.fail_site = fail_site
        self.alive = True
        self.outstanding_candidate_id: str | None = None
        self.active_candidate_id: str | None = None
        self.generated_commands: list[CommandSample] = []
        self.generated_ids: list[tuple[str, str, str | None]] = []
        self.on_generate = None
        self.closed = False

    def require_alive(self) -> None:
        if not self.alive:
            raise ChildProcessDied("MM child is dead")

    def reset(self, config: SessionConfig, *, session_id: str) -> object:
        self.require_alive()
        self.events.append("mm.reset")
        return {"session_id": session_id, "initial_boundary": object()}

    def generate(
        self,
        command: CommandSample,
        *,
        session_id: str,
        candidate_id: str,
        predecessor_id: str | None,
        source_intervals: int,
    ) -> object:
        self.require_alive()
        self.events.append("mm.generate")
        self.generated_commands.append(command)
        self.generated_ids.append((session_id, candidate_id, predecessor_id))
        if source_intervals != 10:
            raise AssertionError("coordinator changed the source interval contract")
        if self.on_generate is not None:
            self.on_generate()
        if self.fail_site == "generation":
            # The request reached the live server and its response was lost.
            # The coordinator-owned ID remains abortable even without a
            # returned SourceChunk.
            self.outstanding_candidate_id = candidate_id
            raise InjectedFailure("generation")
        if self.fail_site == "generation_not_sent":
            # The client proves the request never crossed the pipe boundary.
            raise InjectedFailure("generation")
        self.outstanding_candidate_id = candidate_id
        return {
            "session_id": session_id,
            "candidate_id": candidate_id,
            "predecessor_id": predecessor_id,
            "command": command,
        }

    def commit(self, candidate_id: str) -> None:
        self.events.append("mm.commit")
        if candidate_id != self.outstanding_candidate_id:
            raise AssertionError("wrong candidate committed")
        # A commit response failure is ambiguous: model the server having
        # committed before the response was lost.  The coordinator must never
        # attempt a rollback at or beyond this boundary.
        self.active_candidate_id = candidate_id
        self.outstanding_candidate_id = None
        if self.fail_site == "mm_commit":
            raise InjectedFailure("mm_commit")

    def abort(self, candidate_id: str) -> None:
        self.events.append("mm.abort")
        if not self.alive:
            raise AssertionError("abort was sent to a dead MM server")
        if candidate_id != self.outstanding_candidate_id:
            raise AssertionError("wrong candidate aborted")
        self.outstanding_candidate_id = None

    def close(self) -> None:
        self.events.append("mm.close")
        self.closed = True


class FakeValidator:
    def __init__(self, events: list[str], fail_site: str | None = None) -> None:
        self.events = events
        self.fail_site = fail_site

    def validate_initial(self, raw: object) -> object:
        self.events.append("source.validate_initial")
        return SimpleNamespace(session_id=raw["session_id"])

    def validate_source(self, raw: object) -> object:
        self.events.append("source.validate")
        if self.fail_site == "source_validation":
            raise InjectedFailure("source_validation")
        return SimpleNamespace(**raw)


class FakeTimeline:
    def __init__(self, events: list[str], fail_site: str | None = None) -> None:
        self.events = events
        self.fail_site = fail_site
        self.initial_buffer = make_buffer(0, 1)
        self.canonical_buffer = self.initial_buffer
        self.pending_candidate_id: str | None = None
        self.last_accepted_candidate_id: str | None = None
        self.committed_ids: list[str] = []

    def prepare(self, source: object) -> object:
        self.events.append("target.prepare")
        if self.fail_site == "resampling":
            raise InjectedFailure("resampling")
        if self.pending_candidate_id is not None:
            raise AssertionError("timeline already has a pending candidate")
        first = int(self.canonical_buffer.frame_index[-1]) + 1
        target = SimpleNamespace(
            source_candidate_id=source.candidate_id,
            accepted_chunk_id=f"target-{first:06d}-{first + 19:06d}",
            buffer=make_buffer(first, 20),
        )
        prepared = SimpleNamespace(
            source_candidate_id=source.candidate_id,
            target=target,
        )
        self.pending_candidate_id = source.candidate_id
        return prepared

    def commit(self, prepared: object) -> object:
        self.events.append("timeline.commit")
        if self.fail_site == "timeline_commit":
            raise InjectedFailure("timeline_commit")
        if prepared.source_candidate_id != self.pending_candidate_id:
            raise AssertionError("wrong target candidate committed")
        target = prepared.target
        current = self.canonical_buffer
        self.canonical_buffer = CanonicalTargetBuffer(
            joint_position=np.concatenate(
                (current.joint_position, target.buffer.joint_position), axis=0
            ),
            joint_velocity=np.concatenate(
                (current.joint_velocity, target.buffer.joint_velocity), axis=0
            ),
            body_quat_w=np.concatenate(
                (current.body_quat_w, target.buffer.body_quat_w), axis=0
            ),
            frame_index=np.concatenate(
                (current.frame_index, target.buffer.frame_index)
            ),
        )
        self.pending_candidate_id = None
        self.last_accepted_candidate_id = prepared.source_candidate_id
        self.committed_ids.append(prepared.source_candidate_id)
        return target

    def abort(self, candidate_id: str) -> None:
        self.events.append("timeline.abort")
        if candidate_id != self.pending_candidate_id:
            raise AssertionError("wrong target candidate aborted")
        self.pending_candidate_id = None


class FakeRun:
    def __init__(self, events: list[str], fail_site: str | None = None) -> None:
        self.events = events
        self.fail_site = fail_site
        self.prepared: list[tuple[object, object]] = []
        self.rejection_attempts: list[object] = []
        self.rejections: list[object] = []
        self.abort_decisions: list[object] = []
        self.accepted: list[AcceptedChunk] = []
        self.timings: list[object] = []
        self.readiness: list[object] = []
        self.terminal_attempts: list[object] = []
        self.terminal = None

    def write_readiness(self, readiness: object) -> None:
        self.events.append("artifact.readiness")
        self.readiness.append(readiness)

    def write_prepared(self, source: object, target: object) -> None:
        self.events.append("artifact.enqueue")
        if self.fail_site == "artifact_enqueue":
            raise InjectedFailure("artifact_enqueue")
        self.prepared.append((source, target))

    def write_rejection(self, rejection: object) -> None:
        self.events.append("artifact.rejection")
        self.rejection_attempts.append(rejection)
        if self.fail_site == "artifact_rejection":
            raise InjectedFailure("artifact_rejection")
        self.rejections.append(rejection)

    def write_abort_decision(self, decision: object) -> None:
        self.events.append("artifact.abort")
        self.abort_decisions.append(decision)

    def write_accepted(self, accepted: AcceptedChunk) -> None:
        self.events.append("artifact.accept")
        if self.fail_site == "artifact_accept":
            raise InjectedFailure("artifact_accept")
        self.accepted.append(accepted)

    def write_timing(self, timing: object) -> None:
        self.events.append("artifact.timing")
        if self.fail_site == "artifact_timing":
            raise InjectedFailure("artifact_timing")
        self.timings.append(timing)

    def write_terminal_verdict(self, verdict: object) -> None:
        self.events.append("artifact.terminal")
        self.terminal_attempts.append(verdict)
        if self.fail_site == "artifact_terminal":
            raise InjectedFailure("artifact_terminal")
        if self.terminal is not None:
            raise AssertionError("terminal verdict was overwritten")
        self.terminal = verdict


class TestOnlyDeliveryAuditor:
    """Visible test-only capability; production has no permissive default."""

    def __init__(self) -> None:
        self.calls: list[DeliveryAuditEvidence] = []
        self.expected_rows_override: int | None = None
        self.passed = True

    def audit(self, evidence: DeliveryAuditEvidence) -> DeliveryAudit:
        if not isinstance(evidence, DeliveryAuditEvidence):
            raise AssertionError("auditor did not receive immutable evidence")
        self.calls.append(evidence)
        expected = (
            evidence.canonical_buffer.count
            if self.expected_rows_override is None
            else self.expected_rows_override
        )
        digest = hashlib.sha256(
            evidence.session_id.encode("utf-8")
            + evidence.canonical_buffer.frame_index.tobytes()
            + evidence.official_log_slice
        ).hexdigest()
        return DeliveryAudit(
            expected_rows=expected,
            observed_rows=expected,
            transmitted_indices_exact=self.passed,
            official_rows_exact=self.passed,
            evidence_sha256=digest,
        )


class FakePublisher:
    def __init__(
        self,
        events: list[str],
        target_log: Path,
        *,
        fail_site: str | None = None,
        readiness_match_attempt: int = 1,
    ) -> None:
        self.events = events
        self.target_log = target_log
        self.fail_site = fail_site
        self.readiness_match_attempt = readiness_match_attempt
        self.prepared_messages: list[object] = []
        self.sent: list[object] = []
        self.closed = False
        self.physics_step_count = lambda: 0

    def prepare(
        self,
        buffer: CanonicalTargetBuffer,
        *,
        phase: str,
        attempt: int | None = None,
    ) -> object:
        if phase == "readiness":
            if self.physics_step_count() != 0:
                raise AssertionError("MuJoCo advanced during frame-zero readiness")
            self.events.append(f"zmq.readiness.encode.{attempt}")
        else:
            self.events.append("zmq.encode")
        if self.fail_site == "zmq_encoding" and phase == "timeline":
            raise InjectedFailure("zmq_encoding")
        if phase == "readiness":
            self.events.append(f"artifact.readiness.archive.{attempt}")
        else:
            self.events.append("artifact.transmission.archive")
        if self.fail_site == "artifact_enqueue" and phase == "timeline":
            raise InjectedFailure("artifact_enqueue")
        prepared = SimpleNamespace(
            buffer=buffer,
            phase=phase,
            attempt=attempt,
            exact_bytes=hashlib.sha256(
                buffer.joint_position.tobytes()
                + buffer.joint_velocity.tobytes()
                + buffer.body_quat_w.tobytes()
                + buffer.frame_index.tobytes()
            ).digest(),
        )
        self.prepared_messages.append(prepared)
        return prepared

    def send_prepared(self, prepared: object) -> object:
        if prepared.phase == "readiness":
            if self.physics_step_count() != 0:
                raise AssertionError("MuJoCo advanced during frame-zero readiness")
            self.events.append(f"zmq.readiness.send.{prepared.attempt}")
            row = official_row_bytes(
                prepared.buffer,
                mismatch=prepared.attempt < self.readiness_match_attempt,
            )
            with self.target_log.open("ab") as output:
                output.write(row)
        else:
            self.events.append("zmq.send")
            if self.fail_site == "zmq_send":
                raise InjectedFailure("zmq_send")
        self.sent.append(prepared)
        return {
            "local_send_completed": True,
            "phase": prepared.phase,
            "attempt": prepared.attempt,
            "first_frame_index": int(prepared.buffer.frame_index[0]),
            "last_frame_index": int(prepared.buffer.frame_index[-1]),
        }

    def close(self) -> None:
        self.events.append("publisher.close")
        self.closed = True


class FakeStream:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.events.append("stream.enter")
        self.started = True

    def require_alive(self) -> None:
        if not self.started:
            raise ChildProcessDied("stream is not alive")

    def close(self) -> None:
        self.events.append("stream.close")
        self.closed = True


class FakeGate:
    def __init__(self, events: list[str], fail_site: str | None = None) -> None:
        self.events = events
        self.fail_site = fail_site
        self.paused = False
        self.sim_steps = 0
        self.closed = False
        self.pause_failures_remaining = 0

    @property
    def is_paused(self) -> bool:
        return self.paused

    def require_paused(self) -> None:
        if not self.paused:
            raise RuntimeError("gate is not paused")

    def pause(self) -> None:
        self.events.append("pause")
        if self.pause_failures_remaining:
            self.pause_failures_remaining -= 1
            raise InjectedFailure("process_pause")
        self.paused = True

    def release_steps(self, steps: int) -> AdvanceResult:
        self.require_paused()
        self.paused = False
        self.events.append("process.resume")
        if self.fail_site == "process_resume":
            self.events.append("pause")
            self.paused = True
            raise InjectedFailure("process_resume")
        self.events.append("simulator.advance")
        if self.fail_site == "simulator_advance":
            self.sim_steps += 7
            self.events.append("pause")
            self.paused = True
            raise InjectedFailure("simulator_advance")
        self.sim_steps += steps
        self.events.append("pause")
        self.paused = True
        return AdvanceResult(
            steps=steps,
            sim_time_start_s=(self.sim_steps - steps) * 0.005,
            sim_time_end_s=self.sim_steps * 0.005,
            state_rows=20,
            contact_rows=steps,
        )

    def close(self) -> None:
        self.events.append("gate.close")
        self.closed = True


_DEFAULT_AUDITOR = object()


class Harness:
    def __init__(
        self,
        temporary: tempfile.TemporaryDirectory[str],
        *,
        fail_site: str | None = None,
        readiness_match_attempt: int = 1,
        delivery_auditor: object | None = _DEFAULT_AUDITOR,
    ) -> None:
        self.events: list[str] = []
        self.root = Path(temporary.name)
        self.target_log = self.root / "target.csv"
        self.mm = FakeMM(self.events, fail_site)
        self.validator = FakeValidator(self.events, fail_site)
        self.timeline = FakeTimeline(self.events, fail_site)
        self.run = FakeRun(self.events, fail_site)
        self.publisher = FakePublisher(
            self.events,
            self.target_log,
            fail_site=fail_site,
            readiness_match_attempt=readiness_match_attempt,
        )
        self.stream = FakeStream(self.events)
        self.gate = FakeGate(self.events, fail_site)
        self.publisher.physics_step_count = lambda: self.gate.sim_steps
        self.clock = FakeClock()
        self.auditor = (
            TestOnlyDeliveryAuditor()
            if delivery_auditor is _DEFAULT_AUDITOR
            else delivery_auditor
        )
        self.coordinator = Coordinator(
            mm=self.mm,
            validator=self.validator,
            timeline_factory=lambda _initial: self.timeline,
            run=self.run,
            publisher=self.publisher,
            stream=self.stream,
            gate=self.gate,
            target_motion_logfile=self.target_log,
            steps_per_chunk=80,
            source_intervals=10,
            session_id_factory=lambda: "session-000000",
            monotonic_ns=self.clock.monotonic_ns,
            wall_time_ns=self.clock.wall_time_ns,
            readiness_wait=lambda: None,
            delivery_auditor=self.auditor,
        )

    def preflight(self) -> None:
        self.coordinator.preflight(
            SessionConfig(
                scene_id="sonic-flat-baseline",
                route_id="flat-12s",
                terrain_weight=0.0,
            )
        )
        self.events.clear()

    @staticmethod
    def command(speed: float = 0.5, *, chunk_index: int = 0) -> CommandSample:
        return CommandSample(
            chunk_index=chunk_index,
            requested_velocity_mujoco=(speed, 0.0, 0.0),
            desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
        )


class CoordinatorContractTests(unittest.TestCase):
    def test_explicit_states_and_official_logger_permutation_are_pinned(self) -> None:
        self.assertEqual(
            tuple(state.value for state in CoordinatorState),
            (
                "created",
                "preflight",
                "ready_paused",
                "candidate",
                "committed",
                "advancing",
                "terminal",
            ),
        )
        self.assertEqual(LOGGER_JOINT_PERMUTATION, _PERMUTATION)

    def test_command_value_is_finite_normalized_and_immutable(self) -> None:
        command = Harness.command()
        with self.assertRaises(FrozenInstanceError):
            command.requested_velocity_mujoco = (0.0, 0.0, 0.0)
        for velocity, heading, expected in (
            ((0.0, 0.0), (1.0, 0.0, 0.0, 0.0), "velocity"),
            ((0.0, 0.0, float("nan")), (1.0, 0.0, 0.0, 0.0), "finite"),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), "unit"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    CommandSample(0, velocity, heading)

    def test_session_config_requires_real_protocol_route(self) -> None:
        with self.assertRaisesRegex(ContractError, "route_id"):
            SessionConfig("sonic-flat-baseline", None, 0.0)


class OfficialTargetLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = make_buffer(0, 1)

    def test_expected_row_uses_exact_defaultfloat_text_and_logger_permutation(self) -> None:
        expected = expected_official_target_row(self.buffer)
        manual = np.concatenate(
            (
                np.zeros(3, dtype=np.float32),
                self.buffer.body_quat_w[0],
                self.buffer.joint_position[0, np.asarray(_PERMUTATION)],
            )
        )
        manual_bytes = (
            ",".join(format(float(value), ".6g") for value in manual) + ",\n"
        ).encode("ascii")
        self.assertEqual(expected, manual_bytes)
        parsed = parse_official_target_row(official_row_bytes(self.buffer))
        self.assertIsInstance(parsed, bytes)
        self.assertEqual(parsed, expected)

        perturbed_position = self.buffer.joint_position.copy()
        perturbed_position[0, 0] += np.float32(1.0e-4)
        perturbed = CanonicalTargetBuffer(
            joint_position=perturbed_position,
            joint_velocity=self.buffer.joint_velocity,
            body_quat_w=self.buffer.body_quat_w,
            frame_index=self.buffer.frame_index,
        )
        self.assertNotEqual(expected_official_target_row(perturbed), expected)

    def test_parser_rejects_malformed_nonfinite_partial_and_extra_rows(self) -> None:
        valid = official_row_bytes(self.buffer)
        fields = valid[:-2].split(b",")
        cases = (
            (valid[:-1], "newline"),
            (valid[:-2] + b"\n", "trailing comma"),
            (b",".join(fields[:-1]) + b",\n", "36"),
            (b",".join((*fields, b"1")) + b",\n", "36"),
            (valid.replace(fields[5], b"nan", 1), "finite"),
            (valid.replace(fields[5], b"inf", 1), "finite"),
            (valid.replace(fields[5], b"0.000000", 1), "defaultfloat"),
            (b"\xff" + valid[1:], "ASCII"),
            (valid + valid, "one row"),
        )
        for encoded, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    parse_official_target_row(encoded)

    def test_defaultfloat_contract_matches_a_live_cpp_stream_oracle(self) -> None:
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a C++ compiler is required for the logger-format oracle")
        # These finite binary32 bit patterns exercise signed zero, rounding
        # boundaries, fixed/scientific switching, subnormals, and both signs.
        patterns = (
            0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
            0x3DCCCCCD, 0x3F000000, 0x40490FDB, 0xC0490FDB,
            0x358637BD, 0x358637BC, 0x3727C5AC, 0x4B189680,
            0x49742400, 0x447A0000, 0x3A83126F, 0x3A83126E,
            0x3A831270, 0x00800000, 0x00000001, 0x007FFFFF,
            0x7F7FFFFF, 0xFF7FFFFF, 0x3F7FFFFF, 0x3F800001,
            0x41200000, 0x42C80000, 0x461C4000, 0x47C35000,
            0x4A742400, 0x2EDBE6FF, 0xAEDBE6FF, 0x3EAAAAAB,
            0xBEAAAAAB, 0x3F9E0652, 0x3F9E064B, 0x7F000000,
        )
        source = """
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
  constexpr std::array<std::uint32_t, 36> bits{{BITS}};
  for (const auto encoded : bits) {
    float value = 0.0F;
    std::memcpy(&value, &encoded, sizeof(value));
    std::cout << value << ',';
  }
  std::cout << '\\n';
}
""".replace(
            "BITS", ",".join(f"0x{value:08x}U" for value in patterns)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "defaultfloat_oracle.cpp"
            executable = root / "defaultfloat_oracle"
            source_path.write_text(source, encoding="ascii")
            subprocess.run(
                [compiler, "-std=c++17", str(source_path), "-o", str(executable)],
                check=True,
                capture_output=True,
            )
            completed = subprocess.run(
                [str(executable)], check=True, capture_output=True
            )
        self.assertEqual(parse_official_target_row(completed.stdout), completed.stdout)
        self.assertEqual(completed.stderr, b"")


class CoordinatorReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_frame_zero_repeats_until_exact_official_row_then_records_byte_offset(self) -> None:
        harness = Harness(self.temporary, readiness_match_attempt=2)
        result = harness.coordinator.preflight(
            SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
        )

        mismatch = official_row_bytes(harness.timeline.initial_buffer, mismatch=True)
        matching = official_row_bytes(harness.timeline.initial_buffer)
        self.assertEqual(harness.target_log.read_bytes(), mismatch + matching)
        self.assertEqual(result.scoring_log_offset, len(mismatch) + len(matching))
        self.assertEqual(result.readiness_attempts, 2)
        self.assertEqual(result.session_id, "session-000000")
        self.assertEqual(harness.coordinator.state, CoordinatorState.READY_PAUSED)
        self.assertTrue(harness.gate.is_paused)
        self.assertEqual(harness.gate.sim_steps, 0)
        self.assertEqual(
            harness.events,
            [
                "mm.reset",
                "source.validate_initial",
                "stream.enter",
                "zmq.readiness.encode.1",
                "artifact.readiness.archive.1",
                "zmq.readiness.send.1",
                "zmq.readiness.encode.2",
                "artifact.readiness.archive.2",
                "zmq.readiness.send.2",
                "pause",
                "artifact.readiness",
            ],
        )
        self.assertEqual(
            [message.attempt for message in harness.publisher.sent], [1, 2]
        )
        self.assertEqual(harness.run.readiness, [result])
        self.assertEqual(result.expected_row_sha256, hashlib.sha256(matching).hexdigest())
        self.assertGreater(result.log_inode, 0)
        self.assertGreater(result.log_device, 0)

    def test_partial_last_log_row_is_not_accepted_as_readiness(self) -> None:
        harness = Harness(self.temporary, readiness_match_attempt=2)
        original = harness.publisher.send_prepared

        def partial_then_matching(prepared: object) -> object:
            if prepared.phase == "readiness" and prepared.attempt == 1:
                harness.events.append("zmq.readiness.send.1")
                with harness.target_log.open("ab") as output:
                    output.write(official_row_bytes(prepared.buffer)[:-1])
                harness.publisher.sent.append(prepared)
                return {"local_send_completed": True}
            if prepared.phase == "readiness" and prepared.attempt == 2:
                with harness.target_log.open("ab") as output:
                    output.write(b"\n")
            return original(prepared)

        harness.publisher.send_prepared = partial_then_matching
        result = harness.coordinator.preflight(
            SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
        )
        matching = official_row_bytes(harness.timeline.initial_buffer)
        self.assertEqual(result.readiness_attempts, 2)
        self.assertEqual(harness.target_log.read_bytes(), matching + matching)
        # Attempt two completed the partial first row and then appended its own
        # row before returning.  Pausing GEAR and snapshotting EOF must exclude
        # both handshake rows from subsequent scoring.
        self.assertEqual(result.scoring_log_offset, 2 * len(matching))

    def test_preexisting_matching_rows_cannot_establish_current_readiness(self) -> None:
        harness = Harness(self.temporary, readiness_match_attempt=2)
        stale = official_row_bytes(harness.timeline.initial_buffer)
        harness.target_log.write_bytes(stale)

        result = harness.coordinator.preflight(
            SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
        )

        mismatch = official_row_bytes(harness.timeline.initial_buffer, mismatch=True)
        matching = official_row_bytes(harness.timeline.initial_buffer)
        self.assertEqual(harness.target_log.read_bytes(), stale + mismatch + matching)
        self.assertEqual(result.readiness_attempts, 2)
        self.assertEqual(result.readiness_log_start_offset, len(stale))
        self.assertEqual(
            result.scoring_log_offset, len(stale) + len(mismatch) + len(matching)
        )
        self.assertEqual(harness.gate.sim_steps, 0)

    def test_reset_initial_boundary_must_match_coordinator_session(self) -> None:
        harness = Harness(self.temporary)

        def mismatched_reset(_config, *, session_id):
            harness.events.append("mm.reset")
            return {
                "session_id": session_id,
                "initial_boundary": object(),
            }

        harness.mm.reset = mismatched_reset
        original = harness.validator.validate_initial

        def mismatched_initial(raw):
            original(raw)
            return SimpleNamespace(session_id="different-session")

        harness.validator.validate_initial = mismatched_initial
        with self.assertRaisesRegex(IntegrationFailure, "initial_session"):
            harness.coordinator.preflight(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
            )
        self.assertNotIn("stream.enter", harness.events)
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        self.assertTrue(harness.gate.is_paused)

    def test_preflight_id_and_initial_log_inspection_failures_terminalize(self) -> None:
        invalid_id = Harness(self.temporary)
        invalid_id.coordinator._session_id_factory = lambda: ""
        with self.assertRaisesRegex(IntegrationFailure, "session ID"):
            invalid_id.coordinator.preflight(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
            )
        self.assertEqual(invalid_id.coordinator.state, CoordinatorState.TERMINAL)
        self.assertIsNotNone(invalid_id.run.terminal)

        second_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(second_temporary.cleanup)
        invalid_log = Harness(second_temporary)
        invalid_log.target_log.mkdir()
        with self.assertRaisesRegex(IntegrationFailure, "regular"):
            invalid_log.coordinator.preflight(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
            )
        self.assertEqual(invalid_log.coordinator.state, CoordinatorState.TERMINAL)
        self.assertIsNotNone(invalid_log.run.terminal)

    def test_readiness_rejects_hardlink_and_path_replacement(self) -> None:
        hardlink_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(hardlink_temporary.cleanup)
        hardlink = Harness(hardlink_temporary)
        external = hardlink.root / "external.csv"
        external.write_bytes(b"")
        hardlink.target_log.hardlink_to(external)
        with self.assertRaisesRegex(IntegrationFailure, "hard link"):
            hardlink.coordinator.preflight(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
            )
        self.assertEqual(hardlink.coordinator.state, CoordinatorState.TERMINAL)

        replacement_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(replacement_temporary.cleanup)
        replacement = Harness(replacement_temporary)
        replacement.target_log.write_bytes(b"")
        original_send = replacement.publisher.send_prepared

        def replace_after_send(prepared):
            result = original_send(prepared)
            moved = replacement.root / "original-target.csv"
            replacement.target_log.replace(moved)
            replacement.target_log.write_bytes(
                official_row_bytes(prepared.buffer)
            )
            return result

        replacement.publisher.send_prepared = replace_after_send
        with self.assertRaisesRegex(IntegrationFailure, "identity"):
            replacement.coordinator.preflight(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
            )
        self.assertEqual(
            replacement.coordinator.state, CoordinatorState.TERMINAL
        )


class CoordinatorSuccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.harness = Harness(self.temporary)
        self.harness.preflight()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_success_order_ids_timing_and_mutation_are_exact(self) -> None:
        self.harness.coordinator.queue_command(Harness.command())
        accepted = self.harness.coordinator.run_one_chunk()

        self.assertEqual(
            self.harness.events,
            [
                "pause",
                "mm.generate",
                "source.validate",
                "target.prepare",
                "artifact.enqueue",
                "zmq.encode",
                "artifact.transmission.archive",
                "zmq.send",
                "mm.commit",
                "timeline.commit",
                "process.resume",
                "simulator.advance",
                "pause",
                "artifact.accept",
                "artifact.timing",
            ],
        )
        self.assertIsInstance(accepted, AcceptedChunk)
        self.assertEqual(accepted.target.buffer.frame_index.tolist(), list(range(1, 21)))
        self.assertEqual(accepted.advance.steps, 80)
        self.assertEqual(tuple(accepted.timings_ns), _TIMING_KEYS)
        self.assertTrue(
            all(type(value) is int and value >= 0 for value in accepted.timings_ns.values())
        )
        self.assertIsInstance(accepted.wall_started_ns, int)
        self.assertIsInstance(accepted.wall_finished_ns, int)
        self.assertGreaterEqual(accepted.wall_finished_ns, accepted.wall_started_ns)
        self.assertEqual(
            self.harness.mm.generated_ids,
            [("session-000000", "session-000000:candidate:000000", None)],
        )
        self.assertEqual(
            self.harness.mm.active_candidate_id,
            "session-000000:candidate:000000",
        )
        self.assertEqual(self.harness.timeline.committed_ids, [self.harness.mm.active_candidate_id])
        self.assertEqual(self.harness.timeline.canonical_buffer.count, 21)
        self.assertEqual(self.harness.gate.sim_steps, 80)
        self.assertTrue(self.harness.gate.is_paused)
        self.assertEqual(self.harness.coordinator.state, CoordinatorState.READY_PAUSED)
        self.assertEqual(len(self.harness.run.accepted), 1)
        self.assertEqual(len(self.harness.run.timings), 1)
        self.assertNotIn("ack", repr(self.harness.run.accepted).lower())

    def test_mid_generation_input_is_queued_without_mutating_the_latched_candidate(self) -> None:
        first = Harness.command(0.25)
        second = Harness.command(0.75, chunk_index=1)
        self.harness.coordinator.queue_command(first)
        observed_state: list[CoordinatorState] = []

        def queue_during_generation() -> None:
            observed_state.append(self.harness.coordinator.state)
            self.harness.coordinator.queue_command(second)

        self.harness.mm.on_generate = queue_during_generation
        self.harness.coordinator.run_one_chunk()
        self.harness.mm.on_generate = None
        self.harness.coordinator.run_one_chunk()

        self.assertEqual(observed_state, [CoordinatorState.CANDIDATE])
        self.assertEqual(self.harness.mm.generated_commands, [first, second])
        self.assertEqual(
            self.harness.mm.generated_ids,
            [
                ("session-000000", "session-000000:candidate:000000", None),
                (
                    "session-000000",
                    "session-000000:candidate:000001",
                    "session-000000:candidate:000000",
                ),
            ],
        )
        self.assertEqual(self.harness.timeline.canonical_buffer.count, 41)

    def test_command_chunk_index_must_equal_the_next_acceptance(self) -> None:
        for wrong in (1, 2):
            with self.subTest(initial_index=wrong):
                with self.assertRaisesRegex(ContractError, "chunk_index 0"):
                    self.harness.coordinator.run_one_chunk(
                        Harness.command(chunk_index=wrong)
                    )
                self.assertEqual(
                    self.harness.coordinator.state,
                    CoordinatorState.READY_PAUSED,
                )
                self.assertEqual(self.harness.mm.generated_commands, [])

        self.harness.coordinator.run_one_chunk(Harness.command(chunk_index=0))
        generated = list(self.harness.mm.generated_commands)
        for wrong in (0, 2):
            with self.subTest(successor_index=wrong):
                with self.assertRaisesRegex(ContractError, "chunk_index 1"):
                    self.harness.coordinator.run_one_chunk(
                        Harness.command(chunk_index=wrong)
                    )
                self.assertEqual(
                    self.harness.coordinator.state,
                    CoordinatorState.READY_PAUSED,
                )
                self.assertEqual(self.harness.mm.generated_commands, generated)

    def test_slow_generation_has_no_scientific_timeout(self) -> None:
        self.harness.coordinator.queue_command(Harness.command())

        def slow() -> None:
            time.sleep(0.05)

        self.harness.mm.on_generate = slow
        accepted = self.harness.coordinator.run_one_chunk()
        self.assertEqual(accepted.advance.steps, 80)
        self.assertEqual(self.harness.coordinator.state, CoordinatorState.READY_PAUSED)


class CoordinatorFailureBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporaries: list[tempfile.TemporaryDirectory[str]] = []

    def tearDown(self) -> None:
        for temporary in self.temporaries:
            temporary.cleanup()

    def harness(self, site: str) -> Harness:
        temporary = tempfile.TemporaryDirectory()
        self.temporaries.append(temporary)
        harness = Harness(temporary, fail_site=site)
        # Readiness is not the failure under test.
        harness.publisher.fail_site = None
        harness.preflight()
        harness.publisher.fail_site = site
        harness.coordinator.queue_command(Harness.command())
        return harness

    def test_every_named_boundary_has_the_registered_abort_and_mutation_decision(self) -> None:
        cases = {
            "generation": (True, False, False, 0),
            "source_validation": (True, False, False, 0),
            "resampling": (True, False, False, 0),
            "artifact_enqueue": (True, False, False, 0),
            "zmq_encoding": (True, False, False, 0),
            "zmq_send": (True, False, False, 0),
            # The commit request is the irreversible/ambiguous boundary.
            "mm_commit": (False, True, False, 0),
            "timeline_commit": (False, True, False, 0),
            "process_resume": (False, True, True, 0),
            "simulator_advance": (False, True, True, 7),
        }
        for site, (abort_sent, mm_changed, timeline_changed, sim_steps) in cases.items():
            with self.subTest(site=site):
                harness = self.harness(site)
                before_timeline = harness.timeline.canonical_buffer.frame_index.tobytes()
                with self.assertRaises(IntegrationFailure) as raised:
                    harness.coordinator.run_one_chunk()

                self.assertEqual(raised.exception.site, site)
                self.assertEqual(
                    raised.exception.phase,
                    "pre_commit" if abort_sent or site == "generation" else "post_commit",
                )
                self.assertEqual("mm.abort" in harness.events, abort_sent)
                self.assertEqual(harness.mm.active_candidate_id is not None, mm_changed)
                self.assertEqual(bool(harness.timeline.committed_ids), timeline_changed)
                self.assertEqual(harness.gate.sim_steps, sim_steps)
                if not timeline_changed:
                    self.assertEqual(
                        harness.timeline.canonical_buffer.frame_index.tobytes(),
                        before_timeline,
                    )
                self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
                self.assertTrue(harness.gate.is_paused)
                self.assertIsNotNone(harness.run.terminal)
                self.assertTrue(harness.run.terminal.delivery_audit_required)
                self.assertFalse(harness.run.terminal.delivery_audit_completed)
                self.assertEqual(harness.run.terminal.failure_site, site)
                self.assertEqual(
                    len(harness.run.rejections),
                    1 if abort_sent or site == "generation" else 0,
                )
                if harness.run.rejections:
                    rejection = harness.run.rejections[0]
                    self.assertEqual(rejection.abort_required, abort_sent)
                    self.assertFalse(rejection.abort_sent)
                    self.assertEqual(rejection.mm_alive, True)
                    self.assertEqual(rejection.failure_site, site)
                    self.assertEqual(len(harness.run.abort_decisions), 1)
                    self.assertEqual(
                        harness.run.abort_decisions[0].mm_abort_sent, abort_sent
                    )
                self.assertEqual(len(harness.run.timings), 1)
                timing = harness.run.timings[0]
                self.assertEqual(tuple(timing.durations_ns), _TIMING_KEYS)
                self.assertEqual(timing.failed_stage, site)
                self.assertTrue(
                    all(
                        value is None or (type(value) is int and value >= 0)
                        for value in timing.durations_ns.values()
                    )
                )
                self.assertIsInstance(timing.wall_started_ns, int)
                self.assertIsInstance(timing.wall_finished_ns, int)
                self.assertGreaterEqual(
                    timing.wall_finished_ns, timing.wall_started_ns
                )
                event_count = len(harness.events)
                with self.assertRaisesRegex(StateTransitionError, "terminal"):
                    harness.coordinator.run_one_chunk()
                self.assertEqual(len(harness.events), event_count)

    def test_generation_request_proven_not_sent_does_not_emit_spurious_abort(self) -> None:
        harness = self.harness("generation_not_sent")
        harness.events.clear()
        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.run_one_chunk()
        self.assertEqual(raised.exception.site, "generation")
        self.assertEqual(raised.exception.phase, "pre_commit")
        self.assertNotIn("mm.abort", harness.events)
        self.assertEqual(len(harness.run.rejections), 1)
        self.assertFalse(harness.run.rejections[0].abort_required)
        self.assertFalse(harness.run.rejections[0].abort_sent)
        self.assertTrue(harness.run.rejections[0].mm_alive)
        self.assertIsNone(harness.mm.outstanding_candidate_id)
        self.assertEqual(harness.run.timings[0].failed_stage, "generation")

    def test_initial_pause_failure_terminalizes_and_records_unstarted_stages(self) -> None:
        harness = self.harness("source_validation")
        harness.validator.fail_site = None
        harness.events.clear()
        harness.gate.paused = False
        harness.gate.pause_failures_remaining = 1

        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.run_one_chunk()

        self.assertEqual(raised.exception.site, "process_pause")
        self.assertEqual(raised.exception.phase, "pre_commit")
        self.assertNotIn("mm.generate", harness.events)
        self.assertNotIn("mm.abort", harness.events)
        self.assertTrue(harness.gate.is_paused)
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        self.assertEqual(len(harness.run.rejection_attempts), 1)
        self.assertEqual(len(harness.run.timings), 1)
        self.assertTrue(
            all(value is None for value in harness.run.timings[0].durations_ns.values())
        )

    def test_rejection_append_failure_cannot_suppress_timeline_or_mm_abort(self) -> None:
        harness = self.harness("artifact_rejection")
        harness.validator.fail_site = "source_validation"
        harness.events.clear()

        with self.assertRaises(IntegrationFailure):
            harness.coordinator.run_one_chunk()

        self.assertIn("artifact.rejection", harness.events)
        self.assertIn("mm.abort", harness.events)
        self.assertIsNone(harness.mm.outstanding_candidate_id)
        self.assertEqual(harness.timeline.canonical_buffer.count, 1)
        self.assertTrue(harness.gate.is_paused)
        self.assertEqual(len(harness.run.abort_decisions), 1)
        self.assertTrue(harness.run.abort_decisions[0].mm_abort_sent)

    def test_failure_verdict_write_error_keeps_one_immutable_terminal_cause(self) -> None:
        harness = self.harness("artifact_terminal")
        harness.validator.fail_site = "source_validation"
        harness.events.clear()

        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.run_one_chunk()

        self.assertEqual(raised.exception.phase, "pre_commit")
        self.assertEqual(raised.exception.site, "artifact_terminal")
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        verdict = harness.coordinator.terminal_verdict
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.status, "failed")
        self.assertEqual(verdict.failure_phase, "pre_commit")
        self.assertEqual(verdict.failure_site, "source_validation")
        self.assertEqual(verdict.error_type, "InjectedFailure")
        self.assertEqual(len(harness.run.terminal_attempts), 1)
        self.assertIsNone(harness.run.terminal)
        harness.coordinator.close()
        self.assertIs(harness.coordinator.terminal_verdict, verdict)
        self.assertEqual(len(harness.run.terminal_attempts), 1)
        with self.assertRaisesRegex(StateTransitionError, "terminal"):
            harness.coordinator.run_one_chunk(Harness.command(0.75))

    def test_post_release_artifact_failures_keep_irreversible_accept_count(self) -> None:
        for site in ("artifact_accept", "artifact_timing"):
            with self.subTest(site=site):
                harness = self.harness(site)
                harness.events.clear()
                with self.assertRaises(IntegrationFailure) as raised:
                    harness.coordinator.run_one_chunk()
                self.assertEqual(raised.exception.site, site)
                self.assertEqual(raised.exception.phase, "post_commit")
                self.assertEqual(harness.mm.active_candidate_id is not None, True)
                self.assertEqual(harness.timeline.canonical_buffer.count, 21)
                self.assertEqual(harness.gate.sim_steps, 80)
                self.assertTrue(harness.gate.is_paused)
                self.assertEqual(harness.run.terminal.accepted_chunks, 1)
                self.assertNotIn("mm.abort", harness.events)
                self.assertIn("artifact.accept", harness.events)
                self.assertIn("artifact.timing", harness.events)

    def test_post_advance_assembly_failure_terminalizes_with_accept_count(self) -> None:
        harness = self.harness("source_validation")
        harness.validator.fail_site = None
        harness.events.clear()
        with mock.patch(
            "mm_sonic.coordinator.AcceptedChunk",
            side_effect=InjectedFailure("acceptance_assembly"),
        ):
            with self.assertRaises(IntegrationFailure) as raised:
                harness.coordinator.run_one_chunk()
        self.assertEqual(raised.exception.phase, "post_commit")
        self.assertEqual(raised.exception.site, "acceptance_assembly")
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        self.assertEqual(harness.run.terminal.accepted_chunks, 1)
        self.assertEqual(harness.gate.sim_steps, 80)
        self.assertTrue(harness.gate.is_paused)
        self.assertNotIn("mm.abort", harness.events)

    def test_wall_clock_failure_after_advance_cannot_leave_advancing_state(self) -> None:
        harness = self.harness("source_validation")
        harness.validator.fail_site = None
        harness.events.clear()
        calls = 0

        def failing_wall_clock():
            nonlocal calls
            calls += 1
            if calls == 1:
                return 100
            raise RuntimeError("wall clock unavailable")

        harness.coordinator._wall_time_ns = failing_wall_clock
        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.run_one_chunk()
        self.assertEqual(raised.exception.phase, "post_commit")
        self.assertEqual(raised.exception.site, "acceptance_assembly")
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        self.assertEqual(harness.run.terminal.accepted_chunks, 1)
        self.assertEqual(harness.gate.sim_steps, 80)
        self.assertTrue(harness.gate.is_paused)

    def test_initial_wall_clock_failure_cannot_leave_candidate_state(self) -> None:
        harness = self.harness("source_validation")
        harness.validator.fail_site = None
        harness.events.clear()

        def failing_wall_clock() -> int:
            raise RuntimeError("wall clock unavailable at chunk start")

        harness.coordinator._wall_time_ns = failing_wall_clock
        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.run_one_chunk()

        self.assertEqual(raised.exception.phase, "pre_commit")
        self.assertEqual(raised.exception.site, "timing_start")
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        self.assertEqual(harness.run.terminal.failure_site, "timing_start")
        self.assertTrue(harness.gate.is_paused)
        self.assertNotIn("mm.generate", harness.events)
        self.assertNotIn("mm.abort", harness.events)

    def test_wall_clock_can_move_backward_without_changing_duration_timing(self) -> None:
        harness = self.harness("source_validation")
        harness.validator.fail_site = None
        wall_values = iter((200, 100))
        harness.coordinator._wall_time_ns = lambda: next(wall_values)
        accepted = harness.coordinator.run_one_chunk()
        self.assertEqual(accepted.wall_started_ns, 200)
        self.assertEqual(accepted.wall_finished_ns, 100)
        self.assertEqual(harness.coordinator.state, CoordinatorState.READY_PAUSED)

    def test_dead_server_is_detected_without_abort_and_terminalizes_paused(self) -> None:
        harness = self.harness("source_validation")
        harness.validator.fail_site = None
        harness.mm.alive = False
        harness.events.clear()
        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.run_one_chunk()
        self.assertEqual(raised.exception.site, "generation")
        self.assertNotIn("mm.abort", harness.events)
        self.assertIsNone(harness.mm.active_candidate_id)
        self.assertEqual(harness.timeline.canonical_buffer.count, 1)
        self.assertTrue(harness.gate.is_paused)
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)


class CoordinatorTerminalAndCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.harness = Harness(self.temporary)
        self.harness.preflight()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_success_requires_delivery_audit_and_terminal_verdict_is_immutable(self) -> None:
        self.harness.coordinator.queue_command(Harness.command())
        self.harness.coordinator.run_one_chunk()
        assert isinstance(self.harness.auditor, TestOnlyDeliveryAuditor)
        verdict = self.harness.coordinator.finish()
        self.assertEqual(self.harness.coordinator.state, CoordinatorState.TERMINAL)
        self.assertEqual(verdict.status, "complete")
        self.assertTrue(verdict.delivery_audit_required)
        self.assertTrue(verdict.delivery_audit_completed)
        self.assertIs(self.harness.run.terminal, verdict)
        evidence = self.harness.auditor.calls[-1]
        self.assertEqual(evidence.session_id, "session-000000")
        self.assertEqual(evidence.canonical_buffer.count, 21)
        self.assertEqual(evidence.readiness, self.harness.run.readiness[0])
        self.assertEqual(evidence.official_log_slice, b"")
        self.assertEqual(len(evidence.timeline_publications), 1)
        self.assertIs(evidence.run, self.harness.run)
        with self.assertRaises(FrozenInstanceError):
            verdict.status = "failed"
        with self.assertRaisesRegex(StateTransitionError, "terminal"):
            self.harness.coordinator.finish()
        with self.assertRaisesRegex(StateTransitionError, "terminal"):
            self.harness.coordinator.queue_command(Harness.command())

    def test_invalid_auditor_results_fail_closed_and_cannot_be_retried(self) -> None:
        cases = ("wrong_type", "wrong_count")
        for case in cases:
            with self.subTest(case=case):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                if case == "wrong_type":
                    auditor = SimpleNamespace(audit=lambda _evidence: object())
                else:
                    auditor = TestOnlyDeliveryAuditor()
                    auditor.expected_rows_override = 1
                harness = Harness(temporary, delivery_auditor=auditor)
                harness.preflight()
                harness.coordinator.run_one_chunk(Harness.command())

                with self.assertRaises(IntegrationFailure) as raised:
                    harness.coordinator.finish()

                self.assertEqual(raised.exception.phase, "delivery_audit")
                self.assertEqual(raised.exception.site, "delivery_audit")
                self.assertEqual(
                    harness.coordinator.state, CoordinatorState.TERMINAL
                )
                self.assertEqual(harness.run.terminal.status, "failed")
                self.assertFalse(harness.run.terminal.delivery_audit_completed)
                with self.assertRaisesRegex(StateTransitionError, "terminal"):
                    harness.coordinator.finish()

    def test_finish_excludes_operator_input_while_audit_is_in_flight(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingTestOnlyAuditor(TestOnlyDeliveryAuditor):
            def audit(self, evidence: DeliveryAuditEvidence) -> DeliveryAudit:
                started.set()
                if not release.wait(2.0):
                    raise AssertionError("test did not release delivery audit")
                return super().audit(evidence)

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        harness = Harness(
            temporary, delivery_auditor=BlockingTestOnlyAuditor()
        )
        harness.preflight()
        harness.coordinator.run_one_chunk(Harness.command())
        finish_results: list[object] = []
        queue_results: list[object] = []
        queue_done = threading.Event()

        def finish() -> None:
            try:
                finish_results.append(harness.coordinator.finish())
            except BaseException as error:
                finish_results.append(error)

        def queue() -> None:
            try:
                harness.coordinator.queue_command(Harness.command(0.75))
            except BaseException as error:
                queue_results.append(error)
            finally:
                queue_done.set()

        finish_thread = threading.Thread(target=finish)
        finish_thread.start()
        self.assertTrue(started.wait(1.0))
        queue_thread = threading.Thread(target=queue)
        queue_thread.start()
        self.assertFalse(queue_done.wait(0.05))
        release.set()
        finish_thread.join(2.0)
        queue_thread.join(2.0)

        self.assertFalse(finish_thread.is_alive())
        self.assertFalse(queue_thread.is_alive())
        self.assertEqual(len(finish_results), 1)
        self.assertIsInstance(finish_results[0], object)
        self.assertEqual(getattr(finish_results[0], "status", None), "complete")
        self.assertEqual(len(queue_results), 1)
        self.assertIsInstance(queue_results[0], StateTransitionError)

    def test_no_auditor_means_production_completion_is_impossible(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        harness = Harness(temporary, delivery_auditor=None)
        harness.preflight()
        harness.coordinator.run_one_chunk(Harness.command())
        with self.assertRaisesRegex(ContractError, "delivery auditor"):
            harness.coordinator.finish()
        self.assertEqual(harness.coordinator.state, CoordinatorState.READY_PAUSED)
        self.assertIsNone(harness.run.terminal)
        forged = DeliveryAudit(21, 21, True, True, "a" * 64)
        with self.assertRaises(TypeError):
            harness.coordinator.finish(forged)

    def test_passing_audit_cannot_mark_unpaused_session_complete(self) -> None:
        self.harness.coordinator.run_one_chunk(Harness.command())
        self.harness.gate.paused = False
        self.harness.gate.pause_failures_remaining = 1
        verdict = self.harness.coordinator.finish()
        self.assertEqual(verdict.status, "failed")
        self.assertFalse(verdict.simulation_paused)
        self.assertEqual(verdict.failure_site, "delivery_audit")

    def test_terminal_verdict_write_failure_is_terminal_and_never_completes(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        harness = Harness(temporary, fail_site="artifact_terminal")
        harness.preflight()
        harness.coordinator.run_one_chunk(Harness.command())
        accepted_id = harness.mm.active_candidate_id

        with self.assertRaises(IntegrationFailure) as raised:
            harness.coordinator.finish()

        self.assertEqual(raised.exception.phase, "delivery_audit")
        self.assertEqual(raised.exception.site, "artifact_terminal")
        self.assertEqual(harness.coordinator.state, CoordinatorState.TERMINAL)
        verdict = harness.coordinator.terminal_verdict
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.status, "failed")
        self.assertEqual(verdict.failure_site, "artifact_terminal")
        self.assertTrue(verdict.delivery_audit_completed)
        self.assertIsNotNone(verdict.delivery_audit)
        self.assertIsNone(harness.run.terminal)
        self.assertEqual(len(harness.run.terminal_attempts), 1)
        with self.assertRaisesRegex(StateTransitionError, "terminal"):
            harness.coordinator.finish()
        with self.assertRaisesRegex(StateTransitionError, "terminal"):
            harness.coordinator.run_one_chunk(Harness.command(0.75))
        self.assertEqual(harness.mm.active_candidate_id, accepted_id)

    def test_cleanup_is_idempotent_and_preserves_pause_before_process_teardown(self) -> None:
        self.harness.events.clear()
        self.harness.coordinator.close()
        self.assertEqual(
            self.harness.events,
            [
                "artifact.terminal",
                "publisher.close",
                "gate.close",
                "stream.close",
                "mm.close",
            ],
        )
        self.assertTrue(self.harness.gate.closed)
        self.assertTrue(self.harness.publisher.closed)
        self.assertTrue(self.harness.stream.closed)
        self.assertTrue(self.harness.mm.closed)
        self.harness.coordinator.close()
        self.assertEqual(
            self.harness.events,
            [
                "artifact.terminal",
                "publisher.close",
                "gate.close",
                "stream.close",
                "mm.close",
            ],
        )

    def test_public_operations_reject_the_wrong_state(self) -> None:
        other_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(other_temporary.cleanup)
        created_harness = Harness(other_temporary)
        created = created_harness.coordinator
        with self.assertRaisesRegex(StateTransitionError, "created"):
            created.run_one_chunk()
        with self.assertRaisesRegex(StateTransitionError, "created"):
            created.run_one_chunk(Harness.command())
        created_harness.preflight()
        with self.assertRaisesRegex(StateTransitionError, "queued command"):
            created.run_one_chunk()
        with self.assertRaisesRegex(StateTransitionError, "ready_paused"):
            self.harness.coordinator.preflight(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0)
            )
        with self.assertRaisesRegex(StateTransitionError, "queued command"):
            self.harness.coordinator.run_one_chunk()


_MM_CHILD = r'''
import json
import sys
import time

active = None
outstanding = None
for line in sys.stdin:
    request = json.loads(line)
    op = request["op"]
    request_id = request["request_id"]
    if op == "hello":
        data = {"protocol_version": 1}
    elif op == "reset":
        active = None
        outstanding = None
        data = {
            "session_id": request["session_id"],
            "active_candidate_id": None,
            "scene": {},
            "initial_boundary": {},
        }
    elif op == "generate":
        outstanding = request["candidate_id"]
        if outstanding.endswith("ambiguous"):
            sys.stdout.write('{"v":1,"ok":true,"op":"generate","request_id":"' + request_id + '","data":{},"extra":1}\n')
            sys.stdout.flush()
            continue
        if outstanding.endswith("dead"):
            sys.stderr.write("dead during generation\n")
            sys.stderr.flush()
            raise SystemExit(23)
        time.sleep(0.05)
        data = request
    elif op == "commit":
        active = request["candidate_id"]
        outstanding = None
        data = {
            "session_id": request["session_id"],
            "candidate_id": request["candidate_id"],
            "active_candidate_id": active,
        }
    elif op == "abort":
        outstanding = None
        data = {
            "session_id": request["session_id"],
            "candidate_id": request["candidate_id"],
            "active_candidate_id": active,
        }
    elif op == "close":
        # The real mm_chunk_server v1 close envelope carries an empty data
        # object; process exit is the lifecycle confirmation.
        data = {}
    else:
        raise AssertionError(op)
    response = {
        "v": 1,
        "ok": True,
        "op": op,
        "request_id": request_id,
        "data": data,
    }
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    if op == "close":
        break
'''


class MMChunkClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def client(self) -> MMChunkClient:
        return MMChunkClient(
            run_root=self.root,
            command=(sys.executable, "-u", "-c", _MM_CHILD),
            stdout_archive=self.root / "mm.stdout.jsonl",
            stderr_archive=self.root / "mm.stderr.log",
            poll_interval_s=0.005,
            stop_grace_s=0.1,
            term_grace_s=0.1,
            kill_grace_s=0.1,
        )

    def test_reset_rejects_missing_route_before_protocol_write(self) -> None:
        client = self.client()
        try:
            client.hello()
            before = (self.root / "mm.stdout.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "route_id"):
                client.reset(
                    SessionConfig("sonic-flat-baseline", None, 0.0),
                    session_id="session-client",
                )
            self.assertEqual(
                (self.root / "mm.stdout.jsonl").read_bytes(),
                before,
            )
        finally:
            client.close()

    def test_invalid_command_is_rejected_before_generate_protocol_write(self) -> None:
        client = self.client()
        try:
            client.hello()
            client.reset(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="session-client",
            )
            before = (self.root / "mm.stdout.jsonl").read_bytes()
            invalid = SimpleNamespace(
                requested_velocity_mujoco=(0.0, 0.0),
                desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            with self.assertRaisesRegex(ValueError, "target-basis"):
                client.generate(
                    invalid,
                    session_id="session-client",
                    candidate_id="candidate-invalid-command",
                    predecessor_id=None,
                    source_intervals=10,
                )
            self.assertEqual(
                (self.root / "mm.stdout.jsonl").read_bytes(), before
            )
            self.assertIsNone(client.outstanding_candidate_id)
        finally:
            client.close()

    def test_slow_generate_has_no_deadline_and_client_owns_protocol_state(self) -> None:
        client = self.client()
        try:
            self.assertEqual(client.hello()["protocol_version"], 1)
            reset = client.reset(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="session-client",
            )
            self.assertEqual(reset["session_id"], "session-client")
            started = time.monotonic()
            generated = client.generate(
                CommandSample(
                    chunk_index=0,
                    requested_velocity_mujoco=(0.5, -0.25, 0.125),
                    desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
                ),
                session_id="session-client",
                candidate_id="candidate-slow",
                predecessor_id=None,
                source_intervals=10,
            )
            self.assertGreaterEqual(time.monotonic() - started, 0.04)
            self.assertEqual(
                generated["requested_velocity_holden"], [0.5, 0.125, 0.25]
            )
            self.assertEqual(
                generated["desired_heading_holden_wxyz"], [1.0, 0.0, 0.0, 0.0]
            )
            self.assertEqual(client.outstanding_candidate_id, "candidate-slow")
            client.commit("candidate-slow")
            self.assertIsNone(client.outstanding_candidate_id)
            self.assertEqual(client.active_candidate_id, "candidate-slow")
        finally:
            client.close()
        self.assertIn(b'"op":"generate"', (self.root / "mm.stdout.jsonl").read_bytes())

    def test_malformed_live_response_keeps_inflight_id_abortable(self) -> None:
        client = self.client()
        try:
            client.hello()
            client.reset(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="session-client",
            )
            with self.assertRaisesRegex(ProcessProtocolError, "keys differ"):
                client.generate(
                    Harness.command(),
                    session_id="session-client",
                    candidate_id="candidate-ambiguous",
                    predecessor_id=None,
                    source_intervals=10,
                )
            client.require_alive()
            self.assertEqual(
                client.outstanding_candidate_id, "candidate-ambiguous"
            )
            client.abort("candidate-ambiguous")
            self.assertIsNone(client.outstanding_candidate_id)
        finally:
            client.close()

    def test_dead_child_is_detected_and_cleanup_preserves_stderr(self) -> None:
        client = self.client()
        client.hello()
        client.reset(
            SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
            session_id="session-client",
        )
        with self.assertRaisesRegex(ChildProcessDied, "dead during generation"):
            client.generate(
                Harness.command(),
                session_id="session-client",
                candidate_id="candidate-dead",
                predecessor_id=None,
                source_intervals=10,
            )
        self.assertEqual(client.outstanding_candidate_id, "candidate-dead")
        client.close()
        self.assertIn(b"dead during generation", (self.root / "mm.stderr.log").read_bytes())

    def test_zero_byte_request_failure_is_proven_not_outstanding(self) -> None:
        client = self.client()
        try:
            client.hello()
            client.reset(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="session-client",
            )
            with mock.patch(
                "mm_sonic.process.os.write",
                side_effect=BrokenPipeError("injected before first byte"),
            ):
                with self.assertRaises(ProcessError):
                    client.generate(
                        Harness.command(),
                        session_id="session-client",
                        candidate_id="candidate-zero-byte",
                        predecessor_id=None,
                        source_intervals=10,
                    )
            self.assertIsNone(client.outstanding_candidate_id)
        finally:
            client.close()

    def test_partial_request_write_is_conservatively_abortable(self) -> None:
        client = self.client()
        try:
            client.hello()
            client.reset(
                SessionConfig("sonic-flat-baseline", "flat-12s", 0.0),
                session_id="session-client",
            )
            writes = 0

            def partial_then_fail(_fd, _data):
                nonlocal writes
                writes += 1
                if writes == 1:
                    return 1
                raise BrokenPipeError("injected after first byte")

            with mock.patch("mm_sonic.process.os.write", side_effect=partial_then_fail):
                with self.assertRaises(ProcessError):
                    client.generate(
                        Harness.command(),
                        session_id="session-client",
                        candidate_id="candidate-partial-byte",
                        predecessor_id=None,
                        source_intervals=10,
                    )
            self.assertEqual(
                client.outstanding_candidate_id, "candidate-partial-byte"
            )
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
