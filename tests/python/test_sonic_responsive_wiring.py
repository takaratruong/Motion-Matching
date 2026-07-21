from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.commands import CommandSample
from mm_sonic.coordinator import CandidateSuperseded


def _command(chunk_index: int = 0, speed: float = 0.5) -> CommandSample:
    return CommandSample(
        chunk_index=chunk_index,
        requested_velocity_mujoco=(speed, 0.0, 0.0),
        desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


class _FakeBuffer:
    """A minimal published buffer object; identity is what matters."""


class _FakeTarget:
    def __init__(self, root_rows: np.ndarray) -> None:
        self.virtual_root_position = root_rows
        self._buffer = _FakeBuffer()

    @property
    def buffer(self) -> _FakeBuffer:
        return self._buffer


class _FakePrepared:
    def __init__(self, root_rows: np.ndarray) -> None:
        self.target = _FakeTarget(root_rows)


class _FakeAdvance:
    def __init__(self, state_rows_start: int, state_rows_end: int) -> None:
        self.steps = state_rows_end - state_rows_start
        self.sim_time_start_s = 0.0
        self.sim_time_end_s = 0.4
        self.state_rows = state_rows_end
        self.contact_rows = 0


class _Fakes:
    """A single shared call-order log with duck-typed collaborators."""

    def __init__(self, *, root_rows: np.ndarray | None = None) -> None:
        self.log: list[str] = []
        if root_rows is None:
            root_rows = np.zeros((20, 3), dtype=np.float32)
        self._prepared = _FakePrepared(root_rows)
        self.state_rows_before = 0
        self.state_rows_after = 20

    # mm ----------------------------------------------------------------
    def generate(self, command, **kwargs):
        self.log.append("mm.generate")
        return object()

    def commit(self, candidate_id):
        self.log.append("mm.commit")

    def abort(self, candidate_id):
        self.log.append("mm.abort")

    @property
    def last_accepted_candidate_id(self):
        return None

    # validator ---------------------------------------------------------
    def validate_source(self, raw):
        self.log.append("validate")
        return object()

    # timeline ----------------------------------------------------------
    def prepare(self, checked):
        self.log.append("prepare")
        return self._prepared

    def commit_timeline(self, prepared):
        self.log.append("timeline.commit")

    def abort_timeline(self, candidate_id):
        self.log.append("timeline.abort")

    # publisher / gate / recorder --------------------------------------
    def publish(self, buffer, *, phase, wait):
        self.log.append(f"publish:{phase}:wait={wait}")
        self._published_buffer = buffer

    def release_steps(self, steps):
        self.log.append("release")
        self.state_rows_before = 0
        return _FakeAdvance(0, self.state_rows_after)

    def record(self, command):
        self.log.append("record")


class _MMShim:
    def __init__(self, fakes, *, generate_error=None):
        self._f = fakes
        self._generate_error = generate_error

    def generate(self, command, **kwargs):
        if self._generate_error is not None:
            self._f.log.append("mm.generate")
            raise self._generate_error
        return self._f.generate(command, **kwargs)

    def commit(self, candidate_id):
        self._f.commit(candidate_id)

    def abort(self, candidate_id):
        self._f.abort(candidate_id)


class _TimelineShim:
    def __init__(self, fakes):
        self._f = fakes

    @property
    def last_accepted_candidate_id(self):
        return None

    def prepare(self, checked):
        return self._f.prepare(checked)

    def commit(self, prepared):
        self._f.commit_timeline(prepared)

    def abort(self, candidate_id):
        self._f.abort_timeline(candidate_id)


class _ValidatorShim:
    def __init__(self, fakes):
        self._f = fakes

    def validate_source(self, raw):
        return self._f.validate_source(raw)


class _PublisherShim:
    def __init__(self, fakes, *, publish_error=None):
        self._f = fakes
        self._publish_error = publish_error

    def publish(self, buffer, *, phase, wait):
        if self._publish_error is not None:
            self._f.log.append(f"publish:{phase}:wait={wait}")
            raise self._publish_error
        self._f.publish(buffer, phase=phase, wait=wait)


class _GateShim:
    def __init__(self, fakes):
        self._f = fakes

    def release_steps(self, steps):
        return self._f.release_steps(steps)


class _RecorderShim:
    def __init__(self, fakes):
        self._f = fakes

    def record(self, command):
        self._f.record(command)


def _make_committer(fakes, *, mm=None):
    from mm_sonic.responsive_wiring import ManualChunkCommitter

    return ManualChunkCommitter(
        mm=mm if mm is not None else _MMShim(fakes),
        validator=_ValidatorShim(fakes),
        timeline=_TimelineShim(fakes),
        publish=_PublisherShim(fakes).publish,
        gate=_GateShim(fakes),
        session_id="session",
        steps_per_chunk=20,
        recorder=_RecorderShim(fakes),
    )


class ManualChunkCommitterOrderingTests(unittest.TestCase):
    def test_committer_commits_one_prefix_in_coordinator_order(self):
        fakes = _Fakes()
        committer = _make_committer(fakes)

        result = committer.run_one_chunk(_command(0))

        self.assertIsNotNone(result)
        self.assertEqual(
            fakes.log,
            [
                "mm.generate",
                "validate",
                "prepare",
                "publish:timeline:wait=False",
                "mm.commit",
                "timeline.commit",
                "release",
                "record",
            ],
        )

    def test_stale_candidate_is_aborted_before_publish_and_releases_no_physics(self):
        fakes = _Fakes()
        committer = _make_committer(fakes)

        with self.assertRaises(CandidateSuperseded):
            committer.run_one_chunk(_command(0), command_is_current=lambda _c: False)

        self.assertEqual(
            fakes.log,
            ["mm.generate", "validate", "prepare", "timeline.abort", "mm.abort"],
        )
        self.assertNotIn("release", fakes.log)
        self.assertFalse(any(entry.startswith("publish") for entry in fakes.log))

    def test_generation_failure_releases_no_physics(self):
        fakes = _Fakes()
        mm = _MMShim(fakes, generate_error=RuntimeError("mm exploded"))
        committer = _make_committer(fakes, mm=mm)

        with self.assertRaises(RuntimeError):
            committer.run_one_chunk(_command(0))

        self.assertEqual(fakes.log, ["mm.generate"])
        self.assertNotIn("release", fakes.log)

    def test_publication_failure(self):
        # Publication is the last reversible step before the commit boundary.
        # When the socket publish raises, the committer must abort both the
        # timeline and mm candidate and re-raise, committing nothing and
        # releasing no physics.  Publication is attempted (after the last
        # supersession check) but never followed by mm.commit/timeline.commit.
        fakes = _Fakes()
        from mm_sonic.responsive_wiring import ManualChunkCommitter

        publisher = _PublisherShim(fakes, publish_error=RuntimeError("socket down"))
        committer = ManualChunkCommitter(
            mm=_MMShim(fakes),
            validator=_ValidatorShim(fakes),
            timeline=_TimelineShim(fakes),
            publish=publisher.publish,
            gate=_GateShim(fakes),
            session_id="session",
            steps_per_chunk=20,
            recorder=_RecorderShim(fakes),
        )

        with self.assertRaises(RuntimeError):
            committer.run_one_chunk(_command(0))

        self.assertEqual(
            fakes.log,
            [
                "mm.generate",
                "validate",
                "prepare",
                "publish:timeline:wait=False",
                "timeline.abort",
                "mm.abort",
            ],
        )
        # Nothing committed, no physics released, nothing recorded.
        self.assertNotIn("mm.commit", fakes.log)
        self.assertNotIn("timeline.commit", fakes.log)
        self.assertNotIn("release", fakes.log)
        self.assertNotIn("record", fakes.log)
        # next_chunk does not advance on a failed publication.
        self.assertEqual(committer.next_chunk, 0)

    def test_chunk_index_must_match_next_chunk(self):
        fakes = _Fakes()
        committer = _make_committer(fakes)

        with self.assertRaises(RuntimeError):
            committer.run_one_chunk(_command(3))

    def test_committer_continues_from_configured_next_chunk(self):
        # After a preload commits chunk 0, responsive control resumes at 1.
        fakes = _Fakes()
        from mm_sonic.responsive_wiring import ManualChunkCommitter

        committer = ManualChunkCommitter(
            mm=_MMShim(fakes),
            validator=_ValidatorShim(fakes),
            timeline=_TimelineShim(fakes),
            publish=_PublisherShim(fakes).publish,
            gate=_GateShim(fakes),
            session_id="session",
            steps_per_chunk=20,
            recorder=_RecorderShim(fakes),
            next_chunk=1,
        )

        self.assertEqual(committer.next_chunk, 1)
        # Chunk 0 is rejected; chunk 1 is accepted.
        with self.assertRaises(RuntimeError):
            committer.run_one_chunk(_command(0))
        accepted = committer.run_one_chunk(_command(1))
        self.assertEqual(accepted.presented_prefix_id, "session:candidate:000001")
        self.assertEqual(committer.next_chunk, 2)


def _monotonic_sequence(values):
    it = iter(values)
    return lambda: next(it)


class ManualChunkCommitterTraceTests(unittest.TestCase):
    def test_accepted_chunk_carries_monotone_timings_and_generated_root(self):
        # A known root trajectory: displacement = last row - first row.
        root = np.zeros((20, 3), dtype=np.float32)
        # Values exactly representable in float32 so end-minus-first is exact.
        root[-1] = (0.25, 0.125, 0.0)
        fakes = _Fakes(root_rows=root)
        # Timestamps for: mm_started, mm_completed, publication_sent,
        # committed, physics_release_requested, simulation_advance_completed.
        clock = _monotonic_sequence([10, 20, 30, 40, 50, 60])
        from mm_sonic.responsive_wiring import ManualChunkCommitter

        committer = ManualChunkCommitter(
            mm=_MMShim(fakes),
            validator=_ValidatorShim(fakes),
            timeline=_TimelineShim(fakes),
            publish=_PublisherShim(fakes).publish,
            gate=_GateShim(fakes),
            session_id="session",
            steps_per_chunk=20,
            recorder=_RecorderShim(fakes),
            monotonic_ns=clock,
        )

        accepted = committer.run_one_chunk(_command(0))

        self.assertEqual(accepted.presented_prefix_id, "session:candidate:000000")
        stamps = [
            accepted.mm_started_ns,
            accepted.mm_completed_ns,
            accepted.publication_sent_ns,
            accepted.committed_ns,
            accepted.physics_release_requested_ns,
            accepted.simulation_advance_completed_ns,
        ]
        self.assertEqual(stamps, sorted(stamps))
        # publication_sent precedes committed (honest ordering).
        self.assertLess(accepted.publication_sent_ns, accepted.committed_ns)
        self.assertEqual(
            accepted.generated_virtual_root_displacement_mujoco, (0.25, 0.125, 0.0)
        )

    def test_accepted_chunk_reads_real_observed_root_from_state_reader(self):
        fakes = _Fakes()

        class _FakeStateReader:
            def displacement(self, start_row, end_row):
                # Real end-minus-start pelvis displacement.
                self.calls = (start_row, end_row)
                return (0.25, -0.05, 0.0)

        reader = _FakeStateReader()
        from mm_sonic.responsive_wiring import ManualChunkCommitter

        committer = ManualChunkCommitter(
            mm=_MMShim(fakes),
            validator=_ValidatorShim(fakes),
            timeline=_TimelineShim(fakes),
            publish=_PublisherShim(fakes).publish,
            gate=_GateShim(fakes),
            session_id="session",
            steps_per_chunk=20,
            recorder=_RecorderShim(fakes),
            state_log_reader=reader,
        )

        accepted = committer.run_one_chunk(_command(0))

        self.assertEqual(accepted.observed_mujoco_root_displacement, (0.25, -0.05, 0.0))

    def test_accepted_chunk_observed_root_none_when_reader_absent(self):
        fakes = _Fakes()
        committer = _make_committer(fakes)

        accepted = committer.run_one_chunk(_command(0))

        self.assertIsNone(accepted.observed_mujoco_root_displacement)

    def test_accepted_chunk_observed_root_none_when_reader_returns_none(self):
        fakes = _Fakes()

        class _EmptyReader:
            def displacement(self, start_row, end_row):
                return None

        from mm_sonic.responsive_wiring import ManualChunkCommitter

        committer = ManualChunkCommitter(
            mm=_MMShim(fakes),
            validator=_ValidatorShim(fakes),
            timeline=_TimelineShim(fakes),
            publish=_PublisherShim(fakes).publish,
            gate=_GateShim(fakes),
            session_id="session",
            steps_per_chunk=20,
            recorder=_RecorderShim(fakes),
            state_log_reader=_EmptyReader(),
        )

        accepted = committer.run_one_chunk(_command(0))

        self.assertIsNone(accepted.observed_mujoco_root_displacement)


class BuildBoundaryTraceTests(unittest.TestCase):
    def _accepted(self, observed):
        from mm_sonic.responsive_wiring import AcceptedChunk

        return AcceptedChunk(
            presented_prefix_id="session:candidate:000000",
            mm_started_ns=30,
            mm_completed_ns=40,
            publication_sent_ns=50,
            committed_ns=60,
            physics_release_requested_ns=70,
            simulation_advance_completed_ns=80,
            generated_virtual_root_displacement_mujoco=(0.3, 0.1, 0.0),
            observed_mujoco_root_displacement=observed,
            advance=object(),
        )

    def _prefix(self):
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_scheduler import ScheduledPrefix

        snapshot = IntentSnapshot(revision=7, observed_ns=10, command=_command(0))
        return ScheduledPrefix(snapshot=snapshot, sampled_ns=20, accepted=None)

    def test_build_trace_binds_input_to_prefix_with_real_observed(self):
        from mm_sonic.boundary_trace import BoundaryTrace
        from mm_sonic.responsive_wiring import build_boundary_trace

        prefix = self._prefix()
        accepted = self._accepted((0.25, -0.05, 0.0))

        trace = build_boundary_trace(prefix, accepted, session_id="session")

        self.assertIsInstance(trace, BoundaryTrace)
        self.assertEqual(trace.input_transition_id, "session:rev:000007")
        self.assertEqual(trace.presented_prefix_id, "session:candidate:000000")
        self.assertEqual(trace.input_observed_ns, 10)
        self.assertEqual(trace.sampled_ns, 20)
        self.assertEqual(trace.observed_mujoco_root_displacement, (0.25, -0.05, 0.0))
        self.assertEqual(trace.requested_velocity_mujoco, (0.5, 0.0, 0.0))

    def test_build_trace_keeps_observed_none_when_unavailable(self):
        from mm_sonic.boundary_trace import BoundaryTrace
        from mm_sonic.responsive_wiring import build_boundary_trace

        trace = build_boundary_trace(
            self._prefix(), self._accepted(None), session_id="session"
        )

        self.assertIsInstance(trace, BoundaryTrace)
        # No fabricated zero: the field is explicitly None.
        self.assertIsNone(trace.observed_mujoco_root_displacement)

    def test_trace_record_is_json_serializable_append_only_evidence(self):
        from mm_sonic.responsive_wiring import build_boundary_trace, trace_record

        trace = build_boundary_trace(
            self._prefix(), self._accepted(None), session_id="session"
        )
        import json

        record = trace_record(trace)
        line = json.dumps(record)
        restored = json.loads(line)
        self.assertEqual(restored["input_transition_id"], "session:rev:000007")
        self.assertIsNone(restored["observed_mujoco_root_displacement"])
        self.assertEqual(
            restored["observed_root_available"], False
        )


class StateLogRootReaderTests(unittest.TestCase):
    def _write_log(self, path, rows):
        import json

        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for step, pelvis in rows:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "sim_time_s": step * 0.001,
                            "state": {"pelvis_position_m": list(pelvis)},
                        }
                    )
                    + "\n"
                )

    def test_reader_measures_end_minus_start_pelvis_displacement(self):
        import tempfile
        from pathlib import Path
        from mm_sonic.responsive_wiring import StateLogRootReader

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "state.jsonl"
            self._write_log(
                log,
                [
                    (10, (1.0, 2.0, 0.9)),
                    (20, (1.2, 2.1, 0.9)),
                    (30, (1.5, 2.3, 0.9)),
                ],
            )
            reader = StateLogRootReader(log)
            # Rows 0..2; displacement between row 0 and row 2.
            disp = reader.displacement(0, 2)
            self.assertEqual(len(disp), 3)
            self.assertAlmostEqual(disp[0], 0.5)
            self.assertAlmostEqual(disp[1], 0.3)
            self.assertAlmostEqual(disp[2], 0.0)

    def test_reader_returns_none_when_rows_missing(self):
        import tempfile
        from pathlib import Path
        from mm_sonic.responsive_wiring import StateLogRootReader

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "state.jsonl"
            self._write_log(log, [(10, (1.0, 2.0, 0.9))])
            reader = StateLogRootReader(log)
            # end_row index 5 does not exist -> unavailable, never fabricated.
            self.assertIsNone(reader.displacement(0, 5))

    def test_reader_returns_none_when_file_absent(self):
        from pathlib import Path
        from mm_sonic.responsive_wiring import StateLogRootReader

        reader = StateLogRootReader(Path("/nonexistent/state.jsonl"))
        self.assertIsNone(reader.displacement(0, 1))


class _FakeMailbox:
    """Serves scripted IntentSnapshots, tracking sampled chunk indices."""

    def __init__(self, snapshots) -> None:
        self._snapshots = list(snapshots)
        self.sampled_indices: list[int] = []
        self.current_revision = 1

    def sample_intent(self, chunk_index: int):
        self.sampled_indices.append(chunk_index)
        snapshot = self._snapshots.pop(0)
        self.current_revision = snapshot.revision
        return snapshot, None


class SchedulerCommitterIntegrationTests(unittest.TestCase):
    def test_scheduler_retries_same_chunk_index_on_supersession(self):
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_scheduler import ResponsiveScheduler

        stale = IntentSnapshot(revision=1, observed_ns=100, command=_command(0))
        fresh = IntentSnapshot(revision=2, observed_ns=200, command=_command(0))
        mailbox = _FakeMailbox([stale, fresh])

        # A committer whose first commit is superseded and second accepts.
        class _ScriptedCommitter:
            def __init__(self):
                self.calls = 0

            def run_one_chunk(self, command, *, command_is_current=None):
                self.calls += 1
                if self.calls == 1:
                    raise CandidateSuperseded("session:candidate:000000", command)
                return object()

        committer = _ScriptedCommitter()
        scheduler = ResponsiveScheduler(committer, mailbox)

        result = scheduler.run_one_prefix(0)

        self.assertIsNotNone(result)
        self.assertEqual(mailbox.sampled_indices, [0, 0])
        self.assertEqual(committer.calls, 2)

    def test_terminate_snapshot_returns_none_without_committing(self):
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_scheduler import ResponsiveScheduler

        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=None)
        mailbox = _FakeMailbox([snapshot])

        class _NeverCommitter:
            def __init__(self):
                self.calls = 0

            def run_one_chunk(self, command, *, command_is_current=None):
                self.calls += 1
                return object()

        committer = _NeverCommitter()
        scheduler = ResponsiveScheduler(committer, mailbox)

        result = scheduler.run_one_prefix(0)

        self.assertIsNone(result)
        self.assertEqual(committer.calls, 0)

    def test_real_committer_supersession_is_retried_by_scheduler(self):
        # Drive the real ManualChunkCommitter through the scheduler: the first
        # sampled revision is stale (a newer revision is live), so the committer
        # aborts and raises CandidateSuperseded; the second sample succeeds.
        from mm_sonic.operator_x11 import IntentSnapshot
        from mm_sonic.responsive_scheduler import ResponsiveScheduler

        fakes = _Fakes()
        committer = _make_committer(fakes)

        stale = IntentSnapshot(revision=1, observed_ns=100, command=_command(0))
        fresh = IntentSnapshot(revision=2, observed_ns=200, command=_command(0))

        class _AdvancingMailbox(_FakeMailbox):
            def sample_intent(self, chunk_index):
                snapshot, mapped = super().sample_intent(chunk_index)
                # First sample binds revision 1 but a newer revision is live.
                if len(self.sampled_indices) == 1:
                    self.current_revision = 2
                return snapshot, mapped

        mailbox = _AdvancingMailbox([stale, fresh])
        scheduler = ResponsiveScheduler(committer, mailbox)

        result = scheduler.run_one_prefix(0)

        self.assertIsNotNone(result)
        self.assertEqual(mailbox.sampled_indices, [0, 0])
        # First attempt aborted before publish; second committed and released.
        self.assertEqual(committer.next_chunk, 1)
        self.assertIn("timeline.abort", fakes.log)
        self.assertIn("release", fakes.log)


if __name__ == "__main__":
    unittest.main()
