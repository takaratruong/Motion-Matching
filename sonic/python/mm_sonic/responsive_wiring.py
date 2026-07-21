"""Stage-R1 responsive wiring: adapt manual_demo's committed transaction.

``ManualChunkCommitter`` presents ``manual_demo``'s existing per-boundary
generate -> validate -> prepare -> publish -> commit transaction as the
``run_one_chunk(command, *, command_is_current)`` duck type the
``ResponsiveScheduler`` requires (``responsive_scheduler.py``).  It preserves the
coordinator's fail-closed ordering exactly:

    generate -> validate -> prepare -> (LAST supersession check) ->
    publish -> mm.commit -> timeline.commit -> release physics -> record

Physics is released only after publication and both commits.  Stale candidates
are aborted (``timeline.abort`` + ``mm.abort``) before publication and raise the
typed ``CandidateSuperseded`` the scheduler retries; no physics is released on
generation, validation, preparation, supersession, or publication failure.

The module holds no NVIDIA policy/protocol code; it only orchestrates the
already-authenticated ``mm``, ``validator``, ``timeline``, ``publish``,
``gate``, and ``recorder`` collaborators ``manual_demo`` already owns.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .boundary_trace import BoundaryTrace
from .coordinator import CandidateSuperseded
from .joints import ContractError


@dataclass(frozen=True)
class AcceptedChunk:
    """One committed responsive prefix with honest producer-side evidence.

    Timings are a single monotonic clock read in the real adapter order:
    ``mm_started`` <= ``mm_completed`` <= ``publication_sent`` (wait=False
    socket return) <= ``committed`` (both commits done) <=
    ``physics_release_requested`` (release_steps invoked) <=
    ``simulation_advance_completed`` (release_steps returned).

    ``generated_virtual_root_displacement_mujoco`` is the prepared target's
    last-minus-first ``virtual_root_position`` row.
    ``observed_mujoco_root_displacement`` is a real state-log measurement, or
    ``None`` when the state log does not bound the released chunk -- never a
    fabricated zero.
    """

    presented_prefix_id: str
    mm_started_ns: int
    mm_completed_ns: int
    publication_sent_ns: int
    committed_ns: int
    physics_release_requested_ns: int
    simulation_advance_completed_ns: int
    generated_virtual_root_displacement_mujoco: tuple[float, float, float]
    observed_mujoco_root_displacement: tuple[float, float, float] | None
    advance: object


class StateLogRootReader:
    """Read real pelvis displacement from a scored-sim-logs ``state.jsonl``.

    Each row is ``{"step", "sim_time_s", "state": {"pelvis_position_m": [..]}}``
    exactly as ``GatedSimulatorClient.advance`` writes it.  ``displacement``
    returns the end-minus-start pelvis vector in the MuJoCo world basis, or
    ``None`` when either bounding row is missing.  It never fabricates a value.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _pelvis_rows(self) -> list[tuple[float, float, float]] | None:
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return None
        rows: list[tuple[float, float, float]] = []
        for line in text.splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
                pelvis = row["state"]["pelvis_position_m"]
            except (ValueError, KeyError, TypeError):
                return None
            if not isinstance(pelvis, (list, tuple)) or len(pelvis) != 3:
                return None
            rows.append((float(pelvis[0]), float(pelvis[1]), float(pelvis[2])))
        return rows

    def displacement(
        self, start_row: int, end_row: int
    ) -> tuple[float, float, float] | None:
        rows = self._pelvis_rows()
        if rows is None:
            return None
        if start_row < 0 or end_row < 0 or start_row >= len(rows) or end_row >= len(
            rows
        ):
            return None
        start = rows[start_row]
        end = rows[end_row]
        return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def _generated_root_displacement(
    prepared: object,
) -> tuple[float, float, float]:
    rows = np.asarray(prepared.target.virtual_root_position, dtype=np.float64)
    delta = rows[-1] - rows[0]
    return (float(delta[0]), float(delta[1]), float(delta[2]))


def build_boundary_trace(
    prefix: object, accepted: AcceptedChunk, *, session_id: str
) -> BoundaryTrace:
    """Bind one input transition to the prefix it produced (honest evidence).

    Combines the scheduler's ``ScheduledPrefix`` (the exact successful snapshot
    and its monotonic sampled timestamp) with the committer's ``AcceptedChunk``
    timings and root vectors.  ``observed_mujoco_root_displacement`` is passed
    through as-is: a real measurement or ``None``, never a fabricated zero.
    """

    snapshot = prefix.snapshot
    command = snapshot.command
    return BoundaryTrace(
        input_transition_id=f"{session_id}:rev:{snapshot.revision:06d}",
        presented_prefix_id=accepted.presented_prefix_id,
        input_observed_ns=snapshot.observed_ns,
        sampled_ns=prefix.sampled_ns,
        mm_started_ns=accepted.mm_started_ns,
        mm_completed_ns=accepted.mm_completed_ns,
        publication_sent_ns=accepted.publication_sent_ns,
        committed_ns=accepted.committed_ns,
        physics_release_requested_ns=accepted.physics_release_requested_ns,
        simulation_advance_completed_ns=accepted.simulation_advance_completed_ns,
        requested_velocity_mujoco=command.requested_velocity_mujoco,
        requested_heading_mujoco_wxyz=command.desired_heading_mujoco_wxyz,
        generated_virtual_root_displacement_mujoco=(
            accepted.generated_virtual_root_displacement_mujoco
        ),
        observed_mujoco_root_displacement=(
            accepted.observed_mujoco_root_displacement
        ),
    )


def trace_record(trace: BoundaryTrace) -> dict:
    """Serialize a BoundaryTrace to append-only JSONL-ready evidence.

    ``observed_root_available`` states plainly whether the observed root
    displacement is a real state-log measurement or was unavailable for this
    boundary; the vector itself stays ``None`` rather than a fabricated zero.
    """

    observed = trace.observed_mujoco_root_displacement
    return {
        "schema": "mm-sonic-responsive-boundary-trace/v1",
        "input_transition_id": trace.input_transition_id,
        "presented_prefix_id": trace.presented_prefix_id,
        "input_observed_ns": trace.input_observed_ns,
        "sampled_ns": trace.sampled_ns,
        "mm_started_ns": trace.mm_started_ns,
        "mm_completed_ns": trace.mm_completed_ns,
        "publication_sent_ns": trace.publication_sent_ns,
        "committed_ns": trace.committed_ns,
        "physics_release_requested_ns": trace.physics_release_requested_ns,
        "simulation_advance_completed_ns": trace.simulation_advance_completed_ns,
        "requested_velocity_mujoco": list(trace.requested_velocity_mujoco),
        "requested_heading_mujoco_wxyz": list(trace.requested_heading_mujoco_wxyz),
        "generated_virtual_root_displacement_mujoco": list(
            trace.generated_virtual_root_displacement_mujoco
        ),
        "observed_mujoco_root_displacement": (
            None if observed is None else list(observed)
        ),
        "observed_root_available": observed is not None,
    }


class ManualChunkCommitter:
    """Adapt manual_demo's committed transaction to ``run_one_chunk``."""

    def __init__(
        self,
        *,
        mm: object,
        validator: object,
        timeline: object,
        publish: Callable[..., None],
        gate: object,
        session_id: str,
        steps_per_chunk: int,
        recorder: object,
        state_log_reader: object | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        next_chunk: int = 0,
    ) -> None:
        if type(session_id) is not str or not session_id:
            raise ContractError("committer session_id must be a nonempty string")
        if type(steps_per_chunk) is not int or steps_per_chunk <= 0:
            raise ContractError("committer steps_per_chunk must be a positive integer")
        if type(next_chunk) is not int or next_chunk < 0:
            raise ContractError("committer next_chunk must be a nonnegative integer")
        if not callable(publish):
            raise ContractError("committer publish must be callable")
        if state_log_reader is not None and not hasattr(
            state_log_reader, "displacement"
        ):
            raise ContractError("committer state_log_reader must expose displacement")
        if not callable(monotonic_ns):
            raise ContractError("committer monotonic_ns must be callable")
        self._mm = mm
        self._validator = validator
        self._timeline = timeline
        self._publish = publish
        self._gate = gate
        self._session_id = session_id
        self._steps_per_chunk = steps_per_chunk
        self._recorder = recorder
        self._state_log_reader = state_log_reader
        self._monotonic_ns = monotonic_ns
        self._next_chunk = next_chunk

    @property
    def next_chunk(self) -> int:
        return self._next_chunk

    def run_one_chunk(self, command: object, *, command_is_current=None) -> object:
        """Commit exactly one prefix in the coordinator's failure order."""

        chunk_index = command.chunk_index
        if chunk_index != self._next_chunk:
            raise RuntimeError(
                f"command index {chunk_index} != next chunk {self._next_chunk}"
            )
        candidate = f"{self._session_id}:candidate:{chunk_index:06d}"

        # generate -> validate -> prepare (pre-publication, reversible).
        mm_started_ns = self._monotonic_ns()
        raw = self._mm.generate(
            command,
            session_id=self._session_id,
            candidate_id=candidate,
            predecessor_id=self._timeline.last_accepted_candidate_id,
            source_intervals=10,
        )
        mm_completed_ns = self._monotonic_ns()
        checked = self._validator.validate_source(raw)
        prepared = self._timeline.prepare(checked)

        # LAST supersession check, exactly where the coordinator places it.
        if command_is_current is not None and not command_is_current(command):
            self._timeline.abort(candidate)
            self._mm.abort(candidate)
            raise CandidateSuperseded(candidate, command)

        # publish; on any failure abort the candidate and re-raise (no release).
        try:
            self._publish(prepared.target.buffer, phase="timeline", wait=False)
        except BaseException:
            self._timeline.abort(candidate)
            self._mm.abort(candidate)
            raise
        # wait=False: this timestamp means the socket publication returned, not
        # a GEAR stream-processing acknowledgement (which is WAIT-phase only).
        publication_sent_ns = self._monotonic_ns()

        # irreversible boundary: commit both, then release physics, then record.
        self._mm.commit(candidate)
        self._timeline.commit(prepared)
        committed_ns = self._monotonic_ns()

        state_rows_before = self._state_rows_of(
            getattr(self._gate, "state_rows", None)
        )
        physics_release_requested_ns = self._monotonic_ns()
        advance = self._gate.release_steps(self._steps_per_chunk)
        simulation_advance_completed_ns = self._monotonic_ns()
        self._recorder.record(command)
        self._next_chunk += 1

        observed = self._read_observed_root(state_rows_before, advance)
        return AcceptedChunk(
            presented_prefix_id=candidate,
            mm_started_ns=mm_started_ns,
            mm_completed_ns=mm_completed_ns,
            publication_sent_ns=publication_sent_ns,
            committed_ns=committed_ns,
            physics_release_requested_ns=physics_release_requested_ns,
            simulation_advance_completed_ns=simulation_advance_completed_ns,
            generated_virtual_root_displacement_mujoco=(
                _generated_root_displacement(prepared)
            ),
            observed_mujoco_root_displacement=observed,
            advance=advance,
        )

    @staticmethod
    def _state_rows_of(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _read_observed_root(
        self, state_rows_before: int | None, advance: object
    ) -> tuple[float, float, float] | None:
        """Read a real observed pelvis displacement, or None if unavailable.

        The observed root is only recoverable when a state-log reader is wired
        and the ``AdvanceResult`` exposes the bounding ``state_rows``.  When any
        of that is missing the displacement is explicitly ``None`` -- never a
        fabricated zero, since ``AdvanceResult`` itself carries no root pose.
        """

        if self._state_log_reader is None:
            return None
        state_rows_after = getattr(advance, "state_rows", None)
        if not isinstance(state_rows_after, int) or isinstance(
            state_rows_after, bool
        ):
            return None
        start_index = 0 if state_rows_before is None else state_rows_before
        # state.jsonl rows are 0-indexed; the last logged row is state_rows-1.
        end_index = state_rows_after - 1
        if end_index < start_index:
            return None
        return self._state_log_reader.displacement(start_index, end_index)
