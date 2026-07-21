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
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .boundary_trace import BoundaryTrace
from .coordinator import CandidateSuperseded
from .joints import ContractError
from .transform import holden_to_mujoco_quaternions, holden_to_mujoco_vectors


_SUPPORTED_SOURCE_INTERVALS = (5, 10)
_SOURCE_RATE_HZ = 25.0
_TARGET_ROWS_PER_INTERVAL = 2


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
    applied_velocity_mujoco_first: tuple[float, float, float]
    applied_velocity_mujoco_last: tuple[float, float, float]
    applied_heading_mujoco_wxyz_first: tuple[float, float, float, float]
    applied_heading_mujoco_wxyz_last: tuple[float, float, float, float]
    observed_mujoco_root_displacement: tuple[float, float, float] | None
    advance: object


class StateLogRootReader:
    """Read incremental pelvis displacement from scored simulator state.

    Each row is ``{"step", "sim_time_s", "state": {"pelvis_position_m": [..]}}``
    exactly as ``GatedSimulatorClient.advance`` writes it. Construction records
    the existing row count and last pose, which excludes already-committed
    preload rows. ``measure_advance`` then consumes only the number of new rows
    reported by one ``AdvanceResult`` (that field is a delta, not cumulative).
    It returns ``None`` when the new rows are unavailable and never fabricates
    a value.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        initial_position: tuple[float, float, float] | None = None,
    ) -> None:
        self._path = Path(path)
        self._next_row = 0
        self._last_position = initial_position
        existing = self._pelvis_rows()
        if existing is not None:
            self._next_row = len(existing)
            if existing:
                self._last_position = existing[-1]

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
            try:
                position = (
                    float(pelvis[0]),
                    float(pelvis[1]),
                    float(pelvis[2]),
                )
            except (TypeError, ValueError, OverflowError):
                return None
            if not all(math.isfinite(value) for value in position):
                return None
            rows.append(position)
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

    def measure_advance(
        self, row_count: int
    ) -> tuple[float, float, float] | None:
        """Consume exactly one advance's appended rows and measure its motion."""

        if type(row_count) is not int or row_count < 0:
            raise ContractError("state row count must be a nonnegative integer")
        if row_count == 0:
            return None
        rows = self._pelvis_rows()
        end_exclusive = self._next_row + row_count
        if rows is None or len(rows) < end_exclusive:
            return None
        end = rows[end_exclusive - 1]
        start = self._last_position
        self._next_row = end_exclusive
        self._last_position = end
        if start is None:
            return None
        return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def _generated_root_displacement(
    prepared: object,
) -> tuple[float, float, float]:
    rows = np.asarray(prepared.target.virtual_root_position, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 3:
        raise ContractError(
            "prepared virtual_root_position must be a nonempty Nx3 array"
        )
    delta = rows[-1] - rows[0]
    if not np.all(np.isfinite(delta)):
        raise ContractError("generated virtual root displacement must be finite")
    return (float(delta[0]), float(delta[1]), float(delta[2]))


def _applied_velocity_endpoints_mujoco(
    checked: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Transform the first/last source-step applied velocities to MuJoCo.

    Reads the honest per-step ``applied_velocity_holden`` produced by the
    server and maps both endpoints through the existing Holden-to-MuJoCo
    helper. The values are never inferred from root displacement.
    """

    command = getattr(checked, "command", None)
    if command is None or "applied_velocity_holden" not in command:
        raise ContractError(
            "validated source chunk must expose applied_velocity_holden"
        )
    rows = np.asarray(command["applied_velocity_holden"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 3:
        raise ContractError(
            "applied_velocity_holden must be a nonempty Nx3 array"
        )
    first = holden_to_mujoco_vectors(rows[0])
    last = holden_to_mujoco_vectors(rows[-1])
    return (
        (float(first[0]), float(first[1]), float(first[2])),
        (float(last[0]), float(last[1]), float(last[2])),
    )


def _applied_heading_endpoints_mujoco(
    checked: object,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    """Transform the first/last source-step applied headings to MuJoCo.

    Reads the honest per-step ``applied_heading_holden_wxyz`` produced by the
    turn-profile server and maps both endpoints through the existing
    Holden-to-MuJoCo quaternion helper.
    """

    command = getattr(checked, "command", None)
    if command is None or "applied_heading_holden_wxyz" not in command:
        raise ContractError(
            "validated turn-profile chunk must expose applied headings"
        )
    rows = np.asarray(command["applied_heading_holden_wxyz"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 4:
        raise ContractError("applied headings must be a nonempty Nx4 array")
    converted = holden_to_mujoco_quaternions(rows[[0, -1]])
    return tuple(map(float, converted[0])), tuple(map(float, converted[1]))


def build_boundary_trace(
    prefix: object,
    accepted: AcceptedChunk,
    *,
    session_id: str,
    release: AcceptedChunk | None = None,
) -> BoundaryTrace:
    """Bind one input transition to the prefix it produced (honest evidence).

    Combines the scheduler's ``ScheduledPrefix`` (the exact successful snapshot
    and its monotonic sampled timestamp) with the committer's ``AcceptedChunk``
    generation timings and root vectors. With one-prefix lookahead, the prefix
    committed at one boundary is physically presented by the next boundary's
    release; ``release`` supplies that later physical timing and observation.
    The observed displacement remains real-or-``None``, never fabricated.
    """

    snapshot = prefix.snapshot
    command = snapshot.command
    physical = accepted if release is None else release
    return BoundaryTrace(
        input_transition_id=f"{session_id}:rev:{snapshot.revision:06d}",
        presented_prefix_id=accepted.presented_prefix_id,
        input_observed_ns=snapshot.observed_ns,
        sampled_ns=prefix.sampled_ns,
        mm_started_ns=accepted.mm_started_ns,
        mm_completed_ns=accepted.mm_completed_ns,
        publication_sent_ns=accepted.publication_sent_ns,
        committed_ns=accepted.committed_ns,
        physics_release_requested_ns=physical.physics_release_requested_ns,
        simulation_advance_completed_ns=physical.simulation_advance_completed_ns,
        requested_velocity_mujoco=command.requested_velocity_mujoco,
        requested_heading_mujoco_wxyz=command.desired_heading_mujoco_wxyz,
        generated_virtual_root_displacement_mujoco=(
            accepted.generated_virtual_root_displacement_mujoco
        ),
        applied_velocity_mujoco_first=accepted.applied_velocity_mujoco_first,
        applied_velocity_mujoco_last=accepted.applied_velocity_mujoco_last,
        applied_heading_mujoco_wxyz_first=(
            accepted.applied_heading_mujoco_wxyz_first
        ),
        applied_heading_mujoco_wxyz_last=(
            accepted.applied_heading_mujoco_wxyz_last
        ),
        observed_mujoco_root_displacement=(
            physical.observed_mujoco_root_displacement
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
        # Schema v3: the public field set now also carries the first/last
        # capped applied headings in MuJoCo coordinates.
        "schema": "mm-sonic-responsive-boundary-trace/v3",
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
        "applied_velocity_mujoco_first": list(
            trace.applied_velocity_mujoco_first
        ),
        "applied_velocity_mujoco_last": list(
            trace.applied_velocity_mujoco_last
        ),
        "applied_heading_mujoco_wxyz_first": list(
            trace.applied_heading_mujoco_wxyz_first
        ),
        "applied_heading_mujoco_wxyz_last": list(
            trace.applied_heading_mujoco_wxyz_last
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
        source_intervals: int = 10,
        sim_dt_s: float = 0.02,
        state_log_reader: object | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        next_chunk: int = 0,
    ) -> None:
        if type(session_id) is not str or not session_id:
            raise ContractError("committer session_id must be a nonempty string")
        if type(steps_per_chunk) is not int or steps_per_chunk <= 0:
            raise ContractError("committer steps_per_chunk must be a positive integer")
        if (
            type(source_intervals) is not int
            or source_intervals not in _SUPPORTED_SOURCE_INTERVALS
        ):
            raise ContractError(
                "committer source_intervals must be one of "
                + " or ".join(str(count) for count in _SUPPORTED_SOURCE_INTERVALS)
            )
        if (
            isinstance(sim_dt_s, bool)
            or not isinstance(sim_dt_s, (int, float))
            or not math.isfinite(float(sim_dt_s))
            or float(sim_dt_s) <= 0.0
        ):
            raise ContractError("committer sim_dt_s must be finite and positive")
        # MM, target, and physics operate at different rates. Compare their
        # durations instead of equating row/step counts: five 25-Hz source
        # intervals are ten 50-Hz target rows but forty 200-Hz physics steps.
        expected_duration_s = source_intervals / _SOURCE_RATE_HZ
        physics_duration_s = steps_per_chunk * float(sim_dt_s)
        if not math.isclose(
            physics_duration_s,
            expected_duration_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ContractError(
                "committer physics duration "
                f"{physics_duration_s:.12g}s must equal MM horizon "
                f"{expected_duration_s:.12g}s"
            )
        if type(next_chunk) is not int or next_chunk < 0:
            raise ContractError("committer next_chunk must be a nonnegative integer")
        if not callable(publish):
            raise ContractError("committer publish must be callable")
        if state_log_reader is not None and not hasattr(
            state_log_reader, "measure_advance"
        ):
            raise ContractError(
                "committer state_log_reader must expose measure_advance"
            )
        if not callable(monotonic_ns):
            raise ContractError("committer monotonic_ns must be callable")
        self._mm = mm
        self._validator = validator
        self._timeline = timeline
        self._publish = publish
        self._gate = gate
        self._session_id = session_id
        self._steps_per_chunk = steps_per_chunk
        self._source_intervals = source_intervals
        self._expected_target_rows = (
            source_intervals * _TARGET_ROWS_PER_INTERVAL
        )
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
            source_intervals=self._source_intervals,
        )
        mm_completed_ns = self._monotonic_ns()
        checked = self._validator.validate_source(raw)
        prepared = self._timeline.prepare(checked)
        # Extract evidence while the transaction is still reversible.  Evidence
        # assembly must never be the first failure after physics is released.
        generated_root_displacement = _generated_root_displacement(prepared)
        applied_first, applied_last = _applied_velocity_endpoints_mujoco(checked)
        (
            applied_heading_first,
            applied_heading_last,
        ) = _applied_heading_endpoints_mujoco(checked)
        # Guard the target's 50-Hz horizon here while the transaction remains
        # reversible. Physics duration was independently checked from sim_dt.
        published_rows = int(
            np.asarray(prepared.target.virtual_root_position).shape[0]
        )
        if published_rows != self._expected_target_rows:
            self._timeline.abort(candidate)
            self._mm.abort(candidate)
            raise ContractError(
                "prepared target rows "
                f"{published_rows} must equal source horizon target rows "
                f"{self._expected_target_rows}"
            )

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

        physics_release_requested_ns = self._monotonic_ns()
        advance = self._gate.release_steps(self._steps_per_chunk)
        simulation_advance_completed_ns = self._monotonic_ns()
        self._recorder.record(command)
        self._next_chunk += 1

        observed = self._read_observed_root(advance)
        return AcceptedChunk(
            presented_prefix_id=candidate,
            mm_started_ns=mm_started_ns,
            mm_completed_ns=mm_completed_ns,
            publication_sent_ns=publication_sent_ns,
            committed_ns=committed_ns,
            physics_release_requested_ns=physics_release_requested_ns,
            simulation_advance_completed_ns=simulation_advance_completed_ns,
            generated_virtual_root_displacement_mujoco=generated_root_displacement,
            applied_velocity_mujoco_first=applied_first,
            applied_velocity_mujoco_last=applied_last,
            applied_heading_mujoco_wxyz_first=applied_heading_first,
            applied_heading_mujoco_wxyz_last=applied_heading_last,
            observed_mujoco_root_displacement=observed,
            advance=advance,
        )

    def _read_observed_root(
        self, advance: object
    ) -> tuple[float, float, float] | None:
        """Read a real observed pelvis displacement, or None if unavailable.

        ``AdvanceResult.state_rows`` is the number of rows appended by this
        advance. The stateful reader already captured the prior physical pose
        and file cursor, so it consumes that delta exactly once.
        """

        if self._state_log_reader is None:
            return None
        state_rows = getattr(advance, "state_rows", None)
        if not isinstance(state_rows, int) or isinstance(state_rows, bool):
            return None
        try:
            observed = self._state_log_reader.measure_advance(state_rows)
        except Exception:
            # Motion is already committed and simulated. Evidence availability
            # must not retroactively turn that successful boundary into a
            # command failure.
            return None
        if not isinstance(observed, (tuple, list)) or len(observed) != 3:
            return None
        try:
            vector = tuple(float(value) for value in observed)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in vector):
            return None
        return vector
