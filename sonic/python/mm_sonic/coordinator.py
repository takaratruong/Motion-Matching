"""Transactional MM-to-SONIC coordination and frame-zero readiness.

The coordinator is the sole owner of session/candidate identifiers and command
latching.  Its streaming boundary deliberately records only a successful local
send; GEAR PUB/SUB has no per-message receipt.  A terminal delivery audit is a
separate mandatory evidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np

from .commands import CommandSample
from .joints import ContractError, JointContract
from .process import AdvanceResult
from .schema import InitialBoundary, SourceChunk, parse_initial_boundary, parse_source_chunk
from .timeline import CanonicalTargetBuffer, TargetTimeline


LOGGER_JOINT_PERMUTATION = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10,
    14, 18, 2, 5, 8, 11, 15, 19, 21, 23,
    25, 27, 12, 16, 20, 22, 24, 26, 28,
)

_TIMING_STAGES = (
    "mm_generation",
    "projection_validation",
    "resampling",
    "artifact_enqueue",
    "publication",
    "simulation_advance",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CoordinatorState(str, Enum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    READY_PAUSED = "ready_paused"
    CANDIDATE = "candidate"
    COMMITTED = "committed"
    ADVANCING = "advancing"
    TERMINAL = "terminal"


class StateTransitionError(RuntimeError):
    """A public operation was attempted in an incompatible state."""


class IntegrationFailure(RuntimeError):
    """A terminal integration failure with an explicit transaction boundary."""

    def __init__(self, phase: str, site: str, cause: BaseException):
        self.phase = phase
        self.site = site
        self.cause = cause
        super().__init__(f"{phase} integration failure at {site}: {cause}")


class CandidateSuperseded(RuntimeError):
    """A pre-publication candidate was aborted because a newer command exists.

    This is a typed nonterminal outcome: the coordinator has discarded the
    stale candidate via the existing precommit cleanup, restored READY_PAUSED,
    and released no physics.  It is not a failure and never terminalizes.
    """

    def __init__(self, candidate_id: str, command: CommandSample):
        self.candidate_id = candidate_id
        self.command = command
        super().__init__(f"candidate {candidate_id} superseded before publication")


@dataclass(frozen=True)
class SessionConfig:
    scene_id: str
    route_id: str
    terrain_weight: float

    def __post_init__(self) -> None:
        if type(self.scene_id) is not str or not self.scene_id:
            raise ContractError("session scene_id must be a nonempty string")
        if type(self.route_id) is not str or not self.route_id:
            raise ContractError("session route_id must be a nonempty string")
        if type(self.terrain_weight) not in (int, float) or not math.isfinite(
            float(self.terrain_weight)
        ):
            raise ContractError("session terrain_weight must be finite")
        converted = np.float32(self.terrain_weight)
        if not np.isfinite(converted) or float(converted) != float(
            self.terrain_weight
        ):
            raise ContractError("session terrain_weight must be exact binary32")
        object.__setattr__(self, "terrain_weight", float(converted))


@dataclass(frozen=True)
class DeliveryAudit:
    expected_rows: int
    observed_rows: int
    transmitted_indices_exact: bool
    official_rows_exact: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.expected_rows) is not int
            or type(self.observed_rows) is not int
            or self.expected_rows <= 0
            or self.observed_rows < 0
        ):
            raise ContractError("delivery audit row counts are invalid")
        if (
            type(self.transmitted_indices_exact) is not bool
            or type(self.official_rows_exact) is not bool
        ):
            raise ContractError("delivery audit equality flags must be booleans")
        if (
            type(self.evidence_sha256) is not str
            or _SHA256.fullmatch(self.evidence_sha256) is None
        ):
            raise ContractError("delivery audit evidence must be a SHA-256 digest")

    @property
    def passed(self) -> bool:
        return (
            self.expected_rows == self.observed_rows
            and self.transmitted_indices_exact
            and self.official_rows_exact
        )


@dataclass(frozen=True)
class PreflightResult:
    session_id: str
    readiness_log_start_offset: int
    scoring_log_offset: int
    readiness_attempts: int
    official_log_path: str
    log_device: int
    log_inode: int
    expected_row_sha256: str

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise ContractError("readiness session_id must be nonempty")
        if (
            type(self.readiness_log_start_offset) is not int
            or type(self.scoring_log_offset) is not int
            or self.readiness_log_start_offset < 0
            or self.scoring_log_offset < self.readiness_log_start_offset
        ):
            raise ContractError("readiness log offsets are invalid")
        if type(self.readiness_attempts) is not int or self.readiness_attempts <= 0:
            raise ContractError("readiness attempts must be positive")
        if type(self.official_log_path) is not str or not self.official_log_path:
            raise ContractError("readiness official log path must be nonempty")
        if (
            type(self.log_device) is not int
            or type(self.log_inode) is not int
            or self.log_device < 0
            or self.log_inode <= 0
        ):
            raise ContractError("readiness log identity is invalid")
        if (
            type(self.expected_row_sha256) is not str
            or _SHA256.fullmatch(self.expected_row_sha256) is None
        ):
            raise ContractError("readiness expected row digest is invalid")


@dataclass(frozen=True)
class DeliveryAuditEvidence:
    """Coordinator-owned inputs supplied to an explicit delivery auditor."""

    session_id: str
    canonical_buffer: CanonicalTargetBuffer
    readiness: PreflightResult
    official_log_slice: bytes
    readiness_publications: tuple[Mapping[str, object], ...]
    timeline_publications: tuple[Mapping[str, object], ...]
    run: object

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise ContractError("delivery evidence session_id must be nonempty")
        if not isinstance(self.canonical_buffer, CanonicalTargetBuffer):
            raise ContractError("delivery evidence requires the canonical buffer")
        if not isinstance(self.readiness, PreflightResult):
            raise ContractError("delivery evidence requires readiness evidence")
        if self.readiness.session_id != self.session_id:
            raise ContractError("delivery evidence session does not match readiness")
        if not isinstance(self.official_log_slice, bytes):
            raise ContractError("delivery evidence official log slice must be bytes")
        for label, publications in (
            ("readiness", self.readiness_publications),
            ("timeline", self.timeline_publications),
        ):
            if type(publications) is not tuple or any(
                not isinstance(value, Mapping) for value in publications
            ):
                raise ContractError(
                    f"delivery evidence {label} publications must be mappings"
                )
            object.__setattr__(
                self,
                f"{label}_publications",
                tuple(MappingProxyType(dict(value)) for value in publications),
            )


@dataclass(frozen=True)
class TimingRecord:
    candidate_id: str
    durations_ns: Mapping[str, int | None]
    failed_stage: str | None
    wall_started_ns: int
    wall_finished_ns: int

    def __post_init__(self) -> None:
        durations = dict(self.durations_ns)
        if tuple(durations) != _TIMING_STAGES:
            raise ContractError("timing record stages changed")
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in durations.values()
        ):
            raise ContractError("timing durations must be nonnegative integer ns")
        if (
            type(self.wall_started_ns) is not int
            or type(self.wall_finished_ns) is not int
        ):
            raise ContractError("wall timing must use integer ns")
        object.__setattr__(self, "durations_ns", MappingProxyType(durations))


@dataclass(frozen=True)
class AcceptedChunk:
    target: object
    advance: AdvanceResult
    timings_ns: Mapping[str, int]
    wall_started_ns: int
    wall_finished_ns: int

    def __post_init__(self) -> None:
        timings = dict(self.timings_ns)
        if tuple(timings) != _TIMING_STAGES or any(
            type(value) is not int or value < 0 for value in timings.values()
        ):
            raise ContractError("accepted timings must contain exact integer-ns stages")
        object.__setattr__(self, "timings_ns", MappingProxyType(timings))


@dataclass(frozen=True)
class RejectionRecord:
    candidate_id: str
    failure_site: str
    error_type: str
    error_message: str
    mm_alive: bool
    abort_required: bool
    abort_sent: bool
    timeline_abort_required: bool


@dataclass(frozen=True)
class AbortDecisionRecord:
    candidate_id: str
    mm_abort_required: bool
    mm_abort_sent: bool
    timeline_abort_required: bool
    timeline_aborted: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TerminalVerdict:
    status: str
    failure_phase: str | None
    failure_site: str | None
    error_type: str | None
    error_message: str | None
    accepted_chunks: int
    simulation_paused: bool
    delivery_audit_required: bool
    delivery_audit_completed: bool
    delivery_audit: DeliveryAudit | None


class SourceValidator:
    """Production schema adapter kept injectable for boundary tests."""

    def __init__(self, contract: JointContract):
        self._contract = contract

    def validate_initial(self, raw: object) -> InitialBoundary:
        if type(raw) is dict and "initial_boundary" in raw:
            raw = raw["initial_boundary"]
        return parse_initial_boundary(raw, self._contract)

    def validate_source(self, raw: object) -> SourceChunk:
        return parse_source_chunk(raw, self._contract)


def _defaultfloat(value: float) -> str:
    if not math.isfinite(value):
        raise ContractError("official target row must contain finite values")
    # The pinned g1_deploy_onnx_ref.cpp logger uses operator<< without a
    # precision override.  C++ defaultfloat therefore emits six significant
    # digits for these promoted streamed f32 values.
    return format(value, ".6g")


def expected_official_target_row(buffer: CanonicalTargetBuffer) -> bytes:
    """Render frame zero exactly as the pinned 36-value official logger row."""

    if not isinstance(buffer, CanonicalTargetBuffer):
        raise ContractError("official target row requires a canonical buffer")
    if buffer.count != 1 or int(buffer.frame_index[0]) != 0:
        raise ContractError("readiness requires exactly canonical frame zero")
    permutation = np.asarray(LOGGER_JOINT_PERMUTATION, dtype=np.int64)
    values = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            buffer.body_quat_w[0],
            buffer.joint_position[0, permutation],
        )
    )
    if values.shape != (36,) or not np.all(np.isfinite(values)):
        raise ContractError("official target row must contain 36 finite values")
    text = ",".join(_defaultfloat(float(value)) for value in values)
    return (text + ",\n").encode("ascii")


def parse_official_target_row(line: bytes | bytearray | memoryview) -> bytes:
    """Validate and preserve one exact defaultfloat logger row."""

    if not isinstance(line, (bytes, bytearray, memoryview)):
        raise ContractError("official target row must be bytes")
    encoded = bytes(line)
    if not encoded.endswith(b"\n") or encoded.count(b"\n") != 1:
        raise ContractError(
            "official target row must contain one row ending in exactly one newline"
        )
    if encoded.endswith(b"\r\n"):
        raise ContractError("official target row must use a Unix newline")
    body = encoded[:-1]
    if not body.endswith(b","):
        raise ContractError("official target row requires one trailing comma")
    raw_fields = body[:-1].split(b",")
    if len(raw_fields) != 36 or any(not field for field in raw_fields):
        raise ContractError("official target row must contain exactly 36 values")
    for raw in raw_fields:
        try:
            token = raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ContractError("official target row must contain ASCII values") from error
        try:
            value = float(token)
        except ValueError as error:
            raise ContractError("official target row contains an invalid number") from error
        if not math.isfinite(value):
            raise ContractError("official target row values must be finite")
        if token != _defaultfloat(value):
            raise ContractError(
                "official target row values must use pinned defaultfloat formatting"
            )
    return encoded


class _OfficialTargetLog:
    """Retain one regular-file identity across readiness and final audit."""

    def __init__(self, path: Path):
        self.path = path
        self._descriptor = -1
        self._identity: tuple[int, int] | None = None

    def _bind(self, *, required: bool) -> bool:
        if self._descriptor >= 0:
            self._verify_identity()
            return True
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            if required:
                raise ContractError(
                    f"official target log is missing: {self.path}"
                )
            return False
        except OSError as error:
            raise ContractError(
                f"cannot open official target log: {self.path}"
            ) from error
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ContractError(
                    "official target log must be a regular nonsymlink file"
                )
            if opened.st_nlink != 1:
                raise ContractError("official target log cannot be a hard link")
            try:
                visible = self.path.lstat()
            except OSError as error:
                raise ContractError(
                    "official target log identity disappeared during open"
                ) from error
            if (
                stat.S_ISLNK(visible.st_mode)
                or not stat.S_ISREG(visible.st_mode)
                or (visible.st_dev, visible.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise ContractError(
                    "official target log identity changed during open"
                )
            self._descriptor = descriptor
            self._identity = (int(opened.st_dev), int(opened.st_ino))
            descriptor = -1
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _verify_identity(self) -> os.stat_result:
        if self._descriptor < 0 or self._identity is None:
            raise ContractError("official target log identity is not bound")
        opened = os.fstat(self._descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ContractError("official target log identity is no longer regular")
        try:
            visible = self.path.lstat()
        except OSError as error:
            raise ContractError(
                "official target log identity is no longer visible"
            ) from error
        if (
            stat.S_ISLNK(visible.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or (int(visible.st_dev), int(visible.st_ino)) != self._identity
            or (int(opened.st_dev), int(opened.st_ino)) != self._identity
        ):
            raise ContractError("official target log identity changed")
        return opened

    def anchor(self) -> int:
        if not self._bind(required=False):
            return 0
        return int(self._verify_identity().st_size)

    @property
    def identity(self) -> tuple[int, int]:
        self._bind(required=True)
        self._verify_identity()
        assert self._identity is not None
        return self._identity

    def read_from(self, offset: int) -> bytes:
        if type(offset) is not int or offset < 0:
            raise ContractError("official target log offset must be nonnegative")
        if not self._bind(required=False):
            return b""
        metadata = self._verify_identity()
        if metadata.st_size < offset:
            raise ContractError("official target log was truncated")
        chunks: list[bytes] = []
        position = offset
        remaining = int(metadata.st_size) - offset
        while remaining:
            chunk = os.pread(self._descriptor, min(65536, remaining), position)
            if not chunk:
                raise ContractError("official target log changed during read")
            chunks.append(chunk)
            position += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1


def _complete_rows(data: bytes) -> tuple[tuple[bytes, ...], bytes]:
    rows: list[bytes] = []
    start = 0
    while True:
        newline = data.find(b"\n", start)
        if newline < 0:
            break
        row = data[start : newline + 1]
        rows.append(parse_official_target_row(row))
        start = newline + 1
    return tuple(rows), data[start:]


class _TimingTracker:
    def __init__(
        self,
        candidate_id: str,
        monotonic_ns: Callable[[], int],
        wall_time_ns: Callable[[], int],
    ) -> None:
        self.candidate_id = candidate_id
        self._monotonic_ns = monotonic_ns
        self._wall_time_ns = wall_time_ns
        self.wall_started_ns = self._wall()
        self.durations: dict[str, int | None] = {
            stage: None for stage in _TIMING_STAGES
        }

    def _monotonic(self) -> int:
        value = self._monotonic_ns()
        if type(value) is not int or value < 0:
            raise ContractError("monotonic clock must return nonnegative integer ns")
        return value

    def _wall(self) -> int:
        value = self._wall_time_ns()
        if type(value) is not int or value < 0:
            raise ContractError("wall clock must return nonnegative integer ns")
        return value

    def call(self, stage: str, function: Callable[[], object]) -> object:
        if stage not in self.durations or self.durations[stage] is not None:
            raise ContractError(f"timing stage cannot run twice: {stage}")
        started = self._monotonic()
        try:
            return function()
        finally:
            finished = self._monotonic()
            if finished < started:
                raise ContractError("monotonic clock moved backwards")
            self.durations[stage] = finished - started

    def finish(self, failed_stage: str | None) -> TimingRecord:
        wall_finished = self._wall()
        return TimingRecord(
            candidate_id=self.candidate_id,
            durations_ns=self.durations,
            failed_stage=failed_stage,
            wall_started_ns=self.wall_started_ns,
            wall_finished_ns=wall_finished,
        )


class _MM(Protocol):
    outstanding_candidate_id: str | None
    active_candidate_id: str | None

    def require_alive(self) -> None: ...
    def reset(self, config: SessionConfig, *, session_id: str) -> object: ...
    def generate(self, command: CommandSample, **fields: object) -> object: ...
    def commit(self, candidate_id: str) -> None: ...
    def abort(self, candidate_id: str) -> None: ...
    def close(self) -> None: ...


class _DeliveryAuditor(Protocol):
    def audit(self, evidence: DeliveryAuditEvidence) -> DeliveryAudit: ...


class Coordinator:
    """Explicit generate/validate/send/commit/advance state machine."""

    def __init__(
        self,
        *,
        mm: _MM,
        validator: object,
        timeline_factory: Callable[[object], TargetTimeline],
        run: object,
        publisher: object,
        stream: object,
        gate: object,
        target_motion_logfile: str | Path,
        steps_per_chunk: int,
        source_intervals: int = 10,
        session_id_factory: Callable[[], str] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time_ns: Callable[[], int] = time.time_ns,
        readiness_wait: Callable[[], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        delivery_auditor: _DeliveryAuditor | None = None,
        prepared_target_validator: Callable[[object], object] | None = None,
    ) -> None:
        if type(steps_per_chunk) is not int or steps_per_chunk <= 0:
            raise ContractError("steps_per_chunk must be a positive integer")
        if type(source_intervals) is not int or source_intervals != 10:
            raise ContractError("source_intervals must equal the registered value 10")
        if prepared_target_validator is not None and not callable(
            prepared_target_validator
        ):
            raise ContractError("prepared target validator must be callable")
        self.mm = mm
        self.validator = validator
        self._timeline_factory = timeline_factory
        self.run = run
        self.publisher = publisher
        self.stream = stream
        self.gate = gate
        self.target_motion_logfile = Path(target_motion_logfile)
        self.steps_per_chunk = steps_per_chunk
        self.source_intervals = source_intervals
        self._session_id_factory = session_id_factory or (
            lambda: f"session-{time.time_ns():020d}"
        )
        self._monotonic_ns = monotonic_ns
        self._wall_time_ns = wall_time_ns
        self._readiness_wait = readiness_wait or (lambda: time.sleep(0.001))
        self._cancelled = cancelled
        self._delivery_auditor = delivery_auditor
        self._prepared_target_validator = prepared_target_validator
        self._state = CoordinatorState.CREATED
        self._lock = threading.RLock()
        self._pending_command: CommandSample | None = None
        self._latched_command: CommandSample | None = None
        self._session_id: str | None = None
        self._candidate_number = 0
        self._accepted_chunks = 0
        self._timeline: TargetTimeline | None = None
        self._preflight: PreflightResult | None = None
        self._terminal_verdict: TerminalVerdict | None = None
        self._official_log = _OfficialTargetLog(self.target_motion_logfile)
        self._readiness_publications: list[Mapping[str, object]] = []
        self._timeline_publications: list[Mapping[str, object]] = []
        self._closed = False

    @property
    def state(self) -> CoordinatorState:
        with self._lock:
            return self._state

    @property
    def timeline(self) -> TargetTimeline | None:
        return self._timeline

    @property
    def terminal_verdict(self) -> TerminalVerdict | None:
        return self._terminal_verdict

    def _require_state(self, *allowed: CoordinatorState) -> None:
        if self._state not in allowed:
            expected = ", ".join(state.value for state in allowed)
            raise StateTransitionError(
                f"operation requires {expected}; coordinator is {self._state.value}"
            )

    def queue_command(self, command: CommandSample) -> None:
        if not isinstance(command, CommandSample):
            raise ContractError("queued command must be an immutable CommandSample")
        with self._lock:
            if self._state is CoordinatorState.TERMINAL:
                raise StateTransitionError("coordinator is terminal")
            self._pending_command = command

    def _new_session_id(self) -> str:
        value = self._session_id_factory()
        if type(value) is not str or not value or "\x00" in value:
            raise ContractError("session ID factory returned an invalid ID")
        return value

    def _failure_site(self, error: BaseException, default: str) -> str:
        value = getattr(error, "failure_site", default)
        return value if type(value) is str and value else default

    def _record_publication(
        self,
        result: object,
        *,
        phase: str,
    ) -> Mapping[str, object]:
        if not isinstance(result, Mapping):
            raise ContractError("local publication result must be a mapping")
        copied = dict(result)
        if copied.get("local_send_completed") is not True:
            raise ContractError("local publication did not confirm send completion")
        if "phase" in copied and copied["phase"] != phase:
            raise ContractError("local publication phase does not match the request")
        frozen = MappingProxyType(copied)
        if phase == "readiness":
            self._readiness_publications.append(frozen)
        elif phase == "timeline":
            self._timeline_publications.append(frozen)
        else:
            raise ContractError("local publication phase is invalid")
        return frozen

    def preflight(self, config: SessionConfig) -> PreflightResult:
        if not isinstance(config, SessionConfig):
            raise ContractError("preflight requires an immutable SessionConfig")
        with self._lock:
            self._require_state(CoordinatorState.CREATED)
            self._state = CoordinatorState.PREFLIGHT
        session_id: str | None = None
        try:
            session_id = self._new_session_id()
            with self._lock:
                self._session_id = session_id
            log_start = self._official_log.anchor()
            self.mm.require_alive()
            raw_initial = self.mm.reset(config, session_id=session_id)
            initial = self.validator.validate_initial(raw_initial)
            if getattr(initial, "session_id", None) != session_id:
                error = ContractError(
                    "initial_session does not match the coordinator-owned session"
                )
                error.failure_site = "initial_session"
                raise error
            timeline = self._timeline_factory(initial)
            if not hasattr(timeline, "initial_buffer"):
                raise ContractError("timeline factory did not return an initial buffer")
            initial_buffer = timeline.initial_buffer
            expected = expected_official_target_row(initial_buffer)
            self.stream.start()
            attempts = 0
            while True:
                if self._cancelled is not None and bool(self._cancelled()):
                    raise RuntimeError("operator cancelled frame-zero readiness")
                self.mm.require_alive()
                self.stream.require_alive()
                attempts += 1
                prepared = self.publisher.prepare(
                    initial_buffer,
                    phase="readiness",
                    attempt=attempts,
                )
                sent = self.publisher.send_prepared(prepared)
                self._record_publication(sent, phase="readiness")
                observed = self._official_log.read_from(log_start)
                rows, _partial = _complete_rows(observed)
                if expected in rows:
                    # Stop GEAR first, then snapshot the stable EOF.  All rows
                    # emitted by any readiness send remain before this offset.
                    self.gate.pause()
                    self.gate.require_paused()
                    stable = self._official_log.read_from(log_start)
                    stable_rows, residual = _complete_rows(stable)
                    if residual:
                        raise ContractError(
                            "official target log ended with a partial readiness row"
                        )
                    if expected not in stable_rows:
                        raise ContractError(
                            "official frame-zero row disappeared during pause"
                        )
                    scoring_offset = log_start + len(stable)
                    log_device, log_inode = self._official_log.identity
                    result = PreflightResult(
                        session_id=session_id,
                        readiness_log_start_offset=log_start,
                        scoring_log_offset=scoring_offset,
                        readiness_attempts=attempts,
                        official_log_path=str(self.target_motion_logfile),
                        log_device=log_device,
                        log_inode=log_inode,
                        expected_row_sha256=hashlib.sha256(expected).hexdigest(),
                    )
                    self.run.write_readiness(result)
                    with self._lock:
                        self._timeline = timeline
                        self._preflight = result
                        self._state = CoordinatorState.READY_PAUSED
                    return result
                self._readiness_wait()
        except BaseException as error:
            self._terminalize_failure(
                phase="pre_commit",
                site=self._failure_site(error, "preflight"),
                error=error,
                candidate_id=None,
            )
            raise IntegrationFailure(
                "pre_commit", self._failure_site(error, "preflight"), error
            ) from error

    def _candidate_id(self) -> str:
        assert self._session_id is not None
        return f"{self._session_id}:candidate:{self._candidate_number:06d}"

    def _write_timing(self, timing: TimingRecord) -> None:
        self.run.write_timing(timing)

    def _is_mm_alive(self) -> bool:
        try:
            self.mm.require_alive()
        except BaseException:
            return False
        return True

    def _timeline_pending(self, candidate_id: str) -> bool:
        return (
            self._timeline is not None
            and getattr(self._timeline, "pending_candidate_id", None) == candidate_id
        )

    def _reject_precommit(
        self,
        candidate_id: str,
        site: str,
        error: BaseException,
    ) -> None:
        mm_alive = self._is_mm_alive()
        abort_required = (
            mm_alive
            and getattr(self.mm, "outstanding_candidate_id", None) == candidate_id
        )
        timeline_abort_required = self._timeline_pending(candidate_id)
        rejection = RejectionRecord(
            candidate_id=candidate_id,
            failure_site=site,
            error_type=type(error).__name__,
            error_message=str(error),
            mm_alive=mm_alive,
            abort_required=abort_required,
            # This record is written before cleanup, so it cannot claim that a
            # later command has already succeeded.
            abort_sent=False,
            timeline_abort_required=timeline_abort_required,
        )
        errors: list[BaseException] = []
        try:
            # Evidence of the decision precedes either state-discard command.
            self.run.write_rejection(rejection)
        except BaseException as failure:
            errors.append(failure)
        timeline_aborted = False
        if timeline_abort_required:
            try:
                assert self._timeline is not None
                self._timeline.abort(candidate_id)
                timeline_aborted = True
            except BaseException as failure:
                errors.append(failure)
        mm_abort_sent = False
        if abort_required:
            try:
                self.mm.abort(candidate_id)
                mm_abort_sent = True
            except BaseException as failure:
                errors.append(failure)
        decision = AbortDecisionRecord(
            candidate_id=candidate_id,
            mm_abort_required=abort_required,
            mm_abort_sent=mm_abort_sent,
            timeline_abort_required=timeline_abort_required,
            timeline_aborted=timeline_aborted,
            errors=tuple(
                f"{type(failure).__name__}: {failure}" for failure in errors
            ),
        )
        try:
            self.run.write_abort_decision(decision)
        except BaseException as failure:
            errors.append(failure)
        if errors:
            raise errors[0]

    def _paused_where_possible(self) -> bool:
        paused = bool(getattr(self.gate, "is_paused", False))
        if paused:
            return True
        try:
            self.gate.pause()
        except BaseException:
            return bool(getattr(self.gate, "is_paused", False))
        return bool(getattr(self.gate, "is_paused", False))

    def _terminalize_failure(
        self,
        *,
        phase: str,
        site: str,
        error: BaseException,
        candidate_id: str | None,
    ) -> TerminalVerdict:
        paused = self._paused_where_possible()
        with self._lock:
            self._state = CoordinatorState.TERMINAL
            self._pending_command = None
            self._latched_command = None
        verdict = TerminalVerdict(
            status="failed",
            failure_phase=phase,
            failure_site=site,
            error_type=type(error).__name__,
            error_message=str(error),
            accepted_chunks=self._accepted_chunks,
            simulation_paused=paused,
            delivery_audit_required=True,
            delivery_audit_completed=False,
            delivery_audit=None,
        )
        with self._lock:
            if self._terminal_verdict is not None:
                return self._terminal_verdict
            # Reserve the original immutable failure before the one fallible
            # durable write.  Cleanup must never retry with a different cause.
            self._terminal_verdict = verdict
        try:
            self.run.write_terminal_verdict(verdict)
        except BaseException as evidence_error:
            evidence_site = self._failure_site(
                evidence_error, "artifact_terminal"
            )
            raise IntegrationFailure(
                phase, evidence_site, evidence_error
            ) from evidence_error
        return verdict

    def _fail_chunk(
        self,
        *,
        phase: str,
        site: str,
        error: BaseException,
        candidate_id: str,
        tracker: _TimingTracker,
    ) -> IntegrationFailure:
        evidence_error: BaseException | None = None
        if phase == "pre_commit":
            try:
                self._reject_precommit(candidate_id, site, error)
            except BaseException as failure:
                evidence_error = failure
        try:
            self._write_timing(tracker.finish(site))
        except BaseException as failure:
            if evidence_error is None:
                evidence_error = failure
        terminal_error = error if evidence_error is None else evidence_error
        self._terminalize_failure(
            phase=phase,
            site=site,
            error=terminal_error,
            candidate_id=candidate_id,
        )
        failure = IntegrationFailure(phase, site, terminal_error)
        if evidence_error is not None:
            failure.__context__ = error
        return failure

    def _supersede_chunk(
        self,
        *,
        candidate_id: str,
        latched: CommandSample,
        tracker: _TimingTracker,
    ) -> CandidateSuperseded:
        """Discard a stale pre-publication candidate without terminalizing.

        Reuses the existing pre-commit cleanup to abort the pending timeline
        candidate and the outstanding MM candidate, records the rejection and
        timing, restores READY_PAUSED, clears the latched command, and releases
        no physics.  Any cleanup/evidence error remains a normal terminal
        IntegrationFailure with physics paused.
        """

        outcome = CandidateSuperseded(candidate_id, latched)
        evidence_error: BaseException | None = None
        try:
            self._reject_precommit(candidate_id, "command_superseded", outcome)
        except BaseException as failure:
            evidence_error = failure
        try:
            self._write_timing(tracker.finish("command_superseded"))
        except BaseException as failure:
            if evidence_error is None:
                evidence_error = failure
        if evidence_error is not None:
            self._terminalize_failure(
                phase="pre_commit",
                site="command_superseded",
                error=evidence_error,
                candidate_id=candidate_id,
            )
            raise IntegrationFailure(
                "pre_commit", "command_superseded", evidence_error
            ) from evidence_error
        with self._lock:
            self._latched_command = None
            self._state = CoordinatorState.READY_PAUSED
        return outcome

    def run_one_chunk(
        self,
        command: CommandSample | None = None,
        *,
        command_is_current: Callable[[CommandSample], bool] | None = None,
    ) -> AcceptedChunk:
        if command_is_current is not None and not callable(command_is_current):
            raise ContractError("command_is_current must be callable")
        with self._lock:
            self._require_state(CoordinatorState.READY_PAUSED)
            if command is not None:
                if not isinstance(command, CommandSample):
                    raise ContractError(
                        "chunk command must be an immutable CommandSample"
                    )
                self._pending_command = command
            if self._pending_command is None:
                raise StateTransitionError(
                    "ready_paused coordinator requires a queued command"
                )
            latched = self._pending_command
            if latched.chunk_index != self._accepted_chunks:
                self._pending_command = None
                raise ContractError(
                    "command chunk_index "
                    f"{latched.chunk_index} does not equal next chunk_index "
                    f"{self._accepted_chunks}"
                )
            self._pending_command = None
            self._latched_command = latched
            candidate_id = self._candidate_id()
            predecessor_id = (
                None
                if self._timeline is None
                else getattr(self._timeline, "last_accepted_candidate_id", None)
            )
            try:
                tracker = _TimingTracker(
                    candidate_id,
                    self._monotonic_ns,
                    self._wall_time_ns,
                )
            except BaseException as error:
                site = self._failure_site(error, "timing_start")
                self._terminalize_failure(
                    phase="pre_commit",
                    site=site,
                    error=error,
                    candidate_id=None,
                )
                raise IntegrationFailure("pre_commit", site, error) from error
            self._candidate_number += 1
            self._state = CoordinatorState.CANDIDATE
        try:
            self.gate.pause()
            self.gate.require_paused()
        except BaseException as error:
            site = self._failure_site(error, "process_pause")
            raise self._fail_chunk(
                phase="pre_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        try:
            raw_source = tracker.call(
                "mm_generation",
                lambda: self._generate(
                    latched, candidate_id, predecessor_id
                ),
            )
        except BaseException as error:
            site = self._failure_site(error, "generation")
            raise self._fail_chunk(
                phase="pre_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        try:
            checked_source = tracker.call(
                "projection_validation",
                lambda: self.validator.validate_source(raw_source),
            )
        except BaseException as error:
            site = self._failure_site(error, "source_validation")
            raise self._fail_chunk(
                phase="pre_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        def prepare_and_validate() -> object:
            assert self._timeline is not None
            candidate = self._timeline.prepare(checked_source)
            if self._prepared_target_validator is not None:
                try:
                    self._prepared_target_validator(candidate.target)
                except BaseException as error:
                    if not hasattr(error, "failure_site"):
                        try:
                            setattr(error, "failure_site", "kinematic_validation")
                        except BaseException:
                            pass
                    raise
            return candidate

        try:
            prepared = tracker.call("resampling", prepare_and_validate)
        except BaseException as error:
            site = self._failure_site(error, "resampling")
            raise self._fail_chunk(
                phase="pre_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        try:
            tracker.call(
                "artifact_enqueue",
                lambda: self.run.write_prepared(checked_source, prepared.target),
            )
        except BaseException as error:
            site = self._failure_site(error, "artifact_enqueue")
            raise self._fail_chunk(
                phase="pre_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error

        # Last pre-publication supersession check.  A false result aborts the
        # stale candidate via the existing precommit cleanup, releases no
        # physics, and raises the typed nonterminal CandidateSuperseded.  A
        # predicate error is a normal fail-closed pre-commit failure.
        if command_is_current is not None:
            try:
                current = command_is_current(latched)
            except BaseException as error:
                site = self._failure_site(error, "command_currency")
                raise self._fail_chunk(
                    phase="pre_commit",
                    site=site,
                    error=error,
                    candidate_id=candidate_id,
                    tracker=tracker,
                ) from error
            if not current:
                raise self._supersede_chunk(
                    candidate_id=candidate_id,
                    latched=latched,
                    tracker=tracker,
                )

        def publish() -> object:
            try:
                publication = self.publisher.prepare(
                    prepared.target.buffer,
                    phase="timeline",
                )
            except BaseException as error:
                if not hasattr(error, "failure_site"):
                    try:
                        setattr(error, "failure_site", "zmq_encoding")
                    except BaseException:
                        pass
                raise
            try:
                result = self.publisher.send_prepared(publication)
                return self._record_publication(result, phase="timeline")
            except BaseException as error:
                if not hasattr(error, "failure_site"):
                    try:
                        setattr(error, "failure_site", "zmq_send")
                    except BaseException:
                        pass
                raise

        try:
            tracker.call("publication", publish)
        except BaseException as error:
            site = self._failure_site(error, "zmq_send")
            raise self._fail_chunk(
                phase="pre_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error

        # Invoking commit is the irreversible/ambiguous boundary.  No abort or
        # timeline rollback is attempted at this point or later.
        with self._lock:
            self._state = CoordinatorState.COMMITTED
        try:
            self.mm.commit(candidate_id)
        except BaseException as error:
            site = self._failure_site(error, "mm_commit")
            raise self._fail_chunk(
                phase="post_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        try:
            accepted_target = self._timeline.commit(prepared)
        except BaseException as error:
            site = self._failure_site(error, "timeline_commit")
            raise self._fail_chunk(
                phase="post_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        with self._lock:
            self._state = CoordinatorState.ADVANCING
        try:
            advance = tracker.call(
                "simulation_advance",
                lambda: self.gate.release_steps(self.steps_per_chunk),
            )
            self.gate.require_paused()
        except BaseException as error:
            site = self._failure_site(error, "simulator_advance")
            raise self._fail_chunk(
                phase="post_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        # MM, timeline, and simulation have all advanced irreversibly.  Count
        # the acceptance before assembly or append-only evidence writes.
        with self._lock:
            self._accepted_chunks += 1
        try:
            timing = tracker.finish(None)
            accepted = AcceptedChunk(
                target=accepted_target,
                advance=advance,
                timings_ns={
                    key: value
                    for key, value in timing.durations_ns.items()
                    if value is not None
                },
                wall_started_ns=timing.wall_started_ns,
                wall_finished_ns=timing.wall_finished_ns,
            )
        except BaseException as error:
            site = self._failure_site(error, "acceptance_assembly")
            raise self._fail_chunk(
                phase="post_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
                tracker=tracker,
            ) from error
        # Attempt both writes independently so one evidence failure cannot
        # suppress the other record.
        evidence_failures: list[tuple[str, BaseException]] = []
        try:
            self.run.write_accepted(accepted)
        except BaseException as error:
            evidence_failures.append(
                (self._failure_site(error, "artifact_accept"), error)
            )
        try:
            self._write_timing(timing)
        except BaseException as error:
            evidence_failures.append(
                (self._failure_site(error, "artifact_timing"), error)
            )
        if evidence_failures:
            site, error = evidence_failures[0]
            self._terminalize_failure(
                phase="post_commit",
                site=site,
                error=error,
                candidate_id=candidate_id,
            )
            raise IntegrationFailure("post_commit", site, error) from error
        with self._lock:
            self._latched_command = None
            self._state = CoordinatorState.READY_PAUSED
        return accepted

    def _generate(
        self,
        command: CommandSample,
        candidate_id: str,
        predecessor_id: str | None,
    ) -> object:
        self.mm.require_alive()
        assert self._session_id is not None
        return self.mm.generate(
            command,
            session_id=self._session_id,
            candidate_id=candidate_id,
            predecessor_id=predecessor_id,
            source_intervals=self.source_intervals,
        )

    def finish(self) -> TerminalVerdict:
        with self._lock:
            return self._finish_locked()

    def _finish_locked(self) -> TerminalVerdict:
        """Finalize while excluding successor work and late operator input."""

        self._require_state(CoordinatorState.READY_PAUSED)
        if self._delivery_auditor is None:
            raise ContractError(
                "finish requires an explicit production delivery auditor"
            )
        if self._timeline is None:
            raise ContractError("finish requires an initialized canonical timeline")
        if self._preflight is None or self._session_id is None:
            raise ContractError("finish requires durable readiness evidence")
        paused = self._paused_where_possible()
        try:
            evidence = DeliveryAuditEvidence(
                session_id=self._session_id,
                canonical_buffer=self._timeline.canonical_buffer,
                readiness=self._preflight,
                official_log_slice=self._official_log.read_from(
                    self._preflight.scoring_log_offset
                ),
                readiness_publications=tuple(self._readiness_publications),
                timeline_publications=tuple(self._timeline_publications),
                run=self.run,
            )
            delivery_audit = self._delivery_auditor.audit(evidence)
            if not isinstance(delivery_audit, DeliveryAudit):
                raise ContractError("delivery auditor returned an invalid result")
            canonical_rows = self._timeline.canonical_buffer.count
            if delivery_audit.expected_rows != canonical_rows:
                raise ContractError(
                    "delivery audit expected_rows must equal the canonical timeline "
                    f"row count {canonical_rows}"
                )
        except BaseException as error:
            site = self._failure_site(error, "delivery_audit")
            self._terminalize_failure(
                phase="delivery_audit",
                site=site,
                error=error,
                candidate_id=None,
            )
            raise IntegrationFailure("delivery_audit", site, error) from error
        passed = delivery_audit.passed and paused
        status = "complete" if passed else "failed"
        verdict = TerminalVerdict(
            status=status,
            failure_phase=None if passed else "delivery_audit",
            failure_site=None if passed else "delivery_audit",
            error_type=None,
            error_message=None,
            accepted_chunks=self._accepted_chunks,
            simulation_paused=paused,
            delivery_audit_required=True,
            delivery_audit_completed=True,
            delivery_audit=delivery_audit,
        )
        # Reserve terminal state before the fallible durable verdict write.
        # The public lock remains held throughout finalization, so no queued
        # command or successor can cross the audited terminal boundary.
        self._pending_command = None
        self._latched_command = None
        self._state = CoordinatorState.TERMINAL
        try:
            self.run.write_terminal_verdict(verdict)
        except BaseException as error:
            site = self._failure_site(error, "artifact_terminal")
            self._terminal_verdict = TerminalVerdict(
                status="failed",
                failure_phase="delivery_audit",
                failure_site=site,
                error_type=type(error).__name__,
                error_message=str(error),
                accepted_chunks=self._accepted_chunks,
                simulation_paused=paused,
                delivery_audit_required=True,
                delivery_audit_completed=True,
                delivery_audit=delivery_audit,
            )
            raise IntegrationFailure("delivery_audit", site, error) from error
        self._terminal_verdict = verdict
        return verdict

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary: BaseException | None = None
        if self._terminal_verdict is None:
            try:
                self._terminalize_failure(
                    phase="cleanup",
                    site="closed",
                    error=RuntimeError("coordinator closed before delivery audit"),
                    candidate_id=None,
                )
            except BaseException as error:
                primary = error
        closed_dependencies: list[object] = []
        for dependency in (self.publisher, self.gate, self.stream, self.mm):
            if any(dependency is closed for closed in closed_dependencies):
                continue
            closed_dependencies.append(dependency)
            try:
                dependency.close()
            except BaseException as error:
                if primary is None:
                    primary = error
        try:
            self._official_log.close()
        except BaseException as error:
            if primary is None:
                primary = error
        if primary is not None:
            raise primary
