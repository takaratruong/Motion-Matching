"""Frozen Stage B transport, coverage, and flat-simulator evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from .commands import CommandSample, flat_command_script
from .joints import ContractError
from .timeline import CanonicalTargetBuffer


STAGE_B_COMMAND_COUNT = 30
STAGE_B_FRAME_COUNT = 601
STAGE_B_REFERENCE_DURATION_S = 12.0
STAGE_B_CONTROL_LEAD_ROWS = 16
STREAM_CHUNK_FRAME_COUNT = 20
STREAM_PADDING_FRAME_COUNT = 46


@dataclass(frozen=True)
class SonicTransportCanonicalization:
    buffer: CanonicalTargetBuffer
    subnormal_count: int
    maximum_subnormal_magnitude: float


def canonicalize_sonic_transport(
    canonical: CanonicalTargetBuffer,
) -> SonicTransportCanonicalization:
    """Replace float32 subnormals with +0 at the observed GEAR boundary."""

    if not isinstance(canonical, CanonicalTargetBuffer):
        raise ContractError(
            "SONIC transport canonicalization requires a canonical target buffer"
        )
    tiny = np.finfo(np.float32).tiny
    arrays: list[np.ndarray] = []
    replaced: list[np.ndarray] = []
    for value in (
        canonical.joint_position,
        canonical.joint_velocity,
        canonical.body_quat_w,
    ):
        output = np.ascontiguousarray(value, dtype="<f4").copy()
        mask = (output != 0.0) & (np.abs(output) < tiny)
        if np.any(mask):
            replaced.append(np.abs(output[mask]).astype(np.float64))
            output[mask] = np.float32(0.0)
        arrays.append(output)
    count = sum(int(values.size) for values in replaced)
    maximum = (
        max(float(np.max(values)) for values in replaced)
        if replaced
        else 0.0
    )
    return SonicTransportCanonicalization(
        buffer=CanonicalTargetBuffer(
            joint_position=arrays[0],
            joint_velocity=arrays[1],
            body_quat_w=arrays[2],
            frame_index=np.asarray(canonical.frame_index, dtype="<i8").copy(),
        ),
        subnormal_count=count,
        maximum_subnormal_magnitude=maximum,
    )


@dataclass(frozen=True)
class StreamTransportPlan:
    readiness: CanonicalTargetBuffer
    logical: tuple[CanonicalTargetBuffer, ...]
    padding: CanonicalTargetBuffer
    receipt_fence: CanonicalTargetBuffer

    @property
    def publication_count(self) -> int:
        return 1 + len(self.logical) + 2


def _transport_buffer(
    canonical: CanonicalTargetBuffer, indices: np.ndarray
) -> CanonicalTargetBuffer:
    if (
        indices.ndim != 1
        or indices.size <= 0
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
    ):
        raise ContractError("stream transport indices are invalid")
    source = np.minimum(indices.astype(np.int64, copy=False), canonical.count - 1)
    return CanonicalTargetBuffer(
        joint_position=canonical.joint_position[source],
        joint_velocity=canonical.joint_velocity[source],
        body_quat_w=canonical.body_quat_w[source],
        frame_index=np.asarray(indices, dtype="<i8"),
    )


def build_stream_transport_plan(
    canonical: CanonicalTargetBuffer,
) -> StreamTransportPlan:
    """Build frame zero, exact 20-frame chunks, and an unscored future fence."""

    if not isinstance(canonical, CanonicalTargetBuffer):
        raise ContractError("stream transport requires a canonical target buffer")
    if canonical.count <= 1 or (canonical.count - 1) % STREAM_CHUNK_FRAME_COUNT:
        raise ContractError(
            "stream transport requires frame zero plus complete 20-frame chunks"
        )
    expected = np.arange(canonical.count, dtype=np.int64)
    if not np.array_equal(canonical.frame_index, expected):
        raise ContractError("stream transport frames must be exactly 0..N-1")
    logical = tuple(
        _transport_buffer(
            canonical,
            np.arange(
                start,
                start + STREAM_CHUNK_FRAME_COUNT,
                dtype="<i8",
            ),
        )
        for start in range(1, canonical.count, STREAM_CHUNK_FRAME_COUNT)
    )
    padding_start = canonical.count
    padding_stop = padding_start + STREAM_PADDING_FRAME_COUNT
    return StreamTransportPlan(
        readiness=_transport_buffer(canonical, np.array([0], dtype="<i8")),
        logical=logical,
        padding=_transport_buffer(
            canonical,
            np.arange(padding_start, padding_stop, dtype="<i8"),
        ),
        receipt_fence=_transport_buffer(
            canonical, np.array([padding_stop], dtype="<i8")
        ),
    )


@dataclass(frozen=True)
class StageBCoverage:
    command_count: int
    frame_count: int
    reference_duration_s: float
    control_drive_steps: int
    control_drive_duration_s: float
    sim_dt_s: float
    control_lead_rows: int
    exact_command_coverage: bool
    exact_frame_coverage: bool
    exact_control_duration: bool


def validate_stage_b_coverage(
    *,
    commands: Sequence[CommandSample],
    canonical: CanonicalTargetBuffer,
    accepted_command_indices: Iterable[int],
    observed_target_rows: int,
    control_drive_steps: int,
    control_drive_duration_s: float,
    sim_dt_s: float,
    control_lead_rows: int,
) -> StageBCoverage:
    """Require the one registered flat script and every 50 Hz reference row."""

    values = tuple(commands)
    if values != flat_command_script() or len(values) != STAGE_B_COMMAND_COUNT:
        raise ContractError("Stage B command script is not the registered 30 chunks")
    if not isinstance(canonical, CanonicalTargetBuffer):
        raise ContractError("Stage B coverage requires a canonical target buffer")
    if canonical.count != STAGE_B_FRAME_COUNT or not np.array_equal(
        canonical.frame_index, np.arange(STAGE_B_FRAME_COUNT, dtype=np.int64)
    ):
        raise ContractError("Stage B target frames must be exactly 0..600")
    indices = tuple(accepted_command_indices)
    if (
        indices != tuple(range(STAGE_B_COMMAND_COUNT))
        or any(type(value) is not int for value in indices)
    ):
        raise ContractError("Stage B command coverage must be exactly 0..29")
    if type(observed_target_rows) is not int or observed_target_rows != canonical.count:
        raise ContractError("Stage B target coverage must contain exactly 601 rows")
    if type(control_drive_steps) is not int or control_drive_steps <= 0:
        raise ContractError("Stage B control-drive step count is invalid")
    if (
        type(control_drive_duration_s) not in (int, float)
        or not math.isfinite(float(control_drive_duration_s))
        or type(sim_dt_s) not in (int, float)
        or not math.isfinite(float(sim_dt_s))
        or float(sim_dt_s) <= 0.0
    ):
        raise ContractError("Stage B control-drive timing is invalid")
    duration = float(control_drive_duration_s)
    dt = float(sim_dt_s)
    if control_lead_rows != STAGE_B_CONTROL_LEAD_ROWS:
        raise ContractError("Stage B control lead must match the registry")
    expected_steps = round(STAGE_B_REFERENCE_DURATION_S / dt)
    if (
        expected_steps <= 0
        or abs(expected_steps * dt - STAGE_B_REFERENCE_DURATION_S) > 1.0e-12
        or control_drive_steps != expected_steps
        or abs(duration - STAGE_B_REFERENCE_DURATION_S) > 1.0e-9
        or abs(duration - control_drive_steps * dt) > 1.0e-9
    ):
        raise ContractError(
            "Stage B active CONTROL physics must be exactly 12.0 seconds"
        )
    return StageBCoverage(
        command_count=len(values),
        frame_count=canonical.count,
        reference_duration_s=STAGE_B_REFERENCE_DURATION_S,
        control_drive_steps=control_drive_steps,
        control_drive_duration_s=duration,
        sim_dt_s=dt,
        control_lead_rows=control_lead_rows,
        exact_command_coverage=True,
        exact_frame_coverage=True,
        exact_control_duration=True,
    )


@dataclass(frozen=True)
class FlatSimulatorSafety:
    state_rows: int
    contact_rows: int
    minimum_pelvis_local_height_m: float
    minimum_pelvis_up_dot: float
    forbidden_contact_groups: tuple[str, ...]
    swing_foot_scuff_count: int
    minimum_foot_clearance_m: float
    horizontal_path_drift_m: float
    contact_impulses_ns: tuple[float, ...]


def _read_jsonl(path: Path, label: str) -> tuple[Mapping[str, object], ...]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ContractError(f"Stage B {label} log is unavailable") from error
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError(f"Stage B {label} log is incomplete")
    output: list[Mapping[str, object]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError(f"Stage B {label} log is invalid JSONL") from error
        if type(value) is not dict:
            raise ContractError(f"Stage B {label} row must be an object")
        output.append(MappingProxyType(value))
    return tuple(output)


def _finite_vector(value: object, width: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(f"Stage B {label} must be numeric") from error
    if vector.shape != (width,) or not np.all(np.isfinite(vector)):
        raise ContractError(f"Stage B {label} must be finite width {width}")
    return vector


def _ordered_rows(
    rows: tuple[Mapping[str, object], ...], label: str
) -> None:
    previous_step = 0
    previous_time = -math.inf
    for row in rows:
        step = row.get("step")
        sim_time = row.get("sim_time_s")
        if (
            type(step) is not int
            or step <= previous_step
            or type(sim_time) not in (int, float)
            or not math.isfinite(float(sim_time))
            or float(sim_time) <= previous_time
        ):
            raise ContractError(f"Stage B {label} rows are not strictly ordered")
        previous_step = step
        previous_time = float(sim_time)


def load_flat_simulator_safety(
    log_dir: str | Path,
    *,
    terrain_geoms: Iterable[int],
    allowed_foot_geoms: Iterable[int],
    forbidden_geom_groups: Mapping[str, Iterable[int]],
    sim_dt_s: float,
    expected_final_pelvis_xy: Sequence[float],
    expected_simulator_steps: int,
    expected_state_rows: int,
    expected_contact_rows: int,
) -> FlatSimulatorSafety:
    """Read scored flat MuJoCo logs using registered geom IDs as authority."""

    if type(sim_dt_s) not in (int, float) or not math.isfinite(float(sim_dt_s)):
        raise ContractError("Stage B simulator dt must be finite")
    dt = float(sim_dt_s)
    if dt <= 0.0:
        raise ContractError("Stage B simulator dt must be positive")
    if any(
        type(value) is not int or value <= 0
        for value in (
            expected_simulator_steps,
            expected_state_rows,
            expected_contact_rows,
        )
    ):
        raise ContractError("Stage B simulator log counters are invalid")
    state_stride_value = 0.02 / dt
    state_stride = round(state_stride_value)
    if (
        state_stride <= 0
        or abs(state_stride_value - state_stride) > 1.0e-12
        or expected_contact_rows != expected_simulator_steps
        or expected_state_rows != expected_simulator_steps // state_stride
    ):
        raise ContractError("Stage B simulator log counters are inconsistent")
    expected_xy = _finite_vector(
        expected_final_pelvis_xy, 2, "expected final pelvis xy"
    )
    terrain = tuple(terrain_geoms)
    if not terrain or any(
        type(value) is not int or value < 0 for value in terrain
    ):
        raise ContractError("Stage B terrain geom IDs are invalid")
    allowed = tuple(allowed_foot_geoms)
    if any(type(value) is not int or value < 0 for value in allowed):
        raise ContractError("Stage B allowed foot geom IDs are invalid")
    groups = {
        name: tuple(values) for name, values in dict(forbidden_geom_groups).items()
    }
    if set(groups) != {"pelvis", "knees", "torso", "hands"} or any(
        type(name) is not str
        or any(type(value) is not int or value < 0 for value in values)
        for name, values in groups.items()
    ):
        raise ContractError("Stage B forbidden geom registry is invalid")
    root = Path(log_dir)
    state_rows = _read_jsonl(root / "state.jsonl", "state")
    contact_rows = _read_jsonl(root / "contacts.jsonl", "contact")
    _ordered_rows(state_rows, "state")
    _ordered_rows(contact_rows, "contact")
    if (
        len(state_rows) != expected_state_rows
        or len(contact_rows) != expected_contact_rows
    ):
        raise ContractError("Stage B simulator log coverage is incomplete")
    for index, row in enumerate(contact_rows, start=1):
        if (
            row.get("step") != index
            or abs(float(row["sim_time_s"]) - index * dt) > 1.0e-9
        ):
            raise ContractError("Stage B contact log coverage is not exact")
    for index, row in enumerate(state_rows, start=1):
        expected_step = index * state_stride
        if (
            row.get("step") != expected_step
            or abs(float(row["sim_time_s"]) - expected_step * dt) > 1.0e-9
        ):
            raise ContractError("Stage B state log coverage is not exact")

    heights: list[float] = []
    up_dots: list[float] = []
    pelvis_positions: list[np.ndarray] = []
    for row in state_rows:
        state = row.get("state")
        if type(state) is not dict:
            raise ContractError("Stage B state payload is invalid")
        position = _finite_vector(
            state.get("pelvis_position_m"), 3, "pelvis position"
        )
        quaternion = _finite_vector(
            state.get("pelvis_quaternion_wxyz"), 4, "pelvis quaternion"
        )
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1.0e-6:
            raise ContractError("Stage B pelvis quaternion must be unit length")
        up = _finite_vector(state.get("pelvis_up"), 3, "pelvis up vector")
        if abs(float(np.linalg.norm(up)) - 1.0) > 1.0e-6:
            raise ContractError("Stage B pelvis up vector must be unit length")
        heights.append(float(position[2]))
        up_dots.append(float(up[2]))
        pelvis_positions.append(position)

    forbidden: set[str] = set()
    impulses: list[float] = []
    foot_clearances: list[float] = []
    scuffs = 0
    allowed_set = set(allowed)
    terrain_set = set(terrain)
    group_sets = {name: set(values) for name, values in groups.items()}
    if terrain_set & (
        allowed_set | set().union(*group_sets.values())
    ):
        raise ContractError("Stage B terrain and robot geom IDs overlap")
    for row in contact_rows:
        contacts = row.get("contacts")
        if type(contacts) is not list:
            raise ContractError("Stage B contact payload is invalid")
        for contact in contacts:
            if type(contact) is not dict:
                raise ContractError("Stage B contact entry is invalid")
            first = contact.get("geom1_id")
            second = contact.get("geom2_id")
            if (
                type(first) is not int
                or first < 0
                or type(second) is not int
                or second < 0
            ):
                raise ContractError("Stage B contact geom IDs are invalid")
            distance = contact.get("distance_m")
            if type(distance) not in (int, float) or not math.isfinite(float(distance)):
                raise ContractError("Stage B contact distance is invalid")
            distance_value = float(distance)
            force = _finite_vector(contact.get("force"), 6, "contact force")
            pair = {first, second}
            if not pair & terrain_set:
                continue
            impulses.append(float(np.linalg.norm(force[:3])) * dt)
            if pair & allowed_set:
                foot_clearances.append(distance_value)
                if distance_value < 0.0:
                    scuffs += 1
            for name, ids in group_sets.items():
                if pair & ids:
                    forbidden.add(name)

    minimum_clearance = min(foot_clearances) if foot_clearances else 0.0
    horizontal_drift = float(
        np.linalg.norm(pelvis_positions[-1][:2] - expected_xy)
    )
    return FlatSimulatorSafety(
        state_rows=len(state_rows),
        contact_rows=len(contact_rows),
        minimum_pelvis_local_height_m=min(heights),
        minimum_pelvis_up_dot=min(up_dots),
        forbidden_contact_groups=tuple(
            name for name in ("pelvis", "knees", "torso", "hands") if name in forbidden
        ),
        swing_foot_scuff_count=scuffs,
        minimum_foot_clearance_m=minimum_clearance,
        horizontal_path_drift_m=horizontal_drift,
        contact_impulses_ns=tuple(impulses),
    )
