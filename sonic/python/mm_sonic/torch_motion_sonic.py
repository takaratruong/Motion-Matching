"""Dense transactional bridge from the Torch matcher to released SONIC."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from .commands import CommandSample
from .coordinator import CandidateSuperseded
from .gear_action import isaaclab_to_mujoco_joint_vector
from .joints import ContractError
from .timeline import CanonicalTargetBuffer
from .torch_motion_matcher import (
    MotionMatchResult,
    PreparedMotionMatch,
)


def _cpu_f32(tensor: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        source = tensor.detach().to(device="cpu").numpy()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError) as error:
        raise ContractError(f"{label} must be a Torch tensor") from error
    if source.shape != shape:
        raise ContractError(f"{label} must have shape {shape}")
    output = np.ascontiguousarray(source, dtype="<f4")
    if not np.all(np.isfinite(output)):
        raise ContractError(f"{label} must contain finite values")
    return output


def canonical_target_buffer_from_match(
    result: MotionMatchResult, *, global_frame_start: int
) -> CanonicalTargetBuffer:
    """Copy one exact dense 46-row matcher window into SONIC's wire schema."""
    if not isinstance(result, MotionMatchResult):
        raise ContractError("dense SONIC publication requires MotionMatchResult")
    if type(global_frame_start) is not int or global_frame_start < 0:
        raise ContractError("global_frame_start must be a nonnegative integer")
    joint_position = _cpu_f32(
        result.dense_joint_position_window, (46, 29), "dense joint position"
    )
    joint_velocity = _cpu_f32(
        result.dense_joint_velocity_window, (46, 29), "dense joint velocity"
    )
    root_quaternion = _cpu_f32(
        result.dense_root_orientation_window_wxyz,
        (46, 4),
        "dense root quaternion",
    )
    sampled_indices = np.arange(0, 46, 5)
    sampled_position = _cpu_f32(
        result.joint_position_window, (10, 29), "sampled joint position"
    )
    sampled_velocity = _cpu_f32(
        result.joint_velocity_window, (10, 29), "sampled joint velocity"
    )
    sampled_quaternion = _cpu_f32(
        result.root_orientation_window_wxyz,
        (10, 4),
        "sampled root quaternion",
    )
    if not (
        np.array_equal(sampled_position, joint_position[sampled_indices])
        and np.array_equal(sampled_velocity, joint_velocity[sampled_indices])
        and np.array_equal(sampled_quaternion, root_quaternion[sampled_indices])
    ):
        raise ContractError("sampled matcher rows do not equal dense offsets 0..45")
    return CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=root_quaternion,
        frame_index=np.arange(
            global_frame_start, global_frame_start + 46, dtype="<i8"
        ),
    )


def initial_qpos_from_match(
    result: MotionMatchResult, *, base_qpos: object | None = None
) -> np.ndarray:
    """Construct simulator qpos while preserving optional hand state."""
    if not isinstance(result, MotionMatchResult):
        raise ContractError("initial qpos requires MotionMatchResult")
    if base_qpos is None:
        qpos = np.zeros(36, dtype=np.float64)
    else:
        qpos = np.asarray(base_qpos, dtype=np.float64).copy()
        if qpos.ndim != 1 or qpos.shape[0] < 36 or not np.all(np.isfinite(qpos)):
            raise ContractError("base_qpos must contain at least 36 finite values")
    root_position = _cpu_f32(
        result.root_position_world, (3,), "initial root position"
    )
    root_quaternion = _cpu_f32(
        result.root_orientation_world_wxyz, (4,), "initial root quaternion"
    )
    joint_position = _cpu_f32(
        result.joint_position, (29,), "initial joint position"
    )
    qpos[:3] = root_position
    qpos[3:7] = root_quaternion
    qpos[7:36] = isaaclab_to_mujoco_joint_vector(joint_position)
    return qpos


@dataclass(frozen=True)
class PreparedSonicReference:
    buffer: CanonicalTargetBuffer
    result_sequence: int
    global_frame_start: int
    global_frame_end: int
    publisher_prepared: object
    _adapter_token: object
    _base_acked_sequence: int


@dataclass(frozen=True)
class AcknowledgedSonicReference:
    publication: PreparedSonicReference
    prepared_release: object
    send_result: Mapping[str, object]


class SonicReferenceAdapter:
    def __init__(
        self,
        publisher: object,
        gate: object,
        *,
        initial_acked_sequence: int = 0,
        initial_global_frame_end: int = 45,
    ) -> None:
        if not hasattr(publisher, "prepare") or not hasattr(
            publisher, "send_prepared"
        ):
            raise ContractError("adapter publisher lacks transactional methods")
        if not hasattr(gate, "prepare_release"):
            raise ContractError("adapter gate lacks prepare_release")
        if type(initial_acked_sequence) is not int or initial_acked_sequence < -1:
            raise ContractError("initial_acked_sequence is invalid")
        if type(initial_global_frame_end) is not int or initial_global_frame_end < -1:
            raise ContractError("initial_global_frame_end is invalid")
        self.publisher = publisher
        self.gate = gate
        self._token = object()
        self._acked_sequence = initial_acked_sequence
        self._global_frame_end = initial_global_frame_end
        self._prepared: dict[int, PreparedSonicReference] = {}
        self._sent: set[int] = set()

    def prepare(
        self, result: MotionMatchResult, *, global_frame_start: int
    ) -> PreparedSonicReference:
        expected_sequence = self._acked_sequence + 1
        if result.diagnostics.sequence != expected_sequence:
            raise ContractError(
                f"result sequence {result.diagnostics.sequence} "
                f"!= next acknowledged sequence {expected_sequence}"
            )
        if global_frame_start != self._global_frame_end - 44:
            raise ContractError("global frame window must advance by exactly one")
        buffer = canonical_target_buffer_from_match(
            result, global_frame_start=global_frame_start
        )
        publisher_prepared = self.publisher.prepare(buffer, phase="timeline")
        prepared = PreparedSonicReference(
            buffer=buffer,
            result_sequence=result.diagnostics.sequence,
            global_frame_start=global_frame_start,
            global_frame_end=global_frame_start + 45,
            publisher_prepared=publisher_prepared,
            _adapter_token=self._token,
            _base_acked_sequence=self._acked_sequence,
        )
        self._prepared[prepared.result_sequence] = prepared
        return prepared

    def send_and_wait(
        self, prepared: PreparedSonicReference
    ) -> AcknowledgedSonicReference:
        if (
            not isinstance(prepared, PreparedSonicReference)
            or prepared._adapter_token is not self._token
            or prepared._base_acked_sequence != self._acked_sequence
            or self._prepared.get(prepared.result_sequence) is not prepared
            or prepared.result_sequence in self._sent
        ):
            raise ContractError("SONIC reference is foreign, stale, or already sent")
        send_result = self.publisher.send_prepared(prepared.publisher_prepared)
        self._sent.add(prepared.result_sequence)
        release = self.gate.prepare_release(
            expected_stream_frame_end=prepared.global_frame_end
        )
        self._acked_sequence = prepared.result_sequence
        self._global_frame_end = prepared.global_frame_end
        return AcknowledgedSonicReference(
            publication=prepared,
            prepared_release=release,
            send_result=MappingProxyType(dict(send_result)),
        )


@dataclass(frozen=True)
class TorchCommitRecord:
    command_index: int
    global_frame_start: int
    global_frame_end: int
    matcher_sequence: int
    selected_clip_path: str
    selected_frame: int
    root_position_world: tuple[float, float, float]
    root_orientation_world_wxyz: tuple[float, float, float, float]
    matcher_started_ns: int
    publication_acked_ns: int
    committed_ns: int
    simulation_advance_completed_ns: int
    advance: object


def _heading_yaw(quaternion: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class TorchMotionCommitter:
    def __init__(
        self,
        *,
        matcher: object,
        adapter: SonicReferenceAdapter,
        gate: object,
        recorder: object,
        steps_per_policy: int,
        next_command_index: int = 0,
        next_global_frame: int = 1,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(steps_per_policy) is not int or steps_per_policy <= 0:
            raise ContractError("steps_per_policy must be positive")
        self.matcher = matcher
        self.adapter = adapter
        self.gate = gate
        self.recorder = recorder
        self.steps_per_policy = steps_per_policy
        self.next_command_index = next_command_index
        self.next_global_frame = next_global_frame
        self.monotonic_ns = monotonic_ns

    @property
    def next_chunk(self) -> int:
        return self.next_command_index

    def run_one_step(
        self,
        command: CommandSample,
        *,
        command_is_current: Callable[[CommandSample], bool] | None = None,
    ) -> TorchCommitRecord:
        if command.chunk_index != self.next_command_index:
            raise ContractError("command index does not match Torch committer")
        started = self.monotonic_ns()
        prepared: PreparedMotionMatch = self.matcher.prepare_step(
            (
                command.requested_velocity_mujoco[0],
                command.requested_velocity_mujoco[1],
            ),
            _heading_yaw(command.desired_heading_mujoco_wxyz),
        )
        if command_is_current is not None and not command_is_current(command):
            raise CandidateSuperseded(
                f"torch:{command.chunk_index:06d}", command
            )
        publication = self.adapter.prepare(
            prepared.result, global_frame_start=self.next_global_frame
        )
        acknowledged = self.adapter.send_and_wait(publication)
        acked = self.monotonic_ns()
        result = self.matcher.commit(prepared)
        committed = self.monotonic_ns()
        advance = self.gate.commit_release(
            acknowledged.prepared_release, self.steps_per_policy
        )
        advanced = self.monotonic_ns()
        self.recorder.record(command)
        root_p = tuple(
            float(v)
            for v in result.root_position_world.detach().cpu().tolist()
        )
        root_q = tuple(
            float(v)
            for v in result.root_orientation_world_wxyz.detach().cpu().tolist()
        )
        record = TorchCommitRecord(
            command_index=self.next_command_index,
            global_frame_start=self.next_global_frame,
            global_frame_end=self.next_global_frame + 45,
            matcher_sequence=result.diagnostics.sequence,
            selected_clip_path=result.diagnostics.selected_clip_path,
            selected_frame=result.diagnostics.selected_frame,
            root_position_world=root_p,
            root_orientation_world_wxyz=root_q,
            matcher_started_ns=started,
            publication_acked_ns=acked,
            committed_ns=committed,
            simulation_advance_completed_ns=advanced,
            advance=advance,
        )
        self.next_command_index += 1
        self.next_global_frame += 1
        return record

    def run_one_chunk(
        self,
        command: CommandSample,
        *,
        command_is_current: Callable[[CommandSample], bool] | None = None,
    ) -> TorchCommitRecord:
        return self.run_one_step(
            command, command_is_current=command_is_current
        )


__all__ = [
    "AcknowledgedSonicReference",
    "PreparedSonicReference",
    "SonicReferenceAdapter",
    "TorchCommitRecord",
    "TorchMotionCommitter",
    "canonical_target_buffer_from_match",
    "initial_qpos_from_match",
]
