"""Raw 30 Hz closed-loop terrain-PFNN runtime and acceptance metrics.

The runtime is deliberately transactional: a tick either commits the complete
``t -> t+1`` prediction, or retains every recurrent state value from the last
finite frame.  It contains no IK, foot locking, portal, root projection, or
teacher-forced correction path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)

from .dataset import pfnn_input_sha256
from .kinematics import TorchG1ForwardKinematics, root_tilt_quaternion_wxyz
from .layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from .recurrence import (
    PlannedTrajectory,
    _integrate_root_motion,
    advance_recurrent_state,
    initialize_recurrent_state,
    pack_recurrent_input,
    plan_recurrent_trajectory,
)
from .training import LoadedCheckpoint, load_checkpoint


FPS = 30.0
DT = 1.0 / FPS
TWO_PI = 2.0 * math.pi
TERRAIN_HALF_WIDTH_M = 0.25
MAXIMUM_GRADE_DEGREES = 20.0
MAXIMUM_ROOT_TRANSLATION_STEP_M = 0.060
MAXIMUM_ROOT_ROTATION_STEP_RAD = 0.35
MAXIMUM_JOINT_STEP_RAD = 0.25
MAXIMUM_SOLE_PENETRATION_M = 0.015
MAXIMUM_STANCE_SOLE_SPEED_M_S = 0.10
FUTURE_INDICES = tuple(range(6, 12))
SEGMENT_NAMES = ("flat", "ascent", "summit", "descent", "landing")


TerrainCallback = Callable[[np.ndarray], "TerrainSample | None"]


def _canonical_receipt_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _owned_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != shape or source.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be a real array with shape {shape}")
    output = np.ascontiguousarray(source, dtype=np.float64).copy()
    if not np.isfinite(output).all():
        raise ValueError(f"{label} must be finite")
    output.flags.writeable = False
    return output


def _rotation_2d(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(left, dtype=np.float64)
    bw, bx, by, bz = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float64,
    )


def _root_quaternion(yaw: float, tilt: np.ndarray) -> np.ndarray:
    yaw_quaternion = np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )
    output = _quaternion_multiply_wxyz(
        yaw_quaternion,
        root_tilt_quaternion_wxyz(float(tilt[0]), float(tilt[1])),
    )
    norm = float(np.linalg.norm(output))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("root quaternion is invalid")
    return output / norm


def _quaternion_step(left: np.ndarray, right: np.ndarray) -> float:
    dot = abs(float(np.dot(left, right)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


@dataclass(frozen=True)
class TerrainSample:
    """One world-Z terrain sample with gradient ``(dz/dx, dz/dy)``."""

    height_m: float
    gradient_xy: np.ndarray
    segment: str | None = None

    def __post_init__(self) -> None:
        height = float(self.height_m)
        gradient = _owned_array(self.gradient_xy, (2,), "gradient_xy")
        if not math.isfinite(height):
            raise ValueError("terrain height must be finite")
        if self.segment is not None and (
            type(self.segment) is not str or self.segment not in SEGMENT_NAMES
        ):
            raise ValueError("terrain segment tag is invalid")
        object.__setattr__(self, "height_m", height)
        object.__setattr__(self, "gradient_xy", gradient)

    @property
    def absolute_grade_degrees(self) -> float:
        return math.degrees(math.atan(float(np.linalg.norm(self.gradient_xy))))

    def signed_grade_degrees(self, travel_unit: np.ndarray) -> float:
        direction = np.asarray(travel_unit, dtype=np.float64)
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("travel direction must be finite with shape (2,)")
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-12:
            return 0.0
        return math.degrees(
            math.atan(float(np.dot(self.gradient_xy, direction / norm)))
        )


@dataclass
class PFNNTrajectoryState:
    position_world_xy: np.ndarray
    direction_world_xy: np.ndarray
    semantic_intent: np.ndarray

    def __post_init__(self) -> None:
        self.position_world_xy = _owned_array(
            self.position_world_xy, (12, 2), "trajectory position"
        )
        direction = _owned_array(
            self.direction_world_xy, (12, 2), "trajectory direction"
        )
        norms = np.linalg.norm(direction, axis=1)
        if np.any(norms < 1.0e-12):
            raise ValueError("trajectory directions must be nonzero")
        normalized = np.ascontiguousarray(direction / norms[:, None])
        normalized.flags.writeable = False
        self.direction_world_xy = normalized
        semantic = _owned_array(
            self.semantic_intent, (12, 2), "trajectory semantic intent"
        )
        if not np.isin(semantic, (0.0, 1.0)).all() or not np.all(
            np.sum(semantic, axis=1) == 1.0
        ):
            raise ValueError("trajectory semantic intent must be one-hot")
        self.semantic_intent = semantic


@dataclass(frozen=True)
class PFNNRuntimeFrame:
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position_isaaclab: np.ndarray
    phase: float
    contact_probability: np.ndarray
    trajectory: PFNNTrajectoryState
    supported: bool
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_position_world",
            _owned_array(self.root_position_world, (3,), "root position"),
        )
        quaternion = _owned_array(
            self.root_quaternion_world_wxyz, (4,), "root quaternion"
        )
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-6):
            raise ValueError("root quaternion must be normalized")
        object.__setattr__(self, "root_quaternion_world_wxyz", quaternion)
        object.__setattr__(
            self,
            "joint_position_isaaclab",
            _owned_array(self.joint_position_isaaclab, (29,), "joint position"),
        )
        phase = float(self.phase)
        if not math.isfinite(phase) or not 0.0 <= phase < TWO_PI:
            raise ValueError("runtime phase must be finite in [0,2*pi)")
        object.__setattr__(self, "phase", phase)
        contacts = _owned_array(
            self.contact_probability, (4,), "contact probability"
        )
        if np.any(contacts < 0.0) or np.any(contacts > 1.0):
            raise ValueError("contact probabilities must be in [0,1]")
        object.__setattr__(self, "contact_probability", contacts)
        if not isinstance(self.trajectory, PFNNTrajectoryState):
            raise TypeError("trajectory must be PFNNTrajectoryState")
        if type(self.supported) is not bool or type(self.diagnostics) is not dict:
            raise TypeError("supported and diagnostics have invalid types")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class _TerrainTrack:
    samples: tuple[tuple[TerrainSample, TerrainSample, TerrainSample], ...]

    @property
    def relative_height(self) -> np.ndarray:
        support = self.samples[6][1].height_m
        return np.asarray(
            [[sample.height_m - support for sample in row] for row in self.samples],
            dtype=np.float64,
        )


class TerrainPFNNRuntime:
    """Evaluate raw PFNN outputs in a strict native-G1 closed loop."""

    def __init__(
        self,
        *,
        checkpoint: LoadedCheckpoint | object,
        kinematics: TorchG1ForwardKinematics | object,
        height_and_grade_at: TerrainCallback,
        model: torch.nn.Module | None = None,
        device: str | torch.device = "cpu",
        enforce_motion_envelope: bool = True,
    ) -> None:
        if not callable(height_and_grade_at):
            raise TypeError("height_and_grade_at must be callable")
        if type(enforce_motion_envelope) is not bool:
            raise TypeError("enforce_motion_envelope must be bool")
        signature = getattr(kinematics, "kinematic_signature_sha256", None)
        if signature != getattr(checkpoint, "kinematic_signature_sha256", None):
            raise ValueError("checkpoint kinematic signature mismatch")
        limits = np.asarray(
            torch.as_tensor(getattr(kinematics, "joint_limits"), dtype=torch.float64)
            .detach()
            .cpu(),
            dtype=np.float64,
        )
        checkpoint_limits = np.asarray(
            torch.as_tensor(getattr(checkpoint, "joint_limits"), dtype=torch.float64)
            .detach()
            .cpu(),
            dtype=np.float64,
        )
        if (
            limits.shape != (29, 2)
            or not np.isfinite(limits).all()
            or not np.array_equal(limits, checkpoint_limits)
        ):
            raise ValueError("checkpoint canonical joint limits mismatch")
        self._limits = limits.copy()
        self._height_and_grade_at = height_and_grade_at
        self._checkpoint = checkpoint
        self._enforce_motion_envelope = enforce_motion_envelope
        self._device = torch.device(device)
        self._limits_tensor = torch.as_tensor(
            self._limits, dtype=torch.float64, device=self._device
        )
        self._model = getattr(checkpoint, "build_model")() if model is None else model
        if not isinstance(self._model, torch.nn.Module):
            raise TypeError("model must be a torch module")
        self._model.to(self._device)
        self._model.eval()
        self._normalization = {
            name: np.asarray(
                torch.as_tensor(getattr(checkpoint, "normalization")[name])
                .detach()
                .cpu(),
                dtype=np.float32,
            ).copy()
            for name in ("x_mean", "x_std", "y_mean", "y_std")
        }
        self._x_mean = torch.as_tensor(
            self._normalization["x_mean"], dtype=torch.float32, device=self._device
        )
        self._x_std = torch.as_tensor(
            self._normalization["x_std"], dtype=torch.float32, device=self._device
        )
        self._y_mean = torch.as_tensor(
            self._normalization["y_mean"], dtype=torch.float64, device=self._device
        )
        self._y_std = torch.as_tensor(
            self._normalization["y_std"], dtype=torch.float64, device=self._device
        )
        q99 = float(getattr(checkpoint, "phase_advance_q99"))
        if not math.isfinite(q99) or q99 < 0.0:
            raise ValueError("checkpoint phase_advance_q99 is invalid")
        self._phase_cap = min(math.pi, 1.5 * q99)
        seed = getattr(checkpoint, "runtime_seed", None)
        if type(seed) is not dict:
            raise ValueError("checkpoint runtime seed is invalid")

        def seed_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
            value = getattr(seed[name], "detach", lambda: seed[name])()
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            return _owned_array(value, shape, f"runtime seed {name}")

        phase = float(seed_array("phase", (),).item())
        root_height = float(seed_array("root_height", (),).item())
        tilt = seed_array("root_tilt", (2,))
        joints = seed_array("joint_position", (29,))
        if (
            not 0.0 <= phase < TWO_PI
            or root_height <= 0.0
            or np.any(joints < self._limits[:, 0])
            or np.any(joints > self._limits[:, 1])
        ):
            raise ValueError("runtime seed is outside the native runtime limits")
        root_xy = seed_array("world_xy", (2,)).copy()
        yaw = float(seed_array("world_yaw", ()).item())
        if not math.isfinite(yaw):
            raise ValueError("runtime seed world yaw must be finite")
        support = self._query(root_xy, allow_unsupported_grade=False)
        if support is None:
            raise ValueError("runtime seed origin has no supported terrain")
        rotation = _rotation_2d(yaw)
        local_position = seed_array("trajectory_position", (12, 2))
        local_direction = seed_array("trajectory_direction", (12, 2))
        semantic = seed_array("semantic_intent", (12, 2))
        self._bootstrap_expected_terrain = seed_array(
            "terrain_height", (12, 3)
        ).copy()
        self._bootstrap_expected_input = seed_array(
            "normalized_input", (INPUT_LAYOUT.size,)
        ).astype(np.float32)
        self._bootstrap_input_sha256 = seed.get("normalized_input_sha256")
        if (
            type(self._bootstrap_input_sha256) is not str
            or len(self._bootstrap_input_sha256) != 64
        ):
            raise ValueError("runtime seed normalized input digest is invalid")
        if pfnn_input_sha256(
            self._bootstrap_expected_input
        ) != self._bootstrap_input_sha256:
            raise ValueError("runtime seed normalized input digest mismatch")
        world_position = root_xy + local_position @ rotation.T
        world_direction = local_direction @ rotation.T
        trajectory = PFNNTrajectoryState(world_position, world_direction, semantic)
        contacts = seed_array("contact_label", (4,))
        self._recurrent_state = initialize_recurrent_state(
            trajectory_position_local=torch.tensor(
                local_position, dtype=torch.float32, device=self._device
            ).reshape(1, 12, 2),
            trajectory_direction_local=torch.tensor(
                local_direction, dtype=torch.float32, device=self._device
            ).reshape(1, 12, 2),
            semantic_intent=torch.tensor(
                semantic, dtype=torch.float32, device=self._device
            ).reshape(1, 12, 2),
            previous_body_position_local=torch.tensor(
                seed_array("body_position", (30, 3)),
                dtype=torch.float32,
                device=self._device,
            ).reshape(1, 30, 3),
            previous_body_velocity_local=torch.tensor(
                seed_array("body_velocity", (30, 3)),
                dtype=torch.float32,
                device=self._device,
            ).reshape(1, 30, 3),
            phase=torch.tensor((phase,), dtype=torch.float32, device=self._device),
            root_world_xy=torch.tensor(
                root_xy, dtype=torch.float32, device=self._device
            ).reshape(1, 2),
            root_yaw_world=torch.tensor(
                (yaw,), dtype=torch.float32, device=self._device
            ),
        )
        self._bootstrap_pending = True
        self._wall_tick = 0
        self._hold_count = 0
        self._frame = PFNNRuntimeFrame(
            root_position_world=np.asarray(
                (root_xy[0], root_xy[1], support.height_m + root_height),
                dtype=np.float64,
            ),
            root_quaternion_world_wxyz=_root_quaternion(yaw, tilt),
            joint_position_isaaclab=joints,
            phase=phase,
            contact_probability=contacts,
            trajectory=trajectory,
            supported=True,
            diagnostics={
                "wall_tick": 0,
                "hold_count": 0,
                "runtime_seed_provenance": dict(seed["provenance"]),
                "phase_advance": 0.0,
                "desired_speed_m_s": 0.0,
            },
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        dataset_digest: str,
        model_path: str | Path,
        height_and_grade_at: TerrainCallback,
        device: str | torch.device = "cpu",
        enforce_motion_envelope: bool = True,
    ) -> "TerrainPFNNRuntime":
        kinematics = TorchG1ForwardKinematics.from_mjcf(model_path)
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_dataset_digest=dataset_digest,
            expected_kinematic_signature_sha256=(
                kinematics.kinematic_signature_sha256
            ),
        )
        return cls(
            checkpoint=checkpoint,
            kinematics=kinematics,
            height_and_grade_at=height_and_grade_at,
            device=device,
            enforce_motion_envelope=enforce_motion_envelope,
        )

    @property
    def frame(self) -> PFNNRuntimeFrame:
        return self._frame

    @property
    def previous_body_position(self) -> np.ndarray:
        return np.asarray(
            self._recurrent_state.previous_body_position_local[0]
            .to(device="cpu", dtype=torch.float64)
            .numpy(),
            dtype=np.float64,
        ).copy()

    @property
    def previous_body_velocity(self) -> np.ndarray:
        return np.asarray(
            self._recurrent_state.previous_body_velocity_local[0]
            .to(device="cpu", dtype=torch.float64)
            .numpy(),
            dtype=np.float64,
        ).copy()

    @property
    def joint_limits(self) -> np.ndarray:
        return self._limits.copy()

    def _query(
        self, xy: np.ndarray, *, allow_unsupported_grade: bool
    ) -> TerrainSample | None:
        try:
            raw = self._height_and_grade_at(
                np.ascontiguousarray(xy, dtype=np.float64)
            )
            if raw is None:
                return None
            if not isinstance(raw, TerrainSample):
                raw = TerrainSample(
                    getattr(raw, "height_m"),
                    getattr(raw, "gradient_xy"),
                    getattr(raw, "segment", None),
                )
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            return None
        if (
            not allow_unsupported_grade
            and raw.absolute_grade_degrees > MAXIMUM_GRADE_DEGREES + 1.0e-10
        ):
            return None
        return raw

    @staticmethod
    def _track_points(position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        return np.stack(
            (
                position + TERRAIN_HALF_WIDTH_M * normal,
                position,
                position - TERRAIN_HALF_WIDTH_M * normal,
            )
        )

    def _query_row(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        *,
        allow_unsupported_grade: bool,
    ) -> tuple[TerrainSample, TerrainSample, TerrainSample] | None:
        values = tuple(
            self._query(point, allow_unsupported_grade=allow_unsupported_grade)
            for point in self._track_points(position, direction)
        )
        if any(value is None for value in values):
            return None
        return values  # type: ignore[return-value]

    def _replan_and_sample(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        semantic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, _TerrainTrack, bool] | None:
        first_unsupported: int | None = None
        for index in FUTURE_INDICES:
            row = self._query_row(
                position[index], direction[index], allow_unsupported_grade=False
            )
            if row is None:
                first_unsupported = index
                break
        if first_unsupported == 6:
            return None
        supported = first_unsupported is None
        if first_unsupported is not None:
            last = first_unsupported - 1
            position[first_unsupported:] = position[last]
            direction[first_unsupported:] = direction[last]
            semantic[6:] = (1.0, 0.0)
        rows: list[tuple[TerrainSample, TerrainSample, TerrainSample]] = []
        for index in range(12):
            row = self._query_row(
                position[index], direction[index], allow_unsupported_grade=False
            )
            if row is None:
                return None
            rows.append(row)
        return position, direction, semantic, _TerrainTrack(tuple(rows)), supported

    def _hold(self, reason: str, **details: object) -> PFNNRuntimeFrame:
        self._hold_count += 1
        diagnostics = dict(self._frame.diagnostics)
        diagnostics.update(
            {
                "wall_tick": self._wall_tick,
                "hold_count": self._hold_count,
                "hold_reason": str(reason),
                **details,
            }
        )
        self._frame = replace(self._frame, diagnostics=diagnostics)
        return self._frame

    def step(self, command: object, camera_yaw: float) -> PFNNRuntimeFrame:
        self._wall_tick += 1
        requested = np.asarray(command, dtype=np.float64)
        if (
            requested.shape != (2,)
            or not np.isfinite(requested).all()
            or not math.isfinite(float(camera_yaw))
        ):
            return self._hold("invalid_command")
        desired_world = requested @ _rotation_2d(float(camera_yaw)).T
        requested_speed = float(np.linalg.norm(desired_world))
        bootstrap_tick = self._bootstrap_pending
        if bootstrap_tick:
            planned = PlannedTrajectory(
                position_world_xy=self._recurrent_state.predicted_position_world_xy,
                direction_world_xy=self._recurrent_state.predicted_direction_world_xy,
                semantic_intent=torch.tensor(
                    self._frame.trajectory.semantic_intent,
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ).reshape(1, 12, 2),
            )
        else:
            planned = plan_recurrent_trajectory(
                self._recurrent_state,
                torch.as_tensor(
                    desired_world,
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ).reshape(1, 2),
            )
        position = np.asarray(
            planned.position_world_xy[0].to(device="cpu", dtype=torch.float64),
            dtype=np.float64,
        ).copy()
        direction = np.asarray(
            planned.direction_world_xy[0].to(device="cpu", dtype=torch.float64),
            dtype=np.float64,
        ).copy()
        semantic = np.asarray(
            planned.semantic_intent[0].to(device="cpu", dtype=torch.float64),
            dtype=np.float64,
        ).copy()
        replanned = self._replan_and_sample(position, direction, semantic)
        if replanned is None:
            return self._hold("unsupported_current_terrain")
        position, direction, semantic, terrain, supported = replanned
        if bootstrap_tick and not supported:
            return self._hold("bootstrap_unsupported_terrain")
        if bootstrap_tick and not np.allclose(
            terrain.relative_height,
            self._bootstrap_expected_terrain,
            rtol=0.0,
            atol=1.0e-4,
        ):
            return self._hold("bootstrap_terrain_mismatch")
        effective_speed = requested_speed if supported else 0.0
        try:
            planned = PlannedTrajectory(
                position_world_xy=torch.as_tensor(
                    position,
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ).reshape(1, 12, 2),
                direction_world_xy=torch.as_tensor(
                    direction,
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ).reshape(1, 12, 2),
                semantic_intent=torch.as_tensor(
                    semantic,
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ).reshape(1, 12, 2),
            )
            normalized = pack_recurrent_input(
                state=self._recurrent_state,
                planned=planned,
                terrain_height=torch.as_tensor(
                    terrain.relative_height,
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ).reshape(1, 12, 3),
                x_mean=self._x_mean,
                x_std=self._x_std,
            )
            if bootstrap_tick and not np.allclose(
                np.asarray(normalized[0].to(device="cpu"), dtype=np.float32),
                self._bootstrap_expected_input,
                rtol=0.0,
                atol=3.0e-5,
            ):
                return self._hold("bootstrap_input_digest_mismatch")
            with torch.inference_mode():
                prediction = self._model(
                    normalized,
                    self._recurrent_state.phase,
                )
            if not isinstance(prediction, torch.Tensor) or prediction.shape != (1, 268):
                return self._hold("model_output_shape")
            raw_output = prediction.to(device=self._device, dtype=torch.float64)[0]
        except Exception as error:  # model failures are a rejected transaction
            return self._hold(f"model_exception:{type(error).__name__}")
        if not bool(torch.isfinite(raw_output).all()):
            return self._hold("model_output_nonfinite")
        denormalized = raw_output * self._y_std + self._y_mean
        contact_field = OUTPUT_LAYOUT["contact_logit"]
        physical = torch.cat(
            (denormalized[: contact_field.start], raw_output[contact_field]), dim=0
        )
        if not bool(torch.isfinite(physical).all()):
            return self._hold("denormalized_output_nonfinite")

        preview_envelope_violations: list[str] = []
        local_trajectory_direction = physical[
            OUTPUT_LAYOUT["trajectory_direction"]
        ].reshape(12, 2)
        direction_norm = torch.linalg.vector_norm(
            local_trajectory_direction, dim=-1
        )
        direction_invalid = bool(torch.any(direction_norm < 0.5)) or bool(
            torch.any(direction_norm > 1.5)
        )
        if direction_invalid:
            direction_index = int(
                torch.argmax(
                    torch.maximum(0.5 - direction_norm, direction_norm - 1.5)
                ).item()
            )
            if self._enforce_motion_envelope:
                return self._hold(
                    "trajectory_direction_norm",
                    rejected_trajectory_direction_index=direction_index,
                    rejected_trajectory_direction_norm=float(direction_norm[direction_index]),
                    rejected_minimum_trajectory_direction_norm=float(
                        torch.min(direction_norm)
                    ),
                    rejected_maximum_trajectory_direction_norm=float(
                        torch.max(direction_norm)
                    ),
                )
            preview_envelope_violations.append("trajectory_direction_norm")
        root_height = float(physical[OUTPUT_LAYOUT["root_height"]][0])
        tilt = physical[OUTPUT_LAYOUT["root_tilt"]]
        joints = physical[OUTPUT_LAYOUT["joint_position"]]
        raw_phase_advance = float(physical[OUTPUT_LAYOUT["phase_advance"]][0])
        if root_height <= 0.0:
            if self._enforce_motion_envelope:
                return self._hold("root_height_nonpositive")
            preview_envelope_violations.append("root_height_nonpositive")
        joint_limit_invalid = bool(torch.any(joints < self._limits_tensor[:, 0])) or bool(
            torch.any(joints > self._limits_tensor[:, 1])
        )
        if joint_limit_invalid:
            if self._enforce_motion_envelope:
                return self._hold("joint_limit")
            preview_envelope_violations.append("joint_limit")
        previous_joints = torch.tensor(
            self._frame.joint_position_isaaclab,
            dtype=torch.float64,
            device=self._device,
        )
        joint_delta = torch.abs(joints - previous_joints)
        maximum_joint_step = float(torch.max(joint_delta))
        if maximum_joint_step > MAXIMUM_JOINT_STEP_RAD + 1.0e-10:
            joint_index = int(torch.argmax(joint_delta).item())
            if self._enforce_motion_envelope:
                return self._hold(
                    "joint_step",
                    rejected_max_joint_step_rad=float(joint_delta[joint_index]),
                    rejected_joint_index=joint_index,
                    rejected_joint_name=ISAACLAB_JOINT_NAMES[joint_index],
                )
            preview_envelope_violations.append("joint_step")

        physical_recurrent = physical.to(
            dtype=self._recurrent_state.root_world_xy.dtype
        ).reshape(1, OUTPUT_LAYOUT.size)
        candidate_root_world_xy, candidate_root_yaw_world = _integrate_root_motion(
            self._recurrent_state, physical_recurrent
        )
        candidate_new_xy = np.asarray(
            candidate_root_world_xy[0]
            .to(device="cpu", dtype=torch.float64)
            .numpy(),
            dtype=np.float64,
        )
        candidate_new_yaw = float(
            candidate_root_yaw_world[0].to(device="cpu", dtype=torch.float64)
        )
        new_support = self._query(candidate_new_xy, allow_unsupported_grade=False)
        if new_support is None:
            return self._hold("unsupported_predicted_root_terrain")
        candidate_new_position = np.asarray(
            (
                candidate_new_xy[0],
                candidate_new_xy[1],
                new_support.height_m + root_height,
            ),
            dtype=np.float64,
        )
        tilt_report = np.asarray(
            tilt.to(device="cpu", dtype=torch.float64), dtype=np.float64
        )
        try:
            new_quaternion = _root_quaternion(candidate_new_yaw, tilt_report)
        except ValueError:
            return self._hold("root_quaternion")
        if not np.isclose(np.linalg.norm(new_quaternion), 1.0, atol=1.0e-6):
            return self._hold("root_quaternion")
        root_translation_step = float(
            np.linalg.norm(candidate_new_position - self._frame.root_position_world)
        )
        if root_translation_step > MAXIMUM_ROOT_TRANSLATION_STEP_M + 1.0e-10:
            if self._enforce_motion_envelope:
                return self._hold("root_translation_step")
            preview_envelope_violations.append("root_translation_step")
        root_rotation_step = _quaternion_step(
            self._frame.root_quaternion_world_wxyz, new_quaternion
        )
        if root_rotation_step > MAXIMUM_ROOT_ROTATION_STEP_RAD + 1.0e-10:
            if self._enforce_motion_envelope:
                return self._hold("root_rotation_step")
            preview_envelope_violations.append("root_rotation_step")
        phase_advance = float(np.clip(raw_phase_advance, 0.0, self._phase_cap))
        if not 0.0 <= phase_advance <= self._phase_cap + 1.0e-12:
            return self._hold("phase_advance_cap")
        try:
            advanced_recurrent_state = advance_recurrent_state(
                self._recurrent_state,
                planned,
                physical_recurrent,
                phase_advance_cap=torch.tensor(
                    (self._phase_cap,),
                    dtype=self._recurrent_state.root_world_xy.dtype,
                    device=self._device,
                ),
            )
        except (RuntimeError, TypeError, ValueError):
            return self._hold("recurrent_state")
        if not torch.equal(
            advanced_recurrent_state.root_world_xy, candidate_root_world_xy
        ) or not torch.equal(
            advanced_recurrent_state.root_yaw_world, candidate_root_yaw_world
        ):
            return self._hold("recurrent_root_mismatch")
        new_xy = np.asarray(
            advanced_recurrent_state.root_world_xy[0]
            .to(device="cpu", dtype=torch.float64)
            .numpy(),
            dtype=np.float64,
        )
        new_yaw = float(
            advanced_recurrent_state.root_yaw_world[0].to(
                device="cpu", dtype=torch.float64
            )
        )
        new_quaternion = _root_quaternion(new_yaw, tilt_report)
        new_position = np.asarray(
            (new_xy[0], new_xy[1], new_support.height_m + root_height),
            dtype=np.float64,
        )
        predicted_world_position = np.asarray(
            advanced_recurrent_state.predicted_position_world_xy[0]
            .to(device="cpu", dtype=torch.float64)
            .numpy(),
            dtype=np.float64,
        )
        predicted_world_direction = np.asarray(
            advanced_recurrent_state.predicted_direction_world_xy[0]
            .to(device="cpu", dtype=torch.float64)
            .numpy(),
            dtype=np.float64,
        )
        new_trajectory = PFNNTrajectoryState(
            predicted_world_position,
            predicted_world_direction,
            semantic,
        )
        joints_report = np.asarray(
            joints.to(device="cpu", dtype=torch.float64), dtype=np.float64
        )
        contacts = np.asarray(
            torch.sigmoid(raw_output[contact_field]).to(
                device="cpu", dtype=torch.float64
            ),
            dtype=np.float64,
        )
        diagnostics = {
            "wall_tick": self._wall_tick,
            "hold_count": self._hold_count,
            "desired_speed_m_s": effective_speed,
            "requested_speed_m_s": requested_speed,
            "phase_advance": phase_advance,
            "raw_phase_advance": raw_phase_advance,
            "terrain_grade_degrees": new_support.absolute_grade_degrees,
            "replanned_unsupported_future": not supported,
        }
        if not self._enforce_motion_envelope:
            diagnostics.update(
                {
                    "preview_envelope_violations": tuple(
                        preview_envelope_violations
                    ),
                    "preview_max_joint_step_rad": maximum_joint_step,
                    "preview_root_translation_step_m": root_translation_step,
                    "preview_root_rotation_step_rad": root_rotation_step,
                    "preview_minimum_trajectory_direction_norm": float(
                        torch.min(direction_norm)
                    ),
                    "preview_maximum_trajectory_direction_norm": float(
                        torch.max(direction_norm)
                    ),
                }
            )
        frame = PFNNRuntimeFrame(
            root_position_world=new_position,
            root_quaternion_world_wxyz=new_quaternion,
            joint_position_isaaclab=joints_report,
            phase=float(
                advanced_recurrent_state.phase[0].to(
                    device="cpu", dtype=torch.float64
                )
            ),
            contact_probability=contacts,
            trajectory=new_trajectory,
            supported=supported,
            diagnostics=diagnostics,
        )
        # Commit every recurrent field only after all validation above succeeds.
        self._recurrent_state = advanced_recurrent_state
        self._bootstrap_pending = False
        self._frame = frame
        return frame


@dataclass(frozen=True)
class RuntimeGeometryObservation:
    maximum_sole_penetration_m: float
    maximum_forbidden_body_penetration_m: float
    forbidden_geom_names: tuple[str, ...]
    stance_sole_speeds_m_s: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (
            float(self.maximum_sole_penetration_m),
            float(self.maximum_forbidden_body_penetration_m),
            *map(float, self.stance_sole_speeds_m_s),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("runtime geometry observation must be finite and nonnegative")
        if any(type(name) is not str or not name for name in self.forbidden_geom_names):
            raise ValueError("forbidden geometry names must be nonempty strings")


def _transform_points(rotation: np.ndarray, position: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation.T + position


def _deterministic_collision_surface_samples(
    kind: str,
    size: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
    *,
    vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic world samples spanning an entire collision surface.

    Primitive sampling includes uphill and lateral extrema rather than just the
    globally lowest point.  Meshes use every transformed collision vertex.
    """

    dimensions = _owned_array(size, (3,), "collision geom size")
    attitude = _owned_array(rotation, (3, 3), "collision geom rotation")
    origin = _owned_array(position, (3,), "collision geom position")
    if np.any(dimensions < 0.0):
        raise ValueError("collision geom size must be nonnegative")

    if kind == "mesh":
        if vertices is None:
            raise ValueError("mesh collision samples require vertices")
        local = np.asarray(vertices, dtype=np.float64)
        if (
            local.ndim != 2
            or local.shape[1:] != (3,)
            or len(local) == 0
            or not np.isfinite(local).all()
        ):
            raise ValueError("native collision mesh has invalid vertices")
        return _transform_points(attitude, origin, local)

    azimuth = np.arange(64, dtype=np.float64) * (TWO_PI / 64.0)
    latitude = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 33)
    unit_sphere = np.asarray(
        [
            (
                math.cos(lat) * math.cos(angle),
                math.cos(lat) * math.sin(angle),
                math.sin(lat),
            )
            for lat in latitude
            for angle in azimuth
        ],
        dtype=np.float64,
    )
    if kind == "sphere":
        local = unit_sphere * dimensions[0]
    elif kind == "ellipsoid":
        local = unit_sphere * dimensions
    elif kind == "capsule":
        radius, half_length = dimensions[:2]
        lower = unit_sphere * radius + np.asarray((0.0, 0.0, -half_length))
        upper = unit_sphere * radius + np.asarray((0.0, 0.0, half_length))
        rings = np.asarray(
            [
                (radius * math.cos(angle), radius * math.sin(angle), z)
                for z in (-half_length, 0.0, half_length)
                for angle in azimuth
            ],
            dtype=np.float64,
        )
        local = np.concatenate((lower, upper, rings), axis=0)
    elif kind == "box":
        local = np.asarray(
            [
                (x, y, z)
                for x in (-dimensions[0], dimensions[0])
                for y in (-dimensions[1], dimensions[1])
                for z in (-dimensions[2], dimensions[2])
            ],
            dtype=np.float64,
        )
    elif kind == "cylinder":
        radius, half_length = dimensions[:2]
        local = np.asarray(
            [
                (radius * math.cos(angle), radius * math.sin(angle), z)
                for z in (-half_length, half_length)
                for angle in azimuth
            ]
            + [(0.0, 0.0, -half_length), (0.0, 0.0, half_length)],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported native collision geom kind {kind}")
    return _transform_points(attitude, origin, local)


class NativeG1RuntimeGeometry:
    """Raw native-MuJoCo geometry samples used by the Task-7 recorder."""

    def __init__(self, model: object) -> None:
        import mujoco

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("NativeG1RuntimeGeometry requires a MuJoCo model")
        self._mujoco = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(model.nbody)
        )
        if any(body_names.count(name) != 1 for name in ISAACLAB_BODY_NAMES):
            raise ValueError("native model must contain every canonical G1 body")
        self._body_ids = frozenset(body_names.index(name) for name in ISAACLAB_BODY_NAMES)
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_id < 0:
            free = np.flatnonzero(
                np.asarray(model.jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE)
            )
            if len(free) != 1:
                raise ValueError("native model must have one canonical free joint")
            root_id = int(free[0])
        self._root_qpos = int(model.jnt_qposadr[root_id])
        self._joint_qpos = np.asarray(
            [
                int(
                    model.jnt_qposadr[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    ]
                )
                for name in ISAACLAB_JOINT_NAMES
            ],
            dtype=np.int64,
        )

        def descends(body_id: int, ancestor_id: int) -> bool:
            cursor = int(body_id)
            while cursor > 0:
                if cursor == ancestor_id:
                    return True
                cursor = int(model.body_parentid[cursor])
            return False

        sole_by_foot: list[list[int]] = []
        for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            ankle = body_names.index(body_name)
            geoms = [
                geom
                for geom in range(model.ngeom)
                if descends(int(model.geom_bodyid[geom]), ankle)
                and int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
                and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
            ]
            if len(geoms) != 4:
                raise ValueError(
                    f"{body_name} must have exactly four descendant collision spheres"
                )
            sole_by_foot.append(geoms)
        self._sole_geom_ids = tuple((*sole_by_foot[0], *sole_by_foot[1]))
        self._sole_set = frozenset(self._sole_geom_ids)
        self._forbidden_geom_ids = tuple(
            geom
            for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) in self._body_ids
            and geom not in self._sole_set
            and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
        )
        mujoco.mj_forward(model, self._data)
        groups: list[tuple[int, int]] = []
        for geoms in sole_by_foot:
            ordered = sorted(geoms, key=lambda geom: float(self._data.geom_xpos[geom, 0]))
            groups.extend(((ordered[0], ordered[1]), (ordered[2], ordered[3])))
        self._sole_groups = tuple(groups)
        self._previous_group_xy: np.ndarray | None = None
        self.total_collision_surface_samples = 0
        self.total_terrain_query_points = 0

    @classmethod
    def from_mjcf(cls, path: str | Path) -> "NativeG1RuntimeGeometry":
        import mujoco

        return cls(mujoco.MjModel.from_xml_path(str(Path(path))))

    @property
    def sole_geom_count(self) -> int:
        return len(self._sole_geom_ids)

    @property
    def forbidden_geom_count(self) -> int:
        return len(self._forbidden_geom_ids)

    def _set_qpos(
        self,
        root_position_world: np.ndarray,
        root_quaternion_world_wxyz: np.ndarray,
        joint_position_isaaclab: np.ndarray,
    ) -> None:
        self._data.qpos[:] = self._model.qpos0
        address = self._root_qpos
        self._data.qpos[address : address + 3] = root_position_world
        self._data.qpos[address + 3 : address + 7] = root_quaternion_world_wxyz
        self._data.qpos[self._joint_qpos] = joint_position_isaaclab
        self._mujoco.mj_forward(self._model, self._data)

    def _geom_samples(self, geom: int) -> np.ndarray:
        mujoco = self._mujoco
        model, data = self._model, self._data
        geom_type = int(model.geom_type[geom])
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        position = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            kind = "sphere"
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            kind = "capsule"
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            kind = "box"
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            kind = "cylinder"
        if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            kind = "ellipsoid"
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            stop = start + int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start:stop], dtype=np.float64)
            return _deterministic_collision_surface_samples(
                "mesh", size, rotation, position, vertices=vertices
            )
        if "kind" not in locals():
            raise ValueError(f"unsupported native collision geom type {geom_type}")
        return _deterministic_collision_surface_samples(
            kind, size, rotation, position
        )

    def _sole_bottom_sample(self, geom: int) -> np.ndarray:
        radius = float(self._model.geom_size[geom, 0])
        position = np.asarray(self._data.geom_xpos[geom], dtype=np.float64)
        return position[None, :] + np.asarray(((0.0, 0.0, -radius),))

    @staticmethod
    def _penetration(points: np.ndarray, callback: TerrainCallback) -> float:
        values = np.asarray(points, dtype=np.float64)
        batch = getattr(callback, "collision_heights_at", None)
        if callable(batch):
            try:
                heights = np.asarray(batch(values[:, :2]), dtype=np.float64)
            except (ArithmeticError, TypeError, ValueError) as error:
                raise ValueError("geometry terrain query is missing or invalid") from error
            if heights.shape != (len(values),) or not np.isfinite(heights).all():
                raise ValueError("geometry terrain query is missing or invalid")
            return max(0.0, float(np.max(heights - values[:, 2], initial=0.0)))
        maximum = 0.0
        for point in values:
            try:
                sample = callback(np.asarray(point[:2], dtype=np.float64))
            except (ArithmeticError, TypeError, ValueError):
                sample = None
            if not isinstance(sample, TerrainSample):
                raise ValueError("geometry terrain query is missing or invalid")
            maximum = max(maximum, sample.height_m - float(point[2]))
        return max(0.0, maximum)

    def observe(
        self,
        *,
        root_position_world: object,
        root_quaternion_world_wxyz: object,
        joint_position_isaaclab: object,
        contact_probability: object,
        height_and_grade_at: TerrainCallback,
    ) -> RuntimeGeometryObservation:
        root = _owned_array(root_position_world, (3,), "geometry root position")
        quaternion = _owned_array(
            root_quaternion_world_wxyz, (4,), "geometry root quaternion"
        )
        joints = _owned_array(
            joint_position_isaaclab, (29,), "geometry joint position"
        )
        contacts = _owned_array(contact_probability, (4,), "geometry contacts")
        self._set_qpos(root, quaternion, joints)
        sole_samples = [self._sole_bottom_sample(geom) for geom in self._sole_geom_ids]
        sole_penetration = max(
            self._penetration(samples, height_and_grade_at)
            for samples in sole_samples
        )
        sample_count = sum(len(samples) for samples in sole_samples)
        forbidden_penetration = 0.0
        offenders: list[str] = []
        for geom in self._forbidden_geom_ids:
            samples = self._geom_samples(geom)
            sample_count += len(samples)
            penetration = self._penetration(
                samples, height_and_grade_at
            )
            if penetration > 0.0:
                forbidden_penetration = max(forbidden_penetration, penetration)
                raw_name = self._mujoco.mj_id2name(
                    self._model, self._mujoco.mjtObj.mjOBJ_GEOM, geom
                )
                body = self._model.body(int(self._model.geom_bodyid[geom])).name
                offenders.append(str(raw_name) if raw_name else f"{body}:geom-{geom}")
        self.total_collision_surface_samples += sample_count
        self.total_terrain_query_points += sample_count
        group_xy = np.asarray(
            [
                np.mean(
                    [self._data.geom_xpos[geom, :2] for geom in group], axis=0
                )
                for group in self._sole_groups
            ],
            dtype=np.float64,
        )
        speeds: tuple[float, ...] = ()
        if self._previous_group_xy is not None:
            all_speeds = np.linalg.norm(
                group_xy - self._previous_group_xy, axis=1
            ) * FPS
            speeds = tuple(
                float(all_speeds[index])
                for index in range(4)
                if contacts[index] >= 0.5
            )
        self._previous_group_xy = group_xy
        return RuntimeGeometryObservation(
            maximum_sole_penetration_m=float(sole_penetration),
            maximum_forbidden_body_penetration_m=float(forbidden_penetration),
            forbidden_geom_names=tuple(sorted(set(offenders))),
            stance_sole_speeds_m_s=speeds,
        )


@dataclass(frozen=True)
class _RecordedFrame:
    frame: PFNNRuntimeFrame
    desired_velocity_world: np.ndarray
    terrain_sample: TerrainSample
    geometry: RuntimeGeometryObservation
    traversal_direction: str | None
    realized_delta_world_xy: np.ndarray
    realized_speed_m_s: float
    phase_delta: float
    root_translation_step_m: float
    root_rotation_step_rad: float
    joint_step_rad: float
    command_path_error_m: float


class ClosedLoopRecorder:
    """Accumulate exact global and per-segment closed-loop acceptance gates."""

    def __init__(
        self,
        *,
        joint_limits: object,
        native_geometry: NativeG1RuntimeGeometry | None = None,
        height_and_grade_at: TerrainCallback | None = None,
        traversal_axis_world: object = (1.0, 0.0),
    ) -> None:
        self._limits = _owned_array(joint_limits, (29, 2), "recorder joint limits")
        if np.any(self._limits[:, 0] >= self._limits[:, 1]):
            raise ValueError("recorder joint limits must be ordered")
        self._native_geometry = native_geometry
        self._height_and_grade_at = height_and_grade_at
        self._records: list[_RecordedFrame] = []
        self._traversal_axis = _owned_array(
            traversal_axis_world, (2,), "traversal axis"
        )
        norm = float(np.linalg.norm(self._traversal_axis))
        if norm < 1.0e-12:
            raise ValueError("traversal axis must be nonzero")
        self._traversal_axis = self._traversal_axis / norm
        self._command_position: np.ndarray | None = None

    def record(
        self,
        frame: PFNNRuntimeFrame,
        *,
        desired_velocity_world: object,
        terrain_sample: TerrainSample,
        geometry: RuntimeGeometryObservation | None = None,
        traversal_direction: str | None = None,
    ) -> None:
        if not isinstance(frame, PFNNRuntimeFrame):
            raise TypeError("recorder frame must be PFNNRuntimeFrame")
        desired = _owned_array(desired_velocity_world, (2,), "desired velocity")
        if not isinstance(terrain_sample, TerrainSample):
            raise TypeError("recorder terrain sample must be TerrainSample")
        if traversal_direction not in (None, "forward", "backward"):
            raise ValueError("traversal direction must be forward/backward or None")
        if geometry is None:
            if self._native_geometry is None or self._height_and_grade_at is None:
                raise ValueError("recorder requires native geometry or an observation")
            geometry = self._native_geometry.observe(
                root_position_world=frame.root_position_world,
                root_quaternion_world_wxyz=frame.root_quaternion_world_wxyz,
                joint_position_isaaclab=frame.joint_position_isaaclab,
                contact_probability=frame.contact_probability,
                height_and_grade_at=self._height_and_grade_at,
            )
        previous = self._records[-1] if self._records else None
        if previous is None:
            translation = rotation = joint_step = realized_speed = phase_delta = 0.0
            realized_delta = np.zeros(2, dtype=np.float64)
            self._command_position = frame.root_position_world[:2].copy()
        else:
            realized_delta = (
                frame.root_position_world[:2]
                - previous.frame.root_position_world[:2]
            )
            translation = float(
                np.linalg.norm(
                    frame.root_position_world - previous.frame.root_position_world
                )
            )
            rotation = _quaternion_step(
                previous.frame.root_quaternion_world_wxyz,
                frame.root_quaternion_world_wxyz,
            )
            joint_step = float(
                np.max(
                    np.abs(
                        frame.joint_position_isaaclab
                        - previous.frame.joint_position_isaaclab
                    )
                )
            )
            realized_speed = float(
                np.linalg.norm(
                    frame.root_position_world[:2]
                    - previous.frame.root_position_world[:2]
                )
                * FPS
            )
            raw_delta = frame.phase - previous.frame.phase
            phase_delta = raw_delta
            if phase_delta < -math.pi:
                phase_delta += TWO_PI
            elif phase_delta > math.pi:
                phase_delta -= TWO_PI
        assert self._command_position is not None
        self._command_position += desired * DT
        command_error = float(
            np.linalg.norm(frame.root_position_world[:2] - self._command_position)
        )
        self._records.append(
            _RecordedFrame(
                frame=frame,
                desired_velocity_world=desired,
                terrain_sample=terrain_sample,
                geometry=geometry,
                traversal_direction=traversal_direction,
                realized_delta_world_xy=realized_delta,
                realized_speed_m_s=realized_speed,
                phase_delta=phase_delta,
                root_translation_step_m=translation,
                root_rotation_step_rad=rotation,
                joint_step_rad=joint_step,
                command_path_error_m=command_error,
            )
        )

    @staticmethod
    def _segment(record: _RecordedFrame) -> str:
        if record.terrain_sample.segment is not None:
            return record.terrain_sample.segment
        speed = float(np.linalg.norm(record.desired_velocity_world))
        signed = record.terrain_sample.signed_grade_degrees(
            record.desired_velocity_world if speed > 1.0e-12 else np.zeros(2)
        )
        if signed >= 2.0:
            return "ascent"
        if signed <= -2.0:
            return "descent"
        return "flat"

    def _walking_freezes(self, records: Sequence[_RecordedFrame]) -> int:
        count = 0
        for start in range(max(0, len(records) - 29)):
            window = records[start : start + 30]
            if len(window) < 30:
                continue
            if (
                all(row.frame.supported for row in window)
                and all(np.linalg.norm(row.desired_velocity_world) >= 0.20 for row in window)
                and sum(max(0.0, row.phase_delta) for row in window) < 0.25
            ):
                count += 1
        return count

    def _landing_recovery(self, records: Sequence[_RecordedFrame]) -> tuple[int, int]:
        required = recovered = 0
        for index, record in enumerate(records):
            if self._segment(record) != "landing" or (
                index > 0 and self._segment(records[index - 1]) == "landing"
            ):
                continue
            desired = float(np.linalg.norm(record.desired_velocity_world))
            if desired < 0.20:
                continue
            required += 1
            horizon = records[index : index + 61]
            if any(row.realized_speed_m_s >= 0.5 * desired for row in horizon):
                recovered += 1
        return required, recovered

    def _summary(self, records: Sequence[_RecordedFrame]) -> dict[str, object]:
        values = tuple(records)
        stance = [
            speed
            for row in values
            for speed in row.geometry.stance_sole_speeds_m_s
        ]
        limit_violations = sum(
            bool(
                np.any(row.frame.joint_position_isaaclab < self._limits[:, 0])
                or np.any(row.frame.joint_position_isaaclab > self._limits[:, 1])
            )
            for row in values
        )
        return {
            "frame_count": len(values),
            "duration_seconds": len(values) / FPS,
            "finite": all(
                np.isfinite(row.frame.root_position_world).all()
                and np.isfinite(row.frame.root_quaternion_world_wxyz).all()
                and np.isfinite(row.frame.joint_position_isaaclab).all()
                and math.isfinite(row.frame.phase)
                for row in values
            ),
            "maximum_root_translation_step_m": max(
                (row.root_translation_step_m for row in values), default=0.0
            ),
            "maximum_root_rotation_step_rad": max(
                (row.root_rotation_step_rad for row in values), default=0.0
            ),
            "maximum_joint_step_rad": max(
                (row.joint_step_rad for row in values), default=0.0
            ),
            "joint_limit_violation_count": int(limit_violations),
            "maximum_sole_penetration_m": max(
                (row.geometry.maximum_sole_penetration_m for row in values),
                default=0.0,
            ),
            "maximum_forbidden_body_penetration_m": max(
                (
                    row.geometry.maximum_forbidden_body_penetration_m
                    for row in values
                ),
                default=0.0,
            ),
            "forbidden_geom_names": sorted(
                {
                    name
                    for row in values
                    for name in row.geometry.forbidden_geom_names
                }
            ),
            "median_stance_sole_speed_m_s": (
                float(np.median(stance)) if stance else 0.0
            ),
            "predicted_stance_sample_count": len(stance),
            "phase_reversal_count": sum(row.phase_delta < -1.0e-6 for row in values),
            "walking_freeze_count": self._walking_freezes(values),
            "held_tick_count": sum("hold_reason" in row.frame.diagnostics for row in values),
            "maximum_command_path_error_m": max(
                (row.command_path_error_m for row in values), default=0.0
            ),
        }

    def finalize(self) -> dict[str, object]:
        global_summary = self._summary(self._records)
        def segment_gates(summary: Mapping[str, object]) -> dict[str, bool]:
            return {
                "finite": bool(summary["finite"]),
                "no_phase_reversal": summary["phase_reversal_count"] == 0,
                "no_phase_freeze": summary["walking_freeze_count"] == 0,
                "root_translation_step_within_limit": (
                    summary["maximum_root_translation_step_m"]
                    <= MAXIMUM_ROOT_TRANSLATION_STEP_M
                ),
                "root_rotation_step_within_limit": (
                    summary["maximum_root_rotation_step_rad"]
                    <= MAXIMUM_ROOT_ROTATION_STEP_RAD
                ),
                "joint_step_within_limit": (
                    summary["maximum_joint_step_rad"] <= MAXIMUM_JOINT_STEP_RAD
                ),
                "no_joint_limit_violation": (
                    summary["joint_limit_violation_count"] == 0
                ),
                "sole_penetration_within_limit": (
                    summary["maximum_sole_penetration_m"]
                    <= MAXIMUM_SOLE_PENETRATION_M
                ),
                "zero_forbidden_body_penetration": (
                    summary["maximum_forbidden_body_penetration_m"] == 0.0
                ),
                "stance_sole_speed_within_limit": (
                    summary["predicted_stance_sample_count"] > 0
                    and
                    summary["median_stance_sole_speed_m_s"]
                    <= MAXIMUM_STANCE_SOLE_SPEED_M_S
                ),
                "no_invalid_hold": summary["held_tick_count"] == 0,
            }

        per_segment = {}
        for segment in SEGMENT_NAMES:
            summary = self._summary(
                [row for row in self._records if self._segment(row) == segment]
            )
            per_segment[segment] = {**summary, "gates": segment_gates(summary)}
        realized_grades: dict[str, float] = {"forward": 0.0, "backward": 0.0}
        realized_displacement: dict[str, float] = {"forward": 0.0, "backward": 0.0}
        for row in self._records:
            projected = float(np.dot(row.realized_delta_world_xy, self._traversal_axis))
            direction = (
                "forward" if projected > 1.0e-6
                else "backward" if projected < -1.0e-6
                else None
            )
            if direction is not None:
                realized_displacement[direction] += abs(projected)
                realized_grades[direction] = max(
                    realized_grades[direction],
                    row.terrain_sample.absolute_grade_degrees,
                )
        landing_required, landing_recovered = self._landing_recovery(self._records)
        gates = {
            "finite_20_seconds": bool(
                global_summary["finite"]
                and float(global_summary["duration_seconds"]) >= 20.0
            ),
            "no_phase_reversal": global_summary["phase_reversal_count"] == 0,
            "no_phase_freeze": global_summary["walking_freeze_count"] == 0,
            "root_translation_step_within_limit": (
                global_summary["maximum_root_translation_step_m"]
                <= MAXIMUM_ROOT_TRANSLATION_STEP_M
            ),
            "root_rotation_step_within_limit": (
                global_summary["maximum_root_rotation_step_rad"]
                <= MAXIMUM_ROOT_ROTATION_STEP_RAD
            ),
            "joint_step_within_limit": (
                global_summary["maximum_joint_step_rad"] <= MAXIMUM_JOINT_STEP_RAD
            ),
            "no_joint_limit_violation": (
                global_summary["joint_limit_violation_count"] == 0
            ),
            "sole_penetration_within_limit": (
                global_summary["maximum_sole_penetration_m"]
                <= MAXIMUM_SOLE_PENETRATION_M
            ),
            "zero_forbidden_body_penetration": (
                global_summary["maximum_forbidden_body_penetration_m"] == 0.0
            ),
            "stance_sole_speed_within_limit": (
                global_summary["predicted_stance_sample_count"] > 0
                and
                global_summary["median_stance_sole_speed_m_s"]
                <= MAXIMUM_STANCE_SOLE_SPEED_M_S
            ),
            "traverses_18_9_degrees_both_directions": all(
                realized_displacement[name] > 0.0
                and realized_grades[name] >= 18.85
                for name in realized_grades
            ),
            "responsive_flat_landing_recovery": (
                landing_required > 0 and landing_recovered == landing_required
            ),
            "no_invalid_hold": global_summary["held_tick_count"] == 0,
        }
        numeric_gates = {
            "root_translation_step_m": {
                "value": global_summary["maximum_root_translation_step_m"],
                "limit": MAXIMUM_ROOT_TRANSLATION_STEP_M,
                "gate": "root_translation_step_within_limit",
            },
            "root_rotation_step_rad": {
                "value": global_summary["maximum_root_rotation_step_rad"],
                "limit": MAXIMUM_ROOT_ROTATION_STEP_RAD,
                "gate": "root_rotation_step_within_limit",
            },
            "joint_step_rad": {
                "value": global_summary["maximum_joint_step_rad"],
                "limit": MAXIMUM_JOINT_STEP_RAD,
                "gate": "joint_step_within_limit",
            },
            "sole_penetration_m": {
                "value": global_summary["maximum_sole_penetration_m"],
                "limit": MAXIMUM_SOLE_PENETRATION_M,
                "gate": "sole_penetration_within_limit",
            },
            "stance_sole_speed_m_s": {
                "value": global_summary["median_stance_sole_speed_m_s"],
                "limit": MAXIMUM_STANCE_SOLE_SPEED_M_S,
                "gate": "stance_sole_speed_within_limit",
            },
        }
        return {
            "schema": "mm-sonic-terrain-pfnn-closed-loop/v1",
            "fps": FPS,
            "global": global_summary,
            "segments": per_segment,
            "traversal": {
                "maximum_grade_degrees_by_direction": realized_grades,
                "realized_displacement_m": realized_displacement,
            },
            "landing_recovery": {
                "required": landing_required,
                "recovered": landing_recovered,
            },
            "gates": gates,
            "numeric_gates": numeric_gates,
            "accepted": all(gates.values()),
        }


@dataclass(frozen=True)
class ClosedLoopValidationResult:
    evaluated: bool
    split: str
    hard_failures: int
    normalized_excess: float
    metrics: dict[str, object]

    def __post_init__(self) -> None:
        if type(self.evaluated) is not bool or self.split != "validation":
            raise ValueError("closed-loop selection result must be validation-only")
        if type(self.hard_failures) is not int or self.hard_failures < 0:
            raise ValueError("closed-loop hard_failures must be nonnegative")
        if (
            type(self.normalized_excess) not in (int, float)
            or not math.isfinite(float(self.normalized_excess))
            or self.normalized_excess < 0.0
        ):
            raise ValueError("closed-loop normalized_excess must be nonnegative")
        if type(self.metrics) is not dict:
            raise TypeError("closed-loop metrics must be a dictionary")
        object.__setattr__(self, "normalized_excess", float(self.normalized_excess))
        object.__setattr__(self, "metrics", dict(self.metrics))

    @property
    def failure_penalty(self) -> float:
        return 1000.0 * self.hard_failures + self.normalized_excess

    @classmethod
    def from_metrics(
        cls, metrics: Mapping[str, object], *, split: str
    ) -> "ClosedLoopValidationResult":
        if split != "validation":
            raise ValueError("closed-loop scoring accepts only the validation split")
        gates = metrics.get("gates")
        numeric = metrics.get("numeric_gates")
        if not isinstance(gates, Mapping) or not isinstance(numeric, Mapping):
            raise ValueError("closed-loop metrics must contain gates and numeric_gates")
        if any(type(value) is not bool for value in gates.values()):
            raise ValueError("closed-loop gates must be booleans")
        failed = {str(name) for name, value in gates.items() if not value}
        excess = 0.0
        for name, raw in numeric.items():
            if not isinstance(raw, Mapping):
                raise ValueError("closed-loop numeric gate is invalid")
            try:
                value = float(raw["value"])
                limit = float(raw["limit"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("closed-loop numeric gate is invalid") from error
            if not math.isfinite(value) or not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("closed-loop numeric gate is invalid")
            if value > limit:
                # A numeric bound is an exact gate in its own right.  Its
                # normalized excess and hard failure are retained even when a
                # corresponding boolean summary gate is also false.
                failed.add(f"numeric:{name}")
                excess += value / limit - 1.0
        return cls(
            evaluated=True,
            split=split,
            hard_failures=len(failed),
            normalized_excess=excess,
            metrics=dict(metrics),
        )


def validation_identity_set_receipt(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Seal the exact validation identity set from a dataset manifest."""

    dataset_digest = manifest.get("dataset_digest_sha256")
    split_identities = manifest.get("split_identities")
    identities = (
        split_identities.get("validation")
        if isinstance(split_identities, Mapping)
        else None
    )
    if (
        not _is_sha256(dataset_digest)
        or type(identities) is not list
        or not identities
        or any(type(identity) is not str or not identity for identity in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError("validation identity-set manifest receipt is invalid")
    base: dict[str, object] = {
        "schema": "mm-sonic-validation-identity-set/v1",
        "dataset_digest_sha256": dataset_digest,
        "split": "validation",
        "identities": list(identities),
    }
    return {**base, "receipt_sha256": _canonical_receipt_sha256(base)}


def _validated_scenario_provenance(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("closed-loop scenario provenance is invalid")
    required = {
        "schema",
        "dataset_digest_sha256",
        "split",
        "validation_identity_set_sha256",
        "kinematic_signature_sha256",
        "scenario_id",
        "terrain_source_receipt",
        "provenance_sha256",
    }
    if set(value) != required:
        raise ValueError("closed-loop scenario provenance is invalid")
    terrain = value["terrain_source_receipt"]
    if (
        value["schema"] != "mm-sonic-closed-loop-scenario/v1"
        or value["split"] != "validation"
        or not _is_sha256(value["dataset_digest_sha256"])
        or not _is_sha256(value["validation_identity_set_sha256"])
        or not _is_sha256(value["kinematic_signature_sha256"])
        or type(value["scenario_id"]) is not str
        or not value["scenario_id"]
        or type(terrain) is not dict
        or not {"kind", "source_sha256", "version"}.issubset(terrain)
        or type(terrain["kind"]) is not str
        or not terrain["kind"]
        or not _is_sha256(terrain["source_sha256"])
        or type(terrain["version"]) is not str
        or not terrain["version"]
    ):
        raise ValueError("closed-loop scenario provenance is invalid")
    try:
        canonical_terrain = json.loads(
            json.dumps(
                terrain, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("closed-loop terrain source receipt is invalid") from error
    base = {
        name: (canonical_terrain if name == "terrain_source_receipt" else value[name])
        for name in required - {"provenance_sha256"}
    }
    if value["provenance_sha256"] != _canonical_receipt_sha256(base):
        raise ValueError("closed-loop scenario provenance digest mismatch")
    return {**base, "provenance_sha256": value["provenance_sha256"]}


@dataclass(frozen=True)
class ClosedLoopScenario:
    """A callback paired with fail-closed, manifest-bound provenance."""

    provenance: dict[str, object]
    evaluator: Callable[[], Mapping[str, object]]

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise TypeError("closed-loop scenario evaluator must be callable")
        object.__setattr__(self, "provenance", dict(self.provenance))


def make_validation_scenario(
    *,
    manifest: Mapping[str, object],
    kinematic_signature_sha256: str,
    scenario_id: str,
    terrain_source_receipt: Mapping[str, object],
    evaluator: Callable[[], Mapping[str, object]],
) -> ClosedLoopScenario:
    """Create a validation-only scenario sealed to manifest and terrain assets."""

    identity = validation_identity_set_receipt(manifest)
    if not _is_sha256(kinematic_signature_sha256):
        raise ValueError("scenario kinematic signature is invalid")
    try:
        terrain = json.loads(
            json.dumps(
                dict(terrain_source_receipt),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("scenario terrain source receipt is invalid") from error
    base: dict[str, object] = {
        "schema": "mm-sonic-closed-loop-scenario/v1",
        "dataset_digest_sha256": identity["dataset_digest_sha256"],
        "split": "validation",
        "validation_identity_set_sha256": identity["receipt_sha256"],
        "kinematic_signature_sha256": kinematic_signature_sha256,
        "scenario_id": scenario_id,
        "terrain_source_receipt": terrain,
    }
    provenance = {**base, "provenance_sha256": _canonical_receipt_sha256(base)}
    checked = _validated_scenario_provenance(provenance)
    return ClosedLoopScenario(checked, evaluator)


def evaluate_closed_loop_scenarios(
    *,
    split: str,
    scenarios: Sequence[ClosedLoopScenario],
    expected_dataset_digest: str,
    expected_kinematic_signature_sha256: str,
    expected_identity_set_sha256: str,
) -> ClosedLoopValidationResult:
    """Validate every sealed receipt before opening any scenario callback."""

    if split != "validation":
        raise ValueError("closed-loop scoring accepts only validation scenarios")
    if not scenarios or any(not isinstance(scenario, ClosedLoopScenario) for scenario in scenarios):
        raise ValueError("closed-loop validation requires explicit scenarios")
    checked = [_validated_scenario_provenance(scenario.provenance) for scenario in scenarios]
    for provenance in checked:
        if (
            provenance["dataset_digest_sha256"] != expected_dataset_digest
            or provenance["kinematic_signature_sha256"]
            != expected_kinematic_signature_sha256
            or provenance["validation_identity_set_sha256"]
            != expected_identity_set_sha256
        ):
            raise ValueError("closed-loop scenario provenance does not match")
    results = [
        ClosedLoopValidationResult.from_metrics(scenario.evaluator(), split=split)
        for scenario in scenarios
    ]
    provenance_digest = _canonical_receipt_sha256(
        {"scenarios": [value["provenance_sha256"] for value in checked]}
    )
    return ClosedLoopValidationResult(
        evaluated=True,
        split="validation",
        hard_failures=sum(result.hard_failures for result in results),
        normalized_excess=sum(result.normalized_excess for result in results),
        metrics={
            "schema": "mm-sonic-terrain-pfnn-validation-scenarios/v1",
            "scenario_count": len(results),
            "dataset_digest_sha256": expected_dataset_digest,
            "kinematic_signature_sha256": expected_kinematic_signature_sha256,
            "validation_identity_set_sha256": expected_identity_set_sha256,
            "scenario_provenance_sha256": provenance_digest,
            "scenario_provenance": checked,
            "scenarios": [result.metrics for result in results],
        },
    )


def write_closed_loop_json(path: str | Path, report: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(report), sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(encoded)
    temporary.replace(destination)


__all__ = [
    "ClosedLoopRecorder",
    "ClosedLoopValidationResult",
    "NativeG1RuntimeGeometry",
    "PFNNRuntimeFrame",
    "PFNNTrajectoryState",
    "RuntimeGeometryObservation",
    "TerrainPFNNRuntime",
    "TerrainSample",
    "evaluate_closed_loop_scenarios",
    "write_closed_loop_json",
]
