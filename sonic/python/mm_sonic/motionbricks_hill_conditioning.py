"""Pre-inference terrain-height conditioning for pinned MotionBricks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


HeightQuery = Callable[[object], float]


def canonical_targets_to_world_xy(
    target_root_positions: np.ndarray,
    heading_radians: float,
    first_frame_world_xy: object,
) -> np.ndarray:
    """Map MotionBricks canonical planar roots to MuJoCo world XY."""

    targets = np.asarray(target_root_positions, dtype=np.float64)
    origin = np.asarray(first_frame_world_xy, dtype=np.float64)
    heading = float(heading_radians)
    if (
        targets.ndim != 2
        or targets.shape[0] != 2
        or targets.shape[1] < 1
        or not np.isfinite(targets).all()
    ):
        raise ValueError("target root positions must have finite shape (2, frames)")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("first-frame world XY must contain two finite values")
    if not math.isfinite(heading):
        raise ValueError("canonicalization heading must be finite")

    # MotionBricks motion space is Y-up/Z-forward. Its planar [X, Z] maps to
    # MuJoCo Z-up [Y, X], respectively.
    canonical_xy = np.column_stack((targets[1], targets[0]))
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    return canonical_xy @ rotation.T + origin


def terrain_height_deltas(
    current_world_xy: object,
    target_world_xy: object,
    height_query: HeightQuery,
) -> np.ndarray:
    """Sample terrain elevation targets relative to current support."""

    current = np.asarray(current_world_xy, dtype=np.float64)
    targets = np.asarray(target_world_xy, dtype=np.float64)
    if current.shape != (2,) or not np.isfinite(current).all():
        raise ValueError("current world XY must contain two finite values")
    if (
        targets.ndim != 2
        or targets.shape[1] != 2
        or targets.shape[0] < 1
        or not np.isfinite(targets).all()
    ):
        raise ValueError("target world XY must have finite shape (frames, 2)")
    current_height = float(height_query(current))
    target_heights = np.asarray(
        [float(height_query(target)) for target in targets], dtype=np.float64
    )
    if not math.isfinite(current_height) or not np.isfinite(target_heights).all():
        raise ValueError("terrain height query returned a non-finite value")
    return target_heights - current_height


@dataclass(frozen=True)
class MotionBricksConditioningTrace:
    """Latest root constraints delivered to MotionBricks inference."""

    current_world_xy: tuple[float, float]
    target_world_xy: tuple[tuple[float, float], ...]
    current_height_world: float
    target_heights_world: tuple[float, ...]
    height_deltas: tuple[float, ...]
    original_target_vertical: tuple[float, ...]
    conditioned_target_vertical: tuple[float, ...]


class MotionBricksHillConditioner:
    """Guarded adapter around MotionBricks' target-transform stage."""

    _METHOD_NAME = "_generate_target_joint_transforms"

    def __init__(self, height_query: HeightQuery) -> None:
        if not callable(height_query):
            raise TypeError("height_query must be callable")
        self._height_query = height_query
        self._agent: Any | None = None
        self._original: Callable[..., object] | None = None
        self._wrapper: Callable[..., object] | None = None
        self.latest_trace: MotionBricksConditioningTrace | None = None

    @property
    def is_installed(self) -> bool:
        return self._agent is not None

    def install(self, agent: object) -> None:
        """Install the adapter on one pinned agent instance."""

        if self.is_installed:
            raise RuntimeError("MotionBricks hill conditioner is already installed")
        original = getattr(agent, self._METHOD_NAME, None)
        if not callable(original):
            raise TypeError(
                "MotionBricks agent lacks callable "
                f"{self._METHOD_NAME}"
            )

        def conditioned(inputs: dict[str, object]) -> object:
            result = original(inputs)
            return self._condition_result(inputs, result)

        setattr(agent, self._METHOD_NAME, conditioned)
        self._agent = agent
        self._original = original
        self._wrapper = conditioned

    def remove(self) -> None:
        """Restore the exact callable that was present before installation."""

        if not self.is_installed:
            return
        assert self._agent is not None
        if getattr(self._agent, self._METHOD_NAME, None) is not self._wrapper:
            raise RuntimeError("MotionBricks target-transform hook changed in place")
        setattr(self._agent, self._METHOD_NAME, self._original)
        self._agent = None
        self._original = None
        self._wrapper = None

    @staticmethod
    def _as_numpy(value: object, label: str) -> np.ndarray:
        try:
            array = value.detach().cpu().numpy()  # type: ignore[attr-defined]
        except AttributeError:
            array = np.asarray(value)
        result = np.asarray(array)
        if not np.isfinite(result).all():
            raise ValueError(f"{label} must be finite")
        return result

    def _condition_result(
        self,
        inputs: dict[str, object],
        result: object,
    ) -> object:
        try:
            joint_positions, joint_rotations, root_positions = result  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise TypeError(
                "MotionBricks target-transform result must contain three tensors"
            ) from error

        roots = self._as_numpy(root_positions, "target global root positions")
        if roots.ndim != 3 or roots.shape[0] != 1 or roots.shape[2] != 3:
            raise ValueError(
                "MotionBricks target global root positions require batch-one "
                "shape (1, frames, 3)"
            )
        target_roots = self._as_numpy(
            inputs.get("target_root_positions"), "spring target root positions"
        )
        if (
            target_roots.ndim != 3
            or target_roots.shape[0] != 1
            or target_roots.shape[1] != 2
        ):
            raise ValueError(
                "MotionBricks spring roots require batch-one shape (1, 2, frames)"
            )
        output_frames = int(roots.shape[1])
        target_frames = int(target_roots.shape[2])
        if target_frames not in (1, output_frames):
            raise ValueError(
                "MotionBricks target frame count must be one or match target clip"
            )

        headings = self._as_numpy(
            inputs.get("first_frame_heading_angle"),
            "first-frame heading",
        )
        origins = self._as_numpy(
            inputs.get("first_frame_position"), "first-frame position"
        )
        raw_context = self._as_numpy(
            inputs.get("raw_context_mujoco_qpos"), "raw context qpos"
        )
        if headings.shape != (1,):
            raise ValueError("first-frame heading requires batch-one shape (1,)")
        if origins.shape != (1, 3):
            raise ValueError("first-frame position requires batch-one shape (1, 3)")
        if (
            raw_context.ndim != 3
            or raw_context.shape[0] != 1
            or raw_context.shape[1] < 1
            or raw_context.shape[2] < 2
        ):
            raise ValueError(
                "raw context qpos requires batch-one shape (1, frames, qpos)"
            )

        target_world = canonical_targets_to_world_xy(
            target_roots[0],
            float(headings[0]),
            origins[0, :2],
        )
        if target_frames == 1 and output_frames > 1:
            target_world = np.repeat(target_world, output_frames, axis=0)
        current_world = np.asarray(raw_context[0, -1, :2], dtype=np.float64)
        height_deltas = terrain_height_deltas(
            current_world, target_world, self._height_query
        )
        current_height = float(self._height_query(current_world))
        target_heights = height_deltas + current_height

        try:
            import torch

            delta_tensor = torch.as_tensor(
                height_deltas,
                dtype=root_positions.dtype,
                device=root_positions.device,
            )
            conditioned_roots = root_positions.clone()
            conditioned_roots[0, :, 1] += delta_tensor
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "MotionBricks target global root positions must be a torch tensor"
            ) from error

        original_vertical = roots[0, :, 1].astype(np.float64)
        conditioned_vertical = original_vertical + height_deltas
        self.latest_trace = MotionBricksConditioningTrace(
            current_world_xy=tuple(float(value) for value in current_world),
            target_world_xy=tuple(
                tuple(float(value) for value in point) for point in target_world
            ),
            current_height_world=current_height,
            target_heights_world=tuple(float(value) for value in target_heights),
            height_deltas=tuple(float(value) for value in height_deltas),
            original_target_vertical=tuple(
                float(value) for value in original_vertical
            ),
            conditioned_target_vertical=tuple(
                float(value) for value in conditioned_vertical
            ),
        )
        return joint_positions, joint_rotations, conditioned_roots
