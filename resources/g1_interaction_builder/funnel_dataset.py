"""Deterministic object-local training rows for the interaction funnel model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import read_artifact_set
from .schema import InteractionArtifact, InteractionPhase


APPROACH_HORIZON_FRAMES = 75
KNOT_FRAME_OFFSETS = np.asarray(
    [0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 49, 54, 59, 64, 69, 74],
    dtype=np.int64,
)


@dataclass(frozen=True)
class FunnelDataset:
    conditions: np.ndarray  # [N, 24], float32
    funnels: np.ndarray  # [N, 16, 4], float32
    sequence_indices: np.ndarray  # [N], int32

    def validate(self) -> None:
        if self.conditions.ndim != 2 or self.conditions.shape[1] != 24:
            raise ValueError("conditions must have shape [N, 24]")
        if self.funnels.ndim != 3 or self.funnels.shape[1:] != (16, 4):
            raise ValueError("funnels must have shape [N, 16, 4]")
        if (len(self.conditions) != len(self.funnels) or
                len(self.conditions) != len(self.sequence_indices)):
            raise ValueError("dataset row counts do not match")
        if not np.isfinite(self.conditions).all() or not np.isfinite(self.funnels).all():
            raise ValueError("dataset contains non-finite values")


def is_certifiable_funnel(funnel: np.ndarray) -> bool:
    funnel = np.asarray(funnel, dtype=np.float32)
    if funnel.shape != (16, 4) or not np.isfinite(funnel).all():
        return False
    translation_step = np.linalg.norm(np.diff(funnel[:, :2], axis=0), axis=1)
    yaw_step = np.abs(np.arctan2(
        funnel[1:, 2] * funnel[:-1, 3] - funnel[1:, 3] * funnel[:-1, 2],
        funnel[1:, 2] * funnel[:-1, 2] + funnel[1:, 3] * funnel[:-1, 3],
    ))
    return bool(
        translation_step.max() <= 0.08
        and yaw_step.max() <= np.deg2rad(15.0)
        and translation_step.sum() >= 0.15
    )


def _yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _object_local_delta(delta: np.ndarray, object_yaw: float) -> tuple[float, float]:
    c, s = np.cos(object_yaw), np.sin(object_yaw)
    return float(c * delta[0] + s * delta[2]), float(-s * delta[0] + c * delta[2])


def object_local_execution(
    root_positions_world: np.ndarray,
    root_yaws_world: np.ndarray,
    object_position_world: np.ndarray,
    object_yaw_world: float,
) -> np.ndarray:
    positions = np.asarray(root_positions_world, dtype=np.float64)
    yaws = np.asarray(root_yaws_world, dtype=np.float64)
    object_position = np.asarray(object_position_world, dtype=np.float64)
    if positions.shape != (16, 3) or yaws.shape != (16,) or object_position.shape != (3,):
        raise ValueError("object-local execution inputs have invalid shapes")
    if not np.isfinite(positions).all() or not np.isfinite(yaws).all():
        raise ValueError("object-local execution inputs must be finite")
    delta = positions - object_position
    c, s = np.cos(object_yaw_world), np.sin(object_yaw_world)
    local_x = c * delta[:, 0] + s * delta[:, 2]
    local_z = -s * delta[:, 0] + c * delta[:, 2]
    relative_yaw = yaws - object_yaw_world
    return np.stack(
        (local_x, local_z, np.sin(relative_yaw), np.cos(relative_yaw)), axis=1
    ).astype(np.float32)


def _condition(artifact: InteractionArtifact, clip: int, reach: int) -> np.ndarray:
    hand = int(artifact.active_hands[clip])
    grasp_rotation = np.asarray(artifact.grasp_rotations_object[clip], np.float32)
    # Quaternion order is wxyz; retain the first two rotation-matrix columns.
    w, x, y, z = grasp_rotation
    rotation6 = np.asarray(
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w),
         2 * (x * z - y * w), 2 * (x * y - z * w),
         1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], np.float32
    )
    approach = np.asarray(artifact.approach_directions_object[clip], np.float32)
    norm = float(np.linalg.norm(approach[[0, 2]]))
    if norm <= 1e-8:
        raise ValueError(f"clip {clip} has a degenerate approach direction")
    support_height = float(artifact.object_positions[reach, 1] - artifact.table_positions[clip, 1])
    grasp_height = float(artifact.grasp_positions_object[clip, 1])
    return np.asarray(
        [1.0 if hand == 0 else 0.0, 1.0 if hand == 1 else 0.0,
         *np.asarray(artifact.grasp_positions_object[clip], np.float32), *rotation6,
         float(approach[0] / norm), float(approach[2] / norm),
         *np.asarray(artifact.object_dimensions[clip], np.float32),
         support_height, grasp_height], np.float32)


def _entry_relative_execution(execution_object: np.ndarray) -> np.ndarray:
    execution_object = np.asarray(execution_object, dtype=np.float32)
    if execution_object.shape != (16, 4):
        raise ValueError("execution_object must have shape (16, 4)")
    entry_x, entry_z, entry_sin, entry_cos = execution_object[0]
    delta_x = execution_object[:, 0] - entry_x
    delta_z = execution_object[:, 1] - entry_z
    relative = np.empty_like(execution_object)
    relative[:, 0] = entry_cos * delta_x + entry_sin * delta_z
    relative[:, 1] = -entry_sin * delta_x + entry_cos * delta_z
    relative[:, 2] = (
        execution_object[:, 2] * entry_cos
        - execution_object[:, 3] * entry_sin
    )
    relative[:, 3] = (
        execution_object[:, 3] * entry_cos
        + execution_object[:, 2] * entry_sin
    )
    relative[0] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    return relative


def extract_dataset(artifact: InteractionArtifact) -> FunnelDataset:
    artifact.validate()
    conditions, funnels, indices = [], [], []
    for clip, (start, stop) in enumerate(zip(artifact.range_starts, artifact.range_stops)):
        clip_phases = artifact.phases[start:stop]
        reaches = np.flatnonzero(clip_phases == int(InteractionPhase.REACH))
        contacts = np.flatnonzero(clip_phases == int(InteractionPhase.CONTACT))
        if len(reaches) == 0 or len(contacts) == 0 or int(reaches[0]) < 74:
            continue
        reach = int(start + reaches[0])
        entry = reach - (APPROACH_HORIZON_FRAMES - 1)
        if entry < int(start):
            continue
        object_frame = int(start + contacts[0] - 1)
        object_yaw = _yaw(artifact.object_rotations[object_frame])
        object_position = artifact.object_positions[object_frame]
        frame_indices = entry + KNOT_FRAME_OFFSETS
        execution_object = object_local_execution(
            artifact.positions[frame_indices, 0],
            np.asarray([_yaw(artifact.rotations[frame, 0]) for frame in frame_indices]),
            object_position,
            object_yaw,
        )
        execution_entry = _entry_relative_execution(execution_object)
        # Model convention is outward terminal-to-entrance order.
        funnel = np.ascontiguousarray(execution_entry[::-1], dtype=np.float32)
        funnel[15] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        if not is_certifiable_funnel(funnel):
            continue
        velocity_x, velocity_z = _object_local_delta(
            artifact.velocities[entry, 0], object_yaw)
        entry_condition = np.asarray(
            (*execution_object[0], velocity_x, velocity_z), dtype=np.float32)
        funnels.append(funnel)
        conditions.append(np.concatenate(
            (_condition(artifact, clip, reach), entry_condition)
        ).astype(np.float32))
        indices.append(clip)
    dataset = FunnelDataset(
        np.asarray(conditions, np.float32),
        np.asarray(funnels, np.float32),
        np.asarray(indices, np.int32),
    )
    dataset.validate()
    if len(dataset.conditions) == 0:
        raise ValueError("interaction artifact produced no valid funnel rows")
    return dataset


def load_dataset(pack: Path) -> FunnelDataset:
    artifact, _, _, _, _ = read_artifact_set(Path(pack))
    return extract_dataset(artifact)
