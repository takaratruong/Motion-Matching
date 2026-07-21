"""Frozen full-body motion schema and genuine overlap dataset extraction."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.schema import G1_SKELETON, InteractionArtifact, InteractionHand, InteractionPhase
from resources.g1_terrain_builder.kinematics import G1Kinematics, convert_source_clip
from resources.g1_terrain_builder.schema import HoldenClip, SourceClip


BONE_COUNT = 31
FRAME_DIM = 3 + BONE_COUNT * 6 + 3
WALK_FRAMES = 50
PICKUP_FRAMES = 50
OVERLAP_FRAMES = 20
TIMELINE_FRAMES = 80
FPS = 25.0
_ROTATION_EPSILON = 1e-8
STATIC_CONDITION_DIM = 25
TEMPORAL_CONDITION_DIM = 9


@dataclass(frozen=True)
class DecodedMotion:
    positions: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    contacts: np.ndarray


@dataclass(frozen=True)
class OverlapDataset:
    """Interaction rows with one genuine 20-frame shared source segment."""

    walk_windows: np.ndarray
    pickup_windows: np.ndarray
    static_conditions: np.ndarray
    walk_temporal: np.ndarray
    pickup_temporal: np.ndarray
    sequence_indices: np.ndarray
    object_ids: np.ndarray
    source_walk_ranges: np.ndarray
    source_pickup_ranges: np.ndarray
    continuation_sequence_indices: np.ndarray
    source_continuation_ranges: np.ndarray
    rejection_counts: dict[str, int]


def walk_slice() -> slice:
    return slice(0, WALK_FRAMES)


def pickup_slice() -> slice:
    return slice(WALK_FRAMES - OVERLAP_FRAMES, TIMELINE_FRAMES)


def _array(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _anchor(value: np.ndarray) -> np.ndarray:
    return _array(value, (3,), "anchor")


def _normalised_quaternions(rotations: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(rotations, axis=-1, keepdims=True)
    if np.any(lengths < _ROTATION_EPSILON):
        raise ValueError("rotations contain a zero-length quaternion")
    result = rotations / lengths
    result = np.where(result[..., :1] < 0.0, -result, result)
    return result


def _quaternion_to_rotation6d(rotations: np.ndarray) -> np.ndarray:
    matrices = holden_quat.to_xform(rotations)
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)


def _rotation6d_to_quaternion(rotation6d: np.ndarray) -> np.ndarray:
    first = rotation6d[..., :3]
    first_length = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_length < _ROTATION_EPSILON):
        raise ValueError("rotation6d first column norm is below 1e-8")
    first = first / first_length

    second = rotation6d[..., 3:] - np.sum(
        first * rotation6d[..., 3:], axis=-1, keepdims=True
    ) * first
    second_length = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_length < _ROTATION_EPSILON):
        raise ValueError("rotation6d second column norm is below 1e-8")
    second = second / second_length
    third = np.cross(first, second)
    matrices = np.stack((first, second, third), axis=-1)
    rotations = np.asarray(holden_quat.from_xform(matrices), dtype=np.float64)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
    return np.where(rotations[..., :1] < 0.0, -rotations, rotations)


def _finite_difference_vectors(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if len(values) == 1:
        return result
    result[0] = (values[1] - values[0]) * FPS
    result[-1] = (values[-1] - values[-2]) * FPS
    if len(values) > 2:
        result[1:-1] = (values[2:] - values[:-2]) * (0.5 * FPS)
    return result


def _finite_difference_quaternions(rotations: np.ndarray) -> np.ndarray:
    result = np.zeros(rotations.shape[:-1] + (3,), dtype=np.float64)
    if len(rotations) == 1:
        return result
    unrolled = holden_quat.unroll(rotations.copy())
    result[0] = holden_quat.to_scaled_angle_axis(
        holden_quat.mul(unrolled[1], holden_quat.inv(unrolled[0]))
    ) * FPS
    result[-1] = holden_quat.to_scaled_angle_axis(
        holden_quat.mul(unrolled[-1], holden_quat.inv(unrolled[-2]))
    ) * FPS
    if len(rotations) > 2:
        result[1:-1] = holden_quat.to_scaled_angle_axis(
            holden_quat.mul(unrolled[2:], holden_quat.inv(unrolled[:-2]))
        ) * (0.5 * FPS)
    return result


def encode_motion(
    positions: np.ndarray, rotations: np.ndarray, contacts: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    """Encode a local Holden pose sequence into the strict 192-channel schema."""
    positions = np.asarray(positions, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[1:] != (BONE_COUNT, 3)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("positions must be finite with shape (N, 31, 3)")
    frames = len(positions)
    rotations = _array(rotations, (frames, BONE_COUNT, 4), "rotations")
    contacts = _array(contacts, (frames, 3), "contacts")
    anchor = _anchor(anchor)

    rotation6d = _quaternion_to_rotation6d(_normalised_quaternions(rotations))
    return np.concatenate(
        (positions[:, 0] - anchor, rotation6d.reshape(frames, -1), contacts),
        axis=-1,
    ).astype(np.float32)


def decode_motion(
    encoded: np.ndarray, local_positions: np.ndarray, anchor: np.ndarray,
) -> DecodedMotion:
    """Decode root motion and rotations while restoring frozen local offsets."""
    encoded = np.asarray(encoded, dtype=np.float64)
    if encoded.ndim != 2 or encoded.shape[1] != FRAME_DIM or not np.isfinite(encoded).all():
        raise ValueError(f"encoded motion must be finite with shape (N, {FRAME_DIM})")
    frames = len(encoded)
    if frames == 0:
        raise ValueError("encoded motion must contain at least one frame")
    local_positions = _array(local_positions, (BONE_COUNT, 3), "local_positions")
    anchor = _anchor(anchor)

    rotations = _rotation6d_to_quaternion(
        encoded[:, 3:3 + BONE_COUNT * 6].reshape(frames, BONE_COUNT, 6)
    )
    positions = np.broadcast_to(
        local_positions, (frames, BONE_COUNT, 3)
    ).copy()
    positions[:, 0] = encoded[:, :3] + anchor
    return DecodedMotion(
        positions=positions.astype(np.float32),
        rotations=rotations.astype(np.float32),
        velocities=_finite_difference_vectors(positions).astype(np.float32),
        angular_velocities=_finite_difference_quaternions(rotations).astype(
            np.float32
        ),
        contacts=encoded[:, -3:].astype(np.float32),
    )


def _empty_overlap_dataset(rejections: dict[str, int]) -> OverlapDataset:
    return OverlapDataset(
        np.empty((0, WALK_FRAMES, FRAME_DIM), np.float32),
        np.empty((0, PICKUP_FRAMES, FRAME_DIM), np.float32),
        np.empty((0, STATIC_CONDITION_DIM), np.float32),
        np.empty((0, WALK_FRAMES, TEMPORAL_CONDITION_DIM), np.float32),
        np.empty((0, PICKUP_FRAMES, TEMPORAL_CONDITION_DIM), np.float32),
        np.empty(0, np.int32),
        np.empty(0, "U1"),
        np.empty((0, 2), np.int32),
        np.empty((0, 2), np.int32),
        np.empty(0, np.int32),
        np.empty((0, 2), np.int32),
        dict(rejections),
    )


def _grasp_condition(
    artifact: InteractionArtifact, clip: int, object_frame: int,
) -> np.ndarray:
    """The existing 18-value object/grasp condition used by funnel training."""
    hand = int(artifact.active_hands[clip])
    grasp_rotation = np.asarray(artifact.grasp_rotations_object[clip], np.float64)
    if not np.isfinite(grasp_rotation).all() or np.linalg.norm(grasp_rotation) < _ROTATION_EPSILON:
        raise ValueError("invalid_grasp")
    grasp_rotation /= np.linalg.norm(grasp_rotation)
    w, x, y, z = grasp_rotation
    rotation6 = np.asarray(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y + z * w),
            2 * (x * z - y * w), 2 * (x * y - z * w),
            1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
        ], np.float32,
    )
    approach = np.asarray(artifact.approach_directions_object[clip], np.float64)
    grasp_position = np.asarray(artifact.grasp_positions_object[clip], np.float32)
    dimensions = np.asarray(artifact.object_dimensions[clip], np.float32)
    planar_norm = np.linalg.norm(approach[[0, 2]])
    if (
        not np.isfinite(approach).all()
        or not np.isfinite(grasp_position).all()
        or not np.isfinite(dimensions).all()
        or np.any(dimensions <= 0.0)
        or planar_norm < _ROTATION_EPSILON
    ):
        raise ValueError("invalid_grasp")
    support_height = float(
        artifact.object_positions[object_frame, 1] - artifact.table_positions[clip, 1]
    )
    grasp_height = float(grasp_position[1])
    return np.asarray(
        [
            1.0 if hand == int(InteractionHand.LEFT) else 0.0,
            1.0 if hand == int(InteractionHand.RIGHT) else 0.0,
            *grasp_position,
            *rotation6,
            float(approach[0] / planar_norm),
            float(approach[2] / planar_norm),
            *dimensions,
            support_height,
            grasp_height,
        ],
        np.float32,
    )


def _yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, np.float64)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _route_temporal(
    artifact: InteractionArtifact,
    source: slice,
    object_frame: int,
) -> np.ndarray:
    """Object-local root route and the five explicit source phase channels."""
    object_position = np.asarray(artifact.object_positions[object_frame], np.float64)
    object_yaw = _yaw(artifact.object_rotations[object_frame])
    root = np.asarray(artifact.positions[source, 0], np.float64)
    delta = root - object_position
    c, s = np.cos(object_yaw), np.sin(object_yaw)
    route = np.empty((len(root), 4), np.float32)
    route[:, 0] = c * delta[:, 0] + s * delta[:, 2]
    route[:, 1] = -s * delta[:, 0] + c * delta[:, 2]
    root_yaw = np.asarray([_yaw(q) for q in artifact.rotations[source, 0]])
    route[:, 2] = np.sin(root_yaw - object_yaw)
    route[:, 3] = np.cos(root_yaw - object_yaw)

    source_phases = np.asarray(artifact.phases[source], np.int64)
    one_hot = np.zeros((len(root), 5), np.float32)
    # Interaction packs begin in APPROACH; WALK is reserved for native walk rows.
    one_hot[np.arange(len(root)), source_phases + 1] = 1.0
    return np.concatenate((route, one_hot), axis=1)


def _metadata_object_ids(
    metadata: Sequence[Mapping[str, object]] | None,
    indices: list[int],
) -> np.ndarray:
    if metadata is None:
        return np.full(len(indices), "", dtype="U1")
    if len(metadata) == 0:
        raise ValueError("clip metadata must cover every artifact clip")
    values: list[str] = []
    for index in indices:
        if index >= len(metadata):
            raise ValueError("clip metadata must cover every artifact clip")
        object_id = metadata[index].get("object_id")
        sequence_id = metadata[index].get("sequence_id")
        if not isinstance(object_id, str) or not object_id or not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("clip metadata requires nonempty sequence_id and object_id")
        values.append(object_id)
    return np.asarray(values)


def extract_interaction_pairs(
    artifact: InteractionArtifact,
    clip_metadata: Sequence[Mapping[str, object]] | None = None,
) -> OverlapDataset:
    """Extract only source-contiguous right-hand rows with a post-window Lift."""
    walks: list[np.ndarray] = []
    pickups: list[np.ndarray] = []
    static_conditions: list[np.ndarray] = []
    walk_temporal: list[np.ndarray] = []
    pickup_temporal: list[np.ndarray] = []
    indices: list[int] = []
    walk_ranges: list[tuple[int, int]] = []
    pickup_ranges: list[tuple[int, int]] = []
    continuation_indices: list[int] = []
    continuation_ranges: list[tuple[int, int]] = []
    rejections: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    for clip, (start, stop) in enumerate(zip(artifact.range_starts, artifact.range_stops)):
        start, stop = int(start), int(stop)
        phases = np.asarray(artifact.phases[start:stop], np.int64)
        reaches = np.flatnonzero(phases == int(InteractionPhase.REACH))
        if len(reaches) == 0:
            reject("missing_reach")
            continue
        reach = start + int(reaches[0])
        global_start, global_stop = reach - WALK_FRAMES, reach + (PICKUP_FRAMES - OVERLAP_FRAMES)
        if global_start < start or global_stop > stop:
            reject("short_clip")
            continue
        contacts = np.flatnonzero(phases == int(InteractionPhase.CONTACT))
        if len(contacts) == 0 or start + int(contacts[0]) > reach + OVERLAP_FRAMES + 9:
            reject("contact_after_reach_window")
            continue
        if int(artifact.active_hands[clip]) != int(InteractionHand.RIGHT):
            reject("left_hand")
            continue
        contact = start + int(contacts[0])
        try:
            grasp = _grasp_condition(artifact, clip, contact - 1)
        except ValueError:
            reject("invalid_grasp")
            continue
        lifts = np.flatnonzero(phases == int(InteractionPhase.LIFT))
        lift = start + int(lifts[0]) if len(lifts) else -1
        if lift < global_stop:
            reject("missing_continuation")
            continue

        whole_contacts = np.column_stack((
            artifact.foot_contacts[global_start:global_stop],
            artifact.hand_contacts[global_start:global_stop, int(InteractionHand.RIGHT)],
        ))
        encoded = encode_motion(
            artifact.positions[global_start:global_stop],
            artifact.rotations[global_start:global_stop],
            whole_contacts,
            artifact.positions[global_start, 0],
        )
        walk = np.ascontiguousarray(encoded[:WALK_FRAMES])
        pickup = np.ascontiguousarray(encoded[WALK_FRAMES - OVERLAP_FRAMES:])
        if walk[WALK_FRAMES - OVERLAP_FRAMES:].tobytes() != pickup[:OVERLAP_FRAMES].tobytes():
            raise AssertionError("interaction overlap is not byte-identical")

        route = _route_temporal(artifact, slice(global_start, global_stop), contact - 1)
        initial_velocity = np.concatenate((
            artifact.velocities[global_start, 0],
            artifact.angular_velocities[global_start, 0],
        )).astype(np.float32)
        static = np.concatenate((grasp, initial_velocity, np.array([1.0], np.float32)))
        if static.shape != (STATIC_CONDITION_DIM,) or not np.isfinite(static).all():
            reject("invalid_grasp")
            continue

        walks.append(walk)
        pickups.append(pickup)
        static_conditions.append(static)
        walk_temporal.append(route[:WALK_FRAMES])
        pickup_temporal.append(route[WALK_FRAMES - OVERLAP_FRAMES:])
        indices.append(clip)
        walk_ranges.append((reach - WALK_FRAMES - start, reach - start))
        pickup_ranges.append((reach - OVERLAP_FRAMES - start, global_stop - start))
        continuation_indices.append(clip)
        continuation_ranges.append((global_stop - start, lift + 1 - start))

    if not walks:
        return _empty_overlap_dataset(rejections)
    object_ids = _metadata_object_ids(clip_metadata, indices)
    return OverlapDataset(
        np.asarray(walks, np.float32),
        np.asarray(pickups, np.float32),
        np.asarray(static_conditions, np.float32),
        np.asarray(walk_temporal, np.float32),
        np.asarray(pickup_temporal, np.float32),
        np.asarray(indices, np.int32),
        object_ids,
        np.asarray(walk_ranges, np.int32),
        np.asarray(pickup_ranges, np.int32),
        np.asarray(continuation_indices, np.int32),
        np.asarray(continuation_ranges, np.int32),
        dict(sorted(rejections.items())),
    )


def load_native_g1_walk(source_npz: Path, g1_xml: Path) -> HoldenClip:
    """Convert the native 29-DoF G1 source through the canonical FK exporter."""
    source_npz, g1_xml = Path(source_npz), Path(g1_xml)
    with np.load(source_npz, allow_pickle=False) as raw:
        required = {
            "fps", "dof_names", "body_names", "dof_positions", "body_positions", "body_rotations",
        }
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(f"native G1 source is missing arrays: {missing}")
        fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
        dof_names = tuple(str(name) for name in raw["dof_names"])
        body_names = tuple(str(name) for name in raw["body_names"])
        dof_positions = np.asarray(raw["dof_positions"], np.float32)
        body_positions = np.asarray(raw["body_positions"], np.float32)
        body_rotations = np.asarray(raw["body_rotations"], np.float32)
    kinematics = G1Kinematics(str(g1_xml))
    joint_names = tuple(kinematics.model.joint(index).name for index in range(1, kinematics.model.njnt))
    model_body_names = tuple(kinematics.model.body(index).name for index in range(1, kinematics.model.nbody))
    if dof_names != joint_names or len(dof_names) != 29:
        raise ValueError("native G1 dof_names must exactly match MuJoCo joints 1..29")
    if body_names != model_body_names or len(body_names) != 30:
        raise ValueError("native G1 body_names must exactly match MuJoCo bodies 1..30")
    if (
        dof_positions.ndim != 2 or dof_positions.shape[1] != 29
        or body_positions.shape != (len(dof_positions), 30, 3)
        or body_rotations.shape != (len(dof_positions), 30, 4)
        or not all(np.isfinite(value).all() for value in (dof_positions, body_positions, body_rotations))
    ):
        raise ValueError("native G1 source has invalid 29-DoF/body arrays")
    quaternion_norms = np.linalg.norm(body_rotations[:, 0], axis=1)
    if np.any(np.abs(quaternion_norms - 1.0) > 1e-4):
        raise ValueError("native G1 root rotations must be WXYZ unit quaternions")
    qpos = np.broadcast_to(kinematics.keyframe_or_zero_qpos(), (len(dof_positions), 36)).copy()
    qpos[:, :3] = body_positions[:, 0]
    qpos[:, 3:7] = body_rotations[:, 0]
    qpos[:, 7:] = dof_positions
    source = SourceClip(
        source_npz.stem, fps, qpos.astype(np.float32),
        np.arange(len(qpos), dtype=np.int32), "flat",
    )
    clip, skeleton, _ = convert_source_clip(source, kinematics, target_fps=FPS)
    if skeleton.names != G1_SKELETON.names:
        raise ValueError("native G1 conversion did not produce canonical 31-bone order")
    clip.validate()
    return clip


def extract_walking_windows(clip: HoldenClip, stride: int = 10) -> np.ndarray:
    """Encode source-contained native G1 walking windows without retargeting."""
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    clip.validate()
    if clip.positions.shape[1] != BONE_COUNT:
        raise ValueError("walking clip must use the canonical 31-bone skeleton")
    rows = []
    for start in range(0, len(clip.positions) - WALK_FRAMES + 1, stride):
        stop = start + WALK_FRAMES
        rows.append(encode_motion(
            clip.positions[start:stop], clip.rotations[start:stop],
            np.column_stack((clip.contacts[start:stop], np.zeros(WALK_FRAMES, np.float32))),
            clip.positions[start, 0],
        ))
    if not rows:
        return np.empty((0, WALK_FRAMES, FRAME_DIM), np.float32)
    return np.asarray(rows, np.float32)
