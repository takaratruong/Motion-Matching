#!/usr/bin/env python3
"""Export an aligned 80-frame overlap-motion visualizer pose file."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resources.g1_interaction_builder import overlap_motion
from resources.g1_interaction_builder.artifacts import read_artifact_set
from resources.g1_interaction_builder.schema import G1_SKELETON


MAGIC = b"G1OVLP01"
SCHEMA_VERSION = 1
FPS = 25
FRAME_COUNT = 80
BONE_COUNT = 31
FOOT_CONTACT_COUNT = 2
FRAME_DIM = 195
G1_BONE_NAMES = tuple(G1_SKELETON.names)
G1_SKELETON_SIGNATURE = G1_SKELETON.signature()
_QUATERNION_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class DecodedPose:
    positions: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    rotations: np.ndarray
    contacts: np.ndarray


def _require_finite(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _yaw_quaternion(yaw_radians: float) -> np.ndarray:
    return np.asarray(
        [np.cos(0.5 * yaw_radians), 0.0, np.sin(0.5 * yaw_radians), 0.0],
        np.float32,
    )


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), axis=-1).astype(np.float32)


def _yaw_rotate(vectors: np.ndarray, yaw_radians: float) -> np.ndarray:
    vectors = np.asarray(vectors, np.float32)
    cosine, sine = np.cos(yaw_radians), np.sin(yaw_radians)
    result = vectors.copy()
    result[..., 0] = cosine * vectors[..., 0] + sine * vectors[..., 2]
    result[..., 2] = -sine * vectors[..., 0] + cosine * vectors[..., 2]
    return result


def decode_encoded_motion(
    encoded: np.ndarray,
    local_positions: np.ndarray,
    align_root_position: np.ndarray,
    align_root_yaw_radians: float,
) -> DecodedPose:
    """Decode a 195-channel timeline and place frame zero in a declared world frame."""
    encoded = _require_finite(encoded, (FRAME_COUNT, FRAME_DIM), "encoded")
    local_positions = _require_finite(local_positions, (BONE_COUNT, 3), "local_positions")
    align_root_position = _require_finite(
        align_root_position, (3,), "align_root_position"
    )
    if not np.isfinite(align_root_yaw_radians):
        raise ValueError("align_root_yaw_radians must be finite")
    with np.errstate(divide="ignore", invalid="ignore"):
        decoded = overlap_motion.decode_motion(
            encoded, local_positions, np.zeros(3, np.float32)
        )
    positions = np.asarray(decoded.positions, np.float32).copy()
    velocities = np.asarray(decoded.velocities, np.float32).copy()
    angular_velocities = np.asarray(decoded.angular_velocities, np.float32).copy()
    rotations = np.asarray(decoded.rotations, np.float32).copy()
    origin = positions[0, 0].copy()
    positions[:, 0] = _yaw_rotate(positions[:, 0] - origin, align_root_yaw_radians)
    positions[:, 0] += align_root_position
    velocities[:, 0] = _yaw_rotate(velocities[:, 0], align_root_yaw_radians)
    angular_velocities[:, 0] = _yaw_rotate(
        angular_velocities[:, 0], align_root_yaw_radians
    )
    rotations[:, 0] = _quat_multiply(
        _yaw_quaternion(align_root_yaw_radians), rotations[:, 0]
    )
    return DecodedPose(positions, velocities, angular_velocities, rotations, decoded.contacts)


def _validate_export_fields(
    positions: np.ndarray,
    velocities: np.ndarray,
    angular_velocities: np.ndarray,
    rotations: np.ndarray,
    contacts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = _require_finite(positions, (FRAME_COUNT, BONE_COUNT, 3), "positions")
    velocities = _require_finite(velocities, (FRAME_COUNT, BONE_COUNT, 3), "velocities")
    angular_velocities = _require_finite(
        angular_velocities, (FRAME_COUNT, BONE_COUNT, 3), "angular_velocities"
    )
    rotations = _require_finite(rotations, (FRAME_COUNT, BONE_COUNT, 4), "rotations")
    contacts = _require_finite(contacts, (FRAME_COUNT, 3), "contacts")
    norms = np.linalg.norm(rotations, axis=-1)
    if np.any(np.abs(norms - 1.0) > _QUATERNION_TOLERANCE):
        raise ValueError("rotations must be WXYZ unit quaternions")
    return positions, velocities, angular_velocities, rotations, contacts


def export_offline_overlap(
    output: Path,
    positions: np.ndarray,
    velocities: np.ndarray,
    angular_velocities: np.ndarray,
    rotations: np.ndarray,
    contacts: np.ndarray,
) -> None:
    """Atomically publish the strict little-endian visualizer schema."""
    positions, velocities, angular_velocities, rotations, contacts = _validate_export_fields(
        positions, velocities, angular_velocities, rotations, contacts
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(MAGIC)
            stream.write(np.asarray(
                [SCHEMA_VERSION, FPS, FRAME_COUNT, BONE_COUNT, FOOT_CONTACT_COUNT],
                dtype="<u4",
            ).tobytes())
            signature = G1_SKELETON_SIGNATURE.encode("ascii")
            stream.write(np.asarray([len(signature)], dtype="<u4").tobytes())
            stream.write(signature)
            for name in G1_BONE_NAMES:
                encoded_name = name.encode("ascii")
                stream.write(np.asarray([len(encoded_name)], dtype="<u4").tobytes())
                stream.write(encoded_name)
            for values in (positions, velocities, angular_velocities, rotations):
                stream.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
            stream.write(np.ascontiguousarray(contacts[:, :2] >= 0.5, dtype=np.uint8).tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_npz_array(path: Path, key: str | None, label: str) -> np.ndarray:
    path = Path(path)
    source = np.load(path, allow_pickle=False)
    if isinstance(source, np.ndarray):
        return np.asarray(source)
    try:
        selected = key
        if selected is None:
            if len(source.files) != 1:
                raise ValueError(f"{label} NPZ needs --{label}-key when it has multiple arrays")
            selected = source.files[0]
        if selected not in source.files:
            raise ValueError(f"{label} NPZ is missing key {selected!r}")
        return np.asarray(source[selected])
    finally:
        source.close()


def reconstruct_dataset_row(
    dataset_path: Path, interaction_pack: Path | None, split: str, row: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild one encoded 80-frame interaction row and its frozen local offsets."""
    prefix = f"{split}_"
    with np.load(dataset_path, allow_pickle=False) as dataset:
        required = (
            f"{prefix}walk_windows", f"{prefix}pickup_windows",
            f"{prefix}sequence_indices", f"{prefix}source_walk_ranges",
        )
        missing = [name for name in required if name not in dataset.files]
        if missing:
            raise ValueError(f"dataset is missing arrays: {missing}")
        walk = np.asarray(dataset[f"{prefix}walk_windows"])
        pickup = np.asarray(dataset[f"{prefix}pickup_windows"])
        sequence_indices = np.asarray(dataset[f"{prefix}sequence_indices"])
        source_walk_ranges = np.asarray(dataset[f"{prefix}source_walk_ranges"])
        canonical_local_positions = next((
            np.asarray(dataset[name], np.float32)
            for name in ("canonical_local_positions", "local_positions")
            if name in dataset.files
        ), None)
    if walk.ndim != 3 or pickup.ndim != 3 or walk.shape[1:] != (50, FRAME_DIM) or pickup.shape[1:] != (50, FRAME_DIM):
        raise ValueError("dataset overlap windows must have shape (N, 50, 195)")
    if row < 0 or row >= len(walk) or len(pickup) != len(walk):
        raise ValueError("dataset row is outside the selected split")
    if not np.array_equal(walk[row, 30:50], pickup[row, :20]):
        raise ValueError("dataset row does not preserve its byte-identical overlap")
    encoded = np.concatenate((walk[row], pickup[row, 20:50]), axis=0)
    if canonical_local_positions is not None:
        return encoded, _require_finite(
            canonical_local_positions, (BONE_COUNT, 3), "canonical_local_positions"
        )
    if interaction_pack is None:
        raise ValueError(
            "dataset has no canonical local positions; provide --interaction-pack "
            "to recover frozen local offsets"
        )
    artifact, _, _, _, _ = read_artifact_set(Path(interaction_pack))
    clip = int(sequence_indices[row])
    if clip < 0 or clip >= len(artifact.range_starts):
        raise ValueError("dataset row has an invalid interaction sequence index")
    source_start = int(source_walk_ranges[row, 0])
    source_stop = int(source_walk_ranges[row, 1])
    if source_stop - source_start != 50:
        raise ValueError("dataset row has an invalid source walk range")
    global_start = int(artifact.range_starts[clip]) + source_start
    local_positions = np.asarray(artifact.positions[global_start], np.float32)
    return encoded, local_positions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="overlap dataset NPZ for a ground-truth row")
    source.add_argument("--generated", type=Path, help="NPZ containing one generated [80,195] array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--align-root-position", type=float, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--align-root-yaw-degrees", type=float, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--interaction-pack", type=Path)
    parser.add_argument("--generated-key")
    parser.add_argument("--local-positions", type=Path)
    parser.add_argument("--local-positions-key")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.dataset is not None:
            encoded, local_positions = reconstruct_dataset_row(
                args.dataset, args.interaction_pack, args.split, args.row
            )
        else:
            if args.local_positions is None:
                raise ValueError("--generated requires --local-positions because 195 channels omit bones 2..30 offsets")
            encoded = _load_npz_array(args.generated, args.generated_key, "generated")
            local_positions = _load_npz_array(
                args.local_positions, args.local_positions_key, "local-positions"
            )
        decoded = decode_encoded_motion(
            encoded,
            local_positions,
            np.asarray(args.align_root_position, np.float32),
            np.deg2rad(args.align_root_yaw_degrees),
        )
        export_offline_overlap(
            args.output,
            decoded.positions,
            decoded.velocities,
            decoded.angular_velocities,
            decoded.rotations,
            decoded.contacts,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"export_g1_overlap_pose: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
