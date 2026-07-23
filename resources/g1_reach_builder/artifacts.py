from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct

import numpy as np

from resources.g1_interaction_builder.schema import G1_SKELETON

from .motions import CanonicalReach, ReachAugmentation


DATABASE_MAGIC = b"G1RCHD2\0"
FEATURE_MAGIC = b"G1RCHF2\0"
VERSION = 2
ENDIAN_MARKER = 0x01020304
DATABASE_HEADER = struct.Struct("<8s8I")
FEATURE_HEADER = struct.Struct("<8s4I")


@dataclass
class ReachArtifact:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    source_frames: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contact_frames: np.ndarray
    active_hands: np.ndarray
    augmentations: np.ndarray
    source_indices: np.ndarray
    original_indices: np.ndarray
    endpoint_positions: np.ndarray
    endpoint_rotations: np.ndarray
    approach_directions: np.ndarray
    source_names: tuple[str, ...]

    def validate(self) -> None:
        frames = len(self.positions)
        clips = len(self.range_starts)
        expected = (
            (self.positions, (frames, 31, 3), np.float32, "positions"),
            (self.velocities, (frames, 31, 3), np.float32, "velocities"),
            (self.rotations, (frames, 31, 4), np.float32, "rotations"),
            (
                self.angular_velocities,
                (frames, 31, 3),
                np.float32,
                "angular velocities",
            ),
            (self.foot_contacts, (frames, 2), np.uint8, "foot contacts"),
            (self.source_frames, (frames,), np.int32, "source frames"),
            (self.range_starts, (clips,), np.int32, "range starts"),
            (self.range_stops, (clips,), np.int32, "range stops"),
            (self.contact_frames, (clips,), np.int32, "contact frames"),
            (self.active_hands, (clips,), np.uint8, "active hands"),
            (self.augmentations, (clips,), np.uint8, "augmentations"),
            (self.source_indices, (clips,), np.uint32, "source indices"),
            (self.original_indices, (clips,), np.int32, "original indices"),
            (
                self.endpoint_positions,
                (clips, 3),
                np.float32,
                "endpoint positions",
            ),
            (
                self.endpoint_rotations,
                (clips, 4),
                np.float32,
                "endpoint rotations",
            ),
            (
                self.approach_directions,
                (clips, 3),
                np.float32,
                "approach directions",
            ),
        )
        if frames == 0 or clips == 0:
            raise ValueError("reach artifact must contain frames and clips")
        for value, shape, dtype, label in expected:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError(
                    f"reach {label} must have shape {shape} and dtype {np.dtype(dtype)}"
                )
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"reach {label} contains non-finite values")
        if self.range_starts[0] != 0 or self.range_stops[-1] != frames:
            raise ValueError("reach ranges do not cover all frames")
        if np.any(self.range_starts >= self.range_stops):
            raise ValueError("reach ranges contain an empty clip")
        if np.any(self.range_starts[1:] != self.range_stops[:-1]):
            raise ValueError("reach ranges must be contiguous")
        if np.any(self.contact_frames < self.range_starts) or np.any(
            self.contact_frames >= self.range_stops
        ):
            raise ValueError("reach contact frames are outside clip ranges")
        if np.any(self.active_hands > 1) or np.any(self.augmentations > 1):
            raise ValueError("reach hand or augmentation enum is invalid")
        if not self.source_names or len(set(self.source_names)) != len(self.source_names):
            raise ValueError("reach source names must be unique and nonempty")
        if any(not name for name in self.source_names):
            raise ValueError("reach source name is empty")
        if np.any(self.source_indices >= len(self.source_names)):
            raise ValueError("reach source index is out of range")
        for clip, augmentation in enumerate(self.augmentations):
            original = int(self.original_indices[clip])
            if augmentation == int(ReachAugmentation.CAPTURED) and original != -1:
                raise ValueError("captured reach has synthetic original index")
            if augmentation == int(ReachAugmentation.MIRRORED):
                if original < 0 or original >= clips:
                    raise ValueError("mirrored reach original index is invalid")
                if self.augmentations[original] != int(ReachAugmentation.CAPTURED):
                    raise ValueError("mirrored reach original is not captured")
        if np.max(np.abs(np.linalg.norm(self.rotations, axis=-1) - 1.0)) > 1e-4:
            raise ValueError("reach pose quaternion norm error")
        if np.max(np.abs(np.linalg.norm(self.endpoint_rotations, axis=-1) - 1.0)) > 1e-4:
            raise ValueError("reach endpoint quaternion norm error")
        if np.max(np.abs(np.linalg.norm(self.approach_directions, axis=-1) - 1.0)) > 1e-4:
            raise ValueError("reach approach direction norm error")


@dataclass
class ReachFeatures:
    values: np.ndarray

    def validate(self, clip_count: int | None = None) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != 10:
            raise ValueError("reach features must have shape (C, 10)")
        if self.values.dtype != np.float32 or not np.isfinite(self.values).all():
            raise ValueError("reach features must be finite float32")
        if clip_count is not None and len(self.values) != clip_count:
            raise ValueError("reach feature clip count mismatch")


def assemble_reach_pack(
    reaches: list[CanonicalReach],
) -> tuple[ReachArtifact, ReachFeatures]:
    if not reaches:
        raise ValueError("cannot assemble an empty reach pack")
    for reach in reaches:
        reach.validate()
    ids = [reach.reach_id for reach in reaches]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate reach id")
    id_to_index = {value: index for index, value in enumerate(ids)}
    source_names = tuple(dict.fromkeys(reach.sequence_id for reach in reaches))
    source_to_index = {value: index for index, value in enumerate(source_names)}
    counts = np.asarray([reach.frame_count for reach in reaches], np.int32)
    stops = np.cumsum(counts, dtype=np.int32)
    starts = np.concatenate((np.array([0], np.int32), stops[:-1]))
    contact_frames = starts + np.asarray(
        [reach.contact_index for reach in reaches], np.int32
    )
    artifact = ReachArtifact(
        positions=np.concatenate([reach.positions for reach in reaches]).astype(np.float32),
        velocities=np.concatenate([reach.velocities for reach in reaches]).astype(np.float32),
        rotations=np.concatenate([reach.rotations for reach in reaches]).astype(np.float32),
        angular_velocities=np.concatenate(
            [reach.angular_velocities for reach in reaches]
        ).astype(np.float32),
        foot_contacts=np.concatenate(
            [reach.foot_contacts for reach in reaches]
        ).astype(np.uint8),
        source_frames=np.concatenate(
            [reach.source_frames for reach in reaches]
        ).astype(np.int32),
        range_starts=starts,
        range_stops=stops,
        contact_frames=contact_frames,
        active_hands=np.asarray([int(reach.active_hand) for reach in reaches], np.uint8),
        augmentations=np.asarray(
            [int(reach.augmentation) for reach in reaches], np.uint8
        ),
        source_indices=np.asarray(
            [source_to_index[reach.sequence_id] for reach in reaches], np.uint32
        ),
        original_indices=np.asarray([
            -1 if reach.original_reach_id is None
            else id_to_index[reach.original_reach_id]
            for reach in reaches
        ], np.int32),
        endpoint_positions=np.stack(
            [reach.endpoint_position_root for reach in reaches]
        ).astype(np.float32),
        endpoint_rotations=np.stack(
            [reach.endpoint_rotation_root_wxyz for reach in reaches]
        ).astype(np.float32),
        approach_directions=np.stack(
            [reach.approach_direction_root for reach in reaches]
        ).astype(np.float32),
        source_names=source_names,
    )
    features = ReachFeatures(np.concatenate((
        artifact.endpoint_positions,
        artifact.approach_directions,
        artifact.endpoint_rotations,
    ), axis=1).astype(np.float32))
    artifact.validate()
    features.validate(len(reaches))
    return artifact, features


def _bytes(array: np.ndarray, dtype: str) -> bytes:
    return np.asarray(array, dtype=np.dtype(dtype)).tobytes(order="C")


def _write_database(path: Path, value: ReachArtifact) -> None:
    value.validate()
    frames = len(value.positions)
    clips = len(value.range_starts)
    with path.open("wb") as stream:
        stream.write(DATABASE_HEADER.pack(
            DATABASE_MAGIC, VERSION, ENDIAN_MARKER, 25, 1,
            frames, 31, clips, len(value.source_names),
        ))
        for data, dtype in (
            (G1_SKELETON.parents, "<i4"),
            (value.range_starts, "<i4"),
            (value.range_stops, "<i4"),
            (value.contact_frames, "<i4"),
            (value.positions, "<f4"),
            (value.velocities, "<f4"),
            (value.rotations, "<f4"),
            (value.angular_velocities, "<f4"),
            (value.foot_contacts, "u1"),
            (value.active_hands, "u1"),
            (value.augmentations, "u1"),
            (value.source_indices, "<u4"),
            (value.original_indices, "<i4"),
            (value.endpoint_positions, "<f4"),
            (value.endpoint_rotations, "<f4"),
            (value.approach_directions, "<f4"),
            (value.source_frames, "<i4"),
        ):
            stream.write(_bytes(data, dtype))
        for name in value.source_names:
            encoded = name.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)


def _write_features(path: Path, value: ReachFeatures) -> None:
    value.validate()
    with path.open("wb") as stream:
        stream.write(FEATURE_HEADER.pack(
            FEATURE_MAGIC, VERSION, ENDIAN_MARKER, len(value.values), 10
        ))
        stream.write(_bytes(value.values, "<f4"))


def _take(stream: io.BytesIO, count: int, dtype: str, shape, label: str):
    itemsize = np.dtype(dtype).itemsize
    byte_count = count * itemsize
    remaining = stream.getbuffer().nbytes - stream.tell()
    if byte_count < 0 or byte_count > remaining:
        raise ValueError(f"truncated reach {label}")
    data = stream.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"truncated reach {label}")
    return np.frombuffer(data, dtype=dtype).reshape(shape).astype(
        np.dtype(dtype).newbyteorder("="), copy=True
    )


def _read_database(path: Path) -> ReachArtifact:
    stream = io.BytesIO(path.read_bytes())
    header = stream.read(DATABASE_HEADER.size)
    if len(header) != DATABASE_HEADER.size:
        raise ValueError("truncated reach database header")
    magic, version, endian, fps_num, fps_den, frames, bones, clips, sources = (
        DATABASE_HEADER.unpack(header)
    )
    if (magic, version, endian, fps_num, fps_den, bones) != (
        DATABASE_MAGIC, VERSION, ENDIAN_MARKER, 25, 1, 31
    ):
        raise ValueError("invalid reach database header")
    if frames == 0 or clips == 0 or sources == 0:
        raise ValueError("invalid zero reach database count")
    wire_parents = _take(stream, 31, "<i4", (31,), "parents")
    if not np.array_equal(wire_parents, G1_SKELETON.parents):
        raise ValueError("reach skeleton parents mismatch")
    kwargs = {
        "range_starts": _take(stream, clips, "<i4", (clips,), "range starts"),
        "range_stops": _take(stream, clips, "<i4", (clips,), "range stops"),
        "contact_frames": _take(stream, clips, "<i4", (clips,), "contact frames"),
        "positions": _take(stream, frames * 31 * 3, "<f4", (frames, 31, 3), "positions"),
        "velocities": _take(stream, frames * 31 * 3, "<f4", (frames, 31, 3), "velocities"),
        "rotations": _take(stream, frames * 31 * 4, "<f4", (frames, 31, 4), "rotations"),
        "angular_velocities": _take(stream, frames * 31 * 3, "<f4", (frames, 31, 3), "angular velocities"),
        "foot_contacts": _take(stream, frames * 2, "u1", (frames, 2), "foot contacts"),
        "active_hands": _take(stream, clips, "u1", (clips,), "active hands"),
        "augmentations": _take(stream, clips, "u1", (clips,), "augmentations"),
        "source_indices": _take(stream, clips, "<u4", (clips,), "source indices"),
        "original_indices": _take(stream, clips, "<i4", (clips,), "original indices"),
        "endpoint_positions": _take(stream, clips * 3, "<f4", (clips, 3), "endpoint positions"),
        "endpoint_rotations": _take(stream, clips * 4, "<f4", (clips, 4), "endpoint rotations"),
        "approach_directions": _take(stream, clips * 3, "<f4", (clips, 3), "approach directions"),
        "source_frames": _take(stream, frames, "<i4", (frames,), "source frames"),
    }
    names = []
    for _ in range(sources):
        length_data = stream.read(4)
        if len(length_data) != 4:
            raise ValueError("truncated reach source name length")
        length = struct.unpack("<I", length_data)[0]
        if length == 0 or length > 4096:
            raise ValueError("invalid reach source name length")
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError("truncated reach source name")
        try:
            names.append(encoded.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 reach source name") from error
    if stream.read(1):
        raise ValueError("reach database has trailing bytes")
    artifact = ReachArtifact(source_names=tuple(names), **kwargs)
    artifact.validate()
    return artifact


def _read_features(path: Path) -> ReachFeatures:
    stream = io.BytesIO(path.read_bytes())
    header = stream.read(FEATURE_HEADER.size)
    if len(header) != FEATURE_HEADER.size:
        raise ValueError("truncated reach feature header")
    magic, version, endian, clips, dimension = FEATURE_HEADER.unpack(header)
    if (magic, version, endian, dimension) != (
        FEATURE_MAGIC, VERSION, ENDIAN_MARKER, 10
    ) or clips == 0:
        raise ValueError("invalid reach feature header")
    values = _take(stream, clips * 10, "<f4", (clips, 10), "features")
    if stream.read(1):
        raise ValueError("reach features have trailing bytes")
    result = ReachFeatures(values)
    result.validate()
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"expected private pack directory, got {path}")
        shutil.rmtree(path)


def write_reach_pack(
    output: Path,
    artifact: ReachArtifact,
    features: ReachFeatures,
    manifest: dict,
) -> None:
    artifact.validate()
    features.validate(len(artifact.range_starts))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    previous = output.parent / f".{output.name}.previous-{os.getpid()}"
    _remove_directory(temporary)
    _remove_directory(previous)
    temporary.mkdir()
    try:
        _write_database(temporary / "reach_database.bin", artifact)
        _write_features(temporary / "reach_features.bin", features)
        stops = artifact.range_stops
        contact_frames = artifact.contact_frames
        value = dict(manifest)
        value.update({
            "schema": "g1-reach-pack",
            "version": VERSION,
            "database_magic": DATABASE_MAGIC.rstrip(b"\0").decode("ascii"),
            "feature_magic": FEATURE_MAGIC.rstrip(b"\0").decode("ascii"),
            "target_fps": 25.0,
            "skeleton_names": list(G1_SKELETON.names),
            "skeleton_parents": G1_SKELETON.parents.astype(int).tolist(),
            "skeleton_signature": G1_SKELETON.signature(),
            "frame_count": len(artifact.positions),
            "total_reaches": len(artifact.range_starts),
            "source_names": list(artifact.source_names),
            "feature_dimension": 10,
            "paired_returns": int(np.sum(stops - contact_frames - 1 > 0)),
            "unavailable_returns": int(np.sum(stops - contact_frames - 1 == 0)),
            "return_frame_count": int(np.sum(stops - contact_frames - 1)),
            "minimum_return_frames": int(np.min(stops - contact_frames - 1)),
            "maximum_return_frames": int(np.max(stops - contact_frames - 1)),
            "reach_database_sha256": _sha256(temporary / "reach_database.bin"),
            "reach_features_sha256": _sha256(temporary / "reach_features.bin"),
        })
        (temporary / "manifest.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        read_reach_pack(temporary)
        if output.exists():
            if not output.is_dir():
                raise ValueError(f"reach pack output is not a directory: {output}")
            os.replace(output, previous)
        try:
            os.replace(temporary, output)
        except BaseException:
            if previous.exists() and not output.exists():
                os.replace(previous, output)
            raise
        _remove_directory(previous)
    except BaseException:
        _remove_directory(temporary)
        raise


def read_reach_pack(
    output: Path,
) -> tuple[ReachArtifact, ReachFeatures, dict]:
    output = Path(output)
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    database_path = output / "reach_database.bin"
    feature_path = output / "reach_features.bin"
    if _sha256(database_path) != manifest.get("reach_database_sha256"):
        raise ValueError("reach database checksum mismatch")
    if _sha256(feature_path) != manifest.get("reach_features_sha256"):
        raise ValueError("reach feature checksum mismatch")
    artifact = _read_database(database_path)
    features = _read_features(feature_path)
    features.validate(len(artifact.range_starts))
    if (
        manifest.get("schema") != "g1-reach-pack"
        or manifest.get("version") != VERSION
    ):
        raise ValueError("unsupported reach pack manifest")
    if manifest.get("frame_count") != len(artifact.positions):
        raise ValueError("reach manifest frame count mismatch")
    if manifest.get("total_reaches") != len(artifact.range_starts):
        raise ValueError("reach manifest clip count mismatch")
    if manifest.get("source_names") != list(artifact.source_names):
        raise ValueError("reach manifest source names mismatch")
    captured = int(np.sum(
        artifact.augmentations == int(ReachAugmentation.CAPTURED)
    ))
    mirrored = int(np.sum(
        artifact.augmentations == int(ReachAugmentation.MIRRORED)
    ))
    if manifest.get("captured_reaches") != captured:
        raise ValueError("reach manifest captured reach count mismatch")
    if manifest.get("mirrored_reaches") != mirrored:
        raise ValueError("reach manifest mirrored reach count mismatch")
    return_frames = artifact.range_stops - artifact.contact_frames - 1
    return_metadata = {
        "paired_returns": int(np.sum(return_frames > 0)),
        "unavailable_returns": int(np.sum(return_frames == 0)),
        "return_frame_count": int(np.sum(return_frames)),
        "minimum_return_frames": int(np.min(return_frames)),
        "maximum_return_frames": int(np.max(return_frames)),
    }
    for field, expected in return_metadata.items():
        if manifest.get(field) != expected:
            raise ValueError(f"reach manifest {field} mismatch")
    expected_features = np.concatenate((
        artifact.endpoint_positions,
        artifact.approach_directions,
        artifact.endpoint_rotations,
    ), axis=1)
    if features.values.tobytes() != expected_features.tobytes():
        raise ValueError("reach feature values do not match database endpoints")
    return artifact, features, manifest
