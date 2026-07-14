from collections.abc import Sequence
import dataclasses
import json
import math
import os
from pathlib import Path
import shutil
import struct
from typing import BinaryIO

import numpy as np

from resources.g1_terrain_builder.schema import SkeletonSpec

from .features import FEATURE_GROUPS, build_features
from .schema import (
    EvaluationSplit,
    FeatureSet,
    G1_SKELETON,
    InteractionArtifact,
    InteractionValidationError,
    LabeledInteractionClip,
)
from .splits import partition_clips


DB_MAGIC = b"G1INTDB1"
FEATURE_MAGIC = b"G1INTFT1"
VERSION = 1
ENDIAN_MARKER = 0x01020304

_DB_HEADER = struct.Struct("<8s8I")
_FEATURE_HEADER = struct.Struct("<8s5I")
_FEATURE_DIMENSION = 71
_FEATURE_GROUP_COUNT = 5
_BONE_COUNT = 31
_HAND_DOF_COUNT = 14


def assemble_database(
    clips: Sequence[LabeledInteractionClip],
    skeleton: SkeletonSpec,
) -> InteractionArtifact:
    ordered = _sort_unique_clips(clips)
    return _assemble_ordered_database(ordered, skeleton)


def _sort_unique_clips(
    clips: Sequence[LabeledInteractionClip],
) -> tuple[LabeledInteractionClip, ...]:
    ordered = tuple(
        sorted(clips, key=lambda clip: clip.motion.sequence_id)
    )
    if not ordered:
        raise ValueError("cannot assemble an empty interaction database")
    duplicates = sorted(
        {
            ordered[index].motion.sequence_id
            for index in range(1, len(ordered))
            if (
                ordered[index - 1].motion.sequence_id
                == ordered[index].motion.sequence_id
            )
        }
    )
    if duplicates:
        raise InteractionValidationError(
            "duplicate_sequence_id",
            f"duplicate sequence_id values: {duplicates!r}",
        )
    return ordered


def _assemble_ordered_database(
    ordered: tuple[LabeledInteractionClip, ...],
    skeleton: SkeletonSpec,
) -> InteractionArtifact:
    if skeleton.signature() != G1_SKELETON.signature():
        raise InteractionValidationError(
            "skeleton_mismatch", skeleton.signature()
        )

    lengths = np.array(
        [len(clip.motion.positions) for clip in ordered], np.int32
    )
    stops = np.cumsum(lengths, dtype=np.int32)
    starts = np.r_[np.int32(0), stops[:-1]].astype(np.int32)

    def frames(name: str) -> np.ndarray:
        return np.concatenate(
            [getattr(clip.motion, name) for clip in ordered], axis=0
        )

    artifact = InteractionArtifact(
        fps=25,
        parents=skeleton.parents.astype(np.int32, copy=True),
        range_starts=starts,
        range_stops=stops,
        positions=frames("positions"),
        velocities=frames("velocities"),
        rotations=frames("rotations"),
        angular_velocities=frames("angular_velocities"),
        foot_contacts=frames("foot_contacts"),
        hand_contacts=frames("hand_contacts"),
        hand_dof=frames("hand_dof"),
        hand_dof_velocities=frames("hand_dof_velocities"),
        phases=np.concatenate([clip.phases for clip in ordered]),
        active_hands=np.array(
            [int(clip.active_hand) for clip in ordered], np.uint8
        ),
        time_to_contact=np.concatenate(
            [clip.time_to_contact for clip in ordered]
        ).astype(np.float32),
        object_positions=frames("object_positions"),
        object_rotations=frames("object_rotations"),
        object_velocities=frames("object_velocities"),
        object_angular_velocities=frames("object_angular_velocities"),
        table_positions=np.stack(
            [clip.motion.table_position for clip in ordered]
        ),
        table_rotations=np.stack(
            [clip.motion.table_rotation for clip in ordered]
        ),
        table_sizes=np.stack([clip.motion.table_size for clip in ordered]),
        object_dimensions=np.stack(
            [clip.motion.object_dimensions for clip in ordered]
        ),
        grasp_positions_object=np.stack(
            [clip.grasp_position_object for clip in ordered]
        ),
        grasp_rotations_object=np.stack(
            [clip.grasp_rotation_object for clip in ordered]
        ),
        approach_directions_object=np.stack(
            [clip.approach_direction_object for clip in ordered]
        ),
        source_frames=frames("source_frames"),
    )
    artifact.validate()
    return artifact


def prepare_artifacts(
    clips: Sequence[LabeledInteractionClip],
    split: EvaluationSplit,
    skeleton: SkeletonSpec,
) -> tuple[
    tuple[LabeledInteractionClip, ...],
    InteractionArtifact,
    FeatureSet,
]:
    database_clips, _ = partition_clips(clips, split)
    ordered = _sort_unique_clips(database_clips)
    artifact = _assemble_ordered_database(ordered, skeleton)
    features = build_features(ordered, skeleton)
    return ordered, artifact, features


def _write_array(
    stream: BinaryIO,
    value: np.ndarray,
    dtype: str,
) -> None:
    source = np.asarray(value)
    if not source.flags.c_contiguous:
        raise ValueError("cannot write a non-C-contiguous array")
    output_dtype = np.dtype(dtype).newbyteorder("<")
    array = np.asarray(source, dtype=output_dtype, order="C")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("cannot write a non-finite array")
    stream.write(array.tobytes(order="C"))


def _read_exact(stream: BinaryIO, count: int, label: str) -> bytes:
    if count < 0:
        raise ValueError(f"invalid byte count for {label}: {count}")
    position = stream.tell()
    end = stream.seek(0, os.SEEK_END)
    stream.seek(position, os.SEEK_SET)
    if count > end - position:
        raise ValueError(f"truncated {label}")
    value = stream.read(count)
    if len(value) != count:
        raise ValueError(f"truncated {label}")
    return value


def _read_array(
    stream: BinaryIO,
    shape: tuple[int, ...],
    dtype: str,
    label: str,
) -> np.ndarray:
    count = math.prod(shape)
    file_dtype = np.dtype(dtype).newbyteorder("<")
    byte_count = count * file_dtype.itemsize
    data = _read_exact(stream, byte_count, label)
    return np.frombuffer(data, dtype=file_dtype).astype(
        np.dtype(dtype), copy=True
    ).reshape(shape)


def _write_database(path: Path, value: InteractionArtifact) -> None:
    value.validate()
    frame_count, bone_count = value.positions.shape[:2]
    clip_count = len(value.range_starts)
    with path.open("wb") as stream:
        stream.write(
            _DB_HEADER.pack(
                DB_MAGIC,
                VERSION,
                ENDIAN_MARKER,
                25,
                1,
                frame_count,
                bone_count,
                clip_count,
                _HAND_DOF_COUNT,
            )
        )
        for array, dtype in (
            (value.parents, "i4"),
            (value.range_starts, "i4"),
            (value.range_stops, "i4"),
            (value.positions, "f4"),
            (value.velocities, "f4"),
            (value.rotations, "f4"),
            (value.angular_velocities, "f4"),
            (value.foot_contacts, "u1"),
            (value.hand_contacts, "u1"),
            (value.hand_dof, "f4"),
            (value.hand_dof_velocities, "f4"),
            (value.phases, "u1"),
            (value.active_hands, "u1"),
            (value.time_to_contact, "f4"),
            (value.object_positions, "f4"),
            (value.object_rotations, "f4"),
            (value.object_velocities, "f4"),
            (value.object_angular_velocities, "f4"),
            (value.table_positions, "f4"),
            (value.table_rotations, "f4"),
            (value.table_sizes, "f4"),
            (value.object_dimensions, "f4"),
            (value.grasp_positions_object, "f4"),
            (value.grasp_rotations_object, "f4"),
            (value.approach_directions_object, "f4"),
            (value.source_frames, "i4"),
        ):
            _write_array(stream, array, dtype)


def _validate_database_header(
    header: tuple[bytes, int, int, int, int, int, int, int, int],
    file_size: int,
) -> tuple[int, int]:
    (
        magic,
        version,
        endian_marker,
        fps_numerator,
        fps_denominator,
        frame_count,
        bone_count,
        clip_count,
        hand_dof_count,
    ) = header
    if magic != DB_MAGIC:
        raise ValueError(
            f"invalid database magic: expected {DB_MAGIC!r}, got {magic!r}"
        )
    if version != VERSION:
        raise ValueError(
            f"unsupported database version {version}; expected {VERSION}"
        )
    if endian_marker != ENDIAN_MARKER:
        raise ValueError(
            "invalid database endian marker "
            f"0x{endian_marker:08x}; expected 0x{ENDIAN_MARKER:08x}"
        )
    if (fps_numerator, fps_denominator) != (25, 1):
        raise ValueError(
            "invalid database fps "
            f"{fps_numerator}/{fps_denominator}; expected 25/1"
        )
    if frame_count == 0:
        raise ValueError("invalid database frame count 0")
    if frame_count > file_size:
        raise ValueError(
            f"database frame count {frame_count} exceeds file size "
            f"{file_size}"
        )
    if bone_count != _BONE_COUNT:
        raise ValueError(
            f"invalid database bone count {bone_count}; expected 31"
        )
    if clip_count == 0:
        raise ValueError("invalid database clip count 0")
    if clip_count > file_size:
        raise ValueError(
            f"database clip count {clip_count} exceeds file size {file_size}"
        )
    if hand_dof_count != _HAND_DOF_COUNT:
        raise ValueError(
            "invalid database hand dof count "
            f"{hand_dof_count}; expected 14"
        )
    return frame_count, clip_count


def _read_database(path: Path) -> InteractionArtifact:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = _DB_HEADER.unpack(
            _read_exact(stream, _DB_HEADER.size, "database header")
        )
        frame_count, clip_count = _validate_database_header(
            header, file_size
        )
        artifact = InteractionArtifact(
            fps=25,
            parents=_read_array(stream, (31,), "i4", "parents"),
            range_starts=_read_array(
                stream, (clip_count,), "i4", "range_starts"
            ),
            range_stops=_read_array(
                stream, (clip_count,), "i4", "range_stops"
            ),
            positions=_read_array(
                stream,
                (frame_count, 31, 3),
                "f4",
                "positions",
            ),
            velocities=_read_array(
                stream,
                (frame_count, 31, 3),
                "f4",
                "velocities",
            ),
            rotations=_read_array(
                stream,
                (frame_count, 31, 4),
                "f4",
                "rotations",
            ),
            angular_velocities=_read_array(
                stream,
                (frame_count, 31, 3),
                "f4",
                "angular_velocities",
            ),
            foot_contacts=_read_array(
                stream, (frame_count, 2), "u1", "foot_contacts"
            ),
            hand_contacts=_read_array(
                stream, (frame_count, 2), "u1", "hand_contacts"
            ),
            hand_dof=_read_array(
                stream,
                (frame_count, 14),
                "f4",
                "hand_dof",
            ),
            hand_dof_velocities=_read_array(
                stream,
                (frame_count, 14),
                "f4",
                "hand_dof_velocities",
            ),
            phases=_read_array(
                stream, (frame_count,), "u1", "phases"
            ),
            active_hands=_read_array(
                stream, (clip_count,), "u1", "active_hands"
            ),
            time_to_contact=_read_array(
                stream,
                (frame_count,),
                "f4",
                "time_to_contact",
            ),
            object_positions=_read_array(
                stream,
                (frame_count, 3),
                "f4",
                "object_positions",
            ),
            object_rotations=_read_array(
                stream,
                (frame_count, 4),
                "f4",
                "object_rotations",
            ),
            object_velocities=_read_array(
                stream,
                (frame_count, 3),
                "f4",
                "object_velocities",
            ),
            object_angular_velocities=_read_array(
                stream,
                (frame_count, 3),
                "f4",
                "object_angular_velocities",
            ),
            table_positions=_read_array(
                stream,
                (clip_count, 3),
                "f4",
                "table_positions",
            ),
            table_rotations=_read_array(
                stream,
                (clip_count, 4),
                "f4",
                "table_rotations",
            ),
            table_sizes=_read_array(
                stream, (clip_count, 3), "f4", "table_sizes"
            ),
            object_dimensions=_read_array(
                stream,
                (clip_count, 3),
                "f4",
                "object_dimensions",
            ),
            grasp_positions_object=_read_array(
                stream,
                (clip_count, 3),
                "f4",
                "grasp_positions_object",
            ),
            grasp_rotations_object=_read_array(
                stream,
                (clip_count, 4),
                "f4",
                "grasp_rotations_object",
            ),
            approach_directions_object=_read_array(
                stream,
                (clip_count, 3),
                "f4",
                "approach_directions_object",
            ),
            source_frames=_read_array(
                stream,
                (frame_count,),
                "i4",
                "source_frames",
            ),
        )
        if stream.read(1) != b"":
            raise ValueError("database has trailing bytes")
    artifact.validate()
    return artifact


def _validate_features(
    value: FeatureSet,
    expected_frames: int | None = None,
) -> None:
    values = np.asarray(value.values)
    if values.ndim != 2 or values.shape[1:] != (_FEATURE_DIMENSION,):
        raise InteractionValidationError(
            "feature_shape",
            "feature dimension must be exactly 71; "
            f"got shape {values.shape}",
        )
    if values.shape[0] == 0:
        raise InteractionValidationError(
            "feature_shape", "feature values must contain at least one frame"
        )
    if expected_frames is not None and len(values) != expected_frames:
        raise InteractionValidationError(
            "feature_frame_count",
            "feature/database frame count mismatch: "
            f"{len(values)} != {expected_frames}",
        )
    arrays = (
        ("feature values", values, values.shape),
        (
            "feature offsets",
            np.asarray(value.offsets),
            (_FEATURE_DIMENSION,),
        ),
        (
            "feature scales",
            np.asarray(value.scales),
            (_FEATURE_DIMENSION,),
        ),
    )
    for name, array, shape in arrays:
        if array.shape != shape:
            raise InteractionValidationError(
                "feature_shape",
                f"{name} shape must be {shape}, got {array.shape}",
            )
        if array.dtype != np.dtype(np.float32):
            raise InteractionValidationError(
                "feature_dtype",
                f"{name} dtype must be float32, got {array.dtype}",
            )
        if not np.isfinite(array).all():
            raise InteractionValidationError(
                "non_finite_features",
                f"{name} contains non-finite values",
            )
    if not np.all(np.asarray(value.scales) > 0):
        raise InteractionValidationError(
            "feature_scales", "feature scales must be positive"
        )
    if tuple(value.groups) != FEATURE_GROUPS:
        raise InteractionValidationError(
            "feature_groups",
            "feature groups must match the exact schema-v1 groups",
        )


def _write_features(path: Path, value: FeatureSet) -> None:
    _validate_features(value)
    with path.open("wb") as stream:
        stream.write(
            _FEATURE_HEADER.pack(
                FEATURE_MAGIC,
                VERSION,
                ENDIAN_MARKER,
                len(value.values),
                value.values.shape[1],
                len(value.groups),
            )
        )
        _write_array(
            stream, [group.start for group in value.groups], "u4"
        )
        _write_array(
            stream, [group.stop for group in value.groups], "u4"
        )
        _write_array(stream, value.offsets, "f4")
        _write_array(stream, value.scales, "f4")
        _write_array(stream, value.values, "f4")


def _validate_feature_header(
    header: tuple[bytes, int, int, int, int, int],
    file_size: int,
) -> tuple[int, int]:
    magic, version, endian_marker, frame_count, dimension, group_count = header
    if magic != FEATURE_MAGIC:
        raise ValueError(
            f"invalid feature magic: expected {FEATURE_MAGIC!r}, got {magic!r}"
        )
    if version != VERSION:
        raise ValueError(
            f"unsupported feature version {version}; expected {VERSION}"
        )
    if endian_marker != ENDIAN_MARKER:
        raise ValueError(
            "invalid feature endian marker "
            f"0x{endian_marker:08x}; expected 0x{ENDIAN_MARKER:08x}"
        )
    if frame_count == 0:
        raise ValueError("invalid feature frame count 0")
    if frame_count > file_size:
        raise ValueError(
            f"feature frame count {frame_count} exceeds file size {file_size}"
        )
    if dimension != _FEATURE_DIMENSION:
        raise ValueError(
            f"invalid feature dimension {dimension}; expected 71"
        )
    if group_count != _FEATURE_GROUP_COUNT:
        raise ValueError(
            f"invalid feature group count {group_count}; expected 5"
        )
    return frame_count, group_count


def _read_features(path: Path) -> FeatureSet:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = _FEATURE_HEADER.unpack(
            _read_exact(stream, _FEATURE_HEADER.size, "feature header")
        )
        frame_count, group_count = _validate_feature_header(
            header, file_size
        )
        group_starts = _read_array(
            stream, (group_count,), "u4", "feature group starts"
        )
        group_stops = _read_array(
            stream, (group_count,), "u4", "feature group stops"
        )
        expected_starts = np.array(
            [group.start for group in FEATURE_GROUPS], np.uint32
        )
        expected_stops = np.array(
            [group.stop for group in FEATURE_GROUPS], np.uint32
        )
        if not (
            np.array_equal(group_starts, expected_starts)
            and np.array_equal(group_stops, expected_stops)
        ):
            raise ValueError(
                "invalid feature group ranges; expected "
                "[0,33), [33,45), [45,57), [57,65), [65,71)"
            )
        offsets = _read_array(
            stream, (71,), "f4", "feature offsets"
        )
        scales = _read_array(stream, (71,), "f4", "feature scales")
        values = _read_array(
            stream,
            (frame_count, 71),
            "f4",
            "feature values",
        )
        if stream.read(1) != b"":
            raise ValueError("feature file has trailing bytes")
    feature_set = FeatureSet(values, offsets, scales, FEATURE_GROUPS)
    _validate_features(feature_set)
    return feature_set


def _json_text(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except ValueError as error:
        if "Out of range float values" in str(error):
            raise ValueError(
                "cannot serialize a non-finite JSON value"
            ) from error
        raise
    return encoded + "\n"


def _write_json(path: Path, value: object) -> None:
    _write_text(path, _json_text(value))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite constant {value}")


def _require_finite_json(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} JSON contains a non-finite number")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite_json(child, label)
    elif isinstance(value, list):
        for child in value:
            _require_finite_json(child, label)


def _read_json(path: Path, label: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label} JSON: expected an object")
    _require_finite_json(value, label)
    return value


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_version(value: dict, label: str) -> None:
    version = value.get("schema_version")
    if not _is_integer(version) or version != VERSION:
        raise ValueError(
            f"{label} schema_version must be integer 1, got {version!r}"
        )


_CLIP_COUNT_FIELDS = (
    "source_clips",
    "included_clips",
    "rejected_clips",
)


def _validate_clip_counts(
    value: dict,
    label: str,
) -> dict[str, int]:
    counts = {}
    for name in _CLIP_COUNT_FIELDS:
        count = value.get(name)
        if not _is_integer(count) or count < 0:
            raise ValueError(
                f"{label} {name} must be a nonnegative integer, "
                f"got {count!r}"
            )
        counts[name] = count
    if counts["source_clips"] != (
        counts["included_clips"] + counts["rejected_clips"]
    ):
        raise ValueError(
            f"{label} source_clips must equal included_clips plus "
            "rejected_clips"
        )
    return counts


def _validate_manifest(
    manifest: dict,
    artifact: InteractionArtifact,
) -> None:
    _require_version(manifest, "manifest")
    counts = _validate_clip_counts(manifest, "manifest")
    target_fps = manifest.get("target_fps")
    if (
        isinstance(target_fps, bool)
        or not isinstance(target_fps, (int, float))
        or not math.isfinite(target_fps)
        or target_fps != 25
    ):
        raise ValueError(
            f"manifest target_fps must be 25, got {target_fps!r}"
        )
    signature = manifest.get("skeleton_signature")
    if signature != G1_SKELETON.signature():
        raise ValueError(
            "manifest skeleton_signature must match the exact G1 skeleton"
        )
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise ValueError("manifest clips must be an array")
    if len(clips) != len(artifact.range_starts):
        raise ValueError(
            "manifest clip count does not match database clip count"
        )
    if len(clips) > counts["included_clips"]:
        raise ValueError(
            "database artifact clip count must not exceed manifest "
            f"included_clips: {len(clips)} > {counts['included_clips']}"
        )
    ranges = []
    sequence_ids = set()
    ordered_sequence_ids = []
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise ValueError(f"manifest clip {index} must be an object")
        for field in ("sequence_id", "object_id"):
            item = clip.get(field)
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"manifest clip {index} {field} must be a nonempty string"
                )
        if clip["sequence_id"] in sequence_ids:
            raise ValueError(
                f"manifest contains duplicate sequence_id {clip['sequence_id']!r}"
            )
        sequence_ids.add(clip["sequence_id"])
        ordered_sequence_ids.append(clip["sequence_id"])
        start = clip.get("range_start")
        stop = clip.get("range_stop")
        if not _is_integer(start) or not _is_integer(stop):
            raise ValueError(
                f"manifest clip {index} range must contain integer bounds"
            )
        ranges.append((start, stop))
    if ordered_sequence_ids != sorted(ordered_sequence_ids):
        raise ValueError(
            "manifest sequence_ids must be lexicographically sorted"
        )
    expected_ranges = list(
        zip(
            artifact.range_starts.astype(int).tolist(),
            artifact.range_stops.astype(int).tolist(),
        )
    )
    if ranges != expected_ranges:
        raise ValueError(
            "manifest clip ranges do not match database ranges"
        )


def _validate_split(split: dict, manifest: dict) -> None:
    seed = split.get("seed")
    if not _is_integer(seed) or seed < 0:
        raise ValueError(
            f"evaluation split seed must be a nonnegative integer, got {seed!r}"
        )
    partitions = {}
    for name in ("database_objects", "heldout_objects"):
        values = split.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"evaluation split {name} must be a nonempty array")
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError(
                f"evaluation split {name} must contain nonempty strings"
            )
        if values != sorted(values):
            raise ValueError(f"evaluation split {name} must be sorted")
        if len(values) != len(set(values)):
            raise ValueError(
                f"evaluation split {name} must not contain duplicates"
            )
        partitions[name] = set(values)
    overlap = partitions["database_objects"] & partitions["heldout_objects"]
    if overlap:
        raise ValueError(
            "evaluation split database and heldout objects overlap: "
            f"{sorted(overlap)!r}"
        )
    manifest_objects = {clip["object_id"] for clip in manifest["clips"]}
    database_objects = partitions["database_objects"]
    if manifest_objects != database_objects:
        missing = sorted(database_objects - manifest_objects)
        unexpected = sorted(manifest_objects - database_objects)
        raise ValueError(
            "manifest database object identities must exactly match "
            "evaluation split database_objects; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _validate_report(
    report: dict,
    frame_count: int,
    manifest: dict,
) -> None:
    _require_version(report, "validation report")
    report_counts = _validate_clip_counts(report, "validation report")
    manifest_counts = _validate_clip_counts(manifest, "manifest")
    for name in _CLIP_COUNT_FIELDS:
        if report_counts[name] != manifest_counts[name]:
            raise ValueError(
                f"manifest/report {name} mismatch: "
                f"{manifest_counts[name]} != {report_counts[name]}"
            )
    included_frames = report.get("included_frames")
    if not _is_integer(included_frames) or included_frames != frame_count:
        raise ValueError(
            "validation report included_frames must match database frame "
            f"count {frame_count}, got {included_frames!r}"
        )
    histogram = report.get("rejections_by_code")
    if not isinstance(histogram, dict):
        raise ValueError(
            "validation report rejections_by_code must be an object"
        )
    for code, count in histogram.items():
        if not isinstance(code, str) or not code:
            raise ValueError(
                "validation report rejections_by_code keys must be "
                "nonempty strings"
            )
        if not _is_integer(count) or count <= 0:
            raise ValueError(
                "validation report rejections_by_code values must be "
                "positive integers"
            )
    rejections = report.get("rejections")
    if not isinstance(rejections, list):
        raise ValueError("validation report rejections must be an array")
    if len(rejections) != report_counts["rejected_clips"]:
        raise ValueError(
            "validation report rejections length must equal "
            f"rejected_clips: {len(rejections)} != "
            f"{report_counts['rejected_clips']}"
        )
    rejection_fields = (
        "sequence_id",
        "object_id",
        "stage",
        "code",
        "message",
    )
    rejection_field_set = set(rejection_fields)
    rejection_keys = []
    codes = []
    for index, rejection in enumerate(rejections):
        if not isinstance(rejection, dict):
            raise ValueError(
                f"validation report rejection {index} must be an object"
            )
        actual_fields = set(rejection)
        if actual_fields != rejection_field_set:
            missing = sorted(rejection_field_set - actual_fields)
            extra = sorted(actual_fields - rejection_field_set)
            raise ValueError(
                f"validation report rejection {index} fields must be "
                "exactly sequence_id, object_id, stage, code, message; "
                f"missing={missing!r}, extra={extra!r}"
            )
        for field in rejection_fields:
            item = rejection[field]
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"validation report rejection {index} {field} must be "
                    "a nonempty string"
                )
        key = (
            rejection["sequence_id"],
            rejection["stage"],
            rejection["code"],
        )
        rejection_keys.append(key)
        codes.append(rejection["code"])
    if rejection_keys != sorted(rejection_keys):
        raise ValueError(
            "validation report rejections must be sorted by "
            "(sequence_id, stage, code)"
        )
    expected_histogram = {}
    for code in codes:
        expected_histogram[code] = expected_histogram.get(code, 0) + 1
    if histogram != expected_histogram:
        raise ValueError(
            "validation report rejections_by_code must equal the exact "
            f"rejection histogram: expected {expected_histogram!r}, "
            f"got {histogram!r}"
        )
    bounds = report.get("numeric_bounds")
    if not isinstance(bounds, dict):
        raise ValueError("validation report numeric_bounds must be an object")
    for name in (
        "fk_max_error_m",
        "fk_rotation_max_error_degrees",
        "duration_max_error_s",
        "quaternion_norm_max_error",
    ):
        value = bounds.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "validation report numeric bound "
                f"{name} must be a finite nonnegative number"
            )


def _canonical_json(value: object, label: str) -> tuple[str, dict]:
    text = _json_text(value)
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} JSON must encode an object")
    return text, decoded


def _write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _remove_private_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _artifact_set_validation_error(path: Path) -> Exception | None:
    try:
        read_artifact_set(path)
    except (OSError, TypeError, ValueError) as error:
        return error
    return None


def _recover_interrupted_publish(
    output: Path,
    temporary: Path,
) -> None:
    previous_paths = sorted(
        output.parent.glob(f".{output.name}.previous-*")
    )
    if not previous_paths:
        return

    if output.exists():
        output_error = _artifact_set_validation_error(output)
        if output_error is None:
            for path in previous_paths:
                _remove_private_path(path)
            return
    else:
        output_error = None

    valid_previous = []
    previous_errors = []
    for path in previous_paths:
        error = _artifact_set_validation_error(path)
        if error is None:
            valid_previous.append(path)
        else:
            previous_errors.append(f"{path.name}: {error}")
    if not valid_previous:
        detail = "; ".join(previous_errors)
        if output_error is not None:
            detail = f"public artifact: {output_error}; {detail}"
        raise ValueError(
            f"cannot recover a valid previous artifact for {output}: "
            f"{detail}"
        )

    recovery = valid_previous[-1]
    if output.exists():
        _remove_private_path(temporary)
        os.replace(output, temporary)
        try:
            os.replace(recovery, output)
        except BaseException:
            if not output.exists() and temporary.exists():
                os.replace(temporary, output)
            raise
        _remove_private_path(temporary)
    else:
        os.replace(recovery, output)

    for path in previous_paths:
        _remove_private_path(path)


def write_artifact_set(
    output: Path,
    artifact: InteractionArtifact,
    features: FeatureSet,
    split: EvaluationSplit,
    manifest: dict,
    report: dict,
) -> None:
    output = Path(output)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    previous = output.parent / f".{output.name}.previous-{os.getpid()}"
    _recover_interrupted_publish(output, temporary)

    artifact.validate()
    _validate_features(features, len(artifact.positions))
    manifest_text, manifest_value = _canonical_json(manifest, "manifest")
    split_text, split_value = _canonical_json(
        dataclasses.asdict(split), "evaluation split"
    )
    report_text, report_value = _canonical_json(
        report, "validation report"
    )
    _validate_manifest(manifest_value, artifact)
    _validate_split(split_value, manifest_value)
    _validate_report(report_value, len(artifact.positions), manifest_value)

    if output.exists() and not output.is_dir():
        raise ValueError(f"artifact output is not a directory: {output}")
    _remove_private_path(temporary)
    temporary.mkdir(parents=True)
    moved_previous = False
    published = False
    try:
        _write_database(
            temporary / "interaction_database.bin", artifact
        )
        _write_features(
            temporary / "interaction_features.bin", features
        )
        _write_text(temporary / "manifest.json", manifest_text)
        _write_text(temporary / "evaluation_split.json", split_text)
        _write_text(temporary / "validation_report.json", report_text)
        read_artifact_set(temporary)

        if output.exists():
            os.replace(output, previous)
            moved_previous = True
        os.replace(temporary, output)
        published = True
        if moved_previous:
            shutil.rmtree(previous)
    except BaseException:
        if moved_previous and previous.exists():
            if published and output.exists():
                os.replace(output, temporary)
                published = False
            if not output.exists():
                os.replace(previous, output)
                moved_previous = False
        _remove_private_path(temporary)
        if not moved_previous:
            _remove_private_path(previous)
        raise


def read_artifact_set(
    output: Path,
) -> tuple[InteractionArtifact, FeatureSet, dict, dict, dict]:
    output = Path(output)
    if not output.is_dir():
        raise ValueError(f"artifact output is not a directory: {output}")
    artifact = _read_database(output / "interaction_database.bin")
    features = _read_features(output / "interaction_features.bin")
    if len(features.values) != len(artifact.positions):
        raise ValueError(
            "feature/database frame count mismatch: "
            f"{len(features.values)} != {len(artifact.positions)}"
        )
    manifest = _read_json(output / "manifest.json", "manifest")
    split = _read_json(
        output / "evaluation_split.json", "evaluation split"
    )
    report = _read_json(
        output / "validation_report.json", "validation report"
    )
    _validate_manifest(manifest, artifact)
    _validate_split(split, manifest)
    _validate_report(report, len(artifact.positions), manifest)
    return artifact, features, manifest, split, report
