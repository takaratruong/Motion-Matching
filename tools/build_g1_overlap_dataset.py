#!/usr/bin/env python3
"""Build deterministic native-G1 walk/pickup overlap training data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.artifacts import read_artifact_set
from resources.g1_interaction_builder.overlap_motion import (
    FPS,
    FRAME_DIM,
    PICKUP_FRAMES,
    WALK_FRAMES,
    OverlapDataset,
    _encode_walking_window,
    _yaw,
    _yaw_quaternion,
    extract_interaction_pairs,
    load_native_g1_walk,
)
from resources.g1_interaction_builder.schema import G1_SKELETON
from resources.g1_terrain_builder.schema import HoldenClip


SPLITS = ("train", "validation", "test")
_NATIVE_OFFSET_TOLERANCE_M = 0.002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_npz(path: Path, values: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values)
    return temporary


def _stage_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return temporary


def _temporary_path(parent: Path) -> Path:
    with tempfile.NamedTemporaryFile(dir=parent, delete=False) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    return temporary


def _atomic_dataset_pair(path: Path, values: dict[str, np.ndarray], manifest: dict) -> None:
    """Publish the dataset and manifest together, restoring a prior pair on error."""
    manifest_path = path.with_suffix(".manifest.json")
    data_stage = _stage_npz(path, values)
    manifest_stage = _stage_json(manifest_path, manifest)
    data_backup = manifest_backup = None
    had_data, had_manifest = path.exists(), manifest_path.exists()
    if had_data != had_manifest:
        data_stage.unlink(missing_ok=True)
        manifest_stage.unlink(missing_ok=True)
        raise RuntimeError("dataset publication requires a complete prior output pair")
    try:
        if had_data:
            data_backup, manifest_backup = _temporary_path(path.parent), _temporary_path(path.parent)
            try:
                os.replace(path, data_backup)
                os.replace(manifest_path, manifest_backup)
            except Exception:
                if data_backup.exists():
                    os.replace(data_backup, path)
                if manifest_backup.exists():
                    os.replace(manifest_backup, manifest_path)
                raise
        try:
            os.replace(data_stage, path)
            os.replace(manifest_stage, manifest_path)
        except Exception:
            path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            if had_data:
                os.replace(data_backup, path)
                os.replace(manifest_backup, manifest_path)
            raise
    finally:
        data_stage.unlink(missing_ok=True)
        manifest_stage.unlink(missing_ok=True)
        if data_backup is not None:
            data_backup.unlink(missing_ok=True)
        if manifest_backup is not None:
            manifest_backup.unlink(missing_ok=True)


def _split_indices(count: int, seed: int) -> dict[str, np.ndarray]:
    if count == 0:
        return {name: np.empty(0, np.int64) for name in SPLITS}
    order = np.random.default_rng(seed).permutation(count)
    if count < 3:
        sizes = (count, 0, 0)
    else:
        validation = max(1, int(round(count * 0.10)))
        test = max(1, int(round(count * 0.10)))
        train = count - validation - test
        if train < 1:
            train, validation, test = 1, 1, count - 2
        sizes = (train, validation, test)
    cursor = 0
    result = {}
    for name, size in zip(SPLITS, sizes):
        result[name] = np.sort(order[cursor:cursor + size])
        cursor += size
    return result


def _coverage_balanced_walking_indices(
    clip: HoldenClip, stride: int, seed: int, starts: np.ndarray | None = None,
) -> np.ndarray:
    if starts is None:
        starts = np.arange(0, len(clip.positions) - WALK_FRAMES + 1, stride, dtype=np.int32)
    starts = np.asarray(starts, np.int32)
    if len(starts) == 0:
        return starts
    root = clip.positions[:, 0].astype(np.float64)
    velocity = np.zeros_like(root)
    velocity[1:] = np.diff(root, axis=0) * FPS
    yaw = np.asarray([_yaw(q) for q in clip.rotations[:, 0]], np.float64)
    yaw_rate = np.zeros(len(yaw), np.float64)
    yaw_rate[1:] = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))) * FPS
    speed_bin = np.floor(
        np.linalg.norm(velocity[starts][:, [0, 2]], axis=1) / 0.20
    ).astype(np.int32)
    yaw_bin = np.floor(np.abs(yaw_rate[starts]) / 0.50).astype(np.int32)
    groups: dict[tuple[int, int], list[int]] = {}
    for row, key in enumerate(zip(speed_bin, yaw_bin)):
        groups.setdefault(key, []).append(row)
    target = max(len(rows) for rows in groups.values())
    generator = np.random.default_rng(seed)
    selected = []
    for key in sorted(groups):
        rows = np.asarray(groups[key], np.int32)
        selected.extend(rows.tolist())
        if len(rows) < target:
            selected.extend(generator.choice(rows, size=target - len(rows), replace=True).tolist())
    return generator.permutation(np.asarray(selected, np.int32))


def _split_interaction_by_object(
    object_ids: np.ndarray, seed: int,
) -> dict[str, np.ndarray]:
    object_ids = np.asarray(object_ids)
    if object_ids.ndim != 1 or len(object_ids) == 0 or not np.all(object_ids != ""):
        raise ValueError("interaction object_ids must be a nonempty identity vector")
    identities = np.unique(object_ids)
    groups = _split_indices(len(identities), seed)
    return {
        name: np.flatnonzero(np.isin(object_ids, identities[indices]))
        for name, indices in groups.items()
    }


def _walking_row(clip: HoldenClip, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one source-contained row in its terminal-root goal frame."""
    stop = int(start + WALK_FRAMES)
    window = _encode_walking_window(clip, int(start))
    root = clip.positions[start:stop, 0].astype(np.float64)
    terminal_rotation = _yaw_quaternion(clip.rotations[stop - 1, 0])
    goal_root = holden_quat.inv_mul_vec(terminal_rotation, root - root[-1])
    terminal_yaw = _yaw(terminal_rotation)
    root_yaw = np.asarray([_yaw(q) for q in clip.rotations[start:stop, 0]])
    route = np.column_stack((
        goal_root[:, 0], goal_root[:, 2],
        np.sin(root_yaw - terminal_yaw), np.cos(root_yaw - terminal_yaw),
    )).astype(np.float32)
    phases = np.zeros((WALK_FRAMES, 5), np.float32)
    phases[:, 0] = 1.0
    temporal = np.concatenate((route, phases), axis=1)
    linear = (goal_root[min(1, WALK_FRAMES - 1)] - goal_root[0]) * FPS
    q0 = holden_quat.mul(holden_quat.inv(terminal_rotation), clip.rotations[start, 0])
    q1 = holden_quat.mul(holden_quat.inv(terminal_rotation), clip.rotations[start + 1, 0])
    rotation_delta = holden_quat.mul(q1, holden_quat.inv(q0))
    angular = np.zeros(3, np.float32)
    if np.linalg.norm(rotation_delta[1:]) >= 1e-8:
        angular[:] = holden_quat.to_scaled_angle_axis(rotation_delta) * FPS
    static = np.concatenate((
        np.zeros(18, np.float32), linear.astype(np.float32), angular.astype(np.float32),
        np.array([0.0], np.float32),
    ))
    return window, static, temporal


def _walking_source_regions(frame_count: int) -> dict[str, tuple[int, int]]:
    """Reserve disjoint source-frame regions with a full-window guard between them."""
    guard = WALK_FRAMES - 1
    usable = frame_count - 2 * guard
    if usable < 3 * WALK_FRAMES:
        raise ValueError("native G1 source is too short for three guarded walking partitions")
    held_out = WALK_FRAMES
    if usable >= 4 * WALK_FRAMES:
        held_out = 2 * WALK_FRAMES
    train = usable - 2 * held_out
    regions = {
        "train": (0, train),
        "validation": (train + guard, train + guard + held_out),
        "test": (train + held_out + 2 * guard, frame_count),
    }
    if any(stop - start < WALK_FRAMES for start, stop in regions.values()):
        raise AssertionError("walking source partition cannot contain one full window")
    return regions


def _partitioned_walking_rows(
    clip: HoldenClip, *, stride: int, seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Partition source time first, then balance windows only inside each partition."""
    partitions = {}
    for offset, (name, (region_start, region_stop)) in enumerate(_walking_source_regions(len(clip.positions)).items()):
        first = ((region_start + stride - 1) // stride) * stride
        starts = np.arange(first, region_stop - WALK_FRAMES + 1, stride, dtype=np.int32)
        selected = _coverage_balanced_walking_indices(clip, stride, seed + offset, starts)
        selected_starts = starts[selected]
        rows = [_walking_row(clip, int(start)) for start in selected_starts]
        windows = np.asarray([row[0] for row in rows], np.float32)
        statics = np.asarray([row[1] for row in rows], np.float32)
        temporal = np.asarray([row[2] for row in rows], np.float32)
        ranges = np.column_stack((selected_starts, selected_starts + WALK_FRAMES)).astype(np.int32)
        partitions[name] = (windows, statics, temporal, ranges)
    return partitions


def _assert_disjoint_walking_source_ranges(
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    seen_frames: set[int] = set()
    seen_ranges: set[tuple[int, int]] = set()
    for name in SPLITS:
        ranges = partitions[name][3]
        ranges_in_partition = {tuple(int(value) for value in item) for item in ranges}
        frames_in_partition = {
            frame for start, stop in ranges_in_partition for frame in range(start, stop)
        }
        if seen_ranges.intersection(ranges_in_partition) or seen_frames.intersection(frames_in_partition):
            raise AssertionError(f"walking source leakage into {name} partition")
        seen_ranges.update(ranges_in_partition)
        seen_frames.update(frames_in_partition)


def _manifest_metadata(manifest: dict, artifact) -> list[dict]:
    clips = manifest.get("clips")
    if not isinstance(clips, list) or len(clips) != len(artifact.range_starts):
        raise ValueError("interaction manifest must contain one metadata record per artifact clip")
    metadata = []
    for index, record in enumerate(clips):
        if not isinstance(record, dict):
            raise ValueError("interaction manifest clip metadata must be objects")
        if (record.get("range_start"), record.get("range_stop")) != (
            int(artifact.range_starts[index]), int(artifact.range_stops[index]),
        ):
            raise ValueError("interaction manifest range does not match artifact")
        metadata.append(record)
    return metadata


def _partition_overlap(values: dict[str, np.ndarray], rows: OverlapDataset, split: dict[str, np.ndarray]) -> None:
    fields = {
        "walk_windows": rows.walk_windows,
        "pickup_windows": rows.pickup_windows,
        "static_conditions": rows.static_conditions,
        "walk_temporal": rows.walk_temporal,
        "pickup_temporal": rows.pickup_temporal,
        "sequence_indices": rows.sequence_indices,
        "object_ids": rows.object_ids,
        "source_walk_ranges": rows.source_walk_ranges,
        "source_pickup_ranges": rows.source_pickup_ranges,
        "continuation_sequence_indices": rows.continuation_sequence_indices,
        "source_continuation_ranges": rows.source_continuation_ranges,
    }
    for name, indices in split.items():
        for field, source in fields.items():
            values[f"{name}_{field}"] = source[indices]


def _canonical_skeleton_metadata(walking: HoldenClip) -> dict[str, np.ndarray]:
    """Return the fixed hierarchy and native local offsets used by torch FK.

    The Simulation root and Hips translation are deliberately not offsets: both
    are dynamic channels in the 195-D motion representation.  Every remaining
    local translation must be a single canonical native-G1 value.
    """
    positions = np.asarray(walking.positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (len(G1_SKELETON.names), 3):
        raise ValueError("native G1 walk must contain canonical 31-bone local positions")
    # Resampling the native source and reconstructing local transforms is
    # numerically accurate to the existing 1 mm FK contract, not bit-exact.
    # Median offsets retain the native skeleton while a 2 mm bound rejects a
    # genuinely animated child translation.
    offsets = np.median(positions, axis=0)
    offsets[:2] = 0.0
    if not np.allclose(
        positions[:, 2:], offsets[None, 2:], rtol=0.0, atol=_NATIVE_OFFSET_TOLERANCE_M,
    ):
        raise ValueError("native G1 local offsets for bones 2..30 must be constant")
    if not np.isfinite(offsets).all():
        raise ValueError("canonical native G1 local offsets must be finite")
    return {
        "skeleton_parents": G1_SKELETON.parents.astype(np.int32, copy=True),
        "skeleton_names": np.asarray(G1_SKELETON.names),
        "skeleton_signature": np.asarray(G1_SKELETON.signature()),
        "canonical_local_offsets": offsets.astype(np.float32),
    }


def build_dataset(
    walking_source: Path,
    g1_xml: Path,
    interaction_pack: Path,
    output: Path,
    *,
    seed: int,
) -> dict:
    """Build and atomically publish one dataset plus its auditable manifest."""
    walking_source, g1_xml = Path(walking_source), Path(g1_xml)
    interaction_pack, output = Path(interaction_pack), Path(output)
    artifact, _, manifest, _, _ = read_artifact_set(interaction_pack)
    rows = extract_interaction_pairs(artifact, _manifest_metadata(manifest, artifact))
    if len(rows.walk_windows) == 0:
        raise ValueError("interaction pack produced no valid overlap rows")
    if rows.walk_windows[:, 30:50].tobytes() != rows.pickup_windows[:, :20].tobytes():
        raise AssertionError("interaction dataset overlap audit failed")
    walking = load_native_g1_walk(walking_source, g1_xml)
    skeleton_metadata = _canonical_skeleton_metadata(walking)
    walking_partitions = _partitioned_walking_rows(walking, stride=10, seed=seed)
    _assert_disjoint_walking_source_ranges(walking_partitions)
    if any(len(walking_partitions[name][0]) == 0 for name in SPLITS):
        raise ValueError("native G1 source produced an empty walking partition")

    interaction_split = _split_interaction_by_object(rows.object_ids, seed)
    values: dict[str, np.ndarray] = {}
    values.update(skeleton_metadata)
    _partition_overlap(values, rows, interaction_split)
    for name, (windows, statics, temporal, ranges) in walking_partitions.items():
        values[f"{name}_walking_windows"] = windows
        values[f"{name}_walking_static_conditions"] = statics
        values[f"{name}_walking_temporal"] = temporal
        values[f"{name}_walking_source_ranges"] = ranges

    train_frames = np.concatenate((
        values["train_walk_windows"], values["train_pickup_windows"], values["train_walking_windows"],
    ), axis=0)
    values["normalization_mean"] = train_frames.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    values["normalization_scale"] = np.maximum(
        train_frames.std(axis=(0, 1), dtype=np.float64), 1e-5
    ).astype(np.float32)
    continuation_records = []
    for row, clip in enumerate(rows.continuation_sequence_indices):
        record = manifest["clips"][int(clip)]
        continuation_records.append({
            "object_id": record["object_id"],
            "sequence_id": record["sequence_id"],
            "source_continuation_range": [int(v) for v in rows.source_continuation_ranges[row]],
            "source_pickup_range": [int(v) for v in rows.source_pickup_ranges[row]],
            "source_walk_range": [int(v) for v in rows.source_walk_ranges[row]],
        })
    source_hashes = {
        "walking_source": _sha256(walking_source),
        "g1_xml": _sha256(g1_xml),
        "interaction_database": _sha256(interaction_pack / "interaction_database.bin"),
        "interaction_features": _sha256(interaction_pack / "interaction_features.bin"),
        "interaction_manifest": _sha256(interaction_pack / "manifest.json"),
    }
    audit = {
        "frame_schema": FRAME_DIM,
        "skeleton_parents": skeleton_metadata["skeleton_parents"].astype(int).tolist(),
        "skeleton_names": skeleton_metadata["skeleton_names"].tolist(),
        "skeleton_signature": str(skeleton_metadata["skeleton_signature"].item()),
        "canonical_local_offsets": skeleton_metadata["canonical_local_offsets"].tolist(),
        "hips_channel_variance": train_frames[..., 3:6].var(axis=(0, 1), dtype=np.float64).tolist(),
        "fps": FPS,
        "interaction_rows": int(len(rows.walk_windows)),
        "walking_rows": int(sum(len(walking_partitions[name][0]) for name in SPLITS)),
        "split_interaction_rows": {name: int(len(indices)) for name, indices in interaction_split.items()},
        "split_interaction_object_counts": {
            name: int(len(np.unique(rows.object_ids[indices])))
            for name, indices in interaction_split.items()
        },
        "split_walking_rows": {
            name: int(len(walking_partitions[name][0])) for name in SPLITS
        },
        "walking_source_partitions_disjoint": True,
        "overlap_byte_identical": True,
        "rejection_counts": rows.rejection_counts,
        "normalization_partition": "train",
        "source_hashes": source_hashes,
        "continuations": continuation_records,
    }
    _atomic_dataset_pair(output, values, audit)
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-source", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--interaction-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    audit = build_dataset(
        args.walking_source, args.g1_xml, args.interaction_pack, args.output, seed=args.seed
    )
    summary = {key: value for key, value in audit.items() if key != "continuations"}
    summary["continuation_count"] = len(audit["continuations"])
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
