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
    STATIC_CONDITION_DIM,
    TEMPORAL_CONDITION_DIM,
    WALK_FRAMES,
    OverlapDataset,
    _yaw,
    extract_interaction_pairs,
    extract_walking_windows,
    load_native_g1_walk,
)
from resources.g1_terrain_builder.schema import HoldenClip


SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **values)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    clip: HoldenClip, stride: int, seed: int,
) -> np.ndarray:
    starts = np.arange(0, len(clip.positions) - WALK_FRAMES + 1, stride, dtype=np.int32)
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


def _walking_rows(clip: HoldenClip, *, stride: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows = extract_walking_windows(clip, stride=stride)
    starts = np.arange(0, len(clip.positions) - WALK_FRAMES + 1, stride, dtype=np.int32)
    selected = _coverage_balanced_walking_indices(clip, stride, seed)
    windows = windows[selected]
    starts = starts[selected]
    statics, temporal = [], []
    for start in starts:
        stop = int(start + WALK_FRAMES)
        root = clip.positions[start:stop, 0].astype(np.float64)
        terminal = root[-1]
        terminal_yaw = _yaw(clip.rotations[stop - 1, 0])
        delta = root - terminal
        root_yaw = np.asarray([_yaw(q) for q in clip.rotations[start:stop, 0]])
        route = np.column_stack((
            delta[:, 0], delta[:, 2],
            np.sin(root_yaw - terminal_yaw), np.cos(root_yaw - terminal_yaw),
        )).astype(np.float32)
        phases = np.zeros((WALK_FRAMES, 5), np.float32)
        phases[:, 0] = 1.0
        temporal.append(np.concatenate((route, phases), axis=1))
        linear = (clip.positions[min(start + 1, stop - 1), 0] - clip.positions[start, 0]) * FPS
        angular = np.zeros(3, np.float32)
        if stop - start > 1:
            q0, q1 = clip.rotations[start, 0], clip.rotations[start + 1, 0]
            delta = holden_quat.mul(q1, holden_quat.inv(q0))
            if np.linalg.norm(delta[1:]) >= 1e-8:
                angular[:] = holden_quat.to_scaled_angle_axis(delta) * FPS
        statics.append(np.concatenate((
            np.zeros(18, np.float32), linear.astype(np.float32), angular,
            np.array([0.0], np.float32),
        )))
    return (
        windows,
        np.asarray(statics, np.float32).reshape(-1, STATIC_CONDITION_DIM),
        np.asarray(temporal, np.float32).reshape(-1, WALK_FRAMES, TEMPORAL_CONDITION_DIM),
        np.column_stack((starts, starts + WALK_FRAMES)).astype(np.int32),
    )


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
    walking_windows, walking_static, walking_temporal, walking_ranges = _walking_rows(
        walking, stride=10, seed=seed
    )
    if len(walking_windows) == 0:
        raise ValueError("native G1 source produced no complete walking windows")

    interaction_split = _split_interaction_by_object(rows.object_ids, seed)
    walking_split = _split_indices(len(walking_windows), seed + 1)
    values: dict[str, np.ndarray] = {}
    _partition_overlap(values, rows, interaction_split)
    for name, indices in walking_split.items():
        values[f"{name}_walking_windows"] = walking_windows[indices]
        values[f"{name}_walking_static_conditions"] = walking_static[indices]
        values[f"{name}_walking_temporal"] = walking_temporal[indices]
        values[f"{name}_walking_source_ranges"] = walking_ranges[indices]

    train_frames = np.concatenate((
        values["train_walk_windows"], values["train_pickup_windows"], values["train_walking_windows"],
    ), axis=0)
    values["normalization_mean"] = train_frames.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    values["normalization_scale"] = np.maximum(
        train_frames.std(axis=(0, 1), dtype=np.float64), 1e-5
    ).astype(np.float32)
    _atomic_npz(output, values)

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
        "fps": FPS,
        "interaction_rows": int(len(rows.walk_windows)),
        "walking_rows": int(len(walking_windows)),
        "split_interaction_rows": {name: int(len(indices)) for name, indices in interaction_split.items()},
        "split_interaction_object_counts": {
            name: int(len(np.unique(rows.object_ids[indices])))
            for name, indices in interaction_split.items()
        },
        "split_walking_rows": {name: int(len(indices)) for name, indices in walking_split.items()},
        "overlap_byte_identical": True,
        "rejection_counts": rows.rejection_counts,
        "normalization_partition": "train",
        "source_hashes": source_hashes,
        "continuations": continuation_records,
    }
    _atomic_json(output.with_suffix(".manifest.json"), audit)
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
