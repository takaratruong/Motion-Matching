"""Authenticated broad-corpus adapter for the hybrid terrain LMM.

The large Holden database and terrain sidecars remain immutable external
assets.  The local cache contains only rebuilt 31-D matching features and
small row/range lookup arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import (
    read_support_sidecar,
)
from resources.g1_terrain_builder.database import (
    derive_velocities,
    read_holden_database,
)
from resources.g1_terrain_builder.features import build_matching_features
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    ArtifactSet,
    FeatureSet,
)

EXPECTED_ROWS = 3_970_932
EXPECTED_RANGES = 15_815
EXPECTED_STRICT_RANGES = 15_816
TAKARA_HELDOUT_START = 15_688
EXPECTED_FAMILY_SOURCE_RANGE_COUNTS = (1, 1_769, 1_857, 12_188)
TAKARA_SOURCE_NAME = "takara_walk_50hz"
TAKARA_SOURCE_STOP = 17_432
FPS = 25.0
HORIZONS = (8, 17, 25)
TERRAIN_DISTANCES_M = (0.25, 0.5, 0.75, 1.0)
FAMILY_NAMES = ("flat", "curb", "slope", "stair")
_FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILY_NAMES)}
_TERRAIN_V2_COLUMNS = (1, 3, 5, 7)
_SUPPORT_COLUMNS = (
    "source_root_height_m",
    "source_left_toe_height_m",
    "source_right_toe_height_m",
)
_HEADER = struct.Struct("<4sIII")
_CACHE_SCHEMA = "g1-hybrid-terrain-lmm-corpus/v2-strict"
_SOURCE_RECEIPT_SCHEMA = "g1-hybrid-terrain-lmm-source-receipt/v1"
_SPLIT_CONTRACT = {
    "grail": "every-tenth-sorted-range-per-family-starting-at-zero",
    "takara": "final-ceil-ten-percent-temporal-interval",
    "takara_is_only_within_range_split": True,
}
_NORMALIZATION_CONTRACT = {
    "fit_rows": "train-only",
    "transform_rows": "all",
    "method": "31d-group-offset-and-scale",
}
_ARRAY_SPECS = {
    "features.npy": (np.float32, (None, 31)),
    "feature_offset.npy": (np.float32, (31,)),
    "feature_scale.npy": (np.float32, (31,)),
    "range_family_ids.npy": (np.int16, (None,)),
    "family_ids.npy": (np.int16, (None,)),
    "range_ids.npy": (np.int32, (None,)),
    "range_starts.npy": (np.int32, (None,)),
    "range_stops.npy": (np.int32, (None,)),
    "train_mask.npy": (np.bool_, (None,)),
    "evaluation_mask.npy": (np.bool_, (None,)),
}


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n").encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_descriptor(path: Path, schema: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"source asset is not a regular file: {path}")
    return {
        "path": str(path),
        "schema": schema,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _resolved_manifest_asset(root: Path, descriptor: Mapping[str, Any]) -> Path:
    relative = descriptor.get("path")
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError("manifest asset path must be a non-empty relative path")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("manifest asset path escapes the source root") from error
    if not path.is_file():
        raise ValueError(f"manifest asset is not a regular file: {path}")
    return path


def _authenticate_asset(
    root: Path, descriptor: Mapping[str, Any], label: str,
) -> tuple[Path, dict[str, Any]]:
    if type(descriptor) is not dict:
        raise ValueError(f"{label} descriptor must be an object")
    expected_sha = descriptor.get("sha256")
    if type(expected_sha) is not str or len(expected_sha) != 64:
        raise ValueError(f"{label} descriptor has an invalid SHA-256")
    path = _resolved_manifest_asset(root, descriptor)
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    receipt = {
        "path": str(path),
        "schema": descriptor.get("schema"),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha,
    }
    return path, receipt


def _read_g1tf_v2(path: Path) -> np.ndarray:
    """Read the bank's 12-D descriptor and select 0.25--1.00 m heights."""

    with path.open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError(f"{path}: truncated G1TF/v2 header")
        magic, version, frames, dimensions = _HEADER.unpack(header)
        if (magic, version, dimensions) != (b"G1TF", 2, 12):
            raise ValueError(f"{path}: expected exact G1TF/v2 with 12 dimensions")
        payload = stream.read()
    expected = frames * dimensions * np.dtype("<f4").itemsize
    if len(payload) != expected:
        raise ValueError(
            f"{path}: expected {expected} terrain bytes, got {len(payload)}"
        )
    raw = np.frombuffer(payload, dtype="<f4").reshape(frames, dimensions)
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: G1TF/v2 values must be finite")
    return np.ascontiguousarray(raw[:, _TERRAIN_V2_COLUMNS], dtype=np.float32)


def _read_terrain(path: Path, descriptor: Mapping[str, Any]) -> np.ndarray:
    schema = descriptor.get("schema")
    if schema == "G1TF/v2":
        return _read_g1tf_v2(path)
    raise ValueError(f"unsupported terrain sidecar schema: {schema!r}")


def _row_lookup(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    range_family_ids: np.ndarray,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    range_ids = np.empty(frames, np.int32)
    family_ids = np.empty(frames, np.int16)
    for range_id, (start, stop) in enumerate(zip(range_starts, range_stops)):
        range_ids[int(start):int(stop)] = range_id
        family_ids[int(start):int(stop)] = range_family_ids[range_id]
    return range_ids, family_ids


def _split_masks(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    range_family_ids: np.ndarray,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic masks with the documented Takara exception.

    Every tenth sorted GRAIL range in each non-flat family is held out whole.
    The single Takara range is split temporally: its final ceil(10%) is held
    out and its preceding rows train.  The two Takara intervals do not overlap.
    """

    evaluation = np.zeros(frames, np.bool_)
    flat_ranges = np.flatnonzero(range_family_ids == _FAMILY_TO_ID["flat"])
    if len(flat_ranges) != 2 or not np.array_equal(flat_ranges, [0, 1]):
        raise ValueError("strict corpus requires two leading Takara virtual ranges")
    evaluation[
        int(range_starts[flat_ranges[1]]):int(range_stops[flat_ranges[1]])
    ] = True

    for family in ("curb", "slope", "stair"):
        family_ranges = np.flatnonzero(
            range_family_ids == _FAMILY_TO_ID[family]
        )
        for range_id in family_ranges[::10]:
            start = int(range_starts[range_id])
            stop = int(range_stops[range_id])
            evaluation[start:stop] = True
    return ~evaluation, evaluation


def _refine_takara_range(
    artifacts: ArtifactSet,
    range_family_ids: np.ndarray,
) -> tuple[ArtifactSet, np.ndarray]:
    """Turn the temporal Takara split into two derivative-safe ranges."""

    artifacts.validate()
    families = np.ascontiguousarray(range_family_ids, dtype=np.int16)
    if families.shape != artifacts.range_starts.shape:
        raise ValueError("one family ID is required for every source range")
    flat = np.flatnonzero(families == _FAMILY_TO_ID["flat"])
    if len(flat) == 2:
        if not np.array_equal(flat, [0, 1]):
            raise ValueError("refined Takara ranges must be the first two ranges")
        for start, stop in zip(
            artifacts.range_starts[:2], artifacts.range_stops[:2]
        ):
            _recompute_segment_dynamics(artifacts, int(start), int(stop))
        return artifacts, families
    if len(flat) != 1 or int(flat[0]) != 0 \
            or int(artifacts.range_starts[0]) != 0:
        raise ValueError("source must contain one leading Takara flat range")

    stop = int(artifacts.range_stops[0])
    split = (
        TAKARA_HELDOUT_START
        if stop == TAKARA_SOURCE_STOP
        else stop - max(1, (stop + 9) // 10)
    )
    if split < 1 or stop - split < 1:
        raise ValueError("Takara virtual ranges must both be non-empty")
    for start, segment_stop in ((0, split), (split, stop)):
        _recompute_segment_dynamics(artifacts, start, segment_stop)
    artifacts.range_starts = np.insert(
        np.asarray(artifacts.range_starts, np.int32), 1, split
    ).astype(np.int32, copy=False)
    artifacts.range_stops = np.insert(
        np.asarray(artifacts.range_stops, np.int32), 0, split
    ).astype(np.int32, copy=False)
    families = np.insert(families, 1, families[0]).astype(np.int16, copy=False)
    artifacts.validate()
    return artifacts, np.ascontiguousarray(families)


def _recompute_segment_dynamics(
    artifacts: ArtifactSet, start: int, stop: int,
) -> None:
    positions = artifacts.positions[start:stop]
    rotations = artifacts.rotations[start:stop]
    if len(positions) >= 3:
        # Holden's branch-safe quaternion expression still evaluates the
        # unused zero-angle division before np.where selects its finite arm.
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity, angular = derive_velocities(positions, rotations, FPS)
    elif len(positions) == 2:
        linear_delta = (positions[1] - positions[0]) * FPS
        with np.errstate(divide="ignore", invalid="ignore"):
            angular_delta = holden_quat.to_scaled_angle_axis(holden_quat.abs(
                holden_quat.mul_inv(rotations[1], rotations[0])
            )) * FPS
        velocity = np.repeat(linear_delta[None], 2, axis=0).astype(np.float32)
        angular = np.repeat(angular_delta[None], 2, axis=0).astype(np.float32)
    else:
        velocity = np.zeros_like(positions, dtype=np.float32)
        angular = np.zeros_like(positions, dtype=np.float32)
    artifacts.velocities[start:stop] = velocity
    artifacts.angular_velocities[start:stop] = angular


@dataclass(frozen=True)
class HybridCorpus:
    artifacts: ArtifactSet
    features: FeatureSet
    range_family_ids: np.ndarray
    family_ids: np.ndarray
    range_ids: np.ndarray
    train_mask: np.ndarray
    evaluation_mask: np.ndarray
    source_root: Path
    manifest_receipt: Mapping[str, Any]
    source_assets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    family_names: tuple[str, ...] = FAMILY_NAMES
    cache_manifest_sha256: str | None = None

    def range_for_row(self, row: int) -> int:
        if type(row) is not int or row < 0 or row >= len(self.range_ids):
            raise IndexError("corpus row is out of range")
        return int(self.range_ids[row])

    def validate(self) -> None:
        self.artifacts.validate()
        self.features.validate()
        rows = len(self.artifacts.positions)
        ranges = len(self.artifacts.range_starts)
        if self.features.values.shape[0] != rows:
            raise ValueError("feature row count differs from the database")
        for name, values, shape, dtype in (
            ("range family IDs", self.range_family_ids, (ranges,), np.int16),
            ("family IDs", self.family_ids, (rows,), np.int16),
            ("range IDs", self.range_ids, (rows,), np.int32),
            ("train mask", self.train_mask, (rows,), np.bool_),
            ("evaluation mask", self.evaluation_mask, (rows,), np.bool_),
        ):
            if values.shape != shape or values.dtype != np.dtype(dtype):
                raise ValueError(f"{name} has the wrong shape or dtype")
        if np.any(self.range_family_ids < 0) or np.any(
            self.range_family_ids >= len(self.family_names)
        ):
            raise ValueError("range family IDs are outside the family table")
        if np.any(self.train_mask & self.evaluation_mask) or not np.all(
            self.train_mask | self.evaluation_mask
        ):
            raise ValueError("train/evaluation masks must be disjoint and exhaustive")
        expected_ranges, expected_families = _row_lookup(
            self.artifacts.range_starts,
            self.artifacts.range_stops,
            self.range_family_ids,
            rows,
        )
        if not np.array_equal(self.range_ids, expected_ranges) or not np.array_equal(
            self.family_ids, expected_families
        ):
            raise ValueError("row family/range lookup is inconsistent")


def _assemble_hybrid_corpus(
    artifacts: ArtifactSet,
    range_family_ids: np.ndarray,
    source_root: Path,
    manifest_receipt: Mapping[str, Any],
    source_assets: Mapping[str, Mapping[str, Any]] | None = None,
) -> HybridCorpus:
    artifacts, range_family_ids = _refine_takara_range(
        artifacts, range_family_ids
    )
    rows = len(artifacts.positions)
    range_ids, family_ids = _row_lookup(
        artifacts.range_starts, artifacts.range_stops, range_family_ids, rows
    )
    train_mask, evaluation_mask = _split_masks(
        artifacts.range_starts, artifacts.range_stops, range_family_ids, rows
    )
    features = build_matching_features(
        artifacts,
        FPS,
        HORIZONS,
        normalization_fit_mask=train_mask,
    )
    corpus = HybridCorpus(
        artifacts=artifacts,
        features=features,
        range_family_ids=range_family_ids,
        family_ids=family_ids,
        range_ids=range_ids,
        train_mask=train_mask,
        evaluation_mask=evaluation_mask,
        source_root=Path(source_root),
        manifest_receipt=dict(manifest_receipt),
        source_assets=dict(source_assets or {}),
    )
    corpus.validate()
    return corpus


def _range_families(manifest: Mapping[str, Any], ranges: int) -> np.ndarray:
    motion_banks = manifest.get("motion_banks")
    if type(motion_banks) is not dict or type(motion_banks.get("banks")) is not list:
        raise ValueError("manifest motion_banks.banks is missing")
    result = np.full(ranges, -1, np.int16)
    seen_names: set[str] = set()
    for bank in motion_banks["banks"]:
        if type(bank) is not dict or bank.get("family") not in _FAMILY_TO_ID:
            raise ValueError("manifest contains an unknown motion-bank family")
        family = bank["family"]
        if family in seen_names:
            raise ValueError(f"manifest repeats motion-bank family {family}")
        seen_names.add(family)
        indices = bank.get("range_indices")
        if type(indices) is not list or any(type(index) is not int for index in indices):
            raise ValueError(f"manifest range list for {family} is invalid")
        array = np.asarray(indices, np.int64)
        if len(array) == 0 or np.any(array < 0) or np.any(array >= ranges):
            raise ValueError(f"manifest range list for {family} is invalid")
        if not np.all(array[1:] > array[:-1]):
            raise ValueError(f"manifest range list for {family} is not sorted unique")
        if np.any(result[array] != -1):
            raise ValueError("manifest motion-bank range lists overlap")
        result[array] = _FAMILY_TO_ID[family]
    if seen_names != set(FAMILY_NAMES) or np.any(result < 0):
        raise ValueError("manifest motion-bank families do not cover every range")
    return result


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != "g1-terrain-artifacts/v3":
        raise ValueError("source must use g1-terrain-artifacts/v3")
    if manifest.get("database_frames") != EXPECTED_ROWS:
        raise ValueError(f"source must contain exactly {EXPECTED_ROWS} rows")
    if manifest.get("output_fps") != FPS:
        raise ValueError("source must use exactly 25 Hz")
    if manifest.get("feature_dimensions") != 39:
        raise ValueError("source legacy feature_dimensions metadata must be 39")
    if manifest.get("terrain_dimensions") != 12:
        raise ValueError("source terrain sidecar must have 12 dimensions")
    if manifest.get("support_dimensions") != 3:
        raise ValueError("source support sidecar must have 3 dimensions")
    if manifest.get("terrain_feature_distances_m") != list(TERRAIN_DISTANCES_M):
        raise ValueError("source terrain distances must be exactly 0.25, 0.5, 0.75, 1 m")
    if manifest.get("total_clips") != EXPECTED_RANGES or manifest.get(
        "grail_clips"
    ) != EXPECTED_RANGES - 1:
        raise ValueError("source clip counts are not canonical")
    database = manifest.get("database")
    if type(database) is not dict or database.get("schema") != "holden-database/v1":
        raise ValueError("source database must use holden-database/v1")
    sidecars = manifest.get("sidecars")
    if type(sidecars) is not dict:
        raise ValueError("source sidecar descriptors are missing")
    terrain = sidecars.get("terrain_features")
    if type(terrain) is not dict or (
        terrain.get("schema"), terrain.get("version"), terrain.get("dimensions")
    ) != ("G1TF/v2", 2, 12):
        raise ValueError("source terrain sidecar must use exact G1TF/v2 semantics")
    support = sidecars.get("terrain_support")
    if type(support) is not dict or (
        support.get("schema"), support.get("version"), support.get("dimensions")
    ) != ("G1SP/v1", 1, 3) or tuple(support.get("columns", ())) != _SUPPORT_COLUMNS:
        raise ValueError("source support sidecar must use exact G1SP/v1 semantics")
    skeleton = manifest.get("skeleton")
    if type(skeleton) is not dict or tuple(skeleton.get("names", ())) != tuple(
        G1_SKELETON_NAMES
    ) or tuple(skeleton.get("parents", ())) != tuple(G1_SKELETON_PARENTS) or skeleton.get(
        "signature"
    ) != G1_SKELETON_SIGNATURE:
        raise ValueError("source skeleton is not canonical G1")
    motion_banks = manifest.get("motion_banks")
    if type(motion_banks) is not dict or motion_banks.get(
        "schema"
    ) != "g1-terrain-motion-banks/v1" or motion_banks.get(
        "frame_count"
    ) != EXPECTED_ROWS or len(motion_banks.get("ranges", ())) != EXPECTED_RANGES:
        raise ValueError(f"source must contain exactly {EXPECTED_RANGES} ranges")
    banks = motion_banks.get("banks")
    if type(banks) is not list or len(banks) != len(FAMILY_NAMES):
        raise ValueError("source motion-bank family inventory is not canonical")
    first = 0
    for bank, family, count in zip(
        banks, FAMILY_NAMES, EXPECTED_FAMILY_SOURCE_RANGE_COUNTS
    ):
        expected = list(range(first, first + count))
        if type(bank) is not dict or bank.get("family") != family or bank.get(
            "range_indices"
        ) != expected:
            raise ValueError("source motion-bank family counts or indices changed")
        first += count
    ranges = motion_banks["ranges"]
    takara = ranges[0]
    if type(takara) is not dict or any((
        takara.get("global_start") != 0,
        takara.get("global_stop") != TAKARA_SOURCE_STOP,
        takara.get("source_frame_count") != TAKARA_SOURCE_STOP,
        takara.get("source_name") != TAKARA_SOURCE_NAME,
        takara.get("source_start") != 0,
        takara.get("source_stop") != TAKARA_SOURCE_STOP,
    )) or sum(
        type(item) is dict and item.get("source_name") == TAKARA_SOURCE_NAME
        for item in ranges
    ) != 1:
        raise ValueError("source must contain exactly the canonical Takara range")
    sources = manifest.get("sources")
    if type(sources) is not list or len(sources) != EXPECTED_RANGES or sum(
        type(item) is dict and item.get("name") == TAKARA_SOURCE_NAME
        for item in sources
    ) != 1:
        raise ValueError("source must contain exactly one Takara source identity")
    takara_source = sources[0]
    if type(takara_source) is not dict or any((
        takara_source.get("name") != TAKARA_SOURCE_NAME,
        takara_source.get("output_frames") != TAKARA_SOURCE_STOP,
        takara_source.get("range_start") != 0,
        takara_source.get("range_stop") != TAKARA_SOURCE_STOP,
        takara_source.get("source_fps") != 50.0,
    )):
        raise ValueError("canonical Takara source metadata changed")


def load_hybrid_corpus(root: Path) -> HybridCorpus:
    root = Path(root).resolve(strict=True)
    manifest_path = (root / "manifest.json").resolve(strict=True)
    if manifest_path.parent != root or not manifest_path.is_file():
        raise ValueError("source manifest must be a regular file in the source root")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source manifest is invalid JSON") from error
    if type(manifest) is not dict:
        raise ValueError("source manifest must be a JSON object")
    _validate_source_manifest(manifest)

    database_path, database_asset = _authenticate_asset(
        root, manifest.get("database"), "database"
    )
    sidecars = manifest.get("sidecars")
    if type(sidecars) is not dict:
        raise ValueError("source sidecars descriptor is missing")
    terrain_descriptor = sidecars.get("terrain_features")
    support_descriptor = sidecars.get("terrain_support")
    terrain_path, terrain_asset = _authenticate_asset(
        root, terrain_descriptor, "terrain sidecar"
    )
    support_path, support_asset = _authenticate_asset(
        root, support_descriptor, "support sidecar"
    )

    artifacts = read_holden_database(database_path)
    if len(artifacts.positions) != EXPECTED_ROWS or len(
        artifacts.range_starts
    ) != EXPECTED_RANGES:
        raise ValueError("database row/range count differs from its manifest")
    if not np.array_equal(
        artifacts.parents, np.asarray(G1_SKELETON_PARENTS, np.int32)
    ):
        raise ValueError("database parents are not the canonical G1 skeleton")
    manifest_ranges = manifest["motion_banks"]["ranges"]
    starts = np.asarray([item.get("global_start") for item in manifest_ranges])
    stops = np.asarray([item.get("global_stop") for item in manifest_ranges])
    if not np.array_equal(starts, artifacts.range_starts) or not np.array_equal(
        stops, artifacts.range_stops
    ):
        raise ValueError("database ranges differ from motion_banks ranges")

    artifacts.terrain_features = _read_terrain(terrain_path, terrain_descriptor)
    artifacts.terrain_support = read_support_sidecar(support_path)
    artifacts.validate()
    if not np.all(np.ptp(artifacts.terrain_features, axis=0) > 0.0):
        raise ValueError("all four active terrain channels must vary")

    range_family_ids = _range_families(manifest, EXPECTED_RANGES)
    source_assets = {
        "database": database_asset,
        "terrain_features": terrain_asset,
        "terrain_support": support_asset,
    }
    manifest_receipt = {
        "schema": _SOURCE_RECEIPT_SCHEMA,
        "path": str(manifest_path),
        "size_bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_schema": manifest["schema"],
        "database_rows": EXPECTED_ROWS,
        "source_ranges": EXPECTED_RANGES,
        "output_fps": FPS,
        "legacy_feature_dimensions": manifest["feature_dimensions"],
        "rebuilt_feature_dimensions": 31,
        "terrain_v2_selected_columns": list(_TERRAIN_V2_COLUMNS),
    }
    corpus = _assemble_hybrid_corpus(
        artifacts,
        range_family_ids,
        root,
        manifest_receipt,
        source_assets,
    )
    terrain_scale = corpus.features.scale[27:31]
    if np.any(terrain_scale >= np.finfo(np.float32).max) or not np.all(
        np.isfinite(terrain_scale)
    ):
        raise ValueError("all four terrain feature dimensions must remain active")
    return corpus


def _write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_array(path: Path, dtype: Any, shape: tuple[Any, ...]) -> None:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.dtype(dtype) or values.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(values.shape, shape)
    ):
        raise ValueError(f"staged cache array has wrong contract: {path.name}")
    if values.size and values.dtype.kind == "f" and not np.isfinite(values).all():
        raise ValueError(f"staged cache array is non-finite: {path.name}")


def _source_assets_for_publication(corpus: HybridCorpus) -> dict[str, dict[str, Any]]:
    if corpus.source_assets:
        return {name: dict(value) for name, value in corpus.source_assets.items()}
    schemas = {
        "database": "holden-database/v1",
        "terrain_features": "G1TF/v2",
        "terrain_support": "G1SP/v1",
    }
    filenames = {
        "database": "database.bin",
        "terrain_features": "terrain_features.bin",
        "terrain_support": "terrain_support.bin",
    }
    return {
        name: _asset_descriptor(corpus.source_root / filenames[name], schema)
        for name, schema in schemas.items()
    }


def _publish_hybrid_cache(corpus: HybridCorpus, output: Path) -> Path:
    corpus.validate()
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"hybrid corpus cache already exists: {output}")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    ))
    arrays = {
        "features.npy": corpus.features.values,
        "feature_offset.npy": corpus.features.offset,
        "feature_scale.npy": corpus.features.scale,
        "range_family_ids.npy": corpus.range_family_ids,
        "family_ids.npy": corpus.family_ids,
        "range_ids.npy": corpus.range_ids,
        "range_starts.npy": corpus.artifacts.range_starts,
        "range_stops.npy": corpus.artifacts.range_stops,
        "train_mask.npy": corpus.train_mask,
        "evaluation_mask.npy": corpus.evaluation_mask,
    }
    try:
        for name, values in arrays.items():
            _write_npy(staging / name, values)
        members = {
            name: {
                "path": name,
                "size_bytes": (staging / name).stat().st_size,
                "sha256": _sha256(staging / name),
            }
            for name in sorted(arrays)
        }
        manifest = {
            "schema": _CACHE_SCHEMA,
            "status": "accepted",
            "rows": len(corpus.artifacts.positions),
            "ranges": len(corpus.artifacts.range_starts),
            "source_ranges": len(corpus.artifacts.range_starts) - 1,
            "fps": FPS,
            "horizons": list(HORIZONS),
            "feature_dimensions": 31,
            "family_names": list(corpus.family_names),
            "split": dict(_SPLIT_CONTRACT),
            "virtual_ranges": {
                "takara": {
                    "source_range_count": 1,
                    "refined_range_count": 2,
                    "source_start": int(corpus.artifacts.range_starts[0]),
                    "heldout_start": int(corpus.artifacts.range_stops[0]),
                    "source_stop": int(corpus.artifacts.range_stops[1]),
                    "derivatives": "independent-per-virtual-range",
                    "future_horizons": "clamped-per-virtual-range",
                },
            },
            "normalization": dict(_NORMALIZATION_CONTRACT),
            "counts": {
                "train_rows": int(np.count_nonzero(corpus.train_mask)),
                "evaluation_rows": int(np.count_nonzero(corpus.evaluation_mask)),
            },
            "source_root": str(corpus.source_root.resolve()),
            "source_manifest": dict(corpus.manifest_receipt),
            "source_assets": _source_assets_for_publication(corpus),
            "members": members,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_canonical_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())

        for name, (dtype, shape) in _ARRAY_SPECS.items():
            _validate_array(staging / name, dtype, shape)
        reopened = json.loads(manifest_path.read_bytes())
        if reopened != manifest:
            raise ValueError("staged cache manifest failed reopen verification")
        _fsync_directory(staging)
        if output.exists():
            raise FileExistsError(f"hybrid corpus cache already exists: {output}")
        os.rename(staging, output)
        _fsync_directory(output.parent)
        return output
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_hybrid_cache(root: Path, output: Path) -> HybridCorpus:
    corpus = load_hybrid_corpus(root)
    _publish_hybrid_cache(corpus, output)
    manifest_sha256 = _sha256(Path(output) / "manifest.json")
    return replace(corpus, cache_manifest_sha256=manifest_sha256)


def _load_cache_manifest(output: Path) -> tuple[dict[str, Any], str]:
    manifest_path = output / "manifest.json"
    payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("hybrid cache manifest is invalid JSON") from error
    if type(manifest) is not dict or manifest.get("schema") != _CACHE_SCHEMA or manifest.get(
        "status"
    ) != "accepted":
        raise ValueError("hybrid cache manifest is not accepted v2-strict")
    if payload != _canonical_json_bytes(manifest):
        raise ValueError("hybrid cache manifest is not canonical JSON")
    if set(manifest.get("members", {})) != set(_ARRAY_SPECS):
        raise ValueError("hybrid cache member inventory changed")
    if {path.name for path in output.iterdir()} != set(_ARRAY_SPECS) | {
        "manifest.json"
    }:
        raise ValueError("hybrid cache contains missing or extra members")
    for name, descriptor in manifest["members"].items():
        path = output / name
        if type(descriptor) is not dict or descriptor.get("path") != name:
            raise ValueError(f"invalid cache member descriptor: {name}")
        if path.stat().st_size != descriptor.get("size_bytes") or _sha256(
            path
        ) != descriptor.get("sha256"):
            raise ValueError(f"cache member authentication failed: {name}")
    return manifest, hashlib.sha256(payload).hexdigest()


def _load_external_artifacts(assets: Mapping[str, Any]) -> ArtifactSet:
    expected = {"database", "terrain_features", "terrain_support"}
    expected_schemas = {
        "database": "holden-database/v1",
        "terrain_features": "G1TF/v2",
        "terrain_support": "G1SP/v1",
    }
    if type(assets) is not dict or set(assets) != expected:
        raise ValueError("cache source asset inventory changed")
    paths: dict[str, Path] = {}
    for name in expected:
        descriptor = assets[name]
        if type(descriptor) is not dict or type(descriptor.get("path")) is not str:
            raise ValueError(f"cache source descriptor is invalid: {name}")
        if descriptor.get("schema") != expected_schemas[name]:
            raise ValueError(f"cache source descriptor schema changed: {name}")
        path = Path(descriptor["path"]).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"cache source asset size changed: {name}")
        if _sha256(path) != descriptor.get("sha256"):
            raise ValueError(f"cache source asset SHA-256 changed: {name}")
        paths[name] = path
    artifacts = read_holden_database(paths["database"])
    artifacts.terrain_features = _read_terrain(
        paths["terrain_features"], assets["terrain_features"]
    )
    artifacts.terrain_support = read_support_sidecar(paths["terrain_support"])
    artifacts.validate()
    return artifacts


def _authenticated_primary_manifest(
    source_root: Path,
    receipt: Mapping[str, Any],
    source_assets: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reopen and authenticate the exact primary source authority."""

    expected_receipt = {
        "schema": _SOURCE_RECEIPT_SCHEMA,
        "source_schema": "g1-terrain-artifacts/v3",
        "database_rows": EXPECTED_ROWS,
        "source_ranges": EXPECTED_RANGES,
        "output_fps": FPS,
        "legacy_feature_dimensions": 39,
        "rebuilt_feature_dimensions": 31,
        "terrain_v2_selected_columns": list(_TERRAIN_V2_COLUMNS),
    }
    if type(receipt) is not dict or any(
        receipt.get(name) != expected
        for name, expected in expected_receipt.items()
    ):
        raise ValueError(
            "strict cache requires exact primary source authority: "
            "source receipt contract changed"
        )
    source_root = Path(source_root).resolve(strict=True)
    path = (source_root / "manifest.json").resolve(strict=True)
    if path.parent != source_root or receipt.get("path") != str(path):
        raise ValueError("cache source manifest path is not canonical")
    payload = path.read_bytes()
    if len(payload) != receipt.get("size_bytes") or hashlib.sha256(
        payload
    ).hexdigest() != receipt.get("sha256"):
        raise ValueError("cache source manifest authentication failed")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cache source manifest is invalid JSON") from error
    if type(manifest) is not dict:
        raise ValueError("cache source manifest must be a JSON object")
    _validate_source_manifest(manifest)
    sidecars = manifest["sidecars"]
    expected_assets = {
        "database": _authenticate_asset(
            source_root, manifest["database"], "database"
        )[1],
        "terrain_features": _authenticate_asset(
            source_root, sidecars["terrain_features"], "terrain sidecar"
        )[1],
        "terrain_support": _authenticate_asset(
            source_root, sidecars["terrain_support"], "support sidecar"
        )[1],
    }
    if source_assets != expected_assets:
        raise ValueError("cache source asset receipts differ from source authority")
    return manifest


def _byte_equal(actual: np.ndarray, expected: np.ndarray, dtype: Any) -> bool:
    left = np.ascontiguousarray(actual, dtype=dtype)
    right = np.ascontiguousarray(expected, dtype=dtype)
    return left.shape == right.shape and left.tobytes() == right.tobytes()


def _canonicalize_and_check_cache_arrays(
    artifacts: ArtifactSet,
    loaded: Mapping[str, np.ndarray],
    source_manifest: Mapping[str, Any] | None,
) -> ArtifactSet:
    if source_manifest is not None:
        source_ranges = source_manifest["motion_banks"]["ranges"]
        starts = np.asarray(
            [item.get("global_start") for item in source_ranges], np.int32
        )
        stops = np.asarray(
            [item.get("global_stop") for item in source_ranges], np.int32
        )
        if not _byte_equal(artifacts.range_starts, starts, np.int32) or not _byte_equal(
            artifacts.range_stops, stops, np.int32
        ):
            raise ValueError("immutable database ranges differ from source authority")
        source_families = _range_families(source_manifest, EXPECTED_RANGES)
        artifacts, expected_range_families = _refine_takara_range(
            artifacts, source_families
        )
    else:
        # Private synthetic caches write their already-refined ranges to the
        # external fixture.  They remain fully authenticated and still exercise
        # the canonical lookup/mask reconstruction below.
        expected_range_families = np.asarray(
            loaded["range_family_ids.npy"], np.int16
        )
        artifacts, expected_range_families = _refine_takara_range(
            artifacts, expected_range_families
        )

    rows = len(artifacts.positions)
    expected_range_ids, expected_family_ids = _row_lookup(
        artifacts.range_starts,
        artifacts.range_stops,
        expected_range_families,
        rows,
    )
    expected_train, expected_evaluation = _split_masks(
        artifacts.range_starts,
        artifacts.range_stops,
        expected_range_families,
        rows,
    )
    comparisons = (
        ("range starts", loaded["range_starts.npy"], artifacts.range_starts, np.int32),
        ("range stops", loaded["range_stops.npy"], artifacts.range_stops, np.int32),
        (
            "range families",
            loaded["range_family_ids.npy"],
            expected_range_families,
            np.int16,
        ),
        ("range IDs", loaded["range_ids.npy"], expected_range_ids, np.int32),
        ("family IDs", loaded["family_ids.npy"], expected_family_ids, np.int16),
        ("train mask", loaded["train_mask.npy"], expected_train, np.bool_),
        (
            "evaluation mask",
            loaded["evaluation_mask.npy"],
            expected_evaluation,
            np.bool_,
        ),
    )
    for label, actual, expected, dtype in comparisons:
        if not _byte_equal(actual, expected, dtype):
            if "mask" in label:
                raise ValueError(f"canonical split mask byte comparison failed: {label}")
            raise ValueError(f"canonical cache array byte comparison failed: {label}")
    return artifacts


def _load_hybrid_cache_impl(
    output: Path,
    *,
    expected_cache_manifest_sha256: str | None = None,
    allow_synthetic_test_cache: bool = False,
) -> HybridCorpus:
    output = Path(output).resolve(strict=True)
    manifest, cache_manifest_sha256 = _load_cache_manifest(output)
    if expected_cache_manifest_sha256 is not None:
        if type(expected_cache_manifest_sha256) is not str or re.fullmatch(
            r"[0-9a-f]{64}", expected_cache_manifest_sha256
        ) is None:
            raise ValueError("expected cache manifest SHA-256 is invalid")
        if cache_manifest_sha256 != expected_cache_manifest_sha256:
            raise ValueError("expected cache manifest SHA-256 does not match")
    for name, (dtype, shape) in _ARRAY_SPECS.items():
        _validate_array(output / name, dtype, shape)
    loaded = {
        name: np.load(output / name, mmap_mode="r", allow_pickle=False)
        for name in _ARRAY_SPECS
    }
    source_root = Path(manifest["source_root"])
    receipt = manifest.get("source_manifest", {})
    source_manifest = None
    if allow_synthetic_test_cache and receipt.get(
        "source_schema"
    ) != "g1-terrain-artifacts/v3":
        source_manifest = None
    else:
        source_manifest = _authenticated_primary_manifest(
            source_root,
            receipt,
            manifest.get("source_assets", {}),
        )
    artifacts = _load_external_artifacts(manifest.get("source_assets"))
    artifacts = _canonicalize_and_check_cache_arrays(
        artifacts, loaded, source_manifest
    )
    features = FeatureSet(
        loaded["features.npy"],
        loaded["feature_offset.npy"],
        loaded["feature_scale.npy"],
    )
    corpus = HybridCorpus(
        artifacts=artifacts,
        features=features,
        range_family_ids=loaded["range_family_ids.npy"],
        family_ids=loaded["family_ids.npy"],
        range_ids=loaded["range_ids.npy"],
        train_mask=loaded["train_mask.npy"],
        evaluation_mask=loaded["evaluation_mask.npy"],
        source_root=source_root,
        manifest_receipt=manifest["source_manifest"],
        source_assets=manifest["source_assets"],
        family_names=tuple(manifest["family_names"]),
        cache_manifest_sha256=cache_manifest_sha256,
    )
    corpus.validate()
    takara_virtual = manifest.get("virtual_ranges", {}).get("takara", {})
    if manifest.get("rows") != len(artifacts.positions) or manifest.get(
        "ranges"
    ) != len(artifacts.range_starts) or manifest.get("fps") != FPS or tuple(
        manifest.get("horizons", ())
    ) != HORIZONS or manifest.get("feature_dimensions") != 31:
        raise ValueError("hybrid cache dimensions or rate changed")
    expected_counts = {
        "train_rows": int(np.count_nonzero(corpus.train_mask)),
        "evaluation_rows": int(np.count_nonzero(corpus.evaluation_mask)),
    }
    if tuple(manifest.get("family_names", ())) != FAMILY_NAMES \
            or manifest.get("counts") != expected_counts \
            or manifest.get("split") != _SPLIT_CONTRACT:
        raise ValueError("hybrid cache canonical metadata contract changed")
    if manifest.get("source_ranges") != len(artifacts.range_starts) - 1 or manifest.get(
        "normalization"
    ) != _NORMALIZATION_CONTRACT or takara_virtual != {
        "source_range_count": 1,
        "refined_range_count": 2,
        "source_start": int(artifacts.range_starts[0]),
        "heldout_start": int(artifacts.range_stops[0]),
        "source_stop": int(artifacts.range_stops[1]),
        "derivatives": "independent-per-virtual-range",
        "future_horizons": "clamped-per-virtual-range",
    }:
        raise ValueError("hybrid cache strict split or normalization contract changed")
    if source_manifest is not None and (
        len(artifacts.range_starts) != EXPECTED_STRICT_RANGES
        or int(artifacts.range_stops[0]) != TAKARA_HELDOUT_START
    ):
        raise ValueError("primary cache does not contain the exact strict Takara split")
    return corpus


def _load_synthetic_hybrid_cache_for_tests(
    output: Path,
    *,
    expected_cache_manifest_sha256: str | None = None,
) -> HybridCorpus:
    """Exercise cache mechanics for small fixtures without weakening production."""

    return _load_hybrid_cache_impl(
        output,
        expected_cache_manifest_sha256=expected_cache_manifest_sha256,
        allow_synthetic_test_cache=True,
    )


def load_hybrid_cache(
    output: Path,
    *,
    expected_cache_manifest_sha256: str | None = None,
) -> HybridCorpus:
    """Load only an exact authenticated primary strict cache."""

    return _load_hybrid_cache_impl(
        output,
        expected_cache_manifest_sha256=expected_cache_manifest_sha256,
    )


def _summary(corpus: HybridCorpus, output: Path) -> dict[str, Any]:
    return {
        "status": "accepted",
        "output": str(Path(output).resolve()),
        "rows": len(corpus.artifacts.positions),
        "ranges": len(corpus.artifacts.range_starts),
        "families": list(corpus.family_names),
        "train_rows": int(np.count_nonzero(corpus.train_mask)),
        "evaluation_rows": int(np.count_nonzero(corpus.evaluation_mask)),
        "feature_dimensions": corpus.features.values.shape[1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build an authenticated feature cache")
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        corpus = build_hybrid_cache(arguments.source, arguments.output)
        print(json.dumps(_summary(corpus, arguments.output), sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
