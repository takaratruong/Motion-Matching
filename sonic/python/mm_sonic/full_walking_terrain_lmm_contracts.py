"""Immutable contracts for the complete walking-only terrain LMM corpus."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from resources.g1_terrain_builder.artifacts import (
    canonical_json_bytes,
    sha256_file,
)
from resources.g1_terrain_builder.database import (
    read_holden_database,
    write_holden_database,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    ArtifactSet,
)

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
Family: TypeAlias = Literal["flat", "curb", "slope", "stair"]
Split: TypeAlias = Literal["train", "validation", "test"]
Quality: TypeAlias = Literal["clean", "usable", "quarantined"]

INVENTORY_SCHEMA = "g1-full-walking-terrain-lmm-inventory/v1"
LANE_SCHEMA = "g1-full-walking-terrain-lmm-lane/v1"
SPLIT_LEDGER_SCHEMA = "g1-full-walking-terrain-lmm-split-ledger/v1"
FPS = 60.0
FAMILIES = ("flat", "curb", "slope", "stair")
SPLITS = ("train", "validation", "test")
_SHA256 = frozenset("0123456789abcdef")
_RANGE_KEYS = {
    "range_id",
    "canonical_source_id",
    "terrain_id",
    "mirror_of",
    "family",
    "split_group_id",
    "split",
    "start",
    "stop",
    "quality",
    "authority",
}
_MANIFEST_KEYS = {
    "schema",
    "authority_manifest_sha256",
    "fps",
    "rows",
    "ranges",
    "skeleton",
    "members",
}
_MEMBER_KEYS = {"path", "size_bytes", "sha256"}
_MEMBER_NAMES = ("database.bin", "terrain_grid.npy", "ranges.json")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    canonical_source_id: str
    terrain_id: str
    mirror_of: str | None
    family: Family
    authority: Mapping[str, JSONValue]


@dataclass(frozen=True)
class FullWalkingInventory:
    build_id: str
    sources: tuple[SourceRecord, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class SplitAssignment:
    split_group_id: str
    split: Split
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class RangeRecord:
    range_id: str
    canonical_source_id: str
    terrain_id: str
    mirror_of: str | None
    family: Family
    split_group_id: str
    split: Split
    start: int
    stop: int
    quality: Quality
    authority: Mapping[str, JSONValue]


@dataclass(frozen=True)
class LaneArtifact:
    root: Path
    manifest_sha256: str
    artifacts: ArtifactSet
    terrain_grid: np.ndarray
    ranges: tuple[RangeRecord, ...]


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - _SHA256)
    )


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _normalize_authority(value: object, label: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only finite JSON values") from error
    if type(normalized) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _validate_source(record: SourceRecord) -> None:
    if not isinstance(record, SourceRecord):
        raise TypeError("split sources must be SourceRecord values")
    _require_identifier(record.source_id, "source_id")
    _require_identifier(record.canonical_source_id, "canonical_source_id")
    _require_identifier(record.terrain_id, "terrain_id")
    if record.mirror_of is not None:
        _require_identifier(record.mirror_of, "mirror_of")
        if record.mirror_of == record.source_id:
            raise ValueError("a source cannot mirror itself")
    if record.family not in FAMILIES:
        raise ValueError(f"invalid source family {record.family!r}")
    _normalize_authority(record.authority, "source authority")


def _validated_sources(
    sources: Sequence[SourceRecord],
) -> tuple[SourceRecord, ...]:
    records = tuple(sources)
    if not records:
        raise ValueError("inventory contains no sources")
    for record in records:
        _validate_source(record)
    ids = [record.source_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("source_id values must be unique")
    id_set = set(ids)
    for record in records:
        if record.mirror_of is not None and record.mirror_of not in id_set:
            raise ValueError(
                f"mirror source {record.source_id!r} names missing source "
                f"{record.mirror_of!r}"
            )
    return records


def source_record_json(record: SourceRecord) -> dict[str, JSONValue]:
    _validate_source(record)
    value = asdict(record)
    value["authority"] = _normalize_authority(
        record.authority, f"source {record.source_id!r} authority"
    )
    return value


def inventory_manifest_bytes(inventory: FullWalkingInventory) -> bytes:
    if not isinstance(inventory, FullWalkingInventory):
        raise TypeError("inventory must be FullWalkingInventory")
    sources = _validated_sources(inventory.sources)
    if not _is_sha256(inventory.build_id):
        raise ValueError("inventory build_id must be lowercase SHA-256")
    payload = {
        "schema": INVENTORY_SCHEMA,
        "build_id": inventory.build_id,
        "sources": [source_record_json(record) for record in sources],
    }
    return canonical_json_bytes(payload)


def connected_split_groups(
    sources: Sequence[SourceRecord],
) -> tuple[tuple[str, ...], ...]:
    """Return source components linked by canonical, terrain, or mirror identity."""

    records = _validated_sources(sources)
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_source = {record.source_id: index for index, record in enumerate(records)}
    seen_tokens: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for token in (
            ("canonical", record.canonical_source_id),
            ("terrain", record.terrain_id),
        ):
            if token in seen_tokens:
                union(index, seen_tokens[token])
            else:
                seen_tokens[token] = index
        if record.mirror_of is not None:
            union(index, by_source[record.mirror_of])

    components: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        components[find(index)].append(record.source_id)
    groups = [tuple(sorted(source_ids)) for source_ids in components.values()]
    groups.sort(key=canonical_json_bytes)
    return tuple(groups)


def _split_group_id(source_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(source_ids))).hexdigest()


def build_split_ledger(
    inventory: FullWalkingInventory, *, build_id: str, seed: int
) -> tuple[SplitAssignment, ...]:
    if not isinstance(inventory, FullWalkingInventory):
        raise TypeError("inventory must be FullWalkingInventory")
    records = _validated_sources(inventory.sources)
    _require_identifier(build_id, "build_id")
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    by_id = {record.source_id: record for record in records}
    strata: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for source_ids in connected_split_groups(records):
        families = {by_id[source_id].family for source_id in source_ids}
        if len(families) != 1:
            raise ValueError("one connected split group crosses terrain families")
        family = next(iter(families))
        strata[family].append((_split_group_id(source_ids), source_ids))

    assignments: list[SplitAssignment] = []
    for family in FAMILIES:
        groups = strata.get(family, [])
        if not groups:
            continue
        if len(groups) < 3:
            raise ValueError(
                f"{family} stratum cannot populate validation and test splits"
            )
        ranked = sorted(
            groups,
            key=lambda item: (
                hashlib.sha256(
                    (build_id + str(seed) + item[0]).encode("utf-8")
                ).digest(),
                item[0],
            ),
        )
        validation_count = max(1, round(len(ranked) * 0.1))
        test_count = max(1, round(len(ranked) * 0.1))
        while validation_count + test_count >= len(ranked):
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        train_count = len(ranked) - validation_count - test_count
        split_names = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count
        )
        assignments.extend(
            SplitAssignment(group_id, split_name, source_ids)
            for (group_id, source_ids), split_name in zip(ranked, split_names)
        )
    return tuple(sorted(assignments, key=lambda item: item.split_group_id))


def split_ledger_bytes(
    assignments: Sequence[SplitAssignment], *, build_id: str, seed: int
) -> bytes:
    _require_identifier(build_id, "build_id")
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    rows = []
    seen_groups: set[str] = set()
    seen_sources: set[str] = set()
    for assignment in sorted(assignments, key=lambda item: item.split_group_id):
        if not isinstance(assignment, SplitAssignment):
            raise TypeError("ledger values must be SplitAssignment records")
        if not _is_sha256(assignment.split_group_id):
            raise ValueError("split_group_id must be lowercase SHA-256")
        if assignment.split not in SPLITS:
            raise ValueError(f"invalid split {assignment.split!r}")
        if not assignment.source_ids or tuple(sorted(assignment.source_ids)) \
                != assignment.source_ids:
            raise ValueError("split source_ids must be a non-empty sorted tuple")
        if assignment.split_group_id in seen_groups \
                or seen_sources.intersection(assignment.source_ids):
            raise ValueError("split ledger groups and sources must be unique")
        seen_groups.add(assignment.split_group_id)
        seen_sources.update(assignment.source_ids)
        rows.append(asdict(assignment))
    return canonical_json_bytes(
        {
            "schema": SPLIT_LEDGER_SCHEMA,
            "build_id": build_id,
            "seed": seed,
            "assignments": rows,
        }
    )


def _validate_array(
    values: np.ndarray,
    *,
    label: str,
    dtype: str,
    shape: tuple[int | None, ...],
    finite: bool = True,
) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if values.dtype.str != dtype:
        raise ValueError(f"{label} must use exact little-endian {dtype} storage")
    if values.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(values.shape, shape)
    ):
        raise ValueError(f"{label} has invalid shape {values.shape}")
    if not values.flags.c_contiguous:
        raise ValueError(f"{label} must be C-contiguous")
    if finite and not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")


def _range_json(record: RangeRecord) -> dict[str, JSONValue]:
    if not isinstance(record, RangeRecord):
        raise TypeError("lane ranges must be RangeRecord values")
    for label, value in (
        ("range_id", record.range_id),
        ("canonical_source_id", record.canonical_source_id),
        ("terrain_id", record.terrain_id),
        ("split_group_id", record.split_group_id),
    ):
        _require_identifier(value, label)
    if record.mirror_of is not None:
        _require_identifier(record.mirror_of, "mirror_of")
    if record.family not in FAMILIES:
        raise ValueError(f"invalid range family {record.family!r}")
    if record.split not in SPLITS:
        raise ValueError(f"invalid range split {record.split!r}")
    if record.quality not in ("clean", "usable", "quarantined"):
        raise ValueError(f"invalid range quality {record.quality!r}")
    if type(record.start) is not int or type(record.stop) is not int \
            or record.start < 0 or record.stop <= record.start:
        raise ValueError("range bounds must be non-empty non-negative integers")
    value = asdict(record)
    value["authority"] = _normalize_authority(
        record.authority, f"range {record.range_id!r} authority"
    )
    return value


def _validate_lane(lane: LaneArtifact) -> None:
    if not isinstance(lane, LaneArtifact):
        raise TypeError("lane must be LaneArtifact")
    if not _is_sha256(lane.manifest_sha256):
        raise ValueError("lane authority manifest digest must be lowercase SHA-256")
    artifacts = lane.artifacts
    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("lane artifacts must be ArtifactSet")
    artifacts.validate()
    rows = len(artifacts.positions)
    for label, values, dtype, shape, finite in (
        ("positions", artifacts.positions, "<f4", (rows, 31, 3), True),
        ("velocities", artifacts.velocities, "<f4", (rows, 31, 3), True),
        ("rotations", artifacts.rotations, "<f4", (rows, 31, 4), True),
        (
            "angular_velocities",
            artifacts.angular_velocities,
            "<f4",
            (rows, 31, 3),
            True,
        ),
        ("parents", artifacts.parents, "<i4", (31,), False),
        (
            "range_starts",
            artifacts.range_starts,
            "<i4",
            (len(lane.ranges),),
            False,
        ),
        (
            "range_stops",
            artifacts.range_stops,
            "<i4",
            (len(lane.ranges),),
            False,
        ),
        ("contacts", artifacts.contacts, "|u1", (rows, 2), False),
        ("terrain_features", artifacts.terrain_features, "<f4", (rows, 4), True),
        ("terrain_support", artifacts.terrain_support, "<f4", (rows, 3), True),
        ("terrain_grid", lane.terrain_grid, "<f4", (rows, 36), True),
    ):
        _validate_array(
            values,
            label=label,
            dtype=dtype,
            shape=shape,
            finite=finite,
        )
    if tuple(artifacts.parents.tolist()) != G1_SKELETON_PARENTS:
        raise ValueError("lane does not use the canonical G1 skeleton")
    if np.any(artifacts.contacts > 1):
        raise ValueError("lane contacts must be binary")
    if len(lane.ranges) != len(artifacts.range_starts):
        raise ValueError("lane requires one RangeRecord per artifact range")
    range_ids: set[str] = set()
    expected_start = 0
    for index, record in enumerate(lane.ranges):
        _range_json(record)
        if record.range_id in range_ids:
            raise ValueError("lane range_id values must be unique")
        range_ids.add(record.range_id)
        if record.start != expected_start \
                or record.start != int(artifacts.range_starts[index]) \
                or record.stop != int(artifacts.range_stops[index]):
            raise ValueError("lane ranges must exactly and contiguously cover rows")
        expected_start = record.stop
    if expected_start != rows:
        raise ValueError("lane ranges must cover every row")


def _member_descriptor(path: Path) -> dict[str, JSONValue]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_lane_exclusive(lane: LaneArtifact, output: Path) -> Path:
    """Validate and atomically publish one immutable lane to an absent path."""

    _validate_lane(lane)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable lane output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        write_holden_database(staging / "database.bin", lane.artifacts)
        with (staging / "terrain_grid.npy").open("wb") as stream:
            np.save(stream, lane.terrain_grid, allow_pickle=False)
        (staging / "ranges.json").write_bytes(
            canonical_json_bytes([_range_json(record) for record in lane.ranges])
        )
        for name in _MEMBER_NAMES:
            _fsync_file(staging / name)
        members = {
            name: _member_descriptor(staging / name) for name in _MEMBER_NAMES
        }
        manifest = {
            "schema": LANE_SCHEMA,
            "authority_manifest_sha256": lane.manifest_sha256,
            "fps": FPS,
            "rows": len(lane.artifacts.positions),
            "ranges": len(lane.ranges),
            "skeleton": {
                "names": list(G1_SKELETON_NAMES),
                "parents": list(G1_SKELETON_PARENTS),
                "signature": G1_SKELETON_SIGNATURE,
            },
            "members": members,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        _fsync_file(staging / "manifest.json")
        _fsync_directory(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"immutable lane output already exists: {output}")
        os.rename(staging, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _parse_canonical_json(path: Path, label: str) -> object:
    payload = path.read_bytes()
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if canonical_json_bytes(decoded) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return decoded


def _parse_range(value: object) -> RangeRecord:
    if type(value) is not dict or set(value) != _RANGE_KEYS:
        raise ValueError(f"range keys must be exactly {sorted(_RANGE_KEYS)}")
    try:
        record = RangeRecord(**value)
    except TypeError as error:
        raise ValueError("range record has invalid fields") from error
    _range_json(record)
    return record


def load_lane(
    root: Path, *, expected_manifest_sha256: str | None = None
) -> LaneArtifact:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("lane root must be a directory")
    actual_names = {path.name for path in root.iterdir()}
    expected_names = {*_MEMBER_NAMES, "manifest.json"}
    if actual_names != expected_names:
        raise ValueError(f"lane member names must be exactly {sorted(expected_names)}")
    manifest_path = root / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        if not _is_sha256(expected_manifest_sha256):
            raise ValueError("expected manifest SHA-256 is invalid")
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError("lane manifest SHA-256 mismatch")
    manifest = _parse_canonical_json(manifest_path, "lane manifest")
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        raise ValueError(f"lane manifest keys must be exactly {sorted(_MANIFEST_KEYS)}")
    if manifest["schema"] != LANE_SCHEMA:
        raise ValueError(f"lane manifest schema must be {LANE_SCHEMA}")
    if not _is_sha256(manifest["authority_manifest_sha256"]):
        raise ValueError("lane authority manifest SHA-256 is invalid")
    if manifest["fps"] != FPS \
            or type(manifest["rows"]) is not int \
            or type(manifest["ranges"]) is not int \
            or manifest["rows"] < 1 \
            or manifest["ranges"] < 1:
        raise ValueError("lane manifest dimensions or rate are invalid")
    if manifest["skeleton"] != {
        "names": list(G1_SKELETON_NAMES),
        "parents": list(G1_SKELETON_PARENTS),
        "signature": G1_SKELETON_SIGNATURE,
    }:
        raise ValueError("lane manifest canonical skeleton changed")
    members = manifest["members"]
    if type(members) is not dict or set(members) != set(_MEMBER_NAMES):
        raise ValueError("lane member descriptor keys are invalid")
    for name in _MEMBER_NAMES:
        descriptor = members[name]
        if type(descriptor) is not dict or set(descriptor) != _MEMBER_KEYS \
                or descriptor["path"] != name \
                or type(descriptor["size_bytes"]) is not int \
                or descriptor["size_bytes"] < 0 \
                or not _is_sha256(descriptor["sha256"]):
            raise ValueError(f"{name} member descriptor is invalid")
        path = root / name
        if path.is_symlink() or not path.is_file() \
                or path.stat().st_size != descriptor["size_bytes"] \
                or sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"{name} SHA-256 or size mismatch")

    artifacts = read_holden_database(root / "database.bin")
    with (root / "terrain_grid.npy").open("rb") as stream:
        terrain_grid = np.load(stream, allow_pickle=False)
        if stream.read(1):
            raise ValueError("terrain_grid.npy has trailing bytes")
    ranges_value = _parse_canonical_json(root / "ranges.json", "lane ranges")
    if type(ranges_value) is not list:
        raise ValueError("lane ranges must be a JSON array")
    ranges = tuple(_parse_range(value) for value in ranges_value)
    if len(artifacts.positions) != manifest["rows"] \
            or len(ranges) != manifest["ranges"]:
        raise ValueError("lane manifest row/range counts do not match members")
    lane = LaneArtifact(
        root=root,
        manifest_sha256=manifest["authority_manifest_sha256"],
        artifacts=artifacts,
        terrain_grid=terrain_grid,
        ranges=ranges,
    )
    _validate_lane(lane)
    return LaneArtifact(
        root=root,
        manifest_sha256=manifest_sha256,
        artifacts=artifacts,
        terrain_grid=terrain_grid,
        ranges=ranges,
    )


__all__ = [
    "FullWalkingInventory",
    "JSONValue",
    "LaneArtifact",
    "RangeRecord",
    "SourceRecord",
    "SplitAssignment",
    "build_split_ledger",
    "connected_split_groups",
    "inventory_manifest_bytes",
    "load_lane",
    "publish_lane_exclusive",
    "split_ledger_bytes",
]
