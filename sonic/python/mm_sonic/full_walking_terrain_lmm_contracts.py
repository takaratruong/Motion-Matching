"""Immutable contracts for the complete walking-only terrain LMM corpus."""

from __future__ import annotations

import ctypes
import errno
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
LANE_SCHEMA = "g1-full-walking-terrain-lmm-lane/v2"
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
    "inventory_manifest_sha256",
    "split_ledger_manifest_sha256",
    "fps",
    "rows",
    "ranges",
    "skeleton",
    "members",
}
_MEMBER_KEYS = {"path", "size_bytes", "sha256"}
_MEMBER_NAMES = (
    "database.bin",
    "terrain_features.npy",
    "terrain_support.npy",
    "terrain_grid.npy",
    "source_ids.json",
    "source_left_indices.npy",
    "source_right_indices.npy",
    "source_alpha.npy",
    "ranges.json",
)


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
class SplitLedger:
    build_id: str
    inventory_manifest_sha256: str
    seed: int
    requested: Mapping[str, int]
    actual_family_counts: Mapping[str, Mapping[str, int]]
    assignments: tuple[SplitAssignment, ...]
    manifest_sha256: str


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
    inventory_manifest_sha256: str
    split_ledger_manifest_sha256: str
    artifacts: ArtifactSet
    terrain_grid: np.ndarray
    source_ids: tuple[str, ...]
    source_left_indices: np.ndarray
    source_right_indices: np.ndarray
    source_alpha: np.ndarray
    ranges: tuple[RangeRecord, ...]


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not (set(value) - _SHA256)


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
) -> SplitLedger:
    if not isinstance(inventory, FullWalkingInventory):
        raise TypeError("inventory must be FullWalkingInventory")
    records = _validated_sources(inventory.sources)
    if build_id != inventory.build_id:
        raise ValueError("split build_id must equal the inventory build_id")
    if not _is_sha256(build_id) or not _is_sha256(inventory.manifest_sha256):
        raise ValueError("split inventory authority must use lowercase SHA-256")
    if (
        hashlib.sha256(inventory_manifest_bytes(inventory)).hexdigest()
        != inventory.manifest_sha256
    ):
        raise ValueError("inventory manifest SHA-256 does not match canonical content")
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
    ordered = tuple(sorted(assignments, key=lambda item: item.split_group_id))
    actual = {
        family: {
            split: sum(
                assignment.split == split
                and by_id[assignment.source_ids[0]].family == family
                for assignment in ordered
            )
            for split in SPLITS
        }
        for family in FAMILIES
    }
    provisional = SplitLedger(
        build_id=build_id,
        inventory_manifest_sha256=inventory.manifest_sha256,
        seed=seed,
        requested={"train": 80, "validation": 10, "test": 10},
        actual_family_counts=actual,
        assignments=ordered,
        manifest_sha256="",
    )
    manifest_sha256 = hashlib.sha256(_split_ledger_payload(provisional)).hexdigest()
    return SplitLedger(
        build_id=provisional.build_id,
        inventory_manifest_sha256=provisional.inventory_manifest_sha256,
        seed=provisional.seed,
        requested=provisional.requested,
        actual_family_counts=provisional.actual_family_counts,
        assignments=provisional.assignments,
        manifest_sha256=manifest_sha256,
    )


def _split_ledger_payload(ledger: SplitLedger) -> bytes:
    if not isinstance(ledger, SplitLedger):
        raise TypeError("ledger must be SplitLedger")
    if not _is_sha256(ledger.build_id) or not _is_sha256(
        ledger.inventory_manifest_sha256
    ):
        raise ValueError("split ledger parent identities must be lowercase SHA-256")
    if type(ledger.seed) is not int:
        raise ValueError("split seed must be an integer")
    if dict(ledger.requested) != {"train": 80, "validation": 10, "test": 10}:
        raise ValueError("split requested allocation must be exact 80/10/10")
    actual = {
        family: dict(ledger.actual_family_counts.get(family, {})) for family in FAMILIES
    }
    if set(ledger.actual_family_counts) != set(FAMILIES) or any(
        set(counts) != set(SPLITS)
        or any(type(count) is not int or count < 0 for count in counts.values())
        for counts in actual.values()
    ):
        raise ValueError("split actual per-family counts are invalid")
    rows = []
    seen_groups: set[str] = set()
    seen_sources: set[str] = set()
    if (
        tuple(sorted(ledger.assignments, key=lambda item: item.split_group_id))
        != ledger.assignments
    ):
        raise ValueError("split ledger assignments must be sorted")
    for assignment in ledger.assignments:
        if not isinstance(assignment, SplitAssignment):
            raise TypeError("ledger values must be SplitAssignment records")
        if not _is_sha256(assignment.split_group_id):
            raise ValueError("split_group_id must be lowercase SHA-256")
        if assignment.split not in SPLITS:
            raise ValueError(f"invalid split {assignment.split!r}")
        if (
            not assignment.source_ids
            or tuple(sorted(assignment.source_ids)) != assignment.source_ids
        ):
            raise ValueError("split source_ids must be a non-empty sorted tuple")
        if assignment.split_group_id in seen_groups or seen_sources.intersection(
            assignment.source_ids
        ):
            raise ValueError("split ledger groups and sources must be unique")
        seen_groups.add(assignment.split_group_id)
        seen_sources.update(assignment.source_ids)
        rows.append(
            {
                "split_group_id": assignment.split_group_id,
                "split": assignment.split,
                "source_ids": list(assignment.source_ids),
            }
        )
    return canonical_json_bytes(
        {
            "schema": SPLIT_LEDGER_SCHEMA,
            "build_id": ledger.build_id,
            "inventory_manifest_sha256": ledger.inventory_manifest_sha256,
            "seed": ledger.seed,
            "requested": dict(ledger.requested),
            "actual_family_counts": actual,
            "assignments": rows,
        }
    )


def split_ledger_bytes(ledger: SplitLedger) -> bytes:
    payload = _split_ledger_payload(ledger)
    digest = hashlib.sha256(payload).hexdigest()
    if ledger.manifest_sha256 and ledger.manifest_sha256 != digest:
        raise ValueError("split ledger manifest SHA-256 does not match its content")
    return payload


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
    if (
        type(record.start) is not int
        or type(record.stop) is not int
        or record.start < 0
        or record.stop <= record.start
    ):
        raise ValueError("range bounds must be non-empty non-negative integers")
    value = asdict(record)
    value["authority"] = _normalize_authority(
        record.authority, f"range {record.range_id!r} authority"
    )
    return value


def _validate_lane(
    lane: LaneArtifact,
    *,
    inventory: FullWalkingInventory | None = None,
    split_ledger: SplitLedger | None = None,
    require_unpublished: bool = False,
) -> None:
    if not isinstance(lane, LaneArtifact):
        raise TypeError("lane must be LaneArtifact")
    if require_unpublished:
        if lane.manifest_sha256 != "":
            raise ValueError(
                "an unpublished lane must have an empty self manifest digest"
            )
    elif not _is_sha256(lane.manifest_sha256):
        raise ValueError("lane self manifest digest must be lowercase SHA-256")
    if not _is_sha256(lane.inventory_manifest_sha256):
        raise ValueError("lane inventory manifest SHA-256 is invalid")
    if not _is_sha256(lane.split_ledger_manifest_sha256):
        raise ValueError("lane split ledger manifest SHA-256 is invalid")
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
        (
            "source_left_indices",
            lane.source_left_indices,
            "<i4",
            (rows,),
            False,
        ),
        (
            "source_right_indices",
            lane.source_right_indices,
            "<i4",
            (rows,),
            False,
        ),
        ("source_alpha", lane.source_alpha, "<f4", (rows,), True),
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
    if (
        type(lane.source_ids) is not tuple
        or len(lane.source_ids) != len(lane.ranges)
        or any(
            type(source_id) is not str or not source_id for source_id in lane.source_ids
        )
    ):
        raise ValueError(
            "lane source identities must be one non-empty string per range"
        )
    if (
        np.any(lane.source_left_indices < 0)
        or np.any(lane.source_right_indices < lane.source_left_indices)
        or np.any(lane.source_alpha < 0.0)
        or np.any(lane.source_alpha > 1.0)
    ):
        raise ValueError("lane source interpolation provenance is out of bounds")
    if len(lane.ranges) != len(artifacts.range_starts):
        raise ValueError("lane requires one RangeRecord per artifact range")
    source_records = (
        {record.source_id: record for record in inventory.sources}
        if inventory is not None
        else None
    )
    split_by_source: dict[str, SplitAssignment] | None = None
    if inventory is not None or split_ledger is not None:
        if inventory is None or split_ledger is None:
            raise ValueError(
                "lane authority validation requires inventory and split ledger"
            )
        if (
            hashlib.sha256(inventory_manifest_bytes(inventory)).hexdigest()
            != inventory.manifest_sha256
        ):
            raise ValueError("lane inventory object is not self-authenticating")
        split_ledger_bytes(split_ledger)
        expected_ledger = build_split_ledger(
            inventory, build_id=inventory.build_id, seed=split_ledger.seed
        )
        if split_ledger_bytes(split_ledger) != split_ledger_bytes(expected_ledger):
            raise ValueError(
                "lane split authority does not match connected inventory coverage"
            )
        if (
            split_ledger.inventory_manifest_sha256 != inventory.manifest_sha256
            or split_ledger.manifest_sha256 != lane.split_ledger_manifest_sha256
            or inventory.manifest_sha256 != lane.inventory_manifest_sha256
        ):
            raise ValueError("lane parent manifest authorities do not match")
        split_by_source = {}
        for assignment in split_ledger.assignments:
            for source_id in assignment.source_ids:
                split_by_source[source_id] = assignment
    range_ids: set[str] = set()
    expected_start = 0
    for index, record in enumerate(lane.ranges):
        _range_json(record)
        if record.range_id in range_ids:
            raise ValueError("lane range_id values must be unique")
        range_ids.add(record.range_id)
        if (
            record.start != expected_start
            or record.start != int(artifacts.range_starts[index])
            or record.stop != int(artifacts.range_stops[index])
        ):
            raise ValueError("lane ranges must exactly and contiguously cover rows")
        source_id = lane.source_ids[index]
        authority = _normalize_authority(
            record.authority, f"range {record.range_id!r} authority"
        )
        source_frame_count = authority.get("source_frame_count")
        if source_frame_count is not None and (
            type(source_frame_count) is not int
            or source_frame_count < 1
            or np.any(
                lane.source_right_indices[record.start : record.stop]
                >= source_frame_count
            )
        ):
            raise ValueError("lane source provenance exceeds its source range bounds")
        if source_records is not None and split_by_source is not None:
            source = source_records.get(source_id)
            assignment = split_by_source.get(source_id)
            if source is None or assignment is None:
                raise ValueError(
                    "lane range source is absent from authenticated authorities"
                )
            if (
                source.canonical_source_id != record.canonical_source_id
                or source.terrain_id != record.terrain_id
                or source.mirror_of != record.mirror_of
                or source.family != record.family
                or assignment.split_group_id != record.split_group_id
                or assignment.split != record.split
            ):
                raise ValueError(
                    "lane range source/split disagrees with authenticated ledger"
                )
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


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for immutable publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"immutable lane output already exists: {destination}",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def publish_lane_exclusive(lane: LaneArtifact, output: Path) -> Path:
    """Validate and atomically publish one immutable lane to an absent path."""

    _validate_lane(lane, require_unpublished=True)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable lane output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        write_holden_database(staging / "database.bin", lane.artifacts)
        for name, values in (
            ("terrain_features.npy", lane.artifacts.terrain_features),
            ("terrain_support.npy", lane.artifacts.terrain_support),
            ("terrain_grid.npy", lane.terrain_grid),
        ):
            with (staging / name).open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
        (staging / "source_ids.json").write_bytes(
            canonical_json_bytes(list(lane.source_ids))
        )
        for name, values in (
            ("source_left_indices.npy", lane.source_left_indices),
            ("source_right_indices.npy", lane.source_right_indices),
            ("source_alpha.npy", lane.source_alpha),
        ):
            with (staging / name).open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
        (staging / "ranges.json").write_bytes(
            canonical_json_bytes([_range_json(record) for record in lane.ranges])
        )
        for name in _MEMBER_NAMES:
            _fsync_file(staging / name)
        members = {name: _member_descriptor(staging / name) for name in _MEMBER_NAMES}
        manifest = {
            "schema": LANE_SCHEMA,
            "inventory_manifest_sha256": lane.inventory_manifest_sha256,
            "split_ledger_manifest_sha256": lane.split_ledger_manifest_sha256,
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
        _rename_noreplace(staging, output)
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


def load_split_ledger(
    path: Path,
    *,
    inventory: FullWalkingInventory,
    expected_manifest_sha256: str | None = None,
) -> SplitLedger:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("split ledger must be a regular non-symlink file")
    manifest_sha256 = sha256_file(path)
    if expected_manifest_sha256 is not None and (
        not _is_sha256(expected_manifest_sha256)
        or manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("split ledger manifest SHA-256 mismatch")
    if (
        hashlib.sha256(inventory_manifest_bytes(inventory)).hexdigest()
        != inventory.manifest_sha256
    ):
        raise ValueError("split ledger inventory object is not self-authenticating")
    value = _parse_canonical_json(path, "split ledger")
    keys = {
        "schema",
        "build_id",
        "inventory_manifest_sha256",
        "seed",
        "requested",
        "actual_family_counts",
        "assignments",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value["schema"] != SPLIT_LEDGER_SCHEMA
        or type(value["assignments"]) is not list
    ):
        raise ValueError("split ledger has invalid schema or keys")
    assignments = []
    for item in value["assignments"]:
        if (
            type(item) is not dict
            or set(item) != {"split_group_id", "split", "source_ids"}
            or type(item["source_ids"]) is not list
        ):
            raise ValueError("split ledger assignment keys are invalid")
        assignments.append(
            SplitAssignment(
                split_group_id=item["split_group_id"],
                split=item["split"],
                source_ids=tuple(item["source_ids"]),
            )
        )
    decoded = SplitLedger(
        build_id=value["build_id"],
        inventory_manifest_sha256=value["inventory_manifest_sha256"],
        seed=value["seed"],
        requested=value["requested"],
        actual_family_counts=value["actual_family_counts"],
        assignments=tuple(assignments),
        manifest_sha256=manifest_sha256,
    )
    split_ledger_bytes(decoded)
    expected = build_split_ledger(
        inventory, build_id=inventory.build_id, seed=decoded.seed
    )
    if split_ledger_bytes(decoded) != split_ledger_bytes(expected):
        raise ValueError(
            "split ledger does not match recomputed connected groups, coverage, "
            "or family counts"
        )
    return decoded


def _load_npy(path: Path, label: str) -> np.ndarray:
    with path.open("rb") as stream:
        values = np.load(stream, allow_pickle=False)
        if stream.read(1):
            raise ValueError(f"{label} has trailing bytes")
    return values


def load_lane(
    root: Path,
    *,
    inventory: FullWalkingInventory,
    split_ledger: SplitLedger,
    expected_manifest_sha256: str | None = None,
    expected_inventory_manifest_sha256: str | None = None,
    expected_split_ledger_manifest_sha256: str | None = None,
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
    inventory_sha256 = manifest["inventory_manifest_sha256"]
    split_sha256 = manifest["split_ledger_manifest_sha256"]
    if not _is_sha256(inventory_sha256):
        raise ValueError("lane inventory manifest SHA-256 is invalid")
    if not _is_sha256(split_sha256):
        raise ValueError("lane split ledger manifest SHA-256 is invalid")
    if expected_inventory_manifest_sha256 is not None and (
        not _is_sha256(expected_inventory_manifest_sha256)
        or inventory_sha256 != expected_inventory_manifest_sha256
    ):
        raise ValueError("lane inventory manifest SHA-256 mismatch")
    if expected_split_ledger_manifest_sha256 is not None and (
        not _is_sha256(expected_split_ledger_manifest_sha256)
        or split_sha256 != expected_split_ledger_manifest_sha256
    ):
        raise ValueError("lane split ledger manifest SHA-256 mismatch")
    if (
        manifest["fps"] != FPS
        or type(manifest["rows"]) is not int
        or type(manifest["ranges"]) is not int
        or manifest["rows"] < 1
        or manifest["ranges"] < 1
    ):
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
        if (
            type(descriptor) is not dict
            or set(descriptor) != _MEMBER_KEYS
            or descriptor["path"] != name
            or type(descriptor["size_bytes"]) is not int
            or descriptor["size_bytes"] < 0
            or not _is_sha256(descriptor["sha256"])
        ):
            raise ValueError(f"{name} member descriptor is invalid")
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"{name} SHA-256 or size mismatch")

    artifacts = read_holden_database(root / "database.bin")
    artifacts.terrain_features = _load_npy(
        root / "terrain_features.npy", "terrain_features.npy"
    )
    artifacts.terrain_support = _load_npy(
        root / "terrain_support.npy", "terrain_support.npy"
    )
    terrain_grid = _load_npy(root / "terrain_grid.npy", "terrain_grid.npy")
    source_ids_value = _parse_canonical_json(root / "source_ids.json", "source IDs")
    if type(source_ids_value) is not list or not all(
        type(source_id) is str for source_id in source_ids_value
    ):
        raise ValueError("lane source IDs must be a JSON string array")
    source_left_indices = _load_npy(
        root / "source_left_indices.npy", "source_left_indices.npy"
    )
    source_right_indices = _load_npy(
        root / "source_right_indices.npy", "source_right_indices.npy"
    )
    source_alpha = _load_npy(root / "source_alpha.npy", "source_alpha.npy")
    ranges_value = _parse_canonical_json(root / "ranges.json", "lane ranges")
    if type(ranges_value) is not list:
        raise ValueError("lane ranges must be a JSON array")
    ranges = tuple(_parse_range(value) for value in ranges_value)
    if (
        len(artifacts.positions) != manifest["rows"]
        or len(ranges) != manifest["ranges"]
    ):
        raise ValueError("lane manifest row/range counts do not match members")
    lane = LaneArtifact(
        root=root,
        manifest_sha256=manifest_sha256,
        inventory_manifest_sha256=inventory_sha256,
        split_ledger_manifest_sha256=split_sha256,
        artifacts=artifacts,
        terrain_grid=terrain_grid,
        source_ids=tuple(source_ids_value),
        source_left_indices=source_left_indices,
        source_right_indices=source_right_indices,
        source_alpha=source_alpha,
        ranges=ranges,
    )
    _validate_lane(lane, inventory=inventory, split_ledger=split_ledger)
    return lane


__all__ = [
    "FullWalkingInventory",
    "JSONValue",
    "LaneArtifact",
    "RangeRecord",
    "SourceRecord",
    "SplitAssignment",
    "SplitLedger",
    "build_split_ledger",
    "connected_split_groups",
    "inventory_manifest_bytes",
    "load_lane",
    "load_split_ledger",
    "publish_lane_exclusive",
    "split_ledger_bytes",
]
