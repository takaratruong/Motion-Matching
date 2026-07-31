"""Pinned bulk discovery and deterministic selection for GRAIL terrain data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Sequence

from .registry import ResolvedSource, SourceSpec


_SCHEMA = "g1-grail-terrain-inventory/v1"
_REPOSITORY_ID = "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
_REVISION = "943946a972d5de2eb0d2ff214b236d0e43575fd7"
_PARTITIONS = frozenset(("curb", "stair_p1", "stair_p2"))
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class BulkGrailCandidate:
    partition: str
    stem: str
    robot_relative_path: str
    object_relative_path: str
    usd_relative_path: str
    robot_bytes: int
    object_bytes: int | None
    usd_bytes: int
    robot_sha256: str
    object_sha256: str | None
    usd_sha256: str

    @property
    def logical_name(self) -> str:
        identity = hashlib.sha256(
            f"{self.partition}\0{self.stem}".encode("utf-8")
        ).hexdigest()[:20]
        return f"grail-{self.partition}-{identity}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("inventory file path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"inventory file path is unsafe: {value}")
    return value


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def _file_entry(value: object) -> tuple[str, int, str]:
    if not isinstance(value, dict) or set(value) != {
        "bytes",
        "path",
        "sha256",
    }:
        raise ValueError("inventory file entry is invalid")
    relative = _relative_path(value["path"])
    size = value["bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"inventory file size is invalid: {relative}")
    return relative, size, _digest(value["sha256"], relative)


def _requested_partitions(partitions: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(partitions)
    if not requested or any(partition not in _PARTITIONS for partition in requested):
        raise ValueError("requested GRAIL partition is invalid")
    if len(requested) != len(set(requested)):
        raise ValueError("requested GRAIL partitions contain a duplicate")
    return requested


def _load_inventory(root: Path) -> dict[str, tuple[int, str]]:
    path = root / "g1_mm_inventory.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"GRAIL inventory is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("GRAIL inventory is not valid JSON") from error
    if not isinstance(value, dict) or value.get("schema") != _SCHEMA:
        raise ValueError("GRAIL inventory schema is invalid")
    repository = value.get("repository")
    if (
        not isinstance(repository, dict)
        or repository.get("id") != _REPOSITORY_ID
        or repository.get("type") != "dataset"
    ):
        raise ValueError("GRAIL inventory repository identity is invalid")
    if repository.get("revision") != _REVISION:
        raise ValueError("GRAIL inventory repository revision is invalid")
    files = value.get("files")
    if not isinstance(files, list):
        raise ValueError("GRAIL inventory files must be a list")
    output: dict[str, tuple[int, str]] = {}
    for raw in files:
        relative, size, digest = _file_entry(raw)
        if relative in output:
            raise ValueError(f"GRAIL inventory contains duplicate path: {relative}")
        output[relative] = (size, digest)
    return output


def discover_bulk_grail_candidates(
    dataset_root: str | Path,
    partitions: Iterable[str],
) -> tuple[BulkGrailCandidate, ...]:
    root = Path(dataset_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"GRAIL dataset root is missing or unsafe: {root}")
    requested = _requested_partitions(partitions)
    inventory = _load_inventory(root)
    candidates: list[BulkGrailCandidate] = []
    for partition in requested:
        prefix = f"data/{partition}/robot/"
        robot_paths = sorted(
            relative
            for relative in inventory
            if relative.startswith(prefix) and relative.endswith(".pkl")
        )
        for robot_relative in robot_paths:
            stem = Path(robot_relative).stem
            object_relative = f"data/{partition}/objects/{stem}.pkl"
            usd_relative = f"data/{partition}/object_usd/{stem}.usd"
            if usd_relative not in inventory:
                raise ValueError(f"{stem}: paired USD is absent from inventory")
            object_identity = inventory.get(object_relative)
            if object_identity is None and partition != "curb":
                raise ValueError(
                    f"{stem}: paired object is absent from inventory"
                )
            object_path = root / object_relative
            if not object_path.exists():
                raise ValueError(f"{stem}: paired object file is missing")
            robot_bytes, robot_sha = inventory[robot_relative]
            usd_bytes, usd_sha = inventory[usd_relative]
            object_bytes, object_sha = (
                object_identity if object_identity is not None else (None, None)
            )
            candidates.append(
                BulkGrailCandidate(
                    partition=partition,
                    stem=stem,
                    robot_relative_path=robot_relative,
                    object_relative_path=object_relative,
                    usd_relative_path=usd_relative,
                    robot_bytes=robot_bytes,
                    object_bytes=object_bytes,
                    usd_bytes=usd_bytes,
                    robot_sha256=robot_sha,
                    object_sha256=object_sha,
                    usd_sha256=usd_sha,
                )
            )
    identities = [candidate.logical_name for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("bulk GRAIL candidates contain duplicate logical identity")
    return tuple(candidates)


def select_bulk_grail_candidates(
    candidates: Iterable[BulkGrailCandidate],
    *,
    limit_per_partition: int | None,
    seed: int,
) -> tuple[BulkGrailCandidate, ...]:
    values = tuple(candidates)
    if any(not isinstance(value, BulkGrailCandidate) for value in values):
        raise TypeError("bulk GRAIL candidates have an invalid type")
    identities = [(value.partition, value.stem) for value in values]
    if len(identities) != len(set(identities)):
        raise ValueError("bulk GRAIL candidates contain a duplicate")
    if (
        limit_per_partition is not None
        and (
            not isinstance(limit_per_partition, int)
            or isinstance(limit_per_partition, bool)
            or limit_per_partition <= 0
        )
    ):
        raise ValueError("limit_per_partition must be positive or None")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("selection seed must be an integer")

    selected: list[BulkGrailCandidate] = []
    for partition in sorted({value.partition for value in values}):
        group = sorted(
            (value for value in values if value.partition == partition),
            key=lambda value: value.stem,
        )
        if limit_per_partition is None or limit_per_partition >= len(group):
            selected.extend(group)
            continue
        count = limit_per_partition
        for interval in range(count):
            start = interval * len(group) // count
            stop = (interval + 1) * len(group) // count
            selected.append(
                min(
                    group[start:stop],
                    key=lambda value: hashlib.sha256(
                        (
                            f"{seed}\0{value.partition}\0{value.stem}"
                        ).encode("utf-8")
                    ).digest(),
                )
            )
    return tuple(sorted(selected, key=lambda value: (value.partition, value.stem)))


def select_named_bulk_grail_candidates(
    candidates: Sequence[BulkGrailCandidate],
    logical_names: Sequence[str],
) -> tuple[BulkGrailCandidate, ...]:
    values = tuple(candidates)
    if any(not isinstance(value, BulkGrailCandidate) for value in values):
        raise TypeError("bulk GRAIL candidates have an invalid type")
    requested = tuple(logical_names)
    if not requested:
        raise ValueError("named GRAIL selection must not be empty")
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("named GRAIL selection contains an invalid name")
    if len(requested) != len(set(requested)):
        raise ValueError("named GRAIL selection contains a duplicate")
    by_name = {candidate.logical_name: candidate for candidate in values}
    if len(by_name) != len(values):
        raise ValueError(
            "bulk GRAIL candidates contain duplicate logical identity"
        )
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"named GRAIL candidate not found: {missing[0]}")
    return tuple(by_name[name] for name in requested)


def _checked_file(
    root: Path,
    relative: str,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
) -> tuple[Path, str]:
    lexical = root / relative
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"bulk GRAIL path escapes dataset root: {relative}") from error
    if lexical.is_symlink() or not resolved.is_file():
        raise ValueError(f"bulk GRAIL source must be a real file: {relative}")
    if expected_bytes is not None and resolved.stat().st_size != expected_bytes:
        raise ValueError(f"bulk GRAIL source size changed: {relative}")
    actual = _sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"bulk GRAIL source SHA-256 changed: {relative}")
    return resolved, actual


def resolve_bulk_grail_candidates(
    dataset_root: str | Path,
    candidates: Iterable[BulkGrailCandidate],
) -> tuple[ResolvedSource, ...]:
    root = Path(dataset_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"GRAIL dataset root is missing or unsafe: {root}")
    output: list[ResolvedSource] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, BulkGrailCandidate):
            raise TypeError("bulk GRAIL candidates have an invalid type")
        if candidate.logical_name in seen:
            raise ValueError("bulk GRAIL candidates contain duplicate logical identity")
        seen.add(candidate.logical_name)
        robot, robot_sha = _checked_file(
            root,
            candidate.robot_relative_path,
            expected_bytes=candidate.robot_bytes,
            expected_sha256=candidate.robot_sha256,
        )
        objects, object_sha = _checked_file(
            root,
            candidate.object_relative_path,
            expected_bytes=candidate.object_bytes,
            expected_sha256=candidate.object_sha256,
        )
        usd, usd_sha = _checked_file(
            root,
            candidate.usd_relative_path,
            expected_bytes=candidate.usd_bytes,
            expected_sha256=candidate.usd_sha256,
        )
        spec = SourceSpec(
            logical_name=candidate.logical_name,
            family="grail",
            source_adapter="grail-record",
            motion_relative_path=candidate.robot_relative_path,
            motion_sha256=robot_sha,
            terrain_adapter="grail-usd",
            geometry_relative_paths=(
                candidate.object_relative_path,
                candidate.usd_relative_path,
            ),
            geometry_sha256=(object_sha, usd_sha),
        )
        output.append(
            ResolvedSource(
                spec=spec,
                motion_path=robot,
                geometry_paths=(objects, usd),
                source_sha256=MappingProxyType(
                    {
                        "motion": robot_sha,
                        f"geometry:{candidate.object_relative_path}": object_sha,
                        f"geometry:{candidate.usd_relative_path}": usd_sha,
                    }
                ),
            )
        )
    return tuple(output)
