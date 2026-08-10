"""Range-local 60 Hz derivation for the walking-only GRAIL/Takara lanes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from resources.build_g1_terrain_database import finalize_clip
from resources.g1_terrain_builder.artifacts import (
    canonical_json_bytes,
    read_support_sidecar,
    sha256_file,
)
from resources.g1_terrain_builder.database import (
    combine_clips,
    read_holden_database,
    refresh_lmm_clip_dynamics,
    write_holden_database,
)
from resources.g1_terrain_builder.kinematics import G1Kinematics, convert_source_clip
from resources.g1_terrain_builder.resample import (
    resample_map,
    resample_vectors,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    ArtifactSet,
    HoldenClip,
    SkeletonSpec,
)
from resources.g1_terrain_builder.sources import _grail_source_clip, load_takara
from resources.g1_terrain_builder.terrain import GrailTerrain

from .full_walking_terrain_lmm_contracts import (
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    _fsync_directory,
    _fsync_file,
    _rename_noreplace,
    load_split_ledger,
    publish_lane_exclusive,
)
from .full_walking_terrain_lmm_inventory import (
    STRICT_BANK_MANIFEST_SHA256,
    load_inventory,
)

_HEADER = struct.Struct("<4sIII")
_TAKARA_SHA256 = "adb2d1b624f84132d6768e9021b073db65c12ceb717c5b23d873fc11626ef7b7"
_TAKARA_REMAP_SHA256 = (
    "1195c702bcd6f9b13ed460bd553c0ef30369b0733642220bbf50809a008acf4a"
)
_G1_XML_SHA256 = "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
_INVENTORY_MANIFEST_SHA256 = (
    "7d1efd01abba6eea86280cb7720f999173fd8cedca89ddf56a61caad81a341e4"
)
_SPLIT_LEDGER_MANIFEST_SHA256 = (
    "5185bd42c153518c80b67c3dd68f54e9122e2e81970893209e9235ff5cc2feb5"
)
_INHERITED_WORK_AUTHORITY_SHA256 = (
    "e0e74435e084714a50d530b8d7dcb85b44becc7690f591151127d6287d5c6690"
)
_BANK_MEMBER_FIELDS = (
    ("database",),
    ("sidecars", "terrain_features"),
    ("sidecars", "terrain_support"),
    ("motion_index",),
    ("scene_index",),
    ("validation_file",),
)


def _publish_directory_exclusive(
    output: Path, write_members: Callable[[Path], None]
) -> Path:
    """Publish a complete work directory atomically and never replace it."""

    if not callable(write_members):
        raise TypeError("work-directory writer must be callable")
    destination = Path(output).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"inherited work output exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        write_members(staging)
        for member in sorted(staging.iterdir()):
            if member.is_symlink() or not member.is_file():
                raise ValueError("inherited work staging contains a non-file member")
            _fsync_file(member)
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return destination


def _load_frozen_authorities(inventory: Path, split_ledger: Path):
    inventory_value = load_inventory(
        Path(inventory), expected_manifest_sha256=_INVENTORY_MANIFEST_SHA256
    )
    ledger = load_split_ledger(
        Path(split_ledger),
        inventory=inventory_value,
        expected_manifest_sha256=_SPLIT_LEDGER_MANIFEST_SHA256,
    )
    return inventory_value, ledger


def _canonical_lane_artifacts(artifacts: ArtifactSet) -> ArtifactSet:
    """Return the exact little-endian, C-contiguous lane storage layout."""

    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("artifacts must be an ArtifactSet")
    return ArtifactSet(
        positions=np.ascontiguousarray(artifacts.positions, dtype="<f4"),
        velocities=np.ascontiguousarray(artifacts.velocities, dtype="<f4"),
        rotations=np.ascontiguousarray(artifacts.rotations, dtype="<f4"),
        angular_velocities=np.ascontiguousarray(
            artifacts.angular_velocities, dtype="<f4"
        ),
        parents=np.ascontiguousarray(artifacts.parents, dtype="<i4"),
        range_starts=np.ascontiguousarray(artifacts.range_starts, dtype="<i4"),
        range_stops=np.ascontiguousarray(artifacts.range_stops, dtype="<i4"),
        contacts=np.ascontiguousarray(artifacts.contacts, dtype="|u1"),
        terrain_features=np.ascontiguousarray(artifacts.terrain_features, dtype="<f4"),
        terrain_support=np.ascontiguousarray(artifacts.terrain_support, dtype="<f4"),
    )


def _reconstruct_inherited_terrain_channels(
    terrain_grid: np.ndarray,
    parent_support: np.ndarray,
    source_left_indices: np.ndarray,
    source_right_indices: np.ndarray,
    source_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover omitted v1 work channels from retained exact authorities."""

    grid = np.asarray(terrain_grid)
    support = np.asarray(parent_support)
    left = np.asarray(source_left_indices)
    right = np.asarray(source_right_indices)
    alpha = np.asarray(source_alpha)
    rows = len(grid)
    if (
        grid.shape != (rows, 36)
        or support.ndim != 2
        or support.shape[1] != 3
        or left.shape != (rows,)
        or right.shape != (rows,)
        or alpha.shape != (rows,)
        or np.any(left < 0)
        or np.any(right < left)
        or np.any(right >= len(support))
        or not np.isfinite(grid).all()
        or not np.isfinite(support).all()
        or not np.isfinite(alpha).all()
    ):
        raise ValueError("retained inherited terrain authority is invalid")
    features = np.ascontiguousarray(grid[:, (1, 3, 5, 7)], dtype="<f4")
    blend = alpha.astype(np.float64)[:, None]
    reconstructed_support = (
        support[left].astype(np.float64) * (1.0 - blend)
        + support[right].astype(np.float64) * blend
    )
    return features, np.ascontiguousarray(reconstructed_support, dtype="<f4")


@dataclass(frozen=True)
class BroadBankAuthority:
    root: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    sources: tuple[dict[str, Any], ...]
    members: dict[str, Path]


def _canonical_manifest(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("broad-bank manifest is not UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("broad-bank manifest must be a JSON object")
    return payload, value


def authenticate_broad_bank(bank: Path) -> BroadBankAuthority:
    """Authenticate the frozen broad bank before any binary decoding."""

    root = Path(bank).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("broad-bank authority must be a directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("broad-bank manifest must be a regular non-symlink file")
    digest = sha256_file(manifest_path)
    if digest != STRICT_BANK_MANIFEST_SHA256:
        raise ValueError("broad-bank manifest SHA-256 mismatch")
    _payload, manifest = _canonical_manifest(manifest_path)
    if (
        manifest.get("schema") != "g1-terrain-artifacts/v3"
        or manifest.get("database_frames") != 3_970_932
        or manifest.get("output_fps") != 25.0
        or manifest.get("total_clips") != 15_815
        or manifest.get("grail_clips") != 15_814
        or manifest.get("skipped_clips") != 23
    ):
        raise ValueError("broad-bank cardinality/rate contract changed")
    sources = manifest.get("sources")
    if type(sources) is not list or len(sources) != 15_815:
        raise ValueError("broad-bank source inventory changed")
    members: dict[str, Path] = {}
    for field_path in _BANK_MEMBER_FIELDS:
        descriptor: object = manifest
        for field in field_path:
            if type(descriptor) is not dict:
                break
            descriptor = descriptor.get(field)
        label = ".".join(field_path)
        if (
            type(descriptor) is not dict
            or type(descriptor.get("path")) is not str
            or type(descriptor.get("sha256")) is not str
        ):
            raise ValueError(f"broad-bank {label} descriptor is invalid")
        candidate = (root / descriptor["path"]).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"broad-bank {label} path escapes root") from error
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"broad-bank {label} is not a regular file")
        if sha256_file(candidate) != descriptor["sha256"]:
            raise ValueError(f"broad-bank {label} SHA-256 mismatch")
        members[label] = candidate
    return BroadBankAuthority(root, digest, manifest, tuple(sources), members)


def missing_slope_source_ids(
    inventory_path: Path, authority: BroadBankAuthority
) -> tuple[str, ...]:
    """Return the exact raw GRAIL slope identities absent from the bank."""

    if not isinstance(authority, BroadBankAuthority):
        raise TypeError("authority must be an authenticated BroadBankAuthority")
    inventory = load_inventory(
        Path(inventory_path), expected_manifest_sha256=_INVENTORY_MANIFEST_SHA256
    )
    all_slopes = {
        record.source_id
        for record in inventory.sources
        if record.authority.get("kind") == "grail" and record.family == "slope"
    }
    bank_stems = {
        source["name"]
        for source in authority.sources
        if source.get("terrain_family") == "slope"
    }
    inherited = {
        source_id
        for source_id in all_slopes
        if source_id.rsplit(":", 1)[-1] in bank_stems
    }
    missing = tuple(sorted(all_slopes - inherited))
    if len(all_slopes) != 1_880 or len(inherited) != 1_857 or len(missing) != 23:
        raise ValueError("GRAIL missing-slope inventory changed")
    return missing


def _read_g1tf_v2_raw(path: Path) -> np.ndarray:
    with Path(path).open("rb") as stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("G1TF/v2 header is truncated")
        magic, version, frames, dimensions = _HEADER.unpack(header)
        if (magic, version, dimensions) != (b"G1TF", 2, 12):
            raise ValueError("terrain sidecar must be exact G1TF/v2")
        payload = stream.read()
    expected = frames * dimensions * np.dtype("<f4").itemsize
    if len(payload) != expected:
        raise ValueError("G1TF/v2 payload size changed")
    values = np.frombuffer(payload, dtype="<f4").reshape(frames, dimensions).copy()
    if not np.isfinite(values).all():
        raise ValueError("G1TF/v2 values must be finite")
    return values


def _compatibility_terrain_grid(terrain_v2: np.ndarray) -> np.ndarray:
    """Retain every authenticated v2 value in a deterministic 36-D view.

    The inherited bank contains eight longitudinal and four corridor values,
    not PFNN's native grid.  Repeating the complete descriptor across the three
    lane slots is lossless and explicitly remains an inherited compatibility
    view; source-native PFNN lanes publish their true 3x12 grid.
    """

    values = np.asarray(terrain_v2)
    if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError("inherited terrain descriptor must be finite (rows, 12)")
    return np.ascontiguousarray(np.tile(values, (1, 3)), dtype=np.float32)


def _terrain_grid_from_four_features(features: np.ndarray) -> np.ndarray:
    """Encode four longitudinal heights in G1TF/v2-compatible columns."""

    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("terrain features must be finite (rows, 4)")
    descriptor = np.zeros((len(values), 12), dtype=np.float32)
    descriptor[:, (0, 1)] = values[:, 0, None]
    descriptor[:, (2, 3)] = values[:, 1, None]
    descriptor[:, (4, 5)] = values[:, 2, None]
    descriptor[:, (6, 7)] = values[:, 3, None]
    return _compatibility_terrain_grid(descriptor)


def _clip_from_parent(
    parent: Any, terrain_v2: np.ndarray, start: int, stop: int, name: str
) -> HoldenClip:
    rows = np.arange(start, stop, dtype=np.int32)
    local_rows = np.arange(stop - start, dtype=np.int32)
    return HoldenClip(
        name=name,
        positions=np.ascontiguousarray(parent.positions[start:stop], dtype=np.float32),
        velocities=np.ascontiguousarray(
            parent.velocities[start:stop], dtype=np.float32
        ),
        rotations=np.ascontiguousarray(parent.rotations[start:stop], dtype=np.float32),
        angular_velocities=np.ascontiguousarray(
            parent.angular_velocities[start:stop], dtype=np.float32
        ),
        contacts=np.ascontiguousarray(parent.contacts[start:stop], dtype=np.uint8),
        terrain_features=np.ascontiguousarray(
            terrain_v2[start:stop, (1, 3, 5, 7)], dtype=np.float32
        ),
        terrain_support=np.ascontiguousarray(
            parent.terrain_support[start:stop], dtype=np.float32
        ),
        source_frames=rows.copy(),
        terrain_id=name,
        source_left_indices=local_rows.copy(),
        source_right_indices=local_rows.copy(),
        source_alpha=np.zeros(len(rows), dtype=np.float32),
    )


def _map_sha(values: np.ndarray, dtype: str) -> str:
    payload = np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def resample_inherited_range(
    parent: Any, range_index: int, record: SourceRecord
) -> tuple[HoldenClip, RangeRecord]:
    """Derive one parent range; split metadata is injected by the lane builder."""

    if not isinstance(record, SourceRecord):
        raise TypeError("record must be a SourceRecord")
    authority = dict(record.authority)
    required = {"_split_group_id", "_split", "_parent_manifest_sha256"}
    if not required.issubset(authority):
        raise ValueError("source record lacks authenticated split/parent context")
    start = int(parent.range_starts[range_index])
    stop = int(parent.range_stops[range_index])
    terrain_v2 = authority.pop("_terrain_v2")
    skeleton = SkeletonSpec(
        G1_SKELETON_NAMES, np.asarray(G1_SKELETON_PARENTS, dtype=np.int32)
    )
    source = _clip_from_parent(parent, terrain_v2, start, stop, record.source_id)
    result, mapping = resample_holden_range(
        source, source_fps=25.0, target_fps=60.0, skeleton=skeleton
    )
    range_record = RangeRecord(
        range_id=f"grail-inherited:{range_index:05d}:{record.source_id}",
        canonical_source_id=record.canonical_source_id,
        terrain_id=record.terrain_id,
        mirror_of=record.mirror_of,
        family=record.family,
        split_group_id=str(authority.pop("_split_group_id")),
        split=str(authority.pop("_split")),
        start=0,
        stop=len(result.positions),
        quality="clean",
        authority={
            "kind": "derived-grail-parent-range",
            "parent_manifest_sha256": str(authority.pop("_parent_manifest_sha256")),
            "parent_range_index": int(range_index),
            "parent_start": start,
            "parent_stop": stop,
            "source_frame_count": stop - start,
            "source_map": {
                "schema": "range-local-resample-map/v1",
                "target": "source-range-local-25hz/v1",
                "source_fps": 25.0,
                "target_fps": 60.0,
                "rows": len(result.positions),
                "left_sha256": _map_sha(result.source_left_indices, "<i4"),
                "right_sha256": _map_sha(result.source_right_indices, "<i4"),
                "alpha_sha256": _map_sha(result.source_alpha, "<f4"),
                "local_left_sha256": _map_sha(mapping["left"], "<i4"),
                "local_right_sha256": _map_sha(mapping["right"], "<i4"),
            },
        },
    )
    return result, range_record


def build_inherited_grail_lane(
    *,
    bank: Path,
    inventory: Path,
    split_ledger: Path,
    output: Path,
    work_output: Path | None = None,
) -> Path:
    """Build all 15,814 inherited GRAIL ranges as one immutable 60 Hz lane."""

    inventory_value, ledger = _load_frozen_authorities(inventory, split_ledger)
    authority = authenticate_broad_bank(Path(bank))
    parent = read_holden_database(authority.members["database"])
    terrain_v2 = _read_g1tf_v2_raw(authority.members["sidecars.terrain_features"])
    parent.terrain_support = read_support_sidecar(
        authority.members["sidecars.terrain_support"]
    )
    if (
        len(parent.positions) != 3_970_932
        or terrain_v2.shape != (3_970_932, 12)
        or parent.terrain_support.shape != (3_970_932, 3)
        or len(parent.range_starts) != 15_815
    ):
        raise ValueError("broad-bank decoded dimensions changed")
    source_by_stem = {
        record.source_id.rsplit(":", 1)[-1]: record
        for record in inventory_value.sources
        if record.authority.get("kind") == "grail"
    }
    assignment_by_source = {
        source_id: assignment
        for assignment in ledger.assignments
        for source_id in assignment.source_ids
    }
    clips: list[HoldenClip] = []
    terrain_grids: list[np.ndarray] = []
    ranges: list[RangeRecord] = []
    source_ids: list[str] = []
    left_maps: list[np.ndarray] = []
    right_maps: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    cursor = 0
    for range_index, source_claim in enumerate(authority.sources[1:], start=1):
        name = source_claim.get("name")
        if type(name) is not str or name not in source_by_stem:
            raise ValueError(f"bank source {range_index} has no inventory identity")
        record = source_by_stem[name]
        assignment = assignment_by_source.get(record.source_id)
        if assignment is None:
            raise ValueError(f"bank source {record.source_id} has no split assignment")
        parent_start = int(parent.range_starts[range_index])
        parent_stop = int(parent.range_stops[range_index])
        if (
            source_claim.get("range_start") != parent_start
            or source_claim.get("range_stop") != parent_stop
            or parent_stop - parent_start != 250
            or source_claim.get("output_frames") != 250
            or source_claim.get("source_fps") != 25.0
        ):
            raise ValueError(f"bank source/range mismatch at index {range_index}")
        contextual = replace(
            record,
            authority={
                **record.authority,
                "_split_group_id": assignment.split_group_id,
                "_split": assignment.split,
                "_parent_manifest_sha256": authority.manifest_sha256,
                "_terrain_v2": terrain_v2,
            },
        )
        clip, range_record = resample_inherited_range(parent, range_index, contextual)
        grid = resample_terrain_grid(
            _compatibility_terrain_grid(terrain_v2[parent_start:parent_stop]),
            source_fps=25.0,
            target_fps=60.0,
        )
        stop = cursor + len(clip.positions)
        ranges.append(replace(range_record, start=cursor, stop=stop))
        clips.append(clip)
        terrain_grids.append(grid)
        source_ids.append(record.source_id)
        left_maps.append(clip.source_left_indices)
        right_maps.append(clip.source_right_indices)
        alphas.append(clip.source_alpha)
        cursor = stop
        if range_index % 500 == 0:
            print(
                json.dumps(
                    {
                        "phase": "derive-inherited-grail",
                        "ranges": range_index,
                        "rows": cursor,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(clips) != 15_814 or cursor != 9_456_772:
        raise ValueError("inherited GRAIL lane cardinality changed")
    skeleton = SkeletonSpec(
        G1_SKELETON_NAMES, np.asarray(G1_SKELETON_PARENTS, dtype=np.int32)
    )
    artifacts = _canonical_lane_artifacts(combine_clips(clips, skeleton))
    combined_grid = np.concatenate(terrain_grids).astype(np.float32, copy=False)
    combined_left = np.concatenate(left_maps).astype(np.int32, copy=False)
    combined_right = np.concatenate(right_maps).astype(np.int32, copy=False)
    combined_alpha = np.concatenate(alphas).astype(np.float32, copy=False)
    if work_output is not None:

        def write_work_members(staging: Path) -> None:
            write_holden_database(staging / "database.bin", artifacts)
            for name, values in (
                ("terrain_grid.npy", combined_grid),
                ("terrain_features.npy", artifacts.terrain_features),
                ("terrain_support.npy", artifacts.terrain_support),
                ("source_left_indices.npy", combined_left),
                ("source_right_indices.npy", combined_right),
                ("source_alpha.npy", combined_alpha),
            ):
                with (staging / name).open("xb") as stream:
                    np.save(stream, values, allow_pickle=False)
            (staging / "source_ids.json").write_bytes(canonical_json_bytes(source_ids))
            (staging / "ranges.json").write_bytes(
                canonical_json_bytes([asdict(record) for record in ranges])
            )
            members = {}
            for path in sorted(staging.iterdir()):
                members[path.name] = {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            authority_payload = canonical_json_bytes(
                {
                    "schema": "g1-full-walking-inherited-work/v2",
                    "bank_manifest_sha256": authority.manifest_sha256,
                    "inventory_manifest_sha256": inventory_value.manifest_sha256,
                    "split_ledger_manifest_sha256": ledger.manifest_sha256,
                    "source_map_target": "source-range-local-25hz/v1",
                    "rows": cursor,
                    "ranges": len(ranges),
                    "members": members,
                }
            )
            if (
                hashlib.sha256(authority_payload).hexdigest()
                != _INHERITED_WORK_AUTHORITY_SHA256
            ):
                raise ValueError("inherited work reproduction digest changed")
            (staging / "authority.json").write_bytes(authority_payload)

        return _publish_directory_exclusive(Path(work_output), write_work_members)
    lane = LaneArtifact(
        root=Path("."),
        manifest_sha256="",
        inventory_manifest_sha256=inventory_value.manifest_sha256,
        split_ledger_manifest_sha256=ledger.manifest_sha256,
        artifacts=artifacts,
        terrain_grid=combined_grid,
        source_ids=tuple(source_ids),
        source_left_indices=combined_left,
        source_right_indices=combined_right,
        source_alpha=combined_alpha,
        ranges=tuple(ranges),
    )
    return publish_lane_exclusive(lane, Path(output))


def _read_canonical_json(path: Path, label: str) -> Any:
    payload = Path(path).read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _load_work_npy(path: Path, *, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    if values.dtype != np.dtype(dtype) or values.shape != shape:
        raise ValueError(f"{Path(path).name} dtype or shape changed")
    return values


def load_inherited_grail_work(
    work: Path,
    *,
    inventory: Path,
    split_ledger: Path,
) -> LaneArtifact:
    """Rebind the retained expensive GRAIL payload to a hardened split ledger."""

    root = Path(work).resolve(strict=True)
    expected_names = {
        "authority.json",
        "database.bin",
        "ranges.json",
        "source_alpha.npy",
        "source_ids.json",
        "source_left_indices.npy",
        "source_right_indices.npy",
        "terrain_grid.npy",
        "terrain_features.npy",
        "terrain_support.npy",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("inherited GRAIL work members changed")
    authority_path = root / "authority.json"
    if sha256_file(authority_path) != _INHERITED_WORK_AUTHORITY_SHA256:
        raise ValueError("inherited GRAIL work authority SHA-256 changed")
    authority = _read_canonical_json(authority_path, "work authority")
    inventory_value, ledger = _load_frozen_authorities(inventory, split_ledger)
    member_names = expected_names - {"authority.json"}
    if (
        type(authority) is not dict
        or set(authority)
        != {
            "schema",
            "bank_manifest_sha256",
            "inventory_manifest_sha256",
            "split_ledger_manifest_sha256",
            "source_map_target",
            "rows",
            "ranges",
            "members",
        }
        or authority["schema"] != "g1-full-walking-inherited-work/v2"
        or authority["bank_manifest_sha256"] != STRICT_BANK_MANIFEST_SHA256
        or authority["inventory_manifest_sha256"] != inventory_value.manifest_sha256
        or authority["split_ledger_manifest_sha256"] != ledger.manifest_sha256
        or authority["source_map_target"] != "source-range-local-25hz/v1"
        or authority["rows"] != 9_456_772
        or authority["ranges"] != 15_814
        or type(authority["members"]) is not dict
        or set(authority["members"]) != member_names
    ):
        raise ValueError("inherited GRAIL work authority changed")
    for name in member_names:
        descriptor = authority["members"][name]
        path = root / name
        if (
            type(descriptor) is not dict
            or set(descriptor) != {"path", "size_bytes", "sha256"}
            or descriptor["path"] != name
            or descriptor["size_bytes"] != path.stat().st_size
            or descriptor["sha256"] != sha256_file(path)
        ):
            raise ValueError(f"inherited GRAIL work member changed: {name}")
    source_ids_value = _read_canonical_json(root / "source_ids.json", "source IDs")
    ranges_value = _read_canonical_json(root / "ranges.json", "ranges")
    if (
        type(source_ids_value) is not list
        or len(source_ids_value) != 15_814
        or not all(type(source_id) is str for source_id in source_ids_value)
        or type(ranges_value) is not list
        or len(ranges_value) != 15_814
    ):
        raise ValueError("inherited GRAIL source/range cardinality changed")
    try:
        ranges = tuple(RangeRecord(**value) for value in ranges_value)
    except TypeError as error:
        raise ValueError("inherited GRAIL ranges have invalid fields") from error
    assignment_by_source = {
        source_id: assignment
        for assignment in ledger.assignments
        for source_id in assignment.source_ids
    }
    expected_start = 0
    for source_id, record in zip(source_ids_value, ranges, strict=True):
        assignment = assignment_by_source.get(source_id)
        if (
            assignment is None
            or record.start != expected_start
            or record.stop - record.start != 598
            or record.split_group_id != assignment.split_group_id
            or record.split != assignment.split
        ):
            raise ValueError("work range disagrees with the hardened split ledger")
        expected_start = record.stop
    if expected_start != 9_456_772:
        raise ValueError("work ranges do not exactly cover inherited rows")
    artifacts = read_holden_database(root / "database.bin")
    if (
        artifacts.positions.shape != (9_456_772, 31, 3)
        or len(artifacts.range_starts) != 15_814
    ):
        raise ValueError("inherited database dimensions changed")
    rows = len(artifacts.positions)
    artifacts.terrain_features = _load_work_npy(
        root / "terrain_features.npy", dtype="<f4", shape=(rows, 4)
    )
    artifacts.terrain_support = _load_work_npy(
        root / "terrain_support.npy", dtype="<f4", shape=(rows, 3)
    )
    return LaneArtifact(
        root=Path("."),
        manifest_sha256="",
        inventory_manifest_sha256=inventory_value.manifest_sha256,
        split_ledger_manifest_sha256=ledger.manifest_sha256,
        artifacts=artifacts,
        terrain_grid=_load_work_npy(
            root / "terrain_grid.npy", dtype="<f4", shape=(rows, 36)
        ),
        source_ids=tuple(source_ids_value),
        source_left_indices=_load_work_npy(
            root / "source_left_indices.npy", dtype="<i4", shape=(rows,)
        ),
        source_right_indices=_load_work_npy(
            root / "source_right_indices.npy", dtype="<i4", shape=(rows,)
        ),
        source_alpha=_load_work_npy(
            root / "source_alpha.npy", dtype="<f4", shape=(rows,)
        ),
        ranges=ranges,
    )


def restore_inherited_work_terrain(work: Path, *, bank: Path) -> Path:
    """Restore v1 work terrain/support sidecars from retained exact inputs."""

    root = Path(work).resolve(strict=True)
    feature_path = root / "terrain_features.npy"
    support_path = root / "terrain_support.npy"
    if feature_path.exists() or support_path.exists():
        raise FileExistsError("inherited work terrain restoration already exists")
    authority = authenticate_broad_bank(Path(bank))
    grid = np.load(root / "terrain_grid.npy", mmap_mode="r", allow_pickle=False)
    left = np.load(root / "source_left_indices.npy", mmap_mode="r", allow_pickle=False)
    right = np.load(
        root / "source_right_indices.npy", mmap_mode="r", allow_pickle=False
    )
    alpha = np.load(root / "source_alpha.npy", mmap_mode="r", allow_pickle=False)
    parent_support = read_support_sidecar(authority.members["sidecars.terrain_support"])
    features, support = _reconstruct_inherited_terrain_channels(
        grid, parent_support, left, right, alpha
    )
    try:
        with feature_path.open("xb") as stream:
            np.save(stream, features, allow_pickle=False)
        with support_path.open("xb") as stream:
            np.save(stream, support, allow_pickle=False)
    except BaseException:
        feature_path.unlink(missing_ok=True)
        support_path.unlink(missing_ok=True)
        raise
    return root


def publish_inherited_grail_work(
    work: Path,
    *,
    inventory: Path,
    split_ledger: Path,
    output: Path,
) -> Path:
    lane = load_inherited_grail_work(
        work, inventory=inventory, split_ledger=split_ledger
    )
    return publish_lane_exclusive(lane, Path(output))


def _require_digest(path: Path, expected: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or sha256_file(resolved) != expected
    ):
        raise ValueError(f"{label} SHA-256 changed")
    return resolved


def build_takara_lane(
    *,
    source: Path,
    remap: Path,
    inventory: Path,
    split_ledger: Path,
    g1_xml: Path,
    output: Path,
) -> Path:
    """Convert the complete authenticated Takara walk directly 50 -> 60 Hz."""

    source = _require_digest(source, _TAKARA_SHA256, "Takara motion")
    remap = _require_digest(remap, _TAKARA_REMAP_SHA256, "Takara remap")
    g1_xml = _require_digest(g1_xml, _G1_XML_SHA256, "canonical G1 XML")
    inventory_value, ledger = _load_frozen_authorities(inventory, split_ledger)
    records = [
        record
        for record in inventory_value.sources
        if record.source_id == "takara:takara_walk_50hz"
    ]
    if len(records) != 1:
        raise ValueError("Takara inventory identity changed")
    record = records[0]
    assignment = next(
        (item for item in ledger.assignments if record.source_id in item.source_ids),
        None,
    )
    if assignment is None:
        raise ValueError("Takara split assignment is missing")
    native = load_takara(str(source), str(remap))
    if native.fps != 50.0 or native.qpos.shape != (34_863, 36):
        raise ValueError("Takara native rate/frame contract changed")
    clip, skeleton, report = convert_source_clip(
        native,
        G1Kinematics(str(g1_xml)),
        target_fps=60.0,
        root_filter_mode="nearest",
    )
    clip = refresh_lmm_clip_dynamics(clip, skeleton, 60.0)
    if len(clip.positions) != 41_835:
        raise ValueError("Takara 50 -> 60 Hz row count changed")
    range_record = RangeRecord(
        range_id="takara:takara_walk_50hz:full",
        canonical_source_id=record.canonical_source_id,
        terrain_id=record.terrain_id,
        mirror_of=None,
        family="flat",
        split_group_id=assignment.split_group_id,
        split=assignment.split,
        start=0,
        stop=len(clip.positions),
        quality="usable",
        authority={
            "kind": "source-native-takara",
            "source_frame_count": 34_863,
            "motion_sha256": _TAKARA_SHA256,
            "remap_sha256": _TAKARA_REMAP_SHA256,
            "g1_xml_sha256": _G1_XML_SHA256,
            "conversion": {
                key: float(value) if isinstance(value, (float, np.floating)) else value
                for key, value in report.items()
            },
            "source_map": {
                "left_sha256": _map_sha(clip.source_left_indices, "<i4"),
                "right_sha256": _map_sha(clip.source_right_indices, "<i4"),
                "alpha_sha256": _map_sha(clip.source_alpha, "<f4"),
            },
        },
    )
    lane = LaneArtifact(
        root=Path("."),
        manifest_sha256="",
        inventory_manifest_sha256=inventory_value.manifest_sha256,
        split_ledger_manifest_sha256=ledger.manifest_sha256,
        artifacts=_canonical_lane_artifacts(combine_clips([clip], skeleton)),
        terrain_grid=np.zeros((len(clip.positions), 36), dtype=np.float32),
        source_ids=(record.source_id,),
        source_left_indices=clip.source_left_indices,
        source_right_indices=clip.source_right_indices,
        source_alpha=clip.source_alpha,
        ranges=(range_record,),
    )
    return publish_lane_exclusive(lane, Path(output))


def _authenticated_inventory_payload(
    root: Path, descriptor: Any, label: str
) -> tuple[Path, bytes]:
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"path", "size_bytes", "sha256"}
        or type(descriptor["path"]) is not str
        or type(descriptor["size_bytes"]) is not int
        or type(descriptor["sha256"]) is not str
    ):
        raise ValueError(f"{label} inventory descriptor is invalid")
    root = Path(root).resolve(strict=True)
    path = (root / descriptor["path"]).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes GRAIL root") from error
    payload = path.read_bytes()
    if (
        len(payload) != descriptor["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError(f"{label} source bytes changed")
    return path, payload


def build_missing_slope_lane(
    *,
    grail_root: Path,
    inventory: Path,
    split_ledger: Path,
    g1_xml: Path,
    output: Path,
) -> Path:
    """Process the exact 23 slope identities absent from the broad bank."""

    g1_xml = _require_digest(g1_xml, _G1_XML_SHA256, "canonical G1 XML")
    inventory_value, ledger = _load_frozen_authorities(inventory, split_ledger)
    bank_descriptor = next(
        record.authority["bank_manifest_sha256"]
        for record in inventory_value.sources
        if record.authority.get("kind") == "grail"
    )
    if bank_descriptor != STRICT_BANK_MANIFEST_SHA256:
        raise ValueError("GRAIL inventory does not bind the strict bank")
    # Derive the missing set from the already authenticated manifest claims;
    # the source bytes themselves are independently reauthenticated below.
    bank_root = Path(
        "/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate"
    )
    bank_authority = authenticate_broad_bank(bank_root)
    missing_ids = missing_slope_source_ids(Path(inventory), bank_authority)
    by_id = {record.source_id: record for record in inventory_value.sources}
    assignment_by_source = {
        source_id: assignment
        for assignment in ledger.assignments
        for source_id in assignment.source_ids
    }
    kinematics = G1Kinematics(str(g1_xml))
    clips: list[HoldenClip] = []
    ranges: list[RangeRecord] = []
    grids: list[np.ndarray] = []
    source_ids: list[str] = []
    cursor = 0
    for source_id in missing_ids:
        record = by_id[source_id]
        assignment = assignment_by_source[source_id]
        inputs = record.authority["inputs"]
        authenticated = {
            role: _authenticated_inventory_payload(
                Path(grail_root), descriptor, f"{source_id} {role}"
            )
            for role, descriptor in inputs.items()
        }
        robot_path, robot_payload = authenticated["robot"]
        usd_path, usd_payload = authenticated["object_usd"]
        recon_path, recon_payload = authenticated["recon"]
        try:
            robot_blob = joblib.load(io.BytesIO(robot_payload))
        except Exception as error:
            raise ValueError(f"{source_id} robot payload failed decode") from error
        source_clip = _grail_source_clip(robot_blob, str(robot_path))
        terrain = GrailTerrain.from_authenticated_bytes(
            usd_payload,
            recon_payload,
            usd_source=str(usd_path),
            reconstruction_source=str(recon_path),
            support_calibration_m=0.0,
        )
        clip, skeleton, report = finalize_clip(
            source_clip, terrain, kinematics, output_fps=60.0
        )
        clip = refresh_lmm_clip_dynamics(clip, skeleton, 60.0)
        if len(clip.positions) != 598:
            raise ValueError(f"{source_id} 25 -> 60 Hz row count changed")
        stop = cursor + len(clip.positions)
        ranges.append(
            RangeRecord(
                range_id=f"grail-missing-slope:{source_id}",
                canonical_source_id=record.canonical_source_id,
                terrain_id=record.terrain_id,
                mirror_of=None,
                family="slope",
                split_group_id=assignment.split_group_id,
                split=assignment.split,
                start=cursor,
                stop=stop,
                quality="usable",
                authority={
                    "kind": "source-native-grail-slope",
                    "source_frame_count": 250,
                    "g1_xml_sha256": _G1_XML_SHA256,
                    "terrain_grid_semantics": (
                        "g1tf-v2-four-height-compatibility-3lane/v1"
                    ),
                    "input_sha256": {
                        role: descriptor["sha256"]
                        for role, descriptor in inputs.items()
                    },
                    "conversion": {
                        key: float(value)
                        if isinstance(value, (float, np.floating))
                        else value
                        for key, value in report.items()
                    },
                },
            )
        )
        clips.append(clip)
        grids.append(_terrain_grid_from_four_features(clip.terrain_features))
        source_ids.append(source_id)
        cursor = stop
    if len(clips) != 23 or cursor != 13_754:
        raise ValueError("missing-slope lane cardinality changed")
    artifacts = _canonical_lane_artifacts(combine_clips(clips, skeleton))
    lane = LaneArtifact(
        root=Path("."),
        manifest_sha256="",
        inventory_manifest_sha256=inventory_value.manifest_sha256,
        split_ledger_manifest_sha256=ledger.manifest_sha256,
        artifacts=artifacts,
        terrain_grid=np.concatenate(grids).astype(np.float32, copy=False),
        source_ids=tuple(source_ids),
        source_left_indices=np.concatenate(
            [clip.source_left_indices for clip in clips]
        ).astype(np.int32, copy=False),
        source_right_indices=np.concatenate(
            [clip.source_right_indices for clip in clips]
        ).astype(np.int32, copy=False),
        source_alpha=np.concatenate([clip.source_alpha for clip in clips]).astype(
            np.float32, copy=False
        ),
        ranges=tuple(ranges),
    )
    return publish_lane_exclusive(lane, Path(output))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inherited = commands.add_parser("inherited")
    inherited.add_argument("--bank", type=Path, required=True)
    inherited.add_argument("--inventory", type=Path, required=True)
    inherited.add_argument("--split-ledger", type=Path, required=True)
    inherited.add_argument("--output", type=Path, required=True)
    inherited.add_argument("--work-output", type=Path)
    publish_work = commands.add_parser("publish-inherited-work")
    publish_work.add_argument("--work", type=Path, required=True)
    publish_work.add_argument("--inventory", type=Path, required=True)
    publish_work.add_argument("--split-ledger", type=Path, required=True)
    publish_work.add_argument("--output", type=Path, required=True)
    restore_work = commands.add_parser("restore-inherited-terrain")
    restore_work.add_argument("--work", type=Path, required=True)
    restore_work.add_argument("--bank", type=Path, required=True)
    takara = commands.add_parser("takara")
    takara.add_argument("--source", type=Path, required=True)
    takara.add_argument("--remap", type=Path, required=True)
    takara.add_argument("--inventory", type=Path, required=True)
    takara.add_argument("--split-ledger", type=Path, required=True)
    takara.add_argument("--g1-xml", type=Path, required=True)
    takara.add_argument("--output", type=Path, required=True)
    slopes = commands.add_parser("missing-slopes")
    slopes.add_argument("--grail-root", type=Path, required=True)
    slopes.add_argument("--inventory", type=Path, required=True)
    slopes.add_argument("--split-ledger", type=Path, required=True)
    slopes.add_argument("--g1-xml", type=Path, required=True)
    slopes.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inherited":
        output = build_inherited_grail_lane(
            bank=args.bank,
            inventory=args.inventory,
            split_ledger=args.split_ledger,
            output=args.output,
            work_output=args.work_output,
        )
        print(json.dumps({"path": str(output)}, sort_keys=True), flush=True)
        return 0
    if args.command == "missing-slopes":
        output = build_missing_slope_lane(
            grail_root=args.grail_root,
            inventory=args.inventory,
            split_ledger=args.split_ledger,
            g1_xml=args.g1_xml,
            output=args.output,
        )
        print(json.dumps({"path": str(output)}, sort_keys=True), flush=True)
        return 0
    if args.command == "publish-inherited-work":
        output = publish_inherited_grail_work(
            args.work,
            inventory=args.inventory,
            split_ledger=args.split_ledger,
            output=args.output,
        )
        print(json.dumps({"path": str(output)}, sort_keys=True), flush=True)
        return 0
    if args.command == "restore-inherited-terrain":
        output = restore_inherited_work_terrain(args.work, bank=args.bank)
        print(json.dumps({"path": str(output)}, sort_keys=True), flush=True)
        return 0
    if args.command == "takara":
        output = build_takara_lane(
            source=args.source,
            remap=args.remap,
            inventory=args.inventory,
            split_ledger=args.split_ledger,
            g1_xml=args.g1_xml,
            output=args.output,
        )
        print(json.dumps({"path": str(output)}, sort_keys=True), flush=True)
        return 0
    raise AssertionError(args.command)


def _finite_float32(values: np.ndarray, label: str) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


def resample_terrain_grid(
    terrain_grid: np.ndarray, *, source_fps: float, target_fps: float = 60.0
) -> np.ndarray:
    """Linearly resample one range's 3-lane by 12-station terrain grid."""

    grid = np.asarray(terrain_grid)
    if grid.ndim != 2 or grid.shape[1] != 36:
        raise ValueError("terrain grid must have exact shape (frames, 36)")
    return _finite_float32(
        resample_vectors(grid, source_fps, target_fps), "terrain grid"
    )


def _resample_quaternions_vectorized(
    values: np.ndarray, *, source_fps: float, target_fps: float
) -> np.ndarray:
    """Sign-continuous WXYZ SLERP without per-frame Python loops."""

    quaternions = np.asarray(values, dtype=np.float64).copy()
    if quaternions.ndim < 2 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must end in a four-component WXYZ axis")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.isfinite(quaternions).all() or np.any(norms < 1e-12):
        raise ValueError("quaternion samples must be finite and nonzero")
    quaternions /= norms
    flattened = quaternions.reshape(len(quaternions), -1, 4)
    if len(flattened) > 1:
        adjacent = np.sum(flattened[:-1] * flattened[1:], axis=-1)
        steps = np.where(adjacent < 0.0, -1.0, 1.0)
        signs = np.ones((len(flattened), flattened.shape[1]), np.float64)
        signs[1:] = np.cumprod(steps, axis=0)
        flattened *= signs[..., None]
    left, right, alpha = resample_map(len(flattened), source_fps, target_fps)
    lower = flattened[left]
    upper = flattened[right]
    dot = np.clip(np.sum(lower * upper, axis=-1), -1.0, 1.0)
    blend = alpha.astype(np.float64)[:, None]
    theta = np.arccos(dot)
    sine = np.sin(theta)
    safe_sine = np.where(np.abs(sine) < 1e-12, 1.0, sine)
    spherical = (
        np.sin((1.0 - blend) * theta)[..., None] * lower
        + np.sin(blend * theta)[..., None] * upper
    ) / safe_sine[..., None]
    linear = lower + blend[..., None] * (upper - lower)
    output = np.where((dot > 0.9995)[..., None], linear, spherical)
    output /= np.linalg.norm(output, axis=-1, keepdims=True)
    return output.reshape((len(output),) + quaternions.shape[1:])


def resample_holden_range(
    clip: HoldenClip,
    *,
    source_fps: float,
    target_fps: float = 60.0,
    skeleton: SkeletonSpec,
) -> tuple[HoldenClip, dict[str, Any]]:
    """Resample one continuity-safe range and independently rebuild dynamics."""

    if not isinstance(clip, HoldenClip):
        raise TypeError("clip must be a HoldenClip")
    clip.validate()
    if target_fps != 60.0:
        raise ValueError("full walking terrain LMM lanes require exact 60 Hz")
    left, right, alpha = resample_map(len(clip.positions), source_fps, target_fps)
    positions = _finite_float32(
        resample_vectors(clip.positions, source_fps, target_fps), "positions"
    )
    rotations = _finite_float32(
        _resample_quaternions_vectorized(
            clip.rotations, source_fps=source_fps, target_fps=target_fps
        ),
        "rotations",
    )
    terrain_features = _finite_float32(
        resample_vectors(clip.terrain_features, source_fps, target_fps),
        "terrain features",
    )
    terrain_support = _finite_float32(
        resample_vectors(clip.terrain_support, source_fps, target_fps),
        "terrain support",
    )
    nearest = np.where(alpha < 0.5, left, right)
    source_frames = np.asarray(clip.source_frames)[nearest].copy()
    result = HoldenClip(
        name=clip.name,
        positions=positions,
        velocities=np.zeros_like(positions),
        rotations=rotations,
        angular_velocities=np.zeros_like(positions),
        contacts=np.zeros((len(positions), 2), dtype=np.uint8),
        terrain_features=terrain_features,
        terrain_support=terrain_support,
        source_frames=source_frames,
        terrain_id=clip.terrain_id,
        source_left_indices=np.asarray(clip.source_left_indices, np.int32)[left],
        source_right_indices=np.asarray(clip.source_right_indices, np.int32)[right],
        source_alpha=np.asarray(alpha, np.float32),
    )
    result = refresh_lmm_clip_dynamics(result, skeleton, target_fps)
    provenance: dict[str, Any] = {
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "left": left,
        "right": right,
        "alpha": alpha,
    }
    return result, provenance


__all__ = [
    "BroadBankAuthority",
    "authenticate_broad_bank",
    "build_inherited_grail_lane",
    "build_missing_slope_lane",
    "build_takara_lane",
    "load_inherited_grail_work",
    "main",
    "missing_slope_source_ids",
    "publish_inherited_grail_work",
    "resample_holden_range",
    "resample_inherited_range",
    "resample_terrain_grid",
    "restore_inherited_work_terrain",
]


if __name__ == "__main__":
    raise SystemExit(main())
