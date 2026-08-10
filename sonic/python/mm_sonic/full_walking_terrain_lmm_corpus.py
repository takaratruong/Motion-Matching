"""Verify, pack, normalize, and reproduce the full 60 Hz walking corpus."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import canonical_json_bytes, sha256_file
from resources.g1_terrain_builder.database import (
    derive_lmm_contacts,
    derive_velocities,
)
from resources.g1_terrain_builder.features import _NORMALIZATION_WEIGHTS
from resources.g1_terrain_builder.resample import resample_map
from resources.g1_terrain_builder.scenes import (
    COORDINATE_SIGNATURE,
    SceneDefinition,
    SceneRoute,
    build_scene,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    ArtifactSet,
    FeatureSet,
)
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    surface_semantics_signature,
)

from .full_walking_terrain_lmm_contracts import (
    FAMILIES,
    SPLITS,
    FullWalkingInventory,
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    SplitAssignment,
    SplitLedger,
    build_split_ledger,
    inventory_manifest_bytes,
    load_lane,
    load_split_ledger,
    split_ledger_bytes,
)
from .full_walking_terrain_lmm_inventory import load_inventory

CORPUS_SCHEMA = "g1-full-walking-terrain-lmm-corpus/v1"
VERIFY_SCHEMA = "g1-full-walking-terrain-lmm-corpus-verification/v1"
DETERMINISM_SCHEMA = "g1-full-walking-terrain-lmm-determinism/v1"
FPS = 60.0
HORIZONS = (20, 40, 60)
SCENE_IDS = (
    "flat-standard",
    "grail-curb-default",
    "ramp-10-up-down",
    "stairs-standard",
)
DEFAULT_SCENE_AUTHORITY = Path(
    "/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate"
)
FAMILY_IDS = {name: index for index, name in enumerate(FAMILIES)}
SPLIT_IDS = {name: index for index, name in enumerate(SPLITS)}
QUALITY_IDS = {"clean": 0, "usable": 1, "quarantined": 2}
_FEATURE_GROUPS = (
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (15, 21),
    (21, 27),
    (27, 31),
)
_DISABLED_SCALE = np.finfo(np.float32).max
_SHA256_CHARS = frozenset("0123456789abcdef")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_BLOCK_ROWS = 262_144
_FROZEN_INVENTORY_SHA256 = (
    "7d1efd01abba6eea86280cb7720f999173fd8cedca89ddf56a61caad81a341e4"
)
_FROZEN_SPLIT_LEDGER_SHA256 = (
    "5185bd42c153518c80b67c3dd68f54e9122e2e81970893209e9235ff5cc2feb5"
)
_FROZEN_LANE_MANIFESTS = frozenset(
    {
        "e7cf6a83c65f642a69cf7e1f90b822a0e37cf4a5e8db0fe72298e1d2bfc19d78",
        "21e504ee7f2ec83921174a5c541ec7e0a19b43655c4cf9f7e6650edbcd0f22c8",
        "f8436be15845b1f42693d1161e453a8b628d53ebd5685480e066d3e9cbe4074b",
        "ceff0acb02ac1d189eac9da96cc3a21bbe88fe65259afb85bbc488dc79d706b7",
    }
)
_FROZEN_RANGE_KIND_COUNTS = {
    "derived-grail-parent-range": 15_814,
    "source-native-grail-slope": 23,
    "source-native-takara": 1,
    "pfnn": 1_161,
}
_FROZEN_PFNN_ROOT = Path("/home/ubuntu/datasets/pfnn/pfnn")
_PFNN_GAIT_CACHE: dict[str, np.ndarray] = {}
_PFNN_CONTINUITY_CACHE: dict[str, np.ndarray] = {}
_PFNN_EXCLUDED_GAIT_LABELS = ("jog", "run", "crouch", "jump", "crawl")


@dataclass(frozen=True)
class FullWalkingCorpus:
    root: Path
    artifacts: ArtifactSet
    features: FeatureSet
    raw_features: np.ndarray
    terrain_grid: np.ndarray
    family_ids: np.ndarray
    source_ids: np.ndarray
    source_names: tuple[str, ...]
    canonical_source_ids: np.ndarray
    canonical_source_names: tuple[str, ...]
    terrain_ids: np.ndarray
    terrain_names: tuple[str, ...]
    range_ids: np.ndarray
    range_names: tuple[str, ...]
    split_ids: np.ndarray
    quality_ids: np.ndarray
    eligible_mask: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    eligible_rows: np.ndarray
    source_row_offsets: np.ndarray
    source_rows: np.ndarray
    source_left_indices: np.ndarray
    source_right_indices: np.ndarray
    source_alpha: np.ndarray
    successor: np.ndarray
    successor_valid: np.ndarray
    root_delta_xy: np.ndarray
    root_delta_yaw: np.ndarray
    manifest_sha256: str
    inventory_manifest_sha256: str
    split_ledger_manifest_sha256: str
    lane_manifest_sha256: tuple[str, ...]
    fps: float = FPS
    horizons: tuple[int, int, int] = HORIZONS


@dataclass(frozen=True)
class _RangeInput:
    lane: LaneArtifact
    lane_range_index: int
    record: RangeRecord
    source_id: str

    @property
    def rows(self) -> int:
        return self.record.stop - self.record.start


@dataclass(frozen=True)
class _ScenePayload:
    scene_id: str
    scene_json: bytes
    terrain_bin: bytes
    terrain_obj: bytes
    walkability_bin: bytes


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not (set(value) - _SHA256_CHARS)


def _parse_canonical_bytes(payload: bytes, label: str) -> object:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _parse_canonical_file(path: Path, label: str) -> object:
    return _parse_canonical_bytes(path.read_bytes(), label)


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
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for corpus publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"immutable corpus output already exists: {destination}",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _publish_file_exclusive(payload: bytes, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    temporary = output.parent / f".{output.name}.building-{os.getpid()}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _member_descriptor(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class _BuildWorkspace:
    """Durable per-member staging that can resume after an interrupted build."""

    def __init__(self, output: Path, request_sha256: str):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.staging = self.output.parent / f".{self.output.name}.building"
        self.lock_path = self.output.parent / f".{self.output.name}.lock"
        self.lock = self.lock_path.open("a+b")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX)
        if self.output.exists() or self.output.is_symlink():
            self.close()
            raise FileExistsError(f"immutable corpus output exists: {self.output}")
        # Keep the journal outside the directory that is atomically renamed.
        # It therefore remains durable until publication without ever becoming
        # an unauthenticated member of the immutable output tree.
        self.state_path = self.output.parent / f".{self.output.name}.resume.json"
        if self.staging.exists():
            if self.staging.is_symlink() or not self.staging.is_dir():
                self.close()
                raise ValueError("corpus resume path is not a regular directory")
            state = _parse_canonical_file(self.state_path, "corpus resume state")
            if (
                type(state) is not dict
                or set(state) != {"schema", "request_sha256", "completed"}
                or state["schema"] != "g1-full-walking-corpus-resume/v1"
                or state["request_sha256"] != request_sha256
                or type(state["completed"]) is not dict
            ):
                self.close()
                raise ValueError("corpus resume state belongs to another build")
            self.completed: dict[str, dict[str, object]] = state["completed"]
        else:
            self.staging.mkdir()
            self.completed = {}
            self._write_state(request_sha256)
        self.request_sha256 = request_sha256

    def _write_state(self, request_sha256: str | None = None) -> None:
        payload = canonical_json_bytes(
            {
                "schema": "g1-full-walking-corpus-resume/v1",
                "request_sha256": request_sha256 or self.request_sha256,
                "completed": self.completed,
            }
        )
        temporary = self.staging / ".resume.json.partial"
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)
        _fsync_file(self.state_path)
        _fsync_directory(self.state_path.parent)
        _fsync_directory(self.staging)

    def _completed_valid(self, relative: str) -> bool:
        descriptor = self.completed.get(relative)
        path = self.staging / relative
        return bool(
            type(descriptor) is dict
            and descriptor.get("path") == relative
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == descriptor.get("size_bytes")
            and sha256_file(path) == descriptor.get("sha256")
        )

    def write_bytes(self, relative: str, payload: bytes) -> Path:
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        descriptor = self.completed.get(relative)
        if (
            self._completed_valid(relative)
            and descriptor is not None
            and descriptor.get("size_bytes") == len(payload)
            and descriptor.get("sha256") == expected_sha256
        ):
            return self.staging / relative
        path = self.staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.partial")
        partial.unlink(missing_ok=True)
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
        self.completed[relative] = _member_descriptor(path, relative)
        self._write_state()
        return path

    def write_npy(
        self,
        relative: str,
        *,
        dtype: np.dtype[Any] | str,
        shape: tuple[int, ...],
        fill: Callable[[np.memmap], None],
    ) -> Path:
        path = self.staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.partial")
        partial.unlink(missing_ok=True)
        values = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=shape,
        )
        fill(values)
        values.flush()
        del values
        _fsync_file(partial)
        if self._completed_valid(relative):
            descriptor = self.completed[relative]
            if (
                partial.stat().st_size == descriptor["size_bytes"]
                and sha256_file(partial) == descriptor["sha256"]
            ):
                partial.unlink()
                return path
        os.replace(partial, path)
        _fsync_directory(path.parent)
        self.completed[relative] = _member_descriptor(path, relative)
        self._write_state()
        return path

    def finish(
        self,
        manifest: Mapping[str, object],
        *,
        validator: Callable[[Path], None] | None = None,
    ) -> Path:
        manifest_path = self.staging / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(dict(manifest)))
        _fsync_file(manifest_path)
        try:
            if validator is not None:
                validator(self.staging)
        except BaseException:
            manifest_path.unlink(missing_ok=True)
            _fsync_directory(self.staging)
            raise
        # The external resume journal is intentionally retained until after
        # the no-replace rename. A crash before rename is resumable; a crash
        # after rename has already published the exact validated directory.
        self.completed.pop("resume.json", None)
        _fsync_directory(self.staging)
        for directory, subdirectories, _files in os.walk(self.staging, topdown=False):
            for name in subdirectories:
                _fsync_directory(Path(directory) / name)
            _fsync_directory(Path(directory))
        _rename_noreplace(self.staging, self.output)
        _fsync_directory(self.output.parent)
        self.state_path.unlink(missing_ok=True)
        _fsync_directory(self.output.parent)
        self.close()
        return self.output

    def close(self) -> None:
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()


def _resolve_inventory(
    value: Path | FullWalkingInventory,
) -> FullWalkingInventory:
    if isinstance(value, FullWalkingInventory):
        payload = inventory_manifest_bytes(value)
        if hashlib.sha256(payload).hexdigest() != value.manifest_sha256:
            raise ValueError("inventory object is not self-authenticating")
        return value
    return load_inventory(Path(value))


def _resolve_split_ledger(
    value: Path | SplitLedger,
    inventory: FullWalkingInventory,
) -> SplitLedger:
    if isinstance(value, SplitLedger):
        payload = split_ledger_bytes(value)
        if hashlib.sha256(payload).hexdigest() != value.manifest_sha256:
            raise ValueError("split ledger object is not self-authenticating")
        expected = build_split_ledger(
            inventory, build_id=inventory.build_id, seed=value.seed
        )
        if split_ledger_bytes(expected) != payload:
            raise ValueError("split ledger differs from connected inventory groups")
        return value
    return load_split_ledger(Path(value), inventory=inventory)


def _load_inventory_bytes(payload: bytes) -> FullWalkingInventory:
    value = _parse_canonical_bytes(payload, "corpus inventory")
    if (
        type(value) is not dict
        or set(value) != {"schema", "build_id", "sources"}
        or type(value["sources"]) is not list
    ):
        raise ValueError("corpus inventory schema or keys changed")
    records = tuple(SourceRecord(**record) for record in value["sources"])
    result = FullWalkingInventory(
        build_id=value["build_id"],
        sources=records,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if inventory_manifest_bytes(result) != payload:
        raise ValueError("corpus inventory content is not self-authenticating")
    return result


def _load_split_bytes(payload: bytes, inventory: FullWalkingInventory) -> SplitLedger:
    value = _parse_canonical_bytes(payload, "corpus split ledger")
    required = {
        "schema",
        "build_id",
        "inventory_manifest_sha256",
        "seed",
        "requested",
        "actual_family_counts",
        "assignments",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("corpus split ledger schema or keys changed")
    assignments = tuple(
        SplitAssignment(
            split_group_id=item["split_group_id"],
            split=item["split"],
            source_ids=tuple(item["source_ids"]),
        )
        for item in value["assignments"]
    )
    result = SplitLedger(
        build_id=value["build_id"],
        inventory_manifest_sha256=value["inventory_manifest_sha256"],
        seed=value["seed"],
        requested=value["requested"],
        actual_family_counts=value["actual_family_counts"],
        assignments=assignments,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if split_ledger_bytes(result) != payload:
        raise ValueError("corpus split ledger is not self-authenticating")
    expected = build_split_ledger(
        inventory, build_id=inventory.build_id, seed=result.seed
    )
    if split_ledger_bytes(expected) != payload:
        raise ValueError("corpus split ledger differs from connected inventory groups")
    return result


def _map_digest(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes(order="C")
    ).hexdigest()


def _input_sha256(source: SourceRecord, role: str) -> str | None:
    inputs = source.authority.get("inputs")
    descriptor = inputs.get(role) if isinstance(inputs, Mapping) else None
    digest = descriptor.get("sha256") if isinstance(descriptor, Mapping) else None
    return digest if _is_sha256(digest) else None


def _expected_source_map(
    source: SourceRecord, record: RangeRecord
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a range's map from its independent inventory source kind."""

    authority = record.authority
    source_kind = source.authority.get("kind")
    range_kind = authority.get("kind")
    frames = authority.get("source_frame_count")
    if type(frames) is not int or frames < 1:
        raise ValueError("lane source map requires an exact source frame count")

    if source_kind == "pfnn":
        interval = authority.get("source_interval_120hz")
        fit = authority.get("terrain_fit")
        motion_sha256 = _input_sha256(source, "motion")
        if (
            range_kind != "pfnn"
            or type(interval) is not list
            or len(interval) != 2
            or any(type(value) is not int for value in interval)
            or not (0 <= interval[0] < interval[1] <= frames)
            or not isinstance(fit, Mapping)
            or motion_sha256 is None
            or fit.get("source_sha256") != motion_sha256
            or fit.get("source_start") != interval[0]
            or fit.get("source_stop") != interval[1]
        ):
            raise ValueError("PFNN source map or fit differs from inventory authority")
        first = interval[0] + (interval[0] & 1)
        left = np.arange(first, interval[1], 2, dtype=np.int32)
        right = left.copy()
        alpha = np.zeros(len(left), dtype=np.float32)
        if authority.get("output_rows_60hz") != len(left) or authority.get(
            "absolute_source_rows_sha256"
        ) != _map_digest(left, "<i4"):
            raise ValueError("PFNN absolute 120-to-60 Hz source clock changed")
        identity_keys = (
            "schema",
            "family",
            "mode",
            "source_sha256",
            "patches_sha256",
            "source_start",
            "source_stop",
            "selected_patch_index",
            "beam_seed",
            "maximum_objective_probes",
            "maximum_rbf_centers",
        )
        if any(key not in fit for key in identity_keys):
            raise ValueError("PFNN terrain fit identity is incomplete")
        identity = {key: fit[key] for key in identity_keys}
        if hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != fit.get(
            "fit_id"
        ):
            raise ValueError("PFNN terrain fit identity digest changed")
        return left, right, alpha

    if source_kind == "grail":
        if range_kind == "derived-grail-parent-range":
            if (
                authority.get("parent_manifest_sha256")
                != source.authority.get("bank_manifest_sha256")
                or type(authority.get("parent_start")) is not int
                or type(authority.get("parent_stop")) is not int
                or authority["parent_stop"] - authority["parent_start"] != frames
            ):
                raise ValueError("inherited GRAIL parent authority changed")
        elif range_kind == "source-native-grail-slope":
            input_sha256 = authority.get("input_sha256")
            source_inputs = source.authority.get("inputs")
            expected_inputs = (
                {
                    role: descriptor.get("sha256")
                    for role, descriptor in source_inputs.items()
                    if isinstance(descriptor, Mapping)
                }
                if isinstance(source_inputs, Mapping)
                else None
            )
            if input_sha256 != expected_inputs:
                raise ValueError("source-native GRAIL input authority changed")
        else:
            raise ValueError("GRAIL range kind changed")
        return resample_map(frames, 25.0, FPS)

    if source_kind == "takara":
        if range_kind != "source-native-takara" or authority.get(
            "motion_sha256"
        ) != _input_sha256(source, "motion"):
            raise ValueError("Takara source authority changed")
        return resample_map(frames, 50.0, FPS)

    if source_kind == "fixture":
        source_fps = source.authority.get("source_fps")
        source_map = authority.get("source_map")
        if (
            range_kind != "fixture"
            or type(source_fps) not in (int, float)
            or not isinstance(source_map, Mapping)
            or source_map.get("source_fps") != source_fps
            or source_map.get("target_fps") not in (None, FPS)
        ):
            raise ValueError("fixture source map authority changed")
        return resample_map(frames, float(source_fps), FPS)

    raise ValueError(f"unsupported independent source-map authority {source_kind!r}")


def _verify_terrain_semantics(
    source: SourceRecord,
    record: RangeRecord,
    terrain_features: np.ndarray,
    terrain_grid: np.ndarray,
) -> None:
    """Verify the source-specific 4-D/36-D terrain representation contract."""

    if source.authority.get("kind") == "pfnn":
        terrain_family = record.authority.get("terrain_family")
        terrain_fit = record.authority.get("terrain_fit")
        if (
            record.authority.get("kind") != "pfnn"
            or terrain_family not in ("flat", "rocky", "jumpy", "beam")
            or not isinstance(terrain_fit, Mapping)
            or terrain_fit.get("schema") != "full-pfnn-family-fit/v2"
            or terrain_fit.get("family") != terrain_family
        ):
            raise ValueError("PFNN terrain grid lacks its exact fit authority")
    elif not np.array_equal(terrain_features, terrain_grid[:, (1, 3, 5, 7)]):
        raise ValueError(
            "G1TF compatibility terrain grid columns 1/3/5/7 must exactly equal "
            "the four matching terrain features"
        )


def _verify_lane_contents(
    lane: LaneArtifact,
    inventory: FullWalkingInventory,
    split_ledger: SplitLedger,
) -> None:
    """Independently verify row provenance and all boundary-sensitive channels."""

    if lane.inventory_manifest_sha256 != inventory.manifest_sha256:
        raise ValueError("lane inventory authority changed")
    if lane.split_ledger_manifest_sha256 != split_ledger.manifest_sha256:
        raise ValueError("lane split authority changed")
    source_by_id = {record.source_id: record for record in inventory.sources}
    assignment_by_source = {
        source_id: assignment
        for assignment in split_ledger.assignments
        for source_id in assignment.source_ids
    }
    if len(lane.source_ids) != len(lane.ranges):
        raise ValueError("lane source IDs must be exactly one per range")
    if np.any(lane.artifacts.contacts > 1):
        raise ValueError("lane contacts must remain binary")
    parents = np.asarray(lane.artifacts.parents, dtype=np.int32)
    if tuple(parents.tolist()) != G1_SKELETON_PARENTS:
        raise ValueError("lane skeleton hierarchy changed")
    for index, (record, source_id) in enumerate(zip(lane.ranges, lane.source_ids)):
        source = source_by_id.get(source_id)
        assignment = assignment_by_source.get(source_id)
        if source is None or assignment is None:
            raise ValueError("lane source is absent from inventory/split authority")
        if (
            source.canonical_source_id != record.canonical_source_id
            or source.terrain_id != record.terrain_id
            or source.mirror_of != record.mirror_of
            or source.family != record.family
            or assignment.split_group_id != record.split_group_id
            or assignment.split != record.split
        ):
            raise ValueError("lane range mirror/source/split metadata changed")
        start, stop = record.start, record.stop
        if (
            start != int(lane.artifacts.range_starts[index])
            or stop != int(lane.artifacts.range_stops[index])
            or stop - start < 4
        ):
            raise ValueError("lane range is not a valid range-local 60 Hz interval")
        left = lane.source_left_indices[start:stop]
        right = lane.source_right_indices[start:stop]
        alpha = lane.source_alpha[start:stop]
        expected_left, expected_right, expected_alpha = _expected_source_map(
            source, record
        )
        source_frame_count = record.authority.get("source_frame_count")
        if (
            type(source_frame_count) is not int
            or source_frame_count < 1
            or np.any(left < 0)
            or np.any(left > right)
            or np.any(right >= source_frame_count)
            or np.any(np.diff(left.astype(np.int64)) < 0)
            or np.any(np.diff(right.astype(np.int64)) < 0)
            or np.any(alpha < 0.0)
            or np.any(alpha > 1.0)
        ):
            raise ValueError(
                "lane source map requires an exact source frame count and bounded, "
                "monotone source-local provenance"
            )
        if (
            not np.array_equal(left, expected_left)
            or not np.array_equal(right, expected_right)
            or not np.array_equal(alpha, expected_alpha)
        ):
            raise ValueError(
                "lane source map differs from the independently reconstructed clock"
            )
        source_map = record.authority.get("source_map")
        if isinstance(source_map, Mapping):
            expected_map = {
                "left_sha256": _map_digest(left, "<i4"),
                "right_sha256": _map_digest(right, "<i4"),
                "alpha_sha256": _map_digest(alpha, "<f4"),
            }
            for key, digest in expected_map.items():
                if source_map.get(key) != digest:
                    raise ValueError(
                        f"lane source map {key} provenance SHA-256 changed"
                    )

        positions = lane.artifacts.positions[start:stop]
        rotations = lane.artifacts.rotations[start:stop]
        norms = np.linalg.norm(rotations.astype(np.float64), axis=-1)
        if np.max(np.abs(norms - 1.0)) > 1e-4:
            raise ValueError("lane quaternion normalization changed")
        if np.any(np.sum(rotations[1:] * rotations[:-1], axis=-1) < -1e-6):
            raise ValueError("lane quaternion signs are discontinuous inside a range")
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_velocity, expected_angular = derive_velocities(
                positions, rotations, FPS
            )
        if not np.allclose(
            lane.artifacts.velocities[start:stop],
            expected_velocity,
            rtol=2e-5,
            atol=2e-5,
        ):
            raise ValueError("lane range-local velocity recomputation failed")
        if not np.allclose(
            lane.artifacts.angular_velocities[start:stop],
            expected_angular,
            rtol=2e-5,
            atol=2e-5,
        ):
            raise ValueError("lane range-local angular velocity recomputation failed")
        expected_contacts = derive_lmm_contacts(
            positions,
            rotations,
            parents,
            G1_SKELETON_NAMES.index("LeftToe"),
            G1_SKELETON_NAMES.index("RightToe"),
            FPS,
        )
        if not np.array_equal(lane.artifacts.contacts[start:stop], expected_contacts):
            raise ValueError("lane range-local contact recomputation failed")
        for label, values, columns in (
            ("terrain features", lane.artifacts.terrain_features[start:stop], 4),
            ("terrain support", lane.artifacts.terrain_support[start:stop], 3),
            ("terrain grid", lane.terrain_grid[start:stop], 36),
        ):
            if values.shape != (stop - start, columns) or not np.isfinite(values).all():
                raise ValueError(f"lane {label} is not finite and row aligned")
        _verify_terrain_semantics(
            source,
            record,
            lane.artifacts.terrain_features[start:stop],
            lane.terrain_grid[start:stop],
        )


def _global_kinematics(
    positions: np.ndarray,
    rotations: np.ndarray,
    velocities: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, bones = positions.shape[:2]
    gp = np.empty((frames, bones, 3), np.float64)
    gq = np.empty((frames, bones, 4), np.float64)
    gv = np.empty((frames, bones, 3), np.float64)
    ga = np.empty((frames, bones, 3), np.float64)
    lp = positions.astype(np.float64)
    lq = rotations.astype(np.float64)
    lv = velocities.astype(np.float64)
    la = np.zeros_like(lv)
    # Angular velocities affect child linear velocity. They are derived again
    # from rotations so the raw-feature path does not trust a packed channel.
    with np.errstate(divide="ignore", invalid="ignore"):
        _unused_velocity, la32 = derive_velocities(positions, rotations, FPS)
    la[:] = la32
    for bone, parent in enumerate(parents):
        if parent < 0:
            gp[:, bone], gq[:, bone] = lp[:, bone], lq[:, bone]
            gv[:, bone], ga[:, bone] = lv[:, bone], la[:, bone]
            continue
        rotated = holden_quat.mul_vec(gq[:, parent], lp[:, bone])
        gp[:, bone] = gp[:, parent] + rotated
        gq[:, bone] = holden_quat.mul(gq[:, parent], lq[:, bone])
        gv[:, bone] = (
            gv[:, parent]
            + holden_quat.mul_vec(gq[:, parent], lv[:, bone])
            + np.cross(ga[:, parent], rotated)
        )
        ga[:, bone] = ga[:, parent] + holden_quat.mul_vec(gq[:, parent], la[:, bone])
    return gp, gq, gv


def _raw_features_for_range(
    positions: np.ndarray,
    velocities: np.ndarray,
    rotations: np.ndarray,
    parents: np.ndarray,
    terrain_features: np.ndarray,
) -> np.ndarray:
    gp, gq, gv = _global_kinematics(positions, rotations, velocities, parents)
    root_inverse = holden_quat.inv(gq[:, 0])
    raw = np.empty((len(gp), 31), np.float32)
    for output, bone in ((0, 6), (3, 12)):
        raw[:, output : output + 3] = holden_quat.mul_vec(
            root_inverse, gp[:, bone] - gp[:, 0]
        )
    for output, bone in ((6, 6), (9, 12), (12, 1)):
        raw[:, output : output + 3] = holden_quat.mul_vec(root_inverse, gv[:, bone])
    rows = np.arange(len(gp), dtype=np.int64)
    for slot, horizon in enumerate(HORIZONS):
        future = np.minimum(rows + horizon, len(gp) - 1)
        position = holden_quat.mul_vec(root_inverse, gp[future, 0] - gp[:, 0])
        facing = holden_quat.mul_vec(
            root_inverse,
            holden_quat.mul_vec(gq[future, 0], np.asarray([0.0, 0.0, 1.0], np.float64)),
        )
        raw[:, 15 + slot * 2 : 17 + slot * 2] = position[:, [0, 2]]
        raw[:, 21 + slot * 2 : 23 + slot * 2] = facing[:, [0, 2]]
    raw[:, 27:31] = terrain_features
    if not np.isfinite(raw).all():
        raise ValueError("raw matching features are nonfinite")
    return raw


def _fit_normalization_float64(
    raw: np.ndarray, fit_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fit_rows = np.flatnonzero(fit_mask)
    if not len(fit_rows):
        raise ValueError("normalization requires clean + usable training rows")
    offset = np.empty(31, np.float32)
    scale = np.empty(31, np.float32)
    for (start, stop), weight in zip(_FEATURE_GROUPS, _NORMALIZATION_WEIGHTS):
        count = 0
        mean = np.zeros(stop - start, np.float64)
        m2 = np.zeros(stop - start, np.float64)
        for block_start in range(0, len(fit_rows), _BLOCK_ROWS):
            indices = fit_rows[block_start : block_start + _BLOCK_ROWS]
            values = np.asarray(raw[indices, start:stop], dtype=np.float64)
            block_count = len(values)
            block_mean = values.mean(axis=0)
            block_m2 = np.square(values - block_mean).sum(axis=0)
            if count == 0:
                mean, m2, count = block_mean, block_m2, block_count
                continue
            delta = block_mean - mean
            combined = count + block_count
            m2 += block_m2 + np.square(delta) * count * block_count / combined
            mean += delta * block_count / combined
            count = combined
        group_std = float(np.mean(np.sqrt(m2 / count)))
        offset[start:stop] = mean.astype(np.float32)
        if not np.isfinite(group_std) or group_std <= 0.0 or weight == 0.0:
            scale[start:stop] = _DISABLED_SCALE
        else:
            scale[start:stop] = np.float32(group_std / weight)
    return offset, scale


def _fill_normalized(
    output: np.ndarray, raw: np.ndarray, offset: np.ndarray, scale: np.ndarray
) -> None:
    for block_start in range(0, len(raw), _BLOCK_ROWS):
        block_stop = min(block_start + _BLOCK_ROWS, len(raw))
        values = np.asarray(raw[block_start:block_stop], dtype=np.float64)
        transformed = np.empty(values.shape, np.float32)
        for start, stop in _FEATURE_GROUPS:
            if np.all(scale[start:stop] == _DISABLED_SCALE):
                transformed[:, start:stop] = 0.0
            else:
                transformed[:, start:stop] = (
                    (values[:, start:stop] - offset[start:stop]) / scale[start:stop]
                ).astype(np.float32)
        output[block_start:block_stop] = transformed


def _root_deltas_for_range(
    positions: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(positions)
    delta_xy = np.zeros((rows, 2), np.float32)
    delta_yaw = np.zeros(rows, np.float32)
    if rows == 1:
        return delta_xy, delta_yaw
    facing = holden_quat.mul_vec(
        rotations[:, 0].astype(np.float64),
        np.asarray([0.0, 0.0, 1.0], np.float64),
    )
    yaw = np.unwrap(np.arctan2(facing[:, 0], facing[:, 2]))
    displacement = positions[1:, 0] - positions[:-1, 0]
    cosine, sine = np.cos(yaw[:-1]), np.sin(yaw[:-1])
    delta_xy[:-1, 0] = (cosine * displacement[:, 0] - sine * displacement[:, 2]).astype(
        np.float32
    )
    delta_xy[:-1, 1] = (sine * displacement[:, 0] + cosine * displacement[:, 2]).astype(
        np.float32
    )
    delta_yaw[:-1] = ((np.diff(yaw) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    return delta_xy, delta_yaw


def _flat_scene_definition() -> SceneDefinition:
    return SceneDefinition(
        scene_id="flat-standard",
        label="Flat Standard",
        provenance={
            "kind": "procedural",
            "source_ids": [],
            "parameters": {
                "schema": "g1-full-walking-flat-standard/v1",
                "height_m": 0.0,
            },
        },
        surface=FlatTerrain(),
        heightfield_bounds_xz=(-3.0, 3.0, -2.0, 12.0),
        playable_bounds_xz=(-2.0, 2.0, 0.0, 10.0),
        lookahead_bounds_xz=(-3.0, 3.0, -2.0, 12.0),
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions={
            "certified": ({"id": "flat-route", "bounds_xz": [-2.0, 2.0, 0.0, 10.0]},),
            "stress": (),
            "blocked": (),
        },
        routes=(
            SceneRoute(
                route_id="flat-forward",
                waypoints_xz=(
                    (0.0, 0.0),
                    (0.0, 2.0),
                    (0.0, 5.0),
                    (0.0, 9.0),
                ),
                expected_outcome="traverse",
                walkability_class=1,
                landing_hold_seconds=0.0,
            ),
        ),
        walkability=lambda x, z: int(-2.25 <= x <= 2.25 and -0.25 <= z <= 10.25),
    )


def _authenticate_scene_payload(
    scene_id: str,
    scene_root: Path,
    *,
    expected_scene_sha256: str | None,
) -> _ScenePayload:
    if scene_root.is_symlink() or not scene_root.is_dir():
        raise ValueError(f"{scene_id}: scene authority is not a directory")
    expected_names = {
        "scene.json",
        "terrain.bin",
        "terrain.obj",
        "walkability.bin",
    }
    if {path.name for path in scene_root.iterdir()} != expected_names:
        raise ValueError(f"{scene_id}: scene member set changed")
    payloads = {
        name: (scene_root / name).read_bytes() for name in sorted(expected_names)
    }
    if (
        expected_scene_sha256 is not None
        and hashlib.sha256(payloads["scene.json"]).hexdigest() != expected_scene_sha256
    ):
        raise ValueError(f"{scene_id}: scene index SHA-256 changed")
    metadata = _parse_canonical_bytes(
        payloads["scene.json"], f"{scene_id} scene metadata"
    )
    if (
        type(metadata) is not dict
        or metadata.get("schema") != "g1-terrain-scene/v1"
        or metadata.get("id") != scene_id
        or metadata.get("coordinate_signature") != COORDINATE_SIGNATURE
        or metadata.get("surface_signature") != surface_semantics_signature()
    ):
        raise ValueError(f"{scene_id}: scene identity or coordinate schema changed")
    for descriptor_name, filename in (
        ("heightfield", "terrain.bin"),
        ("mesh", "terrain.obj"),
        ("walkability", "walkability.bin"),
    ):
        descriptor = metadata.get(descriptor_name)
        if (
            type(descriptor) is not dict
            or descriptor.get("path") != filename
            or descriptor.get("sha256")
            != hashlib.sha256(payloads[filename]).hexdigest()
        ):
            raise ValueError(f"{scene_id}: {descriptor_name} authority changed")
    return _ScenePayload(
        scene_id=scene_id,
        scene_json=payloads["scene.json"],
        terrain_bin=payloads["terrain.bin"],
        terrain_obj=payloads["terrain.obj"],
        walkability_bin=payloads["walkability.bin"],
    )


def _scene_pack_snapshot(
    scene_authority: Path | None,
) -> tuple[bytes, tuple[_ScenePayload, ...]]:
    authority = Path(scene_authority or DEFAULT_SCENE_AUTHORITY).resolve(strict=True)
    scenes_root = authority / "scenes" if (authority / "scenes").is_dir() else authority
    index_path = scenes_root / "index.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("scene authority index is missing")
    index_payload = index_path.read_bytes()
    index = _parse_canonical_bytes(index_payload, "scene authority index")
    if (
        type(index) is not dict
        or index.get("schema") != "g1-terrain-scene-index/v1"
        or index.get("coordinate_signature") != COORDINATE_SIGNATURE
        or index.get("surface_signature") != surface_semantics_signature()
        or type(index.get("scenes")) is not list
    ):
        raise ValueError("scene authority index identity changed")
    descriptor_by_id: dict[str, dict[str, object]] = {}
    for descriptor in index["scenes"]:
        if (
            type(descriptor) is not dict
            or set(descriptor) != {"id", "path", "sha256"}
            or type(descriptor.get("id")) is not str
        ):
            raise ValueError("scene authority descriptor changed")
        descriptor_by_id[descriptor["id"]] = descriptor

    payloads: list[_ScenePayload] = []
    for scene_id in SCENE_IDS:
        descriptor = descriptor_by_id.get(scene_id)
        scene_root = scenes_root / scene_id
        if descriptor is None and scene_id == "flat-standard":
            built = build_scene(_flat_scene_definition())
            payloads.append(
                _ScenePayload(
                    scene_id,
                    built.scene_json,
                    built.terrain_bin,
                    built.terrain_obj,
                    built.walkability_bin,
                )
            )
            continue
        if descriptor is None:
            raise ValueError(f"scene authority lacks required {scene_id}")
        expected_path = f"scenes/{scene_id}/scene.json"
        if descriptor["path"] != expected_path or not _is_sha256(descriptor["sha256"]):
            raise ValueError(f"{scene_id}: scene index descriptor changed")
        payloads.append(
            _authenticate_scene_payload(
                scene_id,
                scene_root,
                expected_scene_sha256=descriptor["sha256"],
            )
        )
    packed_index = {
        "schema": "g1-terrain-scene-index/v1",
        "default_scene_id": "flat-standard",
        "scene_ids": list(SCENE_IDS),
        "scenes": [
            {
                "id": scene.scene_id,
                "path": f"scenes/{scene.scene_id}/scene.json",
                "sha256": hashlib.sha256(scene.scene_json).hexdigest(),
            }
            for scene in payloads
        ],
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
    }
    return canonical_json_bytes(packed_index), tuple(payloads)


def _resolve_lanes(
    lanes: Sequence[Path],
    inventory: FullWalkingInventory,
    split_ledger: SplitLedger,
) -> tuple[LaneArtifact, ...]:
    if not lanes:
        raise ValueError("full corpus requires at least one lane")
    loaded = []
    seen_manifest: set[str] = set()
    for lane_path in lanes:
        lane = load_lane(
            Path(lane_path),
            inventory=inventory,
            split_ledger=split_ledger,
            expected_inventory_manifest_sha256=inventory.manifest_sha256,
            expected_split_ledger_manifest_sha256=split_ledger.manifest_sha256,
        )
        if lane.manifest_sha256 in seen_manifest:
            raise ValueError("one immutable lane was supplied more than once")
        seen_manifest.add(lane.manifest_sha256)
        _verify_lane_contents(lane, inventory, split_ledger)
        loaded.append(lane)
    return tuple(sorted(loaded, key=lambda value: value.manifest_sha256))


def _ordered_ranges(
    lanes: Sequence[LaneArtifact], inventory: FullWalkingInventory
) -> tuple[_RangeInput, ...]:
    inputs = [
        _RangeInput(lane, index, record, lane.source_ids[index])
        for lane in lanes
        for index, record in enumerate(lane.ranges)
    ]
    range_ids = [value.record.range_id for value in inputs]
    if len(range_ids) != len(set(range_ids)):
        raise ValueError("full corpus range IDs must be globally unique")
    inventory_ids = {source.source_id for source in inventory.sources}
    terminal_ids = {value.source_id for value in inputs}
    if terminal_ids != inventory_ids:
        missing = sorted(inventory_ids - terminal_ids)
        unexpected = sorted(terminal_ids - inventory_ids)
        raise ValueError(
            "full corpus source terminal coverage changed: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    ranges_by_source: dict[str, set[str]] = {
        source_id: set() for source_id in inventory_ids
    }
    for value in inputs:
        ranges_by_source[value.source_id].add(value.record.range_id)
    for source in inventory.sources:
        expected = source.authority.get("expected_range_ids")
        if expected is None:
            continue
        if (
            type(expected) is not list
            or not expected
            or not all(type(range_id) is str for range_id in expected)
            or ranges_by_source[source.source_id] != set(expected)
        ):
            raise ValueError(
                f"source terminal range coverage changed for {source.source_id}"
            )
    if inventory.manifest_sha256 == _FROZEN_INVENTORY_SHA256:
        lane_manifests = {lane.manifest_sha256 for lane in lanes}
        kind_counts: dict[str, int] = {}
        for value in inputs:
            kind = value.record.authority.get("kind")
            if type(kind) is not str:
                raise ValueError("frozen range terminal kind is missing")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if (
            lane_manifests != _FROZEN_LANE_MANIFESTS
            or kind_counts != _FROZEN_RANGE_KIND_COUNTS
            or len(inputs) != 16_999
            or sum(value.rows for value in inputs) != 9_758_524
        ):
            raise ValueError(
                "frozen full-corpus lane or terminal range coverage changed"
            )
    return tuple(
        sorted(
            inputs,
            key=lambda value: (
                FAMILY_IDS[value.record.family],
                value.record.canonical_source_id,
                value.record.mirror_of or "",
                value.record.range_id,
            ),
        )
    )


def _classify_pfnn_terminal_outcomes(
    gait: np.ndarray,
    accepted_intervals: Sequence[tuple[int, int]],
    continuity_breaks: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    """Classify every omitted native PFNN row and every retained boundary."""

    values = np.asarray(gait)
    if values.ndim != 2 or values.shape[1] != 8 or values.dtype.kind not in "iuf":
        raise ValueError("PFNN gait must have exact numeric shape [T,8]")
    values = values.astype(np.float64, copy=False)
    finite = np.isfinite(values).all(axis=1)
    bounded = ((values >= 0.0) & (values <= 1.0)).all(axis=1)
    base_sum = values[:, :7].sum(axis=1)
    valid = finite & bounded & np.isclose(base_sum, 1.0, rtol=0.0, atol=1e-6)
    safe = np.where(valid[:, None], values, 0.0)
    allowed = safe[:, 0] + safe[:, 1]
    disallowed = safe[:, 2:7].sum(axis=1)
    admitted = valid & (allowed > 0.0) & (allowed >= disallowed)

    continuity = np.asarray(continuity_breaks)
    if continuity.dtype != np.bool_ or continuity.shape != (max(0, len(values) - 1),):
        raise ValueError("PFNN continuity breaks must be bool shape [T-1]")
    accepted_set: set[tuple[int, int]] = set()
    previous_stop = 0
    for start, stop in accepted_intervals:
        if (
            type(start) is not int
            or type(stop) is not int
            or not (0 <= start < stop <= len(values))
            or start < previous_stop
        ):
            raise ValueError("PFNN accepted terminal intervals overlap or are invalid")
        accepted_set.add((start, stop))
        previous_stop = stop

    admitted_intervals: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(admitted):
        if not admitted[cursor]:
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while (
            cursor < len(admitted) and admitted[cursor] and not continuity[cursor - 1]
        ):
            cursor += 1
        admitted_intervals.append((start, cursor))
    admitted_set = set(admitted_intervals)
    if not accepted_set.issubset(admitted_set):
        raise ValueError("PFNN retained range differs from admitted range boundaries")

    reasons: list[str | None] = [None] * len(values)
    for row in np.flatnonzero(~valid):
        reasons[int(row)] = "malformed_gait"
    for row in np.flatnonzero(valid & ~admitted):
        reasons[int(row)] = _PFNN_EXCLUDED_GAIT_LABELS[int(np.argmax(values[row, 2:7]))]
    blocks: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(reasons):
        reason = reasons[cursor]
        if reason is None:
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(reasons) and reasons[cursor] == reason:
            cursor += 1
        blocks.append({"start": start, "stop": cursor, "reason": reason})
    blocks.extend(
        {"start": start, "stop": stop, "reason": "short_fragment"}
        for start, stop in admitted_intervals
        if (start, stop) not in accepted_set
    )
    blocks.sort(key=lambda value: (int(value["start"]), int(value["stop"])))
    boundaries = [
        {"at": int(index + 1), "reason": "retarget_continuity"}
        for index in np.flatnonzero(continuity)
    ]
    return {"excluded_blocks": blocks, "range_boundaries": boundaries}


def _load_frozen_pfnn_gait(source: SourceRecord, frames: int) -> np.ndarray:
    inputs = source.authority.get("inputs")
    descriptor = inputs.get("gait") if isinstance(inputs, Mapping) else None
    if (
        not isinstance(descriptor, Mapping)
        or type(descriptor.get("path")) is not str
        or type(descriptor.get("size_bytes")) is not int
        or not _is_sha256(descriptor.get("sha256"))
    ):
        raise ValueError("PFNN inventory lacks an authenticated gait descriptor")
    root = _FROZEN_PFNN_ROOT.resolve(strict=True)
    path = (root / descriptor["path"]).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("PFNN gait authority escapes its frozen root") from error
    key = f"{path}:{descriptor['sha256']}"
    if key not in _PFNN_GAIT_CACHE:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError("PFNN gait source bytes changed")
        try:
            gait = np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ValueError("PFNN gait source failed strict decode") from error
        gait = np.ascontiguousarray(gait, dtype=np.float64)
        gait.setflags(write=False)
        _PFNN_GAIT_CACHE[key] = gait
    result = _PFNN_GAIT_CACHE[key]
    if result.shape != (frames, 8):
        raise ValueError("PFNN gait source frame count changed")
    return result


def _load_pfnn_continuity(records: Sequence[RangeRecord], frames: int) -> np.ndarray:
    descriptors = [record.authority.get("retarget") for record in records]
    if not descriptors or any(value != descriptors[0] for value in descriptors):
        raise ValueError("PFNN ranges do not share one retarget authority")
    descriptor = descriptors[0]
    if (
        not isinstance(descriptor, Mapping)
        or type(descriptor.get("path")) is not str
        or type(descriptor.get("size_bytes")) is not int
        or not _is_sha256(descriptor.get("sha256"))
    ):
        raise ValueError("PFNN retarget authority is incomplete")
    path = Path(descriptor["path"]).resolve(strict=True)
    key = f"{path}:{descriptor['sha256']}"
    if key not in _PFNN_CONTINUITY_CACHE:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError("PFNN retarget source bytes changed")
        try:
            with np.load(path, allow_pickle=False) as archive:
                root = np.asarray(archive["root_pos"], dtype=np.float64)
                quaternion = np.asarray(archive["root_quat"], dtype=np.float64)
                dof = np.asarray(archive["dof"], dtype=np.float64)
                fps = float(np.asarray(archive["fps"]).item())
        except (OSError, KeyError, ValueError) as error:
            raise ValueError("PFNN retarget source failed strict decode") from error
        if (
            root.shape != (frames, 3)
            or quaternion.shape != (frames, 4)
            or dof.shape != (frames, 29)
            or fps != 120.0
            or not all(np.isfinite(value).all() for value in (root, quaternion, dof))
        ):
            raise ValueError("PFNN retarget source dimensions or values changed")
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("PFNN retarget source contains a zero quaternion")
        normalized = quaternion / norms
        planar = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) > 0.10
        joint = np.max(np.abs(np.diff(dof, axis=0)), axis=1) > 0.50
        dots = np.clip(
            np.abs(np.sum(normalized[1:] * normalized[:-1], axis=1)), 0.0, 1.0
        )
        rotation = 2.0 * np.arccos(dots) > 0.30
        continuity = np.ascontiguousarray(planar | joint | rotation, dtype=np.bool_)
        continuity.setflags(write=False)
        _PFNN_CONTINUITY_CACHE[key] = continuity
    result = _PFNN_CONTINUITY_CACHE[key]
    if result.shape != (max(0, frames - 1),):
        raise ValueError("PFNN retarget continuity frame count changed")
    return result


def _terminal_outcomes(
    records: Sequence[tuple[RangeRecord, str]],
    inventory: FullWalkingInventory,
) -> dict[str, object]:
    """Build an exact per-source quality and excluded-block ledger."""

    by_source: dict[str, list[RangeRecord]] = {
        source.source_id: [] for source in inventory.sources
    }
    quality_range_counts = {quality: 0 for quality in QUALITY_IDS}
    quality_row_counts = {quality: 0 for quality in QUALITY_IDS}
    for record, source_id in records:
        by_source[source_id].append(record)
        quality_range_counts[record.quality] += 1
        quality_row_counts[record.quality] += record.stop - record.start

    sources = []
    excluded_reason_counts: dict[str, int] = {}
    excluded_family_reason_counts: dict[str, dict[str, dict[str, int]]] = {
        family: {} for family in FAMILIES
    }
    boundary_reason_counts: dict[str, int] = {}
    for source in sorted(inventory.sources, key=lambda value: value.source_id):
        source_records = by_source[source.source_id]
        excluded_blocks: list[dict[str, object]] = []
        range_boundaries: list[dict[str, object]] = []
        if source.authority.get("kind") == "pfnn":
            intervals = sorted(
                tuple(record.authority.get("source_interval_120hz", ()))
                for record in source_records
            )
            frame_counts = {
                record.authority.get("source_frame_count") for record in source_records
            }
            if len(frame_counts) != 1 or any(
                len(interval) != 2 or any(type(value) is not int for value in interval)
                for interval in intervals
            ):
                raise ValueError("PFNN terminal outcome source bounds changed")
            frame_count = next(iter(frame_counts))
            gait = _load_frozen_pfnn_gait(source, frame_count)
            continuity = _load_pfnn_continuity(source_records, frame_count)
            classified = _classify_pfnn_terminal_outcomes(
                gait, tuple(intervals), continuity
            )
            excluded_blocks = classified["excluded_blocks"]
            range_boundaries = classified["range_boundaries"]
        for block in excluded_blocks:
            reason = str(block["reason"])
            excluded_reason_counts[reason] = excluded_reason_counts.get(reason, 0) + 1
            family_reasons = excluded_family_reason_counts[source.family]
            counts = family_reasons.setdefault(reason, {"blocks": 0, "rows": 0})
            counts["blocks"] += 1
            counts["rows"] += int(block["stop"]) - int(block["start"])
        for boundary in range_boundaries:
            reason = str(boundary["reason"])
            boundary_reason_counts[reason] = boundary_reason_counts.get(reason, 0) + 1
        sources.append(
            {
                "source_id": source.source_id,
                "family": source.family,
                "ranges": len(source_records),
                "rows": sum(record.stop - record.start for record in source_records),
                "quality_range_counts": {
                    quality: sum(record.quality == quality for record in source_records)
                    for quality in QUALITY_IDS
                },
                "quality_row_counts": {
                    quality: sum(
                        record.stop - record.start
                        for record in source_records
                        if record.quality == quality
                    )
                    for quality in QUALITY_IDS
                },
                "excluded_blocks": excluded_blocks,
                "range_boundaries": range_boundaries,
            }
        )
    return {
        "schema": "g1-full-walking-terminal-outcomes/v1",
        "quality_range_counts": quality_range_counts,
        "quality_row_counts": quality_row_counts,
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "excluded_family_reason_counts": {
            family: dict(sorted(reasons.items()))
            for family, reasons in excluded_family_reason_counts.items()
        },
        "boundary_reason_counts": dict(sorted(boundary_reason_counts.items())),
        "sources": sources,
    }


def _request_identity(
    inventory: FullWalkingInventory,
    split_ledger: SplitLedger,
    lanes: Sequence[LaneArtifact],
    scene_index: bytes,
    scenes: Sequence[_ScenePayload],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "g1-full-walking-corpus-build-request/v1",
                "inventory_manifest_sha256": inventory.manifest_sha256,
                "split_ledger_manifest_sha256": split_ledger.manifest_sha256,
                "lane_manifest_sha256": sorted(lane.manifest_sha256 for lane in lanes),
                "scene_index_sha256": hashlib.sha256(scene_index).hexdigest(),
                "scene_payload_sha256": {
                    scene.scene_id: {
                        "scene.json": hashlib.sha256(scene.scene_json).hexdigest(),
                        "terrain.bin": hashlib.sha256(scene.terrain_bin).hexdigest(),
                        "terrain.obj": hashlib.sha256(scene.terrain_obj).hexdigest(),
                        "walkability.bin": hashlib.sha256(
                            scene.walkability_bin
                        ).hexdigest(),
                    }
                    for scene in scenes
                },
            }
        )
    ).hexdigest()


def _copy_lane_field(
    output: np.ndarray,
    ranges: Sequence[_RangeInput],
    attribute: str,
) -> None:
    cursor = 0
    for value in ranges:
        source = getattr(value.lane.artifacts, attribute)
        rows = value.rows
        output[cursor : cursor + rows] = source[value.record.start : value.record.stop]
        cursor += rows


def _copy_lane_array(
    output: np.ndarray,
    ranges: Sequence[_RangeInput],
    attribute: str,
) -> None:
    cursor = 0
    for value in ranges:
        source = getattr(value.lane, attribute)
        rows = value.rows
        output[cursor : cursor + rows] = source[value.record.start : value.record.stop]
        cursor += rows


def _load_mmap(path: Path) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"corpus array is not a regular file: {path.name}")
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid corpus array {path.name}") from error
    if not isinstance(values, np.ndarray):
        raise TypeError(f"invalid corpus array {path.name}")
    return values


def _fill_raw_features(
    output: np.ndarray,
    artifacts: ArtifactSet,
) -> None:
    for start, stop in zip(artifacts.range_starts, artifacts.range_stops):
        begin, end = int(start), int(stop)
        output[begin:end] = _raw_features_for_range(
            artifacts.positions[begin:end],
            artifacts.velocities[begin:end],
            artifacts.rotations[begin:end],
            artifacts.parents,
            artifacts.terrain_features[begin:end],
        )


def _fill_successor(output: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> None:
    for start, stop in zip(starts, stops):
        begin, end = int(start), int(stop)
        output[begin : end - 1] = np.arange(begin + 1, end, dtype=np.int64)
        output[end - 1] = end - 1


def _fill_successor_valid(
    output: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> None:
    output[:] = True
    output[np.asarray(stops, dtype=np.int64) - 1] = False


def _fill_root_deltas(
    xy_output: np.ndarray,
    yaw_output: np.ndarray,
    artifacts: ArtifactSet,
) -> None:
    for start, stop in zip(artifacts.range_starts, artifacts.range_stops):
        begin, end = int(start), int(stop)
        delta_xy, delta_yaw = _root_deltas_for_range(
            artifacts.positions[begin:end], artifacts.rotations[begin:end]
        )
        xy_output[begin:end] = delta_xy
        yaw_output[begin:end] = delta_yaw


def assemble_full_corpus(
    lanes: Sequence[Path],
    output: Path,
    *,
    inventory: Path | FullWalkingInventory,
    split_ledger: Path | SplitLedger,
    scene_authority: Path | None = None,
) -> Path:
    """Verify source lanes and atomically publish one immutable mmap corpus."""

    inventory_value = _resolve_inventory(inventory)
    split_value = _resolve_split_ledger(split_ledger, inventory_value)
    lane_values = _resolve_lanes(lanes, inventory_value, split_value)
    ordered = _ordered_ranges(lane_values, inventory_value)
    scene_index, scenes = _scene_pack_snapshot(scene_authority)
    output = Path(output)
    request_sha256 = _request_identity(
        inventory_value, split_value, lane_values, scene_index, scenes
    )
    workspace = _BuildWorkspace(output, request_sha256)
    try:
        total_rows = sum(value.rows for value in ordered)
        if total_rows < 1 or total_rows > np.iinfo(np.int32).max:
            raise ValueError("full corpus rows must fit the Holden int32 ABI")
        range_count = len(ordered)
        source_names = tuple(
            sorted(source.source_id for source in inventory_value.sources)
        )
        canonical_names = tuple(
            sorted({source.canonical_source_id for source in inventory_value.sources})
        )
        terrain_names = tuple(
            sorted({source.terrain_id for source in inventory_value.sources})
        )
        range_names = tuple(value.record.range_id for value in ordered)
        source_index = {name: index for index, name in enumerate(source_names)}
        canonical_index = {name: index for index, name in enumerate(canonical_names)}
        terrain_index = {name: index for index, name in enumerate(terrain_names)}

        starts = np.empty(range_count, dtype="<i4")
        stops = np.empty(range_count, dtype="<i4")
        cursor = 0
        corpus_ranges: list[dict[str, object]] = []
        lane_global_ranges: dict[str, list[int]] = {
            lane.manifest_sha256: [] for lane in lane_values
        }
        for index, value in enumerate(ordered):
            starts[index] = cursor
            cursor += value.rows
            stops[index] = cursor
            adjusted = replace(value.record, start=int(starts[index]), stop=cursor)
            corpus_ranges.append(
                {
                    "record": asdict(adjusted),
                    "source_id": value.source_id,
                    "lane_manifest_sha256": value.lane.manifest_sha256,
                    "lane_range_index": value.lane_range_index,
                }
            )
            lane_global_ranges[value.lane.manifest_sha256].append(index)
        if cursor != total_rows:
            raise RuntimeError("full corpus row accounting failed")

        output_parent = output.parent.resolve()
        lane_bindings = []
        for lane in lane_values:
            try:
                relative = lane.root.resolve().relative_to(output_parent).as_posix()
            except ValueError as error:
                raise ValueError(
                    "lane roots must be inside the corpus output parent"
                ) from error
            lane_bindings.append(
                {
                    "path": relative,
                    "manifest_sha256": lane.manifest_sha256,
                    "inventory_manifest_sha256": lane.inventory_manifest_sha256,
                    "split_ledger_manifest_sha256": (lane.split_ledger_manifest_sha256),
                    "rows": len(lane.artifacts.positions),
                    "ranges": len(lane.ranges),
                    "global_range_indices": lane_global_ranges[lane.manifest_sha256],
                }
            )
        lane_bindings.sort(key=lambda value: value["manifest_sha256"])

        workspace.write_bytes(
            "inventory.json", inventory_manifest_bytes(inventory_value)
        )
        workspace.write_bytes("split-ledger.json", split_ledger_bytes(split_value))
        workspace.write_bytes("ranges.json", canonical_json_bytes(corpus_ranges))
        workspace.write_bytes(
            "terminal-outcomes.json",
            canonical_json_bytes(
                _terminal_outcomes(
                    tuple((value.record, value.source_id) for value in ordered),
                    inventory_value,
                )
            ),
        )
        workspace.write_bytes("lane-bindings.json", canonical_json_bytes(lane_bindings))
        workspace.write_bytes(
            "metadata.json",
            canonical_json_bytes(
                {
                    "schema": "g1-full-walking-corpus-metadata/v1",
                    "family_names": list(FAMILIES),
                    "split_names": list(SPLITS),
                    "quality_names": ["clean", "usable", "quarantined"],
                    "source_names": list(source_names),
                    "canonical_source_names": list(canonical_names),
                    "terrain_names": list(terrain_names),
                    "range_names": list(range_names),
                }
            ),
        )
        workspace.write_bytes("scenes/index.json", scene_index)
        for scene in scenes:
            for name, payload in (
                ("scene.json", scene.scene_json),
                ("terrain.bin", scene.terrain_bin),
                ("terrain.obj", scene.terrain_obj),
                ("walkability.bin", scene.walkability_bin),
            ):
                workspace.write_bytes(f"scenes/{scene.scene_id}/{name}", payload)

        workspace.write_npy(
            "parents.npy",
            dtype="<i4",
            shape=(31,),
            fill=lambda values: values.__setitem__(
                slice(None), np.asarray(G1_SKELETON_PARENTS, dtype="<i4")
            ),
        )
        workspace.write_npy(
            "range_starts.npy",
            dtype="<i4",
            shape=(range_count,),
            fill=lambda values: values.__setitem__(slice(None), starts),
        )
        workspace.write_npy(
            "range_stops.npy",
            dtype="<i4",
            shape=(range_count,),
            fill=lambda values: values.__setitem__(slice(None), stops),
        )
        artifact_members = (
            ("positions.npy", "positions", "<f4", (total_rows, 31, 3)),
            ("velocities.npy", "velocities", "<f4", (total_rows, 31, 3)),
            ("rotations.npy", "rotations", "<f4", (total_rows, 31, 4)),
            (
                "angular_velocities.npy",
                "angular_velocities",
                "<f4",
                (total_rows, 31, 3),
            ),
            ("contacts.npy", "contacts", "|u1", (total_rows, 2)),
            (
                "terrain_features.npy",
                "terrain_features",
                "<f4",
                (total_rows, 4),
            ),
            (
                "terrain_support.npy",
                "terrain_support",
                "<f4",
                (total_rows, 3),
            ),
        )
        for filename, attribute, dtype, shape in artifact_members:
            workspace.write_npy(
                filename,
                dtype=dtype,
                shape=shape,
                fill=lambda values, field=attribute: _copy_lane_field(
                    values, ordered, field
                ),
            )
        workspace.write_npy(
            "terrain_grid.npy",
            dtype="<f4",
            shape=(total_rows, 36),
            fill=lambda values: _copy_lane_array(values, ordered, "terrain_grid"),
        )
        for filename, attribute, dtype in (
            ("source_left_indices.npy", "source_left_indices", "<i4"),
            ("source_right_indices.npy", "source_right_indices", "<i4"),
            ("source_alpha.npy", "source_alpha", "<f4"),
        ):
            workspace.write_npy(
                filename,
                dtype=dtype,
                shape=(total_rows,),
                fill=lambda values, field=attribute: _copy_lane_array(
                    values, ordered, field
                ),
            )

        row_specs = {
            "family_ids.npy": np.dtype("|u1"),
            "source_ids.npy": np.dtype("<i4"),
            "canonical_source_ids.npy": np.dtype("<i4"),
            "terrain_ids.npy": np.dtype("<i4"),
            "range_ids.npy": np.dtype("<i4"),
            "split_ids.npy": np.dtype("|u1"),
            "quality_ids.npy": np.dtype("|u1"),
        }

        def fill_row_metadata(name: str, values: np.ndarray) -> None:
            for index, value in enumerate(ordered):
                start, stop = int(starts[index]), int(stops[index])
                record = value.record
                encoded = {
                    "family_ids.npy": FAMILY_IDS[record.family],
                    "source_ids.npy": source_index[value.source_id],
                    "canonical_source_ids.npy": canonical_index[
                        record.canonical_source_id
                    ],
                    "terrain_ids.npy": terrain_index[record.terrain_id],
                    "range_ids.npy": index,
                    "split_ids.npy": SPLIT_IDS[record.split],
                    "quality_ids.npy": QUALITY_IDS[record.quality],
                }[name]
                values[start:stop] = encoded

        for name, dtype in row_specs.items():
            workspace.write_npy(
                name,
                dtype=dtype,
                shape=(total_rows,),
                fill=lambda values, member=name: fill_row_metadata(member, values),
            )

        quality_ids = _load_mmap(workspace.staging / "quality_ids.npy")
        split_ids = _load_mmap(workspace.staging / "split_ids.npy")
        family_ids = _load_mmap(workspace.staging / "family_ids.npy")
        eligible = np.asarray(quality_ids != QUALITY_IDS["quarantined"], dtype=np.bool_)
        masks = {
            "eligible_mask.npy": eligible,
            "train_mask.npy": eligible & (split_ids == SPLIT_IDS["train"]),
            "validation_mask.npy": eligible & (split_ids == SPLIT_IDS["validation"]),
            "test_mask.npy": eligible & (split_ids == SPLIT_IDS["test"]),
        }
        for family_id, family in enumerate(FAMILIES):
            family_mask = family_ids == family_id
            for split, filename in (
                ("train", "train_mask.npy"),
                ("validation", "validation_mask.npy"),
                ("test", "test_mask.npy"),
            ):
                if not np.any(family_mask & masks[filename]):
                    raise ValueError(f"{family} has an empty eligible {split} stratum")
        for name, values in masks.items():
            workspace.write_npy(
                name,
                dtype=np.bool_,
                shape=(total_rows,),
                fill=lambda target, source=values: target.__setitem__(
                    slice(None), source
                ),
            )

        artifacts = ArtifactSet(
            positions=_load_mmap(workspace.staging / "positions.npy"),
            velocities=_load_mmap(workspace.staging / "velocities.npy"),
            rotations=_load_mmap(workspace.staging / "rotations.npy"),
            angular_velocities=_load_mmap(workspace.staging / "angular_velocities.npy"),
            parents=_load_mmap(workspace.staging / "parents.npy"),
            range_starts=_load_mmap(workspace.staging / "range_starts.npy"),
            range_stops=_load_mmap(workspace.staging / "range_stops.npy"),
            contacts=_load_mmap(workspace.staging / "contacts.npy"),
            terrain_features=_load_mmap(workspace.staging / "terrain_features.npy"),
            terrain_support=_load_mmap(workspace.staging / "terrain_support.npy"),
        )
        artifacts.validate()
        workspace.write_npy(
            "raw_features.npy",
            dtype="<f4",
            shape=(total_rows, 31),
            fill=lambda values: _fill_raw_features(values, artifacts),
        )
        raw_features = _load_mmap(workspace.staging / "raw_features.npy")
        offset, scale = _fit_normalization_float64(
            raw_features, masks["train_mask.npy"]
        )
        workspace.write_npy(
            "feature_offset.npy",
            dtype="<f4",
            shape=(31,),
            fill=lambda values: values.__setitem__(slice(None), offset),
        )
        workspace.write_npy(
            "feature_scale.npy",
            dtype="<f4",
            shape=(31,),
            fill=lambda values: values.__setitem__(slice(None), scale),
        )
        workspace.write_npy(
            "features.npy",
            dtype="<f4",
            shape=(total_rows, 31),
            fill=lambda values: _fill_normalized(values, raw_features, offset, scale),
        )

        workspace.write_npy(
            "successor.npy",
            dtype="<i8",
            shape=(total_rows,),
            fill=lambda values: _fill_successor(values, starts, stops),
        )
        workspace.write_npy(
            "successor_valid.npy",
            dtype=np.bool_,
            shape=(total_rows,),
            fill=lambda values: _fill_successor_valid(values, starts, stops),
        )

        def fill_delta_xy(values: np.ndarray) -> None:
            for start, stop in zip(starts, stops):
                begin, end = int(start), int(stop)
                delta_xy, _delta_yaw = _root_deltas_for_range(
                    artifacts.positions[begin:end], artifacts.rotations[begin:end]
                )
                values[begin:end] = delta_xy

        def fill_delta_yaw(values: np.ndarray) -> None:
            for start, stop in zip(starts, stops):
                begin, end = int(start), int(stop)
                _delta_xy, delta_yaw = _root_deltas_for_range(
                    artifacts.positions[begin:end], artifacts.rotations[begin:end]
                )
                values[begin:end] = delta_yaw

        workspace.write_npy(
            "root_delta_xy.npy",
            dtype="<f4",
            shape=(total_rows, 2),
            fill=fill_delta_xy,
        )
        workspace.write_npy(
            "root_delta_yaw.npy",
            dtype="<f4",
            shape=(total_rows,),
            fill=fill_delta_yaw,
        )

        source_ids = _load_mmap(workspace.staging / "source_ids.npy")
        eligible_rows = np.flatnonzero(eligible).astype(np.int64)
        source_order = np.argsort(source_ids[eligible_rows], kind="stable")
        source_rows = eligible_rows[source_order]
        source_counts = np.bincount(
            source_ids[eligible_rows], minlength=len(source_names)
        ).astype(np.int64)
        source_offsets = np.concatenate(
            (np.zeros(1, np.int64), np.cumsum(source_counts, dtype=np.int64))
        )
        workspace.write_npy(
            "eligible_rows.npy",
            dtype="<i8",
            shape=eligible_rows.shape,
            fill=lambda values: values.__setitem__(slice(None), eligible_rows),
        )
        workspace.write_npy(
            "source_rows.npy",
            dtype="<i8",
            shape=source_rows.shape,
            fill=lambda values: values.__setitem__(slice(None), source_rows),
        )
        workspace.write_npy(
            "source_row_offsets.npy",
            dtype="<i8",
            shape=source_offsets.shape,
            fill=lambda values: values.__setitem__(slice(None), source_offsets),
        )
        source_family = {
            source.source_id: source.family for source in inventory_value.sources
        }
        workspace.write_bytes(
            "sampling-groups.json",
            canonical_json_bytes(
                {
                    "schema": "g1-full-walking-family-source-views/v1",
                    "groups": [
                        {
                            "family": source_family[name],
                            "source_id": name,
                            "start": int(source_offsets[index]),
                            "stop": int(source_offsets[index + 1]),
                        }
                        for index, name in enumerate(source_names)
                    ],
                }
            ),
        )

        split_counts = {
            family: {
                split: int(
                    np.count_nonzero(
                        (family_ids == family_id) & masks[f"{split}_mask.npy"]
                    )
                )
                for split in SPLITS
            }
            for family_id, family in enumerate(FAMILIES)
        }
        manifest = {
            "schema": CORPUS_SCHEMA,
            "fps": FPS,
            "horizons": list(HORIZONS),
            "dimensions": {
                "bones": 31,
                "features": 31,
                "terrain_features": 4,
                "terrain_support": 3,
                "terrain_grid": 36,
            },
            "rows": total_rows,
            "ranges": range_count,
            "sources": len(source_names),
            "families": list(FAMILIES),
            "split_row_counts": split_counts,
            "eligible_rows": int(np.count_nonzero(eligible)),
            "inventory_manifest_sha256": inventory_value.manifest_sha256,
            "split_ledger_manifest_sha256": split_value.manifest_sha256,
            "lane_manifest_sha256": sorted(
                lane.manifest_sha256 for lane in lane_values
            ),
            "build_request_sha256": request_sha256,
            "skeleton": {
                "names": list(G1_SKELETON_NAMES),
                "parents": list(G1_SKELETON_PARENTS),
                "signature": G1_SKELETON_SIGNATURE,
            },
            "scene_index": {
                "path": "scenes/index.json",
                "sha256": hashlib.sha256(scene_index).hexdigest(),
                "scene_ids": list(SCENE_IDS),
            },
            "members": dict(sorted(workspace.completed.items())),
        }
        published = workspace.finish(
            manifest,
            validator=lambda path: _verify_full_corpus_contents(path),
        )
    except BaseException:
        workspace.close()
        raise
    load_full_corpus(published)
    return published


def _validate_member_tree(
    root: Path,
    manifest: Mapping[str, object],
    *,
    allow_resume_state: bool = False,
) -> None:
    members = manifest.get("members")
    if type(members) is not dict:
        raise ValueError("corpus manifest members must be an object")
    expected_files = {"manifest.json", *members}
    if allow_resume_state:
        expected_files.add("resume.json")
    actual_files: set[str] = set()
    for directory, subdirectories, files in os.walk(root):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise ValueError("corpus tree cannot contain symlink directories")
        for name in subdirectories:
            if (directory_path / name).is_symlink():
                raise ValueError("corpus tree cannot contain symlink directories")
        for name in files:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("corpus tree members must be regular files")
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            f"corpus member set changed: missing={missing} unexpected={unexpected}"
        )
    for relative, descriptor in members.items():
        if (
            type(relative) is not str
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(descriptor) is not dict
            or set(descriptor) != {"path", "size_bytes", "sha256"}
            or descriptor.get("path") != relative
            or type(descriptor.get("size_bytes")) is not int
            or descriptor["size_bytes"] < 0
            or not _is_sha256(descriptor.get("sha256"))
        ):
            raise ValueError(f"invalid corpus member descriptor {relative!r}")
        path = root / relative
        if (
            path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"{relative} corpus member SHA-256 or size changed")


def _require_array(
    root: Path,
    name: str,
    *,
    dtype: str,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    values = _load_mmap(root / name)
    if values.dtype.str != dtype:
        raise ValueError(f"{name} dtype changed from exact {dtype}")
    if values.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(values.shape, shape)
    ):
        raise ValueError(f"{name} shape changed: {values.shape}")
    if values.flags.writeable:
        raise ValueError(f"{name} must be opened as read-only mmap")
    return values


def _parse_corpus_ranges(
    root: Path, expected_count: int
) -> tuple[tuple[RangeRecord, str, str, int], ...]:
    value = _parse_canonical_file(root / "ranges.json", "corpus ranges")
    if type(value) is not list or len(value) != expected_count:
        raise ValueError("corpus range count changed")
    result = []
    record_keys = {
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
    for item in value:
        if (
            type(item) is not dict
            or set(item)
            != {"record", "source_id", "lane_manifest_sha256", "lane_range_index"}
            or type(item["record"]) is not dict
            or set(item["record"]) != record_keys
            or type(item["source_id"]) is not str
            or not _is_sha256(item["lane_manifest_sha256"])
            or type(item["lane_range_index"]) is not int
            or item["lane_range_index"] < 0
        ):
            raise ValueError("corpus range metadata keys or types changed")
        result.append(
            (
                RangeRecord(**item["record"]),
                item["source_id"],
                item["lane_manifest_sha256"],
                item["lane_range_index"],
            )
        )
    return tuple(result)


def _load_full_corpus_details(
    root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    _allow_resume_state: bool = False,
) -> tuple[
    FullWalkingCorpus,
    FullWalkingInventory,
    SplitLedger,
    tuple[LaneArtifact, ...],
    tuple[tuple[RangeRecord, str, str, int], ...],
    Mapping[str, object],
]:
    root = Path(root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("full corpus root must be a regular directory")
    manifest_path = root / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and (
        not _is_sha256(expected_manifest_sha256)
        or manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("full corpus manifest SHA-256 mismatch")
    manifest = _parse_canonical_file(manifest_path, "full corpus manifest")
    manifest_keys = {
        "schema",
        "fps",
        "horizons",
        "dimensions",
        "rows",
        "ranges",
        "sources",
        "families",
        "split_row_counts",
        "eligible_rows",
        "inventory_manifest_sha256",
        "split_ledger_manifest_sha256",
        "lane_manifest_sha256",
        "build_request_sha256",
        "skeleton",
        "scene_index",
        "members",
    }
    if type(manifest) is not dict or set(manifest) != manifest_keys:
        raise ValueError("full corpus manifest keys changed")
    if (
        manifest["schema"] != CORPUS_SCHEMA
        or manifest["fps"] != FPS
        or manifest["horizons"] != list(HORIZONS)
        or manifest["families"] != list(FAMILIES)
        or manifest["dimensions"]
        != {
            "bones": 31,
            "features": 31,
            "terrain_features": 4,
            "terrain_support": 3,
            "terrain_grid": 36,
        }
        or type(manifest["rows"]) is not int
        or manifest["rows"] < 1
        or type(manifest["ranges"]) is not int
        or manifest["ranges"] < 1
        or type(manifest["sources"]) is not int
        or manifest["sources"] < 1
        or not _is_sha256(manifest["inventory_manifest_sha256"])
        or not _is_sha256(manifest["split_ledger_manifest_sha256"])
        or not _is_sha256(manifest["build_request_sha256"])
        or type(manifest["lane_manifest_sha256"]) is not list
        or not all(_is_sha256(value) for value in manifest["lane_manifest_sha256"])
        or manifest["skeleton"]
        != {
            "names": list(G1_SKELETON_NAMES),
            "parents": list(G1_SKELETON_PARENTS),
            "signature": G1_SKELETON_SIGNATURE,
        }
    ):
        raise ValueError("full corpus fixed 60 Hz ABI changed")
    _validate_member_tree(root, manifest, allow_resume_state=_allow_resume_state)

    inventory_payload = (root / "inventory.json").read_bytes()
    inventory = _load_inventory_bytes(inventory_payload)
    if inventory.manifest_sha256 != manifest["inventory_manifest_sha256"]:
        raise ValueError("corpus inventory manifest binding changed")
    split_payload = (root / "split-ledger.json").read_bytes()
    split_ledger = _load_split_bytes(split_payload, inventory)
    if split_ledger.manifest_sha256 != manifest["split_ledger_manifest_sha256"]:
        raise ValueError("corpus split ledger manifest binding changed")

    bindings = _parse_canonical_file(
        root / "lane-bindings.json", "corpus lane bindings"
    )
    if type(bindings) is not list or not bindings:
        raise ValueError("corpus lane bindings must be a nonempty array")
    lanes: list[LaneArtifact] = []
    binding_keys = {
        "path",
        "manifest_sha256",
        "inventory_manifest_sha256",
        "split_ledger_manifest_sha256",
        "rows",
        "ranges",
        "global_range_indices",
    }
    for binding in bindings:
        if (
            type(binding) is not dict
            or set(binding) != binding_keys
            or type(binding["path"]) is not str
            or Path(binding["path"]).is_absolute()
            or ".." in Path(binding["path"]).parts
            or not _is_sha256(binding["manifest_sha256"])
            or binding["inventory_manifest_sha256"] != inventory.manifest_sha256
            or binding["split_ledger_manifest_sha256"] != split_ledger.manifest_sha256
            or type(binding["global_range_indices"]) is not list
        ):
            raise ValueError("corpus lane binding keys or authority changed")
        lane_path = (root.parent / binding["path"]).resolve(strict=True)
        try:
            lane_path.relative_to(root.parent)
        except ValueError as error:
            raise ValueError("corpus lane binding escapes its run root") from error
        lane = load_lane(
            lane_path,
            inventory=inventory,
            split_ledger=split_ledger,
            expected_manifest_sha256=binding["manifest_sha256"],
            expected_inventory_manifest_sha256=inventory.manifest_sha256,
            expected_split_ledger_manifest_sha256=split_ledger.manifest_sha256,
        )
        if (
            len(lane.artifacts.positions) != binding["rows"]
            or len(lane.ranges) != binding["ranges"]
        ):
            raise ValueError("corpus lane binding counts changed")
        lanes.append(lane)
    lane_hashes = tuple(lane.manifest_sha256 for lane in lanes)
    if list(lane_hashes) != manifest["lane_manifest_sha256"]:
        raise ValueError("corpus lane manifest ordering or coverage changed")
    if lane_hashes != tuple(sorted(set(lane_hashes))):
        raise ValueError("corpus lane bindings must be uniquely digest ordered")

    metadata = _parse_canonical_file(root / "metadata.json", "corpus metadata")
    metadata_keys = {
        "schema",
        "family_names",
        "split_names",
        "quality_names",
        "source_names",
        "canonical_source_names",
        "terrain_names",
        "range_names",
    }
    if (
        type(metadata) is not dict
        or set(metadata) != metadata_keys
        or metadata["schema"] != "g1-full-walking-corpus-metadata/v1"
        or metadata["family_names"] != list(FAMILIES)
        or metadata["split_names"] != list(SPLITS)
        or metadata["quality_names"] != ["clean", "usable", "quarantined"]
        or not all(
            type(metadata[key]) is list
            and all(type(value) is str for value in metadata[key])
            for key in (
                "source_names",
                "canonical_source_names",
                "terrain_names",
                "range_names",
            )
        )
    ):
        raise ValueError("corpus metadata schema changed")
    source_names = tuple(metadata["source_names"])
    canonical_names = tuple(metadata["canonical_source_names"])
    terrain_names = tuple(metadata["terrain_names"])
    range_names = tuple(metadata["range_names"])
    expected_source_names = tuple(
        sorted(source.source_id for source in inventory.sources)
    )
    expected_canonical_names = tuple(
        sorted({source.canonical_source_id for source in inventory.sources})
    )
    expected_terrain_names = tuple(
        sorted({source.terrain_id for source in inventory.sources})
    )
    if (
        source_names != expected_source_names
        or canonical_names != expected_canonical_names
        or terrain_names != expected_terrain_names
    ):
        raise ValueError("corpus metadata name tables differ from inventory authority")
    rows, ranges = manifest["rows"], manifest["ranges"]
    if len(source_names) != manifest["sources"] or len(range_names) != ranges:
        raise ValueError("corpus metadata counts changed")
    range_values = _parse_corpus_ranges(root, ranges)
    if range_names != tuple(value[0].range_id for value in range_values):
        raise ValueError("corpus range name metadata differs from range authority")
    if {value[1] for value in range_values} != set(source_names):
        raise ValueError("corpus range metadata lacks exact source terminal coverage")
    expected_ranges_by_lane: dict[str, list[int]] = {
        digest: [] for digest in lane_hashes
    }
    for index, value in enumerate(range_values):
        try:
            expected_ranges_by_lane[value[2]].append(index)
        except KeyError as error:
            raise ValueError("corpus range names an unbound lane shard") from error
    for binding in bindings:
        if (
            binding["global_range_indices"]
            != expected_ranges_by_lane[binding["manifest_sha256"]]
        ):
            raise ValueError("corpus lane global range shard mapping changed")

    artifacts = ArtifactSet(
        positions=_require_array(
            root, "positions.npy", dtype="<f4", shape=(rows, 31, 3)
        ),
        velocities=_require_array(
            root, "velocities.npy", dtype="<f4", shape=(rows, 31, 3)
        ),
        rotations=_require_array(
            root, "rotations.npy", dtype="<f4", shape=(rows, 31, 4)
        ),
        angular_velocities=_require_array(
            root,
            "angular_velocities.npy",
            dtype="<f4",
            shape=(rows, 31, 3),
        ),
        parents=_require_array(root, "parents.npy", dtype="<i4", shape=(31,)),
        range_starts=_require_array(
            root, "range_starts.npy", dtype="<i4", shape=(ranges,)
        ),
        range_stops=_require_array(
            root, "range_stops.npy", dtype="<i4", shape=(ranges,)
        ),
        contacts=_require_array(root, "contacts.npy", dtype="|u1", shape=(rows, 2)),
        terrain_features=_require_array(
            root, "terrain_features.npy", dtype="<f4", shape=(rows, 4)
        ),
        terrain_support=_require_array(
            root, "terrain_support.npy", dtype="<f4", shape=(rows, 3)
        ),
    )
    artifacts.validate()
    features = FeatureSet(
        values=_require_array(root, "features.npy", dtype="<f4", shape=(rows, 31)),
        offset=_require_array(root, "feature_offset.npy", dtype="<f4", shape=(31,)),
        scale=_require_array(root, "feature_scale.npy", dtype="<f4", shape=(31,)),
    )
    features.validate()
    raw_features = _require_array(
        root, "raw_features.npy", dtype="<f4", shape=(rows, 31)
    )
    terrain_grid = _require_array(
        root, "terrain_grid.npy", dtype="<f4", shape=(rows, 36)
    )
    family_ids = _require_array(root, "family_ids.npy", dtype="|u1", shape=(rows,))
    source_ids = _require_array(root, "source_ids.npy", dtype="<i4", shape=(rows,))
    canonical_source_ids = _require_array(
        root, "canonical_source_ids.npy", dtype="<i4", shape=(rows,)
    )
    terrain_ids = _require_array(root, "terrain_ids.npy", dtype="<i4", shape=(rows,))
    range_ids = _require_array(root, "range_ids.npy", dtype="<i4", shape=(rows,))
    split_ids = _require_array(root, "split_ids.npy", dtype="|u1", shape=(rows,))
    quality_ids = _require_array(root, "quality_ids.npy", dtype="|u1", shape=(rows,))
    eligible_mask = _require_array(
        root, "eligible_mask.npy", dtype="|b1", shape=(rows,)
    )
    train_mask = _require_array(root, "train_mask.npy", dtype="|b1", shape=(rows,))
    validation_mask = _require_array(
        root, "validation_mask.npy", dtype="|b1", shape=(rows,)
    )
    test_mask = _require_array(root, "test_mask.npy", dtype="|b1", shape=(rows,))
    eligible_rows = _require_array(
        root, "eligible_rows.npy", dtype="<i8", shape=(None,)
    )
    source_rows = _require_array(
        root, "source_rows.npy", dtype="<i8", shape=(len(eligible_rows),)
    )
    source_row_offsets = _require_array(
        root,
        "source_row_offsets.npy",
        dtype="<i8",
        shape=(len(source_names) + 1,),
    )
    source_left_indices = _require_array(
        root, "source_left_indices.npy", dtype="<i4", shape=(rows,)
    )
    source_right_indices = _require_array(
        root, "source_right_indices.npy", dtype="<i4", shape=(rows,)
    )
    source_alpha = _require_array(root, "source_alpha.npy", dtype="<f4", shape=(rows,))
    successor = _require_array(root, "successor.npy", dtype="<i8", shape=(rows,))
    successor_valid = _require_array(
        root, "successor_valid.npy", dtype="|b1", shape=(rows,)
    )
    root_delta_xy = _require_array(
        root, "root_delta_xy.npy", dtype="<f4", shape=(rows, 2)
    )
    root_delta_yaw = _require_array(
        root, "root_delta_yaw.npy", dtype="<f4", shape=(rows,)
    )
    if (
        np.any(family_ids >= len(FAMILIES))
        or np.any(source_ids < 0)
        or np.any(source_ids >= len(source_names))
        or np.any(canonical_source_ids < 0)
        or np.any(canonical_source_ids >= len(canonical_names))
        or np.any(terrain_ids < 0)
        or np.any(terrain_ids >= len(terrain_names))
        or np.any(range_ids < 0)
        or np.any(range_ids >= ranges)
        or np.any(split_ids >= len(SPLITS))
        or np.any(quality_ids > QUALITY_IDS["quarantined"])
        or np.any(source_left_indices < 0)
        or np.any(source_left_indices > source_right_indices)
        or np.any(source_alpha < 0.0)
        or np.any(source_alpha > 1.0)
        or np.any(successor < 0)
        or np.any(successor >= rows)
    ):
        raise ValueError("corpus row metadata or source provenance is out of bounds")
    if not all(
        np.isfinite(values).all()
        for values in (
            raw_features,
            terrain_grid,
            root_delta_xy,
            root_delta_yaw,
            source_alpha,
        )
    ):
        raise ValueError("corpus row members contain nonfinite values")

    scene_descriptor = manifest["scene_index"]
    scene_payload = (root / "scenes/index.json").read_bytes()
    scene_index = _parse_canonical_bytes(scene_payload, "corpus scene index")
    if (
        type(scene_descriptor) is not dict
        or scene_descriptor
        != {
            "path": "scenes/index.json",
            "sha256": hashlib.sha256(scene_payload).hexdigest(),
            "scene_ids": list(SCENE_IDS),
        }
        or type(scene_index) is not dict
        or scene_index.get("scene_ids") != list(SCENE_IDS)
        or scene_index.get("default_scene_id") != "flat-standard"
    ):
        raise ValueError("corpus authenticated scene pack changed")

    corpus = FullWalkingCorpus(
        root=root,
        artifacts=artifacts,
        features=features,
        raw_features=raw_features,
        terrain_grid=terrain_grid,
        family_ids=family_ids,
        source_ids=source_ids,
        source_names=source_names,
        canonical_source_ids=canonical_source_ids,
        canonical_source_names=canonical_names,
        terrain_ids=terrain_ids,
        terrain_names=terrain_names,
        range_ids=range_ids,
        range_names=range_names,
        split_ids=split_ids,
        quality_ids=quality_ids,
        eligible_mask=eligible_mask,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        eligible_rows=eligible_rows,
        source_row_offsets=source_row_offsets,
        source_rows=source_rows,
        source_left_indices=source_left_indices,
        source_right_indices=source_right_indices,
        source_alpha=source_alpha,
        successor=successor,
        successor_valid=successor_valid,
        root_delta_xy=root_delta_xy,
        root_delta_yaw=root_delta_yaw,
        manifest_sha256=manifest_sha256,
        inventory_manifest_sha256=inventory.manifest_sha256,
        split_ledger_manifest_sha256=split_ledger.manifest_sha256,
        lane_manifest_sha256=lane_hashes,
    )
    return corpus, inventory, split_ledger, tuple(lanes), range_values, manifest


def load_full_corpus(
    root: Path, *, expected_manifest_sha256: str | None = None
) -> FullWalkingCorpus:
    """Authenticate every combined member and reopen arrays as read-only mmap."""

    corpus, _inventory, _split, _lanes, _ranges, _manifest = _load_full_corpus_details(
        root, expected_manifest_sha256=expected_manifest_sha256
    )
    return corpus


def _independent_raw_features(
    artifacts: ArtifactSet, start: int, stop: int
) -> np.ndarray:
    """Second implementation used only by verification, not by assembly."""

    positions = artifacts.positions[start:stop].astype(np.float64)
    rotations = artifacts.rotations[start:stop].astype(np.float64)
    velocities = artifacts.velocities[start:stop].astype(np.float64)
    angular = artifacts.angular_velocities[start:stop].astype(np.float64)
    parents = artifacts.parents
    frames, bones = positions.shape[:2]
    global_positions = np.empty((frames, bones, 3), np.float64)
    global_rotations = np.empty((frames, bones, 4), np.float64)
    global_velocities = np.empty((frames, bones, 3), np.float64)
    global_angular = np.empty((frames, bones, 3), np.float64)
    for bone in range(bones):
        parent = int(parents[bone])
        if parent == -1:
            global_positions[:, bone] = positions[:, bone]
            global_rotations[:, bone] = rotations[:, bone]
            global_velocities[:, bone] = velocities[:, bone]
            global_angular[:, bone] = angular[:, bone]
        else:
            offset = holden_quat.mul_vec(
                global_rotations[:, parent], positions[:, bone]
            )
            global_positions[:, bone] = global_positions[:, parent] + offset
            global_rotations[:, bone] = holden_quat.mul(
                global_rotations[:, parent], rotations[:, bone]
            )
            global_velocities[:, bone] = (
                global_velocities[:, parent]
                + holden_quat.mul_vec(global_rotations[:, parent], velocities[:, bone])
                + np.cross(global_angular[:, parent], offset)
            )
            global_angular[:, bone] = global_angular[:, parent] + holden_quat.mul_vec(
                global_rotations[:, parent], angular[:, bone]
            )
    inverse = holden_quat.inv(global_rotations[:, 0])
    result = np.empty((frames, 31), np.float32)
    for output, bone in ((0, 6), (3, 12)):
        result[:, output : output + 3] = holden_quat.mul_vec(
            inverse, global_positions[:, bone] - global_positions[:, 0]
        )
    for output, bone in ((6, 6), (9, 12), (12, 1)):
        result[:, output : output + 3] = holden_quat.mul_vec(
            inverse, global_velocities[:, bone]
        )
    row = np.arange(frames, dtype=np.int64)
    for slot, horizon in enumerate(HORIZONS):
        future = np.minimum(row + horizon, frames - 1)
        position = holden_quat.mul_vec(
            inverse, global_positions[future, 0] - global_positions[:, 0]
        )
        world_facing = holden_quat.mul_vec(
            global_rotations[future, 0], np.asarray([0.0, 0.0, 1.0])
        )
        facing = holden_quat.mul_vec(inverse, world_facing)
        result[:, 15 + 2 * slot : 17 + 2 * slot] = position[:, (0, 2)]
        result[:, 21 + 2 * slot : 23 + 2 * slot] = facing[:, (0, 2)]
    result[:, 27:31] = artifacts.terrain_features[start:stop]
    if not np.isfinite(result).all():
        raise ValueError("independently reconstructed raw features are nonfinite")
    return result


def _verify_normalization_independent(corpus: FullWalkingCorpus) -> None:
    selected = np.flatnonzero(corpus.train_mask)
    if not len(selected):
        raise ValueError("normalization train mask is empty")
    raw = corpus.raw_features
    expected_offset = np.empty(31, np.float64)
    expected_scale = np.empty(31, np.float64)
    for (start, stop), weight in zip(_FEATURE_GROUPS, _NORMALIZATION_WEIGHTS):
        sums = np.zeros(stop - start, np.float64)
        count = 0
        for block_start in range(0, len(selected), _BLOCK_ROWS):
            indices = selected[block_start : block_start + _BLOCK_ROWS]
            values = np.asarray(raw[indices, start:stop], dtype=np.float64)
            sums += values.sum(axis=0, dtype=np.float64)
            count += len(values)
        mean = sums / count
        squares = np.zeros(stop - start, np.float64)
        for block_start in range(0, len(selected), _BLOCK_ROWS):
            indices = selected[block_start : block_start + _BLOCK_ROWS]
            values = np.asarray(raw[indices, start:stop], dtype=np.float64)
            squares += np.square(values - mean).sum(axis=0, dtype=np.float64)
        group_std = float(np.mean(np.sqrt(squares / count)))
        expected_offset[start:stop] = mean
        expected_scale[start:stop] = (
            float(_DISABLED_SCALE)
            if not np.isfinite(group_std) or group_std <= 0.0 or weight == 0.0
            else group_std / weight
        )
    if not np.allclose(
        corpus.features.offset,
        expected_offset,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise ValueError("train-only float64 feature normalization offset changed")
    if not np.allclose(
        corpus.features.scale,
        expected_scale,
        rtol=2e-5,
        atol=2e-6,
    ):
        raise ValueError("train-only float64 feature normalization scale changed")
    for block_start in range(0, len(raw), _BLOCK_ROWS):
        block_stop = min(block_start + _BLOCK_ROWS, len(raw))
        source = np.asarray(raw[block_start:block_stop], dtype=np.float64)
        expected = np.empty(source.shape, np.float32)
        for start, stop in _FEATURE_GROUPS:
            if np.all(corpus.features.scale[start:stop] == _DISABLED_SCALE):
                expected[:, start:stop] = 0.0
            else:
                expected[:, start:stop] = (
                    (source[:, start:stop] - corpus.features.offset[start:stop])
                    / corpus.features.scale[start:stop]
                ).astype(np.float32)
        if not np.allclose(
            corpus.features.values[block_start:block_stop],
            expected,
            rtol=2e-6,
            atol=2e-6,
        ):
            raise ValueError("normalized feature table changed from raw features")


def _verify_full_corpus_contents(root: Path) -> Mapping[str, object]:
    corpus, inventory, ledger, lanes, ranges, manifest = _load_full_corpus_details(root)
    expected_inputs = _ordered_ranges(lanes, inventory)
    expected_ranges = []
    cursor = 0
    for value in expected_inputs:
        stop = cursor + value.rows
        expected_ranges.append(
            (
                replace(value.record, start=cursor, stop=stop),
                value.source_id,
                value.lane.manifest_sha256,
                value.lane_range_index,
            )
        )
        cursor = stop
    if tuple(expected_ranges) != ranges:
        raise ValueError("corpus terminal range coverage or packed ordering changed")
    terminal_outcomes = _parse_canonical_file(
        corpus.root / "terminal-outcomes.json", "corpus terminal outcomes"
    )
    expected_terminal_outcomes = _terminal_outcomes(
        tuple((record, source_id) for record, source_id, _lane, _index in ranges),
        inventory,
    )
    if terminal_outcomes != expected_terminal_outcomes:
        raise ValueError("corpus terminal outcome or exclusion ledger changed")
    for lane in lanes:
        _verify_lane_contents(lane, inventory, ledger)
    lane_by_sha = {lane.manifest_sha256: lane for lane in lanes}
    source_index = {name: index for index, name in enumerate(corpus.source_names)}
    canonical_index = {
        name: index for index, name in enumerate(corpus.canonical_source_names)
    }
    terrain_index = {name: index for index, name in enumerate(corpus.terrain_names)}
    ordered_keys = []
    for global_index, (record, source_id, lane_sha, lane_range_index) in enumerate(
        ranges
    ):
        lane = lane_by_sha.get(lane_sha)
        if lane is None or lane_range_index >= len(lane.ranges):
            raise ValueError("corpus range names an absent lane parent")
        parent = lane.ranges[lane_range_index]
        start, stop = record.start, record.stop
        parent_start, parent_stop = parent.start, parent.stop
        if (
            replace(parent, start=start, stop=stop) != record
            or lane.source_ids[lane_range_index] != source_id
            or int(corpus.artifacts.range_starts[global_index]) != start
            or int(corpus.artifacts.range_stops[global_index]) != stop
            or stop - start != parent_stop - parent_start
        ):
            raise ValueError("corpus range/source parent reconstruction changed")
        ordered_keys.append(
            (
                FAMILY_IDS[record.family],
                record.canonical_source_id,
                record.mirror_of or "",
                record.range_id,
            )
        )
        for label, packed, authority in (
            (
                "positions",
                corpus.artifacts.positions[start:stop],
                lane.artifacts.positions[parent_start:parent_stop],
            ),
            (
                "velocity",
                corpus.artifacts.velocities[start:stop],
                lane.artifacts.velocities[parent_start:parent_stop],
            ),
            (
                "rotations",
                corpus.artifacts.rotations[start:stop],
                lane.artifacts.rotations[parent_start:parent_stop],
            ),
            (
                "angular velocity",
                corpus.artifacts.angular_velocities[start:stop],
                lane.artifacts.angular_velocities[parent_start:parent_stop],
            ),
            (
                "contact",
                corpus.artifacts.contacts[start:stop],
                lane.artifacts.contacts[parent_start:parent_stop],
            ),
            (
                "terrain feature",
                corpus.artifacts.terrain_features[start:stop],
                lane.artifacts.terrain_features[parent_start:parent_stop],
            ),
            (
                "terrain support",
                corpus.artifacts.terrain_support[start:stop],
                lane.artifacts.terrain_support[parent_start:parent_stop],
            ),
            (
                "terrain grid",
                corpus.terrain_grid[start:stop],
                lane.terrain_grid[parent_start:parent_stop],
            ),
            (
                "source map left provenance",
                corpus.source_left_indices[start:stop],
                lane.source_left_indices[parent_start:parent_stop],
            ),
            (
                "source map right provenance",
                corpus.source_right_indices[start:stop],
                lane.source_right_indices[parent_start:parent_stop],
            ),
            (
                "source map alpha provenance",
                corpus.source_alpha[start:stop],
                lane.source_alpha[parent_start:parent_stop],
            ),
        ):
            if not np.array_equal(packed, authority):
                raise ValueError(f"corpus {label} differs from authenticated lane")
        expected_raw = _independent_raw_features(corpus.artifacts, start, stop)
        if not np.allclose(
            corpus.raw_features[start:stop], expected_raw, rtol=2e-6, atol=2e-6
        ):
            raise ValueError("corpus raw feature range reconstruction changed")
        expected_row_values = {
            "family": (corpus.family_ids[start:stop], FAMILY_IDS[record.family]),
            "source": (corpus.source_ids[start:stop], source_index[source_id]),
            "canonical source": (
                corpus.canonical_source_ids[start:stop],
                canonical_index[record.canonical_source_id],
            ),
            "terrain": (
                corpus.terrain_ids[start:stop],
                terrain_index[record.terrain_id],
            ),
            "range": (corpus.range_ids[start:stop], global_index),
            "split": (corpus.split_ids[start:stop], SPLIT_IDS[record.split]),
            "quality": (
                corpus.quality_ids[start:stop],
                QUALITY_IDS[record.quality],
            ),
        }
        for label, (values, expected) in expected_row_values.items():
            if not np.all(values == expected):
                raise ValueError(f"corpus {label} row metadata changed")

        expected_successor = np.arange(start + 1, stop + 1, dtype=np.int64)
        expected_successor[-1] = stop - 1
        if not np.array_equal(corpus.successor[start:stop], expected_successor):
            raise ValueError("corpus range-safe successor changed")
        expected_valid = np.ones(stop - start, np.bool_)
        expected_valid[-1] = False
        if not np.array_equal(corpus.successor_valid[start:stop], expected_valid):
            raise ValueError("corpus successor validity crossed a range")
        expected_xy, expected_yaw = _root_deltas_for_range(
            corpus.artifacts.positions[start:stop],
            corpus.artifacts.rotations[start:stop],
        )
        if not np.array_equal(corpus.root_delta_xy[start:stop], expected_xy):
            raise ValueError("corpus SE(2) root delta XY changed")
        if not np.array_equal(corpus.root_delta_yaw[start:stop], expected_yaw):
            raise ValueError("corpus SE(2) root delta yaw changed")
    if ordered_keys != sorted(ordered_keys):
        raise ValueError("corpus deterministic range ordering changed")

    expected_eligible = corpus.quality_ids != QUALITY_IDS["quarantined"]
    expected_masks = (
        expected_eligible & (corpus.split_ids == SPLIT_IDS["train"]),
        expected_eligible & (corpus.split_ids == SPLIT_IDS["validation"]),
        expected_eligible & (corpus.split_ids == SPLIT_IDS["test"]),
    )
    if not np.array_equal(corpus.eligible_mask, expected_eligible):
        raise ValueError("corpus clean + usable eligibility mask changed")
    for label, actual, expected in zip(
        ("train", "validation", "test"),
        (corpus.train_mask, corpus.validation_mask, corpus.test_mask),
        expected_masks,
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"corpus {label} split mask changed")
        for family_id, family in enumerate(FAMILIES):
            if not np.any(expected & (corpus.family_ids == family_id)):
                raise ValueError(f"corpus {family} has empty {label} stratum")
    _verify_normalization_independent(corpus)

    expected_eligible_rows = np.flatnonzero(expected_eligible).astype(np.int64)
    order = np.argsort(corpus.source_ids[expected_eligible_rows], kind="stable")
    expected_source_rows = expected_eligible_rows[order]
    counts = np.bincount(
        corpus.source_ids[expected_eligible_rows], minlength=len(corpus.source_names)
    ).astype(np.int64)
    expected_offsets = np.concatenate(
        (np.zeros(1, np.int64), np.cumsum(counts, dtype=np.int64))
    )
    if not np.array_equal(corpus.eligible_rows, expected_eligible_rows):
        raise ValueError("corpus eligible row view changed")
    if not np.array_equal(corpus.source_rows, expected_source_rows):
        raise ValueError("corpus source-balanced row view changed")
    if not np.array_equal(corpus.source_row_offsets, expected_offsets):
        raise ValueError("corpus source-balanced offsets changed")
    sampling = _parse_canonical_file(
        corpus.root / "sampling-groups.json", "corpus sampling groups"
    )
    if (
        type(sampling) is not dict
        or sampling.get("schema") != "g1-full-walking-family-source-views/v1"
        or type(sampling.get("groups")) is not list
        or len(sampling["groups"]) != len(corpus.source_names)
    ):
        raise ValueError("corpus family/source sampling view schema changed")
    source_by_id = {source.source_id: source for source in inventory.sources}
    expected_groups = [
        {
            "family": source_by_id[name].family,
            "source_id": name,
            "start": int(expected_offsets[index]),
            "stop": int(expected_offsets[index + 1]),
        }
        for index, name in enumerate(corpus.source_names)
    ]
    if sampling["groups"] != expected_groups:
        raise ValueError("corpus family/source sampling groups changed")

    scene_index_path = corpus.root / "scenes/index.json"
    scene_index_payload = scene_index_path.read_bytes()
    scene_index = _parse_canonical_bytes(
        scene_index_payload, "corpus verification scene index"
    )
    if (
        type(scene_index) is not dict
        or scene_index.get("schema") != "g1-terrain-scene-index/v1"
        or scene_index.get("scene_ids") != list(SCENE_IDS)
        or type(scene_index.get("scenes")) is not list
        or len(scene_index["scenes"]) != len(SCENE_IDS)
    ):
        raise ValueError("corpus verification scene index schema changed")
    descriptors = {}
    for value in scene_index["scenes"]:
        if (
            type(value) is not dict
            or set(value) != {"id", "path", "sha256"}
            or value.get("id") not in SCENE_IDS
            or value.get("path") != f"scenes/{value.get('id')}/scene.json"
            or not _is_sha256(value.get("sha256"))
        ):
            raise ValueError("corpus verification scene descriptor changed")
        descriptors[value["id"]] = value
    verified_scenes = []
    for scene_id in SCENE_IDS:
        if scene_id not in descriptors:
            raise ValueError("corpus verification scene coverage changed")
        verified_scenes.append(
            _authenticate_scene_payload(
                scene_id,
                corpus.root / "scenes" / scene_id,
                expected_scene_sha256=descriptors[scene_id]["sha256"],
            )
        )
    expected_request = _request_identity(
        inventory, ledger, lanes, scene_index_payload, verified_scenes
    )
    if manifest["build_request_sha256"] != expected_request:
        raise ValueError("corpus build request identity changed")
    if manifest["eligible_rows"] != int(np.count_nonzero(expected_eligible)):
        raise ValueError("corpus manifest eligible count changed")
    split_counts = {
        family: {
            split: int(
                np.count_nonzero(
                    (corpus.family_ids == family_id) & expected_masks[split_id]
                )
            )
            for split_id, split in enumerate(SPLITS)
        }
        for family_id, family in enumerate(FAMILIES)
    }
    if manifest["split_row_counts"] != split_counts:
        raise ValueError("corpus manifest family/split counts changed")
    return {
        "schema": VERIFY_SCHEMA,
        "status": "accepted",
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "inventory_manifest_sha256": corpus.inventory_manifest_sha256,
        "split_ledger_manifest_sha256": corpus.split_ledger_manifest_sha256,
        "lane_manifest_sha256": list(corpus.lane_manifest_sha256),
        "rows": len(corpus.artifacts.positions),
        "ranges": len(corpus.artifacts.range_starts),
        "sources": len(corpus.source_names),
        "eligible_rows": int(np.count_nonzero(corpus.eligible_mask)),
        "family_split_rows": split_counts,
        "fps": corpus.fps,
        "horizons": list(corpus.horizons),
    }


def verify_full_corpus(root: Path, receipt: Path) -> Path:
    """Independently reconstruct every lane/range and publish a receipt."""

    root = Path(root).resolve(strict=True)
    receipt = Path(receipt).resolve()
    try:
        receipt.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("verification receipt must be outside immutable corpus")
    result = _verify_full_corpus_contents(root)
    return _publish_file_exclusive(canonical_json_bytes(dict(result)), Path(receipt))


def _tree_descriptors(root: Path) -> dict[str, dict[str, object]]:
    descriptors = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        descriptors[relative] = _member_descriptor(path, relative)
    return descriptors


def reproduce_full_corpus(
    corpus: Path,
    lanes: Sequence[Path],
    scratch: Path,
    receipt: Path,
    *,
    inventory: Path | FullWalkingInventory,
    split_ledger: Path | SplitLedger,
    scene_authority: Path | None = None,
) -> Path:
    """Build a second corpus, compare every byte, then remove owned scratch."""

    reference_root = Path(corpus).resolve(strict=True)
    scratch = Path(scratch).resolve()
    receipt = Path(receipt).resolve()
    try:
        receipt.relative_to(scratch)
    except ValueError:
        pass
    else:
        raise ValueError("reproduction receipt must be outside owned scratch")
    try:
        receipt.relative_to(reference_root)
    except ValueError:
        pass
    else:
        raise ValueError("reproduction receipt must be outside immutable corpus")
    reference_result = _verify_full_corpus_contents(reference_root)
    reference = load_full_corpus(reference_root)
    reproduced_path = assemble_full_corpus(
        lanes,
        scratch,
        inventory=inventory,
        split_ledger=split_ledger,
        scene_authority=scene_authority,
    )
    reproduced_result = _verify_full_corpus_contents(reproduced_path)
    reference_files = _tree_descriptors(reference.root)
    reproduced_files = _tree_descriptors(reproduced_path)
    if reference_files != reproduced_files:
        changed = sorted(
            relative
            for relative in set(reference_files) | set(reproduced_files)
            if reference_files.get(relative) != reproduced_files.get(relative)
        )
        raise ValueError(
            f"full corpus reproduction differs at {changed[:5]}; scratch preserved"
        )
    if reference_result != reproduced_result:
        raise ValueError("full corpus reproduction verification receipt changed")
    result = {
        "schema": DETERMINISM_SCHEMA,
        "status": "accepted",
        "reference_manifest_sha256": reference.manifest_sha256,
        "reproduced_manifest_sha256": load_full_corpus(reproduced_path).manifest_sha256,
        "members": reference_files,
    }
    published = _publish_file_exclusive(canonical_json_bytes(result), receipt)
    # This target was supplied as this command's dedicated scratch output and
    # has just been authenticated as the exact reproduced corpus.
    shutil.rmtree(reproduced_path)
    _fsync_directory(reproduced_path.parent)
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--split-ledger", type=Path, required=True)
    build.add_argument("--lane", type=Path, action="append", required=True)
    build.add_argument("--scene-authority", type=Path)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--corpus", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    reproduce = commands.add_parser("reproduce")
    reproduce.add_argument("--corpus", type=Path, required=True)
    reproduce.add_argument("--inventory", type=Path, required=True)
    reproduce.add_argument("--split-ledger", type=Path, required=True)
    reproduce.add_argument("--lane", type=Path, action="append", required=True)
    reproduce.add_argument("--scene-authority", type=Path)
    reproduce.add_argument("--scratch", type=Path, required=True)
    reproduce.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        path = assemble_full_corpus(
            arguments.lane,
            arguments.output,
            inventory=arguments.inventory,
            split_ledger=arguments.split_ledger,
            scene_authority=arguments.scene_authority,
        )
        corpus = load_full_corpus(path)
        result = {
            "command": "build",
            "path": str(path.resolve()),
            "manifest_sha256": corpus.manifest_sha256,
            "rows": len(corpus.artifacts.positions),
            "ranges": len(corpus.artifacts.range_starts),
            "eligible_rows": int(np.count_nonzero(corpus.eligible_mask)),
        }
    elif arguments.command == "verify":
        path = verify_full_corpus(arguments.corpus, arguments.receipt)
        result = {
            "command": "verify",
            "path": str(path.resolve()),
            "receipt_sha256": sha256_file(path),
        }
    else:
        path = reproduce_full_corpus(
            arguments.corpus,
            arguments.lane,
            arguments.scratch,
            arguments.receipt,
            inventory=arguments.inventory,
            split_ledger=arguments.split_ledger,
            scene_authority=arguments.scene_authority,
        )
        result = {
            "command": "reproduce",
            "path": str(path.resolve()),
            "receipt_sha256": sha256_file(path),
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAMILIES",
    "FullWalkingCorpus",
    "assemble_full_corpus",
    "load_full_corpus",
    "main",
    "reproduce_full_corpus",
    "verify_full_corpus",
]
