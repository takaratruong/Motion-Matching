"""Build sealed, provenance-preserving terrain-PFNN NPZ datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Iterable, Mapping
import zipfile

import numpy as np

from mm_sonic.grail_terrain_source import (
    G1MujocoFK,
    load_grail_motion,
    resample_grail_motion,
)
from mm_sonic.terrain_oracle.canonical import (
    CanonicalTerrainMesh,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_oracle.contact import CanonicalMeshQuery, SoleGeometry
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.source_grail import _load_usd_mesh
from mm_sonic.terrain_pfnn.features import (
    PFNNTrainingWindow,
    build_clip_windows_with_audit,
)
from mm_sonic.terrain_pfnn.layout import (
    CONTACT_ORDER,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
    TRAJECTORY_TIMES_S,
)
from mm_sonic.terrain_pfnn.phase import reconstruct_heel_toe_contacts
from mm_sonic.terrain_pfnn.sources import (
    GrailSlopeRecord,
    discover_grail_slope_records,
    load_grail_source,
    load_lafan_source,
)
from mm_sonic.terrain_pfnn.splits import split_identity, terrain_identity


_SCHEMA = "mm-sonic-terrain-pfnn-dataset/v1"
_SPLITS = ("train", "validation", "test")
_TERRAIN_CLASSES = ("flat", "ascent", "descent", "transition")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ROOT_FIELDS = {
    "path", "license_id", "license_manifest_path", "license_manifest_sha256"
}
_SOURCE_RECORD_FIELDS = {
    "source_kind", "motion_sha256", "terrain_sha256", "license_id",
    "split_identity", "split",
}
_SHARD_FIELDS = {
    "x", "y", "phase", "clip_id", "split_identity", "terrain_class",
    "center_frame", "motion_sha256", "terrain_sha256",
}


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(root: Path, relative: object) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be a nonempty relative string")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes the dataset destination")
    return path


def source_root_records(
    grail_root: str | Path, lafan_root: str | Path
) -> dict[str, dict[str, str]]:
    """Validate and bind the two authoritative source-license manifests."""

    grail = Path(grail_root).expanduser().resolve()
    lafan = Path(lafan_root).expanduser().resolve()
    grail_manifest = grail / "README.md"
    lafan_manifest = lafan.parent / "source_manifest.json"
    if not grail.is_dir() or not grail_manifest.is_file():
        raise FileNotFoundError("GRAIL root and README.md are required")
    if not lafan.is_dir() or not lafan_manifest.is_file():
        raise FileNotFoundError("LAFAN g1 root and ../source_manifest.json are required")
    grail_text = grail_manifest.read_text(encoding="utf-8")
    if re.search(r"(?im)^license:\s*apache-2\.0\s*$", grail_text) is None:
        raise ValueError("GRAIL README.md must declare license: apache-2.0")
    try:
        lafan_record = json.loads(lafan_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("LAFAN source_manifest.json is invalid") from error
    if lafan_record.get("license") != "CC BY-NC-ND 4.0":
        raise ValueError("LAFAN manifest license must be CC BY-NC-ND 4.0")
    if lafan_record.get("source_fps") != 30.0:
        raise ValueError("LAFAN manifest source_fps must equal 30.0")
    files = lafan_record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("LAFAN manifest files must be a nonempty list")
    declared: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("LAFAN manifest file record is invalid")
        relative = item.get("path")
        digest = item.get("sha256")
        if type(relative) is not str or type(digest) is not str \
                or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("LAFAN manifest file path or digest is invalid")
        if relative in declared:
            raise ValueError(f"duplicate LAFAN manifest path: {relative}")
        declared[relative] = digest
    actual_paths = sorted(lafan.glob("*.csv"))
    actual_relatives = {path.relative_to(lafan.parent).as_posix() for path in actual_paths}
    declared_for_g1 = {path for path in declared if path.startswith("g1/")}
    if actual_relatives != declared_for_g1:
        raise ValueError("LAFAN manifest inventory does not match the g1 root")
    for path in actual_paths:
        relative = path.relative_to(lafan.parent).as_posix()
        if _sha256(path) != declared[relative]:
            raise ValueError(f"LAFAN manifest digest mismatch: {relative}")
    return {
        "grail": {
            "path": str(grail),
            "license_id": "Apache-2.0",
            "license_manifest_path": str(grail_manifest),
            "license_manifest_sha256": _sha256(grail_manifest),
        },
        "lafan": {
            "path": str(lafan),
            "license_id": "CC-BY-NC-ND-4.0",
            "license_manifest_path": str(lafan_manifest),
            "license_manifest_sha256": _sha256(lafan_manifest),
        },
    }


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    content = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, content)


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return output.getvalue()


def _atomic_validated_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    validate: Callable[[Path], object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_deterministic_npz_bytes(arrays))
            stream.flush()
            os.fsync(stream.fileno())
        validate(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_roots(source_roots: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    if not isinstance(source_roots, Mapping) or not source_roots:
        raise ValueError("source_roots must be a nonempty mapping")
    output: dict[str, dict[str, str]] = {}
    for name, raw in sorted(source_roots.items()):
        if type(name) is not str or not name or not isinstance(raw, Mapping):
            raise ValueError("source root names and records are invalid")
        if set(raw) != _SOURCE_ROOT_FIELDS:
            raise ValueError(f"source root {name!r} has invalid fields")
        record = {field: str(raw[field]) for field in sorted(_SOURCE_ROOT_FIELDS)}
        if not record["path"] or not record["license_id"] or not record["license_manifest_path"]:
            raise ValueError(f"source root {name!r} contains an empty field")
        if _SHA256_RE.fullmatch(record["license_manifest_sha256"]) is None:
            raise ValueError(f"source root {name!r} manifest digest is invalid")
        output[name] = record
    return output


def _validate_source_records(
    source_records: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(source_records, Mapping) or not source_records:
        raise ValueError("source_records must be a nonempty clip-id mapping")
    output: dict[str, dict[str, str]] = {}
    for clip_id, raw in sorted(source_records.items()):
        if type(clip_id) is not str or not clip_id or len(clip_id) > 128:
            raise ValueError("source record clip IDs must be nonempty and at most 128 characters")
        if not isinstance(raw, Mapping) or set(raw) != _SOURCE_RECORD_FIELDS:
            raise ValueError(f"source record {clip_id!r} has invalid fields")
        record = {field: str(raw[field]) for field in sorted(_SOURCE_RECORD_FIELDS)}
        if record["source_kind"] not in ("grail", "lafan"):
            raise ValueError(f"source record {clip_id!r} source_kind is invalid")
        if _SHA256_RE.fullmatch(record["motion_sha256"]) is None:
            raise ValueError(f"source record {clip_id!r} motion digest is invalid")
        if record["terrain_sha256"] and _SHA256_RE.fullmatch(record["terrain_sha256"]) is None:
            raise ValueError(f"source record {clip_id!r} terrain digest is invalid")
        if record["source_kind"] == "lafan" and record["terrain_sha256"]:
            raise ValueError("LAFAN terrain_sha256 must be the empty string")
        if not record["license_id"] or not record["split_identity"]:
            raise ValueError(f"source record {clip_id!r} contains an empty field")
        if len(record["split_identity"]) > 128:
            raise ValueError("split identities must be at most 128 characters")
        if record["split"] not in _SPLITS:
            raise ValueError(f"source record {clip_id!r} split is invalid")
        output[clip_id] = record
    return output


def _split_identities(records: Mapping[str, Mapping[str, str]]) -> dict[str, list[str]]:
    result = {split: [] for split in _SPLITS}
    assignment: dict[str, str] = {}
    for record in records.values():
        identity = record["split_identity"]
        split = record["split"]
        previous = assignment.setdefault(identity, split)
        if previous != split:
            raise ValueError(f"split identity {identity!r} appears in multiple splits")
    for identity, split in sorted(assignment.items()):
        result[split].append(identity)
    return result


def _source_set_payload(
    roots: Mapping[str, Mapping[str, str]],
    records: Mapping[str, Mapping[str, str]],
    identities: Mapping[str, list[str]],
    build_options: Mapping[str, object],
) -> dict[str, object]:
    digest_roots = {
        name: {key: value for key, value in record.items() if key != "path"}
        for name, record in sorted(roots.items())
    }
    return {
        "schema": _SCHEMA,
        "fps": 30.0,
        "input_size": INPUT_LAYOUT.size,
        "output_size": OUTPUT_LAYOUT.size,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "contact_order": list(CONTACT_ORDER),
        "source_roots": digest_roots,
        "source_records": dict(sorted(records.items())),
        "split_identities": {name: sorted(identities[name]) for name in _SPLITS},
        "build_options": dict(sorted(build_options.items())),
    }


def _window_arrays(windows: list[PFNNTrainingWindow]) -> dict[str, np.ndarray]:
    if not windows:
        raise ValueError("cannot write an empty shard")
    for window in windows:
        if len(window.clip_id) > 128:
            raise ValueError("clip_id exceeds the fixed <U128 shard contract")
        if len(window.split_identity) > 128:
            raise ValueError("split_identity exceeds the fixed <U128 shard contract")
        if window.terrain_class not in _TERRAIN_CLASSES:
            raise ValueError("terrain_class is invalid")
        if window.center_frame > np.iinfo(np.int32).max:
            raise ValueError("center_frame exceeds the int32 shard contract")
    return {
        "x": np.ascontiguousarray([window.x for window in windows], dtype=np.float32),
        "y": np.ascontiguousarray([window.y for window in windows], dtype=np.float32),
        "phase": np.asarray([window.phase for window in windows], dtype=np.float32),
        "clip_id": np.asarray([window.clip_id for window in windows], dtype="<U128"),
        "split_identity": np.asarray(
            [window.split_identity for window in windows], dtype="<U128"
        ),
        "terrain_class": np.asarray(
            [window.terrain_class for window in windows], dtype="<U10"
        ),
        "center_frame": np.asarray(
            [window.center_frame for window in windows], dtype=np.int32
        ),
        "motion_sha256": np.asarray(
            [window.motion_sha256 for window in windows], dtype="<U64"
        ),
        "terrain_sha256": np.asarray(
            [window.terrain_sha256 or "" for window in windows], dtype="<U64"
        ),
    }


def _validate_shard(path: Path, *, expected_count: int, expected_split: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _SHARD_FIELDS:
                raise ValueError("shard fields do not match the contract")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid shard {path}") from error
    expected = {
        "x": ((expected_count, INPUT_LAYOUT.size), np.dtype(np.float32)),
        "y": ((expected_count, OUTPUT_LAYOUT.size), np.dtype(np.float32)),
        "phase": ((expected_count,), np.dtype(np.float32)),
        "clip_id": ((expected_count,), np.dtype("<U128")),
        "split_identity": ((expected_count,), np.dtype("<U128")),
        "terrain_class": ((expected_count,), np.dtype("<U10")),
        "center_frame": ((expected_count,), np.dtype(np.int32)),
        "motion_sha256": ((expected_count,), np.dtype("<U64")),
        "terrain_sha256": ((expected_count,), np.dtype("<U64")),
    }
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"shard {path} field {name} has invalid shape or dtype")
    if not np.isfinite(arrays["x"]).all() or not np.isfinite(arrays["y"]).all() \
            or not np.isfinite(arrays["phase"]).all():
        raise ValueError(f"shard {path} contains nonfinite values")
    if not np.all(arrays["split_identity"] == expected_split):
        raise ValueError(f"shard {path} crosses split boundaries")
    if not np.isin(arrays["terrain_class"], _TERRAIN_CLASSES).all():
        raise ValueError(f"shard {path} contains an invalid terrain class")
    if not np.isin(arrays["y"][:, OUTPUT_LAYOUT["contact_logit"]], (0.0, 1.0)).all():
        raise ValueError(f"shard {path} contains nonbinary contacts")
    return arrays


def _write_shard(path: Path, windows: list[PFNNTrainingWindow], split: str) -> str:
    arrays = _window_arrays(windows)
    _atomic_validated_npz(
        path,
        arrays,
        lambda temporary: _validate_shard(
            temporary, expected_count=len(windows), expected_split=split
        ),
    )
    return _sha256(path)


def _normalization_arrays(
    root: Path, shard_records: Iterable[Mapping[str, object]],
    split_identities: Mapping[str, list[str]],
) -> dict[str, np.ndarray]:
    x_sum = np.zeros(INPUT_LAYOUT.size, dtype=np.float64)
    x_squared_sum = np.zeros(INPUT_LAYOUT.size, dtype=np.float64)
    y_sum = np.zeros(OUTPUT_LAYOUT.size, dtype=np.float64)
    y_squared_sum = np.zeros(OUTPUT_LAYOUT.size, dtype=np.float64)
    split_counts = np.zeros(3, dtype=np.int64)
    training_count = 0
    for record in sorted(shard_records, key=lambda item: str(item["path"])):
        split = str(record["split"])
        count = int(record["count"])
        split_counts[_SPLITS.index(split)] += count
        if split != "train":
            continue
        arrays = _validate_shard(
            root / str(record["path"]), expected_count=count, expected_split=split
        )
        x64 = np.asarray(arrays["x"], dtype=np.float64)
        y64 = np.asarray(arrays["y"], dtype=np.float64)
        x_sum += x64.sum(axis=0, dtype=np.float64)
        x_squared_sum += np.square(x64).sum(axis=0, dtype=np.float64)
        y_sum += y64.sum(axis=0, dtype=np.float64)
        y_squared_sum += np.square(y64).sum(axis=0, dtype=np.float64)
        training_count += count
    if training_count < 1:
        raise ValueError("at least one training window is required")
    x_mean64 = x_sum / training_count
    y_mean64 = y_sum / training_count
    x_var = np.maximum(x_squared_sum / training_count - np.square(x_mean64), 0.0)
    y_var = np.maximum(y_squared_sum / training_count - np.square(y_mean64), 0.0)
    x_std64 = np.sqrt(x_var)
    y_std64 = np.sqrt(y_var)
    x_std64[x_std64 < 1.0e-6] = 1.0
    y_std64[y_std64 < 1.0e-6] = 1.0
    contact = OUTPUT_LAYOUT["contact_logit"]
    y_mean64[contact] = 0.0
    y_std64[contact] = 1.0
    return {
        "x_mean": x_mean64.astype(np.float32),
        "x_std": x_std64.astype(np.float32),
        "y_mean": y_mean64.astype(np.float32),
        "y_std": y_std64.astype(np.float32),
        "training_sample_count": np.asarray(training_count, dtype=np.int64),
        "split_counts": split_counts,
        "split_order": np.asarray(_SPLITS, dtype="<U10"),
        "split_identity_digest": np.asarray(
            canonical_json_sha256(
                {name: sorted(split_identities[name]) for name in _SPLITS}
            ),
            dtype="<U64",
        ),
    }


def _validate_normalization(path: Path) -> None:
    expected = {
        "x_mean": ((INPUT_LAYOUT.size,), np.dtype(np.float32)),
        "x_std": ((INPUT_LAYOUT.size,), np.dtype(np.float32)),
        "y_mean": ((OUTPUT_LAYOUT.size,), np.dtype(np.float32)),
        "y_std": ((OUTPUT_LAYOUT.size,), np.dtype(np.float32)),
        "training_sample_count": ((), np.dtype(np.int64)),
        "split_counts": ((3,), np.dtype(np.int64)),
        "split_order": ((3,), np.dtype("<U10")),
        "split_identity_digest": ((), np.dtype("<U64")),
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(expected):
                raise ValueError("normalization fields are invalid")
            for name, (shape, dtype) in expected.items():
                value = np.asarray(data[name])
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError(f"normalization field {name} is invalid")
            for name in ("x_mean", "x_std", "y_mean", "y_std"):
                if not np.isfinite(data[name]).all():
                    raise ValueError(f"normalization field {name} is nonfinite")
            if np.any(data["x_std"] <= 0.0) or np.any(data["y_std"] <= 0.0):
                raise ValueError("normalization standard deviations must be positive")
            np.testing.assert_array_equal(data["y_mean"][OUTPUT_LAYOUT["contact_logit"]], 0.0)
            np.testing.assert_array_equal(data["y_std"][OUTPUT_LAYOUT["contact_logit"]], 1.0)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        raise ValueError(f"invalid normalization archive {path}") from error


def _validate_resume_files(root: Path, manifest: Mapping[str, object]) -> None:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("resume manifest shards are invalid")
    seen_paths: set[Path] = set()
    for record in shards:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "count", "split"}:
            raise ValueError("resume manifest shard record is invalid")
        if (
            record["split"] not in _SPLITS
            or type(record["count"]) is not int
            or record["count"] < 1
            or type(record["sha256"]) is not str
            or _SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise ValueError("resume manifest shard split/count/digest is invalid")
        path = _artifact_path(root, record["path"])
        if path in seen_paths:
            raise ValueError("resume manifest contains a duplicate shard path")
        seen_paths.add(path)
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"resume shard hash mismatch: {path}")
        _validate_shard(path, expected_count=int(record["count"]), expected_split=str(record["split"]))
    normalization = manifest.get("normalization")
    if normalization is not None:
        if not isinstance(normalization, Mapping) or set(normalization) != {"path", "sha256"}:
            raise ValueError("resume normalization record is invalid")
        path = _artifact_path(root, normalization["path"])
        if not path.is_file() or _sha256(path) != normalization["sha256"]:
            raise ValueError("resume normalization hash mismatch")
        _validate_normalization(path)


def _validate_resume_manifest_source(manifest: Mapping[str, object]) -> str:
    try:
        roots = _validate_roots(manifest["source_roots"])
        records = _validate_source_records(manifest["source_records"])
        identities = _split_identities(records)
        if manifest["split_identities"] != identities:
            raise ValueError("resume split identities are stale")
        options = manifest["build_options"]
        if not isinstance(options, Mapping):
            raise ValueError("resume build options are invalid")
        digest = canonical_json_sha256(
            _source_set_payload(roots, records, identities, options)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resume manifest source provenance is invalid") from error
    if manifest.get("source_set_digest_sha256") != digest:
        raise ValueError("resume manifest source set digest is invalid")
    return digest


def _validated_rejections(values: Mapping[str, int]) -> dict[str, int]:
    output: dict[str, int] = {}
    for reason, count in sorted(values.items()):
        if type(reason) is not str or not reason or type(count) is not int or count < 0:
            raise ValueError(
                "rejection counts must map nonempty reasons to nonnegative integers"
            )
        output[reason] = count
    return output


def write_pfnn_dataset(
    output: str | Path,
    windows: Iterable[PFNNTrainingWindow],
    *,
    source_roots: Mapping[str, Mapping[str, str]],
    source_records: Mapping[str, Mapping[str, str]],
    rejection_counts: Mapping[str, int],
    build_options: Mapping[str, object] | None = None,
    max_windows_per_shard: int = 50_000,
    resume: bool = False,
) -> dict[str, object]:
    """Write a deterministic sealed dataset without exceeding one shard buffer."""

    if type(max_windows_per_shard) is not int or not 1 <= max_windows_per_shard <= 50_000:
        raise ValueError("max_windows_per_shard must be in [1, 50000]")
    root = Path(output).expanduser().resolve()
    roots = _validate_roots(source_roots)
    records = _validate_source_records(source_records)
    identities = _split_identities(records)
    options = dict(build_options or {})
    options["max_windows_per_shard"] = max_windows_per_shard
    source_set_digest = canonical_json_sha256(
        _source_set_payload(roots, records, identities, options)
    )
    rejections = _validated_rejections(rejection_counts)

    manifest_path = root / "manifest.json"
    existing: dict[str, object] | None = None
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing to overwrite nonempty destination: {root}")
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("resume requires a valid manifest") from error
        _validate_resume_manifest_source(existing)
        if existing.get("source_set_digest_sha256") != source_set_digest:
            raise ValueError("resume source set digest mismatch")
        _validate_resume_files(root, existing)
        if existing.get("status") == "accepted":
            normalization = existing.get("normalization")
            shards = existing.get("shards")
            if not isinstance(normalization, Mapping) or not isinstance(shards, list):
                raise ValueError("accepted resume manifest is incomplete")
            dataset_payload = {
                "source_set_digest_sha256": source_set_digest,
                "shards": sorted(shards, key=lambda item: str(item["path"])),
                "normalization_sha256": normalization["sha256"],
            }
            if existing.get("dataset_digest_sha256") != canonical_json_sha256(dataset_payload):
                raise ValueError("resume dataset digest mismatch")
            return existing
    elif resume:
        raise ValueError("resume requires a nonempty destination")
    else:
        root.mkdir(parents=True, exist_ok=True)

    shard_records = list(existing.get("shards", [])) if existing else []
    completed_keys: set[tuple[str, int, bytes]] = set()
    next_index = {split: 0 for split in _SPLITS}
    for record in shard_records:
        split = str(record["split"])
        name = Path(str(record["path"])).stem
        try:
            next_index[split] = max(next_index[split], int(name.rsplit("_", 1)[1]) + 1)
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError("resume shard path is invalid") from error
        arrays = _validate_shard(
            _artifact_path(root, record["path"]),
            expected_count=int(record["count"]), expected_split=split,
        )
        for clip_id, frame, phase in zip(
            arrays["clip_id"], arrays["center_frame"], arrays["phase"]
        ):
            key = (str(clip_id), int(frame), np.float32(phase).tobytes())
            if key in completed_keys:
                raise ValueError("resume dataset contains duplicate windows")
            completed_keys.add(key)

    base_manifest: dict[str, object] = {
        "schema": _SCHEMA,
        "status": "building",
        "fps": 30.0,
        "input_size": INPUT_LAYOUT.size,
        "output_size": OUTPUT_LAYOUT.size,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "contact_order": list(CONTACT_ORDER),
        "source_roots": roots,
        "source_records": records,
        "split_identities": identities,
        "shards": shard_records,
        "rejections": rejections,
        "build_options": options,
        "source_set_digest_sha256": source_set_digest,
    }
    _atomic_json(manifest_path, base_manifest)
    active_split: str | None = None
    buffer: list[PFNNTrainingWindow] = []
    seen_keys = set(completed_keys)

    def flush() -> None:
        nonlocal buffer
        if not buffer or active_split is None:
            return
        relative = f"{active_split}/shard_{next_index[active_split]:05d}.npz"
        path = root / relative
        if path.exists():
            raise ValueError(f"resume would overwrite an unlisted shard: {path}")
        digest = _write_shard(path, buffer, active_split)
        shard_records.append(
            {
                "path": relative,
                "sha256": digest,
                "count": len(buffer),
                "split": active_split,
            }
        )
        next_index[active_split] += 1
        base_manifest["shards"] = shard_records
        _atomic_json(manifest_path, base_manifest)
        buffer = []

    for window in windows:
        if not isinstance(window, PFNNTrainingWindow):
            raise TypeError("windows must contain PFNNTrainingWindow values")
        record = records.get(window.clip_id)
        if record is None:
            raise ValueError(f"window clip {window.clip_id!r} has no source record")
        if (
            record["motion_sha256"] != window.motion_sha256
            or record["terrain_sha256"] != (window.terrain_sha256 or "")
            or record["split"] != window.split_identity
        ):
            raise ValueError(
                f"window provenance conflicts with source record {window.clip_id!r}"
            )
        key = (
            window.clip_id,
            window.center_frame,
            np.float32(window.phase).tobytes(),
        )
        if key in completed_keys:
            continue
        if key in seen_keys:
            raise ValueError("input contains duplicate windows")
        seen_keys.add(key)
        if active_split is not None and window.split_identity != active_split:
            flush()
        active_split = window.split_identity
        buffer.append(window)
        if len(buffer) == max_windows_per_shard:
            flush()
    flush()
    base_manifest["rejections"] = _validated_rejections(rejection_counts)

    normalization_path = root / "normalization.npz"
    normalization_arrays = _normalization_arrays(root, shard_records, identities)
    _atomic_validated_npz(
        normalization_path, normalization_arrays, _validate_normalization
    )
    normalization_record = {
        "path": "normalization.npz", "sha256": _sha256(normalization_path)
    }
    dataset_payload = {
        "source_set_digest_sha256": source_set_digest,
        "shards": sorted(shard_records, key=lambda item: str(item["path"])),
        "normalization_sha256": normalization_record["sha256"],
    }
    base_manifest.update(
        {
            "status": "accepted",
            "normalization": normalization_record,
            "dataset_digest_sha256": canonical_json_sha256(dataset_payload),
        }
    )
    _atomic_json(manifest_path, base_manifest)
    return base_manifest


def _terrain_query(record: GrailSlopeRecord) -> CanonicalMeshQuery:
    terrain_digest = _sha256(record.terrain_path)
    mesh = _load_usd_mesh(
        record.terrain_path, source_asset_sha256=terrain_digest
    )
    return CanonicalMeshQuery(
        mesh,
        RigidTransform(
            record.terrain_position_world,
            record.terrain_quaternion_world_from_usd_wxyz,
        ),
    )


def _height_at(query: CanonicalMeshQuery, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    shape = points.shape[:-1]
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("height query expects finite [...,2] points")
    flattened = points.reshape((-1, 2))
    triangles = np.asarray(query._triangles, dtype=np.float64)
    normals = np.asarray(query._normals, dtype=np.float64)
    upward = triangles[normals[:, 2] > 1.0e-8]
    if not len(upward):
        return np.full(shape, np.nan, dtype=np.float64)
    height = np.full(len(flattened), np.nan, dtype=np.float64)
    best_distance = np.full(len(flattened), np.inf, dtype=np.float64)
    for triangle in upward:
        xy_triangle = triangle[:, :2]
        a, b, c = xy_triangle
        edge0 = b - a
        edge1 = c - a
        dot00 = float(np.dot(edge0, edge0))
        dot01 = float(np.dot(edge0, edge1))
        dot11 = float(np.dot(edge1, edge1))
        denominator = dot00 * dot11 - dot01 * dot01
        candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if abs(denominator) > 1.0e-14:
            relative = flattened - a
            dot20 = relative @ edge0
            dot21 = relative @ edge1
            bary_b = (dot11 * dot20 - dot01 * dot21) / denominator
            bary_c = (dot00 * dot21 - dot01 * dot20) / denominator
            bary_a = 1.0 - bary_b - bary_c
            inside = (
                (bary_a >= -1.0e-12)
                & (bary_b >= -1.0e-12)
                & (bary_c >= -1.0e-12)
            )
            inside_height = (
                bary_a * triangle[0, 2]
                + bary_b * triangle[1, 2]
                + bary_c * triangle[2, 2]
            )
            candidates.append(
                (np.zeros(len(flattened), dtype=np.float64), inside_height, inside)
            )
        for start, stop in ((0, 1), (1, 2), (2, 0)):
            direction = xy_triangle[stop] - xy_triangle[start]
            length_squared = float(np.dot(direction, direction))
            if length_squared <= 1.0e-14:
                continue
            fraction = np.clip(
                ((flattened - xy_triangle[start]) @ direction) / length_squared,
                0.0,
                1.0,
            )
            closest = xy_triangle[start] + fraction[:, None] * direction
            distance = np.sum(np.square(flattened - closest), axis=1)
            edge_height = (
                triangle[start, 2]
                + fraction * (triangle[stop, 2] - triangle[start, 2])
            )
            candidates.append(
                (distance, edge_height, np.ones(len(flattened), dtype=bool))
            )
        for distance, candidate_height, valid in candidates:
            closer = valid & (distance < best_distance - 1.0e-12)
            tied_higher = (
                valid
                & (np.abs(distance - best_distance) <= 1.0e-12)
                & (candidate_height > height)
            )
            update = closer | tied_higher
            best_distance[update] = distance[update]
            height[update] = candidate_height[update]
    return height.reshape(shape)


def _flat_query() -> CanonicalMeshQuery:
    extent = 1.0e6
    mesh = CanonicalTerrainMesh(
        vertices_local=np.asarray(
            ((-extent, -extent, 0.0), (extent, -extent, 0.0),
             (extent, extent, 0.0), (-extent, extent, 0.0)),
            dtype=np.float32,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        valid_faces=np.ones(2, dtype=bool),
        source_asset_sha256="0" * 64,
    )
    return CanonicalMeshQuery(
        mesh,
        RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        ),
    )


def _traversed_grade(record: GrailSlopeRecord) -> float | None:
    """Measure the maximum finite grade actually traversed by one variant."""

    raw = load_grail_motion(record.robot_path, expected_frames=record.expected_frames)
    resampled = resample_grail_motion(
        raw.root_position,
        raw.root_quaternion_xyzw,
        raw.dof_mujoco,
        source_fps=25.0,
        target_fps=30.0,
    )
    query = _terrain_query(record)
    points = np.asarray(resampled.root_position[:, :2], dtype=np.float64)
    height = _height_at(query, points)
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    delta = np.abs(np.diff(height))
    usable = (distance > 1.0e-6) & np.isfinite(delta)
    if not np.any(usable):
        return None
    nonflat = delta[usable] / distance[usable] > np.tan(np.deg2rad(1.0))
    if not np.any(nonflat):
        return 0.0
    grades = np.degrees(np.arctan2(delta[usable][nonflat], distance[usable][nonflat]))
    return float(np.max(grades))


def _select_families(
    records: tuple[GrailSlopeRecord, ...], limit: int | None
) -> tuple[tuple[GrailSlopeRecord, ...], dict[str, object]]:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("terrain_family_limit must be a positive integer")
    grouped: dict[str, list[GrailSlopeRecord]] = {}
    for record in records:
        grouped.setdefault(record.terrain_id, []).append(record)
    selected: list[str] = []
    grades: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for identity in sorted(grouped):
        if limit is not None and len(selected) >= limit:
            skipped[identity] = "family_limit"
            continue
        measured: list[float] = []
        for record in sorted(grouped[identity], key=lambda item: item.stem):
            try:
                grade = _traversed_grade(record)
            except (ArithmeticError, OSError, TypeError, ValueError):
                continue
            if grade is not None:
                measured.append(grade)
        if not measured:
            skipped[identity] = "grade_unmeasurable"
            continue
        grade = max(measured)
        grades[identity] = grade
        if grade < 5.0 - 1.0e-6:
            skipped[identity] = "grade_below_5_degrees"
            continue
        if grade > 20.0 + 1.0e-6:
            skipped[identity] = "grade_above_20_degrees"
            continue
        selected.append(identity)
    selected_set = set(selected)
    chosen = tuple(
        record for record in records if record.terrain_id in selected_set
    )
    return chosen, {
        "selected_identities": selected,
        "selected_grades_degrees": grades,
        "skipped_identities": skipped,
    }


def _inventory_source_records(
    grail_records: Iterable[GrailSlopeRecord], lafan_paths: Iterable[Path]
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}

    def add(clip_id: str, record: dict[str, str]) -> None:
        if clip_id in output:
            raise ValueError(f"duplicate or conflicting clip ID: {clip_id}")
        output[clip_id] = record

    for item in grail_records:
        identity = terrain_identity(item.stem)
        add(
            item.stem,
            {
                "source_kind": "grail",
                "motion_sha256": _sha256(item.robot_path),
                "terrain_sha256": _sha256(item.terrain_path),
                "license_id": "Apache-2.0",
                "split_identity": identity,
                "split": split_identity(identity),
            },
        )
    for path in lafan_paths:
        clip_id = path.stem
        identity = terrain_identity(clip_id)
        add(
            clip_id,
            {
                "source_kind": "lafan",
                "motion_sha256": _sha256(path),
                "terrain_sha256": "",
                "license_id": "CC-BY-NC-ND-4.0",
                "split_identity": identity,
                "split": split_identity(identity),
            },
        )
    return output


def _window_stream(
    grail_records: tuple[GrailSlopeRecord, ...],
    lafan_paths: tuple[Path, ...],
    fk: G1MujocoFK,
    geometry: SoleGeometry,
    rejections: Counter[str],
) -> Iterable[PFNNTrainingWindow]:
    grail_accepted = 0
    for split in _SPLITS:
        for record in grail_records:
            if split_identity(record.terrain_id) != split:
                continue
            source = load_grail_source(record, fk)
            query = _terrain_query(record)
            track = reconstruct_heel_toe_contacts(source, query, geometry)
            result = build_clip_windows_with_audit(
                source, track, height_at=lambda xy, q=query: _height_at(q, xy)
            )
            rejections.update(item.reason for item in result.rejections)
            grail_accepted += len(result.windows)
            yield from result.windows
    if grail_records and grail_accepted == 0:
        raise ValueError("selected GRAIL families yielded zero accepted GRAIL windows")

    flat_query = _flat_query()
    for split in _SPLITS:
        for path in lafan_paths:
            if split_identity(path.stem) != split:
                continue
            source = load_lafan_source(path, fk)
            track = reconstruct_heel_toe_contacts(source, flat_query, geometry)
            result = build_clip_windows_with_audit(
                source, track, height_at=lambda xy: np.zeros(np.asarray(xy).shape[:-1])
            )
            rejections.update(item.reason for item in result.rejections)
            yield from result.windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grail-root", type=Path, required=True)
    parser.add_argument("--lafan-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terrain-family-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    grail_root = arguments.grail_root.expanduser().resolve()
    lafan_root = arguments.lafan_root.expanduser().resolve()
    model_path = arguments.model_path.expanduser().resolve()
    if model_path.name != "g1_29dof.xml" or not model_path.is_file():
        raise ValueError("model-path must be the existing g1_29dof.xml")
    roots = source_root_records(grail_root, lafan_root)
    discovered = discover_grail_slope_records(grail_root)
    selected, selection = _select_families(
        discovered, arguments.terrain_family_limit
    )
    if arguments.terrain_family_limit is not None \
            and len(selection["selected_identities"]) != arguments.terrain_family_limit:
        raise ValueError("not enough eligible GRAIL terrain families")
    lafan_paths = tuple(sorted(lafan_root.glob("*.csv")))
    source_records = _inventory_source_records(selected, lafan_paths)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    geometry = SoleGeometry.from_model(model)
    fk = G1MujocoFK(model_path)
    rejection_counts: Counter[str] = Counter(
        {
            "invalid_phase": 0,
            "trajectory_knot_outside_clip": 0,
            "terrain_ray_missing": 0,
            "nonfinite_packed_value": 0,
            "grade_unmeasurable": sum(
                reason == "grade_unmeasurable"
                for reason in selection["skipped_identities"].values()
            ),
            "grade_below_5_degrees": sum(
                reason == "grade_below_5_degrees"
                for reason in selection["skipped_identities"].values()
            ),
            "grade_above_20_degrees": sum(
                reason == "grade_above_20_degrees"
                for reason in selection["skipped_identities"].values()
            ),
        }
    )
    build_options = {
        "terrain_family_limit": arguments.terrain_family_limit,
        "terrain_family_selection": selection,
        "kinematic_model_filename": model_path.name,
        "kinematic_model_sha256": _sha256(model_path),
    }
    manifest = write_pfnn_dataset(
        arguments.output,
        _window_stream(selected, lafan_paths, fk, geometry, rejection_counts),
        source_roots=roots,
        source_records=source_records,
        rejection_counts=rejection_counts,
        build_options=build_options,
        resume=arguments.resume,
    )
    print(json.dumps(
        {
            "status": manifest["status"],
            "selected_identities": selection["selected_identities"],
            "shards": len(manifest["shards"]),
            "samples": sum(item["count"] for item in manifest["shards"]),
            "rejections": manifest["rejections"],
        },
        sort_keys=True,
    ))
    return 0


__all__ = ["canonical_json_sha256", "source_root_records", "write_pfnn_dataset"]


if __name__ == "__main__":
    raise SystemExit(main())
