"""Authenticated primary + PFNN cache for the overnight hybrid terrain LMM.

The cache deliberately does not copy the multi-gigabyte Holden database.  It
stores only jointly normalized matching features and row/range metadata, then
rehashes and reopens both immutable input authorities when materializing a
normal :class:`HybridCorpus` for the existing trainer and runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mm_sonic.hybrid_terrain_lmm_data import (
    FAMILY_NAMES,
    FPS,
    HORIZONS,
    HybridCorpus,
    _canonical_json_bytes,
    _fsync_directory,
    _row_lookup,
    _sha256,
    _validate_array,
    _write_npy,
    load_hybrid_cache,
)
from mm_sonic.hybrid_terrain_lmm_pfnn import (
    SCHEMA as PFNN_SCHEMA,
)
from mm_sonic.hybrid_terrain_lmm_pfnn import (
    PFNNSupplement,
    load_pfnn_supplement,
)
from resources.g1_terrain_builder.features import _normalize
from resources.g1_terrain_builder.schema import ArtifactSet, FeatureSet

SCHEMA = "g1-hybrid-terrain-lmm-combined-cache/v2-strict"
_RECEIPT_SCHEMA = "g1-hybrid-terrain-lmm-combined-receipt/v2-strict"
_PRIMARY_SCHEMA = "g1-hybrid-terrain-lmm-corpus/v2-strict"
_NORMALIZATION_CONTRACT = {
    "evaluation_rows_in_fit": 0,
    "fit_rows": "combined-train-mask",
    "input": "denormalized-authority-31d-features",
    "method": "joint-group-mean-and-mean-standard-deviation",
    "transform_rows": "all-combined-rows",
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


def _manifest_descriptor(root: Path, label: str) -> dict[str, object]:
    candidate = Path(root).expanduser().resolve(strict=True)
    manifest = candidate if candidate.is_file() else candidate / "manifest.json"
    manifest = manifest.resolve(strict=True)
    try:
        value = json.loads(manifest.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} authority manifest is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} authority manifest must be an object")
    return {
        "path": str(manifest),
        "size_bytes": manifest.stat().st_size,
        "sha256": _sha256(manifest),
        "schema": value.get("schema"),
    }


def _authenticate_authority(
    descriptor: object,
    *,
    expected_schema: str,
    label: str,
) -> Path:
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
        "schema",
    }:
        raise ValueError(f"{label} authority descriptor is invalid")
    path_value = descriptor["path"]
    if type(path_value) is not str or not Path(path_value).is_absolute():
        raise ValueError(f"{label} authority path must be absolute")
    try:
        path = Path(path_value).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} authority manifest is missing") from error
    if (
        not path.is_file()
        or path.name != "manifest.json"
        or descriptor["schema"] != expected_schema
        or type(descriptor["size_bytes"]) is not int
        or path.stat().st_size != descriptor["size_bytes"]
        or type(descriptor["sha256"]) is not str
        or _sha256(path) != descriptor["sha256"]
    ):
        raise ValueError(f"{label} authority authentication failed")
    return path.parent


def _denormalize(features: FeatureSet) -> np.ndarray:
    features.validate()
    raw = (
        features.values.astype(np.float64)
        * features.scale.astype(np.float64)
        + features.offset.astype(np.float64)
    )
    if not np.isfinite(raw).all():
        raise ValueError("denormalized matching features are nonfinite")
    return np.ascontiguousarray(raw, dtype=np.float32)


def _joint_features(
    primary: FeatureSet,
    pfnn: FeatureSet,
    train_mask: np.ndarray,
) -> FeatureSet:
    raw = np.concatenate((_denormalize(primary), _denormalize(pfnn)), axis=0)
    return _normalize(raw, np.asarray(train_mask, dtype=np.bool_))


def _pfnn_range_families(supplement: PFNNSupplement) -> np.ndarray:
    records = supplement.receipt.get("ranges")
    starts = np.asarray(supplement.artifacts.range_starts)
    stops = np.asarray(supplement.artifacts.range_stops)
    if type(records) is not list or len(records) != len(starts):
        raise ValueError("PFNN receipt ranges differ from its ArtifactSet")
    family_ids = np.empty(len(records), np.int16)
    for index, (record, start, stop) in enumerate(zip(records, starts, stops)):
        if type(record) is not dict:
            raise ValueError("PFNN range receipt is invalid")
        family = record.get("family")
        contact_digest = record.get("contact_source_rows_sha256")
        if (
            record.get("role") != "train"
            or family not in ("flat", "stair")
            or record.get("start") != int(start)
            or record.get("stop") != int(stop)
            or type(contact_digest) is not str
            or len(contact_digest) != 64
            or any(value not in "0123456789abcdef" for value in contact_digest)
        ):
            raise ValueError("PFNN range-local train receipt is invalid")
        family_ids[index] = FAMILY_NAMES.index(str(family))
    selected_rows = supplement.receipt.get("counts", {}).get(
        "selected_output_rows_25hz"
    )
    if selected_rows != len(supplement.artifacts.positions):
        raise ValueError("PFNN admitted row count differs from its receipt")
    return family_ids


def _combined_arrays(
    primary: HybridCorpus,
    supplement: PFNNSupplement,
) -> dict[str, np.ndarray]:
    primary.validate()
    pfnn_families = _pfnn_range_families(supplement)
    primary_rows = len(primary.artifacts.positions)
    pfnn_rows = len(supplement.artifacts.positions)
    pfnn_ranges = len(supplement.artifacts.range_starts)
    range_starts = np.concatenate(
        (
            np.asarray(primary.artifacts.range_starts, np.int32),
            (
                np.asarray(supplement.artifacts.range_starts, np.int64)
                + primary_rows
            ).astype(np.int32),
        )
    )
    range_stops = np.concatenate(
        (
            np.asarray(primary.artifacts.range_stops, np.int32),
            (
                np.asarray(supplement.artifacts.range_stops, np.int64)
                + primary_rows
            ).astype(np.int32),
        )
    )
    range_family_ids = np.concatenate(
        (np.asarray(primary.range_family_ids, np.int16), pfnn_families)
    )
    total_rows = primary_rows + pfnn_rows
    range_ids, family_ids = _row_lookup(
        range_starts, range_stops, range_family_ids, total_rows
    )
    train_mask = np.concatenate(
        (np.asarray(primary.train_mask, np.bool_), np.ones(pfnn_rows, np.bool_))
    )
    evaluation_mask = np.concatenate(
        (
            np.asarray(primary.evaluation_mask, np.bool_),
            np.zeros(pfnn_rows, np.bool_),
        )
    )
    features = _joint_features(
        primary.features,
        supplement.features,
        train_mask,
    )
    if len(range_starts) != len(primary.artifacts.range_starts) + pfnn_ranges:
        raise AssertionError("combined range count changed during assembly")
    return {
        "features.npy": features.values,
        "feature_offset.npy": features.offset,
        "feature_scale.npy": features.scale,
        "range_family_ids.npy": range_family_ids,
        "family_ids.npy": family_ids,
        "range_ids.npy": range_ids,
        "range_starts.npy": range_starts,
        "range_stops.npy": range_stops,
        "train_mask.npy": train_mask,
        "evaluation_mask.npy": evaluation_mask,
    }


def _family_counts(family_ids: np.ndarray) -> dict[str, int]:
    return {
        family: int(np.count_nonzero(family_ids == index))
        for index, family in enumerate(FAMILY_NAMES)
    }


def _publish(
    *,
    output: Path,
    arrays: Mapping[str, np.ndarray],
    authorities: Mapping[str, Mapping[str, object]],
    primary: HybridCorpus,
    supplement: PFNNSupplement,
) -> Path:
    destination = Path(output).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"combined cache already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent, prefix=f".{destination.name}.staging-"
        )
    )
    try:
        if set(arrays) != set(_ARRAY_SPECS):
            raise ValueError("combined cache array inventory is invalid")
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
        primary_rows = len(primary.artifacts.positions)
        pfnn_rows = len(supplement.artifacts.positions)
        train = np.asarray(arrays["train_mask.npy"])
        evaluation = np.asarray(arrays["evaluation_mask.npy"])
        family_ids = np.asarray(arrays["family_ids.npy"])
        manifest = {
            "schema": SCHEMA,
            "status": "accepted",
            "rows": primary_rows + pfnn_rows,
            "ranges": len(arrays["range_starts.npy"]),
            "fps": FPS,
            "horizons": list(HORIZONS),
            "feature_dimensions": 31,
            "family_names": list(FAMILY_NAMES),
            "authorities": {
                name: dict(descriptor)
                for name, descriptor in sorted(authorities.items())
            },
            "counts": {
                "primary_rows": primary_rows,
                "primary_ranges": len(primary.artifacts.range_starts),
                "primary_train_rows": int(np.count_nonzero(primary.train_mask)),
                "primary_evaluation_rows": int(
                    np.count_nonzero(primary.evaluation_mask)
                ),
                "pfnn_rows": pfnn_rows,
                "pfnn_ranges": len(supplement.artifacts.range_starts),
                "pfnn_train_rows": pfnn_rows,
                "pfnn_evaluation_rows": 0,
                "train_rows": int(np.count_nonzero(train)),
                "evaluation_rows": int(np.count_nonzero(evaluation)),
                "family_rows": _family_counts(family_ids),
            },
            "split": {
                "primary_masks_preserved": True,
                "pfnn_admitted_train_only": True,
                "pfnn_source_role": "train",
                "pfnn_contributes_to_source_held_out": False,
            },
            "normalization": dict(_NORMALIZATION_CONTRACT),
            "authentication": {
                "primary_cache_reopened": True,
                "pfnn_supplement_reopened": True,
                "pfnn_range_boundaries_verified": True,
                "pfnn_contacts_bound_by": "pfnn-supplement/database.bin",
                "pfnn_terrain_bound_by": [
                    "pfnn-supplement/terrain.bin",
                    "pfnn-supplement/support.bin",
                ],
            },
            "members": members,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_canonical_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        for name, (dtype, shape) in _ARRAY_SPECS.items():
            _validate_array(staging / name, dtype, shape)
        if json.loads(manifest_path.read_bytes()) != manifest:
            raise ValueError("combined staged manifest failed reopen verification")
        _fsync_directory(staging)
        if destination.exists():
            raise FileExistsError(f"combined cache already exists: {destination}")
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        return destination / "manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_combined_cache(
    primary_cache: Path,
    pfnn_supplement: Path,
    output: Path,
) -> Path:
    """Publish an immutable cache that references, but never modifies, both inputs."""

    primary_descriptor = _manifest_descriptor(primary_cache, "primary")
    pfnn_descriptor = _manifest_descriptor(pfnn_supplement, "PFNN")
    primary = load_hybrid_cache(
        Path(primary_cache),
        expected_cache_manifest_sha256=str(primary_descriptor["sha256"]),
    )
    supplement = load_pfnn_supplement(
        Path(pfnn_supplement),
        expected_manifest_sha256=str(pfnn_descriptor["sha256"]),
    )
    if primary.cache_manifest_sha256 != primary_descriptor["sha256"]:
        raise ValueError("primary authority changed while it was opened")
    if _manifest_descriptor(primary_cache, "primary") != primary_descriptor:
        raise ValueError("primary authority changed while it was opened")
    if _manifest_descriptor(pfnn_supplement, "PFNN") != pfnn_descriptor:
        raise ValueError("PFNN authority changed while it was opened")
    arrays = _combined_arrays(primary, supplement)
    return _publish(
        output=output,
        arrays=arrays,
        authorities={
            "primary_cache": primary_descriptor,
            "pfnn_supplement": pfnn_descriptor,
        },
        primary=primary,
        supplement=supplement,
    )


def _load_manifest(output: Path) -> tuple[dict[str, Any], str]:
    manifest_path = output / "manifest.json"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("combined cache manifest is invalid") from error
    if (
        type(manifest) is not dict
        or manifest.get("schema") != SCHEMA
        or manifest.get("status") != "accepted"
        or payload != _canonical_json_bytes(manifest)
    ):
        raise ValueError("combined cache manifest is not canonical accepted v2-strict")
    if set(manifest.get("members", {})) != set(_ARRAY_SPECS):
        raise ValueError("combined cache member inventory changed")
    expected_names = set(_ARRAY_SPECS) | {"manifest.json"}
    if {path.name for path in output.iterdir()} != expected_names:
        raise ValueError("combined cache contains missing or extra members")
    for name, descriptor in manifest["members"].items():
        path = output / name
        if (
            type(descriptor) is not dict
            or descriptor.get("path") != name
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"combined cache member authentication failed: {name}")
    return manifest, _sha256(manifest_path)


def _combine_artifacts(primary: ArtifactSet, pfnn: ArtifactSet) -> ArtifactSet:
    primary.validate()
    pfnn.validate()
    if not np.array_equal(primary.parents, pfnn.parents):
        raise ValueError("primary and PFNN skeleton parents differ")
    primary_rows = len(primary.positions)
    result = ArtifactSet(
        positions=np.concatenate((primary.positions, pfnn.positions), axis=0),
        velocities=np.concatenate((primary.velocities, pfnn.velocities), axis=0),
        rotations=np.concatenate((primary.rotations, pfnn.rotations), axis=0),
        angular_velocities=np.concatenate(
            (primary.angular_velocities, pfnn.angular_velocities), axis=0
        ),
        parents=np.asarray(primary.parents, np.int32).copy(),
        range_starts=np.concatenate(
            (
                np.asarray(primary.range_starts, np.int32),
                (
                    np.asarray(pfnn.range_starts, np.int64) + primary_rows
                ).astype(np.int32),
            )
        ),
        range_stops=np.concatenate(
            (
                np.asarray(primary.range_stops, np.int32),
                (np.asarray(pfnn.range_stops, np.int64) + primary_rows).astype(
                    np.int32
                ),
            )
        ),
        contacts=np.concatenate((primary.contacts, pfnn.contacts), axis=0),
        terrain_features=np.concatenate(
            (primary.terrain_features, pfnn.terrain_features), axis=0
        ),
        terrain_support=np.concatenate(
            (primary.terrain_support, pfnn.terrain_support), axis=0
        ),
    )
    result.validate()
    return result


def _validate_loaded_contract(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    primary: HybridCorpus,
    supplement: PFNNSupplement,
) -> None:
    expected = _combined_arrays(primary, supplement)
    for name in (
        "range_family_ids.npy",
        "family_ids.npy",
        "range_ids.npy",
        "range_starts.npy",
        "range_stops.npy",
        "train_mask.npy",
        "evaluation_mask.npy",
    ):
        if not np.array_equal(arrays[name], expected[name]):
            raise ValueError(f"combined {name} differs from its authorities")
    expected_features = FeatureSet(
        expected["features.npy"],
        expected["feature_offset.npy"],
        expected["feature_scale.npy"],
    )
    actual_features = FeatureSet(
        arrays["features.npy"],
        arrays["feature_offset.npy"],
        arrays["feature_scale.npy"],
    )
    if not all(
        np.array_equal(actual, wanted)
        for actual, wanted in (
            (actual_features.values, expected_features.values),
            (actual_features.offset, expected_features.offset),
            (actual_features.scale, expected_features.scale),
        )
    ):
        raise ValueError("combined joint normalization differs from its authorities")
    counts = manifest.get("counts")
    if type(counts) is not dict:
        raise ValueError("combined counts are missing")
    expected_counts = {
        "primary_rows": len(primary.artifacts.positions),
        "primary_ranges": len(primary.artifacts.range_starts),
        "primary_train_rows": int(np.count_nonzero(primary.train_mask)),
        "primary_evaluation_rows": int(np.count_nonzero(primary.evaluation_mask)),
        "pfnn_rows": len(supplement.artifacts.positions),
        "pfnn_ranges": len(supplement.artifacts.range_starts),
        "pfnn_train_rows": len(supplement.artifacts.positions),
        "pfnn_evaluation_rows": 0,
        "train_rows": int(np.count_nonzero(arrays["train_mask.npy"])),
        "evaluation_rows": int(np.count_nonzero(arrays["evaluation_mask.npy"])),
        "family_rows": _family_counts(arrays["family_ids.npy"]),
    }
    if counts != expected_counts:
        raise ValueError("combined counts differ from its authorities")
    if manifest.get("normalization") != _NORMALIZATION_CONTRACT:
        raise ValueError("combined normalization contract changed")


def load_combined_cache(
    output: Path,
) -> HybridCorpus:
    """Authenticate both authorities and return a training/runtime HybridCorpus."""

    root = Path(output).expanduser().resolve(strict=True)
    manifest, manifest_sha256 = _load_manifest(root)
    authorities = manifest.get("authorities")
    if type(authorities) is not dict or set(authorities) != {
        "primary_cache",
        "pfnn_supplement",
    }:
        raise ValueError("combined authority inventory changed")
    primary_root = _authenticate_authority(
        authorities["primary_cache"],
        expected_schema=_PRIMARY_SCHEMA,
        label="primary",
    )
    pfnn_root = _authenticate_authority(
        authorities["pfnn_supplement"],
        expected_schema=PFNN_SCHEMA,
        label="PFNN",
    )

    # These loaders rehash every cache/supplement member before the combined
    # ArtifactSet below is allocated.  Primary raw source assets are also
    # rehashed by default.
    primary = load_hybrid_cache(
        primary_root,
        expected_cache_manifest_sha256=authorities["primary_cache"]["sha256"],
    )
    supplement = load_pfnn_supplement(
        pfnn_root,
        expected_manifest_sha256=str(authorities["pfnn_supplement"]["sha256"]),
    )
    if primary.cache_manifest_sha256 != authorities["primary_cache"]["sha256"]:
        raise ValueError("primary authority digest changed during load")
    try:
        pfnn_manifest = (pfnn_root / "manifest.json").resolve(strict=True)
        pfnn_size = pfnn_manifest.stat().st_size
        pfnn_digest = _sha256(pfnn_manifest)
    except OSError as error:
        raise ValueError("PFNN authority digest changed during load") from error
    if (
        pfnn_size != authorities["pfnn_supplement"]["size_bytes"]
        or pfnn_digest != authorities["pfnn_supplement"]["sha256"]
    ):
        raise ValueError("PFNN authority digest changed during load")
    _pfnn_range_families(supplement)

    for name, (dtype, shape) in _ARRAY_SPECS.items():
        _validate_array(root / name, dtype, shape)
    arrays = {
        name: np.load(root / name, mmap_mode="r", allow_pickle=False)
        for name in _ARRAY_SPECS
    }
    _validate_loaded_contract(manifest, arrays, primary, supplement)
    artifacts = _combine_artifacts(primary.artifacts, supplement.artifacts)
    features = FeatureSet(
        arrays["features.npy"],
        arrays["feature_offset.npy"],
        arrays["feature_scale.npy"],
    )
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "path": str(root / "manifest.json"),
        "size_bytes": (root / "manifest.json").stat().st_size,
        "sha256": manifest_sha256,
        "authorities": manifest["authorities"],
    }
    corpus = HybridCorpus(
        artifacts=artifacts,
        features=features,
        range_family_ids=arrays["range_family_ids.npy"],
        family_ids=arrays["family_ids.npy"],
        range_ids=arrays["range_ids.npy"],
        train_mask=arrays["train_mask.npy"],
        evaluation_mask=arrays["evaluation_mask.npy"],
        source_root=root,
        manifest_receipt=receipt,
        source_assets={},
        family_names=tuple(manifest["family_names"]),
        cache_manifest_sha256=manifest_sha256,
    )
    corpus.validate()
    if (
        manifest.get("rows") != len(artifacts.positions)
        or manifest.get("ranges") != len(artifacts.range_starts)
        or manifest.get("fps") != FPS
        or tuple(manifest.get("horizons", ())) != HORIZONS
        or manifest.get("feature_dimensions") != 31
        or tuple(manifest.get("family_names", ())) != FAMILY_NAMES
    ):
        raise ValueError("combined dimensions, rate, or family table changed")
    return corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--pfnn", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    manifest = build_combined_cache(
        arguments.primary, arguments.pfnn, arguments.output
    )
    payload = json.loads(manifest.read_bytes())
    print(
        json.dumps(
            {
                "status": "accepted",
                "manifest": str(manifest.resolve()),
                "manifest_sha256": _sha256(manifest),
                "rows": payload["rows"],
                "ranges": payload["ranges"],
                "counts": payload["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA",
    "build_combined_cache",
    "load_combined_cache",
]
