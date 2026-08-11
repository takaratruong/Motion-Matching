"""Canonical digest payloads shared by terrain-PFNN writers and readers."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES

from .layout import CONTACT_ORDER, INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S
from .splits import split_identity as sealed_split, terrain_identity


DATASET_SCHEMA = "mm-sonic-terrain-pfnn-dataset/v2"
JOINT_STATE_RECEIPT_SCHEMA = "g1-pfnn-joint-state-receipt/v1"
MAXIMUM_JOINT_STEP_RAD = 0.225
SPLITS = ("train", "validation", "test")


def canonical_joint_state_receipt(
    *,
    accepted_rows_by_source: Mapping[str, int],
    rejected_rows_by_source: Mapping[str, Mapping[str, int]],
    rejected_rows_by_joint: Mapping[str, int],
    state_source_by_clip: Mapping[str, str],
    migration_provenance: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate and canonicalize the exact q/qdot construction audit."""

    accepted = dict(sorted(accepted_rows_by_source.items()))
    modes = dict(sorted(state_source_by_clip.items()))
    rejected = {
        source: dict(sorted(counts.items()))
        for source, counts in sorted(rejected_rows_by_source.items())
    }
    by_joint = dict(sorted(rejected_rows_by_joint.items()))
    migration = dict(sorted((migration_provenance or {}).items()))
    if (
        not accepted
        or set(modes) != set(accepted) | set(rejected)
        or any(
            type(source) is not str
            or not source
            or type(count) is not int
            or count < 0
            for source, count in accepted.items()
        )
        or any(mode not in ("direct_source", "unique_predecessor") for mode in modes.values())
    ):
        raise ValueError("joint-state source receipt is invalid")
    for source, counts in rejected.items():
        if (
            type(source) is not str
            or not source
            or not counts
            or any(
                type(reason) is not str
                or not reason
                or type(count) is not int
                or count < 1
                for reason, count in counts.items()
            )
        ):
            raise ValueError("joint-state rejection receipt is invalid")
    if set(by_joint) != set(ISAACLAB_JOINT_NAMES) or any(
        type(count) is not int or count < 0 for count in by_joint.values()
    ):
        raise ValueError("joint-state per-joint rejection receipt is invalid")
    if migration and (
        set(migration) != {"source_dataset_sha256", "retarget_manifest_sha256"}
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in migration.values()
        )
    ):
        raise ValueError("joint-state migration provenance is invalid")
    return {
        "schema": JOINT_STATE_RECEIPT_SCHEMA,
        "maximum_joint_step_rad": MAXIMUM_JOINT_STEP_RAD,
        "accepted_rows_by_source": accepted,
        "rejected_rows_by_source": rejected,
        "rejected_rows_by_joint": by_joint,
        "state_source_by_clip": modes,
        "migration_provenance": migration,
    }


def validate_joint_state_receipt(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "maximum_joint_step_rad",
        "accepted_rows_by_source",
        "rejected_rows_by_source",
        "rejected_rows_by_joint",
        "state_source_by_clip",
        "migration_provenance",
    }:
        raise ValueError("joint-state receipt fields are invalid")
    if (
        value["schema"] != JOINT_STATE_RECEIPT_SCHEMA
        or value["maximum_joint_step_rad"] != MAXIMUM_JOINT_STEP_RAD
        or not isinstance(value["accepted_rows_by_source"], Mapping)
        or not isinstance(value["rejected_rows_by_source"], Mapping)
        or not isinstance(value["rejected_rows_by_joint"], Mapping)
        or not isinstance(value["state_source_by_clip"], Mapping)
        or not isinstance(value["migration_provenance"], Mapping)
    ):
        raise ValueError("joint-state receipt contract is invalid")
    return canonical_joint_state_receipt(
        accepted_rows_by_source=value["accepted_rows_by_source"],
        rejected_rows_by_source=value["rejected_rows_by_source"],
        rejected_rows_by_joint=value["rejected_rows_by_joint"],
        state_source_by_clip=value["state_source_by_clip"],
        migration_provenance=value["migration_provenance"],
    )


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_split_identities(
    split_identities: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    return {split: sorted(split_identities[split]) for split in SPLITS}


def split_identity_digest(
    split_identities: Mapping[str, list[str]],
) -> str:
    return canonical_json_sha256(canonical_split_identities(split_identities))


def source_set_payload(
    source_roots: Mapping[str, Mapping[str, str]],
    source_records: Mapping[str, Mapping[str, str]],
    split_identities: Mapping[str, list[str]],
    build_options: Mapping[str, object],
) -> dict[str, object]:
    """Return the portable payload; no machine-specific absolute path survives."""

    digest_roots = {
        name: {
            key: value
            for key, value in record.items()
            if key not in ("path", "license_manifest_path")
        }
        for name, record in sorted(source_roots.items())
    }
    return {
        "schema": DATASET_SCHEMA,
        "fps": 30.0,
        "input_size": INPUT_LAYOUT.size,
        "output_size": OUTPUT_LAYOUT.size,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "contact_order": list(CONTACT_ORDER),
        "source_roots": digest_roots,
        "source_records": dict(sorted(source_records.items())),
        "split_identities": canonical_split_identities(split_identities),
        "build_options": dict(sorted(build_options.items())),
    }


def validate_shard_row_provenance(
    arrays: Mapping[str, np.ndarray],
    *,
    split: str,
    source_records: Mapping[str, Mapping[str, str]],
) -> None:
    """Bind each immutable shard row to its sealed manifest source record."""

    if np.any(arrays["center_frame"] < 0):
        raise ValueError("shard center_frame must be nonnegative")

    for clip, identity, motion_hash, terrain_hash in zip(
        arrays["clip_id"],
        arrays["split_identity"],
        arrays["motion_sha256"],
        arrays["terrain_sha256"],
    ):
        clip_id = str(clip)
        record = source_records.get(clip_id)
        if record is None:
            raise ValueError(f"unknown shard clip_id: {clip_id}")
        canonical_identity = terrain_identity(clip_id)
        if (
            record.get("split_identity") != canonical_identity
            or record.get("split") != sealed_split(canonical_identity)
            or str(identity) != canonical_identity
            or record.get("split") != split
            or record.get("motion_sha256") != str(motion_hash)
            or record.get("terrain_sha256") != str(terrain_hash)
        ):
            raise ValueError(f"shard row provenance mismatch: {clip_id}")


def validate_normalization_metadata(
    arrays: Mapping[str, np.ndarray],
    *,
    shard_records: list[Mapping[str, object]],
    split_identities: Mapping[str, list[str]],
) -> None:
    counts = np.zeros(3, dtype=np.int64)
    for record in shard_records:
        split = record.get("split")
        count = record.get("count")
        if split not in SPLITS or type(count) is not int or count < 1:
            raise ValueError("normalization metadata has invalid shard counts")
        counts[SPLITS.index(split)] += count
    training_count = arrays["training_sample_count"]
    split_counts = arrays["split_counts"]
    split_order = arrays["split_order"]
    identity_digest = arrays["split_identity_digest"]
    if (
        training_count.shape != ()
        or training_count.dtype != np.dtype(np.int64)
        or int(training_count) <= 0
        or int(training_count) != int(counts[0])
        or split_counts.shape != (3,)
        or split_counts.dtype != np.dtype(np.int64)
        or not np.array_equal(split_counts, counts)
        or split_order.shape != (3,)
        or split_order.dtype != np.dtype("<U10")
        or tuple(map(str, split_order)) != SPLITS
        or identity_digest.shape != ()
        or identity_digest.dtype != np.dtype("<U64")
        or str(identity_digest) != split_identity_digest(split_identities)
    ):
        raise ValueError("normalization metadata does not match the manifest")


__all__ = [
    "DATASET_SCHEMA",
    "JOINT_STATE_RECEIPT_SCHEMA",
    "MAXIMUM_JOINT_STEP_RAD",
    "SPLITS",
    "canonical_json_sha256",
    "canonical_joint_state_receipt",
    "canonical_split_identities",
    "source_set_payload",
    "split_identity_digest",
    "validate_shard_row_provenance",
    "validate_normalization_metadata",
    "validate_joint_state_receipt",
]
