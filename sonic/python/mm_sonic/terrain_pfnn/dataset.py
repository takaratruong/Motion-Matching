"""Read-only normalized access to sealed terrain-PFNN NPZ shards."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
import re
from typing import Literal, Mapping

import numpy as np

from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES

from .layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    CONTACT_ORDER,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
    TRAJECTORY_TIMES_S,
)
from .features import (
    PFNNTrainingWindow,
    _JOINT_MIRROR,
    _JOINT_MIRROR_SIGN,
    mirror_window,
)
from .provenance import (
    DATASET_SCHEMA,
    SPLITS,
    canonical_json_sha256,
    source_set_payload,
    validate_normalization_metadata,
    validate_shard_row_provenance,
)
from .splits import split_identity as sealed_split, terrain_identity


SplitName = Literal["train", "validation", "test"]
_SPLITS = SPLITS
_SHARD_FIELDS = {
    "x", "y", "phase", "clip_id", "split_identity", "sequence_lane", "terrain_class",
    "center_frame", "motion_sha256", "terrain_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_PATH_RE = re.compile(
    r"^(train|validation|test)/shard_[0-9]{5}\.npz$"
)
_SOURCE_ROOT_FIELDS = {
    "path", "license_id", "license_manifest_path", "license_manifest_sha256"
}
_SOURCE_RECORD_FIELDS = {
    "source_kind", "motion_sha256", "terrain_sha256", "license_id",
    "split_identity", "split",
}
_TERRAIN_CLASSES = ("flat", "ascent", "descent", "transition")
_SEQUENCE_LANES = ("motion", *(f"idle_phase_{index}" for index in range(8)))

JOINT_MIRROR_PERMUTATION = np.asarray(_JOINT_MIRROR, dtype=np.int64).copy()
JOINT_MIRROR_PERMUTATION.flags.writeable = False
JOINT_MIRROR_SIGN = np.asarray(_JOINT_MIRROR_SIGN, dtype=np.float32).copy()
JOINT_MIRROR_SIGN.flags.writeable = False


def pack_classic_g1_input_v3(
    base_input: object,
    joint_position: object,
    joint_velocity: object,
) -> np.ndarray:
    """Append exact physical G1 joint state to one legacy 288-value input."""

    base = np.asarray(base_input, dtype=np.float32)
    position = np.asarray(joint_position, dtype=np.float32)
    velocity = np.asarray(joint_velocity, dtype=np.float32)
    if base.shape != (INPUT_LAYOUT.size,):
        raise ValueError("base_input must have shape (288,)")
    if position.shape != (29,) or velocity.shape != (29,):
        raise ValueError("joint position and velocity must have shape (29,)")
    if not all(np.isfinite(value).all() for value in (base, position, velocity)):
        raise ValueError("classic G1 v3 input values must be finite")
    output = np.empty(CLASSIC_G1_INPUT_LAYOUT_V3.size, dtype=np.float32)
    output[: INPUT_LAYOUT.size] = base
    output[CLASSIC_G1_INPUT_LAYOUT_V3["joint_position"]] = position
    output[CLASSIC_G1_INPUT_LAYOUT_V3["joint_velocity"]] = velocity
    return output


def mirror_classic_g1_joint_state(value: object) -> np.ndarray:
    """Apply the committed 29-joint sagittal permutation/sign convention."""

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (29,) or not np.isfinite(array).all():
        raise ValueError("classic G1 joint state must contain 29 finite values")
    return np.ascontiguousarray(
        array[JOINT_MIRROR_PERMUTATION] * JOINT_MIRROR_SIGN,
        dtype=np.float32,
    )


def mirror_classic_g1_physical_row_v3(
    x: object, y: object, phase: object
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Mirror one physical 346/268 row without normalizing or synthesizing state."""

    input_value = np.asarray(x, dtype=np.float32)
    target_value = np.asarray(y, dtype=np.float32)
    if input_value.shape != (CLASSIC_G1_INPUT_LAYOUT_V3.size,):
        raise ValueError("classic G1 v3 x must have shape (346,)")
    if target_value.shape != (OUTPUT_LAYOUT.size,):
        raise ValueError("classic G1 y must have shape (268,)")
    if not np.isfinite(input_value).all() or not np.isfinite(target_value).all():
        raise ValueError("classic G1 physical row must be finite")
    window = PFNNTrainingWindow(
        x=input_value[: INPUT_LAYOUT.size],
        y=target_value,
        phase=float(phase),
        clip_id="terrain_slopes__slope_000__000",
        split_identity="slope_000",
        split="train",
        sequence_lane="motion",
        center_frame=0,
        motion_sha256="a" * 64,
        terrain_sha256="b" * 64,
        terrain_class="flat",
    )
    mirrored = mirror_window(window)
    mirrored_input = pack_classic_g1_input_v3(
        mirrored.x,
        mirror_classic_g1_joint_state(
            input_value[CLASSIC_G1_INPUT_LAYOUT_V3["joint_position"]]
        ),
        mirror_classic_g1_joint_state(
            input_value[CLASSIC_G1_INPUT_LAYOUT_V3["joint_velocity"]]
        ),
    )
    return mirrored_input, mirrored.y.copy(), np.float32(mirrored.phase)


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
        raise ValueError("artifact path escapes the dataset root")
    return path


def _window_key(
    clip_id: object, sequence_lane: object, center_frame: object
) -> tuple[str, str, int]:
    return str(clip_id), str(sequence_lane), int(center_frame)


def _validate_manifest_contract(manifest: Mapping[str, object]) -> None:
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("dataset manifest schema is invalid")
    expected = {
        "status": "accepted",
        "fps": 30.0,
        "input_size": INPUT_LAYOUT.size,
        "output_size": OUTPUT_LAYOUT.size,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "contact_order": list(CONTACT_ORDER),
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("dataset manifest contract is invalid")
    roots = manifest.get("source_roots")
    records = manifest.get("source_records")
    identities = manifest.get("split_identities")
    options = manifest.get("build_options")
    if (
        not isinstance(roots, dict)
        or not roots
        or not isinstance(records, dict)
        or not records
        or not isinstance(identities, dict)
        or set(identities) != set(_SPLITS)
        or not isinstance(options, dict)
    ):
        raise ValueError("dataset manifest provenance is invalid")
    checked_roots: dict[str, dict[str, str]] = {}
    for name, raw in sorted(roots.items()):
        if type(name) is not str or not isinstance(raw, dict) \
                or set(raw) != _SOURCE_ROOT_FIELDS:
            raise ValueError("dataset source-root provenance is invalid")
        if any(type(raw[field]) is not str or not raw[field] for field in _SOURCE_ROOT_FIELDS):
            raise ValueError("dataset source-root provenance is invalid")
        if _SHA256_RE.fullmatch(raw["license_manifest_sha256"]) is None:
            raise ValueError("dataset source-root digest is invalid")
        checked_roots[name] = dict(raw)
    derived_identities = {split: set() for split in _SPLITS}
    checked_records: dict[str, dict[str, str]] = {}
    assignments: dict[str, str] = {}
    for clip_id, raw in sorted(records.items()):
        if (
            type(clip_id) is not str
            or not clip_id
            or len(clip_id) > 128
            or not isinstance(raw, dict)
            or set(raw) != _SOURCE_RECORD_FIELDS
            or any(type(raw[field]) is not str for field in _SOURCE_RECORD_FIELDS)
        ):
            raise ValueError("dataset source record is invalid")
        if (
            raw["source_kind"] not in ("grail", "lafan")
            or raw["split"] not in _SPLITS
            or not raw["split_identity"]
            or len(raw["split_identity"]) > 128
            or not raw["license_id"]
            or _SHA256_RE.fullmatch(raw["motion_sha256"]) is None
            or (
                raw["terrain_sha256"]
                and _SHA256_RE.fullmatch(raw["terrain_sha256"]) is None
            )
            or (raw["source_kind"] == "lafan" and raw["terrain_sha256"])
            or raw["split_identity"] != terrain_identity(clip_id)
            or raw["split"] != sealed_split(raw["split_identity"])
        ):
            raise ValueError("dataset source record provenance is invalid")
        identity = raw["split_identity"]
        previous = assignments.setdefault(identity, raw["split"])
        if previous != raw["split"]:
            raise ValueError("dataset split identities overlap")
        derived_identities[raw["split"]].add(identity)
        checked_records[clip_id] = dict(raw)
    expected_identities = {
        split: sorted(derived_identities[split]) for split in _SPLITS
    }
    if identities != expected_identities:
        raise ValueError("dataset split identities are stale")
    source_payload = source_set_payload(
        checked_roots, checked_records, expected_identities, options
    )
    if canonical_json_sha256(source_payload) != manifest.get(
        "source_set_digest_sha256"
    ):
        raise ValueError("dataset source-set digest mismatch")


def _validated_input_normalization_values(
    x: object, x_mean: object, x_std: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(x, dtype=np.float32)
    mean = np.asarray(x_mean, dtype=np.float32)
    std = np.asarray(x_std, dtype=np.float32)
    supported_widths = (INPUT_LAYOUT.size, CLASSIC_G1_INPUT_LAYOUT_V3.size)
    if value.shape[-1:] not in tuple((width,) for width in supported_widths):
        raise ValueError("x must end in 288 or 346 features")
    width = value.shape[-1]
    if mean.shape != (width,) or std.shape != (width,):
        raise ValueError("x and normalization arrays must have the same supported shape")
    if not np.isfinite(value).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("x and normalization arrays must be finite")
    if np.any(std <= 0.0):
        raise ValueError("x_std must be positive")
    return value, mean, std


def normalize_pfnn_input(
    x: object, x_mean: object, x_std: object
) -> np.ndarray:
    """Normalize a raw PFNN input and apply the approved recurrent-body scale."""

    value, mean, std = _validated_input_normalization_values(x, x_mean, x_std)
    normalized = np.ascontiguousarray((value - mean) / std, dtype=np.float32)
    for field in ("previous_body_position", "previous_body_velocity"):
        normalized[..., INPUT_LAYOUT[field]] *= np.float32(0.1)
    return normalized


def denormalize_pfnn_input(
    x: object, x_mean: object, x_std: object
) -> np.ndarray:
    value, mean, std = _validated_input_normalization_values(x, x_mean, x_std)
    normalized = np.ascontiguousarray(value, dtype=np.float32).copy()
    for field in ("previous_body_position", "previous_body_velocity"):
        normalized[..., INPUT_LAYOUT[field]] /= np.float32(0.1)
    return np.ascontiguousarray(normalized * std + mean, dtype=np.float32)


def normalize_pfnn_output(
    y: object, y_mean: object, y_std: object
) -> np.ndarray:
    value = np.asarray(y, dtype=np.float32)
    mean = np.asarray(y_mean, dtype=np.float32)
    std = np.asarray(y_std, dtype=np.float32)
    if value.shape[-1:] != (OUTPUT_LAYOUT.size,):
        raise ValueError("y must end in 268 features")
    if mean.shape != (OUTPUT_LAYOUT.size,) or std.shape != (OUTPUT_LAYOUT.size,):
        raise ValueError("y normalization arrays must have shape (268,)")
    if not np.isfinite(value).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("y and normalization arrays must be finite")
    if np.any(std <= 0.0):
        raise ValueError("y_std must be positive")
    return np.ascontiguousarray((value - mean) / std, dtype=np.float32)


def pfnn_input_sha256(value: object) -> str:
    """Hash one canonical normalized 288- or 346-value PFNN input.

    The v1 domain remains byte-compatible for 288-value artifacts. The v3
    domain binds all 346 values, so neither layout can be mistaken for the
    other even if a caller supplies the same binary prefix.
    """

    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    domains = {
        INPUT_LAYOUT.size: b"mm-sonic-normalized-pfnn-input/v1\0",
        CLASSIC_G1_INPUT_LAYOUT_V3.size: b"mm-sonic-normalized-pfnn-input/v3\0",
    }
    if array.ndim != 1 or len(array) not in domains or not np.isfinite(array).all():
        raise ValueError(
            "normalized PFNN input receipt requires 288 or 346 finite values"
        )
    # IEEE negative zero is numerically identical and may arise from an
    # otherwise exact frame rotation, so canonicalize both signs to +0.
    array = array.copy()
    array[array == 0.0] = np.float32(0.0)
    digest = hashlib.sha256(domains[len(array)])
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validated_shard_arrays(
    path: Path,
    record: Mapping[str, object],
    source_records: Mapping[str, Mapping[str, str]],
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != _SHARD_FIELDS:
                raise ValueError("shard fields do not match the contract")
            cache = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid PFNN shard: {path}: {error}") from error
    count = int(record["count"])
    expected = {
        "x": ((count, INPUT_LAYOUT.size), np.dtype(np.float32)),
        "y": ((count, OUTPUT_LAYOUT.size), np.dtype(np.float32)),
        "phase": ((count,), np.dtype(np.float32)),
        "clip_id": ((count,), np.dtype("<U128")),
        "split_identity": ((count,), np.dtype("<U128")),
        "sequence_lane": ((count,), np.dtype("<U16")),
        "terrain_class": ((count,), np.dtype("<U10")),
        "center_frame": ((count,), np.dtype(np.int32)),
        "motion_sha256": ((count,), np.dtype("<U64")),
        "terrain_sha256": ((count,), np.dtype("<U64")),
    }
    for name, (shape, dtype) in expected.items():
        if cache[name].shape != shape or cache[name].dtype != dtype:
            raise ValueError(f"shard field {name} has invalid shape or dtype")
    if (
        not np.isfinite(cache["x"]).all()
        or not np.isfinite(cache["y"]).all()
        or not np.isfinite(cache["phase"]).all()
        or not np.isin(cache["terrain_class"], _TERRAIN_CLASSES).all()
        or not np.isin(cache["sequence_lane"], _SEQUENCE_LANES).all()
        or np.any(cache["phase"] < 0.0)
        or np.any(cache["phase"].astype(np.float64) >= 2.0 * np.pi)
        or not np.isin(
            cache["y"][:, OUTPUT_LAYOUT["contact_logit"]], (0.0, 1.0)
        ).all()
        or any(
            _SHA256_RE.fullmatch(str(value)) is None
            for value in cache["motion_sha256"]
        )
        or any(
            value and _SHA256_RE.fullmatch(str(value)) is None
            for value in cache["terrain_sha256"]
        )
    ):
        raise ValueError("shard values violate the phase/finiteness contract")
    validate_shard_row_provenance(
        cache,
        split=str(record["split"]),
        source_records=source_records,
    )
    return cache


class PFNNShardDataset:
    """Lazy index over one split; source motion and terrain are never reopened."""

    def __init__(self, root: str | Path, split: SplitName) -> None:
        self.root = Path(root).expanduser().resolve()
        if split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        self.split = split
        try:
            manifest = json.loads((self.root / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("dataset manifest is missing or invalid") from error
        _validate_manifest_contract(manifest)
        self._source_records = manifest["source_records"]
        all_records = manifest.get("shards")
        if not isinstance(all_records, list) or not all_records:
            raise ValueError("dataset manifest shards are invalid")
        seen_paths: set[Path] = set()
        seen_windows: set[tuple[str, str, int]] = set()
        for record in all_records:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256", "count", "split"}
                or record.get("split") not in _SPLITS
                or type(record.get("sha256")) is not str
                or _SHA256_RE.fullmatch(record["sha256"]) is None
                or type(record.get("count")) is not int
                or record["count"] < 1
            ):
                raise ValueError("manifest shard record is invalid")
            path_match = _SHARD_PATH_RE.fullmatch(str(record["path"]))
            if path_match is None or path_match.group(1) != record["split"]:
                raise ValueError("manifest shard path violates the sealed grammar")
            shard_path = _artifact_path(self.root, record.get("path"))
            if shard_path in seen_paths:
                raise ValueError("dataset manifest contains a duplicate shard path")
            seen_paths.add(shard_path)
            if not shard_path.is_file() or _sha256(shard_path) != record["sha256"]:
                raise ValueError(f"dataset shard hash mismatch: {shard_path}")
            arrays = _validated_shard_arrays(
                shard_path, record, manifest["source_records"]
            )
            for clip_id, sequence_lane, center_frame in zip(
                arrays["clip_id"], arrays["sequence_lane"], arrays["center_frame"]
            ):
                key = _window_key(clip_id, sequence_lane, center_frame)
                if key in seen_windows:
                    raise ValueError("dataset contains a duplicate window identity")
                seen_windows.add(key)
        self._records = [record for record in all_records if record["split"] == split]
        self._stops: list[int] = []
        total = 0
        for record in self._records:
            count = record["count"]
            total += count
            self._stops.append(total)
        self._length = total
        normalization_record = manifest.get("normalization")
        if (
            not isinstance(normalization_record, dict)
            or set(normalization_record) != {"path", "sha256"}
            or type(normalization_record.get("sha256")) is not str
            or _SHA256_RE.fullmatch(normalization_record["sha256"]) is None
        ):
            raise ValueError("dataset normalization record is invalid")
        normalization_path = _artifact_path(
            self.root, normalization_record.get("path")
        )
        if (
            not normalization_path.is_file()
            or _sha256(normalization_path) != normalization_record.get("sha256")
        ):
            raise ValueError("dataset normalization hash mismatch")
        dataset_payload = {
            "source_set_digest_sha256": manifest.get("source_set_digest_sha256"),
            "shards": sorted(
                manifest.get("shards", ()), key=lambda item: str(item["path"])
            ),
            "normalization_sha256": normalization_record.get("sha256"),
        }
        dataset_digest = canonical_json_sha256(dataset_payload)
        if dataset_digest != manifest.get("dataset_digest_sha256"):
            raise ValueError("dataset digest mismatch")
        try:
            with np.load(normalization_path, allow_pickle=False) as data:
                expected_normalization = {
                    "x_mean": ((INPUT_LAYOUT.size,), np.dtype(np.float32)),
                    "x_std": ((INPUT_LAYOUT.size,), np.dtype(np.float32)),
                    "y_mean": ((OUTPUT_LAYOUT.size,), np.dtype(np.float32)),
                    "y_std": ((OUTPUT_LAYOUT.size,), np.dtype(np.float32)),
                    "training_sample_count": ((), np.dtype(np.int64)),
                    "split_counts": ((3,), np.dtype(np.int64)),
                    "split_order": ((3,), np.dtype("<U10")),
                    "split_identity_digest": ((), np.dtype("<U64")),
                }
                if set(data.files) != set(expected_normalization):
                    raise ValueError("normalization fields are invalid")
                normalization_arrays = {
                    name: np.asarray(data[name]) for name in data.files
                }
                for name, (shape, dtype) in expected_normalization.items():
                    value = normalization_arrays[name]
                    if value.shape != shape or value.dtype != dtype:
                        raise ValueError(f"normalization field {name} is invalid")
                validate_normalization_metadata(
                    normalization_arrays,
                    shard_records=all_records,
                    split_identities=manifest["split_identities"],
                )
                self.x_mean = normalization_arrays["x_mean"]
                self.x_std = normalization_arrays["x_std"]
                self.y_mean = normalization_arrays["y_mean"]
                self.y_std = normalization_arrays["y_std"]
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"invalid normalization archive: {error}") from error
        # Exercise the complete normalization contract at construction time.
        normalize_pfnn_input(np.zeros(INPUT_LAYOUT.size), self.x_mean, self.x_std)
        normalize_pfnn_output(np.zeros(OUTPUT_LAYOUT.size), self.y_mean, self.y_std)
        if not np.array_equal(
            self.y_mean[OUTPUT_LAYOUT["contact_logit"]], np.zeros(4, np.float32)
        ) or not np.array_equal(
            self.y_std[OUTPUT_LAYOUT["contact_logit"]], np.ones(4, np.float32)
        ):
            raise ValueError("contact normalization must be exactly mean=0/std=1")
        self._cache_index: int | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self._length

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_index == shard_index and self._cache is not None:
            return self._cache
        record = self._records[shard_index]
        path = _artifact_path(self.root, record["path"])
        cache = _validated_shard_arrays(path, record, self._source_records)
        self._cache_index = shard_index
        self._cache = cache
        return cache

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, np.integer):
            index = int(index)
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._stops, index)
        start = 0 if shard_index == 0 else self._stops[shard_index - 1]
        row = index - start
        data = self._load_shard(shard_index)
        return {
            "x": normalize_pfnn_input(data["x"][row], self.x_mean, self.x_std),
            "y": normalize_pfnn_output(data["y"][row], self.y_mean, self.y_std),
            "phase": np.asarray(data["phase"][row], dtype=np.float32),
            "clip_id": str(data["clip_id"][row]),
            "split_identity": str(data["split_identity"][row]),
            "split": self.split,
            "sequence_lane": str(data["sequence_lane"][row]),
            "terrain_class": str(data["terrain_class"][row]),
            "center_frame": int(data["center_frame"][row]),
            "motion_sha256": str(data["motion_sha256"][row]),
            "terrain_sha256": str(data["terrain_sha256"][row]),
        }


__all__ = [
    "JOINT_MIRROR_PERMUTATION",
    "JOINT_MIRROR_SIGN",
    "PFNNShardDataset",
    "denormalize_pfnn_input",
    "mirror_classic_g1_joint_state",
    "mirror_classic_g1_physical_row_v3",
    "normalize_pfnn_input",
    "normalize_pfnn_output",
    "pack_classic_g1_input_v3",
    "pfnn_input_sha256",
]
