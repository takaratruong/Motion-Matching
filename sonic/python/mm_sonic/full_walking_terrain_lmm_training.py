"""Selection, frozen-test evaluation, and eligible-row refit at exactly 60 Hz."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np

from resources.g1_terrain_builder.artifacts import canonical_json_bytes
from resources.g1_terrain_builder.features import _NORMALIZATION_WEIGHTS
from resources.g1_terrain_builder.schema import G1_SKELETON_PARENTS

from .hybrid_terrain_lmm_training import (
    HybridGenerator,
    HybridModelConfig,
    evaluate_hybrid_rows,
    load_hybrid_generator,
    train_hybrid_generator,
)

FULL_WALKING_FPS = 60.0
FULL_WALKING_DT = 1.0 / 60.0
FULL_WALKING_HORIZONS = (20, 40, 60)
FULL_WALKING_MODEL_SCHEMA = "g1-full-walking-terrain-lmm-model/v1"
FULL_WALKING_TEST_SCHEMA = "g1-full-walking-terrain-lmm-test/v1"
FULL_WALKING_FAMILIES = ("flat", "curb", "slope", "stair")

_FEATURES = 31
_TERRAIN_GRID = 36
_COMPRESSOR_INPUTS = 908
_TARGET_OUTPUTS = 458
_SHA256_ALPHABET = frozenset("0123456789abcdef")
_SPEED_BIN_EDGES_MPS = (0.10, 0.35, 0.65)
_TURN_BIN_EDGES_RADPS = (-0.35, -0.05, 0.05, 0.35)
_SPEED_THRESHOLDS_MPS = np.asarray(_SPEED_BIN_EDGES_MPS, dtype=np.float32)
_TURN_THRESHOLDS_RADPS = np.asarray(_TURN_BIN_EDGES_RADPS, dtype=np.float32)
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
_DISABLED_SCALE = np.float32(np.finfo(np.float32).max)
_TEST_RECEIPT_KEYS = {
    "schema",
    "accepted",
    "status",
    "finite",
    "one_time_test",
    "corpus_manifest_sha256",
    "inventory_manifest_sha256",
    "split_ledger_manifest_sha256",
    "selection_model_manifest_sha256",
    "selection_validation_receipt_sha256",
    "selection_artifact_identity",
    "test_rows",
    "metrics",
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_ALPHABET for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} SHA-256 is not canonical")
    assert isinstance(value, str)
    return value


def _row_count(corpus: object) -> int:
    artifacts = getattr(corpus, "artifacts", None)
    positions = getattr(artifacts, "positions", None)
    if not isinstance(positions, np.ndarray) or positions.ndim != 3:
        raise TypeError("full walking corpus must expose artifacts.positions")
    return len(positions)


def _row_array(
    corpus: object,
    name: str,
    *,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
) -> np.ndarray:
    label = name.replace("_", " ")
    value = getattr(corpus, name, None)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"full walking corpus must expose {label}")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype or value.shape != shape:
        raise ValueError(
            f"full walking corpus {label} must have shape {shape} and dtype "
            f"{expected_dtype}"
        )
    return value


def _finite_row_array(
    corpus: object,
    name: str,
    *,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = _row_array(corpus, name, dtype=dtype, shape=shape)
    for first in range(0, len(value), 65_536):
        if not np.isfinite(value[first : first + 65_536]).all():
            raise ValueError(f"full walking corpus {name} must be finite")
    return value


def _mask(corpus: object, name: str, frames: int) -> np.ndarray:
    return _row_array(corpus, name, dtype=np.bool_, shape=(frames,))


def _expected_feature_normalization(
    raw_features: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.flatnonzero(train_mask)
    if not len(selected):
        raise ValueError("full walking train split is empty")
    offset = np.empty(_FEATURES, dtype=np.float32)
    scale = np.empty(_FEATURES, dtype=np.float32)
    for (start, stop), weight in zip(_FEATURE_GROUPS, _NORMALIZATION_WEIGHTS):
        values = np.asarray(raw_features[selected, start:stop], dtype=np.float64)
        mean = values.mean(axis=0, dtype=np.float64)
        group_std = float(np.mean(np.std(values, axis=0, dtype=np.float64)))
        offset[start:stop] = mean.astype(np.float32)
        if not np.isfinite(group_std) or group_std <= 0.0 or weight == 0.0:
            scale[start:stop] = _DISABLED_SCALE
        else:
            scale[start:stop] = np.float32(group_std / weight)
    normalized = np.empty_like(raw_features)
    for start, stop in _FEATURE_GROUPS:
        if np.all(scale[start:stop] == _DISABLED_SCALE):
            normalized[:, start:stop] = 0.0
        else:
            normalized[:, start:stop] = (
                (raw_features[:, start:stop] - offset[start:stop]) / scale[start:stop]
            ).astype(np.float32)
    return normalized, offset, scale


def _exact_string_table(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"full walking corpus {label} must be unique strings")
    return value


def _validate_ids(
    values: np.ndarray, names: tuple[str, ...], label: str, *, frames: int
) -> None:
    if values.shape != (frames,) or values.dtype != np.int32:
        raise ValueError(f"full walking corpus {label} IDs must be int32 [rows]")
    if np.any(values < 0) or int(values.max(initial=-1)) >= len(names):
        raise ValueError(f"full walking corpus {label} IDs exceed their name table")
    if not np.array_equal(np.unique(values), np.arange(len(names), dtype=np.int32)):
        raise ValueError(f"full walking corpus {label} name table has unused IDs")


def validate_full_walking_corpus(corpus: object) -> int:
    """Fail closed on the Task 4 mmap accessor and return its row count."""

    frames = _row_count(corpus)
    if type(getattr(corpus, "fps", None)) is not float or corpus.fps != 60.0:
        raise ValueError("full walking training requires exactly 60 Hz")
    if getattr(corpus, "horizons", None) != FULL_WALKING_HORIZONS:
        raise ValueError("full walking feature horizons must be exactly (20,40,60)")
    for name in (
        "manifest_sha256",
        "inventory_manifest_sha256",
        "split_ledger_manifest_sha256",
    ):
        _require_sha256(getattr(corpus, name, None), name.replace("_", " "))
    lane_hashes = getattr(corpus, "lane_manifest_sha256", None)
    if (
        not isinstance(lane_hashes, tuple)
        or not lane_hashes
        or any(not _is_sha256(value) for value in lane_hashes)
        or len(set(lane_hashes)) != len(lane_hashes)
    ):
        raise ValueError("full walking lane manifest SHA-256 authorities are invalid")

    artifacts = corpus.artifacts
    if hasattr(artifacts, "validate"):
        artifacts.validate()
    if artifacts.positions.shape != (frames, 31, 3):
        raise ValueError("full walking corpus must use exactly 31 canonical bones")
    if tuple(np.asarray(artifacts.parents).tolist()) != G1_SKELETON_PARENTS:
        raise ValueError("full walking corpus does not use the canonical G1 hierarchy")
    if artifacts.rotations.shape != (frames, 31, 4):
        raise ValueError("full walking rotations must have shape [rows,31,4]")
    if artifacts.velocities.shape != (frames, 31, 3) or (
        artifacts.angular_velocities.shape != (frames, 31, 3)
    ):
        raise ValueError("full walking velocity arrays have the wrong dimensions")
    if artifacts.contacts.shape != (frames, 2) or (
        artifacts.contacts.dtype != np.uint8
    ):
        raise ValueError("full walking contacts must be uint8 [rows,2]")
    if np.any(artifacts.contacts > 1):
        raise ValueError("full walking contacts must be binary")

    features = getattr(corpus, "features", None)
    if features is None or not all(
        isinstance(getattr(features, name, None), np.ndarray)
        for name in ("values", "offset", "scale")
    ):
        raise TypeError("full walking corpus must expose a FeatureSet")
    if hasattr(features, "validate"):
        features.validate()
    if features.values.shape != (frames, _FEATURES):
        raise ValueError("full walking matching features must be float32 [rows,31]")
    raw_features = _finite_row_array(
        corpus, "raw_features", dtype=np.float32, shape=(frames, _FEATURES)
    )
    _finite_row_array(
        corpus, "terrain_grid", dtype=np.float32, shape=(frames, _TERRAIN_GRID)
    )

    families = _row_array(corpus, "family_ids", dtype=np.uint8, shape=(frames,))
    if not np.array_equal(np.unique(families), np.arange(4, dtype=np.uint8)):
        raise ValueError("full walking corpus must contain all four terrain classes")
    source_ids = _row_array(corpus, "source_ids", dtype=np.int32, shape=(frames,))
    canonical_ids = _row_array(
        corpus, "canonical_source_ids", dtype=np.int32, shape=(frames,)
    )
    _validate_ids(
        source_ids,
        _exact_string_table(getattr(corpus, "source_names", None), "source names"),
        "source",
        frames=frames,
    )
    _validate_ids(
        canonical_ids,
        _exact_string_table(
            getattr(corpus, "canonical_source_names", None),
            "canonical source names",
        ),
        "canonical source",
        frames=frames,
    )
    terrain_ids = _row_array(corpus, "terrain_ids", dtype=np.int32, shape=(frames,))
    _validate_ids(
        terrain_ids,
        _exact_string_table(getattr(corpus, "terrain_names", None), "terrain names"),
        "terrain",
        frames=frames,
    )
    range_ids = _row_array(corpus, "range_ids", dtype=np.int32, shape=(frames,))
    _validate_ids(
        range_ids,
        _exact_string_table(getattr(corpus, "range_names", None), "range names"),
        "range",
        frames=frames,
    )

    split_ids = _row_array(corpus, "split_ids", dtype=np.uint8, shape=(frames,))
    quality_ids = _row_array(corpus, "quality_ids", dtype=np.uint8, shape=(frames,))
    if np.any(split_ids > 2):
        raise ValueError("full walking split IDs must be train/validation/test")
    if np.any(quality_ids > 2):
        raise ValueError("full walking quality IDs must be clean/usable/quarantined")
    eligible = _mask(corpus, "eligible_mask", frames)
    train = _mask(corpus, "train_mask", frames)
    validation = _mask(corpus, "validation_mask", frames)
    test = _mask(corpus, "test_mask", frames)
    if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
        raise ValueError("full walking split masks overlap")
    if not np.array_equal(train | validation | test, eligible):
        raise ValueError("full walking split masks must cover every eligible row")
    if not (
        np.array_equal((split_ids == 0) & eligible, train)
        and np.array_equal((split_ids == 1) & eligible, validation)
        and np.array_equal((split_ids == 2) & eligible, test)
    ):
        raise ValueError("full walking split IDs disagree with split masks")
    if not np.array_equal(eligible, quality_ids <= 1):
        raise ValueError("eligible rows must be exactly clean plus usable rows")
    eligible_rows = _row_array(
        corpus,
        "eligible_rows",
        dtype=np.int64,
        shape=(int(np.count_nonzero(eligible)),),
    )
    if not np.array_equal(eligible_rows, np.flatnonzero(eligible)):
        raise ValueError("full walking eligible row index is inconsistent")
    for label, mask in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        present = np.unique(families[mask])
        if not np.array_equal(present, np.arange(4, dtype=np.uint8)):
            raise ValueError(f"eligible {label} rows lack a required terrain class")
    for values, label in (
        (canonical_ids, "canonical source"),
        (terrain_ids, "terrain identity"),
    ):
        for identity in range(int(values.max(initial=-1)) + 1):
            identity_mask = eligible & (values == identity)
            if not np.any(identity_mask):
                raise ValueError(f"full walking {label} has no eligible rows")
            splits = np.unique(split_ids[identity_mask])
            if len(splits) != 1:
                raise ValueError(
                    f"full walking {label} must appear in exactly one split"
                )

    expected_features, expected_offset, expected_scale = (
        _expected_feature_normalization(raw_features, train)
    )
    if not np.allclose(features.offset, expected_offset, rtol=2e-6, atol=2e-6):
        raise ValueError(
            "full walking normalization offset changed from eligible train rows"
        )
    if not np.allclose(features.scale, expected_scale, rtol=2e-5, atol=2e-6):
        raise ValueError(
            "full walking normalization scale changed from eligible train rows"
        )
    if not np.allclose(features.values, expected_features, rtol=2e-6, atol=2e-6):
        raise ValueError(
            "full walking normalized feature table changed from raw features"
        )

    source_left = _row_array(
        corpus, "source_left_indices", dtype=np.int32, shape=(frames,)
    )
    source_right = _row_array(
        corpus, "source_right_indices", dtype=np.int32, shape=(frames,)
    )
    if np.any(source_left < 0) or np.any(source_left > source_right):
        raise ValueError("full walking source interpolation indices are invalid")
    alpha = _finite_row_array(corpus, "source_alpha", dtype=np.float32, shape=(frames,))
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("full walking source interpolation alpha is invalid")
    successor = _row_array(corpus, "successor", dtype=np.int64, shape=(frames,))
    successor_valid = _mask(corpus, "successor_valid", frames)
    valid_successors = successor[successor_valid]
    if np.any(valid_successors < 0) or np.any(valid_successors >= frames):
        raise ValueError("full walking successor contains an out-of-range row")
    valid_rows = np.flatnonzero(successor_valid)
    if np.any(range_ids[valid_successors] != range_ids[valid_rows]):
        raise ValueError("full walking successor crosses a range boundary")
    _finite_row_array(corpus, "root_delta_xy", dtype=np.float32, shape=(frames, 2))
    _finite_row_array(corpus, "root_delta_yaw", dtype=np.float32, shape=(frames,))
    return frames


@dataclass(frozen=True)
class FullWalkingTrainingView:
    """Hybrid-trainer view whose held-out population is validation only."""

    corpus: object
    train_mask: np.ndarray
    evaluation_mask: np.ndarray
    eligible_mask: np.ndarray
    fit_mask: np.ndarray

    def __getattr__(self, name: str) -> object:
        return getattr(self.corpus, name)


def make_training_view(corpus: object) -> FullWalkingTrainingView:
    """Expose train/validation to selection while retaining an all-eligible fit mask."""

    frames = validate_full_walking_corpus(corpus)
    eligible = _mask(corpus, "eligible_mask", frames)
    train = np.asarray(_mask(corpus, "train_mask", frames) & eligible, dtype=bool)
    validation = np.asarray(
        _mask(corpus, "validation_mask", frames) & eligible, dtype=bool
    )
    fit = np.asarray(eligible, dtype=bool)
    for value in (train, validation, fit):
        value.setflags(write=False)
    return FullWalkingTrainingView(
        corpus=corpus,
        train_mask=train,
        evaluation_mask=validation,
        eligible_mask=fit,
        fit_mask=fit,
    )


@dataclass(frozen=True, order=True)
class HierarchyKey:
    terrain_class: int
    canonical_source: int
    speed_bin: int
    turn_bin: int
    contact_bin: int
    transition_bin: int


def _derived_hierarchy_rows(
    corpus: object, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frames = _row_count(corpus)
    selected = np.asarray(rows, dtype=np.int64)
    if (
        selected.ndim != 1
        or len(selected) == 0
        or np.any(selected < 0)
        or np.any(selected >= frames)
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("hierarchical pool rows must be unique in-range indices")
    family = np.asarray(corpus.family_ids[selected], dtype=np.int32)
    source = np.asarray(corpus.canonical_source_ids[selected], dtype=np.int32)
    root_xy = np.asarray(corpus.root_delta_xy, dtype=np.float32)
    root_yaw = np.asarray(corpus.root_delta_yaw, dtype=np.float32)
    speed_all = np.linalg.norm(root_xy, axis=1) * np.float32(FULL_WALKING_FPS)
    turn_all = root_yaw * np.float32(FULL_WALKING_FPS)
    speed = np.digitize(speed_all[selected], _SPEED_THRESHOLDS_MPS).astype(np.int32)
    turn = np.digitize(turn_all[selected], _TURN_THRESHOLDS_RADPS).astype(np.int32)
    contacts = np.asarray(corpus.artifacts.contacts[selected], dtype=np.uint8)
    contact = (contacts[:, 0] | (contacts[:, 1] << np.uint8(1))).astype(np.int32)
    successor = np.asarray(corpus.successor, dtype=np.int64)
    successor_valid = np.asarray(corpus.successor_valid, dtype=bool)
    moving = speed_all > np.float32(_SPEED_THRESHOLDS_MPS[0])
    transition = np.full(len(selected), 3, dtype=np.int32)
    valid = successor_valid[selected]
    current_moving = moving[selected[valid]]
    next_moving = moving[successor[selected[valid]]]
    transition[valid] = np.where(
        ~current_moving & next_moving,
        1,
        np.where(current_moving & ~next_moving, 2, 0),
    )
    keys = np.column_stack((family, source, speed, turn, contact, transition))
    return selected, np.asarray(keys, dtype=np.int32)


def _nested_tree(
    keys: tuple[HierarchyKey, ...], pools: tuple[np.ndarray, ...], depth: int = 0
) -> object:
    if depth == 6:
        if len(pools) != 1:
            raise AssertionError("hierarchy leaf is not unique")
        return pools[0]
    grouped: dict[int, tuple[list[HierarchyKey], list[np.ndarray]]] = {}
    for key, pool in zip(keys, pools):
        component = (
            key.terrain_class,
            key.canonical_source,
            key.speed_bin,
            key.turn_bin,
            key.contact_bin,
            key.transition_bin,
        )[depth]
        key_group, pool_group = grouped.setdefault(component, ([], []))
        key_group.append(key)
        pool_group.append(pool)
    return tuple(
        (
            component,
            _nested_tree(tuple(group_keys), tuple(group_pools), depth + 1),
        )
        for component, (group_keys, group_pools) in sorted(grouped.items())
    )


def _sample_tree(
    tree: object, count: int, generator: np.random.Generator
) -> list[np.ndarray]:
    if isinstance(tree, np.ndarray):
        if count == 0:
            return []
        return [generator.choice(tree, size=count, replace=True)]
    if not isinstance(tree, tuple) or not tree:
        raise AssertionError("hierarchical sampler tree is malformed")
    child_count = len(tree)
    counts = np.full(child_count, count // child_count, dtype=np.int64)
    remainder = count % child_count
    if remainder:
        counts[generator.permutation(child_count)[:remainder]] += 1
    sampled: list[np.ndarray] = []
    for (_, child), allocation in zip(tree, counts):
        sampled.extend(_sample_tree(child, int(allocation), generator))
    return sampled


@dataclass(frozen=True)
class HierarchicalTrainingPools:
    """Precomputed non-empty nested pools for source-balanced post-coverage draws."""

    keys: tuple[HierarchyKey, ...]
    pools: tuple[np.ndarray, ...]
    tree: object

    @classmethod
    def from_corpus(
        cls, corpus: object, eligible: np.ndarray
    ) -> HierarchicalTrainingPools:
        frames = _row_count(corpus)
        mask = np.asarray(eligible)
        if mask.dtype != np.bool_ or mask.shape != (frames,):
            raise ValueError("hierarchical eligibility must be a row boolean mask")
        rows, matrix = _derived_hierarchy_rows(corpus, np.flatnonzero(mask))
        if set(matrix[:, 0].tolist()) != {0, 1, 2, 3}:
            raise ValueError("hierarchical pools require all four terrain classes")
        order = np.lexsort(
            (
                rows,
                matrix[:, 5],
                matrix[:, 4],
                matrix[:, 3],
                matrix[:, 2],
                matrix[:, 1],
                matrix[:, 0],
            )
        )
        ordered_rows = rows[order]
        ordered_keys = matrix[order]
        boundaries = np.flatnonzero(
            np.r_[True, np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1), True]
        )
        keys: list[HierarchyKey] = []
        pools: list[np.ndarray] = []
        for first, stop in pairwise(boundaries):
            components = ordered_keys[int(first)]
            key = HierarchyKey(*(int(value) for value in components))
            pool = np.asarray(ordered_rows[int(first) : int(stop)], dtype=np.int64)
            pool.setflags(write=False)
            keys.append(key)
            pools.append(pool)
        frozen_keys = tuple(keys)
        frozen_pools = tuple(pools)
        return cls(
            keys=frozen_keys,
            pools=frozen_pools,
            tree=_nested_tree(frozen_keys, frozen_pools),
        )

    def sample(self, batch_size: int, generator: np.random.Generator) -> np.ndarray:
        if type(batch_size) is not int or batch_size < 4:
            raise ValueError("hierarchical batch must include every terrain class")
        if not isinstance(generator, np.random.Generator):
            raise TypeError("hierarchical sampler requires numpy.random.Generator")
        chunks = _sample_tree(self.tree, batch_size, generator)
        if not chunks:
            raise AssertionError("hierarchical sampler produced no rows")
        rows = np.concatenate(chunks).astype(np.int64, copy=False)
        if len(rows) != batch_size:
            raise AssertionError("hierarchical sampler returned a short batch")
        generator.shuffle(rows)
        return rows

    def key_for_row(self, row: int) -> HierarchyKey:
        if type(row) is not int or row < 0:
            raise ValueError("hierarchy row must be a non-negative integer")
        for key, pool in zip(self.keys, self.pools):
            location = int(np.searchsorted(pool, row))
            if location < len(pool) and int(pool[location]) == row:
                return key
        raise ValueError("row is not present in the hierarchical pools")


def _validate_config(config: HybridModelConfig, *, fit_all_rows: bool) -> None:
    if not isinstance(config, HybridModelConfig):
        raise TypeError("config must be HybridModelConfig")
    if config.dt != FULL_WALKING_DT:
        raise ValueError("full walking model config requires dt=1/60 exactly")
    if config.variant not in {"latent32", "latent64"} or config.width != 512:
        raise ValueError("full walking model requires latent32/64 with width 512")
    if config.loss_profile != "visual-articulation":
        raise ValueError("full walking model requires visual-articulation loss")
    if config.fit_all_rows is not fit_all_rows:
        raise ValueError(f"full walking config fit_all_rows must be {fit_all_rows}")


def full_walking_model_identity(
    corpus: object,
    *,
    config: HybridModelConfig,
    stage: Literal["selection", "refit"],
    selection_model_manifest_sha256: str | None = None,
    test_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Return the exact JSON identity embedded in every model member."""

    validate_full_walking_corpus(corpus)
    if stage not in {"selection", "refit"}:
        raise ValueError("full walking model stage must be selection or refit")
    _validate_config(config, fit_all_rows=stage == "refit")
    if stage == "selection":
        if (
            selection_model_manifest_sha256 is not None
            or test_receipt_sha256 is not None
        ):
            raise ValueError("selection identity cannot contain test/refit provenance")
    else:
        _require_sha256(selection_model_manifest_sha256, "selection model manifest")
        _require_sha256(test_receipt_sha256, "test receipt")
    identity: dict[str, object] = {
        "schema": FULL_WALKING_MODEL_SCHEMA,
        "stage": stage,
        "fps": FULL_WALKING_FPS,
        "dt": FULL_WALKING_DT,
        "horizons": list(FULL_WALKING_HORIZONS),
        "dimensions": {
            "matching_features": _FEATURES,
            "compressor_input": _COMPRESSOR_INPUTS,
            "decoder_input": _FEATURES + config.latent_size,
            "target": _TARGET_OUTPUTS,
            "latent": config.latent_size,
            "width": config.width,
        },
        "variant": config.variant,
        "loss_profile": config.loss_profile,
        "sampling": {
            "schema": "g1-full-walking-hierarchical-sampling/v1",
            "coverage": "deterministic-shuffled-every-fit-row-once",
            "hierarchy": [
                "terrain_class",
                "canonical_source",
                "speed_bin",
                "turn_bin",
                "contact_bin",
                "transition_bin",
            ],
            "speed_bin_edges_mps": list(_SPEED_BIN_EDGES_MPS),
            "turn_bin_edges_radps": list(_TURN_BIN_EDGES_RADPS),
            "contact_bin": "left-bit-or-right-bit-shifted-one",
            "transition_bins": [
                "steady",
                "stand-to-walk",
                "walk-to-stand",
                "range-end",
            ],
            "empty_bin_redistribution": (
                "recursive-equal-among-present-children-seeded-remainder"
            ),
        },
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "inventory_manifest_sha256": corpus.inventory_manifest_sha256,
        "split_ledger_manifest_sha256": corpus.split_ledger_manifest_sha256,
        "lane_manifest_sha256": list(corpus.lane_manifest_sha256),
    }
    if stage == "refit":
        identity["selection_model_manifest_sha256"] = selection_model_manifest_sha256
        identity["test_receipt_sha256"] = test_receipt_sha256
    return identity


def _manifest_payload(root: Path) -> tuple[dict[str, object], bytes, str]:
    path = root / "manifest.json"
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection model manifest cannot be decoded") from error
    canonical_model_payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    if not isinstance(value, dict) or canonical_model_payload != payload:
        raise ValueError("selection model manifest is not canonical JSON")
    return value, payload, hashlib.sha256(payload).hexdigest()


def _selection_generator(
    corpus: object, model: Path, *, device: str
) -> tuple[HybridGenerator, dict[str, object], str]:
    view = make_training_view(corpus)
    root = Path(model).expanduser().resolve()
    manifest, _, manifest_sha256 = _manifest_payload(root)
    generator = load_hybrid_generator(root, corpus=view, device=device)
    identity = full_walking_model_identity(
        corpus, config=generator.config, stage="selection"
    )
    if manifest.get("artifact_identity") != identity or (
        generator.artifact_identity != identity
    ):
        raise ValueError("selection model does not have the exact 60 Hz identity")
    if generator.config.fit_all_rows:
        raise ValueError("selection model cannot be an all-row refit")
    if not generator.canonical_selection_verified:
        raise ValueError(
            "selection model validation gates are red; refusing to consume test rows"
        )
    return generator, manifest, manifest_sha256


def train_selection(
    corpus: object,
    output: Path,
    *,
    config: HybridModelConfig,
) -> Mapping[str, object]:
    """Train on eligible train rows and select from validation rows only."""

    view = make_training_view(corpus)
    _validate_config(config, fit_all_rows=False)
    sampler = HierarchicalTrainingPools.from_corpus(view, view.train_mask)
    identity = full_walking_model_identity(corpus, config=config, stage="selection")
    return train_hybrid_generator(
        view,
        output,
        config=config,
        post_coverage_sampler=sampler,
        artifact_identity=identity,
    )


def _metric_receipt_valid(metrics: object) -> bool:
    if not isinstance(metrics, dict) or metrics.get("finite") is not True:
        raise ValueError("test physical metrics are invalid")
    if type(metrics.get("rows")) is not int or metrics["rows"] <= 0:
        raise ValueError("test physical metric row count is invalid")
    numeric_limits = {
        "joint_geodesic_mae_rad": 0.03,
        "joint_frame_max_p95_rad": 0.10,
        "local_position_p95_m": 0.03,
        "fk_body_position_p95_m": 0.08,
        "support_foot_position_p95_m": 0.05,
    }
    expected_limits = {
        **numeric_limits,
        "minimum_contact_f1": 0.85,
    }
    if metrics.get("gate_limits") != expected_limits:
        raise ValueError("test physical gate limits changed")
    accepted = True
    for name, limit in numeric_limits.items():
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"test metric {name} is invalid")
        accepted = accepted and value <= limit
    contact = metrics.get("contact_f1")
    if (
        not isinstance(contact, list)
        or len(contact) != 2
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
            for value in contact
        )
    ):
        raise ValueError("test contact F1 metrics are invalid")
    accepted = accepted and min(contact) >= 0.85
    if metrics.get("accepted") is not bool(accepted):
        raise ValueError("test physical gate result is inconsistent")
    return bool(accepted)


def _publish_file_exclusive(payload: bytes, output: Path) -> Path:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable test receipt already exists: {output}")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.staging-", dir=output.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"immutable test receipt already exists: {output}"
            ) from error
        directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)
    return output


def _reserve_frozen_test(
    selection_root: Path,
    *,
    corpus_manifest_sha256: str,
    selection_model_manifest_sha256: str,
    requested_output: Path,
) -> None:
    reservation_root = selection_root.parent / (
        ".full-walking-frozen-test-"
        f"{corpus_manifest_sha256}-{selection_model_manifest_sha256}"
    )
    try:
        reservation_root.mkdir(mode=0o755)
    except FileExistsError as error:
        raise FileExistsError(
            "frozen test is already reserved for this corpus and selection model"
        ) from error
    descriptor, staging_name = tempfile.mkstemp(
        prefix=".reservation-", dir=reservation_root
    )
    staging = Path(staging_name)
    reservation = {
        "schema": "g1-full-walking-terrain-lmm-frozen-test-reservation/v1",
        "status": "reserved-before-evaluation",
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "selection_model_manifest_sha256": selection_model_manifest_sha256,
        "requested_output": str(requested_output),
    }
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(reservation))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, reservation_root / "reservation.json")
        directory = os.open(
            reservation_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)


def evaluate_frozen_test(
    corpus: object,
    model: Path,
    output: Path,
    *,
    device: str,
) -> Mapping[str, object]:
    """Evaluate the frozen selection checkpoint once on eligible test rows."""

    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable test receipt already exists: {output}")
    generator, manifest, selection_sha256 = _selection_generator(
        corpus, Path(model), device=device
    )
    _reserve_frozen_test(
        Path(model).expanduser().resolve(),
        corpus_manifest_sha256=corpus.manifest_sha256,
        selection_model_manifest_sha256=selection_sha256,
        requested_output=output,
    )
    rows = np.flatnonzero(corpus.test_mask & corpus.eligible_mask).astype(
        np.int64, copy=False
    )
    metrics = evaluate_hybrid_rows(corpus, generator, rows)
    accepted = _metric_receipt_valid(metrics)
    artifacts = manifest.get("artifacts")
    descriptor = (
        artifacts.get("evaluation.json") if isinstance(artifacts, dict) else None
    )
    validation_sha256 = (
        descriptor.get("sha256") if isinstance(descriptor, dict) else None
    )
    _require_sha256(validation_sha256, "selection validation receipt")
    receipt: dict[str, object] = {
        "schema": FULL_WALKING_TEST_SCHEMA,
        "accepted": accepted,
        "status": "test-gates-green" if accepted else "test-gates-red",
        "finite": True,
        "one_time_test": True,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "inventory_manifest_sha256": corpus.inventory_manifest_sha256,
        "split_ledger_manifest_sha256": corpus.split_ledger_manifest_sha256,
        "selection_model_manifest_sha256": selection_sha256,
        "selection_validation_receipt_sha256": validation_sha256,
        "selection_artifact_identity": dict(generator.artifact_identity or {}),
        "test_rows": len(rows),
        "metrics": metrics,
    }
    _publish_file_exclusive(canonical_json_bytes(receipt), output)
    return receipt


def load_test_receipt(path: Path) -> dict[str, object]:
    """Load and validate one canonical immutable frozen-test receipt."""

    try:
        payload = Path(path).expanduser().resolve().read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen test receipt cannot be decoded") from error
    if (
        not isinstance(value, dict)
        or set(value) != _TEST_RECEIPT_KEYS
        or canonical_json_bytes(value) != payload
        or value.get("schema") != FULL_WALKING_TEST_SCHEMA
        or value.get("finite") is not True
        or value.get("one_time_test") is not True
    ):
        raise ValueError("frozen test receipt schema/canonical form changed")
    for name in (
        "corpus_manifest_sha256",
        "inventory_manifest_sha256",
        "split_ledger_manifest_sha256",
        "selection_model_manifest_sha256",
        "selection_validation_receipt_sha256",
    ):
        _require_sha256(value.get(name), name.replace("_", " "))
    accepted = _metric_receipt_valid(value.get("metrics"))
    metrics = value["metrics"]
    assert isinstance(metrics, dict)
    if (
        value.get("accepted") is not accepted
        or value.get("status") != ("test-gates-green" if accepted else "test-gates-red")
        or value.get("test_rows") != metrics.get("rows")
        or not isinstance(value.get("selection_artifact_identity"), dict)
    ):
        raise ValueError("frozen test receipt status/provenance is inconsistent")
    return value


def train_all_rows(
    corpus: object,
    output: Path,
    *,
    selection_model: Path,
    test_receipt: Path,
    config: HybridModelConfig,
) -> Mapping[str, object]:
    """Refit on every eligible row while binding frozen selection/test receipts."""

    view = make_training_view(corpus)
    _validate_config(config, fit_all_rows=True)
    selection, selection_manifest, selection_sha256 = _selection_generator(
        corpus, Path(selection_model), device="cpu"
    )
    receipt_path = Path(test_receipt).expanduser().resolve()
    receipt = load_test_receipt(receipt_path)
    receipt_payload = receipt_path.read_bytes()
    test_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    expected_bindings = {
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "inventory_manifest_sha256": corpus.inventory_manifest_sha256,
        "split_ledger_manifest_sha256": corpus.split_ledger_manifest_sha256,
        "selection_model_manifest_sha256": selection_sha256,
    }
    if any(receipt.get(name) != value for name, value in expected_bindings.items()):
        raise ValueError("test receipt is not bound to this selection model/corpus")
    selection_artifacts = selection_manifest.get("artifacts")
    selection_evaluation = (
        selection_artifacts.get("evaluation.json")
        if isinstance(selection_artifacts, dict)
        else None
    )
    selection_validation_sha256 = (
        selection_evaluation.get("sha256")
        if isinstance(selection_evaluation, dict)
        else None
    )
    if (
        receipt.get("selection_validation_receipt_sha256")
        != selection_validation_sha256
    ):
        raise ValueError("test receipt validation SHA does not match the selection")
    if receipt.get("selection_artifact_identity") != dict(
        selection.artifact_identity or {}
    ):
        raise ValueError("test receipt selection model identity changed")
    if receipt.get("accepted") is not True:
        raise ValueError(
            "all-row refit requires a frozen test receipt with green gates"
        )
    identity = full_walking_model_identity(
        corpus,
        config=config,
        stage="refit",
        selection_model_manifest_sha256=selection_sha256,
        test_receipt_sha256=test_sha256,
    )
    sampler = HierarchicalTrainingPools.from_corpus(view, view.fit_mask)
    return train_hybrid_generator(
        view,
        output,
        config=config,
        selection_model=selection_model,
        selection_model_manifest_sha256=selection_sha256,
        post_coverage_sampler=sampler,
        artifact_identity=identity,
    )


def _load_cli_corpus(path: Path) -> object:
    try:
        from .full_walking_terrain_lmm_corpus import load_full_corpus
    except ImportError as error:
        raise RuntimeError(
            "Task 4 full walking corpus loader is unavailable"
        ) from error
    return load_full_corpus(path)


def _config_from_arguments(
    arguments: argparse.Namespace, *, fit_all_rows: bool
) -> HybridModelConfig:
    return HybridModelConfig(
        variant=arguments.variant,
        seed=arguments.seed,
        device=arguments.device,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        post_coverage_steps=arguments.post_coverage_steps,
        normalization_chunk_size=arguments.normalization_chunk_size,
        evaluation_chunk_size=arguments.evaluation_chunk_size,
        dt=FULL_WALKING_DT,
        fit_all_rows=fit_all_rows,
        loss_profile="visual-articulation",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--variant", choices=("latent32", "latent64"), default="latent32"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--post-coverage-steps", type=int, default=30_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--normalization-chunk-size", type=int, default=2048)
    parser.add_argument("--evaluation-chunk-size", type=int, default=4096)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="train using train/validation only")
    _add_training_arguments(select)
    test = commands.add_parser("test", help="run the one-time frozen test split")
    test.add_argument("--corpus", required=True, type=Path)
    test.add_argument("--model", required=True, type=Path)
    test.add_argument("--output", required=True, type=Path)
    test.add_argument("--device", required=True)
    refit = commands.add_parser("refit", help="refit on all clean/usable rows")
    _add_training_arguments(refit)
    refit.add_argument("--selection", required=True, type=Path)
    refit.add_argument("--test-receipt", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    corpus = _load_cli_corpus(arguments.corpus)
    if arguments.command == "select":
        receipt = train_selection(
            corpus,
            arguments.output,
            config=_config_from_arguments(arguments, fit_all_rows=False),
        )
    elif arguments.command == "test":
        receipt = evaluate_frozen_test(
            corpus,
            arguments.model,
            arguments.output,
            device=arguments.device,
        )
    else:
        receipt = train_all_rows(
            corpus,
            arguments.output,
            selection_model=arguments.selection,
            test_receipt=arguments.test_receipt,
            config=_config_from_arguments(arguments, fit_all_rows=True),
        )
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "FULL_WALKING_DT",
    "FULL_WALKING_FPS",
    "FullWalkingTrainingView",
    "HierarchicalTrainingPools",
    "HierarchyKey",
    "evaluate_frozen_test",
    "full_walking_model_identity",
    "load_test_receipt",
    "main",
    "make_training_view",
    "train_all_rows",
    "train_selection",
    "validate_full_walking_corpus",
]


if __name__ == "__main__":
    raise SystemExit(main())
