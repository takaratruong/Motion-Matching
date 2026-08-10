"""Streaming training and full-corpus evaluation for the 25 Hz hybrid terrain LMM.

The broad corpus is memory mapped on the host.  This module deliberately derives
the 908-D compressor rows and 458-D decoder targets only for the current chunk;
no complete training table is ever materialized or copied to CUDA.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from resources import quat  # noqa: E402

_CUDA_INITIALIZED_AT_MODULE_IMPORT = torch.cuda.is_initialized()

_MODEL_SCHEMA = "g1-hybrid-terrain-lmm-model/v1"
_TRAINING_SCHEMA = "g1-hybrid-terrain-lmm-training/v1"
_EVALUATION_SCHEMA = "g1-hybrid-terrain-lmm-evaluation/v1"
_MANIFEST_SCHEMA = "g1-hybrid-terrain-lmm-manifest/v1"
_FULL_WALKING_MODEL_SCHEMA = "g1-full-walking-terrain-lmm-model/v1"
_FULL_WALKING_LOCOMOTION_ACCEPTANCE_PROFILE = "lower-body-locomotion-v1"
_FULL_WALKING_LOCOMOTION_NON_ROOT_BONES = tuple(range(13))
_FULL_WALKING_LOCOMOTION_GATE_LIMITS = {
    "joint_geodesic_mae_rad": 0.03,
    "locomotion_joint_frame_max_p95_rad": 0.10,
    "fk_body_position_p95_m": 0.08,
    "support_foot_position_p95_m": 0.05,
    "minimum_contact_f1": 0.85,
}
_ARTIFACT_NAMES = ("latent.npy", "model.pt", "training.json", "evaluation.json")
_CONTINUOUS_OUTPUTS = 456
_OUTPUTS = 458
_COMPRESSOR_INPUTS = 908
_FEATURES = 31
_BONES = 31
_CONTACTS = 2
_SUPPORT_BONES = (7, 13)
_LOSS_PROFILES = ("uniform", "visual-articulation")
_LOSS_PROFILE_WEIGHTS = {
    "uniform": {
        "continuous_mean": 1.0,
        "contact_bce": 1.0,
        "latent_l2": 1.0e-4,
    },
    "visual-articulation": {
        "rotation_6d_mean": 1.0,
        "hips_height": 0.25,
        "auxiliary_continuous_mean": 0.05,
        "contact_bce": 0.10,
        "latent_l2": 1.0e-4,
    },
}
_PHYSICAL_ACCEPTANCE_LIMITS = {
    "joint_geodesic_mae_rad": 0.03,
    "joint_frame_max_p95_rad": 0.10,
    "fk_body_position_p95_m": 0.08,
    "support_foot_position_p95_m": 0.05,
}
_MINIMUM_CONTACT_F1 = 0.85


@dataclass(frozen=True)
class HybridModelConfig:
    """Frozen architecture/training configuration for one bounded variant."""

    variant: str = "latent32"
    seed: int = 1234
    device: str = "cuda:5"
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    post_coverage_steps: int = 20_000
    normalization_chunk_size: int = 2048
    evaluation_chunk_size: int = 4096
    dt: float = 0.04
    fit_all_rows: bool = False
    loss_profile: str = "uniform"

    def __post_init__(self) -> None:
        if self.variant not in {"latent32", "latent64", "wider"}:
            raise ValueError("variant must be latent32, latent64, or wider")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.device) is not str or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        for name in ("batch_size", "normalization_chunk_size", "evaluation_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.post_coverage_steps) is not int or self.post_coverage_steps < 0:
            raise ValueError("post_coverage_steps must be a non-negative integer")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if type(self.dt) is not float or self.dt not in (0.04, 1.0 / 60.0):
            raise ValueError(
                "the hybrid terrain LMM ABI requires dt=0.04 or dt=1/60 exactly"
            )
        if type(self.fit_all_rows) is not bool:
            raise ValueError("fit_all_rows must be an exact bool")
        if self.loss_profile not in _LOSS_PROFILES:
            raise ValueError("loss profile must be uniform or visual-articulation")

    @property
    def latent_size(self) -> int:
        return 32 if self.variant == "latent32" else 64

    @property
    def width(self) -> int:
        return 1024 if self.variant == "wider" else 512

    @property
    def architecture(self) -> dict[str, list[int]]:
        return {
            "compressor": [908, self.width, self.width, self.width, self.latent_size],
            "decompressor": [31 + self.latent_size, self.width, self.width, 458],
        }

    @property
    def loss_weights(self) -> dict[str, float]:
        return dict(_LOSS_PROFILE_WEIGHTS[self.loss_profile])


class _Compressor(nn.Module):
    def __init__(self, config: HybridModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.Linear(_COMPRESSOR_INPUTS, config.width),
                nn.Linear(config.width, config.width),
                nn.Linear(config.width, config.width),
                nn.Linear(config.width, config.latent_size),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            values = F.elu(layer(values))
        return self.layers[-1](values)


class _Decompressor(nn.Module):
    def __init__(self, config: HybridModelConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.Linear(_FEATURES + config.latent_size, config.width),
                nn.Linear(config.width, config.width),
                nn.Linear(config.width, _OUTPUTS),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.layers[0](values))
        values = F.relu(self.layers[1](values))
        return self.layers[2](values)


@dataclass(frozen=True)
class TrainingBatch:
    compressor_input: np.ndarray
    target: np.ndarray


class TrainingRowSampler(Protocol):
    """Narrow post-coverage sampling seam shared by corpus-specific wrappers."""

    def sample(self, batch_size: int, generator: np.random.Generator) -> np.ndarray: ...


@dataclass
class HybridGenerator:
    root: Path | None
    config: HybridModelConfig
    compressor: _Compressor
    decompressor: _Decompressor
    compressor_mean: torch.Tensor
    compressor_scale: torch.Tensor
    output_mean: torch.Tensor
    output_scale: torch.Tensor
    latent: np.ndarray | None
    corpus_manifest_sha256: str
    artifact_identity: Mapping[str, object] | None
    device: torch.device
    manifest: Mapping[str, object]
    training_receipt: Mapping[str, object]
    evaluation_receipt: Mapping[str, object]
    manifest_sha256: str | None
    canonical_selection_verified: bool
    selection_provenance_verified: bool
    test_receipt_current: bool
    refit_receipt_current: bool

    def encode_rows(self, corpus: object, rows: np.ndarray) -> np.ndarray:
        indices = _row_indices(rows, _row_count(corpus))
        batch = assemble_training_batch(corpus, indices)
        values = _to_device(batch.compressor_input, self.device)
        with torch.no_grad():
            encoded = self.compressor(
                (values - self.compressor_mean) / self.compressor_scale
            )
        result = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError(
                "learned compressor produced non-finite latent rows"
            )
        return np.ascontiguousarray(result)

    def decode(
        self,
        features: np.ndarray,
        *,
        row_indices: np.ndarray | None = None,
        latent: np.ndarray | None = None,
    ) -> np.ndarray:
        feature_rows = np.asarray(features, dtype=np.float32)
        if feature_rows.ndim != 2 or feature_rows.shape[1] != _FEATURES:
            raise ValueError("decoder features must have shape [rows, 31]")
        if not np.isfinite(feature_rows).all():
            raise ValueError("decoder features must be finite")
        if (row_indices is None) == (latent is None):
            raise ValueError("supply exactly one of row_indices or latent")
        if row_indices is not None:
            if self.latent is None:
                raise ValueError("generator has no published latent table")
            indices = _row_indices(row_indices, len(self.latent))
            latent_rows = np.asarray(self.latent[indices], dtype=np.float32)
        else:
            latent_rows = np.asarray(latent, dtype=np.float32)
        if latent_rows.shape != (len(feature_rows), self.config.latent_size):
            raise ValueError("decoder latent rows do not match features/config")
        if not np.isfinite(latent_rows).all():
            raise FloatingPointError("decoder latent rows are non-finite")
        state = np.concatenate((feature_rows, latent_rows), axis=1)
        with torch.no_grad():
            prediction = self.decompressor(_to_device(state, self.device))
            continuous = (
                prediction[:, :_CONTINUOUS_OUTPUTS] * self.output_scale
                + self.output_mean
            )
            contacts = torch.sigmoid(prediction[:, _CONTINUOUS_OUTPUTS:])
            decoded = torch.cat((continuous, contacts), dim=1)
        result = decoded.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("learned decompressor produced non-finite rows")
        return np.ascontiguousarray(result)

    def decode_rows(self, features: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return self.decode(features, row_indices=rows)

    def decode_corpus_rows(self, corpus: object, rows: np.ndarray) -> np.ndarray:
        indices = _row_indices(rows, _row_count(corpus))
        features = np.asarray(_feature_set(corpus).values[indices], dtype=np.float32)
        if self.latent is None:
            latent = self.encode_rows(corpus, indices)
            return self.decode(features, latent=latent)
        return self.decode(features, row_indices=indices)


def _row_count(corpus: object) -> int:
    artifacts = getattr(corpus, "artifacts", None)
    positions = getattr(artifacts, "positions", None)
    if not isinstance(positions, np.ndarray) or positions.ndim != 3:
        raise TypeError("corpus must expose artifacts.positions")
    return len(positions)


def _feature_set(corpus: object) -> object:
    features = getattr(corpus, "features", None)
    values = getattr(features, "values", None)
    if not isinstance(values, np.ndarray) or values.shape != (_row_count(corpus), 31):
        raise TypeError("corpus must expose a 31-D FeatureSet")
    if values.dtype != np.float32:
        raise ValueError("corpus matching features must be float32")
    return features


def _mask(corpus: object, name: str) -> np.ndarray:
    values = np.asarray(getattr(corpus, name, None))
    if values.dtype != np.bool_ or values.shape != (_row_count(corpus),):
        raise ValueError(f"corpus {name} must be an explicit row boolean mask")
    return values


def _eligible_mask(corpus: object) -> np.ndarray:
    if getattr(corpus, "eligible_mask", None) is None:
        return np.ones(_row_count(corpus), dtype=bool)
    return _mask(corpus, "eligible_mask")


def _training_mask(corpus: object) -> np.ndarray:
    train = _mask(corpus, "train_mask")
    if getattr(corpus, "eligible_mask", None) is not None:
        train = np.asarray(train & _eligible_mask(corpus), dtype=bool)
    return train


def _evaluation_mask(corpus: object) -> np.ndarray:
    if getattr(corpus, "evaluation_mask", None) is not None:
        evaluation = _mask(corpus, "evaluation_mask")
    elif getattr(corpus, "validation_mask", None) is not None:
        evaluation = _mask(corpus, "validation_mask")
    else:
        raise ValueError(
            "corpus must expose evaluation_mask or an explicit validation_mask"
        )
    if getattr(corpus, "eligible_mask", None) is not None:
        evaluation = np.asarray(evaluation & _eligible_mask(corpus), dtype=bool)
    return evaluation


def _fitted_mask(corpus: object, config: HybridModelConfig) -> np.ndarray:
    if not config.fit_all_rows:
        return _training_mask(corpus)
    if getattr(corpus, "fit_mask", None) is not None:
        fitted = _mask(corpus, "fit_mask")
    else:
        fitted = _eligible_mask(corpus)
    if not np.any(fitted):
        raise ValueError("all-row fit mask must select at least one eligible row")
    if np.any(fitted & ~_eligible_mask(corpus)):
        raise ValueError("all-row fit mask includes an ineligible row")
    return fitted


def _validate_corpus(corpus: object) -> None:
    artifacts = getattr(corpus, "artifacts", None)
    if artifacts is None:
        raise TypeError("corpus must expose artifacts")
    if hasattr(artifacts, "validate"):
        artifacts.validate()
    frames = _row_count(corpus)
    if artifacts.positions.shape != (frames, _BONES, 3):
        raise ValueError("hybrid training requires exactly 31 canonical bones")
    if artifacts.rotations.shape != (frames, _BONES, 4):
        raise ValueError("hybrid rotations must have shape [rows,31,4]")
    if (
        artifacts.velocities.shape != artifacts.positions.shape
        or artifacts.angular_velocities.shape != artifacts.positions.shape
    ):
        raise ValueError("hybrid velocity arrays do not match positions")
    if artifacts.contacts.shape != (frames, _CONTACTS):
        raise ValueError("hybrid contacts must have shape [rows,2]")
    if np.any(artifacts.contacts > 1):
        raise ValueError("hybrid contacts must be binary")
    if np.asarray(artifacts.parents).shape != (_BONES,):
        raise ValueError("hybrid parents must contain 31 entries")
    features = _feature_set(corpus).values
    validation_chunk = 65_536
    for first in range(0, frames, validation_chunk):
        if not np.isfinite(features[first : first + validation_chunk]).all():
            raise ValueError("corpus matching features must be finite float32")
    train = _training_mask(corpus)
    evaluation = _evaluation_mask(corpus)
    eligible = _eligible_mask(corpus)
    if np.any(train & evaluation):
        raise ValueError("train/evaluation masks must be disjoint")
    if getattr(corpus, "eligible_mask", None) is None:
        if not np.all(train | evaluation):
            raise ValueError(
                "legacy train/evaluation masks must be a complete partition"
            )
    elif np.any((train | evaluation) & ~eligible):
        raise ValueError("train/evaluation masks include an ineligible row")
    if not np.any(train) or not np.any(evaluation):
        raise ValueError("train and evaluation masks must both be non-empty")
    _family_rows(corpus)
    _corpus_manifest_sha256(corpus)


def _row_indices(rows: np.ndarray, frames: int) -> np.ndarray:
    indices = np.asarray(rows)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("rows must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= frames):
        raise ValueError("rows must be non-empty and within the corpus")
    return indices


def _family_rows(corpus: object) -> np.ndarray:
    frames = _row_count(corpus)
    values = np.asarray(getattr(corpus, "family_ids", None))
    if values.ndim != 1:
        raise ValueError("corpus family_ids must be one-dimensional")
    if len(values) == frames:
        result = values
    else:
        artifacts = corpus.artifacts
        starts = np.asarray(artifacts.range_starts)
        stops = np.asarray(artifacts.range_stops)
        if len(values) != len(starts):
            raise ValueError("family_ids must be row-aligned or range-aligned")
        result = np.empty(frames, dtype=values.dtype)
        covered = np.zeros(frames, dtype=bool)
        for family, start, stop in zip(values, starts, stops):
            result[int(start) : int(stop)] = family
            covered[int(start) : int(stop)] = True
        if not np.all(covered):
            raise ValueError("range family IDs do not cover all rows")
    if len(np.unique(result)) == 0:
        raise ValueError("corpus has no source families")
    return result


def _corpus_manifest_sha256(corpus: object) -> str:
    for name in ("manifest_sha256", "cache_manifest_sha256"):
        digest = getattr(corpus, name, None)
        if isinstance(digest, str):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("corpus manifest SHA-256 is not canonical")
            return digest
    receipt = getattr(corpus, "manifest_receipt", None)
    if receipt is None:
        receipt = getattr(corpus, "manifest", None)
    if not isinstance(receipt, Mapping):
        raise TypeError("corpus does not expose an immutable manifest receipt")
    payload = _json_bytes(dict(receipt))
    return hashlib.sha256(payload).hexdigest()


def sample_training_rows(
    corpus: object,
    *,
    batch_size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Sample a nearly equal count from every family, using training rows only."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be numpy.random.Generator")
    train = _training_mask(corpus)
    families = _family_rows(corpus)
    family_values = np.unique(families)
    if batch_size < len(family_values):
        raise ValueError("batch_size must include at least one row from every family")
    counts = np.full(
        len(family_values), batch_size // len(family_values), dtype=np.int64
    )
    counts[: batch_size % len(family_values)] += 1
    sampled = []
    for family, count in zip(family_values, counts):
        available = np.flatnonzero(train & (families == family))
        if len(available) == 0:
            raise ValueError(f"family {family!r} has no training rows")
        sampled.append(generator.choice(available, size=int(count), replace=True))
    result = np.concatenate(sampled).astype(np.int64, copy=False)
    generator.shuffle(result)
    return result


@dataclass(frozen=True)
class _FamilyTrainingPools:
    pools: tuple[np.ndarray, ...]

    @classmethod
    def from_corpus(
        cls, corpus: object, eligible: np.ndarray | None = None
    ) -> _FamilyTrainingPools:
        train = _training_mask(corpus) if eligible is None else np.asarray(eligible)
        if train.dtype != np.bool_ or train.shape != (_row_count(corpus),):
            raise ValueError("family sampling eligibility must be a row boolean mask")
        families = _family_rows(corpus)
        pools = tuple(
            np.flatnonzero(train & (families == family)).astype(np.int64, copy=False)
            for family in np.unique(families)
        )
        if not pools or any(len(pool) == 0 for pool in pools):
            raise ValueError("every corpus family must have training rows")
        return cls(pools)

    def sample(self, batch_size: int, generator: np.random.Generator) -> np.ndarray:
        if batch_size < len(self.pools):
            raise ValueError(
                "batch_size must include at least one row from every family"
            )
        counts = np.full(len(self.pools), batch_size // len(self.pools), dtype=np.int64)
        counts[: batch_size % len(self.pools)] += 1
        result = np.concatenate(
            [
                generator.choice(pool, size=int(count), replace=True)
                for pool, count in zip(self.pools, counts)
            ]
        ).astype(np.int64, copy=False)
        generator.shuffle(result)
        return result


def iter_coverage_batches(
    corpus: object,
    *,
    batch_size: int,
    generator: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield one deterministic shuffled pass in which every train row occurs once."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be numpy.random.Generator")
    rows = np.flatnonzero(_training_mask(corpus)).astype(np.int64, copy=False)
    yield from _iter_coverage_row_batches(rows, batch_size, generator)


def _iter_coverage_row_batches(
    rows: np.ndarray, batch_size: int, generator: np.random.Generator
) -> Iterator[np.ndarray]:
    batch_starts = np.arange(0, len(rows), batch_size, dtype=np.int64)
    generator.shuffle(batch_starts)
    for first in batch_starts:
        start = int(first)
        yield rows[start : start + batch_size]


def assemble_training_batch(corpus: object, rows: np.ndarray) -> TrainingBatch:
    """Derive the Orange Duck row ABI for only the selected host rows."""

    indices = _row_indices(rows, _row_count(corpus))
    artifacts = corpus.artifacts
    positions = np.ascontiguousarray(artifacts.positions[indices], dtype=np.float32)
    rotations = np.ascontiguousarray(artifacts.rotations[indices], dtype=np.float32)
    velocities = np.ascontiguousarray(artifacts.velocities[indices], dtype=np.float32)
    angular = np.ascontiguousarray(
        artifacts.angular_velocities[indices], dtype=np.float32
    )
    if not all(
        np.isfinite(value).all()
        for value in (positions, rotations, velocities, angular)
    ):
        raise ValueError("selected canonical motion rows contain non-finite values")
    rotation_xy = quat.to_xform_xy(rotations).astype(np.float32)
    global_rotation, global_position, global_velocity, global_angular = quat.fk_vel(
        rotations, positions, velocities, angular, np.asarray(artifacts.parents)
    )
    character_rotation = quat.inv_mul(global_rotation[:, 0:1], global_rotation)
    character_position = quat.inv_mul_vec(
        global_rotation[:, 0:1], global_position - global_position[:, 0:1]
    )
    character_velocity = quat.inv_mul_vec(global_rotation[:, 0:1], global_velocity)
    character_angular = quat.inv_mul_vec(global_rotation[:, 0:1], global_angular)
    character_rotation_xy = quat.to_xform_xy(character_rotation).astype(np.float32)
    root_velocity = quat.inv_mul_vec(rotations[:, 0], velocities[:, 0]).astype(
        np.float32
    )
    root_angular = quat.inv_mul_vec(rotations[:, 0], angular[:, 0]).astype(np.float32)
    contacts = np.asarray(artifacts.contacts[indices], dtype=np.float32)
    count = len(indices)
    compressor_input = np.concatenate(
        (
            positions[:, 1:].reshape(count, -1),
            rotation_xy[:, 1:].reshape(count, -1),
            velocities[:, 1:].reshape(count, -1),
            angular[:, 1:].reshape(count, -1),
            character_position[:, 1:].reshape(count, -1),
            character_rotation_xy[:, 1:].reshape(count, -1),
            character_velocity[:, 1:].reshape(count, -1),
            character_angular[:, 1:].reshape(count, -1),
            root_velocity,
            root_angular,
            contacts,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    target = np.concatenate(
        (
            positions[:, 1:].reshape(count, -1),
            rotation_xy[:, 1:].reshape(count, -1),
            velocities[:, 1:].reshape(count, -1),
            angular[:, 1:].reshape(count, -1),
            root_velocity,
            root_angular,
            contacts,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if compressor_input.shape != (count, _COMPRESSOR_INPUTS) or target.shape != (
        count,
        _OUTPUTS,
    ):
        raise AssertionError("streamed training row assembly violated the 908/458 ABI")
    if not np.isfinite(compressor_input).all() or not np.isfinite(target).all():
        raise ValueError("derived training rows contain non-finite values")
    return TrainingBatch(
        compressor_input=np.ascontiguousarray(compressor_input),
        target=np.ascontiguousarray(target),
    )


def _to_device(values: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return tensor.to(
        device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
    )


def _configure_determinism(seed: int, device: torch.device) -> None:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("deterministic training supports only CPU or CUDA")
    if device.type == "cuda":
        if _CUDA_INITIALIZED_AT_MODULE_IMPORT:
            raise RuntimeError(
                "CUDA was initialized before deterministic hybrid training setup"
            )
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
            raise RuntimeError("CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device is unavailable: {device}")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device is unavailable: {device}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _streaming_normalization(
    corpus: object, rows: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = 0
    compressor_sum = np.zeros(_COMPRESSOR_INPUTS, dtype=np.float64)
    compressor_square = np.zeros(_COMPRESSOR_INPUTS, dtype=np.float64)
    output_sum = np.zeros(_CONTINUOUS_OUTPUTS, dtype=np.float64)
    output_square = np.zeros(_CONTINUOUS_OUTPUTS, dtype=np.float64)
    for first in range(0, len(rows), chunk_size):
        batch = assemble_training_batch(corpus, rows[first : first + chunk_size])
        compressor = batch.compressor_input.astype(np.float64)
        output = batch.target[:, :_CONTINUOUS_OUTPUTS].astype(np.float64)
        count += len(batch.target)
        compressor_sum += compressor.sum(axis=0)
        compressor_square += np.square(compressor).sum(axis=0)
        output_sum += output.sum(axis=0)
        output_square += np.square(output).sum(axis=0)
    if count != len(rows) or count == 0:
        raise AssertionError("normalization pass did not cover every fitted row")
    compressor_mean = compressor_sum / count
    output_mean = output_sum / count
    compressor_variance = np.maximum(
        compressor_square / count - np.square(compressor_mean), 0.0
    )
    output_variance = np.maximum(output_square / count - np.square(output_mean), 0.0)
    compressor_scale = np.maximum(np.sqrt(compressor_variance), 1.0e-5)
    output_scale = np.maximum(np.sqrt(output_variance), 1.0e-5)
    values = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (compressor_mean, compressor_scale, output_mean, output_scale)
    )
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("streaming normalization became non-finite")
    return values


def _training_step(
    corpus: object,
    rows: np.ndarray,
    generator: HybridGenerator,
    optimizer: torch.optim.Optimizer,
) -> float:
    batch = assemble_training_batch(corpus, rows)
    features = np.asarray(_feature_set(corpus).values[rows], dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("selected matching feature rows contain non-finite values")
    compressor_input = _to_device(batch.compressor_input, generator.device)
    target = _to_device(batch.target, generator.device)
    feature_tensor = _to_device(features, generator.device)
    optimizer.zero_grad(set_to_none=True)
    latent = generator.compressor(
        (compressor_input - generator.compressor_mean) / generator.compressor_scale
    )
    prediction = generator.decompressor(torch.cat((feature_tensor, latent), dim=1))
    loss = hybrid_generator_loss(
        prediction,
        target,
        latent,
        generator.output_mean,
        generator.output_scale,
        loss_profile=generator.config.loss_profile,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("hybrid generator training loss became non-finite")
    loss.backward()
    optimizer.step()
    if not all(
        torch.isfinite(parameter).all()
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise FloatingPointError("hybrid generator parameters became non-finite")
    return float(loss.detach().cpu().item())


def hybrid_generator_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor,
    output_mean: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    loss_profile: str = "uniform",
) -> torch.Tensor:
    """Return the deterministic generator objective for one streamed batch.

    ``visual-articulation`` reflects the native runtime ABI: MuJoCo consumes the
    learned local rotations, while runtime root placement cancels decoded Hips
    translation and native conversion ignores the remaining local translations
    and velocity channels.  Auxiliary channels retain a small non-zero loss so
    the published 458-D ABI remains finite and meaningful.
    """

    if loss_profile not in _LOSS_PROFILES:
        raise ValueError("loss profile must be uniform or visual-articulation")
    if prediction.ndim != 2 or prediction.shape[1] != _OUTPUTS:
        raise ValueError("generator prediction must have shape [batch,458]")
    if target.shape != prediction.shape:
        raise ValueError("generator target must match prediction")
    if latent.ndim != 2 or latent.shape[0] != prediction.shape[0]:
        raise ValueError("generator latent must be row-aligned and rank two")
    if output_mean.shape != (_CONTINUOUS_OUTPUTS,) or output_scale.shape != (
        _CONTINUOUS_OUTPUTS,
    ):
        raise ValueError("generator output normalization has the wrong shape")
    if torch.any(output_scale <= 0.0):
        raise ValueError("generator output scale must be positive")

    target_continuous = (target[:, :_CONTINUOUS_OUTPUTS] - output_mean) / output_scale
    element_loss = F.smooth_l1_loss(
        prediction[:, :_CONTINUOUS_OUTPUTS],
        target_continuous,
        reduction="none",
    )
    contact_loss = F.binary_cross_entropy_with_logits(
        prediction[:, _CONTINUOUS_OUTPUTS:], target[:, _CONTINUOUS_OUTPUTS:]
    )
    weights = _LOSS_PROFILE_WEIGHTS[loss_profile]
    if loss_profile == "uniform":
        reconstruction_loss = (
            weights["continuous_mean"] * element_loss.mean()
            + weights["contact_bce"] * contact_loss
        )
    else:
        rotation_loss = element_loss[:, 90:270].mean()
        hips_height_loss = element_loss[:, 1].mean()
        auxiliary_loss = torch.cat(
            (
                element_loss[:, :1],
                element_loss[:, 2:90],
                element_loss[:, 270:],
            ),
            dim=1,
        ).mean()
        reconstruction_loss = (
            weights["rotation_6d_mean"] * rotation_loss
            + weights["hips_height"] * hips_height_loss
            + weights["auxiliary_continuous_mean"] * auxiliary_loss
            + weights["contact_bce"] * contact_loss
        )
    return reconstruction_loss + weights["latent_l2"] * torch.mean(torch.square(latent))


@dataclass
class _MetricAccumulator:
    joint_sum: float
    joint_count: int
    frame_joint_max: list[np.ndarray]
    locomotion_frame_joint_max: list[np.ndarray]
    local_position_max: list[np.ndarray]
    fk_position_max: list[np.ndarray]
    support_position_max: list[np.ndarray]
    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    rows: int

    @classmethod
    def empty(cls) -> _MetricAccumulator:
        return cls(
            0.0,
            0,
            [],
            [],
            [],
            [],
            [],
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            0,
        )

    def add(self, corpus: object, rows: np.ndarray, decoded: np.ndarray) -> None:
        indices = _row_indices(rows, _row_count(corpus))
        output = np.asarray(decoded, dtype=np.float32)
        if output.shape != (len(indices), _OUTPUTS):
            raise ValueError("decoded rows must have shape [rows,458]")
        if not np.isfinite(output).all():
            raise FloatingPointError(
                "decoded physical evaluation contains non-finite values"
            )
        non_root = _BONES - 1
        position_end = 3 * non_root
        rotation_end = position_end + 6 * non_root
        predicted_position = output[:, :position_end].reshape(-1, non_root, 3)
        predicted_xy = output[:, position_end:rotation_end].reshape(-1, non_root, 3, 2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            predicted_rotation = quat.from_xform_xy(predicted_xy).astype(np.float32)
        if not np.isfinite(predicted_rotation).all():
            raise FloatingPointError("decoded rotations are non-finite or degenerate")
        artifacts = corpus.artifacts
        actual_rotation = np.asarray(artifacts.rotations[indices, 1:], dtype=np.float32)
        dot = np.clip(
            np.abs(np.sum(predicted_rotation * actual_rotation, axis=2)), 0.0, 1.0
        )
        joint_error = 2.0 * np.arccos(dot)
        self.joint_sum += float(joint_error.astype(np.float64).sum())
        self.joint_count += int(joint_error.size)
        self.frame_joint_max.append(np.max(joint_error, axis=1).astype(np.float32))
        self.locomotion_frame_joint_max.append(
            np.max(
                joint_error[:, _FULL_WALKING_LOCOMOTION_NON_ROOT_BONES], axis=1
            ).astype(np.float32)
        )
        actual_position = np.asarray(artifacts.positions[indices, 1:], dtype=np.float32)
        local_error = np.linalg.norm(predicted_position - actual_position, axis=2)
        self.local_position_max.append(np.max(local_error, axis=1).astype(np.float32))
        local_rotation = np.concatenate(
            (np.asarray(artifacts.rotations[indices, :1]), predicted_rotation), axis=1
        )
        local_position = np.concatenate(
            (np.asarray(artifacts.positions[indices, :1]), predicted_position), axis=1
        )
        _, predicted_global = quat.fk(
            local_rotation, local_position, np.asarray(artifacts.parents)
        )
        _, actual_global = quat.fk(
            np.asarray(artifacts.rotations[indices]),
            np.asarray(artifacts.positions[indices]),
            np.asarray(artifacts.parents),
        )
        body_error = np.linalg.norm(predicted_global - actual_global, axis=2)
        self.fk_position_max.append(np.max(body_error, axis=1).astype(np.float32))
        self.support_position_max.append(
            np.max(body_error[:, _SUPPORT_BONES], axis=1).astype(np.float32)
        )
        truth = np.asarray(artifacts.contacts[indices], dtype=bool)
        estimate = output[:, _CONTINUOUS_OUTPUTS:] >= np.float32(0.5)
        self.true_positive += np.count_nonzero(truth & estimate, axis=0)
        self.false_positive += np.count_nonzero(~truth & estimate, axis=0)
        self.false_negative += np.count_nonzero(truth & ~estimate, axis=0)
        self.rows += len(indices)

    def finish(self) -> dict[str, object]:
        if self.rows == 0 or self.joint_count == 0:
            raise ValueError("physical evaluation requires at least one row")
        frame_joint = np.concatenate(self.frame_joint_max)
        locomotion_frame_joint = np.concatenate(self.locomotion_frame_joint_max)
        local = np.concatenate(self.local_position_max)
        fk = np.concatenate(self.fk_position_max)
        support = np.concatenate(self.support_position_max)
        contact_f1 = []
        for side in range(2):
            denominator = (
                2 * int(self.true_positive[side])
                + int(self.false_positive[side])
                + int(self.false_negative[side])
            )
            contact_f1.append(
                1.0
                if denominator == 0
                else 2.0 * int(self.true_positive[side]) / denominator
            )
        metrics: dict[str, object] = {
            "finite": True,
            "rows": self.rows,
            "joint_geodesic_mae_rad": self.joint_sum / self.joint_count,
            "joint_frame_max_p95_rad": float(np.percentile(frame_joint, 95.0)),
            "locomotion_joint_frame_max_p95_rad": float(
                np.percentile(locomotion_frame_joint, 95.0)
            ),
            "local_position_p95_m": float(np.percentile(local, 95.0)),
            "fk_body_position_p95_m": float(np.percentile(fk, 95.0)),
            "support_foot_position_p95_m": float(np.percentile(support, 95.0)),
            "contact_f1": contact_f1,
            "gate_limits": {
                **_PHYSICAL_ACCEPTANCE_LIMITS,
                "minimum_contact_f1": _MINIMUM_CONTACT_F1,
            },
        }
        metrics["accepted"] = bool(
            metrics["joint_geodesic_mae_rad"]
            <= _PHYSICAL_ACCEPTANCE_LIMITS["joint_geodesic_mae_rad"]
            and metrics["joint_frame_max_p95_rad"]
            <= _PHYSICAL_ACCEPTANCE_LIMITS["joint_frame_max_p95_rad"]
            and metrics["fk_body_position_p95_m"]
            <= _PHYSICAL_ACCEPTANCE_LIMITS["fk_body_position_p95_m"]
            and metrics["support_foot_position_p95_m"]
            <= _PHYSICAL_ACCEPTANCE_LIMITS["support_foot_position_p95_m"]
            and min(contact_f1) >= _MINIMUM_CONTACT_F1
        )
        return metrics


def calculate_physical_metrics(
    corpus: object, rows: np.ndarray, decoded: np.ndarray
) -> dict[str, object]:
    """Calculate the practical decompressor gates for an explicit row set."""

    accumulator = _MetricAccumulator.empty()
    accumulator.add(corpus, rows, decoded)
    return accumulator.finish()


def _apply_physical_acceptance_profile(
    metrics: dict[str, object], artifact_identity: Mapping[str, object] | None
) -> dict[str, object]:
    """Apply a versioned full-walking gate without changing legacy semantics."""

    profile = (
        artifact_identity.get("acceptance_profile")
        if isinstance(artifact_identity, Mapping)
        and artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
        else None
    )
    if profile is None:
        return metrics
    if profile != _FULL_WALKING_LOCOMOTION_ACCEPTANCE_PROFILE:
        raise ValueError("full walking physical acceptance profile is unsupported")
    contact = metrics.get("contact_f1")
    if not isinstance(contact, list) or len(contact) != 2:
        raise ValueError("full walking contact metrics are invalid")
    metrics["acceptance_profile"] = profile
    metrics["joint_frame_max_scope"] = "all-30-non-root-joints-diagnostic"
    metrics["gate_limits"] = dict(_FULL_WALKING_LOCOMOTION_GATE_LIMITS)
    metrics["accepted"] = bool(
        metrics["joint_geodesic_mae_rad"]
        <= _FULL_WALKING_LOCOMOTION_GATE_LIMITS["joint_geodesic_mae_rad"]
        and metrics["locomotion_joint_frame_max_p95_rad"]
        <= _FULL_WALKING_LOCOMOTION_GATE_LIMITS["locomotion_joint_frame_max_p95_rad"]
        and metrics["fk_body_position_p95_m"]
        <= _FULL_WALKING_LOCOMOTION_GATE_LIMITS["fk_body_position_p95_m"]
        and metrics["support_foot_position_p95_m"]
        <= _FULL_WALKING_LOCOMOTION_GATE_LIMITS["support_foot_position_p95_m"]
        and min(float(value) for value in contact)
        >= _FULL_WALKING_LOCOMOTION_GATE_LIMITS["minimum_contact_f1"]
    )
    return metrics


def evaluate_hybrid_rows(
    corpus: object,
    generator: HybridGenerator,
    rows: np.ndarray,
    *,
    chunk_size: int | None = None,
) -> dict[str, object]:
    """Evaluate one explicit population without inventing a split contract."""

    if not isinstance(generator, HybridGenerator):
        raise TypeError("generator must be a HybridGenerator")
    if generator.corpus_manifest_sha256 != _corpus_manifest_sha256(corpus):
        raise ValueError("generator is bound to a different corpus manifest")
    selected = _row_indices(rows, _row_count(corpus))
    size = generator.config.evaluation_chunk_size if chunk_size is None else chunk_size
    if type(size) is not int or size <= 0:
        raise ValueError("evaluation chunk_size must be a positive integer")
    accumulator = _MetricAccumulator.empty()
    for first in range(0, len(selected), size):
        chunk = selected[first : first + size]
        decoded = generator.decode_corpus_rows(corpus, chunk)
        accumulator.add(corpus, chunk, decoded)
    return _apply_physical_acceptance_profile(
        accumulator.finish(), generator.artifact_identity
    )


def evaluate_hybrid_generator(
    corpus: object,
    generator: HybridGenerator | str | Path,
    *,
    chunk_size: int | None = None,
) -> dict[str, object]:
    """Evaluate every train and source-held-out row in bounded host/device chunks."""

    _validate_corpus(corpus)
    if isinstance(generator, (str, Path)):
        generator = load_hybrid_generator(generator, corpus=corpus, device="cpu")
    if not isinstance(generator, HybridGenerator):
        raise TypeError("generator must be a HybridGenerator or model directory")
    expected_hash = _corpus_manifest_sha256(corpus)
    if generator.corpus_manifest_sha256 != expected_hash:
        raise ValueError("generator is bound to a different corpus manifest")
    size = generator.config.evaluation_chunk_size if chunk_size is None else chunk_size
    if type(size) is not int or size <= 0:
        raise ValueError("evaluation chunk_size must be a positive integer")
    results: dict[str, object] = {}
    for receipt_name, population_mask in (
        ("train", _training_mask(corpus)),
        ("source_held_out", _evaluation_mask(corpus)),
    ):
        selected = np.flatnonzero(population_mask).astype(np.int64, copy=False)
        results[receipt_name] = evaluate_hybrid_rows(
            corpus, generator, selected, chunk_size=size
        )
    held_out = results["source_held_out"]
    assert isinstance(held_out, dict)
    full_walking_selection = bool(
        generator.artifact_identity is not None
        and generator.artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
        and generator.artifact_identity.get("stage") == "selection"
    )
    canonical_selection_accepted = bool(
        held_out["accepted"]
        if full_walking_selection
        else results["train"]["accepted"] and held_out["accepted"]
    )
    receipt = {
        "schema": _EVALUATION_SCHEMA,
        # Canonical Holden reconstruction is the architecture-selection gate.
        # Native G1 hinge limits are only executable in the MuJoCo runtime,
        # so this receipt alone never claims final runtime acceptance.
        "accepted": False,
        "canonical_selection_accepted": canonical_selection_accepted,
        "native_joint_limits_evaluated": False,
        "runtime_acceptance": "pending-native-joint-limit-and-smoke-gates",
        "finite": True,
        "corpus_manifest_sha256": expected_hash,
        "config": _config_receipt(generator.config),
        "train": results["train"],
        "source_held_out": held_out,
        "selection_tuple": [
            held_out["joint_geodesic_mae_rad"],
            held_out["fk_body_position_p95_m"],
            1.0 - min(held_out["contact_f1"]),
        ],
        "evaluation_scope": (
            "post-selection-all-row-refit-diagnostic"
            if generator.config.fit_all_rows
            else "source-held-out-selection"
        ),
    }
    if generator.artifact_identity is not None:
        receipt["artifact_identity"] = dict(generator.artifact_identity)
        if (
            generator.artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
            and generator.artifact_identity.get("stage") == "selection"
        ):
            receipt["selection_population"] = "validation"
            receipt["validation"] = held_out
    _require_json_finite(receipt)
    return receipt


def _config_receipt(config: HybridModelConfig) -> dict[str, object]:
    return {
        "variant": config.variant,
        "latent_size": config.latent_size,
        "width": config.width,
        "architecture": config.architecture,
        "seed": config.seed,
        "device": config.device,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "post_coverage_steps": config.post_coverage_steps,
        "normalization_chunk_size": config.normalization_chunk_size,
        "evaluation_chunk_size": config.evaluation_chunk_size,
        "dt": config.dt,
        "fit_all_rows": config.fit_all_rows,
        "loss_profile": config.loss_profile,
        "loss_weights": config.loss_weights,
    }


def _config_from_receipt(receipt: object) -> HybridModelConfig:
    if not isinstance(receipt, dict):
        raise TypeError("model config receipt is missing")
    keys = {
        "variant",
        "seed",
        "device",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "post_coverage_steps",
        "normalization_chunk_size",
        "evaluation_chunk_size",
        "dt",
        "fit_all_rows",
    }
    if not keys.issubset(receipt):
        raise ValueError("model config receipt is incomplete")
    arguments = {key: receipt[key] for key in keys}
    arguments["loss_profile"] = receipt.get("loss_profile", "uniform")
    config = HybridModelConfig(**arguments)
    if (
        receipt.get("latent_size") != config.latent_size
        or receipt.get("width") != config.width
        or receipt.get("architecture") != config.architecture
        or (
            "loss_profile" in receipt
            and receipt.get("loss_weights") != config.loss_weights
        )
    ):
        raise ValueError("model config architecture receipt changed")
    return config


def _checkpoint_bytes(generator: HybridGenerator) -> bytes:
    checkpoint = {
        "schema": _MODEL_SCHEMA,
        "corpus_manifest_sha256": generator.corpus_manifest_sha256,
        "compressor": generator.compressor.state_dict(),
        "decompressor": generator.decompressor.state_dict(),
        "compressor_mean": generator.compressor_mean.detach().cpu(),
        "compressor_scale": generator.compressor_scale.detach().cpu(),
        "output_mean": generator.output_mean.detach().cpu(),
        "output_scale": generator.output_scale.detach().cpu(),
        "variant": generator.config.variant,
    }
    if generator.artifact_identity is not None:
        checkpoint["artifact_identity"] = dict(generator.artifact_identity)
    output = io.BytesIO()
    torch.save(checkpoint, output)
    return output.getvalue()


def _write_latent(staging: Path, corpus: object, generator: HybridGenerator) -> None:
    frames = _row_count(corpus)
    path = staging / "latent.npy"
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype="<f4", shape=(frames, generator.config.latent_size)
    )
    try:
        chunk_size = generator.config.evaluation_chunk_size
        for first in range(0, frames, chunk_size):
            rows = np.arange(first, min(first + chunk_size, frames), dtype=np.int64)
            encoded = generator.encode_rows(corpus, rows)
            if not np.isfinite(encoded).all():
                raise FloatingPointError("all-row latent export became non-finite")
            output[first : first + len(rows)] = encoded
        output.flush()
    finally:
        del output
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _artifact_descriptor(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {"path": path.name, "size_bytes": size, "sha256": digest.hexdigest()}


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one model directory without replacing a racing owner."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "immutable model publication requires renameat2")
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
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"immutable model output already exists: {destination}",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _json_bytes(value: object) -> bytes:
    _require_json_finite(value)
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _normalized_artifact_identity(
    value: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError("artifact identity must be a non-empty mapping")
    if not all(type(key) is str and key for key in value):
        raise ValueError("artifact identity keys must be non-empty strings")
    try:
        normalized = json.loads(_json_bytes(dict(value)))
    except (TypeError, ValueError) as error:
        raise ValueError("artifact identity must be canonical JSON data") from error
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("artifact identity must be a non-empty JSON object")
    return normalized


def _require_json_finite(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _require_json_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_json_finite(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError("receipt contains a non-finite value")


def _physical_metrics_receipt_accepted(
    metrics: object, *, acceptance_profile: str | None = None
) -> bool:
    if not isinstance(metrics, dict) or metrics.get("finite") is not True:
        raise ValueError("selection model physical metrics are invalid")
    if type(metrics.get("rows")) is not int or metrics["rows"] <= 0:
        raise ValueError("selection model physical metrics row count is invalid")
    if acceptance_profile is None:
        limits = {
            **_PHYSICAL_ACCEPTANCE_LIMITS,
            "minimum_contact_f1": _MINIMUM_CONTACT_F1,
        }
        gated_names = tuple(_PHYSICAL_ACCEPTANCE_LIMITS)
    elif acceptance_profile == _FULL_WALKING_LOCOMOTION_ACCEPTANCE_PROFILE:
        limits = dict(_FULL_WALKING_LOCOMOTION_GATE_LIMITS)
        gated_names = tuple(name for name in limits if name != "minimum_contact_f1")
        if (
            metrics.get("acceptance_profile") != acceptance_profile
            or metrics.get("joint_frame_max_scope")
            != "all-30-non-root-joints-diagnostic"
        ):
            raise ValueError("selection model locomotion metric scope changed")
    else:
        raise ValueError("selection model physical acceptance profile is unsupported")
    if metrics.get("gate_limits") != limits:
        raise ValueError("selection model physical gate limits changed")
    values: dict[str, float] = {}
    diagnostic_names = ("local_position_p95_m", "joint_frame_max_p95_rad")
    for name in (*gated_names, *diagnostic_names):
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"selection model metric {name} is invalid")
        values[name] = float(value)
    contact = metrics.get("contact_f1")
    if (
        not isinstance(contact, list)
        or len(contact) != 2
        or not all(
            type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0
            for value in contact
        )
    ):
        raise ValueError("selection model contact metrics are invalid")
    accepted = bool(
        all(values[name] <= float(limits[name]) for name in gated_names)
        and min(float(value) for value in contact)
        >= float(limits["minimum_contact_f1"])
    )
    if metrics.get("accepted") is not accepted:
        raise ValueError("selection model physical gate result is inconsistent")
    return accepted


def _validated_evaluation_receipt(
    receipt: object,
    *,
    corpus: object,
    config: HybridModelConfig,
    corpus_manifest_sha256: str,
    artifact_identity: Mapping[str, object] | None,
) -> bool:
    if not isinstance(receipt, dict):
        # Artifact-contract violations use ValueError consistently at load seams.
        raise ValueError("model evaluation receipt is invalid")  # noqa: TRY004
    train = receipt.get("train")
    held_out = receipt.get("source_held_out")
    full_walking_selection = bool(
        artifact_identity is not None
        and artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
        and artifact_identity.get("stage") == "selection"
    )
    acceptance_profile = (
        artifact_identity.get("acceptance_profile")
        if artifact_identity is not None
        and artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
        else None
    )
    train_accepted = _physical_metrics_receipt_accepted(
        train, acceptance_profile=acceptance_profile
    )
    held_out_accepted = _physical_metrics_receipt_accepted(
        held_out, acceptance_profile=acceptance_profile
    )
    assert isinstance(train, dict) and isinstance(held_out, dict)
    train_rows = int(np.count_nonzero(_training_mask(corpus)))
    held_out_rows = int(np.count_nonzero(_evaluation_mask(corpus)))
    if train.get("rows") != train_rows or held_out.get("rows") != held_out_rows:
        raise ValueError("model evaluation population row counts are incomplete")
    canonical = bool(
        held_out_accepted
        if full_walking_selection
        else train_accepted and held_out_accepted
    )
    expected_tuple = [
        held_out["joint_geodesic_mae_rad"],
        held_out["fk_body_position_p95_m"],
        1.0 - min(held_out["contact_f1"]),
    ]
    if (
        receipt.get("schema") != _EVALUATION_SCHEMA
        or receipt.get("accepted") is not False
        or receipt.get("finite") is not True
        or receipt.get("corpus_manifest_sha256") != corpus_manifest_sha256
        or receipt.get("config") != _config_receipt(config)
        or receipt.get("canonical_selection_accepted") is not canonical
        or receipt.get("native_joint_limits_evaluated") is not False
        or receipt.get("runtime_acceptance")
        != "pending-native-joint-limit-and-smoke-gates"
        or receipt.get("evaluation_scope")
        != (
            "post-selection-all-row-refit-diagnostic"
            if config.fit_all_rows
            else "source-held-out-selection"
        )
        or receipt.get("selection_tuple") != expected_tuple
        or receipt.get("artifact_identity")
        != (None if artifact_identity is None else dict(artifact_identity))
        or (
            full_walking_selection
            and (
                receipt.get("selection_population") != "validation"
                or receipt.get("validation") != held_out
                or "test" in receipt
            )
        )
    ):
        raise ValueError("model canonical evaluation status is inconsistent")
    return canonical


def _validated_training_receipt(
    receipt: object,
    *,
    corpus: object,
    config: HybridModelConfig,
    corpus_manifest_sha256: str,
    canonical: bool,
    artifact_identity: Mapping[str, object] | None,
) -> None:
    if not isinstance(receipt, dict):
        # Artifact-contract violations use ValueError consistently at load seams.
        raise ValueError("model training receipt is invalid")  # noqa: TRY004
    fitted_rows = int(np.count_nonzero(_fitted_mask(corpus, config)))
    expected_batches = math.ceil(fitted_rows / config.batch_size)
    losses = receipt.get("losses")
    if (
        receipt.get("schema") != _TRAINING_SCHEMA
        or receipt.get("corpus_manifest_sha256") != corpus_manifest_sha256
        or receipt.get("config") != _config_receipt(config)
        or receipt.get("coverage_rows") != fitted_rows
        or receipt.get("normalization_rows") != fitted_rows
        or receipt.get("coverage_batches") != expected_batches
        or receipt.get("post_coverage_steps") != config.post_coverage_steps
        or receipt.get("fit_scope")
        != ("all-rows-refit" if config.fit_all_rows else "source-held-out-selection")
        or receipt.get("evaluation_accepted") is not False
        or receipt.get("canonical_selection_accepted") is not canonical
        or receipt.get("status")
        != (
            "canonical-gates-green-runtime-pending"
            if canonical
            else "canonical-gates-red"
        )
        or not isinstance(losses, list)
        or len(losses) != expected_batches + config.post_coverage_steps
        or not all(
            type(value) in (int, float) and math.isfinite(value) for value in losses
        )
        or not losses
        or receipt.get("final_loss") != losses[-1]
        or receipt.get("artifact_identity")
        != (None if artifact_identity is None else dict(artifact_identity))
    ):
        raise ValueError("model training coverage/status is inconsistent")


def _selection_model_receipt(
    selection_model: str | Path | None,
    *,
    expected_manifest_sha256: str | None,
    corpus: object,
    corpus_manifest_sha256: str,
    config: HybridModelConfig,
) -> tuple[dict[str, object], str, str]:
    if selection_model is None:
        raise ValueError("all-row refit requires an authenticated selection model")
    if not isinstance(selection_model, (str, Path)):
        raise TypeError("selection model must be an immutable model directory path")
    if expected_manifest_sha256 is None:
        raise ValueError(
            "all-row refit requires the expected selection manifest SHA-256"
        )
    if (
        type(expected_manifest_sha256) is not str
        or len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        raise ValueError("expected selection manifest SHA-256 is invalid")
    root = Path(selection_model).expanduser().resolve()
    generator = load_hybrid_generator(root, corpus=corpus, device="cpu")
    if generator.manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "selection manifest SHA-256 does not match the expected authority"
        )
    if generator.corpus_manifest_sha256 != corpus_manifest_sha256:
        raise ValueError("selection model is bound to a different corpus")
    if generator.config.fit_all_rows or not generator.canonical_selection_verified:
        raise ValueError(
            "selection model must contain a canonical-green source-held-out "
            "evaluation bound to this corpus"
        )
    if (
        generator.config.variant != config.variant
        or generator.config.dt != config.dt
        or generator.config.loss_profile != config.loss_profile
        or generator.config.loss_weights != config.loss_weights
        or generator.config.seed != config.seed
        or generator.config.batch_size != config.batch_size
        or generator.config.learning_rate != config.learning_rate
        or generator.config.weight_decay != config.weight_decay
        or generator.config.post_coverage_steps != config.post_coverage_steps
    ):
        raise ValueError("selection model config does not match the refit")
    receipt = dict(generator.evaluation_receipt)
    artifacts = generator.manifest.get("artifacts")
    evaluation_descriptor = (
        artifacts.get("evaluation.json") if isinstance(artifacts, Mapping) else None
    )
    evaluation_sha256 = (
        evaluation_descriptor.get("sha256")
        if isinstance(evaluation_descriptor, Mapping)
        else None
    )
    if type(evaluation_sha256) is not str or len(evaluation_sha256) != 64:
        raise ValueError("selection model evaluation descriptor is invalid")
    return (
        receipt,
        evaluation_sha256,
        generator.manifest_sha256,
    )


def train_hybrid_generator(
    corpus: object,
    output_directory: str | Path,
    *,
    config: HybridModelConfig | None = None,
    selection_model: str | Path | None = None,
    selection_model_manifest_sha256: str | None = None,
    post_coverage_sampler: TrainingRowSampler | None = None,
    artifact_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train one immutable variant with a mandatory full train-row coverage pass."""

    config = HybridModelConfig() if config is None else config
    if not isinstance(config, HybridModelConfig):
        raise TypeError("config must be HybridModelConfig")
    identity = _normalized_artifact_identity(artifact_identity)
    if post_coverage_sampler is not None and not callable(
        getattr(post_coverage_sampler, "sample", None)
    ):
        raise TypeError("post_coverage_sampler must implement sample()")
    _validate_corpus(corpus)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable model output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus_hash = _corpus_manifest_sha256(corpus)
    selection_receipt: dict[str, object] | None = None
    selection_sha256: str | None = None
    selection_manifest_sha256: str | None = None
    if config.fit_all_rows:
        selection_receipt, selection_sha256, selection_manifest_sha256 = (
            _selection_model_receipt(
                selection_model,
                expected_manifest_sha256=selection_model_manifest_sha256,
                corpus=corpus,
                corpus_manifest_sha256=corpus_hash,
                config=config,
            )
        )
    elif selection_model is not None or selection_model_manifest_sha256 is not None:
        raise ValueError(
            "selection model provenance is only valid for an all-row refit"
        )
    device = torch.device(config.device)
    _configure_determinism(config.seed, device)
    fitted_mask = _fitted_mask(corpus, config)
    train_rows = np.flatnonzero(fitted_mask).astype(np.int64, copy=False)
    compressor_mean, compressor_scale, output_mean, output_scale = (
        _streaming_normalization(corpus, train_rows, config.normalization_chunk_size)
    )
    compressor = _Compressor(config).to(device)
    decompressor = _Decompressor(config).to(device)
    generator = HybridGenerator(
        root=None,
        config=config,
        compressor=compressor,
        decompressor=decompressor,
        compressor_mean=_to_device(compressor_mean, device),
        compressor_scale=_to_device(compressor_scale, device),
        output_mean=_to_device(output_mean, device),
        output_scale=_to_device(output_scale, device),
        latent=None,
        corpus_manifest_sha256=corpus_hash,
        artifact_identity=(
            None if identity is None else MappingProxyType(dict(identity))
        ),
        device=device,
        manifest=MappingProxyType({}),
        training_receipt=MappingProxyType({}),
        evaluation_receipt=MappingProxyType({}),
        manifest_sha256=None,
        canonical_selection_verified=False,
        selection_provenance_verified=False,
        test_receipt_current=False,
        refit_receipt_current=False,
    )
    optimizer = torch.optim.AdamW(
        (*compressor.parameters(), *decompressor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    random = np.random.default_rng(config.seed)
    family_pools = (
        _FamilyTrainingPools.from_corpus(corpus, fitted_mask)
        if post_coverage_sampler is None
        else None
    )
    losses: list[float] = []
    coverage_batches = 0
    for rows in _iter_coverage_row_batches(train_rows, config.batch_size, random):
        losses.append(_training_step(corpus, rows, generator, optimizer))
        coverage_batches += 1
    for _ in range(config.post_coverage_steps):
        rows = (
            family_pools.sample(config.batch_size, random)
            if family_pools is not None
            else post_coverage_sampler.sample(config.batch_size, random)
        )
        rows = _row_indices(rows, _row_count(corpus))
        if len(rows) != config.batch_size or np.any(~fitted_mask[rows]):
            raise ValueError(
                "post-coverage sampler returned a wrong-sized or ineligible batch"
            )
        losses.append(_training_step(corpus, rows, generator, optimizer))
    if not losses or not np.isfinite(losses).all():
        raise FloatingPointError(
            "hybrid generator training did not produce finite losses"
        )
    training_receipt = {
        "schema": _TRAINING_SCHEMA,
        "status": "trained",
        "corpus_manifest_sha256": corpus_hash,
        "config": _config_receipt(config),
        "coverage_rows": len(train_rows),
        "coverage_batches": coverage_batches,
        "post_coverage_steps": config.post_coverage_steps,
        "losses": losses,
        "final_loss": losses[-1],
        "normalization_rows": len(train_rows),
        "streaming_cpu_to_device": True,
        "fit_scope": "all-rows-refit"
        if config.fit_all_rows
        else "source-held-out-selection",
    }
    if selection_receipt is not None:
        training_receipt["selection_evaluation_sha256"] = selection_sha256
        training_receipt["selection_model_manifest_sha256"] = selection_manifest_sha256
        training_receipt["selection_tuple"] = selection_receipt["selection_tuple"]
    if identity is not None:
        training_receipt["artifact_identity"] = identity
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        evaluation_receipt = evaluate_hybrid_generator(corpus, generator)
        if not evaluation_receipt["finite"]:
            raise FloatingPointError("hybrid generator evaluation was non-finite")
        _write_latent(staging, corpus, generator)
        _atomic_write(staging / "model.pt", _checkpoint_bytes(generator))
        canonical_green = bool(evaluation_receipt["canonical_selection_accepted"])
        training_receipt["status"] = (
            "canonical-gates-green-runtime-pending"
            if canonical_green
            else "canonical-gates-red"
        )
        training_receipt["evaluation_accepted"] = False
        training_receipt["canonical_selection_accepted"] = canonical_green
        _atomic_write(staging / "training.json", _json_bytes(training_receipt))
        _atomic_write(staging / "evaluation.json", _json_bytes(evaluation_receipt))
        artifacts = {
            name: _artifact_descriptor(staging / name) for name in _ARTIFACT_NAMES
        }
        manifest_schema = (
            _FULL_WALKING_MODEL_SCHEMA
            if identity is not None
            and identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
            else _MANIFEST_SCHEMA
        )
        manifest = {
            "schema": manifest_schema,
            "corpus_manifest_sha256": corpus_hash,
            "rows": _row_count(corpus),
            "latent_size": config.latent_size,
            "config": _config_receipt(config),
            "evaluation_accepted": False,
            "canonical_selection_accepted": canonical_green,
            "runtime_acceptance": "pending-native-joint-limit-and-smoke-gates",
            "artifacts": artifacts,
        }
        if identity is not None:
            manifest["artifact_identity"] = identity
        if selection_receipt is not None:
            manifest["selection_evaluation_sha256"] = selection_sha256
            manifest["selection_model_manifest_sha256"] = selection_manifest_sha256
        _atomic_write(staging / "manifest.json", _json_bytes(manifest))
        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if output.exists():
            raise FileExistsError(
                f"refusing to replace immutable model output: {output}"
            )
        _rename_directory_noreplace(staging, output)
        return training_receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _read_bound_artifact(root: Path, manifest: dict[str, object], name: str) -> bytes:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_ARTIFACT_NAMES):
        raise ValueError("model manifest artifact set changed")
    descriptor = artifacts.get(name)
    if not isinstance(descriptor, dict) or descriptor.get("path") != name:
        raise ValueError(f"model manifest does not bind {name}")
    path = root / name
    try:
        with path.open("rb") as source:
            payload = source.read()
    except OSError as error:
        raise ValueError(f"cannot read bound model artifact {name}") from error
    if len(payload) != descriptor.get("size_bytes") or hashlib.sha256(
        payload
    ).hexdigest() != descriptor.get("sha256"):
        raise ValueError(f"{name} does not match its immutable manifest descriptor")
    return payload


def _load_bound_latent(
    root: Path,
    manifest: dict[str, object],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    payload = _read_bound_artifact(root, manifest, "latent.npy")
    latent = np.load(io.BytesIO(payload), allow_pickle=False)
    if latent.shape != expected_shape or latent.dtype != np.float32:
        raise ValueError("published latent table shape/dtype is invalid")
    latent.setflags(write=False)
    return latent


def _selection_authority_root(root: Path, manifest_sha256: str) -> Path:
    matches: list[Path] = []
    try:
        candidates = sorted(root.parent.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise ValueError("selection authority directory cannot be inspected") from error
    for candidate in candidates:
        if candidate == root or candidate.is_symlink() or not candidate.is_dir():
            continue
        try:
            payload = (candidate / "manifest.json").read_bytes()
        except OSError:
            continue
        if hashlib.sha256(payload).hexdigest() == manifest_sha256:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "selection authority must resolve to exactly one immutable sibling model"
        )
    return matches[0]


def _full_test_receipt_authority(
    root: Path,
    *,
    expected_sha256: str,
    corpus_manifest_sha256: str,
    selection_model_manifest_sha256: str,
    selection_validation_receipt_sha256: str,
) -> tuple[Path, bytes]:
    from .full_walking_terrain_lmm_training import _parse_test_receipt_payload

    matches: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    for pattern in ("*.json", "*/*.json"):
        for candidate in sorted(root.parent.glob(pattern)):
            if candidate in seen or candidate.is_symlink() or not candidate.is_file():
                continue
            seen.add(candidate)
            try:
                payload = candidate.read_bytes()
            except OSError:
                continue
            if hashlib.sha256(payload).hexdigest() == expected_sha256:
                matches.append((candidate, payload))
    if len(matches) != 1:
        raise ValueError(
            "full walking test receipt authority must resolve to exactly one "
            "immutable JSON file"
        )
    path, payload = matches[0]
    current_payload = path.read_bytes()
    if current_payload != payload:
        raise ValueError(
            "full walking test receipt authority changed during authentication"
        )
    receipt = _parse_test_receipt_payload(payload)
    if (
        receipt.get("accepted") is not True
        or receipt.get("corpus_manifest_sha256") != corpus_manifest_sha256
        or receipt.get("selection_model_manifest_sha256")
        != selection_model_manifest_sha256
        or receipt.get("selection_validation_receipt_sha256")
        != selection_validation_receipt_sha256
    ):
        raise ValueError("full walking test receipt authority is inconsistent")
    return path, payload


def load_hybrid_generator(
    output_directory: str | Path,
    *,
    corpus: object,
    device: str = "cpu",
) -> HybridGenerator:
    """Fail-closed load a model and bind it to the exact Task 1 corpus receipt."""

    _validate_corpus(corpus)
    root = Path(output_directory).expanduser().resolve()
    manifest_payload = (root / "manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid hybrid model manifest JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") not in {
        _MANIFEST_SCHEMA,
        _FULL_WALKING_MODEL_SCHEMA,
    }:
        raise ValueError("hybrid model manifest schema is unsupported")
    if manifest_payload != _json_bytes(manifest):
        raise ValueError("hybrid model manifest is not canonical JSON")
    if {path.name for path in root.iterdir()} != set(_ARTIFACT_NAMES) | {
        "manifest.json"
    }:
        raise ValueError("hybrid model directory contains missing or extra files")
    corpus_hash = _corpus_manifest_sha256(corpus)
    if manifest.get("corpus_manifest_sha256") != corpus_hash:
        raise ValueError("model and corpus manifest SHA-256 values do not match")
    config = _config_from_receipt(manifest.get("config"))
    artifact_identity = _normalized_artifact_identity(manifest.get("artifact_identity"))
    is_full_walking_manifest = manifest.get("schema") == _FULL_WALKING_MODEL_SCHEMA
    if is_full_walking_manifest != bool(
        artifact_identity is not None
        and artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
    ):
        raise ValueError("hybrid/full-walking manifest and artifact identity disagree")
    if is_full_walking_manifest:
        from .full_walking_terrain_lmm_training import full_walking_model_identity

        stage = (
            artifact_identity.get("stage") if artifact_identity is not None else None
        )
        acceptance_profile = (
            artifact_identity.get("acceptance_profile")
            if artifact_identity is not None
            else None
        )
        if stage == "selection":
            expected_identity = full_walking_model_identity(
                corpus,
                config=config,
                stage="selection",
                acceptance_profile=acceptance_profile,
            )
        elif stage == "refit":
            expected_identity = full_walking_model_identity(
                corpus,
                config=config,
                stage="refit",
                selection_model_manifest_sha256=manifest.get(
                    "selection_model_manifest_sha256"
                ),
                test_receipt_sha256=artifact_identity.get("test_receipt_sha256")
                if isinstance(artifact_identity, Mapping)
                else None,
                acceptance_profile=acceptance_profile,
            )
        else:
            raise ValueError("full walking model stage is invalid")
        if artifact_identity != expected_identity:
            raise ValueError("full walking model identity does not match this corpus")
    if (
        manifest.get("rows") != _row_count(corpus)
        or manifest.get("latent_size") != config.latent_size
        or type(manifest.get("evaluation_accepted")) is not bool
    ):
        raise ValueError("hybrid model manifest dimensions/status are invalid")
    target_device = torch.device(device)
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("model device must be CPU or CUDA")
    model_payload = _read_bound_artifact(root, manifest, "model.pt")
    training_payload = _read_bound_artifact(root, manifest, "training.json")
    evaluation_payload = _read_bound_artifact(root, manifest, "evaluation.json")
    receipts: dict[str, dict[str, object]] = {}
    for label, payload, schema in (
        ("training", training_payload, _TRAINING_SCHEMA),
        ("evaluation", evaluation_payload, _EVALUATION_SCHEMA),
    ):
        try:
            receipt = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {label} receipt JSON") from error
        if (
            not isinstance(receipt, dict)
            or payload != _json_bytes(receipt)
            or receipt.get("schema") != schema
            or receipt.get("corpus_manifest_sha256") != corpus_hash
            or receipt.get("config") != manifest.get("config")
        ):
            raise ValueError(f"{label} receipt is not bound to this corpus")
        receipts[label] = receipt
    evaluation_receipt = receipts["evaluation"]
    training_receipt = receipts["training"]
    canonical_verified = _validated_evaluation_receipt(
        evaluation_receipt,
        corpus=corpus,
        config=config,
        corpus_manifest_sha256=corpus_hash,
        artifact_identity=artifact_identity,
    )
    _validated_training_receipt(
        training_receipt,
        corpus=corpus,
        config=config,
        corpus_manifest_sha256=corpus_hash,
        canonical=canonical_verified,
        artifact_identity=artifact_identity,
    )
    if (
        manifest.get("canonical_selection_accepted") is not canonical_verified
        or manifest.get("evaluation_accepted") is not False
        or manifest.get("runtime_acceptance")
        != "pending-native-joint-limit-and-smoke-gates"
    ):
        raise ValueError("model canonical manifest status is inconsistent")
    selection_fields = (
        "selection_evaluation_sha256",
        "selection_model_manifest_sha256",
    )
    selection_provenance_verified = False
    if config.fit_all_rows:
        if any(
            type(manifest.get(name)) is not str
            or len(manifest[name]) != 64
            or any(character not in "0123456789abcdef" for character in manifest[name])
            or training_receipt.get(name) != manifest[name]
            for name in selection_fields
        ):
            raise ValueError("all-row model selection provenance is invalid")
        selection_root = _selection_authority_root(
            root, manifest["selection_model_manifest_sha256"]
        )
        try:
            selection = load_hybrid_generator(
                selection_root, corpus=corpus, device="cpu"
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                "selection authority could not be authenticated"
            ) from error
        selection_artifacts = selection.manifest.get("artifacts")
        selection_evaluation = (
            selection_artifacts.get("evaluation.json")
            if isinstance(selection_artifacts, Mapping)
            else None
        )
        if (
            selection.manifest_sha256 != manifest["selection_model_manifest_sha256"]
            or selection.corpus_manifest_sha256 != corpus_hash
            or selection.config.fit_all_rows
            or not selection.canonical_selection_verified
            or selection.config.variant != config.variant
            or selection.config.loss_profile != config.loss_profile
            or selection.config.loss_weights != config.loss_weights
            or not isinstance(selection_evaluation, Mapping)
            or selection_evaluation.get("sha256")
            != manifest["selection_evaluation_sha256"]
        ):
            raise ValueError("selection authority does not match the all-row refit")
        selection_provenance_verified = True
    elif any(name in manifest or name in training_receipt for name in selection_fields):
        raise ValueError("selection provenance is only valid for an all-row refit")
    try:
        checkpoint = torch.load(
            io.BytesIO(model_payload), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError("invalid hybrid model checkpoint") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema") != _MODEL_SCHEMA
        or checkpoint.get("corpus_manifest_sha256") != corpus_hash
        or checkpoint.get("variant") != config.variant
        or checkpoint.get("artifact_identity") != artifact_identity
    ):
        raise ValueError("hybrid checkpoint identity does not match its manifest")
    compressor = _Compressor(config)
    decompressor = _Decompressor(config)
    try:
        compressor.load_state_dict(checkpoint["compressor"], strict=True)
        decompressor.load_state_dict(checkpoint["decompressor"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("hybrid checkpoint network shapes are invalid") from error
    if not all(
        torch.isfinite(parameter).all()
        for model in (compressor, decompressor)
        for parameter in model.parameters()
    ):
        raise ValueError("hybrid checkpoint contains non-finite parameters")
    compressor.to(target_device).eval()
    decompressor.to(target_device).eval()
    normalizations = []
    for name, shape in (
        ("compressor_mean", (_COMPRESSOR_INPUTS,)),
        ("compressor_scale", (_COMPRESSOR_INPUTS,)),
        ("output_mean", (_CONTINUOUS_OUTPUTS,)),
        ("output_scale", (_CONTINUOUS_OUTPUTS,)),
    ):
        value = checkpoint.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"hybrid checkpoint {name} is invalid")
        if name.endswith("scale") and torch.any(value <= 0.0):
            raise ValueError(f"hybrid checkpoint {name} must be positive")
        normalizations.append(value.to(target_device, dtype=torch.float32))
    latent = _load_bound_latent(
        root, manifest, (_row_count(corpus), config.latent_size)
    )
    for first in range(0, len(latent), config.evaluation_chunk_size):
        if not np.isfinite(latent[first : first + config.evaluation_chunk_size]).all():
            raise ValueError("published latent table contains non-finite values")
    full_refit_identity = bool(
        artifact_identity is not None
        and artifact_identity.get("schema") == _FULL_WALKING_MODEL_SCHEMA
        and artifact_identity.get("stage") == "refit"
        and type(artifact_identity.get("test_receipt_sha256")) is str
        and len(artifact_identity["test_receipt_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in artifact_identity["test_receipt_sha256"]
        )
    )
    test_receipt_current = False
    if full_refit_identity:
        if not selection_provenance_verified or manifest.get(
            "selection_model_manifest_sha256"
        ) != artifact_identity.get("selection_model_manifest_sha256"):
            raise ValueError("full walking refit selection provenance is inconsistent")
        _test_receipt_path, _test_receipt_payload = _full_test_receipt_authority(
            root,
            expected_sha256=artifact_identity["test_receipt_sha256"],
            corpus_manifest_sha256=corpus_hash,
            selection_model_manifest_sha256=manifest["selection_model_manifest_sha256"],
            selection_validation_receipt_sha256=manifest["selection_evaluation_sha256"],
        )
        test_receipt_current = True
    return HybridGenerator(
        root=root,
        config=config,
        compressor=compressor,
        decompressor=decompressor,
        compressor_mean=normalizations[0],
        compressor_scale=normalizations[1],
        output_mean=normalizations[2],
        output_scale=normalizations[3],
        latent=latent,
        corpus_manifest_sha256=corpus_hash,
        artifact_identity=(
            None
            if artifact_identity is None
            else MappingProxyType(dict(artifact_identity))
        ),
        device=target_device,
        manifest=MappingProxyType(manifest),
        training_receipt=MappingProxyType(training_receipt),
        evaluation_receipt=MappingProxyType(evaluation_receipt),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        canonical_selection_verified=canonical_verified,
        selection_provenance_verified=selection_provenance_verified,
        test_receipt_current=test_receipt_current,
        refit_receipt_current=(
            full_refit_identity
            and selection_provenance_verified
            and test_receipt_current
        ),
    )


def _load_cli_corpus(path: str | Path) -> object:
    try:
        from .hybrid_terrain_lmm_data import load_hybrid_cache
    except ImportError as error:
        raise RuntimeError(
            "Task 1 hybrid_terrain_lmm_data is required to load a corpus cache"
        ) from error
    return load_hybrid_cache(Path(path).expanduser().resolve())


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the 25 Hz hybrid terrain LMM generator"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train one immutable model variant")
    train.add_argument("--corpus", required=True)
    train.add_argument("--output", required=True)
    train.add_argument(
        "--variant", choices=("latent32", "latent64", "wider"), required=True
    )
    train.add_argument("--device", required=True)
    train.add_argument("--post-coverage-steps", type=int, default=20_000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--normalization-chunk-size", type=int, default=2048)
    train.add_argument("--evaluation-chunk-size", type=int, default=4096)
    train.add_argument("--loss-profile", choices=_LOSS_PROFILES, default="uniform")
    train.add_argument("--fit-all-rows", action="store_true")
    train.add_argument("--selection-model")
    train.add_argument("--selection-model-manifest-sha256")

    evaluate = commands.add_parser(
        "evaluate", help="recompute chunked full train/held-out metrics"
    )
    evaluate.add_argument("--corpus", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--chunk-size", type=int)

    inspect = commands.add_parser("inspect", help="print immutable model receipts")
    inspect.add_argument("--model", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executable entry point used for bounded GPU variants and receipt audit."""

    arguments = _build_argument_parser().parse_args(argv)
    if arguments.command == "train":
        corpus = _load_cli_corpus(arguments.corpus)
        config = HybridModelConfig(
            variant=arguments.variant,
            seed=arguments.seed,
            device=arguments.device,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            post_coverage_steps=arguments.post_coverage_steps,
            normalization_chunk_size=arguments.normalization_chunk_size,
            evaluation_chunk_size=arguments.evaluation_chunk_size,
            fit_all_rows=arguments.fit_all_rows,
            loss_profile=arguments.loss_profile,
        )
        receipt = train_hybrid_generator(
            corpus,
            arguments.output,
            config=config,
            selection_model=arguments.selection_model,
            selection_model_manifest_sha256=(arguments.selection_model_manifest_sha256),
        )
        print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if arguments.command == "evaluate":
        corpus = _load_cli_corpus(arguments.corpus)
        generator = load_hybrid_generator(
            arguments.model, corpus=corpus, device=arguments.device
        )
        receipt = evaluate_hybrid_generator(
            corpus, generator, chunk_size=arguments.chunk_size
        )
        print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
        return 0
    root = Path(arguments.model).expanduser().resolve()
    result = {}
    for label, name in (
        ("manifest", "manifest.json"),
        ("training", "training.json"),
        ("evaluation", "evaluation.json"),
    ):
        try:
            result[label] = json.loads((root / name).read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot inspect {name}") from error
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "HybridGenerator",
    "HybridModelConfig",
    "TrainingBatch",
    "TrainingRowSampler",
    "assemble_training_batch",
    "calculate_physical_metrics",
    "evaluate_hybrid_generator",
    "evaluate_hybrid_rows",
    "hybrid_generator_loss",
    "iter_coverage_batches",
    "load_hybrid_generator",
    "main",
    "sample_training_rows",
    "train_hybrid_generator",
]


if __name__ == "__main__":
    raise SystemExit(main())
