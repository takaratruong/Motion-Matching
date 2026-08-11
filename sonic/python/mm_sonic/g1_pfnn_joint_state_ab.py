"""Bounded deterministic falsification of 288 vs raw 346 G1 PFNN inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from mm_sonic.evaluate_classic_g1_pfnn_transfer import (
    joint_reconstruction_metrics,
)
from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES
from mm_sonic.terrain_pfnn.layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
)
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.train_classic_g1_pfnn import classic_pfnn_loss

EXPERIMENT_CLIPS = (
    "LocomotionFlat02_000",
    "WalkingUpSteps02_000",
    "WalkingUpSteps09_000",
    "WalkingUpSteps12_000",
)
MAXIMUM_TARGET_STEP_RAD = 0.225
LIMB_GATES = {
    "joint_mae_rad": 0.100,
    "joint_rmse_rad": 0.150,
    "frame_max_p95_rad": 0.500,
    "maximum_joint_error_rad": 1.000,
}
TreatmentName = Literal["baseline_288", "raw_346", "periodic_375"]


@dataclass(frozen=True)
class SelectedRow:
    source_index: int
    block: Literal["fit", "held_out"]
    clip_id: str
    sequence_lane: str
    center_frame_120hz: int
    mirrored: bool
    terrain_class: str
    terrain_sha256: str
    row_sha256: str


@dataclass(frozen=True)
class BlockSelection:
    fit_indices: tuple[int, ...]
    held_indices: tuple[int, ...]
    rows: tuple[SelectedRow, ...]
    receipt_sha256: str


@dataclass(frozen=True)
class ExperimentMetrics:
    sample_count: int
    joint_mae_rad: float
    joint_rmse_rad: float
    frame_max_p95_rad: float
    maximum_joint_error_rad: float
    worst_frame_index: int
    worst_joint_index: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExperimentMetrics:
        try:
            result = cls(
                sample_count=int(value["sample_count"]),
                joint_mae_rad=float(value["joint_mae_rad"]),
                joint_rmse_rad=float(value["joint_rmse_rad"]),
                frame_max_p95_rad=float(value["frame_max_p95_rad"]),
                maximum_joint_error_rad=float(value["maximum_joint_error_rad"]),
                worst_frame_index=int(value["worst_frame_index"]),
                worst_joint_index=int(value["worst_joint_index"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("experiment limb metrics are invalid") from error
        numeric = (
            result.joint_mae_rad,
            result.joint_rmse_rad,
            result.frame_max_p95_rad,
            result.maximum_joint_error_rad,
        )
        if (
            result.sample_count < 1
            or not all(np.isfinite(item) and item >= 0.0 for item in numeric)
            or not 0 <= result.worst_frame_index < result.sample_count
            or not 0 <= result.worst_joint_index < 29
        ):
            raise ValueError("experiment limb metrics are invalid")
        return result

    @property
    def gate_failures(self) -> tuple[str, ...]:
        return tuple(
            name for name, threshold in LIMB_GATES.items()
            if getattr(self, name) > threshold
        )


@dataclass(frozen=True)
class ExperimentDecision:
    outcome: str
    accepted: bool
    promotable_treatment: str | None
    periodic_scope: str


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 23456
    update_count: int = 4000
    batch_size: int = 32
    hidden_size: int = 512
    learning_rate: float = 1.0e-4
    dropout_probability: float = 0.30
    evaluation_batch_size: int = 256
    fit_rows_per_clip: int = 20
    held_rows_per_clip: int = 10
    include_periodic: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or type(self.update_count) is not int
            or self.update_count < 1
            or type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.hidden_size) is not int
            or self.hidden_size < 1
            or type(self.evaluation_batch_size) is not int
            or self.evaluation_batch_size < 1
            or type(self.fit_rows_per_clip) is not int
            or self.fit_rows_per_clip < 1
            or type(self.held_rows_per_clip) is not int
            or self.held_rows_per_clip < 1
            or type(self.include_periodic) is not bool
            or type(self.device) is not str
            or not self.device
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not math.isfinite(self.dropout_probability)
            or not 0.0 <= self.dropout_probability < 1.0
        ):
            raise ValueError("experiment training configuration is invalid")


def _array_field(arrays: object, name: str, count: int) -> np.ndarray:
    value = np.asarray(getattr(arrays, name))
    if value.shape[:1] != (count,):
        raise ValueError(f"experiment split {name} has invalid row count")
    return value


def _row_sha256(arrays: object, index: int) -> str:
    metadata = {
        "center_frame_120hz": int(arrays.center_frame_120hz[index]),
        "clip_id": str(arrays.clip_id[index]),
        "mirrored": bool(arrays.mirrored[index]),
        "sequence_lane": str(arrays.sequence_lane[index]),
        "terrain_class": str(arrays.terrain_class[index]),
        "terrain_sha256": str(arrays.terrain_sha256[index]),
    }
    digest = hashlib.sha256(b"g1-pfnn-joint-state-ab-row/v1\0")
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for value in (
        np.asarray(arrays.x[index], dtype="<f4"),
        np.asarray(arrays.y[index], dtype="<f4"),
        np.asarray(arrays.phase[index], dtype="<f4"),
    ):
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def select_fixed_blocks(
    arrays: object,
    *,
    fit_rows_per_clip: int = 20,
    held_rows_per_clip: int = 10,
) -> BlockSelection:
    """Select the earliest fixed blocks in each clip's longest retained run."""

    if (
        type(fit_rows_per_clip) is not int
        or type(held_rows_per_clip) is not int
        or fit_rows_per_clip < 1
        or held_rows_per_clip < 1
    ):
        raise ValueError("experiment block sizes must be positive integers")
    x = np.asarray(arrays.x, dtype=np.float32)
    y = np.asarray(arrays.y, dtype=np.float32)
    if (
        x.ndim != 2
        or x.shape[1] != CLASSIC_G1_INPUT_LAYOUT_V3.size
        or y.shape != (len(x), OUTPUT_LAYOUT.size)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("experiment requires finite physical v3 rows")
    count = len(x)
    for name in (
        "phase", "clip_id", "sequence_lane", "center_frame_120hz", "mirrored",
        "terrain_class", "terrain_sha256",
    ):
        _array_field(arrays, name, count)
    phase = np.asarray(arrays.phase, dtype=np.float32)
    if phase.shape != (count,) or not np.isfinite(phase).all():
        raise ValueError("experiment phase rows are invalid")

    required = fit_rows_per_clip + held_rows_per_clip
    selected_fit: list[int] = []
    selected_held: list[int] = []
    selected_rows: list[SelectedRow] = []
    joints = OUTPUT_LAYOUT["joint_position"]
    for clip in EXPERIMENT_CLIPS:
        eligible = [
            index for index in range(count)
            if str(arrays.clip_id[index]) == clip
            and str(arrays.sequence_lane[index]) == "motion"
            and not bool(arrays.mirrored[index])
        ]
        ordered = sorted(eligible, key=lambda index: int(arrays.center_frame_120hz[index]))
        centers = [int(arrays.center_frame_120hz[index]) for index in ordered]
        if len(set(centers)) != len(centers):
            raise ValueError(f"experiment contains duplicate {clip} motion rows")
        runs: list[list[int]] = []
        for index in ordered:
            if not runs:
                runs.append([index])
                continue
            previous = runs[-1][-1]
            frame = int(arrays.center_frame_120hz[index])
            previous_frame = int(arrays.center_frame_120hz[previous])
            if frame != previous_frame + 4:
                runs.append([index])
                continue
            maximum_step = float(
                np.max(np.abs(y[index, joints] - y[previous, joints]))
            )
            if maximum_step > MAXIMUM_TARGET_STEP_RAD:
                raise ValueError(
                    f"retained {clip} transition exceeds 0.225 rad at frame {frame}"
                )
            runs[-1].append(index)
        candidates = [run for run in runs if len(run) >= required]
        if not candidates:
            raise ValueError(
                f"{clip} has no retained run with {required} consecutive rows"
            )
        chosen = min(
            candidates,
            key=lambda run: (-len(run), int(arrays.center_frame_120hz[run[0]])),
        )[:required]
        fit = chosen[:fit_rows_per_clip]
        held = chosen[fit_rows_per_clip:]
        selected_fit.extend(fit)
        selected_held.extend(held)
        for block, indices in (("fit", fit), ("held_out", held)):
            for index in indices:
                selected_rows.append(
                    SelectedRow(
                        source_index=index,
                        block=block,
                        clip_id=clip,
                        sequence_lane=str(arrays.sequence_lane[index]),
                        center_frame_120hz=int(arrays.center_frame_120hz[index]),
                        mirrored=bool(arrays.mirrored[index]),
                        terrain_class=str(arrays.terrain_class[index]),
                        terrain_sha256=str(arrays.terrain_sha256[index]),
                        row_sha256=_row_sha256(arrays, index),
                    )
                )
    receipt_payload = [row.__dict__ for row in selected_rows]
    receipt = hashlib.sha256(
        b"g1-pfnn-joint-state-ab-selection/v1\0"
        + json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return BlockSelection(
        fit_indices=tuple(selected_fit),
        held_indices=tuple(selected_held),
        rows=tuple(selected_rows),
        receipt_sha256=receipt,
    )


def phase_bank_coverage(
    arrays: object, selection: BlockSelection
) -> dict[str, object]:
    """Describe primary PFNN-bank coverage without changing fixed row selection."""

    if not isinstance(selection, BlockSelection):
        raise TypeError("phase coverage selection is invalid")
    phase = np.asarray(arrays.phase, dtype=np.float32)
    if phase.ndim != 1 or not np.isfinite(phase).all():
        raise ValueError("phase coverage rows are invalid")
    banks = (
        np.floor(np.remainder(phase, np.float32(2.0 * math.pi)) * (4.0 / (2.0 * math.pi)))
        .astype(np.int64)
        % 4
    )
    per_clip: dict[str, object] = {}
    disjoint: list[str] = []
    for clip in EXPERIMENT_CLIPS:
        blocks: dict[str, list[int]] = {}
        occupied: dict[str, set[int]] = {}
        for block in ("fit", "held_out"):
            indices = np.asarray(
                [
                    row.source_index
                    for row in selection.rows
                    if row.clip_id == clip and row.block == block
                ],
                dtype=np.int64,
            )
            if len(indices) < 1 or np.any(indices < 0) or np.any(indices >= len(phase)):
                raise ValueError("phase coverage selection indices are invalid")
            counts = np.bincount(banks[indices], minlength=4).astype(np.int64)
            blocks[block] = counts.tolist()
            occupied[block] = set(np.flatnonzero(counts).tolist())
        if occupied["fit"].isdisjoint(occupied["held_out"]):
            disjoint.append(clip)
        per_clip[clip] = blocks
    return {
        "bank_count": 4,
        "primary_bank_formula": "floor((phase mod 2pi) * 4 / 2pi)",
        "per_clip": per_clip,
        "disjoint_primary_bank_clips": disjoint,
    }


def treatment_inputs(value: object, treatment: TreatmentName) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != CLASSIC_G1_INPUT_LAYOUT_V3.size
        or not np.isfinite(rows).all()
    ):
        raise ValueError("treatment input must contain finite 346-value rows")
    if treatment == "baseline_288":
        result = rows[:, : INPUT_LAYOUT.size]
    elif treatment == "raw_346":
        result = rows
    elif treatment == "periodic_375":
        q = rows[:, CLASSIC_G1_INPUT_LAYOUT_V3["joint_position"]]
        qdot = rows[:, CLASSIC_G1_INPUT_LAYOUT_V3["joint_velocity"]]
        result = np.concatenate(
            (rows[:, : INPUT_LAYOUT.size], np.sin(q), np.cos(q), qdot), axis=1
        )
    else:
        raise ValueError("unknown PFNN joint-state treatment")
    return np.ascontiguousarray(result, dtype=np.float32)


def fit_normalization(x: object, y: object) -> dict[str, np.ndarray]:
    inputs = np.asarray(x, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float32)
    if (
        inputs.ndim != 2
        or targets.shape != (len(inputs), OUTPUT_LAYOUT.size)
        or len(inputs) < 1
        or not np.isfinite(inputs).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("normalization fit rows are invalid")
    x_mean = np.mean(inputs, axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.std(inputs, axis=0, dtype=np.float64).astype(np.float32)
    y_mean = np.mean(targets, axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.std(targets, axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < np.float32(1.0e-6)] = np.float32(1.0)
    y_std[y_std < np.float32(1.0e-6)] = np.float32(1.0)
    y_mean[OUTPUT_LAYOUT["contact_logit"]] = np.float32(0.0)
    y_std[OUTPUT_LAYOUT["contact_logit"]] = np.float32(1.0)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def apply_normalization(
    x: object, y: object, normalization: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(x, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float32)
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)
    if (
        inputs.ndim != 2
        or targets.shape != (len(inputs), OUTPUT_LAYOUT.size)
        or x_mean.shape != (inputs.shape[1],)
        or x_std.shape != (inputs.shape[1],)
        or y_mean.shape != (OUTPUT_LAYOUT.size,)
        or y_std.shape != (OUTPUT_LAYOUT.size,)
        or np.any(x_std <= 0.0)
        or np.any(y_std <= 0.0)
    ):
        raise ValueError("normalization arrays are invalid")
    normalized_x = np.ascontiguousarray((inputs - x_mean) / x_std, dtype=np.float32)
    for field in ("previous_body_position", "previous_body_velocity"):
        normalized_x[:, INPUT_LAYOUT[field]] *= np.float32(0.1)
    normalized_y = np.ascontiguousarray((targets - y_mean) / y_std, dtype=np.float32)
    if not np.isfinite(normalized_x).all() or not np.isfinite(normalized_y).all():
        raise ValueError("normalized experiment rows must be finite")
    return normalized_x, normalized_y


def decide_experiment(
    *,
    baseline_fit: ExperimentMetrics,
    baseline_held: ExperimentMetrics,
    raw_fit: ExperimentMetrics,
    raw_held: ExperimentMetrics,
    periodic_fit: ExperimentMetrics | None,
    periodic_held: ExperimentMetrics | None,
    periodic_scope: str,
) -> ExperimentDecision:
    metrics = (baseline_fit, baseline_held, raw_fit, raw_held)
    if not all(isinstance(value, ExperimentMetrics) for value in metrics):
        raise TypeError("experiment decision metrics are invalid")
    if periodic_scope not in ("executed", "skipped_runtime_deadline"):
        raise ValueError("periodic treatment scope is invalid")
    if (periodic_fit is None) != (periodic_held is None):
        raise ValueError("periodic treatment metrics are incomplete")
    if periodic_scope == "executed" and periodic_fit is None:
        raise ValueError("executed periodic treatment requires metrics")
    if periodic_fit is not None and not all(
        isinstance(value, ExperimentMetrics) for value in (periodic_fit, periodic_held)
    ):
        raise TypeError("periodic treatment metrics are invalid")

    raw_passes = not raw_fit.gate_failures and not raw_held.gate_failures
    baseline_fails_held = bool(baseline_held.gate_failures)
    periodic_passes = (
        periodic_fit is not None
        and periodic_held is not None
        and not periodic_fit.gate_failures
        and not periodic_held.gate_failures
    )
    if raw_passes and baseline_fails_held:
        return ExperimentDecision(
            outcome="raw_346_accepted",
            accepted=True,
            promotable_treatment="raw_346",
            periodic_scope=periodic_scope,
        )
    if not raw_passes and periodic_passes:
        outcome = "periodic_only_stop_for_revision"
    elif not raw_passes:
        outcome = "raw_346_failed"
    else:
        outcome = "baseline_288_not_falsified"
    return ExperimentDecision(
        outcome=outcome,
        accepted=False,
        promotable_treatment=None,
        periodic_scope=periodic_scope,
    )


def deterministic_update_batches(
    count: int, *, batch_size: int, seed: int, update_count: int
) -> tuple[np.ndarray, ...]:
    """Return the exact local row ordinals used by every treatment."""

    if (
        type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(seed) is not int
        or type(update_count) is not int
        or update_count < 1
    ):
        raise ValueError("experiment batch schedule arguments are invalid")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    batches: list[np.ndarray] = []
    while len(batches) < update_count:
        order = rng.permutation(count).astype(np.int64, copy=False)
        for start in range(0, count, batch_size):
            batches.append(np.ascontiguousarray(order[start : start + batch_size]))
            if len(batches) == update_count:
                break
    return tuple(batches)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"receipt contains non-JSON {type(value).__name__}")


def _canonical_mapping_sha256(value: object, *, domain: bytes) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("joint-state provenance receipt is missing")
    try:
        encoded = json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("joint-state provenance receipt is invalid") from error
    return hashlib.sha256(domain + encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _batch_schedule_sha256(batches: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(b"g1-pfnn-joint-state-ab-batches/v1\0")
    for batch in batches:
        value = np.ascontiguousarray(batch, dtype="<i8")
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _code_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "sonic/python/mm_sonic/g1_pfnn_joint_state_ab.py",
        "sonic/python/mm_sonic/build_g1_pfnn_vertical_dataset.py",
        "sonic/python/mm_sonic/evaluate_classic_g1_pfnn_transfer.py",
        "sonic/python/mm_sonic/train_classic_g1_pfnn.py",
        "sonic/python/mm_sonic/terrain_pfnn/layout.py",
        "sonic/python/mm_sonic/terrain_pfnn/model.py",
    )
    source_files = {name: _sha256(root / name) for name in relative_paths}
    tree = hashlib.sha256(b"g1-pfnn-joint-state-ab-source-tree/v1\0")
    for name, digest in source_files.items():
        tree.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        patch = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--", *relative_paths),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("experiment code identity cannot be resolved") from error
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("experiment git commit identity is invalid")
    return {
        "git_commit": commit,
        "source_tree_sha256": tree.hexdigest(),
        "dirty_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "source_files": source_files,
    }


def _environment_identity(device: str) -> dict[str, object]:
    resolved = torch.device(device)
    cuda_device = None
    if resolved.type == "cuda":
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        cuda_device = {
            "logical_index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_device": device,
        "cuda_device": cuda_device,
    }


def _metric_record(
    value: ExperimentMetrics, indices: np.ndarray, arrays: object
) -> dict[str, object]:
    source_index = int(indices[value.worst_frame_index])
    return {
        **value.__dict__,
        "gate_failures": list(value.gate_failures),
        "accepted": not value.gate_failures,
        "worst": {
            "source_index": source_index,
            "clip_id": str(arrays.clip_id[source_index]),
            "sequence_lane": str(arrays.sequence_lane[source_index]),
            "center_frame_120hz": int(arrays.center_frame_120hz[source_index]),
            "joint_index": value.worst_joint_index,
            "joint_name": ISAACLAB_JOINT_NAMES[value.worst_joint_index],
        },
    }


def _physical_metric_records(
    prediction: np.ndarray,
    physical_y: np.ndarray,
    indices: np.ndarray,
    *,
    arrays: object,
) -> tuple[ExperimentMetrics, dict[str, object]]:
    if prediction.shape != (len(indices), OUTPUT_LAYOUT.size):
        raise ValueError("physical metric prediction shape is invalid")
    aggregate = ExperimentMetrics.from_mapping(
        joint_reconstruction_metrics(prediction, physical_y[indices])
    )
    per_clip: dict[str, object] = {}
    selected_clips = np.asarray(arrays.clip_id)[indices].astype(str)
    for clip in EXPERIMENT_CLIPS:
        mask = selected_clips == clip
        clip_indices = indices[mask]
        if len(clip_indices) < 1:
            raise ValueError(f"physical metrics contain no {clip} rows")
        metrics = ExperimentMetrics.from_mapping(
            joint_reconstruction_metrics(
                prediction[mask], physical_y[clip_indices]
            )
        )
        per_clip[clip] = _metric_record(metrics, clip_indices, arrays)
    return aggregate, {
        "aggregate": _metric_record(aggregate, indices, arrays),
        "per_clip": per_clip,
    }


def joint_state_oracle_diagnostics(
    arrays: object, selection: BlockSelection
) -> dict[str, object]:
    """Evaluate copy and one-step constant-velocity joint-state observability."""

    if not isinstance(selection, BlockSelection):
        raise TypeError("joint-state oracle selection is invalid")
    x = np.asarray(arrays.x, dtype=np.float32)
    physical_y = np.asarray(arrays.y, dtype=np.float32)
    if (
        x.ndim != 2
        or x.shape[1] != CLASSIC_G1_INPUT_LAYOUT_V3.size
        or physical_y.shape != (len(x), OUTPUT_LAYOUT.size)
        or not np.isfinite(x).all()
        or not np.isfinite(physical_y).all()
    ):
        raise ValueError("joint-state oracle rows are invalid")
    q = x[:, CLASSIC_G1_INPUT_LAYOUT_V3["joint_position"]]
    qdot = x[:, CLASSIC_G1_INPUT_LAYOUT_V3["joint_velocity"]]
    joints = OUTPUT_LAYOUT["joint_position"]
    result: dict[str, object] = {}
    for name, predicted_q in (
        ("q_copy", q),
        ("q_plus_qdot_dt", q + qdot / np.float32(30.0)),
    ):
        blocks: dict[str, object] = {}
        for block, selected in (
            ("fit", selection.fit_indices),
            ("held_out", selection.held_indices),
        ):
            indices = np.asarray(selected, dtype=np.int64)
            prediction = np.ascontiguousarray(physical_y[indices].copy())
            prediction[:, joints] = predicted_q[indices]
            _, blocks[block] = _physical_metric_records(
                prediction, physical_y, indices, arrays=arrays
            )
        result[name] = blocks
    return {"fps": 30.0, **result}


def _evaluate_treatment(
    model: PhaseFunctionedNetwork,
    normalized_x: np.ndarray,
    physical_y: np.ndarray,
    phase: np.ndarray,
    normalization: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    arrays: object,
    device: torch.device,
    batch_size: int,
) -> tuple[ExperimentMetrics, dict[str, object]]:
    prediction = np.empty((len(indices), OUTPUT_LAYOUT.size), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            prediction[start : start + len(selected)] = (
                model(
                    torch.as_tensor(normalized_x[selected], device=device),
                    torch.as_tensor(phase[selected], device=device),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
    predicted_physical = np.ascontiguousarray(
        prediction * normalization["y_std"] + normalization["y_mean"],
        dtype=np.float32,
    )
    return _physical_metric_records(
        predicted_physical, physical_y, indices, arrays=arrays
    )


def _train_treatment(
    treatment: TreatmentName,
    *,
    arrays: object,
    selection: BlockSelection,
    output: Path,
    config: ExperimentConfig,
    dataset_sha256: str,
    dataset_manifest_sha256: str,
    joint_state_receipt_sha256: str,
    batch_schedule: tuple[np.ndarray, ...],
) -> tuple[dict[str, object], ExperimentMetrics, ExperimentMetrics]:
    joint_state_receipt_sha256 = _digest(
        joint_state_receipt_sha256, "joint-state receipt"
    )
    inputs = treatment_inputs(arrays.x, treatment)
    physical_y = np.ascontiguousarray(np.asarray(arrays.y), dtype=np.float32)
    phase = np.ascontiguousarray(np.asarray(arrays.phase), dtype=np.float32)
    fit_indices = np.asarray(selection.fit_indices, dtype=np.int64)
    held_indices = np.asarray(selection.held_indices, dtype=np.int64)
    normalization = fit_normalization(inputs[fit_indices], physical_y[fit_indices])
    normalized_x, normalized_y = apply_normalization(
        inputs, physical_y, normalization
    )
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = PhaseFunctionedNetwork(
        hidden_size=config.hidden_size,
        dropout_probability=config.dropout_probability,
        input_size=inputs.shape[1],
    ).to(device)
    dropout_seed = config.seed + 1
    torch.manual_seed(dropout_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(dropout_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    final_loss = math.nan
    for local_indices in batch_schedule:
        indices = fit_indices[local_indices]
        optimizer.zero_grad(set_to_none=True)
        loss = classic_pfnn_loss(
            model(
                torch.as_tensor(normalized_x[indices], device=device),
                torch.as_tensor(phase[indices], device=device),
            ),
            torch.as_tensor(normalized_y[indices], device=device),
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    fit_metrics, fit_record = _evaluate_treatment(
        model,
        normalized_x,
        physical_y,
        phase,
        normalization,
        fit_indices,
        arrays=arrays,
        device=device,
        batch_size=config.evaluation_batch_size,
    )
    held_metrics, held_record = _evaluate_treatment(
        model,
        normalized_x,
        physical_y,
        phase,
        normalization,
        held_indices,
        arrays=arrays,
        device=device,
        batch_size=config.evaluation_batch_size,
    )
    checkpoint_path = output / f"{treatment}.pt"
    checkpoint_payload = {
        "schema": "g1-pfnn-joint-state-ab-checkpoint/v1",
        "treatment": treatment,
        "input_size": inputs.shape[1],
        "output_size": OUTPUT_LAYOUT.size,
        "model_config": {
            "hidden_size": config.hidden_size,
            "dropout_probability": config.dropout_probability,
        },
        "model_state": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        },
        "normalization": {
            name: torch.as_tensor(value).detach().cpu().contiguous().clone()
            for name, value in normalization.items()
        },
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "joint_state_receipt_sha256": joint_state_receipt_sha256,
        "selection_receipt_sha256": selection.receipt_sha256,
        "seed": config.seed,
        "dropout_seed": dropout_seed,
        "update_count": config.update_count,
        "batch_size": config.batch_size,
        "batch_schedule_sha256": _batch_schedule_sha256(batch_schedule),
        "learning_rate": config.learning_rate,
        "optimizer": "Adam",
        "representation_class_candidate": treatment == "raw_346",
        "checkpoint_promotion_eligible": False,
        "viewer_checkpoint": False,
    }
    _atomic_checkpoint(checkpoint_path, checkpoint_payload)
    metrics_payload = {
        "schema": "g1-pfnn-joint-state-ab-metrics/v1",
        "treatment": treatment,
        "fit": fit_record,
        "held_out": held_record,
        "final_training_loss": final_loss,
        "thresholds": dict(LIMB_GATES),
    }
    metrics_path = output / f"{treatment}-metrics.json"
    _atomic_json(metrics_path, metrics_payload)
    record = {
        "input_size": inputs.shape[1],
        "diagnostic": treatment != "raw_346",
        "representation_class_candidate": treatment == "raw_346",
        "checkpoint_promotion_eligible": False,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
        },
        "metrics_artifact": {
            "path": metrics_path.name,
            "sha256": _sha256(metrics_path),
        },
        "metrics": metrics_payload,
    }
    return record, fit_metrics, held_metrics


def run_experiment(
    dataset: object,
    *,
    output: str | Path,
    config: ExperimentConfig | None = None,
    dataset_manifest_sha256: str,
) -> dict[str, object]:
    """Train and seal equal-budget treatments over one loaded v3 dataset."""

    if config is None:
        config = ExperimentConfig()
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    torch.use_deterministic_algorithms(True)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {destination}")
    destination.mkdir(parents=True)
    dataset_sha256 = _digest(getattr(dataset, "dataset_sha256", None), "dataset")
    manifest_sha256 = _digest(dataset_manifest_sha256, "dataset manifest")
    splits = getattr(dataset, "splits", None)
    if not isinstance(splits, Mapping) or "train" not in splits:
        raise ValueError("experiment dataset has no train split")
    arrays = splits["train"]
    selection = select_fixed_blocks(
        arrays,
        fit_rows_per_clip=config.fit_rows_per_clip,
        held_rows_per_clip=config.held_rows_per_clip,
    )
    source_provenance = {
        name: _digest(getattr(dataset, name, None), name)
        for name in (
            "selection_sha256",
            "retarget_manifest_sha256",
            "terrain_receipt_set_sha256",
        )
    }
    source_provenance["joint_state_receipt_sha256"] = _canonical_mapping_sha256(
        getattr(dataset, "joint_state_receipt", None),
        domain=b"g1-pfnn-joint-state-receipt-provenance/v1\0",
    )
    source_roles = dict(getattr(dataset, "source_roles", {}))
    if any(type(name) is not str or type(role) is not str for name, role in source_roles.items()):
        raise ValueError("experiment source roles are invalid")
    selection_payload = {
        "schema": "g1-pfnn-joint-state-ab-selection/v1",
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "source_provenance": source_provenance,
        "fit_rows_per_clip": config.fit_rows_per_clip,
        "held_rows_per_clip": config.held_rows_per_clip,
        "maximum_target_step_rad": MAXIMUM_TARGET_STEP_RAD,
        "receipt_sha256": selection.receipt_sha256,
        "rows": [row.__dict__ for row in selection.rows],
    }
    selection_path = destination / "selection.json"
    _atomic_json(selection_path, selection_payload)
    code_identity = _code_identity()
    environment_identity = _environment_identity(config.device)
    phase_coverage = phase_bank_coverage(arrays, selection)
    joint_state_oracles = joint_state_oracle_diagnostics(arrays, selection)
    batch_schedule = deterministic_update_batches(
        len(selection.fit_indices),
        batch_size=config.batch_size,
        seed=config.seed,
        update_count=config.update_count,
    )
    treatments: dict[str, object] = {}
    treatment_metrics: dict[str, tuple[ExperimentMetrics, ExperimentMetrics]] = {}
    names: tuple[TreatmentName, ...] = (
        ("baseline_288", "raw_346", "periodic_375")
        if config.include_periodic
        else ("baseline_288", "raw_346")
    )
    for treatment in names:
        record, fit_metrics, held_metrics = _train_treatment(
            treatment,
            arrays=arrays,
            selection=selection,
            output=destination,
            config=config,
            dataset_sha256=dataset_sha256,
            dataset_manifest_sha256=manifest_sha256,
            joint_state_receipt_sha256=source_provenance[
                "joint_state_receipt_sha256"
            ],
            batch_schedule=batch_schedule,
        )
        treatments[treatment] = record
        treatment_metrics[treatment] = (fit_metrics, held_metrics)
    periodic_scope = "executed" if config.include_periodic else "skipped_runtime_deadline"
    periodic = treatment_metrics.get("periodic_375")
    decision = decide_experiment(
        baseline_fit=treatment_metrics["baseline_288"][0],
        baseline_held=treatment_metrics["baseline_288"][1],
        raw_fit=treatment_metrics["raw_346"][0],
        raw_held=treatment_metrics["raw_346"][1],
        periodic_fit=None if periodic is None else periodic[0],
        periodic_held=None if periodic is None else periodic[1],
        periodic_scope=periodic_scope,
    )
    ending_code_identity = _code_identity()
    if (
        ending_code_identity["source_tree_sha256"]
        != code_identity["source_tree_sha256"]
    ):
        raise ValueError("experiment source tree changed during training")
    code_identity["source_tree_verified_unchanged"] = True
    oracle_passes = all(
        joint_state_oracles[name][block]["aggregate"]["accepted"]
        for name in ("q_copy", "q_plus_qdot_dt")
        for block in ("fit", "held_out")
    )
    if decision.outcome == "raw_346_failed" and oracle_passes:
        failure_interpretation = (
            "learned_candidate_failed_despite_passing_joint_state_oracles"
        )
    else:
        failure_interpretation = None
    report = {
        "schema": "g1-pfnn-joint-state-ab/v1",
        "dataset": {
            "dataset_sha256": dataset_sha256,
            "manifest_sha256": manifest_sha256,
            "source_provenance": source_provenance,
            "source_roles": dict(sorted(source_roles.items())),
        },
        "code": code_identity,
        "environment": environment_identity,
        "selection": {
            "artifact": {
                "path": selection_path.name,
                "sha256": _sha256(selection_path),
            },
            "receipt_sha256": selection.receipt_sha256,
            "fit_row_count": len(selection.fit_indices),
            "held_out_row_count": len(selection.held_indices),
        },
        "training": {
            "seed": config.seed,
            "dropout_seed": config.seed + 1,
            "update_count": config.update_count,
            "batch_size": config.batch_size,
            "batch_schedule_sha256": _batch_schedule_sha256(batch_schedule),
            "hidden_size": config.hidden_size,
            "learning_rate": config.learning_rate,
            "optimizer": "Adam",
            "dropout_probability": config.dropout_probability,
            "device": config.device,
            "torch_deterministic_algorithms": True,
        },
        "thresholds": dict(LIMB_GATES),
        "diagnostics": {
            "phase_bank_coverage": phase_coverage,
            "joint_state_oracles": joint_state_oracles,
            "failure_interpretation": failure_interpretation,
        },
        "periodic_scope": periodic_scope,
        "treatments": treatments,
        "decision": decision.__dict__,
    }
    _atomic_json(destination / "experiment.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=23456)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--dropout-probability", type=float, default=0.30)
    parser.add_argument("--fit-rows-per-clip", type=int, default=20)
    parser.add_argument("--held-rows-per-clip", type=int, default=10)
    parser.add_argument("--skip-periodic-runtime-deadline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from mm_sonic.build_g1_pfnn_vertical_dataset import load_vertical_dataset

    root = arguments.dataset.expanduser().resolve(strict=True)
    root = root if root.is_dir() else root.parent
    manifest = root / "manifest.json"
    dataset = load_vertical_dataset(root)
    torch.use_deterministic_algorithms(True)
    report = run_experiment(
        dataset,
        output=arguments.output,
        config=ExperimentConfig(
            seed=arguments.seed,
            update_count=arguments.updates,
            batch_size=arguments.batch_size,
            hidden_size=arguments.hidden_size,
            learning_rate=arguments.learning_rate,
            dropout_probability=arguments.dropout_probability,
            evaluation_batch_size=arguments.evaluation_batch_size,
            fit_rows_per_clip=arguments.fit_rows_per_clip,
            held_rows_per_clip=arguments.held_rows_per_clip,
            include_periodic=not arguments.skip_periodic_runtime_deadline,
            device=arguments.device,
        ),
        dataset_manifest_sha256=_sha256(manifest),
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["decision"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
