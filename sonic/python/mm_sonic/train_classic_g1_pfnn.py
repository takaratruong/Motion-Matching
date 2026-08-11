"""Train the published one-step PFNN objective on native G1 motion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .terrain_pfnn.dataset import (
    PFNNShardDataset,
    normalize_pfnn_input,
    normalize_pfnn_output,
)
from .build_g1_pfnn_vertical_dataset import (
    VerticalDataset,
    _flat_terrain_sha256,
    load_vertical_dataset,
)
from .terrain_pfnn.features import PFNNTrainingWindow, mirror_window
from .terrain_pfnn.kinematics import TorchG1ForwardKinematics
from .terrain_pfnn.layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
)
from .terrain_pfnn.model import PhaseFunctionedNetwork
from .terrain_pfnn.training import (
    _validate_runtime_seed,
    choose_runtime_seed,
    training_phase_advance_q99,
)
from .terrain_oracle.canonical import ISAACLAB_JOINT_NAMES


CLASSIC_CHECKPOINT_SCHEMA = "classic-g1-pfnn/v3"
_SHA256_CHARS = frozenset("0123456789abcdef")


def classic_pfnn_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Original one-step regression plus binary heel/toe contact supervision."""

    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != OUTPUT_LAYOUT.size
        or prediction.dtype != target.dtype
        or prediction.device != target.device
        or not bool(torch.isfinite(prediction).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("classic PFNN prediction and target must be finite [B,268] tensors")
    contact = OUTPUT_LAYOUT["contact_logit"]
    labels = target[:, contact]
    if not bool(torch.all((labels == 0.0) | (labels == 1.0))):
        raise ValueError("classic PFNN contact targets must be binary")
    return F.mse_loss(prediction[:, : contact.start], target[:, : contact.start]) + F.binary_cross_entropy_with_logits(
        prediction[:, contact], labels
    )


def balanced_epoch_batches(
    source_kind: object,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    source_filter: str = "mixed",
) -> tuple[np.ndarray, ...]:
    """Return deterministic native-G1 or half-GRAIL/half-LAFAN batches."""

    kinds = np.asarray(source_kind)
    names = set(kinds.astype(str).tolist()) if kinds.ndim == 1 else set()
    if (
        kinds.ndim != 1
        or source_filter not in ("grail", "mixed")
        or "grail" not in names
        or (source_filter == "mixed" and names != {"grail", "lafan"})
    ):
        raise ValueError("classic PFNN source kinds must contain grail and lafan")
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError("classic PFNN batch size must be a positive even integer")
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("classic PFNN sampler seed/epoch are invalid")
    half = batch_size // 2
    rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
    if source_filter == "grail":
        indices = rng.permutation(np.flatnonzero(kinds.astype(str) == "grail"))
        batch_count = math.ceil(len(indices) / batch_size)
        expanded = np.resize(indices, batch_count * batch_size)
        return tuple(
            expanded[index * batch_size : (index + 1) * batch_size].astype(
                np.int64, copy=False
            )
            for index in range(batch_count)
        )
    groups = {
        name: rng.permutation(np.flatnonzero(kinds.astype(str) == name))
        for name in ("grail", "lafan")
    }
    batch_count = max(math.ceil(len(indices) / half) for indices in groups.values())
    expanded = {
        name: np.resize(indices, batch_count * half)
        for name, indices in groups.items()
    }
    batches: list[np.ndarray] = []
    for index in range(batch_count):
        start, stop = index * half, (index + 1) * half
        batch = np.concatenate(
            (expanded["grail"][start:stop], expanded["lafan"][start:stop])
        )
        batches.append(rng.permutation(batch).astype(np.int64, copy=False))
    return tuple(batches)


def vertical_epoch_batches(
    count: int, *, batch_size: int, seed: int, epoch: int
) -> tuple[np.ndarray, ...]:
    """Shuffle every released-PFNN row exactly once per deterministic epoch."""

    if (
        type(count) is not int
        or count < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(seed) is not int
        or type(epoch) is not int
        or epoch < 0
    ):
        raise ValueError("released PFNN sampler arguments are invalid")
    rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
    indices = rng.permutation(count).astype(np.int64, copy=False)
    return tuple(indices[start : start + batch_size] for start in range(0, count, batch_size))


def _plain_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value).detach().to(device="cpu").contiguous().clone()
    if isinstance(value, Mapping):
        return {str(key): _plain_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_cpu(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"classic PFNN checkpoint contains unsafe {type(value).__name__}")


def _finite_tree(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(type(key) is str and _finite_tree(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return value is None or type(value) in (str, int, bool)


def _validate_classic_runtime_seed_binding(
    seed: Mapping[str, object], normalization: Mapping[str, torch.Tensor]
) -> None:
    normalized = torch.as_tensor(seed["normalized_input"], dtype=torch.float32)
    for field, seed_name in (
        ("joint_position", "joint_position"),
        ("joint_velocity", "joint_velocity"),
    ):
        section = CLASSIC_G1_INPUT_LAYOUT_V3[field]
        expected = (
            torch.as_tensor(seed[seed_name], dtype=torch.float32)
            - normalization["x_mean"][section]
        ) / normalization["x_std"][section]
        if not torch.allclose(
            normalized[section], expected, rtol=0.0, atol=3.0e-5
        ):
            raise ValueError(
                f"classic PFNN runtime seed {field} binding is invalid"
            )


@dataclass(frozen=True)
class ClassicG1PFNNCheckpoint:
    model_state: dict[str, torch.Tensor]
    input_size: int
    hidden_size: int
    dropout_probability: float
    normalization: dict[str, torch.Tensor]
    runtime_seed: dict[str, object]
    dataset_digest: str
    kinematic_signature_sha256: str
    joint_limits: torch.Tensor
    phase_advance_q99: float
    epoch: int
    validation_loss: float
    seed: int
    source_kind: str
    vertical_slice_receipt_sha256: str
    terrain_receipt_set_sha256: str

    def build_model(self) -> PhaseFunctionedNetwork:
        model = PhaseFunctionedNetwork(
            hidden_size=self.hidden_size,
            dropout_probability=self.dropout_probability,
            input_size=self.input_size,
        )
        model.load_state_dict(self.model_state, strict=True)
        return model


def save_classic_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    normalization: Mapping[str, object],
    runtime_seed: Mapping[str, object],
    dataset_digest: str,
    kinematic_signature_sha256: str,
    joint_limits: object,
    phase_advance_q99: float,
    epoch: int,
    validation_loss: float,
    seed: int,
    source_kind: str,
    vertical_slice_receipt_sha256: str,
    terrain_receipt_set_sha256: str,
) -> None:
    if not isinstance(model, PhaseFunctionedNetwork):
        raise TypeError("classic PFNN checkpoint model is invalid")
    if (
        model.input_size != CLASSIC_G1_INPUT_LAYOUT_V3.size
        or model.W0.shape[2] != CLASSIC_G1_INPUT_LAYOUT_V3.size
    ):
        raise ValueError("classic PFNN checkpoint input contract is invalid")
    if (
        type(dataset_digest) is not str
        or len(dataset_digest) != 64
        or not set(dataset_digest) <= _SHA256_CHARS
        or type(kinematic_signature_sha256) is not str
        or len(kinematic_signature_sha256) != 64
        or not set(kinematic_signature_sha256) <= _SHA256_CHARS
        or type(epoch) is not int
        or epoch < 0
        or type(seed) is not int
        or source_kind not in ("grail", "mixed", "released_pfnn")
        or type(vertical_slice_receipt_sha256) is not str
        or len(vertical_slice_receipt_sha256) != 64
        or not set(vertical_slice_receipt_sha256) <= _SHA256_CHARS
        or type(terrain_receipt_set_sha256) is not str
        or len(terrain_receipt_set_sha256) != 64
        or not set(terrain_receipt_set_sha256) <= _SHA256_CHARS
    ):
        raise ValueError("classic PFNN checkpoint provenance is invalid")
    score = float(validation_loss)
    q99 = float(phase_advance_q99)
    if not math.isfinite(score) or score < 0.0 or not math.isfinite(q99) or q99 <= 0.0:
        raise ValueError("classic PFNN checkpoint metrics are invalid")
    expected_normal = {
        "x_mean": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "x_std": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "y_mean": OUTPUT_LAYOUT.size,
        "y_std": OUTPUT_LAYOUT.size,
    }
    normal = {
        name: torch.tensor(
            np.array(normalization[name], dtype=np.float32, copy=True),
            dtype=torch.float32,
            device="cpu",
        ).contiguous()
        for name in expected_normal
    }
    if any(normal[name].shape != (width,) for name, width in expected_normal.items()):
        raise ValueError("classic PFNN normalization shape is invalid")
    if any(not bool(torch.isfinite(value).all()) for value in normal.values()) or any(
        not bool(torch.all(normal[name] > 0.0)) for name in ("x_std", "y_std")
    ):
        raise ValueError("classic PFNN normalization is invalid")
    limits = torch.as_tensor(joint_limits, dtype=torch.float64, device="cpu").contiguous().clone()
    if limits.shape != (29, 2) or not bool(torch.all(limits[:, 0] < limits[:, 1])):
        raise ValueError("classic PFNN joint limits are invalid")
    checked_runtime_seed = _validate_runtime_seed(
        dict(runtime_seed),
        limits,
        input_layout=CLASSIC_G1_INPUT_LAYOUT_V3,
    )
    _validate_classic_runtime_seed_binding(checked_runtime_seed, normal)
    payload = {
        "schema": CLASSIC_CHECKPOINT_SCHEMA,
        "input_size": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "output_size": OUTPUT_LAYOUT.size,
        "input_layout": [list(field) for field in CLASSIC_G1_INPUT_LAYOUT_V3.fields],
        "output_layout": [list(field) for field in OUTPUT_LAYOUT.fields],
        "canonical_joint_order": list(ISAACLAB_JOINT_NAMES),
        "model_config": {
            "hidden_size": int(model.W0.shape[1]),
            "dropout_probability": float(model.dropout.p),
        },
        "model_state": dict(model.state_dict()),
        "normalization": normal,
        "runtime_seed": checked_runtime_seed,
        "dataset_digest": dataset_digest,
        "kinematic_signature_sha256": kinematic_signature_sha256,
        "joint_limits": limits,
        "phase_advance_q99": q99,
        "epoch": epoch,
        "validation_loss": score,
        "seed": seed,
        "source_kind": source_kind,
        "vertical_slice_receipt_sha256": vertical_slice_receipt_sha256,
        "terrain_receipt_set_sha256": terrain_receipt_set_sha256,
    }
    plain = _plain_cpu(payload)
    if not isinstance(plain, dict) or not _finite_tree(plain):
        raise ValueError("classic PFNN checkpoint must be finite")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(plain, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_classic_checkpoint(path: str | Path) -> ClassicG1PFNNCheckpoint:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("classic PFNN checkpoint cannot be loaded safely") from error
    required = {
        "schema", "input_size", "output_size", "input_layout", "output_layout",
        "canonical_joint_order", "model_config", "model_state", "normalization", "runtime_seed",
        "dataset_digest", "kinematic_signature_sha256", "joint_limits",
        "phase_advance_q99", "epoch", "validation_loss", "seed",
        "source_kind", "vertical_slice_receipt_sha256",
        "terrain_receipt_set_sha256",
    }
    if type(payload) is not dict or set(payload) != required or payload["schema"] != CLASSIC_CHECKPOINT_SCHEMA:
        raise ValueError("classic PFNN checkpoint schema is invalid")
    expected_contract = {
        "input_size": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "output_size": OUTPUT_LAYOUT.size,
        "input_layout": [list(field) for field in CLASSIC_G1_INPUT_LAYOUT_V3.fields],
        "output_layout": [list(field) for field in OUTPUT_LAYOUT.fields],
        "canonical_joint_order": list(ISAACLAB_JOINT_NAMES),
    }
    if any(payload[name] != value for name, value in expected_contract.items()):
        raise ValueError("classic PFNN checkpoint immutable contract mismatch")
    config = payload["model_config"]
    if (
        type(config) is not dict
        or set(config) != {"hidden_size", "dropout_probability"}
        or type(config["hidden_size"]) is not int
        or config["hidden_size"] < 1
        or type(config["dropout_probability"]) is not float
        or not 0.0 <= config["dropout_probability"] < 1.0
    ):
        raise ValueError("classic PFNN model config is invalid")
    normalization = payload["normalization"]
    expected_normal = {
        "x_mean": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "x_std": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "y_mean": OUTPUT_LAYOUT.size,
        "y_std": OUTPUT_LAYOUT.size,
    }
    if type(normalization) is not dict or set(normalization) != set(expected_normal):
        raise ValueError("classic PFNN normalization is invalid")
    if any(
        not isinstance(normalization[name], torch.Tensor)
        or normalization[name].shape != (width,)
        or not bool(torch.isfinite(normalization[name]).all())
        for name, width in expected_normal.items()
    ) or any(
        not bool(torch.all(normalization[name] > 0.0))
        for name in ("x_std", "y_std")
    ):
        raise ValueError("classic PFNN normalization is invalid")
    limits = payload["joint_limits"]
    if (
        not isinstance(limits, torch.Tensor)
        or limits.shape != (29, 2)
        or not bool(torch.isfinite(limits).all())
        or not bool(torch.all(limits[:, 0] < limits[:, 1]))
    ):
        raise ValueError("classic PFNN joint limits are invalid")
    runtime_seed = _validate_runtime_seed(
        payload["runtime_seed"],
        limits.to(dtype=torch.float64),
        exact_tensors=True,
        input_layout=CLASSIC_G1_INPUT_LAYOUT_V3,
    )
    _validate_classic_runtime_seed_binding(runtime_seed, normalization)
    if (
        payload["source_kind"] not in ("grail", "mixed", "released_pfnn")
        or type(payload["vertical_slice_receipt_sha256"]) is not str
        or len(payload["vertical_slice_receipt_sha256"]) != 64
        or not set(payload["vertical_slice_receipt_sha256"]) <= _SHA256_CHARS
        or type(payload["terrain_receipt_set_sha256"]) is not str
        or len(payload["terrain_receipt_set_sha256"]) != 64
        or not set(payload["terrain_receipt_set_sha256"]) <= _SHA256_CHARS
    ):
        raise ValueError("classic PFNN source provenance is invalid")
    checkpoint = ClassicG1PFNNCheckpoint(
        model_state=payload["model_state"],
        input_size=payload["input_size"],
        hidden_size=config["hidden_size"],
        dropout_probability=config["dropout_probability"],
        normalization=payload["normalization"],
        runtime_seed=runtime_seed,
        dataset_digest=payload["dataset_digest"],
        kinematic_signature_sha256=payload["kinematic_signature_sha256"],
        joint_limits=payload["joint_limits"],
        phase_advance_q99=float(payload["phase_advance_q99"]),
        epoch=payload["epoch"],
        validation_loss=float(payload["validation_loss"]),
        seed=payload["seed"],
        source_kind=payload["source_kind"],
        vertical_slice_receipt_sha256=payload[
            "vertical_slice_receipt_sha256"
        ],
        terrain_receipt_set_sha256=payload["terrain_receipt_set_sha256"],
    )
    if not _finite_tree(payload):
        raise ValueError("classic PFNN checkpoint contains nonfinite values")
    checkpoint.build_model()
    return checkpoint


def _materialize(
    dataset: PFNNShardDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((len(dataset), INPUT_LAYOUT.size), dtype=np.float32)
    y = np.empty((len(dataset), OUTPUT_LAYOUT.size), dtype=np.float32)
    phase = np.empty(len(dataset), dtype=np.float32)
    source = np.empty(len(dataset), dtype="<U6")
    clip = np.empty(len(dataset), dtype="<U96")
    for index in range(len(dataset)):
        row = dataset[index]
        x[index] = row["x"]
        y[index] = row["y"]
        phase[index] = row["phase"]
        source[index] = "grail" if str(row["clip_id"]).startswith("terrain_slopes__") else "lafan"
        clip[index] = str(row["clip_id"])
    return x, y, phase, source, clip


class _IndexedTrainDataset:
    """Read-only train view used to bind the runtime seed to optimized rows."""

    split = "train"

    def __init__(self, dataset: PFNNShardDataset, indices: np.ndarray) -> None:
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=np.int64)
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            setattr(self, name, getattr(dataset, name))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._dataset[int(self._indices[index])]


class _VerticalTrainingView:
    """Normalized row view over one compact released-PFNN split."""

    def __init__(self, dataset: VerticalDataset, split: str) -> None:
        if split not in ("train", "validation"):
            raise ValueError("vertical training split is invalid")
        self.split = split
        self._dataset = dataset
        self._arrays = dataset.splits[split]
        self.x_mean = dataset.x_mean
        self.x_std = dataset.x_std
        self.y_mean = dataset.y_mean
        self.y_std = dataset.y_std

    def __len__(self) -> int:
        return len(self._arrays.phase)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = int(index)
        clip = str(self._arrays.clip_id[row])
        return {
            "x": normalize_pfnn_input(
                self._arrays.x[row], self.x_mean, self.x_std
            ),
            "y": normalize_pfnn_output(
                self._arrays.y[row], self.y_mean, self.y_std
            ),
            "phase": float(self._arrays.phase[row]),
            "clip_id": clip,
            "center_frame": int(self._arrays.center_frame_120hz[row]) // 4,
            "split_identity": clip.removesuffix("__mirror"),
            "split": self.split,
            "sequence_lane": str(self._arrays.sequence_lane[row]),
            "terrain_class": str(self._arrays.terrain_class[row]),
            "terrain_sha256": str(self._arrays.terrain_sha256[row]),
            "terrain_fitted": str(self._arrays.terrain_sha256[row])
            != _flat_terrain_sha256(),
            "mirrored": bool(self._arrays.mirrored[row]),
            "root_world_xy": self._arrays.root_world_xy[row].copy(),
            "root_world_yaw": float(self._arrays.root_world_yaw[row]),
        }


def _released_runtime_seed_view(dataset: VerticalDataset) -> _IndexedTrainDataset:
    """Keep appended GRAIL rows from replacing the released-PFNN bootstrap."""

    view = _VerticalTrainingView(dataset, "train")
    indices = np.asarray(
        [
            index
            for index in range(len(view))
            if not str(view[index]["clip_id"]).startswith("terrain_slopes__")
        ],
        dtype=np.int64,
    )
    if len(indices) == 0:
        raise ValueError("released PFNN corpus contains no runtime seed rows")
    return _IndexedTrainDataset(view, indices)


def _materialize_vertical(
    view: _VerticalTrainingView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty(
        (len(view), CLASSIC_G1_INPUT_LAYOUT_V3.size), dtype=np.float32
    )
    y = np.empty((len(view), OUTPUT_LAYOUT.size), dtype=np.float32)
    phase = np.empty(len(view), dtype=np.float32)
    for index in range(len(view)):
        row = view[index]
        x[index] = row["x"]
        y[index] = row["y"]
        phase[index] = row["phase"]
    return x, y, phase


def _grail_train_validation_masks(
    source_kind: object, clip_id: object
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the lexicographically last present variant of each family."""

    source = np.asarray(source_kind).astype(str)
    clips = np.asarray(clip_id).astype(str)
    if source.ndim != 1 or clips.shape != source.shape:
        raise ValueError("classic PFNN source/clip metadata is invalid")
    grail_ids = sorted(set(clips[source == "grail"].tolist()))
    families: dict[str, list[str]] = {}
    for value in grail_ids:
        if not value.startswith("terrain_slopes__") or "__" not in value:
            raise ValueError("classic PFNN GRAIL clip id is invalid")
        families.setdefault(value.rsplit("__", 1)[0], []).append(value)
    if not families or any(len(values) < 2 for values in families.values()):
        raise ValueError("classic PFNN GRAIL family has no holdout variant")
    held_ids = {max(values) for values in families.values()}
    held_out = (source == "grail") & np.isin(clips, tuple(sorted(held_ids)))
    optimized = (source == "grail") & ~held_out
    return optimized, held_out


def _mirror_normalized_examples(
    x: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    normalization: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the original PFNN sagittal mirror in physical feature space."""

    x_array = np.asarray(x, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    phase_array = np.asarray(phase, dtype=np.float32)
    if (
        x_array.ndim != 2
        or x_array.shape[1] != INPUT_LAYOUT.size
        or y_array.shape != (len(x_array), OUTPUT_LAYOUT.size)
        or phase_array.shape != (len(x_array),)
    ):
        raise ValueError("classic PFNN mirror examples are invalid")
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)
    denormalized_x = x_array.copy()
    for field in ("previous_body_position", "previous_body_velocity"):
        denormalized_x[:, INPUT_LAYOUT[field]] /= np.float32(0.1)
    denormalized_x = denormalized_x * x_std + x_mean
    denormalized_y = y_array * y_std + y_mean
    mirrored_x = np.empty_like(x_array)
    mirrored_y = np.empty_like(y_array)
    mirrored_phase = np.empty_like(phase_array)
    for index in range(len(x_array)):
        window = PFNNTrainingWindow(
            x=denormalized_x[index],
            y=denormalized_y[index],
            phase=float(phase_array[index]),
            clip_id="terrain_slopes__slope_000__000",
            split_identity="slope_000",
            split="train",
            sequence_lane="motion",
            center_frame=index,
            motion_sha256="a" * 64,
            terrain_sha256="b" * 64,
            terrain_class="flat",
        )
        mirrored = mirror_window(window)
        mirrored_x[index] = normalize_pfnn_input(
            mirrored.x, x_mean, x_std
        )
        mirrored_y[index] = normalize_pfnn_output(
            mirrored.y, y_mean, y_std
        )
        mirrored_phase[index] = mirrored.phase
    return mirrored_x, mirrored_y, mirrored_phase


def _evaluate(
    model: PhaseFunctionedNetwork,
    x: np.ndarray,
    y: np.ndarray,
    phase: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            prediction = model(
                torch.as_tensor(x[start:stop], device=device),
                torch.as_tensor(phase[start:stop], device=device),
            )
            loss = classic_pfnn_loss(
                prediction, torch.as_tensor(y[start:stop], device=device)
            )
            total += float(loss) * (stop - start)
    return total / len(x)


def train(arguments: argparse.Namespace) -> Path:
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device(arguments.device)
    dataset_path = Path(arguments.dataset).expanduser().resolve()
    root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    vertical_dataset: VerticalDataset | None = None
    if arguments.train_source in ("released-pfnn", "mixed"):
        vertical_dataset = load_vertical_dataset(root)
        train_dataset = _VerticalTrainingView(vertical_dataset, "train")
        validation_dataset = _VerticalTrainingView(vertical_dataset, "validation")
        train_x, train_y, train_phase = _materialize_vertical(train_dataset)
        val_x, val_y, val_phase = _materialize_vertical(validation_dataset)
        train_clip = np.asarray(
            vertical_dataset.splits["train"].clip_id, dtype="<U128"
        )
        train_source = (
            np.where(
                np.char.startswith(train_clip, "terrain_slopes__"),
                "grail",
                "lafan",
            ).astype("<U6")
            if arguments.train_source == "mixed"
            else np.full(len(train_x), "released_pfnn", dtype="<U13")
        )
        manifest = {"dataset_digest_sha256": vertical_dataset.dataset_sha256}
        seed_dataset = _released_runtime_seed_view(vertical_dataset)
    else:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        train_dataset = PFNNShardDataset(root, "train")
        train_x, train_y, train_phase, train_source, train_clip = _materialize(
            train_dataset
        )
    if arguments.train_source == "grail":
        optimized, held_out = _grail_train_validation_masks(
            train_source, train_clip
        )
        if not np.any(optimized) or not np.any(held_out):
            raise ValueError("classic PFNN GRAIL train/validation partition is empty")
        optimized_indices = np.flatnonzero(optimized)
        seed_dataset = _IndexedTrainDataset(train_dataset, optimized_indices)
        val_x = train_x[held_out]
        val_y = train_y[held_out]
        val_phase = train_phase[held_out]
        train_x = train_x[optimized]
        train_y = train_y[optimized]
        train_phase = train_phase[optimized]
        train_source = train_source[optimized]
    kinematics = TorchG1ForwardKinematics.from_mjcf(arguments.model_path)
    runtime_seed = choose_runtime_seed(
        seed_dataset,
        kinematics.joint_limits,
        require_terrain=(
            arguments.train_source == "released-pfnn"
            and arguments.runtime_seed == "terrain"
        ),
        input_layout=CLASSIC_G1_INPUT_LAYOUT_V3,
    )
    phase_q99 = training_phase_advance_q99(seed_dataset)
    normalization = {
        "x_mean": train_dataset.x_mean,
        "x_std": train_dataset.x_std,
        "y_mean": train_dataset.y_mean,
        "y_std": train_dataset.y_std,
    }
    if arguments.train_source == "grail":
        mirror_x, mirror_y, mirror_phase = _mirror_normalized_examples(
            train_x, train_y, train_phase, normalization
        )
        train_x = np.concatenate((train_x, mirror_x), axis=0)
        train_y = np.concatenate((train_y, mirror_y), axis=0)
        train_phase = np.concatenate((train_phase, mirror_phase), axis=0)
        train_source = np.concatenate(
            (train_source, np.full(len(mirror_x), "grail", dtype="<U6")), axis=0
        )
        val_mirror_x, val_mirror_y, val_mirror_phase = _mirror_normalized_examples(
            val_x, val_y, val_phase, normalization
        )
        val_x = np.concatenate((val_x, val_mirror_x), axis=0)
        val_y = np.concatenate((val_y, val_mirror_y), axis=0)
        val_phase = np.concatenate((val_phase, val_mirror_phase), axis=0)
    model = PhaseFunctionedNetwork(
        hidden_size=arguments.hidden_size,
        dropout_probability=0.30,
        input_size=CLASSIC_G1_INPUT_LAYOUT_V3.size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    best = output / "best.pt"
    best_loss = math.inf
    metrics = output / "metrics.jsonl"
    with metrics.open("w", encoding="utf-8") as stream:
        for epoch in range(arguments.epochs):
            model.train()
            train_total = 0.0
            train_count = 0
            batches = (
                vertical_epoch_batches(
                    len(train_x),
                    batch_size=arguments.batch_size,
                    seed=arguments.seed,
                    epoch=epoch,
                )
                if arguments.train_source == "released-pfnn"
                else balanced_epoch_batches(
                    train_source,
                    batch_size=arguments.batch_size,
                    seed=arguments.seed,
                    epoch=epoch,
                    source_filter=arguments.train_source,
                )
            )
            for indices in batches:
                x = torch.as_tensor(train_x[indices], device=device)
                y = torch.as_tensor(train_y[indices], device=device)
                phase = torch.as_tensor(train_phase[indices], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = classic_pfnn_loss(model(x, phase), y)
                loss.backward()
                optimizer.step()
                train_total += float(loss.detach()) * len(indices)
                train_count += len(indices)
            validation_loss = _evaluate(
                model,
                val_x,
                val_y,
                val_phase,
                device=device,
                batch_size=arguments.evaluation_batch_size,
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": train_total / train_count,
                "validation_loss": validation_loss,
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_classic_checkpoint(
                    best,
                    model=model,
                    normalization=normalization,
                    runtime_seed=runtime_seed,
                    dataset_digest=manifest["dataset_digest_sha256"],
                    kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
                    joint_limits=kinematics.joint_limits,
                    phase_advance_q99=phase_q99,
                    epoch=epoch + 1,
                    validation_loss=validation_loss,
                    seed=arguments.seed,
                    source_kind=arguments.train_source.replace("-", "_"),
                    vertical_slice_receipt_sha256=(
                        vertical_dataset.selection_sha256
                        if vertical_dataset is not None
                        else "0" * 64
                    ),
                    terrain_receipt_set_sha256=(
                        vertical_dataset.terrain_receipt_set_sha256
                        if vertical_dataset is not None
                        else "0" * 64
                    ),
                )
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23456)
    parser.add_argument(
        "--train-source",
        choices=("grail", "mixed", "released-pfnn"),
        default="grail",
    )
    parser.add_argument(
        "--runtime-seed", choices=("flat", "terrain"), default="flat"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if (
        arguments.epochs < 1
        or arguments.batch_size < 2
        or arguments.batch_size % 2
        or arguments.evaluation_batch_size < 1
        or arguments.hidden_size < 1
        or not math.isfinite(arguments.learning_rate)
        or arguments.learning_rate <= 0.0
    ):
        raise ValueError("classic PFNN training arguments are invalid")
    checkpoint = train(arguments)
    print(json.dumps({"checkpoint": str(checkpoint), "status": "accepted"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
