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

from .terrain_pfnn.dataset import PFNNShardDataset
from .terrain_pfnn.kinematics import TorchG1ForwardKinematics
from .terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from .terrain_pfnn.model import PhaseFunctionedNetwork
from .terrain_pfnn.training import choose_runtime_seed, training_phase_advance_q99


CLASSIC_CHECKPOINT_SCHEMA = "classic-g1-pfnn/v1"
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
    source_kind: object, *, batch_size: int, seed: int, epoch: int
) -> tuple[np.ndarray, ...]:
    """Return deterministic half-GRAIL/half-LAFAN batches for one epoch."""

    kinds = np.asarray(source_kind)
    if kinds.ndim != 1 or set(kinds.astype(str).tolist()) != {"grail", "lafan"}:
        raise ValueError("classic PFNN source kinds must contain grail and lafan")
    if type(batch_size) is not int or batch_size < 2 or batch_size % 2:
        raise ValueError("classic PFNN batch size must be a positive even integer")
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("classic PFNN sampler seed/epoch are invalid")
    half = batch_size // 2
    rng = np.random.default_rng(np.random.SeedSequence((seed, epoch)))
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


@dataclass(frozen=True)
class ClassicG1PFNNCheckpoint:
    model_state: dict[str, torch.Tensor]
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

    def build_model(self) -> PhaseFunctionedNetwork:
        model = PhaseFunctionedNetwork(
            hidden_size=self.hidden_size,
            dropout_probability=self.dropout_probability,
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
) -> None:
    if not isinstance(model, PhaseFunctionedNetwork):
        raise TypeError("classic PFNN checkpoint model is invalid")
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
    ):
        raise ValueError("classic PFNN checkpoint provenance is invalid")
    score = float(validation_loss)
    q99 = float(phase_advance_q99)
    if not math.isfinite(score) or score < 0.0 or not math.isfinite(q99) or q99 <= 0.0:
        raise ValueError("classic PFNN checkpoint metrics are invalid")
    expected_normal = {
        "x_mean": INPUT_LAYOUT.size,
        "x_std": INPUT_LAYOUT.size,
        "y_mean": OUTPUT_LAYOUT.size,
        "y_std": OUTPUT_LAYOUT.size,
    }
    normal = {
        name: torch.as_tensor(normalization[name], dtype=torch.float32, device="cpu").contiguous().clone()
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
    payload = {
        "schema": CLASSIC_CHECKPOINT_SCHEMA,
        "model_config": {
            "hidden_size": int(model.W0.shape[1]),
            "dropout_probability": float(model.dropout.p),
        },
        "model_state": dict(model.state_dict()),
        "normalization": normal,
        "runtime_seed": dict(runtime_seed),
        "dataset_digest": dataset_digest,
        "kinematic_signature_sha256": kinematic_signature_sha256,
        "joint_limits": limits,
        "phase_advance_q99": q99,
        "epoch": epoch,
        "validation_loss": score,
        "seed": seed,
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
        "schema", "model_config", "model_state", "normalization", "runtime_seed",
        "dataset_digest", "kinematic_signature_sha256", "joint_limits",
        "phase_advance_q99", "epoch", "validation_loss", "seed",
    }
    if type(payload) is not dict or set(payload) != required or payload["schema"] != CLASSIC_CHECKPOINT_SCHEMA:
        raise ValueError("classic PFNN checkpoint schema is invalid")
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
    checkpoint = ClassicG1PFNNCheckpoint(
        model_state=payload["model_state"],
        hidden_size=config["hidden_size"],
        dropout_probability=config["dropout_probability"],
        normalization=payload["normalization"],
        runtime_seed=payload["runtime_seed"],
        dataset_digest=payload["dataset_digest"],
        kinematic_signature_sha256=payload["kinematic_signature_sha256"],
        joint_limits=payload["joint_limits"],
        phase_advance_q99=float(payload["phase_advance_q99"]),
        epoch=payload["epoch"],
        validation_loss=float(payload["validation_loss"]),
        seed=payload["seed"],
    )
    if not _finite_tree(payload):
        raise ValueError("classic PFNN checkpoint contains nonfinite values")
    checkpoint.build_model()
    return checkpoint


def _materialize(dataset: PFNNShardDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((len(dataset), INPUT_LAYOUT.size), dtype=np.float32)
    y = np.empty((len(dataset), OUTPUT_LAYOUT.size), dtype=np.float32)
    phase = np.empty(len(dataset), dtype=np.float32)
    source = np.empty(len(dataset), dtype="<U6")
    for index in range(len(dataset)):
        row = dataset[index]
        x[index] = row["x"]
        y[index] = row["y"]
        phase[index] = row["phase"]
        source[index] = "grail" if str(row["clip_id"]).startswith("terrain_slopes__") else "lafan"
    return x, y, phase, source


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
    root = Path(arguments.dataset).expanduser().resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_dataset = PFNNShardDataset(root, "train")
    validation_dataset = PFNNShardDataset(root, "validation")
    train_x, train_y, train_phase, train_source = _materialize(train_dataset)
    val_x, val_y, val_phase, _ = _materialize(validation_dataset)
    kinematics = TorchG1ForwardKinematics.from_mjcf(arguments.model_path)
    runtime_seed = choose_runtime_seed(train_dataset, kinematics.joint_limits)
    phase_q99 = training_phase_advance_q99(train_dataset)
    normalization = {
        "x_mean": train_dataset.x_mean,
        "x_std": train_dataset.x_std,
        "y_mean": train_dataset.y_mean,
        "y_std": train_dataset.y_std,
    }
    model = PhaseFunctionedNetwork(
        hidden_size=arguments.hidden_size, dropout_probability=0.30
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
            for indices in balanced_epoch_batches(
                train_source,
                batch_size=arguments.batch_size,
                seed=arguments.seed,
                epoch=epoch,
            ):
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
                )
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23456)
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
