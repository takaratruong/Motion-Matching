"""Exact-input limb reconstruction gate for classic native-G1 PFNN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

from mm_sonic.build_g1_pfnn_vertical_dataset import load_vertical_dataset
from mm_sonic.terrain_pfnn.dataset import (
    normalize_pfnn_input,
    normalize_pfnn_output,
)
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.train_classic_g1_pfnn import load_classic_checkpoint


RELEASED_PFNN_GATE = {
    "joint_mae_rad": 0.100,
    "joint_rmse_rad": 0.150,
    "frame_max_p95_rad": 0.500,
    "maximum_joint_error_rad": 1.000,
}
_GRAIL_PREFIX = "terrain_slopes__"


def _joint_rows(value: object) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("joint reconstruction rows must be finite rank-two arrays")
    if rows.shape[1] == 29:
        return rows
    if rows.shape[1] == OUTPUT_LAYOUT.size:
        return rows[:, OUTPUT_LAYOUT["joint_position"]]
    raise ValueError("joint reconstruction rows must contain 29 joints or physical PFNN outputs")


def joint_reconstruction_metrics(
    predicted_physical: object, target_physical: object
) -> dict[str, float | int]:
    """Measure absolute joint-angle reconstruction with worst-frame provenance."""

    predicted = _joint_rows(predicted_physical)
    target = _joint_rows(target_physical)
    if predicted.shape != target.shape or len(predicted) < 1:
        raise ValueError("joint reconstruction prediction and target shape is invalid")
    error = np.abs(
        np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    )
    frame_maximum = np.max(error, axis=1)
    worst_flat = int(np.argmax(error))
    worst_frame, worst_joint = np.unravel_index(worst_flat, error.shape)
    return {
        "sample_count": int(len(error)),
        "joint_mae_rad": float(np.mean(error)),
        "joint_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "frame_max_p95_rad": float(np.percentile(frame_maximum, 95.0)),
        "maximum_joint_error_rad": float(error[worst_frame, worst_joint]),
        "worst_frame_index": int(worst_frame),
        "worst_joint_index": int(worst_joint),
    }


def source_joint_reconstruction_metrics(
    predicted_physical: object, target_physical: object, clip_ids: object
) -> dict[str, dict[str, float | int]]:
    """Keep released-PFNN and GRAIL errors separate before any reduction."""

    predicted = _joint_rows(predicted_physical)
    target = _joint_rows(target_physical)
    clips = np.asarray(clip_ids).astype(str)
    if predicted.shape != target.shape or clips.shape != (len(predicted),):
        raise ValueError("source joint reconstruction rows and clip IDs are invalid")
    grail = np.char.startswith(clips, _GRAIL_PREFIX)
    groups = {
        "released_pfnn": ~grail,
        "grail": grail,
    }
    if any(not np.any(mask) for mask in groups.values()):
        raise ValueError("source-specific evaluation requires released PFNN and GRAIL rows")
    return {
        source: joint_reconstruction_metrics(predicted[mask], target[mask])
        for source, mask in groups.items()
    }


def released_pfnn_gate_failures(metrics: Mapping[str, object]) -> tuple[str, ...]:
    """Return every fixed released-source limb threshold that was exceeded."""

    if not isinstance(metrics, Mapping) or type(metrics.get("sample_count")) is not int:
        raise ValueError("released PFNN reconstruction metrics are invalid")
    if int(metrics["sample_count"]) < 1:
        raise ValueError("released PFNN reconstruction metrics are empty")
    failures: list[str] = []
    for name, threshold in RELEASED_PFNN_GATE.items():
        value = metrics.get(name)
        if type(value) is not float or not np.isfinite(value):
            raise ValueError("released PFNN reconstruction metric is invalid")
        if value > threshold:
            failures.append(name)
    return tuple(failures)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    root = resolved if resolved.is_dir() else resolved.parent
    if resolved.is_file() and resolved.name != "manifest.json":
        raise ValueError("classic PFNN dataset must be a dataset root or manifest")
    return root


def _checkpoint_normalization(checkpoint: object, dataset: object) -> dict[str, np.ndarray]:
    normalization = getattr(checkpoint, "normalization", None)
    values = {
        name: np.ascontiguousarray(
            torch.as_tensor(normalization[name]).detach().cpu().numpy(),
            dtype=np.float32,
        )
        for name in ("x_mean", "x_std", "y_mean", "y_std")
    }
    if any(
        value.shape != (width,)
        for name, width in (("x_mean", INPUT_LAYOUT.size), ("x_std", INPUT_LAYOUT.size),
                            ("y_mean", OUTPUT_LAYOUT.size), ("y_std", OUTPUT_LAYOUT.size))
        for value in (values[name],)
    ) or any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("classic PFNN checkpoint normalization is invalid")
    if any(
        not np.array_equal(values[name], np.asarray(getattr(dataset, name), dtype=np.float32))
        for name in values
    ):
        raise ValueError("classic PFNN checkpoint normalization does not match dataset")
    return values


def evaluate(
    *, checkpoint_path: str | Path, dataset_path: str | Path,
    split: str = "validation", batch_size: int = 256, device: str = "cpu",
) -> dict[str, object]:
    """Evaluate a sealed classic checkpoint on exact float32 normalized rows."""

    if split not in ("train", "validation") or type(batch_size) is not int or batch_size < 1:
        raise ValueError("classic PFNN transfer evaluation arguments are invalid")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve(strict=True)
    root = _dataset_root(Path(dataset_path))
    checkpoint = load_classic_checkpoint(checkpoint_file)
    dataset = load_vertical_dataset(root)
    if checkpoint.dataset_digest != dataset.dataset_sha256:
        raise ValueError("classic PFNN checkpoint dataset digest mismatch")
    normalization = _checkpoint_normalization(checkpoint, dataset)
    arrays = dataset.splits[split]
    x = normalize_pfnn_input(arrays.x, normalization["x_mean"], normalization["x_std"])
    target_normalized = normalize_pfnn_output(
        arrays.y, normalization["y_mean"], normalization["y_std"]
    )
    phase = np.array(arrays.phase, dtype=np.float32, copy=True, order="C")
    if x.dtype != np.dtype(np.float32) or target_normalized.dtype != np.dtype(np.float32) or phase.dtype != np.dtype(np.float32):
        raise ValueError("classic PFNN exact inputs must be float32")
    target_physical = np.ascontiguousarray(arrays.y, dtype=np.float32)
    model = checkpoint.build_model().to(torch.device(device)).eval()
    prediction_normalized = np.empty_like(target_normalized)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            prediction_normalized[start:stop] = model(
                torch.as_tensor(x[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(phase[start:stop], dtype=torch.float32, device=device),
            ).detach().cpu().numpy().astype(np.float32, copy=False)
    predicted_physical = np.ascontiguousarray(
        prediction_normalized * normalization["y_std"] + normalization["y_mean"],
        dtype=np.float32,
    )
    source_metrics = source_joint_reconstruction_metrics(
        predicted_physical, target_physical, arrays.clip_id
    )
    failures = released_pfnn_gate_failures(source_metrics["released_pfnn"])
    return {
        "schema": "classic-g1-pfnn-transfer-evaluation/v1",
        "checkpoint_sha256": _sha256(checkpoint_file),
        "dataset_manifest_sha256": _sha256(root / "manifest.json"),
        "dataset_sha256": dataset.dataset_sha256,
        "split": split,
        "released_pfnn": {
            **source_metrics["released_pfnn"],
            "worst": {
                "frame_index": source_metrics["released_pfnn"]["worst_frame_index"],
                "joint_index": source_metrics["released_pfnn"]["worst_joint_index"],
            },
        },
        "grail": {
            **source_metrics["grail"],
            "worst": {
                "frame_index": source_metrics["grail"]["worst_frame_index"],
                "joint_index": source_metrics["grail"]["worst_joint_index"],
            },
        },
        "released_pfnn_gate": {
            "thresholds": dict(RELEASED_PFNN_GATE),
            "failures": list(failures),
            "accepted": not failures,
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = evaluate(
        checkpoint_path=arguments.checkpoint,
        dataset_path=arguments.dataset,
        split=arguments.split,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite evaluation: {output}")
        _atomic_json(output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["released_pfnn_gate"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
