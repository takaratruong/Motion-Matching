"""Deterministic one-step evaluator for sealed terrain-PFNN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch

from mm_sonic.terrain_pfnn.dataset import PFNNShardDataset
from mm_sonic.terrain_pfnn.kinematics import TorchG1ForwardKinematics
from mm_sonic.terrain_pfnn.training import load_checkpoint, one_step_metrics


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    return candidate.parent if candidate.is_file() else candidate


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def claim_sealed_test_receipt(
    run_directory: str | Path,
    *,
    checkpoint_sha256: str,
    dataset_digest: str,
) -> Path:
    """Atomically reserve the sole sealed-test evaluation for a run directory."""

    if (
        _SHA256_RE.fullmatch(checkpoint_sha256) is None
        or _SHA256_RE.fullmatch(dataset_digest) is None
    ):
        raise ValueError("sealed-test receipt digests must be SHA-256 values")
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "sealed-test-receipt.json"
    encoded = (
        json.dumps(
            {
                "schema": "mm-sonic-terrain-pfnn-sealed-test-receipt/v1",
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_digest_sha256": dataset_digest,
                "selection_pass": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("sealed test has already been evaluated in this run directory") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        receipt.unlink(missing_ok=True)
        raise
    return receipt


def validate_checkpoint_kinematics(checkpoint: object, kinematics: object) -> None:
    if (
        getattr(checkpoint, "kinematic_signature_sha256", None)
        != getattr(kinematics, "kinematic_signature_sha256", None)
    ):
        raise ValueError("checkpoint kinematic signature mismatch")
    try:
        expected_limits = torch.as_tensor(
            getattr(kinematics, "joint_limits"), dtype=torch.float64, device="cpu"
        )
        checkpoint_limits = getattr(checkpoint, "joint_limits")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint canonical joint limits mismatch") from error
    if (
        type(checkpoint_limits) is not torch.Tensor
        or checkpoint_limits.dtype != torch.float64
        or not torch.equal(checkpoint_limits, expected_limits)
    ):
        raise ValueError("checkpoint canonical joint limits mismatch")


def evaluate(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    model_path: str | Path,
    split: str,
    output_path: str | Path,
    sealed_test: bool,
    run_directory: str | Path | None,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    if split not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    if split == "test" and not sealed_test:
        raise ValueError("test evaluation requires --sealed-test")
    if split != "test" and sealed_test:
        raise ValueError("--sealed-test is only valid for the test split")
    root = _dataset_root(dataset_path)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is missing or invalid") from error
    dataset_digest = manifest.get("dataset_digest_sha256")
    if type(dataset_digest) is not str or _SHA256_RE.fullmatch(dataset_digest) is None:
        raise ValueError("dataset digest is invalid")
    kinematics = TorchG1ForwardKinematics.from_mjcf(model_path)
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_dataset_digest=dataset_digest,
        expected_kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
    )
    validate_checkpoint_kinematics(checkpoint, kinematics)
    expected_train = tuple(manifest.get("split_identities", {}).get("train", ()))
    expected_validation = tuple(manifest.get("split_identities", {}).get("validation", ()))
    if (
        checkpoint.train_identities != expected_train
        or checkpoint.validation_identities != expected_validation
    ):
        raise ValueError("checkpoint train/validation identities mismatch")
    dataset = PFNNShardDataset(root, split)
    for name in ("x_mean", "x_std", "y_mean", "y_std"):
        expected = torch.as_tensor(getattr(dataset, name), dtype=torch.float32)
        if not torch.equal(checkpoint.normalization[name], expected):
            raise ValueError("checkpoint normalization arrays mismatch")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_digest = _sha256(checkpoint_path)
    if split == "test":
        claim_sealed_test_receipt(
            Path(output_path).parent if run_directory is None else run_directory,
            checkpoint_sha256=checkpoint_digest,
            dataset_digest=dataset_digest,
        )
    target_device = torch.device(device)
    model = checkpoint.build_model().to(target_device)
    kinematics = kinematics.to(target_device)
    metrics = one_step_metrics(
        model,
        dataset,
        kinematics=kinematics,
        batch_size=batch_size,
        device=target_device,
        loss_weights=checkpoint.loss_weights,
    )
    report: dict[str, object] = {
        "schema": "mm-sonic-terrain-pfnn-evaluation/v1",
        "checkpoint_sha256": checkpoint_digest,
        "dataset_digest_sha256": dataset_digest,
        "kinematic_signature_sha256": kinematics.kinematic_signature_sha256,
        "split": split,
        "sealed_test": split == "test",
        "one_step": metrics,
        "closed_loop": {
            "status": "pending_task_7",
            "failure_penalty": None,
            "accepted": None,
        },
    }
    _atomic_json(Path(output_path), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--sealed-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = evaluate(
        checkpoint_path=arguments.checkpoint,
        dataset_path=arguments.dataset,
        model_path=arguments.model_path,
        split=arguments.split,
        output_path=arguments.output,
        sealed_test=arguments.sealed_test,
        run_directory=arguments.run_dir,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "claim_sealed_test_receipt",
    "evaluate",
    "main",
    "validate_checkpoint_kinematics",
]
