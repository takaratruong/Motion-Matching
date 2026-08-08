"""Train, checkpoint, and provisionally evaluate the sealed terrain PFNN."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from mm_sonic.evaluate_terrain_pfnn import _atomic_json, _dataset_root
from mm_sonic.terrain_pfnn.dataset import PFNNShardDataset
from mm_sonic.terrain_pfnn.kinematics import TorchG1ForwardKinematics
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.training import (
    DEFAULT_LOSS_WEIGHTS,
    LOSS_WEIGHT_KEYS,
    autoregressive_unroll,
    choose_runtime_seed,
    load_checkpoint,
    one_step_metrics,
    pfnn_losses,
    restore_training_state,
    save_checkpoint,
    selection_metadata,
    training_phase_advance_q99,
)


_TERRAIN_CLASSES = ("flat", "ascent", "descent", "transition")


def overfit_gate_accepted(initial: float, final: float, ratio: float) -> bool:
    values = (float(initial), float(final), float(ratio))
    if any(not math.isfinite(value) for value in values) or values[0] <= 0.0:
        return False
    return values[2] > 0.0 and values[1] < values[2] * values[0]


class _MaterializedDataset:
    def __init__(self, source: object, rows: Sequence[dict[str, object]]) -> None:
        self.split = getattr(source, "split")
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            setattr(self, name, np.asarray(getattr(source, name)).copy())
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]


def materialize_subset(source: object, indices: Sequence[int]) -> _MaterializedDataset:
    """Cache a small subset while touching sealed shards in sorted order once."""

    ordered = [int(index) for index in indices]
    if not ordered:
        raise ValueError("materialized subset cannot be empty")
    cache: dict[int, dict[str, object]] = {}
    for index in sorted(set(ordered)):
        sample = source[index]
        cache[index] = {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in sample.items()
        }
    return _MaterializedDataset(source, [cache[index] for index in ordered])


def stratified_subset(
    terrain_classes: Sequence[str], count: int, *, seed: int
) -> list[int]:
    """Select a deterministic, maximally balanced subset without replacement."""

    if type(count) is not int or count < 1 or count > len(terrain_classes):
        raise ValueError("stratified subset count is outside the dataset")
    groups = {
        name: [index for index, value in enumerate(terrain_classes) if value == name]
        for name in _TERRAIN_CLASSES
    }
    if any(not group for group in groups.values()):
        raise ValueError("stratified sampling requires every terrain class")
    generator = random.Random(int(seed))
    for group in groups.values():
        generator.shuffle(group)
    offsets = {name: 0 for name in _TERRAIN_CLASSES}
    output: list[int] = []
    while len(output) < count:
        order = list(_TERRAIN_CLASSES)
        generator.shuffle(order)
        made_progress = False
        for name in order:
            offset = offsets[name]
            if offset < len(groups[name]) and len(output) < count:
                output.append(groups[name][offset])
                offsets[name] += 1
                made_progress = True
        if not made_progress:
            break
    if len(output) != count:
        raise ValueError("could not construct the requested stratified subset")
    return output


def _balanced_epoch_indices(
    candidate_indices: Sequence[int],
    terrain_classes: Sequence[str],
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    groups = {
        name: [index for index in candidate_indices if terrain_classes[index] == name]
        for name in _TERRAIN_CLASSES
    }
    if any(not group for group in groups.values()):
        raise ValueError("balanced epochs require every terrain class")
    generator = random.Random(int(seed) + 1_000_003 * int(epoch))
    for group in groups.values():
        generator.shuffle(group)
    width = max(map(len, groups.values()))
    output: list[int] = []
    for offset in range(width):
        order = list(_TERRAIN_CLASSES)
        generator.shuffle(order)
        output.extend(groups[name][offset % len(groups[name])] for name in order)
    return output


def seed_worker(worker_id: int, *, seed: int, rank: int) -> None:
    worker_seed = int(seed) + 10_007 * int(rank) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def _distributed_context(seed: int) -> tuple[int, int, int, torch.device]:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    return rank, world_size, local_rank, device


def _broadcast_indices(indices: list[int] | None, device: torch.device) -> list[int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        assert indices is not None
        return indices
    rank = dist.get_rank()
    length = torch.tensor(
        -1 if indices is None else len(indices), dtype=torch.int64, device=device
    )
    dist.broadcast(length, src=0)
    if int(length) < 0:
        raise RuntimeError("rank 0 did not provide distributed indices")
    tensor = torch.empty(int(length), dtype=torch.int64, device=device)
    if rank == 0:
        assert indices is not None
        tensor.copy_(torch.as_tensor(indices, dtype=torch.int64, device=device))
    dist.broadcast(tensor, src=0)
    return tensor.cpu().tolist()


def _padded_rank_epoch(
    indices: list[int], *, batch_size: int, rank: int, world_size: int
) -> tuple[list[int], int]:
    divisor = batch_size * world_size
    padding = (-len(indices)) % divisor
    if padding:
        indices = indices + [indices[index % len(indices)] for index in range(padding)]
    return indices[rank::world_size], padding


def _batch(
    dataset: object, indices: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [dataset[index] for index in indices]
    return (
        torch.as_tensor(np.stack([item["x"] for item in samples]), device=device),
        torch.as_tensor(np.asarray([item["phase"] for item in samples]), device=device),
        torch.as_tensor(np.stack([item["y"] for item in samples]), device=device),
    )


def _terrain_classes(dataset: object) -> list[str]:
    return [str(dataset[index]["terrain_class"]) for index in range(len(dataset))]


def _consecutive_starts(
    dataset: PFNNShardDataset, frames: int
) -> tuple[list[tuple[int, ...]], list[str]]:
    if frames < 2:
        raise ValueError("rollout fine-tuning requires at least two frames")
    metadata = [
        (
            str(dataset[index]["clip_id"]),
            int(dataset[index]["center_frame"]),
            str(dataset[index]["terrain_class"]),
        )
        for index in range(len(dataset))
    ]
    sequences: list[tuple[int, ...]] = []
    classes: list[str] = []
    for start in range(len(metadata) - frames + 1):
        clip, center, terrain_class = metadata[start]
        if all(
            metadata[start + offset][0] == clip
            and metadata[start + offset][1] == center + offset
            for offset in range(frames)
        ):
            sequences.append(tuple(range(start, start + frames)))
            classes.append(terrain_class)
    if not sequences:
        raise ValueError("training split has no consecutive rollout sequences")
    return sequences, classes


def _append_metric(stream: object, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def train(arguments: argparse.Namespace) -> dict[str, object] | None:
    import torch.distributed as dist

    rank, world_size, local_rank, device = _distributed_context(arguments.seed)
    root = _dataset_root(arguments.dataset)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is missing or invalid") from error
    train_dataset = PFNNShardDataset(root, "train")
    terrain_classes = _terrain_classes(train_dataset)
    all_indices = list(range(len(train_dataset)))
    if arguments.overfit_samples is not None:
        candidate_indices = (
            stratified_subset(
                terrain_classes, arguments.overfit_samples, seed=arguments.seed
            )
            if rank == 0 else None
        )
        candidate_indices = _broadcast_indices(candidate_indices, device)
        pipeline_overfit = True
        optimization_dataset = materialize_subset(train_dataset, candidate_indices)
        optimization_classes = _terrain_classes(optimization_dataset)
        optimization_indices = list(range(len(optimization_dataset)))
    else:
        candidate_indices = all_indices
        pipeline_overfit = False
        optimization_dataset = train_dataset
        optimization_classes = terrain_classes
        optimization_indices = candidate_indices

    kinematics = TorchG1ForwardKinematics.from_mjcf(arguments.model_path).to(device)
    torch.manual_seed(arguments.seed)
    model: torch.nn.Module = PhaseFunctionedNetwork(
        hidden_size=arguments.hidden_size, dropout_probability=0.30
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    unwrapped = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.Adam(unwrapped.parameters(), lr=arguments.learning_rate)
    restored_step, restored_epoch = 0, 0
    if arguments.resume is not None:
        resumed = load_checkpoint(
            arguments.resume,
            expected_dataset_digest=manifest["dataset_digest_sha256"],
            expected_kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
        )
        restored_step, restored_epoch = restore_training_state(
            resumed,
            unwrapped,
            optimizer,
            normalization=train_dataset,
            train_identities=manifest["split_identities"]["train"],
            validation_identities=manifest["split_identities"]["validation"],
        )
        if restored_step >= arguments.steps:
            raise ValueError("--steps must exceed the resumed checkpoint step")
        if world_size > 1:
            dist.barrier()
    output = Path(arguments.output)
    metrics_path = output / "metrics.jsonl"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    if arguments.resume is None and metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run metrics: {metrics_path}")
    if arguments.resume is not None and not metrics_path.is_file():
        raise FileNotFoundError("resume requires the existing run metrics.jsonl")
    if rank == 0:
        metric_stream = metrics_path.open(
            "a" if arguments.resume is not None else "x",
            encoding="utf-8",
            buffering=1,
        )
    else:
        metric_stream = None

    evaluation_dataset = optimization_dataset
    evaluation_indices: Sequence[int] | None = optimization_indices
    if not pipeline_overfit:
        evaluation_dataset = PFNNShardDataset(root, "validation")
        evaluation_indices = None
    initial_metrics = one_step_metrics(
        model,
        evaluation_dataset,
        kinematics=kinematics,
        indices=evaluation_indices,
        batch_size=arguments.evaluation_batch_size,
        device=device,
        loss_weights=DEFAULT_LOSS_WEIGHTS,
        distributed=world_size > 1,
    )

    started = time.monotonic()
    step = restored_step
    epoch = restored_epoch
    local_epoch: list[int] = []
    local_offset = 0
    padding = 0
    while step < arguments.steps:
        if local_offset + arguments.batch_size > len(local_epoch):
            global_epoch = (
                _balanced_epoch_indices(
                    optimization_indices,
                    optimization_classes,
                    seed=arguments.seed,
                    epoch=epoch,
                )
                if rank == 0 else None
            )
            global_epoch = _broadcast_indices(global_epoch, device)
            local_epoch, padding = _padded_rank_epoch(
                global_epoch,
                batch_size=arguments.batch_size,
                rank=rank,
                world_size=world_size,
            )
            local_offset = 0
            epoch += 1
        batch_indices = local_epoch[local_offset : local_offset + arguments.batch_size]
        local_offset += arguments.batch_size
        x, phase, y = _batch(optimization_dataset, batch_indices, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = pfnn_losses(
            model(x, phase),
            y,
            model=unwrapped,
            normalization=train_dataset,
            kinematics=kinematics,
            loss_weights=DEFAULT_LOSS_WEIGHTS,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"nonfinite training loss at step {step + 1}")
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}")
        optimizer.step()
        step += 1
        values = torch.stack(
            [losses[name].detach() for name in (*LOSS_WEIGHT_KEYS, "total")]
        ).to(torch.float64)
        if world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= world_size
        if rank == 0:
            assert metric_stream is not None
            _append_metric(
                metric_stream,
                {
                    "schema": "mm-sonic-terrain-pfnn-metric/v1",
                    "stage": "one_step",
                    "step": step,
                    "epoch": epoch,
                    "losses": {
                        name: float(values[index])
                        for index, name in enumerate((*LOSS_WEIGHT_KEYS, "total"))
                    },
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "samples": step * arguments.batch_size * world_size,
                    "wall_time_seconds": time.monotonic() - started,
                    "ddp_padding_samples_this_epoch": padding,
                },
            )

    gate_metrics = one_step_metrics(
        model,
        evaluation_dataset,
        kinematics=kinematics,
        indices=evaluation_indices,
        batch_size=arguments.evaluation_batch_size,
        device=device,
        loss_weights=DEFAULT_LOSS_WEIGHTS,
        distributed=world_size > 1,
    )
    one_step_accepted = (
        overfit_gate_accepted(
            float(initial_metrics["one_step_score"]),
            float(gate_metrics["one_step_score"]),
            arguments.overfit_acceptance_ratio,
        )
        if pipeline_overfit
        else math.isfinite(float(gate_metrics["one_step_score"]))
    )

    if arguments.rollout_finetune_frames:
        if not one_step_accepted:
            raise RuntimeError("rollout fine-tuning requires a passing one-step gate")
        sequences, sequence_classes = _consecutive_starts(
            train_dataset, arguments.rollout_finetune_frames
        )
        sequence_candidates = list(range(len(sequences)))
        rollout_steps = (
            arguments.rollout_finetune_steps
            if arguments.rollout_finetune_steps is not None
            else arguments.steps
        )
        for rollout_step in range(rollout_steps):
            global_sequences = (
                _balanced_epoch_indices(
                    sequence_candidates,
                    sequence_classes,
                    seed=arguments.seed + 97,
                    epoch=rollout_step,
                )
                if rank == 0 else None
            )
            global_sequences = _broadcast_indices(global_sequences, device)
            rank_sequences, _ = _padded_rank_epoch(
                global_sequences,
                batch_size=arguments.batch_size,
                rank=rank,
                world_size=world_size,
            )
            chosen = rank_sequences[: arguments.batch_size]
            rows = [[train_dataset[index] for index in sequences[item]] for item in chosen]
            inputs = torch.as_tensor(
                np.stack([[row[frame]["x"] for row in rows] for frame in range(arguments.rollout_finetune_frames)]),
                device=device,
            )
            targets = torch.as_tensor(
                np.stack([[row[frame]["y"] for row in rows] for frame in range(arguments.rollout_finetune_frames)]),
                device=device,
            )
            phases = torch.as_tensor(
                np.asarray([row[0]["phase"] for row in rows]), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            rollout = autoregressive_unroll(
                model,
                inputs,
                phases,
                targets,
                normalization=train_dataset,
                kinematics=kinematics,
                loss_weights=DEFAULT_LOSS_WEIGHTS,
            )
            if not torch.isfinite(rollout.losses["total"]):
                raise FloatingPointError(
                    f"nonfinite rollout loss at step {step + 1}"
                )
            rollout.losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"nonfinite rollout gradient at step {step + 1}"
                )
            optimizer.step()
            step += 1
            rollout_values = torch.stack(
                [
                    rollout.losses[name].detach()
                    for name in (*LOSS_WEIGHT_KEYS, "total")
                ]
            ).to(torch.float64)
            if world_size > 1:
                dist.all_reduce(rollout_values, op=dist.ReduceOp.SUM)
                rollout_values /= world_size
            if rank == 0:
                assert metric_stream is not None
                _append_metric(
                    metric_stream,
                    {
                        "schema": "mm-sonic-terrain-pfnn-metric/v1",
                        "stage": "rollout_finetune",
                        "step": step,
                        "epoch": epoch,
                        "losses": {
                            name: float(rollout_values[index])
                            for index, name in enumerate(
                                (*LOSS_WEIGHT_KEYS, "total")
                            )
                        },
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(gradient_norm),
                        "samples": arguments.batch_size
                        * arguments.rollout_finetune_frames,
                        "wall_time_seconds": time.monotonic() - started,
                    },
                )
        gate_metrics = one_step_metrics(
            model,
            evaluation_dataset,
            kinematics=kinematics,
            indices=evaluation_indices,
            batch_size=arguments.evaluation_batch_size,
            device=device,
            loss_weights=DEFAULT_LOSS_WEIGHTS,
            distributed=world_size > 1,
        )

    one_step_accepted = (
        overfit_gate_accepted(
            float(initial_metrics["one_step_score"]),
            float(gate_metrics["one_step_score"]),
            arguments.overfit_acceptance_ratio,
        )
        if pipeline_overfit
        else math.isfinite(float(gate_metrics["one_step_score"]))
    )

    if rank == 0:
        assert metric_stream is not None
        _append_metric(
            metric_stream,
            {
                "schema": "mm-sonic-terrain-pfnn-metric/v1",
                "stage": "validation" if not pipeline_overfit else "pipeline_overfit_gate",
                "step": step,
                "epoch": epoch,
                "losses": {
                    name: float(gate_metrics[name])
                    for name in LOSS_WEIGHT_KEYS
                    if name != "regularization"
                },
                "validation_score": float(gate_metrics["one_step_score"]),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "samples": int(gate_metrics["samples"]),
                "wall_time_seconds": time.monotonic() - started,
            },
        )

    if rank != 0:
        if world_size > 1:
            dist.barrier()
        return None
    assert metric_stream is not None
    metric_stream.flush()
    os.fsync(metric_stream.fileno())
    metric_stream.close()
    dataset_digest = manifest["dataset_digest_sha256"]
    train_identities = manifest["split_identities"]["train"]
    validation_identities = manifest["split_identities"]["validation"]
    phase_q99 = training_phase_advance_q99(train_dataset)
    runtime_seed = choose_runtime_seed(
        train_dataset, kinematics.joint_limits.detach().cpu()
    )
    selection = (
        selection_metadata(
            one_step_score=float(gate_metrics["one_step_score"]),
            pipeline_overfit=True,
        )
        if pipeline_overfit else None
    )
    candidate = output / f"checkpoint-step-{step:08d}.pt"
    save_checkpoint(
        candidate,
        unwrapped,
        optimizer,
        train_dataset,
        dataset_digest=dataset_digest,
        kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
        runtime_seed=runtime_seed,
        step=step,
        epoch=epoch,
        seed=arguments.seed,
        loss_weights=DEFAULT_LOSS_WEIGHTS,
        joint_limits=kinematics.joint_limits.detach().cpu(),
        phase_advance_q99=phase_q99,
        train_identities=train_identities,
        validation_identities=validation_identities,
        selection=selection,
    )
    load_checkpoint(
        candidate,
        expected_dataset_digest=dataset_digest,
        expected_kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
    )
    best_path: str | None = None
    if pipeline_overfit:
        best = output / "best.pt"
        save_checkpoint(
            best,
            unwrapped,
            optimizer,
            train_dataset,
            dataset_digest=dataset_digest,
            kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
            runtime_seed=runtime_seed,
            step=step,
            epoch=epoch,
            seed=arguments.seed,
            loss_weights=DEFAULT_LOSS_WEIGHTS,
            joint_limits=kinematics.joint_limits.detach().cpu(),
            phase_advance_q99=phase_q99,
            train_identities=train_identities,
            validation_identities=validation_identities,
            selection=selection,
        )
        best_path = str(best)
    report = {
        "schema": "mm-sonic-terrain-pfnn-one-step-report/v1",
        "dataset_digest_sha256": dataset_digest,
        "kinematic_signature_sha256": kinematics.kinematic_signature_sha256,
        "fixed_subset_samples": len(candidate_indices) if pipeline_overfit else None,
        "initial": initial_metrics,
        "final": gate_metrics,
        "loss_ratio": float(gate_metrics["one_step_score"])
        / float(initial_metrics["one_step_score"]),
        "accepted": bool(one_step_accepted),
        "checkpoint_finite_and_reloadable": True,
        "candidate_checkpoint": str(candidate),
        "best_checkpoint": best_path,
        "selection": selection,
        "closed_loop": {
            "status": "pending_task_7",
            "failure_penalty": None,
            "accepted": None,
        },
    }
    _atomic_json(output / "one-step-report.json", report)
    if world_size > 1:
        dist.barrier()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--overfit-acceptance-ratio", type=float, default=0.5)
    parser.add_argument("--rollout-finetune-frames", type=int, default=0)
    parser.add_argument("--rollout-finetune-steps", type=int)
    parser.add_argument("--resume")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.steps < 1 or arguments.batch_size < 1 or arguments.hidden_size < 1:
        raise ValueError("steps, batch size, and hidden size must be positive")
    if arguments.rollout_finetune_frames not in (0, *range(2, 17)):
        raise ValueError("rollout fine-tuning frames must be 0 or in [2,16]")
    report = train(arguments)
    if report is not None:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        if not report["accepted"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "materialize_subset",
    "overfit_gate_accepted",
    "seed_worker",
    "stratified_subset",
    "train",
]
