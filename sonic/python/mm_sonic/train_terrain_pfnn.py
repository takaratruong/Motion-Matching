"""Train, checkpoint, and provisionally evaluate the sealed terrain PFNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from mm_sonic.evaluate_terrain_pfnn import _atomic_json, _dataset_root
from mm_sonic.terrain_pfnn.dataset import PFNNShardDataset
from mm_sonic.terrain_pfnn.kinematics import TorchG1ForwardKinematics
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.runtime import ClosedLoopValidationResult
from mm_sonic.terrain_pfnn.splits import split_identity, terrain_identity
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


def promote_pipeline_best(*, pipeline_overfit: bool, accepted: bool) -> bool:
    return bool(pipeline_overfit and accepted)


def validated_normal_selection(
    *,
    one_step_score: float,
    result: ClosedLoopValidationResult | None,
    dataset_digest: str,
    kinematic_signature_sha256: str,
) -> tuple[dict[str, object] | None, bool]:
    """Return promotable metadata only for an actual matching validation run."""

    if (
        not isinstance(result, ClosedLoopValidationResult)
        or not result.evaluated
        or result.split != "validation"
        or result.metrics.get("dataset_digest_sha256") != dataset_digest
        or result.metrics.get("kinematic_signature_sha256")
        != kinematic_signature_sha256
    ):
        return None, False
    selection = selection_metadata(
        one_step_score=one_step_score,
        pipeline_overfit=False,
        closed_loop_scorer=lambda: result.failure_penalty,
    )
    return selection, True


def resolve_rollout_finetune_frames(
    value: int | None, *, pipeline_overfit: bool
) -> int:
    """Resolve the mode-dependent default without overriding an explicit flag."""

    resolved = (0 if pipeline_overfit else 16) if value is None else value
    if type(resolved) is not int or resolved not in (0, *range(2, 17)):
        raise ValueError("rollout fine-tuning frames must be 0 or in [2,16]")
    return resolved


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


def consecutive_overfit_subset(dataset: object, count: int) -> list[int]:
    """Select source-sealed consecutive GRAIL slope runs for pipeline overfit.

    Identities, clip variants, and maximal runs are traversed lexicographically.
    A center emitted more than once (idle phase augmentation) is excluded and is
    therefore a hard run boundary.  This makes every retained successor an
    actual recurrent sample instead of an unrelated stratified row.
    """

    if getattr(dataset, "split", None) != "train":
        raise ValueError("overfit subset may only be selected from training data")
    if type(count) is not int or count < 1 or count > len(dataset):
        raise ValueError("overfit subset count is outside the dataset")
    grouped: dict[str, dict[str, dict[int, list[tuple[int, str]]]]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        clip = str(sample.get("clip_id", ""))
        identity = str(sample.get("split_identity", ""))
        row_split = sample.get("split")
        center = sample.get("center_frame")
        terrain_class = str(sample.get("terrain_class", ""))
        if (
            not clip
            or not identity
            or row_split != "train"
            or type(center) is not int
            or center < 0
            or terrain_class not in _TERRAIN_CLASSES
            or terrain_identity(clip) != identity
            or split_identity(identity) != "train"
        ):
            raise ValueError("overfit subset row provenance is not source sealed")
        # GRAIL slope identities are the only source with paired terrain and
        # are preferred ahead of all flat LAFAN identities by contract.
        if not identity.startswith("slope_"):
            continue
        grouped.setdefault(identity, {}).setdefault(clip, {}).setdefault(
            center, []
        ).append((index, terrain_class))

    selected: list[int] = []
    selected_classes: list[str] = []
    for identity in sorted(grouped):
        for clip in sorted(grouped[identity]):
            by_center = grouped[identity][clip]
            unique = sorted(
                center for center, rows in by_center.items() if len(rows) == 1
            )
            runs: list[list[int]] = []
            for center in unique:
                if not runs or center != runs[-1][-1] + 1:
                    runs.append([center])
                else:
                    runs[-1].append(center)
            for run in runs:
                for center in run:
                    index, terrain_class = by_center[center][0]
                    selected.append(index)
                    selected_classes.append(terrain_class)
                    if len(selected) == count:
                        break
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("GRAIL training slopes lack the requested consecutive rows")
    if "flat" not in selected_classes or not any(
        value in ("ascent", "descent", "transition")
        for value in selected_classes
    ):
        raise ValueError("consecutive overfit subset must contain flat and slope rows")
    return selected


def fitted_subset_metadata(
    dataset: object, indices: Sequence[int]
) -> dict[str, object]:
    """Return the exact fitted row receipt stored in checkpoint and report."""

    if getattr(dataset, "split", None) != "train":
        raise ValueError("fitted subset receipt must describe training rows")
    rows: list[dict[str, object]] = []
    counts = {name: 0 for name in _TERRAIN_CLASSES}
    seen: set[tuple[str, int]] = set()
    for raw_index in indices:
        sample = dataset[int(raw_index)]
        clip = str(sample.get("clip_id", ""))
        center = sample.get("center_frame")
        terrain_class = str(sample.get("terrain_class", ""))
        identity = str(sample.get("split_identity", ""))
        key = (clip, center) if type(center) is int else (clip, -1)
        if (
            not clip
            or type(center) is not int
            or center < 0
            or terrain_class not in counts
            or sample.get("split") != "train"
            or terrain_identity(clip) != identity
            or split_identity(identity) != "train"
            or key in seen
        ):
            raise ValueError("fitted subset receipt contains an invalid row")
        seen.add(key)
        counts[terrain_class] += 1
        rows.append({"clip_id": clip, "center_frame": center})
    if not rows:
        raise ValueError("fitted subset receipt cannot be empty")
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "split": "train",
        "rows": rows,
        "rows_sha256": hashlib.sha256(encoded).hexdigest(),
        "class_counts": counts,
    }


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


def _local_epoch_at_global_offset(
    indices: list[int],
    *,
    global_offset: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> tuple[list[int], int, int]:
    """Reconstruct a rank's epoch view and exact consumed position."""

    local_epoch, padding = _padded_rank_epoch(
        indices, batch_size=batch_size, rank=rank, world_size=world_size
    )
    padded_count = len(local_epoch) * world_size
    divisor = batch_size * world_size
    if (
        type(global_offset) is not int
        or global_offset < 0
        or global_offset > padded_count
        or global_offset % divisor != 0
    ):
        raise ValueError("checkpoint sampler global offset is invalid")
    return local_epoch, global_offset // world_size, padding


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
    if getattr(dataset, "split", None) != "train":
        raise ValueError("rollout sequences require the training split")
    grouped: dict[str, dict[int, list[tuple[int, str]]]] = {}
    identities: dict[str, str] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        clip = str(sample["clip_id"])
        center = int(sample["center_frame"])
        terrain_class = str(sample["terrain_class"])
        identity = str(sample["split_identity"])
        if (
            not clip
            or center < 0
            or terrain_class not in _TERRAIN_CLASSES
            or not identity
            or sample.get("split") != "train"
            or identity != terrain_identity(clip)
            or split_identity(identity) != "train"
        ):
            raise ValueError("rollout sequence split/identity metadata is invalid")
        previous_identity = identities.setdefault(clip, identity)
        if previous_identity != identity:
            raise ValueError("rollout sequence split/identity mismatch")
        clip_rows = grouped.setdefault(clip, {})
        clip_rows.setdefault(center, []).append((index, terrain_class))
    sequences: list[tuple[int, ...]] = []
    classes: list[str] = []
    for clip in sorted(grouped):
        clip_rows = {
            center: rows[0]
            for center, rows in grouped[clip].items()
            if len(rows) == 1
        }
        centers = sorted(clip_rows)
        for start in range(len(centers) - frames + 1):
            window = centers[start : start + frames]
            if any(
                right != left + 1 for left, right in zip(window, window[1:])
            ):
                continue
            indices = tuple(clip_rows[center][0] for center in window)
            sequences.append(indices)
            classes.append(clip_rows[window[0]][1])
    if not sequences:
        raise ValueError("training split has no consecutive rollout sequences")
    return sequences, classes


def _append_metric(stream: object, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def train(
    arguments: argparse.Namespace,
    *,
    closed_loop_scorer: Callable[[], ClosedLoopValidationResult] | None = None,
) -> dict[str, object] | None:
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
            consecutive_overfit_subset(train_dataset, arguments.overfit_samples)
            if rank == 0 else None
        )
        candidate_indices = _broadcast_indices(candidate_indices, device)
        pipeline_overfit = True
        fitted_receipt = fitted_subset_metadata(train_dataset, candidate_indices)
        optimization_dataset = materialize_subset(train_dataset, candidate_indices)
        optimization_classes = _terrain_classes(optimization_dataset)
        optimization_indices = list(range(len(optimization_dataset)))
    else:
        candidate_indices = all_indices
        pipeline_overfit = False
        fitted_receipt = None
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
    optimizer = torch.optim.Adam(
        unwrapped.parameters(), lr=arguments.learning_rate, weight_decay=0.0
    )
    restored_step, restored_epoch = 0, 0
    restored_sampler_epoch, restored_sampler_global_offset = 0, 0
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
        restored_sampler_epoch = resumed.sampler_epoch
        restored_sampler_global_offset = resumed.sampler_global_offset
        if arguments.seed != resumed.seed:
            raise ValueError("resume checkpoint seed mismatch")
        if restored_step > 0 and restored_epoch != restored_sampler_epoch + 1:
            raise ValueError("resume checkpoint sampler epoch is inconsistent")
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
    if arguments.resume is not None:
        global_epoch = (
            _balanced_epoch_indices(
                optimization_indices,
                optimization_classes,
                seed=arguments.seed,
                epoch=restored_sampler_epoch,
            )
            if rank == 0 else None
        )
        global_epoch = _broadcast_indices(global_epoch, device)
        local_epoch, local_offset, padding = _local_epoch_at_global_offset(
            global_epoch,
            global_offset=restored_sampler_global_offset,
            batch_size=arguments.batch_size,
            rank=rank,
            world_size=world_size,
        )
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
        rollout_dataset = optimization_dataset if pipeline_overfit else train_dataset
        sequences, sequence_classes = _consecutive_starts(
            rollout_dataset, arguments.rollout_finetune_frames
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
            rows = [
                [rollout_dataset[index] for index in sequences[item]]
                for item in chosen
            ]
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
        optimization_dataset, kinematics.joint_limits.detach().cpu()
    )
    closed_loop_result: ClosedLoopValidationResult | None = None
    if pipeline_overfit:
        promote_best = promote_pipeline_best(
            pipeline_overfit=True, accepted=one_step_accepted
        )
        selection = (
            selection_metadata(
                one_step_score=float(gate_metrics["one_step_score"]),
                pipeline_overfit=True,
            )
            if promote_best
            else None
        )
    elif one_step_accepted and closed_loop_scorer is not None:
        closed_loop_result = closed_loop_scorer()
        selection, promote_best = validated_normal_selection(
            one_step_score=float(gate_metrics["one_step_score"]),
            result=closed_loop_result,
            dataset_digest=dataset_digest,
            kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
        )
    else:
        selection, promote_best = None, False
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
        sampler_epoch=max(0, epoch - 1),
        sampler_global_offset=local_offset * world_size,
        fitted_subset=fitted_receipt,
    )
    load_checkpoint(
        candidate,
        expected_dataset_digest=dataset_digest,
        expected_kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
    )
    best_path: str | None = None
    if promote_best:
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
            sampler_epoch=max(0, epoch - 1),
            sampler_global_offset=local_offset * world_size,
            fitted_subset=fitted_receipt,
        )
        best_path = str(best)
    report = {
        "schema": "mm-sonic-terrain-pfnn-one-step-report/v1",
        "dataset_digest_sha256": dataset_digest,
        "kinematic_signature_sha256": kinematics.kinematic_signature_sha256,
        "fixed_subset_samples": len(candidate_indices) if pipeline_overfit else None,
        "fitted_subset": fitted_receipt,
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
            "status": (
                "evaluated"
                if closed_loop_result is not None
                else (
                    "provisional_pipeline_overfit"
                    if pipeline_overfit
                    else "not_evaluated_task_8_scenarios_required"
                )
            ),
            "failure_penalty": (
                None
                if closed_loop_result is None
                else closed_loop_result.failure_penalty
            ),
            "accepted": (
                None
                if closed_loop_result is None
                else closed_loop_result.hard_failures == 0
            ),
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
    parser.add_argument("--rollout-finetune-frames", type=int, default=None)
    parser.add_argument("--rollout-finetune-steps", type=int)
    parser.add_argument("--resume")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.steps < 1 or arguments.batch_size < 1 or arguments.hidden_size < 1:
        raise ValueError("steps, batch size, and hidden size must be positive")
    arguments.rollout_finetune_frames = resolve_rollout_finetune_frames(
        arguments.rollout_finetune_frames,
        pipeline_overfit=arguments.overfit_samples is not None,
    )
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
    "consecutive_overfit_subset",
    "fitted_subset_metadata",
    "overfit_gate_accepted",
    "promote_pipeline_best",
    "validated_normal_selection",
    "resolve_rollout_finetune_frames",
    "seed_worker",
    "stratified_subset",
    "train",
]
