"""Fine-tune a classic native-G1 PFNN through its exact runtime recurrence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

from .build_g1_pfnn_vertical_dataset import load_vertical_dataset
from .terrain_pfnn.kinematics import TorchG1ForwardKinematics
from .terrain_pfnn.training import (
    DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
    autoregressive_unroll,
)
from .train_classic_g1_pfnn import (
    _VerticalTrainingView,
    _evaluate,
    _materialize_vertical,
    classic_pfnn_loss,
    load_classic_checkpoint,
    save_classic_checkpoint,
)


@dataclass(frozen=True)
class VerticalRolloutSequence:
    indices: tuple[int, ...]
    predecessor_index: int


def vertical_rollout_sequences(
    view: _VerticalTrainingView, frames: int
) -> tuple[VerticalRolloutSequence, ...]:
    if getattr(view, "split", None) != "train" or type(frames) is not int or frames < 2:
        raise ValueError("vertical rollout sequence contract is invalid")
    grouped: dict[tuple[str, str], dict[int, int]] = {}
    for index in range(len(view)):
        row = view[index]
        key = (str(row["clip_id"]), str(row["sequence_lane"]))
        center = int(row["center_frame"])
        lane = grouped.setdefault(key, {})
        if center in lane:
            raise ValueError("vertical rollout sequence has duplicate rows")
        lane[center] = index
    result: list[VerticalRolloutSequence] = []
    for key in sorted(grouped):
        lane = grouped[key]
        centers = sorted(lane)
        for offset in range(len(centers) - frames + 1):
            selected = centers[offset : offset + frames]
            if selected[0] - 1 not in lane or any(
                right != left + 1 for left, right in zip(selected, selected[1:])
            ):
                continue
            result.append(
                VerticalRolloutSequence(
                    indices=tuple(lane[center] for center in selected),
                    predecessor_index=lane[selected[0] - 1],
                )
            )
    if not result:
        raise ValueError("vertical corpus has no rollout sequences")
    return tuple(result)


def _rollout_batch(
    view: _VerticalTrainingView,
    sequences: tuple[VerticalRolloutSequence, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [[view[index] for index in sequence.indices] for sequence in sequences]
    frames = len(sequences[0].indices)
    return (
        torch.as_tensor(
            np.stack([[row[frame]["x"] for row in rows] for frame in range(frames)]),
            device=device,
        ),
        torch.as_tensor(
            np.asarray([row[0]["phase"] for row in rows], dtype=np.float32),
            device=device,
        ),
        torch.as_tensor(
            np.stack([[row[frame]["y"] for row in rows] for frame in range(frames)]),
            device=device,
        ),
        torch.as_tensor(
            np.stack([view[sequence.predecessor_index]["y"] for sequence in sequences]),
            device=device,
        ),
    )


def rollout_fine_tune(arguments: argparse.Namespace) -> Path:
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = torch.device(arguments.device)
    checkpoint = load_classic_checkpoint(arguments.checkpoint)
    dataset = load_vertical_dataset(Path(arguments.dataset).expanduser().resolve().parent)
    if dataset.dataset_sha256 != checkpoint.dataset_digest:
        raise ValueError("classic PFNN rollout dataset digest mismatch")
    view = _VerticalTrainingView(dataset, "train")
    sequences = vertical_rollout_sequences(view, arguments.frames)
    kinematics = TorchG1ForwardKinematics.from_mjcf(arguments.model_path)
    if kinematics.kinematic_signature_sha256 != checkpoint.kinematic_signature_sha256:
        raise ValueError("classic PFNN rollout kinematic signature mismatch")
    normal = {
        name: value.to(device=device, dtype=torch.float32)
        for name, value in checkpoint.normalization.items()
    }
    limits = kinematics.joint_limits.to(device=device, dtype=torch.float32)
    phase_cap = min(math.pi, 1.5 * checkpoint.phase_advance_q99)
    model = checkpoint.build_model().to(device)
    model.dropout.p = float(arguments.dropout_probability)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    metrics_path = output / "metrics.jsonl"
    generator = np.random.default_rng(arguments.seed)
    order = generator.permutation(len(sequences))
    cursor = 0
    with metrics_path.open("w", encoding="utf-8") as stream:
        for step in range(arguments.steps):
            if cursor + arguments.batch_size > len(order):
                order = generator.permutation(len(sequences))
                cursor = 0
            selected = order[cursor : cursor + arguments.batch_size]
            cursor += arguments.batch_size
            records = tuple(sequences[int(index)] for index in selected)
            inputs, phases, targets, predecessor = _rollout_batch(
                view, records, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            rollout = autoregressive_unroll(
                model,
                inputs,
                phases,
                targets,
                initial_predecessor_targets=predecessor,
                normalization=normal,
                phase_advance_cap=phase_cap,
                joint_limits=limits,
                envelope_contract=DEFAULT_PHYSICAL_ENVELOPE_OBJECTIVE,
                global_batch_ordinals=torch.arange(
                    len(records), dtype=torch.int64, device=device
                ),
                rank=0,
                world_size=1,
            )
            base = torch.stack(
                [
                    classic_pfnn_loss(prediction, targets[index])
                    for index, prediction in enumerate(rollout.predictions)
                ]
            ).mean()
            loss = base + float(arguments.envelope_weight) * rollout.losses[
                "physical_envelope_total"
            ]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("classic PFNN rollout loss is nonfinite")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("classic PFNN rollout gradient is nonfinite")
            optimizer.step()
            if step == 0 or (step + 1) % arguments.trace_every == 0:
                record = {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "classic_loss": float(base.detach()),
                    "physical_envelope_loss": float(
                        rollout.losses["physical_envelope_total"].detach()
                    ),
                    "gradient_norm": float(gradient.detach()),
                    "sequence_count": len(sequences),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
    validation_view = _VerticalTrainingView(dataset, "validation")
    validation_x, validation_y, validation_phase = _materialize_vertical(
        validation_view
    )
    validation_loss = _evaluate(
        model,
        validation_x,
        validation_y,
        validation_phase,
        device=device,
        batch_size=max(arguments.batch_size, 256),
    )
    if not math.isfinite(validation_loss):
        raise FloatingPointError("classic PFNN rollout validation is nonfinite")
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "step": arguments.steps,
                    "validation_loss": validation_loss,
                    "validation_samples": len(validation_view),
                },
                sort_keys=True,
            )
            + "\n"
        )
    best = output / "best.pt"
    save_classic_checkpoint(
        best,
        model=model,
        normalization=checkpoint.normalization,
        runtime_seed=checkpoint.runtime_seed,
        dataset_digest=checkpoint.dataset_digest,
        kinematic_signature_sha256=checkpoint.kinematic_signature_sha256,
        joint_limits=checkpoint.joint_limits,
        phase_advance_q99=checkpoint.phase_advance_q99,
        epoch=checkpoint.epoch + arguments.steps,
        validation_loss=validation_loss,
        seed=arguments.seed,
        source_kind=checkpoint.source_kind,
        vertical_slice_receipt_sha256=checkpoint.vertical_slice_receipt_sha256,
        terrain_receipt_set_sha256=checkpoint.terrain_receipt_set_sha256,
    )
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--envelope-weight", type=float, default=32.0)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    parser.add_argument("--trace-every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=56789)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    values = (
        arguments.learning_rate,
        arguments.envelope_weight,
        arguments.dropout_probability,
    )
    if (
        arguments.frames < 2
        or arguments.steps < 1
        or arguments.batch_size < 1
        or arguments.trace_every < 1
        or not all(math.isfinite(value) for value in values)
        or arguments.learning_rate <= 0.0
        or arguments.envelope_weight <= 0.0
        or not 0.0 <= arguments.dropout_probability < 1.0
    ):
        raise ValueError("classic PFNN rollout arguments are invalid")
    path = rollout_fine_tune(arguments)
    print(json.dumps({"status": "accepted", "checkpoint": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
