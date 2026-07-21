#!/usr/bin/env python3
"""Evaluate entry-pose proposal generality for the trained object/grasp."""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from resources.g1_interaction_builder.diffusion import sample_checkpoint
from resources.g1_interaction_builder.funnel_dataset import load_dataset
from resources.g1_interaction_builder.proposal_artifact import (
    PROPOSAL_COUNT,
    SAMPLE_COUNT,
    SAMPLE_WIDTH,
    certify_proposals,
    project_object_local_funnels,
)


BEARINGS_DEGREES = tuple(range(0, 360, 45))
RADII_METRES = (0.55, 0.75, 0.95)
YAW_OFFSETS_DEGREES = (-15, 0, 15)


def build_entry_conditions(
    base_condition: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Copy one trained object/grasp prefix across the fixed entry grid."""
    base = np.asarray(base_condition, dtype=np.float32)
    if base.shape != (24,) or not np.isfinite(base).all():
        raise ValueError("base_condition must be one finite 24-value row")

    conditions: list[np.ndarray] = []
    descriptors: list[dict[str, float]] = []
    for bearing_degrees in BEARINGS_DEGREES:
        bearing = math.radians(bearing_degrees)
        for radius_m in RADII_METRES:
            x = radius_m * math.sin(bearing)
            z = radius_m * math.cos(bearing)
            facing_yaw = math.atan2(-x, -z)
            for yaw_offset_degrees in YAW_OFFSETS_DEGREES:
                yaw = facing_yaw + math.radians(yaw_offset_degrees)
                condition = base.copy()
                condition[18:24] = np.asarray(
                    [x, z, math.sin(yaw), math.cos(yaw), 0.0, 0.0],
                    dtype=np.float32,
                )
                conditions.append(condition)
                descriptors.append({
                    "bearing_degrees": bearing_degrees,
                    "radius_m": radius_m,
                    "yaw_offset_degrees": yaw_offset_degrees,
                })
    return np.ascontiguousarray(conditions, dtype=np.float32), descriptors


def _as_numpy_samples(samples: object) -> np.ndarray:
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    result = np.asarray(samples, dtype=np.float32)
    expected = (72, PROPOSAL_COUNT, SAMPLE_COUNT, SAMPLE_WIDTH)
    if result.shape != expected:
        raise ValueError(f"sampler returned {result.shape}, expected {expected}")
    if not np.isfinite(result).all():
        raise ValueError("sampler returned non-finite proposals")
    return result


def evaluate_entry_sweep(
    checkpoint: Path,
    base_condition: np.ndarray,
    *,
    seed: int = 2_026_071_901,
    device: str = "cpu",
    sampler: Callable = sample_checkpoint,
) -> dict:
    """Sample and certify all trained-object entry conditions."""
    conditions, descriptors = build_entry_conditions(base_condition)
    outward = _as_numpy_samples(sampler(
        Path(checkpoint),
        torch.from_numpy(conditions),
        device=device,
        seed=seed,
    ))

    records: list[dict] = []
    accepted_counts: list[int] = []
    for index, (condition, descriptor) in enumerate(
        zip(conditions, descriptors)
    ):
        execution = project_object_local_funnels(
            np.ascontiguousarray(outward[index, :, ::-1], dtype=np.float32))
        expected_entry = np.broadcast_to(
            condition[18:22], execution[:, 0].shape)
        entry_exact = bool(np.array_equal(execution[:, 0], expected_entry))
        entry_error = float(np.max(np.abs(
            execution[:, 0].astype(np.float64) -
            expected_entry.astype(np.float64))))

        certified = certify_proposals(execution) if entry_exact else np.zeros(
            PROPOSAL_COUNT, dtype=bool)
        accepted_count = int(certified.sum())
        accepted_counts.append(accepted_count)
        failure_reason = None
        if not entry_exact:
            failure_reason = "entry_boundary_mismatch"
        elif accepted_count == 0:
            failure_reason = "no_accepted_proposals"

        accepted = execution[certified]
        if len(accepted):
            terminals = accepted[:, -1, :2].astype(np.float64)
            terminal_mean = terminals.mean(axis=0)
            terminal_std = terminals.std(axis=0)
            route_position_diversity = float(np.mean(
                np.std(accepted[:, :, :2].astype(np.float64), axis=0)))
        else:
            terminal_mean = None
            terminal_std = None
            route_position_diversity = None

        records.append({
            "index": index,
            **descriptor,
            "entry_condition": [float(value) for value in condition[18:24]],
            "entry_boundary_exact": entry_exact,
            "entry_boundary_max_abs_error": entry_error,
            "accepted_proposal_count": accepted_count,
            "failure_reason": failure_reason,
            "terminal_position_mean": (
                [float(value) for value in terminal_mean]
                if terminal_mean is not None else None),
            "terminal_position_std": (
                [float(value) for value in terminal_std]
                if terminal_std is not None else None),
            "route_position_diversity": route_position_diversity,
        })

    passing = sum(record["failure_reason"] is None for record in records)
    return {
        "schema_version": 1,
        "scope": "trained_object_entry_pose_proposal_stage",
        "checkpoint": str(Path(checkpoint)),
        "seed": seed,
        "device": device,
        "condition_count": len(records),
        "proposal_count_per_condition": PROPOSAL_COUNT,
        "passing_condition_count": passing,
        "failing_condition_count": len(records) - passing,
        "minimum_accepted_proposals": min(accepted_counts),
        "median_accepted_proposals": float(np.median(accepted_counts)),
        "conditions": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack", type=Path,
        default=Path("build/smart-pickup/full-pack"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("build/g1-funnels/checkpoint-schema3-20k.pt"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "build/g1-funnels/evidence/trained-object-entry-sweep.json"))
    parser.add_argument("--seed", type=int, default=2_026_071_901)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    dataset = load_dataset(args.pack)
    if len(dataset.conditions) == 0:
        raise ValueError("funnel training pack contains no conditions")
    report = evaluate_entry_sweep(
        args.checkpoint,
        dataset.conditions[0],
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"evaluated {report['condition_count']} conditions: "
        f"{report['passing_condition_count']} passed, minimum accepted "
        f"{report['minimum_accepted_proposals']}/32")


if __name__ == "__main__":
    main()
