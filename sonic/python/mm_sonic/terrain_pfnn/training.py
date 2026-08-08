"""Loss, autoregressive-training, and safe-checkpoint contracts for G1 PFNN."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)

from .dataset import normalize_pfnn_input, pfnn_input_sha256
from .layout import (
    CONTACT_ORDER,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
    TRAJECTORY_TIMES_S,
)
from .model import PhaseFunctionedNetwork
from .recurrence import (
    PlannedTrajectory,
    advance_recurrent_state,
    derive_training_desired_velocity,
    initialize_recurrent_state,
    pack_recurrent_input,
    plan_recurrent_trajectory,
)


CHECKPOINT_SCHEMA = "mm-sonic-terrain-pfnn-checkpoint/v5"
_SEQUENCE_LANES = ("motion", *(f"idle_phase_{index}" for index in range(8)))
_NORMALIZATION_CONTRACT = {
    "continuous_loss_domain": "normalized",
    "contact_loss_domain": "raw_logits_binary_labels",
    "structural_loss_domain": "physical_denormalized",
    "recurrent_body_input_scale": 0.1,
    "root_tilt_encoding": "angle_axis_xy",
}
LOSS_WEIGHT_KEYS = (
    "trajectory_mse",
    "body_mse",
    "root_pose_mse",
    "joint_mse",
    "root_motion_mse",
    "phase_mse",
    "contact_bce",
    "trajectory_direction",
    "phase_nonnegative",
    "fk_consistency",
    "regularization",
)
DEFAULT_LOSS_WEIGHTS = {name: 1.0 for name in LOSS_WEIGHT_KEYS}
_STATE_NAMES = ("W0", "b0", "W1", "b1", "W2", "b2")
# Exact torch.optim.Adam group schema for the pinned trainer configuration.
_ADAM_PARAM_GROUP_KEYS = frozenset(
    {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "params",
    }
)
_ADAM_BOOL_FLAGS = (
    "amsgrad",
    "maximize",
    "capturable",
    "differentiable",
    "decoupled_weight_decay",
)
_ADAM_OPTIONAL_BOOL_FLAGS = ("foreach", "fused")


def _as_tensor(
    value: object, *, device: torch.device, dtype: torch.dtype, name: str
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"normalization {name} is invalid") from error
    if not torch.isfinite(tensor).all():
        raise ValueError(f"normalization {name} must be finite")
    return tensor


def _normalization_tensors(
    normalization: object, *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    expected = {
        "x_mean": INPUT_LAYOUT.size,
        "x_std": INPUT_LAYOUT.size,
        "y_mean": OUTPUT_LAYOUT.size,
        "y_std": OUTPUT_LAYOUT.size,
    }
    for name, width in expected.items():
        if isinstance(normalization, Mapping):
            if name not in normalization:
                raise ValueError(f"normalization is missing {name}")
            source = normalization[name]
        else:
            try:
                source = getattr(normalization, name)
            except AttributeError as error:
                raise ValueError(f"normalization is missing {name}") from error
        tensor = _as_tensor(source, device=device, dtype=dtype, name=name)
        if tensor.shape != (width,):
            raise ValueError(f"normalization {name} has invalid shape")
        if name.endswith("_std") and torch.any(tensor <= 0.0):
            raise ValueError(f"normalization {name} must be positive")
        output[name] = tensor
    contact = OUTPUT_LAYOUT["contact_logit"]
    if not torch.equal(
        output["y_mean"][contact], torch.zeros(4, device=device, dtype=dtype)
    ) or not torch.equal(
        output["y_std"][contact], torch.ones(4, device=device, dtype=dtype)
    ):
        raise ValueError("contact normalization must be exactly mean=0/std=1")
    return output


def _checkpoint_normalization_tensors(value: object) -> dict[str, torch.Tensor]:
    expected = {
        "x_mean": INPUT_LAYOUT.size,
        "x_std": INPUT_LAYOUT.size,
        "y_mean": OUTPUT_LAYOUT.size,
        "y_std": OUTPUT_LAYOUT.size,
    }
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("checkpoint normalization fields are invalid")
    output: dict[str, torch.Tensor] = {}
    for name, width in expected.items():
        tensor = value[name]
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.layout != torch.strided
            or tensor.shape != (width,)
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or not torch.isfinite(tensor).all()
            or (name.endswith("_std") and torch.any(tensor <= 0.0))
        ):
            raise ValueError(f"checkpoint normalization {name} is invalid")
        output[name] = tensor.clone()
    contact = OUTPUT_LAYOUT["contact_logit"]
    if not torch.equal(output["y_mean"][contact], torch.zeros(4)) or not torch.equal(
        output["y_std"][contact], torch.ones(4)
    ):
        raise ValueError("checkpoint contact normalization is invalid")
    return output


def _field_cat(value: torch.Tensor, fields: Sequence[str]) -> torch.Tensor:
    return torch.cat([value[..., OUTPUT_LAYOUT[name]] for name in fields], dim=-1)


def _loss_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    candidate = DEFAULT_LOSS_WEIGHTS if weights is None else weights
    if type(candidate) is not dict or set(candidate) != set(LOSS_WEIGHT_KEYS):
        raise ValueError("loss weights do not match the immutable loss groups")
    if any(type(candidate[name]) is not float or candidate[name] != 1.0
           for name in LOSS_WEIGHT_KEYS):
        raise ValueError("loss weights are frozen to float 1.0")
    return {name: 1.0 for name in LOSS_WEIGHT_KEYS}


def pfnn_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    model: nn.Module,
    normalization: object,
    kinematics: nn.Module | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute separately reported PFNN objectives in their approved domains."""

    if (
        prediction.ndim != 2
        or prediction.shape[1] != OUTPUT_LAYOUT.size
        or target.shape != prediction.shape
    ):
        raise ValueError("prediction and target must have shape [B,268]")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must be floating tensors")
    normal = _normalization_tensors(
        normalization, device=prediction.device, dtype=prediction.dtype
    )
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    physical = prediction * normal["y_std"] + normal["y_mean"]

    mse_groups = {
        "trajectory_mse": ("trajectory_position", "trajectory_direction"),
        "body_mse": ("body_position", "body_velocity"),
        "root_pose_mse": ("root_height", "root_tilt"),
        "joint_mse": ("joint_position",),
        "root_motion_mse": ("root_planar_velocity", "root_yaw_velocity"),
        "phase_mse": ("phase_advance",),
    }
    losses = {
        name: F.mse_loss(_field_cat(prediction, fields), _field_cat(target, fields))
        for name, fields in mse_groups.items()
    }
    contact = OUTPUT_LAYOUT["contact_logit"]
    labels = target[:, contact]
    if torch.any((labels != 0.0) & (labels != 1.0)):
        raise ValueError("contact targets must be binary labels")
    losses["contact_bce"] = F.binary_cross_entropy_with_logits(
        prediction[:, contact], labels
    )
    direction = physical[:, OUTPUT_LAYOUT["trajectory_direction"]].reshape(-1, 12, 2)
    losses["trajectory_direction"] = (
        torch.linalg.vector_norm(direction, dim=-1) - 1.0
    ).square().mean()
    phase_advance = physical[:, OUTPUT_LAYOUT["phase_advance"]]
    losses["phase_nonnegative"] = torch.relu(-phase_advance).square().mean()
    if kinematics is None:
        losses["fk_consistency"] = prediction.sum() * 0.0
    else:
        root = torch.cat(
            (
                physical[:, OUTPUT_LAYOUT["root_height"]],
                physical[:, OUTPUT_LAYOUT["root_tilt"]],
            ),
            dim=1,
        )
        joints = physical[:, OUTPUT_LAYOUT["joint_position"]]
        expected_body = kinematics(root, joints)
        predicted_body = physical[:, OUTPUT_LAYOUT["body_position"]].reshape(-1, 30, 3)
        losses["fk_consistency"] = F.mse_loss(predicted_body, expected_body)

    regularized_model = model.module if hasattr(model, "module") else model
    parameters = [getattr(regularized_model, name) for name in _STATE_NAMES]
    if any(not isinstance(parameter, torch.Tensor) for parameter in parameters):
        raise ValueError("model does not expose the six PFNN parameter banks")
    absolute_sum = sum(parameter.abs().sum() for parameter in parameters)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    losses["regularization"] = 0.01 * absolute_sum / parameter_count
    weights = _loss_weights(loss_weights)
    losses["total"] = sum(losses[name] * weights[name] for name in LOSS_WEIGHT_KEYS)
    return losses


@dataclass(frozen=True)
class RolloutResult:
    losses: dict[str, torch.Tensor]
    predictions: tuple[torch.Tensor, ...]
    inputs: tuple[torch.Tensor, ...]
    phases: tuple[torch.Tensor, ...]


def autoregressive_unroll(
    model: nn.Module,
    ground_truth_inputs: torch.Tensor,
    initial_phase: torch.Tensor,
    targets: torch.Tensor,
    *,
    normalization: object,
    phase_advance_cap: float,
    kinematics: nn.Module | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> RolloutResult:
    """Unroll a consecutive training-only sequence without detaching feedback."""

    if (
        ground_truth_inputs.ndim != 3
        or ground_truth_inputs.shape[2] != INPUT_LAYOUT.size
        or targets.shape != (
            ground_truth_inputs.shape[0], ground_truth_inputs.shape[1], OUTPUT_LAYOUT.size
        )
        or initial_phase.shape != (ground_truth_inputs.shape[1],)
    ):
        raise ValueError("rollout expects x[T,B,288], phase[B], and y[T,B,268]")
    normal = _normalization_tensors(
        normalization,
        device=ground_truth_inputs.device,
        dtype=ground_truth_inputs.dtype,
    )
    if (
        isinstance(phase_advance_cap, bool)
        or not isinstance(phase_advance_cap, (int, float))
        or not math.isfinite(float(phase_advance_cap))
        or float(phase_advance_cap) < 0.0
    ):
        raise ValueError("phase_advance_cap must be finite and nonnegative")

    def physical_input(normalized: torch.Tensor) -> torch.Tensor:
        unscaled = normalized.clone()
        for field in ("previous_body_position", "previous_body_velocity"):
            unscaled[:, INPUT_LAYOUT[field]] = (
                unscaled[:, INPUT_LAYOUT[field]] / 0.1
            )
        physical = unscaled * normal["x_std"] + normal["x_mean"]
        semantic = physical[:, INPUT_LAYOUT["semantic_intent"]].reshape(
            len(normalized), 12, 2
        )
        physical[:, INPUT_LAYOUT["semantic_intent"]] = F.one_hot(
            torch.argmax(semantic, dim=-1), num_classes=2
        ).to(physical.dtype).reshape(len(normalized), -1)
        return physical

    batch_size = int(ground_truth_inputs.shape[1])
    first_physical = physical_input(ground_truth_inputs[0])
    recurrent_state = initialize_recurrent_state(
        trajectory_position_local=first_physical[
            :, INPUT_LAYOUT["trajectory_position"]
        ].reshape(batch_size, 12, 2),
        trajectory_direction_local=first_physical[
            :, INPUT_LAYOUT["trajectory_direction"]
        ].reshape(batch_size, 12, 2),
        semantic_intent=first_physical[
            :, INPUT_LAYOUT["semantic_intent"]
        ].reshape(batch_size, 12, 2),
        previous_body_position_local=first_physical[
            :, INPUT_LAYOUT["previous_body_position"]
        ].reshape(batch_size, 30, 3),
        previous_body_velocity_local=first_physical[
            :, INPUT_LAYOUT["previous_body_velocity"]
        ].reshape(batch_size, 30, 3),
        phase=initial_phase,
        root_world_xy=torch.zeros(
            (batch_size, 2),
            dtype=ground_truth_inputs.dtype,
            device=ground_truth_inputs.device,
        ),
        root_yaw_world=torch.zeros(
            batch_size,
            dtype=ground_truth_inputs.dtype,
            device=ground_truth_inputs.device,
        ),
    )
    planned = PlannedTrajectory(
        position_world_xy=recurrent_state.predicted_position_world_xy,
        direction_world_xy=recurrent_state.predicted_direction_world_xy,
        semantic_intent=first_physical[
            :, INPUT_LAYOUT["semantic_intent"]
        ].reshape(batch_size, 12, 2),
    )
    cap = torch.full(
        (batch_size,),
        float(phase_advance_cap),
        dtype=ground_truth_inputs.dtype,
        device=ground_truth_inputs.device,
    )
    current_input = ground_truth_inputs[0]
    predictions: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []
    phases: list[torch.Tensor] = []
    per_step: list[dict[str, torch.Tensor]] = []
    for step in range(len(ground_truth_inputs)):
        inputs.append(current_input)
        phases.append(recurrent_state.phase)
        prediction = model(current_input, recurrent_state.phase)
        predictions.append(prediction)
        per_step.append(
            pfnn_losses(
                prediction,
                targets[step],
                model=model,
                normalization=normal,
                kinematics=kinematics,
                loss_weights=loss_weights,
            )
        )
        if step + 1 < len(ground_truth_inputs):
            physical_output = prediction * normal["y_std"] + normal["y_mean"]
            recurrent_state = advance_recurrent_state(
                recurrent_state,
                planned,
                physical_output,
                phase_advance_cap=cap,
            )
            next_physical = physical_input(ground_truth_inputs[step + 1])
            desired_velocity = derive_training_desired_velocity(
                next_physical[:, INPUT_LAYOUT["trajectory_position"]].reshape(
                    batch_size, 12, 2
                ),
                next_physical[:, INPUT_LAYOUT["semantic_intent"]].reshape(
                    batch_size, 12, 2
                ),
                recurrent_state.root_yaw_world,
            )
            planned = plan_recurrent_trajectory(
                recurrent_state, desired_velocity
            )
            current_input = pack_recurrent_input(
                state=recurrent_state,
                planned=planned,
                terrain_height=next_physical[
                    :, INPUT_LAYOUT["terrain_height"]
                ].reshape(batch_size, 12, 3),
                x_mean=normal["x_mean"],
                x_std=normal["x_std"],
                body_scale=0.1,
            )
    losses = {
        name: torch.stack([item[name] for item in per_step]).mean()
        for name in per_step[0]
    }
    return RolloutResult(
        losses=losses,
        predictions=tuple(predictions),
        inputs=tuple(inputs),
        phases=tuple(phases),
    )


def training_phase_advance_q99(dataset: object) -> float:
    """Compute the physical phase-advance q99 from a training split only."""

    if getattr(dataset, "split", None) != "train":
        raise ValueError("phase_advance_q99 may only be computed from training data")
    values = np.empty(len(dataset), dtype=np.float64)
    mean = float(np.asarray(dataset.y_mean)[OUTPUT_LAYOUT["phase_advance"]][0])
    std = float(np.asarray(dataset.y_std)[OUTPUT_LAYOUT["phase_advance"]][0])
    for index in range(len(dataset)):
        sample = dataset[index]
        values[index] = float(np.asarray(sample["y"])[OUTPUT_LAYOUT["phase_advance"]][0]) * std + mean
    if not np.isfinite(values).all():
        raise ValueError("training phase advances are invalid")
    # PFNNShardDataset has already verified that the sealed physical targets
    # are nonnegative.  A raw zero can reconstruct a few ulps below zero after
    # the mandated float32 normalize/denormalize round trip.
    values = np.maximum(values, 0.0)
    return float(np.quantile(values, 0.99, method="linear"))


def fitted_row_sha256(sample: Mapping[str, object]) -> str:
    """Hash one normalized fitted row and its sealed training provenance."""

    try:
        x = np.ascontiguousarray(np.asarray(sample["x"], dtype="<f4"))
        y = np.ascontiguousarray(np.asarray(sample["y"], dtype="<f4"))
        phase = np.asarray(float(sample["phase"]), dtype="<f4")
        provenance = {
            "clip_id": sample["clip_id"],
            "center_frame": sample["center_frame"],
            "split_identity": sample["split_identity"],
            "split": sample["split"],
            "sequence_lane": sample["sequence_lane"],
            "terrain_class": sample["terrain_class"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fitted row receipt source is invalid") from error
    if (
        x.shape != (INPUT_LAYOUT.size,)
        or y.shape != (OUTPUT_LAYOUT.size,)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(phase)
        or type(provenance["clip_id"]) is not str
        or not provenance["clip_id"]
        or type(provenance["center_frame"]) is not int
        or provenance["center_frame"] < 0
        or type(provenance["split_identity"]) is not str
        or not provenance["split_identity"]
        or provenance["split"] != "train"
        or provenance["sequence_lane"] not in _SEQUENCE_LANES
        or provenance["terrain_class"]
        not in ("flat", "ascent", "descent", "transition")
    ):
        raise ValueError("fitted row receipt source is invalid")
    digest = hashlib.sha256(b"mm-sonic-fitted-pfnn-row/v2\0")
    digest.update(
        json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    digest.update(x.tobytes(order="C"))
    digest.update(y.tobytes(order="C"))
    digest.update(phase.tobytes())
    return digest.hexdigest()


def choose_runtime_seed(
    dataset: object,
    joint_limits: object,
    *,
    fitted_subset: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Choose a fitted recurrent state whose first inference is also fitted.

    A candidate row ``t`` must have its unique same-clip, same-lane predecessor
    ``t-1`` present in this dataset.  The seed is predecessor ``y`` (the reached
    state at ``t``), and its recurrent body/trajectory/phase are checked against
    row ``t`` before it can be serialized.  Thus passing a materialized overfit
    dataset prevents seed selection from escaping the fitted rows.
    """

    if getattr(dataset, "split", None) != "train":
        raise ValueError("runtime seed may only be selected from training data")
    limits = torch.as_tensor(joint_limits, dtype=torch.float64, device="cpu")
    y_mean = np.asarray(dataset.y_mean, dtype=np.float64)
    y_std = np.asarray(dataset.y_std, dtype=np.float64)
    x_mean = np.asarray(dataset.x_mean, dtype=np.float64)
    x_std = np.asarray(dataset.x_std, dtype=np.float64)
    if (
        x_mean.shape != (INPUT_LAYOUT.size,)
        or x_std.shape != (INPUT_LAYOUT.size,)
        or not np.isfinite(x_mean).all()
        or not np.isfinite(x_std).all()
        or np.any(x_std <= 0.0)
    ):
        raise ValueError("runtime seed input normalization is invalid")
    rows: dict[tuple[str, str, int], list[tuple[int, Mapping[str, object]]]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        clip = str(sample.get("clip_id", ""))
        center = sample.get("center_frame")
        if (
            not clip
            or type(center) is not int
            or center < 0
            or sample.get("split") != "train"
            or sample.get("sequence_lane") not in _SEQUENCE_LANES
        ):
            raise ValueError("runtime seed fitted row provenance is invalid")
        lane = str(sample["sequence_lane"])
        rows.setdefault((clip, lane, center), []).append((index, sample))

    unique = {key: value[0] for key, value in rows.items() if len(value) == 1}
    candidates: list[
        tuple[
            tuple[float, str, str, int, str],
            Mapping[str, object],
            Mapping[str, object],
        ]
    ] = []
    for (clip, lane, center), (_, sample) in unique.items():
        predecessor_entry = unique.get((clip, lane, center - 1))
        has_rollout_horizon = all(
            (clip, lane, center + offset) in unique for offset in range(16)
        )
        if (
            sample.get("terrain_class") != "flat"
            or predecessor_entry is None
            or not has_rollout_horizon
        ):
            continue
        physical_target = np.asarray(sample["y"], dtype=np.float64) * y_std + y_mean
        predecessor = predecessor_entry[1]
        predecessor_physical = (
            np.asarray(predecessor["y"], dtype=np.float64) * y_std + y_mean
        )
        predecessor_joints = predecessor_physical[OUTPUT_LAYOUT["joint_position"]]
        if (
            not np.isfinite(physical_target).all()
            or not np.isfinite(predecessor_physical).all()
            or np.any(predecessor_joints < limits[:, 0].numpy())
            or np.any(predecessor_joints > limits[:, 1].numpy())
        ):
            continue
        speed = float(
            np.linalg.norm(physical_target[OUTPUT_LAYOUT["root_planar_velocity"]])
        )
        candidates.append(
            (
                (speed, clip, lane, center, fitted_row_sha256(sample)),
                predecessor,
                sample,
            )
        )
    if not candidates:
        raise ValueError("training split has no fitted consecutive flat runtime seed")
    key, predecessor, first_fitted = min(candidates, key=lambda candidate: candidate[0])
    speed, clip, _, center, _ = key
    predecessor_physical = (
        np.asarray(predecessor["y"], dtype=np.float64) * y_std + y_mean
    )
    joints = predecessor_physical[OUTPUT_LAYOUT["joint_position"]]
    if (
        not np.isfinite(predecessor_physical).all()
        or np.any(joints < limits[:, 0].numpy())
        or np.any(joints > limits[:, 1].numpy())
    ):
        raise ValueError("fitted runtime seed is nonfinite or outside joint limits")

    first_normalized = np.asarray(first_fitted["x"], dtype=np.float32)
    first_input = np.asarray(first_normalized, dtype=np.float64).copy()
    if first_input.shape != (INPUT_LAYOUT.size,) or not np.isfinite(first_input).all():
        raise ValueError("first fitted recurrent input is invalid")
    for field in ("previous_body_position", "previous_body_velocity"):
        first_input[INPUT_LAYOUT[field]] /= 0.1
    first_input = first_input * x_std + x_mean
    recurrent_pairs = (
        ("trajectory_position", "trajectory_position"),
        ("trajectory_direction", "trajectory_direction"),
        ("body_position", "previous_body_position"),
        ("body_velocity", "previous_body_velocity"),
    )
    for output_name, input_name in recurrent_pairs:
        left = predecessor_physical[OUTPUT_LAYOUT[output_name]]
        right = first_input[INPUT_LAYOUT[input_name]]
        if not np.allclose(left, right, rtol=3.0e-5, atol=3.0e-5):
            raise ValueError(
                f"fitted recurrent {output_name} does not reconstruct first input"
            )
    phase_advance = max(
        0.0, float(predecessor_physical[OUTPUT_LAYOUT["phase_advance"]][0])
    )
    phase = (float(predecessor["phase"]) + phase_advance) % (2.0 * math.pi)
    first_phase = float(first_fitted["phase"])
    phase_error = abs((phase - first_phase + math.pi) % (2.0 * math.pi) - math.pi)
    if phase_error > 3.0e-5:
        raise ValueError("fitted recurrent phase does not reconstruct first input")
    reconstructed_semantic = first_input[
        INPUT_LAYOUT["semantic_intent"]
    ].reshape(12, 2)
    semantic = np.zeros((12, 2), dtype=np.float64)
    semantic[np.arange(12), np.argmax(reconstructed_semantic, axis=1)] = 1.0
    if not np.allclose(
        reconstructed_semantic, semantic, rtol=0.0, atol=3.0e-5
    ):
        raise ValueError("first fitted semantic intent is not one-hot")
    expected_terrain = first_input[INPUT_LAYOUT["terrain_height"]].reshape(12, 3).copy()
    reconstructed = np.empty(INPUT_LAYOUT.size, dtype=np.float32)
    reconstructed[INPUT_LAYOUT["trajectory_position"]] = predecessor_physical[
        OUTPUT_LAYOUT["trajectory_position"]
    ]
    reconstructed[INPUT_LAYOUT["trajectory_direction"]] = predecessor_physical[
        OUTPUT_LAYOUT["trajectory_direction"]
    ]
    reconstructed[INPUT_LAYOUT["terrain_height"]] = expected_terrain.reshape(-1)
    reconstructed[INPUT_LAYOUT["semantic_intent"]] = semantic.reshape(-1)
    reconstructed[INPUT_LAYOUT["previous_body_position"]] = predecessor_physical[
        OUTPUT_LAYOUT["body_position"]
    ]
    reconstructed[INPUT_LAYOUT["previous_body_velocity"]] = predecessor_physical[
        OUTPUT_LAYOUT["body_velocity"]
    ]
    reconstructed_normalized = normalize_pfnn_input(
        reconstructed, dataset.x_mean, dataset.x_std
    )
    if not np.allclose(
        reconstructed_normalized, first_normalized, rtol=0.0, atol=3.0e-5
    ):
        raise ValueError("runtime seed does not reproduce first fitted input")
    best = {
            "phase": torch.tensor(phase, dtype=torch.float32),
            "world_xy": torch.zeros(2, dtype=torch.float32),
            "world_yaw": torch.tensor(0.0, dtype=torch.float32),
            "root_height": torch.tensor(
                float(predecessor_physical[OUTPUT_LAYOUT["root_height"]][0]),
                dtype=torch.float32,
            ),
            "root_tilt": torch.as_tensor(
                predecessor_physical[OUTPUT_LAYOUT["root_tilt"]].copy(), dtype=torch.float32
            ),
            "joint_position": torch.as_tensor(joints.copy(), dtype=torch.float32),
            "body_position": torch.as_tensor(
                predecessor_physical[OUTPUT_LAYOUT["body_position"]].reshape(30, 3).copy(),
                dtype=torch.float32,
            ),
            "body_velocity": torch.as_tensor(
                predecessor_physical[OUTPUT_LAYOUT["body_velocity"]].reshape(30, 3).copy(),
                dtype=torch.float32,
            ),
            "trajectory_position": torch.as_tensor(
                predecessor_physical[OUTPUT_LAYOUT["trajectory_position"]].reshape(12, 2).copy(),
                dtype=torch.float32,
            ),
            "trajectory_direction": torch.as_tensor(
                predecessor_physical[OUTPUT_LAYOUT["trajectory_direction"]].reshape(12, 2).copy(),
                dtype=torch.float32,
            ),
            "contact_label": torch.as_tensor(
                predecessor_physical[OUTPUT_LAYOUT["contact_logit"]].copy(), dtype=torch.float32
            ),
            "semantic_intent": torch.as_tensor(semantic, dtype=torch.float32),
            "terrain_height": torch.as_tensor(expected_terrain, dtype=torch.float32),
            "normalized_input": torch.as_tensor(
                first_normalized.copy(), dtype=torch.float32
            ),
            "normalized_input_sha256": pfnn_input_sha256(first_normalized),
            "provenance": {
                "predecessor_clip_id": str(predecessor["clip_id"]),
                "predecessor_center_frame": int(predecessor["center_frame"]),
                "predecessor_sequence_lane": str(predecessor["sequence_lane"]),
                "first_fitted_clip_id": clip,
                "first_fitted_center_frame": center,
                "first_fitted_sequence_lane": str(first_fitted["sequence_lane"]),
                "speed": speed,
                "predecessor_row_sha256": fitted_row_sha256(predecessor),
                "first_fitted_row_sha256": fitted_row_sha256(first_fitted),
                "fitted_subset_rows_sha256": (
                    "0" * 64
                    if fitted_subset is None
                    else str(fitted_subset.get("rows_sha256", ""))
                ),
            },
        }
    checked = _validate_runtime_seed(best, limits)
    if fitted_subset is not None:
        validate_runtime_seed_fitted_subset(checked, fitted_subset)
    return checked


def _sample_batch(
    dataset: object, indices: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [dataset[int(index)] for index in indices]
    return (
        torch.as_tensor(
            np.stack([sample["x"] for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.asarray([sample["phase"] for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            np.stack([sample["y"] for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
    )


def one_step_metrics(
    model: nn.Module,
    dataset: object,
    *,
    kinematics: nn.Module,
    indices: Sequence[int] | None = None,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
    loss_weights: Mapping[str, float] | None = None,
    distributed: bool = False,
) -> dict[str, float | int]:
    """Aggregate validation terms without regularization or batch-size bias."""

    target_device = torch.device(device)
    selected = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    if not selected:
        raise ValueError("one-step evaluation requires at least one sample")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rank, world_size = 0, 1
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("distributed validation requires an initialized process group")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    selected = selected[rank::world_size]
    names = LOSS_WEIGHT_KEYS[:-1]
    sums = torch.zeros(len(names) + 1, dtype=torch.float64, device=target_device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_indices = selected[start : start + batch_size]
            x, phase, y = _sample_batch(dataset, batch_indices, target_device)
            losses = pfnn_losses(
                model(x, phase),
                y,
                model=model.module if hasattr(model, "module") else model,
                normalization=dataset,
                kinematics=kinematics,
                loss_weights=loss_weights,
            )
            count = len(batch_indices)
            for offset, name in enumerate(names):
                sums[offset] += losses[name].to(torch.float64) * count
            sums[-1] += count
    if was_training:
        model.train()
    if distributed:
        import torch.distributed as dist

        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    count = int(sums[-1].item())
    if count < 1:
        raise ValueError("distributed validation produced no samples")
    means = {name: float(sums[index].item() / count) for index, name in enumerate(names)}
    weights = _loss_weights(loss_weights)
    score = sum(means[name] * weights[name] for name in names)
    return {**means, "one_step_score": score, "samples": count}


def finite_runtime_seed() -> dict[str, object]:
    direction = torch.zeros(12, 2, dtype=torch.float32)
    direction[:, 0] = 1.0
    semantic = torch.zeros(12, 2, dtype=torch.float32)
    semantic[:, 0] = 1.0
    normalized = np.zeros(INPUT_LAYOUT.size, dtype=np.float32)
    normalized[INPUT_LAYOUT["trajectory_direction"]] = direction.numpy().reshape(-1)
    normalized[INPUT_LAYOUT["semantic_intent"]] = semantic.numpy().reshape(-1)
    return {
        "phase": torch.tensor(0.0, dtype=torch.float32),
        "world_xy": torch.zeros(2, dtype=torch.float32),
        "world_yaw": torch.tensor(0.0, dtype=torch.float32),
        "root_height": torch.tensor(0.8, dtype=torch.float32),
        "root_tilt": torch.zeros(2, dtype=torch.float32),
        "joint_position": torch.zeros(29, dtype=torch.float32),
        "body_position": torch.zeros(30, 3, dtype=torch.float32),
        "body_velocity": torch.zeros(30, 3, dtype=torch.float32),
        "trajectory_position": torch.zeros(12, 2, dtype=torch.float32),
        "trajectory_direction": direction,
        "contact_label": torch.zeros(4, dtype=torch.float32),
        "semantic_intent": semantic,
        "terrain_height": torch.zeros(12, 3, dtype=torch.float32),
        "normalized_input": torch.as_tensor(normalized.copy()),
        "normalized_input_sha256": pfnn_input_sha256(normalized),
        "provenance": {
            "predecessor_clip_id": "synthetic",
            "predecessor_center_frame": 0,
            "predecessor_sequence_lane": "motion",
            "first_fitted_clip_id": "synthetic",
            "first_fitted_center_frame": 1,
            "first_fitted_sequence_lane": "motion",
            "speed": 0.0,
            "predecessor_row_sha256": "0" * 64,
            "first_fitted_row_sha256": "0" * 64,
            "fitted_subset_rows_sha256": "0" * 64,
        },
    }


def _validate_runtime_seed(
    seed: object, joint_limits: torch.Tensor, *, exact_tensors: bool = False
) -> dict[str, object]:
    template = finite_runtime_seed()
    if type(seed) is not dict or set(seed) != set(template):
        raise ValueError("runtime seed fields are invalid")
    output: dict[str, object] = {}
    for name, expected in template.items():
        value = seed[name]
        if isinstance(expected, torch.Tensor):
            if exact_tensors:
                if (
                    type(value) is not torch.Tensor
                    or value.device.type != "cpu"
                    or value.dtype != torch.float32
                    or value.layout != torch.strided
                    or value.shape != expected.shape
                    or not value.is_contiguous()
                    or value.requires_grad
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"runtime seed {name} is invalid")
                tensor = value.clone()
            else:
                tensor = torch.as_tensor(
                    value, dtype=torch.float32, device="cpu"
                ).detach().contiguous().clone()
            if tensor.shape != expected.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"runtime seed {name} is invalid")
            output[name] = tensor
        elif name == "provenance":
            if (
                type(value) is not dict
                or set(value) != {
                    "predecessor_clip_id", "predecessor_center_frame",
                    "predecessor_sequence_lane", "first_fitted_clip_id",
                    "first_fitted_center_frame", "first_fitted_sequence_lane", "speed",
                    "predecessor_row_sha256", "first_fitted_row_sha256",
                    "fitted_subset_rows_sha256",
                }
                or any(
                    type(value[name]) is not str or not value[name]
                    for name in ("predecessor_clip_id", "first_fitted_clip_id")
                )
                or any(
                    value[name] not in _SEQUENCE_LANES
                    for name in (
                        "predecessor_sequence_lane", "first_fitted_sequence_lane"
                    )
                )
                or any(
                    type(value[name]) is not int or value[name] < 0
                    for name in (
                        "predecessor_center_frame", "first_fitted_center_frame"
                    )
                )
                or (
                    type(value["speed"]) is not float
                    if exact_tensors
                    else type(value["speed"]) not in (int, float)
                )
                or not math.isfinite(float(value["speed"]))
                or any(
                    type(value[name]) is not str
                    or len(value[name]) != 64
                    or any(character not in "0123456789abcdef" for character in value[name])
                    for name in (
                        "predecessor_row_sha256", "first_fitted_row_sha256",
                        "fitted_subset_rows_sha256",
                    )
                )
            ):
                raise ValueError("runtime seed provenance is invalid")
            output[name] = {
                "predecessor_clip_id": value["predecessor_clip_id"],
                "predecessor_center_frame": value["predecessor_center_frame"],
                "predecessor_sequence_lane": value["predecessor_sequence_lane"],
                "first_fitted_clip_id": value["first_fitted_clip_id"],
                "first_fitted_center_frame": value["first_fitted_center_frame"],
                "first_fitted_sequence_lane": value["first_fitted_sequence_lane"],
                "speed": float(value["speed"]),
                "predecessor_row_sha256": value["predecessor_row_sha256"],
                "first_fitted_row_sha256": value["first_fitted_row_sha256"],
                "fitted_subset_rows_sha256": value["fitted_subset_rows_sha256"],
            }
        else:
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("runtime seed normalized input digest is invalid")
            output[name] = value
    joints = output["joint_position"]
    assert isinstance(joints, torch.Tensor)
    if torch.any(joints < joint_limits[:, 0]) or torch.any(joints > joint_limits[:, 1]):
        raise ValueError("runtime seed joint positions exceed canonical limits")
    contacts = output["contact_label"]
    assert isinstance(contacts, torch.Tensor)
    if torch.any((contacts != 0.0) & (contacts != 1.0)):
        raise ValueError("runtime seed contacts must be binary")
    semantic = output["semantic_intent"]
    assert isinstance(semantic, torch.Tensor)
    if not torch.all((semantic == 0.0) | (semantic == 1.0)) or not torch.all(
        semantic.sum(dim=1) == 1.0
    ):
        raise ValueError("runtime seed semantic intent must be one-hot")
    normalized_input = output["normalized_input"]
    normalized_input_sha256 = output["normalized_input_sha256"]
    assert isinstance(normalized_input, torch.Tensor)
    assert isinstance(normalized_input_sha256, str)
    if pfnn_input_sha256(normalized_input.numpy()) != normalized_input_sha256:
        raise ValueError("runtime seed normalized input digest mismatch")
    phase = output["phase"]
    assert isinstance(phase, torch.Tensor)
    if not 0.0 <= float(phase) < 2.0 * math.pi:
        raise ValueError("runtime seed phase must be in [0,2*pi)")
    return output


def _validated_sequence_sampler_state(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    expected_keys = {
        "seed",
        "sequence_count",
        "permutation_number",
        "permutation",
        "cursor",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("checkpoint sequence sampler state is invalid")
    seed = value["seed"]
    sequence_count = value["sequence_count"]
    permutation_number = value["permutation_number"]
    permutation = value["permutation"]
    cursor = value["cursor"]
    if (
        type(seed) is not int
        or type(sequence_count) is not int
        or sequence_count < 1
        or type(permutation_number) is not int
        or permutation_number < 0
        or type(permutation) is not list
        or len(permutation) != sequence_count
        or any(type(index) is not int for index in permutation)
        or type(cursor) is not int
        or cursor < 0
        or cursor >= sequence_count
    ):
        raise ValueError("checkpoint sequence sampler state is invalid")
    expected_permutation = list(range(sequence_count))
    generator = random.Random(seed + 1_000_003 * permutation_number)
    generator.shuffle(expected_permutation)
    if permutation != expected_permutation:
        raise ValueError("checkpoint sequence sampler state is invalid")
    return {
        "seed": seed,
        "sequence_count": sequence_count,
        "permutation_number": permutation_number,
        "permutation": list(permutation),
        "cursor": cursor,
    }


def _validated_sampler_state(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {"epoch", "global_offset", "sequence"}
        or type(value["epoch"]) is not int
        or value["epoch"] < 0
        or type(value["global_offset"]) is not int
        or value["global_offset"] < 0
    ):
        raise ValueError("checkpoint sampler state is invalid")
    return {
        "epoch": value["epoch"],
        "global_offset": value["global_offset"],
        "sequence": _validated_sequence_sampler_state(value["sequence"]),
    }


def _validated_fitted_subset(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {
        "split", "rows", "rows_sha256", "class_counts"
    }:
        raise ValueError("checkpoint fitted subset is invalid")
    rows = value["rows"]
    counts = value["class_counts"]
    if (
        value["split"] != "train"
        or type(rows) is not list
        or not rows
        or type(counts) is not dict
        or set(counts) != {"flat", "ascent", "descent", "transition"}
        or any(type(counts[name]) is not int or counts[name] < 0 for name in counts)
        or sum(counts.values()) != len(rows)
    ):
        raise ValueError("checkpoint fitted subset is invalid")
    checked_rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != {
                "clip_id", "center_frame", "split_identity", "sequence_lane",
                "terrain_class", "row_sha256",
            }
            or type(row["clip_id"]) is not str
            or not row["clip_id"]
            or type(row["center_frame"]) is not int
            or row["center_frame"] < 0
            or type(row["split_identity"]) is not str
            or not row["split_identity"]
            or row["sequence_lane"] not in _SEQUENCE_LANES
            or row["terrain_class"] not in counts
            or type(row["row_sha256"]) is not str
            or len(row["row_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["row_sha256"])
        ):
            raise ValueError("checkpoint fitted subset is invalid")
        key = (row["clip_id"], row["center_frame"], row["sequence_lane"])
        if key in seen:
            raise ValueError("checkpoint fitted subset is invalid")
        seen.add(key)
        checked_rows.append({
            "clip_id": key[0],
            "center_frame": key[1],
            "sequence_lane": key[2],
            "split_identity": row["split_identity"],
            "terrain_class": row["terrain_class"],
            "row_sha256": row["row_sha256"],
        })
    encoded = json.dumps(
        checked_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if value["rows_sha256"] != digest:
        raise ValueError("checkpoint fitted subset digest mismatch")
    return {
        "split": "train",
        "rows": checked_rows,
        "rows_sha256": digest,
        "class_counts": {name: counts[name] for name in (
            "flat", "ascent", "descent", "transition"
        )},
    }


def validate_runtime_seed_fitted_subset(
    runtime_seed: Mapping[str, object], fitted_subset: Mapping[str, object]
) -> None:
    """Bind the recurrent seed pair to two adjacent rows in one fitted receipt."""

    checked_subset = _validated_fitted_subset(dict(fitted_subset))
    if checked_subset is None:
        raise ValueError("runtime seed fitted subset membership is missing")
    provenance = runtime_seed.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime seed fitted subset membership is invalid")
    predecessor_key = (
        provenance.get("predecessor_clip_id"),
        provenance.get("predecessor_center_frame"),
        provenance.get("predecessor_sequence_lane"),
    )
    first_key = (
        provenance.get("first_fitted_clip_id"),
        provenance.get("first_fitted_center_frame"),
        provenance.get("first_fitted_sequence_lane"),
    )
    if (
        type(predecessor_key[0]) is not str
        or type(predecessor_key[1]) is not int
        or type(first_key[0]) is not str
        or type(first_key[1]) is not int
        or predecessor_key[2] not in _SEQUENCE_LANES
        or first_key[2] not in _SEQUENCE_LANES
    ):
        raise ValueError("runtime seed fitted subset membership is invalid")
    rows = {
        (row["clip_id"], row["center_frame"], row["sequence_lane"]): row
        for row in checked_subset["rows"]
    }
    if predecessor_key not in rows or first_key not in rows:
        raise ValueError("runtime seed fitted subset membership mismatch")
    if (
        predecessor_key[0] != first_key[0]
        or predecessor_key[1] + 1 != first_key[1]
        or predecessor_key[2] != first_key[2]
    ):
        raise ValueError("runtime seed fitted rows are not adjacent")
    if (
        provenance.get("predecessor_row_sha256")
        != rows[predecessor_key]["row_sha256"]
        or provenance.get("first_fitted_row_sha256")
        != rows[first_key]["row_sha256"]
        or provenance.get("fitted_subset_rows_sha256")
        != checked_subset["rows_sha256"]
    ):
        raise ValueError("runtime seed fitted subset row hash mismatch")


def validate_resume_fitted_subset(
    checkpoint: object, fitted_subset: Mapping[str, object] | None
) -> None:
    """Reject changed fitted rows before any resume state is restored."""

    current = _validated_fitted_subset(
        None if fitted_subset is None else dict(fitted_subset)
    )
    if getattr(checkpoint, "fitted_subset", None) != current:
        raise ValueError("resume fitted subset mismatch")


def selection_metadata(
    *,
    one_step_score: float,
    pipeline_overfit: bool,
    closed_loop_scorer: Callable[[], float] | None = None,
) -> dict[str, object]:
    score = float(one_step_score)
    if not math.isfinite(score) or score < 0.0:
        raise ValueError("one-step validation score must be finite and nonnegative")
    if pipeline_overfit:
        return {
            "selection_mode": "pipeline_overfit_one_step",
            "one_step_score": score,
            "closed_loop_score": None,
            "validation_score": score,
            "provisional": True,
        }
    if closed_loop_scorer is None:
        raise RuntimeError(
            "normal checkpoint selection requires the Task 7 closed-loop scorer"
        )
    closed_loop_score = float(closed_loop_scorer())
    if not math.isfinite(closed_loop_score) or closed_loop_score < 0.0:
        raise ValueError("closed-loop failure penalty must be finite and nonnegative")
    return {
        "selection_mode": "one_step_plus_closed_loop",
        "one_step_score": score,
        "closed_loop_score": closed_loop_score,
        "validation_score": score + closed_loop_score,
        "provisional": False,
    }


def _validated_selection(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    keys = {
        "selection_mode", "one_step_score", "closed_loop_score",
        "validation_score", "provisional",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("checkpoint selection is invalid")
    one_step = value["one_step_score"]
    validation = value["validation_score"]
    if (
        type(one_step) is not float
        or not math.isfinite(one_step)
        or one_step < 0.0
        or type(validation) is not float
        or not math.isfinite(validation)
        or type(value["provisional"]) is not bool
    ):
        raise ValueError("checkpoint selection is invalid")
    if value["selection_mode"] == "pipeline_overfit_one_step":
        if (
            value["closed_loop_score"] is not None
            or value["provisional"] is not True
            or validation != one_step
        ):
            raise ValueError("checkpoint selection is invalid")
    elif value["selection_mode"] == "one_step_plus_closed_loop":
        closed_loop = value["closed_loop_score"]
        if (
            type(closed_loop) is not float
            or not math.isfinite(closed_loop)
            or closed_loop < 0.0
            or value["provisional"] is not False
            or validation != one_step + closed_loop
        ):
            raise ValueError("checkpoint selection is invalid")
    else:
        raise ValueError("checkpoint selection is invalid")
    return dict(value)


def _code_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _plain_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _plain_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_cpu(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"checkpoint payload contains unsafe type {type(value).__name__}")


def _finite_tree(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return value is None or type(value) in (str, int, bool)


_CHECKPOINT_KEYS = {
    "schema", "input_size", "output_size", "input_layout", "output_layout",
    "fps", "joint_order", "body_order",
    "contact_order", "trajectory_times_s", "normalization_contract", "normalization",
    "model_config", "model_state", "optimizer_state", "code_commit", "dataset_digest",
    "train_identities", "validation_identities", "step", "epoch", "seed",
    "loss_weights", "kinematic_signature_sha256", "joint_limits",
    "phase_advance_q99", "runtime_seed", "selection", "sampler_state",
    "fitted_subset",
}


@dataclass(frozen=True)
class LoadedCheckpoint:
    schema: str
    step: int
    epoch: int
    seed: int
    dataset_digest: str
    kinematic_signature_sha256: str
    runtime_seed: dict[str, object]
    normalization: dict[str, torch.Tensor]
    loss_weights: dict[str, float]
    joint_limits: torch.Tensor
    phase_advance_q99: float
    train_identities: tuple[str, ...]
    validation_identities: tuple[str, ...]
    selection: dict[str, object] | None
    model_config: dict[str, object]
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, object]
    code_commit: str
    sampler_epoch: int
    sampler_global_offset: int
    sequence_sampler_state: dict[str, object] | None
    fitted_subset: dict[str, object] | None

    def build_model(self) -> PhaseFunctionedNetwork:
        model = PhaseFunctionedNetwork(
            hidden_size=int(self.model_config["hidden_size"]),
            dropout_probability=float(self.model_config["dropout_probability"]),
        )
        model.load_state_dict(self.model_state, strict=True)
        return model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    normalization: object,
    *,
    dataset_digest: str,
    kinematic_signature_sha256: str,
    runtime_seed: Mapping[str, object],
    step: int,
    epoch: int = 0,
    seed: int = 0,
    loss_weights: Mapping[str, float] | None = None,
    joint_limits: object | None = None,
    phase_advance_q99: float = 0.0,
    train_identities: Sequence[str] = (),
    validation_identities: Sequence[str] = (),
    selection: Mapping[str, object] | None = None,
    code_commit: str | None = None,
    sampler_epoch: int = 0,
    sampler_global_offset: int = 0,
    sequence_sampler_state: Mapping[str, object] | None = None,
    fitted_subset: Mapping[str, object] | None = None,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    if not isinstance(unwrapped, PhaseFunctionedNetwork):
        raise TypeError("checkpoint model must be a PhaseFunctionedNetwork")
    if type(step) is not int or step < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("checkpoint step and epoch must be nonnegative integers")
    if type(seed) is not int or not dataset_digest or not kinematic_signature_sha256:
        raise ValueError("checkpoint provenance is invalid")
    normal = _normalization_tensors(
        normalization, device=torch.device("cpu"), dtype=torch.float32
    )
    if joint_limits is None:
        limits = torch.tensor([[-math.pi, math.pi]] * 29, dtype=torch.float64)
    else:
        limits = torch.as_tensor(
            joint_limits, dtype=torch.float64, device="cpu"
        ).contiguous().clone()
    if (
        limits.shape != (29, 2)
        or not torch.isfinite(limits).all()
        or torch.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("canonical joint limits are invalid")
    q99 = float(phase_advance_q99)
    if not math.isfinite(q99) or q99 < 0.0:
        raise ValueError("phase_advance_q99 must be finite and nonnegative")
    checked_runtime_seed = _validate_runtime_seed(dict(runtime_seed), limits)
    checked_fitted_subset = _validated_fitted_subset(
        None if fitted_subset is None else dict(fitted_subset)
    )
    if checked_fitted_subset is not None:
        validate_runtime_seed_fitted_subset(
            checked_runtime_seed, checked_fitted_subset
        )
    weights = _loss_weights(loss_weights)
    sampler_state = _validated_sampler_state(
        {
            "epoch": sampler_epoch,
            "global_offset": sampler_global_offset,
            "sequence": (
                None
                if sequence_sampler_state is None
                else dict(sequence_sampler_state)
            ),
        }
    )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "input_size": INPUT_LAYOUT.size,
        "output_size": OUTPUT_LAYOUT.size,
        "input_layout": [list(field) for field in INPUT_LAYOUT.fields],
        "output_layout": [list(field) for field in OUTPUT_LAYOUT.fields],
        "fps": 30.0,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "body_order": list(ISAACLAB_BODY_NAMES),
        "contact_order": list(CONTACT_ORDER),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "normalization_contract": dict(_NORMALIZATION_CONTRACT),
        "normalization": normal,
        "model_config": {
            "hidden_size": int(unwrapped.W0.shape[1]),
            "dropout_probability": float(unwrapped.dropout.p),
        },
        "model_state": dict(unwrapped.state_dict()),
        "optimizer_state": dict(optimizer.state_dict()),
        "code_commit": _code_commit() if code_commit is None else str(code_commit),
        "dataset_digest": dataset_digest,
        "train_identities": list(train_identities),
        "validation_identities": list(validation_identities),
        "step": step,
        "epoch": epoch,
        "seed": seed,
        "loss_weights": weights,
        "kinematic_signature_sha256": kinematic_signature_sha256,
        "joint_limits": limits,
        "phase_advance_q99": q99,
        "runtime_seed": checked_runtime_seed,
        "selection": _validated_selection(
            None if selection is None else dict(selection)
        ),
        "sampler_state": sampler_state,
        "fitted_subset": checked_fitted_subset,
    }
    plain = _plain_cpu(payload)
    if not isinstance(plain, dict) or not _finite_tree(plain):
        raise ValueError("checkpoint payload contains nonfinite values")
    _validate_optimizer_state(plain["optimizer_state"], unwrapped)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(plain, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_string_list(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError(f"checkpoint {name} is invalid")
    return tuple(value)


def _validate_optimizer_state(
    value: object, model: PhaseFunctionedNetwork
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {"state", "param_groups"}
        or type(value["state"]) is not dict
        or type(value["param_groups"]) is not list
        or len(value["param_groups"]) != 1
        or type(value["param_groups"][0]) is not dict
    ):
        raise ValueError("checkpoint optimizer state is invalid")
    group = value["param_groups"][0]
    if set(group) != _ADAM_PARAM_GROUP_KEYS:
        raise ValueError("checkpoint optimizer state is invalid")
    params = group["params"]
    expected_params = list(range(len(tuple(model.parameters()))))
    if (
        type(params) is not list
        or any(type(parameter_id) is not int for parameter_id in params)
        or len(set(params)) != len(params)
        or params != expected_params
    ):
        raise ValueError("checkpoint optimizer state is invalid")
    for name, lower_inclusive in (
        ("lr", False),
        ("eps", False),
        ("weight_decay", True),
    ):
        scalar = group[name]
        if (
            type(scalar) is not float
            or not math.isfinite(scalar)
            or (scalar < 0.0 if lower_inclusive else scalar <= 0.0)
        ):
            raise ValueError("checkpoint optimizer state is invalid")
    betas = group["betas"]
    if (
        type(betas) not in (tuple, list)
        or len(betas) != 2
        or any(
            type(beta) is not float
            or not math.isfinite(beta)
            or not 0.0 <= beta < 1.0
            for beta in betas
        )
    ):
        raise ValueError("checkpoint optimizer state is invalid")
    if any(type(group[name]) is not bool for name in _ADAM_BOOL_FLAGS):
        raise ValueError("checkpoint optimizer state is invalid")
    if any(
        group[name] is not None and type(group[name]) is not bool
        for name in _ADAM_OPTIONAL_BOOL_FLAGS
    ):
        raise ValueError("checkpoint optimizer state is invalid")
    state = value["state"]
    if any(
        type(key) is not int or key not in range(len(expected_params))
        for key in state
    ):
        raise ValueError("checkpoint optimizer state is invalid")
    parameters = list(model.parameters())
    expected_fields = {"step", "exp_avg", "exp_avg_sq"}
    if group.get("amsgrad") is True:
        expected_fields.add("max_exp_avg_sq")
    for parameter_index, raw in state.items():
        if type(raw) is not dict or set(raw) != expected_fields:
            raise ValueError("checkpoint optimizer state is invalid")
        step = raw["step"]
        if (
            type(step) is not torch.Tensor
            or step.device.type != "cpu"
            or step.shape != ()
            or step.dtype != torch.float32
            or step.layout != torch.strided
            or not step.is_contiguous()
            or step.requires_grad
            or not torch.isfinite(step)
            or float(step) < 0.0
        ):
            raise ValueError("checkpoint optimizer state is invalid")
        for name in expected_fields - {"step"}:
            tensor = raw[name]
            if (
                type(tensor) is not torch.Tensor
                or tensor.device.type != "cpu"
                or tensor.shape != parameters[parameter_index].shape
                or tensor.dtype != parameters[parameter_index].dtype
                or tensor.layout != torch.strided
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("checkpoint optimizer state is invalid")
    probe = torch.optim.Adam(model.parameters(), weight_decay=0.0)
    try:
        probe.load_state_dict(value)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer state is invalid") from error
    return value


def load_checkpoint(
    path: str | Path,
    *,
    expected_dataset_digest: str,
    expected_kinematic_signature_sha256: str,
) -> LoadedCheckpoint:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError) as error:
        raise ValueError("checkpoint cannot be loaded safely") from error
    if type(payload) is not dict or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("checkpoint fields or schema are invalid")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint fields or schema are invalid")
    immutable = {
        "input_size": INPUT_LAYOUT.size,
        "output_size": OUTPUT_LAYOUT.size,
        "input_layout": [list(field) for field in INPUT_LAYOUT.fields],
        "output_layout": [list(field) for field in OUTPUT_LAYOUT.fields],
        "fps": 30.0,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "body_order": list(ISAACLAB_BODY_NAMES),
        "contact_order": list(CONTACT_ORDER),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "normalization_contract": _NORMALIZATION_CONTRACT,
    }
    if any(payload.get(name) != value for name, value in immutable.items()):
        raise ValueError("checkpoint immutable PFNN contract mismatch")
    if payload.get("dataset_digest") != expected_dataset_digest:
        raise ValueError("checkpoint dataset digest mismatch")
    if payload.get("kinematic_signature_sha256") != expected_kinematic_signature_sha256:
        raise ValueError("checkpoint kinematic signature mismatch")
    if not _finite_tree(payload):
        raise ValueError("checkpoint contains nonfinite or unsafe values")
    for name in ("step", "epoch"):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError(f"checkpoint {name} is invalid")
    if type(payload["seed"]) is not int:
        raise ValueError("checkpoint seed is invalid")
    if type(payload["code_commit"]) is not str or not payload["code_commit"]:
        raise ValueError("checkpoint code commit is invalid")
    config = payload["model_config"]
    if (
        type(config) is not dict
        or set(config) != {"hidden_size", "dropout_probability"}
        or type(config["hidden_size"]) is not int
        or config["hidden_size"] < 1
        or type(config["dropout_probability"]) is not float
        or not 0.0 <= config["dropout_probability"] < 1.0
    ):
        raise ValueError("checkpoint model config is invalid")
    probe = PhaseFunctionedNetwork(
        hidden_size=config["hidden_size"],
        dropout_probability=config["dropout_probability"],
    )
    model_state = payload["model_state"]
    if type(model_state) is not dict or set(model_state) != set(probe.state_dict()):
        raise ValueError("checkpoint model state is invalid")
    for name, expected in probe.state_dict().items():
        value = model_state[name]
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ValueError("checkpoint model state is invalid")
    try:
        probe.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint model state is invalid") from error
    normal = _checkpoint_normalization_tensors(payload["normalization"])
    limits = payload["joint_limits"]
    if (
        type(limits) is not torch.Tensor
        or limits.device.type != "cpu"
        or limits.dtype != torch.float64
        or limits.layout != torch.strided
        or limits.shape != (29, 2)
        or not limits.is_contiguous()
        or limits.requires_grad
        or not torch.isfinite(limits).all()
        or torch.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("checkpoint canonical joint limits are invalid")
    runtime_seed = _validate_runtime_seed(
        payload["runtime_seed"], limits, exact_tensors=True
    )
    weights = _loss_weights(payload["loss_weights"])
    q99 = payload["phase_advance_q99"]
    if type(q99) is not float or q99 < 0.0:
        raise ValueError("checkpoint phase_advance_q99 is invalid")
    optimizer_state = _validate_optimizer_state(payload["optimizer_state"], probe)
    selection = _validated_selection(payload["selection"])
    sampler_state = _validated_sampler_state(payload["sampler_state"])
    fitted_subset = _validated_fitted_subset(payload["fitted_subset"])
    if fitted_subset is not None:
        validate_runtime_seed_fitted_subset(runtime_seed, fitted_subset)
    return LoadedCheckpoint(
        schema=payload["schema"],
        step=payload["step"],
        epoch=payload["epoch"],
        seed=payload["seed"],
        dataset_digest=payload["dataset_digest"],
        kinematic_signature_sha256=payload["kinematic_signature_sha256"],
        runtime_seed=runtime_seed,
        normalization={name: value.clone() for name, value in normal.items()},
        loss_weights=weights,
        joint_limits=limits.clone(),
        phase_advance_q99=q99,
        train_identities=_require_string_list(payload["train_identities"], "train identities"),
        validation_identities=_require_string_list(payload["validation_identities"], "validation identities"),
        selection=None if selection is None else dict(selection),
        model_config=dict(config),
        model_state={name: value.clone() for name, value in model_state.items()},
        optimizer_state=dict(optimizer_state),
        code_commit=payload["code_commit"],
        sampler_epoch=sampler_state["epoch"],
        sampler_global_offset=sampler_state["global_offset"],
        sequence_sampler_state=(
            None
            if sampler_state["sequence"] is None
            else dict(sampler_state["sequence"])
        ),
        fitted_subset=fitted_subset,
    )


def restore_training_state(
    checkpoint: LoadedCheckpoint,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    normalization: object,
    train_identities: Sequence[str],
    validation_identities: Sequence[str],
) -> tuple[int, int]:
    """Validate and restore an Adam/PFNN resume state on every DDP rank."""

    unwrapped = model.module if hasattr(model, "module") else model
    if not isinstance(unwrapped, PhaseFunctionedNetwork):
        raise TypeError("resume model must be a PhaseFunctionedNetwork")
    expected_config = {
        "hidden_size": int(unwrapped.W0.shape[1]),
        "dropout_probability": float(unwrapped.dropout.p),
    }
    if checkpoint.model_config != expected_config:
        raise ValueError("resume model configuration mismatch")
    if checkpoint.train_identities != tuple(train_identities) or (
        checkpoint.validation_identities != tuple(validation_identities)
    ):
        raise ValueError("resume split identities mismatch")
    normal = _normalization_tensors(
        normalization, device=torch.device("cpu"), dtype=torch.float32
    )
    if any(
        not torch.equal(checkpoint.normalization[name], value)
        for name, value in normal.items()
    ):
        raise ValueError("resume normalization mismatch")
    unwrapped.load_state_dict(checkpoint.model_state, strict=True)
    try:
        optimizer.load_state_dict(checkpoint.optimizer_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("resume optimizer state mismatch") from error
    return checkpoint.step, checkpoint.epoch


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEFAULT_LOSS_WEIGHTS",
    "LOSS_WEIGHT_KEYS",
    "LoadedCheckpoint",
    "RolloutResult",
    "autoregressive_unroll",
    "choose_runtime_seed",
    "fitted_row_sha256",
    "finite_runtime_seed",
    "load_checkpoint",
    "one_step_metrics",
    "pfnn_losses",
    "restore_training_state",
    "save_checkpoint",
    "selection_metadata",
    "training_phase_advance_q99",
    "validate_resume_fitted_subset",
    "validate_runtime_seed_fitted_subset",
]
