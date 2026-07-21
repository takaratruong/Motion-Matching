#!/usr/bin/env python3
"""Deterministically train coupled native-G1 walk/pickup diffusion experts.

Defaults are intentionally a single practical A/B/C pass on one L40S.  They
perform real optimisation and evaluate a frozen validation subset; increase the
three epoch knobs for longer training without ever reading test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from resources.g1_interaction_builder.overlap_diffusion import (
    CONTACT_SLICE,
    DIFFUSION_STEPS,
    FRAME_DIM,
    LOSS_NAMES,
    OVERLAP_FRAMES,
    ROTATION6D_SLICE,
    STATIC_CONDITION_DIM,
    TEMPORAL_CONDITION_DIM,
    TIMELINE_FRAMES,
    WINDOW_FRAMES,
    CANONICAL_LOCAL_OFFSETS_SHA256,
    CoupledCondition,
    FixedFrames,
    MotionWindowDenoiser,
    TaskGuidance,
    _alpha_bars,
    canonical_g1_fk,
    named_training_losses,
    sample_coupled,
)
from resources.g1_interaction_builder.schema import G1_SKELETON


LOSS_WEIGHTS = {
    "epsilon": 1.0,
    "pose_6d": 0.20,
    "fk_hand": 0.20,
    "fk_feet": 0.10,
    "velocity": 0.05,
    "acceleration": 0.02,
    "foot_contact": 0.05,
    "foot_sliding": 0.05,
    "grasp_position": 0.50,
    "grasp_orientation": 0.20,
    "pre_contact_separation": 0.10,
    "attachment": 0.50,
    "overlap_agreement": 0.25,
}

GUIDANCE_STRENGTH = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _atomic_best_pair(value: object, best_path: Path, requested_path: Path) -> None:
    """Publish best.pt and the requested checkpoint as one rollback-safe pair."""
    if best_path == requested_path:
        raise ValueError("best and requested checkpoint paths must differ")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    backups: list[Path | None] = [None, None]
    paths = (best_path, requested_path)
    if best_path.exists() != requested_path.exists():
        raise RuntimeError("best checkpoint publication requires a complete prior pair")
    try:
        for path in paths:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
                staged_path = Path(stream.name)
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append(staged_path)
        if best_path.exists():
            try:
                for index, path in enumerate(paths):
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                        backup = Path(stream.name)
                    backup.unlink()
                    os.replace(path, backup)
                    backups[index] = backup
            except Exception:
                for path, backup in zip(paths, backups):
                    if backup is not None and backup.exists():
                        os.replace(backup, path)
                raise
        try:
            os.replace(staged[0], best_path)
            os.replace(staged[1], requested_path)
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            for path, backup in zip(paths, backups):
                if backup is not None and backup.exists():
                    os.replace(backup, path)
            raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
        for backup in backups:
            if backup is not None:
                backup.unlink(missing_ok=True)


def _atomic_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _atomic_npz(value: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        np.savez_compressed(stream, **value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _require_array(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"dataset {label} must be finite with shape {shape}")
    return result.astype(np.float32, copy=True)


def _read_dataset(path: Path) -> dict[str, object]:
    """Read only train/validation fields; test partitions are intentionally absent."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "normalization_mean", "normalization_scale", "skeleton_parents", "skeleton_names",
            "skeleton_signature", "canonical_local_offsets",
        )
        for key in required:
            if key not in archive:
                raise ValueError(f"dataset is missing {key}")
        parents = np.asarray(archive["skeleton_parents"], np.int32)
        names = tuple(str(item) for item in np.asarray(archive["skeleton_names"]).tolist())
        signature = str(np.asarray(archive["skeleton_signature"]).item())
        if not np.array_equal(parents, G1_SKELETON.parents):
            raise ValueError("dataset skeleton parents do not match canonical G1")
        if names != G1_SKELETON.names or signature != G1_SKELETON.signature():
            raise ValueError("dataset skeleton identity does not match canonical G1")
        offsets = _require_array(archive["canonical_local_offsets"], (31, 3), "canonical_local_offsets")
        if not np.array_equal(offsets[:2], np.zeros((2, 3), np.float32)):
            raise ValueError("dataset root/Hips offsets must remain dynamic channels")
        if hashlib.sha256(np.ascontiguousarray(offsets.astype("<f4", copy=False)).tobytes()).hexdigest() != CANONICAL_LOCAL_OFFSETS_SHA256:
            raise ValueError("dataset canonical local offsets SHA256 does not match the frozen contract")
        mean = _require_array(archive["normalization_mean"], (FRAME_DIM,), "normalization_mean")
        scale = _require_array(archive["normalization_scale"], (FRAME_DIM,), "normalization_scale")
        if np.any(scale <= 0.0):
            raise ValueError("dataset normalization_scale must be positive")

        data: dict[str, object] = {
            "parents": parents.copy(), "names": names, "signature": signature,
            "offsets": offsets, "mean": mean, "scale": scale,
        }
        fields = ("walk_windows", "pickup_windows", "static_conditions", "walk_temporal", "pickup_temporal")
        for partition in ("train", "validation"):
            for field in fields:
                key = f"{partition}_{field}"
                if key not in archive:
                    raise ValueError(f"dataset is missing {key}")
                value = np.asarray(archive[key])
                if field.endswith("windows"):
                    expected = (len(value), WINDOW_FRAMES, FRAME_DIM)
                elif field == "static_conditions":
                    expected = (len(value), STATIC_CONDITION_DIM)
                else:
                    expected = (len(value), WINDOW_FRAMES, TEMPORAL_CONDITION_DIM)
                data[key] = _require_array(value, expected, key)
            walking_fields = ("walking_windows", "walking_static_conditions", "walking_temporal")
            for field in walking_fields:
                key = f"{partition}_{field}"
                if key not in archive:
                    raise ValueError(f"dataset is missing {key}")
                value = np.asarray(archive[key])
                if field == "walking_windows":
                    expected = (len(value), WINDOW_FRAMES, FRAME_DIM)
                elif field == "walking_static_conditions":
                    expected = (len(value), STATIC_CONDITION_DIM)
                else:
                    expected = (len(value), WINDOW_FRAMES, TEMPORAL_CONDITION_DIM)
                data[key] = _require_array(value, expected, key)
    if len(data["train_walk_windows"]) == 0 or len(data["train_pickup_windows"]) == 0:
        raise ValueError("dataset training partitions must be nonempty")
    if len(data["validation_pickup_windows"]) == 0:
        raise ValueError("dataset validation interaction partition must be nonempty")
    return data


def _to_device(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device=device, dtype=torch.float32)


def _batch_order(count: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    if count <= 0:
        raise ValueError("training source must be nonempty")
    order = torch.randperm(count, generator=generator)
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


def _object_terms(static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grasp_position = static[:, 2:5]
    grasp_orientation = static[:, 5:11]
    object_position = torch.zeros_like(grasp_position)
    object_clearance = torch.where(
        static[:, 24] >= 0.5,
        static[:, 13:16].amax(dim=1) * 0.5,
        torch.zeros_like(static[:, 0]),
    )
    return grasp_position, grasp_orientation, object_position, object_clearance


def _denoise(
    model: MotionWindowDenoiser,
    clean: torch.Tensor,
    static: torch.Tensor,
    temporal: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = alpha_bars[timesteps].to(dtype=clean.dtype).view(-1, 1, 1)
    noisy = torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
    epsilon = model(noisy, timesteps, static, temporal)
    return epsilon, (noisy - torch.sqrt(1.0 - alpha) * epsilon) / torch.sqrt(alpha)


def _weighted(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if set(losses) != set(LOSS_NAMES):
        raise ValueError("named loss result does not match the reviewed contract")
    return sum(LOSS_WEIGHTS[name] * losses[name] for name in LOSS_NAMES)


def _raw_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    epsilon: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    overlap_walk: torch.Tensor,
    overlap_pickup: torch.Tensor,
    parents: np.ndarray,
    offsets: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    grasp_position, grasp_orientation, object_position, object_clearance = _object_terms(static)
    return named_training_losses(
        predicted, target, epsilon, noise,
        fk=lambda frames: canonical_g1_fk(frames, parents=parents, local_offsets=offsets),
        grasp_position=grasp_position, grasp_orientation=grasp_orientation,
        object_position=object_position, object_clearance=object_clearance,
        overlap_walk=overlap_walk, overlap_pickup=overlap_pickup,
    )


def _matrix_angle_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_matrix = torch.stack((left[..., :3], left[..., 3:], torch.linalg.cross(left[..., :3], left[..., 3:], dim=-1)), dim=-1)
    right_matrix = torch.stack((right[..., :3], right[..., 3:], torch.linalg.cross(right[..., :3], right[..., 3:], dim=-1)), dim=-1)
    trace = torch.diagonal(torch.matmul(left_matrix.transpose(-1, -2), right_matrix), dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)))


def _target_contact_frame(target_global: torch.Tensor) -> int:
    """Locate the first target active-hand Contact in the global 80-frame row."""
    if tuple(target_global.shape) != (TIMELINE_FRAMES, FRAME_DIM):
        raise ValueError("target global motion must have shape (80, 195)")
    contacts = torch.nonzero(target_global[:, CONTACT_SLICE.stop - 1] >= 0.5, as_tuple=False).flatten()
    if len(contacts) == 0:
        raise ValueError("validation target has no active-hand Contact")
    return int(contacts[0].item())


def _fixed_target_zero(target_global: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> FixedFrames:
    values = (target_global - mean.view(1, -1)) / scale.view(1, -1)
    mask = torch.zeros(TIMELINE_FRAMES, dtype=torch.bool, device=target_global.device)
    mask[0] = True
    return FixedFrames(values=values, mask=mask)


def _sampling_guidance(
    target_global: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    offsets: torch.Tensor,
    parents: np.ndarray,
) -> TaskGuidance:
    """Guide the clean global motion to the conditioned grasp and root route."""
    pickup_contact = _target_contact_frame(target_global) - 30
    if pickup_contact < 0:
        raise ValueError("guided Contact must occur in the pickup global range")

    def objective(
        clean: torch.Tensor, condition: CoupledCondition, timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        normal_mean = mean.detach().clone()
        normal_scale = scale.detach().clone()
        normal_offsets = offsets.detach().clone()
        static = condition.static.detach().clone()
        temporal = condition.temporal.detach().clone()
        geometry_clean = clean.clamp(-20.0, 20.0)
        raw = geometry_clean * normal_scale.view(1, 1, -1) + normal_mean.view(1, 1, -1)
        fk = canonical_g1_fk(
            raw[:, 30:80], parents=parents, local_offsets=normal_offsets,
        )
        position = ((fk["hand"][:, pickup_contact] - static[2:5]) / 0.04).square().mean(-1)
        orientation = (
            (fk["hand_orientation"][:, pickup_contact] - static[5:11]) / 0.15
        ).square().mean(-1)
        route = (
            (raw[:, :, (0, 2)] - temporal[None, :, :2]) / 0.20
        ).square().mean((1, 2))
        bounded_pose = torch.nn.functional.smooth_l1_loss(
            clean, clean.clamp(-3.0, 3.0), reduction="none",
        ).mean((1, 2))
        return position + 4.0 * orientation + 0.25 * route + 0.10 * bounded_pose

    return TaskGuidance(objective=objective, strength=GUIDANCE_STRENGTH)


def _passes_no_stop(frames: torch.Tensor, target_contact_frame: int) -> bool:
    """Apply the exact conditional global [30,55) pre-Contact dwell rule."""
    if tuple(frames.shape) != (TIMELINE_FRAMES, FRAME_DIM):
        raise ValueError("no-stop frames must have shape (80, 195)")
    if not isinstance(target_contact_frame, int) or not 0 <= target_contact_frame < TIMELINE_FRAMES:
        raise ValueError("target Contact frame must be in [0, 80)")
    planar_speed = torch.linalg.vector_norm(
        (frames[1:, (0, 2)] - frames[:-1, (0, 2)]) * 25.0, dim=-1,
    )
    entry_speed = float(planar_speed[29].item())
    if entry_speed <= 0.10:
        return True
    run = 0
    for frame in range(30, min(55, target_contact_frame)):
        if float(planar_speed[frame - 1].item()) < 0.03:
            run += 1
            if run >= 3:
                return False
        else:
            run = 0
    return True


def _contact_metrics(
    generated_raw: torch.Tensor, target_global: torch.Tensor, static: torch.Tensor, offsets: torch.Tensor | np.ndarray,
) -> dict[str, object]:
    """Score global candidates at the target's actual active-hand Contact frame."""
    if generated_raw.ndim != 3 or tuple(generated_raw.shape[1:]) != (TIMELINE_FRAMES, FRAME_DIM):
        raise ValueError("generated validation motion must have shape (B, 80, 195)")
    contact = _target_contact_frame(target_global)
    if contact < 30:
        raise ValueError("validation target Contact must occur in the pickup global range")
    fk = canonical_g1_fk(generated_raw[:, 30:80], parents=G1_SKELETON.parents, local_offsets=offsets)
    pickup_contact = contact - 30
    position = torch.linalg.vector_norm(fk["hand"][:, pickup_contact] - static[2:5], dim=-1)
    orientation = _matrix_angle_degrees(fk["hand_orientation"][:, pickup_contact], static[5:11])
    attach = (position <= 0.04) & (orientation <= 15.0)
    score = position / 0.04 + orientation / 15.0
    return {
        "target_contact_frame": contact,
        "position_error_m": position,
        "orientation_error_degrees": orientation,
        "attach": attach,
        "selected": int(torch.argmin(score).item()),
    }


@torch.inference_mode()
def _quality_summary(
    walk_model: MotionWindowDenoiser,
    pickup_model: MotionWindowDenoiser,
    data: Mapping[str, object],
    offsets: torch.Tensor,
    *,
    seed: int,
    subset: int,
) -> dict[str, dict[str, float]]:
    """Measure real deterministic validation samples at both required DDIM counts."""
    static = np.asarray(data["validation_static_conditions"], np.float32)
    temporal = np.asarray(data["validation_pickup_temporal"], np.float32)
    if subset <= 0:
        raise ValueError("validation_subset must be positive")
    order = np.random.default_rng(seed + 701).permutation(len(static))[:min(subset, len(static))]
    device = next(walk_model.parameters()).device
    mean = _to_device(np.asarray(data["mean"]), device)
    scale = _to_device(np.asarray(data["scale"]), device)
    parents = np.asarray(data["parents"], np.int32)
    summaries: dict[str, dict[str, float]] = {}
    for step_count in (20, 50):
        attach_rows: list[float] = []
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        no_stop_rows: list[float] = []
        for row in order:
            target_global = _to_device(np.concatenate((
                np.asarray(data["validation_walk_windows"], np.float32)[row, :30],
                np.asarray(data["validation_pickup_windows"], np.float32)[row],
            ), axis=0), device)
            condition = CoupledCondition(
                static=_to_device(static[row], device),
                temporal=_to_device(np.concatenate((
                    np.asarray(data["validation_walk_temporal"], np.float32)[row, :30], temporal[row],
                ), axis=0), device),
            )
            generated = sample_coupled(
                walk_model, pickup_model, condition, seed=seed + int(row) * 101,
                steps=step_count, fixed_frames=_fixed_target_zero(target_global, mean, scale),
                task_guidance=_sampling_guidance(
                    target_global, mean, scale, offsets, parents,
                ),
            )
            raw = generated * scale.view(1, 1, -1) + mean.view(1, 1, -1)
            metrics = _contact_metrics(raw, target_global, condition.static, offsets)
            selected = int(metrics["selected"])
            attach_rows.append(float(torch.any(metrics["attach"]).item()))
            position_errors.append(float(metrics["position_error_m"][selected].item()))
            orientation_errors.append(float(metrics["orientation_error_degrees"][selected].item()))
            no_stop_rows.append(float(_passes_no_stop(raw[selected], int(metrics["target_contact_frame"]))))
        summary = {
            "attach_proxy_at_8": float(np.mean(attach_rows)),
            "median_grasp_position_m": float(np.median(position_errors)),
            "median_grasp_orientation_degrees": float(np.median(orientation_errors)),
            "no_stop_proxy": float(np.mean(no_stop_rows)),
            "rows": float(len(order)),
        }
        if not all(math.isfinite(value) for value in summary.values()):
            raise ValueError("validation quality evaluation produced a non-finite metric")
        summaries[str(step_count)] = summary
    return summaries


@torch.inference_mode()
def _emit_preview(
    payload: Mapping[str, object], data: Mapping[str, object], offsets: torch.Tensor,
    output: Path, *, seed: int, validation_subset: int, device: torch.device,
) -> Path:
    """Atomically emit one evidence-rich preview from the actual selected C weights."""
    schema = payload["model_schema"]
    walk = MotionWindowDenoiser(width=int(schema["width"]), blocks=int(schema["blocks"]), heads=int(schema["heads"])).to(device)
    pickup = MotionWindowDenoiser(width=int(schema["width"]), blocks=int(schema["blocks"]), heads=int(schema["heads"])).to(device)
    walk.load_state_dict(payload["walk_model"], strict=True)
    pickup.load_state_dict(payload["pickup_model"], strict=True)
    walk.eval()
    pickup.eval()
    static_rows = np.asarray(data["validation_static_conditions"], np.float32)
    order = np.random.default_rng(seed + 701).permutation(len(static_rows))[:min(validation_subset, len(static_rows))]
    row = int(order[0])
    mean = _to_device(np.asarray(data["mean"]), device)
    scale = _to_device(np.asarray(data["scale"]), device)
    target_global = _to_device(np.concatenate((
        np.asarray(data["validation_walk_windows"], np.float32)[row, :30],
        np.asarray(data["validation_pickup_windows"], np.float32)[row],
    ), axis=0), device)
    condition = CoupledCondition(
        static=_to_device(static_rows[row], device),
        temporal=_to_device(np.concatenate((
            np.asarray(data["validation_walk_temporal"], np.float32)[row, :30],
            np.asarray(data["validation_pickup_temporal"], np.float32)[row],
        ), axis=0), device),
    )
    steps = int(payload["selected_sampler_steps"])
    sample_seed = seed + row * 101
    fixed = _fixed_target_zero(target_global, mean, scale)
    generated = sample_coupled(
        walk, pickup, condition, seed=sample_seed, steps=steps, fixed_frames=fixed,
        task_guidance=_sampling_guidance(
            target_global, mean, scale, offsets, np.asarray(data["parents"], np.int32),
        ),
    )
    if not torch.equal(generated[:, 0], fixed.values[0].expand_as(generated[:, 0])):
        raise AssertionError("preview sampler did not preserve exact fixed global frame zero")
    raw = generated * scale.view(1, 1, -1) + mean.view(1, 1, -1)
    raw[:, 0] = target_global[0]
    metrics = _contact_metrics(raw, target_global, condition.static, offsets)
    selected = int(metrics["selected"])
    preview_path = Path(output).with_suffix(".preview.npz")
    _atomic_npz({
        "selected_raw_generated": raw[selected].detach().cpu().numpy().astype(np.float32),
        "local_positions": np.asarray(data["offsets"], dtype=np.float32),
        "validation_row": np.asarray(row, np.int64),
        "target_contact_frames": np.asarray([metrics["target_contact_frame"]], np.int64),
        "seed": np.asarray(sample_seed, np.int64),
        "steps": np.asarray(steps, np.int64),
        "position_error_m": np.asarray(metrics["position_error_m"][selected].item(), np.float32),
        "orientation_error_degrees": np.asarray(metrics["orientation_error_degrees"][selected].item(), np.float32),
        "fixed_frame_zero": target_global[0].detach().cpu().numpy().astype(np.float32),
        "generated_frame_zero": raw[selected, 0].detach().cpu().numpy().astype(np.float32),
        "fixed_frame_zero_exact": np.asarray(True),
        "selected_contact_score": np.asarray(
            (metrics["position_error_m"][selected] / 0.04 + metrics["orientation_error_degrees"][selected] / 15.0).item(),
            np.float32,
        ),
    }, preview_path)
    return preview_path


def _select_sampler_steps(quality: Mapping[str, Mapping[str, float]]) -> int:
    twenty, fifty = quality["20"], quality["50"]
    if (
        fifty["attach_proxy_at_8"] > 0.0
        and
        abs(twenty["attach_proxy_at_8"] - fifty["attach_proxy_at_8"]) <= 0.020000001
        and abs(twenty["no_stop_proxy"] - fifty["no_stop_proxy"]) <= 0.020000001
        and twenty["median_grasp_position_m"] <= fifty["median_grasp_position_m"] + 0.002000001
        and twenty["median_grasp_orientation_degrees"] <= fifty["median_grasp_orientation_degrees"] + 2.0000001
    ):
        return 20
    return 50


def _validation_rank(quality: Mapping[str, Mapping[str, float]]) -> tuple[float, float, float]:
    selected = quality[str(_select_sampler_steps(quality))]
    return (
        -selected["attach_proxy_at_8"],
        selected["median_grasp_position_m"],
        -selected["no_stop_proxy"],
    )


def _checkpoint_payload(
    walk_model: MotionWindowDenoiser,
    pickup_model: MotionWindowDenoiser,
    data: Mapping[str, object],
    dataset_sha256: str,
    stage_history: list[dict[str, object]],
    quality: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    selected_stage: Mapping[str, object],
) -> dict[str, object]:
    schema = walk_model.schema
    return {
        "schema_version": "overlap-diffusion-schema-v1",
        "frame_dim": FRAME_DIM,
        "windows": (50, 50, 20, 80),
        "fps": 25.0,
        "prediction_type": "epsilon",
        "seed": seed,
        "model_schema": {"width": schema.width, "blocks": schema.blocks, "heads": schema.heads},
        "walk_model": {name: value.detach().cpu().clone() for name, value in walk_model.state_dict().items()},
        "pickup_model": {name: value.detach().cpu().clone() for name, value in pickup_model.state_dict().items()},
        "normalization": {
            "mean": torch.from_numpy(np.asarray(data["mean"], np.float32)).clone(),
            "scale": torch.from_numpy(np.asarray(data["scale"], np.float32)).clone(),
        },
        "skeleton_signature": str(data["signature"]),
        "skeleton_names": tuple(data["names"]),
        "skeleton_parents": tuple(int(value) for value in np.asarray(data["parents"])),
        "canonical_local_offsets": torch.from_numpy(np.asarray(data["offsets"], np.float32)).clone(),
        "dataset_sha256": dataset_sha256,
        "loss_weights": dict(LOSS_WEIGHTS),
        "stage_history": list(stage_history),
        "selected_stage": dict(selected_stage),
        "selected_sampler_steps": _select_sampler_steps(quality),
        "validation_quality": {key: dict(value) for key, value in quality.items()},
    }


def train_overlap(
    dataset: Path,
    output: Path,
    *,
    seed: int,
    device: str = "cuda",
    epochs_a: int = 1,
    epochs_b: int = 1,
    epochs_c: int = 1,
    batch_size: int = 64,
    validation_subset: int = 4,
    validation_interval: int = 1,
    model_width: int = 256,
    model_blocks: int = 8,
    model_heads: int = 8,
) -> dict[str, object]:
    """Run stage A/B/C without loading a single test-partition value."""
    if any(not isinstance(value, int) or value < 0 for value in (epochs_a, epochs_b, epochs_c)):
        raise ValueError("stage epoch counts must be nonnegative integers")
    if epochs_c <= 0:
        raise ValueError("epochs_c must be positive because only C-stage weights are eligible for best output")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(validation_interval, int) or validation_interval <= 0:
        raise ValueError("validation_interval must be a positive integer")
    data = _read_dataset(Path(dataset))
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    walk_model = MotionWindowDenoiser(width=model_width, blocks=model_blocks, heads=model_heads).to(target_device)
    pickup_model = MotionWindowDenoiser(width=model_width, blocks=model_blocks, heads=model_heads).to(target_device)
    walk_optimizer = torch.optim.AdamW(walk_model.parameters(), lr=3e-4, weight_decay=1e-4)
    pickup_optimizer = torch.optim.AdamW(pickup_model.parameters(), lr=3e-4, weight_decay=1e-4)
    mean = _to_device(np.asarray(data["mean"]), target_device)
    scale = _to_device(np.asarray(data["scale"]), target_device)
    offsets = _to_device(np.asarray(data["offsets"]), target_device)
    alpha_bars = _alpha_bars(target_device)
    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    output = Path(output)
    last_path = output.with_name("last.pt")
    best_path = output.with_name("best.pt")
    log_path = output.with_suffix(".training.jsonl")
    validation_path = output.with_suffix(".validation.json")
    stage_history: list[dict[str, object]] = []
    best_rank: tuple[float, float, float] | None = None
    best_quality: dict[str, dict[str, float]] | None = None
    best_payload: dict[str, object] | None = None
    dataset_sha256 = _sha256(Path(dataset))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def train_single(model, optimizer, windows, statics, temporals) -> float:
        model.train()
        totals: list[float] = []
        for indices in _batch_order(len(windows), batch_size, rng):
            clean_raw = _to_device(windows[indices.numpy()], target_device)
            static = _to_device(statics[indices.numpy()], target_device)
            temporal = _to_device(temporals[indices.numpy()], target_device)
            clean = (clean_raw - mean) / scale
            timestep = torch.randint(DIFFUSION_STEPS, (len(indices),), generator=rng, dtype=torch.long).to(target_device)
            noise = torch.randn(clean.shape, generator=rng, dtype=clean.dtype).to(target_device)
            epsilon, _ = _denoise(model, clean, static, temporal, noise, timestep, alpha_bars)
            loss = torch.nn.functional.mse_loss(epsilon, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        return float(np.mean(totals))

    def train_coupled(walks, pickups, statics, walk_temporal, pickup_temporal) -> float:
        walk_model.train()
        pickup_model.train()
        totals: list[float] = []
        parents = np.asarray(data["parents"], np.int32)
        for indices in _batch_order(len(walks), batch_size, rng):
            item = indices.numpy()
            walk_raw, pickup_raw = _to_device(walks[item], target_device), _to_device(pickups[item], target_device)
            static = _to_device(statics[item], target_device)
            walk_condition, pickup_condition = _to_device(walk_temporal[item], target_device), _to_device(pickup_temporal[item], target_device)
            walk_clean, pickup_clean = (walk_raw - mean) / scale, (pickup_raw - mean) / scale
            timestep = torch.randint(DIFFUSION_STEPS, (len(indices),), generator=rng, dtype=torch.long).to(target_device)
            global_noise = torch.randn((len(indices), TIMELINE_FRAMES, FRAME_DIM), generator=rng, dtype=walk_clean.dtype).to(target_device)
            walk_noise, pickup_noise = global_noise[:, :50], global_noise[:, 30:80]
            walk_epsilon, walk_predicted = _denoise(walk_model, walk_clean, static, walk_condition, walk_noise, timestep, alpha_bars)
            pickup_epsilon, pickup_predicted = _denoise(pickup_model, pickup_clean, static, pickup_condition, pickup_noise, timestep, alpha_bars)
            walk_raw_predicted, pickup_raw_predicted = walk_predicted * scale + mean, pickup_predicted * scale + mean
            walk_losses = _raw_losses(
                walk_raw_predicted, walk_raw, walk_epsilon, walk_noise, static,
                walk_raw_predicted[:, 30:50], pickup_raw_predicted[:, :20], parents, offsets,
            )
            pickup_losses = _raw_losses(
                pickup_raw_predicted, pickup_raw, pickup_epsilon, pickup_noise, static,
                walk_raw_predicted[:, 30:50], pickup_raw_predicted[:, :20], parents, offsets,
            )
            loss = 0.5 * (_weighted(walk_losses) + _weighted(pickup_losses))
            walk_optimizer.zero_grad(set_to_none=True)
            pickup_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(walk_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pickup_model.parameters(), 1.0)
            walk_optimizer.step()
            pickup_optimizer.step()
            totals.append(float(loss.detach().cpu()))
        return float(np.mean(totals))

    stages = (
        ("A", epochs_a), ("B", epochs_b), ("C", epochs_c),
    )
    for stage, epochs in stages:
        for epoch in range(epochs):
            if stage == "A":
                interaction_loss = train_single(
                    walk_model, walk_optimizer, data["train_walk_windows"],
                    data["train_static_conditions"], data["train_walk_temporal"],
                )
                walking_loss = train_single(
                    walk_model, walk_optimizer, data["train_walking_windows"],
                    data["train_walking_static_conditions"], data["train_walking_temporal"],
                )
                train_loss = 0.5 * (interaction_loss + walking_loss)
            elif stage == "B":
                train_loss = train_single(
                    pickup_model, pickup_optimizer, data["train_pickup_windows"],
                    data["train_static_conditions"], data["train_pickup_temporal"],
                )
            else:
                train_loss = train_coupled(
                    data["train_walk_windows"], data["train_pickup_windows"],
                    data["train_static_conditions"], data["train_walk_temporal"], data["train_pickup_temporal"],
                )
            record: dict[str, object] = {"stage": stage, "epoch": epoch + 1, "train_loss": train_loss}
            stage_history.append(record)
            if stage != "C":
                _atomic_torch_save({"stage_history": list(stage_history), "stage": stage}, last_path)
                quality: Mapping[str, Mapping[str, float]] | None = None
            elif (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs_c:
                quality = _quality_summary(
                    walk_model, pickup_model, data, offsets, seed=seed, subset=validation_subset,
                )
                record["validation_rank"] = list(_validation_rank(quality))
                payload = _checkpoint_payload(
                    walk_model, pickup_model, data, dataset_sha256, stage_history, quality,
                    seed=seed, selected_stage={"stage": "C", "epoch": epoch + 1},
                )
                _atomic_torch_save(payload, last_path)
                rank = _validation_rank(quality)
                if best_rank is None or rank < best_rank:
                    _atomic_best_pair(payload, best_path, output)
                    best_rank, best_quality, best_payload = rank, dict(quality), payload
            else:
                quality = None
                _atomic_torch_save({"stage_history": list(stage_history), "stage": stage}, last_path)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**record, **({"validation_quality": quality} if quality is not None else {})}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    if best_quality is None or best_payload is None:
        raise AssertionError("nonzero staged training did not produce validation quality")
    preview_path = _emit_preview(
        best_payload, data, offsets, output, seed=seed, validation_subset=validation_subset,
        device=target_device,
    )
    final = {
        "dataset": str(Path(dataset)), "dataset_sha256": dataset_sha256,
        "device": str(target_device),
        "gpu": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None,
        "validation_subset": min(validation_subset, len(data["validation_pickup_windows"])),
        "quality": best_quality, "selected_sampler_steps": _select_sampler_steps(best_quality),
        "selected_stage": dict(best_payload["selected_stage"]),
        "preview": str(preview_path),
        "loss_weights": dict(LOSS_WEIGHTS),
    }
    _atomic_json(final, validation_path)
    return final


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs-a", type=int, default=1, help="Stage A epochs (default: 1)")
    parser.add_argument("--epochs-b", type=int, default=1, help="Stage B epochs (default: 1)")
    parser.add_argument("--epochs-c", type=int, default=1, help="Stage C epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--validation-subset", type=int, default=4, help="Frozen validation rows (default: 4)")
    parser.add_argument("--validation-interval", type=int, default=1, help="C-stage quality interval (default: 1)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = train_overlap(
        args.dataset, args.output, seed=args.seed, device=args.device,
        epochs_a=args.epochs_a, epochs_b=args.epochs_b, epochs_c=args.epochs_c,
        batch_size=args.batch_size, validation_subset=args.validation_subset,
        validation_interval=args.validation_interval,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
