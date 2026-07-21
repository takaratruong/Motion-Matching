"""Coupled full-body overlap diffusion with one authoritative global latent.

The two experts intentionally have independent parameters.  Sampling never has
two independently evolving window latents: each reverse step reads two slices
from, and writes one DDIM update to, a single ``[candidates, 80, 195]`` tensor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn

from resources.g1_interaction_builder.schema import G1_SKELETON


FRAME_DIM = 195
ROOT_TRANSLATION_SLICE = slice(0, 3)
HIPS_TRANSLATION_SLICE = slice(3, 6)
ROTATION6D_SLICE = slice(6, 6 + 31 * 6)
CONTACT_SLICE = slice(6 + 31 * 6, FRAME_DIM)
STATIC_CONDITION_DIM = 25
TEMPORAL_CONDITION_DIM = 9
WINDOW_FRAMES = 50
OVERLAP_FRAMES = 20
TIMELINE_FRAMES = 80
DIFFUSION_STEPS = 1000
MODEL_WIDTH = 256
MODEL_BLOCKS = 8
MODEL_HEADS = 8


def _require_tensor(
    value: torch.Tensor, shape: tuple[int, ...], label: str, *, boolean: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if boolean:
        if value.dtype is not torch.bool:
            raise ValueError(f"{label} must have dtype torch.bool")
        return
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be floating point and finite")


def _require_motion(value: torch.Tensor, label: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[1:] != (WINDOW_FRAMES, FRAME_DIM)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{label} must be a finite floating tensor with shape [B, {WINDOW_FRAMES}, {FRAME_DIM}]"
        )


def _require_expert_epsilon(
    value: torch.Tensor,
    label: str,
    *,
    candidates: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Reject any expert result that cannot be one global-DDIM prediction."""
    expected_shape = (candidates, WINDOW_FRAMES, FRAME_DIM)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}")
    if value.dtype != dtype:
        raise ValueError(f"{label} must have dtype {dtype}")
    if value.device != device:
        raise ValueError(f"{label} must stay on device {device}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")


@dataclass(frozen=True)
class CoupledCondition:
    """The one-request static and global temporal conditioning contract.

    ``static`` is one 25-value condition shared by every candidate.  ``temporal``
    is one global 80-frame, 9-channel condition; the sampler selects its two
    window slices so both experts see the same overlap convention as the latent.
    """

    static: torch.Tensor
    temporal: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor(self.static, (STATIC_CONDITION_DIM,), "static")
        _require_tensor(
            self.temporal, (TIMELINE_FRAMES, TEMPORAL_CONDITION_DIM), "temporal"
        )
        if self.static.device != self.temporal.device:
            raise ValueError("static and temporal conditions must be on the same device")


@dataclass(frozen=True)
class FixedFrames:
    """Whole-frame inpainting values and an explicit global-frame mask."""

    values: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor(self.values, (TIMELINE_FRAMES, FRAME_DIM), "fixed values")
        _require_tensor(self.mask, (TIMELINE_FRAMES,), "fixed mask", boolean=True)
        if self.values.device != self.mask.device:
            raise ValueError("fixed values and mask must be on the same device")


GuidanceObjective = Callable[[torch.Tensor, CoupledCondition, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TaskGuidance:
    """A differentiable global clean-motion objective used during sampling.

    The objective receives a clean estimate ``[B,80,195]``, the validated
    condition, and a same-device scalar timestep.  It must return a finite
    scalar or a finite per-candidate tensor.  Its gradient is applied to the
    clean estimate, so route/floor/contact/grasp/attachment objectives can be
    supplied without a second, divergent sampler.
    """

    objective: GuidanceObjective
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise ValueError("task guidance objective must be callable")
        if not math.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("task guidance strength must be finite and nonnegative")


@dataclass(frozen=True)
class DenoiserSchema:
    frame_dim: int
    static_dim: int
    temporal_dim: int
    width: int
    blocks: int
    heads: int


def _timestep_embedding(timestep: torch.Tensor, width: int = MODEL_WIDTH) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class _TemporalTransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.attention_norm(hidden), self.attention_norm(hidden), self.attention_norm(hidden),
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class MotionWindowDenoiser(nn.Module):
    """The schema-shared 50-frame epsilon denoiser used by each expert."""

    def __init__(
        self,
        frame_dim: int = FRAME_DIM,
        static_dim: int = STATIC_CONDITION_DIM,
        temporal_dim: int = TEMPORAL_CONDITION_DIM,
        *,
        width: int = MODEL_WIDTH,
        blocks: int = MODEL_BLOCKS,
        heads: int = MODEL_HEADS,
    ) -> None:
        super().__init__()
        if (frame_dim, static_dim, temporal_dim) != (
            FRAME_DIM, STATIC_CONDITION_DIM, TEMPORAL_CONDITION_DIM,
        ):
            raise ValueError("MotionWindowDenoiser uses the frozen 195/25/9 schema")
        if (
            not isinstance(width, int) or width < 2
            or not isinstance(blocks, int) or blocks <= 0
            or not isinstance(heads, int) or heads <= 0 or width % heads
        ):
            raise ValueError("denoiser width/blocks/heads must form a positive attention schema")
        self.schema = DenoiserSchema(
            frame_dim, static_dim, temporal_dim, width, blocks, heads
        )
        self.frame_projection = nn.Linear(frame_dim, width)
        self.static_projection = nn.Linear(static_dim, width)
        self.temporal_projection = nn.Linear(temporal_dim, width)
        self.timestep_projection = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [_TemporalTransformerBlock(width, heads) for _ in range(blocks)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, frame_dim)

    def forward(
        self,
        noisy_frames: torch.Tensor,
        timestep: torch.Tensor,
        static_condition: torch.Tensor,
        temporal_condition: torch.Tensor,
    ) -> torch.Tensor:
        _require_motion(noisy_frames, "noisy_frames")
        batch = noisy_frames.shape[0]
        _require_tensor(static_condition, (batch, STATIC_CONDITION_DIM), "static_condition")
        _require_tensor(
            temporal_condition, (batch, WINDOW_FRAMES, TEMPORAL_CONDITION_DIM), "temporal_condition"
        )
        if (
            timestep.ndim != 1
            or timestep.shape[0] != batch
            or timestep.device != noisy_frames.device
            or timestep.dtype not in (torch.int32, torch.int64)
            or torch.any(timestep < 0)
            or torch.any(timestep >= DIFFUSION_STEPS)
        ):
            raise ValueError("timestep must be [B] integer diffusion indices in [0, 1000)")
        if static_condition.device != noisy_frames.device or temporal_condition.device != noisy_frames.device:
            raise ValueError("denoiser inputs must be on one device")
        hidden = (
            self.frame_projection(noisy_frames)
            + self.static_projection(static_condition).unsqueeze(1)
            + self.temporal_projection(temporal_condition)
            + self.timestep_projection(_timestep_embedding(timestep, self.schema.width)).unsqueeze(1)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))


def overlap_weight(index: int) -> float:
    """Pickup expert weight for shared global frame ``30 + index``."""
    if not isinstance(index, int) or not 0 <= index < OVERLAP_FRAMES:
        raise ValueError("overlap index must be an integer in [0, 20)")
    return index / (OVERLAP_FRAMES - 1)


def _ddim_timesteps(steps: int) -> tuple[int, ...]:
    if not isinstance(steps, int) or not 1 <= steps <= DIFFUSION_STEPS:
        raise ValueError("steps must be an integer in [1, 1000]")
    if steps == 1:
        return (DIFFUSION_STEPS - 1,)
    return tuple(
        math.floor(index * (DIFFUSION_STEPS - 1) / (steps - 1))
        for index in range(steps - 1, -1, -1)
    )


def _alpha_bars(device: torch.device) -> torch.Tensor:
    betas = torch.linspace(0.0001, 0.02, DIFFUSION_STEPS, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0).to(device=device, dtype=torch.float32)


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return fallback if buffer is None else buffer.device


def _fixed_values(
    fixed_frames: FixedFrames | None, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if fixed_frames is None:
        return None, None
    return (
        fixed_frames.values.to(device=device, dtype=dtype),
        fixed_frames.mask.to(device=device),
    )


def _apply_fixed(
    values: torch.Tensor, fixed_values: torch.Tensor | None, fixed_mask: torch.Tensor | None,
) -> torch.Tensor:
    if fixed_values is None or fixed_mask is None:
        return values
    result = values.clone()
    result[:, fixed_mask] = fixed_values[fixed_mask]
    return result


def _apply_task_guidance(
    clean: torch.Tensor,
    guidance: TaskGuidance | None,
    condition: CoupledCondition,
    timestep: int,
) -> torch.Tensor:
    if guidance is None or guidance.strength == 0.0:
        return clean
    candidate = clean.detach().requires_grad_(True)
    value = guidance.objective(
        candidate, condition, torch.tensor(timestep, device=clean.device, dtype=torch.long)
    )
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("task guidance objective must return a finite floating tensor")
    if value.ndim > 1 or (value.ndim == 1 and value.shape[0] != clean.shape[0]):
        raise ValueError("task guidance objective must return a scalar or one value per candidate")
    if not value.requires_grad:
        raise ValueError("task guidance objective must be differentiable with respect to clean motion")
    gradient = torch.autograd.grad(value.sum(), candidate, allow_unused=False)[0]
    guided = clean - guidance.strength * gradient
    if not torch.isfinite(guided).all():
        raise ValueError("task guidance produced non-finite clean motion")
    return guided.detach()


@dataclass(frozen=True)
class CoupledStepTrace:
    """CPU-testable evidence that both experts read one latent and update once."""

    walk_input: torch.Tensor
    pickup_input: torch.Tensor
    global_epsilon: torch.Tensor
    unguided_clean: torch.Tensor
    guided_clean: torch.Tensor
    global_latent: torch.Tensor


def sample_coupled(
    walk_model: nn.Module,
    pickup_model: nn.Module,
    condition: CoupledCondition,
    *,
    seed: int,
    candidates: int = 8,
    steps: int = 20,
    fixed_frames: FixedFrames | None = None,
    task_guidance: TaskGuidance | None = None,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[CoupledStepTrace, ...]]:
    """Deterministically sample eight coupled 80-frame timelines.

    Noise is *always* generated by a CPU ``Generator`` before one transfer to
    the shared model device.  This makes a CPU seed independent of ambient RNG
    state and avoids CUDA generator/device-specific random streams.
    """
    if not isinstance(condition, CoupledCondition):
        raise ValueError("condition must be a CoupledCondition")
    if not isinstance(candidates, int) or candidates <= 0:
        raise ValueError("candidates must be a positive integer")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if fixed_frames is not None and not isinstance(fixed_frames, FixedFrames):
        raise ValueError("fixed_frames must be FixedFrames or None")
    if task_guidance is not None and not isinstance(task_guidance, TaskGuidance):
        raise ValueError("task_guidance must be TaskGuidance or None")
    timesteps = _ddim_timesteps(steps)
    walk_device = _model_device(walk_model, condition.static.device)
    pickup_device = _model_device(pickup_model, condition.static.device)
    if walk_device != pickup_device:
        raise ValueError("walk and pickup models must be on the same device")
    device = walk_device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(
        (candidates, TIMELINE_FRAMES, FRAME_DIM),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device)
    static = condition.static.to(device=device, dtype=latent.dtype).expand(candidates, -1)
    temporal = condition.temporal.to(device=device, dtype=latent.dtype)
    guidance_condition = CoupledCondition(static=static[0], temporal=temporal)
    fixed_values, fixed_mask = _fixed_values(fixed_frames, device, latent.dtype)
    latent = _apply_fixed(latent, fixed_values, fixed_mask)
    alpha_bars = _alpha_bars(device)
    traces: list[CoupledStepTrace] = []

    walk_training = walk_model.training
    pickup_training = pickup_model.training
    try:
        walk_model.eval()
        pickup_model.eval()
        for index, timestep in enumerate(timesteps):
            walk_input = latent[:, :WINDOW_FRAMES].clone()
            pickup_input = latent[:, WINDOW_FRAMES - OVERLAP_FRAMES:].clone()
            timestep_tensor = torch.full(
                (candidates,), timestep, device=device, dtype=torch.long
            )
            with torch.inference_mode():
                walk_epsilon = walk_model(
                    walk_input, timestep_tensor, static,
                    temporal[:WINDOW_FRAMES].expand(candidates, -1, -1),
                )
                pickup_epsilon = pickup_model(
                    pickup_input, timestep_tensor, static,
                    temporal[WINDOW_FRAMES - OVERLAP_FRAMES:].expand(candidates, -1, -1),
                )
            _require_expert_epsilon(
                walk_epsilon, "walk epsilon", candidates=candidates,
                device=device, dtype=latent.dtype,
            )
            _require_expert_epsilon(
                pickup_epsilon, "pickup epsilon", candidates=candidates,
                device=device, dtype=latent.dtype,
            )
            epsilon = torch.empty_like(latent)
            epsilon[:, :WINDOW_FRAMES - OVERLAP_FRAMES] = walk_epsilon[:, :WINDOW_FRAMES - OVERLAP_FRAMES]
            epsilon[:, WINDOW_FRAMES:] = pickup_epsilon[:, OVERLAP_FRAMES:]
            for overlap_index in range(OVERLAP_FRAMES):
                pickup_weight = overlap_weight(overlap_index)
                epsilon[:, WINDOW_FRAMES - OVERLAP_FRAMES + overlap_index] = (
                    (1.0 - pickup_weight) * walk_epsilon[:, WINDOW_FRAMES - OVERLAP_FRAMES + overlap_index]
                    + pickup_weight * pickup_epsilon[:, overlap_index]
                )
            if not torch.isfinite(epsilon).all():
                raise ValueError("global epsilon must be finite")
            alpha_bar = alpha_bars[timestep]
            unguided_clean = (
                latent - torch.sqrt(1.0 - alpha_bar) * epsilon
            ) / torch.sqrt(alpha_bar)
            guided_clean = _apply_fixed(unguided_clean, fixed_values, fixed_mask)
            guided_clean = _apply_task_guidance(
                guided_clean, task_guidance, guidance_condition, timestep
            )
            guided_clean = _apply_fixed(guided_clean, fixed_values, fixed_mask)
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            previous_bar = (
                torch.ones((), device=device, dtype=latent.dtype)
                if previous < 0
                else alpha_bars[previous]
            )
            latent = (
                torch.sqrt(previous_bar) * guided_clean
                + torch.sqrt(1.0 - previous_bar) * epsilon
            )
            latent = _apply_fixed(latent, fixed_values, fixed_mask)
            if return_trace:
                traces.append(
                    CoupledStepTrace(
                        walk_input.detach().clone(), pickup_input.detach().clone(),
                        epsilon.detach().clone(), unguided_clean.detach().clone(),
                        guided_clean.detach().clone(), latent.detach().clone(),
                    )
                )
    finally:
        walk_model.train(walk_training)
        pickup_model.train(pickup_training)
    result = latent.detach()
    return (result, tuple(traces)) if return_trace else result


def _finite_difference(values: torch.Tensor) -> torch.Tensor:
    return values[:, 1:] - values[:, :-1]


def _mean_square(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean()


def _validate_fk_result(result: Mapping[str, torch.Tensor], batch: int, label: str) -> Mapping[str, torch.Tensor]:
    required = ("hand", "hand_orientation", "left_foot", "right_foot")
    if not isinstance(result, Mapping) or any(name not in result for name in required):
        raise ValueError(f"{label} must map hand, hand_orientation, left_foot, and right_foot")
    for name in ("hand", "left_foot", "right_foot"):
        _require_tensor(result[name], (batch, WINDOW_FRAMES, 3), f"{label}[{name}]")
    _require_tensor(result["hand_orientation"], (batch, WINDOW_FRAMES, 6), f"{label}[hand_orientation]")
    return result


def named_training_losses(
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    predicted_epsilon: torch.Tensor,
    target_epsilon: torch.Tensor,
    *,
    fk: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
    grasp_position: torch.Tensor,
    grasp_orientation: torch.Tensor,
    object_position: torch.Tensor,
    object_clearance: torch.Tensor,
    overlap_walk: torch.Tensor,
    overlap_pickup: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return Task 3 losses under the contact-transition contract.

    ``object_clearance`` is the finite, nonnegative per-object radius used only
    before first hand contact.  ``grasp_position`` and ``grasp_orientation``
    are the already conditioned object-relative grasp transform expressed as
    the target hand pose; first-contact grasp and sustained attachment losses
    must never substitute the object center for that target.
    """
    for value, label in (
        (predicted_clean, "predicted_clean"), (target_clean, "target_clean"),
        (predicted_epsilon, "predicted_epsilon"), (target_epsilon, "target_epsilon"),
    ):
        _require_motion(value, label)
    if (
        predicted_clean.shape != target_clean.shape
        or predicted_epsilon.shape != target_epsilon.shape
        or predicted_clean.shape != predicted_epsilon.shape
    ):
        raise ValueError("clean and epsilon tensors must have identical shapes")
    batch = predicted_clean.shape[0]
    _require_tensor(grasp_position, (batch, 3), "grasp_position")
    _require_tensor(grasp_orientation, (batch, 6), "grasp_orientation")
    _require_tensor(object_position, (batch, 3), "object_position")
    _require_tensor(object_clearance, (batch,), "object_clearance")
    if torch.any(object_clearance < 0.0):
        raise ValueError("object_clearance must be finite and nonnegative")
    expected_overlap = (batch, OVERLAP_FRAMES, FRAME_DIM)
    _require_tensor(overlap_walk, expected_overlap, "overlap_walk")
    _require_tensor(overlap_pickup, expected_overlap, "overlap_pickup")
    if not callable(fk):
        raise ValueError("fk must be callable")
    predicted_fk = _validate_fk_result(fk(predicted_clean), batch, "predicted fk")
    target_fk = _validate_fk_result(fk(target_clean), batch, "target fk")
    pose_channels = ROTATION6D_SLICE
    foot_contacts = target_clean[..., CONTACT_SLICE.start:CONTACT_SLICE.stop - 1]
    active_hand_contact = target_clean[..., CONTACT_SLICE.stop - 1] >= 0.5
    contact_count = active_hand_contact.cumsum(dim=1)
    first_contact = active_hand_contact & (contact_count == 1)
    post_contact = active_hand_contact & (contact_count > 1)
    pre_contact = contact_count == 0
    first_contact_weight = first_contact.to(predicted_clean.dtype).unsqueeze(-1)
    post_contact_weight = post_contact.to(predicted_clean.dtype).unsqueeze(-1)
    pre_contact_weight = pre_contact.to(predicted_clean.dtype).unsqueeze(-1)
    first_contact_count = first_contact_weight.sum().clamp_min(1.0)
    post_contact_count = post_contact_weight.sum().clamp_min(1.0)
    pre_contact_count = pre_contact_weight.sum().clamp_min(1.0)
    foot_weight = foot_contacts.sum().clamp_min(1.0)
    predicted_foot_velocity = (
        _finite_difference(predicted_fk["left_foot"]).square() * foot_contacts[:, 1:, :1]
        + _finite_difference(predicted_fk["right_foot"]).square() * foot_contacts[:, 1:, 1:]
    )
    return {
        "epsilon": _mean_square(predicted_epsilon - target_epsilon),
        "pose_6d": _mean_square(predicted_clean[..., pose_channels] - target_clean[..., pose_channels]),
        "fk_hand": _mean_square(predicted_fk["hand"] - target_fk["hand"]),
        "fk_feet": 0.5 * (
            _mean_square(predicted_fk["left_foot"] - target_fk["left_foot"])
            + _mean_square(predicted_fk["right_foot"] - target_fk["right_foot"])
        ),
        "velocity": _mean_square(
            _finite_difference(predicted_clean[..., :CONTACT_SLICE.start])
            - _finite_difference(target_clean[..., :CONTACT_SLICE.start])
        ),
        "acceleration": _mean_square(
            _finite_difference(_finite_difference(predicted_clean[..., :CONTACT_SLICE.start]))
            - _finite_difference(_finite_difference(target_clean[..., :CONTACT_SLICE.start]))
        ),
        "foot_contact": _mean_square(
            predicted_clean[..., CONTACT_SLICE.start:CONTACT_SLICE.stop - 1]
            - target_clean[..., CONTACT_SLICE.start:CONTACT_SLICE.stop - 1]
        ),
        "foot_sliding": predicted_foot_velocity.sum() / (foot_weight * 3.0),
        "grasp_position": (
            ((predicted_fk["hand"] - grasp_position[:, None]).square() * first_contact_weight).sum()
            / (first_contact_count * 3.0)
        ),
        "grasp_orientation": (
            ((predicted_fk["hand_orientation"] - grasp_orientation[:, None]).square() * first_contact_weight).sum()
            / (first_contact_count * 6.0)
        ),
        "pre_contact_separation": (
            (
                torch.relu(
                    object_clearance[:, None, None]
                    - torch.linalg.vector_norm(predicted_fk["hand"] - object_position[:, None], dim=-1, keepdim=True)
                ).square()
                * pre_contact_weight
            ).sum()
            / pre_contact_count
        ),
        "attachment": (
            0.5 * (
                ((predicted_fk["hand"] - grasp_position[:, None]).square() * post_contact_weight).sum()
                / (post_contact_count * 3.0)
            )
            + 0.5 * (
                (
                    (predicted_fk["hand_orientation"] - grasp_orientation[:, None]).square()
                    * post_contact_weight
                ).sum()
                / (post_contact_count * 6.0)
            )
        ),
        "overlap_agreement": _mean_square(overlap_walk - overlap_pickup),
    }


LOSS_NAMES = (
    "epsilon", "pose_6d", "fk_hand", "fk_feet", "velocity", "acceleration",
    "foot_contact", "foot_sliding", "grasp_position", "grasp_orientation",
    "pre_contact_separation", "attachment", "overlap_agreement",
)


def _rotation6d_matrices(rotation6d: torch.Tensor) -> torch.Tensor:
    """Gram--Schmidt 6-D rotations without any identity-FK fallback."""
    first = torch.nn.functional.normalize(rotation6d[..., :3], dim=-1, eps=1e-8)
    second = rotation6d[..., 3:] - (first * rotation6d[..., 3:]).sum(-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def canonical_g1_fk(
    frames: torch.Tensor,
    *,
    parents: torch.Tensor | tuple[int, ...] | list[int],
    local_offsets: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Differentiably FK the real canonical G1 hierarchy from 195 channels.

    Simulation-root and Hips translations are read from their dynamic channels;
    only bones 2..30 use the frozen dataset offsets.  The returned active hand
    is the canonical right wrist (bone 30), with LeftToe/RightToe at 7/13.
    """
    _require_motion(frames, "frames")
    expected_parents = tuple(int(value) for value in G1_SKELETON.parents)
    actual_parents = tuple(int(value) for value in torch.as_tensor(parents).cpu().tolist())
    if actual_parents != expected_parents:
        raise ValueError("FK parents must match the canonical G1 hierarchy")
    offsets = torch.as_tensor(local_offsets, device=frames.device, dtype=frames.dtype)
    if tuple(offsets.shape) != (31, 3) or not torch.isfinite(offsets).all():
        raise ValueError("FK local_offsets must be finite with shape (31, 3)")
    if not torch.equal(offsets[:2], torch.zeros_like(offsets[:2])):
        raise ValueError("FK local offsets for Simulation and Hips must be dynamic zeros")
    batch, steps, _ = frames.shape
    local = offsets.view(1, 1, 31, 3).expand(batch, steps, -1, -1).clone()
    local[:, :, 0] = frames[..., ROOT_TRANSLATION_SLICE]
    local[:, :, 1] = frames[..., HIPS_TRANSLATION_SLICE]
    rotations = _rotation6d_matrices(frames[..., ROTATION6D_SLICE].reshape(batch, steps, 31, 6))
    world_positions: list[torch.Tensor] = []
    world_rotations: list[torch.Tensor] = []
    for bone, parent in enumerate(expected_parents):
        if parent < 0:
            world_positions.append(local[:, :, bone])
            world_rotations.append(rotations[:, :, bone])
            continue
        parent_rotation = world_rotations[parent]
        world_positions.append(
            world_positions[parent] + torch.matmul(parent_rotation, local[:, :, bone].unsqueeze(-1)).squeeze(-1)
        )
        world_rotations.append(torch.matmul(parent_rotation, rotations[:, :, bone]))
    hand_rotation = world_rotations[30]
    return {
        "hand": world_positions[30],
        "hand_orientation": torch.cat((hand_rotation[..., :, 0], hand_rotation[..., :, 1]), dim=-1),
        "left_foot": world_positions[7],
        "right_foot": world_positions[13],
    }


@dataclass(frozen=True)
class LoadedOverlapModels:
    walk_model: MotionWindowDenoiser
    pickup_model: MotionWindowDenoiser
    payload: Mapping[str, object]


def _checkpoint_tensor(value: object, shape: tuple[int, ...], label: str, *, positive: bool = False) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
        raise ValueError(f"checkpoint {label} must be a floating tensor with shape {shape}")
    if not torch.isfinite(value).all() or (positive and torch.any(value <= 0.0)):
        raise ValueError(f"checkpoint {label} must be finite" + (" and positive" if positive else ""))
    return value


def _validate_state_dict(value: object, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"checkpoint {label} must be a nonempty state dictionary")
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint {label} has an invalid state entry")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint {label} has non-finite state")
    return value


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _contains_test_partition(value: object) -> bool:
    if isinstance(value, Mapping):
        return any("test" in str(key).lower() or _contains_test_partition(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_test_partition(item) for item in value)
    return False


def load_overlap_checkpoint(path: Path, device: str | torch.device = "cpu") -> LoadedOverlapModels:
    """Strictly load the self-contained schema-v1 coupled-expert checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "overlap-diffusion-schema-v1":
        raise ValueError("unsupported overlap checkpoint schema")
    if payload.get("frame_dim") != FRAME_DIM or tuple(payload.get("windows", ())) != (50, 50, 20, 80):
        raise ValueError("checkpoint has an unsupported overlap frame/window schema")
    if payload.get("fps") != 25.0 or payload.get("prediction_type") != "epsilon":
        raise ValueError("checkpoint has an unsupported prediction contract")
    if payload.get("skeleton_signature") != G1_SKELETON.signature():
        raise ValueError("checkpoint skeleton signature does not match canonical G1")
    if tuple(payload.get("skeleton_names", ())) != G1_SKELETON.names:
        raise ValueError("checkpoint skeleton names do not match canonical G1")
    if tuple(payload.get("skeleton_parents", ())) != tuple(int(value) for value in G1_SKELETON.parents):
        raise ValueError("checkpoint skeleton parents do not match canonical G1")
    offsets = _checkpoint_tensor(payload.get("canonical_local_offsets"), (31, 3), "canonical_local_offsets")
    if not torch.equal(offsets[:2], torch.zeros_like(offsets[:2])):
        raise ValueError("checkpoint canonical local offsets must leave root/Hips dynamic")
    digest = payload.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("checkpoint dataset SHA256 is invalid")
    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("checkpoint normalization is missing")
    _checkpoint_tensor(normalization.get("mean"), (FRAME_DIM,), "normalization mean")
    _checkpoint_tensor(normalization.get("scale"), (FRAME_DIM,), "normalization scale", positive=True)
    weights = payload.get("loss_weights")
    if not isinstance(weights, Mapping) or set(weights) != set(LOSS_NAMES):
        raise ValueError("checkpoint loss weights do not match named loss contract")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("checkpoint loss weights must be finite and nonnegative")
    steps = payload.get("selected_sampler_steps")
    if steps not in (20, 50):
        raise ValueError("checkpoint selected sampler steps must be 20 or 50")
    history = payload.get("stage_history")
    if not isinstance(history, list) or not history or any(not isinstance(row, Mapping) for row in history):
        raise ValueError("checkpoint stage history is invalid")
    for row in history:
        if (
            set(row) != {"stage", "epoch", "train_loss", "validation_rank"}
            or row["stage"] not in {"A", "B", "C"}
            or not isinstance(row["epoch"], int) or row["epoch"] <= 0
            or not _finite_number(row["train_loss"])
            or not isinstance(row["validation_rank"], (list, tuple))
            or len(row["validation_rank"]) != 3
            or not all(_finite_number(item) for item in row["validation_rank"])
        ):
            raise ValueError("checkpoint stage history has invalid values")
    quality = payload.get("validation_quality")
    if not isinstance(quality, Mapping) or set(quality) != {"20", "50"}:
        raise ValueError("checkpoint validation-only quality summary is invalid")
    quality_fields = {
        "attach_proxy_at_8", "median_grasp_position_m", "median_grasp_orientation_degrees",
        "no_stop_proxy", "rows",
    }
    if any(
        not isinstance(summary, Mapping) or set(summary) != quality_fields
        or not all(_finite_number(value) for value in summary.values())
        for summary in quality.values()
    ):
        raise ValueError("checkpoint validation quality contains invalid values")
    if _contains_test_partition(payload):
        raise ValueError("checkpoint must not serialize test-partition statistics")
    schema = payload.get("model_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("checkpoint model schema is missing")
    values = tuple(schema.get(key) for key in ("width", "blocks", "heads"))
    if not all(isinstance(value, int) for value in values):
        raise ValueError("checkpoint model schema is invalid")
    walk = MotionWindowDenoiser(width=values[0], blocks=values[1], heads=values[2]).to(device)
    pickup = MotionWindowDenoiser(width=values[0], blocks=values[1], heads=values[2]).to(device)
    walk.load_state_dict(_validate_state_dict(payload.get("walk_model"), "walk_model"), strict=True)
    pickup.load_state_dict(_validate_state_dict(payload.get("pickup_model"), "pickup_model"), strict=True)
    walk.eval()
    pickup.eval()
    return LoadedOverlapModels(walk, pickup, payload)
