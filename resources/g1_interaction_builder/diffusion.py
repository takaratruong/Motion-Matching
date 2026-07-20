"""The fixed conditional DDPM used for object-interaction motion funnels."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn


MODEL_CONDITION_DIM = 18
FUNNEL_CHANNELS = 4
FUNNEL_SAMPLES = 16
DIFFUSION_STEPS = 1000
DDIM_STEPS = 50


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_schedule() -> DiffusionSchedule:
    """Build the reviewed float64 schedule in increasing timestep order."""
    betas = torch.linspace(0.0001, 0.02, DIFFUSION_STEPS, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas, alphas, alpha_bars)


def _timestep_embedding(timestep: torch.Tensor, width: int = 128) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class _FiLMBlock(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.film = nn.Linear(256, width * 2)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        y = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, dim=1)
        y = y * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        y = self.conv2(torch.nn.functional.silu(self.norm2(y)))
        return x + y


class FunnelDenoiser(nn.Module):
    """Compact conditional epsilon predictor for [B, 4, 16] local funnels."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(FUNNEL_CHANNELS, 64, 3, padding=1)
        self.timestep = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.condition = nn.Sequential(
            nn.Linear(MODEL_CONDITION_DIM, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.blocks = nn.ModuleList([_FiLMBlock(64) for _ in range(4)])
        self.output_norm = nn.GroupNorm(8, 64)
        self.output = nn.Conv1d(64, FUNNEL_CHANNELS, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (FUNNEL_CHANNELS, FUNNEL_SAMPLES):
            raise ValueError(f"expected x shape [B, 4, 16], got {tuple(x.shape)}")
        if condition.ndim != 2 or condition.shape[1] != MODEL_CONDITION_DIM:
            raise ValueError("condition must have shape [B, 18]")
        embedding = torch.cat(
            (self.timestep(_timestep_embedding(timestep)), self.condition(condition)),
            dim=1,
        )
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))


def ddim_timesteps(step_count: int = DDIM_STEPS) -> tuple[int, ...]:
    if step_count <= 0 or step_count > DIFFUSION_STEPS:
        raise ValueError("step_count must be in [1, 1000]")
    return tuple(math.floor(i * 999 / (step_count - 1)) for i in range(step_count - 1, -1, -1))


@torch.no_grad()
def sample_ddim(
    model: FunnelDenoiser,
    condition: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    seed: int,
    proposal_count: int = 32,
    step_count: int = DDIM_STEPS,
) -> torch.Tensor:
    """Sample a deterministic proposal-major batch, shaped [B, 32, 16, 4]."""
    if condition.ndim != 2 or condition.shape[1] != MODEL_CONDITION_DIM:
        raise ValueError("condition must have shape [B, 18]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = condition.shape[0]
    x = torch.randn(
        (batch * proposal_count, FUNNEL_CHANNELS, FUNNEL_SAMPLES),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(condition.device)
    repeated_condition = condition.repeat_interleave(proposal_count, dim=0)
    timesteps = ddim_timesteps(step_count)
    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        t = torch.full((x.shape[0],), timestep, dtype=torch.long, device=x.device)
        epsilon = model(x, t, repeated_condition)
        alpha_bar = schedule.alpha_bars[timestep].to(x.device, torch.float32)
        previous_bar = (
            torch.tensor(1.0, device=x.device)
            if previous < 0
            else schedule.alpha_bars[previous].to(x.device, torch.float32)
        )
        x0 = (x - torch.sqrt(1.0 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar)
        x = torch.sqrt(previous_bar) * x0 + torch.sqrt(1.0 - previous_bar) * epsilon
    return x.reshape(batch, proposal_count, FUNNEL_SAMPLES, FUNNEL_CHANNELS).contiguous()


def train_funnel(
    conditions: torch.Tensor,
    funnels: torch.Tensor,
    output: Path,
    *,
    steps: int = 20_000,
    batch_size: int = 256,
    seed: int = 2_026_071_901,
    device: str | None = None,
) -> dict:
    """Train and publish one self-contained, normalized funnel checkpoint."""
    if conditions.ndim != 2 or tuple(conditions.shape[1:]) != (18,):
        raise ValueError("conditions must have shape [N, 18]")
    if funnels.ndim != 3 or tuple(funnels.shape[1:]) != (16, 4):
        raise ValueError("funnels must have shape [N, 16, 4]")
    if len(conditions) != len(funnels) or len(conditions) == 0:
        raise ValueError("training rows must be nonempty and aligned")
    torch.manual_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    condition_mean = conditions.mean(dim=0)
    condition_scale = conditions.std(dim=0, unbiased=False).clamp_min(1e-6)
    funnel_mean = funnels.mean(dim=(0, 1), keepdim=True)
    funnel_scale = funnels.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized_conditions = ((conditions - condition_mean) / condition_scale).to(target_device)
    normalized_funnels = ((funnels - funnel_mean) / funnel_scale).permute(0, 2, 1).to(target_device)
    schedule = make_schedule()
    model = FunnelDenoiser().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0001)
    generator = torch.Generator(device=target_device).manual_seed(seed + 1)
    model.train()
    for step in range(steps):
        indices = torch.randint(len(normalized_funnels), (min(batch_size, len(normalized_funnels)),), generator=generator, device=target_device)
        clean = normalized_funnels[indices]
        condition = normalized_conditions[indices]
        timestep = torch.randint(DIFFUSION_STEPS, (len(indices),), generator=generator, device=target_device)
        alpha_bar = schedule.alpha_bars.to(target_device)[timestep].to(torch.float32).view(-1, 1, 1)
        noise = torch.randn(clean.shape, generator=generator, device=target_device)
        noised = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
        prediction = model(noised, timestep, condition)
        loss = torch.nn.functional.mse_loss(prediction, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    checkpoint = {
        "schema_version": 1,
        "seed": seed,
        "model": model.cpu().state_dict(),
        "condition_mean": condition_mean.cpu(),
        "condition_scale": condition_scale.cpu(),
        "funnel_mean": funnel_mean.cpu(),
        "funnel_scale": funnel_scale.cpu(),
        "schedule": make_schedule(),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return checkpoint


def load_funnel_checkpoint(path: Path, *, device: str = "cpu") -> tuple[FunnelDenoiser, DiffusionSchedule, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported funnel checkpoint schema")
    model = FunnelDenoiser().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = checkpoint["schedule"]
    if not isinstance(schedule, DiffusionSchedule):
        raise ValueError("checkpoint has no valid diffusion schedule")
    return model, schedule, checkpoint


@torch.no_grad()
def sample_checkpoint(
    path: Path,
    condition: torch.Tensor,
    *,
    device: str = "cpu",
    seed: int,
    step_count: int = DDIM_STEPS,
) -> torch.Tensor:
    model, schedule, checkpoint = load_funnel_checkpoint(path, device=device)
    raw_condition = condition.to(device=device, dtype=torch.float32)
    normalized = (raw_condition - checkpoint["condition_mean"].to(device)) / checkpoint["condition_scale"].to(device)
    normalized_samples = sample_ddim(model, normalized, schedule, seed=seed, step_count=step_count)
    samples = normalized_samples * checkpoint["funnel_scale"].to(device) + checkpoint["funnel_mean"].to(device)
    yaw = samples[..., 2:4].to(torch.float64)
    norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
    if not torch.isfinite(norm).all() or (norm < 1e-8).any():
        raise ValueError("sampled yaw direction is degenerate")
    samples[..., 2:4] = (yaw / norm).to(torch.float32)
    return samples
