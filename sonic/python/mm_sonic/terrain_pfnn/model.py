from __future__ import annotations

import math

import torch
from torch import nn

from .layout import INPUT_LAYOUT, OUTPUT_LAYOUT


def catmull_rom_phase_banks(banks: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    if banks.ndim < 2 or banks.shape[0] != 4:
        raise ValueError("phase banks must have shape [4,...]")
    if phase.ndim != 1:
        raise ValueError("phase must have shape [B]")
    coordinate = torch.remainder(phase, 2.0 * math.pi) * (4.0 / (2.0 * math.pi))
    k1 = torch.floor(coordinate).to(torch.long) % 4
    amount = coordinate - torch.floor(coordinate)
    k0, k2, k3 = (k1 - 1) % 4, (k1 + 1) % 4, (k1 + 2) % 4
    p0, p1, p2, p3 = banks[k0], banks[k1], banks[k2], banks[k3]
    shape = (len(phase),) + (1,) * (banks.ndim - 1)
    w = amount.reshape(shape)
    return (
        p1
        + 0.5 * w * (p2 - p0)
        + w.square() * (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3)
        + w.pow(3) * (1.5 * p1 - 1.5 * p2 + 0.5 * p3 - 0.5 * p0)
    )


def _batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias


class PhaseFunctionedNetwork(nn.Module):
    def __init__(self, hidden_size: int = 512, dropout_probability: float = 0.30) -> None:
        super().__init__()
        self.W0 = nn.Parameter(torch.empty(4, hidden_size, INPUT_LAYOUT.size))
        self.b0 = nn.Parameter(torch.zeros(4, hidden_size))
        self.W1 = nn.Parameter(torch.empty(4, hidden_size, hidden_size))
        self.b1 = nn.Parameter(torch.zeros(4, hidden_size))
        self.W2 = nn.Parameter(torch.empty(4, OUTPUT_LAYOUT.size, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(4, OUTPUT_LAYOUT.size))
        self.dropout = nn.Dropout(dropout_probability)
        self.activation = nn.ELU()
        for weight in (self.W0, self.W1, self.W2):
            for bank in weight:
                nn.init.xavier_uniform_(bank)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != INPUT_LAYOUT.size or phase.shape != (len(x),):
            raise ValueError("PFNN expects x[B,288] and phase[B]")
        h0 = self.activation(_batched_linear(x, catmull_rom_phase_banks(self.W0, phase), catmull_rom_phase_banks(self.b0, phase)))
        h1 = self.activation(_batched_linear(self.dropout(h0), catmull_rom_phase_banks(self.W1, phase), catmull_rom_phase_banks(self.b1, phase)))
        return _batched_linear(self.dropout(h1), catmull_rom_phase_banks(self.W2, phase), catmull_rom_phase_banks(self.b2, phase))
