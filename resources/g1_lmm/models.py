"""Exact Orange Duck learned-motion-matching network architectures."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .dataset import G1LmmDimensions


class Compressor(nn.Module):
    """Training-only ``908 -> 512 -> 512 -> 512 -> 32`` encoder."""

    def __init__(self, dimensions: G1LmmDimensions) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimensions.compressor_input, 512),
                nn.Linear(512, 512),
                nn.Linear(512, 512),
                nn.Linear(512, dimensions.latent),
            ]
        )
        self.linear0, self.linear1, self.linear2, self.linear3 = self.layers

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.elu(self.linear0(values))
        values = F.elu(self.linear1(values))
        values = F.elu(self.linear2(values))
        return self.linear3(values)


class Decompressor(nn.Module):
    """Runtime ``63 -> 512 -> 458`` pose decompressor."""

    def __init__(self, dimensions: G1LmmDimensions) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimensions.state, 512),
                nn.Linear(512, dimensions.decompressor_output),
            ]
        )
        self.linear0, self.linear1 = self.layers

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear1(F.relu(self.linear0(values)))


class Stepper(nn.Module):
    """Runtime ``63 -> 512 -> 512 -> 63`` recurrent derivative model."""

    def __init__(self, dimensions: G1LmmDimensions) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimensions.state, 512),
                nn.Linear(512, 512),
                nn.Linear(512, dimensions.state),
            ]
        )
        self.linear0, self.linear1, self.linear2 = self.layers

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.linear0(values))
        values = F.relu(self.linear1(values))
        return self.linear2(values)


class Projector(nn.Module):
    """Runtime ``31 -> 512 -> 512 -> 512 -> 512 -> 63`` projector."""

    def __init__(self, dimensions: G1LmmDimensions) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(dimensions.features, 512),
                nn.Linear(512, 512),
                nn.Linear(512, 512),
                nn.Linear(512, 512),
                nn.Linear(512, dimensions.state),
            ]
        )
        self.linear0, self.linear1, self.linear2, self.linear3, self.linear4 = self.layers

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.relu(self.linear0(values))
        values = F.relu(self.linear1(values))
        values = F.relu(self.linear2(values))
        values = F.relu(self.linear3(values))
        return self.linear4(values)
