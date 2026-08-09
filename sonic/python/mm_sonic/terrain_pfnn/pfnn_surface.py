"""Exact released-PFNN terrain placement in the native G1 world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mm_sonic.pfnn_terrain_fit import (
    PFNNTerrainFit,
    load_terrain_fit,
    terrain_height_g1,
)


PFNN_G1_SCALE = 0.875
PFNN_G1_Z_OFFSET_M = 0.05224985936713168


@dataclass(frozen=True)
class PlacedPFNNSurface:
    """A rank-zero PFNN fit under the frozen source-to-G1 transform."""

    fit: PFNNTerrainFit
    scale: float = PFNN_G1_SCALE
    z_offset: float = PFNN_G1_Z_OFFSET_M

    def __post_init__(self) -> None:
        if not isinstance(self.fit, PFNNTerrainFit):
            raise TypeError("fit must be a PFNNTerrainFit")
        if float(self.scale) != PFNN_G1_SCALE:
            raise ValueError("PFNN terrain scale is frozen for the G1 corpus")
        if float(self.z_offset) != PFNN_G1_Z_OFFSET_M:
            raise ValueError("PFNN terrain Z offset is frozen for the G1 corpus")

    def height_at(self, xy: object) -> np.ndarray:
        query = np.asarray(xy)
        if query.ndim < 1 or query.shape[-1] != 2:
            raise ValueError("PFNN terrain query must end in two coordinates")
        leading_shape = query.shape[:-1]
        height = terrain_height_g1(
            self.fit,
            query.reshape(-1, 2),
            scale=self.scale,
            z_offset=self.z_offset,
        )
        return height.reshape(leading_shape)

    def gradient_at(self, xy: object, epsilon: float = 1.0e-4) -> np.ndarray:
        query = np.asarray(xy, dtype=np.float64)
        if query.ndim < 1 or query.shape[-1] != 2:
            raise ValueError("PFNN terrain query must end in two coordinates")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("gradient epsilon must be positive and finite")
        x_offset = np.array((float(epsilon), 0.0), dtype=np.float64)
        y_offset = np.array((0.0, float(epsilon)), dtype=np.float64)
        dx = (
            self.height_at(query + x_offset) - self.height_at(query - x_offset)
        ) / (2.0 * float(epsilon))
        dy = (
            self.height_at(query + y_offset) - self.height_at(query - y_offset)
        ) / (2.0 * float(epsilon))
        return np.stack((dx, dy), axis=-1)


def load_placed_pfnn_surface(path: Path) -> PlacedPFNNSurface:
    """Load a safe numeric PFNN fit under the frozen G1 placement."""

    return PlacedPFNNSurface(load_terrain_fit(Path(path)))


__all__ = [
    "PFNN_G1_SCALE",
    "PFNN_G1_Z_OFFSET_M",
    "PlacedPFNNSurface",
    "load_placed_pfnn_surface",
]
