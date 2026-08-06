"""One deterministic smooth hill shared by conditioning and visualization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GentleHillProfile:
    """Wide cosine hill with flat approach and exit aprons."""

    domain_x: tuple[float, float] = (-3.0, 12.0)
    half_width: float = 3.0
    hill_start_x: float = 1.5
    hill_length: float = 7.0
    height_m: float = 0.35

    def __post_init__(self) -> None:
        start, end = self.domain_x
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and start < self.hill_start_x
            and self.hill_start_x + self.hill_length < end
        ):
            raise ValueError("hill and aprons must lie inside an ordered domain")
        if (
            not math.isfinite(self.half_width)
            or self.half_width <= 0.0
            or not math.isfinite(self.hill_length)
            or self.hill_length <= 0.0
            or not math.isfinite(self.height_m)
            or self.height_m <= 0.0
        ):
            raise ValueError("hill dimensions must be finite and positive")

    def height(self, xy: object) -> float:
        """Return exact terrain Z for one certified world XY query."""

        point = np.asarray(xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("terrain XY must contain two finite values")
        x, y = (float(value) for value in point)
        if not (
            self.domain_x[0] <= x <= self.domain_x[1]
            and abs(y) <= self.half_width
        ):
            raise ValueError("terrain XY is outside the certified hill domain")
        offset = x - self.hill_start_x
        if offset <= 0.0 or offset >= self.hill_length:
            return 0.0
        phase = offset / self.hill_length
        return 0.5 * self.height_m * (1.0 - math.cos(2.0 * math.pi * phase))

    @property
    def max_slope_degrees(self) -> float:
        """Maximum absolute longitudinal tangent angle."""

        maximum_derivative = self.height_m * math.pi / self.hill_length
        return math.degrees(math.atan(maximum_derivative))

    def profile_vertices(self, sample_count: int = 151) -> np.ndarray:
        """Return ordered longitudinal ``(x, z)`` profile samples."""

        if not isinstance(sample_count, int) or sample_count < 2:
            raise ValueError("sample_count must be an integer of at least two")
        x_values = np.linspace(
            self.domain_x[0], self.domain_x[1], sample_count, dtype=np.float64
        )
        z_values = np.asarray(
            [self.height((float(x), 0.0)) for x in x_values], dtype=np.float64
        )
        return np.column_stack((x_values, z_values))

    def mesh(self, sample_count: int = 151) -> tuple[np.ndarray, np.ndarray]:
        """Return a two-sided strip mesh whose vertices obey :meth:`height`."""

        profile = self.profile_vertices(sample_count)
        vertices = np.empty((sample_count * 2, 3), dtype=np.float64)
        vertices[0::2, 0] = profile[:, 0]
        vertices[1::2, 0] = profile[:, 0]
        vertices[0::2, 1] = -self.half_width
        vertices[1::2, 1] = self.half_width
        vertices[0::2, 2] = profile[:, 1]
        vertices[1::2, 2] = profile[:, 1]

        faces = np.empty(((sample_count - 1) * 2, 3), dtype=np.int32)
        for index in range(sample_count - 1):
            left = 2 * index
            right = left + 2
            faces[2 * index] = (left, right, right + 1)
            faces[2 * index + 1] = (left, right + 1, left + 1)
        return vertices, faces
