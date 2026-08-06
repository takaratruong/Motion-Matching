"""One deterministic smooth hill shared by conditioning and visualization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_HILL_SLOPE_DEGREES = 18.0
DEFAULT_HILL_DIAMETER_M = 7.0
DEFAULT_HILL_HEIGHT_M = (
    math.tan(math.radians(DEFAULT_HILL_SLOPE_DEGREES))
    * DEFAULT_HILL_DIAMETER_M
    / math.pi
)


@dataclass(frozen=True)
class GentleHillProfile:
    """Local radial cosine mound surrounded by flat ground."""

    domain_x: tuple[float, float] = (-5.0, 15.0)
    half_width: float = 8.0
    hill_start_x: float = 1.5
    hill_length: float = DEFAULT_HILL_DIAMETER_M
    height_m: float = DEFAULT_HILL_HEIGHT_M

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
        """Return exact terrain Z for the mound on an otherwise flat plane."""

        point = np.asarray(xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("terrain XY must contain two finite values")
        x, y = (float(value) for value in point)
        radius = 0.5 * self.hill_length
        center_x = self.hill_start_x + radius
        radial_distance = math.hypot(x - center_x, y)
        if radial_distance >= radius:
            return 0.0
        return 0.5 * self.height_m * (
            1.0 + math.cos(math.pi * radial_distance / radius)
        )

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
        """Return a rectangular grid mesh whose vertices obey :meth:`height`."""

        if not isinstance(sample_count, int) or sample_count < 2:
            raise ValueError("sample_count must be an integer of at least two")
        span_x = self.domain_x[1] - self.domain_x[0]
        span_y = 2.0 * self.half_width
        y_intervals = max(
            1, int(round((sample_count - 1) * span_y / span_x))
        )
        y_count = y_intervals + 1
        x_values = np.linspace(
            self.domain_x[0], self.domain_x[1], sample_count, dtype=np.float64
        )
        y_values = np.linspace(
            -self.half_width, self.half_width, y_count, dtype=np.float64
        )
        x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="ij")
        z_grid = np.asarray(
            [
                self.height((float(x), float(y)))
                for x, y in zip(x_grid.ravel(), y_grid.ravel())
            ],
            dtype=np.float64,
        ).reshape(x_grid.shape)
        vertices = np.column_stack(
            (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
        )

        faces = np.empty(
            ((sample_count - 1) * (y_count - 1) * 2, 3), dtype=np.int32
        )
        face_index = 0
        for x_index in range(sample_count - 1):
            for y_index in range(y_count - 1):
                lower_left = x_index * y_count + y_index
                lower_right = (x_index + 1) * y_count + y_index
                faces[face_index] = (
                    lower_left,
                    lower_right,
                    lower_right + 1,
                )
                faces[face_index + 1] = (
                    lower_left,
                    lower_right + 1,
                    lower_left + 1,
                )
                face_index += 2
        return vertices, faces
