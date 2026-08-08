"""One authoritative mesh/query map containing three smooth test hills."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HillSegment:
    """Longitudinal bounds and requested grade for one symmetric hill."""

    start_x: float
    end_x: float
    grade_deg: float


class TerrainPFNNHillMap:
    """C1 10, 15 and 18.9 degree hills shared by control and rendering.

    The sampled triangle mesh is authoritative at runtime.  The analytic
    profile is retained only for construction checks and diagnostics.
    """

    requested_grades_deg = (10.0, 15.0, 18.9)

    def __init__(
        self,
        *,
        grid_spacing_m: float = 0.05,
        half_width_m: float = 2.0,
        flat_apron_m: float = 2.0,
        inter_hill_flat_m: float = 1.0,
        blend_m: float = 0.5,
        flank_m: float = 1.25,
        summit_m: float = 0.8,
    ) -> None:
        values = (
            grid_spacing_m,
            half_width_m,
            flat_apron_m,
            inter_hill_flat_m,
            blend_m,
            flank_m,
            summit_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("hill-map dimensions must be finite and positive")

        self.grid_spacing_m = float(grid_spacing_m)
        self.half_width_m = float(half_width_m)
        self.flat_apron_m = float(flat_apron_m)
        self.inter_hill_flat_m = float(inter_hill_flat_m)
        self.blend_m = float(blend_m)
        self.flank_m = float(flank_m)
        self.summit_m = float(summit_m)

        hill_length = 2.0 * blend_m + 2.0 * flank_m + summit_m
        cursor = 0.0
        hills: list[HillSegment] = []
        for grade in self.requested_grades_deg:
            hills.append(HillSegment(cursor, cursor + hill_length, grade))
            cursor += hill_length + inter_hill_flat_m
        self.hills = tuple(hills)
        self.x_min = -float(flat_apron_m)
        self.x_max = float(hills[-1].end_x + flat_apron_m)
        self.y_min = -float(half_width_m)
        self.y_max = float(half_width_m)

        x_count = int(math.ceil((self.x_max - self.x_min) / grid_spacing_m)) + 1
        y_count = int(math.ceil((self.y_max - self.y_min) / grid_spacing_m)) + 1
        self.x_coordinates = np.linspace(
            self.x_min, self.x_max, x_count, dtype=np.float64
        )
        self.y_coordinates = np.linspace(
            self.y_min, self.y_max, y_count, dtype=np.float64
        )
        heights = np.asarray(
            [self.analytic_height_at(float(x)) for x in self.x_coordinates],
            dtype=np.float64,
        )
        x_grid, y_grid = np.meshgrid(
            self.x_coordinates, self.y_coordinates, indexing="ij"
        )
        z_grid = np.broadcast_to(heights[:, None], x_grid.shape).copy()
        self._height_grid = z_grid
        self.vertices = np.column_stack(
            (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
        )

        faces = np.empty(((x_count - 1) * (y_count - 1) * 2, 3), dtype=np.int32)
        face_index = 0
        for x_index in range(x_count - 1):
            for y_index in range(y_count - 1):
                lower_left = x_index * y_count + y_index
                lower_right = (x_index + 1) * y_count + y_index
                faces[face_index] = (lower_left, lower_right, lower_right + 1)
                faces[face_index + 1] = (
                    lower_left,
                    lower_right + 1,
                    lower_left + 1,
                )
                face_index += 2
        self.faces = faces

    def _local_profile(self, local_x: float, grade_deg: float) -> tuple[float, float]:
        blend = self.blend_m
        flank = self.flank_m
        summit = self.summit_m
        slope = math.tan(math.radians(grade_deg))
        first_shoulder = 0.5 * slope * blend
        upper_shoulder = first_shoulder + slope * flank

        if local_x <= 0.0:
            return 0.0, 0.0
        if local_x < blend:
            s = local_x / blend
            return (
                slope * blend * (s**3 - 0.5 * s**4),
                slope * (3.0 * s**2 - 2.0 * s**3),
            )
        local_x -= blend
        if local_x < flank:
            return first_shoulder + slope * local_x, slope
        local_x -= flank
        if local_x < summit:
            s = local_x / summit
            return (
                upper_shoulder + slope * summit * math.sin(math.pi * s) / math.pi,
                slope * math.cos(math.pi * s),
            )
        local_x -= summit
        if local_x < flank:
            return upper_shoulder - slope * local_x, -slope
        local_x -= flank
        if local_x < blend:
            s = local_x / blend
            return (
                first_shoulder
                - slope * blend * (s - s**3 + 0.5 * s**4),
                -slope * (1.0 - 3.0 * s**2 + 2.0 * s**3),
            )
        return 0.0, 0.0

    def analytic_height_at(self, x: float) -> float:
        """Return the unsampled longitudinal construction profile."""

        if not math.isfinite(x):
            raise ValueError("terrain x must be finite")
        for hill in self.hills:
            if hill.start_x <= x <= hill.end_x:
                return self._local_profile(x - hill.start_x, hill.grade_deg)[0]
        return 0.0

    def analytic_slope_at(self, x: float) -> float:
        """Return dz/dx from the analytic construction profile."""

        if not math.isfinite(x):
            raise ValueError("terrain x must be finite")
        for hill in self.hills:
            if hill.start_x <= x <= hill.end_x:
                return self._local_profile(x - hill.start_x, hill.grade_deg)[1]
        return 0.0

    def height_at(self, xy: object) -> float | None:
        """Interpolate the exact rendered triangle beneath a world XY point."""

        point = np.asarray(xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("terrain XY must contain two finite values")
        x, y = (float(value) for value in point)
        tolerance = 1.0e-12
        if (
            x < self.x_min - tolerance
            or x > self.x_max + tolerance
            or y < self.y_min - tolerance
            or y > self.y_max + tolerance
        ):
            return None
        x = min(max(x, self.x_min), self.x_max)
        y = min(max(y, self.y_min), self.y_max)
        x_index = min(
            int(np.searchsorted(self.x_coordinates, x, side="right")) - 1,
            len(self.x_coordinates) - 2,
        )
        y_index = min(
            int(np.searchsorted(self.y_coordinates, y, side="right")) - 1,
            len(self.y_coordinates) - 2,
        )
        x_index = max(0, x_index)
        y_index = max(0, y_index)
        x0 = float(self.x_coordinates[x_index])
        x1 = float(self.x_coordinates[x_index + 1])
        y0 = float(self.y_coordinates[y_index])
        y1 = float(self.y_coordinates[y_index + 1])
        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)
        z00 = float(self._height_grid[x_index, y_index])
        z10 = float(self._height_grid[x_index + 1, y_index])
        z01 = float(self._height_grid[x_index, y_index + 1])
        z11 = float(self._height_grid[x_index + 1, y_index + 1])
        if tx >= ty:
            return z00 + tx * (z10 - z00) + ty * (z11 - z10)
        return z00 + tx * (z11 - z01) + ty * (z01 - z00)

    def grade_degrees_at(self, xy: object) -> float | None:
        """Return the signed grade of the authoritative mesh triangle."""

        point = np.asarray(xy, dtype=np.float64)
        height = self.height_at(point)
        if height is None:
            return None
        x = float(point[0])
        x_index = min(
            max(int(np.searchsorted(self.x_coordinates, x, side="right")) - 1, 0),
            len(self.x_coordinates) - 2,
        )
        dx = float(self.x_coordinates[x_index + 1] - self.x_coordinates[x_index])
        dz = float(
            self._height_grid[x_index + 1, 0] - self._height_grid[x_index, 0]
        )
        return math.degrees(math.atan2(dz, dx))
