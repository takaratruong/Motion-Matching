"""Native Z-up GRAIL object surfaces and deterministic height grids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from resources.g1_terrain_builder.terrain import triangulate_faces

from .corpus import PinnedStairSource


_PROJECTED_EPSILON = 1e-12
_BARYCENTRIC_TOLERANCE = 1e-9


def _quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("object quaternion must be finite and nonzero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        np.float64,
    )


def transform_object_vertices(
    vertices: np.ndarray,
    *,
    position_world: np.ndarray,
    quaternion_world_xyzw: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, np.float64)
    position = np.asarray(position_world, np.float64)
    quaternion = np.asarray(quaternion_world_xyzw, np.float64)
    scale = np.asarray(scale, np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("object vertices must have non-empty shape (N, 3)")
    if position.shape != (3,) or quaternion.shape != (4,) or scale.shape != (3,):
        raise ValueError("object transform shapes are invalid")
    if not (
        np.isfinite(vertices).all()
        and np.isfinite(position).all()
        and np.isfinite(quaternion).all()
        and np.isfinite(scale).all()
        and np.all(scale > 0.0)
    ):
        raise ValueError("object transform values must be finite with positive scale")
    rotation = _quaternion_matrix_xyzw(quaternion)
    return (rotation @ (vertices * scale).T).T + position


class ZUpTriangleSurface:
    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        *,
        exterior_height_z: float = 0.0,
    ) -> None:
        vertices = np.asarray(vertices, np.float64)
        triangles = np.asarray(triangles)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("surface vertices must be finite non-empty (N, 3)")
        if (
            triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or not len(triangles)
            or triangles.dtype.kind not in "iu"
        ):
            raise ValueError("surface triangle indices must have shape (M, 3)")
        triangles = np.asarray(triangles, np.int64)
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("surface triangle index is outside the vertex array")
        if not math.isfinite(float(exterior_height_z)):
            raise ValueError("surface exterior height must be finite")
        triangle_vertices = vertices[triangles]
        projected = triangle_vertices[..., :2]
        self.vertices = vertices.copy()
        self.triangles = triangles.copy()
        self.exterior_height_z = float(exterior_height_z)
        self._triangle_vertices = triangle_vertices.copy()
        self._minimum_xy = projected.min(axis=1)
        self._maximum_xy = projected.max(axis=1)
        for value in (
            self.vertices,
            self.triangles,
            self._triangle_vertices,
            self._minimum_xy,
            self._maximum_xy,
        ):
            value.setflags(write=False)

    def height_xy(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, np.float64)
        if points.ndim == 1:
            if points.shape != (2,):
                raise ValueError("surface query point must have shape (2,)")
            points = points[None, :]
            scalar = True
        else:
            if points.ndim != 2 or points.shape[1:] != (2,):
                raise ValueError("surface query points must have shape (N, 2)")
            scalar = False
        if not np.isfinite(points).all():
            raise ValueError("surface query points must be finite")

        output = np.full(len(points), self.exterior_height_z, np.float64)
        for row, point in enumerate(points):
            candidates = np.flatnonzero(
                (self._minimum_xy[:, 0] - _BARYCENTRIC_TOLERANCE <= point[0])
                & (point[0] <= self._maximum_xy[:, 0] + _BARYCENTRIC_TOLERANCE)
                & (self._minimum_xy[:, 1] - _BARYCENTRIC_TOLERANCE <= point[1])
                & (point[1] <= self._maximum_xy[:, 1] + _BARYCENTRIC_TOLERANCE)
            )
            if not len(candidates):
                continue
            triangle = self._triangle_vertices[candidates]
            a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
            v0 = b[:, :2] - a[:, :2]
            v1 = c[:, :2] - a[:, :2]
            p = point - a[:, :2]
            determinant = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
            projected = np.abs(determinant) > _PROJECTED_EPSILON
            u = np.zeros_like(determinant)
            v = np.zeros_like(determinant)
            u[projected] = (
                p[projected, 0] * v1[projected, 1]
                - p[projected, 1] * v1[projected, 0]
            ) / determinant[projected]
            v[projected] = (
                v0[projected, 0] * p[projected, 1]
                - v0[projected, 1] * p[projected, 0]
            ) / determinant[projected]
            w = 1.0 - u - v
            inside = (
                projected
                & (u >= -_BARYCENTRIC_TOLERANCE)
                & (v >= -_BARYCENTRIC_TOLERANCE)
                & (w >= -_BARYCENTRIC_TOLERANCE)
            )
            if np.any(inside):
                height = (
                    w[inside] * a[inside, 2]
                    + u[inside] * b[inside, 2]
                    + v[inside] * c[inside, 2]
                )
                output[row] = max(self.exterior_height_z, float(height.max()))
        return output[0] if scalar else output


@dataclass(frozen=True)
class ZUpHeightGrid:
    origin_xy: np.ndarray
    cell_size_m: float
    height_z: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_xy, np.float32)
        height = np.asarray(self.height_z, np.float32)
        cell = float(self.cell_size_m)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("height grid origin must be finite shape (2,)")
        if not math.isfinite(cell) or cell <= 0.0:
            raise ValueError("height grid cell size must be finite and positive")
        if (
            height.ndim != 2
            or min(height.shape) < 2
            or not np.isfinite(height).all()
        ):
            raise ValueError("height grid values must be finite shape (ny>=2, nx>=2)")
        origin = np.ascontiguousarray(origin)
        height = np.ascontiguousarray(height)
        origin.setflags(write=False)
        height.setflags(write=False)
        object.__setattr__(self, "origin_xy", origin)
        object.__setattr__(self, "cell_size_m", cell)
        object.__setattr__(self, "height_z", height)

    @property
    def maximum_xy(self) -> np.ndarray:
        ny, nx = self.height_z.shape
        return self.origin_xy.astype(np.float64) + self.cell_size_m * np.array(
            [nx - 1, ny - 1], np.float64
        )

    def sample_xy(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy)
        if points.shape == (2,):
            flat = np.asarray(points, np.float64)[None, :]
            output_shape: tuple[int, ...] = ()
        elif points.ndim >= 2 and points.shape[-1] == 2:
            flat = np.asarray(points, np.float64).reshape(-1, 2)
            output_shape = points.shape[:-1]
        else:
            raise ValueError("height grid query points must end in shape (2,)")
        if not np.isfinite(flat).all():
            raise ValueError("height grid query points must be finite")
        coordinate = (
            flat - self.origin_xy.astype(np.float64)
        ) / self.cell_size_m
        ny, nx = self.height_z.shape
        tolerance = 1e-7
        outside = (
            (coordinate[:, 0] < -tolerance)
            | (coordinate[:, 1] < -tolerance)
            | (coordinate[:, 0] > nx - 1 + tolerance)
            | (coordinate[:, 1] > ny - 1 + tolerance)
        )
        if np.any(outside):
            raise ValueError("height grid query is outside the authoritative domain")
        coordinate[:, 0] = np.clip(coordinate[:, 0], 0.0, nx - 1)
        coordinate[:, 1] = np.clip(coordinate[:, 1], 0.0, ny - 1)
        ix = np.minimum(np.floor(coordinate[:, 0]).astype(np.int64), nx - 2)
        iy = np.minimum(np.floor(coordinate[:, 1]).astype(np.int64), ny - 2)
        tx = coordinate[:, 0] - ix
        ty = coordinate[:, 1] - iy
        h00 = self.height_z[iy, ix].astype(np.float64)
        h10 = self.height_z[iy, ix + 1].astype(np.float64)
        h01 = self.height_z[iy + 1, ix].astype(np.float64)
        h11 = self.height_z[iy + 1, ix + 1].astype(np.float64)
        first = h00 + tx * (h10 - h00) + ty * (h11 - h10)
        second = h00 + tx * (h11 - h01) + ty * (h01 - h00)
        values = np.where(tx >= ty, first, second).astype(np.float32)
        return values.reshape(output_shape)

    def as_npz_fields(self) -> dict[str, np.ndarray]:
        return {
            "origin_xy": self.origin_xy,
            "cell_size_m": np.array([self.cell_size_m], np.float32),
            "height_z": self.height_z,
        }

    @staticmethod
    def load(path: str | Path) -> "ZUpHeightGrid":
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"origin_xy", "cell_size_m", "height_z"}:
                    raise ValueError("height grid archive fields are invalid")
                cell = np.asarray(data["cell_size_m"])
                if cell.size != 1:
                    raise ValueError("height grid cell size must contain one value")
                return ZUpHeightGrid(
                    origin_xy=np.asarray(data["origin_xy"], np.float32),
                    cell_size_m=float(cell.reshape(-1)[0]),
                    height_z=np.asarray(data["height_z"], np.float32),
                )
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"cannot load height grid: {path}") from error


def rasterize_zup_surface(
    surface: ZUpTriangleSurface,
    *,
    bounds_xy: tuple[float, float, float, float],
    cell_size_m: float = 0.02,
) -> ZUpHeightGrid:
    if not isinstance(surface, ZUpTriangleSurface):
        raise TypeError("surface must be a ZUpTriangleSurface")
    try:
        minimum_x, maximum_x, minimum_y, maximum_y = (
            float(value) for value in bounds_xy
        )
    except (TypeError, ValueError) as error:
        raise ValueError("height grid bounds must contain four numbers") from error
    if not np.isfinite(bounds_xy).all() or maximum_x <= minimum_x or maximum_y <= minimum_y:
        raise ValueError("height grid bounds must be finite and increasing")
    cell = float(np.float32(cell_size_m))
    if not math.isfinite(cell) or cell <= 0.0:
        raise ValueError("height grid cell size must be finite and positive")
    origin = np.array([minimum_x, minimum_y], np.float32)
    nx = int(math.ceil((maximum_x - float(origin[0])) / cell)) + 1
    ny = int(math.ceil((maximum_y - float(origin[1])) / cell)) + 1
    if nx < 2 or ny < 2 or nx * ny > 50_000_000:
        raise ValueError("height grid dimensions are invalid")
    xs = float(origin[0]) + np.arange(nx, dtype=np.float64) * cell
    ys = float(origin[1]) + np.arange(ny, dtype=np.float64) * cell
    heights = np.empty((ny, nx), np.float32)
    for row, y in enumerate(ys):
        points = np.column_stack((xs, np.full(nx, y)))
        heights[row] = surface.height_xy(points).astype(np.float32)
    return ZUpHeightGrid(origin, cell, heights)


def load_source_surface(source: PinnedStairSource) -> ZUpTriangleSurface:
    if not isinstance(source, PinnedStairSource):
        raise TypeError("source must be a PinnedStairSource")
    stage = Usd.Stage.Open(str(source.usd_path))
    if stage is None:
        raise ValueError(f"{source.base}: unable to open USD")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int32)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32)
            cache = UsdGeom.XformCache()
            transform = cache.GetLocalToWorldTransform(prim)
            stage_vertices = np.array(
                [
                    transform.Transform(Gf.Vec3d(*[float(v) for v in point]))
                    for point in points
                ],
                np.float64,
            )
            world = transform_object_vertices(
                stage_vertices,
                position_world=source.object_position_world,
                quaternion_world_xyzw=source.object_quaternion_world_xyzw,
                scale=source.object_scale,
            )
            triangles = triangulate_faces(world, counts, indices)
            return ZUpTriangleSurface(world, triangles, exterior_height_z=0.0)
    raise ValueError(f"{source.base}: USD contains no mesh")


def build_source_height_grid(
    source: PinnedStairSource,
    *,
    cell_size_m: float = 0.02,
    margin_m: float = 2.0,
) -> ZUpHeightGrid:
    surface = load_source_surface(source)
    root_xy = np.asarray(source.robot_qpos_mujoco[:, :2], np.float64)
    minimum = np.minimum(surface.vertices[:, :2].min(axis=0), root_xy.min(axis=0))
    maximum = np.maximum(surface.vertices[:, :2].max(axis=0), root_xy.max(axis=0))
    minimum -= margin_m
    maximum += margin_m
    return rasterize_zup_surface(
        surface,
        bounds_xy=(
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
        ),
        cell_size_m=cell_size_m,
    )
