"""Exact canonical-mesh queries and practical flat/tread support extraction.

This first global-information bootstrap deliberately covers the geometry we can
use immediately: flat surfaces and stair treads represented by canonical
triangles.  It has no USD or simulator dependency.  Support candidates are
tested with the complete sole polygon, not merely the ankle or sole centre.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from mm_sonic.joints import ContractError

from .canonical import CanonicalTerrainMesh, TerrainBinding
from .contact import CanonicalMeshQuery
from .math3d import RigidTransform


_EPSILON = 1.0e-10


def _readonly(value: object, dtype: np.dtype | type = np.float64) -> np.ndarray:
    output = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    output.flags.writeable = False
    return output


def _identity_transform() -> RigidTransform:
    return RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )


def _signed_polygon_area(polygon: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )


def _convex_hull(points: object) -> np.ndarray:
    """Return the counter-clockwise hull of finite 2D points."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or not np.isfinite(values).all():
        raise ContractError("polygon points must be finite with shape [N,2]")
    unique = sorted({(float(point[0]), float(point[1])) for point in values})
    if len(unique) < 3:
        raise ContractError("support polygon needs at least three distinct points")

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= _EPSILON:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= _EPSILON:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3 or _signed_polygon_area(hull) <= _EPSILON:
        raise ContractError("support polygon has no positive area")
    return hull


def _contains_convex(polygon: np.ndarray, points: object, tolerance: float) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64)
    scalar = query.shape == (2,)
    if scalar:
        query = query[None, :]
    if query.ndim != 2 or query.shape[1:] != (2,) or not np.isfinite(query).all():
        raise ContractError("candidate points must be finite with shape [N,2]")
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = query[:, None, :] - polygon[None, :, :]
    cross = (
        edges[None, :, 0] * relative[:, :, 1]
        - edges[None, :, 1] * relative[:, :, 0]
    )
    result = np.all(cross >= -float(tolerance), axis=1)
    return result[0] if scalar else result


@dataclass(frozen=True)
class SupportConfig:
    """Thresholds for grouping upward, connected, coplanar triangles."""

    max_slope_rad: float = 0.20
    coplanar_tolerance_m: float = 1.0e-4
    normal_tolerance_rad: float = 1.0e-3
    minimum_area_m2: float = 1.0e-5

    def __post_init__(self) -> None:
        values = (
            self.max_slope_rad,
            self.coplanar_tolerance_m,
            self.normal_tolerance_rad,
            self.minimum_area_m2,
        )
        if not all(np.isfinite(value) for value in values):
            raise ContractError("support thresholds must be finite")
        if (
            not 0.0 <= self.max_slope_rad < math.pi / 2.0
            or self.coplanar_tolerance_m <= 0.0
            or self.normal_tolerance_rad < 0.0
            or self.minimum_area_m2 < 0.0
        ):
            raise ContractError("support thresholds are outside their useful range")


@dataclass(frozen=True)
class SupportPatch:
    """One connected horizontal support surface in world coordinates.

    ``polygon_xy`` is the exact convex boundary used by the current flat/stair
    bootstrap.  ``world_from_patch`` places its XY plane at ``height_m``.
    """

    patch_id: str
    world_from_patch: RigidTransform
    polygon_xy: np.ndarray
    normal_world: np.ndarray
    height_m: float
    face_indices: np.ndarray

    def __post_init__(self) -> None:
        if type(self.patch_id) is not str or not self.patch_id:
            raise ContractError("patch_id must be nonempty")
        polygon = _convex_hull(self.polygon_xy)
        normal = np.asarray(self.normal_world, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ContractError("normal_world must be a finite 3-vector")
        norm = float(np.linalg.norm(normal))
        if norm <= _EPSILON:
            raise ContractError("normal_world must be nonzero")
        normal /= norm
        faces = np.asarray(self.face_indices, dtype=np.int32)
        if faces.ndim != 1:
            raise ContractError("face_indices must be one-dimensional")
        if not np.isfinite(self.height_m):
            raise ContractError("height_m must be finite")
        object.__setattr__(self, "polygon_xy", _readonly(polygon))
        object.__setattr__(self, "normal_world", _readonly(normal))
        object.__setattr__(self, "face_indices", _readonly(faces, np.int32))

    @property
    def area_m2(self) -> float:
        return _signed_polygon_area(self.polygon_xy)

    def contains_point(self, point_xy: object, tolerance_m: float = 1.0e-9) -> bool:
        return bool(_contains_convex(self.polygon_xy, point_xy, tolerance_m))

    def contains_sole(
        self,
        centre_xy: object,
        sole_polygon_xy: object,
        *,
        yaw_rad: float = 0.0,
        tolerance_m: float = 1.0e-9,
    ) -> bool:
        """Return whether the complete translated/rotated sole fits the patch."""

        centre = np.asarray(centre_xy, dtype=np.float64)
        sole = np.asarray(sole_polygon_xy, dtype=np.float64)
        if centre.shape != (2,) or sole.ndim != 2 or sole.shape[1:] != (2,):
            raise ContractError("sole containment needs centre [2] and polygon [N,2]")
        if not np.isfinite(centre).all() or not np.isfinite(sole).all():
            raise ContractError("sole candidate must be finite")
        cosine = math.cos(float(yaw_rad))
        sine = math.sin(float(yaw_rad))
        rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
        corners = centre + sole @ rotation.T
        return bool(np.all(_contains_convex(self.polygon_xy, corners, tolerance_m)))


@dataclass(frozen=True)
class ErodedSupportPatch:
    """Legal sole-centre region after complete-footprint erosion."""

    source_patch_id: str
    polygon_xy: np.ndarray
    height_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "polygon_xy", _readonly(_convex_hull(self.polygon_xy)))

    @property
    def area_m2(self) -> float:
        return _signed_polygon_area(self.polygon_xy)

    def contains_point(self, point_xy: object, tolerance_m: float = 1.0e-9) -> bool:
        return bool(_contains_convex(self.polygon_xy, point_xy, tolerance_m))


@dataclass(frozen=True)
class RayHit:
    position_world: np.ndarray
    normal_world: np.ndarray
    distance_m: float
    face_index: int


class TerrainMeshIndex:
    """World-space index over the valid triangles of a canonical mesh."""

    def __init__(
        self,
        mesh: CanonicalTerrainMesh,
        binding: TerrainBinding | RigidTransform,
    ) -> None:
        if not isinstance(mesh, CanonicalTerrainMesh):
            raise ContractError("mesh must be a CanonicalTerrainMesh")
        if isinstance(binding, TerrainBinding):
            transform = binding.world_from_terrain
            terrain_binding: TerrainBinding | None = binding
        elif isinstance(binding, RigidTransform):
            transform = binding
            terrain_binding = None
        else:
            raise ContractError("binding must be TerrainBinding or RigidTransform")

        vertices_world = np.asarray(
            transform.apply_points(mesh.vertices_local), dtype=np.float64
        )
        valid_indices = np.flatnonzero(mesh.valid_faces)
        faces = np.asarray(mesh.faces[valid_indices], dtype=np.int32)
        triangles = vertices_world[faces] if len(faces) else np.empty((0, 3, 3))
        raw_normal = (
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            if len(triangles)
            else np.empty((0, 3))
        )
        length = np.linalg.norm(raw_normal, axis=1)
        nondegenerate = length > _EPSILON

        self.mesh = mesh
        self.binding = terrain_binding
        self.world_from_terrain = transform
        self.vertices_world = _readonly(vertices_world)
        self.face_indices = _readonly(valid_indices[nondegenerate], np.int32)
        self.faces = _readonly(faces[nondegenerate], np.int32)
        self.triangles_world = _readonly(triangles[nondegenerate])
        self.normals_world = _readonly(
            raw_normal[nondegenerate] / length[nondegenerate, None]
            if np.any(nondegenerate)
            else np.empty((0, 3))
        )

    @classmethod
    def from_canonical(
        cls,
        mesh: CanonicalTerrainMesh,
        binding: TerrainBinding | RigidTransform,
    ) -> "TerrainMeshIndex":
        return cls(mesh, binding)

    @property
    def is_watertight_for_collision(self) -> bool:
        if len(self.faces) == 0:
            return False
        counts: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for left, right in (
                (int(face[0]), int(face[1])),
                (int(face[1]), int(face[2])),
                (int(face[2]), int(face[0])),
            ):
                edge = (min(left, right), max(left, right))
                counts[edge] = counts.get(edge, 0) + 1
        return bool(counts) and all(count == 2 for count in counts.values())

    def raycast(self, origin: object, direction: object) -> RayHit | None:
        """Return the nearest deterministic Möller–Trumbore ray hit."""

        ray_origin = np.asarray(origin, dtype=np.float64)
        ray_direction = np.asarray(direction, dtype=np.float64)
        if ray_origin.shape != (3,) or ray_direction.shape != (3,):
            raise ContractError("ray origin and direction must be 3-vectors")
        if not np.isfinite(ray_origin).all() or not np.isfinite(ray_direction).all():
            raise ContractError("ray origin and direction must be finite")
        norm = float(np.linalg.norm(ray_direction))
        if norm <= _EPSILON:
            raise ContractError("ray direction must be nonzero")
        direction_unit = ray_direction / norm
        if len(self.triangles_world) == 0:
            return None

        edge1 = self.triangles_world[:, 1] - self.triangles_world[:, 0]
        edge2 = self.triangles_world[:, 2] - self.triangles_world[:, 0]
        h = np.cross(np.broadcast_to(direction_unit, edge2.shape), edge2)
        determinant = np.sum(edge1 * h, axis=1)
        valid = np.abs(determinant) > _EPSILON
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        relative = ray_origin - self.triangles_world[:, 0]
        bary_u = inverse * np.sum(relative * h, axis=1)
        q = np.cross(relative, edge1)
        bary_v = inverse * np.sum(direction_unit * q, axis=1)
        distance = inverse * np.sum(edge2 * q, axis=1)
        hits = (
            valid
            & (bary_u >= -_EPSILON)
            & (bary_v >= -_EPSILON)
            & (bary_u + bary_v <= 1.0 + _EPSILON)
            & (distance >= -_EPSILON)
        )
        if not np.any(hits):
            return None
        candidates = np.flatnonzero(hits)
        order = np.lexsort((self.face_indices[candidates], distance[candidates]))
        index = int(candidates[order[0]])
        hit_distance = max(0.0, float(distance[index]))
        return RayHit(
            position_world=_readonly(ray_origin + hit_distance * direction_unit),
            normal_world=_readonly(self.normals_world[index]),
            distance_m=hit_distance,
            face_index=int(self.face_indices[index]),
        )

    def signed_distance(self, points: object) -> np.ndarray:
        """Return oriented closest-surface distance.

        The sign follows the closest triangle normal.  This is useful for the
        open flat/stair bootstrap; a later collision audit remains responsible
        for inside/outside sign on arbitrary watertight bodies.
        """

        query_points = np.asarray(points, dtype=np.float64)
        scalar = query_points.shape == (3,)
        if scalar:
            query_points = query_points[None, :]
        if query_points.ndim != 2 or query_points.shape[1:] != (3,):
            raise ContractError("signed-distance points must have shape [N,3]")
        if len(self.triangles_world) == 0:
            result = np.full(len(query_points), np.inf, dtype=np.float64)
        else:
            result_query = CanonicalMeshQuery(
                self.mesh, self.world_from_terrain
            ).query(query_points)
            offset = query_points - result_query.closest_point_world
            result = np.sum(offset * result_query.surface_normal_world, axis=1)
        result.flags.writeable = False
        return result[0] if scalar else result


def _connected_support_components(
    index: TerrainMeshIndex, config: SupportConfig
) -> list[np.ndarray]:
    normals = np.asarray(index.normals_world)
    triangles = np.asarray(index.triangles_world)
    if len(triangles) == 0:
        return []
    # Triangle winding is not an input to support semantics.  A horizontal
    # tread remains a support whether its source face was wound up or down.
    oriented = normals.copy()
    oriented[oriented[:, 2] < 0.0] *= -1.0
    support = oriented[:, 2] >= math.cos(config.max_slope_rad)
    candidates = np.flatnonzero(support)
    if len(candidates) == 0:
        return []

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_local in candidates:
        face = index.faces[face_local]
        for left, right in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (min(left, right), max(left, right))
            edge_faces.setdefault(edge, []).append(int(face_local))

    adjacency: dict[int, set[int]] = {int(value): set() for value in candidates}
    minimum_dot = math.cos(config.normal_tolerance_rad)
    for neighbours in edge_faces.values():
        if len(neighbours) < 2:
            continue
        for left_index in range(len(neighbours)):
            for right_index in range(left_index + 1, len(neighbours)):
                left = neighbours[left_index]
                right = neighbours[right_index]
                if float(np.dot(oriented[left], oriented[right])) < minimum_dot:
                    continue
                plane_offset = abs(
                    float(
                        np.dot(
                            triangles[right, 0] - triangles[left, 0],
                            oriented[left],
                        )
                    )
                )
                if plane_offset <= config.coplanar_tolerance_m:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

    components: list[np.ndarray] = []
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        group: list[int] = []
        remaining.remove(seed)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        components.append(np.asarray(sorted(group), dtype=np.int32))
    return components


def extract_support_patches(
    mesh: TerrainMeshIndex | CanonicalTerrainMesh,
    config: SupportConfig | None = None,
    binding: TerrainBinding | RigidTransform | None = None,
) -> tuple[SupportPatch, ...]:
    """Extract connected flat/stair support patches from canonical triangles."""

    settings = config if config is not None else SupportConfig()
    if not isinstance(settings, SupportConfig):
        raise ContractError("config must be SupportConfig")
    if isinstance(mesh, TerrainMeshIndex):
        if binding is not None:
            raise ContractError("binding is already carried by TerrainMeshIndex")
        index = mesh
    elif isinstance(mesh, CanonicalTerrainMesh):
        index = TerrainMeshIndex.from_canonical(
            mesh, binding if binding is not None else _identity_transform()
        )
    else:
        raise ContractError("mesh must be CanonicalTerrainMesh or TerrainMeshIndex")

    raw: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for component in _connected_support_components(index, settings):
        triangle_vertices = index.triangles_world[component].reshape(-1, 3)
        polygon = _convex_hull(triangle_vertices[:, :2])
        area = _signed_polygon_area(polygon)
        if area < settings.minimum_area_m2:
            continue
        normals = np.array(index.normals_world[component], copy=True)
        normals[normals[:, 2] < 0.0] *= -1.0
        normal = np.mean(normals, axis=0)
        normal /= np.linalg.norm(normal)
        height = float(np.mean(triangle_vertices[:, 2]))
        centroid = np.mean(polygon, axis=0)
        raw.append(
            (
                height,
                float(centroid[0]) * 1.0e3 + float(centroid[1]),
                polygon,
                normal,
                index.face_indices[component],
            )
        )

    raw.sort(key=lambda value: (value[0], value[1]))
    patches: list[SupportPatch] = []
    for patch_index, (height, _, polygon, normal, face_indices) in enumerate(raw):
        patches.append(
            SupportPatch(
                patch_id=f"support-{patch_index:06d}",
                world_from_patch=RigidTransform(
                    translation_world=np.array((0.0, 0.0, height), dtype=np.float32),
                    quaternion_world_from_local_wxyz=np.array(
                        (1.0, 0.0, 0.0, 0.0), dtype=np.float32
                    ),
                ),
                polygon_xy=polygon,
                normal_world=normal,
                height_m=height,
                face_indices=face_indices,
            )
        )
    return tuple(patches)


def _clip_halfplane(
    polygon: np.ndarray, normal: np.ndarray, lower_bound: float
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(np.dot(normal, previous) - lower_bound)
    previous_inside = previous_value >= -_EPSILON
    for current in polygon:
        current_value = float(np.dot(normal, current) - lower_bound)
        current_inside = current_value >= -_EPSILON
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            fraction = previous_value / denominator
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(output, dtype=np.float64)


def erode_support_for_sole(
    patch: SupportPatch, sole_polygon_xy: object
) -> ErodedSupportPatch:
    """Erode a convex support patch by a reflected sole footprint.

    The resulting polygon contains exactly the legal translations of the sole
    centre for the supplied (unrotated) sole polygon.
    """

    if not isinstance(patch, SupportPatch):
        raise ContractError("patch must be a SupportPatch")
    sole = np.asarray(sole_polygon_xy, dtype=np.float64)
    if sole.ndim != 2 or sole.shape[1:] != (2,) or len(sole) < 3:
        raise ContractError("sole_polygon_xy must have shape [N>=3,2]")
    if not np.isfinite(sole).all():
        raise ContractError("sole_polygon_xy must be finite")

    eroded = np.array(patch.polygon_xy, copy=True)
    boundary = patch.polygon_xy
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0)):
        edge = end - start
        inward = np.array((-edge[1], edge[0]), dtype=np.float64)
        lower_bound = float(np.dot(inward, start) - np.min(sole @ inward))
        eroded = _clip_halfplane(eroded, inward, lower_bound)
        if len(eroded) < 3:
            raise ValueError(
                f"sole footprint does not fit support patch {patch.patch_id}"
            )
    return ErodedSupportPatch(
        source_patch_id=patch.patch_id,
        polygon_xy=eroded,
        height_m=patch.height_m,
    )


def mesh_from_heightfield(
    height: object,
    valid: object,
    *,
    spacing_m: float | tuple[float, float] = 1.0,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> CanonicalTerrainMesh:
    """Triangulate valid heightfield cells without inventing zero-height ground."""

    values = np.asarray(height, dtype=np.float64)
    validity = np.asarray(valid, dtype=np.bool_)
    if values.ndim != 2 or values.shape != validity.shape:
        raise ContractError("height and valid must have the same 2D shape")
    if not np.isfinite(values).all():
        raise ContractError("height must be finite")
    if isinstance(spacing_m, tuple):
        dx, dy = map(float, spacing_m)
    else:
        dx = dy = float(spacing_m)
    if dx <= 0.0 or dy <= 0.0 or not np.isfinite((dx, dy)).all():
        raise ContractError("heightfield spacing must be positive and finite")
    rows, columns = values.shape
    x = float(origin_xy[0]) + dx * np.arange(columns)
    y = float(origin_xy[1]) + dy * np.arange(rows)
    grid_x, grid_y = np.meshgrid(x, y)
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), values.ravel()))
    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            d = (row + 1) * columns + column
            c = d + 1
            if validity[row, column] and validity[row, column + 1] and validity[row + 1, column + 1]:
                faces.append((a, b, c))
            if validity[row, column] and validity[row + 1, column + 1] and validity[row + 1, column]:
                faces.append((a, c, d))
    face_array = np.asarray(faces, dtype=np.int32).reshape((-1, 3))
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    digest.update(validity.tobytes(order="C"))
    digest.update(np.asarray((dx, dy, *origin_xy), dtype="<f8").tobytes())
    return CanonicalTerrainMesh(
        vertices_local=np.asarray(vertices, dtype=np.float32),
        faces=face_array,
        valid_faces=np.ones(len(face_array), dtype=np.bool_),
        source_asset_sha256=digest.hexdigest(),
    )


__all__ = (
    "ErodedSupportPatch",
    "RayHit",
    "SupportConfig",
    "SupportPatch",
    "TerrainMeshIndex",
    "erode_support_for_sole",
    "extract_support_patches",
    "mesh_from_heightfield",
)
