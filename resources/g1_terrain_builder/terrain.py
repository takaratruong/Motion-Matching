from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import pickle
import struct
import warnings

import numpy as np
from pxr import Usd, UsdGeom

from resources import quat as holden_quat

from .sources import load_grail_object_pose


LOOKAHEAD = np.array([0.25, 0.50, 0.75, 1.00], np.float64)
USD_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/object_usd"
RECON_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/recon"
# GRAIL object poses are expressed for the Isaac-imported asset basis.  Raw
# released terrain USD points use the source asset basis; this proper rotation
# maps those local axes into the simulation frame before applying root_quat.
GRAIL_ASSET_TO_SIMULATION_BASIS = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
], np.float64)
BBOX_TOLERANCE_M = 1e-12
PROJECTED_AREA_EPSILON_M2 = 1e-12
BARYCENTRIC_TOLERANCE = 1e-10
HEIGHTFIELD_HEADER = struct.Struct("<4sIII4f")
HEIGHTFIELD_VERSION = 2
HEIGHTFIELD_DIAGONAL = "min-x-min-z_to_max-x-max-z"
HEIGHTFIELD_INTERPOLATION = "fixed-diagonal-triangles"
HEIGHTFIELD_SCALAR_ENCODING = "ieee754-binary32-little-endian"
HEIGHTFIELD_DOMAIN_POLICY = "inclusive-authoritative-node-rectangle"
HEIGHTFIELD_GRID_LINE_POLICY = "positive-index-cell-except-maximum-edge"
HEIGHTFIELD_DIAGONAL_TIE_POLICY = \
    "tx-greater-or-equal-tz-uses-p00-p10-p11"
HEIGHTFIELD_RASTER_BOUNDS_POLICY = \
    "float32-minimum-rounded-down-and-maximum-ceil-covered"
HEIGHTFIELD_OBJ_VERTEX_ORDER = "z-major-x-minor"
HEIGHTFIELD_OBJ_FACE_ORDER = "p00-p11-p10_then_p00-p01-p11"
HEIGHTFIELD_OBJ_FLOAT_FORMAT = ".9g-final-newline"
HEIGHTFIELD_EXTERIOR_NORMAL = (0.0, 1.0, 0.0)
HEIGHTFIELD_MAX_SAMPLES = np.iinfo(np.int32).max
HEIGHTFIELD_MIN_NORMAL = float(np.finfo(np.float32).tiny)
HEIGHTFIELD_SOURCE_NODE_ENCODING = \
    "binary32-header-values-promoted-to-binary64-arithmetic"
HEIGHTFIELD_RUNTIME_QUERY_ENCODING = \
    "normal-or-zero-binary32-canonicalized-positive-and-promoted-to-binary64"
HEIGHTFIELD_SCALAR_DOMAIN = "normal-or-zero-binary32"
HEIGHTFIELD_CELL_DOMAIN = "positive-normal-binary32"
HEIGHTFIELD_RUNTIME_NODE_DOMAIN = "normal-or-zero-binary32"
HEIGHTFIELD_RUNTIME_QUERY_DOMAIN = "normal-or-zero-binary32-coordinates"
HEIGHTFIELD_RUNTIME_PARITY_DOMAIN = "normal-or-zero-binary32-coordinates"
HEIGHTFIELD_DENORMAL_POLICY = "reject-nonzero-binary32-subnormals"
HEIGHTFIELD_EVALUATION_PRECISION = \
    "binary64-from-binary32-samples-and-promoted-node-weights"
HEIGHTFIELD_RUNTIME_HEIGHT_OUTPUT = \
    "finite-binary64-interpolation-rounded-to-binary32"
HEIGHTFIELD_NORMAL_EVALUATION = \
    "selected-triangle-binary64-gradient-scale-safe-unit-normalization"
HEIGHTFIELD_RUNTIME_NORMAL_OUTPUT = \
    "unit-normal-components-rounded-to-binary32"
HEIGHTFIELD_RUNTIME_OUTPUT_FTZ_POLICY = \
    "binary32-subnormals-and-signed-zero-canonicalized-to-positive-zero"
HEIGHTFIELD_ZERO_ENCODING = "canonical-positive-zero"
HEIGHTFIELD_RUNTIME_NODE_DISTINGUISHABILITY_POLICY = \
    "normal-or-positive-zero-strictly-increasing-proven-by-endpoints-" \
    "near-zero-candidates-max-binary32-spacing-and-aligned-equality"
HEIGHTFIELD_OBJ_COORDINATE_QUANTIZATION = \
    "binary32-round-of-promoted-origin-plus-index-times-cell"

SURFACE_SEMANTICS = {
    "schema": "g1-terrain-surface/v1",
    "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
    "source_query": "vertical-triangle-top",
    "polygon_triangulation": "fan-from-first-index",
    "overlap_height_policy": "maximum-y",
    "projected_boundary_policy": "closed",
    "triangle_winding_policy": "orientation-independent",
    "degenerate_projected_triangle_policy": "ignore",
    "projected_area_measure": "absolute-two-times-area",
    "bbox_tolerance_m": BBOX_TOLERANCE_M,
    "projected_area_epsilon_m2": PROJECTED_AREA_EPSILON_M2,
    "barycentric_tolerance": BARYCENTRIC_TOLERANCE,
    "heightfield_schema": "G1HF/v2",
    "heightfield_version": HEIGHTFIELD_VERSION,
    "heightfield_interpolation": HEIGHTFIELD_INTERPOLATION,
    "heightfield_diagonal": HEIGHTFIELD_DIAGONAL,
    "heightfield_scalar_encoding": HEIGHTFIELD_SCALAR_ENCODING,
    "heightfield_domain_policy": HEIGHTFIELD_DOMAIN_POLICY,
    "heightfield_grid_line_policy": HEIGHTFIELD_GRID_LINE_POLICY,
    "heightfield_diagonal_tie_policy": HEIGHTFIELD_DIAGONAL_TIE_POLICY,
    "heightfield_exterior_normal": HEIGHTFIELD_EXTERIOR_NORMAL,
    "heightfield_source_node_encoding": HEIGHTFIELD_SOURCE_NODE_ENCODING,
    "heightfield_runtime_query_encoding": HEIGHTFIELD_RUNTIME_QUERY_ENCODING,
    "heightfield_scalar_domain": HEIGHTFIELD_SCALAR_DOMAIN,
    "heightfield_cell_domain": HEIGHTFIELD_CELL_DOMAIN,
    "heightfield_runtime_node_domain": HEIGHTFIELD_RUNTIME_NODE_DOMAIN,
    "heightfield_runtime_query_domain": HEIGHTFIELD_RUNTIME_QUERY_DOMAIN,
    "heightfield_runtime_parity_domain": HEIGHTFIELD_RUNTIME_PARITY_DOMAIN,
    "heightfield_denormal_policy": HEIGHTFIELD_DENORMAL_POLICY,
    "heightfield_evaluation_precision": HEIGHTFIELD_EVALUATION_PRECISION,
    "heightfield_runtime_height_output": HEIGHTFIELD_RUNTIME_HEIGHT_OUTPUT,
    "heightfield_normal_evaluation": HEIGHTFIELD_NORMAL_EVALUATION,
    "heightfield_runtime_normal_output": HEIGHTFIELD_RUNTIME_NORMAL_OUTPUT,
    "heightfield_runtime_output_ftz_policy":
        HEIGHTFIELD_RUNTIME_OUTPUT_FTZ_POLICY,
    "heightfield_zero_encoding": HEIGHTFIELD_ZERO_ENCODING,
    "heightfield_runtime_node_distinguishability_policy":
        HEIGHTFIELD_RUNTIME_NODE_DISTINGUISHABILITY_POLICY,
    "heightfield_obj_coordinate_quantization":
        HEIGHTFIELD_OBJ_COORDINATE_QUANTIZATION,
    "heightfield_raster_bounds_policy": HEIGHTFIELD_RASTER_BOUNDS_POLICY,
    "heightfield_obj_vertex_order": HEIGHTFIELD_OBJ_VERTEX_ORDER,
    "heightfield_obj_face_order": HEIGHTFIELD_OBJ_FACE_ORDER,
    "heightfield_obj_float_format": HEIGHTFIELD_OBJ_FLOAT_FORMAT,
    "cell_size_m": 0.02,
    "exterior_height_m": 0.0,
}


def surface_semantics() -> dict:
    return json.loads(json.dumps(SURFACE_SEMANTICS, sort_keys=True))


def surface_semantics_signature() -> str:
    payload = json.dumps(
        SURFACE_SEMANTICS, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FlatTerrain:
    def height(self, x: float, z: float) -> float:
        return 0.0


@dataclass(frozen=True)
class StepTerrain:
    edge_x: float
    height_value: float

    def __init__(self, edge_x: float, height: float):
        object.__setattr__(self, "edge_x", float(edge_x))
        object.__setattr__(self, "height_value", float(height))

    def height(self, x: float, z: float) -> float:
        return self.height_value if x >= self.edge_x else 0.0


def _normalized(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return fallback.copy() if length < 1e-8 else vector / length


def build_facing_centerline(
    root_xz: np.ndarray,
    headings_xz: np.ndarray,
    path_xz: np.ndarray,
) -> np.ndarray:
    root = np.asarray(root_xz, np.float64)
    headings = np.asarray(headings_xz, np.float64)
    path = np.asarray(path_xz, np.float64)
    if root.shape != (2,):
        raise ValueError(f"root_xz must have shape (2,), got {root.shape}")
    if headings.ndim != 2 or headings.shape[1:] != (2,) or not len(headings):
        raise ValueError(
            f"headings_xz must have non-empty shape (N, 2), got {headings.shape}")
    if path.shape != headings.shape:
        raise ValueError(
            f"path_xz shape {path.shape} does not match headings {headings.shape}")
    if not (
        np.all(np.isfinite(root))
        and np.all(np.isfinite(headings))
        and np.all(np.isfinite(path))
    ):
        raise ValueError("centerline inputs must be finite")

    points = [root]
    last_direction = _normalized(headings[0], np.array([0.0, 1.0]))
    for point, heading in zip(path[1:], headings[1:]):
        segment = point - points[-1]
        if np.linalg.norm(segment) > 1e-6:
            points.append(point.copy())
        last_direction = _normalized(heading, last_direction)

    travelled = sum(
        np.linalg.norm(b - a) for a, b in zip(points, points[1:]))
    if travelled < LOOKAHEAD[-1]:
        points.append(points[-1] + last_direction * (LOOKAHEAD[-1] - travelled))
    return np.asarray(points, np.float64)


def _point_at_arc_distance(line: np.ndarray, distance: float) -> np.ndarray:
    remaining = distance
    for start, end in zip(line, line[1:]):
        length = np.linalg.norm(end - start)
        if remaining <= length:
            return start + (remaining / max(length, 1e-8)) * (end - start)
        remaining -= length
    return line[-1]


def sample_terrain_features(terrain, centerline: np.ndarray) -> np.ndarray:
    line = np.asarray(centerline, np.float64)
    if line.ndim != 2 or line.shape[1:] != (2,) or not len(line):
        raise ValueError(f"centerline must have non-empty shape (N, 2), got {line.shape}")
    if not np.all(np.isfinite(line)):
        raise ValueError("centerline coordinates must be finite")
    root_height = terrain.height(*line[0])
    features = np.array([
        terrain.height(*_point_at_arc_distance(line, distance)) - root_height
        for distance in LOOKAHEAD
    ], np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("terrain features must be finite")
    return features


def _load_usd_mesh_path(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise FileNotFoundError(f"unable to open GRAIL terrain USD: {path}")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.array(
                mesh.GetPointsAttr().Get(), np.float64, copy=True)
            counts = _checked_int32_topology(
                mesh.GetFaceVertexCountsAttr().Get(), "face counts")
            indices = _checked_int32_topology(
                mesh.GetFaceVertexIndicesAttr().Get(), "face indices")
            if points.ndim != 2 or points.shape[1:] != (3,):
                raise ValueError(f"invalid GRAIL mesh points in {path}")
            if counts.ndim != 1 or indices.ndim != 1:
                raise ValueError(f"invalid GRAIL mesh topology in {path}")
            _validate_mesh_topology(points, counts, indices)
            return points, counts, indices
    raise ValueError(f"no mesh found in GRAIL terrain USD: {path}")


def _load_usd_mesh(base: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_usd_mesh_path(os.path.join(USD_DIR, base + ".usd"))


def _object_pose0(base: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(RECON_DIR, base + ".pkl")
    with open(path, "rb") as stream:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="numpy.core.numeric is deprecated",
                category=DeprecationWarning,
            )
            reconstruction = pickle.load(stream)
    object_data = reconstruction["obj_data"]
    rotation = np.asarray(object_data["obj_R"], np.float64)[0]
    translation = np.asarray(object_data["obj_t"], np.float64)[0]
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"invalid GRAIL reconstruction pose in {path}")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"non-finite GRAIL reconstruction pose in {path}")
    return rotation, translation


def _validate_mesh_topology(
    vertices: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("terrain vertices must have non-empty shape (N, 3)")
    if face_counts.ndim != 1 or face_indices.ndim != 1:
        raise ValueError("face counts and indices must be one-dimensional")
    if not len(face_counts):
        raise ValueError("terrain faces must be non-empty")
    if np.any(face_counts < 3):
        raise ValueError("terrain faces must have at least three vertices")
    if int(face_counts.sum()) != len(face_indices):
        raise ValueError("face counts do not match the number of face indices")
    vertex_count = len(vertices)
    if np.any(face_indices < 0) or np.any(face_indices >= vertex_count):
        raise ValueError(
            f"face indices must be in [0, {vertex_count})")


def _checked_int32_topology(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        limits = np.iinfo(np.int32)
        if minimum < limits.min or maximum > limits.max:
            raise ValueError(f"{name} must fit int32")
    return np.array(array, np.int32, copy=True)


def triangulate_faces(
    vertices: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, np.float64)
    counts = _checked_int32_topology(face_counts, "face counts")
    indices = _checked_int32_topology(face_indices, "face indices")
    _validate_mesh_topology(vertices, counts, indices)
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor:cursor + int(count)]
        cursor += int(count)
        triangles.extend(
            (int(face[0]), int(face[index]), int(face[index + 1]))
            for index in range(1, int(count) - 1)
        )
    return np.asarray(triangles, np.int32)


class VerticalTriangleSurface:
    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        exterior_height: float = 0.0,
    ):
        self.vertices = np.array(vertices, np.float64, copy=True)
        self.triangles = _checked_int32_topology(
            triangles, "surface triangle indices")
        if self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,) \
                or not len(self.vertices):
            raise ValueError("surface vertices must have non-empty shape (N, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1:] != (3,) \
                or not len(self.triangles):
            raise ValueError("surface triangles must have non-empty shape (M, 3)")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(self.vertices)):
            raise ValueError("surface triangle index is outside the vertex array")
        if not np.isfinite(self.vertices).all() or not np.isfinite(exterior_height):
            raise ValueError("surface vertices and exterior height must be finite")
        self.exterior_height = float(exterior_height)
        self._triangle_vertices = self.vertices[self.triangles]
        projected = self._triangle_vertices[:, :, (0, 2)]
        self._minimum_xz = projected.min(axis=1)
        self._maximum_xz = projected.max(axis=1)
        for array in (
            self.vertices,
            self.triangles,
            self._triangle_vertices,
            self._minimum_xz,
            self._maximum_xz,
        ):
            array.setflags(write=False)

    def height(self, x: float, z: float) -> float:
        x, z = float(x), float(z)
        if not np.isfinite(x) or not np.isfinite(z):
            raise ValueError("terrain query coordinates must be finite")
        candidates = np.flatnonzero(
            (self._minimum_xz[:, 0] - BBOX_TOLERANCE_M <= x)
            & (x <= self._maximum_xz[:, 0] + BBOX_TOLERANCE_M)
            & (self._minimum_xz[:, 1] - BBOX_TOLERANCE_M <= z)
            & (z <= self._maximum_xz[:, 1] + BBOX_TOLERANCE_M)
        )
        if not len(candidates):
            return self.exterior_height
        triangle = self._triangle_vertices[candidates]
        a = triangle[:, 0]
        b = triangle[:, 1]
        c = triangle[:, 2]
        v0x, v0z = b[:, 0] - a[:, 0], b[:, 2] - a[:, 2]
        v1x, v1z = c[:, 0] - a[:, 0], c[:, 2] - a[:, 2]
        px, pz = x - a[:, 0], z - a[:, 2]
        determinant = v0x * v1z - v0z * v1x
        projected = np.abs(determinant) > PROJECTED_AREA_EPSILON_M2
        u = np.zeros_like(determinant)
        v = np.zeros_like(determinant)
        u[projected] = (
            px[projected] * v1z[projected]
            - pz[projected] * v1x[projected]
        ) / determinant[projected]
        v[projected] = (
            v0x[projected] * pz[projected]
            - v0z[projected] * px[projected]
        ) / determinant[projected]
        w = 1.0 - u - v
        inside = projected \
            & (u >= -BARYCENTRIC_TOLERANCE) \
            & (v >= -BARYCENTRIC_TOLERANCE) \
            & (w >= -BARYCENTRIC_TOLERANCE)
        if not np.any(inside):
            return self.exterior_height
        heights = w[inside] * a[inside, 1] \
            + u[inside] * b[inside, 1] \
            + v[inside] * c[inside, 1]
        return float(np.max(heights))


def _mujoco_to_holden(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))


class GrailTerrain(VerticalTriangleSurface):
    def __init__(self, render_vertices, face_counts, face_indices):
        self._vertices = np.array(render_vertices, np.float64, copy=True)
        self._face_counts = _checked_int32_topology(
            face_counts, "face counts")
        self._face_indices = _checked_int32_topology(
            face_indices, "face indices")
        _validate_mesh_topology(
            self._vertices, self._face_counts, self._face_indices)
        super().__init__(
            self._vertices,
            triangulate_faces(
                self._vertices, self._face_counts, self._face_indices),
            exterior_height=0.0,
        )
        self._max_height = float(self._vertices[:, 1].max())
        self._vertices.setflags(write=False)
        self._face_counts.setflags(write=False)
        self._face_indices.setflags(write=False)

    @classmethod
    def from_base(cls, base: str) -> "GrailTerrain":
        vertices, face_counts, face_indices = _load_usd_mesh(base)
        rotation, translation = _object_pose0(base)
        world_vertices = (rotation @ vertices.T).T + translation
        return cls(_mujoco_to_holden(world_vertices), face_counts, face_indices)

    @classmethod
    def from_release(cls, usd_path: str, objects_path: str) -> "GrailTerrain":
        vertices, face_counts, face_indices = _load_usd_mesh_path(usd_path)
        pose = load_grail_object_pose(objects_path)
        scaled_vertices = vertices * pose.scale
        simulation_vertices = (
            GRAIL_ASSET_TO_SIMULATION_BASIS @ scaled_vertices.T).T
        rotation = holden_quat.to_xform(pose.root_quat)
        world_vertices = (rotation @ simulation_vertices.T).T + pose.root_pos
        return cls(_mujoco_to_holden(world_vertices), face_counts, face_indices)

    def footprint(self) -> dict:
        top = self._vertices[
            self._vertices[:, 1] > self._max_height - 0.02]
        return {
            "x": (float(top[:, 0].min()), float(top[:, 0].max())),
            "z": (float(top[:, 2].min()), float(top[:, 2].max())),
            "height": self._max_height,
        }

    def xz_bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self._vertices[:, 0].min()),
            float(self._vertices[:, 0].max()),
            float(self._vertices[:, 2].min()),
            float(self._vertices[:, 2].max()),
        )

    def export_obj(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            for x, y, z in self._vertices:
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            offset = 0
            for count in self._face_counts:
                indices = self._face_indices[offset:offset + count] + 1
                offset += count
                stream.write("f " + " ".join(str(int(i)) for i in indices) + "\n")


def _authoritative_float32(value, name: str) -> float:
    try:
        value = float(value)
        with np.errstate(over="ignore", invalid="ignore"):
            encoded = np.float32(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"heightfield {name} must encode as float32") from error
    decoded = float(encoded)
    if not math.isfinite(decoded):
        raise ValueError(f"heightfield {name} must encode as finite float32")
    return 0.0 if decoded == 0.0 else decoded


def _validate_heightfield_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise ValueError("heightfield heights must have shape (nz, nx), both at least 2")
    nz, nx = int(shape[0]), int(shape[1])
    if nx > HEIGHTFIELD_MAX_SAMPLES or nz > HEIGHTFIELD_MAX_SAMPLES \
            or nx * nz > HEIGHTFIELD_MAX_SAMPLES:
        raise ValueError("heightfield dimensions and sample count must fit INT_MAX")
    return nz, nx


def _runtime_node_coordinate(
    origin: float,
    index: int,
    cell_size: float,
) -> tuple[float, float]:
    source_coordinate = origin + index * cell_size
    with np.errstate(over="ignore", invalid="ignore"):
        runtime_coordinate = float(np.float32(source_coordinate))
    if runtime_coordinate == 0.0:
        runtime_coordinate = 0.0
    return source_coordinate, runtime_coordinate


def _is_normal_or_zero_binary32(value: float) -> bool:
    return value == 0.0 or abs(value) >= HEIGHTFIELD_MIN_NORMAL


def _require_normal_or_zero_binary32(value: float, name: str) -> None:
    if not _is_normal_or_zero_binary32(value):
        raise ValueError(
            f"heightfield {name} must encode as normal-or-zero binary32")


def _maximum_inward_spacing(
    first: float,
    last: float,
    binary32: bool,
) -> float:
    def next_toward(value: float, direction: float) -> float:
        if binary32:
            return float(np.nextafter(
                np.float32(value), np.float32(direction)))
        return math.nextafter(value, direction)

    if first >= 0.0:
        return last - next_toward(last, -math.inf)
    if last <= 0.0:
        return next_toward(first, math.inf) - first
    return max(
        next_toward(first, math.inf) - first,
        last - next_toward(last, -math.inf),
    )


def _validate_runtime_axis(
    origin: float,
    count: int,
    cell_size: float,
    axis: str,
) -> None:
    first_source, first_runtime = _runtime_node_coordinate(
        origin, 0, cell_size)
    second_source, second_runtime = _runtime_node_coordinate(
        origin, 1, cell_size)
    penultimate_source, penultimate_runtime = _runtime_node_coordinate(
        origin, count - 2, cell_size)
    last_source, last_runtime = _runtime_node_coordinate(
        origin, count - 1, cell_size)
    nodes = (
        first_source, first_runtime, second_source, second_runtime,
        penultimate_source, penultimate_runtime, last_source, last_runtime,
    )
    if not all(math.isfinite(value) for value in nodes) \
            or second_source <= first_source \
            or second_runtime <= first_runtime \
            or last_source <= penultimate_source \
            or last_runtime <= penultimate_runtime:
        raise ValueError(
            f"heightfield {axis} nodes must be finite and runtime-distinguishable")
    if not all(_is_normal_or_zero_binary32(value) for value in (
            first_runtime, second_runtime,
            penultimate_runtime, last_runtime)):
        raise ValueError(
            f"heightfield {axis} nodes must be normal-or-zero binary32")

    if first_source < 0.0 < last_source:
        zero_index = -origin / cell_size
        base = math.floor(zero_index)
        for index in range(max(0, base - 1), min(count, base + 3)):
            _, runtime = _runtime_node_coordinate(origin, index, cell_size)
            if not _is_normal_or_zero_binary32(runtime):
                raise ValueError(
                    f"heightfield {axis} nodes must be "
                    "normal-or-zero binary32")
    if count <= 3:
        return

    # Float rounding is monotone, and representable spacing grows only with
    # distance from zero. Endpoint-inward gaps therefore bound every rounding
    # bucket in this uniform interval. A step larger than that bound cannot
    # collapse; equality is safe only on the same representable cell lattice.
    maximum_spacing = max(
        _maximum_inward_spacing(
            first_source, last_source, binary32=False),
        _maximum_inward_spacing(
            first_runtime, last_runtime, binary32=True),
    )
    quotient = origin / cell_size
    aligned_equality = cell_size == maximum_spacing \
        and math.isfinite(quotient) and quotient.is_integer()
    if cell_size < maximum_spacing \
            or (cell_size == maximum_spacing and not aligned_equality):
        raise ValueError(
            f"heightfield {axis} nodes must be finite and runtime-distinguishable")


@dataclass(frozen=True)
class HeightGrid:
    heights: np.ndarray
    origin_x: float
    origin_z: float
    cell_size: float
    exterior_height: float
    version: int = field(default=HEIGHTFIELD_VERSION, init=False)

    def __post_init__(self) -> None:
        source = np.asarray(self.heights)
        nz, nx = _validate_heightfield_shape(source.shape)
        if np.iscomplexobj(source):
            raise ValueError("heightfield heights must be real-valued")
        origin_x = _authoritative_float32(self.origin_x, "origin_x")
        origin_z = _authoritative_float32(self.origin_z, "origin_z")
        cell_size = _authoritative_float32(self.cell_size, "cell_size")
        exterior_height = _authoritative_float32(
            self.exterior_height, "exterior_height")
        _require_normal_or_zero_binary32(origin_x, "origin_x")
        _require_normal_or_zero_binary32(origin_z, "origin_z")
        _require_normal_or_zero_binary32(exterior_height, "exterior_height")
        if cell_size < HEIGHTFIELD_MIN_NORMAL:
            raise ValueError(
                "heightfield cell_size must encode as positive normal binary32")
        _validate_runtime_axis(origin_x, nx, cell_size, "X")
        _validate_runtime_axis(origin_z, nz, cell_size, "Z")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                heights = np.array(source, dtype="<f4", order="C", copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "heightfield heights must be real-valued float32 samples") \
                from error
        if not np.all(np.isfinite(heights)):
            raise ValueError("heightfield heights must encode as finite float32")
        subnormal = (heights != 0.0) \
            & (np.abs(heights) < HEIGHTFIELD_MIN_NORMAL)
        if np.any(subnormal):
            raise ValueError(
                "heightfield heights must encode as normal-or-zero binary32")
        heights[heights == 0.0] = np.float32(0.0)
        heights.setflags(write=False)
        object.__setattr__(self, "heights", heights)
        object.__setattr__(self, "origin_x", origin_x)
        object.__setattr__(self, "origin_z", origin_z)
        object.__setattr__(self, "cell_size", cell_size)
        object.__setattr__(self, "exterior_height", exterior_height)

    @property
    def nx(self) -> int:
        return int(self.heights.shape[1])

    @property
    def nz(self) -> int:
        return int(self.heights.shape[0])

    @property
    def max_x(self) -> float:
        return self.origin_x + (self.nx - 1) * self.cell_size

    @property
    def max_z(self) -> float:
        return self.origin_z + (self.nz - 1) * self.cell_size

    @staticmethod
    def _query_coordinate(value, name: str) -> float:
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"heightfield query {name} must be finite") from error
        if not math.isfinite(value):
            raise ValueError(f"heightfield query {name} must be finite")
        return 0.0 if value == 0.0 else value

    def _axis_cell(
        self,
        value: float,
        origin: float,
        count: int,
    ) -> tuple[int, float]:
        coordinate = (value - origin) / self.cell_size
        index = min(max(int(math.floor(coordinate)), 0), count - 2)
        while index > 0 and value < origin + index * self.cell_size:
            index -= 1
        while index < count - 2 \
                and value >= origin + (index + 1) * self.cell_size:
            index += 1
        fraction = (value - (origin + index * self.cell_size)) \
            / self.cell_size
        return index, min(max(fraction, 0.0), 1.0)

    def _cell(self, x, z) -> tuple[int, int, float, float] | None:
        x = self._query_coordinate(x, "x")
        z = self._query_coordinate(z, "z")
        if x < self.origin_x or x > self.max_x \
                or z < self.origin_z or z > self.max_z:
            return None
        ix, tx = self._axis_cell(x, self.origin_x, self.nx)
        iz, tz = self._axis_cell(z, self.origin_z, self.nz)
        return ix, iz, tx, tz

    def height(self, x: float, z: float) -> float:
        cell = self._cell(x, z)
        if cell is None:
            return self.exterior_height
        ix, iz, tx, tz = cell
        h00 = float(self.heights[iz, ix])
        h10 = float(self.heights[iz, ix + 1])
        h01 = float(self.heights[iz + 1, ix])
        h11 = float(self.heights[iz + 1, ix + 1])
        if tx >= tz:
            return h00 + tx * (h10 - h00) + tz * (h11 - h10)
        return h00 + tx * (h11 - h01) + tz * (h01 - h00)

    def normal(self, x: float, z: float) -> np.ndarray:
        cell = self._cell(x, z)
        if cell is None:
            return np.asarray(HEIGHTFIELD_EXTERIOR_NORMAL, np.float64)
        ix, iz, tx, tz = cell
        h00 = float(self.heights[iz, ix])
        h10 = float(self.heights[iz, ix + 1])
        h01 = float(self.heights[iz + 1, ix])
        h11 = float(self.heights[iz + 1, ix + 1])
        if tx >= tz:
            slope_x = (h10 - h00) / self.cell_size
            slope_z = (h11 - h10) / self.cell_size
        else:
            slope_x = (h11 - h01) / self.cell_size
            slope_z = (h01 - h00) / self.cell_size
        normal = np.array([-slope_x, 1.0, -slope_z], np.float64)
        scaled = normal / np.max(np.abs(normal))
        return scaled / np.linalg.norm(scaled)

    def g1hf_bytes(self) -> bytes:
        return HEIGHTFIELD_HEADER.pack(
            b"G1HF",
            self.version,
            self.nx,
            self.nz,
            self.origin_x,
            self.origin_z,
            self.cell_size,
            self.exterior_height,
        ) + self.heights.tobytes(order="C")

    def obj_bytes(self) -> bytes:
        lines = []
        for iz in range(self.nz):
            _, z = _runtime_node_coordinate(
                self.origin_z, iz, self.cell_size)
            for ix in range(self.nx):
                _, x = _runtime_node_coordinate(
                    self.origin_x, ix, self.cell_size)
                y = float(self.heights[iz, ix])
                lines.append(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for iz in range(self.nz - 1):
            for ix in range(self.nx - 1):
                p00 = iz * self.nx + ix + 1
                p10 = p00 + 1
                p01 = p00 + self.nx
                p11 = p01 + 1
                lines.append(f"f {p00} {p11} {p10}\n")
                lines.append(f"f {p00} {p01} {p11}\n")
        return "".join(lines).encode("ascii")

    def metadata(self) -> dict:
        return {
            "schema": "G1HF/v2",
            "version": self.version,
            "nx": self.nx,
            "nz": self.nz,
            "origin_x": self.origin_x,
            "origin_z": self.origin_z,
            "cell_size_m": self.cell_size,
            "exterior_height_m": self.exterior_height,
            "interpolation": HEIGHTFIELD_INTERPOLATION,
            "diagonal": HEIGHTFIELD_DIAGONAL,
        }


def _raster_dimension(origin: float, maximum: float, cell_size: float) -> int:
    span = (maximum - origin) / cell_size
    if not math.isfinite(span) or span > HEIGHTFIELD_MAX_SAMPLES - 1:
        raise ValueError("heightfield raster dimension must fit INT_MAX")
    steps = int(math.ceil(span))
    if origin + steps * cell_size < maximum:
        steps += 1
    count = steps + 1
    if count > HEIGHTFIELD_MAX_SAMPLES:
        raise ValueError("heightfield raster dimension must fit INT_MAX")
    return count


def rasterize_heightfield(
    terrain,
    bounds: tuple[float, float, float, float],
    cell_size: float = 0.02,
) -> HeightGrid:
    try:
        xmin, xmax, zmin, zmax = (float(value) for value in bounds)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("heightfield bounds must contain four numbers") from error
    if not np.all(np.isfinite([xmin, xmax, zmin, zmax])):
        raise ValueError("heightfield bounds must be finite")
    if xmax <= xmin:
        raise ValueError("heightfield xmax must be greater than xmin")
    if zmax <= zmin:
        raise ValueError("heightfield zmax must be greater than zmin")
    encoded_cell = _authoritative_float32(cell_size, "cell_size")
    if encoded_cell < HEIGHTFIELD_MIN_NORMAL:
        raise ValueError(
            "heightfield cell_size must encode as positive normal binary32")

    def rounded_down(value: float, name: str) -> float:
        encoded = np.float32(_authoritative_float32(value, name))
        if float(encoded) > value:
            encoded = np.nextafter(encoded, np.float32(-np.inf))
        decoded = float(encoded)
        if not math.isfinite(decoded):
            raise ValueError(f"heightfield {name} cannot be rounded down in float32")
        _require_normal_or_zero_binary32(decoded, name)
        return decoded

    origin_x = rounded_down(xmin, "origin_x")
    origin_z = rounded_down(zmin, "origin_z")
    nx = _raster_dimension(origin_x, xmax, encoded_cell)
    nz = _raster_dimension(origin_z, zmax, encoded_cell)
    if nx * nz > HEIGHTFIELD_MAX_SAMPLES:
        raise ValueError("heightfield raster sample count must fit INT_MAX")
    _validate_runtime_axis(origin_x, nx, encoded_cell, "X")
    _validate_runtime_axis(origin_z, nz, encoded_cell, "Z")

    heights = np.empty((nz, nx), np.float64)
    for iz in range(nz):
        z = origin_z + iz * encoded_cell
        for ix in range(nx):
            x = origin_x + ix * encoded_cell
            heights[iz, ix] = terrain.height(x, z)
    return HeightGrid(heights, origin_x, origin_z, encoded_cell, 0.0)


def _fixed_triangle_height(grid, ix, iz, tx, tz):
    h00 = float(grid.heights[iz, ix])
    h10 = float(grid.heights[iz, ix + 1])
    h01 = float(grid.heights[iz + 1, ix])
    h11 = float(grid.heights[iz + 1, ix + 1])
    if tx >= tz:
        return h00 + tx * (h10 - h00) + tz * (h11 - h10)
    return h00 + tx * (h11 - h01) + tz * (h01 - h00)


def grail_surface_parity(terrain, grid):
    if not isinstance(terrain, GrailTerrain) or not isinstance(grid, HeightGrid):
        raise TypeError("GRAIL parity requires GrailTerrain and HeightGrid")
    source_nodes = np.empty_like(grid.heights, dtype=np.float64)
    for iz in range(grid.nz):
        z = grid.origin_z + iz * grid.cell_size
        for ix in range(grid.nx):
            x = grid.origin_x + ix * grid.cell_size
            source_nodes[iz, ix] = terrain.height(x, z)
    node_error = float(np.max(np.abs(
        source_nodes - grid.heights.astype(np.float64))))

    cell_count = (grid.nx - 1) * (grid.nz - 1)
    sample_count = min(cell_count, 4096)
    linear_cells = np.unique(np.linspace(
        0, cell_count - 1, sample_count, dtype=np.int64))
    local_probes = ((0.25, 0.125), (0.75, 0.25),
                    (0.25, 0.75), (0.75, 0.875))
    within_cell_error = 0.0
    source_away_error = 0.0
    away_edge_probe_count = 0
    for linear in linear_cells:
        iz, ix = divmod(int(linear), grid.nx - 1)
        corner_heights = [
            float(grid.heights[iz, ix]),
            float(grid.heights[iz, ix + 1]),
            float(grid.heights[iz + 1, ix]),
            float(grid.heights[iz + 1, ix + 1]),
        ]
        probes = []
        for tx, tz in local_probes:
            x = grid.origin_x + (ix + tx) * grid.cell_size
            z = grid.origin_z + (iz + tz) * grid.cell_size
            queried = grid.height(x, z)
            explicit = _fixed_triangle_height(grid, ix, iz, tx, tz)
            within_cell_error = max(
                within_cell_error, abs(queried - explicit))
            probes.append((queried, terrain.height(x, z)))
        local_source = corner_heights + [source for _, source in probes]
        if max(local_source) - min(local_source) <= 0.005:
            for queried, source in probes:
                source_away_error = max(
                    source_away_error, abs(queried - source))
                away_edge_probe_count += 1

    footprint = terrain.footprint()
    threshold = footprint["height"] - 0.02
    top_iz, top_ix = np.nonzero(grid.heights >= threshold)
    if not len(top_ix):
        raise ValueError("GRAIL grid contains no measured top nodes")
    grid_top_bounds = (
        grid.origin_x + int(top_ix.min()) * grid.cell_size,
        grid.origin_x + int(top_ix.max()) * grid.cell_size,
        grid.origin_z + int(top_iz.min()) * grid.cell_size,
        grid.origin_z + int(top_iz.max()) * grid.cell_size,
    )
    source_top_bounds = (
        footprint["x"][0], footprint["x"][1],
        footprint["z"][0], footprint["z"][1],
    )
    edge_movement = max(
        abs(float(actual) - float(expected))
        for actual, expected in zip(grid_top_bounds, source_top_bounds)
    )
    return {
        "node_error_m": node_error,
        "within_cell_error_m": float(within_cell_error),
        "source_away_edge_error_m": float(source_away_error),
        "top_edge_movement_m": float(edge_movement),
        "away_edge_probe_count": away_edge_probe_count,
    }


def export_heightfield_obj(grid: HeightGrid, path: str) -> None:
    if not isinstance(grid, HeightGrid):
        raise TypeError("heightfield OBJ export requires a HeightGrid")
    payload = grid.obj_bytes()
    with open(path, "wb") as stream:
        stream.write(payload)


def export_heightfield(
    terrain,
    bounds: tuple[float, float, float, float],
    cell_size: float,
    path: str,
) -> dict:
    xmin, xmax, zmin, zmax = (float(value) for value in bounds)
    if not np.all(np.isfinite([xmin, xmax, zmin, zmax, cell_size])):
        raise ValueError("heightfield bounds and cell size must be finite")
    if xmax <= xmin:
        raise ValueError("heightfield xmax must be greater than xmin")
    if zmax <= zmin:
        raise ValueError("heightfield zmax must be greater than zmin")
    if cell_size <= 0.0:
        raise ValueError("heightfield cell size must be positive")
    nx = int(np.ceil((xmax - xmin) / cell_size)) + 1
    nz = int(np.ceil((zmax - zmin) / cell_size)) + 1
    values = np.empty((nz, nx), dtype="<f4")
    for iz in range(nz):
        for ix in range(nx):
            values[iz, ix] = terrain.height(
                xmin + ix * cell_size,
                zmin + iz * cell_size,
            )
    if not np.all(np.isfinite(values)):
        raise ValueError("heightfield contains non-finite values")
    with open(path, "wb") as stream:
        stream.write(struct.pack(
            "<4sIII4f",
            b"G1HF",
            1,
            nx,
            nz,
            xmin,
            zmin,
            cell_size,
            0.0,
        ))
        stream.write(values.tobytes(order="C"))
    return {
        "nx": nx,
        "nz": nz,
        "origin_x": xmin,
        "origin_z": zmin,
        "cell_size": cell_size,
        "exterior_height": 0.0,
    }
