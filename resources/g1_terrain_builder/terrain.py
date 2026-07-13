from dataclasses import dataclass
import os
import pickle
import struct
import warnings

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import cKDTree


LOOKAHEAD = np.array([0.25, 0.50, 0.75, 1.00], np.float64)
USD_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/object_usd"
RECON_DIR = "/home/ubuntu/datasets/GRAIL/data/curb/recon"


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


def _load_usd_mesh(base: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(USD_DIR, base + ".usd")
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise FileNotFoundError(f"unable to open GRAIL terrain USD: {path}")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            counts = np.asarray(
                mesh.GetFaceVertexCountsAttr().Get(), np.int32)
            indices = np.asarray(
                mesh.GetFaceVertexIndicesAttr().Get(), np.int32)
            if points.ndim != 2 or points.shape[1:] != (3,):
                raise ValueError(f"invalid GRAIL mesh points in {path}")
            if counts.ndim != 1 or indices.ndim != 1:
                raise ValueError(f"invalid GRAIL mesh topology in {path}")
            _validate_mesh_topology(points, counts, indices)
            return points, counts, indices
    raise ValueError(f"no mesh found in GRAIL terrain USD: {path}")


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


def _densify_faces(
    vertices: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
    resolution: int = 6,
) -> np.ndarray:
    vertices = np.asarray(vertices, np.float64)
    face_counts = np.asarray(face_counts, np.int32)
    face_indices = np.asarray(face_indices, np.int32)
    _validate_mesh_topology(vertices, face_counts, face_indices)
    if resolution < 2:
        raise ValueError("face sampling resolution must be at least 2")

    dense = [vertices]
    offset = 0
    uv = np.linspace(0.0, 1.0, resolution)
    u, v = np.meshgrid(uv, uv)
    u = u.ravel()
    v = v.ravel()
    for count in face_counts:
        indices = face_indices[offset:offset + count]
        offset += count
        if count == 4:
            a, b, c, d = vertices[indices]
            bottom = a[None] * (1.0 - u[:, None]) + b[None] * u[:, None]
            top = d[None] * (1.0 - u[:, None]) + c[None] * u[:, None]
            dense.append(bottom * (1.0 - v[:, None]) + top * v[:, None])
            continue
        if count < 3:
            raise ValueError(f"terrain face has only {count} vertices")
        for index in range(1, count - 1):
            a, b, c = vertices[[indices[0], indices[index], indices[index + 1]]]
            samples = []
            for ib in range(resolution + 1):
                for ic in range(resolution + 1 - ib):
                    wb = ib / resolution
                    wc = ic / resolution
                    samples.append(a * (1.0 - wb - wc) + b * wb + c * wc)
            dense.append(np.asarray(samples, np.float64))
    return np.concatenate(dense, axis=0)


def _mujoco_to_holden(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    return np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))


class GrailTerrain:
    def __init__(
        self,
        render_vertices: np.ndarray,
        face_counts: np.ndarray,
        face_indices: np.ndarray,
        query_points: np.ndarray,
        radius: float = 0.14,
    ):
        self._vertices = np.asarray(render_vertices, np.float64)
        self._face_counts = np.asarray(face_counts, np.int32)
        self._face_indices = np.asarray(face_indices, np.int32)
        self._points = np.asarray(query_points, np.float64)
        _validate_mesh_topology(
            self._vertices, self._face_counts, self._face_indices)
        if (
            self._points.ndim != 2
            or self._points.shape[1:] != (3,)
            or not len(self._points)
        ):
            raise ValueError(
                "GRAIL terrain query points must have non-empty shape (N, 3)")
        if (
            not np.all(np.isfinite(self._vertices))
            or not np.all(np.isfinite(self._points))
        ):
            raise ValueError("GRAIL terrain points must be finite")
        if radius <= 0.0:
            raise ValueError(
                "GRAIL terrain requires points and a positive query radius")
        self._radius = float(radius)
        self._tree = cKDTree(self._points[:, (0, 2)])
        self._max_height = float(self._points[:, 1].max())

    @classmethod
    def from_base(cls, base: str) -> "GrailTerrain":
        vertices, face_counts, face_indices = _load_usd_mesh(base)
        rotation, translation = _object_pose0(base)
        dense = _densify_faces(vertices, face_counts, face_indices)
        world_vertices = (rotation @ vertices.T).T + translation
        world_dense = (rotation @ dense.T).T + translation
        return cls(
            _mujoco_to_holden(world_vertices),
            face_counts,
            face_indices,
            _mujoco_to_holden(world_dense),
        )

    def height(self, x: float, z: float) -> float:
        if not np.isfinite(x) or not np.isfinite(z):
            raise ValueError("terrain query coordinates must be finite")
        indices = self._tree.query_ball_point([x, z], self._radius)
        if not indices:
            return 0.0
        return float(self._points[indices, 1].max())

    def footprint(self) -> dict:
        top = self._points[self._points[:, 1] > self._max_height - 0.02]
        return {
            "x": (float(top[:, 0].min()), float(top[:, 0].max())),
            "z": (float(top[:, 2].min()), float(top[:, 2].max())),
            "height": self._max_height,
        }

    def xz_bounds(self) -> tuple[float, float, float, float]:
        """Return complete render/query bounds as xmin, xmax, zmin, zmax."""
        return (
            float(min(self._vertices[:, 0].min(), self._points[:, 0].min())),
            float(max(self._vertices[:, 0].max(), self._points[:, 0].max())),
            float(min(self._vertices[:, 2].min(), self._points[:, 2].min())),
            float(max(self._vertices[:, 2].max(), self._points[:, 2].max())),
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
