"""Exact foot geometry, terrain queries, and hysteretic contact reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from mm_sonic.joints import ContractError

from .canonical import (
    ISAACLAB_BODY_NAMES,
    CanonicalClip,
    CanonicalTerrainMesh,
)
from .math3d import RigidTransform
from .storage import mesh_digest


CONTACT_PLACEHOLDER_TAG = "contacts-unreconstructed"
CONTACT_RECONSTRUCTION_TAG = "contacts-reconstructed-exact-mesh-v1"
_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_GEOMETRY_EPSILON = 1.0e-12


def _readonly_float32(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iuf":
        raise ContractError(f"{label} must be a real numeric array")
    output = np.ascontiguousarray(source, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ContractError(f"{label} must contain finite float32 values")
    output = output.copy(order="C")
    output.flags.writeable = False
    return output


def _readonly_bool(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "biu":
        raise ContractError(f"{label} must be a boolean array")
    output = np.ascontiguousarray(source, dtype=np.bool_).copy(order="C")
    output.flags.writeable = False
    return output


def _readonly_int32(value: object, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iu":
        raise ContractError(f"{label} must be an integer array")
    output = np.ascontiguousarray(source, dtype=np.int32).copy(order="C")
    output.flags.writeable = False
    return output


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors with normalized wxyz quaternions using broadcasting."""

    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q_vector = q[..., 1:]
    twice_cross = 2.0 * np.cross(q_vector, v)
    return v + q[..., :1] * twice_cross + np.cross(q_vector, twice_cross)


@dataclass(frozen=True)
class SoleGeometry:
    """Named ankle-local sole probes derived from collision geometry."""

    body_names: tuple[str, str]
    corner_positions_body: np.ndarray
    heel_positions_body: np.ndarray = field(init=False)
    toe_positions_body: np.ndarray = field(init=False)
    sole_positions_body: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if tuple(self.body_names) != _FOOT_BODY_NAMES:
            raise ContractError(
                "sole geometry body_names must name left/right ankle-roll links"
            )
        object.__setattr__(self, "body_names", _FOOT_BODY_NAMES)
        corners = np.asarray(self.corner_positions_body)
        if (
            corners.shape != (2, 4, 3)
            or corners.dtype.kind not in "iuf"
            or not np.isfinite(corners).all()
        ):
            raise ContractError(
                "corner_positions_body must be finite with shape [2,4,3]"
            )
        ordered = np.empty((2, 4, 3), dtype=np.float64)
        for foot in range(2):
            order = np.lexsort((corners[foot, :, 1], corners[foot, :, 0]))
            ordered[foot] = corners[foot, order]
            heel = ordered[foot, :2]
            toe = ordered[foot, 2:]
            if (
                float(np.max(heel[:, 0]))
                >= float(np.min(toe[:, 0])) - 1.0e-8
                or abs(float(heel[1, 1] - heel[0, 1])) < 1.0e-8
                or abs(float(toe[1, 1] - toe[0, 1])) < 1.0e-8
            ):
                raise ContractError(
                    f"{self.body_names[foot]} must have two distinct heel "
                    "and two distinct toe probes"
                )
        readonly_corners = _readonly_float32(ordered, "corner_positions_body")
        object.__setattr__(
            self, "corner_positions_body", readonly_corners
        )
        object.__setattr__(
            self,
            "heel_positions_body",
            _readonly_float32(
                np.mean(ordered[:, :2], axis=1), "heel_positions_body"
            ),
        )
        object.__setattr__(
            self,
            "toe_positions_body",
            _readonly_float32(
                np.mean(ordered[:, 2:], axis=1), "toe_positions_body"
            ),
        )
        object.__setattr__(
            self,
            "sole_positions_body",
            _readonly_float32(
                np.mean(ordered, axis=1), "sole_positions_body"
            ),
        )

    @classmethod
    def from_model(cls, model: object) -> "SoleGeometry":
        """Extract the four exact collision-sphere bottoms under each ankle."""

        try:
            import mujoco
        except ImportError as error:
            raise ContractError("SoleGeometry.from_model requires mujoco") from error
        if not isinstance(model, mujoco.MjModel):
            raise ContractError("SoleGeometry.from_model requires a MuJoCo MjModel")
        robot_body_names = tuple(
            str(model.body(index).name) for index in range(1, model.nbody)
        )
        if (
            len(robot_body_names) != len(ISAACLAB_BODY_NAMES)
            or set(robot_body_names) != set(ISAACLAB_BODY_NAMES)
        ):
            raise ContractError(
                "MuJoCo model robot body names must equal the canonical G1 body set"
            )

        ankle_ids: list[int] = []
        for name in _FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id <= 0:
                raise ContractError(f"MuJoCo model is missing named body {name}")
            ankle_ids.append(int(body_id))

        def descended_from(body_id: int, ancestor_id: int) -> bool:
            cursor = int(body_id)
            while cursor > 0:
                if cursor == ancestor_id:
                    return True
                cursor = int(model.body_parentid[cursor])
            return False

        geom_ids_by_foot: list[list[int]] = []
        for body_name, ankle_id in zip(
            _FOOT_BODY_NAMES, ankle_ids, strict=True
        ):
            geom_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if descended_from(int(model.geom_bodyid[geom_id]), ankle_id)
                and int(model.geom_type[geom_id])
                == int(mujoco.mjtGeom.mjGEOM_SPHERE)
                and (
                    int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                )
                and float(model.geom_size[geom_id, 0]) > 0.0
            ]
            if len(geom_ids) != 4:
                raise ContractError(
                    f"{body_name} must have exactly four descendant collision "
                    f"spheres; found {len(geom_ids)}"
                )
            geom_ids_by_foot.append(geom_ids)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        corners = np.empty((2, 4, 3), dtype=np.float64)
        for foot, (ankle_id, geom_ids) in enumerate(
            zip(ankle_ids, geom_ids_by_foot, strict=True)
        ):
            ankle_rotation_world = np.asarray(
                data.xmat[ankle_id], dtype=np.float64
            ).reshape(3, 3)
            ankle_position_world = np.asarray(
                data.xpos[ankle_id], dtype=np.float64
            )
            for corner, geom_id in enumerate(geom_ids):
                center_body = ankle_rotation_world.T @ (
                    np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
                    - ankle_position_world
                )
                corners[foot, corner] = center_body + (
                    0.0,
                    0.0,
                    -float(model.geom_size[geom_id, 0]),
                )
        return cls(body_names=_FOOT_BODY_NAMES, corner_positions_body=corners)

    @property
    def probe_positions_body(self) -> np.ndarray:
        """Return four corners followed by explicit heel and toe probes."""

        probes = np.concatenate(
            (
                self.corner_positions_body,
                self.heel_positions_body[:, None, :],
                self.toe_positions_body[:, None, :],
            ),
            axis=1,
        )
        return _readonly_float32(probes, "probe_positions_body")


@dataclass(frozen=True)
class SurfaceQueryResult:
    closest_point_world: np.ndarray
    surface_normal_world: np.ndarray
    distance_m: np.ndarray
    closest_face_index: np.ndarray
    downward_ray_distance_m: np.ndarray
    downward_ray_normal_world: np.ndarray
    downward_ray_face_index: np.ndarray

    def __post_init__(self) -> None:
        closest = _readonly_float32(
            self.closest_point_world, "closest_point_world"
        )
        if closest.ndim != 2 or closest.shape[1:] != (3,):
            raise ContractError("closest_point_world must have shape [N,3]")
        count = len(closest)
        expected = {
            "surface_normal_world": (count, 3),
            "distance_m": (count,),
            "closest_face_index": (count,),
            "downward_ray_distance_m": (count,),
            "downward_ray_normal_world": (count, 3),
            "downward_ray_face_index": (count,),
        }
        object.__setattr__(self, "closest_point_world", closest)
        for name in ("surface_normal_world", "distance_m"):
            value = _readonly_float32(getattr(self, name), name)
            if value.shape != expected[name]:
                raise ContractError(f"{name} must have shape {expected[name]}")
            object.__setattr__(self, name, value)
        ray_distance = np.ascontiguousarray(
            self.downward_ray_distance_m, dtype=np.float32
        )
        if ray_distance.shape != expected["downward_ray_distance_m"] or np.any(
            np.isnan(ray_distance)
        ):
            raise ContractError(
                "downward_ray_distance_m must have shape [N] without NaN"
            )
        ray_distance = ray_distance.copy()
        ray_distance.flags.writeable = False
        object.__setattr__(self, "downward_ray_distance_m", ray_distance)
        ray_normal = _readonly_float32(
            self.downward_ray_normal_world, "downward_ray_normal_world"
        )
        if ray_normal.shape != expected["downward_ray_normal_world"]:
            raise ContractError(
                "downward_ray_normal_world must have shape [N,3]"
            )
        object.__setattr__(
            self, "downward_ray_normal_world", ray_normal
        )
        for name in ("closest_face_index", "downward_ray_face_index"):
            value = _readonly_int32(getattr(self, name), name)
            if value.shape != expected[name]:
                raise ContractError(f"{name} must have shape {expected[name]}")
            object.__setattr__(self, name, value)


class CanonicalMeshQuery:
    """Exact world-space triangle closest-point, ray, and normal queries."""

    __slots__ = (
        "_triangles",
        "_normals",
        "_face_indices",
        "mesh",
        "mesh_sha256",
        "source_asset_sha256",
        "world_from_terrain",
        "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CanonicalMeshQuery is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        mesh: CanonicalTerrainMesh,
        world_from_terrain: RigidTransform,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(mesh, CanonicalTerrainMesh):
            raise ContractError(
                "CanonicalMeshQuery mesh must be a CanonicalTerrainMesh"
            )
        if not isinstance(world_from_terrain, RigidTransform):
            raise ContractError(
                "CanonicalMeshQuery requires a RigidTransform"
            )
        vertices = np.asarray(mesh.vertices_local)
        faces = np.asarray(mesh.faces)
        valid_faces = np.asarray(mesh.valid_faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or vertices.dtype.kind not in "iuf"
            or not np.isfinite(vertices).all()
            or len(vertices) < 3
        ):
            raise ContractError(
                "mesh vertices must be finite with shape [N>=3,3]"
            )
        if (
            faces.ndim != 2
            or faces.shape[1:] != (3,)
            or faces.dtype.kind not in "iu"
            or len(faces) < 1
        ):
            raise ContractError("mesh faces must be integer triangles")
        if valid_faces.shape != (len(faces),) or valid_faces.dtype.kind != "b":
            raise ContractError(
                "mesh valid_faces must be a boolean mask with shape [M]"
            )
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ContractError("mesh face indices are outside the vertex array")
        if not np.any(valid_faces):
            raise ContractError("mesh must contain at least one valid face")

        vertices_world = np.asarray(
            world_from_terrain.apply_points(vertices), dtype=np.float64
        )
        valid_indices = np.flatnonzero(valid_faces)
        triangles = vertices_world[np.asarray(faces[valid_indices], dtype=np.int64)]
        raw_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normal_lengths = np.linalg.norm(raw_normals, axis=1)
        nondegenerate = normal_lengths > _GEOMETRY_EPSILON
        if not np.any(nondegenerate):
            raise ContractError("mesh has no nondegenerate valid triangle")
        cached_triangles = np.ascontiguousarray(
            triangles[nondegenerate], dtype=np.float64
        ).copy(order="C")
        cached_normals = np.ascontiguousarray(
            raw_normals[nondegenerate]
            / normal_lengths[nondegenerate, None],
            dtype=np.float64,
        ).copy(order="C")
        cached_face_indices = np.ascontiguousarray(
            valid_indices[nondegenerate], dtype=np.int32
        ).copy(order="C")
        cached_triangles.flags.writeable = False
        cached_normals.flags.writeable = False
        cached_face_indices.flags.writeable = False
        self._triangles = cached_triangles
        self._normals = cached_normals
        self._face_indices = cached_face_indices
        self.mesh = mesh
        self.mesh_sha256 = mesh_digest(mesh)
        self.source_asset_sha256 = mesh.source_asset_sha256
        self.world_from_terrain = world_from_terrain
        object.__setattr__(self, "_sealed", True)

    @staticmethod
    def _closest_points_for_one(
        point: np.ndarray, triangles: np.ndarray, normals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        a = triangles[:, 0]
        b = triangles[:, 1]
        c = triangles[:, 2]

        def closest_on_segments(
            start: np.ndarray, end: np.ndarray
        ) -> np.ndarray:
            direction = end - start
            denominator = np.sum(direction * direction, axis=1)
            fraction = np.sum((point - start) * direction, axis=1) / denominator
            return start + np.clip(fraction, 0.0, 1.0)[:, None] * direction

        candidates = [
            closest_on_segments(a, b),
            closest_on_segments(b, c),
            closest_on_segments(c, a),
        ]
        plane_distance = np.sum((point - a) * normals, axis=1)
        projection = point - plane_distance[:, None] * normals
        edge0 = b - a
        edge1 = c - a
        relative = projection - a
        dot00 = np.sum(edge0 * edge0, axis=1)
        dot01 = np.sum(edge0 * edge1, axis=1)
        dot11 = np.sum(edge1 * edge1, axis=1)
        dot20 = np.sum(relative * edge0, axis=1)
        dot21 = np.sum(relative * edge1, axis=1)
        denominator = dot00 * dot11 - dot01 * dot01
        bary_v = (dot11 * dot20 - dot01 * dot21) / denominator
        bary_w = (dot00 * dot21 - dot01 * dot20) / denominator
        bary_u = 1.0 - bary_v - bary_w
        inside = (
            (bary_u >= -1.0e-12)
            & (bary_v >= -1.0e-12)
            & (bary_w >= -1.0e-12)
        )
        candidates.append(
            np.where(inside[:, None], projection, np.inf)
        )
        candidate_array = np.stack(candidates, axis=1)
        squared = np.sum(
            (candidate_array - point) ** 2, axis=2
        )
        candidate_index = np.argmin(squared, axis=1)
        triangle_index = np.arange(len(triangles))
        return (
            candidate_array[triangle_index, candidate_index],
            squared[triangle_index, candidate_index],
        )

    def query(self, points_world: object) -> SurfaceQueryResult:
        points = np.asarray(points_world, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.isfinite(points).all()
        ):
            raise ContractError("query points must be finite with shape [N,3]")
        count = len(points)
        closest_points = np.empty((count, 3), dtype=np.float64)
        closest_normals = np.empty((count, 3), dtype=np.float64)
        closest_distances = np.empty(count, dtype=np.float64)
        closest_faces = np.empty(count, dtype=np.int32)
        ray_distances = np.full(count, np.inf, dtype=np.float64)
        ray_normals = np.zeros((count, 3), dtype=np.float64)
        ray_faces = np.full(count, -1, dtype=np.int32)

        triangles = self._triangles
        normals = self._normals
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        direction = np.array((0.0, 0.0, -1.0), dtype=np.float64)
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.sum(edge1 * h, axis=1)
        determinant_valid = np.abs(determinant) > _GEOMETRY_EPSILON
        inverse_determinant = np.zeros_like(determinant)
        inverse_determinant[determinant_valid] = (
            1.0 / determinant[determinant_valid]
        )

        for index, point in enumerate(points):
            triangle_closest, triangle_squared = self._closest_points_for_one(
                point, triangles, normals
            )
            best = int(np.argmin(triangle_squared))
            closest_points[index] = triangle_closest[best]
            closest_normals[index] = normals[best]
            closest_distances[index] = np.sqrt(triangle_squared[best])
            closest_faces[index] = self._face_indices[best]

            relative = point - triangles[:, 0]
            bary_u = inverse_determinant * np.sum(relative * h, axis=1)
            q = np.cross(relative, edge1)
            bary_v = inverse_determinant * np.sum(direction * q, axis=1)
            ray_t = inverse_determinant * np.sum(edge2 * q, axis=1)
            hits = (
                determinant_valid
                & (bary_u >= -1.0e-12)
                & (bary_v >= -1.0e-12)
                & (bary_u + bary_v <= 1.0 + 1.0e-12)
                & (ray_t >= -1.0e-12)
            )
            if np.any(hits):
                hit_indices = np.flatnonzero(hits)
                hit = int(hit_indices[np.argmin(ray_t[hit_indices])])
                ray_distances[index] = max(0.0, float(ray_t[hit]))
                ray_normals[index] = normals[hit]
                ray_faces[index] = self._face_indices[hit]

        return SurfaceQueryResult(
            closest_point_world=closest_points,
            surface_normal_world=closest_normals,
            distance_m=closest_distances,
            closest_face_index=closest_faces,
            downward_ray_distance_m=ray_distances,
            downward_ray_normal_world=ray_normals,
            downward_ray_face_index=ray_faces,
        )


@dataclass(frozen=True)
class ContactConfig:
    geometry: SoleGeometry
    enter_distance_m: float = 0.012
    leave_distance_m: float = 0.025
    enter_tangential_speed_m_s: float = 0.12
    leave_tangential_speed_m_s: float = 0.20
    enter_angular_speed_rad_s: float = 0.75
    leave_angular_speed_rad_s: float = 1.25
    minimum_surface_normal_z: float = 0.50

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, SoleGeometry):
            raise ContractError("ContactConfig.geometry must be SoleGeometry")
        values = (
            self.enter_distance_m,
            self.leave_distance_m,
            self.enter_tangential_speed_m_s,
            self.leave_tangential_speed_m_s,
            self.enter_angular_speed_rad_s,
            self.leave_angular_speed_rad_s,
            self.minimum_surface_normal_z,
        )
        if not all(
            type(value) in (int, float) and np.isfinite(value)
            for value in values
        ):
            raise ContractError("contact thresholds must be finite numbers")
        if (
            self.enter_distance_m <= 0.0
            or self.leave_distance_m <= self.enter_distance_m
            or self.enter_tangential_speed_m_s <= 0.0
            or self.leave_tangential_speed_m_s
            <= self.enter_tangential_speed_m_s
            or self.enter_angular_speed_rad_s <= 0.0
            or self.leave_angular_speed_rad_s
            <= self.enter_angular_speed_rad_s
            or not 0.0 <= self.minimum_surface_normal_z < 1.0
        ):
            raise ContractError(
                "contact leave thresholds must exceed positive enter "
                "thresholds and normal_z must be in [0,1)"
            )


@dataclass(frozen=True)
class ContactReconstruction:
    clip_id: str
    sole_position_world: np.ndarray
    sole_quaternion_world_wxyz: np.ndarray
    heel_position_world: np.ndarray
    toe_position_world: np.ndarray
    contact: np.ndarray
    contact_confidence: np.ndarray

    def __post_init__(self) -> None:
        if type(self.clip_id) is not str or not self.clip_id:
            raise ContractError("contact reconstruction clip_id must be nonempty")
        sole = _readonly_float32(
            self.sole_position_world, "sole_position_world"
        )
        if sole.ndim != 3 or sole.shape[1:] != (2, 3):
            raise ContractError("sole_position_world must have shape [T,2,3]")
        frame_count = len(sole)
        expected = {
            "sole_quaternion_world_wxyz": (frame_count, 2, 4),
            "heel_position_world": (frame_count, 2, 3),
            "toe_position_world": (frame_count, 2, 3),
            "contact": (frame_count, 2),
            "contact_confidence": (frame_count, 2),
        }
        object.__setattr__(self, "sole_position_world", sole)
        for name in (
            "sole_quaternion_world_wxyz",
            "heel_position_world",
            "toe_position_world",
            "contact_confidence",
        ):
            value = _readonly_float32(getattr(self, name), name)
            if value.shape != expected[name]:
                raise ContractError(f"{name} must have shape {expected[name]}")
            object.__setattr__(self, name, value)
        contact = _readonly_bool(self.contact, "contact")
        if contact.shape != expected["contact"]:
            raise ContractError(f"contact must have shape {expected['contact']}")
        object.__setattr__(self, "contact", contact)
        quaternion_norm = np.linalg.norm(
            self.sole_quaternion_world_wxyz, axis=-1
        )
        if not np.allclose(quaternion_norm, 1.0, atol=1.0e-5):
            raise ContractError(
                "sole_quaternion_world_wxyz must be normalized"
            )
        if np.any(self.contact_confidence < 0.0) or np.any(
            self.contact_confidence > 1.0
        ):
            raise ContractError("contact_confidence must be in [0,1]")

    def apply(self, clip: CanonicalClip) -> CanonicalClip:
        """Replace only provisional contact fields and their provenance tag."""

        if not isinstance(clip, CanonicalClip):
            raise ContractError("ContactReconstruction.apply requires CanonicalClip")
        clip.validate()
        if clip.clip_id != self.clip_id or clip.frame_count != len(self.contact):
            raise ContractError(
                "contact reconstruction belongs to a different canonical clip"
            )
        tags = tuple(
            tag
            for tag in clip.action_tags
            if tag
            not in (CONTACT_PLACEHOLDER_TAG, CONTACT_RECONSTRUCTION_TAG)
        )
        reconstructed = replace(
            clip,
            sole_position_world=self.sole_position_world,
            sole_quaternion_world_wxyz=self.sole_quaternion_world_wxyz,
            heel_position_world=self.heel_position_world,
            toe_position_world=self.toe_position_world,
            contact=np.asarray(self.contact, dtype=np.float32),
            contact_confidence=self.contact_confidence,
            action_tags=(*tags, CONTACT_RECONSTRUCTION_TAG),
        )
        reconstructed.validate()
        return reconstructed


def reconstruct_contacts(
    clip: CanonicalClip,
    query: CanonicalMeshQuery | None,
    config: ContactConfig,
) -> ContactReconstruction:
    """Reconstruct full-sole contact from exact probes, motion, and terrain."""

    if not isinstance(clip, CanonicalClip):
        raise ContractError("reconstruct_contacts requires a CanonicalClip")
    clip.validate()
    if query is None:
        raise ContractError(
            "contact reconstruction requires an explicit surface query"
        )
    if not isinstance(query, CanonicalMeshQuery):
        raise ContractError("query must be a CanonicalMeshQuery")
    if not isinstance(config, ContactConfig):
        raise ContractError("config must be a ContactConfig")

    geometry = config.geometry
    try:
        body_indices = np.asarray(
            [clip.body_names.index(name) for name in geometry.body_names],
            dtype=np.int64,
        )
    except ValueError as error:
        raise ContractError(
            "canonical clip is missing a named ankle-roll body"
        ) from error
    body_position = np.asarray(
        clip.body_position_world[:, body_indices], dtype=np.float64
    )
    body_quaternion = np.asarray(
        clip.body_quaternion_world_wxyz[:, body_indices], dtype=np.float64
    )
    body_linear_velocity = np.asarray(
        clip.body_linear_velocity_world[:, body_indices], dtype=np.float64
    )
    body_angular_velocity = np.asarray(
        clip.body_angular_velocity_world[:, body_indices], dtype=np.float64
    )

    sole_offset = np.asarray(geometry.sole_positions_body, dtype=np.float64)
    heel_offset = np.asarray(geometry.heel_positions_body, dtype=np.float64)
    toe_offset = np.asarray(geometry.toe_positions_body, dtype=np.float64)
    probe_offset = np.asarray(geometry.probe_positions_body, dtype=np.float64)
    quaternion_expanded = body_quaternion[:, :, None, :]
    rotated_probes = _rotate_wxyz(
        quaternion_expanded,
        np.broadcast_to(
            probe_offset[None, ...],
            (clip.frame_count, 2, probe_offset.shape[1], 3),
        ),
    )
    probe_world = body_position[:, :, None, :] + rotated_probes
    surface = query.query(probe_world.reshape((-1, 3)))
    probe_count = probe_offset.shape[1]
    closest_distance = np.asarray(surface.distance_m).reshape(
        (clip.frame_count, 2, probe_count)
    )
    closest_normal = np.asarray(surface.surface_normal_world).reshape(
        (clip.frame_count, 2, probe_count, 3)
    )
    ray_distance = np.asarray(surface.downward_ray_distance_m).reshape(
        (clip.frame_count, 2, probe_count)
    )
    ray_normal = np.asarray(surface.downward_ray_normal_world).reshape(
        (clip.frame_count, 2, probe_count, 3)
    )
    probe_velocity = body_linear_velocity[:, :, None, :] + np.cross(
        body_angular_velocity[:, :, None, :], rotated_probes
    )
    angular_speed = np.linalg.norm(body_angular_velocity, axis=-1)

    distance = np.array(closest_distance, copy=True)
    normal = np.array(closest_normal, copy=True)
    maximum_tangential_speed = np.empty(
        (clip.frame_count, 2), dtype=np.float64
    )
    contact = np.zeros((clip.frame_count, 2), dtype=np.bool_)
    for foot in range(2):
        active = False
        for frame in range(clip.frame_count):
            if active:
                distance_threshold = config.leave_distance_m
                tangential_threshold = config.leave_tangential_speed_m_s
                angular_threshold = config.leave_angular_speed_rad_s
            else:
                distance_threshold = config.enter_distance_m
                tangential_threshold = config.enter_tangential_speed_m_s
                angular_threshold = config.enter_angular_speed_rad_s
            ray_support = (
                np.isfinite(ray_distance[frame, foot])
                & (ray_distance[frame, foot] <= distance_threshold)
                & (
                    ray_normal[frame, foot, :, 2]
                    >= config.minimum_surface_normal_z
                )
            )
            closest_support = (
                closest_distance[frame, foot] <= distance_threshold
            ) & (
                closest_normal[frame, foot, :, 2]
                >= config.minimum_surface_normal_z
            )
            distance[frame, foot] = np.where(
                ray_support,
                ray_distance[frame, foot],
                closest_distance[frame, foot],
            )
            normal[frame, foot] = np.where(
                ray_support[:, None],
                ray_normal[frame, foot],
                closest_normal[frame, foot],
            )
            normal_velocity = np.sum(
                probe_velocity[frame, foot] * normal[frame, foot], axis=-1
            )
            tangential_velocity = probe_velocity[frame, foot] - (
                normal_velocity[:, None] * normal[frame, foot]
            )
            maximum_tangential_speed[frame, foot] = float(
                np.max(np.linalg.norm(tangential_velocity, axis=-1))
            )
            active = bool(
                np.all(ray_support | closest_support)
                and maximum_tangential_speed[frame, foot]
                <= tangential_threshold
                and angular_speed[frame, foot] <= angular_threshold
            )
            contact[frame, foot] = active

    distance_score = np.clip(
        1.0 - distance / config.leave_distance_m, 0.0, 1.0
    )
    normal_score = np.clip(
        (
            normal[..., 2] - config.minimum_surface_normal_z
        )
        / (1.0 - config.minimum_surface_normal_z),
        0.0,
        1.0,
    )
    support_score = np.mean(distance_score * normal_score, axis=2)
    tangential_score = np.clip(
        1.0
        - maximum_tangential_speed
        / config.leave_tangential_speed_m_s,
        0.0,
        1.0,
    )
    angular_score = np.clip(
        1.0 - angular_speed / config.leave_angular_speed_rad_s,
        0.0,
        1.0,
    )
    confidence = np.clip(
        support_score * tangential_score * angular_score, 0.0, 1.0
    )
    if not np.isfinite(confidence).all():
        raise ContractError("contact reconstruction produced nonfinite confidence")

    sole_position = body_position + _rotate_wxyz(
        body_quaternion,
        np.broadcast_to(
            sole_offset[None, ...], (clip.frame_count, 2, 3)
        ),
    )
    heel_position = body_position + _rotate_wxyz(
        body_quaternion,
        np.broadcast_to(
            heel_offset[None, ...], (clip.frame_count, 2, 3)
        ),
    )
    toe_position = body_position + _rotate_wxyz(
        body_quaternion,
        np.broadcast_to(
            toe_offset[None, ...], (clip.frame_count, 2, 3)
        ),
    )
    return ContactReconstruction(
        clip_id=clip.clip_id,
        sole_position_world=sole_position,
        sole_quaternion_world_wxyz=body_quaternion,
        heel_position_world=heel_position,
        toe_position_world=toe_position,
        contact=contact,
        contact_confidence=confidence,
    )


__all__ = (
    "CONTACT_RECONSTRUCTION_TAG",
    "CanonicalMeshQuery",
    "ContactConfig",
    "ContactReconstruction",
    "SoleGeometry",
    "SurfaceQueryResult",
    "reconstruct_contacts",
)
