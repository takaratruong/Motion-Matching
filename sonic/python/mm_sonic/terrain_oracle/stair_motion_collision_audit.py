"""Exact-mesh full-G1 collision audit for reconstructed stair motion."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .math3d import RigidTransform
from .source_grail import _load_usd_mesh
from .stitch import StitchedMotion
from .terrain_mesh import TerrainMeshIndex


_AUDIT_MODEL_CACHE: OrderedDict[tuple[object, ...], tuple[object, object]] = (
    OrderedDict()
)
_AUDIT_MODEL_CACHE_SIZE = 2


@dataclass(frozen=True)
class GeometryClasses:
    """Terrain, allowed-foot, and forbidden-body geometry identifiers."""

    terrain_geom_ids: frozenset[int]
    foot_geom_ids: frozenset[int]
    forbidden_body_geom_ids: frozenset[int]


@dataclass(frozen=True)
class FrameCollisionObservation:
    """Penetrating terrain contacts observed in one kinematic frame."""

    maximum_foot_penetration_m: float
    maximum_forbidden_body_penetration_m: float
    foot_contact_count: int
    forbidden_body_contact_count: int


@dataclass(frozen=True)
class StairMotionCollisionAuditResult:
    """JSON-ready full-motion terrain collision audit."""

    accepted: bool
    frame_count: int
    maximum_foot_penetration_threshold_m: float
    maximum_forbidden_body_penetration_threshold_m: float
    maximum_foot_penetration_m: float
    maximum_forbidden_body_penetration_m: float
    foot_contact_count: int
    forbidden_body_contact_count: int
    foot_collision_frame_indices: tuple[int, ...]
    forbidden_body_collision_frame_indices: tuple[int, ...]
    foot_threshold_exceedance_frame_indices: tuple[int, ...]
    forbidden_body_threshold_exceedance_frame_indices: tuple[int, ...]
    per_frame_max_foot_penetration_m: tuple[float, ...]
    per_frame_max_forbidden_body_penetration_m: tuple[float, ...]
    per_frame_foot_contact_count: tuple[int, ...]
    per_frame_forbidden_body_contact_count: tuple[int, ...]
    terrain_geom_count: int
    foot_geom_count: int
    forbidden_body_geom_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "frame_count": self.frame_count,
            "thresholds": {
                "maximum_foot_penetration_m": (
                    self.maximum_foot_penetration_threshold_m
                ),
                "maximum_forbidden_body_penetration_m": (
                    self.maximum_forbidden_body_penetration_threshold_m
                ),
            },
            "maximum_foot_penetration_m": self.maximum_foot_penetration_m,
            "maximum_forbidden_body_penetration_m": (
                self.maximum_forbidden_body_penetration_m
            ),
            "foot_contact_count": self.foot_contact_count,
            "forbidden_body_contact_count": (
                self.forbidden_body_contact_count
            ),
            "foot_collision_frame_indices": list(
                self.foot_collision_frame_indices
            ),
            "forbidden_body_collision_frame_indices": list(
                self.forbidden_body_collision_frame_indices
            ),
            "foot_threshold_exceedance_frame_indices": list(
                self.foot_threshold_exceedance_frame_indices
            ),
            "forbidden_body_threshold_exceedance_frame_indices": list(
                self.forbidden_body_threshold_exceedance_frame_indices
            ),
            "per_frame_max_foot_penetration_m": list(
                self.per_frame_max_foot_penetration_m
            ),
            "per_frame_max_forbidden_body_penetration_m": list(
                self.per_frame_max_forbidden_body_penetration_m
            ),
            "per_frame_foot_contact_count": list(
                self.per_frame_foot_contact_count
            ),
            "per_frame_forbidden_body_contact_count": list(
                self.per_frame_forbidden_body_contact_count
            ),
            "geometry_counts": {
                "terrain": self.terrain_geom_count,
                "foot": self.foot_geom_count,
                "forbidden_body": self.forbidden_body_geom_count,
            },
        }


def _descends_from(
    body_id: int,
    ancestor_id: int,
    body_parent_ids: Sequence[int],
) -> bool:
    cursor = int(body_id)
    while cursor != 0:
        if cursor == ancestor_id:
            return True
        cursor = int(body_parent_ids[cursor])
    return cursor == ancestor_id


def classify_geometry_ids(
    *,
    geom_names: Sequence[str | None],
    geom_body_ids: Sequence[int],
    body_names: Sequence[str | None],
    body_parent_ids: Sequence[int],
    foot_body_names: tuple[str, str] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ),
) -> GeometryClasses:
    """Classify exact terrain and robot geoms using the G1 body hierarchy."""

    terrain = frozenset(
        index
        for index, name in enumerate(geom_names)
        if name is not None and str(name).startswith("terrain_")
    )
    foot_body_ids = tuple(
        index
        for index, name in enumerate(body_names)
        if name in foot_body_names
    )
    feet = frozenset(
        geom_id
        for geom_id, body_id in enumerate(geom_body_ids)
        if body_id != 0
        and any(
            _descends_from(
                int(body_id),
                foot_body_id,
                body_parent_ids,
            )
            for foot_body_id in foot_body_ids
        )
    )
    forbidden = frozenset(
        geom_id
        for geom_id, body_id in enumerate(geom_body_ids)
        if body_id != 0 and geom_id not in feet
    )
    return GeometryClasses(terrain, feet, forbidden)


def classify_terrain_contacts(
    contacts: Sequence[tuple[int, int, float]],
    *,
    terrain_geom_ids: frozenset[int],
    foot_geom_ids: frozenset[int],
    forbidden_body_geom_ids: frozenset[int],
) -> FrameCollisionObservation:
    """Reduce MuJoCo-style geom pairs and signed distances for one frame."""

    foot_maximum = 0.0
    body_maximum = 0.0
    foot_count = 0
    body_count = 0
    for geom1, geom2, signed_distance in contacts:
        if geom1 in terrain_geom_ids and geom2 not in terrain_geom_ids:
            robot_geom = geom2
        elif geom2 in terrain_geom_ids and geom1 not in terrain_geom_ids:
            robot_geom = geom1
        else:
            continue
        penetration = max(0.0, -float(signed_distance))
        if penetration <= 0.0:
            continue
        if robot_geom in foot_geom_ids:
            foot_count += 1
            foot_maximum = max(foot_maximum, penetration)
        elif robot_geom in forbidden_body_geom_ids:
            body_count += 1
            body_maximum = max(body_maximum, penetration)
    return FrameCollisionObservation(
        maximum_foot_penetration_m=foot_maximum,
        maximum_forbidden_body_penetration_m=body_maximum,
        foot_contact_count=foot_count,
        forbidden_body_contact_count=body_count,
    )


def aggregate_collision_frames(
    frames: Sequence[FrameCollisionObservation],
    *,
    maximum_foot_penetration_m: float = 0.005,
    maximum_forbidden_body_penetration_m: float = 1.0e-6,
    terrain_geom_count: int = 0,
    foot_geom_count: int = 0,
    forbidden_body_geom_count: int = 0,
) -> StairMotionCollisionAuditResult:
    """Aggregate per-frame contacts under simple foot/body acceptance limits."""

    values = tuple(frames)
    foot_penetration = tuple(
        float(value.maximum_foot_penetration_m) for value in values
    )
    body_penetration = tuple(
        float(value.maximum_forbidden_body_penetration_m)
        for value in values
    )
    foot_counts = tuple(int(value.foot_contact_count) for value in values)
    body_counts = tuple(
        int(value.forbidden_body_contact_count) for value in values
    )
    foot_maximum = max(foot_penetration, default=0.0)
    body_maximum = max(body_penetration, default=0.0)
    foot_limit = float(maximum_foot_penetration_m)
    body_limit = float(maximum_forbidden_body_penetration_m)
    return StairMotionCollisionAuditResult(
        accepted=foot_maximum <= foot_limit and body_maximum <= body_limit,
        frame_count=len(values),
        maximum_foot_penetration_threshold_m=foot_limit,
        maximum_forbidden_body_penetration_threshold_m=body_limit,
        maximum_foot_penetration_m=foot_maximum,
        maximum_forbidden_body_penetration_m=body_maximum,
        foot_contact_count=sum(foot_counts),
        forbidden_body_contact_count=sum(body_counts),
        foot_collision_frame_indices=tuple(
            index for index, count in enumerate(foot_counts) if count > 0
        ),
        forbidden_body_collision_frame_indices=tuple(
            index for index, count in enumerate(body_counts) if count > 0
        ),
        foot_threshold_exceedance_frame_indices=tuple(
            index
            for index, penetration in enumerate(foot_penetration)
            if penetration > foot_limit
        ),
        forbidden_body_threshold_exceedance_frame_indices=tuple(
            index
            for index, penetration in enumerate(body_penetration)
            if penetration > body_limit
        ),
        per_frame_max_foot_penetration_m=foot_penetration,
        per_frame_max_forbidden_body_penetration_m=body_penetration,
        per_frame_foot_contact_count=foot_counts,
        per_frame_forbidden_body_contact_count=body_counts,
        terrain_geom_count=int(terrain_geom_count),
        foot_geom_count=int(foot_geom_count),
        forbidden_body_geom_count=int(forbidden_body_geom_count),
    )


def _model_geometry_classes(model: object, mujoco: object) -> GeometryClasses:
    geom_names = tuple(
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            index,
        )
        for index in range(model.ngeom)
    )
    body_names = tuple(
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            index,
        )
        for index in range(model.nbody)
    )
    return classify_geometry_ids(
        geom_names=geom_names,
        geom_body_ids=tuple(int(value) for value in model.geom_bodyid),
        body_names=body_names,
        body_parent_ids=tuple(int(value) for value in model.body_parentid),
    )


_TRIANGLE_PRISM_FACES = np.asarray(
    (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4),
        (0, 4, 3),
        (1, 2, 5),
        (1, 5, 4),
        (2, 0, 3),
        (2, 3, 5),
    ),
    dtype=np.int32,
)


def _triangle_prism_vertices(
    triangle: object,
    *,
    thickness_m: float,
) -> np.ndarray:
    """Return a thin convex prism whose mid-surface is one mesh triangle."""

    points = np.asarray(triangle, dtype=np.float64)
    thickness = float(thickness_m)
    if points.shape != (3, 3):
        raise ValueError("terrain triangle must have shape [3,3]")
    if not np.isfinite(points).all():
        raise ValueError("terrain triangle must be finite")
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("terrain face prism thickness must be positive")
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("terrain triangle is degenerate")
    offset = 0.5 * thickness * normal / norm
    return np.concatenate((points - offset, points + offset), axis=0)


def _build_collision_scene_model(
    model_spec: object,
    mesh: object,
    transform: RigidTransform,
    mujoco: object,
    *,
    terrain_face_thickness_m: float = 0.001,
) -> object:
    """Compile exact face-wise terrain collision prisms and the G1 model.

    A single MuJoCo mesh geom collides as its convex hull.  That would bridge
    all stair treads with one diagonal hull and report large false
    penetrations.  One thin convex prism per source triangle retains the
    original non-convex staircase surface while still using MuJoCo's robust
    convex collision pipeline.
    """

    spec = model_spec.copy()
    faces = np.asarray(mesh.faces[mesh.valid_faces], dtype=np.int32)
    vertices = np.asarray(mesh.vertices_local, dtype=np.float64)
    for face_index, face in enumerate(faces):
        name = f"terrain_oracle_face_{face_index:06d}"
        prism = _triangle_prism_vertices(
            vertices[face],
            thickness_m=terrain_face_thickness_m,
        )
        spec.add_mesh(
            name=name,
            uservert=prism.ravel(),
            userface=_TRIANGLE_PRISM_FACES.ravel(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
        )
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            pos=np.asarray(transform.translation_world, dtype=np.float64),
            quat=np.asarray(
                transform.quaternion_world_from_local_wxyz,
                dtype=np.float64,
            ),
            contype=1,
            conaffinity=1,
            rgba=[0.20, 0.34, 0.24, 1.0],
        )
    for geom in spec.geoms:
        if geom.parent.name != "world":
            geom.contype = 1
            geom.conaffinity = 1
    return spec.compile()


def audit_stair_motion_collisions(
    motion: StitchedMotion,
    *,
    model_path: str | Path,
    archive_path: str | Path | None = None,
    target_clip_index: int | None = None,
    target_mesh: TerrainMeshIndex | None = None,
    joint_names: Sequence[str] | None = None,
    maximum_foot_penetration_m: float = 0.005,
    maximum_forbidden_body_penetration_m: float = 1.0e-6,
) -> StairMotionCollisionAuditResult:
    """Audit every kinematic frame against an exact target terrain mesh.

    Archive-backed callers may retain ``target_clip_index``.  Generic runtime
    callers pass ``target_mesh`` and ``joint_names`` directly, so collision
    auditing has no scene or target-motion identity dependency.
    """

    import mujoco
    import zarr

    if len(motion.root_position_world) == 0:
        raise ValueError("stitched motion is empty")
    archive = None
    if target_mesh is None or joint_names is None:
        if archive_path is None:
            raise ValueError(
                "archive_path is required unless target_mesh and joint_names "
                "are supplied"
            )
        archive = zarr.open_group(str(Path(archive_path)), mode="r")
    if target_mesh is None:
        if target_clip_index is None:
            raise ValueError("target_mesh or target_clip_index is required")
        target = int(target_clip_index)
        terrain_path = Path(str(archive["terrain_usd_path"][target]))
        terrain_transform = RigidTransform(
            np.asarray(archive["terrain_position_env"][target]),
            np.asarray(archive["terrain_rotation_env_wxyz"][target]),
        )
        mesh = _load_usd_mesh(
            terrain_path,
            source_asset_sha256="0" * 64,
        )
        target_mesh = TerrainMeshIndex(mesh, terrain_transform)
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    if joint_names is None:
        assert archive is not None
        joint_names = tuple(str(value) for value in archive["joint_names"][:])
    else:
        joint_names = tuple(str(value) for value in joint_names)

    mesh_digest = hashlib.sha256()
    mesh_digest.update(
        np.ascontiguousarray(target_mesh.mesh.vertices_local).tobytes()
    )
    mesh_digest.update(np.ascontiguousarray(target_mesh.mesh.faces).tobytes())
    mesh_digest.update(
        np.ascontiguousarray(target_mesh.mesh.valid_faces).tobytes()
    )
    resolved_model_path = Path(model_path).resolve()
    cache_key = (
        str(resolved_model_path),
        mesh_digest.hexdigest(),
        tuple(
            float(value)
            for value in target_mesh.world_from_terrain.translation_world
        ),
        tuple(
            float(value)
            for value in target_mesh.world_from_terrain.quaternion_world_from_local_wxyz
        ),
    )
    cached = _AUDIT_MODEL_CACHE.get(cache_key)
    if cached is None:
        spec = mujoco.MjSpec.from_file(str(resolved_model_path))
        model = _build_collision_scene_model(
            spec,
            target_mesh.mesh,
            target_mesh.world_from_terrain,
            mujoco,
        )
        classes = _model_geometry_classes(model, mujoco)
        _AUDIT_MODEL_CACHE[cache_key] = (model, classes)
        _AUDIT_MODEL_CACHE.move_to_end(cache_key)
        while len(_AUDIT_MODEL_CACHE) > _AUDIT_MODEL_CACHE_SIZE:
            _AUDIT_MODEL_CACHE.popitem(last=False)
    else:
        model, classes = cached
        _AUDIT_MODEL_CACHE.move_to_end(cache_key)

    root_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "floating_base_joint",
    )
    if root_id < 0:
        raise ValueError("G1 model has no floating_base_joint")
    root_address = int(model.jnt_qposadr[root_id])
    joint_addresses: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {name}")
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))

    data = mujoco.MjData(model)
    observations: list[FrameCollisionObservation] = []
    for frame in range(len(motion.root_position_world)):
        data.qpos[root_address : root_address + 3] = (
            motion.root_position_world[frame]
        )
        data.qpos[root_address + 3 : root_address + 7] = (
            motion.root_quaternion_world_wxyz[frame]
        )
        for address, value in zip(
            joint_addresses,
            motion.joint_position[frame],
            strict=True,
        ):
            data.qpos[address] = value
        data.time = frame / float(motion.fps)
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)
        observations.append(
            classify_terrain_contacts(
                tuple(
                    (
                        int(data.contact[index].geom1),
                        int(data.contact[index].geom2),
                        float(data.contact[index].dist),
                    )
                    for index in range(data.ncon)
                ),
                terrain_geom_ids=classes.terrain_geom_ids,
                foot_geom_ids=classes.foot_geom_ids,
                forbidden_body_geom_ids=(
                    classes.forbidden_body_geom_ids
                ),
            )
        )
    return aggregate_collision_frames(
        observations,
        maximum_foot_penetration_m=maximum_foot_penetration_m,
        maximum_forbidden_body_penetration_m=(
            maximum_forbidden_body_penetration_m
        ),
        terrain_geom_count=len(classes.terrain_geom_ids),
        foot_geom_count=len(classes.foot_geom_ids),
        forbidden_body_geom_count=len(classes.forbidden_body_geom_ids),
    )


__all__ = (
    "FrameCollisionObservation",
    "GeometryClasses",
    "StairMotionCollisionAuditResult",
    "aggregate_collision_frames",
    "audit_stair_motion_collisions",
    "classify_geometry_ids",
    "classify_terrain_contacts",
)
