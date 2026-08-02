"""Canonical adapter for the complete clean c490 GRAIL terrain bank."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from mm_sonic.grail_terrain_source import (
    G1MujocoFK,
    GrailClipRecord,
    discover_grail_clips,
    load_grail_motion,
    mujoco_to_isaaclab_joints,
    resample_grail_motion,
)

from . import storage
from .canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CanonicalTerrainMesh,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
    TerrainBinding,
)
from .math3d import (
    RigidTransform,
    angular_velocity_world_wxyz,
    finite_difference,
    unroll_quaternions_wxyz,
)


C490_FAMILIES = (
    "c490_stair_p1",
    "c490_stair_p2",
    "c490_slope",
    "c490_curb",
)
C490_EXPECTED_COUNTS = {
    "c490_stair_p1": 166,
    "c490_stair_p2": 174,
    "c490_slope": 77,
    "c490_curb": 72,
}

_STAIR_FAMILIES = ("c490_stair_p1", "c490_stair_p2")
_LEGACY_FAMILY_CATEGORY = {
    "c490_slope": "slope",
    "c490_curb": "curb",
}
_LEGACY_FRAME_COUNT = 250
_LEGACY_POSITION_ENV = np.zeros(3, dtype=np.float32)
_LEGACY_ROTATION_ENV_WXYZ = np.array(
    (np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)), dtype=np.float32
)
_LEGACY_POSE_SOURCE = "legacy_default_yaw"
_FOOT_BODY_INDICES = (18, 19)
_INFERRED_FACING_HORIZON_FRAMES = 24
_CONTACT_PLACEHOLDER_TAG = "contacts-unreconstructed"


@dataclass(frozen=True)
class GrailCanonicalSource:
    """One canonical motion clip and its explicitly paired canonical mesh."""

    clip: CanonicalClip
    mesh: CanonicalTerrainMesh
    source_record: GrailClipRecord


def _requested_families(families: Sequence[str]) -> tuple[str, ...]:
    if isinstance(families, (str, bytes)):
        raise ValueError("GRAIL families must be a nonempty sequence")
    requested = tuple(families)
    if (
        not requested
        or any(type(family) is not str for family in requested)
        or len(set(requested)) != len(requested)
        or any(family not in C490_FAMILIES for family in requested)
    ):
        raise ValueError(
            "GRAIL families must be a unique nonempty subset of C490_FAMILIES"
        )
    return requested


def _legacy_records(root: Path, family: str) -> tuple[GrailClipRecord, ...]:
    category = _LEGACY_FAMILY_CATEGORY[family]
    records: list[GrailClipRecord] = []
    seen: set[str] = set()
    shards = sorted(
        path
        for path in root.glob(f"{family}_*")
        if path.is_dir() and (path / "clips.json").is_file()
    )
    if not shards:
        raise ValueError(f"no committed clips.json shards found for {family}")
    for shard in shards:
        metadata_path = shard / "clips.json"
        try:
            rows = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid shard metadata: {metadata_path}") from error
        if not isinstance(rows, list):
            raise ValueError(f"{metadata_path} must contain a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("legacy GRAIL records must be objects")
            stem = row.get("stem")
            if not isinstance(stem, str) or not stem:
                raise ValueError("legacy GRAIL stem must be a nonempty string")
            if stem in seen:
                raise ValueError(f"duplicate GRAIL stem across shards: {stem}")
            if row.get("category") != category:
                raise ValueError(
                    f"{stem}: legacy {family} category must equal {category}"
                )
            if row.get("terrain") != category:
                raise ValueError(
                    f"{stem}: legacy {family} terrain must equal {category}"
                )
            if row.get("n_frames") != _LEGACY_FRAME_COUNT:
                raise ValueError(
                    f"{stem}: legacy {family} n_frames must equal "
                    f"{_LEGACY_FRAME_COUNT}"
                )
            pose_source = row.get("pose_source")
            if pose_source is not None and pose_source != _LEGACY_POSE_SOURCE:
                raise ValueError(
                    f"{stem}: legacy pose_source must equal "
                    f"{_LEGACY_POSE_SOURCE}"
                )
            robot_path = shard / "robot" / f"{stem}.pkl"
            usd_path = shard / "object_usd" / f"{stem}.usd"
            if not robot_path.is_file():
                raise ValueError(f"{stem}: missing paired robot pickle")
            if not usd_path.is_file():
                raise ValueError(f"{stem}: missing paired USD")
            seen.add(stem)
            records.append(
                GrailClipRecord(
                    family=family,
                    shard_path=shard,
                    stem=stem,
                    robot_path=robot_path,
                    usd_path=usd_path,
                    n_frames=_LEGACY_FRAME_COUNT,
                    terrain_position_env=_LEGACY_POSITION_ENV.copy(),
                    terrain_rotation_env_wxyz=(
                        _LEGACY_ROTATION_ENV_WXYZ.copy()
                    ),
                    pose_source=_LEGACY_POSE_SOURCE,
                )
            )
    return tuple(records)


def discover_clean_c490_records(
    shard_root: Path,
    families: Sequence[str] = C490_FAMILIES,
) -> tuple[GrailClipRecord, ...]:
    """Discover exactly the clean c490 shards, with one audited legacy fallback."""

    root = Path(shard_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if "/sonic-rollouts/" in f"{root.as_posix().rstrip('/')}/":
        raise ValueError("clean GRAIL discovery never reads sonic-rollouts")
    requested = _requested_families(families)
    stair_families = tuple(
        family for family in requested if family in _STAIR_FAMILIES
    )
    records: list[GrailClipRecord] = []
    if stair_families:
        records.extend(
            discover_grail_clips(
                root,
                families=stair_families,
                expected_family_counts={
                    family: C490_EXPECTED_COUNTS[family]
                    for family in stair_families
                },
            )
        )
    for family in requested:
        if family in _LEGACY_FAMILY_CATEGORY:
            records.extend(_legacy_records(root, family))
    counts = Counter(record.family for record in records)
    expected = Counter(
        {family: C490_EXPECTED_COUNTS[family] for family in requested}
    )
    if counts != expected:
        raise ValueError(
            "GRAIL inventory count mismatch; "
            f"expected={dict(expected)}, actual={dict(counts)}"
        )
    stems = [record.stem for record in records]
    if len(stems) != len(set(stems)):
        raise ValueError("duplicate GRAIL stem across clean c490 families")
    return tuple(records)


def _load_usd_mesh(
    path: Path, *, source_asset_sha256: str
) -> CanonicalTerrainMesh:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as error:
        raise RuntimeError("USD Python bindings are required") from error
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"could not open USD: {path}")
    if (
        UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
        or UsdGeom.GetStageMetersPerUnit(stage) != 1.0
    ):
        raise ValueError("GRAIL USD must declare Z up and metersPerUnit=1")
    meshes = tuple(
        UsdGeom.Mesh(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    )
    if len(meshes) != 1:
        raise ValueError(
            f"expected one terrain mesh in {path}, found {len(meshes)}"
        )
    usd_mesh = meshes[0]
    points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(
        usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64
    )
    indices = np.asarray(
        usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    )
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 3
        or counts.ndim != 1
        or len(counts) < 1
        or np.any(counts < 3)
        or indices.ndim != 1
        or int(np.sum(counts)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(points))
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"invalid terrain mesh arrays in {path}")
    transform = usd_mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transformed = np.asarray(
        [
            tuple(transform.Transform(Gf.Vec3d(*map(float, point))))
            for point in points
        ],
        dtype=np.float64,
    )
    left_handed = (
        usd_mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    )
    faces: list[tuple[int, int, int]] = []
    cursor = 0
    for count_value in counts:
        polygon = indices[cursor : cursor + int(count_value)]
        cursor += int(count_value)
        for offset in range(1, len(polygon) - 1):
            triangle = (
                int(polygon[0]),
                int(polygon[offset]),
                int(polygon[offset + 1]),
            )
            if left_handed:
                triangle = (triangle[0], triangle[2], triangle[1])
            faces.append(triangle)
    triangles = np.asarray(faces, dtype=np.int32)
    edges_a = transformed[triangles[:, 1]] - transformed[triangles[:, 0]]
    edges_b = transformed[triangles[:, 2]] - transformed[triangles[:, 0]]
    valid = np.linalg.norm(np.cross(edges_a, edges_b), axis=1) > 1.0e-12
    return CanonicalTerrainMesh(
        vertices_local=transformed,
        faces=triangles,
        valid_faces=valid,
        source_asset_sha256=source_asset_sha256,
    )


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _infer_commands(
    root_quaternion_world_wxyz: np.ndarray,
    root_linear_velocity_world: np.ndarray,
    root_angular_velocity_world: np.ndarray,
) -> CommandTrack:
    frame_count = len(root_quaternion_world_wxyz)
    yaw = _yaw_from_wxyz(root_quaternion_world_wxyz)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    velocity = np.asarray(root_linear_velocity_world)[:, :2]
    local_velocity = np.stack(
        (
            cosine * velocity[:, 0] + sine * velocity[:, 1],
            -sine * velocity[:, 0] + cosine * velocity[:, 1],
        ),
        axis=1,
    )
    future = np.minimum(
        np.arange(frame_count) + _INFERRED_FACING_HORIZON_FRAMES,
        frame_count - 1,
    )
    facing_delta = np.arctan2(
        np.sin(yaw[future] - yaw), np.cos(yaw[future] - yaw)
    )
    return CommandTrack(
        observed_travel_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_facing_stick_xy=np.zeros((frame_count, 2), dtype=np.float32),
        observed_mask=np.zeros(frame_count, dtype=np.bool_),
        inferred_velocity_local_xy=local_velocity,
        inferred_facing_local_xy=np.stack(
            (np.cos(facing_delta), np.sin(facing_delta)), axis=1
        ),
        inferred_yaw_rate_rad_s=np.asarray(root_angular_velocity_world)[:, 2],
    )


def _action_tags(family: str) -> tuple[str, ...]:
    category = (
        "stairs"
        if family in _STAIR_FAMILIES
        else _LEGACY_FAMILY_CATEGORY[family]
    )
    return ("grail", "terrain", category, _CONTACT_PLACEHOLDER_TAG)


def _canonicalize_record(
    record: GrailClipRecord, fk: G1MujocoFK
) -> GrailCanonicalSource:
    motion = load_grail_motion(
        record.robot_path, expected_frames=record.n_frames
    )
    resampled = resample_grail_motion(
        motion.root_position,
        motion.root_quaternion_xyzw,
        motion.dof_mujoco,
        source_fps=motion.fps,
        target_fps=50.0,
    )
    forward = fk.forward(
        resampled.root_position,
        resampled.root_quaternion_xyzw,
        resampled.dof_mujoco,
    )
    if tuple(forward.body_names) != ISAACLAB_BODY_NAMES:
        raise ValueError("GRAIL FK result does not use the canonical body order")
    root_wxyz = unroll_quaternions_wxyz(
        resampled.root_quaternion_xyzw[:, (3, 0, 1, 2)]
    )
    body_wxyz = unroll_quaternions_wxyz(
        forward.body_quaternion_world_xyzw[..., (3, 0, 1, 2)]
    )
    joint_position = mujoco_to_isaaclab_joints(resampled.dof_mujoco)
    root_linear_velocity = finite_difference(
        resampled.root_position, 50.0
    )
    root_angular_velocity = angular_velocity_world_wxyz(root_wxyz, 50.0)
    joint_velocity = finite_difference(joint_position, 50.0)
    body_linear_velocity = finite_difference(
        forward.body_position_world, 50.0
    )
    body_angular_velocity = angular_velocity_world_wxyz(body_wxyz, 50.0)
    sole_position = forward.body_position_world[:, _FOOT_BODY_INDICES]
    sole_quaternion = body_wxyz[:, _FOOT_BODY_INDICES]

    robot_bytes = record.robot_path.read_bytes()
    asset_bytes = record.usd_path.read_bytes()
    asset_sha256 = hashlib.sha256(asset_bytes).hexdigest()
    mesh = _load_usd_mesh(
        record.usd_path, source_asset_sha256=asset_sha256
    )
    terrain = TerrainBinding(
        asset_path=str(record.usd_path.resolve()),
        asset_size_bytes=len(asset_bytes),
        asset_sha256=asset_sha256,
        asset_license_id="UNRECORDED",
        mesh_sha256=storage.mesh_digest(mesh),
        world_from_terrain=RigidTransform(
            record.terrain_position_env,
            record.terrain_rotation_env_wxyz,
        ),
        validity_mask_path=None,
    )
    frame_count = len(resampled.root_position)
    clip = CanonicalClip(
        clip_id=f"{record.family}/{record.stem}",
        fps=50.0,
        source=SourceIdentity(
            source_format="grail-clean-robot-joblib-25hz",
            source_path=str(record.robot_path.resolve()),
            source_size_bytes=len(robot_bytes),
            source_sha256=hashlib.sha256(robot_bytes).hexdigest(),
            source_license_id="UNRECORDED",
            coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
            quaternion_convention="wxyz",
            pose_origin=CLEAN_POSE_ORIGIN,
        ),
        joint_names=ISAACLAB_JOINT_NAMES,
        body_names=ISAACLAB_BODY_NAMES,
        root_position_world=resampled.root_position,
        root_quaternion_world_wxyz=root_wxyz,
        joint_position=joint_position,
        root_linear_velocity_world=root_linear_velocity,
        root_angular_velocity_world=root_angular_velocity,
        joint_velocity=joint_velocity,
        body_position_world=forward.body_position_world,
        body_quaternion_world_wxyz=body_wxyz,
        body_linear_velocity_world=body_linear_velocity,
        body_angular_velocity_world=body_angular_velocity,
        sole_position_world=sole_position,
        sole_quaternion_world_wxyz=sole_quaternion,
        heel_position_world=sole_position,
        toe_position_world=sole_position,
        contact=np.zeros((frame_count, 2), dtype=np.float32),
        contact_confidence=np.zeros((frame_count, 2), dtype=np.float32),
        commands=_infer_commands(
            root_wxyz, root_linear_velocity, root_angular_velocity
        ),
        terrain=terrain,
        action_tags=_action_tags(record.family),
    )
    clip.validate()
    return GrailCanonicalSource(clip=clip, mesh=mesh, source_record=record)


def iter_grail_clips(
    shard_root: Path,
    families: Sequence[str],
    model_path: Path,
) -> Iterator[GrailCanonicalSource]:
    """Yield each clean c490 clip together with its paired canonical mesh."""

    records = discover_clean_c490_records(shard_root, families=families)
    fk = G1MujocoFK(model_path)
    for record in records:
        yield _canonicalize_record(record, fk)
