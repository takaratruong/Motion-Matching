"""Generic globally privileged terrain-conditioned motion matching.

The older terrain experiments in this repository classified one obstacle as a
staircase and then planned entry/exit portals through it.  That is useful for
debugging one asset, but it is not the controller used by games: it bakes stair
topology into selection and cannot naturally generalise to a curb, ramp, or
irregular height field.

This module deliberately has no stair vocabulary.  Every canonical motion row
stores the future trajectory of both soles and the terrain height/normal below
those soles.  At runtime ordinary pose/trajectory motion matching produces a
short list; the known target mesh is sampled under each candidate's prospective
feet and re-ranks that list.  A staircase is therefore only one possible local
height profile.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_data import G1_TAKARA_LAYOUT, MotionClip, MotionFolder
from .torch_motion_features import (
    FEATURE_HORIZON_FRAMES,
    CommandTrajectory,
    GeneratedFeatureState,
    TorchMotionDatabase,
    extract_query_features,
)
from .terrain_oracle.canonical import CanonicalClip
from .terrain_oracle.contact import SoleGeometry
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.storage import load_corpus, read_clip, read_mesh
from .terrain_oracle.terrain_mesh import TerrainMeshIndex, mesh_from_heightfield
from .torch_motion_matcher import (
    ForcedMotionAlignment,
    MatcherConfig,
    MotionMatchResult,
    PreparedMotionMatch,
    ShapedCommand,
    TorchMotionMatcher,
    predict_command_trajectory,
    predict_takara_ball_trajectory,
    search_is_due,
)


TERRAIN_HORIZON_FRAMES = tuple(range(0, FEATURE_HORIZON_FRAMES[-1] + 1, 5))
CONTACT_HORIZON_FRAMES = (0, 5, 10, 15)
CONTINUITY_HORIZON_FRAMES = 10


def _yaw_from_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_xy(value: np.ndarray, yaw: np.ndarray | float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    angle = np.asarray(yaw, dtype=np.float64)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    while cosine.ndim < source[..., 0].ndim:
        cosine = cosine[..., None]
        sine = sine[..., None]
    return np.stack(
        (
            cosine * source[..., 0] - sine * source[..., 1],
            sine * source[..., 0] + cosine * source[..., 1],
        ),
        axis=-1,
    )


def _rotate_wxyz_numpy(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by wxyz quaternions with NumPy broadcasting."""

    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    twice_cross = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * twice_cross + np.cross(q[..., 1:], twice_cross)


def _rotate_wxyz_torch(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by wxyz quaternions with Torch broadcasting."""

    twice_cross = 2.0 * torch.linalg.cross(quaternion[..., 1:], vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(quaternion[..., 1:], twice_cross, dim=-1)
    )


class MeshHeightField:
    """Vectorised top-surface queries over an arbitrary canonical mesh.

    Motion matching needs the walkable surface, so vertical faces are ignored
    and the highest triangle intersecting each world XY coordinate is returned.
    This is the globally privileged height-field bootstrap requested for the
    first version.  It supports flat ground, ramps, curbs, uneven treads, and
    generated mixed courses without classifying any of them.
    """

    def __init__(self, index: TerrainMeshIndex) -> None:
        if not isinstance(index, TerrainMeshIndex):
            raise ContractError("MeshHeightField requires a TerrainMeshIndex")
        triangles = np.asarray(index.triangles_world, dtype=np.float64)
        normals = np.asarray(index.normals_world, dtype=np.float64).copy()
        normals[normals[:, 2] < 0.0] *= -1.0
        self.index = index
        self._triangles = triangles
        self._normals = normals

    def sample(self, points_world_xy: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return top height, upward normal, and hit mask for ``[N,2]`` XY."""

        points = np.asarray(points_world_xy, dtype=np.float64)
        original_shape = points.shape[:-1]
        if points.ndim < 2 or points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ContractError("height-field points must be finite with shape [...,2]")
        flat = points.reshape(-1, 2)
        height = np.full(len(flat), -np.inf, dtype=np.float64)
        normal = np.zeros((len(flat), 3), dtype=np.float64)
        tolerance = 1.0e-10

        # Looping over terrain triangles and vectorising over query points is
        # substantially faster than one Python raycast per candidate probe.
        for triangle, triangle_normal in zip(
            self._triangles, self._normals, strict=True
        ):
            a, b, c = triangle
            edge0 = b[:2] - a[:2]
            edge1 = c[:2] - a[:2]
            denominator = edge0[0] * edge1[1] - edge0[1] * edge1[0]
            if abs(float(denominator)) <= tolerance:
                continue
            relative = flat - a[:2]
            u = (relative[:, 0] * edge1[1] - relative[:, 1] * edge1[0]) / denominator
            v = (edge0[0] * relative[:, 1] - edge0[1] * relative[:, 0]) / denominator
            inside = (
                (u >= -tolerance)
                & (v >= -tolerance)
                & (u + v <= 1.0 + tolerance)
            )
            surface_z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
            replace = inside & (surface_z > height)
            if np.any(replace):
                height[replace] = surface_z[replace]
                normal[replace] = triangle_normal

        hit = np.isfinite(height)
        height[~hit] = np.nan
        return (
            height.reshape(original_shape),
            normal.reshape((*original_shape, 3)),
            hit.reshape(original_shape),
        )


class RegularGridHeightField:
    """Fast bilinear height/normal queries for large procedural arenas.

    ``MeshHeightField`` is exact for an arbitrary authored triangle mesh but
    its triangle scan is intentionally simple.  A large game-style heightmap
    would make that scan dominate every database search.  This representation
    keeps the same ``sample`` interface while reducing a query to four array
    gathers.  It also owns a canonical mesh index for the articulated renderer.
    """

    def __init__(
        self,
        height: object,
        *,
        spacing_m: float | tuple[float, float],
        origin_xy: tuple[float, float],
        valid: object | None = None,
    ) -> None:
        values = np.asarray(height, dtype=np.float64)
        validity = (
            np.ones(values.shape, dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if values.ndim != 2 or validity.shape != values.shape:
            raise ContractError("regular-grid height and validity must be equal 2D arrays")
        if not np.isfinite(values).all() or min(values.shape) < 2:
            raise ContractError("regular-grid height must be finite and at least 2x2")
        if isinstance(spacing_m, tuple):
            dx, dy = map(float, spacing_m)
        else:
            dx = dy = float(spacing_m)
        origin = tuple(float(value) for value in origin_xy)
        if (
            not np.isfinite((dx, dy, *origin)).all()
            or dx <= 0.0
            or dy <= 0.0
        ):
            raise ContractError("regular-grid spacing and origin are invalid")
        dz_dy, dz_dx = np.gradient(values, dy, dx)
        normal = np.stack((-dz_dx, -dz_dy, np.ones_like(values)), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1.0e-12)
        self.height = np.ascontiguousarray(values)
        self.valid = np.ascontiguousarray(validity)
        self.normal = np.ascontiguousarray(normal)
        self.spacing_m = (dx, dy)
        self.origin_xy = origin
        mesh = mesh_from_heightfield(
            values,
            validity,
            spacing_m=(dx, dy),
            origin_xy=origin,
        )
        self.index = TerrainMeshIndex(
            mesh,
            RigidTransform(
                np.zeros(3, dtype=np.float32),
                np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            ),
        )

    def sample(self, points_world_xy: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points_world_xy, dtype=np.float64)
        original_shape = points.shape[:-1]
        if points.ndim < 2 or points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ContractError("height-field points must be finite with shape [...,2]")
        flat = points.reshape(-1, 2)
        dx, dy = self.spacing_m
        column = (flat[:, 0] - self.origin_xy[0]) / dx
        row = (flat[:, 1] - self.origin_xy[1]) / dy
        rows, columns = self.height.shape
        inside = (
            (column >= 0.0)
            & (column <= columns - 1)
            & (row >= 0.0)
            & (row <= rows - 1)
        )
        c0 = np.floor(column).astype(np.int64).clip(0, columns - 1)
        r0 = np.floor(row).astype(np.int64).clip(0, rows - 1)
        c1 = np.minimum(c0 + 1, columns - 1)
        r1 = np.minimum(r0 + 1, rows - 1)
        tx = np.clip(column - c0, 0.0, 1.0)
        ty = np.clip(row - r0, 0.0, 1.0)
        corner_valid = (
            self.valid[r0, c0]
            & self.valid[r0, c1]
            & self.valid[r1, c0]
            & self.valid[r1, c1]
        )
        hit = inside & corner_valid

        def bilinear(array: np.ndarray) -> np.ndarray:
            a = array[r0, c0]
            b = array[r0, c1]
            c = array[r1, c0]
            d = array[r1, c1]
            x = tx if a.ndim == 1 else tx[:, None]
            y = ty if a.ndim == 1 else ty[:, None]
            return (1.0 - y) * ((1.0 - x) * a + x * b) + y * (
                (1.0 - x) * c + x * d
            )

        sampled_height = bilinear(self.height)
        sampled_normal = bilinear(self.normal)
        sampled_normal /= np.linalg.norm(
            sampled_normal, axis=-1, keepdims=True
        ).clip(min=1.0e-12)
        sampled_height[~hit] = np.nan
        sampled_normal[~hit] = 0.0
        return (
            sampled_height.reshape(original_shape),
            sampled_normal.reshape((*original_shape, 3)),
            hit.reshape(original_shape),
        )


@dataclass(frozen=True)
class CanonicalTerrainLibrary:
    corpus_root: Path
    clip_ids: tuple[str, ...]
    canonical_clips: tuple[CanonicalClip, ...]
    height_fields: tuple[MeshHeightField, ...]
    folder: MotionFolder


def _select_records(
    records: Sequence[object],
    *,
    clip_ids: Sequence[str] | None,
    include_prefixes: Sequence[str],
    maximum_clips: int | None,
) -> list[object]:
    requested = None if clip_ids is None else set(clip_ids)
    selected = []
    for record in records:
        clip_id = str(getattr(record, "clip_id"))
        if requested is not None:
            if clip_id not in requested:
                continue
        elif include_prefixes and not any(
            clip_id.startswith(prefix) for prefix in include_prefixes
        ):
            continue
        selected.append(record)
        if maximum_clips is not None and len(selected) >= maximum_clips:
            break
    if requested is not None:
        missing = requested - {str(getattr(record, "clip_id")) for record in selected}
        if missing:
            raise ContractError(f"canonical clip ids not found: {sorted(missing)}")
    if not selected:
        raise ContractError("canonical terrain selection is empty")
    return selected


def load_canonical_terrain_library(
    corpus_root: str | Path,
    *,
    clip_ids: Sequence[str] | None = None,
    include_prefixes: Sequence[str] = (),
    maximum_clips: int | None = None,
) -> CanonicalTerrainLibrary:
    """Load canonical clips directly into the ordinary Torch MM layout."""

    root = Path(corpus_root).expanduser().resolve()
    manifest = load_corpus(root)
    records = _select_records(
        manifest.clips,
        clip_ids=clip_ids,
        include_prefixes=include_prefixes,
        maximum_clips=maximum_clips,
    )
    mesh_records = {record.sha256: record for record in manifest.meshes}
    clips: list[CanonicalClip] = []
    fields: list[MeshHeightField] = []
    motion_clips: list[MotionClip] = []
    digest = hashlib.sha256()

    worker_count = max(1, min(8, int(os.environ.get("MM_TERRAIN_LOAD_WORKERS", "1"))))

    def read_record(record: object) -> tuple[object, CanonicalClip, object, object] | None:
        clip = read_clip(root / record.relative_path)
        if clip.frame_count <= FEATURE_HORIZON_FRAMES[-1]:
            return None
        if clip.terrain is None:
            raise ContractError(f"canonical clip has no terrain binding: {clip.clip_id}")
        mesh_record = mesh_records.get(clip.terrain.mesh_sha256)
        if mesh_record is None:
            raise ContractError(f"canonical clip mesh is absent: {clip.clip_id}")
        mesh = read_mesh(root / mesh_record.relative_path)
        return record, clip, mesh_record, mesh

    if worker_count == 1:
        loaded = map(read_record, records)
    else:
        executor = ThreadPoolExecutor(max_workers=worker_count)
        loaded = executor.map(read_record, records)
    try:
        for item in loaded:
            if item is None:
                continue
            record, clip, mesh_record, mesh = item
            index = TerrainMeshIndex(mesh, clip.terrain)
            clips.append(clip)
            fields.append(MeshHeightField(index))
            motion_clips.append(
                MotionClip(
                    relative_path=f"canonical/{clip.clip_id}/motion.npz",
                    fps=50,
                    joint_position=clip.joint_position,
                    joint_velocity=clip.joint_velocity,
                    body_position_world=clip.body_position_world,
                    body_quaternion_world_wxyz=clip.body_quaternion_world_wxyz,
                    body_linear_velocity_world=clip.body_linear_velocity_world,
                    body_angular_velocity_world=clip.body_angular_velocity_world,
                )
            )
            digest.update(record.sha256.encode("ascii"))
            digest.update(mesh_record.sha256.encode("ascii"))
    finally:
        if worker_count != 1:
            executor.shutdown(wait=True)

    if not clips:
        raise ContractError("canonical terrain selection has no searchable clip")
    return CanonicalTerrainLibrary(
        corpus_root=root,
        clip_ids=tuple(clip.clip_id for clip in clips),
        canonical_clips=tuple(clips),
        height_fields=tuple(fields),
        folder=MotionFolder(
            root=root,
            layout=G1_TAKARA_LAYOUT,
            clips=tuple(motion_clips),
            inventory_sha256=digest.hexdigest(),
        ),
    )


@dataclass(frozen=True)
class TerrainRowFeatures:
    future_foot_local_xy: torch.Tensor
    future_foot_height_from_root: torch.Tensor
    source_height_from_root: torch.Tensor
    source_normal_local: torch.Tensor
    source_hit: torch.Tensor
    source_path_height_from_root: torch.Tensor
    source_path_normal_local: torch.Tensor
    source_path_hit: torch.Tensor
    contact_phase: torch.Tensor
    contact_future: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    maximum_future_joint_step_rad: torch.Tensor
    maximum_future_root_step_m: torch.Tensor
    maximum_future_root_yaw_step_rad: torch.Tensor


def _exact_sole_corners_world(
    clip: CanonicalClip, geometry: SoleGeometry
) -> np.ndarray:
    """Reconstruct the four rigid collision-sphere bottoms for both feet.

    The canonical clip intentionally stores compact sole/heel/toe averages.
    Those three averages are useful for contact reporting but are not enough
    at a riser: one lateral corner can collide while every average remains
    clear.  Retrieval therefore reconstructs the actual G1 collision
    footprint from the named ankle bodies when model geometry is available.
    """

    if not isinstance(geometry, SoleGeometry):
        raise ContractError("exact sole corners require SoleGeometry")
    try:
        body_indices = np.asarray(
            [clip.body_names.index(name) for name in geometry.body_names],
            dtype=np.int64,
        )
    except ValueError as error:
        raise ContractError(
            "canonical clip is missing a named ankle-roll body"
        ) from error
    position = np.asarray(
        clip.body_position_world[:, body_indices], dtype=np.float64
    )
    quaternion = np.asarray(
        clip.body_quaternion_world_wxyz[:, body_indices], dtype=np.float64
    )
    offsets = np.broadcast_to(
        np.asarray(geometry.corner_positions_body, dtype=np.float64)[None],
        (clip.frame_count, 2, 4, 3),
    )
    return position[:, :, None, :] + _rotate_wxyz_numpy(
        quaternion[:, :, None, :], offsets
    )


def build_terrain_row_features(
    library: CanonicalTerrainLibrary,
    database: TorchMotionDatabase,
    *,
    sole_geometry: SoleGeometry | None = None,
) -> TerrainRowFeatures:
    """Build dense future foot and command-path terrain probes for every row."""

    row_clip = database._search_clip_index.detach().cpu().numpy()
    row_frame = database._search_frame_index.detach().cpu().numpy()
    row_count = len(row_clip)
    horizons = np.asarray(TERRAIN_HORIZON_FRAMES, dtype=np.int64)
    probe_count = 4 if sole_geometry is not None else 3
    foot_local = np.empty(
        (row_count, 2, len(horizons), probe_count, 2), dtype=np.float32
    )
    foot_height = np.empty(
        (row_count, 2, len(horizons), probe_count), dtype=np.float32
    )
    source_height = np.empty(
        (row_count, 2, len(horizons), probe_count), dtype=np.float32
    )
    source_normal = np.empty(
        (row_count, 2, len(horizons), probe_count, 3), dtype=np.float32
    )
    source_hit = np.zeros(
        (row_count, 2, len(horizons), probe_count), dtype=np.bool_
    )
    source_path_height = np.empty((row_count, len(horizons)), dtype=np.float32)
    source_path_normal = np.empty(
        (row_count, len(horizons), 3), dtype=np.float32
    )
    source_path_hit = np.zeros((row_count, len(horizons)), dtype=np.bool_)
    contact_phase = np.zeros(row_count, dtype=np.int8)
    contact_future = np.zeros((row_count, 2, 4), dtype=np.float32)
    joint_position = np.zeros((row_count, 29), dtype=np.float32)
    joint_velocity = np.zeros((row_count, 29), dtype=np.float32)
    maximum_future_joint_step = np.zeros(row_count, dtype=np.float32)
    maximum_future_root_step = np.zeros(row_count, dtype=np.float32)
    maximum_future_root_yaw_step = np.zeros(row_count, dtype=np.float32)

    for clip_index, (clip, height_field) in enumerate(
        zip(library.canonical_clips, library.height_fields, strict=True)
    ):
        rows = np.flatnonzero(row_clip == clip_index)
        if rows.size == 0:
            continue
        frames = row_frame[rows]
        future = frames[:, None] + horizons[None]
        root = np.asarray(clip.root_position_world[frames], dtype=np.float64)
        yaw = _yaw_from_wxyz(clip.root_quaternion_world_wxyz[frames])
        # [row,horizon,foot,probe,xyz] -> [row,foot,horizon,probe,xyz].
        # Prefer all four exact collision corners.  The compact three-probe
        # fallback keeps synthetic/unit-test libraries model independent.
        if sole_geometry is None:
            probes = np.stack(
                (
                    np.asarray(
                        clip.sole_position_world[future], dtype=np.float64
                    ),
                    np.asarray(
                        clip.heel_position_world[future], dtype=np.float64
                    ),
                    np.asarray(
                        clip.toe_position_world[future], dtype=np.float64
                    ),
                ),
                axis=3,
            ).transpose(0, 2, 1, 3, 4)
        else:
            corners = _exact_sole_corners_world(clip, sole_geometry)
            probes = np.asarray(corners[future], dtype=np.float64).transpose(
                0, 2, 1, 3, 4
            )
        relative_xy = probes[..., :2] - root[:, None, None, None, :2]
        foot_local[rows] = _rotate_xy(relative_xy, -yaw).astype(np.float32)
        # Actual heel/sole/toe height is distinct from the terrain height
        # beneath those probes.  Without it a stair candidate can match the
        # target height profile while its airborne foot passes through the
        # next riser.  Express it from the current root so the same rigid
        # registration used by the matcher applies at runtime.
        foot_height[rows] = (
            probes[..., 2] - root[:, None, None, None, 2]
        ).astype(np.float32)

        height, normal_world, hit = height_field.sample(probes[..., :2])
        source_height[rows] = (
            height - root[:, None, None, None, 2]
        ).astype(np.float32)
        normal_local = normal_world.copy()
        normal_local[..., :2] = _rotate_xy(normal_world[..., :2], -yaw)
        source_normal[rows] = normal_local.astype(np.float32)
        source_hit[rows] = hit
        future_root = np.asarray(
            clip.root_position_world[future], dtype=np.float64
        )
        path_height, path_normal_world, path_hit = height_field.sample(
            future_root[..., :2]
        )
        source_path_height[rows] = (
            path_height - root[:, None, 2]
        ).astype(np.float32)
        path_normal_local = path_normal_world.copy()
        path_normal_local[..., :2] = _rotate_xy(
            path_normal_world[..., :2], -yaw
        )
        source_path_normal[rows] = path_normal_local.astype(np.float32)
        source_path_hit[rows] = path_hit
        current_contact = np.asarray(clip.contact[frames]) >= 0.5
        contact_phase[rows] = (
            current_contact[:, 0].astype(np.int8)
            + 2 * current_contact[:, 1].astype(np.int8)
        )
        future_contact_frames = (
            frames[:, None]
            + np.asarray(CONTACT_HORIZON_FRAMES, dtype=np.int64)[None]
        )
        contact_future[rows] = np.asarray(
            clip.contact[future_contact_frames], dtype=np.float32
        ).transpose(0, 2, 1)
        joint_position[rows] = np.asarray(
            clip.joint_position[frames], dtype=np.float32
        )
        joint_velocity[rows] = np.asarray(
            clip.joint_velocity[frames], dtype=np.float32
        )
        continuity_offsets = np.arange(
            CONTINUITY_HORIZON_FRAMES, dtype=np.int64
        )
        continuity_frames = frames[:, None] + continuity_offsets[None]
        joint_steps = np.max(
            np.abs(np.diff(np.asarray(clip.joint_position), axis=0)), axis=1
        )
        root_steps = np.linalg.norm(
            np.diff(np.asarray(clip.root_position_world), axis=0), axis=1
        )
        root_yaw = _yaw_from_wxyz(clip.root_quaternion_world_wxyz)
        yaw_steps = np.abs(
            np.arctan2(np.sin(np.diff(root_yaw)), np.cos(np.diff(root_yaw)))
        )
        maximum_future_joint_step[rows] = np.max(
            joint_steps[continuity_frames], axis=1
        ).astype(np.float32)
        maximum_future_root_step[rows] = np.max(
            root_steps[continuity_frames], axis=1
        ).astype(np.float32)
        maximum_future_root_yaw_step[rows] = np.max(
            yaw_steps[continuity_frames], axis=1
        ).astype(np.float32)

    device = database.device
    return TerrainRowFeatures(
        future_foot_local_xy=torch.as_tensor(foot_local, device=device),
        future_foot_height_from_root=torch.as_tensor(
            foot_height, device=device
        ),
        source_height_from_root=torch.as_tensor(source_height, device=device),
        source_normal_local=torch.as_tensor(source_normal, device=device),
        source_hit=torch.as_tensor(source_hit, device=device),
        source_path_height_from_root=torch.as_tensor(
            source_path_height, device=device
        ),
        source_path_normal_local=torch.as_tensor(
            source_path_normal, device=device
        ),
        source_path_hit=torch.as_tensor(source_path_hit, device=device),
        contact_phase=torch.as_tensor(contact_phase, device=device),
        contact_future=torch.as_tensor(contact_future, device=device),
        joint_position=torch.as_tensor(joint_position, device=device),
        joint_velocity=torch.as_tensor(joint_velocity, device=device),
        maximum_future_joint_step_rad=torch.as_tensor(
            maximum_future_joint_step, device=device
        ),
        maximum_future_root_step_m=torch.as_tensor(
            maximum_future_root_step, device=device
        ),
        maximum_future_root_yaw_step_rad=torch.as_tensor(
            maximum_future_root_yaw_step, device=device
        ),
    )


@dataclass(frozen=True)
class GenericTerrainSearchConfig:
    preselection_count: int = 2048
    terrain_height_weight: float = 8.0
    terrain_height_scale_m: float = 0.12
    terrain_normal_weight: float = 0.75
    prospective_foot_clearance_weight: float = 8.0
    prospective_foot_clearance_scale_m: float = 0.04
    # Reject a target route that would require lifting this source motion's
    # authored flight by more than a physically modest amount.  Unlike an
    # absolute foot-height cutoff, this compares each pose with its own source
    # terrain and therefore cancels link/probe representation offsets.
    maximum_prospective_flight_clearance_deficit_m: float = math.inf
    # This is an emergency impossibility cutoff, not the final contact gate.
    # The soft cost above should choose the cleanest available flight.  A
    # 5-mm search-time cutoff can dead-end when inertialized root height lags a
    # large terrain change even though the downstream rigid reconstruction is
    # still feasible; exact collision is audited on the resulting motion.
    minimum_prospective_foot_clearance_m: float = -0.12
    trajectory_terrain_height_weight: float = 12.0
    trajectory_terrain_normal_weight: float = 0.5
    transition_penalty: float = 1.0
    compatible_transition_penalty: float = 8.0
    transition_release_rms_start_m: float = 0.025
    transition_release_rms_stop_m: float = 0.075
    # A low terrain residual only says the incumbent still fits the ground;
    # it does not say that it still fits a newly changed joystick command.
    # Release the same hysteresis when the normalized motion-query mismatch
    # becomes large, otherwise a flat forward clip can dominate lateral or
    # backward commands simply because every flat clip has zero terrain cost.
    transition_release_feature_start: float = 6.0
    transition_release_feature_stop: float = 12.0
    contact_timing_weight: float = 4.0
    joint_position_weight: float = 0.25
    joint_position_scale_rad: float = 0.50
    joint_velocity_weight: float = 0.05
    joint_velocity_scale_rad_s: float = 3.0
    maximum_source_joint_step_rad: float = 0.25
    maximum_source_root_step_m: float = 0.08
    maximum_source_root_yaw_step_rad: float = 0.35
    exclusion_frames: int = 20


@dataclass(frozen=True)
class TerrainCandidateSelection:
    selected_row: int
    selected_clip_index: int
    selected_frame_index: int
    feature_cost: float
    terrain_cost: float
    contact_cost: float
    pose_cost: float
    terrain_rms_m: float
    minimum_prospective_foot_clearance_m: float
    maximum_prospective_flight_clearance_deficit_m: float
    total_cost: float
    preselection_count: int


@dataclass(frozen=True)
class ContinuousTerrainSearchEvent:
    """One terrain-aware database search made by the rolling controller."""

    sequence: int
    selection: TerrainCandidateSelection
    source_clip_id: str
    transitioned: bool
    alignment: ForcedMotionAlignment | None


@dataclass(frozen=True)
class ContinuousTerrainRuntimeConfig:
    """Small continuous adaptations applied after database selection."""

    vertical_adaptation: bool = True
    output_grounding: bool = False
    terrain_command_adaptation: bool = True
    vertical_halflife_s: float = 0.08
    maximum_vertical_step_m: float = 0.012


def select_terrain_candidate(
    database: TorchMotionDatabase,
    terrain_rows: TerrainRowFeatures,
    state: GeneratedFeatureState,
    trajectory: CommandTrajectory,
    target_height_field: MeshHeightField,
    *,
    current_clip_index: int | None = None,
    current_frame_index: int | None = None,
    incumbent_row: int | None = None,
    current_joint_position: torch.Tensor | None = None,
    current_joint_velocity: torch.Tensor | None = None,
    excluded_clip_indices: Iterable[int] = (),
    config: GenericTerrainSearchConfig = GenericTerrainSearchConfig(),
) -> TerrainCandidateSelection:
    """Two-stage pose/trajectory then candidate-foot terrain search."""

    query = database.normalization.normalize(extract_query_features(state, trajectory))
    feature_cost = torch.sum(
        torch.square(database._search_features - query.unsqueeze(0)), dim=1
    )
    quaternion = state.root_orientation_world_wxyz
    w, x, y, z = quaternion.unbind()
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)

    # A fixed command-path terrain profile prevents a candidate from hiding a
    # riser simply because its particular sparse footprints miss the edge.
    path_knots = torch.cat(
        (state.root_position_world[:2].unsqueeze(0), trajectory.position_world_xy),
        dim=0,
    ).detach().cpu().numpy()
    knot_frames = np.asarray((0, *FEATURE_HORIZON_FRAMES), dtype=np.float64)
    dense_frames = np.asarray(TERRAIN_HORIZON_FRAMES, dtype=np.float64)
    target_path_xy = np.stack(
        tuple(
            np.interp(dense_frames, knot_frames, path_knots[:, axis])
            for axis in range(2)
        ),
        axis=-1,
    )
    target_path_height_np, target_path_normal_np, target_path_hit_np = (
        target_height_field.sample(target_path_xy)
    )
    if not bool(np.all(target_path_hit_np)):
        raise ContractError("command trajectory leaves the target terrain")
    target_path_height = torch.as_tensor(
        target_path_height_np, dtype=torch.float32, device=database.device
    ) - state.root_position_world[2]
    target_path_normal = torch.as_tensor(
        target_path_normal_np, dtype=torch.float32, device=database.device
    )
    path_nx = cosine * target_path_normal[..., 0] + sine * target_path_normal[..., 1]
    path_ny = -sine * target_path_normal[..., 0] + cosine * target_path_normal[..., 1]
    target_path_normal = torch.stack(
        (path_nx, path_ny, target_path_normal[..., 2]), dim=-1
    )
    path_height_residual = (
        terrain_rows.source_path_height_from_root - target_path_height
    )
    normalized_path_height_sq = torch.square(
        path_height_residual / config.terrain_height_scale_m
    )
    path_height_cost = config.trajectory_terrain_height_weight * (
        torch.mean(normalized_path_height_sq, dim=1)
        + 0.5 * torch.amax(normalized_path_height_sq, dim=1)
    )
    path_normal_dot = torch.sum(
        terrain_rows.source_path_normal_local * target_path_normal, dim=-1
    ).clamp(-1.0, 1.0)
    path_normal_cost = config.trajectory_terrain_normal_weight * torch.mean(
        1.0 - path_normal_dot, dim=1
    )
    path_cost = path_height_cost + path_normal_cost
    eligible = (
        terrain_rows.source_hit.reshape(len(feature_cost), -1).all(dim=1)
        & terrain_rows.source_path_hit.all(dim=1)
        & (
            terrain_rows.maximum_future_joint_step_rad
            <= config.maximum_source_joint_step_rad
        )
        & (
            terrain_rows.maximum_future_root_step_m
            <= config.maximum_source_root_step_m
        )
        & (
            terrain_rows.maximum_future_root_yaw_step_rad
            <= config.maximum_source_root_yaw_step_rad
        )
    )
    for clip_index in excluded_clip_indices:
        eligible &= database._search_clip_index != int(clip_index)
    if current_clip_index is not None and current_frame_index is not None:
        local = (
            (database._search_clip_index == int(current_clip_index))
            & (
                torch.abs(database._search_frame_index - int(current_frame_index))
                <= config.exclusion_frames
            )
        )
        eligible &= ~local
    if incumbent_row is not None:
        eligible[int(incumbent_row)] = True
    eligible_count = int(torch.count_nonzero(eligible).item())
    if eligible_count == 0:
        raise ContractError("generic terrain search has no eligible row")

    masked = (feature_cost + path_cost).masked_fill(~eligible, math.inf)
    count = min(config.preselection_count, eligible_count)
    candidates = torch.topk(masked, k=count, largest=False, sorted=False).indices
    if incumbent_row is not None and not bool(torch.any(candidates == incumbent_row)):
        candidates[-1] = int(incumbent_row)

    offsets = terrain_rows.future_foot_local_xy[candidates]
    world_offsets = torch.stack(
        (
            cosine * offsets[..., 0] - sine * offsets[..., 1],
            sine * offsets[..., 0] + cosine * offsets[..., 1],
        ),
        dim=-1,
    )
    sample_xy = state.root_position_world[:2] + world_offsets
    height_np, normal_np, hit_np = target_height_field.sample(
        sample_xy.detach().cpu().numpy()
    )
    target_height = torch.as_tensor(
        height_np, dtype=torch.float32, device=database.device
    ) - state.root_position_world[2]
    target_normal = torch.as_tensor(
        normal_np, dtype=torch.float32, device=database.device
    )
    # Express target normals in the same current-heading frame as source rows.
    nx = cosine * target_normal[..., 0] + sine * target_normal[..., 1]
    ny = -sine * target_normal[..., 0] + cosine * target_normal[..., 1]
    target_normal = torch.stack((nx, ny, target_normal[..., 2]), dim=-1)
    hit = torch.as_tensor(hit_np, dtype=torch.bool, device=database.device)

    height_residual = (
        terrain_rows.source_height_from_root[candidates] - target_height
    )
    target_foot_clearance = (
        terrain_rows.future_foot_height_from_root[candidates] - target_height
    )
    source_foot_clearance = (
        terrain_rows.future_foot_height_from_root[candidates]
        - terrain_rows.source_height_from_root[candidates]
    )
    clearance_loss = torch.relu(
        source_foot_clearance - target_foot_clearance - 0.005
    )
    clearance_cost = config.prospective_foot_clearance_weight * torch.mean(
        torch.square(
            clearance_loss / config.prospective_foot_clearance_scale_m
        ),
        dim=tuple(range(1, clearance_loss.ndim)),
    )
    minimum_target_clearance = torch.amin(
        target_foot_clearance,
        dim=tuple(range(1, target_foot_clearance.ndim)),
    )
    if clearance_loss.shape[2] == terrain_rows.contact_future.shape[2]:
        dense_future_contact = terrain_rows.contact_future[candidates]
    elif clearance_loss.shape[2] == len(TERRAIN_HORIZON_FRAMES):
        contact_horizon_index = torch.as_tensor(
            np.argmin(
                np.abs(
                    np.asarray(CONTACT_HORIZON_FRAMES, dtype=np.int64)[:, None]
                    - np.asarray(TERRAIN_HORIZON_FRAMES, dtype=np.int64)[None]
                ),
                axis=0,
            ),
            dtype=torch.long,
            device=database.device,
        )
        dense_future_contact = torch.index_select(
            terrain_rows.contact_future[candidates], 2, contact_horizon_index
        )
    else:
        raise ContractError("terrain and contact clearance horizons disagree")
    flight_mask = dense_future_contact < 0.5
    while flight_mask.ndim < clearance_loss.ndim:
        flight_mask = flight_mask.unsqueeze(-1)
    flight_clearance_deficit = clearance_loss.masked_fill(
        ~flight_mask, 0.0
    )
    maximum_flight_clearance_deficit = torch.amax(
        flight_clearance_deficit,
        dim=tuple(range(1, clearance_loss.ndim)),
    )
    height_dimensions = tuple(range(1, height_residual.ndim))
    terrain_rms = torch.sqrt(
        (
            torch.sum(torch.square(height_residual), dim=height_dimensions)
            + torch.sum(
                torch.square(path_height_residual[candidates]), dim=1
            )
        )
        / float(
            math.prod(height_residual.shape[1:])
            + len(TERRAIN_HORIZON_FRAMES)
        )
    )
    height_cost = config.terrain_height_weight * torch.mean(
        torch.square(height_residual / config.terrain_height_scale_m),
        dim=height_dimensions,
    )
    normal_dot = torch.sum(
        terrain_rows.source_normal_local[candidates] * target_normal, dim=-1
    ).clamp(-1.0, 1.0)
    normal_dimensions = tuple(range(1, normal_dot.ndim))
    normal_cost = config.terrain_normal_weight * torch.mean(
        1.0 - normal_dot, dim=normal_dimensions
    )
    terrain_cost = (
        height_cost + normal_cost + path_cost[candidates] + clearance_cost
    )
    terrain_cost = terrain_cost.masked_fill(
        (~hit.reshape(len(candidates), -1).all(dim=1))
        | (
            minimum_target_clearance
            < config.minimum_prospective_foot_clearance_m
        ),
        math.inf,
    )
    terrain_cost = terrain_cost.masked_fill(
        maximum_flight_clearance_deficit
        > config.maximum_prospective_flight_clearance_deficit_m,
        math.inf,
    )
    contact_cost = torch.zeros_like(terrain_cost)
    if incumbent_row is not None:
        incumbent_contact = terrain_rows.contact_future[int(incumbent_row)]
        contact_cost = config.contact_timing_weight * torch.mean(
            torch.abs(
                terrain_rows.contact_future[candidates] - incumbent_contact
            ),
            dim=(1, 2),
        )
    pose_cost = torch.zeros_like(terrain_cost)
    if current_joint_position is not None:
        if (
            tuple(current_joint_position.shape) != (29,)
            or current_joint_position.device != database.device
        ):
            raise ContractError("current joint position must be a device-local 29-vector")
        pose_cost = pose_cost + config.joint_position_weight * torch.mean(
            torch.square(
                (
                    terrain_rows.joint_position[candidates]
                    - current_joint_position
                )
                / config.joint_position_scale_rad
            ),
            dim=1,
        )
    if current_joint_velocity is not None:
        if (
            tuple(current_joint_velocity.shape) != (29,)
            or current_joint_velocity.device != database.device
        ):
            raise ContractError("current joint velocity must be a device-local 29-vector")
        pose_cost = pose_cost + config.joint_velocity_weight * torch.mean(
            torch.square(
                (
                    terrain_rows.joint_velocity[candidates]
                    - current_joint_velocity
                )
                / config.joint_velocity_scale_rad_s
            ),
            dim=1,
        )
    effective_transition_penalty = float(config.transition_penalty)
    if incumbent_row is not None:
        incumbent_locations = torch.nonzero(
            candidates == int(incumbent_row), as_tuple=False
        ).flatten()
        if len(incumbent_locations):
            incumbent_rms = float(
                terrain_rms[int(incumbent_locations[0])].detach().cpu().item()
            )
            release = float(
                np.clip(
                    (
                        incumbent_rms
                        - config.transition_release_rms_start_m
                    )
                    / max(
                        config.transition_release_rms_stop_m
                        - config.transition_release_rms_start_m,
                        1.0e-6,
                    ),
                    0.0,
                    1.0,
                )
            )
            incumbent_feature_cost = float(
                feature_cost[int(incumbent_row)].detach().cpu().item()
            )
            feature_release = float(
                np.clip(
                    (
                        incumbent_feature_cost
                        - config.transition_release_feature_start
                    )
                    / max(
                        config.transition_release_feature_stop
                        - config.transition_release_feature_start,
                        1.0e-6,
                    ),
                    0.0,
                    1.0,
                )
            )
            release = max(release, feature_release)
            effective_transition_penalty = (
                config.compatible_transition_penalty * (1.0 - release)
                + config.transition_penalty * release
            )
    total = (
        feature_cost[candidates]
        + terrain_cost
        + contact_cost
        + pose_cost
        + effective_transition_penalty
    )
    if incumbent_row is not None:
        total = torch.where(
            candidates == int(incumbent_row),
            total - effective_transition_penalty,
            total,
        )
    if not bool(torch.isfinite(total).any().item()):
        raise ContractError(
            "generic terrain search found no candidate supported by the target "
            "surface; best prospective clearance was "
            f"{float(torch.amax(minimum_target_clearance).item()):.6f} m and "
            "minimum maximum relative clearance deficit was "
            f"{float(torch.amin(maximum_flight_clearance_deficit).item()):.6f} m"
        )
    best_local = int(torch.argmin(total).item())
    best_row = int(candidates[best_local].item())
    values = torch.stack(
        (
            feature_cost[best_row],
            terrain_cost[best_local],
            contact_cost[best_local],
            pose_cost[best_local],
            terrain_rms[best_local],
            minimum_target_clearance[best_local],
            maximum_flight_clearance_deficit[best_local],
            total[best_local],
        )
    ).detach().cpu().tolist()
    return TerrainCandidateSelection(
        selected_row=best_row,
        selected_clip_index=int(database._search_clip_index[best_row].item()),
        selected_frame_index=int(database._search_frame_index[best_row].item()),
        feature_cost=float(values[0]),
        terrain_cost=float(values[1]),
        contact_cost=float(values[2]),
        pose_cost=float(values[3]),
        terrain_rms_m=float(values[4]),
        minimum_prospective_foot_clearance_m=float(values[5]),
        maximum_prospective_flight_clearance_deficit_m=float(values[6]),
        total_cost=float(values[7]),
        preselection_count=count,
    )


class ContinuousTerrainMotionMatcher:
    """Rolling terrain-conditioned motion matcher with no obstacle labels.

    ``TorchMotionMatcher`` remains responsible for command filtering,
    inertialisation, dense windows, and transactional commits.  This wrapper
    replaces only its ordinary nearest-neighbour choice at each search tick:
    it previews the exact same filtered command, re-ranks source rows by the
    known target surface under their prospective feet, and then asks the base
    matcher to play the selected row.

    A newly selected source row is rigidly registered root-to-root in XYZ and
    yaw.  That is the coordinate transform assumed by
    :func:`select_terrain_candidate`: future source-foot offsets are sampled
    around the generated character's current root.  Between search ticks the
    controller advances to the selected clip's ordinary successor, preserving
    both source timing and the established registration.
    """

    def __init__(
        self,
        library: CanonicalTerrainLibrary,
        matcher: TorchMotionMatcher,
        target_height_field: MeshHeightField,
        *,
        terrain_rows: TerrainRowFeatures | None = None,
        sole_geometry: SoleGeometry | None = None,
        search_config: GenericTerrainSearchConfig = GenericTerrainSearchConfig(),
        runtime_config: ContinuousTerrainRuntimeConfig = ContinuousTerrainRuntimeConfig(),
    ) -> None:
        if not isinstance(library, CanonicalTerrainLibrary):
            raise ContractError("continuous terrain matcher requires a canonical library")
        if not isinstance(matcher, TorchMotionMatcher):
            raise ContractError("continuous terrain matcher requires TorchMotionMatcher")
        if matcher.folder.inventory_sha256 != library.folder.inventory_sha256:
            raise ContractError("matcher and canonical terrain library disagree")
        if not hasattr(target_height_field, "sample"):
            raise ContractError("target height field must provide sample(points_xy)")
        self.library = library
        self.matcher = matcher
        self.target_height_field = target_height_field
        self.terrain_rows = (
            build_terrain_row_features(
                library, matcher.database, sole_geometry=sole_geometry
            )
            if terrain_rows is None
            else terrain_rows
        )
        self.search_config = search_config
        self.runtime_config = runtime_config
        layout = library.folder.layout
        reference = library.canonical_clips[0]
        foot_indices = np.asarray(
            (layout.left_foot_body_index, layout.right_foot_body_index),
            dtype=np.int64,
        )
        foot_quaternion = np.asarray(
            reference.body_quaternion_world_wxyz[:, foot_indices],
            dtype=np.float64,
        )
        inverse_quaternion = foot_quaternion.copy()
        inverse_quaternion[..., 1:] *= -1.0
        if sole_geometry is None:
            world_probe = np.stack(
                (
                    reference.sole_position_world,
                    reference.heel_position_world,
                    reference.toe_position_world,
                ),
                axis=2,
            )
            world_offset = np.asarray(
                world_probe
                - reference.body_position_world[:, foot_indices, None, :],
                dtype=np.float64,
            )
            probe_offset_body = np.median(
                _rotate_wxyz_numpy(
                    inverse_quaternion[:, :, None, :], world_offset
                ),
                axis=0,
            )
        else:
            # Index zero stays the sole centre for output grounding and drift
            # diagnostics; the following four entries are the exact corners.
            probe_offset_body = np.concatenate(
                (
                    np.asarray(
                        sole_geometry.sole_positions_body, dtype=np.float64
                    )[:, None],
                    np.asarray(
                        sole_geometry.corner_positions_body, dtype=np.float64
                    ),
                ),
                axis=1,
            )
        self._foot_probe_offset_body = torch.as_tensor(
            probe_offset_body, dtype=torch.float32, device=matcher.device
        )
        self.last_search_event: ContinuousTerrainSearchEvent | None = None
        self._pending_search_events: dict[int, ContinuousTerrainSearchEvent | None] = {}

    @classmethod
    def from_library(
        cls,
        library: CanonicalTerrainLibrary,
        target_height_field: MeshHeightField,
        *,
        device: str | torch.device = "auto",
        matcher_config: MatcherConfig = MatcherConfig(trajectory_model="takara_ball"),
        search_config: GenericTerrainSearchConfig = GenericTerrainSearchConfig(),
        runtime_config: ContinuousTerrainRuntimeConfig = ContinuousTerrainRuntimeConfig(),
        sole_geometry: SoleGeometry | None = None,
    ) -> "ContinuousTerrainMotionMatcher":
        matcher = TorchMotionMatcher.from_motion_folder(
            library.folder,
            device=device,
            config=matcher_config,
        )
        return cls(
            library,
            matcher,
            target_height_field,
            sole_geometry=sole_geometry,
            search_config=search_config,
            runtime_config=runtime_config,
        )

    @property
    def current_root_world_yaw(self) -> float:
        return self.matcher.current_root_world_yaw

    @property
    def motion_inventory_sha256(self) -> str:
        return self.matcher.motion_inventory_sha256

    @property
    def database(self) -> TorchMotionDatabase:
        return self.matcher.database

    @property
    def config(self) -> MatcherConfig:
        return self.matcher.config

    @property
    def device(self) -> torch.device:
        return self.matcher.device

    def _generated_feature_state(self) -> GeneratedFeatureState:
        state = self.matcher._state
        if state is None:
            raise ContractError("continuous terrain matcher must be reset first")
        layout = self.library.folder.layout
        return GeneratedFeatureState(
            root_position_world=state.root_position,
            root_orientation_world_wxyz=state.root_quaternion,
            root_linear_velocity_world=state.root_linear_velocity,
            left_foot_position_world=state.feature_body_position[
                layout.left_foot_body_index
            ],
            right_foot_position_world=state.feature_body_position[
                layout.right_foot_body_index
            ],
            left_foot_velocity_world=state.feature_body_velocity[
                layout.left_foot_body_index
            ],
            right_foot_velocity_world=state.feature_body_velocity[
                layout.right_foot_body_index
            ],
        )

    def _preview_command(
        self,
        requested_velocity_world_xy: tuple[float, float],
        requested_heading_world_yaw: float,
        *,
        successor: int | None,
    ) -> ShapedCommand:
        state = self.matcher._state
        if state is None:
            raise ContractError("continuous terrain matcher must be reset first")
        requested_velocity = torch.tensor(
            requested_velocity_world_xy,
            dtype=torch.float32,
            device=self.device,
        )
        requested_heading = torch.tensor(
            requested_heading_world_yaw,
            dtype=torch.float32,
            device=self.device,
        )
        if self.config.trajectory_model == "takara_ball":
            return predict_takara_ball_trajectory(
                state.simulation_position,
                state.simulation_velocity,
                state.simulation_acceleration,
                state.simulation_heading,
                state.simulation_heading_velocity,
                requested_velocity,
                requested_heading,
                has_valid_successor=successor is not None,
                config=self.config,
            ).command
        return predict_command_trajectory(
            state.root_position[:2],
            state.shaped_velocity,
            state.shaped_heading,
            requested_velocity,
            requested_heading,
            has_valid_successor=successor is not None,
            config=self.config,
        )

    def _terrain_adapted_command(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
    ) -> tuple[tuple[float, float], float]:
        """Smoothly align unsupported oblique travel with steep terrain."""

        if not self.runtime_config.terrain_command_adaptation:
            return velocity_world_xy, heading_world_yaw
        state = self.matcher._state
        if state is None:
            raise ContractError("continuous terrain matcher must be reset first")
        requested = np.asarray(velocity_world_xy, dtype=np.float64)
        speed = float(np.linalg.norm(requested))
        if speed < 1.0e-5:
            return velocity_world_xy, heading_world_yaw
        direction = requested / speed
        root_xy = state.root_position[:2].detach().cpu().numpy().astype(np.float64)
        distances = np.asarray((0.20, 0.35, 0.50, 0.70), dtype=np.float64)
        points = root_xy[None] + distances[:, None] * direction[None]
        current_height, _normal, current_hit = self.target_height_field.sample(
            root_xy[None]
        )
        height, normal, hit = self.target_height_field.sample(points)
        usable = np.asarray(hit) & bool(current_hit[0])
        if not np.any(usable):
            return velocity_world_xy, heading_world_yaw
        rise = np.abs(np.asarray(height) - float(current_height[0]))
        horizontal_normal = np.linalg.norm(np.asarray(normal)[..., :2], axis=-1)
        score = horizontal_normal + rise / np.maximum(distances, 1.0e-6)
        score[~usable] = -np.inf
        probe = int(np.argmax(score))
        steepness = float(horizontal_normal[probe])
        height_change = float(rise[probe])
        if steepness < 0.08 and height_change < 0.045:
            return velocity_world_xy, heading_world_yaw
        gradient = -np.asarray(normal[probe, :2], dtype=np.float64)
        magnitude = float(np.linalg.norm(gradient))
        if magnitude < 1.0e-6:
            return velocity_world_xy, heading_world_yaw
        gradient /= magnitude
        crossing = float(np.dot(direction, gradient))
        terrain_strength = float(
            np.clip(max(steepness / 0.20, height_change / 0.12), 0.0, 1.0)
        )
        if abs(crossing) >= 0.10:
            feasible = math.copysign(1.0, crossing) * gradient
            intent = float(np.clip((abs(crossing) - 0.05) / 0.55, 0.0, 1.0))
            blend = terrain_strength * (0.55 + 0.45 * intent)
        else:
            # A nearly side-on command should travel along the obstacle edge,
            # not slowly push a planted foot into the riser.
            feasible = direction - crossing * gradient
            feasible_norm = float(np.linalg.norm(feasible))
            if feasible_norm < 1.0e-6:
                feasible = direction
            else:
                feasible /= feasible_norm
            blend = 0.80 * terrain_strength
        adapted_direction = (1.0 - blend) * direction + blend * feasible
        adapted_direction /= max(float(np.linalg.norm(adapted_direction)), 1.0e-6)
        adapted_speed = speed * (
            1.0
            - 0.30 * terrain_strength * (1.0 - min(abs(crossing), 1.0))
        )
        adapted_velocity = adapted_speed * adapted_direction
        travel_heading = math.atan2(adapted_direction[1], adapted_direction[0])
        heading_delta = math.atan2(
            math.sin(travel_heading - heading_world_yaw),
            math.cos(travel_heading - heading_world_yaw),
        )
        adapted_heading = heading_world_yaw + 0.75 * terrain_strength * heading_delta
        return (
            (float(adapted_velocity[0]), float(adapted_velocity[1])),
            float(adapted_heading),
        )

    def _root_registration(self, row: int) -> ForcedMotionAlignment:
        state = self.matcher._state
        if state is None:
            raise ContractError("continuous terrain matcher must be reset first")
        clip_index = int(self.database._search_clip_index[row].item())
        frame_index = int(self.database._search_frame_index[row].item())
        clip = self.library.canonical_clips[clip_index]
        root = self.library.folder.layout.root_body_index
        source_position = np.asarray(
            clip.body_position_world[frame_index, root], dtype=np.float64
        )
        source_yaw = float(
            _yaw_from_wxyz(clip.body_quaternion_world_wxyz[frame_index, root])
        )
        current_quaternion = state.root_quaternion
        w, x, y, z = (float(value) for value in current_quaternion.detach().cpu())
        current_yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        yaw_offset = math.atan2(
            math.sin(current_yaw - source_yaw),
            math.cos(current_yaw - source_yaw),
        )
        cosine = math.cos(yaw_offset)
        sine = math.sin(yaw_offset)
        rotated_source = np.asarray(
            (
                cosine * source_position[0] - sine * source_position[1],
                sine * source_position[0] + cosine * source_position[1],
                source_position[2],
            ),
            dtype=np.float64,
        )
        current_position = state.root_position.detach().cpu().numpy().astype(np.float64)
        translation = current_position - rotated_source
        return ForcedMotionAlignment(
            yaw_offset_rad=yaw_offset,
            translation_world_xyz=tuple(float(value) for value in translation),
            # Preserve the filtered joystick velocity/facing.  Only the board
            # origin is recentered onto the newly registered character.
            synchronize_simulation_character=False,
        )

    def _source_successor_is_continuous(self, row: int) -> bool:
        """Reject both hard seams and rows that would enter one before re-search."""

        state = self.matcher._state
        if state is None:
            raise ContractError("continuous terrain matcher must be reset first")
        clip_index = int(self.database._search_clip_index[row].item())
        frame_index = int(self.database._search_frame_index[row].item())
        if clip_index != state.clip_index or frame_index != state.frame_index + 1:
            return False
        clip = self.library.canonical_clips[clip_index]
        previous = state.frame_index
        joint_step = float(
            np.max(
                np.abs(
                    np.asarray(clip.joint_position[frame_index])
                    - np.asarray(clip.joint_position[previous])
                )
            )
        )
        root_step = float(
            np.linalg.norm(
                np.asarray(clip.root_position_world[frame_index])
                - np.asarray(clip.root_position_world[previous])
            )
        )
        yaw = _yaw_from_wxyz(
            clip.root_quaternion_world_wxyz[[previous, frame_index]]
        )
        yaw_step = float(
            abs(math.atan2(math.sin(float(yaw[1] - yaw[0])), math.cos(float(yaw[1] - yaw[0]))))
        )
        future_is_safe = bool(
            self.terrain_rows.maximum_future_joint_step_rad[row]
            <= self.search_config.maximum_source_joint_step_rad
        ) and bool(
            self.terrain_rows.maximum_future_root_step_m[row]
            <= self.search_config.maximum_source_root_step_m
        ) and bool(
            self.terrain_rows.maximum_future_root_yaw_step_rad[row]
            <= self.search_config.maximum_source_root_yaw_step_rad
        )
        return bool(
            future_is_safe
            and joint_step <= self.search_config.maximum_source_joint_step_rad
            and root_step <= self.search_config.maximum_source_root_step_m
            and yaw_step <= self.search_config.maximum_source_root_yaw_step_rad
        )

    def _vertical_registration(
        self,
        row: int,
        base_alignment: ForcedMotionAlignment,
    ) -> ForcedMotionAlignment:
        """Track the target support height with one bounded common Z offset."""

        if not self.runtime_config.vertical_adaptation:
            return base_alignment
        clip_index = int(self.database._search_clip_index[row].item())
        frame_index = int(self.database._search_frame_index[row].item())
        clip = self.library.canonical_clips[clip_index]
        source_xy = np.asarray(
            clip.sole_position_world[frame_index, :, :2], dtype=np.float64
        )
        source_height, _source_normal, source_hit = self.library.height_fields[
            clip_index
        ].sample(source_xy)
        yaw = float(base_alignment.yaw_offset_rad)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        target_xy = np.stack(
            (
                cosine * source_xy[:, 0] - sine * source_xy[:, 1],
                sine * source_xy[:, 0] + cosine * source_xy[:, 1],
            ),
            axis=-1,
        )
        target_xy += np.asarray(base_alignment.translation_world_xyz[:2])
        target_height, _target_normal, target_hit = self.target_height_field.sample(
            target_xy
        )
        usable = np.asarray(source_hit) & np.asarray(target_hit)
        contact = np.asarray(clip.contact[frame_index]) >= 0.5
        support = usable & contact
        if not np.any(support):
            support = usable
        if not np.any(support):
            return base_alignment
        desired = float(np.median(target_height[support] - source_height[support]))
        current = float(base_alignment.translation_world_xyz[2])
        alpha = 1.0 - math.exp(
            -math.log(2.0) * self.config.dt
            / max(self.runtime_config.vertical_halflife_s, 1.0e-6)
        )
        change = float(
            np.clip(
                alpha * (desired - current),
                -self.runtime_config.maximum_vertical_step_m,
                self.runtime_config.maximum_vertical_step_m,
            )
        )
        return ForcedMotionAlignment(
            yaw_offset_rad=base_alignment.yaw_offset_rad,
            translation_world_xyz=(
                float(base_alignment.translation_world_xyz[0]),
                float(base_alignment.translation_world_xyz[1]),
                current + change,
            ),
            synchronize_simulation_character=False,
        )

    def _generated_sole_positions(self, result: MotionMatchResult) -> torch.Tensor:
        return self._generated_foot_probe_positions(result)[:, 0]

    def _generated_foot_probe_positions(
        self, result: MotionMatchResult
    ) -> torch.Tensor:
        layout = self.library.folder.layout
        indices = torch.as_tensor(
            (layout.left_foot_body_index, layout.right_foot_body_index),
            dtype=torch.long,
            device=self.device,
        )
        position = result.body_position_world[indices, None, :]
        quaternion = result.body_orientation_world_wxyz[indices, None, :]
        return position + _rotate_wxyz_torch(
            quaternion, self._foot_probe_offset_body
        )

    def _output_grounding_correction(
        self,
        result: MotionMatchResult,
        row: int,
    ) -> float:
        """Return a bounded common-Z correction for the actual blended pose."""

        clip_index = int(self.database._search_clip_index[row].item())
        frame_index = int(self.database._search_frame_index[row].item())
        contact = np.asarray(
            self.library.canonical_clips[clip_index].contact[frame_index]
        ) >= 0.5
        if not np.any(contact):
            return 0.0
        sole = self._generated_sole_positions(result).detach().cpu().numpy()
        height, _normal, hit = self.target_height_field.sample(sole[:, :2])
        support = contact & np.asarray(hit)
        if not np.any(support):
            return 0.0
        residual = np.asarray(height)[support] - sole[support, 2]
        # Raise by the worst supported penetration.  When all support soles
        # hover, settle toward their median rather than snapping downward.
        desired = (
            float(np.max(residual))
            if float(np.max(residual)) > 0.0
            else float(np.median(residual))
        )
        return float(
            np.clip(
                desired,
                -self.runtime_config.maximum_vertical_step_m,
                self.runtime_config.maximum_vertical_step_m,
            )
        )

    def reset_to_row(
        self,
        row: int,
        *,
        alignment: ForcedMotionAlignment | None = None,
    ) -> MotionMatchResult:
        self.last_search_event = None
        self._pending_search_events.clear()
        return self.matcher.reset_to_row(row, alignment=alignment)

    def reset(self) -> MotionMatchResult:
        self.last_search_event = None
        self._pending_search_events.clear()
        return self.matcher.reset()

    def prepare_step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
    ) -> PreparedMotionMatch:
        state = self.matcher._state
        if state is None:
            raise ContractError("continuous terrain matcher must be reset first")
        velocity_world_xy, heading_world_yaw = self._terrain_adapted_command(
            velocity_world_xy, heading_world_yaw
        )
        successor = self.database.row_for_source(
            state.clip_index, state.frame_index + 1
        )
        if successor is not None and not self._source_successor_is_continuous(
            int(successor)
        ):
            successor = None
        shaped = self._preview_command(
            velocity_world_xy,
            heading_world_yaw,
            successor=successor,
        )
        do_search = (
            successor is None
            or search_is_due(state.sequence, shaped.force_search, self.config)
        )
        selection: TerrainCandidateSelection | None = None
        event: ContinuousTerrainSearchEvent | None = None
        selected_row = successor
        alignment = None
        if do_search:
            selection = select_terrain_candidate(
                self.database,
                self.terrain_rows,
                self._generated_feature_state(),
                shaped.trajectory,
                self.target_height_field,
                current_clip_index=state.clip_index,
                current_frame_index=state.frame_index,
                incumbent_row=successor,
                current_joint_position=state.joint_position,
                current_joint_velocity=state.joint_velocity,
                config=self.search_config,
            )
            selected_row = selection.selected_row
            transitioned = successor is None or selected_row != successor
            if transitioned:
                alignment = self._root_registration(selected_row)
            event = ContinuousTerrainSearchEvent(
                sequence=state.sequence + 1,
                selection=selection,
                source_clip_id=self.library.clip_ids[
                    selection.selected_clip_index
                ],
                transitioned=transitioned,
                alignment=alignment,
            )
        if selected_row is None:
            raise ContractError("terrain search did not produce a playable row")
        if self.runtime_config.vertical_adaptation:
            if alignment is None:
                base_alignment = ForcedMotionAlignment(
                    yaw_offset_rad=float(state.yaw_offset.detach().cpu().item()),
                    translation_world_xyz=(
                        float(state.translation_xy[0].detach().cpu().item()),
                        float(state.translation_xy[1].detach().cpu().item()),
                        float(state.height_offset.detach().cpu().item()),
                    ),
                    synchronize_simulation_character=False,
                )
            else:
                base_alignment = alignment
            adjusted_alignment = self._vertical_registration(
                int(selected_row), base_alignment
            )
            changed_height = (
                abs(
                    adjusted_alignment.translation_world_xyz[2]
                    - base_alignment.translation_world_xyz[2]
                )
                > 1.0e-8
            )
            if alignment is not None or changed_height:
                alignment = adjusted_alignment
                if event is not None:
                    event = ContinuousTerrainSearchEvent(
                        sequence=event.sequence,
                        selection=event.selection,
                        source_clip_id=event.source_clip_id,
                        transitioned=event.transitioned,
                        alignment=alignment,
                    )
        prepared = self.matcher.prepare_step(
            velocity_world_xy,
            heading_world_yaw,
            dt=dt,
            forced_row=int(selected_row),
            forced_alignment=alignment,
        )
        if (
            self.runtime_config.vertical_adaptation
            and self.runtime_config.output_grounding
        ):
            grounding_change = self._output_grounding_correction(
                prepared.result, int(selected_row)
            )
            if abs(grounding_change) > 1.0e-8:
                if alignment is None:
                    alignment = ForcedMotionAlignment(
                        yaw_offset_rad=float(state.yaw_offset.detach().cpu().item()),
                        translation_world_xyz=(
                            float(state.translation_xy[0].detach().cpu().item()),
                            float(state.translation_xy[1].detach().cpu().item()),
                            float(state.height_offset.detach().cpu().item()),
                        ),
                        synchronize_simulation_character=False,
                    )
                alignment = ForcedMotionAlignment(
                    yaw_offset_rad=alignment.yaw_offset_rad,
                    translation_world_xyz=(
                        alignment.translation_world_xyz[0],
                        alignment.translation_world_xyz[1],
                        alignment.translation_world_xyz[2] + grounding_change,
                    ),
                    synchronize_simulation_character=False,
                )
                prepared = self.matcher.prepare_step(
                    velocity_world_xy,
                    heading_world_yaw,
                    dt=dt,
                    forced_row=int(selected_row),
                    forced_alignment=alignment,
                )
                if event is not None:
                    event = ContinuousTerrainSearchEvent(
                        sequence=event.sequence,
                        selection=event.selection,
                        source_clip_id=event.source_clip_id,
                        transitioned=event.transitioned,
                        alignment=alignment,
                    )
        self._pending_search_events[id(prepared)] = event
        return prepared

    def commit(self, prepared: PreparedMotionMatch) -> MotionMatchResult:
        result = self.matcher.commit(prepared)
        event = self._pending_search_events.pop(id(prepared), None)
        if event is not None:
            self.last_search_event = event
        return result

    def step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        dt: float = 0.02,
    ) -> MotionMatchResult:
        return self.commit(
            self.prepare_step(
                velocity_world_xy,
                heading_world_yaw,
                dt=dt,
            )
        )


def generated_state_from_clip(
    clip: CanonicalClip,
    frame_index: int,
    *,
    device: str | torch.device,
) -> GeneratedFeatureState:
    """Build the ordinary MM query state for one canonical clean frame."""

    dev = torch.device(device)
    frame = int(frame_index)
    left = G1_TAKARA_LAYOUT.left_foot_body_index
    right = G1_TAKARA_LAYOUT.right_foot_body_index
    root = G1_TAKARA_LAYOUT.root_body_index
    # Canonical arrays are intentionally read-only; own the tiny query frame
    # so Torch never exposes a writable view onto immutable corpus storage.
    position = torch.tensor(clip.body_position_world[frame], device=dev)
    velocity = torch.tensor(clip.body_linear_velocity_world[frame], device=dev)
    return GeneratedFeatureState(
        root_position_world=position[root],
        root_orientation_world_wxyz=torch.tensor(
            clip.body_quaternion_world_wxyz[frame, root], device=dev
        ),
        root_linear_velocity_world=velocity[root],
        left_foot_position_world=position[left],
        right_foot_position_world=position[right],
        left_foot_velocity_world=velocity[left],
        right_foot_velocity_world=velocity[right],
    )


def exact_trajectory_from_clip(
    clip: CanonicalClip,
    frame_index: int,
    *,
    device: str | torch.device,
) -> CommandTrajectory:
    """Return the authored 0.3/0.6/0.9-second root trajectory query."""

    dev = torch.device(device)
    frames = int(frame_index) + np.asarray(FEATURE_HORIZON_FRAMES)
    root = G1_TAKARA_LAYOUT.root_body_index
    position = torch.tensor(
        clip.body_position_world[frames, root, :2], device=dev
    )
    yaw = _yaw_from_wxyz(clip.body_quaternion_world_wxyz[frames, root])
    facing = torch.as_tensor(
        np.stack((np.cos(yaw), np.sin(yaw)), axis=-1).astype(np.float32),
        device=dev,
    )
    return CommandTrajectory(position_world_xy=position, facing_world_xy=facing)
