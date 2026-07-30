"""Strict native-Z-up terrain features for the Torch motion matcher.

The database and live query paths use the same height-grid sampler and feature
definitions.  Recorded clips sample their own aligned scene grids directly.
Live queries first map matcher world coordinates into the selected recorded
scene, keeping global scene placement out of the search feature itself.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_data import G1_TAKARA_LAYOUT, MotionClip, MotionFolder
from .torch_motion_features import (
    CommandTrajectory,
    GeneratedFeatureState,
    resolve_torch_device,
)


DENSE_FORWARD_M = tuple(-0.15 + 0.15 * index for index in range(13))
DENSE_LATERAL_M = tuple(-0.45 + 0.15 * index for index in range(7))
LEGACY_DISTANCE_M = (0.25, 0.50, 0.75, 1.00)
TERRAIN_DATASET_SCHEMA = "g1-torch-stair-slice/v1"
EXPANDED_TERRAIN_DATASET_SCHEMA = "g1-torch-terrain-corpus/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_xy(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate trailing-dimension planar vectors by positive Z-up ``yaw``."""
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack(
        (cosine * x - sine * y, sine * x + cosine * y), dim=-1
    )


@dataclass(frozen=True)
class TerrainSceneAlignment:
    """Rigid planar map from matcher world into one recorded scene."""

    translation_scene_xy: torch.Tensor
    yaw_scene_from_matcher: torch.Tensor

    def __post_init__(self) -> None:
        translation = self.translation_scene_xy
        yaw = self.yaw_scene_from_matcher
        if (
            not isinstance(translation, torch.Tensor)
            or tuple(translation.shape) != (2,)
            or not translation.dtype.is_floating_point
            or not torch.isfinite(translation).all()
        ):
            raise ContractError(
                "terrain scene translation must be a finite floating shape-(2,) tensor"
            )
        if (
            not isinstance(yaw, torch.Tensor)
            or yaw.numel() != 1
            or yaw.device != translation.device
            or yaw.dtype != translation.dtype
            or not torch.isfinite(yaw).all()
        ):
            raise ContractError(
                "terrain scene yaw must be a finite scalar matching translation"
            )

    def matcher_to_scene_xy(self, points_xy: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(points_xy, torch.Tensor)
            or points_xy.ndim < 1
            or points_xy.shape[-1] != 2
            or points_xy.device != self.translation_scene_xy.device
            or points_xy.dtype != self.translation_scene_xy.dtype
            or not torch.isfinite(points_xy).all()
        ):
            raise ContractError(
                "terrain alignment points must be finite [..., 2] tensors "
                "matching alignment device/dtype"
            )
        return (
            _rotate_xy(points_xy, self.yaw_scene_from_matcher)
            + self.translation_scene_xy
        )


@dataclass(frozen=True)
class _TorchHeightGrid:
    origin_xy: torch.Tensor
    cell_size_m: float
    height_z: torch.Tensor

    @staticmethod
    def load(
        path: Path, device: torch.device
    ) -> "_TorchHeightGrid":
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "origin_xy",
                    "cell_size_m",
                    "height_z",
                }:
                    raise ValueError("height grid archive fields are invalid")
                origin = np.asarray(archive["origin_xy"], np.float32)
                cell_value = np.asarray(archive["cell_size_m"])
                height = np.asarray(archive["height_z"], np.float32)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"cannot load height grid: {path}") from error
        if (
            origin.shape != (2,)
            or not np.isfinite(origin).all()
            or cell_value.size != 1
            or height.ndim != 2
            or min(height.shape) < 2
            or not np.isfinite(height).all()
        ):
            raise ValueError("height grid arrays are invalid")
        cell = float(cell_value.reshape(-1)[0])
        if not math.isfinite(cell) or cell <= 0.0:
            raise ValueError("height grid cell size must be finite and positive")
        return _TorchHeightGrid(
            origin_xy=torch.tensor(
                origin, dtype=torch.float32, device=device
            ),
            cell_size_m=cell,
            height_z=torch.tensor(
                height, dtype=torch.float32, device=device
            ),
        )

    def sample_xy(self, points_xy: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(points_xy, torch.Tensor)
            or points_xy.ndim < 1
            or points_xy.shape[-1] != 2
            or points_xy.device != self.height_z.device
            or points_xy.dtype != torch.float32
            or not torch.isfinite(points_xy).all()
        ):
            raise ContractError(
                "height-grid queries must be finite float32 [..., 2] tensors "
                "on the grid device"
            )
        shape = points_xy.shape[:-1]
        coordinate = (
            points_xy.reshape(-1, 2) - self.origin_xy
        ) / self.cell_size_m
        ny, nx = self.height_z.shape
        tolerance = 1e-5
        outside = (
            (coordinate[:, 0] < -tolerance)
            | (coordinate[:, 1] < -tolerance)
            | (coordinate[:, 0] > nx - 1 + tolerance)
            | (coordinate[:, 1] > ny - 1 + tolerance)
        )
        if bool(outside.any().item()):
            raise ContractError(
                "height-grid query is outside the authoritative terrain domain"
            )

        x = coordinate[:, 0].clamp(0.0, float(nx - 1))
        y = coordinate[:, 1].clamp(0.0, float(ny - 1))
        ix = torch.floor(x).to(torch.long).clamp(max=nx - 2)
        iy = torch.floor(y).to(torch.long).clamp(max=ny - 2)
        tx = x - ix
        ty = y - iy
        h00 = self.height_z[iy, ix]
        h10 = self.height_z[iy, ix + 1]
        h01 = self.height_z[iy + 1, ix]
        h11 = self.height_z[iy + 1, ix + 1]

        # Match ZUpHeightGrid's fixed lower-left-to-upper-right diagonal.
        first = h00 + tx * (h10 - h00) + ty * (h11 - h10)
        second = h00 + tx * (h11 - h01) + ty * (h01 - h00)
        return torch.where(tx >= ty, first, second).reshape(shape)


@dataclass(frozen=True)
class TerrainDataset:
    """Validated motion folder plus one optional aligned grid per clip."""

    root: Path
    folder: MotionFolder
    clip_grids: Sequence[_TorchHeightGrid | None]
    manifest_sha256: str
    _manifest: dict
    device: torch.device

    @property
    def manifest(self) -> dict:
        return deepcopy(self._manifest)

    @staticmethod
    def load(
        root: str | Path, *, device: str | torch.device
    ) -> "TerrainDataset":
        root_path = Path(root).resolve()
        manifest_path = root_path / "manifest.json"
        if not manifest_path.is_file():
            raise ContractError(f"terrain manifest is missing: {manifest_path}")
        raw_manifest = manifest_path.read_bytes()
        try:
            manifest = json.loads(raw_manifest)
        except Exception as error:
            raise ContractError("terrain manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise ContractError("terrain manifest must be an object")
        schema = manifest.get("schema")
        if schema not in (
            TERRAIN_DATASET_SCHEMA,
            EXPANDED_TERRAIN_DATASET_SCHEMA,
        ):
            raise ContractError("terrain manifest schema is invalid")
        if manifest.get("output_fps") != 50:
            raise ContractError("terrain manifest output_fps must equal 50")
        if manifest.get("layout") != G1_TAKARA_LAYOUT.identity:
            raise ContractError("terrain manifest layout is invalid")
        descriptors = manifest.get("clips")
        if not isinstance(descriptors, list) or not descriptors:
            raise ContractError("terrain manifest must describe clips")
        if schema == TERRAIN_DATASET_SCHEMA and len(descriptors) != 5:
            raise ContractError(
                "terrain manifest must describe exactly five clips"
            )
        if (
            schema == EXPANDED_TERRAIN_DATASET_SCHEMA
            and manifest.get("accepted_clips") != descriptors
        ):
            raise ContractError(
                "expanded terrain accepted clips must match clip inventory"
            )

        resolved_device = resolve_torch_device(device)
        folder = MotionFolder.load(root_path)
        if len(folder.clips) != len(descriptors):
            raise ContractError("terrain manifest clip count does not match motion folder")

        grids: list[_TorchHeightGrid | None] = []
        for index, (clip, descriptor) in enumerate(
            zip(folder.clips, descriptors)
        ):
            if not isinstance(descriptor, dict):
                raise ContractError(f"terrain clip descriptor {index} is invalid")
            relative = descriptor.get("relative_motion_path")
            if relative != clip.relative_path:
                raise ContractError(
                    "terrain manifest clip mapping does not exactly match "
                    f"motion inventory at index {index}"
                )
            motion_path = root_path / clip.relative_path
            expected_motion_hash = descriptor.get("motion_sha256")
            if (
                not isinstance(expected_motion_hash, str)
                or _sha256(motion_path) != expected_motion_hash
            ):
                raise ContractError(
                    f"terrain manifest motion hash mismatch: {clip.relative_path}"
                )
            terrain = descriptor.get("terrain")
            if not isinstance(terrain, dict):
                raise ContractError(
                    f"terrain descriptor is invalid: {clip.relative_path}"
                )
            kind = terrain.get("kind")
            if kind == "flat":
                if index != 0 or descriptor.get("kind") != "flat":
                    raise ContractError(
                        "only the first flat control clip may omit a height grid"
                    )
                grids.append(None)
                continue
            expected_kinds = (
                ("stair",)
                if schema == TERRAIN_DATASET_SCHEMA
                else ("terrain",)
            )
            if (
                kind != "heightgrid"
                or descriptor.get("kind") not in expected_kinds
            ):
                raise ContractError(
                    f"terrain kind is invalid: {clip.relative_path}"
                )
            relative_grid = terrain.get("path")
            expected_grid_hash = terrain.get("sha256")
            if (
                not isinstance(relative_grid, str)
                or not relative_grid
                or not isinstance(expected_grid_hash, str)
            ):
                raise ContractError(
                    f"terrain grid identity is invalid: {clip.relative_path}"
                )
            grid_path = root_path / relative_grid
            if not grid_path.is_file() or _sha256(grid_path) != expected_grid_hash:
                raise ContractError(
                    f"terrain grid hash mismatch: {clip.relative_path}"
                )
            try:
                torch_grid = _TorchHeightGrid.load(
                    grid_path, resolved_device
                )
            except Exception as error:
                raise ContractError(
                    f"terrain grid cannot be loaded: {clip.relative_path}"
                ) from error
            shape = terrain.get("shape")
            origin = terrain.get("origin_xy")
            cell = terrain.get("cell_size_m")
            expected_origin = [
                float(value) for value in torch_grid.origin_xy.cpu().tolist()
            ]
            if (
                shape != list(torch_grid.height_z.shape)
                or not isinstance(origin, list)
                or len(origin) != 2
                or any(
                    not isinstance(value, (int, float))
                    or not math.isclose(
                        float(value), expected, rel_tol=0.0, abs_tol=1e-6
                    )
                    for value, expected in zip(origin, expected_origin)
                )
                or not isinstance(cell, (int, float))
                or not math.isclose(
                    float(cell),
                    torch_grid.cell_size_m,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
            ):
                raise ContractError(
                    f"terrain grid metadata mismatch: {clip.relative_path}"
                )
            grids.append(torch_grid)

        return TerrainDataset(
            root=root_path,
            folder=folder,
            clip_grids=tuple(grids),
            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
            _manifest=deepcopy(manifest),
            device=resolved_device,
        )


def _dense_local_points(
    *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    forward = torch.tensor(DENSE_FORWARD_M, device=device, dtype=dtype)
    lateral = torch.tensor(DENSE_LATERAL_M, device=device, dtype=dtype)
    grid_forward, grid_lateral = torch.meshgrid(
        forward, lateral, indexing="ij"
    )
    return torch.stack((grid_forward, grid_lateral), dim=-1).reshape(-1, 2)


def _dense_rows(
    root_xy: torch.Tensor,
    root_yaw: torch.Tensor,
    grid: _TorchHeightGrid,
) -> torch.Tensor:
    local = _dense_local_points(device=root_xy.device, dtype=root_xy.dtype)
    scene_points = root_xy[:, None, :] + _rotate_xy(
        local[None, :, :], root_yaw[:, None]
    )
    heights = grid.sample_xy(scene_points)
    base = grid.sample_xy(root_xy)
    return heights - base[:, None]


def _facing_unit(
    facings_xy: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    if facings_xy.numel():
        for index in range(len(facings_xy) - 1, -1, -1):
            candidate = facings_xy[index]
            norm = torch.linalg.vector_norm(candidate)
            if bool((norm > 1e-6).item()):
                return candidate / norm
    return fallback / torch.linalg.vector_norm(fallback).clamp_min(1e-6)


def _arc_distance_points(
    root_xy: torch.Tensor,
    future_positions_xy: torch.Tensor,
    future_facings_xy: torch.Tensor,
    root_facing_xy: torch.Tensor,
) -> torch.Tensor:
    """Sample a predicted centerline, extending short/stopped paths by facing."""
    polyline = torch.cat((root_xy[None, :], future_positions_xy), dim=0)
    segments = polyline[1:] - polyline[:-1]
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    fallback = _facing_unit(future_facings_xy, root_facing_xy)
    output: list[torch.Tensor] = []
    distances = torch.tensor(
        LEGACY_DISTANCE_M, device=root_xy.device, dtype=root_xy.dtype
    )
    for target in distances:
        traversed = torch.zeros((), device=root_xy.device, dtype=root_xy.dtype)
        selected: torch.Tensor | None = None
        last_point = root_xy
        for index in range(len(segments)):
            length = lengths[index]
            if bool((length > 1e-6).item()):
                if bool((traversed + length >= target).item()):
                    fraction = (target - traversed) / length
                    selected = polyline[index] + fraction * segments[index]
                    break
                traversed = traversed + length
                last_point = polyline[index + 1]
        if selected is None:
            selected = last_point + (target - traversed) * fallback
        output.append(selected)
    return torch.stack(output)


def _legacy_rows(
    clip: MotionClip,
    grid: _TorchHeightGrid,
    *,
    root_body_index: int,
    device: torch.device,
) -> torch.Tensor:
    position = torch.tensor(
        clip.body_position_world[:, root_body_index, :2],
        dtype=torch.float32,
        device=device,
    )
    quaternion = torch.tensor(
        clip.body_quaternion_world_wxyz[:, root_body_index],
        dtype=torch.float32,
        device=device,
    )
    yaw = _yaw_from_wxyz(quaternion)
    facing = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    frames = torch.arange(clip.valid_frame_stop, device=device)
    offsets = torch.arange(46, device=device)
    window_indices = frames[:, None] + offsets[None, :]
    position_window = position[window_indices]
    segments = position_window[:, 1:] - position_window[:, :-1]
    length = torch.linalg.vector_norm(segments, dim=-1)
    cumulative = torch.cumsum(length, dim=1)
    targets = torch.tensor(
        LEGACY_DISTANCE_M, dtype=torch.float32, device=device
    )
    crossed = cumulative[:, :, None] >= targets[None, None, :]
    has_crossing = crossed.any(dim=1)
    segment_index = torch.argmax(crossed.to(torch.int64), dim=1)
    gather_xy = segment_index[:, :, None].expand(-1, -1, 2)
    selected_start = torch.gather(
        position_window[:, :-1], 1, gather_xy
    )
    selected_segment = torch.gather(segments, 1, gather_xy)
    selected_length = torch.gather(length, 1, segment_index).clamp_min(1e-8)
    previous_index = (segment_index - 1).clamp_min(0)
    previous = torch.gather(cumulative, 1, previous_index)
    previous = torch.where(
        segment_index == 0, torch.zeros_like(previous), previous
    )
    fraction = (targets[None, :] - previous) / selected_length
    interpolated = selected_start + fraction[:, :, None] * selected_segment

    total = cumulative[:, -1]
    final_facing = facing[frames + 45]
    extended = position_window[:, -1, None, :] + (
        targets[None, :] - total[:, None]
    )[:, :, None] * final_facing[:, None, :]
    samples = torch.where(
        has_crossing[:, :, None], interpolated, extended
    )
    return grid.sample_xy(samples) - grid.sample_xy(position[frames])[:, None]


@dataclass(frozen=True)
class TerrainFeatureExtension:
    """One frozen terrain condition for database construction and live queries."""

    dataset: TerrainDataset
    condition: str
    query_clip_index: int
    query_grid: _TorchHeightGrid
    alignment: TerrainSceneAlignment
    weight: float
    name: str = "terrain"

    @property
    def dimension(self) -> int:
        return 4 if self.condition == "legacy" else 91

    @staticmethod
    def for_condition(
        dataset: TerrainDataset,
        *,
        condition: str,
        query_scene: str,
        weight: float,
    ) -> "TerrainFeatureExtension":
        if not isinstance(dataset, TerrainDataset):
            raise ContractError("terrain dataset is invalid")
        if condition not in ("legacy", "dense"):
            raise ContractError(
                "terrain condition must be either 'legacy' or 'dense'"
            )
        if (
            not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ContractError("terrain feature weight must be finite and positive")
        matching = [
            index
            for index, clip in enumerate(dataset.folder.clips)
            if clip.relative_path == query_scene
        ]
        if len(matching) != 1:
            raise ContractError(
                f"terrain query scene must resolve exactly once: {query_scene}"
            )
        clip_index = matching[0]
        query_grid = dataset.clip_grids[clip_index]
        if query_grid is None:
            raise ContractError("terrain query scene must have a height grid")
        clip = dataset.folder.clips[clip_index]
        root = dataset.folder.layout.root_body_index
        translation = torch.tensor(
            clip.body_position_world[0, root, :2],
            dtype=torch.float32,
            device=dataset.device,
        )
        quaternion = torch.tensor(
            clip.body_quaternion_world_wxyz[0, root],
            dtype=torch.float32,
            device=dataset.device,
        )
        alignment = TerrainSceneAlignment(
            translation_scene_xy=translation,
            yaw_scene_from_matcher=_yaw_from_wxyz(quaternion),
        )
        return TerrainFeatureExtension(
            dataset=dataset,
            condition=condition,
            query_clip_index=clip_index,
            query_grid=query_grid,
            alignment=alignment,
            weight=float(weight),
        )

    def database_rows(
        self, folder: MotionFolder, device: torch.device
    ) -> Sequence[torch.Tensor]:
        if folder is not self.dataset.folder:
            if (
                folder.root != self.dataset.folder.root
                or folder.inventory_sha256 != self.dataset.folder.inventory_sha256
            ):
                raise ContractError(
                    "terrain extension motion folder does not match its dataset"
                )
        if torch.device(device) != self.dataset.device:
            raise ContractError(
                "terrain extension database device does not match its dataset"
            )
        root = folder.layout.root_body_index
        rows: list[torch.Tensor] = []
        for clip, grid in zip(folder.clips, self.dataset.clip_grids):
            if grid is None:
                rows.append(
                    torch.zeros(
                        (clip.valid_frame_stop, self.dimension),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                continue
            if self.condition == "dense":
                position = torch.tensor(
                    clip.body_position_world[: clip.valid_frame_stop, root, :2],
                    dtype=torch.float32,
                    device=device,
                )
                quaternion = torch.tensor(
                    clip.body_quaternion_world_wxyz[
                        : clip.valid_frame_stop, root
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                rows.append(
                    _dense_rows(position, _yaw_from_wxyz(quaternion), grid)
                )
            else:
                rows.append(
                    _legacy_rows(
                        clip,
                        grid,
                        root_body_index=root,
                        device=torch.device(device),
                    )
                )
        return tuple(rows)

    def query_row(
        self, state: GeneratedFeatureState, trajectory: CommandTrajectory
    ) -> torch.Tensor:
        root = state.root_position_world
        if (
            not isinstance(root, torch.Tensor)
            or tuple(root.shape) != (3,)
            or root.device != self.dataset.device
            or root.dtype != torch.float32
            or not torch.isfinite(root).all()
        ):
            raise ContractError(
                "terrain query root must be finite float32 shape (3,) "
                "on the dataset device"
            )
        quaternion = state.root_orientation_world_wxyz
        if (
            not isinstance(quaternion, torch.Tensor)
            or tuple(quaternion.shape) != (4,)
            or quaternion.device != root.device
            or quaternion.dtype != root.dtype
            or not torch.isfinite(quaternion).all()
        ):
            raise ContractError(
                "terrain query orientation must be finite float32 shape (4,)"
            )
        yaw = _yaw_from_wxyz(quaternion)
        root_scene = self.alignment.matcher_to_scene_xy(root[:2])
        if self.condition == "dense":
            local = _dense_local_points(device=root.device, dtype=root.dtype)
            matcher_points = root[:2] + _rotate_xy(local, yaw)
            scene_points = self.alignment.matcher_to_scene_xy(matcher_points)
            return (
                self.query_grid.sample_xy(scene_points)
                - self.query_grid.sample_xy(root_scene)
            ).to(dtype=torch.float32)

        positions = trajectory.position_world_xy
        facings = trajectory.facing_world_xy
        if (
            not isinstance(positions, torch.Tensor)
            or tuple(positions.shape) != (3, 2)
            or positions.device != root.device
            or positions.dtype != root.dtype
            or not torch.isfinite(positions).all()
            or not isinstance(facings, torch.Tensor)
            or tuple(facings.shape) != (3, 2)
            or facings.device != root.device
            or facings.dtype != root.dtype
            or not torch.isfinite(facings).all()
        ):
            raise ContractError(
                "terrain command trajectory must contain finite float32 "
                "shape-(3, 2) tensors on the dataset device"
            )
        root_facing = torch.stack((torch.cos(yaw), torch.sin(yaw)))
        matcher_samples = _arc_distance_points(
            root[:2], positions, facings, root_facing
        )
        scene_samples = self.alignment.matcher_to_scene_xy(matcher_samples)
        return (
            self.query_grid.sample_xy(scene_samples)
            - self.query_grid.sample_xy(root_scene)
        ).to(dtype=torch.float32)


@dataclass(frozen=True)
class TerrainFootClearanceValidator:
    """Validate ankle clearance over a composed inertialized body window."""

    extension: TerrainFeatureExtension
    preview_steps: int
    minimum_clearance_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.extension, TerrainFeatureExtension):
            raise ContractError(
                "terrain clearance validator requires a terrain extension"
            )
        if (
            type(self.preview_steps) is not int
            or not 1 <= self.preview_steps <= 46
        ):
            raise ContractError(
                "terrain clearance preview steps must be an integer in [1, 46]"
            )
        clearance = self.minimum_clearance_m
        if (
            not isinstance(clearance, (int, float))
            or isinstance(clearance, bool)
            or not math.isfinite(float(clearance))
        ):
            raise ContractError(
                "minimum terrain foot clearance must be finite"
            )
        object.__setattr__(
            self, "minimum_clearance_m", float(clearance)
        )

    def __call__(self, body_window: torch.Tensor) -> bool:
        if (
            not isinstance(body_window, torch.Tensor)
            or tuple(body_window.shape) != (46, 3, 3)
            or body_window.dtype != torch.float32
            or body_window.device != self.extension.dataset.device
            or not torch.isfinite(body_window).all()
        ):
            raise ContractError(
                "terrain clearance body window must be finite float32 "
                "shape (46, 3, 3) on the terrain dataset device"
            )
        preview = body_window[: self.preview_steps]
        scene_xy = self.extension.alignment.matcher_to_scene_xy(
            preview[:, :, :2]
        )
        surface = self.extension.query_grid.sample_xy(scene_xy)
        clearance = preview[:, 1:, 2] - surface[:, 1:]
        return bool(
            (clearance.min() >= self.minimum_clearance_m).item()
        )
