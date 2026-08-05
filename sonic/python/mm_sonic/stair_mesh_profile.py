"""Measure the exact stair half traversed by a clean GRAIL clip."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .joints import ContractError
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


@dataclass(frozen=True)
class UniformStairMeshProfile:
    """Physical tread profile measured from a source USD, not its metadata.

    The historical name is retained for API compatibility.  ``tread_edges_m``
    and ``tread_heights_m`` preserve non-uniform steps; when omitted they are
    generated from the legacy scalar rise/tread fields.
    """

    rise_m: float
    tread_m: float
    level_count: int
    tread_edges_m: tuple[float, ...] = ()
    tread_heights_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.rise_m))
            or not math.isfinite(float(self.tread_m))
            or self.rise_m <= 0.0
            or self.tread_m <= 0.0
            or type(self.level_count) is not int
            or self.level_count <= 1
        ):
            raise ContractError("uniform stair mesh profile is invalid")
        if not self.tread_edges_m and not self.tread_heights_m:
            edges = tuple(
                float(value)
                for value in np.linspace(
                    0.0,
                    self.tread_m * self.level_count,
                    self.level_count + 1,
                )
            )
            heights = tuple(
                float(self.rise_m * (level + 1))
                for level in range(self.level_count)
            )
            object.__setattr__(self, "tread_edges_m", edges)
            object.__setattr__(self, "tread_heights_m", heights)
        edges = np.asarray(self.tread_edges_m, dtype=np.float64)
        heights = np.asarray(self.tread_heights_m, dtype=np.float64)
        if (
            edges.shape != (self.level_count + 1,)
            or heights.shape != (self.level_count,)
            or not np.isfinite(edges).all()
            or not np.isfinite(heights).all()
            or abs(float(edges[0])) > 1.0e-6
            or np.any(np.diff(edges) <= 0.0)
            or np.any(heights <= 0.0)
        ):
            raise ContractError("exact stair tread profile is invalid")
        object.__setattr__(
            self,
            "tread_edges_m",
            tuple(float(value) for value in edges),
        )
        object.__setattr__(
            self,
            "tread_heights_m",
            tuple(float(value) for value in heights),
        )

    @property
    def run_m(self) -> float:
        return float(self.tread_edges_m[-1])

    @property
    def height_m(self) -> float:
        return float(max(self.tread_heights_m))


def measure_archive_stair_profile(
    archive: object,
    clip_index: int,
    *,
    sample_spacing_m: float = 0.005,
) -> UniformStairMeshProfile:
    """Raycast the physical stair half traversed by one archive clip."""

    traversal = str(archive["clip_traversal"][clip_index])
    if traversal not in {"up", "down"}:
        raise ContractError("archive clip is not a stair traversal")
    origin = np.asarray(archive["terrain_position_env"][clip_index], dtype=np.float64)
    rotation = np.asarray(
        archive["terrain_rotation_env_wxyz"][clip_index], dtype=np.float64
    )
    yaw = float(archive["travel_yaw_rad"][clip_index])
    mesh = _load_usd_mesh(
        Path(str(archive["terrain_usd_path"][clip_index])),
        source_asset_sha256="0" * 64,
    )
    index = TerrainMeshIndex(mesh, RigidTransform(origin, rotation))
    travel = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
    relative_vertices = np.asarray(index.vertices_world, dtype=np.float64) - origin
    maximum_x = float(np.max(relative_vertices[:, :2] @ travel))
    spacing = float(sample_spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0 or maximum_x <= spacing:
        raise ContractError("stair profile sampling interval is invalid")
    sample_x = np.arange(spacing * 0.5, maximum_x, spacing, dtype=np.float64)
    ray_height = float(np.max(index.vertices_world[:, 2]) + 1.0)
    height = np.full(len(sample_x), np.nan, dtype=np.float64)
    for sample, distance in enumerate(sample_x):
        xy = origin[:2] + distance * travel
        hit = index.raycast(
            (float(xy[0]), float(xy[1]), ray_height),
            (0.0, 0.0, -1.0),
        )
        if hit is not None:
            height[sample] = float(hit.position_world[2] - origin[2])
    delta = np.diff(height)
    sign = 1.0 if traversal == "up" else -1.0
    transitions = np.flatnonzero(sign * delta > 0.04)
    if len(transitions) < 1:
        raise ContractError("could not measure stair risers from the source mesh")
    transition_edges = sample_x[transitions + 1]
    opposite = np.flatnonzero(
        (np.arange(len(delta)) > int(transitions[-1]))
        & (sign * delta < -0.04)
    )
    if traversal == "up" and len(opposite):
        # Up clips stop on the centre of the top plateau of a two-sided stair
        # asset.  Split that plateau halfway between the final ascent and the
        # first descent rather than treating the far-side staircase as run.
        run_m = 0.5 * (
            float(transition_edges[-1])
            + float(sample_x[int(opposite[0]) + 1])
        )
    else:
        finite = np.flatnonzero(np.isfinite(height))
        if not len(finite):
            raise ContractError("stair raycast contains no finite surface")
        run_m = min(
            maximum_x,
            float(sample_x[int(finite[-1])] + 0.5 * spacing),
        )
    internal = transition_edges[transition_edges < run_m - spacing]
    edges = np.concatenate(
        (np.asarray((0.0,), dtype=np.float64), internal, (run_m,))
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    segment_height = np.empty(len(centers), dtype=np.float64)
    for segment, distance in enumerate(centers):
        xy = origin[:2] + distance * travel
        hit = index.raycast(
            (float(xy[0]), float(xy[1]), ray_height),
            (0.0, 0.0, -1.0),
        )
        if hit is None:
            raise ContractError("exact stair segment has no top surface")
        segment_height[segment] = float(hit.position_world[2] - origin[2])
    level_count = len(segment_height)
    if traversal == "up":
        rises = np.diff(np.concatenate(((0.0,), segment_height)))
    else:
        rises = -np.diff(np.concatenate((segment_height, (0.0,))))
    positive_rises = rises[rises > 0.04]
    if not len(positive_rises):
        raise ContractError("exact stair profile has no positive riser")
    rise = float(np.median(positive_rises))
    tread = float(np.median(np.diff(edges)))
    return UniformStairMeshProfile(
        rise,
        tread,
        level_count,
        tuple(float(value) for value in edges),
        tuple(float(value) for value in segment_height),
    )


def paired_uniform_stair_profile(
    archive: object,
    ascent_clip_index: int,
    descent_clip_index: int,
) -> UniformStairMeshProfile:
    """Measure and reconcile a physically compatible up/down archive pair."""

    ascent = measure_archive_stair_profile(archive, ascent_clip_index)
    descent = measure_archive_stair_profile(archive, descent_clip_index)
    descent_edges = descent.run_m - np.asarray(
        descent.tread_edges_m[::-1], dtype=np.float64
    )
    descent_heights = np.asarray(
        descent.tread_heights_m[::-1], dtype=np.float64
    )
    ascent_edges = np.asarray(ascent.tread_edges_m, dtype=np.float64)
    ascent_heights = np.asarray(ascent.tread_heights_m, dtype=np.float64)
    if (
        ascent.level_count != descent.level_count
        or ascent_edges.shape != descent_edges.shape
        or ascent_heights.shape != descent_heights.shape
        or float(np.max(np.abs(ascent_edges - descent_edges))) > 0.02
        or float(np.max(np.abs(ascent_heights - descent_heights))) > 0.01
    ):
        raise ContractError(
            "ascent and descent source meshes do not describe the same staircase"
        )
    edges = 0.5 * (ascent_edges + descent_edges)
    heights = 0.5 * (ascent_heights + descent_heights)
    return UniformStairMeshProfile(
        0.5 * (ascent.rise_m + descent.rise_m),
        0.5 * (ascent.tread_m + descent.tread_m),
        ascent.level_count,
        tuple(float(value) for value in edges),
        tuple(float(value) for value in heights),
    )


__all__ = [
    "UniformStairMeshProfile",
    "measure_archive_stair_profile",
    "paired_uniform_stair_profile",
]
