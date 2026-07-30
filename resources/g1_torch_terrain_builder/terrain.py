"""Authoritative source-terrain adapters for expanded G1 motion clips."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from resources.g1_torch_stair_builder.conversion import NativeMotionArrays
from resources.g1_torch_stair_builder.corpus import load_pinned_sources
from resources.g1_torch_stair_builder.surface import (
    ZUpHeightGrid,
    build_source_height_grid,
)

from .registry import ResolvedSource


_CELL_M = 0.02
_MARGIN_M = 2.5
_DEFAULT_CHAIR_CONFIG = Path(
    "/home/ubuntu/projects/rmr_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/chair_step/chair_step_env_cfg.py"
)
_DEFAULT_CHAIR_CONFIG_SHA256 = (
    "c6187a72e9ab3126faa33b516d4f9f99e78539e764105236fed669007547da74"
)


@dataclass(frozen=True)
class TerrainEvidence:
    adapter: str
    grid: ZUpHeightGrid
    geometry_sha256: Mapping[str, str]
    motion_to_terrain_xy_yaw: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("terrain adapter identity must be non-empty")
        if not isinstance(self.grid, ZUpHeightGrid):
            raise TypeError("terrain evidence grid must be a ZUpHeightGrid")
        transform = self.motion_to_terrain_xy_yaw
        if (
            type(transform) is not tuple
            or len(transform) != 3
            or not all(math.isfinite(float(value)) for value in transform)
        ):
            raise ValueError("motion-to-terrain transform must be finite")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _boxes_grid(
    boxes: tuple[tuple[float, float, float, float, float], ...],
    motion: NativeMotionArrays,
) -> ZUpHeightGrid:
    if not isinstance(motion, NativeMotionArrays):
        raise TypeError("motion must be NativeMotionArrays")
    motion.validate()
    if not boxes:
        raise ValueError("terrain must contain at least one box")
    bounds = np.asarray(boxes, np.float64)
    if (
        bounds.ndim != 2
        or bounds.shape[1:] != (5,)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] <= bounds[:, 0])
        or np.any(bounds[:, 3] <= bounds[:, 2])
        or np.any(bounds[:, 4] <= 0.0)
    ):
        raise ValueError("terrain box bounds are invalid")
    root_xy = np.asarray(motion.body_position_world[:, 0, :2], np.float64)
    minimum = np.minimum(
        root_xy.min(axis=0), bounds[:, (0, 2)].min(axis=0)
    ) - _MARGIN_M
    maximum = np.maximum(
        root_xy.max(axis=0), bounds[:, (1, 3)].max(axis=0)
    ) + _MARGIN_M
    origin = np.floor(minimum / _CELL_M) * _CELL_M
    count = np.ceil((maximum - origin) / _CELL_M).astype(np.int64) + 1
    xs = origin[0] + np.arange(count[0], dtype=np.float64) * _CELL_M
    ys = origin[1] + np.arange(count[1], dtype=np.float64) * _CELL_M
    height = np.zeros((count[1], count[0]), np.float32)
    for minimum_x, maximum_x, minimum_y, maximum_y, top_z in boxes:
        x = (xs >= minimum_x - 1e-9) & (xs <= maximum_x + 1e-9)
        y = (ys >= minimum_y - 1e-9) & (ys <= maximum_y + 1e-9)
        height[np.ix_(y, x)] = np.maximum(
            height[np.ix_(y, x)], np.float32(top_z)
        )
    return ZUpHeightGrid(
        np.asarray(origin, np.float32), _CELL_M, height
    )


def _fixed_staircase(motion: NativeMotionArrays) -> ZUpHeightGrid:
    return _boxes_grid(
        (
            (-0.056, 0.274, -0.192, 0.430, 0.1778),
            (-0.376, -0.046, -0.192, 0.430, 0.3556),
            (-0.705, -0.375, -0.192, 0.430, 0.5334),
        ),
        motion,
    )


def _karen_boxes(path: Path) -> tuple[
    tuple[float, float, float, float, float], ...
]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("Karen staircase metadata cannot be loaded") from error
    if (
        not isinstance(value, dict)
        or value.get("axis_conversion")
        != "fbx_y_up_to_z_up_with_negated_y"
        or not isinstance(value.get("stairs"), list)
        or not value["stairs"]
    ):
        raise ValueError("Karen staircase metadata contract is invalid")
    boxes = []
    names = set()
    for step in value["stairs"]:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str):
            raise ValueError("Karen staircase step identity is invalid")
        if step["name"] in names:
            raise ValueError("Karen staircase step names must be unique")
        names.add(step["name"])
        minimum = np.asarray(step.get("bounds_min_m"), np.float64)
        maximum = np.asarray(step.get("bounds_max_m"), np.float64)
        if (
            minimum.shape != (3,)
            or maximum.shape != (3,)
            or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all()
            or np.any(maximum <= minimum)
            or abs(float(minimum[2])) > 1e-6
        ):
            raise ValueError("Karen staircase bounds are invalid")
        boxes.append(
            (
                float(minimum[0]),
                float(maximum[0]),
                float(minimum[1]),
                float(maximum[1]),
                float(maximum[2]),
            )
        )
    return tuple(boxes)


def _literal_assignment(path: Path, name: str) -> np.ndarray:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("chair configuration cannot be parsed") from error
    values = []
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            try:
                values.append(ast.literal_eval(node.value))
            except Exception as error:
                raise ValueError(
                    f"chair configuration {name} must be literal"
                ) from error
    if len(values) != 1:
        raise ValueError(
            f"chair configuration must define {name} exactly once"
        )
    array = np.asarray(values[0], np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"chair configuration {name} must be finite shape (3,)")
    return array


def _chair_box(
    path: Path,
    *,
    expected_sha256: str | None,
) -> tuple[tuple[float, float, float, float, float], ...]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("chair configuration must be a real file")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("chair configuration SHA-256 changed")
    position = _literal_assignment(path, "BOX_POSITION")
    size = _literal_assignment(path, "BOX_SIZE")
    if np.any(size <= 0.0):
        raise ValueError("chair configuration BOX_SIZE must be positive")
    minimum = position - 0.5 * size
    maximum = position + 0.5 * size
    return (
        (
            float(minimum[0]),
            float(maximum[0]),
            float(minimum[1]),
            float(maximum[1]),
            float(maximum[2]),
        ),
    )


def _geometry_hashes(source: ResolvedSource) -> Mapping[str, str]:
    return MappingProxyType(
        {
            key: value
            for key, value in source.source_sha256.items()
            if key.startswith("geometry:")
        }
    )


def build_source_terrain(
    source: ResolvedSource,
    motion: NativeMotionArrays,
    *,
    chair_config_path: str | Path = _DEFAULT_CHAIR_CONFIG,
    chair_config_sha256: str | None = _DEFAULT_CHAIR_CONFIG_SHA256,
) -> TerrainEvidence:
    if not isinstance(source, ResolvedSource):
        raise TypeError("source must be a ResolvedSource")
    adapter = source.spec.terrain_adapter
    hashes = dict(_geometry_hashes(source))
    if adapter == "fixed-staircase":
        grid = _fixed_staircase(motion)
    elif adapter == "karen-metadata":
        if len(source.geometry_paths) != 1:
            raise ValueError("Karen terrain requires one metadata file")
        grid = _boxes_grid(_karen_boxes(source.geometry_paths[0]), motion)
    elif adapter == "chair-object":
        config = Path(chair_config_path).resolve()
        boxes = _chair_box(config, expected_sha256=chair_config_sha256)
        grid = _boxes_grid(boxes, motion)
        hashes[f"chair-config:{config}"] = _sha256(config)
    elif adapter == "grail-usd":
        root = source.motion_path.parents[1]
        candidates = load_pinned_sources(root)
        pinned = next(
            (
                candidate
                for candidate in candidates
                if candidate.robot_path == source.motion_path
            ),
            None,
        )
        if pinned is None:
            raise ValueError("registered GRAIL source is not pinned")
        grid = build_source_height_grid(pinned)
    else:
        raise ValueError(f"unsupported terrain adapter: {adapter}")
    return TerrainEvidence(
        adapter=adapter,
        grid=grid,
        geometry_sha256=MappingProxyType(hashes),
        motion_to_terrain_xy_yaw=(0.0, 0.0, 0.0),
    )
