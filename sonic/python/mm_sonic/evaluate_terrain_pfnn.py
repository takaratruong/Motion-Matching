"""Deterministic one-step evaluator for sealed terrain-PFNN checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Sequence

import numpy as np
import torch

from mm_sonic.terrain_pfnn.dataset import PFNNShardDataset
from mm_sonic.terrain_pfnn.kinematics import TorchG1ForwardKinematics
from mm_sonic.terrain_pfnn.layout import TRAJECTORY_TIMES_S
from mm_sonic.terrain_pfnn.runtime import (
    ClosedLoopRecorder,
    NativeG1RuntimeGeometry,
    TerrainPFNNRuntime,
    TerrainSample,
)
from mm_sonic.terrain_pfnn.training import (
    evaluate_fitted_transition_envelope,
    fitted_adjacent_indices,
    fitted_row_sha256,
    load_checkpoint,
    one_step_metrics,
    validate_fitted_transition_report,
)
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh
from mm_sonic.terrain_oracle.math3d import RigidTransform


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_TRAIN_TICKS = 600
_ROUTE_TURNAROUND_CLEARANCE_M = 0.85
_USD_DECODE_SCHEMA = "mm-sonic-usd-mesh/v1"
_GRAIL_PAIR_SCHEMA = "mm-sonic-grail-pair/v1"
_USD_DECODE_MAXIMUM_BYTES = 64 * 1024 * 1024
_DEFAULT_USD_PYTHON = Path("/home/ubuntu/miniconda3/bin/python")
_DEFAULT_MODEL_PATH = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/assets/skeletons/g1/g1_29dof.xml"
)


def select_known_train_grail_record(
    manifest: dict[str, object], *, split: str
) -> tuple[str, dict[str, object]]:
    """Select the lexicographically first paired GRAIL record in ``split``."""

    if split not in ("train", "validation", "test"):
        raise ValueError("known terrain split is invalid")
    records = manifest.get("source_records")
    if type(records) is not dict:
        raise ValueError("dataset source records are invalid")
    candidates = [
        (str(name), record)
        for name, record in records.items()
        if type(name) is str
        and type(record) is dict
        and record.get("source_kind") == "grail"
        and record.get("split") == split
    ]
    if not candidates:
        raise ValueError(f"dataset has no paired GRAIL record in {split}")
    name, record = min(candidates, key=lambda item: item[0])
    return name, dict(record)


def known_train_command(
    tick: int, warm_command: object
) -> tuple[np.ndarray, str | None]:
    """Exact 600-tick known-slope command script approved for Task 7."""

    if type(tick) is not int or not 0 <= tick < _KNOWN_TRAIN_TICKS:
        raise ValueError("known-train script tick must be in [0,600)")
    warm = np.asarray(warm_command, dtype=np.float64)
    if warm.shape != (2,) or not np.isfinite(warm).all():
        raise ValueError("known-train warm command must be finite with shape (2,)")
    if tick < 60:
        return warm.copy(), "forward"
    if 60 <= tick < 360:
        return np.asarray((0.35, 0.0), dtype=np.float64), "forward"
    if 420 <= tick < 540:
        # Facing remains a recurrent trajectory state; only travel reverses.
        return np.asarray((-0.35, 0.0), dtype=np.float64), "backward"
    return np.zeros(2, dtype=np.float64), None


def _load_decoded_usd_npz(
    path: str | Path, *, expected_sha256: str
) -> CanonicalTerrainMesh:
    artifact = Path(path)
    if (
        not artifact.is_file()
        or artifact.stat().st_size <= 0
        or artifact.stat().st_size > _USD_DECODE_MAXIMUM_BYTES
    ):
        raise ValueError("USD helper artifact size is invalid")
    try:
        with np.load(artifact, allow_pickle=False) as payload:
            if set(payload.files) != {
                "schema", "source_sha256", "vertices", "faces", "valid_faces"
            }:
                raise ValueError("USD helper artifact fields are invalid")
            schema = np.asarray(payload["schema"])
            source_hash = np.asarray(payload["source_sha256"])
            vertices = np.asarray(payload["vertices"])
            faces = np.asarray(payload["faces"])
            valid_faces = np.asarray(payload["valid_faces"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("USD helper"):
            raise
        raise ValueError("USD helper artifact cannot be loaded safely") from error
    if schema.shape != () or schema.item() != _USD_DECODE_SCHEMA:
        raise ValueError("USD helper artifact schema is invalid")
    if source_hash.shape != () or source_hash.item() != expected_sha256:
        raise ValueError("USD helper artifact source hash mismatch")
    if (
        vertices.dtype != np.dtype(np.float32)
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
        or faces.dtype != np.dtype(np.int32)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(faces) < 1
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or valid_faces.dtype != np.dtype(np.bool_)
        or valid_faces.shape != (len(faces),)
        or not np.any(valid_faces)
    ):
        raise ValueError("USD helper numeric arrays are invalid")
    return CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=faces,
        valid_faces=valid_faces,
        source_asset_sha256=expected_sha256,
    )


def _decode_usd_mesh(
    usd_path: str | Path,
    *,
    expected_sha256: str,
    python_path: str | Path = _DEFAULT_USD_PYTHON,
) -> CanonicalTerrainMesh:
    """Decode a caller-selected USD through the existing reviewed PXR reader."""

    source = Path(usd_path).expanduser().resolve()
    interpreter = Path(python_path).expanduser().resolve()
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise ValueError("USD source hash mismatch before helper launch")
    if not interpreter.is_file():
        raise ValueError("USD helper interpreter is not a file")
    module_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(module_root)
        if not existing_pythonpath
        else os.pathsep.join((str(module_root), existing_pythonpath))
    )
    with tempfile.TemporaryDirectory(prefix="terrain-pfnn-usd-") as temporary:
        output = Path(temporary) / "mesh.npz"
        try:
            completed = subprocess.run(
                (
                    str(interpreter),
                    "-m",
                    "mm_sonic.terrain_pfnn.usd_decode",
                    "--usd",
                    str(source),
                    "--expected-sha256",
                    expected_sha256,
                    "--output",
                    str(output),
                ),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError("USD helper process failed") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise ValueError(f"USD helper process failed: {detail}")
        if _sha256(source) != expected_sha256:
            raise ValueError("USD source hash changed during helper execution")
        return _load_decoded_usd_npz(output, expected_sha256=expected_sha256)


@dataclass(frozen=True)
class _DecodedGrailPair:
    mesh: CanonicalTerrainMesh
    robot_sha256: str
    anchor_root_xy: np.ndarray
    anchor_yaw: float
    anchor_support: float
    anchor_center_frame: int


def _load_decoded_grail_npz(
    path: str | Path,
    *,
    expected_terrain_sha256: str,
    expected_robot_sha256: str,
    expected_center_frame: int,
) -> _DecodedGrailPair:
    artifact = Path(path)
    if (
        not artifact.is_file()
        or artifact.stat().st_size <= 0
        or artifact.stat().st_size > _USD_DECODE_MAXIMUM_BYTES
    ):
        raise ValueError("GRAIL helper artifact size is invalid")
    expected_fields = {
        "schema", "source_sha256", "robot_sha256", "vertices", "faces",
        "valid_faces", "anchor_root_xy", "anchor_yaw", "anchor_support",
        "anchor_center_frame",
    }
    try:
        with np.load(artifact, allow_pickle=False) as payload:
            if set(payload.files) != expected_fields:
                raise ValueError("GRAIL helper artifact fields are invalid")
            values = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("GRAIL helper"):
            raise
        raise ValueError("GRAIL helper artifact cannot be loaded safely") from error
    if (
        values["schema"].shape != ()
        or values["schema"].item() != _GRAIL_PAIR_SCHEMA
        or values["source_sha256"].shape != ()
        or values["source_sha256"].item() != expected_terrain_sha256
        or values["robot_sha256"].shape != ()
        or values["robot_sha256"].item() != expected_robot_sha256
        or values["anchor_center_frame"].shape != ()
        or values["anchor_center_frame"].dtype != np.dtype(np.int32)
        or int(values["anchor_center_frame"]) != expected_center_frame
    ):
        raise ValueError("GRAIL helper artifact provenance mismatch")
    vertices = values["vertices"]
    faces = values["faces"]
    valid_faces = values["valid_faces"]
    root_xy = values["anchor_root_xy"]
    yaw = values["anchor_yaw"]
    support = values["anchor_support"]
    if (
        vertices.dtype != np.dtype(np.float32)
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
        or faces.dtype != np.dtype(np.int32)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(faces) < 1
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or valid_faces.dtype != np.dtype(np.bool_)
        or valid_faces.shape != (len(faces),)
        or not np.any(valid_faces)
        or root_xy.dtype != np.dtype(np.float32)
        or root_xy.shape != (2,)
        or not np.isfinite(root_xy).all()
        or yaw.dtype != np.dtype(np.float32)
        or yaw.shape != ()
        or not np.isfinite(yaw)
        or support.dtype != np.dtype(np.float32)
        or support.shape != ()
        or not np.isfinite(support)
    ):
        raise ValueError("GRAIL helper numeric arrays are invalid")
    mesh = CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=faces,
        valid_faces=valid_faces,
        source_asset_sha256=expected_terrain_sha256,
    )
    return _DecodedGrailPair(
        mesh=mesh,
        robot_sha256=expected_robot_sha256,
        anchor_root_xy=np.asarray(root_xy, dtype=np.float64),
        anchor_yaw=float(yaw),
        anchor_support=float(support),
        anchor_center_frame=expected_center_frame,
    )


def _decode_grail_pair(
    usd_path: str | Path,
    robot_path: str | Path,
    *,
    expected_terrain_sha256: str,
    expected_robot_sha256: str,
    center_frame: int,
    python_path: str | Path = _DEFAULT_USD_PYTHON,
) -> _DecodedGrailPair:
    """Decode one caller-bound GRAIL robot/USD pair in the PXR/joblib env."""

    terrain = Path(usd_path).expanduser().resolve()
    robot = Path(robot_path).expanduser().resolve()
    interpreter = Path(python_path).expanduser().resolve()
    if (
        not terrain.is_file()
        or _sha256(terrain) != expected_terrain_sha256
        or not robot.is_file()
        or _sha256(robot) != expected_robot_sha256
    ):
        raise ValueError("GRAIL pair hash mismatch before helper launch")
    if not interpreter.is_file():
        raise ValueError("GRAIL helper interpreter is not a file")
    module_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(module_root)
        if not existing_pythonpath
        else os.pathsep.join((str(module_root), existing_pythonpath))
    )
    with tempfile.TemporaryDirectory(prefix="terrain-pfnn-grail-") as temporary:
        output = Path(temporary) / "pair.npz"
        try:
            completed = subprocess.run(
                (
                    str(interpreter), "-m", "mm_sonic.terrain_pfnn.usd_decode",
                    "--usd", str(terrain),
                    "--expected-sha256", expected_terrain_sha256,
                    "--robot", str(robot),
                    "--expected-robot-sha256", expected_robot_sha256,
                    "--center-frame", str(center_frame),
                    "--output", str(output),
                ),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError("GRAIL helper process failed") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise ValueError(f"GRAIL helper process failed: {detail}")
        if (
            _sha256(terrain) != expected_terrain_sha256
            or _sha256(robot) != expected_robot_sha256
        ):
            raise ValueError("GRAIL pair hash changed during helper execution")
        return _load_decoded_grail_npz(
            output,
            expected_terrain_sha256=expected_terrain_sha256,
            expected_robot_sha256=expected_robot_sha256,
            expected_center_frame=center_frame,
        )


class KnownSlopeTerrain:
    """Exact paired top-surface query with one bounded synthetic flat apron."""

    def __init__(self, triangles_world: np.ndarray, *, source_sha256: str) -> None:
        triangles = np.ascontiguousarray(triangles_world, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
            raise ValueError("known-slope triangles must have shape [N,3,3]")
        raw_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(raw_normals, axis=1)
        usable = (lengths > 1.0e-12) & (raw_normals[:, 2] > 1.0e-12)
        if not np.any(usable):
            raise ValueError("known-slope mesh has no upward top surface")
        triangles = triangles[usable]
        normals = raw_normals[usable] / lengths[usable, None]
        gradients = -normals[:, :2] / normals[:, 2:3]
        grades = np.degrees(np.arctan(np.linalg.norm(gradients, axis=1)))
        flank = (grades >= 2.0) & (grades <= 20.0 + 1.0e-8)
        if not np.any(flank):
            raise ValueError("known-slope mesh has no supported measured flank")
        flank_triangles = triangles[flank]
        flank_gradients = gradients[flank]
        minimum_height = float(np.min(flank_triangles[:, :, 2]))
        boundary_mask = np.isclose(
            flank_triangles[:, :, 2], minimum_height, atol=1.0e-7, rtol=0.0
        )
        boundary_points = flank_triangles[boundary_mask]
        if len(boundary_points) < 2:
            raise ValueError("known-slope lower boundary cannot be identified")
        boundary_xy = np.mean(boundary_points[:, :2], axis=0)
        adjacent = np.any(boundary_mask, axis=1)
        uphill = np.mean(flank_gradients[adjacent], axis=0)
        uphill_norm = float(np.linalg.norm(uphill))
        if uphill_norm < 1.0e-12:
            raise ValueError("known-slope uphill axis is ambiguous")
        uphill /= uphill_norm
        lateral = np.asarray((-uphill[1], uphill[0]), dtype=np.float64)
        self._triangles = triangles
        self._gradients = gradients
        self._boundary_xy = boundary_xy
        self._boundary_height = minimum_height
        self._uphill = uphill
        self._lateral = lateral
        self.alignment_report = {
            "approach": "synthetic_flat_apron",
            "apron_start_runtime_x_m": -0.25,
            "mesh_boundary_runtime_x_m": 0.50,
            "runtime_origin_outside_boundary_m": 0.50,
            "source_boundary_xy": boundary_xy.tolist(),
            "source_boundary_height_m": minimum_height,
            "source_uphill_axis_xy": uphill.tolist(),
            "source_sha256": source_sha256,
            "maximum_supported_mesh_grade_degrees": float(np.max(grades[flank])),
            "reaches_exact_mesh_flank": True,
        }
        self.apron_query_count = 0
        self.exact_mesh_query_count = 0

    @classmethod
    def from_mesh(
        cls,
        mesh: CanonicalTerrainMesh,
        *,
        world_from_mesh: RigidTransform | None = None,
    ) -> "KnownSlopeTerrain":
        if not isinstance(mesh, CanonicalTerrainMesh):
            raise TypeError("KnownSlopeTerrain requires CanonicalTerrainMesh")
        transform = (
            RigidTransform(np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0)))
            if world_from_mesh is None
            else world_from_mesh
        )
        vertices = np.asarray(transform.apply_points(mesh.vertices_local), dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        valid = np.asarray(mesh.valid_faces, dtype=np.bool_)
        return cls(
            vertices[faces[valid]],
            source_sha256=mesh.source_asset_sha256,
        )

    @staticmethod
    def _inside_height_gradient(
        point: np.ndarray, triangle: np.ndarray, gradient: np.ndarray
    ) -> tuple[float, np.ndarray] | None:
        a, b, c = triangle[:, :2]
        edge0, edge1 = b - a, c - a
        denominator = (
            float(np.dot(edge0, edge0)) * float(np.dot(edge1, edge1))
            - float(np.dot(edge0, edge1)) ** 2
        )
        if abs(denominator) <= 1.0e-14:
            return None
        relative = point - a
        dot00 = float(np.dot(edge0, edge0))
        dot01 = float(np.dot(edge0, edge1))
        dot11 = float(np.dot(edge1, edge1))
        dot20 = float(np.dot(relative, edge0))
        dot21 = float(np.dot(relative, edge1))
        weight_b = (dot11 * dot20 - dot01 * dot21) / denominator
        weight_c = (dot00 * dot21 - dot01 * dot20) / denominator
        weight_a = 1.0 - weight_b - weight_c
        if min(weight_a, weight_b, weight_c) < -1.0e-10:
            return None
        height = (
            weight_a * triangle[0, 2]
            + weight_b * triangle[1, 2]
            + weight_c * triangle[2, 2]
        )
        return float(height), gradient

    def __call__(self, xy: np.ndarray) -> TerrainSample | None:
        runtime_xy = np.asarray(xy, dtype=np.float64)
        if runtime_xy.shape != (2,) or not np.isfinite(runtime_xy).all():
            return None
        along = float(runtime_xy[0] - 0.50)
        if -0.75 - 1.0e-10 <= along < 0.0:
            self.apron_query_count += 1
            return TerrainSample(0.0, np.zeros(2), "flat")
        source_xy = (
            self._boundary_xy
            + along * self._uphill
            + float(runtime_xy[1]) * self._lateral
        )
        candidates = [
            self._inside_height_gradient(source_xy, triangle, gradient)
            for triangle, gradient in zip(self._triangles, self._gradients, strict=True)
        ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            return None
        self.exact_mesh_query_count += 1
        source_height, source_gradient = max(candidates, key=lambda item: item[0])
        runtime_gradient = np.asarray(
            (
                float(np.dot(source_gradient, self._uphill)),
                float(np.dot(source_gradient, self._lateral)),
            ),
            dtype=np.float64,
        )
        return TerrainSample(
            source_height - self._boundary_height,
            runtime_gradient,
            None,
        )


class SourceAlignedTerrain:
    """Dataset-identical terrain heights in the seed root's source frame."""

    def __init__(
        self,
        mesh: CanonicalTerrainMesh,
        *,
        world_from_mesh: RigidTransform,
        source_root_xy: object,
        source_root_yaw: float,
        source_support_height: float,
        runtime_root_xy: object = (0.0, 0.0),
    ) -> None:
        vertices = np.asarray(
            world_from_mesh.apply_points(mesh.vertices_local), dtype=np.float64
        )
        faces = np.asarray(mesh.faces, dtype=np.int64)
        valid = np.asarray(mesh.valid_faces, dtype=np.bool_)
        triangles = vertices[faces[valid]]
        raw_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        usable = raw_normals[:, 2] > 1.0e-8
        if not np.any(usable):
            raise ValueError("source-aligned terrain has no upward surface")
        self._triangles = triangles[usable]
        normals = raw_normals[usable]
        self._gradients = -normals[:, :2] / normals[:, 2:3]
        self._source_root_xy = np.asarray(source_root_xy, dtype=np.float64)
        self._runtime_root_xy = np.asarray(runtime_root_xy, dtype=np.float64)
        self._source_root_yaw = float(source_root_yaw)
        self._source_support = float(source_support_height)
        if (
            self._source_root_xy.shape != (2,)
            or not np.isfinite(self._source_root_xy).all()
            or self._runtime_root_xy.shape != (2,)
            or not np.isfinite(self._runtime_root_xy).all()
            or not math.isfinite(self._source_root_yaw)
            or not math.isfinite(self._source_support)
        ):
            raise ValueError("source-aligned terrain anchor is invalid")
        cosine, sine = math.cos(self._source_root_yaw), math.sin(
            self._source_root_yaw
        )
        self._source_from_runtime = np.asarray(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        self.exact_mesh_query_count = 0
        self.extrapolated_query_count = 0
        self._runtime_route_maximum_x: float | None = None
        self.alignment_report = {
            "approach": "seed_source_anchor",
            "source_root_xy": self._source_root_xy.tolist(),
            "runtime_root_xy": self._runtime_root_xy.tolist(),
            "source_root_yaw": self._source_root_yaw,
            "source_support_height_m": self._source_support,
            "source_sha256": mesh.source_asset_sha256,
        }

    @classmethod
    def from_supported_flat_seed(
        cls,
        mesh: CanonicalTerrainMesh,
        *,
        world_from_mesh: RigidTransform,
        runtime_root_xy: object,
        runtime_root_yaw: float,
        seed_trajectory_position: object,
        seed_trajectory_direction: object,
        expected_relative_terrain: object,
    ) -> "SourceAlignedTerrain":
        """Deterministically align a flat recurrent seed on exact mesh support."""

        root_xy = np.asarray(runtime_root_xy, dtype=np.float64)
        positions = np.asarray(seed_trajectory_position, dtype=np.float64)
        directions = np.asarray(seed_trajectory_direction, dtype=np.float64)
        expected = np.asarray(expected_relative_terrain, dtype=np.float64)
        if (
            root_xy.shape != (2,)
            or positions.shape != (12, 2)
            or directions.shape != (12, 2)
            or expected.shape != (12, 3)
            or not all(
                np.isfinite(value).all()
                for value in (root_xy, positions, directions, expected)
            )
            or not math.isfinite(float(runtime_root_yaw))
        ):
            raise ValueError("supported-flat seed alignment inputs are invalid")
        vertices = np.asarray(
            world_from_mesh.apply_points(mesh.vertices_local), dtype=np.float64
        )
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = vertices[faces[np.asarray(mesh.valid_faces, dtype=np.bool_)]]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        usable = normals[:, 2] > 1.0e-8
        triangles, normals = triangles[usable], normals[usable]
        gradients = -normals[:, :2] / normals[:, 2:3]
        grades = np.degrees(np.arctan(np.linalg.norm(gradients, axis=1)))
        nonflat = np.flatnonzero(grades >= 2.0)
        flat = np.flatnonzero(grades <= 0.5)
        if not len(nonflat) or not len(flat):
            raise ValueError("known terrain needs exact flat and sloped support")
        headings: list[float] = []
        for index in nonflat:
            gradient = gradients[index]
            angle = math.atan2(float(gradient[1]), float(gradient[0]))
            for heading in (angle, math.atan2(-math.sin(angle), -math.cos(angle))):
                if all(
                    abs(math.atan2(math.sin(heading - other), math.cos(heading - other)))
                    > 1.0e-6
                    for other in headings
                ):
                    headings.append(heading)
        barycentric = np.asarray(
            [
                (first / 10.0, second / 10.0, (10 - first - second) / 10.0)
                for first in range(11)
                for second in range(11 - first)
            ],
            dtype=np.float64,
        )
        candidates = sorted(
            {
                tuple(np.round(weights @ triangles[index, :, :2], 10))
                for index in flat
                for weights in barycentric
            }
        )
        runtime_rotation = np.asarray(
            (
                (math.cos(runtime_root_yaw), -math.sin(runtime_root_yaw)),
                (math.sin(runtime_root_yaw), math.cos(runtime_root_yaw)),
            ),
            dtype=np.float64,
        )
        span = float(np.linalg.norm(np.ptp(vertices[:, :2], axis=0))) + 0.5
        def assess(
            heading: float, candidate: np.ndarray
        ) -> tuple[tuple[float, ...], SourceAlignedTerrain] | None:
            # The candidate lies on a known flat triangle, so the highest
            # containing upward face defines its exact support height.
            provisional = cls(
                mesh,
                world_from_mesh=world_from_mesh,
                source_root_xy=candidate,
                source_root_yaw=heading,
                source_support_height=0.0,
                runtime_root_xy=root_xy,
            )
            try:
                support, _, _ = provisional._source_height_gradient(candidate)
            except ValueError:
                return None
            terrain = cls(
                mesh,
                world_from_mesh=world_from_mesh,
                source_root_xy=candidate,
                source_root_yaw=heading,
                source_support_height=support,
                runtime_root_xy=root_xy,
            )
            actual = np.empty((12, 3), dtype=np.float64)
            supported = True
            for index in range(12):
                norm = float(np.linalg.norm(directions[index]))
                if norm < 1.0e-12:
                    supported = False
                    break
                facing = directions[index] / norm
                normal = np.asarray((-facing[1], facing[0]))
                for probe, local in enumerate(
                    (
                        positions[index] + 0.25 * normal,
                        positions[index],
                        positions[index] - 0.25 * normal,
                    )
                ):
                    runtime_point = root_xy + local @ runtime_rotation.T
                    sample = terrain(runtime_point)
                    if sample is None or sample.absolute_grade_degrees > 0.5:
                        supported = False
                        break
                    actual[index, probe] = sample.height_m
                if not supported:
                    break
            if not supported:
                return None
            error = float(np.max(np.abs(actual - expected)))
            if error > 1.0e-4:
                return None
            flank_entry: float | None = None
            continuation: float | None = None
            last_supported = 0.0
            route_maximum_grade = 0.0
            for distance in np.linspace(0.0, span, max(2, int(span / 0.025) + 1)):
                runtime_points = [
                    root_xy
                    + np.asarray((distance, lateral)) @ runtime_rotation.T
                    for lateral in (0.25, 0.0, -0.25)
                ]
                samples = [terrain(point) for point in runtime_points]
                if any(sample is None for sample in samples):
                    break
                last_supported = float(distance)
                grade = max(sample.absolute_grade_degrees for sample in samples)
                route_maximum_grade = max(route_maximum_grade, grade)
                if flank_entry is None and grade >= 2.0:
                    flank_entry = float(distance)
                elif (
                    continuation is None
                    and flank_entry is not None
                    and grade <= 0.5
                    and distance > flank_entry + 0.05
                ):
                    continuation = float(distance)
            # Keep a full future-knot horizon inside exact support at the
            # turnaround.  The runtime replanner may stop earlier, never later.
            turnaround = last_supported - _ROUTE_TURNAROUND_CLEARANCE_M
            if (
                flank_entry is None
                or turnaround <= flank_entry + 0.10
                or route_maximum_grade < 2.0
            ):
                return None
            terrain.alignment_report.update(
                {
                    "approach": "supported_flat_mesh",
                    "runtime_to_source_yaw": heading,
                    "maximum_seed_probe_error_m": error,
                    "verified_seed_probe_count": 36,
                    "native_maximum_grade_degrees": float(np.max(grades)),
                    "route_maximum_grade_degrees": route_maximum_grade,
                    "flank_entry_runtime_x_m": flank_entry,
                    "continuation_runtime_x_m": continuation,
                    "has_post_flank_continuation": continuation is not None,
                    "supported_route_extent_m": last_supported,
                    "turnaround_runtime_x_m": turnaround,
                    "turnaround_future_clearance_m": (
                        _ROUTE_TURNAROUND_CLEARANCE_M
                    ),
                }
            )
            score = (
                -route_maximum_grade,
                -turnaround,
                error,
                flank_entry,
                heading,
                float(candidate[0]),
                float(candidate[1]),
            )
            return score, terrain

        best: tuple[tuple[float, ...], SourceAlignedTerrain] | None = None
        for heading in sorted(headings):
            for candidate_tuple in candidates:
                assessed = assess(
                    heading, np.asarray(candidate_tuple, dtype=np.float64)
                )
                if assessed is not None and (best is None or assessed[0] < best[0]):
                    best = assessed
        if best is None:
            raise ValueError(
                "no exact supported flat route has a safe flank turnaround"
            )
        selected = best[1]
        selected._runtime_route_maximum_x = float(
            selected.alignment_report["turnaround_runtime_x_m"]
        )
        selected.exact_mesh_query_count = 0
        selected.extrapolated_query_count = 0
        return selected

    def _source_height_gradient(
        self, point: np.ndarray
    ) -> tuple[float, np.ndarray, bool]:
        best_height = -math.inf
        best_gradient: np.ndarray | None = None
        for triangle, gradient in zip(
            self._triangles, self._gradients, strict=True
        ):
            xy = triangle[:, :2]
            a, b, c = xy
            edge0, edge1 = b - a, c - a
            dot00 = float(np.dot(edge0, edge0))
            dot01 = float(np.dot(edge0, edge1))
            dot11 = float(np.dot(edge1, edge1))
            denominator = dot00 * dot11 - dot01 * dot01
            if abs(denominator) > 1.0e-14:
                relative = point - a
                dot20 = float(np.dot(relative, edge0))
                dot21 = float(np.dot(relative, edge1))
                weight_b = (dot11 * dot20 - dot01 * dot21) / denominator
                weight_c = (dot00 * dot21 - dot01 * dot20) / denominator
                weight_a = 1.0 - weight_b - weight_c
                if min(weight_a, weight_b, weight_c) >= -1.0e-12:
                    height = float(
                        weight_a * triangle[0, 2]
                        + weight_b * triangle[1, 2]
                        + weight_c * triangle[2, 2]
                    )
                    if height > best_height:
                        best_height = height
                        best_gradient = gradient
        if best_gradient is None or not math.isfinite(best_height):
            raise ValueError("source-aligned terrain query failed")
        return best_height, best_gradient, True

    def collision_heights_at(self, xy: object) -> np.ndarray:
        """Vectorized exact heights for every native collision-surface sample."""

        runtime = np.asarray(xy, dtype=np.float64)
        if runtime.ndim != 2 or runtime.shape[1:] != (2,) or not np.isfinite(runtime).all():
            raise ValueError("collision terrain points must be finite [N,2]")
        source = self._source_root_xy + (
            runtime - self._runtime_root_xy
        ) @ self._source_from_runtime.T
        heights = np.full(len(source), -math.inf, dtype=np.float64)
        for triangle in self._triangles:
            a, b, c = triangle[:, :2]
            edge0, edge1 = b - a, c - a
            dot00 = float(np.dot(edge0, edge0))
            dot01 = float(np.dot(edge0, edge1))
            dot11 = float(np.dot(edge1, edge1))
            denominator = dot00 * dot11 - dot01 * dot01
            if abs(denominator) <= 1.0e-14:
                continue
            relative = source - a
            dot20 = relative @ edge0
            dot21 = relative @ edge1
            weight_b = (dot11 * dot20 - dot01 * dot21) / denominator
            weight_c = (dot00 * dot21 - dot01 * dot20) / denominator
            weight_a = 1.0 - weight_b - weight_c
            inside = (
                (weight_a >= -1.0e-12)
                & (weight_b >= -1.0e-12)
                & (weight_c >= -1.0e-12)
            )
            if np.any(inside):
                values = (
                    weight_a[inside] * triangle[0, 2]
                    + weight_b[inside] * triangle[1, 2]
                    + weight_c[inside] * triangle[2, 2]
                )
                heights[inside] = np.maximum(heights[inside], values)
        if np.any(~np.isfinite(heights)):
            raise ValueError("collision terrain query left exact mesh support")
        return heights - self._source_support

    def __call__(self, xy: np.ndarray) -> TerrainSample | None:
        runtime_xy = np.asarray(xy, dtype=np.float64)
        if runtime_xy.shape != (2,) or not np.isfinite(runtime_xy).all():
            return None
        if (
            self._runtime_route_maximum_x is not None
            and runtime_xy[0] - self._runtime_root_xy[0]
            > self._runtime_route_maximum_x + 1.0e-10
        ):
            return None
        source_xy = (
            self._source_root_xy
            + (runtime_xy - self._runtime_root_xy) @ self._source_from_runtime.T
        )
        try:
            height, source_gradient, exact = self._source_height_gradient(source_xy)
        except ValueError:
            return None
        if exact:
            self.exact_mesh_query_count += 1
        else:
            self.extrapolated_query_count += 1
        runtime_gradient = source_gradient @ self._source_from_runtime
        return TerrainSample(
            height - self._source_support,
            np.asarray(runtime_gradient, dtype=np.float64),
            None,
        )


def _known_train_terrain(
    manifest: dict[str, object], *, split: str, checkpoint: object
) -> tuple[str, SourceAlignedTerrain]:
    seed = getattr(checkpoint, "runtime_seed", None)
    provenance = seed.get("provenance") if type(seed) is dict else None
    if type(provenance) is not dict:
        raise ValueError("checkpoint runtime seed provenance is invalid")
    records = manifest.get("source_records")
    fitted_subset = getattr(checkpoint, "fitted_subset", None)
    fitted_rows = fitted_subset.get("rows") if type(fitted_subset) is dict else None
    if type(records) is not dict or type(fitted_rows) is not list:
        raise ValueError("checkpoint fitted source receipt is missing")
    candidates = sorted(
        {
            row["clip_id"]
            for row in fitted_rows
            if type(row) is dict
            and type(row.get("clip_id")) is str
            and type(records.get(row["clip_id"])) is dict
            and records[row["clip_id"]].get("source_kind") == "grail"
            and records[row["clip_id"]].get("split") == split
            and row.get("terrain_class") in ("ascent", "descent", "transition")
        }
    )
    if not candidates:
        raise ValueError("checkpoint fitted GRAIL source record is missing")
    roots = manifest.get("source_roots")
    if type(roots) is not dict or type(roots.get("grail")) is not dict:
        raise ValueError("dataset GRAIL source root is invalid")
    root_value = roots["grail"].get("path")
    if type(root_value) is not str:
        raise ValueError("dataset GRAIL source root is invalid")
    root = Path(root_value).expanduser().resolve()
    expected_terrain_parent = (root / "data" / "slope" / "object_usd").resolve()
    world_from_usd = RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.asarray((math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)), dtype=np.float32),
    )
    runtime_root_xy = np.asarray(seed["world_xy"], dtype=np.float64)
    runtime_root_yaw = float(np.asarray(seed["world_yaw"]))
    positions = np.asarray(seed["trajectory_position"], dtype=np.float64)
    directions = np.asarray(seed["trajectory_direction"], dtype=np.float64)
    expected = np.asarray(seed["terrain_height"], dtype=np.float64)
    route_candidates: list[
        tuple[tuple[float, float, str], str, str, SourceAlignedTerrain]
    ] = []
    for candidate_name in candidates:
        record = dict(records[candidate_name])
        terrain_path = (
            root / "data" / "slope" / "object_usd" / f"{candidate_name}.usd"
        ).resolve()
        terrain_hash = record.get("terrain_sha256")
        if (
            terrain_path.parent != expected_terrain_parent
            or not terrain_path.is_file()
            or type(terrain_hash) is not str
            or _SHA256_RE.fullmatch(terrain_hash) is None
        ):
            raise ValueError("paired GRAIL terrain path/hash is invalid")
        mesh = _decode_usd_mesh(terrain_path, expected_sha256=terrain_hash)
        try:
            route = SourceAlignedTerrain.from_supported_flat_seed(
                mesh,
                world_from_mesh=world_from_usd,
                runtime_root_xy=runtime_root_xy,
                runtime_root_yaw=runtime_root_yaw,
                seed_trajectory_position=positions,
                seed_trajectory_direction=directions,
                expected_relative_terrain=expected,
            )
        except ValueError:
            continue
        route_candidates.append(
            (
                (
                    -float(route.alignment_report["route_maximum_grade_degrees"]),
                    -float(route.alignment_report["turnaround_runtime_x_m"]),
                    candidate_name,
                ),
                candidate_name,
                terrain_hash,
                route,
            )
        )
    if not route_candidates:
        raise ValueError("no fitted GRAIL terrain has an exact supported route")
    _, name, terrain_hash, aligned = min(route_candidates, key=lambda item: item[0])
    rotation = np.asarray(
        (
            (math.cos(runtime_root_yaw), -math.sin(runtime_root_yaw)),
            (math.sin(runtime_root_yaw), math.cos(runtime_root_yaw)),
        ),
        dtype=np.float64,
    )
    actual = np.empty((12, 3), dtype=np.float64)
    for index in range(12):
        normal = np.asarray((-directions[index, 1], directions[index, 0]))
        for probe, point in enumerate((
            positions[index] + 0.25 * normal,
            positions[index],
            positions[index] - 0.25 * normal,
        )):
            sample = aligned(runtime_root_xy + point @ rotation.T)
            if sample is None:
                raise ValueError("seed terrain alignment has a missing probe")
            actual[index, probe] = sample.height_m
    error = float(np.max(np.abs(actual - expected)))
    if error > 1.0e-4:
        raise ValueError(f"seed terrain alignment mismatch: {error:.9g} m")
    aligned.alignment_report.update({
        "source_record": name,
        "terrain_sha256": terrain_hash,
        "seed_first_fitted_clip_id": provenance["first_fitted_clip_id"],
        "seed_first_fitted_center_frame": provenance["first_fitted_center_frame"],
        "maximum_seed_probe_error_m": error,
        "verified_seed_probe_count": 36,
        "fitted_route_candidate_count": len(candidates),
        "valid_route_candidate_count": len(route_candidates),
    })
    return name, aligned


def run_known_train_rollout(
    *,
    checkpoint: object,
    kinematics: TorchG1ForwardKinematics,
    model_path: str | Path,
    manifest: dict[str, object],
    split: str,
    device: str | torch.device,
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Run the provisional 600-tick known-train-slope Task-7 gate."""

    if split != "train":
        raise ValueError("known-slope provisional rollout is train-only")
    source_name, terrain = _known_train_terrain(
        manifest, split=split, checkpoint=checkpoint
    )
    seed_trajectory = np.asarray(
        checkpoint.runtime_seed["trajectory_position"], dtype=np.float64
    )
    warm_command = (
        seed_trajectory[11] - seed_trajectory[6]
    ) / float(TRAJECTORY_TIMES_S[11] - TRAJECTORY_TIMES_S[6])
    runtime = TerrainPFNNRuntime(
        checkpoint=checkpoint,
        kinematics=kinematics,
        height_and_grade_at=terrain,
        device=device,
    )
    geometry = NativeG1RuntimeGeometry.from_mjcf(model_path)
    recorder = ClosedLoopRecorder(
        joint_limits=runtime.joint_limits,
        native_geometry=geometry,
        height_and_grade_at=terrain,
        traversal_axis_world=(1.0, 0.0),
    )
    first_hold: dict[str, object] | None = None
    realized_x: list[float] = []
    realized_grade: list[float] = []
    initial_x = float(runtime.frame.root_position_world[0])
    for tick in range(_KNOWN_TRAIN_TICKS):
        command, traversal_direction = known_train_command(tick, warm_command)
        frame = runtime.step(command, camera_yaw=0.0)
        if first_hold is None and "hold_reason" in frame.diagnostics:
            first_hold = {"tick": tick, **frame.diagnostics}
        sample = terrain(frame.root_position_world[:2])
        if sample is None:
            raise ValueError("accepted runtime root left the known terrain query")
        recorder.record(
            frame,
            desired_velocity_world=command,
            terrain_sample=sample,
            traversal_direction=traversal_direction,
        )
        realized_x.append(float(frame.root_position_world[0]) - initial_x)
        realized_grade.append(sample.absolute_grade_degrees)
    report = recorder.finalize()
    runtime_gate_names = (
        "finite_20_seconds",
        "no_invalid_hold",
        "no_phase_reversal",
        "no_phase_freeze",
        "root_translation_step_within_limit",
        "root_rotation_step_within_limit",
        "joint_step_within_limit",
        "no_joint_limit_violation",
        "sole_penetration_within_limit",
        "zero_forbidden_body_penetration",
        "stance_sole_speed_within_limit",
    )
    flank_entry = float(terrain.alignment_report["flank_entry_runtime_x_m"])
    forward_maximum = max(realized_x, default=0.0)
    backward_minimum = min(realized_x[420:], default=forward_maximum)
    displacement = report["traversal"]["realized_displacement_m"]
    grade_by_direction = report["traversal"]["maximum_grade_degrees_by_direction"]
    signed_speeds = [
        (realized_x[index] - realized_x[index - 1]) * 30.0
        for index in range(1, len(realized_x))
    ]
    flat_return_x = max(0.0, flank_entry - 0.05)
    return_tick = next(
        (
            tick
            for tick in range(421, len(realized_x))
            if max(realized_x[:tick], default=0.0) >= flank_entry + 0.10
            and realized_x[tick] <= flat_return_x
            and realized_grade[tick] <= 0.5
        ),
        None,
    )
    responsive_return = bool(
        return_tick is not None
        and any(
            speed <= -0.175
            for speed in signed_speeds[
                max(0, return_tick - 1) : min(len(signed_speeds), return_tick + 60)
            ]
        )
    )
    known_gates: dict[str, bool] = {
        name: bool(report["gates"][name]) for name in runtime_gate_names
    }
    known_gates.update(
        {
            "queried_exact_paired_mesh_flank": terrain.exact_mesh_query_count > 0,
            "crossed_supported_known_terrain": bool(
                forward_maximum >= flank_entry + 0.10
                and grade_by_direction["forward"] >= 2.0
            ),
            "returned_to_supported_continuation": bool(
                return_tick is not None and grade_by_direction["backward"] >= 2.0
            ),
            "responsive_flat_motion_after_return": responsive_return,
            "realized_nonstationary_traversal": bool(
                displacement["forward"] >= 0.25
                and displacement["backward"] >= 0.25
            ),
        }
    )
    split_identities = manifest.get("split_identities", {}).get("train")
    if type(split_identities) is not list or not split_identities:
        raise ValueError("known-train identity-set provenance is missing")
    identity_base = {
        "schema": "mm-sonic-train-identity-set/v1",
        "dataset_digest_sha256": manifest["dataset_digest_sha256"],
        "split": "train",
        "identities": split_identities,
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity_base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    scenario_base = {
        "schema": "mm-sonic-known-train-scenario/v1",
        "dataset_digest_sha256": manifest["dataset_digest_sha256"],
        "train_identity_set_sha256": identity_sha,
        "kinematic_signature_sha256": kinematics.kinematic_signature_sha256,
        "source_record": source_name,
        "terrain_source_sha256": terrain.alignment_report["terrain_sha256"],
        "terrain_query_version": "exact-upward-triangle/v2",
        "command_script_version": "known-train-20s/v2",
    }
    scenario_sha = hashlib.sha256(
        json.dumps(scenario_base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_digest_sha256": manifest["dataset_digest_sha256"],
            "kinematic_signature_sha256": (
                kinematics.kinematic_signature_sha256
            ),
            "split": "train",
            "source_record": source_name,
            "terrain_alignment": {
                **terrain.alignment_report,
                "exact_mesh_query_count": terrain.exact_mesh_query_count,
                "extrapolated_query_count": terrain.extrapolated_query_count,
            },
            "native_geometry_audit": {
                "sole_geom_count": geometry.sole_geom_count,
                "forbidden_geom_count": geometry.forbidden_geom_count,
                "collision_surface_sample_count": (
                    geometry.total_collision_surface_samples
                ),
                "terrain_query_point_count": geometry.total_terrain_query_points,
                "mean_surface_samples_per_frame": (
                    geometry.total_collision_surface_samples / _KNOWN_TRAIN_TICKS
                ),
            },
            "command_script": {
                "duration_seconds": 20.0,
                "warm_0_2_m_s": warm_command.tolist(),
                "warm_speed_m_s": float(np.linalg.norm(warm_command)),
                "forward_2_12_m_s": 0.35,
                "idle_12_14_seconds": True,
                "backward_14_18_m_s": -0.35,
                "idle_18_20_seconds": True,
            },
            "known_train_gate": {
                **known_gates,
                "accepted": all(known_gates.values()),
                "claim_scope": "finite_known_train_slope_only",
                "unseen_hill_quality_claimed": False,
                "traverses_18_9_degrees_both_directions": False,
                "traverses_18_9_degrees_both_directions_applicable": False,
                "native_maximum_grade_degrees": terrain.alignment_report[
                    "native_maximum_grade_degrees"
                ],
                "realized_forward_maximum_x_m": forward_maximum,
                "realized_backward_minimum_x_m": backward_minimum,
                "flat_return_tick": return_tick,
                "flat_return_threshold_x_m": flat_return_x,
                "responsive_flat_motion_after_return": responsive_return,
                "maximum_grade_degrees_by_direction": grade_by_direction,
                "realized_displacement_m": displacement,
            },
            "scenario_provenance": {
                **scenario_base,
                "scenario_provenance_sha256": scenario_sha,
            },
            "first_hold": first_hold,
        }
    )
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    return candidate.parent if candidate.is_file() else candidate


def _fitted_subset_indices(
    dataset: PFNNShardDataset, fitted_subset: object
) -> list[int]:
    """Resolve every hash-bound fitted row without accepting a same-key duplicate."""

    rows = fitted_subset.get("rows") if type(fitted_subset) is dict else None
    if type(rows) is not list or not rows:
        raise ValueError("pipeline checkpoint fitted subset receipt is missing")
    wanted = {
        (
            row.get("clip_id"), row.get("sequence_lane"),
            row.get("center_frame"),
        ): row.get("row_sha256")
        for row in rows
        if type(row) is dict
    }
    if len(wanted) != len(rows):
        raise ValueError("pipeline checkpoint fitted subset receipt is invalid")
    matches: dict[tuple[object, object, object], list[int]] = {
        key: [] for key in wanted
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        key = (
            sample.get("clip_id"), sample.get("sequence_lane"),
            sample.get("center_frame"),
        )
        if key in wanted and fitted_row_sha256(sample) == wanted[key]:
            matches[key].append(index)
    if any(len(indices) != 1 for indices in matches.values()):
        raise ValueError("fitted subset row membership is not unique")
    return [
        matches[(row["clip_id"], row["sequence_lane"], row["center_frame"])][0]
        for row in rows
    ]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def claim_sealed_test_receipt(
    run_directory: str | Path,
    *,
    checkpoint_sha256: str,
    dataset_digest: str,
) -> Path:
    """Atomically reserve the sole sealed-test evaluation for a run directory."""

    if (
        _SHA256_RE.fullmatch(checkpoint_sha256) is None
        or _SHA256_RE.fullmatch(dataset_digest) is None
    ):
        raise ValueError("sealed-test receipt digests must be SHA-256 values")
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "sealed-test-receipt.json"
    encoded = (
        json.dumps(
            {
                "schema": "mm-sonic-terrain-pfnn-sealed-test-receipt/v1",
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_digest_sha256": dataset_digest,
                "selection_pass": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("sealed test has already been evaluated in this run directory") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        receipt.unlink(missing_ok=True)
        raise
    return receipt


def validate_checkpoint_kinematics(checkpoint: object, kinematics: object) -> None:
    if (
        getattr(checkpoint, "kinematic_signature_sha256", None)
        != getattr(kinematics, "kinematic_signature_sha256", None)
    ):
        raise ValueError("checkpoint kinematic signature mismatch")
    try:
        expected_limits = torch.as_tensor(
            getattr(kinematics, "joint_limits"), dtype=torch.float64, device="cpu"
        )
        checkpoint_limits = getattr(checkpoint, "joint_limits")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("checkpoint canonical joint limits mismatch") from error
    if (
        type(checkpoint_limits) is not torch.Tensor
        or checkpoint_limits.dtype != torch.float64
        or not torch.equal(checkpoint_limits, expected_limits)
    ):
        raise ValueError("checkpoint canonical joint limits mismatch")


def evaluate(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    model_path: str | Path,
    split: str,
    output_path: str | Path,
    sealed_test: bool,
    run_directory: str | Path | None,
    batch_size: int,
    device: str,
    closed_loop_seconds: float | None = None,
    promote_pipeline_checkpoint: bool = False,
) -> dict[str, object]:
    if split not in ("train", "validation", "test"):
        raise ValueError("evaluation split must be train, validation, or test")
    if split == "train" and closed_loop_seconds != 20.0:
        raise ValueError("train evaluation requires the exact 20 second closed-loop gate")
    if split != "train" and closed_loop_seconds is not None:
        raise ValueError(
            "validation/test closed-loop evaluation requires explicit Task-8 scenarios"
        )
    if split == "test" and not sealed_test:
        raise ValueError("test evaluation requires --sealed-test")
    if split != "test" and sealed_test:
        raise ValueError("--sealed-test is only valid for the test split")
    root = _dataset_root(dataset_path)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is missing or invalid") from error
    dataset_digest = manifest.get("dataset_digest_sha256")
    if type(dataset_digest) is not str or _SHA256_RE.fullmatch(dataset_digest) is None:
        raise ValueError("dataset digest is invalid")
    kinematics = TorchG1ForwardKinematics.from_mjcf(model_path)
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_dataset_digest=dataset_digest,
        expected_kinematic_signature_sha256=kinematics.kinematic_signature_sha256,
    )
    validate_checkpoint_kinematics(checkpoint, kinematics)
    expected_train = tuple(manifest.get("split_identities", {}).get("train", ()))
    expected_validation = tuple(manifest.get("split_identities", {}).get("validation", ()))
    if (
        checkpoint.train_identities != expected_train
        or checkpoint.validation_identities != expected_validation
    ):
        raise ValueError("checkpoint train/validation identities mismatch")
    dataset = PFNNShardDataset(root, split)
    for name in ("x_mean", "x_std", "y_mean", "y_std"):
        expected = torch.as_tensor(getattr(dataset, name), dtype=torch.float32)
        if not torch.equal(checkpoint.normalization[name], expected):
            raise ValueError("checkpoint normalization arrays mismatch")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_digest = _sha256(checkpoint_path)
    if split == "test":
        claim_sealed_test_receipt(
            Path(output_path).parent if run_directory is None else run_directory,
            checkpoint_sha256=checkpoint_digest,
            dataset_digest=dataset_digest,
        )
    target_device = torch.device(device)
    model = checkpoint.build_model().to(target_device)
    kinematics = kinematics.to(target_device)
    metrics = one_step_metrics(
        model,
        dataset,
        kinematics=kinematics,
        batch_size=batch_size,
        device=target_device,
        loss_weights=checkpoint.loss_weights,
    )
    closed_loop: dict[str, object]
    pipeline_promotion: dict[str, object] | None = None
    fitted_transition_report: dict[str, object] | None = None
    if split == "train":
        fitted_subset = getattr(checkpoint, "fitted_subset", None)
        selection = getattr(checkpoint, "selection", None)
        if type(fitted_subset) is dict:
            fixed_indices = _fitted_subset_indices(dataset, fitted_subset)
            adjacent = fitted_adjacent_indices(dataset, fixed_indices)
            transition_model = checkpoint.build_model().to(torch.device("cpu"))
            transition = evaluate_fitted_transition_envelope(
                transition_model,
                dataset,
                adjacent,
                normalization=checkpoint.normalization,
                joint_limits=checkpoint.joint_limits,
                phase_advance_q99=checkpoint.phase_advance_q99,
            )
            fitted_transition_report = validate_fitted_transition_report(
                transition
            )
            fixed_metrics = one_step_metrics(
                model,
                dataset,
                kinematics=kinematics,
                indices=fixed_indices,
                batch_size=batch_size,
                device=target_device,
                loss_weights=checkpoint.loss_weights,
            )
            receipt: dict[str, object] | None = None
            promoted = False
            if fitted_transition_report["accepted"] is True:
                if promote_pipeline_checkpoint and type(selection) is not dict:
                    raise ValueError(
                        "pipeline promotion requires a provisional fitted checkpoint"
                    )
                closed_loop = run_known_train_rollout(
                    checkpoint=checkpoint,
                    kinematics=kinematics,
                    model_path=model_path,
                    manifest=manifest,
                    split=split,
                    device=target_device,
                    checkpoint_sha256=checkpoint_digest,
                )
                if type(selection) is dict:
                    from mm_sonic.train_terrain_pfnn import (
                        build_pipeline_promotion_receipt,
                        promote_pipeline_best,
                    )

                    scenario_sha = closed_loop["scenario_provenance"][
                        "scenario_provenance_sha256"
                    ]
                    receipt = build_pipeline_promotion_receipt(
                        checkpoint_path=checkpoint_path,
                        dataset_digest_sha256=dataset_digest,
                        kinematic_signature_sha256=(
                            kinematics.kinematic_signature_sha256
                        ),
                        scenario_provenance_sha256=scenario_sha,
                        fitted_subset_rows_sha256=fitted_subset["rows_sha256"],
                        fitted_transition_report=fitted_transition_report,
                        fitted_transition_report_sha256=fitted_transition_report[
                            "report_sha256"
                        ],
                        expected_fixed_sample_score=float(
                            selection["one_step_score"]
                        ),
                        observed_fixed_sample_score=float(
                            fixed_metrics["one_step_score"]
                        ),
                        expected_fixed_sample_count=len(fixed_indices),
                        observed_fixed_sample_count=int(fixed_metrics["samples"]),
                        closed_loop_metrics=closed_loop,
                    )
                    if promote_pipeline_checkpoint:
                        promoted = promote_pipeline_best(
                            candidate_path=checkpoint_path,
                            best_path=checkpoint_path.parent / "best.pt",
                            receipt=receipt,
                            dataset_digest_sha256=dataset_digest,
                            kinematic_signature_sha256=(
                                kinematics.kinematic_signature_sha256
                            ),
                            scenario_provenance_sha256=scenario_sha,
                            fitted_subset_rows_sha256=fitted_subset["rows_sha256"],
                            fitted_transition_report=fitted_transition_report,
                            fitted_transition_report_sha256=(
                                fitted_transition_report["report_sha256"]
                            ),
                            expected_fixed_sample_score=float(
                                selection["one_step_score"]
                            ),
                            expected_fixed_sample_count=len(fixed_indices),
                            observed_fixed_sample_score=float(
                                fixed_metrics["one_step_score"]
                            ),
                            observed_fixed_sample_count=int(
                                fixed_metrics["samples"]
                            ),
                        )
                        if promoted:
                            _atomic_json(
                                checkpoint_path.parent
                                / "pipeline-promotion-receipt.json",
                                receipt,
                            )
            else:
                closed_loop = {
                    "status": "rejected_fitted_transition_envelope",
                    "failure_penalty": None,
                    "accepted": False,
                    "first_failure": fitted_transition_report["first_failure"],
                }
            if type(selection) is dict:
                pipeline_promotion = {
                    "receipt": receipt,
                    "fixed_sample_metrics": fixed_metrics,
                    "promoted": promoted,
                }
        else:
            if promote_pipeline_checkpoint:
                raise ValueError(
                    "pipeline promotion requires a provisional fitted checkpoint"
                )
            closed_loop = run_known_train_rollout(
                checkpoint=checkpoint,
                kinematics=kinematics,
                model_path=model_path,
                manifest=manifest,
                split=split,
                device=target_device,
                checkpoint_sha256=checkpoint_digest,
            )
    else:
        closed_loop = {
            "status": "task_8_scenarios_required",
            "failure_penalty": None,
            "accepted": None,
        }
    report: dict[str, object] = {
        "schema": "mm-sonic-terrain-pfnn-evaluation/v1",
        "checkpoint_sha256": checkpoint_digest,
        "dataset_digest_sha256": dataset_digest,
        "kinematic_signature_sha256": kinematics.kinematic_signature_sha256,
        "split": split,
        "sealed_test": split == "test",
        "one_step": metrics,
        "fitted_transition_report": fitted_transition_report,
        "fitted_transition_report_sha256": (
            None
            if fitted_transition_report is None
            else fitted_transition_report["report_sha256"]
        ),
        "closed_loop": closed_loop,
        "pipeline_promotion": pipeline_promotion,
    }
    _atomic_json(Path(output_path), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", default=str(_DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--output")
    parser.add_argument("--run-dir")
    parser.add_argument("--sealed-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--closed-loop-seconds", type=float)
    parser.add_argument("--promote-pipeline-best", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = (
        Path(arguments.checkpoint).resolve().parent
        / f"{arguments.split}-evaluation.json"
        if arguments.output is None
        else Path(arguments.output)
    )
    report = evaluate(
        checkpoint_path=arguments.checkpoint,
        dataset_path=arguments.dataset,
        model_path=arguments.model_path,
        split=arguments.split,
        output_path=output,
        sealed_test=arguments.sealed_test,
        run_directory=arguments.run_dir,
        batch_size=arguments.batch_size,
        device=arguments.device,
        closed_loop_seconds=arguments.closed_loop_seconds,
        promote_pipeline_checkpoint=arguments.promote_pipeline_best,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if arguments.split == "train":
        transition = report.get("fitted_transition_report")
        if type(transition) is dict and transition.get("accepted") is not True:
            return 2
        known = report.get("closed_loop", {}).get("known_train_gate")
        if type(known) is not dict or known.get("accepted") is not True:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "claim_sealed_test_receipt",
    "KnownSlopeTerrain",
    "evaluate",
    "known_train_command",
    "main",
    "select_known_train_grail_record",
    "run_known_train_rollout",
    "validate_checkpoint_kinematics",
]
