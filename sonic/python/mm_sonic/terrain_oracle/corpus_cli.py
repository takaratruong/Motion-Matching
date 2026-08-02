"""Command-line publication for the audited canonical terrain corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
import errno
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Iterable, Iterator, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from mm_sonic.joints import ContractError

from . import (
    coverage,
    source_flat,
    source_grail,
    source_justin,
    source_lafan,
    storage,
)
from .audit import (
    ClipAudit,
    audit_clip,
    structural_model_sha256,
    terrain_query_sha256,
)
from .canonical import CanonicalClip, CanonicalTerrainMesh, TerrainBinding
from .contact import (
    CanonicalMeshQuery,
    ContactConfig,
    SoleGeometry,
    reconstruct_contacts,
)
from .math3d import RigidTransform


_PHASE1_PRIMARY_INVENTORY_SHA256 = (
    "a7d6157116eec567b5a4df2daf21c95f"
    "4a6363b8dc8c6dfe1a8a5c6652c6063c"
)
_PHASE1_LAFAN_INVENTORY_SHA256 = (
    "1b1f8097c532a08c26e39e7f504f7bb"
    "2f4e887636129c59f95930ea0ef41d844"
)
_PHASE1_G1_ROOT_NAME = "g1_29dof_rev_1_0.xml"
_PHASE1_G1_ROOT_SHA256 = (
    "2e92915c253c6774d305cbd9ee13e7ea"
    "523d3c06a28551f8ce2f92fdc6138e87"
)
_PHASE1_G1_VFS_DIRECTORY_COUNT = 2
_PHASE1_G1_VFS_FILE_COUNT = 88
_PHASE1_G1_VFS_SIZE_BYTES = 60_248_622
_PHASE1_G1_VFS_TREE_SHA256 = (
    "e41a1311012dd9eaf3d5e733d5aefc82"
    "62f6024dd5c2f8e55dc617ab1db3977d"
)


@dataclass(frozen=True)
class _FreezeModelSnapshot:
    asset_path: str
    asset_payload: bytes
    compiled_mjb_payload: bytes
    model: object


def _reject_unsafe_cli_path(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if any(part in ("", "..") for part in candidate.parts):
        raise ContractError(f"{label} path must not contain traversal")
    absolute = candidate.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ContractError(f"{label} path must not traverse a symlink")
    return absolute


def _cli_input_directory(path: Path, label: str) -> Path:
    resolved = _reject_unsafe_cli_path(path, label).resolve()
    if not resolved.is_dir():
        raise ContractError(f"{label} is not a directory: {resolved}")
    return resolved


def _cli_input_file(path: Path, label: str) -> Path:
    resolved = _reject_unsafe_cli_path(path, label).resolve()
    if not resolved.is_file():
        raise ContractError(f"{label} is not a file: {resolved}")
    return resolved


def _cli_output_path(path: Path, label: str) -> Path:
    output = _reject_unsafe_cli_path(path, label)
    if output.name in ("", ".", ".."):
        raise ContractError(f"{label} must be a narrow path")
    return output


def _load_fixture(
    path: Path,
) -> tuple[
    tuple[CanonicalClip, ...],
    tuple[CanonicalTerrainMesh, ...],
    dict[str, object],
]:
    root = _cli_input_directory(path, "fixture corpus")
    manifest = storage.load_corpus(root)
    clips = tuple(
        storage.read_clip(root / record.relative_path)
        for record in manifest.clips
    )
    meshes = tuple(
        storage.read_mesh(root / record.relative_path)
        for record in manifest.meshes
    )
    return clips, meshes, dict(manifest.metadata)


def _publish_bundle(
    output: Path,
    clips: Iterable[CanonicalClip],
    meshes: Iterable[CanonicalTerrainMesh],
    metadata: dict[str, object],
) -> None:
    destination = _cli_output_path(output, "import output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        clip_records = tuple(
            sorted(
                (
                    storage.write_clip(stage / "clips", clip)
                    for clip in clips
                ),
                key=lambda record: (record.clip_id, record.sha256),
            )
        )
        mesh_records = tuple(
            sorted(
                (
                    storage.write_mesh(stage / "meshes", mesh)
                    for mesh in meshes
                ),
                key=lambda record: record.sha256,
            )
        )
        manifest_directory = storage.publish_corpus(
            stage / "_manifest",
            clip_records,
            {
                **metadata,
                "mesh_records": [
                    record.to_dict() for record in mesh_records
                ],
            },
        )
        os.replace(
            manifest_directory / "manifest.json",
            stage / "manifest.json",
        )
        (manifest_directory / storage.COMPLETION_MARKER).unlink()
        manifest_directory.rmdir()
        storage._seal_directory(
            stage,
            excluded_top_level=("coverage.json", "render-audit"),
        )
        storage._publish_directory_no_replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _import_fixture(arguments: argparse.Namespace) -> None:
    clips, meshes, metadata = _load_fixture(arguments.fixture)
    _publish_bundle(arguments.output, clips, meshes, metadata)


def _import_from_arguments(arguments: argparse.Namespace) -> None:
    real_options = (
        arguments.inventory,
        arguments.lafan_inventory,
        arguments.flat,
        arguments.justin,
        arguments.grail_root,
        arguments.grail_families,
        arguments.lafan,
        arguments.model,
        arguments.justin_terrain,
    )
    if arguments.fixture is not None:
        if any(value is not None for value in real_options):
            raise ContractError(
                "--fixture cannot be combined with real-source import options"
            )
        _import_fixture(arguments)
        return
    if any(value is None for value in real_options):
        raise ContractError(
            "real import requires --inventory, --flat, --justin, "
            "--grail-root, --grail-families, --lafan-inventory, "
            "--lafan, --model, and --justin-terrain"
        )
    _import_real_sources(arguments)


def _validated_inventory_document(
    path: Path,
    label: str,
    expected_sources: frozenset[str],
    trusted_file_sha256: str,
) -> dict[str, object]:
    inventory_path = _cli_input_file(path, label)
    document, payload = _load_canonical_json(inventory_path, label)
    if (
        _validated_sha256(
            trusted_file_sha256,
            f"{label} trusted file SHA-256",
        )
        != hashlib.sha256(payload).hexdigest()
    ):
        raise ContractError(
            f"{label} is not the trusted phase-1 inventory"
        )
    if set(document) != {
        "schema",
        "sources",
        "summary",
        "content_sha256",
    }:
        raise ContractError(f"{label} fields do not match v1")
    without_hash = dict(document)
    content_sha256 = without_hash.pop("content_sha256")
    if (
        document["schema"] != "terrain-oracle-source-inventory/v1"
        or _validated_sha256(content_sha256, f"{label} content SHA-256")
        != hashlib.sha256(
            storage._canonical_json_bytes(without_hash, label)
        ).hexdigest()
    ):
        raise ContractError(f"{label} hash is stale")
    sources = document["sources"]
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise ContractError(
            f"{label} must contain exactly {sorted(expected_sources)}"
        )
    counts_by_source: dict[str, int] = {}
    counts_by_format: Counter[str] = Counter()
    for name, source in sorted(sources.items()):
        if not isinstance(source, dict):
            raise ContractError(f"{label} source {name} must be an object")
        count = source.get("count")
        formats = source.get("counts_by_format")
        if type(count) is not int or count < 1:
            raise ContractError(
                f"{label} source {name} count must be positive"
            )
        if (
            not isinstance(formats, dict)
            or not formats
            or any(
                type(format_name) is not str
                or not format_name
                or type(format_count) is not int
                or format_count < 1
                for format_name, format_count in formats.items()
            )
            or sum(formats.values()) != count
        ):
            raise ContractError(
                f"{label} source {name} format counts are invalid"
            )
        counts_by_source[name] = count
        counts_by_format.update(formats)
    expected_summary = {
        "clip_count": sum(counts_by_source.values()),
        "counts_by_source": counts_by_source,
        "counts_by_format": dict(sorted(counts_by_format.items())),
    }
    if document["summary"] != expected_summary:
        raise ContractError(f"{label} summary is stale")
    return document


def _load_import_inventories(
    primary_path: Path,
    lafan_path: Path,
    *,
    primary_sha256: str = _PHASE1_PRIMARY_INVENTORY_SHA256,
    lafan_sha256: str = _PHASE1_LAFAN_INVENTORY_SHA256,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Load the two independently hashed source authorities for real import."""

    primary = _validated_inventory_document(
        primary_path,
        "primary source inventory",
        frozenset(("flat", "justin", "grail")),
        primary_sha256,
    )
    lafan = _validated_inventory_document(
        lafan_path,
        "LAFAN source inventory",
        frozenset(("lafan",)),
        lafan_sha256,
    )
    if primary["content_sha256"] == lafan["content_sha256"]:
        raise ContractError(
            "primary and LAFAN inventories must have distinct authority hashes"
        )
    sources = {
        **primary["sources"],
        **lafan["sources"],
    }
    return primary, lafan, sources


_BASE_PLANE_HALF_EXTENT_M = 16.0
_BASE_PLANE_VERTICES_WORLD = np.array(
    (
        (-_BASE_PLANE_HALF_EXTENT_M, -_BASE_PLANE_HALF_EXTENT_M, 0.0),
        (_BASE_PLANE_HALF_EXTENT_M, -_BASE_PLANE_HALF_EXTENT_M, 0.0),
        (_BASE_PLANE_HALF_EXTENT_M, _BASE_PLANE_HALF_EXTENT_M, 0.0),
        (-_BASE_PLANE_HALF_EXTENT_M, _BASE_PLANE_HALF_EXTENT_M, 0.0),
    ),
    dtype=np.float32,
)
_BASE_PLANE_FACES = np.array(
    ((0, 1, 2), (0, 2, 3)),
    dtype=np.int32,
)


def _normalized_obstacle_recipe(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "path",
        "size_bytes",
        "sha256",
        "license_id",
    }:
        raise ContractError("terrain obstacle recipe fields do not match v1")
    if value["kind"] not in ("justin-urdf", "grail-usd"):
        raise ContractError("terrain obstacle recipe kind is invalid")
    if (
        type(value["path"]) is not str
        or not value["path"]
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] < 1
        or type(value["license_id"]) is not str
        or not value["license_id"]
    ):
        raise ContractError("terrain obstacle recipe identity is invalid")
    return {
        "kind": value["kind"],
        "path": value["path"],
        "size_bytes": value["size_bytes"],
        "sha256": _validated_sha256(
            value["sha256"],
            "terrain obstacle SHA-256",
        ),
        "license_id": value["license_id"],
    }


def _bind_terrain_recipe(
    model_record: dict[str, object],
    obstacle: dict[str, object] | None,
    obstacle_mesh: CanonicalTerrainMesh | None,
    world_from_terrain: RigidTransform,
) -> tuple[
    CanonicalTerrainMesh,
    TerrainBinding,
    dict[str, object],
]:
    """Append the canonical world floor and bind the exact derivation recipe."""

    normalized_model = _normalized_model_record(
        model_record,
        "terrain recipe model",
    )
    normalized_obstacle = _normalized_obstacle_recipe(obstacle)
    if not isinstance(world_from_terrain, RigidTransform):
        raise ContractError("terrain recipe requires a RigidTransform")
    if (normalized_obstacle is None) != (obstacle_mesh is None):
        raise ContractError(
            "terrain recipe obstacle identity and mesh must be paired"
        )
    if (
        obstacle_mesh is not None
        and obstacle_mesh.source_asset_sha256
        != normalized_obstacle["sha256"]
    ):
        raise ContractError(
            "terrain obstacle mesh does not match its source authority"
        )
    recipe = {
        "schema": "terrain-oracle-terrain-recipe/v1",
        "base_plane": {
            "authority_model": normalized_model,
            "geom_name": "floor",
            "z_m": 0.0,
            "half_extent_m": _BASE_PLANE_HALF_EXTENT_M,
            "vertices_world": _BASE_PLANE_VERTICES_WORLD.tolist(),
            "faces": _BASE_PLANE_FACES.tolist(),
        },
        "obstacle": normalized_obstacle,
    }
    recipe_payload = storage._canonical_json_bytes(
        recipe,
        "terrain recipe",
    )
    recipe_sha256 = hashlib.sha256(recipe_payload).hexdigest()
    plane_local = world_from_terrain.inverse().apply_points(
        _BASE_PLANE_VERTICES_WORLD
    )
    if obstacle_mesh is None:
        vertices = plane_local
        faces = _BASE_PLANE_FACES
        valid_faces = np.ones(len(faces), dtype=np.bool_)
        license_id = "LicenseRef-Procedural"
    else:
        vertex_count = len(obstacle_mesh.vertices_local)
        vertices = np.concatenate(
            (obstacle_mesh.vertices_local, plane_local),
            axis=0,
        )
        faces = np.concatenate(
            (
                obstacle_mesh.faces,
                _BASE_PLANE_FACES + vertex_count,
            ),
            axis=0,
        )
        valid_faces = np.concatenate(
            (
                obstacle_mesh.valid_faces,
                np.ones(len(_BASE_PLANE_FACES), dtype=np.bool_),
            ),
            axis=0,
        )
        license_id = str(normalized_obstacle["license_id"])
    mesh = CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=faces,
        valid_faces=valid_faces,
        source_asset_sha256=recipe_sha256,
    )
    binding = TerrainBinding(
        asset_path=f"recipe://{recipe_sha256}",
        asset_size_bytes=len(recipe_payload),
        asset_sha256=recipe_sha256,
        asset_license_id=license_id,
        mesh_sha256=storage.mesh_digest(mesh),
        world_from_terrain=world_from_terrain,
        validity_mask_path=None,
    )
    return mesh, binding, recipe


def _urdf_vector(value: object, label: str) -> np.ndarray:
    if type(value) is not str:
        raise ContractError(f"Justin terrain {label} must be declared")
    try:
        vector = np.asarray(
            tuple(float(component) for component in value.split()),
            dtype=np.float64,
        )
    except ValueError as error:
        raise ContractError(
            f"Justin terrain {label} must contain three numbers"
        ) from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ContractError(
            f"Justin terrain {label} must contain three finite numbers"
        )
    return vector


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _load_justin_obstacle(
    path: Path,
) -> tuple[dict[str, object], CanonicalTerrainMesh]:
    """Parse the four authoritative Justin collision boxes without URDF code."""

    resolved = _cli_input_file(path, "Justin terrain URDF")
    payload = storage._read_regular_file_nofollow(resolved)
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ContractError("Justin terrain URDF must not declare entities")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ContractError("Justin terrain URDF is invalid XML") from error
    collisions = root.findall("./link/collision") if root.tag == "robot" else []
    if len(collisions) != 4:
        raise ContractError(
            "Justin terrain URDF must contain exactly four collision boxes"
        )
    unit_vertices = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    box_faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int32,
    )
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for collision_index, collision in enumerate(collisions):
        if set(collision.attrib):
            raise ContractError(
                "Justin terrain collision elements must not have attributes"
            )
        children = list(collision)
        if len(children) != 2 or {child.tag for child in children} != {
            "origin",
            "geometry",
        }:
            raise ContractError(
                "Justin terrain collisions require one origin and geometry"
            )
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        if origin is None or geometry is None or set(origin.attrib) != {
            "xyz",
            "rpy",
        }:
            raise ContractError(
                "Justin terrain collision origin fields do not match v1"
            )
        geometry_children = list(geometry)
        if (
            geometry.attrib
            or len(geometry_children) != 1
            or geometry_children[0].tag != "box"
            or set(geometry_children[0].attrib) != {"size"}
        ):
            raise ContractError(
                "Justin terrain collision geometry must be one box"
            )
        translation = _urdf_vector(
            origin.attrib["xyz"],
            "origin xyz",
        )
        rotation = _rpy_rotation(
            _urdf_vector(origin.attrib["rpy"], "origin rpy")
        )
        size = _urdf_vector(
            geometry_children[0].attrib["size"],
            "box size",
        )
        if np.any(size <= 0.0):
            raise ContractError(
                "Justin terrain box sizes must be positive"
            )
        box_vertices = (
            (unit_vertices * (size / 2.0)) @ rotation.T
            + translation
        )
        vertices.append(box_vertices)
        faces.append(box_faces + collision_index * len(unit_vertices))
    asset_sha256 = hashlib.sha256(payload).hexdigest()
    mesh = CanonicalTerrainMesh(
        vertices_local=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        valid_faces=np.ones(4 * len(box_faces), dtype=np.bool_),
        source_asset_sha256=asset_sha256,
    )
    obstacle = {
        "kind": "justin-urdf",
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": asset_sha256,
        "license_id": "UNRECORDED",
    }
    return obstacle, mesh


_REAL_IMPORT_COUNTS = {
    "flat": 173,
    "justin": 18,
    "grail": 489,
    "lafan": 40,
}


def _inventory_authority_metadata(
    path: Path,
    document: dict[str, object],
    label: str,
) -> dict[str, object]:
    resolved = _cli_input_file(path, label)
    payload = storage._read_regular_file_nofollow(resolved)
    try:
        snapshot = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} changed after validation") from error
    if snapshot != document:
        raise ContractError(f"{label} changed after validation")
    return {
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": document["content_sha256"],
        "summary": document["summary"],
    }


def _require_exact_inventory_sources(
    arguments: argparse.Namespace,
    declared_sources: dict[str, object],
) -> tuple[Path, Path, Path, tuple[str, ...], Path]:
    flat = _cli_input_directory(arguments.flat, "flat source root")
    justin = _cli_input_directory(arguments.justin, "Justin source root")
    grail = _cli_input_directory(arguments.grail_root, "GRAIL source root")
    lafan = _cli_input_directory(arguments.lafan, "LAFAN source root")
    families = tuple(str(arguments.grail_families).split(","))
    regenerated = {
        "flat": _flat_inventory(flat),
        "justin": _justin_inventory(justin),
        "grail": _grail_inventory(
            grail,
            str(arguments.grail_families),
        ),
        "lafan": _lafan_inventory(lafan),
    }
    for name in ("flat", "justin", "grail", "lafan"):
        if regenerated[name] != declared_sources[name]:
            raise ContractError(
                f"{name} source bytes or metadata differ from inventory authority"
            )
        if regenerated[name]["count"] != _REAL_IMPORT_COUNTS[name]:
            raise ContractError(
                f"{name} source count must equal "
                f"{_REAL_IMPORT_COUNTS[name]} for the phase-1 release"
            )
    return flat, justin, grail, families, lafan


@contextmanager
def _snapshot_real_import_roots(
    flat: Path,
    justin: Path,
    lafan: Path,
) -> Iterator[tuple[Path, Path, Path]]:
    """Yield private no-follow snapshots for every safe real-source adapter."""

    with tempfile.TemporaryDirectory(
        prefix="terrain-oracle-import-snapshot-"
    ) as temporary:
        snapshot_root = Path(temporary)
        os.chmod(snapshot_root, 0o700)
        flat_snapshot = snapshot_root / "flat"
        justin_snapshot = snapshot_root / "justin.zarr"
        lafan_release_snapshot = snapshot_root / "lafan-release"
        for destination in (
            flat_snapshot,
            justin_snapshot,
            lafan_release_snapshot,
        ):
            destination.mkdir(mode=0o700)
        _snapshot_directory_nofollow(flat, flat_snapshot)
        _snapshot_directory_nofollow(justin, justin_snapshot)
        _snapshot_directory_nofollow(
            lafan.parent,
            lafan_release_snapshot,
        )
        lafan_snapshot = lafan_release_snapshot / lafan.name
        if not lafan_snapshot.is_dir() or lafan_snapshot.is_symlink():
            raise ContractError(
                "LAFAN snapshot does not contain the authoritative g1 root"
            )
        yield flat_snapshot, justin_snapshot, lafan_snapshot


def _verified_inventory_file_payload(
    path: Path,
    identity: object,
    label: str,
) -> bytes:
    if not isinstance(identity, dict):
        raise ContractError(f"{label} inventory identity must be an object")
    required = {"path", "size_bytes", "sha256"}
    if not required.issubset(identity):
        raise ContractError(f"{label} inventory identity is incomplete")
    resolved = _cli_input_file(path, label)
    if str(resolved) != identity["path"]:
        raise ContractError(f"{label} path differs from inventory authority")
    payload = storage._read_regular_file_nofollow(resolved)
    if (
        len(payload) != identity["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != identity["sha256"]
    ):
        raise ContractError(f"{label} bytes differ from inventory authority")
    return payload


def _grail_inventory_record(
    root: Path,
    record: object,
) -> dict[str, object]:
    metadata_path = record.shard_path / "clips.json"  # type: ignore[attr-defined]
    metadata = _file_identity(
        metadata_path,
        metadata_path.relative_to(root).as_posix(),
    )
    robot_path = record.robot_path  # type: ignore[attr-defined]
    terrain_path = record.usd_path  # type: ignore[attr-defined]
    robot_identity = _file_identity(
        robot_path,
        robot_path.relative_to(root).as_posix(),
    )
    terrain_identity = _file_identity(
        terrain_path,
        terrain_path.relative_to(root).as_posix(),
    )
    return {
        "family": record.family,  # type: ignore[attr-defined]
        "stem": record.stem,  # type: ignore[attr-defined]
        "frame_count": record.n_frames,  # type: ignore[attr-defined]
        "robot": {
            **robot_identity,
            "path": str(robot_path.resolve()),
            "license_id": "UNRECORDED",
        },
        "terrain": {
            **terrain_identity,
            "path": str(terrain_path.resolve()),
            "license_id": "UNRECORDED",
            "world_from_terrain": {
                "translation_world": np.asarray(
                    record.terrain_position_env,  # type: ignore[attr-defined]
                    dtype=np.float32,
                ).tolist(),
                "quaternion_world_from_local_wxyz": np.asarray(
                    record.terrain_rotation_env_wxyz,  # type: ignore[attr-defined]
                    dtype=np.float32,
                ).tolist(),
            },
        },
        "shard_metadata": metadata,
        "pose_source": record.pose_source,  # type: ignore[attr-defined]
    }


def _grail_record_from_inventory(
    root: Path,
    value: dict[str, object],
) -> source_grail.GrailClipRecord:
    """Reconstruct one record from the code-anchored inventory authority."""

    if not isinstance(value, dict) or set(value) != {
        "family",
        "stem",
        "frame_count",
        "robot",
        "terrain",
        "shard_metadata",
        "pose_source",
    }:
        raise ContractError("GRAIL inventory record fields do not match v1")

    def source_path(
        identity: object,
        label: str,
        *,
        require_absolute_authority: bool,
    ) -> Path:
        if not isinstance(identity, dict):
            raise ContractError(f"{label} identity must be an object")
        relative_value = identity.get("relative_path")
        if type(relative_value) is not str or not relative_value:
            raise ContractError(f"{label} relative path is invalid")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_value
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ContractError(f"{label} relative path is not canonical")
        path = root / relative
        if require_absolute_authority and identity.get("path") != str(path):
            raise ContractError(
                f"{label} absolute path differs from inventory root"
            )
        return path

    robot = value["robot"]
    terrain = value["terrain"]
    metadata = value["shard_metadata"]
    robot_path = source_path(
        robot,
        "GRAIL robot",
        require_absolute_authority=True,
    )
    terrain_path = source_path(
        terrain,
        "GRAIL terrain",
        require_absolute_authority=True,
    )
    metadata_path = source_path(
        metadata,
        "GRAIL shard metadata",
        require_absolute_authority=False,
    )
    if metadata_path.name != "clips.json":
        raise ContractError("GRAIL shard metadata must name clips.json")
    if not isinstance(terrain, dict):
        raise ContractError("GRAIL terrain identity must be an object")
    transform_value = terrain.get("world_from_terrain")
    if not isinstance(transform_value, dict) or set(transform_value) != {
        "translation_world",
        "quaternion_world_from_local_wxyz",
    }:
        raise ContractError("GRAIL inventory terrain transform is invalid")
    try:
        transform = RigidTransform(
            transform_value["translation_world"],
            transform_value["quaternion_world_from_local_wxyz"],
        )
    except (TypeError, ValueError) as error:
        raise ContractError(
            "GRAIL inventory terrain transform is invalid"
        ) from error
    family = value["family"]
    stem = value["stem"]
    frame_count = value["frame_count"]
    pose_source = value["pose_source"]
    if (
        type(family) is not str
        or not family
        or type(stem) is not str
        or not stem
        or type(frame_count) is not int
        or frame_count < 1
        or type(pose_source) is not str
        or not pose_source
    ):
        raise ContractError("GRAIL inventory record semantics are invalid")
    return source_grail.GrailClipRecord(
        family=family,
        shard_path=metadata_path.parent,
        stem=stem,
        robot_path=robot_path,
        usd_path=terrain_path,
        n_frames=frame_count,
        terrain_position_env=np.asarray(
            transform.translation_world,
            dtype=np.float32,
        ),
        terrain_rotation_env_wxyz=np.asarray(
            transform.quaternion_world_from_local_wxyz,
            dtype=np.float32,
        ),
        pose_source=pose_source,
    )


def _load_verified_grail_source(
    record: object,
    inventory_record: dict[str, object],
    fk: object,
    *,
    grail_root: Path,
) -> source_grail.GrailCanonicalSource:
    """Snapshot exact PKL/USD bytes before either unsafe parser can observe them."""

    if _grail_inventory_record(grail_root, record) != inventory_record:
        raise ContractError(
            "GRAIL record metadata or semantics differ from inventory authority"
        )
    robot_identity = inventory_record.get("robot")
    terrain_identity = inventory_record.get("terrain")
    if not isinstance(robot_identity, dict) or not isinstance(
        terrain_identity, dict
    ):
        raise ContractError("GRAIL inventory pair identity is incomplete")
    robot_payload = _verified_inventory_file_payload(
        record.robot_path,  # type: ignore[attr-defined]
        robot_identity,
        f"GRAIL robot {record.stem}",  # type: ignore[attr-defined]
    )
    terrain_payload = _verified_inventory_file_payload(
        record.usd_path,  # type: ignore[attr-defined]
        terrain_identity,
        f"GRAIL terrain {record.stem}",  # type: ignore[attr-defined]
    )
    try:
        import joblib
        from mm_sonic.grail_terrain_source import (
            parse_grail_motion_blob,
        )

        blob = joblib.load(io.BytesIO(robot_payload))
        motion = parse_grail_motion_blob(
            blob,
            expected_frames=record.n_frames,  # type: ignore[attr-defined]
        )
    except Exception as error:
        raise ContractError(
            f"verified GRAIL robot payload is invalid: "
            f"{record.stem}"  # type: ignore[attr-defined]
        ) from error
    terrain_sha256 = hashlib.sha256(terrain_payload).hexdigest()
    with tempfile.TemporaryDirectory(
        prefix="terrain-oracle-grail-usd-"
    ) as temporary:
        snapshot_root = Path(temporary)
        os.chmod(snapshot_root, 0o700)
        snapshot_path = snapshot_root / "terrain.usd"
        storage._write_no_replace(snapshot_path, terrain_payload)
        try:
            mesh = source_grail._load_usd_mesh(
                snapshot_path,
                source_asset_sha256=terrain_sha256,
            )
        except Exception as error:
            raise ContractError(
                f"verified GRAIL terrain payload is invalid: "
                f"{record.stem}"  # type: ignore[attr-defined]
            ) from error
    return source_grail._canonicalize_verified_record(
        record,
        fk,
        motion=motion,
        mesh=mesh,
        robot_bytes=robot_payload,
        asset_bytes=terrain_payload,
    )


def _import_real_sources(arguments: argparse.Namespace) -> None:
    primary, lafan_inventory, declared_sources = (
        _load_import_inventories(
            arguments.inventory,
            arguments.lafan_inventory,
        )
    )
    flat_root, justin_root, grail_root, _families, lafan_root = (
        _require_exact_inventory_sources(arguments, declared_sources)
    )
    model_path = _cli_input_file(arguments.model, "mechanical model")
    try:
        import mujoco
    except ImportError as error:
        raise ContractError("real import requires MuJoCo") from error
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_record = _model_record(model_path, model)
    contact_config = ContactConfig(SoleGeometry.from_model(model))
    try:
        fk = source_grail.G1MujocoFK(model_path)
    except Exception as error:
        raise ContractError("could not initialize canonical G1 FK") from error

    identity = RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )
    plane_mesh, plane_binding, plane_recipe = _bind_terrain_recipe(
        model_record,
        None,
        None,
        identity,
    )
    justin_obstacle, justin_obstacle_mesh = _load_justin_obstacle(
        arguments.justin_terrain
    )
    justin_mesh, justin_binding, justin_recipe = _bind_terrain_recipe(
        model_record,
        justin_obstacle,
        justin_obstacle_mesh,
        identity,
    )

    meshes_by_sha256: dict[str, CanonicalTerrainMesh] = {}
    recipes_by_sha256: dict[str, dict[str, object]] = {}

    def register_terrain(
        mesh: CanonicalTerrainMesh,
        binding: TerrainBinding,
        recipe: dict[str, object],
    ) -> None:
        digest = storage.mesh_digest(mesh)
        if (
            digest != binding.mesh_sha256
            or mesh.source_asset_sha256 != binding.asset_sha256
        ):
            raise ContractError("terrain recipe binding is internally stale")
        existing_mesh = meshes_by_sha256.setdefault(digest, mesh)
        if storage.mesh_digest(existing_mesh) != digest:
            raise ContractError("terrain mesh digest collision")
        existing_recipe = recipes_by_sha256.setdefault(
            binding.asset_sha256,
            recipe,
        )
        if existing_recipe != recipe:
            raise ContractError("terrain recipe SHA-256 collision")

    register_terrain(plane_mesh, plane_binding, plane_recipe)
    register_terrain(justin_mesh, justin_binding, justin_recipe)

    primary_sources = primary["sources"]
    lafan_sources = lafan_inventory["sources"]
    flat_records = {
        str(record["clip_id"]): record
        for record in primary_sources["flat"]["records"]
    }
    justin_records = {
        str(record["clip_id"]): record
        for record in primary_sources["justin"]["records"]
    }
    grail_records = {
        (str(record["family"]), str(record["stem"])): record
        for record in primary_sources["grail"]["records"]
    }
    lafan_records = {
        str(record["clip_id"]): record
        for record in lafan_sources["lafan"]["records"]
    }
    expected_ids = {
        "flat": set(flat_records),
        "justin": set(justin_records),
        "grail": set(grail_records),
        "lafan": set(lafan_records),
    }
    seen_ids: dict[str, set[object]] = {
        name: set() for name in expected_ids
    }
    published_clip_ids: set[str] = set()
    frame_counts: Counter[str] = Counter()
    metadata: dict[str, object] = {
        "schema": "terrain-oracle-real-import/v1",
        "source_inventories": {
            "primary": _inventory_authority_metadata(
                arguments.inventory,
                primary,
                "primary source inventory",
            ),
            "lafan": _inventory_authority_metadata(
                arguments.lafan_inventory,
                lafan_inventory,
                "LAFAN source inventory",
            ),
        },
        "model": model_record,
        "terrain_recipes": recipes_by_sha256,
    }

    def bind_clip(
        clip: CanonicalClip,
        *,
        source_name: str,
        clip_id: str,
        mesh: CanonicalTerrainMesh,
        binding: TerrainBinding,
    ) -> CanonicalClip:
        if clip_id in published_clip_ids:
            raise ContractError(f"duplicate imported clip ID: {clip_id}")
        renamed = replace(clip, clip_id=clip_id, terrain=binding)
        reconstructed = reconstruct_contacts(
            renamed,
            CanonicalMeshQuery(mesh, binding.world_from_terrain),
            contact_config,
        ).apply(renamed)
        published_clip_ids.add(clip_id)
        frame_counts[source_name] += reconstructed.frame_count
        return reconstructed

    def validate_simple_record(
        clip: CanonicalClip,
        expected: dict[str, object],
        label: str,
    ) -> None:
        actual = _clip_inventory_record(clip)
        if actual != expected:
            raise ContractError(
                f"{label} adapter output differs from inventory authority"
            )

    flat_import_root = flat_root
    justin_import_root = justin_root
    lafan_import_root = lafan_root

    def clip_stream() -> Iterable[CanonicalClip]:
        for clip in source_flat.iter_flat_clips(
            flat_import_root,
            tags=("flat", "other"),
            authority_root=flat_root,
        ):
            expected = flat_records.get(clip.clip_id)
            if expected is None or clip.clip_id in seen_ids["flat"]:
                raise ContractError(
                    f"unexpected flat clip during import: {clip.clip_id}"
                )
            validate_simple_record(clip, expected, "flat")
            seen_ids["flat"].add(clip.clip_id)
            yield bind_clip(
                clip,
                source_name="flat",
                clip_id=f"flat/{clip.clip_id}",
                mesh=plane_mesh,
                binding=plane_binding,
            )

        for clip in source_justin.iter_justin_clips(
            justin_import_root,
            justin_binding,
            authority_path=justin_root,
        ):
            expected = justin_records.get(clip.clip_id)
            if expected is None or clip.clip_id in seen_ids["justin"]:
                raise ContractError(
                    f"unexpected Justin clip during import: {clip.clip_id}"
                )
            validate_simple_record(clip, expected, "Justin")
            seen_ids["justin"].add(clip.clip_id)
            yield bind_clip(
                clip,
                source_name="justin",
                clip_id=f"justin/{clip.clip_id}",
                mesh=justin_mesh,
                binding=justin_binding,
            )

        discovered = tuple(
            _grail_record_from_inventory(
                grail_root,
                inventory_record,
            )
            for _, inventory_record in sorted(grail_records.items())
        )
        for source_record in discovered:
            key = (source_record.family, source_record.stem)
            inventory_record = grail_records.get(key)
            if inventory_record is None or key in seen_ids["grail"]:
                raise ContractError(
                    f"unexpected GRAIL clip during import: {key}"
                )
            source = _load_verified_grail_source(
                source_record,
                inventory_record,
                fk,
                grail_root=grail_root,
            )
            robot = inventory_record["robot"]
            if (
                source.clip.source.source_path != robot["path"]
                or source.clip.source.source_size_bytes
                != robot["size_bytes"]
                or source.clip.source.source_sha256 != robot["sha256"]
                or source.clip.source.source_license_id
                != robot["license_id"]
            ):
                raise ContractError(
                    f"GRAIL adapter provenance differs for {key}"
                )
            terrain = inventory_record["terrain"]
            obstacle = {
                "kind": "grail-usd",
                "path": terrain["path"],
                "size_bytes": terrain["size_bytes"],
                "sha256": terrain["sha256"],
                "license_id": terrain["license_id"],
            }
            world_from_terrain = source.clip.terrain.world_from_terrain
            mesh, binding, recipe = _bind_terrain_recipe(
                model_record,
                obstacle,
                source.mesh,
                world_from_terrain,
            )
            register_terrain(mesh, binding, recipe)
            seen_ids["grail"].add(key)
            yield bind_clip(
                source.clip,
                source_name="grail",
                clip_id=f"grail/{source.clip.clip_id}",
                mesh=mesh,
                binding=binding,
            )

        for clip_name, inventory_record in sorted(lafan_records.items()):
            authority_path = Path(
                str(inventory_record["source_path"])
            )
            try:
                relative_path = authority_path.relative_to(lafan_root)
            except ValueError as error:
                raise ContractError(
                    "LAFAN inventory path escapes its authoritative root"
                ) from error
            clip = source_lafan.load_lafan_csv(
                lafan_import_root / relative_path,
                model_path,
                terrain=plane_binding,
                authority_path=authority_path,
                fk=fk,
            )
            expected_identity = {
                key: inventory_record[key]
                for key in (
                    "clip_id",
                    "source_format",
                    "source_path",
                    "source_size_bytes",
                    "source_sha256",
                    "source_license_id",
                )
            }
            if (
                {
                    "clip_id": clip.clip_id,
                    "source_format": clip.source.source_format,
                    "source_path": clip.source.source_path,
                    "source_size_bytes": clip.source.source_size_bytes,
                    "source_sha256": clip.source.source_sha256,
                    "source_license_id": clip.source.source_license_id,
                }
                != expected_identity
                or clip.clip_id in seen_ids["lafan"]
            ):
                raise ContractError(
                    f"LAFAN adapter provenance differs for {clip_name}"
                )
            seen_ids["lafan"].add(clip.clip_id)
            if "obstacle" in clip.action_tags:
                unbound_id = f"lafan/{clip.clip_id}"
                if unbound_id in published_clip_ids:
                    raise ContractError(
                        f"duplicate imported clip ID: {unbound_id}"
                    )
                published_clip_ids.add(unbound_id)
                frame_counts["lafan"] += clip.frame_count
                yield replace(clip, clip_id=unbound_id, terrain=None)
            else:
                yield bind_clip(
                    clip,
                    source_name="lafan",
                    clip_id=f"lafan/{clip.clip_id}",
                    mesh=plane_mesh,
                    binding=plane_binding,
                )

        for name, expected in expected_ids.items():
            if seen_ids[name] != expected:
                raise ContractError(
                    f"{name} adapter did not exactly cover its inventory"
                )
        metadata["summary"] = {
            "clip_count": len(published_clip_ids),
            "frame_count": sum(frame_counts.values()),
            "counts_by_source": dict(_REAL_IMPORT_COUNTS),
            "frames_by_source": dict(sorted(frame_counts.items())),
        }

    with _snapshot_real_import_roots(
        flat_root,
        justin_root,
        lafan_root,
    ) as snapshots:
        (
            flat_import_root,
            justin_import_root,
            lafan_import_root,
        ) = snapshots
        _publish_bundle(
            arguments.output,
            clip_stream(),
            meshes_by_sha256.values(),
            metadata,
        )


def _model_record(path: Path, model: object) -> dict[str, object]:
    resolved = _cli_input_file(path, "mechanical model")
    payload = resolved.read_bytes()
    return {
        "asset_path": str(resolved),
        "asset_size_bytes": len(payload),
        "asset_sha256": hashlib.sha256(payload).hexdigest(),
        "structural_sha256": structural_model_sha256(model),
    }


def _validated_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalized_model_record(
    value: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "asset_path",
        "asset_size_bytes",
        "asset_sha256",
        "structural_sha256",
    }:
        raise ContractError(f"{label} fields do not match v1")
    asset_path = value["asset_path"]
    asset_size_bytes = value["asset_size_bytes"]
    if type(asset_path) is not str or not asset_path:
        raise ContractError(f"{label} asset_path must be a nonempty string")
    if type(asset_size_bytes) is not int or asset_size_bytes < 1:
        raise ContractError(
            f"{label} asset_size_bytes must be a positive integer"
        )
    return {
        "asset_path": asset_path,
        "asset_size_bytes": asset_size_bytes,
        "asset_sha256": _validated_sha256(
            value["asset_sha256"],
            f"{label} asset_sha256",
        ),
        "structural_sha256": _validated_sha256(
            value["structural_sha256"],
            f"{label} structural_sha256",
        ),
    }


def _normalized_manifest_audit_metadata(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "source_corpus_manifest_sha256",
        "model",
    }:
        raise ContractError("manifest audit metadata fields do not match v1")
    if value["schema"] != "terrain-oracle-audit-index/v1":
        raise ContractError("manifest audit metadata schema is stale")
    return {
        "schema": value["schema"],
        "source_corpus_manifest_sha256": _validated_sha256(
            value["source_corpus_manifest_sha256"],
            "manifest source corpus SHA-256",
        ),
        "model": _normalized_model_record(
            value["model"],
            "manifest audit model",
        ),
    }


def _audit_entry(
    report: ClipAudit,
    relative_path: str,
    report_sha256: str,
) -> dict[str, object]:
    return {
        "clip_id": report.clip_id,
        "clip_sha256": report.clip_sha256,
        "source_sha256": report.source_sha256,
        "model_sha256": report.model_sha256,
        "terrain_sha256": report.terrain_sha256,
        "status": report.status,
        "accepted_intervals": [
            list(interval) for interval in report.accepted_intervals
        ],
        "reasons": [reason.to_dict() for reason in report.reasons],
        "relative_path": relative_path,
        "sha256": report_sha256,
    }


def _verify_audit_stage(
    stage: Path,
    expected_index: dict[str, object],
) -> None:
    manifest = storage.load_corpus(stage)
    for record in manifest.clips:
        clip = storage.read_clip(stage / record.relative_path)
        if (
            clip.clip_id != record.clip_id
            or clip.frame_count != record.frame_count
        ):
            raise ContractError("audited clip record does not match its bytes")
    for record in manifest.meshes:
        storage.read_mesh(stage / record.relative_path)
    index_path = stage / "audit-index.json"
    loaded_index = json.loads(index_path.read_text("ascii"))
    if loaded_index != expected_index:
        raise ContractError("reloaded audit index does not match publication")
    reports = loaded_index["reports"]
    if not isinstance(reports, list) or len(reports) != len(manifest.clips):
        raise ContractError("audit index must contain exactly one report per clip")
    for entry in reports:
        if not isinstance(entry, dict):
            raise ContractError("audit index reports must be objects")
        report_path = stage / str(entry["relative_path"])
        payload = report_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ContractError("audit report hash mismatch after publication")
        report = ClipAudit.from_dict(json.loads(payload.decode("ascii")))
        if _audit_entry(
            report,
            str(entry["relative_path"]),
            str(entry["sha256"]),
        ) != entry:
            raise ContractError("audit report and index evidence mismatch")


def _publish_audit_bundle(
    output: Path,
    clips: Sequence[CanonicalClip],
    meshes: Sequence[CanonicalTerrainMesh],
    metadata: dict[str, object],
    reports: Sequence[ClipAudit],
    source_manifest_sha256: str,
    model_record: dict[str, object],
) -> None:
    destination = _cli_output_path(output, "audit output")
    if len(reports) != len(clips):
        raise ContractError("mechanical audit requires one report per clip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        clip_records = tuple(
            storage.write_clip(stage / "clips", clip)
            for clip in clips
        )
        mesh_records = tuple(
            storage.write_mesh(stage / "meshes", mesh)
            for mesh in meshes
        )
        manifest_directory = storage.publish_corpus(
            stage / "_manifest",
            clip_records,
            {
                **metadata,
                "audit": {
                    "schema": "terrain-oracle-audit-index/v1",
                    "source_corpus_manifest_sha256": (
                        source_manifest_sha256
                    ),
                    "model": model_record,
                },
                "mesh_records": [
                    record.to_dict() for record in mesh_records
                ],
            },
        )
        os.replace(
            manifest_directory / "manifest.json",
            stage / "manifest.json",
        )
        (manifest_directory / storage.COMPLETION_MARKER).unlink()
        manifest_directory.rmdir()
        output_manifest_sha256 = hashlib.sha256(
            (stage / "manifest.json").read_bytes()
        ).hexdigest()
        entries: list[dict[str, object]] = []
        for record, report in zip(
            clip_records,
            reports,
            strict=True,
        ):
            if report.clip_sha256 != record.sha256:
                raise ContractError(
                    "audit report does not bind the republished clip bytes"
                )
            relative_path = f"audits/{record.sha256}.json"
            payload = storage._canonical_json_bytes(
                report.to_dict(),
                "mechanical audit report",
            )
            report_sha256 = hashlib.sha256(payload).hexdigest()
            storage._write_no_replace(stage / relative_path, payload)
            entries.append(
                _audit_entry(report, relative_path, report_sha256)
            )
        entries.sort(
            key=lambda entry: (
                str(entry["clip_id"]),
                str(entry["clip_sha256"]),
            )
        )
        without_hash = {
            "schema": "terrain-oracle-audit-index/v1",
            "source_corpus_manifest_sha256": source_manifest_sha256,
            "corpus_manifest_sha256": output_manifest_sha256,
            "model": model_record,
            "reports": entries,
        }
        content_sha256 = hashlib.sha256(
            storage._canonical_json_bytes(
                without_hash,
                "mechanical audit index",
            )
        ).hexdigest()
        index = {
            **without_hash,
            "content_sha256": content_sha256,
        }
        storage._write_no_replace(
            stage / "audit-index.json",
            storage._canonical_json_bytes(
                index,
                "mechanical audit index",
            ),
        )
        storage._seal_directory(
            stage,
            excluded_top_level=("coverage.json", "render-audit"),
        )
        _verify_audit_stage(stage, index)
        storage._publish_directory_no_replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _audit_corpus(arguments: argparse.Namespace) -> None:
    root = _cli_input_directory(arguments.corpus, "corpus")
    manifest = storage.load_corpus(root)
    clips = tuple(
        storage.read_clip(root / record.relative_path)
        for record in manifest.clips
    )
    meshes = tuple(
        storage.read_mesh(root / record.relative_path)
        for record in manifest.meshes
    )
    mesh_by_sha256 = {
        record.sha256: mesh
        for record, mesh in zip(
            manifest.meshes,
            meshes,
            strict=True,
        )
    }
    if len(mesh_by_sha256) != len(meshes):
        raise ContractError("corpus mesh records must be unique")
    try:
        import mujoco
    except ImportError as error:
        raise ContractError("mechanical audit requires MuJoCo") from error
    model_path = _cli_input_file(arguments.model, "mechanical model")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_record = _model_record(model_path, model)
    reports: list[ClipAudit] = []
    for clip, record in zip(
        clips,
        manifest.clips,
        strict=True,
    ):
        if (
            clip.clip_id != record.clip_id
            or clip.frame_count != record.frame_count
        ):
            raise ContractError("corpus clip record does not match its bytes")
        query = None
        if clip.terrain is not None:
            mesh = mesh_by_sha256.get(clip.terrain.mesh_sha256)
            if mesh is None:
                raise ContractError(
                    f"clip {clip.clip_id} terrain mesh is absent from corpus"
                )
            query = CanonicalMeshQuery(
                mesh,
                clip.terrain.world_from_terrain,
            )
        reports.append(audit_clip(clip, model, query))
    source_manifest_sha256 = hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()
    _publish_audit_bundle(
        arguments.output,
        clips,
        meshes,
        dict(manifest.metadata),
        reports,
        source_manifest_sha256,
        model_record,
    )


def _canonical_relative_path(value: object, directory: str) -> str:
    if type(value) is not str or not value:
        raise ContractError("artifact relative path must be a nonempty string")
    path = Path(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or not path.parts
        or path.parts[0] != directory
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ContractError(
            f"artifact relative path must be canonical beneath {directory}/"
        )
    return value


def _load_audit_evidence(
    root: Path,
    *,
    model_snapshot: _FreezeModelSnapshot | None = None,
) -> tuple[
    storage.CorpusManifest,
    tuple[CanonicalClip, ...],
    dict[str, object],
    dict[str, ClipAudit],
]:
    manifest_document, manifest_payload = _load_canonical_json(
        root / "manifest.json",
        "audited corpus manifest",
    )
    if set(manifest_document) != {
        "schema",
        "coordinate_frame",
        "quaternion_convention",
        "fps",
        "joint_order",
        "body_order",
        "clips",
        "meshes",
        "metadata",
    }:
        raise ContractError("audited corpus manifest fields do not match v1")
    manifest = storage.load_corpus(root)
    clips = tuple(
        storage.read_clip(root / record.relative_path)
        for record in manifest.clips
    )
    meshes = tuple(
        storage.read_mesh(root / record.relative_path)
        for record in manifest.meshes
    )
    mesh_by_sha256 = {
        record.sha256: mesh
        for record, mesh in zip(
            manifest.meshes,
            meshes,
            strict=True,
        )
    }
    if len(mesh_by_sha256) != len(meshes):
        raise ContractError("audited corpus mesh records must be unique")
    try:
        index_payload = (root / "audit-index.json").read_bytes()
        index = json.loads(index_payload.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("invalid mechanical audit index") from error
    if not isinstance(index, dict) or set(index) != {
        "schema",
        "source_corpus_manifest_sha256",
        "corpus_manifest_sha256",
        "model",
        "reports",
        "content_sha256",
    }:
        raise ContractError("mechanical audit index fields do not match v1")
    without_hash = dict(index)
    declared_content_sha256 = without_hash.pop("content_sha256")
    if (
        index["schema"] != "terrain-oracle-audit-index/v1"
        or declared_content_sha256
        != hashlib.sha256(
            storage._canonical_json_bytes(
                without_hash,
                "mechanical audit index",
            )
        ).hexdigest()
        or index["corpus_manifest_sha256"]
        != hashlib.sha256(manifest_payload).hexdigest()
    ):
        raise ContractError("mechanical audit index hash or corpus binding is stale")
    source_manifest_sha256 = _validated_sha256(
        index["source_corpus_manifest_sha256"],
        "audit source corpus SHA-256",
    )
    _validated_sha256(
        index["corpus_manifest_sha256"],
        "audited corpus manifest SHA-256",
    )
    model_record = _normalized_model_record(
        index["model"],
        "mechanical audit model",
    )
    manifest_audit = _normalized_manifest_audit_metadata(
        manifest.metadata.get("audit")
    )
    if manifest_audit != {
        "schema": "terrain-oracle-audit-index/v1",
        "source_corpus_manifest_sha256": source_manifest_sha256,
        "model": model_record,
    }:
        raise ContractError(
            "manifest audit metadata does not match index authority"
        )
    reconstructed_source = dict(manifest_document)
    reconstructed_metadata = dict(reconstructed_source["metadata"])
    removed_audit = reconstructed_metadata.pop("audit", None)
    if removed_audit is None:
        raise ContractError("audited corpus manifest lacks audit metadata")
    reconstructed_source["metadata"] = reconstructed_metadata
    if source_manifest_sha256 != hashlib.sha256(
        storage._canonical_json_bytes(
            reconstructed_source,
            "reconstructed source corpus manifest",
        )
    ).hexdigest():
        raise ContractError(
            "audit source manifest authority does not match reconstructed bytes"
        )
    if model_snapshot is None:
        model_path = _cli_input_file(
            Path(str(model_record["asset_path"])),
            "mechanical audit model asset",
        )
        if str(model_path) != model_record["asset_path"]:
            raise ContractError(
                "mechanical audit model asset path is not canonical"
            )
        model_payload = storage._read_regular_file_nofollow(model_path)
        try:
            import mujoco
        except ImportError as error:
            raise ContractError("render audit requires MuJoCo") from error
        model = mujoco.MjModel.from_xml_path(str(model_path))
    else:
        if model_snapshot.asset_path != model_record["asset_path"]:
            raise ContractError(
                "freeze model snapshot path does not match audit authority"
            )
        model_payload = model_snapshot.asset_payload
        model = model_snapshot.model
    if (
        len(model_payload) != model_record["asset_size_bytes"]
        or hashlib.sha256(model_payload).hexdigest()
        != model_record["asset_sha256"]
    ):
        raise ContractError("mechanical audit model asset hash is stale")
    if structural_model_sha256(model) != model_record["structural_sha256"]:
        raise ContractError("mechanical audit structural model hash is stale")
    raw_entries = index["reports"]
    if not isinstance(raw_entries, list):
        raise ContractError("mechanical audit reports must be a list")
    reports: dict[str, ClipAudit] = {}
    entries_by_sha256: dict[str, dict[str, object]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict) or set(entry) != {
            "clip_id",
            "clip_sha256",
            "source_sha256",
            "model_sha256",
            "terrain_sha256",
            "status",
            "accepted_intervals",
            "reasons",
            "relative_path",
            "sha256",
        }:
            raise ContractError("mechanical audit report index fields do not match v1")
        clip_sha256 = str(entry["clip_sha256"])
        expected_path = f"audits/{clip_sha256}.json"
        relative_path = _canonical_relative_path(
            entry["relative_path"],
            "audits",
        )
        if relative_path != expected_path:
            raise ContractError("mechanical audit report path is not canonical")
        report_path = root / relative_path
        if report_path.is_symlink() or not report_path.is_file():
            raise ContractError("mechanical audit report is missing or a symlink")
        payload = report_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ContractError("mechanical audit report hash mismatch")
        try:
            report = ClipAudit.from_dict(
                json.loads(payload.decode("ascii"))
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("invalid mechanical audit report JSON") from error
        if _audit_entry(
            report,
            relative_path,
            str(entry["sha256"]),
        ) != entry:
            raise ContractError("mechanical audit report evidence is stale")
        if clip_sha256 in reports:
            raise ContractError("mechanical audit report clip hash is duplicated")
        reports[clip_sha256] = report
        entries_by_sha256[clip_sha256] = entry
    expected_sha256 = {record.sha256 for record in manifest.clips}
    if set(reports) != expected_sha256 or len(reports) != len(manifest.clips):
        raise ContractError("mechanical audit reports must exactly cover corpus clips")
    for record, clip in zip(manifest.clips, clips, strict=True):
        report = reports[record.sha256]
        if (
            clip.clip_id != record.clip_id
            or clip.frame_count != record.frame_count
            or report.clip_id != clip.clip_id
            or report.frame_count != clip.frame_count
            or report.clip_sha256 != record.sha256
            or report.source_sha256 != clip.source.source_sha256
        ):
            raise ContractError("mechanical audit and canonical clip mismatch")
        if report.model_sha256 != model_record["structural_sha256"]:
            raise ContractError(
                "mechanical audit model authority mismatch"
            )
        if clip.terrain is None:
            expected_terrain_sha256 = "0" * 64
            if (
                report.status != "rejected"
                or report.accepted_intervals
                or not any(
                    reason.code == "terrain_registration"
                    for reason in report.reasons
                )
            ):
                raise ContractError(
                    "unbound clip audit must be a terrain-registration rejection"
                )
        else:
            mesh = mesh_by_sha256.get(clip.terrain.mesh_sha256)
            if (
                mesh is None
                or mesh.source_asset_sha256
                != clip.terrain.asset_sha256
            ):
                raise ContractError(
                    "canonical clip terrain binding does not match exact mesh"
                )
            expected_terrain_sha256 = terrain_query_sha256(
                CanonicalMeshQuery(
                    mesh,
                    clip.terrain.world_from_terrain,
                )
            )
        if report.terrain_sha256 != expected_terrain_sha256:
            raise ContractError(
                "mechanical audit terrain query authority mismatch"
            )
    return manifest, clips, index, reports


def _normalized_renderer_argument(value: str) -> str:
    candidate = Path(value).expanduser()
    return str(candidate.resolve()) if candidate.is_file() else value


def _accepted_interval_key(
    clip_sha256: str,
    start_frame: int,
    end_frame: int,
) -> str:
    return f"{clip_sha256}:{start_frame}:{end_frame}"


def _action_class(clip: CanonicalClip) -> str:
    tags = set(clip.action_tags)
    aliases = {
        "up": "ascent",
        "down": "descent",
        "sprint": "run",
    }
    for alias, action_class in aliases.items():
        if alias in tags:
            return action_class
    for action_class in coverage.ACTION_CLASSES:
        if action_class != "other" and action_class in tags:
            return action_class
    return "other"


def _terrain_class(clip: CanonicalClip) -> str:
    tags = set(clip.action_tags)
    for terrain in ("stairs", "slope", "curb", "flat"):
        if terrain in tags:
            return terrain
    return "other"


def _direction_class(
    clip: CanonicalClip,
    start_frame: int,
    end_frame: int,
) -> str:
    velocity = np.mean(
        clip.commands.inferred_velocity_local_xy[start_frame:end_frame],
        axis=0,
        dtype=np.float64,
    )
    if float(np.linalg.norm(velocity)) < 0.1:
        return "stationary"
    if abs(float(velocity[0])) >= abs(float(velocity[1])):
        return "forward" if velocity[0] >= 0.0 else "backward"
    return "left" if velocity[1] >= 0.0 else "right"


def _render_input(
    record: storage.ClipRecord,
    clip: CanonicalClip,
    report: ClipAudit,
    interval: object,
    model_record: object,
) -> dict[str, object]:
    if clip.terrain is None:
        raise ContractError(
            f"accepted clip {clip.clip_id} has no terrain binding"
        )
    start_frame = int(interval.start_frame)
    end_frame = int(interval.end_frame)
    return {
        "interval_key": _accepted_interval_key(
            record.sha256,
            start_frame,
            end_frame,
        ),
        "clip_id": clip.clip_id,
        "clip_sha256": record.sha256,
        "source_sha256": clip.source.source_sha256,
        "model": model_record,
        "terrain_sha256": report.terrain_sha256,
        "terrain_mesh_sha256": clip.terrain.mesh_sha256,
        "interval": [start_frame, end_frame],
    }


def _trusted_media_model(model_record: object) -> dict[str, object]:
    normalized = _normalized_model_record(
        model_record,
        "trusted renderer model",
    )
    path = _cli_input_file(
        Path(str(normalized["asset_path"])),
        "pinned exact G1 model",
    )
    if (
        str(path) != normalized["asset_path"]
        or path.name != _PHASE1_G1_ROOT_NAME
        or normalized["asset_sha256"] != _PHASE1_G1_ROOT_SHA256
    ):
        raise ContractError(
            "trusted rendering requires the pinned exact G1 model authority"
        )
    dependency_vfs = dict(
        _pinned_g1_dependency_vfs(str(path.parent))
    )
    return {
        "path": str(path),
        "size_bytes": normalized["asset_size_bytes"],
        "sha256": normalized["asset_sha256"],
        "structural_sha256": normalized["structural_sha256"],
        "dependency_vfs": dependency_vfs,
    }


@lru_cache(maxsize=4)
def _pinned_g1_dependency_vfs(root_value: str) -> dict[str, object]:
    root = Path(root_value)
    directories, files, tree_sha256 = storage._directory_tree_evidence(
        root,
        (),
    )
    size_bytes = sum(int(record["size_bytes"]) for record in files)
    root_records = [
        record
        for record in files
        if record["relative_path"] == _PHASE1_G1_ROOT_NAME
    ]
    if (
        len(directories) != _PHASE1_G1_VFS_DIRECTORY_COUNT
        or len(files) != _PHASE1_G1_VFS_FILE_COUNT
        or size_bytes != _PHASE1_G1_VFS_SIZE_BYTES
        or tree_sha256 != _PHASE1_G1_VFS_TREE_SHA256
        or len(root_records) != 1
        or root_records[0]["sha256"] != _PHASE1_G1_ROOT_SHA256
    ):
        raise ContractError(
            "trusted rendering requires the pinned exact G1 dependency VFS"
        )
    return {
        "root_path": str(root),
        "root_relative_path": _PHASE1_G1_ROOT_NAME,
        "directory_count": len(directories),
        "file_count": len(files),
        "size_bytes": size_bytes,
        "tree_sha256": tree_sha256,
    }


def _trusted_media_input(
    root: Path,
    record: storage.ClipRecord,
    clip: CanonicalClip,
    report: ClipAudit,
    interval: object,
    mesh_record: storage.MeshRecord,
    mesh: CanonicalTerrainMesh,
    *,
    request_root: Path | None = None,
) -> dict[str, object]:
    if clip.terrain is None:
        raise ContractError(
            f"accepted clip {clip.clip_id} has no terrain binding"
        )
    clip_path = (root / record.relative_path).resolve()
    mesh_path = (root / mesh_record.relative_path).resolve()
    authority_root = root if request_root is None else request_root
    if not authority_root.is_absolute():
        raise ContractError("trusted render request root must be absolute")
    request_clip_path = authority_root / record.relative_path
    request_mesh_path = authority_root / mesh_record.relative_path
    if (
        clip_path.is_symlink()
        or mesh_path.is_symlink()
        or not clip_path.is_file()
        or not mesh_path.is_file()
        or clip_path.stem != record.sha256
        or mesh_path.stem != mesh_record.sha256
    ):
        raise ContractError("trusted render input artifacts are not canonical")
    start_frame = int(interval.start_frame)
    end_frame = int(interval.end_frame)
    interval_key = _accepted_interval_key(
        record.sha256,
        start_frame,
        end_frame,
    )
    return {
        "interval_key": interval_key,
        "interval": [start_frame, end_frame],
        "clip": {
            "path": str(request_clip_path),
            "size_bytes": clip_path.stat(follow_symlinks=False).st_size,
            "sha256": record.sha256,
            "clip_id": clip.clip_id,
            "source_sha256": clip.source.source_sha256,
            "frame_count": clip.frame_count,
        },
        "terrain_mesh": {
            "path": str(request_mesh_path),
            "size_bytes": mesh_path.stat(follow_symlinks=False).st_size,
            "sha256": mesh_record.sha256,
            "source_asset_sha256": mesh.source_asset_sha256,
        },
        "terrain_query": {
            "mesh_sha256": mesh_record.sha256,
            "query_sha256": report.terrain_sha256,
            "world_from_terrain": {
                "translation_world": (
                    clip.terrain.world_from_terrain.translation_world.tolist()
                ),
                "quaternion_world_from_local_wxyz": (
                    clip.terrain.world_from_terrain
                    .quaternion_world_from_local_wxyz.tolist()
                ),
            },
        },
    }


def _trusted_media_request(
    kind: str,
    inputs: Sequence[dict[str, object]],
    model_record: object,
) -> dict[str, object]:
    from .render_media import FIXED_RENDER_CONFIG

    ordered = sorted(
        inputs,
        key=lambda item: str(item["interval_key"]),
    )
    if not ordered:
        raise ContractError("trusted render request cannot be empty")
    return {
        "schema": "terrain-oracle-render-request/v1",
        "kind": kind,
        "interval_keys": [
            str(item["interval_key"]) for item in ordered
        ],
        "inputs": ordered,
        "model": _trusted_media_model(model_record),
        "render_config": FIXED_RENDER_CONFIG,
    }


def _artifact_record(path: Path, relative_path: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ContractError("renderer did not produce a regular artifact file")
    payload = path.read_bytes()
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validated_trusted_renderer_result(
    *,
    stdout: bytes,
    stderr: bytes,
    request: dict[str, object],
    request_payload: bytes,
    video_path: Path,
    overlay_path: Path,
    video_relative: str,
    overlay_relative: str,
) -> dict[str, object]:
    from . import render_media

    try:
        result = json.loads(stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(
            "trusted renderer did not emit canonical result JSON"
        ) from error
    if (
        stderr
        or not isinstance(result, dict)
        or stdout
        != storage._canonical_json_bytes(
            result,
            "trusted renderer result",
        )
        or set(result)
        != {
            "schema",
            "request_sha256",
            "kind",
            "interval_keys",
            "video",
            "contact_overlay",
            "runtime_identity",
            "render_evidence",
            "completed",
        }
        or result["schema"] != "terrain-oracle-render-result/v1"
        or result["request_sha256"]
        != hashlib.sha256(request_payload).hexdigest()
        or result["kind"] != request["kind"]
        or result["interval_keys"] != request["interval_keys"]
        or result["completed"] is not True
    ):
        raise ContractError("trusted renderer result authority is stale")
    frame_count = render_media._encoded_frame_count(
        str(request["kind"]),
        tuple(
            {
                "start": int(item["interval"][0]),
                "end": int(item["interval"][1]),
            }
            for item in request["inputs"]
        ),
    )
    video_metadata = render_media.validate_video(
        video_path,
        width=int(render_media.FIXED_RENDER_CONFIG["width"]),
        height=int(render_media.FIXED_RENDER_CONFIG["height"]),
        fps=int(render_media.FIXED_RENDER_CONFIG["fps"]),
        frame_count=frame_count,
    )
    overlay_metadata = render_media.validate_png(
        overlay_path,
        width=int(render_media.FIXED_RENDER_CONFIG["width"]),
        height=int(render_media.FIXED_RENDER_CONFIG["height"]),
        interval_keys=request["interval_keys"],
    )

    def normalized_artifact(
        value: object,
        path: Path,
        relative_path: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "size_bytes",
            "sha256",
            "validated_metadata",
        }:
            raise ContractError(
                "trusted renderer artifact result fields do not match v1"
            )
        payload = storage._read_regular_file_nofollow(path)
        if (
            value["path"] != str(path)
            or value["size_bytes"] != len(payload)
            or value["sha256"] != hashlib.sha256(payload).hexdigest()
            or value["validated_metadata"] != metadata
        ):
            raise ContractError(
                "trusted renderer artifact result is stale"
            )
        return {
            "relative_path": relative_path,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "validated_metadata": metadata,
        }

    if result["runtime_identity"] != render_media.runtime_identity():
        raise ContractError("trusted renderer runtime identity is stale")
    _validate_trusted_render_evidence(
        result["render_evidence"],
        request,
    )
    return {
        **result,
        "video": normalized_artifact(
            result["video"],
            video_path,
            video_relative,
            video_metadata,
        ),
        "contact_overlay": normalized_artifact(
            result["contact_overlay"],
            overlay_path,
            overlay_relative,
            overlay_metadata,
        ),
    }


def _validate_trusted_render_evidence(
    value: object,
    request: dict[str, object],
) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "visual_mesh_geom_count",
            "terrain_face_count",
            "model_dependency_vfs",
        }
        or any(
            type(value[name]) is not int or value[name] < 1
            for name in ("visual_mesh_geom_count", "terrain_face_count")
        )
    ):
        raise ContractError("trusted renderer visual evidence is incomplete")
    model = request.get("model")
    if (
        not isinstance(model, dict)
        or value["model_dependency_vfs"] != model.get("dependency_vfs")
    ):
        raise ContractError(
            "trusted renderer model dependency VFS evidence is stale"
        )


def _run_renderer(
    *,
    stage: Path,
    renderer: Path,
    renderer_arguments: tuple[str, ...],
    renderer_sha256: str,
    request: dict[str, object],
    semantic_arguments: list[str],
    timeout_seconds: float,
    trusted_request: dict[str, object] | None = None,
) -> tuple[dict[str, object], str]:
    request_id = hashlib.sha256(
        storage._canonical_json_bytes(request, "render request")
    ).hexdigest()
    video_relative = f"artifacts/{request_id}.mp4"
    overlay_relative = f"artifacts/{request_id}.contact.png"
    request_record: dict[str, object] | None = None
    request_payload: bytes | None = None
    if trusted_request is None:
        normalized_argv = [
            str(renderer),
            *renderer_arguments,
            *semantic_arguments,
            "--video",
            video_relative,
            "--overlay",
            overlay_relative,
        ]
        actual_argv = [
            *normalized_argv[:-3],
            str(stage / video_relative),
            "--overlay",
            str(stage / overlay_relative),
        ]
    else:
        request_relative = f"requests/{request_id}.json"
        request_payload = storage._canonical_json_bytes(
            trusted_request,
            "trusted render request",
        )
        storage._write_no_replace(
            stage / request_relative,
            request_payload,
        )
        request_record = {
            "relative_path": request_relative,
            "sha256": hashlib.sha256(request_payload).hexdigest(),
            "size_bytes": len(request_payload),
        }
        normalized_argv = [
            str(renderer),
            *renderer_arguments,
            "--request",
            request_relative,
            "--video",
            video_relative,
            "--overlay",
            overlay_relative,
        ]
        actual_argv = [
            str(renderer),
            *renderer_arguments,
            "--request",
            str(stage / request_relative),
            "--video",
            str(stage / video_relative),
            "--overlay",
            str(stage / overlay_relative),
        ]
    stdout_capture = (
        tempfile.TemporaryFile()
        if trusted_request is not None
        else None
    )
    stderr_capture = (
        tempfile.TemporaryFile()
        if trusted_request is not None
        else None
    )
    try:
        process = subprocess.Popen(
            actual_argv,
            shell=False,
            stdout=(
                stdout_capture
                if stdout_capture is not None
                else subprocess.DEVNULL
            ),
            stderr=(
                stderr_capture
                if stderr_capture is not None
                else subprocess.DEVNULL
            ),
            start_new_session=True,
        )
    except BaseException:
        if stdout_capture is not None:
            stdout_capture.close()
        if stderr_capture is not None:
            stderr_capture.close()
        raise

    def kill_process_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        kill_process_group()
        process.wait()
        if stdout_capture is not None:
            stdout_capture.close()
        if stderr_capture is not None:
            stderr_capture.close()
        raise ContractError(
            f"renderer timed out after {timeout_seconds:g} seconds"
        ) from error
    except BaseException:
        if process.poll() is None:
            kill_process_group()
            process.wait()
        if stdout_capture is not None:
            stdout_capture.close()
        if stderr_capture is not None:
            stderr_capture.close()
        raise
    kill_process_group()
    capture_limit = 1024 * 1024
    stdout = b""
    stderr = b""
    if stdout_capture is not None and stderr_capture is not None:
        stdout_capture.seek(0)
        stderr_capture.seek(0)
        stdout = stdout_capture.read(capture_limit + 1)
        stderr = stderr_capture.read(capture_limit + 1)
        stdout_capture.close()
        stderr_capture.close()
        if len(stdout) > capture_limit or len(stderr) > capture_limit:
            raise ContractError(
                "trusted renderer output exceeded the 1 MiB evidence cap"
            )
    if returncode != 0:
        raise ContractError(
            f"renderer failed with exit {returncode}"
        )
    renderer_record = {
        "argv": normalized_argv,
        "executable_sha256": renderer_sha256,
        "invocation_sha256": hashlib.sha256(
            storage._canonical_json_bytes(
                normalized_argv,
                "normalized renderer argv",
            )
        ).hexdigest(),
        "shell": False,
        "returncode": returncode,
    }
    artifacts = {
        "video": _artifact_record(
            stage / video_relative,
            video_relative,
        ),
        "contact_overlay": _artifact_record(
            stage / overlay_relative,
            overlay_relative,
        ),
    }
    if trusted_request is not None:
        assert request_payload is not None
        assert request_record is not None
        renderer_record["request"] = request_record
        renderer_record["result"] = _validated_trusted_renderer_result(
            stdout=stdout,
            stderr=stderr,
            request=trusted_request,
            request_payload=request_payload,
            video_path=stage / video_relative,
            overlay_path=stage / overlay_relative,
            video_relative=video_relative,
            overlay_relative=overlay_relative,
        )
    return {**renderer_record, **artifacts}, request_id


def _write_render_receipt(
    stage: Path,
    relative_path: str,
    receipt: dict[str, object],
) -> dict[str, str]:
    payload = storage._canonical_json_bytes(
        receipt,
        "render audit receipt",
    )
    storage._write_no_replace(stage / relative_path, payload)
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _renderer_receipt_evidence(
    rendered: dict[str, object],
) -> dict[str, object]:
    fields = (
        "argv",
        "executable_sha256",
        "invocation_sha256",
        "shell",
        "returncode",
    )
    evidence = {name: rendered[name] for name in fields}
    if "request" in rendered or "result" in rendered:
        if "request" not in rendered or "result" not in rendered:
            raise ContractError(
                "trusted renderer request and result evidence must be paired"
            )
        evidence["request"] = rendered["request"]
        evidence["result"] = rendered["result"]
    return evidence


def _verify_render_stage(
    stage: Path,
    expected_index: dict[str, object],
    expected_receipts: dict[str, dict[str, object]],
) -> None:
    loaded_index = json.loads(
        (stage / "render-index.json").read_text("ascii")
    )
    if loaded_index != expected_index:
        raise ContractError("reloaded render index does not match publication")
    actual_receipt_paths = {
        path.relative_to(stage).as_posix()
        for directory in ("receipts", "contact-sheet", "strata")
        for path in (stage / directory).glob("*.json")
    }
    if actual_receipt_paths != set(expected_receipts):
        raise ContractError("render receipts do not form the exact required set")
    expected_artifacts: set[str] = set()
    expected_requests: set[str] = set()
    for relative_path, expected_receipt in expected_receipts.items():
        path = stage / relative_path
        if path.is_symlink() or not path.is_file():
            raise ContractError("render receipt is missing or a symlink")
        loaded_receipt = json.loads(path.read_text("ascii"))
        if loaded_receipt != expected_receipt:
            raise ContractError("reloaded render receipt does not match publication")
        for field in ("video", "contact_overlay"):
            artifact = loaded_receipt[field]
            relative_artifact = _canonical_relative_path(
                artifact["relative_path"],
                "artifacts",
            )
            expected_artifacts.add(relative_artifact)
            artifact_path = stage / relative_artifact
            if artifact_path.is_symlink() or not artifact_path.is_file():
                raise ContractError("render artifact is missing or a symlink")
            payload = artifact_path.read_bytes()
            if (
                len(payload) != artifact["size_bytes"]
                or hashlib.sha256(payload).hexdigest()
                != artifact["sha256"]
            ):
                raise ContractError("render artifact failed reload verification")
        renderer = loaded_receipt["renderer"]
        if not isinstance(renderer, dict) or "request" not in renderer:
            raise ContractError(
                "render receipt lacks trusted request evidence"
            )
        request = renderer["request"]
        if not isinstance(request, dict) or set(request) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ContractError(
                "trusted request record fields do not match v1"
            )
        request_relative = _canonical_relative_path(
            request["relative_path"],
            "requests",
        )
        expected_requests.add(request_relative)
        request_payload = storage._read_regular_file_nofollow(
            stage / request_relative
        )
        if (
            request["size_bytes"] != len(request_payload)
            or request["sha256"]
            != hashlib.sha256(request_payload).hexdigest()
        ):
            raise ContractError("trusted request failed reload verification")
    actual_artifacts = {
        path.relative_to(stage).as_posix()
        for path in (stage / "artifacts").iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual_artifacts != expected_artifacts:
        raise ContractError("render artifacts do not form the exact required set")
    actual_requests = {
        path.relative_to(stage).as_posix()
        for path in (stage / "requests").iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual_requests != expected_requests:
        raise ContractError("trusted requests do not form the exact required set")


def _expected_render_inputs(
    manifest: storage.CorpusManifest,
    clips: tuple[CanonicalClip, ...],
    audit_index: dict[str, object],
    reports: dict[str, ClipAudit],
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, str, str, str], list[dict[str, object]]],
]:
    accepted_inputs: list[dict[str, object]] = []
    strata: dict[
        tuple[str, str, str, str],
        list[dict[str, object]],
    ] = {}
    for record, clip in zip(manifest.clips, clips, strict=True):
        report = reports[record.sha256]
        for interval in report.accepted_intervals:
            input_record = _render_input(
                record,
                clip,
                report,
                interval,
                audit_index["model"],
            )
            accepted_inputs.append(input_record)
            key = (
                clip.source.source_format,
                _action_class(clip),
                _terrain_class(clip),
                _direction_class(
                    clip,
                    interval.start_frame,
                    interval.end_frame,
                ),
            )
            strata.setdefault(key, []).append(input_record)
    accepted_inputs.sort(key=lambda item: str(item["interval_key"]))
    for inputs in strata.values():
        inputs.sort(key=lambda item: str(item["interval_key"]))
    return accepted_inputs, strata


def _load_canonical_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing or a symlink")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or payload != storage._canonical_json_bytes(
        value,
        label,
    ):
        raise ContractError(f"{label} is not canonical JSON")
    return value, payload


def _render_request_id(request: dict[str, object]) -> str:
    return hashlib.sha256(
        storage._canonical_json_bytes(request, "render request")
    ).hexdigest()


def _validate_render_artifact(
    render_root: Path,
    value: object,
    expected_relative_path: str,
) -> str:
    if not isinstance(value, dict) or set(value) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ContractError("render artifact fields do not match v1")
    relative_path = _canonical_relative_path(
        value["relative_path"],
        "artifacts",
    )
    if relative_path != expected_relative_path:
        raise ContractError("render artifact path does not match its request")
    path = render_root / relative_path
    if path.is_symlink() or not path.is_file():
        raise ContractError("render artifact is missing or a symlink")
    payload = path.read_bytes()
    if (
        type(value["size_bytes"]) is not int
        or value["size_bytes"] < 0
        or len(payload) != value["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != value["sha256"]
    ):
        raise ContractError("render artifact size or hash mismatch")
    return relative_path


def _validate_renderer_evidence(
    value: object,
    expected_argv: list[str],
    executable_sha256: str,
    render_root: Path,
    expected_request: dict[str, object],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "argv",
        "executable_sha256",
        "invocation_sha256",
        "shell",
        "returncode",
        "request",
        "result",
    }:
        raise ContractError("trusted renderer evidence fields do not match v2")
    if (
        value["argv"] != expected_argv
        or value["executable_sha256"] != executable_sha256
        or value["invocation_sha256"]
        != hashlib.sha256(
            storage._canonical_json_bytes(
                expected_argv,
                "normalized renderer argv",
            )
        ).hexdigest()
        or value["shell"] is not False
        or type(value["returncode"]) is not int
        or value["returncode"] != 0
    ):
        raise ContractError("renderer invocation evidence is stale")
    request_record = value["request"]
    if not isinstance(request_record, dict) or set(request_record) != {
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ContractError("trusted request record fields do not match v1")
    expected_payload = storage._canonical_json_bytes(
        expected_request,
        "trusted render request",
    )
    request_relative = _canonical_relative_path(
        request_record["relative_path"],
        "requests",
    )
    request_payload = storage._read_regular_file_nofollow(
        render_root / request_relative
    )
    if (
        request_payload != expected_payload
        or request_record["size_bytes"] != len(request_payload)
        or request_record["sha256"]
        != hashlib.sha256(request_payload).hexdigest()
    ):
        raise ContractError("trusted render request authority is stale")
    result = value["result"]
    if not isinstance(result, dict) or set(result) != {
        "schema",
        "request_sha256",
        "kind",
        "interval_keys",
        "video",
        "contact_overlay",
        "runtime_identity",
        "render_evidence",
        "completed",
    }:
        raise ContractError("trusted renderer result fields do not match v1")
    if (
        result["schema"] != "terrain-oracle-render-result/v1"
        or result["request_sha256"]
        != hashlib.sha256(expected_payload).hexdigest()
        or result["kind"] != expected_request["kind"]
        or result["interval_keys"] != expected_request["interval_keys"]
        or result["completed"] is not True
    ):
        raise ContractError("trusted renderer result binding is stale")

    from . import render_media

    if result["runtime_identity"] != render_media.runtime_identity():
        raise ContractError("trusted renderer runtime identity is stale")
    _validate_trusted_render_evidence(
        result["render_evidence"],
        expected_request,
    )
    frame_count = render_media._encoded_frame_count(
        str(expected_request["kind"]),
        tuple(
            {
                "start": int(item["interval"][0]),
                "end": int(item["interval"][1]),
            }
            for item in expected_request["inputs"]
        ),
    )
    result_artifacts: dict[str, tuple[dict[str, object], Path]] = {}
    for name, suffix in (
        ("video", ".mp4"),
        ("contact_overlay", ".contact.png"),
    ):
        artifact = result[name]
        if not isinstance(artifact, dict) or set(artifact) != {
            "relative_path",
            "size_bytes",
            "sha256",
            "validated_metadata",
        }:
            raise ContractError(
                "trusted result artifact fields do not match v1"
            )
        relative = _canonical_relative_path(
            artifact["relative_path"],
            "artifacts",
        )
        if not relative.endswith(suffix):
            raise ContractError("trusted result artifact suffix is stale")
        result_artifacts[name] = (
            artifact,
            render_root / relative,
        )
    expected_artifacts = (
        (
            "video",
            render_media.validate_video(
                result_artifacts["video"][1],
                width=int(render_media.FIXED_RENDER_CONFIG["width"]),
                height=int(render_media.FIXED_RENDER_CONFIG["height"]),
                fps=int(render_media.FIXED_RENDER_CONFIG["fps"]),
                frame_count=frame_count,
            ),
        ),
        (
            "contact_overlay",
            render_media.validate_png(
                result_artifacts["contact_overlay"][1],
                width=int(render_media.FIXED_RENDER_CONFIG["width"]),
                height=int(render_media.FIXED_RENDER_CONFIG["height"]),
                interval_keys=expected_request["interval_keys"],
            ),
        ),
    )
    for name, metadata in expected_artifacts:
        artifact, artifact_path = result_artifacts[name]
        payload = storage._read_regular_file_nofollow(
            artifact_path
        )
        if (
            artifact["size_bytes"] != len(payload)
            or artifact["sha256"] != hashlib.sha256(payload).hexdigest()
            or artifact["validated_metadata"] != metadata
        ):
            raise ContractError("trusted result artifact authority is stale")


def _load_render_receipt(
    render_root: Path,
    *,
    relative_path: str,
    expected_fields: set[str],
    expected_kind: str,
    expected_argv: list[str],
    expected_request_id: str,
    executable_sha256: str,
    expected_request: dict[str, object],
) -> tuple[dict[str, object], dict[str, str], set[str]]:
    receipt, payload = _load_canonical_json(
        render_root / relative_path,
        "render receipt",
    )
    if (
        set(receipt) != expected_fields
        or receipt["schema"] != "terrain-oracle-render-receipt/v1"
        or receipt["kind"] != expected_kind
        or receipt["completed"] is not True
    ):
        raise ContractError("render receipt fields do not match v1")
    _validate_renderer_evidence(
        receipt["renderer"],
        expected_argv,
        executable_sha256,
        render_root,
        expected_request,
    )
    artifacts = {
        _validate_render_artifact(
            render_root,
            receipt["video"],
            f"artifacts/{expected_request_id}.mp4",
        ),
        _validate_render_artifact(
            render_root,
            receipt["contact_overlay"],
            f"artifacts/{expected_request_id}.contact.png",
        ),
    }
    return (
        receipt,
        {
            "relative_path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        artifacts,
    )


def _load_render_evidence(
    root: Path,
    *,
    model_snapshot: _FreezeModelSnapshot | None = None,
    request_root: Path | None = None,
) -> dict[str, object]:
    """Reload and recompute the exact exhaustive render evidence set."""

    corpus_root = Path(root).expanduser().resolve()
    if not corpus_root.is_dir() or corpus_root.is_symlink():
        raise ContractError("audited corpus is missing or a symlink")
    manifest, clips, audit_index, reports = _load_audit_evidence(
        corpus_root,
        model_snapshot=model_snapshot,
    )
    meshes = tuple(
        storage.read_mesh(corpus_root / record.relative_path)
        for record in manifest.meshes
    )
    mesh_records_by_sha256 = {
        record.sha256: record for record in manifest.meshes
    }
    meshes_by_sha256 = {
        record.sha256: mesh
        for record, mesh in zip(manifest.meshes, meshes, strict=True)
    }
    trusted_inputs_by_key: dict[str, dict[str, object]] = {}
    for record, clip in zip(manifest.clips, clips, strict=True):
        report = reports[record.sha256]
        for interval in report.accepted_intervals:
            if clip.terrain is None:
                raise ContractError(
                    f"accepted clip {clip.clip_id} has no terrain binding"
                )
            mesh_record = mesh_records_by_sha256.get(
                clip.terrain.mesh_sha256
            )
            mesh = meshes_by_sha256.get(clip.terrain.mesh_sha256)
            if mesh_record is None or mesh is None:
                raise ContractError(
                    f"accepted clip {clip.clip_id} mesh is unavailable"
                )
            trusted_input = _trusted_media_input(
                corpus_root,
                record,
                clip,
                report,
                interval,
                mesh_record,
                mesh,
                request_root=request_root,
            )
            trusted_inputs_by_key[
                str(trusted_input["interval_key"])
            ] = trusted_input
    render_root = corpus_root / "render-audit"
    if not render_root.is_dir() or render_root.is_symlink():
        raise ContractError("canonical render-audit directory is missing")
    storage._validate_complete_directory(render_root)
    allowed_directories = {
        "artifacts",
        "receipts",
        "contact-sheet",
        "strata",
        "requests",
    }
    for directory in allowed_directories:
        path = render_root / directory
        if not path.is_dir() or path.is_symlink():
            raise ContractError(
                f"render-audit {directory} directory is missing or a symlink"
            )
    index, _ = _load_canonical_json(
        render_root / "render-index.json",
        "render audit index",
    )
    if set(index) != {
        "schema",
        "corpus_manifest_sha256",
        "audit_index_sha256",
        "accepted_interval_keys",
        "renderer",
        "interval_receipts",
        "contact_sheet_receipt",
        "stratum_receipts",
        "content_sha256",
    }:
        raise ContractError("render audit index fields do not match v1")
    without_hash = dict(index)
    content_sha256 = without_hash.pop("content_sha256")
    if (
        index["schema"] != "terrain-oracle-render-index/v2"
        or content_sha256
        != hashlib.sha256(
            storage._canonical_json_bytes(
                without_hash,
                "render audit index",
            )
        ).hexdigest()
        or index["corpus_manifest_sha256"]
        != hashlib.sha256(
            (corpus_root / "manifest.json").read_bytes()
        ).hexdigest()
        or index["audit_index_sha256"]
        != hashlib.sha256(
            (corpus_root / "audit-index.json").read_bytes()
        ).hexdigest()
    ):
        raise ContractError("render audit index hash or corpus binding is stale")
    renderer_record = index["renderer"]
    if not isinstance(renderer_record, dict) or set(renderer_record) != {
        "argv_prefix",
        "executable_sha256",
        "mode",
        "module_sha256",
    }:
        raise ContractError("render index renderer fields do not match v2")
    argv_prefix = renderer_record["argv_prefix"]
    if (
        not isinstance(argv_prefix, list)
        or not argv_prefix
        or any(type(value) is not str or not value for value in argv_prefix)
    ):
        raise ContractError("render argv prefix must be a nonempty string list")
    expected_prefix = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "mm_sonic.terrain_oracle.render_media",
    ]
    executable = Path(argv_prefix[0])
    from . import render_media

    if (
        argv_prefix != expected_prefix
        or renderer_record["mode"] != "trusted-package-media-v1"
        or not executable.is_absolute()
        or executable != executable.resolve()
        or not executable.is_file()
        or executable.is_symlink()
        or hashlib.sha256(executable.read_bytes()).hexdigest()
        != renderer_record["executable_sha256"]
        or hashlib.sha256(Path(render_media.__file__).read_bytes()).hexdigest()
        != renderer_record["module_sha256"]
    ):
        raise ContractError("render executable identity is stale")
    accepted_inputs, strata = _expected_render_inputs(
        manifest,
        clips,
        audit_index,
        reports,
    )
    accepted_keys = [
        str(item["interval_key"]) for item in accepted_inputs
    ]
    if index["accepted_interval_keys"] != accepted_keys:
        raise ContractError(
            "render index accepted intervals do not match mechanical audit"
        )
    expected_receipt_paths: set[str] = set()
    expected_artifact_paths: set[str] = set()
    expected_request_paths: set[str] = set()
    interval_references: list[dict[str, str]] = []
    input_by_key = {
        str(item["interval_key"]): item for item in accepted_inputs
    }
    for interval_key in accepted_keys:
        input_record = input_by_key[interval_key]
        clip_sha256 = str(input_record["clip_sha256"])
        start_frame, end_frame = input_record["interval"]
        request = {
            "kind": "accepted_interval",
            "clip_sha256": clip_sha256,
            "start": start_frame,
            "end": end_frame,
        }
        request_id = _render_request_id(request)
        relative_path = f"receipts/{request_id}.json"
        trusted_request = _trusted_media_request(
            "accepted_interval",
            (trusted_inputs_by_key[interval_key],),
            audit_index["model"],
        )
        expected_argv = [
            *argv_prefix,
            "--request",
            f"requests/{request_id}.json",
            "--video",
            f"artifacts/{request_id}.mp4",
            "--overlay",
            f"artifacts/{request_id}.contact.png",
        ]
        receipt, reference, artifacts = _load_render_receipt(
            render_root,
            relative_path=relative_path,
            expected_fields={
                "schema",
                "kind",
                "clip_id",
                "clip_sha256",
                "model",
                "terrain_sha256",
                "terrain_mesh_sha256",
                "interval",
                "renderer",
                "video",
                "contact_overlay",
                "completed",
            },
            expected_kind="accepted_interval",
            expected_argv=expected_argv,
            expected_request_id=request_id,
            executable_sha256=str(renderer_record["executable_sha256"]),
            expected_request=trusted_request,
        )
        if {
            "clip_id": receipt["clip_id"],
            "clip_sha256": receipt["clip_sha256"],
            "model": receipt["model"],
            "terrain_sha256": receipt["terrain_sha256"],
            "terrain_mesh_sha256": receipt["terrain_mesh_sha256"],
            "interval": receipt["interval"],
        } != {
            name: input_record[name]
            for name in (
                "clip_id",
                "clip_sha256",
                "model",
                "terrain_sha256",
                "terrain_mesh_sha256",
                "interval",
            )
        }:
            raise ContractError("interval render receipt binding is stale")
        interval_references.append(reference)
        expected_receipt_paths.add(relative_path)
        expected_artifact_paths.update(artifacts)
        expected_request_paths.add(f"requests/{request_id}.json")
    interval_references.sort(key=lambda item: item["relative_path"])
    if index["interval_receipts"] != interval_references:
        raise ContractError("interval render receipts are not the exact required set")
    sheet_request = {
        "kind": "contact_sheet",
        "interval_keys": accepted_keys,
    }
    sheet_id = _render_request_id(sheet_request)
    sheet_relative = f"contact-sheet/{sheet_id}.json"
    trusted_sheet_request = _trusted_media_request(
        "contact_sheet",
        tuple(trusted_inputs_by_key[key] for key in accepted_keys),
        audit_index["model"],
    )
    sheet_argv = [
        *argv_prefix,
        "--request",
        f"requests/{sheet_id}.json",
        "--video",
        f"artifacts/{sheet_id}.mp4",
        "--overlay",
        f"artifacts/{sheet_id}.contact.png",
    ]
    sheet, sheet_reference, artifacts = _load_render_receipt(
        render_root,
        relative_path=sheet_relative,
        expected_fields={
            "schema",
            "kind",
            "inputs",
            "interval_keys",
            "renderer",
            "video",
            "contact_overlay",
            "completed",
        },
        expected_kind="contact_sheet",
        expected_argv=sheet_argv,
        expected_request_id=sheet_id,
        executable_sha256=str(renderer_record["executable_sha256"]),
        expected_request=trusted_sheet_request,
    )
    if (
        sheet["inputs"] != accepted_inputs
        or sheet["interval_keys"] != accepted_keys
        or index["contact_sheet_receipt"] != sheet_reference
    ):
        raise ContractError("contact sheet does not cover every accepted interval")
    expected_receipt_paths.add(sheet_relative)
    expected_artifact_paths.update(artifacts)
    expected_request_paths.add(f"requests/{sheet_id}.json")
    stratum_references: list[dict[str, str]] = []
    for stratum_key in sorted(strata):
        stratum = {
            name: value
            for name, value in zip(
                ("source", "action_class", "terrain", "direction"),
                stratum_key,
                strict=True,
            )
        }
        inputs = strata[stratum_key]
        interval_keys = [
            str(item["interval_key"]) for item in inputs
        ]
        request = {
            "kind": "full_video",
            "stratum": stratum,
            "interval_keys": interval_keys,
        }
        request_id = _render_request_id(request)
        relative_path = f"strata/{request_id}.json"
        trusted_stratum_request = _trusted_media_request(
            "full_video",
            tuple(
                trusted_inputs_by_key[key] for key in interval_keys
            ),
            audit_index["model"],
        )
        expected_argv = [
            *argv_prefix,
            "--request",
            f"requests/{request_id}.json",
            "--video",
            f"artifacts/{request_id}.mp4",
            "--overlay",
            f"artifacts/{request_id}.contact.png",
        ]
        receipt, reference, artifacts = _load_render_receipt(
            render_root,
            relative_path=relative_path,
            expected_fields={
                "schema",
                "kind",
                "stratum",
                "inputs",
                "interval_keys",
                "renderer",
                "video",
                "contact_overlay",
                "completed",
            },
            expected_kind="full_video",
            expected_argv=expected_argv,
            expected_request_id=request_id,
            executable_sha256=str(renderer_record["executable_sha256"]),
            expected_request=trusted_stratum_request,
        )
        if (
            receipt["stratum"] != stratum
            or receipt["inputs"] != inputs
            or receipt["interval_keys"] != interval_keys
        ):
            raise ContractError(
                "full-video receipt stratum was not canonically recomputed"
            )
        stratum_references.append(reference)
        expected_receipt_paths.add(relative_path)
        expected_artifact_paths.update(artifacts)
        expected_request_paths.add(f"requests/{request_id}.json")
    stratum_references.sort(key=lambda item: item["relative_path"])
    if index["stratum_receipts"] != stratum_references:
        raise ContractError("full-video receipts are not the exact stratum set")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in render_root.rglob("*"):
        relative_path = path.relative_to(render_root).as_posix()
        if path.is_symlink():
            raise ContractError("render evidence must not contain symlinks")
        if path.is_dir():
            actual_directories.add(relative_path)
        elif path.is_file():
            actual_files.add(relative_path)
        else:
            raise ContractError("render evidence contains a non-regular entry")
    if actual_directories != allowed_directories:
        raise ContractError("render evidence directories do not match v1")
    if actual_files != {
        storage.COMPLETION_MARKER,
        "render-index.json",
        *expected_receipt_paths,
        *expected_artifact_paths,
        *expected_request_paths,
    }:
        raise ContractError("render evidence files do not form the exact set")
    return index


def _render_interval_receipts(arguments: argparse.Namespace) -> None:
    root = _cli_input_directory(arguments.corpus, "audited corpus")
    destination = _cli_output_path(
        arguments.output,
        "render-audit output",
    )
    if destination.parent.resolve() != root or destination.name in ("", ".", ".."):
        raise ContractError("render-audit output must be a direct corpus child")
    manifest, clips, audit_index, reports = _load_audit_evidence(root)
    clip_by_sha256 = {
        record.sha256: clip
        for record, clip in zip(manifest.clips, clips, strict=True)
    }
    meshes = tuple(
        storage.read_mesh(root / record.relative_path)
        for record in manifest.meshes
    )
    mesh_records_by_sha256 = {
        record.sha256: record for record in manifest.meshes
    }
    meshes_by_sha256 = {
        record.sha256: mesh
        for record, mesh in zip(manifest.meshes, meshes, strict=True)
    }
    trusted_renderer = arguments.renderer is None
    if trusted_renderer:
        renderer = _cli_input_file(
            Path(sys.executable).resolve(),
            "Python renderer executable",
        )
        renderer_arguments = (
            "-B",
            "-m",
            "mm_sonic.terrain_oracle.render_media",
        )
    else:
        renderer = _cli_input_file(
            arguments.renderer,
            "negative-test renderer executable",
        )
        renderer_arguments = tuple(
            _normalized_renderer_argument(value)
            for value in arguments.renderer_arg
        )
    renderer_sha256 = hashlib.sha256(renderer.read_bytes()).hexdigest()
    timeout_seconds = float(arguments.renderer_timeout_seconds)
    if (
        not np.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
        or timeout_seconds > 3600.0
    ):
        raise ContractError(
            "renderer timeout must be finite in (0, 3600] seconds"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        for directory in (
            "artifacts",
            "receipts",
            "contact-sheet",
            "strata",
            *(("requests",) if trusted_renderer else ()),
        ):
            (stage / directory).mkdir()
        accepted_inputs: list[dict[str, object]] = []
        strata: dict[
            tuple[str, str, str, str],
            list[dict[str, object]],
        ] = {}
        interval_references: list[dict[str, str]] = []
        expected_receipts: dict[str, dict[str, object]] = {}
        trusted_inputs_by_key: dict[str, dict[str, object]] = {}
        for record in manifest.clips:
            clip = clip_by_sha256[record.sha256]
            report = reports[record.sha256]
            if not report.accepted_intervals:
                continue
            if clip.terrain is None:
                raise ContractError(
                    f"accepted clip {clip.clip_id} has no terrain binding"
                )
            for interval in report.accepted_intervals:
                input_record = _render_input(
                    record,
                    clip,
                    report,
                    interval,
                    audit_index["model"],
                )
                accepted_inputs.append(input_record)
                mesh_record = mesh_records_by_sha256.get(
                    clip.terrain.mesh_sha256
                )
                mesh = meshes_by_sha256.get(
                    clip.terrain.mesh_sha256
                )
                if mesh_record is None or mesh is None:
                    raise ContractError(
                        f"accepted clip {clip.clip_id} mesh is unavailable"
                    )
                trusted_input = _trusted_media_input(
                    root,
                    record,
                    clip,
                    report,
                    interval,
                    mesh_record,
                    mesh,
                )
                trusted_inputs_by_key[
                    str(input_record["interval_key"])
                ] = trusted_input
                stratum_key = (
                    clip.source.source_format,
                    _action_class(clip),
                    _terrain_class(clip),
                    _direction_class(
                        clip,
                        interval.start_frame,
                        interval.end_frame,
                    ),
                )
                strata.setdefault(stratum_key, []).append(input_record)
                request = {
                    "kind": "accepted_interval",
                    "clip_sha256": record.sha256,
                    "start": interval.start_frame,
                    "end": interval.end_frame,
                }
                rendered, request_id = _run_renderer(
                    stage=stage,
                    renderer=renderer,
                    renderer_arguments=renderer_arguments,
                    renderer_sha256=renderer_sha256,
                    request=request,
                    semantic_arguments=[
                        "--kind",
                        "accepted_interval",
                        "--clip-sha256",
                        record.sha256,
                        "--start",
                        str(interval.start_frame),
                        "--end",
                        str(interval.end_frame),
                    ],
                    timeout_seconds=timeout_seconds,
                    trusted_request=(
                        _trusted_media_request(
                            "accepted_interval",
                            (trusted_input,),
                            audit_index["model"],
                        )
                        if trusted_renderer
                        else None
                    ),
                )
                if not trusted_renderer:
                    raise ContractError(
                        "external renderers are restricted to negative "
                        "failure and timeout diagnostics"
                    )
                receipt_relative = f"receipts/{request_id}.json"
                receipt = {
                    "schema": "terrain-oracle-render-receipt/v1",
                    "kind": "accepted_interval",
                    "clip_id": clip.clip_id,
                    "clip_sha256": record.sha256,
                    "model": audit_index["model"],
                    "terrain_sha256": report.terrain_sha256,
                    "terrain_mesh_sha256": clip.terrain.mesh_sha256,
                    "interval": [
                        interval.start_frame,
                        interval.end_frame,
                    ],
                    "renderer": _renderer_receipt_evidence(rendered),
                    "video": rendered["video"],
                    "contact_overlay": rendered["contact_overlay"],
                    "completed": True,
                }
                reference = _write_render_receipt(
                    stage,
                    receipt_relative,
                    receipt,
                )
                interval_references.append(reference)
                expected_receipts[receipt_relative] = receipt
        accepted_inputs.sort(key=lambda item: str(item["interval_key"]))
        accepted_keys = [
            str(item["interval_key"]) for item in accepted_inputs
        ]
        sheet_request = {
            "kind": "contact_sheet",
            "interval_keys": accepted_keys,
        }
        sheet_rendered, sheet_id = _run_renderer(
            stage=stage,
            renderer=renderer,
            renderer_arguments=renderer_arguments,
            renderer_sha256=renderer_sha256,
            request=sheet_request,
            semantic_arguments=[
                "--kind",
                "contact_sheet",
                *[
                    value
                    for key in accepted_keys
                    for value in ("--interval-key", key)
                ],
            ],
            timeout_seconds=timeout_seconds,
            trusted_request=(
                _trusted_media_request(
                    "contact_sheet",
                    tuple(
                        trusted_inputs_by_key[key]
                        for key in accepted_keys
                    ),
                    audit_index["model"],
                )
                if trusted_renderer
                else None
            ),
        )
        if not trusted_renderer:
            raise ContractError(
                "external renderers cannot publish contact-sheet evidence"
            )
        sheet_receipt = {
            "schema": "terrain-oracle-render-receipt/v1",
            "kind": "contact_sheet",
            "inputs": accepted_inputs,
            "interval_keys": accepted_keys,
            "renderer": _renderer_receipt_evidence(sheet_rendered),
            "video": sheet_rendered["video"],
            "contact_overlay": sheet_rendered["contact_overlay"],
            "completed": True,
        }
        sheet_relative = f"contact-sheet/{sheet_id}.json"
        sheet_reference = _write_render_receipt(
            stage,
            sheet_relative,
            sheet_receipt,
        )
        expected_receipts[sheet_relative] = sheet_receipt
        stratum_references: list[dict[str, str]] = []
        for stratum_key in sorted(strata):
            stratum = {
                name: value
                for name, value in zip(
                    ("source", "action_class", "terrain", "direction"),
                    stratum_key,
                    strict=True,
                )
            }
            inputs = sorted(
                strata[stratum_key],
                key=lambda item: str(item["interval_key"]),
            )
            interval_keys = [
                str(item["interval_key"]) for item in inputs
            ]
            request = {
                "kind": "full_video",
                "stratum": stratum,
                "interval_keys": interval_keys,
            }
            stratum_json = json.dumps(
                stratum,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            rendered, request_id = _run_renderer(
                stage=stage,
                renderer=renderer,
                renderer_arguments=renderer_arguments,
                renderer_sha256=renderer_sha256,
                request=request,
                semantic_arguments=[
                    "--kind",
                    "full_video",
                    "--stratum",
                    stratum_json,
                    *[
                        value
                        for key in interval_keys
                        for value in ("--interval-key", key)
                    ],
                ],
                timeout_seconds=timeout_seconds,
                trusted_request=(
                    _trusted_media_request(
                        "full_video",
                        tuple(
                            trusted_inputs_by_key[key]
                            for key in interval_keys
                        ),
                        audit_index["model"],
                    )
                    if trusted_renderer
                    else None
                ),
            )
            receipt = {
                "schema": "terrain-oracle-render-receipt/v1",
                "kind": "full_video",
                "stratum": stratum,
                "inputs": inputs,
                "interval_keys": interval_keys,
                "renderer": _renderer_receipt_evidence(rendered),
                "video": rendered["video"],
                "contact_overlay": rendered["contact_overlay"],
                "completed": True,
            }
            relative_path = f"strata/{request_id}.json"
            stratum_references.append(
                _write_render_receipt(
                    stage,
                    relative_path,
                    receipt,
                )
            )
            expected_receipts[relative_path] = receipt
        interval_references.sort(
            key=lambda item: item["relative_path"]
        )
        stratum_references.sort(
            key=lambda item: item["relative_path"]
        )
        without_hash = {
            "schema": "terrain-oracle-render-index/v2",
            "corpus_manifest_sha256": hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest(),
            "audit_index_sha256": hashlib.sha256(
                (root / "audit-index.json").read_bytes()
            ).hexdigest(),
            "accepted_interval_keys": accepted_keys,
            "renderer": {
                "argv_prefix": [str(renderer), *renderer_arguments],
                "executable_sha256": renderer_sha256,
                "mode": "trusted-package-media-v1",
                "module_sha256": hashlib.sha256(
                    Path(
                        sys.modules[
                            "mm_sonic.terrain_oracle.render_media"
                        ].__file__
                    ).read_bytes()
                ).hexdigest(),
            },
            "interval_receipts": interval_references,
            "contact_sheet_receipt": sheet_reference,
            "stratum_receipts": stratum_references,
        }
        index = {
            **without_hash,
            "content_sha256": hashlib.sha256(
                storage._canonical_json_bytes(
                    without_hash,
                    "render audit index",
                )
            ).hexdigest(),
        }
        storage._write_no_replace(
            stage / "render-index.json",
            storage._canonical_json_bytes(
                index,
                "render audit index",
            ),
        )
        _verify_render_stage(stage, index, expected_receipts)
        for directory in (
            "artifacts",
            "receipts",
            "contact-sheet",
            "strata",
        ):
            storage._fsync_directory(stage / directory)
        storage._seal_directory(stage)
        storage._publish_directory_no_replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _procedural_family_id(clip: CanonicalClip) -> str | None:
    if not clip.source.source_format.startswith("grail-clean-"):
        return None
    family, separator, _ = clip.clip_id.partition("/")
    if not separator or not family:
        raise ContractError(
            "GRAIL clip ID must retain its procedural family prefix"
        )
    return family


def _accepted_coverage_records(
    root: Path,
    *,
    model_snapshot: _FreezeModelSnapshot | None = None,
) -> tuple[coverage.AcceptedClipRecord, ...]:
    manifest, clips, audit_index, reports = _load_audit_evidence(
        root,
        model_snapshot=model_snapshot,
    )
    meshes = tuple(
        storage.read_mesh(root / record.relative_path)
        for record in manifest.meshes
    )
    mesh_by_sha256 = {
        record.sha256: mesh
        for record, mesh in zip(manifest.meshes, meshes, strict=True)
    }
    if len(mesh_by_sha256) != len(meshes):
        raise ContractError("corpus mesh records must be unique")
    accepted: list[coverage.AcceptedClipRecord] = []
    for clip_record, clip in zip(
        manifest.clips,
        clips,
        strict=True,
    ):
        report = reports[clip_record.sha256]
        if report.status == "rejected":
            if report.accepted_intervals:
                raise ContractError(
                    "rejected audit unexpectedly contains accepted intervals"
                )
            continue
        if not report.accepted_intervals:
            raise ContractError(
                "non-rejected audit must contain accepted intervals"
            )
        if clip.terrain is None:
            raise ContractError(
                f"accepted clip {clip.clip_id} has no terrain binding"
            )
        mesh = mesh_by_sha256.get(clip.terrain.mesh_sha256)
        if mesh is None:
            raise ContractError(
                f"accepted clip {clip.clip_id} terrain mesh is absent"
            )
        query = CanonicalMeshQuery(
            mesh,
            clip.terrain.world_from_terrain,
        )
        accepted.append(
            coverage.AcceptedClipRecord(
                clip=clip,
                clip_record=clip_record,
                audit=report,
                accepted_intervals=report.accepted_intervals,
                terrain_mesh=mesh,
                terrain_query=query,
                model_sha256=report.model_sha256,
                action_class=_action_class(clip),
                procedural_family_id=_procedural_family_id(clip),
            )
        )
    return tuple(accepted)


def _publish_coverage(arguments: argparse.Namespace) -> None:
    root = _cli_input_directory(arguments.corpus, "audited corpus")
    destination = _cli_output_path(arguments.output, "coverage output")
    if (
        destination.parent.resolve() != root
        or destination.name != "coverage.json"
    ):
        raise ContractError(
            "coverage output must be the canonical corpus child coverage.json"
        )
    manifest = coverage.build_coverage(
        _accepted_coverage_records(root)
    )
    coverage.freeze_coverage(destination, manifest)


def _load_coverage_evidence(
    root: Path,
    *,
    model_snapshot: _FreezeModelSnapshot | None = None,
) -> tuple[coverage.CoverageManifest, bytes]:
    path = root / "coverage.json"
    value, payload = _load_canonical_json(
        path,
        "coverage manifest",
    )
    loaded = coverage.CoverageManifest.from_dict(value)
    expected = coverage.build_coverage(
        _accepted_coverage_records(
            root,
            model_snapshot=model_snapshot,
        )
    )
    if loaded.to_dict() != expected.to_dict():
        raise ContractError(
            "coverage manifest does not equal recomputed Task 8 coverage"
        )
    return loaded, payload


def _stable_descriptor_bytes(descriptor: int, label: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"{label} is not a regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if before_identity != after_identity or len(payload) != after.st_size:
        raise ContractError(f"{label} changed while it was snapshotted")
    return payload


def _snapshot_directory_nofollow(source: Path, destination: Path) -> None:
    """Copy one exact tree through directory descriptors without following links."""

    if destination.is_symlink() or not destination.is_dir():
        raise ContractError("freeze snapshot destination must be a directory")
    if any(destination.iterdir()):
        raise ContractError("freeze snapshot destination must be empty")

    def copy_directory(source_descriptor: int, target: Path) -> None:
        names = sorted(os.listdir(source_descriptor))
        for name in names:
            if name in ("", ".", "..") or "/" in name:
                raise ContractError("freeze source contains a noncanonical name")
            before = os.stat(
                name,
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
            output = target / name
            if stat.S_ISDIR(before.st_mode):
                os.mkdir(output, mode=0o700)
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    dir_fd=source_descriptor,
                )
                try:
                    opened = os.fstat(child_descriptor)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (before.st_dev, before.st_ino)
                    ):
                        raise ContractError(
                            "freeze source directory changed during snapshot"
                        )
                    copy_directory(child_descriptor, output)
                    closed = os.fstat(child_descriptor)
                    if (closed.st_dev, closed.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise ContractError(
                            "freeze source directory identity changed"
                        )
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(
                    "freeze source contains a symlink or non-regular entry"
                )
            file_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=source_descriptor,
            )
            try:
                opened = os.fstat(file_descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                ):
                    raise ContractError(
                        "freeze source file changed during snapshot"
                    )
                payload = _stable_descriptor_bytes(
                    file_descriptor,
                    f"freeze source file {name}",
                )
            finally:
                os.close(file_descriptor)
            storage._write_no_replace(output, payload)
        if sorted(os.listdir(source_descriptor)) != names:
            raise ContractError(
                "freeze source directory listing changed during snapshot"
            )

    source_descriptor = os.open(
        source,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        copy_directory(source_descriptor, destination)
    finally:
        os.close(source_descriptor)
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        storage._fsync_directory(directory)
    storage._fsync_directory(destination)


def _snapshot_freeze_model(
    corpus_snapshot: Path,
    temporary_parent: Path,
) -> _FreezeModelSnapshot:
    index, _ = _load_canonical_json(
        corpus_snapshot / "audit-index.json",
        "freeze model audit index",
    )
    model_record = _normalized_model_record(
        index.get("model"),
        "freeze model audit authority",
    )
    model_path = _cli_input_file(
        Path(str(model_record["asset_path"])),
        "freeze mechanical model",
    )
    if str(model_path) != model_record["asset_path"]:
        raise ContractError("freeze mechanical model path is not canonical")
    model_payload = storage._read_regular_file_nofollow(model_path)
    if (
        len(model_payload) != model_record["asset_size_bytes"]
        or hashlib.sha256(model_payload).hexdigest()
        != model_record["asset_sha256"]
    ):
        raise ContractError("freeze mechanical model bytes are stale")
    try:
        import mujoco
        from . import render_media

        trusted_model = _trusted_media_model(model_record)
        model_spec, _model_vfs = render_media._authenticated_model_spec(
            trusted_model
        )
        source_model = model_spec.compile()
    except Exception as error:
        raise ContractError(
            "mechanical source model cannot be compiled for freeze"
        ) from error
    if storage._read_regular_file_nofollow(model_path) != model_payload:
        raise ContractError(
            "mechanical source model changed during freeze snapshot"
        )
    if (
        structural_model_sha256(source_model)
        != model_record["structural_sha256"]
    ):
        raise ContractError("freeze source model structural hash is stale")

    descriptor, compiled_name = tempfile.mkstemp(
        prefix=".terrain-oracle-model.",
        suffix=".mjb",
        dir=temporary_parent,
    )
    os.close(descriptor)
    compiled_path = Path(compiled_name)
    try:
        mujoco.mj_saveModel(source_model, str(compiled_path))
        compiled_payload = storage._read_regular_file_nofollow(
            compiled_path
        )
        released_model = mujoco.MjModel.from_binary_path(
            str(compiled_path)
        )
    except Exception as error:
        raise ContractError(
            "freeze MJB model snapshot cannot be produced or reloaded"
        ) from error
    finally:
        compiled_path.unlink(missing_ok=True)
    if (
        not compiled_payload
        or structural_model_sha256(released_model)
        != model_record["structural_sha256"]
    ):
        raise ContractError("freeze MJB structural model hash is stale")
    return _FreezeModelSnapshot(
        asset_path=str(model_record["asset_path"]),
        asset_payload=model_payload,
        compiled_mjb_payload=compiled_payload,
        model=released_model,
    )


def _freeze_source_paths(
    root: Path,
    manifest: storage.CorpusManifest,
    audit_index: dict[str, object],
) -> tuple[str, ...]:
    expected_files = {
        storage.COMPLETION_MARKER,
        "manifest.json",
        "audit-index.json",
        "coverage.json",
        *(record.relative_path for record in manifest.clips),
        *(record.relative_path for record in manifest.meshes),
        *(
            str(entry["relative_path"])
            for entry in audit_index["reports"]
        ),
        *(
            path.relative_to(root).as_posix()
            for path in (root / "render-audit").rglob("*")
            if path.is_file()
        ),
    }
    expected_directories = {
        parent.as_posix()
        for relative_path in expected_files
        for parent in Path(relative_path).parents
        if parent.as_posix() != "."
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ContractError("freeze source must not contain symlinks")
        if path.is_dir():
            actual_directories.add(relative_path)
        elif path.is_file():
            actual_files.add(relative_path)
        else:
            raise ContractError("freeze source contains a non-regular entry")
    if actual_files != expected_files:
        raise ContractError(
            "freeze source files do not form the exact evidence set"
        )
    if actual_directories != expected_directories:
        raise ContractError(
            "freeze source directories do not form the exact evidence set"
        )
    return tuple(
        sorted(expected_files - {storage.COMPLETION_MARKER})
    )


def _frozen_file_record(path: Path, relative_path: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _canonical_frozen_path(value: object) -> str:
    if type(value) is not str or not value:
        raise ContractError("frozen file path must be a nonempty string")
    path = Path(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ContractError("frozen file path must be canonical and relative")
    return value


def _load_frozen_corpus(root: Path) -> dict[str, object]:
    """Validate a frozen release without consulting its source model tree."""

    frozen_root = Path(root).expanduser().resolve()
    if not frozen_root.is_dir() or frozen_root.is_symlink():
        raise ContractError("frozen corpus is missing or a symlink")
    storage.load_corpus(frozen_root)
    index, _ = _load_canonical_json(
        frozen_root / "freeze-index.json",
        "frozen corpus index",
    )
    if set(index) != {
        "schema",
        "source",
        "model",
        "files",
        "content_sha256",
    }:
        raise ContractError("frozen corpus index fields do not match v1")
    without_hash = dict(index)
    content_sha256 = without_hash.pop("content_sha256")
    if (
        index["schema"] != "terrain-oracle-frozen-corpus/v1"
        or content_sha256
        != hashlib.sha256(
            storage._canonical_json_bytes(
                without_hash,
                "frozen corpus index",
            )
        ).hexdigest()
    ):
        raise ContractError("frozen corpus index content hash is stale")
    source = index["source"]
    if not isinstance(source, dict) or set(source) != {
        "corpus_manifest_sha256",
        "audit_index_sha256",
        "render_index_sha256",
        "coverage_sha256",
    }:
        raise ContractError("frozen source evidence fields do not match v1")
    source_paths = {
        "corpus_manifest_sha256": frozen_root / "manifest.json",
        "audit_index_sha256": frozen_root / "audit-index.json",
        "render_index_sha256": (
            frozen_root / "render-audit" / "render-index.json"
        ),
        "coverage_sha256": frozen_root / "coverage.json",
    }
    for field, path in source_paths.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != source[field]
        ):
            raise ContractError(f"frozen {field} binding is stale")
    declared_files = index["files"]
    if not isinstance(declared_files, list):
        raise ContractError("frozen corpus file evidence must be a list")
    expected_paths: set[str] = set()
    file_record_by_path: dict[str, dict[str, object]] = {}
    for record in declared_files:
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ContractError("frozen corpus file fields do not match v1")
        relative_path = _canonical_frozen_path(record["relative_path"])
        if relative_path in expected_paths:
            raise ContractError("frozen corpus file path is duplicated")
        path = frozen_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ContractError("frozen corpus file is missing or a symlink")
        if _frozen_file_record(path, relative_path) != record:
            raise ContractError("frozen corpus file hash mismatch")
        expected_paths.add(relative_path)
        file_record_by_path[relative_path] = record
    if [
        str(record["relative_path"]) for record in declared_files
    ] != sorted(expected_paths):
        raise ContractError("frozen corpus files are not canonically sorted")
    actual_paths: set[str] = set()
    actual_directories: set[str] = set()
    for path in frozen_root.rglob("*"):
        relative_path = path.relative_to(frozen_root).as_posix()
        if path.is_symlink():
            raise ContractError("frozen corpus must not contain symlinks")
        if path.is_dir():
            actual_directories.add(relative_path)
        elif path.is_file():
            actual_paths.add(relative_path)
        else:
            raise ContractError("frozen corpus contains a non-regular entry")
    if actual_paths != {
        storage.COMPLETION_MARKER,
        "freeze-index.json",
        *expected_paths,
    }:
        raise ContractError("frozen corpus files do not form the exact set")
    expected_directories = {
        parent.as_posix()
        for relative_path in actual_paths
        for parent in Path(relative_path).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise ContractError("frozen corpus directories do not form the exact set")
    model = index["model"]
    if not isinstance(model, dict) or set(model) != {
        "relative_path",
        "sha256",
        "size_bytes",
        "original_model_file_sha256",
        "structural_sha256",
    }:
        raise ContractError("frozen model evidence fields do not match v1")
    model_relative_path = _canonical_relative_path(
        model["relative_path"],
        "model",
    )
    if (
        model_relative_path
        != f"model/{model['sha256']}.mjb"
        or file_record_by_path.get(model_relative_path)
        != {
            "relative_path": model_relative_path,
            "sha256": model["sha256"],
            "size_bytes": model["size_bytes"],
        }
    ):
        raise ContractError("frozen model file evidence is stale")
    audit_value, _ = _load_canonical_json(
        frozen_root / "audit-index.json",
        "frozen audit index",
    )
    audit_model = audit_value.get("model")
    if (
        not isinstance(audit_model, dict)
        or model["original_model_file_sha256"]
        != audit_model.get("asset_sha256")
        or model["structural_sha256"]
        != audit_model.get("structural_sha256")
    ):
        raise ContractError("frozen model provenance does not match audit")
    try:
        import mujoco

        released_model = mujoco.MjModel.from_binary_path(
            str(frozen_root / model_relative_path)
        )
    except Exception as error:
        raise ContractError("frozen MJB model cannot be reloaded") from error
    if (
        structural_model_sha256(released_model)
        != model["structural_sha256"]
    ):
        raise ContractError("frozen MJB structural model hash is stale")
    coverage_value, _ = _load_canonical_json(
        frozen_root / "coverage.json",
        "frozen coverage manifest",
    )
    coverage.CoverageManifest.from_dict(coverage_value)
    return index


def _verify_frozen_stage(
    stage: Path,
    expected_index: dict[str, object],
) -> None:
    index = _load_frozen_corpus(stage)
    if index != expected_index:
        raise ContractError("reloaded frozen corpus index does not match")


def _freeze_corpus(arguments: argparse.Namespace) -> None:
    root = _cli_input_directory(arguments.corpus, "audited corpus")
    destination = _cli_output_path(arguments.output, "freeze output")
    if destination == root or root in destination.parents:
        raise ContractError("freeze output must not be inside its source corpus")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        _snapshot_directory_nofollow(root, stage)
        model_snapshot = _snapshot_freeze_model(
            stage,
            destination.parent,
        )
        manifest, _, audit_index, _ = _load_audit_evidence(
            stage,
            model_snapshot=model_snapshot,
        )
        _load_render_evidence(
            stage,
            model_snapshot=model_snapshot,
            request_root=root,
        )
        _, coverage_payload = _load_coverage_evidence(
            stage,
            model_snapshot=model_snapshot,
        )
        source_paths = _freeze_source_paths(
            stage,
            manifest,
            audit_index,
        )
        model_record = audit_index["model"]
        (stage / storage.COMPLETION_MARKER).unlink()
        model_payload = model_snapshot.compiled_mjb_payload
        model_sha256 = hashlib.sha256(model_payload).hexdigest()
        model_relative_path = f"model/{model_sha256}.mjb"
        storage._write_no_replace(
            stage / model_relative_path,
            model_payload,
        )
        file_records = [
            _frozen_file_record(stage / relative_path, relative_path)
            for relative_path in sorted(
                (*source_paths, model_relative_path)
            )
        ]
        without_hash = {
            "schema": "terrain-oracle-frozen-corpus/v1",
            "source": {
                "corpus_manifest_sha256": hashlib.sha256(
                    (stage / "manifest.json").read_bytes()
                ).hexdigest(),
                "audit_index_sha256": hashlib.sha256(
                    (stage / "audit-index.json").read_bytes()
                ).hexdigest(),
                "render_index_sha256": hashlib.sha256(
                    (
                        stage
                        / "render-audit"
                        / "render-index.json"
                    ).read_bytes()
                ).hexdigest(),
                "coverage_sha256": hashlib.sha256(
                    coverage_payload
                ).hexdigest(),
            },
            "model": {
                "relative_path": model_relative_path,
                "sha256": model_sha256,
                "size_bytes": len(model_payload),
                "original_model_file_sha256": model_record["asset_sha256"],
                "structural_sha256": model_record["structural_sha256"],
            },
            "files": file_records,
        }
        index = {
            **without_hash,
            "content_sha256": hashlib.sha256(
                storage._canonical_json_bytes(
                    without_hash,
                    "frozen corpus index",
                )
            ).hexdigest(),
        }
        storage._write_no_replace(
            stage / "freeze-index.json",
            storage._canonical_json_bytes(
                index,
                "frozen corpus index",
            ),
        )
        for path in sorted(
            (path for path in stage.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            storage._fsync_directory(path)
        storage._seal_directory(stage)
        _verify_frozen_stage(stage, index)
        storage._publish_directory_no_replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _inventory_source_root(path: Path, label: str) -> Path:
    return _cli_input_directory(path, f"{label} source root")


def _clip_inventory_record(clip: CanonicalClip) -> dict[str, object]:
    clip.validate()
    source_path = Path(clip.source.source_path)
    if source_path.is_symlink():
        raise ContractError(
            f"source for clip {clip.clip_id} must not be a symlink"
        )
    return {
        "clip_id": clip.clip_id,
        "frame_count": clip.frame_count,
        "source_format": clip.source.source_format,
        "source_path": clip.source.source_path,
        "source_size_bytes": clip.source.source_size_bytes,
        "source_sha256": clip.source.source_sha256,
        "source_license_id": clip.source.source_license_id,
    }


def _duplicate_source_groups(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_sha256: dict[str, list[str]] = {}
    for record in records:
        by_sha256.setdefault(
            str(record["source_sha256"]),
            [],
        ).append(str(record["clip_id"]))
    return [
        {
            "source_sha256": sha256,
            "clip_ids": sorted(clip_ids),
        }
        for sha256, clip_ids in sorted(by_sha256.items())
        if len(clip_ids) > 1
    ]


def _validated_clip_inventory(
    clips: Sequence[CanonicalClip],
    label: str,
) -> dict[str, object]:
    records = sorted(
        (_clip_inventory_record(clip) for clip in clips),
        key=lambda record: (
            str(record["clip_id"]),
            str(record["source_path"]),
        ),
    )
    clip_ids = [str(record["clip_id"]) for record in records]
    if len(clip_ids) != len(set(clip_ids)):
        raise ContractError(f"{label} inventory contains duplicate clip IDs")
    counts = Counter(str(record["source_format"]) for record in records)
    return {
        "count": len(records),
        "counts_by_format": dict(sorted(counts.items())),
        "records": records,
        "duplicate_source_sha256": _duplicate_source_groups(records),
    }


def _flat_inventory(path: Path) -> dict[str, object]:
    root = _inventory_source_root(path, "flat")
    clips = tuple(
        source_flat.iter_flat_clips(
            root,
            tags=("flat", "other"),
        )
    )
    inventory = _validated_clip_inventory(clips, "flat")
    return {"root": str(root), **inventory}


def _inventory_terrain_binding() -> TerrainBinding:
    return TerrainBinding(
        asset_path="inventory://unresolved-terrain",
        asset_size_bytes=1,
        asset_sha256="0" * 64,
        asset_license_id="UNRECORDED",
        mesh_sha256="1" * 64,
        world_from_terrain=RigidTransform(
            np.zeros(3, dtype=np.float32),
            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        ),
        validity_mask_path=None,
    )


def _justin_inventory(path: Path) -> dict[str, object]:
    root = _inventory_source_root(path, "Justin")
    clips = tuple(
        source_justin.iter_justin_clips(
            root,
            _inventory_terrain_binding(),
        )
    )
    inventory = _validated_clip_inventory(clips, "Justin")
    return {"root": str(root), **inventory}


def _file_identity(path: Path, relative_path: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"required source file is missing: {path}")
    payload = path.read_bytes()
    return {
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _huggingface_metadata_identity(
    release: Path,
    relative_path: str,
) -> dict[str, object]:
    metadata_relative_path = (
        f".cache/huggingface/download/{relative_path}.metadata"
    )
    metadata_path = release / metadata_relative_path
    identity = _file_identity(
        metadata_path,
        metadata_relative_path,
    )
    try:
        lines = metadata_path.read_text("utf-8").splitlines()
    except UnicodeError as error:
        raise ContractError(
            f"Hugging Face metadata is not UTF-8: {metadata_path}"
        ) from error
    if (
        len(lines) < 2
        or lines[0] != source_lafan.LAFAN1_UPSTREAM_REVISION
        or not lines[1]
    ):
        raise ContractError(
            f"Hugging Face metadata commit is not pinned for {relative_path}"
        )
    return {
        **identity,
        "commit": lines[0],
        "etag": lines[1],
    }


def _validate_lafan_license(readme: bytes, license_payload: bytes) -> None:
    try:
        readme_text = readme.decode("utf-8")
        license_text = license_payload.decode("utf-8")
    except UnicodeError as error:
        raise ContractError("LAFAN README and LICENSE must be UTF-8") from error
    normalized_readme = " ".join(readme_text.lower().split())
    if (
        "license: cc-by-nc-nd-4.0" not in normalized_readme
        and (
            "licensed under creative commons "
            "attribution-noncommercial-noderivatives 4.0 international"
        )
        not in normalized_readme
    ):
        raise ContractError(
            "LAFAN README must declare dataset license CC-BY-NC-ND-4.0"
        )
    normalized_license = " ".join(license_text.lower().split())
    if (
        not normalized_license.startswith("bsd 3-clause license")
        or "redistribution and use in source and binary forms"
        not in normalized_license
    ):
        raise ContractError(
            "LAFAN code LICENSE does not identify BSD-3-Clause"
        )


def _lafan_inventory(path: Path) -> dict[str, object]:
    root = _inventory_source_root(path, "LAFAN")
    if root.name != "g1":
        raise ContractError("LAFAN inventory root must be the release g1 directory")
    release = root.parent
    readme_path = release / "README.md"
    license_path = release / "LICENSE"
    readme_identity = _file_identity(readme_path, "README.md")
    license_identity = _file_identity(license_path, "LICENSE")
    readme_payload = readme_path.read_bytes()
    license_payload = license_path.read_bytes()
    _validate_lafan_license(readme_payload, license_payload)
    supporting_files = {
        "LICENSE": {
            **license_identity,
            "metadata": _huggingface_metadata_identity(
                release,
                "LICENSE",
            ),
        },
        "README.md": {
            **readme_identity,
            "metadata": _huggingface_metadata_identity(
                release,
                "README.md",
            ),
        },
    }
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    if not entries:
        raise ContractError("LAFAN g1 inventory is empty")
    records: list[dict[str, object]] = []
    source_format = (
        "lafan1-retargeted-g1-csv-30hz@"
        + source_lafan.LAFAN1_UPSTREAM_REVISION
    )
    for csv_path in entries:
        if (
            csv_path.is_symlink()
            or not csv_path.is_file()
            or csv_path.suffix != ".csv"
        ):
            raise ContractError(
                f"LAFAN g1 contains a noncanonical entry: {csv_path.name}"
            )
        rows = source_lafan._load_rows(csv_path)
        source_lafan.classify_lafan_name(csv_path.stem)
        identity = _file_identity(
            csv_path,
            f"g1/{csv_path.name}",
        )
        records.append(
            {
                "clip_id": csv_path.stem,
                "source_frame_count": int(rows.shape[0]),
                "source_format": source_format,
                "source_path": str(csv_path.resolve()),
                "source_size_bytes": identity["size_bytes"],
                "source_sha256": identity["sha256"],
                "source_license_id": "CC-BY-NC-ND-4.0",
                "metadata": _huggingface_metadata_identity(
                    release,
                    f"g1/{csv_path.name}",
                ),
            }
        )
    clip_ids = [str(record["clip_id"]) for record in records]
    if len(clip_ids) != len(set(clip_ids)):
        raise ContractError("LAFAN inventory contains duplicate clip IDs")
    return {
        "root": str(root),
        "repository": "lvhaidong/LAFAN1_Retargeting_Dataset",
        "revision": source_lafan.LAFAN1_UPSTREAM_REVISION,
        "license_id": "CC-BY-NC-ND-4.0",
        "code_license_id": "BSD-3-Clause",
        "supporting_files": supporting_files,
        "count": len(records),
        "counts_by_format": {source_format: len(records)},
        "records": records,
        "duplicate_source_sha256": _duplicate_source_groups(records),
    }


def _grail_inventory(
    path: Path,
    families_value: str,
) -> dict[str, object]:
    root = _inventory_source_root(path, "GRAIL")
    if type(families_value) is not str:
        raise ContractError("GRAIL families must be comma-separated")
    families = tuple(families_value.split(","))
    records = source_grail.discover_clean_c490_records(
        root,
        families=families,
    )
    output_records: list[dict[str, object]] = []
    for record in records:
        if (
            record.robot_path.is_symlink()
            or not record.robot_path.is_file()
            or record.usd_path.is_symlink()
            or not record.usd_path.is_file()
        ):
            raise ContractError(
                f"GRAIL pair is missing or symlinked: {record.stem}"
            )
        output_records.append(_grail_inventory_record(root, record))
    output_records.sort(
        key=lambda record: (
            str(record["family"]),
            str(record["stem"]),
        )
    )
    stems = [str(record["stem"]) for record in output_records]
    if len(stems) != len(set(stems)):
        raise ContractError("GRAIL inventory contains duplicate stems")
    family_counts = Counter(
        str(record["family"]) for record in output_records
    )
    return {
        "root": str(root),
        "families": list(families),
        "count": len(output_records),
        "counts_by_family": dict(sorted(family_counts.items())),
        "counts_by_format": {
            "grail-clean-paired-record": len(output_records)
        },
        "records": output_records,
    }


def _publish_inventory(arguments: argparse.Namespace) -> None:
    sources: dict[str, object] = {}
    if arguments.flat is not None:
        sources["flat"] = _flat_inventory(arguments.flat)
    if arguments.justin is not None:
        sources["justin"] = _justin_inventory(arguments.justin)
    if arguments.grail_root is not None:
        sources["grail"] = _grail_inventory(
            arguments.grail_root,
            arguments.grail_families
            or ",".join(source_grail.C490_FAMILIES),
        )
    elif arguments.grail_families is not None:
        raise ContractError("--grail-families requires --grail-root")
    if arguments.lafan is not None:
        sources["lafan"] = _lafan_inventory(arguments.lafan)
    if not sources:
        raise ContractError("inventory requires at least one source option")
    counts_by_source = {
        name: int(value["count"])
        for name, value in sorted(sources.items())
    }
    counts_by_format: Counter[str] = Counter()
    for value in sources.values():
        counts_by_format.update(value["counts_by_format"])
    without_hash = {
        "schema": "terrain-oracle-source-inventory/v1",
        "sources": sources,
        "summary": {
            "clip_count": sum(counts_by_source.values()),
            "counts_by_source": counts_by_source,
            "counts_by_format": dict(sorted(counts_by_format.items())),
        },
    }
    inventory = {
        **without_hash,
        "content_sha256": hashlib.sha256(
            storage._canonical_json_bytes(
                without_hash,
                "source inventory",
            )
        ).hexdigest(),
    }
    output = _cli_output_path(arguments.output, "inventory output")
    _write_inventory_no_replace(
        output,
        storage._canonical_json_bytes(inventory, "source inventory"),
    )


def _write_inventory_no_replace(destination: Path, payload: bytes) -> None:
    """Atomically publish a file on Linux filesystems and NFS without replace."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            storage._rename_noreplace(temporary, destination)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
            try:
                os.link(
                    temporary,
                    destination,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raise
            finally:
                temporary.unlink(missing_ok=True)
        storage._fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mm_sonic.terrain_oracle.corpus_cli"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory_parser = commands.add_parser(
        "inventory",
        help="validate and hash canonical source inventories",
    )
    inventory_parser.add_argument("--flat", type=Path)
    inventory_parser.add_argument("--justin", type=Path)
    inventory_parser.add_argument("--grail-root", type=Path)
    inventory_parser.add_argument("--grail-families")
    inventory_parser.add_argument("--lafan", type=Path)
    inventory_parser.add_argument("--output", type=Path, required=True)
    inventory_parser.set_defaults(handler=_publish_inventory)
    import_parser = commands.add_parser(
        "import",
        help="publish canonical source clips without replacing prior output",
    )
    import_parser.add_argument("--fixture", type=Path)
    import_parser.add_argument("--inventory", type=Path)
    import_parser.add_argument("--lafan-inventory", type=Path)
    import_parser.add_argument("--flat", type=Path)
    import_parser.add_argument("--justin", type=Path)
    import_parser.add_argument("--grail-root", type=Path)
    import_parser.add_argument("--grail-families")
    import_parser.add_argument("--lafan", type=Path)
    import_parser.add_argument("--model", type=Path)
    import_parser.add_argument("--justin-terrain", type=Path)
    import_parser.add_argument("--output", type=Path, required=True)
    import_parser.set_defaults(handler=_import_from_arguments)
    audit_parser = commands.add_parser(
        "audit",
        help="mechanically audit every canonical clip and republish evidence",
    )
    audit_parser.add_argument("--corpus", type=Path, required=True)
    audit_parser.add_argument("--model", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(handler=_audit_corpus)
    render_parser = commands.add_parser(
        "render-audit",
        help="render every mechanically accepted interval",
    )
    render_parser.add_argument("--corpus", type=Path, required=True)
    render_parser.add_argument(
        "--renderer",
        type=Path,
        help=argparse.SUPPRESS,
    )
    render_parser.add_argument(
        "--renderer-arg",
        action="append",
        default=[],
    )
    render_parser.add_argument(
        "--renderer-timeout-seconds",
        type=float,
        default=3600.0,
    )
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.set_defaults(handler=_render_interval_receipts)
    coverage_parser = commands.add_parser(
        "coverage",
        help="freeze Task 8 coverage from mechanically accepted intervals",
    )
    coverage_parser.add_argument("--corpus", type=Path, required=True)
    coverage_parser.add_argument("--output", type=Path, required=True)
    coverage_parser.set_defaults(handler=_publish_coverage)
    freeze_parser = commands.add_parser(
        "freeze",
        help="publish the exact audited, rendered, covered corpus",
    )
    freeze_parser.add_argument("--corpus", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.set_defaults(handler=_freeze_corpus)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one corpus command, returning 2 for an actionable contract error."""

    try:
        arguments = _parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    try:
        arguments.handler(arguments)
    except (
        ContractError,
        FileExistsError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
