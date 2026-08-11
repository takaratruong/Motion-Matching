"""Authenticate and cache the published PFNN exporter and display terrain."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .published_pfnn_g1_scenes import SceneSpec
from .published_pfnn_heightmap import build_mesh


RELEASED_PFNN_CPP_SHA256 = (
    "6deb74a9af58874e2b9c88cca760f9aa1c9e6d4db10fa10ae02b39609e3d3154"
)
_EXPORT_RECEIPT = ".published_pfnn_g1_export.json"
_COMPILE_COMMAND = [
    "g++",
    "-std=gnu++11",
    "-Wall",
    "-O3",
    "-ffast-math",
    "pfnn.cpp",
    "-lGL",
    "-lGLEW",
    "-lSDL2",
    "-g",
    "-o",
    "pfnn_export",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_exporter(
    source_demo: Path, cache_demo: Path, resources: Path
) -> Path:
    """Authenticate, patch, and compile the released PFNN demo in ``cache_demo``."""
    source_demo = Path(source_demo)
    cache_demo = Path(cache_demo)
    resources = Path(resources)
    source_cpp = source_demo / "pfnn.cpp"
    try:
        source_digest = _sha256(source_cpp)
    except OSError as error:
        raise ValueError("released pfnn.cpp SHA could not be authenticated") from error
    if source_digest != RELEASED_PFNN_CPP_SHA256:
        raise ValueError("released pfnn.cpp SHA-256 mismatch")

    patch_path = resources / "pfnn_export.patch"
    try:
        patch_digest = _sha256(patch_path)
    except OSError as error:
        raise ValueError("published PFNN exporter patch is unavailable") from error
    executable = cache_demo / "pfnn_export"
    receipt_path = cache_demo / _EXPORT_RECEIPT
    expected_receipt = {
        "patch_sha256": patch_digest,
        "source_pfnn_cpp_sha256": source_digest,
    }
    try:
        cached_receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        cached_receipt = None
    if cached_receipt == expected_receipt and executable.is_file():
        return executable

    cache_demo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_demo, cache_demo, dirs_exist_ok=True)
    subprocess.run(
        ["patch", "-p1", "--input", str(patch_path.resolve())],
        cwd=cache_demo,
        check=True,
    )
    subprocess.run(_COMPILE_COMMAND, cwd=cache_demo, check=True)
    if not executable.is_file():
        raise RuntimeError("PFNN exporter compilation produced no executable")
    _write_json_atomic(receipt_path, expected_receipt)
    return executable


def terrain_cache_path(
    scene: SceneSpec, source_demo: Path, cache_root: Path
) -> Path:
    """Return the content-addressed terrain path without writing it."""
    source_heightmap = Path(source_demo) / "heightmaps" / scene.heightmap
    heightmap_digest = _sha256(source_heightmap)
    parameters = {
        "heightmap_sha256": heightmap_digest,
        "stride": scene.display_stride,
        "uniform_scale": 0.875,
    }
    cache_key = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Path(cache_root) / "terrain" / f"{Path(scene.heightmap).stem}-{cache_key}.npz"


def prepare_terrain(
    scene: SceneSpec, source_demo: Path, cache_root: Path
) -> Path:
    """Build or reuse the morphology-scaled display mesh for ``scene``."""
    source_heightmap = Path(source_demo) / "heightmaps" / scene.heightmap
    output = terrain_cache_path(scene, source_demo, cache_root)
    terrain_root = output.parent
    terrain_root.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return output

    vertices, faces = build_mesh(
        source_heightmap,
        uniform_scale=0.875,
        stride=scene.display_stride,
    )
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, vertices=vertices, faces=faces)
    temporary.replace(output)
    return output


__all__ = [
    "RELEASED_PFNN_CPP_SHA256",
    "prepare_exporter",
    "prepare_terrain",
    "terrain_cache_path",
]
