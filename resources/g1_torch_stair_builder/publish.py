"""Transactional publication of the five-clip native Torch stair slice."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from mm_sonic.torch_motion_data import MotionFolder
from resources.g1_terrain_builder.artifacts import _rename_exchange
from resources.g1_terrain_builder.kinematics import G1Kinematics

from .conversion import convert_source_to_native_50hz
from .corpus import PinnedStairSource, load_pinned_sources
from .surface import ZUpHeightGrid, build_source_height_grid


SCHEMA = "g1-torch-stair-slice/v1"
_CLIP_DIRECTORIES = ("0000", "0001", "0002", "updown-0000")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _deterministic_npz_bytes(fields: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(fields):
            array_bytes = io.BytesIO()
            np.lib.format.write_array(
                array_bytes,
                np.asanyarray(fields[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_bytes.getvalue())
    return output.getvalue()


def _copy_fsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output:
        shutil.copyfileobj(input_stream, output, 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_candidate(path: Path, manifest: dict) -> None:
    folder = MotionFolder.load(path)
    if len(folder.clips) != 5:
        raise ValueError("published stair slice must contain exactly five clips")
    for descriptor in manifest["clips"]:
        motion = path / descriptor["relative_motion_path"]
        if _sha256(motion) != descriptor["motion_sha256"]:
            raise ValueError("published motion hash changed")
        terrain = descriptor["terrain"]
        if terrain["kind"] == "heightgrid":
            grid_path = path / terrain["path"]
            ZUpHeightGrid.load(grid_path)
            if _sha256(grid_path) != terrain["sha256"]:
                raise ValueError("published terrain hash changed")


def _source_descriptor(
    source: PinnedStairSource,
    *,
    relative_motion: str,
    motion_sha256: str,
    relative_terrain: str,
    terrain_sha256: str,
    grid: ZUpHeightGrid,
) -> dict:
    return {
        "kind": "stair",
        "base": source.base,
        "relative_motion_path": relative_motion,
        "source_fps": source.source_fps,
        "source_frames": len(source.robot_qpos_mujoco),
        "output_frames": 499,
        "source_sha256": dict(source.source_sha256),
        "motion_sha256": motion_sha256,
        "terrain": {
            "kind": "heightgrid",
            "path": relative_terrain,
            "sha256": terrain_sha256,
            "origin_xy": [float(value) for value in grid.origin_xy],
            "cell_size_m": grid.cell_size_m,
            "shape": list(grid.height_z.shape),
        },
    }


def publish_stair_slice(
    *,
    output: str | Path,
    grail_root: str | Path,
    g1_xml: str | Path,
    flat_motion: str | Path,
    kinematics=None,
    grid_builder=None,
    validate_candidate=None,
) -> dict:
    output = Path(output).resolve()
    grail_root = Path(grail_root).resolve()
    g1_xml = Path(g1_xml).resolve()
    flat_motion = Path(flat_motion).resolve()
    if not g1_xml.is_file():
        raise ValueError(f"G1 XML is missing: {g1_xml}")
    if not flat_motion.is_file():
        raise ValueError(f"flat motion is missing: {flat_motion}")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ValueError("existing output must be a real directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = load_pinned_sources(grail_root)
    if kinematics is None:
        kinematics = G1Kinematics(str(g1_xml))
    if grid_builder is None:
        grid_builder = build_source_height_grid
    if validate_candidate is None:
        validate_candidate = lambda _path: None
    if not callable(grid_builder) or not callable(validate_candidate):
        raise TypeError("grid_builder and validate_candidate must be callable")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-", dir=output.parent
        )
    )
    committed = False
    try:
        flat_destination = staging / "flat" / "motion.npz"
        _copy_fsync(flat_motion, flat_destination)
        clips: list[dict] = [
            {
                "kind": "flat",
                "relative_motion_path": "flat/motion.npz",
                "source_fps": 50.0,
                "output_frames": int(
                    MotionFolder.load(flat_motion.parent).clips[0].frame_count
                ),
                "source_sha256": {"motion": _sha256(flat_motion)},
                "motion_sha256": _sha256(flat_destination),
                "terrain": {"kind": "flat"},
            }
        ]
        for source, directory in zip(sources, _CLIP_DIRECTORIES):
            converted = convert_source_to_native_50hz(source, kinematics)
            grid = grid_builder(source)
            if not isinstance(grid, ZUpHeightGrid):
                raise TypeError("grid_builder must return a ZUpHeightGrid")
            motion_relative = f"stair/{directory}/motion.npz"
            terrain_relative = f"stair/{directory}/terrain.npz"
            motion_path = staging / motion_relative
            terrain_path = staging / terrain_relative
            _write_bytes(
                motion_path, _deterministic_npz_bytes(converted.as_npz_fields())
            )
            _write_bytes(
                terrain_path, _deterministic_npz_bytes(grid.as_npz_fields())
            )
            clips.append(
                _source_descriptor(
                    source,
                    relative_motion=motion_relative,
                    motion_sha256=_sha256(motion_path),
                    relative_terrain=terrain_relative,
                    terrain_sha256=_sha256(terrain_path),
                    grid=grid,
                )
            )

        manifest = {
            "schema": SCHEMA,
            "output_fps": 50,
            "layout": "g1-29dof-isaaclab-v1",
            "g1_xml_sha256": _sha256(g1_xml),
            "clips": clips,
        }
        _write_bytes(
            staging / "manifest.json",
            (
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _validate_candidate(staging, manifest)
        validate_candidate(staging)
        _validate_candidate(staging, manifest)
        _fsync_tree(staging)

        if output.exists():
            _rename_exchange(str(staging), str(output))
            committed = True
            shutil.rmtree(staging)
        else:
            os.replace(staging, output)
            committed = True
        parent_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return manifest
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)

