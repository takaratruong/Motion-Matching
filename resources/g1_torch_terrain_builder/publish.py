"""Transactional publisher for the authenticated expanded terrain corpus."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable

from mm_sonic.torch_motion_data import MotionFolder
from mm_sonic.torch_terrain_features import TerrainDataset
from resources.g1_terrain_builder.artifacts import _rename_exchange
from resources.g1_terrain_builder.kinematics import G1Kinematics
from resources.g1_torch_stair_builder.conversion import (
    _body_permutation,
    convert_source_to_native_50hz,
)
from resources.g1_torch_stair_builder.corpus import load_pinned_sources
from resources.g1_torch_stair_builder.publish import (
    _deterministic_npz_bytes,
    _fsync_tree,
    _sha256,
    _write_bytes,
)
from resources.g1_torch_stair_builder.surface import ZUpHeightGrid

from .admission import AdmissionReport, admit_candidate
from .motion import load_native_motion_50hz
from .registry import ResolvedSource, resolve_registered_sources
from .terrain import TerrainEvidence, build_source_terrain


SCHEMA = "g1-torch-terrain-corpus/v1"
LAYOUT = "g1-29dof-isaaclab-v1"


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _distribution(value) -> dict[str, float | None]:
    return {str(key): _json_number(number) for key, number in value.items()}


def _admission_descriptor(report: AdmissionReport) -> dict:
    return {
        "logical_name": report.logical_name,
        "accepted": report.accepted,
        "reason": report.reason,
        "output_frames": report.output_frames,
        "contact_sample_count": report.contact_sample_count,
        "elevated_contact_sample_count": (
            report.elevated_contact_sample_count
        ),
        "contact_height_error_m": _distribution(
            report.contact_height_error_m
        ),
        "fk_position_error_m": _distribution(report.fk_position_error_m),
        "fk_full_body_position_error_m": _distribution(
            report.fk_full_body_position_error_m
        ),
        "duplicate_of": report.duplicate_of,
    }


def _candidate_failure(
    source: ResolvedSource, reason: str, error: Exception
) -> dict:
    return {
        "logical_name": source.spec.logical_name,
        "accepted": False,
        "reason": reason,
        "stage_error": {
            "type": type(error).__name__,
            "message": str(error)[:1000],
        },
        "source_sha256": dict(source.source_sha256),
    }


def _terrain_descriptor(
    terrain: TerrainEvidence,
    *,
    relative_path: str,
    sha256: str,
) -> dict:
    grid = terrain.grid
    return {
        "kind": "heightgrid",
        "adapter": terrain.adapter,
        "path": relative_path,
        "sha256": sha256,
        "origin_xy": [float(value) for value in grid.origin_xy],
        "cell_size_m": grid.cell_size_m,
        "shape": list(grid.height_z.shape),
        "geometry_sha256": dict(terrain.geometry_sha256),
        "motion_to_terrain_xy_yaw": [
            float(value) for value in terrain.motion_to_terrain_xy_yaw
        ],
    }


def _default_motion_loader(
    source: ResolvedSource,
    *,
    grail_root: Path,
    kinematics,
):
    if source.spec.source_adapter == "native-npz":
        return load_native_motion_50hz(
            source.motion_path,
            layout_provenance=source.spec.expected_layout,
        )
    if source.spec.source_adapter != "grail-record":
        raise ValueError(
            f"unsupported source adapter: {source.spec.source_adapter}"
        )
    pinned = next(
        (
            candidate
            for candidate in load_pinned_sources(grail_root)
            if candidate.robot_path == source.motion_path
        ),
        None,
    )
    if pinned is None:
        raise ValueError("registered GRAIL motion is not pinned")
    return convert_source_to_native_50hz(pinned, kinematics)


def _validate_candidate(path: Path, manifest: dict) -> None:
    folder = MotionFolder.load(path)
    descriptors = manifest["clips"]
    if len(folder.clips) != len(descriptors):
        raise ValueError("published clip count changed")
    for descriptor in descriptors:
        motion = path / descriptor["relative_motion_path"]
        if _sha256(motion) != descriptor["motion_sha256"]:
            raise ValueError("published motion hash changed")
        terrain = descriptor["terrain"]
        if terrain["kind"] == "heightgrid":
            grid_path = path / terrain["path"]
            ZUpHeightGrid.load(grid_path)
            if _sha256(grid_path) != terrain["sha256"]:
                raise ValueError("published terrain hash changed")
    TerrainDataset.load(path, device="cpu")


def publish_expanded_corpus(
    *,
    output: str | Path,
    source_root: str | Path,
    grail_root: str | Path,
    g1_xml: str | Path,
    flat_motion: str | Path,
    resolved_sources: Iterable[ResolvedSource] | None = None,
    motion_loader: Callable[[ResolvedSource], object] | None = None,
    terrain_builder: Callable[[ResolvedSource, object], object] | None = None,
    fk: Callable | None = None,
    kinematics=None,
    validate_candidate: Callable[[Path], None] | None = None,
) -> dict:
    output = Path(output).resolve()
    source_root = Path(source_root).resolve()
    grail_root = Path(grail_root).resolve()
    g1_xml = Path(g1_xml).resolve()
    flat_motion = Path(flat_motion).resolve()
    if not g1_xml.is_file() or g1_xml.is_symlink():
        raise ValueError(f"G1 XML is missing or unsafe: {g1_xml}")
    if not flat_motion.is_file() or flat_motion.is_symlink():
        raise ValueError(f"flat motion is missing or unsafe: {flat_motion}")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise ValueError("existing output must be a real directory")
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = (
        tuple(resolved_sources)
        if resolved_sources is not None
        else resolve_registered_sources(
            source_root=source_root, grail_root=grail_root
        )
    )
    if any(not isinstance(source, ResolvedSource) for source in sources):
        raise TypeError("resolved sources must contain only ResolvedSource")
    if kinematics is None and (motion_loader is None or fk is None):
        kinematics = G1Kinematics(str(g1_xml))
    if motion_loader is None:
        motion_loader = lambda source: _default_motion_loader(
            source, grail_root=grail_root, kinematics=kinematics
        )
    if terrain_builder is None:
        terrain_builder = build_source_terrain
    if fk is None:
        permutation = _body_permutation(tuple(kinematics.names))

        def fk(qpos):
            position, quaternion = kinematics.world_from_qpos(qpos)
            return position[:, permutation], quaternion[:, permutation]

    if validate_candidate is None:
        validate_candidate = lambda _path: None
    if not all(
        callable(value)
        for value in (motion_loader, terrain_builder, fk, validate_candidate)
    ):
        raise TypeError("publisher dependencies must be callable")

    flat = load_native_motion_50hz(
        flat_motion, layout_provenance=LAYOUT
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-", dir=output.parent
        )
    )
    committed = False
    try:
        flat_relative = "flat/motion.npz"
        flat_path = staging / flat_relative
        _write_bytes(
            flat_path, _deterministic_npz_bytes(flat.as_npz_fields())
        )
        accepted: list[dict] = [
            {
                "logical_name": "flat-takara",
                "kind": "flat",
                "family": "flat",
                "relative_motion_path": flat_relative,
                "source_fps": 50.0,
                "output_frames": len(flat.joint_position),
                "source_sha256": {"motion": _sha256(flat_motion)},
                "motion_sha256": _sha256(flat_path),
                "terrain": {"kind": "flat"},
            }
        ]
        rejected: list[dict] = []
        accepted_hashes: dict[str, str] = {}
        for source in sources:
            try:
                motion = motion_loader(source)
                motion.validate()
            except Exception as error:
                rejected.append(
                    _candidate_failure(source, "motion_load_failed", error)
                )
                continue
            try:
                terrain = terrain_builder(source, motion)
            except Exception as error:
                rejected.append(
                    _candidate_failure(source, "terrain_adapter_failed", error)
                )
                continue
            try:
                report = admit_candidate(
                    source,
                    motion,
                    terrain,
                    fk=fk,
                    accepted_motion_hashes=accepted_hashes,
                )
            except Exception as error:
                rejected.append(
                    _candidate_failure(source, "admission_failed", error)
                )
                continue
            evidence = _admission_descriptor(report)
            if not report.accepted:
                rejected.append(
                    {
                        **evidence,
                        "source_sha256": dict(source.source_sha256),
                    }
                )
                continue

            base = (
                f"terrain/{source.spec.family}/"
                f"{source.spec.logical_name}"
            )
            motion_relative = f"{base}/motion.npz"
            terrain_relative = f"{base}/terrain.npz"
            motion_path = staging / motion_relative
            terrain_path = staging / terrain_relative
            _write_bytes(
                motion_path,
                _deterministic_npz_bytes(motion.as_npz_fields()),
            )
            _write_bytes(
                terrain_path,
                _deterministic_npz_bytes(terrain.grid.as_npz_fields()),
            )
            accepted.append(
                {
                    "logical_name": source.spec.logical_name,
                    "kind": "terrain",
                    "family": source.spec.family,
                    "source_adapter": source.spec.source_adapter,
                    "terrain_adapter": source.spec.terrain_adapter,
                    "relative_motion_path": motion_relative,
                    "source_fps": 50.0,
                    "output_frames": len(motion.joint_position),
                    "source_sha256": dict(source.source_sha256),
                    "motion_sha256": _sha256(motion_path),
                    "terrain": _terrain_descriptor(
                        terrain,
                        relative_path=terrain_relative,
                        sha256=_sha256(terrain_path),
                    ),
                    "admission": evidence,
                }
            )
            accepted_hashes[source.spec.motion_sha256] = (
                source.spec.logical_name
            )

        clips = sorted(
            accepted, key=lambda entry: entry["relative_motion_path"]
        )
        manifest = {
            "schema": SCHEMA,
            "output_fps": 50,
            "layout": LAYOUT,
            "g1_xml_sha256": _sha256(g1_xml),
            "clips": clips,
            "accepted_clips": clips,
            "rejected_candidates": rejected,
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
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)
