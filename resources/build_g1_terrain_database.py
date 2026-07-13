#!/usr/bin/env python3
"""Build the versioned G1 motion-matching terrain artifact set."""

import argparse
import glob
import os
import sys

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import publish_artifacts
from resources.g1_terrain_builder.database import (
    ContactConfig,
    combine_clips,
    derive_contacts,
    derive_velocities,
    forward_kinematics_arrays,
    sample_terrain_support,
)
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
)
from resources.g1_terrain_builder.sources import load_grail, load_takara
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    build_facing_centerline,
    export_heightfield,
    sample_terrain_features,
)


DEFAULTS = {
    "output": "resources/g1_terrain",
    "grail_glob": "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl",
    "g1_xml": "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
    "takara": "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz",
    "remap": "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy",
    "runtime_terrain": "terrain_curbs__curb_000__000",
}

SCHEMA = "g1-terrain-artifacts/v1"
OUTPUT_FPS = 25.0
TERRAIN_DISTANCES = [0.25, 0.50, 0.75, 1.00]
HEIGHTFIELD_CELL_SIZE = 0.02
HEIGHTFIELD_BORDER = 2.0


def finalize_clip(source, terrain, kin):
    clip, skeleton, report = convert_source_clip(source, kin, OUTPUT_FPS)
    gp, gq = forward_kinematics_arrays(
        clip.positions, clip.rotations, skeleton.parents)
    # quat.to_scaled_angle_axis uses np.where around its zero-angle branch;
    # NumPy evaluates both operands even though the finite fallback is selected.
    with np.errstate(divide="ignore", invalid="ignore"):
        clip.velocities, clip.angular_velocities = derive_velocities(
            clip.positions, clip.rotations, OUTPUT_FPS)
    clip.contacts = derive_contacts(
        gp, terrain,
        skeleton.names.index("LeftToe"),
        skeleton.names.index("RightToe"), OUTPUT_FPS)
    clip.terrain_support = sample_terrain_support(
        gp,
        terrain,
        skeleton.names.index("Simulation"),
        skeleton.names.index("LeftToe"),
        skeleton.names.index("RightToe"),
    )
    for frame in range(len(clip.positions)):
        stop = min(frame + 51, len(clip.positions))
        path = gp[frame:stop, 0][:, [0, 2]]
        headings3 = holden_quat.mul_vec(
            gq[frame:stop, 0],
            np.array([0.0, 0.0, 1.0], np.float64))
        headings = headings3[:, [0, 2]]
        centerline = build_facing_centerline(path[0], headings, path)
        clip.terrain_features[frame] = sample_terrain_features(
            terrain, centerline)
    clip.validate()
    return clip, skeleton, report


def _require_file(path: str, description: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {description}: {path}")


def _heightfield_contract(terrain: GrailTerrain) -> tuple[tuple[float, ...], dict]:
    xmin, xmax, zmin, zmax = terrain.xz_bounds()
    bounds = (
        float(xmin) - HEIGHTFIELD_BORDER,
        float(xmax) + HEIGHTFIELD_BORDER,
        float(zmin) - HEIGHTFIELD_BORDER,
        float(zmax) + HEIGHTFIELD_BORDER,
    )
    if not np.all(np.isfinite(bounds)):
        raise ValueError("runtime terrain XZ bounds must be finite")
    nx = int(np.ceil((bounds[1] - bounds[0]) / HEIGHTFIELD_CELL_SIZE)) + 1
    nz = int(np.ceil((bounds[3] - bounds[2]) / HEIGHTFIELD_CELL_SIZE)) + 1
    metadata = {
        "nx": nx,
        "nz": nz,
        "origin_x": bounds[0],
        "origin_z": bounds[2],
        "cell_size": HEIGHTFIELD_CELL_SIZE,
        "exterior_height": 0.0,
    }
    return bounds, metadata


def _source_manifest_entry(source, clip, range_start: int) -> dict:
    output_frames = len(clip.positions)
    return {
        "name": source.name,
        "terrain_id": source.terrain_id,
        "source_fps": float(source.fps),
        "source_frames": int(len(source.qpos)),
        "output_frames": int(output_frames),
        "range_start": int(range_start),
        "range_stop": int(range_start + output_frames),
        "source_frame_map": [int(value) for value in clip.source_frames],
    }


def build_artifacts(args: argparse.Namespace) -> dict:
    if args.grail_limit is not None and args.grail_limit < 0:
        raise ValueError("--grail-limit must be non-negative")
    _require_file(args.g1_xml, "G1 XML")
    _require_file(args.takara, "Takara motion")
    _require_file(args.remap, "Takara joint remap")

    grail_paths = sorted(glob.glob(args.grail_glob))
    if args.grail_limit != 0 and not grail_paths:
        raise FileNotFoundError(
            f"GRAIL glob matched no clips: {args.grail_glob}")
    if args.grail_limit is not None:
        grail_paths = grail_paths[:args.grail_limit]
    for path in grail_paths:
        _require_file(path, "GRAIL clip")

    kin = G1Kinematics(args.g1_xml)
    sources = [load_takara(args.takara, args.remap)]
    sources.extend(load_grail(path) for path in grail_paths)
    seen_names = set()
    duplicate_names = set()
    for source in sources:
        if source.name in seen_names:
            duplicate_names.add(source.name)
        seen_names.add(source.name)
    if duplicate_names:
        raise ValueError(f"duplicate source names: {sorted(duplicate_names)}")

    clips = []
    reports = []
    source_manifest = []
    expected_skeleton = None
    range_cursor = 0
    for source in sources:
        terrain = (
            FlatTerrain() if source.terrain_id == "flat"
            else GrailTerrain.from_base(source.terrain_id)
        )
        clip, skeleton, report = finalize_clip(source, terrain, kin)
        if expected_skeleton is None:
            expected_skeleton = skeleton
        elif skeleton.signature() != expected_skeleton.signature():
            raise ValueError(f"{source.name}: skeleton signature changed")
        clips.append(clip)
        reports.append(report)
        source_manifest.append(
            _source_manifest_entry(source, clip, range_cursor))
        range_cursor += len(clip.positions)

    if expected_skeleton is None:
        raise ValueError("no source clips were built")
    if len(expected_skeleton.names) != 31:
        raise ValueError(
            f"G1 skeleton must contain 31 bones, got "
            f"{len(expected_skeleton.names)}")
    artifacts = combine_clips(clips, expected_skeleton)
    if range_cursor != len(artifacts.positions):
        raise ValueError("source ranges do not cover the combined database")

    runtime_terrain = GrailTerrain.from_base(args.runtime_terrain)
    runtime_bounds, heightfield_metadata = _heightfield_contract(runtime_terrain)
    contact_config = ContactConfig()
    manifest = {
        "schema": SCHEMA,
        "output_fps": OUTPUT_FPS,
        "feature_dimensions": 31,
        "terrain_dimensions": 4,
        "total_clips": len(sources),
        "grail_clips": len(grail_paths),
        "skipped_clips": 0,
        "database_frames": len(artifacts.positions),
        "diagnostic_mode": args.grail_limit is not None,
        "sources": source_manifest,
        "skeleton": {
            "names": list(expected_skeleton.names),
            "parents": expected_skeleton.parents.tolist(),
            "signature": expected_skeleton.signature(),
        },
        "contact": {
            "speed_threshold": contact_config.speed_threshold,
            "height_threshold": contact_config.height_threshold,
            "median_filter_frames": contact_config.median_filter_frames,
        },
        "terrain": {
            "distances_m": TERRAIN_DISTANCES,
            "coordinate_mapping": "mujoco_xyz_to_holden_x_z_neg_y",
            "runtime_base": args.runtime_terrain,
            "cell_size_m": HEIGHTFIELD_CELL_SIZE,
            "border_m": HEIGHTFIELD_BORDER,
            "heightfield": heightfield_metadata,
        },
        "validation": {
            "fk_max_error_m": [
                float(report["fk_max_error_m"]) for report in reports],
            "duration_error_s": [
                float(report["duration_error_s"]) for report in reports],
            "quaternion_norm_max_error": [
                float(report["quaternion_norm_max_error"])
                for report in reports
            ],
        },
    }

    def terrain_writer(staging: str) -> None:
        emitted = export_heightfield(
            runtime_terrain,
            runtime_bounds,
            HEIGHTFIELD_CELL_SIZE,
            os.path.join(staging, "terrain.bin"),
        )
        if emitted != heightfield_metadata:
            raise ValueError(
                "runtime heightfield metadata changed during publication: "
                f"expected {heightfield_metadata}, got {emitted}")
        runtime_terrain.export_obj(os.path.join(staging, "terrain.obj"))

    publish_artifacts(args.output, artifacts, manifest, terrain_writer)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build validated Holden G1 terrain motion artifacts")
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--grail-glob", default=DEFAULTS["grail_glob"])
    parser.add_argument("--grail-limit", type=int)
    parser.add_argument("--g1-xml", default=DEFAULTS["g1_xml"])
    parser.add_argument("--takara", default=DEFAULTS["takara"])
    parser.add_argument("--remap", default=DEFAULTS["remap"])
    parser.add_argument(
        "--runtime-terrain", default=DEFAULTS["runtime_terrain"])
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_artifacts(args)
    except (OSError, TypeError, ValueError) as error:
        print(f"BUILD FAILED {args.output}: {error}", file=sys.stderr)
        return 1
    print(
        f"BUILT {manifest['schema']} frames={manifest['database_frames']} "
        f"clips={manifest['total_clips']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
