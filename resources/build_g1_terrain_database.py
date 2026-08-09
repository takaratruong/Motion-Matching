#!/usr/bin/env python3
"""Build the versioned G1 motion-matching terrain artifact set."""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import (
    publish_artifacts,
    publish_flat_artifacts,
)
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
from resources.g1_terrain_builder.features import (
    FEATURE_NAMES,
    FEATURE_WEIGHTS,
    build_matching_features,
)
from resources.g1_terrain_builder.scenes import (
    REQUIRED_SCENE_IDS,
    all_scene_definitions,
    build_scene_pack,
    select_grail_scene_bases,
)
from resources.g1_terrain_builder.sources import (
    load_grail,
    load_retarget_npz,
    load_takara,
)
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    GrailTerrain,
    build_facing_centerline,
    sample_terrain_features,
    surface_semantics,
    surface_semantics_signature,
)


DEFAULTS = {
    "output": "resources/g1_terrain",
    "grail_glob": "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl",
    "g1_xml": "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml",
    "takara": "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/motion.npz",
    "remap": "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy",
}

SCHEMA = "g1-terrain-artifacts/v2"
OUTPUT_FPS = 25.0
TERRAIN_DISTANCES = [0.25, 0.50, 0.75, 1.00]


def _odd_frame_count(seconds: float, fps: float) -> int:
    count = int(round(seconds * fps))
    return count if count % 2 else count + 1


def finalize_clip(source, terrain, kin, output_fps=OUTPUT_FPS):
    clip, skeleton, report = convert_source_clip(source, kin, output_fps)
    gp, gq = forward_kinematics_arrays(
        clip.positions, clip.rotations, skeleton.parents)
    # quat.to_scaled_angle_axis uses np.where around its zero-angle branch;
    # NumPy evaluates both operands even though the finite fallback is selected.
    with np.errstate(divide="ignore", invalid="ignore"):
        clip.velocities, clip.angular_velocities = derive_velocities(
            clip.positions, clip.rotations, output_fps)
    clip.contacts = derive_contacts(
        gp, terrain,
        skeleton.names.index("LeftToe"),
        skeleton.names.index("RightToe"), output_fps,
        ContactConfig(median_filter_frames=_odd_frame_count(0.1, output_fps)))
    clip.terrain_support = sample_terrain_support(
        gp,
        terrain,
        skeleton.names.index("Simulation"),
        skeleton.names.index("LeftToe"),
        skeleton.names.index("RightToe"),
    )
    for frame in range(len(clip.positions)):
        stop = min(frame + int(round(2.0 * output_fps)) + 1,
                   len(clip.positions))
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


def _require_loaded_grail_source(source, base):
    if source.name != base or source.terrain_id != base:
        raise ValueError(
            f"{base}: loaded source name/terrain identity changed")


def _run_candidate_validator(staging, args):
    validator = os.path.join(
        REPOSITORY_ROOT, "resources", "validate_g1_terrain_database.py")
    argv = [sys.executable, validator, staging]
    if not getattr(args, "flat_only", False) and args.grail_limit is None:
        argv.extend([
            "--full-source-validation",
            "--grail-glob", args.grail_glob,
            "--g1-xml", args.g1_xml,
            "--takara", args.takara,
            "--remap", args.remap,
        ])
    try:
        subprocess.run(argv, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "candidate validator failed with exit status "
            f"{error.returncode}") from error


def _candidate_validation_policy(args):
    def validate_candidate(staging):
        return _run_candidate_validator(staging, args)

    return validate_candidate


def _inspect_grail_corpus(args):
    all_paths = tuple(sorted(glob.glob(args.grail_glob)))
    if not all_paths:
        raise FileNotFoundError(
            f"GRAIL glob matched no clips: {args.grail_glob}")
    for path in all_paths:
        _require_file(path, "GRAIL clip")

    path_by_base = {}
    for path in all_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        if not base or base in path_by_base:
            raise ValueError("duplicate GRAIL terrain base name")
        path_by_base[base] = path

    measured_max_heights = {}
    for base in path_by_base:
        terrain = GrailTerrain.from_base(base)
        measured_max_heights[base] = float(terrain.footprint()["height"])
        del terrain
    selected_scene_bases = select_grail_scene_bases(measured_max_heights)
    expected_scene_ids = tuple(REQUIRED_SCENE_IDS[:4])
    if tuple(selected_scene_bases) != expected_scene_ids \
            or len(set(selected_scene_bases.values())) != 4 \
            or not set(selected_scene_bases.values()) <= set(path_by_base):
        raise ValueError("GRAIL scene selection must contain four exact bases")

    motion_paths = all_paths
    if args.grail_limit is not None:
        motion_paths = all_paths[:args.grail_limit]
    return (
        all_paths, motion_paths, path_by_base,
        measured_max_heights, selected_scene_bases,
    )


def _flat_feature_signature(names, horizons, weights, offset, scale):
    payload = {
        "names": list(names), "horizons": list(horizons),
        "weights": list(weights), "offset": offset.tolist(),
        "scale": scale.tolist(),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _assemble_flat_candidate(args):
    if float(args.output_fps) != 60.0:
        raise ValueError("--flat-only requires --output-fps 60")
    if not args.retarget_npz or not args.retarget_receipt:
        raise ValueError(
            "--flat-only requires --retarget-npz and --retarget-receipt")
    _require_file(args.g1_xml, "G1 XML")
    _require_file(args.retarget_npz, "released-PFNN G1 retarget")
    _require_file(args.retarget_receipt, "released-PFNN G1 retarget receipt")
    source = load_retarget_npz(args.retarget_npz, args.retarget_receipt)
    if source.fps != 120.0 or source.terrain_id != "flat":
        raise ValueError("flat released-PFNN source must be 120 Hz and flat")
    kinematics = G1Kinematics(args.g1_xml)
    clip, skeleton, report = finalize_clip(
        source, FlatTerrain(), kinematics, output_fps=60.0)
    if len(skeleton.names) != 31:
        raise ValueError("flat G1 skeleton must contain exactly 31 bones")
    artifacts = combine_clips([clip], skeleton)
    horizons = (20, 40, 60)
    features = build_matching_features(artifacts, 60.0, horizons)
    provenance = source.provenance
    if type(provenance) is not dict:
        raise ValueError("flat source has no authenticated provenance")
    manifest_base = {
        "schema": "g1-lmm-flat-data/v1",
        "output_fps": 60.0,
        "trajectory_horizons": list(horizons),
        "feature_dimensions": 31,
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": list(FEATURE_WEIGHTS),
        "feature_offset": features.offset.tolist(),
        "feature_scale": features.scale.tolist(),
        "feature_signature": _flat_feature_signature(
            FEATURE_NAMES, horizons, FEATURE_WEIGHTS,
            features.offset, features.scale),
        "terrain_features": {
            "indices": [27, 28, 29, 30],
            "semantics": "authenticated-flat-root-relative-height-deltas",
            "value_m": 0.0,
        },
        "database_frames": len(artifacts.positions),
        "total_clips": 1,
        "dimensions": {"bones": 31, "features": 31, "contacts": 2},
        "skeleton": {
            "names": list(skeleton.names),
            "parents": skeleton.parents.tolist(),
            "basis": "holden-y-up-right-handed-forward-plus-z",
            "signature": skeleton.signature(),
        },
        "ranges": [{
            "start": 0, "stop": len(artifacts.positions),
            "motion_class": "flat-walk", "terrain_class": "flat",
            "source": source.name,
        }],
        "time_filters": {
            "root_position_frames": 31,
            "root_position_order": 3,
            "root_direction_frames": 61,
            "root_direction_order": 3,
            "contact_median_frames": 7,
            "forward_terrain_path_rows": 121,
        },
        "contact": {
            "speed_threshold": 0.15,
            "height_threshold": 0.06,
            "median_filter_frames": 7,
        },
        "sources": [{
            "name": source.name,
            "terrain_id": "flat",
            "path": provenance["path"],
            "sha256": provenance["sha256"],
            "receipt_path": provenance["receipt_path"],
            "receipt_sha256": provenance["receipt_sha256"],
            "receipt_schema": provenance["receipt"]["schema"],
            "receipt_status": provenance["receipt"]["status"],
            "source_fps": source.fps,
            "source_frames": len(source.qpos),
            "output_frames": len(clip.positions),
            "left_source_index": clip.source_left_indices.tolist(),
            "right_source_index": clip.source_right_indices.tolist(),
            "source_alpha": clip.source_alpha.tolist(),
        }],
        "validation": {
            "fk_max_error_m": report["fk_max_error_m"],
            "duration_error_s": report["duration_error_s"],
            "quaternion_norm_max_error": report[
                "quaternion_norm_max_error"],
        },
    }
    return artifacts, features, manifest_base


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


def _assemble_candidate(args):
    if args.grail_limit is not None and args.grail_limit < 0:
        raise ValueError("--grail-limit must be non-negative")
    _require_file(args.g1_xml, "G1 XML")
    _require_file(args.takara, "Takara motion")
    _require_file(args.remap, "Takara joint remap")
    (
        _all_grail_paths, grail_paths, path_by_base,
        measured_max_heights, selected_scene_bases,
    ) = _inspect_grail_corpus(args)
    if "takara_walk_50hz" in path_by_base:
        raise ValueError("duplicate source names: ['takara_walk_50hz']")

    kinematics = G1Kinematics(args.g1_xml)
    clips = []
    reports = []
    source_manifest = []
    clips_by_terrain = {}
    expected_skeleton = None
    range_cursor = 0

    def append_motion_source(source, terrain, expected_name, expected_terrain):
        nonlocal expected_skeleton, range_cursor
        if source.name != expected_name or source.terrain_id != expected_terrain:
            raise ValueError(
                f"{expected_name}: loaded source name/terrain identity changed")
        clip, skeleton, report = finalize_clip(source, terrain, kinematics)
        if expected_skeleton is None:
            expected_skeleton = skeleton
        elif skeleton.signature() != expected_skeleton.signature():
            raise ValueError(f"{source.name}: skeleton signature changed")
        clips.append(clip)
        reports.append({
            "fk_max_error_m": float(report["fk_max_error_m"]),
            "duration_error_s": float(report["duration_error_s"]),
            "quaternion_norm_max_error": float(
                report["quaternion_norm_max_error"]),
        })
        source_manifest.append(
            _source_manifest_entry(source, clip, range_cursor))
        range_cursor += len(clip.positions)
        if source.terrain_id in selected_scene_bases.values():
            clips_by_terrain[source.terrain_id] = clip

    source = load_takara(args.takara, args.remap)
    terrain = FlatTerrain()
    append_motion_source(
        source, terrain, "takara_walk_50hz", "flat")
    del source, terrain

    for path in grail_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        source = load_grail(path)
        _require_loaded_grail_source(source, base)
        terrain = GrailTerrain.from_base(base)
        append_motion_source(source, terrain, base, base)
        del source, terrain

    if expected_skeleton is None:
        raise ValueError("no source clips were built")
    if len(expected_skeleton.names) != 31:
        raise ValueError(
            f"G1 skeleton must contain 31 bones, got "
            f"{len(expected_skeleton.names)}")
    artifacts = combine_clips(clips, expected_skeleton)
    del clips
    if range_cursor != len(artifacts.positions):
        raise ValueError("source ranges do not cover the combined database")

    for base in selected_scene_bases.values():
        if base in clips_by_terrain:
            continue
        route_source = load_grail(path_by_base[base])
        _require_loaded_grail_source(route_source, base)
        route_clip, route_skeleton, _route_report = convert_source_clip(
            route_source, kinematics, OUTPUT_FPS)
        if route_skeleton.signature() != expected_skeleton.signature():
            raise ValueError(
                f"{base}: scene-route skeleton signature changed")
        clips_by_terrain[base] = route_clip
        del route_source, route_clip, route_skeleton, _route_report

    scene_pack = build_scene_pack(all_scene_definitions(
        measured_max_heights, clips_by_terrain))
    del clips_by_terrain

    contact_config = ContactConfig()
    manifest_base = {
        "schema": SCHEMA,
        "output_fps": OUTPUT_FPS,
        "feature_dimensions": 31,
        "terrain_dimensions": 4,
        "support_dimensions": 3,
        "terrain_feature_distances_m": list(TERRAIN_DISTANCES),
        "total_clips": len(source_manifest),
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
        "surface": {
            "semantics": surface_semantics(),
            "signature": surface_semantics_signature(),
        },
        "validation": {
            "schema": "g1-terrain-validation/v1",
            "fk_max_error_m": [
                report["fk_max_error_m"] for report in reports],
            "duration_error_s": [
                report["duration_error_s"] for report in reports],
            "quaternion_norm_max_error": [
                report["quaternion_norm_max_error"] for report in reports],
        },
    }
    return artifacts, manifest_base, scene_pack


def build_artifacts(args: argparse.Namespace) -> dict:
    if getattr(args, "flat_only", False):
        artifacts, features, manifest_base = _assemble_flat_candidate(args)
        return publish_flat_artifacts(
            args.output, artifacts, features, manifest_base,
            _candidate_validation_policy(args))
    artifacts, manifest_base, scene_pack = _assemble_candidate(args)
    validate_candidate = _candidate_validation_policy(args)
    return publish_artifacts(
        args.output, artifacts, manifest_base, scene_pack,
        validate_candidate)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build validated Holden G1 terrain motion artifacts")
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--output-fps", type=float, default=OUTPUT_FPS)
    parser.add_argument("--flat-only", action="store_true")
    parser.add_argument("--retarget-npz")
    parser.add_argument("--retarget-receipt")
    parser.add_argument("--grail-glob", default=DEFAULTS["grail_glob"])
    parser.add_argument("--grail-limit", type=int)
    parser.add_argument("--g1-xml", default=DEFAULTS["g1_xml"])
    parser.add_argument("--takara", default=DEFAULTS["takara"])
    parser.add_argument("--remap", default=DEFAULTS["remap"])
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_artifacts(args)
    except (OSError, TypeError, ValueError) as error:
        print(f"BUILD FAILED {args.output}: {error}", file=sys.stderr)
        return 1
    scenes = 0 if args.flat_only else len(REQUIRED_SCENE_IDS)
    print(
        f"BUILT {manifest['schema']} frames={manifest['database_frames']} "
        f"clips={manifest['total_clips']} scenes={scenes} "
        f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
