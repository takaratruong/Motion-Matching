#!/usr/bin/env python3
"""Build the versioned G1 motion-matching terrain artifact set."""

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import (
    _authenticate_flat_manifest_inputs,
    canonical_json_bytes,
    publish_artifacts,
    publish_flat_artifacts,
    publish_lmm_terrain_artifacts,
    read_features,
    sha256_file,
)
from resources.g1_terrain_builder.database import (
    ContactConfig,
    combine_clips,
    derive_contacts,
    derive_lmm_contacts,
    derive_velocities,
    forward_kinematics_arrays,
    read_holden_database,
    refresh_lmm_clip_dynamics,
    sample_terrain_support,
    slice_holden_clip,
)
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    convert_source_clip,
    load_mujoco_xml_assets,
    split_continuity_ranges,
)
from resources.g1_terrain_builder.features import (
    FEATURE_NAMES,
    FEATURE_WEIGHTS,
    build_lmm_terrain_features,
    build_matching_features,
)
from resources.g1_terrain_builder.scenes import (
    BuiltScene,
    REQUIRED_SCENE_IDS,
    ScenePack,
    all_scene_definitions,
    authored_slope_scene_definition,
    build_scene,
    build_scene_pack,
    select_grail_scene_bases,
)
from resources.g1_terrain_builder.sources import (
    AUTHORED_SLOPE_NAME,
    AUTHORED_SLOPE_ROBOT_SHA256,
    AUTHORED_SLOPE_ROBOT_SIZE_BYTES,
    load_authenticated_grail_slope,
    load_grail,
    load_retarget_npz,
    load_takara,
)
from resources.g1_terrain_builder.resample import resample_map, resample_vectors
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    HoldenClip,
    SkeletonSpec,
    SourceClip,
    require_canonical_g1_skeleton,
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
CANONICAL_FLAT_RETARGET_NAME = (
    "LocomotionFlat01_000-walk-only-7659-8171-120hz")
CANONICAL_FLAT_OUTPUT_SHA256 = (
    "bbdeb79760950480582ae937e54b913c376caa49f344896a8958476b82f3317f")
CANONICAL_FLAT_RECEIPT_SHA256 = (
    "2d0e93f485bab9c54773c66cf07c14d25837f5f4e50f5c007f9f8e2c5c20520f")
CANONICAL_FLAT_SOURCE_SHA256 = (
    "4a01768df71c6f7b5bbb71312c1e94eae489af32b21c3e300fd1c8d1ffc24bc5")
CANONICAL_FLAT_PREPARED_SHA256 = (
    "d6dbbac84e68d419d27aff0356b5a8245522f39694d94a6fc83e300ab6feaf8d")
CANONICAL_FLAT_GROUNDING_OFFSET_M = 0.06142798715901732
CANONICAL_FLAT_KINEMATICS_MODEL = {
    "asset": "g1_29dof.xml",
    "sha256": "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
    "size_bytes": 26914,
}
AUTHORED_SLOPE_SUPPORT_CALIBRATION_M = 0.012000000104308128
AUTHORED_SLOPE_INPUTS = {
    "usd": {
        "sha256":
            "8d1e696fb5bd2aecfa17797549bddd001093b060773cb313185a6a46db7eb5a5",
        "size_bytes": 6656,
    },
    "reconstruction": {
        "sha256":
            "d05d6c5a7d6a13eff7e69da0e9e96ab5a79f700bb706611699f0b9c44c9b51ec",
        "size_bytes": 411476,
    },
    "metadata": {
        "sha256":
            "7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d",
        "size_bytes": 85,
    },
}
AUTHORED_SLOPE_FEATURE_MINIMUM = (
    -0.05486539751291275,
    -0.09694159030914307,
    -0.12729622423648834,
    -0.12729622423648834,
)
AUTHORED_SLOPE_FEATURE_MAXIMUM = (
    0.05370394513010979,
    0.09506101161241531,
    0.12704572081565857,
    0.12723475694656372,
)
AUTHORED_SLOPE_FEATURE_STD = (
    0.02696162149121752,
    0.04589913504391204,
    0.05615584307619531,
    0.06524118885644932,
)
LMM_TERRAIN_SCHEMA = "g1-lmm-terrain-data/v1"
CANONICAL_FLAT_DATA_MANIFEST_SHA256 = (
    "5b5c48ccbb1dbabdbf87842d8b033c15b307199d72a8d90e4e39208ba5382db1"
)


@dataclass(frozen=True)
class AuthoredSlopeSourceCandidate:
    source: HoldenClip
    terrain: GrailTerrain
    scene: object
    provenance: dict


@dataclass(frozen=True)
class LMMTerrainCandidate:
    artifacts: object
    features: object
    manifest_base: dict
    validation: dict
    scene_pack: ScenePack


def _odd_frame_count(seconds: float, fps: float) -> int:
    count = int(round(seconds * fps))
    return count if count % 2 else count + 1


def finalize_clip(
    source, terrain, kin, output_fps=OUTPUT_FPS,
    root_filter_mode="interp",
):
    clip, skeleton, report = convert_source_clip(
        source, kin, output_fps, root_filter_mode=root_filter_mode)
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


def _read_canonical_flat_g1_xml(path: str) -> bytes:
    try:
        with open(path, "rb") as stream:
            payload = stream.read(
                CANONICAL_FLAT_KINEMATICS_MODEL["size_bytes"] + 1)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"canonical G1 XML authentication failed: {error}") from error
    if len(payload) != CANONICAL_FLAT_KINEMATICS_MODEL["size_bytes"] \
            or hashlib.sha256(payload).hexdigest() \
            != CANONICAL_FLAT_KINEMATICS_MODEL["sha256"]:
        raise ValueError(
            "canonical G1 XML content SHA-256/size changed")
    return payload


def _read_authored_slope_input(path: str, kind: str) -> bytes:
    descriptor = AUTHORED_SLOPE_INPUTS[kind]
    _require_file(path, f"authored slope {kind}")
    try:
        with open(path, "rb") as stream:
            payload = stream.read(descriptor["size_bytes"] + 1)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"authored slope {kind} authentication failed: {error}") from error
    if len(payload) != descriptor["size_bytes"] \
            or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError(
            f"authored slope {kind} content SHA-256/size changed")
    return payload


def _require_loaded_grail_source(source, base):
    if source.name != base or source.terrain_id != base:
        raise ValueError(
            f"{base}: loaded source name/terrain identity changed")


def _run_candidate_validator(staging, args):
    validator = os.path.join(
        REPOSITORY_ROOT, "resources", "validate_g1_terrain_database.py")
    argv = [sys.executable, validator, staging]
    if getattr(args, "flat_only", False):
        argv.extend(["--g1-xml", args.g1_xml])
    elif args.grail_limit is None:
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


def _maximum_local_rotation_steps(rotations):
    rotations = np.asarray(rotations, np.float64)
    if rotations.ndim != 3 or rotations.shape[-1] != 4 \
            or len(rotations) < 2 or not np.isfinite(rotations).all():
        raise ValueError("local rotations must be a finite (T, B, 4) array")
    norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("local rotations contain a zero quaternion")
    normalized = rotations / norms
    dots = np.abs(np.sum(normalized[:-1] * normalized[1:], axis=-1))
    return np.max(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)), axis=1)


def _require_canonical_flat_retarget(source):
    provenance = source.provenance
    if type(provenance) is not dict or type(provenance.get("receipt")) is not dict:
        raise ValueError("canonical flat retarget has no authenticated receipt")
    receipt = provenance["receipt"]
    expected = {
        "schema": "native-g1-pfnn-sample-retarget/v1",
        "status": "accepted",
        "source_sha256": CANONICAL_FLAT_SOURCE_SHA256,
        "prepared_sha256": CANONICAL_FLAT_PREPARED_SHA256,
        "output_sha256": CANONICAL_FLAT_OUTPUT_SHA256,
        "gmr_commit": "bb1bbe40774794fceb2a7c579a3464a28e68c844",
        "retarget_project_commit":
            "fb3433a6310ab4198102d3905e74b73944fc1f6b",
        "fps": 120.0,
        "source_frame_count": 8171,
        "start_frame": 7659,
        "frame_count": 512,
        "warmup_frames": 120,
        "aliases": [["Spine1", "Spine2"]],
        "root_quaternion_order": "xyzw",
        "grounding_offset_m": CANONICAL_FLAT_GROUNDING_OFFSET_M,
        "grounding": "flat",
        "pfnn_position_scale": 5.6444,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"canonical flat retarget receipt {key} changed")
    if source.name != CANONICAL_FLAT_RETARGET_NAME \
            or source.fps != 120.0 or source.terrain_id != "flat" \
            or len(source.qpos) != 512 \
            or not np.array_equal(
                source.source_frames, np.arange(7659, 8171, dtype=np.int32)) \
            or provenance.get("sha256") != CANONICAL_FLAT_OUTPUT_SHA256 \
            or provenance.get("receipt_sha256") \
            != CANONICAL_FLAT_RECEIPT_SHA256:
        raise ValueError("canonical flat retarget identity changed")


def _flat_contact_receipt(
    contacts, range_starts, range_stops, *, median_filter_frames,
):
    contacts = np.asarray(contacts)
    starts = np.asarray(range_starts)
    stops = np.asarray(range_stops)
    if contacts.ndim != 2 or contacts.shape[1] != 2 \
            or contacts.dtype != np.dtype(np.uint8) \
            or np.any((contacts != 0) & (contacts != 1)):
        raise ValueError("bilateral contact rows must be binary uint8 (T, 2)")
    if starts.ndim != 1 or stops.shape != starts.shape or len(starts) < 1 \
            or starts[0] != 0 or stops[-1] != len(contacts) \
            or np.any(starts[1:] != stops[:-1]) \
            or np.any(starts < 0) or np.any(stops <= starts):
        raise ValueError("bilateral contact ranges must exactly cover rows")
    if type(median_filter_frames) is not int or median_filter_frames < 1:
        raise ValueError("bilateral contact filter size is invalid")

    run_lengths = ([], [])
    events_by_range = []
    for start, stop in zip(starts.tolist(), stops.tolist()):
        events = []
        for side in range(2):
            values = contacts[start:stop, side].astype(np.int8, copy=False)
            edges = np.diff(np.pad(values, (1, 1)))
            run_starts = np.flatnonzero(edges == 1)
            run_stops = np.flatnonzero(edges == -1)
            for run_start, run_stop in zip(run_starts, run_stops):
                run_lengths[side].append(int(run_stop - run_start))
                events.append((int(run_start), side))
        events.sort()
        events_by_range.append(events)
    transitions = sum(
        left[1] != right[1]
        for events in events_by_range
        for left, right in zip(events, events[1:])
    )
    receipt = {
        "schema": "g1-lmm-bilateral-contact/v1",
        "left_contact_frames": int(contacts[:, 0].sum()),
        "right_contact_frames": int(contacts[:, 1].sum()),
        "left_run_count": len(run_lengths[0]),
        "right_run_count": len(run_lengths[1]),
        "left_max_run_frames": max(run_lengths[0], default=0),
        "right_max_run_frames": max(run_lengths[1], default=0),
        "alternating_run_transition_count": int(transitions),
    }
    if receipt["left_contact_frames"] == 0 \
            or receipt["right_contact_frames"] == 0 \
            or receipt["left_max_run_frames"] < median_filter_frames \
            or receipt["right_max_run_frames"] < median_filter_frames \
            or transitions < 1:
        raise ValueError(
            "bilateral contact observations lack meaningful alternating runs")
    return receipt


def _flat_source_fragment(source, first_source_frame, last_source_frame):
    first_source_frame = int(first_source_frame)
    last_source_frame = int(last_source_frame)
    if len(source.source_frames) != len(source.qpos) \
            or not np.array_equal(
                source.source_frames,
                np.arange(
                    int(source.source_frames[0]),
                    int(source.source_frames[0]) + len(source.source_frames),
                    dtype=source.source_frames.dtype)):
        raise ValueError("flat source frame provenance must be contiguous")
    local_first = first_source_frame - int(source.source_frames[0])
    local_last = last_source_frame - int(source.source_frames[0])
    if local_first < 0 or local_last < local_first \
            or local_last >= len(source.qpos):
        raise ValueError("flat continuity fragment source bounds are invalid")
    stop = local_last + 1
    fragment = SourceClip(
        source.name,
        source.fps,
        source.qpos[local_first:stop].copy(),
        source.source_frames[local_first:stop].copy(),
        source.terrain_id,
        source.provenance,
    )
    fragment.validate()
    return fragment


def _slice_holden_clip(clip: HoldenClip, start: int) -> HoldenClip:
    sliced = HoldenClip(
        clip.name,
        np.array(clip.positions[start:], np.float32, copy=True),
        np.zeros_like(clip.positions[start:], np.float32),
        np.array(clip.rotations[start:], np.float32, copy=True),
        np.zeros_like(clip.positions[start:], np.float32),
        np.zeros((len(clip.positions) - start, 2), np.uint8),
        np.zeros((len(clip.positions) - start, 4), np.float32),
        np.zeros((len(clip.positions) - start, 3), np.float32),
        np.array(clip.source_frames[start:], copy=True),
        clip.terrain_id,
        np.array(clip.source_left_indices[start:], np.int32, copy=True),
        np.array(clip.source_right_indices[start:], np.int32, copy=True),
        np.array(clip.source_alpha[start:], np.float32, copy=True),
    )
    sliced.validate()
    return sliced


def _finalize_authored_slope_clip(clip, terrain, skeleton, output_fps):
    gp, gq = forward_kinematics_arrays(
        clip.positions, clip.rotations, skeleton.parents)
    with np.errstate(divide="ignore", invalid="ignore"):
        clip.velocities, clip.angular_velocities = derive_velocities(
            clip.positions, clip.rotations, output_fps)
    clip.contacts = derive_lmm_contacts(
        clip.positions, clip.rotations, skeleton.parents,
        skeleton.names.index("LeftToe"),
        skeleton.names.index("RightToe"),
        output_fps,
    )
    clip.terrain_support = sample_terrain_support(
        gp, terrain,
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
        centerline = build_facing_centerline(
            path[0], headings3[:, [0, 2]], path)
        clip.terrain_features[frame] = sample_terrain_features(
            terrain, centerline)
    clip.validate()


def _require_authored_slope_feature_statistics(features):
    values = np.asarray(features)
    if values.shape != (595, 4) or values.dtype != np.dtype(np.float32) \
            or not np.isfinite(values).all():
        raise ValueError(
            "authored slope terrain features must be finite float32 (595, 4)")
    observed_minimum = np.min(values, axis=0)
    observed_maximum = np.max(values, axis=0)
    expected_minimum = np.asarray(
        AUTHORED_SLOPE_FEATURE_MINIMUM, np.float32)
    expected_maximum = np.asarray(
        AUTHORED_SLOPE_FEATURE_MAXIMUM, np.float32)
    if not np.array_equal(observed_minimum, expected_minimum):
        raise ValueError(
            "authored slope terrain feature minimum changed: "
            f"{observed_minimum}")
    if not np.array_equal(observed_maximum, expected_maximum):
        raise ValueError(
            "authored slope terrain feature maximum changed: "
            f"{observed_maximum}")
    observed_std = np.std(values.astype(np.float64), axis=0)
    expected_std = np.asarray(AUTHORED_SLOPE_FEATURE_STD, np.float64)
    if np.max(np.abs(observed_std - expected_std)) > 1e-12:
        raise ValueError(
            "authored slope terrain feature standard deviation changed: "
            f"{observed_std}")


def _merge_admitted_holden_fields(provisional, admitted):
    if len(provisional.positions) != 598 or len(admitted.positions) != 595:
        raise ValueError("authored slope provisional/admitted merge changed")
    for attribute in (
        "velocities", "angular_velocities", "contacts",
        "terrain_features", "terrain_support",
    ):
        getattr(provisional, attribute)[3:] = getattr(admitted, attribute)
    provisional.validate()


def _assemble_authored_slope_source(args):
    if float(args.output_fps) != 60.0:
        raise ValueError("authored slope source requires --output-fps 60")
    for attribute, description in (
        ("slope_robot", "robot"),
        ("slope_usd", "USD"),
        ("slope_recon", "reconstruction"),
        ("slope_metadata", "metadata"),
        ("g1_xml", "G1 XML"),
    ):
        path = getattr(args, attribute, None)
        if not path:
            raise ValueError(f"authored slope source requires --{attribute.replace('_', '-')}")
        _require_file(path, f"authored slope {description}")

    raw_source = load_authenticated_grail_slope(args.slope_robot)
    usd_bytes = _read_authored_slope_input(args.slope_usd, "usd")
    reconstruction_bytes = _read_authored_slope_input(
        args.slope_recon, "reconstruction")
    metadata_bytes = _read_authored_slope_input(
        args.slope_metadata, "metadata")
    metadata = pickle.loads(metadata_bytes)
    if type(metadata) is not dict \
            or type(metadata.get("scene_scale")) is not float \
            or metadata["scene_scale"] != 1.0:
        raise ValueError("authored slope metadata scene_scale changed")

    terrain = GrailTerrain.from_authenticated_bytes(
        usd_bytes, reconstruction_bytes,
        usd_source=os.path.abspath(args.slope_usd),
        reconstruction_source=os.path.abspath(args.slope_recon),
        support_calibration_m=AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
    )
    if terrain.exterior_height != -AUTHORED_SLOPE_SUPPORT_CALIBRATION_M:
        raise ValueError("authored slope exterior-flat calibration changed")

    xml_bytes = _read_canonical_flat_g1_xml(args.g1_xml)
    kinematics = G1Kinematics.from_xml_bytes(
        xml_bytes, load_mujoco_xml_assets(args.g1_xml))
    provisional, skeleton, report = convert_source_clip(
        raw_source, kinematics, target_fps=60.0)
    provisional.validate()
    require_canonical_g1_skeleton(skeleton, "authored slope skeleton")
    if len(provisional.positions) != 598:
        raise ValueError("authored slope provisional frame count changed")
    if np.any(provisional.positions[:, 0, 1] != 0.0):
        raise ValueError("authored slope Simulation is not planar")

    # The exact composed-MuJoCo preflight rejects provisional [0,3). Slice
    # before computing derivatives, contacts, support, or terrain features.
    admitted = _slice_holden_clip(provisional, 3)
    _finalize_authored_slope_clip(admitted, terrain, skeleton, 60.0)
    if len(admitted.positions) != 595:
        raise ValueError("authored slope admitted frame count changed")
    _require_authored_slope_feature_statistics(admitted.terrain_features)
    if not np.all(np.any(
        admitted.terrain_features[165:539] != 0.0, axis=1)):
        raise ValueError("authored slope intended nonflat exposure changed")

    reconstructed = terrain.source_to_runtime(terrain.source_vertices)
    recovered = terrain.runtime_to_source(reconstructed)
    round_trip_error = float(np.max(np.linalg.norm(
        recovered - terrain.source_vertices, axis=1)))
    if round_trip_error > 1e-9:
        raise ValueError(
            f"authored slope inverse round trip changed: {round_trip_error} m")

    source_span = (len(raw_source.qpos) - 1) / raw_source.fps
    output_span = (len(provisional.positions) - 1) / 60.0
    output_shortfall = source_span - output_span
    if source_span != 9.96 or output_span != 9.95 \
            or output_shortfall != 0.010000000000001563 \
            or report["duration_error_s"] != output_shortfall:
        raise ValueError("authored slope source/output span changed")
    receipt = {
        "schema": "g1-lmm-authored-slope-source/v1",
        "status": "provisional",
        "clip_id": AUTHORED_SLOPE_NAME,
        "hashes": {
            "robot_sha256": raw_source.provenance["sha256"],
            "usd_sha256": AUTHORED_SLOPE_INPUTS["usd"]["sha256"],
            "reconstruction_sha256":
                AUTHORED_SLOPE_INPUTS["reconstruction"]["sha256"],
            "metadata_sha256": AUTHORED_SLOPE_INPUTS["metadata"]["sha256"],
            "g1_xml_sha256": CANONICAL_FLAT_KINEMATICS_MODEL["sha256"],
        },
        "metadata": {"scene_scale": 1.0},
        "source_fps": 25.0,
        "target_fps": 60.0,
        "source_frames": 250,
        "provisional_output_frames": 598,
        "admitted_output_frames": 595,
        "source_span_s": source_span,
        "output_span_s": output_span,
        "output_shortfall_s": output_shortfall,
        "rejected_output_ranges": [[0, 3]],
        "admitted_output_range": [3, 598],
        "intended_nonflat_provisional_range": [168, 542],
        "intended_nonflat_admitted_range": [165, 539],
        "interpolation": {
            "left_source_index": provisional.source_left_indices.tolist(),
            "right_source_index": provisional.source_right_indices.tolist(),
            "source_alpha": provisional.source_alpha.tolist(),
        },
        "basis": "z-up-to-holden-shared-with-robot",
        "object_rotation_binary32":
            terrain.reconstruction_rotation.tolist(),
        "object_translation_binary32":
            terrain.reconstruction_translation.tolist(),
        "inverse_round_trip_max_error_m": round_trip_error,
        "exterior_policy": {
            "source_height_m": 0.0,
            "runtime_height_m": -AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
            "gradient_xz": [0.0, 0.0],
            "mesh_wins_inside": True,
        },
        "support_calibration_m": AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
        "support_calibration_applied_to": "terrain-only",
        "fk_max_error_m": report["fk_max_error_m"],
        "quaternion_norm_max_error": report["quaternion_norm_max_error"],
    }
    scene = authored_slope_scene_definition(admitted, terrain, receipt)
    _merge_admitted_holden_fields(provisional, admitted)
    return AuthoredSlopeSourceCandidate(provisional, terrain, scene, receipt)


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
    _require_canonical_flat_retarget(source)
    xml_bytes = _read_canonical_flat_g1_xml(args.g1_xml)
    kinematics = G1Kinematics.from_xml_bytes(
        xml_bytes, load_mujoco_xml_assets(args.g1_xml))
    preliminary, skeleton, _ = convert_source_clip(
        source, kinematics, target_fps=60.0, root_filter_mode="nearest")
    if len(preliminary.positions) != 256:
        raise ValueError("flat released-PFNN frame count changed")
    require_canonical_g1_skeleton(skeleton, "flat released-PFNN skeleton")
    native_dofs = resample_vectors(source.qpos[:, 7:], source.fps, 60.0)
    native_steps = np.max(np.abs(np.diff(native_dofs, axis=0)), axis=1)
    local_steps = _maximum_local_rotation_steps(preliminary.rotations)
    continuity_plan = split_continuity_ranges(
        native_steps, local_steps, threshold=0.25, minimum_frames=61)
    expected_plan = {
        "source_native_rejected_edge_count": 0,
        "database_local_rejected_edge_count": 0,
        "union_rejected_edge_count": 0,
        "dropped_fragment_count": 0,
        "dropped_frame_count": 0,
    }
    for key, expected in expected_plan.items():
        if continuity_plan[key] != expected:
            raise ValueError(
                f"released-PFNN continuity {key} changed: "
                f"{continuity_plan[key]} != {expected}")
    retained = continuity_plan["retained_ranges"]
    if len(retained) != 1 or sum(stop - start for start, stop in retained) \
            != 256:
        raise ValueError("released-PFNN retained continuity ranges changed")

    full_left, full_right, _full_alpha = resample_map(
        len(source.qpos), source.fps, 60.0)
    clips = []
    reports = []
    ranges = []
    database_cursor = 0
    expected_signature = G1_SKELETON_SIGNATURE
    for output_start, output_stop in retained:
        first_source = int(source.source_frames[full_left[output_start]])
        last_source = int(source.source_frames[full_right[output_stop - 1]])
        fragment = _flat_source_fragment(
            source, first_source, last_source)
        clip, fragment_skeleton, report = finalize_clip(
            fragment, FlatTerrain(), kinematics, output_fps=60.0,
            root_filter_mode="nearest")
        clip.contacts = derive_lmm_contacts(
            clip.positions,
            clip.rotations,
            fragment_skeleton.parents,
            fragment_skeleton.names.index("LeftToe"),
            fragment_skeleton.names.index("RightToe"),
            60.0,
        )
        require_canonical_g1_skeleton(
            fragment_skeleton, "flat continuity fragment skeleton")
        if fragment_skeleton.signature() != expected_signature \
                or len(clip.positions) != output_stop - output_start:
            raise ValueError("flat continuity fragment conversion changed")
        range_stop = database_cursor + len(clip.positions)
        ranges.append({
            "start": database_cursor,
            "stop": range_stop,
            "source": source.name,
            "source_first_frame": int(clip.source_left_indices[0]),
            "source_last_frame": int(clip.source_right_indices[-1]),
            "motion_class": "flat-walk",
            "terrain_class": "flat",
        })
        clips.append(clip)
        reports.append(report)
        database_cursor = range_stop

    artifacts = combine_clips(clips, skeleton)
    if database_cursor != 256 \
            or artifacts.range_starts.tolist() != [
                entry["start"] for entry in ranges] \
            or artifacts.range_stops.tolist() != [
                entry["stop"] for entry in ranges]:
        raise ValueError("flat continuity database ranges changed")
    horizons = (20, 40, 60)
    features = build_matching_features(artifacts, 60.0, horizons)
    left_source_index = np.concatenate([
        clip.source_left_indices for clip in clips]).astype(np.int32)
    right_source_index = np.concatenate([
        clip.source_right_indices for clip in clips]).astype(np.int32)
    source_alpha = np.concatenate([
        clip.source_alpha for clip in clips]).astype(np.float32)
    admitted_native = max(
        float(np.max(native_steps[start:stop - 1]))
        for start, stop in retained)
    admitted_local = max(
        float(np.max(_maximum_local_rotation_steps(
            artifacts.rotations[start:stop])))
        for start, stop in zip(
            artifacts.range_starts, artifacts.range_stops))
    range_rows = np.asarray([[
        entry["start"], entry["stop"], entry["source_first_frame"],
        entry["source_last_frame"],
    ] for entry in ranges], dtype="<i4")
    range_digest = hashlib.sha256(
        range_rows.tobytes(order="C")).hexdigest()
    source_map_digest = hashlib.sha256(b"".join((
        left_source_index.astype("<i4", copy=False).tobytes(order="C"),
        right_source_index.astype("<i4", copy=False).tobytes(order="C"),
        source_alpha.astype("<f4", copy=False).tobytes(order="C"),
    ))).hexdigest()
    provenance = source.provenance
    if type(provenance) is not dict:
        raise ValueError("flat source has no authenticated provenance")
    manifest_base = {
        "schema": "g1-lmm-flat-data/v3",
        "kinematics_model": dict(CANONICAL_FLAT_KINEMATICS_MODEL),
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
        "source_count": 1,
        "range_count": len(ranges),
        "dimensions": {"bones": 31, "features": 31, "contacts": 2},
        "skeleton": {
            "names": list(skeleton.names),
            "parents": skeleton.parents.tolist(),
            "basis": "holden-y-up-right-handed-forward-plus-z",
            "signature": G1_SKELETON_SIGNATURE,
        },
        "ranges": ranges,
        "continuity": {
            "schema": "g1-lmm-continuity/v1",
            "threshold_rad_per_frame": 0.25,
            "minimum_range_frames": 61,
            **{key: continuity_plan[key] for key in expected_plan},
            "published_range_count": len(ranges),
            "published_frame_count": len(artifacts.positions),
            "maximum_admitted_native_step_rad": admitted_native,
            "maximum_admitted_local_rotation_step_rad": admitted_local,
            "range_digest_sha256": range_digest,
            "source_map_digest_sha256": source_map_digest,
        },
        "time_filters": {
            "root_position_frames": 31,
            "root_position_order": 3,
            "root_direction_frames": 61,
            "root_direction_order": 3,
            "contact_median_frames": 6,
            "forward_terrain_path_rows": 121,
        },
        "contact": {
            "semantics": "bundled-orange-duck-global-toe-speed-only",
            "speed_threshold": 0.15,
            "median_filter_frames": 6,
            "median_filter_mode": "nearest",
        },
        "contact_observations": _flat_contact_receipt(
            artifacts.contacts,
            artifacts.range_starts,
            artifacts.range_stops,
            median_filter_frames=6,
        ),
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
            "output_frames": len(artifacts.positions),
            "left_source_index": left_source_index.tolist(),
            "right_source_index": right_source_index.tolist(),
            "source_alpha": source_alpha.tolist(),
        }],
        "validation": {
            "fk_max_error_m": max(
                float(report["fk_max_error_m"]) for report in reports),
            "duration_error_s": max(
                float(report["duration_error_s"]) for report in reports),
            "quaternion_norm_max_error": max(float(
                report["quaternion_norm_max_error"]) for report in reports),
        },
    }
    return artifacts, features, manifest_base


def _require_flat_data_v3(flat_data, g1_xml):
    flat_data = os.path.abspath(os.fspath(flat_data))
    manifest_path = os.path.join(flat_data, "manifest.json")
    _require_file(manifest_path, "accepted flat-v3 manifest")
    with open(manifest_path, "rb") as stream:
        manifest_bytes = stream.read()
    if hashlib.sha256(manifest_bytes).hexdigest() \
            != CANONICAL_FLAT_DATA_MANIFEST_SHA256:
        raise ValueError("accepted flat-v3 manifest SHA-256 changed")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("accepted flat-v3 manifest is invalid") from error
    if canonical_json_bytes(manifest) != manifest_bytes \
            or manifest.get("schema") != "g1-lmm-flat-data/v3" \
            or manifest.get("status") != "accepted":
        raise ValueError("accepted flat-v3 manifest schema/status changed")
    _authenticate_flat_manifest_inputs(manifest)
    artifact_descriptors = manifest.get("artifacts")
    if type(artifact_descriptors) is not dict \
            or set(artifact_descriptors) != {"database.bin", "features.bin"}:
        raise ValueError("accepted flat-v3 artifact descriptors changed")
    for name in ("database.bin", "features.bin"):
        descriptor = artifact_descriptors[name]
        path = os.path.join(flat_data, name)
        _require_file(path, f"accepted flat-v3 {name}")
        node = os.stat(path)
        if type(descriptor) is not dict \
                or descriptor != {
                    "path": name,
                    "size_bytes": node.st_size,
                    "sha256": sha256_file(path),
                }:
            raise ValueError(f"accepted flat-v3 {name} descriptor changed")

    source = manifest["sources"][0]
    rebuilt_artifacts, rebuilt_features, rebuilt_base = \
        _assemble_flat_candidate(SimpleNamespace(
            output_fps=60.0,
            retarget_npz=source["path"],
            retarget_receipt=source["receipt_path"],
            g1_xml=g1_xml,
        ))
    expected_base = dict(manifest)
    del expected_base["status"]
    del expected_base["artifacts"]
    if canonical_json_bytes(rebuilt_base) != canonical_json_bytes(expected_base):
        raise ValueError("independently rebuilt flat-v3 manifest changed")
    loaded_artifacts = read_holden_database(
        os.path.join(flat_data, "database.bin"))
    loaded_features = read_features(os.path.join(flat_data, "features.bin"))
    for field in (
        "positions", "velocities", "rotations", "angular_velocities",
        "parents", "range_starts", "range_stops", "contacts",
    ):
        if not np.array_equal(
                getattr(rebuilt_artifacts, field),
                getattr(loaded_artifacts, field)):
            raise ValueError(f"independently rebuilt flat-v3 {field} changed")
    for field in ("values", "offset", "scale"):
        if not np.array_equal(
                getattr(rebuilt_features, field), getattr(loaded_features, field)):
            raise ValueError(
                f"independently rebuilt flat-v3 feature {field} changed")
    return flat_data, manifest, rebuilt_artifacts, rebuilt_features, rebuilt_base


def _flat_artifact_clip(artifacts, flat_manifest):
    source = flat_manifest["sources"][0]
    clip = HoldenClip(
        "flat",
        np.array(artifacts.positions, np.float32, copy=True),
        np.array(artifacts.velocities, np.float32, copy=True),
        np.array(artifacts.rotations, np.float32, copy=True),
        np.array(artifacts.angular_velocities, np.float32, copy=True),
        np.array(artifacts.contacts, np.uint8, copy=True),
        np.array(artifacts.terrain_features, np.float32, copy=True),
        np.array(artifacts.terrain_support, np.float32, copy=True),
        np.array(source["left_source_index"], np.int32),
        "flat",
        np.array(source["left_source_index"], np.int32),
        np.array(source["right_source_index"], np.int32),
        np.array(source["source_alpha"], np.float32),
    )
    clip.validate()
    return clip


def _file_input_descriptor(path):
    path = os.path.abspath(os.fspath(path))
    _require_file(path, "terrain LMM input")
    return {
        "path": path,
        "size_bytes": os.stat(path).st_size,
        "sha256": sha256_file(path),
    }


def _preauthenticate_lmm_terrain_cli_inputs(args):
    flat_manifest = os.path.join(
        os.path.abspath(os.fspath(args.flat_data)), "manifest.json")
    expected = (
        (flat_manifest, None, CANONICAL_FLAT_DATA_MANIFEST_SHA256,
         "flat-v3 manifest"),
        (args.slope_robot, AUTHORED_SLOPE_ROBOT_SIZE_BYTES,
         AUTHORED_SLOPE_ROBOT_SHA256, "slope robot"),
        (args.slope_usd, AUTHORED_SLOPE_INPUTS["usd"]["size_bytes"],
         AUTHORED_SLOPE_INPUTS["usd"]["sha256"], "slope USD"),
        (args.slope_recon,
         AUTHORED_SLOPE_INPUTS["reconstruction"]["size_bytes"],
         AUTHORED_SLOPE_INPUTS["reconstruction"]["sha256"],
         "slope reconstruction"),
        (args.slope_metadata, AUTHORED_SLOPE_INPUTS["metadata"]["size_bytes"],
         AUTHORED_SLOPE_INPUTS["metadata"]["sha256"], "slope metadata"),
        (args.g1_xml, CANONICAL_FLAT_KINEMATICS_MODEL["size_bytes"],
         CANONICAL_FLAT_KINEMATICS_MODEL["sha256"], "canonical G1 XML"),
    )
    for path, size, digest, label in expected:
        _require_file(path, label)
        node = os.stat(path)
        if (size is not None and node.st_size != size) \
                or sha256_file(path) != digest:
            raise ValueError(f"{label} SHA-256/size changed")


def _authored_query_counts(clip, skeleton, terrain):
    gp, gq = forward_kinematics_arrays(
        clip.positions, clip.rotations, skeleton.parents)

    def count(points):
        exact = sum(
            bool(terrain.contains_projected_point(float(x), float(z)))
            for x, z in np.asarray(points, np.float64))
        return {"exact_mesh": exact, "flat_extension": len(points) - exact}

    feature_points = []
    for frame in range(len(gp)):
        stop = min(frame + 121, len(gp))
        path = gp[frame:stop, 0][:, [0, 2]]
        headings3 = holden_quat.mul_vec(
            gq[frame:stop, 0], np.array([0.0, 0.0, 1.0], np.float64))
        centerline = build_facing_centerline(
            path[0], headings3[:, [0, 2]], path)
        feature_points.append(centerline[0])
        for distance in TERRAIN_DISTANCES:
            remaining = float(distance)
            point = centerline[-1]
            for start, end in zip(centerline, centerline[1:]):
                length = float(np.linalg.norm(end - start))
                if remaining <= length:
                    point = start + (
                        remaining / max(length, 1e-8)) * (end - start)
                    break
                remaining -= length
            feature_points.append(point)
    support_indices = [
        skeleton.names.index(name)
        for name in ("Simulation", "LeftToe", "RightToe")
    ]
    support_points = gp[:, support_indices][:, :, [0, 2]].reshape(-1, 2)
    route_points = gp[:, 0][:, [0, 2]]
    return {
        "features": count(np.asarray(feature_points)),
        "support": count(support_points),
        "route": count(route_points),
    }


def _authored_scene_pack(scene_definition, terrain_object, receipt):
    source = build_scene(scene_definition)
    metadata = source.metadata
    receipt_sha = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    surface_signature = hashlib.sha256(
        canonical_json_bytes(terrain_object)).hexdigest()
    metadata["schema"] = "g1-lmm-terrain-scene/v1"
    metadata["surface_signature"] = surface_signature
    metadata["provenance"] = {
        "kind": "authenticated-grail",
        "source_ids": [AUTHORED_SLOPE_NAME],
        "parameters": {
            "receipt_schema": receipt["schema"],
            "source_receipt_sha256": receipt_sha,
            "route_source": "admitted-holden-simulation-path",
            "surface_source": "authenticated-exact-mesh",
            "exterior_policy": "explicit-flat-zero-before-calibration",
            "support_calibration_m": AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
            "support_calibration_applied_to": "terrain-only",
        },
    }
    payloads = {
        "heightfield": source.terrain_bin,
        "mesh": source.terrain_obj,
        "walkability": source.walkability_bin,
    }
    for name, payload in payloads.items():
        metadata[name]["size_bytes"] = len(payload)
    scene_json = canonical_json_bytes(metadata)
    built = BuiltScene(
        "authored-slope", scene_json, source.terrain_bin,
        source.terrain_obj, source.walkability_bin)
    index = {
        "schema": "g1-lmm-terrain-scene-index/v1",
        "default_scene_id": "authored-slope",
        "scene_ids": ["authored-slope"],
        "scenes": [{
            "id": "authored-slope",
            "path": "scenes/authored-slope/scene.json",
            "schema": "g1-lmm-terrain-scene/v1",
            "size_bytes": len(scene_json),
            "sha256": hashlib.sha256(scene_json).hexdigest(),
        }],
        "coordinate_signature":
            "holden-y-up-right-handed-forward-plus-z",
        "surface_signature": surface_signature,
    }
    return ScenePack(canonical_json_bytes(index), (built,))


def _assemble_lmm_terrain_candidate(args):
    if float(getattr(args, "output_fps", 0.0)) != 60.0:
        raise ValueError("authored-slope terrain bundle requires --output-fps 60")
    for attribute in (
        "flat_data", "slope_robot", "slope_usd", "slope_recon",
        "slope_metadata", "g1_xml",
    ):
        if not getattr(args, attribute, None):
            raise ValueError(
                "authored-slope terrain bundle requires --"
                + attribute.replace("_", "-"))

    _preauthenticate_lmm_terrain_cli_inputs(args)

    flat_data, flat_manifest, flat_artifacts, _flat_features, flat_base = \
        _require_flat_data_v3(args.flat_data, args.g1_xml)
    slope_candidate = _assemble_authored_slope_source(args)
    skeleton = SkeletonSpec(
        G1_SKELETON_NAMES, np.asarray(G1_SKELETON_PARENTS, np.int32))
    require_canonical_g1_skeleton(skeleton, "terrain LMM skeleton")
    flat_clip = refresh_lmm_clip_dynamics(
        _flat_artifact_clip(flat_artifacts, flat_manifest), skeleton, 60.0)
    slope_clip = refresh_lmm_clip_dynamics(
        slice_holden_clip(slope_candidate.source, 3), skeleton, 60.0)
    artifacts = combine_clips([flat_clip, slope_clip], skeleton)
    features = build_lmm_terrain_features(artifacts)

    receipt = json.loads(canonical_json_bytes(slope_candidate.provenance))
    receipt_sha = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    query_counts = _authored_query_counts(slope_clip, skeleton, slope_candidate.terrain)
    terrain_object = {
        "schema": "g1-lmm-authored-slope-surface/v1",
        "feature_distances_m": list(TERRAIN_DISTANCES),
        "source_receipt_sha256": receipt_sha,
        "basis": "z-up-to-holden-shared-with-robot",
        "support_calibration_m": AUTHORED_SLOPE_SUPPORT_CALIBRATION_M,
        "support_calibration_applied_to": "terrain-only",
        "exterior_policy": dict(receipt["exterior_policy"]),
        "query_counts": query_counts,
    }
    scene_pack = _authored_scene_pack(
        slope_candidate.scene, terrain_object, receipt)

    flat_source = flat_manifest["sources"][0]
    slope_map = receipt["interpolation"]
    sources = [{
        "name": "flat",
        "terrain_id": "flat",
        "range": {"start": 0, "stop": 256},
        "source_fps": 120.0,
        "source_frames": 512,
        "output_frames": 256,
        "source_map": {
            "left_source_index": list(flat_source["left_source_index"]),
            "right_source_index": list(flat_source["right_source_index"]),
            "source_alpha": list(flat_source["source_alpha"]),
        },
    }, {
        "name": AUTHORED_SLOPE_NAME,
        "terrain_id": AUTHORED_SLOPE_NAME,
        "range": {"start": 256, "stop": 851},
        "source_fps": 25.0,
        "source_frames": 250,
        "output_frames": 595,
        "source_map": {
            "left_source_index": list(slope_map["left_source_index"][3:]),
            "right_source_index": list(slope_map["right_source_index"][3:]),
            "source_alpha": list(slope_map["source_alpha"][3:]),
        },
    }]
    ranges = [
        {"start": 0, "stop": 256, "source_index": 0},
        {"start": 256, "stop": 851, "source_index": 1},
    ]
    range_digest = hashlib.sha256(canonical_json_bytes(ranges)).hexdigest()
    source_map_digest = hashlib.sha256(canonical_json_bytes([
        source["source_map"] for source in sources
    ])).hexdigest()
    raw_slope = load_authenticated_grail_slope(args.slope_robot)
    native_steps = np.max(np.abs(np.diff(raw_slope.qpos[:, 7:], axis=0)), axis=1)
    provisional_local_steps = _maximum_local_rotation_steps(
        slope_candidate.source.rotations)
    admitted_local_steps = _maximum_local_rotation_steps(slope_clip.rotations)
    continuity = {
        "schema": "g1-lmm-continuity/v1",
        "threshold_rad_per_frame": 0.25,
        "minimum_range_frames": 61,
        "source_native_rejected_edge_count": int(np.sum(native_steps > 0.25)),
        "database_local_rejected_edge_count": int(np.sum(
            provisional_local_steps > 0.25)),
        "union_rejected_edge_count": int(
            np.sum(native_steps > 0.25) + np.sum(provisional_local_steps > 0.25)),
        "dropped_fragment_count": 1,
        "dropped_frame_count": 3,
        "published_range_count": 2,
        "published_frame_count": 851,
        "maximum_admitted_native_step_rad": max(
            float(flat_base["continuity"]["maximum_admitted_native_step_rad"]),
            float(np.max(native_steps))),
        "maximum_admitted_local_rotation_step_rad": max(
            float(flat_base["continuity"]
                  ["maximum_admitted_local_rotation_step_rad"]),
            float(np.max(admitted_local_steps))),
        "range_digest_sha256": range_digest,
        "source_map_digest_sha256": source_map_digest,
    }
    temporal_partition = {
        "boundary_row": 256,
        "derivatives": "per-source",
        "contacts": "per-source",
        "feature_future": "clamp-at-range-stop",
        "cross_range_operations": 0,
    }
    contact_observations = _flat_contact_receipt(
        artifacts.contacts, artifacts.range_starts, artifacts.range_stops,
        median_filter_frames=6)
    feature_signature = _flat_feature_signature(
        FEATURE_NAMES, (20, 40, 60), FEATURE_WEIGHTS,
        features.offset, features.scale)
    inputs = {
        "flat_data": {
            "path": flat_data,
            "manifest_path": "manifest.json",
            "schema": "g1-lmm-flat-data/v3",
            "manifest_sha256": CANONICAL_FLAT_DATA_MANIFEST_SHA256,
        },
        "slope_robot": _file_input_descriptor(args.slope_robot),
        "slope_usd": _file_input_descriptor(args.slope_usd),
        "slope_recon": _file_input_descriptor(args.slope_recon),
        "slope_metadata": _file_input_descriptor(args.slope_metadata),
        "g1_xml": _file_input_descriptor(args.g1_xml),
    }
    manifest_base = {
        "schema": LMM_TERRAIN_SCHEMA,
        "output_fps": 60.0,
        "database_frames": 851,
        "source_count": 2,
        "range_count": 2,
        "trajectory_horizons": [20, 40, 60],
        "dimensions": {
            "bones": 31, "features": 31, "latent": 32, "contacts": 2,
        },
        "skeleton": {
            "names": list(G1_SKELETON_NAMES),
            "parents": list(G1_SKELETON_PARENTS),
            "basis": "holden-y-up-right-handed-forward-plus-z",
            "signature": G1_SKELETON_SIGNATURE,
        },
        "feature": {
            "names": list(FEATURE_NAMES),
            "weights": list(FEATURE_WEIGHTS),
            "offset": features.offset.tolist(),
            "scale": features.scale.tolist(),
            "signature": feature_signature,
            "terrain_indices": [27, 28, 29, 30],
        },
        "terrain": terrain_object,
        "support": {
            "schema": "G1SP/v1", "dimensions": 3,
            "columns": [
                "source_root_height_m", "source_left_toe_height_m",
                "source_right_toe_height_m",
            ],
        },
        "contact": {
            "semantics": "bundled-orange-duck-global-toe-speed-only",
            "speed_threshold": 0.15,
            "median_filter_frames": 6,
            "median_filter_mode": "nearest",
            "observations": contact_observations,
        },
        "time_filters": dict(flat_manifest["time_filters"]),
        "inputs": inputs,
        "slope_source_receipt": receipt,
        "sources": sources,
        "ranges": ranges,
        "continuity": continuity,
        "temporal_partition": temporal_partition,
    }
    validation = {
        "schema": "g1-lmm-terrain-data-validation/v1",
        "database": {
            "frames": 851, "bones": 31, "contacts": 2,
            "ranges": [[0, 256], [256, 851]],
        },
        "features": {
            "rows": 851, "dimensions": 31,
            "terrain_indices": [27, 28, 29, 30],
            "terrain_scale": features.scale[27:31].tolist(),
        },
        "sidecars": {
            "terrain_features": {"rows": 851, "dimensions": 4},
            "terrain_support": {
                "rows": 851, "dimensions": 3,
                "columns": list(manifest_base["support"]["columns"]),
            },
        },
        "partition": dict(temporal_partition),
        "inputs": {
            "flat_manifest_sha256": CANONICAL_FLAT_DATA_MANIFEST_SHA256,
            "slope_source_receipt_sha256": receipt_sha,
            "g1_xml_sha256": CANONICAL_FLAT_KINEMATICS_MODEL["sha256"],
        },
    }
    return LMMTerrainCandidate(
        artifacts, features, manifest_base, validation, scene_pack)


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
    _validate_build_mode_args(args)
    if getattr(args, "authored_slope_terrain", False):
        candidate = _assemble_lmm_terrain_candidate(args)
        # Task 3 adds the independent full-source validator. The Task 2
        # publisher itself performs its complete staged validation three times.
        return publish_lmm_terrain_artifacts(
            args.output, candidate.artifacts, candidate.features,
            candidate.manifest_base, candidate.validation,
            candidate.scene_pack, lambda staging: None)
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


def _validate_build_mode_args(args):
    terrain_mode = bool(getattr(args, "authored_slope_terrain", False))
    flat_mode = bool(getattr(args, "flat_only", False))
    terrain_fields = (
        "flat_data", "slope_robot", "slope_usd", "slope_recon",
        "slope_metadata",
    )
    if terrain_mode:
        for field in (*terrain_fields, "g1_xml"):
            if not getattr(args, field, None):
                raise ValueError(
                    "authored-slope terrain mode requires --"
                    + field.replace("_", "-"))
        if getattr(args, "retarget_npz", None) \
                or getattr(args, "retarget_receipt", None):
            raise ValueError(
                "authored-slope terrain mode rejects flat retarget options")
        if getattr(args, "grail_limit", None) is not None \
                or getattr(args, "grail_glob", DEFAULTS["grail_glob"]) \
                != DEFAULTS["grail_glob"] \
                or getattr(args, "takara", DEFAULTS["takara"]) \
                != DEFAULTS["takara"] \
                or getattr(args, "remap", DEFAULTS["remap"]) \
                != DEFAULTS["remap"]:
            raise ValueError(
                "authored-slope terrain mode rejects general source options")
        return
    if any(getattr(args, field, None) for field in terrain_fields):
        mode = "flat" if flat_mode else "general"
        raise ValueError(f"{mode} mode rejects terrain-only options")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build validated Holden G1 terrain motion artifacts")
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--output-fps", type=float, default=OUTPUT_FPS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--flat-only", action="store_true")
    modes.add_argument("--authored-slope-terrain", action="store_true")
    parser.add_argument("--retarget-npz")
    parser.add_argument("--retarget-receipt")
    parser.add_argument("--flat-data")
    parser.add_argument("--slope-robot")
    parser.add_argument("--slope-usd")
    parser.add_argument("--slope-recon")
    parser.add_argument("--slope-metadata")
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
    scenes = 1 if args.authored_slope_terrain else (
        0 if args.flat_only else len(REQUIRED_SCENE_IDS))
    clips = manifest.get("total_clips", manifest.get("source_count"))
    print(
        f"BUILT {manifest['schema']} frames={manifest['database_frames']} "
        f"clips={clips} scenes={scenes} "
        f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
