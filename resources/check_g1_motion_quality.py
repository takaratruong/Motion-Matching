#!/usr/bin/env python3
"""Certify deterministic full-pack pre-IK G1 motion-quality logs."""

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from resources.check_g1_runtime_log import (
    LOCKED_SCENE_IDS,
    check_gate_c,
    check_gate_f,
    read_rows,
)
from resources.summarize_g1_motion_quality import summarize_route
from resources.g1_terrain_builder.terrain import (
    _descriptor_height,
    classify_terrain_profile,
)


MATRIX_SCHEMA = "g1-motion-quality-matrix/v1"
REPORT_SCHEMA = "g1-motion-quality-report/v1"
RUN_ORIGIN_SCHEMA = "g1-motion-quality-run-origin/v1"
EVIDENCE_SCHEMA = "g1-motion-quality-evidence/v1"
CANONICAL_BASELINE_SHA256 = (
    "fc3e0e94173ee7156f5b5b0f3ff0a6da7abc98148a1749fbc94ca270b1a247bd")
FULL_CANDIDATE_SOURCES = 15_815
FULL_CANDIDATE_FRAMES = 3_970_932
SECTOR_MASKS = (0x002, 0x004, 0x008, 0x010,
                0x020, 0x040, 0x080, 0x100)
SECTOR_NAMES = (
    "forward", "forward-right", "right", "back-right",
    "backward", "back-left", "left", "forward-left",
)
POSITIVE_Z_HEADINGS = (
    "forward", "diagonal-negative-x", "negative-x",
    "backward-negative-x", "backward", "backward-positive-x",
    "positive-x", "diagonal-positive-x",
)
POSITIVE_X_HEADINGS = (
    "positive-x", "diagonal-positive-x", "forward",
    "diagonal-negative-x", "negative-x", "backward-negative-x",
    "backward", "backward-positive-x",
)
HEADING_VECTORS = {
    "forward": (0.0, 1.0),
    "diagonal-positive-x": (math.sqrt(0.5), math.sqrt(0.5)),
    "positive-x": (1.0, 0.0),
    "backward-positive-x": (math.sqrt(0.5), -math.sqrt(0.5)),
    "backward": (0.0, -1.0),
    "backward-negative-x": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "negative-x": (-1.0, 0.0),
    "diagonal-negative-x": (-math.sqrt(0.5), math.sqrt(0.5)),
}
CORRIDOR_OFFSET_M = 0.148506455
CLASSIFIER_CONFIDENCE_TOLERANCE = 0.125
GATE_C_ROUTE_IDS = {
    "slope-fixed-forward",
    "stair-fixed-forward",
    "gate-c-stairs-shallow-forward",
    "gate-c-ramp-05-forward",
    "stair-unseen-forward",
    "mixed-full-course",
}
SECTOR_FAMILIES = ("flat", "slope", "stair")
THRESHOLD_NAMES = (
    "flat_wrong_family_frames",
    "activated_terrain_wrong_family_frames",
    "lateral_selected_cost_p95",
    "lateral_transition_rate",
    "stance_slip_total_m",
    "sole_worst_penetration_m",
)


def _production_route_specs():
    specs = []
    for sector, name in enumerate(SECTOR_NAMES):
        specs.append({
            "id": f"flat-fixed-{name}", "scene": "stairs-shallow",
            "route": "flat-positive-z", "frames": 100,
            "expected_family": "flat", "sector": sector,
            "heading_case": "fixed", "heading": POSITIVE_Z_HEADINGS[sector],
        })
        changed = {
            "id": f"flat-changed-{name}", "scene": "stairs-shallow",
            "route": "flat-positive-x", "frames": 100,
            "expected_family": "flat", "sector": sector,
            "heading_case": "changed", "heading": POSITIVE_X_HEADINGS[sector],
        }
        if sector == 2:
            changed["comparison_id"] = "stairs-shallow/flat-positive-x"
        specs.append(changed)
    specs[0]["comparison_id"] = "stairs-shallow/flat-positive-z"

    slope_scenes = (
        "ramp-10-up-down", "ramp-10-up-down", "ramp-10-up-down",
        "cross-slope-10", "ramp-10-up-down", "cross-slope-10",
        "cross-slope-10", "cross-slope-10",
    )
    for sector, (name, scene) in enumerate(zip(SECTOR_NAMES, slope_scenes)):
        spec = {
            "id": f"slope-fixed-{name}", "scene": scene,
            "route": ("up-landing-down" if scene == "ramp-10-up-down"
                      else "forward-cross-slope"),
            "frames": 800, "expected_family": "slope", "sector": sector,
            "heading_case": "fixed", "heading": POSITIVE_Z_HEADINGS[sector],
        }
        if sector == 0:
            spec["comparison_id"] = "ramp-10-up-down/up-landing-down"
        specs.append(spec)

    for sector, name in enumerate(SECTOR_NAMES):
        spec = {
            "id": f"stair-fixed-{name}", "scene": "stairs-standard",
            "route": "ascent-landing-descent", "frames": 800,
            "expected_family": "stair", "sector": sector,
            "heading_case": "fixed", "heading": POSITIVE_Z_HEADINGS[sector],
        }
        if sector == 0:
            spec["comparison_id"] = (
                "stairs-standard/ascent-landing-descent")
        elif sector == 6:
            spec["comparison_id"] = (
                "stairs-standard/ascent-landing-descent@positive-x")
        specs.append(spec)
    specs.extend((
        {
            "id": "gate-c-stairs-shallow-forward",
            "scene": "stairs-shallow", "route": "ascent-landing-descent",
            "frames": 800, "expected_family": "stair", "sector": 0,
            "heading_case": "fixed", "heading": "forward",
        },
        {
            "id": "gate-c-ramp-05-forward", "scene": "ramp-05-up-down",
            "route": "up-landing-down", "frames": 800,
            "expected_family": "slope", "sector": 0,
            "heading_case": "fixed", "heading": "forward",
        },
        {
            "id": "stair-unseen-forward", "scene": "stairs-unseen-variable",
            "route": "ascent-landing-descent", "frames": 800,
            "expected_family": "stair", "sector": 0,
            "heading_case": "fixed", "heading": "forward",
        },
        {
            "id": "mixed-tangent", "scene": "mixed-multilevel",
            "route": "tangent-level-boundary", "frames": 800,
            "expected_family": "mixed", "sector": 0,
            "heading_case": "fixed", "heading": "forward",
            "comparison_id": "mixed-multilevel/tangent-level-boundary",
        },
        {
            "id": "mixed-full-course", "scene": "mixed-multilevel",
            "route": "full-course", "frames": 800,
            "expected_family": "mixed", "sector": 0,
            "heading_case": "fixed", "heading": "forward",
        },
    ))
    return tuple(specs)


PRODUCTION_ROUTE_SPECS = _production_route_specs()
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path, label, *, executable=False):
    try:
        information = os.lstat(path)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(information.st_mode) or (
            executable and not os.access(path, os.X_OK)):
        kind = "regular executable" if executable else "regular file"
        raise ValueError(f"{label} must be a {kind}")
    return pathlib.Path(path)


def _authenticated_relative_file(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    relative_path = pathlib.Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"{label} path must be relative")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes candidate artifacts") from error
    return _regular_file(path, label)


def _read_authenticated_json(path, expected_sha256, label):
    _regular_file(path, label)
    try:
        payload = pathlib.Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    return value, digest


def _load_authenticated_candidate_artifacts(
        artifact_directory, manifest_sha256):
    """Load checker oracles through the candidate's authenticated hash chain."""
    if not isinstance(artifact_directory, str) or not os.path.isabs(
            artifact_directory):
        raise ValueError("candidate artifact directory must be absolute")
    root = pathlib.Path(artifact_directory).resolve()
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(
            manifest_sha256):
        raise ValueError("candidate manifest SHA-256 is invalid")
    manifest_path = _regular_file(root / "manifest.json", "candidate manifest")
    manifest, _ = _read_authenticated_json(
        manifest_path, manifest_sha256, "candidate manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != \
            "g1-terrain-artifacts/v3":
        raise ValueError("candidate manifest schema is invalid")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("candidate manifest sources are unavailable")
    authenticated_sources = []
    previous_stop = None
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"candidate source {index} is invalid")
        required = {
            "name", "terrain_id", "terrain_family",
            "range_start", "range_stop",
        }
        if not required.issubset(source):
            raise ValueError(f"candidate source {index} metadata is incomplete")
        start = source["range_start"]
        stop = source["range_stop"]
        if (type(start) is not int or type(stop) is not int or
                start < 0 or stop <= start or
                (previous_stop is None and start != 0) or
                (previous_stop is not None and start != previous_stop) or
                not all(isinstance(source[name], str) and source[name]
                        for name in ("name", "terrain_id", "terrain_family")) or
                source["terrain_family"] not in
                ("flat", "curb", "slope", "stair")):
            raise ValueError(f"candidate source {index} metadata is invalid")
        authenticated_sources.append({
            "index": index,
            "name": source["name"],
            "terrain_id": source["terrain_id"],
            "terrain_family": source["terrain_family"],
            "range_start": start,
            "range_stop": stop,
        })
        previous_stop = stop

    scene_index = manifest.get("scene_index")
    if (not isinstance(scene_index, dict) or
            scene_index.get("schema") != "g1-terrain-scene-index/v1" or
            not isinstance(scene_index.get("sha256"), str) or
            not _SHA256.fullmatch(scene_index["sha256"])):
        raise ValueError("candidate scene index identity is invalid")
    index_path = _authenticated_relative_file(
        root, scene_index.get("path"), "candidate scene index")
    index, index_sha256 = _read_authenticated_json(
        index_path, scene_index["sha256"], "candidate scene index")
    descriptors = index.get("scenes") if isinstance(index, dict) else None
    if not isinstance(descriptors, list):
        raise ValueError("candidate scene index descriptors are invalid")
    descriptor_by_id = {}
    for descriptor in descriptors:
        if (not isinstance(descriptor, dict) or
                set(descriptor) != {"id", "path", "sha256"} or
                not isinstance(descriptor["id"], str) or
                descriptor["id"] in descriptor_by_id or
                not isinstance(descriptor["sha256"], str) or
                not _SHA256.fullmatch(descriptor["sha256"])):
            raise ValueError("candidate scene index descriptor is invalid")
        descriptor_by_id[descriptor["id"]] = descriptor

    required_scenes = {spec["scene"] for spec in PRODUCTION_ROUTE_SPECS}
    if not required_scenes.issubset(descriptor_by_id):
        raise ValueError("candidate scene index lacks a production scene")
    from resources.validate_g1_terrain_database import _parse_heightfield

    scenes = {}
    for scene_id in sorted(required_scenes):
        descriptor = descriptor_by_id[scene_id]
        scene_path = _authenticated_relative_file(
            root, descriptor["path"], f"{scene_id} scene metadata")
        scene, scene_sha256 = _read_authenticated_json(
            scene_path, descriptor["sha256"], f"{scene_id} scene metadata")
        if (not isinstance(scene, dict) or
                scene.get("schema") != "g1-terrain-scene/v1" or
                scene.get("id") != scene_id or
                not isinstance(scene.get("heightfield"), dict) or
                not isinstance(scene.get("routes"), list)):
            raise ValueError(f"{scene_id} authenticated scene is invalid")
        heightfield = scene["heightfield"]
        terrain_path = _authenticated_relative_file(
            scene_path.parent, heightfield.get("path"),
            f"{scene_id} heightfield")
        try:
            grid = _parse_heightfield(terrain_path, heightfield)
        except ValueError as error:
            raise ValueError(
                f"{scene_id} heightfield SHA-256 or format invalid: {error}") \
                from error
        routes = {}
        for route in scene["routes"]:
            if (not isinstance(route, dict) or
                    not isinstance(route.get("id"), str) or
                    route["id"] in routes or
                    not isinstance(route.get("waypoints_xz"), list) or
                    len(route["waypoints_xz"]) < 2 or
                    isinstance(route.get("landing_hold_seconds"), bool) or
                    not isinstance(route.get("landing_hold_seconds"),
                                   (int, float))):
                raise ValueError(f"{scene_id} route metadata is invalid")
            points = []
            for point in route["waypoints_xz"]:
                if (not isinstance(point, list) or len(point) != 2 or
                        any(isinstance(value, bool) or
                            not isinstance(value, (int, float)) or
                            not math.isfinite(float(value)) for value in point)):
                    raise ValueError(f"{scene_id} route waypoint is invalid")
                points.append(tuple(float(value) for value in point))
            routes[route["id"]] = {
                "id": route["id"],
                "landing_hold_seconds": float(
                    route["landing_hold_seconds"]),
                "waypoints_xz": tuple(points),
            }
        scenes[scene_id] = {
            "metadata": scene,
            "scene_sha256": scene_sha256,
            "heightfield_sha256": heightfield["sha256"],
            "grid": grid,
            "routes": routes,
        }
    for spec in PRODUCTION_ROUTE_SPECS:
        if spec["route"] not in scenes[spec["scene"]]["routes"]:
            raise ValueError(
                f"candidate scene lacks production route {spec['route']!r}")
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "scene_index_sha256": index_sha256,
        "sources": tuple(authenticated_sources),
        "source_starts": tuple(
            source["range_start"] for source in authenticated_sources),
        "scenes": scenes,
    }


def _resolve_authenticated_source_frame(artifacts, frame, label):
    value = _integer(frame, label)
    try:
        index = bisect_right(artifacts["source_starts"], value) - 1
        source = artifacts["sources"][index] if index >= 0 else None
    except (KeyError, TypeError) as error:
        raise ValueError("authenticated source index is unavailable") from error
    if (source is None or value < source["range_start"] or
            value >= source["range_stop"]):
        raise ValueError(f"{label} {value} has no authenticated source")
    return source


def _resolve_authenticated_source(
        artifacts, *, selected_frame, current_frame):
    source = _resolve_authenticated_source_frame(
        artifacts, selected_frame, "selected database frame")
    current_source = _resolve_authenticated_source_frame(
        artifacts, current_frame, "current database frame")
    if current_source["index"] != source["index"]:
        raise ValueError(
            "current database frame differs from authenticated source")
    return source


def _runtime_height(grid, x, z):
    return _descriptor_height(
        grid, np.asarray((float(x), float(z)), np.float64))


def _f32_bits(value):
    encoded = np.asarray(np.float32(float(value)))
    return int(encoded.view(np.uint32))


def _same_f32(first, second):
    return _f32_bits(first) == _f32_bits(second)


def _runtime_height_difference(first, second):
    first_f32 = _f32(first)
    second_f32 = _f32(second)
    return _f32(first_f32 - second_f32)


def _authenticated_route(artifacts, spec):
    if not isinstance(artifacts, dict):
        raise ValueError("authenticated candidate artifacts are unavailable")
    try:
        scene = artifacts["scenes"][spec["scene"]]
        route = scene["routes"][spec["route"]]
    except (KeyError, TypeError) as error:
        raise ValueError("authenticated production route is unavailable") \
            from error
    return scene, route


def _f32(value):
    result = float(np.float32(value))
    if not math.isfinite(result):
        raise ValueError("binary32 route arithmetic overflowed")
    return 0.0 if result == 0.0 else result


def _route_schedule(route):
    """Reconstruct route timing, waypoint, completion, and commanded speed."""
    step = _f32(_f32(.50) * _f32(.04))
    segments = []
    for start, stop in zip(
            route["waypoints_xz"], route["waypoints_xz"][1:]):
        dx = _f32(_f32(stop[0]) - _f32(start[0]))
        dz = _f32(_f32(stop[1]) - _f32(start[1]))
        squared = _f32(_f32(dx * dx) + _f32(dz * dz))
        length = _f32(math.sqrt(squared))
        if length <= 1.0e-6:
            raise ValueError("authenticated route has a degenerate segment")
        frames = max(1, math.ceil(_f32(length / step)))
        segments.append({
            "length": length,
            "frames": frames,
        })
    hold_seconds = route["landing_hold_seconds"]
    hold_frames = (math.ceil(_f32(_f32(hold_seconds) / _f32(.04)))
                   if hold_seconds else 0)
    return tuple(segments), hold_frames


def _route_schedule_sample(route, frame):
    if type(frame) is not int or frame < 0:
        raise ValueError("route schedule frame is invalid")
    segments, hold_frames = _route_schedule(route)
    cursor = 0
    for index, segment in enumerate(segments):
        if frame < cursor + segment["frames"]:
            return {
                "waypoint": index + 1,
                "complete": 0,
                "speed": .5,
            }
        cursor += segment["frames"]
        if index + 1 == 2 and hold_frames:
            if frame < cursor + hold_frames:
                return {
                    "waypoint": 2, "complete": 0,
                    "speed": 0.0,
                }
            cursor += hold_frames
    return {
        "waypoint": len(route["waypoints_xz"]) - 1,
        "complete": 1,
        "speed": 0.0,
    }


def _route_heading(spec, route, frame):
    del route, frame
    return HEADING_VECTORS[spec["heading"]]


def _logged_profile_classification(grid, row):
    points = [
        (float(row["terrain_root_point_x"]),
         float(row["terrain_root_point_z"])),
    ]
    points.extend(
        (float(row[f"terrain_center_point{sample}_x"]),
         float(row[f"terrain_center_point{sample}_z"]))
        for sample in range(8))
    distances = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    sampled = []
    for distance in distances:
        scaled = min(8.0, float(distance) / .125)
        start = min(7, int(math.floor(scaled)))
        alpha = scaled - start
        sampled.append((
            points[start][0] + alpha * (
                points[start + 1][0] - points[start][0]),
            points[start][1] + alpha * (
                points[start + 1][1] - points[start][1]),
        ))
    heights = np.asarray([
        _runtime_height(grid, x, z) for x, z in sampled
    ], np.float64)
    normals = np.asarray([
        np.asarray(grid.normal(*np.asarray((x, z), np.float32)),
                   np.float32).astype(np.float64)
        for x, z in sampled
    ], np.float64)
    return classify_terrain_profile(distances, heights, normals)


def _advance_bank_state(state, family, elevation, confident):
    current, current_elevation, pending, pending_elevation, pending_count = \
        state
    transitioned = 0
    if current is None:
        pending, pending_elevation, pending_count = None, 0, 0
        if confident:
            current, current_elevation = family, elevation
            transitioned = 1
            reason = "initial_confident"
        else:
            reason = "pending_cleared_low_confidence"
    elif (family, elevation) == (current, current_elevation):
        pending, pending_elevation, pending_count = None, 0, 0
        reason = "retained_same"
    elif not confident:
        pending, pending_elevation, pending_count = None, 0, 0
        reason = "pending_cleared_low_confidence"
    elif (pending, pending_elevation) != (family, elevation):
        pending, pending_elevation, pending_count = family, elevation, 1
        reason = "pending_started"
    else:
        pending_count += 1
        if pending_count < 2:
            reason = "pending_advanced"
        else:
            current, current_elevation = pending, pending_elevation
            pending, pending_elevation, pending_count = None, 0, 0
            transitioned = 1
            reason = "confirmed"
    return (
        (current, current_elevation, pending, pending_elevation,
         pending_count),
        (family, current, current_elevation, transitioned, reason),
    )


def _authenticated_bank_trace(rows, grid):
    possible = {(None, 0, None, 0, 0)}
    trace = []
    for index, row in enumerate(rows):
        classification = _logged_profile_classification(grid, row)
        reconstructed = float(classification.confidence)
        logged = _finite(
            row["classifier_confidence"],
            f"row {index} classifier confidence")
        confidence_valid = abs(logged - reconstructed) <= \
            CLASSIFIER_CONFIDENCE_TOLERANCE + 1.0e-9
        lower = max(0.0, reconstructed - CLASSIFIER_CONFIDENCE_TOLERANCE)
        upper = min(1.0, reconstructed + CLASSIFIER_CONFIDENCE_TOLERANCE)
        # The authenticated reconstruction bounds telemetry plausibility.  The
        # accepted logged scalar is still the exact value consumed by runtime,
        # so its side of the 0.60 threshold owns the deterministic state branch.
        confidence_branches = [logged >= .60]
        candidates = []
        for state in possible:
            for confident in confidence_branches:
                candidates.append(_advance_bank_state(
                    state, classification.family,
                    classification.elevation_mode, confident))
        actual = (
            row["requested_family"], row["active_family"],
            _integer(row["elevation_mode"], f"row {index} elevation mode"),
            _integer(row["bank_transition"], f"row {index} bank transition"),
            row["bank_transition_reason"],
        )
        matching = {state for state, output in candidates if output == actual}
        state_valid = bool(matching)
        if matching:
            possible = matching
        else:
            possible = {state for state, _ in candidates}
        trace.append({
            "requested_family": classification.family,
            "requested_elevation": classification.elevation_mode,
            "reconstructed_confidence": reconstructed,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "confidence_valid": confidence_valid,
            "state_valid": state_valid,
        })
    return trace


def _validate_canonical_baseline(baseline, payload):
    if not isinstance(payload, bytes):
        raise ValueError("canonical baseline payload must be bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CANONICAL_BASELINE_SHA256:
        raise ValueError("canonical baseline SHA-256 mismatch")
    try:
        parsed = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid canonical baseline JSON: {error}") from error
    if parsed != baseline:
        raise ValueError("parsed canonical baseline differs from supplied baseline")
    _thresholds(baseline)
    return digest


def _validate_candidate_identity(matrix, validator):
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be a JSON object")
    candidate = matrix.get("candidate")
    required_candidate = {
        "artifact_directory", "manifest_sha256",
        "binary_path", "binary_sha256",
    }
    if not isinstance(candidate, dict) or set(candidate) != required_candidate:
        raise ValueError("matrix candidate identity is incomplete")
    artifact_directory = candidate["artifact_directory"]
    binary_path = candidate["binary_path"]
    if not isinstance(artifact_directory, str) or not os.path.isabs(
            artifact_directory):
        raise ValueError("candidate artifact directory must be absolute")
    if not isinstance(binary_path, str) or not os.path.isabs(binary_path):
        raise ValueError("candidate binary path must be absolute")
    _regular_file(binary_path, "candidate binary", executable=True)
    for name in ("manifest_sha256", "binary_sha256"):
        if not isinstance(candidate[name], str) or not _SHA256.fullmatch(
                candidate[name]):
            raise ValueError(f"candidate {name} is invalid")

    environment = matrix.get("environment")
    expected_environment = {
        "G1_TERRAIN_DIR": artifact_directory,
        "MM_TERRAIN_WEIGHT": "4",
        "MM_IK": "0",
    }
    if environment != expected_environment:
        raise ValueError(
            "matrix environment must bind candidate path, terrain weight 4, "
            "and IK 0")

    manifest_path = os.path.join(artifact_directory, "manifest.json")
    try:
        manifest_payload = pathlib.Path(manifest_path).read_bytes()
        manifest = json.loads(
            manifest_payload, object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read candidate manifest: {error}") from error
    actual_manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    if actual_manifest_sha != candidate["manifest_sha256"]:
        raise ValueError("candidate manifest SHA-256 mismatch")
    if not isinstance(manifest, dict) or manifest.get("schema") != \
            "g1-terrain-artifacts/v3":
        raise ValueError("candidate manifest schema is invalid")
    if manifest.get("diagnostic_mode") is not False:
        raise ValueError("candidate pack must be non-diagnostic")
    if manifest.get("total_clips") != FULL_CANDIDATE_SOURCES:
        raise ValueError("candidate source total is not the locked full total")
    if manifest.get("database_frames") != FULL_CANDIDATE_FRAMES:
        raise ValueError("candidate frame total is not the locked full total")

    try:
        actual_binary_sha = _sha256_file(binary_path)
    except OSError as error:
        raise ValueError(f"cannot read candidate binary: {error}") from error
    if actual_binary_sha != candidate["binary_sha256"]:
        raise ValueError("candidate binary SHA-256 mismatch")
    if not callable(validator):
        raise ValueError("candidate validator is unavailable")
    validation = validator(artifact_directory, full_source_validation=True)
    expected_validation = {
        "frames": FULL_CANDIDATE_FRAMES,
        "clips": FULL_CANDIDATE_SOURCES,
        "bones": 31,
        "scenes": 14,
    }
    if (not isinstance(validation, dict) or
            any(validation.get(key) != value
                for key, value in expected_validation.items()) or
            type(validation.get("source_rows")) is not int or
            validation["source_rows"] <= 0):
        raise ValueError(
            "candidate full-source validator summary does not match the full pack")
    return {
        "artifact_directory": artifact_directory,
        "manifest_sha256": actual_manifest_sha,
        "binary_path": binary_path,
        "binary_sha256": actual_binary_sha,
        "environment": dict(expected_environment),
        "manifest_schema": manifest["schema"],
        "sources": manifest["total_clips"],
        "frames": manifest["database_frames"],
        "diagnostic_mode": manifest["diagnostic_mode"],
        "validator_passed": True,
        "validator_summary": dict(validation),
    }


def canonical_json(report):
    return json.dumps(
        report,
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ) + "\n"


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite numeric")
    return number


def _integer(value, label):
    number = _finite(value, label)
    if number != int(number):
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _thresholds(baseline):
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be a JSON object")
    if baseline.get("schema") != "g1-motion-quality-baseline/v1":
        raise ValueError("unsupported motion-quality baseline schema")
    if baseline.get("ik_enabled") is not False:
        raise ValueError("baseline must declare IK disabled")
    if _integer(baseline.get("sample_rate_hz"), "baseline sample_rate_hz") != 25:
        raise ValueError("baseline sample_rate_hz must be 25")
    thresholds = baseline.get("candidate_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(THRESHOLD_NAMES):
        raise ValueError("baseline candidate thresholds are incomplete")
    for name in THRESHOLD_NAMES:
        if not isinstance(thresholds[name], dict):
            raise ValueError(f"baseline threshold {name} must be an object")
    for name in ("flat_wrong_family_frames",
                 "activated_terrain_wrong_family_frames"):
        threshold = thresholds[name]
        if threshold.get("comparison") != "equal":
            raise ValueError(f"baseline threshold {name} comparison changed")
        _integer(threshold.get("target"), f"baseline threshold {name} target")
        routes = threshold.get("routes")
        if not isinstance(routes, list) or not routes or not all(
                isinstance(route, str) and route for route in routes):
            raise ValueError(f"baseline threshold {name} routes are invalid")
    for name in THRESHOLD_NAMES[2:]:
        threshold = thresholds[name]
        if threshold.get("comparison") != "strictly_less_than_baseline":
            raise ValueError(f"baseline threshold {name} comparison changed")
        _finite(threshold.get("baseline"),
                f"baseline threshold {name} baseline")
    for name in ("lateral_selected_cost_p95", "lateral_transition_rate"):
        if not isinstance(thresholds[name].get("route"), str) or not \
                thresholds[name]["route"]:
            raise ValueError(f"baseline threshold {name} route is invalid")
    return thresholds


def _route_specs(matrix):
    if not isinstance(matrix, dict):
        raise ValueError("matrix must be a JSON object")
    expected_matrix_keys = {
        "schema", "sample_rate_hz", "candidate", "environment",
        "routes", "scene_cycle",
    }
    if set(matrix) != expected_matrix_keys:
        raise ValueError("matrix key set differs from the production contract")
    if "candidate_thresholds" in matrix or "thresholds" in matrix:
        raise ValueError("candidate matrix may not provide thresholds")
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise ValueError("unsupported motion-quality matrix schema")
    if _integer(matrix.get("sample_rate_hz"), "matrix sample_rate_hz") != 25:
        raise ValueError("matrix sample_rate_hz must be 25")
    routes = matrix.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("matrix routes must be a nonempty list")
    observed_ids = [route.get("id") if isinstance(route, dict) else None
                    for route in routes]
    expected_ids = {spec["id"] for spec in PRODUCTION_ROUTE_SPECS}
    if (len(routes) != len(PRODUCTION_ROUTE_SPECS) or
            set(observed_ids) != expected_ids or
            len(observed_ids) != len(set(observed_ids))):
        raise ValueError("matrix must contain the exact production route roster")
    evidence = {}
    for index, route in enumerate(routes):
        expected_route_keys = {
            "id", "log", "log_sha256", "origin", "origin_sha256",
        }
        if not isinstance(route, dict) or set(route) != expected_route_keys:
            raise ValueError(
                f"matrix route {index} must contain hash-bound log and "
                "run-origin evidence only")
        identifier = route["id"]
        if (not all(isinstance(route[name], str) and route[name]
                    for name in ("log", "origin")) or
                not all(isinstance(route[name], str) and
                        _SHA256.fullmatch(route[name])
                        for name in ("log_sha256", "origin_sha256"))):
            raise ValueError(
                f"matrix route {identifier!r} hash/origin evidence is invalid")
        evidence[identifier] = {
            name: route[name] for name in expected_route_keys - {"id"}
        }
    for field in ("log", "origin"):
        values = [item[field] for item in evidence.values()]
        if len(set(values)) != len(values):
            raise ValueError(
                f"matrix route evidence {field} paths must be unique")
    scene_cycle = matrix.get("scene_cycle")
    expected_scene_keys = {
        "log", "log_sha256", "cleanup", "cleanup_sha256",
        "origin", "origin_sha256",
    }
    if (not isinstance(scene_cycle, dict) or
            set(scene_cycle) != expected_scene_keys or
            not all(isinstance(scene_cycle[name], str) and scene_cycle[name]
                    for name in ("log", "cleanup", "origin")) or
            not all(isinstance(scene_cycle[name], str) and
                    _SHA256.fullmatch(scene_cycle[name])
                    for name in (
                        "log_sha256", "cleanup_sha256", "origin_sha256"))):
        raise ValueError(
            "matrix scene-cycle evidence hash/origin binding is incomplete")
    return tuple({**spec, **evidence[spec["id"]]}
                 for spec in PRODUCTION_ROUTE_SPECS)


def _validate_run_origin(spec, candidate, log_path, origin):
    expected_candidate_keys = {
        "artifact_directory", "manifest_sha256",
        "binary_path", "binary_sha256",
    }
    if (not isinstance(candidate, dict) or
            set(candidate) != expected_candidate_keys):
        raise ValueError("run origin candidate identity is invalid")
    log_path = pathlib.Path(log_path).resolve()
    environment = {
        "G1_TERRAIN_DIR": candidate["artifact_directory"],
        "MM_TERRAIN_WEIGHT": "4",
        "MM_IK": "0",
        "MM_TERRAIN_SCENE": spec["scene"],
        "MM_TEST_MODE": "route",
        "MM_TEST_ROUTE": spec["route"],
        "MM_TEST_FRAMES": str(spec["frames"]),
        "MM_TEST_HEADING": spec["heading"],
        "MM_LOG": str(log_path),
    }
    expected = {
        "schema": RUN_ORIGIN_SCHEMA,
        "binary_sha256": candidate["binary_sha256"],
        "manifest_sha256": candidate["manifest_sha256"],
        "argv": [candidate["binary_path"]],
        "environment": environment,
    }
    if (not isinstance(origin, dict) or origin != expected or
            any(type(origin.get(name)) is not type(value)
                for name, value in expected.items())):
        raise ValueError(
            f"run origin for route {spec['id']!r} differs from the exact "
            "candidate invocation")
    return origin


def _validate_scene_cycle_origin(
        candidate, log_path, cleanup_path, origin):
    expected = {
        "schema": RUN_ORIGIN_SCHEMA,
        "binary_sha256": candidate["binary_sha256"],
        "manifest_sha256": candidate["manifest_sha256"],
        "argv": [candidate["binary_path"]],
        "environment": {
            "G1_TERRAIN_DIR": candidate["artifact_directory"],
            "MM_TERRAIN_WEIGHT": "4",
            "MM_IK": "0",
            "MM_TEST_MODE": "scene-cycle",
            "MM_SCENE_DWELL_FRAMES": "25",
            "MM_TEST_FRAMES": "700",
            "MM_LOG": str(pathlib.Path(log_path).resolve()),
            "MM_CLEANUP_LOG": str(pathlib.Path(cleanup_path).resolve()),
        },
    }
    if origin != expected:
        raise ValueError(
            "scene-cycle run origin differs from the exact candidate invocation")
    return origin


def _polyline_lengths(points):
    lengths = tuple(math.hypot(stop[0] - start[0], stop[1] - start[1])
                    for start, stop in zip(points, points[1:]))
    return lengths, sum(lengths)


def _project_route(points, x, z):
    lengths, _ = _polyline_lengths(points)
    best = None
    prefix = 0.0
    segments = tuple(zip(points, points[1:]))
    for segment_index, (length, (start, stop)) in enumerate(
            zip(lengths, segments)):
        dx, dz = stop[0] - start[0], stop[1] - start[1]
        raw_alpha = ((x - start[0]) * dx + (z - start[1]) * dz) / (
            length * length)
        alpha = max(0.0, raw_alpha)
        if segment_index + 1 != len(segments):
            alpha = min(1.0, alpha)
        px, pz = start[0] + alpha * dx, start[1] + alpha * dz
        distance = math.hypot(x - px, z - pz)
        candidate = (distance, prefix + alpha * length)
        if best is None or candidate < best:
            best = candidate
        prefix += length
    return best


def _route_metric(spec, rows, artifacts=None):
    identifier = spec["id"]
    failures = []
    scene = None
    route = None
    grid = None
    expected_trace = []
    source_mismatches = []
    source_claim_mismatches = []
    try:
        scene, route = _authenticated_route(artifacts, spec)
        grid = scene["grid"]
        expected_trace = _authenticated_bank_trace(rows, grid)
        for index, row in enumerate(rows):
            source = _resolve_authenticated_source(
                artifacts,
                selected_frame=row["selected_database_frame"],
                current_frame=row["database_frame"])
            query_frame = _integer(
                row["query_database_frame"], f"row {index} query frame")
            query_source = _resolve_authenticated_source_frame(
                artifacts, query_frame, "query database frame")
            claimed = (
                _integer(row["source_index"],
                         f"row {index} source index"),
                _integer(row["range"], f"row {index} pose range"),
                _integer(row["source_range"],
                         f"row {index} source range"),
                row["source_name"], row["source_terrain"],
                row["source_family"],
            )
            authenticated = (
                source["index"], source["index"], source["index"],
                source["name"], source["terrain_id"],
                source["terrain_family"],
            )
            query_range = _integer(
                row["query_range"], f"row {index} query range")
            if (claimed != authenticated or
                    query_range != query_source["index"]):
                source_claim_mismatches.append(index)
            if source["terrain_family"] != row["active_family"]:
                source_mismatches.append(index)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"authenticated physical oracle invalid: {error}")
    try:
        summary = summarize_route(rows)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "passed": False,
            "failures": failures + [f"runtime log invalid: {error}"],
            "expected_family": spec["expected_family"],
            "expected_direction_mask": SECTOR_MASKS[spec["sector"]],
            "heading": spec["heading"],
            "bank_correctness": {
                "activation_frame": None,
                "wrong_family_frames": len(source_mismatches),
                "phase_mismatched_frames": 0,
                "passed": not source_mismatches,
            },
        }

    if len(rows) != spec["frames"]:
        failures.append(f"frame count {len(rows)} does not equal locked "
                        f"production duration {spec['frames']}")
    for index, row in enumerate(rows):
        if row.get("scene_id") != spec["scene"] or \
                row.get("mode") != "route" or row.get("route") != spec["route"]:
            failures.append(f"row {index}: production scene/mode/route mismatch")
            break

    schedule_mismatches = []
    progress_mismatches = []
    previous_progress = None
    if route is not None:
        for index, row in enumerate(rows):
            expected = _route_schedule_sample(route, index)
            commanded = _finite(
                row["commanded_speed"],
                f"route {identifier!r} row {index} command")
            actual_schedule = (
                _integer(row["frame"], f"row {index} frame"),
                _integer(row["route_waypoint"],
                         f"row {index} route waypoint"),
                _integer(row["route_complete"],
                         f"row {index} route complete"),
                _integer(row["scene_generation"],
                         f"row {index} scene generation"),
                _integer(row["scene_frame"], f"row {index} scene frame"),
                _integer(row["scene_reset_count"],
                         f"row {index} reset count"),
                _integer(row["scene_switch_failed"],
                         f"row {index} scene switch flag"),
                _integer(row["motion_pack_load_count"],
                         f"row {index} motion-pack count"),
                _integer(row["model_load_count"],
                         f"row {index} model-load count"),
                _integer(row["model_unload_count"],
                         f"row {index} model-unload count"),
                _integer(row["live_model_count"],
                         f"row {index} live-model count"),
            )
            wanted_schedule = (
                index, expected["waypoint"], expected["complete"],
                0, index, 1, 0, 1, 1, 0, 1,
            )
            if (actual_schedule != wanted_schedule or
                    abs(commanded - expected["speed"]) > 2.0e-6 or
                    abs(_finite(row["fixed_dt"], f"row {index} fixed dt") -
                        _f32(.04)) > 2.0e-9):
                schedule_mismatches.append(index)
            projection = _project_route(
                route["waypoints_xz"],
                _finite(row["simulation_x"], f"row {index} simulation x"),
                _finite(row["simulation_z"], f"row {index} simulation z"))
            if (projection[0] > .25 or
                    (previous_progress is not None and
                     projection[1] + 1.0e-5 < previous_progress)):
                progress_mismatches.append(index)
            previous_progress = projection[1]
    if schedule_mismatches:
        failures.append(
            f"exact route/runtime schedule mismatched on "
            f"{len(schedule_mismatches)} frames")
    if progress_mismatches:
        failures.append(
            f"route progress regressed or left the authenticated path on "
            f"{len(progress_mismatches)} frames")

    empty_frames = []
    heading_mismatches = []
    flag_mismatches = []
    unusable_searches = []
    moving_frames = 0
    for index, row in enumerate(rows):
        empty = _integer(row["empty_compatible_set"],
                         f"route {identifier!r} row {index} empty flag") == 1
        if empty:
            empty_frames.append(index)
        commanded = _finite(row["commanded_speed"],
                            f"route {identifier!r} row {index} command")
        if commanded > 1.0e-4 and not empty:
            moving_frames += 1
            direction = _integer(
                row["direction_mask"],
                f"route {identifier!r} row {index} direction")
            if direction != SECTOR_MASKS[spec["sector"]]:
                heading_mismatches.append(index)
        if (float(row["effective_terrain_weight"]) != 4.0 or
                int(row["matching_enabled"]) != 1 or
                int(row["support_retargeting_enabled"]) != 1 or
                int(row["ik_enabled"]) != 0):
            flag_mismatches.append(index)
        if int(row["searched"]) == 1 and (
                int(row["eligible_frame_count"]) <= 0 or
                int(row["evaluated_frame_count"]) <= 0 or
                int(row["considered_bound_count"]) <= 0):
            unusable_searches.append(index)
    if moving_frames == 0:
        failures.append("heading oracle observed no moving nonempty frame")
    if heading_mismatches:
        failures.append(
            f"heading oracle mismatched {len(heading_mismatches)} frames")
    if empty_frames:
        failures.append(
            f"missing compatible coverage safe-stopped {len(empty_frames)} frames")
    if flag_mismatches:
        failures.append(
            f"weight/matching/support/IK flags changed on "
            f"{len(flag_mismatches)} frames")
    if unusable_searches:
        failures.append(
            f"search accounting unusable on {len(unusable_searches)} frames")
    if summary["search_count"] == 0:
        failures.append("search accounting observed no indexed search")

    phase_mismatches = [
        index for index, expected in enumerate(expected_trace)
        if not expected["state_valid"]
    ]
    confidence_mismatches = [
        index for index, expected in enumerate(expected_trace)
        if not expected["confidence_valid"]
    ]
    if phase_mismatches:
        failures.append(
            f"authenticated phase oracle mismatched "
            f"{len(phase_mismatches)} frames")
    if confidence_mismatches:
        failures.append(
            f"classifier confidence differs from authenticated reconstruction "
            f"on {len(confidence_mismatches)} frames")
    if source_claim_mismatches:
        failures.append(
            f"selected source claims differ from authenticated manifest on "
            f"{len(source_claim_mismatches)} frames")

    tangent_activation = None
    if identifier == "mixed-tangent":
        tangent_activation = next((
            index for index, row in enumerate(rows)
            if max(abs(float(row[f"terrain{sample}"]))
                   for sample in range(8, 12)) > 0.02), None)
        if tangent_activation is None:
            failures.append(
                "tangent partial-support corridor never activated")
    activation = tangent_activation
    if activation is None:
        activation = next((
            index for index, row in enumerate(rows)
            if row.get("active_family") not in (None, "flat")), None)

    geometry_mismatches = []
    points = route["waypoints_xz"] if route is not None else ()
    for index, row in enumerate(rows):
        root_y = float(row["terrain_root_point_y"])
        centers = []
        bad = None
        root_x = float(row["terrain_root_point_x"])
        root_z = float(row["terrain_root_point_z"])
        if grid is None or not points:
            geometry_mismatches.append(
                (index, "authenticated descriptor geometry"))
            continue
        if not _same_f32(root_y, _runtime_height(grid, root_x, root_z)):
            bad = "authenticated descriptor geometry root height"
        root_projection = _project_route(
            points, root_x, root_z)
        heading_x, heading_z = _route_heading(spec, route, index)
        expected_left = (-heading_z * CORRIDOR_OFFSET_M,
                         heading_x * CORRIDOR_OFFSET_M)
        if root_projection[0] > 0.25:
            bad = "descriptor geometry root path"
        previous_xz = (root_x, root_z)
        accumulated_chord = 0.0
        for sample in range(8):
            center = tuple(float(row[
                f"terrain_center_point{sample}_{axis}"]) for axis in "xyz")
            centers.append(center)
            if not _same_f32(center[1], _runtime_height(
                    grid, center[0], center[2])):
                bad = "authenticated descriptor geometry center height"
            if not _same_f32(
                    row[f"terrain{sample}"],
                    _runtime_height_difference(center[1], root_y)):
                bad = "descriptor geometry center value"
            if sample < 4 and any(not _same_f32(
                    row[f"terrain_point{sample}_{axis}"],
                    center[axis_index])
                    for axis_index, axis in enumerate("xyz")):
                bad = "descriptor geometry legacy point"
            planar_step = math.hypot(
                center[0] - previous_xz[0], center[2] - previous_xz[1])
            accumulated_chord += planar_step
            if planar_step < .05 or planar_step > .14:
                bad = "descriptor geometry 0.125m center arc"
            previous_xz = (center[0], center[2])
        if not .94 <= accumulated_chord <= 1.02:
            bad = "descriptor geometry one-metre horizon"
        for sample, center_index in enumerate((1, 3, 5, 7)):
            center = centers[center_index]
            left = tuple(float(row[
                f"terrain_left_point{sample}_{axis}"]) for axis in "xyz")
            right = tuple(float(row[
                f"terrain_right_point{sample}_{axis}"]) for axis in "xyz")
            midpoint = ((left[0] + right[0]) * .5,
                        (left[2] + right[2]) * .5)
            left_offset = (left[0] - center[0], left[2] - center[2])
            right_offset = (right[0] - center[0], right[2] - center[2])
            if (not _same_f32(
                    left[1], _runtime_height(grid, left[0], left[2])) or
                    not _same_f32(right[1], _runtime_height(
                        grid, right[0], right[2]))):
                bad = "authenticated descriptor geometry corridor height"
            if (max(abs(midpoint[0] - center[0]),
                    abs(midpoint[1] - center[2])) > 2e-6 or
                    max(abs(left_offset[0] - expected_left[0]),
                        abs(left_offset[1] - expected_left[1]),
                        abs(right_offset[0] + expected_left[0]),
                        abs(right_offset[1] + expected_left[1])) > 2e-6):
                bad = "heading geometry corridor"
            if not _same_f32(
                    row[f"terrain{sample + 8}"],
                    _runtime_height_difference(left[1], right[1])):
                bad = "descriptor geometry corridor value"
        if bad is not None:
            geometry_mismatches.append((index, bad))
    if geometry_mismatches:
        labels = sorted({label for _, label in geometry_mismatches})
        failures.append(
            f"{labels[0]} mismatched on {len(geometry_mismatches)} frames")

    start = points[0] if points else (math.inf, math.inf)
    endpoint = points[-1] if points else (math.inf, math.inf)
    simulated = [(float(row["simulation_x"]), float(row["simulation_z"]))
                 for row in rows]
    start_error = math.hypot(simulated[0][0] - start[0],
                             simulated[0][1] - start[1])
    endpoint_error = math.hypot(simulated[-1][0] - endpoint[0],
                                simulated[-1][1] - endpoint[1])
    if start_error > 0.25 or endpoint_error > 0.25:
        failures.append(
            f"physical progression endpoint mismatch: start={start_error:.6g} "
            f"end={endpoint_error:.6g}")
    if identifier == "mixed-tangent":
        waypoint = (0.62, 2.0)
        if (not any(int(row["route_waypoint"]) == 1 for row in rows) or
                not any(int(row["route_waypoint"]) == 2 for row in rows) or
                min(math.hypot(x - waypoint[0], z - waypoint[1])
                    for x, z in simulated) > 0.25):
            failures.append("tangent waypoint sector did not change at waypoint 1")

    gate_report = None
    gate_required = identifier in GATE_C_ROUTE_IDS
    gate_completed = not gate_required
    try:
        if gate_required:
            gate_report = check_gate_c(rows)
            if gate_report is None:
                failures.append(
                    "physical completion gate returned no composed report")
            else:
                gate_completed = True
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"physical completion gate failed: {error}")
    if not summary["route_complete"]:
        failures.append("route did not complete")
    if summary["ik_enabled_frames"]:
        failures.append("IK was enabled")

    wrong_family_frames = len(source_mismatches)

    return {
        **summary,
        "passed": not failures,
        "failures": failures,
        "expected_family": spec["expected_family"],
        "bank_correctness": {
            "activation_frame": activation,
            "wrong_family_frames": wrong_family_frames,
            "phase_mismatched_frames": len(phase_mismatches),
            "confidence_mismatched_frames": len(confidence_mismatches),
            "source_claim_mismatched_frames": len(source_claim_mismatches),
            "passed": (not phase_mismatches and not confidence_mismatches and
                       not source_mismatches and
                       not source_claim_mismatches),
        },
        "coverage_empty_frames": len(empty_frames),
        "heading_invariance": {
            "heading_case": spec["heading_case"],
            "heading": spec["heading"],
            "expected_direction_mask": SECTOR_MASKS[spec["sector"]],
            "mismatched_frames": len(heading_mismatches),
            "geometry_mismatched_frames": sum(
                "heading geometry" in label
                for _, label in geometry_mismatches),
            "moving_frames": moving_frames,
            "passed": (moving_frames > 0 and not heading_mismatches and
                       not any("heading geometry" in label
                               for _, label in geometry_mismatches)),
        },
        "physical_completion": {
            "start_error_m": start_error,
            "endpoint_error_m": endpoint_error,
            "composed_gate": gate_report,
            "passed": start_error <= .25 and endpoint_error <= .25 and
                      gate_completed,
        },
    }


def _comparison_gate(actual, threshold, comparison):
    if actual is None:
        return False
    if comparison == "equal":
        return actual == threshold
    if comparison == "strictly_less_than_baseline":
        return actual < threshold
    raise ValueError(f"unsupported comparison {comparison!r}")


def _scene_cycle_metric(rows, cleanup):
    failures = []
    gate_f = None
    try:
        gate_f = check_gate_f(rows, LOCKED_SCENE_IDS)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"Gate F failed: {error}")
    if len(rows) != 700:
        failures.append(
            f"scene cycle frame count {len(rows)} does not equal 700")
    expected_gate_f = {
        "frames": 700,
        "generations": 28,
        "complete_cycles": 2,
        "motion_pack_loads": 1,
        "model_loads": 28,
        "model_unloads_before_final_cleanup": 27,
    }
    if (not isinstance(gate_f, dict) or
            set(gate_f) != set(expected_gate_f) or
            any(type(gate_f[name]) is not int or
                gate_f[name] != expected
                for name, expected in expected_gate_f.items())):
        failures.append(
            "Gate F must return the exact composed 14-scene x 25 x 2 report")
    expected_cleanup = {
        "exit_code": 0,
        "live_model_count": 0,
        "log_closed": True,
        "model_load_count": 28,
        "model_unload_count": 28,
        "motion_pack_load_count": 1,
        "window_closed": True,
    }
    if (not isinstance(cleanup, dict) or set(cleanup) != set(expected_cleanup) or
            any(type(cleanup[name]) is not type(expected) or
                cleanup[name] != expected
                for name, expected in expected_cleanup.items())):
        failures.append(
            "normal cleanup must close log/window and unload all 28 models")
    return {
        "passed": not failures,
        "failures": failures,
        "gate_f": gate_f,
        "cleanup": cleanup,
    }


def _build_quality_gates(thresholds, comparisons):
    required = set()
    for name in ("flat_wrong_family_frames",
                 "activated_terrain_wrong_family_frames"):
        required.update(thresholds[name]["routes"])
    required.add(thresholds["lateral_selected_cost_p95"]["route"])
    required.add(thresholds["lateral_transition_rate"]["route"])
    if set(comparisons) != required or len(comparisons) != 6:
        raise ValueError(
            "checker-owned comparison ownership must be exactly six routes")

    gates = {}
    flat_routes = thresholds["flat_wrong_family_frames"]["routes"]
    flat_actual = sum(comparisons[route]["bank_correctness"][
        "wrong_family_frames"] for route in flat_routes)
    threshold = thresholds["flat_wrong_family_frames"]
    gates["flat_wrong_family_frames"] = {
        "actual": flat_actual, "comparison": threshold["comparison"],
        "threshold": threshold["target"], "routes": list(flat_routes),
        "passed": _comparison_gate(
            flat_actual, threshold["target"], threshold["comparison"]),
    }

    activated_routes = thresholds[
        "activated_terrain_wrong_family_frames"]["routes"]
    activated_actual = sum(comparisons[route]["bank_correctness"][
        "wrong_family_frames"] for route in activated_routes)
    activated_complete = all(comparisons[route]["bank_correctness"][
        "activation_frame"] is not None for route in activated_routes)
    threshold = thresholds["activated_terrain_wrong_family_frames"]
    gates["activated_terrain_wrong_family_frames"] = {
        "actual": activated_actual, "comparison": threshold["comparison"],
        "threshold": threshold["target"], "routes": list(activated_routes),
        "all_routes_activated": activated_complete,
        "passed": activated_complete and _comparison_gate(
            activated_actual, threshold["target"], threshold["comparison"]),
    }

    for name, metric in (
            ("lateral_selected_cost_p95", ("selected_cost", "p95")),
            ("lateral_transition_rate", ("transition_rate",))):
        threshold = thresholds[name]
        actual = comparisons[threshold["route"]]
        for key in metric:
            actual = actual.get(key) if isinstance(actual, dict) else None
        gates[name] = {
            "actual": actual, "comparison": threshold["comparison"],
            "threshold": threshold["baseline"], "route": threshold["route"],
            "passed": _comparison_gate(
                actual, threshold["baseline"], threshold["comparison"]),
        }

    reports = [comparisons[route] for route in sorted(required)]
    slip_actual = round(sum(
        report["stance_slip"]["total_m"][side]
        for report in reports for side in ("left", "right")), 9)
    threshold = thresholds["stance_slip_total_m"]
    gates["stance_slip_total_m"] = {
        "actual": slip_actual, "comparison": threshold["comparison"],
        "threshold": threshold["baseline"], "routes": sorted(required),
        "passed": _comparison_gate(
            slip_actual, threshold["baseline"], threshold["comparison"]),
    }
    penetration_actual = max(
        report["sole_clearance"]["worst_penetration_m"]
        for report in reports)
    threshold = thresholds["sole_worst_penetration_m"]
    gates["sole_worst_penetration_m"] = {
        "actual": penetration_actual, "comparison": threshold["comparison"],
        "threshold": threshold["baseline"], "routes": sorted(required),
        "passed": _comparison_gate(
            penetration_actual, threshold["baseline"], threshold["comparison"]),
    }
    return gates


def _build_evidence_report(matrix, matrix_payload, evidence):
    if matrix_payload is None:
        matrix_payload = canonical_json(matrix).encode("utf-8")
    if not isinstance(matrix_payload, bytes):
        raise ValueError("matrix payload must be bytes")
    matrix_sha256 = hashlib.sha256(matrix_payload).hexdigest()
    if evidence is None:
        evidence = {
            "routes": {
                route["id"]: {
                    name: route[name] for name in (
                        "log", "log_sha256", "origin", "origin_sha256")
                }
                for route in matrix["routes"]
            },
            "scene_cycle": dict(matrix["scene_cycle"]),
            "files_verified": False,
        }
    if (not isinstance(evidence, dict) or
            set(evidence) != {"routes", "scene_cycle", "files_verified"} or
            type(evidence["files_verified"]) is not bool):
        raise ValueError("computed evidence digest report is invalid")
    return {
        "schema": EVIDENCE_SCHEMA,
        "matrix_sha256": matrix_sha256,
        "files_verified": evidence["files_verified"],
        "origin_authentication": (
            "self-attested hash-bound invocation metadata; no independent "
            "process attestation"),
        "routes": evidence["routes"],
        "scene_cycle": evidence["scene_cycle"],
    }


def evaluate_candidate(
    baseline,
    matrix,
    logs,
    scene_cycle_rows,
    cleanup,
    *,
    baseline_payload,
    candidate_validator,
    matrix_payload=None,
    evidence=None,
):
    """Evaluate exact production evidence against immutable Task-3 gates."""
    baseline_sha256 = _validate_canonical_baseline(
        baseline, baseline_payload)
    thresholds = _thresholds(baseline)
    specs = _route_specs(matrix)
    identity = _validate_candidate_identity(matrix, candidate_validator)
    evidence_report = _build_evidence_report(
        matrix, matrix_payload, evidence)
    artifacts = _load_authenticated_candidate_artifacts(
        matrix["candidate"]["artifact_directory"],
        matrix["candidate"]["manifest_sha256"])
    if not isinstance(logs, dict):
        raise ValueError("logs must map production route ids to rows")
    expected_ids = {spec["id"] for spec in specs}
    if set(logs) != expected_ids:
        raise ValueError(
            f"matrix/log route mismatch: missing={sorted(expected_ids - set(logs))} "
            f"unexpected={sorted(set(logs) - expected_ids)}")

    scene_cycle = _scene_cycle_metric(scene_cycle_rows, cleanup)
    routes = {}
    comparisons = {}
    report_failures = []
    sectors_by_family = {family: set() for family in SECTOR_FAMILIES}
    sectors_by_heading = {"fixed": set(), "changed": set()}
    for spec in specs:
        route_report = _route_metric(
            spec, logs[spec["id"]], artifacts)
        routes[spec["id"]] = route_report
        if not route_report["passed"]:
            report_failures.extend(
                f"{spec['id']}: {failure}"
                for failure in route_report["failures"])
        if spec["expected_family"] in sectors_by_family:
            sectors_by_family[spec["expected_family"]].add(spec["sector"])
        sectors_by_heading[spec["heading_case"]].add(spec["sector"])
        if "comparison_id" in spec:
            comparisons[spec["comparison_id"]] = route_report
    if not scene_cycle["passed"]:
        report_failures.extend(
            f"scene-cycle: {failure}" for failure in scene_cycle["failures"])

    expected_sectors = set(range(8))
    coverage_failures = []
    for family in SECTOR_FAMILIES:
        missing = sorted(expected_sectors - sectors_by_family[family])
        if missing:
            coverage_failures.append(
                f"{family} missing travel sectors {missing}")
    for heading_case in ("fixed", "changed"):
        missing = sorted(expected_sectors - sectors_by_heading[heading_case])
        if missing:
            coverage_failures.append(
                f"{heading_case} heading missing travel sectors {missing}")

    cross_rows = [row for spec in specs if spec["scene"] == "cross-slope-10"
                  for row in logs[spec["id"]]]
    cross_values = [float(row[f"terrain{sample}"])
                    for row in cross_rows for sample in range(8, 12)]
    derived_coverage = {
        "flat": all(routes[spec["id"]]["bank_correctness"]["passed"]
                    for spec in specs if spec["expected_family"] == "flat"),
        "slope-up": any(int(row["elevation_mode"]) == 1
                        for spec in specs if spec["scene"].startswith("ramp-")
                        for row in logs[spec["id"]]),
        "slope-down": any(int(row["elevation_mode"]) == -1
                          for spec in specs if spec["scene"].startswith("ramp-")
                          for row in logs[spec["id"]]),
        "cross-slope-positive": any(value > .02 for value in cross_values),
        "cross-slope-negative": any(value < -.02 for value in cross_values),
        "stairs-up": any(int(row["elevation_mode"]) == 1
                         for spec in specs if spec["scene"].startswith("stairs-")
                         and spec["route"] == "ascent-landing-descent"
                         for row in logs[spec["id"]]),
        "stairs-down": any(int(row["elevation_mode"]) == -1
                           for spec in specs if spec["scene"].startswith("stairs-")
                           and spec["route"] == "ascent-landing-descent"
                           for row in logs[spec["id"]]),
        "lateral": all(sector in sectors_by_family["flat"]
                       for sector in (2, 6)),
        "diagonal-left": all(sector in sectors_by_family["flat"]
                             for sector in (5, 7)),
        "diagonal-right": all(sector in sectors_by_family["flat"]
                              for sector in (1, 3)),
        "tangent": routes["mixed-tangent"]["physical_completion"]["passed"],
        "partial-support": routes["mixed-tangent"][
            "bank_correctness"]["activation_frame"] is not None,
        "multilevel": routes["mixed-full-course"][
            "physical_completion"]["composed_gate"] is not None,
        "scene-switch-cleanup": scene_cycle["passed"],
    }
    for name, passed in derived_coverage.items():
        if not passed:
            coverage_failures.append(f"derived coverage {name} failed")

    paired_distinct = True
    by_id = {spec["id"]: spec for spec in specs}
    for name in SECTOR_NAMES:
        fixed = logs[f"flat-fixed-{name}"]
        changed = logs[f"flat-changed-{name}"]
        fixed_geometry = tuple(
            (row["terrain_root_point_x"], row["terrain_root_point_z"])
            for row in fixed)
        changed_geometry = tuple(
            (row["terrain_root_point_x"], row["terrain_root_point_z"])
            for row in changed)
        if fixed_geometry == changed_geometry:
            paired_distinct = False
            coverage_failures.append(
                f"paired heading geometry is indistinguishable for {name}")
    del by_id
    report_failures.extend(coverage_failures)
    coverage_report = {
        "passed": not coverage_failures,
        "failures": coverage_failures,
        "sectors_by_family": {
            family: sorted(values)
            for family, values in sectors_by_family.items()
        },
        "sectors_by_heading_case": {
            case: sorted(values) for case, values in sectors_by_heading.items()
        },
        "derived": {name: bool(value)
                    for name, value in sorted(derived_coverage.items())},
        "paired_heading_geometry_distinct": paired_distinct,
    }

    gates = _build_quality_gates(thresholds, comparisons)

    for name, gate in gates.items():
        if not gate["passed"]:
            report_failures.append(
                f"gate {name} failed: actual={gate['actual']!r} "
                f"threshold={gate['threshold']!r}")
    report_failures = sorted(set(report_failures))
    return {
        "schema": REPORT_SCHEMA,
        "passed": not report_failures,
        "sample_rate_hz": 25,
        "ik_enabled": False,
        "baseline_schema": baseline["schema"],
        "baseline_sha256": baseline_sha256,
        "candidate": identity,
        "evidence": evidence_report,
        "coverage": coverage_report,
        "scene_cycle": scene_cycle,
        "routes": {identifier: routes[identifier]
                   for identifier in sorted(routes)},
        "gates": {name: gates[name] for name in sorted(gates)},
        "failures": report_failures,
    }


def _read_json(path, label):
    try:
        payload = pathlib.Path(path).read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    return value


def _read_json_with_payload(path, label):
    try:
        payload = pathlib.Path(path).read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error
    return value, payload


def _verified_evidence_path(root, relative, expected_sha256, label):
    if (not isinstance(expected_sha256, str) or
            not _SHA256.fullmatch(expected_sha256)):
        raise ValueError(f"{label} SHA-256 is invalid")
    root = pathlib.Path(root).resolve()
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    relative_path = pathlib.Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"{label} must be relative")
    lexical = root / relative_path
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes logs directory") from error
    _regular_file(lexical, label)
    computed = _sha256_file(lexical)
    if computed != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return resolved, computed


def _load_logs(directory, matrix):
    root = pathlib.Path(directory).resolve()
    logs = {}
    evidence = {"routes": {}, "scene_cycle": {}}
    for spec in _route_specs(matrix):
        label = f"matrix route {spec['id']!r} log"
        path, log_sha256 = _verified_evidence_path(
            root, spec["log"], spec["log_sha256"], label)
        origin_path, origin_sha256 = _verified_evidence_path(
            root, spec["origin"], spec["origin_sha256"],
            f"matrix route {spec['id']!r} origin")
        origin = _read_json(origin_path, f"route {spec['id']!r} origin")
        _validate_run_origin(spec, matrix["candidate"], path, origin)
        logs[spec["id"]] = read_rows(path)
        evidence["routes"][spec["id"]] = {
            "log": spec["log"], "log_sha256": log_sha256,
            "origin": spec["origin"], "origin_sha256": origin_sha256,
        }
    scene_spec = matrix["scene_cycle"]
    scene_path, scene_sha256 = _verified_evidence_path(
        root, scene_spec["log"], scene_spec["log_sha256"],
        "matrix scene-cycle log")
    cleanup_path, cleanup_sha256 = _verified_evidence_path(
        root, scene_spec["cleanup"], scene_spec["cleanup_sha256"],
        "matrix cleanup report")
    origin_path, origin_sha256 = _verified_evidence_path(
        root, scene_spec["origin"], scene_spec["origin_sha256"],
        "matrix scene-cycle origin")
    origin = _read_json(origin_path, "scene-cycle run origin")
    _validate_scene_cycle_origin(
        matrix["candidate"], scene_path, cleanup_path, origin)
    scene_rows = read_rows(scene_path)
    cleanup = _read_json(cleanup_path, "cleanup report")
    evidence["scene_cycle"] = {
        "log": scene_spec["log"], "log_sha256": scene_sha256,
        "cleanup": scene_spec["cleanup"],
        "cleanup_sha256": cleanup_sha256,
        "origin": scene_spec["origin"], "origin_sha256": origin_sha256,
    }
    evidence["files_verified"] = True
    return logs, scene_rows, cleanup, evidence


def _write_transactionally(path, text):
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check full-pack pre-IK G1 motion-quality logs")
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--logs", required=True,
        help="directory containing matrix.json and its runtime CSV files")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        baseline, baseline_payload = _read_json_with_payload(
            arguments.baseline, "baseline")
        matrix_path = pathlib.Path(arguments.logs) / "matrix.json"
        matrix, matrix_payload = _read_json_with_payload(
            matrix_path, "matrix")
        logs, scene_rows, cleanup, evidence = _load_logs(
            arguments.logs, matrix)
        from resources.validate_g1_terrain_database import (
            validate_artifact_directory,
        )
        report = evaluate_candidate(
            baseline, matrix, logs, scene_rows, cleanup,
            baseline_payload=baseline_payload,
            candidate_validator=validate_artifact_directory,
            matrix_payload=matrix_payload,
            evidence=evidence)
        _write_transactionally(arguments.output, canonical_json(report))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"{'PASS' if report['passed'] else 'FAIL'} "
        f"routes={len(report['routes'])} failures={len(report['failures'])} "
        f"report={arguments.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
