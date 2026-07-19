import copy
import hashlib
import json
import math
import os
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from resources.check_g1_runtime_log import LOCKED_SCENE_IDS
from resources.g1_terrain_builder.terrain import classify_terrain_profile
from resources import check_g1_motion_quality as quality
from tests.python.test_motion_quality_summary import quality_rows
from tests.python.test_runtime_log import runtime_row, scene_cycle_rows


COMPARISON_IDS = (
    "stairs-shallow/flat-positive-z",
    "stairs-shallow/flat-positive-x",
    "stairs-standard/ascent-landing-descent",
    "ramp-10-up-down/up-landing-down",
    "stairs-standard/ascent-landing-descent@positive-x",
    "mixed-multilevel/tangent-level-boundary",
)
SECTOR_MASKS = (0x002, 0x004, 0x008, 0x010,
                0x020, 0x040, 0x080, 0x100)
REQUIRED_TAGS = (
    "flat",
    "slope-up",
    "slope-down",
    "cross-slope-positive",
    "cross-slope-negative",
    "stairs-up",
    "stairs-down",
    "lateral",
    "diagonal-left",
    "diagonal-right",
    "tangent",
    "partial-support",
    "multilevel",
    "scene-switch-cleanup",
)

# This is the checker-owned production command roster.  Matrix JSON may name
# evidence files for these IDs, but may not invent scenes, routes, durations,
# headings, families, sectors, or comparison ownership.
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


def expected_production_roster():
    roster = []
    for sector, name in enumerate(SECTOR_NAMES):
        roster.append({
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
            changed["comparison_id"] = COMPARISON_IDS[1]
        roster.append(changed)
    roster[0]["comparison_id"] = COMPARISON_IDS[0]

    slope_scenes = (
        "ramp-10-up-down", "ramp-10-up-down", "ramp-10-up-down",
        "cross-slope-10", "ramp-10-up-down", "cross-slope-10",
        "cross-slope-10", "cross-slope-10",
    )
    for sector, (name, scene) in enumerate(zip(SECTOR_NAMES, slope_scenes)):
        item = {
            "id": f"slope-fixed-{name}", "scene": scene,
            "route": ("up-landing-down" if scene == "ramp-10-up-down"
                      else "forward-cross-slope"),
            "frames": 800, "expected_family": "slope", "sector": sector,
            "heading_case": "fixed", "heading": POSITIVE_Z_HEADINGS[sector],
        }
        if sector == 0:
            item["comparison_id"] = COMPARISON_IDS[3]
        roster.append(item)

    for sector, name in enumerate(SECTOR_NAMES):
        item = {
            "id": f"stair-fixed-{name}", "scene": "stairs-standard",
            "route": "ascent-landing-descent", "frames": 800,
            "expected_family": "stair", "sector": sector,
            "heading_case": "fixed", "heading": POSITIVE_Z_HEADINGS[sector],
        }
        if sector == 0:
            item["comparison_id"] = COMPARISON_IDS[2]
        elif sector == 6:
            item["comparison_id"] = COMPARISON_IDS[4]
        roster.append(item)
    roster.extend((
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
            "comparison_id": COMPARISON_IDS[5],
        },
        {
            "id": "mixed-full-course", "scene": "mixed-multilevel",
            "route": "full-course", "frames": 800,
            "expected_family": "mixed", "sector": 0,
            "heading_case": "fixed", "heading": "forward",
        },
    ))
    return roster


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
CORRIDOR_OFFSET = 0.148506455


FIXTURE_SCENE_ROUTES = {
    "stairs-shallow": (
        ("ascent-landing-descent", 2.0,
         ((0.0, 0.0), (0.0, 1.75), (0.0, 4.199999809265137),
          (0.0, 6.650000095367432), (0.0, 7.400000095367432))),
        ("flat-positive-z", 0.0, ((0.0, 0.0), (0.0, 1.0))),
        ("flat-positive-x", 0.0, ((0.0, 0.0), (1.0, 0.0))),
    ),
    "stairs-standard": (
        ("ascent-landing-descent", 2.0,
         ((0.0, 0.0), (0.0, 1.75), (0.0, 3.9600000381469727),
          (0.0, 6.170000076293945), (0.0, 6.920000076293945))),
    ),
    "stairs-unseen-variable": (
        ("ascent-landing-descent", 2.0,
         ((0.0, 0.0), (0.0, 1.75), (0.0, 4.239999771118164),
          (0.0, 6.730000019073486), (0.0, 7.480000019073486))),
    ),
    "ramp-05-up-down": (
        ("up-landing-down", 2.0,
         ((0.0, 0.0), (0.0, 1.75), (0.0, 7.114819049835205),
          (0.0, 12.47963809967041), (0.0, 13.22963809967041))),
    ),
    "ramp-10-up-down": (
        ("up-landing-down", 2.0,
         ((0.0, 0.0), (0.0, 1.75), (0.0, 5.041661262512207),
          (0.0, 8.333322525024414), (0.0, 9.083322525024414))),
    ),
    "cross-slope-10": (
        ("forward-cross-slope", 0.0,
         ((0.0, 0.0), (0.0, 2.75), (0.0, 5.0),
          (0.0, 7.25), (0.0, 8.0))),
    ),
    "mixed-multilevel": (
        ("full-course", 2.0,
         ((0.0, 0.0), (0.0, 1.75), (0.0, 4.699999809265137),
          (0.0, 6.5), (0.0, 7.099999904632568),
          (0.0, 7.699999809265137), (0.0, 9.814809799194336),
          (0.0, 10.814809799194336))),
        ("tangent-level-boundary", 0.0,
         ((0.0, 0.0), (0.6200000047683716, 2.0),
          (0.6200000047683716, 6.0))),
    ),
}


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def write_candidate_fixture(root):
    candidate = root / "candidate"
    candidate.mkdir()
    scene_descriptors = []
    heightfield_header = struct.Struct("<4sIII4f")
    origin_x, origin_z, cell = -3.0, -1.0, _f32(.02)
    nx, nz = 301, 801
    for scene, routes in FIXTURE_SCENE_ROUTES.items():
        scene_spec = next(
            item for item in expected_production_roster()
            if item["scene"] == scene and (
                scene != "stairs-shallow" or
                item["route"] == "ascent-landing-descent"))
        shape = _route_shape(scene_spec)
        scene_root = candidate / "scenes" / scene
        scene_root.mkdir(parents=True)
        heights = bytearray()
        for iz in range(nz):
            z = _f32(origin_z + iz * cell)
            for ix in range(nx):
                x = _f32(origin_x + ix * cell)
                heights.extend(struct.pack(
                    "<f", _surface_height(shape, x, z)))
        terrain = heightfield_header.pack(
            b"G1HF", 2, nx, nz, origin_x, origin_z, cell, 0.0) + heights
        terrain_path = scene_root / "terrain.bin"
        terrain_path.write_bytes(terrain)
        scene_payload = {
            "schema": "g1-terrain-scene/v1",
            "id": scene,
            "heightfield": {
                "path": "terrain.bin", "schema": "G1HF/v2", "version": 2,
                "nx": nx, "nz": nz, "origin_x": origin_x,
                "origin_z": origin_z, "cell_size_m": cell,
                "exterior_height_m": 0.0,
                "interpolation": "fixed-diagonal-triangles",
                "diagonal": "min-x-min-z_to_max-x-max-z",
                "sha256": hashlib.sha256(terrain).hexdigest(),
            },
            "routes": [
                {
                    "id": identifier,
                    "landing_hold_seconds": hold,
                    "waypoints_xz": [list(point) for point in points],
                }
                for identifier, hold, points in routes
            ],
        }
        scene_bytes = json.dumps(
            scene_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        scene_path = scene_root / "scene.json"
        scene_path.write_bytes(scene_bytes)
        scene_descriptors.append({
            "id": scene,
            "path": f"scenes/{scene}/scene.json",
            "sha256": hashlib.sha256(scene_bytes).hexdigest(),
        })
    index_payload = {
        "schema": "g1-terrain-scene-index/v1",
        "scenes": scene_descriptors,
    }
    index_bytes = json.dumps(
        index_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    index_path = candidate / "scenes" / "index.json"
    index_path.write_bytes(index_bytes)
    manifest = {
        "schema": "g1-terrain-artifacts/v3",
        "total_clips": 15_815,
        "database_frames": 3_970_932,
        "diagnostic_mode": False,
        "scene_index": {
            "path": "scenes/index.json",
            "schema": "g1-terrain-scene-index/v1",
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "sources": [
            {
                "name": f"{family}-fixture", "terrain_id": family,
                "terrain_family": family, "range_start": index * 1000,
                "range_stop": (index + 1) * 1000,
            }
            for index, family in enumerate(("flat", "curb", "slope", "stair"))
        ],
    }
    manifest_path = candidate / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    binary = root / "controller"
    binary.write_bytes(b"release-controller")
    binary.chmod(0o755)
    return candidate, manifest, binary


def _route_shape(spec):
    scene = spec["scene"]
    route = spec["route"]
    points = next(
        tuple(waypoints)
        for identifier, _, waypoints in FIXTURE_SCENE_ROUTES[scene]
        if identifier == route)
    if route in ("flat-positive-z", "flat-positive-x"):
        rises, runs = (0.08,) * 4, (0.30,) * 4
        ascent_end = 2.0 + sum(runs)
        descent_start = ascent_end + 2.0
        return {
            "kind": "stair", "rises": rises, "runs": runs,
            "ascent_end": ascent_end, "descent_start": descent_start,
            "descent_end": descent_start + sum(runs),
            "top": sum(rises), "points": points,
        }
    if scene.startswith("ramp-"):
        angle = 5.0 if scene == "ramp-05-up-down" else 10.0
        run = 0.36 / math.tan(math.radians(angle))
        ascent_end = 2.0 + run
        descent_start = ascent_end + 2.0
        descent_end = descent_start + run
        return {
            "kind": "ramp", "ascent_end": ascent_end,
            "descent_start": descent_start, "descent_end": descent_end,
            "top": 0.36,
            "points": points,
        }
    if scene.startswith("stairs-") and route == "ascent-landing-descent":
        definitions = {
            "stairs-shallow": ((0.08,) * 4, (0.30,) * 4),
            "stairs-standard": ((0.12,) * 3, (0.32,) * 3),
            "stairs-unseen-variable": (
                (0.06, 0.10, 0.08, 0.12),
                (0.24, 0.34, 0.28, 0.38)),
        }
        rises, runs = definitions[scene]
        ascent_end = 2.0 + sum(runs)
        descent_start = ascent_end + 2.0
        descent_end = descent_start + sum(runs)
        return {
            "kind": "stair", "rises": rises, "runs": runs,
            "ascent_end": ascent_end, "descent_start": descent_start,
            "descent_end": descent_end, "top": sum(rises),
            "points": points,
        }
    if scene == "cross-slope-10":
        return {
            "kind": "cross", "grade_start": 3.0, "grade_end": 7.0,
            "points": points,
        }
    if route == "tangent-level-boundary":
        return {
            "kind": "tangent",
            "points": points,
        }
    if route == "full-course":
        ramp_end = 8.0 + 0.32 / math.tan(math.radians(10.0))
        return {
            "kind": "mixed", "ramp_end": ramp_end,
            "points": points,
        }
    raise AssertionError((scene, route))


def _polyline_lengths(points):
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:])]
    return lengths, sum(lengths)


def _point_on_route(points, distance):
    lengths, total = _polyline_lengths(points)
    remaining = max(0.0, distance)
    for segment, (start, stop) in enumerate(zip(points, points[1:])):
        length = lengths[segment]
        if remaining < length - 1e-12:
            alpha = remaining / length if length else 0.0
            return (
                start[0] + alpha * (stop[0] - start[0]),
                start[1] + alpha * (stop[1] - start[1]),
                segment + 1,
            )
        remaining -= length
    start, stop = points[-2:]
    length = lengths[-1]
    return (
        stop[0] + remaining * (stop[0] - start[0]) / length,
        stop[1] + remaining * (stop[1] - start[1]) / length,
        len(points) - 1,
    )


def _mixed_height(x, z):
    if abs(x) > 0.6 or z < 2.0:
        return 0.0
    height = 0.0
    for rise, end in zip((0.08,) * 4, (2.3, 2.6, 2.9, 3.2)):
        height += rise
        if z < end:
            return height
    if z < 6.2:
        return 0.32
    if z < 6.8:
        return 0.40
    if z < 7.4:
        return 0.28
    if z < 8.0:
        return 0.32
    ramp_end = 8.0 + 0.32 / math.tan(math.radians(10.0))
    if z < ramp_end:
        return 0.32 * (ramp_end - z) / (ramp_end - 8.0)
    return 0.0


def _surface_height(shape, x, z):
    kind = shape["kind"]
    if kind == "flat":
        return 0.0
    if kind == "ramp":
        if z < 2.0:
            return 0.0
        if z < shape["ascent_end"]:
            return shape["top"] * (z - 2.0) / (shape["ascent_end"] - 2.0)
        if z < shape["descent_start"]:
            return shape["top"]
        if z < shape["descent_end"]:
            return shape["top"] * (shape["descent_end"] - z) / (
                shape["descent_end"] - shape["descent_start"])
        return 0.0
    if kind == "stair":
        if z < 2.0:
            return 0.0
        height = 0.0
        ascent_ends = []
        cursor = 2.0
        for rise, run in zip(shape["rises"], shape["runs"]):
            cursor += run
            ascent_ends.append(cursor)
            height += rise
            if z < cursor:
                return height
        if z < shape["descent_start"]:
            return height
        cursor = shape["descent_start"]
        for rise, run in zip(reversed(shape["rises"]),
                             reversed(shape["runs"])):
            cursor += run
            if z < cursor:
                return height
            height -= rise
        return 0.0
    if kind == "cross":
        phase = z - shape["grade_start"]
        if phase <= 0.0 or z >= shape["grade_end"] or abs(x) > 0.6:
            return 0.0
        factor = min(1.0, phase / 0.5, (4.0 - phase) / 0.5)
        return x * math.tan(math.radians(10.0)) * factor
    return _mixed_height(x, z)


def _support_height(shape, z):
    kind = shape["kind"]
    if kind == "ramp":
        return _surface_height(shape, 0.0, z)
    if kind == "stair":
        if z < 2.0:
            return 0.0
        if z < shape["ascent_end"]:
            return shape["top"] * (z - 2.0) / (shape["ascent_end"] - 2.0)
        if z < shape["descent_start"]:
            return shape["top"]
        if z < shape["descent_end"]:
            return shape["top"] * (shape["descent_end"] - z) / (
                shape["descent_end"] - shape["descent_start"])
        return 0.0
    if kind == "mixed":
        if z < 2.0:
            return 0.0
        if z < 3.2:
            return .32 * (z - 2.0) / 1.2
        if z < 6.2:
            return .32
        if z < 6.26:
            return .32 + .08 * (z - 6.2) / .06
        if z < 6.8:
            return .40
        if z < 6.86:
            return .40 - .12 * (z - 6.8) / .06
        if z < 7.4:
            return .28
        if z < 7.46:
            return .28 + .04 * (z - 7.4) / .06
        if z < 8.0:
            return .32
        if z < shape["ramp_end"]:
            return .32 * (shape["ramp_end"] - z) / (
                shape["ramp_end"] - 8.0)
        return 0.0
    return 0.0


def _family_elevation(shape, z):
    kind = shape["kind"]
    if kind == "ramp":
        if 2.0 <= z < shape["ascent_end"]:
            return "slope", 1
        if shape["descent_start"] <= z < shape["descent_end"]:
            return "slope", -1
        return "flat", 0
    if kind == "stair":
        if 2.0 <= z < shape["ascent_end"]:
            return "stair", 1
        if shape["descent_start"] <= z < shape["descent_end"]:
            return "stair", -1
        return "flat", 0
    if kind == "cross":
        return ("slope", 0) if 3.0 <= z < 7.0 else ("flat", 0)
    if kind == "mixed":
        if 2.0 <= z < 3.2:
            return "stair", 1
        if 8.0 <= z < shape["ramp_end"]:
            return "slope", -1
    return "flat", 0


def _fixture_height_cell(shape, x, z):
    origin_x = -3.0
    origin_z = -1.0
    cell = _f32(.02)
    x, z = _f32(x), _f32(z)
    coordinate_x = (x - origin_x) / cell
    coordinate_z = (z - origin_z) / cell
    ix = min(max(math.floor(coordinate_x), 0), 299)
    iz = min(max(math.floor(coordinate_z), 0), 799)
    node_x = origin_x + ix * cell
    node_z = origin_z + iz * cell
    tx = min(max((x - node_x) / cell, 0.0), 1.0)
    tz = min(max((z - node_z) / cell, 0.0), 1.0)
    heights = tuple(
        _f32(_surface_height(
            shape, _f32(origin_x + node_ix * cell),
            _f32(origin_z + node_iz * cell)))
        for node_iz, node_ix in (
            (iz, ix), (iz, ix + 1), (iz + 1, ix), (iz + 1, ix + 1)))
    return cell, tx, tz, heights


def _fixture_height(shape, x, z):
    x, z = _f32(x), _f32(z)
    maximum_x = -3.0 + 300 * _f32(.02)
    maximum_z = -1.0 + 800 * _f32(.02)
    if x < -3.0 or x > maximum_x or z < -1.0 or z > maximum_z:
        raise ValueError("fixture descriptor query outside authenticated grid")
    _, tx, tz, (h00, h10, h01, h11) = _fixture_height_cell(shape, x, z)
    if tx >= tz:
        value = h00 + tx * (h10 - h00) + tz * (h11 - h10)
    else:
        value = h00 + tx * (h11 - h01) + tz * (h01 - h00)
    return float(_f32(value))


def _surface_normal(shape, x, z):
    x, z = _f32(x), _f32(z)
    maximum_x = -3.0 + 300 * _f32(.02)
    maximum_z = -1.0 + 800 * _f32(.02)
    if x < -3.0 or x > maximum_x or z < -1.0 or z > maximum_z:
        raise ValueError("fixture normal query outside authenticated grid")
    cell, tx, tz, (h00, h10, h01, h11) = \
        _fixture_height_cell(shape, x, z)
    if tx >= tz:
        gradient_x = (h10 - h00) / cell
        gradient_z = (h11 - h10) / cell
    else:
        gradient_x = (h11 - h01) / cell
        gradient_z = (h01 - h00) / cell
    vector = np.array((-gradient_x, 1.0, -gradient_z), np.float64)
    return vector / np.linalg.norm(vector)


def _test_profile_classification(shape, points, distance):
    distances = np.linspace(0.0, 1.0, 51)
    samples = [_point_on_route(points, distance + float(offset))
               for offset in distances]
    heights = np.array([
        _fixture_height(shape, x, z) for x, z, _ in samples
    ], np.float64)
    normals = np.array([
        _surface_normal(shape, x, z) for x, z, _ in samples
    ], np.float64)
    return classify_terrain_profile(distances, heights, normals)


def _test_logged_profile_classification(shape, row):
    points = [
        (float(row["terrain_root_point_x"]),
         float(row["terrain_root_point_z"])),
    ] + [
        (float(row[f"terrain_center_point{sample}_x"]),
         float(row[f"terrain_center_point{sample}_z"]))
        for sample in range(8)
    ]
    distances = np.linspace(0.0, 1.0, 51)
    samples = []
    for distance in distances:
        scaled = min(8.0, float(distance) / .125)
        start = min(7, math.floor(scaled))
        alpha = scaled - start
        samples.append((
            points[start][0] + alpha * (
                points[start + 1][0] - points[start][0]),
            points[start][1] + alpha * (
                points[start + 1][1] - points[start][1]),
        ))
    heights = np.asarray([
        _fixture_height(shape, x, z) for x, z in samples
    ], np.float64)
    normals = np.asarray([
        np.asarray(_surface_normal(shape, x, z), np.float32)
        for x, z in samples
    ], np.float64)
    return classify_terrain_profile(distances, heights, normals)


def _test_claimed_confidence_replay(observations, confidences):
    current = None
    current_elevation = 0
    pending = None
    pending_count = 0
    trace = []
    for (family, elevation), confidence in zip(observations, confidences):
        transitioned = 0
        if current is None:
            pending = None
            pending_count = 0
            if confidence >= .60:
                current, current_elevation = family, elevation
                transitioned = 1
                reason = "initial_confident"
            else:
                reason = "pending_cleared_low_confidence"
        elif (family, elevation) == (current, current_elevation):
            pending = None
            pending_count = 0
            reason = "retained_same"
        elif confidence < .60:
            pending = None
            pending_count = 0
            reason = "pending_cleared_low_confidence"
        elif pending != (family, elevation):
            pending = (family, elevation)
            pending_count = 1
            reason = "pending_started"
        else:
            pending_count += 1
            if pending_count < 2:
                reason = "pending_advanced"
            else:
                current, current_elevation = pending
                pending = None
                pending_count = 0
                transitioned = 1
                reason = "confirmed"
        trace.append({
            "requested_family": family,
            "active_family": current,
            "active_elevation": current_elevation,
            "transitioned": transitioned,
            "reason": reason,
        })
    return trace


def _refresh_query_bits(row):
    values = [0.0] * 27 + [float(row[f"terrain{sample}"])
                           for sample in range(12)]
    row["query_bits_hex"] = "".join(
        struct.pack(">f", value).hex() for value in values)


def _test_segment_schedule(points):
    step = _f32(_f32(.5) * _f32(.04))
    schedule = []
    prefix = 0.0
    for start, stop in zip(points, points[1:]):
        dx = _f32(_f32(stop[0]) - _f32(start[0]))
        dz = _f32(_f32(stop[1]) - _f32(start[1]))
        squared = _f32(_f32(dx * dx) + _f32(dz * dz))
        length = _f32(math.sqrt(squared))
        frames = max(1, math.ceil(_f32(length / step)))
        schedule.append({
            "start": start, "stop": stop, "length": length,
            "frames": frames, "prefix": prefix,
            "direction": (dx / length, dz / length),
        })
        prefix += float(length)
    return schedule


def _test_route_command(spec, frame):
    shape = _route_shape(spec)
    schedule = _test_segment_schedule(shape["points"])
    hold = next(
        seconds for identifier, seconds, _ in FIXTURE_SCENE_ROUTES[
            spec["scene"]] if identifier == spec["route"])
    hold_frames = math.ceil(_f32(_f32(hold) / _f32(.04))) if hold else 0
    cursor = 0
    for index, segment in enumerate(schedule):
        if frame < cursor + segment["frames"]:
            local = frame - cursor
            distance = segment["prefix"] + min(
                float(segment["length"]), local * .02)
            return {
                "waypoint": index + 1, "complete": False,
                "command": tuple(.5 * value for value in segment["direction"]),
                "distance": distance,
            }
        cursor += segment["frames"]
        if index + 1 == 2 and hold_frames:
            if frame < cursor + hold_frames:
                return {
                    "waypoint": 2, "complete": False,
                    "command": (0.0, 0.0),
                    "distance": segment["prefix"] + segment["length"],
                }
            cursor += hold_frames
    return {
        "waypoint": len(shape["points"]) - 1, "complete": True,
        "command": (0.0, 0.0),
        "distance": sum(segment["length"] for segment in schedule),
    }


def _bind_fixture_source_provenance(rows):
    previous_current = None
    previous_source_index = None
    for frame, row in enumerate(rows):
        source_index = int(row["source_index"])
        selected = source_index * 1000 + 10 + frame
        query = selected if previous_current is None else previous_current
        query_source = (source_index if previous_source_index is None
                        else previous_source_index)
        transitioned = int(selected != query)
        row["query_database_frame"] = str(query)
        row["query_range"] = str(query_source)
        row["selected_database_frame"] = str(selected)
        row["database_frame"] = str(selected + 1)
        row["range"] = str(source_index)
        row["source_range"] = str(source_index)
        row["transitioned"] = str(transitioned)
        if transitioned:
            row["searched"] = "1"
            row["eligible_frame_count"] = "8"
            row["evaluated_frame_count"] = "8"
            row["considered_bound_count"] = "8"
        previous_current = selected + 1
        previous_source_index = source_index


def production_route_rows(spec):
    shape = _route_shape(spec)
    points = shape["points"]
    explicit_heading = HEADING_VECTORS.get(spec["heading"])
    rows = []
    observations = []
    confidences = []
    for frame in range(spec["frames"]):
        command = _test_route_command(spec, frame)
        distance = command["distance"]
        root_x, root_z, waypoint = _point_on_route(points, distance)
        if explicit_heading is None:
            horizon = _test_route_command(spec, frame + 25)
            if horizon["command"] == (0.0, 0.0):
                start, stop = points[-2:]
                length = math.hypot(stop[0] - start[0], stop[1] - start[1])
                heading_x = (stop[0] - start[0]) / length
                heading_z = (stop[1] - start[1]) / length
            else:
                heading_x, heading_z = horizon["command"]
                length = math.hypot(heading_x, heading_z)
                heading_x, heading_z = heading_x / length, heading_z / length
        else:
            heading_x, heading_z = explicit_heading
        left_x, left_z = -heading_z, heading_x
        complete = command["complete"]
        commanded_speed = math.hypot(*command["command"])
        root_y = _fixture_height(shape, root_x, root_z)
        support = _support_height(shape, root_z)
        family, elevation = _family_elevation(shape, root_z)
        searched = int((frame + 1) % 3 == 0)
        row = runtime_row(
            frame, scene_id=spec["scene"], route=spec["route"],
            simulation_x=root_x, simulation_z=root_z,
            commanded_speed=commanded_speed,
            applied_speed=commanded_speed,
            route_waypoint=command["waypoint"], route_complete=int(complete),
            route_target_height=shape.get("top", .32),
            searched=searched, eligible_frame_count=(8 if searched else 0),
            evaluated_frame_count=(8 if searched else 0),
            considered_bound_count=(8 if searched else 0),
            direction_mask=(1 if commanded_speed == 0 else
                            SECTOR_MASKS[spec["sector"]]),
            speed_mask=(1 if commanded_speed == 0 else 2),
            elevation_mode=elevation,
            requested_family=family, active_family=family,
            source_family=family, source_terrain=family,
            source_name=f"{family}-production-fixture",
            source_index=(0 if family == "flat" else 1),
            runtime_root_surface_height=root_y,
            runtime_support_root_height=support,
            runtime_support_left_toe_height=support,
            runtime_support_right_toe_height=support,
            support_root_delta=support, support_left_toe_delta=support,
            support_right_toe_delta=support, support_height=support,
            support_retargeted_hips_y=.8 + support,
            ik_adjusted_hips_y=.8 + support,
            raw_selected_hips_y=.8 + support,
            inertialized_hips_y=.8 + support,
            rendered_hips_y=.8 + support,
        )
        root = (root_x, root_y, root_z)
        for axis, value in zip("xyz", root):
            row[f"terrain_root_point_{axis}"] = str(value)
        centers = []
        for sample in range(8):
            cx, cz, _ = _point_on_route(points, distance + .125 * (sample + 1))
            center = (cx, _fixture_height(shape, cx, cz), cz)
            centers.append(center)
            for axis, value in zip("xyz", center):
                row[f"terrain_center_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample}"] = str(center[1] - root_y)
            if sample < 4:
                for axis, value in zip("xyz", center):
                    row[f"terrain_point{sample}_{axis}"] = str(value)
        for sample, center_index in enumerate((1, 3, 5, 7)):
            center = centers[center_index]
            lx = center[0] + left_x * CORRIDOR_OFFSET
            lz = center[2] + left_z * CORRIDOR_OFFSET
            rx = center[0] - left_x * CORRIDOR_OFFSET
            rz = center[2] - left_z * CORRIDOR_OFFSET
            left = (lx, _fixture_height(shape, lx, lz), lz)
            right = (rx, _fixture_height(shape, rx, rz), rz)
            for label, point in (("left", left), ("right", right)):
                for axis, value in zip("xyz", point):
                    row[f"terrain_{label}_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample + 8}"] = str(left[1] - right[1])
        _refresh_query_bits(row)
        rows.append(row)
        classification = _test_logged_profile_classification(shape, row)
        observations.append(
            (classification.family, classification.elevation_mode))
        confidence = classification.confidence
        confidences.append(confidence)
        row["classifier_confidence"] = str(confidence)
    states = _test_claimed_confidence_replay(observations, confidences)
    for row, state, confidence in zip(rows, states, confidences):
        row["requested_family"] = state["requested_family"]
        row["active_family"] = state["active_family"]
        row["source_family"] = state["active_family"]
        source_index = {
            "flat": 0, "curb": 1, "slope": 2, "stair": 3,
        }[state["active_family"]]
        row["source_terrain"] = state["active_family"]
        row["source_name"] = f"{state['active_family']}-fixture"
        row["source_index"] = str(source_index)
        row["elevation_mode"] = str(state["active_elevation"])
        row["classifier_confidence"] = str(confidence)
        row["bank_transition"] = str(state["transitioned"])
        row["bank_transition_reason"] = state["reason"]
    _bind_fixture_source_provenance(rows)
    return rows


def replay_rows_from_claimed_confidence(
        spec, rows, *, state_confidences=None):
    shape = _route_shape(spec)
    points = shape["points"]
    observations = []
    for row in rows:
        classification = _test_logged_profile_classification(shape, row)
        observations.append(
            (classification.family, classification.elevation_mode))
    confidences = ([float(row["classifier_confidence"]) for row in rows]
                   if state_confidences is None else state_confidences)
    states = _test_claimed_confidence_replay(observations, confidences)
    for row, state in zip(rows, states):
        row["requested_family"] = state["requested_family"]
        row["active_family"] = state["active_family"]
        row["source_family"] = state["active_family"]
        source_index = {
            "flat": 0, "curb": 1, "slope": 2, "stair": 3,
        }[state["active_family"]]
        row["source_terrain"] = state["active_family"]
        row["source_name"] = f"{state['active_family']}-fixture"
        row["source_index"] = str(source_index)
        row["elevation_mode"] = str(state["active_elevation"])
        row["bank_transition"] = str(state["transitioned"])
        row["bank_transition_reason"] = state["reason"]
    _bind_fixture_source_provenance(rows)


def synthetic_baseline():
    return {
        "schema": "g1-motion-quality-baseline/v1",
        "ik_enabled": False,
        "sample_rate_hz": 25,
        "candidate_thresholds": {
            "flat_wrong_family_frames": {
                "comparison": "equal", "target": 0,
                "routes": list(COMPARISON_IDS[:2]),
            },
            "activated_terrain_wrong_family_frames": {
                "comparison": "equal", "target": 0,
                "routes": list(COMPARISON_IDS[2:]),
                "activation": "first accepted non-flat classifier activation",
            },
            "lateral_selected_cost_p95": {
                "comparison": "strictly_less_than_baseline",
                "baseline": 10.0, "route": COMPARISON_IDS[1],
            },
            "lateral_transition_rate": {
                "comparison": "strictly_less_than_baseline",
                "baseline": 0.75, "route": COMPARISON_IDS[1],
            },
            "stance_slip_total_m": {
                "comparison": "strictly_less_than_baseline",
                "baseline": 10.0,
                "aggregation": "sum over both feet and all routes",
            },
            "sole_worst_penetration_m": {
                "comparison": "strictly_less_than_baseline",
                "baseline": 0.5,
                "aggregation": "maximum over all routes",
            },
        },
    }


def route_rows(spec):
    rows = copy.deepcopy(quality_rows())
    family = spec["expected_family"]
    for index, row in enumerate(rows):
        row["scene_id"] = spec["scene"]
        row["route"] = spec["route"]
        row["direction_mask"] = str(SECTOR_MASKS[spec["sector"]])
        row["requested_family"] = family
        row["active_family"] = family
        row["source_family"] = family
        row["source_terrain"] = family
        row["source_name"] = f"{family}-fixture"
        row["route_complete"] = "1" if index == len(rows) - 1 else "0"
    return rows


def passing_fixture():
    routes = []
    logs = {}
    comparison = {
        ("flat", 0): COMPARISON_IDS[0],
        ("flat", 2): COMPARISON_IDS[1],
        ("stair", 0): COMPARISON_IDS[2],
        ("slope", 0): COMPARISON_IDS[3],
        ("stair", 2): COMPARISON_IDS[4],
    }
    tags = list(REQUIRED_TAGS)
    for family in ("flat", "slope", "stair"):
        for sector in range(8):
            route_id = f"{family}-fixed-{sector}"
            spec = {
                "id": route_id,
                "log": f"{route_id}.csv",
                "scene": f"{family}-scene",
                "route": f"sector-{sector}",
                "frames": 4,
                "expected_family": family,
                "sector": sector,
                "heading_case": "fixed",
                "heading_delta_degrees": 0.0,
                "coverage": tags if family == "flat" and sector == 0 else [],
            }
            if (family, sector) in comparison:
                spec["comparison_id"] = comparison[(family, sector)]
            routes.append(spec)
            logs[route_id] = route_rows(spec)
    for sector in range(8):
        route_id = f"flat-changed-{sector}"
        spec = {
            "id": route_id,
            "log": f"{route_id}.csv",
            "scene": "flat-scene",
            "route": f"changed-sector-{sector}",
            "frames": 4,
            "expected_family": "flat",
            "sector": sector,
            "heading_case": "changed",
            "heading_delta_degrees": -15.0 if sector % 2 else 15.0,
            "coverage": [],
        }
        routes.append(spec)
        logs[route_id] = route_rows(spec)
    mixed = {
        "id": "mixed-fixed-0",
        "log": "mixed-fixed-0.csv",
        "scene": "mixed-multilevel",
        "route": "tangent-level-boundary",
        "frames": 4,
        "expected_family": "curb",
        "sector": 0,
        "heading_case": "fixed",
        "heading_delta_degrees": 0.0,
        "coverage": [],
        "comparison_id": COMPARISON_IDS[5],
    }
    routes.append(mixed)
    logs[mixed["id"]] = route_rows(mixed)
    matrix = {
        "schema": "g1-motion-quality-matrix/v1",
        "sample_rate_hz": 25,
        "routes": routes,
    }
    return synthetic_baseline(), matrix, logs


class ProductionContractRedTests(unittest.TestCase):
    def test_checker_owns_the_exact_command_roster_and_durations(self):
        expected = expected_production_roster()
        self.assertEqual(len(expected), 37)
        self.assertEqual(quality.PRODUCTION_ROUTE_SPECS, tuple(expected))
        self.assertEqual(
            sum(spec["frames"] for spec in expected),
            16 * 100 + 21 * 800)

    def test_fabricated_four_frame_matrix_cannot_supply_its_own_semantics(self):
        baseline, matrix, logs = passing_fixture()
        with self.assertRaisesRegex(ValueError, "production"):
            quality._route_specs(matrix)

    def test_exact_ids_cannot_hide_a_fabricated_four_frame_matrix(self):
        routes = []
        logs = {}
        for index, locked in enumerate(expected_production_roster()):
            family = locked["expected_family"]
            if family == "mixed":
                family = "curb"
            candidate = {
                "id": locked["id"], "log": f"{locked['id']}.csv",
                "scene": f"invented-{family}-scene",
                "route": f"invented-sector-{locked['sector']}",
                "frames": 4, "expected_family": family,
                "sector": locked["sector"],
                "heading_case": locked["heading_case"],
                "heading_delta_degrees": (
                    0.0 if locked["heading_case"] == "fixed" else 15.0),
                "coverage": list(REQUIRED_TAGS) if index == 0 else [],
            }
            if "comparison_id" in locked:
                candidate["comparison_id"] = locked["comparison_id"]
            routes.append(candidate)
            logs[candidate["id"]] = route_rows(candidate)
        matrix = {
            "schema": "g1-motion-quality-matrix/v1",
            "sample_rate_hz": 25,
            "routes": routes,
        }
        with self.assertRaisesRegex(ValueError, "production"):
            quality._route_specs(matrix)

    def test_checker_locks_canonical_baseline_and_full_candidate_totals(self):
        self.assertEqual(
            quality.CANONICAL_BASELINE_SHA256,
            "fc3e0e94173ee7156f5b5b0f3ff0a6da7abc98148a1749fbc94ca270b1a247bd")
        self.assertEqual(quality.FULL_CANDIDATE_SOURCES, 15_815)
        self.assertEqual(quality.FULL_CANDIDATE_FRAMES, 3_970_932)

    def test_canonical_baseline_is_checked_as_bytes_and_as_parsed_content(self):
        path = pathlib.Path(__file__).with_name(
            "test_g1_motion_quality.py").parents[1] / \
            "fixtures" / "g1_motion_quality_baseline.json"
        payload = path.read_bytes()
        baseline = json.loads(payload)
        quality._validate_canonical_baseline(baseline, payload)

        changed = copy.deepcopy(baseline)
        changed["candidate_thresholds"][
            "lateral_transition_rate"]["baseline"] = 999.0
        with self.assertRaisesRegex(ValueError, "parsed canonical baseline"):
            quality._validate_canonical_baseline(changed, payload)
        with self.assertRaisesRegex(ValueError, "canonical baseline SHA-256"):
            quality._validate_canonical_baseline(
                baseline, payload.replace(b"0.05", b"0.06", 1))

    def test_candidate_identity_requires_full_manifest_binary_and_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            manifest_path = candidate / "manifest.json"
            manifest = {
                "schema": "g1-terrain-artifacts/v3",
                "total_clips": 15_815,
                "database_frames": 3_970_932,
                "diagnostic_mode": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            binary = root / "controller"
            binary.write_bytes(b"release-controller")
            matrix = {
                "candidate": {
                    "artifact_directory": str(candidate.resolve()),
                    "manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()).hexdigest(),
                    "binary_path": str(binary.resolve()),
                    "binary_sha256": hashlib.sha256(
                        binary.read_bytes()).hexdigest(),
                },
                "environment": {
                    "G1_TERRAIN_DIR": str(candidate.resolve()),
                    "MM_TERRAIN_WEIGHT": "4", "MM_IK": "0",
                },
            }
            calls = []

            def validate(path, *, full_source_validation):
                calls.append((path, full_source_validation))
                return {"frames": 3_970_932, "clips": 15_815,
                        "bones": 31, "scenes": 14, "source_rows": 4_250}

            with self.assertRaisesRegex(ValueError, "regular executable"):
                quality._validate_candidate_identity(matrix, validate)
            binary.chmod(0o755)
            report = quality._validate_candidate_identity(matrix, validate)
            self.assertEqual(calls, [(str(candidate.resolve()), True)])
            self.assertEqual(report["manifest_sha256"],
                             matrix["candidate"]["manifest_sha256"])
            self.assertEqual(report["binary_sha256"],
                             matrix["candidate"]["binary_sha256"])
            self.assertTrue(report["validator_passed"])

            for field, value, message in (
                    ("diagnostic_mode", True, "non-diagnostic"),
                    ("total_clips", 15_814, "source total"),
                    ("database_frames", 3_970_931, "frame total")):
                changed = dict(manifest)
                changed[field] = value
                manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                matrix["candidate"]["manifest_sha256"] = hashlib.sha256(
                    manifest_path.read_bytes()).hexdigest()
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, message):
                        quality._validate_candidate_identity(matrix, validate)

            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            matrix["candidate"]["manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()
            binary.write_bytes(b"substituted-controller")
            with self.assertRaisesRegex(ValueError, "binary SHA-256"):
                quality._validate_candidate_identity(matrix, validate)

            binary.write_bytes(b"release-controller")
            binary.chmod(0o755)
            matrix["environment"]["MM_TERRAIN_WEIGHT"] = "0"
            with self.assertRaisesRegex(ValueError, "environment"):
                quality._validate_candidate_identity(matrix, validate)

            matrix["environment"]["MM_TERRAIN_WEIGHT"] = "4"
            linked = root / "linked-controller"
            linked.symlink_to(binary)
            matrix["candidate"]["binary_path"] = str(linked.resolve())
            # Preserve the lexical symlink path instead of resolve(), because
            # the production evidence contract must reject symlink identity.
            matrix["candidate"]["binary_path"] = str(linked.absolute())
            with self.assertRaisesRegex(ValueError, "regular executable"):
                quality._validate_candidate_identity(matrix, validate)

    def test_authenticated_candidate_artifacts_own_scenes_and_source_metadata(self):
        self.assertTrue(
            hasattr(quality, "_load_authenticated_candidate_artifacts"),
            "checker must load the manifest-to-heightfield hash chain")
        self.assertTrue(
            hasattr(quality, "_resolve_authenticated_source"),
            "checker must resolve source metadata from authenticated ranges")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            candidate, manifest, _ = write_candidate_fixture(root)
            manifest_sha = hashlib.sha256(
                (candidate / "manifest.json").read_bytes()).hexdigest()
            artifacts = quality._load_authenticated_candidate_artifacts(
                str(candidate.resolve()), manifest_sha)
            source = quality._resolve_authenticated_source(
                artifacts, selected_frame=2_100, current_frame=2_101)
            self.assertEqual(source["terrain_family"], "slope")
            self.assertEqual(source["name"], "slope-fixture")
            self.assertEqual(
                set(artifacts["scenes"]), set(FIXTURE_SCENE_ROUTES))

            terrain = candidate / "scenes" / "ramp-10-up-down" / "terrain.bin"
            terrain_payload = terrain.read_bytes()
            terrain.write_bytes(
                terrain_payload[:-4] + struct.pack("<f", 1.0))
            with self.assertRaisesRegex(ValueError, "heightfield SHA-256"):
                quality._load_authenticated_candidate_artifacts(
                    str(candidate.resolve()), manifest_sha)
            terrain.write_bytes(terrain_payload)

            # Authenticated source ownership is interval-derived, not accepted
            # from the CSV's claimed source index/family strings.
            manifest["sources"][2]["range_start"] = 2_200
            (candidate / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            changed_sha = hashlib.sha256(
                (candidate / "manifest.json").read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "source"):
                quality._load_authenticated_candidate_artifacts(
                    str(candidate.resolve()), changed_sha)

    def test_matrix_requires_hash_bound_run_origin_evidence(self):
        matrix = {
            "schema": "g1-motion-quality-matrix/v1",
            "sample_rate_hz": 25,
            "candidate": {
                "artifact_directory": "/candidate",
                "manifest_sha256": "a" * 64,
                "binary_path": "/controller",
                "binary_sha256": "b" * 64,
            },
            "environment": {
                "G1_TERRAIN_DIR": "/candidate",
                "MM_TERRAIN_WEIGHT": "4", "MM_IK": "0",
            },
            "routes": [
                {"id": spec["id"], "log": f"{spec['id']}.csv"}
                for spec in expected_production_roster()
            ],
            "scene_cycle": {
                "log": "scene-cycle.csv",
                "cleanup": "scene-cycle-cleanup.json",
            },
        }
        with self.assertRaisesRegex(ValueError, "hash.*origin|origin.*hash"):
            quality._route_specs(matrix)

    def test_run_origin_binds_candidate_environment_and_command(self):
        self.assertTrue(
            hasattr(quality, "_validate_run_origin"),
            "checker needs a hashed per-invocation origin validator")
        spec = expected_production_roster()[0]
        candidate = {
            "artifact_directory": "/candidate",
            "manifest_sha256": "a" * 64,
            "binary_path": "/controller",
            "binary_sha256": "b" * 64,
        }
        log_path = pathlib.Path("/logs/flat-fixed-forward.csv")
        origin = {
            "schema": "g1-motion-quality-run-origin/v1",
            "binary_sha256": candidate["binary_sha256"],
            "manifest_sha256": candidate["manifest_sha256"],
            "argv": [candidate["binary_path"]],
            "environment": {
                "G1_TERRAIN_DIR": candidate["artifact_directory"],
                "MM_TERRAIN_WEIGHT": "4", "MM_IK": "0",
                "MM_TERRAIN_SCENE": spec["scene"],
                "MM_TEST_MODE": "route", "MM_TEST_ROUTE": spec["route"],
                "MM_TEST_FRAMES": str(spec["frames"]),
                "MM_TEST_HEADING": spec["heading"],
                "MM_LOG": str(log_path),
            },
        }
        self.assertEqual(
            quality._validate_run_origin(spec, candidate, log_path, origin),
            origin)
        for name, mutate in {
                "binary": lambda value: value.update(binary_sha256="c" * 64),
                "manifest": lambda value: value.update(
                    manifest_sha256="c" * 64),
                "argv": lambda value: value.update(argv=["/other"]),
                "scene": lambda value: value["environment"].update(
                    MM_TERRAIN_SCENE="other"),
                "heading": lambda value: value["environment"].update(
                    MM_TEST_HEADING="positive-x"),
                "log": lambda value: value["environment"].update(
                    MM_LOG="/logs/other.csv"),
        }.items():
            changed = copy.deepcopy(origin)
            mutate(changed)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "run origin"):
                    quality._validate_run_origin(
                        spec, candidate, log_path, changed)

    def test_evidence_file_hashes_are_verified_before_parsing(self):
        self.assertTrue(
            hasattr(quality, "_verified_evidence_path"),
            "checker needs a raw evidence hash verifier")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "route.csv"
            evidence.write_bytes(b"immutable runtime bytes\n")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            path, computed = quality._verified_evidence_path(
                root, "route.csv", digest, "route evidence")
            self.assertEqual(path, evidence.resolve())
            self.assertEqual(computed, digest)

            evidence.write_bytes(b"tampered runtime bytes\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                quality._verified_evidence_path(
                    root, "route.csv", digest, "route evidence")
            evidence.write_bytes(b"immutable runtime bytes\n")
            linked = root / "linked.csv"
            linked.symlink_to(evidence)
            with self.assertRaisesRegex(ValueError, "regular file"):
                quality._verified_evidence_path(
                    root, "linked.csv", digest, "route evidence")


class ProductionRouteOracleRedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {spec["id"]: spec for spec in expected_production_roster()}
        cls.temporary = tempfile.TemporaryDirectory()
        candidate, _, _ = write_candidate_fixture(
            pathlib.Path(cls.temporary.name))
        manifest_sha256 = hashlib.sha256(
            (candidate / "manifest.json").read_bytes()).hexdigest()
        cls.artifacts = quality._load_authenticated_candidate_artifacts(
            str(candidate.resolve()), manifest_sha256)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def metric(self, identifier, rows=None):
        spec = self.specs[identifier]
        if rows is None:
            rows = production_route_rows(spec)
        return quality._route_metric(spec, rows, self.artifacts)

    def test_representative_physical_routes_pass_the_locked_oracles(self):
        for identifier in (
                "flat-fixed-forward", "flat-changed-forward",
                "slope-fixed-forward", "slope-fixed-back-right",
                "stair-fixed-forward", "stair-unseen-forward",
                "mixed-tangent", "mixed-full-course"):
            with self.subTest(identifier=identifier):
                report = self.metric(identifier)
                self.assertTrue(report["passed"], report["failures"])

    def test_each_exact_terrain_phase_mutation_is_rejected(self):
        cases = (
            ("ramp-ascent-family", "slope-fixed-forward", 2.50,
             {"requested_family": "flat", "active_family": "flat",
              "source_family": "flat", "elevation_mode": "0"}),
            ("ramp-landing-family", "slope-fixed-forward", 5.00,
             {"requested_family": "slope", "active_family": "slope",
              "source_family": "slope"}),
            ("ramp-descent-elevation", "slope-fixed-forward", 7.00,
             {"elevation_mode": "1"}),
            ("ramp-flat-exit", "slope-fixed-forward", 8.50,
             {"requested_family": "slope", "active_family": "slope",
              "source_family": "slope", "elevation_mode": "-1"}),
            ("stair-ascent-elevation", "stair-fixed-forward", 2.50,
             {"elevation_mode": "0"}),
            ("stair-landing-family", "stair-fixed-forward", 3.50,
             {"requested_family": "stair", "active_family": "stair",
              "source_family": "stair"}),
            ("stair-descent-elevation", "stair-fixed-forward", 5.30,
             {"elevation_mode": "0"}),
            ("cross-grade-family", "slope-fixed-back-right", 4.00,
             {"requested_family": "flat", "active_family": "flat",
              "source_family": "flat"}),
        )
        for name, identifier, target_z, changes in cases:
            rows = production_route_rows(self.specs[identifier])
            row = min(rows, key=lambda item: abs(
                float(item["simulation_z"]) - target_z))
            row.update(changes)
            with self.subTest(name=name):
                report = self.metric(identifier, rows)
                self.assertFalse(report["passed"], name)
                self.assertTrue(any("phase oracle" in failure
                                    for failure in report["failures"]))

    def test_corridor_geometry_authenticates_heading_not_just_direction_mask(self):
        identifier = "flat-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        row = rows[20]
        # Forge positive-X heading geometry while retaining the exact forward
        # direction mask.  Runtime row validation alone accepts these finite
        # points; the quality checker must bind them to MM_TEST_HEADING=forward.
        for sample, center_index in enumerate((1, 3, 5, 7)):
            cx = float(row[f"terrain_center_point{center_index}_x"])
            cz = float(row[f"terrain_center_point{center_index}_z"])
            row[f"terrain_left_point{sample}_x"] = str(cx)
            row[f"terrain_left_point{sample}_z"] = str(cz + CORRIDOR_OFFSET)
            row[f"terrain_right_point{sample}_x"] = str(cx)
            row[f"terrain_right_point{sample}_z"] = str(cz - CORRIDOR_OFFSET)
        report = self.metric(identifier, rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("heading geometry" in failure
                            for failure in report["failures"]))

    def test_descriptor_values_must_match_logged_physical_points(self):
        identifier = "slope-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        row = rows[120]
        row["terrain0"] = str(float(row["terrain0"]) + 0.05)
        _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("descriptor geometry" in failure
                            for failure in report["failures"]))

    def test_descriptor_points_must_match_authenticated_heightfield_y(self):
        identifier = "slope-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        row = rows[120]
        for name in ("terrain_root_point_y", "runtime_root_surface_height"):
            row[name] = str(float(row[name]) + 1.0)
        for sample in range(8):
            name = f"terrain_center_point{sample}_y"
            row[name] = str(float(row[name]) + 1.0)
            if sample < 4:
                legacy = f"terrain_point{sample}_y"
                row[legacy] = str(float(row[legacy]) + 1.0)
        for side in ("left", "right"):
            for sample in range(4):
                name = f"terrain_{side}_point{sample}_y"
                row[name] = str(float(row[name]) + 1.0)
        report = self.metric(identifier, rows)
        self.assertFalse(
            report["passed"],
            "self-consistent +1 m CSV Y claims must not replace G1HF truth")

    def test_descriptor_queries_round_to_f32_and_heights_require_exact_bits(self):
        identifier = "slope-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        row = rows[120]
        original_z = np.float32(float(row["terrain_center_point0_z"]))
        same_f32_z = np.nextafter(
            float(original_z), math.inf, dtype=np.float64)
        self.assertEqual(_f32(same_f32_z), float(original_z))
        row["terrain_center_point0_z"] = str(same_f32_z)
        row["terrain_point0_z"] = str(same_f32_z)
        report = self.metric(identifier, rows)
        self.assertTrue(report["passed"], report["failures"])

        rows = production_route_rows(self.specs[identifier])
        row = rows[120]
        original_y = np.float32(float(row["terrain_center_point0_y"]))
        changed_y = np.nextafter(original_y, np.float32(math.inf))
        self.assertNotEqual(changed_y.view(np.uint32),
                            original_y.view(np.uint32))
        row["terrain_center_point0_y"] = str(float(changed_y))
        row["terrain_point0_y"] = str(float(changed_y))
        row["terrain0"] = str(_f32(
            float(changed_y) - float(row["terrain_root_point_y"])))
        _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertFalse(report["passed"], "one Y ULP must fail authentication")

    def test_descriptor_center_samples_cannot_collapse_to_the_root(self):
        identifier = "flat-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        row = rows[20]
        root = tuple(float(row[f"terrain_root_point_{axis}"])
                     for axis in "xyz")
        heading_x, heading_z = HEADING_VECTORS["forward"]
        left_offset = (-heading_z * CORRIDOR_OFFSET,
                       heading_x * CORRIDOR_OFFSET)
        for sample in range(8):
            for axis, value in zip("xyz", root):
                row[f"terrain_center_point{sample}_{axis}"] = str(value)
                if sample < 4:
                    row[f"terrain_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample}"] = "0"
        for sample in range(4):
            left = (root[0] + left_offset[0], root[1],
                    root[2] + left_offset[1])
            right = (root[0] - left_offset[0], root[1],
                     root[2] - left_offset[1])
            for side, point in (("left", left), ("right", right)):
                for axis, value in zip("xyz", point):
                    row[f"terrain_{side}_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample + 8}"] = "0"
        _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertFalse(
            report["passed"],
            "eight collapsed centers must not claim a one-metre horizon")

    def test_real_task11_predicted_arc_drift_is_not_route_projected(self):
        identifier = "flat-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        row = rows[20]
        shape = _route_shape(self.specs[identifier])
        root = tuple(float(row[f"terrain_root_point_{axis}"])
                     for axis in "xyz")
        # Actual Task 11 telemetry reached 1.00119853 instead of the route-
        # projected 1.04502023 at horizon sample 7: a 0.0438217 m progress
        # difference while the predicted-trajectory arc remained valid.
        progress_difference = .0438217
        arc_step = .125 - progress_difference / 8.0
        centers = []
        for sample in range(8):
            x = root[0]
            z = root[2] + arc_step * (sample + 1)
            center = (x, _fixture_height(shape, x, z), z)
            centers.append(center)
            for axis, value in zip("xyz", center):
                row[f"terrain_center_point{sample}_{axis}"] = str(value)
                if sample < 4:
                    row[f"terrain_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample}"] = str(_f32(
                _f32(center[1]) - _f32(root[1])))
        for sample, center_index in enumerate((1, 3, 5, 7)):
            center = centers[center_index]
            for side, sign in (("left", -1.0), ("right", 1.0)):
                x, z = center[0] + sign * CORRIDOR_OFFSET, center[2]
                point = (x, _fixture_height(shape, x, z), z)
                for axis, value in zip("xyz", point):
                    row[f"terrain_{side}_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample + 8}"] = str(_f32(
                _f32(float(row[f"terrain_left_point{sample}_y"])) -
                _f32(float(row[f"terrain_right_point{sample}_y"]))))
        _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertTrue(report["passed"], report["failures"])

    def test_classifier_confidence_is_independently_bounded(self):
        identifier = "slope-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        target = next(row for row in rows
                      if row["requested_family"] == "slope" and
                      float(row["classifier_confidence"]) > .64)
        target["classifier_confidence"] = "0"
        replay_rows_from_claimed_confidence(self.specs[identifier], rows)
        report = self.metric(identifier, rows)
        self.assertFalse(
            report["passed"],
            "logged confidence and its self-consistent state are not an oracle")

    def test_stair_confidence_interval_allows_runtime_raster_boundary(self):
        identifier = "stair-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        target = next(row for row in rows if math.isclose(
            float(row["classifier_confidence"]), 2.0 / 3.0,
            rel_tol=0.0, abs_tol=1e-9))
        target["classifier_confidence"] = "0.59927"
        replay_rows_from_claimed_confidence(self.specs[identifier], rows)
        report = self.metric(identifier, rows)
        self.assertTrue(
            report["passed"],
            "0.66667 reconstruction must permit observed 0.59927 runtime state")

    def test_plausible_logged_confidence_owns_the_exact_hysteresis_branch(self):
        identifier = "stair-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        target_index = next(index for index, row in enumerate(rows)
                            if math.isclose(
                                float(row["classifier_confidence"]), 2.0 / 3.0,
                                rel_tol=0.0, abs_tol=1e-9))
        rows[target_index]["classifier_confidence"] = ".65"
        forged_state_confidences = [
            float(row["classifier_confidence"]) for row in rows
        ]
        forged_state_confidences[target_index] = .599
        replay_rows_from_claimed_confidence(
            self.specs[identifier], rows,
            state_confidences=forged_state_confidences)
        report = self.metric(identifier, rows)
        self.assertFalse(
            report["passed"],
            "accepted .65 telemetry must take the >=.60 hysteresis branch")

    def test_selected_source_claim_is_authenticated_independently(self):
        identifier = "stair-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        target_index = next(index for index, row in enumerate(rows)
                            if row["active_family"] == "stair")
        # Preserve a runtime-valid two-transition trace, but make the first
        # accepted stair pose actually select an authenticated curb range
        # while retaining forged stair provenance claims.
        rows[target_index].update(
            database_frame="1001", selected_database_frame="1000",
            range="1", source_range="1")
        rows[target_index + 1].update(
            query_database_frame="1001", query_range="1",
            transitioned="1", searched="1", eligible_frame_count="8",
            evaluated_frame_count="8", considered_bound_count="8")
        report = self.metric(identifier, rows)
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["bank_correctness"]["wrong_family_frames"], 1)
        self.assertEqual(
            report["bank_correctness"]["source_claim_mismatched_frames"], 1)

    def test_completion_requires_progress_endpoint_and_tangent_waypoint(self):
        identifier = "flat-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        for row in rows:
            row["simulation_x"] = "0"
            row["simulation_z"] = "0"
        report = self.metric(identifier, rows)
        self.assertTrue(report["route_complete"])
        self.assertFalse(report["passed"])

        identifier = "mixed-tangent"
        rows = production_route_rows(self.specs[identifier])
        for row in rows:
            row["route_waypoint"] = "1"
        report = self.metric(identifier, rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("tangent waypoint" in failure
                            for failure in report["failures"]))

    def test_weight_runtime_flags_search_and_length_are_hard_gates(self):
        mutations = {
            "terrain-weight": lambda rows: rows[10].update(
                effective_terrain_weight="0"),
            "matching": lambda rows: rows[10].update(matching_enabled="0"),
            "support": lambda rows: rows[10].update(
                support_retargeting_enabled="0"),
            "ik": lambda rows: rows[10].update(ik_enabled="1"),
            "search": lambda rows: [row.update(
                searched="0", eligible_frame_count="0",
                evaluated_frame_count="0", considered_bound_count="0")
                for row in rows],
            "underlength": lambda rows: rows.pop(),
            "completion": lambda rows: rows[-1].update(route_complete="0"),
        }
        for name, mutate in mutations.items():
            rows = production_route_rows(self.specs["slope-fixed-forward"])
            mutate(rows)
            with self.subTest(name=name):
                self.assertFalse(self.metric(
                    "slope-fixed-forward", rows)["passed"])

    def test_tangent_requires_flat_centerline_and_partial_support_corridor(self):
        identifier = "mixed-tangent"
        rows = production_route_rows(self.specs[identifier])
        activated = [row for row in rows if max(
            abs(float(row[f"terrain{sample}"])) for sample in range(8, 12)
        ) > 0.02]
        self.assertTrue(activated)
        for row in activated:
            for sample in range(8, 12):
                row[f"terrain{sample}"] = "0"
            for sample in range(4):
                row[f"terrain_left_point{sample}_y"] = "0"
                row[f"terrain_right_point{sample}_y"] = "0"
            _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertFalse(report["passed"])
        self.assertTrue(any("partial-support" in failure
                            for failure in report["failures"]))

    def test_tangent_corridor_preserves_fixed_forward_heading_through_turn(self):
        identifier = "mixed-tangent"
        spec = self.specs[identifier]
        rows = production_route_rows(spec)
        shape = _route_shape(spec)
        for row in rows:
            heading = (0.0, 1.0)
            left_x, left_z = -heading[1], heading[0]
            for sample, center_index in enumerate((1, 3, 5, 7)):
                center = tuple(float(row[
                    f"terrain_center_point{center_index}_{axis}"])
                               for axis in "xyz")
                left = (
                    center[0] + left_x * CORRIDOR_OFFSET,
                    _fixture_height(
                        shape, center[0] + left_x * CORRIDOR_OFFSET,
                        center[2] + left_z * CORRIDOR_OFFSET),
                    center[2] + left_z * CORRIDOR_OFFSET,
                )
                right = (
                    center[0] - left_x * CORRIDOR_OFFSET,
                    _fixture_height(
                        shape, center[0] - left_x * CORRIDOR_OFFSET,
                        center[2] - left_z * CORRIDOR_OFFSET),
                    center[2] - left_z * CORRIDOR_OFFSET,
                )
                for side, point in (("left", left), ("right", right)):
                    for axis, value in zip("xyz", point):
                        row[f"terrain_{side}_point{sample}_{axis}"] = str(value)
                row[f"terrain{sample + 8}"] = str(left[1] - right[1])
            _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertTrue(report["passed"], report["failures"])
        self.assertAlmostEqual(
            float(rows[79]["terrain_left_point0_x"])
            - float(rows[79]["terrain_center_point1_x"]),
            -CORRIDOR_OFFSET, places=12)
        self.assertAlmostEqual(
            float(rows[105]["terrain_left_point0_x"])
            - float(rows[105]["terrain_center_point1_x"]),
            -CORRIDOR_OFFSET, places=12)

        forged = production_route_rows(spec)
        row = forged[79]
        travel = (0.6200000047683716 /
                  math.hypot(0.6200000047683716, 2.0),
                  2.0 / math.hypot(0.6200000047683716, 2.0))
        left_x, left_z = -travel[1], travel[0]
        for sample, center_index in enumerate((1, 3, 5, 7)):
            center = tuple(float(row[
                f"terrain_center_point{center_index}_{axis}"])
                           for axis in "xyz")
            for side, sign in (("left", 1.0), ("right", -1.0)):
                x = center[0] + sign * left_x * CORRIDOR_OFFSET
                z = center[2] + sign * left_z * CORRIDOR_OFFSET
                point = (x, _fixture_height(shape, x, z), z)
                for axis, value in zip("xyz", point):
                    row[f"terrain_{side}_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample + 8}"] = str(
                float(row[f"terrain_left_point{sample}_y"]) -
                float(row[f"terrain_right_point{sample}_y"]))
        _refresh_query_bits(row)
        self.assertFalse(self.metric(identifier, forged)["passed"])

    def test_exact_route_schedule_and_runtime_provenance_are_hard_gates(self):
        identifier = "flat-fixed-forward"
        mutations = {
            "early-waypoint": lambda rows: rows[10].update(route_waypoint="2"),
            "early-complete": lambda rows: rows[10].update(route_complete="1"),
            "command-speed": lambda rows: rows[10].update(commanded_speed=".25"),
            "generation": lambda rows: [row.update(scene_generation="7")
                                          for row in rows],
            "reset-count": lambda rows: [row.update(scene_reset_count="7")
                                           for row in rows],
            "scene-switch": lambda rows: [row.update(scene_switch_failed="1")
                                            for row in rows],
            "model-counts": lambda rows: [row.update(
                model_load_count="2", model_unload_count="1")
                for row in rows],
            "progress-regression": lambda rows: rows[20].update(
                simulation_z=str(float(rows[19]["simulation_z"]) - .10)),
        }
        for name, mutate in mutations.items():
            rows = production_route_rows(self.specs[identifier])
            mutate(rows)
            with self.subTest(name=name):
                report = self.metric(identifier, rows)
                self.assertFalse(report["passed"], name)

    def test_predicted_masks_may_remain_moving_during_hold_and_completion(self):
        identifier = "slope-fixed-forward"
        rows = production_route_rows(self.specs[identifier])
        stopped = [row for row in rows
                   if float(row["commanded_speed"]) == 0.0]
        self.assertTrue(any(int(row["route_complete"]) == 0
                            for row in stopped), "fixture needs landing hold")
        self.assertTrue(any(int(row["route_complete"]) == 1
                            for row in stopped), "fixture needs completion")
        for row in stopped:
            row["direction_mask"] = str(SECTOR_MASKS[0])
            row["speed_mask"] = "2"
        report = self.metric(identifier, rows)
        self.assertTrue(report["passed"], report["failures"])

    def test_descriptor_extension_after_stop_need_not_follow_route(self):
        identifier = "flat-fixed-right"
        rows = production_route_rows(self.specs[identifier])
        row = next(item for item in rows if int(item["route_complete"]) == 1)
        shape = _route_shape(self.specs[identifier])
        root = tuple(float(row[f"terrain_root_point_{axis}"])
                     for axis in "xyz")
        centers = []
        for sample in range(8):
            x, z = root[0] - .125 * (sample + 1), root[2]
            center = (x, _fixture_height(shape, x, z), z)
            centers.append(center)
            for axis, value in zip("xyz", center):
                row[f"terrain_center_point{sample}_{axis}"] = str(value)
                if sample < 4:
                    row[f"terrain_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample}"] = str(center[1] - root[1])
        for sample, center_index in enumerate((1, 3, 5, 7)):
            center = centers[center_index]
            for side, sign in (("left", -1.0), ("right", 1.0)):
                x, z = center[0], center[2] + sign * CORRIDOR_OFFSET
                point = (x, _fixture_height(shape, x, z), z)
                for axis, value in zip("xyz", point):
                    row[f"terrain_{side}_point{sample}_{axis}"] = str(value)
            row[f"terrain{sample + 8}"] = str(
                float(row[f"terrain_left_point{sample}_y"]) -
                float(row[f"terrain_right_point{sample}_y"]))
        _refresh_query_bits(row)
        report = self.metric(identifier, rows)
        self.assertTrue(report["passed"], report["failures"])

    def test_gate_c_ramp05_requires_authentic_runtime_slope_activation(self):
        ramp = production_route_rows(
            self.specs["gate-c-ramp-05-forward"])
        report = self.metric("gate-c-ramp-05-forward", ramp)
        self.assertTrue(report["passed"], report["failures"])
        self.assertTrue(any(row["source_family"] == "slope" for row in ramp))

    def test_gate_c_and_mixed_checks_are_composed(self):

        stair = production_route_rows(self.specs["stair-fixed-forward"])
        for row in stair:
            row["runtime_support_root_height"] = "0"
            row["runtime_support_left_toe_height"] = "0"
            row["runtime_support_right_toe_height"] = "0"
            row["support_root_delta"] = "0"
            row["support_left_toe_delta"] = "0"
            row["support_right_toe_delta"] = "0"
            row["support_height"] = "0"
            row["support_retargeted_hips_y"] = ".8"
            row["ik_adjusted_hips_y"] = ".8"
            row["rendered_hips_y"] = ".8"
        self.assertFalse(self.metric("stair-fixed-forward", stair)["passed"])

        mixed = production_route_rows(self.specs["mixed-full-course"])
        for row in mixed:
            z = float(row["simulation_z"])
            if 6.85 < z < 7.35:
                row["runtime_support_root_height"] = ".32"
        self.assertFalse(self.metric("mixed-full-course", mixed)["passed"])

        mixed = production_route_rows(self.specs["mixed-full-course"])
        mixed[200]["support_retargeted_hips_y"] = str(
            float(mixed[200]["support_retargeted_hips_y"]) + .01)
        report = self.metric("mixed-full-course", mixed)
        self.assertFalse(report["passed"])
        self.assertTrue(any("physical completion gate" in failure
                            for failure in report["failures"]))

        for name, behavior in (
                ("none", {"return_value": None}),
                ("raised", {"side_effect": ValueError("bad Gate C")})):
            mixed = production_route_rows(self.specs["mixed-full-course"])
            with self.subTest(composed_gate=name), mock.patch.object(
                    quality, "check_gate_c", **behavior):
                report = self.metric("mixed-full-course", mixed)
                self.assertFalse(report["physical_completion"]["passed"])


class SceneCycleCleanupRedTests(unittest.TestCase):
    def passing(self):
        rows = scene_cycle_rows(list(LOCKED_SCENE_IDS), dwell=25, cycles=2)
        cleanup = {
            "exit_code": 0, "live_model_count": 0, "log_closed": True,
            "model_load_count": 28, "model_unload_count": 28,
            "motion_pack_load_count": 1, "window_closed": True,
        }
        return rows, cleanup

    def test_exact_700_frame_gate_f_and_normal_cleanup_pass(self):
        rows, cleanup = self.passing()
        report = quality._scene_cycle_metric(rows, cleanup)
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["gate_f"]["frames"], 700)
        self.assertEqual(report["gate_f"]["generations"], 28)
        self.assertEqual(report["gate_f"]["complete_cycles"], 2)
        self.assertEqual(report["gate_f"]["model_loads"], 28)
        self.assertEqual(
            report["gate_f"]["model_unloads_before_final_cleanup"], 27)
        self.assertEqual(report["cleanup"]["model_unload_count"], 28)

    def test_scene_cycle_length_order_switch_and_cleanup_mutations_fail(self):
        cases = {
            "underlength": lambda rows, cleanup: rows.pop(),
            "switch-failure": lambda rows, cleanup: rows[100].update(
                scene_switch_failed="1"),
            "wrong-scene": lambda rows, cleanup: rows[25].update(
                scene_id="stairs-standard"),
            "cleanup-unload": lambda rows, cleanup: cleanup.update(
                model_unload_count=27, live_model_count=1),
            "cleanup-log": lambda rows, cleanup: cleanup.update(
                log_closed=False),
            "cleanup-exit": lambda rows, cleanup: cleanup.update(exit_code=2),
        }
        for name, mutate in cases.items():
            rows, cleanup = self.passing()
            mutate(rows, cleanup)
            with self.subTest(name=name):
                report = quality._scene_cycle_metric(rows, cleanup)
                self.assertFalse(report["passed"], name)

    def test_cleanup_requires_exact_json_scalar_types(self):
        for name, mutate in (
                ("boolean-exit-code", lambda value: value.update(
                    exit_code=False)),
                ("integer-log-closed", lambda value: value.update(
                    log_closed=1)),
                ("boolean-live-count", lambda value: value.update(
                    live_model_count=False))):
            rows, cleanup = self.passing()
            mutate(cleanup)
            with self.subTest(name=name):
                report = quality._scene_cycle_metric(rows, cleanup)
                self.assertFalse(report["passed"], name)

    def test_gate_f_requires_a_composed_report(self):
        rows, cleanup = self.passing()
        with mock.patch.object(quality, "check_gate_f", return_value=None):
            report = quality._scene_cycle_metric(rows, cleanup)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["gate_f"])


class ProductionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.temporary.name)
        cls.root = root
        cls.candidate, _, cls.binary = write_candidate_fixture(root)
        cls.manifest_path = cls.candidate / "manifest.json"
        cls.matrix = {
            "schema": "g1-motion-quality-matrix/v1",
            "sample_rate_hz": 25,
            "candidate": {
                "artifact_directory": str(cls.candidate.resolve()),
                "manifest_sha256": hashlib.sha256(
                    cls.manifest_path.read_bytes()).hexdigest(),
                "binary_path": str(cls.binary.resolve()),
                "binary_sha256": hashlib.sha256(
                    cls.binary.read_bytes()).hexdigest(),
            },
            "environment": {
                "G1_TERRAIN_DIR": str(cls.candidate.resolve()),
                "MM_TERRAIN_WEIGHT": "4", "MM_IK": "0",
            },
            "routes": [
                {
                    "id": spec["id"], "log": f"{spec['id']}.csv",
                    "log_sha256": "a" * 64,
                    "origin": f"{spec['id']}.origin.json",
                    "origin_sha256": "b" * 64,
                }
                for spec in expected_production_roster()
            ],
            "scene_cycle": {
                "log": "scene-cycle.csv",
                "log_sha256": "c" * 64,
                "cleanup": "scene-cycle-cleanup.json",
                "cleanup_sha256": "d" * 64,
                "origin": "scene-cycle.origin.json",
                "origin_sha256": "e" * 64,
            },
        }
        cls.logs = {
            spec["id"]: production_route_rows(spec)
            for spec in expected_production_roster()
        }
        cls.scene_rows = scene_cycle_rows(
            list(LOCKED_SCENE_IDS), dwell=25, cycles=2)
        cls.cleanup = {
            "exit_code": 0, "live_model_count": 0, "log_closed": True,
            "model_load_count": 28, "model_unload_count": 28,
            "motion_pack_load_count": 1, "window_closed": True,
        }
        cls.baseline_path = pathlib.Path(__file__).parents[1] / \
            "fixtures" / "g1_motion_quality_baseline.json"
        cls.baseline_payload = cls.baseline_path.read_bytes()
        cls.baseline = json.loads(cls.baseline_payload)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def validator(path, *, full_source_validation):
        if not full_source_validation:
            raise AssertionError("quality checker skipped full-source validation")
        return {"frames": 3_970_932, "clips": 15_815,
                "bones": 31, "scenes": 14, "source_rows": 4_250}

    def evaluate(self, logs=None):
        return quality.evaluate_candidate(
            self.baseline, self.matrix, self.logs if logs is None else logs,
            self.scene_rows, self.cleanup,
            baseline_payload=self.baseline_payload,
            candidate_validator=self.validator)

    def test_full_production_matrix_passes_and_owns_exactly_six_comparisons(self):
        report = self.evaluate()
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(len(report["routes"]), 37)
        self.assertEqual(report["candidate"]["sources"], 15_815)
        self.assertEqual(report["candidate"]["frames"], 3_970_932)
        self.assertTrue(report["candidate"]["validator_passed"])
        self.assertEqual(
            report["evidence"]["matrix_sha256"],
            hashlib.sha256(quality.canonical_json(
                self.matrix).encode("utf-8")).hexdigest())
        self.assertFalse(report["evidence"]["files_verified"])
        self.assertIn("self-attested", report["evidence"][
            "origin_authentication"])
        self.assertEqual(report["baseline_sha256"],
                         quality.CANONICAL_BASELINE_SHA256)
        comparison_routes = set()
        for gate in report["gates"].values():
            comparison_routes.update(gate.get("routes", []))
            if "route" in gate:
                comparison_routes.add(gate["route"])
        self.assertEqual(comparison_routes, set(COMPARISON_IDS))
        self.assertTrue(all(report["coverage"]["derived"].values()))
        self.assertTrue(report["scene_cycle"]["passed"])

    def test_each_comparative_gate_fails_independently_at_its_boundary(self):
        thresholds = quality._thresholds(self.baseline)
        comparisons = {
            route: {
                "bank_correctness": {
                    "wrong_family_frames": 0,
                    "activation_frame": (
                        0 if route in thresholds[
                            "activated_terrain_wrong_family_frames"]["routes"]
                        else None),
                },
                "selected_cost": {"p95": 0.0},
                "transition_rate": 0.0,
                "stance_slip": {"total_m": {"left": 0.0, "right": 0.0}},
                "sole_clearance": {"worst_penetration_m": 0.0},
            }
            for route in COMPARISON_IDS
        }
        self.assertTrue(all(
            gate["passed"] for gate in
            quality._build_quality_gates(thresholds, comparisons).values()))

        mutations = {
            "flat_wrong_family_frames": lambda reports: reports[
                thresholds["flat_wrong_family_frames"]["routes"][0]][
                    "bank_correctness"].update(wrong_family_frames=1),
            "activated_terrain_wrong_family_frames": lambda reports: reports[
                thresholds[
                    "activated_terrain_wrong_family_frames"]["routes"][0]][
                        "bank_correctness"].update(wrong_family_frames=1),
            "lateral_selected_cost_p95": lambda reports: reports[
                thresholds["lateral_selected_cost_p95"]["route"]][
                    "selected_cost"].update(
                        p95=thresholds["lateral_selected_cost_p95"]["baseline"]),
            "lateral_transition_rate": lambda reports: reports[
                thresholds["lateral_transition_rate"]["route"]].update(
                    transition_rate=thresholds[
                        "lateral_transition_rate"]["baseline"]),
            "stance_slip_total_m": lambda reports: reports[
                COMPARISON_IDS[0]]["stance_slip"]["total_m"].update(
                    left=thresholds["stance_slip_total_m"]["baseline"]),
            "sole_worst_penetration_m": lambda reports: reports[
                COMPARISON_IDS[0]]["sole_clearance"].update(
                    worst_penetration_m=thresholds[
                        "sole_worst_penetration_m"]["baseline"]),
        }
        for expected_failure, mutate in mutations.items():
            with self.subTest(gate=expected_failure):
                changed = copy.deepcopy(comparisons)
                mutate(changed)
                gates = quality._build_quality_gates(thresholds, changed)
                self.assertFalse(gates[expected_failure]["passed"])
                self.assertTrue(all(
                    gate["passed"] for name, gate in gates.items()
                    if name != expected_failure))

        for changed in (
                {key: value for key, value in comparisons.items()
                 if key != COMPARISON_IDS[0]},
                {**comparisons, "invented/comparison": comparisons[
                    COMPARISON_IDS[0]]}):
            with self.subTest(comparisons=sorted(changed)):
                with self.assertRaisesRegex(ValueError, "exactly six routes"):
                    quality._build_quality_gates(thresholds, changed)

    def test_matrix_cannot_author_semantics_or_coverage(self):
        for field, value in (
                ("frames", 4), ("scene", "invented"),
                ("heading", "invented"), ("coverage", list(REQUIRED_TAGS)),
                ("comparison_id", "invented")):
            matrix = copy.deepcopy(self.matrix)
            matrix["routes"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "hash-bound|production"):
                    quality._route_specs(matrix)

    def test_missing_locked_route_and_scene_cycle_binding_are_rejected(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["routes"].pop()
        with self.assertRaisesRegex(ValueError, "production route roster"):
            quality._route_specs(matrix)
        matrix = copy.deepcopy(self.matrix)
        matrix["scene_cycle"]["frames"] = 700
        with self.assertRaisesRegex(ValueError, "scene-cycle evidence"):
            quality._route_specs(matrix)

    def test_cli_writes_canonical_report_and_uses_full_validator(self):
        (self.root / "matrix.json").write_text(
            json.dumps(self.matrix), encoding="utf-8")
        output = self.root / "quality-report.json"
        with mock.patch(
                "resources.validate_g1_terrain_database."
                "validate_artifact_directory",
                side_effect=self.validator) as validator, mock.patch.object(
                    quality, "_load_logs",
                    return_value=(
                        self.logs, self.scene_rows, self.cleanup,
                        {
                            "routes": {
                                route["id"]: {
                                    name: route[name] for name in (
                                        "log", "log_sha256", "origin",
                                        "origin_sha256")
                                }
                                for route in self.matrix["routes"]
                            },
                            "scene_cycle": dict(self.matrix["scene_cycle"]),
                            "files_verified": True,
                        })):
            status = quality.main([
                "--baseline", str(self.baseline_path),
                "--logs", str(self.root), "--output", str(output),
            ])
        self.assertEqual(status, 0)
        validator.assert_called_once_with(
            str(self.candidate.resolve()), full_source_validation=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["evidence"]["files_verified"])
        self.assertEqual(
            report["evidence"]["matrix_sha256"], hashlib.sha256(
                (self.root / "matrix.json").read_bytes()).hexdigest())
        self.assertEqual(output.read_text(encoding="utf-8"),
                         quality.canonical_json(report))


if __name__ == "__main__":
    unittest.main()
