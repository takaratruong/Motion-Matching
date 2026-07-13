from dataclasses import dataclass
import hashlib
import json
import re
import struct
from typing import Callable

import numpy as np

from .artifacts import walkability_bytes
from .terrain import (
    HEIGHTFIELD_DIAGONAL,
    HEIGHTFIELD_INTERPOLATION,
    HeightGrid,
    rasterize_heightfield,
    surface_semantics_signature,
)


REQUIRED_SCENE_IDS = (
    "grail-curb-default",
    "grail-curb-low",
    "grail-curb-medium",
    "grail-curb-high",
    "stairs-shallow",
    "stairs-standard",
    "stairs-unseen-variable",
    "ramp-05-up-down",
    "ramp-10-up-down",
    "ramp-15-stress",
    "cross-slope-05",
    "cross-slope-10",
    "mixed-multilevel",
    "blocked-course",
)
COORDINATE_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
TERRAIN_DISTANCES = (0.25, 0.50, 0.75, 1.00)
SCENE_CELL_SIZE = 0.02
WALKABILITY_CLASSIFICATION_HALO = 0.25
SCENE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
OUTCOME_CLASS = {
    "traverse": 1,
    "safe-stop": 0,
    "traverse-or-safe-stop": 2,
}


def _deep_normalize_json(value, label="JSON"):
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} object keys must be strings")
        return {
            key: _deep_normalize_json(child, f"{label}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _deep_normalize_json(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return 0.0 if value == 0.0 else value
    if type(value) is int or type(value) in (str, bool) or value is None:
        return value
    raise TypeError(f"{label} contains non-JSON scalar {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    normalized = _deep_normalize_json(value)
    return (json.dumps(
        normalized, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_real(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a real scalar, not bool")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _runtime_f32(value, label):
    source = _finite_real(value, label)
    try:
        bits = struct.unpack("<I", struct.pack("<f", source))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{label} overflows float32") from error
    exponent = bits & 0x7f800000
    magnitude = bits & 0x7fffffff
    if exponent == 0x7f800000:
        raise ValueError(f"{label} overflows float32")
    if magnitude != 0 and exponent == 0:
        raise ValueError(f"{label} becomes a float32 subnormal")
    decoded = struct.unpack("<f", struct.pack("<I", bits))[0]
    return 0.0 if decoded == 0.0 else float(decoded)


def _f32_add(left, right, label):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) + np.float32(right))
    return _runtime_f32(value, label)


def _f32_sub(left, right, label):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) - np.float32(right))
    return _runtime_f32(value, label)


def _f32_mul(left, right, label):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) * np.float32(right))
    return _runtime_f32(value, label)


def _f32_div(left, right, label):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) / np.float32(right))
    return _runtime_f32(value, label)


def _f32_sqrt(value, label):
    with np.errstate(invalid="ignore"):
        result = np.float32(np.sqrt(np.float32(value)))
    return _runtime_f32(result, label)


@dataclass(frozen=True)
class SceneRoute:
    route_id: str
    waypoints_xz: tuple[tuple[float, float], ...]
    expected_outcome: str
    walkability_class: int
    landing_hold_seconds: float

    def normalized_points(self):
        if not isinstance(self.waypoints_xz, (tuple, list)) \
                or len(self.waypoints_xz) < 2:
            raise ValueError("scene route needs an ID and at least two waypoints")
        points = []
        for index, point in enumerate(self.waypoints_xz):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError("scene route waypoints must be finite XZ pairs")
            points.append((
                _runtime_f32(point[0], f"route waypoint {index} x"),
                _runtime_f32(point[1], f"route waypoint {index} z"),
            ))
        for start, stop in zip(points, points[1:]):
            if start == stop:
                raise ValueError("scene route requires every nonzero segment")
        return tuple(points)

    def validate(self):
        if type(self.route_id) is not str \
                or not SCENE_ID_PATTERN.fullmatch(self.route_id):
            raise ValueError("scene route needs a safe non-empty ID")
        points = self.normalized_points()
        if type(self.expected_outcome) is not str:
            raise TypeError("scene route outcome must be a string")
        expected = OUTCOME_CLASS.get(self.expected_outcome)
        if type(self.walkability_class) is not int:
            raise TypeError("walkability class must be an integer, not bool")
        if expected != self.walkability_class:
            raise ValueError("scene route outcome and walkability class disagree")
        hold = _runtime_f32(
            self.landing_hold_seconds, "landing hold seconds")
        if hold < 0.0:
            raise ValueError("landing hold seconds must be finite and nonnegative")
        if hold > 0.0 and len(points) < 4:
            raise ValueError(
                "landing hold requires published waypoint 2 and a later exit")
        return points, hold

    def to_json(self):
        points, hold = self.validate()
        return {
            "id": self.route_id,
            "waypoints_xz": [[x, z] for x, z in points],
            "expected_outcome": self.expected_outcome,
            "walkability_class": self.walkability_class,
            "landing_hold_seconds": hold,
        }


@dataclass
class SceneDefinition:
    scene_id: str
    label: str
    provenance: dict
    surface: object
    heightfield_bounds_xz: tuple[float, float, float, float]
    playable_bounds_xz: tuple[float, float, float, float]
    lookahead_bounds_xz: tuple[float, float, float, float]
    spawn_position: tuple[float, float, float]
    spawn_yaw_radians: float
    regions: dict
    routes: tuple[SceneRoute, ...]
    walkability: Callable[[float, float], int]


@dataclass(frozen=True)
class BuiltScene:
    scene_id: str
    scene_json: bytes
    terrain_bin: bytes
    terrain_obj: bytes
    walkability_bin: bytes

    @property
    def metadata(self):
        return json.loads(self.scene_json)


@dataclass(frozen=True)
class ScenePack:
    index_json: bytes
    scenes: tuple[BuiltScene, ...]

    @property
    def index(self):
        return json.loads(self.index_json)


def _bounds_contains(bounds, x, z):
    xmin, xmax, zmin, zmax = bounds
    return xmin <= x <= xmax and zmin <= z <= zmax


def _strict_bounds(values, label):
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError(f"{label} bounds must contain xmin,xmax,zmin,zmax")
    bounds = tuple(
        _runtime_f32(value, f"{label} bounds[{index}]")
        for index, value in enumerate(values)
    )
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError(f"{label} bounds must be strict and nondegenerate")
    return bounds


def _bounds_contains_bounds(outer, inner):
    return outer[0] <= inner[0] <= inner[1] <= outer[1] \
        and outer[2] <= inner[2] <= inner[3] <= outer[3]


def _classify_grid(scene_id, walkability, grid):
    values = np.empty((grid.nz, grid.nx), np.uint8)
    for iz in range(grid.nz):
        z = grid.origin_z + iz * grid.cell_size
        for ix in range(grid.nx):
            x = grid.origin_x + ix * grid.cell_size
            value = walkability(x, z)
            if isinstance(value, (bool, np.bool_)) \
                    or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"{scene_id}: walkability must return an "
                    "integer class, not bool")
            if int(value) not in (0, 1, 2):
                raise ValueError(
                    f"{scene_id}: invalid walkability class {value}")
            values[iz, ix] = int(value)
    return values


def _walkability_indices(grid, x, z):
    x = _runtime_f32(x, "route sample x")
    z = _runtime_f32(z, "route sample z")
    gx = _f32_div(
        _f32_sub(x, grid.origin_x, "walkability gx numerator"),
        grid.cell_size, "walkability gx")
    gz = _f32_div(
        _f32_sub(z, grid.origin_z, "walkability gz numerator"),
        grid.cell_size, "walkability gz")
    if not (0.0 <= gx <= grid.nx - 1 and 0.0 <= gz <= grid.nz - 1):
        raise ValueError("route sample is outside walkability grid")
    ix = min(int(np.floor(np.float32(
        _f32_add(gx, 0.5, "walkability gx tie offset")))), grid.nx - 1)
    iz = min(int(np.floor(np.float32(
        _f32_add(gz, 0.5, "walkability gz tie offset")))), grid.nz - 1)
    return ix, iz


def _walkability_at(grid, values, x, z):
    ix, iz = _walkability_indices(grid, x, z)
    return int(values[iz, ix])


def _route_samples(points, maximum_step):
    output = [points[0]]
    step = _runtime_f32(maximum_step, "route maximum sample step")
    if step <= 0.0:
        raise ValueError("route maximum sample step must be positive")
    for start, stop in zip(points, points[1:]):
        dx = _f32_sub(stop[0], start[0], "route segment dx")
        dz = _f32_sub(stop[1], start[1], "route segment dz")
        squared = _f32_add(
            _f32_mul(dx, dx, "route segment dx squared"),
            _f32_mul(dz, dz, "route segment dz squared"),
            "route segment squared length")
        distance = _f32_sqrt(squared, "route segment length")
        ratio = _f32_div(distance, step, "route segment sample ratio")
        count = max(1, int(np.ceil(np.float32(ratio))))
        for index in range(1, count + 1):
            if index == count:
                output.append(stop)
                continue
            alpha = _f32_div(index, count, "route sample alpha")
            output.append((
                _f32_add(
                    start[0], _f32_mul(alpha, dx, "route sample dx"),
                    "route sample x"),
                _f32_add(
                    start[1], _f32_mul(alpha, dz, "route sample dz"),
                    "route sample z"),
            ))
    return tuple(output)


def _segment_cell_supercover(grid, start, stop):
    start_ix, start_iz = _walkability_indices(grid, *start)
    stop_ix, stop_iz = _walkability_indices(grid, *stop)
    if abs(stop_ix - start_ix) > 1 or abs(stop_iz - start_iz) > 1:
        raise ValueError("route half-cell subsegment skipped a grid cell")
    return tuple(
        (iz, ix)
        for iz in range(min(start_iz, stop_iz), max(start_iz, stop_iz) + 1)
        for ix in range(min(start_ix, stop_ix), max(start_ix, stop_ix) + 1)
    )


def _route_cell_covers(grid, points):
    samples = _route_samples(
        points, _f32_div(
            grid.cell_size, 2.0, "route half-cell sample step"))
    return tuple(
        _segment_cell_supercover(grid, start, stop)
        for start, stop in zip(samples, samples[1:])
    )


def _region_cell_indices(grid, bounds):
    xmin, xmax, zmin, zmax = bounds
    minimum_ix, minimum_iz = _walkability_indices(grid, xmin, zmin)
    maximum_ix, maximum_iz = _walkability_indices(grid, xmax, zmax)
    if minimum_ix > maximum_ix or minimum_iz > maximum_iz:
        raise ValueError("region nearest-node mapping is not monotone")
    return tuple(
        (iz, ix)
        for iz in range(minimum_iz, maximum_iz + 1)
        for ix in range(minimum_ix, maximum_ix + 1)
    )


def _validate_route_classes(routes, grid, classes, lookahead):
    for route in routes:
        points = tuple(tuple(point) for point in route["waypoints_xz"])
        for x, z in points:
            if not _bounds_contains(lookahead, x, z):
                raise ValueError("scene route leaves lookahead bounds")
        cover_classes = tuple(
            frozenset(int(classes[iz, ix]) for iz, ix in cover)
            for cover in _route_cell_covers(grid, points)
        )
        expected = route["walkability_class"]
        if expected in (1, 2):
            if any(values != {expected} for values in cover_classes):
                raise ValueError("expected route enters wrong walkability class")
        else:
            boundary = [
                index for index, values in enumerate(cover_classes)
                if values == {0, 1}
            ]
            if len(boundary) != 1:
                raise ValueError(
                    "safe-stop route needs exactly one certified-blocked cover")
            split = boundary[0]
            if split == 0 or split == len(cover_classes) - 1 \
                    or any(values != {1}
                           for values in cover_classes[:split]) \
                    or any(values != {0}
                           for values in cover_classes[split + 1:]):
                raise ValueError(
                    "safe-stop route must have pure certified and blocked sides")


def _validate_region_classes(regions, grid, classes):
    expected_classes = {"blocked": 0, "certified": 1, "stress": 2}
    for class_name, entries in regions.items():
        for entry in entries:
            observed = tuple(
                int(classes[iz, ix])
                for iz, ix in _region_cell_indices(
                    grid, entry["bounds_xz"])
            )
            if any(value != expected_classes[class_name]
                   for value in observed):
                raise ValueError(
                    f"{class_name} region disagrees with G1WM classes")


def _validate_definition(definition):
    scene_id = definition.scene_id
    label = definition.label
    surface = definition.surface
    walkability = definition.walkability
    if type(scene_id) is not str \
            or not SCENE_ID_PATTERN.fullmatch(scene_id):
        raise ValueError("scene ID must be a safe non-empty ID")
    if type(label) is not str or not label:
        raise ValueError("scene label must be a non-empty string")
    if type(definition.provenance) is not dict \
            or set(definition.provenance) != {
                "kind", "source_ids", "parameters"}:
        raise ValueError("scene provenance fields are not locked")
    if definition.provenance["kind"] not in ("grail", "procedural"):
        raise ValueError("scene provenance kind must be grail or procedural")
    source_ids = definition.provenance["source_ids"]
    parameters = definition.provenance["parameters"]
    if not isinstance(source_ids, (tuple, list)) \
            or any(type(value) is not str or not value for value in source_ids) \
            or type(parameters) is not dict:
        raise ValueError("scene provenance values are invalid")
    if (definition.provenance["kind"] == "procedural" and source_ids) \
            or (definition.provenance["kind"] == "grail"
                and len(source_ids) != 2):
        raise ValueError("scene provenance source IDs do not match kind")
    provenance = json.loads(canonical_json_bytes(definition.provenance))

    heightfield = _strict_bounds(
        definition.heightfield_bounds_xz, "heightfield")
    playable = _strict_bounds(definition.playable_bounds_xz, "playable")
    lookahead = _strict_bounds(definition.lookahead_bounds_xz, "lookahead")
    if not _bounds_contains_bounds(heightfield, lookahead):
        raise ValueError("scene lookahead bounds must lie inside heightfield")
    if not _bounds_contains_bounds(lookahead, playable):
        raise ValueError("scene playable bounds must lie inside lookahead")
    if not isinstance(definition.spawn_position, (tuple, list)) \
            or len(definition.spawn_position) != 3:
        raise ValueError("scene spawn position must contain XYZ")
    spawn = tuple(
        _runtime_f32(value, f"spawn position[{index}]")
        for index, value in enumerate(definition.spawn_position)
    )
    yaw = _runtime_f32(definition.spawn_yaw_radians, "spawn yaw")
    sx, _, sz = spawn
    if not _bounds_contains(playable, sx, sz):
        raise ValueError("scene spawn must lie inside playable bounds")
    if type(definition.regions) is not dict \
            or set(definition.regions) != {"certified", "stress", "blocked"}:
        raise ValueError("scene regions must define certified, stress, blocked")
    region_ids = set()
    regions = {"certified": [], "stress": [], "blocked": []}
    for class_name in ("certified", "stress", "blocked"):
        entries = definition.regions[class_name]
        if not isinstance(entries, (tuple, list)):
            raise TypeError(f"{class_name} regions must be a sequence")
        for entry in entries:
            if type(entry) is not dict or set(entry) != {"id", "bounds_xz"} \
                    or type(entry["id"]) is not str \
                    or not SCENE_ID_PATTERN.fullmatch(entry["id"]):
                raise ValueError(f"invalid {class_name} region")
            if entry["id"] in region_ids:
                raise ValueError(f"duplicate region ID {entry['id']}")
            region_ids.add(entry["id"])
            bounds = _strict_bounds(
                entry["bounds_xz"], f"{class_name} region {entry['id']}")
            if not _bounds_contains_bounds(lookahead, bounds):
                raise ValueError(f"{class_name} region leaves lookahead bounds")
            regions[class_name].append({
                "id": entry["id"], "bounds_xz": list(bounds),
            })

    if not isinstance(definition.routes, (tuple, list)) \
            or not definition.routes:
        raise ValueError("scene must define at least one route")
    route_ids = set()
    routes = []
    for route in definition.routes:
        if not isinstance(route, SceneRoute):
            raise TypeError("scene routes must be SceneRoute values")
        route_json = route.to_json()
        if route_json["id"] in route_ids:
            raise ValueError(f"duplicate route ID {route_json['id']}")
        route_ids.add(route_json["id"])
        if tuple(route_json["waypoints_xz"][0]) != (spawn[0], spawn[2]):
            raise ValueError("scene route must start at scene spawn")
        for x, z in route_json["waypoints_xz"]:
            if not _bounds_contains(lookahead, x, z):
                raise ValueError("scene route leaves lookahead bounds")
        routes.append(route_json)
    return {
        "scene_id": scene_id,
        "label": label,
        "surface": surface,
        "walkability": walkability,
        "provenance": provenance,
        "heightfield_bounds": heightfield,
        "playable_bounds": playable,
        "lookahead_bounds": lookahead,
        "spawn": spawn,
        "yaw": yaw,
        "regions": regions,
        "routes": routes,
    }


def _build_scene(normalized):
    grid = rasterize_heightfield(
        normalized["surface"],
        normalized["heightfield_bounds"],
        SCENE_CELL_SIZE,
    )
    classes = _classify_grid(
        normalized["scene_id"], normalized["walkability"], grid)
    terrain_bin = grid.g1hf_bytes()
    terrain_obj = grid.obj_bytes()
    walkability_bin = walkability_bytes(classes)
    _, _, nx, nz, origin_x, origin_z, cell_size, exterior = \
        struct.unpack_from("<4sIII4f", terrain_bin)
    decoded_heights = np.frombuffer(terrain_bin, "<f4", nx * nz, 32)
    minimum_y = float(decoded_heights.min())
    maximum_y = float(decoded_heights.max())
    heightfield_xmin = float(origin_x)
    heightfield_xmax = \
        heightfield_xmin + (nx - 1) * float(cell_size)
    heightfield_zmin = float(origin_z)
    heightfield_zmax = \
        heightfield_zmin + (nz - 1) * float(cell_size)

    def obj_coordinate(value):
        return _runtime_f32(value, "OBJ coordinate")

    mesh_xmin = obj_coordinate(heightfield_xmin)
    mesh_xmax = obj_coordinate(heightfield_xmax)
    mesh_zmin = obj_coordinate(heightfield_zmin)
    mesh_zmax = obj_coordinate(heightfield_zmax)
    metadata = {
        "schema": "g1-terrain-scene/v1",
        "id": normalized["scene_id"],
        "label": normalized["label"],
        "provenance": normalized["provenance"],
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
        "terrain_feature_distances_m": list(TERRAIN_DISTANCES),
        "heightfield": {
            "path": "terrain.bin", "schema": "G1HF/v2", "version": 2,
            "nx": nx, "nz": nz,
            "origin_x": float(origin_x), "origin_z": float(origin_z),
            "cell_size_m": float(cell_size),
            "exterior_height_m": float(exterior),
            "interpolation": HEIGHTFIELD_INTERPOLATION,
            "diagonal": HEIGHTFIELD_DIAGONAL,
            "sha256": sha256_hex(terrain_bin),
        },
        "mesh": {
            "path": "terrain.obj", "schema": "obj/v1",
            "sha256": sha256_hex(terrain_obj),
        },
        "walkability": {
            "path": "walkability.bin", "schema": "G1WM/v1", "version": 1,
            "nx": nx, "nz": nz,
            "classes": {"blocked": 0, "certified": 1, "stress": 2},
            "sha256": sha256_hex(walkability_bin),
        },
        "bounds": {
            "mesh_min_xyz": [mesh_xmin, minimum_y, mesh_zmin],
            "mesh_max_xyz": [mesh_xmax, maximum_y, mesh_zmax],
            "heightfield_min_xyz": [
                heightfield_xmin, minimum_y, heightfield_zmin],
            "heightfield_max_xyz": [
                heightfield_xmax, maximum_y, heightfield_zmax],
            "playable_min_xz": [
                normalized["playable_bounds"][0],
                normalized["playable_bounds"][2]],
            "playable_max_xz": [
                normalized["playable_bounds"][1],
                normalized["playable_bounds"][3]],
            "lookahead_min_xz": [
                normalized["lookahead_bounds"][0],
                normalized["lookahead_bounds"][2]],
            "lookahead_max_xz": [
                normalized["lookahead_bounds"][1],
                normalized["lookahead_bounds"][3]],
        },
        "spawn": {
            "position": list(normalized["spawn"]),
            "yaw_radians": normalized["yaw"],
        },
        "regions": normalized["regions"],
        "routes": normalized["routes"],
    }
    _validate_region_classes(normalized["regions"], grid, classes)
    _validate_route_classes(
        normalized["routes"], grid, classes, normalized["lookahead_bounds"])
    scene_json = canonical_json_bytes(metadata)
    # One parse/write round trip proves the cached bytes contain a deep,
    # JSON-native normalization and are the only publication authority.
    scene_json = canonical_json_bytes(json.loads(scene_json))
    return BuiltScene(
        normalized["scene_id"], scene_json,
        terrain_bin, terrain_obj, walkability_bin)


def build_scene(definition):
    return _build_scene(_validate_definition(definition))


def build_scene_pack(definitions):
    definitions = tuple(definitions)  # materialize a generator exactly once
    ids = tuple(definition.scene_id for definition in definitions)
    if ids != REQUIRED_SCENE_IDS:
        raise ValueError(
            f"required scene order is {REQUIRED_SCENE_IDS}, got {ids}")
    normalized = tuple(
        _validate_definition(definition) for definition in definitions)
    scenes = tuple(_build_scene(value) for value in normalized)
    built_ids = tuple(scene.scene_id for scene in scenes)
    if built_ids != REQUIRED_SCENE_IDS:
        raise ValueError(
            f"required built scene order is {REQUIRED_SCENE_IDS}, "
            f"got {built_ids}")
    index = {
        "schema": "g1-terrain-scene-index/v1",
        "default_scene_id": "grail-curb-default",
        "scene_ids": list(REQUIRED_SCENE_IDS),
        "scenes": [{
            "id": scene.scene_id,
            "path": f"scenes/{scene.scene_id}/scene.json",
            "sha256": sha256_hex(scene.scene_json),
        } for scene in scenes],
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
    }
    index_json = canonical_json_bytes(index)
    index_json = canonical_json_bytes(json.loads(index_json))
    return ScenePack(index_json, scenes)
