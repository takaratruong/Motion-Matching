from dataclasses import dataclass
import hashlib
import json
import re
import struct
from typing import Callable

import numpy as np

from resources import quat as holden_quat

from .artifacts import walkability_bytes
from .terrain import (
    GrailTerrain,
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


COURSE_HALF_WIDTH = 0.60
FLAT_SPAWN_LENGTH = 2.0
LOOKAHEAD_MARGIN = 1.0
GRAIL_DEFAULT_BASE = "terrain_curbs__curb_000__000"
GRAIL_TARGETS = (
    ("grail-curb-low", 0.12),
    ("grail-curb-medium", 0.24),
    ("grail-curb-high", 0.36),
)


def _runtime_f32_upper_ceiling(value, label):
    target = _finite_real(value, label)
    result = _runtime_f32(target, label)
    if result < target:
        result = _runtime_f32(
            np.nextafter(np.float32(result), np.float32(np.inf)),
            f"{label} upper ceiling")
    return result


def _walkability_classification_bounds(bounds):
    # The JSON playable/region bounds remain center bounds. Only the promoted
    # source-node G1WM callback receives this exact binary32 footprint halo.
    values = tuple(
        _runtime_f32(value, f"walkability center bound {index}")
        for index, value in enumerate(bounds)
    )
    halo = _runtime_f32(
        WALKABILITY_CLASSIFICATION_HALO,
        "walkability classification halo")
    return (
        _f32_sub(values[0], halo, "walkability classification xmin"),
        _f32_add(values[1], halo, "walkability classification xmax"),
        _f32_sub(values[2], halo, "walkability classification zmin"),
        _f32_add(values[3], halo, "walkability classification zmax"),
    )


@dataclass(frozen=True)
class LongitudinalProfileSurface:
    profile: Callable[[float], float]
    half_width: float = COURSE_HALF_WIDTH
    exterior_height: float = 0.0

    def height(self, x, z):
        x, z = float(x), float(z)
        if not np.isfinite([x, z]).all():
            raise ValueError("profile query must be finite")
        if abs(x) > self.half_width:
            return self.exterior_height
        value = float(self.profile(z))
        if not np.isfinite(value):
            raise ValueError("profile height must be finite")
        return value


@dataclass(frozen=True)
class CrossSlopeSurface:
    angle_degrees: float
    grade_start_z: float
    grade_length: float = 4.0
    transition_length: float = 0.5
    half_width: float = COURSE_HALF_WIDTH

    def height(self, x, z):
        x, z = float(x), float(z)
        if not np.isfinite([x, z]).all():
            raise ValueError("cross-slope query must be finite")
        if abs(x) > self.half_width:
            return 0.0
        phase = z - self.grade_start_z
        if phase <= 0.0 or phase >= self.grade_length:
            return 0.0
        factor = min(
            1.0,
            phase / self.transition_length,
            (self.grade_length - phase) / self.transition_length,
        )
        return float(
            x * np.tan(np.deg2rad(self.angle_degrees)) * factor)


@dataclass(frozen=True)
class BlockedCourseSurface:
    wall_center_x: float = -0.8
    ramp_center_x: float = 0.8
    lane_half_width: float = 0.6
    obstacle_start_z: float = FLAT_SPAWN_LENGTH
    wall_height: float = 0.45
    wall_top_length: float = 0.50
    ramp_rise: float = 0.36
    ramp_angle_degrees: float = 25.0
    ramp_top_length: float = 0.50

    @property
    def ramp_run(self):
        return float(
            self.ramp_rise /
            np.tan(np.deg2rad(self.ramp_angle_degrees)))

    def height(self, x, z):
        x, z = float(x), float(z)
        if not np.isfinite([x, z]).all():
            raise ValueError("blocked-course query must be finite")
        wall_min_x = self.wall_center_x - self.lane_half_width
        wall_max_x = self.wall_center_x + self.lane_half_width
        ramp_min_x = self.ramp_center_x - self.lane_half_width
        ramp_max_x = self.ramp_center_x + self.lane_half_width
        wall_end_z = self.obstacle_start_z + self.wall_top_length
        ramp_ascent_end_z = self.obstacle_start_z + self.ramp_run
        ramp_end_z = ramp_ascent_end_z + self.ramp_top_length
        if wall_min_x <= x <= wall_max_x:
            if self.obstacle_start_z <= z \
                    <= wall_end_z:
                return self.wall_height
        if ramp_min_x <= x <= ramp_max_x:
            if self.obstacle_start_z <= z < ramp_ascent_end_z:
                phase = z - self.obstacle_start_z
                return float(
                    phase *
                    np.tan(np.deg2rad(self.ramp_angle_degrees)))
            if ramp_ascent_end_z <= z <= ramp_end_z:
                return self.ramp_rise
        return 0.0


def _region(region_id, bounds):
    return {"id": region_id, "bounds_xz": [float(v) for v in bounds]}


def _corridor_definition(
    scene_id, label, surface, course_end_z, parameters, route,
    walkability_class,
):
    heightfield_end_z = _runtime_f32_upper_ceiling(
        course_end_z + LOOKAHEAD_MARGIN, "heightfield zmax")
    playable = (-COURSE_HALF_WIDTH, COURSE_HALF_WIDTH, 0.0, course_end_z)
    classification = _walkability_classification_bounds(playable)
    bounds = (
        -COURSE_HALF_WIDTH - LOOKAHEAD_MARGIN,
        COURSE_HALF_WIDTH + LOOKAHEAD_MARGIN,
        -LOOKAHEAD_MARGIN,
        heightfield_end_z,
    )
    region_name = "certified" if walkability_class == 1 else "stress"
    regions = {"certified": (), "stress": (), "blocked": ()}
    regions[region_name] = (_region("course", playable),)
    return SceneDefinition(
        scene_id=scene_id,
        label=label,
        provenance={
            "kind": "procedural", "source_ids": [],
            "parameters": parameters,
        },
        surface=surface,
        heightfield_bounds_xz=bounds,
        playable_bounds_xz=playable,
        lookahead_bounds_xz=bounds,
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions=regions,
        routes=(route,),
        walkability=lambda x, z, c=walkability_class, b=classification: (
            c if _bounds_contains(b, x, z) else 0),
    )


def _stair_profile(rises, runs, landing_length=2.0):
    rises = tuple(rises)
    runs = tuple(runs)
    pairs = tuple(zip(rises, runs))
    ascent_ends = tuple(
        FLAT_SPAWN_LENGTH + sum(runs[:index + 1])
        for index in range(len(runs))
    )
    ascent_end = ascent_ends[-1]
    descent_start = ascent_end + landing_length
    reversed_pairs = tuple(reversed(pairs))
    reversed_runs = tuple(run for _, run in reversed_pairs)
    descent_ends = tuple(
        descent_start + sum(reversed_runs[:index + 1])
        for index in range(len(reversed_runs))
    )
    course_end = descent_ends[-1] + 1.0

    def profile(z):
        if z < FLAT_SPAWN_LENGTH:
            return 0.0
        height = 0.0
        for (rise, _), end in zip(pairs, ascent_ends):
            height += rise
            if z < end:
                return height
        if z < descent_start:
            return height
        for (rise, _), end in zip(reversed_pairs, descent_ends):
            if z < end:
                return height
            height -= rise
        return 0.0

    return profile, ascent_end, descent_start, course_end


def _stair_definition(scene_id, label, rises, runs, unseen):
    profile, ascent_end, descent_start, course_end = _stair_profile(rises, runs)
    course_end = _runtime_f32(course_end, f"{scene_id} course end z")
    parameters = {
        "primitive": "stairs-up-landing-down",
        "rises_m": list(rises),
        "runs_m": list(runs),
        "width_m": 1.2,
        "landing_length_m": 2.0,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_exit_length_m": 1.0,
        "unseen_geometry": unseen,
        "ascent_end_z_m": ascent_end,
        "descent_start_z_m": descent_start,
        "course_end_z_m": course_end,
    }
    route = SceneRoute(
        "ascent-landing-descent",
        ((0.0, 0.0), (0.0, 1.75),
         (0.0, ascent_end + 1.0),
         (0.0, descent_start + sum(runs) + 0.25),
         (0.0, course_end)),
        "traverse", 1, 2.0,
    )
    return _corridor_definition(
        scene_id, label, LongitudinalProfileSurface(profile),
        course_end, parameters, route, 1)


def _ramp_profile(angle_degrees):
    rise = 0.36
    run = float(rise / np.tan(np.deg2rad(angle_degrees)))
    ascent_end = FLAT_SPAWN_LENGTH + run
    descent_start = ascent_end + 2.0
    course_end = descent_start + run + 1.0

    def profile(z):
        if z < FLAT_SPAWN_LENGTH:
            return 0.0
        if z < ascent_end:
            return (z - FLAT_SPAWN_LENGTH) * rise / run
        if z < descent_start:
            return rise
        if z < descent_start + run:
            return rise - (z - descent_start) * rise / run
        return 0.0

    return profile, run, ascent_end, descent_start, course_end


def _ramp_definition(scene_id, label, angle_degrees, stress):
    profile, run, ascent_end, descent_start, course_end = _ramp_profile(
        angle_degrees)
    course_end = _runtime_f32(course_end, f"{scene_id} course end z")
    walkability_class = 2 if stress else 1
    expected_outcome = "traverse-or-safe-stop" if stress else "traverse"
    parameters = {
        "primitive": "ramp-up-landing-down",
        "angle_degrees": float(angle_degrees),
        "rise_m": 0.36,
        "run_m": run,
        "width_m": 1.2,
        "landing_length_m": 2.0,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_exit_length_m": 1.0,
        "ascent_end_z_m": ascent_end,
        "descent_start_z_m": descent_start,
        "course_end_z_m": course_end,
    }
    route = SceneRoute(
        "up-landing-down",
        ((0.0, 0.0), (0.0, 1.75),
         (0.0, ascent_end + 1.0),
         (0.0, descent_start + run + 0.25),
         (0.0, course_end)),
        expected_outcome, walkability_class, 2.0,
    )
    return _corridor_definition(
        scene_id, label, LongitudinalProfileSurface(profile),
        course_end, parameters, route, walkability_class)


def _cross_slope_definition(angle_degrees):
    flat_entry_length = 1.0
    grade_start = FLAT_SPAWN_LENGTH + flat_entry_length
    course_end = grade_start + 4.0 + 1.0
    scene_id = f"cross-slope-{angle_degrees:02d}"
    course_end = _runtime_f32(course_end, f"{scene_id} course end z")
    parameters = {
        "primitive": "cross-slope",
        "angle_degrees": float(angle_degrees),
        "width_m": 1.2,
        "grade_length_m": 4.0,
        "transition_length_m": 0.5,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_entry_length_m": flat_entry_length,
        "flat_exit_length_m": 1.0,
        "course_end_z_m": course_end,
    }
    route = SceneRoute(
        "forward-cross-slope",
        ((0.0, 0.0), (0.0, 2.75), (0.0, 5.0),
         (0.0, 7.25), (0.0, course_end)),
        "traverse", 1, 0.0,
    )
    return _corridor_definition(
        scene_id, f"Cross Slope {angle_degrees} Degrees",
        CrossSlopeSurface(float(angle_degrees), grade_start), course_end,
        parameters, route, 1)


def _mixed_definition():
    stair_runs = (0.30, 0.30, 0.30, 0.30)
    stair_rises = (0.08, 0.08, 0.08, 0.08)
    stair_ascent_ends = tuple(
        FLAT_SPAWN_LENGTH + sum(stair_runs[:index + 1])
        for index in range(len(stair_runs))
    )
    ascent_end = stair_ascent_ends[-1]
    elevated_end = ascent_end + 3.0
    changes = (0.08, -0.12, 0.04)
    block_starts = tuple(elevated_end + 0.60 * i for i in range(3))
    ramp_start = elevated_end + 3 * 0.60
    block_ends = block_starts[1:] + (ramp_start,)
    ramp_run = float(
        sum(stair_rises) / np.tan(np.deg2rad(10.0)))
    course_end = ramp_start + ramp_run + 1.0
    course_end = _runtime_f32(course_end, "mixed-multilevel course end z")

    def profile(z):
        if z < FLAT_SPAWN_LENGTH:
            return 0.0
        height = 0.0
        for rise, end in zip(stair_rises, stair_ascent_ends):
            height += rise
            if z < end:
                return height
        if z < elevated_end:
            return height
        for change, end in zip(changes, block_ends):
            height += change
            if z < end:
                return height
        if z < ramp_start + ramp_run:
            return height * (1.0 - (z - ramp_start) / ramp_run)
        return 0.0

    parameters = {
        "primitive": "mixed-multilevel",
        "stair_rises_m": list(stair_rises),
        "stair_runs_m": list(stair_runs),
        "width_m": 1.2,
        "elevated_walk_length_m": 3.0,
        "block_height_changes_m": list(changes),
        "block_top_length_m": 0.60,
        "block_starts_z_m": list(block_starts),
        "return_ramp_angle_degrees": 10.0,
        "return_ramp_run_m": ramp_run,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "flat_exit_length_m": 1.0,
        "course_end_z_m": course_end,
    }
    route_points = [
        (0.0, 0.0), (0.0, 1.75), (0.0, ascent_end + 1.5),
    ]
    route_points.extend((0.0, start + 0.30) for start in block_starts)
    route_points.extend(((0.0, ramp_start + ramp_run), (0.0, course_end)))
    return _corridor_definition(
        "mixed-multilevel", "Mixed Multilevel Course",
        LongitudinalProfileSurface(profile), course_end, parameters,
        SceneRoute(
            "full-course", tuple(route_points), "traverse", 1, 2.0),
        1)


def _blocked_definition():
    surface = BlockedCourseSurface()
    obstacle_start = surface.obstacle_start_z
    ramp_run = surface.ramp_run
    ramp_ascent_end = obstacle_start + ramp_run
    ramp_end = ramp_ascent_end + surface.ramp_top_length
    course_end = _runtime_f32(
        ramp_end + 1.0, "blocked-course course end z")
    heightfield_end_z = _runtime_f32_upper_ceiling(
        course_end + LOOKAHEAD_MARGIN,
        "blocked-course heightfield zmax")
    playable = (-1.4, 1.4, 0.0, course_end)
    classification = _walkability_classification_bounds(playable)
    bounds = (-2.4, 2.4, -1.0, heightfield_end_z)

    def walkability(x, z):
        if not _bounds_contains(classification, x, z):
            return 0
        # Expand only the outer course bounds. This obstacle threshold is the
        # published safety boundary and deliberately receives no halo.
        return 1 if z <= obstacle_start - SCENE_CELL_SIZE else 0

    parameters = {
        "primitive": "blocked-course",
        "lane_width_m": 1.2,
        "wall_center_x_m": surface.wall_center_x,
        "wall_height_m": surface.wall_height,
        "wall_top_length_m": surface.wall_top_length,
        "ramp_center_x_m": surface.ramp_center_x,
        "ramp_rise_m": surface.ramp_rise,
        "ramp_angle_degrees": surface.ramp_angle_degrees,
        "ramp_run_m": ramp_run,
        "ramp_top_length_m": surface.ramp_top_length,
        "flat_spawn_length_m": FLAT_SPAWN_LENGTH,
        "course_end_z_m": course_end,
    }
    return SceneDefinition(
        scene_id="blocked-course",
        label="Blocked Wall And Ramp Course",
        provenance={
            "kind": "procedural", "source_ids": [],
            "parameters": parameters,
        },
        surface=surface,
        heightfield_bounds_xz=bounds,
        playable_bounds_xz=playable,
        lookahead_bounds_xz=bounds,
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions={
            "certified": (_region(
                "approach", (-1.4, 1.4, 0.0,
                             obstacle_start - SCENE_CELL_SIZE)),),
            "stress": (),
            "blocked": (
                _region("wall", (-1.4, -0.2, obstacle_start, course_end)),
                _region("ramp", (0.2, 1.4, obstacle_start, course_end)),
            ),
        },
        routes=(
            SceneRoute(
                "wall-safe-stop",
                ((0.0, 0.0), (-0.8, 1.5), (-0.8, 2.25)),
                "safe-stop", 0, 0.0),
            SceneRoute(
                "ramp-safe-stop",
                ((0.0, 0.0), (0.8, 1.5),
                 (0.8, obstacle_start + ramp_run / 2.0)),
                "safe-stop", 0, 0.0),
        ),
        walkability=walkability,
    )


def procedural_scene_definitions():
    return (
        _stair_definition(
            "stairs-shallow", "Shallow Stairs",
            (0.08, 0.08, 0.08, 0.08),
            (0.30, 0.30, 0.30, 0.30), False),
        _stair_definition(
            "stairs-standard", "Standard Stairs",
            (0.12, 0.12, 0.12), (0.32, 0.32, 0.32), False),
        _stair_definition(
            "stairs-unseen-variable", "Unseen Variable Stairs",
            (0.06, 0.10, 0.08, 0.12),
            (0.24, 0.34, 0.28, 0.38), True),
        _ramp_definition(
            "ramp-05-up-down", "Ramp 5 Degrees Up And Down", 5, False),
        _ramp_definition(
            "ramp-10-up-down", "Ramp 10 Degrees Up And Down", 10, False),
        _ramp_definition(
            "ramp-15-stress", "Ramp 15 Degree Stress Case", 15, True),
        _cross_slope_definition(5),
        _cross_slope_definition(10),
        _mixed_definition(),
        _blocked_definition(),
    )


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


def select_grail_scene_bases(measured_max_heights):
    if not isinstance(measured_max_heights, dict) or not measured_max_heights:
        raise ValueError("GRAIL measurements must be a non-empty mapping")
    measured = {}
    for base, height in measured_max_heights.items():
        if type(base) is not str or not base:
            raise ValueError("GRAIL base names must be non-empty strings")
        measured[base] = _finite_real(
            height, f"GRAIL measured height for {base}")
    if GRAIL_DEFAULT_BASE not in measured:
        raise ValueError(f"missing default GRAIL base {GRAIL_DEFAULT_BASE}")
    result = {"grail-curb-default": GRAIL_DEFAULT_BASE}
    for scene_id, target in GRAIL_TARGETS:
        result[scene_id] = min(
            measured,
            key=lambda base: (abs(measured[base] - target), base),
        )
    return result


def _root_route_and_yaw(clip):
    positions = np.asarray(clip.positions, np.float64)
    rotations = np.asarray(clip.rotations, np.float64)
    if positions.ndim != 3 or positions.shape[0] < 2 \
            or positions.shape[1] < 1 or positions.shape[2] != 3:
        raise ValueError("converted GRAIL clip has invalid root positions")
    if rotations.ndim != 3 \
            or rotations.shape != positions.shape[:2] + (4,):
        raise ValueError("converted GRAIL clip has invalid root rotations")
    if not np.isfinite(positions).all() \
            or not np.isfinite(rotations).all():
        raise ValueError("converted GRAIL root transform is invalid")
    indices = sorted(set(
        int(round(value))
        for value in np.linspace(0, len(positions) - 1, 5)
    ))
    if len(indices) < 2:
        raise ValueError("converted GRAIL route needs two distinct frames")
    path = positions[:, 0][:, (0, 2)]
    route = tuple(
        (float(path[index, 0]), float(path[index, 1]))
        for index in indices
    )
    root_rotation = rotations[0, 0]
    root_norm = float(np.linalg.norm(root_rotation))
    if root_norm < 1e-8 or abs(root_norm - 1.0) > 1e-4:
        raise ValueError(
            "converted GRAIL root quaternion is not unit length")
    root_rotation = root_rotation / root_norm
    facing = holden_quat.mul_vec(
        root_rotation, np.array([0.0, 0.0, 1.0], np.float64))
    horizontal = np.array([facing[0], facing[2]], np.float64)
    if not np.isfinite(horizontal).all() or np.linalg.norm(horizontal) < 1e-8:
        raise ValueError("converted GRAIL root facing is invalid")
    yaw = float(np.arctan2(horizontal[0], horizontal[1]))
    spawn = tuple(float(value) for value in positions[0, 0])
    return path, route, spawn, yaw


def grail_scene_definition(
    scene_id, base, clip, target_height_m,
):
    if scene_id not in REQUIRED_SCENE_IDS[:4]:
        raise ValueError(f"unknown GRAIL scene ID {scene_id}")
    if clip.terrain_id != base:
        raise ValueError("GRAIL scene requires its matching converted clip")
    terrain = GrailTerrain.from_base(base)
    maximum_height = float(terrain.footprint()["height"])
    path, route_points, spawn, yaw = _root_route_and_yaw(clip)
    mesh_xmin, mesh_xmax, mesh_zmin, mesh_zmax = terrain.xz_bounds()
    path_xmin, path_zmin = path.min(axis=0)
    path_xmax, path_zmax = path.max(axis=0)
    playable = (
        float(path_xmin - COURSE_HALF_WIDTH),
        float(path_xmax + COURSE_HALF_WIDTH),
        float(path_zmin - COURSE_HALF_WIDTH),
        float(path_zmax + COURSE_HALF_WIDTH),
    )
    classification = _walkability_classification_bounds(playable)
    bounds = (
        float(min(mesh_xmin, playable[0]) - LOOKAHEAD_MARGIN),
        float(max(mesh_xmax, playable[1]) + LOOKAHEAD_MARGIN),
        float(min(mesh_zmin, playable[2]) - LOOKAHEAD_MARGIN),
        float(max(mesh_zmax, playable[3]) + LOOKAHEAD_MARGIN),
    )
    certified = maximum_height <= 0.16
    walkability_class = 1 if certified else 2
    expected_outcome = "traverse" if certified else "traverse-or-safe-stop"
    region_name = "certified" if certified else "stress"
    regions = {"certified": (), "stress": (), "blocked": ()}
    regions[region_name] = (_region("curb-route", playable),)
    labels = {
        "grail-curb-default": "GRAIL Default Curb",
        "grail-curb-low": "GRAIL Low Curb",
        "grail-curb-medium": "GRAIL Medium Curb",
        "grail-curb-high": "GRAIL High Curb",
    }
    return SceneDefinition(
        scene_id=scene_id,
        label=labels[scene_id],
        provenance={
            "kind": "grail",
            "source_ids": [base, clip.name],
            "parameters": {
                "selection_rule": (
                    "fixed-default" if target_height_m is None
                    else "nearest-measured-maximum-then-lexical"),
                "target_height_m": (
                    None if target_height_m is None
                    else float(target_height_m)),
                "measured_maximum_height_m": maximum_height,
                "route_source": "converted-holden-root-path",
            },
        },
        surface=terrain,
        heightfield_bounds_xz=bounds,
        playable_bounds_xz=playable,
        lookahead_bounds_xz=bounds,
        spawn_position=spawn,
        spawn_yaw_radians=yaw,
        regions=regions,
        routes=(SceneRoute(
            "curb-forward", route_points, expected_outcome,
            walkability_class, 0.0),),
        walkability=lambda x, z, c=walkability_class, b=classification: (
            c if _bounds_contains(b, x, z) else 0),
    )


def grail_scene_definitions(measured_max_heights, clips_by_terrain):
    selected = select_grail_scene_bases(measured_max_heights)
    targets = {scene_id: target for scene_id, target in GRAIL_TARGETS}
    targets["grail-curb-default"] = None
    definitions = []
    for scene_id in REQUIRED_SCENE_IDS[:4]:
        base = selected[scene_id]
        if base not in clips_by_terrain:
            raise ValueError(f"missing converted scene clip for {base}")
        definitions.append(grail_scene_definition(
            scene_id, base, clips_by_terrain[base], targets[scene_id]))
    return tuple(definitions)


def all_scene_definitions(measured_max_heights, clips_by_terrain):
    definitions = grail_scene_definitions(
        measured_max_heights, clips_by_terrain) \
        + procedural_scene_definitions()
    if tuple(scene.scene_id for scene in definitions) != REQUIRED_SCENE_IDS:
        raise ValueError("complete scene definition order changed")
    return definitions


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
