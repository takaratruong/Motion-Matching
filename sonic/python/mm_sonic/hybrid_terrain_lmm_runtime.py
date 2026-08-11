"""Exact-search runtime for the supported-terrain hybrid G1 LMM proof of concept."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from resources import quat

from .preliminary_learned_slope import decoded_pose_to_native_qpos

FPS = 25.0
DT = 1.0 / FPS
SEARCH_INTERVAL_S = 0.1
MAX_SPEED_MPS = 0.45
MAX_NATIVE_LIMIT_CANDIDATE_REJECTIONS = 32
# Broad native-walking mechanical plausibility guards for the interactive
# diagnostic.  They are not learned-pose quality scores or formal acceptance
# thresholds.
DIAGNOSTIC_MIN_PELVIS_SUPPORT_CLEARANCE_M = 0.4
DIAGNOSTIC_MAX_PELVIS_SUPPORT_CLEARANCE_M = 1.2
DIAGNOSTIC_MAX_SUCCESSOR_CLEARANCE_STEP_M = 0.1
TERRAIN_INDICES = slice(27, 31)
TRAJECTORY_INDICES = slice(15, 27)


def _finite_scalar(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _frozen(values: object, dtype: np.dtype | str) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SE2Transform:
    """Immutable planar translation and yaw in the native world frame."""

    xy: np.ndarray
    yaw: float

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("SE(2) translation must be one finite XY point")
        yaw = _finite_scalar(self.yaw, "SE(2) yaw")
        object.__setattr__(self, "xy", _frozen(xy, np.float64))
        object.__setattr__(self, "yaw", math.remainder(yaw, 2.0 * math.pi))


def compose_root_delta(
    world: SE2Transform, local_delta_xy: np.ndarray, local_delta_yaw: float
) -> SE2Transform:
    """Compose one native-planar local motion delta into a world transform."""

    if not isinstance(world, SE2Transform):
        raise TypeError("root composition requires an SE2Transform")
    delta = np.asarray(local_delta_xy, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("local root delta must be one finite XY vector")
    yaw_delta = _finite_scalar(local_delta_yaw, "local root yaw delta")
    cosine, sine = math.cos(world.yaw), math.sin(world.yaw)
    rotated = np.asarray(
        (
            cosine * delta[0] - sine * delta[1],
            sine * delta[0] + cosine * delta[1],
        ),
        dtype=np.float64,
    )
    return SE2Transform(world.xy + rotated, world.yaw + yaw_delta)


@dataclass(frozen=True)
class CommandState:
    """Normalized forward-speed and steering request in ``[-1, 1]``."""

    speed: float = 0.0
    steering: float = 0.0

    def __post_init__(self) -> None:
        speed = _finite_scalar(self.speed, "command speed")
        steering = _finite_scalar(self.steering, "command steering")
        object.__setattr__(self, "speed", float(np.clip(speed, -1.0, 1.0)))
        object.__setattr__(self, "steering", float(np.clip(steering, -1.0, 1.0)))

    @classmethod
    def from_keyboard(cls, pressed: Iterable[str]) -> CommandState:
        keys = {str(value).lower() for value in pressed}
        if " " in keys or "space" in keys:
            return cls()
        return cls(
            speed=float("w" in keys) - float("s" in keys),
            steering=float("a" in keys) - float("d" in keys),
        )

    @classmethod
    def from_gamepad(cls, *, speed_axis: float, steering_axis: float) -> CommandState:
        # Linux left-stick Y is negative in the forward direction.
        return cls(
            speed=-_finite_scalar(speed_axis, "gamepad speed axis"),
            steering=_finite_scalar(steering_axis, "gamepad steering axis"),
        )


class TerrainAuthority:
    """One height authority shared by feature queries and rendered geometry."""

    def __init__(
        self,
        height_at: Callable[[np.ndarray], float],
        *,
        domain_contains: Callable[[np.ndarray], bool] | None = None,
        name: str = "terrain",
    ) -> None:
        if not callable(height_at):
            raise TypeError("terrain height_at must be callable")
        if domain_contains is not None and not callable(domain_contains):
            raise TypeError("terrain domain predicate must be callable")
        if not str(name):
            raise ValueError("terrain name must be non-empty")
        self._height_at = height_at
        self._domain_contains = domain_contains
        self.name = str(name)

    @classmethod
    def flat(cls, height: float = 0.0) -> TerrainAuthority:
        value = _finite_scalar(height, "flat terrain height")
        return cls(lambda _xy, h=value: h, name="flat")

    @classmethod
    def multi_hill(cls) -> TerrainAuthority:
        """Small analytic scene suitable for deterministic overnight smoke runs."""

        def height_at(xy: np.ndarray) -> float:
            x, y = np.asarray(xy, dtype=np.float64)
            first = 0.32 * math.exp(-((x / 1.1) ** 2 + ((y + 1.8) / 0.9) ** 2))
            second = 0.22 * math.exp(-(((x + 0.8) / 0.8) ** 2 + ((y + 4.0) / 1.2) ** 2))
            trough = 0.06 * math.exp(-(((x - 0.7) / 0.7) ** 2 + ((y + 3.1) / 0.7) ** 2))
            return first + second - trough

        return cls(height_at, name="analytic-multi-hill")

    def height_at(self, world_xy: object) -> float:
        point = np.asarray(world_xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("terrain query must be one finite world XY point")
        value = _finite_scalar(self._height_at(point), "terrain height")
        return value

    def contains(self, world_xy: object) -> bool:
        point = np.asarray(world_xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("terrain domain query must be one finite world XY point")
        if self._domain_contains is None:
            return True
        result = self._domain_contains(point)
        if not isinstance(result, (bool, np.bool_)):
            raise TypeError("terrain domain predicate must return one boolean")
        return bool(result)

    def sample(
        self,
        root_xy: object,
        heading: float,
        distances: tuple[float, float, float, float] = (0.25, 0.5, 0.75, 1.0),
    ) -> np.ndarray:
        root = np.asarray(root_xy, dtype=np.float64)
        yaw = _finite_scalar(heading, "terrain sample heading")
        if root.shape != (2,) or not np.isfinite(root).all():
            raise ValueError("terrain sample root must be finite world XY")
        forward = np.asarray((math.sin(yaw), -math.cos(yaw)), dtype=np.float64)
        result = np.asarray(
            [
                self.height_at(root + float(distance) * forward)
                for distance in distances
            ],
            dtype=np.float64,
        )
        return result - self.height_at(root)

    def sample_points(self, root_xy: object, world_xy: object) -> np.ndarray:
        root = np.asarray(root_xy, dtype=np.float64)
        points = np.asarray(world_xy, dtype=np.float64)
        if root.shape != (2,) or not np.isfinite(root).all():
            raise ValueError("terrain sample root must be finite world XY")
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("terrain sample points must be four finite world XY rows")
        support = self.height_at(root)
        return np.asarray(
            [self.height_at(point) - support for point in points], dtype=np.float64
        )

    def support(self, root_xy: object) -> float:
        return self.height_at(root_xy)

    def terrain_class(self, heights: object) -> str:
        values = np.asarray(heights, dtype=np.float64)
        return "flat" if float(np.ptp(values)) < 0.025 else "hill"

    def height_grid(
        self,
        *,
        bounds: tuple[float, float, float, float] = (-6.0, 6.0, -2.0, 8.0),
        shape: tuple[int, int] = (121, 101),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the exact query authority for MuJoCo heightfield construction."""

        xmin, xmax, ymin, ymax = map(float, bounds)
        rows, columns = map(int, shape)
        if xmin >= xmax or ymin >= ymax or rows < 2 or columns < 2:
            raise ValueError("terrain grid bounds or shape are invalid")
        xs = np.linspace(xmin, xmax, columns, dtype=np.float64)
        ys = np.linspace(ymin, ymax, rows, dtype=np.float64)
        grid = np.asarray(
            [[self.height_at((x, y)) for x in xs] for y in ys], dtype=np.float64
        )
        return xs, ys, grid


@dataclass(frozen=True)
class SearchResult:
    row: int
    distance: float
    candidate_count: int


class _NativeJointLimitError(ValueError):
    """A finite 36D pose exceeded an authenticated native joint range."""


class _DiagnosticPoseError(ValueError):
    """A diagnostic pose failed before native-valid commit."""


class _OutsideSupportError(ValueError):
    """A diagnostic placement or preview left authenticated terrain support."""


class _DiagnosticMechanicalRowError(ValueError):
    """A diagnostic candidate has implausible pelvis-to-support clearance."""


@dataclass(frozen=True)
class HybridRuntimeState:
    row: int
    range_index: int
    family: str
    qpos: np.ndarray
    query: np.ndarray
    terrain_features: np.ndarray
    root_position_world: np.ndarray
    heading: float
    search_distance: float
    decode_count: int
    fallback_count: int
    joint_clamp_count: int
    diagnostic_pose_rejection_count: int
    unsupported_hold_count: int
    max_joint_clamp_magnitude: float
    candidate_limit_rejection_count: int
    first_candidate_limit_rejection_row: int | None
    max_candidate_limit_rejections_per_step: int
    candidate_exhaustion_count: int
    candidate_exhausted: bool
    candidate_retry_budget: int
    pose_source: str
    support_height: float
    support_status: str
    terrain_class: str
    searchable_row_count: int
    searchable_family_counts: tuple[tuple[str, int], ...]
    total_searchable_row_count: int
    total_searchable_family_counts: tuple[tuple[str, int], ...]
    search_scope: str
    search_acceptance_eligible: bool
    transition_penalty: float
    fps: float
    dt: float
    horizons: tuple[int, int, int]
    walking_speed_p95_mps: float
    root_motion_source: str


class HybridMatcher:
    """Range-safe exact NN matcher with a transactional display-pose decode."""

    def __init__(
        self,
        corpus: object,
        generator: object,
        terrain: TerrainAuthority,
        *,
        native_model: object | None = None,
        pose_converter: Callable[
            [np.ndarray, np.ndarray, np.ndarray, object], np.ndarray
        ]
        | None = None,
        transition_penalty: float = 0.1,
        max_root_step_m: float = 1.0,
        max_search_rows: int | None = None,
        search_device: str | None = None,
        initial_root_xy: object = (0.0, 0.0),
        initial_heading: float = 0.0,
        cache_manifest_path: str | Path | None = None,
        diagnostic_stability: bool = False,
        diagnostic_canonical_source_pose: bool = False,
    ) -> None:
        if type(diagnostic_stability) is not bool:
            raise TypeError("diagnostic stability must be a boolean")
        if type(diagnostic_canonical_source_pose) is not bool:
            raise TypeError("diagnostic canonical source pose must be a boolean")
        if diagnostic_canonical_source_pose and not diagnostic_stability:
            raise ValueError(
                "diagnostic canonical source pose requires diagnostic stability"
            )
        self.diagnostic_stability = diagnostic_stability
        self.diagnostic_canonical_source_pose = diagnostic_canonical_source_pose
        physical_search_device: int | None = None
        if search_device is not None:
            if type(search_device) is not str or not search_device.startswith("cuda:"):
                raise ValueError(
                    "search device must be canonical cuda:<nonnegative integer>"
                )
            index_text = search_device.removeprefix("cuda:")
            if (
                not index_text
                or not index_text.isascii()
                or not index_text.isdigit()
                or (len(index_text) > 1 and index_text.startswith("0"))
            ):
                raise ValueError(
                    "search device must be canonical cuda:<nonnegative integer>"
                )
            physical_search_device = int(index_text)
        self.corpus = corpus
        self.generator = generator
        self.search_device = search_device
        self.cache_manifest_path = (
            None
            if cache_manifest_path is None
            else Path(cache_manifest_path).expanduser().resolve(strict=True)
        )
        if not isinstance(terrain, TerrainAuthority):
            raise TypeError("terrain must be a TerrainAuthority")
        self.terrain = terrain
        self.native_model = native_model
        self.pose_converter = pose_converter or decoded_pose_to_native_qpos
        self.transition_penalty = _finite_scalar(
            transition_penalty, "transition penalty"
        )
        if self.transition_penalty < 0.0:
            raise ValueError("transition penalty must be nonnegative")
        self.max_root_step_m = _finite_scalar(max_root_step_m, "maximum root step")
        if self.max_root_step_m <= 0.0:
            raise ValueError("maximum root step must be positive")

        corpus_has_rate = hasattr(corpus, "fps")
        rate_value = getattr(corpus, "fps", FPS)
        self.fps = _finite_scalar(rate_value, "corpus fps")
        if self.fps <= 0.0:
            raise ValueError("corpus fps must be positive")
        self._legacy_runtime = self.fps == FPS
        self.dt = 1.0 / self.fps
        default_horizons = tuple(
            round(seconds * self.fps) for seconds in (0.32, 0.68, 1.0)
        )
        horizons = tuple(getattr(corpus, "horizons", default_horizons))
        if len(horizons) != 3 or any(
            type(value) is not int or value < 1 for value in horizons
        ):
            raise ValueError("corpus horizons must be three positive integers")
        self.horizons = horizons
        retry_budget = getattr(
            corpus, "candidate_retry_budget", MAX_NATIVE_LIMIT_CANDIDATE_REJECTIONS
        )
        if type(retry_budget) is not int or retry_budget < 1:
            raise ValueError("candidate retry budget must be a positive integer")
        self.candidate_retry_budget = retry_budget

        self.artifacts = corpus.artifacts
        feature_set = corpus.features
        self.features = np.asarray(
            getattr(feature_set, "values", feature_set), dtype=np.float32
        )
        self.feature_offset = np.asarray(
            getattr(feature_set, "offset", np.zeros(31)), dtype=np.float64
        )
        self.feature_scale = np.asarray(
            getattr(feature_set, "scale", np.ones(31)), dtype=np.float64
        )
        if self.features.ndim != 2 or self.features.shape[1] != 31:
            raise ValueError("hybrid corpus features must have shape (rows, 31)")
        if self.features.shape[0] != len(self.artifacts.positions):
            raise ValueError("hybrid feature and artifact row counts differ")
        if (
            self.feature_offset.shape != (31,)
            or self.feature_scale.shape != (31,)
            or not np.isfinite(self.features).all()
            or not np.isfinite(self.feature_offset).all()
            or not np.isfinite(self.feature_scale).all()
            or np.any(self.feature_scale <= 0.0)
        ):
            raise ValueError("hybrid feature normalization is invalid")
        if not self.diagnostic_stability:
            terrain_columns = self.features[:, TERRAIN_INDICES]
            self.terrain_feature_min = np.min(terrain_columns, axis=0).astype(
                np.float64
            )
            self.terrain_feature_max = np.max(terrain_columns, axis=0).astype(
                np.float64
            )

        starts = np.asarray(self.artifacts.range_starts, dtype=np.int64)
        stops = np.asarray(self.artifacts.range_stops, dtype=np.int64)
        if (
            starts.ndim != 1
            or starts.shape != stops.shape
            or not len(starts)
            or starts[0] != 0
            or stops[-1] != len(self.features)
            or np.any(starts >= stops)
        ):
            raise ValueError("hybrid source ranges are invalid")
        self.range_starts = starts
        self.range_stops = stops
        self.row_ranges = np.empty(len(self.features), dtype=np.int32)
        range_families = np.asarray(
            getattr(
                corpus,
                "range_family_ids",
                getattr(corpus, "family_ids", np.zeros(len(starts))),
            )
        )
        if range_families.shape == (len(self.features),):
            range_families = range_families[starts]
        if range_families.shape != (len(starts),):
            raise ValueError("family IDs must be defined per range or per row")
        self.family_ids = range_families.astype(np.int64, copy=False)
        self.family_names = tuple(
            str(value) for value in getattr(corpus, "family_names", ())
        )
        safe: list[np.ndarray] = []
        for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            self.row_ranges[start:stop] = index
            # Terminal rows can be displayed, but never become search seeds.
            if stop - start > 1:
                safe.append(np.arange(start, stop - 1, dtype=np.int64))
        if not safe:
            raise ValueError("hybrid corpus has no range-safe searchable rows")
        searchable = np.concatenate(safe)
        explicit_speed = getattr(corpus, "walking_speed_p95_mps", None)
        if explicit_speed is not None:
            walking_speed = _finite_scalar(explicit_speed, "corpus walking speed p95")
        elif corpus_has_rate:
            planar_speeds: list[np.ndarray] = []
            root_xz = np.asarray(self.artifacts.positions[:, 0], np.float64)[:, (0, 2)]
            for start, stop in zip(starts, stops, strict=True):
                if stop - start > 1:
                    planar_speeds.append(
                        np.linalg.norm(np.diff(root_xz[start:stop], axis=0), axis=1)
                        * self.fps
                    )
            walking_speed = float(np.percentile(np.concatenate(planar_speeds), 95.0))
        else:
            # Authenticated legacy wrappers predate corpus-owned rate metadata.
            walking_speed = MAX_SPEED_MPS
        if not math.isfinite(walking_speed) or walking_speed < 0.0:
            raise ValueError("corpus walking speed p95 must be finite and nonnegative")
        self.walking_speed_p95_mps = walking_speed
        row_families = self.family_ids[self.row_ranges]
        total_searchable = np.array(searchable, dtype=np.int64, copy=True)
        self.diagnostic_mechanical_clearance_bounds_m: tuple[float, ...] = ()
        self.diagnostic_mechanical_successor_clearance_limit_m: float | None = None
        self.diagnostic_mechanical_rejected_row_count: int | None = None
        self.diagnostic_mechanical_retained_row_count: int | None = None
        self.diagnostic_mechanical_rejected_searchable_row_count: int | None = None
        self.diagnostic_mechanical_retained_searchable_row_count: int | None = None
        self.diagnostic_mechanical_discontinuous_successor_edge_count: int | None = None
        self._diagnostic_mechanical_clearance: np.ndarray | None = None
        self._diagnostic_mechanical_safe_rows: np.ndarray | None = None
        self._diagnostic_mechanical_discontinuous_successors: np.ndarray | None = None
        mechanically_searchable = total_searchable
        if self.diagnostic_stability:
            root_y = np.asarray(self.artifacts.positions[:, 0, 1], dtype=np.float64)
            if (
                root_y.shape != (len(self.features),)
                or not np.isfinite(root_y).all()
                or np.any(np.abs(root_y) > 1.0e-6)
            ):
                raise ValueError(
                    "diagnostic clearance requires authenticated Simulation-root Y=0"
                )
            for component in (1, 3):
                root_tilt = np.asarray(
                    self.artifacts.rotations[:, 0, component], dtype=np.float64
                )
                if (
                    root_tilt.shape != (len(self.features),)
                    or not np.isfinite(root_tilt).all()
                    or np.any(np.abs(root_tilt) > 1.0e-6)
                ):
                    raise ValueError(
                        "diagnostic clearance requires authenticated yaw-only "
                        "Simulation roots"
                    )
            clearance = np.asarray(
                self.artifacts.positions[:, 1, 1], dtype=np.float64
            ) - np.asarray(self.artifacts.terrain_support[:, 0], dtype=np.float64)
            if (
                clearance.shape != (len(self.features),)
                or not np.isfinite(clearance).all()
            ):
                raise ValueError(
                    "pelvis-to-support clearance must be finite per corpus row"
                )
            self.diagnostic_mechanical_clearance_bounds_m = (
                DIAGNOSTIC_MIN_PELVIS_SUPPORT_CLEARANCE_M,
                DIAGNOSTIC_MAX_PELVIS_SUPPORT_CLEARANCE_M,
            )
            self.diagnostic_mechanical_successor_clearance_limit_m = (
                DIAGNOSTIC_MAX_SUCCESSOR_CLEARANCE_STEP_M
            )
            self._diagnostic_mechanical_clearance = clearance
            self._diagnostic_mechanical_safe_rows = np.logical_and(
                clearance >= DIAGNOSTIC_MIN_PELVIS_SUPPORT_CLEARANCE_M,
                clearance <= DIAGNOSTIC_MAX_PELVIS_SUPPORT_CLEARANCE_M,
            )
            self.diagnostic_mechanical_rejected_row_count = int(
                np.count_nonzero(~self._diagnostic_mechanical_safe_rows)
            )
            self.diagnostic_mechanical_retained_row_count = int(
                np.count_nonzero(self._diagnostic_mechanical_safe_rows)
            )
            discontinuous_successors = np.zeros(len(self.features), dtype=np.bool_)
            discontinuous_successors[:-1] = (
                self._diagnostic_mechanical_safe_rows[:-1]
                & self._diagnostic_mechanical_safe_rows[1:]
                & (self.row_ranges[:-1] == self.row_ranges[1:])
                & (
                    np.abs(clearance[1:] - clearance[:-1])
                    > DIAGNOSTIC_MAX_SUCCESSOR_CLEARANCE_STEP_M
                )
            )
            self._diagnostic_mechanical_discontinuous_successors = (
                discontinuous_successors
            )
            self.diagnostic_mechanical_discontinuous_successor_edge_count = int(
                np.count_nonzero(discontinuous_successors)
            )
            next_rows = total_searchable + 1
            continuation_safe = (
                self._diagnostic_mechanical_safe_rows[total_searchable]
                & self._diagnostic_mechanical_safe_rows[next_rows]
                & ~discontinuous_successors[total_searchable]
            )
            mechanically_searchable = total_searchable[continuation_safe]
            if not len(mechanically_searchable):
                raise ValueError(
                    "diagnostic mechanical filter removed every searchable row"
                )
            searchable = mechanically_searchable
            self.diagnostic_mechanical_rejected_searchable_row_count = int(
                len(total_searchable) - len(mechanically_searchable)
            )
            self.diagnostic_mechanical_retained_searchable_row_count = int(
                len(mechanically_searchable)
            )
            retained_terrain = self.features[mechanically_searchable, TERRAIN_INDICES]
            self.terrain_feature_min = np.min(retained_terrain, axis=0).astype(
                np.float64
            )
            self.terrain_feature_max = np.max(retained_terrain, axis=0).astype(
                np.float64
            )

        def family_counts(rows: np.ndarray) -> tuple[tuple[str, int], ...]:
            counts: list[tuple[str, int]] = []
            for family in np.unique(row_families[rows]):
                family_id = int(family)
                name = (
                    self.family_names[family_id]
                    if 0 <= family_id < len(self.family_names)
                    else str(family_id)
                )
                counts.append(
                    (name, int(np.count_nonzero(row_families[rows] == family)))
                )
            return tuple(counts)

        self.total_searchable_row_count = len(total_searchable)
        self.total_searchable_family_counts = family_counts(total_searchable)
        if max_search_rows is not None:
            cap = int(max_search_rows)
            if cap < 1:
                raise ValueError("maximum search rows must be positive")
            if len(searchable) > cap:
                family_values = np.unique(row_families[searchable])
                capacities = {
                    int(family): int(
                        np.count_nonzero(row_families[searchable] == family)
                    )
                    for family in family_values
                }
                quotas = {int(family): 0 for family in family_values}
                remaining = cap
                active = [int(family) for family in family_values]
                while remaining and active:
                    share, extra = divmod(remaining, len(active))
                    granted = 0
                    for slot, family in enumerate(active):
                        available = capacities[family] - quotas[family]
                        request = share + int(slot < extra)
                        addition = min(available, request)
                        quotas[family] += addition
                        granted += addition
                    remaining -= granted
                    active = [
                        family
                        for family in active
                        if quotas[family] < capacities[family]
                    ]
                    if granted == 0:
                        raise AssertionError("family water-fill made no progress")
                selected: list[np.ndarray] = []
                for family in family_values:
                    rows = searchable[row_families[searchable] == family]
                    count = quotas[int(family)]
                    if count:
                        slots = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
                        chosen = rows[slots]
                        selected.append(chosen)
                searchable = np.sort(np.concatenate(selected))
        self.searchable_rows = searchable.astype(np.int64, copy=False)
        self.search_acceptance_eligible = (
            len(self.searchable_rows) == self.total_searchable_row_count
            and np.array_equal(self.searchable_rows, total_searchable)
            and not self.diagnostic_stability
        )
        retained_view_complete = np.array_equal(
            self.searchable_rows, mechanically_searchable
        )
        if self.diagnostic_stability:
            self.search_scope = (
                "diagnostic-mechanically-filtered-corpus"
                if retained_view_complete
                else "diagnostic-mechanically-filtered-cap"
            )
        else:
            self.search_scope = (
                "full-range-safe-corpus"
                if self.search_acceptance_eligible
                else "diagnostic-stratified-cap"
            )
        if physical_search_device is not None:
            if self.diagnostic_stability and not retained_view_complete:
                raise ValueError(
                    "single-GPU search requires the complete mechanically filtered view"
                )
            if not self.diagnostic_stability and not self.search_acceptance_eligible:
                raise ValueError(
                    "single-GPU search requires the full-range-safe searchable corpus"
                )
        self.search_view_sha256 = hashlib.sha256(
            np.ascontiguousarray(self.searchable_rows, dtype="<i8").tobytes()
        ).hexdigest()
        self.searchable_family_counts = family_counts(self.searchable_rows)
        self.last_search_elapsed_ms: float | None = None
        self.warm_search_elapsed_ms: float | None = None
        if physical_search_device is None:
            self.search_backend_identity = (
                "cpu-ckdtree-mechanically-filtered-exact"
                if self.diagnostic_stability
                else "cpu-ckdtree-exact"
            )
            self._tree_values = np.asarray(self.features[searchable], dtype=np.float64)
            self._tree = cKDTree(
                self._tree_values, compact_nodes=True, balanced_tree=True
            )
        else:
            from .hybrid_terrain_lmm_gpu_search import SingleGpuExactSearch

            contact_values = np.asarray(self.artifacts.contacts)
            if contact_values.shape != (len(self.features), 2) or not np.all(
                (contact_values == 0) | (contact_values == 1)
            ):
                raise ValueError(
                    "single-GPU runtime contacts must have shape (rows, 2) "
                    "and contain only booleans"
                )
            searchable_contacts = np.asarray(
                contact_values[self.searchable_rows], dtype=np.uint8
            )
            contact_codes = searchable_contacts[:, 0] | (searchable_contacts[:, 1] << 1)
            self._searchable_contact_counts = tuple(
                int(value) for value in np.bincount(contact_codes, minlength=4)
            )
            self.search_backend_identity = (
                f"single-gpu-mechanically-filtered-fp32:{search_device}"
                if self.diagnostic_stability
                else f"single-gpu-full-row-fp32:{search_device}"
            )
            self._gpu_search = SingleGpuExactSearch(
                self.features,
                self.searchable_rows,
                self.row_ranges,
                self.artifacts.contacts,
                physical_device_index=physical_search_device,
                transition_penalty=self.transition_penalty,
                exclusion_budget=(
                    self.candidate_retry_budget + 2
                    if self.diagnostic_stability
                    else self.candidate_retry_budget
                ),
            )

        self._elapsed_since_search = SEARCH_INTERVAL_S
        self._last_command: CommandState | None = None
        self._last_terrain_class: str | None = None
        spawn_xy = np.asarray(initial_root_xy, dtype=np.float64)
        spawn_heading = _finite_scalar(initial_heading, "initial heading")
        if spawn_xy.shape != (2,) or not np.isfinite(spawn_xy).all():
            raise ValueError("initial root XY must be one finite native point")
        self._initial_root_xy = np.array(spawn_xy, copy=True)
        self._initial_heading = math.remainder(spawn_heading, 2.0 * math.pi)
        self._root_xy = np.array(self._initial_root_xy, copy=True)
        self._heading = self._initial_heading
        self._world_transform = SE2Transform(self._root_xy, self._heading)
        initial_row = int(self.searchable_rows[0]) if self.diagnostic_stability else 0
        initial_range = int(self.row_ranges[initial_row])
        self._active_source_root = self._source_root_transform(initial_row)
        initial_terrain = self._preview_terrain(CommandState())
        initial_domain_supported = self._preview_domain_supported(CommandState())
        initial_query = np.array(
            self.features[initial_row], dtype=np.float64, copy=True
        )
        initial_query[TERRAIN_INDICES] = self._normalize_terrain(initial_terrain)
        self.state = HybridRuntimeState(
            row=initial_row,
            range_index=initial_range,
            family=self._family_name(initial_range),
            qpos=_frozen(np.zeros(36), np.float64),
            query=_frozen(initial_query, np.float64),
            terrain_features=_frozen(initial_query[TERRAIN_INDICES], np.float64),
            root_position_world=_frozen((0.0, 0.0, 0.0), np.float64),
            heading=0.0,
            search_distance=0.0,
            decode_count=0,
            fallback_count=0,
            joint_clamp_count=0,
            diagnostic_pose_rejection_count=0,
            unsupported_hold_count=0,
            max_joint_clamp_magnitude=0.0,
            candidate_limit_rejection_count=0,
            first_candidate_limit_rejection_row=None,
            max_candidate_limit_rejections_per_step=0,
            candidate_exhaustion_count=0,
            candidate_exhausted=False,
            candidate_retry_budget=self.candidate_retry_budget,
            pose_source="uninitialized",
            support_height=self.terrain.support(self._root_xy),
            support_status=self._terrain_support_status(
                initial_query[TERRAIN_INDICES],
                domain_supported=initial_domain_supported,
            ),
            terrain_class=self.terrain.terrain_class(initial_terrain),
            searchable_row_count=len(self.searchable_rows),
            searchable_family_counts=self.searchable_family_counts,
            total_searchable_row_count=self.total_searchable_row_count,
            total_searchable_family_counts=self.total_searchable_family_counts,
            search_scope=self.search_scope,
            search_acceptance_eligible=self.search_acceptance_eligible,
            transition_penalty=self.transition_penalty,
            fps=self.fps,
            dt=self.dt,
            horizons=self.horizons,
            walking_speed_p95_mps=self.walking_speed_p95_mps,
            root_motion_source="canonical-simulation-se2",
        )
        self._commit_row(
            initial_row,
            initial_query,
            search_distance=0.0,
            count_decode=False,
            live_raw=initial_terrain,
            live_domain_supported=initial_domain_supported,
        )

    def _family_name(self, range_index: int) -> str:
        family_id = int(self.family_ids[range_index])
        return (
            self.family_names[family_id]
            if 0 <= family_id < len(self.family_names)
            else str(family_id)
        )

    def _source_root_transform(self, row: int) -> SE2Transform:
        position = np.asarray(self.artifacts.positions[row, 0], dtype=np.float64)
        rotation = np.asarray(self.artifacts.rotations[row, 0], dtype=np.float64)
        if (
            position.shape != (3,)
            or rotation.shape != (4,)
            or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
        ):
            raise ValueError("source root transform is invalid")
        forward = quat.mul_vec(rotation, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
        horizontal_norm = float(np.linalg.norm(forward[[0, 2]]))
        if not math.isfinite(horizontal_norm) or horizontal_norm <= 1.0e-8:
            raise ValueError("source root heading is degenerate")
        yaw = math.atan2(float(forward[0]), float(forward[2]))
        return SE2Transform((position[0], -position[2]), yaw)

    def _set_world_transform(self, transform: SE2Transform) -> None:
        self._world_transform = transform
        self._root_xy[:] = transform.xy
        self._heading = transform.yaw

    def _advance_world_to_source_row(self, row: int) -> None:
        source = self._source_root_transform(row)
        absolute_delta = source.xy - self._active_source_root.xy
        cosine = math.cos(self._active_source_root.yaw)
        sine = math.sin(self._active_source_root.yaw)
        local_delta = np.asarray(
            (
                cosine * absolute_delta[0] + sine * absolute_delta[1],
                -sine * absolute_delta[0] + cosine * absolute_delta[1],
            ),
            dtype=np.float64,
        )
        local_yaw = math.remainder(
            source.yaw - self._active_source_root.yaw, 2.0 * math.pi
        )
        self._set_world_transform(
            compose_root_delta(self._world_transform, local_delta, local_yaw)
        )

    def _normalize_terrain(self, raw: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = (
                np.asarray(raw, dtype=np.float64) - self.feature_offset[TERRAIN_INDICES]
            ) / self.feature_scale[TERRAIN_INDICES]
        if result.shape != (4,) or not np.isfinite(result).all():
            raise ValueError("live terrain normalization produced invalid channels")
        return result

    def _terrain_support_status(
        self, normalized: object, *, domain_supported: bool = True
    ) -> str:
        if not domain_supported:
            return "OUT OF TRAINED SUPPORT"
        values = np.asarray(normalized, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            return "OUT OF TRAINED SUPPORT"
        tolerance = (
            32.0
            * np.finfo(np.float32).eps
            * np.maximum(
                1.0,
                np.maximum(
                    np.abs(self.terrain_feature_min),
                    np.abs(self.terrain_feature_max),
                ),
            )
        )
        supported = np.all(values >= self.terrain_feature_min - tolerance) and np.all(
            values <= self.terrain_feature_max + tolerance
        )
        return "SUPPORTED" if supported else "OUT OF TRAINED SUPPORT"

    def _candidate_score(self, query: np.ndarray, rows: np.ndarray) -> np.ndarray:
        delta = self.features[rows].astype(np.float64) - query[None, :]
        score = np.einsum("ij,ij->i", delta, delta, dtype=np.float64)
        if self.transition_penalty and hasattr(self, "state"):
            current_range = int(self.state.range_index)
            score += self.transition_penalty * (self.row_ranges[rows] != current_range)
        return score

    @staticmethod
    def _best(rows: np.ndarray, scores: np.ndarray) -> tuple[int, float]:
        order = np.lexsort((rows, scores))
        index = int(order[0])
        return int(rows[index]), float(scores[index])

    def _normalized_excluded_rows(self, excluded_rows: Iterable[int]) -> np.ndarray:
        try:
            values = tuple(excluded_rows)
        except TypeError as error:
            raise TypeError(
                "excluded rows must be an iterable of row integers"
            ) from error
        normalized: list[int] = []
        for value in values:
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError("excluded rows must contain only row integers")
            row = int(value)
            if not 0 <= row < len(self.features):
                raise IndexError("excluded row is outside the corpus")
            normalized.append(row)
        if not normalized:
            return np.empty(0, dtype=np.int64)
        unique = np.unique(np.asarray(normalized, dtype=np.int64))
        indices = np.searchsorted(self.searchable_rows, unique)
        present = indices < len(self.searchable_rows)
        present[present] &= self.searchable_rows[indices[present]] == unique[present]
        return unique[present]

    def _match_exclusions(self, excluded_rows: Iterable[int]) -> np.ndarray:
        excluded = self._normalized_excluded_rows(excluded_rows)
        if not hasattr(self, "state"):
            return excluded
        contacts = np.asarray(self.artifacts.contacts)
        if contacts.shape != (len(self.features), 2):
            raise ValueError("runtime contact table must have shape (rows, 2)")
        active = np.asarray(contacts[self.state.row], dtype=np.bool_)
        compatible = np.all(
            np.asarray(contacts[self.searchable_rows], dtype=np.bool_) == active,
            axis=1,
        )
        incompatible = self.searchable_rows[~compatible]
        if not len(excluded):
            return incompatible.astype(np.int64, copy=False)
        return np.union1d(excluded, incompatible).astype(np.int64, copy=False)

    @staticmethod
    def _exclude_rows(rows: np.ndarray, excluded: np.ndarray) -> np.ndarray:
        if not len(excluded):
            return rows
        return rows[~np.isin(rows, excluded, assume_unique=True)]

    def brute_force_match(
        self, query: object, *, excluded_rows: Iterable[int] = ()
    ) -> SearchResult:
        value = np.asarray(query, dtype=np.float64)
        if value.shape != (31,) or not np.isfinite(value).all():
            raise ValueError("search query must be one finite 31-D row")
        excluded = self._match_exclusions(excluded_rows)
        remaining = len(self.searchable_rows) - len(excluded)
        if remaining < 1:
            raise ValueError("candidate exclusions removed every searchable row")
        best_row: int | None = None
        best = math.inf
        for start in range(0, len(self.searchable_rows), 131_072):
            rows = self.searchable_rows[start : start + 131_072]
            rows = self._exclude_rows(rows, excluded)
            if not len(rows):
                continue
            score = self._candidate_score(value, rows)
            row, candidate = self._best(rows, score)
            if candidate < best or (
                candidate == best and (best_row is None or row < best_row)
            ):
                best_row, best = row, candidate
        if best_row is None:
            raise AssertionError("range-safe search view became empty")
        return SearchResult(best_row, math.sqrt(max(best, 0.0)), remaining)

    def _single_gpu_match(
        self, query: np.ndarray, *, excluded_rows: Iterable[int]
    ) -> SearchResult:
        excluded = self._normalized_excluded_rows(excluded_rows)
        current_row = int(self.state.row)
        current_range = int(self.row_ranges[current_row])
        contacts = np.asarray(self.artifacts.contacts)
        if contacts.shape != (len(self.features), 2):
            raise ValueError("runtime contact table must have shape (rows, 2)")
        current_contact = np.asarray(contacts[current_row], dtype=np.uint8)
        active_contact_code = int(current_contact[0]) | (int(current_contact[1]) << 1)
        candidates = self._gpu_search.match_candidates(
            query,
            current_range=current_range,
            active_contact_code=active_contact_code,
            excluded_rows=excluded,
        )

        try:
            raw_rows = np.asarray(candidates.rows)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "single-GPU candidate rows must be a one-dimensional integer array"
            ) from error
        if raw_rows.ndim != 1 or raw_rows.dtype.kind not in "iu":
            raise ValueError(
                "single-GPU candidate rows must be a one-dimensional integer array"
            )
        if not 1 <= len(raw_rows) <= 128:
            raise ValueError(
                "single-GPU candidate rows must contain between 1 and 128 rows"
            )
        if raw_rows.dtype.kind == "u" and np.any(raw_rows > np.iinfo(np.int64).max):
            raise ValueError("single-GPU candidate row cannot be represented exactly")
        rows = raw_rows.astype(np.int64, copy=False)
        if np.any(rows[1:] <= rows[:-1]):
            raise ValueError(
                "single-GPU candidate rows must be strictly sorted and unique"
            )
        searchable_positions = np.searchsorted(self.searchable_rows, rows)
        searchable = searchable_positions < len(self.searchable_rows)
        searchable[searchable] &= (
            self.searchable_rows[searchable_positions[searchable]] == rows[searchable]
        )
        if not np.all(searchable):
            raise ValueError("single-GPU candidate row is not searchable")
        if len(excluded) and np.any(np.isin(rows, excluded, assume_unique=True)):
            raise ValueError("single-GPU candidate row was explicitly excluded")
        candidate_contacts = np.asarray(contacts[rows])
        if (
            current_contact.shape != (2,)
            or candidate_contacts.shape != (len(rows), 2)
            or not np.all((current_contact == 0) | (current_contact == 1))
            or not np.all((candidate_contacts == 0) | (candidate_contacts == 1))
            or not np.all(candidate_contacts == current_contact)
        ):
            raise ValueError("single-GPU candidate row violates current hard contacts")

        def exact_integer_metadata(name: str) -> int:
            try:
                value = getattr(candidates, name)
            except AttributeError as error:
                raise ValueError(
                    f"single-GPU {name} must be an exact integer"
                ) from error
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise ValueError(f"single-GPU {name} must be an exact integer")
            return int(value)

        candidate_count = exact_integer_metadata("candidate_count")
        close_candidate_count = exact_integer_metadata("close_candidate_count")
        if not len(rows) <= candidate_count <= len(self.searchable_rows):
            raise ValueError(
                "single-GPU candidate_count is inconsistent with returned "
                "and searchable rows"
            )
        excluded_contacts = np.asarray(contacts[excluded])
        if excluded_contacts.shape != (len(excluded), 2) or not np.all(
            (excluded_contacts == 0) | (excluded_contacts == 1)
        ):
            raise ValueError("single-GPU explicit exclusion contacts are invalid")
        compatible_exclusion_count = int(
            np.count_nonzero(np.all(excluded_contacts == current_contact, axis=1))
        )
        expected_candidate_count = (
            self._searchable_contact_counts[active_contact_code]
            - compatible_exclusion_count
        )
        if candidate_count != expected_candidate_count:
            raise ValueError(
                "single-GPU candidate_count is inconsistent with hard contacts "
                "and explicit exclusions"
            )
        if not 1 <= close_candidate_count <= candidate_count:
            raise ValueError(
                "single-GPU close_candidate_count is inconsistent with candidates"
            )

        def finite_metadata(name: str) -> float:
            try:
                value = getattr(candidates, name)
            except AttributeError as error:
                raise ValueError(f"single-GPU {name} must be finite") from error
            if not isinstance(
                value, (int, float, np.integer, np.floating)
            ) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"single-GPU {name} must be finite")
            try:
                result = float(value)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(f"single-GPU {name} must be finite") from error
            if not math.isfinite(result):
                raise ValueError(f"single-GPU {name} must be finite")
            return result

        device_minimum_score = finite_metadata("device_minimum_score")
        if device_minimum_score < 0.0:
            raise ValueError("single-GPU device_minimum_score must be nonnegative")
        elapsed_ms = finite_metadata("elapsed_ms")
        if elapsed_ms < 0.0:
            raise ValueError(
                "single-GPU search elapsed time must be finite and nonnegative"
            )
        self.last_search_elapsed_ms = elapsed_ms
        if self.warm_search_elapsed_ms is None:
            self.warm_search_elapsed_ms = elapsed_ms
        if close_candidate_count > len(rows):
            return self.brute_force_match(query, excluded_rows=excluded)
        scores = self._candidate_score(query, rows)
        row, best = self._best(rows, scores)
        return SearchResult(
            row,
            math.sqrt(max(best, 0.0)),
            candidate_count,
        )

    def match(
        self, query: object, *, excluded_rows: Iterable[int] = ()
    ) -> SearchResult:
        value = np.asarray(query, dtype=np.float64)
        if value.shape != (31,) or not np.isfinite(value).all():
            raise ValueError("search query must be one finite 31-D row")
        if self.search_device is not None:
            return self._single_gpu_match(value, excluded_rows=excluded_rows)
        excluded = self._match_exclusions(excluded_rows)
        count = len(self.searchable_rows)
        if len(excluded) == count:
            raise ValueError("candidate exclusions removed every searchable row")
        current_rows = np.empty(0, dtype=np.int64)
        unseen_penalty = 0.0
        if self.transition_penalty and hasattr(self, "state"):
            current_range = int(self.state.range_index)
            current_rows = self.searchable_rows[
                self.row_ranges[self.searchable_rows] == current_range
            ]
            current_rows = self._exclude_rows(current_rows, excluded)
            # Every current-range row is scored explicitly below. Therefore any
            # still-unseen tree row must pay the range-transition penalty.
            unseen_penalty = self.transition_penalty
        # A nonzero range-transition cost can make the exact winner lie beyond
        # the nearest raw-feature neighbors. Score the complete current range,
        # then expand far enough to prove the penalized unseen lower bound.
        for requested in (32, 128, 512, 2_048, 8_192):
            k = min(requested, count)
            distances, indices = self._tree.query(value, k=k, workers=1)
            indices = np.atleast_1d(indices).astype(np.int64, copy=False)
            tree_distances = np.atleast_1d(distances).astype(np.float64, copy=False)
            tree_rows = self.searchable_rows[indices]
            tree_rows = self._exclude_rows(tree_rows, excluded)
            rows = (
                np.unique(np.concatenate((tree_rows, current_rows)))
                if len(current_rows)
                else tree_rows
            )
            if not len(rows):
                continue
            scores = self._candidate_score(value, rows)
            row, best = self._best(rows, scores)
            if k == count:
                return SearchResult(
                    row, math.sqrt(max(best, 0.0)), count - len(excluded)
                )
            # Every unseen feature distance is at least the kth distance. Strict
            # separation also proves that no omitted equal-distance row can win
            # the stable lowest-row tie break.
            lower_bound = float(tree_distances[-1]) ** 2 + unseen_penalty
            tolerance = np.finfo(np.float64).eps * max(1.0, abs(best), lower_bound) * 16
            if best < lower_bound - tolerance:
                return SearchResult(row, math.sqrt(max(best, 0.0)), len(rows))
        # Rare large tie/penalty ambiguity: a full float64 rescore preserves the
        # exact contract instead of silently accepting cKDTree tie ordering.
        return self.brute_force_match(value, excluded_rows=excluded)

    def successor(self, row: int) -> int:
        if type(row) is not int or not 0 <= row < len(self.features):
            raise IndexError("runtime row is outside the corpus")
        range_index = int(self.row_ranges[row])
        candidate = min(row + 1, int(self.range_stops[range_index]) - 1)
        if self.diagnostic_stability:
            safe_rows = self._diagnostic_mechanical_safe_rows
            discontinuous = self._diagnostic_mechanical_discontinuous_successors
            if safe_rows is None or discontinuous is None:
                raise AssertionError("diagnostic mechanical inventory is unavailable")
            if not safe_rows[candidate] or discontinuous[row]:
                return row
        return candidate

    def _command_query(
        self, command: CommandState, live_terrain: np.ndarray
    ) -> np.ndarray:
        query = np.array(self.features[self.state.row], dtype=np.float64, copy=True)
        horizons = np.asarray(self.horizons, dtype=np.float64) / self.fps
        speed = command.speed * self.walking_speed_p95_mps
        angular_speed = command.steering * 1.5
        future_yaw = angular_speed * horizons
        distance = speed * horizons
        trajectory = self._arc_local(distance, future_yaw).reshape(-1)
        facing = np.column_stack((np.sin(future_yaw), np.cos(future_yaw))).reshape(-1)
        raw = np.concatenate((trajectory, facing))
        query[TRAJECTORY_INDICES] = (
            raw - self.feature_offset[TRAJECTORY_INDICES]
        ) / self.feature_scale[TRAJECTORY_INDICES]
        query[TERRAIN_INDICES] = self._normalize_terrain(live_terrain)
        if not np.isfinite(query).all():
            raise ValueError("live command query became non-finite")
        return query

    @staticmethod
    def _arc_local(distance: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        travel = np.asarray(distance, dtype=np.float64)
        angle = np.asarray(yaw, dtype=np.float64)
        if (
            travel.shape != angle.shape
            or not np.isfinite(travel).all()
            or not np.isfinite(angle).all()
        ):
            raise ValueError("command arc inputs must be aligned and finite")
        small = np.abs(angle) < 1.0e-8
        x = np.empty_like(travel)
        z = np.empty_like(travel)
        x[small] = 0.0
        z[small] = travel[small]
        x[~small] = travel[~small] * (1.0 - np.cos(angle[~small])) / angle[~small]
        z[~small] = travel[~small] * np.sin(angle[~small]) / angle[~small]
        return np.column_stack((x, z))

    def _preview_terrain(self, command: CommandState) -> np.ndarray:
        return self.terrain.sample_points(self._root_xy, self._preview_points(command))

    def _preview_points(self, command: CommandState) -> np.ndarray:
        distances = np.asarray((0.25, 0.5, 0.75, 1.0), dtype=np.float64)
        speed = command.speed * self.walking_speed_p95_mps
        angular_speed = command.steering * 1.5
        if abs(speed) < 1.0e-8:
            signed_distance = distances
            if self.walking_speed_p95_mps < 1.0e-8:
                future_yaw = np.zeros_like(distances)
            else:
                future_yaw = angular_speed * distances / self.walking_speed_p95_mps
        else:
            signed_distance = math.copysign(1.0, speed) * distances
            future_yaw = angular_speed * distances / abs(speed)
        local = self._arc_local(signed_distance, future_yaw)
        cosine, sine = math.cos(self._heading), math.sin(self._heading)
        world = np.empty_like(local)
        world[:, 0] = self._root_xy[0] + cosine * local[:, 0] + sine * local[:, 1]
        world[:, 1] = self._root_xy[1] + sine * local[:, 0] - cosine * local[:, 1]
        return world

    def _preview_domain_supported(self, command: CommandState) -> bool:
        points = self._preview_points(command)
        return self.terrain.contains(self._root_xy) and all(
            self.terrain.contains(point) for point in points
        )

    def _latent_for_row(self, row: int) -> np.ndarray:
        for name in ("latent", "latents", "latent_rows"):
            if hasattr(self.generator, name):
                table = np.asarray(getattr(self.generator, name))
                if table.ndim == 2 and table.shape[0] == len(self.features):
                    return np.asarray(table[row], dtype=np.float32)
        raise ValueError("hybrid generator has no row-aligned latent table")

    def _decode(self, row: int, features: np.ndarray) -> np.ndarray:
        latent = self._latent_for_row(row)
        if callable(getattr(self.generator, "decode_rows", None)):
            output = self.generator.decode_rows(
                np.asarray(features, dtype=np.float32)[None],
                np.asarray((row,), dtype=np.int64),
            )
        elif callable(getattr(self.generator, "decode_row", None)):
            output = self.generator.decode_row(row, features)
        elif callable(getattr(self.generator, "decode", None)):
            output = self.generator.decode(
                np.asarray(features, dtype=np.float32)[None], latent=latent[None]
            )
        elif callable(getattr(self.generator, "decompressor", None)):
            state = np.concatenate((features, latent)).astype(np.float32)
            output = self.generator.decompressor(state[None])
        else:
            raise TypeError("hybrid generator provides no decode interface")
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        result = np.asarray(output, dtype=np.float32).squeeze()
        if result.shape != (458,) or not np.isfinite(result).all():
            raise ValueError("learned generator produced an invalid decoded pose")
        return result

    def _canonical_decoded(self, row: int) -> np.ndarray:
        positions = np.asarray(self.artifacts.positions[row], dtype=np.float32)
        rotations = np.asarray(self.artifacts.rotations[row], dtype=np.float32)
        velocities = np.asarray(self.artifacts.velocities[row], dtype=np.float32)
        angular = np.asarray(self.artifacts.angular_velocities[row], dtype=np.float32)
        rotation_xy = quat.to_xform_xy(rotations).astype(np.float32)
        root_velocity = quat.inv_mul_vec(rotations[0], velocities[0]).astype(np.float32)
        root_angular = quat.inv_mul_vec(rotations[0], angular[0]).astype(np.float32)
        decoded = np.concatenate(
            (
                positions[1:].reshape(-1),
                rotation_xy[1:].reshape(-1),
                velocities[1:].reshape(-1),
                angular[1:].reshape(-1),
                root_velocity,
                root_angular,
                np.asarray(self.artifacts.contacts[row], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        if decoded.shape != (458,) or not np.isfinite(decoded).all():
            raise ValueError("canonical source pose is invalid")
        return decoded

    def _source_clearance(self, row: int, canonical: np.ndarray) -> float:
        source_position = np.asarray(self.artifacts.positions[row, 0], np.float32)
        source_rotation = np.asarray(self.artifacts.rotations[row, 0], np.float32)
        source_qpos = np.asarray(
            self.pose_converter(
                canonical, source_position, source_rotation, self.native_model
            ),
            dtype=np.float64,
        )
        if source_qpos.shape != (36,) or not np.isfinite(source_qpos).all():
            raise ValueError("canonical source conversion is invalid")
        clearance = float(source_qpos[2]) - float(
            self.artifacts.terrain_support[row, 0]
        )
        if not math.isfinite(clearance):
            raise ValueError("selected support-relative clearance is invalid")
        return clearance

    def _placed_transform(
        self, decoded: np.ndarray, target_root_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        half = 0.5 * self._heading
        simulation_rotation = np.asarray(
            (math.cos(half), 0.0, math.sin(half), 0.0), dtype=np.float32
        )
        desired_holden = np.asarray(
            (target_root_world[0], target_root_world[2], -target_root_world[1]),
            dtype=np.float32,
        )
        local_root = np.asarray(decoded[:3], dtype=np.float32)
        simulation_position = desired_holden - quat.mul_vec(
            simulation_rotation, local_root
        ).astype(np.float32)
        return simulation_position, simulation_rotation

    def _validate_qpos(self, qpos: np.ndarray) -> None:
        if qpos.shape != (36,) or not np.isfinite(qpos).all():
            raise ValueError("runtime qpos must be finite canonical G1 qpos")
        if self.native_model is not None:
            for joint_id in range(1, int(self.native_model.njnt)):
                address = int(self.native_model.jnt_qposadr[joint_id])
                lower, upper = np.asarray(
                    self.native_model.jnt_range[joint_id], np.float64
                )
                if qpos[address] < lower - 1e-5 or qpos[address] > upper + 1e-5:
                    raise _NativeJointLimitError(
                        "runtime decoded joint left its native limit"
                    )
        if self.state.pose_source != "uninitialized":
            step = float(np.linalg.norm(qpos[:3] - self.state.qpos[:3]))
            if step > self.max_root_step_m:
                raise ValueError("runtime decoded root exceeded the step bound")

    def _clamp_canonical_hinges(self, qpos: np.ndarray) -> tuple[np.ndarray, float]:
        if self.native_model is None:
            raise ValueError("canonical hinge clamp requires a native model")
        result = np.array(qpos, dtype=np.float64, copy=True)
        if result.shape != (36,) or not np.isfinite(result).all():
            raise ValueError("runtime qpos must be finite canonical G1 qpos")
        root = np.array(result[:7], copy=True)
        max_magnitude = 0.0
        joint_types = np.asarray(self.native_model.jnt_type)
        joint_limited = getattr(self.native_model, "jnt_limited", None)
        for joint_id in range(1, int(self.native_model.njnt)):
            if int(joint_types[joint_id]) != 3:
                continue
            if joint_limited is not None and not bool(joint_limited[joint_id]):
                continue
            address = int(self.native_model.jnt_qposadr[joint_id])
            if address < 7 or address >= len(result):
                raise ValueError("native hinge qpos address is invalid")
            lower, upper = np.asarray(
                self.native_model.jnt_range[joint_id], dtype=np.float64
            )
            if not np.isfinite((lower, upper)).all() or lower > upper:
                raise ValueError("native hinge joint range is invalid")
            clamped = float(np.clip(result[address], lower, upper))
            max_magnitude = max(max_magnitude, abs(clamped - result[address]))
            result[address] = clamped
        if not np.array_equal(result[:7], root):
            raise AssertionError("canonical hinge clamp modified the free root")
        return result, max_magnitude

    def _commit_row(
        self,
        row: int,
        query: np.ndarray,
        *,
        search_distance: float,
        count_decode: bool = True,
        live_raw: np.ndarray | None = None,
        live_domain_supported: bool | None = None,
        reject_learned_native_limit: bool = False,
    ) -> HybridRuntimeState:
        if (
            self.diagnostic_stability
            and not self._diagnostic_mechanical_safe_rows[int(row)]
        ):
            raise _DiagnosticMechanicalRowError(
                "diagnostic candidate is outside the mechanical clearance band"
            )
        range_index = int(self.row_ranges[row])
        if live_raw is None:
            live_raw = self._preview_terrain(CommandState())
        if live_domain_supported is None:
            live_domain_supported = self._preview_domain_supported(CommandState())
        live_raw = np.asarray(live_raw, dtype=np.float64)
        live_normalized = self._normalize_terrain(live_raw)
        seeded_features = np.array(self.features[row], dtype=np.float32, copy=True)
        seeded_features[TERRAIN_INDICES] = live_normalized.astype(np.float32)
        canonical = self._canonical_decoded(row)
        support = self.terrain.support(self._root_xy)
        target_root = np.asarray(
            (
                self._root_xy[0],
                self._root_xy[1],
                support + self._source_clearance(row, canonical),
            ),
            dtype=np.float64,
        )

        learned_count = self.state.decode_count
        fallback_count = self.state.fallback_count
        joint_clamp_count = self.state.joint_clamp_count
        diagnostic_pose_rejection_count = self.state.diagnostic_pose_rejection_count
        unsupported_hold_count = self.state.unsupported_hold_count
        max_joint_clamp_magnitude = 0.0
        pose_source = (
            "canonical-diagnostic"
            if self.diagnostic_canonical_source_pose
            else "learned"
        )
        support_status = self._terrain_support_status(
            live_normalized, domain_supported=bool(live_domain_supported)
        )
        if self.diagnostic_stability and support_status != "SUPPORTED":
            raise _OutsideSupportError(
                "diagnostic placement is outside authenticated terrain support"
            )
        try:
            if support_status != "SUPPORTED":
                raise ValueError("live terrain is outside the corpus envelope")
            decoded = (
                canonical
                if self.diagnostic_canonical_source_pose
                else self._decode(row, seeded_features)
            )
            simulation_position, simulation_rotation = self._placed_transform(
                decoded, target_root
            )
            proposed_qpos = np.asarray(
                self.pose_converter(
                    decoded, simulation_position, simulation_rotation, self.native_model
                ),
                dtype=np.float64,
            )
            self._validate_qpos(proposed_qpos)
            if count_decode and not self.diagnostic_canonical_source_pose:
                learned_count += 1
        except _NativeJointLimitError:
            if reject_learned_native_limit or self.diagnostic_stability:
                raise
            simulation_position, simulation_rotation = self._placed_transform(
                canonical, target_root
            )
            proposed_qpos = np.asarray(
                self.pose_converter(
                    canonical,
                    simulation_position,
                    simulation_rotation,
                    self.native_model,
                ),
                dtype=np.float64,
            )
            try:
                self._validate_qpos(proposed_qpos)
            except _NativeJointLimitError:
                proposed_qpos, max_joint_clamp_magnitude = self._clamp_canonical_hinges(
                    proposed_qpos
                )
                self._validate_qpos(proposed_qpos)
                joint_clamp_count += int(count_decode)
                pose_source = "canonical-fallback-joint-clamped"
            fallback_count += int(count_decode)
            if max_joint_clamp_magnitude == 0.0:
                pose_source = "canonical-fallback"
        except (
            AttributeError,
            FloatingPointError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            if self.diagnostic_stability:
                source = (
                    "canonical" if self.diagnostic_canonical_source_pose else "learned"
                )
                raise _DiagnosticPoseError(
                    f"diagnostic {source} pose candidate failed validation"
                ) from error
            simulation_position, simulation_rotation = self._placed_transform(
                canonical, target_root
            )
            proposed_qpos = np.asarray(
                self.pose_converter(
                    canonical,
                    simulation_position,
                    simulation_rotation,
                    self.native_model,
                ),
                dtype=np.float64,
            )
            try:
                self._validate_qpos(proposed_qpos)
            except _NativeJointLimitError:
                proposed_qpos, max_joint_clamp_magnitude = self._clamp_canonical_hinges(
                    proposed_qpos
                )
                self._validate_qpos(proposed_qpos)
                joint_clamp_count += int(count_decode)
                pose_source = "canonical-fallback-joint-clamped"
            fallback_count += int(count_decode)
            if max_joint_clamp_magnitude == 0.0:
                pose_source = "canonical-fallback"

        proposed = HybridRuntimeState(
            row=int(row),
            range_index=range_index,
            family=self._family_name(range_index),
            qpos=_frozen(proposed_qpos, np.float64),
            query=_frozen(query, np.float64),
            terrain_features=_frozen(live_normalized, np.float64),
            root_position_world=_frozen(target_root, np.float64),
            heading=float(self._heading),
            search_distance=float(search_distance),
            decode_count=learned_count,
            fallback_count=fallback_count,
            joint_clamp_count=joint_clamp_count,
            diagnostic_pose_rejection_count=diagnostic_pose_rejection_count,
            unsupported_hold_count=unsupported_hold_count,
            max_joint_clamp_magnitude=max_joint_clamp_magnitude,
            candidate_limit_rejection_count=(
                self.state.candidate_limit_rejection_count
            ),
            first_candidate_limit_rejection_row=(
                self.state.first_candidate_limit_rejection_row
            ),
            max_candidate_limit_rejections_per_step=(
                self.state.max_candidate_limit_rejections_per_step
            ),
            candidate_exhaustion_count=self.state.candidate_exhaustion_count,
            candidate_exhausted=False,
            candidate_retry_budget=self.candidate_retry_budget,
            pose_source=pose_source,
            support_height=float(support),
            support_status=support_status,
            terrain_class=self.terrain.terrain_class(live_raw),
            searchable_row_count=len(self.searchable_rows),
            searchable_family_counts=self.searchable_family_counts,
            total_searchable_row_count=self.total_searchable_row_count,
            total_searchable_family_counts=self.total_searchable_family_counts,
            search_scope=self.search_scope,
            search_acceptance_eligible=self.search_acceptance_eligible,
            transition_penalty=self.transition_penalty,
            fps=self.fps,
            dt=self.dt,
            horizons=self.horizons,
            walking_speed_p95_mps=self.walking_speed_p95_mps,
            root_motion_source="canonical-simulation-se2",
        )
        self.state = proposed
        self._active_source_root = self._source_root_transform(int(row))
        return proposed

    def _hold_outside_support(self, baseline: HybridRuntimeState) -> HybridRuntimeState:
        proposed = replace(
            baseline,
            pose_source="held-outside-support",
            support_status="OUT OF TRAINED SUPPORT",
            unsupported_hold_count=baseline.unsupported_hold_count + 1,
            candidate_exhausted=False,
        )
        self.state = proposed
        return proposed

    def _commit_with_native_limit_retry(
        self,
        row: int,
        query: np.ndarray,
        *,
        search_distance: float,
        live_raw: np.ndarray,
        live_domain_supported: bool,
        advance_source: bool = False,
        command: CommandState | None = None,
        base_excluded_rows: Iterable[int] = (),
    ) -> HybridRuntimeState:
        baseline = self.state
        baseline_world = self._world_transform
        baseline_source = self._active_source_root
        original_row = int(row)
        original_distance = float(search_distance)
        rejected: list[int] = []
        native_rejected: list[int] = []
        diagnostic_pose_rejected: list[int] = []
        base_exclusions = self._normalized_excluded_rows(base_excluded_rows)
        candidate_advances_source = bool(advance_source)

        base_query = np.array(query, dtype=np.float64, copy=True)

        def prepare_candidate(
            candidate_row: int,
        ) -> tuple[np.ndarray, np.ndarray, bool]:
            candidate_query = np.array(base_query, dtype=np.float64, copy=True)
            candidate_live_raw = np.asarray(live_raw, dtype=np.float64)
            candidate_live_domain_supported = bool(live_domain_supported)
            if candidate_advances_source:
                self._set_world_transform(baseline_world)
                self._active_source_root = baseline_source
                self._advance_world_to_source_row(int(candidate_row))
                active_command = command or CommandState()
                candidate_live_raw = self._preview_terrain(active_command)
                candidate_live_domain_supported = self._preview_domain_supported(
                    active_command
                )
                candidate_query[TERRAIN_INDICES] = self._normalize_terrain(
                    candidate_live_raw
                )
            return (
                candidate_query,
                np.asarray(candidate_live_raw, dtype=np.float64),
                bool(candidate_live_domain_supported),
            )

        (
            prepared_query,
            prepared_live_raw,
            prepared_live_domain_supported,
        ) = prepare_candidate(int(row))
        original_query = np.array(prepared_query, dtype=np.float64, copy=True)
        original_live_raw = np.asarray(prepared_live_raw, dtype=np.float64, copy=True)
        original_live_domain_supported = bool(prepared_live_domain_supported)
        while True:
            try:
                if self.diagnostic_stability and not candidate_advances_source:
                    clearance = self._diagnostic_mechanical_clearance
                    if clearance is None:
                        raise AssertionError(
                            "diagnostic mechanical clearance is unavailable"
                        )
                    if (
                        abs(float(clearance[row]) - float(clearance[baseline.row]))
                        > DIAGNOSTIC_MAX_SUCCESSOR_CLEARANCE_STEP_M
                    ):
                        raise _DiagnosticMechanicalRowError(
                            "diagnostic search jump exceeded the clearance-step limit"
                        )
                proposed = self._commit_row(
                    row,
                    prepared_query,
                    search_distance=search_distance,
                    live_raw=prepared_live_raw,
                    live_domain_supported=prepared_live_domain_supported,
                    reject_learned_native_limit=True,
                )
                break
            except _OutsideSupportError:
                self._set_world_transform(baseline_world)
                self._active_source_root = baseline_source
                proposed = self._hold_outside_support(baseline)
                break
            except (
                _DiagnosticMechanicalRowError,
                _DiagnosticPoseError,
                _NativeJointLimitError,
            ) as error:
                rejected.append(int(row))
                if isinstance(error, _NativeJointLimitError):
                    native_rejected.append(int(row))
                elif isinstance(
                    error, (_DiagnosticMechanicalRowError, _DiagnosticPoseError)
                ):
                    diagnostic_pose_rejected.append(int(row))
                self._set_world_transform(baseline_world)
                self._active_source_root = baseline_source
                combined_exclusions = np.concatenate(
                    (base_exclusions, np.asarray(rejected, dtype=np.int64))
                )
                if self.search_device is None:
                    retry_exclusions = self._match_exclusions(combined_exclusions)
                    contact_exhausted = len(retry_exclusions) == len(
                        self.searchable_rows
                    )
                else:
                    retry_exclusions = self._normalized_excluded_rows(
                        combined_exclusions
                    )
                    contacts = np.asarray(self.artifacts.contacts)
                    active_contact = np.asarray(contacts[self.state.row])
                    retry_contacts = np.asarray(contacts[retry_exclusions])
                    if (
                        active_contact.shape != (2,)
                        or retry_contacts.shape != (len(retry_exclusions), 2)
                        or not np.all((active_contact == 0) | (active_contact == 1))
                        or not np.all((retry_contacts == 0) | (retry_contacts == 1))
                    ):
                        raise ValueError("single-GPU retry contacts are invalid")
                    retry_exclusions = retry_exclusions[
                        np.all(retry_contacts == active_contact, axis=1)
                    ]
                    active_contact_code = int(active_contact[0]) | (
                        int(active_contact[1]) << 1
                    )
                    contact_exhausted = (
                        len(retry_exclusions)
                        >= (self._searchable_contact_counts[active_contact_code])
                    )
                exhausted = (
                    len(rejected) >= self.candidate_retry_budget or contact_exhausted
                )
                if exhausted:
                    if self._legacy_runtime and not self.diagnostic_stability:
                        proposed = self._commit_row(
                            original_row,
                            original_query,
                            search_distance=original_distance,
                            live_raw=original_live_raw,
                            live_domain_supported=original_live_domain_supported,
                        )
                        break
                    proposed = replace(
                        baseline,
                        candidate_limit_rejection_count=(
                            baseline.candidate_limit_rejection_count
                            + len(native_rejected)
                        ),
                        first_candidate_limit_rejection_row=(
                            baseline.first_candidate_limit_rejection_row
                            if baseline.first_candidate_limit_rejection_row is not None
                            else (native_rejected[0] if native_rejected else None)
                        ),
                        max_candidate_limit_rejections_per_step=max(
                            baseline.max_candidate_limit_rejections_per_step,
                            len(native_rejected),
                        ),
                        diagnostic_pose_rejection_count=(
                            baseline.diagnostic_pose_rejection_count
                            + len(diagnostic_pose_rejected)
                        ),
                        candidate_exhaustion_count=(
                            baseline.candidate_exhaustion_count + 1
                        ),
                        candidate_exhausted=True,
                    )
                    self.state = proposed
                    break
                retry_query = (
                    base_query if self.diagnostic_stability else prepared_query
                )
                result = self.match(retry_query, excluded_rows=retry_exclusions)
                row = result.row
                search_distance = result.distance
                if self.diagnostic_stability:
                    # Only a real within-range successor composes source motion.
                    # A retry selected by search is a jump and reanchors at the
                    # unchanged baseline world transform.
                    candidate_advances_source = False
                (
                    prepared_query,
                    prepared_live_raw,
                    prepared_live_domain_supported,
                ) = prepare_candidate(int(row))
        if rejected and not proposed.candidate_exhausted:
            proposed = replace(
                proposed,
                candidate_limit_rejection_count=(
                    baseline.candidate_limit_rejection_count + len(native_rejected)
                ),
                first_candidate_limit_rejection_row=(
                    baseline.first_candidate_limit_rejection_row
                    if baseline.first_candidate_limit_rejection_row is not None
                    else (native_rejected[0] if native_rejected else None)
                ),
                max_candidate_limit_rejections_per_step=max(
                    baseline.max_candidate_limit_rejections_per_step,
                    len(native_rejected),
                ),
                diagnostic_pose_rejection_count=(
                    baseline.diagnostic_pose_rejection_count
                    + len(diagnostic_pose_rejected)
                ),
                candidate_exhausted=False,
            )
            self.state = proposed
        return proposed

    def select_query(self, query: object) -> HybridRuntimeState:
        value = np.asarray(query, dtype=np.float64)
        result = self.match(value)
        command = CommandState()
        return self._commit_with_native_limit_retry(
            result.row,
            value,
            search_distance=result.distance,
            live_raw=self._preview_terrain(command),
            live_domain_supported=self._preview_domain_supported(command),
        )

    def step(
        self,
        command: CommandState,
        *,
        dt: float | None = None,
        force_search: bool = False,
    ) -> HybridRuntimeState:
        if not isinstance(command, CommandState):
            raise TypeError("runtime command must be a CommandState")
        elapsed = self.dt if dt is None else _finite_scalar(dt, "runtime dt")
        if elapsed <= 0.0:
            raise ValueError("runtime dt must be positive")
        snapshot = (
            self.state,
            self._root_xy.copy(),
            self._heading,
            self._elapsed_since_search,
            self._last_command,
            self._last_terrain_class,
            self._world_transform,
            self._active_source_root,
        )
        try:
            live = self._preview_terrain(command)
            live_domain_supported = self._preview_domain_supported(command)
            terrain_class = self.terrain.terrain_class(live)
            self._elapsed_since_search += elapsed
            if self.diagnostic_stability:
                preview_support_status = self._terrain_support_status(
                    self._normalize_terrain(live),
                    domain_supported=live_domain_supported,
                )
                if preview_support_status != "SUPPORTED":
                    self._last_command = command
                    self._last_terrain_class = terrain_class
                    return self._hold_outside_support(self.state)
            command_event = self._last_command is None or command != self._last_command
            terrain_event = (
                self._last_terrain_class is not None
                and terrain_class != self._last_terrain_class
            )
            range_index = int(self.row_ranges[self.state.row])
            raw_successor = min(
                self.state.row + 1, int(self.range_stops[range_index]) - 1
            )
            successor_row = self.successor(self.state.row)
            range_terminal = successor_row == self.state.row
            mechanically_unsafe_successor_boundary = (
                self.diagnostic_stability
                and raw_successor != self.state.row
                and self._diagnostic_mechanical_safe_rows is not None
                and not self._diagnostic_mechanical_safe_rows[raw_successor]
            )
            discontinuous_successor_boundary = (
                self.diagnostic_stability
                and raw_successor != self.state.row
                and self._diagnostic_mechanical_discontinuous_successors is not None
                and self._diagnostic_mechanical_discontinuous_successors[self.state.row]
            )
            periodic_search = (
                not self.diagnostic_stability
                and self._elapsed_since_search >= SEARCH_INTERVAL_S
            )
            event_search = bool(force_search) or command_event or terrain_event
            active_boundary_search = (
                self.diagnostic_stability
                and range_terminal
                and command != CommandState()
            )
            search = event_search or periodic_search or active_boundary_search
            if self.diagnostic_stability and command == CommandState() and not search:
                self._last_command = command
                self._last_terrain_class = terrain_class
                return self.state
            query = self._command_query(command, live)
            if discontinuous_successor_boundary:
                boundary_exclusions = (self.state.row, raw_successor)
            elif mechanically_unsafe_successor_boundary:
                boundary_exclusions = (self.state.row,)
            else:
                boundary_exclusions = ()
            if search:
                result = self.match(query, excluded_rows=boundary_exclusions)
                row = result.row
                distance = result.distance
                self._elapsed_since_search = 0.0
            else:
                row = successor_row
                distance = self.state.search_distance
            proposed = self._commit_with_native_limit_retry(
                row,
                query,
                search_distance=distance,
                live_raw=live,
                live_domain_supported=live_domain_supported,
                advance_source=not search,
                command=command,
                base_excluded_rows=boundary_exclusions,
            )
            if proposed.candidate_exhausted:
                (
                    _state,
                    root_xy,
                    self._heading,
                    self._elapsed_since_search,
                    self._last_command,
                    self._last_terrain_class,
                    self._world_transform,
                    self._active_source_root,
                ) = snapshot
                self._root_xy[:] = root_xy
                return proposed
            self._last_command = command
            self._last_terrain_class = terrain_class
            return proposed
        except BaseException:
            (
                self.state,
                root_xy,
                self._heading,
                self._elapsed_since_search,
                self._last_command,
                self._last_terrain_class,
                self._world_transform,
                self._active_source_root,
            ) = snapshot
            self._root_xy[:] = root_xy
            raise

    def reset(self) -> HybridRuntimeState:
        """Reset kinematic placement without rebuilding corpus, tree, or generator."""

        snapshot = (
            self.state,
            self._root_xy.copy(),
            self._heading,
            self._elapsed_since_search,
            self._last_command,
            self._last_terrain_class,
            self._world_transform,
            self._active_source_root,
        )
        try:
            self._elapsed_since_search = SEARCH_INTERVAL_S
            self._last_command = None
            self._last_terrain_class = None
            self._set_world_transform(
                SE2Transform(self._initial_root_xy, self._initial_heading)
            )
            live = self._preview_terrain(CommandState())
            live_domain_supported = self._preview_domain_supported(CommandState())
            reset_row = int(self.searchable_rows[0]) if self.diagnostic_stability else 0
            query = np.array(self.features[reset_row], dtype=np.float64, copy=True)
            query[TERRAIN_INDICES] = self._normalize_terrain(live)
            counters = (
                self.state.decode_count,
                self.state.fallback_count,
                self.state.joint_clamp_count,
            )
            reset_state = self._commit_row(
                reset_row,
                query,
                search_distance=0.0,
                count_decode=False,
                live_raw=live,
                live_domain_supported=live_domain_supported,
            )
            self.state = replace(
                reset_state,
                decode_count=counters[0],
                fallback_count=counters[1],
                joint_clamp_count=counters[2],
            )
            return self.state
        except BaseException:
            (
                self.state,
                root_xy,
                self._heading,
                self._elapsed_since_search,
                self._last_command,
                self._last_terrain_class,
                self._world_transform,
                self._active_source_root,
            ) = snapshot
            self._root_xy[:] = root_xy
            raise


def _scripted_command(frame: int, frames: int) -> CommandState:
    phase = frame / max(frames, 1)
    if phase < 0.12:
        return CommandState()
    if phase < 0.36:
        return CommandState(1.0, 0.0)
    if phase < 0.52:
        return CommandState(1.0, 0.65)
    if phase < 0.68:
        return CommandState(1.0, -0.65)
    if phase < 0.78:
        return CommandState()
    if phase < 0.9:
        return CommandState(-0.4, 0.0)
    return CommandState(0.8, 0.0)


def run_headless_smoke(
    matcher: HybridMatcher, *, frames: int = 1_000
) -> dict[str, object]:
    """Exercise the corpus-rate matcher without importing MuJoCo or windows."""

    if not isinstance(matcher, HybridMatcher):
        raise TypeError("headless smoke requires a HybridMatcher")
    if type(frames) is not int or frames < 1:
        raise ValueError("headless smoke frames must be a positive integer")
    selected_ranges: set[int] = set()
    selected_families: set[str] = set()
    classes: set[str] = set()
    terrain_rows: list[np.ndarray] = []
    parity_samples = 0
    parity_failures = 0
    retrieval_samples = 0
    retrieval_failures = 0
    retrieval_exact_feature_matches = 0
    max_joint_clamp_magnitude = 0.0
    initial_decode_count = matcher.state.decode_count
    initial_fallback_count = matcher.state.fallback_count
    initial_joint_clamp_count = matcher.state.joint_clamp_count
    initial_exhaustion_count = matcher.state.candidate_exhaustion_count

    row_families = matcher.family_ids[matcher.row_ranges[matcher.searchable_rows]]
    for family in np.unique(row_families):
        family_rows = matcher.searchable_rows[row_families == family]
        audit_row = int(family_rows[len(family_rows) // 2])
        audit_query = np.asarray(matcher.features[audit_row], dtype=np.float64)
        tree = matcher.match(audit_query)
        brute = matcher.brute_force_match(audit_query)
        retrieval_samples += 1
        if tree.row != brute.row or not math.isclose(
            tree.distance, brute.distance, rel_tol=0.0, abs_tol=1e-12
        ):
            retrieval_failures += 1
        retrieval_exact_feature_matches += int(
            np.array_equal(matcher.features[tree.row], matcher.features[audit_row])
        )

    parity_frames = {0, max(0, frames // 2)}
    for frame in range(frames):
        state = matcher.step(_scripted_command(frame, frames), dt=matcher.dt)
        if not np.isfinite(state.qpos).all():
            raise RuntimeError(f"non-finite qpos at smoke frame {frame}")
        range_index = int(matcher.row_ranges[state.row])
        if range_index != state.range_index:
            raise RuntimeError(f"range crossing at smoke frame {frame}")
        selected_ranges.add(range_index)
        selected_families.add(state.family)
        classes.add(state.terrain_class)
        terrain_rows.append(np.asarray(state.terrain_features, dtype=np.float64))
        max_joint_clamp_magnitude = max(
            max_joint_clamp_magnitude, state.max_joint_clamp_magnitude
        )
        if frame in parity_frames:
            tree = matcher.match(state.query)
            brute = matcher.brute_force_match(state.query)
            if tree.row != brute.row or not math.isclose(
                tree.distance, brute.distance, rel_tol=0.0, abs_tol=1e-12
            ):
                parity_failures += 1
            parity_samples += 1
    terrain_variance = float(np.var(np.stack(terrain_rows), axis=0).max())
    learned_decode_count = matcher.state.decode_count - initial_decode_count
    fallback_count = matcher.state.fallback_count - initial_fallback_count
    joint_clamp_count = matcher.state.joint_clamp_count - initial_joint_clamp_count
    candidate_exhaustion_count = (
        matcher.state.candidate_exhaustion_count - initial_exhaustion_count
    )
    canonical_fallback_count = max(0, fallback_count - joint_clamp_count)
    acceptance_failures: list[str] = []
    gates = (
        (frames >= 1_000, "minimum-1000-frames"),
        (learned_decode_count > 0, "learned-decode-required"),
        (fallback_count == 0, "zero-fallback-required"),
        (joint_clamp_count == 0, "zero-native-limit-clamp-required"),
        (candidate_exhaustion_count == 0, "zero-candidate-exhaustion-required"),
        (terrain_variance > 0.0, "nonflat-terrain-variance-required"),
        (parity_samples > 0 and parity_failures == 0, "tree-brute-parity"),
        (
            retrieval_samples > 0 and retrieval_failures == 0,
            "manifest-row-retrieval-audit",
        ),
        (len(selected_ranges) >= 2, "range-diversity-required"),
        (matcher.search_acceptance_eligible, "full-search-required"),
    )
    for passed, name in gates:
        if not passed:
            acceptance_failures.append(name)
    diagnostic_checks_passed = not acceptance_failures
    acceptance_failures.append("mujoco-evidence-required")
    return {
        "schema": "hybrid-terrain-lmm-headless-smoke/v1",
        "accepted": False,
        "acceptance_eligible": False,
        "evidence_status": "runtime-only-diagnostic-not-acceptance",
        "diagnostic_checks_passed": diagnostic_checks_passed,
        "frames": frames,
        "selected_ranges": sorted(selected_ranges),
        "selected_families": sorted(selected_families),
        "terrain_classes": sorted(classes),
        "acceptance_failures": acceptance_failures,
        "learned_decode_count": learned_decode_count,
        "fallback_count": fallback_count,
        "canonical_fallback_count": canonical_fallback_count,
        "clamped_fallback_count": joint_clamp_count,
        "joint_clamp_count": joint_clamp_count,
        "max_joint_clamp_magnitude": max_joint_clamp_magnitude,
        "candidate_limit_rejection_count": (
            matcher.state.candidate_limit_rejection_count
        ),
        "first_candidate_limit_rejection_row": (
            matcher.state.first_candidate_limit_rejection_row
        ),
        "max_candidate_limit_rejections_per_step": (
            matcher.state.max_candidate_limit_rejections_per_step
        ),
        "candidate_exhaustion_count": candidate_exhaustion_count,
        "candidate_exhausted": matcher.state.candidate_exhausted,
        "candidate_retry_budget": matcher.candidate_retry_budget,
        "terrain_variance": terrain_variance,
        "tree_brute_parity_samples": parity_samples,
        "tree_brute_parity_failures": parity_failures,
        "retrieval_audit_samples": retrieval_samples,
        "retrieval_audit_failures": retrieval_failures,
        "retrieval_exact_feature_matches": retrieval_exact_feature_matches,
        "range_diversity_count": len(selected_ranges),
        "family_diversity_count": len(selected_families),
        "search_scope": matcher.search_scope,
        "search_acceptance_eligible": matcher.search_acceptance_eligible,
        "total_safe_row_count": matcher.total_searchable_row_count,
        "searched_row_count": len(matcher.searchable_rows),
        "searchable_row_count": len(matcher.searchable_rows),
        "searchable_family_counts": dict(matcher.searchable_family_counts),
        "total_searchable_family_counts": dict(matcher.total_searchable_family_counts),
        "search_view_sha256": matcher.search_view_sha256,
        "transition_penalty": matcher.transition_penalty,
        "fps": matcher.fps,
        "dt": matcher.dt,
        "horizons": list(matcher.horizons),
        "walking_speed_p95_mps": matcher.walking_speed_p95_mps,
        "root_motion_source": "canonical-simulation-se2",
    }


__all__ = (
    "CommandState",
    "HybridMatcher",
    "HybridRuntimeState",
    "SE2Transform",
    "SearchResult",
    "TerrainAuthority",
    "compose_root_delta",
    "run_headless_smoke",
)
