"""Robot-relative terrain perception and finite platform test geometry.

The analytic scene in this module is a simulator sensor backend, not an input
contract.  It rasterizes the known test geometry into the same ego height map
that a real depth/point-cloud frontend would provide.  Matching consumes only
the resulting robot-relative observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .joints import ContractError
from .terrain_catalog import (
    JUSTIN_STAIR_RISE_M,
    JUSTIN_STAIR_RUN_M,
    JUSTIN_STAIR_TREADS,
    JUSTIN_STAIR_WIDTH_M,
)
from .terrain_motion import (
    TERRAIN_COLS,
    TERRAIN_FORWARD_RANGE_M,
    TERRAIN_LATERAL_RANGE_M,
    TERRAIN_ROWS,
    StairGeometry,
    detect_stair_geometry,
    terrain_observation_from_height_map,
)


def _finite(
    value: object, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{name} must have shape {shape} and be finite")
    return np.ascontiguousarray(array)


def _rotation(yaw: float) -> np.ndarray:
    value = float(yaw)
    if not math.isfinite(value):
        raise ContractError("yaw must be finite")
    c, s = math.cos(value), math.sin(value)
    return np.asarray(((c, -s), (s, c)), dtype=np.float64)


@dataclass(frozen=True)
class StairPlacement:
    """One canonical staircase placed in a renderer/world coordinate system."""

    origin_world_xyz: np.ndarray
    ascent_yaw_world: float
    name: str = "stair"

    def __post_init__(self) -> None:
        origin = _finite(
            self.origin_world_xyz, (3,), "origin_world_xyz"
        ).astype(np.float32)
        yaw = float(self.ascent_yaw_world)
        if not math.isfinite(yaw):
            raise ContractError("ascent_yaw_world must be finite")
        name = str(self.name)
        if not name:
            raise ContractError("stair name must be non-empty")
        object.__setattr__(self, "origin_world_xyz", origin)
        object.__setattr__(
            self, "ascent_yaw_world", math.remainder(yaw, 2.0 * math.pi)
        )
        object.__setattr__(self, "name", name)

    def stair_to_world_xyz(self, position_stair: np.ndarray) -> np.ndarray:
        points = np.asarray(position_stair, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ContractError(
                "position_stair must end in three finite coordinates"
            )
        result = np.asarray(points).copy()
        result[..., :2] = (
            points[..., :2] @ _rotation(self.ascent_yaw_world).T
        )
        result[..., :2] += self.origin_world_xyz[:2]
        result[..., 2] += float(self.origin_world_xyz[2])
        return np.ascontiguousarray(result, dtype=np.float32)

    def world_to_stair_xyz(self, position_world: np.ndarray) -> np.ndarray:
        points = np.asarray(position_world, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ContractError(
                "position_world must end in three finite coordinates"
            )
        delta = points - self.origin_world_xyz
        result = np.asarray(delta)
        result[..., :2] = (
            delta[..., :2] @ _rotation(-self.ascent_yaw_world).T
        )
        return np.ascontiguousarray(result, dtype=np.float32)


@dataclass(frozen=True)
class ObservedStair:
    """A stair fitted only in the robot's current causal sensor frame."""

    geometry: StairGeometry
    base_height_robot_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, StairGeometry):
            raise ContractError("geometry must be StairGeometry")
        height = float(self.base_height_robot_m)
        if not math.isfinite(height):
            raise ContractError("base_height_robot_m must be finite")
        object.__setattr__(self, "base_height_robot_m", height)

    def placement_from_robot(
        self,
        *,
        root_position_world: np.ndarray,
        root_yaw_world: float,
        name: str = "observed_stair",
    ) -> StairPlacement:
        root = _finite(
            root_position_world, (3,), "root_position_world"
        )
        yaw = float(root_yaw_world)
        origin_world_xy = (
            root[:2]
            + _rotation(yaw) @ self.geometry.origin_robot_xy
        )
        return StairPlacement(
            origin_world_xyz=np.asarray(
                (
                    origin_world_xy[0],
                    origin_world_xy[1],
                    root[2] + self.base_height_robot_m,
                ),
                dtype=np.float32,
            ),
            ascent_yaw_world=yaw + self.geometry.ascent_yaw_robot,
            name=name,
        )


class CausalStairObservationFilter:
    """Robustly fuse a short history of ego-frame stair detections.

    The stored coordinate frame is only the kinematic session's internal
    relative frame.  No external odometry, map, or scene-global root position
    is consumed.  Each update composes the new robot-local detection through
    the session's own relative pose, takes a coordinate-wise median over a
    bounded causal window, and immediately re-expresses the result in the
    current robot frame.
    """

    def __init__(
        self,
        *,
        history_frames: int = 24,
        maximum_association_distance_m: float = 0.75,
        maximum_association_yaw_rad: float = math.radians(30.0),
        maximum_missing_frames: int = 6,
    ) -> None:
        self.history_frames = int(history_frames)
        self.maximum_association_distance_m = float(
            maximum_association_distance_m
        )
        self.maximum_association_yaw_rad = float(
            maximum_association_yaw_rad
        )
        self.maximum_missing_frames = int(maximum_missing_frames)
        if (
            self.history_frames < 3
            or not math.isfinite(self.maximum_association_distance_m)
            or self.maximum_association_distance_m <= 0.0
            or not math.isfinite(self.maximum_association_yaw_rad)
            or not 0.0 < self.maximum_association_yaw_rad < math.pi
            or self.maximum_missing_frames < 0
        ):
            raise ContractError("causal stair filter parameters are invalid")
        self._measurements: list[
            tuple[StairPlacement, StairGeometry]
        ] = []
        self._missing_frames = 0

    def reset(self) -> None:
        self._measurements.clear()
        self._missing_frames = 0

    @staticmethod
    def _median_yaw(values: np.ndarray) -> float:
        reference = float(values[0])
        unwrapped = reference + np.asarray(
            [
                math.remainder(float(value) - reference, 2.0 * math.pi)
                for value in values
            ],
            dtype=np.float64,
        )
        return math.remainder(
            float(np.median(unwrapped)), 2.0 * math.pi
        )

    def _associated(self, placement: StairPlacement) -> bool:
        if not self._measurements:
            return True
        origins = np.asarray(
            [
                measurement.origin_world_xyz
                for measurement, _geometry in self._measurements
            ],
            dtype=np.float64,
        )
        yaws = np.asarray(
            [
                measurement.ascent_yaw_world
                for measurement, _geometry in self._measurements
            ],
            dtype=np.float64,
        )
        origin = np.median(origins, axis=0)
        yaw = self._median_yaw(yaws)
        return (
            float(
                np.linalg.norm(
                    placement.origin_world_xyz[:2] - origin[:2]
                )
            )
            <= self.maximum_association_distance_m
            and abs(
                math.remainder(
                    placement.ascent_yaw_world - yaw,
                    2.0 * math.pi,
                )
            )
            <= self.maximum_association_yaw_rad
        )

    def update(
        self,
        stair: ObservedStair | None,
        *,
        root_position_world: np.ndarray,
        root_yaw_world: float,
    ) -> ObservedStair | None:
        root = _finite(
            root_position_world, (3,), "root_position_world"
        )
        yaw = float(root_yaw_world)
        if not math.isfinite(yaw):
            raise ContractError("root_yaw_world must be finite")
        if stair is None:
            self._missing_frames += 1
            if self._missing_frames > self.maximum_missing_frames:
                self.reset()
            return None
        if not isinstance(stair, ObservedStair):
            raise ContractError("stair must be an ObservedStair or None")
        self._missing_frames = 0
        placement = stair.placement_from_robot(
            root_position_world=root,
            root_yaw_world=yaw,
        )
        if not self._associated(placement):
            self.reset()
        self._measurements.append((placement, stair.geometry))
        if len(self._measurements) > self.history_frames:
            del self._measurements[: -self.history_frames]

        origins = np.asarray(
            [
                measurement.origin_world_xyz
                for measurement, _geometry in self._measurements
            ],
            dtype=np.float64,
        )
        yaws = np.asarray(
            [
                measurement.ascent_yaw_world
                for measurement, _geometry in self._measurements
            ],
            dtype=np.float64,
        )
        geometries = [
            geometry for _measurement, geometry in self._measurements
        ]
        filtered_origin = np.median(origins, axis=0)
        filtered_yaw = self._median_yaw(yaws)
        rotation = _rotation(yaw)
        origin_robot = (
            filtered_origin[:2] - root[:2]
        ) @ rotation
        return ObservedStair(
            StairGeometry(
                float(np.median([value.run_m for value in geometries])),
                float(np.median([value.rise_m for value in geometries])),
                float(np.median([value.width_m for value in geometries])),
                int(
                    round(
                        float(
                            np.median(
                                [
                                    value.tread_count
                                    for value in geometries
                                ]
                            )
                        )
                    )
                ),
                np.ascontiguousarray(origin_robot, dtype=np.float32),
                math.remainder(filtered_yaw - yaw, 2.0 * math.pi),
                float(
                    np.median([value.confidence for value in geometries])
                ),
            ),
            base_height_robot_m=float(filtered_origin[2] - root[2]),
        )


def _nearest_height(
    height_map: np.ndarray, forward_m: float, lateral_m: float
) -> float | None:
    forward = np.linspace(
        *TERRAIN_FORWARD_RANGE_M, TERRAIN_ROWS, dtype=np.float64
    )
    lateral = np.linspace(
        *TERRAIN_LATERAL_RANGE_M, TERRAIN_COLS, dtype=np.float64
    )
    row = int(np.argmin(np.abs(forward - float(forward_m))))
    column = int(np.argmin(np.abs(lateral - float(lateral_m))))
    value = float(height_map[row, column])
    return value if math.isfinite(value) else None


def observe_stair_from_height_map(
    height_map_robot_m: np.ndarray,
) -> ObservedStair | None:
    """Fit a stair and its base elevation from a robot-centred height map."""

    height_map = np.asarray(height_map_robot_m, dtype=np.float32)
    observation = terrain_observation_from_height_map(height_map)
    geometry = detect_stair_geometry(observation)
    if geometry is None:
        return None
    ascent = np.asarray(
        (
            math.cos(geometry.ascent_yaw_robot),
            math.sin(geometry.ascent_yaw_robot),
        ),
        dtype=np.float64,
    )
    side = np.asarray((-ascent[1], ascent[0]), dtype=np.float64)
    samples: list[float] = []
    for behind in (0.06, 0.12, 0.18):
        for lateral in (-0.2, 0.0, 0.2):
            point = (
                geometry.origin_robot_xy
                - ascent * behind
                + side * lateral
            )
            value = _nearest_height(height_map, point[0], point[1])
            if value is not None:
                samples.append(value)
    if not samples:
        return None
    return ObservedStair(
        geometry=geometry,
        base_height_robot_m=float(np.median(samples)),
    )


def path_has_unsafe_height_change(
    height_map_robot_m: np.ndarray,
    future_root_xy: np.ndarray,
    *,
    stair: ObservedStair | None,
    maximum_flat_drop_m: float = 0.12,
    corridor_margin_m: float = 0.12,
) -> bool:
    """Reject flat previews crossing a rise/drop away from stair mode."""

    height_map = np.asarray(height_map_robot_m, dtype=np.float32)
    if height_map.shape != (TERRAIN_ROWS, TERRAIN_COLS):
        raise ContractError(
            f"height map must have shape {(TERRAIN_ROWS, TERRAIN_COLS)}"
        )
    path = np.asarray(future_root_xy, dtype=np.float64)
    if (
        path.ndim != 2
        or path.shape[1:] != (2,)
        or len(path) == 0
        or not np.isfinite(path).all()
    ):
        raise ContractError("future_root_xy must have shape [N,2]")
    limit = float(maximum_flat_drop_m)
    margin = float(corridor_margin_m)
    if not math.isfinite(limit) or limit <= 0.0 or margin < 0.0:
        raise ContractError("drop and corridor thresholds are invalid")
    current_height = _nearest_height(height_map, 0.0, 0.0)
    if current_height is None:
        return True
    geometry = None if stair is None else stair.geometry
    if geometry is not None:
        ascent = np.asarray(
            (
                math.cos(geometry.ascent_yaw_robot),
                math.sin(geometry.ascent_yaw_robot),
            ),
            dtype=np.float64,
        )
        side = np.asarray((-ascent[1], ascent[0]), dtype=np.float64)
        origin = np.asarray(geometry.origin_robot_xy, dtype=np.float64)
        length = geometry.run_m * geometry.tread_count

    # The height map has no sample exactly at the robot origin and a short
    # executed path can cross a riser between adjacent raster cells.  Sweep a
    # narrow, directionally look-ahead corridor instead of checking only the
    # sparse trajectory knots.  Directional probing is important: it sees a
    # support edge before crossing it without trapping a robot that is backing
    # away from the edge.
    forward_spacing = (
        TERRAIN_FORWARD_RANGE_M[1] - TERRAIN_FORWARD_RANGE_M[0]
    ) / (TERRAIN_ROWS - 1)
    lateral_spacing = (
        TERRAIN_LATERAL_RANGE_M[1] - TERRAIN_LATERAL_RANGE_M[0]
    ) / (TERRAIN_COLS - 1)
    lookahead = max(forward_spacing, lateral_spacing)
    sweep_spacing = 0.5 * min(forward_spacing, lateral_spacing)
    half_width = lateral_spacing
    previous = np.zeros(2, dtype=np.float64)
    for endpoint in path:
        segment = endpoint - previous
        distance = float(np.linalg.norm(segment))
        if distance <= 1.0e-9:
            previous = endpoint
            continue
        direction = segment / distance
        lateral_direction = np.asarray(
            (-direction[1], direction[0]), dtype=np.float64
        )
        sample_count = max(1, int(math.ceil(distance / sweep_spacing)))
        for fraction in np.linspace(
            1.0 / sample_count, 1.0, sample_count
        ):
            centre = previous + segment * fraction + direction * lookahead
            for lateral_offset in (-half_width, 0.0, half_width):
                point = centre + lateral_direction * lateral_offset
                height = _nearest_height(
                    height_map, point[0], point[1]
                )
                if height is None:
                    return True
                if abs(current_height - height) <= limit:
                    continue
                in_stair_corridor = False
                if geometry is not None:
                    delta = point - origin
                    along = float(delta @ ascent)
                    lateral = abs(float(delta @ side))
                    in_stair_corridor = (
                        -margin <= along <= length + margin
                        and lateral
                        <= geometry.width_m * 0.5 + margin
                    )
                if not in_stair_corridor:
                    return True
        previous = endpoint
    return False


def path_has_unsafe_drop(
    height_map_robot_m: np.ndarray,
    future_root_xy: np.ndarray,
    *,
    stair: ObservedStair | None,
    maximum_flat_drop_m: float = 0.12,
    corridor_margin_m: float = 0.12,
) -> bool:
    """Backward-compatible name for the bidirectional support-edge guard."""

    return path_has_unsafe_height_change(
        height_map_robot_m,
        future_root_xy,
        stair=stair,
        maximum_flat_drop_m=maximum_flat_drop_m,
        corridor_margin_m=corridor_margin_m,
    )


def flat_motion_preview_is_safe(
    height_map_robot_m: np.ndarray,
    *,
    current_root_position_world: np.ndarray,
    current_root_yaw_world: float,
    dense_root_position_world: np.ndarray,
    dense_body_position_world: np.ndarray,
    foot_center_to_sole_m: float = 0.035,
    maximum_sole_penetration_m: float = 0.005,
    check_nominal_soles: bool = True,
) -> bool:
    """Check the flat matcher's selected motion, rather than its command.

    Motion matching is approximate: the selected clip can travel differently
    from the command trajectory used to query it.  This guard consumes the
    selected dense root/body preview in the current robot-centred sensor frame
    and rejects it before matcher state is committed.  Ordinarily the archive
    ankle-roll point uses a conservative nominal sole offset.  A caller that
    separately gates the next published pose with exact articulated geometry
    may disable only that nominal proxy; the dense support-edge sweep remains.
    """

    height_map = np.asarray(height_map_robot_m, dtype=np.float32)
    if height_map.shape != (TERRAIN_ROWS, TERRAIN_COLS):
        raise ContractError(
            f"height map must have shape {(TERRAIN_ROWS, TERRAIN_COLS)}"
        )
    current_root = _finite(
        current_root_position_world,
        (3,),
        "current_root_position_world",
    )
    yaw = float(current_root_yaw_world)
    roots = np.asarray(dense_root_position_world, dtype=np.float64)
    bodies = np.asarray(dense_body_position_world, dtype=np.float64)
    sole_offset = float(foot_center_to_sole_m)
    penetration = float(maximum_sole_penetration_m)
    if (
        not math.isfinite(yaw)
        or roots.ndim != 2
        or roots.shape[1:] != (3,)
        or len(roots) == 0
        or bodies.shape != (len(roots), 30, 3)
        or not np.isfinite(roots).all()
        or not np.isfinite(bodies).all()
        or not math.isfinite(sole_offset)
        or sole_offset < 0.0
        or not math.isfinite(penetration)
        or penetration < 0.0
        or type(check_nominal_soles) is not bool
    ):
        raise ContractError("flat selected motion preview is invalid")

    robot_from_world = _rotation(-yaw)
    root_path_robot = (
        roots[:, :2] - current_root[:2]
    ) @ robot_from_world.T
    if path_has_unsafe_height_change(
        height_map,
        root_path_robot,
        stair=None,
    ):
        return False
    if not check_nominal_soles:
        return True

    feet = bodies[:, (18, 19)]
    feet_xy_robot = (
        feet[..., :2] - current_root[:2]
    ) @ robot_from_world.T
    feet_height_robot = feet[..., 2] - current_root[2]
    for frame in range(len(feet)):
        for foot in range(2):
            terrain_height = _nearest_height(
                height_map,
                feet_xy_robot[frame, foot, 0],
                feet_xy_robot[frame, foot, 1],
            )
            if terrain_height is None:
                return False
            clearance = (
                float(feet_height_robot[frame, foot])
                - sole_offset
                - terrain_height
            )
            if clearance < -penetration:
                return False
    return True


class StairIntentLatch:
    """Require a sustained command corridor before stair mode may arm."""

    def __init__(
        self,
        *,
        required_frames: int = 10,
        minimum_progress_m: float = 0.05,
        lateral_margin_m: float = 0.10,
    ) -> None:
        if type(required_frames) is not int or required_frames <= 0:
            raise ContractError("required_frames must be a positive integer")
        if minimum_progress_m <= 0.0 or lateral_margin_m < 0.0:
            raise ContractError("intent latch distances are invalid")
        self.required_frames = required_frames
        self.minimum_progress_m = float(minimum_progress_m)
        self.lateral_margin_m = float(lateral_margin_m)
        self.aligned_frames = 0

    def reset(self) -> None:
        self.aligned_frames = 0

    def update(
        self, stair: ObservedStair | None, future_root_xy: np.ndarray
    ) -> bool:
        path = np.asarray(future_root_xy, dtype=np.float64)
        if (
            stair is None
            or path.ndim != 2
            or path.shape[1:] != (2,)
            or len(path) == 0
            or not np.isfinite(path).all()
        ):
            self.reset()
            return False
        geometry = stair.geometry
        ascent = np.asarray(
            (
                math.cos(geometry.ascent_yaw_robot),
                math.sin(geometry.ascent_yaw_robot),
            ),
            dtype=np.float64,
        )
        side = np.asarray((-ascent[1], ascent[0]), dtype=np.float64)
        origin = np.asarray(geometry.origin_robot_xy, dtype=np.float64)
        start_along = float((-origin) @ ascent)
        end_delta = path[-1] - origin
        end_along = float(end_delta @ ascent)
        end_lateral = abs(float(end_delta @ side))
        stair_length = geometry.run_m * geometry.tread_count
        if start_along < 0.0:
            signed_progress = end_along - start_along
        elif start_along > stair_length:
            signed_progress = start_along - end_along
        else:
            signed_progress = abs(end_along - start_along)
        aligned = (
            signed_progress >= self.minimum_progress_m
            and end_lateral
            <= geometry.width_m * 0.5 + self.lateral_margin_m
        )
        self.aligned_frames = self.aligned_frames + 1 if aligned else 0
        return self.aligned_frames >= self.required_frames


def shape_terrain_approach_velocity(
    local_velocity_xy: np.ndarray,
    stair: ObservedStair | None,
    *,
    maximum_approach_speed_mps: float = 0.34,
    minimum_alignment_cosine: float = math.cos(math.radians(45.0)),
    ascent_alignment_standoff_m: float = 0.30,
    ascent_lateral_tolerance_m: float = 0.10,
    descent_standoff_margin_m: float = 0.08,
    descent_braking_gain_s_inv: float = 1.2,
    lateral_centering_gain_s_inv: float = 1.0,
    maximum_lateral_correction_mps: float = 0.18,
    maximum_shaping_distance_m: float = 1.0,
) -> np.ndarray:
    """Cap only commands aimed into an observed stair corridor.

    The cap keeps the flat approach in the speed regime for which the current
    Justin catalog has demonstrated a contact-compatible splice. It is a
    robot-frame terrain response: no world pose or global map is consulted.
    """

    velocity = _finite(
        local_velocity_xy, (2,), "local_velocity_xy"
    )
    maximum_speed = float(maximum_approach_speed_mps)
    alignment = float(minimum_alignment_cosine)
    ascent_standoff = float(ascent_alignment_standoff_m)
    lateral_tolerance = float(ascent_lateral_tolerance_m)
    standoff_margin = float(descent_standoff_margin_m)
    braking_gain = float(descent_braking_gain_s_inv)
    centering_gain = float(lateral_centering_gain_s_inv)
    maximum_lateral = float(maximum_lateral_correction_mps)
    maximum_distance = float(maximum_shaping_distance_m)
    if (
        not math.isfinite(maximum_speed)
        or maximum_speed <= 0.0
        or not math.isfinite(alignment)
        or not -1.0 <= alignment <= 1.0
        or not math.isfinite(ascent_standoff)
        or ascent_standoff <= 0.0
        or not math.isfinite(lateral_tolerance)
        or lateral_tolerance < 0.0
        or not math.isfinite(standoff_margin)
        or standoff_margin < 0.0
        or not math.isfinite(braking_gain)
        or braking_gain <= 0.0
        or not math.isfinite(centering_gain)
        or centering_gain < 0.0
        or not math.isfinite(maximum_lateral)
        or maximum_lateral < 0.0
        or not math.isfinite(maximum_distance)
        or maximum_distance <= 0.0
    ):
        raise ContractError("terrain approach shaping values are invalid")
    speed = float(np.linalg.norm(velocity))
    if stair is None or speed <= 1.0e-8:
        return velocity.astype(np.float64, copy=True)
    origin = np.asarray(
        stair.geometry.origin_robot_xy, dtype=np.float64
    )
    ascent = np.asarray(
        (
            math.cos(stair.geometry.ascent_yaw_robot),
            math.sin(stair.geometry.ascent_yaw_robot),
        ),
        dtype=np.float64,
    )
    side = np.asarray((-ascent[1], ascent[0]), dtype=np.float64)
    origin_along = float(origin @ ascent)
    origin_lateral = float(origin @ side)
    root_along = -origin_along
    root_lateral = -origin_lateral
    velocity_along = float(velocity @ ascent)
    velocity_lateral = float(velocity @ side)
    stair_length = (
        stair.geometry.run_m * stair.geometry.tread_count
    )
    approaching_from_below = (
        root_along < 0.0
        and velocity_along > 0.0
        and velocity_along / speed >= alignment
    )
    approaching_from_above = (
        root_along > stair_length
        and velocity_along < 0.0
        and -velocity_along / speed >= alignment
    )
    if not (approaching_from_below or approaching_from_above):
        return velocity.astype(np.float64, copy=True)
    distance_to_stairs = (
        -root_along
        if approaching_from_below
        else root_along - stair_length
    )
    if distance_to_stairs > maximum_distance:
        return velocity.astype(np.float64, copy=True)

    velocity_lateral += float(
        np.clip(
            -centering_gain * root_lateral,
            -maximum_lateral,
            maximum_lateral,
        )
    )
    if (
        approaching_from_below
        and abs(root_lateral) > lateral_tolerance
    ):
        # Steer through a robot-relative staging point instead of stopping at
        # a hard alignment gate.  Moving the point farther back as lateral
        # error grows produces a safe approach arc and, crucially, cannot
        # deadlock at a non-zero lateral offset.
        stage_along = -ascent_standoff - min(
            0.35, abs(root_lateral)
        )
        to_stage_along = stage_along - root_along
        to_stage_lateral = -root_lateral
        to_stage = np.asarray(
            (to_stage_along, to_stage_lateral), dtype=np.float64
        )
        stage_distance = float(np.linalg.norm(to_stage))
        if stage_distance > 1.0e-8:
            stage_speed = min(speed, maximum_speed)
            velocity_along = stage_speed * to_stage_along / stage_distance
            velocity_lateral = float(
                np.clip(
                    stage_speed * to_stage_lateral / stage_distance,
                    -maximum_lateral,
                    maximum_lateral,
                )
            )
    elif approaching_from_above:
        gate_along = stair_length + standoff_margin
        remaining = max(0.0, root_along - gate_along)
        velocity_along = max(
            velocity_along, -braking_gain * remaining
        )
    shaped = velocity_along * ascent + velocity_lateral * side
    shaped_speed = float(np.linalg.norm(shaped))
    if shaped_speed <= maximum_speed:
        return np.ascontiguousarray(shaped)
    return np.ascontiguousarray(
        shaped * (maximum_speed / shaped_speed), dtype=np.float64
    )


def shape_stair_handoff_velocity(
    requested_local_velocity_xy: np.ndarray,
    alignment_local_velocity_xy: np.ndarray,
    *,
    handoff_active: bool,
    phase_staging_active: bool = False,
    alignment_deadband_mps: float = 0.008,
    maximum_alignment_speed_mps: float = 0.16,
    maximum_phase_staging_speed_mps: float = 0.12,
) -> np.ndarray:
    """Let near-stair alignment replace, rather than fight, operator travel.

    The ordinary approach command remains untouched until the terrain matcher
    is armed.  Once armed, a non-trivial support-foot correction temporarily
    becomes the flat motion-matching target.  The command ball still filters
    this target, so the reversal is smooth; using it as an additive nudge
    cannot work at a riser because the larger operator command keeps asking
    the flat matcher to cross the obstacle.
    """

    requested = _finite(
        requested_local_velocity_xy,
        (2,),
        "requested_local_velocity_xy",
    )
    alignment = _finite(
        alignment_local_velocity_xy,
        (2,),
        "alignment_local_velocity_xy",
    )
    deadband = float(alignment_deadband_mps)
    maximum = float(maximum_alignment_speed_mps)
    phase_maximum = float(maximum_phase_staging_speed_mps)
    if (
        type(handoff_active) is not bool
        or type(phase_staging_active) is not bool
        or not math.isfinite(deadband)
        or deadband < 0.0
        or not math.isfinite(maximum)
        or maximum <= 0.0
        or not math.isfinite(phase_maximum)
        or phase_maximum <= 0.0
    ):
        raise ContractError("stair handoff shaping values are invalid")
    magnitude = float(np.linalg.norm(alignment))
    if not handoff_active:
        return requested.astype(np.float64, copy=True)
    if magnitude <= deadband:
        if not phase_staging_active:
            return requested.astype(np.float64, copy=True)
        # Keep advancing the selected gait phase instead of switching to a
        # standing motion while waiting for the paired stair support phase.
        # The caller's dense flat-preview collision oracle still rejects any
        # proposal that would actually cross the riser.
        requested_speed = float(np.linalg.norm(requested))
        if requested_speed <= phase_maximum:
            return requested.astype(np.float64, copy=True)
        return np.ascontiguousarray(
            requested * (phase_maximum / requested_speed),
            dtype=np.float64,
        )
    if magnitude <= maximum:
        return alignment.astype(np.float64, copy=True)
    return np.ascontiguousarray(
        alignment * (maximum / magnitude), dtype=np.float64
    )


def shape_terrain_approach_facing_yaw(
    *,
    current_yaw_world: float,
    requested_facing_yaw_world: float,
    local_velocity_xy: np.ndarray,
    stair: ObservedStair | None,
    facing_override_active: bool,
    minimum_alignment_cosine: float = math.cos(math.radians(45.0)),
    maximum_assist_distance_m: float = 1.0,
) -> float:
    """Face along an intended stair traversal when the facing stick is idle.

    All geometry is robot-relative.  Explicit operator facing input always
    wins, and merely observing a stair does not turn the robot unless the
    translation command is actually aimed through its corridor.
    """

    current_yaw = float(current_yaw_world)
    requested_yaw = float(requested_facing_yaw_world)
    if not math.isfinite(current_yaw) or not math.isfinite(requested_yaw):
        raise ContractError("terrain approach facing yaw must be finite")
    if facing_override_active or stair is None:
        return math.remainder(requested_yaw, 2.0 * math.pi)
    velocity = _finite(
        local_velocity_xy, (2,), "local_velocity_xy"
    )
    speed = float(np.linalg.norm(velocity))
    if speed <= 1.0e-8:
        return math.remainder(requested_yaw, 2.0 * math.pi)
    geometry = stair.geometry
    origin = np.asarray(geometry.origin_robot_xy, dtype=np.float64)
    ascent = np.asarray(
        (
            math.cos(geometry.ascent_yaw_robot),
            math.sin(geometry.ascent_yaw_robot),
        ),
        dtype=np.float64,
    )
    root_along = -float(origin @ ascent)
    stair_length = geometry.run_m * geometry.tread_count
    velocity_along = float(velocity @ ascent)
    alignment = float(minimum_alignment_cosine)
    maximum_distance = float(maximum_assist_distance_m)
    if (
        not math.isfinite(alignment)
        or not -1.0 <= alignment <= 1.0
        or not math.isfinite(maximum_distance)
        or maximum_distance <= 0.0
    ):
        raise ContractError("terrain facing assistance values are invalid")
    if (
        root_along < 0.0
        and -root_along <= maximum_distance
        and velocity_along / speed >= alignment
    ):
        target_yaw = current_yaw + geometry.ascent_yaw_robot
        return math.remainder(target_yaw, 2.0 * math.pi)
    if (
        root_along > stair_length
        and root_along - stair_length <= maximum_distance
        and -velocity_along / speed >= alignment
    ):
        target_yaw = (
            current_yaw + geometry.ascent_yaw_robot + math.pi
        )
        return math.remainder(target_yaw, 2.0 * math.pi)
    return math.remainder(requested_yaw, 2.0 * math.pi)


@dataclass(frozen=True)
class TerrainBoxGeometry:
    name: str
    center_world_xyz: np.ndarray
    half_size_xyz: np.ndarray
    yaw_world: float

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ContractError("terrain box name must be non-empty")
        center = _finite(
            self.center_world_xyz, (3,), "center_world_xyz"
        ).astype(np.float32)
        size = _finite(
            self.half_size_xyz, (3,), "half_size_xyz"
        ).astype(np.float32)
        if np.any(size <= 0.0):
            raise ContractError("terrain box half sizes must be positive")
        yaw = float(self.yaw_world)
        if not math.isfinite(yaw):
            raise ContractError("terrain box yaw must be finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "center_world_xyz", center)
        object.__setattr__(self, "half_size_xyz", size)
        object.__setattr__(self, "yaw_world", yaw)


class FourWayPlatformScene:
    """A finite raised platform with one Justin staircase on every side."""

    def __init__(
        self,
        *,
        platform_half_extent_m: float = 1.35,
    ) -> None:
        half = float(platform_half_extent_m)
        if not math.isfinite(half) or half <= JUSTIN_STAIR_WIDTH_M:
            raise ContractError("platform_half_extent_m is too small")
        self.platform_half_extent_m = half
        length = JUSTIN_STAIR_RUN_M * JUSTIN_STAIR_TREADS
        self.stairs = (
            StairPlacement(
                np.asarray((-half - length, 0.0, 0.0)),
                0.0,
                "west",
            ),
            StairPlacement(
                np.asarray((half + length, 0.0, 0.0)),
                math.pi,
                "east",
            ),
            StairPlacement(
                np.asarray((0.0, -half - length, 0.0)),
                math.pi / 2,
                "south",
            ),
            StairPlacement(
                np.asarray((0.0, half + length, 0.0)),
                -math.pi / 2,
                "north",
            ),
        )

    @property
    def platform_height_m(self) -> float:
        return JUSTIN_STAIR_RISE_M * JUSTIN_STAIR_TREADS

    def box_geometries(self) -> tuple[TerrainBoxGeometry, ...]:
        boxes = [
            TerrainBoxGeometry(
                "platform",
                np.asarray(
                    (0.0, 0.0, self.platform_height_m * 0.5)
                ),
                np.asarray(
                    (
                        self.platform_half_extent_m,
                        self.platform_half_extent_m,
                        self.platform_height_m * 0.5,
                    )
                ),
                0.0,
            )
        ]
        for stair in self.stairs:
            for tread in range(1, JUSTIN_STAIR_TREADS + 1):
                height = tread * JUSTIN_STAIR_RISE_M
                center = stair.stair_to_world_xyz(
                    np.asarray(
                        (
                            (tread - 0.5) * JUSTIN_STAIR_RUN_M,
                            0.0,
                            height * 0.5,
                        )
                    )
                )
                boxes.append(
                    TerrainBoxGeometry(
                        f"{stair.name}_{tread}",
                        center,
                        np.asarray(
                            (
                                JUSTIN_STAIR_RUN_M * 0.5,
                                JUSTIN_STAIR_WIDTH_M * 0.5,
                                height * 0.5,
                            )
                        ),
                        stair.ascent_yaw_world,
                    )
                )
        return tuple(boxes)

    def height_at_world_xy(self, world_xy: object) -> float:
        point = _finite(world_xy, (2,), "world_xy")
        half = self.platform_half_extent_m
        height = (
            self.platform_height_m
            if abs(point[0]) <= half and abs(point[1]) <= half
            else 0.0
        )
        point_xyz = np.asarray((point[0], point[1], 0.0))
        length = JUSTIN_STAIR_RUN_M * JUSTIN_STAIR_TREADS
        for stair in self.stairs:
            local = stair.world_to_stair_xyz(point_xyz)
            if (
                0.0 <= local[0] <= length
                and abs(local[1]) <= JUSTIN_STAIR_WIDTH_M * 0.5
            ):
                tread = min(
                    JUSTIN_STAIR_TREADS,
                    int(math.floor(local[0] / JUSTIN_STAIR_RUN_M)) + 1,
                )
                height = max(height, tread * JUSTIN_STAIR_RISE_M)
        return float(height)

    def height_map(
        self,
        *,
        root_position_world: np.ndarray,
        root_yaw_world: float,
    ) -> np.ndarray:
        """Rasterize scene surfaces into the causal robot sensor frame."""

        root = _finite(
            root_position_world, (3,), "root_position_world"
        )
        forward = np.linspace(
            *TERRAIN_FORWARD_RANGE_M, TERRAIN_ROWS, dtype=np.float64
        )
        lateral = np.linspace(
            *TERRAIN_LATERAL_RANGE_M, TERRAIN_COLS, dtype=np.float64
        )
        ff, ll = np.meshgrid(forward, lateral, indexing="ij")
        local = np.stack((ff, ll), axis=-1)
        world = root[:2] + local @ _rotation(root_yaw_world).T
        half = self.platform_half_extent_m
        height_world = np.where(
            (np.abs(world[..., 0]) <= half)
            & (np.abs(world[..., 1]) <= half),
            self.platform_height_m,
            0.0,
        )
        length = JUSTIN_STAIR_RUN_M * JUSTIN_STAIR_TREADS
        for stair in self.stairs:
            delta = world - stair.origin_world_xyz[:2]
            stair_xy = delta @ _rotation(-stair.ascent_yaw_world).T
            inside = (
                (stair_xy[..., 0] >= 0.0)
                & (stair_xy[..., 0] <= length)
                & (
                    np.abs(stair_xy[..., 1])
                    <= JUSTIN_STAIR_WIDTH_M * 0.5
                )
            )
            tread = np.clip(
                np.floor(
                    stair_xy[..., 0] / JUSTIN_STAIR_RUN_M
                ).astype(np.int32)
                + 1,
                1,
                JUSTIN_STAIR_TREADS,
            )
            height_world = np.maximum(
                height_world,
                np.where(
                    inside,
                    tread * JUSTIN_STAIR_RISE_M,
                    0.0,
                ),
            )
        return np.ascontiguousarray(
            height_world - root[2], dtype=np.float32
        )
