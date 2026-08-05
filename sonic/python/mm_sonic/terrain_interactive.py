"""Pure control and playback state for the terrain MM kinematic viewer.

The classes in this module deliberately do not import MuJoCo, X11, or zarr.
The visible entry point lives in :mod:`mm_sonic.terrain_interactive_viewer`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .holden_control import NormalizedControlState
from .joints import ContractError
from .terrain_catalog import (
    JUSTIN_STAIR_RISE_M,
    JUSTIN_STAIR_RUN_M,
    JUSTIN_STAIR_TREADS,
    JUSTIN_STAIR_WIDTH_M,
)
from .terrain_motion import (
    MotionMode,
    StairGeometry,
    TerrainFrameDatabase,
    TerrainMatch,
    TerrainMatcherConfig,
    TerrainMatchQuery,
    TerrainMotionMatcher,
    complete_route_ready_frames,
    wrap_angle,
)


def _finite_scalar(value: object, name: str) -> float:
    if type(value) not in (int, float, np.float32, np.float64):
        raise ContractError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} must be a finite number")
    return number


def _finite_xy(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ContractError(f"{name} must contain two finite values")
    return array


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s), (s, c)), dtype=np.float64)


def _takara_fast_negexp(value: float) -> float:
    """Rational exp(-x) approximation used by Takara's MM runtime."""

    return 1.0 / (
        1.0
        + value
        + 0.48 * value * value
        + 0.235 * value * value * value
    )


def _takara_position_step(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    desired_velocity: np.ndarray,
    *,
    halflife_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance Takara's persistent position/velocity/acceleration spring."""

    y = (4.0 * math.log(2.0) / (halflife_s + 1.0e-5)) / 2.0
    j0 = velocity - desired_velocity
    j1 = acceleration + j0 * y
    decay = _takara_fast_negexp(y * dt_s)
    next_position = (
        decay * ((-j1 / (y * y)) + ((-j0 - j1 * dt_s) / y))
        + j1 / (y * y)
        + j0 / y
        + desired_velocity * dt_s
        + position
    )
    next_velocity = decay * (j0 + j1 * dt_s) + desired_velocity
    next_acceleration = decay * (
        acceleration - j1 * y * dt_s
    )
    return next_position, next_velocity, next_acceleration


@dataclass(frozen=True)
class TerrainInputIntent:
    """One control sample consumed by the command-ball filter."""

    local_velocity_xy: np.ndarray
    desired_facing_yaw_world: float
    camera_azimuth_rad: float
    camera_altitude_rad: float
    camera_distance_m: float
    stand: bool
    terminate: bool

    def __post_init__(self) -> None:
        velocity = _finite_xy(self.local_velocity_xy, "local_velocity_xy")
        object.__setattr__(self, "local_velocity_xy", velocity)
        for name in (
            "desired_facing_yaw_world",
            "camera_azimuth_rad",
            "camera_altitude_rad",
            "camera_distance_m",
        ):
            object.__setattr__(
                self, name, _finite_scalar(getattr(self, name), name)
            )
        if self.camera_distance_m <= 0.0:
            raise ContractError("camera_distance_m must be positive")
        if type(self.stand) is not bool or type(self.terminate) is not bool:
            raise ContractError("stand and terminate must be boolean")


class TerrainKeyboardMapper:
    """Map the existing Sonic keys onto independent travel and facing intent.

    ``WASD`` always denotes robot-local travel and never silently changes the
    facing target. With Control held, the arrow keys point the facing stick in
    the camera/world plane. Without Control they orbit the camera.
    """

    RUN_SPEEDS_MPS = (0.70, 0.52, 0.52)
    WALK_SPEEDS_MPS = (0.46, 0.36, 0.40)
    CAMERA_RATE_RAD_S = 2.0
    ZOOM_RATE_M_S = 6.0

    def __init__(
        self,
        *,
        initial_yaw_world: float,
        initial_altitude_rad: float = 0.32,
        initial_distance_m: float = 2.5,
    ) -> None:
        yaw = _finite_scalar(initial_yaw_world, "initial_yaw_world")
        altitude = _finite_scalar(initial_altitude_rad, "initial_altitude_rad")
        distance = _finite_scalar(initial_distance_m, "initial_distance_m")
        if distance <= 0.0:
            raise ContractError("initial_distance_m must be positive")
        self._desired_facing_yaw_world = yaw
        self._camera_azimuth = yaw
        self._camera_altitude = float(np.clip(altitude, 0.0, 0.4 * math.pi))
        self._camera_distance = float(np.clip(distance, 0.4, 20.0))

    @staticmethod
    def _stick(x: float, z: float) -> tuple[float, float]:
        magnitude = math.hypot(x, z)
        if magnitude <= 0.2:
            return 0.0, 0.0
        scaled = min(1.0, magnitude * magnitude)
        return x * scaled / magnitude, z * scaled / magnitude

    def update(
        self,
        state: NormalizedControlState,
        *,
        current_yaw_world: float,
        dt_s: float,
    ) -> TerrainInputIntent:
        if type(state) is not NormalizedControlState:
            raise ContractError(
                "terrain keyboard mapper requires NormalizedControlState"
            )
        _finite_scalar(current_yaw_world, "current_yaw_world")
        dt = _finite_scalar(dt_s, "dt_s")
        if dt <= 0.0:
            raise ContractError("dt_s must be positive")
        left_x, left_z = self._stick(state.left_x, state.left_z)
        right_x, right_z = self._stick(state.right_x, state.right_z)

        if state.strafe:
            if math.hypot(right_x, right_z) > 0.01:
                forward = -right_z
                left = -right_x
                c, s = (
                    math.cos(self._camera_azimuth),
                    math.sin(self._camera_azimuth),
                )
                world_forward = c * forward - s * left
                world_left = s * forward + c * left
                self._desired_facing_yaw_world = math.atan2(
                    world_left, world_forward
                )
        else:
            self._camera_azimuth = math.remainder(
                self._camera_azimuth
                - self.CAMERA_RATE_RAD_S * dt * right_x,
                2.0 * math.pi,
            )
            self._camera_altitude = float(
                np.clip(
                    self._camera_altitude
                    + self.CAMERA_RATE_RAD_S * dt * right_z,
                    0.0,
                    0.4 * math.pi,
                )
            )
        self._camera_distance = float(
            np.clip(
                self._camera_distance + self.ZOOM_RATE_M_S * dt * state.zoom,
                0.4,
                20.0,
            )
        )

        forward = -left_z
        left = -left_x
        speeds = self.WALK_SPEEDS_MPS if state.walk else self.RUN_SPEEDS_MPS
        forward_speed = speeds[0] if forward >= 0.0 else speeds[2]
        local_velocity = np.asarray(
            (forward * forward_speed, left * speeds[1]), dtype=np.float64
        )
        if state.stand:
            local_velocity[:] = 0.0
        return TerrainInputIntent(
            local_velocity_xy=local_velocity,
            desired_facing_yaw_world=self._desired_facing_yaw_world,
            camera_azimuth_rad=self._camera_azimuth,
            camera_altitude_rad=self._camera_altitude,
            camera_distance_m=self._camera_distance,
            stand=state.stand,
            terminate=state.terminate,
        )


class TerrainCommandBall:
    """Critically damped velocity/facing state with robot-centred previews.

    Only velocity, velocity derivative, facing yaw, and yaw rate persist. No
    world or root position is stored, so each preview necessarily starts at the
    robot's current origin.
    """

    def __init__(
        self,
        *,
        velocity_halflife_s: float = 0.27,
        facing_halflife_s: float = 0.27,
    ) -> None:
        velocity_halflife = _finite_scalar(
            velocity_halflife_s, "velocity_halflife_s"
        )
        facing_halflife = _finite_scalar(
            facing_halflife_s, "facing_halflife_s"
        )
        if velocity_halflife <= 0.0 or facing_halflife <= 0.0:
            raise ContractError("command half-lives must be positive")
        self._velocity_halflife_s = velocity_halflife
        self._facing_halflife_s = facing_halflife
        self._velocity_world = np.zeros(2, dtype=np.float64)
        self._velocity_rate_world = np.zeros(2, dtype=np.float64)
        self._facing_yaw_world = 0.0
        self._facing_yaw_rate = 0.0

    @property
    def velocity_world(self) -> np.ndarray:
        return self._velocity_world.copy()

    @property
    def facing_yaw_world(self) -> float:
        return self._facing_yaw_world

    def reset(
        self,
        *,
        current_yaw_world: float,
        initial_velocity_world: np.ndarray | None = None,
    ) -> None:
        yaw = _finite_scalar(current_yaw_world, "current_yaw_world")
        velocity = (
            np.zeros(2, dtype=np.float64)
            if initial_velocity_world is None
            else _finite_xy(initial_velocity_world, "initial_velocity_world")
        )
        self._velocity_world = velocity.copy()
        self._velocity_rate_world = np.zeros(2, dtype=np.float64)
        self._facing_yaw_world = yaw
        self._facing_yaw_rate = 0.0

    def _step_yaw(
        self,
        yaw: float,
        yaw_rate: float,
        goal_yaw: float,
        dt_s: float,
    ) -> tuple[float, float]:
        y = (
            4.0
            * math.log(2.0)
            / (self._facing_halflife_s + 1.0e-5)
        ) / 2.0
        j0 = float(wrap_angle(yaw - goal_yaw))
        j1 = yaw_rate + j0 * y
        decay = _takara_fast_negexp(y * dt_s)
        next_yaw = float(
            wrap_angle(decay * (j0 + j1 * dt_s) + goal_yaw)
        )
        next_rate = decay * (
            yaw_rate - j1 * y * dt_s
        )
        return next_yaw, next_rate

    def update(
        self,
        local_velocity_xy: np.ndarray,
        desired_facing_yaw_world: float,
        current_yaw_world: float,
        dt_s: float,
    ) -> None:
        local_velocity = _finite_xy(
            local_velocity_xy, "local_velocity_xy"
        )
        desired_yaw = _finite_scalar(
            desired_facing_yaw_world, "desired_facing_yaw_world"
        )
        current_yaw = _finite_scalar(current_yaw_world, "current_yaw_world")
        dt = _finite_scalar(dt_s, "dt_s")
        if dt <= 0.0:
            raise ContractError("dt_s must be positive")
        target_velocity_world = _rotation(current_yaw) @ local_velocity
        (
            _unused_position,
            self._velocity_world,
            self._velocity_rate_world,
        ) = _takara_position_step(
                np.zeros(2, dtype=np.float64),
                self._velocity_world,
                self._velocity_rate_world,
                target_velocity_world,
                halflife_s=self._velocity_halflife_s,
                dt_s=dt,
        )
        self._facing_yaw_world, self._facing_yaw_rate = self._step_yaw(
            self._facing_yaw_world,
            self._facing_yaw_rate,
            desired_yaw,
            dt,
        )

    def preview(
        self,
        local_velocity_xy: np.ndarray,
        desired_facing_yaw_world: float,
        current_yaw_world: float,
        offsets_frames: tuple[int, ...],
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preview future travel/facing from the current robot origin."""

        local_velocity = _finite_xy(
            local_velocity_xy, "local_velocity_xy"
        )
        desired_yaw = _finite_scalar(
            desired_facing_yaw_world, "desired_facing_yaw_world"
        )
        current_yaw = _finite_scalar(current_yaw_world, "current_yaw_world")
        rate = _finite_scalar(fps, "fps")
        if rate <= 0.0:
            raise ContractError("fps must be positive")
        offsets = tuple(offsets_frames)
        if (
            not offsets
            or any(type(value) is not int or value <= 0 for value in offsets)
            or tuple(sorted(set(offsets))) != offsets
        ):
            raise ContractError(
                "offsets_frames must be unique increasing positive integers"
            )

        velocity = self._velocity_world.copy()
        velocity_rate = self._velocity_rate_world.copy()
        yaw = self._facing_yaw_world
        yaw_rate = self._facing_yaw_rate
        displacement_world = np.zeros(2, dtype=np.float64)
        roots = np.empty((len(offsets), 2), dtype=np.float32)
        facings = np.empty_like(roots)
        dt = 1.0 / rate
        output_index = 0
        for frame in range(1, offsets[-1] + 1):
            yaw, yaw_rate = self._step_yaw(
                yaw, yaw_rate, desired_yaw, dt
            )
            # The travel stick is robot-local. During a predicted turn its
            # world target rotates with the filtered future facing, producing
            # an arc rather than an instantaneous right-angle corner.
            target_velocity_world = _rotation(yaw) @ local_velocity
            displacement_world, velocity, velocity_rate = (
                _takara_position_step(
                    displacement_world,
                    velocity,
                    velocity_rate,
                    target_velocity_world,
                    halflife_s=self._velocity_halflife_s,
                    dt_s=dt,
                )
            )
            if frame == offsets[output_index]:
                roots[output_index] = (
                    _rotation(-current_yaw) @ displacement_world
                )
                relative_yaw = yaw - current_yaw
                facings[output_index] = (
                    math.cos(relative_yaw),
                    math.sin(relative_yaw),
                )
                output_index += 1
                if output_index == len(offsets):
                    break
        return roots, facings


def strict_interactive_matcher(
    database: TerrainFrameDatabase,
) -> TerrainMotionMatcher:
    """Construct the deliberately conservative first interactive matcher."""

    return TerrainMotionMatcher(
        database,
        TerrainMatcherConfig(
            feature_weight=0.03,
            trajectory_weight=80.0,
            facing_weight=20.0,
            stair_pose_weight=25.0,
            planted_foot_weight=200.0,
            continuation_bonus=0.10,
            maximum_stair_origin_error_m=0.040,
            maximum_stair_yaw_error_rad=math.radians(5.0),
            maximum_planted_foot_error_m=0.010,
            minimum_stair_confidence=0.65,
        ),
    )


class _Matcher(Protocol):
    database: TerrainFrameDatabase

    def select(self, query: TerrainMatchQuery) -> TerrainMatch:
        raise NotImplementedError


@dataclass(frozen=True)
class TerrainStepOutcome:
    row: int
    source_clip: int
    source_frame: int
    searched: bool
    switched: bool
    held: bool
    cost: float
    candidate_count: int
    maximum_planted_foot_mismatch_m: float
    reason: str


class KinematicTerrainSession:
    """Own source continuity and perform a strict search every N frames."""

    _TIME_WARP_REASON_PREFIX = "time-warped stationary support"
    _MAXIMUM_TIME_WARP_ADVANCE_FRAMES = 4
    _MINIMUM_MOVING_COMMAND_ENDPOINT_M = 0.12
    _MAXIMUM_SOURCE_PROGRESS_M = 0.08
    _MAXIMUM_TIME_WARP_JOINT_STEP_RAD = 0.05
    _MAXIMUM_TIME_WARP_ROOT_STEP_M = 0.010
    _MAXIMUM_TIME_WARP_ROOT_HEIGHT_STEP_M = 0.010
    _MAXIMUM_TIME_WARP_PLANTED_FOOT_STEP_M = 0.010
    _MAXIMUM_TIME_WARP_STAIR_YAW_STEP_RAD = math.radians(2.0)

    def __init__(
        self,
        database: TerrainFrameDatabase,
        *,
        initial_row: int,
        matcher: _Matcher | None = None,
        search_interval_frames: int = 5,
        maximum_joint_switch_error_rad: float = 0.30,
    ) -> None:
        if type(database) is not TerrainFrameDatabase:
            raise ContractError("database must be a TerrainFrameDatabase")
        if type(initial_row) is not int or not 0 <= initial_row < database.row_count:
            raise ContractError("initial_row is outside the terrain database")
        if (
            type(search_interval_frames) is not int
            or search_interval_frames <= 0
        ):
            raise ContractError("search_interval_frames must be positive")
        maximum_joint_switch_error = float(
            maximum_joint_switch_error_rad
        )
        if (
            not math.isfinite(maximum_joint_switch_error)
            or maximum_joint_switch_error <= 0.0
        ):
            raise ContractError(
                "maximum_joint_switch_error_rad must be finite and positive"
            )
        self.database = database
        self.matcher = (
            strict_interactive_matcher(database) if matcher is None else matcher
        )
        if self.matcher.database is not database:
            raise ContractError("matcher must own the session database")
        self.search_interval_frames = search_interval_frames
        self.maximum_joint_switch_error_rad = (
            maximum_joint_switch_error
        )
        self.active_row = initial_row
        self.tick = 0
        self.switch_count = 0
        self.hold_count = 0
        self._lookup = {
            (int(clip), int(frame)): row
            for row, (clip, frame) in enumerate(
                zip(
                    database.source_clip,
                    database.source_frame,
                    strict=True,
                )
            )
        }
        # Lifecycle annotations distinguish complete stair traversals from
        # useful-but-partial fragments.  Partial fragments may contribute
        # poses to an offline catalog, but switching into one online can strand
        # playback at its final frame.  Preserve legacy behavior for databases
        # with no annotated exits at all.
        exit_flag = np.asarray(database.exit, dtype=bool)
        self._complete_route_clips: frozenset[int] | None = None
        if bool(np.any(exit_flag)):
            self._complete_route_clips = frozenset(
                complete_route_ready_frames(
                    database,
                    tread_rise_m=JUSTIN_STAIR_RISE_M,
                )
            )

    @property
    def active_clip(self) -> int:
        return int(self.database.source_clip[self.active_row])

    @property
    def active_frame(self) -> int:
        return int(self.database.source_frame[self.active_row])

    def reset(self, row: int) -> None:
        if type(row) is not int or not 0 <= row < self.database.row_count:
            raise ContractError("reset row is outside the terrain database")
        self.active_row = row
        self.tick = 0
        self.switch_count = 0
        self.hold_count = 0

    def snapshot_state(self) -> tuple[int, int, int, int]:
        """Capture the small mutable state needed for a transactional step."""

        return (
            int(self.active_row),
            int(self.tick),
            int(self.switch_count),
            int(self.hold_count),
        )

    def restore_state(self, snapshot: object) -> None:
        """Roll back a proposed step rejected by the exact pose oracle."""

        if (
            not isinstance(snapshot, tuple)
            or len(snapshot) != 4
            or not all(type(value) is int for value in snapshot)
        ):
            raise ContractError("invalid terrain session snapshot")
        active_row, tick, switch_count, hold_count = snapshot
        if (
            not 0 <= active_row < self.database.row_count
            or min(tick, switch_count, hold_count) < 0
        ):
            raise ContractError("terrain session snapshot is outside bounds")
        self.active_row = active_row
        self.tick = tick
        self.switch_count = switch_count
        self.hold_count = hold_count

    def _sequential_match(self, reason: str) -> TerrainMatch:
        key = (self.active_clip, self.active_frame + 1)
        row = self._lookup.get(key)
        if row is None:
            return TerrainMatch(
                False,
                -1,
                -1,
                -1,
                float("inf"),
                f"hold: {reason}; no sequential frame",
                0,
            )
        return TerrainMatch(
            True,
            row,
            int(self.database.source_clip[row]),
            int(self.database.source_frame[row]),
            0.0,
            f"sequential: {reason}",
            1,
        )

    def _time_warp_stationary_support(
        self,
        match: TerrainMatch,
        future_root_xy: np.ndarray,
    ) -> TerrainMatch:
        """Compact redundant planted frames without crossing a gait event.

        Stair captures can contain long preparation holds before a step.  A
        game-style locomotion database should treat those as compressible
        timing, not force the player to wait through the authored pause.  Only
        an exact sequential continuation may be accelerated, and every
        skipped frame must retain double support, tread identity, fixed-world
        stair pose, planted feet, and a small joint/root displacement.
        """

        db = self.database
        if (
            not match.accepted
            or int(match.source_clip) != self.active_clip
            or int(match.source_frame) != self.active_frame + 1
            or not bool(np.all(db.contact[self.active_row]))
        ):
            return match
        command = np.asarray(future_root_xy, dtype=np.float32)
        if (
            command.shape != db.future_root_xy.shape[1:]
            or not np.isfinite(command).all()
        ):
            raise ContractError(
                "future_root_xy has the wrong shape for time warping"
            )
        endpoint = command[-1]
        endpoint_norm = float(np.linalg.norm(endpoint))
        if endpoint_norm < self._MINIMUM_MOVING_COMMAND_ENDPOINT_M:
            return match
        command_direction = endpoint / endpoint_norm
        source_progress = float(
            np.dot(
                np.asarray(
                    db.future_root_xy[self.active_row, -1],
                    dtype=np.float32,
                ),
                command_direction,
            )
        )
        if source_progress > self._MAXIMUM_SOURCE_PROGRESS_M:
            return match

        active = self.active_row
        active_contact = np.asarray(db.contact[active], dtype=bool)
        active_tread = np.asarray(db.tread_id[active], dtype=np.int16)
        best = int(match.row)
        for advance in range(
            2, self._MAXIMUM_TIME_WARP_ADVANCE_FRAMES + 1
        ):
            row = self._lookup.get(
                (self.active_clip, self.active_frame + advance)
            )
            if row is None:
                break
            candidate_contact = np.asarray(db.contact[row], dtype=bool)
            candidate_tread = np.asarray(db.tread_id[row], dtype=np.int16)
            if (
                not np.array_equal(candidate_contact, active_contact)
                or not np.array_equal(candidate_tread, active_tread)
                or float(
                    np.max(
                        np.abs(
                            db.feature[row, :29]
                            - db.feature[active, :29]
                        ),
                        initial=0.0,
                    )
                )
                > self._MAXIMUM_TIME_WARP_JOINT_STEP_RAD
                or float(
                    np.linalg.norm(
                        db.stair_origin_root_xy[row]
                        - db.stair_origin_root_xy[active]
                    )
                )
                > self._MAXIMUM_TIME_WARP_ROOT_STEP_M
                or abs(
                    float(
                        db.root_height_above_stair_base_m[row]
                        - db.root_height_above_stair_base_m[active]
                    )
                )
                > self._MAXIMUM_TIME_WARP_ROOT_HEIGHT_STEP_M
                or abs(
                    math.remainder(
                        float(
                            db.stair_ascent_yaw_root[row]
                            - db.stair_ascent_yaw_root[active]
                        ),
                        2.0 * math.pi,
                    )
                )
                > self._MAXIMUM_TIME_WARP_STAIR_YAW_STEP_RAD
            ):
                break
            planted_error = np.linalg.norm(
                db.feet_xyz_stair[row, active_contact]
                - db.feet_xyz_stair[active, active_contact],
                axis=1,
            )
            if (
                float(np.max(planted_error, initial=0.0))
                > self._MAXIMUM_TIME_WARP_PLANTED_FOOT_STEP_M
            ):
                break
            best = int(row)
        if best == int(match.row):
            return match
        return TerrainMatch(
            True,
            best,
            int(db.source_clip[best]),
            int(db.source_frame[best]),
            float(match.cost),
            f"{self._TIME_WARP_REASON_PREFIX}: {match.reason}",
            int(match.candidate_count),
        )

    def _query(
        self, future_root_xy: np.ndarray, future_facing_xy: np.ndarray
    ) -> TerrainMatchQuery:
        row = self.active_row
        db = self.database
        stair = StairGeometry(
            JUSTIN_STAIR_RUN_M,
            JUSTIN_STAIR_RISE_M,
            JUSTIN_STAIR_WIDTH_M,
            JUSTIN_STAIR_TREADS,
            np.asarray(db.stair_origin_root_xy[row], dtype=np.float32),
            float(db.stair_ascent_yaw_root[row]),
            1.0,
        )
        return TerrainMatchQuery(
            feature=db.feature[row],
            future_root_xy=np.asarray(future_root_xy, dtype=np.float32),
            future_facing_xy=np.asarray(future_facing_xy, dtype=np.float32),
            stair=stair,
            contact=db.contact[row],
            tread_id=db.tread_id[row],
            feet_xyz_stair=db.feet_xyz_stair[row],
            mode=MotionMode.STAIR_COMMITTED,
            root_height_above_stair_base_m=float(
                db.root_height_above_stair_base_m[row]
            ),
            active_clip=self.active_clip,
            active_frame=self.active_frame,
        )

    def _maximum_planted_mismatch(self, candidate_row: int) -> float:
        contact = np.asarray(self.database.contact[self.active_row], dtype=bool)
        if not np.any(contact):
            return 0.0
        errors = np.linalg.norm(
            self.database.feet_xyz_stair[candidate_row, contact]
            - self.database.feet_xyz_stair[self.active_row, contact],
            axis=1,
        )
        return float(np.max(errors, initial=0.0))

    def _commit_match(
        self,
        match: TerrainMatch,
        *,
        searched: bool,
        prior_clip: int,
        prior_frame: int,
    ) -> TerrainStepOutcome:
        held = not match.accepted
        mismatch = 0.0
        if match.accepted:
            mismatch = self._maximum_planted_mismatch(match.row)
            self.active_row = match.row
        else:
            self.hold_count += 1
        switched = (
            match.accepted
            and not match.reason.startswith(
                self._TIME_WARP_REASON_PREFIX
            )
            and (
                self.active_clip != prior_clip
                or self.active_frame != prior_frame + 1
            )
        )
        if switched:
            self.switch_count += 1
        self.tick += 1
        return TerrainStepOutcome(
            row=self.active_row,
            source_clip=self.active_clip,
            source_frame=self.active_frame,
            searched=searched,
            switched=switched,
            held=held,
            cost=float(match.cost),
            candidate_count=int(match.candidate_count),
            maximum_planted_foot_mismatch_m=mismatch,
            reason=match.reason,
        )

    def step_sequential(
        self, reason: str = "transactional fallback"
    ) -> TerrainStepOutcome:
        """Advance exactly one source frame without a cross-clip search."""

        if not isinstance(reason, str) or not reason:
            raise ContractError("sequential fallback reason must be nonempty")
        prior_clip, prior_frame = self.active_clip, self.active_frame
        return self._commit_match(
            self._sequential_match(reason),
            searched=False,
            prior_clip=prior_clip,
            prior_frame=prior_frame,
        )

    def step(
        self,
        future_root_xy: np.ndarray,
        future_facing_xy: np.ndarray,
        *,
        stop_requested: bool = False,
    ) -> TerrainStepOutcome:
        if type(stop_requested) is not bool:
            raise ContractError("stop_requested must be boolean")
        if stop_requested and bool(
            np.all(self.database.contact[self.active_row])
        ):
            self.tick += 1
            self.hold_count += 1
            return TerrainStepOutcome(
                row=self.active_row,
                source_clip=self.active_clip,
                source_frame=self.active_frame,
                searched=False,
                switched=False,
                held=True,
                cost=0.0,
                candidate_count=0,
                maximum_planted_foot_mismatch_m=0.0,
                reason="command stop held at double support",
            )
        searched = self.tick % self.search_interval_frames == 0
        prior_row = self.active_row
        prior_clip, prior_frame = self.active_clip, self.active_frame
        if searched:
            match = self.matcher.select(
                self._query(future_root_xy, future_facing_xy)
            )
            if match.accepted and match.row == self.active_row:
                match = self._sequential_match("matcher returned current frame")
            elif (
                match.accepted
                and self._complete_route_clips is not None
                and int(match.source_clip) not in self._complete_route_clips
            ):
                match = self._sequential_match(
                    "rejected cross-switch into incomplete route"
                )
            elif (
                match.accepted
                and self.database.feature.shape[1] >= 29
                and not (
                    int(match.source_clip) == prior_clip
                    and int(match.source_frame) == prior_frame + 1
                )
                and float(
                    np.max(
                        np.abs(
                            self.database.feature[match.row, :29]
                            - self.database.feature[self.active_row, :29]
                        )
                    )
                )
                > self.maximum_joint_switch_error_rad
            ):
                match = self._sequential_match(
                    "rejected search joint discontinuity"
                )
        else:
            match = self._sequential_match("between search boundaries")

        match = self._time_warp_stationary_support(
            match,
            future_root_xy,
        )
        return self._commit_match(
            match,
            searched=searched,
            prior_clip=prior_clip,
            prior_frame=prior_frame,
        )
