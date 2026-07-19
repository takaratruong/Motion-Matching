"""Pure Holden control mapping with no X11 or MuJoCo dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .commands import CommandSample
from .joints import ContractError


@dataclass(frozen=True)
class NormalizedControlState:
    """One device-independent control snapshot with axes in ``[-1, 1]``."""

    left_x: float = 0.0
    left_z: float = 0.0
    right_x: float = 0.0
    right_z: float = 0.0
    strafe: bool = False
    walk: bool = False
    zoom: float = 0.0
    stand: bool = False
    terminate: bool = False

    def __post_init__(self) -> None:
        for name in ("left_x", "left_z", "right_x", "right_z", "zoom"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ContractError(f"normalized control {name} must be finite")
            if not -1.0 <= float(value) <= 1.0:
                raise ContractError(f"normalized control {name} must be in [-1, 1]")
            object.__setattr__(self, name, float(value))
        for name in ("strafe", "walk", "stand", "terminate"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"normalized control {name} must be boolean")


@dataclass(frozen=True)
class CameraState:
    """The immutable orbit camera pose derived from the right stick and zoom."""

    sequence: int
    azimuth_rad: float
    altitude_rad: float
    distance_m: float

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ContractError("camera sequence must be a nonnegative integer")
        for name in ("azimuth_rad", "altitude_rad", "distance_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ContractError(f"camera {name} must be finite")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class MappedControlState:
    """The pure command output for one control frame."""

    velocity_mujoco: tuple[float, float, float]
    desired_heading_mujoco_wxyz: tuple[float, float, float, float]
    camera: CameraState
    strafe: bool
    walk_blend: float
    stand: bool
    terminate: bool

    def __post_init__(self) -> None:
        if (
            type(self.velocity_mujoco) is not tuple
            or len(self.velocity_mujoco) != 3
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in self.velocity_mujoco
            )
        ):
            raise ContractError("velocity_mujoco must be three finite values")
        if (
            type(self.desired_heading_mujoco_wxyz) is not tuple
            or len(self.desired_heading_mujoco_wxyz) != 4
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in self.desired_heading_mujoco_wxyz
            )
        ):
            raise ContractError("desired_heading_mujoco_wxyz must be four finite values")
        if type(self.camera) is not CameraState:
            raise ContractError("camera must be a CameraState")
        for name in ("strafe", "stand", "terminate"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"mapped control {name} must be boolean")
        if (
            type(self.walk_blend) not in (int, float)
            or not math.isfinite(float(self.walk_blend))
        ):
            raise ContractError("walk_blend must be finite")
        object.__setattr__(
            self,
            "velocity_mujoco",
            tuple(float(value) for value in self.velocity_mujoco),
        )
        object.__setattr__(
            self,
            "desired_heading_mujoco_wxyz",
            tuple(float(value) for value in self.desired_heading_mujoco_wxyz),
        )
        object.__setattr__(self, "walk_blend", float(self.walk_blend))


class HoldenControlMapper:
    """Port the Holden locomotion controls to MuJoCo command basis."""

    RUN_SPEEDS_MPS = (0.9, 0.6, 0.6)
    WALK_SPEEDS_MPS = (0.5, 0.4, 0.4)
    GAIT_HALFLIFE_S = 0.1
    CAMERA_RATE_RAD_S = 2.0
    ZOOM_RATE_M_S = 10.0
    MIN_DISTANCE_M = 0.1
    MAX_DISTANCE_M = 100.0

    def __init__(
        self,
        *,
        initial_heading_yaw_rad: float,
        initial_altitude_rad: float = 0.4,
        initial_distance_m: float = 3.0,
    ) -> None:
        values = (
            initial_heading_yaw_rad,
            initial_altitude_rad,
            initial_distance_m,
        )
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in values
        ):
            raise ContractError("Holden control initial state must be finite")
        self._desired_yaw = float(initial_heading_yaw_rad)
        self._camera_azimuth = float(initial_heading_yaw_rad)
        self._camera_altitude = min(max(float(initial_altitude_rad), 0.0), 0.4 * math.pi)
        self._camera_distance = min(max(float(initial_distance_m), 0.1), 100.0)
        self._camera_sequence = 0
        self._gait = 0.0
        self._gait_velocity = 0.0

    @staticmethod
    def _stick(x: float, z: float) -> tuple[float, float]:
        magnitude = math.hypot(x, z)
        if magnitude <= 0.2:
            return 0.0, 0.0
        clipped = min(1.0, magnitude * magnitude)
        return x * clipped / magnitude, z * clipped / magnitude

    def update(
        self,
        state: NormalizedControlState,
        dt_s: float,
    ) -> MappedControlState:
        if type(state) is not NormalizedControlState:
            raise ContractError("Holden mapper requires NormalizedControlState")
        if type(dt_s) not in (int, float) or not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ContractError("Holden mapper dt_s must be positive and finite")
        dt = float(dt_s)
        left_x, left_z = self._stick(state.left_x, state.left_z)
        right_x, right_z = self._stick(state.right_x, state.right_z)

        goal = 1.0 if state.walk else 0.0
        y = (4.0 * math.log(2.0) / (self.GAIT_HALFLIFE_S + 1.0e-5)) / 2.0
        j0 = self._gait - goal
        j1 = self._gait_velocity + j0 * y
        x = y * dt
        eydt = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        self._gait = eydt * (j0 + j1 * dt) + goal
        self._gait_velocity = eydt * (self._gait_velocity - j1 * y * dt)

        old_camera = (
            self._camera_azimuth,
            self._camera_altitude,
            self._camera_distance,
        )
        if not state.strafe:
            self._camera_azimuth = math.remainder(
                self._camera_azimuth - self.CAMERA_RATE_RAD_S * dt * right_x,
                2.0 * math.pi,
            )
            self._camera_altitude = min(
                max(
                    self._camera_altitude
                    + self.CAMERA_RATE_RAD_S * dt * right_z,
                    0.0,
                ),
                0.4 * math.pi,
            )
        self._camera_distance = min(
            max(
                self._camera_distance + self.ZOOM_RATE_M_S * dt * state.zoom,
                self.MIN_DISTANCE_M,
            ),
            self.MAX_DISTANCE_M,
        )
        if old_camera != (
            self._camera_azimuth,
            self._camera_altitude,
            self._camera_distance,
        ):
            self._camera_sequence += 1

        forward = -left_z
        left = -left_x
        ca = math.cos(self._camera_azimuth)
        sa = math.sin(self._camera_azimuth)
        world_forward = ca * forward - sa * left
        world_left = sa * forward + ca * left

        ch = math.cos(self._desired_yaw)
        sh = math.sin(self._desired_yaw)
        local_forward = ch * world_forward + sh * world_left
        local_left = -sh * world_forward + ch * world_left
        run_forward, run_side, run_back = self.RUN_SPEEDS_MPS
        walk_forward, walk_side, walk_back = self.WALK_SPEEDS_MPS
        forward_speed = run_forward + (walk_forward - run_forward) * self._gait
        side_speed = run_side + (walk_side - run_side) * self._gait
        back_speed = run_back + (walk_back - run_back) * self._gait
        scaled_forward = local_forward * (
            forward_speed if local_forward >= 0.0 else back_speed
        )
        scaled_left = local_left * side_speed
        velocity_x = ch * scaled_forward - sh * scaled_left
        velocity_y = sh * scaled_forward + ch * scaled_left

        if state.strafe:
            heading_forward = 1.0
            heading_left = 0.0
            if math.hypot(right_x, right_z) > 0.01:
                heading_forward = -right_z
                heading_left = -right_x
            heading_x = ca * heading_forward - sa * heading_left
            heading_y = sa * heading_forward + ca * heading_left
            self._desired_yaw = math.atan2(heading_y, heading_x)
        elif math.hypot(left_x, left_z) > 0.01:
            self._desired_yaw = math.atan2(world_left, world_forward)

        if state.stand:
            velocity_x = 0.0
            velocity_y = 0.0
        half = 0.5 * self._desired_yaw
        return MappedControlState(
            velocity_mujoco=(velocity_x, velocity_y, 0.0),
            desired_heading_mujoco_wxyz=(
                math.cos(half), 0.0, 0.0, math.sin(half)
            ),
            camera=CameraState(
                self._camera_sequence,
                self._camera_azimuth,
                self._camera_altitude,
                self._camera_distance,
            ),
            strafe=state.strafe,
            walk_blend=self._gait,
            stand=state.stand,
            terminate=state.terminate,
        )

    def command(
        self,
        chunk_index: int,
        mapped: MappedControlState,
    ) -> CommandSample | None:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("Holden command chunk index must be nonnegative")
        if type(mapped) is not MappedControlState:
            raise ContractError("Holden command requires MappedControlState")
        if mapped.terminate:
            return None
        return CommandSample(
            chunk_index=chunk_index,
            requested_velocity_mujoco=(
                (0.0, 0.0, 0.0)
                if mapped.stand
                else mapped.velocity_mujoco
            ),
            desired_heading_mujoco_wxyz=mapped.desired_heading_mujoco_wxyz,
        )
