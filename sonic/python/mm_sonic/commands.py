"""Frozen target-basis command registries and deterministic route compiler."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Iterable, Mapping, Sequence

import numpy as np

from .joints import ContractError


SOURCE_RATE_HZ = 25
TARGET_RATE_HZ = 50
CHUNK_INTERVALS = 10
ROUTE_SPEED_MPS = 0.5
COMMAND_SCRIPT_SCHEMA = "mm-sonic-command-script/v1"

PERTURBATIONS = (
    ("lateral_p003", 0.03, 0.0),
    ("lateral_m003", -0.03, 0.0),
    ("lateral_p006", 0.06, 0.0),
    ("lateral_m006", -0.06, 0.0),
    ("yaw_p002", 0.0, math.radians(2.0)),
    ("yaw_m002", 0.0, math.radians(-2.0)),
    ("yaw_p004", 0.0, math.radians(4.0)),
    ("yaw_m004", 0.0, math.radians(-4.0)),
    ("combined_p", 0.03, math.radians(2.0)),
    ("combined_m", -0.03, math.radians(-2.0)),
)
TERRAIN_CONDITIONS = (("aware", 4.0), ("blind", 0.0))
REGISTERED_TERRAIN_ROUTES = (
    ("grail-curb-low", "curb-forward"),
    ("ramp-10-up-down", "up-landing-down"),
    ("stairs-shallow", "ascent-landing-descent"),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite_tuple(value: object, width: int, label: str) -> tuple[float, ...]:
    if type(value) not in (tuple, list) or len(value) != width:
        raise ContractError(f"command {label} must have width {width}")
    output: list[float] = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ContractError(f"command {label} must contain finite values")
        converted = float(item)
        output.append(0.0 if converted == 0.0 else converted)
    return tuple(output)


@dataclass(frozen=True)
class CommandSample:
    """One immutable chunk command expressed only in the MuJoCo target basis."""

    chunk_index: int
    requested_velocity_mujoco: tuple[float, float, float]
    desired_heading_mujoco_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ContractError("command chunk_index must be a nonnegative integer")
        velocity = _finite_tuple(self.requested_velocity_mujoco, 3, "velocity")
        heading = _finite_tuple(
            self.desired_heading_mujoco_wxyz, 4, "heading quaternion"
        )
        norm = math.sqrt(sum(value * value for value in heading))
        if abs(norm - 1.0) > 1.0e-6:
            raise ContractError("command heading quaternion must be unit length")
        object.__setattr__(self, "requested_velocity_mujoco", velocity)
        object.__setattr__(self, "desired_heading_mujoco_wxyz", heading)


@dataclass(frozen=True)
class StageCTrialIdentity:
    """One scored physical trial with all scientific identities explicit."""

    scene_id: str
    route_id: str
    condition: str
    terrain_weight: float
    perturbation_id: str
    physical_lateral_offset_m: float
    physical_yaw_offset_rad: float
    command_artifact_id: str
    command_sha256: str
    duration_s: float
    mm_initial_state_sha256: str
    mm_reference_sha256: str

    def __post_init__(self) -> None:
        if (self.scene_id, self.route_id) not in REGISTERED_TERRAIN_ROUTES:
            raise ContractError("Stage C trial scene/route is not registered")
        conditions = dict(TERRAIN_CONDITIONS)
        if self.condition not in conditions:
            raise ContractError("Stage C trial condition is not registered")
        if (
            type(self.terrain_weight) not in (int, float)
            or not math.isfinite(float(self.terrain_weight))
            or float(self.terrain_weight) != conditions[self.condition]
        ):
            raise ContractError("Stage C trial terrain weight is not registered")
        perturbations = {
            perturbation_id: (lateral, yaw)
            for perturbation_id, lateral, yaw in PERTURBATIONS
        }
        if self.perturbation_id not in perturbations:
            raise ContractError("Stage C physical perturbation is not registered")
        physical = (
            self.physical_lateral_offset_m,
            self.physical_yaw_offset_rad,
        )
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in physical
            )
            or tuple(float(value) for value in physical)
            != perturbations[self.perturbation_id]
        ):
            raise ContractError("Stage C physical perturbation values changed")
        if (
            type(self.command_artifact_id) is not str
            or not self.command_artifact_id
            or "\x00" in self.command_artifact_id
        ):
            raise ContractError("Stage C command artifact identity is invalid")
        for label, digest in (
            ("command", self.command_sha256),
            ("initial state", self.mm_initial_state_sha256),
            ("reference", self.mm_reference_sha256),
        ):
            if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
                raise ContractError(f"Stage C {label} identity must be SHA-256")
        if (
            type(self.duration_s) not in (int, float)
            or not math.isfinite(float(self.duration_s))
            or float(self.duration_s) <= 0.0
        ):
            raise ContractError("Stage C duration_s must be finite and positive")
        object.__setattr__(self, "terrain_weight", float(self.terrain_weight))
        object.__setattr__(
            self,
            "physical_lateral_offset_m",
            float(self.physical_lateral_offset_m),
        )
        object.__setattr__(
            self,
            "physical_yaw_offset_rad",
            float(self.physical_yaw_offset_rad),
        )
        object.__setattr__(self, "duration_s", float(self.duration_s))


def validate_stage_c_scored_trials(
    trials: Iterable[StageCTrialIdentity],
) -> tuple[StageCTrialIdentity, ...]:
    """Prove one scene's 20 scored identities contain no experiment confound."""

    values = tuple(trials)
    if len(values) < len(TERRAIN_CONDITIONS) * len(PERTURBATIONS):
        raise ContractError("Stage C trial expansion has a missing trial")
    if len(values) > len(TERRAIN_CONDITIONS) * len(PERTURBATIONS):
        raise ContractError("Stage C trial expansion has an extra trial")
    if any(not isinstance(value, StageCTrialIdentity) for value in values):
        raise ContractError("Stage C trial expansion contains an invalid identity")
    pairs = tuple((value.condition, value.perturbation_id) for value in values)
    if len(set(pairs)) != len(pairs):
        raise ContractError("Stage C trial expansion has a duplicate trial")
    expected_pairs = {
        (condition, perturbation_id)
        for condition, _weight in TERRAIN_CONDITIONS
        for perturbation_id, _lateral, _yaw in PERTURBATIONS
    }
    if set(pairs) != expected_pairs:
        raise ContractError("Stage C trial expansion has a missing trial")

    first = values[0]
    global_fields = (
        ("scene_id", "scene identity"),
        ("route_id", "route identity"),
        ("command_artifact_id", "command artifact"),
        ("command_sha256", "command_sha256"),
        ("duration_s", "duration_s"),
        ("mm_initial_state_sha256", "initial state"),
    )
    for field, label in global_fields:
        expected = getattr(first, field)
        if any(getattr(value, field) != expected for value in values[1:]):
            raise ContractError(f"Stage C {label} changed across scored trials")
    for condition, _weight in TERRAIN_CONDITIONS:
        condition_values = tuple(
            value for value in values if value.condition == condition
        )
        reference = condition_values[0].mm_reference_sha256
        if any(
            value.mm_reference_sha256 != reference
            for value in condition_values[1:]
        ):
            raise ContractError(
                f"Stage C {condition} reference changed across perturbations"
            )
    return values


def expand_stage_c_scored_trials(
    *,
    scene_id: str,
    route_id: str,
    command_artifact_id: str,
    command_sha256: str,
    duration_s: float,
    mm_initial_state_sha256: str,
    mm_reference_sha256_by_condition: Mapping[str, str],
) -> tuple[StageCTrialIdentity, ...]:
    """Expand only the registered weights and ten physical perturbations."""

    if not isinstance(mm_reference_sha256_by_condition, Mapping) or set(
        mm_reference_sha256_by_condition
    ) != {condition for condition, _weight in TERRAIN_CONDITIONS}:
        raise ContractError(
            "Stage C reference identities require exactly aware and blind"
        )
    trials = tuple(
        StageCTrialIdentity(
            scene_id=scene_id,
            route_id=route_id,
            condition=condition,
            terrain_weight=terrain_weight,
            perturbation_id=perturbation_id,
            physical_lateral_offset_m=lateral_offset_m,
            physical_yaw_offset_rad=yaw_offset_rad,
            command_artifact_id=command_artifact_id,
            command_sha256=command_sha256,
            duration_s=duration_s,
            mm_initial_state_sha256=mm_initial_state_sha256,
            mm_reference_sha256=mm_reference_sha256_by_condition[condition],
        )
        for condition, terrain_weight in TERRAIN_CONDITIONS
        for perturbation_id, lateral_offset_m, yaw_offset_rad in PERTURBATIONS
    )
    return validate_stage_c_scored_trials(trials)


@dataclass(frozen=True)
class RouteDefinition:
    scene_id: str
    route_id: str
    waypoints_xz_holden: tuple[tuple[float, float], ...]
    landing_hold_seconds: float

    def __post_init__(self) -> None:
        for label, value in (("scene_id", self.scene_id), ("route_id", self.route_id)):
            if type(value) is not str or not value or "\x00" in value:
                raise ContractError(f"route {label} must be a nonempty string")
        if (
            type(self.waypoints_xz_holden) is not tuple
            or len(self.waypoints_xz_holden) < 2
            or any(
                type(point) is not tuple or len(point) != 2
                for point in self.waypoints_xz_holden
            )
        ):
            raise ContractError(
                "route requires at least two immutable waypoint tuples"
            )
        waypoints: list[tuple[float, float]] = []
        minimum_normal = float(np.finfo(np.float32).tiny)
        for index, point in enumerate(self.waypoints_xz_holden):
            values = tuple(
                _binary32(value, f"route waypoint {index} finite binary32")
                for value in point
            )
            if any(value != 0.0 and abs(value) < minimum_normal for value in values):
                raise ContractError(
                    f"route waypoint {index} must be finite binary32 normal or zero"
                )
            waypoints.append((values[0], values[1]))
        for start, stop in zip(waypoints[:-1], waypoints[1:], strict=True):
            _segment(start, stop)
        hold = _binary32(
            self.landing_hold_seconds,
            "route landing hold finite binary32",
        )
        if hold < 0.0 or (hold > 0.0 and len(waypoints) < 4):
            raise ContractError("route landing hold is invalid")
        object.__setattr__(self, "waypoints_xz_holden", tuple(waypoints))
        object.__setattr__(self, "landing_hold_seconds", hold)


@dataclass(frozen=True)
class RouteFrameSample:
    frame_index: int
    waypoint_index: int
    velocity_holden: tuple[float, float, float]
    complete: bool = False


def _object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"route JSON contains duplicate key: {key}")
        output[key] = value
    return output


def _exact_mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ContractError(f"{label} keys do not match the registered contract")
    return value


def _binary32(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ContractError(f"{label} must be finite binary32")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(value)
    if not np.isfinite(converted) or float(converted) != float(value):
        raise ContractError(f"{label} must be exact binary32")
    result = float(converted)
    return 0.0 if result == 0.0 else result


def _read_regular_single_link(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    if path.name in ("", ".", ".."):
        raise ContractError(f"{label} must be a regular single-link file")
    parent_descriptor = _open_directory_no_symlinks(path.parent, label=label)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        os.close(parent_descriptor)
        raise ContractError(f"{label} must be a regular single-link file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ContractError(f"{label} must be a regular single-link file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise ContractError(f"{label} exceeds {maximum_bytes} bytes")
        return b"".join(chunks)
    except OSError as error:
        raise ContractError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _open_directory_no_symlinks(path: Path, *, label: str) -> int:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if ".." in candidate.parts:
        raise ContractError(f"{label} requires a symlink-free directory chain")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open("/", flags)
        for part in candidate.parts[1:]:
            if part in ("", "."):
                continue
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ContractError(
            f"{label} requires a symlink-free directory chain"
        ) from error


def load_registered_route(scene_json: str | Path, route_id: str) -> RouteDefinition:
    """Load one protected route without rewriting or normalizing its source JSON."""

    if type(route_id) is not str or not route_id:
        raise ContractError("route_id must be a nonempty string")
    path = Path(scene_json)
    raw = _read_regular_single_link(
        path,
        label="registered scene JSON",
        maximum_bytes=16 * 1024 * 1024,
    )
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"invalid route JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid registered scene JSON: {path}") from error
    if type(document) is not dict:
        raise ContractError("registered scene JSON must contain an object")
    scene_id = document.get("id")
    routes = document.get("routes")
    if type(scene_id) is not str or not scene_id or type(routes) is not list:
        raise ContractError("registered scene JSON has invalid scene/route identity")
    matches = [
        route
        for route in routes
        if type(route) is dict and route.get("id") == route_id
    ]
    if len(matches) != 1:
        raise ContractError(
            f"registered scene {scene_id} must contain exactly one route {route_id}"
        )
    route = _exact_mapping(
        matches[0],
        {
            "id",
            "waypoints_xz",
            "expected_outcome",
            "walkability_class",
            "landing_hold_seconds",
        },
        "registered route",
    )
    if route["expected_outcome"] != "traverse" or route["walkability_class"] != 1:
        raise ContractError("registered scored route must remain traverse/class 1")
    raw_waypoints = route["waypoints_xz"]
    if type(raw_waypoints) is not list or len(raw_waypoints) < 2:
        raise ContractError("registered route requires at least two waypoints")
    waypoints: list[tuple[float, float]] = []
    for index, point in enumerate(raw_waypoints):
        if type(point) is not list or len(point) != 2:
            raise ContractError(f"route waypoint {index} must contain x/z")
        waypoints.append(
            (
                _binary32(point[0], f"route waypoint {index} x"),
                _binary32(point[1], f"route waypoint {index} z"),
            )
        )
    hold = _binary32(route["landing_hold_seconds"], "landing_hold_seconds")
    if hold < 0.0 or (hold > 0.0 and len(waypoints) < 4):
        raise ContractError("registered route landing hold is invalid")
    return RouteDefinition(
        scene_id=scene_id,
        route_id=route_id,
        waypoints_xz_holden=tuple(waypoints),
        landing_hold_seconds=hold,
    )


def _f32(value: float) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        converted = np.float32(value)
    if not np.isfinite(converted):
        raise ContractError("route binary32 arithmetic overflowed")
    result = float(converted)
    return 0.0 if result == 0.0 else result


def _segment(
    start: tuple[float, float], stop: tuple[float, float]
) -> tuple[float, float, float]:
    dx = _f32(_f32(stop[0]) - _f32(start[0]))
    dz = _f32(_f32(stop[1]) - _f32(start[1]))
    dx2 = _f32(dx * dx)
    dz2 = _f32(dz * dz)
    squared = _f32(dx2 + dz2)
    length = _f32(math.sqrt(squared))
    if length <= 1.0e-6:
        raise ContractError("route contains a degenerate segment")
    return dx, dz, length


def _route_parts(
    route: RouteDefinition,
) -> tuple[tuple[int, int, tuple[float, float, float]], ...]:
    dt = _f32(1.0 / SOURCE_RATE_HZ)
    speed = _f32(ROUTE_SPEED_MPS)
    maximum_step = _f32(speed * dt)
    parts: list[tuple[int, int, tuple[float, float, float]]] = []
    for index, (start, stop) in enumerate(
        zip(
            route.waypoints_xz_holden[:-1],
            route.waypoints_xz_holden[1:],
            strict=True,
        )
    ):
        dx, dz, length = _segment(start, stop)
        ratio = _f32(length / maximum_step)
        frames = int(math.ceil(ratio))
        unit_x = _f32(dx / length)
        unit_z = _f32(dz / length)
        command = (_f32(speed * unit_x), 0.0, _f32(speed * unit_z))
        parts.append((frames, index + 1, command))
        if index + 1 == 2 and route.landing_hold_seconds > 0.0:
            hold_frames = int(
                math.ceil(_f32(_f32(route.landing_hold_seconds) / dt))
            )
            parts.append((hold_frames, 2, (0.0, 0.0, 0.0)))
    return tuple(parts)


def route_frame_schedule(route: RouteDefinition) -> tuple[RouteFrameSample, ...]:
    if not isinstance(route, RouteDefinition):
        raise ContractError("route schedule requires a RouteDefinition")
    frames: list[RouteFrameSample] = []
    cursor = 0
    for count, waypoint, command in _route_parts(route):
        for _ in range(count):
            frames.append(
                RouteFrameSample(
                    frame_index=cursor,
                    waypoint_index=waypoint,
                    velocity_holden=command,
                )
            )
            cursor += 1
    return tuple(frames)


def _heading_for_velocity(
    velocity_mujoco: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    yaw = math.atan2(velocity_mujoco[1], velocity_mujoco[0])
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def compile_route_commands(route: RouteDefinition) -> tuple[CommandSample, ...]:
    """Mean each ten-frame route block, padding only its final partial block."""

    frames = route_frame_schedule(route)
    if not frames:
        raise ContractError("registered route produced no command frames")
    commands: list[CommandSample] = []
    preceding_heading = (1.0, 0.0, 0.0, 0.0)
    for begin in range(0, len(frames), CHUNK_INTERVALS):
        block = list(frames[begin : begin + CHUNK_INTERVALS])
        padded = [sample.velocity_holden for sample in block]
        padded.extend([(0.0, 0.0, 0.0)] * (CHUNK_INTERVALS - len(padded)))
        mean_x = _f32(sum(value[0] for value in padded) / CHUNK_INTERVALS)
        mean_z = _f32(sum(value[2] for value in padded) / CHUNK_INTERVALS)
        target_velocity = (mean_x, _f32(-mean_z), 0.0)
        if math.hypot(target_velocity[0], target_velocity[1]) > 0.0:
            preceding_heading = _heading_for_velocity(target_velocity)
        commands.append(
            CommandSample(
                chunk_index=len(commands),
                requested_velocity_mujoco=target_velocity,
                desired_heading_mujoco_wxyz=preceding_heading,
            )
        )
    return tuple(commands)


def flat_command_script() -> tuple[CommandSample, ...]:
    commands: list[CommandSample] = []
    for index in range(30):
        if index < 5:
            yaw = 0.0
            speed = 0.0
        elif index < 15:
            yaw = 0.0
            speed = 0.5
        elif index < 25:
            yaw = math.radians(4.5 * (index - 14))
            speed = 0.5
        else:
            yaw = math.radians(45.0)
            speed = 0.0
        commands.append(
            CommandSample(
                chunk_index=index,
                requested_velocity_mujoco=(
                    speed * math.cos(yaw),
                    speed * math.sin(yaw),
                    0.0,
                ),
                desired_heading_mujoco_wxyz=(
                    math.cos(0.5 * yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * yaw),
                ),
            )
        )
    return tuple(commands)


def _validated_commands(commands: Iterable[CommandSample]) -> tuple[CommandSample, ...]:
    values = tuple(commands)
    if not values or any(not isinstance(value, CommandSample) for value in values):
        raise ContractError("command script requires immutable CommandSample values")
    if tuple(value.chunk_index for value in values) != tuple(range(len(values))):
        raise ContractError("command script chunk indices must be exactly 0..N-1")
    return values


def command_script_bytes(
    *,
    scene_id: str,
    route_id: str,
    commands: Sequence[CommandSample],
) -> bytes:
    if type(scene_id) is not str or not scene_id:
        raise ContractError("command script scene_id must be nonempty")
    if type(route_id) is not str or not route_id:
        raise ContractError("command script route_id must be nonempty")
    values = _validated_commands(commands)
    document = {
        "schema": COMMAND_SCRIPT_SCHEMA,
        "scene_id": scene_id,
        "route_id": route_id,
        "chunk_intervals": CHUNK_INTERVALS,
        "source_rate_hz": SOURCE_RATE_HZ,
        "target_rate_hz": TARGET_RATE_HZ,
        "chunk_count": len(values),
        "duration_s": len(values) * CHUNK_INTERVALS / SOURCE_RATE_HZ,
        "commands": [
            {
                "chunk_index": command.chunk_index,
                "requested_velocity_mujoco": list(
                    command.requested_velocity_mujoco
                ),
                "desired_heading_mujoco_wxyz": list(
                    command.desired_heading_mujoco_wxyz
                ),
            }
            for command in values
        ],
    }
    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError(
            "command script cannot be serialized canonically"
        ) from error


def command_script_sha256(**fields: object) -> str:
    return hashlib.sha256(command_script_bytes(**fields)).hexdigest()


def freeze_command_script(
    path: str | Path,
    *,
    scene_id: str,
    route_id: str,
    commands: Sequence[CommandSample],
) -> str:
    """Create once, or prove an existing frozen script is byte-identical."""

    destination = Path(path)
    encoded = command_script_bytes(
        scene_id=scene_id,
        route_id=route_id,
        commands=commands,
    )
    if destination.name in ("", ".", ".."):
        raise ContractError("frozen command path requires a file name")
    parent_descriptor = -1
    descriptor = -1
    try:
        parent_descriptor = _open_directory_no_symlinks(
            destination.parent,
            label="frozen command",
        )
        descriptor = os.open(
            destination.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=parent_descriptor,
        )
    except FileExistsError:
        try:
            descriptor = os.open(
                destination.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o222
            ):
                raise ContractError(
                    "frozen command must be a read-only regular single-link file"
                )
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > len(encoded):
                    break
            existing = b"".join(chunks)
        except OSError as error:
            raise ContractError(
                "frozen command must be a read-only regular single-link file"
            ) from error
        if existing != encoded:
            raise ContractError("existing frozen command script changed")
    except OSError as error:
        raise ContractError("cannot create frozen command script") from error
    else:
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("zero-byte frozen command write")
                view = view[written:]
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o222
            ):
                raise ContractError(
                    "frozen command must be a read-only regular single-link file"
                )
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        except OSError as error:
            raise ContractError("cannot write frozen command script") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    return hashlib.sha256(encoded).hexdigest()
