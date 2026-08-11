"""MuJoCo viewer and headless CLI for the hybrid terrain G1 LMM."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import struct
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hybrid_terrain_lmm_runtime import (
    CommandState,
    HybridMatcher,
    HybridRuntimeState,
    TerrainAuthority,
    _scripted_command,
)
from .hybrid_terrain_lmm_gpu_search import configure_single_gpu_visibility

LABEL = (
    "HYBRID TERRAIN LMM POC (EXACT SEARCH + LEARNED GENERATOR; SUPPORTED TERRAIN ONLY)"
)
DIAGNOSTIC_LABEL = (
    "HYBRID TERRAIN LMM POC "
    "(DIAGNOSTIC CAPPED SEARCH + LEARNED GENERATOR; NOT ACCEPTANCE EVIDENCE)"
)
DIAGNOSTIC_TERRAIN_LABEL = (
    "HYBRID TERRAIN LMM POC "
    "(DIAGNOSTIC UNAUTHENTICATED TERRAIN; NOT ACCEPTANCE EVIDENCE)"
)
DIAGNOSTIC_MODEL_LABEL = (
    "HYBRID TERRAIN LMM POC (DIAGNOSTIC UNVERIFIED GENERATOR; NOT ACCEPTANCE EVIDENCE)"
)
_DISPLAY_POSTPROCESSOR_IDENTITY = "existing-pose-inertializer-repair-foot-lock/v2"
_DISPLAY_POSTPROCESSOR_POLICY = (
    "PoseInertializer + G1TerrainPoseRepair + G1TerrainFootLock"
)
_FORMAL_ARTIFACT_AUTHORITIES = {
    "cache_manifest_sha256": (
        "084c168b473e730ec24419a49f4be1526226e5f95a2dd81ae4d367c729889cdb"
    ),
    "model_manifest_sha256": (
        "f200db342dd514df6deb6203f3be014054efd4f9038dc0486cc3a4b274a6d826"
    ),
    "g1_xml_sha256": (
        "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
    ),
    "g1_asset_inventory_sha256": (
        "47dcad0d533434233e7769486ae79074751be9eeb583af39f38e219c96c71a28"
    ),
    "corpus_row_count": 3_973_057,
    "corpus_range_count": 15_833,
    "corpus_train_row_count": 3_575_813,
    "corpus_evaluation_row_count": 397_244,
    "generator_loader_authority": True,
    "total_safe_row_count": 3_957_224,
    "total_safe_family_counts": {
        "flat": 18_174,
        "curb": 440_481,
        "slope": 462_393,
        "stair": 3_036_176,
    },
}


def _formal_artifact_authorities(identity: object) -> bool:
    getter = getattr(identity, "get", lambda *_args: None)
    return all(
        getter(name) == expected
        for name, expected in _FORMAL_ARTIFACT_AUTHORITIES.items()
    )


def _runtime_label(
    matcher: HybridMatcher,
    terrain: SceneTerrainAdapter | None = None,
    *,
    scene_authentication_current: bool | None = None,
    search_acceptance_current: bool | None = None,
    formal_authorities_current: bool | None = None,
) -> str:
    if (
        getattr(matcher, "search_backend_identity", "cpu-ckdtree-exact")
        != "cpu-ckdtree-exact"
    ):
        return DIAGNOSTIC_LABEL
    search_eligible = (
        matcher.search_acceptance_eligible
        if search_acceptance_current is None
        else search_acceptance_current
    )
    if not search_eligible:
        return DIAGNOSTIC_LABEL
    if not _generator_acceptance_status(getattr(matcher, "generator", None))[
        "accepted"
    ]:
        return DIAGNOSTIC_MODEL_LABEL
    if formal_authorities_current is False:
        return DIAGNOSTIC_MODEL_LABEL
    if terrain is not None and (
        not terrain.scene_authenticated or scene_authentication_current is False
    ):
        return DIAGNOSTIC_TERRAIN_LABEL
    return LABEL


DEFAULT_TERRAIN_ROOT = Path(
    "/home/ubuntu/projects/motion-matching/resources/g1_terrain/scenes"
)
DEFAULT_G1_XML = Path(
    "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
)


class KeyboardCommandSource:
    """Thread-safe keyboard levels with edge-triggered reset."""

    _LOGICAL_KEYS = {
        "w": "w",
        "a": "a",
        "s": "s",
        "d": "d",
        " ": " ",
        "arrow-up": "w",
        "arrow-down": "s",
        "arrow-left": "a",
        "arrow-right": "d",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._reset = False
        self._stopped = False

    @staticmethod
    def _key(character: str | None) -> str:
        value = character.lower() if isinstance(character, str) else ""
        return " " if value == "space" else value

    def press(self, character: str | None = None, *, escape: bool = False) -> None:
        value = self._key(character)
        with self._lock:
            newly_pressed = value not in self._pressed
            if value in self._LOGICAL_KEYS or value == "r":
                self._pressed.add(value)
            if value == "r" and newly_pressed:
                self._reset = True
            if value in {"x", "q"} or escape:
                self._stopped = True

    def release(self, character: str | None = None) -> None:
        value = self._key(character)
        with self._lock:
            self._pressed.discard(value)

    def snapshot(self) -> tuple[CommandState, bool, bool]:
        with self._lock:
            logical_pressed = {
                self._LOGICAL_KEYS[value]
                for value in self._pressed
                if value in self._LOGICAL_KEYS
            }
            command = CommandState.from_keyboard(logical_pressed)
            reset = self._reset
            self._reset = False
            return command, reset, self._stopped

    def hard_stop_active(self) -> bool:
        """Return the level-triggered Space state used above every controller."""

        with self._lock:
            return " " in self._pressed


def _select_control_command(
    *,
    keyboard_command: CommandState,
    gamepad_command: CommandState,
    gamepad_connected: bool,
    hard_stop: bool,
) -> CommandState:
    if hard_stop:
        return CommandState()
    if gamepad_connected and gamepad_command != CommandState():
        return gamepad_command
    return keyboard_command


def _keyboard_command_token(key: object, keyboard_module: object) -> str | None:
    specials = (
        (getattr(keyboard_module.Key, "up", None), "arrow-up"),
        (getattr(keyboard_module.Key, "down", None), "arrow-down"),
        (getattr(keyboard_module.Key, "left", None), "arrow-left"),
        (getattr(keyboard_module.Key, "right", None), "arrow-right"),
        (getattr(keyboard_module.Key, "space", None), " "),
    )
    for special, token in specials:
        if special is not None and key == special:
            return token
    return getattr(key, "char", None)


def handle_key_press(
    keys: KeyboardCommandSource,
    character: str | None,
    *,
    escape: bool,
) -> bool | None:
    """Listener seam that never consumes the reset edge via ``snapshot``."""

    keys.press(character, escape=escape)
    return (
        False
        if escape or (isinstance(character, str) and character.lower() in {"x", "q"})
        else None
    )


class EvdevCommandSource:
    """Normalize an optional evdev-like axis provider, failing neutral."""

    def __init__(self, device: object | None, *, deadzone: float = 0.2) -> None:
        zone = float(deadzone)
        if not math.isfinite(zone) or not 0.0 <= zone < 1.0:
            raise ValueError("gamepad deadzone must be in [0, 1)")
        self.device = device
        self.deadzone = zone
        self.connected = device is not None

    def snapshot(self) -> CommandState:
        if not self.connected or self.device is None:
            return CommandState()
        try:
            speed_axis, steering_axis = self.device.read_axes()
            speed = float(np.clip(float(speed_axis), -1.0, 1.0))
            steering = float(np.clip(float(steering_axis), -1.0, 1.0))
            if not math.isfinite(speed) or not math.isfinite(steering):
                raise ValueError("non-finite evdev axis")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            self.connected = False
            return CommandState()
        if abs(speed) < self.deadzone:
            speed = 0.0
        if abs(steering) < self.deadzone:
            steering = 0.0
        return CommandState.from_gamepad(speed_axis=speed, steering_axis=steering)


class _LinuxEvdevAxes:
    """Nonblocking left-stick adapter; the last complete axis state is retained."""

    def __init__(self, path: str) -> None:
        evdev = importlib.import_module("evdev")
        self._evdev = evdev
        self._device = evdev.InputDevice(path)
        self._x = 0.0
        self._y = 0.0
        self._ranges: dict[int, tuple[float, float]] = {}
        for code in (evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y):
            info = self._device.absinfo(code)
            self._ranges[code] = (float(info.min), float(info.max))

    def _normalize(self, code: int, value: float) -> float:
        lower, upper = self._ranges[code]
        if upper <= lower:
            raise OSError("evdev absolute axis has an invalid range")
        return float(np.clip(2.0 * (value - lower) / (upper - lower) - 1.0, -1.0, 1.0))

    def read_axes(self) -> tuple[float, float]:
        try:
            events = self._device.read()
        except BlockingIOError:
            events = ()
        for event in events:
            if event.type != self._evdev.ecodes.EV_ABS:
                continue
            if event.code == self._evdev.ecodes.ABS_X:
                self._x = self._normalize(event.code, event.value)
            elif event.code == self._evdev.ecodes.ABS_Y:
                self._y = self._normalize(event.code, event.value)
        return self._y, self._x


def optional_evdev_source(
    path: str | None,
    *,
    factory: Callable[[str], object] = _LinuxEvdevAxes,
) -> EvdevCommandSource:
    if path is None:
        return EvdevCommandSource(None)
    try:
        device = factory(path)
    except (ModuleNotFoundError, OSError, RuntimeError, ValueError):
        device = None
    return EvdevCommandSource(device)


class FixedRateAccumulator:
    """Advance a fixed-rate matcher from a variable-rate render clock."""

    def __init__(self, *, step_hz: float = 25.0) -> None:
        rate = float(step_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("fixed matcher rate must be positive and finite")
        self.step_dt = 1.0 / rate
        self.remainder = 0.0

    def advance(self, elapsed: float, callback: Callable[[float], object]) -> int:
        value = float(elapsed)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("render elapsed time must be finite and nonnegative")
        if not callable(callback):
            raise TypeError("fixed-step callback must be callable")
        # Prevent a paused debugger/window from requesting an unbounded catch-up.
        self.remainder += min(value, 0.25)
        count = 0
        tolerance = np.finfo(np.float64).eps * 8.0
        while self.remainder + tolerance >= self.step_dt:
            callback(self.step_dt)
            self.remainder -= self.step_dt
            count += 1
        return count


def _runtime_step_hz(matcher: object) -> float:
    """Preserve legacy 25 Hz viewers while honoring authenticated new runtimes."""

    rate = float(getattr(matcher, "fps", 25.0))
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("matcher runtime FPS must be positive and finite")
    return rate


def reset_without_mutation_on_failure(
    current: object, build_reset: Callable[[], object]
) -> tuple[object, BaseException | None]:
    """Publish a separately prepared reset only after preparation succeeds."""

    try:
        proposed = build_reset()
    except BaseException as error:  # noqa: BLE001 - reset is transactional on cancellation
        return current, error
    return proposed, None


@dataclass(frozen=True)
class SceneTerrainAdapter:
    """G1HF/v2 scene grid used by both runtime queries and render geometry."""

    authority: TerrainAuthority
    source_heights: np.ndarray
    origin_x: float
    origin_z: float
    cell_size_m: float
    exterior_height_m: float
    source_path: Path | None = None
    spawn_native_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    spawn_heading: float = 0.0
    scene_id: str = "generated"
    terrain_sha256: str = ""
    scene_json_sha256: str | None = None
    generated_grid_sha256: str | None = None
    scene_authenticated: bool = False
    scene_evidence_status: str = "diagnostic-unindexed"
    source_manifest_sha256: str | None = None
    scene_index_sha256: str | None = None
    scene_index_scene_sha256: str | None = None

    def render_heights(self) -> np.ndarray:
        return np.array(self.source_heights, dtype=np.float32, copy=True)

    def native_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        rows, columns = self.source_heights.shape
        xs = self.origin_x + np.arange(columns, dtype=np.float64) * self.cell_size_m
        zs = self.origin_z + np.arange(rows, dtype=np.float64) * self.cell_size_m
        vertices = np.asarray(
            [
                (x, -z, self.source_heights[j, i])
                for j, z in enumerate(zs)
                for i, x in enumerate(xs)
            ],
            dtype=np.float64,
        )
        faces: list[tuple[int, int, int]] = []
        for row in range(rows - 1):
            for column in range(columns - 1):
                a = row * columns + column
                b = a + 1
                c = a + columns
                d = c + 1
                faces.extend(((a, d, b), (a, c, d)))
        return vertices, np.asarray(faces, dtype=np.int32)


def _scene_adapter_from_grid(
    heights: np.ndarray,
    *,
    origin_x: float,
    origin_z: float,
    cell: float,
    exterior: float,
    name: str,
    source_path: Path | None,
    spawn_native_xy: object = (0.0, 0.0),
    spawn_heading: float = 0.0,
    scene_id: str | None = None,
    terrain_sha256: str | None = None,
    scene_json_sha256: str | None = None,
    scene_authenticated: bool = False,
    scene_evidence_status: str | None = None,
    source_manifest_sha256: str | None = None,
    scene_index_sha256: str | None = None,
    scene_index_scene_sha256: str | None = None,
) -> SceneTerrainAdapter:
    grid = np.asarray(heights, dtype=np.float32)
    if grid.ndim != 2 or min(grid.shape) < 2 or not np.isfinite(grid).all():
        raise ValueError("scene heightfield must be a finite two-dimensional grid")
    rows, columns = grid.shape

    def domain_contains(native_xy: np.ndarray) -> bool:
        # Source G1HF is Holden X/Z; native MuJoCo XY is X/-Z.
        x = float(native_xy[0])
        z = -float(native_xy[1])
        u = (x - origin_x) / cell
        v = (z - origin_z) / cell
        return 0.0 <= u <= columns - 1 and 0.0 <= v <= rows - 1

    def height_at(native_xy: np.ndarray) -> float:
        # Source G1HF is Holden X/Z; native MuJoCo XY is X/-Z.
        x = float(native_xy[0])
        z = -float(native_xy[1])
        u = (x - origin_x) / cell
        v = (z - origin_z) / cell
        if u < 0.0 or v < 0.0 or u > columns - 1 or v > rows - 1:
            return exterior
        i0, j0 = math.floor(u), math.floor(v)
        i1, j1 = min(i0 + 1, columns - 1), min(j0 + 1, rows - 1)
        fu, fv = u - i0, v - j0
        a = float(grid[j0, i0])
        b = float(grid[j0, i1])
        c = float(grid[j1, i0])
        d = float(grid[j1, i1])
        # Both query and native_mesh use the a-to-d fixed cell diagonal.
        return (
            a + fu * (b - a) + fv * (d - b)
            if fu >= fv
            else a + fv * (c - a) + fu * (d - c)
        )

    frozen = np.array(grid, copy=True)
    frozen.setflags(write=False)
    spawn = np.asarray(spawn_native_xy, dtype=np.float64)
    if spawn.shape != (2,) or not np.isfinite(spawn).all():
        raise ValueError("scene spawn must be one finite native XY point")
    spawn = np.array(spawn, copy=True)
    spawn.setflags(write=False)
    yaw = float(spawn_heading)
    if not math.isfinite(yaw):
        raise ValueError("scene spawn yaw must be finite")
    grid_identity = hashlib.sha256(
        json.dumps(
            {
                "shape": list(grid.shape),
                "origin_x": float(origin_x),
                "origin_z": float(origin_z),
                "cell_size_m": float(cell),
                "exterior_height_m": float(exterior),
                "height_bytes_sha256": hashlib.sha256(
                    np.ascontiguousarray(grid, dtype="<f4").tobytes()
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return SceneTerrainAdapter(
        authority=TerrainAuthority(
            height_at, domain_contains=domain_contains, name=name
        ),
        source_heights=frozen,
        origin_x=float(origin_x),
        origin_z=float(origin_z),
        cell_size_m=float(cell),
        exterior_height_m=float(exterior),
        source_path=source_path,
        spawn_native_xy=spawn,
        spawn_heading=math.remainder(yaw, 2.0 * math.pi),
        scene_id=str(scene_id or name),
        terrain_sha256=str(terrain_sha256 or grid_identity),
        scene_json_sha256=scene_json_sha256,
        generated_grid_sha256=grid_identity if source_path is None else None,
        scene_authenticated=bool(scene_authenticated),
        scene_evidence_status=str(
            scene_evidence_status
            or (
                "diagnostic-generated"
                if source_path is None
                else "diagnostic-unindexed"
            )
        ),
        source_manifest_sha256=source_manifest_sha256,
        scene_index_sha256=scene_index_sha256,
        scene_index_scene_sha256=scene_index_scene_sha256,
    )


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _authenticated_file(
    path: Path, expected_sha256: object, label: str, *, expected_size: object = None
) -> tuple[bytes, str]:
    if type(expected_sha256) is not str or len(expected_sha256) != 64:
        raise ValueError(f"{label} has no valid SHA-256 authority")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    payload = resolved.read_bytes()
    if expected_size is not None and (
        type(expected_size) is not int or len(payload) != expected_size
    ):
        raise ValueError(f"{label} size changed")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return payload, digest


def _primary_source_receipt(corpus: object) -> tuple[Path, dict[str, object]]:
    receipt = getattr(corpus, "manifest_receipt", None)
    source_root = getattr(corpus, "source_root", None)
    if type(receipt) is not dict:
        raise ValueError("corpus has no source-manifest receipt")
    if receipt.get("schema") == "g1-hybrid-terrain-lmm-source-receipt/v1":
        return Path(source_root).resolve(strict=True), receipt
    if receipt.get("schema") != "g1-hybrid-terrain-lmm-combined-receipt/v2-strict":
        raise ValueError("corpus source-manifest receipt schema is unsupported")
    if set(receipt) != {
        "schema",
        "path",
        "size_bytes",
        "sha256",
        "authorities",
    }:
        raise ValueError("combined corpus receipt shape changed")
    combined_path_value = receipt.get("path")
    if (
        type(combined_path_value) is not str
        or not Path(combined_path_value).is_absolute()
    ):
        raise ValueError("combined cache manifest path must be absolute")
    combined_path = Path(combined_path_value).resolve(strict=True)
    expected_combined_path = (Path(source_root) / "manifest.json").resolve(strict=True)
    if combined_path != expected_combined_path:
        raise ValueError("combined receipt does not identify its cache manifest")
    combined_payload, _ = _authenticated_file(
        combined_path,
        receipt.get("sha256"),
        "combined cache manifest",
        expected_size=receipt.get("size_bytes"),
    )
    combined_manifest = _json_object(combined_payload, "combined cache manifest")
    if combined_manifest.get(
        "schema"
    ) != "g1-hybrid-terrain-lmm-combined-cache/v2-strict" or combined_manifest.get(
        "authorities"
    ) != receipt.get("authorities"):
        raise ValueError("combined cache receipt authority changed")
    authorities = combined_manifest.get("authorities")
    if type(authorities) is not dict or set(authorities) != {
        "primary_cache",
        "pfnn_supplement",
    }:
        raise ValueError("combined cache authority inventory changed")
    primary = authorities.get("primary_cache") if type(authorities) is dict else None
    if type(primary) is not dict or set(primary) != {
        "path",
        "schema",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("combined corpus has no primary cache authority")
    if primary.get("schema") != "g1-hybrid-terrain-lmm-corpus/v2-strict":
        raise ValueError("combined primary cache descriptor schema changed")
    primary_path_value = primary.get("path")
    if (
        type(primary_path_value) is not str
        or not Path(primary_path_value).is_absolute()
    ):
        raise ValueError("combined primary cache manifest path must be absolute")
    primary_path = Path(primary_path_value).resolve(strict=True)
    if primary_path.name != "manifest.json":
        raise ValueError("combined primary cache manifest path changed")
    primary_payload, _ = _authenticated_file(
        primary_path,
        primary.get("sha256"),
        "combined primary cache manifest",
        expected_size=primary.get("size_bytes"),
    )
    primary_manifest = _json_object(primary_payload, "primary cache manifest")
    if primary_manifest.get("schema") != "g1-hybrid-terrain-lmm-corpus/v2-strict":
        raise ValueError("combined primary cache schema changed")
    pfnn = authorities["pfnn_supplement"]
    if type(pfnn) is not dict or set(pfnn) != {
        "path",
        "schema",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("combined PFNN supplement authority changed")
    if pfnn.get("schema") != "pfnn-terrain-lmm-supplement/v1":
        raise ValueError("combined PFNN supplement schema changed")
    pfnn_path_value = pfnn.get("path")
    if type(pfnn_path_value) is not str or not Path(pfnn_path_value).is_absolute():
        raise ValueError("combined PFNN supplement manifest path must be absolute")
    pfnn_path = Path(pfnn_path_value).resolve(strict=True)
    if pfnn_path.name != "manifest.json":
        raise ValueError("combined PFNN supplement manifest path changed")
    pfnn_payload, _ = _authenticated_file(
        pfnn_path,
        pfnn.get("sha256"),
        "combined PFNN supplement manifest",
        expected_size=pfnn.get("size_bytes"),
    )
    pfnn_manifest = _json_object(pfnn_payload, "PFNN supplement manifest")
    if pfnn_manifest.get("schema") != "pfnn-terrain-lmm-supplement/v1":
        raise ValueError("combined PFNN supplement manifest schema changed")
    nested = primary_manifest.get("source_manifest")
    if type(nested) is not dict:
        raise ValueError("primary cache has no source-manifest authority")
    return Path(str(primary_manifest.get("source_root"))).resolve(strict=True), nested


def _indexed_scene_authority(
    corpus: object, scene_id: str
) -> tuple[Path, dict[str, str]]:
    full_walking = type(getattr(corpus, "manifest_receipt", None)) is not dict
    members: dict[str, object] | None = None
    if full_walking:
        source_root_value = getattr(corpus, "root", None)
        source_sha_value = getattr(corpus, "manifest_sha256", None)
        if not isinstance(source_root_value, Path):
            raise ValueError("full walking corpus has no scene authority root")
        source_root = source_root_value.resolve(strict=True)
        manifest_path = (source_root / "manifest.json").resolve(strict=True)
        manifest_payload, source_sha = _authenticated_file(
            manifest_path, source_sha_value, "full walking corpus manifest"
        )
        manifest = _json_object(manifest_payload, "full walking corpus manifest")
        if manifest.get("schema") != "g1-full-walking-terrain-lmm-corpus/v1":
            raise ValueError("full walking corpus schema cannot authorize scenes")
        members_value = manifest.get("members")
        if type(members_value) is not dict:
            raise ValueError("full walking corpus member inventory changed")
        members = members_value
        index_descriptor = manifest.get("scene_index")
        if (
            type(index_descriptor) is not dict
            or set(index_descriptor) != {"path", "scene_ids", "sha256"}
            or index_descriptor.get("path") != "scenes/index.json"
            or type(index_descriptor.get("scene_ids")) is not list
            or scene_id not in index_descriptor.get("scene_ids", ())
        ):
            raise ValueError("full walking scene-index descriptor changed")
        index_member = members.get("scenes/index.json")
        if (
            type(index_member) is not dict
            or index_member.get("path") != "scenes/index.json"
            or index_member.get("sha256") != index_descriptor.get("sha256")
        ):
            raise ValueError("full walking scene-index member changed")
        index_size = index_member.get("size_bytes")
    else:
        source_root, receipt = _primary_source_receipt(corpus)
        manifest_path = (source_root / "manifest.json").resolve(strict=True)
        receipt_path = Path(str(receipt.get("path"))).resolve(strict=True)
        if receipt_path != manifest_path:
            raise ValueError(
                "source receipt does not identify source_root/manifest.json"
            )
        manifest_payload, source_sha = _authenticated_file(
            manifest_path,
            receipt.get("sha256"),
            "source manifest",
            expected_size=receipt.get("size_bytes"),
        )
        manifest = _json_object(manifest_payload, "source manifest")
        if manifest.get("schema") != "g1-terrain-artifacts/v3":
            raise ValueError("source manifest schema cannot authorize terrain scenes")
        index_descriptor = manifest.get("scene_index")
        if (
            type(index_descriptor) is not dict
            or index_descriptor.get("path") != "scenes/index.json"
            or index_descriptor.get("schema") != "g1-terrain-scene-index/v1"
        ):
            raise ValueError("source scene-index descriptor changed")
        index_size = None
    index_path = (source_root / "scenes" / "index.json").resolve(strict=True)
    index_payload, index_sha = _authenticated_file(
        index_path,
        index_descriptor.get("sha256"),
        "source scene index",
        expected_size=index_size,
    )
    index = _json_object(index_payload, "source scene index")
    ids = index.get("scene_ids")
    descriptors = index.get("scenes")
    if (
        index.get("schema") != "g1-terrain-scene-index/v1"
        or index.get("coordinate_signature")
        != "holden-y-up-right-handed-forward-plus-z"
        or type(ids) is not list
        or type(descriptors) is not list
        or len(ids) != len(descriptors)
        or len(set(ids)) != len(ids)
        or scene_id not in ids
    ):
        raise ValueError("requested scene is not registered by the source scene index")
    position = ids.index(scene_id)
    descriptor = descriptors[position]
    expected_path = f"scenes/{scene_id}/scene.json"
    if (
        type(descriptor) is not dict
        or set(descriptor) != {"id", "path", "sha256"}
        or descriptor.get("id") != scene_id
        or descriptor.get("path") != expected_path
    ):
        raise ValueError("source scene descriptor changed")
    scene_path = (source_root / expected_path).resolve(strict=True)
    scene_size = None
    if members is not None:
        scene_member = members.get(expected_path)
        if (
            type(scene_member) is not dict
            or scene_member.get("path") != expected_path
            or scene_member.get("sha256") != descriptor.get("sha256")
        ):
            raise ValueError("full walking scene member changed")
        scene_size = scene_member.get("size_bytes")
    scene_payload, scene_sha = _authenticated_file(
        scene_path,
        descriptor.get("sha256"),
        "source scene.json",
        expected_size=scene_size,
    )
    scene = _json_object(scene_payload, "source scene.json")
    if (
        scene.get("schema") != "g1-terrain-scene/v1"
        or scene.get("id") != scene_id
        or scene.get("coordinate_signature")
        != "holden-y-up-right-handed-forward-plus-z"
        or scene.get("surface_signature") != index.get("surface_signature")
    ):
        raise ValueError("source scene identity differs from its index")
    return scene_path.parent, {
        "source_manifest_sha256": source_sha,
        "scene_index_sha256": index_sha,
        "scene_index_scene_sha256": scene_sha,
    }


def load_scene_terrain(
    scene: str | Path,
    *,
    terrain_root: Path = DEFAULT_TERRAIN_ROOT,
    corpus: object | None = None,
) -> SceneTerrainAdapter:
    value = str(scene)
    if value in {"hills", "analytic-multi-hill"}:
        authority = TerrainAuthority.multi_hill()
        xs, ys, heights = authority.height_grid(
            bounds=(-6.0, 6.0, -8.0, 2.0), shape=(101, 121)
        )
        # Convert native Y samples to monotonically increasing Holden Z.
        return _scene_adapter_from_grid(
            heights[::-1],
            origin_x=float(xs[0]),
            origin_z=float(-ys[-1]),
            cell=float(xs[1] - xs[0]),
            exterior=0.0,
            name=authority.name,
            source_path=None,
        )
    if value == "flat":
        return _scene_adapter_from_grid(
            np.zeros((2, 2), np.float32),
            origin_x=-10.0,
            origin_z=-10.0,
            cell=20.0,
            exterior=0.0,
            name="flat",
            source_path=None,
        )
    authority_identity: dict[str, str] | None = None
    path = Path(scene).expanduser()
    if corpus is not None and not path.exists() and Path(value).name == value:
        path, authority_identity = _indexed_scene_authority(corpus, value)
    elif corpus is not None and path.exists():
        explicit_root = path if path.is_dir() else path.parent
        metadata_path = explicit_root / "scene.json"
        if metadata_path.is_file():
            metadata = _json_object(metadata_path.read_bytes(), "scene.json")
            explicit_id = metadata.get("id")
            if type(explicit_id) is str and explicit_id:
                try:
                    indexed_root, candidate_identity = _indexed_scene_authority(
                        corpus, explicit_id
                    )
                except ValueError:
                    pass
                else:
                    if indexed_root == explicit_root.resolve(strict=True):
                        path = indexed_root
                        authority_identity = candidate_identity
    if not path.exists():
        path = Path(terrain_root).expanduser() / value
    terrain_path = path / "terrain.bin" if path.is_dir() else path
    payload = terrain_path.read_bytes()
    if len(payload) < 32:
        raise ValueError("scene terrain.bin has a truncated G1HF header")
    magic, version, nx, nz, origin_x, origin_z, cell, exterior = struct.unpack(
        "<4sIIIffff", payload[:32]
    )
    expected = 32 + int(nx) * int(nz) * 4
    if (
        magic != b"G1HF"
        or version != 2
        or nx < 2
        or nz < 2
        or expected != len(payload)
        or not math.isfinite(cell)
        or cell <= 0.0
    ):
        raise ValueError("scene terrain.bin is not a valid G1HF/v2 heightfield")
    heights = np.frombuffer(payload, dtype="<f4", offset=32).reshape((nz, nx))
    terrain_sha256 = hashlib.sha256(payload).hexdigest()
    spawn_native_xy = np.zeros(2, dtype=np.float64)
    spawn_heading = 0.0
    scene_id = path.name
    scene_json_sha256 = None
    scene_json_path = path / "scene.json" if path.is_dir() else None
    if scene_json_path is not None and scene_json_path.is_file():
        scene_payload = scene_json_path.read_bytes()
        scene_json_sha256 = hashlib.sha256(scene_payload).hexdigest()
        try:
            descriptor = json.loads(scene_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("scene.json is invalid JSON") from error
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("schema") != "g1-terrain-scene/v1"
            or descriptor.get("coordinate_signature")
            != "holden-y-up-right-handed-forward-plus-z"
        ):
            raise ValueError("scene.json terrain contract is unsupported")
        heightfield = descriptor.get("heightfield")
        if (
            not isinstance(heightfield, dict)
            or heightfield.get("sha256") != terrain_sha256
        ):
            raise ValueError("scene.json does not bind terrain.bin")
        spawn = descriptor.get("spawn")
        if not isinstance(spawn, dict):
            raise ValueError("scene.json has no spawn")
        position = np.asarray(spawn.get("position"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("scene.json spawn position is invalid")
        spawn_native_xy = np.asarray((position[0], -position[2]), np.float64)
        spawn_heading = float(spawn.get("yaw_radians"))
        scene_id = str(descriptor.get("id"))
        if not scene_id or not math.isfinite(spawn_heading):
            raise ValueError("scene.json spawn identity is invalid")
    return _scene_adapter_from_grid(
        heights,
        origin_x=float(origin_x),
        origin_z=float(origin_z),
        cell=float(cell),
        exterior=float(exterior),
        name=path.name,
        source_path=terrain_path.resolve(),
        spawn_native_xy=spawn_native_xy,
        spawn_heading=spawn_heading,
        scene_id=scene_id,
        terrain_sha256=terrain_sha256,
        scene_json_sha256=scene_json_sha256,
        scene_authenticated=authority_identity is not None,
        scene_evidence_status=(
            "authenticated-indexed"
            if authority_identity is not None
            else "diagnostic-unindexed"
        ),
        source_manifest_sha256=(
            authority_identity.get("source_manifest_sha256")
            if authority_identity is not None
            else None
        ),
        scene_index_sha256=(
            authority_identity.get("scene_index_sha256")
            if authority_identity is not None
            else None
        ),
        scene_index_scene_sha256=(
            authority_identity.get("scene_index_scene_sha256")
            if authority_identity is not None
            else None
        ),
    )


def scene_authentication_is_current(
    corpus: object, terrain: SceneTerrainAdapter
) -> bool:
    """Recheck the complete indexed scene authority at evidence time."""

    if not terrain.scene_authenticated or terrain.source_path is None:
        return False
    try:
        indexed_root, identity = _indexed_scene_authority(corpus, terrain.scene_id)
        expected_terrain = (indexed_root / "terrain.bin").resolve(strict=True)
        source_path = terrain.source_path.resolve(strict=True)
        return (
            source_path == expected_terrain
            and _sha256_file(source_path) == terrain.terrain_sha256
            and identity.get("source_manifest_sha256") == terrain.source_manifest_sha256
            and identity.get("scene_index_sha256") == terrain.scene_index_sha256
            and identity.get("scene_index_scene_sha256")
            == terrain.scene_index_scene_sha256
        )
    except (OSError, TypeError, ValueError):
        return False


def overlay_text(
    state: HybridRuntimeState | object,
    *,
    paused: bool,
    scene_evidence_status: str = "diagnostic-unindexed",
    generator_accepted: bool = False,
    search_backend_identity: str | None = None,
    last_search_elapsed_ms: float | None = None,
    first_runtime_search_elapsed_ms: float | None = None,
    diagnostic_mechanical_retained_searchable_row_count: int | None = None,
    diagnostic_mechanical_clearance_bounds_m: tuple[float, float] | None = None,
    display_postprocessor_identity: object | None = None,
) -> tuple[str, str]:
    terrain = np.asarray(state.terrain_features, dtype=np.float64).reshape(4)
    mode = "PAUSED" if paused else str(state.pose_source).upper()
    family_counts = " ".join(
        f"{name}={int(count)}"
        for name, count in getattr(state, "searchable_family_counts", ())
    )
    fallback_count = int(state.fallback_count)
    joint_clamp_count = int(getattr(state, "joint_clamp_count", 0))
    canonical_fallback_count = max(0, fallback_count - joint_clamp_count)
    candidate_rejections = int(getattr(state, "candidate_limit_rejection_count", 0))
    first_rejected_row = getattr(state, "first_candidate_limit_rejection_row", None)
    max_rejections_per_step = int(
        getattr(state, "max_candidate_limit_rejections_per_step", 0)
    )
    search_scope = str(getattr(state, "search_scope", "diagnostic-unknown"))
    searched = int(getattr(state, "searchable_row_count", 0))
    total = int(getattr(state, "total_searchable_row_count", searched))
    exact = search_scope == "full-range-safe-corpus" and searched == total
    mechanical_scope = search_scope in (
        "diagnostic-mechanically-filtered-corpus",
        "diagnostic-mechanically-filtered-cap",
    )
    mechanical_cap = search_scope == "diagnostic-mechanically-filtered-cap"
    mechanical_retained_value = getattr(
        state,
        "diagnostic_mechanical_retained_searchable_row_count",
        diagnostic_mechanical_retained_searchable_row_count,
    )
    mechanical_retained = (
        int(mechanical_retained_value)
        if mechanical_retained_value is not None
        else (searched if mechanical_scope and not mechanical_cap else None)
    )
    authenticated_scene = scene_evidence_status == "authenticated-indexed"
    if exact:
        search_text = f"full exact rows {searched}/{total}"
    elif mechanical_cap:
        retained_label = (
            "unknown" if mechanical_retained is None else mechanical_retained
        )
        search_text = f"diagnostic capped rows {searched}/{retained_label}"
    elif mechanical_scope:
        search_text = f"mechanically filtered rows {mechanical_retained}/{total}"
    else:
        search_text = f"diagnostic capped rows {searched}/{total}"
    mechanical_fields = []
    if mechanical_scope:
        if mechanical_cap and mechanical_retained is not None:
            mechanical_fields.append(
                f"mechanically filtered rows {mechanical_retained}/{total}"
            )
        clearance_bounds = getattr(
            state,
            "diagnostic_mechanical_clearance_bounds_m",
            diagnostic_mechanical_clearance_bounds_m,
        )
        if clearance_bounds is not None:
            lower, upper = (float(value) for value in clearance_bounds)
            mechanical_fields.append(f"clearance {lower:.3f}..{upper:.3f} m")
        diagnostic_pose_rejections = getattr(
            state, "diagnostic_pose_rejection_count", None
        )
        if diagnostic_pose_rejections is not None:
            mechanical_fields.append(
                f"diagnostic pose rejects {int(diagnostic_pose_rejections)}"
            )
        unsupported_holds = getattr(state, "unsupported_hold_count", None)
        if unsupported_holds is not None:
            mechanical_fields.append(f"unsupported holds {int(unsupported_holds)}")
    mechanical_text = ""
    if mechanical_fields:
        mechanical_text = " | ".join(mechanical_fields) + "\n"
    diagnostic_backend = search_backend_identity not in (None, "cpu-ckdtree-exact")
    if diagnostic_backend:
        title = DIAGNOSTIC_LABEL
    elif exact and authenticated_scene and generator_accepted:
        title = LABEL
    elif exact and authenticated_scene:
        title = DIAGNOSTIC_MODEL_LABEL
    else:
        title = DIAGNOSTIC_TERRAIN_LABEL if exact else DIAGNOSTIC_LABEL
    backend_text = ""
    if search_backend_identity is not None:
        latency_fields = []
        if last_search_elapsed_ms is not None:
            latency_fields.append(f"last search {float(last_search_elapsed_ms):.3f} ms")
        if first_runtime_search_elapsed_ms is not None:
            latency_fields.append(
                f"first runtime search {float(first_runtime_search_elapsed_ms):.3f} ms"
            )
        backend_text = f"search backend {search_backend_identity}"
        if latency_fields:
            backend_text += " | " + " | ".join(latency_fields)
        backend_text += "\n"
    postprocessor_title, postprocessor_body = _display_postprocessor_overlay(
        display_postprocessor_identity
    )
    if postprocessor_title:
        title += f" | {postprocessor_title}"
    return title, (
        f"family {state.family} | range {int(state.range_index)} | row {int(state.row)} | "
        f"distance {float(state.search_distance):.6f}\n"
        f"terrain [{', '.join(f'{value:.3f}' for value in terrain)}] | "
        f"decode {int(state.decode_count)} | fallback {fallback_count} | "
        f"canonical {canonical_fallback_count} | clamped {joint_clamp_count} "
        f"max {float(getattr(state, 'max_joint_clamp_magnitude', 0.0)):.6f}\n"
        f"native candidate rejects {candidate_rejections} | "
        f"first {first_rejected_row if first_rejected_row is not None else 'none'} | "
        f"max/step {max_rejections_per_step}\n"
        f"support {state.support_status} | height {float(state.support_height):.3f} | {mode}\n"
        f"search {search_text} {family_counts} | transition penalty "
        f"{float(getattr(state, 'transition_penalty', 0.0)):.3f}\n"
        f"{mechanical_text}"
        f"{backend_text}"
        f"scene evidence {scene_evidence_status}\n"
        f"{postprocessor_body}"
        "Up/Down speed | Left/Right steer | WASD aliases | Space stop | R reset | X/Esc exit"
    )


def _display_postprocessor_overlay(identity: object | None) -> tuple[str, str]:
    getter = getattr(identity, "get", lambda *_args: None)
    if getter("diagnostic_display_postprocessor") != _DISPLAY_POSTPROCESSOR_IDENTITY:
        return "", ""
    half_life = float(getter("inertialization_halflife_s"))
    title = f"{_DISPLAY_POSTPROCESSOR_POLICY} | {half_life:.2f}s half-life"
    body = (
        f"display postprocess {_DISPLAY_POSTPROCESSOR_POLICY} | "
        f"{half_life:.2f}s half-life | "
        f"repair accepted {int(getter('pose_repair_count', 0))} | "
        "lock accept/bypass "
        f"{int(getter('foot_lock_accept_count', 0))}/"
        f"{int(getter('foot_lock_bypass_count', 0))} | "
        f"repair rejected {int(getter('pose_repair_rejection_count', 0))} | "
        f"last reason {getter('last_reason', 'none')}\n"
    )
    return title, body


def _generator_acceptance_status(generator: object) -> dict[str, bool]:
    manifest = getattr(generator, "manifest", {})
    manifest_green = bool(
        getattr(manifest, "get", lambda *_args: False)(
            "canonical_selection_accepted", False
        )
    )
    fit_all_rows = getattr(getattr(generator, "config", None), "fit_all_rows", None)
    status = {
        "canonical_selection_accepted": manifest_green,
        "canonical_selection_verified": (
            getattr(generator, "canonical_selection_verified", None) is True
        ),
        "fit_all_rows": fit_all_rows is True,
        "selection_provenance_verified": (
            getattr(generator, "selection_provenance_verified", None) is True
        ),
    }
    status["accepted"] = all(status.values())
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "view"):
        command = commands.add_parser(name)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--scene", default="ramp-10-up-down")
        command.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
        command.add_argument("--search-rows", type=int, default=None)
        command.add_argument("--transition-penalty", type=float, default=0.1)
        command.add_argument("--receipt", type=Path)
    smoke = commands.choices["smoke"]
    smoke.add_argument("--frames", type=int, default=1_000)
    view = commands.choices["view"]
    view.add_argument("--gamepad")
    view.add_argument("--max-render-frames", type=int, default=0)
    view.add_argument("--search-device")
    return parser


def _load_generator(path: Path, corpus: object) -> object:
    module = importlib.import_module("mm_sonic.hybrid_terrain_lmm_training")
    loader = module.load_hybrid_generator
    try:
        return loader(path, corpus=corpus)
    except TypeError:
        try:
            return loader(path, corpus)
        except TypeError:
            return loader(path)


def _load_matcher(
    arguments: argparse.Namespace,
) -> tuple[HybridMatcher, SceneTerrainAdapter]:
    data_module = importlib.import_module("mm_sonic.hybrid_terrain_lmm_data")
    corpus = data_module.load_hybrid_cache(arguments.cache)
    generator = _load_generator(arguments.model, corpus)
    adapter = load_scene_terrain(arguments.scene, corpus=corpus)
    import mujoco

    native_model = mujoco.MjModel.from_xml_path(str(arguments.g1_xml))
    matcher = HybridMatcher(
        corpus,
        generator,
        adapter.authority,
        native_model=native_model,
        max_search_rows=arguments.search_rows,
        search_device=getattr(arguments, "search_device", None),
        transition_penalty=arguments.transition_penalty,
        initial_root_xy=adapter.spawn_native_xy,
        initial_heading=adapter.spawn_heading,
        cache_manifest_path=Path(arguments.cache) / "manifest.json",
    )
    return matcher, adapter


def build_viewer_model(
    g1_xml: Path,
    terrain: SceneTerrainAdapter,
    *,
    interactive_visuals: bool = False,
):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(g1_xml))
    if interactive_visuals:
        for texture in spec.textures:
            if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
                texture.rgb1[:] = (0.3, 0.5, 0.7)
                texture.rgb2[:] = (0.05, 0.08, 0.12)
        spec.visual.headlight.ambient[:] = (0.3, 0.3, 0.3)
    vertices, faces = terrain.native_mesh()
    spec.add_mesh(
        name="hybrid_lmm_authoritative_terrain",
        uservert=vertices.ravel(),
        userface=faces.ravel(),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="hybrid_lmm_authoritative_terrain",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="hybrid_lmm_authoritative_terrain",
        contype=0,
        conaffinity=0,
        rgba=(0.30, 0.50, 0.22, 1.0),
    )
    model = spec.compile()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_rgba[floor, 3] = 0.0
        model.geom_contype[floor] = 0
        model.geom_conaffinity[floor] = 0
    return model


def build_diagnostic_collision_model(g1_xml: Path, terrain: SceneTerrainAdapter):
    """Build a separate collidable terrain oracle for display pose utilities."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(g1_xml))
    heights = np.asarray(terrain.source_heights, dtype=np.float64)
    if np.all(heights == heights.flat[0]):
        spec.worldbody.add_geom(
            name="terrain_hybrid_lmm_authoritative",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=(0.0, 0.0, float(heights.flat[0])),
            size=(0.0, 0.0, 0.05),
            contype=1,
            conaffinity=1,
        )
    else:
        vertices, faces = terrain.native_mesh()
        spec.add_mesh(
            name="terrain_hybrid_lmm_authoritative_mesh",
            uservert=vertices.ravel(),
            userface=faces.ravel(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
        )
        spec.worldbody.add_geom(
            name="terrain_hybrid_lmm_authoritative",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="terrain_hybrid_lmm_authoritative_mesh",
            contype=1,
            conaffinity=1,
        )
    model = spec.compile()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_contype[floor] = 0
        model.geom_conaffinity[floor] = 0
    return model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _g1_xml_asset_identity(path: Path) -> tuple[str, dict[str, dict[str, object]]]:
    xml_path = Path(path).resolve(strict=True)
    inventory: dict[str, dict[str, object]] = {}
    visited_xml: set[Path] = set()
    active_xml: set[Path] = set()

    def name_for(resolved: Path) -> str:
        if not resolved.is_file():
            raise ValueError("G1 XML references a non-file asset")
        try:
            return resolved.relative_to(xml_path.parent).as_posix()
        except ValueError:
            return str(resolved)

    def add_file(resolved: Path) -> None:
        payload = resolved.read_bytes()
        inventory[name_for(resolved)] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    def walk(
        current: Path,
        inherited_dirs: tuple[Path | None, Path | None, Path | None] | None = None,
    ) -> None:
        resolved_xml = current.resolve(strict=True)
        if resolved_xml in active_xml:
            raise ValueError("G1 XML include graph contains a cycle")
        if resolved_xml in visited_xml:
            return
        active_xml.add(resolved_xml)
        try:
            payload = resolved_xml.read_bytes()
            root = ET.fromstring(payload)
        except (ET.ParseError, OSError) as error:
            raise ValueError("G1 XML asset inventory is unavailable") from error
        compiler = root.find("compiler")
        attributes = compiler.attrib if compiler is not None else {}
        inherited_asset, inherited_mesh, inherited_texture = (
            inherited_dirs if inherited_dirs is not None else (None, None, None)
        )
        explicit_asset = (
            xml_path.parent / attributes["assetdir"]
            if "assetdir" in attributes
            else inherited_asset
        )
        explicit_mesh = (
            xml_path.parent / attributes["meshdir"]
            if "meshdir" in attributes
            else inherited_mesh
        )
        explicit_texture = (
            xml_path.parent / attributes["texturedir"]
            if "texturedir" in attributes
            else inherited_texture
        )
        asset_dir = explicit_asset or resolved_xml.parent
        mesh_dir = explicit_mesh or explicit_asset or resolved_xml.parent
        texture_dir = explicit_texture or explicit_asset or resolved_xml.parent
        explicit_directories = (
            explicit_asset,
            explicit_mesh,
            explicit_texture,
        )
        for element in root.iter():
            file_value = element.attrib.get("file")
            if not file_value:
                continue
            relative = Path(file_value)
            if relative.is_absolute():
                candidate = relative
            elif element.tag == "include":
                candidate = resolved_xml.parent / relative
            elif element.tag == "mesh":
                candidate = mesh_dir / relative
            elif element.tag == "texture":
                candidate = texture_dir / relative
            else:
                candidate = asset_dir / relative
            resolved = candidate.resolve(strict=True)
            add_file(resolved)
            if element.tag == "include":
                walk(resolved, explicit_directories)
        active_xml.remove(resolved_xml)
        visited_xml.add(resolved_xml)

    walk(xml_path)
    payload = (
        json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest(), inventory


def _model_manifest_sha256(generator: object) -> str:
    root = getattr(generator, "root", None)
    if root is not None:
        path = Path(root) / "manifest.json"
        if path.is_file():
            return _sha256_file(path)
    manifest = getattr(generator, "manifest", None)
    if not isinstance(manifest, dict) and not hasattr(manifest, "items"):
        raise ValueError("hybrid generator has no model manifest identity")
    payload = (
        json.dumps(
            dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_manifest_identity(
    matcher: HybridMatcher, expected_sha256: str
) -> tuple[str | None, str | None, bool]:
    path_value = getattr(matcher, "cache_manifest_path", None)
    if path_value is None:
        receipt = getattr(matcher.corpus, "manifest_receipt", None)
        if (
            type(receipt) is dict
            and receipt.get("schema")
            == "g1-hybrid-terrain-lmm-combined-receipt/v2-strict"
        ):
            path_value = receipt.get("path")
    if path_value is None:
        return None, None, False
    try:
        path = Path(path_value).resolve(strict=True)
        digest = _sha256_file(path)
    except (OSError, TypeError, ValueError):
        return str(path_value), None, False
    return str(path), digest, digest == expected_sha256


def _model_manifest_identity(
    generator: object, cache_sha256: str
) -> tuple[str | None, str, bool]:
    manifest = getattr(generator, "manifest", None)
    if not isinstance(manifest, dict) and not hasattr(manifest, "items"):
        raise ValueError("hybrid generator has no model manifest identity")
    expected_payload = (
        json.dumps(
            dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode()
    mapping_digest = hashlib.sha256(expected_payload).hexdigest()
    root = getattr(generator, "root", None)
    if root is None:
        return None, mapping_digest, False
    path = Path(root) / "manifest.json"
    try:
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
    except (OSError, TypeError, ValueError):
        return str(path), mapping_digest, False
    digest = hashlib.sha256(payload).hexdigest()
    published_digest = getattr(generator, "manifest_sha256", None)
    published_current = published_digest is None or published_digest == digest
    corpus_current = dict(manifest).get("corpus_manifest_sha256") == cache_sha256
    return (
        str(resolved),
        digest,
        payload == expected_payload and published_current and corpus_current,
    )


def _search_identity_is_current(matcher: HybridMatcher) -> bool:
    rows = np.asarray(matcher.searchable_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        return False
    if (
        len(rows) == 0
        or int(rows[0]) < 0
        or int(rows[-1]) >= len(matcher.row_ranges)
        or np.any(np.diff(rows) <= 0)
    ):
        return False
    range_indices = matcher.row_ranges[rows]
    range_safe = bool(
        np.all(rows >= matcher.range_starts[range_indices])
        and np.all(rows < matcher.range_stops[range_indices] - 1)
    )
    if not range_safe:
        return False
    digest = hashlib.sha256(
        np.ascontiguousarray(rows, dtype="<i8").tobytes()
    ).hexdigest()
    searched = len(rows)
    expected_total = int(
        np.sum(matcher.range_stops - matcher.range_starts - 1, dtype=np.int64)
    )
    full = searched == expected_total
    if full:
        cursor = 0
        for start, stop in zip(matcher.range_starts, matcher.range_stops, strict=True):
            count = int(stop - start - 1)
            if not np.array_equal(
                rows[cursor : cursor + count],
                np.arange(start, stop - 1, dtype=rows.dtype),
            ):
                full = False
                break
            cursor += count

    def family_counts(ids: np.ndarray, counts: np.ndarray) -> dict[str, int]:
        result: dict[str, int] = {}
        for family, count in zip(ids, counts, strict=True):
            family_id = int(family)
            name = (
                matcher.family_names[family_id]
                if 0 <= family_id < len(matcher.family_names)
                else str(family_id)
            )
            result[name] = int(count)
        return result

    searched_ids, searched_counts = np.unique(
        matcher.family_ids[range_indices], return_counts=True
    )
    range_lengths = matcher.range_stops - matcher.range_starts - 1
    total_ids = np.unique(matcher.family_ids)
    total_counts = np.asarray(
        [
            np.sum(range_lengths[matcher.family_ids == family], dtype=np.int64)
            for family in total_ids
        ]
    )
    expected_scope = "full-range-safe-corpus" if full else "diagnostic-stratified-cap"
    return (
        digest == matcher.search_view_sha256
        and matcher.search_scope == expected_scope
        and matcher.search_acceptance_eligible is full
        and matcher.total_searchable_row_count == expected_total
        and dict(matcher.searchable_family_counts)
        == family_counts(searched_ids, searched_counts)
        and dict(matcher.total_searchable_family_counts)
        == family_counts(total_ids, total_counts)
    )


def _all_row_canonical_evaluation_identity(
    generator: object,
) -> dict[str, object]:
    manifest = getattr(generator, "manifest", {})
    manifest_get = getattr(manifest, "get", lambda *_args: None)
    artifacts = manifest_get("artifacts", {})
    evaluation = (
        artifacts.get("evaluation.json") if isinstance(artifacts, dict) else None
    )
    receipt = getattr(generator, "evaluation_receipt", {})
    receipt_get = getattr(receipt, "get", lambda *_args: None)
    return {
        "fit_all_rows": (
            getattr(getattr(generator, "config", None), "fit_all_rows", None) is True
        ),
        "evaluation_scope": receipt_get("evaluation_scope"),
        "evaluation_artifact_sha256": (
            evaluation.get("sha256") if isinstance(evaluation, dict) else None
        ),
        "selection_evaluation_sha256": manifest_get("selection_evaluation_sha256"),
        "selection_model_manifest_sha256": manifest_get(
            "selection_model_manifest_sha256"
        ),
        "native_joint_limits_evaluated": (
            receipt_get("native_joint_limits_evaluated") is True
        ),
    }


def _generator_has_loader_authority(generator: object, model_sha: str) -> bool:
    from .hybrid_terrain_lmm_training import HybridGenerator

    return (
        type(generator) is HybridGenerator
        and isinstance(getattr(generator, "root", None), Path)
        and getattr(generator, "manifest_sha256", None) == model_sha
    )


def _runtime_identity(
    matcher: HybridMatcher,
    terrain: SceneTerrainAdapter,
    g1_xml: Path,
) -> dict[str, object]:
    cache_sha = getattr(matcher.corpus, "cache_manifest_sha256", None)
    if not isinstance(cache_sha, str) or len(cache_sha) != 64:
        cache_sha = getattr(matcher.corpus, "manifest_sha256", None)
    if not isinstance(cache_sha, str) or len(cache_sha) != 64:
        raise ValueError("hybrid corpus has no exact cache manifest SHA-256")
    corpus_receipt = getattr(matcher.corpus, "manifest_receipt", None)
    strict_combined_corpus = (
        type(corpus_receipt) is dict
        and corpus_receipt.get("schema")
        == "g1-hybrid-terrain-lmm-combined-receipt/v2-strict"
    )
    cache_path, cache_observed_sha, cache_current = _cache_manifest_identity(
        matcher, cache_sha
    )
    model_path, model_sha, model_current = _model_manifest_identity(
        matcher.generator, cache_sha
    )
    asset_inventory_sha, asset_inventory = _g1_xml_asset_identity(Path(g1_xml))
    corpus_artifacts = getattr(matcher.corpus, "artifacts", None)
    corpus_positions = np.asarray(getattr(corpus_artifacts, "positions", ()))
    corpus_range_starts = np.asarray(getattr(corpus_artifacts, "range_starts", ()))
    corpus_train_mask = np.asarray(getattr(matcher.corpus, "train_mask", ()))
    corpus_evaluation_mask = np.asarray(getattr(matcher.corpus, "evaluation_mask", ()))
    generator_loader_authority = _generator_has_loader_authority(
        matcher.generator, model_sha
    )
    return {
        "identity_capture_succeeded": True,
        "strict_combined_corpus": strict_combined_corpus,
        "cache_manifest_sha256": cache_sha,
        "cache_manifest_path": cache_path,
        "cache_manifest_observed_sha256": cache_observed_sha,
        "cache_manifest_authority_current": cache_current,
        "model_manifest_path": model_path,
        "model_manifest_sha256": model_sha,
        "model_manifest_authority_current": model_current,
        "scene_id": terrain.scene_id,
        "scene_terrain_sha256": terrain.terrain_sha256,
        "scene_json_sha256": terrain.scene_json_sha256,
        "scene_generated_grid_sha256": terrain.generated_grid_sha256,
        "scene_evidence_status": terrain.scene_evidence_status,
        "scene_authentication_current": scene_authentication_is_current(
            matcher.corpus, terrain
        ),
        "source_manifest_sha256": terrain.source_manifest_sha256,
        "scene_index_sha256": terrain.scene_index_sha256,
        "scene_index_scene_sha256": terrain.scene_index_scene_sha256,
        "g1_xml_sha256": _sha256_file(Path(g1_xml)),
        "g1_asset_inventory_sha256": asset_inventory_sha,
        "g1_asset_inventory": asset_inventory,
        "corpus_row_count": len(corpus_positions),
        "corpus_range_count": len(corpus_range_starts),
        "corpus_train_row_count": int(np.count_nonzero(corpus_train_mask)),
        "corpus_evaluation_row_count": int(np.count_nonzero(corpus_evaluation_mask)),
        "generator_loader_authority": generator_loader_authority,
        "search_view_sha256": matcher.search_view_sha256,
        "search_scope": matcher.search_scope,
        "total_safe_row_count": matcher.total_searchable_row_count,
        "searched_row_count": len(matcher.searchable_rows),
        "searched_family_counts": dict(matcher.searchable_family_counts),
        "total_safe_family_counts": dict(matcher.total_searchable_family_counts),
        "transition_penalty": matcher.transition_penalty,
        "search_backend_identity": getattr(
            matcher, "search_backend_identity", "cpu-ckdtree-exact"
        ),
        "last_search_elapsed_ms": getattr(matcher, "last_search_elapsed_ms", None),
        "first_runtime_search_elapsed_ms": getattr(
            matcher, "warm_search_elapsed_ms", None
        ),
        "search_identity_current": _search_identity_is_current(matcher),
        "generator_acceptance_status": _generator_acceptance_status(matcher.generator),
        "all_row_canonical_evaluation_identity": (
            _all_row_canonical_evaluation_identity(matcher.generator)
        ),
    }


def _capture_runtime_identity(
    matcher: HybridMatcher,
    terrain: SceneTerrainAdapter,
    g1_xml: Path,
) -> dict[str, object]:
    try:
        return _runtime_identity(matcher, terrain, g1_xml)
    except Exception as error:  # noqa: BLE001 - a rejected receipt records TOCTOU
        return {
            "identity_capture_succeeded": False,
            "identity_capture_error": f"{type(error).__name__}: {error}",
        }


def _search_audit(matcher: HybridMatcher) -> dict[str, int]:
    row_families = matcher.family_ids[matcher.row_ranges[matcher.searchable_rows]]
    samples = 0
    failures = 0
    exact_feature_matches = 0
    for family in np.unique(row_families):
        family_rows = matcher.searchable_rows[row_families == family]
        row = int(family_rows[len(family_rows) // 2])
        query = np.asarray(matcher.features[row], dtype=np.float64)
        tree = matcher.match(query)
        brute = matcher.brute_force_match(query)
        samples += 1
        failures += int(
            tree.row != brute.row
            or not math.isclose(
                tree.distance, brute.distance, rel_tol=0.0, abs_tol=1.0e-12
            )
        )
        exact_feature_matches += int(
            np.array_equal(matcher.features[tree.row], matcher.features[row])
        )
    return {
        "samples": samples,
        "failures": failures,
        "exact_feature_matches": exact_feature_matches,
    }


def run_mujoco_headless_smoke(
    matcher: HybridMatcher,
    terrain: SceneTerrainAdapter,
    *,
    g1_xml: Path = DEFAULT_G1_XML,
    frames: int = 1_000,
) -> dict[str, object]:
    """Run the delivered smoke gate through a real MuJoCo model/forward pass."""

    if not isinstance(matcher, HybridMatcher):
        raise TypeError("MuJoCo smoke requires a HybridMatcher")
    if not isinstance(terrain, SceneTerrainAdapter):
        raise TypeError("MuJoCo smoke requires one scene terrain adapter")
    if type(frames) is not int or frames < 1:
        raise ValueError("MuJoCo smoke frames must be a positive integer")
    import mujoco

    authority_pre = _capture_runtime_identity(matcher, terrain, Path(g1_xml))
    model = build_viewer_model(Path(g1_xml), terrain)
    data = mujoco.MjData(model)
    retrieval_audit = _search_audit(matcher)
    generator_status_value = authority_pre.get("generator_acceptance_status")
    generator_status = (
        generator_status_value
        if isinstance(generator_status_value, dict)
        else {
            "accepted": False,
            "canonical_selection_accepted": False,
            "canonical_selection_verified": False,
            "fit_all_rows": False,
            "selection_provenance_verified": False,
        }
    )
    all_row_evaluation_identity = authority_pre.get(
        "all_row_canonical_evaluation_identity", {}
    )
    initial_fallback = matcher.state.fallback_count
    initial_decode = matcher.state.decode_count
    initial_joint_clamp = matcher.state.joint_clamp_count
    selected_ranges: set[int] = set()
    selected_families: set[str] = set()
    terrain_classes: set[str] = set()
    terrain_rows: list[np.ndarray] = []
    finite = True
    native_limits = True
    native_limit_violation_count = 0
    forward_count = 0
    parity_samples = 0
    parity_failures = 0
    max_joint_clamp_magnitude = 0.0
    failure_frame: int | None = None
    failure_reason: str | None = None
    last_pose_source = matcher.state.pose_source
    parity_frames = {0, max(0, frames // 2)}
    for frame in range(frames):
        try:
            state = matcher.step(_scripted_command(frame, frames), dt=1.0 / 25.0)
            last_pose_source = state.pose_source
            max_joint_clamp_magnitude = max(
                max_joint_clamp_magnitude, state.max_joint_clamp_magnitude
            )
            qpos = np.asarray(state.qpos, dtype=np.float64)
            finite = (
                finite
                and qpos.shape == (int(model.nq),)
                and bool(np.isfinite(qpos).all())
            )
            for joint_id in range(1, int(model.njnt)):
                address = int(model.jnt_qposadr[joint_id])
                lower, upper = np.asarray(model.jnt_range[joint_id], np.float64)
                joint_valid = (
                    float(qpos[address]) >= float(lower) - 1.0e-5
                    and float(qpos[address]) <= float(upper) + 1.0e-5
                )
                native_limit_violation_count += int(not joint_valid)
                native_limits = native_limits and joint_valid
            if not finite or not native_limits:
                failure_frame = frame
                failure_reason = (
                    "runtime qpos was non-finite or outside native joint limits"
                )
                break
            data.qpos[:] = qpos
            data.time = (frame + 1) / 25.0
            mujoco.mj_forward(model, data)
            forward_count += 1
            finite = finite and all(
                np.isfinite(values).all()
                for values in (data.qpos, data.xpos, data.xquat)
            )
            selected_ranges.add(state.range_index)
            selected_families.add(state.family)
            terrain_classes.add(state.terrain_class)
            terrain_rows.append(np.asarray(state.terrain_features, dtype=np.float64))
            if frame in parity_frames:
                tree = matcher.match(state.query)
                brute = matcher.brute_force_match(state.query)
                parity_samples += 1
                parity_failures += int(
                    tree.row != brute.row
                    or not math.isclose(
                        tree.distance,
                        brute.distance,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
            if not finite:
                failure_frame = frame
                failure_reason = "MuJoCo forward produced non-finite state"
                break
        except Exception as error:  # noqa: BLE001 - evidence records arbitrary step faults
            failure_frame = frame
            failure_reason = f"{type(error).__name__}: {error}"
            break
    fallback_count = matcher.state.fallback_count - initial_fallback
    learned_count = matcher.state.decode_count - initial_decode
    joint_clamp_count = matcher.state.joint_clamp_count - initial_joint_clamp
    canonical_fallback_count = max(0, fallback_count - joint_clamp_count)
    crash_count = int(failure_reason is not None)
    terrain_variance = (
        float(np.var(np.stack(terrain_rows), axis=0).max()) if terrain_rows else 0.0
    )
    authority_post = _capture_runtime_identity(matcher, terrain, Path(g1_xml))
    authority_current_fields = (
        "identity_capture_succeeded",
        "cache_manifest_authority_current",
        "model_manifest_authority_current",
        "scene_authentication_current",
        "search_identity_current",
    )
    authority_current = all(
        authority_pre.get(name) is True and authority_post.get(name) is True
        for name in authority_current_fields
    )
    authority_unchanged = authority_pre == authority_post
    scene_authentication_current = (
        authority_pre.get("scene_authentication_current") is True
        and authority_post.get("scene_authentication_current") is True
        and authority_pre.get("scene_terrain_sha256")
        == authority_post.get("scene_terrain_sha256")
        and authority_pre.get("scene_json_sha256")
        == authority_post.get("scene_json_sha256")
    )
    strict_combined_corpus = (
        authority_pre.get("strict_combined_corpus") is True
        and authority_post.get("strict_combined_corpus") is True
    )
    full_search = (
        authority_pre.get("search_identity_current") is True
        and authority_post.get("search_identity_current") is True
        and authority_pre.get("search_scope") == "full-range-safe-corpus"
        and authority_pre.get("searched_row_count")
        == authority_pre.get("total_safe_row_count")
    )
    cpu_search_backend_only = (
        authority_pre.get("search_backend_identity") == "cpu-ckdtree-exact"
        and authority_post.get("search_backend_identity") == "cpu-ckdtree-exact"
    )
    transition_penalty_exact = (
        authority_pre.get("transition_penalty") == 0.1
        and authority_post.get("transition_penalty") == 0.1
    )
    formal_artifact_authorities = _formal_artifact_authorities(
        authority_pre
    ) and _formal_artifact_authorities(authority_post)
    acceptance_failures: list[str] = []
    gates = (
        (frames >= 1_000, "minimum-1000-frames"),
        (forward_count == frames, "all-requested-mujoco-forwards"),
        (finite, "finite-runtime-and-mujoco-state"),
        (
            native_limits and native_limit_violation_count == 0,
            "zero-native-limit-violations",
        ),
        (len(selected_ranges) >= 2, "range-diversity-required"),
        (len(terrain_classes) >= 2, "terrain-class-diversity-required"),
        (terrain_variance > 0.0, "nonflat-terrain-variance-required"),
        (
            parity_samples > 0 and parity_failures == 0,
            "tree-brute-parity",
        ),
        (
            retrieval_audit["samples"] > 0 and retrieval_audit["failures"] == 0,
            "manifest-row-retrieval-audit",
        ),
        (
            retrieval_audit["samples"] > 0
            and retrieval_audit["exact_feature_matches"] == retrieval_audit["samples"],
            "exact-feature-retrieval-required",
        ),
        (full_search, "full-search-required"),
        (cpu_search_backend_only, "cpu-exact-search-backend-required"),
        (strict_combined_corpus, "strict-combined-corpus-required"),
        (
            formal_artifact_authorities,
            "frozen-artifact-authorities-required",
        ),
        (
            authority_pre.get("scene_authentication_current") is True,
            "authenticated-indexed-scene-required",
        ),
        (
            transition_penalty_exact,
            "transition-penalty-exactly-0.1",
        ),
        (learned_count > 0, "learned-decode-required"),
        (fallback_count == 0, "zero-fallback-required"),
        (joint_clamp_count == 0, "zero-native-limit-clamp-required"),
        (failure_reason is None, "zero-crash-required"),
        (
            generator_status["canonical_selection_accepted"]
            and generator_status["canonical_selection_verified"],
            "canonical-selection-green-required",
        ),
        (generator_status["fit_all_rows"], "all-row-refit-required"),
        (
            generator_status["selection_provenance_verified"],
            "selection-provenance-verified-required",
        ),
        (authority_current, "evidence-authorities-current"),
        (authority_unchanged, "evidence-authority-unchanged"),
    )
    for passed, name in gates:
        if not passed:
            acceptance_failures.append(name)
    accepted = not acceptance_failures
    return {
        "schema": "hybrid-terrain-lmm-mujoco-smoke/v2",
        "acceptance_contract": (
            "strict-combined-frozen-authorities-scripted-runtime/v2"
        ),
        "label": _runtime_label(
            matcher,
            terrain,
            scene_authentication_current=scene_authentication_current,
            search_acceptance_current=full_search,
            formal_authorities_current=formal_artifact_authorities,
        ),
        "accepted": accepted,
        "acceptance_failures": acceptance_failures,
        "frames": frames,
        "mujoco_forward_count": forward_count,
        "finite": bool(finite),
        "native_joint_limits": bool(native_limits),
        "native_limit_violation_count": native_limit_violation_count,
        "selected_ranges": sorted(selected_ranges),
        "selected_families": sorted(selected_families),
        "terrain_classes": sorted(terrain_classes),
        "terrain_variance": terrain_variance,
        "tree_brute_parity_samples": parity_samples,
        "tree_brute_parity_failures": parity_failures,
        "retrieval_audit_samples": retrieval_audit["samples"],
        "retrieval_audit_failures": retrieval_audit["failures"],
        "retrieval_exact_feature_matches": retrieval_audit["exact_feature_matches"],
        "range_diversity_count": len(selected_ranges),
        "family_diversity_count": len(selected_families),
        "learned_decode_count": learned_count,
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
        "last_pose_source": last_pose_source,
        "failure_frame": failure_frame,
        "failure_reason": failure_reason,
        "crash_count": crash_count,
        "completed_without_exception": crash_count == 0,
        "canonical_selection_accepted": generator_status[
            "canonical_selection_accepted"
        ],
        "canonical_selection_verified": generator_status[
            "canonical_selection_verified"
        ],
        "fit_all_rows": generator_status["fit_all_rows"],
        "selection_provenance_verified": generator_status[
            "selection_provenance_verified"
        ],
        "strict_combined_corpus": strict_combined_corpus,
        "formal_artifact_authorities": formal_artifact_authorities,
        "model_identity_status": (
            "verified-all-row-refit"
            if generator_status["accepted"]
            else "diagnostic-selection-or-unverified-model"
        ),
        "search_scope": authority_pre.get("search_scope"),
        "search_acceptance_eligible": full_search,
        "cpu_search_backend_only": cpu_search_backend_only,
        "total_safe_row_count": authority_pre.get("total_safe_row_count"),
        "searched_row_count": authority_pre.get("searched_row_count"),
        "searched_family_counts": authority_pre.get("searched_family_counts"),
        "total_safe_family_counts": authority_pre.get("total_safe_family_counts"),
        "search_view_sha256": authority_pre.get("search_view_sha256"),
        "transition_penalty": authority_pre.get("transition_penalty"),
        "native_limit_audit_scope": "scripted-runtime-mujoco-forward-only",
        "full_dataset_native_limit_audit_performed": False,
        "all_row_canonical_evaluation_identity": all_row_evaluation_identity,
        "evidence_authority_current": authority_current,
        "evidence_authority_unchanged": authority_unchanged,
        "evidence_authority_pre": authority_pre,
        "evidence_authority_post": authority_post,
        "identity": authority_pre,
    }


def _load_keyboard_module():
    return importlib.import_module("pynput.keyboard")


def run_interactive(
    matcher: HybridMatcher,
    terrain: SceneTerrainAdapter,
    *,
    g1_xml: Path = DEFAULT_G1_XML,
    gamepad: str | None = None,
    max_render_frames: int = 0,
    formal_authority_predicate: Callable[[object], bool] | None = None,
    runtime_label_resolver: Callable[..., str] | None = None,
    runtime_step_hz_resolver: Callable[[object], float] | None = None,
    runtime_identity_resolver: Callable[
        [HybridMatcher, SceneTerrainAdapter, Path], dict[str, object]
    ]
    | None = None,
    display_postprocessor_factory: Callable[[object, object, object], object]
    | None = None,
) -> dict[str, object]:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("interactive hybrid terrain viewer requires DISPLAY")
    import mujoco
    import mujoco.viewer

    authority_predicate = (
        _formal_artifact_authorities
        if formal_authority_predicate is None
        else formal_authority_predicate
    )
    label_resolver = (
        _runtime_label if runtime_label_resolver is None else runtime_label_resolver
    )
    step_hz_resolver = (
        _runtime_step_hz
        if runtime_step_hz_resolver is None
        else runtime_step_hz_resolver
    )
    identity_resolver = (
        _runtime_identity
        if runtime_identity_resolver is None
        else runtime_identity_resolver
    )

    keyboard_module = _load_keyboard_module()
    keys = KeyboardCommandSource()
    joystick = optional_evdev_source(gamepad)

    def on_press(key: object) -> bool | None:
        escape = key == keyboard_module.Key.esc
        return handle_key_press(
            keys, _keyboard_command_token(key, keyboard_module), escape=escape
        )

    def on_release(key: object) -> None:
        keys.release(_keyboard_command_token(key, keyboard_module))

    listener = keyboard_module.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    model = build_viewer_model(g1_xml, terrain, interactive_visuals=True)
    postprocessor = (
        None
        if display_postprocessor_factory is None
        else display_postprocessor_factory(model, matcher, terrain)
    )
    identity = identity_resolver(matcher, terrain, Path(g1_xml))
    interactive_scene_authentication_current = (
        identity.get("scene_authentication_current") is True
    )
    interactive_formal_authorities_current = authority_predicate(identity)
    interactive_generator_accepted = (
        identity.get("generator_acceptance_status", {}).get("accepted") is True
        and interactive_formal_authorities_current
    )
    overlay_scene_status = terrain.scene_evidence_status
    data = mujoco.MjData(model)
    accumulator = FixedRateAccumulator(step_hz=step_hz_resolver(matcher))
    render_frames = 0
    reset_failures = 0
    started = time.monotonic()
    last_clock = started
    next_scene_authentication_check = started
    display_qpos = np.asarray(matcher.state.qpos)

    def fixed_rate_step(dt: float, command: CommandState) -> None:
        nonlocal display_qpos
        state = matcher.step(command, dt=dt)
        display_qpos = (
            state.qpos
            if postprocessor is None
            else postprocessor.step(
                state.qpos,
                row=state.row,
                range_index=state.range_index,
                source_contact=np.asarray(
                    matcher.artifacts.contacts[state.row], dtype=bool
                ),
                dt_s=dt,
            )
        )

    try:
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                frame_started = time.monotonic()
                elapsed = frame_started - last_clock
                last_clock = frame_started
                keyboard_command, reset, stopped = keys.snapshot()
                if stopped:
                    break
                gamepad_command = joystick.snapshot()
                command = _select_control_command(
                    keyboard_command=keyboard_command,
                    gamepad_command=gamepad_command,
                    gamepad_connected=joystick.connected,
                    hard_stop=keys.hard_stop_active(),
                )
                if reset:
                    previous = matcher.state
                    _, error = reset_without_mutation_on_failure(
                        previous, matcher.reset
                    )
                    reset_failures += int(error is not None)
                    if error is None and postprocessor is not None:
                        postprocessor.reset()
                accumulator.advance(
                    elapsed, lambda dt, command=command: fixed_rate_step(dt, command)
                )
                data.qpos[:] = display_qpos
                mujoco.mj_forward(model, data)
                if (
                    interactive_scene_authentication_current
                    and frame_started >= next_scene_authentication_check
                ):
                    interactive_scene_authentication_current = (
                        scene_authentication_is_current(matcher.corpus, terrain)
                    )
                    next_scene_authentication_check = frame_started + 1.0
                overlay_scene_status = (
                    terrain.scene_evidence_status
                    if interactive_scene_authentication_current
                    else "diagnostic-authentication-not-current"
                )
                display_postprocessor_identity = (
                    None if postprocessor is None else postprocessor.identity()
                )
                title, body = overlay_text(
                    matcher.state,
                    paused=False,
                    scene_evidence_status=overlay_scene_status,
                    generator_accepted=interactive_generator_accepted,
                    search_backend_identity=getattr(
                        matcher, "search_backend_identity", "cpu-ckdtree-exact"
                    ),
                    last_search_elapsed_ms=getattr(
                        matcher, "last_search_elapsed_ms", None
                    ),
                    first_runtime_search_elapsed_ms=getattr(
                        matcher, "warm_search_elapsed_ms", None
                    ),
                    diagnostic_mechanical_retained_searchable_row_count=getattr(
                        matcher,
                        "diagnostic_mechanical_retained_searchable_row_count",
                        None,
                    ),
                    diagnostic_mechanical_clearance_bounds_m=getattr(
                        matcher, "diagnostic_mechanical_clearance_bounds_m", None
                    ),
                    display_postprocessor_identity=display_postprocessor_identity,
                )
                title = label_resolver(
                    matcher,
                    terrain,
                    scene_authentication_current=(
                        interactive_scene_authentication_current
                    ),
                    formal_authorities_current=(interactive_formal_authorities_current),
                )
                postprocessor_title, _postprocessor_body = (
                    _display_postprocessor_overlay(display_postprocessor_identity)
                )
                if postprocessor_title and _DISPLAY_POSTPROCESSOR_POLICY not in title:
                    title += f" | {postprocessor_title}"
                with viewer.lock():
                    viewer.cam.lookat[:] = data.qpos[:3]
                    wireframe = int(mujoco.mjtRndFlag.mjRND_WIREFRAME)
                    viewer.user_scn.flags[wireframe] = 0
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        title,
                        body,
                    )
                )
                viewer.sync()
                render_frames += 1
                if max_render_frames and render_frames >= max_render_frames:
                    break
                remaining = 1.0 / 60.0 - (time.monotonic() - frame_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        listener.stop()
        listener.join(timeout=1.0)
    identity.update(
        {
            "search_backend_identity": getattr(
                matcher, "search_backend_identity", "cpu-ckdtree-exact"
            ),
            "last_search_elapsed_ms": getattr(matcher, "last_search_elapsed_ms", None),
            "first_runtime_search_elapsed_ms": getattr(
                matcher, "warm_search_elapsed_ms", None
            ),
        }
    )
    if postprocessor is not None:
        identity.update(postprocessor.identity())
    return {
        "schema": "hybrid-terrain-lmm-viewer-runtime/v1",
        "label": label_resolver(
            matcher,
            terrain,
            scene_authentication_current=interactive_scene_authentication_current,
            formal_authorities_current=interactive_formal_authorities_current,
        ),
        "evidence_status": "interactive-diagnostic-not-acceptance",
        "acceptance_eligible": False,
        "formal_artifact_authorities": interactive_formal_authorities_current,
        "scene": terrain.authority.name,
        "scene_evidence_status": terrain.scene_evidence_status,
        "overlay_scene_evidence_status": overlay_scene_status,
        "search_scope": matcher.search_scope,
        "render_frames": render_frames,
        "elapsed_s": time.monotonic() - started,
        "matcher_row": matcher.state.row,
        "learned_decode_count": matcher.state.decode_count,
        "fallback_count": matcher.state.fallback_count,
        "candidate_limit_rejection_count": getattr(
            matcher.state, "candidate_limit_rejection_count", 0
        ),
        "first_candidate_limit_rejection_row": getattr(
            matcher.state, "first_candidate_limit_rejection_row", None
        ),
        "max_candidate_limit_rejections_per_step": getattr(
            matcher.state, "max_candidate_limit_rejections_per_step", 0
        ),
        "reset_failures": reset_failures,
        "identity": identity,
    }


def write_receipt_exclusive(target: Path, receipt: dict[str, object]) -> Path:
    """Publish canonical evidence atomically without replacing prior evidence."""

    if type(receipt) is not dict:
        raise TypeError("runtime receipt must be a JSON object")
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    destination = Path(target).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        published = True
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        temporary.unlink()
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if temporary.exists():
            temporary.unlink()
        raise
    if not published:
        raise AssertionError("receipt publication did not complete")
    return destination


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (
        arguments.command == "view"
        and arguments.search_device is not None
        and arguments.search_rows is not None
    ):
        raise ValueError("--search-device cannot be combined with --search-rows")
    if arguments.command == "view" and arguments.search_device is not None:
        configure_single_gpu_visibility(arguments.search_device)
    matcher, terrain = _load_matcher(arguments)
    if arguments.command == "smoke":
        receipt = run_mujoco_headless_smoke(
            matcher,
            terrain,
            g1_xml=arguments.g1_xml,
            frames=arguments.frames,
        )
    else:
        receipt = run_interactive(
            matcher,
            terrain,
            g1_xml=arguments.g1_xml,
            gamepad=arguments.gamepad,
            max_render_frames=arguments.max_render_frames,
        )
    payload = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    print(payload, flush=True)
    if getattr(arguments, "receipt", None) is not None:
        write_receipt_exclusive(arguments.receipt, receipt)
    return 0 if bool(receipt.get("accepted", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "LABEL",
    "EvdevCommandSource",
    "FixedRateAccumulator",
    "KeyboardCommandSource",
    "SceneTerrainAdapter",
    "build_parser",
    "build_viewer_model",
    "build_diagnostic_collision_model",
    "handle_key_press",
    "load_scene_terrain",
    "main",
    "optional_evdev_source",
    "overlay_text",
    "reset_without_mutation_on_failure",
    "run_interactive",
    "run_mujoco_headless_smoke",
    "scene_authentication_is_current",
    "write_receipt_exclusive",
)
