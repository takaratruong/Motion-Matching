"""Step-counted MuJoCo JSONL child for the unmodified GEAR checkout.

This module is safe to import in CPU-only tests.  GEAR, MuJoCo, Unitree, and
configuration modules are resolved lazily, inside the dedicated child process.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Callable, IO, Mapping, Protocol

import numpy as np

from .process import (
    AdvanceResult,
    ProcessError,
    _canonical_run_root,
    _confined_existing_path,
    _create_exclusive_directory,
    _open_confined_parent_fd,
    _validate_confined_output_path,
)


PROTOCOL_NAME = "gated-sim/v1"
PROTOCOL_VERSION = 1
PINNED_GEAR_COMMIT = "ddd41cbe5c5de32b6ff8f2822c73cb92bfc1c190"
_STATE_PERIOD_S = 1.0 / 50.0
_EXACT_TOLERANCE = 1.0e-12


class ProtocolError(ValueError):
    """A request, backend, or sampled value violates the gated contract."""


class SimulatorBackend(Protocol):
    @property
    def model(self) -> object:
        raise NotImplementedError

    @property
    def data(self) -> object:
        raise NotImplementedError

    @property
    def sim_dt(self) -> float:
        raise NotImplementedError

    @property
    def wall_clock_pacing(self) -> bool:
        raise NotImplementedError

    def reset_from_qpos(
        self,
        qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
        *,
        elastic_band_enabled: bool,
    ) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def prime_low_state(self) -> None:
        raise NotImplementedError

    def refresh_low_state(self) -> None:
        raise NotImplementedError

    def low_command_snapshot(self) -> Mapping[str, object]:
        raise NotImplementedError

    def sample(self) -> Mapping[str, object]:
        raise NotImplementedError

    def set_camera(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance_m: float,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float, np.float32, np.float64):
        raise ProtocolError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{label} must be a finite number")
    return number


def physical_qpos_with_perturbation(
    initial_qpos: np.ndarray,
    *,
    lateral_offset_m: float,
    yaw_offset_rad: float,
) -> np.ndarray:
    """Copy qpos and perturb only physical root y and world-yaw.

    The caller's target/MM boundary is never mutated.  MuJoCo free-root
    quaternions are interpreted in wxyz order.
    """

    source = np.asarray(initial_qpos, dtype=np.float64)
    if source.ndim != 1 or source.size < 7:
        raise ProtocolError("initial_qpos must contain a free root")
    if not np.all(np.isfinite(source)):
        raise ProtocolError("initial_qpos must contain only finite values")
    lateral = _finite_number(lateral_offset_m, "lateral_offset_m")
    yaw = _finite_number(yaw_offset_rad, "yaw_offset_rad")
    output = source.copy()
    output[1] += lateral

    root = output[3:7].copy()
    norm = float(np.linalg.norm(root))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ProtocolError("initial_qpos root quaternion is not normalizable")
    root /= norm
    half = 0.5 * yaw
    cy = math.cos(half)
    sz = math.sin(half)
    w, x, y, z = root
    rotated = np.array(
        [
            cy * w - sz * z,
            cy * x - sz * y,
            cy * y + sz * x,
            cy * z + sz * w,
        ],
        dtype=np.float64,
    )
    rotated /= np.linalg.norm(rotated)
    output[3:7] = rotated
    return output


def _json_safe(value: object, label: str = "sample") -> object:
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), label)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), label)
    if type(value) in (list, tuple):
        return [
            _json_safe(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ProtocolError(f"{label} contains a non-string or empty key")
            if key in output:
                raise ProtocolError(f"{label} contains duplicate key {key!r}")
            output[key] = _json_safe(item, f"{label}.{key}")
        return output
    raise ProtocolError(f"{label} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True)
class _SceneIdentity:
    path: Path
    parent_fd: int
    file_fd: int
    name: str
    device: int
    inode: int
    sha256: str


def _scene_descriptor_sha256(file_fd: int) -> tuple[str, os.stat_result]:
    before = os.fstat(file_fd)
    if not stat.S_ISREG(before.st_mode):
        raise ProtocolError("scene_xml must be a regular file")
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(file_fd, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(file_fd)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise ProtocolError("scene identity changed while hashing")
    return digest.hexdigest(), after


def _close_scene_identity(identity: _SceneIdentity | None) -> None:
    if identity is None:
        return
    for descriptor in (identity.file_fd, identity.parent_fd):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _capture_scene_identity(run_root: Path, scene: Path) -> _SceneIdentity:
    parent_fd: int | None = None
    file_fd: int | None = None
    retained = False
    try:
        parent_fd, name = _open_confined_parent_fd(run_root, scene, "scene_xml")
        file_fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        digest, opened = _scene_descriptor_sha256(file_fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
            or current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
            or current.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ProtocolError("scene identity changed while opening")
        retained = True
        return _SceneIdentity(
            path=scene,
            parent_fd=parent_fd,
            file_fd=file_fd,
            name=name,
            device=opened.st_dev,
            inode=opened.st_ino,
            sha256=digest,
        )
    except ProtocolError:
        raise
    except (OSError, ProcessError) as error:
        raise ProtocolError("scene_xml identity is unavailable") from error
    finally:
        if not retained:
            if file_fd is not None:
                try:
                    os.close(file_fd)
                except OSError:
                    pass
            if parent_fd is not None:
                try:
                    os.close(parent_fd)
                except OSError:
                    pass


def _scene_identity_is_current(
    expected: _SceneIdentity,
    run_root: Path,
    scene: Path,
) -> bool:
    if scene != expected.path:
        return False
    current_identity: _SceneIdentity | None = None
    try:
        retained_digest, retained = _scene_descriptor_sha256(expected.file_fd)
        if (
            retained.st_dev != expected.device
            or retained.st_ino != expected.inode
            or retained_digest != expected.sha256
        ):
            return False
        current_identity = _capture_scene_identity(run_root, scene)
        return (
            current_identity.device == expected.device
            and current_identity.inode == expected.inode
            and current_identity.sha256 == expected.sha256
        )
    except (OSError, ProtocolError):
        return False
    finally:
        _close_scene_identity(current_identity)


def _require_current_scene(
    expected: _SceneIdentity,
    run_root: Path,
    scene: Path,
) -> None:
    if not _scene_identity_is_current(expected, run_root, scene):
        raise ProtocolError(
            "scene identity/path changed; fresh simulator process required"
        )


class GatedSimulatorRunner:
    """Own one idle backend and advance it only by explicit integer requests."""

    def __init__(
        self,
        backend_factory: Callable[[Path], SimulatorBackend],
        *,
        run_root: str | Path,
    ) -> None:
        self._backend_factory = backend_factory
        try:
            self.run_root = _canonical_run_root(run_root)
        except ProcessError as error:
            raise ProtocolError(str(error)) from error
        self._backend: SimulatorBackend | None = None
        self._scene_identity: _SceneIdentity | None = None
        self._state_file: IO[str] | None = None
        self._contact_file: IO[str] | None = None
        self._steps = 0
        self._state_rows = 0
        self._contact_rows = 0
        self._state_stride = 0
        self._closed = False

    @property
    def backend(self) -> SimulatorBackend:
        if self._backend is None:
            raise ProtocolError("reset is required before simulator access")
        return self._backend

    def _close_logs(self) -> None:
        for stream in (self._state_file, self._contact_file):
            if stream is not None:
                stream.close()
        self._state_file = None
        self._contact_file = None

    def _close_backend(self) -> None:
        backend = self._backend
        identity = self._scene_identity
        self._backend = None
        self._scene_identity = None
        try:
            if backend is not None:
                backend.close()
        finally:
            _close_scene_identity(identity)

    def _close_active(self) -> None:
        self._close_logs()
        self._close_backend()

    def reset(
        self,
        *,
        scene_xml: str | Path,
        initial_qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
        log_dir: str | Path,
        elastic_band_enabled: bool,
    ) -> dict[str, object]:
        if self._closed:
            raise ProtocolError("simulator runner is closed")
        if type(elastic_band_enabled) is not bool:
            raise ProtocolError("elastic_band_enabled must be a boolean")
        try:
            scene = _confined_existing_path(
                self.run_root,
                scene_xml,
                "scene_xml",
                directory=False,
            )
            logs = _validate_confined_output_path(
                self.run_root,
                log_dir,
                "simulator log directory",
            )
        except ProcessError as error:
            raise ProtocolError(str(error)) from error
        lateral = _finite_number(lateral_offset_m, "lateral_offset_m")
        yaw = _finite_number(yaw_offset_rad, "yaw_offset_rad")

        if self._backend is None:
            identity = _capture_scene_identity(self.run_root, scene)
            try:
                backend = self._backend_factory(scene)
            except BaseException:
                _close_scene_identity(identity)
                raise
            self._backend = backend
            self._scene_identity = identity
            try:
                _require_current_scene(identity, self.run_root, scene)
            except BaseException:
                self._close_backend()
                raise
        else:
            backend = self._backend
            assert self._scene_identity is not None
            _require_current_scene(self._scene_identity, self.run_root, scene)
        self._close_logs()
        try:
            nq = getattr(backend.model, "nq", None)
            if type(nq) is not int or nq <= 0:
                raise ProtocolError("backend model.nq must be a positive integer")
            qpos = np.asarray(initial_qpos, dtype=np.float64)
            if qpos.ndim != 1 or qpos.size != nq:
                actual = qpos.size if qpos.ndim == 1 else "non-vector"
                raise ProtocolError(
                    f"initial_qpos must contain exactly {nq} values; found {actual}"
                )
            if not np.all(np.isfinite(qpos)):
                raise ProtocolError("initial_qpos must contain only finite values")
            sim_dt = _finite_number(backend.sim_dt, "backend.sim_dt")
            if sim_dt <= 0.0:
                raise ProtocolError("backend.sim_dt must be positive")
            if type(backend.wall_clock_pacing) is not bool:
                raise ProtocolError("backend.wall_clock_pacing must be boolean")
            ratio = _STATE_PERIOD_S / sim_dt
            stride = round(ratio)
            if stride <= 0 or abs(ratio - stride) > _EXACT_TOLERANCE:
                raise ProtocolError(
                    "backend.sim_dt must divide the exact 50 Hz state period"
                )
            backend.reset_from_qpos(
                qpos.copy(),
                lateral,
                yaw,
                elastic_band_enabled=elastic_band_enabled,
            )
            start_time = _finite_number(backend.data.time, "backend.data.time")
            try:
                _create_exclusive_directory(
                    self.run_root,
                    logs,
                    "simulator log directory",
                )
            except ProcessError as error:
                raise ProtocolError(str(error)) from error
            self._state_file = (logs / "state.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            )
            self._contact_file = (logs / "contacts.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            )
            self._steps = 0
            self._state_rows = 0
            self._contact_rows = 0
            self._state_stride = int(stride)
            return {
                "nq": nq,
                "sim_dt_s": sim_dt,
                "sim_time_s": start_time,
                "elastic_band_enabled": elastic_band_enabled,
            }
        except BaseException:
            self._close_active()
            raise

    def advance(self, steps: int) -> AdvanceResult:
        if type(steps) is not int or steps <= 0:
            raise ProtocolError("steps must be a positive integer")
        backend = self.backend
        assert self._state_file is not None
        assert self._contact_file is not None
        start = _finite_number(backend.data.time, "backend.data.time")
        wall_deadline = time.monotonic()
        state_start = self._state_rows
        contact_start = self._contact_rows
        for _ in range(steps):
            backend.step()
            self._steps += 1
            sim_time = _finite_number(backend.data.time, "backend.data.time")
            sample = backend.sample()
            if not isinstance(sample, Mapping):
                raise ProtocolError("backend.sample() must return a mapping")
            safe_sample = _json_safe(sample)
            assert type(safe_sample) is dict
            contacts = safe_sample.get("contacts", [])
            contact_row = {
                "step": self._steps,
                "sim_time_s": sim_time,
                "contacts": contacts,
            }
            self._contact_file.write(
                json.dumps(contact_row, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
            self._contact_rows += 1
            if self._steps % self._state_stride == 0:
                state = dict(safe_sample)
                state.pop("contacts", None)
                state_row = {
                    "step": self._steps,
                    "sim_time_s": sim_time,
                    "state": state,
                }
                self._state_file.write(
                    json.dumps(state_row, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
                self._state_rows += 1
            if backend.wall_clock_pacing:
                wall_deadline += float(backend.sim_dt)
                remaining = wall_deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
        self._state_file.flush()
        self._contact_file.flush()
        end = _finite_number(backend.data.time, "backend.data.time")
        expected_delta = steps * float(backend.sim_dt)
        if abs((end - start) - expected_delta) > _EXACT_TOLERANCE:
            raise ProtocolError(
                "backend MuJoCo time delta does not match requested step count: "
                f"expected {expected_delta:.17g}, found {end - start:.17g}"
            )
        return AdvanceResult(
            steps=steps,
            sim_time_start_s=start,
            sim_time_end_s=end,
            state_rows=self._state_rows - state_start,
            contact_rows=self._contact_rows - contact_start,
        )

    def prime_low_state(self) -> dict[str, object]:
        """Publish the reset observation without integrating scored physics."""

        backend = self.backend
        before = self.snapshot()
        backend.prime_low_state()
        after = self.snapshot()
        if after != before:
            raise ProtocolError(
                "LowState prime changed simulator time or evidence counters"
            )
        return {"published": True, **after}

    def refresh_low_state(self) -> dict[str, object]:
        """Republish current LowState without clearing LowCmd or integrating."""

        backend = self.backend
        before = self.snapshot()
        backend.refresh_low_state()
        after = self.snapshot()
        if after != before:
            raise ProtocolError(
                "LowState refresh changed simulator time or evidence counters"
            )
        return {"published": True, **after}

    def low_command_snapshot(self) -> dict[str, object]:
        """Copy the latest body LowCmd received by the simulator bridge."""

        source = self.backend.low_command_snapshot()
        if not isinstance(source, Mapping) or set(source) != {
            "received",
            "q_target",
        }:
            raise ProtocolError("backend LowCmd snapshot shape changed")
        received = source["received"]
        if type(received) is not bool:
            raise ProtocolError("backend LowCmd received flag must be boolean")
        if not received:
            if source["q_target"] is not None:
                raise ProtocolError("absent backend LowCmd must have null target")
            return {"received": False, "q_target": None}
        try:
            target = np.asarray(source["q_target"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ProtocolError("backend LowCmd target must be numeric") from error
        if target.shape != (29,) or not np.all(np.isfinite(target)):
            raise ProtocolError(
                "backend LowCmd target must contain 29 finite values"
            )
        return {
            "received": True,
            "q_target": tuple(float(value) for value in target),
        }

    def snapshot(self) -> dict[str, object]:
        backend = self.backend
        return {
            "steps": self._steps,
            "sim_time_s": _finite_number(
                backend.data.time, "backend.data.time"
            ),
            "state_rows": self._state_rows,
            "contact_rows": self._contact_rows,
        }

    def set_camera(
        self,
        sequence: int,
        azimuth_deg: float,
        elevation_deg: float,
        distance_m: float,
    ) -> dict[str, object]:
        if type(sequence) is not int or sequence < 0:
            raise ProtocolError("sequence must be a nonnegative integer")
        azimuth = _finite_number(azimuth_deg, "azimuth_deg")
        elevation = _finite_number(elevation_deg, "elevation_deg")
        distance = _finite_number(distance_m, "distance_m")
        if not (0.1 <= distance <= 100.0):
            raise ProtocolError("distance_m must lie within [0.1, 100.0]")
        backend = self.backend
        backend.set_camera(azimuth, elevation, distance)
        return {
            "sequence": sequence,
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "distance_m": distance,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_active()


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _loads_request(text: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProtocolError(f"invalid JSON constant: {value}")
            ),
        )
    except ProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid JSON request: {error}") from error


def _exact_request(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ProtocolError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _request_identity(value: object) -> tuple[str, str]:
    if type(value) is not dict:
        return "<invalid>", "<invalid>"
    op = value.get("op")
    request_id = value.get("request_id")
    return (
        op if type(op) is str and op else "<invalid>",
        request_id if type(request_id) is str and request_id else "<invalid>",
    )


def _validated_base(value: object) -> tuple[dict[str, object], str, str]:
    if type(value) is not dict:
        raise ProtocolError("request must be an object")
    if value.get("v") != PROTOCOL_VERSION or type(value.get("v")) is not int:
        raise ProtocolError("request v must equal integer 1")
    op = value.get("op")
    request_id = value.get("request_id")
    if type(op) is not str or not op:
        raise ProtocolError("request op must be a nonempty string")
    if type(request_id) is not str or not request_id:
        raise ProtocolError("request_id must be a nonempty string")
    return value, op, request_id


def _handle_request(
    runner: GatedSimulatorRunner,
    value: object,
) -> tuple[str, str, dict[str, object]]:
    source, op, request_id = _validated_base(value)
    if op == "hello":
        _exact_request(source, {"v", "op", "request_id"}, "hello request")
        return op, request_id, {"protocol": PROTOCOL_NAME}
    if op == "reset":
        _exact_request(
            source,
            {
                "v",
                "op",
                "request_id",
                "scene_xml",
                "initial_qpos",
                "lateral_offset_m",
                "yaw_offset_rad",
                "log_dir",
                "elastic_band_enabled",
            },
            "reset request",
        )
        if type(source["scene_xml"]) is not str or not source["scene_xml"]:
            raise ProtocolError("scene_xml must be a nonempty string")
        if type(source["elastic_band_enabled"]) is not bool:
            raise ProtocolError("elastic_band_enabled must be a JSON boolean")
        if type(source["log_dir"]) is not str or not source["log_dir"]:
            raise ProtocolError("log_dir must be a nonempty string")
        if type(source["initial_qpos"]) is not list:
            raise ProtocolError("initial_qpos must be an array")
        qpos = np.asarray(
            [
                _finite_number(item, f"initial_qpos[{index}]")
                for index, item in enumerate(source["initial_qpos"])
            ],
            dtype=np.float64,
        )
        return (
            op,
            request_id,
            runner.reset(
                scene_xml=source["scene_xml"],
                initial_qpos=qpos,
                lateral_offset_m=_finite_number(
                    source["lateral_offset_m"], "lateral_offset_m"
                ),
                yaw_offset_rad=_finite_number(
                    source["yaw_offset_rad"], "yaw_offset_rad"
                ),
                log_dir=source["log_dir"],
                elastic_band_enabled=source["elastic_band_enabled"],
            ),
        )
    if op == "advance":
        _exact_request(
            source, {"v", "op", "request_id", "steps"}, "advance request"
        )
        if type(source["steps"]) is not int or source["steps"] <= 0:
            raise ProtocolError("steps must be a positive integer")
        result = runner.advance(source["steps"])
        return op, request_id, {
            "steps": result.steps,
            "sim_time_start_s": result.sim_time_start_s,
            "sim_time_end_s": result.sim_time_end_s,
            "state_rows": result.state_rows,
            "contact_rows": result.contact_rows,
        }
    if op == "prime_low_state":
        _exact_request(
            source,
            {"v", "op", "request_id"},
            "prime_low_state request",
        )
        return op, request_id, runner.prime_low_state()
    if op == "refresh_low_state":
        _exact_request(
            source,
            {"v", "op", "request_id"},
            "refresh_low_state request",
        )
        return op, request_id, runner.refresh_low_state()
    if op == "low_command":
        _exact_request(
            source,
            {"v", "op", "request_id"},
            "low_command request",
        )
        return op, request_id, runner.low_command_snapshot()
    if op == "camera":
        _exact_request(
            source,
            {
                "v",
                "op",
                "request_id",
                "sequence",
                "azimuth_deg",
                "elevation_deg",
                "distance_m",
            },
            "camera request",
        )
        if type(source["sequence"]) is not int or source["sequence"] < 0:
            raise ProtocolError("sequence must be a nonnegative integer")
        return (
            op,
            request_id,
            runner.set_camera(
                sequence=source["sequence"],
                azimuth_deg=_finite_number(source["azimuth_deg"], "azimuth_deg"),
                elevation_deg=_finite_number(
                    source["elevation_deg"], "elevation_deg"
                ),
                distance_m=_finite_number(source["distance_m"], "distance_m"),
            ),
        )
    if op == "snapshot":
        _exact_request(source, {"v", "op", "request_id"}, "snapshot request")
        return op, request_id, runner.snapshot()
    if op == "close":
        _exact_request(source, {"v", "op", "request_id"}, "close request")
        runner.close()
        return op, request_id, {"closed": True}
    raise ProtocolError(f"unknown operation: {op}")


def serve_jsonl(
    *,
    run_root: str | Path,
    backend_factory: Callable[[Path], SimulatorBackend],
    input_stream: IO[str] = sys.stdin,
    output_stream: IO[str] = sys.stdout,
) -> None:
    """Serve exact JSONL until EOF or a valid close request."""

    runner = GatedSimulatorRunner(backend_factory, run_root=run_root)
    try:
        for raw in input_stream:
            value: object = None
            try:
                if not raw.endswith("\n"):
                    raise ProtocolError("request must end with one newline")
                text = raw[:-1]
                if not text:
                    raise ProtocolError("request line must not be empty")
                value = _loads_request(text)
                op, request_id, data = _handle_request(runner, value)
                response = {
                    "v": PROTOCOL_VERSION,
                    "ok": True,
                    "op": op,
                    "request_id": request_id,
                    "data": data,
                }
            except Exception as error:
                op, request_id = _request_identity(value)
                response = {
                    "v": PROTOCOL_VERSION,
                    "ok": False,
                    "op": op,
                    "request_id": request_id,
                    "error": {
                        "code": "invalid_request",
                        "message": str(error) or type(error).__name__,
                    },
                }
            output_stream.write(
                json.dumps(response, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
            output_stream.flush()
            if response["ok"] is True and response["op"] == "close":
                break
    finally:
        runner.close()


@dataclass(frozen=True)
class _ExternalBindings:
    checkout: Path
    base_simulator: type
    sim_loop_config: type
    mujoco: object


def _git(checkout: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(checkout), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "")
        raise ProtocolError(
            f"failed to verify GEAR checkout: {detail or error}"
        ) from error
    return completed.stdout.strip()


def _verified_checkout(path: str | Path) -> Path:
    checkout = Path(path).resolve(strict=True)
    if not checkout.is_dir():
        raise ProtocolError(f"gear_checkout is not a directory: {checkout}")
    top = Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != checkout:
        raise ProtocolError("gear_checkout must be the Git worktree root")
    commit = _git(checkout, "rev-parse", "--verify", "HEAD")
    if commit != PINNED_GEAR_COMMIT:
        raise ProtocolError(
            f"GEAR commit mismatch: expected {PINNED_GEAR_COMMIT}, found {commit}"
        )
    dirty = _git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if dirty:
        raise ProtocolError("GEAR checkout must be clean, including untracked files")
    return checkout


def _require_module_origin(
    module: object,
    *,
    base: Path,
    label: str,
) -> None:
    origin_text = getattr(module, "__file__", None)
    search_locations = getattr(module, "__path__", None)
    has_origin = type(origin_text) is str and bool(origin_text)
    if not has_origin and origin_text is not None:
        raise ProtocolError(f"{label} module origin is unavailable")
    if has_origin:
        try:
            origin = Path(origin_text).resolve(strict=True)
            origin.relative_to(base)
        except (OSError, ValueError) as error:
            raise ProtocolError(
                f"{label} module origin is outside {base}: {origin_text}"
            ) from error
    if search_locations is None:
        if not has_origin:
            raise ProtocolError(f"{label} module origin is unavailable")
        return
    try:
        locations = tuple(search_locations)
    except TypeError as error:
        raise ProtocolError(f"{label} package search path is invalid") from error
    if not locations and not has_origin:
        raise ProtocolError(f"{label} package search path is unavailable")
    for location_text in locations:
        if type(location_text) is not str or not location_text:
            raise ProtocolError(f"{label} package search path is invalid")
        try:
            location = Path(location_text).resolve(strict=True)
            location.relative_to(base)
        except (OSError, ValueError) as error:
            raise ProtocolError(
                f"{label} package search path is outside {base}: {location_text}"
            ) from error


def _require_unitree_origins(unitree_source: Path) -> None:
    names = sorted(
        name
        for name in sys.modules
        if name == "unitree_sdk2py" or name.startswith("unitree_sdk2py.")
    )
    if "unitree_sdk2py" not in names:
        raise ProtocolError("unitree_sdk2py module origin is unavailable")
    for name in names:
        _require_module_origin(
            sys.modules.get(name),
            base=unitree_source,
            label=name,
        )


def _require_class_origin(
    cls: type,
    *,
    checkout: Path,
    label: str,
) -> None:
    _require_module_origin(
        sys.modules.get(cls.__module__),
        base=checkout,
        label=label,
    )


def _resolve_unitree_source(checkout: Path) -> Path:
    """Locate the pinned Unitree Python package inside the verified checkout.

    The source must be a real in-checkout directory: an escaping symlink is
    rejected rather than followed, and the package directory itself must exist.
    """

    source = checkout / "external_dependencies" / "unitree_sdk2_python"
    if source.is_symlink():
        raise ProtocolError(
            f"unitree source must be a real directory, not a symlink: {source}"
        )
    if not source.is_dir():
        raise ProtocolError(f"unitree source directory is missing: {source}")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(checkout)
    except ValueError as error:
        raise ProtocolError(
            f"unitree source escapes verified checkout: {source}"
        ) from error
    package = resolved / "unitree_sdk2py"
    if package.is_symlink() or not package.is_dir():
        raise ProtocolError(
            f"unitree source lacks the unitree_sdk2py package: {package}"
        )
    return resolved


def load_external_bindings(gear_checkout: str | Path) -> _ExternalBindings:
    """Resolve the pinned external imports without constructing a simulator."""

    checkout = _verified_checkout(gear_checkout)
    unitree_source = _resolve_unitree_source(checkout)
    checkout_text = str(checkout)
    unitree_text = str(unitree_source)
    for entry in (checkout_text, unitree_text):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, unitree_text)
    sys.path.insert(0, checkout_text)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import unitree_sdk2py

        # Authenticate the complete cached namespace before importing GEAR:
        # BaseSimulator imports Unitree children at module load time and must
        # never observe a shadow module.
        _require_unitree_origins(unitree_source)
        from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        _require_unitree_origins(unitree_source)
        import mujoco
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode

    _require_class_origin(
        BaseSimulator,
        checkout=checkout,
        label="BaseSimulator",
    )
    _require_class_origin(
        SimLoopConfig,
        checkout=checkout,
        label="SimLoopConfig",
    )
    # Recheck commit and cleanliness after imports to close the verification
    # window and prove that importing did not write into the external checkout.
    _verified_checkout(checkout)

    return _ExternalBindings(
        checkout=checkout,
        base_simulator=BaseSimulator,
        sim_loop_config=SimLoopConfig,
        mujoco=mujoco,
    )


class ExternalGearBackend:
    """Lazy adapter over the pinned official ``BaseSimulator``."""

    def __init__(
        self,
        gear_checkout: str | Path,
        scene_xml: str | Path,
        *,
        wall_clock_pacing: bool = True,
        onscreen: bool = False,
        freeze_on_fall: bool = False,
    ) -> None:
        if type(wall_clock_pacing) is not bool:
            raise ProtocolError("wall_clock_pacing must be a boolean")
        if type(onscreen) is not bool:
            raise ProtocolError("onscreen must be a boolean")
        if type(freeze_on_fall) is not bool:
            raise ProtocolError("freeze_on_fall must be a boolean")
        if freeze_on_fall and not onscreen:
            raise ProtocolError("freeze_on_fall requires onscreen")
        scene = Path(scene_xml).resolve(strict=True)
        if not scene.is_file():
            raise ProtocolError(f"scene_xml is not a file: {scene}")
        # Keep all diagnostics away from the JSONL stdout channel.
        with redirect_stdout(sys.stderr):
            bindings = load_external_bindings(gear_checkout)
            config_loader = bindings.sim_loop_config()
            config = config_loader.load_wbc_yaml()
            config["ROBOT_SCENE"] = str(scene)
            simulator = bindings.base_simulator(
                config=config,
                env_name=config_loader.env_name,
                # This child is a deterministic JSONL physics gate.  Rendering is
                # explicit and opt-in: onscreen is forwarded exactly as requested
                # while offscreen rendering and image publishing stay disabled.
                onscreen=onscreen,
                offscreen=False,
                enable_image_publish=False,
            )
        self._bindings = bindings
        self._simulator = simulator
        self._wall_clock_pacing = wall_clock_pacing
        self._freeze_on_fall = freeze_on_fall
        self._frozen = False
        if freeze_on_fall:
            # Diagnostic-only: replace only this live instance's fall callback so
            # the first fall latches the fallen pose instead of rewinding the
            # official simulator and tearing down the visible viewer.
            self._simulator.sim_env.check_fall = self._freeze_on_fall_callback

    @property
    def model(self) -> object:
        return self._simulator.sim_env.mj_model

    @property
    def data(self) -> object:
        return self._simulator.sim_env.mj_data

    @property
    def sim_dt(self) -> float:
        return float(self._simulator.sim_dt)

    @property
    def wall_clock_pacing(self) -> bool:
        return self._wall_clock_pacing

    def reset_from_qpos(
        self,
        qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
        *,
        elastic_band_enabled: bool,
    ) -> None:
        if type(elastic_band_enabled) is not bool:
            raise ProtocolError("elastic_band_enabled must be a boolean")
        if qpos.shape != (int(self.model.nq),):
            raise ProtocolError(
                f"initial_qpos must contain exactly {int(self.model.nq)} values"
            )
        physical = physical_qpos_with_perturbation(
            qpos,
            lateral_offset_m=lateral_offset_m,
            yaw_offset_rad=yaw_offset_rad,
        )
        band = getattr(self._simulator.sim_env, "elastic_band", None)
        if band is None or not hasattr(band, "enable"):
            raise ProtocolError(
                "pinned simulator is missing its elastic_band object"
            )
        try:
            angular_gain = float(band.kp_ang)
        except (AttributeError, TypeError, ValueError) as error:
            raise ProtocolError(
                "pinned simulator elastic_band angular gain is invalid"
            ) from error
        if not math.isfinite(angular_gain) or angular_gain < 0.0:
            raise ProtocolError(
                "pinned simulator elastic_band angular gain is invalid"
            )
        if not hasattr(self, "_elastic_band_angular_gain"):
            self._elastic_band_angular_gain = angular_gain
        if elastic_band_enabled:
            try:
                anchor = np.asarray(band.point, dtype=np.float64)
            except (AttributeError, TypeError, ValueError) as error:
                raise ProtocolError(
                    "pinned simulator elastic_band point is invalid"
                ) from error
            if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
                raise ProtocolError(
                    "pinned simulator elastic_band point is invalid"
                )
            translated_anchor = anchor.copy()
            translated_anchor[:2] = physical[:2]
            band.point = translated_anchor
            # The upstream spring is hard-coded to world-identity orientation.
            # That is valid only for flat identity-yaw starts; an authenticated
            # route may begin at any heading. Keep angular damping, but remove
            # the invalid absolute-orientation spring during bootstrap.
            band.kp_ang = 0.0
        else:
            band.kp_ang = self._elastic_band_angular_gain
        band.enable = elastic_band_enabled
        self._bindings.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = physical
        self.data.qvel[:] = 0.0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0.0
        self._bindings.mujoco.mj_forward(self.model, self.data)
        self._viewer_step_count = 0

    def _freeze_on_fall_callback(self) -> None:
        # Instance-local replacement for the official ``check_fall``. A pelvis
        # height below the official 0.2 m threshold latches the first fallen
        # pose once, logs one diagnostic, and never calls the upstream reset.
        if self._frozen:
            return
        sim_env = self._simulator.sim_env
        if float(sim_env.mj_data.qpos[2]) >= 0.2:
            return
        self._frozen = True
        # Preserve the fallen qpos; only stop the dynamics.
        sim_env.mj_data.qvel[:] = 0.0
        sim_env.mj_data.qacc[:] = 0.0
        if sim_env.mj_data.ctrl.size:
            sim_env.mj_data.ctrl[:] = 0.0
        # Refresh MuJoCo's derived position/velocity caches without integrating.
        # In particular, ``prepare_obs`` calls ``mj_objectVelocity`` for the
        # torso; without this forward pass it would retain the final falling
        # velocity from the preceding physics step. ``mj_forward`` recomputes
        # acceleration, so zero it again to keep the frozen LowState consistent.
        self._bindings.mujoco.mj_forward(self.model, self.data)
        sim_env.mj_data.qacc[:] = 0.0
        print(
            "onscreen fall freeze: preserving first fallen pose",
            file=sys.stderr,
            flush=True,
        )

    def prime_low_state(self) -> None:
        """Clear bootstrap receipts and publish reset state without `mj_step`."""

        with redirect_stdout(sys.stderr):
            sim_env = self._simulator.sim_env
            try:
                sim_env.unitree_bridge.reset()
                sim_env.obs = sim_env.prepare_obs()
                sim_env.unitree_bridge.PublishLowState(sim_env.obs)
            except (AttributeError, TypeError, ValueError) as error:
                raise ProtocolError(
                    "simulator cannot publish a no-step LowState prime"
                ) from error

    def refresh_low_state(self) -> None:
        """Publish the latest observation without resetting the LowCmd receiver."""

        with redirect_stdout(sys.stderr):
            sim_env = self._simulator.sim_env
            try:
                sim_env.obs = sim_env.prepare_obs()
                sim_env.unitree_bridge.PublishLowState(sim_env.obs)
            except (AttributeError, TypeError, ValueError) as error:
                raise ProtocolError(
                    "simulator cannot refresh LowState without stepping"
                ) from error

    def low_command_snapshot(self) -> Mapping[str, object]:
        bridge = self._simulator.sim_env.unitree_bridge
        try:
            lock = bridge.low_cmd_lock
            with lock:
                received = bridge.low_cmd_received
                count = bridge.num_body_motor
                if type(received) is not bool:
                    raise ProtocolError(
                        "simulator LowCmd receipt flag must be boolean"
                    )
                if not received:
                    return {"received": False, "q_target": None}
                if type(count) is not int or count != 29:
                    raise ProtocolError(
                        "simulator LowCmd motor count must equal 29"
                    )
                target = np.asarray(
                    [bridge.low_cmd.motor_cmd[index].q for index in range(count)],
                    dtype=np.float64,
                )
        except ProtocolError:
            raise
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ProtocolError(
                "simulator cannot snapshot received LowCmd"
            ) from error
        if target.shape != (29,) or not np.all(np.isfinite(target)):
            raise ProtocolError(
                "simulator received LowCmd must contain 29 finite targets"
            )
        return {"received": True, "q_target": target.copy()}

    def _frozen_step(self) -> None:
        # Diagnostic-only frozen service: keep GEAR connected and advance the
        # protocol clock without touching physics.
        sim_env = self._simulator.sim_env
        viewer = getattr(sim_env, "viewer", None)
        if viewer is None or not viewer.is_running():
            raise ProtocolError("MuJoCo viewer is closed")
        try:
            sim_env.obs = sim_env.prepare_obs()
            sim_env.unitree_bridge.PublishLowState(sim_env.obs)
            if sim_env.unitree_bridge.joystick:
                sim_env.unitree_bridge.PublishWirelessController()
        except (AttributeError, TypeError, ValueError) as error:
            raise ProtocolError(
                "frozen simulator cannot republish LowState"
            ) from error
        # Advance the simulated clock by exactly one sim_dt without physics.
        sim_env.mj_data.time += float(self.sim_dt)

    def step(self) -> None:
        # The official simulator can print fall diagnostics.  Stdout is the
        # JSONL protocol channel in this child, so preserve those diagnostics
        # on stderr instead of allowing a non-JSON line to corrupt the peer.
        with redirect_stdout(sys.stderr):
            sim_env = self._simulator.sim_env
            viewer = getattr(sim_env, "viewer", None)
            if viewer is not None and not viewer.is_running():
                raise ProtocolError("MuJoCo viewer is closed")
            if getattr(self, "_freeze_on_fall", False) and self._frozen:
                self._frozen_step()
            else:
                sim_env.sim_step()
            # Presentation only: synchronize the passive viewer at its pinned
            # 50 Hz cadence, independent of the 200 Hz physics integration.
            if viewer is not None:
                viewer_step_count = getattr(self, "_viewer_step_count", 0) + 1
                self._viewer_step_count = viewer_step_count
                viewer_stride = max(1, round(_STATE_PERIOD_S / self.sim_dt))
                if viewer_step_count % viewer_stride == 0:
                    sim_env.update_viewer()

    def set_camera(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        distance_m: float,
    ) -> None:
        # Presentation only: adjust an existing running passive viewer and
        # synchronize exactly once.  This never advances physics and stays on
        # stderr like the other viewer diagnostics.
        with redirect_stdout(sys.stderr):
            viewer = getattr(self._simulator.sim_env, "viewer", None)
            if viewer is None or not viewer.is_running():
                raise ProtocolError("passive viewer is not running")
            viewer.cam.azimuth = azimuth_deg
            viewer.cam.elevation = elevation_deg
            viewer.cam.distance = distance_m
            viewer.sync()

    def _geom_name(self, geom_id: int) -> str:
        name = self._bindings.mujoco.mj_id2name(
            self.model,
            self._bindings.mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )
        return "" if name is None else str(name)

    def sample(self) -> Mapping[str, object]:
        contacts: list[dict[str, object]] = []
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            force = np.zeros(6, dtype=np.float64)
            self._bindings.mujoco.mj_contactForce(
                self.model, self.data, index, force
            )
            contacts.append(
                {
                    "geom1_id": int(contact.geom1),
                    "geom2_id": int(contact.geom2),
                    "geom1": self._geom_name(int(contact.geom1)),
                    "geom2": self._geom_name(int(contact.geom2)),
                    "distance_m": float(contact.dist),
                    "position_m": np.asarray(contact.pos, dtype=np.float64).copy(),
                    "frame": np.asarray(contact.frame, dtype=np.float64).copy(),
                    "force": force,
                }
            )
        pelvis = self.data.body("pelvis")
        return {
            "qpos": np.asarray(self.data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float64).copy(),
            "qacc": np.asarray(self.data.qacc, dtype=np.float64).copy(),
            "ctrl": np.asarray(self.data.ctrl, dtype=np.float64).copy(),
            "actuator_force": np.asarray(
                self.data.actuator_force, dtype=np.float64
            ).copy(),
            "pelvis_position_m": np.asarray(
                pelvis.xpos, dtype=np.float64
            ).copy(),
            "pelvis_quaternion_wxyz": np.asarray(
                pelvis.xquat, dtype=np.float64
            ).copy(),
            "pelvis_up": np.asarray(pelvis.xmat, dtype=np.float64)
            .reshape(3, 3)[:, 2]
            .copy(),
            "contacts": contacts,
        }

    def close(self) -> None:
        with redirect_stdout(sys.stderr):
            self._simulator.close()


def _preflight(gear_checkout: Path) -> int:
    try:
        with redirect_stdout(sys.stderr):
            bindings = load_external_bindings(gear_checkout)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "not_run",
                    "stage": "import_preflight",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                allow_nan=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "gear_commit": PINNED_GEAR_COMMIT,
                "base_simulator": (
                    f"{bindings.base_simulator.__module__}."
                    f"{bindings.base_simulator.__name__}"
                ),
                "config_loader": (
                    f"{bindings.sim_loop_config.__module__}."
                    f"{bindings.sim_loop_config.__name__}"
                ),
            },
            allow_nan=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gear-checkout", required=True, type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--import-preflight", action="store_true")
    parser.add_argument("--unpaced-physics", action="store_true")
    parser.add_argument("--onscreen", action="store_true")
    parser.add_argument("--freeze-on-fall", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.import_preflight:
        return _preflight(args.gear_checkout)
    if args.run_root is None:
        print("gated simulator requires --run-root", file=sys.stderr, flush=True)
        return 2
    try:
        checkout = _verified_checkout(args.gear_checkout)
        run_root = _canonical_run_root(args.run_root)
    except Exception as error:
        print(f"gated simulator checkout error: {error}", file=sys.stderr, flush=True)
        return 2
    protocol_stdout = sys.stdout
    # Reserve the original stdout stream exclusively for protocol frames.
    # Official DDS/simulator threads use ``print`` and therefore follow the
    # process-wide redirected ``sys.stdout`` to stderr as well.
    with redirect_stdout(sys.stderr):
        serve_jsonl(
            run_root=run_root,
            backend_factory=lambda scene: ExternalGearBackend(
                checkout,
                scene,
                wall_clock_pacing=not args.unpaced_physics,
                onscreen=args.onscreen,
                freeze_on_fall=args.freeze_on_fall,
            ),
            output_stream=protocol_stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
