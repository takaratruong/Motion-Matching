"""Step-counted MuJoCo JSONL child for the unmodified GEAR checkout.

This module is safe to import in CPU-only tests.  GEAR, MuJoCo, Unitree, and
configuration modules are resolved lazily, inside the dedicated child process.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import json
import math
from pathlib import Path
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
    _validate_confined_output_path,
)


PROTOCOL_NAME = "gated-sim/v1"
PROTOCOL_VERSION = 1
PINNED_GEAR_COMMIT = "60de0df7ffedeef415fe58d435e92cc5b01ba3d9"
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

    def reset_from_qpos(
        self,
        qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
    ) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def sample(self) -> Mapping[str, object]:
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

    def _close_active(self) -> None:
        for stream in (self._state_file, self._contact_file):
            if stream is not None:
                stream.close()
        self._state_file = None
        self._contact_file = None
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def reset(
        self,
        *,
        scene_xml: str | Path,
        initial_qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
        log_dir: str | Path,
    ) -> dict[str, object]:
        if self._closed:
            raise ProtocolError("simulator runner is closed")
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

        self._close_active()
        backend = self._backend_factory(scene)
        self._backend = backend
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
            ratio = _STATE_PERIOD_S / sim_dt
            stride = round(ratio)
            if stride <= 0 or abs(ratio - stride) > _EXACT_TOLERANCE:
                raise ProtocolError(
                    "backend.sim_dt must divide the exact 50 Hz state period"
                )
            backend.reset_from_qpos(qpos.copy(), lateral, yaw)
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
            return {"nq": nq, "sim_dt_s": sim_dt, "sim_time_s": start_time}
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
            },
            "reset request",
        )
        if type(source["scene_xml"]) is not str or not source["scene_xml"]:
            raise ProtocolError("scene_xml must be a nonempty string")
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

    def __init__(self, gear_checkout: str | Path, scene_xml: str | Path) -> None:
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
                onscreen=config.get("ENABLE_ONSCREEN", True),
                offscreen=config.get("ENABLE_OFFSCREEN", False),
                enable_image_publish=False,
            )
        self._bindings = bindings
        self._simulator = simulator

    @property
    def model(self) -> object:
        return self._simulator.sim_env.mj_model

    @property
    def data(self) -> object:
        return self._simulator.sim_env.mj_data

    @property
    def sim_dt(self) -> float:
        return float(self._simulator.sim_dt)

    def reset_from_qpos(
        self,
        qpos: np.ndarray,
        lateral_offset_m: float,
        yaw_offset_rad: float,
    ) -> None:
        if qpos.shape != (int(self.model.nq),):
            raise ProtocolError(
                f"initial_qpos must contain exactly {int(self.model.nq)} values"
            )
        physical = physical_qpos_with_perturbation(
            qpos,
            lateral_offset_m=lateral_offset_m,
            yaw_offset_rad=yaw_offset_rad,
        )
        self._bindings.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = physical
        self.data.qvel[:] = 0.0
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0.0
        self._bindings.mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        started = time.monotonic()
        # The official simulator can print fall diagnostics.  Stdout is the
        # JSONL protocol channel in this child, so preserve those diagnostics
        # on stderr instead of allowing a non-JSON line to corrupt the peer.
        with redirect_stdout(sys.stderr):
            self._simulator.sim_env.sim_step()
        remaining = self.sim_dt - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)

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
            backend_factory=lambda scene: ExternalGearBackend(checkout, scene),
            output_stream=protocol_stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
