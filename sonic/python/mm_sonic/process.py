"""Explicit subprocess lifecycles for the gated SONIC simulation.

The orchestration boundary is deliberately dependency-light: importing this
module never imports GEAR, MuJoCo, Unitree, or DDS.  Those imports belong to the
``mm_sonic.gated_sim`` child only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pty
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Callable, Mapping, Sequence

import numpy as np


_PROTOCOL_VERSION = 1
_MANAGED_GEAR_FLAGS = frozenset(
    {
        "--input-type",
        "--target-motion-logfile",
        "--logs-dir",
        "--enable-csv-logs",
        "--zmq-conflate",
        "--zmq-verbose",
    }
)
_DEFAULT_STARTUP_MARKERS = ("Initialized ZMQ endpoint interface",)
_DEFAULT_ACTIVE_MARKERS = (
    "[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state",
    "ZMQ STREAMING MODE: ENABLED",
)


class ProcessError(RuntimeError):
    """Base class for a controlled child-process failure."""


class ProcessProtocolError(ProcessError):
    """A JSONL peer violated the gated simulator protocol."""


class ChildProcessDied(ProcessError):
    """A required child exited before completing its operation."""


class OperatorCancelled(ProcessError):
    """The operator requested cancellation."""


def _canonical_run_root(value: str | Path) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise ProcessError("run_root must be an absolute path")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise ProcessError(f"run_root is unavailable: {raw}") from error
    if raw != resolved or raw.is_symlink():
        raise ProcessError("run_root must be canonical and cannot be a symlink")
    try:
        mode = raw.stat().st_mode
    except OSError as error:
        raise ProcessError(f"run_root is unavailable: {raw}") from error
    if not stat.S_ISDIR(mode):
        raise ProcessError("run_root must be a directory")
    return resolved


def _confined_candidate(run_root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ProcessError(f"{label} must be absolute and beneath run_root")
    try:
        relative = candidate.relative_to(run_root)
    except ValueError as error:
        raise ProcessError(f"{label} must be beneath run_root") from error
    if relative == Path(".") or ".." in relative.parts:
        raise ProcessError(f"{label} must be a descendant beneath run_root")
    return candidate


def _reject_existing_symlink_components(
    run_root: Path,
    candidate: Path,
    *,
    label: str,
    include_leaf: bool,
) -> None:
    relative = candidate.relative_to(run_root)
    parts = relative.parts if include_leaf else relative.parts[:-1]
    current = run_root
    for part in parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as error:
            raise ProcessError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(mode):
            raise ProcessError(f"{label} contains a symlink component: {current}")
        if current != candidate and not stat.S_ISDIR(mode):
            raise ProcessError(f"{label} parent is not a directory: {current}")


def _validate_confined_output_path(
    run_root: Path,
    value: str | Path,
    label: str,
) -> Path:
    candidate = _confined_candidate(run_root, value, label)
    _reject_existing_symlink_components(
        run_root,
        candidate,
        label=label,
        include_leaf=True,
    )
    if candidate.exists() or candidate.is_symlink():
        raise ProcessError(f"{label} already exists: {candidate}")
    return candidate


def _create_confined_parents(
    run_root: Path,
    candidate: Path,
    *,
    label: str,
) -> None:
    relative_parent = candidate.parent.relative_to(run_root)
    current = run_root
    for part in relative_parent.parts:
        current /= part
        try:
            current.mkdir()
        except FileExistsError:
            try:
                mode = current.lstat().st_mode
            except OSError as error:
                raise ProcessError(
                    f"cannot inspect {label} parent: {current}"
                ) from error
            if stat.S_ISLNK(mode):
                raise ProcessError(
                    f"{label} contains a symlink component: {current}"
                )
            if not stat.S_ISDIR(mode):
                raise ProcessError(f"{label} parent is not a directory: {current}")
        except OSError as error:
            raise ProcessError(f"cannot create {label} parent: {current}") from error


def _prepare_confined_output_path(
    run_root: Path,
    value: str | Path,
    label: str,
) -> Path:
    candidate = _validate_confined_output_path(run_root, value, label)
    _create_confined_parents(run_root, candidate, label=label)
    return _validate_confined_output_path(run_root, candidate, label)


def _create_exclusive_directory(
    run_root: Path,
    value: str | Path,
    label: str,
) -> Path:
    candidate = _prepare_confined_output_path(run_root, value, label)
    try:
        candidate.mkdir()
    except FileExistsError as error:
        raise ProcessError(f"{label} already exists: {candidate}") from error
    except OSError as error:
        raise ProcessError(f"cannot create {label}: {candidate}") from error
    return candidate


def _open_exclusive_binary_output(
    run_root: Path,
    value: str | Path,
    label: str,
):
    candidate = _prepare_confined_output_path(run_root, value, label)
    try:
        return candidate.open("xb")
    except FileExistsError as error:
        raise ProcessError(f"{label} already exists: {candidate}") from error
    except OSError as error:
        raise ProcessError(f"cannot create {label}: {candidate}") from error


def _confined_existing_path(
    run_root: Path,
    value: str | Path,
    label: str,
    *,
    directory: bool,
) -> Path:
    candidate = _confined_candidate(run_root, value, label)
    _reject_existing_symlink_components(
        run_root,
        candidate,
        label=label,
        include_leaf=True,
    )
    try:
        resolved = candidate.resolve(strict=True)
        mode = candidate.stat().st_mode
    except OSError as error:
        raise ProcessError(f"{label} is unavailable: {candidate}") from error
    if resolved != candidate:
        raise ProcessError(f"{label} cannot traverse a symlink")
    expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise ProcessError(f"{label} must be a {kind}: {candidate}")
    return candidate


@dataclass(frozen=True)
class AdvanceResult:
    steps: int
    sim_time_start_s: float
    sim_time_end_s: float
    state_rows: int
    contact_rows: int


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProcessProtocolError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _loads_exact(text: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProcessProtocolError(f"invalid JSON constant: {value}")
            ),
        )
    except ProcessProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProcessProtocolError(f"invalid JSON response: {error}") from error


def _exact_object(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ProcessProtocolError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ProcessProtocolError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ProcessProtocolError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ProcessProtocolError(f"{label} must be a finite JSON number")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ProcessProtocolError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _cancelled(callback: Callable[[], bool] | None) -> bool:
    return callback is not None and bool(callback())


def _wait_process(process: subprocess.Popen[bytes], seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    return process.poll() is not None


def _verify_new_process_group(
    process: subprocess.Popen[bytes],
    *,
    timeout_s: float = 1.0,
) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            raise ChildProcessDied(
                f"child {process.pid} exited {process.returncode} "
                "before PGID verification"
            )
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            pgid = -1
        if pgid == process.pid and pgid != os.getpgrp():
            return pgid
        if time.monotonic() >= deadline:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            raise ProcessError(
                f"child {process.pid} did not enter its verified new process group"
            )
        time.sleep(0.005)


def _cleanup_expected_new_group(
    process: subprocess.Popen[bytes],
    *,
    term_grace_s: float,
    kill_grace_s: float,
) -> bool:
    """Boundedly clean the group promised by ``start_new_session=True``.

    The expected PGID is the freshly allocated child PID.  We never signal a
    caller/ambient group, and when the leader is still live we additionally
    require it to have entered that exact group before using ``killpg``.
    """

    expected_pgid = process.pid
    if expected_pgid == os.getpgrp():
        raise ProcessError("refusing to clean an ambient process group")

    def group_exists() -> bool:
        return bool(_linux_group_states(expected_pgid))

    def signal_expected(sig: signal.Signals) -> bool:
        if process.poll() is None:
            try:
                current = os.getpgid(process.pid)
            except ProcessLookupError:
                current = None
            if current != expected_pgid:
                return False
        if not group_exists():
            return False
        try:
            os.killpg(expected_pgid, sig)
        except ProcessLookupError:
            return False
        return True

    def wait_absent(seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while group_exists() and time.monotonic() < deadline:
            process.poll()
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return not group_exists()

    if group_exists():
        signal_expected(signal.SIGTERM)
        wait_absent(term_grace_s)
    elif process.poll() is None:
        # A child that somehow failed to create the promised group is still
        # safe to terminate directly; never infer or signal its ambient PGID.
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        _wait_process(process, term_grace_s)
    if group_exists():
        signal_expected(signal.SIGKILL)
        wait_absent(kill_grace_s)
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.05, kill_grace_s))
        except subprocess.TimeoutExpired:
            pass
    return not group_exists() and process.poll() is not None


def _safe_kill_created_group(
    process: subprocess.Popen[bytes],
    pgid: int,
    sig: signal.Signals,
) -> bool:
    """Signal only the group verified for ``process`` at creation time."""

    if pgid != process.pid or pgid == os.getpgrp():
        raise ProcessError("refusing to signal an unverified process group")
    if process.poll() is None:
        try:
            current = os.getpgid(process.pid)
        except ProcessLookupError:
            current = None
        if current != pgid:
            raise ProcessError(
                f"refusing to signal changed PGID: expected {pgid}, found {current}"
            )
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


class GatedSimulatorClient:
    """Synchronous, duplicate-key-safe JSONL client with no request deadline."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        command: Sequence[str] | None = None,
        gear_checkout: str | Path | None = None,
        stdout_archive: str | Path,
        stderr_archive: str | Path,
        cancelled: Callable[[], bool] | None = None,
        poll_interval_s: float = 0.05,
        stop_grace_s: float = 1.0,
        term_grace_s: float = 1.0,
        kill_grace_s: float = 1.0,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.run_root = _canonical_run_root(run_root)
        if command is None:
            if gear_checkout is None:
                raise ValueError(
                    "gear_checkout is required without an injected command"
                )
            checkout = Path(gear_checkout).resolve(strict=True)
            command = (
                sys.executable,
                "-u",
                "-B",
                "-m",
                "mm_sonic.gated_sim",
                "--gear-checkout",
                str(checkout),
                "--run-root",
                str(self.run_root),
            )
        if not command or any(type(item) is not str or not item for item in command):
            raise ValueError("command must contain nonempty strings")
        if poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        self.command = tuple(command)
        self.stdout_archive = _validate_confined_output_path(
            self.run_root,
            stdout_archive,
            "simulator stdout archive",
        )
        self.stderr_archive = _validate_confined_output_path(
            self.run_root,
            stderr_archive,
            "simulator stderr archive",
        )
        if self.stdout_archive == self.stderr_archive:
            raise ProcessError("simulator archive output paths must be distinct")
        self._cancelled = cancelled
        self._poll_interval_s = poll_interval_s
        self._stop_grace_s = max(0.0, stop_grace_s)
        self._term_grace_s = max(0.0, term_grace_s)
        self._kill_grace_s = max(0.0, kill_grace_s)
        self._request_number = 0
        self._read_buffer = bytearray()
        self._closed = False
        self._sim_dt_s: float | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._pgid: int | None = None
        self._stdout_file = _open_exclusive_binary_output(
            self.run_root,
            self.stdout_archive,
            "simulator stdout archive",
        )
        try:
            self._stderr_file = _open_exclusive_binary_output(
                self.run_root,
                self.stderr_archive,
                "simulator stderr archive",
            )
        except BaseException:
            self._stdout_file.close()
            raise
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                cwd=None if cwd is None else str(Path(cwd)),
                env=None if env is None else dict(env),
                start_new_session=True,
                bufsize=0,
            )
            self._pgid = _verify_new_process_group(self._process)
        except BaseException as error:
            cleanup_ok = True
            if self._process is not None:
                cleanup_ok = _cleanup_expected_new_group(
                    self._process,
                    term_grace_s=self._term_grace_s,
                    kill_grace_s=self._kill_grace_s,
                )
                for stream in (self._process.stdin, self._process.stdout):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            self._stdout_file.close()
            self._stderr_file.close()
            if not cleanup_ok:
                raise ProcessError(
                    f"unverified child group {self._process.pid} survived cleanup"
                ) from error
            raise

    @property
    def pid(self) -> int:
        assert self._process is not None
        return self._process.pid

    @property
    def pgid(self) -> int:
        assert self._pgid is not None
        return self._pgid

    @property
    def returncode(self) -> int | None:
        assert self._process is not None
        return self._process.poll()

    @property
    def sim_dt(self) -> float:
        if self._sim_dt_s is None:
            raise ProcessError(
                "gated simulator reset must report SIMULATE_DT before gate creation"
            )
        return self._sim_dt_s

    def _stderr_tail(self) -> str:
        self._stderr_file.flush()
        try:
            raw = self.stderr_archive.read_bytes()[-8192:]
        except OSError:
            return ""
        return raw.decode("utf-8", errors="replace").strip()

    def _death_error(self) -> ChildProcessDied:
        returncode = self._process.poll()
        if returncode is None:
            try:
                returncode = self._process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                returncode = "unknown"
        detail = self._stderr_tail()
        suffix = f": {detail}" if detail else ""
        return ChildProcessDied(
            f"gated simulator child {self.pid} exited {returncode}{suffix}"
        )

    def require_alive(self) -> None:
        if self._closed:
            raise ChildProcessDied("gated simulator client is closed")
        if self._process.poll() is not None:
            raise self._death_error()

    def _next_request_id(self) -> str:
        result = f"g{self._request_number}"
        self._request_number += 1
        return result

    def _write_request(self, request: Mapping[str, object]) -> None:
        self.require_alive()
        assert self._process.stdin is not None
        payload = json.dumps(
            request,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            if self._process.poll() is not None:
                raise self._death_error() from error
            raise ProcessError(f"failed to write simulator request: {error}") from error

    def _read_line(
        self,
        *,
        deadline: float | None,
        wait_label: str,
    ) -> str:
        assert self._process.stdout is not None
        fd = self._process.stdout.fileno()
        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._read_buffer[: newline + 1])
                del self._read_buffer[: newline + 1]
                self._stdout_file.write(line)
                self._stdout_file.flush()
                try:
                    return line[:-1].decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ProcessProtocolError(
                        f"simulator response is not valid UTF-8: {error}"
                    ) from error
            if _cancelled(self._cancelled):
                raise OperatorCancelled("operator cancelled simulator request")
            if deadline is not None and time.monotonic() >= deadline:
                raise ProcessError(f"timed out waiting for {wait_label}")
            wait = self._poll_interval_s
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
            ready, _, _ = select.select(
                [fd],
                [],
                [],
                wait,
            )
            if ready:
                chunk = os.read(fd, 65536)
                if chunk:
                    self._read_buffer.extend(chunk)
                    continue
                if self._read_buffer:
                    partial = bytes(self._read_buffer)
                    self._stdout_file.write(partial)
                    self._stdout_file.flush()
                    self._read_buffer.clear()
                    raise ProcessProtocolError(
                        "simulator stdout ended with an unterminated JSONL response"
                    )
            if self._process.poll() is not None:
                raise self._death_error()

    def _read_line_without_deadline(self) -> str:
        return self._read_line(deadline=None, wait_label="simulator response")

    def _request(
        self,
        op: str,
        *,
        deadline: float | None = None,
        wait_label: str = "simulator response",
        **fields: object,
    ) -> dict[str, object]:
        request_id = self._next_request_id()
        request = {"v": _PROTOCOL_VERSION, "op": op, "request_id": request_id}
        request.update(fields)
        self._write_request(request)
        response = _loads_exact(
            self._read_line(deadline=deadline, wait_label=wait_label)
        )
        if type(response) is not dict:
            raise ProcessProtocolError("simulator response must be an object")
        if response.get("ok") is True:
            source = _exact_object(
                response,
                {"v", "ok", "op", "request_id", "data"},
                "success response",
            )
            data = source["data"]
        elif response.get("ok") is False:
            source = _exact_object(
                response,
                {"v", "ok", "op", "request_id", "error"},
                "error response",
            )
            error = _exact_object(
                source["error"], {"code", "message"}, "error response.error"
            )
            if type(error["code"]) is not str or type(error["message"]) is not str:
                raise ProcessProtocolError("error code and message must be strings")
            data = None
        else:
            raise ProcessProtocolError("simulator response ok must be a boolean")
        if source["v"] != _PROTOCOL_VERSION or type(source["v"]) is not int:
            raise ProcessProtocolError("simulator response version must equal 1")
        if source["op"] != op or type(source["op"]) is not str:
            raise ProcessProtocolError(
                "simulator response op mismatch: "
                f"expected {op!r}, found {source['op']!r}"
            )
        if source["request_id"] != request_id:
            raise ProcessProtocolError(
                "simulator response request_id mismatch: "
                f"expected {request_id!r}, found {source['request_id']!r}"
            )
        if source["ok"] is False:
            assert type(error) is dict
            raise ProcessProtocolError(f"{error['code']}: {error['message']}")
        if type(data) is not dict:
            raise ProcessProtocolError("success response data must be an object")
        return data

    def _drain_stdout_until(self, deadline: float) -> bytes:
        """Archive bytes after the close response until EOF or its deadline."""

        assert self._process.stdout is not None
        residual = bytearray()
        if self._read_buffer:
            buffered = bytes(self._read_buffer)
            self._read_buffer.clear()
            self._stdout_file.write(buffered)
            self._stdout_file.flush()
            residual.extend(buffered)
        fd = self._process.stdout.fileno()
        while time.monotonic() < deadline:
            wait = min(
                self._poll_interval_s,
                max(0.0, deadline - time.monotonic()),
            )
            ready, _, _ = select.select([fd], [], [], wait)
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                self._stdout_file.write(chunk)
                self._stdout_file.flush()
                residual.extend(chunk)
                continue
            if self._process.poll() is not None:
                # A final select iteration observes the pipe EOF after exit.
                continue
        return bytes(residual)

    def hello(self) -> dict[str, object]:
        data = _exact_object(self._request("hello"), {"protocol"}, "hello data")
        if data["protocol"] != "gated-sim/v1":
            raise ProcessProtocolError("unexpected gated simulator protocol")
        return data

    def reset(
        self,
        *,
        scene_xml: str | Path,
        initial_qpos: np.ndarray | Sequence[float],
        lateral_offset_m: float,
        yaw_offset_rad: float,
        log_dir: str | Path,
    ) -> dict[str, object]:
        self._sim_dt_s = None
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
        qpos = np.asarray(initial_qpos)
        if qpos.ndim != 1:
            raise ValueError("initial_qpos must be one-dimensional")
        data = _exact_object(
            self._request(
                "reset",
                scene_xml=str(scene),
                initial_qpos=qpos.tolist(),
                lateral_offset_m=float(lateral_offset_m),
                yaw_offset_rad=float(yaw_offset_rad),
                log_dir=str(logs),
            ),
            {"nq", "sim_dt_s", "sim_time_s"},
            "reset data",
        )
        if type(data["nq"]) is not int or data["nq"] <= 0:
            raise ProcessProtocolError("reset data.nq must be a positive integer")
        sim_dt = _finite_number(data["sim_dt_s"], "reset data.sim_dt_s")
        if sim_dt <= 0.0:
            raise ProcessProtocolError("reset data.sim_dt_s must be positive")
        sim_time = _finite_number(data["sim_time_s"], "reset data.sim_time_s")
        self._sim_dt_s = sim_dt
        return {"nq": data["nq"], "sim_dt_s": sim_dt, "sim_time_s": sim_time}

    def advance(self, steps: int) -> AdvanceResult:
        _positive_integer(steps, "steps")
        data = _exact_object(
            self._request("advance", steps=steps),
            {
                "steps",
                "sim_time_start_s",
                "sim_time_end_s",
                "state_rows",
                "contact_rows",
            },
            "advance data",
        )
        result = AdvanceResult(
            steps=_nonnegative_integer(data["steps"], "advance data.steps"),
            sim_time_start_s=_finite_number(
                data["sim_time_start_s"], "advance data.sim_time_start_s"
            ),
            sim_time_end_s=_finite_number(
                data["sim_time_end_s"], "advance data.sim_time_end_s"
            ),
            state_rows=_nonnegative_integer(
                data["state_rows"], "advance data.state_rows"
            ),
            contact_rows=_nonnegative_integer(
                data["contact_rows"], "advance data.contact_rows"
            ),
        )
        if result.steps != steps:
            raise ProcessProtocolError(
                f"simulator advanced {result.steps} steps, expected {steps}"
            )
        return result

    def snapshot(self) -> dict[str, object]:
        data = _exact_object(
            self._request("snapshot"),
            {"steps", "sim_time_s", "state_rows", "contact_rows"},
            "snapshot data",
        )
        return {
            "steps": _nonnegative_integer(data["steps"], "snapshot data.steps"),
            "sim_time_s": _finite_number(
                data["sim_time_s"], "snapshot data.sim_time_s"
            ),
            "state_rows": _nonnegative_integer(
                data["state_rows"], "snapshot data.state_rows"
            ),
            "contact_rows": _nonnegative_integer(
                data["contact_rows"], "snapshot data.contact_rows"
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        process = self._process
        assert process is not None
        assert self._pgid is not None
        close_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            group_exists = lambda: bool(_linux_group_states(self._pgid))
            if process.poll() is None:
                deadline = time.monotonic() + self._stop_grace_s
                try:
                    data = _exact_object(
                        self._request(
                            "close",
                            deadline=deadline,
                            wait_label="gated simulator close response",
                        ),
                        {"closed"},
                        "close data",
                    )
                    if data["closed"] is not True:
                        raise ProcessProtocolError(
                            "close data.closed must be boolean true"
                        )
                except BaseException as error:
                    close_error = error
                try:
                    residual = self._drain_stdout_until(deadline)
                    if residual and close_error is None:
                        close_error = ProcessProtocolError(
                            "residual stdout followed the gated simulator "
                            "close response"
                        )
                except BaseException as error:
                    if close_error is None:
                        close_error = error
            if group_exists():
                _safe_kill_created_group(process, self._pgid, signal.SIGTERM)
                deadline = time.monotonic() + self._term_grace_s
                while group_exists() and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.01)
            if group_exists():
                _safe_kill_created_group(process, self._pgid, signal.SIGKILL)
                deadline = time.monotonic() + self._kill_grace_s
                while group_exists() and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.01)
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, self._kill_grace_s))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            cleanup_error = error
        finally:
            try:
                final_deadline = time.monotonic() + max(
                    0.1,
                    self._poll_interval_s,
                    self._kill_grace_s,
                )
                residual = self._drain_stdout_until(final_deadline)
                if residual and close_error is None:
                    close_error = ProcessProtocolError(
                        "residual stdout remained at gated simulator shutdown"
                    )
            except BaseException as error:
                if close_error is None:
                    close_error = error
            self._closed = True
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._stdout_file.close()
            self._stderr_file.close()
        if _linux_group_states(self._pgid):
            raise ProcessError(
                f"gated simulator process group {self._pgid} survived cleanup"
            ) from (cleanup_error or close_error)
        if cleanup_error is not None:
            raise cleanup_error
        if close_error is not None:
            raise close_error

    def __enter__(self) -> "GatedSimulatorClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _linux_group_states(pgid: int) -> dict[int, str]:
    states: dict[int, str] = {}
    proc = Path("/proc")
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return states
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            close = raw.rfind(") ")
            fields = raw[close + 2 :].split()
            state = fields[0]
            member_pgid = int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if member_pgid == pgid:
            states[int(entry.name)] = state
    return states


class GearProcess:
    """Official GEAR deployment wrapper with a PTY and verified process group."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        command: Sequence[str],
        target_motion_logfile: str | Path,
        logs_dir: str | Path,
        stdout_archive: str | Path,
        stderr_archive: str | Path,
        startup_markers: Sequence[str] = _DEFAULT_STARTUP_MARKERS,
        active_markers: Sequence[str] = _DEFAULT_ACTIVE_MARKERS,
        cancelled: Callable[[], bool] | None = None,
        readiness_timeout_s: float | None = 60.0,
        readiness_poll_s: float = 0.05,
        signal_poll_s: float = 0.01,
        stop_grace_s: float = 2.0,
        term_grace_s: float = 2.0,
        kill_grace_s: float = 2.0,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if not command or any(type(item) is not str or not item for item in command):
            raise ValueError("command must contain nonempty strings")
        managed = [
            item
            for item in command
            if any(
                item == flag or item.startswith(f"{flag}=")
                for flag in _MANAGED_GEAR_FLAGS
            )
        ]
        if managed:
            raise ValueError(
                f"command contains managed GEAR flag {managed[0]!r}; "
                "the wrapper owns all integration flags"
            )
        self.run_root = _canonical_run_root(run_root)
        target = _validate_confined_output_path(
            self.run_root,
            target_motion_logfile,
            "target motion logfile",
        )
        logs = _validate_confined_output_path(
            self.run_root,
            logs_dir,
            "GEAR logs directory",
        )
        stdout_path = _validate_confined_output_path(
            self.run_root,
            stdout_archive,
            "GEAR stdout archive",
        )
        stderr_path = _validate_confined_output_path(
            self.run_root,
            stderr_archive,
            "GEAR stderr archive",
        )
        if len({target, stdout_path, stderr_path}) != 3:
            raise ProcessError("GEAR file output paths must be distinct")
        if logs in (target, stdout_path, stderr_path):
            raise ProcessError("GEAR logs directory cannot also be a file output")
        _create_exclusive_directory(
            self.run_root,
            logs,
            "GEAR logs directory",
        )
        for path, label in (
            (target, "target motion logfile"),
            (stdout_path, "GEAR stdout archive"),
            (stderr_path, "GEAR stderr archive"),
        ):
            _create_confined_parents(self.run_root, path, label=label)
        self.argv = tuple(
            command
        ) + (
            "--input-type",
            "zmq",
            "--target-motion-logfile",
            str(target),
            "--logs-dir",
            str(logs),
            "--enable-csv-logs",
            "--zmq-verbose",
        )
        self.target_motion_logfile = target
        self.logs_dir = logs
        self.stdout_archive = stdout_path
        self.stderr_archive = stderr_path
        if readiness_poll_s <= 0.0 or signal_poll_s <= 0.0:
            raise ValueError("poll intervals must be positive")
        self._startup_markers = tuple(startup_markers)
        self._active_markers = tuple(active_markers)
        markers = (*self._startup_markers, *self._active_markers)
        if any(type(marker) is not str or not marker for marker in markers):
            raise ValueError("readiness markers must be nonempty strings")
        self._cancelled = cancelled
        self._readiness_timeout_s = readiness_timeout_s
        self._readiness_poll_s = readiness_poll_s
        self._signal_poll_s = signal_poll_s
        self._stop_grace_s = max(0.0, stop_grace_s)
        self._term_grace_s = max(0.0, term_grace_s)
        self._kill_grace_s = max(0.0, kill_grace_s)
        self._env = None if env is None else dict(env)
        self._cwd = None if cwd is None else str(Path(cwd))
        self._process: subprocess.Popen[bytes] | None = None
        self._pgid: int | None = None
        self._master_fd: int | None = None
        self._stdout_file = None
        self._stderr_file = None
        self._reader_threads: list[threading.Thread] = []
        self._reader_errors: list[tuple[str, BaseException]] = []
        self._observed = ""
        self._output_condition = threading.Condition()
        self._closed = False
        self._ready = False
        self.signal_history: list[signal.Signals] = []
        self.cleanup_history: list[str] = []

    @property
    def pid(self) -> int:
        if self._process is None:
            raise ProcessError("GEAR process has not started")
        return self._process.pid

    @property
    def pgid(self) -> int:
        if self._pgid is None:
            raise ProcessError("GEAR process has not started")
        return self._pgid

    @property
    def returncode(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    @property
    def ready(self) -> bool:
        return self._ready

    def _reader(self, name: str, stream, archive) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                archive.write(chunk)
                archive.flush()
                text = chunk.decode("utf-8", errors="replace")
                with self._output_condition:
                    self._observed += text
                    self._output_condition.notify_all()
        except BaseException as error:
            with self._output_condition:
                self._reader_errors.append((name, error))
                self._output_condition.notify_all()
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _reader_failure_locked(self) -> ProcessError | None:
        if not self._reader_errors:
            return None
        name, error = self._reader_errors[0]
        failure = ProcessError(
            f"GEAR {name} archive reader failed: "
            f"{type(error).__name__}: {error}"
        )
        failure.__cause__ = error
        return failure

    def _reader_failure(self) -> ProcessError | None:
        with self._output_condition:
            return self._reader_failure_locked()

    def _wait_markers(
        self,
        markers: Sequence[str],
        *,
        after_offset: int = 0,
    ) -> None:
        if not markers:
            return
        deadline = (
            None
            if self._readiness_timeout_s is None
            else time.monotonic() + self._readiness_timeout_s
        )
        with self._output_condition:
            while not all(
                marker in self._observed[after_offset:] for marker in markers
            ):
                reader_failure = self._reader_failure_locked()
                if reader_failure is not None:
                    raise reader_failure
                if _cancelled(self._cancelled):
                    raise OperatorCancelled("operator cancelled GEAR readiness wait")
                if self._process is not None and self._process.poll() is not None:
                    missing = [
                        marker
                        for marker in markers
                        if marker not in self._observed[after_offset:]
                    ]
                    raise ChildProcessDied(
                        f"GEAR child {self.pid} exited {self._process.returncode} "
                        f"before readiness markers {missing!r}"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    missing = [
                        marker
                        for marker in markers
                        if marker not in self._observed[after_offset:]
                    ]
                    raise ProcessError(
                        f"timed out waiting for GEAR readiness markers {missing!r}"
                    )
                wait = self._readiness_poll_s
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - time.monotonic()))
                self._output_condition.wait(wait)
            reader_failure = self._reader_failure_locked()
            if reader_failure is not None:
                raise reader_failure

    def _write_key_then_wait(self, key: bytes, marker: str) -> None:
        # Prevent the reader from appending a key-triggered marker until the
        # causal boundary has been captured.
        with self._output_condition:
            boundary = len(self._observed)
            self.write_keys(key)
        self._wait_markers((marker,), after_offset=boundary)

    def start(self) -> None:
        if self._closed:
            raise ProcessError("GEAR process is closed")
        if self._process is not None:
            raise ProcessError("GEAR process has already started")
        master, slave = pty.openpty()
        self._master_fd = master
        try:
            logs = _confined_existing_path(
                self.run_root,
                self.logs_dir,
                "GEAR logs directory",
                directory=True,
            )
            try:
                if any(logs.iterdir()):
                    raise ProcessError(
                        "GEAR logs directory must remain empty before start"
                    )
            except OSError as error:
                raise ProcessError("cannot inspect GEAR logs directory") from error
            _validate_confined_output_path(
                self.run_root,
                self.target_motion_logfile,
                "target motion logfile",
            )
            self._stdout_file = _open_exclusive_binary_output(
                self.run_root,
                self.stdout_archive,
                "GEAR stdout archive",
            )
            self._stderr_file = _open_exclusive_binary_output(
                self.run_root,
                self.stderr_archive,
                "GEAR stderr archive",
            )
            try:
                self._process = subprocess.Popen(
                    self.argv,
                    stdin=slave,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self._env,
                    cwd=self._cwd,
                    start_new_session=True,
                    bufsize=0,
                )
            finally:
                os.close(slave)
            self._pgid = _verify_new_process_group(self._process)
            assert self._process.stdout is not None
            assert self._process.stderr is not None
            for name, stream, archive in (
                ("stdout", self._process.stdout, self._stdout_file),
                ("stderr", self._process.stderr, self._stderr_file),
            ):
                thread = threading.Thread(
                    target=self._reader,
                    args=(name, stream, archive),
                    name=f"gear-{name}-{self.pid}",
                    daemon=True,
                )
                thread.start()
                self._reader_threads.append(thread)
            self._wait_markers(self._startup_markers)
            if len(self._active_markers) != 2:
                raise ProcessError(
                    "GEAR active_markers must contain CONTROL then STREAMING"
                )
            self._write_key_then_wait(b"]", self._active_markers[0])
            self._write_key_then_wait(b"\n", self._active_markers[1])
            self._ready = True
        except BaseException as error:
            try:
                os.close(slave)
            except OSError:
                pass
            cleanup_ok = True
            if self._process is not None and self._pgid is None:
                cleanup_ok = _cleanup_expected_new_group(
                    self._process,
                    term_grace_s=self._term_grace_s,
                    kill_grace_s=self._kill_grace_s,
                )
            self.close()
            if not cleanup_ok:
                raise ProcessError(
                    f"unverified GEAR group {self._process.pid} survived cleanup"
                ) from error
            raise

    def write_keys(self, keys: bytes) -> None:
        if type(keys) is not bytes or not keys:
            raise ValueError("keys must be nonempty bytes")
        self.require_alive()
        if self._master_fd is None:
            raise ProcessError("GEAR PTY is closed")
        try:
            written = os.write(self._master_fd, keys)
        except OSError as error:
            raise ProcessError(f"failed to write GEAR PTY: {error}") from error
        if written != len(keys):
            raise ProcessError(
                f"short GEAR PTY write: wrote {written} of {len(keys)} bytes"
            )

    def require_alive(self) -> None:
        reader_failure = self._reader_failure()
        if reader_failure is not None:
            raise reader_failure
        if self._process is None:
            raise ChildProcessDied("GEAR process has not started")
        returncode = self._process.poll()
        if returncode is not None:
            raise ChildProcessDied(f"GEAR child {self.pid} exited {returncode}")

    def _send_group_signal(self, sig: signal.Signals) -> bool:
        if self._process is None or self._pgid is None:
            raise ProcessError("GEAR process has not started")
        sent = _safe_kill_created_group(self._process, self._pgid, sig)
        if sent:
            self.signal_history.append(sig)
        return sent

    def group_is_stopped(self) -> bool:
        if self._pgid is None:
            return False
        states = _linux_group_states(self._pgid)
        live = [state for state in states.values() if state != "Z"]
        return bool(live) and all(state in ("T", "t") for state in live)

    def group_is_resumed(self) -> bool:
        if self._pgid is None:
            return False
        states = _linux_group_states(self._pgid)
        live = [state for state in states.values() if state != "Z"]
        return bool(live) and all(state not in ("T", "t") for state in live)

    def stop_group(self) -> None:
        if self._process is None or self._pgid is None:
            raise ProcessError("GEAR process has not started")
        if not _linux_group_states(self._pgid):
            self.require_alive()
        self._send_group_signal(signal.SIGSTOP)
        deadline = time.monotonic() + 2.0
        while not self.group_is_stopped():
            if not _linux_group_states(self._pgid):
                self.require_alive()
            if time.monotonic() >= deadline:
                raise ProcessError(f"GEAR process group {self.pgid} did not stop")
            time.sleep(self._signal_poll_s)
        # Report policy-leader death only after every surviving descendant is
        # safely stopped.
        self.require_alive()

    def continue_group(self) -> None:
        self.require_alive()
        self._send_group_signal(signal.SIGCONT)
        deadline = time.monotonic() + 2.0
        while not self.group_is_resumed():
            self.require_alive()
            if time.monotonic() >= deadline:
                raise ProcessError(f"GEAR process group {self.pgid} did not continue")
            time.sleep(self._signal_poll_s)

    def _group_exists(self) -> bool:
        if self._pgid is None:
            return False
        return bool(_linux_group_states(self._pgid))

    def _wait_group_exit(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while self._group_exists() and time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is None:
                self._process.poll()
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        if (
            self._process is not None
            and self._process.poll() is None
            and not self._group_exists()
        ):
            try:
                self._process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        return not self._group_exists()

    def _finish_reader_threads(
        self,
        process: subprocess.Popen[bytes] | None,
    ) -> ProcessError | None:
        deadline = time.monotonic() + max(0.1, self._kill_grace_s)
        for thread in self._reader_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        incomplete = [thread for thread in self._reader_threads if thread.is_alive()]
        if not incomplete:
            return None

        # Group termination should have produced EOF.  Force-close the parent
        # pipe handles only after the bounded EOF join, then enforce a final
        # bounded join so daemon readers cannot silently outlive cleanup.
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        forced_deadline = time.monotonic() + 0.1
        for thread in incomplete:
            thread.join(timeout=max(0.0, forced_deadline - time.monotonic()))
        names = ", ".join(thread.name for thread in incomplete)
        return ProcessError(
            f"GEAR reader threads did not drain to EOF before bound: {names}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        cleanup_error: BaseException | None = None
        evidence_error: BaseException | None = None
        try:
            if process is not None and self._group_exists():
                self.cleanup_history.append("official-stop-key")
                if self._master_fd is not None:
                    try:
                        os.write(self._master_fd, b"o")
                    except OSError:
                        pass
                if self.group_is_stopped():
                    try:
                        self._send_group_signal(signal.SIGCONT)
                    except ProcessError:
                        pass
                self._wait_group_exit(self._stop_grace_s)
            if process is not None and self._group_exists():
                self.cleanup_history.append("SIGTERM")
                try:
                    self._send_group_signal(signal.SIGTERM)
                except ProcessError:
                    pass
                self._wait_group_exit(self._term_grace_s)
            if process is not None and self._group_exists():
                self.cleanup_history.append("SIGKILL")
                try:
                    self._send_group_signal(signal.SIGKILL)
                except ProcessError:
                    pass
                self._wait_group_exit(self._kill_grace_s)
            if process is not None:
                try:
                    process.wait(timeout=max(0.1, self._kill_grace_s))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            cleanup_error = error
        finally:
            if self._master_fd is not None:
                try:
                    os.close(self._master_fd)
                except OSError:
                    pass
                self._master_fd = None
            incomplete = self._finish_reader_threads(process)
            if incomplete is not None:
                evidence_error = incomplete
            reader_failure = self._reader_failure()
            if reader_failure is not None and evidence_error is None:
                evidence_error = reader_failure
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            for archive in (self._stdout_file, self._stderr_file):
                if archive is not None:
                    try:
                        archive.close()
                    except BaseException as error:
                        if evidence_error is None:
                            evidence_error = ProcessError(
                                f"failed to close GEAR archive: {error}"
                            )
            self._ready = False
        if self._group_exists():
            raise ProcessError(
                f"GEAR process group {self.pgid} survived cleanup"
            ) from (cleanup_error or evidence_error)
        if cleanup_error is not None:
            raise cleanup_error
        if evidence_error is not None:
            raise evidence_error

    def __enter__(self) -> "GearProcess":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class SimulationPolicyGate:
    """Release GEAR only while the simulator advances an exact step count."""

    def __init__(
        self,
        gear: GearProcess,
        simulator: GatedSimulatorClient,
        *,
        tolerance: float = 1.0e-12,
    ) -> None:
        sim_dt = simulator.sim_dt
        if (
            type(sim_dt) not in (int, float)
            or not math.isfinite(float(sim_dt))
            or sim_dt <= 0
        ):
            raise ValueError("sim_dt must be a positive finite number")
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and nonnegative")
        self.gear = gear
        self.simulator = simulator
        self.sim_dt = float(sim_dt)
        self.tolerance = float(tolerance)
        self._paused = False
        self._closed = False

    def _require_bound_sim_dt(self) -> None:
        current = self.simulator.sim_dt
        if (
            type(current) not in (int, float)
            or not math.isfinite(float(current))
            or float(current) != self.sim_dt
        ):
            raise ProcessProtocolError(
                "simulator-reported SIMULATE_DT changed after gate binding"
            )

    @property
    def is_paused(self) -> bool:
        return self._paused and self.gear.group_is_stopped()

    def require_paused(self) -> None:
        if not self.is_paused:
            raise RuntimeError("simulation policy gate must be paused")

    def pause(self) -> None:
        if self._closed:
            raise RuntimeError("simulation policy gate is closed")
        try:
            if not self._paused or not self.gear.group_is_stopped():
                self.gear.stop_group()
        finally:
            self._paused = self.gear.group_is_stopped()
        if not self.gear.group_is_stopped():
            raise ProcessError("GEAR process group did not verify as stopped")
        self.simulator.require_alive()

    def release_steps(self, steps: int) -> AdvanceResult:
        _positive_integer(steps, "steps")
        self.require_paused()
        self._require_bound_sim_dt()
        self.gear.require_alive()
        self.simulator.require_alive()
        self._paused = False
        result: AdvanceResult | None = None
        operation_error: BaseException | None = None
        try:
            self.gear.continue_group()
            result = self.simulator.advance(steps)
        except BaseException as error:
            operation_error = error
        stop_error: BaseException | None = None
        try:
            self.gear.stop_group()
        except BaseException as error:
            stop_error = error
        self._paused = self.gear.group_is_stopped()
        if operation_error is not None:
            if stop_error is not None and not self._paused:
                raise stop_error from operation_error
            raise operation_error
        if stop_error is not None:
            raise stop_error
        assert result is not None
        if result.steps != steps:
            raise ProcessProtocolError(
                f"simulator reply reports {result.steps} steps, expected {steps}"
            )
        expected = steps * self.sim_dt
        actual = result.sim_time_end_s - result.sim_time_start_s
        if not math.isfinite(actual) or abs(actual - expected) > self.tolerance:
            raise ProcessProtocolError(
                "MuJoCo time delta mismatch: "
                f"expected {expected:.17g}, found {actual:.17g}"
            )
        return result

    def advance(self, duration_s: float) -> AdvanceResult:
        if type(duration_s) not in (int, float) or not math.isfinite(
            float(duration_s)
        ):
            raise ValueError("duration / SIMULATE_DT must be an exact positive integer")
        ratio = float(duration_s) / self.sim_dt
        nearest = round(ratio)
        if nearest <= 0 or abs(ratio - nearest) > self.tolerance:
            raise ValueError("duration / SIMULATE_DT must be an exact positive integer")
        return self.release_steps(int(nearest))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        simulator_error = None
        try:
            self.simulator.close()
        except BaseException as error:  # preserve GEAR cleanup even on child failure
            simulator_error = error
        self.gear.close()
        if simulator_error is not None:
            raise simulator_error

    def __enter__(self) -> "SimulationPolicyGate":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
