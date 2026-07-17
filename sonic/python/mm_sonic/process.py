"""Explicit subprocess lifecycles for the gated SONIC simulation.

The orchestration boundary is deliberately dependency-light: importing this
module never imports GEAR, MuJoCo, Unitree, or DDS.  Those imports belong to the
``mm_sonic.gated_sim`` child only.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import pty
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from .joints import ContractError
from .transform import mujoco_to_holden_quaternions, mujoco_to_holden_vectors


_PROTOCOL_VERSION = 1
_MANAGED_GEAR_FLAGS = frozenset(
    {
        "--input-type",
        "--target-motion-logfile",
        "--logs-dir",
        "--enable-csv-logs",
        "--disable-crc-check",
        "--zmq-conflate",
        "--zmq-verbose",
    }
)
_DEFAULT_STARTUP_MARKERS = ("Initialized ZMQ endpoint interface",)
_WAIT_FOR_CONTROL_MARKER = "Init Done"
_CONTROL_ACTIVE_MARKER = (
    "[Control] DEBUG: operator_state.start=true, transitioning to CONTROL state"
)
_DEFAULT_ACTIVE_MARKERS = (
    _CONTROL_ACTIVE_MARKER,
    "ZMQ STREAMING MODE: ENABLED",
)
_LOADED_MOTION_STARTUP_MARKERS = (
    "✓ Motion data loaded successfully!",
    "Started with motion:",
    "(paused at frame 0)",
    "Initialized keyboard input interface (default)",
)
_LOADED_MOTION_ACTIVE_MARKERS = (
    _CONTROL_ACTIVE_MARKER,
    "Playing motion 0 from frame 0 to end (",
)
_LOADED_MOTION_RESET_MARKER = "Reset motion 0 to frame 0 (paused)"
_STREAM_PROCESSING_START_MARKER = (
    "[ZMQEndpointInterface] *** Starting ZMQ processing ***"
)
_STREAM_PROCESSING_END_MARKER = (
    "[ZMQEndpointInterface] *** End of ZMQ decoding processing ***"
)
_STREAM_MERGER_PROCESSING_PREFIX = "[StreamedMotionMerger] Processing "
_STREAM_MERGER_MERGED_PREFIX = "[StreamedMotionMerger] Merged motion: "
_POST_ENABLE_LEFT_LINE = "Delta heading left: 0.1 rad"
_POST_ENABLE_RIGHT_LINE = "Delta heading right: 0 rad"
_POST_ENABLE_FENCE_SEMANTICS = (
    "post-enable-reset-tail-complete-with-net-zero-heading"
)
_GEAR_LAUNCH_PROFILES = frozenset({"zmq_stream", "loaded_motion"})
_SHELL_SAFE_ABSOLUTE_PATH = re.compile(r"/[A-Za-z0-9._/-]*\Z")
_OWNED_DIRECTORY_STAGING_PREFIX = ".mm-sonic-owned-"
_RENAME_NOREPLACE = 1


class ProcessError(RuntimeError):
    """Base class for a controlled child-process failure."""


class ProcessProtocolError(ProcessError):
    """A JSONL peer violated the gated simulator protocol."""


class ChildProcessDied(ProcessError):
    """A required child exited before completing its operation."""


class OperatorCancelled(ProcessError):
    """The operator requested cancellation."""


def _at_failure_site(error: BaseException, site: str) -> BaseException:
    """Preserve an exception type while attaching its coordinator phase."""

    if getattr(error, "failure_site", None) is None:
        try:
            error.failure_site = site
        except BaseException:
            tagged = ProcessError(str(error))
            tagged.failure_site = site
            tagged.__cause__ = error
            return tagged
    return error


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


def _require_shell_safe_absolute_path(value: Path, label: str) -> None:
    if not value.is_absolute() or _SHELL_SAFE_ABSOLUTE_PATH.fullmatch(str(value)) is None:
        raise ProcessError(
            f"{label} must be a conservative shell-safe absolute path"
        )


def _gear_process_argv(
    command: Sequence[str],
    *,
    launch_profile: str,
    target_motion_logfile: str | Path,
    logs_dir: str | Path,
) -> tuple[str, ...]:
    integration_flags = (
        "--input-type",
        "zmq" if launch_profile == "zmq_stream" else "keyboard",
        "--target-motion-logfile",
        str(target_motion_logfile),
        "--logs-dir",
        str(logs_dir),
        "--enable-csv-logs",
        "--disable-crc-check",
    )
    if launch_profile == "zmq_stream":
        integration_flags += ("--zmq-verbose",)
    return tuple(command) + integration_flags


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


@dataclass(frozen=True)
class _OwnedDirectory:
    parent_fd: int
    leaf_fd: int
    name: str
    device: int
    inode: int


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _rename_noreplace(parent_fd: int, source: str, target: str) -> None:
    """Atomically publish one directory name without replacing a peer."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ProcessError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def _open_confined_parent_fd(
    run_root: Path,
    candidate: Path,
    label: str,
) -> tuple[int, str]:
    """Open a candidate's parent from the authenticated root without links."""

    try:
        relative = candidate.relative_to(run_root)
    except ValueError as error:
        raise ProcessError(f"{label} must be beneath run_root") from error
    parts = relative.parts
    if not parts:
        raise ProcessError(f"{label} must be a descendant beneath run_root")
    flags = _directory_open_flags()
    parent_fd: int | None = None
    retained = False
    try:
        expected_root = run_root.stat(follow_symlinks=False)
        parent_fd = os.open(run_root, flags)
        opened_root = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or opened_root.st_dev != expected_root.st_dev
            or opened_root.st_ino != expected_root.st_ino
        ):
            raise ProcessError(f"{label} run_root identity changed")
        for part in parts[:-1]:
            next_fd = os.open(part, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        retained = True
        return parent_fd, parts[-1]
    except ProcessError:
        raise
    except OSError as error:
        raise ProcessError(f"cannot open {label} parent") from error
    finally:
        if parent_fd is not None and not retained:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _create_owned_directory(
    run_root: Path,
    value: str | Path,
    label: str,
) -> tuple[Path, _OwnedDirectory]:
    """Create, authenticate, and atomically publish an owned directory."""

    candidate = _prepare_confined_output_path(run_root, value, label)
    parent_fd, name = _open_confined_parent_fd(run_root, candidate, label)
    leaf_fd: int | None = None
    staging_name: str | None = None
    staging_device: int | None = None
    staging_inode: int | None = None
    published = False
    retained = False
    try:
        for _ in range(8):
            proposed = _OWNED_DIRECTORY_STAGING_PREFIX + secrets.token_hex(16)
            try:
                os.mkdir(proposed, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            staging_name = proposed
            break
        if staging_name is None:
            raise ProcessError(
                f"cannot allocate private staging directory for {label}"
            )
        created_staging = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(created_staging.st_mode):
            raise ProcessError(f"{label} staging identity changed during creation")
        staging_device = created_staging.st_dev
        staging_inode = created_staging.st_ino
        leaf_fd = os.open(
            staging_name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        leaf = os.fstat(leaf_fd)
        staged = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(leaf.st_mode)
            or not stat.S_ISDIR(staged.st_mode)
            or leaf.st_dev != staging_device
            or leaf.st_ino != staging_inode
            or staged.st_dev != leaf.st_dev
            or staged.st_ino != leaf.st_ino
        ):
            raise ProcessError(f"{label} identity changed during creation")
        try:
            _rename_noreplace(parent_fd, staging_name, name)
        except FileExistsError as error:
            raise ProcessError(f"{label} already exists: {candidate}") from error
        published = True
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != leaf.st_dev
            or current.st_ino != leaf.st_ino
        ):
            raise ProcessError(f"{label} identity changed during creation")
        retained = True
        return candidate, _OwnedDirectory(
            parent_fd=parent_fd,
            leaf_fd=leaf_fd,
            name=name,
            device=leaf.st_dev,
            inode=leaf.st_ino,
        )
    except ProcessError:
        raise
    except OSError as error:
        raise ProcessError(f"cannot create {label}: {candidate}") from error
    finally:
        if not retained:
            remove_created = False
            cleanup_name = name if published else staging_name
            if cleanup_name is not None:
                try:
                    if leaf_fd is not None:
                        original = os.fstat(leaf_fd)
                        expected_device = original.st_dev
                        expected_inode = original.st_ino
                        original_is_directory = stat.S_ISDIR(original.st_mode)
                    else:
                        expected_device = staging_device
                        expected_inode = staging_inode
                        original_is_directory = (
                            expected_device is not None
                            and expected_inode is not None
                        )
                    current = os.stat(
                        cleanup_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    remove_created = (
                        original_is_directory
                        and stat.S_ISDIR(current.st_mode)
                        and current.st_dev == expected_device
                        and current.st_ino == expected_inode
                    )
                except OSError:
                    remove_created = False
            if remove_created:
                try:
                    assert cleanup_name is not None
                    os.rmdir(cleanup_name, dir_fd=parent_fd)
                except OSError:
                    pass
            if leaf_fd is not None:
                try:
                    os.close(leaf_fd)
                except OSError:
                    pass
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _remove_owned_empty_directory(owned: _OwnedDirectory) -> None:
    """Remove only the empty leaf created under the retained parent."""

    try:
        try:
            original = os.fstat(owned.leaf_fd)
        except OSError:
            return
        try:
            current = os.stat(
                owned.name,
                dir_fd=owned.parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            return
        if (
            not stat.S_ISDIR(original.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or original.st_dev != owned.device
            or original.st_ino != owned.inode
            or current.st_dev != original.st_dev
            or current.st_ino != original.st_ino
        ):
            return
        try:
            os.rmdir(owned.name, dir_fd=owned.parent_fd)
        except OSError:
            # Nonempty evidence or a concurrent identity change is retained.
            pass
    finally:
        try:
            os.close(owned.leaf_fd)
        except OSError:
            pass
        try:
            os.close(owned.parent_fd)
        except OSError:
            pass


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


class _RemoteMMError(ProcessProtocolError):
    """A valid MM error response, which proves no successful candidate reply."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class MMChunkClient:
    """Strict persistent MM JSONL client with unbounded generation waits.

    Candidate ownership becomes conservative after the first request byte is
    written.  A zero-byte write failure is the only request failure that proves
    the candidate cannot be outstanding; partial/full writes and malformed or
    lost responses retain the coordinator-owned candidate ID for abort.
    """

    def __init__(
        self,
        *,
        run_root: str | Path,
        command: Sequence[str],
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
        if not command or any(type(item) is not str or not item for item in command):
            raise ValueError("MM command must contain nonempty strings")
        if poll_interval_s <= 0.0:
            raise ValueError("MM poll_interval_s must be positive")
        self.command = tuple(command)
        self.stdout_archive = _validate_confined_output_path(
            self.run_root, stdout_archive, "MM stdout archive"
        )
        self.stderr_archive = _validate_confined_output_path(
            self.run_root, stderr_archive, "MM stderr archive"
        )
        if self.stdout_archive == self.stderr_archive:
            raise ProcessError("MM archive output paths must be distinct")
        self._cancelled = cancelled
        self._poll_interval_s = poll_interval_s
        self._stop_grace_s = max(0.0, stop_grace_s)
        self._term_grace_s = max(0.0, term_grace_s)
        self._kill_grace_s = max(0.0, kill_grace_s)
        self._request_number = 0
        self._read_buffer = bytearray()
        self._closed = False
        self._session_id: str | None = None
        self.outstanding_candidate_id: str | None = None
        self.active_candidate_id: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._pgid: int | None = None
        self._stdout_file = _open_exclusive_binary_output(
            self.run_root, self.stdout_archive, "MM stdout archive"
        )
        try:
            self._stderr_file = _open_exclusive_binary_output(
                self.run_root, self.stderr_archive, "MM stderr archive"
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
                    f"unverified MM group {self._process.pid} survived cleanup"
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

    def _stderr_tail(self) -> str:
        self._stderr_file.flush()
        try:
            raw = self.stderr_archive.read_bytes()[-8192:]
        except OSError:
            return ""
        return raw.decode("utf-8", errors="replace").strip()

    def _death_error(self) -> ChildProcessDied:
        assert self._process is not None
        returncode: int | str | None = self._process.poll()
        if returncode is None:
            try:
                returncode = self._process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                returncode = "unknown"
        detail = self._stderr_tail()
        suffix = f": {detail}" if detail else ""
        return ChildProcessDied(
            f"MM chunk server {self.pid} exited {returncode}{suffix}"
        )

    def require_alive(self) -> None:
        if self._closed:
            raise ChildProcessDied("MM chunk client is closed")
        assert self._process is not None
        if self._process.poll() is not None:
            raise self._death_error()

    def _next_request_id(self) -> str:
        value = f"m{self._request_number}"
        self._request_number += 1
        return value

    def _write_request(
        self,
        request: Mapping[str, object],
        *,
        candidate_id: str | None = None,
    ) -> None:
        self.require_alive()
        assert self._process is not None and self._process.stdin is not None
        try:
            payload = json.dumps(
                request,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            raise ProcessProtocolError(f"invalid MM request: {error}") from error
        view = memoryview(payload)
        written_total = 0
        fd = self._process.stdin.fileno()
        try:
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("zero-byte MM request write")
                written_total += written
                if candidate_id is not None and written_total > 0:
                    self.outstanding_candidate_id = candidate_id
                view = view[written:]
        except (BrokenPipeError, OSError) as error:
            if self._process.poll() is not None:
                raise self._death_error() from error
            qualifier = "before first byte" if written_total == 0 else "after partial write"
            raise ProcessError(f"failed to write MM request {qualifier}: {error}") from error

    def _read_line(self) -> str:
        assert self._process is not None and self._process.stdout is not None
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
                        f"MM response is not valid UTF-8: {error}"
                    ) from error
            if _cancelled(self._cancelled):
                raise OperatorCancelled("operator cancelled MM request")
            ready, _, _ = select.select([fd], [], [], self._poll_interval_s)
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
                        "MM stdout ended with an unterminated JSONL response"
                    )
            if self._process.poll() is not None:
                raise self._death_error()

    def _request(
        self,
        op: str,
        *,
        candidate_write_id: str | None = None,
        **fields: object,
    ) -> dict[str, object]:
        request_id = self._next_request_id()
        request = {"v": _PROTOCOL_VERSION, "op": op, "request_id": request_id}
        request.update(fields)
        self._write_request(request, candidate_id=candidate_write_id)
        response = _loads_exact(self._read_line())
        if type(response) is not dict:
            raise ProcessProtocolError("MM response must be an object")
        if response.get("ok") is True:
            envelope = _exact_object(
                response,
                {"v", "ok", "op", "request_id", "data"},
                "MM success response",
            )
            data = envelope["data"]
        elif response.get("ok") is False:
            envelope = _exact_object(
                response,
                {"v", "ok", "op", "request_id", "error"},
                "MM error response",
            )
            remote = _exact_object(
                envelope["error"], {"code", "message"}, "MM error response.error"
            )
            if type(remote["code"]) is not str or type(remote["message"]) is not str:
                raise ProcessProtocolError("MM error code and message must be strings")
            data = None
        else:
            raise ProcessProtocolError("MM response ok must be a boolean")
        if envelope["v"] != _PROTOCOL_VERSION or type(envelope["v"]) is not int:
            raise ProcessProtocolError("MM response version must equal 1")
        if envelope["op"] != op or type(envelope["op"]) is not str:
            raise ProcessProtocolError("MM response op mismatch")
        if envelope["request_id"] != request_id:
            raise ProcessProtocolError("MM response request_id mismatch")
        if envelope["ok"] is False:
            assert type(remote) is dict
            raise _RemoteMMError(remote["code"], remote["message"])
        if type(data) is not dict:
            raise ProcessProtocolError("MM success response data must be an object")
        return data

    @staticmethod
    def _identifier(value: object, label: str) -> str:
        if type(value) is not str or not value or "\x00" in value:
            raise ValueError(f"{label} must be a nonempty string")
        return value

    def hello(self) -> dict[str, object]:
        data = self._request("hello")
        if data.get("protocol_version") != 1 or type(data.get("protocol_version")) is not int:
            raise ProcessProtocolError("unexpected MM protocol version")
        return data

    def reset(self, config: object, *, session_id: str) -> dict[str, object]:
        if self.outstanding_candidate_id is not None:
            raise ProcessError("cannot reset MM with an outstanding candidate")
        session = self._identifier(session_id, "session_id")
        scene_id = self._identifier(getattr(config, "scene_id", None), "scene_id")
        route_id = self._identifier(getattr(config, "route_id", None), "route_id")
        terrain_weight = getattr(config, "terrain_weight", None)
        if type(terrain_weight) not in (int, float) or not math.isfinite(
            float(terrain_weight)
        ):
            raise ValueError("terrain_weight must be finite")
        data = self._request(
            "reset",
            session_id=session,
            scene_id=scene_id,
            route_id=route_id,
            terrain_weight=float(terrain_weight),
        )
        source = _exact_object(
            data,
            {"session_id", "active_candidate_id", "scene", "initial_boundary"},
            "MM reset data",
        )
        if source["session_id"] != session or source["active_candidate_id"] is not None:
            raise ProcessProtocolError("MM reset state identity mismatch")
        if type(source["scene"]) is not dict or type(source["initial_boundary"]) is not dict:
            raise ProcessProtocolError("MM reset data contains invalid objects")
        self._session_id = session
        self.active_candidate_id = None
        self.outstanding_candidate_id = None
        return source

    def generate(
        self,
        command: object,
        *,
        session_id: str,
        candidate_id: str,
        predecessor_id: str | None,
        source_intervals: int,
    ) -> dict[str, object]:
        if self.outstanding_candidate_id is not None:
            raise ProcessError("MM already has an outstanding candidate")
        session = self._identifier(session_id, "session_id")
        candidate = self._identifier(candidate_id, "candidate_id")
        if session != self._session_id:
            raise ProcessError("MM generate session does not match reset")
        if predecessor_id is not None:
            predecessor_id = self._identifier(predecessor_id, "predecessor_id")
        if predecessor_id != self.active_candidate_id:
            raise ProcessError("MM generate predecessor does not match active state")
        if type(source_intervals) is not int or source_intervals != 10:
            raise ValueError("MM source_intervals must equal 10")
        try:
            velocity = mujoco_to_holden_vectors(
                getattr(command, "requested_velocity_mujoco")
            )
            heading = mujoco_to_holden_quaternions(
                getattr(command, "desired_heading_mujoco_wxyz")
            )
        except (AttributeError, ContractError) as error:
            raise ValueError("MM command has an invalid target-basis value") from error
        try:
            data = self._request(
                "generate",
                candidate_write_id=candidate,
                session_id=session,
                candidate_id=candidate,
                predecessor_id=predecessor_id,
                source_intervals=source_intervals,
                requested_velocity_holden=velocity.tolist(),
                desired_heading_holden_wxyz=heading.tolist(),
            )
        except _RemoteMMError:
            # A valid server error response proves it did not publish a
            # successful candidate for this request.
            self.outstanding_candidate_id = None
            raise
        if data.get("session_id") != session or data.get("candidate_id") != candidate:
            raise ProcessProtocolError("MM generated candidate identity mismatch")
        if data.get("predecessor_id") != predecessor_id:
            raise ProcessProtocolError("MM generated predecessor identity mismatch")
        return data

    def commit(self, candidate_id: str) -> None:
        candidate = self._identifier(candidate_id, "candidate_id")
        if candidate != self.outstanding_candidate_id or self._session_id is None:
            raise ProcessError("MM commit candidate does not match outstanding state")
        data = self._request(
            "commit", session_id=self._session_id, candidate_id=candidate
        )
        source = _exact_object(
            data,
            {"session_id", "candidate_id", "active_candidate_id"},
            "MM commit data",
        )
        if (
            source["session_id"] != self._session_id
            or source["candidate_id"] != candidate
            or source["active_candidate_id"] != candidate
        ):
            raise ProcessProtocolError("MM commit identity mismatch")
        self.active_candidate_id = candidate
        self.outstanding_candidate_id = None

    def abort(self, candidate_id: str) -> None:
        candidate = self._identifier(candidate_id, "candidate_id")
        if candidate != self.outstanding_candidate_id or self._session_id is None:
            raise ProcessError("MM abort candidate does not match outstanding state")
        data = self._request(
            "abort", session_id=self._session_id, candidate_id=candidate
        )
        source = _exact_object(
            data,
            {"session_id", "candidate_id", "active_candidate_id"},
            "MM abort data",
        )
        if (
            source["session_id"] != self._session_id
            or source["candidate_id"] != candidate
            or source["active_candidate_id"] != self.active_candidate_id
        ):
            raise ProcessProtocolError("MM abort identity mismatch")
        self.outstanding_candidate_id = None

    def _drain_stdout(self, deadline: float) -> bytes:
        assert self._process is not None and self._process.stdout is not None
        residual = bytearray()
        if self._read_buffer:
            residual.extend(self._read_buffer)
            self._stdout_file.write(self._read_buffer)
            self._stdout_file.flush()
            self._read_buffer.clear()
        fd = self._process.stdout.fileno()
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [fd], [], [], min(self._poll_interval_s, max(0.0, deadline - time.monotonic()))
            )
            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                residual.extend(chunk)
                self._stdout_file.write(chunk)
                self._stdout_file.flush()
            elif self._process.poll() is not None:
                continue
        return bytes(residual)

    def close(self) -> None:
        if self._closed:
            return
        process = self._process
        assert process is not None and self._pgid is not None
        close_error: BaseException | None = None
        try:
            if process.poll() is None:
                try:
                    data = self._request("close")
                    if data != {}:
                        raise ProcessProtocolError(
                            "MM close data must be the protocol v1 empty object"
                        )
                except BaseException as error:
                    close_error = error
                deadline = time.monotonic() + self._stop_grace_s
                try:
                    residual = self._drain_stdout(deadline)
                    if residual and close_error is None:
                        close_error = ProcessProtocolError(
                            "residual stdout followed the MM close response"
                        )
                except BaseException as error:
                    if close_error is None:
                        close_error = error
            if _linux_group_states(self._pgid):
                _safe_kill_created_group(process, self._pgid, signal.SIGTERM)
                deadline = time.monotonic() + self._term_grace_s
                while _linux_group_states(self._pgid) and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.01)
            if _linux_group_states(self._pgid):
                _safe_kill_created_group(process, self._pgid, signal.SIGKILL)
                deadline = time.monotonic() + self._kill_grace_s
                while _linux_group_states(self._pgid) and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.01)
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, self._kill_grace_s))
                except subprocess.TimeoutExpired:
                    pass
        finally:
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
            raise ProcessError(f"MM process group {self._pgid} survived cleanup") from close_error
        if close_error is not None and not isinstance(close_error, ChildProcessDied):
            raise close_error

    def __enter__(self) -> "MMChunkClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


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
        launch_profile: str = "zmq_stream",
        startup_markers: Sequence[str] | None = None,
        active_markers: Sequence[str] | None = None,
        wait_for_control_marker: str = _WAIT_FOR_CONTROL_MARKER,
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
        try:
            selected_command = tuple(command)
        except TypeError as error:
            raise ValueError("command must contain nonempty strings") from error
        if not selected_command or any(
            type(item) is not str or not item for item in selected_command
        ):
            raise ValueError("command must contain nonempty strings")
        if launch_profile not in _GEAR_LAUNCH_PROFILES:
            raise ValueError(
                "launch_profile must be 'zmq_stream' or 'loaded_motion'"
            )
        managed = [
            item
            for item in selected_command
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
        if readiness_poll_s <= 0.0 or signal_poll_s <= 0.0:
            raise ValueError("poll intervals must be positive")
        default_startup = (
            _DEFAULT_STARTUP_MARKERS
            if launch_profile == "zmq_stream"
            else _LOADED_MOTION_STARTUP_MARKERS
        )
        default_active = (
            _DEFAULT_ACTIVE_MARKERS
            if launch_profile == "zmq_stream"
            else _LOADED_MOTION_ACTIVE_MARKERS
        )
        selected_startup_markers = tuple(
            default_startup if startup_markers is None else startup_markers
        )
        selected_active_markers = tuple(
            default_active if active_markers is None else active_markers
        )
        markers = (*selected_startup_markers, *selected_active_markers)
        if any(type(marker) is not str or not marker for marker in markers):
            raise ValueError("readiness markers must be nonempty strings")
        if (
            type(wait_for_control_marker) is not str
            or not wait_for_control_marker
            or "\n" in wait_for_control_marker
        ):
            raise ValueError("wait_for_control_marker must be one nonempty line")
        selected_environment = None if env is None else dict(env)
        selected_cwd = None if cwd is None else str(Path(cwd))
        selected_stop_grace_s = max(0.0, stop_grace_s)
        selected_term_grace_s = max(0.0, term_grace_s)
        selected_kill_grace_s = max(0.0, kill_grace_s)
        output_condition = threading.Condition()

        run_root_path = _canonical_run_root(run_root)
        _require_shell_safe_absolute_path(run_root_path, "run_root")
        target = _validate_confined_output_path(
            run_root_path,
            target_motion_logfile,
            "target motion logfile",
        )
        logs = _validate_confined_output_path(
            run_root_path,
            logs_dir,
            "GEAR logs directory",
        )
        _require_shell_safe_absolute_path(logs, "GEAR logs directory")
        stdout_path = _validate_confined_output_path(
            run_root_path,
            stdout_archive,
            "GEAR stdout archive",
        )
        stderr_path = _validate_confined_output_path(
            run_root_path,
            stderr_archive,
            "GEAR stderr archive",
        )
        if len({target, stdout_path, stderr_path}) != 3:
            raise ProcessError("GEAR file output paths must be distinct")
        if logs in (target, stdout_path, stderr_path):
            raise ProcessError("GEAR logs directory cannot also be a file output")
        for path, label in (
            (target, "target motion logfile"),
            (stdout_path, "GEAR stdout archive"),
            (stderr_path, "GEAR stderr archive"),
        ):
            _create_confined_parents(run_root_path, path, label=label)
        argv = _gear_process_argv(
            selected_command,
            launch_profile=launch_profile,
            target_motion_logfile=target,
            logs_dir=logs,
        )
        logs, owned_logs = _create_owned_directory(
            run_root_path,
            logs,
            "GEAR logs directory",
        )

        self.run_root = run_root_path
        self.argv = argv
        self.launch_profile = launch_profile
        self.target_motion_logfile = target
        self.logs_dir = logs
        self.stdout_archive = stdout_path
        self.stderr_archive = stderr_path
        self._startup_markers = selected_startup_markers
        self._active_markers = selected_active_markers
        self._wait_for_control_marker = wait_for_control_marker
        self._cancelled = cancelled
        self._readiness_timeout_s = readiness_timeout_s
        self._readiness_poll_s = readiness_poll_s
        self._signal_poll_s = signal_poll_s
        self._stop_grace_s = selected_stop_grace_s
        self._term_grace_s = selected_term_grace_s
        self._kill_grace_s = selected_kill_grace_s
        self._env = selected_environment
        self._cwd = selected_cwd
        self._process: subprocess.Popen[bytes] | None = None
        self._pgid: int | None = None
        self._master_fd: int | None = None
        self._stdout_file = None
        self._stderr_file = None
        self._reader_threads: list[threading.Thread] = []
        self._reader_errors: list[tuple[str, BaseException]] = []
        self._observed = b""
        self._output_condition = output_condition
        self._closed = False
        self._ready = False
        self._startup_markers_ready = False
        self._wait_for_control_ready = False
        self._input_prepared = False
        self._control_active = False
        self.signal_history: list[signal.Signals] = []
        self.cleanup_history: list[str] = []
        self._owned_logs_directory = owned_logs

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
    def startup_markers_ready(self) -> bool:
        """Whether authenticated model/interface cold loading has completed."""

        return self._startup_markers_ready

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def wait_for_control_ready(self) -> bool:
        """Whether the authenticated process reached WAIT_FOR_CONTROL."""

        return self._wait_for_control_ready

    @property
    def input_prepared(self) -> bool:
        """Whether file playback or the ZMQ input was prepared in WAIT."""

        return self._input_prepared

    @property
    def control_active(self) -> bool:
        """Whether the exact WAIT_FOR_CONTROL -> CONTROL marker was observed."""

        return self._control_active

    def _reader(self, name: str, stream, archive) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                archive.write(chunk)
                archive.flush()
                if name == "stdout":
                    with self._output_condition:
                        self._observed += chunk
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
        encoded_markers = tuple(marker.encode("utf-8") for marker in markers)
        deadline = (
            None
            if self._readiness_timeout_s is None
            else time.monotonic() + self._readiness_timeout_s
        )
        with self._output_condition:
            while not all(
                marker in self._observed[after_offset:]
                for marker in encoded_markers
            ):
                reader_failure = self._reader_failure_locked()
                if reader_failure is not None:
                    raise reader_failure
                if _cancelled(self._cancelled):
                    raise OperatorCancelled("operator cancelled GEAR readiness wait")
                if self._process is not None and self._process.poll() is not None:
                    missing = [
                        marker
                        for marker, encoded in zip(markers, encoded_markers)
                        if encoded not in self._observed[after_offset:]
                    ]
                    raise ChildProcessDied(
                        f"GEAR child {self.pid} exited {self._process.returncode} "
                        f"before readiness markers {missing!r}"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    missing = [
                        marker
                        for marker, encoded in zip(markers, encoded_markers)
                        if encoded not in self._observed[after_offset:]
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

    def _wait_exact_line_range(
        self, line: str, *, after_offset: int
    ) -> tuple[int, int]:
        if not line.endswith("\n") or "\n" in line[:-1]:
            raise ProcessError("GEAR exact-line marker is invalid")
        encoded_line = line.encode("utf-8")
        deadline = (
            None
            if self._readiness_timeout_s is None
            else time.monotonic() + self._readiness_timeout_s
        )
        with self._output_condition:
            while encoded_line not in self._observed[after_offset:].splitlines(
                keepends=True
            ):
                reader_failure = self._reader_failure_locked()
                if reader_failure is not None:
                    raise reader_failure
                if _cancelled(self._cancelled):
                    raise OperatorCancelled("operator cancelled GEAR readiness wait")
                if self._process is not None and self._process.poll() is not None:
                    raise ChildProcessDied(
                        f"GEAR child {self.pid} exited {self._process.returncode} "
                        f"before exact output line {line.rstrip()!r}"
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    raise ProcessError(
                        f"timed out waiting for exact GEAR output line "
                        f"{line.rstrip()!r}"
                    )
                wait = self._readiness_poll_s
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - time.monotonic()))
                self._output_condition.wait(wait)
            reader_failure = self._reader_failure_locked()
            if reader_failure is not None:
                raise reader_failure

            cursor = after_offset
            for observed_line in self._observed[after_offset:].splitlines(
                keepends=True
            ):
                start = cursor
                cursor += len(observed_line)
                if observed_line == encoded_line:
                    return start, cursor
        raise AssertionError("exact GEAR line vanished while holding output lock")

    def _wait_exact_line(self, line: str, *, after_offset: int) -> int:
        return self._wait_exact_line_range(line, after_offset=after_offset)[1]

    def start_to_wait_for_control(self) -> None:
        """Cold-start GEAR through exact authenticated WAIT_FOR_CONTROL only."""

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
            self._startup_markers_ready = True
            self._wait_exact_line(f"{self._wait_for_control_marker}\n", after_offset=0)
            self._wait_for_control_ready = True
            if len(self._active_markers) != 2:
                raise ProcessError(
                    "GEAR active_markers must contain CONTROL then input activation"
                )
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

    def start(self) -> None:
        """Backward-compatible full startup; phased callers use explicit APIs."""

        self.start_to_wait_for_control()
        # Preserve the long-standing convenience method's authenticated key
        # order.  Stage A intentionally does not use this path: it prepares
        # input in WAIT, resets physics, then activates CONTROL explicitly.
        self._write_key_then_wait(b"]", self._active_markers[0])
        self._control_active = True
        self._ready = True
        profile_key = b"\n" if self.launch_profile == "zmq_stream" else b"t"
        self._write_key_then_wait(profile_key, self._active_markers[1])
        self._input_prepared = True

    def _require_wait_preparation_state(self, profile: str) -> None:
        if self.launch_profile != profile:
            raise ProcessError(f"input preparation requires {profile}")
        if not self._wait_for_control_ready or self._control_active:
            raise ProcessError("input preparation requires authenticated WAIT_FOR_CONTROL")
        if self._input_prepared:
            raise ProcessError("GEAR input is already prepared")
        if self.group_is_stopped():
            raise ProcessError("input preparation requires a running WAIT process")
        self.require_alive()

    def enable_stream_for_preload(self) -> Mapping[str, object]:
        """Enable and clear the ZMQ input while policy control remains inactive."""

        self._require_wait_preparation_state("zmq_stream")
        with self._output_condition:
            activation_boundary = len(self._observed)
            self.write_keys(b"\n")
        enabled_end = self._wait_exact_line(
            f"{self._active_markers[1]}\n",
            after_offset=activation_boundary,
        )
        with self._output_condition:
            self.write_keys(b"qe")
        left_end = self._wait_exact_line(
            f"{_POST_ENABLE_LEFT_LINE}\n",
            after_offset=enabled_end,
        )
        fence_end = self._wait_exact_line(
            f"{_POST_ENABLE_RIGHT_LINE}\n",
            after_offset=left_end,
        )
        self._input_prepared = True
        return MappingProxyType(
            {
                "boundary": enabled_end,
                "end_offset": fence_end,
                "key_sequence": "qe",
                "left_line": _POST_ENABLE_LEFT_LINE,
                "right_line": _POST_ENABLE_RIGHT_LINE,
                "semantics": _POST_ENABLE_FENCE_SEMANTICS,
            }
        )

    def prepare_loaded_motion_for_scoring(self) -> None:
        """Reset frame zero and arm loaded playback while still in WAIT."""

        self._require_wait_preparation_state("loaded_motion")
        self._write_key_then_wait(b"r", _LOADED_MOTION_RESET_MARKER)
        self._write_key_then_wait(b"t", self._active_markers[1])
        self._input_prepared = True

    def activate_control(self) -> None:
        """Enter CONTROL once and continue running after the exact marker."""

        if (
            not self._wait_for_control_ready
            or not self._input_prepared
            or self._control_active
        ):
            raise ProcessError(
                "CONTROL activation requires one prepared WAIT_FOR_CONTROL epoch"
            )
        if self.group_is_stopped():
            raise ProcessError("CONTROL activation requires the GEAR group resumed")
        with self._output_condition:
            boundary = len(self._observed)
            self.write_keys(b"]")
        marker = self._active_markers[0]
        self._wait_exact_line(f"{marker}\n", after_offset=boundary)
        self._control_active = True
        self._ready = True

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

    def reset_loaded_motion_for_scoring(self) -> None:
        """Compatibility spelling for WAIT-phase loaded-motion preparation."""

        self.prepare_loaded_motion_for_scoring()

    def activate_loaded_motion_for_scoring(self) -> None:
        """Resume a prepared file epoch and enter CONTROL without re-stopping."""

        if self.launch_profile != "loaded_motion" or not self._input_prepared:
            raise ProcessError("loaded_motion scoring activation requires preparation")
        if not self.group_is_stopped():
            raise ProcessError("loaded_motion scoring activation requires a stopped group")
        self.continue_group()
        self.activate_control()

    def require_alive(self) -> None:
        reader_failure = self._reader_failure()
        if reader_failure is not None:
            raise reader_failure
        if self._process is None:
            raise ChildProcessDied("GEAR process has not started")
        returncode = self._process.poll()
        if returncode is not None:
            raise ChildProcessDied(f"GEAR child {self.pid} exited {returncode}")

    def publication_boundary(self) -> int:
        """Capture a causal stdout boundary before a stream publication."""

        if (
            self.launch_profile != "zmq_stream"
            or not self._wait_for_control_ready
            or not self._input_prepared
        ):
            raise ProcessError(
                "publication boundaries require a prepared zmq_stream process"
            )
        self.require_alive()
        with self._output_condition:
            reader_failure = self._reader_failure_locked()
            if reader_failure is not None:
                raise reader_failure
            return len(self._observed)

    def wait_for_stream_processing(
        self,
        after_offset: int,
        *,
        frame_count: int,
        global_start: int,
        merged_count: int,
    ) -> Mapping[str, object]:
        """Authenticate one WAIT-phase publication's pinned verbose transcript."""

        if (
            self.launch_profile != "zmq_stream"
            or not self._wait_for_control_ready
            or not self._input_prepared
            or self._control_active
            or type(after_offset) is not int
            or after_offset < 0
            or type(frame_count) is not int
            or frame_count <= 0
            or type(global_start) is not int
            or global_start < 0
            or type(merged_count) is not int
            or merged_count != global_start + frame_count
        ):
            raise ProcessError(
                "stream processing wait requires one prepared WAIT publication"
            )
        with self._output_condition:
            if after_offset > len(self._observed):
                raise ProcessError("stream publication boundary is in the future")
        processing = (
            f"{_STREAM_MERGER_PROCESSING_PREFIX}{frame_count} frames, "
            f"incoming_frame_start={global_start}, frame_step=1\n"
        )
        copied = merged_count - frame_count
        merged = (
            f"{_STREAM_MERGER_MERGED_PREFIX}{merged_count} frames "
            f"(copied: {copied} + incoming: {frame_count})\n"
        )
        start_line = _STREAM_PROCESSING_START_MARKER
        start_range = self._wait_exact_line_range(
            f"{start_line}\n", after_offset=after_offset
        )
        processing_range = self._wait_exact_line_range(
            processing, after_offset=start_range[1]
        )
        merged_range = self._wait_exact_line_range(
            merged, after_offset=processing_range[1]
        )
        end_line = _STREAM_PROCESSING_END_MARKER
        end_range = self._wait_exact_line_range(
            f"{end_line}\n", after_offset=merged_range[1]
        )
        return MappingProxyType(
            {
                "boundary": after_offset,
                "end_offset": end_range[1],
                "frame_count": frame_count,
                "global_start": global_start,
                "merged_count": merged_count,
                "start_line": start_line,
                "processing_line": processing.rstrip("\n"),
                "merged_line": merged.rstrip("\n"),
                "end_line": end_line,
                "start_range": start_range,
                "processing_range": processing_range,
                "merged_range": merged_range,
                "end_range": end_range,
            }
        )

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
        # A /proc-wide descendant inventory is intentionally deferred until
        # after SIGSTOP while the verified leader is live.  On busy hosts that
        # scan can span several 50 Hz control ticks.  The creation-time PGID
        # proof and live leader identity are sufficient to signal first; the
        # full inventory below still proves every surviving member stopped.
        if self._process.poll() is not None and not _linux_group_states(self._pgid):
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

    def terminate_stopped_at_scoring_boundary(self) -> None:
        """Kill a stopped scored group without a SIGCONT that could log row N+1."""

        if self._closed:
            return
        self.require_alive()
        if not self.group_is_stopped():
            raise ProcessError(
                "scoring-boundary termination requires a stopped GEAR group"
            )
        self.cleanup_history.append("scoring-boundary-SIGKILL")
        self._send_group_signal(signal.SIGKILL)
        if not self._wait_group_exit(self._kill_grace_s):
            raise ProcessError(
                f"GEAR process group {self.pgid} survived scoring-boundary SIGKILL"
            )
        # With the group already dead, the generic finalizer only drains and
        # closes evidence descriptors; it cannot send the official key or
        # resume the group.
        self.close()

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
            self._startup_markers_ready = False
            self._wait_for_control_ready = False
            self._input_prepared = False
            self._control_active = False
            # The wrapper creates this directory exclusively.  GEAR fills it
            # during a normal run; an earlier peer failure can leave it empty,
            # and empty unregistered directories cannot enter sealed evidence.
            # Retained no-follow identity prevents a child-controlled path swap
            # from redirecting cleanup outside the run bundle.
            owned_logs = self._owned_logs_directory
            self._owned_logs_directory = None
            if owned_logs is not None:
                _remove_owned_empty_directory(owned_logs)
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
        try:
            self._require_bound_sim_dt()
        except BaseException as error:
            raise _at_failure_site(error, "simulator_advance")
        try:
            self.gear.require_alive()
        except BaseException as error:
            raise _at_failure_site(error, "process_resume")
        try:
            self.simulator.require_alive()
        except BaseException as error:
            raise _at_failure_site(error, "simulator_advance")
        self._paused = False
        result: AdvanceResult | None = None
        operation_error: BaseException | None = None
        try:
            try:
                self.gear.continue_group()
            except BaseException as error:
                operation_error = _at_failure_site(error, "process_resume")
            if operation_error is None:
                try:
                    result = self.simulator.advance(steps)
                except BaseException as error:
                    operation_error = _at_failure_site(
                        error, "simulator_advance"
                    )
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
        try:
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
        except BaseException as error:
            raise _at_failure_site(error, "simulator_advance")
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
