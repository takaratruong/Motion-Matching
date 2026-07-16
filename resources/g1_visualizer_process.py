#!/usr/bin/env python3
"""Capture and revalidate Linux process identities before signaling them."""

import argparse
import json
import os
from pathlib import Path
import re
import signal as signal_module
import sys
import time


MAX_CAPTURE_TIMEOUT_MS = 60_000
DEFAULT_POLL_INTERVAL_MS = 10

_IDENTITY_KEYS = frozenset({
    "pid",
    "start_time",
    "executable",
    "executable_device",
    "executable_inode",
    "cmdline_hex",
})
_LOWER_HEX = re.compile(r"[0-9a-f]*\Z")
_SIGNALS = {
    "STOP": signal_module.SIGSTOP,
    "CONT": signal_module.SIGCONT,
    "TERM": signal_module.SIGTERM,
    "KILL": signal_module.SIGKILL,
}


class ProcessIdentityError(RuntimeError):
    """The requested process cannot be identified without ambiguity."""


class ProcessCaptureTimeout(TimeoutError):
    """The pinned process did not become the expected executable in time."""


class _ProcessAbsent(Exception):
    pass


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _require_integer(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"identity {name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"identity {name} must be {relation}")
    return value


def parse_proc_stat_start_time(raw_stat, expected_pid=None):
    """Return Linux ``/proc/PID/stat`` field 22 from raw file contents."""

    if isinstance(raw_stat, bytes):
        try:
            text = raw_stat.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("process stat is not ASCII") from error
    elif isinstance(raw_stat, str):
        text = raw_stat
    else:
        raise ValueError("process stat must be bytes or text")

    opening = text.find("(")
    closing = text.rfind(")")
    if opening <= 0 or closing <= opening:
        raise ValueError("process stat has no complete command field")
    try:
        pid = int(text[:opening].strip(), 10)
    except ValueError as error:
        raise ValueError("process stat has an invalid PID") from error
    if pid <= 0:
        raise ValueError("process stat PID must be positive")
    if expected_pid is not None:
        _require_integer(expected_pid, "expected PID", positive=True)
        if pid != expected_pid:
            raise ValueError(
                f"process stat PID {pid} does not match {expected_pid}")

    # After the command (field 2), item zero is state (field 3), so item
    # nineteen is starttime (field 22). The command itself may contain spaces
    # and ')' bytes; locating its final ')' is required by procfs semantics.
    trailing_fields = text[closing + 1:].split()
    if len(trailing_fields) < 20:
        raise ValueError("process stat ends before field 22")
    try:
        start_time = int(trailing_fields[19], 10)
    except ValueError as error:
        raise ValueError("process stat field 22 is not an integer") from error
    if start_time < 0:
        raise ValueError("process stat field 22 must be nonnegative")
    return start_time


def parse_identity(document):
    """Parse and strictly validate a process identity or JSON ``null``."""

    if isinstance(document, bytes):
        try:
            document = document.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("identity document is not UTF-8") from error
    if isinstance(document, str):
        try:
            document = json.loads(
                document,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError("identity document is not valid JSON") from error

    if document is None:
        return None
    if not isinstance(document, dict):
        raise ValueError("identity document must be an object or null")
    if set(document) != _IDENTITY_KEYS:
        missing = sorted(_IDENTITY_KEYS - set(document))
        extra = sorted(set(document) - _IDENTITY_KEYS)
        raise ValueError(
            f"identity fields differ (missing={missing}, extra={extra})")

    identity = dict(document)
    _require_integer(identity["pid"], "pid", positive=True)
    _require_integer(identity["start_time"], "start_time")
    _require_integer(identity["executable_device"], "executable_device")
    _require_integer(identity["executable_inode"], "executable_inode")

    executable = identity["executable"]
    if not isinstance(executable, str) or not executable:
        raise ValueError("identity executable must be nonempty text")
    if not os.path.isabs(executable):
        raise ValueError("identity executable must be absolute")
    if os.path.normpath(executable) != executable:
        raise ValueError("identity executable must be normalized")

    cmdline_hex = identity["cmdline_hex"]
    if not isinstance(cmdline_hex, str):
        raise ValueError("identity cmdline_hex must be text")
    if len(cmdline_hex) % 2 or _LOWER_HEX.fullmatch(cmdline_hex) is None:
        raise ValueError("identity cmdline_hex must be lowercase raw-byte hex")

    return identity


def canonical_identity_bytes(identity):
    """Serialize one validated identity in its canonical JSON form."""

    parsed = parse_identity(identity)
    return (json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("ascii")


def compare_identities(expected, observed):
    """Byte-compare every canonical field of two validated identities."""

    return canonical_identity_bytes(expected) == canonical_identity_bytes(
        observed)


def write_identity(path, identity):
    path = Path(path)
    with path.open("wb") as stream:
        stream.write(canonical_identity_bytes(identity))


def load_identity(path):
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ProcessIdentityError(
            f"could not read identity document {path}: {error}") from error
    return parse_identity(raw)


def _require_pid(pid):
    return _require_integer(pid, "pid", positive=True)


def _process_directory(proc_root, pid):
    return Path(proc_root) / str(_require_pid(pid))


def _missing_or_broken(proc_root, pid, label, error):
    process_directory = _process_directory(proc_root, pid)
    if not process_directory.is_dir():
        raise _ProcessAbsent(pid) from error
    raise ProcessIdentityError(
        f"process {pid} has missing or unreadable {label}") from error


def _read_start_time(pid, *, proc_root):
    process_directory = _process_directory(proc_root, pid)
    if not process_directory.is_dir():
        raise _ProcessAbsent(pid)
    try:
        raw_stat = (process_directory / "stat").read_bytes()
    except OSError as error:
        _missing_or_broken(proc_root, pid, "stat", error)
    try:
        return parse_proc_stat_start_time(raw_stat, expected_pid=pid)
    except ValueError as error:
        raise ProcessIdentityError(
            f"process {pid} has malformed stat data: {error}") from error


def _read_resolved_executable(pid, *, proc_root):
    process_directory = _process_directory(proc_root, pid)
    if not process_directory.is_dir():
        raise _ProcessAbsent(pid)
    executable_link = process_directory / "exe"
    try:
        return str(executable_link.resolve(strict=True))
    except (OSError, RuntimeError) as error:
        _missing_or_broken(proc_root, pid, "executable link", error)


def _resolve_expected_executable(executable):
    try:
        raw_path = os.fspath(executable)
    except TypeError as error:
        raise ValueError("executable must be a filesystem path") from error
    if not os.path.isabs(raw_path):
        raise ValueError("executable must be an absolute path")
    try:
        return str(Path(raw_path).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ProcessIdentityError(
            f"cannot resolve expected executable {raw_path}") from error


def read_process_identity(pid, *, proc_root=Path("/proc")):
    """Read all identity fields for one currently present process."""

    pid = _require_pid(pid)
    process_directory = _process_directory(proc_root, pid)
    if not process_directory.is_dir():
        raise _ProcessAbsent(pid)

    start_time = _read_start_time(pid, proc_root=proc_root)
    executable = _read_resolved_executable(pid, proc_root=proc_root)
    executable_link = process_directory / "exe"
    try:
        executable_stat = executable_link.stat()
    except OSError as error:
        _missing_or_broken(proc_root, pid, "executable metadata", error)
    try:
        cmdline = (process_directory / "cmdline").read_bytes()
    except OSError as error:
        _missing_or_broken(proc_root, pid, "command line", error)
    final_start_time = _read_start_time(pid, proc_root=proc_root)
    if final_start_time != start_time:
        raise ProcessIdentityError(
            f"process {pid} start time changed while reading identity")

    return parse_identity({
        "pid": pid,
        "start_time": start_time,
        "executable": executable,
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "cmdline_hex": cmdline.hex(),
    })


def discover(executable, *, proc_root=Path("/proc")):
    """Return the sole exact executable match, ``None``, or fail ambiguous."""

    expected_executable = _resolve_expected_executable(executable)
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError as error:
        raise ProcessIdentityError(
            f"cannot enumerate proc root {proc_root}") from error

    pids = sorted({
        int(entry.name)
        for entry in entries
        if entry.name.isascii() and entry.name.isdigit()
        and int(entry.name) > 0
    })
    matches = []
    for pid in pids:
        try:
            resolved_executable = _read_resolved_executable(
                pid, proc_root=proc_root)
        except (_ProcessAbsent, ProcessIdentityError):
            # Processes may disappear while /proc is enumerated, and unrelated
            # processes may be inaccessible to this user.
            continue
        if resolved_executable != expected_executable:
            continue
        try:
            first_identity = read_process_identity(pid, proc_root=proc_root)
            second_identity = read_process_identity(pid, proc_root=proc_root)
        except _ProcessAbsent:
            continue
        if not compare_identities(first_identity, second_identity):
            raise ProcessIdentityError(
                f"process {pid} changed during discovery")
        if second_identity["executable"] != expected_executable:
            continue
        matches.append(second_identity)
        if len(matches) > 1:
            raise ProcessIdentityError(
                f"more than one process resolves to {expected_executable}")
    return matches[0] if matches else None


def _pinned_start_time(pid, proc_root):
    try:
        return _read_start_time(pid, proc_root=proc_root)
    except _ProcessAbsent as error:
        raise ProcessIdentityError(
            f"process {pid} is missing at capture") from error


def _current_pinned_start_time(pid, pinned_start_time, proc_root):
    try:
        current_start_time = _read_start_time(pid, proc_root=proc_root)
    except _ProcessAbsent as error:
        raise ProcessIdentityError(
            f"pinned process {pid} disappeared") from error
    if current_start_time != pinned_start_time:
        raise ProcessIdentityError(
            f"pinned process {pid} start time changed from "
            f"{pinned_start_time} to {current_start_time}")
    return current_start_time


def capture_wait(
        pid,
        executable,
        timeout_ms,
        *,
        proc_root=Path("/proc"),
        monotonic=time.monotonic,
        sleeper=time.sleep,
        poll_interval_ms=DEFAULT_POLL_INTERVAL_MS):
    """Pin a PID/start-time pair and wait for its unchanged expected exec."""

    pid = _require_pid(pid)
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
        raise ValueError("timeout_ms must be an integer")
    if timeout_ms < 0 or timeout_ms > MAX_CAPTURE_TIMEOUT_MS:
        raise ValueError(
            f"timeout_ms must be between 0 and {MAX_CAPTURE_TIMEOUT_MS}")
    if isinstance(poll_interval_ms, bool) or not isinstance(
            poll_interval_ms, (int, float)) or poll_interval_ms <= 0:
        raise ValueError("poll_interval_ms must be positive")

    # Pin before polling for the executable. No later PID with the same number
    # is eligible, even if it resolves to the requested path.
    pinned_start_time = _pinned_start_time(pid, proc_root)
    expected_executable = _resolve_expected_executable(executable)
    deadline = monotonic() + timeout_ms / 1000.0

    while True:
        _current_pinned_start_time(pid, pinned_start_time, proc_root)
        try:
            current_executable = _read_resolved_executable(
                pid, proc_root=proc_root)
        except _ProcessAbsent as error:
            raise ProcessIdentityError(
                f"pinned process {pid} disappeared") from error

        if current_executable == expected_executable:
            try:
                first = read_process_identity(pid, proc_root=proc_root)
                second = read_process_identity(pid, proc_root=proc_root)
            except _ProcessAbsent as error:
                raise ProcessIdentityError(
                    f"pinned process {pid} disappeared") from error
            if not compare_identities(first, second):
                raise ProcessIdentityError(
                    f"pinned process {pid} changed between identity reads")
            if first["start_time"] != pinned_start_time:
                raise ProcessIdentityError(
                    f"pinned process {pid} start time changed")
            if first["executable"] == expected_executable:
                return first

        now = monotonic()
        if now >= deadline:
            raise ProcessCaptureTimeout(
                f"process {pid} did not resolve to {expected_executable} "
                f"within {timeout_ms} ms")
        remaining = deadline - now
        sleeper(min(poll_interval_ms / 1000.0, remaining))


def signal_identity(
        identity,
        signal_name,
        *,
        proc_root=Path("/proc"),
        kill_callback=None):
    """Signal only an identity that still exactly matches all saved fields."""

    if signal_name not in _SIGNALS:
        raise ValueError(
            f"unsupported signal {signal_name!r}; expected "
            f"{'|'.join(_SIGNALS)}")
    expected = parse_identity(identity)
    if expected is None:
        return False
    if kill_callback is None:
        kill_callback = os.kill

    try:
        first_observed = read_process_identity(
            expected["pid"], proc_root=proc_root)
        second_observed = read_process_identity(
            expected["pid"], proc_root=proc_root)
    except _ProcessAbsent:
        return False
    if not compare_identities(first_observed, second_observed):
        raise ProcessIdentityError(
            f"process {expected['pid']} changed during identity validation; "
            "refusing to signal")
    if not compare_identities(expected, second_observed):
        raise ProcessIdentityError(
            f"process {expected['pid']} identity changed; refusing to signal")

    try:
        kill_callback(expected["pid"], _SIGNALS[signal_name])
    except ProcessLookupError:
        return False
    return True


def signal_identity_file(
        identity_path,
        signal_name,
        *,
        proc_root=Path("/proc"),
        kill_callback=None):
    return signal_identity(
        load_identity(identity_path),
        signal_name,
        proc_root=proc_root,
        kill_callback=kill_callback,
    )


def _absolute_path(raw_path):
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _positive_pid(raw_pid):
    try:
        pid = int(raw_pid, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("PID must be an integer") from error
    if pid <= 0:
        raise argparse.ArgumentTypeError("PID must be positive")
    return pid


def build_parser():
    parser = argparse.ArgumentParser(
        description="Capture and safely signal an exact visualizer process")
    commands = parser.add_subparsers(dest="command", required=True)

    discover_parser = commands.add_parser("discover")
    discover_parser.add_argument(
        "--executable", required=True, type=_absolute_path)
    discover_parser.add_argument("--output", required=True, type=_absolute_path)

    capture_parser = commands.add_parser("capture-wait")
    capture_parser.add_argument("--pid", required=True, type=_positive_pid)
    capture_parser.add_argument(
        "--executable", required=True, type=_absolute_path)
    capture_parser.add_argument("--timeout-ms", required=True, type=int)
    capture_parser.add_argument("--output", required=True, type=_absolute_path)

    signal_parser = commands.add_parser("signal")
    signal_parser.add_argument(
        "--identity", required=True, type=_absolute_path)
    signal_parser.add_argument(
        "--signal", required=True, choices=tuple(_SIGNALS))
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "discover":
            identity = discover(
                arguments.executable, proc_root=Path("/proc"))
            write_identity(arguments.output, identity)
        elif arguments.command == "capture-wait":
            identity = capture_wait(
                arguments.pid,
                arguments.executable,
                arguments.timeout_ms,
                proc_root=Path("/proc"),
            )
            write_identity(arguments.output, identity)
        else:
            signal_identity_file(
                arguments.identity,
                arguments.signal,
                proc_root=Path("/proc"),
                kill_callback=os.kill,
            )
    except (OSError, ProcessIdentityError, ProcessCaptureTimeout,
            ValueError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
