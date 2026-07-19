"""Pure thread-safe terminal key decoding for the SONIC OperatorSampler."""

from __future__ import annotations

from collections.abc import Callable
import os
import select
import termios
import threading
import tty

from .joints import ContractError
from .operator import OperatorState


# Case-insensitive key -> OperatorState field name. CR/LF are ignored.
_KEY_TO_FIELD = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "q": "heading_left",
    "e": "heading_right",
    " ": "stand",
    "x": "terminate",
}
_IGNORED = frozenset(("\r", "\n"))


def _key_label(character: str) -> str:
    if character == " ":
        return "<SPACE>"
    if character in _IGNORED:
        return "<ENTER>"
    if character == "\x1b":
        return "<ESC>"
    if character.isprintable():
        return character.upper()
    return ascii(character)[1:-1]


def _accepted_key_event(character: str) -> str:
    action = _KEY_TO_FIELD.get(character.lower(), "ignored")
    return f"KEY {_key_label(character)} -> {action}"


def _ignored_key_event(keys: str) -> str:
    return f"KEY {''.join(_key_label(character) for character in keys)} -> ignored"


def _fields_from_keys(keys: str) -> set[str]:
    """Validate a whole key string and collect the OperatorState fields it sets."""
    if type(keys) is not str:
        raise ContractError("terminal keys must be a string")
    fields: set[str] = set()
    for character in keys:
        if character in _IGNORED:
            continue
        field = _KEY_TO_FIELD.get(character.lower())
        if field is None:
            raise ContractError(f"unknown terminal key {character!r}")
        fields.add(field)
    return fields


def operator_state_from_keys(keys: str) -> OperatorState:
    """Decode a key string into a complete immutable OperatorState.

    Accepts case-insensitive W/S/A/D/Q/E, space, CR, and LF. CR/LF are
    ignored; any other character raises ContractError before any mutation.
    Repetition is idempotent and opposing keys remain simultaneously true.
    """
    return OperatorState(**{field: True for field in _fields_from_keys(keys)})


class TerminalKeyBuffer:
    """Union pending terminal keys until an atomic boundary sample.

    Safe for one input thread feeding while a coordinator thread samples.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: set[str] = set()

    def feed(self, keys: str) -> None:
        # Validate the whole feed before touching pending state so an invalid
        # feed leaves previously accepted keys intact.
        fields = _fields_from_keys(keys)
        with self._lock:
            self._pending |= fields

    def sample(self) -> OperatorState:
        with self._lock:
            fields = self._pending
            self._pending = set()
        return OperatorState(**{field: True for field in fields})


class TerminalInputReader:
    """Capture raw single-byte keys from a real TTY on a daemon thread."""

    def __init__(
        self,
        fd: int,
        buffer: TerminalKeyBuffer,
        *,
        event_sink: Callable[[str], None] | None = None,
        join_timeout: float = 2.0,
    ) -> None:
        if type(fd) is not int or fd < 0:
            raise ContractError("terminal reader fd must be a file descriptor")
        if not isinstance(buffer, TerminalKeyBuffer):
            raise ContractError("terminal reader requires a TerminalKeyBuffer")
        try:
            self._original = termios.tcgetattr(fd)
        except termios.error as error:
            raise ContractError("terminal reader requires a real TTY") from error
        if not os.isatty(fd):
            raise ContractError("terminal reader requires a real TTY")
        self._fd = fd
        self._buffer = buffer
        self._event_sink = event_sink
        self._join_timeout = float(join_timeout)
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: ContractError | None = None
        self._restored = False

    def __enter__(self) -> "TerminalInputReader":
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout)
        restore_error = self._restore()
        if self._thread is not None and self._thread.is_alive():
            raise ContractError("terminal reader thread did not stop")
        if exc_type is not None:
            return False
        if self._error is not None:
            raise self._error
        if restore_error is not None:
            raise restore_error
        return False

    def wait_closed(self, timeout: float | None = None) -> None:
        """Block until the reader thread stops, surfacing reader failures."""
        if not self._closed.wait(timeout):
            raise ContractError("terminal reader did not close in time")
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.05)
                if not ready:
                    continue
                data = os.read(self._fd, 1024)
                if not data:
                    raise ContractError("terminal input stream closed")
                keys = data.decode("utf-8")
                try:
                    self._buffer.feed(keys)
                except ContractError:
                    if self._event_sink is not None:
                        self._event_sink(_ignored_key_event(keys))
                    continue
                if self._event_sink is not None:
                    for character in keys:
                        self._event_sink(_accepted_key_event(character))
        except ContractError as error:
            self._error = error
        except (OSError, UnicodeDecodeError) as error:
            self._error = ContractError(f"terminal reader failed: {error}")
        finally:
            self._closed.set()

    def _restore(self) -> ContractError | None:
        if self._restored:
            return None
        self._restored = True
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)
        except termios.error as error:
            return ContractError(f"terminal attribute restore failed: {error}")
        return None
