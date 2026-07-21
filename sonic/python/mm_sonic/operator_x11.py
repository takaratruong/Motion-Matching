"""Continuous focus-gated X11 key levels feeding the pure Holden mapper.

This module reads real keyboard levels through ``ctypes``/libX11, gates them by
MuJoCo/G1 window focus, latches Space/X edges, and advances the pure
``HoldenControlMapper`` on a background thread. It adds no third-party
dependencies and never imports MuJoCo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import ctypes
import os
import threading
import time

from .commands import CommandSample
from .holden_control import (
    HoldenControlMapper,
    MappedControlState,
    NormalizedControlState,
)
from .joints import ContractError


# Exact X11 keysyms for every control the desktop frontend understands.
KEYSYMS = {
    "W": 0x0077,
    "A": 0x0061,
    "S": 0x0073,
    "D": 0x0064,
    "Q": 0x0071,
    "E": 0x0065,
    "X": 0x0078,
    "SPACE": 0x0020,
    "LEFT": 0xFF51,
    "UP": 0xFF52,
    "RIGHT": 0xFF53,
    "DOWN": 0xFF54,
    "LEFT_SHIFT": 0xFFE1,
    "LEFT_CTRL": 0xFFE3,
}

# Human-readable transition label for each key.
_ACTIONS = {
    "W": "forward",
    "S": "backward",
    "A": "left",
    "D": "right",
    "Q": "zoom_in",
    "E": "zoom_out",
    "LEFT": "camera_left",
    "RIGHT": "camera_right",
    "UP": "camera_up",
    "DOWN": "camera_down",
    "SPACE": "stand",
    "X": "terminate",
    "LEFT_SHIFT": "walk",
    "LEFT_CTRL": "strafe",
}

_FOCUS_TITLE_MARKERS = (
    "MuJoCo",
    "G1 CONTROLS",
    "G1 terrain motion matching",
)
_STALENESS_S = 0.1
_X11_BAD_WINDOW = 3

# Sonic command keys passively grabbed away from the focused MuJoCo viewer, in
# their canonical order. Standalone Shift/Control are deliberately excluded so
# AnyModifier grabs preserve the walk and strafe modifier combinations.
_X11_GRABBED_CONTROL_KEYS = (
    "W",
    "A",
    "S",
    "D",
    "Q",
    "E",
    "X",
    "SPACE",
    "LEFT",
    "UP",
    "RIGHT",
    "DOWN",
)
# ``AnyModifier`` matches every modifier combination for a passive key grab.
_X11_ANY_MODIFIER = 1 << 15
# ``GrabModeAsync`` keeps event delivery flowing to the keymap while grabbed.
_X11_GRAB_MODE_ASYNC = 1
# This connection polls levels and never consumes X events, so every sync must
# discard its private queue to bound autorepeat/key-event memory.
_X11_DISCARD_EVENTS = 1


class _XErrorEvent(ctypes.Structure):
    """The stable leading layout of Xlib's ``XErrorEvent``."""

    _fields_ = (
        ("type", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("resourceid", ctypes.c_ulong),
        ("serial", ctypes.c_ulong),
        ("error_code", ctypes.c_ubyte),
        ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    )


_X_ERROR_HANDLER_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_XErrorEvent)
)
_X11_ERROR_LOCK = threading.Lock()
_X11_ERRORS: dict[int, list[int]] = {}
_X11_ERROR_HANDLER_INSTALLED = False


@_X_ERROR_HANDLER_TYPE
def _record_x11_error(
    display: ctypes.c_void_p, event: ctypes.POINTER(_XErrorEvent)
) -> int:
    """Capture asynchronous X errors so Xlib cannot terminate the process."""

    display_key = int(display or 0)
    error_code = int(event.contents.error_code) if event else 0
    with _X11_ERROR_LOCK:
        _X11_ERRORS.setdefault(display_key, []).append(error_code)
    return 0


def _take_x11_errors(display: ctypes.c_void_p) -> tuple[int, ...]:
    display_key = int(display.value or 0)
    with _X11_ERROR_LOCK:
        return tuple(_X11_ERRORS.pop(display_key, ()))


def _classify_x11_errors(error_codes: tuple[int, ...]) -> bool:
    """Return true for a transient destroyed-window race, else fail closed."""

    if not error_codes:
        return False
    unexpected = tuple(code for code in error_codes if code != _X11_BAD_WINDOW)
    if unexpected:
        joined = ", ".join(str(code) for code in unexpected)
        raise ContractError(f"X11 provider received protocol error code(s): {joined}")
    return True


def _install_x11_error_handler(lib: ctypes.CDLL) -> None:
    """Install one process-wide nonterminating Xlib protocol-error handler."""

    global _X11_ERROR_HANDLER_INSTALLED
    with _X11_ERROR_LOCK:
        if _X11_ERROR_HANDLER_INSTALLED:
            return
        lib.XSetErrorHandler(_record_x11_error)
        _X11_ERROR_HANDLER_INSTALLED = True


def _is_target_window_title(title: str) -> bool:
    """Return whether an X11 title belongs to an operator-control window."""

    return any(marker in title for marker in _FOCUS_TITLE_MARKERS)


@dataclass(frozen=True)
class KeyLevels:
    """One instantaneous focus flag plus the set of held control keys."""

    focused: bool = False
    pressed: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if type(self.focused) is not bool:
            raise ContractError("KeyLevels focused must be boolean")
        if type(self.pressed) is not frozenset:
            raise ContractError("KeyLevels pressed must be a frozenset")
        for key in self.pressed:
            if key not in KEYSYMS:
                raise ContractError(f"KeyLevels pressed contains unknown key {key!r}")


def normalized_state_from_pressed(pressed: frozenset) -> NormalizedControlState:
    """Translate a held-key set into the pure device-independent snapshot."""

    if type(pressed) is not frozenset:
        raise ContractError("pressed keys must be a frozenset")
    for key in pressed:
        if key not in KEYSYMS:
            raise ContractError(f"pressed contains unknown key {key!r}")
    return NormalizedControlState(
        left_x=float("D" in pressed) - float("A" in pressed),
        left_z=float("S" in pressed) - float("W" in pressed),
        right_x=float("RIGHT" in pressed) - float("LEFT" in pressed),
        right_z=float("DOWN" in pressed) - float("UP" in pressed),
        strafe="LEFT_CTRL" in pressed,
        walk="LEFT_SHIFT" in pressed,
        zoom=float("Q" in pressed) - float("E" in pressed),
        stand="SPACE" in pressed,
        terminate="X" in pressed,
    )


@dataclass(frozen=True)
class IntentSnapshot:
    """One immutable revisioned locomotion intent latched at a boundary."""

    revision: int
    observed_ns: int
    command: CommandSample | None

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision <= 0:
            raise ContractError("intent revision must be a positive integer")
        if type(self.observed_ns) is not int or self.observed_ns < 0:
            raise ContractError("intent observed_ns must be a nonnegative integer")
        if self.command is not None and not isinstance(self.command, CommandSample):
            raise ContractError("intent command must be a CommandSample or None")


class BoundaryControlMailbox:
    """Atomic latest-state boundary that latches Space/X until consumed.

    It also tracks an effective locomotion ``revision`` that increments only
    when velocity, heading, stand, or terminate changes; camera-only updates
    retain the revision.  A same chunk index may be re-sampled only after a
    newer locomotion revision has arrived (a supersession retry); otherwise the
    existing repeated-index rejection is preserved.
    """

    def __init__(self, monotonic_ns: Callable[[], int] = time.monotonic_ns) -> None:
        if not callable(monotonic_ns):
            raise ContractError("mailbox monotonic_ns must be callable")
        self._lock = threading.Lock()
        self._monotonic_ns = monotonic_ns
        self._latest: MappedControlState | None = None
        self._stand_latched = False
        self._terminate_latched = False
        self._last_index = -1
        self._revision = 1
        self._last_sampled_revision = 0
        self._effective_key: tuple[object, ...] | None = None

    @property
    def current_revision(self) -> int:
        with self._lock:
            return self._revision

    @staticmethod
    def _locomotion_key(mapped: MappedControlState) -> tuple[object, ...]:
        return (
            mapped.velocity_mujoco,
            mapped.desired_heading_mujoco_wxyz,
            mapped.stand,
            mapped.terminate,
        )

    def _monotonic(self) -> int:
        value = self._monotonic_ns()
        if type(value) is not int or value < 0:
            raise ContractError("mailbox clock must return nonnegative integer ns")
        return value

    def publish(self, mapped: MappedControlState) -> None:
        if type(mapped) is not MappedControlState:
            raise ContractError("mailbox publish requires a MappedControlState")
        with self._lock:
            key = self._locomotion_key(mapped)
            if self._effective_key is not None and key != self._effective_key:
                self._revision += 1
            self._effective_key = key
            self._latest = mapped
            if mapped.stand:
                self._stand_latched = True
            if mapped.terminate:
                self._terminate_latched = True

    def _sample(
        self, chunk_index: int
    ) -> tuple[IntentSnapshot, MappedControlState]:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("mailbox chunk_index must be a nonnegative integer")
        observed_ns = self._monotonic()
        with self._lock:
            if chunk_index < self._last_index or (
                chunk_index == self._last_index
                and self._revision <= self._last_sampled_revision
            ):
                raise ContractError("mailbox chunk_index must be strictly increasing")
            if self._latest is None:
                raise ContractError("mailbox has no published control state")
            latest = self._latest
            revision = self._revision
            stand = latest.stand or self._stand_latched
            terminate = latest.terminate or self._terminate_latched
            self._stand_latched = False
            self._terminate_latched = False
            self._last_index = chunk_index
            self._last_sampled_revision = revision
        if terminate:
            command: CommandSample | None = None
        else:
            command = CommandSample(
                chunk_index=chunk_index,
                requested_velocity_mujoco=(
                    (0.0, 0.0, 0.0) if stand else latest.velocity_mujoco
                ),
                desired_heading_mujoco_wxyz=latest.desired_heading_mujoco_wxyz,
            )
        snapshot = IntentSnapshot(
            revision=revision, observed_ns=observed_ns, command=command
        )
        return snapshot, latest

    def sample_intent(
        self, chunk_index: int
    ) -> tuple[IntentSnapshot, MappedControlState]:
        return self._sample(chunk_index)

    def sample(
        self, chunk_index: int
    ) -> tuple[CommandSample | None, MappedControlState]:
        snapshot, latest = self._sample(chunk_index)
        return snapshot.command, latest


class ContinuousControlLoop:
    """Sample a key-level provider on one thread and advance the pure mapper."""

    def __init__(
        self,
        provider: object,
        mapper: HoldenControlMapper,
        *,
        event_sink: Callable[[str], None] | None = None,
        period_s: float = 0.02,
        cancel_event: threading.Event | None = None,
        join_timeout_s: float = 2.0,
    ) -> None:
        if not hasattr(provider, "sample"):
            raise ContractError("control loop provider must expose sample()")
        if type(mapper) is not HoldenControlMapper:
            raise ContractError("control loop requires a HoldenControlMapper")
        if (
            type(period_s) not in (int, float)
            or period_s <= 0.0
            or not (period_s < float("inf"))
        ):
            raise ContractError("control loop period_s must be positive and finite")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise ContractError("control loop cancel_event must be a threading.Event")
        self._provider = provider
        self._mapper = mapper
        self._event_sink = event_sink
        self._period_s = float(period_s)
        self._cancel_event = cancel_event
        self._join_timeout_s = float(join_timeout_s)
        self.mailbox = BoundaryControlMailbox()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._cond = threading.Condition()
        self._sequence = 0
        self._error: ContractError | None = None
        self._thread: threading.Thread | None = None
        self._prev_focused = False
        self._prev_pressed: frozenset = frozenset()

    def __enter__(self) -> "ContinuousControlLoop":
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_s)
            if self._thread.is_alive():
                raise ContractError("control loop thread did not stop")
        if exc_type is not None:
            return False
        if self._error is not None:
            raise self._error
        return False

    def wait_for_sequence(self, count: int, timeout_s: float) -> bool:
        if type(count) is not int or count < 0:
            raise ContractError("wait_for_sequence count must be nonnegative")
        deadline = time.monotonic() + float(timeout_s)
        with self._cond:
            while self._sequence < count and not self._finished.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._cond.wait(remaining)
            return self._sequence >= count

    def raise_if_failed(self) -> None:
        """Surface a terminal sampler failure without waiting for context exit."""

        with self._cond:
            error = self._error
            finished = self._finished.is_set()
        if error is not None:
            raise error
        if finished:
            raise ContractError("control loop stopped before viewer acquisition")

    def _emit(self, message: str) -> None:
        if self._event_sink is not None:
            self._event_sink(message)

    def _process_transitions(self, focused: bool, pressed: frozenset) -> None:
        if self._prev_focused and not focused:
            self._emit("FOCUS LOST -> neutral")
        for key in sorted(pressed - self._prev_pressed):
            self._emit(f"KEY {key} DOWN -> {_ACTIONS[key]}")
            if key == "X" and self._cancel_event is not None:
                self._cancel_event.set()
        for key in sorted(self._prev_pressed - pressed):
            self._emit(f"KEY {key} UP -> {_ACTIONS[key]}")
        self._prev_focused = focused
        self._prev_pressed = pressed

    def _publish(self, focused: bool, pressed: frozenset) -> None:
        self._process_transitions(focused, pressed)
        state = normalized_state_from_pressed(pressed)
        mapped = self._mapper.update(state, self._period_s)
        self.mailbox.publish(mapped)
        with self._cond:
            self._sequence += 1
            self._cond.notify_all()

    def _run(self) -> None:
        last_success = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    levels = self._provider.sample()
                except Exception as error:  # provider disconnect is fatal
                    raise ContractError(
                        f"control loop provider failed: {error}"
                    ) from error
                if levels is None:
                    if time.monotonic() - last_success > _STALENESS_S:
                        raise ContractError(
                            "control loop received no fresh sample in time"
                        )
                    self._stop.wait(self._period_s)
                    continue
                if type(levels) is not KeyLevels:
                    raise ContractError("control loop provider must return KeyLevels")
                last_success = time.monotonic()
                pressed = levels.pressed if levels.focused else frozenset()
                self._publish(levels.focused, pressed)
                self._stop.wait(self._period_s)
        except ContractError as error:
            self._error = error
            try:  # publish one neutral focus-lost state before stopping
                self._process_transitions(False, frozenset())
                mapped = self._mapper.update(NormalizedControlState(), self._period_s)
                self.mailbox.publish(mapped)
            except ContractError:
                pass
        finally:
            self._finished.set()
            with self._cond:
                self._cond.notify_all()


class X11KeyStateProvider:
    """Read real key levels through libX11, gated by MuJoCo/G1 window focus."""

    def __init__(self, display_name: str | None = None) -> None:
        name = display_name if display_name is not None else os.environ.get("DISPLAY")
        if not name:
            raise ContractError("X11 provider requires the DISPLAY environment")
        try:
            lib = ctypes.CDLL("libX11.so.6")
        except OSError as error:
            raise ContractError("X11 provider requires libX11") from error
        self._configure(lib)
        _install_x11_error_handler(lib)
        self._lib = lib
        display = lib.XOpenDisplay(name.encode("utf-8"))
        if not display:
            raise ContractError(f"X11 provider cannot open display {name!r}")
        self._display = ctypes.c_void_p(display)
        self._closed = False
        self._grabbed_window = 0
        self._keycodes: dict[str, int] = {}
        for key, keysym in KEYSYMS.items():
            keycode = int(lib.XKeysymToKeycode(self._display, ctypes.c_ulong(keysym)))
            if keycode != 0:
                self._keycodes[key] = keycode

    @staticmethod
    def _configure(lib: ctypes.CDLL) -> None:
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        lib.XCloseDisplay.restype = ctypes.c_int
        lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char * 32]
        lib.XQueryKeymap.restype = ctypes.c_int
        lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        lib.XKeysymToKeycode.restype = ctypes.c_ubyte
        lib.XGetInputFocus.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.XGetInputFocus.restype = ctypes.c_int
        lib.XFetchName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.XFetchName.restype = ctypes.c_int
        lib.XQueryTree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        lib.XQueryTree.restype = ctypes.c_int
        lib.XFree.argtypes = [ctypes.c_void_p]
        lib.XFree.restype = ctypes.c_int
        lib.XSetErrorHandler.argtypes = [_X_ERROR_HANDLER_TYPE]
        lib.XSetErrorHandler.restype = ctypes.c_void_p
        lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.XSync.restype = ctypes.c_int
        lib.XGrabKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.XGrabKey.restype = ctypes.c_int
        lib.XUngrabKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_ulong,
        ]
        lib.XUngrabKey.restype = ctypes.c_int

    @property
    def target_bound(self) -> bool:
        """Whether command-key grabs are installed on a live target window."""

        return not self._closed and self._grabbed_window != 0

    def _fetch_name(self, window: int) -> str | None:
        name_ptr = ctypes.c_char_p()
        status = int(
            self._lib.XFetchName(
                self._display, ctypes.c_ulong(window), ctypes.byref(name_ptr)
            )
        )
        if status == 0 or not name_ptr.value:
            return None
        try:
            return name_ptr.value.decode("utf-8", "replace")
        finally:
            self._lib.XFree(ctypes.cast(name_ptr, ctypes.c_void_p))

    def _parent(self, window: int) -> int:
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        status = int(
            self._lib.XQueryTree(
                self._display,
                ctypes.c_ulong(window),
                ctypes.byref(root),
                ctypes.byref(parent),
                ctypes.byref(children),
                ctypes.byref(count),
            )
        )
        if children:
            self._lib.XFree(ctypes.cast(children, ctypes.c_void_p))
        if status == 0:
            return 0
        return int(parent.value)

    def _focus_window(self) -> int:
        window = ctypes.c_ulong()
        revert = ctypes.c_int()
        self._lib.XGetInputFocus(
            self._display, ctypes.byref(window), ctypes.byref(revert)
        )
        return int(window.value)

    def _target_window(self, window: int) -> int:
        """Return the titled operator ancestor for ``window``, or zero."""

        current = window
        for _ in range(64):
            if current == 0:
                return 0
            title = self._fetch_name(current)
            if title is not None and _is_target_window_title(title):
                return current
            parent = self._parent(current)
            if parent == 0 or parent == current:
                return 0
            current = parent
        return 0

    def _focused_on_target(self, window: int) -> bool:
        return self._target_window(window) != 0

    def _ungrab_window(self, window: int) -> None:
        """Release every command-key grab held on ``window``."""

        if window == 0:
            return
        for key in _X11_GRABBED_CONTROL_KEYS:
            keycode = self._keycodes.get(key)
            if keycode is None:
                continue
            self._lib.XUngrabKey(
                self._display,
                keycode,
                _X11_ANY_MODIFIER,
                window,
            )

    def _release_grabbed_window(self) -> bool:
        """Release the tracked window and report a destroyed-window race."""

        window = self._grabbed_window
        self._grabbed_window = 0
        if window == 0:
            return False
        _take_x11_errors(self._display)
        self._ungrab_window(window)
        self._lib.XSync(self._display, _X11_DISCARD_EVENTS)
        return _classify_x11_errors(_take_x11_errors(self._display))

    def _discard_partial_grabs(self, window: int) -> None:
        """Best-effort cleanup after a failed multi-key grab transaction."""

        _take_x11_errors(self._display)
        self._ungrab_window(window)
        self._lib.XSync(self._display, _X11_DISCARD_EVENTS)
        _take_x11_errors(self._display)

    def _bind_target_window(self, window: int) -> bool:
        """Passively grab all Sonic command keys on the focused target window.

        Idempotent for the currently bound window. When replacing a different
        window it releases the prior grabs first. A destroyed-window
        ``BadWindow`` race clears the binding so a later focus can reacquire it;
        any other X11 error fails closed with a ``ContractError``.
        """

        if window == 0:
            return False
        if window == self._grabbed_window:
            return True
        if self._grabbed_window != 0:
            self._release_grabbed_window()
        _take_x11_errors(self._display)
        for key in _X11_GRABBED_CONTROL_KEYS:
            keycode = self._keycodes.get(key)
            if keycode is None:
                continue
            self._lib.XGrabKey(
                self._display,
                keycode,
                _X11_ANY_MODIFIER,
                window,
                0,
                _X11_GRAB_MODE_ASYNC,
                _X11_GRAB_MODE_ASYNC,
            )
        self._lib.XSync(self._display, _X11_DISCARD_EVENTS)
        errors = _take_x11_errors(self._display)
        try:
            destroyed = _classify_x11_errors(errors)
        except ContractError:
            self._discard_partial_grabs(window)
            raise
        if destroyed:
            # Destroyed-window race: drop the binding and allow reacquisition.
            self._grabbed_window = 0
            return False
        self._grabbed_window = window
        return True

    def sample(self) -> KeyLevels:
        if self._closed:
            raise ContractError("X11 provider is closed")
        _take_x11_errors(self._display)
        focus_window = self._focus_window()
        target_window = self._target_window(focus_window)
        focused = target_window != 0
        pressed: frozenset = frozenset()
        if focused:
            if not self._bind_target_window(target_window):
                return KeyLevels(focused=False, pressed=frozenset())
            keymap = (ctypes.c_char * 32)()
            self._lib.XQueryKeymap(self._display, keymap)
            raw = bytes(keymap)
            pressed = frozenset(
                key
                for key, keycode in self._keycodes.items()
                if raw[keycode >> 3] & (1 << (keycode & 7))
            )
        self._lib.XSync(self._display, _X11_DISCARD_EVENTS)
        if _classify_x11_errors(_take_x11_errors(self._display)):
            self._release_grabbed_window()
            return KeyLevels(focused=False, pressed=frozenset())
        return KeyLevels(focused=focused, pressed=pressed)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        release_error: ContractError | None = None
        try:
            self._release_grabbed_window()
        except ContractError as error:
            release_error = error
        finally:
            self._lib.XCloseDisplay(self._display)
        if release_error is not None:
            raise release_error
