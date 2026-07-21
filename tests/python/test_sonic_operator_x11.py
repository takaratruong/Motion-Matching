from __future__ import annotations

import ctypes
import os
import threading
import time
import unittest

from mm_sonic import operator_x11
from mm_sonic.commands import CommandSample
from mm_sonic.holden_control import (
    CameraState,
    HoldenControlMapper,
    MappedControlState,
    NormalizedControlState,
)
from mm_sonic.joints import ContractError
from mm_sonic.operator_x11 import (
    KEYSYMS,
    BoundaryControlMailbox,
    ContinuousControlLoop,
    IntentSnapshot,
    KeyLevels,
    X11KeyStateProvider,
    _X11_ANY_MODIFIER,
    _X11_GRABBED_CONTROL_KEYS,
    _classify_x11_errors,
    _is_target_window_title,
    normalized_state_from_pressed,
)


class _StepClock:
    """Monotonic integer-ns clock returning a fresh value on each read."""

    def __init__(self, start: int = 1_000) -> None:
        self._value = start

    def __call__(self) -> int:
        self._value += 7
        return self._value


class FakeProvider:
    def __init__(self, frames: list[KeyLevels]) -> None:
        self.frames = frames

    def sample(self) -> KeyLevels:
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]


class SteadyProvider:
    """Return one healthy sample forever without raising or exhausting."""

    def __init__(self, levels: KeyLevels) -> None:
        self._levels = levels

    def sample(self) -> KeyLevels:
        return self._levels


class RaisingProvider:
    def sample(self) -> KeyLevels:
        raise RuntimeError("device gone")


class AbsentProvider:
    def sample(self) -> None:
        return None


def _mapper() -> HoldenControlMapper:
    return HoldenControlMapper(initial_heading_yaw_rad=0.0)


class KeysymRegistryTests(unittest.TestCase):
    def test_registry_has_exact_frozen_keysyms(self) -> None:
        self.assertEqual(
            KEYSYMS,
            {
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
            },
        )


class NormalizedStateFromPressedTests(unittest.TestCase):
    def test_maps_pressed_keys_to_normalized_axes(self) -> None:
        state = normalized_state_from_pressed(
            frozenset({"D", "S", "RIGHT", "DOWN", "LEFT_CTRL", "LEFT_SHIFT", "Q", "SPACE", "X"})
        )
        self.assertEqual(state.left_x, 1.0)
        self.assertEqual(state.left_z, 1.0)
        self.assertEqual(state.right_x, 1.0)
        self.assertEqual(state.right_z, 1.0)
        self.assertTrue(state.strafe)
        self.assertTrue(state.walk)
        self.assertEqual(state.zoom, 1.0)
        self.assertTrue(state.stand)
        self.assertTrue(state.terminate)

    def test_opposed_keys_cancel(self) -> None:
        state = normalized_state_from_pressed(frozenset({"W", "S", "A", "D"}))
        self.assertEqual(state.left_x, 0.0)
        self.assertEqual(state.left_z, 0.0)

    def test_empty_pressed_is_neutral(self) -> None:
        self.assertEqual(normalized_state_from_pressed(frozenset()), NormalizedControlState())


class KeyLevelsTests(unittest.TestCase):
    def test_rejects_unknown_key(self) -> None:
        with self.assertRaises(ContractError):
            KeyLevels(focused=True, pressed=frozenset({"Z"}))

    def test_rejects_nonbool_focus(self) -> None:
        with self.assertRaises(ContractError):
            KeyLevels(focused=1, pressed=frozenset())


class ContinuousControlLoopTests(unittest.TestCase):
    def test_hold_release_and_focus_loss_publish_exact_transitions(self) -> None:
        events: list[str] = []
        provider = FakeProvider([
            KeyLevels(focused=True, pressed=frozenset({"W"})),
            KeyLevels(focused=True, pressed=frozenset({"W", "LEFT_CTRL"})),
            KeyLevels(focused=False, pressed=frozenset()),
        ])
        loop = ContinuousControlLoop(
            provider,
            _mapper(),
            event_sink=events.append,
            period_s=0.001,
        )
        with loop:
            self.assertTrue(loop.wait_for_sequence(3, timeout_s=1.0))
        self.assertIn("KEY W DOWN -> forward", events)
        self.assertIn("KEY LEFT_CTRL DOWN -> strafe", events)
        self.assertIn("FOCUS LOST -> neutral", events)

    def test_healthy_unfocused_samples_keep_loop_alive_and_neutral(self) -> None:
        # Regression (decision-001): a successful focused=False sample is fresh
        # input, not a stale/absent sample. Many such samples over >0.1s must
        # keep the loop alive and continuously publish neutral state.
        provider = SteadyProvider(KeyLevels(focused=False, pressed=frozenset()))
        loop = ContinuousControlLoop(provider, _mapper(), period_s=0.001)
        with loop:
            self.assertTrue(loop.wait_for_sequence(150, timeout_s=2.0))
            command, mapped = loop.mailbox.sample(0)
        self.assertIsNotNone(command)
        self.assertEqual(command.requested_velocity_mujoco, (0.0, 0.0, 0.0))
        self.assertEqual(mapped.velocity_mujoco, (0.0, 0.0, 0.0))

    def test_provider_exception_is_fatal(self) -> None:
        loop = ContinuousControlLoop(RaisingProvider(), _mapper(), period_s=0.001)
        with self.assertRaises(ContractError):
            with loop:
                loop.wait_for_sequence(1, timeout_s=1.0)

    def test_absent_samples_beyond_staleness_are_fatal(self) -> None:
        loop = ContinuousControlLoop(AbsentProvider(), _mapper(), period_s=0.001)
        with self.assertRaises(ContractError):
            with loop:
                time.sleep(0.2)

    def test_x_rising_edge_sets_cancel_event(self) -> None:
        event = threading.Event()
        provider = FakeProvider([
            KeyLevels(focused=True, pressed=frozenset()),
            KeyLevels(focused=True, pressed=frozenset({"X"})),
        ])
        loop = ContinuousControlLoop(
            provider,
            _mapper(),
            cancel_event=event,
            period_s=0.001,
        )
        with loop:
            self.assertTrue(loop.wait_for_sequence(2, timeout_s=1.0))
            self.assertTrue(event.wait(timeout=1.0))
        self.assertTrue(event.is_set())


class BoundaryControlMailboxTests(unittest.TestCase):
    @staticmethod
    def _mapped(*, stand: bool, terminate: bool) -> MappedControlState:
        return MappedControlState(
            velocity_mujoco=(0.5, 0.0, 0.0),
            desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
            camera=CameraState(0, 0.0, 0.4, 3.0),
            strafe=False,
            walk_blend=0.0,
            stand=stand,
            terminate=terminate,
        )

    def test_space_and_x_edges_survive_release_until_consumed(self) -> None:
        mailbox = BoundaryControlMailbox()
        mailbox.publish(self._mapped(stand=True, terminate=False))
        mailbox.publish(self._mapped(stand=False, terminate=False))
        first, _ = mailbox.sample(0)
        self.assertEqual(first.requested_velocity_mujoco, (0.0, 0.0, 0.0))
        mailbox.publish(self._mapped(stand=False, terminate=True))
        mailbox.publish(self._mapped(stand=False, terminate=False))
        terminated, _ = mailbox.sample(1)
        self.assertIsNone(terminated)

    def test_sample_requires_monotonic_sequence(self) -> None:
        mailbox = BoundaryControlMailbox()
        mailbox.publish(self._mapped(stand=False, terminate=False))
        mailbox.sample(0)
        with self.assertRaises(ContractError):
            mailbox.sample(0)

    def test_sample_before_publish_is_rejected(self) -> None:
        mailbox = BoundaryControlMailbox()
        with self.assertRaises(ContractError):
            mailbox.sample(0)


class RevisionedMailboxTests(unittest.TestCase):
    @staticmethod
    def _mapped(
        *,
        velocity: tuple[float, float, float] = (0.5, 0.0, 0.0),
        heading: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        camera: CameraState | None = None,
        stand: bool = False,
        terminate: bool = False,
    ) -> MappedControlState:
        return MappedControlState(
            velocity_mujoco=velocity,
            desired_heading_mujoco_wxyz=heading,
            camera=camera if camera is not None else CameraState(0, 0.0, 0.4, 3.0),
            strafe=False,
            walk_blend=0.0,
            stand=stand,
            terminate=terminate,
        )

    def test_effective_locomotion_revision_starts_at_one(self) -> None:
        mailbox = BoundaryControlMailbox(monotonic_ns=_StepClock())
        self.assertEqual(mailbox.current_revision, 1)
        mailbox.publish(self._mapped())
        self.assertEqual(mailbox.current_revision, 1)

    def test_sample_intent_returns_snapshot_with_revision_and_stamp(self) -> None:
        clock = _StepClock()
        mailbox = BoundaryControlMailbox(monotonic_ns=clock)
        mailbox.publish(self._mapped(velocity=(0.5, 0.0, 0.0)))
        snapshot, mapped = mailbox.sample_intent(0)
        self.assertIsInstance(snapshot, IntentSnapshot)
        self.assertIsInstance(mapped, MappedControlState)
        self.assertEqual(snapshot.revision, 1)
        self.assertIsInstance(snapshot.observed_ns, int)
        self.assertGreater(snapshot.observed_ns, 0)
        self.assertEqual(
            snapshot.command.requested_velocity_mujoco, (0.5, 0.0, 0.0)
        )

    def test_locomotion_changes_increment_revision(self) -> None:
        mailbox = BoundaryControlMailbox(monotonic_ns=_StepClock())
        mailbox.publish(self._mapped(velocity=(0.5, 0.0, 0.0)))
        self.assertEqual(mailbox.current_revision, 1)
        mailbox.publish(self._mapped(velocity=(-0.5, 0.0, 0.0)))
        self.assertEqual(mailbox.current_revision, 2)
        mailbox.publish(
            self._mapped(
                velocity=(-0.5, 0.0, 0.0), heading=(0.0, 0.0, 0.0, 1.0)
            )
        )
        self.assertEqual(mailbox.current_revision, 3)
        mailbox.publish(
            self._mapped(velocity=(-0.5, 0.0, 0.0), heading=(0.0, 0.0, 0.0, 1.0), stand=True)
        )
        self.assertEqual(mailbox.current_revision, 4)
        mailbox.publish(
            self._mapped(velocity=(-0.5, 0.0, 0.0), heading=(0.0, 0.0, 0.0, 1.0), stand=True, terminate=True)
        )
        self.assertEqual(mailbox.current_revision, 5)

    def test_camera_only_change_retains_revision(self) -> None:
        mailbox = BoundaryControlMailbox(monotonic_ns=_StepClock())
        mailbox.publish(self._mapped(camera=CameraState(0, 0.0, 0.4, 3.0)))
        self.assertEqual(mailbox.current_revision, 1)
        mailbox.publish(self._mapped(camera=CameraState(1, 0.3, 0.5, 2.0)))
        self.assertEqual(mailbox.current_revision, 1)

    def test_same_prefix_sample_allowed_only_after_supersession(self) -> None:
        mailbox = BoundaryControlMailbox(monotonic_ns=_StepClock())
        mailbox.publish(self._mapped(velocity=(0.5, 0.0, 0.0)))
        first, _ = mailbox.sample_intent(0)
        self.assertEqual(first.revision, 1)
        # No newer locomotion revision -> repeated index is still rejected.
        with self.assertRaises(ContractError):
            mailbox.sample_intent(0)
        # A newer locomotion revision permits retrying the same chunk index.
        mailbox.publish(self._mapped(velocity=(-0.5, 0.0, 0.0)))
        retried, _ = mailbox.sample_intent(0)
        self.assertEqual(retried.revision, 2)
        # Once consumed at the new revision, the same index is rejected again.
        with self.assertRaises(ContractError):
            mailbox.sample_intent(0)

    def test_terminate_snapshot_carries_no_command(self) -> None:
        mailbox = BoundaryControlMailbox(monotonic_ns=_StepClock())
        mailbox.publish(self._mapped(terminate=True))
        snapshot, _ = mailbox.sample_intent(0)
        self.assertIsNone(snapshot.command)

    def test_sample_wrapper_preserves_tuple_shape(self) -> None:
        mailbox = BoundaryControlMailbox(monotonic_ns=_StepClock())
        mailbox.publish(self._mapped(velocity=(0.5, 0.0, 0.0)))
        command, mapped = mailbox.sample(0)
        self.assertEqual(command.requested_velocity_mujoco, (0.5, 0.0, 0.0))
        self.assertIsInstance(mapped, MappedControlState)


class IntentSnapshotTests(unittest.TestCase):
    @staticmethod
    def _command() -> CommandSample:
        return CommandSample(
            chunk_index=0,
            requested_velocity_mujoco=(0.5, 0.0, 0.0),
            desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    def test_is_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        snapshot = IntentSnapshot(
            revision=1, observed_ns=100, command=self._command()
        )
        with self.assertRaises(FrozenInstanceError):
            snapshot.revision = 2

    def test_rejects_nonpositive_revision(self) -> None:
        with self.assertRaises(ContractError):
            IntentSnapshot(revision=0, observed_ns=100, command=self._command())

    def test_rejects_negative_stamp(self) -> None:
        with self.assertRaises(ContractError):
            IntentSnapshot(revision=1, observed_ns=-1, command=self._command())

    def test_allows_none_command_for_terminate(self) -> None:
        snapshot = IntentSnapshot(revision=1, observed_ns=100, command=None)
        self.assertIsNone(snapshot.command)


EXPECTED_GRABBED_KEYS = (
    "W", "A", "S", "D", "Q", "E", "X", "SPACE",
    "LEFT", "UP", "RIGHT", "DOWN",
)


class _RecordingX11Lib:
    """Record the grab lifecycle without touching a real display."""

    def __init__(self, *, grab_errors: tuple[int, ...] = ()) -> None:
        self.grabs: list[tuple[int, int, int]] = []
        self.ungrabbed_windows: list[int] = []
        self.events: list[tuple] = []
        self.query_keymap_calls = 0
        self.sync_discards: list[int] = []
        self._grab_errors = list(grab_errors)

    def XGrabKey(self, display, keycode, modifiers, window, owner, pmode, kmode):
        self.grabs.append((int(keycode), int(modifiers), int(window)))
        return 0

    def XUngrabKey(self, display, keycode, modifiers, window):
        win = int(window)
        if not self.ungrabbed_windows or self.ungrabbed_windows[-1] != win:
            self.ungrabbed_windows.append(win)
        self.events.append(("ungrab", win))
        return 0

    def XSync(self, display, discard):
        self.events.append(("sync",))
        self.sync_discards.append(int(discard))
        # Deliver any queued asynchronous grab errors on the next sync.
        display_key = int(getattr(display, "value", display) or 0)
        if self._grab_errors:
            with operator_x11._X11_ERROR_LOCK:
                operator_x11._X11_ERRORS.setdefault(display_key, []).extend(
                    self._grab_errors
                )
            self._grab_errors = []
        return 0

    def XQueryKeymap(self, display, keymap):
        self.query_keymap_calls += 1
        return 1

    def XCloseDisplay(self, display):
        self.events.append(("close",))
        return 0


class X11GrabLifecycleTests(unittest.TestCase):
    def _recording_provider(
        self, *, grab_errors: tuple[int, ...] = ()
    ) -> X11KeyStateProvider:
        provider = X11KeyStateProvider.__new__(X11KeyStateProvider)
        provider._lib = _RecordingX11Lib(grab_errors=grab_errors)
        provider._display = ctypes.c_void_p(0x5150)
        provider._closed = False
        provider._grabbed_window = 0
        provider._keycodes = {
            key: index + 10 for index, key in enumerate(KEYSYMS)
        }
        return provider

    def test_controller_grabs_exclude_standalone_modifiers(self) -> None:
        self.assertEqual(_X11_GRABBED_CONTROL_KEYS, EXPECTED_GRABBED_KEYS)
        self.assertNotIn("LEFT_SHIFT", _X11_GRABBED_CONTROL_KEYS)
        self.assertNotIn("LEFT_CTRL", _X11_GRABBED_CONTROL_KEYS)

    def test_binding_target_grabs_each_control_key_once_for_all_modifiers(
        self,
    ) -> None:
        provider = self._recording_provider()
        provider._bind_target_window(91)
        provider._bind_target_window(91)
        self.assertEqual(
            provider._lib.grabs,
            [
                (provider._keycodes[key], _X11_ANY_MODIFIER, 91)
                for key in _X11_GRABBED_CONTROL_KEYS
            ],
        )
        self.assertEqual(provider._grabbed_window, 91)

    def test_rebinding_releases_old_window_before_grabbing_new_window(
        self,
    ) -> None:
        provider = self._recording_provider()
        provider._bind_target_window(91)
        provider._bind_target_window(92)
        self.assertEqual(provider._lib.ungrabbed_windows, [91])
        self.assertEqual(provider._grabbed_window, 92)

    def test_close_releases_grabs_before_display(self) -> None:
        provider = self._recording_provider()
        provider._bind_target_window(91)
        provider.close()
        last_ungrab = max(
            index
            for index, event in enumerate(provider._lib.events)
            if event == ("ungrab", 91)
        )
        self.assertEqual(
            provider._lib.events[last_ungrab + 1 :],
            [("sync",), ("close",)],
        )

    def test_bind_unexpected_error_fails_closed(self) -> None:
        provider = self._recording_provider(grab_errors=(2,))
        with self.assertRaises(ContractError):
            provider._bind_target_window(91)
        self.assertEqual(provider._grabbed_window, 0)
        self.assertEqual(provider._lib.ungrabbed_windows, [91])

    def test_bind_bad_window_race_clears_binding_for_reacquisition(
        self,
    ) -> None:
        provider = self._recording_provider(
            grab_errors=(operator_x11._X11_BAD_WINDOW,)
        )
        provider._bind_target_window(91)
        self.assertEqual(provider._grabbed_window, 0)

    def test_target_window_returns_the_titled_ancestor(self) -> None:
        provider = self._recording_provider()
        names = {17: None, 41: "MuJoCo", 1: "desktop"}
        parents = {17: 41, 41: 1, 1: 1}
        provider._fetch_name = lambda window: names[window]
        provider._parent = lambda window: parents[window]

        self.assertEqual(provider._target_window(17), 41)
        self.assertTrue(provider._focused_on_target(17))

    def test_sample_grabs_titled_ancestor_not_raw_focus_child(self) -> None:
        provider = self._recording_provider()
        provider._focus_window = lambda: 17
        provider._target_window = lambda window: 41

        levels = provider.sample()

        self.assertEqual(levels, KeyLevels(focused=True, pressed=frozenset()))
        self.assertEqual(provider._grabbed_window, 41)
        self.assertEqual({window for _, _, window in provider._lib.grabs}, {41})
        self.assertTrue(provider._lib.sync_discards)
        self.assertEqual(set(provider._lib.sync_discards), {1})

    def test_sample_bad_window_during_bind_returns_neutral_without_key_query(
        self,
    ) -> None:
        provider = self._recording_provider(
            grab_errors=(operator_x11._X11_BAD_WINDOW,)
        )
        provider._focus_window = lambda: 17
        provider._target_window = lambda window: 41

        levels = provider.sample()

        self.assertEqual(levels, KeyLevels(focused=False, pressed=frozenset()))
        self.assertEqual(provider._grabbed_window, 0)
        self.assertEqual(provider._lib.query_keymap_calls, 0)

    def test_sample_rebinds_before_reading_a_new_target_window(self) -> None:
        provider = self._recording_provider()
        provider._focus_window = lambda: 17
        targets = iter((41, 42))
        provider._target_window = lambda window: next(targets)

        provider.sample()
        provider.sample()

        self.assertEqual(provider._lib.ungrabbed_windows, [41])
        self.assertEqual(provider._grabbed_window, 42)
        self.assertEqual(provider._lib.query_keymap_calls, 2)

    def test_close_is_idempotent(self) -> None:
        provider = self._recording_provider()
        provider._bind_target_window(91)
        provider.close()
        provider.close()
        self.assertEqual(
            [event for event in provider._lib.events if event == ("close",)],
            [("close",)],
        )


class X11KeyStateProviderTests(unittest.TestCase):
    def test_destroyed_focus_window_degrades_to_neutral_sample(self) -> None:
        self.assertTrue(_classify_x11_errors((3,)))
        self.assertTrue(_classify_x11_errors((3, 3)))

    def test_non_bad_window_x11_error_remains_fatal(self) -> None:
        with self.assertRaises(ContractError):
            _classify_x11_errors((2,))

    def test_live_holden_viewer_title_is_an_operator_target(self) -> None:
        self.assertTrue(
            _is_target_window_title("G1 terrain motion matching - Holden runtime")
        )
        self.assertTrue(_is_target_window_title("MuJoCo"))
        self.assertTrue(_is_target_window_title("G1 CONTROLS"))
        self.assertFalse(_is_target_window_title("ubuntu"))

    def test_missing_display_hard_fails(self) -> None:
        previous = os.environ.pop("DISPLAY", None)
        try:
            with self.assertRaises(ContractError):
                X11KeyStateProvider()
        finally:
            if previous is not None:
                os.environ["DISPLAY"] = previous


if __name__ == "__main__":
    unittest.main()
