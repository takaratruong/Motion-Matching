# G1 SONIC Viewer Key Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the existing Sonic `WASD` controls while preventing those controls from mutating MuJoCo viewer state or hiding terrain.

**Architecture:** The existing `X11KeyStateProvider` will own passive X11 grabs for every non-modifier Sonic command key on the authenticated focused operator window. Sonic will continue reading physical levels with `XQueryKeymap`; the MuJoCo client will no longer receive the shortcut events. The onscreen backend will separately normalize the two presentation bits required for terrain visibility when the viewer is created.

**Tech Stack:** Python 3.10, `ctypes`/libX11, MuJoCo passive viewer, `unittest`, existing `uv` Sonic environment.

## Global Constraints

- Preserve `W`, `A`, `S`, `D`, `Q`, `E`, `X`, `Space`, arrow, Shift, and Control mappings.
- Headless and non-X11 modes remain unchanged.
- Do not change motion matching, the control-gate protocol, physics stepping, scene registration, collision assets, or the GEAR policy.
- Passive grabs cover every modifier combination and omit independent Shift/Control grabs.
- X11 grab failure is fail-closed; focus loss still publishes neutral.
- Use test-first RED/GREEN cycles and preserve a clean worktree after each commit.

---

### Task 1: Isolate Sonic command keys from MuJoCo shortcuts

**Files:**
- Modify: `sonic/python/mm_sonic/operator_x11.py:27-69,435-581`
- Test: `tests/python/test_sonic_operator_x11.py:1-26,358-383`

**Interfaces:**
- Consumes: `KEYSYMS`, focused-window ancestry, `_take_x11_errors`, and `_classify_x11_errors`.
- Produces: `_GRABBED_CONTROL_KEYS: tuple[str, ...]`, `X11KeyStateProvider._target_window(window: int) -> int`, `X11KeyStateProvider._bind_target_window(window: int) -> None`, and idempotent grab release during `close()`.

- [ ] **Step 1: Write failing provider-boundary tests**

Add a small recording X11 library double and construct the provider with `__new__` so the test does not require a real display. The tests must assert the exact command-key set and call contract:

```python
EXPECTED_GRABBED_KEYS = {
    "W", "A", "S", "D", "Q", "E", "X", "SPACE",
    "LEFT", "UP", "RIGHT", "DOWN",
}

def test_controller_grabs_exclude_standalone_modifiers(self) -> None:
    self.assertEqual(set(_GRABBED_CONTROL_KEYS), EXPECTED_GRABBED_KEYS)
    self.assertNotIn("LEFT_SHIFT", _GRABBED_CONTROL_KEYS)
    self.assertNotIn("LEFT_CTRL", _GRABBED_CONTROL_KEYS)

def test_binding_target_grabs_each_control_key_once_for_all_modifiers(self) -> None:
    provider = self._recording_provider()
    provider._bind_target_window(91)
    provider._bind_target_window(91)
    self.assertEqual(
        provider._lib.grabs,
        [(provider._keycodes[key], _X11_ANY_MODIFIER, 91)
         for key in _GRABBED_CONTROL_KEYS],
    )

def test_rebinding_releases_old_window_before_grabbing_new_window(self) -> None:
    provider = self._recording_provider()
    provider._bind_target_window(91)
    provider._bind_target_window(92)
    self.assertEqual(provider._lib.ungrabbed_windows, [91])
    self.assertEqual(provider._grabbed_window, 92)

def test_close_releases_grabs_before_display(self) -> None:
    provider = self._recording_provider()
    provider._bind_target_window(91)
    provider.close()
    self.assertEqual(provider._lib.events[-2:], [("ungrab", 91), ("close",)])
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
UV_NO_SYNC=1 uv run --project sonic --extra test \
  python -m unittest tests.python.test_sonic_operator_x11.X11KeyStateProviderTests -v
```

Expected: FAIL because `_GRABBED_CONTROL_KEYS`, `_X11_ANY_MODIFIER`, and `_bind_target_window` do not exist.

- [ ] **Step 3: Implement the minimal X11 grab lifecycle**

Add exact constants and configure the two Xlib calls:

```python
_GRABBED_CONTROL_KEYS = tuple(
    key for key in KEYSYMS if key not in {"LEFT_SHIFT", "LEFT_CTRL"}
)
_X11_ANY_MODIFIER = 1 << 15
_X11_GRAB_MODE_ASYNC = 1

lib.XGrabKey.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.XGrabKey.restype = ctypes.c_int
lib.XUngrabKey.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_ulong,
]
lib.XUngrabKey.restype = ctypes.c_int
```

Initialize `_grabbed_window = 0`. Replace the boolean ancestry helper with one
that returns the authenticated ancestor window. `_bind_target_window` must be
idempotent for the same window, ungrab the prior window before replacement,
call `XGrabKey(..., owner_events=0, pointer_mode=1, keyboard_mode=1)` once for
each available command keycode, `XSync`, and classify asynchronous errors. A
non-`BadWindow` error raises `ContractError`; a destroyed-window race clears
the binding and returns a neutral sample. `sample()` binds before querying the
keymap. `close()` ungrabs before `XCloseDisplay` and remains idempotent.

- [ ] **Step 4: Run focused and complete X11 tests and confirm GREEN**

Run:

```bash
UV_NO_SYNC=1 uv run --project sonic --extra test \
  python -m unittest tests.python.test_sonic_operator_x11 -v
```

Expected: all tests pass with no X11 warnings.

- [ ] **Step 5: Commit the isolated input fix**

```bash
git add sonic/python/mm_sonic/operator_x11.py \
  tests/python/test_sonic_operator_x11.py
git commit -m "fix: isolate Sonic controls from MuJoCo shortcuts"
```

### Task 2: Normalize terrain presentation when the viewer starts

**Files:**
- Modify: `sonic/python/mm_sonic/gated_sim.py:1085-1133`
- Test: `tests/python/test_sonic_gated_sim.py:1097-1167`

**Interfaces:**
- Consumes: `bindings.mujoco.mjtVisFlag.mjVIS_STATIC`, `simulator.sim_env.viewer.opt.flags`, and `viewer.opt.geomgroup`.
- Produces: `_normalize_onscreen_viewer(bindings: _ExternalBindings, simulator: object) -> None`, invoked only for `onscreen=True`.

- [ ] **Step 1: Write failing onscreen normalization tests**

Add a helper fixture with mutable flag/group arrays and tests that establish the
expected boundary:

```python
def test_onscreen_backend_enables_static_terrain_presentation(self) -> None:
    viewer = SimpleNamespace(
        opt=SimpleNamespace(flags=[0] * 31, geomgroup=[0] * 6),
        is_running=lambda: True,
    )
    simulator = SimpleNamespace(sim_env=SimpleNamespace(viewer=viewer))
    bindings = SimpleNamespace(
        mujoco=SimpleNamespace(
            mjtVisFlag=SimpleNamespace(mjVIS_STATIC=22),
        )
    )
    _normalize_onscreen_viewer(bindings, simulator)
    self.assertEqual(viewer.opt.flags[22], 1)
    self.assertEqual(viewer.opt.geomgroup[2], 1)

def test_onscreen_normalization_rejects_missing_or_closed_viewer(self) -> None:
    # Check both viewer=None and is_running() == False.
    with self.assertRaisesRegex(ProtocolError, "viewer"):
        _normalize_onscreen_viewer(bindings, simulator)
```

Update the existing explicit-onscreen constructor fixture so `onscreen=True`
returns a running viewer with `opt.flags` and `opt.geomgroup` arrays.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
UV_NO_SYNC=1 uv run --project sonic --extra test \
  python -m unittest \
  tests.python.test_sonic_gated_sim.ExternalGearBackendBoundaryTests -v
```

Expected: FAIL because `_normalize_onscreen_viewer` does not exist and the
onscreen constructor does not normalize presentation state.

- [ ] **Step 3: Implement startup-only viewer normalization**

Implement a private helper that rejects a missing/closed viewer, validates the
option arrays have the required indices, assigns `1` to `mjVIS_STATIC` and geom
group `2`, and performs no physics step or viewer sync. Call it immediately
after the official simulator is constructed, but only when `onscreen=True`.
Translate malformed pinned-viewer state into `ProtocolError` with a clear
`viewer presentation` message.

- [ ] **Step 4: Run focused simulator tests and confirm GREEN**

Run:

```bash
UV_NO_SYNC=1 uv run --project sonic --extra test \
  python -m unittest tests.python.test_sonic_gated_sim -v
```

Expected: all gated-simulator tests pass.

- [ ] **Step 5: Commit viewer normalization**

```bash
git add sonic/python/mm_sonic/gated_sim.py \
  tests/python/test_sonic_gated_sim.py
git commit -m "fix: normalize Sonic terrain viewer state"
```

### Task 3: Regression, real visible canary, and cleanup proof

**Files:**
- Modify only if a test exposes a defect in Task 1 or Task 2.
- Evidence root: `/home/ubuntu/mm-sonic-control-gate-live/operator-visible-key-isolation`

**Interfaces:**
- Consumes: the Task 1 X11 grab lifecycle and Task 2 viewer normalization.
- Produces: protected test output, a visible terrain screenshot, right-command evidence, `mm_terrain` contact evidence, and zero leaked processes.

- [ ] **Step 1: Run the protected Python suites**

```bash
UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --project sonic --extra test python -m unittest \
  tests.python.test_sonic_operator_x11 \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_responsive_scheduler \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_gated_sim -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the real visible curb canary**

Run:

```bash
DISPLAY=:1 PYTHONUNBUFFERED=1 UV_NO_SYNC=1 \
  uv run --project sonic --extra integration python -m mm_sonic.manual_demo \
  --mode interactive --chunks 300 --onscreen --responsive \
  --responsive-source-intervals 5 \
  --scene-id grail-curb-default --route-id curb-forward --terrain-weight 4 \
  --movement-model holden-turn-v1 --input-source x11 \
  --output-root \
    /home/ubuntu/mm-sonic-control-gate-live/operator-visible-key-isolation \
  --source-run \
    /home/ubuntu/mm-flat-walk-stage-b-r13-command-responsive.alXnDZ/stage-b/stage-b-20260718T004729900042Z-f334fb47 \
  --gear-checkout \
    /home/ubuntu/projects/gear-sonic-worktrees/simulation-control-gate \
  --runtime \
    /home/ubuntu/.local/share/motion-matching-deps/gear-sonic/5e22ddc69abcea2a9aafc40536b14c232d3f9d7f \
  --terrain-dir /home/ubuntu/projects/motion-matching/resources/g1_terrain \
  --mm-server /tmp/g1-control-gate-mm-build/mm_chunk_server
```

Expected: the process reports `LIVE X11`, `SONIC first action ready`, and
`SONIC policy command received` while the MuJoCo window remains running.

- [ ] **Step 3: Exercise `D` and prove render/control isolation**

Focus the MuJoCo window, hold `D` long enough for the continuous provider to
sample it, and release. Confirm terminal output contains `KEY D DOWN -> right`
and `KEY D UP -> right`. Capture the actual MuJoCo window and inspect that the
terrain remains visible after the key release. Task 2's focused test must
already prove the startup values `mjVIS_STATIC == 1` and geom group `2 == 1`;
the post-`D` screenshot proves that the live key never toggled them away.

- [ ] **Step 4: Prove collision and clean shutdown**

Exit through `X`. Confirm the scored contact log contains contacts whose
`geom1` or `geom2` is `mm_terrain`, the run exits without a controller error,
and `ps` shows no child `manual_demo`, `gated_sim`, `mm_chunk_server`, or
`g1_deploy_onnx_ref` process for this run.

- [ ] **Step 5: Review and commit any verification-only correction**

Run `git diff --check`, inspect the complete diff against this plan, and commit
only if the canary required a focused correction. Do not add project-external
or Reliable Claude framework changes for a task-local failure.
