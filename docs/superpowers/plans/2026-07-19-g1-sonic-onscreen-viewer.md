# G1 SONIC Onscreen Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open the official passive MuJoCo viewer on the active desktop while the existing interactive G1 SONIC pipeline remains the sole owner of physics and controls.

**Architecture:** An opt-in `onscreen: bool` travels from `manual_demo` through `GatedSimulatorClient` and the gated-simulator CLI into `ExternalGearBackend`. The backend asks pinned GEAR `BaseSimulator` for its passive viewer and synchronizes it after each existing physics step. Headless behavior remains the default.

**Tech Stack:** Python 3.12, `argparse`, MuJoCo Python viewer through pinned GEAR `BaseSimulator`, `unittest`, Bash, X11 display `:1`.

## Global Constraints

- The same scene, physics state, controller, rolling reference, and terminal controls must remain authoritative.
- Onscreen rendering is explicit and opt-in; automated evidence paths remain headless by default.
- Viewer synchronization must not add a physics step.
- Protocol stdout remains JSONL-only and viewer diagnostics remain on stderr.
- Viewer startup failure must be visible and must not silently fall back to headless mode.
- `/home/ubuntu/drive-g1-sonic.sh` must request GLFW onscreen rendering on desktop display `:1`.

---

### Task 1: Propagate the explicit onscreen contract

**Files:**
- Modify: `sonic/python/mm_sonic/process.py:1277-1320`
- Modify: `sonic/python/mm_sonic/manual_demo.py:215-235,426-445`
- Test: `tests/python/test_sonic_process.py`
- Test: `tests/python/test_sonic_manual_demo.py`

**Interfaces:**
- Consumes: existing `GatedSimulatorClient(..., unpaced_physics: bool, ...)` constructor and `manual_demo._parser()`.
- Produces: `_gated_simulator_command(run_root: Path, gear_checkout: Path, *, unpaced_physics: bool, onscreen: bool) -> tuple[str, ...]`; `GatedSimulatorClient(..., onscreen: bool = False, ...)`; generated simulator command contains `--onscreen` only when true; `manual_demo --onscreen` forwards true.

- [ ] **Step 1: Write failing propagation tests**

Import `_gated_simulator_command` in `tests/python/test_sonic_process.py` and add a pure generated-command assertion:

```python
headless = _gated_simulator_command(
    self.root,
    self.root / "gear",
    unpaced_physics=False,
    onscreen=False,
)
visible = _gated_simulator_command(
    self.root,
    self.root / "gear",
    unpaced_physics=False,
    onscreen=True,
)
self.assertNotIn("--onscreen", headless)
self.assertIn("--onscreen", visible)
with self.assertRaisesRegex(ValueError, "onscreen must be a boolean"):
    _gated_simulator_command(
        self.root,
        self.root / "gear",
        unpaced_physics=False,
        onscreen=1,
    )
```

Add a parser contract test in `tests/python/test_sonic_manual_demo.py`:

```python
from mm_sonic.manual_demo import _parser

def test_onscreen_is_explicit_and_opt_in(self) -> None:
    self.assertFalse(_parser().parse_args([]).onscreen)
    self.assertTrue(_parser().parse_args(["--onscreen"]).onscreen)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m pytest -q \
  tests/python/test_sonic_process.py \
  tests/python/test_sonic_manual_demo.py
```

Expected: failure because `--onscreen` and `onscreen` do not exist.

- [ ] **Step 3: Implement minimal flag propagation**

In `GatedSimulatorClient.__init__`, add and validate:

```python
onscreen: bool = False,

if type(onscreen) is not bool:
    raise ValueError("onscreen must be a boolean")
```

Append the generated child argument only when requested:

```python
*(("--onscreen",) if onscreen else ()),
```

Move the existing generated command tuple into this pure helper and call it
from the constructor:

```python
def _gated_simulator_command(
    run_root: Path,
    gear_checkout: Path,
    *,
    unpaced_physics: bool,
    onscreen: bool,
) -> tuple[str, ...]:
    if type(onscreen) is not bool:
        raise ValueError("onscreen must be a boolean")
    return (
        sys.executable,
        "-u",
        "-B",
        "-m",
        "mm_sonic.gated_sim",
        "--gear-checkout",
        str(gear_checkout),
        "--run-root",
        str(run_root),
        *(("--unpaced-physics",) if unpaced_physics else ()),
        *(("--onscreen",) if onscreen else ()),
    )
```

In `manual_demo`, pass `onscreen=namespace.onscreen`, and add:

```python
parser.add_argument("--onscreen", action="store_true")
```

- [ ] **Step 4: Run propagation tests and verify GREEN**

Run the Step 2 command.

Expected: all process and manual-demo tests pass.

---

### Task 2: Activate and synchronize the official passive viewer

**Files:**
- Modify: `sonic/python/mm_sonic/gated_sim.py:928-1020,1114-1150`
- Test: `tests/python/test_sonic_gated_sim.py:580-680`

**Interfaces:**
- Consumes: `ExternalGearBackend(..., wall_clock_pacing: bool = True)` and pinned GEAR `BaseSimulator` methods `sim_env.sim_step()` and `sim_env.update_viewer()`.
- Produces: `ExternalGearBackend(..., onscreen: bool = False)`; gated-simulator CLI `--onscreen`; exactly one viewer sync after each physics step when a viewer exists.

- [ ] **Step 1: Write failing backend tests**

Extend the existing constructor test to cover both values:

```python
ExternalGearBackend("/authenticated/gear", scene, onscreen=True)
self.assertTrue(captured["onscreen"])
self.assertFalse(captured["offscreen"])
```

Add an ordered step test with a fake environment:

```python
events = []
sim_env = SimpleNamespace(
    viewer=object(),
    sim_step=lambda: events.append("step"),
    update_viewer=lambda: events.append("sync"),
)
backend._simulator = SimpleNamespace(sim_dt=0.005, sim_env=sim_env)
backend._wall_clock_pacing = False
backend.step()
self.assertEqual(events, ["step", "sync"])
```

Add parser assertions that `--onscreen` defaults false and parses true.

- [ ] **Step 2: Run the focused backend tests and verify RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m pytest -q \
  tests/python/test_sonic_gated_sim.py
```

Expected: failures for the missing constructor/CLI flag and missing viewer sync.

- [ ] **Step 3: Implement minimal viewer activation**

Add `onscreen: bool = False` to `ExternalGearBackend`, validate it, and construct:

```python
simulator = bindings.base_simulator(
    config=config,
    env_name=config_loader.env_name,
    onscreen=onscreen,
    offscreen=False,
    enable_image_publish=False,
)
```

After the existing single `sim_step()` call, synchronize only an existing viewer:

```python
if self._simulator.sim_env.viewer is not None:
    self._simulator.sim_env.update_viewer()
```

Add `parser.add_argument("--onscreen", action="store_true")` and pass
`onscreen=args.onscreen` through the backend factory.

- [ ] **Step 4: Run focused backend tests and verify GREEN**

Run the Step 2 command.

Expected: all gated-simulator tests pass.

- [ ] **Step 5: Run the complete affected Python slice**

Run:

```bash
PYTHONPATH=sonic/python sonic/.venv/bin/python -m pytest -q \
  tests/python/test_sonic_process.py \
  tests/python/test_sonic_manual_demo.py \
  tests/python/test_sonic_gated_sim.py
```

Expected: all tests pass with no protocol-output regression.

- [ ] **Step 6: Commit the repository change**

```bash
git add sonic/python/mm_sonic/process.py \
  sonic/python/mm_sonic/manual_demo.py \
  sonic/python/mm_sonic/gated_sim.py \
  tests/python/test_sonic_process.py \
  tests/python/test_sonic_manual_demo.py \
  tests/python/test_sonic_gated_sim.py
git commit -m "feat: add G1 SONIC onscreen viewer mode"
```

---

### Task 3: Launch on the active desktop and prove operator control

**Files:**
- Modify: `/home/ubuntu/drive-g1-sonic.sh`

**Interfaces:**
- Consumes: `manual_demo --onscreen`, X11 `DISPLAY=:1`, existing terminal command loop.
- Produces: one visible MuJoCo window and the existing W/S/A/D, Q/E, space, X terminal controls.

- [ ] **Step 1: Update the convenience launcher**

Keep the existing worktree and command, but replace EGL with GLFW and request onscreen mode:

```bash
DISPLAY=${DISPLAY:-:1} \
MUJOCO_GL=glfw \
sonic/.venv/bin/python -u -m mm_sonic.manual_demo \
  --mode interactive \
  --onscreen \
  --chunks "$chunks"
```

- [ ] **Step 2: Validate launcher syntax**

Run:

```bash
bash -n /home/ubuntu/drive-g1-sonic.sh
```

Expected: exit 0 with no output.

- [ ] **Step 3: Start the desktop smoke run**

Run the launcher in a PTY and wait for both the MuJoCo window and terminal `LIVE` marker.

Expected: an X11 window owned by the gated-simulator process appears on display `:1`; the terminal prints `LIVE`.

- [ ] **Step 4: Exercise one command and clean exit**

Send `w`, wait for a boundary showing positive forward velocity, send space to stand, then send `x`.

Expected: the visible robot walks forward, returns to stand, the launcher exits 0, and:

```bash
pgrep -af 'mm_sonic|g1_deploy_onnx_ref|gated_sim' || true
```

shows no process from the completed smoke run.

- [ ] **Step 5: Relaunch for the operator**

Start `/home/ubuntu/drive-g1-sonic.sh` in a persistent PTY, wait for the visible window and `LIVE`, then leave it running for the user.
