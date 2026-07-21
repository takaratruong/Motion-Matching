# Onscreen Fall Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the terrain and first fallen pose in manual onscreen SONIC runs without changing headless/scored simulator behavior.

**Architecture:** Plumb one strict, opt-in `freeze_on_fall` flag into the existing simulator adapter. The adapter suppresses the upstream automatic reset only in visible manual runs, latches the first fall, and services later step requests with a frozen LowState plus exact protocol-clock advancement.

**Tech Stack:** Python 3.10, `unittest`, NumPy, the existing JSONL gated-simulator protocol, and the pinned official GEAR simulator.

## Global Constraints

- Do not edit or fork the official GEAR checkout.
- `freeze_on_fall=True` is valid only with `onscreen=True`.
- Headless, scored, and injected-command clients remain behaviorally unchanged.
- Preserve the first fallen `qpos`; do not reset or continue physics afterward.
- Keep GEAR connected by publishing frozen LowState on every frozen step.
- Advance `data.time` by exactly `sim_dt` per frozen step so the existing protocol remains exact.
- Use test-first implementation and run warning-strict gates.

---

### Task 1: Test and implement the adapter fall latch

**Files:**
- Modify: `tests/python/test_sonic_gated_sim.py`
- Modify: `sonic/python/mm_sonic/gated_sim.py`

**Interfaces:**
- Consumes: `ExternalGearBackend(..., onscreen: bool, freeze_on_fall: bool = False)`.
- Produces: adapter-owned fall callback and frozen-step path; no protocol schema change.

- [ ] **Step 1: Write failing adapter tests**

Add tests that construct a lightweight `ExternalGearBackend` with fake viewer,
MuJoCo data, bridge, and simulator environment. Assert that default stepping
still calls `sim_step`, while freeze mode rejects headless use, prevents the
official reset callback, preserves fallen qpos, zeros `qvel`/`qacc`/`ctrl`, and
on the next step publishes LowState, increments time by exactly `sim_dt`, and
syncs the viewer without calling physics.

- [ ] **Step 2: Run RED**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_gated_sim
```

Expected: FAIL because `freeze_on_fall` and the adapter latch do not exist.

- [ ] **Step 3: Implement the minimum adapter behavior**

Add the strict constructor option, install an instance-local fall callback only
when enabled, and add a private frozen-step helper. Preserve the normal `step()`
path byte-for-byte in effect when disabled. Redirect the single fall diagnostic
to stderr through the existing `redirect_stdout` boundary.

- [ ] **Step 4: Run GREEN**

Re-run the exact Task 1 command. Expected: all gated-simulator tests pass with no
warnings.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/gated_sim.py tests/python/test_sonic_gated_sim.py
git commit -m "feat: freeze onscreen simulator after falls"
```

### Task 2: Test and implement explicit flag propagation

**Files:**
- Modify: `tests/python/test_sonic_process.py`
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `sonic/python/mm_sonic/process.py`
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/gated_sim.py`

**Interfaces:**
- Consumes: `_gated_simulator_command(..., onscreen: bool, freeze_on_fall: bool)` and `GatedSimulatorClient(..., freeze_on_fall: bool = False)`.
- Produces: `--freeze-on-fall` on the child command only for visible manual runs.

- [ ] **Step 1: Write failing propagation tests**

Extend command-generation tests to require strict booleans, reject
`freeze_on_fall=True` with `onscreen=False`, and include `--freeze-on-fall`
exactly once only when enabled. Extend manual-demo construction coverage to
assert `freeze_on_fall` equals `namespace.onscreen`. Extend parser/backend-factory
coverage to assert the child flag reaches `ExternalGearBackend`.

- [ ] **Step 2: Run RED**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_process tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_gated_sim
```

Expected: FAIL on the missing constructor, command, and parser arguments.

- [ ] **Step 3: Implement minimum propagation**

Add strict validation and command assembly in `process.py`, add the parser flag
and backend-factory forwarding in `gated_sim.py`, and pass
`freeze_on_fall=namespace.onscreen` only from `manual_demo.py`.

- [ ] **Step 4: Run GREEN and regression gates**

Run the exact focused command above, then:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error \
  PYTHONPATH=sonic/python sonic/.venv/bin/python -B -m unittest discover -v \
  -s tests/python -p 'test_sonic_*.py'
```

Expected: all tests pass with no warnings; optional real-artifact skips remain
explicit skips rather than failures.

- [ ] **Step 5: Commit**

```bash
git add sonic/python/mm_sonic/process.py sonic/python/mm_sonic/manual_demo.py \
  sonic/python/mm_sonic/gated_sim.py tests/python/test_sonic_process.py \
  tests/python/test_sonic_manual_demo.py tests/python/test_sonic_gated_sim.py
git commit -m "feat: enable fall freeze for visible manual runs"
```

### Task 3: Protected verification and live confirmation

**Files:**
- Verify only: repository and retained live-run artifacts

**Interfaces:**
- Consumes: Tasks 1-2 and the existing terrain-aware manual launch command.
- Produces: warning-strict test evidence and one retained visible diagnostic run.

- [ ] **Step 1: Run protected diff and test gates**

Run `git diff --check`, the focused suites from Task 2, and the full Python suite
from Task 2 in a controller-owned clean projection. Expected: zero failures and
no warnings.

- [ ] **Step 2: Relaunch the terrain-aware demo**

Launch `holden-turn-v1` with `--onscreen --responsive
--responsive-source-intervals 5` in an isolated loopback namespace. Confirm the
child command contains `--freeze-on-fall` and the scene remains loaded.

- [ ] **Step 3: Confirm the live fall contract**

If a fall occurs, require one fall diagnostic, monotonically increasing protocol
time, unchanged frozen qpos, a running viewer, visible terrain, and orderly X
shutdown. If no fall occurs during the bounded test, retain the passing automated
evidence and do not fabricate live-fall confirmation.

