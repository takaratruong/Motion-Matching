# Holden Command Cadence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the repo's default bounded stop/deceleration and 10 Hz command authority in the active single-GPU full-walking diagnostic.

**Architecture:** Persist one local planar shaped velocity in `HybridMatcher`, advance it with the existing `bounded_velocity_step`, and use the resulting speed consistently for terrain preview and feature queries. Permit diagnostic periodic search only when that filtered command is active on the GPU backend; carry Space separately from the viewer as an immediate hard stop.

**Tech Stack:** Python 3.11, NumPy, PyTorch control primitive, pytest, Ruff.

## Global Constraints

- Reuse `mm_sonic.torch_motion_matcher.bounded_velocity_step` and `MatcherConfig` defaults; do not duplicate their equations.
- Acceleration is 1.5 m/s^2, deceleration is 2.0 m/s^2, stop threshold is 0.05 m/s, and search cadence is 0.10 seconds.
- Preserve arrow signs, source SE(2), terrain height placement, scene masks, postprocessing, CPU diagnostic cadence, and formal/default semantics.
- Space remains immediate hard stop; ordinary key release decelerates.
- Write every regression before production code and observe the intended failure.
- Do not stage or modify `.superpowers/sdd/task-1-report.md` or `.superpowers/sdd/task-2-report.md`.

---

### Task 1: Transactional Holden velocity and GPU cadence

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_runtime.py`
- Test: `tests/python/test_hybrid_terrain_lmm_runtime.py`

**Interfaces:**
- Consumes: `bounded_velocity_step(current_velocity_world_xy, target_velocity_world_xy, *, config, dt)` and `MatcherConfig()` from `mm_sonic.torch_motion_matcher`.
- Produces: `HybridMatcher.step(command, *, dt=None, force_search=False, hard_stop=False)`, plus read-only controller identity attributes.

- [ ] **Step 1: Write the failing cadence and release tests**

Add focused tests that construct a diagnostic matcher with a fake single-GPU
search backend, hold `CommandState(1.0, 0.0)` across at least 0.10 seconds, and
assert a second search. Add a release test that records effective query speed,
proves the first released tick remains positive and decreases by at most
`2.0 * dt`, and proves exact neutral hold after the 0.05 m/s threshold.

- [ ] **Step 2: Write failing isolation and transaction tests**

Assert CPU diagnostic remains event-only, default mode retains its existing
cadence, `hard_stop=True` holds immediately, exhaustion restores shaped
velocity, and `reset()` zeros it. Patch `_command_query` and `_preview_points`
in one test to prove both receive the same shaped `CommandState.speed`.

- [ ] **Step 3: Run the focused RED tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_runtime.py \
  -k 'holden_velocity or gpu_diagnostic_periodic or hard_stop'
```

Expected: failures caused by the absent shaped-velocity state, absent
`hard_stop` keyword, and suppressed diagnostic periodic search.

- [ ] **Step 4: Implement the minimal runtime behavior**

Import `MatcherConfig` and `bounded_velocity_step`. Persist a two-element local
velocity initialized to zero. At each enabled single-GPU diagnostic step, shape
the target `(0.0, command.speed * walking_speed_p95_mps)`, clamp values below
`MatcherConfig.stop_speed_mps` to exact zero, and create an effective
`CommandState` whose speed is the shaped forward component divided by
`walking_speed_p95_mps` and whose steering is unchanged. Use that same command
for terrain preview and `_command_query`. Include shaped velocity in snapshot,
rollback, and reset. If `hard_stop` is true, zero it and return the current
diagnostic state without search. Enable `SEARCH_INTERVAL_S` matching only when
the GPU diagnostic effective command remains active.

- [ ] **Step 5: Run focused GREEN and runtime regression tests**

Run the Step 3 command, then:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_runtime.py
```

Expected: all pass.

### Task 2: Preserve Space and disclose the reused controller

**Files:**
- Modify: `sonic/python/mm_sonic/hybrid_terrain_lmm_viewer.py`
- Modify: `sonic/python/mm_sonic/full_walking_terrain_lmm_viewer.py`
- Test: `tests/python/test_hybrid_terrain_lmm_viewer.py`
- Test: `tests/python/test_full_walking_terrain_lmm_viewer.py`

**Interfaces:**
- Consumes: `HybridMatcher.step(..., hard_stop: bool = False)` and its controller identity attributes.
- Produces: hard-stop-aware fixed-rate interactive wiring and truthful full-runtime identity/label fields.

- [ ] **Step 1: Write failing viewer wiring and identity tests**

Drive ordinary arrow release and Space through the real command selection seam.
Assert ordinary release calls `step` without hard stop, while held Space calls
`step(..., hard_stop=True)`. Assert full runtime identity contains the exact
controller name and the four numeric defaults.

- [ ] **Step 2: Run viewer RED tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:resources:sonic/python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m pytest -q \
  tests/python/test_hybrid_terrain_lmm_viewer.py \
  tests/python/test_full_walking_terrain_lmm_viewer.py \
  -k 'hard_stop or holden_control_identity'
```

Expected: failures because hard-stop provenance and identity fields are absent.

- [ ] **Step 3: Implement minimal wiring and identity**

Capture `keys.hard_stop_active()` separately from the selected command, pass it
through the fixed-rate callback to `HybridMatcher.step`, and retain the existing
neutral command selection. Add controller identity, acceleration, deceleration,
stop threshold, and search interval to the full diagnostic identity and label;
do not alter formal acceptance identity.

- [ ] **Step 4: Run focused GREEN and the bounded aggregate**

Run Task 1 runtime tests plus both viewer files. Then run Ruff check,
Ruff format-check, `py_compile`, and `git diff --check` on only the four scoped
production/test files. Expected: all pass with no warnings.

- [ ] **Step 5: Request independent review and commit**

Review the scoped diff for command-coordinate preservation, transaction rollback,
Space behavior, CPU/formal isolation, and truthful identity. Stage only the four
production/test files and commit with:

```bash
git commit -m "fix: restore Holden command cadence"
```

- [ ] **Step 6: Replace the live viewer after review**

Verify the existing viewer PID by start ticks, cwd, and full command line before
sending SIGTERM. Relaunch the same ramp command on physical GPU5 from the new
commit. Confirm Up remains command-driven across the ramp and ordinary key
release visibly decelerates before neutral hold; do not run another 600-frame
validation.
