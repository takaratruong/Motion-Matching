# G1 Stable Mount Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every selected horizontal-grid mount seed ends at a stable supported boundary.

**Architecture:** Filter source motion windows for mount tasks before signature scoring. Preserve the existing window set for interior and dismount tasks, then rerun the same placement certification and ranking.

**Tech Stack:** Python 3.10, NumPy, unittest, existing GRAIL motion-placement pipeline.

## Global Constraints

- Maximum consecutive unsupported mount frames: 4.
- Minimum terminal double-support mount frames: 8.
- Apply the gate to mount tasks only.
- Report a lane infeasible rather than emitting a contract-violating mount.

---

### Task 1: Mount-window admission

**Files:**
- Modify: `resources/run_g1_horizontal_grid_phase_coverage.py`
- Test: `tests/python/test_run_g1_horizontal_grid_phase_coverage.py`

**Interfaces:**
- Consumes: `RawMotionWindow` and a source support mask shaped `(frames, 2)`.
- Produces: `_stable_mount_window(window, support_mask) -> bool`.

- [ ] **Step 1: Write the failing boundary tests**

Add tests proving that a 20-frame aerial interval and a terminal
single-support interval are rejected, while a one-frame flight followed by
eight double-support frames is accepted.

- [ ] **Step 2: Run the focused test and verify RED**

Run:
`PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_run_g1_horizontal_grid_phase_coverage`

Expected: import or assertion failure for the missing admission function.

- [ ] **Step 3: Implement the admission predicate**

Validate the support slice selected by `window.start_frame:window.stop_frame`,
measure the maximum unsupported run and terminal double-support run, and return
whether both configured limits pass.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Mount-only scan integration

**Files:**
- Modify: `resources/run_g1_horizontal_grid_phase_coverage.py`
- Test: `tests/python/test_run_g1_horizontal_grid_phase_coverage.py`

**Interfaces:**
- Consumes: per-task phase kinds and the admitted mount-window subset.
- Produces: ordinary rows for interior/dismount tasks and stable rows for mount tasks.

- [ ] **Step 1: Write a failing task-routing test**

Construct one stable and one unstable window and prove that mount scoring sees
only the stable window while interior scoring sees both.

- [ ] **Step 2: Run the focused test and verify RED**

Run the focused unittest command. Expected: assertion failure because task
kinds are not yet routed.

- [ ] **Step 3: Route mount tasks through the admitted subset**

Pass phase kinds into `_scan_one` and select the stable subset only when the
task kind equals `mount`.

- [ ] **Step 4: Run focused and related tests**

Run:
`PYTHONPATH=sonic/python:. sonic/.torch-mm-venv/bin/python -m unittest tests.python.test_run_g1_horizontal_grid_phase_coverage tests.python.test_run_g1_feature_seed_playlist`

Expected: all tests pass.

### Task 3: Full-grid qualification

**Files:**
- Generate: `build/g1-horizontal-grid-phase-20cm-stable-mount-v1/`

**Interfaces:**
- Consumes: the full GRAIL terrain dataset and target staircase.
- Produces: regenerated phase summary, connectors, and complete mount audit.

- [ ] **Step 1: Rerun the 20 cm phase grid**

Use the same target, bounds, G1 XML, shortlist, and worker count as the prior
`g1-horizontal-grid-phase-20cm-v2` run.

- [ ] **Step 2: Audit every selected mount**

Recompute maximum unsupported runs and terminal double-support runs directly
from source support masks. Expected: every selected mount passes `<= 4` and
`>= 8`.

- [ ] **Step 3: Build and launch a full mount playlist**

Include every selected mount with holds and teleport boundaries, then launch
the MuJoCo viewer. Do not substitute a single hand-picked candidate.

