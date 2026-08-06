# G1 Arbitrary-Heading Root-Path Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the existing certified multi-window root-path search evaluate a straight 45-degree staircase path without changing its retrieval, placement, transition, or terrain-contact thresholds.

**Architecture:** Add an endpoint-based path contract to the existing chain runner while retaining its legacy constant-Y CLI. Resolve either input form into a start point, stop point, unit heading, and Euclidean length before the existing search pipeline runs.

**Tech Stack:** Python 3.10, NumPy, existing GRAIL path-placement and terrain-contact modules, `unittest`.

## Global Constraints

- Kinematic playback only; do not connect Sonic or physics tracking.
- Preserve raw GRAIL joint positions and the existing certification thresholds.
- Reject ambiguous or degenerate path specifications.
- Preserve the legacy `--start-x`, `--stop-x`, and `--path-y` behavior.
- Do not add heading-specific retrieval parameters.

---

### Task 1: Arbitrary endpoint path contract

**Files:**
- Modify: `resources/run_g1_root_path_motion_search.py`
- Modify: `tests/python/test_run_g1_root_path_motion_search.py`

**Interfaces:**
- Consumes: either legacy `start_x`, `stop_x`, `path_y` arguments or explicit `path_start`, `path_stop` XY endpoints.
- Produces: `_resolve_path(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]`.

- [ ] **Step 1: Write failing endpoint-resolution tests**

Add tests proving that explicit endpoints `(-1, 1)` and `(1, -1)` produce a normalized `(sqrt(0.5), -sqrt(0.5))` heading and `sqrt(8)` length, legacy arguments still produce heading `(1, 0)`, and mixed/degenerate specifications raise `ContractError`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_root_path_motion_search -v
```

Expected: import failure because `_resolve_path` does not exist.

- [ ] **Step 3: Implement the minimal endpoint contract**

Add optional `--path-start X Y` and `--path-stop X Y` arguments, make `--path-y` optional, implement `_resolve_path`, and replace the hard-coded horizontal geometry in `main`. Store resolved endpoints in the coarse-pool cache contract so incompatible paths cannot reuse a cache.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_root_path_motion_search \
  tests.python.test_run_g1_path_motion_placement \
  tests.python.test_sonic_torch_root_path_motion_graph -v
```

Expected: all tests pass.

- [ ] **Step 5: Run and inspect the 45-degree experiment**

Run the chain search on the approved horizontal path rotated by -45 degrees around its midpoint. Require complete graph coverage and successful composite certification before rendering and launching the MuJoCo viewer. If coverage is incomplete, report the exact uncovered interval and rejection reasons rather than synthesizing motion.
