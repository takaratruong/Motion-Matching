# G1 Complete Horizontal Grid Playlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt all 11 horizontal lanes and visualize every certified complete traversal while explicitly reporting infeasible heights.

**Architecture:** A focused runner composes each lane from its own retrieved
phases when available, optionally tests a rigidly reused complete traversal,
and certifies oriented-sole contact. It emits one viewer-compatible playlist
containing only complete feasible lanes plus results for all 11 attempts.

**Tech Stack:** Python 3, NumPy, PyTorch terrain sampling, MuJoCo G1 kinematics, unittest.

## Global Constraints

- Use all 11 lane centers from `-1.0 m` through `+1.0 m` at `0.2 m` spacing.
- Every displayed lane must contain mount, crossing, and dismount.
- Do not accept reuse across different terrain heights without independent certification.
- Keep sole-clearance `-0.03 m`, stance-error `0.03 m`, and zero-unsupported-frame gates frozen.

---

### Task 1: Complete-grid translator and certifier

**Files:**
- Create: `resources/run_g1_complete_horizontal_grid.py`
- Create: `tests/python/test_run_g1_complete_horizontal_grid.py`

**Interfaces:**
- Consumes: accepted `composite.npz`, phase-grid summary, target dataset/config, and G1 XML.
- Produces: `complete-grid-playlist.npz`, `complete-grid-playlist.json`, and `metrics.json`.

- [ ] **Step 1: Write failing tests**

Test that all 11 lane attempts are reported, only complete certified connectors
enter the playlist, and every playlist segment identifies its lane.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=.:sonic/python sonic/.torch-mm-venv/bin/python -m unittest \
  tests.python.test_run_g1_complete_horizontal_grid -v
```

Expected: module import fails.

- [ ] **Step 3: Implement the runner**

Load each lane's mount/interior/dismount artifacts when present, compose and
certify the complete traversal, record missing or rejected phases as
infeasible, and concatenate complete lanes with 10-frame frozen separators.

- [ ] **Step 4: Verify GREEN and run the real artifact**

Run the unit test, then build the real grid and require 11 reported lane
attempts and at least the already demonstrated feasible lanes.

- [ ] **Step 5: Launch and verify**

Run the existing stair viewer with grid summary and playlist metadata at
50 Hz, confirm it remains live, and run the related regression tests.
