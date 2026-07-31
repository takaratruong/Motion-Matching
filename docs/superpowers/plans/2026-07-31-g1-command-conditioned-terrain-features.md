# G1 Command-Conditioned Terrain Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Augment the body-relative 91-height feature with four command-path heights without regressing terrain safety.

**Architecture:** Preserve the 13 by 7 body-yaw grid so terrain slope remains related to the robot's feet, then append four arc-distance samples from the shaped/recorded trajectory. Database, live query, and viewer share the same point definitions.

**Tech Stack:** Python 3, PyTorch, NumPy, unittest, MuJoCo viewer.

## Global Constraints

- Remain privileged-height-map kinematics only; do not add SONIC or tracking.
- Preserve the 13 by 7 footprint, 0.15 m spacing, and legacy encoder.
- Dense features are exactly 95 float32 values: 91 body-relative heights followed by four command-path heights.
- Pelvis clearance remains a safety diagnostic and is not a search feature.
- Query and database terrain features must use the same sampling primitive.
- Preserve untracked build artifacts and unrelated project changes.

---

### Task 1: Hybrid feature contract

**Files:**
- Modify: `tests/python/test_sonic_torch_terrain_features.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_features.py`

**Interfaces:**
- Consumes: `GeneratedFeatureState`, `CommandTrajectory`, and recorded root trajectories.
- Produces: `TerrainFeatureExtension.dimension == 95` and shared body/path sampling primitives.

- [ ] **Step 1: Write failing query tests**

Add assertions that command changes preserve the first 91 body-relative values,
change the final four path values, and bend the four-point profile on turns.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTHONPATH=sonic/python:. python -m unittest tests.python.test_sonic_torch_terrain_features -v`

Expected: failures showing the old dense row has shape 91 and does not include
the command trajectory.

- [ ] **Step 3: Implement the minimal shared sampler**

Retain the body-yaw sampler, append the existing four-point arc-distance profile,
and use the combined 95-value definition in both query and database paths.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_sonic_torch_terrain_features.py \
  sonic/python/mm_sonic/torch_terrain_features.py
git commit -m "feat: condition dense terrain features on command paths"
```

### Task 2: Viewer/search identity

**Files:**
- Modify: `tests/python/test_sonic_torch_terrain_live_viewer.py`
- Modify: `sonic/python/mm_sonic/torch_terrain_live_viewer.py`

**Interfaces:**
- Consumes: the shared path-aligned point constructor from Task 1.
- Produces: viewer markers at the exact 95 query sample locations.

- [ ] **Step 1: Write a failing viewer identity test**

Assert that changing the command trajectory changes viewer sample locations and
that the returned height coordinates equal authoritative grid samples.

- [ ] **Step 2: Run the viewer test and verify RED**

Run: `PYTHONPATH=sonic/python:. python -m unittest tests.python.test_sonic_torch_terrain_live_viewer -v`

Expected: failure because the viewer currently builds a root-yaw-rigid patch.

- [ ] **Step 3: Reuse the shared point constructor in the viewer**

Pass the current shaped trajectory into marker construction and remove the
duplicated rigid patch math.

- [ ] **Step 4: Run viewer and focused terrain tests**

Run both modules from Tasks 1 and 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_sonic_torch_terrain_live_viewer.py \
  sonic/python/mm_sonic/torch_terrain_live_viewer.py
git commit -m "fix: render command-conditioned terrain samples"
```

### Task 3: Automated and visual qualification

**Files:**
- Modify only if a test exposes a representation regression.

**Interfaces:**
- Consumes: the 95-value database/query feature and hybrid viewer.
- Produces: regression evidence and an interactive visual verdict.

- [ ] **Step 1: Run focused motion, terrain, contact, rollout, and viewer suites**

Run the relevant `tests.python.test_sonic_torch_*` modules. Expected: PASS with only documented opt-in skips.

- [ ] **Step 2: Run the full Python suite**

Run: `PYTHONPATH=sonic/python:. python -m unittest discover -s tests/python -v`

Expected: PASS with only documented skips.

- [ ] **Step 3: Run GPU adversarial terrain rollouts**

Use the representative GRAIL corpus and current contact-segment config. Compare
stalling, command progress, support, slide, and clearance against the retained
baseline; reject the change if safety regresses or command-dependent sampling
does not alter selection.

- [ ] **Step 4: Launch one clean viewer**

Close old terrain viewers and launch the representative staircase scene. Test
side approaches, diagonal ascent/descent, reversal, turning on treads, and side
exit. Report the visual result honestly; do not infer success from tests.
