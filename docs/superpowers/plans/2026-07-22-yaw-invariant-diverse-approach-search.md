# Yaw-Invariant Diverse Approach Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every collision-safe approach side eligible in the coverage viewer so retrieval does not lose diversity under object yaw.

**Architecture:** Remove the coverage viewer's fixed world-front rejection policy and its obsolete diagnostic. Leave trajectory alignment, retrieval, IK, collision evaluation, and the optional library-side filtering API unchanged.

**Tech Stack:** C++17, raylib viewer, Python `unittest`, GNU Make.

## Global Constraints

- Keep the 12-option display cap and exact-first axis-fallback search.
- Keep the 4 cm position, 15 degree approach-axis, and 60 degree full-orientation gates.
- Keep object and full-body furniture collision hard.
- Keep Enter-only search, existing controls, and option cycling.
- Do not alter the interaction pack, live controller, mesh renderer, or terrain renderer.

---

### Task 1: Retain every collision-safe approach side

**Files:**
- Modify: `tests/python/test_hand_trajectory_viewer.py`
- Modify: `hand_trajectory_viewer.cpp`

**Interfaces:**
- Consumes: the existing exact and axis-fallback candidate vectors.
- Produces: `TrajectorySet::valid` without fixed world-front rejection.

- [ ] **Step 1: Write the failing viewer contract test**

In `test_rebuilds_world_grasp_shaped_valid_set_live`, remove the positive
`wrong_side` assertion. Rename
`test_uses_exact_first_axis_fallback_and_front_side_filter` to
`test_uses_exact_first_axis_fallback_and_all_approach_sides` and add:

```python
self.assertNotIn("kSceneFront", source)
self.assertNotIn("starts_on_allowed_side(", source)
self.assertNotIn("wrong_side", source)
```

Retain assertions for `kTargetValidTrajectories = 12U`, exact-first search,
axis fallback, compatibility counters, and tier labels.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. python3 -m unittest \
  tests.python.test_hand_trajectory_viewer -v
```

Expected: FAIL because the viewer still defines `kSceneFront`, calls
`starts_on_allowed_side`, and reports `wrong_side`.

- [ ] **Step 3: Remove only the viewer directional policy**

Delete:

```cpp
constexpr vec3 kSceneFront(0.0F, 0.0F, -1.0F);
```

Delete `TrajectorySet::wrong_side` and this block from `process_candidates`:

```cpp
if (!interaction::starts_on_allowed_side(
        candidate, query, kSceneFront)) {
    ++result.wrong_side;
    continue;
}
```

Change the rejection HUD from:

```cpp
"wrong-side %i | IK %i | object %i | environment %i"
```

to:

```cpp
"IK %i | object %i | environment %i"
```

and remove the `trajectories.wrong_side` format argument. Do not change
`interaction_hand_trajectories.cpp` or its `starts_on_allowed_side` API.

- [ ] **Step 4: Verify GREEN and build**

```bash
PYTHONPATH=. python3 -m unittest \
  tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
```

Expected: 10 viewer tests pass, the C++ trajectory binary exits zero, the
release viewer builds, and `git diff --check` reports no errors.

- [ ] **Step 5: Commit**

```bash
git add hand_trajectory_viewer.cpp tests/python/test_hand_trajectory_viewer.py
git commit -m "feat: retain yaw-invariant approach diversity"
```

### Task 2: Regression gate and live viewer replacement

**Files:**
- Modify only Task 1 files if verification exposes a task-related defect.

**Interfaces:**
- Consumes: the release viewer and `build/smart-pickup/table-ground-pack`.
- Produces: one verified live coverage viewer while preserving the controller.

- [ ] **Step 1: Run the complete branch gate**

```bash
python3 -m unittest \
  tests.python.test_interaction_sources \
  tests.python.test_interaction_build_cli \
  tests.python.test_interaction_artifacts \
  tests.python.test_hand_trajectory_viewer
make -B build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_trajectory_database
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make hand_trajectory_viewer
git diff --check
git status --short
```

Expected: 105 Python tests pass, both C++ binaries exit zero, the release
viewer builds, and the worktree is clean.

- [ ] **Step 2: Replace only the coverage viewer**

Resolve the single process matching `^./hand_trajectory_viewer$`, verify its
cwd is this worktree, send SIGTERM only to that PID, and wait for exit. Launch
the new viewer on `DISPLAY=:1` with
`build/smart-pickup/table-ground-pack`. Do not signal controller PID `556617`.

- [ ] **Step 3: Verify the live process**

Confirm exactly one viewer, successful raylib initialization, matching
disk/live SHA-256, correct `DISPLAY` and pack environment, no logged errors,
and RSS below 1.1 GiB for 10 seconds. Do not use screenshots.
