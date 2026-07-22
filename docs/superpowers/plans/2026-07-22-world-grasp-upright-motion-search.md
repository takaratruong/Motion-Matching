# World-Grasp Upright Motion Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make object transforms re-run grasp-based motion search while keeping full-body candidates upright and using arm IK for final grasp convergence.

**Architecture:** Replace full object-frame body mapping with the matcher's yaw/XZ-only scene alignment. Rank clips by their aligned recorded Contact wrist against the requested world grasp, shape each remaining clip through the existing seven-joint arm IK, then collision-filter and visualize only valid options.

**Tech Stack:** C++17, existing interaction pose/matcher/IK math, raylib, C++ unit tests, Python source-contract tests, Make.

## Global Constraints

- Requested world-space grasp position/orientation is the primary motion-search query.
- Target object pitch and roll never transform the candidate root, torso, or legs.
- Candidate scene alignment preserves world Y and uses only yaw plus X/Z translation.
- Use 0.12 m and 0.436332313 radians as Contact correction request limits.
- Position-only queries omit orientation ranking and correction.
- Object controls re-run selection, IK, and collision filtering in the same rendered frame.
- Keep the standalone viewer free of controller, diffusion, mesh, terrain, and capture code.

---

### Task 1: Upright world-grasp selector

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Produces `Transform hand_trajectory_scene_alignment(const HandTrajectory&, const HandTrajectoryQuery&)`.
- `select_hand_trajectories` returns only candidates whose aligned recorded Contact wrist is within configured world-grasp correction limits, sorted by normalized position/orientation/dimension cost.
- `map_hand_trajectory` returns the raw yaw/XZ-aligned wrist/elbow path without an exact full-body residual.

- [ ] **Step 1: Add failing regression tests**

Add two clips with Contact wrist rotations of identity and 0.30 radians about world X. Roll the target object and requested grasp by 0.30 radians. Assert the rolled-wrist clip becomes rank zero. Assert:

```cpp
const Transform alignment = hand_trajectory_scene_alignment(selected[0], query);
require(near(
    quat_mul_vec3(alignment.rotation, vec3(0.0F, 1.0F, 0.0F)),
    vec3(0.0F, 1.0F, 0.0F)),
    "object roll tilted the candidate body");
```

Add candidates at 0.119/0.121 m and 24.9/25.1 degrees from the requested Contact grasp and prove only the in-limit candidates remain. For position-only mode, prove orientation does not affect ranking.

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: the rolled query keeps the old rank or the new alignment API is missing.

- [ ] **Step 3: Implement matcher-consistent scene alignment and selection**

Copy the established yaw extraction/shortest-angle equations from `interaction_matcher.cpp::scene_alignment`. Build each trajectory before scoring, map its actual recorded Contact wrist through the upright alignment, and compare that pose directly with `query.grasp_world_position/rotation`. Set defaults to 0.12 m and 0.436332313 radians. Remove object-local grasp position/orientation scoring.

Refactor raw path mapping to apply only `hand_trajectory_scene_alignment`; remove `hand_trajectory_world_mapping` so no API can accidentally apply full target pitch/roll to a body.

- [ ] **Step 4: Verify and commit**

Run the focused C++ test and `git diff --check`; expect exit 0.

Commit with `git commit -m "fix: search upright motions by world grasp"`.

### Task 2: IK-shaped candidate motion

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Modify: `Makefile`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Add `ShapedHandTrajectory { std::vector<Pose> poses; MappedHandTrajectory path; bool contact_accepted; Reason reason; }`.
- Add `shape_hand_trajectory(const Database&, const HandTrajectory&, const HandTrajectoryQuery&, const IKConfig&)`.

- [ ] **Step 1: Add failing shaping tests**

Create a query with a bounded Contact offset. Assert the shaping API exists, returns one pose per Reach-through-Lift sample, leaves Simulation/Hips/leg local transforms unchanged from the upright-mapped source pose, and places the Contact wrist within configured solver acceptance. Add an over-limit query and prove `contact_accepted` is false.

- [ ] **Step 2: Verify RED**

Run the focused C++ build; expect missing shaping types/functions.

- [ ] **Step 3: Implement phase-weighted seven-joint IK**

For every source sample, apply scene alignment only to the Simulation root. Compute the aligned base wrist pose. Smoothstep correction weight from Reach entry to Contact and hold weight one afterward. Blend position toward the Contact translation residual and orientation toward the Contact orientation residual; omit orientation correction in position-only mode. Call `solve_hand_ik` on the active arm only and store the resulting pose and world wrist/elbow.

The Contact target passed to IK must equal the requested grasp exactly. Set `contact_accepted` from the Contact solve result. Link `interaction_ik.cpp` into the focused test and viewer targets.

- [ ] **Step 4: Verify and commit**

Run C++ tests and `git diff --check`; expect exit 0.

Commit with `git commit -m "feat: shape matched grasp motions with arm IK"`.

### Task 3: Live valid-set viewer

**Files:**
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`

**Interfaces:**
- Rebuilds geometric search, IK shaping, and collision filtering after every object transform change.
- Cycles only IK-accepted, collision-safe candidates and renders their shaped poses/paths.

- [ ] **Step 1: Extend source-contract tests and verify RED**

Require `shape_hand_trajectory(`, `hand_trajectory_scene_alignment(`, selection inside the object-change branch, zero-valid handling, and absence of `hand_trajectory_world_mapping(`. Run the Python viewer tests and confirm expected failures.

- [ ] **Step 2: Rebuild the valid set live**

Replace the static candidate list with a function that calls `select_hand_trajectories`, shapes every candidate, collision-tests the shaped wrist/elbow path, and retains only accepted safe motions. Preserve the selected clip when it remains valid; otherwise select rank zero. Empty results render the table/object and a clear `0 valid motions` status without indexing vectors.

- [ ] **Step 3: Render shaped path and skeleton**

Use the selected `ShapedHandTrajectory::poses` for animated skeleton rendering and its path for highlighting. Do not apply another object transform during drawing. Show valid count, selected clip, position/orientation correction, and current animation phase.

- [ ] **Step 4: Verify and commit**

Run focused C++ tests, Python viewer tests, `make -j2 hand_trajectory_viewer`, and `git diff --check`; expect all commands to exit 0.

Commit with `git commit -m "fix: resample valid motions when grasp changes"`.

### Task 4: Replace the incorrect live process

- [ ] Stop only the exact `^./hand_trajectory_viewer$` PID.
- [ ] Launch one freshly built viewer on `DISPLAY=:1` with the existing full pack.
- [ ] Verify one viewer, zero old diffusion controllers, zero diffusion environment variables, no runtime error lines, matching disk/running binary hash, and a clean worktree.
- [ ] Do not use screenshot or display-capture commands.
