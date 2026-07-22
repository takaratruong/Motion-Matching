# Grasp-Anchored Live Motion Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search and display complete, collision-safe pickup motions from a staged world-space grasp without using object identity or rotation to transform the body.

**Architecture:** Replace object-anchored scene alignment with Contact-wrist grasp anchoring that applies only yaw and X/Z translation. Extract through the final Lift frame, shape the active arm with the existing IK solver, collision-test every shaped skeleton, and rebuild/rank-zero the valid set after every grasp transform.

**Tech Stack:** C++17, existing interaction pose/IK math, raylib, C++ regression tests, Python source-contract tests, Make.

## Global Constraints

- Requested world-space grasp pose is the retrieval anchor.
- Object identity, dimensions, pitch, and roll do not affect retrieval ranking or whole-body alignment.
- Whole-body alignment uses yaw and X/Z translation only and preserves world Y/up.
- Position-only mode retains grasp heading for alignment but omits orientation ranking and IK correction.
- Motions span first Reach through the final contiguous Lift frame.
- Grasp transforms only update the marker and mark results stale; `Enter` runs
  search, IK, collision filtering, and rank-zero selection once.
- `/` and `]` cycle forward, `[` cycles backward, and `Enter` forces refresh.
- Complete shaped skeletons are collision-tested against table and object geometry.
- Keep the viewer free of mesh, terrain, diffusion, controller, and capture code.

---

### Task 1: Grasp-anchored retrieval and complete Lift extraction

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- `HandTrajectoryQuery` retains an always-present `quat grasp_world_rotation` and adds `bool constrain_grasp_orientation = true`.
- `hand_trajectory_scene_alignment(const HandTrajectory&, const HandTrajectoryQuery&)` aligns recorded Contact wrist yaw/XZ to the requested grasp.
- `HandTrajectory::lift_frame` is the final contiguous Lift frame.

- [ ] **Step 1: Write failing retrieval and duration tests**

Add tests that construct two clips with identical recorded Contact grasps but different object dimensions/orientations. Assert both receive identical rank/cost for the same requested grasp. Add clips with different recorded Contact wrist pitch/roll and assert rotating `grasp_world_rotation` reranks them. Assert scene alignment maps the source Contact wrist X/Z and heading to the requested grasp while mapping world up to world up.

Extend the fixture to contain multiple Lift frames and assert:

```cpp
require(selected[0].lift_frame == expected_last_lift,
        "trajectory stopped at the first Lift frame");
require(selected[0].hands_in_source_object.size() ==
            static_cast<size_t>(expected_last_lift - expected_reach + 1),
        "trajectory omitted complete Lift samples");
```

- [ ] **Step 2: Verify RED**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
```

Expected: compile failure for the new query contract or assertion failure because object dimensions still gate search and only the first Lift frame is extracted.

- [ ] **Step 3: Implement grasp anchoring and final-Lift extraction**

Replace object-based alignment with a helper whose inputs are recorded Contact wrist and requested grasp:

```cpp
Transform upright_grasp_alignment(
    const Transform& source_contact,
    const HandTrajectoryQuery& query) {
    const float yaw = shortest_angle(
        yaw_radians(query.grasp_world_rotation) -
        yaw_radians(source_contact.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated_source = quat_mul_vec3(
        rotation, source_contact.position);
    return {
        vec3(query.grasp_world_position.x - rotated_source.x,
             0.0F,
             query.grasp_world_position.z - rotated_source.z),
        rotation,
    };
}
```

Build each candidate first, reconstruct its recorded Contact wrist, and use this alignment for scoring and mapping. Remove object-dimension gating/cost. Score remaining Contact position and, when `constrain_grasp_orientation` is true, full quaternion orientation residual.

Track both first and last Lift while scanning the clip. Require Reach < Contact < first Lift and set `lift_frame` to the last contiguous Lift frame beginning at first Lift. Extract every frame through that endpoint.

Update IK shaping to always use `query.grasp_world_rotation` for scene heading and only apply orientation residual when `constrain_grasp_orientation` is true.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git diff --check
```

Expected: all exit 0.

Commit:

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "fix: anchor motion retrieval on world grasp"
```

### Task 2: Complete shaped-skeleton collision filtering

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Produces `evaluate_shaped_trajectory_feasibility(const ShapedHandTrajectory&, size_t, Hand, const OrientedBox&, const ShelfGeometry&, const TrajectoryCollisionConfig&)`.
- `TrajectoryCollisionConfig` adds conservative joint, limb, and torso radii while retaining wrist/forearm defaults.

- [ ] **Step 1: Write failing full-skeleton collision tests**

Create shaped poses whose wrist/forearm remain clear while an upper-arm capsule crosses the tabletop; expect `ShelfCollision`. Add a torso/table penetration case with the same expectation. Add an object case proving a non-active body segment remains rejected after Contact, while the active wrist/forearm may overlap the object at and after Contact.

Use real `Pose` and `world_pose` values; do not mock collision calls.

- [ ] **Step 2: Verify RED**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
```

Expected: missing `evaluate_shaped_trajectory_feasibility` or the upper-arm/torso penetration is accepted.

- [ ] **Step 3: Implement pose collision traversal**

For each shaped pose, compute `WorldPose`. Test a sphere at every non-Simulation joint and a capsule for every valid parent-child edge. Choose radius by bone group: torso/head uses `torso_radius_m`, ordinary limbs use `limb_radius_m`, and active elbow-to-wrist chain uses existing forearm/wrist radii.

For table boxes, test every segment at every sample with no exemptions. For the target object, test every segment before Contact. At and after Contact, exempt only active elbow-to-wrist-chain geometry; continue testing upper arm, torso, opposite arm, and legs. Return the first failing sample and preserve existing reason values.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git diff --check
```

Expected: all exit 0.

Commit:

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "fix: reject full-body pickup collisions"
```

### Task 3: Live rank-zero viewer and complete controls

**Files:**
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`

**Interfaces:**
- `rebuild_valid_trajectories` calls grasp search, IK shaping, and shaped-skeleton collision filtering.
- Object/grasp changes mark results stale; `KEY_ENTER` replaces the result set and selects rank zero.

- [ ] **Step 1: Write failing viewer contract tests**

Require `KEY_SLASH`, `KEY_ENTER`, and `evaluate_shaped_trajectory_feasibility(`. Assert the grasp-change branch updates the query and sets `search_stale = true` without calling `rebuild_valid_trajectories(`. Assert the `KEY_ENTER` branch calls the rebuild, assigns `selected_index = 0U`, and clears stale state. Assert no previous-clip preservation code remains:

```python
self.assertNotIn("previous_clip", source)
self.assertNotIn("preserved", source)
```

Require `constrain_grasp_orientation` in query construction and retain the zero-valid guard.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.python.test_hand_trajectory_viewer
```

Expected: failures for missing slash/Enter bindings, old collision API, and previous-clip preservation.

- [ ] **Step 3: Implement viewer rebuild behavior**

Construct an always-present grasp rotation and set `constrain_grasp_orientation = !position_only`. In `rebuild_valid_trajectories`, collision-test `shaped` poses through `evaluate_shaped_trajectory_feasibility`.

Handle controls as:

```cpp
if (IsKeyPressed(KEY_SLASH) || IsKeyPressed(KEY_RIGHT_BRACKET)) {
    if (!trajectories.valid.empty()) {
        selected_index = (selected_index + 1U) % trajectories.valid.size();
        animation_seconds = 0.0F;
    }
}
if (IsKeyPressed(KEY_ENTER)) object_changed = true;
```

After any grasp transform, update `query` and set `search_stale = true` without rebuilding. When `KEY_ENTER` is pressed, rebuild the complete result set, assign `selected_index = 0U`, and set `search_stale = false`; do not preserve the prior clip. Display `SEARCH STALE - press Enter` while stale. Update help text to show `/ or ]` and `Enter: run search`. Continue to avoid indexing when `valid.empty()`.

- [ ] **Step 4: Verify GREEN, build, and commit**

Run:

```bash
python3 -m unittest tests.python.test_hand_trajectory_viewer
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make -j2 hand_trajectory_viewer
git diff --check
```

Expected: all exit 0.

Commit:

```bash
git add hand_trajectory_viewer.cpp tests/python/test_hand_trajectory_viewer.py
git commit -m "fix: rerun grasp search and cycle ranked options"
```

### Task 4: Replace and verify the live viewer

**Files:**
- No source changes expected.

**Interfaces:**
- Runs exactly one freshly built `./hand_trajectory_viewer` on `DISPLAY=:1` using the existing full interaction pack.

- [ ] **Step 1: Verify the release candidate**

Run focused tests, `git diff --check`, and confirm a clean worktree. Verify the new executable hash differs from the currently running old executable before replacement.

- [ ] **Step 2: Replace only the exact viewer process**

Resolve the exact PID whose command is `^./hand_trajectory_viewer$`, stop that PID normally, and launch the new binary with:

```bash
DISPLAY=:1 MM_INTERACTION_PACK=/home/ubuntu/worktrees/motion-matching/g1-tabletop-placement/build/smart-pickup/full-pack ./hand_trajectory_viewer
```

Do not use screenshot, display-capture, mesh, terrain, controller, or diffusion processes.

- [ ] **Step 3: Verify runtime state**

Confirm exactly one viewer process, matching disk/running binary hashes, the expected DISPLAY/pack environment, no viewer runtime error output, and a clean worktree. Leave the branch/worktree available for the user's visual iteration.
