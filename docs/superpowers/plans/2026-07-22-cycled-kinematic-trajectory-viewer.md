# Cycled Kinematic Trajectory Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the oversized cabinet with the recorded table fixture, update paths continuously under ordinary controls, and animate a cycled full G1 skeleton for every compatible kinematic option.

**Architecture:** Extend the pure trajectory module with the exact rigid source-world mapping already implicit in wrist mapping and a recorded-table five-box builder. The standalone raylib viewer consumes those interfaces, keeps all candidate paths visible, and maps one database skeleton through the selected candidate at 25 FPS.

**Tech Stack:** C++17, raylib, existing interaction database/pose APIs, Python `unittest` source contracts, Make.

## Global Constraints

- Keep the viewer independent of the controller, diffusion runtime, robot mesh, and terrain renderer.
- Preserve all compatible candidates; do not introduce a top-K cutoff below the existing 4096 safety limit.
- Cycle safe and rejected candidates; rejected selections remain visible and red.
- Use the recorded table transform and dimensions; do not author another cabinet scale.
- Recompute mapped paths and collision status every frame while movement keys are held.
- Draw the G1 skeleton directly from the 31-joint hierarchy; do not load a model.
- Do not use screenshots or DCV/X11 capture tools.

---

### Task 1: Exact source-world mapping interface

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `HandTrajectory`, `HandTrajectoryQuery`, and the existing exact Contact residual.
- Produces: `Transform hand_trajectory_world_mapping(const HandTrajectory&, const HandTrajectoryQuery&)`.

- [ ] **Step 1: Write the failing mapping test**

Add a test that reconstructs each selected source wrist in world coordinates and proves the new mapping produces the same positions and rotations as `map_hand_trajectory`:

```cpp
const Transform mapping = interaction::hand_trajectory_world_mapping(
    selected[0], moved);
for (size_t sample = 0; sample < mapped.hands.size(); ++sample) {
    const Transform source_hand = compose(
        selected[0].source_object,
        selected[0].hands_in_source_object[sample]);
    const Transform via_mapping = compose(mapping, source_hand);
    require(near(via_mapping.position, mapped.hands[sample].position),
            "world mapping disagrees with mapped wrist");
    require(near_rotation(via_mapping.rotation, mapped.hands[sample].rotation),
            "world mapping rotation disagrees with mapped wrist");
}
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
```

Expected: compilation fails because `hand_trajectory_world_mapping` is absent.

- [ ] **Step 3: Implement one rigid mapping authority**

Declare the function in the header. In the implementation, build:

```cpp
const Transform source_world_to_target_object = compose(
    normalized_target_object, inverse(trajectory.source_object));
const Transform mapped_contact = compose(
    source_world_to_target_object,
    compose(trajectory.source_object,
            trajectory.hands_in_source_object[trajectory.contact_point]));
const Transform residual = query.grasp_world_rotation.has_value()
    ? compose(requested_grasp, inverse(mapped_contact))
    : Transform{query.grasp_world_position - mapped_contact.position, quat()};
return compose(residual, source_world_to_target_object);
```

Refactor `map_hand_trajectory` to apply this returned mapping to source-world wrist and elbow samples. Keep validation and exact full-pose/position-only behavior unchanged.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git diff --check
```

Expected: compile and test exit 0.

Commit:

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp \
  tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: expose exact trajectory world mapping"
```

### Task 2: Recorded table collision fixture

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: recorded `Transform table_world` and positive `vec3 table_dimensions`.
- Produces: `ShelfGeometry make_recorded_table_geometry(const Transform&, vec3, float leg_thickness_m = 0.04F)` containing tabletop plus four legs.

- [ ] **Step 1: Write failing geometry tests**

Assert that box 0 exactly preserves the recorded tabletop and boxes 1-4 are slender positive legs below its four corners:

```cpp
const Transform table{{1.0F, 0.40F, -2.0F}, quat()};
const vec3 dimensions{1.20F, 0.08F, 0.70F};
const auto geometry = interaction::make_recorded_table_geometry(
    table, dimensions);
require(near(geometry.boxes[0].world.position, table.position),
        "tabletop position changed");
require(near(geometry.boxes[0].dimensions, dimensions),
        "tabletop dimensions changed");
for (size_t leg = 1U; leg < geometry.boxes.size(); ++leg) {
    require(geometry.boxes[leg].dimensions.y > 0.0F &&
            geometry.boxes[leg].world.position.y < table.position.y,
            "table leg is not below tabletop");
}
```

Also assert an invalid tabletop at or below ground throws `std::invalid_argument`.

- [ ] **Step 2: Run and verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compilation fails because `make_recorded_table_geometry` is absent.

- [ ] **Step 3: Implement the five boxes**

Use box 0 for the exact recorded tabletop. Compute leg height from the underside to world ground, inset each 0.04 m square leg from the X/Z edges, and construct leg centers through `compose(table_world, local_leg_transform)`. Reject non-finite, non-positive, or below-ground geometry.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused C++ test and `git diff --check`; expect exit 0.

Commit:

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp \
  tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: build collision geometry from recorded table"
```

### Task 3: Live cycled skeleton viewer

**Files:**
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`

**Interfaces:**
- Consumes: Tasks 1-2, database `pose_at_frame`/`world_pose`, candidate Reach/Lift bounds.
- Produces: W/S height controls, bracket cycling, continuous path classification, highlighted selected path, and selected G1 skeleton animation.

- [ ] **Step 1: Extend source-contract tests and verify RED**

Require these exact behaviors:

```python
self.assertIn("IsKeyDown(KEY_W)", source)
self.assertIn("IsKeyDown(KEY_S)", source)
self.assertIn("KEY_LEFT_BRACKET", source)
self.assertIn("KEY_RIGHT_BRACKET", source)
self.assertIn("make_recorded_table_geometry(", source)
self.assertIn("hand_trajectory_world_mapping(", source)
self.assertIn("pose_at_frame(", source)
self.assertIn("world_pose(", source)
self.assertIn("g1_skeleton::kParents", source)
self.assertIn("DrawCylinderEx(", source)
self.assertNotIn("make_shelf(", source)
```

Run `PYTHONPATH=. python -m unittest tests.python.test_hand_trajectory_viewer -v`.

Expected: failures for missing controls, cycling, table builder, and skeleton rendering.

- [ ] **Step 2: Load and recenter recorded scene geometry**

Extend `CanonicalGrasp` with `source_object` and `table_world/table_dimensions`. Read them from the selected clip. Apply only `(-table_world.position.x, 0, -table_world.position.z)` to both initial object and table so source feet retain authored height. Replace `make_shelf()` with `make_recorded_table_geometry(...)`.

- [ ] **Step 3: Add continuous controls and cycling**

Map W/S and Page Up/Page Down to Y translation. On bracket presses, wrap `selected_index` over `candidates.size()` without filtering rejected candidates and restart animation time. While any translation/rotation key is down, rebuild the query and classify every candidate in that same frame.

- [ ] **Step 4: Animate the selected full skeleton**

Advance `animation_seconds += dt`, loop over inclusive Reach-to-Lift samples, reconstruct `WorldPose source = world_pose(pose_at_frame(database, frame))`, and map every joint with `hand_trajectory_world_mapping(candidates[selected_index], query)`. Draw each joint and its parent edge. Use red for a rejected selected candidate and dark/sky blue for accepted.

- [ ] **Step 5: Highlight and identify selection**

Draw all paths first, then redraw the selected path with an emphasized color and larger sample/contact marks. Add UI text containing option index/count, clip index, collision status, animation phase, W/S controls, and bracket controls. Rename shelf-rejected UI text to table-rejected.

- [ ] **Step 6: Verify and commit**

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
PYTHONPATH=. python -m unittest tests.python.test_hand_trajectory_viewer -v
make -j2 hand_trajectory_viewer
git diff --check
```

Expected: focused tests and build exit 0.

Commit:

```bash
git add hand_trajectory_viewer.cpp tests/python/test_hand_trajectory_viewer.py
git commit -m "feat: cycle animated pickup kinematics"
```

### Task 4: One-process live handoff

**Files:**
- Runtime only; no source changes expected.

**Interfaces:**
- Consumes: freshly built `hand_trajectory_viewer` and existing full interaction pack.
- Produces: exactly one live viewer process with no diffusion environment.

- [ ] **Step 1: Stop only the exact previous viewer**

Resolve `pgrep -f '^./hand_trajectory_viewer$'` and terminate only that PID. Do not run screenshot or capture commands.

- [ ] **Step 2: Launch the replacement**

Launch on `DISPLAY=:1` with the existing full-pack path and redirect logs to `build/hand-trajectory-viewer.log`.

- [ ] **Step 3: Audit**

Verify one viewer remains alive, zero `controller.offline-overlap-demo` processes exist, no diffusion variables occur in `/proc/<pid>/environ`, the log has no error/fatal/failed lines, and the worktree is clean.

