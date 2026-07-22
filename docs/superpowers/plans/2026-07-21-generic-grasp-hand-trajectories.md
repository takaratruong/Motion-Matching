# Generic Grasp Hand-Trajectories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw every recorded pickup path compatible with a generic object and requested grasp pose or position, with exact Contact convergence and motion matching as the only playback authority.

**Architecture:** A pure C++ module selects clips with explicit tolerance gates and stores wrist/elbow samples in source-object coordinates. It maps each path into a live query by object alignment plus a Contact residual, then filters object and shelf collisions. A separate lightweight raylib viewer renders accepted and rejected paths without the controller, diffusion runtime, robot mesh, or terrain renderer.

**Tech Stack:** C++17, existing G1 interaction database and FK, raylib, Make, Python `unittest`, Bash.

## Global Constraints

- Never use screenshots, X11 capture, or DCV capture.
- Flat terrain and blue G1 skeleton only; skip robot and terrain mesh loads.
- Exactly one controller process.
- No top-K truncation: draw every clip passing tolerances, with a 4096-clip safety rejection.
- Diffusion may later stitch windows but cannot select or execute pickup motion in this baseline.

---

### Task 1: Generic query, selection, and exact convergence

**Files:**
- Create: `interaction_hand_trajectories.h`
- Create: `interaction_hand_trajectories.cpp`
- Create: `tests/cpp/test_interaction_hand_trajectories.cpp`
- Modify: `Makefile`

**Interfaces:**
- Produces `HandTrajectoryQuery`, `HandTrajectoryConfig`, `HandTrajectory`, `select_hand_trajectories(...)`, and `map_hand_trajectory(...)`.
- Consumes `Database`, `pose_at_frame`, `world_pose`, and G1 wrist indices.

- [ ] **Step 1: Write the failing C++ test**

Create right/left clips, malformed phase clips, close/far grasp clips, and tied
costs. Exercise:

```cpp
HandTrajectoryQuery query{};
query.object_world = target.object_world;
query.object_dimensions = target.object_dimensions;
query.hand = Hand::Right;
const Transform grasp = compose(
    target.object_world, target.affordances.front().hand_in_object);
query.grasp_world_position = grasp.position;
query.grasp_world_rotation = grasp.rotation;
const auto selected = select_hand_trajectories(
    database, query, HandTrajectoryConfig{});
```

Assert same-hand and phase filtering, position/orientation/dimension gates,
`(cost, clip)` ordering, every passing clip retained, and the 4096 limit. Assert
`map_hand_trajectory` makes Contact equal the requested full pose. Clear
`grasp_world_rotation` and assert exact position while preserving mapped source
Contact rotation.

- [ ] **Step 2: Verify red**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compile failure because the new header does not exist.

- [ ] **Step 3: Implement selector types and logic**

Define:

```cpp
struct HandTrajectoryQuery {
    Transform object_world{};
    vec3 object_dimensions{};
    Hand hand = Hand::Right;
    vec3 grasp_world_position{};
    std::optional<quat> grasp_world_rotation;
};

struct HandTrajectory {
    int32_t clip = -1;
    int32_t reach_frame = -1;
    int32_t contact_frame = -1;
    int32_t lift_frame = -1;
    size_t contact_point = 0U;
    float cost = 0.0F;
    Transform source_object{};
    std::vector<Transform> hands_in_source_object;
    std::vector<vec3> elbows_in_source_object;
};
```

Find first Reach, Contact, and Lift. Compare source grasp metadata to the query
in target-object coordinates. Gate before scoring. Store wrist transforms and
elbow points from Reach through Lift in source-object coordinates.

- [ ] **Step 4: Implement exact mapping**

Map stored poses through `query.object_world`. For pose queries, apply
`requested_contact * inverse(mapped_contact)` to every pose. For position-only
queries, translate every position by the Contact error and preserve rotations.

- [ ] **Step 5: Wire Make and verify green**

Add the source to `INTERACTION_NATIVE_G1_SOURCES`, add the test to
`CPP_TEST_BINS`, and link it with `interaction_pose.cpp` and
`interaction_target.cpp`.

Run:

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
```

Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add Makefile interaction_hand_trajectories.* \
  tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: select generic grasp hand trajectories"
```

### Task 2: Shelf collision feasibility

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Modify: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes Task 1 mapped wrist/elbow samples.
- Produces `ShelfGeometry`, `TrajectoryFeasibility`, and `evaluate_trajectory_feasibility(...)`.

- [ ] **Step 1: Write failing geometry tests**

Create an oriented target object and five shelf boxes. Assert a clear approach
passes, a pre-Contact wrist sphere through the object is object-rejected, a
forearm capsule through a side wall is shelf-rejected, and object overlap at
Contact is exempt while shelf overlap at Contact is rejected.

- [ ] **Step 2: Verify red**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compile failure for missing feasibility interfaces.

- [ ] **Step 3: Implement collision primitives**

Implement point/segment distance against oriented boxes by transforming samples
to each box's local frame. Expand boxes by wrist radius for sphere tests and by
forearm radius for elbow-to-wrist capsule tests.

- [ ] **Step 4: Implement phase-aware feasibility**

Test object collision only for samples before `contact_point`. Test all shelf
boxes for every sample. Return separate `ObjectCollision` and `ShelfCollision`
reasons and never silently mutate or truncate trajectories.

- [ ] **Step 5: Verify and commit**

Run the focused C++ test and expect exit 0, then commit the selector and
collision module:

```bash
./build/tests/test_interaction_hand_trajectories
git add Makefile interaction_hand_trajectories.* \
  tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: select collision-safe grasp trajectories"
```

### Task 3: Separate interactive shelf trajectory viewer

**Files:**
- Create: `hand_trajectory_viewer.cpp`
- Create: `tests/python/test_hand_trajectory_viewer.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes Tasks 1-2 and an interaction pack path.
- Produces standalone `hand_trajectory_viewer` with shelf/object controls and accepted/rejected path rendering.

- [ ] **Step 1: Write failing viewer source-contract tests**

Assert the standalone source handles `Q/E`, `R/F`, `Z/C`, arrow keys,
Page Up/Page Down, `V`, and Backspace. Assert five shelf boxes, accepted and
rejected counters, exact-query rebuilding, collision evaluation, and no
controller/diffusion/mesh/terrain includes or screenshot calls.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. python -m unittest \
  tests.python.test_hand_trajectory_viewer -v
```

Expected: failure because the viewer source is absent.

- [ ] **Step 3: Implement viewer setup and controls**

Load `interaction_database.bin`, construct the demo object's dimensions and
object-local grasp, select all close trajectories, and construct floor/back/
left/right/top shelf boxes. Apply object translation and yaw/pitch/roll at a
fixed per-frame rate; Backspace restores the initial transform.

- [ ] **Step 4: Implement rendering**

Draw shelf and object oriented boxes, accepted paths in cost colors, Contact
markers, and optionally rejected paths in translucent red. Show controls and
close/safe/object-rejected/shelf-rejected counts. Recompute mapping and
feasibility whenever the object changes.

- [ ] **Step 5: Add Make target and verify**

```bash
./build/tests/test_interaction_hand_trajectories
PYTHONPATH=. python -m unittest \
  tests.python.test_hand_trajectory_viewer -v
make -j2 hand_trajectory_viewer
git diff --check
```

Expected: tests and build exit 0.

- [ ] **Step 6: Launch and audit**

Stop only the exact old diffusion-preview process. Launch exactly one
`hand_trajectory_viewer` on `DISPLAY=:1`; after eight seconds verify it remains
alive, has no diffusion environment variables, and reports no startup error.

- [ ] **Step 7: Commit**

```bash
git add Makefile hand_trajectory_viewer.cpp \
  tests/python/test_hand_trajectory_viewer.py
git commit -m "feat: add interactive shelf trajectory lab"
```
