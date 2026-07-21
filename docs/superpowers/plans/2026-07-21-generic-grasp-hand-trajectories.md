# Generic Grasp Hand-Trajectories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw every recorded pickup path compatible with a generic object and requested grasp pose or position, with exact Contact convergence and motion matching as the only playback authority.

**Architecture:** A pure C++ module selects clips with explicit tolerance gates and stores wrist poses in source-object coordinates. It maps each path into a live query by object alignment plus a Contact residual. The controller renders all selected paths; a launcher removes every diffusion authority variable.

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
};
```

Find first Reach, Contact, and Lift. Compare source grasp metadata to the query
in target-object coordinates. Gate before scoring. Store wrist transforms from
Reach through Lift using `compose(inverse(source_object), source_hand_world)`.

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

### Task 2: Live all-path controller overlay

**Files:**
- Modify: `controller.cpp`
- Create: `tests/python/test_generic_hand_trajectory_controller.py`

**Interfaces:**
- Consumes Task 1 selection/mapping.
- Produces `MM_INTERACTION_HAND_TRAJECTORIES=1` and `MM_G1_SKELETON_ONLY=1` gates.

- [ ] **Step 1: Write failing source-contract tests**

Assert the controller includes the new header, parses both variables, builds a
query from the current target and affordance, selects without top-K, maps using
the current target each frame, draws every polyline and Contact marker, and
skips mesh load/update/draw/unload in skeleton-only mode.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. python -m unittest \
  tests.python.test_generic_hand_trajectory_controller -v
```

Expected: failures for missing gates and calls.

- [ ] **Step 3: Implement setup and selection**

Accept only `MM_INTERACTION_HAND_TRAJECTORIES=1`. After target registration,
form the full-pose query from
`compose(target.object_world, affordance.hand_in_object)` and select once. Treat
zero candidates and over-limit selection as controlled startup errors.

- [ ] **Step 4: Render live paths**

At render time, rebuild the query object/grasp transforms from the current
target, map every selected path, draw all segments in a rank gradient, mark
Contact spheres, emphasize rank zero in green, and display candidate count.

- [ ] **Step 5: Gate skeleton-only rendering**

With `MM_G1_SKELETON_ONLY=1`, set `show_g1_mesh=false` and skip mesh renderer
load/update/draw/unload while preserving `show_g1_bones=true`.

- [ ] **Step 6: Verify and commit**

```bash
./build/tests/test_interaction_hand_trajectories
PYTHONPATH=. python -m unittest \
  tests.python.test_generic_hand_trajectory_controller \
  tests.python.test_native_g1_diffusion_controller -v
make -j2 controller
git add controller.cpp tests/python/test_generic_hand_trajectory_controller.py
git commit -m "feat: render generic grasp trajectory field"
```

Expected: tests and build exit 0 before commit.

### Task 3: Motion-matching-only launcher and live audit

**Files:**
- Create: `tools/run_g1_motion_matching_pickup_baseline.sh`
- Create: `tests/python/test_motion_matching_pickup_baseline_launcher.py`

**Interfaces:**
- Produces one flat skeleton controller with trajectory visualization enabled and diffusion disabled.

- [ ] **Step 1: Write failing launcher test**

Require trajectory, flat-terrain, skeleton-only, pack, and feature variables.
Require `env -u` for `MM_G1_OFFLINE_OVERLAP`,
`MM_INTERACTION_FUNNEL_WORKER`, and `MM_INTERACTION_FUNNEL_CHECKPOINT`. Reject
screenshot/capture commands.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. python -m unittest \
  tests.python.test_motion_matching_pickup_baseline_launcher -v
```

Expected: failure because the launcher is absent.

- [ ] **Step 3: Implement launcher**

Resolve repository root and `exec env`:

```bash
env -u MM_G1_OFFLINE_OVERLAP \
  -u MM_INTERACTION_FUNNEL_WORKER \
  -u MM_INTERACTION_FUNNEL_CHECKPOINT \
  MM_INTERACTION_HAND_TRAJECTORIES=1 \
  MM_INTERACTION_FLAT_TERRAIN=1 \
  MM_G1_SKELETON_ONLY=1 \
  MM_INTERACTION_PACK="$pack" \
  MM_FEATURES_OUTPUT="$features" \
  "$root/controller"
```

- [ ] **Step 4: Final verification**

```bash
./build/tests/test_interaction_hand_trajectories
PYTHONPATH=. python -m unittest \
  tests.python.test_generic_hand_trajectory_controller \
  tests.python.test_motion_matching_pickup_baseline_launcher \
  tests.python.test_native_g1_diffusion_controller -v
git diff --check
```

Expected: all tests pass and whitespace check exits 0.

- [ ] **Step 5: Launch and audit**

Stop only the exact old diffusion-preview process. Launch baseline on
`DISPLAY=:1`; after eight seconds verify one controller, all baseline variables,
no diffusion variables, no mesh-load lines, and no runtime error.

- [ ] **Step 6: Commit**

```bash
git add tools/run_g1_motion_matching_pickup_baseline.sh \
  tests/python/test_motion_matching_pickup_baseline_launcher.py
git commit -m "feat: launch motion matching pickup baseline"
```
