# Tiered Grasp Coverage Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable exact-first grasp search with axis-constrained fallback, front-side filtering, and a right shelf plus lower left table in the standalone skeleton viewer.

**Architecture:** Put orientation metrics, mapped-root filtering, and variable collision geometry in the tested trajectory library. Keep only exact-then-fallback orchestration, controls, drawing, and diagnostic copy in the viewer.

**Tech Stack:** C++17, existing G1 pose/IK and compact trajectory database, raylib, Python `unittest`, GNU Make.

## Global Constraints

- Enter is the only search trigger; object edits only mark results stale.
- Exact matches precede fallback matches.
- Run fallback only when fewer than 12 exact valid trajectories survive.
- Keep the 25-degree retrieval and 15-degree final orientation limits.
- Do not rebuild the existing 2,510-clip pack.
- Scene front is `(0, 0, -1)`; zero-dot side approaches are accepted.
- Ground-only packs omit table-derived furniture.
- Run only the skeleton viewer, never mesh or terrain renderers.

## File Map

- `interaction_hand_trajectories.h`: retrieval modes, match tier, mapped root, environment API.
- `interaction_hand_trajectories.cpp`: matching, shaping, side filter, furniture, collision.
- `tests/cpp/test_interaction_hand_trajectories.cpp`: library behavior.
- `hand_trajectory_viewer.cpp`: tier orchestration and visualization.
- `tests/python/test_hand_trajectory_viewer.py`: viewer integration contract.

---

### Task 1: Exact and approach-axis retrieval

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `Database::approach_directions_object` and existing upright alignment.
- Produces: `GraspOrientationMode`, `TrajectoryMatchTier`, query world approach direction, and candidate source approach direction.

- [ ] **Step 1: Write failing tests**

Add `vec3 approach_direction{1.0F, 0.0F, 0.0F};` to `ClipSpec` and write it to `database.approach_directions_object`. Add and register tests with these assertions:

```cpp
ClipSpec exact{};
exact.grasp_position = vec3(0.10F, 0.20F, 0.30F);
ClipSpec twist = exact;
twist.grasp_rotation = quat_from_angle_axis(
    1.570796327F, vec3(1.0F, 0.0F, 0.0F));
ClipSpec wrong_axis = exact;
wrong_axis.approach_direction = vec3(0.0F, 0.0F, 1.0F);
auto query = identity_query();
query.orientation_mode = interaction::GraspOrientationMode::ApproachAxis;
query.approach_world_direction = vec3(1.0F, 0.0F, 0.0F);
const auto axis = interaction::select_hand_trajectories(
    make_database({exact, twist, wrong_axis}), query);
require(axis.size() == 2U, "axis search did not isolate twist");
require(axis[0].clip == 0 && axis[1].clip == 1, "axis order changed");
require(axis[0].match_tier ==
            interaction::TrajectoryMatchTier::AxisFallback,
        "axis tier missing");

query.orientation_mode = interaction::GraspOrientationMode::ExactPose;
const auto selected_exact = interaction::select_hand_trajectories(
    make_database({exact, twist}), query);
require(selected_exact.size() == 1U && selected_exact[0].clip == 0,
        "exact search ignored wrist twist");
```

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compilation fails because the mode and tier types are absent.

- [ ] **Step 3: Implement the minimal public types and metric**

Replace `constrain_grasp_orientation` with:

```cpp
enum class GraspOrientationMode : uint8_t {
    ExactPose = 0U,
    ApproachAxis = 1U,
    PositionOnly = 2U,
};
enum class TrajectoryMatchTier : uint8_t {
    Exact = 0U,
    AxisFallback = 1U,
    PositionOnly = 2U,
};
```

Add normalized `approach_world_direction` to the query and `source_approach_direction_object` plus `match_tier` to each trajectory. For exact mode use quaternion angle. For fallback map the stored candidate direction through the world alignment and use:

```cpp
float direction_angle(vec3 a, vec3 b) {
    a = normalize(a);
    b = normalize(b);
    return std::acos(std::clamp(dot(a, b), -1.0F, 1.0F));
}
```

Position-only mode uses zero orientation cost. Validate finite, horizontal, nonzero approach directions. Preserve `(cost, clip)` sorting.

- [ ] **Step 4: Verify GREEN and commit**

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: add axis-constrained grasp retrieval"
```

Expected: all trajectory tests pass.

### Task 2: Axis-aware shaping and front-side filtering

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: Task 1 modes and candidate approach direction.
- Produces: `HandTrajectory::start_root_in_source_object` and `starts_on_allowed_side(...)`.

- [ ] **Step 1: Write failing tests**

Add `vec3 start_root{};` to `ClipSpec` and write it to the simulation-root position at every clip frame. Add and register these tests:

```cpp
ClipSpec twist{};
twist.grasp_position = vec3(0.10F, 0.20F, 0.30F);
twist.grasp_rotation = quat_from_angle_axis(
    1.570796327F, vec3(1.0F, 0.0F, 0.0F));
auto axis_query = identity_query();
axis_query.orientation_mode = interaction::GraspOrientationMode::ApproachAxis;
axis_query.approach_world_direction = vec3(1.0F, 0.0F, 0.0F);
auto axis_database = make_database({twist});
const auto axis_selected = interaction::select_hand_trajectories(
    axis_database, axis_query);
require(interaction::shape_hand_trajectory(
            axis_database, axis_selected.front(), axis_query).contact_accepted,
        "fallback still required exact wrist twist");

const vec3 front(0.0F, 0.0F, -1.0F);
ClipSpec front_spec{};
front_spec.grasp_position = vec3(0.10F, 0.20F, 0.30F);
front_spec.start_root = vec3(0.10F, 0.0F, -1.0F);
ClipSpec side_spec = front_spec;
side_spec.start_root = vec3(1.0F, 0.0F, 0.30F);
ClipSpec rear_spec = front_spec;
rear_spec.start_root = vec3(0.10F, 0.0F, 1.0F);
const auto sides = interaction::select_hand_trajectories(
    make_database({front_spec, side_spec, rear_spec}), identity_query());
require(interaction::starts_on_allowed_side(
            sides[0], identity_query(), front),
        "front start rejected");
require(interaction::starts_on_allowed_side(
            sides[1], identity_query(), front),
        "side start rejected");
require(!interaction::starts_on_allowed_side(
            sides[2], identity_query(), front),
        "rear start accepted");
```

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compilation fails because root-side filtering is absent.

- [ ] **Step 3: Store and filter the mapped start root**

During selection, store the clip-start simulation root in source-object space. Implement `starts_on_allowed_side(trajectory, query, scene_front, minimum_dot = 0.0F)` by mapping the root with `hand_trajectory_scene_alignment`, flattening `mapped_root - grasp_world_position`, and accepting a normalized dot greater than or equal to the threshold. Accept horizontal distance below `1e-6 m`; reject invalid front vectors or thresholds.

- [ ] **Step 4: Implement axis-mode final acceptance**

At contact derive the candidate axis in the base wrist frame:

```cpp
const vec3 axis_in_hand = quat_mul_vec3(
    quat_inv(base_contact.rotation), mapped_candidate_axis);
```

For `ApproachAxis`, run positional IK with quaternion residuals disabled. Reconstruct the solved axis as `quat_mul_vec3(solved_hand.rotation, axis_in_hand)` and accept only if position error is at most `accepted_position_m` and axis error is at most `accepted_orientation_radians`. Use `Reason::CorrectionLimit` for an axis miss. Leave exact and position-only acceptance unchanged.

- [ ] **Step 5: Verify GREEN and commit**

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: validate fallback axes and approach sides"
```

### Task 3: Variable environment and coverage furniture

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `OrientedBox`, recorded table transform/dimensions, existing collision tests.
- Produces: `EnvironmentGeometry`, `make_recorded_table_geometry`, and `make_coverage_environment`.

- [ ] **Step 1: Write failing tests**

Add and register checks for a 13-box environment and collision with a non-first box:

```cpp
const auto environment = interaction::make_coverage_environment(
    {vec3(0.0F, 0.76F, 0.0F), quat()},
    vec3(1.20F, 0.06F, 0.70F));
require(environment.boxes.size() == 13U, "wrong furniture box count");
require(near(environment.boxes[5].dimensions,
             vec3(0.45F, 0.04F, 0.32F)), "wrong shelf board");
require(near(environment.boxes[8].dimensions,
             vec3(0.65F, 0.06F, 0.50F)), "wrong lower table top");
```

Build a two-box environment with the second box intersecting the test skeleton and require `EnvironmentCollision`. Add invalid table-dimension coverage.

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compilation fails because variable environment geometry is absent.

- [ ] **Step 3: Implement variable geometry**

Replace fixed `ShelfGeometry` with:

```cpp
struct EnvironmentGeometry {
    std::vector<OrientedBox> boxes;
};
```

Keep five center-table boxes. Append a `0.04 m` thick shelf board `0.32 m` above the right table third plus two supports. Clamp shelf width/depth to table bounds. Append a `0.65 x 0.06 x 0.50 m` lower top `0.24 m` below and `0.12 m` left of the center table plus four legs. Rename `ShelfCollision` to `EnvironmentCollision`; check every vector box. Empty geometry is valid for ground-only packs.

- [ ] **Step 4: Verify GREEN and commit**

```bash
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: add grasp coverage furniture geometry"
```

### Task 4: Viewer orchestration and diagnostics

**Files:**
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`

**Interfaces:**
- Consumes: all Task 1-3 APIs.
- Produces: Enter-triggered tiered search, furniture rendering, and classified HUD.

- [ ] **Step 1: Write failing viewer tests**

Assert the source contains `kTargetValidTrajectories = 12U`, both orientation modes, `starts_on_allowed_side`, `wrong_side`, `EnvironmentGeometry`, `make_coverage_environment`, `"EXACT"`, and `"AXIS-FALLBACK"`. Preserve the current assertion that the `grasp_changed` block marks stale without calling `rebuild_valid_trajectories`.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. python3 -m unittest tests.python.test_hand_trajectory_viewer -v`.

Expected: tier and environment assertions fail.

- [ ] **Step 3: Implement exact-first fallback fill**

Add `constexpr size_t kTargetValidTrajectories = 12U`. Track separate exact/fallback compatible and valid counts plus wrong-side, IK, object, and environment rejections. Process exact candidates first. Before shaping, reject rear starts. If fewer than 12 exact valid paths survive, query `ApproachAxis`, skip clip IDs already processed, and append valid fallback paths until 12 total paths exist or candidates end. Preserve P as explicit position-only diagnostic mode without automatic fallback.

- [ ] **Step 4: Render and label the same collision environment**

For table/mixed packs use `make_coverage_environment`; for ground-only packs use an empty environment. Draw every `environment.boxes` entry passed to collision. Expand the HUD and label each selected option by tier. Preserve complete start-through-lift animation and Enter-only recomputation.

- [ ] **Step 5: Verify GREEN and commit**

```bash
PYTHONPATH=. python3 -m unittest tests.python.test_hand_trajectory_viewer -v
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make -j2 hand_trajectory_viewer
git diff --check
git add hand_trajectory_viewer.cpp tests/python/test_hand_trajectory_viewer.py
git commit -m "feat: visualize tiered grasp coverage"
```

### Task 5: Regression and bounded live handoff

**Files:**
- Modify only files already listed if verification exposes a defect.

**Interfaces:**
- Consumes: completed viewer and existing mixed pack.
- Produces: passing evidence and exactly one live viewer.

- [ ] **Step 1: Run full verification**

```bash
PYTHONPATH=. python3 -m unittest discover -s tests/python -v
make -B build/tests/test_interaction_trajectory_database
./build/tests/test_interaction_trajectory_database
make -B build/tests/test_interaction_hand_trajectories
./build/tests/test_interaction_hand_trajectories
make -j2 hand_trajectory_viewer
git diff --check
git status --short
```

Expected: every test passes, viewer builds, diff check is clean, and no source changes remain.

- [ ] **Step 2: Replace only the current viewer**

Resolve the exact PID matching `^./hand_trajectory_viewer$`, send SIGTERM only to that PID, and wait for exit. Do not touch the controller. Launch:

```bash
DISPLAY=:1 MM_INTERACTION_PACK=/home/ubuntu/projects/motion-matching/.worktrees/g1-motion-matching-pickup-baseline-20260721/build/smart-pickup/table-ground-pack ./hand_trajectory_viewer
```

- [ ] **Step 3: Verify live identity and memory**

Confirm one viewer process; compare `sha256sum hand_trajectory_viewer` with `/proc/<pid>/exe`; verify `DISPLAY` and `MM_INTERACTION_PACK` in `/proc/<pid>/environ`; sample RSS for at least 10 seconds. Require the process to stay alive below `1.1 GiB` without monotonic growth. Do not use screenshots.

- [ ] **Step 4: Record evidence**

Report commits, test counts, PID, binary hash, environment, and peak RSS. Do not claim completion unless every preceding check passed.
