# Exhaustive Motion Reuse Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit every same-hand table/ground pickup clip for Contact IK, complete trajectory shaping, and object/furniture collision within 30 seconds, while displaying the best 12 reusable motions.

**Architecture:** Refactor Contact solving into one trajectory-library primitive shared by Contact-only and full shaping. Add a raylib-independent four-worker audit engine with deterministic indexed outcomes and a hard deadline, then integrate it behind the viewer's `A` key and a headless real-pack benchmark probe.

**Tech Stack:** C++17, `std::thread`, atomics, raylib, Python `unittest`, GNU Make.

## Global Constraints

- Audit all complete clips for the query's active hand; do not mirror hands.
- Use a 45 cm Contact correction envelope and 180 degree retrieval envelope.
- Keep final limits at 4 cm position, 15 degree approach axis, and 60 degree full orientation.
- Fully shape and collision-check every Contact-IK survivor.
- Use exactly four workers in the viewer and a 30,000 ms deadline.
- Publish no partial reusable total and preserve prior displayed paths on `INCOMPLETE`.
- Keep ordinary Enter search unchanged.
- Do not launch mesh or terrain renderers or signal controller PID `556617`.

---

### Task 1: Share Contact solving with full trajectory shaping

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Modify: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `Database`, `HandTrajectory`, `HandTrajectoryQuery`, `IKConfig`.
- Produces:

```cpp
struct ShapedHandContact {
    Pose pose{};
    Transform hand{};
    vec3 elbow{};
    float achieved_orientation_error_radians = 0.0F;
    bool accepted = false;
    Reason reason = Reason::None;
};

ShapedHandContact shape_hand_trajectory_contact(
    const Database& database,
    const HandTrajectory& trajectory,
    const HandTrajectoryQuery& query,
    const IKConfig& config = IKConfig{});
```

- [ ] **Step 1: Write failing Contact/full-shape parity tests**

Select an axis-fallback clip with a 30 degree wrist twist and 5 cm target-height offset. Call the new Contact function and full shaping, then assert:

```cpp
require(contact.accepted == shaped.contact_accepted,
        "Contact-only and full shaping acceptance diverged");
require(contact.reason == shaped.reason,
        "Contact-only and full shaping reasons diverged");
require(near(contact.hand.position,
             shaped.path.hands[selected[0].contact_point].position),
        "Contact-only and full shaping positions diverged");
require(near_rotation(contact.hand.rotation,
                      shaped.path.hands[selected[0].contact_point].rotation),
        "Contact-only and full shaping rotations diverged");
require(std::abs(contact.achieved_orientation_error_radians -
                 shaped.achieved_orientation_error_radians) <= 1.0e-5F,
        "Contact-only and full shaping orientation errors diverged");
```

Repeat parity for `PositionOnly`. Add a 46 cm target-height fixture using `IKConfig::maximum_request_position_m = 0.45F` and require `Reason::CorrectionLimit`, proving the widened envelope remains hard.

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_hand_trajectories`.

Expected: compilation fails because `ShapedHandContact` and `shape_hand_trajectory_contact` do not exist.

- [ ] **Step 3: Extract one shared Contact solver**

In the anonymous namespace, add a `ShapeContext` containing alignment, mapped Contact hand, Contact translation, orientation correction, mapped approach axis, and approach axis expressed in the hand frame. Add `make_shape_context(...)`, `shape_solve_config(...)`, and `solve_shaped_sample(...)`.

`shape_solve_config` preserves existing behavior:

```cpp
if (mode == GraspOrientationMode::ApproachAxis) {
    result.maximum_request_orientation_radians = pi;
    result.orientation_scale_m_per_radian = 0.10F;
    result.maximum_iterations = std::max(16, config.maximum_iterations);
} else if (mode == GraspOrientationMode::PositionOnly) {
    result.maximum_request_orientation_radians = pi;
    result.accepted_orientation_radians = pi;
    result.orientation_scale_m_per_radian = 0.0F;
}
```

At Contact, use solver acceptance for exact and position-only. For axis fallback require position `<= accepted_position_m`, approach axis `<= accepted_orientation_radians`, and full orientation `<= 1.047197551F`.

Implement `shape_hand_trajectory_contact` by loading only the recorded Contact pose, applying root alignment, constructing the weight-1 Contact target through the shared context, and calling `solve_shaped_sample`. Refactor full shaping to use the same helper and copy its Contact result.

- [ ] **Step 4: Verify GREEN and commit**

Run `make -B build/tests/test_interaction_hand_trajectories && ./build/tests/test_interaction_hand_trajectories && git diff --check`.

Commit:

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "refactor: share pickup Contact shaping"
```

### Task 2: Add the bounded exhaustive audit engine

**Files:**
- Create: `interaction_reuse_audit.h`
- Create: `interaction_reuse_audit.cpp`
- Create: `tests/cpp/test_interaction_reuse_audit.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Task 1's Contact and full shaping APIs.
- Produces:

```cpp
enum class ReuseAuditStatus : uint8_t { Complete = 0U, Incomplete = 1U };

struct ReuseAuditConfig {
    float maximum_contact_correction_m = 0.45F;
    size_t worker_count = 4U;
    uint32_t deadline_milliseconds = 30000U;
    size_t display_limit = 12U;
};

struct ReuseAuditCounts {
    size_t total = 0U;
    size_t processed = 0U;
    size_t contact_accepted = 0U;
    size_t fully_shaped = 0U;
    size_t object_rejected = 0U;
    size_t environment_rejected = 0U;
    size_t reusable = 0U;
};

struct ReuseAuditMotion {
    HandTrajectory source;
    ShapedHandTrajectory shaped;
};

struct ReuseAuditResult {
    ReuseAuditStatus status = ReuseAuditStatus::Incomplete;
    ReuseAuditCounts counts{};
    uint64_t elapsed_milliseconds = 0U;
    std::vector<ReuseAuditMotion> displayed;
};

ReuseAuditResult audit_reusable_hand_trajectories(
    const Database& database,
    const HandTrajectoryQuery& query,
    const OrientedBox& object,
    const EnvironmentGeometry& environment,
    const TrajectoryCollisionConfig& collision,
    const ReuseAuditConfig& config = ReuseAuditConfig{});
```

- [ ] **Step 1: Write failing engine tests**

Build a four-clip synthetic database: wrong hand, Contact-unreachable, Contact-reachable but environment-colliding, and reusable. Require exact stage counts. Run with one worker and four workers and require identical counts and displayed clip IDs. Run with `deadline_milliseconds = 0U` and require `Incomplete`, `processed < total`, and `displayed.empty()`. Require worker counts 0 and 5, zero display limit, and invalid correction values to throw.

- [ ] **Step 2: Verify RED**

Run `make -B build/tests/test_interaction_reuse_audit`.

Expected: Make has no rule or compilation fails because the engine is absent.

- [ ] **Step 3: Implement deterministic four-worker auditing**

Validate worker count `[1,4]`, positive display limit, finite nonnegative correction, valid object, and query mode `ApproachAxis` or `PositionOnly`. Start the deadline before selection. Count matching entries in `database.active_hands` as `total`.

Select with:

```cpp
HandTrajectoryConfig selection{};
selection.maximum_grasp_position_error_m =
    config.maximum_contact_correction_m;
selection.maximum_grasp_orientation_error_radians = 3.141592654F;
selection.maximum_compatible_clips = 4096U;
```

Count same-hand clips outside the 45 cm selector envelope as processed Contact rejections because the identical IK request gate rejects them. Set `IKConfig::maximum_request_position_m` to the same envelope.

Allocate one indexed outcome per selected trajectory. Workers use an atomic next index and check the deadline before claiming work. Each claimed clip receives Contact shaping; survivors receive full shaping and shaped feasibility evaluation. Clear all pose buffers before storing accepted background paths. Capture one worker exception, stop claims, join, and rethrow.

Merge outcomes by candidate index. If any candidate is unprocessed, return `Incomplete`, processed/total counts, and no displayed motions. Otherwise compute counts, stable-sort reusable motions by achieved orientation error, source cost, and clip ID, trim to `display_limit`, and return `Complete`.

- [ ] **Step 4: Add build target, verify, and commit**

Add `build/tests/test_interaction_reuse_audit` to `CPP_TEST_BINS`. Compile with `-pthread`, the audit source, trajectory, IK, pose, and target sources.

Run `make -B build/tests/test_interaction_reuse_audit && ./build/tests/test_interaction_reuse_audit && make -B build/tests/test_interaction_hand_trajectories && ./build/tests/test_interaction_hand_trajectories && git diff --check`.

Commit:

```bash
git add interaction_reuse_audit.h interaction_reuse_audit.cpp tests/cpp/test_interaction_reuse_audit.cpp Makefile
git commit -m "feat: add bounded exhaustive reuse audit"
```

### Task 3: Integrate audit controls, HUD, and real-pack probe

**Files:**
- Create: `interaction_reuse_audit_probe.cpp`
- Modify: `hand_trajectory_viewer.cpp`
- Modify: `tests/python/test_hand_trajectory_viewer.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Task 2's audit result.
- Produces: `A`-key audit and `interaction_reuse_audit_probe <pack> <shelf|under-table>`.

- [ ] **Step 1: Write failing viewer and Makefile tests**

Require `KEY_A`, `audit_reusable_hand_trajectories(`, `ReuseAuditStatus::Incomplete`, `deadline_milliseconds = 30000U`, `worker_count = 4U`, all audit stage labels, and `audit_stale = true` in the grasp-change block. Require no audit call in the Enter block. Require Makefile target `interaction_reuse_audit_probe:` and audit engine dependencies in `hand_trajectory_viewer:`.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=. python3 -m unittest tests.python.test_hand_trajectory_viewer -v`.

Expected: audit control, engine, HUD, and probe assertions fail.

- [ ] **Step 3: Integrate the viewer**

Add `KEY_A`. Store `std::optional<ReuseAuditResult> audit` and `bool audit_stale`. Grasp changes mark audit stale. On `A`, create an `ApproachAxis` audit query in full mode or `PositionOnly` in position mode, set four workers and 30,000 ms, and run the engine.

On `Complete`, replace valid trajectories from `displayed`, reset selection, set `searched_query`, rebuild selected animation, and mark audit fresh. On `Incomplete`, preserve prior trajectories and animation. Draw processed/total, Contact accepted, fully shaped, object, environment, reusable, elapsed milliseconds, and explicit `AUDIT INCOMPLETE`/`AUDIT STALE`. Enter remains outside the audit path.

- [ ] **Step 4: Add the headless real-pack probe**

The probe loads the compact database, chooses the first table Contact clip, and builds `make_coverage_environment`. For `shelf`, place the canonical object on environment box 5. For `under-table`, place it at central-table X/Z and half object height above ground. Build an `ApproachAxis` query, apply the viewer's +2 cm collision margin, run the audit, and print JSON containing scenario, status, elapsed milliseconds, and every count. Exit 0 for `Complete`, 3 for `Incomplete`, and 2 for invalid input/errors.

Add release targets with `-O3 -DNDEBUG -pthread` for the probe and add audit source/header plus `-pthread` to the viewer target.

- [ ] **Step 5: Verify and commit**

Run `PYTHONPATH=. python3 -m unittest tests.python.test_hand_trajectory_viewer -v && make hand_trajectory_viewer interaction_reuse_audit_probe && git diff --check`.

Commit:

```bash
git add hand_trajectory_viewer.cpp interaction_reuse_audit_probe.cpp tests/python/test_hand_trajectory_viewer.py Makefile
git commit -m "feat: expose exhaustive reuse audit"
```

### Task 4: Real-pack performance, regression, and live replacement

**Files:**
- Modify only prior task files for benchmark-proven defects.

**Interfaces:**
- Consumes: mixed pack and release binaries.
- Produces: two complete under-30-second reports and one verified viewer.

- [ ] **Step 1: Preserve memory before benchmarking**

Resolve the single `^./hand_trajectory_viewer$` process, verify its cwd, and terminate only that viewer. Verify controller PID `556617` remains alive. This avoids loading two compact databases simultaneously.

- [ ] **Step 2: Benchmark both real scenarios**

Run `timeout --signal=TERM --kill-after=2s 32s ./interaction_reuse_audit_probe build/smart-pickup/table-ground-pack shelf`.

Run `timeout --signal=TERM --kill-after=2s 32s ./interaction_reuse_audit_probe build/smart-pickup/table-ground-pack under-table`.

Expected: both exit 0, print `"status":"complete"`, report `elapsed_milliseconds <= 30000`, and provide exhaustive counts. If not, profile selection, Contact IK, full shaping, and collision separately; optimize only the dominant stage without weakening acceptance or collision, then rerun both.

- [ ] **Step 3: Run the complete regression gate**

Run `python3 -m unittest tests.python.test_interaction_sources tests.python.test_interaction_build_cli tests.python.test_interaction_artifacts tests.python.test_hand_trajectory_viewer`.

Run all three C++ binaries: `test_interaction_trajectory_database`, `test_interaction_hand_trajectories`, and `test_interaction_reuse_audit`; build both release binaries; run `git diff --check` and require clean status.

- [ ] **Step 4: Launch and verify only the viewer**

Launch one viewer on `DISPLAY=:1` with `build/smart-pickup/table-ground-pack`. Confirm raylib initialization, one viewer process, matching disk/live SHA-256, correct environment, no logged errors, and RSS below 1.1 GiB for 10 seconds. Do not use screenshots or touch the controller.
