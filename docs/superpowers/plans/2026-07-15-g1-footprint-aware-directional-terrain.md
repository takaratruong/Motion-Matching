# G1 Footprint-Aware Directional Terrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make forward, backward, diagonal, and heading-preserving lateral G1 motion use one footprint-aware terrain, planting, and certified-IK path without allowing terrain code to rotate the requested heading.

**Architecture:** Keep Holden's accepted 31-dimensional nearest-neighbor query unchanged. Publish an immutable command snapshot with independent travel and heading, build a conservative two-foot observation beside the existing centerline features, and feed that observation into the already-planned actual-center clearance and reversible IK transaction. Unsafe geometry slows or stops travel through the established latch while heading remains untouched.

**Tech Stack:** C++17, Holden motion matching, Raylib/Raygui, strict-FP C++ clearance object, Python 3/NumPy artifact builder and validator, `unittest`, ASan/UBSan, deterministic CSV gates at exact 25 Hz.

## Global Constraints

- `desired_velocity` and `desired_rotation` remain independent. Terrain, support, clearance, and IK may not derive or mutate heading.
- The footprint path applies identically to forward, backward, diagonal, and lateral travel. There is no forward/lateral mode branch; the physical probes rotate only with the independently predicted body heading.
- The accepted matching query stays exactly 31 finite values: 27 Holden dimensions plus the existing four root-centerline terrain values.
- The first implementation uses the current 1,770-clip, 459,682-frame motion pack without rebuilding features or relabeling turning clips as strafes.
- Exact runtime frequency is binary32 `0.04f` (`0x3d23d70a`), or 25 Hz.
- `g1_clearance.cpp` is always a separate strict-FP translation unit compiled with `-fno-fast-math -ffp-contract=off -frounding-math`; callers may use `-ffast-math`, but the final link command may not.
- Every new public state/result operation is checked and transactional. Finite unavailability safe-stops; malformed input/field or arithmetic failure exits through controlled cleanup.
- Certified routes must complete. Safe-stop is not an accepted substitute for forward, backward, diagonal, or lateral traversal on class-1 terrain.
- G1 mesh rendering and motion-data augmentation remain out of scope. The skeleton stays the acceptance visualizer.

## Preconditions

- Start only after combined clearance Tasks 3+4 are independently clean, merged, and verified on the execution branch.
- Run each source task in an isolated worktree created with the
  `using-git-worktrees` skill, review its commit, then integrate that commit
  into `/home/ubuntu/projects/motion-matching` before starting the next task.
  Do not copy ignored artifacts into a task worktree. Every step that reads,
  replaces, hashes, or launches from `resources/g1_terrain`, and every
  `Generate, never commit` step, is coordinator-owned and runs only in that
  integration worktree after the corresponding source commit passes review.
  Do not disturb the user's modified `resources/database.bin`,
  `resources/features.bin`, binaries, logs, or `mm/` directory.
- Preserve both the running old visualizer and every byte of
  `resources/g1_terrain` until Task 8 has certified a replacement against the
  external candidate pack. The only active-pack publication is the final
  checked exchange in Task 8, and it has an exchange-back recovery path.

## File and Ownership Map

- `resources/g1_terrain_builder/scenes.py`: owns the authenticated tangential,
  landing-exit-stress, and two rotated flat routes.
- `resources/validate_g1_terrain_database.py`: independently locks every new
  route ID, order, outcome, class, waypoint binary32 bit, and hold bit.
- `scene_runtime.h`: owns the same independent native route contract with a
  three-route maximum per scene.
- `route_runtime.h`: owns four-horizon deterministic route-command prediction;
  route mode never predicts from the synthetic gamepad.
- `g1_command_runtime.h`: owns independent command intent, deterministic heading overrides, and immutable four-sample command snapshots.
- `g1_footprint_runtime.h`: owns current and predicted physical-foot probes, swept corridor height envelopes, work budgets, and root/foot split diagnostics.
- `g1_clearance.cpp`: remains the only strict certified point/sphere/capsule/foot/leg/pose implementation.
- `g1_ik_runtime.h`: owns the 41-stage actual-pose swing transaction and consumes the footprint observation by const reference.
- `g1_controller_state.h`: owns the complete accepted mutable locomotion state
  across reset/switch and exposes a checked, allocation-free deep copy.
- `g1_frame_transaction.h`: owns the only three finite-rejection publications:
  immutable requested command intent, rejection diagnostics, and a one-frame
  safe-stop/forced-search latch.
- `controller.cpp`: orders input, matching, support, footprint observation, IK transaction, logging, and next-frame safe-stop handoff.
- `motion_match_log.h` and `resources/check_g1_runtime_log.py`: append diagnostics and own Gates L1--L5.
- `g1_candidate_audit.h` and
  `resources/audit_g1_directional_candidates.py`: produce deterministic,
  behavior-inert candidate-count/cost/range evidence for the data decision.
- Focused tests live in `tests/cpp/test_g1_command_runtime.cpp`,
  `tests/cpp/test_route_runtime.cpp`,
  `tests/cpp/test_g1_footprint_runtime.cpp`,
  `tests/cpp/test_g1_clearance.cpp`, `tests/cpp/test_g1_ik.cpp`,
  `tests/cpp/test_g1_controller_state.cpp`,
  `tests/cpp/test_g1_frame_transaction.cpp`,
  `tests/cpp/test_g1_candidate_audit.cpp`, `tests/python/test_scenes.py`,
  `tests/python/test_runtime_log.py`, and
  `tests/python/test_directional_candidates.py`.

---

### Task 1: Complete Certified Foot, Leg, Pose, and Actual-Sweep Aggregation

**Files:**
- Modify: `g1_clearance.cpp`
- Modify: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Consumes: certified point/sphere/capsule primitives from the merged clearance Tasks 3+4 and the fixed `G1LegConfig` geometry.
- Produces working `g1_foot_clearance`, `g1_swept_foot_clearance`, `g1_measure_leg_clearance`, `g1_measure_pose_clearance`, checked swing history, `g1_apply_swing_lift_y`, and actual-center `g1_swing_clearance_validate`.
- Safety values remain binary64. The public stubs are removed; no sampled or predicted-lift path is added.

- [ ] **Step 1: Add aggregate and actual-center RED modes**

Extend `tests/cpp/test_g1_clearance.cpp` with `--task5-aggregate` and `--task5-swing` modes. The fixtures must:

- independently call four sphere certificates and prove `g1_foot_clearance` returns the same minimum lower result, feasible witness, stable primitive key, and summed work;
- independently call four previous-to-current capsule certificates and prove `g1_swept_foot_clearance` uses the supplied actual endpoint bits and is reversal invariant;
- transform the configured knee, ankle, toe, four foot sphere centers, thigh endpoints, and shin endpoints from explicit global ankle/knee transforms and compare every `G1LegClearance` member against a direct public primitive call;
- require pose primitive order Hips `0`, left `1..9`, and right `10..18`, including stable minimum tie selection;
- seed every output/history with unique bits and require unchanged state for all six non-`Ok` statuses and checked-history failures;
- compare `g1_swing_clearance_validate` against four direct target-subtracted previous/current capsules using actual current center bits, exact `0.04f`, and the contact bypass contract; and
- reject a fake `baseline_y + lift` endpoint that differs by one binary32 ULP from the supplied current center.

- [ ] **Step 2: Run the aggregate RED against the strict object**

```bash
mkdir -p /tmp/g1-clearance-task5-red
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -frounding-math -I. -c g1_clearance.cpp -o /tmp/g1-clearance-task5-red/g1-clearance.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. -c tests/cpp/test_g1_clearance.cpp -o /tmp/g1-clearance-task5-red/test.o
g++ /tmp/g1-clearance-task5-red/test.o /tmp/g1-clearance-task5-red/g1-clearance.o -o /tmp/g1-clearance-task5-red/test
/tmp/g1-clearance-task5-red/test --task5-aggregate
```

Expected: compilation succeeds, but the focused run fails because the aggregate public functions still return the contract-stub status.

- [ ] **Step 3: Implement shared-budget aggregation in the strict translation unit**

Inside `g1_clearance.cpp`, add private helpers that preflight the complete requested primitive count against the immutable supplied budget, invoke the already-certified primitive core with monotonically consumed remaining work, add each `G1ClearanceWork` field with checked `uint32_t` arithmetic, and aggregate lower and feasible-witness results independently. Assign the public aggregate only after every member succeeds.

Use these exact physical sources:

```cpp
const vec3 sphere_center =
    global_positions(config.ankle) +
    quat_mul_vec3(global_rotations(config.ankle),
                  config.foot_sphere_centers_local[index]);
const vec3 thigh_a =
    global_positions(config.hip) +
    quat_mul_vec3(global_rotations(config.hip),
                  config.thigh_start_local);
const vec3 thigh_b =
    global_positions(config.hip) +
    quat_mul_vec3(global_rotations(config.hip),
                  config.thigh_end_local);
const vec3 shin_a =
    global_positions(config.knee) +
    quat_mul_vec3(global_rotations(config.knee),
                  config.shin_start_local);
const vec3 shin_b =
    global_positions(config.knee) +
    quat_mul_vec3(global_rotations(config.knee),
                  config.shin_end_local);
```

Point certificates own Hips, knee, ankle, and toe. Sphere/capsule certificates own foot, thigh, and shin. Copy the full winning `G1ClearanceResult` into each `minimum` member; do not reconstruct a float minimum.

- [ ] **Step 4: Implement checked history, lift materialization, and actual-center validation**

`g1_apply_swing_lift_y` performs exactly `RN32(double(input_y)+double(lift_m))`, accepts only finite nonnegative lift through `0.08f`, canonicalizes zero, rejects nonzero subnormal output, and assigns only on `Ok`.

`g1_swing_history_reset` and `g1_swing_history_commit` validate all twelve center components into a local candidate before assignment. `g1_swing_clearance_validate` requires initialized history and exact 25 Hz; contact returns `Ok` with `sweep_evaluated=false`, while swing certifies four history-to-supplied-current capsules after subtracting the configured swing clearance in binary64. It never receives or reconstructs a lift.

- [ ] **Step 5: Run focused, full, parity, negative-guard, and sanitizer GREEN**

```bash
mkdir -p /tmp/g1-clearance-task5
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -frounding-math -I. -c g1_clearance.cpp -o /tmp/g1-clearance-task5/kernel.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. -c tests/cpp/test_g1_clearance.cpp -o /tmp/g1-clearance-task5/test.o
g++ /tmp/g1-clearance-task5/test.o /tmp/g1-clearance-task5/kernel.o -o /tmp/g1-clearance-task5/test
/tmp/g1-clearance-task5/test --task5-aggregate
/tmp/g1-clearance-task5/test --task5-swing
/tmp/g1-clearance-task5/test
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. -c tests/cpp/test_g1_clearance.cpp -o /tmp/g1-clearance-task5/test-fast.o
g++ /tmp/g1-clearance-task5/test-fast.o /tmp/g1-clearance-task5/kernel.o -o /tmp/g1-clearance-task5/test-fast
/tmp/g1-clearance-task5/test --query-parity > /tmp/g1-clearance-task5/strict.txt
/tmp/g1-clearance-task5/test-fast --query-parity > /tmp/g1-clearance-task5/fast.txt
cmp /tmp/g1-clearance-task5/strict.txt /tmp/g1-clearance-task5/fast.txt

if g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
     -c g1_clearance.cpp \
     -o /tmp/g1-clearance-task5/forbidden-fast-kernel.o \
     2>/tmp/g1-clearance-task5/forbidden-fast-kernel.stderr; then
  echo 'ERROR: strict clearance kernel accepted -ffast-math' >&2
  exit 1
fi
rg 'strict|fast-math|FAST_MATH' \
  /tmp/g1-clearance-task5/forbidden-fast-kernel.stderr

san_flags=(-O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all)
g++ -std=c++17 "${san_flags[@]}" -fno-fast-math -ffp-contract=off \
  -frounding-math -I. -c g1_clearance.cpp \
  -o /tmp/g1-clearance-task5/kernel-san.o
g++ -std=c++17 "${san_flags[@]}" -I. \
  -c tests/cpp/test_g1_clearance.cpp \
  -o /tmp/g1-clearance-task5/test-san.o
g++ "${san_flags[@]}" /tmp/g1-clearance-task5/test-san.o \
  /tmp/g1-clearance-task5/kernel-san.o \
  -o /tmp/g1-clearance-task5/test-san
ASAN_OPTIONS=detect_leaks=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/g1-clearance-task5/test-san
```

Expected: every positive test exits zero, parity is byte-identical, the
sanitizer is clean, the direct fast kernel is rejected, and no contract stub
remains for any completed aggregate or actual-center API.

- [ ] **Step 6: Commit certified aggregate clearance**

```bash
git add g1_clearance.cpp tests/cpp/test_g1_clearance.cpp
git commit -m "feat: certify G1 foot and pose clearance"
```

---

### Task 2: Publish Independent Travel and Heading Commands

**Files:**
- Create: `g1_command_runtime.h`
- Create: `tests/cpp/test_g1_command_runtime.cpp`
- Modify: `route_runtime.h`
- Modify: `tests/cpp/test_route_runtime.cpp`
- Modify: `g1_controller_state.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`

**Interfaces:**
- Produces: `G1CommandIntent`, `G1CommandSnapshot`, `G1TestHeadingOverride`, `g1_test_heading_override_parse`, `g1_command_snapshot_build`, and `g1_command_snapshot_is_valid`.
- Produces checked `deterministic_route_predict_commands`, which samples the
  selected route at the same four horizons used by the matcher.
- `G1CommandSnapshot` owns requested and applied travel separately plus exactly four predicted desired velocities, root positions, root rotations, and desired headings.
- Terrain may return only an applied velocity. No terrain API receives a mutable heading reference.

- [ ] **Step 1: Write the command-contract RED**

Create `tests/cpp/test_g1_command_runtime.cpp` with table-driven checks using these exact string/bit pairs:

```cpp
static const struct {
    const char* text;
    quat expected;
} headings[] = {
    {"forward", quat(1.0f, 0.0f, 0.0f, 0.0f)},
    {"backward", quat(0.0f, 0.0f, 1.0f, 0.0f)},
    {"positive-x", quat(0.707106769f, 0.0f, 0.707106769f, 0.0f)},
    {"negative-x", quat(0.707106769f, 0.0f, -0.707106769f, 0.0f)},
    {"diagonal-positive-x", quat(0.923879504f, 0.0f, 0.382683426f, 0.0f)},
    {"diagonal-negative-x", quat(0.923879504f, 0.0f, -0.382683426f, 0.0f)},
};
```

Require null input to mean no override; require each exact string to produce exact quaternion bits; reject empty, whitespace, case variants, suffixes, and every numeric string transactionally. Build two snapshots with identical requested travel/headings/trajectories but different applied velocity, and require every heading bit to remain identical. Poison output and require invalid shapes, nonfinite values, non-unit quaternions, and aliasing to leave it unchanged.

Extend `tests/cpp/test_route_runtime.cpp` with bit-exact four-sample prediction
fixtures. The output is unchanged on every failure. Require:

- a corner whose sample 0 is on segment zero and later samples are on segment
  one, proving prediction follows route time rather than the gamepad;
- the exact `tangent-level-boundary` schedule;
- a positive landing hold whose horizons before, inside, and after the hold
  produce moving, zero, and moving commands respectively;
- route completion to produce canonical positive-zero commands;
- exact speed-scale bits at `1.0f`, `0.5f`, and `0.0f`; and
- a consumed safe-stop latch to canonicalize all four commands to zero while
  setting `force_search=true` without receiving a quaternion.

- [ ] **Step 2: Run the compile RED**

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_g1_command_runtime.cpp -o /tmp/test_g1_command_runtime_red
```

Expected: compilation fails because `g1_command_runtime.h` and its interfaces do not exist.

- [ ] **Step 3: Implement the fixed command types and parser**

Create `g1_command_runtime.h` with these public layouts:

```cpp
#pragma once

#include "ik.h"
#include "terrain_runtime.h"

#include <cstdint>
#include <cstring>

static constexpr int G1CommandTrajectorySampleCount = 4;

struct G1CommandIntent
{
    vec3 requested_velocity;
    quat desired_heading;
};

struct G1CommandSnapshot
{
    G1CommandIntent intent;
    vec3 applied_velocity;
    vec3 predicted_desired_velocities[G1CommandTrajectorySampleCount];
    vec3 predicted_root_positions[G1CommandTrajectorySampleCount];
    quat predicted_root_rotations[G1CommandTrajectorySampleCount];
    quat predicted_desired_headings[G1CommandTrajectorySampleCount];
};

struct G1TestHeadingOverride
{
    bool active = false;
    quat heading;
};

static inline bool g1_test_heading_override_parse(
    G1TestHeadingOverride& output,
    const char* text,
    char* error,
    int error_capacity);

static inline bool g1_command_snapshot_build(
    G1CommandSnapshot& output,
    G1CommandIntent intent,
    vec3 applied_velocity,
    const slice1d<vec3> predicted_desired_velocities,
    const slice1d<vec3> predicted_root_positions,
    const slice1d<quat> predicted_root_rotations,
    const slice1d<quat> predicted_desired_headings,
    char* error,
    int error_capacity);

static inline bool g1_command_snapshot_is_valid(
    const G1CommandSnapshot& value);
```

Implement the parser as exact string-to-constant lookup using the six table entries from Step 1. Do not call trigonometry or parse a float. `g1_command_snapshot_build` validates all values and exact size four into a local candidate, canonicalizes signed-zero velocity/position components with `terrain_runtime_canonicalize_output`, requires `ik_quat_is_unit` for every quaternion, and assigns `output` only after `g1_command_snapshot_is_valid(candidate)` passes.

Add this interface to `route_runtime.h`:

`route_runtime.h` directly includes `g1_command_runtime.h`; it does not rely on
include order for `G1CommandTrajectorySampleCount`. Extend the existing
standalone header-compilation fixture so `route_runtime.h` is the first project
header included.

```cpp
struct deterministic_route_prediction
{
    vec3 commands[G1CommandTrajectorySampleCount];
    int sampled_frames[G1CommandTrajectorySampleCount] = {};
    bool force_search = false;
};

static inline bool deterministic_route_predict_commands(
    deterministic_route_prediction& output,
    const scene_route& route,
    int current_frame,
    vec3 current_applied_velocity,
    float dt,
    float speed,
    float trajectory_sample_time,
    float future_speed_scale,
    bool safe_stop_latched,
    char* error,
    int error_capacity);
```

Require exact `dt == 0x3d23d70a`, output count four, finite runtime values,
`0 <= future_speed_scale <= 1`, and positive finite speed/sample time. Sample
zero is the supplied current applied velocity. For samples `1..3`, compute
`ceilf(RN32(i * trajectory_sample_time) / dt)` with the checked binary32
helpers, checked-add it to `current_frame`, call
`deterministic_route_command`, and checked-multiply only X/Z by
`future_speed_scale`. When `safe_stop_latched` is true, all four commands are
canonical zero and `force_search=true`. Build a complete local candidate and
assign only on success.

- [ ] **Step 4: Move heading selection before terrain velocity limiting**

In `controller.cpp`, parse `MM_TEST_HEADING` beside the other bounded test options before Raylib initialization. Reject an active override outside route mode. Reorder the frame input path to this ownership:

```cpp
const vec3 commanded_velocity = desired_velocity_curr;
quat desired_rotation_curr = desired_rotation_update(
    state.desired_rotation,
    gamepadstick_left,
    gamepadstick_right,
    state.camera_azimuth,
    desired_strafe,
    commanded_velocity);
if (test_heading.active) {
    desired_rotation_curr = test_heading.heading;
}

traversability_diagnostics traversal = {};
desired_velocity_curr = traversability_limit_command(
    state.traversal_speed_scale,
    state.traversal_speed_scale_velocity,
    traversal,
    active_scene.walkability,
    active_scene.terrain,
    state.simulation_position,
    commanded_velocity,
    dt);
```

Delete the old post-traversability call to `desired_rotation_update`. This prevents a zeroed safe-stop velocity from becoming a heading input.

For an active deterministic heading override, fill all four
`trajectory_desired_rotations` entries from the override instead of calling
the input-device predictor. In route mode, call
`deterministic_route_predict_commands` after traversability has materialized
sample zero and copy its four commands to
`state.trajectory_desired_velocities`; never call
`trajectory_desired_velocities_predict` in route mode. In non-route mode keep
the existing input-device predictor. Route travel and heading therefore remain
independent even at a corner, landing hold, or finite safe-stop. After both
prediction paths finish, build:

```cpp
G1CommandIntent command_intent = {};
command_intent.requested_velocity = commanded_velocity;
command_intent.desired_heading = desired_rotation_curr;
if (!g1_command_snapshot_build(
        state.command,
        command_intent,
        desired_velocity_curr,
        state.trajectory_desired_velocities,
        state.trajectory_positions,
        state.trajectory_rotations,
        state.trajectory_desired_rotations,
        artifact_error,
        static_cast<int>(sizeof(artifact_error)))) {
    controlled_runtime_error(artifact_error);
    return;
}
```

- [ ] **Step 5: Add command state to reset and swap**

Include `g1_command_runtime.h` from `g1_controller_state.h`, add `G1CommandSnapshot command;`, swap it in `g1_controller_state_swap`, and initialize a valid zero-travel four-sample snapshot at scene spawn. Extend `tests/cpp/test_g1_controller_state.cpp` to poison every member and require reset to restore zero requested/applied travel and four copies of the spawn position/rotation.

- [ ] **Step 6: Run command and controller GREEN tests**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_g1_command_runtime.cpp -o /tmp/test_g1_command_runtime
/tmp/test_g1_command_runtime
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_route_runtime.cpp -o /tmp/test_route_runtime_command
/tmp/test_route_runtime_command
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_controller_state_command
/tmp/test_g1_controller_state_command
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src -c controller.cpp -o /tmp/controller_command.o
```

Expected: all commands exit zero with no warnings. A bounded route launch with `MM_TEST_HEADING=positive-x` succeeds; invalid or live-mode use exits before opening a window.

- [ ] **Step 7: Commit the independent command contract**

```bash
git add g1_command_runtime.h route_runtime.h g1_controller_state.h controller.cpp \
  tests/cpp/test_g1_command_runtime.cpp tests/cpp/test_route_runtime.cpp \
  tests/cpp/test_g1_controller_state.cpp
git commit -m "feat: separate G1 travel and heading commands"
```

---

### Task 3: Authenticate Directional Certification Routes and Build an External Candidate Pack

**Files:**
- Modify: `resources/g1_terrain_builder/scenes.py`
- Modify: `resources/validate_g1_terrain_database.py`
- Modify: `scene_runtime.h`
- Modify: `tests/python/test_scenes.py`
- Modify: `tests/cpp/test_scene_runtime.cpp`
- Generate, never commit: `/tmp/g1-terrain-footprint-scene-source-v1/`
- Generate, never commit: `/tmp/g1-terrain-footprint-runtime-v1/`
- Generate, never commit: `/tmp/g1-footprint-prechange-oracle/`
- Snapshot, never commit: `/tmp/g1-terrain-active-v1.*`

**Interfaces:**
- Consumes: the old authenticated active pack and Task-2 route-aware command
  prediction.
- Produces four additional zero-hold, class-1 routes, independently locked by
  exact binary32 bits:

| Scene | Ordered route | Waypoints `(x,z)` |
|---|---|---|
| `stairs-shallow` | `flat-positive-z` | `(0,0)`, `(0,1)` |
| `stairs-shallow` | `flat-positive-x` | `(0,0)`, `(1,0)` |
| `stairs-standard` | `landing-side-exit-stress` | `(0,0)`, `(0,1.75)`, `(0,3.96)`, `(0,5.53)` |
| `mixed-multilevel` | `tangent-level-boundary` | `(0,0)`, `(0.62,2)`, `(0.62,6)` |

- Produces `/tmp/g1-terrain-footprint-runtime-v1`, a complete validated
  candidate pack. It does **not** replace or modify `resources/g1_terrain`.

- [ ] **Step 1: Snapshot and validate the old active pack before source integration**

This coordinator-owned step runs from the integration worktree before the
Task-3 source commit is merged. It retains a self-contained Python import tree
compatible with the old one-route pack and records the complete entry tree,
not only four payloads. The retained validator may not import any post-Task-3
repository module:

```bash
test ! -e /tmp/g1-terrain-active-v1-python
test ! -e /tmp/g1-terrain-active-v1.TREE
test ! -e /tmp/g1-terrain-active-v1.SHA256SUMS
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
mkdir -p /tmp/g1-terrain-active-v1-python/resources
install -m 0644 resources/validate_g1_terrain_database.py \
  /tmp/g1-terrain-active-v1-python/resources/validate_g1_terrain_database.py
install -m 0644 resources/quat.py \
  /tmp/g1-terrain-active-v1-python/resources/quat.py
cp -a resources/g1_terrain_builder \
  /tmp/g1-terrain-active-v1-python/resources/g1_terrain_builder
find /tmp/g1-terrain-active-v1-python -type d -name __pycache__ \
  -prune -exec rm -rf {} +
(cd /tmp && PYTHONNOUSERSITE=1 \
  PYTHONPATH=/tmp/g1-terrain-active-v1-python \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    /tmp/g1-terrain-active-v1-python/resources/validate_g1_terrain_database.py \
    /home/ubuntu/projects/motion-matching/resources/g1_terrain)
(
  cd resources/g1_terrain
  find . -printf '%y %m %P -> %l\n' | LC_ALL=C sort
) > /tmp/g1-terrain-active-v1.TREE
(
  cd resources/g1_terrain
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > /tmp/g1-terrain-active-v1.SHA256SUMS
(
  cd /tmp/g1-terrain-active-v1-python
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > /tmp/g1-terrain-active-v1.python.SHA256SUMS
```

Expected: both old-validator invocations print `VALID`; all three snapshots
are nonempty. Before every later old-pack validation, verify the retained
Python-tree hashes, change directory to `/tmp`, set `PYTHONNOUSERSITE=1` and
`PYTHONPATH` only to the retained root, and invoke the retained script by
absolute path. These files are the rollback oracle. The later validator
intentionally rejects this old route catalog.

- [ ] **Step 2: Write route-order and exact-bit RED tests**

In `tests/python/test_scenes.py`, require these ordered IDs:

```python
expected = {
    "stairs-shallow": (
        "ascent-landing-descent", "flat-positive-z", "flat-positive-x"),
    "stairs-standard": (
        "ascent-landing-descent", "landing-side-exit-stress"),
    "mixed-multilevel": ("full-course", "tangent-level-boundary"),
}
```

For each new route, independently pack every coordinate with
`struct.pack("<f", value)` and require these exact words plus exact positive
zero hold `0x00000000`:

```python
expected_bits = {
    ("stairs-shallow", "flat-positive-z"):
        ((0x00000000, 0x00000000), (0x00000000, 0x3f800000)),
    ("stairs-shallow", "flat-positive-x"):
        ((0x00000000, 0x00000000), (0x3f800000, 0x00000000)),
    ("stairs-standard", "landing-side-exit-stress"): (
        (0x00000000, 0x00000000), (0x00000000, 0x3fe00000),
        (0x00000000, 0x407d70a4), (0x00000000, 0x40b0f5c3)),
    ("mixed-multilevel", "tangent-level-boundary"): (
        (0x00000000, 0x00000000), (0x3f1eb852, 0x40000000),
        (0x3f1eb852, 0x40c00000)),
}
```

Extend `tests/cpp/test_scene_runtime.cpp` with the same four arrays and compare
each loaded component via `terrain_float_bits`, as well as route order,
`expected_outcome == "traverse"`, class `1`, and
`terrain_float_bits(landing_hold_seconds) == 0x00000000`. Mutate each ID,
coordinate bit, hold sign bit, outcome, and class independently and require
native loading to fail transactionally.

- [ ] **Step 3: Run the focused RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes.ProceduralSceneTests -v
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_routes_red
/tmp/test_scene_runtime_routes_red
```

Expected: both fail because the four added routes do not exist.

- [ ] **Step 4: Add only the four ordered routes to the producer**

Make this localized signature/return change to the existing
`_corridor_definition`; all code between the signature and the final
`SceneDefinition` remains byte-for-byte unchanged:

```python
def _corridor_definition(
    scene_id, label, surface, course_end_z, parameters, route,
    walkability_class, extra_routes=(),
):
    extra_routes = tuple(extra_routes)
    heightfield_end_z = _runtime_f32_upper_ceiling(
        course_end_z + LOOKAHEAD_MARGIN, "heightfield zmax")
```

The shown `heightfield_end_z` line is the existing first body statement; leave
the remainder of lines 445--501 in place. At the existing `SceneDefinition`
constructor, change only `routes=(route,)` to
`routes=(route,) + extra_routes`. This is a localized edit against the current
function, so no undefined replacement body is introduced.

In `_stair_definition`, insert immediately before its existing return:

```python
extra_routes = ()
if scene_id == "stairs-shallow":
    extra_routes = (
        SceneRoute("flat-positive-z", ((0.0, 0.0), (0.0, 1.0)),
                   "traverse", 1, 0.0),
        SceneRoute("flat-positive-x", ((0.0, 0.0), (1.0, 0.0)),
                   "traverse", 1, 0.0),
    )
elif scene_id == "stairs-standard":
    extra_routes = (SceneRoute(
        "landing-side-exit-stress",
        ((0.0, 0.0), (0.0, 1.75), (0.0, 3.96), (0.0, 5.53)),
        "traverse", 1, 0.0),)
```

Pass `extra_routes` as the final `_corridor_definition` argument. In
`_mixed_definition`, pass exactly:

```python
(SceneRoute(
    "tangent-level-boundary",
    ((0.0, 0.0), (0.62, 2.0), (0.62, 6.0)),
    "traverse", 1, 0.0,
),)
```

Do not change any surface, bounds, region, classifier, spawn, or existing
route. `landing-side-exit-stress` stays class-1 `traverse` in the authenticated
schema; only the dedicated Gate-L2 diagnostic later permits a pre-commit safe
stop as a scoped stress outcome.

- [ ] **Step 5: Lock route bits independently in Python and C++**

In `resources/validate_g1_terrain_database.py`, set:

```python
EXPECTED_ROUTE_IDS["stairs-shallow"] = (
    "ascent-landing-descent", "flat-positive-z", "flat-positive-x")
EXPECTED_ROUTE_IDS["stairs-standard"] = (
    "ascent-landing-descent", "landing-side-exit-stress")
EXPECTED_ROUTE_IDS["mixed-multilevel"] = (
    "full-course", "tangent-level-boundary")
EXPECTED_WAYPOINT_COUNTS["stairs-shallow"] = (5, 2, 2)
EXPECTED_WAYPOINT_COUNTS["stairs-standard"] = (5, 4)
EXPECTED_WAYPOINT_COUNTS["mixed-multilevel"] = (8, 3)
```

Add `EXPECTED_ADDED_ROUTE_BITS` using the complete Step-2 dictionary. After
`_validate_route`, derive each observed pair with
`struct.unpack("<I", struct.pack("<f", component))[0]`, derive the hold word
the same way, and require exact tuple equality and `hold == 0x00000000` for
every dictionary entry. This lock is independent of producer objects and
NumPy.

In `scene_runtime.h`, expand the native contract's second dimension from two
to three and set:

```cpp
static const int route_counts[14] = {
    1,1,1,1,3,2,1,1,1,1,1,1,2,2};
```

Add a `waypoint_bit_contract` pointer/count and `landing_hold_bits` to the
native contract. Populate the four Step-2 arrays as `UINT32_C` literals and
compare each parsed route component with `terrain_float_bits`; do not obtain
the expected values from JSON or the Python builder. Existing route ID,
outcome, and class locks remain. A three-route scene is the maximum; no
dynamic contract allocation is introduced.

- [ ] **Step 6: Run source GREEN tests and commit**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_scenes -v
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_routes
/tmp/test_scene_runtime_routes
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_route_runtime.cpp -o /tmp/test_route_runtime_routes
/tmp/test_route_runtime_routes
git diff --check
```

```bash
git add resources/g1_terrain_builder/scenes.py \
  resources/validate_g1_terrain_database.py scene_runtime.h \
  tests/python/test_scenes.py tests/cpp/test_scene_runtime.cpp
git commit -m "test: authenticate directional G1 terrain routes"
```

Expected: all source tests pass, including route-class coverage and exact-bit
mutation tests.

- [ ] **Step 7: Build a disposable scene source and full external runtime candidate**

After Task 3 passes both reviews and is integrated, run from the integration
worktree. Both destinations must be absent:

```bash
test ! -e /tmp/g1-terrain-footprint-scene-source-v1
test ! -e /tmp/g1-terrain-footprint-runtime-v1
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/build_g1_terrain_database.py --grail-limit 0 \
  --output /tmp/g1-terrain-footprint-scene-source-v1
```

Expected: `BUILT g1-terrain-artifacts/v2 ... scenes=14`.

- [ ] **Step 8: Materialize and validate the external candidate without publication**

Run this repository-root program through the diffsim Python. It makes a full
external pack, replaces only its scenes, updates only its scene-index digest,
proves protected motion bytes are unchanged, fsyncs it, and validates it. It
never opens the active pack for writing:

```python
import hashlib
import json
import os
import shutil
import subprocess

from resources.g1_terrain_builder.artifacts import (
    _fsync_tree, canonical_json_bytes,
)

active = os.path.abspath("resources/g1_terrain")
source_scenes = "/tmp/g1-terrain-footprint-scene-source-v1/scenes"
candidate = "/tmp/g1-terrain-footprint-runtime-v1"
protected_names = (
    "database.bin", "terrain_features.bin", "terrain_support.bin",
    "validation.json",
)
if os.path.lexists(candidate):
    raise RuntimeError(f"refusing pre-existing candidate: {candidate}")

protected = {}
for name in protected_names:
    path = os.path.join(active, name)
    with open(path, "rb") as stream:
        protected[name] = hashlib.file_digest(stream, "sha256").hexdigest()

shutil.copytree(active, candidate, copy_function=os.link)
shutil.rmtree(os.path.join(candidate, "scenes"))
shutil.copytree(source_scenes, os.path.join(candidate, "scenes"))

manifest_path = os.path.join(candidate, "manifest.json")
with open(manifest_path, "rb") as stream:
    manifest = json.load(stream)
with open(os.path.join(candidate, "scenes", "index.json"), "rb") as stream:
    index_bytes = stream.read()
manifest["scene_index"]["sha256"] = hashlib.sha256(index_bytes).hexdigest()
temporary = manifest_path + ".new"
with open(temporary, "xb") as stream:
    stream.write(canonical_json_bytes(manifest))
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, manifest_path)

for name, expected in protected.items():
    with open(os.path.join(candidate, name), "rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != expected:
        raise RuntimeError(f"protected payload changed: {name}")

subprocess.run([
    "/home/ubuntu/miniconda3/envs/diffsim/bin/python",
    "resources/validate_g1_terrain_database.py", candidate,
], check=True)
_fsync_tree(candidate)
print("VALID external footprint candidate", candidate)
```

Expected: `VALID ... scenes=14`; active `resources/g1_terrain` still matches
both `/tmp/g1-terrain-active-v1.*` snapshots byte-for-byte and the old
visualizer remains alive.

- [ ] **Step 9: Capture immutable pre-footprint flat and forward-terrain oracles**

Build the integrated Task-3 source before `g1_footprint_runtime.h` exists:

```bash
mkdir -p /tmp/g1-footprint-prechange-oracle
g++ -std=c++17 -O2 -DNDEBUG -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/g1-footprint-prechange-oracle/controller \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Capture these eight 100-frame flat runs. Together they prove forward,
backward, left, and right relative travel under two different absolute world
travel/heading pairs:

```bash
flat_cases=(
  flat-positive-z:forward:forward
  flat-positive-z:backward:backward
  flat-positive-z:positive-x:left
  flat-positive-z:negative-x:right
  flat-positive-x:positive-x:forward
  flat-positive-x:negative-x:backward
  flat-positive-x:forward:right
  flat-positive-x:backward:left
)
for specification in "${flat_cases[@]}"; do
  IFS=: read -r route heading relative <<<"$specification"
  stem="${route}__${heading}__${relative}"
  DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
    MM_TERRAIN_SCENE=stairs-shallow MM_TEST_MODE=route \
    MM_TEST_ROUTE="$route" MM_TEST_FRAMES=100 \
    MM_TEST_HEADING="$heading" MM_TERRAIN_WEIGHT=4 \
    MM_LOG="/tmp/g1-footprint-prechange-oracle/flat-${stem}.csv" \
    /tmp/g1-footprint-prechange-oracle/controller
done
```

Capture full 800-frame forward oracles for every certified geometry family
used as the directional performance reference:

```bash
forward_routes=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  grail-curb-low:curb-forward
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
)
for specification in "${forward_routes[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TEST_HEADING=forward MM_TERRAIN_WEIGHT=4 \
    MM_LOG="/tmp/g1-footprint-prechange-oracle/forward-${scene}__${route}.csv" \
    /tmp/g1-footprint-prechange-oracle/controller
done
sha256sum /tmp/g1-footprint-prechange-oracle/controller \
  /tmp/g1-footprint-prechange-oracle/*.csv \
  > /tmp/g1-footprint-prechange-oracle/SHA256SUMS
sha256sum -c /tmp/g1-footprint-prechange-oracle/SHA256SUMS
```

Expected: all 13 runs exit zero. These logs freeze common/full-route query
bits, selected frame/cost, simulation XZ, route cursor/completion, support,
contact levels, and quality metrics before footprint/IK integration. They are
never regenerated from later source.

---

### Task 4: Observe the Swept Physical Footprint in Every Direction

**Files:**
- Create: `g1_footprint_runtime.h`
- Create: `tests/cpp/test_g1_footprint_runtime.cpp`

**Interfaces:**
- Consumes: `G1CommandSnapshot`, a four-sample two-foot
  `G1FootContactSchedule`, fixed `G1LegConfig` geometry, adjusted-pose global
  transforms, G1HF/v2 terrain, and its matching G1WM grid.
- Produces: `G1FootprintStatus`, `G1FootprintBudget`,
  `G1FootContactSchedule`, `G1FootprintProbe`,
  `G1FootprintFootObservation`, `G1FootprintObservation`,
  `g1_footprint_budget`, checked `g1_foot_contact_schedule_build`, and
  transactional `g1_footprint_observe_v2`.
- Uses four configured 0.02 m sphere/sole probes per foot and exactly three predicted segments per probe: 24 total sweeps. It contains no strafe/forward mode flag.

- [ ] **Step 1: Write footprint RED fixtures**

Create `tests/cpp/test_g1_footprint_runtime.cpp` using the existing G1HF/v2 test-field helpers and explicit 31-bone global arrays. Add these independently checked cases:

1. A lateral level field with height `0.32f` for `x <= 0.60f` and `0.0f` for `x >= 0.62f`; put the root at `x=0.62f` and one physical foot probe at `x=0.58f`. Require `maximum_root_split_m >= 0.32`, while a root-centerline query down `x=0.62f` remains all zero.
2. With both feet non-contact, transport the same four physical probes over
   identical world root positions under all six heading constants.
   Independently rotate each local offset with `quat_mul_vec3` and require
   exact predicted-probe bits. Do not pass a movement-mode enum. Repeat with
   contacts `{true,true,false,false}` and require the planted samples to retain
   the exact sample-zero world bits until the release edge, followed by the
   same transported swing corridor.
3. Put a one-cell height change between prediction horizons and require the swept segment envelope, not only endpoint samples, to report it.
4. Mirror left/right geometry and reverse every segment. Require identical combined min/max heights, walkability class, split, and work counts.
5. Exercise blocked cells, one-ULP outside domain, malformed G1HF/G1WM, nonfinite command/pose data, budget one below exact work, exact budget, and aliased/poisoned output. Every non-`Ok` status leaves output unchanged.
6. Drive a non-contact foot over a `0.32 m` down-step with contact samples
   `{false,false,true,true}`. Require landing sample `2`, the exact lower
   G1HF/v2 center height/normal, predicted landing-centroid X/Z, four selected
   landing-probe samples, and a ready coplanar patch. Require samples after the
   rising edge to remain locked to the contact-edge geometry. Reverse it for
   the up-step. A current contact with no later rising edge suppresses landing
   selection; no rising edge leaves `landing_expected=false`.
7. Straddle the step at the predicted contact sample and require
   `landing_expected=true`, `landing_patch_ready=false`, and a plane residual
   above `0.005 m`; no upper/lower average may be fabricated.
8. Require exactly 24 walkability sweeps, no more than 35 surface queries, at
   most 65,536 visited height nodes, and no source token matching `strafe`,
   `forward_mode`, or `lateral_mode` in the production header.
9. Build the contact schedule from a two-range contact table. Require samples
   at horizon-frame offsets `0,9,17,25` for `dt=0.04f` and
   `trajectory_sample_time=1/3f`, clamped to the final frame of the current
   range. Pass the controller's current left/right contact bits and require
   exact equality with schedule sample zero. Reject either mismatch, a frame
   outside every range, overlap/gap, wrong two-column shape, bad horizon,
   nonfinite input, and poisoned output transactionally.

- [ ] **Step 2: Run the footprint compile RED**

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_g1_footprint_runtime.cpp -o /tmp/test_g1_footprint_runtime_red
```

Expected: compilation fails because `g1_footprint_runtime.h` is absent.

- [ ] **Step 3: Define fixed-size transactional footprint state**

Create these public definitions in `g1_footprint_runtime.h`:

```cpp
#pragma once

#include "g1_command_runtime.h"
#include "g1_ik.h"

#include <cstdint>

enum G1FootprintStatus
{
    G1FootprintOk,
    G1FootprintOutsideDomain,
    G1FootprintBudgetExceeded,
    G1FootprintInvalidInput,
    G1FootprintInvalidField,
    G1FootprintArithmeticFailure,
};

struct G1FootprintBudget
{
    uint32_t maximum_sweeps = 24;
    uint32_t maximum_surface_queries = 35;
    uint32_t maximum_node_visits = 65536;
};

struct G1FootprintWork
{
    uint32_t sweeps = 0;
    uint32_t surface_queries = 0;
    uint32_t node_visits = 0;
};

struct G1FootContactSchedule
{
    bool contact[2][G1CommandTrajectorySampleCount] = {};
};

struct G1FootprintProbe
{
    vec3 current_sphere_center;
    vec3 current_sole_point;
    G1SurfaceSample current_surface;
    vec3 predicted_sphere_centers[G1CommandTrajectorySampleCount];
    vec3 predicted_sole_points[G1CommandTrajectorySampleCount];
    G1SurfaceQueryStatus predicted_surface_status[
        G1CommandTrajectorySampleCount] = {
            G1SurfaceQueryInvalid, G1SurfaceQueryInvalid,
            G1SurfaceQueryInvalid, G1SurfaceQueryInvalid};
    G1SurfaceSample predicted_surfaces[G1CommandTrajectorySampleCount];
    G1SurfaceSample selected_landing_surface;
    float corridor_minimum_height = 0.0f;
    float corridor_maximum_height = 0.0f;
    int encountered_walkability_class = 1;
};

struct G1FootprintFootObservation
{
    G1FootprintProbe probes[4];
    bool current_contact = false;
    bool landing_expected = false;
    bool landing_patch_ready = false;
    uint32_t landing_sample = UINT32_MAX;
    vec3 predicted_landing_sole_center;
    G1SurfaceQueryStatus predicted_landing_surface_status =
        G1SurfaceQueryInvalid;
    G1SurfaceSample predicted_landing_surface;
    int predicted_landing_walkability_class = 0;
    double landing_patch_maximum_residual_m = 0.0;
    float corridor_minimum_height = 0.0f;
    float corridor_maximum_height = 0.0f;
    double maximum_root_split_m = 0.0;
    int encountered_walkability_class = 1;
    bool multilevel = false;
};

struct G1FootprintObservation
{
    G1SurfaceSample root_surface;
    G1FootprintFootObservation feet[2];
    bool blocked = false;
    walkability_reason blocked_reason = walkability_clear;
    G1FootprintWork work;
};

static inline G1FootprintBudget g1_footprint_budget()
{
    return G1FootprintBudget{};
}

static inline bool g1_foot_contact_schedule_build(
    G1FootContactSchedule& output,
    const slice2d<bool> database_contacts,
    const slice1d<int> range_starts,
    const slice1d<int> range_stops,
    int current_frame,
    bool current_left_contact,
    bool current_right_contact,
    float dt,
    float trajectory_sample_time,
    char* error,
    int error_capacity);

static inline G1FootprintStatus g1_footprint_observe_v2(
    G1FootprintObservation& output,
    const G1FootprintBudget& limits,
    const heightfield& field,
    const walkability_grid& grid,
    const G1CommandSnapshot& command,
    const G1FootContactSchedule& contacts,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity);
```

Define the implementation inline like the existing terrain/support helpers. It assigns `output` only for `G1FootprintOk`; status text is diagnostic only and never drives branching.

- [ ] **Step 4: Implement exact probe transport and conservative corridor envelopes**

`g1_foot_contact_schedule_build` independently locates `current_frame` in one
half-open database range. For horizon `i`, compute the same checked
`ceilf(RN32(i * trajectory_sample_time) / dt)` offset as Task 2, clamp only to
that range's `stop - 1`, and copy both database contact bits. It builds a local
schedule and assigns only after all four indices are valid; no route heading or
travel mode enters this helper.

For each named leg and probe, materialize current world geometry from the ankle transform:

```cpp
current_sphere = global_positions(config.ankle) +
    quat_mul_vec3(global_rotations(config.ankle),
                  config.foot_sphere_centers_local[probe]);
current_sole = global_positions(config.ankle) +
    quat_mul_vec3(global_rotations(config.ankle),
                  config.sole_points_local[probe]);
```

First compute a root-transported candidate for samples one through three by
preserving each probe's offset from predicted root sample zero and transporting
sphere and sole independently with each predicted root transform:

```cpp
const vec3 root_local = quat_inv_mul_vec3(
    command.predicted_root_rotations[0],
    current_sphere - command.predicted_root_positions[0]);
transported[sample] = command.predicted_root_positions[sample] +
    quat_mul_vec3(command.predicted_root_rotations[sample], root_local);
```

Set predicted sample zero to the exact current sphere/sole bits. Validate the
two supplied current-contact bits and require them to equal schedule sample
zero before assigning either schedule or observation. Walk samples in order
with one per-probe contact anchor. While contact remains true, retain the
anchor's exact world bits. A `true -> false` edge releases into that sample's
root-transported geometry. During `false` samples use the root-transported
swing geometry. At the first `false -> true` edge, use that sample's transported
geometry as the landing geometry and new world-space contact anchor; later
contact samples retain its exact bits until another release. This is the only
phase logic and is identical in every travel direction.

For every sole probe at every sample, query the exact G1HF/v2 surface at that
sample's X/Z and retain its status/sample. For a currently non-contact foot,
select the first `false -> true` edge in samples `1..3`; do not invent a
landing when no edge exists. At the selected sample, compute the sole centroid
with checked binary64 sums and one final binary32 rounding, query its exact
G1HF/v2 surface, and use the four already-queried predicted sole probes. Let
each sampled probe point be `(predicted_x, sampled_height, predicted_z)`.
Against the center tangent plane, compute the maximum absolute binary64
dot-product residual.
Publish the selected center query status, exact surface, horizon, and
walkability class even when the patch is not ready. `landing_patch_ready` is
true only when every query is valid, the swept cells are class 1, and that
maximum is at most exact binary32 `0.005f`
(`0x3ba3d70a`). A discontinuous upper/lower footprint remains an observed but
unready landing; never average its levels.

For each of the 24 consecutive probe segments:

- call `walkability_sweep(grid, field, start, stop, 0.02f)` once;
- map `walkability_out_of_bounds` to `OutsideDomain`, malformed/nonfinite to the matching global error, and blocked-cell results to `candidate.blocked=true` while retaining the first blocked reason;
- compute a checked XZ AABB expanded by `0.02f`, convert it to a clamped node window with the existing checked floor/ceil helpers, add a one-node halo on every in-domain side, and preflight its node count against the remaining aggregate budget;
- enumerate the window in `z` then `x`, validate each visited height, and aggregate node heights as a conservative envelope for continuous fixed-diagonal cells touched by the swept probe; and
- count every sweep and height-node visit before assigning the candidate.

Sample the current root, all eight current sole points, and all 24 future
phase-aware sole points with `g1_surface_query_v2`; the two possible selected
landing-centroid queries raise the fixed maximum to 35.
Compute each split in binary64 as the maximum absolute difference between root
height and current/corridor height extrema. Set `multilevel` exactly when the
maximum is at least `0.04`. Do not replace or rewrite the four matcher terrain
values.

- [ ] **Step 5: Run strict, fast-caller parity, and sanitizer GREEN**

```bash
mkdir -p /tmp/g1-footprint
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_g1_footprint_runtime.cpp -o /tmp/g1-footprint/test-strict
/tmp/g1-footprint/test-strict --parity > /tmp/g1-footprint/strict.txt
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. tests/cpp/test_g1_footprint_runtime.cpp -o /tmp/g1-footprint/test-fast
/tmp/g1-footprint/test-fast --parity > /tmp/g1-footprint/fast.txt
cmp /tmp/g1-footprint/strict.txt /tmp/g1-footprint/fast.txt
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero -fno-sanitize-recover=all -I. tests/cpp/test_g1_footprint_runtime.cpp -o /tmp/g1-footprint/test-san
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 /tmp/g1-footprint/test-san
! rg -n 'strafe|forward_mode|lateral_mode' g1_footprint_runtime.h
```

Expected: all tests exit zero, strict and fast parity records are byte-identical, the sanitizer is clean, and the mode-token scan finds nothing.

- [ ] **Step 6: Commit the direction-agnostic footprint observer**

```bash
git add g1_footprint_runtime.h tests/cpp/test_g1_footprint_runtime.cpp
git commit -m "feat: observe swept G1 foot terrain footprints"
```

---

### Task 5: Compose the Footprint-Aware Reversible IK Transaction

**Files:**
- Create: `g1_ik_runtime.h`
- Create: `tests/cpp/compile_g1_ik_production.cpp`
- Create: `tests/cpp/compile_g1_ik_seam_negative.cpp`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Consumes: `G1FootprintObservation`, actual-center clearance APIs, named-leg IK, exact surface locks, and the immutable 41-entry lift ladder.
- Produces: `G1IkState`, `G1FootIkState`, `G1IkFrameTransaction`,
  `G1FootFrameResult`, `G1IkFrameResult`, `G1IkSafeStopHandoff`,
  `g1_ik_state_reset`, checked `g1_ik_frame_begin`,
  `g1_ik_frame_stage_foot`, `g1_ik_frame_finish`, the convenience wrapper
  `g1_ik_frame_evaluate`, `g1_ik_safe_stop_handoff`, and
  `g1_ik_stop_reason_name`.
- The transaction receives no mutable command or heading owner. The outer controller alone decides whether scratch pose/state is committed.

- [ ] **Step 1: Write real-stage, footprint, and rollback RED tests**

Extend `tests/cpp/test_g1_ik.cpp` under `G1_IK_ENABLE_TEST_SEAMS` to:

- invoke all 41 real production stages from one immutable support-retargeted
  baseline through the diagnostic-only test wrapper and require the full
  selector to choose the first stage with `G1ClearanceOk`, nonnegative
  binary64 margin, and passed controller constraints. The wrapper returns no
  staged positions, rotations, mutable directive, trace, or private geometry;
- require selected lift bits, materialized command-Y bits, twelve actual FK
  sphere-center bits, clearance result, and published work to equal the
  independently invoked stage diagnostic. For the selected public output,
  recompute checked FK and materialize the four centers independently of the
  runtime helper before comparing all twelve words;
- construct a nontrivially rotated pose and feed the real
  `g1_footprint_observe_v2` output directly into `g1_ik_frame_begin` under both
  strict and optimized-fast callers. Require all eight Task-4 current-center
  words to match Task 5 recomputation exactly; a hand-built footprint is not
  sufficient evidence for this cross-module boundary;
- supply a footprint whose current probe bits disagree by one ULP with FK and require transactional `InvalidInput`;
- supply `footprint.blocked=true` and require successful diagnostic safe-stop with unchanged pose/state/history;
- supply a root/foot split of `0.32` with valid class-1 probes and require it to proceed through planting/staging rather than being flattened or treated as blocked;
- supply a down-step with `landing_expected=true`,
  `landing_patch_ready=true`, and a selected surface `0.32 m` below the
  current sole. Require that lower surface height/normal to replace the swing
  base target before candidate zero and require each ladder entry to add to
  that lower base, never to the old sole Y. Drive the real per-foot stage, not
  only begin/materialization, against terrain consistent with that height;
- mirror the fixture for an up-step, and require the predicted landing X/Z to
  become the staged swing base X/Z. Require `target.surface.point` to equal the
  predicted landing X/Z and exact sampled surface height, require its normal to
  equal the sampled landing normal, and require the sole-center base Y to be
  the one-rounding surface-height-plus-swing-clearance value. Drive the real
  per-foot stage against the corresponding upper terrain;
- set `landing_expected=true` and `landing_patch_ready=false` and require a
  successful `G1IkStopLandingPatchUnavailable` safe stop with unchanged
  pose/state/history; set `landing_expected=false` and require the normal
  actual swing target path;
- produce every clearance status through real strict-kernel inputs:
  `OutsideDomain`, `BudgetExceeded`, and `Uncertified` continue to the next
  lift, while `InvalidInput`, `InvalidField`, and `ArithmeticFailure` abort
  unchanged. Because the certified validator leaves its output unchanged on
  every non-`Ok` status, those finite calls publish zero margins/work and
  contribute zero to `total_clearance_work`; never relabel work from an `Ok`
  call as work from a non-`Ok` status;
- independently make an early real candidate return `G1ClearanceOk` with a
  negative lower margin, and exercise every checked controller
  reach/correction/residual/invariance predicate with genuine stage inputs or
  the same production predicate used by the real stage, then require a later
  real ladder entry to pass. Each is candidate-local rejection, not a frame
  abort. Test-only result mutation is forbidden;
- stage both feet as swing feet with different winning indices. Require the
  left winner to remain in outer scratch while the right candidates run, and
  require both selected rotations and both twelve-word endpoint records to
  survive the final FK equality check;
- require recorded contact to bypass the ladder, retain its world-space lock across a root level change, and use `target.sole_center` until checked release; and
- snapshot a sentinel `G1CommandSnapshot` around every call and require byte identity, proving the IK API cannot alter requested travel or heading; and
- exercise `g1_ik_safe_stop_handoff` with positive and negative signed zeros.
  When unlatched, every requested-velocity word must be copied exactly. When
  latched, only X/Z become canonical `+0.0f`; Y retains its exact input bits,
  including `-0.0f`.

- [ ] **Step 2: Run the missing-runtime RED with a separate strict kernel**

```bash
mkdir -p /tmp/g1-ik-footprint-task5-red
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -frounding-math -I. -c g1_clearance.cpp -o /tmp/g1-ik-footprint-task5-red/kernel.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp -o /tmp/g1-ik-footprint-task5-red/test.o
```

Expected: the kernel builds; the test compile fails because `g1_ik_runtime.h` and the footprint-aware frame interfaces are absent.

- [ ] **Step 3: Define fixed state and diagnostics**

Create these public owners in `g1_ik_runtime.h`:

```cpp
#pragma once

#include "g1_clearance.h"
#include "g1_footprint_runtime.h"
#include "g1_ik.h"

#include <cstdint>
#include <cstring>

enum G1IkStopReason
{
    G1IkStopNone = 0,
    G1IkStopFootprintBlocked,
    G1IkStopFootprintOutsideDomain,
    G1IkStopFootprintBudgetExceeded,
    G1IkStopLandingPatchUnavailable,
    G1IkStopTargetUnreachable,
    G1IkStopNoSwingCandidate,
    G1IkStopPoseClearanceRejected,
};

static constexpr uint32_t G1SwingLiftCandidateCount = 41;
static constexpr uint32_t G1SwingNoCandidate = UINT32_MAX;

struct G1SwingCandidateDiagnostic
{
    uint32_t candidate_index = G1SwingNoCandidate;
    uint32_t lift_bits = 0;
    uint32_t materialized_command_y_bits = 0;
    uint32_t actual_sphere_center_bits[4][3] = {};
    G1ClearanceStatus clearance_status = G1ClearanceInvalidInput;
    bool controller_constraints_passed = false;
    bool clearance_certified = false;
    double lower_margin_m = 0.0;
    double witness_upper_margin_m = 0.0;
    G1ClearanceWork clearance_work;
};

struct G1SwingSelectionDiagnostic
{
    uint32_t candidates_evaluated = 0;
    uint32_t selected_index = G1SwingNoCandidate;
    G1SwingCandidateDiagnostic selected;
    G1ClearanceWork total_clearance_work;
};

struct G1FootIkState
{
    G1FootLockState lock;
    G1SwingHistory swing;
};

struct G1IkState
{
    bool initialized = false;
    G1FootIkState feet[2];
};

struct G1FootFrameResult
{
    bool recorded_contact = false;
    G1FootTarget target;
    G1SwingSelectionDiagnostic swing_selection;
    G1SwingClearanceValidation defensive_swing;
    G1LegSolveResult position;
    G1FootOrientationResult orientation;
};

struct G1IkFrameResult
{
    bool applied = false;
    bool safe_stop_requested = false;
    G1IkStopReason stop_reason = G1IkStopNone;
    float max_correction_radians = 0.0f;
    G1FootFrameResult feet[2];
};

struct G1IkFrameTransaction
{
    bool initialized = false;
    uint32_t next_foot = 0;
    G1IkState candidate_state;
    G1IkFrameResult candidate_result;
};

struct G1IkSafeStopHandoff
{
    vec3 applied_velocity;
    bool cancel_planar_inertia = false;
    bool force_search = false;
};

static inline bool g1_ik_safe_stop_handoff(
    G1IkSafeStopHandoff& output,
    bool latched,
    vec3 requested_velocity,
    char* error,
    int error_capacity);

static inline const char* g1_ik_stop_reason_name(G1IkStopReason reason);

static inline bool g1_ik_state_reset(
    G1IkState& output,
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> bone_parents,
    char* error,
    int error_capacity);

static inline bool g1_ik_frame_begin(
    G1IkFrameTransaction& transaction,
    array1d<vec3>& scratch_positions,
    array1d<quat>& scratch_rotations,
    const G1IkState& state,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const slice1d<bool> contacts,
    const heightfield& field,
    const G1FootprintObservation& footprint,
    bool enabled,
    float dt,
    char* error,
    int error_capacity);

static inline bool g1_ik_frame_stage_foot(
    G1IkFrameTransaction& transaction,
    array1d<vec3>& scratch_positions,
    array1d<quat>& scratch_rotations,
    uint32_t foot_index,
    const slice1d<int> bone_parents,
    const slice1d<bool> contacts,
    const heightfield& field,
    const G1FootprintObservation& footprint,
    bool enabled,
    float dt,
    char* error,
    int error_capacity);

static inline bool g1_ik_frame_finish(
    G1IkState& output_state,
    G1IkFrameResult& output_result,
    G1IkFrameTransaction& transaction,
    array1d<vec3>& scratch_positions,
    array1d<quat>& scratch_rotations,
    const slice1d<int> bone_parents,
    const heightfield& field,
    float dt,
    char* error,
    int error_capacity);

static inline bool g1_ik_frame_evaluate(
    array1d<vec3>& output_positions,
    array1d<quat>& output_rotations,
    G1IkState& state,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const slice1d<bool> contacts,
    const heightfield& field,
    const G1FootprintObservation& footprint,
    bool enabled,
    float dt,
    G1IkFrameResult& result,
    char* error,
    int error_capacity);
```

Define the immutable ladder exactly:

```cpp
static constexpr uint32_t G1SwingLiftCandidateBits[
    G1SwingLiftCandidateCount] = {
    0x00000000u, 0x3b03126fu, 0x3b83126fu, 0x3bc49ba6u,
    0x3c03126fu, 0x3c23d70au, 0x3c449ba6u, 0x3c656042u,
    0x3c83126fu, 0x3c9374bcu, 0x3ca3d70au, 0x3cb43958u,
    0x3cc49ba6u, 0x3cd4fdf4u, 0x3ce56042u, 0x3cf5c28fu,
    0x3d03126fu, 0x3d0b4396u, 0x3d1374bcu, 0x3d1ba5e3u,
    0x3d23d70au, 0x3d2c0831u, 0x3d343958u, 0x3d3c6a7fu,
    0x3d449ba6u, 0x3d4ccccdu, 0x3d54fdf4u, 0x3d5d2f1bu,
    0x3d656042u, 0x3d6d9168u, 0x3d75c28fu, 0x3d7df3b6u,
    0x3d83126fu, 0x3d872b02u, 0x3d8b4396u, 0x3d8f5c29u,
    0x3d9374bcu, 0x3d978d50u, 0x3d9ba5e3u, 0x3d9fbe77u,
    0x3da3d70au,
};
```

Entry `i` is `RN32(i/500 m)`, from `0x00000000` through `0x3da3d70a`.
Load bits with `memcpy`; never generate the ladder arithmetically.
Use a constexpr full-table ordering check in production and an independent
exact-rational test oracle for all 41 words; checking only count, endpoints, or
the production table against itself is insufficient.
The two diagnostic structs above are copied verbatim from the authoritative
certified-clearance contract; do not replace selected-candidate status/work or
aggregate work with a rejection mask. `g1_ik_safe_stop_handoff` validates
finite requested velocity into a local candidate, copies Y, canonicalizes zero
X/Z only when latched, requests one forced matcher search when latched, assigns
`output` only on success, and receives no heading owner.
`g1_ik_stop_reason_name` maps the enum in order to `none`,
`footprint-blocked`, `footprint-outside-domain`,
`footprint-budget-exceeded`, `landing-patch-unavailable`,
`target-unreachable`, `no-swing-candidate`, and `pose-clearance-rejected`; an
unknown value returns `invalid` for diagnostics and fails validation before
state publication.

Under `G1_IK_ENABLE_TEST_SEAMS`, expose exactly one diagnostic-only wrapper:

```cpp
static inline bool g1_ik_stage_swing_candidate_for_test(
    G1SwingCandidateDiagnostic& diagnostic,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const G1FootTarget& target,
    uint32_t candidate_index,
    float dt,
    char* error,
    int error_capacity);
```

It calls the real private stage once, discards staged positions/rotations, and
publishes only the diagnostic on success. Do not expose directive setters,
trace storage, mutable seam state, staged pose arrays, adapters, or any second
test symbol. Real geometry/status fixtures and shared production predicates
provide branch coverage without laundering results.

`g1_ik_frame_begin` performs the complete common validation, initializes a
local transaction, and copies the support-retargeted baseline into caller-owned
working scratch only after preflight. `g1_ik_frame_stage_foot` requires
`foot_index == transaction.next_foot`, implements the immutable per-foot
snapshot/winner composition below, and increments only after a complete foot
result. `g1_ik_frame_finish` requires both feet staged, performs final FK,
endpoint equality, defensive certificates, and checked histories, then assigns
the output state/result. `g1_ik_frame_evaluate` is exactly the checked sequence
begin, foot zero, foot one, finish; it adds no alternate behavior. Task 6 uses
the split calls so its real first-foot and second-foot checkpoints cannot be
simulated no-ops.

- [ ] **Step 4: Implement checked reset and footprint validation**

`g1_ik_state_reset` validates shapes/configuration, computes actual FK sphere centers, resets both locks and histories into a local candidate, and assigns only after both feet pass.

At the start of `g1_ik_frame_evaluate`, validate exact 25 Hz, complete baseline shapes, initialized state, G1HF/v2, footprint work bounds, and all current footprint probe bits against sphere centers recomputed from baseline FK. `footprint.blocked` returns success with `safe_stop_requested=true`, `stop_reason=G1IkStopFootprintBlocked`, unchanged pose/state/history, and completed diagnostics. Outside-domain and budget statuses remain outer-controller finite rejections; malformed footprint data is a controlled error.

Call the existing checked `g1_foot_lock_update` first for both feet. A recorded
contact remains authoritative: its world-space `target.sole_center` and
surface normal survive root-level changes and never enter the lift ladder. For
a non-contact foot:

- when `landing_expected` is false, use the actual lock/update output as the
  base target;
- when `landing_expected` is true but `landing_patch_ready` is false, return a
  successful safe-stop result with
  `G1IkStopLandingPatchUnavailable` and leave all caller-owned values
  unchanged; and
- when both are true, require selected status `G1SurfaceQueryValid`, class 1,
  and a valid normal. Set `target.surface.point` to the predicted landing
  centroid X/Z and exact `predicted_landing_surface.height`, copy the predicted
  landing X/Z into the base sole center, set base Y to
  `RN32(double(predicted_landing_surface.height) +
  double(config.swing_clearance_m))` through a checked helper, and replace the
  base orientation normal with `predicted_landing_surface.normal`.

`landing_sample` remains diagnostic lookahead, but
`predicted_landing_sole_center.x/z` is the authoritative future swing target
when the patch is ready. The actual current FK centers remain authoritative
for every clearance certificate. This lets the swing foot advance across and
lower to a `0.32 m` lower landing or rise to an upper landing while a planted
foot stays fixed in world space; it cannot certify future height at the old
upper-platform X/Z.

- [ ] **Step 5: Implement one real 41-stage swing transaction**

Process feet in fixed left-then-right order. Before foot zero, copy the current
outer scratch pose into one immutable per-foot snapshot. Evaluate all foot-zero
candidates from that same snapshot; on the first winner, move its already-solved
pose and result into outer scratch. Before foot one, take a new immutable
per-foot snapshot from that updated outer scratch, then evaluate all foot-one
candidates from it and move its winner into outer scratch. Thus candidates for
one foot never accumulate across ladder entries, while the second foot cannot
discard the first foot's winner.

For each non-contact foot, iterate indices `0..40`. Every iteration starts from
that foot's immutable snapshot and performs this exact production order, where
`base_sole_center` is the checked landing-aware base defined in Step 4:

```cpp
status = g1_apply_swing_lift_y(
    materialized_y, base_sole_center.y, lift, error, capacity);
// On Ok: named bounded position IK, surface-normal foot orientation,
// controller correction checks, then checked full FK.
status = g1_swing_clearance_validate(
    validation,
    g1_swing_foot_clearance_budget(),
    baseline_state.swing,
    field,
    config,
    actual_current_sphere_centers,
    false,
    dt,
    error,
    capacity);
```

Accept only exact `G1ClearanceOk`, nonnegative
`validation.lower_margin_m`, and passed controller constraints. Advance to the
next ladder entry on `OutsideDomain`, `BudgetExceeded`, `Uncertified`, an
`Ok` certificate with negative lower margin, or a checked controller-local
reach, correction-limit, residual, or endpoint-invariance rejection. The
strict validator assigns `G1SwingClearanceValidation` only on `Ok`: copy and
checked-add work only from those published `Ok` results, including `Ok`
certificates rejected by a controller predicate or negative margin. Record the
explicit finite non-`Ok` status with default zero margin/work; never fabricate
attempted work. Abort the whole transaction unchanged
only for `InvalidInput`, `InvalidField`, `ArithmeticFailure`, or a malformed
checked controller call. Stop at the first pass, move its already-solved pose/result
and complete `G1SwingCandidateDiagnostic` into outer scratch, and do not solve
again. If all 41 reject, return successful safe-stop with unchanged accepted
state, `selected_index=G1SwingNoCandidate`, default selected fields, and exact
aggregate published-`Ok` work.

After both feet compose sequentially in outer scratch, run one checked FK,
compare both feet's selected endpoint bits, run fresh actual-center defensive
certificates, and commit checked histories only into scratch state. Any later
failure rolls the accepted frame back. The split API's transaction and pose
arrays are explicitly caller-owned working scratch and may remain dirty on a
returned global error; Task 6 must discard them with its complete working
state. `g1_ik_frame_evaluate` remains fully transactional because its scratch
is local and is never published on failure.

- [ ] **Step 6: Run strict, fast-caller, seam, and sanitizer GREEN**

The `--parity` mode below supplies one fixed set of already-materialized
binary32 endpoint bits to both caller builds and compares only strict-kernel
results and authoritative diagnostic serialization. It does not independently
stage controller IK in the strict and fast callers, so caller rounding is
never mistaken for a strict-kernel parity failure.

```bash
mkdir -p /tmp/g1-ik-footprint-task5
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp -o /tmp/g1-ik-footprint-task5/kernel.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-footprint-task5/test-strict.o
g++ /tmp/g1-ik-footprint-task5/test-strict.o \
  /tmp/g1-ik-footprint-task5/kernel.o \
  -o /tmp/g1-ik-footprint-task5/test-strict
/tmp/g1-ik-footprint-task5/test-strict
/tmp/g1-ik-footprint-task5/test-strict --parity \
  > /tmp/g1-ik-footprint-task5/strict.txt

g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-footprint-task5/test-fast.o
g++ /tmp/g1-ik-footprint-task5/test-fast.o \
  /tmp/g1-ik-footprint-task5/kernel.o \
  -o /tmp/g1-ik-footprint-task5/test-fast
/tmp/g1-ik-footprint-task5/test-fast
/tmp/g1-ik-footprint-task5/test-fast --parity \
  > /tmp/g1-ik-footprint-task5/fast.txt
cmp /tmp/g1-ik-footprint-task5/strict.txt \
  /tmp/g1-ik-footprint-task5/fast.txt

g++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-fast-math -ffp-contract=off \
  -frounding-math -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-footprint-task5/kernel-san.o
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -DG1_IK_ENABLE_TEST_SEAMS -I. \
  -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-footprint-task5/test-san.o
g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  /tmp/g1-ik-footprint-task5/test-san.o \
  /tmp/g1-ik-footprint-task5/kernel-san.o \
  -o /tmp/g1-ik-footprint-task5/test-san
ASAN_OPTIONS=detect_leaks=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/g1-ik-footprint-task5/test-san

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/compile_g1_ik_production.cpp \
  -o /tmp/g1-ik-footprint-task5/production.o
! nm -C /tmp/g1-ik-footprint-task5/production.o | \
  rg 'g1_ik_stage_swing_candidate_for_test'
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
     -c tests/cpp/compile_g1_ik_seam_negative.cpp \
     -o /tmp/g1-ik-footprint-task5/negative.o \
     2>/tmp/g1-ik-footprint-task5/negative.stderr; then
  echo 'ERROR: private IK seam compiled without G1_IK_ENABLE_TEST_SEAMS' >&2
  exit 1
fi
rg 'g1_ik_stage_swing_candidate_for_test' \
  /tmp/g1-ik-footprint-task5/negative.stderr
```

Expected: first-pass and all-41 fixtures pass, footprint mismatch rolls back, class-1 multilevel proceeds, blocked footprint stops, command bytes remain identical, and sanitizers report nothing.

- [ ] **Step 7: Commit the reversible footprint-aware frame transaction**

```bash
git add g1_ik_runtime.h tests/cpp/test_g1_ik.cpp \
  tests/cpp/compile_g1_ik_production.cpp \
  tests/cpp/compile_g1_ik_seam_negative.cpp
git commit -m "feat: compose footprint-aware G1 terrain IK"
```

---

### Task 6: Integrate Footprint Observation, IK Commit, and Heading-Preserving Safe Stop

**Files:**
- Create: `g1_frame_transaction.h`
- Create: `g1_controller_frame_runtime.h`
- Create: `tests/cpp/test_g1_frame_transaction.cpp`
- Create: `tests/cpp/test_g1_frame_transaction_production.cpp`
- Create: `tests/cpp/compile_g1_frame_transaction_runner_negative.cpp`
- Modify: `g1_controller_state.h`
- Modify: `g1_ik_runtime.h`
- Modify: `scene_switch.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `tests/cpp/test_g1_ik.cpp`
- Modify: `tests/cpp/test_scene_switch.cpp`

**Interfaces:**
- Consumes: Tasks 2, 4, and 5 command, footprint, and IK interfaces.
- Produces a complete accepted/working frame transaction, accepted
  footprint/IK state, a next-update planar safe-stop/forced-search latch,
  startup-only exact `MM_IK=0|1`, an immutable value-only accepted diagnostic
  record, atomic accepted/working reset and scene switch, and the final
  rendered accepted pose.
- A finite rejection leaves every accepted matcher, inertializer, simulation,
  support, contact, route, timer, pose, footprint, IK, and history byte
  unchanged. Only immutable requested intent/heading, rejection diagnostics,
  the one-frame latch, and the fresh attempt presentation frame may publish
  outside that state.
- The live production stage runner is one named non-capturing function linked
  into both the controller and its production test. Every behavior-relevant
  mode, tuning value, and sampled device input reaches it through one typed,
  immutable context; it has no hidden mutable global, Raylib-input, log,
  model, cleanup, accepted-state, or publication access.

- [ ] **Step 1: Write controller ownership and ordering RED tests**

Extend `tests/cpp/test_g1_controller_state.cpp` to poison and reset:

- accepted `G1FootprintObservation` and its exact status;
- accepted `G1IkState`, candidate/accepted local and global pose arrays;
- `G1IkFrameResult`, accepted/candidate `G1PoseClearance`, exact candidate clearance status, rejection flag, and safe-stop latch; and
- the existing `G1CommandSnapshot` before/after finite rejection, global error,
  successful IK, and scene reset. `MM_IK` is startup-only; there is no live
  toggle claim.
- publication `presentation_frame` and accepted-diagnostic adjustment,
  clamping, IK, and effective-terrain-weight values, including nonfinite/range
  validity mutations and finite/global rollback sentinels.
- exact `MM_SEARCHT` bits in `search_time`, `search_timer`, and
  `force_search_timer` for both independently reset states, plus malformed,
  nonfinite, and out-of-range startup rejection before either state installs.

Create `tests/cpp/test_g1_frame_transaction.cpp`. Give every scalar a distinct
bit pattern and every array element a distinct digest contribution. Inject a
finite failure after each of these stages: input/route command, matcher search,
inertialization, simulation update, support observation, support retarget,
contact update, footprint observation, first-foot IK, second-foot IK, final
FK, and pose certificate. After every injection require a byte-identical
accepted-state digest and exact sentinel values for every scalar/array owner.
The only changed values may be:

```text
publication.requested_intent
publication.rejection
publication.ik_safe_stop_latched
publication.presentation_frame
```

Inject a global error at the same stages and require controlled failure with
the accepted state still identical. Inject success and require the complete
working state to publish together. A mutation test that omits any scalar or
array from deep copy/swap must fail at least one sentinel case.

Add a retry whose working-state buffers have the exact required shapes and are
fully disjoint but whose scalar/value semantics are deliberately poisoned,
including nonfinite values. The coordinator must accept that destination as
storage, overwrite it from accepted state, and then produce the same result as
a clean working destination. A pre-copy rejection caused by dirty working
semantics fails this test.

The production coordinator in `g1_frame_transaction.h` owns one
`G1FrameRuntime` privately and exposes only its `working_state` to stage code
between the initial checked copy and the final checked swap. The runtime owns
accepted state, working state, publication, and the last accepted value-only
diagnostic as one reset/switch unit. The coordinator invokes a checkpoint
after every named stage above. Under
`G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM` only, a registered hook may request
`continue`, one of the real finite rejections, or a global error at a
checkpoint; the production build has no hook storage or branch. The stage
runner's function type accepts only mutable working state, transaction
scratch, and one const typed external context—never mutable accepted state or
publication. The live controller update calls this coordinator with the named
production runner declared in `g1_controller_frame_runtime.h`, not a parallel
test simulation or a capturing lambda.

Create `tests/cpp/test_g1_frame_transaction_production.cpp` against that exact
coordinator and the production runner symbol linked from `controller.cpp`
compiled with `G1_CONTROLLER_NO_MAIN`. Exercise live, sequential, flat,
terrain, route, and scene-cycle modes with explicit immutable input/tuning
snapshots. Authenticate exact lowering for validated `MM_SEARCHT`, the
existing `MM_HALFLIFE` and `MM_SIMROT_HL` startup values, strafe, scripted
azimuth, and autodrive inputs;
none may be reread inside the runner. Inject both finite and global failure
after every real checkpoint
and prove the accepted state and accepted diagnostic are byte-identical and
only the publication whitelist changes. The negative compile fixture has three
translation-unit modes and each must fail: a runner that adds mutable `void*`,
one that adds `G1FramePublication*`, and one that adds
`g1_controller_state* accepted_state`. A narrow source guard also rejects
direct `accepted_state` or publication use between the coordinator call's
stage markers and rejects runner-body references to `getenv`, Raylib input,
GUI, camera/model/log/cleanup objects, `g_frame`, or `g_log`. Add a dedicated
finite injection immediately after
`G1FrameStageInputRouteCommand`: require
`scratch.requested_intent_ready == true`, exact requested-intent bits in the
finite publication, `G1FrameTransactionFiniteRejected`, and an unchanged
accepted-state digest. Add a six-frame down-step regression
spanning `{false,false,true,true}`: at least four successive frames across the
edge accept, route/root XZ advance, no safe-stop latch appears, the swing base
advances toward the predicted lower landing, and the first committed contact
target has the exact predicted landing X/Z, lower surface height, and normal.
Add three genuine production-IK safe-stop cases, not coordinator checkpoint
injections or hand-built transactions. First, make successful
`g1_ik_frame_begin` request a footprint-blocked or unavailable-landing-patch
stop and prove neither foot stage runs. Second, let begin continue and make
foot 0 produce a real target-unreachable or no-swing-candidate stop; prove
`next_foot == 1` and that foot 1 does not run. Third, let foot 0 complete
successfully and make foot 1 produce a real stop; prove `next_foot == 2` and
that finish does not run. At each checkpoint the matching typed rejection
snapshot must succeed, both other checkpoint values and every poisoned
stage/completed-foot field must fail without changing the output. The complete
finite diagnostic stage must respectively be
`G1FrameStageFootprintObservation`, `G1FrameStageFirstFootIk`, or
`G1FrameStageSecondFootIk`.

Compile the `G1_CONTROLLER_NO_MAIN` controller object under
`-Wall -Wextra -Werror`. Its source/preprocessor guard must prove that every
file-scope static root not reachable from the runner—including every main-only
parser, startup validator, render/debug helper, and its main-only helper
closure—is excluded, while the named runner and every pure dependency it calls
remain compiled. Any unused static root warning or hiding the runner behind the
guard fails the production-authenticity test.

Add a controller-order fixture requiring:

```text
g1_controller_state_copy(working, accepted)
< deterministic_route_command/input snapshot
< matcher search/inertialization/simulation
< support_pose_apply
< contact update
< g1_ik_checked_forward_kinematics
< g1_footprint_observe_v2
< g1_ik_frame_begin
< checked begin-time safe-stop-result inspection
< g1_ik_frame_stage_foot(0)
< g1_ik_frame_stage_foot(1)
< g1_ik_frame_finish
< g1_measure_pose_clearance
< accepted pose/state assignment
< deterministic_log.write
```

The production runner must not contain or call `g1_ik_frame_evaluate`; that
one-call wrapper exists only for non-controller callers and focused tests.

Require `g1_ik_safe_stop_handoff` to zero only X/Z desired command and planar inertia on the next update; it may not receive or return a quaternion, touch Y components, or clear desired heading.

Extend `tests/cpp/test_scene_switch.cpp` with accepted/working pair sentinels.
For reset and every scene-load, state-reset, and model-load failure, require
both state digests and buffer identities, publication, accepted diagnostic,
scene, model, index, and route cursors to remain unchanged. On success require
semantically equal, completely disjoint accepted/working states, identical
route cursors, and cleared rejection/latch state. Authenticate the zero-travel
command, support-retarget baseline, checked FK, `G1FootprintOk`, initialized
IK, `G1ClearanceOk`, and minimum `>= -0.01` independently for each member of
the pair; poisoning either candidate gate must preserve the entire live unit.

- [ ] **Step 2: Run the controller-state RED**

```bash
mkdir -p /tmp/g1-footprint-controller-red
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -fno-fast-math -ffp-contract=off -frounding-math -I. -c g1_clearance.cpp -o /tmp/g1-footprint-controller-red/kernel.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. -c tests/cpp/test_g1_controller_state.cpp -o /tmp/g1-footprint-controller-red/test.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_g1_frame_transaction.cpp \
  -o /tmp/g1-footprint-controller-red/transaction.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM -I. \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o /tmp/g1-footprint-controller-red/transaction-production.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_scene_switch.cpp \
  -o /tmp/g1-footprint-controller-red/scene-switch.o
```

Expected: the strict kernel builds; the four positive test compiles fail on
the missing owners, runtime aggregate, production coordinator/checkpoints,
pair-aware reset/switch, accepted diagnostic, and exact runner boundary. After
the production header and positive coordinator compile exist, compile each
negative mode and accept its failure only when stderr specifically identifies
the forbidden conversion to `G1FrameStageRunner`; a missing header, undeclared
symbol, or unrelated diagnostic is not valid negative-API evidence.

- [ ] **Step 3: Extend state and implement an allocation-free whole-state copy**

Add these owners to `g1_controller_state`:

```cpp
G1FootprintStatus footprint_status = G1FootprintInvalidInput;
G1FootprintObservation footprint;
G1IkState ik;
G1IkFrameResult ik_frame;
array1d<vec3> ik_bone_positions;
array1d<quat> ik_bone_rotations;
array1d<vec3> ik_global_bone_positions;
array1d<quat> ik_global_bone_rotations;
array1d<vec3> ik_candidate_bone_positions;
array1d<quat> ik_candidate_bone_rotations;
array1d<vec3> ik_candidate_global_bone_positions;
array1d<quat> ik_candidate_global_bone_rotations;
G1PoseClearance ik_clearance;
G1PoseClearance ik_candidate_clearance;
G1ClearanceStatus ik_candidate_clearance_status = G1ClearanceInvalidInput;
bool ik_candidate_rejected = false;
```

The safe-stop latch does not belong to `g1_controller_state`; it is stored only
in the publication channel below. Swap every listed owner.

Add this checked API to `g1_controller_state.h`:

```cpp
static inline bool g1_controller_state_copy(
    g1_controller_state& output,
    const g1_controller_state& input,
    char* error,
    int error_capacity);
```

It first validates every source scalar and source shape. For the destination it
validates storage only: exact shapes, stable identities, non-null buffers,
complete source/destination and cross-array disjointness, and byte-count
multiplication. It must not validate stale destination scalar/value semantics
before overwrite. Only after that complete preflight does it perform the exact,
unconditional array `memcpy` operations and scalar/aggregate assignments and
return success. It performs no semantic validation, `resize`, allocation, I/O,
or state-dependent branch after the first write. Add every new owner to
`g1_controller_state_swap`, copy, reset, and the test-only digest in the same
commit.

Do not install either reset state directly from this state-level helper. Step 4
defines the runtime aggregate and the only pair-aware reset/switch entry
points that may install these independently built states.

- [ ] **Step 4: Define the finite-rejection publication side channel**

Create `g1_frame_transaction.h`:

```cpp
#pragma once

#include "g1_controller_state.h"
#include "motion_match_log.h"
#include "route_runtime.h"

enum G1FrameRejectionStage
{
    G1FrameRejectNone = 0,
    G1FrameRejectFootprint,
    G1FrameRejectLandingPatch,
    G1FrameRejectIkCandidate,
    G1FrameRejectPoseCertificate,
};

struct G1FrameRejectionDiagnostic
{
    bool rejected = false;
    G1FrameRejectionStage stage = G1FrameRejectNone;
    G1IkStopReason stop_reason = G1IkStopNone;
    bool attempted_footprint_available = false;
    G1FootprintStatus footprint_status = G1FootprintInvalidInput;
    G1FootprintObservation attempted_footprint;
    bool attempted_ik_available = false;
    G1IkFrameResult ik_frame;
    bool attempted_pose_available = false;
    G1ClearanceStatus pose_status = G1ClearanceInvalidInput;
    G1PoseClearance pose_clearance;
};

struct G1FramePublication
{
    G1CommandIntent requested_intent;
    G1FrameRejectionDiagnostic rejection;
    bool ik_safe_stop_latched = false;
    int presentation_frame = 0;
};

struct G1FrameAcceptedDiagnostic
{
    bool ready = false;
    int presentation_frame = 0;
    int scene_frame = 0;
    deterministic_route_sample route;
    traversability_diagnostics traversal;
    float query[31] = {};
    int query_database_frame = -1;
    int query_range = -1;
    int selected_database_frame = -1;
    terrain_centerline_snapshot terrain_query = {};
    motion_match_pose_diagnostic raw_selected;
    motion_match_pose_diagnostic inertialized;
    motion_match_pose_diagnostic support_retargeted;
    motion_match_pose_diagnostic rendered;
    bool matching_enabled = true;
    bool adjustment_enabled = true;
    bool clamping_enabled = true;
    bool ik_enabled = false;
    float effective_terrain_weight = 0.0f;
};

struct G1FrameRuntime
{
    g1_controller_state accepted_state;
    g1_controller_state working_state;
    G1FramePublication publication;
    G1FrameAcceptedDiagnostic accepted_diagnostic;
};

enum G1FrameTransactionStage
{
    G1FrameStageInputRouteCommand = 0,
    G1FrameStageMatcherSearch,
    G1FrameStageInertialization,
    G1FrameStageSimulationUpdate,
    G1FrameStageSupportObservation,
    G1FrameStageSupportRetarget,
    G1FrameStageContactUpdate,
    G1FrameStageFootprintObservation,
    G1FrameStageFirstFootIk,
    G1FrameStageSecondFootIk,
    G1FrameStageFinalFk,
    G1FrameStagePoseCertificate,
    G1FrameStageCount,
};

enum G1FrameStageOutcome
{
    G1FrameStageContinue = 0,
    G1FrameStageFiniteReject,
    G1FrameStageGlobalError,
};

enum G1FrameTransactionStatus
{
    G1FrameTransactionAccepted = 0,
    G1FrameTransactionFiniteRejected,
    G1FrameTransactionGlobalError,
};

struct G1FrameTransactionScratch
{
    bool prior_safe_stop_latched = false;
    bool requested_intent_ready = false;
    G1CommandIntent requested_intent;
    vec3 commanded_velocity;
    vec3 traversal_input;
    G1IkSafeStopHandoff safe_stop_handoff;
    bool force_search = false;
    deterministic_route_sample route_sample;
    traversability_diagnostics traversal;
    float query[31] = {};
    int query_database_frame = -1;
    int query_range = -1;
    int selected_database_frame = -1;
    terrain_centerline_snapshot terrain_query = {};
    motion_match_pose_diagnostic raw_selected_diagnostic;
    motion_match_pose_diagnostic inertialized_diagnostic;
    motion_match_pose_diagnostic support_retargeted_diagnostic;
    motion_match_pose_diagnostic rendered_diagnostic;
    G1FootContactSchedule contact_schedule;
    G1FootprintStatus footprint_status = G1FootprintInvalidInput;
    G1FootprintObservation footprint;
    G1IkFrameTransaction ik_transaction;
    G1ClearanceStatus pose_status = G1ClearanceInvalidInput;
    G1PoseClearance pose_clearance;
    G1FrameRejectionDiagnostic rejection;
    G1FrameAcceptedDiagnostic accepted_diagnostic_candidate;
    bool accepted_diagnostic_ready = false;
};

enum g1_test_mode
{
    G1_TestLive,
    G1_TestSequential,
    G1_TestFlat,
    G1_TestTerrain,
    G1_TestRoute,
    G1_TestSceneCycle,
};

struct G1FrameInputSnapshot
{
    vec3 move_stick;
    vec3 look_stick;
    float gait_target = 0.0f;
    float camera_zoom_axis = 0.0f;
    float scripted_azimuth_delta = 0.0f;
    bool desired_strafe = false;
    int presentation_frame = 0;
};

struct G1FrameTuning
{
    g1_test_mode mode = G1_TestLive;
    int frame_limit = 0;
    int scene_dwell_frames = 25;
    float dt = 1.0f / 25.0f;
    float trajectory_sample_time = 1.0f / 3.0f;
    float route_speed = 0.50f;
    float future_speed_scale = 1.0f;
    float walkability_radius = 0.20f;
    float effective_terrain_weight = 0.0f;
    float initial_search_time = 0.10f;
    float inertialize_blending_halflife = 0.10f;
    float desired_velocity_change_threshold = 50.0f;
    float desired_rotation_change_threshold = 50.0f;
    float simulation_velocity_halflife = 0.27f;
    float simulation_rotation_halflife = 0.27f;
    float simulation_run_forward_speed = 0.90f;
    float simulation_run_side_speed = 0.60f;
    float simulation_run_back_speed = 0.60f;
    float simulation_walk_forward_speed = 0.50f;
    float simulation_walk_side_speed = 0.40f;
    float simulation_walk_back_speed = 0.40f;
    bool synchronization_enabled = false;
    float synchronization_data_factor = 1.0f;
    bool adjustment_enabled = true;
    bool adjustment_by_velocity_enabled = true;
    float adjustment_position_halflife = 0.10f;
    float adjustment_rotation_halflife = 0.20f;
    float adjustment_position_max_ratio = 0.50f;
    float adjustment_rotation_max_ratio = 0.50f;
    bool clamping_enabled = true;
    float clamping_max_distance = 0.15f;
    float clamping_max_angle = 0.5f * PIf;
    bool ik_enabled = false;
};

struct G1FrameExternalInputs
{
    const database* db = nullptr;
    const terrain_support_set* support = nullptr;
    const scene_pack* scene = nullptr;
    const scene_route* route = nullptr;
    G1TestHeadingOverride heading_override;
    G1FrameInputSnapshot input;
    G1FrameTuning tuning;
};

using G1FrameStageRunner = G1FrameStageOutcome (*)(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
enum G1FrameInjectedOutcome
{
    G1FrameInjectContinue = 0,
    G1FrameInjectFiniteReject,
    G1FrameInjectGlobalError,
};

struct G1FrameTransactionTestControl
{
    G1FrameTransactionStage injected_stage = G1FrameStageCount;
    G1FrameInjectedOutcome injected_outcome = G1FrameInjectContinue;
};

using G1FrameTransactionTestHook = G1FrameInjectedOutcome (*)(
    G1FrameTransactionStage stage,
    const g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    const G1FrameTransactionTestControl& control,
    char* error,
    int error_capacity);

struct G1FrameTransactionTestSeam
{
    G1FrameTransactionTestHook hook = nullptr;
    G1FrameTransactionTestControl control;
};
#endif

static inline G1FrameTransactionStatus g1_frame_transaction_run(
    G1FrameRuntime& runtime,
    G1FrameStageRunner run_stage,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity);

struct G1FrameResetConfig
{
    bool route_mode = false;
    const char* route_id = nullptr;
    bool ik_enabled = false;
    float dt = 1.0f / 25.0f;
    float trajectory_sample_time = 1.0f / 3.0f;
    float initial_search_time = 0.10f;
};

static inline bool g1_frame_runtime_reset(
    G1FrameRuntime& output,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const G1FrameResetConfig& config,
    char* error,
    int error_capacity);
```

Create `g1_controller_frame_runtime.h` as the shared declaration of the exact
named production runner implemented in `controller.cpp`:

```cpp
#pragma once

#include "g1_frame_transaction.h"

G1FrameStageOutcome g1_controller_frame_stage_run(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);
```

Add checked validity/reset helpers that build local candidates and assign only
on success. The attempted footprint, IK, and pose members are immutable
value-only diagnostic snapshots containing no controller state, history owner,
pointer, or array buffer. This exact rejection type is the finite-rejection
whitelist. The accepted diagnostic is a separate transactional value owner: it
contains the exact query, terrain snapshot, route/traversal values, selected
frame identifiers, and pose diagnostics needed by post-coordinator logging and
render diagnostics, plus the accepted adjustment/clamping/IK flags and
effective terrain weight used by the existing log row, but no pointer or
mutable controller owner. Its validator requires nonnegative frame/range
identifiers where ready, finite pose/query/traversal values, and finite
effective terrain weight in `[0,10]`. The runner
fills only `scratch.accepted_diagnostic_candidate`; the coordinator publishes
it only with an accepted state.

Task 4 writes its footprint output only on `G1FootprintOk`, and the clearance
kernel writes `G1PoseClearance` only on `G1ClearanceOk`. Therefore
`attempted_footprint_available` is true only for `Ok` observations, including
blocked and unavailable-landing-patch observations, and
`attempted_pose_available` is true only for an `Ok` pose certificate rejected
by the controller thresholds. Outside-domain, budget-exceeded, and uncertified
statuses retain the exact status but set the corresponding availability false
and leave the value object at its canonical default. Add a checked Task-5
rejection-result snapshot/validator in `g1_ik_runtime.h`:

```cpp
enum G1IkRejectionCheckpoint
{
    G1IkRejectionAfterBegin = 0,
    G1IkRejectionAfterFoot0,
    G1IkRejectionAfterFoot1,
};

static inline bool g1_ik_frame_rejection_snapshot(
    G1IkFrameResult& output,
    const G1IkFrameTransaction& transaction,
    G1IkRejectionCheckpoint expected_checkpoint,
    char* error,
    int error_capacity);
```

Reject an unknown checkpoint. Every checkpoint requires an initialized
transaction, `candidate_result.applied == false`,
`candidate_result.safe_stop_requested == true`, a valid non-`None`
`candidate_result.stop_reason`, exact `+0.0f`
`candidate_result.max_correction_radians`, valid begin-produced
recorded-contact/target fields for both feet, and canonical finish-only fields.
Then enforce this exact stage table:

| Expected checkpoint | Required cursor and reason | Required foot-stage fields |
|---|---|---|
| `G1IkRejectionAfterBegin` | `next_foot == 0`; footprint-blocked or landing-patch-unavailable | Both feet retain their valid begin-produced contact/target and canonical foot-stage-only fields. |
| `G1IkRejectionAfterFoot0` | `next_foot == 1`; target-unreachable or no-swing-candidate | Foot 0 is a complete reason-coherent rejecting result; foot 1 retains its valid begin-produced contact/target and canonical foot-stage-only fields. |
| `G1IkRejectionAfterFoot1` | `next_foot == 2`; target-unreachable or no-swing-candidate | Foot 0 is a complete successful staged result and foot 1 is a complete reason-coherent rejecting result. |

A complete successful staged contact foot has valid passing position and
orientation diagnostics with canonical swing-selection fields. A complete
successful staged swing foot has a bounded nonzero candidate count, a selected
index inside that count, matching selected-candidate diagnostics, and valid
passing position/orientation diagnostics. A target-unreachable rejecting foot
is recorded contact with complete failing position/orientation diagnostics; a
no-swing-candidate rejecting foot is non-contact, evaluated all
`G1SwingLiftCandidateCount` candidates, selected none, and retains canonical
unselected position/orientation fields. Only after every common and
checkpoint-specific check does the helper assign the output. Thus
`attempted_ik_available` becomes true only after that helper validates and
assigns a complete snapshot. Unavailable values are never logged as evidence.

`G1FrameExternalInputs` is the complete runner context. Validate every pointer,
shape, input value, tuning scalar, mode/route relationship, and exact 25 Hz
before copying state. `mode == G1_TestRoute` if and only if `route != nullptr`,
and the route pointer, scene, and accepted route index must agree. Live,
sequential, flat, terrain, route, and scene-cycle modes all use this one
interface. It contains only pointers-to-const external artifacts and immutable
value snapshots; accepted state, working state, publication, logs, models,
cleanup, and mutable intermediates cannot be represented in it. Every mutable
per-frame intermediate lives in `G1FrameTransactionScratch`. The currently
disabled learned-motion path is not represented; enabling it later requires a
separate typed transactional interface rather than mutable nnet globals.

Parse `MM_SEARCHT` once with full-string conversion and require a finite
binary32 value in `[0,10]`; null selects exact `0.10f`. Store the result in both
`G1FrameResetConfig.initial_search_time` and the immutable process/frame
tuning, require their bits and the accepted state's immutable `search_time`
bits to agree on every coordinator call, and never call `getenv` from the
runner. The countdown timers remain ordinary transactional state.
Retain current `MM_HALFLIFE` and `MM_SIMROT_HL` behavior by lowering their
existing startup values into the corresponding tuning fields. Retain `MM_STRAFE`,
scripted camera azimuth, and `MM_AUTODRIVE` by lowering them into the immutable
input snapshot. Six-mode production tests authenticate all of these mappings.

`g1_frame_runtime_reset` resolves and configures the route cursor entirely in
local candidates and independently resets accepted and working states. Before
equality/disjointness comparison, each state must independently pass the full
certified reset gates: zero-travel command with spawn heading, exact initial
`search_time`, `search_timer`, and `force_search_timer`, support-retargeted
baseline, checked global FK, `G1FootprintOk`, initialized valid `G1IkState`,
exact `G1ClearanceOk`, and pose minimum at least `-0.01`. Then prove semantic
equality and complete buffer disjointness, initialize publication from the
reset command intent with no rejection/latch and presentation frame zero,
reset the accepted diagnostic, and swap the complete candidate runtime only
after every check passes. Change `scene_reset_current` and
`scene_switch_transaction` to
accept `G1FrameRuntime&` and `G1FrameResetConfig`. Scene switch completely
builds the candidate scene, both states, publication, accepted diagnostic,
route cursors, and model before a non-failing scene/model/runtime/index swap
sequence; it never installs one live state and repairs the other afterward.

The test seam is equally typed: `G1FrameTransactionTestControl` contains only
the requested checkpoint and injected enum outcome. There is no opaque context
or pointer field in either production or test runner APIs. The hook receives
that const control value, may complete the stage-appropriate diagnostic in
transaction scratch, and cannot receive accepted state or publication.

Implement `compile_g1_frame_transaction_runner_negative.cpp` with exactly one
of `G1_FRAME_NEGATIVE_VOID_CONTEXT`,
`G1_FRAME_NEGATIVE_PUBLICATION_CONTEXT`, or
`G1_FRAME_NEGATIVE_ACCEPTED_CONTEXT` required. The selected mode declares a
function with the same runner arguments except that the fourth argument is,
respectively, mutable `void*`, `G1FramePublication*`, or
`g1_controller_state*`. The translation unit performs:

```cpp
G1FrameStageRunner forbidden_runner = &runner_with_forbidden_context;
```

Each mode must fail at that initialization. The fixture has no cast, adapter,
overload, template, or alternate typedef that could make the proof vacuous.

`g1_frame_transaction_run` preflights the runner/external inputs and the full
semantics of accepted state, incoming publication, and accepted diagnostic.
Before overwrite it validates only the working destination's exact array
shapes, stable buffer identities, non-null storage, byte counts, and complete
within-state/cross-state/error-buffer disjointness; stale working scalar/value
semantics are deliberately irrelevant. It then performs the checked
accepted-to-working copy internally and validates working semantics after the
copy and again after all stages. It initializes
`scratch.prior_safe_stop_latched` from the immutable incoming publication
before any stage; the runner never receives publication or the accepted
diagnostic itself.
It calls the runner once for each enum value in order. Only after a real stage
returns `Continue` does the test build invoke its hook. A finite rejection
requires ready requested intent and a valid complete rejection snapshot; the
coordinator assigns a local publication candidate, latches safe stop, leaves
accepted state and accepted diagnostic untouched, and returns
`FiniteRejected`. A global error leaves accepted state, publication, and the
accepted diagnostic untouched and returns `GlobalError`; working state may be
dirty and is overwritten by the next checked copy.
Only after all stages continue does the coordinator validate working state,
the complete success publication, and the ready accepted-diagnostic candidate.
It then enters a non-failing publication tail: swap accepted/working exactly
once, assign the accepted diagnostic and publication, and clear
rejection/consumed latch. Both accepted and finite publication candidates copy
the fresh `external.input.presentation_frame`; global error leaves the incoming
publication unchanged. No allocation, formatting, logging, I/O, or operation
that can fail follows the first publication write. The runner and hook never
receive accepted state, publication, or the accepted diagnostic.

- [ ] **Step 5: Parse startup IK and snapshot typed input before mutation**

Parse `MM_IK` and validated `MM_SEARCHT` before either state reset and before
Raylib; accept only null/`0`/`1` for IK, defaulting to disabled, and the exact
finite `[0,10]` search-time contract from Step 4. Both values are immutable for
the process lifetime. Atomically reset both states with the same search-time
bits and assert all three initial search timer fields agree. Keep one
`G1FrameRuntime`.
Process a pending atomic pair reset/switch first, then sample Raylib/device
state and copy every current GUI/runtime tuning value into one local
`G1FrameExternalInputs`. Sampling may read external APIs but may not mutate
controller state. The coordinator is the first locomotion/controller-state
mutation of the ordinary frame; no controller code copies or swaps either
state directly:

```cpp
const G1FrameTransactionStatus frame_status = g1_frame_transaction_run(
    frame_runtime,
    g1_controller_frame_stage_run, frame_external,
    artifact_error, static_cast<int>(sizeof(artifact_error)));
if (frame_status == G1FrameTransactionGlobalError) {
    controlled_runtime_error(artifact_error);
    return;
}
```

`g1_controller_frame_stage_run` is the named non-capturing production runner
used by the live loop and linked into the production test. Its
`InputRouteCommand` case consumes only the immutable sampled input, applies a
pre-command scripted azimuth delta to working state, samples the configured
route when present, independently selects heading, and writes the validated
intent only to transaction scratch. Live gait input is lowered to exact
`gait_target`; ordinary camera look/zoom state is updated in working state
before the final stage continues. `MM_AUTODRIVE` and `MM_DISCRETE` similarly
lower to the input snapshot; the runner contains no environment or static
script state.
It receives its mutable state as the runner's `working_state` argument and
aliases it locally as `state`. Immediately before traversal, that case consumes
the prior latch with:

Place that externally linked runner and its complete pure dependency closure
outside `G1_CONTROLLER_NO_MAIN`. Guard `main` **and every file-scope static
root not reachable from the runner, including parser/startup-validation,
render/debug, and their main-only helper closures**, with
`#if !defined(G1_CONTROLLER_NO_MAIN)`. This lets the
production test link the exact runner translation unit under
`-Wall -Wextra -Werror` without a duplicate entry point or unused static
function, while preventing the guard from hiding or cloning runner behavior.

```cpp
const bool consume_safe_stop = scratch.prior_safe_stop_latched;
if (!g1_ik_safe_stop_handoff(
        scratch.safe_stop_handoff, consume_safe_stop,
        scratch.commanded_velocity,
        error, error_capacity)) {
    return G1FrameStageGlobalError;
}
if (scratch.safe_stop_handoff.cancel_planar_inertia) {
    state.simulation_velocity.x = 0.0f;
    state.simulation_velocity.z = 0.0f;
    state.simulation_acceleration.x = 0.0f;
    state.simulation_acceleration.z = 0.0f;
}
scratch.traversal_input = scratch.safe_stop_handoff.applied_velocity;
scratch.force_search = scratch.safe_stop_handoff.force_search;
```

The handoff contains only applied velocity, planar-cancel, and forced-search
fields. It receives no quaternion. When it is consumed in route mode, do not
advance `route_frames` during that accepted zero-command retry; the same route
sample is retried on the following frame. Build the requested intent/heading
in a local immutable value. Only after the complete command snapshot and
intent validate, publish it to transaction scratch and mark it ready:

```cpp
scratch.requested_intent = command_intent;
scratch.requested_intent_ready = true;
return G1FrameStageContinue;
```

No earlier failure may set the readiness bit. Do not write publication or
accepted state from the runner.
Clear a consumed latch only when this retry accepts; relatch on a new finite
rejection.

- [ ] **Step 6: Build contact horizons, observe, and stage after support retargeting**

After `support_pose_apply`, compute checked FK for the support-retargeted
baseline. Build contact horizons from the current matched database frame, then
call footprint observation with an immutable factory budget:

```cpp
if (!g1_foot_contact_schedule_build(
        scratch.contact_schedule, external.db->contact_states,
        external.db->range_starts, external.db->range_stops,
        state.frame_index,
        state.curr_bone_contacts(0),
        state.curr_bone_contacts(1),
        external.tuning.dt, external.tuning.trajectory_sample_time,
        error, error_capacity)) {
    return G1FrameStageGlobalError;
}
const G1FootprintBudget footprint_limits = g1_footprint_budget();
scratch.footprint_status = g1_footprint_observe_v2(
    scratch.footprint,
    footprint_limits,
    external.scene->terrain,
    external.scene->walkability,
    state.command,
    scratch.contact_schedule,
    state.global_bone_positions,
    state.global_bone_rotations,
    error,
    error_capacity);
```

The footprint stage returns an enum outcome directly. For `OutsideDomain` or
`BudgetExceeded`, record the exact status, keep
`attempted_footprint_available == false` and the attempted observation at its
canonical default, validate the rejection, then return
`G1FrameStageFiniteReject`. For
`InvalidInput`, `InvalidField`, `ArithmeticFailure`, or any unknown status,
preserve the diagnostic text already written by the checked observer and
return `G1FrameStageGlobalError`. It must not call controller cleanup.

For `Ok`, snapshot the complete validated observation and set
`attempted_footprint_available == true`; `scratch.footprint.blocked` then takes
the finite-rejection path with that available observation. Otherwise begin the
split IK transaction into the working candidate pose arrays exactly once:

```cpp
if (!g1_ik_frame_begin(
        scratch.ik_transaction,
        state.ik_candidate_bone_positions,
        state.ik_candidate_bone_rotations,
        state.ik,
        state.adjusted_bone_positions,
        state.adjusted_bone_rotations,
        external.db->bone_parents,
        state.curr_bone_contacts,
        external.scene->terrain,
        scratch.footprint,
        external.tuning.ik_enabled, external.tuning.dt,
        error, error_capacity)) {
    return G1FrameStageGlobalError;
}
if (scratch.ik_transaction.candidate_result.safe_stop_requested) {
    G1IkFrameResult attempted_ik = {};
    if (!g1_ik_frame_rejection_snapshot(
            attempted_ik, scratch.ik_transaction,
            G1IkRejectionAfterBegin,
            error, error_capacity)) {
        return G1FrameStageGlobalError;
    }
    scratch.rejection.attempted_footprint_available = true;
    scratch.rejection.footprint_status = G1FootprintOk;
    scratch.rejection.attempted_footprint = scratch.footprint;
    scratch.rejection.attempted_ik_available = true;
    scratch.rejection.ik_frame = attempted_ik;
    // Complete stage, reason, and canonical unavailable-pose fields, validate
    // the rejection, then stop before either foot-stage call.
    return G1FrameStageFiniteReject;
}
return G1FrameStageContinue;
```

No non-`Ok` footprint status may copy `scratch.footprint` into rejection or
state. The accepted observation is assigned to `state.footprint` only in the
fully accepted working-frame path.
Successful begin is therefore not equivalent to continuation: its transaction
is inspected immediately, and a landing-patch or other begin-time terminal
safe stop is published as a checked finite rejection at the footprint-stage
checkpoint before `g1_ik_frame_stage_foot(0)` can run.

- [ ] **Step 7: Evaluate the whole working frame and publish only on acceptance**

All matcher search, inertialization, simulation, trajectory, support, contact,
footprint, IK, pose certification, timer, scene-frame, and route-cursor writes
up to this point target the runner's `working_state` only. The three named IK
runner stages use the split Task-5 transaction, so each checkpoint follows a
real production mutation boundary:

```cpp
// G1FrameStageFirstFootIk
if (!g1_ik_frame_stage_foot(
        scratch.ik_transaction,
        state.ik_candidate_bone_positions,
        state.ik_candidate_bone_rotations,
        0, external.db->bone_parents, state.curr_bone_contacts,
        external.scene->terrain, scratch.footprint,
        external.tuning.ik_enabled, external.tuning.dt,
        error, error_capacity)) {
    return G1FrameStageGlobalError;
}
if (scratch.ik_transaction.candidate_result.safe_stop_requested) {
    G1IkFrameResult attempted_ik = {};
    if (!g1_ik_frame_rejection_snapshot(
            attempted_ik, scratch.ik_transaction,
            G1IkRejectionAfterFoot0,
            error, error_capacity)) {
        return G1FrameStageGlobalError;
    }
    // Publish attempted_ik through the checked finite diagnostic.
    return G1FrameStageFiniteReject;
}
return G1FrameStageContinue;

// G1FrameStageSecondFootIk
if (!g1_ik_frame_stage_foot(
        scratch.ik_transaction,
        state.ik_candidate_bone_positions,
        state.ik_candidate_bone_rotations,
        1, external.db->bone_parents, state.curr_bone_contacts,
        external.scene->terrain, scratch.footprint,
        external.tuning.ik_enabled, external.tuning.dt,
        error, error_capacity)) {
    return G1FrameStageGlobalError;
}
if (scratch.ik_transaction.candidate_result.safe_stop_requested) {
    G1IkFrameResult attempted_ik = {};
    if (!g1_ik_frame_rejection_snapshot(
            attempted_ik, scratch.ik_transaction,
            G1IkRejectionAfterFoot1,
            error, error_capacity)) {
        return G1FrameStageGlobalError;
    }
    // Publish attempted_ik through the checked finite diagnostic.
    return G1FrameStageFiniteReject;
}
return G1FrameStageContinue;

// G1FrameStageFinalFk
if (!g1_ik_frame_finish(
        state.ik, state.ik_frame, scratch.ik_transaction,
        state.ik_candidate_bone_positions,
        state.ik_candidate_bone_rotations,
        external.db->bone_parents, external.scene->terrain,
        external.tuning.dt,
        error, error_capacity)) {
    return G1FrameStageGlobalError;
}
if (state.ik_frame.safe_stop_requested) {
    // Validate the complete checked IK result, snapshot it, then validate the
    // finite diagnostic before returning.
    return G1FrameStageFiniteReject;
}
return G1FrameStageContinue;
```

The runner checks every boolean return before reading result fields. A
candidate-local safe stop becomes a complete finite rejection in transaction
scratch; a checked-call failure becomes `G1FrameStageGlobalError`.

Run checked FK and `g1_measure_pose_clearance` on the staged pose with
`g1_pose_clearance_budget()`. Accept only exact `G1ClearanceOk`, planted
toe/foot lower bounds at least `-0.005`, and overall minimum at least `-0.01`.
An `Ok` certificate rejected by those thresholds sets
`attempted_pose_available == true`. `OutsideDomain`, `BudgetExceeded`, or
`Uncertified` is a finite rejection with the exact status but
`attempted_pose_available == false`; the unchanged clearance output is not
copied. Invalid input/field/arithmetic or an unknown status is global error.

Before the pose stage returns `Continue`, update ordinary camera scalar state,
capture a complete `G1FrameAcceptedDiagnostic` with the pre-increment log row
labels, current adjustment/clamping/IK flags, and effective terrain weight,
then validate it and complete every search/force-search timer, route cursor, and
`scene_frame` increment in `working_state`. Suppress `route_frames` advancement
on an accepted latch-consuming retry. The coordinator alone validates and
publishes the state, accepted diagnostic, and publication. No timer, route,
camera, contact, pose, or scene-frame owner is mutated after it returns.

The outer loop then constructs one deterministic row only from
`frame_runtime.accepted_state`, `frame_runtime.accepted_diagnostic`, and
`frame_runtime.publication`; it never reads working scratch. A finite row uses
the unchanged accepted locomotion/diagnostic plus the current validated
requested intent, rejection, latch, and fresh publication
`presentation_frame`. An accepted row requires the accepted diagnostic and
publication presentation frames to agree. A log failure after an accepted swap
is an outer fatal I/O failure, not transaction rollback. After successful
logging, publish the UI snapshot, schedule scene-cycle changes, derive
`Camera3D` from accepted camera scalars and accepted IK pose, render only
`accepted_state.ik_global_bone_positions`, and increment the outer
presentation/frame-limit counter.

On `OutsideDomain`, `BudgetExceeded`, blocked footprint, unavailable landing
patch, no finite swing candidate, target-unreachable, or finite pose-clearance
rejection, construct a complete local `G1FrameRejectionDiagnostic`, validate
it, including every available attempted per-foot landing observation, selected
surface, readiness/residual, checked IK target/reach result, available pose
certificate, and stop reason. Availability bits determine which value objects
may be read; unavailable objects remain canonical defaults. Store the result
only in `scratch.rejection`, and return
`G1FrameStageFiniteReject`. The coordinator sets the latch and does **not**
swap. Require
the accepted digest and every sentinel owner to remain identical. The log may
publish the immutable attempted `requested_intent`, rejection diagnostic, and
latch beside the unchanged accepted locomotion record. No other working value
is observable.

On `InvalidInput`, `InvalidField`, `ArithmeticFailure`, copy failure, or
publication-validation failure, a runner or coordinator returns
`G1FrameStageGlobalError`/`G1FrameTransactionGlobalError` with the original
diagnostic text. Only the outer controller call site invokes the existing
controlled error path; the once-guarded cleanup retains exact order: close
logs, unload the model, close Raylib, then write the cleanup report. The
accepted state and accepted diagnostic are still unchanged. The
latched retry forces a matcher search, zeros only planar applied motion/inertia
for one accepted frame, freezes the route cursor for that frame, and then
clears, so the next frame can retry the same route sample instead of remaining
permanently stopped.

- [ ] **Step 8: Keep startup IK-off byte-reversible**

With `MM_IK=0`, still compute footprint and explicit pose certificates, but
commit a byte copy of the support-retargeted rotations/positions. Each process
starts with checked locks/history reset from its initial accepted pose. There
is no runtime toggle; comparing separately launched `MM_IK=0` and `MM_IK=1`
processes must show that neither mode clears or rewrites heading.

- [ ] **Step 9: Run focused controller, strict-kernel, release-caller, and sanitizer tests**

```bash
mkdir -p /tmp/g1-frame-transaction
raylib_include=(-I /home/ubuntu/apps/raylib/src -I /home/ubuntu/apps/raygui/src)
raylib_link=(-L /home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread \
  -ldl -lrt -lX11)
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp -o /tmp/g1-frame-transaction/kernel.o
for name in g1_controller_state g1_frame_transaction \
            g1_footprint_runtime g1_clearance g1_ik scene_switch; do
  define=()
  test "$name" = g1_ik && define=(-DG1_IK_ENABLE_TEST_SEAMS)
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic "${define[@]}" \
    -I. -c "tests/cpp/test_${name}.cpp" \
    -o "/tmp/g1-frame-transaction/${name}-strict.o"
  g++ "/tmp/g1-frame-transaction/${name}-strict.o" \
    /tmp/g1-frame-transaction/kernel.o \
    -o "/tmp/g1-frame-transaction/${name}-strict"
  "/tmp/g1-frame-transaction/${name}-strict"

  g++ -std=c++17 -O3 -ffast-math -DNDEBUG "${define[@]}" \
    -I. -c "tests/cpp/test_${name}.cpp" \
    -o "/tmp/g1-frame-transaction/${name}-fast.o"
  g++ "/tmp/g1-frame-transaction/${name}-fast.o" \
    /tmp/g1-frame-transaction/kernel.o \
    -o "/tmp/g1-frame-transaction/${name}-fast"
  "/tmp/g1-frame-transaction/${name}-fast"
done

# Link the exact controller-owned runner into the dynamic production test.
for flavor in strict fast; do
  if test "$flavor" = strict; then
    caller_flags=(-O2 -Wall -Wextra -Werror -pedantic)
  else
    caller_flags=(-O3 -ffast-math -DNDEBUG)
  fi
  g++ -std=c++17 "${caller_flags[@]}" \
    -DG1_CONTROLLER_NO_MAIN -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
    -I. "${raylib_include[@]}" -c controller.cpp \
    -o "/tmp/g1-frame-transaction/controller-runner-${flavor}.o"
  g++ -std=c++17 "${caller_flags[@]}" \
    -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM -I. \
    -c tests/cpp/test_g1_frame_transaction_production.cpp \
    -o "/tmp/g1-frame-transaction/frame-production-${flavor}.o"
  g++ "${caller_flags[@]}" \
    "/tmp/g1-frame-transaction/frame-production-${flavor}.o" \
    "/tmp/g1-frame-transaction/controller-runner-${flavor}.o" \
    /tmp/g1-frame-transaction/kernel.o "${raylib_link[@]}" \
    -o "/tmp/g1-frame-transaction/frame-production-${flavor}"
  "/tmp/g1-frame-transaction/frame-production-${flavor}"
done

# Compile and link the real production controller with no frame/IK test seams.
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. "${raylib_include[@]}" -c controller.cpp \
  -o /tmp/g1-frame-transaction/controller-production.o
g++ /tmp/g1-frame-transaction/controller-production.o \
  /tmp/g1-frame-transaction/kernel.o "${raylib_link[@]}" \
  -o /tmp/g1-frame-transaction/controller-production
nm -C /tmp/g1-frame-transaction/controller-production.o | \
  rg 'g1_controller_frame_stage_run'
if nm -C /tmp/g1-frame-transaction/controller-production.o | \
     rg 'G1FrameTransactionTest|G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM'; then
  echo "ERROR: frame transaction seam leaked into production" >&2
  exit 1
fi

for mode in VOID_CONTEXT PUBLICATION_CONTEXT ACCEPTED_CONTEXT; do
  stderr="/tmp/g1-frame-transaction/runner-negative-${mode}.stderr"
  if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
       -D"G1_FRAME_NEGATIVE_${mode}" \
       -c tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
       -o "/tmp/g1-frame-transaction/runner-negative-${mode}.o" \
       2>"$stderr"; then
    echo "ERROR: frame runner accepted forbidden ${mode} context" >&2
    exit 1
  fi
  rg 'G1FrameStageRunner|convert|conversion|argument' "$stderr"
done

san_flags=(-O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all)
g++ -std=c++17 "${san_flags[@]}" -fno-fast-math -ffp-contract=off \
  -frounding-math -I. -c g1_clearance.cpp \
  -o /tmp/g1-frame-transaction/kernel-san.o
for name in g1_controller_state g1_frame_transaction \
            g1_ik scene_switch; do
  define=()
  test "$name" = g1_ik && define=(-DG1_IK_ENABLE_TEST_SEAMS)
  g++ -std=c++17 "${san_flags[@]}" "${define[@]}" -I. \
    -c "tests/cpp/test_${name}.cpp" \
    -o "/tmp/g1-frame-transaction/${name}-san.o"
  g++ "${san_flags[@]}" "/tmp/g1-frame-transaction/${name}-san.o" \
    /tmp/g1-frame-transaction/kernel-san.o \
    -o "/tmp/g1-frame-transaction/${name}-san"
  ASAN_OPTIONS=detect_leaks=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "/tmp/g1-frame-transaction/${name}-san"
done

g++ -std=c++17 "${san_flags[@]}" \
  -DG1_CONTROLLER_NO_MAIN -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM \
  -I. "${raylib_include[@]}" -c controller.cpp \
  -o /tmp/g1-frame-transaction/controller-runner-san.o
g++ -std=c++17 "${san_flags[@]}" \
  -DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM -I. \
  -c tests/cpp/test_g1_frame_transaction_production.cpp \
  -o /tmp/g1-frame-transaction/frame-production-san.o
g++ "${san_flags[@]}" \
  /tmp/g1-frame-transaction/frame-production-san.o \
  /tmp/g1-frame-transaction/controller-runner-san.o \
  /tmp/g1-frame-transaction/kernel-san.o "${raylib_link[@]}" \
  -o /tmp/g1-frame-transaction/frame-production-san
ASAN_OPTIONS=detect_leaks=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/g1-frame-transaction/frame-production-san
```

Expected: all tests exit zero; the exact linked production runner covers every
current mode; IK-off pose bytes and matching state are unchanged; accepted
multilevel footprints commit; pair reset/switch is atomic; finite failures
latch a next-frame planar stop without changing accepted diagnostics; global
failures clean up; no path changes command heading; and the normal production
controller contains no test seam.

- [ ] **Step 10: Commit controller integration**

```bash
git add g1_frame_transaction.h g1_controller_frame_runtime.h \
  g1_controller_state.h g1_ik_runtime.h scene_switch.h controller.cpp \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp \
  tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
  tests/cpp/test_g1_controller_state.cpp tests/cpp/test_g1_ik.cpp \
  tests/cpp/test_scene_switch.cpp
git commit -m "feat: integrate footprint-aware G1 terrain IK"
```

---

### Task 7: Log Physical Footprints and Enforce Directional Gates

**Files:**
- Create: `g1_candidate_audit.h`
- Create: `tests/cpp/test_g1_candidate_audit.cpp`
- Create: `resources/audit_g1_directional_candidates.py`
- Create: `tests/python/test_directional_candidates.py`
- Create: `resources/g1_visualizer_process.py`
- Create: `tests/python/test_visualizer_process.py`
- Modify: `motion_match_log.h`
- Modify: `controller.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`

**Interfaces:**
- Appends the already-approved Gate-E IK suffix, then one new immutable directional suffix. Existing columns are never renamed or reordered.
- Produces corrected Gate D physical-clearance checks, `--gate-l` validation
  for Gates L1--L5, paired Gate-L2 leg-role validation, and behavior-inert
  candidate-count/cost/range evidence.

- [ ] **Step 1: Write schema, Gate-D, and Gate-L RED tests**

Extend the synthetic-row factory in `tests/python/test_runtime_log.py` with the exact suffixes below. Add mutations proving:

- a blocked log with `blocked_distance=0.04` and `rendered_min_clearance=-0.40` fails Gate D;
- a one-bit desired or predicted heading change fails Gate L;
- a rendered heading-error-threshold-only mutation passes
  `--gate-l-safety-only` but fails full Gate L, while every exact-heading,
  completion, rollback, support, or physical mutation fails both;
- a class-1 route that safe-stops, fails to reach the supplied endpoint, or does not complete fails Gate L;
- a runtime root/contact-foot surface split of at least `0.04` without a footprint multilevel observation fails;
- planted toe/foot penetration below `-0.005`, physical minimum below `-0.01`, Hips frame step above `0.05`, or excessive heading error fails;
- treatment matching/query/support-command columns differing from its paired IK-off control fail; and
- a 100-frame Gate-L4 run differing in any common invariant column from its
  immutable pre-footprint flat-ground oracle fails;
- either log in a paired lateral stair run that does not collectively show
  left/right as both the higher and lower planted support foot for at least
  three consecutive rows fails Gate L2;
- the dedicated landing-exit stress log passes only by coherent completion or
  by a safe-stop before any rejected working state changes the accepted-state
  digest, and its rejected row carries the attempted landing/surface,
  readiness/residual, reach, clearance, and stop-reason evidence needed to
  justify that stop; a safe-stop in any normal certified route still fails;
- a rejected non-`Ok` pose status with
  `rejected_attempted_pose_available=0` ignores the canonical default pose
  value, while changing that availability bit to `1` without an `Ok` complete
  certificate fails schema validation;
- each forward/backward/left/right flat category missing either its +Z or
  rotated +X oracle pairing fails Gate L4;
- changing any accepted-state digest across a finite rejected row fails; and
- all forward/backward/diagonal/lateral heading codes pass the same checker path without a movement-mode flag.

Create C++ audit tests that compare all eligible 31-dimensional scalar costs
against an independent brute-force loop, lock stable `(cost,index)` ties, and
count candidates at exact best-cost deltas `0.25f`, `1.0f`, and `4.0f`.
Create Python tests that join every emitted range ordinal/frame to the exact
`manifest.json` source range/name and reject malformed, duplicate, unsorted,
or nonfinite audit records.
Add process-identity tests over a synthetic proc tree that reject changed PID
start time, executable path, device/inode, or command bytes before the mocked
signal call. Require `capture-wait` to reject a missing PID, timeout on a wrong
executable, and reject a changed pinned start time during a simulated exec; it
emits the same canonical identity as `discover` once the expected executable
appears for the unchanged process.

- [ ] **Step 2: Run the checker RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_runtime_log -v
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_candidate_audit.cpp \
  -o /tmp/test_g1_candidate_audit_red
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_directional_candidates -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_visualizer_process -v
```

Expected: new tests fail because the suffix, Gate L, audit interfaces,
process-identity helper, and corrected Gate D physical-clearance semantics do
not exist.

- [ ] **Step 3: Append the existing exact IK suffix**

Append these columns first, exactly in this order:

```text
ik_applied,ik_safe_stop_requested,ik_stop_reason,max_ik_correction,
actual_simulation_speed,ik_candidate_rejected,ik_candidate_clearance_status,
left_candidate_toe_clearance,left_candidate_foot_clearance,
right_candidate_toe_clearance,right_candidate_foot_clearance,
ik_candidate_minimum_clearance,
left_recorded_contact,right_recorded_contact,left_locked,right_locked,
left_observed_lock_drift,right_observed_lock_drift,
left_lock_drift,right_lock_drift,
left_sole_normal_alignment,right_sole_normal_alignment,
left_contact_residual,right_contact_residual,
left_target_height,right_target_height,
left_target_normal_x,left_target_normal_y,left_target_normal_z,
right_target_normal_x,right_target_normal_y,right_target_normal_z,
left_swing_candidates_evaluated,left_swing_selected_index,
left_swing_selected_lift_bits,left_swing_materialized_command_y_bits,
left_swing_actual_sphere_center_bits_hex,
left_swing_selected_clearance_status,
left_swing_selected_controller_constraints_passed,
left_swing_selected_clearance_certified,
left_swing_lower_margin,left_swing_witness_upper_margin,
left_swing_selected_work_point_queries,left_swing_selected_work_cells_visited,
left_swing_selected_work_primitive_triangle_pairs,
left_swing_selected_work_face_patches,left_swing_selected_work_candidate_tests,
left_swing_selected_work_subdivision_nodes,
left_swing_total_work_point_queries,left_swing_total_work_cells_visited,
left_swing_total_work_primitive_triangle_pairs,
left_swing_total_work_face_patches,left_swing_total_work_candidate_tests,
left_swing_total_work_subdivision_nodes,
right_swing_candidates_evaluated,right_swing_selected_index,
right_swing_selected_lift_bits,right_swing_materialized_command_y_bits,
right_swing_actual_sphere_center_bits_hex,
right_swing_selected_clearance_status,
right_swing_selected_controller_constraints_passed,
right_swing_selected_clearance_certified,
right_swing_lower_margin,right_swing_witness_upper_margin,
right_swing_selected_work_point_queries,right_swing_selected_work_cells_visited,
right_swing_selected_work_primitive_triangle_pairs,
right_swing_selected_work_face_patches,right_swing_selected_work_candidate_tests,
right_swing_selected_work_subdivision_nodes,
right_swing_total_work_point_queries,right_swing_total_work_cells_visited,
right_swing_total_work_primitive_triangle_pairs,
right_swing_total_work_face_patches,right_swing_total_work_candidate_tests,
right_swing_total_work_subdivision_nodes,
left_reachable,right_reachable,left_knee_clearance,left_ankle_clearance,
left_toe_clearance,left_foot_clearance,left_shin_clearance,
left_thigh_clearance,right_knee_clearance,right_ankle_clearance,
right_toe_clearance,right_foot_clearance,right_shin_clearance,
right_thigh_clearance,ik_hips_clearance,ik_minimum_clearance
```

Fill them only from accepted state and the exact authoritative
candidate/selection objects. Each actual-center bit string is twelve lowercase
eight-hex-digit words in sphere-major `(x,y,z)` order with no separators.
Ignored selected fields retain their authoritative defaults when
`selected_index == G1SwingNoCandidate`; there is no rejection mask. Print all
binary64 safety values with `%.17g`; never round through float.

- [ ] **Step 4: Append the independent-command and footprint suffix**

Append these columns after the IK suffix:

```text
requested_velocity_x,requested_velocity_y,requested_velocity_z,
applied_velocity_x,applied_velocity_y,applied_velocity_z,
desired_heading_bits_hex,predicted_heading_bits_hex,
simulation_heading_error_deg,rendered_heading_error_deg,
footprint_status,footprint_blocked,footprint_blocked_reason,
footprint_root_height,
left_footprint_min_height,left_footprint_max_height,
right_footprint_min_height,right_footprint_max_height,
left_maximum_root_split,right_maximum_root_split,
left_footprint_multilevel,right_footprint_multilevel,
left_landing_expected,left_landing_patch_ready,left_landing_sample,
left_landing_surface_status,left_landing_walkability_class,
left_predicted_landing_center_x,left_predicted_landing_center_y,
left_predicted_landing_center_z,left_predicted_landing_height,
left_predicted_landing_normal_x,left_predicted_landing_normal_y,
left_predicted_landing_normal_z,left_landing_patch_maximum_residual,
right_landing_expected,right_landing_patch_ready,right_landing_sample,
right_landing_surface_status,right_landing_walkability_class,
right_predicted_landing_center_x,right_predicted_landing_center_y,
right_predicted_landing_center_z,right_predicted_landing_height,
right_predicted_landing_normal_x,right_predicted_landing_normal_y,
right_predicted_landing_normal_z,right_landing_patch_maximum_residual,
footprint_sweeps,footprint_surface_queries,footprint_node_visits,
frame_rejected,frame_rejection_stage,ik_safe_stop_latched,
rejected_attempted_footprint_available,rejected_attempted_ik_available,
rejected_stop_reason,rejected_attempted_pose_available,rejected_pose_status,
rejected_pose_minimum_clearance,
rejected_left_landing_expected,rejected_left_landing_patch_ready,
rejected_left_landing_sample,rejected_left_landing_center_x,
rejected_left_landing_center_y,rejected_left_landing_center_z,
rejected_left_landing_surface_status,rejected_left_landing_surface_height,
rejected_left_landing_surface_normal_x,
rejected_left_landing_surface_normal_y,
rejected_left_landing_surface_normal_z,
rejected_left_landing_walkability_class,
rejected_left_landing_patch_maximum_residual,
rejected_left_target_x,rejected_left_target_y,rejected_left_target_z,
rejected_left_target_normal_x,rejected_left_target_normal_y,
rejected_left_target_normal_z,rejected_left_reachable,
rejected_left_correction_limited,rejected_left_selected_clearance_status,
rejected_left_selected_lower_margin,rejected_left_selected_witness_upper,
rejected_right_landing_expected,rejected_right_landing_patch_ready,
rejected_right_landing_sample,rejected_right_landing_center_x,
rejected_right_landing_center_y,rejected_right_landing_center_z,
rejected_right_landing_surface_status,rejected_right_landing_surface_height,
rejected_right_landing_surface_normal_x,
rejected_right_landing_surface_normal_y,
rejected_right_landing_surface_normal_z,
rejected_right_landing_walkability_class,
rejected_right_landing_patch_maximum_residual,
rejected_right_target_x,rejected_right_target_y,rejected_right_target_z,
rejected_right_target_normal_x,rejected_right_target_normal_y,
rejected_right_target_normal_z,rejected_right_reachable,
rejected_right_correction_limited,rejected_right_selected_clearance_status,
rejected_right_selected_lower_margin,rejected_right_selected_witness_upper,
accepted_state_digest_hex
```

Serialize `desired_heading_bits_hex` as four lowercase eight-hex-digit words in `(w,x,y,z)` order and `predicted_heading_bits_hex` as sixteen words in sample-major order. The two strings contain no separators. Calculate heading errors from yaw-only normalized rotations for diagnostics; these diagnostics never feed behavior.

Every pre-existing IK/footprint/state column above is serialized only from
`accepted_state`, even on a rejected row. The `rejected_*` columns are
serialized only from the validated immutable
`publication.rejection` attempted diagnostic; accepted rows carry their fixed
schema defaults. They never read `working_state` directly. Binary64 attempted
residual/margin/clearance values use `%.17g`. The checker and serializer read
`rejected_pose_minimum_clearance` only when
`rejected_attempted_pose_available` is true; otherwise the canonical default
is ignored and cannot justify a stop. Availability true requires exact
`G1ClearanceOk` and a complete validated pose certificate.

`accepted_state_digest_hex` is a deterministic 64-bit test/log digest over
every scalar and every byte of every accepted-state array in declared-owner
order. It is not used for acceptance. On a finite rejection it must equal the
preceding accepted digest; the requested intent, rejection fields, latch, and
fresh attempt presentation frame are deliberately outside it.

- [ ] **Step 5: Correct Gate D's mislabeled clearance**

Keep the traversability reserve check, but name it accurately and add physical checks:

```python
minimum_blocked_distance = min(
    _finite(rows[index], "blocked_distance", index)
    for index in stopped_tail)
if minimum_blocked_distance < 0.02 - 1e-4:
    raise ValueError("Gate D stopped distance fell below 0.02 m")

minimum_rendered_clearance = min(
    _finite(rows[index], "rendered_min_clearance", index)
    for index in stopped_tail)
minimum_certified_clearance = min(
    _finite(rows[index], "ik_minimum_clearance", index)
    for index in stopped_tail)
if minimum_rendered_clearance < -0.01 or \
        minimum_certified_clearance < -0.01:
    raise ValueError("Gate D physical clearance fell below -0.01 m")
```

Return all three values under distinct report keys. Delete the old assignment that labeled `blocked_distance` as `minimum_clearance`.

- [ ] **Step 6: Implement one Gate-L checker for all directions**

Add CLI options:

```text
--gate-l
--gate-l-safety-only
--compare-ik-off PATH
--compare-forward PATH
--compare-forward-baseline PATH
--compare-flat-baseline PATH
--gate-l2-pair PATH
--gate-l2-exit-stress
--expected-end-x FLOAT
--expected-end-z FLOAT
--expected-heading {forward,backward,positive-x,negative-x,diagonal-positive-x,diagonal-negative-x}
--expected-relative-direction {forward,backward,left,right}
--require-multilevel
```

`check_gate_l` must enforce:

- exact 25 Hz, one scene/route/generation, `MM_IK=1`, class-1 completion, no IK or footprint safe-stop, and final XZ within `0.25 m` of the supplied endpoint;
- exact desired-heading bits on every row and four repetitions of those bits in every predicted-heading record;
- bit-identical matcher query, selected-frame, cost, terrain, support-root, simulation-XZ, and command-intent columns against the paired IK-off log;
- every planted toe/foot certified lower bound at least `-0.005`, every overall rendered/certified lower bound at least `-0.01`, correction no more than `0.35 rad`, and per-frame rendered Hips Y change no more than `0.05 m`;
- after the first 50 frames, median rendered heading error at most `10 degrees`, 95th percentile at most `20 degrees`, and maximum at most `35 degrees`;
- when `--require-multilevel` is present, at least one root/contact-foot runtime surface split of `0.04 m` and a same-or-earlier footprint multilevel report; and
- maximum absolute support velocity no greater than `max(forward_peak + 0.25, 1.25 * forward_peak)` when a forward comparison is supplied. The tangential route without a forward pair has an absolute `1.50 m/s` cap.

`--gate-l-safety-only` runs the identical parser, pairing, exact-heading-bit,
completion, safe-stop, multilevel, support-bound, rollback, contact, and
physical-clearance checks but defers only the three rendered heading-error
thresholds and subjective motion-quality disposition. It may not waive route
completion, exact requested heading, or any physical/transactional failure.
It is used solely to decide whether candidate auditing is safe and meaningful
when a full quality verdict fails.

`--compare-forward-baseline` verifies the immutable Task-3 SHA record, accepts
the older pre-footprint schema, and compares all shared query bits,
selected-frame/cost, simulation-XZ, route cursor/completion, support, contacts,
and quality metrics. It is valid only for the five forward reference routes.
The final forward IK-off run must match all bitwise invariants and may not
regress the baseline's completion, clearance, contact-level coverage, Hips
step, or heading-error maxima.

`--compare-flat-baseline` is Gate L4 and is mutually exclusive with endpoint
and multilevel requirements. It accepts the older Task-3 schema only after its
SHA256 record verifies, requires exactly 100 rows from the same scene, route,
  heading, and 25 Hz schedule, and compares every shared matcher-query,
  selected-frame, selected-cost, simulation-XZ, and support column bit-for-bit.
  The final row's heading suffix is checked against the fixed command constant.
  The final run must have `MM_IK=0`, no footprint
block/stop, and no physical surface split of `0.04 m`. It computes relative
travel from world route delta transformed by the fixed body heading and
requires the supplied relative category. Gate L4 accepts only the exact eight
Task-3 pairs and requires two distinct absolute headings/world routes for each
of forward, backward, left, and right.

`--gate-l2-pair PATH` is valid only for the normal
`stairs-standard/ascent-landing-descent` positive-X/negative-X pair after both
individually pass the safety portion of Gate L. For every run of at least three consecutive
rows where both feet are recorded planted and target heights differ by at
least `0.04 m`, classify the higher target as `uphill` and the lower target as
`downhill`. Across the pair require exact role coverage
`{left: {uphill,downhill}, right: {uphill,downhill}}`, route completion, no
safe-stop, and bounded physical clearance. This proves leg roles from recorded
targets/contacts rather than inferring them from heading.

`--gate-l2-exit-stress` is standalone and valid only for
`stairs-standard/landing-side-exit-stress` with positive-X heading. It accepts
one of two explicit branches: (a) coherent route completion with all normal
physical gates, or (b) `landing-patch-unavailable`, `no-swing-candidate`, or
pose-certificate safe-stop before unsafe publication. Branch (b) requires the
accepted-state digest to remain identical across every rejected row, no new
contact/target level to commit after the unready patch, and physical minima to
remain bounded. It must also prove the selected landing is outside the
certified reach/landing-patch contract using only the rejection-only attempted
columns: either the relevant `rejected_*_reachable` flag is false, or
`rejected_*_landing_patch_ready` is false with a rejected attempted residual
above exact `0.005f`. Its attempted stop reason and selected-clearance status
must agree with the rejection stage. This is the only class-1 route where safe-stop is an accepted
diagnostic outcome; it does not pass general Gate L and grants no waiver to
the normal paired stair routes.

The heading thresholds are final-quality gates. If geometry passes but they fail, Task 8 classifies the remaining problem as data coverage; the checker does not rotate heading or waive them.

- [ ] **Step 7: Add a deterministic behavior-inert candidate audit**

Create `g1_candidate_audit.h` with a fixed top-16 result. The checked function
accepts the immutable normalized 31-value query, feature matrix, ranges,
incumbent index, `ignore_range_end=20`, and `ignore_surrounding=20`. It visits
every eligible frame in ascending order, accumulates the 31 squared binary32
differences in feature order, and stable-sorts by `(cost,index)`. It reports:

```cpp
struct G1CandidateAuditEntry
{
    int frame = -1;
    int range = -1;
    uint32_t cost_bits = 0;
};

struct G1CandidateAudit
{
    uint32_t eligible_count = 0;
    uint32_t within_best_plus_0_25 = 0;
    uint32_t within_best_plus_1 = 0;
    uint32_t within_best_plus_4 = 0;
    uint32_t top_count = 0;
    G1CandidateAuditEntry top[16];
};
```

The three deltas use exact words `0x3e800000`, `0x3f800000`, and
`0x40800000`. Preflight all shapes/counts and checked additions; assign only a
complete finite result. This audit does not call or replace
`motion_matching_search`.

Parse optional `MM_CANDIDATE_AUDIT=/absolute/path.jsonl` only in bounded test
mode. After normal matching/acceptance decisions are complete, audit immutable
query/database data every 25th requested frame and every finite rejection,
then write query bits, scene/route/heading, accepted frame, the three counts,
and 16 `(frame,range,cost_bits)` entries. The result is never copied into
controller state and never feeds matching, terrain, IK, or acceptance.

`resources/audit_g1_directional_candidates.py` validates the JSONL and the
candidate pack's `manifest.json`, maps every range/frame to the exact
`sources[range]` name and `[range_start,range_stop)`, and emits canonical JSON
with per-cell minimum/median candidate counts, top costs, unique top-16 source
coverage, and source names. It fails if manifest ranges disagree with the
runtime records. Tests use a synthetic manifest and mutation-sensitive stable
tie/count fixtures.

Create `resources/g1_visualizer_process.py` with three CLI subcommands:

```text
discover --executable /absolute/path --output /absolute/identity.json
capture-wait --pid PID --executable /absolute/path --timeout-ms 2000 --output /absolute/identity.json
signal --identity /absolute/identity.json --signal STOP|CONT|TERM|KILL
```

`discover` enumerates numeric proc entries and requires zero or one exact
resolved executable match. It writes either JSON `null` or canonical JSON with
PID, `/proc/PID/stat` field-22 start time, executable path, executable
device/inode, and raw command-line hex. `capture-wait` immediately pins the
named PID and `/proc/PID/stat` start time, then polls for at most the supplied
bounded timeout until that unchanged process resolves to the exact executable.
Every poll rejects absence or a changed start time; after the executable
matches it reads the full identity twice and requires byte equality before
writing the canonical non-null document. It never accepts a replacement
process or a merely matching later PID. `signal` re-reads and byte-compares all
fields immediately before `os.kill`; absence is a successful no-op, and any
missing/changed field fails without signaling. Expose pure parse/compare
functions and injectable proc root/kill callback for tests, while the CLI
always fixes proc root to `/proc` and callback to `os.kill`.

- [ ] **Step 8: Run checker, logging, and audit GREEN tests**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest tests.python.test_runtime_log -v
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_route_runtime.cpp -o /tmp/test_route_runtime_logging
/tmp/test_route_runtime_logging
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_candidate_audit.cpp -o /tmp/test_g1_candidate_audit
/tmp/test_g1_candidate_audit
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_directional_candidates -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_visualizer_process -v
git diff --check
```

Expected: all synthetic schema/status/heading/clearance mutations are rejected for the intended reason; valid fixtures for all six heading codes pass.

- [ ] **Step 9: Commit physical directional diagnostics**

```bash
git add g1_candidate_audit.h motion_match_log.h controller.cpp \
  resources/check_g1_runtime_log.py \
  resources/audit_g1_directional_candidates.py \
  resources/g1_visualizer_process.py \
  tests/cpp/test_g1_candidate_audit.cpp tests/python/test_runtime_log.py \
  tests/python/test_directional_candidates.py \
  tests/python/test_visualizer_process.py
git commit -m "test: enforce directional G1 terrain gates"
```

---

### Task 8: Certify the Directional Matrix and Replace the Live Visualizer

**Files:**
- Create: `docs/superpowers/evidence/2026-07-15-g1-footprint-directional-certification.md`
- Generate, never commit: `/tmp/controller_g1_footprint_certified`
- Generate, never commit: `/tmp/g1-footprint-certification/`
- Consume without modifying: `/tmp/g1-terrain-footprint-runtime-v1/`
- Create only at final handoff, ignored: `resources/.g1_terrain-footprint-stage/`
- Update, ignored: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes the committed Tasks 1--7 source, immutable Task-3 flat/forward
  oracles, the external candidate pack, the full old-pack tree snapshot, and a
  dynamically discovered old live visualizer identity if one still exists.
- Produces a strict-kernel/fast-caller skeleton binary, Gates L1--L5 evidence,
  an explicit current-pack data decision, and exactly one verified replacement
  visualizer on `DISPLAY=:1`.

- [ ] **Step 1: Complete broad source reviews, freeze HEAD, and revalidate artifacts**

Before any certification binary/log is created, use `requesting-code-review`.
Give one fresh reviewer the approved spec, this complete plan, all producing
commits, and per-task review/test records for spec compliance. Resolve every
line-specific finding. Then give a different fresh reviewer the revised range
for failure atomicity, strict-FP boundaries, ownership, test authenticity, and
code quality. If either review causes a source change, rerun every affected
focused/full suite and repeat both broad reviews. Freeze `certified_head` only
after both are clean.

Then require:

```bash
git diff --check
test -d /tmp/g1-terrain-footprint-runtime-v1
test -d /tmp/g1-terrain-active-v1-python
test -s /tmp/g1-terrain-active-v1.TREE
test -s /tmp/g1-terrain-active-v1.SHA256SUMS
test -s /tmp/g1-terrain-active-v1.python.SHA256SUMS
sha256sum -c /tmp/g1-footprint-prechange-oracle/SHA256SUMS
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py \
  /tmp/g1-terrain-footprint-runtime-v1
(
  cd /tmp/g1-terrain-active-v1-python
  sha256sum -c /tmp/g1-terrain-active-v1.python.SHA256SUMS
)
(
  cd /tmp
  PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/g1-terrain-active-v1-python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      /tmp/g1-terrain-active-v1-python/resources/validate_g1_terrain_database.py \
      /home/ubuntu/projects/motion-matching/resources/g1_terrain
)
(
  cd resources/g1_terrain
  find . -printf '%y %m %P -> %l\n' | LC_ALL=C sort
) > /tmp/g1-terrain-active-v1.current.TREE
cmp /tmp/g1-terrain-active-v1.TREE \
  /tmp/g1-terrain-active-v1.current.TREE
(
  cd resources/g1_terrain
  sha256sum -c /tmp/g1-terrain-active-v1.SHA256SUMS
)
git rev-parse HEAD > /tmp/g1-footprint-certification.certified-head
```

Record and compare SHA256 values for `database.bin`, `terrain_features.bin`,
`terrain_support.bin`, and `validation.json` in the active and candidate packs;
all four pairs must match. Record their different scene-index digests.

Discover the old visualizer without a hard-coded PID. Enumerate `/proc/[0-9]*`,
select at most one live process whose resolved executable is
`/tmp/controller_g1_cert_task2`, and save its PID, `/proc/<pid>/stat` start-time
field 22, resolved executable, executable device/inode, and NUL-decoded command
line to `/tmp/g1-footprint-certification.old-visualizer`. Do not signal it.
Immediately before every later signal, re-read and compare all saved identity
fields so a reused PID can never be targeted.

Use this exact discovery program:

```bash
test ! -e /tmp/g1-footprint-certification.old-visualizer
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py discover \
  --executable /tmp/controller_g1_cert_task2 \
  --output /tmp/g1-footprint-certification.old-visualizer
```

- [ ] **Step 2: Run the complete C++ and Python unit suites**

Compile the strict kernel once, compile every caller to an object, and link
every test with that kernel using final link commands that contain no
`-ffast-math`:

```bash
mkdir -p /tmp/g1-footprint-certification/tests
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-footprint-certification/tests/g1-clearance.o
caller_tests=(
  cleanup_runtime g1_candidate_audit g1_clearance g1_command_runtime
  g1_controller_state g1_footprint_runtime g1_frame_transaction
  g1_frame_transaction_production g1_ik
  g1_skeleton motion_match_log route_runtime
  scene_runtime scene_switch support_matching support_runtime
  terrain_database terrain_runtime
)
for name in "${caller_tests[@]}"; do
  source="tests/cpp/test_${name}.cpp"
  define=()
  test "$name" = g1_ik && define=(-DG1_IK_ENABLE_TEST_SEAMS)
  test "$name" = g1_frame_transaction_production && \
    define=(-DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    "${define[@]}" -I. -c "$source" \
    -o "/tmp/g1-footprint-certification/tests/${name}-strict.o"
  g++ "/tmp/g1-footprint-certification/tests/${name}-strict.o" \
    /tmp/g1-footprint-certification/tests/g1-clearance.o \
    -o "/tmp/g1-footprint-certification/tests/${name}-strict"
  "/tmp/g1-footprint-certification/tests/${name}-strict"

  g++ -std=c++17 -O3 -ffast-math -DNDEBUG "${define[@]}" \
    -I. -c "$source" \
    -o "/tmp/g1-footprint-certification/tests/${name}-fast.o"
  g++ "/tmp/g1-footprint-certification/tests/${name}-fast.o" \
    /tmp/g1-footprint-certification/tests/g1-clearance.o \
    -o "/tmp/g1-footprint-certification/tests/${name}-fast"
  "/tmp/g1-footprint-certification/tests/${name}-fast"
done

/tmp/g1-footprint-certification/tests/g1_clearance-strict --query-parity \
  > /tmp/g1-footprint-certification/tests/clearance-strict.txt
/tmp/g1-footprint-certification/tests/g1_clearance-fast --query-parity \
  > /tmp/g1-footprint-certification/tests/clearance-fast.txt
cmp /tmp/g1-footprint-certification/tests/clearance-strict.txt \
  /tmp/g1-footprint-certification/tests/clearance-fast.txt
/tmp/g1-footprint-certification/tests/g1_ik-strict --parity \
  > /tmp/g1-footprint-certification/tests/ik-strict.txt
/tmp/g1-footprint-certification/tests/g1_ik-fast --parity \
  > /tmp/g1-footprint-certification/tests/ik-fast.txt
cmp /tmp/g1-footprint-certification/tests/ik-strict.txt \
  /tmp/g1-footprint-certification/tests/ik-fast.txt

/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  discover -s tests/python -v
```

Expected: every strict/fast caller passes and both parity transcripts are
byte-identical. In particular, all transitive clearance callers—including
controller state and scene switch—link the same strict object. The IK parity
transcript is the Task-5 fixed-supplied-endpoint fixture: both callers feed the
same endpoint bits into the strict kernel. Independently staged strict/fast
controller endpoint bits are not compared for byte equality.

- [ ] **Step 3: Run sanitizers and inspect the production seam**

Build and run the sanitizer matrix, direct fast-kernel negative, and production
seam checks exactly as follows:

```bash
san_flags=(-O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all)
g++ -std=c++17 "${san_flags[@]}" -fno-fast-math -ffp-contract=off \
  -frounding-math -I. -c g1_clearance.cpp \
  -o /tmp/g1-footprint-certification/tests/g1-clearance-san.o
san_tests=(g1_clearance g1_footprint_runtime g1_ik g1_command_runtime \
  g1_controller_state g1_frame_transaction \
  g1_frame_transaction_production g1_candidate_audit)
for name in "${san_tests[@]}"; do
  define=()
  test "$name" = g1_ik && define=(-DG1_IK_ENABLE_TEST_SEAMS)
  test "$name" = g1_frame_transaction_production && \
    define=(-DG1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
  g++ -std=c++17 "${san_flags[@]}" "${define[@]}" -I. \
    -c "tests/cpp/test_${name}.cpp" \
    -o "/tmp/g1-footprint-certification/tests/${name}-san.o"
  g++ "${san_flags[@]}" \
    "/tmp/g1-footprint-certification/tests/${name}-san.o" \
    /tmp/g1-footprint-certification/tests/g1-clearance-san.o \
    -o "/tmp/g1-footprint-certification/tests/${name}-san"
  ASAN_OPTIONS=detect_leaks=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "/tmp/g1-footprint-certification/tests/${name}-san"
done

if g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
     -c g1_clearance.cpp \
     -o /tmp/g1-footprint-certification/tests/forbidden-fast-kernel.o \
     2>/tmp/g1-footprint-certification/tests/forbidden-fast-kernel.stderr; then
  echo 'ERROR: clearance kernel accepted -ffast-math' >&2
  exit 1
fi
rg 'strict|fast-math|FAST_MATH' \
  /tmp/g1-footprint-certification/tests/forbidden-fast-kernel.stderr

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/compile_g1_ik_production.cpp \
  -o /tmp/g1-footprint-certification/tests/g1-ik-production.o
! nm -C /tmp/g1-footprint-certification/tests/g1-ik-production.o | \
  rg 'g1_ik_stage_swing_candidate_for_test'
if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
     -c tests/cpp/compile_g1_ik_seam_negative.cpp \
     -o /tmp/g1-footprint-certification/tests/g1-ik-negative.o \
     2>/tmp/g1-footprint-certification/tests/g1-ik-negative.stderr; then
  echo 'ERROR: private IK seam compiled in production mode' >&2
  exit 1
fi
rg 'g1_ik_stage_swing_candidate_for_test' \
  /tmp/g1-footprint-certification/tests/g1-ik-negative.stderr
for mode in VOID_CONTEXT PUBLICATION_CONTEXT ACCEPTED_CONTEXT; do
  stderr="/tmp/g1-footprint-certification/tests/frame-runner-${mode}.stderr"
  if g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
       -D"G1_FRAME_NEGATIVE_${mode}" \
       -c tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
       -o "/tmp/g1-footprint-certification/tests/frame-runner-${mode}.o" \
       2>"$stderr"; then
    echo "ERROR: frame runner accepted forbidden ${mode} context" >&2
    exit 1
  fi
  rg 'G1FrameStageRunner|convert|conversion|argument' "$stderr"
done
! rg -n 'strafe|forward_mode|lateral_mode' g1_footprint_runtime.h
```

Expected: every suite exits zero, no sanitizer emits a diagnostic, no private
IK seam is present, the production frame-runner type rejects opaque,
publication, and mutable accepted-state contexts, and the footprint
implementation contains no direction-specific behavior switch.

- [ ] **Step 4: Build the final binary with a strict kernel and fast caller**

```bash
mkdir -p /tmp/g1-footprint-certification/build
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -frounding-math \
  -frecord-gcc-switches \
  -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-footprint-certification/build/g1-clearance.o
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src -c controller.cpp \
  -o /tmp/g1-footprint-certification/build/controller.o
g++ /tmp/g1-footprint-certification/build/controller.o \
  /tmp/g1-footprint-certification/build/g1-clearance.o \
  -o /tmp/controller_g1_footprint_certified \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
sha256sum /tmp/controller_g1_footprint_certified \
  > /tmp/g1-footprint-certification/controller.sha256
readelf -S /tmp/g1-footprint-certification/build/g1-clearance.o | \
  rg '\.GCC\.command\.line'
readelf -p .GCC.command.line \
  /tmp/g1-footprint-certification/build/g1-clearance.o \
  > /tmp/g1-footprint-certification/build/g1-clearance.switches
rg -- '-fno-fast-math' \
  /tmp/g1-footprint-certification/build/g1-clearance.switches
rg -- '-ffp-contract=off' \
  /tmp/g1-footprint-certification/build/g1-clearance.switches
rg -- '-frounding-math' \
  /tmp/g1-footprint-certification/build/g1-clearance.switches
! rg -- '(^| )-ffast-math( |$)' \
  /tmp/g1-footprint-certification/build/g1-clearance.switches
test "$(cat /tmp/g1-footprint-certification.certified-head)" = \
  "$(git rev-parse HEAD)"
```

The final link command deliberately contains no `-ffast-math`. Store the
compiler version, command lines, binary hash, source HEAD, and scene-index hash
in the evidence report.

- [ ] **Step 5: Run Gate L3 in all six directions on every certified family**

Generate IK-off/IK-on pairs for this exact 30-cell matrix:

| Scene | Route | Heading codes |
|---|---|---|
| `stairs-shallow` | `ascent-landing-descent` | all six |
| `stairs-standard` | `ascent-landing-descent` | all six |
| `grail-curb-low` | `curb-forward` | all six |
| `ramp-05-up-down` | `up-landing-down` | all six |
| `ramp-10-up-down` | `up-landing-down` | all six |

Run:

```bash
mkdir -p /tmp/g1-footprint-certification/gate-l
quality_verdicts=/tmp/g1-footprint-certification/gate-l/quality-verdicts.tsv
: > "$quality_verdicts"
routes=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  grail-curb-low:curb-forward
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
)
headings=(
  forward backward positive-x negative-x
  diagonal-positive-x diagonal-negative-x
)
for specification in "${routes[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  read -r end_x end_z <<<"$(jq -r --arg route "$route" \
    '.routes[] | select(.id == $route) | .waypoints_xz[-1] | @tsv' \
    "/tmp/g1-terrain-footprint-runtime-v1/scenes/${scene}/scene.json")"
  for heading in "${headings[@]}"; do
    stem="${scene}__${route}__${heading}"
    off="/tmp/g1-footprint-certification/gate-l/${stem}-off.csv"
    on="/tmp/g1-footprint-certification/gate-l/${stem}-on.csv"
    for ik in 0 1; do
      log="$off"
      test "$ik" = 1 && log="$on"
      DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
        MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route \
        MM_TEST_ROUTE="$route" MM_TEST_FRAMES=800 \
        MM_TEST_HEADING="$heading" MM_TERRAIN_WEIGHT=4 MM_IK="$ik" \
        MM_LOG="$log" /tmp/controller_g1_footprint_certified
    done
    extra=(--compare-forward \
      "/tmp/g1-footprint-certification/gate-l/${scene}__${route}__forward-on.csv")
    test "$heading" = forward && extra=()
    baseline=()
    if test "$heading" = forward; then
      baseline=(--compare-forward-baseline \
        "/tmp/g1-footprint-prechange-oracle/forward-${scene}__${route}.csv")
    fi
    common=(--compare-ik-off "$off" "${extra[@]}" "${baseline[@]}" \
      --expected-end-x "$end_x" --expected-end-z "$end_z" \
      --expected-heading "$heading" --require-multilevel)
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      resources/check_g1_runtime_log.py "$on" --gate-l-safety-only \
      "${common[@]}"
    if /home/ubuntu/miniconda3/envs/diffsim/bin/python \
         resources/check_g1_runtime_log.py "$on" --gate-l \
         "${common[@]}"; then
      printf '%s\tPASS\n' "$stem" >> "$quality_verdicts"
    else
      printf '%s\tFAIL\n' "$stem" >> "$quality_verdicts"
    fi
  done
done

/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-footprint-certification/gate-l/stairs-standard__ascent-landing-descent__positive-x-on.csv \
  --gate-l2-pair \
  /tmp/g1-footprint-certification/gate-l/stairs-standard__ascent-landing-descent__negative-x-on.csv

for ik in 0 1; do
  DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
    MM_TERRAIN_SCENE=stairs-standard MM_TEST_MODE=route \
    MM_TEST_ROUTE=landing-side-exit-stress MM_TEST_FRAMES=800 \
    MM_TEST_HEADING=positive-x MM_TERRAIN_WEIGHT=4 MM_IK="$ik" \
    MM_LOG="/tmp/g1-footprint-certification/gate-l/landing-exit-stress-${ik}.csv" \
    /tmp/controller_g1_footprint_certified
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-footprint-certification/gate-l/landing-exit-stress-1.csv \
  --gate-l2-exit-stress --compare-ik-off \
  /tmp/g1-footprint-certification/gate-l/landing-exit-stress-0.csv
```

Expected: 30 `VALID gate-l-safety` summaries and exactly 30 recorded full
quality verdicts, plus one `VALID gate-l2-pair` and one branch-explicit
`VALID gate-l2-exit-stress`. Every normal route completes at exact 25 Hz
without a footprint/IK safe-stop; forward and backward are certified by the
same physical path as diagonal and lateral cases. Both
`stairs-standard` lateral directions prove from target/contact levels that each
leg served as uphill and downhill support. The dedicated 0.36 m stress route
either completes coherently or safe-stops before unsafe publication. A full
quality failure does not abort before audit; it remains a failing Gate L cell
until Task 8 classifies it from behavior-inert candidate evidence. The five
forward runs match their immutable pre-footprint oracles and remain the
performance reference for the other headings.

- [ ] **Step 6: Run the tangential boundary and flat invariance gates**

First run Gate L1 on the authenticated boundary route:

```bash
for ik in 0 1; do
  DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
    MM_TERRAIN_SCENE=mixed-multilevel MM_TEST_MODE=route \
    MM_TEST_ROUTE=tangent-level-boundary MM_TEST_FRAMES=800 \
    MM_TEST_HEADING=positive-x MM_TERRAIN_WEIGHT=4 MM_IK="$ik" \
    MM_LOG="/tmp/g1-footprint-certification/gate-l/tangent-${ik}.csv" \
    /tmp/controller_g1_footprint_certified
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-footprint-certification/gate-l/tangent-1.csv --gate-l \
  --compare-ik-off \
    /tmp/g1-footprint-certification/gate-l/tangent-0.csv \
  --expected-end-x 0.62 --expected-end-z 6.0 \
  --expected-heading positive-x --require-multilevel
```

Then run Gate L4 over the exact eight 100-frame route/heading/category pairs
captured before footprint integration:

```bash
flat_cases=(
  flat-positive-z:forward:forward
  flat-positive-z:backward:backward
  flat-positive-z:positive-x:left
  flat-positive-z:negative-x:right
  flat-positive-x:positive-x:forward
  flat-positive-x:negative-x:backward
  flat-positive-x:forward:right
  flat-positive-x:backward:left
)
for specification in "${flat_cases[@]}"; do
  IFS=: read -r route heading relative <<<"$specification"
  stem="${route}__${heading}__${relative}"
  final="/tmp/g1-footprint-certification/gate-l/flat-${stem}.csv"
  baseline="/tmp/g1-footprint-prechange-oracle/flat-${stem}.csv"
  DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
    MM_TERRAIN_SCENE=stairs-shallow MM_TEST_MODE=route \
    MM_TEST_ROUTE="$route" MM_TEST_FRAMES=100 \
    MM_TEST_HEADING="$heading" MM_TERRAIN_WEIGHT=4 MM_IK=0 \
    MM_LOG="$final" /tmp/controller_g1_footprint_certified
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$final" --gate-l \
    --compare-flat-baseline "$baseline" --expected-heading "$heading" \
    --expected-relative-direction "$relative"
done
```

Expected: tangent traversal completes, reports a footprint split no later than
the root/contact split, and retains exact positive-X heading bits. All eight
flat runs match their immutable oracle bit-for-bit, report no multilevel
footprint or stop, preserve the expected heading constant, and collectively
cover each cardinal relative direction under two absolute world headings.

- [ ] **Step 7: Re-run every inherited terrain, IK, switching, and cleanup gate**

First rerun the exact inherited 375-frame Gate A:

```bash
mkdir -p /tmp/g1-footprint-certification/inherited
DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
  MM_TERRAIN_SCENE=grail-curb-default MM_TEST_MODE=terrain \
  MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=375 MM_IK=0 \
  MM_LOG=/tmp/g1-footprint-certification/inherited/gate-a.csv \
  /tmp/controller_g1_footprint_certified
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-footprint-certification/inherited/gate-a.csv --gate-a
```

Generate fresh 800-frame weight-zero/weight-four IK-off pairs for the six Gate
C routes, then check each weight-four log against its paired control:

```bash
gate_c=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  stairs-unseen-variable:ascent-landing-descent
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
  mixed-multilevel:full-course
)
for specification in "${gate_c[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  stem="${scene}__${route}"
  for weight in 0 4; do
    DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
      MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route \
      MM_TEST_ROUTE="$route" MM_TEST_FRAMES=800 \
      MM_TERRAIN_WEIGHT="$weight" MM_IK=0 \
      MM_LOG="/tmp/g1-footprint-certification/inherited/gate-c-${stem}-w${weight}.csv" \
      /tmp/controller_g1_footprint_certified
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "/tmp/g1-footprint-certification/inherited/gate-c-${stem}-w4.csv" \
    --gate-c --compare-control \
    "/tmp/g1-footprint-certification/inherited/gate-c-${stem}-w0.csv"
done
```

Generate both 600-frame Gate-D routes with IK enabled so physical clearance is
certified rather than inferred from blocked distance:

```bash
for route in wall-safe-stop ramp-safe-stop; do
  log="/tmp/g1-footprint-certification/inherited/gate-d-${route}.csv"
  DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
    MM_TERRAIN_SCENE=blocked-course MM_TEST_MODE=route \
    MM_TEST_ROUTE="$route" MM_TEST_FRAMES=600 \
    MM_TERRAIN_WEIGHT=4 MM_IK=1 MM_LOG="$log" \
    /tmp/controller_g1_footprint_certified
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$log" --gate-d
done
```

Re-run matched Gate E pairs for the accepted nine-route matrix:

```bash
gate_e=(
  grail-curb-low:curb-forward
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  stairs-unseen-variable:ascent-landing-descent
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
  cross-slope-05:forward-cross-slope
  cross-slope-10:forward-cross-slope
  mixed-multilevel:full-course
)
for specification in "${gate_e[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  stem="${scene}__${route}"
  off="/tmp/g1-footprint-certification/inherited/gate-e-${stem}-off.csv"
  on="/tmp/g1-footprint-certification/inherited/gate-e-${stem}-on.csv"
  for ik in 0 1; do
    log="$off"; test "$ik" = 1 && log="$on"
    DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
      MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route \
      MM_TEST_ROUTE="$route" MM_TEST_FRAMES=800 \
      MM_TERRAIN_WEIGHT=4 MM_IK="$ik" MM_LOG="$log" \
      /tmp/controller_g1_footprint_certified
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$on" --gate-e \
    --ik-baseline "$off" --expected-scene "$scene" \
    --expected-route "$route"
done
```

Run the four inherited class-2 Gate-E stress pairs:

```bash
gate_e_stress=(
  grail-curb-default:curb-forward
  grail-curb-medium:curb-forward
  grail-curb-high:curb-forward
  ramp-15-stress:up-landing-down
)
for specification in "${gate_e_stress[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  stem="${scene}__${route}"
  off="/tmp/g1-footprint-certification/inherited/gate-e-stress-${stem}-off.csv"
  on="/tmp/g1-footprint-certification/inherited/gate-e-stress-${stem}-on.csv"
  for ik in 0 1; do
    log="$off"; test "$ik" = 1 && log="$on"
    DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
      MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route \
      MM_TEST_ROUTE="$route" MM_TEST_FRAMES=800 \
      MM_TERRAIN_WEIGHT=4 MM_IK="$ik" MM_LOG="$log" \
      /tmp/controller_g1_footprint_certified
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$on" --gate-e-stress \
    --ik-baseline "$off" --expected-scene "$scene" \
    --expected-route "$route"
done
```

Keep each checker-reported `traverse` or `safe-stop` branch; do not relabel a
stress scene as class 1.

Finally run two complete ordered scene cycles and one malformed-candidate
cycle:

```bash
SCENES=grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,mixed-multilevel,blocked-course
DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=700 \
  MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_LOG=/tmp/g1-footprint-certification/inherited/gate-f.csv \
  MM_CLEANUP_LOG=/tmp/g1-footprint-certification/inherited/gate-f-cleanup.json \
  /tmp/controller_g1_footprint_certified
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-footprint-certification/inherited/gate-f.csv \
  --gate-f --expected-scenes "$SCENES"
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
path = "/tmp/g1-footprint-certification/inherited/gate-f-cleanup.json"
with open(path, encoding="utf-8") as stream:
    value = json.load(stream)
assert value == {
    "exit_code": 0, "live_model_count": 0, "log_closed": True,
    "model_load_count": 28, "model_unload_count": 28,
    "motion_pack_load_count": 1, "window_closed": True,
}, value
print("VALID gate-f cleanup")
PY
```

Construct and run the malformed overlay exactly:

```bash
BAD_ROOT=$(mktemp -d /tmp/g1-footprint-malformed-pack.XXXXXX)
for file in database.bin terrain_features.bin terrain_support.bin \
            manifest.json validation.json; do
  ln -s "$(realpath /tmp/g1-terrain-footprint-runtime-v1/$file)" \
    "$BAD_ROOT/$file"
done
mkdir "$BAD_ROOT/scenes"
ln -s "$(realpath /tmp/g1-terrain-footprint-runtime-v1/scenes/index.json)" \
  "$BAD_ROOT/scenes/index.json"
for scene in ${SCENES//,/ }; do
  if test "$scene" = stairs-shallow; then
    cp -a "/tmp/g1-terrain-footprint-runtime-v1/scenes/$scene" \
      "$BAD_ROOT/scenes/$scene"
    truncate -s 1 "$BAD_ROOT/scenes/$scene/terrain.obj"
  else
    ln -s "$(realpath /tmp/g1-terrain-footprint-runtime-v1/scenes/$scene)" \
      "$BAD_ROOT/scenes/$scene"
  fi
done
DISPLAY=:1 G1_TERRAIN_DIR="$BAD_ROOT" \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=350 \
  MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_LOG=/tmp/g1-footprint-certification/inherited/gate-f-malformed.csv \
  MM_CLEANUP_LOG=/tmp/g1-footprint-certification/inherited/gate-f-malformed-cleanup.json \
  /tmp/controller_g1_footprint_certified
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-footprint-certification/inherited/gate-f-malformed.csv \
  --expect-switch-failure
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
path = "/tmp/g1-footprint-certification/inherited/gate-f-malformed-cleanup.json"
with open(path, encoding="utf-8") as stream:
    value = json.load(stream)
assert value["exit_code"] == 0, value
assert value["live_model_count"] == 0, value
assert value["log_closed"] is True and value["window_closed"] is True, value
assert value["model_load_count"] == value["model_unload_count"], value
assert value["motion_pack_load_count"] == 1, value
print("VALID gate-f malformed cleanup")
PY
```

Require the previous valid scene to survive, one motion-pack load, balanced
candidate ownership, a closed CSV/window, and zero live models.

Expected inherited evidence: one Gate A pass, six Gate C passes, two corrected Gate D passes,
nine Gate E passes, four branch-aware Gate-E-stress passes, one 28-generation
Gate F pass, and one controlled malformed-switch pass. No previously accepted
terrain, descent, multilevel, rate, ownership, or cleanup contract is waived.

- [ ] **Step 8: Produce candidate-coverage evidence and make the data decision**

Only after all 30 safety-only cells and every inherited gate pass, run the
behavior-inert audit for all 30 exact scene/route/heading cells before
interpreting any recorded full-quality failure. A physical, transactional,
completion, support, or exact-heading-bit failure remains a code defect and
does not enter this audit phase.

```bash
mkdir -p /tmp/g1-footprint-certification/audit
routes=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  grail-curb-low:curb-forward
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
)
headings=(
  forward backward positive-x negative-x
  diagonal-positive-x diagonal-negative-x
)
for specification in "${routes[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  for heading in "${headings[@]}"; do
    stem="${scene}__${route}__${heading}"
    jsonl="/tmp/g1-footprint-certification/audit/${stem}.jsonl"
    summary="/tmp/g1-footprint-certification/audit/${stem}.summary.json"
    audit_log="/tmp/g1-footprint-certification/audit/${stem}.csv"
    control_log="/tmp/g1-footprint-certification/gate-l/${stem}-on.csv"
    DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
      MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route \
      MM_TEST_ROUTE="$route" MM_TEST_FRAMES=800 \
      MM_TEST_HEADING="$heading" MM_TERRAIN_WEIGHT=4 MM_IK=1 \
      MM_CANDIDATE_AUDIT="$jsonl" MM_LOG="$audit_log" \
      /tmp/controller_g1_footprint_certified
    cmp "$control_log" "$audit_log"
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      resources/audit_g1_directional_candidates.py \
      --manifest /tmp/g1-terrain-footprint-runtime-v1/manifest.json \
      --audit "$jsonl" --output "$summary"
  done
done
sha256sum /tmp/g1-footprint-certification/audit/* \
  > /tmp/g1-footprint-certification/audit/SHA256SUMS
sha256sum -c /tmp/g1-footprint-certification/audit/SHA256SUMS
```

Require each summary to contain samples, exact three-threshold counts, finite
top costs, range/source mappings, and top-16 source coverage. Hash all audit
inputs/outputs. Require exactly 30 summaries and exactly 30 quality-verdict
rows. Every audit-enabled CSV must be byte-identical to its original control,
so the original safety and quality verdicts carry over exactly; any mismatch
is a code defect in the supposedly behavior-inert audit.

Then classify the observed result using only recorded evidence:

- If Gates L1--L5 and inherited gates pass, all 30 full-quality verdicts pass,
  and the matrix logs plus visual smoke show coherent planting, the current motion pack
  is sufficient for this geometry phase.
- If physical clearance, planting, route completion, support bounds, and exact
  heading bits pass but rendered lateral/diagonal quality or heading-error
  thresholds fail, do not rotate toward travel and do not relabel turning
  clips. For every exact failed heading/scene/route cell, record its selected
  source clips, candidate counts/costs/top-16 coverage, and the already-measured
  sustained-terrain-strafe coverage.
  Conclude that genuine independent-heading lateral G1 data is the next task
  and explicitly mark the overall lateral-terrain feature not ready despite a
  completed geometry phase.
- If a physical, transactional, or inherited gate fails, classify it as a code
  defect, retain the old live visualizer, and return to the owning task. Do not
  call it a data problem.

If any gate/audit result causes a source fix, discard every binary, log, hash,
and summary produced in Task 8, repeat the two broad reviews, freeze the new
HEAD, and rerun Task 8 from Step 1. No pre-fix certification evidence may be
reused.

Create
`docs/superpowers/evidence/2026-07-15-g1-footprint-directional-certification.md`
with source/artifact hashes, compiler commands, complete gate counts, numeric
worst cases, both review outcomes, the data decision, old/new PID disposition,
and any failure/rollback action. Create `docs/superpowers/evidence/` first if
absent. Use actual values only.

- [ ] **Step 9: Freeze evidence and smoke the external candidate visualizer**

Only when every safety/inherited gate and all 30 full-quality verdicts pass,
the audit is byte-inert, and the data decision is sufficient, re-run:

```bash
git diff --check
git status --short
sha256sum -c /tmp/g1-footprint-certification/controller.sha256
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py \
  /tmp/g1-terrain-footprint-runtime-v1
test "$(cat /tmp/g1-footprint-certification.certified-head)" = \
  "$(git rev-parse HEAD)"
(
  cd resources/g1_terrain
  sha256sum -c /tmp/g1-terrain-active-v1.SHA256SUMS
)
```

Require status to contain only the user's pre-existing modified resources and
untracked binaries/logs plus the intended evidence file. Leave the evidence
draft unstaged. Start a candidate-path skeleton while the old visualizer and
active pack remain untouched:

```bash
mkdir -p /tmp/g1-footprint-certification/live
DISPLAY=:1 G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  /tmp/controller_g1_footprint_certified \
  >/tmp/g1-footprint-certification/live/candidate.stdout \
  2>/tmp/g1-footprint-certification/live/candidate.stderr &
candidate_pid=$!
echo "$candidate_pid" \
  > /tmp/g1-footprint-certification/live/candidate.pid
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py capture-wait \
  --pid "$candidate_pid" \
  --executable /tmp/controller_g1_footprint_certified \
  --timeout-ms 2000 \
  --output /tmp/g1-footprint-certification/live/candidate.identity.json
sleep 2
kill -0 "$candidate_pid"
DISPLAY=:1 xdotool search --onlyvisible --pid "$candidate_pid" >/dev/null
test ! -s /tmp/g1-footprint-certification/live/candidate.stderr
```

Keep it alive for at least 50 update frames and inspect the mixed terrain,
skeleton, upper/lower planting, lateral travel, descent, ramp, and open side
aprons. If it fails, signal only the captured candidate identity through the
checked process helper (`TERM`, then checked `KILL` only if still identical and
live); because no publication has occurred, the old active pack and old
visualizer require no rollback.

- [ ] **Step 10: Atomically publish, verify the active-pack visualizer, or exchange back**

Only after the candidate-path visual smoke passes, revalidate the saved old
process identity and pause it with `SIGSTOP`; a missing saved process is fine,
but any identity mismatch aborts without signaling. Then stage and exchange:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification.old-visualizer \
  --signal STOP
test ! -e resources/.g1_terrain-footprint-stage
verify_old_pack() {
  root=$1
  tree_output=$2
  (
    cd "$root"
    find . -printf '%y %m %P -> %l\n' | LC_ALL=C sort
  ) > "$tree_output"
  cmp /tmp/g1-terrain-active-v1.TREE "$tree_output" && (
    cd "$root"
    sha256sum -c /tmp/g1-terrain-active-v1.SHA256SUMS
  )
}
reverse_exchange_prelaunch() {
  /home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import os
from resources.g1_terrain_builder.artifacts import (
    _fsync_parent, _locked_parent, _rename_exchange,
)

active = os.path.abspath("resources/g1_terrain")
stage = os.path.abspath("resources/.g1_terrain-footprint-stage")
with _locked_parent(os.path.dirname(active)) as descriptor:
    _rename_exchange(stage, active)
    _fsync_parent(descriptor)
print("ROLLED BACK old active pack before final launch")
PY
}
recover_prelaunch() {
  if ! verify_old_pack resources/g1_terrain \
       /tmp/g1-terrain-active-v1.prelaunch-active.TREE; then
    if ! verify_old_pack resources/.g1_terrain-footprint-stage \
         /tmp/g1-terrain-active-v1.prelaunch-stage.TREE; then
      return 1
    fi
    reverse_exchange_prelaunch || return 1
    verify_old_pack resources/g1_terrain \
      /tmp/g1-terrain-active-v1.prelaunch-rollback.TREE || return 1
  fi
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/g1_visualizer_process.py signal \
    --identity /tmp/g1-footprint-certification/live/candidate.identity.json \
    --signal TERM
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/g1_visualizer_process.py signal \
    --identity /tmp/g1-footprint-certification.old-visualizer \
    --signal CONT
}
if ! /home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
import os
import shutil

from resources.g1_terrain_builder.artifacts import (
    _fsync_parent, _fsync_tree, _locked_parent, _rename_exchange,
)

candidate = "/tmp/g1-terrain-footprint-runtime-v1"
active = os.path.abspath("resources/g1_terrain")
stage = os.path.abspath("resources/.g1_terrain-footprint-stage")
if os.path.lexists(stage):
    raise RuntimeError(f"refusing pre-existing final stage: {stage}")
shutil.copytree(candidate, stage, copy_function=shutil.copy2)
_fsync_tree(stage)
exchanged = False
with _locked_parent(os.path.dirname(active)) as descriptor:
    try:
        _rename_exchange(stage, active)
        exchanged = True
        _fsync_parent(descriptor)
    except BaseException:
        if exchanged:
            _rename_exchange(stage, active)
            _fsync_parent(descriptor)
        raise
print("PUBLISHED candidate; old pack retained at", stage)
PY
then
  if ! recover_prelaunch; then
    echo 'ERROR: pre-launch old-pack recovery could not be proved' >&2
    exit 2
  fi
  exit 1
fi
if ! /home/ubuntu/miniconda3/envs/diffsim/bin/python \
       resources/validate_g1_terrain_database.py resources/g1_terrain || \
   ! verify_old_pack resources/.g1_terrain-footprint-stage \
       /tmp/g1-terrain-active-v1.prelaunch-stage.TREE; then
  if ! recover_prelaunch; then
    echo 'ERROR: pre-launch old-pack recovery could not be proved' >&2
    exit 2
  fi
  exit 1
fi
```

The exchange happens only after the old PID has been identity-checked and
paused. The stage now contains the complete old pack and remains until final
evidence is committed. The displayed pre-launch recovery is independent of
`final.pid`: on staging/exchange failure it first proves whether active already
matches the old TREE/SHA; otherwise it requires stage to match old, reverses
the exchange, and re-proves active old. On validation or backup-verification
failure it takes the same path. Only after an old-pack proof does it terminate
the candidate-path smoke, checked-resume the old identity, and exit without a
final launch. Otherwise launch a second replacement that resolves the active
path:

```bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  /tmp/controller_g1_footprint_certified \
  >/tmp/g1-footprint-certification/live/final.stdout \
  2>/tmp/g1-footprint-certification/live/final.stderr &
final_pid=$!
echo "$final_pid" > /tmp/g1-footprint-certification/live/final.pid
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py capture-wait \
  --pid "$final_pid" \
  --executable /tmp/controller_g1_footprint_certified \
  --timeout-ms 2000 \
  --output /tmp/g1-footprint-certification/live/final.identity.json
sleep 2
kill -0 "$final_pid"
DISPLAY=:1 xdotool search --onlyvisible --pid "$final_pid" >/dev/null
test ! -s /tmp/g1-footprint-certification/live/final.stderr
```

Keep it alive for 50 updates and repeat the candidate smoke. On any launch,
window, asset, clearance, or manual-smoke failure, terminate `final_pid` and
the candidate-path PID, then reverse the exchange with the same locked-parent
transaction:

```bash
final_pid=$(cat /tmp/g1-footprint-certification/live/final.pid)
candidate_pid=$(cat /tmp/g1-footprint-certification/live/candidate.pid)
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification/live/final.identity.json \
  --signal TERM
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification/live/candidate.identity.json \
  --signal TERM
for identity in final candidate; do
  pid_file="/tmp/g1-footprint-certification/live/${identity}.pid"
  pid=$(cat "$pid_file")
  for unused in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      resources/g1_visualizer_process.py signal \
      --identity "/tmp/g1-footprint-certification/live/${identity}.identity.json" \
      --signal KILL
  fi
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import os
from resources.g1_terrain_builder.artifacts import (
    _fsync_parent, _locked_parent, _rename_exchange,
)

active = os.path.abspath("resources/g1_terrain")
stage = os.path.abspath("resources/.g1_terrain-footprint-stage")
reversed_exchange = False
with _locked_parent(os.path.dirname(active)) as descriptor:
    try:
        _rename_exchange(stage, active)
        reversed_exchange = True
        _fsync_parent(descriptor)
    except BaseException:
        if reversed_exchange:
            _rename_exchange(stage, active)
            _fsync_parent(descriptor)
        raise
print("ROLLED BACK old active pack")
PY
```

After rollback, compare the active pack against both complete old snapshots;
do **not** run the new validator against it. Revalidate the saved PID identity,
send `SIGCONT`, and prove the old window is visible again. Retain the failed
candidate and logs for diagnosis.

```bash
(
  cd resources/g1_terrain
  find . -printf '%y %m %P -> %l\n' | LC_ALL=C sort
) > /tmp/g1-terrain-active-v1.rollback.TREE
cmp /tmp/g1-terrain-active-v1.TREE \
  /tmp/g1-terrain-active-v1.rollback.TREE
(
  cd resources/g1_terrain
  sha256sum -c /tmp/g1-terrain-active-v1.SHA256SUMS
)
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification.old-visualizer \
  --signal CONT
```

On success, terminate the candidate-path smoke PID. Revalidate the old PID's
saved start time, executable, device/inode, and command line once more; send it
`SIGTERM` followed by `SIGCONT` so the stopped process can exit. Poll for at
most five seconds, identity-check again before any fallback `SIGKILL`, and
confirm only `final_pid` owns a visible project window. Never signal a reused
PID.

```bash
candidate_pid=$(cat /tmp/g1-footprint-certification/live/candidate.pid)
final_pid=$(cat /tmp/g1-footprint-certification/live/final.pid)
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification/live/candidate.identity.json \
  --signal TERM
for unused in $(seq 1 50); do
  kill -0 "$candidate_pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$candidate_pid" 2>/dev/null; then
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/g1_visualizer_process.py signal \
    --identity /tmp/g1-footprint-certification/live/candidate.identity.json \
    --signal KILL
fi
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification.old-visualizer \
  --signal TERM
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/g1_visualizer_process.py signal \
  --identity /tmp/g1-footprint-certification.old-visualizer \
  --signal CONT
old_pid=$(/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
value = json.load(open(
    "/tmp/g1-footprint-certification.old-visualizer", encoding="utf-8"))
print("" if value is None else value["pid"])
PY
)
if test -n "$old_pid"; then
  for unused in $(seq 1 50); do
    kill -0 "$old_pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$old_pid" 2>/dev/null; then
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      resources/g1_visualizer_process.py signal \
      --identity /tmp/g1-footprint-certification.old-visualizer \
      --signal KILL
  fi
fi
kill -0 "$final_pid"
DISPLAY=:1 xdotool search --onlyvisible --pid "$final_pid" >/dev/null
```

Reload `candidate_pid` and `final_pid` from their recorded files when Step-10
commands run in separate shells; process signaling itself always uses the
captured identity documents and never depends on a shell-local identity cache.

Record the new PID and final live scene in the evidence/progress ledger. Then
commit only the completed evidence file:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
sha256sum /tmp/controller_g1_footprint_certified \
  resources/g1_terrain/manifest.json \
  > /tmp/g1-footprint-certification/live/final-hashes.txt
git add docs/superpowers/evidence/2026-07-15-g1-footprint-directional-certification.md
git diff --cached --check
git commit -m "docs: certify directional G1 terrain footprints"
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
rm -rf resources/.g1_terrain-footprint-stage
kill -0 "$final_pid"
```

After the commit and one final artifact validation, remove only the verified
old-pack directory `resources/.g1_terrain-footprint-stage`. Leave the one
certified skeleton visualizer open for the user.

Expected: the user receives one current mixed-multilevel visualizer whose
forward/backward behavior retains the accepted quality and whose lateral,
diagonal, tangential, stair, curb, and ramp steps all use the same certified
physical-footprint path without terrain-induced rotation.
