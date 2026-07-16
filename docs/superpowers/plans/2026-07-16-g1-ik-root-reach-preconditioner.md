# G1 IK Root-Reach Preconditioner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep exact recorded-contact terrain targets reachable by applying the smallest dual-certified, reversible world-Y translation to only the IK candidate root: the nearest binary32 delta both admitted by the baseline analytic interval and authenticated against the adjusted production target.

**Architecture:** Move the shared reach-shell and physical-sole target arithmetic into a strict-FP translation unit, then use it from both the existing production solver and a fixed-size two-foot interval planner. `g1_ik_frame_begin` computes the plan from the untouched support-retargeted pose, applies its binary32 delta only to scratch bone `G1_Simulation`, and carries the value-only plan through existing result, rollback, equality, digest, rejection, and accepted-diagnostic owners.

**Tech Stack:** C++17; Daniel Holden array/vector/quaternion types; IEEE-754 binary32/binary64; strict-FP `g1_ik_root_reach.cpp`; existing `g1_ik.h`, `g1_ik_runtime.h`, `g1_frame_transaction.h`, `g1_clearance.cpp`, Raylib controller; standalone C++ and Python certification gates.

## Global Constraints

- The runtime cadence remains exact 25 Hz (`dt` binary32 `0.04f`).
- The correction cap is exactly `[-0.05 m, +0.05 m]`.
- Choose the dual-certified binary32 root-Y delta nearest positive zero: remain inside the baseline analytic proof domain, move a rounded boundary toward that interval's interior, and authenticate the adjusted production target through production projection. Do not probe outside the analytic domain merely because adjusted recomputation makes a nearer float reachable.
- Change only reversible `scratch_positions(G1_Simulation).y`; do not mutate support state, simulation X/Z, command, trajectory, heading, matcher input, contact timing, target, lock, local rotation, terrain artifact, or motion data.
- IK disabled, no recorded contact, and already-reachable contacts are exact no-ops; canonical zero is positive zero.
- Empty/out-of-cap/minimum-shell intersections leave the root unchanged so the ordinary foot stage publishes the existing finite `target-unreachable` result.
- Nonfinite input, topology/alias failure, or arithmetic failure is a global error with no output assignment.
- Preserve the existing left-then-right IK order, 41-entry swing ladder, clearance gates, and atomic outer rollback.
- Add no runtime option and no CSV column. Existing `support_retargeted_hips_y` and `ik_adjusted_hips_y` expose the delta.
- `g1_clearance.cpp` and `g1_ik_root_reach.cpp` are separate strict-FP objects built with `-fno-fast-math -ffp-contract=off -frounding-math`; callers may use `-ffast-math`, final links may not, and LTO is forbidden.
- Terrain packs remain separate. Do not replace the running visualizer until the complete Task 8 certification passes.

## File Structure

- Create `g1_ik_root_reach.cpp`: strict-FP shared reach shell, physical-sole target, fixed-size interval planner, boundary materialization, and checked root-Y application.
- Modify `ik.h`: add the value-only effective-shell type/declaration and make `ik_project_target` consume the shared strict shell.
- Modify `g1_ik.h`: add the value-only physical target and root plan types/declarations; make named physical-sole IK consume the shared target derivation.
- Modify `g1_ik_runtime.h`: retain the plan in `G1IkFrameResult`, invoke/apply it at frame begin, and authenticate canonical/accepted/rejected forms.
- Modify `g1_controller_state.h`: require a valid plan on accepted controller state and authenticate its relationship to the current contact schedule.
- Modify `g1_frame_transaction.h`: include the plan in equality and validate the rendered/support root-Y relationship.
- Modify `controller.cpp`: include the plan in the accepted-state logical digest only; do not change logging schema or runtime controls.
- Modify `tests/cpp/test_g1_ik.cpp`: strict math, authentic geometry, no-op, interval, boundary, alias, rollback, and parity tests.
- Modify `tests/cpp/test_g1_frame_transaction.cpp`: logical ownership, accepted/rejected publication, immutable command/heading, and rollback tests.
- Modify `tests/cpp/test_g1_frame_transaction_production.cpp`: production-stage root-only mutation, diagnostic relation, no-seam, closure, and controller build/link tests.
- Modify `tests/cpp/compile_g1_ik_production.cpp`: compile the public production surface without a test seam.
- Modify `docs/superpowers/specs/2026-07-16-g1-ik-root-reach-preconditioner-design.md` only if implementation review discovers a real design correction; do not rewrite it to match an accidental implementation.

---

### Task 1: Shared Strict Reach and Physical-Sole Math

**Files:**
- Create: `g1_ik_root_reach.cpp`
- Modify: `ik.h` near `IKTargetProjection` and `ik_project_target`
- Modify: `g1_ik.h` near `G1FootTarget` and `g1_apply_named_physical_sole_ik`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Consumes: `vec3`, `quat`, `G1LegConfig`, `G1FootTarget`, checked FK, surface-aligned rotation, `terrain_v2_round_output`, and current `ik_project_target` semantics.
- Produces:

```cpp
struct IKReachShell {
    double minimum_distance_m = 0.0;
    double maximum_distance_m = 0.0;
    float minimum_distance_f32_m = 0.0f;
    float maximum_distance_f32_m = 0.0f;
};

bool ik_effective_reach_shell(
    IKReachShell& output,
    vec3 root,
    vec3 middle,
    vec3 end,
    float reach_buffer_m);

struct G1PhysicalSolePositionTarget {
    quat contact_rotation;
    vec3 contact_origin;
    vec3 ankle_target;
};

bool g1_physical_sole_position_target(
    G1PhysicalSolePositionTarget& output,
    vec3 current_contact_origin,
    quat current_contact_rotation,
    vec3 current_ankle_origin,
    const G1LegConfig& config,
    vec3 desired_sole_center,
    vec3 desired_sole_normal,
    char* error,
    int error_capacity);
```

- [ ] **Step 1: Write RED tests for the shared reach shell**

Add tests which call `ik_effective_reach_shell` on the current ordinary, nearly extended, folded/minimum-shell, exact-boundary, one-ULP-inside, and one-ULP-outside fixtures. For each valid fixture, call `ik_project_target` with the same points and assert bit equality between `projection.minimum_distance_m` / `maximum_distance_m` and the shell float fields. Poison the output before invalid-input and output/input-alias calls and assert byte-for-byte unchanged output.

The essential oracle is:

```cpp
IKReachShell shell = {};
check(ik_effective_reach_shell(
          shell, hip, knee, ankle, config.reach_buffer_m),
      "shared effective reach shell materializes");
IKTargetProjection projection = {};
check(ik_project_target(
          projection, hip, knee, ankle, requested,
          config.reach_buffer_m),
      "production projection consumes the shared shell");
check(terrain_float_bits(projection.minimum_distance_m) ==
          terrain_float_bits(shell.minimum_distance_f32_m) &&
      terrain_float_bits(projection.maximum_distance_m) ==
          terrain_float_bits(shell.maximum_distance_f32_m),
      "projection and planner own one shell implementation");
```

- [ ] **Step 2: Run RED and record the missing-interface failure**

Run:

```bash
mkdir -p /tmp/g1-root-reach-plan/task1
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-root-reach-plan/task1/test-red.o
```

Expected: compilation fails only because `IKReachShell`, `ik_effective_reach_shell`, or the physical-target interface is not yet defined.

- [ ] **Step 3: Add RED physical-sole target tests**

Use the existing G1 fixture FK and both named leg configs. Cover flat `(0,1,0)`, ramp, and cross-slope normals; verify the helper's contact rotation is unit, its sole centroid equals the requested center within existing checked binary32 residual rules, and its ankle target exactly matches `G1LegSolveResult::requested_ankle_target` from `g1_apply_named_physical_sole_ik`. Repeat with sentinel outputs for nonunit rotations, malformed config, nonfinite center/normal, and all output/input/error overlap classes.

```cpp
G1PhysicalSolePositionTarget target = {};
check(g1_physical_sole_position_target(
          target,
          globals(config.contact), rotations(config.contact),
          globals(config.ankle), config,
          desired_center, desired_normal, error, sizeof(error)),
      "shared physical sole target materializes");
G1LegSolveResult position = {};
G1FootOrientationResult orientation = {};
check(g1_apply_named_physical_sole_ik(
          output_rotations, local_positions, local_rotations, parents,
          config, desired_center, desired_normal,
          position, orientation, error, sizeof(error)),
      "production physical sole solve succeeds");
check(g1_ik_vec3_bits_equal(
          target.ankle_target, position.requested_ankle_target),
      "planner and production solver consume one ankle target");
```

- [ ] **Step 4: Implement the strict shared functions and refactor production consumers**

In `ik.h`, declare `IKReachShell` and `ik_effective_reach_shell`. Replace the duplicated segment-length/minimum/maximum block in `ik_project_target` with one call and consume its precise doubles for reach comparisons and its binary32 fields for diagnostics.

In `g1_ik.h`, declare `G1PhysicalSolePositionTarget` and `g1_physical_sole_position_target`. In `g1_apply_named_physical_sole_ik`, call the helper after checked FK, pass its `ankle_target` directly to `g1_apply_named_position_ik`, and retain its `contact_rotation` for the existing bounded orientation stage. Remove only the now-duplicated target rotation/sole offset/contact-origin/ankle-target math.

Create `g1_ik_root_reach.cpp` with this guard and transactional assignment pattern:

```cpp
#if defined(__FAST_MATH__)
#error "G1 root-reach kernel must be compiled without fast math"
#endif

#include "g1_ik.h"

bool ik_effective_reach_shell(
    IKReachShell& output,
    vec3 root,
    vec3 middle,
    vec3 end,
    float reach_buffer_m)
{
    if (!ik_vec3_is_runtime_value(root) ||
        !ik_vec3_is_runtime_value(middle) ||
        !ik_vec3_is_runtime_value(end) ||
        !terrain_float_is_positive_normal(reach_buffer_m)) {
        return false;
    }

    double upper = 0.0;
    double lower = 0.0;
    double current = 0.0;
    float upper_f32 = 0.0f;
    float lower_f32 = 0.0f;
    float current_f32 = 0.0f;
    if (!ik_checked_distance_precise(
            upper, upper_f32, middle, root) ||
        !ik_checked_distance_precise(
            lower, lower_f32, end, middle) ||
        !ik_checked_distance_precise(
            current, current_f32, end, root) ||
        upper <= static_cast<double>(reach_buffer_m) ||
        lower <= static_cast<double>(reach_buffer_m)) {
        return false;
    }

    const volatile double nominal_minimum =
        std::fabs(upper - lower) +
        static_cast<double>(reach_buffer_m);
    const volatile double nominal_maximum =
        upper + lower - static_cast<double>(reach_buffer_m);
    if (!terrain_double_is_finite(nominal_minimum) ||
        !terrain_double_is_finite(nominal_maximum) ||
        nominal_minimum <= 0.0 ||
        nominal_maximum < nominal_minimum) {
        return false;
    }

    IKReachShell candidate = {};
    candidate.minimum_distance_m =
        current < nominal_minimum ? current : nominal_minimum;
    candidate.maximum_distance_m =
        current > nominal_maximum ? current : nominal_maximum;
    if (!ik_checked_binary32_commit(
            candidate.minimum_distance_m,
            candidate.minimum_distance_f32_m) ||
        !ik_checked_binary32_commit(
            candidate.maximum_distance_m,
            candidate.maximum_distance_f32_m)) {
        return false;
    }
    output = candidate;
    return true;
}

bool g1_physical_sole_position_target(
    G1PhysicalSolePositionTarget& output,
    vec3 current_contact_origin,
    quat current_contact_rotation,
    vec3 current_ankle_origin,
    const G1LegConfig& config,
    vec3 desired_sole_center,
    vec3 desired_sole_normal,
    char* error,
    int error_capacity)
{
    const std::size_t error_bytes =
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U;
    if (error_capacity < 0 ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &config, sizeof(config)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &output, sizeof(output)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &config, sizeof(config)) ||
        !g1_foot_runtime_config_validate(
            config, error, error_capacity) ||
        !g1_ik_vec3_is_runtime_value(current_contact_origin) ||
        !ik_quat_is_unit(current_contact_rotation) ||
        !g1_ik_vec3_is_runtime_value(current_ankle_origin) ||
        !g1_ik_vec3_is_runtime_value(desired_sole_center) ||
        !g1_ik_surface_normal_is_valid(desired_sole_normal)) {
        return false;
    }

    G1PhysicalSolePositionTarget target = {};
    vec3 sole_offset;
    vec3 contact_to_ankle;
    if (!g1_surface_aligned_foot_rotation(
            target.contact_rotation,
            current_contact_rotation,
            config,
            desired_sole_normal,
            error,
            error_capacity) ||
        !g1_ik_checked_physical_sole_centroid(
            sole_offset,
            vec3(),
            target.contact_rotation,
            config) ||
        !ik_checked_vec3_subtract(
            target.contact_origin,
            desired_sole_center,
            sole_offset) ||
        !ik_checked_vec3_subtract(
            contact_to_ankle,
            current_contact_origin,
            current_ankle_origin) ||
        !ik_checked_vec3_subtract(
            target.ankle_target,
            target.contact_origin,
            contact_to_ankle) ||
        !ik_quat_is_unit(target.contact_rotation) ||
        !g1_ik_vec3_is_runtime_value(target.contact_origin) ||
        !g1_ik_vec3_is_runtime_value(target.ankle_target)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 physical sole target arithmetic failed");
    }
    output = target;
    return true;
}
```

- [ ] **Step 5: Run focused strict, fast-caller/strict-kernel parity, and sanitizers**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_ik_root_reach.cpp \
  -o /tmp/g1-root-reach-plan/task1/root-reach.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-root-reach-plan/task1/clearance.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-root-reach-plan/task1/test-strict.o
g++ /tmp/g1-root-reach-plan/task1/test-strict.o \
  /tmp/g1-root-reach-plan/task1/root-reach.o \
  /tmp/g1-root-reach-plan/task1/clearance.o \
  -o /tmp/g1-root-reach-plan/task1/test-strict
/tmp/g1-root-reach-plan/task1/test-strict
/tmp/g1-root-reach-plan/task1/test-strict --parity \
  > /tmp/g1-root-reach-plan/task1/strict.txt

g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-root-reach-plan/task1/test-fast.o
g++ /tmp/g1-root-reach-plan/task1/test-fast.o \
  /tmp/g1-root-reach-plan/task1/root-reach.o \
  /tmp/g1-root-reach-plan/task1/clearance.o \
  -o /tmp/g1-root-reach-plan/task1/test-fast
/tmp/g1-root-reach-plan/task1/test-fast
/tmp/g1-root-reach-plan/task1/test-fast --parity \
  > /tmp/g1-root-reach-plan/task1/fast.txt
cmp /tmp/g1-root-reach-plan/task1/strict.txt \
  /tmp/g1-root-reach-plan/task1/fast.txt

san='-fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero -fno-sanitize-recover=all'
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer $san \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_ik_root_reach.cpp \
  -o /tmp/g1-root-reach-plan/task1/root-reach-san.o
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer $san \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-root-reach-plan/task1/clearance-san.o
g++ -std=c++17 -O1 -g -fno-omit-frame-pointer $san \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-root-reach-plan/task1/test-san.o
g++ $san /tmp/g1-root-reach-plan/task1/test-san.o \
  /tmp/g1-root-reach-plan/task1/root-reach-san.o \
  /tmp/g1-root-reach-plan/task1/clearance-san.o \
  -o /tmp/g1-root-reach-plan/task1/test-san
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/g1-root-reach-plan/task1/test-san
```

Expected: all executables exit `0`, parity files are byte-identical, and sanitizers are silent.

- [ ] **Step 6: Prove the strict-object guard and commit**

Run:

```bash
if g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
     -c g1_ik_root_reach.cpp \
     -o /tmp/g1-root-reach-plan/task1/forbidden.o \
     2>/tmp/g1-root-reach-plan/task1/forbidden.stderr; then
  echo 'ERROR: root-reach kernel accepted fast math' >&2
  exit 1
fi
rg 'must be compiled without fast math' \
  /tmp/g1-root-reach-plan/task1/forbidden.stderr
git add ik.h g1_ik.h g1_ik_root_reach.cpp tests/cpp/test_g1_ik.cpp
git diff --cached --check
git commit -m "refactor: share strict G1 IK reach geometry"
```

---

### Task 2: Fixed-Size Two-Foot Root-Reach Planner

**Files:**
- Modify: `g1_ik.h`
- Modify: `g1_ik_root_reach.cpp`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Consumes: Task 1 `IKReachShell`, `g1_physical_sole_position_target`, exact 31-bone pose slices, two contact flags, and two `G1FootTarget` values.
- Produces:

```cpp
static constexpr float G1RootReachMaximumAdjustmentM = 0.05f;

struct G1RootReachPlan {
    bool active = false;
    bool common_interval_found = false;
    bool applied = false;
    float root_y_delta_m = 0.0f;
};

bool g1_root_reach_plan_is_valid(const G1RootReachPlan& plan);

bool g1_plan_recorded_contact_root_reach(
    G1RootReachPlan& output,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const slice1d<bool> recorded_contacts,
    const G1FootTarget& left_target,
    const G1FootTarget& right_target,
    char* error,
    int error_capacity);

bool g1_apply_root_reach_plan_y(
    float& output_root_y,
    float baseline_root_y,
    const G1RootReachPlan& plan);

#if defined(G1_IK_ENABLE_TEST_SEAMS)
enum G1RootReachAuditStatus : uint32_t {
    G1RootReachAuditRejected = 1U,
    G1RootReachAuditAccepted = 2U,
};

struct G1RootReachAuditCursor {
    uint32_t interval_index;
    uint32_t initial_delta_bits;
};

struct G1RootReachAuditAttempt {
    uint32_t cursor_index;
    uint32_t delta_bits;
    G1RootReachAuditStatus status;
};

struct G1RootReachPlannerAudit {
    G1RootReachAuditCursor cursors[4];
    G1RootReachAuditAttempt attempts[32];
    uint32_t cursor_count;
    uint32_t attempt_count;
};

bool g1_plan_recorded_contact_root_reach_audited(
    G1RootReachPlan& output,
    G1RootReachPlannerAudit& audit,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const slice1d<bool> recorded_contacts,
    const G1FootTarget& left_target,
    const G1FootTarget& right_target,
    char* error,
    int error_capacity);
#endif
```

- [ ] **Step 1: Write the authentic two-contact RED regression**

Materialize the exact live low-curb/Takara support-retargeted fixture used by the debugger probe. Assert planning is active, a common interval exists, `applied` is true, the delta is in `[-0.0161f, -0.0157f]`, and applying it changes only root Y. Run the named production physical-sole solver for both feet and require `reachable`, no correction limit, and the unchanged exact promoted `<= 0.005f` contact-convergence rule. The current strict kernel selects delta bits `0xbc80e8f0` and yields precise final physical-sole residuals of approximately `0.0020950164 m` left and `0.0001104668 m` right; test the tighter observed envelope `< 0.0022 m`. The older `0.000174 m` / `0.000134 m` debugger values came from a disposable binary predating Task 1 strict projection and remain historical feasibility evidence only.

Also translate the candidate root and both exact world-space locks by level bits `0x38d1b717`. Assert the planner selects `0xbc80e8f5`. Recompute the baseline analytic endpoint and prove `0xbc80e8e7` through `0xbc80e8f4` are outside analytic admission, even though adjusted production-target projection accepts `0xbc80e8e7` through `0xbc80e8f5` and rejects `0xbc80e8e6`. This owns the contract that "nearest" means nearest dual-certified float and that the planner never probes outside its analytic proof domain; the intentionally excluded production-valid gap is only about 26 nm and does not justify changing the named solver.

- [ ] **Step 2: Write interval-set and order RED tests**

Add deterministic fixtures for:

- one recorded left contact plus one swing right foot;
- the same two contacts passed with geometric intervals in reversed lower-bound order;
- an already-reachable contact selecting exact positive zero;
- an established world-space foot lock retained across a root/support level change;
- no contacts producing the canonical inactive plan;
- two disjoint maximum-shell intervals;
- a minimum-shell hole that splits one foot into two intervals;
- a minimum-shell conflict after two-foot intersection;
- nearest dual-certified corrections just inside, exactly at, and one ULP outside both `-0.05f` and `+0.05f`;
- a rounded lower/upper endpoint which becomes valid only after `nextafter` toward the interval interior;
- a candidate which analytic interval math admits but production `ik_project_target` rejects, proving revalidation refuses publication.

Under `G1_IK_ENABLE_TEST_SEAMS`, call the audited wrapper and assert the
production trace, rather than a manually reconstructed approximation:

- the authentic frame materializes initial bits `0xbc80e8ed` and attempts
  `ed`, `ee`, `ef`, `f0` on cursor zero with statuses rejected, rejected,
  rejected, accepted;
- the projection-refusal frame emits exactly 32 rejected attempts and never a
  33rd;
- a real minimum-shell two-cursor frame switches cursors according to the
  global nearest-first ordering before acceptance; and
- a symmetric high-level minimum-shell frame records two cursors and exactly
  32 alternating rejected attempts before the canonical unavailable plan,
  proving the ceiling is planner-wide rather than per interval;
- invalid input/alias/arithmetic leaves poisoned plan and audit byte-exact.

The one-cursor 32-rejection trace catches a 33/128 loop bound, the real
reject-then-other-cursor acceptance trace catches sequential interval search,
and the alternating two-cursor 32-rejection trace catches independent
per-cursor budgets. Check trace capacity before each write so any hypothetical
33rd global attempt returns false transactionally instead of being silently
truncated.

For every `-0.05f` and `+0.05f` inside/exact/outside case, assert all four
plan fields and exact delta bits: inside and exact are active/common/applied
with their expected selected bits, while outside is the canonical
active/non-common/unapplied/positive-zero disposition.

For every order pair, compare all four plan fields by bits. Poison output for invalid pose sizes, topology, target, contact slice, alias, nonfinite arithmetic, and error overlap; require false with unchanged output.

- [ ] **Step 3: Run RED**

Use the Task 1 strict compile/link command. Expected: compilation fails only because `G1RootReachPlan` and planner/apply functions are absent.

- [ ] **Step 4: Implement the fixed-size interval planner**

Use internal value-only objects in `g1_ik_root_reach.cpp`:

```cpp
struct G1RootReachInterval {
    double lower_m;
    double upper_m;
};

struct G1RootReachIntervalSet {
    G1RootReachInterval values[4];
    uint32_t count;
};
```

For each recorded foot:

1. Run checked FK once for the untouched baseline.
2. Derive the exact requested ankle target with Task 1's helper.
3. Derive the unchanged effective reach shell with Task 1's helper.
4. Solve `minimum <= length(target - (hip + (0,delta,0))) <= maximum` as up to two closed Y intervals; intersect fixed arrays without allocation and sort by `(lower, upper)`.
5. Intersect with the exact double cap corresponding to binary32 `[-0.05f,+0.05f]`.
6. Treat the common baseline analytic intersection as a required proof domain. Create one cursor per common interval and examine only analytically admitted candidate floats through a single deterministic global frontier ordered by absolute distance from positive zero, then `(lower, upper, original index)`. Materialize positive zero when contained; otherwise round the nearest endpoint to binary32 and move one ULP toward the interval interior when needed. Test at most 32 adjacent boundary floats total across all cursors, advancing only the rejected cursor; this is one planner-wide budget, not 32 attempts per interval. Never probe outside the analytic intervals even if adjusted recomputation would authenticate a nearer float.
7. For each candidate, copy the fixed 31 local positions, apply the candidate to local `G1_Simulation.y` with `g1_apply_root_reach_plan_y`, and rerun the same checked strict FK. For every recorded foot, rederive the physical ankle target from the adjusted contact origin/rotation and ankle origin plus the original desired sole center/normal, then call `ik_project_target` on the actual adjusted hip/knee/ankle points. Translating the untouched FK's global points or projecting the baseline-derived target is not numerically equivalent to the named production solver and must not authenticate publication.
8. Publish only a candidate for which every projection is reachable and its clamped target bits equal the rederived exact production ankle-target bits.

Keep one private implementation for both public wrappers. The ordinary
wrapper supplies no audit. The test-only audited wrapper supplies a local
fixed-size trace and assigns it only after planner success; it rejects audit
overlap with the plan, every input owner, and the error buffer. Record each
sorted materializable cursor's interval index and first candidate bits, then
record every rejected/accepted global attempt after real adjusted-target
revalidation. Invalid execution assigns neither output. Compile all audit
code and its public declaration only under `G1_IK_ENABLE_TEST_SEAMS`.

If contact planning is active but no candidate survives, return `true` with `{active=true, common_interval_found=false, applied=false, root_y_delta_m=+0}`. Reserve `false` for malformed input/arithmetic/alias failures. `g1_apply_root_reach_plan_y` uses checked double addition and binary32 commit, canonicalizes zero, and assigns output only after validation.

- [ ] **Step 5: Run Task 1's full strict/fast/parity/sanitizer matrix**

Expected: all Task 1 and Task 2 tests pass; strict and fast parity records remain byte-identical; sanitizers emit no report.

Build the focused audited tests with both caller and strict root-reach object
using `G1_IK_ENABLE_TEST_SEAMS`. Separately compile the ordinary test binary
and root-reach object without that macro and run it successfully. Use `nm -C`
on the no-seam root object and require no `audit` or test-seam symbol, then run
the existing positive public/no-seam compile and neutral controller link.

- [ ] **Step 6: Commit the planner**

```bash
git add g1_ik.h g1_ik_root_reach.cpp tests/cpp/test_g1_ik.cpp
git diff --cached --check
git commit -m "feat: plan bounded G1 IK root reach"
```

---

### Task 3: Reversible Runtime Integration and Provenance

**Files:**
- Modify: `g1_ik_runtime.h`
- Modify: `g1_controller_state.h`
- Modify: `g1_frame_transaction.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_ik.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `tests/cpp/test_g1_frame_transaction.cpp`
- Modify: `tests/cpp/test_g1_frame_transaction_production.cpp`
- Modify: `tests/cpp/compile_g1_ik_production.cpp`

**Interfaces:**
- Consumes: Task 2 `G1RootReachPlan`, `g1_plan_recorded_contact_root_reach`, and `g1_apply_root_reach_plan_y`.
- Produces: `G1IkFrameResult::root_reach`, runtime begin integration, equality/digest ownership, rejection snapshot ownership, and accepted diagnostic relation `rendered.hips_y == support_retargeted.hips_y + root_reach.root_y_delta_m`.

- [ ] **Step 1: Write runtime begin RED tests**

Extend authentic `g1_ik_frame_begin` tests to assert:

```cpp
check(transaction.candidate_result.root_reach.active,
      "recorded contacts activate frame-level root reach");
check(transaction.candidate_result.root_reach.applied,
      "authentic extended pose applies bounded root reach");
check(terrain_float_bits(scratch_positions(G1_Simulation).x) ==
          terrain_float_bits(baseline_positions(G1_Simulation).x) &&
      terrain_float_bits(scratch_positions(G1_Simulation).z) ==
          terrain_float_bits(baseline_positions(G1_Simulation).z),
      "preconditioner cannot own planar root motion");
float expected_y = 0.0f;
check(g1_apply_root_reach_plan_y(
          expected_y, baseline_positions(G1_Simulation).y,
          transaction.candidate_result.root_reach) &&
      terrain_float_bits(scratch_positions(G1_Simulation).y) ==
          terrain_float_bits(expected_y),
      "scratch root Y authenticates the retained plan");
```

Add exact byte no-op assertions for IK disabled, no contacts, and already-reachable contacts, including positive/negative-zero inputs. Add a later-foot failure and a finish/clearance failure to prove the caller's accepted state, support, lock/history, command, route, heading, and published root plan all roll back.

Add an infeasible two-contact fixture and an out-of-cap fixture. `g1_ik_frame_begin` must succeed with an active/non-common/canonical-zero plan and unchanged root; the ordinary left-then-right stage must then publish the existing finite `G1IkStopTargetUnreachable` diagnostic. No planner-specific stop reason or target mutation is permitted.

- [ ] **Step 2: Write frame-transaction ownership RED tests**

Add one-bit/one-field mutations of `active`, `common_interval_found`, `applied`, and `root_y_delta_m` to the independent logical digest/equality tests. Authentic accepted and finite-rejected production fixtures must reject every forgery while preserving accepted-state, publication, command, and heading digests. Assert the only pose-position difference before/after accepted IK is `G1_Simulation.y`; all root X/Z and every non-root local position remain bit-equal.

Add accepted diagnostic mutations for `support_retargeted.hips_y`, `rendered.hips_y`, and delta so that each broken sum is rejected. Use `g1_apply_root_reach_plan_y` to form the independent expected binary32 value rather than ordinary `+` in a fast caller.

In `tests/cpp/test_g1_controller_state.cpp`, mutate each plan field on an otherwise authentic accepted state and require `g1_controller_state_is_valid` to reject it. Preserve a canonical inactive plan for disabled IK; for applied IK require `g1_root_reach_plan_is_valid(value.root_reach)` and `value.root_reach.active == (contacts(0) || contacts(1))`.

- [ ] **Step 3: Run RED focused binaries**

Compile/link `test_g1_ik`, `test_g1_frame_transaction`, and `test_g1_frame_transaction_production` against strict `g1_clearance.cpp` and `g1_ik_root_reach.cpp` objects. Expected: failures identify the missing `root_reach` field, missing digest/equality ownership, and missing runtime application.

- [ ] **Step 4: Integrate the planner at frame begin**

Add `G1RootReachPlan root_reach;` immediately before `feet[2]` in `G1IkFrameResult`. In `g1_ik_frame_begin`, keep the existing disabled branch byte-identical. For enabled, non-safe-stopped frames, after both exact `G1FootTarget` values exist and before copying the pose to scratch:

```cpp
G1FootTarget targets[2] = {
    candidate.candidate_result.feet[0].target,
    candidate.candidate_result.feet[1].target,
};
if (!g1_plan_recorded_contact_root_reach(
        candidate.candidate_result.root_reach,
        baseline_positions, baseline_rotations, bone_parents, contacts,
        targets[0], targets[1], error, error_capacity)) {
    return false;
}
```

Copy the untouched baseline arrays as today. If `root_reach.applied`, compute the exact root Y through `g1_apply_root_reach_plan_y` and assign only `scratch_positions(G1_Simulation).y`. Do not change a target or request a new stop reason when no interval exists; the existing recorded-contact stage remains the failure owner.

- [ ] **Step 5: Authenticate every result owner**

Update:

- `g1_ik_runtime_is_disabled_noop` to require the canonical inactive positive-zero plan;
- `g1_ik_frame_rejection_snapshot` to require a valid plan, canonical at begin-time footprint/landing rejection, otherwise active exactly when a recorded contact was eligible for planning;
- accepted finish to retain the same plan unchanged;
- `g1_controller_state_ik_frame_is_valid` to authenticate plan shape and contact activation;
- `g1_frame_ik_result_equal`, rejected-result validators, and canonical-result validators;
- `g1_frame_accepted_diagnostic_matches_success` to authenticate strict checked support/rendered Hips Y plus the retained delta;
- `g1_log_hash_ik_frame_result` in `controller.cpp` and both independent logical hash oracles in the frame transaction tests.

Do not add a CSV field. The existing row builder continues to write support and rendered Hips values from their authoritative owners.

- [ ] **Step 6: Run focused strict, fast-caller, parity, sanitizer, and production closure gates**

Run all Task 2 commands plus equivalent strict/fast/sanitized builds for:

```bash
tests/cpp/test_g1_frame_transaction.cpp
tests/cpp/test_g1_frame_transaction_production.cpp
```

Link each executable with both strict objects. Then run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/compile_g1_ik_production.cpp \
  -o /tmp/g1-root-reach-plan/task3/production.o
! nm -C /tmp/g1-root-reach-plan/task3/production.o | \
  rg 'for_test|test_seam'
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/compile_g1_frame_transaction_runner_negative.cpp \
  -o /tmp/g1-root-reach-plan/task3/runner-negative.o \
  2>/tmp/g1-root-reach-plan/task3/runner-negative.stderr && exit 1 || true
```

Expected: strict/fast/sanitizer tests exit `0`; authentic rejection and acceptance fixtures publish; all plan forgeries fail; no private seam is present in production; the negative runner compile remains rejected.

- [ ] **Step 7: Commit runtime integration**

```bash
git add g1_ik_runtime.h g1_controller_state.h g1_frame_transaction.h \
  controller.cpp tests/cpp/test_g1_ik.cpp \
  tests/cpp/test_g1_controller_state.cpp \
  tests/cpp/test_g1_frame_transaction.cpp \
  tests/cpp/test_g1_frame_transaction_production.cpp \
  tests/cpp/compile_g1_ik_production.cpp
git diff --cached --check
git commit -m "feat: precondition recorded-contact G1 IK reach"
```

---

### Task 4: Live Regression and Complete Certification

**Files:**
- Modify: `.superpowers/sdd/progress.md` (ignored local ledger only)
- Verify only: runtime source, tests, terrain packs, checker, controller process

**Interfaces:**
- Consumes: Tasks 1-3, finished Task 7 logging/provenance changes, separate terrain pack `/tmp/g1-terrain-footprint-runtime-v1`, and the existing Gate E/Gate L checker.
- Produces: a disposable certified controller and evidence sufficient for Task 8 packaging; it does not replace the running visualizer.

- [ ] **Step 1: Build a disposable controller with both strict kernels**

```bash
mkdir -p /tmp/g1-root-reach-cert/build
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -DNDEBUG -I. \
  -c g1_clearance.cpp -o /tmp/g1-root-reach-cert/build/clearance.o
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -DNDEBUG -I. \
  -c g1_ik_root_reach.cpp -o /tmp/g1-root-reach-cert/build/root-reach.o
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src \
  -c controller.cpp -o /tmp/g1-root-reach-cert/build/controller.o
g++ /tmp/g1-root-reach-cert/build/controller.o \
  /tmp/g1-root-reach-cert/build/clearance.o \
  /tmp/g1-root-reach-cert/build/root-reach.o \
  -o /tmp/controller_g1_root_reach_candidate \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
sha256sum /tmp/controller_g1_root_reach_candidate \
  > /tmp/g1-root-reach-cert/controller.sha256
```

Expected: build exits `0`; the final link command has no `-ffast-math`; recorded switches for both strict objects include all three strict flags and exclude fast math.

- [ ] **Step 2: Run the exact low-curb regression at 1, 32, then 800 frames**

Run:

```bash
mkdir -p /tmp/g1-root-reach-cert/low-curb
for frames in 1 32 800; do
  for ik in 0 1; do
    DISPLAY=:1 \
      G1_TERRAIN_DIR=/tmp/g1-terrain-footprint-runtime-v1 \
      MM_TERRAIN_SCENE=grail-curb-low \
      MM_TEST_MODE=route MM_TEST_ROUTE=curb-forward \
      MM_TEST_FRAMES="$frames" MM_TEST_HEADING=forward \
      MM_TERRAIN_WEIGHT=4 MM_IK="$ik" \
      MM_LOG="/tmp/g1-root-reach-cert/low-curb/${frames}-${ik}.csv" \
      /tmp/controller_g1_root_reach_candidate
  done
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -c '
import sys
from resources.check_g1_runtime_log import read_rows, check_rows
for path, ik in ((sys.argv[1], 0), (sys.argv[2], 1)):
    rows = read_rows(path)
    check_rows(rows, allow_ik=True)
    assert len(rows) == int(sys.argv[3])
    assert all(int(row["ik_enabled"]) == ik for row in rows)
    assert all(int(row["frame_rejected"]) == 0 for row in rows)
    assert all(int(row["ik_safe_stop_latched"]) == 0 for row in rows)
    if ik:
        assert all(int(row["ik_applied"]) == 1 for row in rows)
' "/tmp/g1-root-reach-cert/low-curb/${frames}-0.csv" \
  "/tmp/g1-root-reach-cert/low-curb/${frames}-1.csv" "$frames"
done
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-root-reach-cert/low-curb/800-1.csv --gate-l \
  --compare-ik-off /tmp/g1-root-reach-cert/low-curb/800-0.csv \
  --expected-end-x -0.14127154648303986 \
  --expected-end-z 2.825143814086914 \
  --expected-heading forward --require-multilevel
```

Expected at frame 1: accepted, `ik_applied=1`, no safe stop, both recorded feet reachable, and rendered/support Hips delta approximately `-0.016 m`. Expected at 32 and 800: no footprint/IK safe-stop, route advances/completes, exact heading bits, sole residuals and clearances pass, and rendered Hips per-frame step stays at most `0.05 m`.

- [ ] **Step 3: Run inherited focused and full source tests**

Run the complete Python suite and these exact native tests in strict, release caller/strict kernels, and sanitizer configurations: `test_cleanup_runtime`, `test_g1_candidate_audit`, `test_g1_candidate_audit_controller`, `test_g1_clearance`, `test_g1_command_runtime`, `test_g1_controller_logging`, `test_g1_controller_state`, `test_g1_footprint_runtime`, `test_g1_frame_transaction`, `test_g1_frame_transaction_production`, `test_g1_ik`, `test_g1_skeleton`, `test_motion_match_log`, `test_route_runtime`, `test_scene_runtime`, `test_scene_switch`, `test_support_matching`, `test_support_runtime`, `test_terrain_database`, and `test_terrain_runtime`. Run both negative compile fixtures `compile_g1_ik_seam_negative.cpp` and `compile_g1_frame_transaction_runner_negative.cpp`, and the positive production fixture `compile_g1_ik_production.cpp`.

Expected: every invocation exits `0`, strict builds emit no warning, sanitizer output is empty, strict/fast parity records are identical, the database/terrain pack validator remains unchanged, and `git diff --check` is silent.

- [ ] **Step 4: Re-run inherited Gate E terrain behavior**

Generate fresh IK-off/IK-on logs for curb, stair, multilevel, ramp, slope, lateral, diagonal, tangential-boundary, blocked, and landing-exit safe-stop routes using the disposable controller. Run the existing exact Gate E commands from `docs/superpowers/plans/2026-07-15-g1-footprint-aware-directional-terrain.md`.

Expected: normal routes have no finite rejection; stress routes either complete coherently or safe-stop before unsafe publication; heading quality, physical clearance, contact levels, support ownership, and immutable IK-off fields remain within existing thresholds.

- [ ] **Step 5: Re-run the complete 30-cell Gate L matrix**

Run the five scene/route families times six headings defined in the existing Gate L plan, each as a fresh IK-off/IK-on pair, plus Gate L2 lateral pair, Gate L2 exit stress, tangential boundary, and eight flat invariance cases. Do not accept only `--gate-l-safety-only`; every normal cell must pass full `--gate-l`.

Expected: 30/30 full Gate L cells pass, all normal routes complete at exact 25 Hz without a footprint/IK safe-stop, support/query/matcher/command/heading invariants remain exact, Hips step remains at most `0.05 m`, and the exact 305-column schema plus Task 3 corpus audit remain unchanged.

- [ ] **Step 6: Request independent spec and quality reviews**

Give each reviewer the design spec, this implementation plan, Task 1-3 commits, changed-file list, strict/fast/sanitizer evidence, 1/32/800 low-curb logs, and complete Gate E/L summaries. Require separate answers to:

1. Does the implementation satisfy every design requirement without broadening ownership?
2. Can any one-bit plan/result/pose mutation bypass provenance?
3. Can IK-on mutate any IK-off/support/matcher/command/heading field?
4. Are interval boundaries, minimum-shell holes, and projection revalidation numerically sound?
5. Is the runtime ready for Task 8 packaging?

Resolve every Critical or Important finding test-first and rerun the affected focused and complete gates.

- [ ] **Step 7: Record certification without replacing the visualizer**

Update `.superpowers/sdd/progress.md` with exact test counts, Gate E/L totals, binary hash, review verdicts, and the statement that PID `2898959` still runs `/tmp/controller_g1_pre_task6_current`. Leave that process untouched. Task 8 alone copies/packages the certified binary and atomically restarts the visualizer after confirming the new process is alive.
