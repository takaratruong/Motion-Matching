# Carry Seam and Fast-Math Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Smooth every initial and switched Carry candidate before publication, while making Carry's non-finite input rejection reliable in optimized fast-math builds.

**Architecture:** Recorded and layered branches will produce a candidate identified by a mode/recorded-selection-epoch key. A single explicit publication transition will capture the last safe pose, advance cumulative progress over 0.50 seconds, keep the root live, bound all continuous non-root changes, and publish the solved grasp. Carry will separately adopt bit-level float/double finite checks and a dedicated always-active optimized test binary.

**Tech Stack:** C++17, GNU Make, Python `unittest`, existing interaction fixture and graphical evidence validator.

## Global Constraints

- Work only in `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching` on `g1-manipulation-motion-matching`.
- Never access, stat, hash, execute, rebuild, modify, stage, or delete the protected untracked repo-root file `interaction_query_probe` without an extension.
- Preserve the validated full pack and use unique isolated pack/evidence paths for the graphical gate.
- Preserve `--allow-rejections`, the greater-than-0.20 m evidence contract, and layered Carry's live 60 Hz simulation-root policy.
- Same-range contiguous recorded playback must not reset the recorded-selection epoch; explicit retargets, terminal restarts, and range changes must reset it.
- Keep public Carry defaults and IK request bounds unchanged.

---

### Task 1: Pin unconditional layered and recorded seam behavior

**Files:**
- Modify: `tests/cpp/test_interaction_carry.cpp:710-850`
- Test: `tests/cpp/test_interaction_carry.cpp`

**Interfaces:**
- Consumes: `CarryController::start`, `CarryController::update`, `CarryController::recorded`, existing fixture/query helpers.
- Produces: three behavioral regressions for initial layered seam, initial recorded seam, and recorded-range switch/contiguous progression.

- [ ] **Step 1: Replace the IK-rejection-dependent layered seam fixture**

Use default configuration and mutate only the inactive lower-body branch:

```cpp
locomotion.pose = hold;
const quat live_hip = quat_mul(
    hold.rotations[g1_skeleton::LeftHipPitch],
    quat_from_angle_axis(0.90F, vec3(1.0F, 0.0F, 0.0F)));
locomotion.pose.rotations[g1_skeleton::LeftHipPitch] = live_hip;
const Pose first = controller.update(locomotion, 0.04F);
assert(!near(
    first.rotations[g1_skeleton::LeftHipPitch], live_hip, 0.10F));
assert(near(root_world(first), root_world(locomotion.pose), 2.0e-5F));
```

Retain the existing hand/object tolerance, Y-continuity, greater-than-0.20 m motion, and convergence assertions. Require convergence to `live_hip` after enough 0.04 s ticks.

- [ ] **Step 2: Add initial recorded seam regression**

Build one certified range with a constant recorded `LeftHipPitch` rotation distinct from the Hold pose, force its pose/trajectory costs to win, and assert that the first recorded update reports recorded mode but publishes only a bounded fraction:

```cpp
const Pose first = controller.update(locomotion, 0.04F);
assert(controller.recorded());
assert(!near(first.rotations[g1_skeleton::LeftHipPitch], recorded_hip, 0.10F));
assert(near(root_world(first), root_world(locomotion.pose), 2.0e-5F));
assert(hand_error(
    first, Hand::Right, affordance, controller.object_world()) <=
    IKConfig{}.accepted_position_m);
```

Advance ordinary recorded frames and require monotonic transition progress and convergence without a restart.

- [ ] **Step 3: Add recorded-range switch and epoch regression**

Create nonoverlapping certified ranges `[115,132)` and `[132,150)` with distinct constant `LeftHipPitch` rotations. Make all rows in range one tie at zero cost so ordinary frame progression continues without retargeting. Then invert costs at a search boundary so range two wins. Assert the first switched output is between the prior publication and range-two target rather than equal to the target, remains attached, and converges over subsequent small ticks.

- [ ] **Step 4: Run the focused binary and capture RED**

Run:

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
```

Expected: FAIL first on the default layered assertion because current code publishes the complete 0.90 radian `LeftHipPitch` candidate. After isolating each new test if necessary, also capture initial-recorded and recorded-switch failures caused by direct candidate returns.

---

### Task 2: Implement unified publication-stage seam state

**Files:**
- Modify: `interaction_carry.h:45-75`
- Modify: `interaction_carry.cpp:625-750,929-1265`
- Test: `tests/cpp/test_interaction_carry.cpp`

**Interfaces:**
- Consumes: candidate `Pose`, candidate `Transform`, candidate recorded flag, and seam key.
- Produces: internal transition source/progress/key state and one bounded candidate-publication path.

- [ ] **Step 1: Add private seam and recorded-selection state**

Add frozen internal constants in `interaction_carry.cpp` and private fields in `CarryController`:

```cpp
constexpr float kCarrySeamSeconds = 0.50F;
constexpr int64_t kLayeredSeamKey = -1;
constexpr int64_t kUnsetSeamKey = -2;

Pose transition_source_pose_{};
float transition_progress_ = 0.0F;
int64_t recorded_selection_epoch_ = 0;
int64_t published_seam_key_ = -2;
int64_t transition_seam_key_ = -2;
bool transition_active_ = false;
```

Reset all fields in `start`. Extend `CarryUpdateTransaction` so exception rollback restores the epoch, keys, progress, active flag, and transition source.

- [ ] **Step 2: Distinguish contiguous progression from explicit retargets**

Before search assignment, retain the previous range/frame. Increment `recorded_selection_epoch_` only inside `!continue_current` when a new selected frame is assigned. Natural source-frame advancement must not change it. Use positive epoch values as recorded seam keys.

- [ ] **Step 3: Convert recorded and layered returns into candidates**

Use a local candidate with no publication side effects:

```cpp
struct CarryCandidate {
    Pose pose{};
    Transform object{};
    bool recorded = false;
    bool directly_solved = false;
    int64_t seam_key = kLayeredSeamKey;
};
```

Recorded success fills the candidate with its mapped, solved pose/object and current positive epoch. Layered evaluation fills a layered candidate. If raw layered IK rejects, retain the raw non-root target for transition and mark it not directly solved; on later stable updates seed active-arm rotations from `last_safe_pose_` before retrying direct IK.

- [ ] **Step 4: Publish every candidate through explicit progress**

When the key changes or no key has been published, capture `last_safe_pose_`, reset progress, and activate the transition. Compute `requested = min(1, progress + dt / 0.50)`. Blend from the captured source to the current candidate at cumulative requested progress, overwrite all root channels from locomotion, run IK, and require existing object continuity.

If full-channel solve fails, retry with active-arm rotations from `last_safe_pose_`. If that fails, bisect only `[current_progress, requested]` for at most eight attempts. Never lower accepted progress. Publish the solved transition pose/object and update `last_safe_pose_` only after validation.

- [ ] **Step 5: Eliminate the completion boundary snap**

At progress 1, publish through the transition path. Deactivate only when the same-key candidate is directly solvable from the just-published state; otherwise remain at progress 1 and continue the active-arm-corrected path. The following direct publication must be transform-equivalent within existing tolerances.

- [ ] **Step 6: Run focused GREEN and existing regressions**

Run:

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
make build/tests/test_interaction_controller_adapter && \
  build/tests/test_interaction_controller_adapter
```

Expected: both binaries exit 0, including last-safe rejection, live-root, layered mask, initial/switch seam, and no-boundary-snap assertions.

- [ ] **Step 7: Commit item 1**

```sh
git add -- interaction_carry.h interaction_carry.cpp \
  tests/cpp/test_interaction_carry.cpp
git commit -m "fix: smooth every carry candidate transition"
```

---

### Task 3: Harden Carry finite checks under release fast-math

**Files:**
- Create: `tests/cpp/test_interaction_carry_fast_math.cpp`
- Modify: `tests/python/test_interaction_gate1.py:280-325`
- Modify: `Makefile:105-205`
- Modify: `interaction_carry.cpp:1-30,985-1065`

**Interfaces:**
- Produces: `test-interaction-carry-release-fast-math` and `build/tests/test_interaction_carry_release_fast_math`.
- Consumes: `CarryController` public constructor/start/update API and the runtime fixture.

- [ ] **Step 1: Add Make policy RED**

Extend the Python Make policy tests to require a Carry optimized compile line containing `-O3 -DNDEBUG -ffast-math -I.`, all required Carry link sources, one execution of the binary, and inclusion in `test-interaction-safe`.

Run:

```sh
python -m unittest \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests
```

Expected: FAIL because the Carry optimized target is absent.

- [ ] **Step 2: Add the dedicated always-active C++ test and Make target**

The test source must reject an incorrectly configured compilation:

```cpp
#ifndef NDEBUG
#error "Carry fast-math validation requires NDEBUG"
#endif
#ifndef __FAST_MATH__
#error "Carry fast-math validation requires -ffast-math"
#endif
```

Use a throwing `require` helper, not `assert`, to prove that `start` rejects a final Hold pose containing `quiet_NaN()` and `update` rejects `quiet_NaN()` dt. Add the binary dependency/link recipe and phony target to Make, then include it in `test-interaction-safe`.

- [ ] **Step 3: Run optimized C++ RED**

Run:

```sh
make test-interaction-carry-release-fast-math
```

Expected: the always-active test exits nonzero because current `std::isfinite` checks accept at least one NaN under the actual optimized flags.

- [ ] **Step 4: Replace Carry finite checks with IEEE-754 bit checks**

Include `<cstdint>` and `<cstring>`, then use:

```cpp
bool finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite(double value) {
    uint64_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) !=
        0x7ff0000000000000ULL;
}
```

Replace Carry's remaining `std::isfinite` checks for accumulated time and match costs with `finite`.

- [ ] **Step 5: Run normal and optimized GREEN**

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
make test-interaction-carry-release-fast-math
python -m unittest \
  tests.python.test_interaction_gate1.ReleaseFastMathMakefileTests
```

Expected: every command exits 0. The normal suite continues to own inclusive floating-threshold behavior; the optimized binary owns invalid-input safety.

- [ ] **Step 6: Commit item 2**

```sh
git add -- Makefile interaction_carry.cpp \
  tests/cpp/test_interaction_carry_fast_math.cpp \
  tests/python/test_interaction_gate1.py
git commit -m "fix: harden carry validation under fast math"
```

---

### Task 4: Run final safe and graphical gates, then amend the report

**Files:**
- Modify: `.superpowers/sdd/playable-final-gate-fix-report.md`

**Interfaces:**
- Consumes: all prior commits and the existing exact gate.
- Produces: fresh artifact/evidence measurements and final commit hashes.

- [ ] **Step 1: Run static and focused verification**

```sh
git diff --check
make test-interaction-safe
```

Expected: exit 0; safe suite executes both optimized validation binaries.

- [ ] **Step 2: Run a unique exact graphical gate**

```sh
DISPLAY=:1 \
INTERACTION_DEMO_PACK=build/task12/final-demo-pack-carry-seam-fastmath-20260715 \
PLAYABLE_EVIDENCE_DIR=playable-evidence/final-carry-seam-fastmath-20260715 \
make gate-playable-interaction
```

Expected: exit 0, real graphical evidence valid, more than 0.20 m root/object displacement, bounded Hold-to-Carry seam, screenshot larger than 10,000 bytes, reset observed.

- [ ] **Step 3: Verify artifact and evidence measurements**

Read only the exact fresh manifest/report/evidence paths. Record source/included/rejected counts, rejection histogram, hashes, probes, ordered states, Carry/forward counts, Hold-to-Carry deltas, Carry root/object displacement, object Y range, screenshot bytes, and validator output.

- [ ] **Step 4: Commit correction implementation**

If Task 2 and Task 3 were kept as local commits, retain their exact hashes. Otherwise make one scoped correction commit after `git diff --check` and all verification succeeds.

- [ ] **Step 5: Amend and commit the durable report**

Update the ignored-but-versioned report with review RED/GREEN evidence, design/implementation hashes, optimized target output, exact fresh gate output, measurements, and remaining concerns. Force-add only the exact report path and commit it separately so it can reference the implementation hashes.

- [ ] **Step 6: Final tracked-clean verification**

```sh
git status --short --untracked-files=no
git log -5 --format='%H %s'
```

Expected: tracked status is empty and all design, implementation, optimized-validation, and report hashes are present.
