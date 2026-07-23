# Bilateral Posture-Aware Reach IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace arm-only exhaustive reach shaping with the proven posture-aware ten-joint solver for both left and right reaches.

**Architecture:** Port the reviewed left-side solver and tests from the isolated IK comparison spike, then generalize its metadata and kinematic objectives around an explicit `Hand` parameter. Integrate it only into `reach_coverage.cpp`, applying temporally seeded correction during the final 0.6 seconds before the swept object/environment collision gate.

**Tech Stack:** C++17, G1 local/world pose FK, numerical Jacobian damped least squares, Make, Python `unittest`.

## Global Constraints

- Optimize exactly three waist plus seven selected-arm G1 hinge joints.
- Support left and right directly; do not reflect right poses through left space.
- Preserve root, legs, inactive arm, and all non-owned pose channels.
- Preserve adaptive damping, numerical Jacobian, hard limits, `0.10 rad` step cap, and 30-iteration cap.
- Apply smoothstep correction only in the final `0.6 s` at pack FPS.
- Use no old arm-only IK fallback for accepted options.
- Preserve the final 1 mm grasp gate and all swept object/furniture collision gates.
- Process all 4,608 candidate identities within the existing 30-second fixture deadline.

---

## File Structure

- Create `interaction_posture_ik.h/.cpp`: bilateral posture-aware solver.
- Create `tests/cpp/g1_posture_ik_fixture.h`: symmetric G1 test poses and ownership assertions.
- Create `tests/cpp/test_interaction_posture_ik.cpp`: solver unit tests for both hands.
- Modify `reach_coverage.cpp`: final-0.6-second temporally seeded posture shaping.
- Modify `tests/cpp/test_reach_coverage.cpp`: timing, side, ownership, and final-error integration regressions.
- Modify `Makefile`: solver test and all affected viewer/probe/test link dependencies.
- Modify `.superpowers/sdd/progress.md`: combined IK/collision evidence.

### Task 1: Port and Generalize the Posture Solver

**Files:**
- Create: `interaction_posture_ik.h`
- Create: `interaction_posture_ik.cpp`
- Create: `tests/cpp/g1_posture_ik_fixture.h`
- Create: `tests/cpp/test_interaction_posture_ik.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `Pose`, `WorldPose`, `Transform`, `Hand`, `HingeJoint`, `kWaist`, `kLeftArm`, and `kRightArm`.
- Produces: `UpperBodyAngles`, `PostureIKConfig`, `PostureIKResult`, `decompose_upper_body`, `apply_upper_body`, `hand_elbow_pole`, and `solve_hand_posture_ik`.

- [ ] **Step 1: Add the reviewed left-solver tests before implementation**

Bring the latest committed test fixture and solver tests from:

```text
/home/ubuntu/projects/motion-matching/.worktrees/
g1-reach-ik-comparison-spike-20260723/tests/cpp/
```

Retain tests for byte-exact zero offset, reachable convergence, finite bounded
unreachable output, best-seen source pose, elbow-pole geometry, and no elbow
flip. Add the Make target:

```make
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_interaction_posture_ik

$(CPP_TEST_DIR)/test_interaction_posture_ik: \
  tests/cpp/test_interaction_posture_ik.cpp \
  tests/cpp/g1_posture_ik_fixture.h \
  interaction_posture_ik.cpp interaction_posture_ik.h \
  interaction_pose.cpp interaction_pose.h interaction_ik.h \
  g1_arm_joint_metadata.h g1_skeleton.h vec.h quat.h | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) tests/cpp/test_interaction_posture_ik.cpp \
	  interaction_posture_ik.cpp interaction_pose.cpp -o $@
```

- [ ] **Step 2: Run the test to verify RED**

```bash
make build/tests/test_interaction_posture_ik
```

Expected: FAIL because `interaction_posture_ik.h/.cpp` do not exist.

- [ ] **Step 3: Port the latest reviewed solver**

Bring `interaction_posture_ik.h/.cpp` from spike commit `dde3233`, including
the later truthful-diagnostics fixes through `fb81cca`. Run:

```bash
make build/tests/test_interaction_posture_ik
build/tests/test_interaction_posture_ik
```

Expected: the original left-side tests PASS.

- [ ] **Step 4: Write failing bilateral API tests**

Change the test-facing contract to:

```cpp
inline constexpr size_t kUpperBodyJointCount = 10U;
using UpperBodyAngles = std::array<float, kUpperBodyJointCount>;

UpperBodyAngles decompose_upper_body(const Pose&, Hand);
void apply_upper_body(Pose&, Hand, const UpperBodyAngles&);
ElbowPole hand_elbow_pole(const Pose&, Hand);
PostureIKResult solve_hand_posture_ik(
    Pose&, Hand, Transform target, const Pose& source,
    const UpperBodyAngles& temporal_seed,
    const PostureIKConfig& = PostureIKConfig{});
```

Parameterize every solver test over `{Hand::Left, Hand::Right}`. Require the
owned set to equal `kWaist + kLeftArm` or `kWaist + kRightArm`, and require the
inactive arm's rotations to remain byte-identical.

- [ ] **Step 5: Run the bilateral tests to verify RED**

```bash
make build/tests/test_interaction_posture_ik
```

Expected: compilation FAIL because the imported solver exposes left-only
symbols.

- [ ] **Step 6: Generalize the solver directly by hand**

Thread `Hand hand` through metadata selection, hand FK, elbow-pole FK,
decompose/apply, evaluation, Jacobian, bounds, result construction, and solve.
Select metadata with:

```cpp
const HingeJoint& metadata(Hand hand, size_t joint) {
    if (joint < kWaist.size()) return kWaist[joint];
    const size_t arm_joint = joint - kWaist.size();
    return hand == Hand::Left
        ? kLeftArm[arm_joint]
        : kRightArm[arm_joint];
}
```

Select shoulder/elbow/wrist bones by `hand`; do not negate angles or
coordinates. Preserve the spike's residual dimensions, weights, damping,
limits, best-pose logic, and diagnostics.

- [ ] **Step 7: Run bilateral tests to verify GREEN**

```bash
make build/tests/test_interaction_posture_ik
build/tests/test_interaction_posture_ik
```

Expected: PASS with no output.

- [ ] **Step 8: Commit**

```bash
git add interaction_posture_ik.h interaction_posture_ik.cpp \
  tests/cpp/g1_posture_ik_fixture.h \
  tests/cpp/test_interaction_posture_ik.cpp Makefile
git commit -m "feat: add bilateral posture-aware G1 IK"
```

### Task 2: Integrate Final-Window Posture Shaping

**Files:**
- Modify: `reach_coverage.cpp`
- Modify: `tests/cpp/test_reach_coverage.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `solve_hand_posture_ik`, pack FPS, placed source poses, existing per-sample grasp targets, and `CoverageConfig`.
- Produces: posture-shaped `Evaluation::poses` for collision validation and rendering.

- [ ] **Step 1: Write failing final-window integration tests**

Add reach-coverage tests proving:

```text
correction_seconds == 0.6
correction sample count == ceil(0.6 * fps)
all earlier poses equal placed source poses
left candidates call left posture ownership
right candidates call right posture ownership
root, legs, and inactive arm remain unchanged
final position error remains <= 0.001 m when accepted
```

Add a source contract requiring `solve_hand_posture_ik(` and forbidding
`solve_hand_ik(` inside `shape_candidate`.

- [ ] **Step 2: Run integration tests to verify RED**

```bash
make build/tests/test_reach_coverage
build/tests/test_reach_coverage
```

Expected: FAIL because `shape_candidate` still invokes arm-only IK throughout
the clip.

- [ ] **Step 3: Add exact correction timing**

Derive:

```cpp
const float fps =
    static_cast<float>(pack.database.fps_numerator) /
    static_cast<float>(pack.database.fps_denominator);
const size_t correction_intervals = static_cast<size_t>(
    std::ceil(0.6F * fps));
const size_t correction_start = frame_count - 1U >
        correction_intervals
    ? frame_count - 1U - correction_intervals
    : 0U;
```

Return weight zero through `correction_start`, smoothstep to one at the final
sample, and never invoke posture IK at weight zero.

- [ ] **Step 4: Replace arm-only solving with temporal posture solving**

For each corrected sample, use the current placed pose as `source_pose`.
Compute `source_angles = decompose_upper_body(source_pose, hand)`. Seed the
first corrected sample from `source_angles`; thereafter add wrapped
`previous_solution - previous_source` to each current source angle.

Call:

```cpp
solve_hand_posture_ik(
    pose, hand, desired, source_pose, temporal_seed, posture_config);
```

Configure final acceptance from `CoverageConfig`, retain the spike's posture
weights, and accumulate active-arm deformation over the selected arm plus
waist without moving root or legs.

- [ ] **Step 5: Link every affected target**

Add `interaction_posture_ik.cpp/.h` to the viewer, probe,
`test_reach_coverage`, and any search target that links `reach_coverage.cpp`.
Build all affected targets to catch missing dependencies.

- [ ] **Step 6: Run integration and solver tests to verify GREEN**

```bash
make build/tests/test_interaction_posture_ik \
  build/tests/test_reach_coverage build/tests/test_reach_search \
  g1_reach_coverage_probe g1_reach_coverage_viewer
build/tests/test_interaction_posture_ik
build/tests/test_reach_coverage
build/tests/test_reach_search
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add reach_coverage.cpp tests/cpp/test_reach_coverage.cpp Makefile
git commit -m "feat: shape reaches with bilateral posture IK"
```

### Task 3: Combined Collision/IK Evidence and Handoff

**Files:**
- Modify: `.superpowers/sdd/progress.md`
- Generate: `build/g1-reaches/contact-anchored-coverage-report-v4.json` (do not commit)

**Interfaces:**
- Consumes: completed swept-collision plan and Tasks 1-2.
- Produces: reviewed real-pack evidence and exactly one refreshed mesh viewer.

- [ ] **Step 1: Run all focused and regression gates**

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_interaction_posture_ik \
  build/tests/test_interaction_hand_trajectories \
  build/tests/test_reach_coverage build/tests/test_reach_search \
  build/tests/test_interaction_ik g1_reach_coverage_probe \
  g1_reach_coverage_viewer
build/tests/test_interaction_posture_ik
build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_reach_search
build/tests/test_interaction_ik
git diff --check
```

Expected: all PASS. The old `interaction_ik` implementation remains available
to unrelated paths but is no longer called by exhaustive reach shaping.

- [ ] **Step 2: Run real exhaustive evidence**

```bash
./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 \
  --json build/g1-reaches/contact-anchored-coverage-report-v4.json
jq '{search_integrity_passed, fixtures: (.shared_grasps | \
  map_values({accepted, processed_instances, rejections, elapsed_seconds}))}' \
  build/g1-reaches/contact-anchored-coverage-report-v4.json
```

Expected: every fixture processes 4,608/4,608 within 30 seconds. Record, but do
not conceal, any coverage reduction caused by stricter posture quality and
swept collision.

- [ ] **Step 3: Request focused review**

Review from collision/IK design baseline through `HEAD` for bilateral metadata,
unowned-channel preservation, temporal seeding, exact 0.6-second timing,
endpoint-only object contact, collision ordering, finite failure behavior, and
runtime. Resolve every Critical and Important finding test-first.

- [ ] **Step 4: Record and commit evidence**

Append exact commits, tests, report counts/timings, and review verdict to
`.superpowers/sdd/progress.md`, then:

```bash
git add -f .superpowers/sdd/progress.md
git commit -m "docs: record posture IK and swept collision evidence"
```

- [ ] **Step 5: Replace only the exact viewer**

Terminate only the PID matching:

```bash
pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'
```

Launch exactly one:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer \
  build/g1-reaches/reach-pack-v2
```

Verify one PID and bounded RSS. Test with `G`, then `Enter`; accepted mesh
motions must use posture IK and must not sweep the grasp through the object.
