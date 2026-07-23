# Posture-IK Coverage Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore at least 94 strict-collision open-space reach options while retaining bilateral posture-aware motion quality.

**Architecture:** Keep the current ten-joint posture solver as the quality pass, then invoke the same solver with regularizers removed only when the wrist task remains infeasible; a task-feasible polish always outranks a prettier miss. Shape the complete trajectory with a 0.6-second correction window whose final five frames are fully aligned, then enforce the measured pre-regression real-pack coverage in the probe.

**Tech Stack:** C++17, numerical damped-least-squares IK, existing G1 kinematics, Make, C++ `assert` tests, Python `unittest`, raylib viewer.

## Global Constraints

- Accepted wrist-position error remains at most `0.001F` metres.
- Accepted approach-axis error remains at most `0.261799388F` radians.
- Accepted grasp-orientation error remains at most the query's existing configured gate.
- Root translation and root height may not be changed by IK.
- Only three waist and seven active-arm joints may be changed.
- Hard G1 joint limits remain enforced.
- No legacy seven-joint fallback and no acceptance-threshold relaxation.
- Every complete trajectory still passes strict swept object and environment collision checks.
- Final contact remains the only object-contact exemption.
- Open-space verification processes all 4,608 instances in at most 30 seconds.

---

### Task 1: Add a task-priority posture-IK polish

**Files:**
- Modify: `interaction_posture_ik.h:21-75`
- Modify: `interaction_posture_ik.cpp:341-478`
- Test: `tests/cpp/test_interaction_posture_ik.cpp`

**Interfaces:**
- Consumes: existing `solve_hand_posture_ik(Pose&, Hand, Transform, const Pose&, const UpperBodyAngles&, const PostureIKConfig&)`.
- Produces: `solve_hand_posture_ik_task_priority(Pose&, Hand, Transform, const Pose&, const UpperBodyAngles&, const PostureIKConfig&) -> PostureIKResult`.
- Produces: left-hand compatibility wrapper `solve_left_hand_posture_ik_task_priority(...)`.

- [ ] **Step 1: Write failing bilateral task-priority tests**

Add a helper that constructs a reachable target from a known bounded
upper-body pose, then configures source and temporal scales large enough for
the quality solve to prefer a wrist miss:

```cpp
void assert_task_priority_reaches(
    interaction::Hand hand,
    const interaction::UpperBodyAngles& delta) {
    const interaction::Pose source = make_pose();
    interaction::Pose goal = source;
    auto goal_angles = interaction::decompose_upper_body(goal, hand);
    for (size_t joint = 0U; joint < goal_angles.size(); ++joint) {
        goal_angles[joint] += delta[joint];
    }
    interaction::apply_upper_body(goal, hand, goal_angles);

    interaction::PostureIKConfig config{};
    config.source_scale_m_per_radian.fill(0.30F);
    config.temporal_scale_m_per_radian.fill(0.30F);
    interaction::Pose quality = source;
    const interaction::UpperBodyAngles seed =
        interaction::decompose_upper_body(source, hand);
    const auto quality_result = interaction::solve_hand_posture_ik(
        quality, hand, hand_transform(goal, hand), source, seed, config);
    assert(!quality_result.accepted);

    interaction::Pose solved = source;
    const auto result = interaction::solve_hand_posture_ik_task_priority(
        solved, hand, hand_transform(goal, hand), source, seed, config);
    assert(result.accepted);
    assert(result.position_error_m <= config.accepted_position_m);
    assert(result.orientation_error_radians <=
           config.accepted_orientation_radians);
    assert_finite_bounded(result.joint_angles, hand);
    assert_non_owned_local_channels_equal(source, solved, hand);
    assert(exact(
        source.positions[g1_skeleton::Simulation],
        solved.positions[g1_skeleton::Simulation]));
}
```

Call it for left and right with bounded, nontrivial deltas:

```cpp
assert_task_priority_reaches(
    interaction::Hand::Left,
    {0.06F, -0.04F, 0.03F, -0.10F, 0.08F,
     0.05F, 0.12F, -0.06F, 0.05F, -0.04F});
assert_task_priority_reaches(
    interaction::Hand::Right,
    {0.05F, -0.03F, 0.02F, -0.08F, -0.06F,
     0.04F, 0.10F, 0.05F, -0.04F, 0.03F});
```

Add an unreachable-target test asserting a finite bounded result, unchanged
root, and `accepted == false`.

- [ ] **Step 2: Run the posture tests to verify RED**

Run:

```bash
make build/tests/test_interaction_posture_ik
```

Expected: compilation fails because
`solve_hand_posture_ik_task_priority` is undeclared.

- [ ] **Step 3: Declare the bilateral and compatibility APIs**

Add to `interaction_posture_ik.h`:

```cpp
PostureIKResult solve_hand_posture_ik_task_priority(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const Pose& source_pose,
    const UpperBodyAngles& temporal_seed,
    const PostureIKConfig& config = PostureIKConfig{});

PostureIKResult solve_left_hand_posture_ik_task_priority(
    Pose& pose,
    Transform target_hand_world,
    const Pose& source_pose,
    const LeftUpperBodyAngles& temporal_seed,
    const PostureIKConfig& config = PostureIKConfig{});
```

- [ ] **Step 4: Implement quality-first, feasibility-second selection**

Implement the bilateral function in `interaction_posture_ik.cpp`. Preserve
the quality pose separately, run the polish from the quality joint angles,
and choose a task-feasible result before comparing weighted objectives:

```cpp
PostureIKResult solve_hand_posture_ik_task_priority(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const Pose& source_pose,
    const UpperBodyAngles& temporal_seed,
    const PostureIKConfig& config) {
    Pose quality_pose = source_pose;
    PostureIKResult quality = solve_hand_posture_ik(
        quality_pose, hand, target_hand_world, source_pose,
        temporal_seed, config);
    if (quality.accepted) {
        pose = quality_pose;
        return quality;
    }

    PostureIKConfig polish_config = config;
    polish_config.source_scale_m_per_radian.fill(0.0F);
    polish_config.temporal_scale_m_per_radian.fill(0.0F);
    polish_config.elbow_pole_scale_m = 0.0F;
    polish_config.maximum_iterations =
        std::max(polish_config.maximum_iterations, 45);

    Pose polish_pose = quality_pose;
    PostureIKResult polish = solve_hand_posture_ik(
        polish_pose, hand, target_hand_world, source_pose,
        quality.joint_angles, polish_config);
    if (polish.accepted ||
        (!quality.accepted &&
         std::tie(
             polish.position_error_m,
             polish.orientation_error_radians,
             polish.objective) <
         std::tie(
             quality.position_error_m,
             quality.orientation_error_radians,
             quality.objective))) {
        pose = polish_pose;
        return polish;
    }
    pose = quality_pose;
    return quality;
}
```

If the exact test fixture shows the polish needs more than 45 iterations,
raise only the polish cap while keeping the normal quality pass at 30.
Do not modify tolerances or joint limits.

Implement the left wrapper as a direct call with `Hand::Left`.

- [ ] **Step 5: Run the posture tests to verify GREEN**

Run:

```bash
make build/tests/test_interaction_posture_ik
build/tests/test_interaction_posture_ik
```

Expected: build succeeds and the test exits 0.

- [ ] **Step 6: Commit the task-priority solver**

```bash
git add interaction_posture_ik.h interaction_posture_ik.cpp \
  tests/cpp/test_interaction_posture_ik.cpp
git commit -m "fix: prioritize feasible posture IK grasps"
```

---

### Task 2: Restore a locked final approach

**Files:**
- Modify: `reach_coverage.cpp:14-15,264-440`
- Test: `tests/cpp/test_reach_coverage.cpp`

**Interfaces:**
- Consumes: `solve_hand_posture_ik_task_priority(...)` from Task 1.
- Produces: unchanged public `shape_candidate(...)`; its final five frame
  intervals use fully applied contact translation, approach alignment, and
  wrist-orientation correction.

- [ ] **Step 1: Write a failing final-approach regression**

Add a 25-frame fixture test that rotates the requested approach by 90
degrees and verifies the final five-frame displacement rather than only the
terminal frame:

```cpp
void test_final_five_frames_lock_to_requested_approach() {
    const reach::Pack pack = fixture(25U);
    reach::Query query = zero_query(pack);
    query.approach_world = normalize(quat_mul_vec3(
        quat_from_angle_axis(
            1.570796327F, vec3(0.0F, 1.0F, 0.0F)),
        query.approach_world));
    const reach::Candidate candidate =
        reach::select_candidates(pack, query).front();

    const reach::Evaluation result =
        reach::shape_candidate(pack, candidate, query);

    assert(!result.poses.empty());
    const size_t final_sample = result.poses.size() - 1U;
    const interaction::WorldPose start =
        interaction::world_pose(result.poses[final_sample - 5U]);
    const interaction::WorldPose finish =
        interaction::world_pose(result.poses[final_sample]);
    const vec3 measured = normalize(
        finish.positions[g1_skeleton::LeftWrist] -
        start.positions[g1_skeleton::LeftWrist]);
    assert(std::acos(std::clamp(
        dot(measured, normalize(query.approach_world)),
        -1.0F, 1.0F)) <= 0.261799388F);
}
```

Extend `test_final_contact_uses_posture_window_and_one_millimetre_gate` to
assert all local simulation-root positions and rotations remain exactly
equal to `place_pose(...)` for every sample.

- [ ] **Step 2: Run coverage tests to verify RED**

Run:

```bash
make build/tests/test_reach_coverage
build/tests/test_reach_coverage
```

Expected: the new 90-degree final-approach assertion fails with the current
contact-only completion schedule.

- [ ] **Step 3: Finish correction five frames before contact**

In `reach_coverage.cpp`, add:

```cpp
constexpr size_t kLockedApproachIntervals = 5U;
```

Keep `correction_start` at the beginning of the existing 0.6-second window,
then compute the full-alignment sample:

```cpp
const size_t aligned_sample = available_intervals >
        kLockedApproachIntervals
    ? available_intervals - kLockedApproachIntervals
    : available_intervals;
const size_t correction_start = available_intervals >
        correction_intervals
    ? available_intervals - correction_intervals
    : 0U;
if (requires_correction && aligned_sample <= correction_start) {
    evaluation.rejection = Rejection::PositionError;
    return evaluation;
}
```

Replace the current weight with a clamped blend that stays at one through
the locked region:

```cpp
const float correction_u = sample <= correction_start
    ? 0.0F
    : (sample >= aligned_sample
        ? 1.0F
        : static_cast<float>(sample - correction_start) /
              static_cast<float>(aligned_sample - correction_start));
const float correction_weight = smoothstep(correction_u);
```

Use this one weight for contact translation, approach alignment, and wrist
orientation exactly as the current code does.

- [ ] **Step 4: Route shaping through task-priority posture IK**

Replace the call inside `shape_candidate`:

```cpp
ik = interaction::solve_hand_posture_ik_task_priority(
    pose,
    hand,
    desired,
    placed_pose,
    temporal_seed,
    posture_config);
```

Keep source-relative temporal seed transport and deformation accounting
unchanged. Never write the simulation root from the IK result.

- [ ] **Step 5: Run focused coverage, posture, and collision tests**

Run:

```bash
make build/tests/test_reach_coverage \
  build/tests/test_interaction_posture_ik \
  build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_interaction_posture_ik
build/tests/test_interaction_hand_trajectories
```

Expected: all three tests exit 0.

- [ ] **Step 6: Commit the locked approach**

```bash
git add reach_coverage.cpp tests/cpp/test_reach_coverage.cpp
git commit -m "fix: lock shaped reaches before contact"
```

---

### Task 3: Turn the measured baseline into a regression gate

**Files:**
- Modify: `g1_reach_coverage_probe.cpp:21-29,254-271`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: the unchanged JSON fixture report.
- Produces: `valid_report(...) == true` only when open-space coverage reaches
  94 total, 47 per hand, and five azimuth sectors.

- [ ] **Step 1: Write failing source-contract tests**

Add assertions to `tests/python/test_g1_reach_coverage_viewer.py`:

```python
self.assertIn("constexpr size_t kMinimumOpenAccepted = 94U;", probe)
self.assertIn(
    "constexpr size_t kMinimumOpenAcceptedPerHand = 47U;", probe
)
self.assertIn(
    "constexpr size_t kMinimumOpenAzimuthSectors = 5U;", probe
)
self.assertIn(
    "report.accepted >= kMinimumOpenAccepted", probe
)
self.assertIn(
    "report.hands[0] >= kMinimumOpenAcceptedPerHand", probe
)
self.assertIn(
    "report.hands[1] >= kMinimumOpenAcceptedPerHand", probe
)
self.assertIn(
    "report.root_azimuth_sectors >= kMinimumOpenAzimuthSectors", probe
)
```

- [ ] **Step 2: Run the Python test to verify RED**

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests -v
```

Expected: failure because the constants are absent.

- [ ] **Step 3: Add exact pre-regression thresholds**

Add beside the existing probe constants:

```cpp
constexpr size_t kMinimumOpenAccepted = 94U;
constexpr size_t kMinimumOpenAcceptedPerHand = 47U;
constexpr size_t kMinimumOpenAzimuthSectors = 5U;
```

Replace the permissive open-space gate with:

```cpp
if (fixtures[index].name == "open_space") {
    valid = valid &&
        report.accepted >= kMinimumOpenAccepted &&
        report.hands[0] >= kMinimumOpenAcceptedPerHand &&
        report.hands[1] >= kMinimumOpenAcceptedPerHand &&
        report.root_azimuth_sectors >= kMinimumOpenAzimuthSectors;
}
```

- [ ] **Step 4: Run the Python test to verify GREEN**

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests -v
```

Expected: all viewer/probe contract tests pass.

- [ ] **Step 5: Commit the real-pack gate**

```bash
git add g1_reach_coverage_probe.cpp \
  tests/python/test_g1_reach_coverage_viewer.py
git commit -m "test: gate open-space reach coverage"
```

---

### Task 4: Verify the real pack and relaunch one flat viewer

**Files:**
- Generate, do not commit: `build/g1-reaches/contact-anchored-coverage-report-v5.json`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: `build/g1-reaches/reach-pack-v2`.
- Produces: benchmark evidence and exactly one current flat-terrain viewer.

- [ ] **Step 1: Run the complete focused verification**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_interaction_posture_ik \
  build/tests/test_interaction_hand_trajectories \
  build/tests/test_reach_coverage \
  build/tests/test_reach_search \
  build/tests/test_reach_database \
  build/tests/test_interaction_ik \
  g1_reach_coverage_probe \
  g1_reach_coverage_viewer
build/tests/test_interaction_posture_ik
build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_reach_search
build/tests/test_reach_database
build/tests/test_interaction_ik
```

Expected: every command exits 0.

- [ ] **Step 2: Run the strict real-pack gate**

Run:

```bash
./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 \
  --json build/g1-reaches/contact-anchored-coverage-report-v5.json
jq '.search_integrity_passed, .shared_grasps.open_space' \
  build/g1-reaches/contact-anchored-coverage-report-v5.json
```

Expected:

```text
true
```

and open-space values of at least 94 accepted, at least 47 for each hand,
at least five azimuth sectors, at most 0.001 m accepted position error, all
4,608 candidates processed, and elapsed time at most 30 seconds.

- [ ] **Step 3: Record evidence**

Append one dated entry to `.superpowers/sdd/progress.md` containing the
commit hashes, test commands, exact open-space counts, maximum errors,
sectors, elapsed time, and the fact that root-height and collision tests
passed.

- [ ] **Step 4: Commit evidence**

```bash
git add .superpowers/sdd/progress.md
git commit -m "docs: record restored posture IK coverage"
```

- [ ] **Step 5: Replace the existing viewer with the verified binary**

Resolve the exact current viewer PID:

```bash
pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'
```

Terminate only that explicit PID, wait for it to exit, then launch:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer \
  build/g1-reaches/reach-pack-v2
```

Expected: exactly one viewer process, flat grid/coverage furniture only,
responsive camera and object controls, Enter recomputation, and the
recovered open-space options visible after pressing `G` and Enter.

- [ ] **Step 6: Verify repository and process state**

Run:

```bash
git status --short
pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'
ps -o pid,rss,etime,cmd -p \
  "$(pgrep -n -f '(^|/)g1_reach_coverage_viewer( |$)')"
```

Expected: only known generated binaries/reports remain untracked, exactly
one viewer is running, and RSS stays bounded near the existing approximately
145 MiB level.
