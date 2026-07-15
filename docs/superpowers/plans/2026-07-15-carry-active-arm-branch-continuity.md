# Carry Active-Arm Branch Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep every published Carry active-arm joint within the inclusive existing IK-derived per-tick budget, including initial seam, progress 1, completion, and direct handoff.

**Architecture:** A deterministic recorded-range regression supplies a redundant in-limit right-arm IK branch with the same grasp transform. Carry canonicalizes each directly solved candidate from the last published arm branch under the existing strict grasp tolerances, then applies the same active-arm distance gate to every transition solution before any state publication.

**Tech Stack:** C++17, GNU Make, existing Carry/runtime fixture, existing numerical hand IK, Python `unittest` safe gate.

## Global Constraints

- Work only in `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching` on `g1-manipulation-motion-matching`.
- Never access, stat, hash, execute, rebuild, modify, stage, or delete the protected untracked repo-root file `interaction_query_probe` without an extension.
- Keep this correction limited to Carry active-arm branch continuity; Head/Neck adapter fallback remains separate.
- Preserve all public Carry/IK defaults and every existing IK request, accepted-error, Carry drift, root, evidence, recorded-epoch, and rollback contract.
- The active-arm publication budget is inclusive `min(pi, 2 * IKConfig.maximum_step_radians)`, which is 0.20 radians under defaults.
- Use the real recorded Carry production path; do not add test-only production hooks or mutable controller state.

---

### Task 1: Reproduce the redundant-branch publication snap

**Files:**
- Modify: `tests/cpp/test_interaction_carry.cpp`
- Test: `build/tests/test_interaction_carry`

**Interfaces:**
- Consumes: `CarryController::start`, `CarryController::update`, `classify_carry_ranges`, `pose_at_frame`, fixture database writers, and default `IKConfig`.
- Produces: `test_recorded_carry_keeps_active_arm_on_one_continuous_ik_branch`, a production-path regression spanning 20 ordinary Carry ticks.

- [ ] **Step 1: Add test-only measurement and object writers**

Add shortest-arc local arm measurement next to the existing `near` helpers:

```cpp
float rotation_distance(quat left, quat right) {
    left = left / quat_length(left);
    right = right / quat_length(right);
    const float cosine = std::clamp(
        std::abs(quat_dot(left, right)), 0.0F, 1.0F);
    return 2.0F * std::acos(cosine);
}

float maximum_right_arm_step(
    const interaction::Pose& previous,
    const interaction::Pose& current) {
    float maximum = 0.0F;
    for (int32_t bone = g1_skeleton::RightShoulderPitch;
         bone <= g1_skeleton::RightWrist;
         ++bone) {
        maximum = std::max(
            maximum,
            rotation_distance(
                previous.rotations[static_cast<size_t>(bone)],
                current.rotations[static_cast<size_t>(bone)]));
    }
    return maximum;
}
```

Add a complete object writer beside `write_object_rotation`:

```cpp
void write_object_transform(
    interaction::Database& database,
    int32_t frame,
    interaction::Transform value) {
    interaction::runtime_fixture_detail::write_vec3(
        database.object_positions,
        static_cast<size_t>(frame),
        value.position);
    write_object_rotation(database, frame, value.rotation);
}
```

- [ ] **Step 2: Add the certified alternate-arm regression**

Use these frozen right-arm local quaternions, which are in-limit and were solved from the fixture's final Hold wrist target:

```cpp
const std::array<quat, 7> alternate_arm = {{
    {0.706170559F, -0.0991955474F, 0.0976584703F, 0.694223464F},
    {0.99653852F, 0.0831327066F, 0.0F, 0.0F},
    {0.891591847F, 0.0F, -0.452839911F, 0.0F},
    {0.792196512F, 0.0F, 0.0F, -0.610266089F},
    {0.943883598F, -0.330278367F, 0.0F, 0.0F},
    {0.997371793F, 0.0F, 0.0F, -0.0724532753F},
    {0.999991298F, 0.0F, 0.0041709533F, 0.0F},
}};
```

Construct frames `[115,150)` by retaining original roots/positions, copying final Hold rotations to every non-root bone, overriding bones 24 through 30 with `alternate_arm`, and setting this constant marker in `hand_dof[0]`:

```cpp
constexpr float kRecordedMarker = 42.0F;
for (int32_t frame = 115; frame < 150; ++frame) {
    for (size_t bone = 1U; bone < g1_skeleton::BoneCount; ++bone) {
        write_bone_rotation(
            fixture.database, frame, bone, hold.rotations[bone]);
    }
    for (size_t joint = 0U; joint < alternate_arm.size(); ++joint) {
        write_bone_rotation(
            fixture.database,
            frame,
            static_cast<size_t>(g1_skeleton::RightShoulderPitch) + joint,
            alternate_arm[joint]);
    }
    fixture.database.hand_dof.at(
        static_cast<size_t>(frame) * 14U) = kRecordedMarker;
```

For each mutated frame, re-read its pose and write:

```cpp
const Transform recorded_object = compose(
    hand_world(recorded_pose, Hand::Right),
    inverse(affordance.hand_in_object));
write_object_transform(fixture.database, frame, recorded_object);
}
```

Force all pose/trajectory rows in the range to equal the unmodified Hold query, classify and require exactly `[115,150)`, then run 20 updates at `0.04F`. Track the maximum arm step and its tick; on every tick require recorded mode, exact live root, and existing hand/object tolerance. Require the constant hand-DOF marker at ticks 12, 13, and 14 so the test observes progress 1, completion, and direct handoff:

```cpp
Pose previous = hold;
float maximum_step = 0.0F;
int32_t maximum_tick = -1;
for (int32_t tick = 0; tick < 20; ++tick) {
    const Pose output = controller.update(locomotion, 0.04F);
    const float step = maximum_right_arm_step(previous, output);
    if (step > maximum_step) {
        maximum_step = step;
        maximum_tick = tick;
    }
    assert(controller.recorded());
    assert(near(root_world(output), root_world(locomotion.pose), 2.0e-5F));
    assert(hand_error(
        output,
        Hand::Right,
        affordance,
        controller.object_world()) <= IKConfig{}.accepted_position_m);
    if (tick >= 12 && tick <= 14) {
        assert(near(output.hand_dof[0], kRecordedMarker, 2.0e-4F));
    }
    previous = output;
}
```

End with:

```cpp
const float maximum_allowed =
    2.0F * IKConfig{}.maximum_step_radians;
if (maximum_step > maximum_allowed + 2.0e-4F) {
    std::fprintf(
        stderr,
        "active-arm max step=%.9g tick=%d limit=%.9g\n",
        maximum_step,
        maximum_tick,
        maximum_allowed);
}
assert(maximum_step <= maximum_allowed + 2.0e-4F);
```

Include `<cstdio>` for the failure-only diagnostic and call the new test from `main` immediately after the initial recorded seam regression.

- [ ] **Step 3: Build and capture the required RED**

Run:

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
```

Expected: the binary aborts only at the new final continuity assertion after exercising all 20 updates. The diagnostic must report approximately `active-arm max step=1.30238 tick=10 limit=0.2`; the measured sequence also contains the approximately 0.303116-radian initial correction. Do not edit production code before this RED is captured.

---

### Task 2: Canonicalize candidates and gate every publication

**Files:**
- Modify: `interaction_carry.cpp`
- Test: `tests/cpp/test_interaction_carry.cpp`

**Interfaces:**
- Consumes: `last_safe_pose_`, `CarryCandidate`, `arm(Hand)`, `rotation_distance`, `within_inclusive`, `solve_hand_ik`, and existing Carry/IK configs.
- Produces: internal `maximum_active_arm_step` and `continuous_active_arm` helpers plus branch-canonical candidates; no public API or mutable state.

- [ ] **Step 1: Add the derived inclusive arm gate**

After `rotation_distance`, add:

```cpp
float maximum_active_arm_step(const IKConfig& config) {
    constexpr float kPi = 3.141592654F;
    return config.maximum_step_radians >= 0.5F * kPi
        ? kPi
        : 2.0F * config.maximum_step_radians;
}

bool continuous_active_arm(
    const Pose& previous,
    const Pose& candidate,
    Hand hand,
    const IKConfig& config) {
    const float maximum = maximum_active_arm_step(config);
    for (const HingeJoint& joint : arm(hand)) {
        const size_t bone = static_cast<size_t>(joint.bone);
        if (!within_inclusive(
                rotation_distance(
                    previous.rotations[bone],
                    candidate.rotations[bone]),
                maximum)) {
            return false;
        }
    }
    return true;
}
```

- [ ] **Step 2: Build one strict publication IK config before key handling**

Immediately after candidate validity, create `publication_ik` by copying `ik_config_` and clamping only accepted position/orientation to the unchanged Carry drift limits:

```cpp
IKConfig publication_ik = ik_config_;
publication_ik.accepted_position_m = std::min(
    publication_ik.accepted_position_m,
    config_.maximum_grasp_drift_m);
publication_ik.accepted_orientation_radians = std::min(
    publication_ik.accepted_orientation_radians,
    config_.maximum_grasp_drift_radians);
```

Use this object in the existing transition solve and fallback validation instead of recreating `transition_ik` later.

- [ ] **Step 3: Canonicalize every directly solved candidate from the published branch**

Before seam-key handling, if `candidate.directly_solved`, copy its pose, overwrite only active-arm rotations from `last_safe_pose_`, and solve the unchanged candidate hand target with `publication_ik`. Recompute the object from the solved hand and adopt the branch candidate only if all checks pass:

```cpp
Pose branch_pose = candidate.pose;
for (const HingeJoint& joint : arm(hand_)) {
    const size_t bone = static_cast<size_t>(joint.bone);
    branch_pose.rotations[bone] = last_safe_pose_.rotations[bone];
}
const IKResult branch_result = solve_hand_ik(
    branch_pose,
    hand_,
    compose(candidate.object, affordance_.hand_in_object),
    publication_ik);
bool branch_ready = branch_result.accepted &&
    continuous_active_arm(
        last_safe_pose_, branch_pose, hand_, ik_config_);
Transform branch_object{};
if (branch_ready) {
    branch_object = compose(
        hand_transform(branch_pose, hand_),
        inverse(affordance_.hand_in_object));
    branch_ready = valid_transform(branch_object) &&
        continuous_object(
            desired_object,
            branch_object,
            config_,
            publication_ik);
}
if (branch_ready) {
    candidate.pose = branch_pose;
    candidate.object = branch_object;
} else {
    candidate.directly_solved = false;
}
```

Do not change candidate mode/key or any controller state during this evaluation.

- [ ] **Step 4: Gate both transition retry outputs before publication**

After each `solve_hand_ik` acceptance and before object publication, add:

```cpp
if (!continuous_active_arm(
        last_safe_pose_, transition, hand_, ik_config_)) {
    continue;
}
```

Leave retry order, progress bisection, completion/deactivation, live-root fallback, epochs, and transaction commit points unchanged. Rename later `transition_ik` references to `publication_ik`.

- [ ] **Step 5: Run focused GREEN and record exact maximum**

Run:

```sh
make build/tests/test_interaction_carry && build/tests/test_interaction_carry
```

Expected: exit 0. To record the corrected exact measurement, temporarily add this test-only line immediately before the final assertion, run the same focused command, and then remove only this line:

```cpp
std::fprintf(
    stderr,
    "corrected active-arm max step=%.9g tick=%d\n",
    maximum_step,
    maximum_tick);
```

Run the focused command again after removing the line. Expected: exit 0 with no diagnostic output.

---

### Task 3: Verify integration, optimized safety, and commit

**Files:**
- Verify: `interaction_carry.cpp`
- Verify: `tests/cpp/test_interaction_carry.cpp`
- Verify: `interaction_runtime.cpp`, `interaction_controller_adapter.cpp`, `tests/cpp/test_interaction_carry_fast_math.cpp`

**Interfaces:**
- Consumes: the corrected Carry implementation and all existing safe targets.
- Produces: one scoped implementation commit and exact RED/GREEN evidence.

- [ ] **Step 1: Run focused integration binaries**

```sh
make build/tests/test_interaction_runtime && \
  build/tests/test_interaction_runtime
make build/tests/test_interaction_controller_adapter && \
  build/tests/test_interaction_controller_adapter
make test-interaction-carry-release-fast-math
```

Expected: every command exits 0 with no changed public defaults or invalid-input behavior.

- [ ] **Step 2: Run the full safe suite**

```sh
git diff --check
make test-interaction-safe
```

Expected: exit 0; the Python suite, all normal C++ binaries, target fast-math validation, and Carry fast-math validation pass. Only the two environment-gated graphical/full-pack tests remain skipped in the safe Python run.

- [ ] **Step 3: Review the scoped diff and commit**

Confirm only `interaction_carry.cpp` and `tests/cpp/test_interaction_carry.cpp` changed after the committed plan. Remove temporary production diagnostics, run `git diff --check`, stage only those files, and commit:

```sh
git add -- interaction_carry.cpp tests/cpp/test_interaction_carry.cpp
git commit -m "fix: keep carry arm on a continuous ik branch"
```

- [ ] **Step 4: Final tracked-clean evidence**

```sh
git status --short --untracked-files=no
git log -4 --format='%H %s'
```

Expected: tracked status is empty and the spec, plan, and implementation commits are visible. Report the exact RED maximum/tick, corrected maximum/tick, all verification commands, implementation hash, and the separate Head/Neck plus 25 Hz/60 Hz visual-continuity concerns.
