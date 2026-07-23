# Straight-In Grasp Approach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank reusable G1 reaches by straight-in grasp compatibility and lazily retarget the selected reach through a collision-free 20 cm pre-grasp corridor while preserving candidate diversity.

**Architecture:** Add a focused `reach_straight_approach` module for grasp-frame metrics and selected-candidate corridor retargeting. Keep the exhaustive search's existing accepted list intact, add a separately sorted preferred list, and integrate raw/preferred/retargeted modes into the restored reach-coverage viewer. Retargeting reuses the posture-aware waist-plus-arm IK and existing full-body trajectory collision oracle, with a conservative dynamic open-gripper proxy.

**Tech Stack:** C++17, existing G1 `interaction::Pose`, posture-aware damped least-squares IK, reach exhaustive search, raylib viewer, Make, Python `unittest`.

## Global Constraints

- `approach_world` is a finite unit vector pointing from pre-grasp toward contact.
- Default corridor length is exactly `0.20F` metres.
- Default corridor radius is exactly `0.03F` metres.
- Default gripper close distance is exactly `0.03F` metres.
- Default maximum backward axial step is exactly `0.002F` metres.
- Default pre-grasp orientation tolerance is 10 degrees.
- Approach preference must never remove or reject an existing raw accepted candidate.
- Retargeting is lazy and applies only to the selected candidate.
- Retargeting may modify only the three waist and seven active-arm joints plus the active hand's seven DOFs.
- Root translation and all motion before the correction blend remain bit-identical.
- Every retargeted frame must be finite and remain within hard G1 joint limits.
- Object contact is forbidden before the existing final-contact-only interval.
- Locomotion, carrying, placement, episode playback, outbound return search, and grasp-policy training are out of scope.
- The v3 reach pack and all generated `build/g1-reaches/` artifacts remain untracked.

---

### Task 1: Grasp-Frame Straight-Approach Metrics

**Files:**
- Create: `reach_straight_approach.h`
- Create: `reach_straight_approach.cpp`
- Create: `tests/cpp/test_reach_straight_approach.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `std::vector<vec3>` wrist paths, contact transform, and `approach_world`.
- Produces:

```cpp
namespace reach {

struct StraightApproachConfig {
    float corridor_length_m = 0.20F;
    float corridor_radius_m = 0.03F;
    float close_distance_m = 0.03F;
    float maximum_backward_step_m = 0.002F;
    float maximum_blend_seconds = 0.20F;
    float maximum_pregrasp_orientation_radians = 0.174532925F;
    float open_gripper_radius_m = 0.065F;
    float closed_gripper_radius_m = 0.040F;
    std::array<float, 7U> open_active_hand_dof{};
};

struct StraightApproachQuality {
    bool finite = false;
    bool reaches_pregrasp_plane = false;
    float maximum_lateral_m = 0.0F;
    float rms_lateral_m = 0.0F;
    float backward_ratio = 0.0F;
    float maximum_angle_radians = 0.0F;
    float rms_angle_radians = 0.0F;
    size_t corridor_start = 0U;
};

StraightApproachQuality measure_straight_approach(
    const std::vector<vec3>& wrist_path,
    vec3 contact_world,
    vec3 approach_world,
    const StraightApproachConfig& config = {});

bool straight_approach_quality_less(
    const StraightApproachQuality& left,
    const StraightApproachQuality& right);

}  // namespace reach
```

- [ ] **Step 1: Write failing metric tests**

Add real paths that prove straight motion outranks lateral sliding and hooks,
that backward motion is measured, that a path shorter than 20 cm reports no
pre-grasp coverage, and that a rigid transform leaves all scalar metrics
unchanged:

```cpp
void test_straight_path_outranks_slide_and_hook() {
    const vec3 contact(0.0F, 0.0F, 0.0F);
    const vec3 approach(1.0F, 0.0F, 0.0F);
    const std::vector<vec3> straight = {
        {-0.20F, 0.0F, 0.0F},
        {-0.10F, 0.0F, 0.0F},
        { 0.00F, 0.0F, 0.0F},
    };
    const std::vector<vec3> slide = {
        {-0.20F, 0.06F, 0.0F},
        {-0.10F, 0.04F, 0.0F},
        { 0.00F, 0.00F, 0.0F},
    };
    const auto direct = reach::measure_straight_approach(
        straight, contact, approach);
    const auto lateral = reach::measure_straight_approach(
        slide, contact, approach);
    require(direct.reaches_pregrasp_plane, "straight path missed pre-grasp");
    require(direct.maximum_lateral_m < 1.0e-6F,
        "straight path gained lateral drift");
    require(reach::straight_approach_quality_less(direct, lateral),
        "lateral slide outranked straight approach");
}
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
make build/tests/test_reach_straight_approach
```

Expected: compilation fails because `reach_straight_approach.h` and its API do
not exist.

- [ ] **Step 3: Implement validation, suffix selection, and metrics**

Implement strict config/query validation. For each path sample compute:

```cpp
const vec3 from_contact = contact_world - wrist_path[sample];
const float remaining = dot(from_contact, approach);
const vec3 lateral_vector =
    from_contact - remaining * approach;
```

Use the earliest final-contiguous sample with
`remaining <= corridor_length_m`; set `reaches_pregrasp_plane` only when a
sample reaches `corridor_length_m - 1.0e-4F`. Compute segment angle only for
segments longer than `1.0e-6F`. Return `finite=false` and infinite sortable
metrics for invalid paths.

Implement the exact preference tuple:

```cpp
return std::tie(
    !left.reaches_pregrasp_plane,
    left.maximum_lateral_m,
    left.backward_ratio,
    left.rms_angle_radians,
    left.maximum_angle_radians) <
  std::tie(
    !right.reaches_pregrasp_plane,
    right.maximum_lateral_m,
    right.backward_ratio,
    right.rms_angle_radians,
    right.maximum_angle_radians);
```

- [ ] **Step 4: Add the Make target and verify GREEN**

Link the new test only against the new metric module and math headers. Run:

```bash
make build/tests/test_reach_straight_approach
./build/tests/test_reach_straight_approach
```

Expected: `straight approach metrics PASS`.

- [ ] **Step 5: Commit**

```bash
git add reach_straight_approach.h reach_straight_approach.cpp \
  tests/cpp/test_reach_straight_approach.cpp Makefile
git commit -m "feat: measure straight-in grasp approaches"
```

---

### Task 2: Preserve Raw Acceptance and Add Preferred Ordering

**Files:**
- Modify: `reach_search.h`
- Modify: `reach_search.cpp`
- Modify: `tests/cpp/test_reach_search.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `measure_straight_approach` from Task 1.
- Produces:

```cpp
struct CompactEvaluation {
    Evaluation evaluation{};
    std::vector<vec3> hand_path;
    StraightApproachQuality straight_approach{};
};

struct SearchResult {
    bool complete = false;
    size_t total = 0U;
    size_t processed = 0U;
    std::chrono::steady_clock::duration elapsed{};
    std::vector<CompactEvaluation> evaluations;
    std::vector<size_t> accepted;   // Existing raw accepted order.
    std::vector<size_t> preferred;  // Same identities, straight-in order.
};
```

- [ ] **Step 1: Write failing preservation and ranking tests**

Construct three accepted compact evaluations: straight, sliding, and hooked.
Require:

```cpp
require(result.accepted.size() == result.preferred.size(),
    "preference removed accepted candidates");
std::vector<size_t> raw_identities = result.accepted;
std::vector<size_t> preferred_identities = result.preferred;
std::sort(raw_identities.begin(), raw_identities.end());
std::sort(preferred_identities.begin(), preferred_identities.end());
require(raw_identities == preferred_identities,
    "preferred list changed candidate identity");
require(result.preferred.front() == straight_index,
    "straight candidate was not preferred");
```

Also assert the old `accepted` order remains controlled by
`detail::accepted_quality_less`.

- [ ] **Step 2: Run and verify RED**

```bash
make build/tests/test_reach_search
```

Expected: compilation fails because `CompactEvaluation::straight_approach` and
`SearchResult::preferred` do not exist.

- [ ] **Step 3: Populate compact metrics and stable preferred order**

After extracting each active wrist path:

```cpp
compact.straight_approach = measure_straight_approach(
    compact.hand_path,
    query.target.position,
    query.approach_world,
    config.straight_approach);
```

Add `StraightApproachConfig straight_approach{}` to `SearchConfig`. After the
existing accepted sort:

```cpp
result.preferred = result.accepted;
std::stable_sort(
    result.preferred.begin(), result.preferred.end(),
    [&](size_t left, size_t right) {
        const auto& l = result.evaluations[left];
        const auto& r = result.evaluations[right];
        if (straight_approach_quality_less(
                l.straight_approach, r.straight_approach)) return true;
        if (straight_approach_quality_less(
                r.straight_approach, l.straight_approach)) return false;
        return detail::accepted_quality_less(
            l.evaluation, r.evaluation);
    });
```

`regenerate` must recompute the metrics and compare every scalar within
`1.0e-5F`, including the pre-grasp boolean.

- [ ] **Step 4: Verify focused and existing reach tests**

```bash
make build/tests/test_reach_straight_approach \
  build/tests/test_reach_search build/tests/test_reach_coverage
./build/tests/test_reach_straight_approach
./build/tests/test_reach_search
./build/tests/test_reach_coverage
```

Expected: all three binaries exit 0; raw acceptance counts are unchanged.

- [ ] **Step 5: Commit**

```bash
git add reach_search.h reach_search.cpp tests/cpp/test_reach_search.cpp Makefile
git commit -m "feat: rank reaches by straight-in preference"
```

---

### Task 3: Lazy Posture-Aware Corridor Retarget

**Files:**
- Modify: `reach_straight_approach.h`
- Modify: `reach_straight_approach.cpp`
- Modify: `tests/cpp/test_reach_straight_approach.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: a regenerated accepted `Evaluation`, hand, grasp target, object,
  environment, and collision configuration.
- Produces:

```cpp
enum class CorridorRetargetFailure : uint8_t {
    None = 0U,
    NoPregraspCoverage,
    InvalidInput,
    InvalidSolver,
    OutsideCorridor,
    BackwardMotion,
    ObjectCollision,
    EnvironmentCollision,
};

struct CorridorRetargetResult {
    bool accepted = false;
    CorridorRetargetFailure failure =
        CorridorRetargetFailure::InvalidInput;
    size_t failure_sample = 0U;
    size_t blend_start = 0U;
    size_t corridor_start = 0U;
    float active_arm_deformation = 0.0F;
    StraightApproachQuality quality{};
    std::vector<interaction::Pose> poses;
};

CorridorRetargetResult retarget_straight_approach(
    const std::vector<interaction::Pose>& source,
    Hand hand,
    const interaction::Transform& hand_world,
    vec3 approach_world,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    float fps,
    const StraightApproachConfig& config = {},
    const interaction::PostureIKConfig& ik_config = {},
    const interaction::TrajectoryCollisionConfig& collision_config = {});
```

- [ ] **Step 1: Write failing retarget tests**

Use a synthetic 31-bone pose sequence with a lateral final approach. Require:

- samples before `blend_start` are bit-identical
- simulation/root position and rotation are unchanged at every sample
- the retargeted wrist stays within `0.03F` of the requested line
- no axial step is less than `-0.002F`
- final wrist position/orientation are exact within existing tolerances
- only waist, active arm, and active hand DOFs change
- the active hand stays at configured open DOFs until 3 cm remains
- colliding object/environment fixtures fail closed
- a short path reports `NoPregraspCoverage`

The root assertion must compare every frame:

```cpp
bool same_vec3_bits(vec3 left, vec3 right) {
    return std::memcmp(&left, &right, sizeof(vec3)) == 0;
}

bool same_quat_bits(quat left, quat right) {
    return std::memcmp(&left, &right, sizeof(quat)) == 0;
}

require(
    same_vec3_bits(
        result.poses[i].positions[g1_skeleton::Simulation],
        source[i].positions[g1_skeleton::Simulation]) &&
    same_quat_bits(
        result.poses[i].rotations[g1_skeleton::Simulation],
        source[i].rotations[g1_skeleton::Simulation]),
    "corridor retarget moved the root");
```

Include `<cstring>` in the test for these exact-bit helpers.

- [ ] **Step 2: Run and verify RED**

```bash
make build/tests/test_reach_straight_approach
```

Expected: compilation fails because the retarget API does not exist.

- [ ] **Step 3: Implement spatial targets and temporal blend**

Normalize the approach, find the corridor start from Task 1, and compute:

```cpp
const size_t maximum_blend_samples = static_cast<size_t>(
    std::ceil(config.maximum_blend_seconds * fps));
const size_t blend_start =
    corridor_start > maximum_blend_samples
        ? corridor_start - maximum_blend_samples
        : 0U;
```

For the corridor, derive monotonic normalized progress from the source axial
projection, clamp it to the previous progress, then smoothstep it:

```cpp
const vec3 pregrasp =
    hand_world.position -
    config.corridor_length_m * approach;
const vec3 desired_position = lerp(
    pregrasp, hand_world.position, smoothstep(progress));
```

Blend to pre-grasp during `[blend_start, corridor_start]`. Hold
`hand_world.rotation` throughout the corridor. Solve each changed sample with
`solve_hand_posture_ik_task_priority`, using the previous solution as the
temporal seed and hard G1 limits.

Modify only active hand indices `[0, 7)` for left or `[7, 14)` for right.
Keep them open until remaining distance is at most `close_distance_m`, then
smoothstep to the source contact DOFs. Recompute only those active hand DOF
velocities at the native fps.

- [ ] **Step 4: Add dynamic gripper proxy and full collision validation**

For each retargeted sample, use a wrist-centred sphere whose radius interpolates
from `open_gripper_radius_m` to `closed_gripper_radius_m` over the close phase.
Sweep that sphere from the previous wrist position to the current wrist
position against the object and every environment box. Object contact is
allowed only at the terminal sample under the existing endpoint-only epsilon.

Then build `interaction::ShapedHandTrajectory` from all solved poses and call
`evaluate_shaped_trajectory_feasibility` to revalidate the complete G1 body.
Map either failure to the corresponding `CorridorRetargetFailure`.

- [ ] **Step 5: Verify GREEN and regression tests**

```bash
make build/tests/test_reach_straight_approach \
  build/tests/test_reach_coverage build/tests/test_reach_search
./build/tests/test_reach_straight_approach
./build/tests/test_reach_coverage
./build/tests/test_reach_search
```

Expected: all pass with explicit `PASS` output.

- [ ] **Step 6: Commit**

```bash
git add reach_straight_approach.h reach_straight_approach.cpp \
  tests/cpp/test_reach_straight_approach.cpp Makefile
git commit -m "feat: retarget straight-in grasp corridor"
```

---

### Task 4: Coverage Viewer Raw, Preferred, and Retargeted Modes

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `SearchResult::accepted`, `SearchResult::preferred`, and
  `retarget_straight_approach`.
- Produces: `A` mode switching, cached bracket cycling, corridor/path drawing,
  and HUD diagnostics.

- [ ] **Step 1: Write failing viewer contract tests**

Require all of:

```python
self.assertIn("enum class ApproachDisplayMode", source)
self.assertIn("KEY_A", source)
self.assertIn("results->preferred", source)
self.assertIn("reach::retarget_straight_approach(", source)
self.assertIn("RAW", source)
self.assertIn("PREFERRED", source)
self.assertIn("RETARGETED", source)
self.assertIn("MAX LATERAL", source)
self.assertIn("RMS LATERAL", source)
self.assertIn("BACKWARD", source)
self.assertIn("NO PREGRASP COVERAGE", source)
self.assertIn("CORRIDOR RETARGET FAILED", source)
```

Also assert `KEY_LEFT_BRACKET` and `KEY_RIGHT_BRACKET` remain, and source-order
checks prove bracket handling never calls `reach::search_all`.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer -v
```

Expected: new mode/diagnostic assertions fail.

- [ ] **Step 3: Add mode and cached selection**

Implement:

```cpp
enum class ApproachDisplayMode : uint8_t {
    Raw = 0U,
    Preferred = 1U,
    Retargeted = 2U,
};
```

`A` cycles modes and resets only selection/animation/selected retarget; it must
not clear `results` or invoke search. `RAW` indexes `results->accepted`;
`PREFERRED` and `RETARGETED` index `results->preferred`. Brackets wrap within
the active vector and lazily regenerate only the selected full evaluation.

- [ ] **Step 4: Add selected-only retarget and visualization**

In `RETARGETED` mode, call the corridor retarget API immediately after
regenerating the selected evaluation. Cache the result until selection, object,
grasp, environment, or mode changes.

Draw:

- requested pre-grasp point as a gold sphere
- requested 20 cm corridor as a gold cylinder
- raw wrist path in translucent purple
- accepted retarget path in emphasized green
- failed retarget path in orange with its failure string

The animated mesh/bones use retargeted poses only when the retarget is accepted;
otherwise they use the regenerated recorded poses.

- [ ] **Step 5: Add HUD diagnostics**

Show mode, raw/preferred rank, total accepted count, max/RMS lateral drift,
max/RMS angle, backward ratio, pre-grasp coverage, retarget status, and active
arm deformation. Preserve existing rejection/collision diagnostics and all
object/camera/environment controls.

- [ ] **Step 6: Verify viewer tests and build**

```bash
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer -v
make g1_reach_coverage_viewer
```

Expected: Python tests pass and the release viewer builds.

- [ ] **Step 7: Commit**

```bash
git add g1_reach_coverage_viewer.cpp \
  tests/python/test_g1_reach_coverage_viewer.py Makefile
git commit -m "feat: browse preferred straight-in reaches"
```

---

### Task 5: Protected Verification and Live v3 Acceptance

**Files:**
- Modify only if a scoped defect is found in the files authorized by Tasks 1–4.
- Do not add generated packs, binaries, screenshots, or scratch reports.

**Interfaces:**
- Consumes: final branch implementation and
  `build/g1-reaches/reach-pack-v3`.
- Produces: protected evidence and a live viewer left running for user review.

- [ ] **Step 1: Run the protected task gate**

The controller-owned verifier must check:

- only the allowed implementation/test/Make paths changed
- required metric, ordering, retarget, collision, and viewer tests exist
- RED/GREEN-focused binaries pass
- raw accepted identities equal preferred identities
- root and pre-blend preservation tests are present and passing
- `[`/`]` do not rerun exhaustive search

Expected public success:

```text
verified straight-in preference -> lazy corridor retarget -> cached viewer cycling
```

- [ ] **Step 2: Run focused repository verification**

```bash
make build/tests/test_reach_straight_approach \
  build/tests/test_reach_search build/tests/test_reach_coverage \
  g1_reach_coverage_viewer
./build/tests/test_reach_straight_approach
./build/tests/test_reach_search
./build/tests/test_reach_coverage
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer -v
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Build and validate v3 search data**

Run the existing v3 pack loader/probe without changing generated artifacts:

```bash
./g1_reach_coverage_probe \
  build/g1-reaches/reach-pack-v3 \
  --json build/g1-reaches/straight-approach-v3-coverage.json
```

Require full 4,608-instance processing and search-integrity success.

- [ ] **Step 4: Replace the viewer only after candidate verification**

Keep the current known-good coverage viewer alive until the candidate viewer
launches successfully. Then close the old process, launch:

```bash
./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v3
```

- [ ] **Step 5: Drive the viewer**

Using synthetic key input and screenshots:

1. Press `G` for open environment and `Enter` to search.
2. Confirm multiple accepted and preferred candidates.
3. Cycle at least five candidates with `]` and one with `[`.
4. Confirm search elapsed/counts do not change while cycling.
5. Cycle `A` through raw, preferred, and retargeted.
6. Confirm preferred option 1 has no worse straight-in tuple than option 2.
7. Confirm accepted retarget animation stays inside the visible corridor.
8. Move and rotate the object, confirm results become stale, rerun `Enter`.
9. Repeat one retarget in the strict coverage environment.

- [ ] **Step 6: Independent review and final commit**

Request read-only review of the implementation against the design. Fix only
concrete findings, rerun the focused and protected gates, then commit any
review repairs. Leave the verified coverage viewer running.
