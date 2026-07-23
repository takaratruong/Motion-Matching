# Direct Reach Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank direct wrist approaches before motions that move away from the grasp and hook back, without rejecting or removing any valid motion.

**Architecture:** Add a deterministic, rigid-transform-invariant wrist-path quality measurement to the reach-coverage module and publish it on every evaluation. The exhaustive search uses the combined directness cost as its first accepted-result sort key, while the viewer exposes the component metrics for visual validation.

**Tech Stack:** C++17, existing `vec3` math helpers, Make-based C++ tests, raylib coverage viewer.

## Global Constraints

- Candidate enumeration, posture IK, grasp placement, collision checks, acceptance thresholds, and accepted-motion counts must not change.
- `directness_cost = backtrack_ratio + 0.25 * excess_path_ratio`.
- Normalize by `max(distance(path.front(), path.back()), 0.05 m)`.
- Directness changes ordering only; hooked reaches remain cycleable fallbacks.
- Real-pack open-space coverage remains exactly 477 accepted motions: 231 left, 246 right, and ten root-azimuth sectors.
- The full 4,608-instance search must complete within 30 seconds.

---

### Task 1: Measure wrist-path directness

**Files:**
- Modify: `reach_coverage.h`
- Modify: `reach_coverage.cpp`
- Test: `tests/cpp/test_reach_coverage.cpp`

**Interfaces:**
- Consumes: `std::vector<vec3>` and the existing `length(vec3)` math helper.
- Produces: `reach::WristPathQuality` and `reach::measure_wrist_path_quality(const std::vector<vec3>&)`.

- [ ] **Step 1: Write failing metric tests**

Add this public behavior test to `tests/cpp/test_reach_coverage.cpp`:

```cpp
bool near(float left, float right, float tolerance = 1.0e-5F) {
    return std::abs(left - right) <= tolerance;
}

void test_wrist_path_quality_penalizes_hooks_and_is_rigid_invariant() {
    const std::vector<vec3> direct = {
        vec3(0.0F, 0.0F, 0.0F),
        vec3(0.5F, 0.0F, 0.0F),
        vec3(1.0F, 0.0F, 0.0F),
    };
    const std::vector<vec3> hooked = {
        vec3(0.0F, 0.0F, 0.0F),
        vec3(-0.25F, 0.0F, 0.0F),
        vec3(0.5F, 0.0F, 0.0F),
        vec3(1.0F, 0.0F, 0.0F),
    };
    const reach::WristPathQuality direct_quality =
        reach::measure_wrist_path_quality(direct);
    const reach::WristPathQuality hooked_quality =
        reach::measure_wrist_path_quality(hooked);
    assert(near(direct_quality.backtrack_ratio, 0.0F));
    assert(near(direct_quality.excess_path_ratio, 0.0F));
    assert(near(direct_quality.directness_cost, 0.0F));
    assert(near(hooked_quality.backtrack_ratio, 0.25F));
    assert(near(hooked_quality.excess_path_ratio, 0.50F));
    assert(near(hooked_quality.directness_cost, 0.375F));

    const quat yaw = quat_from_angle_axis(
        1.1F, vec3(0.0F, 1.0F, 0.0F));
    std::vector<vec3> transformed;
    for (const vec3 point : hooked) {
        transformed.push_back(
            quat_mul_vec3(yaw, point) + vec3(4.0F, -2.0F, 7.0F));
    }
    const reach::WristPathQuality transformed_quality =
        reach::measure_wrist_path_quality(transformed);
    assert(near(
        transformed_quality.backtrack_ratio,
        hooked_quality.backtrack_ratio));
    assert(near(
        transformed_quality.excess_path_ratio,
        hooked_quality.excess_path_ratio));
    assert(near(
        transformed_quality.directness_cost,
        hooked_quality.directness_cost));
}
```

Call the test from `main()`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
make build/cpp-tests/test_reach_coverage
```

Expected: compilation fails because `WristPathQuality` and
`measure_wrist_path_quality` do not exist.

- [ ] **Step 3: Add the metric interface**

Add to `reach_coverage.h`:

```cpp
struct WristPathQuality {
    float backtrack_ratio = 0.0F;
    float excess_path_ratio = 0.0F;
    float directness_cost = 0.0F;
};

WristPathQuality measure_wrist_path_quality(
    const std::vector<vec3>& path);
```

- [ ] **Step 4: Implement the metric**

Add to `reach_coverage.cpp` in namespace `reach`, outside the anonymous
namespace:

```cpp
WristPathQuality measure_wrist_path_quality(
    const std::vector<vec3>& path) {
    const float invalid = std::numeric_limits<float>::infinity();
    for (const vec3 point : path) {
        if (!finite(point)) return {invalid, invalid, invalid};
    }
    if (path.size() < 2U) return {};

    const vec3 endpoint = path.back();
    const float denominator = std::max(
        length(endpoint - path.front()), 0.05F);
    float previous_distance = length(endpoint - path.front());
    float backtrack = 0.0F;
    float path_length = 0.0F;
    for (size_t sample = 1U; sample < path.size(); ++sample) {
        const float current_distance = length(endpoint - path[sample]);
        backtrack += std::max(0.0F, current_distance - previous_distance);
        path_length += length(path[sample] - path[sample - 1U]);
        previous_distance = current_distance;
    }

    WristPathQuality result{};
    result.backtrack_ratio = backtrack / denominator;
    result.excess_path_ratio = std::max(
        0.0F, path_length / denominator - 1.0F);
    result.directness_cost =
        result.backtrack_ratio + 0.25F * result.excess_path_ratio;
    return result;
}
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
make build/cpp-tests/test_reach_coverage &&
./build/cpp-tests/test_reach_coverage
```

Expected: build succeeds and the binary exits 0.

- [ ] **Step 6: Commit**

```bash
git add reach_coverage.h reach_coverage.cpp \
  tests/cpp/test_reach_coverage.cpp
git commit -m "feat: measure reach path directness"
```

### Task 2: Publish metrics and rank direct reaches first

**Files:**
- Modify: `reach_coverage.h`
- Modify: `reach_coverage.cpp`
- Modify: `reach_search.h`
- Modify: `reach_search.cpp`
- Test: `tests/cpp/test_reach_search.cpp`

**Interfaces:**
- Consumes: `measure_wrist_path_quality(const std::vector<vec3>&)`.
- Produces: three path-quality fields on `reach::Evaluation` and
  `reach::detail::accepted_quality_less(const Evaluation&, const Evaluation&)`.

- [ ] **Step 1: Write failing ranking and parity tests**

Add to `tests/cpp/test_reach_search.cpp`:

```cpp
void test_direct_reach_ranks_before_hooked_fallback() {
    reach::Evaluation direct{};
    direct.rejection = reach::Rejection::None;
    direct.directness_cost = 0.05F;
    direct.active_arm_deformation = 1.0F;
    direct.candidate.clip = 1U;

    reach::Evaluation hooked{};
    hooked.rejection = reach::Rejection::None;
    hooked.directness_cost = 0.50F;
    hooked.active_arm_deformation = 0.0F;
    hooked.candidate.clip = 0U;

    assert(reach::detail::accepted_quality_less(direct, hooked));
    assert(!reach::detail::accepted_quality_less(hooked, direct));
    assert(direct.rejection == reach::Rejection::None);
    assert(hooked.rejection == reach::Rejection::None);
}
```

Extend `test_selected_regeneration_matches_compact_metrics()`:

```cpp
const reach::Evaluation& compact =
    result.evaluations.front().evaluation;
assert(std::abs(
    regenerated.backtrack_ratio - compact.backtrack_ratio) <= 1.0e-5F);
assert(std::abs(
    regenerated.excess_path_ratio - compact.excess_path_ratio) <= 1.0e-5F);
assert(std::abs(
    regenerated.directness_cost - compact.directness_cost) <= 1.0e-5F);
```

Call the ranking test from `main()`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
make build/cpp-tests/test_reach_search
```

Expected: compilation fails because the evaluation fields and comparator do
not exist.

- [ ] **Step 3: Publish and populate evaluation metrics**

Add to `Evaluation` in `reach_coverage.h`:

```cpp
float backtrack_ratio = 0.0F;
float excess_path_ratio = 0.0F;
float directness_cost = 0.0F;
```

After `shape_candidate()` has populated `evaluation.poses`, extract the active
wrist positions, call `measure_wrist_path_quality`, and assign all three
fields:

```cpp
std::vector<vec3> wrist_path;
wrist_path.reserve(evaluation.poses.size());
const size_t wrist = wrist_bone(query.hand);
for (const interaction::Pose& pose : evaluation.poses) {
    wrist_path.push_back(interaction::world_pose(pose).positions[wrist]);
}
const WristPathQuality path_quality =
    measure_wrist_path_quality(wrist_path);
evaluation.backtrack_ratio = path_quality.backtrack_ratio;
evaluation.excess_path_ratio = path_quality.excess_path_ratio;
evaluation.directness_cost = path_quality.directness_cost;
```

Include these fields in the existing finite-value validation.

- [ ] **Step 4: Add and use the accepted-result comparator**

Declare in `reach_search.h` under `namespace detail`:

```cpp
bool accepted_quality_less(
    const Evaluation& left,
    const Evaluation& right);
```

Implement in `reach_search.cpp`:

```cpp
bool accepted_quality_less(
    const Evaluation& left,
    const Evaluation& right) {
    return std::tie(
               left.directness_cost,
               left.active_arm_deformation,
               left.orientation_error_radians,
               left.approach_error_radians,
               left.candidate.clip,
               left.candidate.yaw_index) <
           std::tie(
               right.directness_cost,
               right.active_arm_deformation,
               right.orientation_error_radians,
               right.approach_error_radians,
               right.candidate.clip,
               right.candidate.yaw_index);
}
```

Replace the inline accepted-result sort tuple with:

```cpp
return detail::accepted_quality_less(left, right);
```

Extend `regenerate()` parity checks with `same_metric` comparisons for all
three new metrics.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
make build/cpp-tests/test_reach_coverage \
  build/cpp-tests/test_reach_search &&
./build/cpp-tests/test_reach_coverage &&
./build/cpp-tests/test_reach_search
```

Expected: both binaries exit 0.

- [ ] **Step 6: Commit**

```bash
git add reach_coverage.h reach_coverage.cpp reach_search.h reach_search.cpp \
  tests/cpp/test_reach_search.cpp
git commit -m "feat: rank direct reaches before hooked fallbacks"
```

### Task 3: Expose quality and verify real-pack behavior

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Modify: `docs/superpowers/specs/2026-07-23-direct-reach-ranking-design.md`

**Interfaces:**
- Consumes: the three path-quality fields on `reach::Evaluation`.
- Produces: selected-option HUD diagnostics and final real-pack evidence.

- [ ] **Step 1: Add HUD diagnostics**

Add a selected-option line:

```cpp
DrawText(
    TextFormat(
        "DIRECT %.3f | BACKTRACK %.1f%% | EXCESS PATH %.1f%%",
        evaluation.directness_cost,
        100.0F * evaluation.backtrack_ratio,
        100.0F * evaluation.excess_path_ratio),
    26,
    289,
    16,
    evaluation.directness_cost <= 0.10F ? DARKGREEN : DARKGRAY);
```

Move the collision line to Y=314, the rejected-path diagnostic to Y=343, and
the controls footer to Y=367. Increase the HUD rectangle height from 370 to
400 pixels.

- [ ] **Step 2: Run all focused builds and tests**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/cpp-tests/test_reach_coverage \
  build/cpp-tests/test_reach_search \
  build/cpp-tests/test_interaction_posture_ik \
  build/cpp-tests/test_interaction_hand_trajectories \
  g1_reach_coverage_probe g1_reach_coverage_viewer
./build/cpp-tests/test_reach_coverage
./build/cpp-tests/test_reach_search
./build/cpp-tests/test_interaction_posture_ik
./build/cpp-tests/test_interaction_hand_trajectories
```

Expected: 12 Python viewer tests pass, all C++ binaries exit 0, and both
viewer/probe release builds succeed.

- [ ] **Step 3: Run the real-pack coverage gate**

Run:

```bash
./g1_reach_coverage_probe \
  build/g1-reaches/reach-pack-v2 \
  build/g1-reaches/contact-anchored-coverage-report-v6.json
```

Expected: exit 0; open space reports 477 accepted, 231 left, 246 right, ten
root-azimuth sectors, all 4,608 instances processed, and elapsed time no more
than 30 seconds.

- [ ] **Step 4: Record final evidence**

Append a `## Result` section to the design spec containing the exact focused
test counts, real-pack acceptance counts, search elapsed time, and confirmation
that directness changed ordering without changing the accepted set.

- [ ] **Step 5: Commit**

```bash
git add g1_reach_coverage_viewer.cpp \
  docs/superpowers/specs/2026-07-23-direct-reach-ranking-design.md
git commit -m "docs: record direct reach ranking evidence"
```

- [ ] **Step 6: Relaunch the flat viewer**

Stop only the exact old coverage-viewer process, then launch the rebuilt
viewer on display `:1`:

```bash
pkill -f '^./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v2$'
DISPLAY=:1 ./g1_reach_coverage_viewer \
  build/g1-reaches/reach-pack-v2 \
  >/tmp/g1_reach_coverage_viewer.log 2>&1 &
```

Verify exactly one matching process is running and the log contains no startup
error. In the viewer, press `G` for OPEN, press Enter, wait for completion, and
cycle the first accepted options with `[` and `]`.
