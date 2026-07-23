# Contact-Anchored Exhaustive Grasp Motion Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every bilateral recorded reach reusable at any physically reachable grasp by contact-anchoring the complete motion, sampling 12 upright body yaws, retaining the current IK solver, and accepting candidates only after complete kinematic and scene-collision evaluation.

**Architecture:** Add a pure placement layer that maps canonical clips into the world from their terminal wrist contact. Build exhaustive shaping on that layer, then expose one bounded parallel search service shared by the flat viewer and headless evidence probe. The viewer retains compact paths and diagnostics for all 4,608 instances while regenerating a full pose sequence only for the selected option.

**Tech Stack:** C++17, existing G1 pose/FK and seven-joint IK, raylib, Python 3 `unittest`, Make, JSON evidence.

## Global Constraints

- The real v2 pack contains 192 captured-left and 192 mirrored-right reaches.
- Every search enumerates all 384 clips at exactly 12 world-up yaw placements: 4,608 raw instances.
- Query translation and rotation never change the raw candidate population.
- Recorded endpoint distance, wrist orientation, and approach direction are not eligibility filters.
- Whole-body placement aligns terminal contact only in X/Z, preserves every
  recorded root Y sample, and never inherits object pitch or roll. Vertical
  grasp displacement is handled by gradual active-arm IK. This live-review
  correction supersedes the original full-3D placement snippets in Task 1.
- Keep `interaction_ik.h`, `interaction_ik.cpp`, and `g1_arm_joint_metadata.h` unchanged.
- Accepted final wrist position error is at most `0.001 m`; approach error remains at most 15 degrees and orientation error at most 60 degrees.
- Complete shaped skeletons are tested against target-object and furniture geometry.
- Search is explicit, deterministic, complete within 30 seconds, and never publishes partial results as exhaustive.
- The viewer starts with the table/shelf/lower-table environment visible and runs as exactly one flat viewer without controller, mesh, terrain, diffusion, or screenshots.

---

## File Structure

- Create `reach_placement.h/.cpp`: pure 12-yaw enumeration and contact-relative whole-pose placement.
- Modify `reach_coverage.h/.cpp`: exhaustive candidate identity, placed active-arm shaping, 1 mm final gate, deformation metric, and existing collision classification.
- Create `reach_search.h/.cpp`: bounded parallel exhaustive evaluation, compact result storage, deterministic merge, and deadline semantics.
- Modify `g1_reach_coverage_viewer.cpp`: consume exhaustive compact search results, start with furniture, draw/cycle truthful options, and regenerate one selected pose.
- Modify `g1_reach_coverage_probe.cpp`: evaluate shared arbitrary grasp fixtures rather than clip-local perturbations.
- Modify `tests/cpp/test_reach_coverage.cpp`: placement and shaping regressions.
- Create `tests/cpp/test_reach_search.cpp`: exhaustive population, bilateral, deterministic, deadline, and compact-storage tests.
- Modify `tests/python/test_g1_reach_coverage_viewer.py`: viewer/probe/build source contracts.
- Modify `Makefile`: compile the placement/search units and their tests with pthread support.

---

### Task 1: Pure Contact-Relative Placement

**Files:**
- Create: `reach_placement.h`
- Create: `reach_placement.cpp`
- Modify: `tests/cpp/test_reach_coverage.cpp`
- Modify: `Makefile:156-178,551-552`

**Interfaces:**
- Consumes: `reach::Pack`, canonical clip index, yaw index, target wrist position, and source database frame.
- Produces: `reach::kYawPlacementCount`, `reach::placement_yaw`, `reach::contact_alignment`, `reach::place_pose`, and `reach::place_direction`.

- [ ] **Step 1: Add failing exact-placement and world-up tests**

Add these assertions to the existing fixture-based coverage test:

```cpp
void test_contact_placement_is_exact_and_upright() {
    const reach::Pack pack = fixture(20U);
    const vec3 target(1.25F, 0.83F, -0.70F);
    for (uint8_t yaw = 0U; yaw < reach::kYawPlacementCount; ++yaw) {
        const interaction::Pose placed = reach::place_pose(
            pack, 0U, yaw, target, 19);
        const interaction::WorldPose world = interaction::world_pose(placed);
        const size_t wrist = static_cast<size_t>(g1_skeleton::LeftWrist);
        assert(length(world.positions[wrist] - target) <= 1.0e-5F);
        const vec3 up = quat_mul_vec3(
            world.rotations[g1_skeleton::Simulation], vec3(0, 1, 0));
        assert(length(up - vec3(0, 1, 0)) <= 1.0e-5F);
    }
}

void test_contact_placement_translates_every_sample_rigidly() {
    const reach::Pack pack = fixture(20U);
    const vec3 first_target(0.4F, 0.9F, -0.2F);
    const vec3 moved_target = first_target + vec3(1.0F, -0.3F, 0.5F);
    for (int32_t frame = 0; frame < 20; ++frame) {
        const interaction::WorldPose first = interaction::world_pose(
            reach::place_pose(pack, 0U, 3U, first_target, frame));
        const interaction::WorldPose moved = interaction::world_pose(
            reach::place_pose(pack, 0U, 3U, moved_target, frame));
        for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
            assert(length(
                (moved.positions[bone] - first.positions[bone]) -
                (moved_target - first_target)) <= 1.0e-5F);
        }
    }
}
```

Include `reach_placement.h` and call both tests from `main`.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
make build/tests/test_reach_coverage && build/tests/test_reach_coverage
```

Expected: compilation fails because `reach_placement.h`, `place_pose`, and `kYawPlacementCount` do not exist.

- [ ] **Step 3: Define the placement API**

Create `reach_placement.h`:

```cpp
#pragma once

#include "reach_motion.h"

#include <cstddef>
#include <cstdint>

namespace reach {

inline constexpr uint8_t kYawPlacementCount = 12U;

float placement_yaw(uint8_t yaw_index);

interaction::Transform contact_alignment(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position);

interaction::Pose place_pose(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position,
    int32_t database_frame);

vec3 place_direction(vec3 direction, uint8_t yaw_index);

}  // namespace reach
```

- [ ] **Step 4: Implement exact rigid placement**

Create `reach_placement.cpp` with input validation and this transform:

```cpp
constexpr float kTwoPi = 6.28318530717958647692F;

float placement_yaw(uint8_t yaw_index) {
    if (yaw_index >= kYawPlacementCount) {
        throw std::out_of_range("reach yaw placement index outside range");
    }
    return kTwoPi * static_cast<float>(yaw_index) /
        static_cast<float>(kYawPlacementCount);
}

interaction::Transform contact_alignment(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position) {
    const quat rotation = quat_from_angle_axis(
        placement_yaw(yaw_index), vec3(0, 1, 0));
    const vec3 endpoint = endpoint_transform(pack.database, clip).position;
    return {
        target_position - quat_mul_vec3(rotation, endpoint),
        rotation,
    };
}

interaction::Pose place_pose(
    const Pack& pack,
    size_t clip,
    uint8_t yaw_index,
    vec3 target_position,
    int32_t database_frame) {
    interaction::Pose pose = pose_at_frame(pack.database, database_frame);
    const interaction::Transform alignment = contact_alignment(
        pack, clip, yaw_index, target_position);
    const size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const interaction::Transform placed_root = interaction::compose(
        alignment, {pose.positions[root], pose.rotations[root]});
    pose.positions[root] = placed_root.position;
    pose.rotations[root] = placed_root.rotation;
    return pose;
}

vec3 place_direction(vec3 direction, uint8_t yaw_index) {
    return quat_mul_vec3(
        quat_from_angle_axis(placement_yaw(yaw_index), vec3(0, 1, 0)),
        direction);
}
```

Reject non-finite target positions and invalid clip/frame/yaw indices before producing a pose.

- [ ] **Step 5: Add placement sources to existing builds**

Add `reach_placement.cpp reach_placement.h` to the viewer, probe, and `test_reach_coverage` dependencies and add `reach_placement.cpp` to each corresponding compile command.

- [ ] **Step 6: Run focused placement tests**

Run:

```bash
make build/tests/test_reach_coverage && build/tests/test_reach_coverage
```

Expected: PASS, including all existing coverage tests.

- [ ] **Step 7: Commit**

```bash
git add reach_placement.h reach_placement.cpp tests/cpp/test_reach_coverage.cpp Makefile
git commit -m "feat: place reach motions from grasp contact"
```

---

### Task 2: Exhaustive Candidate Identity and Placed Shaping

**Files:**
- Modify: `reach_coverage.h`
- Modify: `reach_coverage.cpp`
- Modify: `tests/cpp/test_reach_coverage.cpp`

**Interfaces:**
- Consumes: Task 1 placement functions and the unchanged `interaction::solve_hand_ik`.
- Produces: `reach::enumerate_candidates`, yaw-aware `reach::Candidate`, placed `shape_candidate`, `Evaluation::active_arm_deformation`, and a default 1 mm final position gate.

- [ ] **Step 1: Add failing exhaustive-population tests**

Extend the fixture to create one left and one right clip, then add:

```cpp
void test_enumeration_is_bilateral_exhaustive_and_query_invariant() {
    const reach::Pack pack = bilateral_fixture();
    const auto candidates = reach::enumerate_candidates(pack);
    assert(candidates.size() == 2U * reach::kYawPlacementCount);
    for (size_t clip = 0U; clip < 2U; ++clip) {
        for (uint8_t yaw = 0U; yaw < reach::kYawPlacementCount; ++yaw) {
            const size_t index = clip * reach::kYawPlacementCount + yaw;
            assert(candidates[index].clip == clip);
            assert(candidates[index].yaw_index == yaw);
        }
    }
}

void test_shared_grasp_shapes_both_hands_at_every_yaw() {
    const reach::Pack pack = bilateral_fixture();
    reach::Query query{};
    query.target.position = vec3(1.1F, 0.9F, -0.6F);
    query.target.rotation = quat();
    query.approach_world = normalize(vec3(-1, 0, 0));
    bool saw_left = false;
    bool saw_right = false;
    for (const reach::Candidate& candidate : reach::enumerate_candidates(pack)) {
        query.hand = static_cast<reach::Hand>(
            pack.database.active_hands.at(candidate.clip));
        saw_left = saw_left || query.hand == reach::Hand::Left;
        saw_right = saw_right || query.hand == reach::Hand::Right;
        const reach::Evaluation result = reach::shape_candidate(
            pack, candidate, query);
        assert(!result.poses.empty());
        assert(result.rejection != reach::Rejection::OutsideEnvelope);
        if (result.rejection == reach::Rejection::None) {
            assert(result.position_error_m <= 0.001F);
        }
    }
    assert(saw_left && saw_right);
}
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
make build/tests/test_reach_coverage && build/tests/test_reach_coverage
```

Expected: compilation fails because `enumerate_candidates` and `Candidate::yaw_index` do not exist.

- [ ] **Step 3: Define exhaustive identity and final metrics**

Change the relevant declarations in `reach_coverage.h` to:

```cpp
struct CoverageConfig {
    float maximum_request_position_m = 0.45F;
    float accepted_position_m = 0.001F;
    float accepted_approach_radians = 0.261799388F;
    float accepted_orientation_radians = 1.047197551F;
    size_t maximum_candidates = 8192U;
};

struct Candidate {
    size_t clip = 0U;
    uint8_t yaw_index = 0U;
    float yaw_radians = 0.0F;
    float source_approach_error_radians = 0.0F;
    float cost = 0.0F;
};

struct Evaluation {
    Candidate candidate{};
    std::vector<interaction::Pose> poses;
    Rejection rejection = Rejection::None;
    bool joint_limit_saturated = false;
    bool object_collision_observed = false;
    bool environment_collision_observed = false;
    float position_error_m = 0.0F;
    float approach_error_radians = 0.0F;
    float orientation_error_radians = 0.0F;
    float active_arm_deformation = 0.0F;
    size_t collision_sample = 0U;
};

std::vector<Candidate> enumerate_candidates(const Pack& pack);
```

Keep `select_candidates` as the hand-specific compatibility wrapper used by focused tests; it filters `enumerate_candidates` by `query.hand` but applies no endpoint envelope.

- [ ] **Step 4: Enumerate every clip and yaw deterministically**

Implement:

```cpp
std::vector<Candidate> enumerate_candidates(const Pack& pack) {
    std::vector<Candidate> result;
    result.reserve(
        static_cast<size_t>(pack.database.clip_count) * kYawPlacementCount);
    for (size_t clip = 0U; clip < pack.database.clip_count; ++clip) {
        for (uint8_t yaw = 0U; yaw < kYawPlacementCount; ++yaw) {
            result.push_back({clip, yaw, placement_yaw(yaw), 0.0F, 0.0F});
        }
    }
    return result;
}
```

`select_candidates` must derive the placed source approach with `place_direction`, compute approach error only as a cost, sort by `(cost, clip, yaw_index)`, and never compare source endpoint position with the target.

- [ ] **Step 5: Place each complete pose before IK**

In `shape_candidate`, remove the source-endpoint envelope rejection and create each sample with:

```cpp
interaction::Pose pose = place_pose(
    pack,
    candidate.clip,
    candidate.yaw_index,
    query.target.position,
    start + static_cast<int32_t>(sample));
const interaction::Transform placed_hand = hand_transform(pose, query.hand);
const vec3 relative = placed_hand.position - query.target.position;
const vec3 rotated = quat_mul_vec3(approach_alignment, relative);
const interaction::Transform desired{
    placed_hand.position + approach_weight * (rotated - relative),
    quat_mul(
        quat_nlerp_shortest(quat(), correction, translation_weight),
        placed_hand.rotation),
};
```

Compute `source_approach` with `place_direction`. Compute `correction` from the placed terminal wrist rotation, not the canonical endpoint rotation. At the final sample always set `sample_ik_config.accepted_position_m` to `config.accepted_position_m`; do the same during nonzero approach warping. Do not edit any `interaction_ik` source or metadata file.

- [ ] **Step 6: Measure active-arm deformation**

Before each IK call retain the rigidly placed pose. After the call, measure the shortest quaternion angle for each of the seven active-arm bones, accumulate its square, and set:

```cpp
evaluation.active_arm_deformation =
    deformation_sum /
    static_cast<float>(frame_count * interaction::kLeftArm.size());
```

Use `kLeftArm` or `kRightArm` according to the candidate hand. Reject non-finite deformation as `InvalidSolver`.

- [ ] **Step 7: Replace obsolete local-retarget assertions**

Delete `test_small_translation_without_approach_warp_uses_public_tolerance`, because preserving an unchanged hand one centimetre away contradicts exact contact. Replace it with:

```cpp
void test_final_contact_always_uses_one_millimetre_gate() {
    const reach::Pack pack = fixture();
    reach::Query query = zero_query(pack);
    query.target.position.y += 0.01F;
    const reach::Evaluation result = reach::shape_candidate(
        pack, reach::select_candidates(pack, query)[0], query);
    assert(result.position_error_m <= 0.001F);
}
```

Update aggregate `Candidate` construction in tests and the probe to include or default the yaw index intentionally.

- [ ] **Step 8: Run shaping and IK regressions**

Run:

```bash
make build/tests/test_reach_coverage build/tests/test_interaction_ik
build/tests/test_reach_coverage
build/tests/test_interaction_ik
git diff --exit-code 50f5542 -- interaction_ik.h interaction_ik.cpp g1_arm_joint_metadata.h
```

Expected: both binaries PASS and the final command reports no IK-file differences.

- [ ] **Step 9: Commit**

```bash
git add reach_coverage.h reach_coverage.cpp tests/cpp/test_reach_coverage.cpp
git commit -m "feat: shape every reach from grasp-relative placement"
```

---

### Task 3: Bounded Parallel Exhaustive Search

**Files:**
- Create: `reach_search.h`
- Create: `reach_search.cpp`
- Create: `tests/cpp/test_reach_search.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `reach::enumerate_candidates` and `reach::evaluate_candidate` from Task 2.
- Produces: `reach::ExhaustiveQuery`, `reach::SearchConfig`, `reach::CompactEvaluation`, `reach::SearchResult`, `reach::search_all`, and `reach::regenerate`.

- [ ] **Step 1: Add a failing standalone search test target**

Create `tests/cpp/test_reach_search.cpp` with a two-clip bilateral fixture and these tests:

```cpp
void test_search_processes_every_bilateral_yaw_instance() {
    const reach::Pack pack = bilateral_fixture();
    const reach::SearchResult result = reach::search_all(
        pack,
        {{{1.0F, 0.85F, -0.25F}, quat()}, normalize(vec3(-1, 0, 0))},
        {{{0.94F, 0.85F, -0.25F}, quat()}, vec3(0.10F)},
        interaction::EnvironmentGeometry{},
        reach::SearchConfig{2U, std::chrono::seconds(30)});
    assert(result.complete);
    assert(result.total == 2U * reach::kYawPlacementCount);
    assert(result.processed == result.total);
    assert(result.evaluations.size() == result.total);
    for (const reach::CompactEvaluation& value : result.evaluations) {
        assert(value.evaluation.poses.empty());
        assert(!value.hand_path.empty());
    }
}

void test_search_order_is_independent_of_worker_count() {
    const reach::SearchResult serial = run_fixture_search(1U);
    const reach::SearchResult parallel = run_fixture_search(4U);
    assert(serial.evaluations.size() == parallel.evaluations.size());
    for (size_t i = 0U; i < serial.evaluations.size(); ++i) {
        assert(serial.evaluations[i].evaluation.candidate.clip ==
               parallel.evaluations[i].evaluation.candidate.clip);
        assert(serial.evaluations[i].evaluation.candidate.yaw_index ==
               parallel.evaluations[i].evaluation.candidate.yaw_index);
        assert(serial.evaluations[i].evaluation.rejection ==
               parallel.evaluations[i].evaluation.rejection);
    }
}
```

Add a deadline-zero test that asserts `complete == false`, `processed < total`, and no exhaustive acceptance count is published.

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
make build/tests/test_reach_search
```

Expected: compilation fails because `reach_search.h` and the search types do not exist.

- [ ] **Step 3: Define the compact search contract**

Create `reach_search.h`:

```cpp
#pragma once

#include "reach_coverage.h"

#include <chrono>
#include <cstddef>
#include <vector>

namespace reach {

struct ExhaustiveQuery {
    interaction::Transform target{};
    vec3 approach_world{1, 0, 0};
};

struct SearchConfig {
    size_t worker_count = 4U;
    std::chrono::steady_clock::duration deadline = std::chrono::seconds(30);
    CoverageConfig coverage{};
    interaction::TrajectoryCollisionConfig collision{};
};

struct CompactEvaluation {
    Evaluation evaluation{};
    std::vector<vec3> hand_path;
};

struct SearchResult {
    bool complete = false;
    size_t total = 0U;
    size_t processed = 0U;
    std::chrono::steady_clock::duration elapsed{};
    std::vector<CompactEvaluation> evaluations;
    std::vector<size_t> accepted;
};

SearchResult search_all(
    const Pack& pack,
    const ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config = SearchConfig{});

Evaluation regenerate(
    const Pack& pack,
    const CompactEvaluation& compact,
    const ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const SearchConfig& config = SearchConfig{});

}  // namespace reach
```

- [ ] **Step 4: Implement bounded deterministic parallel search**

In `reach_search.cpp`, validate nonzero bounded worker count, finite query, and nonnegative deadline. Enumerate candidates once. Pre-size one optional result slot per candidate and use an atomic index:

```cpp
std::atomic<size_t> next{0U};
const auto started = std::chrono::steady_clock::now();
const auto expires = started + config.deadline;
auto worker = [&] {
    while (std::chrono::steady_clock::now() < expires) {
        const size_t index = next.fetch_add(1U);
        if (index >= candidates.size()) return;
        const Candidate candidate = candidates[index];
        Query hand_query{};
        hand_query.hand = static_cast<Hand>(
            pack.database.active_hands.at(candidate.clip));
        hand_query.target = query.target;
        hand_query.approach_world = query.approach_world;
        Evaluation evaluation = evaluate_candidate(
            pack, candidate, hand_query, object, environment,
            config.coverage, config.collision);
        CompactEvaluation compact{};
        compact.evaluation = evaluation;
        compact.hand_path = extract_active_wrist_path(
            evaluation.poses, hand_query.hand);
        compact.evaluation.poses.clear();
        slots[index] = std::move(compact);
        processed.fetch_add(1U);
    }
};
```

Join workers, merge slots strictly by candidate index, set `complete` only when every slot is populated, and populate `accepted` only when complete. Sort accepted indices by `(active_arm_deformation, orientation_error_radians, approach_error_radians, clip, yaw_index)`.

- [ ] **Step 5: Implement exact selected regeneration**

`regenerate` reconstructs the candidate hand query, calls `evaluate_candidate`, and verifies rejection plus final errors against the compact record within `1e-5`. Throw `std::runtime_error` if deterministic regeneration disagrees.

- [ ] **Step 6: Add build rules**

Add `reach_search.cpp reach_search.h` and `-pthread` to viewer/probe builds. Add:

```make
CPP_TEST_BINS += $(CPP_TEST_DIR)/test_reach_search

$(CPP_TEST_DIR)/test_reach_search: tests/cpp/test_reach_search.cpp \
  reach_search.cpp reach_search.h reach_coverage.cpp reach_coverage.h \
  reach_placement.cpp reach_placement.h reach_database.cpp reach_motion.cpp \
  interaction_hand_trajectories.cpp interaction_ik.cpp interaction_pose.cpp \
  interaction_target.cpp | $(CPP_TEST_DIR)
	$(CXX) $(CPP_TEST_FLAGS) -pthread tests/cpp/test_reach_search.cpp \
	  reach_search.cpp reach_coverage.cpp reach_placement.cpp \
	  reach_database.cpp reach_motion.cpp interaction_hand_trajectories.cpp \
	  interaction_ik.cpp interaction_pose.cpp interaction_target.cpp -o $@
```

- [ ] **Step 7: Run search, coverage, collision, and IK tests**

Run:

```bash
make build/tests/test_reach_search build/tests/test_reach_coverage \
  build/tests/test_interaction_hand_trajectories build/tests/test_interaction_ik
build/tests/test_reach_search
build/tests/test_reach_coverage
build/tests/test_interaction_hand_trajectories
build/tests/test_interaction_ik
```

Expected: all four binaries PASS.

- [ ] **Step 8: Commit**

```bash
git add reach_search.h reach_search.cpp tests/cpp/test_reach_search.cpp Makefile
git commit -m "feat: evaluate exhaustive grasp placements in parallel"
```

---

### Task 4: Truthful Exhaustive Coverage Viewer

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: `reach::search_all` and `reach::regenerate` from Task 3.
- Produces: one furniture-first flat viewer with explicit exhaustive search, compact background paths, accepted cycling, and selected full-pose playback.

- [ ] **Step 1: Update viewer source-contract tests first**

Replace obsolete expectations with:

```python
for required in (
    "reach::search_all(", "reach::regenerate(",
    "use_coverage_environment = true",
    "SEARCH INCOMPLETE", "RAW INSTANCES", "PROCESSED",
    "YAW PLACEMENT", "CAPTURED LEFT", "MIRRORED RIGHT",
    "displayed", "accepted", "KEY_ENTER", "KEY_G",
):
    self.assertIn(required, source)
for forbidden in (
    "reach::select_candidates(",
    "use_coverage_environment = false",
    "g1_mesh_renderer", "terrain_runtime", "TakeScreenshot(",
):
    self.assertNotIn(forbidden, source)
self.assertIn("std::optional<reach::SearchResult>", source)
self.assertIn("results = std::move(completed)", source)
```

Retain existing flat-skeleton, object-control, stale-state, collision-label, and no-controller assertions.

- [ ] **Step 2: Run the viewer test to verify RED**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
```

Expected: FAIL because the viewer still performs hand-specific local search and starts in open space.

- [ ] **Step 3: Replace viewer-local search records**

Replace `SearchResults` with the library `SearchResult` plus view state. Store:

```cpp
std::optional<reach::SearchResult> results;
std::optional<reach::Evaluation> selected_full;
size_t selected = 0U;
bool search_incomplete = false;
```

Build one `reach::ExhaustiveQuery` from the object-derived grasp. Do not carry a reset hand into retrieval; the candidate hand comes from clip metadata.

- [ ] **Step 4: Make explicit search atomic**

On Enter:

```cpp
const reach::SearchResult completed = reach::search_all(
    pack,
    exhaustive_query,
    {object, options.object_size},
    use_coverage_environment ? coverage : open,
    search_config);
if (completed.complete) {
    results = completed;
    selected = 0U;
    selected_full.reset();
    if (!results->accepted.empty()) {
        selected_full = reach::regenerate(
            pack,
            results->evaluations[results->accepted[0]],
            exhaustive_query,
            {object, options.object_size},
            use_coverage_environment ? coverage : open,
            search_config);
    }
    stale = false;
    search_incomplete = false;
} else {
    search_incomplete = true;
}
```

Do not replace the previous complete result on incomplete search. Moving or rotating the object only marks stale.

- [ ] **Step 5: Cycle accepted options and regenerate one full pose**

Bracket/slash controls index `results->accepted`. Each selection calls `regenerate` once and resets animation time. The selected skeleton and emphasized path come from the regenerated `Evaluation`; background paths come from `CompactEvaluation::hand_path` and never require retained full poses.

- [ ] **Step 6: Start with furniture and report exhaustive truth**

Set `use_coverage_environment = true`, initialize the raylib window before any
search, and leave `results` empty until the user presses Enter. The first frame
must therefore show the coverage boxes immediately with `PRESS ENTER TO SEARCH`
instead of blocking before a window exists. Draw every accepted compact hand
path at stride five; `displayed_paths` equals `results->accepted.size()` and no
accepted path is silently capped. Draw compact rejected paths at stride ten
only while the diagnostic toggle is active. Extend the HUD with:

```cpp
TextFormat("RAW INSTANCES %zu | PROCESSED %zu | ACCEPTED %zu",
    results->total, results->processed, results->accepted.size())
TextFormat("YAW PLACEMENT %u/12 | %s",
    candidate.yaw_index + 1U, provenance(pack, candidate.clip))
TextFormat("displayed %zu / accepted %zu",
    displayed_paths, results->accepted.size())
```

Show `SEARCH INCOMPLETE - previous complete results retained` when the deadline is hit. Keep rejected paths behind the diagnostic toggle with orange/red styling and explicit rejection labels.

- [ ] **Step 7: Run source tests and warning-clean builds**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -pthread -I. \
  -I.deps/raylib/src -fsyntax-only g1_reach_coverage_viewer.cpp
make g1_reach_coverage_viewer
```

Expected: Python tests PASS, syntax check emits no diagnostics, and release viewer builds.

- [ ] **Step 8: Commit**

```bash
git add g1_reach_coverage_viewer.cpp tests/python/test_g1_reach_coverage_viewer.py
git commit -m "feat: visualize exhaustive grasp-relative reach coverage"
```

---

### Task 5: Shared-Grasp Coverage Evidence and Real-Pack Gate

**Files:**
- Modify: `g1_reach_coverage_probe.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: the same `reach::search_all` service as the viewer and `build/g1-reaches/reach-pack-v2`.
- Produces: `build/g1-reaches/contact-anchored-coverage-report-v3.json` with per-fixture exhaustive counts, errors, diversity, timing, and completion status.

- [ ] **Step 1: Write failing probe-contract assertions**

Change the probe test to require:

```python
for required in (
    "reach::search_all(", "shared_grasps", "open_space", "table",
    "shelf", "below_table", "lower_table", "raw_instances",
    "processed_instances", "root_azimuth_sectors", "elapsed_seconds",
    "complete", "4608", "0.001F", "30",
):
    self.assertIn(required, source)
for forbidden in ("own_query(", "position_perturbations"):
    self.assertNotIn(forbidden, source)
```

- [ ] **Step 2: Run the probe test to verify RED**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
```

Expected: FAIL because the probe still perturbs every clip around its own endpoint.

- [ ] **Step 3: Define shared grasp fixtures**

Use the viewer's table transform and dimensions and define:

```cpp
const interaction::Transform table_world{vec3(0.0F, 0.74F, 0.0F), quat()};
const vec3 table_dimensions(1.20F, 0.06F, 0.75F);
const interaction::EnvironmentGeometry coverage =
    interaction::make_coverage_environment(table_world, table_dimensions);
const std::array<Fixture, 5U> shared_grasps{{
    {"open_space", {{0.75F, 0.90F, 0.00F}, quat()}, {}},
    {"table", {{0.00F, 0.82F, 0.00F}, quat()}, coverage},
    {"shelf", {{0.38F, 1.12F, 0.00F}, quat()}, coverage},
    {"below_table", {{0.00F, 0.48F, 0.00F}, quat()}, coverage},
    {"lower_table", {{-0.93F, 0.58F, 0.00F}, quat()}, coverage},
}};
```

For each fixture, offset the object center from the wrist target by the existing 4 cm contact proxy, and use a fixed normalized approach axis. The open-space fixture uses an empty environment; the others use coverage geometry.

- [ ] **Step 4: Replace clip-local probing with exhaustive shared searches**

For each fixture call `search_all` once with a 30-second deadline. Report:

```json
{
  "raw_instances": 4608,
  "processed_instances": 4608,
  "complete": true,
  "accepted": 0,
  "hands": {"left": 0, "right": 0},
  "root_azimuth_sectors": 0,
  "maximum_accepted_position_m": 0.0,
  "maximum_accepted_approach_radians": 0.0,
  "maximum_accepted_orientation_radians": 0.0,
  "rejections": {},
  "observed_collisions": {"object": 0, "environment": 0},
  "elapsed_seconds": 0.0
}
```

Derive root azimuth by regenerating the best accepted option in each occupied yaw index and measuring terminal grasp-to-root direction. Exit nonzero if any fixture is incomplete, has a raw count other than 4,608, or reports an accepted position over `0.001F`.

- [ ] **Step 5: Run focused and complete automated gates**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_reach_search build/tests/test_reach_coverage \
  build/tests/test_reach_database build/tests/test_interaction_hand_trajectories \
  build/tests/test_interaction_ik g1_reach_coverage_probe \
  g1_reach_coverage_viewer
build/tests/test_reach_search
build/tests/test_reach_coverage
build/tests/test_reach_database
build/tests/test_interaction_hand_trajectories
build/tests/test_interaction_ik
```

Expected: all tests and release builds PASS.

- [ ] **Step 6: Run the real shared-grasp evidence gate**

Run:

```bash
/usr/bin/time -f 'elapsed=%e rss_kb=%M' ./g1_reach_coverage_probe \
  build/g1-reaches/reach-pack-v2 \
  --json build/g1-reaches/contact-anchored-coverage-report-v3.json
jq '.shared_grasps | map_values({complete, raw_instances, processed_instances, accepted, root_azimuth_sectors, maximum_accepted_position_m, elapsed_seconds})' \
  build/g1-reaches/contact-anchored-coverage-report-v3.json
```

Expected: all five fixtures report `complete: true`, `raw_instances: 4608`, `processed_instances: 4608`, maximum accepted position at most `0.001`, and each fixture finishes within 30 seconds. Open space must contain both hands and more than one occupied root-azimuth sector.

- [ ] **Step 7: Record the durable task ledger**

Append a `Contact-Anchored Exhaustive Grasp Motion Matching` section to `.superpowers/sdd/progress.md` listing Task 1 through Task 5 commits, exact test commands, real report path, per-fixture completion, and the unchanged-IK diff check.

- [ ] **Step 8: Commit**

```bash
git add g1_reach_coverage_probe.cpp tests/python/test_g1_reach_coverage_viewer.py \
  .superpowers/sdd/progress.md
git commit -m "test: measure exhaustive shared-grasp reach coverage"
```

---

### Task 6: Final Review and Single-Viewer Handoff

**Files:**
- Verify only; modify files only for reviewer-confirmed Critical or Important findings.

**Interfaces:**
- Consumes: Tasks 1-5 and the real v3 coverage report.
- Produces: reviewed branch state and exactly one running corrected viewer.

- [ ] **Step 1: Run the final unchanged-IK check**

```bash
git diff --exit-code 50f5542 -- interaction_ik.h interaction_ik.cpp g1_arm_joint_metadata.h
```

Expected: no output and exit code zero.

- [ ] **Step 2: Request whole-feature code review**

Generate a review package from `50f5542` through `HEAD`. Ask the reviewer to verify coordinate-space correctness, exact endpoint anchoring, 4,608 exhaustive identity, thread safety, deterministic merge, deadline behavior, compact memory, collision completeness, viewer truthfulness, and the unchanged IK files. Resolve every Critical or Important finding with a failing test before a fix.

- [ ] **Step 3: Repeat the complete verification gate after review fixes**

Run Task 5 Steps 5 and 6 again. Expected: all automated tests/builds pass and the real shared-grasp report remains complete and within bounds.

- [ ] **Step 4: Replace only the flat viewer process**

Resolve the exact existing PID with:

```bash
pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'
```

Terminate only that resolved PID, verify no matching process remains, then launch exactly one:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v2
```

Do not launch a mesh viewer, terrain viewer, controller, or screenshot process.

- [ ] **Step 5: Verify the live process and bounded memory**

```bash
pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'
ps -o pid,rss,etime,cmd -p "$(pgrep -n -f '(^|/)g1_reach_coverage_viewer( |$)')"
```

Expected: exactly one matching viewer, coverage geometry visible at startup, and no unbounded RSS growth while cycling options.

- [ ] **Step 6: Commit review fixes if any**

If review required code changes, commit only the reviewed fixes and their tests with:

```bash
git commit -m "fix: address exhaustive grasp coverage review"
```

If there were no fixes, do not create an empty commit.
