# Reusable Reach Warp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover every distinct outbound reach and make each clip reusable across multiple requested approach directions and wrist orientations while retaining strict final-pose and collision diagnostics.

**Architecture:** Replace whole-file neutral-run segmentation with prominent endpoint peaks, then keep retrieval spatial and hand-specific only. Shape every candidate by smoothly translating its endpoint, rotating its terminal hand path onto the requested approach axis, and applying requested wrist orientation through the existing seven-joint arm IK. Collision evaluation remains full-body except for the active wrist contact chain during the terminal five-frame object-contact window, and reports observed collisions separately from the exclusive acceptance reason.

**Tech Stack:** Python 3, NumPy, existing G1 corpus builders, C++17, raylib, existing damped-least-squares arm IK, `unittest`, Make.

## Global Constraints

- Reach identity is independent of recorded wrist orientation and recorded approach direction.
- Endpoint position and active hand are the only hard retrieval filters; the spatial envelope remains `0.45 m`.
- Root, waist, legs, and inactive arm remain unchanged.
- Final gates remain `4 cm` position, `15 degrees` approach axis, and `60 degrees` full wrist orientation.
- The active wrist contact chain is exempt from target-object collision only during the last five frames; no body part is exempt from environment collision.
- Do not use mesh, terrain, screenshots, locomotion stitching, diffusion, waist IK, or full-body IK.
- Keep one flat viewer process at most and preserve the current pack until the rebuilt pack passes all checks.

---

### Task 1: Recover Prominent Reach Endpoints

**Files:**
- Modify: `resources/g1_reach_builder/segmentation.py`
- Modify: `tests/python/test_g1_reach_segmentation.py`

**Interfaces:**
- Consumes: root-relative wrist traces `(T, 3)`, source frames, and `SegmentationConfig`.
- Produces: `prominent_endpoint_frames(distance, config) -> list[int]` and one `ReachProposal` per distinct endpoint.

- [ ] **Step 1: Add a failing long-run multi-peak test**

```python
def drifting_neutral_trace() -> np.ndarray:
    keyframes = np.array([
        [0.12, 0.00, 0.00], [0.42, 0.00, 0.00],
        [0.14, 0.02, 0.00], [0.14, 0.39, 0.00],
        [0.15, 0.04, 0.00], [-0.24, 0.04, 0.00],
        [0.13, 0.02, 0.00],
    ])
    frames = [0, 30, 60, 90, 120, 150, 180]
    trace = np.empty((181, 3), np.float64)
    for left in range(6):
        trace[frames[left]:frames[left + 1] + 1] = np.linspace(
            keyframes[left], keyframes[left + 1], 31
        )
    return trace

def test_splits_prominent_endpoints_inside_one_global_outside_run(self):
    trace = drifting_neutral_trace()
    proposals = propose_wrist_trace(
        trace, np.arange(len(trace)), "pickup_north_2"
    )
    self.assertEqual(len(proposals), 3)
    np.testing.assert_allclose(
        [trace[p.grab_frame] for p in proposals],
        [trace[30], trace[90], trace[150]], atol=0.02,
    )
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_segmentation -v
```

Expected: FAIL because the current implementation emits one maximum for the complete outside-neutral run.

- [ ] **Step 3: Implement pure-NumPy prominent endpoint selection**

Add `minimum_peak_prominence_m = 0.08` to `SegmentationConfig` and implement:

```python
def prominent_endpoint_frames(distance, config):
    window = int(round(config.maximum_outbound_s * config.fps))
    candidates = np.flatnonzero(
        (distance[1:-1] >= distance[:-2]) &
        (distance[1:-1] > distance[2:]) &
        (distance[1:-1] >= config.minimum_excursion_m)
    ) + 1
    prominent = []
    for peak in candidates.tolist():
        left = float(np.min(distance[max(0, peak - window):peak + 1]))
        right = float(np.min(distance[peak:min(len(distance), peak + window + 1)]))
        if distance[peak] - max(left, right) >= config.minimum_peak_prominence_m:
            prominent.append(peak)
    return _merge_close_peaks(prominent, distance, config)
```

For each peak, choose the minimum-distance departure in the preceding `3.6 s`, then move grab backward by at most 25 frames to the nearest frame whose preceding five-frame displacement is at least `1 cm`. Preserve source ordering and source-frame-based IDs.

- [ ] **Step 4: Add threshold and ordering tests**

Assert sub-`0.08 m` ripples are ignored, peaks less than `0.60 s` apart retain the larger endpoint, and the existing two-reach fixture still emits two proposals.

- [ ] **Step 5: Run all segmentation tests**

Run: `python3 -m unittest tests.python.test_g1_reach_segmentation -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add resources/g1_reach_builder/segmentation.py tests/python/test_g1_reach_segmentation.py
git commit -m "fix: recover repeated outbound reach peaks"
```

### Task 2: Warp One Reach Across Directions and Orientations

**Files:**
- Modify: `reach_coverage.cpp`
- Modify: `tests/cpp/test_reach_coverage.cpp`

**Interfaces:**
- Consumes: `reach::Pack`, `reach::Query`, `reach::Candidate`, and active-arm IK.
- Produces: unchanged spatial candidate eligibility and terminal approach-position warping inside `shape_candidate`.

- [ ] **Step 1: Add failing retrieval-invariance tests**

```cpp
const auto baseline = clip_set(reach::select_candidates(pack, query));
query.target.rotation = quat_mul(
    quat_from_angle_axis(1.570796327F, vec3(0, 1, 0)),
    query.target.rotation);
assert(clip_set(reach::select_candidates(pack, query)) == baseline);
query.approach_world = normalize(vec3(-0.3F, 0.7F, 0.2F));
assert(clip_set(reach::select_candidates(pack, query)) == baseline);
```

Candidate order may change with approach warp cost; the eligible clip set may not.

- [ ] **Step 2: Add a failing approach-warp test**

Request a feasible rotated approach at the fixture endpoint. Assert final five-frame displacement follows the requested direction within `15 degrees`, while root and inactive-arm local transforms remain bit-identical.

- [ ] **Step 3: Run the C++ test to verify RED**

Run: `make build/tests/test_reach_coverage && build/tests/test_reach_coverage`

Expected: FAIL because the current shaper translates but does not rotate terminal travel.

- [ ] **Step 4: Implement robust direction alignment**

```cpp
quat direction_alignment(vec3 source, vec3 target) {
    source = normalize(source);
    target = normalize(target);
    const float cosine = std::clamp(dot(source, target), -1.0F, 1.0F);
    if (cosine > 1.0F - 1.0e-6F) return quat();
    if (cosine < -1.0F + 1.0e-6F) {
        const vec3 basis = std::abs(source.x) < 0.8F
            ? vec3(1, 0, 0) : vec3(0, 1, 0);
        return quat_from_angle_axis(kPi, normalize(cross(source, basis)));
    }
    return quat_between(source, target);
}
```

- [ ] **Step 5: Warp terminal endpoint-relative positions before IK**

```cpp
const size_t aligned_sample = frame_count > 6U ? frame_count - 6U : 0U;
const size_t ramp_start = aligned_sample > 10U ? aligned_sample - 10U : 0U;
const float approach_u = sample <= ramp_start ? 0.0F
    : (sample >= aligned_sample ? 1.0F
       : static_cast<float>(sample - ramp_start) /
         static_cast<float>(aligned_sample - ramp_start));
const float approach_weight = smoothstep(approach_u);
const vec3 relative = source_hand.position - source_endpoint.position;
const vec3 rotated = quat_mul_vec3(alignment, relative);
const vec3 desired_position = source_hand.position
    + translation_weight * endpoint_offset
    + approach_weight * (rotated - relative);
```

Retain shortest endpoint-orientation correction and active-arm-only IK. Never add a wrist-orientation or approach-direction eligibility filter.

- [ ] **Step 6: Run coverage and IK tests**

```bash
make build/tests/test_reach_coverage build/tests/test_interaction_ik
build/tests/test_reach_coverage
build/tests/test_interaction_ik
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add reach_coverage.cpp tests/cpp/test_reach_coverage.cpp
git commit -m "feat: warp reaches across requested approaches"
```

### Task 3: Correct Contact Collision Semantics and Diagnostics

**Files:**
- Modify: `interaction_hand_trajectories.h`
- Modify: `interaction_hand_trajectories.cpp`
- Modify: `reach_coverage.h`
- Modify: `reach_coverage.cpp`
- Modify: `g1_reach_coverage_probe.cpp`
- Modify: `tests/cpp/test_interaction_hand_trajectories.cpp`
- Modify: `tests/cpp/test_reach_coverage.cpp`

**Interfaces:**
- Produces: `TrajectoryCollisionConfig::active_object_contact_window_samples`, `Evaluation::object_collision_observed`, and `Evaluation::environment_collision_observed`.

- [ ] **Step 1: Add failing terminal-contact tests**

Construct a final pose whose active wrist chain touches the target box. Assert a five-frame contact window permits wrist-roll/wrist-pitch/wrist contact while elbow or elbow-to-wrist-roll intersection still reports `ObjectCollision`.

- [ ] **Step 2: Add failing observed-collision tests**

```cpp
assert(result.rejection == reach::Rejection::ApproachAxisError);
assert(!result.object_collision_observed);
assert(result.environment_collision_observed);
```

Also assert an empty environment never sets `environment_collision_observed`.

- [ ] **Step 3: Run tests to verify RED**

```bash
make build/tests/test_interaction_hand_trajectories build/tests/test_reach_coverage
build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
```

Expected: compilation FAIL because the new fields do not exist.

- [ ] **Step 4: Add bounded contact-window semantics**

Add `size_t active_object_contact_window_samples = 1U` to `TrajectoryCollisionConfig`. Skip active wrist-roll, wrist-pitch, and wrist spheres during terminal target contact. Skip a capsule only when both endpoints are in that wrist chain, so elbow-to-wrist-roll remains checked. Never apply an exemption to environment boxes. `reach::evaluate_candidate` uses five samples.

- [ ] **Step 5: Evaluate collision observations for every finite shape**

Remove the post-shape early return. Whenever poses exist, run collision evaluation, set both observed booleans, and assign collision as the exclusive reason only when no earlier kinematic rejection exists.
Extend the probe's `Report` with `object_collision_observed` and
`environment_collision_observed`, increment them in `record`, merge them across
workers, and serialize them under `observed_collisions`:

```cpp
output << ",\"observed_collisions\":{\"environment\":"
       << report.environment_collision_observed
       << ",\"object\":" << report.object_collision_observed << '}';
```

- [ ] **Step 6: Run collision and IK regressions**

```bash
make build/tests/test_interaction_hand_trajectories build/tests/test_reach_coverage build/tests/test_interaction_ik
build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_interaction_ik
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_hand_trajectories.h interaction_hand_trajectories.cpp reach_coverage.h reach_coverage.cpp g1_reach_coverage_probe.cpp tests/cpp/test_interaction_hand_trajectories.cpp tests/cpp/test_reach_coverage.cpp
git commit -m "fix: distinguish terminal grasp contact from collision"
```

### Task 4: Make Viewer State Unambiguous

**Files:**
- Modify: `g1_reach_coverage_viewer.cpp`
- Modify: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: Task 3 rejection and observed-collision fields.
- Produces: accepted-only lime, rejected orange, and observed collision HUD labels.

- [ ] **Step 1: Add failing viewer assertions**

```python
self.assertIn(
    "evaluation.rejection == reach::Rejection::None ? LIME : ORANGE",
    source,
)
self.assertIn("object_collision_observed", source)
self.assertIn("environment_collision_observed", source)
self.assertIn("OBSERVED OBJECT COLLISION", source)
self.assertIn("OBSERVED ENVIRONMENT COLLISION", source)
```

- [ ] **Step 2: Run to verify RED**

Run: `python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v`

Expected: FAIL because selected rejected paths are currently lime.

- [ ] **Step 3: Render truthful selected state**

```cpp
const Color selected_color = evaluation.rejection == reach::Rejection::None
    ? LIME : ORANGE;
draw_path(evaluation, hand, selected_color, 1U, true);
```

Display `SAFE` only for `Rejection::None`; otherwise display first rejection and observed collision flags. Preserve the older viewer style, complete rejected animation, explicit Enter search, and stale state.

- [ ] **Step 4: Test and compile with warnings as errors**

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. -I.deps/raylib/src -fsyntax-only g1_reach_coverage_viewer.cpp
make g1_reach_coverage_viewer
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add g1_reach_coverage_viewer.cpp tests/python/test_g1_reach_coverage_viewer.py
git commit -m "fix: show truthful reach acceptance diagnostics"
```

### Task 5: Rebuild and Measure the Complete Bilateral Corpus

**Files:**
- Create: `resources/confirm_g1_reach_defaults.py`
- Create: `tests/python/test_confirm_g1_reach_defaults.py`
- Generate: `build/g1-reaches/review-v2/`
- Generate: `build/g1-reaches/annotations-v2.json`
- Generate: `build/g1-reaches/reach-pack-v2/`
- Generate: `build/g1-reaches/coverage-report-v2.json`

**Interfaces:**
- Consumes: recovered proposals and the user's confirmation that source recordings are valid reaches.
- Produces: deterministic annotations, bilateral v2 pack, per-source counts, and coverage evidence.

- [ ] **Step 1: Add a failing confirmation-CLI test**

```python
result = confirm_main([
    "--review", str(review), "--output", str(annotations),
])
self.assertEqual(result, 0)
document = load_annotations(annotations, review)
self.assertEqual(
    [a.status for a in document.annotations],
    ["accepted", "accepted", "rejected"],
)
```

Assert a second run is byte-identical and rejected notes contain the structural reason.

- [ ] **Step 2: Run to verify RED**

Run: `python3 -m unittest tests.python.test_confirm_g1_reach_defaults -v`

Expected: import FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic confirmation**

Create pending annotations, attempt `update_annotation(..., status="accepted")` and `build_captured_reach` for every proposal, accept successful defaults, and reject structural failures. Print total and per-source counts, write atomically, and load the completed document to verify it.

- [ ] **Step 4: Run Python builder tests**

```bash
python3 -m unittest tests.python.test_confirm_g1_reach_defaults tests.python.test_g1_reach_annotations tests.python.test_g1_reach_motions tests.python.test_g1_reach_mirror tests.python.test_g1_reach_artifacts tests.python.test_g1_reach_build_cli -v
```

Expected: PASS.

- [ ] **Step 5: Build v2 beside the baseline**

```bash
python3 -m resources.prepare_g1_reach_review --archive /home/ubuntu/Downloads/g1_retargeted_motions.zip --g1-xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml --output build/g1-reaches/review-v2
python3 -m resources.confirm_g1_reach_defaults --review build/g1-reaches/review-v2 --output build/g1-reaches/annotations-v2.json
python3 -m resources.build_g1_reach_database --review build/g1-reaches/review-v2 --annotations build/g1-reaches/annotations-v2.json --output build/g1-reaches/reach-pack-v2
```

Expected: all eight sources appear, no annotations are pending, mirrored count equals captured count, and captured count is at least 140.

- [ ] **Step 6: Run the real probe**

```bash
./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 --json build/g1-reaches/coverage-report-v2.json
```

Expected: zero-retarget has zero failures and JSON contains per-hand, source, height, direction, augmentation, union, rejection, and observed-collision counts.

- [ ] **Step 7: Run focused full verification**

```bash
python3 -m unittest tests.python.test_g1_reach_sources tests.python.test_g1_reach_review tests.python.test_g1_reach_segmentation tests.python.test_g1_reach_annotations tests.python.test_g1_reach_motions tests.python.test_g1_reach_mirror tests.python.test_g1_reach_artifacts tests.python.test_g1_reach_build_cli tests.python.test_confirm_g1_reach_defaults tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_reach_database build/tests/test_reach_coverage build/tests/test_interaction_hand_trajectories build/tests/test_interaction_ik
build/tests/test_reach_database build/tests/python-reach-pack
build/tests/test_reach_coverage
build/tests/test_interaction_hand_trajectories
build/tests/test_interaction_ik
```

Expected: PASS.

- [ ] **Step 8: Commit source changes**

```bash
git add resources/confirm_g1_reach_defaults.py tests/python/test_confirm_g1_reach_defaults.py
git commit -m "feat: confirm recovered reach corpus deterministically"
```

- [ ] **Step 9: Replace only the live flat viewer after verification**

Terminate only the existing `g1_reach_coverage_viewer`, then run:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer build/g1-reaches/reach-pack-v2
```

Verify Enter recomputes, `/` cycles complete motions, accepted is lime, rejected is orange, open space has no environment collision, and one clip remains eligible across several directions and wrist orientations.
