# Swept Grasp/Object Collision Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject reach candidates whose active grasping wrist chain enters or tunnels through the oriented object before the final grasp contact.

**Architecture:** Extend the existing shaped-trajectory collision evaluator with an analytical swept-sphere versus oriented-box interval test. Keep the final active wrist chain contact exemption, but reduce it to the final sample and permit a terminal sweep intersection only when first contact occurs at the terminal endpoint.

**Tech Stack:** C++17, existing interaction trajectory/collision types, Make, Python `unittest`.

## Global Constraints

- Use the queried oriented object box, including its full position and rotation.
- Permit active wrist-chain object contact only at the final grasp sample.
- Reject sampled penetration and between-sample tunneling.
- Preserve existing elbow, forearm, inactive-hand, body, and furniture collision checks.
- Preserve existing rejection categories, IK, placement, ranking, and rendering.
- Use a small explicit endpoint epsilon; do not restore a multi-sample contact window.

---

## File Structure

- Modify `interaction_hand_trajectories.cpp`: return slab entry/exit intervals and apply active wrist-chain temporal sweeps during shaped collision validation.
- Modify `interaction_hand_trajectories.h`: remove the obsolete configurable multi-sample contact window.
- Modify `reach_coverage.cpp`: remove the five-sample active-wrist contact exemption.
- Modify `tests/cpp/test_interaction_hand_trajectories.cpp`: add multi-sample fixtures and focused penetration, tunneling, terminal-contact, and rotation regressions.
- Modify `tests/python/test_g1_reach_coverage_viewer.py`: lock the exhaustive reach path to final-contact-only collision semantics.
- Modify `.superpowers/sdd/progress.md`: record test and real-pack evidence after implementation.

### Task 1: Continuous Active-Hand/Object Collision

**Files:**
- Modify: `interaction_hand_trajectories.cpp`
- Modify: `interaction_hand_trajectories.h`
- Test: `tests/cpp/test_interaction_hand_trajectories.cpp`

**Interfaces:**
- Consumes: `WorldPose`, `OrientedBox`, `Hand`, `TrajectoryCollisionConfig`, and `evaluate_shaped_trajectory_feasibility(...)`.
- Produces: private `SegmentBoxInterval segment_box_interval(vec3, vec3, float, const OrientedBox&)` and final-contact-only swept wrist validation.

- [ ] **Step 1: Add a multi-sample shaped-pose test fixture**

Add a helper that preserves distinct world positions at each sample:

```cpp
interaction::ShapedHandTrajectory shaped_trajectory_with_world_positions(
    const std::vector<std::array<vec3, g1_skeleton::BoneCount>>& samples,
    interaction::Hand hand) {
    interaction::ShapedHandTrajectory shaped{};
    for (const auto& positions : samples) {
        shaped.poses.push_back(pose_from_world_positions(positions));
        const interaction::WorldPose world =
            interaction::world_pose(shaped.poses.back());
        const size_t wrist = hand == interaction::Hand::Left
            ? g1_skeleton::LeftWrist : g1_skeleton::RightWrist;
        const size_t elbow = hand == interaction::Hand::Left
            ? g1_skeleton::LeftElbow : g1_skeleton::RightElbow;
        shaped.path.hands.push_back(
            {world.positions[wrist], world.rotations[wrist]});
        shaped.path.elbows.push_back(world.positions[elbow]);
    }
    shaped.contact_accepted = true;
    return shaped;
}
```

Keep `shaped_pose_with_world_positions(...)` as a wrapper around this helper so existing tests remain readable.

- [ ] **Step 2: Write failing collision tests**

Add focused tests around a `0.20 m` oriented object:

```cpp
void test_swept_active_wrist_rejects_clear_endpoints_crossing_object();
void test_terminal_sweep_allows_only_endpoint_contact();
void test_swept_active_wrist_respects_object_rotation();
void test_only_final_sample_exempts_active_wrist_chain();
```

The fixtures must prove these exact outcomes:

- wrist centers at `x=-0.30` and `x=+0.30` are individually clear but their sweep through an identity object is `ObjectCollision`;
- a same-side sweep from `x=+0.30` to the expanded-box contact boundary `x=+0.14` is accepted at the final sample;
- an opposite-side sweep from `x=-0.30` to that same `x=+0.14` final target is rejected;
- after a 90-degree object yaw, the equivalent crossing and same-side contact tests operate on world Z rather than world X;
- the returned collision sample is the destination sample of the first forbidden sweep.
- five repeated penetrating wrist poses are rejected at sample zero even when
  contact is sample four, while a clear prefix followed by one final contact
  pose remains accepted.

Replace the existing
`test_terminal_contact_window_exempts_only_the_active_wrist_chain` regression,
which intentionally permits five contact samples, and register all new tests
in `main()`.

- [ ] **Step 3: Run the focused binary to verify RED**

Run:

```bash
make build/tests/test_interaction_hand_trajectories
build/tests/test_interaction_hand_trajectories
```

Expected: FAIL at the clear-endpoint tunneling or opposite-side terminal-sweep assertion because collision validation currently checks only sampled poses.

- [ ] **Step 4: Return an analytical oriented-box intersection interval**

Replace the boolean-only slab implementation used by `capsule_intersects_box`
with a private interval:

```cpp
struct SegmentBoxInterval {
    bool intersects = false;
    float entry = 0.0F;
    float exit = 0.0F;
};

SegmentBoxInterval segment_box_interval(
    vec3 start,
    vec3 stop,
    float radius,
    const OrientedBox& box);
```

Transform both endpoints with `point_in_box`, expand each half extent by
`radius`, initialize the slab range to `[0, 1]`, and return the clipped entry
and exit parameters. Preserve `capsule_intersects_box(...)` as a boolean
wrapper over `.intersects` so all existing spatial skeleton checks retain
their behavior.

- [ ] **Step 5: Apply temporal sweeps to the active wrist chain**

Remove `active_object_contact_window_samples` from
`TrajectoryCollisionConfig`; final-only contact is an invariant rather than a
caller-adjustable relaxation.

In `evaluate_shaped_trajectory_feasibility(...)`, cache each sample's
`WorldPose`. For samples after zero, sweep the active
`WristRoll`, `WristPitch`, and `Wrist` joint centers from the previous pose to
the current pose using each joint's existing `joint_radius(...)`.

Reject an intersecting sweep unless both conditions hold:

```cpp
const bool terminal_sweep = sample == contact_point;
const bool endpoint_only_contact =
    interval.entry >= 1.0F - 1.0e-5F;
```

Only `terminal_sweep && endpoint_only_contact` is permitted. Set
`object_collision_observed=true`, `reason=ObjectCollision`, and
`sample=sample` for the first forbidden sweep. Continue existing pose and
environment checks so diagnostics still observe later collisions.

For the existing per-pose skeleton check, define terminal contact only as:

```cpp
const bool terminal_contact = sample == contact_point;
```

- [ ] **Step 6: Run focused tests to verify GREEN**

Run:

```bash
make build/tests/test_interaction_hand_trajectories
build/tests/test_interaction_hand_trajectories
```

Expected: PASS with no output.

- [ ] **Step 7: Commit**

```bash
git add interaction_hand_trajectories.cpp interaction_hand_trajectories.h \
  tests/cpp/test_interaction_hand_trajectories.cpp
git commit -m "feat: reject swept grasp penetration"
```

### Task 2: Enforce Final-Contact Semantics in Exhaustive Reach Search

**Files:**
- Modify: `reach_coverage.cpp`
- Test: `tests/python/test_g1_reach_coverage_viewer.py`

**Interfaces:**
- Consumes: `reach::evaluate_candidate(...)` and `TrajectoryCollisionConfig`.
- Produces: exhaustive evaluations that use the invariant final-sample contact policy and report swept failures as `Rejection::ObjectCollision`.

- [ ] **Step 1: Write a failing source/behavior regression**

Add `REACH_COVERAGE = ROOT / "reach_coverage.cpp"` and this source contract:

```python
def test_reach_collision_uses_final_contact_only(self):
    source = self.source(REACH_COVERAGE)
    self.assertNotIn(
        "active_object_contact_window_samples = 5U", source
    )
```

- [ ] **Step 2: Run the regression to verify RED**

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_coverage_viewer.G1ReachCoverageViewerTests.test_reach_collision_uses_final_contact_only \
  -v
```

Expected: FAIL because `reach_coverage.cpp` currently sets the window to five
samples.

- [ ] **Step 3: Remove the widened contact window**

Delete the now-invalid override:

```cpp
reach_collision_config.active_object_contact_window_samples = 5U;
```

Pass the caller's remaining radius configuration unchanged to
`evaluate_shaped_trajectory_feasibility(...)`.

- [ ] **Step 4: Run focused and exhaustive tests**

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_interaction_hand_trajectories \
  build/tests/test_reach_coverage build/tests/test_reach_search
build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_reach_search
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add reach_coverage.cpp \
  tests/python/test_g1_reach_coverage_viewer.py
git commit -m "fix: allow object contact only at final grasp"
```

### Task 3: Real-Pack Evidence and Viewer Handoff

**Files:**
- Modify: `.superpowers/sdd/progress.md`
- Generate: `build/g1-reaches/contact-anchored-coverage-report-v4.json` (do not commit)

**Interfaces:**
- Consumes: Tasks 1-2, `build/g1-reaches/reach-pack-v2`, and the existing mesh viewer.
- Produces: reviewed collision-filter commits, real rejection evidence, and exactly one refreshed viewer process.

- [ ] **Step 1: Run the complete automated gate**

```bash
python3 -m unittest tests.python.test_g1_reach_coverage_viewer -v
make build/tests/test_interaction_hand_trajectories \
  build/tests/test_reach_coverage build/tests/test_reach_search \
  build/tests/test_interaction_ik g1_reach_coverage_probe \
  g1_reach_coverage_viewer
build/tests/test_interaction_hand_trajectories
build/tests/test_reach_coverage
build/tests/test_reach_search
build/tests/test_interaction_ik
git diff --exit-code 50f5542 -- \
  interaction_ik.h interaction_ik.cpp g1_arm_joint_metadata.h
git diff --check
```

Expected: all tests/builds pass and the unchanged-IK command emits no output.

- [ ] **Step 2: Regenerate real collision evidence**

```bash
./g1_reach_coverage_probe build/g1-reaches/reach-pack-v2 \
  --json build/g1-reaches/contact-anchored-coverage-report-v4.json
jq '{search_integrity_passed, fixtures: (.shared_grasps | \
  map_values({accepted, processed_instances, rejections, elapsed_seconds}))}' \
  build/g1-reaches/contact-anchored-coverage-report-v4.json
```

Expected: each fixture processes `4608/4608` within 30 seconds. Accepted counts
may decrease because penetrative approaches are now rejected; do not weaken
collision or IK gates to preserve old counts.

- [ ] **Step 3: Record evidence and request focused review**

Append the commits, test commands, real-pack counts, object-collision
rejections, and unchanged-IK evidence to `.superpowers/sdd/progress.md`.
Request review from the pre-design baseline through `HEAD`, specifically for
oriented-box math, endpoint-only contact, first-collision diagnostics, and
regressions. Resolve all Critical and Important findings test-first.

- [ ] **Step 4: Commit evidence**

```bash
git add -f .superpowers/sdd/progress.md
git commit -m "docs: record swept grasp collision evidence"
```

- [ ] **Step 5: Replace only the exact running viewer**

Resolve and terminate only:

```bash
pgrep -af '(^|/)g1_reach_coverage_viewer( |$)'
```

Launch exactly one:

```bash
DISPLAY=:1 ./g1_reach_coverage_viewer \
  build/g1-reaches/reach-pack-v2
```

Verify one matching PID with bounded RSS. In the viewer, press `G`, then
`Enter`; motions whose active grasp sweeps through the object must now appear
under rejected object-collision candidates rather than accepted options.
