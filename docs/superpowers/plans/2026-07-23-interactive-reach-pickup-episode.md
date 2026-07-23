# Interactive Reach Pickup Episode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a flat playable G1 viewer where WASD locomotion, one frozen F request, grasp-relative reach matching, attachment, and hand-correct carry form one continuous pickup episode.

**Architecture:** Convert the user's compact `walking` and `carry_walking` recordings into three 25 Hz G1 motion-matching databases: free walking, left carry, and mirrored right carry. A dedicated viewer combines a small flat locomotion matcher, the existing exhaustive reach search/posture IK/collision pipeline, the existing attachment authority, and a deterministic episode state machine. The live GraspNet and diffusion boundaries are represented by interfaces but use deterministic providers in this milestone.

**Tech Stack:** C++17, raylib, existing G1 pose/IK/reach/attachment modules, Python 3, NumPy, existing GMR/MuJoCo conversion helpers, Make.

## Global Constraints

- Use a dedicated `g1_interaction_episode_viewer`; do not modify production `controller` behavior.
- Render flat geometry only; do not load the terrain viewer or terrain scene.
- Do not load `build/smart-pickup/table-ground-pack` or another gigabyte-scale interaction pack.
- Run simulation at a fixed 25 Hz and rendering/input at 60 Hz.
- F freezes one immutable object/grasp attempt and never recomputes while active.
- The exhaustive search deadline is exactly three seconds; the target is under one second.
- Use the complete bilateral reach pack at `build/g1-reaches/reach-pack-v2`.
- Carry uses the captured left `carry_walking` recording and a deterministic mirrored-right copy.
- Diffusion, live GraspNet, placement, release, multi-object selection, and production-controller integration are out of scope.
- Keep generated motion packs under `build/g1-episode/`; do not commit generated binaries.

---

### Task 1: Build the compact episode locomotion pack

**Files:**
- Create: `resources/build_g1_episode_motion_pack.py`
- Create: `tests/python/test_g1_episode_motion_pack.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: trusted GMR zip accepted by `resources.g1_reach_builder.sources.load_gmr_archive`, `G1Kinematics`, `convert_source_clip`, and `resources.retime_flat_database.write_database`.
- Produces: `build/g1-episode/walking_database.bin`, `carry_left_database.bin`, `carry_right_database.bin`, and `manifest.json`, all at 25 Hz with 31 bones.

- [ ] **Step 1: Write failing builder tests**

Add tests that call pure functions with synthetic 31-bone local transforms:

```python
from resources.build_g1_episode_motion_pack import (
    EpisodeMotion,
    mirror_episode_motion,
    validate_episode_motion,
)


def test_mirror_swaps_limbs_and_reflects_root_z():
    source = synthetic_episode_motion()
    mirrored = mirror_episode_motion(source)
    assert np.allclose(
        mirrored.positions[:, 0, 2],
        -source.positions[:, 0, 2],
    )
    assert np.allclose(
        mirrored.foot_contacts,
        source.foot_contacts[:, ::-1],
    )
    assert mirrored.positions.shape == source.positions.shape


def test_validate_rejects_wrong_skeleton_width():
    source = synthetic_episode_motion()
    source.positions = source.positions[:, :-1]
    with pytest.raises(ValueError, match="31 bones"):
        validate_episode_motion(source)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m pytest tests/python/test_g1_episode_motion_pack.py -q
```

Expected: collection fails because `resources.build_g1_episode_motion_pack` does not exist.

- [ ] **Step 3: Implement deterministic conversion and mirroring**

Define:

```python
@dataclass
class EpisodeMotion:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    foot_contacts: np.ndarray
    source_frames: np.ndarray


@dataclass(frozen=True)
class EpisodePackPaths:
    walking: Path
    carry_left: Path
    carry_right: Path
    manifest: Path
```

Implement these exact call signatures:

- `convert_named_source(archive: Path, sequence_id: str,
  kinematics: G1Kinematics) -> EpisodeMotion`
- `mirror_episode_motion(source: EpisodeMotion) -> EpisodeMotion`
- `write_episode_pack(output: Path, walking: EpisodeMotion,
  carry_left: EpisodeMotion, carry_right: EpisodeMotion) ->
  EpisodePackPaths`

Use the canonical `G1_SKELETON`, `MIRROR_BONES`, quaternion reflection from
`resources/g1_reach_builder/mirror.py`, and finite-difference/contact helpers
already used by the reach builder. Write each motion as a one-range legacy
`FlatDatabase`; this lets C++ reuse `database_load` and
`database_build_matching_features`. The manifest must bind source archive
SHA-256, target FPS, skeleton signature, frame counts, and each output SHA-256.
Use temporary-directory plus atomic rename semantics from the reach pack
builder.

The CLI is:

```bash
python3 -m resources.build_g1_episode_motion_pack \
  --archive /home/ubuntu/Downloads/g1_retargeted_motions.zip \
  --output build/g1-episode
```

- [ ] **Step 4: Run unit tests and build the real pack**

Run:

```bash
python3 -m pytest tests/python/test_g1_episode_motion_pack.py -q
python3 -m resources.build_g1_episode_motion_pack \
  --archive /home/ubuntu/Downloads/g1_retargeted_motions.zip \
  --output build/g1-episode
python3 - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("build/g1-episode/manifest.json").read_text())
assert m["target_fps"] == 25.0
assert m["bone_count"] == 31
assert set(m["motions"]) == {"walking", "carry_left", "carry_right"}
print({k: v["frame_count"] for k, v in m["motions"].items()})
PY
```

Expected: all tests pass; the manifest reports approximately 1,700–2,000
frames per source and the right carry frame count equals the left.

- [ ] **Step 5: Ignore generated artifacts and commit**

Add `/build/g1-episode/` to `.gitignore`, then run:

```bash
git add .gitignore resources/build_g1_episode_motion_pack.py \
  tests/python/test_g1_episode_motion_pack.py
git commit -m "feat: build compact G1 episode motion pack"
```

---

### Task 2: Add the known-grasp provider and playable reach ranking

**Files:**
- Create: `episode_grasp_provider.h`
- Create: `episode_grasp_provider.cpp`
- Create: `episode_reach_planner.h`
- Create: `episode_reach_planner.cpp`
- Create: `tests/cpp/test_episode_reach_planner.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `interaction::Transform`, `interaction::OrientedBox`,
  `reach::Pack`, `reach::SearchResult`, and `reach::Evaluation`.
- Produces: `episode::KnownGraspProvider::query(ObjectSnapshot)` and
  `episode::choose_reach_plan(const reach::Pack&, const
  reach::SearchResult&, const reach::ExhaustiveQuery&, const
  interaction::OrientedBox&, const interaction::EnvironmentGeometry&, const
  reach::SearchConfig&, const interaction::Pose&) ->
  std::optional<ReachPlan>`.

- [ ] **Step 1: Write failing provider and ranking tests**

Cover object-relative transform invariance, both-hand eligibility,
entry-before-directness ranking, and deterministic ties:

```cpp
const episode::ObjectSnapshot object{
    7U,
    {vec3(2.0F, 0.7F, -1.0F),
     quat_from_angle_axis(0.5F, vec3(0, 1, 0))},
    vec3(0.1F, 0.1F, 0.1F)};
const auto grasps = provider.query(object);
require(grasps.size() == 1U, "known provider changed count");
require(near(
    grasps[0].hand_world.position,
    interaction::compose(object.world, local_grasp).position),
    "known grasp was not object-relative");

episode::ReachPlanCost near_hooked{0.30F, 0.05F, 0.08F, 0.20F, 4U, 0U};
episode::ReachPlanCost far_direct{2.00F, 0.00F, 0.00F, 0.00F, 1U, 0U};
require(
    episode::reach_plan_cost_less(near_hooked, far_direct),
    "directness incorrectly outranked playable entry distance");
```

- [ ] **Step 2: Run the planner test and verify RED**

Run:

```bash
make build/test_episode_reach_planner
./build/test_episode_reach_planner
```

Expected: build fails because the episode provider/planner files do not exist.

- [ ] **Step 3: Implement provider and plan types**

The public contract is:

```cpp
namespace episode {

struct ObjectSnapshot {
    uint64_t generation = 0U;
    interaction::Transform world{};
    vec3 dimensions{};
};

struct GraspCandidate {
    interaction::Transform hand_world{};
    vec3 approach_world{1.0F, 0.0F, 0.0F};
    std::optional<interaction::Hand> preferred_hand{};
    uint64_t grasp_id = 0U;
};

class GraspProvider {
public:
    virtual ~GraspProvider() = default;
    virtual std::vector<GraspCandidate> query(
        const ObjectSnapshot& object) const = 0;
};

class KnownGraspProvider final : public GraspProvider {
public:
    KnownGraspProvider(
        interaction::Transform hand_in_object,
        vec3 approach_in_object);
    std::vector<GraspCandidate> query(
        const ObjectSnapshot& object) const override;
private:
    interaction::Transform hand_in_object_{};
    vec3 approach_in_object_{};
};

}
```

Reject nonfinite transforms, zero approach vectors, and nonpositive object
dimensions.

- [ ] **Step 4: Implement entry-aware selection**

Define:

```cpp
struct ReachPlanCost {
    float entry_distance_m;
    float entry_facing_error_radians;
    float grasp_error;
    float directness_cost;
    size_t clip;
    uint8_t yaw_index;
};

struct ReachPlan {
    uint64_t object_generation;
    GraspCandidate grasp;
    reach::Evaluation reach;
    reach::Hand hand;
    interaction::Transform entry_root_world;
    ReachPlanCost cost;
};

std::optional<ReachPlan> choose_reach_plan(
    const reach::Pack& pack,
    const reach::SearchResult& compact,
    const reach::ExhaustiveQuery& query,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    const reach::SearchConfig& search_config,
    const interaction::Pose& live_pose);
```

For each accepted compact evaluation, derive the entry pose from the clip's
range start and candidate placement. Reject a candidate if the straight planar
segment from the live root to entry intersects an environment box expanded by
0.28 m. Sort lexicographically by entry distance, entry facing error, summed
post-IK grasp error, directness, clip, and yaw index. Regenerate only the
winner's full corrected poses.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
make build/test_episode_reach_planner
./build/test_episode_reach_planner
```

Expected: `episode reach planner PASS`.

Commit:

```bash
git add episode_grasp_provider.* episode_reach_planner.* \
  tests/cpp/test_episode_reach_planner.cpp Makefile
git commit -m "feat: plan playable grasp-relative reaches"
```

---

### Task 3: Extract a compact flat G1 motion matcher

**Files:**
- Create: `g1_flat_motion_matcher.h`
- Create: `g1_flat_motion_matcher.cpp`
- Create: `tests/cpp/test_g1_flat_motion_matcher.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: the three legacy 31-bone databases from Task 1 and a planar
  `episode::LocomotionCommand`.
- Produces: continuous `interaction::LocomotionSnapshot` values and supports
  transactional database switches for walking/left carry/right carry.

- [ ] **Step 1: Write failing continuity and control tests**

Use a small in-memory two-range database and assert:

```cpp
episode::FlatMotionMatcher matcher(database_fixture());
const interaction::Pose before = matcher.snapshot().pose;
matcher.update({vec3(0.0F, 0.0F, 1.0F), quat()}, 1.0F / 25.0F);
const interaction::Pose after = matcher.snapshot().pose;
require(after.positions[0].z > before.positions[0].z,
        "forward command did not move root");

const vec3 live_root = after.positions[0];
matcher.switch_database(carry_fixture());
require(length(matcher.snapshot().pose.positions[0] - live_root) < 1.0e-5F,
        "database switch moved live root");

for (int tick = 0; tick < 8; ++tick) {
    matcher.update(command, 1.0F / 25.0F);
    require(maximum_joint_step(previous, matcher.snapshot().pose) < 0.35F,
            "motion-match transition exceeded joint-step bound");
    previous = matcher.snapshot().pose;
}
```

- [ ] **Step 2: Run the matcher test and verify RED**

Run:

```bash
make build/test_g1_flat_motion_matcher
./build/test_g1_flat_motion_matcher
```

Expected: build fails because `g1_flat_motion_matcher.h` is missing.

- [ ] **Step 3: Implement database loading and pose mapping**

Define:

```cpp
namespace episode {

struct LocomotionCommand {
    vec3 desired_velocity_world{};
    quat desired_heading_world{};
};

class FlatMotionMatcher {
public:
    explicit FlatMotionMatcher(std::filesystem::path database_path);
    explicit FlatMotionMatcher(database source);
    void update(const LocomotionCommand& command, float dt);
    void switch_database(std::filesystem::path database_path);
    const interaction::LocomotionSnapshot& snapshot() const;
    float planar_speed() const;
private:
    database database_{};
    int32_t source_frame_ = 0;
    double source_frame_exact_ = 0.0;
    float search_elapsed_seconds_ = 0.0F;
    interaction::Transform world_from_source_{};
    interaction::LocomotionSnapshot snapshot_{};
    interaction::Pose transition_source_{};
    float transition_elapsed_seconds_ = 0.20F;
    bool transition_active_ = false;
};

}
```

On load, require 31 bones and exact `g1_skeleton::kParents`; add a zero
`terrain_features` array and call `database_build_matching_features` with
`LeftToe`, `RightToe`, `Hips`, and terrain weight zero. Convert each database
row into the native `interaction::Pose` contract.

- [ ] **Step 4: Implement flat matching and root continuity**

At each fixed tick:

1. preserve the current pose portion of the 31-value query by denormalizing
   dimensions 0–14 from the active source frame;
2. populate dimensions 15–20 from desired root positions at 0.32, 0.68, and
   1.00 seconds;
3. populate dimensions 21–26 with desired facing;
4. set dimensions 27–30 to zero;
5. call `database_search` every 0.10 seconds or on a command edge;
6. on a selected-frame change, compute a planar source-to-world alignment that
   maps the new source root exactly onto the currently displayed root; and
7. smoothstep local positions/rotations from the prior displayed pose over
   0.20 seconds while root translation continues from mapped source deltas.

Advance using an accumulator in exact 1/25-second steps. Publish future root
samples at 0.32, 0.68, and 1.00 seconds from the requested command.

- [ ] **Step 5: Run tests against fixture and real walking data**

Run:

```bash
make build/test_g1_flat_motion_matcher
./build/test_g1_flat_motion_matcher
./build/test_g1_flat_motion_matcher build/g1-episode/walking_database.bin
```

Expected: both invocations print `flat G1 motion matcher PASS`; the real-data
mode reports nonzero root displacement, finite poses, and bounded transitions.

- [ ] **Step 6: Commit**

```bash
git add g1_flat_motion_matcher.* \
  tests/cpp/test_g1_flat_motion_matcher.cpp Makefile
git commit -m "feat: add compact flat G1 motion matcher"
```

---

### Task 4: Implement the deterministic pickup episode coordinator

**Files:**
- Create: `interaction_episode.h`
- Create: `interaction_episode.cpp`
- Create: `tests/cpp/test_interaction_episode.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: frozen `ReachPlan`, `FlatMotionMatcher`, `TargetRegistry`,
  `AttachmentController`, and fixed-tick commands.
- Produces: `EpisodeOutput { pose, object_world, owns_pose, state,
  selected_hand, failure }`.

- [ ] **Step 1: Write failing state, freeze, and attachment tests**

Use a short synthetic reach with distinct world roots and a fake planner:

```cpp
runtime.press_interact(object_generation, grasp, plan);
require(runtime.state() == EpisodeState::Approach,
        "F did not commit one plan");
runtime.press_interact(object_generation, grasp, replacement_plan);
require(runtime.plan().reach.candidate.clip == plan.reach.candidate.clip,
        "repeated F replaced a committed plan");

advance_until(runtime, EpisodeState::Reach);
require(maximum_root_step(samples) < 0.08F,
        "approach/bridge root continuity failed");
advance_until(runtime, EpisodeState::Carry);
require(runtime.output().attached, "object did not attach at contact");
require(runtime.output().selected_hand == interaction::Hand::Left,
        "selected hand was not propagated into carry");
```

Also test cancel before contact, object generation invalidation, contact
rejection, timeout, and reset.

- [ ] **Step 2: Run the runtime test and verify RED**

Run:

```bash
make build/test_interaction_episode
./build/test_interaction_episode
```

Expected: build fails because `interaction_episode.h` does not exist.

- [ ] **Step 3: Implement immutable attempt and state transitions**

Define:

```cpp
enum class EpisodeState : uint8_t {
    FreeLocomotion,
    Searching,
    Approach,
    Bridge,
    Reach,
    CarryBlend,
    Carry,
    Failed,
};

struct EpisodeConfig {
    float entry_position_m = 0.08F;
    float entry_yaw_radians = 0.174532925F;
    float entry_speed_mps = 0.08F;
    int stable_entry_ticks = 3;
    float approach_timeout_seconds = 12.0F;
    float bridge_seconds = 0.24F;
    float carry_blend_seconds = 0.36F;
};

struct FrozenAttempt {
    ObjectSnapshot object;
    GraspCandidate grasp;
    ReachPlan plan;
    uint64_t request_id;
};
```

`commit` accepts an attempt only in `FreeLocomotion`. `update` rejects object
generation changes before contact. It never reads a mutable viewer object after
commit.

- [ ] **Step 4: Implement approach, bridge, and reach playback**

Approach computes a speed-clamped planar command toward
`plan.entry_root_world`; within 0.35 m it also converges heading. It enters
Bridge only after all entry tolerances hold for three consecutive 25 Hz ticks.

Bridge uses:

```cpp
Pose bridge_pose(
    const Pose& live,
    const Pose& reach_entry,
    float alpha) {
    Pose out = interaction::interpolate_pose(
        live, reach_entry, smoothstep(alpha));
    out.positions[g1_skeleton::Simulation] =
        lerp(live.positions[g1_skeleton::Simulation],
             reach_entry.positions[g1_skeleton::Simulation],
             smoothstep(alpha));
    return out;
}
```

Play the complete corrected `ReachPlan::reach.poses` at 25 Hz. The final frame
is the contact frame because the reach pack is outbound-only.

- [ ] **Step 5: Use existing attachment authority and switch carry database**

At commit, create a `TargetRegistry` target and reserve it with the frozen
request ID. Construct the affordance from:

```cpp
affordance.hand_in_object =
    interaction::compose(
        interaction::inverse(attempt.object.world),
        attempt.grasp.hand_world);
affordance.approach_direction_object =
    quat_inv_mul_vec3(
        attempt.object.world.rotation,
        attempt.grasp.approach_world);
```

On the final reach frame, build a `ContactMeasurement` from the displayed
selected wrist and call `AttachmentController::try_contact`. Enter
`CarryBlend` only on success.

Switch `FlatMotionMatcher` to `carry_left_database.bin` or
`carry_right_database.bin` according to the selected reach hand. Blend from
the contact pose to its continuous carry pose for 0.36 seconds. During
`CarryBlend` and `Carry`, call `AttachmentController::update` with the displayed
wrist so the object follows the actual mesh pose.

If carry matching produces an invalid pose after attachment, retain the last
finite carry pose, remap its root to the current locomotion root, and run
`solve_hand_posture_ik_task_priority` against the last valid wrist transform.
This procedural fallback remains active until reset and never releases or
teleports the object.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
make build/test_interaction_episode
./build/test_interaction_episode
```

Expected: `interaction episode PASS`, including both left and right carry
fixtures.

Commit:

```bash
git add interaction_episode.* tests/cpp/test_interaction_episode.cpp Makefile
git commit -m "feat: coordinate deterministic pickup episode"
```

---

### Task 5: Build the flat interactive viewer

**Files:**
- Create: `g1_interaction_episode_viewer.cpp`
- Create: `tests/python/test_g1_interaction_episode_viewer.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Tasks 1–4 and `resources/g1_mesh/g1_raylib.glb`.
- Produces: `g1_interaction_episode_viewer EPISODE_PACK REACH_PACK`.

- [ ] **Step 1: Write failing viewer contract tests**

Assert source/build contracts:

```python
SOURCE = Path("g1_interaction_episode_viewer.cpp").read_text()

def test_viewer_uses_flat_episode_runtime():
    assert '#include "interaction_episode.h"' in SOURCE
    assert "G1_TERRAIN_DIR" not in SOURCE
    assert "table-ground-pack" not in SOURCE

def test_controls_and_search_are_edge_triggered():
    assert "IsKeyPressed(KEY_F)" in SOURCE
    assert "IsKeyDown(KEY_W)" in SOURCE
    assert "IsKeyDown(KEY_A)" in SOURCE
    assert "IsKeyDown(KEY_S)" in SOURCE
    assert "IsKeyDown(KEY_D)" in SOURCE
```

- [ ] **Step 2: Run viewer tests and verify RED**

Run:

```bash
python3 -m pytest tests/python/test_g1_interaction_episode_viewer.py -q
```

Expected: tests fail because the viewer source does not exist.

- [ ] **Step 3: Implement viewer loading, controls, and asynchronous search**

The CLI is:

```text
g1_interaction_episode_viewer \
  [build/g1-episode] [build/g1-reaches/reach-pack-v2]
```

Load only the three compact locomotion databases, reach pack, G1 GLB, and flat
coverage geometry. Use the existing orbit-camera and G1 mesh update code from
`g1_reach_coverage_viewer.cpp`.

Controls:

- WASD: character locomotion in camera space.
- Arrow keys: object X/Z while no attempt is active.
- U/J: object up/down while no attempt is active.
- Q/E: object yaw while no attempt is active.
- F: freeze and start one search.
- Escape: cancel before attachment.
- R: reset episode and object.
- M/B: mesh/bones.
- Left/middle mouse and wheel: orbit/pan/zoom.

Run `reach::search_all` in one owned worker. The render thread displays
`SEARCHING`, never launches a second search, and joins the worker on reset or
window close. When complete, call `choose_reach_plan` on the render thread and
commit the immutable attempt.

- [ ] **Step 4: Render actionable diagnostics**

Draw flat floor, table, shelf, lower table, object, known grasp axes, selected
entry root/facing, straight root approach, and corrected wrist path. HUD text
must include:

```text
STATE <state> | HAND <left/right> | CLIP <index>
SEARCH <milliseconds> ms | ACCEPTED <count>
ENTRY <meters> m <degrees> deg | DIRECT <cost>
CONTACT <centimeters> cm <degrees> deg | ATTACHED <yes/no>
FAILURE <reason>
```

- [ ] **Step 5: Build and run automated viewer checks**

Run:

```bash
python3 -m pytest tests/python/test_g1_interaction_episode_viewer.py -q
make g1_interaction_episode_viewer
timeout 5s env DISPLAY=:1 ./g1_interaction_episode_viewer \
  build/g1-episode build/g1-reaches/reach-pack-v2
```

Expected: Python tests pass, build succeeds, and the timed graphical smoke run
does not crash or report missing terrain/interaction-pack assets.

- [ ] **Step 6: Commit**

```bash
git add g1_interaction_episode_viewer.cpp \
  tests/python/test_g1_interaction_episode_viewer.py Makefile
git commit -m "feat: add playable G1 pickup episode viewer"
```

---

### Task 6: Verify the complete baseline and capture evidence

**Files:**
- Create: `docs/superpowers/evidence/2026-07-23-interactive-reach-pickup-episode.md`
- Modify only if a verified defect is found: files from Tasks 1–5 and their tests.

**Interfaces:**
- Consumes: complete viewer and generated packs.
- Produces: reproducible automated and graphical acceptance evidence.

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
python3 -m pytest \
  tests/python/test_g1_episode_motion_pack.py \
  tests/python/test_g1_interaction_episode_viewer.py \
  tests/python/test_g1_reach_coverage_viewer.py -q
make build/test_episode_reach_planner \
  build/test_g1_flat_motion_matcher \
  build/test_interaction_episode
./build/test_episode_reach_planner
./build/test_g1_flat_motion_matcher build/g1-episode/walking_database.bin
./build/test_interaction_episode
```

Expected: every command passes.

- [ ] **Step 2: Run reach/IK/collision regressions**

Run the existing focused binaries:

```bash
make build/test_reach_database build/test_reach_search \
  build/test_reach_coverage build/test_interaction_posture_ik \
  build/test_interaction_hand_trajectories
./build/test_reach_database
./build/test_reach_search
./build/test_reach_coverage
./build/test_interaction_posture_ik
./build/test_interaction_hand_trajectories
```

Expected: all pass with the pre-existing accepted coverage hashes unchanged.

- [ ] **Step 3: Launch the real viewer and perform graphical acceptance**

Stop the old coverage viewer, then run:

```bash
env DISPLAY=:1 ./g1_interaction_episode_viewer \
  build/g1-episode build/g1-reaches/reach-pack-v2
```

Verify:

1. WASD visibly moves the G1 mesh before interaction.
2. Moving/rotating the object does not trigger search.
3. F starts one bounded search without freezing the window.
4. Approach does not jump to the origin or orbit an old waypoint.
5. Bridge and reach have no root rubber-band.
6. The hand reaches the displayed grasp.
7. The object attaches on the final reach frame.
8. WASD resumes in carry with the object following the selected wrist.
9. Reset restores both G1 and object.
10. Repeat from enough bearings to observe both left and right carry.

- [ ] **Step 4: Record evidence and commit**

Record exact commands, pass counts, search latency, selected clips/hands, and
any known nonblocking visual defects in the evidence file. Then:

```bash
git add docs/superpowers/evidence/2026-07-23-interactive-reach-pickup-episode.md
git commit -m "docs: record interactive pickup episode evidence"
```
