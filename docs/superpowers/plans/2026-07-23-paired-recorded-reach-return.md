# Paired Recorded Reach Return Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay each selected reach recording's warped post-grab return, then carry with its reachable final arm posture instead of continuous carry IK.

**Architecture:** Reach-pack version 2 stores one full outbound-and-return clip plus an absolute contact frame per reach. Search still shapes and scores only the outbound interval. A new return shaper applies the inverse-time continuation of the contact warp, validates the attached trajectory, and supplies an explicit episode `Return` state; after it finishes, walking owns locomotion while the recorded final arm remains a joint-space carry layer.

**Tech Stack:** Python 3, NumPy, C++17, existing G1 posture-aware IK and collision helpers, Make, Python `unittest`.

## Global Constraints

- Preserve the existing outbound endpoint features, candidate ranking, pickup playback, placement search, release, and reversed placement playback.
- Source archives, review motions, and accepted annotations remain unchanged.
- Every paired return comes from the selected captured source interval; mirrored reaches use the exact bilateral mirror.
- The contact warp has weight `1` at contact and `0` at the return endpoint.
- The frozen `hand_in_object` transform owns the object throughout attached playback.
- A return failure rejects the candidate before attachment.
- Walking-quality changes are outside this repair.
- Generated `build/g1-reaches/` artifacts and root-level viewer/probe binaries remain untracked.

---

### Task 1: Extract Stable Paired Returns

**Files:**
- Create: `resources/g1_reach_builder/returns.py`
- Modify: `resources/g1_reach_builder/motions.py`
- Test: `tests/python/test_g1_reach_returns.py`
- Test: `tests/python/test_g1_reach_motions.py`

**Interfaces:**
- Consumes: root-relative 25 Hz wrist trace, accepted `departure_frame`, and accepted `grab_frame`.
- Produces: `ReturnSegmentationConfig` and `find_return_stop(trace, departure_frame, grab_frame, config) -> int`, where the result is an exclusive source-local stop.
- Produces: `CanonicalReach.contact_index: int`; `CanonicalReach.positions` spans outbound start through the paired return endpoint.

- [ ] **Step 1: Write failing return-segmentation tests**

Add deterministic synthetic traces covering a normal return, a contact pause
that must not be mistaken for the return endpoint, and a missing return:

```python
class PairedReturnSegmentationTest(unittest.TestCase):
    def test_finds_stable_return_after_contact_pause(self):
        trace = synthetic_reach(
            outbound=30, contact_pause=3, inbound=24, nominal_pause=6
        )
        stop = find_return_stop(trace, 0, 29)
        self.assertEqual(stop, len(trace))

    def test_rejects_motion_without_retraction(self):
        trace = np.zeros((80, 3), np.float64)
        trace[20:, 0] = 0.45
        with self.assertRaisesRegex(ValueError, "paired return"):
            find_return_stop(trace, 0, 20)
```

Run:

```bash
python3 -m unittest tests.python.test_g1_reach_returns -v
```

Expected: import failure because `returns.py` does not exist.

- [ ] **Step 2: Implement bounded stable-return detection**

Create:

```python
@dataclass(frozen=True)
class ReturnSegmentationConfig:
    fps: float = 25.0
    maximum_return_s: float = 8.0
    stable_window_s: float = 0.20
    maximum_stable_speed_mps: float = 0.40
    minimum_retraction_m: float = 0.08


def find_return_stop(
    trace: np.ndarray,
    departure_frame: int,
    grab_frame: int,
    config: ReturnSegmentationConfig = ReturnSegmentationConfig(),
) -> int:
    # Validate finite (T, 3) input and frame ordering.
    # Search only grab+1 through grab+maximum_return_s.
    # Require at least minimum_retraction_m away from the grab wrist.
    # Among complete low-speed windows, select the one minimizing distance
    # to the departure wrist; return the window's exclusive stop.
    # Raise ValueError when no stable retracted window exists.
```

Use finite differences at `config.fps`, require the complete stability window,
and break ties by the earliest frame.

- [ ] **Step 3: Extend captured reach construction**

Change `build_captured_reach` to load the source-local wrist trace, call
`find_return_stop`, retain frames from the existing outbound start through the
exclusive return stop, and pass:

```python
contact_index = annotation.grab_frame - retained_start
```

Update `_reach_from_local` so endpoint position, endpoint rotation, and
approach direction are measured at `contact_index`, not `positions[-1]`.
Validate `0 < contact_index < frame_count - 1` and contiguous source frames.

- [ ] **Step 4: Prove the complete accepted corpus pairs**

Add a data-backed test that loads `review-v2` and `annotations-v2.json`, builds
all 192 captured reaches, and checks:

```python
self.assertEqual(len(captured), 192)
self.assertTrue(all(r.contact_index + 1 < r.frame_count for r in captured))
self.assertTrue(all(
    np.array_equal(np.diff(r.source_frames), np.ones(r.frame_count - 1))
    for r in captured
))
```

Tune only the explicit `ReturnSegmentationConfig` thresholds if this test
identifies a genuine stable authored return. Do not drop annotations.

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_returns \
  tests.python.test_g1_reach_motions -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_reach_builder/returns.py \
  resources/g1_reach_builder/motions.py \
  tests/python/test_g1_reach_returns.py \
  tests/python/test_g1_reach_motions.py
git commit -m "feat: pair recorded returns with captured reaches"
```

---

### Task 2: Publish Reach-Pack Version 2

**Files:**
- Modify: `resources/g1_reach_builder/mirror.py`
- Modify: `resources/g1_reach_builder/artifacts.py`
- Modify: `resources/g1_reach_builder/build.py`
- Modify: `resources/build_g1_reach_database.py`
- Test: `tests/python/test_g1_reach_mirror.py`
- Test: `tests/python/test_g1_reach_artifacts.py`
- Test: `tests/python/test_g1_reach_build_cli.py`

**Interfaces:**
- Consumes: `CanonicalReach.contact_index`.
- Produces: reach database magic `G1RCHD2`, feature magic `G1RCHF2`, version `2`, and `ReachArtifact.contact_frames: np.ndarray[int32]`.
- Guarantees: `range_starts[clip] <= contact_frames[clip] < range_stops[clip] - 1`.

- [ ] **Step 1: Write failing mirror and wire-format tests**

Assert the mirrored clip preserves `contact_index`, mirrors every return pose,
and mirrors twice back to the captured full clip. Extend artifact round-trip
tests to require:

```python
self.assertEqual(manifest["version"], 2)
np.testing.assert_array_equal(
    loaded.contact_frames,
    artifact.contact_frames,
)
```

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_mirror \
  tests.python.test_g1_reach_artifacts -v
```

Expected: failures because contact metadata and version 2 are absent.

- [ ] **Step 2: Mirror complete clips and contact metadata**

Set the mirrored reach's `contact_index=source.contact_index`. Keep the current
whole-pose reflection and provenance logic; because captured poses now include
the return, no separate mirroring path is needed.

- [ ] **Step 3: Add contact frames to the artifact**

Use:

```python
DATABASE_MAGIC = b"G1RCHD2\0"
FEATURE_MAGIC = b"G1RCHF2\0"
VERSION = 2

contact_frames = starts + np.asarray(
    [reach.contact_index for reach in reaches], np.int32
)
```

Write `contact_frames` immediately after `range_stops` and read it in the same
position. Validate contact bounds and require at least one return frame.
Features remain the same ten floats and must equal the contact endpoint
metadata byte-for-byte.

- [ ] **Step 4: Report return metadata**

Add manifest fields:

```python
"paired_returns": len(reaches),
"return_frame_count": int(np.sum(stops - contact_frames - 1)),
"minimum_return_frames": int(np.min(stops - contact_frames - 1)),
"maximum_return_frames": int(np.max(stops - contact_frames - 1)),
```

Run:

```bash
python3 -m unittest \
  tests.python.test_g1_reach_mirror \
  tests.python.test_g1_reach_artifacts \
  tests.python.test_g1_reach_build_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add resources/g1_reach_builder/mirror.py \
  resources/g1_reach_builder/artifacts.py \
  resources/g1_reach_builder/build.py \
  resources/build_g1_reach_database.py \
  tests/python/test_g1_reach_mirror.py \
  tests/python/test_g1_reach_artifacts.py \
  tests/python/test_g1_reach_build_cli.py
git commit -m "feat: publish paired-return reach pack v2"
```

---

### Task 3: Load Contact-Bounded Clips in C++

**Files:**
- Modify: `reach_database.h`
- Modify: `reach_database.cpp`
- Modify: `reach_motion.h`
- Modify: `reach_motion.cpp`
- Modify: `reach_coverage.cpp`
- Test: `tests/cpp/test_reach_database.cpp`
- Test: `tests/cpp/test_reach_coverage.cpp`

**Interfaces:**
- Produces: `Database::contact_frames`.
- Produces: `clip_contact_frame(const Database&, size_t) -> int32_t`,
  `clip_return_start(...) -> int32_t`, and
  `clip_return_stop(...) -> int32_t`.
- Guarantees: outbound shaping iterates `[range_start, contact_frame]`;
  return frames are `[contact_frame + 1, range_stop)`.

- [ ] **Step 1: Write failing loader and outbound-boundary tests**

Update the binary fixture to write version 2 contact frames. Assert malformed
contact-before-start, contact-at-stop, and no-return records throw
`interaction::FormatError`. Add a coverage fixture whose frames after contact
move wildly and assert the shaped outbound pose count is still:

```cpp
contact_frame - range_start + 1
```

Run:

```bash
make build/tests/test_reach_database build/tests/test_reach_coverage
./build/tests/test_reach_database
./build/tests/test_reach_coverage
```

Expected: loader/version failures.

- [ ] **Step 2: Implement the version 2 loader**

Change magic/version constants to `G1RCHD2`, `G1RCHF2`, and `2`. Read
`contact_frames` after range stops, validate one per clip, and enforce:

```cpp
start <= contact && contact + 1 < stop
```

Add the three checked accessors in `reach_motion.cpp`.

- [ ] **Step 3: Bound search and collision checks at contact**

In `shape_candidate`, replace the outbound `stop` with:

```cpp
const int32_t stop = clip_contact_frame(pack.database, candidate.clip) + 1;
```

Keep placement alignment, endpoint correction, directness, and trajectory
collision checks otherwise unchanged.

Run the two focused tests and expect PASS.

- [ ] **Step 4: Commit**

```bash
git add reach_database.h reach_database.cpp reach_motion.h reach_motion.cpp \
  reach_coverage.cpp tests/cpp/test_reach_database.cpp \
  tests/cpp/test_reach_coverage.cpp
git commit -m "feat: load contact-bounded reach returns"
```

---

### Task 4: Shape and Preflight the Warped Recorded Return

**Files:**
- Create: `reach_return.h`
- Create: `reach_return.cpp`
- Modify: `reach_coverage.h`
- Modify: `reach_search.cpp`
- Modify: `episode_reach_planner.h`
- Modify: `episode_reach_planner.cpp`
- Modify: `Makefile`
- Test: `tests/cpp/test_reach_return.cpp`
- Test: `tests/cpp/test_episode_reach_planner.cpp`

**Interfaces:**
- Produces:

```cpp
enum class ReturnRejection : uint8_t {
    None, InvalidSolver, Seam, ObjectCollision, EnvironmentCollision
};

struct ShapedReturn {
    std::vector<interaction::Pose> poses;
    ReturnRejection rejection = ReturnRejection::None;
    size_t rejected_sample = 0U;
};

ShapedReturn shape_recorded_return(
    const Pack&,
    const Candidate&,
    const Query&,
    const interaction::Pose& solved_contact,
    interaction::Transform hand_in_object,
    vec3 object_dimensions,
    const interaction::EnvironmentGeometry&,
    const SearchConfig&);
```

- Extends `ReachPlan` with `std::vector<interaction::Pose> return_poses`.

- [ ] **Step 1: Write failing warp-continuity tests**

Create a compact version 2 fixture with a known contact and return. Assert:

- first returned pose equals `solved_contact`;
- the first correction weight is one;
- the final correction weight is zero;
- the final wrist equals the aligned recorded return wrist;
- every solved wrist reconstructs the object from the frozen
  `hand_in_object`;
- a deliberately unreachable return reports `InvalidSolver`;
- a return crossing a box reports `EnvironmentCollision`.

Run:

```bash
make build/tests/test_reach_return
```

Expected: compilation failure because `reach_return.h` is absent.

- [ ] **Step 2: Implement inverse-time contact correction**

Align each recorded return pose with the candidate's existing placement yaw
and root translation. Compute:

```cpp
const Transform delta = compose(
    solved_contact_hand,
    inverse(aligned_source_contact_hand));
const float weight = 1.0F - smoothstep(
    static_cast<float>(sample) /
    static_cast<float>(return_count));
const Transform weighted_delta{
    weight * delta.position,
    quat_nlerp_shortest(quat(), delta.rotation, weight),
};
const Transform target = compose(weighted_delta, aligned_source_hand);
```

Publish `solved_contact` as sample zero. For each later frame, solve the
existing posture-aware task-priority IK against `target`, using that recorded
pose as source and transporting the previous correction in the temporal seed.
Reject non-finite or unaccepted frames.

- [ ] **Step 3: Validate attached return collisions**

For each solved frame, derive:

```cpp
object.world = compose(
    solved_hand,
    inverse(hand_in_object));
```

Run the existing body-vs-object and body-vs-environment feasibility check for
that single frame with the active-hand exemption. Add a focused oriented-box
SAT helper for held-object-vs-environment overlap and reject on the first
colliding sample.

- [ ] **Step 4: Integrate return feasibility into plan selection**

After `regenerate` succeeds, derive `hand_in_object` from the frozen object and
target grasp, shape the return, and accept the candidate only if its return is
valid. If it fails, continue through the remaining accepted candidates instead
of returning `nullopt` immediately. Store the valid `return_poses` in
`ReachPlan`.

Run:

```bash
make build/tests/test_reach_return build/tests/test_episode_reach_planner
./build/tests/test_reach_return
./build/tests/test_episode_reach_planner
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reach_return.h reach_return.cpp reach_coverage.h reach_search.cpp \
  episode_reach_planner.h episode_reach_planner.cpp Makefile \
  tests/cpp/test_reach_return.cpp tests/cpp/test_episode_reach_planner.cpp
git commit -m "feat: shape and preflight recorded reach returns"
```

---

### Task 5: Play Return and Carry Without Continuous IK

**Files:**
- Modify: `episode_layered_carry.h`
- Modify: `episode_layered_carry.cpp`
- Modify: `interaction_episode.h`
- Modify: `interaction_episode.cpp`
- Modify: `g1_interaction_episode_viewer.cpp`
- Test: `tests/cpp/test_episode_layered_carry.cpp`
- Test: `tests/cpp/test_interaction_episode.cpp`
- Test: `tests/python/test_g1_interaction_episode_viewer.py`

**Interfaces:**
- Adds `EpisodeState::Return`.
- Replaces `LayeredCarry::start(final_hold_pose, hand, hand_in_object,
  object_world)` with `start(nominal_return_pose, hand, hand_in_object)`.
- `LayeredCarry::update(locomotion)` copies only the selected arm rotations
  from the nominal return pose over live walking and derives object world from
  the displayed wrist; it performs no IK solve.

- [ ] **Step 1: Write failing episode-state and carry tests**

Add a frozen attempt with three return poses. Assert:

```cpp
Reach -> Return -> Carry
```

and verify the object remains attached on every return frame. In the layered
carry test, drive 100 walking ticks and require root/leg motion while selected
arm rotations stay equal to the nominal return layer. Remove expectations for
`last_solve_accepted()`.

Run:

```bash
make build/tests/test_episode_layered_carry \
  build/tests/test_interaction_episode
./build/tests/test_episode_layered_carry
./build/tests/test_interaction_episode
```

Expected: compilation/state failures because `Return` is absent.

- [ ] **Step 2: Add recorded return playback**

At successful contact:

```cpp
state_ = EpisodeState::Return;
reach_frame_ = 0U;
publish(attempt_->plan.return_poses.front());
```

Advance at the fixed 25 Hz tick, publish every shaped return pose, update the
attached object from the wrist, and ignore locomotion commands. At the final
frame, rebase the walking matcher, start the carry layer from that final pose,
and enter `Carry`.

- [ ] **Step 3: Replace carry IK with a recorded arm layer**

Start from `locomotion.pose`, copy the seven selected-arm local rotations from
the final recorded return pose, optionally retain the existing 25% spine
nlerp, and derive the object from the active wrist and frozen
`hand_in_object`. Delete temporal IK state, object-in-root targeting, rejection
fallback, and carry-IK diagnostics.

Placement approach continues to use this same layer until `PlaceBridge`.

- [ ] **Step 4: Update viewer state text and contracts**

Display `RETURN` for the new state. Ensure input remains locked during Return
and the viewer no longer advertises carry IK rejection diagnostics.

Run:

```bash
make build/tests/test_episode_layered_carry \
  build/tests/test_interaction_episode g1_interaction_episode_viewer
./build/tests/test_episode_layered_carry
./build/tests/test_interaction_episode
python3 -m unittest tests.python.test_g1_interaction_episode_viewer -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add episode_layered_carry.h episode_layered_carry.cpp \
  interaction_episode.h interaction_episode.cpp \
  g1_interaction_episode_viewer.cpp \
  tests/cpp/test_episode_layered_carry.cpp \
  tests/cpp/test_interaction_episode.cpp \
  tests/python/test_g1_interaction_episode_viewer.py
git commit -m "feat: replay recorded return before carry"
```

---

### Task 6: Rebuild Data and Verify the Complete Episode

**Files:**
- Modify only if a test exposes a scoped defect in Tasks 1–5.
- Generated, untracked: `build/g1-reaches/reach-pack-v3/`
- Generated, untracked: `g1_interaction_episode_viewer`

**Interfaces:**
- Produces the validated pack and live viewer used for manual acceptance.

- [ ] **Step 1: Run all builder and focused C++ tests**

```bash
python3 -m unittest discover -s tests/python -p 'test_g1_reach_*.py' -v
make build/tests/test_reach_database \
  build/tests/test_reach_coverage \
  build/tests/test_reach_return \
  build/tests/test_episode_reach_planner \
  build/tests/test_episode_layered_carry \
  build/tests/test_interaction_episode
./build/tests/test_reach_database
./build/tests/test_reach_coverage
./build/tests/test_reach_return
./build/tests/test_episode_reach_planner
./build/tests/test_episode_layered_carry
./build/tests/test_interaction_episode
```

Expected: all PASS.

- [ ] **Step 2: Build the validated paired-return pack**

```bash
python3 -m resources.build_g1_reach_database \
  --review build/g1-reaches/review-v2 \
  --annotations build/g1-reaches/annotations-v2.json \
  --output build/g1-reaches/reach-pack-v3
```

Read the generated pack back through both Python and C++ loaders. Require 192
captured and 192 mirrored reaches, 384 paired returns, unchanged ten-dimensional
features, and no extraction failures.

- [ ] **Step 3: Run the full safe regression suite**

```bash
make test-safe
```

Expected: PASS with no new failures.

- [ ] **Step 4: Drive the viewer**

Build and launch:

```bash
make g1_interaction_episode_viewer
./g1_interaction_episode_viewer \
  build/g1-episode build/g1-reaches/reach-pack-v3
```

Manually exercise:

1. WASD before pickup.
2. Pickup from at least three angles and two heights.
3. Watch the full attached recorded return with no contact jump.
4. Walk while carrying and confirm animated legs with a stable arm layer.
5. Place on the shelf and lower table.
6. Re-pick and place again.

Capture logs and reject any `return`, attachment, non-finite, skeleton, or
collision diagnostic.

- [ ] **Step 5: Commit any scoped verification correction**

If verification required no source correction, do not create an empty commit.
Never add generated packs or binaries.

