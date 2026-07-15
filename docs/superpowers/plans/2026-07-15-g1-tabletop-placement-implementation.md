# G1 Tabletop Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to execute this plan one task at a time. Each
> task receives a fresh implementation subagent, then spec-compliance and code-
> quality review before the next task starts. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Add a deterministic flat-ground `walk -> pick -> carry -> place`
primitive with true recorded-place priority, a certified and explicitly labelled
reversed-pickup fallback, atomic object release, and headless plus graphical 25 Hz
gates.

**Architecture:** Keep flat locomotion, pickup, Carry, and the existing playable
gate unchanged. Add a separate placement-surface registry, an atomic Held-to-Free
commit, and a focused place controller that supplies one contiguous recorded or
reversed motion to `InteractionRuntime`. Runtime appends four place states; the
controller resolves one explicit destination slot and publishes the same object
transform across the attachment-release seam.

**Tech Stack:** C++17, existing G1 interaction database and FK/IK code, Raylib,
Python `unittest`, GNU Make, fixed 25 Hz runtime.

## Global Constraints

- Start from integrated commit `c2b6ff814eced5fbe394409e22c5c7bd6019be10`.
- Work only on `g1-tabletop-placement-*` branches/worktrees; do not touch or
  merge the terrain checkout.
- Do not access, stat, hash, execute, modify, stage, or delete the protected
  repository-root artifact `interaction_query_probe`. Use
  `build/task12/interaction_query_probe_safe` through the existing safe targets.
- Runtime, flat locomotion, interaction playback, auto-demo, and evidence all
  advance exactly once per `1/25` second.
- Existing pickup behavior and `gate-playable-interaction` remain green and keep
  their current evidence contract.
- True recorded placement has hard selection priority. The current demo fallback
  is always labelled `reversed_pickup`.
- Reverse playback starts at the end of the earliest certified five-sample
  Hold/contact window, never blindly from a clip's final frame.
- Terrain, terrain IK, learned control, doors, drawers, and shelf-cavity clearance
  are out of scope.
- Every implementation task follows RED -> GREEN -> focused regression -> commit.

---

### Task 1: Placement Surface and Slot Contracts

**Files:**
- Create: `interaction_place_target.h`
- Create: `interaction_place_target.cpp`
- Create: `tests/cpp/test_interaction_place_target.cpp`
- Modify: `interaction_matcher.h`
- Modify: `Makefile`

**Interfaces:**
- Produces: `SurfaceHandle`, `PlaceAffordance`, `PlacementSurface`,
  `PlacementSurfaceRegistry`, `PlaceRequest`, `placement_goal_world`, and
  `evaluate_placement_fit`.
- Consumes: `Transform`, `vec3`, quaternion helpers, and the existing exception
  conventions from `interaction_target.*`.

- [ ] **Step 1: Write the failing public-contract and validation tests**

Add compile-time and runtime assertions equivalent to:

```cpp
static_assert(std::is_same_v<
    decltype(PlaceAffordance{}.object_in_surface), Transform>);

PlacementSurface surface = make_surface();
PlacementSurfaceRegistry registry;
const SurfaceHandle first = registry.upsert(surface);
assert(first == (SurfaceHandle{41U, 1U}));
assert(registry.find(first) != nullptr);

const Transform goal = placement_goal_world(
    *registry.find(first), registry.find_affordance(first, 7U)->object_in_surface);
assert(near(goal.position, vec3(0.25F, 0.82F, 3.50F)));

const PlacementFit exact = evaluate_placement_fit(
    *registry.find(first),
    *registry.find_affordance(first, 7U),
    vec3(0.20F, 0.30F, 0.20F));
assert(exact.accepted);
assert(exact.support_gap_m == 0.020F);

PlacementSurface outside = surface;
outside.affordances[0].object_in_surface.position.x += 0.000001F;
assert(!evaluate_placement_fit(
    outside, outside.affordances[0], vec3(0.20F, 0.30F, 0.20F)).accepted);
assert(evaluate_placement_fit(
    outside, outside.affordances[0], vec3(0.20F, 0.30F, 0.20F)).reason ==
       Reason::PlacementOutOfBounds);
```

Cover zero IDs/generations, duplicate affordance IDs, non-finite transforms,
non-unit quaternions, non-positive extents, negative clearance, non-unit approach
axis, surface tilt exactly 5 degrees and 5.001 degrees, support gaps exactly
`-0.005`/`+0.020 m` and one micrometre outside, oriented-footprint boundary and
boundary-plus-epsilon, stale handles, and generation increment on `upsert`.

- [ ] **Step 2: Run RED**

Run:

```bash
make build/tests/test_interaction_place_target
```

Expected: compilation fails because `interaction_place_target.h` does not exist.

- [ ] **Step 3: Implement the exact surface API**

Expose:

```cpp
struct SurfaceHandle {
    uint64_t id = 0;
    uint32_t generation = 0;
    friend constexpr bool operator==(
        SurfaceHandle left, SurfaceHandle right) {
        return left.id == right.id && left.generation == right.generation;
    }
};

struct PlaceAffordance {
    uint32_t id = 0;
    Transform object_in_surface{};
    vec3 support_point_object{};
    vec3 approach_direction_surface{};
    float clearance_radius = 0.04F;
};

struct PlacementSurface {
    SurfaceHandle handle{};
    Transform surface_world{};
    float half_extent_x_m = 0.0F;
    float half_extent_z_m = 0.0F;
    float overhead_clearance_m = 0.0F;
    std::vector<PlaceAffordance> affordances;
};

struct PlacementFit {
    bool accepted = false;
    Reason reason = Reason::None;
    float support_gap_m = 0.0F;
};

class PlacementSurfaceRegistry {
public:
    SurfaceHandle upsert(PlacementSurface surface);
    const PlacementSurface* find(SurfaceHandle handle) const;
    const PlacementSurface* find_by_id(uint64_t id) const;
    const PlaceAffordance* find_affordance(
        SurfaceHandle handle, uint32_t affordance_id) const;
    std::optional<SurfaceHandle> resolve_single_surface(
        vec3 character_root, float maximum_distance_m) const;
};

struct PlaceRequest {
    TargetHandle held_target{};
    SurfaceHandle surface{};
    uint32_t affordance_id = 0;
    uint64_t request_id = 0;
};

Transform placement_goal_world(
    const PlacementSurface&, const Transform& object_in_surface);
PlacementFit evaluate_placement_fit(
    const PlacementSurface&,
    const PlaceAffordance&,
    vec3 object_dimensions);
```

Use an oriented-box footprint projection into surface X/Z, including
`clearance_radius`. Define the support gap as the surface-local Y coordinate of
`compose(affordance.object_in_surface,
Transform{affordance.support_point_object, quat()})`; accept it inclusively in
`[-0.005F, 0.020F]`. Require the surface normal angle to world up to be at most
`0.087266463F` radians.

`resolve_single_surface` measures planar distance to each composed affordance
goal, ignores invalid or out-of-range surfaces, and returns a handle only when
exactly one surface qualifies.

Append these `Reason` values after existing values without renumbering them:

```text
SurfaceUnavailable, SurfaceChanged, PlacementOutOfBounds,
ReleasePosition, ReleaseOrientation
```

- [ ] **Step 4: Run focused and safe GREEN**

Run:

```bash
make build/tests/test_interaction_place_target
build/tests/test_interaction_place_target
make test-interaction-safe
```

Expected: all commands exit 0; Python reports 259 or more tests with only the
existing environment-dependent skips; all C++ and fast-math binaries pass.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place_target.h interaction_place_target.cpp \
  tests/cpp/test_interaction_place_target.cpp interaction_matcher.h Makefile
git commit -m "feat: define deterministic placement surfaces"
```

Review checkpoint: validate transform conventions, inclusive numeric boundaries,
and that no target-selection policy leaked into runtime code.

### Task 2: Atomic Held-to-Free Placement Commit

**Files:**
- Modify: `interaction_target.h`
- Modify: `interaction_target.cpp`
- Modify: `interaction_attachment.h`
- Modify: `interaction_attachment.cpp`
- Modify: `tests/cpp/test_interaction_target.cpp`
- Modify: `tests/cpp/test_interaction_attachment.cpp`

**Interfaces:**
- Produces: `TargetRegistry::place_held` and
  `AttachmentController::commit_place`.
- Consumes: the exact Held target generation and original pickup owner already
  retained by `AttachmentController`.

- [ ] **Step 1: Write failing transactional release tests**

Add assertions equivalent to:

```cpp
const Transform placed{vec3(1.0F, 0.82F, 4.0F), quat()};
const TargetHandle old_handle = fixture.request.target;
const auto next = fixture.attachment.commit_place(placed);
assert(next.has_value());
assert(next->id == old_handle.id);
assert(next->generation == old_handle.generation + 1U);
assert(fixture.attachment.state() == ObjectState::Free);
assert(fixture.attachment.result() == ResultCode::Succeeded);
assert(exact(fixture.attachment.object_world(), placed));
const InteractionTarget* stored = fixture.registry.find(*next);
assert(stored != nullptr && stored->state == ObjectState::Free);
assert(stored->owner_request == 0U);
assert(exact(stored->object_world, placed));
```

Also prove wrong owner, stale generation, non-Held state, invalid transform, and
generation overflow cannot partially mutate pose/state/owner/generation. Call
`commit_place` twice and prove the second call fails without changing the first
commit.

- [ ] **Step 2: Run RED**

```bash
make build/tests/test_interaction_target build/tests/test_interaction_attachment
```

Expected: compilation fails because the two commit methods are absent.

- [ ] **Step 3: Implement a single mutation boundary**

Add:

```cpp
std::optional<TargetHandle> TargetRegistry::place_held(
    TargetHandle held,
    uint64_t owner_request,
    Transform placed_world);

std::optional<TargetHandle> AttachmentController::commit_place(
    Transform placed_world);
```

Validate `placed_world` before locating or mutating the target. Require exact
`Held` state and owner. On success increment generation, write the supplied pose,
set `Free`, clear owner, and return the new handle. Do not call `release()` and
then `replace_pose()`. `AttachmentController` updates its local pose/state/result
only after the registry transaction succeeds.

- [ ] **Step 4: Run focused, optimized, and safe GREEN**

```bash
make build/tests/test_interaction_target build/tests/test_interaction_attachment
build/tests/test_interaction_target
build/tests/test_interaction_attachment
make test-interaction-target-release-fast-math
make test-interaction-safe
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_target.h interaction_target.cpp interaction_attachment.h \
  interaction_attachment.cpp tests/cpp/test_interaction_target.cpp \
  tests/cpp/test_interaction_attachment.cpp
git commit -m "feat: commit placed objects atomically"
```

Review checkpoint: prove every failure path is mutation-free and the successful
registry/object pose is exactly the supplied hand-derived pose.

### Task 3: Recorded-Priority Place Selection and Certified Reverse Playback

**Files:**
- Create: `interaction_place.h`
- Create: `interaction_place.cpp`
- Create: `tests/cpp/test_interaction_place.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `Database`, retained pickup `MatchCandidate`, held
  `GraspAffordance`, current Carry pose/object, and a requested placement goal.
- Produces: `PlaceMotionMode`, `PlacePhase`, `RecordedPlaceClip`,
  `PlaceMotionLibrary`, `PlaceCandidate`, `select_place_motion`, and
  `PlacePlayer`.

- [ ] **Step 1: Write failing selector and reverse-player tests**

Freeze these public contracts:

```cpp
enum class PlaceMotionMode : uint8_t {
    None, RecordedPlace, ReversedPickup,
};
enum class PlacePhase : uint8_t {
    Align, Lower, Release, Retract, Finished,
};

struct RecordedPlaceClip {
    uint64_t id = 0;
    uint32_t fps_numerator = 25;
    uint32_t fps_denominator = 1;
    std::vector<Pose> poses;
    std::vector<Transform> object_poses;
    int32_t entry_frame = -1;
    int32_t commit_frame = -1;
    int32_t release_frame = -1;
    int32_t retract_stop_frame = -1;
    Hand hand = Hand::Right;
    Transform hand_in_object{};
};

struct PlaceMotionLibrary {
    std::vector<RecordedPlaceClip> recorded;
};

struct PlaceMatchInput {
    const Database* pickup_database = nullptr;
    const PlaceMotionLibrary* library = nullptr;
    MatchCandidate pickup_candidate{};
    Pose current_pose{};
    Transform current_object_world{};
    GraspAffordance held_affordance{};
    PlacementSurface surface{};
    PlaceAffordance place_affordance{};
    vec3 object_dimensions{};
};
```

Tests must prove:

```cpp
const PlaceResult preferred = select_place_motion(input_with_both_modes());
assert(preferred.accepted);
assert(preferred.candidate.mode == PlaceMotionMode::RecordedPlace);

Database late_contact_loss = reversible_pickup_database();
late_contact_loss.hand_contacts[(late_contact_loss.range_stops[0] - 1) * 2 + 1]
    = 0U;
const PlaceResult fallback = select_place_motion(
    reverse_only_input(late_contact_loss));
assert(fallback.accepted);
assert(fallback.candidate.mode == PlaceMotionMode::ReversedPickup);
assert(fallback.candidate.entry_frame == expected_stable_window_stop);
assert(fallback.candidate.entry_frame !=
       late_contact_loss.range_stops[0] - 1);
```

Advance a 1.0x fallback player for five `0.04F` updates and assert source frames
are exactly `start, start-1, ..., start-5`; pose velocity, angular velocity, and
hand-DOF velocity are the negatives of the source sample; foot contacts equal the
source frame; release occurs exactly at `contact_frame`; and sampling never
crosses `entry_frame`. Make the first Hold window fail stability and the second
pass; assert the second window's last frame is selected. Add rejection fixtures
for contact loss between `contact_frame` and the selected sample, no stable Hold
window, 2.0001 cm within-window hand/object drift, 10.001-degree within-window
drift, malformed events, wrong hand, and correction-limit-plus-epsilon.

- [ ] **Step 2: Run RED**

```bash
make build/tests/test_interaction_place
```

Expected: compilation fails because `interaction_place.h` does not exist.

- [ ] **Step 3: Implement deterministic two-tier selection**

Expose:

```cpp
struct PlaceCandidate {
    PlaceMotionMode mode = PlaceMotionMode::None;
    int32_t clip = -1;
    int32_t entry_frame = -1;
    int32_t commit_frame = -1;
    int32_t release_frame = -1;
    int32_t stop_frame = -1;
    int32_t direction = 0;
    Transform scene_from_source{};
    vec3 entry_root_offset{};
    float entry_yaw_offset = 0.0F;
    float total_cost = 0.0F;
};

struct PlaceResult {
    bool accepted = false;
    PlaceCandidate candidate{};
    Reason reason = Reason::None;
};

struct PlaceSample {
    Pose pose{};
    Transform source_object{};
    int32_t source_frame = -1;
    PlacePhase phase = PlacePhase::Align;
};
```

Validate every in-memory recorded clip as a 25 Hz contiguous pose/object sequence
with `entry <= commit < release < retract_stop`, finite channels, valid
quaternions, and a compatible active hand. Evaluate all compatible true-place
rows first. Return the minimum-cost accepted recorded candidate without comparing
it to fallback cost. Only when that tier is empty or every recorded row fails a
hard filter, scan five-sample windows wholly inside Hold and select the last frame
of the earliest window with continuous contact and within-window hand-in-object
drift `<=0.02F` / `<=0.174532925F`. Also require active contact continuously
from the semantic release event through that selected sample.

For true-place candidates, map the source release object pose to the requested
goal. For fallback candidates, map the source active hand at `contact_frame` to
`placement_goal_world * held_affordance.hand_in_object`; do not drive the held
object from the source object trajectory. `PlacePlayer` uses double-precision
elapsed/source accumulators, exact 25 Hz, shortest-arc pose interpolation,
direction-aware finish checks, and sign-negated derivative channels for direction
`-1`. Do not inspect or depend on frames after the certified reverse start.

Expose `PlacePlayer::start(const PlaceCandidate&, const PlaceMatchInput&, float
speed)`, `advance(float dt)`, `PlaceSample sample()`, `source_frame()`, `phase()`,
`release_due()`, and `finished()`. The player stores only validated pointers or
copies whose lifetime is guaranteed by its owning `PlaceController`.

- [ ] **Step 4: Run focused and safe GREEN**

```bash
make build/tests/test_interaction_place
build/tests/test_interaction_place
make test-interaction-safe
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place.h interaction_place.cpp \
  tests/cpp/test_interaction_place.cpp Makefile
git commit -m "feat: select deterministic place motions"
```

Review checkpoint: confirm recorded tier priority is structural, fallback reads no
late clip frame, and every reverse derivative has correct time direction.

### Task 4: Bounded Place Controller and Release Gate

**Files:**
- Create: `interaction_place_controller.h`
- Create: `interaction_place_controller.cpp`
- Create: `tests/cpp/test_interaction_place_controller.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: selected `PlaceCandidate`, current Carry pose/object, held grasp,
  surface/affordance, `IKConfig`, and exact 25 Hz updates.
- Produces: `PlaceControllerConfig`, `PlaceBeginInput`, `PlaceStep`, and
  `PlaceController`.

- [ ] **Step 1: Write failing correction, release, and recovery tests**

Freeze configuration:

```cpp
struct PlaceControllerConfig {
    float canonical_fps = 25.0F;
    float playback_speed = 1.0F;
    float entry_blend_seconds = 0.25F;
    float maximum_alignment_seconds = 1.00F;
    float commit_horizon_seconds = 0.50F;
    float maximum_root_correction_m = 0.25F;
    float maximum_yaw_correction_radians = 0.436332313F;
    float maximum_hand_correction_m = 0.12F;
    float maximum_hand_orientation_radians = 0.436332313F;
    float release_position_m = 0.02F;
    float release_orientation_radians = 0.174532925F;
};
```

Tests assert exact boundaries are accepted and `std::nextafter(boundary, +inf)`
is rejected. On every attached step assert:

```cpp
assert(near(
    step.object_world,
    compose(hand_world(step.pose, held.hand), inverse(held.hand_in_object))));
```

At the release step assert `release_due` is true exactly once, final position and
orientation errors are within `0.02 m / 10 degrees`, and `object_world` is still
hand-derived. After acknowledging release, assert every retraction step returns
the identical fixed object transform. A failed release gate must set
`recover_to_carry`, retain attachment, and never emit `release_due` later.

Add exact `dt` rejection for `0.0F`, `1/60`, NaN, and the adjacent float around
`1/25`; only the exact constant passes.

- [ ] **Step 2: Run RED**

```bash
make build/tests/test_interaction_place_controller
```

Expected: compilation fails because the controller files do not exist.

- [ ] **Step 3: Implement the focused controller**

Expose:

```cpp
struct PlaceStep {
    Pose pose{};
    Transform object_world{};
    PlacePhase phase = PlacePhase::Align;
    int32_t source_frame = -1;
    bool committed = false;
    bool release_due = false;
    bool retract_finished = false;
    bool recover_to_carry = false;
    Reason reason = Reason::None;
    float hand_position_error_m = 0.0F;
    float hand_orientation_error_radians = 0.0F;
};

struct PlaceBeginInput {
    const Database* pickup_database = nullptr;
    const PlaceMotionLibrary* library = nullptr;
    PlaceCandidate candidate{};
    Pose current_pose{};
    Transform current_object_world{};
    GraspAffordance held_affordance{};
    PlacementSurface surface{};
    PlaceAffordance place_affordance{};
    vec3 object_dimensions{};
};

struct PlaceBeginResult {
    bool accepted = false;
    Reason reason = Reason::None;
};

class PlaceController {
public:
    PlaceBeginResult begin(const PlaceBeginInput& input);
    PlaceStep update(float dt);
    void acknowledge_release(Transform placed_world);
    void cancel();
};
```

Blend entry for exactly seven 25 Hz ticks, distribute planar root/yaw correction
with smoothstep to zero by release, and ramp bounded hand IK to full release
weight. Check the object sweep against the destination support volume, exempting
only the final support contact. Keep `last_safe_pose` and the hand-derived object
for attached recovery. After `acknowledge_release`, freeze exactly the supplied
transform while playback retracts.

- [ ] **Step 4: Run focused, fast-math, and safe GREEN**

```bash
make build/tests/test_interaction_place_controller
build/tests/test_interaction_place_controller
make test-interaction-carry-release-fast-math
make test-interaction-safe
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place_controller.h interaction_place_controller.cpp \
  tests/cpp/test_interaction_place_controller.cpp Makefile
git commit -m "feat: play bounded tabletop placement"
```

Review checkpoint: verify correction limits are hard gates, object ownership has
one source in each phase, and recovery never detaches.

### Task 5: Integrate Placement into InteractionRuntime

**Files:**
- Modify: `interaction_runtime.h`
- Modify: `interaction_runtime.cpp`
- Modify: `interaction_debug_draw.h`
- Modify: `tests/cpp/interaction_runtime_fixture.h`
- Modify: `tests/cpp/test_interaction_runtime.cpp`

**Interfaces:**
- Consumes: placement surface registry, place library/controller, existing pickup
  candidate, attachment owner, and `RuntimeInput::place_request`.
- Produces: appended runtime states, place diagnostics, and full place lifecycle.

- [ ] **Step 1: Write failing API and state-machine tests**

Append enum values and freeze old values:

```cpp
static_assert(static_cast<uint8_t>(RuntimeState::Carry) == 6U);
static_assert(static_cast<uint8_t>(RuntimeState::PlacePreflight) == 7U);
static_assert(static_cast<uint8_t>(RuntimeState::PlaceAlign) == 8U);
static_assert(static_cast<uint8_t>(RuntimeState::PlaceReplay) == 9U);
static_assert(static_cast<uint8_t>(RuntimeState::PlaceRelease) == 10U);
```

Run pickup to Carry, pulse Interact with one `PlaceRequest`, and assert exact
collapsed success order:

```cpp
const std::array expected = {
    RuntimeState::Carry,
    RuntimeState::PlacePreflight,
    RuntimeState::PlaceAlign,
    RuntimeState::PlaceReplay,
    RuntimeState::PlaceRelease,
    RuntimeState::Locomotion,
};
```

Assert `PlacePreflight` occupies one update, all place states own pose and
suppress steering, attachment remains true through the last PlaceReplay record,
and the first PlaceRelease record is Free/unattached with a generation increment
of one. Registry, runtime output, and scene object transforms must compare exact.

Add independent tests for: missing request, stale surface, wrong held object,
cancel in PlaceAlign, candidate rejection, post-commit release-position failure,
surface replacement before release, duplicate Place edge, reset during placement,
recorded-mode diagnostics, reversed-mode diagnostics, and deterministic replay.
Every pre-release failure returns Carry attached; no failure returns Locomotion
with a silently dropped object.

For every `Place*` state, compare otherwise identical updates with
`reset_pressed=true` and `reset_pressed=false`; their outputs and subsequent
state must match exactly. Normal placement progress may continue, but Reset may
not add a pause, transition, pose mutation, attachment mutation, target-generation
change, or source-frame discontinuity. Existing Carry Reset behavior remains
unchanged before a Place request is accepted.

- [ ] **Step 2: Run RED**

```bash
make build/tests/test_interaction_runtime
```

Expected: compilation fails because the new runtime states and request field are
absent.

- [ ] **Step 3: Add placement dependencies without breaking pickup callers**

Retain the existing constructor and add an overload:

```cpp
InteractionRuntime(
    const Database&,
    const Features&,
    TargetRegistry&,
    PlacementSurfaceRegistry&,
    const PlaceMotionLibrary&,
    RuntimeConfig);
```

The old constructor leaves placement unavailable but preserves every pickup test.
Add `std::optional<PlaceRequest> place_request` to `RuntimeInput` and the surface,
mode, release, goal, and support-error fields to diagnostics.
Add `PlaceControllerConfig place{}` to `RuntimeConfig` and validate every field at
construction with the same mutation-free exception behavior as the current
matcher/playback/IK/carry configuration.

In Carry, an Interact edge publishes `PlacePreflight` without advancing Carry or
mutating attachment. Validate on the following update. Delegate pose generation
to `PlaceController`. At `release_due`, run the final gate and call
`AttachmentController::commit_place(step.object_world)` once. Pass that same
transform to `acknowledge_release`, publish the returned target generation, then
continue retraction. Cancellation or pre-release failure restores Carry without
restarting its object transform.

Update debug names exhaustively and retain `ResultCode::Succeeded/Reason::None`
through successful PlaceRelease and final Locomotion.

- [ ] **Step 4: Run focused and safe GREEN**

```bash
make build/tests/test_interaction_runtime
build/tests/test_interaction_runtime
make test-interaction-safe
```

Expected: all commands exit 0 and all prior pickup runtime tests remain green.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_runtime.h interaction_runtime.cpp \
  interaction_debug_draw.h tests/cpp/interaction_runtime_fixture.h \
  tests/cpp/test_interaction_runtime.cpp
git commit -m "feat: coordinate carry to place lifecycle"
```

Review checkpoint: audit every place transition, failure terminal, output
ownership flag, attachment mutation, and target-generation update.

### Task 6: Wire Manual Pick/Place and the Destination Table

**Files:**
- Modify: `interaction_controller_adapter.h`
- Modify: `interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `tests/python/test_playable_interaction_evidence.py`
- Modify: `controller.cpp`
- Modify: `interaction_debug_draw.h`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: current runtime state, one placement surface/slot, shared Interact
  edge, runtime placement output, and scene handoff.
- Produces: `F` pick/place behavior, destination-table drawing, and continuous
  release publication.

- [ ] **Step 1: Write failing adapter and controller-policy tests**

In C++, prove all new place states keep the runtime ownership epoch active, full
body place poses are not treated as layered Carry, and the first free scene sample
equals the final attached runtime object transform bit-for-bit. Prove a new target
generation ends the hand constraint without applying an IK solve to the free
object, while the normal 0.25-second pose release starts from the last displayed
PlaceRelease pose.

Add Python static policy tests asserting:

- controller constructs exactly one destination `PlacementSurface`;
- `F` resolves `PickRequest` only in Locomotion and `PlaceRequest` only in Carry;
- no place state calls `resolve_single_target` to replace the held object;
- destination goal is composed from `surface_world` and `object_in_surface`;
- scene release does not call `reset` or restore the source pose; and
- `SetTargetFPS(25)` and the synchronous scheduler remain unchanged.

- [ ] **Step 2: Run RED**

```bash
make build/tests/test_interaction_controller_adapter
python -m unittest \
  tests.python.test_playable_interaction_evidence.Task12PolicyTests -v
```

Expected: new assertions fail because controller placement wiring is absent.

- [ ] **Step 3: Implement the thin controller seam**

Construct a destination table and top-plane surface once from the authored demo
scene. Copy the source table size/height and translate its center exactly `1.20 m`
farther along world `+Z`. Give it one top-center affordance whose object transform
and object-local support point reproduce the source object's known supported
origin offset. Derive that point from the last stable pre-lift sample by
projecting the source object origin along the source table normal onto the table
top plane, then transforming the projected point through the inverse source
object transform; do not infer support from mesh bounds. Pass an empty
`PlaceMotionLibrary::recorded` so the demo reports `ReversedPickup`.

Extend the scheduler with a place resolver while preserving the current pick
resolver API. On `F`, resolve exactly one surface within the 1.00 m local envelope
when the cached state is Carry. Draw the support rectangle, final object frame,
approach axis, release errors, and place mode. Change help text to
`F pick/place  X cancel  R reset`.

During PlacePreflight/Align/Replay keep scene authority on the attached runtime
object. On atomic release, refresh the object handle by stable ID and let the Free
registry pose take authority. Do not interpolate or overwrite that pose.

- [ ] **Step 4: Run focused build and safe GREEN**

```bash
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
make controller
make test-interaction-safe
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_controller_adapter.h interaction_controller_adapter.cpp \
  tests/cpp/test_interaction_controller_adapter.cpp controller.cpp \
  tests/python/test_playable_interaction_evidence.py interaction_debug_draw.h \
  Makefile README.md
git commit -m "feat: control tabletop placement in scene"
```

Review checkpoint: inspect target/constraint identity across generation change and
prove manual pickup behavior is unchanged when no place request is active.

### Task 7: Add the Focused Headless Place Gate

**Files:**
- Create: `interaction_place_probe.cpp`
- Create: `tests/python/test_place_probe.py`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: deterministic fixture pack, pickup runtime, Carry, one placement
  surface, and the reversed fallback.
- Produces: `interaction_place_probe <pack> --json` and
  `make gate-place-headless`.

- [ ] **Step 1: Write failing probe-output tests**

Require one compact sorted JSON record:

```json
{
  "attachment_transitions": 1,
  "final_attached": false,
  "final_object_state": "Free",
  "final_reason": "None",
  "final_result": "Succeeded",
  "mode": "reversed_pickup",
  "release_frame": 139,
  "reverse_start_frame": 181,
  "state_sequence": [
    "Carry", "PlacePreflight", "PlaceAlign", "PlaceReplay",
    "PlaceRelease", "Locomotion"
  ]
}
```

Tests must derive expected contact and stable-window indices from the fixture
rather than assuming the illustrative numbers above. Make the first five-sample
Hold window unstable and the next one stable, assert the earliest certified
window wins, and mutate only the clip's last contact to zero while expecting the
probe to keep the same certified reverse start.

- [ ] **Step 2: Run RED**

```bash
make interaction_place_probe
python -m unittest tests.python.test_place_probe -v
```

Expected: build fails because the probe source is absent.

- [ ] **Step 3: Implement the deterministic probe and target**

The probe loads the existing interaction pack, performs pickup to Carry, submits
one explicit `PlaceRequest`, advances only with `dt=1/25`, and checks every state
and attachment transition before printing JSON. It exits nonzero on rejection,
failure, timeout, non-contiguous source frames, unexpected mode, or release-pose
mismatch.

Add:

```make
.PHONY: gate-place-headless
gate-place-headless: test-interaction-safe interaction_place_probe
	./interaction_place_probe "$(INTERACTION_DEMO_PACK)" --json
	python -m unittest tests.python.test_place_probe -v
```

The target must use only the existing safe interaction suite and never invoke the
protected root query probe.

- [ ] **Step 4: Run GREEN**

```bash
make demo-interaction-pack
make gate-place-headless
```

Expected: safe tests pass, probe exits 0, mode is `reversed_pickup`, and the exact
place state sequence is reported.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place_probe.cpp tests/python/test_place_probe.py \
  Makefile README.md
git commit -m "test: gate headless tabletop placement"
```

Review checkpoint: run the last-frame-contact mutation and verify the probe still
uses only the earlier certified Hold sample.

### Task 8: Prove Genuine Walk, Pickup, Carry, and Place Graphically

**Files:**
- Create: `tests/python/test_playable_placement_evidence.py`
- Modify: `controller.cpp`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: ordinary flat controller input, placement runtime, destination scene,
  X11 display, and final rendered FK.
- Produces: `MM_INTERACTION_PLACE_AUTODEMO=1`, atomic placement JSONL/PNG
  evidence, and `make gate-playable-placement`.

- [ ] **Step 1: Write the failing placement evidence validator**

Define exact state and action contracts:

```python
EXPECTED_STATES = (
    "Locomotion", "Preflight", "Align", "PickupReplay", "Hold", "Carry",
    "PlacePreflight", "PlaceAlign", "PlaceReplay", "PlaceRelease",
    "Locomotion",
)
CONTROL_RATE_HZ = 25
MIN_WALK_TICKS = 25
MIN_WALK_DISPLACEMENT_M = 0.50
INITIAL_PICKUP_DISTANCE_M = 1.25
INTERACT_DISTANCE_M = 0.80
CARRY_COMMAND_COUNT = 63
```

The synthetic fixture and validator must reject:

- initial pickup distance `<=1.25 m`;
- fewer than 25 consecutive initial `walk` rows;
- any owned pose, attachment, non-Locomotion state, or canonical-snapshot flag in
  the walking prefix;
- displayed-root walk displacement `<0.50 m`;
- Interact when pickup distance `>0.80 m`;
- skipped/duplicated 25 Hz runtime ticks or nonzero scheduler phase;
- fewer or more than 63 consecutive Carry movement rows;
- Carry root or object displacement `<=0.20 m`;
- missing/duplicate Place action or any Reset action;
- release before PlaceRelease or more than one attached-to-free edge;
- final position error `>0.02 m`, orientation error `>10 degrees`, or support gap
  outside `[-0.005 m, +0.020 m]`;
- final state other than `Locomotion/Succeeded/None/Free/unattached`;
- place mode other than `recorded_place` or `reversed_pickup`; and
- the existing distal-foot 12 m/s, authority-seam 0.20 m, or 60-degree joint
  rotation limits.

- [ ] **Step 2: Run RED against missing evidence**

```bash
PLACEMENT_LOG=/tmp/missing-placement.jsonl \
PLACEMENT_SCREENSHOT=/tmp/missing-placement.png \
python -m unittest tests.python.test_playable_placement_evidence -v
```

Expected: unit tests pass and the environment-enabled real-evidence test fails
because both files are absent.

- [ ] **Step 3: Implement a separate placement auto-demo**

Add `MM_INTERACTION_PLACE_AUTODEMO=1`; do not change the existing pickup
auto-demo. Its input sequence is state-driven and bounded:

1. Record the initial displayed-root-to-object planar distance and require it to
   exceed `1.25 m`.
2. Command the ordinary left-stick locomotion seam for at least 25 ticks. During
   this prefix use the real flat-controller `LocomotionSnapshot`; never call or
   substitute `make_autodemo_canonical_entry`, and never let the interaction
   runtime own the pose.
3. Continue ordinary walking toward world `+Z` until displayed-root displacement
   is at least `0.50 m` and pickup distance is at most `0.80 m`; only then pulse
   Interact. Convert the requested world direction through the existing camera
   basis to ordinary left-stick input; do not write the root directly.
4. Complete pickup, command exactly 63 ordinary Carry movement ticks toward world
   `+Z` and the destination table 1.20 m past the source, and require root/object
   displacement above `0.20 m`.
5. Pulse Place once when the destination resolver returns its exact slot.
6. Complete PlaceRelease, drain the normal seven-frame visual handoff without
   input, capture the placed object, publish evidence atomically, and exit.

Bound walking to 250 ticks, pickup-to-Carry to 375 ticks, placement to 250 ticks,
and total evidence to 900 records. Any timeout, window close, rejected runtime
result, fallback mode mislabel, or evidence I/O failure exits nonzero.

Log the current pickup distance, walking-origin displacement, canonical-snapshot
flag, place goal/error/gap, attachment transition count, place source/mode, and
the same final-FK joint/grasp data used by the pickup evidence.

Require the canonical-snapshot flag to remain false on every placement auto-demo
record, including pickup matching after Interact. The existing pickup-only
auto-demo retains its current canonical-fixture behavior.

- [ ] **Step 4: Add the isolated graphical gate**

Add configurable paths:

```make
PLACEMENT_EVIDENCE_DIR ?= playable-evidence/placement
PLACEMENT_LOG_PATH ?= $(PLACEMENT_EVIDENCE_DIR)/placement.jsonl
PLACEMENT_SCREENSHOT_PATH ?= $(PLACEMENT_EVIDENCE_DIR)/placement.png
PLACEMENT_FEATURES_OUTPUT ?= $(PLACEMENT_EVIDENCE_DIR)/locomotion-features.bin
```

Add a target that depends on `gate-place-headless`, then serially bootstraps and
builds the controller, checks `DISPLAY`, runs the controller with a 60-second
external timeout, verifies tracked `resources/features.bin` is unchanged, and
runs the placement validator with its two evidence paths. It must not depend on
or invoke the unsafe root query probe.

- [ ] **Step 5: Run the real graphical gate**

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-placement
```

Expected: headless gate and safe suite pass; the controller exits 0; evidence
shows genuine non-owned flat walking before Interact, then pickup, 63 Carry
commands, one `reversed_pickup` place, one release, and final supported object;
the placement validator exits 0.

- [ ] **Step 6: Inspect visual evidence**

Capture a native-25-Hz lossless video through the working nested-display method
used by the pickup gate. Inspect approach, pickup entry/contact, moving Carry,
PlaceAlign, lowering, release, retraction, and locomotion handoff. Reject the gate
if the object, root, head/neck, active elbow, inactive arm, or skeleton visibly
flips even when numeric limits pass.

- [ ] **Step 7: Document and commit**

README must include manual `F pick/place`, exact headless and graphical commands,
evidence paths, current expected `reversed_pickup` mode, the absence of local true
place data, and the flat-ground/non-terrain scope.

```bash
git add tests/python/test_playable_placement_evidence.py controller.cpp \
  Makefile README.md
git commit -m "test: prove walk pick carry and place"
```

Review checkpoint: independently inspect the lossless video and JSONL. A passing
synthetic or headless probe cannot substitute for genuine ordinary flat walking
in the graphical evidence.

## Final Verification and Branch Review

- [ ] Run the complete safe and place gates from the placement branch:

```bash
make test-interaction-safe
make gate-place-headless
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-placement
```

- [ ] Verify the existing pickup graphical gate remains unchanged and green:

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-interaction
```

- [ ] Run static branch checks:

```bash
git diff --check c2b6ff8..HEAD
git status --short --untracked-files=no
```

- [ ] Dispatch a fresh whole-branch spec reviewer and code-quality reviewer with
  the design, plan, commit range, headless JSON, placement JSONL/PNG/video, and all
  verification output. Fix every Critical or Important finding in one reviewed
  correction wave and rerun every affected gate.

- [ ] Hand off the isolated branch without merging into terrain. The next design
  after this gate is a parameterized height/depth/yaw sweep; constrained shelves
  and articulated objects remain separate specifications.
