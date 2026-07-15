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
gate unchanged. Add explicit object-local bounds, a separate placement-surface
registry with physical support context, a read-only selected-motion staging
preview, an atomic Held-to-Free pose-and-support commit, and a focused place
controller that supplies one contiguous recorded or reversed motion to
`InteractionRuntime`. Runtime appends four place states; the controller reaches
the selected staging root through ordinary Carry locomotion and publishes the
same object transform across the attachment-release seam.

**Tech Stack:** C++17, existing G1 interaction database and FK/IK code, Raylib,
Python `unittest`, GNU Make, fixed 25 Hz runtime.

## Global Constraints

- Before Task 1, create a fresh isolated worktree from the then-latest reviewed
  `g1-manipulation-motion-matching` HEAD, including the reviewed elbow and lower-
  handoff visual fixes. That HEAD must contain
  `38a02137a60bfb0c2c4ac6ca9f4979aaad613f26` as an ancestor. This docs correction
  is based on `38a0213` only for clean cherry-pick; implementation must record its
  newer start hash and must not reset back to `38a0213`, an older placement-doc
  commit, or terrain.
- Work only on `g1-tabletop-placement-*` branches/worktrees; do not touch or
  merge the terrain checkout.
- Do not access, stat, hash, execute, modify, stage, or delete the protected
  repository-root artifact `interaction_query_probe`. Use
  `build/task12/interaction_query_probe_safe` through the existing safe targets.
- Runtime, flat locomotion, interaction playback, auto-demo, and evidence all
  advance exactly once per `0.04` second.
- Existing pickup behavior and `gate-playable-interaction` remain green and keep
  their current evidence contract.
- True recorded placement has hard selection priority. The current demo fallback
  is always labelled `reversed_pickup`.
- Reverse playback starts at the end of the earliest certified five-sample
  Hold/contact window, never blindly from a clip's final frame.
- A placement commit atomically replaces both object pose and pickup support
  context; a released target must be immediately re-pickable from the destination.
- Requested-goal fit never authorizes release by itself. The exact hand-derived
  pose must pass support/footprint/bounds/clearance checks at the release event.
- Assisted staging uses ordinary Carry input and a read-only selected-motion
  preview; it never writes the character root.
- Intrinsic candidate certification never rejects or demotes a row for current
  entry distance. Preview may be accepted-but-not-ready; only frozen preflight
  enforces the `0.25 m` / `25 degree` entry bounds.
- The frozen auto-demo uses its retained authored destination handle regardless of
  distance. The `1.00 m` surface resolver remains a manual-only nearby convenience.
- Selection identity includes the held target and complete frozen current
  pose/object snapshot; a pre-staging ID cannot authorize post-staging playback.
- Every candidate has one motion-commit event: authored for recorded place and
  deterministically derived for reverse, with maximum-align validation.
- Every allowed playback speed clamps to and publishes the exact release sample
  exactly once before retraction.
- Terrain, terrain IK, learned control, doors, drawers, and shelf-cavity clearance
  are out of scope.
- Every implementation task follows RED -> GREEN -> focused regression -> commit.

---

### Task 1: Placement Surface and Slot Contracts

Before RED, verify and record the isolated implementation base:

```bash
git merge-base --is-ancestor \
  38a02137a60bfb0c2c4ac6ca9f4979aaad613f26 HEAD
test "$(git rev-parse HEAD)" = \
  "$(git rev-parse g1-manipulation-motion-matching)"
git branch --show-current
PLACEMENT_BASE="$(git rev-parse HEAD)"
printf '%s\n' "$PLACEMENT_BASE"
```

The recorded `HEAD` becomes `PLACEMENT_BASE` for final branch review. Stop if it
is not the latest reviewed `g1-manipulation-motion-matching` tip containing the
elbow/lower-handoff fixes; do not repair that mismatch by resetting to the minimum
docs ancestor.

**Files:**
- Create: `interaction_place_target.h`
- Create: `interaction_place_target.cpp`
- Create: `tests/cpp/test_interaction_place_target.cpp`
- Modify: `interaction_target.h`
- Modify: `interaction_target.cpp`
- Modify: `tests/cpp/test_interaction_target.cpp`
- Modify: `tests/cpp/test_interaction_attachment.cpp`
- Modify: `tests/cpp/interaction_runtime_fixture.h`
- Modify: `interaction_runtime_probe.cpp`
- Modify: `interaction_controller_adapter.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `interaction_matcher.h`
- Modify: `Makefile`

**Interfaces:**
- Produces: `ObjectLocalBounds`, target object-profile/bounds metadata,
  `SurfaceHandle`, `PlaceAffordance`, `PlacementSurface`,
  `PlacementSurfaceRegistry`, `PlaceRequest`, `placement_goal_world`,
  `evaluate_placement_fit`, and `evaluate_actual_placement_fit`.
- Consumes: `Transform`, `vec3`, quaternion helpers, and the existing exception
  conventions from `interaction_target.*`; updates every mandatory
  `InteractionTarget` author that must satisfy the new profile/bounds invariant.

- [ ] **Step 1: Write the failing public-contract and validation tests**

Add compile-time and runtime assertions equivalent to:

```cpp
static_assert(std::is_same_v<
    decltype(PlaceAffordance{}.object_in_surface), Transform>);

PlacementSurface surface = make_surface();
const InteractionTarget target = make_target();
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
    ObjectLocalBounds{vec3(0.03F, 0.01F, -0.02F),
                      vec3(0.10F, 0.15F, 0.10F)});
assert(exact.accepted);
assert(exact.support_gap_m == 0.020F);

PlacementSurface outside = surface;
outside.affordances[0].object_in_surface.position.x += 0.000001F;
assert(!evaluate_placement_fit(
    outside, outside.affordances[0], target.object_bounds).accepted);
assert(evaluate_placement_fit(
    outside, outside.affordances[0], target.object_bounds).reason ==
       Reason::PlacementOutOfBounds);
```

Cover zero object-profile IDs, non-finite bounds centers, non-positive bound
half-extents, zero surface IDs/generations, duplicate affordance IDs, non-finite
transforms, non-unit quaternions, non-positive support-volume dimensions,
non-positive overhead clearance, non-unit approach axis, surface tilt exactly 5
degrees and 5.001 degrees, support-volume top face offset exactly `0.001 m` and
one micrometre beyond, usable half-extents exactly half the physical support X/Z
size and one float beyond, finite and non-finite support points, clearance radius
exactly zero, negative, NaN, infinity, and positive finite, support gaps exactly
`-0.005`/`+0.020 m` and one micrometre outside, nonzero bounds-center footprint
boundaries, oriented-footprint boundary and boundary-plus-epsilon, lowest-corner
penetration, highest-corner overhead violation, stale handles, and generation
increment on `upsert`.

Update and assert every mandatory target author in the same RED wave:

- `tests/cpp/test_interaction_target.cpp` target helpers;
- `tests/cpp/test_interaction_attachment.cpp` attachment fixture;
- `tests/cpp/interaction_runtime_fixture.h` shared runtime fixture;
- `interaction_runtime_probe.cpp::demo_target`;
- `interaction_controller_adapter.cpp::make_controller_demo_target`; and
- every direct target fixture plus factory assertions in
  `tests/cpp/test_interaction_controller_adapter.cpp`.

Each authors a nonzero stable object-profile ID plus a finite explicit
`ObjectLocalBounds`. Test/demo objects with a reviewed centered mesh use an
explicit zero center and half the measured `object_dimensions`; this is authored
fixture metadata, never a registry inference. The existing graphical pickup path
must continue to construct the same scene pose and dimensions.

- [ ] **Step 2: Run RED**

Run:

```bash
mkdir -p build/red
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_interaction_place_target.cpp \
  -o build/red/test_interaction_place_target.o
```

Expected: the compiler exits nonzero because `interaction_place_target.h` does
not exist. Do not use a nonexistent Make target as the RED signal; this
repository has explicit test rules and no generic C++ test pattern.

- [ ] **Step 3: Implement the exact surface API**

Expose:

```cpp
struct ObjectLocalBounds {
    vec3 center_object{};
    vec3 half_extents_object{};
};

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
    Transform support_volume_world{};
    vec3 support_volume_size{};
    float half_extent_x_m = 0.0F;
    float half_extent_z_m = 0.0F;
    float overhead_clearance_m = 0.0F;
    std::vector<PlaceAffordance> affordances;
};

struct PlacementFit {
    bool accepted = false;
    Reason reason = Reason::None;
    float support_gap_m = 0.0F;
    float lowest_corner_m = 0.0F;
    float highest_corner_m = 0.0F;
    bool footprint_valid = false;
    bool overhead_valid = false;
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
    uint64_t selection_id = 0;
};

Transform placement_goal_world(
    const PlacementSurface&, const Transform& object_in_surface);
PlacementFit evaluate_placement_fit(
    const PlacementSurface&,
    const PlaceAffordance&,
    ObjectLocalBounds);
PlacementFit evaluate_actual_placement_fit(
    const PlacementSurface&,
    const PlaceAffordance&,
    Transform actual_object_world,
    ObjectLocalBounds);
```

Define `ObjectLocalBounds` in `interaction_target.h`. Add nonzero
`object_profile_id` and validated `ObjectLocalBounds object_bounds` to
`InteractionTarget`; preserve `object_dimensions` for the existing matcher.
Update every target fixture with explicit values.

For either requested or actual fit, enumerate the eight points
`center_object + sign * half_extents_object`, transform them through the tested
object pose and into surface local space, and compute inclusive X/Z footprint,
lowest Y, and highest Y. Expand X/Z by `clearance_radius`. Define support gap
from the transformed authored support point and accept it inclusively in
`[-0.005F, 0.020F]`; require lowest Y `>=-0.005F` and highest Y
`<=overhead_clearance_m`. `evaluate_placement_fit` tests
`surface_world * object_in_surface`; `evaluate_actual_placement_fit` first maps
the supplied world pose through `inverse(surface_world)`. Require surface normal
angle at most `0.087266463F` radians and require the support-volume top face to
coincide with the support plane within `0.001F m` / `0.001745329F rad`.

Require `half_extent_x_m <= 0.5F * support_volume_size.x` and the equivalent Z
bound, inclusively. Validate finite `support_point_object` and finite
`clearance_radius >= 0.0F`. `resolve_single_surface` measures planar distance to
each composed affordance goal, ignores invalid or out-of-range surfaces, and
returns a handle only when exactly one surface qualifies; later tasks use it only
for manual convenience.

Append these `Reason` values after existing values without renumbering them:

```text
SurfaceUnavailable, SurfaceChanged, PlacementOutOfBounds,
ReleasePosition, ReleaseOrientation
```

- [ ] **Step 4: Run focused and safe GREEN**

Run:

```bash
make build/tests/test_interaction_place_target \
  build/tests/test_interaction_target \
  build/tests/test_interaction_attachment \
  build/tests/test_interaction_controller_adapter interaction_runtime_probe
build/tests/test_interaction_place_target
build/tests/test_interaction_target
build/tests/test_interaction_attachment
build/tests/test_interaction_controller_adapter
make test-interaction-safe
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-interaction
```

Expected: all commands exit 0; Python reports 261 or more tests with only the
existing environment-dependent skips; all C++ and fast-math binaries pass; the
pre-existing graphical pickup evidence contract remains unchanged and green.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place_target.h interaction_place_target.cpp \
  interaction_target.h interaction_target.cpp \
  tests/cpp/test_interaction_target.cpp \
  tests/cpp/test_interaction_attachment.cpp \
  tests/cpp/interaction_runtime_fixture.h interaction_runtime_probe.cpp \
  interaction_controller_adapter.cpp \
  tests/cpp/test_interaction_controller_adapter.cpp \
  tests/cpp/test_interaction_place_target.cpp interaction_matcher.h Makefile
git commit -m "feat: define deterministic placement surfaces"
```

Review checkpoint: validate transform conventions, explicit off-center bound
math, support-volume/plane/usable-extent coincidence, affordance finiteness and
inclusive numeric boundaries, every mandatory target author, unchanged graphical
pickup evidence, and that no target-selection policy leaked into runtime code.

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
  `AttachmentController::commit_place`, both committing destination pickup
  support with the object pose.
- Consumes: the exact Held target generation and original pickup owner already
  retained by `AttachmentController`.

- [ ] **Step 1: Write failing transactional release tests**

Add assertions equivalent to:

```cpp
const Transform placed{vec3(1.0F, 0.82F, 4.0F), quat()};
const PlacedSupportContext destination{
    Transform{vec3(0.0F, 0.70F, 4.0F), quat()},
    vec3(2.0F, 0.04F, 0.60F)};
const TargetHandle old_handle = fixture.request.target;
const auto next = fixture.attachment.commit_place(placed, destination);
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
assert(exact(stored->table_world, destination.table_world));
assert(exact(stored->table_size, destination.table_size));
```

Also prove wrong owner, stale generation, non-Held state, invalid object pose,
invalid destination support transform/size, and generation overflow cannot
partially mutate object pose, table pose/size, state, owner, or generation. Call
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
    Transform placed_world,
    PlacedSupportContext destination_support);

std::optional<TargetHandle> AttachmentController::commit_place(
    Transform placed_world,
    PlacedSupportContext destination_support);
```

Define `PlacedSupportContext` in `interaction_target.h` with `Transform
table_world` and `vec3 table_size`. Validate `placed_world` and the finite unit
support transform/strictly positive support size before locating or mutating the
target. Require exact `Held` state and owner. On success increment generation,
write the supplied object pose and destination `table_world/table_size`, set
`Free`, clear owner, and return the new handle. Do not call `release()` and then
`replace_pose()`. `AttachmentController` updates its local pose/state/result only
after the registry transaction succeeds.

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
registry object pose and pickup-support context exactly match the supplied
hand-derived pose and destination surface snapshot.

### Task 3: Recorded-Priority Place Selection and Certified Reverse Playback

**Files:**
- Create: `interaction_place.h`
- Create: `interaction_place.cpp`
- Create: `tests/cpp/test_interaction_place.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `Database`, retained pickup `MatchCandidate`, exact held
  `TargetHandle`/`GraspAffordance`, complete current Carry pose/object snapshot,
  and a requested placement goal.
- Produces: `PlaceTimingConfig`, `PlaceMatchConfig`, `PlaceMotionMode`,
  `PlacePhase`, `RecordedPlaceClip`, `PlaceMotionLibrary`, `PlaceCandidate`,
  `select_place_motion`, `PlaceStagingPreview`, `preview_place_motion`, and
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

struct PlaceTimingConfig {
    float canonical_fps = 25.0F;
    float playback_speed = 1.0F;
    float entry_blend_seconds = 0.25F;
    float reversed_commit_seconds = 0.50F;
    float maximum_alignment_seconds = 1.00F;
};

struct PlaceMatchConfig {
    float maximum_entry_root_error_m = 0.25F;
    float maximum_entry_yaw_error_radians = 0.436332313F;
    float maximum_hand_correction_m = 0.12F;
    float maximum_hand_orientation_radians = 0.436332313F;
};

struct RecordedPlaceClip {
    uint64_t id = 0;
    uint64_t object_profile_id = 0;
    uint32_t fps_numerator = 25;
    uint32_t fps_denominator = 1;
    std::vector<Pose> poses;
    std::vector<Transform> object_poses;
    std::vector<uint8_t> active_hand_contacts;
    int32_t entry_frame = -1;
    int32_t commit_frame = -1;
    int32_t release_frame = -1;
    int32_t retract_stop_frame = -1;
    Hand hand = Hand::Right;
    Transform hand_in_object{};
    ObjectLocalBounds object_bounds{};
    PlacementSurface source_surface{};
    uint32_t source_affordance_id = 0;
};

struct PlaceMotionLibrary {
    std::vector<RecordedPlaceClip> recorded;
};

struct PlaceMatchInput {
    const Database* pickup_database = nullptr;
    const PlaceMotionLibrary* library = nullptr;
    TargetHandle held_target{};
    MatchCandidate pickup_candidate{};
    Pose current_pose{};
    Transform current_object_world{};
    uint64_t held_object_profile_id = 0;
    ObjectLocalBounds held_object_bounds{};
    GraspAffordance held_affordance{};
    PlacementSurface surface{};
    PlaceAffordance place_affordance{};
    vec3 object_dimensions{};
    PlaceTimingConfig timing{};
    PlaceMatchConfig match{};
};
```

Tests must prove:

```cpp
const PlaceResult preferred = select_place_motion(input_with_both_modes());
assert(preferred.accepted);
assert(preferred.candidate.mode == PlaceMotionMode::RecordedPlace);

PlaceMatchInput far = far_input_with_both_modes();
const PlaceStagingPreview far_preview = preview_place_motion(far);
assert(far_preview.accepted);
assert(!far_preview.ready);
assert(far_preview.candidate.mode == PlaceMotionMode::RecordedPlace);

PlaceMatchInput staged = stage_carry_snapshot(
    far, far_preview.staging_root_world);
const PlaceStagingPreview staged_preview = preview_place_motion(staged);
assert(staged_preview.accepted);
assert(staged_preview.ready);
assert(staged_preview.candidate.source_id ==
       far_preview.candidate.source_id);
assert(staged_preview.candidate.selection_id !=
       far_preview.candidate.selection_id);

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

Also place root/yaw exactly on each configured default readiness boundary, then
at `std::nextafter(boundary, +inf)`, and prove ready then not-ready without source
or tier fallback. Repeat with one valid tighter `PlaceMatchConfig`. Invalid timing
or match configs reject the whole query before recorded/fallback selection.

Advance a 1.0x fallback player for five `0.04F` updates and assert source frames
are exactly `start, start-1, ..., start-5`; pose velocity, angular velocity, and
hand-DOF velocity are the negatives of the source sample; foot contacts equal the
source frame; release occurs exactly at `contact_frame`; and sampling never
crosses `entry_frame`. Make the first Hold window fail stability and the second
pass; assert the second window's last frame is selected. Add rejection fixtures
for contact loss between `contact_frame` and the selected sample, no stable Hold
window, 2.0001 cm within-window hand/object drift, 10.001-degree within-window
drift, malformed events, malformed or discontinuous active-hand contact through
release, wrong object-profile ID, bounds-center or half-extent difference above
`0.001 m`, wrong hand, hand-in-object translation above `0.02 m`, hand-in-object
rotation above `10 degrees`, invalid source support, and intrinsic mapped hand/
object/clearance correction-limit-plus-epsilon. Do not include current entry-root
translation or yaw in this intrinsic rejection family.

Give recorded fixtures nonzero unique clip IDs. Validate structure/physics first;
a malformed row that repeats a valid row's declared ID is removed before the
uniqueness pass and cannot poison that row. Reject each zero-ID row and every
structurally valid row participating in a duplicate-ID group without rejecting
unrelated unique rows or the library as a whole. At the recorded release sample,
run `evaluate_actual_placement_fit` with
the clip's `source_surface`, selected source affordance, and explicit bounds;
exercise exact supported boundaries and boundary-plus-epsilon failure. Through
entry-to-release inclusive, compare active-hand FK with
`object_pose * hand_in_object` at `0.02 m` / `10 degree` boundaries and reject one
discontinuous sample. Prove an invalid recorded row beside a valid row cannot
prevent that valid recorded row from winning, and a library containing only
invalid recorded rows still admits a certified reversed fallback.

For snapshot identity, perturb one value independently in current pose positions,
velocities, rotations, angular velocities, hand DOF, hand-DOF velocities, foot
contacts, current object position/rotation, held target ID/generation, held
profile/bounds/grasp/dimensions, every requested-surface and affordance field,
timing/match configuration, pickup-candidate/source content, every consulted
recorded row, source events, and mapped corrections. Each perturbation must change
the nonzero `selection_id`. Pointer-address-only changes with identical canonical
database/library content must not change it. Quaternion-sign-equivalent and
negative-zero-equivalent snapshots canonicalize to the same ID. Task 5 uses the
far/stale IDs as rejected preflight requests.

At 0.85x and 1.15x, advance across the reverse release event and assert the
published source frame is clamped exactly to `contact_frame`, `release_due` is
true exactly once, repeated sampling before acknowledgement does not advance,
and the first post-acknowledgement update begins retraction from contact. Assert
reverse derivative channels equal `-speed * source`, and mapped root linear and
angular velocities are rotated through `scene_from_source`.

For recorded playback, require the candidate commit frame to equal the authored
clip event. For reverse, derive it as
`start - floor(reversed_commit_seconds * 25 * playback_speed)` and reject zero
offset, commit at/beyond either start or release, or elapsed entry-to-commit time
below `entry_blend_seconds` or above `maximum_alignment_seconds`. Assert these
rules at 0.85x, 1.0x, and 1.15x.

- [ ] **Step 2: Run RED**

```bash
mkdir -p build/red
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_interaction_place.cpp \
  -o build/red/test_interaction_place.o
```

Expected: the compiler exits nonzero because `interaction_place.h` does not
exist; no missing Make rule is accepted as RED evidence.

- [ ] **Step 3: Implement deterministic two-tier selection**

Expose:

```cpp
struct PlaceCandidate {
    PlaceMotionMode mode = PlaceMotionMode::None;
    uint64_t source_id = 0;
    uint64_t selection_id = 0;
    PlaceTimingConfig timing{};
    PlaceMatchConfig match{};
    int32_t clip = -1;
    int32_t entry_frame = -1;
    int32_t commit_frame = -1;
    int32_t release_frame = -1;
    int32_t stop_frame = -1;
    int32_t direction = 0;
    Transform scene_from_source{};
    Transform staging_root_world{};
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
    bool committed = false;
};

struct PlaceStagingPreview {
    bool accepted = false;
    bool ready = false;
    Reason reason = Reason::None;
    PlaceCandidate candidate{};
    Transform staging_root_world{};
    float root_error_m = 0.0F;
    float yaw_error_radians = 0.0F;
};
```

Require finite `PlaceTimingConfig`, exact `canonical_fps == 25.0F`, playback speed
in `[0.85F, 1.15F]`, positive entry-blend and reversed-commit seconds, and positive
maximum align time no smaller than either and no greater than `1.00F`. Require
every `PlaceMatchConfig` threshold to be finite and positive, with entry root/yaw
and hand position/orientation at most `0.25F` / `0.436332313F` and `0.12F` /
`0.436332313F`, respectively. Tighter positive values are valid.

Validate each in-memory recorded row independently as a 25 Hz contiguous
pose/object/contact sequence with strict
`entry < commit < release < retract_stop`, finite channels, valid quaternions,
nonzero unique clip ID, nonzero object-profile ID, positive explicit bounds, and
a valid complete source surface/affordance. Remove malformed rows before checking
ID uniqueness; then mark every remaining member of a duplicate-ID group invalid,
retain per-row rejection diagnostics, and continue. Row malformation never aborts
selection or poisons an otherwise valid row that declared the same ID. Contact
values are binary, active-hand contact is true
from entry through release, and every later retract sample is false. Require
active-hand FK to match `object_pose * hand_in_object` at every attached sample
within `<=0.02F` / `<=0.174532925F`. Require the source release object to pass
`evaluate_actual_placement_fit` against that same source surface, source
affordance, and explicit bounds.

Before cost evaluation, require exact object-profile and active-hand identity,
object-bound center/half-extents within `0.001F` per component, and demonstrated
versus held hand-in-object error `<=0.02F` / `<=0.174532925F`. Evaluate all
remaining intrinsically compatible true-place rows first. Current staging-root
translation/yaw are readiness diagnostics only: they never affect certification
or ranking. Before testing entry pose or hand-local correction, rigidly align the
current pose and attached object to the candidate staging root with the same rigid
transform; otherwise far global translation would be misclassified as an IK or
object-path failure. Return the minimum-cost certified recorded candidate without
comparing it to fallback cost. Only when that tier is empty or every recorded row
fails an intrinsic hard filter, scan five-sample windows wholly inside Hold and
select the last frame of the earliest window with continuous contact and
within-window hand-in-object drift `<=0.02F` / `<=0.174532925F`. Also require
active contact continuously from the semantic release event through that selected
sample.

For true-place candidates, map the source release object pose to the requested
goal. For fallback candidates, map the source active hand at `contact_frame` to
`placement_goal_world * held_affordance.hand_in_object`; do not drive the held
object from the source object trajectory. `PlacePlayer` uses double-precision
elapsed/source accumulators, exact 25 Hz, shortest-arc pose interpolation,
direction-aware finish checks, and sign-negated derivative channels for direction
`-1`. Multiply derivative channels by playback speed and rotate root-world
vectors through `scene_from_source`. Do not inspect or depend on frames after the
certified reverse start.

For recorded candidates, copy the authored commit event. For reverse, derive the
commit source frame with the configured seconds/speed formula above. For either
direction, require commit strictly between entry and release and require positive
runtime entry-to-commit time within
`[entry_blend_seconds, maximum_alignment_seconds]`. The candidate stores this one
commit event; no later layer derives another horizon.

For either mode, compute `staging_root_world` from the mapped source entry root.
`preview_place_motion` is pure and returns the exact intrinsically certified
selector result plus planar root/yaw errors against the current Carry root. A far
candidate remains `accepted=true` and retains its source/tier/staging root while
`ready=false`; readiness compares against the inclusive `input.match` entry
thresholds, capped at `0.25F` / `0.436332313F`, and is not candidate rejection.

Compute deterministic nonzero `selection_id` from a canonical serialization of
the exact held `TargetHandle`; every channel of current `Pose`; current object;
held profile, bounds, grasp and dimensions; every field of the requested surface
and affordance snapshot; canonical content fingerprints of the pickup source and
every recorded row consulted; complete timing/match configs; selected source/
events; scene mapping, staging root, and every selection-relevant correction.
Never serialize pointer addresses.
Canonicalize normalized quaternion sign and negative zero. Preflight recomputes
the serialization from its frozen Carry snapshot and requires the same ID. A new
staged snapshot gets a new ID even when source candidate identity is unchanged.

Copy the validated `PlaceMatchInput::timing` and `match` into the candidate. Expose
`PlacePlayer::start(const PlaceCandidate&, const PlaceMatchInput&)`, `advance(float
dt)`, `PlaceSample sample()`, `source_frame()`, `phase()`, `committed()`,
`release_due()`, `acknowledge_release()`, and `finished()`. The player takes speed
only from `candidate.timing.playback_speed`; there is no second speed argument.
When an advance would cross commit, clamp and publish that exact source sample
and discard the sub-tick remainder before continuing on a later update. When an
advance would cross release, clamp the accumulator to the exact event, discard the
sub-tick remainder, emit the event once, and refuse to advance until
acknowledgement. The player stores only validated pointers or copies whose
lifetime is guaranteed by its owning `PlaceController`.

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

Review checkpoint: confirm recorded tier priority applies only after unique-ID,
profile, bounds, hand, grasp, attached-trajectory, contact, and source-support
validation; malformed rows cannot suppress fallback; far entry never demotes a
recorded row; snapshot identity covers every selection input; commit is singular;
fallback reads no late clip frame; and every allowed speed clamps events with
correct derivative time direction.

### Task 4: Bounded Place Controller and Release Gate

**Files:**
- Create: `interaction_place_controller.h`
- Create: `interaction_place_controller.cpp`
- Create: `tests/cpp/test_interaction_place_controller.cpp`
- Create: `tests/cpp/test_interaction_place_fast_math.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: selected `PlaceCandidate`, current Carry pose/object, held grasp,
  object-local bounds, surface/support volume/affordance, `IKConfig`, and exact
  25 Hz updates.
- Produces: `PlaceControllerConfig`, `PlaceBeginInput`, `PlaceStep`, and explicit
  `PlaceController(PlaceControllerConfig, IKConfig)` construction.

- [ ] **Step 1: Write failing correction, release, and recovery tests**

Freeze configuration:

```cpp
struct PlaceControllerConfig {
    PlaceTimingConfig timing{};
    PlaceMatchConfig match{};
    float release_position_m = 0.02F;
    float release_orientation_radians = 0.174532925F;
};
```

Tests assert exact boundaries are accepted and `std::nextafter(boundary, +inf)`
is rejected. Construct `PlaceController(valid_place, valid_ik)` successfully.
Independently mutate every timing, match, and release scalar through zero
where forbidden, NaN, infinity, reversed min/max relation, non-25-Hz rate, and
boundary-plus-epsilon; mutate every `IKConfig` scalar and iteration count through
its existing invalid families. Every invalid constructor call must throw
`std::invalid_argument` before `begin` and without observable state mutation.
Require finite `release_position_m` in `(0, 0.02F]` and finite
`release_orientation_radians` in `(0, 0.174532925F]`; valid tighter values remain
observable through diagnostics.
On every attached step assert:

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

Construct a requested goal whose footprint is exactly on the surface boundary.
Shift the hand-derived release pose outward by exactly `0.02 m` while keeping goal
error legal and assert release is rejected because actual footprint is invalid.
Repeat for an authored `+0.020 m` support gap shifted upward, a lowest corner
shifted below `-0.005 m`, a rotated bound exceeding the footprint, and an actual
bound corner exceeding overhead clearance. Mutate the support volume so the
conservative pre-release swept envelope crosses it and assert `BlockedPath`;
include a rotating off-center bound whose endpoint OBBs are disjoint but whose
angularly inflated envelope intersects. Allow only the release-clamped interval
whose actual support fit passes.

Advance PlaceAlign for at least four ticks, then cancel. Assert the recovery step
contains the exact current displayed safe pose/object. Add the same assertion for
a post-commit release-gate failure. These fixtures feed Task 5's Carry reseed
tests rather than assuming the previously paused Carry controller is current.

For both an authored recorded commit and derived reverse commit, assert the last
strictly pre-commit sample remains `PlacePhase::Align` and cancellable, the exact
commit sample is published once as the first committed replay sample, and every
later sample remains committed. A cancel on the last pre-commit sample returns
attached recovery; cancel on the update publishing commit and the following
sample is ignored. Assert entry-to-commit runtime time lies within
`[entry_blend_seconds, maximum_alignment_seconds]` from `config.timing` at 0.85x,
1.0x, and 1.15x.

Add exact `dt` rejection for `0.0F`, `1.0F / 60.0F`, NaN, and the adjacent float
around `0.04F`; only the exact constant passes.

- [ ] **Step 2: Run RED**

```bash
mkdir -p build/red
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_interaction_place_controller.cpp \
  -o build/red/test_interaction_place_controller.o
```

Expected: the compiler exits nonzero because `interaction_place_controller.h`
does not exist; a missing Make rule is not RED evidence.

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
    PlacementFit actual_fit{};
    bool support_sweep_clear = false;
};

struct PlaceBeginInput {
    const Database* pickup_database = nullptr;
    const PlaceMotionLibrary* library = nullptr;
    TargetHandle held_target{};
    PlaceCandidate candidate{};
    Pose current_pose{};
    Transform current_object_world{};
    uint64_t held_object_profile_id = 0;
    ObjectLocalBounds held_object_bounds{};
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
    PlaceController(PlaceControllerConfig config, IKConfig ik_config);
    PlaceBeginResult begin(const PlaceBeginInput& input);
    PlaceStep update(float dt);
    void acknowledge_release(Transform placed_world);
    PlaceStep cancel();
};
```

There is no default constructor. Store validated configuration by value.
`begin` revalidates that the candidate's timing/match fields and single commit
event match the constructor configuration before accepting playback.

The controller applies the mapped planar root/yaw correction that runtime already
authorized from the frozen snapshot. It does not recompute dynamic readiness,
reject on current entry root/yaw, change source tier, or select fallback. Task 5's
`PlacePreflight` is the sole configured entry gate, capped at `0.25 m` /
`25 degrees`; Task 4 verifies only that the authorized warp is applied continuously
and remains finite.

Blend entry for `ceil(config.timing.entry_blend_seconds * 25)` output ticks
(exactly seven at the default), completing no later than the candidate's commit
sample. Distribute planar root/yaw correction with smoothstep to zero by release,
and ramp bounded hand IK to full release weight. At every sample, form the object
OBB from the explicit object-local center and half-extents. Reject pre-release OBB
overlap with the support-volume OBB using 15-axis SAT. For each interval, transform
all previous/current bound corners into support local space and form their
componentwise envelope. Expand every axis by
`r_max * (1 - cos(theta / 2))`, using the maximum object-origin-to-corner radius
and the shortest-arc object rotation `theta`, then slab-test that conservative
envelope against the support-volume AABB. Exempt only the interval clamped to
release, and only after `evaluate_actual_placement_fit` accepts that exact pose.

Keep `last_safe_pose` and its hand-derived object for attached recovery. `cancel`
and every pre-release failure return that pair with `recover_to_carry=true`.
After `acknowledge_release`, freeze exactly the supplied transform while playback
retracts. An unacknowledged clamped event remains fixed and cannot emit release a
second time.

- [ ] **Step 4: Run focused, fast-math, and safe GREEN**

```bash
make build/tests/test_interaction_place_controller
build/tests/test_interaction_place_controller
make test-interaction-place-release-fast-math
make test-interaction-safe
```

Expected: all commands exit 0.

`test-interaction-place-release-fast-math` must compile
`tests/cpp/test_interaction_place_fast_math.cpp` and all place-controller
dependencies with `-O3 -DNDEBUG -ffast-math`. The test uses an explicit
`require()` helper rather than disabled `assert` calls and covers actual-pose
boundary rejection, one-shot clamped release at 0.85x/1.15x, and mutation-free
attached recovery. Add this target to `test-interaction-safe`; the existing Carry
fast-math binary is not placement coverage.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place_controller.h interaction_place_controller.cpp \
  tests/cpp/test_interaction_place_controller.cpp \
  tests/cpp/test_interaction_place_fast_math.cpp Makefile
git commit -m "feat: play bounded tabletop placement"
```

Review checkpoint: verify correction limits and actual-pose support validity are
hard gates where owned here, swept support math uses explicit off-center bounds
plus conservative angular inflation, constructor validation is exhaustive and
mutation-free, and current entry-root/yaw readiness is not re-enforced after
preflight. Verify the authored/derived commit boundary has no off-by-one
cancellation ambiguity, release is one-shot at every speed, object ownership has
one source in each phase, and recovery never detaches.

### Task 5: Integrate Placement into InteractionRuntime

**Files:**
- Modify: `interaction_runtime.h`
- Modify: `interaction_runtime.cpp`
- Modify: `interaction_debug_draw.h`
- Modify: `tests/cpp/interaction_runtime_fixture.h`
- Modify: `tests/cpp/test_interaction_runtime.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: placement surface registry, place library/controller, existing pickup
  candidate, selected-motion preview identity bound to the frozen held
  target/pose/object, attachment owner, and `RuntimeInput::place_request`.
- Produces: appended runtime states, place diagnostics, full place lifecycle, and
  complete runtime/controller/probe link dependencies in `Makefile`.

- [ ] **Step 1: Write failing API and state-machine tests**

Append enum values and freeze old values:

```cpp
static_assert(static_cast<uint8_t>(RuntimeState::Carry) == 6U);
static_assert(static_cast<uint8_t>(RuntimeState::PlacePreflight) == 7U);
static_assert(static_cast<uint8_t>(RuntimeState::PlaceAlign) == 8U);
static_assert(static_cast<uint8_t>(RuntimeState::PlaceReplay) == 9U);
static_assert(static_cast<uint8_t>(RuntimeState::PlaceRelease) == 10U);
static_assert(std::is_same_v<
    decltype(RuntimeConfig{}.place), PlaceControllerConfig>);
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
The released registry target's `table_world/table_size` must equal the
destination surface's `support_volume_world/size`; immediately resolve and
preflight a new ordinary pick of the returned Free handle and assert its query
uses that destination support rather than the source table.

For each invalid `PlaceControllerConfig` family from Task 4, place it in
`RuntimeConfig.place` and assert both runtime constructors reject before mutating
the target or surface registries. With valid distinct non-default timing, match,
and release values, assert placement diagnostics, readiness, and commit timing
reflect that exact config, proving runtime did not silently construct a default
controller.

Add independent tests for: missing request, wrong held object, unready staging
errors, selection ID from the far pre-staging snapshot, any stale snapshot digest,
stale surface between request and preflight, missing placement dependencies via
the legacy runtime constructor, cancel in PlaceAlign, candidate rejection,
post-motion-commit release-position failure, surface replacement before release,
mutation-free atomic commit rejection, duplicate Place edge, reset during
placement, recorded-mode diagnostics, reversed-mode diagnostics, and deterministic
replay. Every pre-release failure returns Carry attached; no failure returns
Locomotion with a silently dropped object.

Split continuity tests by whether playback began:

- For selection-ID mismatch, unready root/yaw, stale preflight surface, and missing
  placement dependencies, capture the frozen Carry pose/object at request time.
  Assert the rejection publication equals that pair exactly, the original
  `CarryController` was neither destroyed nor reconstructed, and the next update
  advances it exactly once within normal seam limits.
- For cancellation after four PlaceAlign updates, post-motion-commit release-gate
  failure, and atomic commit rejection, capture the last safe place pose/object.
  Assert the recovery publication equals them exactly, a fresh Carry controller
  starts from that pair, and its first advanced update remains within normal seam
  limits without restoring the pre-place object. Prove the paused controller is
  destroyed rather than resumed. Force atomic rejection with a Held target at
  `UINT32_MAX` generation so `place_held` rejects generation overflow without
  mutating state, owner, support, or pose.

For recorded-authored and reverse-derived candidates, test cancellation on the
last published sample before commit, the exact commit update, and the immediately
following update. Before commit returns reconstructed Carry; exact and after
commit ignore cancellation and remain contiguous `PlaceReplay` while attached.

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
mode, selection/staging preview, release, goal, actual-fit, sweep, and
support-error fields to diagnostics.
Add `PlaceControllerConfig place{}` to `RuntimeConfig`; validate it with the same
mutation-free exception behavior as matcher/playback/IK/carry configuration. The
placement-enabled constructor creates its controller only as
`PlaceController(config.place, config.ik)`, so runtime has one exact config path.
Every preview and frozen preflight receives `config.place.timing` and
`config.place.match`; no default or separately authored thresholds are allowed.
The legacy constructor retains no surface/library/controller dependencies and an
attempted Place request rejects unchanged from preflight.

In Carry, an Interact edge publishes `PlacePreflight` without advancing Carry or
mutating attachment. Validate on the following update by recomputing the complete
selected candidate from the frozen held target/pose/object snapshot and requiring
its canonical `selection_id` to match the request. Enforce staging root/yaw
against the exact configured limits here and nowhere in intrinsic selection;
those limits are capped at `0.25 m` / `25 degrees`. Do not apply the manual
surface-resolver distance. Any failure before `PlaceController::begin` publishes
unchanged Carry once and retains the original Carry controller. Delegate pose
generation only after all preflight gates pass.

At `release_due`, re-fetch the exact surface generation and affordance, recompute
`evaluate_actual_placement_fit` from `step.object_world`, require the controller's
support sweep to be clear, and call `AttachmentController::commit_place` exactly
once with that object transform plus
`PlacedSupportContext{surface.support_volume_world,
surface.support_volume_size}`. Pass the same object transform to
`acknowledge_release`, publish the returned target generation, then continue
retraction. If the transaction returns no handle, do not acknowledge or retry the
clamped release event. Verify the exact target remains Held by the same owner,
discard the place controller/player, and enter fresh Carry reconstruction from
that last safe release pair; the transaction and runtime output remain attached
and mutation-free.

On cancellation or pre-release failure, destroy the paused Carry controller,
construct a fresh one, and call `start(last_safe_pose, hand, affordance,
last_safe_object)`. Publish that exact pair on the recovery update without an
advance; advance the newly seeded controller on the following update. Do not
restore the object transform from before PlaceAlign.

Use the candidate's sole commit event for state transition. Publish samples
strictly before it as `PlaceAlign`; the exact event is the first `PlaceReplay`
sample. Process cancellation against the published event so the last pre-commit
sample cancels and exact/after-commit samples do not.

Define shared Make variables before either use:

```make
INTERACTION_PLACE_SOURCES := interaction_place_target.cpp \
  interaction_place.cpp interaction_place_controller.cpp
```

Append `$(INTERACTION_PLACE_SOURCES)` to both top-level `INTERACTION_SOURCES` and
`INTERACTION_RUNTIME_SOURCES`. Add all three place headers to the explicit
runtime test, controller-adapter test, and `interaction_runtime_probe`
prerequisites so compile/link and incremental rebuilds use the same dependency
closure.

Update debug names exhaustively and retain `ResultCode::Succeeded/Reason::None`
through successful PlaceRelease and final Locomotion.

- [ ] **Step 4: Run focused and safe GREEN**

```bash
make build/tests/test_interaction_runtime \
  build/tests/test_interaction_controller_adapter \
  interaction_runtime_probe controller
build/tests/test_interaction_runtime
build/tests/test_interaction_controller_adapter
make test-interaction-safe
```

Expected: all commands exit 0 and all prior pickup runtime tests remain green.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_runtime.h interaction_runtime.cpp \
  interaction_debug_draw.h tests/cpp/interaction_runtime_fixture.h \
  tests/cpp/test_interaction_runtime.cpp Makefile
git commit -m "feat: coordinate carry to place lifecycle"
```

Review checkpoint: audit every place transition, failure terminal, output
ownership flag, attachment mutation, target-generation/support update, exact
candidate/snapshot preview validation, preflight-unchanged versus post-begin
reconstruction choice, before/at/after commit cancellation, both frames of every
Carry continuity case, RuntimeConfig forwarding, and complete Make link closure.

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
  edge, read-only selected-motion preview, runtime placement output, and scene
  handoff.
- Produces: `F` pick/place behavior with ordinary Carry staging, destination-table
  drawing, and continuous release publication.

- [ ] **Step 1: Write failing adapter and controller-policy tests**

In C++, prove all new place states keep the runtime ownership epoch active, full
body place poses are not treated as layered Carry, and the first free scene sample
equals the final attached runtime object transform bit-for-bit. Prove a new target
generation ends the hand constraint without applying an IK solve to the free
object, while the normal 0.25-second pose release starts from the last displayed
PlaceRelease pose.

Add Python static policy tests asserting:

- controller constructs exactly one destination `PlacementSurface`;
- `F` resolves `PickRequest` only in Locomotion; in Carry it latches one surface
  through the manual-only `1.00 m` nearby-surface resolver and starts
  preview-guided staging without immediately pulsing runtime Interact;
- scene construction retains the exact returned destination handle/affordance so
  scripted/auto control can select it directly at any distance without calling
  the manual resolver;
- controller constructs one named `RuntimeConfig`, passes it to runtime, and uses
  that same value's `place.timing`/`place.match` for every preview rather than
  separately default-constructing selection thresholds;
- staging converts preview root/yaw error through the existing camera/control
  basis into ordinary Carry input and never writes simulation or displayed root;
- runtime Place is pulsed only when preview is ready and request `selection_id`
  equals the newly recomputed live staged preview, never a prior far preview;
- no place state calls `resolve_single_target` to replace the held object;
- destination goal is composed from `surface_world` and `object_in_surface`;
- destination support volume exactly becomes the released target's
  `table_world/table_size`;
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
farther along world `+Z`. Set `support_volume_world/size` to that physical table,
set `surface_world` to its top plane, and author `2.00 m` overhead clearance. Give
it one top-center affordance whose object transform and object-local support point
reproduce the source object's known supported origin offset. Derive that point
from the last stable pre-lift sample by
projecting the source object origin along the source table normal onto the table
top plane, then transforming the projected point through the inverse source
object transform; do not infer support from mesh bounds. Pass an empty
`PlaceMotionLibrary::recorded` so the demo reports `ReversedPickup`. Retain the
exact `SurfaceHandle` returned by `upsert` together with its affordance ID as the
stable authored destination identity; replacement must explicitly refresh this
pair rather than resolving by distance.

Author the one demo target with nonzero object-profile ID, explicit zero
object-local bounds center, and half-extents equal to half its measured
`object_dimensions`. This explicit center is a demo-scene authoring decision, not
a general inference rule.

Extend the scheduler with a place resolver and preview callback while preserving
the current pick resolver API. On `F`, resolve exactly one surface within the
1.00 m manual convenience envelope when the cached state is Carry and latch it.
This resolver is never called by the frozen auto-demo. Construct one named
`RuntimeConfig`, pass it into runtime, and capture its exact `place.timing` and
`place.match` values in the preview callback. Each 25 Hz tick recomputes the
preview and feeds staging error through the ordinary left-stick/Carry seam.
An accepted far preview retains candidate/staging information with `ready=false`.
`X` clears the latch. When the current preview reports root error `<=0.25 m`, yaw
error `<=25 degrees`, and ready, submit its newly computed explicit request once.
Draw the support rectangle/volume, explicit object bounds, final object frame,
approach axis, staging root/error/readiness, actual release fit, and place mode.
Change help text to `F pick/place  X cancel  R reset`.

During PlacePreflight/Align/Replay keep scene authority on the attached runtime
object. On atomic release, refresh the object handle by stable ID and let the Free
registry pose and destination `table_world/table_size` take authority. Do not
interpolate or overwrite that pose.

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
prove manual pickup behavior is unchanged when no place request is active,
the `1.00 m` surface resolver is manual-only, the authored destination identity is
stable and directly usable at any distance, staging never writes a root, stale
preview IDs are not submitted, and the released target is immediately selectable
with destination support context.

### Task 7: Add the Focused Headless Place Gate

**Files:**
- Create: `interaction_place_probe.cpp`
- Create: `tests/python/test_place_probe.py`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: deterministic fixture pack, pickup runtime, Carry, one placement
  surface with an explicit retained handle, a staged-ready Carry snapshot, and
  the reversed fallback.
- Produces: `interaction_place_probe <pack> --json` and
  `make gate-place-headless`.

- [ ] **Step 1: Write failing probe-output tests**

Require one compact sorted JSON record:

```json
{
  "actual_fit": true,
  "attachment_transitions": 1,
  "destination_selected_directly": true,
  "far_preview_accepted": true,
  "far_preview_ready": false,
  "final_attached": false,
  "final_object_state": "Free",
  "final_reason": "None",
  "final_result": "Succeeded",
  "mode": "reversed_pickup",
  "release_frame": 139,
  "repick_support_is_destination": true,
  "reverse_start_frame": 181,
  "staged_preview_ready": true,
  "staged_selection_id_changed": true,
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
After release, require the target's table transform/size to equal the destination
support and submit a fresh ordinary pick preflight for the returned generation;
assert its query context uses the destination support.

Before submitting Place, require an initial far Carry snapshot whose preview is
accepted with the expected source/staging root but not ready. Construct a second
internally consistent Carry snapshot at the returned staging root/yaw, including
its hand-derived held object, and require preview ready. The ready ID must differ
from the far ID. Submit only the ready request and assert a request carrying the
far or otherwise stale ID takes the unchanged-preflight Carry path.

- [ ] **Step 2: Run RED**

```bash
mkdir -p build/red
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
  -c interaction_place_probe.cpp -o build/red/interaction_place_probe.o
```

Expected: the compiler exits nonzero because `interaction_place_probe.cpp` is
absent. Do not treat a missing Make rule as the RED result. Run the Python
fixture-only validator tests after adding them; they must pass independently of
the real probe binary.

- [ ] **Step 3: Implement the deterministic probe and target**

The probe loads the existing interaction pack, performs pickup to Carry, and
constructs one authored destination surface while retaining its exact returned
handle/affordance. It never uses the manual distance resolver. From Carry it first
uses the same named `RuntimeConfig.place.timing/match` supplied to runtime to
prove far accepted/not-ready preview, then constructs a staged/ready Carry
snapshot and submits exactly one `PlaceRequest` with that snapshot's ID. It
advances only with `dt=1.0F / 25.0F` and checks every state and attachment
transition before printing JSON. It exits nonzero on rejection, failure, timeout,
non-contiguous source frames, unexpected mode, stale-ID acceptance, or
release-pose mismatch. It also verifies actual-pose support fit, exact destination
support commit, and one successful re-pick preflight using that destination
context.

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
place state sequence plus far-to-staged preview evidence is reported.

- [ ] **Step 5: Commit and review**

```bash
git add interaction_place_probe.cpp tests/python/test_place_probe.py \
  Makefile README.md
git commit -m "test: gate headless tabletop placement"
```

Review checkpoint: run the last-frame-contact mutation and verify the probe still
uses only the earlier certified Hold sample, then verify the released generation
re-picks against the destination support. Inspect the constructed Carry snapshot
and prove Place used the ready staged ID, not the far ID or a distance resolver.

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
MIN_WALK_DISPLACEMENT_M = 2.00
INITIAL_PICKUP_DISTANCE_M = 2.80
STANDOFF_MIN_M = 0.35
STANDOFF_MAX_M = 0.45
SETTLE_TICKS = 5
SETTLE_MAX_SPEED_MPS = 0.10
MIN_CARRY_STAGING_TICKS = 25
MAX_CARRY_STAGING_TICKS = 150
```

The synthetic fixture and validator must reject:

- initial pickup distance `<=2.80 m` (the stricter accepted bound also proves the
  requested start distance is greater than `1.25 m`);
- fewer than 25 consecutive initial `approach` rows;
- any owned pose, attachment, non-Locomotion state, or canonical-snapshot flag in
  the walking prefix;
- any root-relocation flag, scheduler provider other than `live_flat`, or
  simulation/displayed root initialization from the clip-0 Reach row;
- displayed-root walk displacement `<2.00 m` or no net pickup-distance progress;
- fewer than five consecutive settled rows with pickup distance in
  `[0.35 m, 0.45 m]` and displayed-root speed `<=0.10 m/s` immediately before
  Interact;
- Interact without the unchanged 1.00 m pickup-target resolver and live flat pose;
- skipped/duplicated 25 Hz runtime ticks or nonzero scheduler phase;
- any placement-surface resolver call in auto-demo mode, any destination handle/
  affordance different from the retained authored pair, or any destination
  generation change before Place;
- initial Carry preview not accepted, missing staging root, or incorrectly ready
  while outside `0.25 m` / `25 degrees`;
- fewer than 25 or more than 150 consecutive ordinary Carry staging rows;
- Carry root or object displacement `<=0.20 m`;
- any direct simulation/displayed-root write during staging, missing selected
  staging root, preview root error `>0.25 m`, preview yaw error `>25 degrees`, or
  false readiness on the runtime Place edge;
- selection ID on the Place edge equal to the far pre-staging ID or unequal to
  the current ready live preview ID;
- missing/duplicate Place action or any Reset action;
- release before PlaceRelease or more than one attached-to-free edge;
- final position error `>0.02 m`, orientation error `>10 degrees`, actual support
  gap outside `[-0.005 m, +0.020 m]`, invalid actual footprint/bound corners/
  overhead clearance, or blocked release sweep;
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

1. Keep the controller's default spawn. Record displayed-root-to-object planar
   distance and require it to exceed `2.80 m`; never call the placement-mode path
   that initializes the autodemo canonical world.
2. Replace placement-mode use of `make_autodemo_canonical_entry` with a focused
   `make_autodemo_reach_waypoint` helper that maps only clip-0's data-derived Reach
   root into the demo scene. It returns a root transform, not a pose or
   `LocomotionSnapshot`, and is used only as a navigation waypoint.
3. Command the ordinary left-stick locomotion seam toward that waypoint for at
   least 25 ticks and at least `2.00 m` displayed-root displacement. Convert the
   desired world direction through the existing camera basis; do not assign
   simulation position, displayed root, flat pose, or interaction pose. Every
   scheduler provider call returns the current live flat-controller snapshot.
4. Within the waypoint neighborhood, release movement input and require five
   consecutive ticks with pickup distance in `[0.35 m, 0.45 m]` and displayed-root
   speed at most `0.10 m/s`. Reset the settle counter whenever either bound fails.
   Then resolve with the unchanged 1.00 m target resolver and pulse Interact using
   that live pose. Preserve the matcher's unchanged `0.25 m` root correction
   bound.
5. Complete pickup and latch the retained authored destination handle/affordance
   directly regardless of current distance. Do not call `resolve_single_surface`
   in placement auto-demo mode. Require the first preview to be intrinsically
   accepted with a staging root and not ready. Recompute exact preview each tick
   and convert staging-root/yaw error through the existing camera basis to
   ordinary Carry left-stick input. Run for at least 25 and at most 150 ticks,
   require root/object displacement above `0.20 m`, and never write either root
   directly.
6. Let the controller pulse runtime Place once only when the exact live preview
   reports ready with root/yaw error at most `0.25 m` / `25 degrees`; submit the
   current staged-snapshot ID and prove it differs from the initial far ID.
7. Complete PlaceRelease, drain the normal seven-frame visual handoff without
   input, capture the placed object, publish evidence atomically, and exit.

Bound walking to 250 ticks, pickup-to-Carry to 375 ticks, placement to 250 ticks,
and total evidence to 900 records. Any timeout, window close, rejected runtime
result, fallback mode mislabel, or evidence I/O failure exits nonzero.

Log the current pickup distance, walking-origin displacement, displayed-root
speed, approach/settle phase and counter, locomotion-provider kind,
canonical-snapshot and root-relocation flags, Reach navigation waypoint, latched
destination handle/generation/affordance, placement-surface-resolver call count,
far and current selection IDs, candidate certification, staging
root/error/readiness, direct-root-write flag, place goal/error, requested and
actual fit/gap/bounds/overhead/sweep,
attachment transition count, destination support committed to the target, place
source/mode, and the same final-FK joint/grasp data used by the pickup evidence.

Require the canonical-snapshot flag to remain false on every placement auto-demo
record, including pickup matching after Interact. The existing pickup-only
auto-demo retains its current canonical-fixture behavior.

Add static placement-policy tests proving the placement-mode branch cannot call
`initialize_autodemo_canonical_world`, cannot select
`use_autodemo_canonical_snapshot`, and cannot return
`autodemo_canonical_entry->snapshot`. It may read the Reach row only inside
`make_autodemo_reach_waypoint`, whose return type contains no pose. Retain the
legacy helpers solely behind the existing pickup-only auto-demo mode.

Also prove the placement-mode branch reads only the retained authored
destination handle/affordance, never calls `resolve_single_surface`, and sends
only the current ready preview's selection ID. Keep manual `F` behavior and its
`1.00 m` resolver in the non-auto branch.

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
starts at the unchanged default spawn more than `2.80 m` away, shows at least 25
progressing `approach` rows and `2.00 m` of genuine non-owned flat walking, five
settled live-provider rows, then pickup, a direct stable authored-destination
selection, an accepted/not-ready far preview, ordinary preview-guided Carry
staging, a new ready selection ID, one `reversed_pickup` place, one release, and a
final supported object. No row uses a placement-surface resolver, canonical
snapshot substitution, or root relocation; the placement validator exits 0.

- [ ] **Step 6: Inspect visual evidence**

Capture a native-25-Hz lossless video through the working nested-display method
used by the pickup gate. Inspect approach, pickup entry/contact, moving Carry,
selected-motion staging, PlaceAlign, lowering, release, retraction, and locomotion
handoff. Reject the gate
if the object, root, head/neck, active elbow, inactive arm, or skeleton visibly
flips even when numeric limits pass.

- [ ] **Step 7: Document and commit**

README must include manual `F pick/place`, exact headless and graphical commands,
evidence paths, current expected `reversed_pickup` mode, the absence of local true
place data, the flat-ground/non-terrain scope, and the placement gate's live
default-spawn approach. Document that its data-derived Reach root is only a
navigation waypoint and that canonical relocation/snapshot substitution is
disabled in placement mode. Distinguish the manual `1.00 m` surface resolver from
the auto-demo's exact retained authored destination.

```bash
git add tests/python/test_playable_placement_evidence.py controller.cpp \
  Makefile README.md
git commit -m "test: prove walk pick carry and place"
```

Review checkpoint: independently inspect the lossless video and JSONL. Require a
start beyond `2.80 m`, progressing approach rows, the five-row settled band, the
`live_flat` provider throughout, and false canonical-snapshot/root-relocation
flags throughout. Require a stable direct destination handle, zero placement-
surface resolver calls, an accepted/not-ready far preview, and a distinct ready
live selection ID at Place. A passing synthetic or headless probe cannot
substitute for genuine ordinary flat walking in the graphical evidence.

## Final Verification and Branch Review

- [ ] Run the complete safe and place gates from the placement branch:

```bash
make test-interaction-safe
make test-interaction-place-release-fast-math
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
: "${PLACEMENT_BASE:?set to the full Task 1 recorded implementation-base hash}"
git diff --check "$PLACEMENT_BASE"..HEAD
git status --short --untracked-files=no
```

- [ ] Dispatch a fresh whole-branch spec reviewer and code-quality reviewer with
  the design, plan, commit range, headless JSON, placement JSONL/PNG/video, and all
  verification output. Fix every Critical or Important finding in one reviewed
  correction wave and rerun every affected gate.

- [ ] Hand off the isolated branch without merging into terrain. The next design
  after this gate is a parameterized height/depth/yaw sweep; constrained shelves
  and articulated objects remain separate specifications.
