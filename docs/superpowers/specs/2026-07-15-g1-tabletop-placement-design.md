# G1 Tabletop Placement Design

Date: 2026-07-15

## Summary

Extend the existing flat-ground interaction runtime with one deterministic game
primitive:

`walk -> pick -> carry -> place -> locomotion`

Placement starts only after an explicit Interact request while the runtime owns a
held object in `Carry`. The production baseline selects one contiguous recorded
place motion, applies the same bounded root warping and hand IK used by pickup,
releases the object at an authored event, and retracts before returning control.
When no compatible recorded place motion exists, the runtime may reverse the
exact pickup segment that produced the held object. That path is always reported
as `reversed_pickup`; it is never presented as recorded placement data.

The local GRAIL checkout contains `pickup_table` and `pickup_ground`, but no place
category. The current diagnostic clip ends with the object lifted rather than
supported. Reversal is therefore justified for the first playable placement
gate, while true recorded placement retains hard selection priority when such a
library is supplied.

This level remains learning-free, flat-ground, single-object, single-hand, and
fixed at 25 Hz. It does not merge or depend on terrain work.

## Goals

- Preserve the existing flat locomotion, pickup, Carry, and playable-pickup gates.
- Let a player or higher-level sequencer submit an explicit place request in
  `Carry`.
- Place the held object on an authored support slot without an object or skeleton
  teleport.
- Prefer a true recorded place clip, with a strictly labelled reversed-pickup
  fallback for the current data gap.
- Keep root, hand, orientation, clearance, and playback corrections bounded and
  observable.
- Parameterize support surfaces so later table-height and shelf-height trials use
  the same primitive.
- Prove the final graphical chain begins with genuine ordinary flat locomotion,
  not a canonical near-table startup.

## Non-goals

- Terrain, slopes, stairs, or terrain-aware IK.
- Learned grasp or placement-affordance inference.
- Physics settling, throwing, dropping, or dynamic balance control.
- Two-handed, deformable, stacked, or articulated objects.
- Shelf-cavity collision generalization in the first gate.
- Doors, drawers, refrigerators, navigation, or high-level task sequencing.
- Ingesting an unreviewed external place corpus in this implementation phase.
- Replacing the existing pickup state machine or interaction database schema.

## Chosen Approach

Three approaches were considered:

1. **Recorded place motion plus bounded cleanup.** This is the production-style
   baseline and has hard priority whenever a compatible true-place candidate is
   available.
2. **Reverse the selected pickup.** This preserves recorded full-body motion and
   contact timing, but reversed dynamics and hand intent are not genuine place
   data. It is accepted only as an explicit fallback because the available GRAIL
   release has no placement clips.
3. **Procedural hand IK over locomotion.** This is useful as a diagnostic but is
   not the Level-2 baseline because it does not author the hips, spine, free arm,
   or weight transfer.

The implementation introduces a place-motion source abstraction with
`recorded_place` and `reversed_pickup` modes. The shipped demo has an empty true
place library and exercises the certified reversal path. A later data-ingestion
change can populate the recorded tier without changing requests, surface
semantics, release semantics, runtime states, or tests.

## Runtime Contracts

### Placement surfaces

A placement surface is independent of the pickable object registry:

```cpp
struct SurfaceHandle {
    uint64_t id = 0;
    uint32_t generation = 0;
};

struct ObjectLocalBounds {
    vec3 center_object{};
    vec3 half_extents_object{};
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
```

`surface_world` is centered on the top support plane. Local `+Y` is the outward
normal; local `+X` and `+Z` span the usable rectangle. `support_volume_world`
and `support_volume_size` describe the physical table box below that plane. Its
top face must coincide with `surface_world`, including orientation, within
`0.001 m` and `0.1 degree`. `overhead_clearance_m` is the certified free height
above the plane within the usable rectangle; the open demo table authors
`2.00 m`. It is not an inferred ceiling or shelf-cavity model.

`object_in_surface` is the explicit final object-origin transform, and
`support_point_object` is the object-local point authored to touch the plane.
`ObjectLocalBounds` is an object-frame box with an explicit center and positive
half-extents. The placement runtime never assumes the object origin is the box
center and never guesses an origin, support height, or bounds center from a
mesh. The one demo object authors its measured dimensions and an explicit zero
bounds center; later object ingestion must author both fields.

`InteractionTarget` adds a nonzero `object_profile_id` and
`ObjectLocalBounds object_bounds`. Existing `object_dimensions` remains the
motion-query input and the interaction database format is unchanged. The profile
and explicit bounds are runtime scene metadata used to prevent recorded-place
selection across incompatible objects and to make support math independent of
mesh-origin conventions.

Level 2 accepts only static surfaces whose normal is within 5 degrees of world
up. Transform all eight object-bound corners through the tested object pose and
then into the surface frame. Their X/Z min/max, expanded by
`clearance_radius`, must fit within both surface half-extents. The support gap is
the surface-local Y coordinate of the tested object pose composed with
`support_point_object`; it must lie in `[-0.005 m, +0.020 m]`. The lowest bound
corner may not be below `-0.005 m`, and the highest may not exceed
`overhead_clearance_m`. These tests run once for the requested goal and again
for the actual hand-derived release pose. The first demo has one destination
table and one authored slot; no runtime policy chooses between multiple slots.

Before release, every sampled object OBB must be disjoint from the support-volume
OBB by separating-axis testing. Each 25 Hz interval also gets a conservative
support-local swept test. Transform all eight previous and current bound corners
into the support-volume frame, take their componentwise envelope, and expand it
on every axis by
`r_max * (1 - cos(shortest_arc_angle / 2))`, where `r_max` is the farthest bound
corner from the object origin. Slab-test that inflated envelope against the
support-volume AABB. The expansion bounds each rotating corner's deviation from
its endpoint chord, so rotation cannot tunnel between disjoint endpoint samples.
Only the interval clamped to the release event may touch the support volume, and
that exception is accepted only when the actual-pose footprint, bound-corner,
overhead-clearance, and support-gap tests all pass. Shelf walls, ceilings below
the authored free height, and arbitrary scene obstacles remain a later
constrained-clearance extension.

`PlacementSurfaceRegistry` validates nonzero handles, generations, finite unit
transforms, positive support-volume size, positive extents and overhead
clearance, top-face/plane coincidence, unique nonzero affordance IDs, nonzero
normalized approach directions, and exact-generation lookup. Object-bound
validation requires finite centers and strictly positive half-extents. Replacing
a surface increments its generation so a stale place request cannot commit.

### Place request

```cpp
struct PlaceRequest {
    TargetHandle held_target{};
    SurfaceHandle surface{};
    uint32_t affordance_id = 0;
    uint64_t request_id = 0;
    uint64_t selection_id = 0;
};
```

`RuntimeInput` retains `interact_pressed` and `pick_request` and adds
`place_request`. In `Locomotion`, Interact resolves a pick request. In `Carry`,
the same edge resolves a place request. The runtime accepts exactly the held
target it already owns and never substitutes another object or surface.

Selection policy remains outside `InteractionRuntime`. A read-only
`preview_place_motion` call runs the same deterministic selector used by
preflight and returns the selected source identity, mapped entry-root staging
transform, root/yaw error, correction status, and a nonzero `selection_id`. It
does not reserve a surface, mutate the held target, advance Carry, or grant pose
ownership. Preflight recomputes the preview from the same 25 Hz snapshot and
accepts only an exact `selection_id` match, so alignment cannot silently stage
for one motion and execute another.

The debug controller may resolve exactly one valid nearby slot and use this
preview for game-style assisted alignment. A future UI, script, or VLM may drive
ordinary Carry locomotion toward the same staging transform and submit the same
explicit request contract.

Coarse surface resolution measures planar distance from the character root to
each affordance's composed final object position and returns a handle only when
exactly one valid surface is inside the configured envelope. Coarse resolution
does not imply placement readiness. Readiness additionally requires the selected
motion's staging-root translation and yaw to satisfy the `0.25 m` / `25 degree`
entry-correction bounds. The displayed cue and assisted controller target the
selected staging root, not the object origin.

### State machine

Append these values after the existing `Carry` enum value so current numeric
contracts do not change:

```text
PlacePreflight
PlaceAlign
PlaceReplay
PlaceRelease
```

The success flow is:

```text
Carry
  -> PlacePreflight
  -> PlaceAlign
  -> PlaceReplay
  -> PlaceRelease
  -> Locomotion
```

- `PlacePreflight` is observable for exactly one 25 Hz update. The runtime still
  owns the Carry pose and attached object. It validates held-object ownership,
  the exact surface generation and affordance, range, support fit, clearance,
  candidate certification, match quality, and correction bounds.
- `PlaceAlign` begins contiguous playback and entry blending. It owns the pose,
  keeps the object attached, suppresses steering, permits cancellation, and lasts
  no more than 1.00 second.
- `PlaceReplay` begins at the commit horizon. Steering and arbitrary motion jumps
  are forbidden. The selected source advances contiguously to its release event.
- `PlaceRelease` begins on the update that atomically releases the object. The
  object remains fixed while the selected motion retracts the hand. On completion
  the existing 0.25-second controller handoff returns visual ownership to flat
  locomotion.

Cancellation in `PlaceAlign` keeps the same target generation, owner, and
attachment and begins a bounded recovery from the current displayed place pose
and hand-derived object transform. A pre-commit rejection before playback begins
returns to unchanged Carry. A post-commit failure before release keeps the object
attached and starts from the last safe place pose; it never finishes a hand
retraction while pretending the object detached.

Recovery reconstructs and starts a fresh `CarryController` from that exact safe
pose, active hand, held affordance, and object transform. The recovery update
publishes the same pose/object once, then the following 25 Hz update advances the
new Carry controller normally. Both the transition frame and the first advanced
Carry frame must satisfy object-transform equality and the visual seam limits;
the stale controller that was paused when placement began is never resumed. Once
atomic release succeeds, the object is never reattached or restored by placement
recovery.

All four place states own the runtime pose and suppress steering. This starts
only after the explicit place request and prevents background locomotion from
fighting local alignment.

### Diagnostics

Add `PlaceMotionMode { None, RecordedPlace, ReversedPickup }` and expose:

- surface handle and affordance ID;
- requested final object pose;
- selection ID, staging-root pose, staging translation/yaw error, and readiness;
- selected source kind, clip, frame, and semantic place phase;
- time to release and committed/released flags;
- requested and applied root, yaw, hand-position, and hand-orientation correction;
- support position/orientation error and support gap;
- requested-goal and actual-release footprint, support-gap, bound-corner,
  overhead-clearance, and support-volume sweep status; and
- rejection or recovery reason.

`ResultCode` remains unchanged. Existing reasons are reused where exact
(`OutOfRange`, `BlockedPath`, `CorrectionLimit`, `JointLimit`, `Cancelled`, and
`ClipEnded`); append `SurfaceUnavailable`, `SurfaceChanged`,
`PlacementOutOfBounds`, `ReleasePosition`, and `ReleaseOrientation` for failures
that cannot be diagnosed accurately with the pickup vocabulary.

## Motion Selection and Playback

### True recorded place tier

A true-place candidate defines a contiguous forward range with entry, commit,
release, and retract-stop events, a nonzero object-profile ID, explicit
object-local bounds, active hand, demonstrated hand-in-object transform, source
support-volume transform/size, poses, contacts, and object motion. The
whole-clip selector first hard-filters for exact object-profile and active-hand
identity, object bounds within `0.001 m` per center/half-extent component, and a
demonstrated hand-in-object transform within `0.02 m` / `10 degrees` of the held
grasp. Recorded active-hand contact must be binary and continuous from entry
through the release sample, then absent throughout retraction. It then maps the
demonstrated release object to the requested
`surface_world * object_in_surface`, evaluates current Carry pose/root/hand
continuity, and applies hard correction and clearance filters.

If one or more compatible true-place candidates pass, the minimum-cost
true-place candidate wins. A wrong-profile, wrong-bounds, wrong-hand, or
wrong-grasp row is not a true-place candidate for this request. A reversed
candidate may not beat an accepted compatible true-place candidate merely with a
lower numeric cost. This keeps the fallback from silently becoming the primary
behavior without allowing hard recorded priority to select an unrelated object
or grasp.

### Certified reversed-pickup tier

The fallback uses the exact `MatchCandidate` retained from the successful pickup;
it does not search a different pickup clip. Its reverse range is:

```text
reverse start: end of the earliest certified five-sample Hold window
release event: candidate.contact_frame
retract stop:  candidate.entry_frame
```

Starting at `candidate.hold_frame`, the certifier scans consecutive five-sample
windows that lie entirely in Hold. It chooses the last sample of the earliest
window whose active-hand contact is true throughout and whose hand-in-object
transform varies by at most 0.02 m and 10 degrees within that window. The
fallback is certified only when:

- the start remains inside the selected clip;
- active-hand contact is continuously true from `contact_frame` through the
  selected stable Hold sample;
- the selected Hold window, rather than the entire lift, passes the 0.02 m and
  10-degree relative-transform stability test;
- the selected interval contains the expected Contact, Lift, and Hold ordering;
- every source pose and transform is finite and every quaternion is valid; and
- reversed root, hand, object, and clearance corrections fit the runtime limits.

The clip's final frame is irrelevant and is never used as the reverse start. A
late clip frame may have lost contact or continued carrying; that does not
invalidate a certified earlier stable Hold sample.

In the current diagnostic pack, Contact begins at frame 139, Lift at frame 153,
and Hold at frame 176. Hold window `[176, 180]` fails the relative-grasp
stability test, `[177, 181]` passes it with continuous contact, and the final
frame 249 has no active contact. The deterministic fallback therefore starts at
181; these observed pack facts are fixture requirements, not hard-coded runtime
indices.

At 1.0x and 25 Hz, the source index decreases by exactly one per runtime tick.
At every allowed speed, source indices use a double-precision accumulator.
Linear, angular, and hand-DOF velocities are multiplied by playback speed and
sign-negated; root-frame vectors are also rotated through the scene mapping.
Quaternion sampling uses normalized shortest-arc interpolation. Foot contacts
remain the recorded sample for each reversed pose.

An update that would cross the authored release source frame clamps to that exact
frame, publishes its exact pose and hand-derived object transform, and emits
`release_due` once. It discards the sub-tick remainder and does not enter
retraction until the runtime acknowledges a successful atomic commit. A failed
actual-pose release gate enters attached recovery and can never emit
`release_due` again. The next 25 Hz update after acknowledgement resumes from the
release frame toward retraction. This one-frame event clamp applies at 0.85x,
1.0x, and 1.15x, so no allowed speed can skip release.

For the fallback, the source hand at `contact_frame` is mapped to the requested
goal hand `goal_object_world * held_hand_in_object`. While attached, object motion
is always derived from the corrected active hand and the held `hand_in_object`
transform. The source object trajectory is used only for contact/window
certification and diagnostics; it never independently drives or teleports the
scene object. A true recorded-place clip may use its authored object trajectory
for selection, but the held runtime object remains hand-derived until release.

### Bounded correction

Placement uses the existing hard limits:

| Correction | Maximum |
| --- | ---: |
| Interaction-controlled approach | 1.00 m |
| Planar root warp | 0.25 m |
| Root yaw warp | 25 degrees |
| Hand positional IK request | 0.12 m |
| Hand orientation IK request | 25 degrees |
| Uniform playback speed | 0.85x to 1.15x |

Entry correction decays smoothly to zero by release. Hand IK ramps to full weight
at release and may not exceed G1 joint limits. Existing flat foot contact handling
runs after the interaction handoff. There is no terrain sample or terrain IK.

Release accepts final object error at or below 0.02 m and 10 degrees, inclusive.
That tolerance never substitutes for support validity: the exact hand-derived
object pose is recomposed into the exact current surface generation and must
independently pass footprint, support-gap, lowest/highest-bound,
overhead-clearance, and support-volume sweep tests immediately before commit.
Larger correction or actual-pose support errors recover while attached rather
than snapping or dropping the object.

## Atomic Release and Scene Ownership

Placement must not compose the current `release()` and `replace_pose()` calls.
Add one transactional registry operation:

```cpp
struct PlacedSupportContext {
    Transform table_world{};
    vec3 table_size{};
};

std::optional<TargetHandle> TargetRegistry::place_held(
    TargetHandle held,
    uint64_t owner_request,
    Transform placed_world,
    PlacedSupportContext destination_support);
```

It validates the object transform and positive finite destination support before
mutation and succeeds only for the exact `Held` generation and pickup owner.
The support context is copied from the exact current placement surface's
`support_volume_world/size`; it is not supplied from stale request data. Success
increments generation, writes `placed_world`, replaces the target's
`table_world/table_size` with the destination support, sets `Free`, clears the
owner, and returns the new handle. Failure does not mutate pose, support,
generation, state, or owner.

`AttachmentController::commit_place` wraps this operation and accepts the same
placed transform and destination support snapshot. `placed_world` is the current
hand-derived object transform at the release event, not the independently
authored goal. The authored goal is only a gate and IK target. Runtime output,
registry state, and the first free scene sample therefore contain the same object
transform across the attached-to-free edge.

On release, diagnostics switch to the new target generation. The existing scene
handoff drops runtime authority because the registry object is `Free`; equality of
the published transforms prevents a scene snap. The object stays fixed during
hand retraction. A subsequent ordinary pick of the new Free handle queries the
destination support volume, never the original source table; this re-pick is part
of the headless acceptance contract even though the graphical demo stops after
one placement.

## Controller and Demo Scene

Manual control remains game-like:

- ordinary movement controls drive flat locomotion;
- `F` / right-face-left picks in `Locomotion`; in `Carry` it latches one explicit
  destination and begins selected-motion staging;
- while staging, the controller converts staging-root error into ordinary Carry
  locomotion input, never a direct root write, and submits Place automatically
  only when the preview is ready;
- `X` / right-face-up cancels staging or committed-preflight alignment before
  place commitment; and
- `R` remains a debug reset, not part of the successful placement chain.

While any `Place*` state owns the interaction, `R` has no effect. This prevents a
debug restore from teleporting an object during lowering or after irrevocable
release. Pre-commit `X` is the only placement cancellation path.

The scene contains a source table/object and one destination table/slot. For the
frozen auto-demo, the destination table has the source table's size and height
and is translated 1.20 m farther along world `+Z`; its slot is centered on the
top plane. The destination support transform is drawn with its usable bounds,
final object frame, approach axis, and correction envelope.

The demo support point is derived once from the last stable pre-lift source
sample: project the source object origin along the source table normal onto the
source top plane, then transform that projected world point into object-local
space. The destination reuses that explicit local point and supported origin
offset. This is recorded scene geometry, not a mesh-bottom estimate.

The placement auto-demo starts from the controller's default spawn, more than
`2.80 m` from the pickup object. It may map the clip-0 Reach root into the demo
scene once, but that transform is only a navigation waypoint. It may not initialize
the simulation/displayed root from that row, copy its pose into locomotion, or
return its synthetic `LocomotionSnapshot` from the scheduler provider.

The demo commands ordinary live flat locomotion toward that waypoint for at
least 25 consecutive 25 Hz approach ticks and at least `2.00 m` of displayed-root
displacement. It then releases movement input and requires five consecutive
settled ticks with root-to-object planar distance in `[0.35 m, 0.45 m]` and
displayed-root speed at most `0.10 m/s`. This hysteresis prevents a one-frame
threshold crossing. Only then may the unchanged 1.00 m target resolver produce a
pick request and Interact pulse; the selected clip must still satisfy the
unchanged `0.25 m` root-warp bound. Every placement-auto-demo scheduler update
uses the current live flat-controller snapshot before and after Interact. The
legacy pickup-only canonical relocation/substitution path may remain for its
separate gate, but it is disabled and unreachable in placement mode.

After pickup reaches Carry, the demo latches the destination and recomputes the
selected-motion preview every tick. It commands ordinary Carry locomotion toward
the preview's staging root for at least 25 and at most 150 consecutive ticks,
requires more than `0.20 m` root and object displacement, and submits Place only
when staging translation is at most `0.25 m`, staging yaw is at most `25 degrees`,
and the exact preview reports ready. It never writes the root, pulses Reset, or
uses a fixed tick count as a proxy for readiness. The screenshot is taken after
successful release with the object visible on the destination table.

## Verification

### Focused headless gate

The headless place gate uses synthetic recorded and reverse fixtures and runs at
exactly `0.04` second per update. It proves:

- existing runtime enum values and pickup behavior remain unchanged;
- `PlacePreflight` is observable for one tick;
- success collapses to `Carry, PlacePreflight, PlaceAlign, PlaceReplay,
  PlaceRelease, Locomotion`;
- true recorded place wins whenever a compatible true-place candidate exists;
- reverse playback starts at the end of the earliest certified five-sample Hold
  window, never the clip's final frame;
- a fixture with invalid late-frame contact still succeeds when its earlier
  stable Hold/contact sample is certified;
- source indices decrement contiguously and reversed velocities have the correct
  sign;
- 0.85x and 1.15x updates clamp to the exact release sample, emit one release
  event, and begin retraction only after acknowledgement;
- exact correction and release boundaries pass and boundary-plus-epsilon fails;
- an exact-boundary goal whose allowed release error would put the actual object
  outside the footprint, support-gap, bound-corner, overhead, or sweep contract
  is rejected while still attached;
- attachment remains true before release, changes exactly once, and never returns;
- release increments object generation once and publishes an identical transform
  through runtime, registry, and scene handoff;
- release atomically replaces the target's support with the destination support,
  and an immediate re-pick query observes that destination;
- cancellation and pre-release failure reseed Carry from the last safe pose and
  remain continuous on both the recovery publication and first advanced Carry
  frame; and
- replaying the same inputs produces bit-identical selection and state output.

### Graphical gate

The new graphical gate is separate from and does not weaken
`gate-playable-interaction`. Its evidence must show:

1. at least 25 live-flat `approach` records in initial `Locomotion`, with no
   runtime pose ownership, attachment, root relocation, or canonical snapshot
   anywhere in the run;
2. default-spawn pickup distance greater than `2.80 m` (and therefore greater
   than `1.25 m`);
3. at least `2.00 m` ordinary displayed-root displacement and a decreasing
   pickup-distance progression before Interact;
4. five consecutive settled rows in the `[0.35 m, 0.45 m]` stand-off band at no
   more than `0.10 m/s`, followed by Interact from the live pose through the
   unchanged 1.00 m resolver;
5. collapsed states exactly `Locomotion, Preflight, Align, PickupReplay, Hold,
   Carry, PlacePreflight, PlaceAlign, PlaceReplay, PlaceRelease, Locomotion`;
6. 25 through 150 consecutive ordinary Carry staging commands, more than 0.20 m
   Carry root and object displacement, and a ready selected-motion preview before
   Place;
7. one latched Place action, no Reset action, and one attached-to-free transition;
8. final object error at most 0.02 m / 10 degrees, with the actual committed pose
   passing footprint, support gap `[-0.005 m, +0.020 m]`, bound-corner,
   overhead-clearance, and support-volume checks;
9. final `Locomotion`, `Succeeded`, `None`, `Free`, and unattached state; and
10. synchronous 25 Hz ticks plus the existing visual continuity limits: distal
    feet at most 12 m/s, authority-seam translation at most 0.20 m, and every
    joint rotation step at most 60 degrees.

The evidence records `recorded_place` or `reversed_pickup`. For the current local
GRAIL pack the expected mode is `reversed_pickup`.

## Follow-on Evaluation

After the single table-to-table gate passes, a deterministic grid may vary support
height, slot X/Z offset, yaw, and initial Carry approach. Each trial records
success, rejection reason, selected mode, correction magnitudes, release error,
support gap, and transition continuity. Shelves reuse the same surface and request
contracts but require a separate constrained-clearance specification and gate.
