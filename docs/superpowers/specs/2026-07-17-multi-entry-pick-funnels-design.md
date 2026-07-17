# Industry-baseline Smart Pickup

Date: 2026-07-17
Status: approved for implementation

## Purpose

Implement the conventional Smart Object interaction pipeline while changing
only the animation-selection mechanism needed to test the research hypothesis.

The character navigates with ordinary flat locomotion to an authored slot owned
by the selected grasp affordance and expressed in the target object's planar
frame. The selected slot is frozen. At the slot, the existing interaction
motion matcher chooses a recorded pickup continuation from the full tabletop
pack. Existing bounded root alignment, hand IK, contact validation, attachment,
Carry, placement, and release complete the interaction.

This isolates one research question:

> Can motion matching replace the usual interaction-montage chooser while the
> rest of the pickup pipeline remains conventional?

## Decision

Checkpoint 1 uses authored slots on the grasp affordance, not runtime-generated
funnels and not slots coupled to individual motion clips.

The considered approaches were:

1. **Authored Smart Object-style slots (selected).** Put several frozen local
   root poses on each affordance, navigate to one, then query the ordinary
   interaction motion database from the live slot pose.
2. **Procedural approach funnels.** Generate curved paths and terminal poses at
   runtime. This adds a custom local planner before the baseline is measured.
3. **Learned approach proposals.** Predict routes or terminal poses. This needs
   successful-route data and is deferred until authored-slot failures exist.

The earlier five-Bezier-funnel design and motion-linked slot sidecar are retired
from Checkpoint 1. If authored slots later leave meaningful coverage gaps,
procedural or learned proposals can be compared against this baseline.

## Scope

Checkpoint 1 includes:

- exact 25 Hz control, simulation, database, and interaction timing;
- existing ordinary flat locomotion and motion matching;
- one rigid, right-hand tabletop target;
- at least three authored slots on its selected grasp affordance;
- the validated full tabletop pack, containing the distinct compatible source
  continuations used to author those slots rather than the one-clip diagnostic
  pack;
- direct local navigation to the best clear slot within the unchanged one-metre
  interaction-controlled travel cap;
- frozen target, affordance, and slot identity for one attempt;
- existing runtime pickup preview and global interaction motion selection from
  the live slot pose;
- existing bounded root/yaw alignment and hand IK;
- existing contact-gated attachment, Carry, table-to-table placement, and
  release; and
- deterministic headless and graphical evidence across varied starts and target
  transforms.

Checkpoint 1 does not include:

- runtime slot generation, splines, or funnels;
- a new slot artifact or interaction database schema;
- restricting a slot to one clip or a private clip subset;
- global navigation around arbitrary obstacles;
- ground pickup ingestion;
- shelves or articulated objects;
- several simultaneously selectable targets;
- two-handed or deformable objects;
- inferred affordances; or
- diffusion or another learned approach model.

The slot representation is height-agnostic. Full GRAIL tabletop data is used in
this checkpoint; pickup_ground is a later data checkpoint using the same
affordance/runtime boundary.

## Conventional runtime flow

The manual flow is:

    player requests target
        -> enumerate authored slots on the chosen affordance
        -> reject out-of-range or blocked slots
        -> choose and freeze one slot
        -> ordinary flat locomotion directly to that slot
        -> brake and settle
        -> query the global interaction motion database
        -> bounded alignment and contiguous pickup playback
        -> contact-gated attachment
        -> Carry
    -> place and release

Ordinary manual Smart Pickup requires an explicit loaded full-pack shape (25/1
Hz, 2,045 clips, 511,250 frames, matching feature frames) and fails startup on
the one-clip diagnostic pack or an implicit fallback. Exact binary hashes remain
the authoring/headless/graphical gate authority. The diagnostic pack remains
permitted only inside separately selected legacy placement/interaction fixtures.

The single-player checkpoint does not reserve the physical object while walking.
The assist freezes target handle/generation, affordance ID, slot ID, object pose,
and mapped root locally and revalidates them every tick. The ordinary scheduler
remains the sole owner of TargetRegistry reservation during Preflight.

A future multi-agent claim service may move logical slot claiming earlier
without changing slot mapping, locomotion, motion selection, or playback.

## Target and affordance data

Add one narrow runtime record:

    struct GraspInteractionSlot {
        uint32_t id;
        float root_x_object_m;
        float root_z_object_m;
        float root_yaw_object_radians;
    };

GraspAffordance owns a vector of these slots. IDs are nonzero and unique within
the affordance. All numeric fields are finite. Empty slot arrays remain legal
for legacy fixtures and non-manual use, but manual Smart Pickup fails
NoAuthoredSlot for an empty selected affordance.

Target validation and exact runtime metadata comparison include the ordered slot
array. Changing slot order, ID, or geometry therefore changes authored target
metadata and invalidates a frozen attempt.

Slots for the controller demo are baked constants derived offline from distinct,
target-affordance-compatible, validated GRAIL pickup_table Reach roots in the
full pack. Runtime never calls make_pick_reach_waypoint or rotates a clip-0 root
to invent a manual slot. Documentation records the source sequence IDs,
canonical local entry frames, and extracted local values used for each authored
demo slot.

## Object-planar slot frame and affordance ownership

Slots are Smart Object-style markers owned by a grasp affordance. Their geometry
is expressed relative to the target object's XZ/yaw frame, matching the existing
interaction matcher alignment exactly. Ownership freezes which grasp contract a
slot belongs to; it does not make the pitched/rolled hand transform a navigation
frame.

For a target object:

- frame origin is object_world.position projected to XZ;
- frame rotation is the world-yaw component of object_world.rotation;
- frame +X is object-right and frame +Z is object-forward; and
- object Y, pitch, and roll do not tilt the flat navigation plane.

The mapped slot root is:

    root_xz = object_origin_xz
            + object_yaw * (root_x_object_m, root_z_object_m)
    root_yaw = object_yaw + root_yaw_object_radians
    root_y = live_flat_root_y

Projection failure or non-finite output rejects the slot.

A common target XZ translation or world-yaw rotation transforms every mapped
slot rigidly while preserving slot ID and relative geometry. A compatible novel
object can reuse or author the slot template when it supplies the same
grasp-affordance contract with its own object pose, hand-in-object grasp,
approach direction, clearance, collision bounds, and ordered slots. Checkpoint 1
does not infer slots from a novel mesh or semantic object identity.

## Authored slot extraction

The initial slot constants are derived offline, not synthesized during play:

1. build and validate the complete pickup_table pack at exactly 25 Hz;
2. select clips with the required right hand and complete Reach, Contact, Lift,
   and Hold continuation;
3. require the recorded contact hand transform to be compatible with the one
   selected demo affordance under the unchanged 0.12 m and 25 degree hand
   correction bounds;
4. use the source object transform at Contact minus one, matching the current
   matcher reference frame exactly;
5. express the first Reach Simulation-root XZ/yaw in that source object's
   planar frame;
6. reproduce the existing zero-correction matcher root/hand path-clearance gate
   through the entire continuation and reject failures;
7. against the published full pack, call unrestricted runtime preview from
   representative stationary live-flat snapshots at each prospective root and
   retain only roots with realized-transition path feasibility and a ready match;
8. sort by stable manifest provenance and greedily remove near duplicates
   within 0.10 m and 10 degrees; and
9. choose at least three distinct clear approach-side roots as authored demo
   slots.

Stable authoring provenance is `(dataset_id, schema_version, sequence_id,
entry_frame - range_start, active_hand)`. Clip ordinals and global frames are
pack-local and are not stable identities. The runtime slot ID remains a compact,
nonzero authored ID and does not serialize this provenance.

This is an authoring/data-preparation operation. The resulting values are
reviewable target data. The runtime does not retain source clip IDs and does not
force the motion matcher to replay the clip from which a slot was authored.

## Reachability and selection

On manual activation, map every ordered slot on the selected affordance into the
live scene. The result is a direct-segment eligibility check, not a navmesh or
global pathfinding claim. It evaluates each slot using:

- direct planar distance from the post-step live root;
- total interaction-controlled travel no greater than 1.00 m plus the existing
  2e-5 m tolerance;
- a closed, continuous planar segment against the table's yaw-only rectangle
  expanded by the interaction matcher's existing 0.24 m root proxy;
- a closed, continuous three-dimensional segment at fixed live-root Y against
  the same immutable world-axis-aligned obstacle arrays used by ordinary
  locomotion, requiring distance strictly greater than the controller's 0.60 m
  root-sphere radius plus the existing 2e-5 m safety margin; and
- no separate root/target collision proxy: the target is governed by terminal
  slot alignment and the existing preview hand-path, correction, IK, and
  contact gates.

Table pitch/roll and Y are ignored exactly like the matcher's root/table proxy.
Generic obstacle Y remains significant exactly like ordinary locomotion.
Table boundaries and equality at the generic-obstacle safety envelope count as
blocked, with no start or terminal exemption. Malformed or nonpositive obstacle data fails closed, and
the lowest failing obstacle index is diagnostic authority.

Checkpoint 1 deliberately uses a direct local segment. Euclidean distance alone
is never called reachable. A blocked segment rejects that slot; the selector
may choose another authored slot. It does not search around an obstacle.

Eligible slots are ordered by:

1. route length quantized to millimetres;
2. absolute initial heading change quantized to milliradians; and
3. authored slot ID.

For a nonzero route, initial heading is the XZ heading from the post-step live
root to the mapped slot. For a route of at most 1e-5 m, it is the mapped slot
yaw. Heading change is the absolute shortest-angle difference from the live
root yaw. Quantization computes `floor(value * scale + 0.5)` in finite double
precision and rejects values that cannot fit in `uint64_t`.

Quantization is deterministic half-up and rejects non-finite/overflowing input.
The winner is frozen. Runtime match costs cannot switch navigation slots.
Every later 25 Hz observation rechecks the remaining direct segment, current
root/table clearance, and the same immutable obstacle arrays. For generic
obstacles, both remaining-segment endpoints are evaluated at the current
observed live-root Y; clearance never interpolates toward the Y stored in the
frozen navigation transform, and that frozen transform is not mutated. Failure cancels
the attempt without slot switching or snapping. The table test remains an
interaction precondition; Checkpoint 1 does not claim that the ordinary
locomotion controller physically collides with the table.

## Native-25-Hz ordering

The current manual-assist order remains authoritative:

1. read raw input;
2. pre-submit X cancellation wins over simultaneous F;
3. on accepted F, consume it, resolve/latch target and affordance, and zero both
   character sticks;
4. advance exactly one ordinary 0.04 s flat locomotion/motion-matching step;
5. materialize exactly one post-step live_flat_snapshot and fingerprint;
6. map, validate, select, and freeze one slot from that snapshot;
7. call the assist observation boundary exactly once;
8. give the scheduler provider the same snapshot unchanged; and
9. begin slot-navigation movement no earlier than the following tick.

The assist emits the same ordinary left-stick navigation used today. Within the
existing slow radius it uses arrival_navigation_stick, force-strafe, and
arrival_facing_stick to brake and match the frozen slot yaw. Preserved raw
camera input, not synthetic facing, owns camera orbit.

The assist never writes displayed/simulation roots, joints, locomotion database
frames, or canonical poses.

The pending-intent and post-step activation logic lives in one raylib-free
production `SmartPickupController` coordinator. The native controller and the
headless oracle both bracket their ordinary locomotion step with this same
pre-step/post-step API. The coordinator owns the pending target snapshot,
assist observation, at-most-one unrestricted preview, and one-shot request
extraction; it never advances locomotion or submits directly to the scheduler.
`controller.cpp` contains only input/snapshot/runtime adaptation around it, not
a second implementation of the state machine.

## Simplified manual assist

The manual state flow becomes:

    Idle
        -> SlotApproach
        -> Settling
        -> FinalPreview
        -> ReadyToSubmit
        -> Submitted

The manual path removes:

- clip-0 make_pick_reach_waypoint activation;
- synthetic Plus/Minus arc construction;
- the common 0.60 m pre-entry waypoint;
- the initial two-slot preview state; and
- hand-score selection.

Those legacy functions remain temporarily for the separately validated placement
auto-demo. This checkpoint changes only manual Smart Pickup wiring and tests.

Every post-activation observation revalidates the exact frozen target,
affordance, slot, and mapped root. Slot mutation, target generation change,
target pose change, object ownership change, or runtime leaving Locomotion fails
the assist without choosing another slot.

## Arrival, preview, and request

Manual Smart Pickup arrival and settling use the frozen slot transform as their
only navigation-pose authority:

- approach latch root error at most 0.03 m, simulation speed at most 0.05 m/s,
  and yaw error at most 20 degrees;
- settling root error at most 0.15 m, displayed speed at most 0.10 m/s, and yaw
  error at most 20 degrees; and
- five consecutive stable 25 Hz observations.

The object-origin standoff is a derived diagnostic only. It is not a manual
gate: object origins may be offset, the exact slot already determines the
distance, and all meaningful safety remains in table/obstacle clearance and the
existing preview, correction, IK, and contact gates. Do not store a duplicate
per-slot standoff. The shared legacy ArrivalConfig standoff fields and
arrival_ready behavior remain untouched for the placement auto-demo.

After settling, runtime preview receives the exact frozen prospective root and
the same post-step snapshot/fingerprint. It searches the ordinary global
interaction database. Pose, trajectory, grasp, root-target, and context features
therefore choose the compatible pickup motion; the slot does not dictate a clip.

A hard path/correction rejection fails immediately. A sole PoorMatch may retain
zero movement and retry while all arrival invariants remain true, bounded by the
existing arrival deadline. It may not change slot.

The first certified preview synthesizes exactly one Interact edge. PickRequest
remains target handle, affordance ID, and request ID; slot ID remains assist
diagnostic state rather than execution authority. Preflight independently reruns
ordinary selection from the same live snapshot. The preview candidate is never
submitted or cached as authority.

## Existing alignment, IK, and contact

After Preflight, the current runtime remains authoritative:

- planar root correction at most 0.25 m;
- yaw correction at most 25 degrees;
- hand positional IK at most 0.12 m;
- hand orientation IK at most 25 degrees;
- total normalized match cost at most 9.0;
- contact position/orientation and joint gates unchanged;
- object remains Free until contact succeeds;
- attachment uses the frozen authored target affordance's hand_in_object
  transform;
- source playback remains contiguous through contact, lift, and hold; and
- Carry and placement receive the existing handoff.

IK corrects small contact residuals. It does not rescue a blocked slot,
unsupported height, incompatible object, or unsuitable motion.

## Generalization contract

Checkpoint 1 proves:

- **new object pose:** translating/yaw-rotating the target maps the same authored
  slots and preserves deterministic behavior;
- **new player start:** different clear starts choose the best reachable slot and
  use ordinary locomotion;
- **compatible novel object:** a new object profile with a compatible authored
  grasp affordance and explicit ordered slot template passes headless pickup
  selection; and
- **unsupported object:** missing slots, incompatible hand/geometry, blocked
  routes, or unavailable motion fail explicitly.

Visual appearance and semantic category are irrelevant. The system does not
infer grasps. A future affordance model supplies the grasp frame, hand, approach
direction, clearance, and authored/template slot set; Smart Pickup consumes that
boundary unchanged.

## Failure behavior

Failure before Preflight leaves the object Free and ordinary locomotion active.
Every failure has one stable primary reason:

- NoAuthoredSlot;
- InvalidGeometry;
- OutsideTravelEnvelope;
- TableBlocked;
- ObstacleBlocked;
- AllSlotsBlocked;
- TargetUnavailable;
- TargetChanged;
- SlotChanged;
- RuntimeChanged;
- ArrivalDeadline;
- PoorMatch;
- FinalPreviewRejected; or
- Cancelled.

There is no slot hopping, runtime slot generation, root snap, pose snap,
threshold expansion, arbitrary clip fallback, second request, or target switch.

## Diagnostics

The native scene draws every mapped authored slot:

- red for ineligible/blocked;
- green for eligible;
- cyan for frozen;
- a direct navigation segment to the winner; and
- the target object's planar slot frame.

The overlay and structured evidence include target/affordance/slot IDs, authored
local and mapped root values, route length, heading cost, first rejection reason,
root/yaw/speed errors, derived object-origin and bounds-center distances, settle
count, final preview result/cost/frames, snapshot fingerprints, request count,
and build/data/configuration hashes.

## Tests

Raylib-free TDD coverage must include:

1. slot validation, nonzero/unique IDs, exact ordered metadata comparison, and
   legacy empty-slot behavior;
2. object-planar mapping, translation/yaw invariance, ignored object pitch/roll,
   projection failure, and live-root-Y preservation;
3. direct swept route validation against rotated tables, the actual immutable
   controller obstacle arrays, collisions between endpoints, obstacle Y
   separation, and target-route exemption with derived-distance diagnostics;
4. route length/heading/ID selection, exact ties, non-finite input, overflow,
   blocked alternatives, and frozen selection;
5. target/affordance/slot mutation and cancellation;
6. simplified approach, arrival, five-tick settle, PoorMatch retry, hard final
   rejection, and one submission;
7. exact post-step snapshot/fingerprint parity between preview and scheduler;
8. global motion matching from the frozen slot with no clip restriction;
9. no root/joint/canonical-pose writes;
10. unchanged placement auto-demo Plus/Minus behavior;
11. unchanged matcher, IK, attachment, Carry, placement, and release tests; and
12. normal and release-fast-math agreement on slot IDs, selection, reasons, and
    request count.

## Graphical acceptance

The exact-25-Hz controller must complete:

    walk -> authored slot -> settle -> motion-matched pickup ->
    carry -> walk to destination -> place -> release

This acceptance uses the validated full tabletop pack, never the one-clip
diagnostic pack.

The checked-in matrix contains:

- at least three target-relative player starts separated by 0.30 m or more;
- at least one 45-degree starting-yaw difference;
- at least two frozen slot IDs across successful starts;
- the same target translated and yaw-rotated on the table;
- one direct-path-blocked case selecting another authored slot; and
- one all-slots-blocked case failing without reservation/request.

Every scenario runs ten times from reset. Slot ID, result/reason, and request
count repeat exactly. Successful attempts issue exactly one pickup request.

Evidence retains existing continuity limits: non-owned root step at most 0.20 m,
ordinary joint translation at most 0.20 m, distal lower-limb speed at most
12 m/s, every joint rotation step at most 60 degrees, quaternion norm error at
most 1e-3, grasp composition error at most 1e-5, active-hand position/orientation
error at most 0.01 m / 2 degrees through pickup and Carry, layered inactive-hand
elevation at most 0.10 m, placement root/yaw error at most 0.25 m / 25 degrees,
final object error at most 0.02 m / 10 degrees, and support gap in
[-0.005, 0.020] m. Existing placement IK request/acceptance limits remain
0.12 m / 25 degrees and 0.04 m / 15 degrees. These are pass/fail ceilings from
the first run; measured maxima are diagnostics and never calibrate the run that
produced them.

The deterministic native mode publishes one tick-numbered capture frame for
each 25 Hz JSONL row through a scoped capture-directory interface. The artifact
includes validated JSONL, nonblank phase screenshots, a video assembled at
exactly 25 fps without interpolation, and exact build/data hashes.

## Delivery order

1. Rebuild and validate the full 25 Hz tabletop pack and extract at least three
   distinct target-affordance-compatible source entries with stable provenance.
2. Add authored slot data, validation, mapping, and the extracted demo constants.
3. Add direct route eligibility and deterministic frozen-slot selection.
4. Simplify the manual assist state machine around one frozen slot.
5. Replace only manual controller activation/preview/debug wiring while
   preserving placement legacy behavior.
6. Add headless repeated-start, target-transform, blocked-slot, and compatible
   novel-object evidence.
7. Add a deterministic, fail-closed native scenario/evidence mode that delegates
   to the same production coordinator and existing placement workflow.
8. Run native walk-pick-carry-place graphical acceptance against the full pack.
9. Measure match, height, object, and slot coverage without changing authored
   slot behavior.
10. Add pickup_ground as a separate data checkpoint.
11. Compare measured failures before considering procedural or learned approach
   proposals.

## Acceptance criteria

Checkpoint 1 is accepted when authored object-planar slots owned by the grasp
affordance remove the exact
starting-position dependency across the checked-in matrix, ordinary locomotion
reaches a frozen clear slot, global motion matching selects and executes a
compatible pickup without teleportation, the complete walk-pick-carry-place
loop remains continuous at 25 Hz, and every existing safety bound remains
unchanged.
