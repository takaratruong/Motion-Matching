# G1 Playable Tabletop Pickup Design

**Status:** Approved for implementation planning on 2026-07-14.

## Purpose

Build the first playable visual proof of the tabletop-manipulation hypothesis in
the existing Raylib locomotion demo. The player can walk a G1 character to one
tabletop object, explicitly request a pickup, watch a target-relative recorded
pickup execute, and regain locomotion control while carrying the object.

This design consumes the schema-v1, 25 Hz interaction artifacts defined by
`2026-07-14-g1-tabletop-interaction-motion-matching-design.md` and implemented by
the Gate 1 data pipeline. It does not change their wire format. All correction,
contact, and attachment limits from that design remain binding unless this
document narrows the behavior further.

The result is a whole-clip motion-matching baseline, not merely an animation
viewer: the current character state, target affordance, table/object context, and
correction envelope determine whether an interaction clip is valid and which
valid clip is selected.

## Scope

The playable checkpoint includes:

- the existing flat-ground locomotion matcher and controls;
- one table and one rigid, single-hand object in the scene;
- explicit Interact input;
- an explicit, extensible target and affordance request boundary;
- whole-clip interaction selection from the interaction feature database;
- a 0.25-second inertialized entry blend plus at most 1.00 second of bounded
  local alignment;
- sequential reach, contact, lift, and hold playback;
- contact-gated object attachment using the recorded grasp transform;
- a controllable carry state;
- genuine carry locomotion when certified carry frames exist;
- layered carry locomotion as the required fallback;
- a temporary Drop/Reset command for repeatable demonstrations; and
- live interaction diagnostics in the Raylib window.

The checkpoint does not include:

- staged frame-level approach/reach matching;
- long-distance navigation to an object;
- player-facing selection among several objects;
- learned affordance prediction;
- two-handed, deformable, or articulated-object interaction;
- general placement or physically simulated release;
- terrain-aware interaction; or
- polished production carrying animation.

The work remains isolated on `g1-manipulation-motion-matching`. It does not merge
or rebase the moving terrain-aware branch.

## Player Experience and Acceptance Flow

The required experience is:

1. Start the existing controller with a valid interaction pack.
2. Walk normally using the existing flat-ground controls.
3. Approach the single tabletop object and see it highlighted as the current
   eligible target.
4. Press Interact. The controller either reports a precise rejection and leaves
   locomotion active, or reserves the target and begins a bounded alignment.
5. See one selected interaction clip advance contiguously through reach,
   contact, lift, and hold without changing source clips mid-grasp.
6. See the object remain on the table until the contact gate passes; it must not
   teleport into the hand merely because a contact-labelled frame was reached.
7. Regain movement control while the character visibly carries the object.
8. Press Drop/Reset to restore the object and repeat the demonstration.

Success requires the complete loop. Loading data or replaying a clip without
locomotion input, target validation, contact-gated attachment, and the carry
handoff is not sufficient.

## Architecture

The runtime state flow is:

`Locomotion -> Preflight -> Align -> PickupReplay -> Hold -> Carry`

Failure before commitment returns directly to `Locomotion`. Failure after
commitment completes a safe contiguous remainder without attaching the object,
then enters `Locomotion`. Drop/Reset from `Carry` returns to `Locomotion` after
restoring the demonstration target.

Interaction behavior is implemented outside the existing monolithic
`controller.cpp`. The controller remains the owner of the window, input, camera,
ordinary locomotion matcher, and skeleton drawing. A narrow adapter supplies a
locomotion snapshot to the interaction runtime and applies the runtime's output
pose, object state, and diagnostics. The only controller changes are wiring at
those boundaries and drawing the scene/debug data.

The logical components are:

1. **Target registry and resolver** — owns stable target handles and produces an
   explicit target selection. The first resolver returns the sole registered
   eligible target; it does not embed multi-object scoring.
2. **Interaction coordinator** — owns requests, state transitions, cancellation,
   result codes, and target revalidation.
3. **Whole-clip selector** — applies hard filters, constructs the runtime query,
   computes normalized feature costs, and returns one valid pre-contact entry.
4. **Sequential interaction player** — samples the selected 25 Hz segment,
   interpolates it for rendering, applies bounded corrections, and never crosses
   its clip range.
5. **Attachment controller** — evaluates contact gates and drives the object from
   the inverse recorded hand-in-object grasp transform after attachment.
6. **Carry controller** — selects certified carry motion when available or uses
   the layered fallback while preserving the grasp.
7. **Debug view** — renders target and correction geometry plus deterministic
   state, cost, error, and result values.

Each component exposes a Raylib-free C++ interface so its behavior can be unit
tested without a window or GPU.

## Target and Request Boundary

An interaction target has a stable ID and revision plus:

- current world transform and rigid-object state;
- object dimensions;
- support/table transform and dimensions;
- one or more object-local grasp affordances; and
- eligibility, reservation, attachment, and held state.

Each affordance has a stable ID, allowed hand, object-local hand grasp frame,
object-local approach direction, and clearance envelope. For the first scene,
the affordance is authored from the chosen recorded object's canonical grasp
metadata; no inference model is involved.

Interact creates a `PickRequest` containing the target ID and revision,
affordance ID, and request sequence ID. The motion runtime consumes this explicit
request. It never chooses among household objects and never infers the player's
high-level intent.

The coordinator revalidates the same target ID and revision until commitment.
Movement, deletion, reservation by another actor, or affordance invalidation
causes a pre-commit rejection rather than silently switching targets.

The one-object implementation still uses the registry and request contract. A
future camera/raycast selector, UI cycling policy, script, or VLM can therefore
choose among several targets without changing matching, playback, attachment,
or carry code.

## Whole-Clip Selection

One candidate entry is considered for every contiguous pre-contact segment that
has a valid future path through contact, lift, and hold within one clip range.
The preferred entry is the first `REACH` frame. An earlier `APPROACH` entry may be
used only when it remains inside the one-metre interaction-controlled approach
limit inherited from the Level 1 design.

Hard filters reject candidates with:

- a hand or affordance mismatch;
- an invalid phase order or future clip range;
- incompatible object/support context;
- a target outside the interaction-controlled approach envelope;
- a residual root, hand, orientation, or time correction beyond the inherited
  hard limits;
- insufficient hand/path clearance; or
- a target state or revision mismatch.

For surviving entries, the runtime builds the same normalized 71-D feature query
as the offline database. Each group contributes the mean squared standardized
difference across its dimensions. The initial weights are pose `1.0`, trajectory
`1.0`, grasp `2.0`, root-target `2.0`, and context `1.0`; their weighted sum is
divided by the total weight. A candidate is acceptable only when that normalized
cost is at most `9.0`. These values are named, tested configuration and appear in
diagnostics rather than being hidden inside selection code.

Once selected, the source frame advances sequentially through the remainder of
the chosen segment. The whole-clip baseline does not re-search or jump to another
recording during reach, grasp, lift, or hold.

## Alignment and Playback

Selection does not mutate the recorded database. Corrections are downstream of
selection and use the existing Level 1 limits:

- at most 1.00 m of interaction-controlled local approach;
- at most 0.25 m of residual planar root warp;
- at most 25 degrees of residual yaw warp;
- at most 0.12 m of hand positional IK;
- at most 25 degrees of hand orientation IK; and
- uniform playback speed between 0.85x and 1.15x.

Align begins with a 0.25-second inertialized blend. Any interaction-controlled
local step may continue for at most 1.00 second, while root/yaw warp is
distributed through the remaining pre-contact motion. Align never snaps the
object or the character across an out-of-envelope error. Hand IK ramps in during
reach and reaches full weight at contact. Existing foot locking is applied after
root correction to protect planted contacts.

The canonical source clock remains exactly 25 Hz. Runtime time accumulates in
seconds, source frames are sampled from the selected clip range, and adjacent
poses/transforms are interpolated at the display update rate. Rendering at 60 Hz
or another rate does not create a second animation database or change contact
ordering.

The player may cancel during `Align` before commitment. At the inherited
pre-contact commit horizon, steering and arbitrary cancellation stop so the
contiguous contact/lift motion can finish safely.

## Contact, Attachment, and Hold

The object begins `Free`, becomes `Targeted` after preflight, and remains at its
scene transform throughout approach and reach.

At the demonstrated stable-contact event, attachment requires the same gates as
the Level 1 design, including:

- the same target/revision and allowed hand;
- corrected hand position within 4 cm of the grasp frame;
- corrected hand orientation within 15 degrees of the grasp frame;
- valid joint and correction limits; and
- a passing hand/object clearance check.

Failure does not teleport or attach the object. On success, the object becomes
`Attached` and is driven kinematically by the active hand using the inverse of
the single recorded object-local hand grasp transform.

The lift must raise the object origin at least 15 cm above its last stable
pre-lift height. The object must remain attached at that height for one second.
If the recorded segment ends before the one-second hold completes, the final
valid hold pose is maintained while the timer completes. The runtime then marks
the object `Held` and enters `Carry`.

## Carrying

Carrying is animated locomotion, not a frozen character translated through the
scene.

A sequence is eligible as genuine carry motion only when a contiguous post-pick
range contains at least 25 `HOLD` frames, continuous active-hand contact,
hand-in-object drift no greater than 2 cm and 10 degrees, at least 0.30 m of
planar root travel, and average planar root speed of at least 0.20 m/s. Such
sequences are searched as a dedicated carry set and remain within their clip
ranges. The debug view reports `recorded` when this path is active.

When no certified carry candidate exists, the required fallback is layered:

- the existing locomotion matcher owns the root and lower body;
- the final recorded hold supplies the active-arm carry posture;
- a partial spine blend preserves locomotion sway rather than freezing the
  torso;
- the unused arm continues a damped locomotion motion;
- active-hand IK maintains the recorded grasp; and
- the object follows the solved hand through the recorded grasp transform.

The fallback reports `layered` in diagnostics. It is expected to be less natural
than a genuine carry database, but it provides responsive, visible carrying
motion without claiming unsupported data coverage.

Drop/Reset is a debug operation, not a placement primitive. It detaches the
object, restores the configured table pose and target revision, clears the carry
layer, and returns ordinary locomotion control.

## Rendering and Diagnostics

The existing G1 skeleton renderer remains the character visualization. The first
scene draws the table and object as oriented primitive proxies from the artifact
transforms and dimensions; loading production USD meshes is not required for the
behavioral proof.

The scene also renders:

- selected-target highlighting and interaction range;
- the affordance/grasp frame and approach direction;
- the selected active hand and corrected hand target;
- requested versus applied root correction; and
- attachment and held state.

The text overlay exposes:

- artifact availability and validation errors;
- coordinator and object state;
- target and affordance IDs/revisions;
- selected clip, frame, phase, and hand;
- total and per-group match cost;
- root, yaw, hand-position, and hand-orientation correction;
- contact error and attachment/rejection reason;
- carry source (`recorded` or `layered`); and
- the latest `InteractionResult`.

Diagnostics are deterministic inputs to tests and debugging, not optional UI
polish.

## Error Handling

If the interaction pack is absent or invalid, the controller starts with
interaction disabled, reports the loader error, and preserves ordinary
locomotion. It must not partially consume an invalid pack.

Preflight failures return a single actionable reason, including target invalid,
target out of range, no compatible candidate, poor match quality, blocked path,
or correction limit exceeded. They leave locomotion and the object unchanged.

Cancellation or target invalidation before commitment inertializes back to
locomotion without attachment. After commitment, an unexpected contact or
correction failure finishes the safe contiguous character motion without
attaching the object, records failure, and returns to locomotion-ready state.

The runtime never substitutes an arbitrary clip, changes targets mid-request,
exceeds a hard correction envelope, or attaches solely because time reached a
labelled source frame.

## Testing and Verification

Raylib-free C++ tests cover:

- target registry, stable IDs/revisions, and the single-target resolver;
- `PickRequest` validation and target revalidation;
- hard filtering and deterministic lowest-cost whole-clip selection;
- clip-range-safe sequential playback and render-time interpolation;
- every coordinator transition, cancel path, and failure result;
- root/hand correction limits;
- contact gates, attachment timing, lift/hold success, and no-teleport failure;
- recorded-carry qualification and layered fallback selection;
- grasp preservation during layered locomotion; and
- Drop/Reset restoration.

An adapter test drives deterministic locomotion snapshots through the runtime
without Raylib and proves that disabled or idle interaction leaves the existing
locomotion output unchanged.

The interaction artifact tests remain the authority for schema, phase, grasp,
feature, and Python/C++ parity. Runtime tests use both synthetic edge fixtures and
a small published GRAIL pack.

The playable gate additionally requires a reproducible Raylib build. The current
environment has no discoverable Raylib installation, so the implementation plan
must establish and document a pinned local or system build path before claiming
the visual gate passes. A headless-only pass is insufficient.

The final manual acceptance run records:

- normal locomotion before Interact;
- target highlight and explicit request;
- selected clip and bounded corrections;
- sequential pickup with contact-gated attachment;
- stable lift and hold;
- resumed player control with visible carrying motion;
- recorded versus layered carry source; and
- a successful Drop/Reset and second attempt.

## Follow-on Work

After this baseline is playable and measured, the next manipulation milestone
replaces only the whole-clip transition policy with staged approach/reach motion
matching and a pre-contact commit rule. It reuses the target contract,
coordinator, correction stack, contact gates, attachment, carry handoff,
diagnostics, and acceptance scene.

Later levels add multi-object player selection, shelves, placement, articulated
doors/drawers, learned affordances, and high-level sequencing. Those systems
submit explicit target/affordance commands; they do not move household intent
selection into the low-level motion matcher.
