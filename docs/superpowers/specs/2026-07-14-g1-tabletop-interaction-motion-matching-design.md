# G1 Tabletop Interaction Motion Matching Design

Date: 2026-07-14

## Summary

Build a game-style tabletop pickup prototype for the Unitree G1 that preserves the
existing flat-ground locomotion controller and activates manipulation only after
an explicit player interaction request. The interaction controller may perform
one or two local alignment steps, selects a target-relative pickup motion from
GRAIL, commits to a contiguous grasp-and-lift segment shortly before contact, and
uses bounded root warping, hand IK, contact locking, and kinematic object
attachment to reach the authored grasp frame.

The first phase is deliberately learning-free. Its research question is whether a
small, target-relative motion database can generalize tabletop pickups to rigid
objects that were not present in that database. A production-style whole-clip
selector using the same affordance, warping, and IK stack is the required baseline.
Frame-level interaction motion matching is useful only if it improves success,
coverage, continuity, or required correction over that baseline.

The runtime flow is:

`locomotion -> explicit interact request -> local approach -> committed grasp -> lift -> hold`

Level 1 ends after a stable hold and a clean handoff back to a locomotion-ready
state. The object may remain kinematically attached while ordinary locomotion
resumes, but carrying animation quality and object placement are not Level 1
acceptance criteria.

## Context and Decision

The current flat-ground motion-matching controller already provides usable,
player-driven G1 locomotion. It remains a stable dependency rather than becoming
part of a new unified locomotion-and-manipulation search space.

GRAIL supplies 2,991 ten-second tabletop pickup motions across 685 objects. Each
sequence contains native G1 body and hand trajectories, object motion, contacts,
table metadata, and an object asset. The source is broad enough to test object
identity holdouts and data-efficiency curves without introducing a learned
controller.

The selected architecture is staged, contact-aware interaction motion matching:

1. Existing motion matching remains authoritative during ordinary locomotion.
2. The player explicitly selects a nearby object and requests interaction.
3. A preflight step validates its authored grasp affordance and the current
   character state.
4. Frame-level matching controls only the short alignment and early reach.
5. Roughly 0.5 seconds before demonstrated contact, the controller commits to one
   contiguous grasp-and-lift segment and stops arbitrary database transitions.
6. Bounded procedural correction aligns the recorded motion to the target.
7. The object attaches only when contact timing and final hand error both pass.

This follows the common game pattern of responsive locomotion, an explicit
interaction request, an authored environmental reference, a short automatic
alignment, a committed interaction animation, and procedural cleanup. The
research contribution is asking target-relative motion matching to infer the body
motion from a grasp affordance rather than authoring a complete body slot for
every object.

Two alternatives remain part of the comparison or roadmap:

- **Whole-clip selection** is the production-style Level 1 baseline. It is simpler
  and may prove sufficient.
- **One unified locomotion and interaction database** is deferred. It makes
  control, feature weighting, contact transitions, and diagnosis unnecessarily
  difficult for the first experiment.
- **Learned selection, residual motion, or grasp inference** is deferred until the
  deterministic system exposes a measured coverage gap that additional data,
  matching features, warping, or IK cannot address.

## Goals

- Preserve the behavior and data path of the existing flat locomotion controller.
- Start interaction only after an explicit game-style input.
- Permit at most one or two interaction-controlled alignment steps rather than
  long-range navigation or pathfinding.
- Pick up a rigid tabletop object whose identity and motions are absent from the
  interaction database.
- Require only an authored object-local grasp affordance at runtime, not a full
  final body pose or hand-authored standing slot.
- Use GRAIL's native G1 pickup data through a deterministic offline conversion and
  validation pipeline.
- Compare whole-clip selection and staged interaction motion matching under the
  same correction and evaluation conditions.
- Measure how much interaction data is actually required.
- Make match cost, correction magnitude, contact error, rejection, and state
  transitions directly observable.
- Keep all implementation and artifacts isolated from the terrain-aware branch
  and from the existing default motion-matching resources.

## Non-goals

- Learned grasp-affordance prediction, vision, or raw-mesh semantic reasoning.
- A universal grasp planner. Level 1 receives a valid grasp frame.
- Two-handed, deformable, articulated, moving, or dynamically thrown objects.
- Shelves, constrained cavities, drawers, doors, or refrigerators.
- Long-distance automatic navigation after the player presses Interact.
- High-quality carrying locomotion or placement after the pickup.
- Force closure, tactile feedback, finger-force control, or dynamic grasp
  simulation.
- Torque control, balance-policy training, sim-to-real deployment, or physical G1
  execution.
- Replacing the existing locomotion database or merging manipulation features into
  its ordinary search query.

## Level 1 Scene and Control Contract

The scene contains a flat floor, one static table, and one or more static rigid
objects. The table and each target object's world transform are known to the game
runtime. Every interactable object has at least one authored grasp affordance.

The player uses the existing movement controls normally. The game or debug demo
selects a candidate object through its normal targeting mechanism; the first demo
may use the closest valid object inside a view or distance filter. Pressing
Interact submits that object and affordance to the interaction coordinator.

An interaction request is accepted only inside the supported local-approach
region. The interaction controller may close at most 1 metre of distance, turn
the character, and play recorded local stepping from the pickup data. It does not
route around obstacles or take over ordinary traversal.

Before the commit point, the player may cancel and immediately return to normal
locomotion through an inertialized blend. After commitment, the pickup finishes
or follows its bounded failure recovery; arbitrary player steering and arbitrary
motion-database jumps remain disabled until the result is known.

## Repository and Artifact Ownership

The canonical implementation lives in the isolated
`g1-manipulation-motion-matching` branch and worktree. The terrain-aware checkout
is not modified.

Interaction-generated data lives under `resources/g1_interaction/` and must not
overwrite `resources/database.bin`, `resources/features.bin`, terrain artifacts,
or LAFAN backups. The semantic artifact set is:

- `interaction_database.bin`: canonical poses, velocities, clip ranges, active
  hand, phases, time-to-contact, contacts, and object/table context;
- `interaction_features.bin`: normalized target-relative matching features;
- `manifest.json`: schema version, coordinate and skeleton conventions, sources,
  feature groups and weights, phase rules, filters, correction limits, and frame
  counts;
- `evaluation_split.json`: object-disjoint database and held-out identities plus
  fixed subset seeds; and
- `validation_report.json`: rejected clips, numeric validation bounds, and summary
  statistics.

The implementation may reuse the repository's existing binary-array serializer,
but it must keep this interaction artifact set versioned and independently
loadable. Generated large binaries are reproducible outputs rather than a reason
to modify the existing resources in place.

## Components

### 1. Existing locomotion adapter

The existing flat controller continues to own input prediction, ordinary database
search, pose continuation, and inertialization while the interaction coordinator
is in `Locomotion`.

A narrow adapter exposes only the state needed to enter an interaction: current
root transform and velocity, current G1 pose and joint velocities, planted-foot
state, and the ability to request an inertialized handoff. Interaction code does
not reach into or reimplement the locomotion matcher's internal search.

### 2. Interaction coordinator

The coordinator owns the game-facing state machine, validates interaction
requests, selects the active affordance, switches controller ownership, processes
cancellation, and returns control. It depends on interfaces for locomotion state,
affordance queries, interaction matching, correction, object attachment, and
diagnostics; it does not implement any of those internals itself.

### 3. Grasp affordance

An affordance describes constraints on the pickup, not a finished character pose.
Its runtime data contains:

- an object-local 6-DoF grasp frame for the hand;
- one or more allowed hands;
- a grip-type tag used as a hard compatibility filter;
- an object-local approach axis and allowed approach cone;
- a clearance radius or bounds used by the simple reach-path test; and
- object dimensions used for clearance and optional soft matching context.

Level 1 uses the single tag `generic_single_hand` unless a source provides a
stronger compatible label. Object attachment derives its hand-relative transform
from the grasp frame, so an author does not have to specify the same relationship
twice.

The authored grasp frame deliberately omits a world-space root transform, final
foot locations, and full-body pose. Candidate approach roots are derived from the
database's demonstrated reach envelope and the current locomotion state.

Training clips receive the same affordance representation automatically. For each
valid single-handed pickup, the builder estimates the demonstrated hand-in-object
transform over stable contact frames and records that as the clip's grasp frame.

### 4. GRAIL interaction ingestion

The offline importer reads the tabletop category's `robot`, `objects`, and `meta`
records, plus table metadata and the USD asset only when geometry or visualization
requires it. Every source adapter produces a canonical interaction sample:

- source clip, source object, source frame, and timestamp;
- floating-root transform and velocity;
- ordered G1 body and hand state and velocities;
- left and right hand transforms and velocities;
- left and right foot transforms, velocities, and contacts;
- object transform and velocity;
- hand-object contact flags;
- table transform and dimensions; and
- active hand, interaction phase, time-to-contact, and demonstrated grasp frame.

GRAIL is already in G1 joint space. The retarget stage is therefore an explicit
coordinate, ordering, FK, and resampling adapter rather than a cross-embodiment
solve. Keeping the stage explicit allows future human or non-G1 sources to target
the same canonical contract.

Data is converted to the repository's established runtime coordinates and kept
at the locomotion database and matcher's fixed 25 Hz rate. GRAIL therefore stays
at its native sampling rate; future non-25-Hz sources use time-based translation
interpolation and normalized shortest-arc quaternion interpolation. Velocities
use canonical timestamps and never cross clip boundaries. Rendering may
interpolate canonical poses at the display rate, but that does not change motion
database indices, contact timing, or feature horizons.

### 5. Clip validation and phase derivation

Level 1 admits single-handed tabletop pickups with coherent contact and object
motion. Two-handed clips, clips with missing fields, non-finite state, state
outside the configured G1 joint limits, inconsistent frame counts, or incoherent
contact/object motion are excluded with an explicit validation reason.

The builder derives one monotonic phase label per frame:

- `approach`: all valid motion before the final one-second reach window;
- `reach`: the final second before stable hand-object contact;
- `contact`: stable contact before the object has risen 5 cm;
- `lift`: contact while the object rises at least 5 cm from its initial support
  height; and
- `hold`: post-lift contact after object vertical speed remains below 0.1 m/s for
  0.2 seconds.

Stable contact begins at the first three consecutive source contact samples for
which the hand-object relative transform changes by at most 2 cm and 10 degrees
across the window. The demonstrated grasp frame is the robust average
hand-in-object transform over the first 0.2 seconds of stable contact. A clip is
rejected if phase ordering, contact persistence, or the grasp estimate cannot be
established. All vertical lift thresholds use the object's origin height at the
last stable pre-lift sample, not the bottom of its mesh.

### 6. Target-relative interaction features

Every database frame and runtime query uses the same feature semantics. Continuous
features are grouped as follows:

- **pose continuity:** root-relative positions and velocities for the feet, hips,
  chest, and active hand, plus root planar and yaw velocity;
- **local trajectory:** future root position and facing samples over the next
  second;
- **grasp target:** active-hand position, orientation, and velocity relative to
  the grasp frame;
- **root target:** root position, facing, and velocity relative to the grasp
  frame; and
- **context:** grasp height relative to the table, approach direction, and object
  dimensions.

Object identity, raw mesh vertices, and source clip identity never enter the
distance function. Active hand, grip type, phase compatibility, contact state,
valid future horizon, and clip-range safety are hard filters rather than weighted
features.

Feature-group normalization follows the existing matcher convention. During
early local approach, pose continuity and root trajectory dominate. As predicted
contact approaches, the hand-to-grasp position and orientation weights increase.
The schedule and group weights are configuration stored in the manifest and shown
in diagnostics.

### 7. Staged interaction matcher

The matcher receives the current inertialized pose and velocity, a short desired
root trajectory, the target affordance in world space, the allowed hand, and the
current interaction phase.

It searches valid approach and reach frames at the existing controller's normal
search cadence. Candidate frames must pass hard filters and improve on the
continuation cost by the configured transition threshold. Between searches, the
selected clip advances sequentially and never crosses its clip range.

When the selected frame is within the 0.5-second commit horizon of demonstrated
contact, the controller locks to that clip's contiguous contact, lift, and hold
segment. It may still apply the approved procedural corrections, but it may not
jump to another database frame. This protects contact timing, hand intent, and
object continuity.

### 8. Whole-clip baseline

The baseline receives the same current state, affordance, filters, and normalized
feature groups. At interaction start it chooses the lowest-cost valid pre-contact
entry frame and then plays the remainder of that clip sequentially through hold,
without re-searching. It uses the exact same root warping, time scaling, IK,
contact test, attachment logic, state machine, and failure thresholds as the
staged matcher.

Only the selection and transition policy differs. This makes the comparison an
actual test of frame-level interaction matching rather than a comparison between
different cleanup stacks.

### 9. Bounded correction stack

Corrections run downstream of selected motion and may not alter database state or
matching costs.

- Planar root translation and yaw warping are distributed over the local approach
  and end at the selected segment's target alignment. Root height remains driven
  by recorded flat-ground motion.
- Hand position and orientation IK ramp in during reach and reach full weight at
  contact.
- Existing or G1-specific foot contact locking prevents root warping from causing
  visible planted-foot drift.
- Joint limits and correction limits are checked every frame.
- Time scaling stays uniform within the selected segment and cannot change
  contact ordering.

Correction magnitudes are measured from the uncorrected selected animation to the
final corrected pose. Root correction is planar translation plus shortest yaw;
hand orientation correction is the shortest quaternion angle.

Initial hard limits are:

| Correction | Maximum |
| --- | ---: |
| Interaction-controlled approach | 1.00 m |
| Residual planar root warp | 0.25 m |
| Residual root yaw warp | 25 degrees |
| Hand positional IK | 0.12 m |
| Hand orientation IK | 25 degrees |
| Uniform playback speed | 0.85x to 1.15x |

These limits are named configuration values recorded in the manifest. Evaluation
may justify making them stricter, but a result that requires exceeding them is a
rejection rather than a success.

### 10. Contact and object state

The object has the states `Free`, `Targeted`, `Attached`, and `Held`. Merely
playing a contact-labelled frame does not attach it.

Attachment requires all of the following on the demonstrated contact event:

- the same target and affordance remain valid;
- the selected hand matches the affordance;
- corrected hand position is within 4 cm of the grasp frame;
- corrected hand orientation is within 15 degrees of the grasp frame;
- joint and correction limits pass; and
- the simple hand/object clearance test passes.

After attachment, Level 1 drives the object kinematically using the inverse of the
authored object-local hand grasp frame. A successful lift raises the object origin
at least 15 cm from its last stable pre-lift height and maintains attachment for
one second in `Held`.

### 11. Diagnostics

The live debug view and deterministic evaluation log expose:

- coordinator and object state;
- target object and affordance frame;
- active hand and phase;
- current source clip and frame;
- incumbent, selected, and per-feature-group costs;
- transition and commit events;
- requested and applied root warp, time scale, and hand IK correction;
- planted-foot drift;
- hand position/orientation error at contact;
- attach, reject, cancel, and recovery reasons; and
- per-search runtime.

The scene view renders the grasp frame, approach cone, supported root region,
predicted local trajectory, corrected hand target, and correction-envelope
violations. Diagnostics are part of the acceptance harness, not optional polish.

## Runtime State Machine and Data Flow

### `Locomotion`

The existing controller owns the character. An explicit Interact input supplies a
target object and affordance to `Preflight`.

### `Preflight`

The coordinator verifies affordance validity, active-hand compatibility, local
approach distance, coarse root/hand reachability, target and table stability,
simple approach and hand-path clearance, correction bounds, and the availability
of a database candidate below the configured quality threshold. Clearance uses a
capsule sweep along the candidate warped root path and a sphere sweep, sized by
the affordance clearance radius, along the candidate warped hand path. The target
object is exempt from the final hand sweep at contact; the table and all other
scene colliders are not.

Failure leaves locomotion active and emits one actionable rejection reason.
Success reserves the target and transfers the current pose and velocities to
`LocalApproach` through inertialization.

### `LocalApproach`

The interaction matcher owns one or two local steps and early reach. The player
may cancel. Loss of the target, loss of clearance, an invalidated affordance, or a
required out-of-envelope correction aborts through an inertialized blend back to
`Locomotion`.

### `CommittedGrasp`

At the commit horizon, arbitrary matching and ordinary steering stop. The chosen
clip advances sequentially while root warping, time scaling, hand IK, and foot
locking finish alignment. Cancellation is no longer immediate because contact
continuity takes priority.

### `AttachLift`

At the demonstrated contact event, final contact gates decide whether the object
attaches. A passing attachment advances through the selected lift. A failed gate
does not teleport or attach the object; the safe remainder of the selected motion
finishes without attachment and then returns control with a recorded failure.

### `Hold`

The object must remain attached and at least 15 cm above its support height for
one second. The pickup is then successful. The controller exposes a
locomotion-ready handoff. Ordinary movement may resume with the object still
attached for demonstration purposes, but a dedicated carry matcher, carry pose,
and placement behavior belong to later levels.

## Error Handling

The offline builder fails without publishing an artifact set when it encounters:

- missing or inconsistent GRAIL robot, object, contact, or table records;
- skeleton ordering or coordinate-conversion mismatch;
- non-finite transforms, velocities, features, or quaternion norms;
- frame-count, timestamp, or clip-range inconsistency;
- cross-clip derivative or interpolation leakage;
- invalid phase ordering or insufficient stable contact;
- an incoherent demonstrated grasp transform;
- an object-identity leak between database and held-out splits; or
- a feature or manifest count that disagrees with the interaction database.

Invalid source clips may be excluded only when their identity and exact reason
are written to the validation report. The normal build reports the included and
excluded counts and fails if the requested evaluation split cannot be satisfied.

The runtime rejects an interaction before commitment when target state,
reachability, clearance, match quality, or correction bounds fail. It never
silently falls back to an arbitrary clip, an extreme warp, or unconditional
attachment. After commitment, unexpected failure completes a safe contiguous
motion without attaching the object and returns to locomotion-ready state.

## Testing and Evaluation

### Offline correctness tests

- Source-to-canonical and canonical-to-runtime FK agree on sampled frames for
  every G1 body within 1 mm positional error and 0.1 degree rotational error.
- Resampling preserves source duration within one 25 Hz frame and does not cross
  clip boundaries.
- Contact, phase, time-to-contact, and demonstrated grasp derivation pass fixed
  synthetic fixtures and representative GRAIL clips.
- Object-relative transforms are invariant under common world translation and
  yaw rotation of the character, table, and object.
- Feature normalization, hard filters, clip ranges, and held-out identity checks
  are deterministic.
- Every artifact header, frame count, skeleton signature, and manifest field
  agrees before runtime loading.

### Runtime behavior tests

- No interaction begins without explicit Interact input.
- A request outside the local approach region leaves locomotion unchanged.
- Cancellation before commitment restores locomotion without an object attach.
- Database indices remain sequential between searches and throughout the
  committed segment.
- Contact attachment cannot occur early or outside final contact tolerances.
- Every correction stays inside its configured envelope.
- Failure after commitment never teleports or attaches the object.
- The same deterministic input and artifact set reproduce selection, correction,
  state transitions, and result.

### Held-out generalization experiment

The primary evaluation excludes every motion belonging to 20 selected object
identities from the interaction database. The held-out objects span the supported
single-hand grasp heights, orientations, approach directions, and dimensions.
They retain authored grasp affordances, because learned affordance inference is
not part of Level 1.

The deterministic harness runs 300 trials:

- 20 held-out objects;
- 5 valid tabletop positions or yaw orientations per object; and
- 3 valid initial character approach poses per placement.

Trials remain inside the declared supported table, grasp, and approach envelope.
Out-of-envelope trials separately test correct rejection and do not inflate the
supported success denominator.

Both the whole-clip baseline and staged matcher run with interaction databases of
10, 25, 50, 100, and all valid database clips. Small subsets use fixed,
stratified seeds so grasp height, hand, approach direction, and root displacement
remain represented. Every condition uses the same trials and correction settings.

### Metrics

- overall pickup success on supported trials;
- accepted coverage, conditional success, and correct rejection;
- hand position and orientation error at contact;
- root warp, time scaling, and hand IK magnitude;
- planted-foot displacement;
- pose and velocity discontinuity at controller transitions;
- time from Interact to contact and stable hold;
- matcher and correction runtime, including 95th percentile;
- failure reasons; and
- source-clip diversity among successful trials.

### Level 1 acceptance criteria

- At least 80% of supported held-out trials complete attachment, a 15 cm lift,
  and a one-second stable hold.
- No successful trial exceeds the correction envelope or attachment tolerances.
- Runtime remains real-time, with 95th-percentile matching and correction below
  one 25 Hz motion-update frame on the evaluation machine.
- The playable demo supports normal walking, explicit Interact, at most one or two
  local alignment steps, pickup, stable hold, and a locomotion-ready handoff.
- Results and failures are reproducible from logged inputs and artifact versions.

The staged matcher justifies its added complexity only if, relative to whole-clip
selection, it either:

- improves overall success by at least 10 percentage points; or
- reduces either median normalized correction score or median pose-continuity
  transition cost by at least 20% without reducing success.

Normalized correction score is the root-mean-square of residual root translation,
root yaw, hand translation, and hand orientation, with each divided by its hard
limit. Pose-continuity transition cost is the normalized pose-continuity feature
group evaluated immediately before each selected transition.

If neither condition holds, Level 1 concludes that production-style whole-clip
selection is sufficient for this interaction class. That is a valid result, not a
reason to hide the baseline.

If the full valid database passes the 80% Level 1 success criterion, the
data-efficiency result is the smallest fixed database subset whose overall
success is within five percentage points of that full database. If the full
database does not pass, the curve is still reported but no subset is called
sufficient. This directly tests the hypothesis that manipulation matching does
not require much data without allowing a weak full-database result to define an
artificially easy target.

## Validation Gates

Implementation advances through reversible gates:

1. **Data replay:** canonical GRAIL G1, hand, object, and table motion reproduces
   representative sources with correct contacts and no matching or correction.
2. **Whole-clip baseline:** one known and then held-out objects complete the full
   request, local alignment, contact, lift, hold, and return flow.
3. **Staged matching:** early approach transitions and the contact commit rule pass
   deterministic state and range tests with cleanup disabled.
4. **Bounded cleanup:** root warping, hand IK, foot locking, and attachment each
   improve their target metric and remain independently disableable.
5. **Generalization:** the complete object-disjoint experiment runs for both
   selectors and every database size.
6. **Playable demo:** the player can repeatedly walk, request a pickup, observe a
   successful or explainable rejected interaction, and regain control.

No later gate may obscure a failed earlier gate. In particular, IK and attachment
cannot be used to claim that matching passed when raw selection or playback is
incorrect.

## Learning Escalation Rule

Learning is not introduced merely because a result looks imperfect. After the
deterministic evaluation, failures are grouped into data coverage, selection,
transition, target alignment, collision, and contact categories.

A learned component becomes a justified follow-up only if a substantial set of
valid targets remains unsupported after feature-weight tuning and bounded
correction, and the nearest demonstrations contain an appropriate interaction
whose residual cannot be represented by the deterministic stack. The whole-clip
and staged deterministic results remain permanent baselines.

The expected future learned component is an affordance model that proposes the
grasp frame. It should feed the same affordance contract rather than changing the
motion controller interface.

## Roadmap Beyond Level 1

1. **Tabletop pickup:** this specification—explicit request, local approach,
   unseen rigid object, pickup, and stable hold.
2. **Shelves:** add height and constrained-clearance coverage while keeping the
   same object affordance and contact-aware matcher.
3. **Articulated objects:** add handle affordances, joint state, synchronized
   object trajectories, and committed open/close interactions for drawers and
   doors.
4. **Chained game interactions:** compose locomotion, refrigerator-door opening,
   object pickup, carrying, placement, and release through a higher-level
   interaction sequencer.

Each level receives its own specification and retains the accepted baselines from
the previous level.

## Implementation Order

1. Freeze the canonical clip, affordance, phase, artifact, and evaluation-split
   contracts.
2. Build and validate representative GRAIL tabletop replay.
3. Derive phases, demonstrated grasps, target-relative features, and object-held
   splits.
4. Add the explicit interaction coordinator around the unchanged flat controller.
5. Implement the whole-clip selection baseline and bounded correction stack.
6. Add staged approach/reach matching and the pre-contact commit rule.
7. Add deterministic diagnostics and automated held-out evaluation.
8. Tune only through recorded configuration, run the data-efficiency study, and
   produce the playable comparison demo.

The detailed implementation plan will assign exact files, tests, and commits only
after this design is reviewed and approved.

## References

- NVIDIA GRAIL dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>
- Holden et al., Learned Motion Matching: <https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/2EVXGPN6ynTrrJaaAUHmZS/060e223cce6a84e47e925105cac5e17f/Learned_Motion_Matching.pdf>
- EA, Environmental and Motion Matched Interactions: <https://www.gdcvault.com/play/1027465/Animation-Summit-Environmental-and-Motion>
- Unreal Engine Smart Objects overview: <https://dev.epicgames.com/documentation/en-us/unreal-engine/smart-objects-in-unreal-engine---overview>
- Unreal Engine Motion Warping: <https://dev.epicgames.com/documentation/unreal-engine/motion-warping-in-unreal-engine>
