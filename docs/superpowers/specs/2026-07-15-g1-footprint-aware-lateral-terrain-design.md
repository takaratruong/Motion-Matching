# G1 Footprint-Aware Tangential and Lateral Terrain Design

**Date:** 2026-07-15

**Status:** Approved direction; written specification awaiting user review

## Goal

Make tangential approaches and heading-preserving lateral traversal behave
correctly on multilevel terrain. The runtime must observe terrain under the
whole support and swing-foot footprint, keep planted feet coherent across
level changes, and use certified clearance and terrain IK without coupling
travel direction to heading.

The immediate work uses the current 1,770-clip G1 motion pack. If the resulting
system is safe and geometrically coherent but lateral motion still looks poor,
data augmentation becomes a separate, evidence-driven task. The runtime must
not hide missing data by turning the character toward its travel direction.

## Hard Command Contract

`desired_velocity` and `desired_rotation` are independent inputs.

- `desired_velocity` may point anywhere in the horizontal plane, including
  sideways or backward relative to heading.
- `desired_rotation` is controlled independently and remains the heading
  target. Terrain logic may read it but may not derive, replace, or rotate it.
- Traversability and clearance logic may reduce horizontal speed or stop when a
  safe step cannot be found. It must not change heading as a fallback.
- Current keyboard, camera, and gamepad controls feed this contract. A future
  dual-stick joystick can feed the same two values without changing terrain,
  matching, support, or IK code.
- Deterministic route tests provide both inputs explicitly, even when the
  present route happens to hold heading constant.

This contract applies to current desired state, predicted desired trajectories,
safe-stop handling, candidate evaluation, support correction, and IK.

## Observed Failures

### Lateral platform exit

The current runtime was driven laterally off the `stairs-standard` landing. In
the decisive interval the root centerline had moved to the lower surface while
the trailing right foot remained on the 0.36 m landing. All four matcher
terrain features reported flat ground, the selected motion switched to the
flat Takara clip, and support velocity reached 1.77134228 m/s while it tried to
reconcile the trailing foot.

The failure is therefore not merely a sparse-data symptom. The matcher observes
a one-dimensional root-centered terrain profile, while support later observes
independent root and toe surfaces. Those two subsystems disagree about the
terrain state presented to the pose.

### Near-tangential edge approach

The `mixed-multilevel` scene contains a lateral level boundary: surface height
is 0.32 m at approximately `x=0.60` and 0 m at approximately `x=0.62`. A root
centerline can remain on the low side while a foot occupies the raised side.
The current centerline-only query cannot represent that split.

### Lateral motion coverage

The GRAIL clips contain many instantaneous lateral frames, but sustained true
terrain strafes are sparse. With a one-second future-motion definition of at
least 60 degrees lateral travel while heading changes by no more than 15
degrees, only 105 usable frames appear in 21 of the 1,769 GRAIL clips.

This may limit animation quality after the geometry defect is fixed, but it
does not justify coupling heading to travel direction.

## Chosen Approach

Use a footprint-aware observation and staged pose acceptance downstream of the
existing 31-dimensional match query.

The alternatives are deliberately deferred:

1. Expanding the feature database with separate left/right terrain tracks
   would make terrain footprint information part of nearest-neighbor search,
   but requires a motion-pack schema change, a full database rebuild, new
   normalization, and recertification. It also cannot create missing sustained
   lateral motions.
2. Synthesizing or retargeting lateral terrain clips could improve style, but
   it does not repair the current query/support disagreement and would make it
   harder to tell whether geometry or data caused an improvement.

The chosen first pass keeps the accepted matcher feature schema unchanged,
fixes the geometric observation and placement path, and measures the remaining
data limitation afterward.

Automatic rotation toward travel direction is explicitly forbidden.

## Runtime Architecture

### 1. Independent command snapshot

At the start of each frame, publish one immutable command snapshot containing:

- horizontal desired travel velocity;
- desired heading rotation; and
- their independently predicted trajectories.

Terrain and IK stages receive this snapshot by const reference. Safe-stop may
return a bounded replacement velocity, including zero, but returns no heading.
The controller remains the only owner of desired heading.

### 2. Footprint-aware terrain observation

Retain the four current root-centerline terrain features exactly as matcher
features. Alongside them, construct a separate footprint observation from the
authoritative G1HF/v2 surface and walkability field.

The observation covers:

- the current root, left sole/toe, and right sole/toe surface points;
- predicted left and right foot corridors over the controller lookahead;
- the lateral area swept between consecutive foot-corridor positions; and
- surface height, normal, walkability, domain status, and level discontinuity
  for each relevant patch.

The predicted corridors use the current pose, contact phase, desired travel,
and desired heading as separate inputs. They are not formed by rotating the
heading toward the velocity. Current contact points remain authoritative for
planting; predicted points are lookahead observations only.

The observation explicitly records a root/foot surface split. A split cannot be
silently collapsed into four zero centerline features.

### 3. Staged pose acceptance

Nearest-neighbor matching still begins with the existing 31-dimensional query.
Before the selected motion pose is committed for rendering, the runtime
evaluates its named left and right leg geometry against the footprint
observation using the certified sphere/capsule clearance kernel and the staged
IK transaction. The bounded candidates in this stage are the existing 41 swing
lift candidates; this design does not add an unplanned k-nearest motion search.

- A candidate that yields valid plant targets and certified swing clearance may
  be committed.
- A finite candidate rejected for clearance, domain, budget, or reach reasons
  does not modify controller, support, history, or heading state.
- Candidate ordering and the certification work budget are fixed and logged.
- If no staged swing candidate is admissible, the controller requests the existing
  traversability safe-stop. It preserves desired heading and retries on later
  frames as the pose/contact state changes.

This stage may reject unsafe continuation into a source clip; it may not invent
a yaw change or reinterpret a turning clip as a strafe.

### 4. Contact and level transitions

A planted foot owns a world-space sole target until its contact release is
accepted. Crossing a level boundary laterally therefore behaves as a real step:

- the trailing planted foot remains on the upper level;
- the swing foot targets the lower or upper authoritative surface under its
  predicted landing footprint;
- root support is derived from the active plant set without snapping merely
  because the root centerline crossed the edge; and
- support-source changes are staged and bounded rather than applied as a
  discontinuous height correction.

Tangential entry uses the same rule. A foot touching a different level is an
explicit multilevel state, not an unobserved perturbation of a flat-ground
match.

### 5. Certified IK and clearance

The existing clearance plan remains authoritative:

- planted feet are locked to exact G1HF/v2 surface points and normals;
- swing candidates are evaluated using actual materialized foot, shin, and leg
  centers, not sampled predicted clearance;
- unresolved certification, unreachable targets, or out-of-domain queries fail
  closed into candidate rejection and safe-stop;
- IK is downstream and reversible; rejected poses leave the accepted matcher,
  support, and history state unchanged; and
- enabling IK must not mutate command heading or matcher query values.

## Deterministic Regression Gates

### Gate L1: Tangential level boundary

Add a `mixed-multilevel` route with exact binary32 waypoints
`(0.0,0.0) -> (0.62,2.0) -> (0.62,6.0)` and a fixed heading independent of
its travel vector.

The route must expose at least one root/contact-foot surface split of 0.04 m or
more. The footprint observation must report that split before or when the
contact becomes active. Because the route lies in published traversable space,
it must complete without rendered leg penetration, unbounded support
correction, or terrain-induced heading change. A safe-stop is diagnostic
failure for this gate, not a passing alternative.

### Gate L2: True lateral platform crossing

Traverse the existing certified `stairs-standard` ascent and descent while
holding heading at 90 degrees to travel. Run mirrored left- and right-leading
versions so each leg is tested as the uphill and downhill support foot. Add a
separate lateral exit from the 0.36 m landing as a stress diagnostic: it must
either make a certified coherent step or stop before committing an unsafe
step, while retaining heading.

The command and predicted desired-heading bit patterns must match an oracle that
contains no terrain-driven heading writes. A valid run must produce staged
upper/lower plant targets and must not collapse the footprint observation to a
flat query while a contacting foot remains on the other level. The certified
ascent/descent routes must complete; safe-stop passes only for the separate
0.36 m stress diagnostic when the surface is outside the certified reach
contract.

### Gate L3: Flat-ground invariance

Repeat forward, backward, and left/right travel on flat ground with at least two
fixed headings. The footprint layer and IK-off mode must not change selected
frames, matcher queries, desired heading, or accepted support behavior relative
to the certified flat baseline.

### Gate L4: Existing terrain preservation

All accepted stair, descent, ramp, multilevel, blocked-course, switching, and
25 Hz gates remain required. A tangent/lateral fix may not weaken route
completion, walkability, clearance, cleanup, determinism, or strict/fast caller
parity.

### Correct clearance measurement

Runtime acceptance must use `rendered_min_clearance` and the certified physical
diagnostics. The existing Gate D label that currently reports blocked distance
as `minimum_clearance` is not accepted as skeleton-clearance evidence.

## Data Decision Gate

After Gates L1--L4 pass with the unmodified 1,770-clip pack, perform a visual and
logged lateral-quality evaluation.

- If geometry, planting, heading, and support are correct, the geometry phase
  is complete. The overall lateral-terrain feature is not declared ready while
  its motion quality remains visibly unacceptable.
- If lateral style remains poor, measure candidate availability and selected
  source coverage for the exact failed commands. Propose a separate motion-data
  expansion only when those logs show that the matcher lacks suitable
  heading-preserving candidates.
- Any later data work must add or derive genuine independent-heading lateral
  motion. It may not relabel turning motion or add automatic heading rotation.

## Failure Behavior

Invalid footprint input, malformed terrain, outside-domain queries, exhausted
certification budgets, unreachable IK targets, or lack of an admissible pose all
fail closed. The runtime reduces travel speed through the established safe-stop
path, preserves the requested heading, preserves the last committed pose and
plant history, records the exact rejection reason, and retries without partial
state publication.

## Scope

Included now:

- decoupled command contract enforcement;
- footprint observation;
- staged candidate/IK use of the observation;
- coherent lateral and tangential contact transitions;
- corrected physical clearance gates; and
- current-pack data sufficiency measurement.

Deferred:

- joystick device/UI integration;
- a new motion-feature schema;
- synthesis, retargeting, or rebuilding of lateral motion data; and
- G1 mesh rendering, which remains the final optional visualization task.
