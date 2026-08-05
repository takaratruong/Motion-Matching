# G1 heading-general footprint planner

## Purpose

Build a kinematic terrain traversal planner that follows a constant but
arbitrary commanded heading without fixed terrain lanes, heading-specific
motion windows, or manually selected splice frames.

The planner constructs a nominal sequence of alternating sole footprints,
searches locally for terrain-valid placements, and retrieves and retargets
motion segments that realize those contacts. The current horizontal stair
traversal is a regression case, not a special planning mode.

This design is limited to constant-heading traversal. Turning and
time-varying command trajectories are deliberately deferred, but the data
contracts use local heading coordinates so that extension remains possible.

## Current behavior being replaced

The current experimental route:

- derives one direction from the staircase alignment;
- samples two fixed scene-space lanes;
- uses a selected source interval and selected drop splice frames;
- adjusts contact anchors locally but does not construct an explicit commanded
  footprint path;
- can therefore succeed on the demonstrated route without demonstrating
  heading generalization.

Those assumptions must not appear in the general planner. Existing
retargeting, full-sole terrain queries, contact validation, center-of-mass
targets, and collision correction remain reusable implementation pieces.

## Inputs and output

The planner input is:

- a G1 starting root pose and joint pose;
- a normalized constant heading in the terrain scene frame;
- requested travel distance;
- requested speed;
- a height grid and its matcher-to-scene transform;
- a motion corpus with contact and kinematic metadata;
- bounded search and quality thresholds.

The output is either:

- one 50 Hz kinematic traversal with root pose, joint pose, contact mask,
  planned footprints, segment provenance, and validation metrics; or
- a structured failure explaining whether footprint feasibility, corpus
  coverage, transition continuity, or final trajectory validation failed.

The planner must not silently return an unsupported pose, relabel a floating
foot as contact, or hold an incomplete swing at the end.

## Heading-local route frame

Let `h` be the normalized commanded heading in the scene XY plane and
`l = (-h_y, h_x)` its lateral axis. The nominal route centerline is:

`c(s) = c_0 + s h`

All contact features and motion-search features are represented in the
heading-local `(forward, lateral, vertical)` frame. Scene-space points are
used only for height-grid queries and final output.

This makes heading a runtime input. Horizontal, diagonal, and head-on routes
execute the same algorithm with different `h`; they do not select different
code paths or hardcoded clip ranges.

## Nominal footprint path

The footprint generator produces alternating left and right contacts along
the centerline. Nominal forward spacing and lateral width come from the
requested speed and corpus gait statistics rather than fixed scene
coordinates.

Each nominal footprint contains:

- foot identity;
- center XY;
- sole yaw aligned to the commanded heading;
- nominal contact time and gait phase;
- predecessor contact identity;
- desired forward and lateral displacement from the predecessor.

The first footprint is derived from the actual starting support state. The
last footprint must finish a complete contact phase; the planner may shorten
the requested distance by less than one nominal step rather than terminate in
mid-swing.

## Terrain-valid footprint candidates

For every nominal footprint, the planner enumerates a bounded local search in
heading-local forward and lateral offsets. The search is centered on the
nominal point and ordered by deviation cost.

For each candidate:

1. Transform the complete sole sample pattern into the scene frame.
2. Query height beneath every sole sample.
3. Reject placements that cross a riser, edge, void, or disallowed slope.
4. Assign contact height from the supporting surface, not from the center
   sample alone.
5. Retain a configurable safety margin from terrain discontinuities.
6. Reject step height and relative foot separation outside robot and corpus
   limits.

The candidate search may move a footprint in XY. Merely lifting the nominal
point in Z is insufficient because a center can be valid while half the sole
is unsupported.

The output of this stage is a layered contact graph: each nominal step index
has zero or more terrain-valid candidate footprints.

## Motion corpus index

The corpus is indexed by contact-to-contact windows. Each window records:

- source clip and frame interval;
- starting support foot and contact phase;
- ending support state;
- root displacement and yaw in the starting-heading frame;
- next-foot displacement in forward, lateral, and vertical coordinates;
- start and end velocity;
- contact duration and flight duration;
- foot clearance envelope;
- joint, root, and center-of-mass boundary state;
- whether the window contains authenticated contact or an explicitly planned
  flight phase.

Indexing is offline and independent of a particular scene. Runtime retrieval
queries this index using the desired contact transition, speed, phase, and
height change.

## Candidate realization

For every edge between consecutive footprint candidates:

1. Retrieve the nearest corpus windows with matching support phase and local
   contact displacement.
2. Apply an SE(2) transform so the source heading matches the commanded
   heading and the source stance sole matches the predecessor footprint.
3. Retarget the swing foot toward the successor footprint while keeping the
   stance sole locked.
4. Preserve the source center-of-mass offset relative to the active support
   polygon.
5. Correct swing clearance against the terrain using a rate-limited target
   envelope.
6. Reject the realization if the complete motion violates contact,
   collision, reachability, joint-step, root-step, or support constraints.

Motion retrieval proposes kinematics; it never overrides terrain validity.
Retargeting is bounded. A large deformation must be rejected so that missing
corpus coverage remains visible rather than producing an untrackable pose.

## Sequence search

Use bounded beam search over the layered contact graph. A search state holds:

- current footprint candidate;
- support phase;
- realized terminal body state;
- accumulated progress;
- accumulated cost;
- selected motion-window provenance.

The transition cost includes:

- commanded centerline and distance deviation;
- footprint displacement from its nominal position;
- motion-retrieval feature error;
- joint and root boundary discontinuity;
- stance drift and sole contact error;
- swing collision and clearance margin;
- center-of-mass support error;
- foot sliding;
- excessive double support, hover, or off-edge dwell;
- bounded retargeting magnitude.

Hard validation failures are not converted into large soft costs; the edge is
removed. The beam retains multiple contact and motion hypotheses so a locally
nearest footprint cannot force a bad later transition.

## Final validation

Every returned traversal must pass an independent replay validator that did
not participate in candidate scoring. It verifies:

- the complete sole footprint for every declared contact;
- contact labels against measured sole height and velocity;
- stance drift over each complete support interval;
- minimum swing and body clearance;
- no unsupported interval outside an explicitly planned flight;
- no terminal mid-swing or frozen unsupported foot;
- joint and root position continuity;
- center-of-mass relationship to the active support polygon;
- commanded heading and progress error;
- provenance for every output frame.

The planner reports both absolute metrics and the worst frame/contact for each
metric.

## Failure behavior

Failures are classified as:

- `no_terrain_footprint`: no full-sole placement exists near a nominal step;
- `no_motion_coverage`: terrain contacts exist but the corpus has no nearby
  contact transition;
- `retarget_unreachable`: a retrieved window cannot reach the contacts within
  deformation limits;
- `sequence_dead_end`: individual edges exist but no complete sequence
  survives beam search;
- `validation_failed`: a complete candidate fails independent replay.

The failure record includes the heading, nominal step index, attempted
footprints, retrieved windows, and rejection reasons. This lets corpus
coverage problems be distinguished from planner bugs.

## Testing and acceptance

### Unit tests

- heading-local transforms round-trip for arbitrary headings;
- nominal footprints rotate and translate without changing local geometry;
- full-sole search rejects edge-straddling center-valid contacts;
- corpus queries are invariant to global heading;
- stance alignment preserves the complete planted sole;
- beam search selects a slightly displaced feasible contact over an
  infeasible nominal contact;
- incomplete terminal swings are rejected.

### Integration tests

Run the same planner configuration, without frame or lane changes, for at
least:

- horizontal traversal at the current heading;
- the same route rotated by 45 degrees;
- a head-on stair heading;
- two additional constant headings that cross different stair features;
- shifted start positions;
- perturbed tread widths and heights within corpus reachability.

### Acceptance criteria

The implementation is accepted only if:

1. no runtime configuration contains fixed scene lanes or heading-specific
   source frame selections;
2. the current horizontal regression remains complete and terrain-valid;
3. at least three distinct headings produce independently validated
   traversals using the same planner;
4. invalid headings or insufficient corpus coverage return a classified
   failure rather than a malformed motion;
5. every declared contact has a supported full sole and measured contact
   consistency;
6. final output ends at a complete contact phase.

## Deferred scope

The following are intentionally deferred:

- time-varying headings and online turning;
- Sonic tracking and physics robustness;
- learned depth-to-contact prediction;
- real-time latency optimization;
- automatic corpus augmentation.

The local-coordinate contracts, contact graph, and corpus index should remain
usable when those capabilities are added later.
