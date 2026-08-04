# G1 Stair Contact Motion Graph Plan

## Goal

Build an offline, stair-relative motion graph that can answer two separate
questions:

1. Which complete ascent, descent, lateral, and diagonal traversals already
   exist in the authenticated motion corpus?
2. Which direction changes require a new connector, and can that connector be
   synthesized offline from one or more source motions while preserving exact
   terrain contact?

Runtime motion matching remains responsible for pose continuity and motion
prior retrieval. The graph is responsible for persistent global route intent
and reachability across contact boundaries.

## Evidence and prior work

The current fixed-horizon matcher searches globally across its entire clip
inventory, but exact two-edge lookahead still reaches a node with no valid
outgoing action. Adding one short contact-transfer action opens that node and
allows the 345-frame route to execute, but locally valid continuation reaches
only 0.327 of the commanded pivot progress. A transient command-change flag
cannot repair this because the requested transition must survive until a later
safe contact boundary.

This architecture follows three established ideas:

- Kovar, Gleicher, and Pighin's Motion Graphs represent original motion pieces
  and generated transitions as directed graph edges, then search graph walks
  that satisfy path constraints:
  https://graphics.cs.wisc.edu/Papers/2002/KGP02/
- Rose et al. generate motion transitions with spacetime and inverse-kinematic
  constraints:
  https://eecs.vuse.vanderbilt.edu/People/bobbyb/pubs/spacetime96.html
- Wu and Popović optimize per-footstep end-effector trajectories for forward,
  backward, sideways, turning, and stair locomotion:
  https://grail.cs.washington.edu/projects/loco/

Two negative boundaries from the literature matter here. Kovar's motion-graph
construction uses interpolation only where source motions are already
geometrically similar; it does not claim that blending can create a
semantically different contact sequence. Holden et al.'s Learned Motion
Matching uses sparse foot-contact terrain samples and explicitly relies on
avoiding extrapolation. Increasing our terrain-grid density therefore cannot
create a missing sideways stair step. PFNN can emit terrain-adaptive motion
because it is trained end-to-end on motions fitted to many environments, not
because phase alone repairs absent contact coverage.

The closest match to the required synthesis step is Wu and Popović's
per-footstep end-effector planner: optimize a short swing/contact trajectory
against terrain and task constraints, with offline optimization supplying the
motion family. Our implementation keeps GRAIL poses as the prior and uses this
planner only for graph holes, rather than replacing the whole database-driven
system.

## Coordinate and state contract

The graph uses one fixed staircase frame:

- `u`: ascent direction;
- `v`: lateral direction across the stair width;
- `h`: vertical;
- tread index: quantized from exact query-terrain height;
- lateral cell: quantized in the `v` direction;
- heading bin: eight 45-degree bins relative to `u`.

A graph decision node is not merely an `(u, v)` intersection. It contains:

- root `(u, v)` grid cell and root-height level;
- root heading bin and planar velocity bin;
- left and right foothold `(tread, lateral-cell)` pairs;
- support mask;
- most recently landed foot;
- gait/contact phase;
- exact source or optimized boundary pose identity.

The first graph exposes only safe double-support decision nodes. Single-support
states remain internal to edges. This keeps graph search small without
discarding the phase and foothold information needed to determine whether two
edges can be connected.

## Directed edges

Each edge owns:

- start and end node identities;
- one of eight commanded traversal headings;
- source clip/frame intervals or optimized splice provenance;
- complete root, joint, foot, sole, support, and terrain traces;
- commanded displacement and yaw outcome;
- exact-contact validation result;
- slide, smoothness, and source-deviation costs.

An authenticated source edge is immutable. An optimized connector is a new
artifact with its own deterministic identity; it never edits the source
motion.

## Offline connector optimization

For a missing transition from edge A to edge B, optimize a bounded window
containing A's suffix and B's prefix. Variables are:

- root position/orientation trajectory;
- G1 joint trajectory;
- optional source-time warp;
- swing-foot path and touchdown pose;
- pelvis-height correction;
- splice duration.

Hard constraints:

- exact boundary pose and velocity continuity outside the optimization window;
- source-labelled support feet remain fixed to their stair footholds;
- touchdown sole samples share the intended tread surface;
- unsupported sole and lower-leg geometry clear the staircase;
- joint limits and a 13 rad/s output-speed ceiling;
- prescribed contact order and terminal graph node.

Soft costs:

- deviation from the source motions;
- global stair-frame path and heading error;
- joint acceleration/jerk;
- root acceleration/jerk;
- stance slip;
- timing distortion.

The first implementation is kinematic. A compact support-polygon/COM
constraint is added before any connector is declared trackable; physics and
Sonic remain out of scope until the kinematic graph is reliable.

## Measured correction to the baseline

The first 17-route extraction is not a valid synthesis inventory. Its ankle
and labelled-support metrics pass, but a full G1 sole audit finds repeated
penetration of the next stair riser, commonly between 0.20 and 0.37 m. A
bounded 26-frame double-support splice also achieves sub-millimetre ankle
targets, a 4.5 mm outgoing seam, and a 2.16 rad/s speed ceiling while its right
sole still penetrates the riser by 0.189 m. Ankle feasibility is therefore not
a proxy for stair feasibility.

Every graph edge is now filtered by the same MuJoCo G1 sole geometry and
authenticated query-height field used for final validation. The edge is
discarded when any sole sample in its complete landing-to-landing interval is
below `-0.025 m`. Route-level success may identify an artifact worth auditing,
but it cannot admit an edge by itself.

The connector experiment also establishes two boundary rules:

- use a backward difference for an incoming endpoint and a forward difference
  for an outgoing endpoint;
- do not use the saved motion-matcher velocity at an unplayed source boundary,
  because it describes the source clip that was discarded. A generated edge
  does preserve and authenticate its emitted terminal joint velocity when it
  becomes the incoming edge for a later search.

The fixed-endpoint double-support connector remains a useful diagnostic, but
is not added to the graph. If the sole-filtered inventory has no feasible
90-degree boundary pair, the optimizer must move the touchdown/contact
schedule and swing path rather than interpolate between collision-invalid
endpoints.

A second audit adds measured output speed to edge admission. The only
sole-valid elevated endpoint in the old matrix contains a `0.8306 rad` joint
jump in one 20 ms frame (`41.53 rad/s`). Once every edge is constrained by
finite differences of emitted joint positions, rather than saved matcher
velocities, the old 97-edge graph collapses to one 5-frame edge with no
successor. This is the honest baseline for the visual stutter and freeze.

The first three completed 12,646-clip full-corpus shards do not add directional
connectivity. Their safe prefixes contribute 15 variants of the same
heading-bin-1 reset/approach edge between only two quantized nodes, with zero
exact or quantized successors. More clips therefore improve neither the
missing turn nor the graph topology under the current retrieval/assembly
rule; synthesized contact actions are still required.

The first synthesized alternative starts from that one clean boundary. A
single 90-degree pivot step is either unreachable or places the sole through a
riser. Two alternating 45-degree transfers pass sole and speed checks, but
collapse the terminal stance to about `0.12 m`; that result is rejected after
visual inspection. Adding a hard `0.18--0.40 m` stance-width contract requires
three alternating 30-degree transfers. With foothold offsets expressed in the
fixed staircase frame, both directions initially passed the nonpenetration,
ankle-target, stance-width, and speed gates:

- left 90 degrees: `-0.022569 m` minimum full-sole clearance,
  `4.976912 rad/s` maximum finite-difference joint speed, and
  `0.266/0.214/0.191 m` terminal stance widths after the three steps;
- right 90 degrees: `-0.018216 m` minimum full-sole clearance,
  `5.059051 rad/s` maximum finite-difference joint speed, and
  `0.270/0.224/0.194 m` stance widths;
- no online motion-matcher jump or fixed outgoing source endpoint is used.

Those paths are subsequently rejected. The acceptance check constrained only
the ankle origins and the minimum of all sole clearances. At their terminal
states one nominally planted sole spans `0.152--0.156 m` of surface residual,
so it is tilted across or floating above a riser even though no sample
penetrates by more than `0.025 m`. The analogous provisional 180-degree
reversal uses six alternating transfers and reports
`-0.023155 m`, its maximum finite-difference joint speed is
`5.809293 rad/s`, but its two terminal sole spreads are `0.177` and
`0.108 m`; it is rejected for the same reason.

Pure lateral translation is asymmetric at the tested boundary. A `+0.20 m`
stair-lateral request passes the earlier ankle-level test, while the opposite
request reaches only
`-0.12 m` under the same foothold lattice and then loses IK-valid or
stance-width-valid successors. Neither is promoted until it passes the new
full-sole contact contract.

Position continuity and a speed ceiling are not sufficient smoothness
contracts. The first provisional pivots had `200--291 rad/s^2` acceleration
spikes exactly where the swing foot changed, matching the visible step-join
stutter. The cause was geometric rather than a missing source clip: the swing
height used a parabolic bump whose vertical derivative was nonzero at liftoff
and touchdown. Replacing it with a quartic bump that has zero endpoint
velocity reduces the left-90 result from `4.976912` to `2.134343 rad/s` peak
joint speed and from `248.843` to `32.954 rad/s^2` peak acceleration while
improving minimum sole clearance from `-0.022569` to `-0.022417 m`. A
minimum-jerk horizontal time law and sixth-order swing-height bump, both with
zero endpoint velocity and acceleration, reduce the same left-90 path again
to `2.130139 rad/s` speed and `16.088330 rad/s^2` acceleration with
`-0.022532 m` clearance. Finite-difference joint acceleration is now a
recorded and gated output metric, with a first qualification ceiling of
`100 rad/s^2`. These smoother ankle-level results remain rejected by the
full-sole contact audit above.

The corrected synthesis constrains all seven oriented sole samples for both
feet throughout every step. A planted sole keeps its complete world pose; the
swing sole follows a rigid yaw-and-translation path and must finish with every
sample within `0.025 m` of the queried surface. The source endpoint is used
only as a stationary pose seed, not played as an unaudited incoming clip; two
stationary history frames are prepended to the speed and acceleration audit.
The first 30-degree transfer passes this stricter contract with `0.013763 m`
maximum terminal sole error, `0.009537 m` footprint spread,
`2.027818 rad/s` maximum speed, and `13.811058 rad/s^2` maximum acceleration.
Adding acceleration-constrained IK and refining the offline foothold lattice
from `0.08` to `0.04 m` qualifies the complete three-transfer left-90 path.
At 91 frames per transfer it reports `-0.014300 m` minimum sole clearance,
`0.014300 m` maximum terminal sole error, `0.011677 m` maximum footprint
spread, `3.719281 rad/s` maximum joint speed, and `50.0 rad/s^2` maximum
acceleration. The result is serialized as an `artifact_kind=optimized` graph
edge with its connector identity and provenance. The high-acceleration source
edge is used only to locate the stationary pose seed and is excluded from the
qualified optimized graph.

The accepted route is no longer collapsed into one opaque graph edge. Its
three inclusive 91-frame landing-to-landing segments are serialized as three
separate optimized artifacts with identical shared boundary hashes at the two
internal landings. The resulting graph reports two exact transition pairs
instead of zero, so later direction changes can be attached at real contact
intersections rather than only at the end of the entire 90-degree route.

The earlier right-turn failure was pelvis reachability, not absence of a
terrain-valid foothold. Searching the discrete pelvis-height lattice
`{0, -0.04, +0.04} m` qualifies the first two rightward 30-degree transfers
with the `-0.04 m` choice: the 60-degree path reports `-0.013879 m` minimum
sole clearance, `0.013879 m` maximum terminal sole error,
`2.194261 rad/s` maximum speed, and `13.771179 rad/s^2` maximum
acceleration. A third 30-degree transfer is unreachable from every surviving
beam endpoint. Splitting only that remaining turn into two 15-degree contact
transfers qualifies the full right-90 route. The added edges respectively
report at most `0.014579 m` and `0.014871 m` sole error, `1.375261 rad/s`
and `1.373377 rad/s` speed, and `6.540521 rad/s^2` and
`6.626275 rad/s^2` acceleration. Their graph has three route edges and two
exact transition pairs. A 361-frame stitched MuJoCo contact sheet confirms
the feet remain on the stair surfaces throughout the complete mirrored turn;
the upper body is intentionally still a static kinematic prior.

Graph schema v3 records joint acceleration, terminal sole-contact error, and
artifact kind on every edge. Extraction serializes the complete staircase
frame and dataset manifest; synthesis refuses a graph whose manifest does not
match the loaded dataset and uses the graph's authenticated ascent direction
instead of reconstructing a second frame from configuration. Node
`root_height_level` now quantizes pelvis height rather than duplicating the
already-recorded foothold surface level. Exact missing directions are reported
per incoming boundary, separately from coarse quantized-node coverage.

## Execution stages

### Stage 1: graph schema and coverage

- Implement immutable node, edge, and graph artifacts.
- Quantize stair-frame headings, footholds, and safe landing states.
- Extract candidate traversal edges from the authenticated corpus.
- Emit an eight-direction coverage matrix and missing-transition list.

### Stage 2: complete traversal baselines

- Find or compose one exact-contact path for each of:
  up, down, left, right, up-left, up-right, down-left, down-right.
- Render all eight on the same staircase.
- Reject paths that finish without commanded progress, even when every frame
  is contact-valid.

### Stage 3: optimized connectors

- Choose the first missing 90-degree direction change on the middle tread.
- Seed from the best incoming/outgoing source edges.
- Optimize one splice window.
- Validate the optimized edge with the existing emitted sole/contact oracle.
- Add the edge only if it improves graph reachability without violating the
  output-speed, slide, or clearance contracts.

### Stage 4: arbitrary command sequences

- At every safe landing node, request each of the other seven headings.
- Plan through the graph in the global stair frame.
- Report reachability, connector count, path progress, slide, clearance, and
  smoothness.
- Drive the same graph interactively in MuJoCo kinematic visualization.

## First acceptance gate

The prototype is ready for visual feedback when:

- all eight traversal directions have a rendered exact-contact path, or are
  explicitly reported as missing;
- the graph distinguishes foothold/contact-incompatible intersections;
- at least one previously missing direction-change edge is synthesized and
  passes exact sole/contact validation;
- arbitrary graph walks never freeze silently: they either return a complete
  path or an explicit missing edge;
- the original fixed-horizon matcher remains unchanged when graph mode is off.
