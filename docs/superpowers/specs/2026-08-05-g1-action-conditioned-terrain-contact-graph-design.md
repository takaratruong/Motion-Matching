# G1 Action-Conditioned Terrain Contact Graph

## Purpose

Replace the nominal-footprint-first terrain planner with an offline contact
graph whose edges are realizable motion-library actions. The planner must
produce natural 50 Hz G1 kinematics across arbitrary constant headings and
arbitrary height-field objects without object labels, heading-specific lanes,
or selected source-frame ranges.

The method is general over terrain geometry, not universally capable. Terrain
outside the robot's reachable workspace or the motion library's coverage must
produce a classified failure rather than a malformed motion.

Sonic tracking and online physics control remain out of scope. An optional
whole-body dynamics refinement stage may consume a completed kinematic route,
but it is not part of graph feasibility.

## Why the Existing Decomposition Fails

The existing planner first lays a flat-ground contact grid over the commanded
line, moves each footprint independently to a terrain-valid location, and only
then retrieves GRAIL actions for pairs of endpoints.

This is rotation-equivariant in coordinates but not motion-aware in search.
At diagonal terrain boundaries it can:

- move successive contacts to opposite sides of a discontinuity;
- request a same-foot stride far outside the source gait distribution;
- choose endpoint-compatible clips whose swing phase intersects terrain;
- discover action infeasibility only after the complete route is selected;
- repeatedly choose a different infeasible clip with the same swept-volume
  defect.

The general correction is to make motion actions the graph edges. Footprints
are consequences of feasible action expansions, not independently planned
targets that the library must fit afterward.

## Alternatives Considered

### Geometry-first footstep optimization

Optimize a sequence of full-sole poses on the height field and then fit source
motions. This gives strong geometric coverage but can return contact sequences
that no natural source action can realize. It retains the central failure of
the current planner.

### Full contact-implicit trajectory optimization

Optimize the complete motion in a differentiable simulator, as in Direct
Simulation-based Multiple Shooting. This can resolve contact, friction, and
actuation jointly, but it is expensive, nonconvex, and dependent on a useful
initial reference. It is appropriate as an optional route-refinement or
library-augmentation stage.

### Action-conditioned contact graph

Expand authenticated library actions from the current contact state, retarget
their landing poses onto terrain, and reject the complete swept motion before
adding the graph edge. This preserves source-motion naturalness, exposes
coverage failures, and uses the same algorithm for every terrain orientation.
This is the selected approach.

## Inputs and Outputs

Inputs:

- G1 root pose, joint pose, velocity estimate, and measured support state;
- a constant commanded scene heading, distance, and target speed;
- a height field with scene/world transforms;
- a motion-action library with contact, kinematic, and boundary metadata;
- bounded retargeting, terrain, continuity, and search thresholds.

Successful output:

- root position and orientation at 50 Hz;
- 29 joint positions at 50 Hz;
- measured and source contact masks;
- realized sole trajectories;
- selected action and source-frame provenance;
- graph node and edge records;
- independent terrain, contact, continuity, command, and naturalness metrics.

Failure output:

- failure class;
- graph depth and progress;
- attempted actions and landing corrections;
- rejection counts by terrain, sweep, reachability, transition, and validation
  reason;
- the best partial state when one exists.

## Terrain Model

The planner consumes only height queries and their spatial gradients or finite
differences. It does not classify stairs, curbs, ramps, blocks, beams, or
edges.

Every sole placement is evaluated using the complete sole sample set plus a
configurable uncertainty halo. A placement is valid only when:

- all required sole samples have one supporting surface within variation and
  slope bounds;
- the sole does not bridge a discontinuity or void;
- the support polygon retains an edge-safety margin;
- the pose is within joint and action retargeting reach.

Rotating or translating the terrain and command together must rotate or
translate the solution without changing action identities or local costs,
within deterministic tie-breaking.

## Motion Action Library

Each action is an authenticated contact-to-contact source interval represented
in a canonical local frame. It records:

- source clip and frame interval;
- starting and ending support topology;
- landing order and gait phase;
- local root pose and velocity trajectory;
- local left/right sole pose and velocity trajectories;
- key-body or conservative body-clearance samples;
- command twist and planar displacement;
- joint and root boundary state;
- contact onset and release frames;
- source swing-clearance envelope;
- unsupported intervals, if explicitly authenticated;
- naturalness statistics such as duration, same-foot stride, width, and joint
  velocity.

Periodic flat and crawling gaits may additionally share a normalized phase and
command-twist index, following the command-conditioned library formulation in
Shooting for Contact. Nonperiodic terrain actions retain their explicit
contact phase.

The library is scene-independent. Terrain is queried only during graph
expansion.

## Graph State

A graph node stores:

- both realized sole poses in heading-local and scene coordinates;
- support topology and phase;
- terminal root pose, orientation, and velocity;
- terminal joint pose and velocity;
- source action identity;
- commanded progress and lateral deviation;
- accumulated cost and parent edge.

The state is not keyed by an object class or fixed terrain lane. Equivalent
states may be merged only when sole poses, support phase, boundary kinematics,
and command progress are all within declared quantization bounds.

## Action-Conditioned Expansion

For each retained node:

1. Query actions compatible with its support topology, phase, command twist,
   and boundary state.
2. Transform each source action from its canonical frame into the node's
   scene frame.
3. Construct bounded landing corrections in forward, lateral, vertical, yaw,
   and optional duration coordinates.
4. Evaluate corrected landing soles against the height field using full-sole
   samples.
5. Retarget the action under explicit stance-lock, target-error, joint-step,
   root-step, and source-distortion bounds.
6. Query terrain beneath every swing-sole and conservative body sample for the
   complete transformed action.
7. Reject the edge immediately on penetration, insufficient clearance,
   unsupported declared contact, stance drift, unreachable IK, or malformed
   support timing.
8. Compute its terminal node and edge cost only after all hard feasibility
   checks pass.

The landing search is coarse-to-fine and batched on the GPU. It is centered on
the source action's natural landing, not on a rigid flat-ground footprint
grid. Small preparatory steps emerge by selecting short actions; they are not
inserted by staircase-specific rules.

## Cheap and Exact Feasibility Layers

Expansion uses two deterministic layers:

### Batched conservative filter

- full-sole landing support;
- sampled swing-sole height envelope;
- conservative key-body clearance;
- source reach and retarget bounds;
- support topology and contact timing;
- boundary-state compatibility.

This filter may reject false positives but must not accept known collisions.

### Exact bounded realization

Surviving candidates run the existing MuJoCo kinematics and bounded
retargeter. Exact realization measures stance error, sole clearance, joint and
root continuity, and complete contact consistency. Failed exact edges are
cached by node/action/correction signature so graph search cannot repeatedly
select the same infeasible transition.

## Search and Cost

Use deterministic weighted A* or a bounded best-first beam. Because this is an
offline planner, correctness and motion quality take priority over real-time
latency.

Hard failures remove edges. Soft cost includes:

- commanded progress and lateral path deviation;
- heading error;
- boundary joint/root position and velocity error;
- landing correction magnitude;
- source same-foot stride and width distortion;
- duration and gait-phase distortion;
- minimum terrain and edge margin;
- stance slip;
- excessive double support or hover;
- joint acceleration and jerk;
- repeated actions or low-progress cycles.

The heuristic is remaining commanded distance divided by the best
terrain-compatible progress bound. Search terminates only at a complete
support phase after meeting the progress tolerance.

## Naturalness Contract

Naturalness is protected structurally:

- source action motion is retained as the prior;
- retargeting is bounded and measured;
- stride, width, phase, duration, and body-path distortions are penalized;
- action boundaries require pose and velocity compatibility;
- no independent per-frame foot projection is allowed;
- no contact label may be changed to make a trajectory pass.

The result report includes maximum and distributional source distortion, not
only collision/contact metrics.

## Optional Dynamic Refinement

An accepted kinematic route may seed a contact-implicit multiple-shooting
optimizer. The refinement tracks state and key-body trajectories while
resolving simulation contact, friction, joint limits, self-collision, torque,
and command-rate costs.

The graph route remains the initial contact and motion prior. Dynamic
refinement failure does not retroactively make an invalid graph edge valid.
Refined motions are stored as new library evidence only after independent
validation and provenance recording.

## Independent Validation

The final validator is separate from graph scoring and verifies:

- complete-sole support for every declared contact;
- measured contact height and velocity consistency;
- minimum swing-sole and body clearance;
- stance drift and foot slide;
- unsupported intervals only where explicitly authenticated;
- terminal complete support;
- joint/root position and velocity continuity;
- command heading and progress;
- source provenance for every frame;
- stride, phase, duration, retarget, acceleration, and jerk bounds.

The validator reports worst frames and contacts. A pass bit without these
metrics is insufficient.

## Generality Tests

Synthetic and corpus-backed tests must use the same planner configuration on:

- flat terrain;
- finite curbs and blocks;
- ascending and descending stairs;
- ramps and uneven plateaus;
- narrow supported regions and gaps;
- translated and rotated copies of every terrain;
- perturbed heights and tread widths within source reach;
- deliberately unreachable variants.

Constant headings are swept across at least every 15 degrees. Tests must also
randomize path offsets so a successful solution cannot depend on one lane.

Metamorphic requirements:

- joint rotation of terrain, start state, and command rotates the route and
  preserves local costs;
- translation preserves action identities and costs;
- adding clearance cannot invalidate an otherwise identical edge;
- raising terrain beyond reach changes success into a classified
  reachability or coverage failure;
- no test or runtime config names an angle or object type.

## Acceptance Criteria

The replacement is accepted when:

1. the horizontal baseline remains independently valid;
2. feasible rotated copies succeed with the same runtime thresholds and no
   action blacklist;
3. diagonal routes reject swept collisions during edge expansion rather than
   after full-plan realization;
4. successful routes stay within declared source-motion distortion bounds;
5. all returned contacts have complete-sole support;
6. unreachable terrain returns a classified failure;
7. no heading, lane, staircase, curb, or object-specific branch exists in the
   planner;
8. every result includes full action and frame provenance.

## Scope

This specification covers constant-heading, offline kinematic traversal over
height-field terrain. It deliberately leaves time-varying steering commands,
depth-to-height estimation, Sonic tracking, online policy distillation, and
dynamic multiple-shooting integration for subsequent specifications.
