# G1 Contact-Path Terrain Motion Generation

## Purpose

Generate natural, terrain-valid G1 kinematics for a constant planar travel
heading over an arbitrary height field. The first acceptance cases are a
horizontal staircase crossing and a 45-degree crossing, but the algorithm must
not contain staircase labels, fixed approach headings, or hand-authored contact
coordinates.

This stage produces kinematics only. Sonic tracking, physics control, and
real-time latency are out of scope.

## Correction to the Previous Design

The previous action-conditioned graph made complete GRAIL actions the primary
search edges. That preserved source motion quality, but it made elementary
footstep placement depend on whether a matching terrain clip already existed.
At diagonal boundaries, endpoint-compatible actions could still have invalid
swing paths or implausible strides.

The corrected decomposition keeps three decisions separate:

1. terrain geometry decides where support is possible;
2. a normal gait template decides contact order and approximate timing;
3. a constrained motion generator decides the whole-body pose trajectory.

The contact planner never searches motion clips. The motion generator never
chooses footholds.

## Alternatives

### Continue tuning motion matching

Search more GRAIL actions, add gait-phase costs, and reject complete swing
collisions. This can improve library retrieval, but it cannot generate missing
uneven gaits and has already produced long strides, freezes, and endpoint-only
matches. It is retained only as a source of natural gait statistics and
optional key poses.

### Optimize all joint trajectories directly

Use contact-implicit trajectory optimization from the start. This is the most
general formulation, but it is expensive and sensitive to initialization. It
is useful later for repairing or augmenting accepted motions, not as the first
contact planner.

### Project a gait contact path, then synthesize it

Lay a normal alternating gait along the commanded line, project each foot onto
full-sole terrain support, insert contacts around skipped terrain transitions,
derive a smooth root path, and synthesize the body from sparse constraints.
This is the selected approach because it is angle-general, debuggable, and
does not require a terrain-specific source clip.

## Coordinate Model

All planning occurs in a command-local frame:

- `s`: progress along the requested heading;
- `l`: lateral displacement, positive toward the left foot;
- `h`: terrain height.

The scene heading and its perpendicular form an orthonormal basis. The two foot
lanes are centered at `l = +/- nominal_half_width`. Rotating or translating
the terrain, start state, and command together must rotate or translate the
same contact plan.

Terrain is queried as a height field. A sole pose is valid only if all sole
samples:

- lie inside known terrain;
- are supported within the sole-height variation tolerance;
- satisfy the edge-safety margin;
- do not bridge a discontinuity or void.

## Contact-Path Planner

### Nominal gait

Extract nominal cadence, same-foot stride, step width, stance duration, and
double-support duration from a clean flat GRAIL walk. If extraction is
unavailable, use explicit robot-scale defaults stored in configuration.

Lay alternating left/right contacts along `s`. Contact yaw remains equal to
the commanded heading. This creates a normal flat gait before terrain is
considered.

### Per-lane terrain profiles

Sample dense full-sole feasibility profiles along the left and right walking
lanes. Each sample records:

- progress;
- support height;
- full-sole validity and margin;
- local height variation;
- the nearest terrain discontinuity before and after the sample.

The two profiles are independent. A diagonal crossing naturally causes the
left and right feet to encounter a terrain boundary at different progress.

### Projection and transition insertion

For every nominal contact, search a bounded progress and lateral window for the
closest valid full-sole placement. The objective preserves nominal stride,
width, cadence, and edge margin. A contact may move, but the planner must never
silently request a same-foot stride outside configured reach.

Detect support-height transitions crossed between consecutive planned
contacts. If a nominal gait step would skip a transition that needs an
additional support exchange, insert a contact immediately before or after that
transition. Prefer alternating feet; allow a short double-support preparation
when strict alternation is geometrically infeasible.

This rule is geometric, not object-specific. It applies equally to stairs,
curbs, blocks, ramps with discontinuities, and rotated terrain.

### Contact intervals

Each touchdown expands into a stance interval using the gait cadence. During a
stance interval the foot position and yaw remain fixed. Swing intervals receive
clearance waypoints derived from the maximum sampled terrain height between
liftoff and touchdown plus a safety margin.

The output is an immutable contact schedule containing both feet's pose,
support mask, touchdown/liftoff frames, nominal-vs-realized offsets, and
terrain feasibility evidence.

## Root Path

The planar root follows a smooth monotone curve through the midpoint of the
support feet while remaining close to the commanded line. Root height is the
support-weighted terrain height plus the nominal pelvis clearance from the
flat gait. Cubic interpolation with bounded velocity and acceleration prevents
the pelvis from copying sharp height-field discontinuities.

The root heading remains constant for the current milestone. Later turning
uses the same planner along a curved command path with locally varying tangent.

## Synthesis Backend Priority

The released ARDY G1 checkpoints were trained on Bones Rigplay 1, not GRAIL.
The GRAIL robot trajectories come from a separate synthetic-video,
reconstruction, retargeting, and SONIC realization pipeline. ARDY therefore
cannot be assumed to reproduce the GRAIL terrain-motion distribution.

The primary synthesis backend is a globally regularized retargeting of an
authenticated GRAIL gait. ARDY receives the same schedule as an experimental
second backend. Both outputs pass the same validator and visual-quality review;
the planner does not prefer ARDY merely because it generated a complete
sequence.

## GRAIL-Native Trajectory Retargeting

Select a clean GRAIL gait with cadence and commanded velocity close to the
requested motion. Preserve its contact phase and use its root-relative joint
trajectory as the motion prior.

Solve the complete traversal, or overlapping contact-to-contact windows, as one
temporally regularized kinematic optimization. Variables are root position and
G1 joint positions for every frame. The objective preserves:

- source joint pose and joint velocity;
- source root-relative foot swing shape;
- source pelvis and torso orientation;
- contact phase and stance duration;
- smooth joint acceleration and root acceleration.

Hard constraints enforce:

- planned stance sole position and yaw for every stance frame;
- heel and toe support rather than ankle-point support;
- swing clearance waypoints;
- the planned root corridor and bounded pelvis height;
- G1 joint limits and exact start/terminal state when required.

This differs from the previous framewise overlay: the complete motion is solved
jointly, stance feet include orientation and sole extent, and continuity is an
optimization constraint rather than a post-hoc blend.

## Optional ARDY Constraint Synthesis

The pinned official G1 ARDY model runs at 25 Hz. The adapter converts MuJoCo
Z-up/XY-ground coordinates to ARDY Y-up/XZ-ground coordinates.

ARDY's supplied G1 foot constraint class also constrains pelvis state and needs
a complete sparse pose. The adapter therefore constructs coherent sparse key
poses rather than isolated XYZ targets:

- a dense or regularly sampled `Root2DConstraintSet` provides the planar root
  path and constant heading;
- contact-boundary poses come from the clean flat gait, transformed to the
  planned root and foot contacts;
- left/right foot constraint sets are applied at stance samples using those
  coherent poses;
- sparse full-body constraints are used only at the exact start and terminal
  states.

The ARDY comparison batch varies diffusion seed and conditioning density.
Candidates are ranked only after independent terrain validation. ARDY is an
optional motion synthesizer, not the terrain feasibility oracle or the primary
backend.

Output is converted back to MuJoCo coordinates and resampled to 50 Hz.

## Bounded Repair

If an ARDY candidate is natural and close to the schedule but misses a small
number of stance or clearance constraints, bounded MuJoCo IK repairs only the
affected interval. It must preserve:

- planted-foot position and yaw;
- joint limits and maximum joint step;
- root-path tolerance;
- neighboring pose and velocity continuity.

Repair is rejected when it exceeds declared bounds. It may not turn an
unsupported or malformed route into an apparent success by changing the
contact schedule.

## Validation and Ranking

Every candidate is replayed kinematically against the same terrain. Hard
acceptance requires:

- every declared stance has valid full-sole support;
- planted feet do not slide beyond tolerance;
- swing soles and conservative body samples do not penetrate terrain;
- foot yaw and ankle orientation remain within bounds;
- joint limits, joint steps, root steps, and endpoint continuity pass;
- progress does not freeze while a nonzero command is active;
- all requested terrain transitions are completed.

Accepted candidates are ranked by:

- deviation from flat-gait stride, width, phase, and joint motion;
- root jerk and joint acceleration;
- contact margin and swing clearance;
- constraint and repair magnitude;
- endpoint error.

The planner emits explicit failure classes for no foothold, reach violation,
generator constraint miss, collision, continuity failure, and motion-quality
failure.

## Acceptance Experiments

The first benchmark uses the same staircase and start state for:

1. horizontal crossing: step up, walk across with unequal foot heights, step
   down while preserving heading;
2. 45-degree crossing;
3. head-on ascent/descent as a control case.

Each experiment first produces a GRAIL-native globally retargeted candidate,
then optionally generates ARDY candidates from the identical contact and root
constraints. Contact validity is mandatory. A contact-valid motion is not
declared good unless rendered motion also avoids visible floating, crouch
shuffling, inward-splayed ankles, large lunges, and sliding dismounts.

After those pass, sweep headings every 15 degrees and translate/rotate the
terrain to test equivariance and prevent overfitting to one approach.

## Scope Boundaries

- No Sonic integration or tracking.
- No real-time requirement.
- No terrain object classification.
- No assumption that every object is traversable.
- No hard-coded staircase coordinates or headings.
- No modification of the official ARDY repository.
