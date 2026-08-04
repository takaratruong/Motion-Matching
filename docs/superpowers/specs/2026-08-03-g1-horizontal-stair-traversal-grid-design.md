# G1 Horizontal Stair Traversal Grid Design

## Goal

Build the first directional family of an offline traversal grid drawn over a
staircase. Each horizontal grid line crosses the staircase from one lateral
side to the other, perpendicular to the staircase ascent direction.

This is a kinematic motion-quality experiment. Sonic tracking, physics
control, depth inference, and realtime latency are out of scope.

## Coordinate frame and grid

The authenticated staircase frame defines:

- `u`: staircase ascent direction;
- `v`: horizontal direction across the staircase width;
- `z`: world height.

Generate one horizontal line every `0.10 m` in `u` over the complete stair
footprint. Each line extends beyond both lateral staircase boundaries so its
route begins and ends on surrounding flat ground. Evaluate both `+v` and `-v`
travel directions.

The line offset determines the tread height that the route must mount. Some
lines are expected to be unreachable because their curb height exceeds the
available motion or kinematic envelope. Unreachable lines are valid reported
outcomes, not reasons to weaken contact constraints.

## Required route structure

Every successful route contains five ordered phases:

1. flat walking toward the lateral side of the staircase;
2. a curb-up action that mounts the selected tread from its side;
3. horizontal walking across the tread;
4. a curb-down action that descends from the opposite side;
5. flat walking away from the staircase.

A route is incomplete if it starts on the stair, stops at the far edge, omits
the mount or descent, or reaches the opposite side only through root sliding.

## Motion inventory

Use the full accepted GRAIL curb and staircase corpora for mount and descent
actions and the accepted flat locomotion corpus for approach, crossing, and
departure. Side mounting is expected to favor curb actions, but a staircase
action is admissible when its contact geometry and travel direction match the
horizontal route more closely.

Each source clip is segmented into landing-to-landing contact actions. Record:

- source family and clip identity;
- root displacement and heading in the source terrain frame;
- left and right contact schedule;
- initial and terminal foot-height pattern;
- maximum curb height crossed;
- boundary joint pose and velocity;
- full oriented sole trajectory.

Mount and descent phases must use curb- or staircase-family terrain actions.
The solver may not silently replace them with flat walking or the geometric
pivot generator. Geometric correction may align an otherwise suitable terrain
action to a target edge, but it may not invent the entire mount or descent.

## Placement and retargeting

Rigidly align candidate actions to the global staircase frame and target
travel direction. Retarget only the root trajectory and leg joints needed to
match the local foot contacts. Preserve the source action timing, contact
order, upper-body motion, and non-leg joint prior.

Candidate actions are rejected before graph insertion when:

- the commanded progress has the wrong sign or is too small;
- the contact-height change disagrees with the selected tread height;
- a nominally planted sole crosses a riser or unsupported edge;
- the action requires a joint-limit violation;
- the retargeted action exceeds the output speed or acceleration contract.

## Offline route solver

Create contact-boundary nodes on both flat regions, at the two lateral stair
edges, and at valid intermediate tread landings. Add a directed edge for each
qualified flat or curb contact action.

For each horizontal line and direction, use dynamic programming over the five
required phases. The route cost ranks:

1. completion of the required phase sequence;
2. exact full-sole contact validity;
3. terminal progress error;
4. source-motion deviation;
5. seam velocity and acceleration;
6. foot slide and clearance margin.

The solver returns either one complete route or an explicit failure with the
phase and rejection histogram that made the line unreachable. It must never
freeze or return a partial route as success.

## Validation

Every emitted frame is checked with the authenticated staircase height query
and the MuJoCo G1 seven-point oriented sole model.

A successful route must satisfy:

- complete flat-to-flat lateral traversal;
- terrain-action source provenance for the mount and descent, including the
  selected curb or staircase family;
- correct commanded progress;
- no full-sole penetration below `-0.025 m`;
- terminal planted-sole error at most `0.025 m`;
- stance width and joint limits;
- maximum joint speed and acceleration limits;
- exact authenticated boundary identity at every splice;
- no unreported missing phase.

The existing geometric pivot output is not a motion-quality reference for
this experiment.

## Outputs

Produce:

- one route artifact and metrics file for every successful line and direction;
- one structured failure record for every unreachable line and direction;
- an aggregate horizontal coverage matrix indexed by 10 cm `u` offset and
  travel direction;
- a MuJoCo contact sheet for every route;
- one interactive MuJoCo viewer that cycles only through complete qualified
  routes.

The aggregate visualizer draws successful grid lines in green and unreachable
lines in red over the staircase.

## First acceptance gate

The prototype is ready for feedback when:

- every 10 cm horizontal line has a success or explicit failure in both
  directions;
- at least one line completes the full
  approach–mount–cross–descend–depart route;
- the successful example visibly retains the selected curb- or stair-style
  body and leg motion;
- all successful routes pass the full-sole, progress, seam, speed, and
  acceleration contracts;
- the coverage report explains why higher unreachable lines failed.

After this gate, the same representation can be extended to the remaining six
45-degree traversal families and their intersections.
