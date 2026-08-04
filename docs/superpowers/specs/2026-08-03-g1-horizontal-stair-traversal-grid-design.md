# G1 Horizontal Stair Traversal Grid Design

## Goal

Build the first directional family of an offline traversal library drawn over
a staircase. Each horizontal grid line crosses the staircase from one lateral
side to the other, perpendicular to the staircase ascent direction.

This is a kinematic motion-quality experiment. Sonic tracking, physics
control, depth inference, realtime latency, online motion matching, and
interactive command selection are out of scope. Motion retrieval and graph
search are generation tools only; the product of this stage is a set of
complete validated motion artifacts.

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

## Warm-start inventory

Use the full accepted GRAIL curb and staircase corpora for mount and descent
actions and the accepted flat locomotion corpus for approach, crossing, and
departure. Side mounting is expected to favor curb actions, but a staircase
action is admissible when its contact geometry and travel direction match the
horizontal route more closely.

Each source clip is segmented into landing-to-landing contact actions used as
warm starts and discrete contact-schedule hypotheses. Record:

- source family and clip identity;
- root displacement and heading in the source terrain frame;
- left and right contact schedule;
- initial and terminal foot-height pattern;
- maximum curb height crossed;
- boundary joint pose and velocity;
- full oriented sole trajectory.

Mount and descent hypotheses must be initialized from curb- or
staircase-family terrain actions. Flat walking may initialize the approach,
crossing, and departure phases. The optimizer may change the continuous
trajectory substantially, but every retained route records which source
actions and contact schedule initialized it.

## Hybrid trajectory synthesis

Rigidly align a sequence of flat and terrain actions to the global staircase
frame and target travel direction. Concatenate them into one complete
flat-to-flat anchor sequence.

Use the pinned upstream ARDY G1 model only for short in-between windows between
those anchors. Supply the adjacent GRAIL full-body states, root trajectory,
feet, and contact-boundary states as kinematic constraints. ARDY does not
choose footholds, contact order, mount height, or route semantics.

The first integration pins ARDY upstream commit
`693f74d13b3d04a0a22ce127ee79c929dd89756b` and uses the released G1 25 FPS
checkpoint. Resample only the generated transition windows to the library's
50 Hz output. Original GRAIL anchor frames remain unchanged.

After in-betweening, apply bounded leg IK and temporal smoothing only where
needed to restore the prescribed planted sole poses. Reject a result rather
than moving a foothold or changing the source contact schedule.

Hard validation enforces:

- the five required route phases and commanded lateral progress;
- the GRAIL-derived contact order and footholds;
- full oriented sole support on every planted frame;
- swing-foot and body collision clearance;
- exact flat-ground start and end boundaries;
- joint position, speed, and acceleration limits.

Warm-start sequences are rejected before continuous optimization when:

- the commanded progress has the wrong sign or is too small;
- the contact-height change disagrees with the selected tread height;
- a nominally planted sole crosses a riser or unsupported edge;
- the action requires a joint-limit violation;
- the retargeted action exceeds the output speed or acceleration contract.

## Offline route generator

Create contact-boundary nodes on both flat regions, at the two lateral stair
edges, and at valid intermediate tread landings. Add a directed edge for each
qualified flat or terrain contact action. This graph is an internal search
representation, not the runtime product.

For each horizontal line and direction, use dynamic programming to enumerate
promising anchor sequences over the five required phases. Preserve each
GRAIL action and invoke ARDY only on the splice windows. Admit only
independently validated solutions.

The generator retains every non-dominated complete route that differs in
source-action sequence or contact schedule. Dominated duplicates are removed
by motion identity and validation metrics. If no complete route exists, it
returns an explicit failure with the phase and rejection histogram that made
the line unreachable. It must never freeze or retain a partial route as a
valid traversal.

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

- a versioned traversal-library manifest;
- one or more route artifacts and metrics files for every successful line and
  direction;
- one structured failure record for every unreachable line and direction;
- an aggregate horizontal coverage matrix indexed by 10 cm `u` offset and
  travel direction;
- a MuJoCo contact sheet for every route;
- one interactive MuJoCo viewer that cycles only through complete qualified
  routes.

Each library entry records its line, direction, complete source-action
sequence, contact schedule, validation metrics, and deterministic identity.
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
45-degree traversal families. Intersections, direction changes, motion
matching, and command-driven selection are separate later stages built from
the qualified traversal library.
