# G1 Ordered-Contact Diagonal Route Design

## Goal

Search the complete GRAIL corpus for a kinematic 45-degree staircase route
that contacts every terrain level crossed by the path and finishes with a
supported exit on the opposite flat ground.

The decisive route is:

```text
ground -> tread 1 -> tread 2 -> opposite-side ground
```

This stage evaluates raw motion retrieval and contact-compatible splicing. It
does not use Sonic, physics tracking, inverse kinematics, ARDY, MotionBricks,
foot projection, or edited joint trajectories.

## Experiment Path

The first route travels at -45 degrees in scene XY from:

```text
start = (-0.4835621836, 1.4414988213)
stop  = ( 0.7000000000, 0.2579366377)
length = 1.6738096920 m
```

It starts on flat ground, crosses the first and second physical staircase
treads, exits through the opposite side of the staircase, and ends on flat
ground. The experiment path is fixed, but the algorithm must derive its
required terrain levels from the supplied height grid rather than hard-code
the staircase coordinates or heights.

## Ordered Terrain-Level Contract

The path planner samples nominal complete-sole footprints along the directed
path. Consecutive surface heights within the existing contact tolerance are
clustered into ordered terrain levels. Repeated samples on the same level are
merged, while a return to an earlier level remains a separate required phase.

For the decisive path, the derived sequence is approximately:

```text
0.000 m -> 0.187 m -> 0.349 m -> 0.000 m
```

The final ground level additionally requires two alternating ground
touchdowns so the route cannot end on the first impact of a drop.

The required sequence records:

- ordered terrain-level identifiers and representative heights;
- the path interval over which each level is geometrically available;
- the allowed sole-center region for each touchdown;
- whether the level is entry ground, an intermediate tread, or exit ground.

## Actual-Contact Admission

Nominal footprints only define the route requirement. A raw GRAIL window is
admitted using its actual foot contacts after rigid placement.

Each actual touchdown is assigned to a required terrain level only when:

- its complete sole lies within that level's path interval and support region;
- its sampled surface height matches the level within the existing stance
  tolerance;
- its supported sole and body geometry pass the existing penetration checks;
- its path progress is nondecreasing.

A candidate may remain on the current level or advance to the immediately
next required level. It may not advance over an unvisited level. Therefore a
raw `0.000 -> 0.349 m` step cannot satisfy the required
`0.000 -> 0.187 -> 0.349 m` route.

No joint positions are edited. Placement remains limited to global yaw, XY
translation, and one root-Z translation per raw window.

## Level-Aware Path Graph

The path-coverage graph augments each state with the index of the next
required terrain level. A graph edge records:

- its covered path interval;
- the ordered required levels satisfied by its actual touchdowns;
- starting and ending support-foot identity;
- boundary joint pose and velocity;
- root pose and velocity;
- support masks and contact-certification metrics.

An edge is reachable only if it starts from the predecessor's terrain-level
state and does not skip a required level. Existing contact-compatible
transition certification remains unchanged. The search still minimizes
coverage gaps, invalid contacts, boundary mismatch, placement count, and raw
motion cost, but a route is complete only after all required levels have been
satisfied.

## Mandatory Exit

The exit is part of the same route rather than a postprocessing option. A
complete result must:

- contain an elevated-to-ground touchdown after the final tread;
- contain a subsequent alternating ground touchdown;
- finish with both feet supported on the opposite flat surface;
- place its terminal root beyond the staircase side boundary;
- pass the existing sole-clearance, stance-error, planted-foot, joint-step,
  and root-step gates.

A chain that stops elevated or ends on a single unsupported landing is
incomplete.

## Outputs and Failure Behavior

The run writes:

- `required-levels.json`, containing the terrain-derived ordered contract;
- `diagnostic.json`, containing candidate admissions and stable rejection
  reasons including `skipped-required-level` and `missing-ground-exit`;
- `report.json`, containing the selected raw sources, satisfied level
  sequence, transition metrics, and final support state;
- `traversal.npz` only for a complete certified chain;
- a contact sheet and 50 Hz MuJoCo viewer command only after numerical and
  visual verification.

If no raw chain exists, the run reports the exact missing phase and path
interval. It must not shorten the requested route, omit the exit, synthesize
contacts, or claim partial coverage as success.

## Acceptance Criteria

The 45-degree route is accepted only when:

- actual touchdown heights satisfy the complete ordered sequence
  `ground -> tread 1 -> tread 2 -> ground`;
- no selected edge skips a required terrain level;
- the final two alternating touchdowns are on opposite-side ground;
- both feet are supported on ground at the terminal frame;
- the complete 1.6738096920 m path is covered;
- raw source joint positions are preserved exactly outside the existing
  contact-compatible boundary blend;
- maximum supported-sole error is at most 0.03 m;
- minimum complete-sole clearance is at least -0.03 m;
- the existing transition, joint-step, root-step, and planted-foot gates pass;
- the live MuJoCo playback visibly takes both risers and executes the exit.

If these criteria cannot be met by the current corpus, the correct outcome is
an explicit infeasibility report identifying the first unavailable required
level or exit transition.
