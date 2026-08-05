# G1 Path Motion Placement Design

## Goal

Given a directed path across the fixed GRAIL staircase, find and rigidly place
existing GRAIL motion windows that collectively walk forward along the path.
Prefer whole clips and long raw windows, allow compatible walking cycles to
repeat, and report uncovered transition gaps explicitly.

This stage proves raw motion retrieval and placement. It does not claim that
arbitrary gaps can already be synthesized naturally.

## Scope

The first implementation targets:

- the fixed staircase currently used for horizontal traversal evaluation;
- straight, constant-heading paths sampled from the staircase grid;
- kinematic G1 playback at 50 Hz;
- the complete GRAIL corpus, regardless of source family labels;
- rigid placement of raw joint motion using only global yaw, XY translation,
  and root-height translation;
- whole-clip, variable-window, segmented, and repeatable-cycle solutions;
- explicit path-coverage and transition-gap reports.

It excludes Sonic, physics tracking, realtime search, arbitrary curved paths,
joint-space IK, foot projection, spatial scaling, and an assumption that
MotionBricks or ARDY can fill every transition.

## Path Contract

A query path contains:

- a scene-frame centerline with monotonically increasing arc length;
- a forward heading aligned with increasing arc length;
- a start and stop arc-length coordinate;
- the target staircase height grid.

The path does not prescribe timing, speed, gait phase, leading foot, step
length, or exact footprints. Search candidates supply those properties.

For the first version, a path is a straight line and is represented by its
scene-frame start point, unit travel direction, and length.

## Approximate Footprint Query

Nominal walking footprints are retrieval hints rather than hard constraints.
For each path, the system creates hypotheses spanning:

- left-foot-first and right-foot-first;
- several phase offsets;
- observed GRAIL walking step lengths and step durations;
- observed GRAIL lateral stance widths.

Each hypothesis samples the complete sole against the staircase and records:

- touchdown order;
- approximate touchdown arc length and lateral offset;
- approximate touchdown time;
- terrain height beneath each sole;
- level, split-height, mount, and dismount changes.

These signatures prefilter raw motion windows. A candidate is never accepted
because it matches a nominal footprint. Final acceptance uses the candidate's
actual root trajectory, feet, soles, and body geometry.

## Motion-Window Index

Every admissible raw motion window is represented in a character-local frame
whose forward axis is the window's root travel direction. The record contains:

- source clip and half-open frame interval;
- duration and forward progress;
- root trajectory and heading error;
- touchdown order, timing, and local XY positions;
- touchdown terrain heights;
- per-frame support masks;
- starting and ending joint position and velocity;
- starting and ending root velocity;
- starting and ending support-foot identity;
- motion-quality statistics for penetration, sliding, joint acceleration, and
  crouching.

Windows are variable length. Source labels such as `stair` and `curb` are
provenance only and never constrain retrieval.

## Rigid Placement

For a candidate window and path interval, placement:

1. rotates the candidate's character-forward axis onto the path direction;
2. translates its root trajectory onto the path interval;
3. chooses a single root-height translation minimizing supported-contact error;
4. leaves every joint angle unchanged.

Placement is rejected when:

- p95 character-heading error exceeds 15 degrees;
- root lateral deviation from the path exceeds 0.15 m;
- maximum supported-foot terrain error exceeds 0.03 m;
- minimum complete-sole clearance is below -0.03 m;
- a continuously supported foot moves more than 0.01 m per frame;
- the candidate makes insufficient positive path progress;
- a complete sole straddles an invalid riser;
- knees, shins, pelvis, or other non-foot collision samples penetrate terrain;
- the raw window violates the retained motion-quality limits.

The result is either a certified placed window covering a half-open path
interval or a rejection with a stable reason code.

## Path-Coverage Graph

Each certified placement is a directed graph edge covering
`[start_arc_length, stop_arc_length)`. Graph state also includes the placed
window's boundary pose, velocity, gait phase, and support identity.

Edges may represent:

- one complete raw clip;
- a variable-length raw window;
- a compatible repetition of a walking cycle;
- another raw window beginning after a transition gap.

The path search minimizes, in order:

1. uncovered path length;
2. invalid terrain contacts;
3. required transition distance;
4. boundary pose, velocity, heading, phase, and support mismatch;
5. number of placements;
6. footprint-query mismatch;
7. raw motion-quality cost.

Whole clips and longer raw windows win naturally through the placement-count
and transition-distance costs. A repeated cycle is admitted only when its end
state is compatible with its transformed start state.

The solver does not invent transitions. It returns:

- the selected ordered placements;
- covered path intervals;
- uncovered intervals;
- measured boundary mismatches for every neighboring placement;
- candidate inbetween endpoints for later MotionBricks or ARDY evaluation.

## Outputs

For each grid path, the offline run writes:

- `placements.json`: source windows, rigid transforms, coverage, costs, and
  rejection reasons;
- `traversal.npz`: concatenated raw placements only when their boundaries are
  directly compatible;
- `coverage.json`: covered intervals and explicit transition gaps;
- `contact-sheet.png`: fixed-camera visual evidence;
- an interactive 50 Hz MuJoCo kinematic playback command.

The summary compares every requested grid line and marks it as:

- fully covered by compatible raw placements;
- partially covered with explicit gaps;
- infeasible under the current corpus and rigid-placement contract.

## Staged Evaluation

### Experiment 1: Automatic Rediscovery

Input only the staircase and the horizontal path used by the approved live
preview. The system must rediscover a valid forward-facing raw placement
without a hard-coded clip identifier, source family, frame range, or rigid
transform.

### Experiment 2: Repetition or Chaining

Use a longer path that cannot be covered by the first placement alone. The
system must either select a certified repeatable cycle or return multiple raw
placements with measured boundary compatibility.

### Experiment 3: Grid Coverage

Run the same solver over parallel grid lines on the fixed staircase. Produce
the coverage classification and visualization for every line without changing
the search algorithm per line.

## Acceptance Criteria

The stage succeeds only if:

- Experiment 1 automatically finds and places a visually forward-walking
  traversal with no hard-coded result identity;
- all emitted placements preserve raw joint positions exactly;
- each certified placement satisfies the rigid-placement thresholds above;
- Experiment 2 demonstrates raw repetition or chaining, or reports the exact
  uncovered interval without claiming completion;
- Experiment 3 produces deterministic coverage results for all configured
  grid lines;
- a fresh rerun produces identical selected source windows and transforms;
- the MuJoCo visualizer shows the selected raw placements on the staircase.

The stage does not fail merely because some grid lines are infeasible. It
fails if it hides gaps, requires unreported joint edits, uses source-family
labels as semantic gates, or accepts placements that violate the terrain
contract.

## Follow-on Decision

Only after the raw coverage graph is measured will MotionBricks or ARDY be
evaluated against the actual reported gaps. Long gaps are permitted as
evidence, but they are not presumed solvable. Generated transitions must be
evaluated separately and must preserve the selected terrain-contact endpoints.
