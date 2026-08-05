# G1 Horizontal Grid Coverage Design

## Goal

Measure how well path-only raw-motion retrieval covers a fixed staircase when
the same horizontal traversal is requested at parallel centerlines spaced
every 0.20 m. Show the complete grid and all accepted traversals in one
kinematic MuJoCo playback instead of presenting one successful lane.

## Fixed Experiment

The target scene is
`grail-stair_p1-db7949fce1b2e48d39f4`. The grid contains 11 directed paths:

```text
y = -1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
     0.2,  0.4,  0.6,  0.8,  1.0 m
```

Every path:

- starts at scene X `-1.2795985755 m`;
- stops at scene X `1.2590216406 m`;
- travels in scene `+X`;
- is 2.5386202161 m long;
- uses a nominal 0.20 m step width;
- searches the complete admitted height-grid GRAIL corpus;
- receives no source clip, family, frame, or transform override.

The lanes at `y = -1.0 m` and `y = 1.0 m` intentionally test the two
staircase boundary-straddle cases. Interior lanes test different tread
heights and split-height patterns.

## Approaches Considered

### Shared source index with independent lane search

Extract label-independent raw contact windows once, score them independently
against every lane's terrain/contact signature, and physically certify each
lane's shortlist. This is the selected approach because it performs a genuine
search per lane without repeating the corpus extraction 11 times.

### Eleven independent single-path processes

Invoke the qualified single-path program once per lane. This reuses more code
but repeatedly scans 12,645 clips, wastes runtime, and makes a single
cross-lane coverage report harder to verify.

### Translate the known winning motion between lanes

Rigidly move the approved horizontal clip to each lane and validate it. This
is a useful diagnostic but not an acceptable grid-search result because it
does not test whether path/contact retrieval chooses appropriate motions for
different height patterns.

## Architecture

### Grid definition

A pure grid constructor produces ordered immutable lane requests. It validates
finite endpoints, positive spacing, exact inclusion of both requested lateral
bounds, and deterministic low-to-high lane ordering.

### Shared raw-window index

The system loads every admitted height-grid motion once and extracts the same
variable contact windows used by the qualified single-path search. Each index
record retains:

- source clip and frame interval;
- raw contact order, positions, times, and relative terrain heights;
- root start, progress, and heading error;
- enough provenance to reload raw joint, root, foot, support, and sole data
  for physical certification.

Source family names remain provenance only. They never gate or prioritize a
lane.

### Per-lane query and shortlist

For each centerline, nominal footprint hypotheses span both leading feet,
observed step lengths, and observed speeds. Terrain heights are sampled at the
lane's approximate footprint centers.

Every indexed window is scored against the lane independently. The shortlist
preserves:

- best contact-signature cost;
- best complete-path coverage;
- the blended contact/coverage score;
- contact-count diversity.

Selection for one lane has no effect on another lane.

### Physical certification

Shortlisted windows are rigidly placed on their lane using only:

- one global yaw;
- one global XY translation;
- one root-Z translation.

Joint positions remain byte-for-byte equal to the raw source window.
Acceptance uses the existing qualified limits:

- heading error p95 no greater than 15 degrees;
- root lateral deviation no greater than 0.15 m;
- maximum stance-height error no greater than 0.03 m;
- minimum complete-sole clearance no lower than -0.03 m.

The grid runner retains the best certified placement even when it covers only
part of a lane, but it never labels that lane complete.

## Classification and Metrics

Each lane is classified as:

- `full`: certified coverage contains the complete requested interval;
- `partial`: at least one certified placement exists but leaves an explicit
  uncovered suffix or prefix;
- `infeasible`: no raw placement passes physical certification.

The aggregate report includes:

- count and fraction of full, partial, and infeasible lanes;
- total covered arc length divided by total requested arc length;
- selected source and frames for every accepted lane;
- rigid transform and physical metrics for every accepted lane;
- uncovered intervals for partial lanes;
- deterministic rejection counts for every lane;
- exact corpus and grid configuration.

No smoothing, IK, source-identity override, or hidden gap is permitted.

## Outputs

One run writes:

```text
grid-summary.json
grid-playlist.npz
grid-playlist.json
lanes/
  lane-neg-1p0/
    placements.json
    coverage.json
    traversal.npz
  ...
  lane-pos-1p0/
    placements.json
    coverage.json
    traversal.npz
dataset/
```

`grid-playlist.npz` concatenates every full and partial certified traversal in
low-to-high lane order. It inserts a short duplicate-pose hold before and
after each lane. Teleports between lanes are playlist boundaries, not claimed
motion transitions, and are recorded in `grid-playlist.json`.

## Visualizer

The MuJoCo viewer loads the playlist and grid summary. It:

- draws all 11 requested horizontal paths on the staircase;
- colors full lanes green, partial lanes amber, and infeasible lanes red;
- highlights the currently playing accepted lane in white;
- plays accepted lanes sequentially at 50 Hz;
- pauses briefly at each lane's start and end;
- loops the complete playlist;
- retains Space, arrow, Backspace, and X controls.

Failed lanes remain visible in red even though no motion is played for them.
The viewer must not visually interpolate the teleport between playlist lanes.

## Determinism and Acceptance

The experiment is accepted only when:

- exactly 11 lanes at 0.20 m spacing are evaluated;
- all lanes use the same corpus, search algorithm, thresholds, and path
  length;
- no lane receives a selected source identity as input;
- every emitted placement preserves raw joint positions exactly;
- every accepted placement passes physical certification;
- all partial gaps and infeasible lanes remain explicit;
- two fresh runs produce identical selected sources, frames, transforms,
  classifications, coverage, and playlist arrays;
- the live MuJoCo viewer shows the colored grid and cycles through every
  accepted lane.

The experiment may legitimately report poor coverage. Its purpose is to
measure the method honestly, not force all grid lines to succeed.
