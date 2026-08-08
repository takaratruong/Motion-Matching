# G1 Staircase-Intersecting Parallel Grid Design

## Goal

Evaluate constant-heading terrain traversal on a 20 cm parallel grid whose
routes actually cross the staircase. Every admitted playback must start on
ground, mount elevated staircase geometry, travel across it, and dismount to
opposite ground. Flat-only routes are not staircase results and must never be
used to fill missing grid coverage.

The same geometry and search pipeline must support horizontal, 45-degree, and
later arbitrary constant headings. Existing accepted horizontal artifacts are
preserved as regression inputs; this experiment does not overwrite them.

## Problem in the Current Preview

The current 45-degree preview is selection-biased. The requested offsets span
`-1.0 m` through `+1.0 m`, but only one displayed route crosses the staircase.
The other four displayed successes at `+0.4 m` through `+1.0 m` remain on flat
ground, while failed staircase-intersecting lanes are omitted.

The current internal certification also admits visually unnatural results
because it does not make all of the following hard admission criteria:

- complete-sole support and contact consistency;
- stance-foot horizontal locking;
- gait cadence and excessive double-support duration;
- terrain-action semantics of source windows;
- natural, contact-compatible segment boundaries.

Consequently, the corrected grid must fix both experiment geometry and motion
quality admission.

## Staircase-Intersecting Grid Contract

For a requested heading `h`, define the perpendicular lateral axis
`l = (-h_y, h_x)`. Generate parallel centerlines at 0.20 m lateral spacing.
The lateral bounds are derived from the supplied terrain footprint, not from a
fixed symmetric offset range or angle-specific constants.

A route is in the evaluation grid only if its terrain profile contains:

1. an initial ground interval long enough to establish a supported approach;
2. at least one elevated staircase interval;
3. measurable progress across that elevated geometry; and
4. an opposite-side ground interval long enough to complete the exit.

Routes that never encounter elevated geometry are recorded as
`flat_only_excluded` and are never placed in the viewer. Routes whose crossed
height or required step exceeds corpus capability remain in the grid report as
classified failures; they are not replaced with easier paths.

The grid manifest records every sampled route, including its scene-space
endpoints, lateral offset, ordered terrain profile, crossed tread identifiers,
and inclusion or exclusion reason.

## Terrain-Event Decomposition

Each included route is decomposed from its sampled terrain profile into three
required semantic phases:

```text
mount -> elevated traversal -> dismount
```

- `mount` begins in a stable ground support phase and ends with supported
  progress on the first elevated region.
- `elevated traversal` covers every subsequent terrain region intersected by
  the route without skipping a required level or fabricating an unsupported
  contact.
- `dismount` begins at the last elevated support and ends after alternating
  supported ground contacts on the opposite side.

The phases are requirements for retrieval and validation, not permission to
splice at arbitrary boundaries. A single source window may cover multiple
phases when it satisfies the entire terrain contract.

## Hierarchical Motion Retrieval

Search the corpus in this order:

1. Prefer a single minimally edited source window that realizes the largest
   contiguous portion of the route.
2. Otherwise retrieve contact-to-contact windows for mount, elevated walking,
   and dismount using heading-local root displacement, actual support foot,
   touchdown displacement and height, cadence, and boundary body state.
3. Repeat a verified same-height gait cycle when a wide continuous surface
   requires more travel than one source window provides.
4. Use short contact-constrained interpolation or MotionBricks only to connect
   compatible source boundaries. It must not invent the main terrain-changing
   motion or replace absent corpus coverage.

Flat terrain windows may serve only continuous same-height flat/elevated
walking. A curb or stair transition window may serve only a compatible height
change. Source labels are useful priors, but admission is determined by actual
contact and displacement evidence rather than filenames alone.

All placement uses a rigid scene transform first. Any retargeting is bounded
and preserves the active stance sole. If the target contacts require excessive
joint deformation, ankle rotation, lateral foot crossing, or root correction,
the candidate is rejected.

## Sequence Search and Naturalness Ranking

Use a layered beam search whose state includes terrain-event progress, support
foot, gait phase, terminal pose and velocity, cadence state, and source
provenance. Required terrain events and hard physical gates remove invalid
edges before soft ranking.

Among valid edges, rank candidates by:

- coverage and root-path agreement;
- complete-sole support margin;
- stance-foot translation and rotation drift;
- pose and velocity continuity at joins;
- deviation from the route's robust alternating-foot cadence;
- excessive double-support, hover, shuffle, or repeated pawing;
- lateral foot crossing and support-polygon margin;
- total retargeting and interpolation magnitude;
- source fragmentation and number of joins.

The search must prefer a longer coherent clip over a lower-feature-error chain
of many short fragments when both cover the terrain.

## Independent Admission Validator

A completed route enters the viewer only if an independent replay validator,
separate from search scoring, confirms:

- the route contains initial ground, elevated traversal, and opposite ground;
- actual touchdown order covers every required terrain event;
- every declared stance has complete-sole support;
- stance soles remain horizontally and vertically locked within thresholds;
- swing soles and the body do not penetrate terrain;
- alternating same-height contacts have consistent cadence;
- height-changing intervals are exempt from cadence normalization but remain
  contact-valid;
- joint/root continuity and foot orientation stay within trackable limits;
- the terminal state has stable ground support and no incomplete swing;
- source provenance exists for every output frame.

Numerical thresholds come from the existing independent validator and accepted
horizontal regression, then are recorded explicitly in the run manifest. A
search-internal success that fails independent validation is
`validation_failed`, not a playable success.

## Viewer and Reporting Behavior

The grid viewer contains only independently admitted staircase traversals. It
orders them by lateral offset and displays the offset, crossed terrain levels,
and quality summary for the current route.

The report contains all requested staircase-intersecting lanes, including
failures. Missing lanes are shown in a separate grid coverage summary with one
of these stable reasons:

- `no_terrain_footprint`;
- `height_out_of_corpus_range`;
- `no_mount_motion`;
- `no_elevated_traversal_motion`;
- `no_dismount_motion`;
- `no_contact_compatible_chain`;
- `validation_failed`.

The viewer must not silently omit failed lanes or substitute flat routes. The
existing live preview remains available until a replacement has at least one
independently admitted result.

## Generalization Requirements

Heading, spacing, terrain footprint, and route endpoints are runtime inputs.
No source frame ranges, lateral offsets, tread coordinates, or terrain heights
may be selected by heading-specific branches.

The same code must regenerate:

- the accepted horizontal traversal as a non-regression case;
- the corrected 45-degree 20 cm staircase grid;
- a rotated or translated equivalent scene with identical local decisions;
- additional constant headings without implementation changes.

Angle-specific run manifests may select an experiment and retain its output,
but may not alter search semantics.

## Testing and Acceptance

### Unit tests

- terrain-derived lateral bounds exclude flat-only centerlines;
- every included centerline has ground/elevated/opposite-ground intervals;
- excluded and infeasible lanes remain visible in the coverage report;
- no viewer playlist can contain a `flat_only_excluded` route;
- terrain-event decomposition preserves ordered level changes;
- semantic retrieval rejects terrain-changing clips on flat phases and flat
  clips for unsupported height changes;
- complete-sole, stance-lock, cadence, foot-crossing, and terminal-support
  violations fail admission;
- heading rotation and scene translation preserve heading-local results.

### Integration tests

- regenerate the horizontal accepted artifact without modifying it;
- run the 45-degree grid at 0.20 m spacing over the staircase-intersecting
  band;
- independently validate every playlist entry;
- verify every playlist entry actually traverses elevated staircase geometry;
- verify all requested grid lanes appear in the report even when infeasible;
- replay admitted results at 50 Hz in MuJoCo.

### Acceptance criteria

The next feedback build is ready only when:

1. no displayed route is flat-only;
2. grid offsets are derived from and cover the staircase footprint;
3. every displayed route completes mount, elevated traversal, and dismount;
4. every displayed route passes independent contact and naturalness gates;
5. failed staircase lanes remain explicit rather than being hidden;
6. horizontal behavior is preserved and no heading-specific search logic is
   introduced;
7. the 45-degree playlist contains only motions suitable for meaningful visual
   feedback, even if that means fewer playable lanes than sampled lanes.

