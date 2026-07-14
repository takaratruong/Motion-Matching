# G1 Wide Playable Scenes Design

**Date:** 2026-07-14

**Status:** User-approved design; implementation planning pending written-spec review

## Goal

Give every G1 terrain scene enough lateral room for free manual exploration.
Each scene must provide at least a six-metre-wide playable floor while keeping
the existing stairs, ramps, curbs, blocks, captured GRAIL geometry, routes,
spawn transforms, and motion database unchanged.

The wider floor is scene data, not precomputed motion. The existing immutable
G1 motion pack remains shared by every scene.

## Scope

This change covers all fourteen required scenes:

- the four GRAIL curb scenes;
- three stair scenes;
- three longitudinal ramp scenes;
- two cross-slope scenes;
- the mixed multilevel course; and
- the blocked wall/ramp course.

It rebuilds each scene's `terrain.bin`, `terrain.obj`, `walkability.bin`, and
`scene.json`, followed by `scenes/index.json`. It does not rebuild or modify
`database.bin`, `features.bin`, `terrain_features.bin`, or
`terrain_support.bin`.

## Chosen Approach

Add a common flat side apron to the existing scenes. Do not scale the central
terrain and do not add duplicate sandbox variants.

The alternatives were rejected because stretching terrain would change stair,
ramp, and curb tests, while duplicate scenes would make scene cycling and
verification needlessly larger. A shared apron preserves the current tests and
makes every existing scene directly explorable.

## Bounds Contract

Introduce a playable half-width of exactly `3.0 m`, separate from the existing
`COURSE_HALF_WIDTH == 0.60 m` feature half-width.

- Procedural scenes whose spawn X is zero publish playable X bounds
  `[-3.0, +3.0]`.
- A GRAIL scene publishes the union of its old playable X interval and the
  interval `[spawn_x - 3.0, spawn_x + 3.0]`. This preserves the complete old
  route envelope and guarantees at least six metres around the spawn.
- Existing playable Z bounds do not change.
- Existing route waypoints, landing holds, spawn positions, and spawn yaw do
  not change.
- Heightfield and lookahead X bounds extend at least `1.0 m` beyond the new
  playable bounds on both sides. Existing mesh/source and lookahead coverage
  remains included. Z heightfield and lookahead bounds do not shrink.
- The existing `0.25 m` classification-only body-footprint halo remains an
  outer-boundary rule. It does not move an obstacle or an internal class
  boundary.

The builder directly expands the existing X bounds and rasterizes the same
surface at the existing `0.02 m` G1HF/v2 resolution. It does not introduce a
second grid format or a custom padding subsystem. Because changing a
binary32 heightfield origin can move internal sample coordinates by rounding,
the locked runtime center-surface probes must agree with the narrow reference
to within `1e-6 m`; analytic feature parameters and boundaries remain exact.

## Surface Contract

Central geometry is unchanged:

- `COURSE_HALF_WIDTH` continues to own the `1.20 m` stair, ramp, cross-slope,
  and mixed-course feature width.
- The blocked course keeps its wall and steep-ramp positions, dimensions, and
  heights.
- GRAIL vertices, transforms, fixed-diagonal interpolation, and exterior
  height remain unchanged.

New procedural columns outside the feature footprint are flat at the scene's
existing exterior ground height. New GRAIL columns outside the captured mesh
are flat at the authoritative `GrailTerrain.exterior_height`. No edge is
extrapolated and no captured curb is stretched into the apron.

The OBJ continues to be generated from the same G1HF/v2 node grid and fixed
diagonal as runtime queries. A visually cheaper mesh with different geometry
is out of scope.

## Walkability Contract

The apron is meant to be usable, not merely visible.

- Known-flat added side cells are class `certified`.
- Existing central certified/stress classification is preserved.
- The 15-degree ramp remains stress only over its central feature lane; its
  known-flat side aprons are certified.
- In GRAIL scenes, the captured mesh/source envelope retains the scene's old
  certified or stress class. More precisely, every cell in the old playable
  rectangle keeps its old class; cells between that rectangle and the source
  mesh X extrema use the same scene class; only cells strictly outside the
  source mesh X extrema can enter the newly certified exterior-flat apron.
- The blocked course retains blocked direct wall and steep-ramp routes. The
  new outer flat lanes are certified through the full playable Z interval, so
  a manually controlled character can walk around the obstacles. The narrow
  central gap is not reclassified as a bypass.
- Internal class boundaries receive no classification halo. The existing
  outer playable-boundary halo remains exact.

Published region rectangles must describe these classes without disagreeing
with any covered G1WM node or leaving an unclassified playable G1WM node.
Existing route outcome and landing-hold metadata remain unchanged, and every
existing route must continue to validate against the same class sequence as
before.

## Builder Structure

Keep the change isolated in the scene-artifact layer:

1. Define named constants for the `3.0 m` playable half-width and existing
   `0.60 m` feature half-width.
2. Add one checked helper that expands existing X bounds to cover the target
   playable/lookahead interval while retaining the same Z bounds and cell
   size.
3. Let scene definitions describe core geometry and the expanded playable
   interval separately.
4. Classify core terrain and added flat aprons explicitly; do not infer
   certification merely from a larger bounding rectangle.
5. Reuse the existing canonical JSON, G1HF/v2, G1WM/v1, OBJ, hash, and atomic
   scene-pack writers.

Runtime scene loading, matching, support retargeting, controller timing, and
scene switching require no new format or branch.

## Failure and Transaction Rules

Scene construction fails before publishing output when:

- a requested expansion is nonfinite, smaller than the retained core, or not
  representable on the runtime lattice;
- a locked center-surface probe differs from the narrow reference by more than
  `1e-6 m`;
- playable/lookahead/heightfield containment fails;
- an apron cell is not proven flat before being marked certified;
- a route changes class or leaves lookahead coverage;
- a region disagrees with G1WM classes; or
- dimension/sample-count arithmetic exceeds existing format caps.

The normal atomic pack writer remains responsible for preventing a partial
scene-pack replacement.

## Verification

Add RED/GREEN coverage for:

- every required scene having at least `6.0 m` playable X width;
- procedural playable X bounds being exactly `[-3.0, +3.0]`;
- GRAIL bounds containing both the old playable interval and
  `[spawn_x - 3.0, spawn_x + 3.0]`;
- one metre of lookahead beyond both playable X edges;
- identical analytic feature dimensions, route JSON, spawn JSON, and central
  walkability classes, plus runtime center-height agreement within `1e-6 m`;
- flat, zero-slope certified probes in both side aprons;
- central stress remaining stress while adjacent flat apron cells are
  certified;
- blocked wall/ramp routes remaining blocked and both outer bypass lanes being
  certified before, beside, and after the obstacles;
- GRAIL source-envelope classification remaining unchanged while exterior
  aprons are certified;
- deterministic repeated scene-pack hashes;
- full scene-schema, region, route, terrain-runtime, scene-runtime, and
  controller-state suites;
- one-frame loading of every rebuilt scene; and
- a 60-second rendered smoke run on the current machine whose median rendered
  rate is at least 25 frames per second, with no terrain-load failure or
  safe-stop caused solely by the expanded outer bounds.

The rebuilt artifact hashes are expected to change. The protected shared
motion-pack hashes must remain exactly unchanged.

## Acceptance Criteria

The change is accepted when all fourteen scenes load through the ordinary
runtime, expose at least six metres of lateral playable area, preserve their
central challenge and existing automated routes, allow manual walking on both
flat side aprons, and pass deterministic artifact and controller verification
without changing the shared motion database.
