# G1 Torch Stair Kinematics Design

Date: 2026-07-29

## Summary

Build a small, renderer-independent terrain motion-matching experiment on the
qualified native Torch matcher. The experiment uses one four-step GRAIL stair
object, four recorded traversals, and the existing flat Takara walk. It compares
flat matching, the legacy four-height feature, and a dense local height patch
entirely kinematically before SONIC or depth estimation is introduced.

The flat Torch matcher remains unchanged on
`research/g1-low-latency-driver`. This work lives on
`research/g1-torch-terrain-kinematics`.

## Question This Experiment Answers

Can exact motion matching select useful recorded stair kinematics when given an
ideal local terrain observation?

If the dense privileged-height condition cannot select and continue appropriate
stair motion on this deliberately small corpus, depth learning and SONIC tracking
are out of scope until the matcher or feature representation is improved. If it
can, the dense height observation becomes the teacher-side target that a later
depth/proprioception encoder must reproduce.

This experiment does not claim physical stability or sim-to-real performance.

## Frozen Data Slice

The terrain data contains one four-step ash-hardwood practice stair in GRAIL
`stair_p1`. The four robot recordings share object identity
`04d99a9e43`:

- ordinary traversal `__0000`;
- ordinary traversal `__0001`;
- ordinary traversal `__0002`; and
- bidirectional traversal `_updown__0000`.

Each local PKL declares `fps=25.0` and contains 250 frames. The matching dataset
also contains the existing flat Takara walk as a control. No other GRAIL motion
is discovered implicitly, even when additional files exist on disk.

The converter resolves the four source files by an explicit checked manifest,
not a broad glob. It records source paths, SHA-256 identities, source frame
counts, source rates, output frame counts, and conversion parameters.

## Coordinate and Rate Contract

The runtime stays in the existing Torch matcher's native convention:

- 50 Hz with `dt=0.02`;
- right-handed Z-up world;
- planar axes X/Y;
- positive X as the G1 forward axis;
- wxyz quaternions; and
- the existing `g1-29dof-isaaclab-v1` joint/body layout.

The four GRAIL clips are converted from their recorded 25 Hz samples to 50 Hz
once, offline. Root and body translations use timestamp interpolation. Root and
body rotations use normalized shortest-arc quaternion interpolation with
temporal sign unrolling. Joint positions use timestamp interpolation. Joint,
body-linear, and body-angular velocities are derived consistently at 50 Hz.
The conversion preserves the first and last source timestamps and never
extrapolates.

No Y-up Holden representation appears in the new runtime or generated motion
folder.

## Generated Motion-Folder Contract

Each converted stair clip is written as a native `motion.npz` accepted by
`MotionFolder.load`, containing the existing required arrays:

- `fps`;
- `joint_pos` and `joint_vel`;
- `body_pos_w` and `body_quat_w`;
- `body_lin_vel_w` and `body_ang_vel_w`.

Each stair clip also has a terrain sidecar associated by relative clip path. The
sidecar identifies the source stair surface and contains database-frame terrain
observations for both experimental encoders. The flat Takara control receives
zero-valued terrain observations through an explicit flat provider; the original
Takara archive is not modified.

Generated data is ignored and written outside tracked source files. The checked
manifest is tracked so another machine can reproduce the same five-clip
inventory.

## Terrain Conditions

Every rollout uses the same motion data, initial state, command stream, search
cadence, transition penalty, and inertialization parameters. Only the terrain
feature condition changes.

### Flat

The existing 27 motion features are used without terrain dimensions. This is the
negative control.

### Legacy Four-Height

Four terrain-height deltas are sampled along the predicted centerline at
0.25, 0.50, 0.75, and 1.00 metres. Each value is relative to the terrain height
at the character's current planar root location. This reproduces the semantic
idea of the old feature in native 50 Hz/Z-up Torch code; it is a comparison
condition, not the target implementation.

### Dense Local Height Patch

The preferred condition samples a root-heading-local grid:

- forward coordinates: -0.15 through 1.65 metres at 0.15-metre spacing
  (13 rows);
- lateral coordinates: -0.45 through 0.45 metres at 0.15-metre spacing
  (7 columns);
- 91 scalar height values in forward-major order; and
- every value relative to terrain height at the current planar root location.

Database observations use each recording's aligned GRAIL terrain. Query
observations use the test scene's privileged height provider. The same sampling
function produces database and query rows. Terrain dimensions form one
independently weighted normalized feature group.

The patch deliberately includes both foot corridors and a small region behind
the root. It does not include global root height, object identity, scene identity,
or privileged future motion labels.

## Torch Architecture

The current 27-dimensional feature extractor remains the sole flat-motion
definition. Terrain support is additive:

1. a strict small-corpus converter produces native 50 Hz clips;
2. a terrain provider samples aligned GRAIL surfaces in native coordinates;
3. interchangeable terrain encoders produce no feature, four heights, or the
   dense patch;
4. the database concatenates and normalizes the selected terrain group;
5. the runtime query calls the same encoder on the active test surface; and
6. exact GPU search, continuation, command shaping, and inertialization reuse the
   qualified Torch matcher behavior.

The terrain encoder is selected when constructing a matcher. It cannot change
mid-rollout, which prevents normalization or dimensionality from changing under
live state.

Terrain-aware code must not add a second implementation of the existing 27
motion features.

## Kinematic Rollout and Visualization

The first benchmark uses a deterministic straight traversal aligned to the
selected stair recording. It initializes on the approach floor and applies one
recorded-speed command stream to all three conditions. No MuJoCo integration step
or SONIC policy inference occurs.

Each step returns the normal dense 50 Hz root and joint windows plus enough body
kinematics for foot-clearance evaluation. A lightweight viewer consumes saved
rollouts and renders the stair surface, G1 kinematic skeleton, root trajectory,
selected source clip/frame, and current terrain samples. Rendering never owns or
changes matcher state.

## Measurements and Decision Rule

Every rollout records:

- selected source clip and frame;
- search and transition events;
- motion-feature, terrain-feature, and total costs;
- per-step and search latency;
- root trajectory and orientation;
- left/right foot height relative to the test surface;
- minimum pelvis height relative to the test surface;
- below-surface foot penetration; and
- progress along the commanded stair direction.

The experiment is promising only if the dense condition:

- selects a stair-source frame no later than 0.20 metres before the first riser;
- reaches at least 90% of the reference traversal's horizontal progress;
- reaches at least 80% of the reference traversal's root-height gain;
- keeps sampled foot penetration at or above -0.03 metres; and
- either reaches the upper landing when the flat condition does not, or reduces
  integrated below-surface foot penetration by at least 50% relative to flat.

The reference progress and height gain are computed from the converted `__0000`
recording before any matcher condition is run and are written to the benchmark
configuration. The legacy condition is reported but is not required to succeed.

## Validation

Automated tests cover:

- the explicit four-file inventory and source identities;
- 25-to-50 Hz endpoint and quaternion interpolation;
- native G1 shapes, finite values, and quaternion norms;
- flat, legacy, and dense encoder shape/order semantics;
- identical database/query encoder math;
- terrain group normalization and weighting;
- flat-mode equivalence with the qualified 27-feature matcher;
- deterministic rollout identity;
- body/foot output consistency; and
- structured metrics and latency output.

A protected integration test converts the four real local GRAIL clips and checks
that the resulting folder loads through the strict native motion loader. Large
generated clips are never committed.

## Failure Handling

Missing GRAIL files, mismatched object identities, non-25-Hz sources, malformed
records, missing terrain geometry, non-finite interpolation, or invalid
quaternions fail before publishing generated data. Dataset publication is
transactional so an interrupted conversion cannot leave a partially valid
motion folder.

At runtime, an unavailable or out-of-domain terrain query is a contract error,
not an implicit flat sample.

## Non-Goals

- Loading the complete 12,188-file stair corpus.
- Reproducing the old C++ matcher numerically.
- Training a depth encoder.
- Using global XY or privileged global root height as a feature.
- SONIC tracking, physics, balance, contact control, or real-robot deployment.
- Obstacle avoidance or manipulation.

## Follow-On Work

If the dense privileged-height experiment is promising:

1. test additional held-out stair geometries;
2. define the depth/proprioception observation window;
3. train an encoder to predict the useful terrain representation or matching
   decision;
4. measure end-to-end observation-to-kinematic latency; and
5. qualify selected 50 Hz kinematics through SONIC.
