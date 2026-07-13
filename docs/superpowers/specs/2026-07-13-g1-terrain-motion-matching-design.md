# G1 Terrain-Aware Motion Matching Design

Date: 2026-07-13

## Summary

Build the first usable G1 terrain-aware motion-matching demo by retaining Daniel
Holden's C++ database search, continuation, trajectory prediction, and pose
inertialization. Python remains an offline data tool that converts the native G1
motions and GRAIL terrain assets into validated runtime artifacts. Raylib is the
authoritative phase-one runtime and renders a debug-quality G1 skeleton.

The existing `g1_mm` Python programs remain prototypes and diagnostic references;
they do not participate in the production runtime. In particular, the new system
does not use periodic nearest-neighbor restarts, commitment timers, artificial
root-height ramps, or partial joint-only inertialization.

## Context and Decision

Two implementations currently exist:

- `motion-matching` contains Holden's complete C++ runtime, modified for the G1
  skeleton. Its remaining leg-collapse behavior has not been isolated from the
  demo's LAFAN-specific contact and IK assumptions.
- `g1_mm` contains a useful Python terrain and database prototype, but its runtime
  is not a faithful motion-matching implementation. It omits live velocity state,
  searches without Holden's transition controls, and uses playback-specific
  heuristics.

The GRAIL robot clips are already native G1 data: a floating root and 29 G1 joint
degrees of freedom. They do not need motion retargeting. They do need a
deterministic representation conversion from MuJoCo/Isaac G1 state into Holden's
ordered local bone transforms and animation database fields.

The selected architecture is therefore:

1. Python offline ingestion and validation for Takara, GRAIL, and terrain assets.
2. Holden's C++ runtime for matching, continuation, inertialization, and playback.
3. Raylib skeleton and terrain visualization in the same C++ process.
4. G1-specific terrain contact/IK as a separate, later validation stage.

The alternatives were rejected for phase one:

- A faithful Python port of the full Holden runtime would integrate naturally with
  MuJoCo but would duplicate already-working transition machinery and expand the
  initial debugging surface.
- Extending `run_terrain_mm.py` would preserve a small prototype, but it would
  require recreating continuation-aware search, full-pose velocity state,
  inertialization, clip-boundary behavior, and contact handling.

## Goals

- Produce a live-driven, kinematic G1 skeleton demo that selects recorded GRAIL
  step motion in response to terrain ahead.
- Ingest all applicable local GRAIL curb robot clips by default, with an optional
  clip limit only for development and tests.
- Preserve complete natural strides between searches and use recorded GRAIL root
  height for climbing.
- Keep terrain-off and terrain-on behavior directly comparable under identical
  input.
- Separate database, matching, playback, terrain, and IK failures so each can be
  tested independently.
- Preserve the original Holden/LAFAN artifacts and the user's current G1 work.

## Non-goals

- Dynamic balance, torque control, or deployment to the physical G1.
- A MuJoCo production viewer in phase one.
- A skinned G1 mesh; the Raylib G1 skeleton is sufficient.
- Retraining or using Holden's learned motion-matching networks.
- Generalizing beyond the locally available GRAIL terrain categories before the
  curb pipeline passes its acceptance tests.
- Rewriting unrelated parts of Holden's demo.

## Repository Ownership

`motion-matching` becomes the canonical implementation. New builder code and
runtime artifacts live there. `g1_mm` remains unchanged as a reference for its
validated terrain alignment, source loading, and historical comparison videos.

New generated artifacts live under a separate directory such as
`resources/g1_terrain/`; the existing `resources/database.bin`,
`resources/features.bin`, and LAFAN backup are not overwritten by the builder.

## Components

### 1. G1 motion ingestion

The Python ingestion layer has adapters for:

- Takara's native 50 Hz G1 motion.
- GRAIL's native 25 Hz G1 root and 29-DoF robot clips.
- GRAIL per-clip USD/reconstruction terrain geometry.

Every adapter produces the same canonical sample structure: timestamp, G1 root
transform, ordered joint state, source clip identity, original frame identity,
and a terrain-height provider. Source-specific loading stops at this boundary.

### 2. Coordinate and skeleton conversion

The canonical runtime representation is Holden's right-handed, Y-up world with a
planar simulation bone whose canonical forward direction is `+Z`. Input G1 motion
is Z-up. The existing verified G1 conversion supplies the skeleton names, parent
ordering, and the proper world-frame change of basis
`(x, y, z) -> (x, z, -y)`. World orientations are changed by quaternion
conjugation, after which local transforms are reconstructed from parent/child
world transforms.

The conversion operates directly on G1 transforms. BVH may remain a debug export,
but it is not required as an intermediate production format. This avoids Euler
round-tripping and makes the GRAIL path identical to the Takara path.

Both sources use a fixed 25 Hz runtime rate: GRAIL remains at its native rate and
Takara is downsampled from 50 Hz. Translation uses time-based interpolation.
Rotation uses shortest-arc quaternion interpolation,
followed by normalization and unrolling. Velocities and angular velocities use
the resampled timestamps and central differences. Clip edges are handled within
each clip and never borrow samples from an adjacent clip.

The C++ runtime also advances at a fixed 25 Hz. Its three trajectory feature
horizons use database offsets 8, 17, and 25 frames (approximately one-third,
two-thirds, and one second); live prediction uses exact one-third-second steps.

The simulation bone keeps the G1 conventions already established by the current
generator: planar position derived from the torso reference and heading derived
from the projected G1 pelvis forward axis. Each source clip becomes an explicit
database range.

### 3. Contacts

Left and right contact channels are aligned with the configured G1 foot/contact
bones. Contact extraction uses foot speed together with ground-relative foot
height so a slowly moving airborne foot cannot be classified as planted. A short
temporal majority filter removes isolated contact flips. Contact computation uses
each clip's terrain provider rather than a global flat floor.

Contact thresholds are configuration values recorded in the manifest, not hidden
constants in source adapters.

### 4. Terrain representation

The offline terrain interface exposes `height(world_x, world_z)` in Holden's Y-up
coordinate system. The existing GRAIL USD plus reconstruction transform remains
the source of truth for per-clip alignment. The builder exports a compact runtime
heightfield and a corresponding Raylib mesh for the terrain selected by the demo.

All height queries define missing or out-of-bounds terrain as an error during
database construction. The runtime heightfield may define a documented flat
exterior region around the exported terrain so live steering outside the curb
footprint remains well-defined.

### 5. Feature database

Holden's native 27 dimensions remain unchanged:

- left and right foot position: 6 values;
- left and right foot velocity: 6 values;
- hip velocity: 3 values;
- three future trajectory positions: 6 values; and
- three future trajectory directions: 6 values.

Four terrain dimensions are appended, making a 31-dimensional matching vector.
The five dead legacy slots and the incompatible 34-dimensional `g1_mm` feature
layout are not imported.

The terrain values are ground-relative height changes at arc distances
`0.25, 0.50, 0.75, 1.00 m` along a geometric facing centerline:

`terrain_height(sample) - terrain_height(character_root)`

For database frames, the centerline starts at the root and bends according to the
recorded future root headings. For live queries, it bends according to Holden's
predicted trajectory headings. Arc distance is geometric look-ahead distance, not
distance the character is guaranteed to travel within the ordinary one-second
trajectory horizon. The centerline is therefore extended from the final available
heading until it reaches 1 m. If translation is near zero, the current facing
direction supplies the centerline. This makes the terrain query defined while the
character is stopped and gives recorded and live curved paths identical semantics.

The terrain block has its own tunable group weight. Feature normalization follows
Holden's existing group-normalization convention. Terrain weight zero is the
control condition and removes terrain from search cost without changing artifacts.

### 6. Artifact contract

The builder writes a versioned artifact set under `resources/g1_terrain/`:

- `database.bin`: Holden-compatible animation, range, and contact arrays;
- `terrain_features.bin`: header plus four `float32` values per database frame;
- `manifest.json`: schema version, ordered skeleton names and parents, coordinate
  convention, source clips, source-frame mapping, resampling parameters, contact
  parameters, feature semantics, frame/range counts, and terrain assets;
- runtime terrain heightfield and Raylib mesh; and
- validation summary with numeric error bounds.

The terrain sidecar header contains a magic value, schema version, frame count,
and feature dimension. The C++ loader requires exact agreement among the base
database, sidecar, manifest skeleton signature, and compile-time/runtime feature
layout before allocating the matching database.

Keeping terrain data in a validated sidecar preserves compatibility with Holden's
unversioned base database format while avoiding any dependence on the original
resources directory.

### 7. C++ matching runtime

The ordinary database matcher remains authoritative. It initializes the current
frame as the incumbent, compares candidates against the continuation cost, ignores
nearby frames as Holden already specifies, excludes range ends needed by the
future horizon, and advances the selected clip sequentially between searches.

The live query copies current pose and velocity features from the current
inertialized motion state, computes future trajectory features from live input,
and samples terrain along that predicted path. It does not zero the foot/root
velocity block and does not restart search from an unconstrained global nearest
neighbor.

When a lower-cost candidate is selected, Holden's complete pose inertializer
transitions bone positions, velocities, rotations, and angular velocities. Root
height comes from the selected recorded motion. There is no artificial ground
ramp, fixed-duration commitment timer, or joint-angle-only offset.

The learned matching path stays disabled. Its existing networks were trained for
a different skeleton and feature layout.

### 8. Visualization and diagnostics

Raylib is the only phase-one viewer. It renders:

- final G1 skeleton transforms after matching and inertialization;
- the selected GRAIL terrain mesh;
- predicted trajectory and four terrain sample points;
- contact state and lock targets when IK is enabled; and
- current source clip/frame, incumbent and selected cost, terrain contribution,
  and transition events.

The deterministic autoplay harness writes the same state to a machine-readable
CSV or line-oriented log. Rendering is diagnostic output from the C++ runtime,
not a second playback implementation.

### 9. G1 terrain IK

IK remains disabled until raw sequential playback and motion matching pass. The
existing Holden foot IK assumes a LAFAN-style two-joint leg, a particular knee
pole axis, a `+X` toe direction, and a flat world-height clamp. Those assumptions
are not accepted for G1 by default.

The G1 IK layer uses explicit per-leg configuration for hip, knee, ankle, contact
bone, knee pole, and toe axis. Ground targets come from the runtime terrain-height
provider. Corrections are bounded and may not change root motion or database
state. Turning IK off must always recover the accepted unadjusted result, making
IK a reversible downstream stage.

## Error Handling

The builder fails without publishing artifacts when it finds:

- missing or unmapped G1 bodies/joints;
- inconsistent G1 skeleton order or parents;
- non-finite transforms, features, or terrain samples;
- invalid quaternion norms or discontinuities after unrolling;
- range overlap, empty ranges, or source-frame discontinuity;
- unavailable GRAIL terrain/reconstruction pairs; or
- FK round-trip error beyond the accepted tolerance.

Artifacts are written to a temporary output set and published only after all
validation passes. A skipped GRAIL clip is allowed only in an explicit
best-effort diagnostic mode; the normal full build treats a skipped clip as a
failure and lists the source path and reason.

The C++ runtime fails fast with an actionable message for schema, frame-count,
feature-dimension, skeleton, or terrain mismatch. It never silently falls back to
zero terrain features.

## Validation Gates

Implementation advances only after the current gate passes.

### Gate 1: database correctness

- Reconstructed Holden FK agrees with source G1 FK on sampled frames and every
  body within 1 mm maximum positional error.
- Every quaternion is finite and within `1e-4` of unit length.
- Bone-local offsets remain constant within 1 mm.
- Resampled duration differs from source duration by no more than one 25 Hz frame.
- Frame counts, source maps, and range boundaries agree across every artifact.
- Terrain samples agree with direct GRAIL mesh queries within the exported
  heightfield resolution.

### Gate 2: sequential playback

Play complete representative Takara and GRAIL clips in Raylib with matching,
inertialization transitions, adjustment, clamping, and IK disabled. Acceptance:

- no root discontinuity above 10 degrees or 5 cm between adjacent resampled
  frames unless the source contains the same discontinuity;
- no non-finite transforms;
- constant bone lengths; and
- visual reproduction of the source G1 motion without leg collapse.

### Gate 3: flat motion matching

Enable Holden search and inertialization on flat terrain with terrain weight zero.
Drive forward and turn using a deterministic input script. Acceptance:

- database indices advance sequentially between logged searches;
- no repeated sub-stride cycle shorter than 0.5 seconds;
- transitions report a candidate cost below the incumbent continuation cost;
- no cross-range sequential advancement; and
- no root spin or leg collapse with IK still disabled.

### Gate 4: terrain selection

Run the same scripted curb approach twice from identical initial state:

- control: terrain weight zero;
- treatment: validated nonzero terrain weight.

Acceptance for the treatment:

- terrain queries become positive before the curb edge;
- selected frames come from a compatible GRAIL approach/step range;
- the selected range advances through the recorded step rather than restarting a
  short slice;
- root height rises through recorded motion without a synthetic height ramp; and
- the terrain-aware run has materially lower terrain-feature error than control.

The control is evidence about feature effect, not required to climb successfully.

### Gate 5: G1 terrain IK

Enable the G1-specific IK after Gate 4 passes. Acceptance:

- planted-foot horizontal drift decreases relative to IK-off playback;
- foot targets use sampled terrain height and do not penetrate the heightfield;
- joint corrections stay within configured bounds;
- disabling IK reproduces the accepted Gate 4 output; and
- IK does not alter selected database frames or matching costs.

## Phase-One Completion Criteria

Phase one is complete when the repository can reproducibly:

1. build validated Takara plus GRAIL G1 terrain artifacts without overwriting the
   existing resources;
2. run deterministic flat and curb tests through Holden's C++ matcher;
3. run a live-driven Raylib G1 skeleton demo with terrain-aware selection;
4. produce logs showing continuation costs, selected costs, terrain samples,
   source ranges, and transitions; and
5. produce a terrain-aware comparison video backed by passing validation gates.

IK improvement is part of the complete phase-one demo, but failure in IK cannot
invalidate or obscure the accepted matching result because Gate 4 is preserved as
an IK-off baseline.

## Implementation Order

1. Freeze artifact formats, skeleton configuration, and validation fixtures.
2. Build direct native-G1 ingestion and FK round-trip tests.
3. Add GRAIL terrain alignment, path sampling, and sidecar generation.
4. Add strict C++ artifact loading and the 31-dimensional database feature layout.
5. Add live terrain queries and feature/cost diagnostics.
6. Validate sequential playback, then flat matching, then terrain matching.
7. Replace the G1-incompatible IK assumptions and validate the reversible IK
   stage.
8. Record the deterministic comparison and live skeleton demo.
