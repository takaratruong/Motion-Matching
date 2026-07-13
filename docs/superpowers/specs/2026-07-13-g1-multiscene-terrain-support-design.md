# G1 Multiscene Terrain, Support Retargeting, and Clearance Design

Date: 2026-07-13

Status: approved in conversation; written specification pending user review

## Summary

Extend the approved Task 5 G1 terrain motion-matching runtime from one visual
GRAIL curb scene into a reproducible multiscene terrain playground. The same
459,682-frame motion library will drive dataset-derived curbs, procedural
blocks, stairs, ramps, elevated landings, mixed courses, and mild cross-slopes.

The extension must also correct the behavior observed during Task 5 testing:
the character can pass beneath a raised surface, a foot or shin can intersect a
step, and a later recorded pelvis rise can look like collision response. The
runtime currently has no terrain collision and keeps the `Simulation` bone
planar at world Y=0. Terrain affects only the 31-dimensional search cost; pelvis
height arrives later through the selected pose and inertialization, and the G1
IK stage is disabled.

The design adds four independent layers:

1. one immutable motion pack plus independently validated scene packs;
2. a contact-aware support-frame transform that preserves motion matching on
   arbitrary world levels;
3. a traversability guard that stops at terrain outside the certified motion
   envelope; and
4. bounded G1 terrain IK for planted-foot locking, surface alignment, and swing
   clearance.

These layers do not fake arbitrary climbing. Novel geometry inside the tested
motion envelope is expected to traverse. Geometry outside the certified
envelope must stop the controller before penetration.

## Approved Requirements

- Keep Daniel Holden's database matcher, full-pose inertialization, and 25 Hz
  runtime authoritative.
- Keep the existing 27 motion/trajectory features plus four longitudinal
  terrain features. Do not change the 31-dimensional matching contract in this
  increment.
- Load the motion library once and allow selection among multiple terrain
  scenes without rebuilding or duplicating the motion database.
- Include GRAIL-derived scenes and procedural blocks, stairs, ramps, elevated
  landings, mixed courses, and mild cross-slopes.
- Include stair layouts and dimensions that are not exact copies of GRAIL.
- Continue ordinary motion matching after climbing to a different world level.
  Selecting a flat clip on a raised landing must not pull the character back to
  world zero.
- Support ascent and descent. A ramp must change support height continuously;
  stairs may change it in contact-consistent increments.
- Stop before steps, walls, or slopes that exceed the certified capability.
  Never lift the body through an unsupported obstacle merely to avoid a visual
  intersection.
- Use one surface definition for matching queries, clearance checks, IK, and
  the visible terrain. The current roughly 14 cm GRAIL query dilation around
  mesh edges must not remain.
- Keep terrain IK downstream of matching. Enabling IK must not alter selected
  database frames, ranges, costs, or terrain queries.
- Log enough state to distinguish selection, inertialization, support-frame,
  adjustment/clamping, traversability, and IK failures.

## Current Root Cause

The observed effect is not a physics collision response.

- Offline conversion creates a planar `Simulation` bone from torso XZ and
  explicitly writes its Y coordinate as zero. Floating pelvis height is stored
  in the child `Hips` local position.
- The live simulation controller integrates desired XZ motion without sampling
  terrain.
- Four relative terrain heights influence database search, but they do not
  constrain the root, pelvis, feet, or shins.
- On a transition, `Hips` and every other non-root bone are inertialized toward
  the selected recording. A recorded rise can therefore arrive after the
  visible terrain boundary and appear to push the character upward.
- Root adjustment and the 15 cm clamp currently operate in 3D against the
  independent planar simulation position, which can displace a recorded step
  relative to the terrain.
- Contact channels are metadata only while IK is disabled. The inherited IK is
  intentionally unusable for G1 because it assumes a LAFAN leg chain, a flat
  floor, and a hard-coded toe direction.
- The current GRAIL height provider takes the maximum point height within a
  0.14 m radius. The rendered OBJ uses the source mesh. The sampled top surface
  can therefore start about 13--14 cm before the visible top.

A read-only check of the first native GRAIL range found the unblended source
pose clear of its source terrain: minimum pelvis clearance was about 0.582 m,
toe clearance about 0.035 m, and ankle clearance about 0.052 m. This supports a
runtime selection/alignment/blending cause rather than corrupt source motion.

The existing terrain sidecar contains observed relative changes from
`-0.372856557 m` to `+0.372856557 m`. This is an upper data envelope, not an
automatic certification that every 0.37 m obstacle is traversable.

## Architecture

### 1. Motion pack and scene packs

Keep one generated motion pack and add a scene directory:

```text
resources/g1_terrain/
  database.bin
  terrain_features.bin
  terrain_support.bin
  manifest.json
  validation.json
  scenes/
    index.json
    grail-curb-default/
      scene.json
      terrain.bin
      terrain.obj
      walkability.bin
    stairs-shallow/
      scene.json
      terrain.bin
      terrain.obj
      walkability.bin
    ...
```

`database.bin`, `terrain_features.bin`, `terrain_support.bin`, the motion
manifest, and motion validation are loaded once. A scene switch replaces only
the runtime heightfield, terrain model, walkability grid, and scene metadata.
The motion manifest records the support-sidecar schema and SHA-256, the exact
GRAIL surface-semantics signature, and the scene-index schema. A runtime may not
combine artifacts with different signatures.

`scenes/index.json` contains ordered stable scene IDs. Each `scene.json`
contains:

- schema and scene ID;
- human-readable label;
- provenance (`grail` or `procedural`) and source identifiers or primitive
  parameters;
- Holden Y-up coordinate signature;
- terrain feature distances `[0.25, 0.50, 0.75, 1.00]`;
- heightfield interpolation convention and cell size;
- SHA-256 hashes of `terrain.bin`, `terrain.obj`, and `walkability.bin`;
- mesh, heightfield, playable, and one-metre-lookahead bounds;
- spawn position and yaw;
- certified and stress-test regions; and
- expected deterministic routes and outcomes.

`walkability.bin` uses magic `G1WM`, version 1, the same `nx` and `nz` as the
scene heightfield, then one row-major `uint8` per cell: 0 is blocked, 1 is
certified traversable, and 2 is stress. Any other value or grid mismatch is an
error.

Startup selection uses `MM_TERRAIN_SCENE=<id>`, defaulting to
`grail-curb-default`. Live UI provides Previous, Next, and Reset controls and
shows the active scene ID. Scene switching is transactional:

1. parse and validate the candidate metadata;
2. load and validate its heightfield and walkability grid;
3. load the candidate Raylib model and verify it is ready;
4. commit all candidate objects together;
5. reset the complete controller state to the scene spawn; and
6. unload the previous model.

Any failure preserves the prior scene and controller state and reports the
candidate path and exact validation error.

### 2. Exact query/render surface

Scene heightfields use a versioned fixed-diagonal triangle grid. Runtime height
sampling uses the same triangle interpolation and diagonal as the exported OBJ.
The OBJ is generated from the height grid rather than from a separate visual
approximation. Every terrain query, normal query, clearance test, IK target, and
visible triangle therefore refers to one surface.

Legacy G1HF/v1 remains readable only for migration. New scene packs use the
existing `G1HF` magic with version 2 and the same `nx`, `nz`, origin, cell-size,
and exterior-height fields. Version 2 fixes every cell diagonal from its
minimum-X/minimum-Z corner to its maximum-X/maximum-Z corner; no runtime option
may change it. `scene.json` repeats this convention so a mixed metadata/binary
pair is rejected.

GRAIL scene generation rasterizes the transformed USD mesh with vertical
triangle intersection. It does not use radius-based maximum-point dilation.
Because the same GRAIL provider creates per-motion terrain rows, support rows,
contacts, and runtime scenes, changing this semantic requires one complete
offline artifact rebuild and validation.

The surface-parity gates are:

- grid-node query and OBJ vertex heights agree within `1e-6 m`;
- deterministic within-cell probes agree within `1e-4 m`;
- away from discontinuities, rasterized GRAIL height differs from the source
  mesh top by at most `0.005 m`; and
- a discontinuity edge may move by at most one grid cell (`0.02 m`), never the
  previous 0.14 m query radius.

### 3. Source support sidecar

Add `terrain_support.bin` with a 16-byte little-endian header containing magic
`G1SP`, version 1, frame count, and dimension 3, followed by three row-major
`float32` values per database frame:

1. source terrain height beneath the planar `Simulation` root;
2. source terrain height beneath the left contact/toe XZ; and
3. source terrain height beneath the right contact/toe XZ.

Takara flat rows contain zero. GRAIL rows use the exact per-clip terrain
provider also used for contacts and the four matching features. The loader
requires exact frame parity with `database.bin` and rejects non-finite values,
truncation, trailing bytes, unsupported versions, or dimensions other than
three before Raylib initialization.

The support sidecar is coordinate/placement metadata, not a matching feature.
It must not change feature normalization, bounds, costs, or selected frames.

### 4. Contact-aware support-frame retargeting

The runtime keeps two related roots:

- the existing planar simulation state controls XZ position and yaw; and
- a scalar world support transform places the recorded pose on the active
  terrain level.

For every selected frame, the runtime forms source-to-runtime height deltas:

```text
root_delta  = runtime_height(root_xz)       - source_root_height(frame)
left_delta  = runtime_height(left_toe_xz)   - source_left_height(frame)
right_delta = runtime_height(right_toe_xz)  - source_right_height(frame)
```

Active recorded contacts choose the support estimate. With one active contact,
use that foot's delta. With two contacts, use their finite mean. With no active
contact, retain the last support estimate and blend toward `root_delta` only
after the airborne interval exceeds two frames, using a 0.10 s critically
damped half-life at 25 Hz. This prevents the support level from falling to zero
during an ordinary swing while keeping a prolonged no-contact state bounded.

The support transform is applied as world Y on the `Simulation` bone before
final FK. It shifts the whole recorded pose without changing local joint
geometry or recorded pelvis motion relative to its source terrain. Adjustment
and clamping become horizontal-only; they may modify XZ/yaw but never pull the
support transform toward world zero.

When search changes the selected source frame or range, rebase the support
transform and the pose inertializer together so the rendered world pose and
vertical velocity remain continuous. The transform does not decay toward zero.
It changes only from contact-consistent runtime/source support deltas.

This gives the required multilevel invariant:

```text
flat terrain features on a +H landing may select an ordinary flat clip,
but its rendered pose remains translated by +H until descent establishes
a lower support level.
```

No synthetic climb curve is added. The selected recording still supplies the
relative pelvis and leg trajectory. Support retargeting supplies only the
coordinate transform between source ground and runtime ground.

### 5. Matching behavior

The first implementation retains the approved 31-dimensional matcher:

- 27 Holden pose/trajectory values; and
- four centerline terrain changes at 0.25, 0.50, 0.75, and 1.00 m.

Longitudinal ramps appear as gradual values in this vector. A raised flat
landing produces zero relative terrain values and may correctly select flat
locomotion; the support frame preserves the landing elevation.

Cross-slope is not represented by the centerline vector. Mild cross-slopes are
handled by per-foot terrain targets, surface normals, and bounded IK. This
increment does not add lateral terrain dimensions without evidence from the
deterministic logs that matching, rather than IK reachability, is the limiting
factor. Any later feature-schema change requires a separate design and complete
artifact rebuild.

Terrain weight shown in the UI must always equal the effective weight used to
build the in-memory feature normalization and search bounds. Changing it either
rebuilds immediately or presents an explicit unapplied state; it may never look
active while the database still uses a prior weight.

### 6. Traversability guard

Motion matching chooses motion; it is not a general collision solver. Each scene
therefore supplies a walkability grid with `traversable`, `stress`, and
`blocked` cells. The generator derives it from primitive parameters or measured
GRAIL surface changes, then records the decision in `scene.json`.

The controller sweeps the planar character footprint over the next XZ movement.
If the sweep reaches a blocked cell, it smoothly reduces desired speed to zero
without advancing through the boundary. It displays the blocked sample in red
and logs the reason. The guard may not alter world Y to bypass the block.

The observed `0.372856557 m` rise/drop is a hard upper data bound. Actual
certification is evidence-based:

- procedural step rises of `0.08 m`, `0.12 m`, and `0.16 m` begin as expected
  traversable cases;
- a `0.45 m` wall is always blocked;
- 5-degree and 10-degree longitudinal ramps begin as expected traversable;
- a 15-degree ramp is a stress case and must either traverse without violating
  clearance gates or stop safely; and
- ramps at or above 20 degrees begin blocked until a later passing validation
  explicitly certifies them.

The runtime's certified limits may become stricter after testing. They may not
be loosened merely because a dimension lies within the raw feature range.

### 7. G1 terrain IK and clearance

The planned G1 IK remains a reversible downstream stage and is extended in
three ways:

1. planted contacts lock to the sampled height and normal beneath each foot;
2. ankle/foot orientation aligns to the local support plane within configured
   correction bounds; and
3. a swing-foot target samples the swept toe/foot path and adds only the bounded
   clearance needed to pass the next tread or ramp surface.

Each leg names its G1 hip, knee, ankle, contact bone, knee pole, and foot axis.
No parent-walk inference or hard-coded LAFAN `+X` toe direction is allowed.

Foot, toe, and shin clearance probes are evaluated against the same scene
surface. IK corrections are clamped to the reachable leg shell and to `0.35`
radians per configured joint per solve. An unreachable target requests a safe
slow/stop from the traversability layer; it never stretches a leg, moves through
the obstacle, or changes matching state.

IK-off output remains the support-retargeted accepted baseline. IK-on and IK-off
runs must have byte-identical selected frame, range, transition, query, and cost
columns.

## Required Scene Set

The initial index contains these deterministic scenes:

1. `grail-curb-default`: corrected exact-surface version of the current demo.
2. `grail-curb-low`, `grail-curb-medium`, and `grail-curb-high`: choose the
   lexically first GRAIL candidate nearest measured maximum heights 0.12, 0.24,
   and 0.36 m respectively.
3. `stairs-shallow`: four 0.08 m rises, 0.30 m runs, 1.20 m width, and a 2.0 m
   raised landing.
4. `stairs-standard`: three 0.12 m rises, 0.32 m runs, 1.20 m width, and a 2.0 m
   raised landing.
5. `stairs-unseen-variable`: rises `[0.06, 0.10, 0.08, 0.12] m`, runs
   `[0.24, 0.34, 0.28, 0.38] m`, 1.20 m width, and a 2.0 m landing. This is the
   required stair geometry not copied from the dataset.
6. `ramp-05-up-down` and `ramp-10-up-down`: 1.20 m-wide longitudinal ramps
   rising 0.36 m, a 2.0 m elevated landing, and descent at the same grade. Ramp
   run is exactly `0.36 / tan(angle)` metres.
7. `ramp-15-stress`: the same 1.20 m-wide, 0.36 m-rise layout marked stress
   rather than certified.
8. `cross-slope-05` and `cross-slope-10`: 4.0 m-long, 1.20 m-wide lateral
   grades with 1.0 m flat entry and exit zones.
9. `mixed-multilevel`: four 0.08 m stairs up, 3.0 m of elevated ordinary
   walking, three full-width blocks adding signed height changes
   `[+0.08, -0.12, +0.04] m` with 0.60 m tops, and a 10-degree ramp back to the
   base level.
10. `blocked-course`: a 1.20 m-wide, 0.45 m-high wall with a 0.50 m top and a
    separate 1.20 m-wide, 0.36 m-rise, 25-degree ramp, both marked blocked.

Every procedural course provides at least 2.0 m of flat spawn area and at least
1.0 m of valid lookahead margin around every certified route.

## Runtime Data Flow

At each fixed 25 Hz update:

1. read user or deterministic input;
2. sweep the planned XZ movement against walkability and reduce blocked motion;
3. predict the ordinary Holden trajectory;
4. sample the four terrain query values from the active scene;
5. build and search the 31-dimensional query when scheduled;
6. advance the selected database range sequentially;
7. update full-pose inertialization;
8. compute source/runtime support deltas from contacts and support rows;
9. apply the world support transform and horizontal-only adjustment/clamping;
10. run FK for the support-retargeted IK-off pose;
11. optionally apply bounded G1 terrain IK and rerun affected FK;
12. compute clearance and acceptance diagnostics; and
13. render the shared query/clearance surface, skeleton, trajectory, samples,
    support state, contacts, and blocked markers.

Scene switching occurs only between updates and performs the complete reset
after candidate validation.

## Diagnostics

The deterministic CSV records at least:

- frame, fixed timestep, scene ID, input mode, and scene route;
- current database frame, range, source name, source terrain, and transition;
- incumbent, selected, continuation, and terrain-only costs;
- effective terrain weight and four query values/points;
- root, left-foot, and right-foot source support heights;
- corresponding runtime heights, chosen support delta, and support source;
- raw selected, inertialized, support-retargeted, and IK-adjusted Hips Y;
- adjustment and clamp displacement, separated into XZ and Y;
- contact state, contact lock, target height, target normal, and IK correction;
- signed heightfield clearance for Hips, knees, ankles, toes, feet, and shin
  capsules;
- walkability class, blocked reason, distance to boundary, and commanded versus
  applied speed; and
- enabled-state flags for matching, adjustment, clamping, support retargeting,
  and IK.

Logs distinguish the invisible planar simulation reference from physical joint
clearance. Bone 0 being below a raised top is not itself a failure.

## Error Handling

Before opening Raylib, reject motion-pack errors including missing files, bad
magic/version/dimensions, frame mismatch, non-finite support values, skeleton or
feature signature mismatch, and changed surface semantics without a matching
artifact rebuild.

Reject a candidate scene transactionally for:

- unknown or duplicate ID;
- path traversal or overlong path;
- missing, empty, truncated, or trailing artifact data;
- non-finite or invalid bounds, spawn, grid, primitive, or route values;
- hash mismatch;
- mesh/heightfield interpolation or coordinate-signature mismatch;
- spawn/playable/lookahead bounds outside the heightfield;
- an expected route entering a blocked or out-of-bounds cell; or
- surface-parity failure.

At runtime, a non-finite query, support estimate, pose, IK correction, or
clearance value requests a controlled diagnostic exit. It must not silently
fall back to zero terrain or continue with a partially loaded scene.

## Acceptance Gates

### Gate A: Baseline diagnosis

Use the planned Task 6 logger before changing support behavior. Reproduce one
deterministic curb approach with terrain weight 4 and record the first positive
query, selected range, Hips inertial offset, support/clearance values, and any
adjustment/clamp displacement. The log must identify whether raw selected pose
or only the blended/rendered pose penetrates.

### Gate B: Artifacts and surface parity

- All Python and C++ artifact tests pass in standard, strict-warning,
  fast-math/release, and sanitizer configurations where applicable.
- The full 1,770-clip motion pack rebuild passes existing FK, duration,
  quaternion, range, source-map, and skeleton gates.
- Contacts, terrain features, and support rows use the exact same GRAIL surface.
- Every required scene passes hash, bounds, route, and surface-parity checks.

### Gate C: Matching and multilevel support

For each expected-traversable stair and ramp route:

- terrain queries become active before ascent;
- a compatible GRAIL range is selected and advances without a repeated cycle
  shorter than 0.5 s;
- every transition beats the incumbent continuation cost;
- world elevation on the landing agrees with scene elevation within `0.02 m`;
- at least 2.0 s of ordinary flat motion matching continues on the elevated
  landing without support height drifting toward zero;
- the descent returns to the lower level within `0.02 m`;
- no non-source one-frame Hips displacement exceeds `0.05 m`; and
- every logged transform and cost remains finite.

Terrain-weight-zero and terrain-weight-four runs use identical scripted input.
The treatment must have lower terrain-feature error and must satisfy the
existing sub-stride detector.

### Gate D: Traversability

On `blocked-course` and every stress case that cannot satisfy traversal gates:

- the swept planar footprint never enters a blocked cell;
- applied speed reaches zero before the visible boundary;
- closest horizontal clearance remains at least `0.02 m`;
- support height does not rise more than `0.02 m` in response to the blocked
  surface; and
- no transition loop or non-finite state occurs while stopped.

### Gate E: IK and physical clearance

Compare matched IK-off and IK-on runs for each certified route:

- selected frames, ranges, transitions, queries, and costs are identical;
- planted-foot horizontal drift decreases with IK;
- planted toe/foot penetration is no more than `0.005 m`;
- all logged physical-joint and foot/shin capsule penetration is no more than
  `0.01 m`;
- IK corrections remain within the configured reachable shell and 0.35-radian
  correction bound;
- unreachable targets cause safe slowing/stopping rather than stretching or
  body lift; and
- disabling IK reproduces the support-retargeted Gate C baseline.

### Gate F: Scene switching and live use

- Repeatedly cycle through every scene without reloading the motion library,
  leaking a Raylib model, or retaining prior controller/contact state.
- A malformed-scene switch preserves the active scene and continues safely.
- UI scene ID, effective terrain weight, support level, source range, contacts,
  and blocked state agree with the CSV.
- The controller maintains its fixed 25 Hz target and closes through normal
  model/window cleanup.
- Leave the approved multiscene build open on `DISPLAY=:1` for user testing.

## Non-goals

- Full rigid-body simulation or general capsule-versus-mesh physics.
- Dynamic obstacles, jumping, climbing, hand contacts, or navigation/path
  planning around arbitrary objects.
- Guaranteed traversal of stairs or ramps beyond the validated motion/IK
  envelope.
- Generating new motion by relabeling existing clips against terrain they were
  not recorded on.
- Adding lateral terrain dimensions to the matcher without separate evidence
  and design review.

## Relationship to Existing Plans

This specification extends the approved G1 terrain design and supersedes two
narrow assumptions where they conflict:

- “recorded root height only” now means recorded pose height relative to its
  source support plus a world support-frame coordinate transform; it does not
  prohibit multilevel placement; and
- “do not change the simulation root” in the IK plan still prohibits IK from
  changing matching/root state, but the pre-IK support-retargeting stage may
  assign world Y to the rendered `Simulation` bone.

The recorded local pose, database features, selected frames, and costs remain
immutable. Support retargeting is part of the accepted IK-off runtime baseline,
not an IK correction.

## Implementation Order

1. Complete the existing deterministic Task 6 logger and reproduce the failure.
2. Implement exact GRAIL terrain sampling, G1HF/v2, scene schemas, procedural
   generators, surface-parity tests, and scene publication.
3. Add the support sidecar, rebuild the complete motion pack, and validate it.
4. Add transactional runtime scene loading, selection, and full reset.
5. Add contact-aware support retargeting and horizontal-only root
   adjustment/clamping, then pass multilevel Gate C.
6. Add the walkability grid and safe-stop guard, then pass Gate D.
7. Complete existing terrain matching Tasks 7--8 across the scene set.
8. Implement the G1 terrain IK plan extended with normals and swing clearance,
   then pass Gate E.
9. Run a broad whole-branch review, record deterministic comparison evidence,
   launch the approved build, and leave it open for user testing.
