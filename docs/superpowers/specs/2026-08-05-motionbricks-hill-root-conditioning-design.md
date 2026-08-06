# MotionBricks Gentle-Hill Root Conditioning Design

Date: 2026-08-05

## Summary

Test whether the released G1 MotionBricks model can generate coherent,
operator-controlled kinematics over a smooth hill when terrain elevation is
supplied as a root target before inference. The first experiment is
deliberately narrow: one fixed gentle hill in the existing browser two-stick
viewer, with no portals, post-generation root warping, inverse kinematics,
retraining, or physics.

MotionBricks remains the motion authority. The experiment changes only the
vertical positions of its sparse future root constraints. The generated root,
orientation, and joint trajectories are evaluated and rendered without
terrain correction.

## User Outcome

The first result is a controllable browser prototype:

- the existing two-stick/WASD interface controls MotionBricks continuously;
- one fixed smooth hill replaces the flat-only scene;
- the operator may approach, cross, turn on, and leave the hill; and
- a debug overlay shows sampled support height and conditioned/generated root
  height.

A deterministic A/B smoke test accompanies the viewer:

- control: the released MotionBricks flat controller follows a straight command
  over flat ground;
- treatment: the same controller, seed, initial context, and horizontal command
  follows a smooth 8--10 degree hill because its future root constraints carry
  the corresponding terrain-height changes.

The prototype and smoke test answer whether the released model responds
coherently enough to vertical root conditioning to justify foot IK, richer
maps, or physics work.

## Current Baseline

The branch's global terrain viewer uses official MotionBricks for flat
locomotion and swaps to precompiled terrain-course playback at a portal. Before
calling MotionBricks on a raised flat surface, it subtracts one constant support
height from the context and adds that same constant to a newly generated batch.
The flat-terrain guard stops the controller before a non-flat support change.

Inside the released MotionBricks agent:

- the latest four G1 poses provide context;
- movement direction, facing direction, mode, target speed, and seed provide
  control;
- a spring model constructs sparse future planar root and heading targets;
- a clip-selected target pose and the root targets condition latent
  in-betweening; and
- the generated `mujoco_qpos` frames are played kinematically.

The target transform code replaces only the two planar root components. Its
vertical root component still comes from a flat locomotion target clip. This
experiment extends that internal target transform at runtime without modifying
the external MotionBricks checkout.

## Selected Architecture

### 1. Fixed smooth hill

Add a deterministic exact-mesh hill scene with flat approach and exit regions.
The first treatment route is a smooth cosine-style longitudinal profile wide
enough that a straight traversal never reaches a lateral boundary. Its maximum
absolute grade is between 8 and 10 degrees, inside the approved 5--12 degree
range. It contains no height discontinuities, steps, curbs, holes, or
cross-slope component.

One mathematical profile owns both mesh construction and height sampling so
the conditioner, renderer, and evaluator cannot disagree about the surface.

### 2. Pre-inference root-height conditioner

Add a small Motion-Matching-side adapter around the loaded MotionBricks agent.
The adapter wraps the agent's target-transform stage. It does not fork or edit
the pinned external MotionBricks source.

On each real generation:

1. Let the unmodified MotionBricks spring compute the four target planar root
   positions and headings.
2. Transform those canonical planar targets back into world XY using the
   current canonicalization frame.
3. Sample the treatment terrain at the current root and at each target XY.
4. Compute each target elevation change:

   `delta_height[t] = H(target_xy[t]) - H(current_root_xy)`

5. Add `delta_height[t]` to the corresponding vertical component of
   `target_global_root_positions` after the ordinary target clip is selected
   and before the root and pose models run.

Adding the delta rather than replacing the target height preserves the
locomotion clip's ordinary pelvis-height oscillation. The existing constant
support canonicalization still removes the current support datum before
inference and restores it once to a newly generated batch afterward. The
generated change in height therefore comes from MotionBricks inference, not
from output projection.

The conditioner is batch-size-one and preview-checkpoint-specific. It validates
the expected private method, tensor ranks, target-frame count, coordinate
mapping, finite values, and device before installing. Any mismatch is a hard
error; it never falls back to output warping.

### 3. Raw generated output

The treatment uses MotionBricks' generated `mujoco_qpos` unchanged after the
ordinary one-time world support offset:

- no terrain-normal rotation;
- no root-height replacement or smoothing;
- no foot locking or IK;
- no joint projection;
- no collision repair; and
- no terrain portal or recorded terrain-course playback.

The root quaternion remains whatever MotionBricks generates. In particular,
the experiment does not align the robot root to the surface normal.

### 4. Interactive browser prototype

Add a hill-only mode to the existing MotionBricks browser viewer. This mode:

1. starts from a valid MotionBricks idle context on a flat apron;
2. keeps the existing two-stick/WASD command mapping and regeneration cadence;
3. disables terrain portal selection, portal playback, and the flat-terrain
   stop guard;
4. installs the pre-inference root-height conditioner for every real
   MotionBricks generation;
5. displays the fixed hill mesh, current trajectory, current terrain height,
   and latest conditioned target heights; and
6. keeps the session bounded to the hill mesh, stopping rather than sampling
   outside its certified height domain.

The first implementation may use a dedicated hill-viewer entry point when
that keeps portal behavior unchanged and reduces integration risk. It must
reuse the existing browser control and rendering components rather than create
a second input protocol.

### 5. Deterministic canary

Add a bounded command-line canary that:

1. loads the same pinned MotionBricks checkpoint for both arms;
2. fixes all random seeds;
3. starts from the same four-frame idle context;
4. applies the same straight forward command at the same speed and duration;
5. runs a flat control with zero elevation deltas;
6. resets the agent and runs the hill treatment with root conditioning;
7. records every raw `qpos`, generation boundary, planar spring target, sampled
   terrain height, conditioned root target, and elapsed time; and
8. writes immutable-by-convention artifacts under a caller-selected output
   directory.

The canary is the reproducible control for the interactive prototype rather
than its replacement.

## Artifacts

Each canary writes:

- `flat_motion.npz` and `hill_motion.npz`, using the existing stitched-motion
  arrays needed by the renderer;
- `flat.mp4` and `hill.mp4`, rendered against their exact meshes;
- `flat_trace.npz` and `hill_trace.npz`, containing conditioning diagnostics;
- `comparison.json`, containing configuration, provenance, metrics, and the
  verdict; and
- `root_height_comparison.png`, plotting generated root Z, terrain height,
  ground-relative root clearance, and conditioned target height over time.

The comparison records the Motion-Matching commit, external MotionBricks commit,
checkpoint paths or hashes, seed, command, terrain parameters, frame rate, and
duration.

The interactive viewer writes no evidence verdict. It may optionally record a
raw session trace for debugging, but only the deterministic canary publishes
`comparison.json`.

## Metrics

Metrics are descriptive for this first untrained probe:

- finite output fraction;
- total planar displacement;
- target terrain height range;
- generated root height range;
- correlation between generated root elevation and terrain elevation;
- root terrain-clearance median, range, and maximum frame-to-frame change;
- maximum root vertical velocity and acceleration;
- maximum root rotation and joint step;
- left and right sole terrain clearance from raw FK;
- maximum raw sole penetration; and
- flat/treatment command and seed parity.

Foot penetration is measured, not repaired. It cannot invalidate the execution
contract unless it is non-finite, because the user explicitly deferred IK and
wants to observe the released model's raw response.

## Verdict

The canary has three preregistered outcomes:

- `responsive`: execution is valid, the generated root clearly follows the
  hill's rise and fall, and visual review finds no catastrophic pose collapse
  or teleport;
- `unresponsive`: execution is valid, but the generated root does not
  materially follow the conditioned elevation; or
- `invalid`: the treatment was not actually delivered before inference, the
  A/B inputs differ outside the declared terrain condition, provenance is
  incomplete, output is non-finite, or required artifacts are missing.

For an automatic `responsive` candidate, require:

- all generated values finite;
- identical seed, initial context, horizontal command, and requested frame
  count between control and treatment;
- generated-root versus terrain-height Pearson correlation at least `0.80`
  over moving treatment frames;
- generated root elevation range at least `60%` of the traversed terrain
  elevation range; and
- no root translation step above `0.10 m`, root rotation step above `0.45 rad`,
  or joint step above `0.45 rad`.

These gates qualify only responsiveness and continuity. They do not establish
good foot contact, physical balance, tracking, or sim-to-real readiness.
Visual review is still required before proceeding.

## Error Handling

Fail before publishing a verdict when:

- the external MotionBricks checkout or checkpoint is unavailable;
- CUDA or a required renderer dependency is unavailable;
- the target-transform hook no longer matches the pinned preview API;
- canonical/world coordinate reconstruction fails;
- a height query is outside the hill domain or non-finite;
- a generation event returns an unexpected batch, frame, or `qpos` shape;
- the control and treatment inputs are not otherwise identical;
- an output array contains non-finite values; or
- rendering or metric generation omits a required artifact.

Write artifacts through temporary paths and publish `comparison.json` last so a
partial directory cannot look complete.

## Tests

Focused CPU tests cover:

- the analytic profile's flat aprons, continuity, and 8--10 degree grade bound;
- exact agreement between profile sampling and generated mesh vertices;
- canonical target XY to world XY reconstruction at zero and rotated headings;
- vertical delta construction for flat, uphill, crest, and downhill targets;
- preservation of the target clip's original vertical oscillation;
- conditioner installation and removal on a fake MotionBricks agent;
- no conditioning on cached batches;
- hard failure on private-API or tensor-shape drift;
- flat conditioning producing exactly zero target elevation deltas; and
- deterministic metric/verdict selection from synthetic traces.

The GPU integration test runs one short flat/treatment pair and verifies target
delivery, output shape, artifact provenance, and renderability. The full
traversal remains a manual, bounded canary because model quality is the subject
of the experiment.

## Non-goals

- Multiple hills, side slopes, arbitrary maps, or procedural terrain.
- Stairs, curbs, discontinuities, or portals.
- Post-generation root placement.
- Foot IK, locking, collision correction, or joint repair.
- MotionBricks retraining or checkpoint modification.
- Physics, SONIC tracking, actuator limits, or hardware deployment.
- A polished map, production UI, or quality guarantee from the first
  time-bounded prototype.

## Next Decision

If the canary is `responsive` and the interactive prototype is visually
coherent, the next design may add foot IK or a richer fixed heightfield while
retaining pre-inference root conditioning.

If it is `unresponsive`, inspect delivered target tensors and model outputs once
under the bounded diagnostic contract. If delivery is correct, stop rather than
adding output warping; the next decision is whether to add terrain-compatible
target poses or train terrain-conditioned MotionBricks.

If it is `invalid`, repair only the violated execution or provenance contract
and rerun the same canary.
