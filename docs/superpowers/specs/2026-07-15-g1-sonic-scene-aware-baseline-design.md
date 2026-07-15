# G1 Terrain Motion Matching to GEAR-SONIC Baseline Design

**Date:** 2026-07-15

**Status:** Approved conversational design, awaiting written-spec review

**Base:** g1-terrain-motion-matching at c1f97c201228376b22a7a114dfd4c757090383b4

**Implementation branch:** g1-sonic-scene-aware-baseline

## Purpose

The first experiment asks one narrow question:

> Can the existing privileged, terrain-aware G1 motion matcher produce
> kinematic references that an unmodified GEAR-SONIC policy can track well
> enough to traverse curbs, ramps, and stairs in MuJoCo?

This is an idealized feasibility test. Motion matching may use the existing
privileged terrain representation, and simulation may pause while a complete
reference chunk is generated and validated. If this composition does not work
under those conditions, adding noisy depth, state-estimation error, network
delay, or hardware dynamics will not rescue it.

The experiment must distinguish four outcomes:

1. The integration contract is wrong.
2. The motion-matched kinematic reference is unsuitable.
3. The reference is suitable, but SONIC cannot track it in the scene.
4. The terrain-aware MM plus SONIC composition works and outperforms the
   terrain-blind control.

A manually steered video is useful as a demonstration, but it is not sufficient
evidence for the fourth outcome.

## Goals

- Run the existing 25 Hz terrain matcher as a persistent, headless reference
  generator.
- Convert its output into the official 50 Hz, 29-DoF SONIC joint-reference
  contract.
- Produce both a replayable SONIC reference directory and a ZMQ protocol-v1
  stream from one canonical buffer.
- Gate MuJoCo physics and the low-level control loop until the next complete
  chunk is valid and buffered.
- Keep motion matching open-loop for the first experiment: the MM virtual root
  is not corrected from the simulated robot state.
- Evaluate terrain-aware and terrain-blind MM with matched commands and initial
  conditions.
- Record enough reference, physical-state, contact, and timing data to identify
  the failing layer.
- Leave clean interfaces for later latency, depth, and imitation-learning
  experiments without implementing those experiments now.

## Non-goals

This branch does not implement:

- depth, height-map reconstruction, lidar, or other exteroceptive perception;
- measured root-position or root-height feedback;
- obstacle detection or obstacle planning;
- closed-loop correction of the MM virtual root from MuJoCo;
- SONIC retraining or fine-tuning;
- PFNN-style replacement models, behavior cloning, DAgger, or dataset
  aggregation;
- real-time deadlines, artificial delay, chunk-size sweeps, or action-chunking
  compensation;
- deployment to the physical G1.

Timing is measured in this branch, but no timing threshold determines the
scientific verdict.

## Selected integration approach

The system uses a canonical short-reference chunk with two sinks:

1. An official SONIC reference directory for deterministic preflight and
   replay.
2. A GEAR-SONIC ZMQ protocol-v1 publisher for interactive chunk streaming.

A file-only integration would be easy to inspect but would not support steering.
Embedding motion matching inside the GEAR controller would remove a process
boundary but would couple two large systems before feasibility is established.
The selected approach keeps each responsibility independently testable while
exercising the supported SONIC streaming boundary.

GEAR-SONIC is an external, pinned dependency. The initial external code revision
is NVlabs/GR00T-WholeBodyControl commit
60de0df7ffedeef415fe58d435e92cc5b01ba3d9. Its deployment documentation lists
ZMQ as a supported motion input, and its motion-reference documentation defines
50 Hz joint-position and joint-velocity files with 29 G1 joints:

- https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html
- https://nvlabs.github.io/GR00T-WholeBodyControl/references/motion_reference.html
- https://nvlabs.github.io/GR00T-WholeBodyControl/references/deployment_code.html

The SONIC checkpoint, observation configuration, G1 model, and external
repository revision are hashed into every run manifest. No model weights or
external repository source are copied into this repository.

## Architecture

### 1. Headless MM chunk server

The C++ chunk server owns:

- the current terrain database and sidecars;
- scene selection and privileged terrain queries;
- the existing 25 Hz query, search, transition, inertialization, support, and
  virtual-root state;
- command smoothing and trajectory prediction;
- conversion of the inertialized G1 pose into named joint coordinates;
- transactional candidate state.

It does not know about SONIC, ZMQ, MuJoCo stepping, CSV formatting, or operator
devices.

The server is a long-lived subprocess. It reads one versioned JSON object per
line from stdin and writes one versioned JSON object per line to stdout.
Human-readable diagnostics use stderr exclusively. The protocol supports:

- **hello**: return protocol, build, skeleton, and artifact identity;
- **reset**: select a scene, route, terrain weight, and initial matcher state;
- **generate**: latch one velocity and heading command and produce one
  candidate chunk;
- **commit**: make the candidate state active;
- **abort**: discard the candidate and retain the preceding active state;
- **close**: terminate cleanly.

Only one candidate may be outstanding. Generate never mutates the active state.
A second generate before commit or abort is a protocol error.

The existing runtime controller state is split into immutable resources, active
state, and candidate state. Candidate generation advances a deep state copy for
ten 25 Hz steps. Commit swaps that candidate into the active slot; abort destroys
it. This preserves the matcher's selected frame, transition offsets,
inertialization, command smoothing, contacts, support state, and virtual root
across accepted chunks while making validation failure reversible.

### 2. G1 joint projection

The motion database stores a hierarchy of local body rotations, while SONIC v1
expects 29 named joint coordinates. A focused projection module converts the
inertialized MM pose into the G1 hinge coordinates using a committed joint
contract derived from the source G1 MJCF:

- source bone name;
- source parent;
- MuJoCo joint name;
- joint axis;
- static parent-to-child rotation;
- sign and zero offset;
- joint limits;
- SONIC target name.

The initial SONIC reference uses the inertialized motion-matched pose before the
display-oriented foot-lock/contact IK pass. That IK can rotate local bones away
from their one-DoF joint manifolds; it is therefore neither silently projected
nor sent to SONIC. Privileged terrain matching, transitions, inertialization,
command adjustment, and support-level diagnostics remain active.

For every projected frame, the module:

1. removes the static parent-to-child rotation;
2. extracts the signed twist around the registered joint axis;
3. reconstructs the local body rotation;
4. rejects the frame if the off-axis residual exceeds 0.001 radians;
5. projects the local angular velocity onto the same registered axis;
6. rejects values outside the registered joint range or any non-finite value.

The physical pelvis global orientation, not the planar Simulation-bone heading,
is the SONIC root orientation. Physical pelvis position and the planar MM virtual
root are both retained as diagnostics.

The projection contract must round-trip source G1 qpos through MuJoCo forward
kinematics, the existing Z-up-to-Holden conversion, and joint extraction with:

- maximum joint-angle error no greater than 0.0001 radians;
- maximum reconstructed local-rotation error no greater than 0.0001 radians;
- exact joint-name coverage with no positional fallback.

### 3. SONIC bridge

The Python bridge owns:

- source-chunk schema validation;
- source-to-target joint-name mapping;
- Holden-to-MuJoCo coordinate conversion;
- 25 Hz to 50 Hz resampling;
- final reference validation;
- official reference-directory export;
- ZMQ protocol-v1 encoding and publishing;
- run-local reference recording.

All target indices come from a checked name manifest. No component assumes that
the MM database order, MuJoCo qpos order, actuator order, and IsaacLab/SONIC
order are equal.

The existing terrain artifacts use the coordinate signature
holden-y-up-right-handed-forward-plus-z. Their inverse source conversion is:

    mujoco_x = holden_x
    mujoco_y = -holden_z
    mujoco_z = holden_y

The bridge applies the same proper rotation to pelvis orientation. Quaternion
arrays are normalized, hemisphere-unrolled, and emitted in wxyz order. Unit
basis vectors, a known pelvis pose, and the terrain mesh bounds are checked
before a run; a guessed sign or axis swap is a hard failure.

The official reference directory and the ZMQ message are serialized from the
same validated float32 target buffer. The directory contains 29-column
joint_pos.csv and joint_vel.csv files, root-only body_quat.csv, zero-valued
body_pos.csv for the position-untracked baseline, and root-only metadata.
Actual MM pelvis/root positions are stored in a separate diagnostic file and
are not supplied to SONIC.

Protocol v1 messages contain:

- joint_pos with shape [N, 29];
- joint_vel with shape [N, 29];
- root body_quat with shape [N, 4];
- contiguous, monotonically increasing frame_index values.

The bridge does not send root position, root height, or a terrain observation.

### 4. Experiment coordinator

The coordinator owns:

- scripted commands and manual operator input;
- session and chunk identifiers;
- generate, validate, enqueue, commit, and abort sequencing;
- the simulation/control gate;
- trial reset and initial perturbations;
- result collection and evaluation.

The baseline provides two execution modes:

- **preflight** writes a complete reference directory without starting SONIC;
- **stream** publishes accepted chunks through ZMQ and advances gated MuJoCo.

Operator velocity and desired heading are sampled and latched only at a chunk
boundary. The existing MM command prediction and smoothing run for every 25 Hz
step within the chunk. Later operator input may be queued, but it cannot mutate
an outstanding candidate.

The simulation gate controls the GEAR low-level process and MuJoCo runner as one
logical clock. Neither advances while a candidate is being generated or
validated. The coordinator validates and locally enqueues the complete chunk,
performs the ZMQ send while simulation is paused, commits the MM candidate after
that send returns successfully, and only then releases exactly 0.4 seconds of
simulated time. The gate stops earlier on a terminal failure and pauses again at
the boundary before another command is latched. The adapter changes simulator
orchestration only; it does not change the SONIC policy, observations, actions,
or network.

GEAR's ZMQ input is publish/subscribe and supplies no per-chunk receipt. A
one-time readiness preflight must establish the subscriber and observe the
initial reference frame before physics is released. Candidate commit occurs
after successful local validation, enqueue, and ZMQ send. The post-run SONIC
target-motion record must contain every expected frame index exactly once; a
missing, duplicated, or reordered frame invalidates the trial as an integration
failure.

### 5. Scene adapter

Motion matching and MuJoCo consume one registered scene definition. The scene
adapter reads the selected terrain scene's mesh and transform, converts the
Holden OBJ into the MuJoCo basis above, and emits a run-local MuJoCo scene
overlay. It does not independently reconstruct the geometry.

The run manifest records:

- MM scene and route identifiers;
- source terrain mesh and heightfield hashes;
- source and target coordinate signatures;
- the exact rigid transform;
- generated MuJoCo scene hash;
- allowed foot geoms and forbidden body-contact groups.

The MM virtual root follows its existing open-loop route and command state.
Simulated pelvis and base state are logged but are not fed back into MM.

## Temporal contract

The initial chunk duration is 0.4 seconds.

- Source rate: exactly 25 Hz.
- Source intervals per chunk: 10.
- Source boundary samples per chunk: 11.
- Target rate: exactly 50 Hz.
- Target intervals per chunk: 20.
- Target boundary samples before seam removal: 21.
- Newly emitted target frames per accepted chunk: 20.

At session start, the initial target boundary is emitted exactly once as frame
zero. The first chunk then emits frames 1 through 20, the second emits frames 21
through 40, and so on. The first boundary of a candidate must equal the final
accepted boundary of the preceding chunk; it is validated but not emitted
again.

Joint positions are interpolated with cubic Hermite interpolation using the
25 Hz joint position and velocity endpoints. The analytic Hermite derivative is
the 50 Hz joint velocity. Pelvis orientation uses normalized shortest-path
spherical interpolation. Interpolation never silently clips joint values.

The baseline chunk length is configurable only through an integer number of
25 Hz source intervals. The 0.4-second value remains fixed for the experiments
in this specification; other lengths belong to the later latency/action-chunk
study.

## Canonical chunk data

Every source candidate contains:

- schema version, session ID, candidate ID, and predecessor ID;
- source rate, interval count, and timestamps;
- latched requested and applied command;
- scene, route, terrain-weight, and transform identity;
- ordered source and target joint names;
- 11 frames of 29 joint positions and velocities;
- 11 physical pelvis positions and orientations;
- 11 planar MM virtual-root positions and orientations;
- selected database frame, search/transition flags, and terrain costs per step;
- privileged terrain samples and support diagnostics per step;
- source artifact hashes and executable commit.

Every accepted target chunk contains:

- schema version, session ID, accepted chunk ID, and source candidate ID;
- 20 new 50 Hz frame indices and timestamps;
- 20 frames of mapped joint position and velocity;
- 20 physical pelvis orientations;
- diagnostic pelvis and virtual-root trajectories;
- command, scene, route, and terrain condition;
- coordinate, joint-map, source-chunk, and target-buffer hashes;
- generation, validation, resampling, enqueue, and publication timings.

Unknown schema versions, duplicate keys, missing fields, extra joints, and
non-finite JSON numbers are rejected.

## Validation and failure behavior

Validation is transactional and fail-closed. Before commit, the system checks:

- exact schema, shape, rate, interval, and name contracts;
- finite and normalizable numeric values;
- registered joint coverage and joint limits;
- joint-projection residuals;
- quaternion unit norm and hemisphere continuity;
- exact source endpoints after resampling;
- equality of the candidate's first boundary and the preceding accepted
  boundary within 0.000001 for scalar/vector values and 0.000001 radians for
  orientation;
- exact target frame count and contiguous frame indices;
- source/scene/model hashes;
- decoded CSV and ZMQ values equal to the canonical float32 buffer;
- subscriber readiness before the first physics release.

On a generation, validation, enqueue, or publication failure before commit:

1. The coordinator records a structured rejection artifact.
2. It sends abort for the outstanding MM candidate when the server is alive.
3. The active MM state remains unchanged.
4. Simulation and the low-level controller remain paused.
5. No preceding chunk is repeated and no neutral fallback is inserted.
6. The current trial terminates with an integration-layer verdict.

A failure observed after commit or after physics release terminates the trial
without generating a successor chunk. The accepted MM state remains committed
for faithful diagnostics; the trial is never resumed from an earlier state.

A process-health watchdog may terminate a dead or unresponsive subprocess, but
chunk-generation duration is not treated as a real-time failure in this
baseline.

## Experiment ladder

Later stages run only after earlier integration gates pass.

### Stage A: contract and reference preflight

1. Run the G1 joint-projection round trip over registered source clips.
2. Validate basis vectors, root orientation, joint limits, and scene alignment.
3. Generate a flat MM reference directory and replay it kinematically in the
   registered G1 MuJoCo model.
4. Run the official GEAR-SONIC quick-start reference in the bundled flat MuJoCo
   environment and register its target and actual-state tracking metrics.
5. Stream that same known-good reference through this branch's ZMQ publisher.

If file replay succeeds but the identical decoded stream fails, the verdict is
an integration failure and no MM hypothesis is evaluated.

### Stage B: flat MM plus SONIC

Use the flat scene and terrain weight 0.0. The deterministic 12-second command
script is:

- 2.0 seconds standing;
- 4.0 seconds forward at 0.5 m/s;
- 4.0 seconds forward at 0.5 m/s while the desired heading changes by 45
  degrees;
- 2.0 seconds standing.

The trial passes when:

- the command script and all expected frames complete;
- no hand, knee, torso, or pelvis geom contacts the environment;
- pelvis height above the flat surface remains at least 0.45 metres and the
  pelvis up vector's dot product with world up remains at least 0.5;
- joint-position RMSE and pelvis-orientation error are each no greater than
  1.5 times the corresponding known-good SONIC reference metric;
- no chunk boundary creates a validation or reference-delivery failure.

Manual flat steering is enabled only after this scripted gate passes.

### Stage C: terrain-aware causal test

The first three terrain classes and existing route IDs are:

1. grail-curb-low with curb-forward;
2. ramp-10-up-down with up-landing-down;
3. stairs-shallow with ascent-landing-descent.

For each class, the existing route waypoints are compiled once into a
deterministic sequence of 0.4-second command chunks. That frozen sequence and
its duration are replayed unchanged in every condition and perturbation. The
target region is the horizontal circle of radius 0.25 metres around the route's
final waypoint. Run two matched conditions:

- **aware:** privileged terrain feature weight 4.0;
- **blind:** terrain feature weight 0.0.

All other matcher, bridge, policy, scene, and simulator settings are identical.
The nominal run is a smoke test. The scored matrix contains ten fixed physical
initial perturbations relative to the unchanged MM reference:

- lateral offsets of +0.03, -0.03, +0.06, and -0.06 metres;
- yaw offsets of +2, -2, +4, and -4 degrees;
- combined offsets of (+0.03 metres, +2 degrees) and
  (-0.03 metres, -2 degrees).

A scored trial succeeds when:

- the pelvis reaches the route target region within 1.25 times the nominal
  scripted duration;
- pelvis height above local terrain remains at least 0.45 metres;
- the pelvis up vector's dot product with world up remains at least 0.5;
- no hand, knee, torso, or pelvis geom contacts the environment;
- every expected target frame is observed exactly once.

Foot contacts are allowed. Swing-foot scuffs, minimum foot clearance, path
drift, joint tracking, pelvis tracking, and contact impulses are recorded as
secondary metrics.

For each condition, also replay the MM reference kinematically against the same
scene. Reference penetration by any forbidden body geom greater than 0.005
metres is an MM-reference defect. The diagnostic does not suppress the safe
dynamic MuJoCo trial; it determines failure attribution.

The naive scene-aware composition hypothesis is supported only when, for every
terrain class:

- the aware condition succeeds on at least 8 of 10 scored trials; and
- the aware condition records at least three more successes than the blind
  condition.

If both conditions succeed on at least 8 of 10 trials, composition feasibility
is supported but the scene-awareness effect is inconclusive; a harder,
separately designed terrain tier is required before making a causal claim.

### Failure attribution

- Known-good reference fails in file and stream modes: GEAR setup, model,
  simulator, or general harness failure.
- File reference succeeds but decoded ZMQ stream fails: bridge/protocol failure.
- Flat MM kinematic replay fails: joint or coordinate conversion failure.
- Flat MM replay succeeds but dynamic tracking fails: reference distribution or
  SONIC tracking failure.
- Terrain MM kinematic replay has forbidden collisions: motion-matching
  reference failure.
- Terrain kinematic replay succeeds but SONIC falls: low-level
  tracking/contact-domain mismatch.
- Aware and blind both pass: terrain is not discriminative enough.
- Aware passes and blind fails by the registered margin: evidence for the
  terrain-aware MM plus unmodified SONIC hypothesis.

Manual scene steering follows the deterministic matrix and is reported as a
demonstration, not included in the scored verdict.

## Metrics and run artifacts

Each run writes to a new immutable run directory. The directory contains:

- experiment and environment manifest;
- MM, GEAR-SONIC, and adapter commit identities;
- checkpoint, observation-config, model, motion-pack, joint-map, and scene
  hashes;
- command stream and initial-condition record;
- raw MM candidate and accept/abort JSONL logs;
- canonical 25 Hz and 50 Hz reference arrays;
- official SONIC reference directory;
- transmitted and decoded ZMQ payload record;
- SONIC target-motion log;
- simulated joint, base, contact, and action logs;
- kinematic collision report;
- timing samples and p50, p95, p99 summaries;
- video for diagnostic runs;
- machine-readable per-trial verdict and aggregate hypothesis verdict.

Timing covers MM generation, projection, validation, resampling, enqueue,
publication, SONIC policy execution when exposed, and simulated-time advance.
These measurements inform the later latency design but do not alter the current
verdict.

## Repository and dependency isolation

Work occurs in the ignored worktree:

    /home/ubuntu/projects/motion-matching/.worktrees/g1-sonic-scene-aware-baseline

The branch was created directly from the committed terrain branch. Existing
modified database/features binaries, logs, executables, videos, and untracked
terrain artifacts in the primary terrain workspace are not copied, modified,
cleaned, or committed.

New code is isolated below a focused sonic directory so the root wildcard
controller build does not acquire a second main function:

    sonic/
      cpp/
      python/mm_sonic/
      schemas/
      configs/

Tests follow the existing tests/cpp and tests/python structure. Generated
references, converted scenes, logs, videos, checkpoints, and run directories
remain ignored artifacts.

The implementation accepts explicit read-only paths for:

- the terrain motion pack and scene artifacts;
- the source G1 MJCF used to certify joint projection;
- the external GEAR-SONIC checkout;
- the SONIC checkpoint and observation configuration.

Preflight hashes every input before launching a trial. It never writes through
those input paths. Scene overlays, reference files, and logs are written only
inside the isolated branch's ignored run area.

## Test strategy

### C++ unit tests

- JSONL request and response validation.
- Exact 10-step candidate generation.
- One-outstanding-candidate state machine.
- Commit preserves candidate state.
- Abort leaves active state unchanged.
- Failed generation leaves active state unchanged.
- Deterministic chunks for identical initial state, command, and artifacts.
- Joint projection, range checks, axis residuals, and source-qpos round trip.
- Initial and successor boundary ownership.

### Python unit tests

- Exact joint-name mapping and rejection of duplicates or omissions.
- Holden-to-MuJoCo basis and quaternion conversion.
- Hermite values and analytic derivatives at endpoints.
- Shortest-path quaternion interpolation.
- 11-source-boundary to 20-new-frame chunk conversion.
- Cross-chunk boundary and frame-index continuity.
- Joint-limit, quaternion, shape, and finite-value rejection.
- Official reference-directory round trip.
- ZMQ v1 encode/decode loopback.
- Decoded CSV and ZMQ float32 parity with the canonical buffer.
- Scene mesh transform and hash registration.
- Metric and verdict calculations at threshold boundaries.

### Integration tests

- Start the MM subprocess, generate, abort, regenerate, and compare candidates.
- Generate and replay a complete flat reference without SONIC.
- Stream the known-good reference through a loopback subscriber.
- Run the known-good reference through GEAR-SONIC and MuJoCo.
- Run the scripted flat MM gate.
- Run nominal curb, ramp, and stair smoke tests.
- Run the matched aware/blind perturbation matrix.

GPU and MuJoCo policy tests are explicit experiment commands rather than
ordinary CPU unit tests. Their output manifests and verdicts are retained as
evidence.

## Extension seams

This baseline deliberately creates three future replacement points:

- A depth or lidar scene provider can replace privileged terrain queries while
  retaining the MM chunk output contract.
- A latency experiment can vary chunk interval count, generation delay, and
  simulation-gate policy without changing reference semantics.
- A learned teacher/student or DAgger-style model can consume the recorded
  command, target kinematics, proprioception, depth, and actual kinematics, then
  produce the same canonical reference contract.

Those extensions require separate designs and verdicts. They must not be added
to this branch until the privileged vanilla-composition result is known.
