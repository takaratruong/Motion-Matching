# G1 Learned Motion Matching Terrain Canary Design

Date: 2026-08-09

Status: approved for implementation

## Objective

Adapt Orange Duck's complete Learned Motion Matching (LMM) pipeline to the
Unitree G1 morphology using an offline-retargeted, walking-only terrain corpus.
The first deliverable is a small 60 Hz canary that proves the retargeted data,
learned pose manifold, learned dynamics, learned projector, and interactive
runtime before the corpus is expanded.

The implementation uses Orange Duck's decompressor, stepper, projector,
network binary format, and runtime orchestration. It does not continue the
custom PFNN as the primary locomotion model. Existing ordinary G1 Motion
Matching is migrated to the same 60 Hz data and controller contract and remains
the operator-selectable baseline and fallback. Its exact database
nearest-neighbor calculation is the learned lane's data oracle.

## Scope

The canary contains exactly three walking motion classes:

1. flat walking;
2. continuous uphill/downhill walking on a smooth slope; and
3. stair walking.

The canary excludes running, sprinting, dancing, fighting, falling, crawling,
gestures, manipulation, and physics-control training. All motions are
kinematic G1 animation at 60 Hz.

The data may come from:

- Orange Duck's 60 Hz re-solved LAFAN1 walking/obstacle motions;
- existing retargeted released-PFNN walking motions;
- Takara G1 walking; and
- GRAIL G1 slopes or stairs.

Each selected source must pass the same provenance, continuity, kinematics,
and terrain-registration checks. No source receives an assumed frame rate.
The loader reads the authoritative source rate and resamples to 60 Hz.
When an original higher-rate source exists, the builder must use it rather than
upsampling a convenience 25 Hz derivative. A source available only at 25 Hz may
still be interpolated to the 60 Hz runtime grid, but the manifest identifies
that native limitation and no gate describes the added samples as new motion
information.

All three canary motions use one neutral walking style with compatible arm,
hand, and torso behavior. The canary has no style tag, so visibly different
upper-body styles cannot be mixed and left for the projector to average.

LAFAN1-derived data remain subject to their noncommercial dataset terms. A
commercial deliverable must replace those sources with appropriately licensed
motion before promotion.

## Why LMM Is the Primary Path

The failed PFNN transfer attempted to reconstruct the next G1 joint pose from
body positions, velocities, phase, trajectory, and terrain. Several G1 twist
and leaf joints are not observable in those inputs, and retargeted targets also
contained branch discontinuities. The result could fit a small homogeneous
segment while producing poor articulation across a mixed corpus.

LMM addresses the principal observability problem differently:

- the compressor learns a latent state from full pose and velocity data;
- the decompressor receives features plus that latent state and reconstructs
  full local bone rotations and motion;
- the stepper advances both features and latent state; and
- the projector maps a control/terrain query onto the learned feature-latent
  manifold when a transition is required.

This does not excuse bad retargets. Joint branch discontinuities, invalid
hinge rotations, broken contacts, incorrect scale, or cross-clip windows must
still be rejected before training.

## Canonical Runtime and Data Rate

The complete pipeline runs at exactly 60 Hz:

- database samples: 60 Hz;
- feature horizons: time-derived for 60 Hz;
- decompressor integration `dt`: `1/60` seconds;
- stepper integration `dt`: `1/60` seconds;
- ordinary G1 Motion Matching controller update: 60 Hz;
- learned G1 Motion Matching controller update: 60 Hz;
- terrain, contact, and collision evaluation: 60 Hz; and
- evaluation and interactive replay: 60 Hz.

The G1 database builder, feature generator, ordinary matcher, LMM networks,
controller, and evaluator share this one rate receipt. The prior convenient
25 Hz G1 resampling and 8/17/25-frame horizons are retired for this artifact.
The 60 Hz G1 horizons are 20/40/60 frames. Rate-dependent APIs derive frame
indices from the validated artifact rate instead of changing the ABI of the
shipped legacy 23-bone resources. The ordinary and learned G1 lanes receive
the same timestamped command at the same frame and are compared frame by
frame as well as by wall-clock time.

Source data may have a different native rate. Vector quantities are
interpolated in time and quaternions are resampled with sign-continuous SLERP.
The manifest records source rate, source frame count, output frame count, and
three arrays for every output row: `left_source_index` int32,
`right_source_index` int32, and `source_alpha` float32. Interior samples use
the two bracketing source frames and `alpha` in `[0,1]`; an exact source sample
uses equal indices and zero alpha; a clamped endpoint uses the first or last
index twice and zero alpha. These arrays are both interpolation provenance and
the authoritative reconstruction map. A separate nearest-frame shortcut is
not accepted. Resampling must preserve duration to within one output frame.

## Canonical G1 Skeleton and Pose Contract

The learned database uses the existing canonical 31-bone hierarchy:

- bone 0: synthetic `Simulation` root;
- bones 1-30: canonical G1 bodies in the order and basis fixed by
  `g1_kinematic_contract.h`;
- two contact channels: left foot and right foot; and
- local rotations stored as normalized WXYZ quaternions.

The database stores, per frame:

- local bone positions `[31, 3]`;
- local bone rotations `[31, 4]`;
- local bone linear velocities `[31, 3]`;
- local bone angular velocities `[31, 3]`;
- the fixed parent array;
- the owning clip range; and
- two foot-contact values.

Non-root local translations describe the rigid G1 morphology, not deformable
bone lengths. A decoded pose whose non-root local translation differs from the
canonical G1 value by more than `1 mm` is invalid.

The offline retargeter first produces canonical G1 root pose plus native 29-DoF
joint state. MuJoCo FK then produces the database bone representation. An
independent FK reconstruction must match the source conversion within
`1e-5 m` before a clip is admitted.

Quaternion signs are unrolled before differentiation. Native joint coordinates
are unwrapped only where the joint topology permits it. Frames are split or
rejected when an adjacent target joint exceeds `6.75 rad/s`; at 60 Hz this is
`0.1125 rad/frame`. A clip range never spans a rejected interval.

## Feature Contract

The canary uses 31 matching features:

| Slice | Meaning |
| --- | --- |
| 0-2 | left-foot position in the current root frame |
| 3-5 | right-foot position in the current root frame |
| 6-8 | left-foot velocity in the current root frame |
| 9-11 | right-foot velocity in the current root frame |
| 12-14 | hip velocity in the current root frame |
| 15-20 | future planar root positions at 1/3, 2/3, and 1 second |
| 21-26 | future planar facing directions at the same horizons |
| 27-30 | root-relative terrain heights at 0.25, 0.50, 0.75, and 1.00 m |

At 60 Hz, the future trajectory frames are 20, 40, and 60. Feature generation
and runtime query generation share one definition and one normalization
receipt. `features.bin` contains normalized float32 rows. For a sampled source
row, tests first compare an independently generated raw runtime query with the
denormalized stored row, then normalize that query and compare it with
`features.bin`; both comparisons require maximum absolute error `<= 1e-6`.

The four terrain values are exogenous. They use raw, pre-walkability-mask scene
heights; walkability affects command limiting and diagnostics but never changes
the learned terrain definition. Raw root-relative heights are normalized with
the manifest-bound feature offset and scale before indices 27-30 are written
into a recurrent state. Drift metrics denormalize predicted values before
comparing them with the terrain.

Each committed LMM tick has this non-circular order:

1. sample raw terrain at the committed root, normalize it, and overwrite
   current recurrent indices 27-30;
2. construct the raw control query and normalize all 31 components with the
   data-manifest receipt. If projection is requested, evaluate the projector
   transactionally from that normalized query, overwrite projected feature
   indices 27-30 with the same normalized committed-root terrain, validate the
   projected feature and latent values, and atomically select that projected
   state as the single stepper input. Otherwise select the overwritten current
   recurrent state. A failed projection mutates nothing and is a hard gate
   failure;
3. advance the selected feature-latent state exactly once with the stepper;
4. run a noncommitting provisional decompressor evaluation from the same
   committed root to obtain a candidate planar root;
5. sample raw terrain at that candidate, normalize it, and overwrite indices
   27-30 in the stepped state;
6. run the final decompressor evaluation from the original committed root;
7. require final-versus-provisional root displacement `<= 1 mm`, yaw difference
   `<= 0.001 rad`, and final-root terrain re-sample error `<= 1e-4 m`; and
8. commit the final root, pose, features, and latent exactly once.

Any convergence failure rejects the LMM tick and fails an acceptance run. The
same terrain transform is used for selection, root registration, collision
evaluation, and rendering.

The first canary intentionally retains the existing four centerline heights.
It does not introduce the PFNN 36-height lateral grid. A richer height field is
a separately versioned follow-up only after this canary works.

The synthetic Simulation root remains planar, matching Orange Duck's runtime
contract. World vertical placement is the authenticated mapping from the
motion's authored terrain support to the selected scene terrain. That mapping
is part of the terrain registration contract, not an IK or display correction.
The evaluator reports both the raw terrain-local learned pose and its registered
world pose so registration cannot hide articulation or contact errors.

## Immutable Canary Artifacts

The builder publishes an ignored, replace-atomically immutable data bundle
with:

- `database.bin`;
- `features.bin`;
- `terrain_features.bin` or equivalent manifest-bound terrain rows;
- `manifest.json`;
- `validation.json`; and
- visual retarget inspection assets for the three clips.

The manifest binds:

- schema version and accepted status;
- exact 60 Hz output rate;
- skeleton names, parents, basis, and signature;
- feature names, indices, horizons, weights, offsets, scales, and signature;
- source paths/identities, source rates, source hashes, and the exact
  left-index/right-index/alpha resampling arrays;
- per-clip terrain identity and transform;
- per-range motion class, terrain class, and destination labels used by the
  projector oracle;
- the authenticated G1 MuJoCo model, composed terrain-scene/mesh hashes, and a
  resolved collision-geometry receipt containing every geom ID, owning body,
  type, local pose, size, collision mask, and allowed-sole classification;
- clip ranges and split identities;
- contact-generation parameters;
- every artifact path, size, and SHA-256; and
- the complete validation receipt.

Loading is fail-closed. A mismatched skeleton, rate, dimension, source hash,
terrain identity, range, or normalization receipt is rejected before a model
or runtime state is constructed.

Training publishes a second replace-atomically immutable model bundle with:

- `latent.bin`;
- `decompressor.bin`;
- `stepper.bin`;
- `projector.bin`;
- `training.json`;
- `evaluation.json`; and
- `manifest.json` referencing the exact data-manifest SHA-256.

The data bundle is never mutated to add learned files. Runtime accepts only a
data/model manifest pair whose reciprocal schema, dimensions, and data digest
match exactly.

Training windows are discovered independently inside each `[start, stop)`
range. No decompressor, stepper, or projector sample may cross a clip boundary
or a rejected interval.

## LMM Dimensions

The canary keeps Orange Duck's 32-dimensional latent state. With 31 features,
31 bones, and two contacts, the derived contracts are:

- feature-latent state: `31 + 32 = 63`;
- decompressor: `63 -> 512 -> 458`;
- stepper: `63 -> 512 -> 512 -> 63`;
- projector: `31 -> four 512-wide hidden layers -> 63`; and
- training-only compressor input: `908`, producing 32 latent values.

The decompressor output size is
`15 * (31 - 1) + 6 + 2 = 458`: local positions, 6D local rotations,
linear velocities, angular velocities, root linear/angular velocity, and two
contacts. Network dimensions are derived from validated artifacts and checked
against these formulas; they are not copied from the shipped 23-bone LAFAN
weights.

All shipped LAFAN learned binaries are incompatible and remain untouched.
The G1 canary produces new learned files in the model bundle, each bound to the
data-manifest digest and training receipt.

## Binary ABI

New G1 bundles use an explicit little-endian ABI. Array headers are unsigned
32-bit row/column or length fields; floating payloads are IEEE-754 float32;
parents and ranges are signed int32; contacts are uint8 values restricted to
`0` or `1`; and quaternions are WXYZ. Network matrix dimensions use unsigned
32-bit fields and matrices/vectors use little-endian float32 in Orange Duck's
layer order.

Every reader validates exact expected dimensions, finite payloads, normalized
quaternions, strictly positive finite active-feature scales, monotonic in-range
clip ranges, exact payload byte count, and EOF immediately after the final
field. Native-endian files, trailing bytes, truncation, nonfinite values, and
zero active scales are rejected. The original shipped resource binaries are
not rewritten.

## Training Sequence and Stop Gates

Training is deliberately staged so the first broken boundary stops the run.

### Gate 1: Offline retarget inspection

Render source and G1 retarget side by side on their exact terrain. Verify root
scale, facing, limb semantics, joint limits, contact timing, and terrain
registration. There is no IK, pose damping, post-retarget smoothing, or
display-only correction. The existing Savitzky-Golay extraction of planar
Simulation-root position and heading is permitted only with manifest-bound
window/order parameters and a reported raw-versus-filtered root delta; it does
not modify native G1 articulation.

Start with the single flat walking clip and obtain visual approval of that raw
retarget before converting the two terrain canary clips. All three clips must
then pass before neural training begins.

After those clips pass, build the shared 60 Hz data bundle and qualify the
ordinary G1 matcher before fitting a network. Its preregistered flat, slope,
and stair routes must use the same per-frame commands and terrain queries,
show offline/runtime feature parity within `1e-6`, commit every frame without
a hold or reset, and satisfy the Gate 5 continuity and collision bounds. A
broken 60 Hz ordinary control lane stops the canary; LMM is not trained against
an invalid oracle.

### Gate 2: Compressor/decompressor canary

Train only the latent autoencoder/decompressor on the three clips. Evaluate
both fitted frames and withheld contiguous blocks from every clip. The data
manifest binds every withheld `[start, stop)` range before training. Optimizer
batches exclude each withheld row and the complete derivative, feature, and
training-window halo that could read any withheld row. The manifest records
the exact halo width and the independently rediscovered fitted/withheld row
digests.

The following reconstruction gates are computed and must pass separately on
the fitted population and on every clip's withheld population:

- native joint MAE `<= 0.010 rad`;
- per-frame maximum-joint-error p95 `<= 0.050 rad`;
- maximum joint error `<= 0.100 rad`;
- maximum FK body-position error `<= 0.010 m`;
- maximum sole-position error `<= 0.010 m`;
- maximum non-root local-translation error `<= 0.001 m`;
- no joint-limit violation;
- no off-axis hinge residual above the kinematic audit tolerance;
- finite root velocity and angular velocity; and
- contact F1 `>= 0.95` for each foot on fitted frames.

The withheld populations use the same joint, FK, sole, translation, limit,
hinge, and finite-velocity thresholds above; their per-foot contact F1
threshold is `>= 0.90`. An aggregate fitted pass cannot mask a withheld block
failure, and one withheld clip cannot mask another.

Failure stops the experiment. The corpus is not expanded and the stepper or
projector is not trained until the articulation gate passes visually and
numerically.

### Gate 3: Stepper rollout

Train the stepper with range-safe windows. From exact database seeds, run
closed-loop rollouts of 2, 5, and 8 seconds on each clip. Later frames must
depend on prior predictions without detaching the recurrent path during
training.

The rollout must remain finite, remain on the learned walking manifold, retain
contact timing, and stay within the joint/root continuity envelopes. Against
the known continuation of each canary clip, the 8-second rollout must keep
native joint MAE `<= 0.050 rad`, planar root error `<= 0.10 m`, and per-foot
contact agreement `>= 90%`. Every committed frame must also satisfy the
interactive joint/root limits defined below.

The data manifest preregisters one seed per clip with at least 480 valid
in-range successors. Range clamping, terminal-row repetition, wraparound, or a
cross-range successor invalidates the test.

### Gate 4: Projector and transitions

Train the projector against the same database features and latents. Exercise
the following manifest-bound deterministic matrix: stopped-to-flat-walk,
flat-walk-to-stop, flat-to-uphill, uphill-to-flat, flat-to-downhill,
downhill-to-flat, flat-to-stair, and stair-to-flat. Every case fixes the source
row, raw command/query, normalized query, scene transform, intended destination
motion class, destination terrain identity, exact brute-force nearest database
target, and a 30-tick transition deadline. Each case runs for 120 ticks.

Oracle coverage is a prerequisite, not a learned-model result. Before training,
the unrestricted exact nearest row for each normalized query must carry the
intended destination class and terrain identity, and its normalized feature
RMSE must be no greater than the case's preregistered bound. Every case-specific
bound is stored before training and must itself be `<= 0.50`; a case outside
that support rejects the data canary instead of training against a misleading
nearest row.

All projector inputs, outputs, costs, nearest-neighbor searches, transition
distances, and thresholds use normalized feature space. In particular, the
cost is formed from `q_norm - projected_normalized_features`; raw queries are
never subtracted from normalized network output. Denormalization occurs only
for named physical diagnostics.

Against the exact brute-force nearest feature/latent target, projector output
must have normalized feature RMSE `<= 0.05`, normalized latent RMSE `<= 0.10`,
and maximum normalized component error `<= 0.50`. Every decoded transition
frame must also pass Gate 2 articulation and Gate 5 continuity/collision
bounds. The transition deadline is met only by the first committed LMM state
whose exact nearest database row has the intended destination class and terrain
identity, whose normalized feature RMSE to the active query is `<= 0.10`, and
whose decoded pose passes every applicable pose, continuity, contact, and
collision gate. Aggregation is all-cases-pass: any invalid projection, missed
deadline, hold, reset, rejection, ordinary-MM commit, or fallback frame fails
Gate 4.

### Gate 5: Interactive 60 Hz runtime

Enable the G1 LMM path as an explicit runtime mode; do not simply flip the
existing dormant constant. The LMM mode owns pose advancement and cannot also
commit an ordinary database pose in the same tick.

Run an A/B viewer with timestamped identical commands and terrain:

- A: rebuilt 60 Hz ordinary G1 Motion Matching;
- B: learned G1 Motion Matching.

Both lanes use the same 60 Hz clock, database rate, feature horizons, command
frame, and terrain transform, so their inputs are frame-aligned. Learned poses
are not expected to be bitwise identical to ordinary-MM poses.
During the LMM acceptance lane, every route frame must be committed by LMM.
Engine ownership is logged per frame; LMM commits must equal route frames and
hold, reset, rejected-pose, ordinary-MM-commit, and automatic-fallback counters
must all equal zero. Ordinary MM is available only through an explicit operator
mode switch outside the acceptance run.

The evaluator and the agent perform a real interactive W/A/S/D drive over the
flat, hill, and stair canary. Acceptance requires:

- responsive start, stop, turn, and reverse behavior;
- complete smooth-hill ascent/descent when approached from both sides, plus
  the stair clip's authored traversal direction and exit;
- no freeze, hold loop, fall, leg crossing, or pose explosion;
- complete-foot penetration `<= 5 mm`;
- forbidden-body penetration `= 0`;
- planted-foot drift `<= 10 mm`;
- maximum native joint step `<= 0.25 rad`;
- maximum root translation `<= 60 mm/frame`;
- maximum root rotation `<= 0.35 rad/frame`;
- maximum denormalized pre-overwrite terrain prediction error `<= 0.020 m`;
  and
- learned output visually no worse than the ordinary-MM control on the same
  canary route.

No hidden IK, damping, joint clamping, smoothing, root tilt removal, or
display-only pose correction may make a failed learned output appear accepted.
The only world-space adjustment in the canary is the manifest-bound terrain
registration described above. If bounded IK is later needed, it receives a
separate design and its raw-versus-corrected effect remains visible.

Gate 5 uses one deterministic 1,200-tick (20-second) route in each traversal
direction plus a separate interactive drive. Complete-foot penetration is the
maximum of `max(0, terrain_y - probe_y)` over both feet, every frame, and the
four canonical kinematic sole points in each ankle frame: heel-medial
`(-0.05,-0.05,-0.025)`, heel-lateral `(-0.05,-0.05,+0.025)`, toe-medial
`(+0.12,-0.05,-0.030)`, and toe-lateral `(+0.12,-0.05,+0.030)` metres.

A planted interval begins after three consecutive frames whose decoded contact
value is at least `0.5` and ends on the first frame below `0.5`. Planted-foot
drift is the maximum terrain-tangent displacement of the four-probe centroid
from its stance-onset position. These points are probes, not collision geoms.

Forbidden-body penetration uses deterministic MuJoCo collision evaluation,
not sampled proxy geometry. The evaluator loads the authenticated G1 MJCF and
exact registered terrain mesh into the manifest-bound composed scene. The
allowed terrain-contact set is exactly the eight resolved ankle-descendant
collision spheres, four per foot. Every other enabled robot collision geom is
forbidden from contacting terrain, and every self-contact not disabled by the
authenticated scene is forbidden. At every committed pose the evaluator sets
the native G1 root and 29 DoF state, runs `mj_forward` (including its
deterministic `mj_collision` stage), and audits every resulting contact.
Forbidden penetration is `max(0, -contact.dist)` across forbidden contacts.
Acceptance requires no
forbidden contact deeper than the `1e-6 m` numerical tolerance; the reported
gate remains `0` after that tolerance. The manifest binds model/mesh hashes and
the complete resolved geom receipt, so an ID or collision-mask change fails
before evaluation. Every reported maximum aggregates both sides and every
route frame.

## Runtime Integration Boundaries

The current controller deliberately keeps LMM dormant because its bundled
networks are for the original 23-bone LAFAN character. The G1 integration must:

- rebuild and load both ordinary-MM and LMM G1 data at exactly 60 Hz, rejecting
  stale 25 Hz G1 artifacts before controller state is constructed;
- derive G1 trajectory horizons, velocities, contacts, and continuity limits
  from the validated 60 Hz rate while leaving legacy 23-bone asset ABIs alone;
- load only manifest-compatible G1 learned artifacts;
- allocate network evaluation buffers from validated layer sizes;
- keep feature/latent state in resettable G1 controller state;
- initialize state from a valid database row and matching latent row;
- ensure exactly one locomotion engine advances a tick;
- retain ordinary MM as a selectable fallback and A/B oracle;
- reject unsafe projected poses before commit; and
- preserve terrain, support, collision, and diagnostic instrumentation.

Unsafe projection is transactional: it mutates no root, pose, feature, latent,
contact, or terrain state. Outside evaluation an operator may explicitly switch
to ordinary MM. During every LMM gate, rejection is a hard failure and never
causes an automatic ordinary-MM commit.

A projected learned pose has no exact database frame identity. Runtime logging
and safety code must represent that explicitly rather than inventing a frame
index.

## Testing Strategy

Unit and contract tests cover:

- source-rate discovery and 60 Hz duration preservation;
- exact left/right/alpha resampling provenance and endpoint rules;
- quaternion sign continuity and SLERP;
- G1 skeleton/basis/signature validation;
- joint discontinuity splitting;
- range-safe window discovery;
- exact database and features binary round trips;
- exact offline/runtime 31-feature parity;
- 20/40/60-frame G1 trajectory horizons and rejection of stale 8/17/25 G1
  horizons;
- identical 60 Hz command, terrain, and query inputs for ordinary and learned
  G1 lanes;
- exogenous terrain overwrite ordering;
- transactional projected-state selection before the single stepper advance;
- normalized-only projector costs and oracle distances;
- manifest-bound MuJoCo collision-model and geom-receipt validation;
- network dimension formulas and binary validation;
- no double-advance between ordinary MM and LMM;
- model/manifest tamper rejection before state construction; and
- deterministic seeded training/reload for a tiny synthetic corpus.

Integration tests cover the five stop gates above. Generated motion, terrain,
networks, logs, and screenshots are ignored artifacts with hashes recorded in
the task report; they are not committed as source.

## Expansion After Canary Acceptance

Only after the complete three-clip interactive canary passes may the corpus
expand. Expansion remains walking-only and proceeds in this order:

1. additional flat starts, stops, speeds, and turns;
2. additional continuous slopes and cross-slope headings;
3. additional GRAIL curbs and stairs; and
4. continuity-clean, terrain-paired released-PFNN walking cycles.

Every expansion rebuilds a new immutable database and retrains all learned
artifacts. Evaluation uses clip-identity-disjoint validation terrain and never
promotes a model solely because mean training loss improved.

## Non-Goals

This canary does not provide:

- running or non-locomotion actions;
- a torque controller or real-robot stability guarantee;
- online retargeting;
- procedural terrain-motion generation outside the trained manifold;
- PFNN compatibility;
- a 36-sample lateral height field;
- automatic correction that hides raw learned failures; or
- commercial rights to third-party motion data.

## Completion Definition

The canary is complete only when:

1. all three retargeted clips pass raw visual and mechanical inspection;
2. the decompressor passes the articulation gates;
3. stepper rollouts pass for 8 seconds;
4. the projector performs valid terrain transitions;
5. the 60 Hz LMM runtime is controllable through the same user interaction as
   the ordinary matcher;
6. the automated terrain and continuity gates pass; and
7. the agent has personally driven the viewer over the canary terrain and
   visually confirmed that the complete body motion, not only the root, is
   coherent.

Until all seven conditions hold, ordinary G1 Motion Matching remains the
working fallback and the learned model is not called complete.
