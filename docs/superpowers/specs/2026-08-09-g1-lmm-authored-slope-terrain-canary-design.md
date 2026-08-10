# G1 LMM Authored-Slope Terrain Canary Design

Date: 2026-08-09

Status: approved for test-first implementation

## Objective

Verify that the G1 Orange Duck Learned Motion Matching data path, learned
decompressor, learned recurrent stepper, and runtime terrain registration work
on real paired terrain. The accepted flat-forward clip has already proved the
G1 retarget, pose representation, decompressor capacity, and short recurrent
rollout. This canary adds exactly one native-G1 GRAIL slope traversal and asks
whether the learned system can reproduce it on its authenticated terrain.

This is an authenticated, support-calibrated terrain overfit and plumbing
canary. It is not a command coverage, unseen-terrain generalization, or
production locomotion claim.

## Decision and Alternatives

Three approaches were considered:

1. **One paired GRAIL slope plus the accepted flat clip — selected.** This is
   the smallest test with real G1 slope-adapted articulation, exact paired
   geometry under an explicit preregistered support-datum calibration, nonzero
   terrain inputs, and enough frames for a four-second recurrent rollout.
2. **Released-PFNN ascent plus its fitted/scaled heightmap — rejected for this
   gate.** The presently audited sample is only 60 output frames at 60 Hz and
   retains large stance-placement errors. It remains a visualization
   diagnostic, not learned-terrain evidence.
3. **A synthetic tilted plane — smoke test only.** It can catch coordinate,
   sign, sampling, and rendering defects, but flat motion rigidly placed on a
   plane is not terrain-conditioned learning.

The implementation runs the synthetic-plane smoke before building the real
bundle. Only the paired GRAIL experiment can pass this canary.

## Explicit Non-Goals

The canary does not attempt:

- broad W/A/S/D support, arbitrary turning, variable speed, or stopping;
- running, stairs, curbs, lateral foot-placement generalization, or physics
  control;
- held-out terrain or clip generalization;
- automatic ordinary-MM fallback, nearest-row substitution, pose clamping,
  IK, smoothing, or display-only correction; or
- promotion of the result as a generally controllable terrain model.

Interaction is deliberately authored-route playback: hold `W` to advance and
release it to pause. The viewer retains free camera control. Any broader input
is visibly rejected as outside this canary's scope.

## Frozen Sources

### Flat range

Reuse the accepted immutable `g1-lmm-flat-data/v3` bundle at
`sonic/runs/g1-lmm-flat-60hz/data-v3`:

- manifest SHA-256
  `5b5c48ccbb1dbabdbf87842d8b033c15b307199d72a8d90e4e39208ba5382db1`;
- 256 rows at exactly 60 Hz;
- terrain features exactly zero; and
- one continuity-safe `LocomotionFlat01_000[7659,8171)` range.

The new builder independently reloads and authenticates this source. It does
not copy unverified arrays from the old manifest.

### Slope range

Use the complete native-G1 GRAIL pair
`terrain_slopes__slope_000__000`:

- robot motion:
  `/home/ubuntu/datasets/GRAIL/data/slope/robot/terrain_slopes__slope_000__000.pkl`,
  SHA-256
  `b77480d5f8f3339a3064276d6f9d443ac3a3456f20eb9195d48add176e561ee1`;
- paired USD:
  `/home/ubuntu/datasets/GRAIL/data/slope/object_usd/terrain_slopes__slope_000__000.usd`,
  SHA-256
  `8d1e696fb5bd2aecfa17797549bddd001093b060773cb313185a6a46db7eb5a5`;
- reconstruction pose:
  `/home/ubuntu/datasets/GRAIL/data/slope/recon/terrain_slopes__slope_000__000.pkl`,
  SHA-256
  `d05d6c5a7d6a13eff7e69da0e9e96ab5a79f700bb706611699f0b9c44c9b51ec`;
- metadata SHA-256
  `7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d`,
  including `scene_scale=1.0`;
- 250 authoritative frames at 25 Hz, floor-to-source-span interpolation to
  598 provisional rows at 60 Hz. The source span is `9.96 s`, the output span
  is `9.95 s`, and the exact `0.010000000000001563 s` shortfall is bound in the
  receipt; and
- canonical G1 XML SHA-256
  `749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376`.

The motion is already native Unitree G1. Upsampling to 60 Hz adds no new motion
information; the manifest states that limitation explicitly.

The pair contains a flat approach and exit plus two complete traversals of a
10.107-degree, 0.127562 m-high ramp. The intended nonflat exposure is output
rows `[168,542)` in the provisional 598-row conversion. Exact composed-MuJoCo
preflight finds enabled wrist/hip self-contact only in provisional rows
`[0,3)`, so those three rows are rejected before derivative or feature fitting.
The admitted slope range has 595 rows; its intended nonflat exposure is
`[165,539)`. Rows outside that interval are labeled by measured terrain, not
by clip identity.

## Terrain and Registration Contract

The new immutable schema is `g1-lmm-terrain-data/v1`. It contains two source
ranges and 851 admitted rows: flat `[0,256)` and slope `[256,851)`. The slope
receipt separately preserves the complete 598-row conversion and the exact
`[0,3)` mechanical rejection. Windows never cross a source or rejected-
continuity boundary.

The four terrain features remain root-relative heights at 0.25, 0.50, 0.75,
and 1.00 m along the facing centerline. They are sampled from the authenticated
paired top surface after the explicit support calibration and before
normalization. The same calibrated heightfield, transform, and interpolation
policy are used by:

- offline feature generation;
- ordinary nearest-neighbor queries;
- LMM recurrent overwrites;
- root/support registration;
- collision evaluation; and
- rendering.

The legacy GRAIL alignment receipt is preserved as an authenticated provenance
cross-check:

- source center `82`;
- source root XY `[-0.15535800158977509, 0.03107992745935917]`;
- source yaw `-1.4274415969848633` rad;
- source support `0.012000000104308128` m; and
- 36-probe maximum alignment error
  `5.757581494902908e-09` m.

The anchor is approximately 31.08 mm outside the finite USD footprint. Its
historical `0.012 m` support value came from nearest-edge extrapolation, while
the production paired surface returns the explicit exterior-flat height of
zero there. The 36 relative-height probes therefore authenticate source
identity but do not establish an absolute support datum. The receipt's PFNN
seed yaw is likewise not reused to reorient this already-paired GRAIL motion.

The authoritative paired rigid transform is the reconstruction pose followed
by the same Z-up-to-Holden basis change applied to the robot. Separately, this
canary preregisters a relative `0.012000000104308128 m` support-datum
calibration between the GRAIL motion convention and the canonical G1 XML foot
spheres. It is applied to terrain only and is not described as rigid
registration. Define the native/global conversion first:

```text
B([x, y, z]) = [x, z, -y]

native_global_bone_position = B(source_global_bone_position)
runtime_mesh_vertex = B(R_object @ usd_vertex + t_object)

h_runtime(x, z) = h_reconstructed_Holden(x, z) - support_calibration
support_calibration = 0.012000000104308128 m
```

`R_object` and `t_object` are the first authenticated reconstruction pose and
their exact binary32 values are stored in the manifest. The robot and mesh use
the same `B`; no second yaw, axis swap, sign flip, or root recentering is
allowed. A viewer-only common rigid planar transform would be mathematically
equivalent, but this canary fixes it to identity to remove another degree of
freedom.

The Holden database then prepends the established `Simulation` bone. It is a
filtered planar controller frame: its Y coordinate is exactly zero and its
rotation contains heading yaw only. The native/global pelvis (`Hips`) height,
pitch, and roll are preserved in the `Hips` local transform beneath
`Simulation`; forward kinematics of `Simulation -> Hips` reconstructs the
native/global pelvis. Terrain centerlines and predicted-root resampling use
only `Simulation` XZ and heading. Planar root error, planar step, yaw step, and
fixed-point gates also mean `Simulation` XZ/yaw. Rendering, sole probes,
forbidden collision, pelvis motion, and FK gates use the complete global bone
transforms after forward kinematics. The same representation is used for both
flat and slope ranges. The authenticated source additionally requires global
`Hips` 3D step `<= 0.025 m/frame`.

The paired GRAIL global pelvis already follows the ramp through the `Hips`
local transform. Runtime therefore never adds a per-frame terrain height to
`Simulation` or `Hips`, never snaps either bone to the terrain, and never
shifts the authored pose vertically. The constant 12 mm correction is
applied once to the reconstructed terrain surface only. Applying it to both
motion and terrain would be a coherent common transform, would cancel the
relative calibration, and would reproduce the unaided 17.864829 mm sole
penetration rather than the calibrated 5.864829 mm result. The manifest and
viewer label this value `support_calibration_m`, never `registration_offset`.
The offline receipt requires float64 reconstruction/basis round trips within
`1e-9 m`; the C++ float32 transform requires the separately recorded `1e-6 m`
maximum. Exact reconstruction-transform, explicit exterior-flat, calibration,
and root/support-clearance parity are required before publication. Both
terrain-local and registered world poses are logged.

Offline features use raw exact-mesh heights. Runtime may use the rasterized
heightfield only after every manifest-bound route sample proves raw exact-mesh
versus runtime height parity within `1e-6 m`. For this smooth fully walkable
route, applying the walkability path must not change a terrain feature by more
than `1e-6 m`. A boundary or mask mismatch stops before training.

The finite USD contains the ramp surface but not every flat approach/exit
probe. Outside the upward-face XZ support of the reconstructed Holden mesh, the
scene uses one explicit paired flat continuation with pre-calibration height
`0.0 m` and gradient `[0,0]`; after calibration its runtime height is
`-0.012000000104308128 m`. The exact mesh always wins inside its domain. No
nearest-edge extension, unbounded triangle extrapolation, or implicit second
ground plane is allowed. The manifest records exact-mesh versus flat-extension
query counts for features, sole probes, and the authored route.

## Pre-Training Data Gates

The build stops before GPU work unless all of the following pass:

- exact 60 Hz output with complete left/right/alpha source provenance;
- 31 canonical G1 bones, 31 features, 32 latent dimensions, and two contacts;
- FK reconstruction error `<= 1e-5 m`;
- no joint-limit violation;
- maximum native/local joint step `<= 0.25 rad/frame`;
- maximum `Simulation` planar step `<= 0.025 m/frame`, global `Hips` 3D step
  `<= 0.025 m/frame`, and `Simulation` yaw step `<= 0.145834 rad/frame`;
- bilateral contacts with at least two meaningful runs per foot;
- all four terrain columns have standard deviation greater than `1e-6 m`,
  finite positive normalization scales, and no `FLT_MAX` placeholder;
- offline/runtime raw and normalized terrain-feature parity is `<= 1e-6`;
- all 374 intended nonflat route rows contain nonzero terrain vectors;
- range-safe 240-frame successors exist for the exact frozen rollout seeds;
- the authenticated source motion itself passes the source-applicable
  mechanical preflight defined below before any learned model is fit; and
- no training derivative or successor window crosses the dataset source
  boundary at global row `256` or a rejected continuity split. Windows are
  explicitly allowed and required to cross the continuous flat-approach,
  ramp-entry, grade, and ramp-exit transitions inside the GRAIL range.

Centered covariance rank, singular values, and flat-versus-slope normalized
separation are recorded as diagnostics, not acceptance gates. They describe
this ramp but are not prerequisites for an honest authored-route replay.

The audited slope values are frozen as regression expectations:

| Probe distance | Minimum (m) | Maximum (m) | Std. dev. (m) |
| --- | ---: | ---: | ---: |
| 0.25 m | -0.054865 | 0.053704 | 0.026962 |
| 0.50 m | -0.096942 | 0.095061 | 0.045899 |
| 0.75 m | -0.127296 | 0.127046 | 0.056156 |
| 1.00 m | -0.127296 | 0.127235 | 0.065241 |

Any material change requires a new schema/receipt and a fresh design review.

## Learned-Model Contract

The architecture stays Orange Duck compatible:

- compressor `908 -> 512 -> 512 -> 512 -> 32`;
- decompressor `63 -> 512 -> 458`;
- stepper `63 -> 512 -> 512 -> 63`.

Training uses the already verified physical Orange Duck objective and compiled
CUDA kernel. The canary is an all-row overfit capacity test. It makes no
held-out claim.

The complete sole-fit configuration is frozen now:

- base seed `1234`; 64-frame overfit seed `1234`, decompressor seed `1235`,
  and stepper seed `1236` for both PyTorch and NumPy generators;
- binary32 training rows, batch size `32`, exact `dt=1/60`, and stepper window
  `20`;
- constant learning rate `1e-3` with no scheduler;
- AdamW `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=1e-3`,
  `amsgrad=true`, `maximize=false`, `foreach=false`, `fused=false`,
  `capturable=false`, and `differentiable=false`;
- overfit budget `1,000`, decompressor budget `100,000`, and stepper budget
  `100,000` optimizer steps;
- deterministic PyTorch algorithms and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; and
- physical GPU 3, UUID
  `GPU-87fb0777-4169-d4e9-12cc-382bdf7c0730`, exposed as logical `cuda:0`.

No projector is trained, exported, or loaded for this canary: runtime begins
from an authenticated exact database feature and latent seed and advances the
authored route recurrently. This removes command coverage and synthetic-query
behavior from the terrain experiment. Architecture, physical objective,
thresholds, and the complete configuration above are bound into the model
receipt before the sole full run. If that physical GPU is unavailable or not
idle, the run waits; substituting hardware requires a new preregistration
before any model output is observed.

The model schema is `g1-lmm-terrain-model/v1` with:

- `model_scope=flat-plus-authored-slope-overfit-canary`;
- `evaluation_scope=exact-paired-slope-route-overfit`; and
- `generalization_claim=none`.

A numerically successful model manifest has `status=accepted` only within
`acceptance_scope=numerical-training-only`; it also carries
`artifact_role=canary_candidate` and `viewer_authorized=false`. It is immutable
and may be loaded only by the deterministic evidence mode until a separate
`g1-lmm-terrain-canary-evaluation/v1` receipt binds its manifest hash, data/XML
hashes, complete runtime trace/log hashes, evaluator-code hash, all runtime and
mechanical reductions, `status=accepted`, and `viewer_authorized=true`. A
rejected receipt remains immutable and cannot authorize the viewer. Normal
interactive startup fails before window creation unless that final accepted
receipt authenticates the exact data/model pair; training acceptance alone is
never presented as terrain-canary acceptance.

The C++ loader authenticates exactly `latent.bin`, `decompressor.bin`, and
`stepper.bin` plus the training/evaluation receipts; `projector.bin` is absent
and rejected for this schema. A canary-only transactional entry point,
`g1_runtime_step_lmm_authored`, requires exactly two allocated network
evaluations, clones all recurrent/pose/output state, initializes feature and
latent state from a manifest-bound database row, samples and overwrites the
current authoritative terrain, advances the stepper once, performs provisional
and final terrain-resampled decompression, applies finite/fixed-point/terrain/
continuity gates, emits the complete decoded global-bone candidate to the
evidence trace, and swaps the clone only on success. Exact MuJoCo geometry and
stateful planted-run reductions are acceptance-time offline gates over that
deterministic trace; the lightweight viewer does not claim to run MuJoCo before
each swap. It never invokes `projector_evaluate_normalized` and never allocates
a dummy projector. The model reload receipt binds this engine mode, exact
artifact set, seed rows, stepper count, LMM commit count, and zero
projector-evaluation count.

## Model Gates

### Decompressor

Evaluate all rows and require:

- native joint MAE `<= 0.010 rad`;
- per-frame maximum-joint-error p95 `<= 0.050 rad`;
- maximum joint error `<= 0.100 rad`;
- maximum FK and sole-position error `<= 0.010 m`;
- maximum non-root local-translation error `<= 0.001 m`;
- contact F1 `>= 0.95` for each foot; and
- finite root/local velocities with no limit or continuity failure.

### Stepper

Use these exact database-global seeds and half-open reference populations:

| Population | Seed | 60 ticks | 120 ticks | 240 ticks |
| --- | ---: | --- | --- | --- |
| flat control | `0` | `[0,61)` | `[0,121)` | `[0,241)` |
| first ascent (admitted slope-local `165`) | `421` | `[421,482)` | `[421,542)` | `[421,662)` |
| first descent (admitted slope-local `274`) | `530` | `[530,591)` | `[530,651)` | `[530,771)` |

The first row is the exact seed and each population contains the named number
of successor transitions. The slope populations cover ascent and descent and
remain inside the global slope range `[256,851)`. On every closed-loop tick,
terrain is resampled from the predicted root and heading before the final
decompression and commit. Require joint MAE `<= 0.050 rad`, planar root error
`<= 0.10 m`, per-foot contact agreement `>= 90%`, no terminal clamping, and
all runtime continuity bounds on every tick, separately for all nine rollout
populations.

A deterministic zeroed/shuffled-terrain replay may be recorded after the
accepted full-terrain run as a report-only sensitivity diagnostic. It cannot
fail or rescue the canary and is not called terrain causality: one paired clip
cannot disentangle terrain input from route/clip identity.

## Runtime and Visual Gate

The viewer adds an explicit `authored-slope` LMM canary route. Holding `W`
advances along the fixed registered route; releasing `W` pauses without
advancing learned state. Unsupported steering inputs reject visibly and mutate
nothing.

Reset initializes the complete pose/contact/output state from global database
row `256` (slope-local row `0`) and copies `features[256]` and `latent[256]`
into recurrent state. The displayed reset pose is row `256`; the next
successful `W` transaction commits successor row `257`. Acceptance is exactly
`594` advancing transactions through rows `257..850`, so the one lane includes
both ramp passes and the complete admitted approach/exit. At row `850`, further
`W` input reports `AUTHORED ROUTE COMPLETE` and returns before recurrent
evaluation; pose, recurrent state, contacts, counters, and cursor remain
bitwise unchanged. Release and unsupported controls have the same no-mutation
property. All runtime counters and maximum reductions cover reset through the
594 committed successors, while mechanical maxima exclude the reset pose only
when the named reducer is transition-specific.

The source contains complete ascent and descent intervals, so the full lane
runs and inspects both authored slope directions. Acceptance requires:

- every advancing frame committed by LMM;
- zero hold, reset, fallback, rejected-pose, or ordinary-MM commit while `W`
  is held;
- joint step `<= 0.25 rad/frame`;
- `Simulation` planar step `<= 0.025 m/frame`, global `Hips` 3D step
  `<= 0.025 m/frame`, and `Simulation` yaw step `<= 0.145834 rad/frame`;
- fixed-point root difference `<= 0.001 m`, yaw `<= 0.001 rad`, and terrain
  resample error `<= 1e-4 m`;
- complete-foot penetration `<= 0.006 m` after the explicit 12 mm authored
  support-datum calibration;
- forbidden-body penetration `= 0` after `1e-6 m` numerical tolerance;
- planted-foot terrain-tangent drift `<= 0.010 m`;
- planted-foot support gap `<= 0.01025 m`;
- denormalized pre-overwrite terrain prediction error `<= 0.020 m`; and
- visually coherent whole-body slope adaptation, contact timing, and root
  placement in source-versus-learned A/B inspection.

The runtime writes an authenticated little-endian evidence trace containing,
for reset and every committed successor, all 31 global bone positions and
quaternions after final decompression, contacts, `Simulation` XZ/yaw, terrain
channels, cursor, and ownership counters. A deterministic Python evaluator
authenticates the XML/terrain/data/model/trace hashes and applies the exact
mechanical gates before the model or viewer can be accepted. This is an
acceptance gate, not an in-transaction collision guard.

There is no lossy decoded-pose-to-hinge-qpos conversion. The evaluator forward
kinematics the decoded local transforms, maps each canonical Holden bone to its
authenticated XML body, and composes every XML collision geom's local transform
with that decoded global bone transform. This preserves decoder translations
and arbitrary rotations. For source rows, this direct-geom path must match
MuJoCo-forward geom transforms within `1e-5 m` and `1e-5 rad`. Forbidden
robot-terrain depth uses the existing exact sphere/capsule/box-to-surface
evaluator. Enabled self-contact is evaluated by placing those same decoded
world geom transforms into a temporary authenticated MuJoCo data instance and
running its collision filter/distance path; no hinge projection is permitted.
The identical direct-geom reduction is applied to source and learned traces.

Mechanical reductions are executable and identical for the source preflight
and learned runtime where applicable:

- complete-foot penetration is
  `max(0, h_runtime(probe_xz) - probe_y)`, maximized over every advancing tick
  and the authenticated XML's eight sphere-bottom sole probes. Each
  `left_ankle_roll_link` / `right_ankle_roll_link` body has four radius-0.02 m
  spheres at local centers `(-0.05,+/-0.025,-0.03)` and
  `(0.12,+/-0.03,-0.03)` m; the probe is the sphere's world-vertical bottom,
  not a synthetic point attached to a different joint;
- forbidden penetration is evaluated with the authenticated G1 MuJoCo model
  against the registered ramp mesh and its explicit exterior-flat extension.
  The eight sole probes use the exact native ankle-roll
  (`left_ankle_roll_link` / `right_ankle_roll_link`, mapped to the Holden
  `LeftToe` / `RightToe` bones) body transforms and the transformed MJCF
  sphere-bottom geometry; they must not be attached to the ankle-pitch bodies.
  Only those eight named support spheres may contact terrain; any other
  robot-terrain contact or enabled self-contact deeper than `1e-6 m` is
  forbidden. This is the gate that rejects provisional source rows `[0,3)`;
- a candidate planted run starts on the first Orange Duck contact tick and is
  admitted only after it lasts at least seven consecutive 60 Hz ticks; once
  admitted, its anchor remains that original first tick. It ends on its first
  value `< 0.5`. Drift is the maximum displacement, over the complete admitted
  run, of the four-probe sole centroid from its run-start anchor after
  projecting the displacement into the local terrain tangent plane; the
  reported gate is the maximum over both feet and all runs;
- planted support gap is evaluated over the same runs and four probes. For
  each planted tick it is
  `min_probe(max(probe_y - h_runtime(probe_xz), 0))`; penetrating probes
  contribute zero rather than being discarded. The gate is the maximum over
  all planted ticks and both feet. The authenticated source maximum is
  `0.00021573805692624848 m` in the authoritative float64 evaluator and must
  remain `<= 0.00025 m` in data preflight. The serialized binary32 diagnostic
  is separately frozen as `0.0002157502790678112 m`;
  Learned runtime allows `0.01025 m`: the frozen source ceiling plus the
  independently gated `0.010 m` maximum sole-position reconstruction error;
- pre-overwrite terrain error is the maximum absolute error over all four
  denormalized stepper terrain channels and every advancing runtime tick,
  measured immediately before authoritative terrain resampling; and
- fixed-point and terrain-resample gates use maxima over every advancing tick,
  never averages or p95 values.

Source preflight applies only joint/root continuity, sole penetration, planted
support gap, planted tangent drift, forbidden robot-terrain contacts, and
enabled self-contact. LMM ownership/commit counts, fixed-point convergence,
stepper terrain-prediction error, decoded-pose comparison, and visual A/B are
learned-runtime gates and are explicitly excluded from source preflight.

The 6 mm probe gate is frozen before training from the authenticated float64
source audit (`0.0058648290435704235 m` maximum) after the explicit relative
support calibration. The binary32 parity value is
`0.0058648294757275045 m`. The unaided paired surface measures
`0.017864829147878552 m`; all values are recorded so the calibration cannot be
mistaken for a raw-registration result. This is a canary bound, not a
production terrain collision guarantee.

The authenticated native-sphere source audit measures maximum seven-tick
planted-run terrain-tangent drift `0.009003773199 m` under this exact
onset/anchor rule, so the 10 mm gate remains feasible and frozen. Its
authoritative float64 / serialized binary32 support-gap values are the two
separately named values above. The rejected prefix does not contain either
maximum.

## Stop Rules and Completion

There is exactly one full model fit after data/runtime preflight freezes all
hashes. Any preflight, numerical, reload, runtime, collision, or visual failure
rejects the artifact and stops. Do not change thresholds, seed, budget, source,
support offset, or scene transform after observing model output.

The canary is complete only when:

1. the two-range terrain data bundle passes every pre-training gate;
2. the decompressor and stepper gates pass for the exact flat, ascent, and
   descent populations;
3. an immutable data/model pair reloads safely in C++;
4. the exact registered slope route passes the mechanical/runtime gates; and
5. the agent personally drives and visually confirms the learned whole-body
   motion on the slope before leaving the viewer ready for the user.

Passing means only that G1 LMM can learn and replay this authenticated paired
terrain under the frozen support-datum calibration. It does not establish
arbitrary control or terrain generalization.
