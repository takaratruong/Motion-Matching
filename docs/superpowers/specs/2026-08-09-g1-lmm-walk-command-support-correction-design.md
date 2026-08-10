# G1 LMM Walk Command-Support Correction Design

> **Status: paused on 2026-08-09.** The user explicitly prioritized terrain
> verification over command coverage. Do not execute this design unless that
> priority changes. The active design is
> `2026-08-09-g1-lmm-authored-slope-terrain-canary-design.md`.

## Purpose

The accepted 60 Hz G1 retarget/build pipeline, learned pose decoder, and learned
stepper have passed on the 256-frame `LocomotionFlat01_000` walk-only canary.
The projector rejection cannot be corrected honestly by increasing its budget
or relaxing its thresholds. The real 60 Hz controller query audit showed that
the single curved source clip contains no support for steady user-directed
walking commands. This correction expands only the flat walking canary enough
to support basic cardinal control, then binds the projector and runtime to the
same measured command population.

This is still a canary. It does not claim terrain, running, strafe animation,
backward-walk animation, broad held-out generalization, or production corpus
coverage.

## Root-cause evidence

- The rejected decoder passes every all-row articulation/contact gate.
- The rejected stepper passes deterministic 1, 2, and 4 second source rollouts.
- The projector's old 512-query gate injects isotropic noise into every feature,
  including four terrain features that are always zero at flat runtime.
- Only 130 of those 512 synthetic queries meet the design's prerequisite
  oracle-support RMSE bound of `0.50`.
- The published Orange Duck projector also fails the old synthetic gate under
  the same `.05/.10/.50` thresholds, so that population is not a demonstrated
  reference acceptance test.
- An unchanged ordinary 60 Hz controller trace on the current 256-row database
  has zero ticks satisfying the strict command-support envelope. The mismatch
  is concentrated in future trajectory position/direction features. The clip
  is a curved authored path rather than steady commanded locomotion.

Therefore the correction is data support plus a shared query contract, not a
projector threshold change.

## Scope

### Included motion classes

The expanded flat canary contains only:

- `stand`;
- `walk-start`;
- `steady-straight-walk`;
- `walk-turn-left`;
- `walk-turn-right`; and
- `walk-stop`.

The exact source intervals are:

1. `LocomotionFlat08_000.bvh [0, 840)` for stand-to-walk and mixed gentle turns;
2. `LocomotionFlat08_000.bvh [11160, 12000)` for right plus straight walking;
3. `LocomotionFlat08_000.bvh [12090, 12930)` for straight plus left walking; and
4. `LocomotionFlat11_000.bvh [17062, 17790)` for walk-to-stand and idle.

Every interval is authoritative 120 Hz source motion and is independently
retargeted to G1 with the existing exact pinned GMR/retarget-project commits,
`pfnn_position_scale=5.6444`, and `grounding=flat`. Duration-preserving
resampling produces 420, 420, 420, and 364 frames at 60 Hz before any
continuity split. No run, jog, jump, or crouch row may be admitted.

The existing 256-frame Flat01 bundle remains immutable evidence but is not an
input to the expanded command-support model.

### Per-row label derivation

Each BVH's authenticated released `.gait` matrix is resampled from 120 to 60
Hz with the same left/right/alpha map as motion. Rows are rejected when any
non-stand/non-walk gait column 2:8 exceeds `1e-6`. The remaining rows receive
two independent labels:

- gait label `stand` when stand weight is `>=0.8`;
- gait label `walk` when walk weight is `>=0.8`;
- otherwise `walk-start` when the next in-range walk weight is not less than
  the previous in-range walk weight; or
- otherwise `walk-stop`.

Walk rows with at least 60 in-range successors receive a trajectory label from
the Simulation-root displacement and facing change one second ahead. Require
forward displacement in `[0.25,0.85] m`; then label signed yaw change
`[-10,+10]` degrees as `straight`, `[-35,-10)` as `right`, and `(10,35]` as
`left`. Other walk rows receive `other` and remain usable inside training
windows but cannot satisfy a command destination or seed requirement. Rows
without 60 successors likewise receive `other`. Backward displacement,
nonfinite geometry, or an excluded gait column rejects the row and splits the
admitted continuity range.

The data-v4 preflight requires at least 100 `stand` rows, one nonempty
`walk-start` and `walk-stop` run, 300 `walk/straight` rows, 180 `walk/left`
rows, and 180 `walk/right` rows after retargeting and range splitting. These
counts are preregistered lower bounds, not targets adjusted after inspection.

### Explicit exclusions

- LMM-canary mode fixes `desired_gait=1`, the controller's walk endpoint.
- The gait toggle is disabled in LMM-canary mode. The existing Shift binding
  selects the walk endpoint rather than run and is not relabeled.
- Ctrl/strafe mode is rejected.
- The controller uses the walk gait endpoint, never the run endpoint.
- Cardinal A/S/D commands turn the character toward the commanded world-space
  direction and then use forward walking; they do not claim side-step or
  backward-walk animation.
- Terrain dimensions remain four exact zero values and slopes/stairs are not
  loaded.

## Immutable artifacts

### Motion data bundle

Publish `g1-lmm-flat-data/v4` transactionally. It contains:

- `database.bin`;
- `features.bin`;
- `motion_labels.bin` containing the exact per-row gait and trajectory labels;
  and
- `manifest.json`.

The manifest authenticates every source BVH, gait file, retarget NPZ/receipt,
the exact 18-field retarget contract, canonical G1 XML bytes, canonical
31-bone skeleton, source-frame maps, continuity ranges, physical audits,
bilateral contacts, labels, 60 Hz rate, 20/40/60 horizons, and all artifact
hashes/sizes. Each published range belongs to one source interval and one
  source/continuity range. Motion labels are per-row metadata, not training
  ranges. Windows may cross adjacent allowed stand/start/walk/turn/stop labels
  within one source/continuity range so the networks can learn those
  transitions. Windows and derivatives may not cross source, continuity, or
  excluded-gait boundaries.

### Command-support bundle

Publish a second immutable `g1-lmm-command-support/v1` bundle after the motion
bundle is frozen. It references the exact data-manifest SHA-256 and contains:

- `queries.bin`: little-endian float32 normalized 31D queries;
- `targets.bin`: exact nearest row indices, distances, classes, and terrain;
- `command_matrix.json`: the complete command tapes and per-tick receipts; and
- `manifest.json`: artifact hashes, controller/source hash, fixed seeds, exact
  60 Hz clock, walk profile, thresholds, and split digests.

This separate lifecycle avoids mutating the immutable data bundle after the
ordinary runtime has generated its queries.

## Command matrix and pre-training gate

The actual ordinary 60 Hz G1 controller generates the command matrix using the
exact data-v4 bundle and a shared query-extraction function also used by the
viewer. It fixes `desired_gait=1`, disables the gait toggle and strafe mode,
and records the resulting applied-velocity and heading bits rather than
assuming scalar speeds. The bound build identity contains exact source hashes,
compiler identity, flags, and recorder binary SHA-256.

Every tape is exactly 120 ticks. Key masks use bits `W=1`, `A=2`, `D=4`,
`S=8`, no diagonals, fixed camera yaw zero, and no gait/strafe modifiers:

- idle -> W -> idle: masks `0` for ticks `[0,30)`, `1` for `[30,90)`,
  then `0` for `[90,120)`;
- idle -> A -> idle: `0`, `2`, `0` on the same boundaries;
- idle -> D -> idle: `0`, `4`, `0`;
- idle -> S -> idle: `0`, `8`, `0`;
- W -> A -> W: `1` on `[0,30)`, `2` on `[30,60)`, `1` on `[60,120)`;
- W -> D -> W: `1`, `4`, `1` on the same boundaries; and
- W -> S -> W: `1`, `8`, `1` on the same boundaries.

Seed rows are selected before publication from per-row motion labels. For each
tape, eligible rows are sorted by `(source_id, source_frame, database_row)`.
Idle-start tapes draw from `stand`; moving-start tapes draw from
`steady-straight-walk`. Training seeds are the eight unique rows nearest odd
fractional ranks `1/17,3/17,...,15/17`; evaluation seeds are the eight unique
rows nearest even ranks `2/17,4/17,...,16/17`. A collision, missing class, or
fewer than sixteen distinct eligible rows rejects preflight. The seven shared
tape templates therefore expose at most 6,720 tick queries per split before
deduplication. The exact selected rows, query bytes, and split digests are
frozen in the support manifest before GPU training.

For every tick, the bundle stores raw and normalized query bits, exact nearest
database row, nearest/second-nearest distance, intended and observed motion
class, terrain identity, and the 30-tick destination deadline result.

Before any GPU training:

- every value must be finite;
- terrain features 27:31 must be numerically zero and are canonicalized to
  positive-zero before serialization;
- the exact nearest row must have an allowed transient/destination label and
  `flat` terrain;
- query-to-oracle normalized feature RMSE (`L2/sqrt(31)`) must be `<= 0.50`
  on every tick; stored L2 and RMSE fields are named separately;
- every destination must be reached within 30 ticks; and
- training and evaluation cases share the seven tape templates but must have
  disjoint seed rows and disjoint serialized query bytes.

One failing tick rejects the data/support bundle. There is no GPU attempt.

## Projector training and gate

The Orange Duck projector architecture and objective stay unchanged:

- `31 -> 512 -> 512 -> 512 -> 512 -> 63`;
- AdamW, learning rate `1e-3`, AMSGrad, weight decay `1e-3`;
- feature L1 + `5 * latent L1` + `0.3 * distance L1`;
- 50,000 updates and batch size 32.

The training population contains 8,192 deterministic supported queries:

- 4,096 no-replacement samples from preregistered ordinary-runtime command
  traces (preflight rejects if fewer than 4,096 distinct query byte strings);
  and
- 4,096 Orange Duck-style noisy queries retained only when terrain dimensions
  remain zero and exact-oracle support RMSE is `<= 0.50`.

The noise generator examines exactly 65,536 deterministic candidates per
split. It must retain at least 4,096 training and 512 evaluation queries, with
no repeated query bytes within a split and no query bytes shared across
splits. Otherwise preflight rejects. The gate uses disjoint command traces plus
the 512 held-out supported-noise queries. The old broad unfiltered-noise
population remains report-only.

Acceptance thresholds do not change:

- normalized feature RMSE `<= 0.05`;
- raw autoencoder-latent RMSE `<= 0.10`; and
- maximum component error across normalized features and raw latent values
  `<= 0.50`.

These preserve the thresholds and units already used by the rejected run and
the Orange Duck objective. Receipts rename the old misleading
`normalized_latent_rmse` field to `raw_latent_rmse` and the mixed-unit maximum
accordingly; no latent rescaling is introduced in this correction.

Thresholds apply to every preregistered command-tape tick individually before
the exogenous terrain overwrite. Aggregate population metrics remain
diagnostic and cannot mask a failing tick.

## Model and runtime contract

Accepted output uses `g1-lmm-model/v2` and includes
`projector_support.json`. The model manifest binds:

- the exact data-v4 manifest hash;
- the exact command-support manifest hash;
- query/target/oracle-index digests;
- generator versions, seeds, counts, and split identities;
- fixed support/output thresholds;
- `model_scope=expanded-flat-walk-overfit-canary`;
- `evaluation_scope=preregistered-cardinal-command-overfit-canary`; and
- `generalization_claim=none`.

The C++ loader validates every field and artifact before allocating network
evaluation buffers.

Each LMM tick performs a brute-force exact nearest-row diagnostic against the
small canary database before committing. It transactionally rejects when:

- query-to-oracle RMSE exceeds `0.50`;
- projected normalized-feature RMSE to the oracle exceeds `0.05`;
- raw projected-latent RMSE exceeds `0.10`;
- maximum component error across normalized features and raw latent exceeds
  `0.50`; or
- any pose, continuity, collision, contact, or finiteness gate fails.

The runtime never substitutes the exact nearest row, clamps the query, holds a
pose, or falls back to ordinary MM. The overlay identifies unsupported input
and shows the measured support/projector errors.

## Training and acceptance sequence

1. Retarget and mechanically audit all four bounded intervals.
2. Publish and independently validate data v4.
3. Generate and independently validate the ordinary-runtime command-support
   bundle.
4. Freeze source/data/support/code hashes.
5. Run exactly one full CUDA training job with decoder 100k, stepper 100k, and
   projector 50k updates.
6. Require all existing decoder/stepper gates plus the supported projector
   gate and safe reload.
7. Run the exact scripted command matrix under LMM with zero ordinary commits,
   holds, resets, fallbacks, unsupported ticks, or rejected poses.
8. Personally drive and visually inspect the unchanged accepted model before
   opening the same viewer for the user.

Any data, support, training, reload, scripted-runtime, or visual failure stops
the canary. There is no threshold, seed, budget, or corpus tuning and no second
fit under this frozen contract.
