# G1 Overnight Steerable Terrain LMM Design

## Objective and honest completion claim

By 08:00 America/Los_Angeles, produce an end-to-end MuJoCo proof of concept in
which a G1 can be steered with W/A/S/D or a Linux joystick across the supported
terrain scenes.  The visible pose must pass through a model trained on the
assembled multi-source G1 walking corpus; it must not be source-route replay.

The overnight system is a **hybrid terrain learned motion matcher**:

- deterministic terrain- and command-conditioned nearest-neighbour search
  selects a supported database state;
- a learned encoder/latent table and learned decompressor reconstruct the
  displayed G1 pose;
- a learned derivative stepper may advance between searches if it passes its
  rollout gate; otherwise range-safe database successors advance between
  searches;
- live terrain samples authoritatively replace the four terrain channels and
  root height is placed from the selected row's support-relative clearance.

This is not a learned projector and it is not evidence of arbitrary-terrain
generalization.  It is allowed to claim steering on the terrain classes and
height envelope covered by the assembled corpus and tested scenes.  Queries
outside that envelope must be visible in telemetry and use the classical
source-pose fallback rather than emit an unconstrained learned pose.

## Data milestones

### Working broad corpus (blocking for the first trained model)

Use the existing immutable bank candidate at
`/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate`:

- 3,970,932 canonical G1 rows at 25 Hz;
- 15,815 source ranges: full Takara walk, 1,769 GRAIL curb clips,
  1,857 GRAIL slope clips, and 12,188 GRAIL stair clips;
- 31-D matching features reconstructed after range-local conversion;
- authenticated 4-D terrain and 3-D support sidecars;
- all fourteen existing terrain scenes.

For the overnight model the bank remains at its authenticated 25 Hz rate and
uses `dt=0.04` plus joint 31-D features at horizons `(8, 17, 25)`.  This avoids
making a new multi-million-row quaternion resampler part of the critical path.
No derivative, successor or feature horizon crosses a range boundary.

The model is trained and evaluated over all resampled rows.  A deterministic source-range
split, not a row split, reserves at least 10% of GRAIL ranges for evaluation;
Takara receives non-overlapping temporal evaluation ranges.

### Augmented walking corpus (parallel, incorporated before the final fit when ready)

Append without changing the working milestone:

- the authenticated 595-row GRAIL authored slope clip;
- the already-retargeted PFNN walking/turning/terrain clips and their released
  terrain fits;
- additional readily processed GRAIL slope/stair sources.

Every appended clip is converted to the canonical 31-bone Holden layout,
retains an independent range, and uses a family label for balanced sampling.
Every appended source is converted range-locally to 25 Hz before combination.
The learned model therefore has one exact `dt = 0.04` contract.

The much larger all-source PFNN/GRAIL corpus continues as a resumable background
build.  Its incompleteness cannot block delivery of the broad-corpus visualizer,
and it cannot be described as processed if its terminal receipt is absent.

The overnight PFNN supplement contains only the 17 authenticated clips whose
released role is `train`.  Those 2,125 rows improve the fitted corpus, but they
do not contribute source-held-out evaluation rows.  The reported held-out
metrics therefore measure the primary Takara/GRAIL split and do not establish
PFNN-specific generalization.

## Model and experiments

The primary model follows the Orange Duck representation:

- compressor input: 908;
- matching features: 31, including terrain at indices 27:31;
- latent: 32 or 64;
- decompressor output: 458;
- contacts: two logits;
- deterministic AdamW training with family-balanced mini-batches.

Run bounded experiments in parallel:

1. baseline: latent 32, existing 512-wide compressor/decompressor;
2. capacity variant: latent 64 with the same hidden width;
3. only if both fail reconstruction: a wider deterministic autoencoder.

A VAE is not the first fallback because stochastic latent regularization works
against exact pose reconstruction.  It is considered only if deterministic
models reconstruct training rows but fail source-held-out rows in a way that
indicates latent-manifold regularization, not merely insufficient capacity or
optimization.

Practical proof-of-concept decompressor gates on both train and source-held-out
rows are:

- finite outputs and native joint limits;
- joint MAE <= 0.03 rad;
- joint frame-max p95 <= 0.10 rad;
- FK body-position p95 <= 0.08 m;
- support-foot position p95 <= 0.05 m;
- local translation p95 <= 0.03 m;
- each contact F1 >= 0.85.

The best model is selected by the held-out tuple `(joint MAE, FK p95, contact
error)`, never by training loss alone.  All attempted configurations and gates
are written to immutable evaluation receipts.

### Implementation variance: native-limit model gate

The original gate above required every train and source-held-out decoded row to
pass native G1 joint limits.  The delivered model does **not** meet that
full-population requirement: an exhaustive diagnostic found 6,127 learned
row-aligned reconstructions outside a native limit at the runtime tolerance.
That result is retained rather than relaxing the XML limits or presenting the
model artifact as having passed the original gate.

For this proof of concept, acceptance is instead scoped to authenticated,
scripted, supported-terrain MuJoCo routes.  Exact search retries candidates in
score order when the learned reconstruction is outside a native joint limit.
The selected pose is checked again against the captured native model before it
can be committed; canonical fallback remains a separately bounded diagnostic
path rather than an exclusion-retry trigger.  Formal evidence must show zero
committed joint-limit violations, zero clamps, and zero fallback.  This is a
runtime feasibility contract, not a claim that every stored row is natively
valid or that arbitrary live-terrain-conditioned queries have been exhausted.

## Runtime motion matching

Build a `scipy.spatial.cKDTree` over normalized 31-D rows.  At startup, on every
command change, on a terrain-envelope change, and at least every six source
ticks:

1. keep pose/velocity channels from the current accepted state;
2. write desired future trajectory/facing from the filtered joystick command;
3. sample the live heightfield at 0.25, 0.50, 0.75 and 1.00 m;
4. query a candidate set from the tree;
5. score candidates with feature distance and a same-range transition penalty;
6. decode the selected feature plus its learned latent;
7. place the decoded root on live terrain using the selected row's stored
   root-support clearance;
8. commit only finite, joint-limited poses.

Between searches, use the learned stepper only if its rate-matched short-horizon
gate passes.  Otherwise use the selected range's next row and decode that row.
This fallback remains learned display reconstruction; it is not route playback,
because the selected range can change with command and terrain at every search.
The delivered PoC has no separate successor-distance term during a search;
range-safe successor preference is implemented as deterministic next-row
advancement between searches.

Controls:

- keyboard: W/S forward-speed command, A/D steering, Space stop, R reset;
- optional Linux evdev gamepad: left stick Y speed, left stick X steering;
- unsupported or absent joystick devices fall back to keyboard without failing
  model load.

## Evaluation and delivery

Automated evaluation must include:

- full train and source-held-out decompressor metrics;
- search retrieval accuracy on manifest-bound row reconstruction queries;
- scripted idle, forward, left, right, stop and terrain-crossing commands;
- at least 1,000 headless MuJoCo frames with no NaN, joint-limit violation or
  out-of-range row access;
- evidence that at least two terrain classes and at least two source ranges are
  selected;
- live terrain-channel variance and nonzero learned-model evaluation counts.

The interactive viewer displays the selected source family/range/row, search
distance, terrain channels, learned decoder count, fallback count and support
status.  The handoff includes one launch command and one headless verification
command.  The accepted label is exactly:

`HYBRID TERRAIN LMM POC (EXACT SEARCH + LEARNED GENERATOR; SUPPORTED TERRAIN ONLY)`

## Failure policy

- Data augmentation failure: continue with the immutable 3,970,932-row broad
  corpus and report excluded families precisely.
- Baseline model failure: select the latent-64 or wider deterministic model if
  and only if its held-out receipt is better and green.
- Stepper failure: search/decode every source tick; never block the viewer.
- Learned decode failure at runtime: use the selected canonical source pose for
  that tick, increment a visible fallback counter, and continue searching.
- Unsupported terrain: display `OUT OF TRAINED SUPPORT`; do not claim success
  for that terrain even if the visual fallback remains upright.
- No experiment may overwrite another model or receipt.
