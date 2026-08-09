# G1 PFNN Joint-State Input Design

Date: 2026-08-09

## Decision

Keep the phase-functioned neural network, trajectory conditioning, terrain
sampling, 30 Hz controller, and native G1 output pose. Make the G1 pose state
observable by appending the current 29 joint angles and 29 joint velocities to
the existing 288 physical inputs. Phase remains a separate scalar argument.

The resulting model input has 346 values:

```text
existing PFNN trajectory/terrain/body state   288
current G1 joint position q(t)                 29
current G1 joint velocity qdot(t)              29
                                               ---
                                               346
```

This is the smallest reliable G1 adaptation of the released PFNN architecture.
It preserves the four cubic phase banks and direct G1 actuation while removing
the ambiguity that caused averaged and twisted limbs.

### Periodic angle encoding

The diagnostic also evaluates `sin(q(t)), cos(q(t)), qdot(t)` as a controlled
375-value treatment. It is not the default contract. Every canonical G1 hinge
range is narrower than `2*pi` and lies in one bounded branch; the widest is
`5.7596 rad`, and the wrist-yaw ranges are only `[-1.61443, 1.61443] rad`.
Raw `q` therefore preserves meaningful distance to a mechanical limit, while
periodic encoding can make opposite ends of a bounded hinge look artificially
close.

The raw 346-value treatment is selected only when it passes all diagnostic
gates. If raw `q` fails but the periodic treatment passes, stop and use that
evidence to propose a separately versioned periodic contract; do not silently
change this design. The output remains physical `q(t+1)`; predicting or
decoding `sin/cos` outputs is outside this correction.

## Evidence for the correction

The rejected released-only checkpoint proves that the current 288-input
contract is structurally insufficient:

- A homogeneous 64-frame clip can be memorized at `0.00666 rad` joint MAE.
- The 160-epoch full-corpus checkpoint reaches only `0.16678 rad` validation
  MAE, `1.99155 rad` frame-max p95, and `2.32623 rad` maximum error.
- Body positions are predicted well while elbows, shoulder yaw, wrist roll,
  wrist yaw, hip yaw, and ankle roll fail.
- Wrist yaw and ankle roll have exactly zero body-position Jacobian; wrist-roll
  positional sensitivity is only `0.00135 m/rad`.
- Released retargets contain broad twist targets and discontinuities, including
  32 adjacent 30 Hz steps above `0.5 rad` and a `2.4923 rad` wrist-roll step.
- The original released PFNN predicts 93 joint-rotation values. The current
  G1 adaptation replaced those with 29 hinge angles without adding prior joint
  angles or body orientations to the input.

Normalization, phase interpolation, mirroring, checkpoint selection, and
viewer joint ordering were independently ruled out as the primary cause.

## Scope

### Included

- A versioned 346-value classic-G1 PFNN input contract.
- Dataset rows containing physical `q(t)` and `qdot(t)` in canonical IsaacLab
  joint order.
- Exact train-only normalization for the added fields.
- Mirroring of added joint state using the existing canonical joint
  permutation and sign convention.
- Model/checkpoint/evaluator/runtime support for the new input width.
- A bounded multi-clip A/B diagnostic before any full retraining.
- Rebuilding the released-PFNN corpus beyond the current small vertical slice,
  then adding the corrected GRAIL slope rows under the same contract.
- Final offline, closed-loop, and interactive slope validation.

### Excluded

- Foot IK, pose damping, output clipping, hidden pose correction, or animation
  smoothing used to conceal model errors.
- Changing the terrain scale or the fixed released-PFNN limb thresholds to
  make a model pass.
- Promoting or initializing from the rejected `released-transfer-v3`
  checkpoint.
- Changing joint-loss weighting in the first representation experiment. That
  is a separate variable and may be tested only after the observability A/B
  result is known.

## Data contract

### Authoritative state

For a training window centered at frame `t`:

- Existing 288 values retain their current physical meaning.
- `q(t)` is the native G1 joint vector at the recurrent input frame.
- `qdot(t)` is the finite-difference G1 joint velocity at that same frame in
  radians per second.
- The output joint target remains `q(t+1)`.
- All joint arrays use `ISAACLAB_JOINT_NAMES` order and float32 storage.

When source motion is available, the builder packs `q(t)` and `qdot(t)`
directly from the source clip. A migration adapter may derive them from exact
same-clip, same-lane predecessor targets only when the required predecessor
rows are unique and consecutive. It must drop boundary rows rather than infer
missing history.

### Continuity and branch flips

No joint trajectory is silently unwrapped or corrected. Every split is audited
in physical units. A row whose target transition exceeds `0.225 rad` from its
unique same-clip, same-lane predecessor is rejected, matching the existing
mixed-corpus safety boundary. Rejection creates a sequence boundary. The
dataset receipt records accepted and rejected counts by source and joint.

This removes discontinuous retarget targets without inventing a pose. If a
source loses all usable motion rows, dataset construction fails closed.

### Normalization and mirroring

The 58 appended fields receive train-only mean and standard deviation arrays.
Values with standard deviation below `1e-6` use `1.0`, consistent with the
existing dataset contract. Validation and held-out data never influence these
statistics.

Mirroring applies the existing 29-joint permutation and sign vector to both
`q` and `qdot`. Applying the mirror twice must reproduce the original row
bit-for-bit.

## Model and artifact contracts

`PhaseFunctionedNetwork` gains an explicit input width. Its default remains
288 so existing terrain-PFNN tests and artifacts retain their current meaning.
The classic G1 path constructs it with input width 346. The phase banks,
hidden width, cubic interpolation, activations, dropout behavior, and 268-value
output remain unchanged.

New artifacts fail closed under versioned schemas:

- vertical/mixed dataset: `g1-pfnn-vertical-dataset/v3`
- classic checkpoint: `classic-g1-pfnn/v3`
- exact-input transfer evaluation: `classic-g1-pfnn-transfer-evaluation/v2`

Each checkpoint binds the input layout/version, dataset digest, normalization,
canonical joint order, kinematic signature, and runtime seed. A v2 checkpoint
cannot be loaded as a v3 model.

The rejected v3 run directory remains immutable evidence and is never reused
as an output destination.

## Runtime recurrence

The runtime continues to construct the existing 288-value trajectory,
terrain, semantic, and body state. Immediately before inference it appends the
last committed `q` and `qdot`.

- Bootstrap uses receipt-bound joint position and velocity from the checkpoint
  runtime seed.
- On a committed frame, `q` becomes the raw predicted `q(t+1)` and `qdot`
  becomes `(q(t+1) - q(t)) * 30`.
- On a held or invalid frame, both values remain unchanged with the rest of the
  transactional recurrent state.
- The model output is not blended, damped, clipped, or corrected.

Joint limits, maximum step envelopes, finite checks, terrain support checks,
and hold reporting remain authorization gates rather than post-processing.

## Required falsification experiment

Before rebuilding or training the full corpus, run a paired deterministic
multi-clip experiment:

1. Select non-mirrored motion-lane rows from `LocomotionFlat02_000` and the
   discontinuity-heavy `WalkingUpSteps02_000`, `WalkingUpSteps09_000`, and
   `WalkingUpSteps12_000`.
2. Use fixed fit blocks and immediately following held-out blocks from every
   clip after applying the `0.225 rad` continuity boundary.
3. Train a baseline with the current 288 inputs, the primary treatment with
   raw `q(t), qdot(t)`, and a periodic diagnostic treatment with
   `sin(q(t)), cos(q(t)), qdot(t)`. Use the same phase, targets, seed, model
   depth, hidden width, optimizer, update budget, batching, and dropout
   setting.
4. Evaluate all three checkpoints through the same exact-input physical limb
   gate.

The representation hypothesis is accepted for this design only if the raw
joint-state treatment passes all released-PFNN limb thresholds on both fitted
and held-out blocks while the baseline fails at least one held-out threshold.
The periodic result is supporting evidence only. If periodic passes while raw
fails, this design stops for revision. All results and selected row receipts
are persisted. No diagnostic checkpoint is promotable.

If the treatment also fails, stop before a full run and re-examine the retarget
continuity and target representation. Do not add loss weighting or smoothing
to rescue the experiment.

## Full corpus and GRAIL sequence

After the A/B treatment passes:

1. Retarget and pack the remaining released PFNN motions using the pinned
   offline retargeter and the v3 joint-state contract. The corpus must no
   longer be limited to the 17 training clips in the current vertical slice.
2. Preserve clip-disjoint train/validation identities and source provenance.
3. Convert corrected GRAIL slope rows to the same 346-input physical contract.
4. Build train-only normalization once over the accepted mixed training split.
5. Train a released-only checkpoint and require the fixed exact-input limb
   gates before using it as mixed-model initialization.
6. Add GRAIL rows and train/fine-tune without allowing GRAIL metrics to dilute
   released-PFNN metrics.

Any failed stage preserves its candidate and stops the next stage.

## Verification and acceptance

The completed solution requires all of the following:

- Exact input/dataset/checkpoint round-trip and tamper tests.
- Mirror involution and q/qdot time-alignment tests.
- Multi-clip A/B treatment acceptance as defined above.
- Released-PFNN train and validation limb gates:
  - joint MAE `<= 0.100 rad`
  - joint RMSE `<= 0.150 rad`
  - frame-max p95 `<= 0.500 rad`
  - maximum joint error `<= 1.000 rad`
- No unsafe fitted transition, joint-limit violation, nonfinite output, hidden
  correction, or source-metric dilution.
- Deterministic undamped flat and stair comparisons that pass the same gates.
- A 20-second interactive run over flat terrain, ascent, crest, descent, and
  turnaround with no holds, implausible twisting, foot dragging, or terrain
  loss.
- Final visual inspection performed through the same controls and viewer the
  user will run.

The viewer is launched for handoff only after all offline and closed-loop gates
pass.

## Failure handling

- Schema, receipt, normalization, ordering, or source mismatches reject before
  model construction or inference where possible.
- Missing predecessor state, sequence gaps, and unsafe transitions are explicit
  row rejections.
- A failed A/B diagnostic blocks corpus-scale work.
- A failed released-only model blocks GRAIL mixing.
- A failed mixed model blocks viewer promotion.
- No failed artifact is overwritten, promoted, or described as working.
