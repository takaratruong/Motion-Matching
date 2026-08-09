# G1 PFNN Transfer Correction Design

Date: 2026-08-09

## Goal

Deliver a controllable G1 PFNN whose articulated pose first reproduces the
released PFNN motion after offline retargeting, then extends that proven model
with corrected GRAIL slope examples. Root terrain placement alone is not an
acceptable result.

## Confirmed defect

`build_g1_pfnn_mixed_dataset._grail_physical_rows` currently reconstructs raw
GRAIL inputs with `x_normalized * x_std + x_mean`. PFNN input normalization
applies an additional `0.1` scale to `previous_body_position` and
`previous_body_velocity`, so those two slices must first be divided by `0.1`.

An exact source-to-mixed comparison found 2,285 matched GRAIL rows. All other
input fields and all targets agree within approximately `2.1e-7`, while the
two recurrent body fields differ by as much as:

- body position: `0.2967620622 m`
- body velocity: `2.5130044918 m/s`

The current model consequently produces poor joints even on fitted inputs. On
released-PFNN training rows its joint MAE is `0.14169 rad`, with individual
wrist errors up to `1.88312 rad`. The viewer's IsaacLab-to-MuJoCo joint mapping
round-trips exactly, so joint-order rendering is not the source of this error.

## Sequence of work

### 1. Correct the physical-input boundary

Introduce one authoritative inverse of `normalize_pfnn_input`. It will undo
the recurrent-body `0.1` scaling before applying `x_std` and `x_mean`.
The mixed-dataset builder will use this inverse instead of an open-coded
affine transform.

The corrected builder must reproduce every matched source GRAIL input and
target within float32 reconstruction tolerance. Existing dataset artifacts are
immutable; the corrected corpus is written to a new directory and gets a new
digest.

### 2. Prove released-PFNN transfer in isolation

Before mixing sources, train a released-PFNN-only G1 model using the existing
offline-retargeted 30 Hz training and validation corpus. The proof uses three
levels:

1. exact-input joint metrics on train and validation rows;
2. a deterministic closed-loop sequence from a known retargeted clip;
3. a MuJoCo side-by-side visual comparison of retargeted source and prediction.

The model cannot pass merely because root motion follows the course. Limb
motion must be numerically close to the retargeted targets and visually retain
the same gait, arm swing, and contact timing. No render-only smoothing is used
to decide this gate.

### 3. Add corrected GRAIL data

Only after the released-only gate passes, build the corrected mixed corpus and
fine-tune from the accepted released-only checkpoint. GRAIL addition must not
materially regress the released-PFNN limb metrics. Evaluation reports metrics
separately for released PFNN and GRAIL rather than only as one aggregate.

### 4. Final interactive verification

Run the actual viewer with the promoted checkpoint and the transferred PFNN
height surface. Drive it with W/A/S/D through flat, ascent, descent, left and
right turns, reverse, stop, and restart. Record the raw predicted joint traces
and inspect the rendered motion. If articulated motion is visibly broken, the
checkpoint is rejected even if root and terrain diagnostics pass.

## Acceptance gates

The implementation will freeze exact numeric thresholds after measuring the
retargeted-source noise floor and a small released-only diagnostic fit. At a
minimum, every promoted checkpoint must satisfy all of these qualitative
contracts:

- corrected source-to-mixed inputs reconstruct within float32 tolerance;
- exact fitted inputs no longer contain large limb errors comparable to the
  current `1.88 rad` failures;
- released-only closed-loop motion remains finite and articulated rather than
  collapsing to a mean pose;
- adding GRAIL does not hide a released-PFNN regression in an aggregate score;
- viewer joint values equal the model's canonical output after the tested
  IsaacLab-to-MuJoCo permutation;
- the final W/A/S/D drive is personally inspected before reporting success.

If the released-only diagnostic cannot overfit a small contiguous sample, work
stops before full training and returns to the feature, phase, or objective
boundary. If released-only succeeds but mixed training fails, the fault stays
isolated to the GRAIL merge/fine-tune stage.

## Testing strategy

Implementation follows red-green TDD:

- a regression test first demonstrates the current recurrent-body
  reconstruction error;
- unit tests cover the exact inverse normalization and reject invalid shapes,
  nonfinite values, and nonpositive standard deviations;
- a source-to-mixed parity test covers all input slices and targets;
- evaluation tests require source-specific joint metrics and reject aggregate-
  only promotion;
- viewer tests retain the exact joint-order mapping and prohibit display
  damping from participating in acceptance measurements.

## Out of scope

This correction does not add foot IK, physics control, foot locking, root
projection, or new terrain synthesis. Those features cannot compensate for a
PFNN that fails to reproduce its own retargeted motion.
