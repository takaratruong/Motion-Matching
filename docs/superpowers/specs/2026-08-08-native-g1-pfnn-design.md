# Native G1 PFNN Design

## Goal

Build the SIGGRAPH 2017 Phase-Functioned Neural Network pipeline with Unitree
G1 morphology. The delivered controller must be trained from an offline
retarget of the authors' released motion corpus, use the paper's terrain-fitting
and runtime structure, and produce visually convincing interactive G1 motion on
flat ground and hills.

## Reference implementation

The behavioral source of truth is the authors' release at
`/home/ubuntu/datasets/pfnn/pfnn`:

- `generate_database.py` defines the motion, phase, gait, terrain, and
  input/output parameterization.
- `train_pfnn.py` defines normalization, four-bank cubic PFNN training, dropout,
  regularization, batch size, optimizer, and epoch count.
- `demo/pfnn.cpp` defines the 120-sample trajectory controller, autoregressive
  state, predicted-root integration, phase update, pose smoothing, and
  contact-driven terrain IK.

The released assets are licensed for academic and non-commercial use. This
implementation will retain the required PFNN citation and will not redistribute
the original data or pretrained weights in Git.

## Scope

This milestone includes:

- all 40 annotated original BVH captures and their 40 released mirrored
  variants (`rest.bvh` is a calibration skeleton, not a motion clip);
- the released phase, gait, and footstep annotations;
- offline whole-body retargeting to the 29-DoF G1 model;
- the released terrain-patch generation and ten-patches-per-cycle fitting
  procedure, converted consistently from source units to metres;
- a four-control-point cubic PFNN trained on native G1 features;
- the reference real-time trajectory and phase controller;
- G1 contact locking, two-link leg IK, and foot orientation on terrain;
- a W/A/S/D MuJoCo viewer with a gravity-aligned root; and
- automated, quantitative, and human visual acceptance evidence.

This milestone does not use MotionBricks, motion matching, portals, the existing
288-to-268 terrain PFNN checkpoint, command-driven root dragging, or online
retargeting of the released human checkpoint.

## G1 representation

The network preserves the reference representation rather than predicting
actuator angles directly. The G1 model has 30 articulated bodies below the
world body. Let `B = 30`, `K = 12` trajectory knots, and `G = 6` gait channels.

The phase remains a separate scalar in `[0, 2π)` and selects four cyclic
Catmull-Rom parameter banks.

The input has 336 values:

| Group | Width |
|---|---:|
| Root-relative trajectory positions, X/Z | 24 |
| Root-relative trajectory directions, X/Z | 24 |
| Six gait values at twelve knots | 72 |
| Previous G1 body positions, XYZ | 90 |
| Previous G1 body velocities, XYZ | 90 |
| Right/centre/left terrain heights at twelve knots | 36 |

The output has 302 values:

| Group | Width |
|---|---:|
| Root planar delta, yaw delta, phase delta | 4 |
| Left/right heel and toe contacts | 4 |
| Six future trajectory positions, X/Z | 12 |
| Six future trajectory directions, X/Z | 12 |
| G1 body positions, XYZ | 90 |
| G1 body velocities, XYZ | 90 |
| G1 body exponential-map rotations, XYZ | 90 |

The network is therefore `336 -> ELU(512) -> ELU(512) -> 302`, with four
cyclic Catmull-Rom control-point banks and dropout retention probability 0.7.

Predicted body rotations are converted to local parent-relative rotations and
projected onto each G1 hinge axis to produce MuJoCo joint coordinates. Predicted
positions and velocities remain the autoregressive state and participate in the
same 0.5 pose/velocity blend used by the reference runtime.

## Offline retargeting

Each source BVH is loaded at its released rate and downsampled to 30 Hz at the
same point as the reference pipeline. A deterministic scale is computed from
the summed source and target hip-to-knee-to-ankle chain lengths. Root
translations and terrain dimensions use this uniform scale; joint rotations do
not.

Each frame is solved onto G1 with a warm-started constrained least-squares IK:

- pelvis position and yaw;
- both ankle positions and orientations;
- both knee bend directions;
- torso and head orientation;
- both wrist positions and orientations; and
- a joint-space continuity term relative to the preceding frame.

The solver respects the native MuJoCo joint limits. The result is rejected if
any frame is non-finite, violates a joint limit, swaps a left/right target, or
exceeds these physical errors:

- stance heel or toe position: 0.02 m;
- swing ankle position: 0.04 m;
- wrist position: 0.06 m;
- pelvis position: 0.02 m; or
- orientation error for a constrained end effector: 12 degrees.

Released phase and gait labels are preserved. Foot contacts are recomputed from
the accepted G1 heel/toe trajectories and checked against the released
footstep intervals. A disagreement lasting more than two consecutive frames
rejects that cycle rather than silently changing its phase.

Before dataset generation, an interactive source-motion viewer must show at
least one walk, run, turn, stair ascent, stair descent, jump, and crouch clip on
G1 without persistent arm elevation, deep crouch, foot sliding, or pose snaps.

## Terrain fitting and dataset generation

The released heightmaps and patch sampler are reused. For each accepted
locomotion cycle, the fitting score uses stance-foot error, swing-foot
clearance, and the reference special cases for jumps and beams. The ten best
patches are refined with the reference radial-basis residual fit at stance
locations.

Each fitted cycle produces the G1 input, output, and phase arrays described
above. Terrain values are root-height-relative. Body positions, velocities, and
rotations are expressed in the smoothed yaw-root frame exactly as in the
reference preprocessing code.

Splits are by original, non-mirrored capture identity. Mirrored clips and every
terrain fit of a capture remain in the same split. Normalization is fit on the
training split only. The manifest records source-file hashes, retarget settings,
accepted and rejected cycles, terrain-patch hashes, split identities, feature
layout, normalization hash, and final dataset hash.

## Training

Training follows the reference recipe:

- four cubic phase banks;
- two hidden layers of 512 ELU units;
- dropout retention probability 0.7;
- batch size 32;
- Adam learning rate `1e-4`;
- 20 complete shuffled epochs;
- normalized mean-squared prediction error over every output, including
  contacts; and
- parameter regularization coefficient `0.01` using the reference layer-cost
  semantics.

Trajectory-related dimensions share group standard deviations. Input body
position and velocity dimensions receive the reference 0.1 importance scaling.
Unused or structurally redundant target bodies may be masked only through a
documented fixed input weight derived from the G1 body tree; outputs are never
masked.

The selected checkpoint is the lowest validation loss among the 20 epochs.
Checkpoint contents include CPU tensors, normalization arrays, body and joint
orders, feature layout, dataset digest, retarget digest, G1 kinematic signature,
epoch, and validation score.

## Runtime and terrain IK

The runtime ports the reference controller structurally:

- a 120-frame past/future trajectory sampled every ten frames for the network;
- smoothed desired velocity, facing, and six gait channels;
- nonlinear future position and direction blending;
- terrain sampling at right, centre, and left offsets;
- PFNN-predicted root delta and future trajectory integration;
- phase advance modulated by the stand gait;
- autoregressive G1 body position and velocity state; and
- predicted-position versus integrated-velocity pose smoothing.

Finite network output is not overwritten with commanded root motion. The
viewer renders PFNN-predicted translation and yaw while keeping root roll and
pitch aligned with gravity.

Four predicted contact values drive G1 heel/toe locks. Locked targets follow
the reference threshold and fade behavior. Leg IK adjusts hip, knee, and ankle
DoFs to the locked foot target, and foot orientation samples the local terrain
under heel, toe, and side points. IK corrections are applied after inference
and do not feed corrected joint coordinates back as a replacement for the
network's body-state recurrence.

## Failure handling

Preprocessing, training, checkpoint loading, and runtime fail closed on invalid
shapes, hashes, non-finite values, joint-limit violations, missing source
annotations, or unsupported terrain queries. A runtime failure holds the last
committed frame and reports the exact field and tick; it never applies a pose,
root, or MotionBricks fallback.

## Acceptance

Automated acceptance requires:

- deterministic retargeting and dataset hashes across two clean runs;
- all required motion categories represented in the training split;
- no train/validation/test capture identity overlap;
- finite training and validation metrics for 20 epochs;
- a safely reloadable checkpoint with identical fixed-batch inference;
- 600 continuous runtime ticks without a held frame; and
- nonzero realized traversal on flat ground, ascent, crest, descent, and
  landing.

Final acceptance is visual. Using the exact W/A/S/D viewer, the implementer
must personally test idle, start, walk, jog, left/right turns, stop, forward and
reverse hill traversal, cresting, descending, and stopping on a slope. The
result is rejected for persistent crouching, raised or frozen arms, root
dragging, phase freeze while moving, visible foot penetration, floating stance
feet, excessive foot sliding, discontinuous pose changes, or failure to respond
to control input.

The viewer is shown to the user only after this manual pass succeeds.
