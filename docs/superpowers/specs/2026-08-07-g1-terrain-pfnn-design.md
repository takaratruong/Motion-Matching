# Native G1 Terrain PFNN Design

- **Status:** Architecture approved; written design pending final review
- **Date:** 2026-08-07
- **Target:** Purely kinematic, continuously controllable Unitree G1 locomotion

## Goal

Build a native Unitree G1 Phase-Functioned Neural Network (PFNN) that produces
continuous idle and walking motion over flat ground and 5--20 degree slopes.
The controller receives camera-relative W/A/S/D intent and terrain heights,
then predicts the complete next G1 pose, root motion, gait phase, future
trajectory, and foot contacts at 30 Hz.

The first milestone renders the network output directly. It has no
MotionBricks runtime, terrain portals, runtime root-height heuristic, inverse
kinematics, or world-space foot locks.

This design supersedes
`2026-08-06-motionbricks-hill-portals-design.md` as the active hill-controller
direction. The portal and online-IK prototypes remain historical comparison
baselines.

## Reference Method and Local Source

The architecture follows Holden, Komura, and Saito, *Phase-Functioned Neural
Networks for Character Control*:

- <https://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf>
- <https://theorangeduck.com/page/phase-functioned-neural-networks-character-control>

The author's complete code, data, human-character checkpoint, and demo are
already available read-only at:

```text
/home/ubuntu/datasets/pfnn/pfnn
```

That package is restricted to academic and non-commercial use. It is a
behavioral and file-format reference only. The repository implementation is a
new PyTorch implementation of the published equations and does not copy the
human-specific Theano or C++ runtime. The supplied human checkpoint is not
retargeted to G1.

## First-Milestone Scope

### In scope

- Native 29-DoF G1 qpos output.
- Pure kinematic playback at 30 Hz.
- Idle and walking intent.
- Camera-relative W/A/S/D movement with facing following travel direction.
- Continuous control before, during, and after a slope.
- Paired terrain-aware training from GRAIL slope motions.
- Flat and turning support from G1-retargeted LAFAN locomotion.
- Runtime terrain sampling along the desired trajectory.
- Raw network output without runtime pose correction.
- A small test map with flat ground and 10, 15, and 18.9 degree symmetric
  hills.

### Out of scope

- Running, jumping, crouching, stairs, curbs, and hand contacts.
- Physics, balance recovery, torque control, or SONIC tracking.
- Midair actions or non-cyclic phase functions.
- Separate travel and body-facing sticks.
- MotionBricks generation or matching at runtime.
- Foot IK, stance pinning, or root projection.
- Slopes steeper than 20 degrees.

## Coordinate and Timing Contract

All canonical data uses:

- Z-up right-handed world coordinates;
- root quaternion in WXYZ order inside the PFNN pipeline;
- the repository's native 29-joint G1 order;
- 30 Hz after source conversion; and
- a gravity-aligned local trajectory frame whose origin is the current ground
  support point and whose X axis is current root yaw.

Root yaw belongs to the trajectory frame. Root height and roll/pitch belong to
the predicted character pose. This lets the pelvis lean for a slope without
tilting the controller's definition of forward or rotating gravity.

The trajectory has 12 knots at these offsets in seconds:

```text
-1.000, -0.833, -0.667, -0.500, -0.333, -0.167,
 0.000,  0.167,  0.333,  0.500,  0.667,  0.833
```

Past knots are recorded states. Current/future knots are blended from user
intent and the prior PFNN trajectory prediction.

## Network Contract

### Phase function

Phase `p` is a scalar in `[0, 2*pi)`. It is not appended to the feature vector.
Instead it selects the complete regression weights by cyclic cubic Catmull-Rom
interpolation over four learned parameter banks.

For every weight or bias tensor, the four banks are `a0`, `a1`, `a2`, and
`a3`. The interpolation is periodic and differentiable at the wrap boundary.
Unit tests compare the implementation to Equation 7 of the paper and verify
value and first-derivative continuity at phase zero.

### Input vector: 288 values

The normalized input vector contains:

| Feature | Shape | Values |
|---|---:|---:|
| local trajectory position | `12 x 2` | 24 |
| local trajectory facing direction | `12 x 2` | 24 |
| relative terrain height: left/center/right | `12 x 3` | 36 |
| semantic intent: idle/walk | `12 x 2` | 24 |
| previous local G1 body positions | `30 x 3` | 90 |
| previous local G1 body velocities | `30 x 3` | 90 |
| **Total** | | **288** |

Terrain samples are taken at the trajectory center and `0.25 m` left and
right, perpendicular to the sampled facing direction. Heights are relative to
the current center support height.

Body positions and velocities use the current gravity-aligned trajectory
frame. They include all 30 canonical G1 bodies, including the pelvis.

### Output vector: 268 values

The denormalized output vector contains:

| Feature | Shape | Values |
|---|---:|---:|
| next-frame local trajectory position | `12 x 2` | 24 |
| next-frame local trajectory direction | `12 x 2` | 24 |
| current local G1 body positions | `30 x 3` | 90 |
| current local G1 body velocities | `30 x 3` | 90 |
| root height above center support | `1` | 1 |
| root roll/pitch exponential map | `2` | 2 |
| G1 joint angles | `29` | 29 |
| local planar root velocity | `2` | 2 |
| root yaw velocity | `1` | 1 |
| positive phase advance | `1` | 1 |
| heel/toe contact logits | `4` | 4 |
| **Total** | | **268** |

Contact order is left heel, left toe, right heel, right toe.

### Regression network

Each phase-selected network is:

```text
288 -> ELU(512) -> ELU(512) -> 268
```

The learned tensors have these shapes:

```text
W0: [4, 512, 288]   b0: [4, 512]
W1: [4, 512, 512]   b1: [4, 512]
W2: [4, 268, 512]   b2: [4, 268]
```

The first runtime uses cubic interpolation directly. Exporting 50 constant
phase slices is an optional optimization only after the cubic model passes.

## Training Data

### GRAIL slopes

The primary terrain source is:

```text
/home/ubuntu/datasets/GRAIL/data/slope
```

It contains 1,880 robot PKLs paired by stem with 1,880 USD terrains. Each
source motion has 250 frames at 25 Hz with root translation, XYZW root
quaternion, and 29 MuJoCo-order joint values.

Only terrains whose measured traversed surface grade is within `[5, 20]`
degrees enter the first milestone. The paired USD is always queried with the
documented GRAIL transform. Source and mesh hashes are retained in every
training-window record.

Translations and joints are interpolated from 25 to 30 Hz; root quaternions
use SLERP. Velocities are recomputed after resampling. MuJoCo forward
kinematics produces the canonical 30-body positions, orientations, and
velocities.

### Flat and turning locomotion

The flat source is:

```text
/home/ubuntu/.cache/g1-lafan-flat/g1
```

It contains 12 G1-retargeted LAFAN walking sequences and 86,859 frames at
30 Hz. The source license and manifest remain attached to derived records.
These sequences provide starts, stops, speed variation, and turns that the
mostly straight GRAIL slope clips do not cover densely.

The Takara walk artifact may be added only if its provenance and split can be
made independent. It is not required for the first accepted checkpoint.

### Split and sampling policy

GRAIL is split deterministically by `slope_NNN` terrain identity:

- 80% training;
- 10% validation; and
- 10% final test.

All ten motion variants for one terrain stay in the same split. LAFAN is split
by original base sequence, not by overlapping windows. Test identities are
sealed before model fitting and are not used for normalization, hyperparameter
selection, early stopping, or checkpoint selection. Validation data may select
the checkpoint but never updates weights or normalization statistics.

Training minibatches are stratified across flat, ascent, descent, and
flat/slope transition frames. Mirrored samples swap left/right bodies,
contacts, lateral trajectory values, roll, and yaw signs according to the
canonical G1 symmetry map.

## Contact and Phase Reconstruction

Contacts are reconstructed from the exact G1 sole spheres and paired terrain,
not inferred from root height alone. Heel and toe channels use their own probe
groups with hysteresis:

- enter distance: `0.012 m`;
- leave distance: `0.025 m`;
- enter tangential speed: `0.12 m/s`;
- leave tangential speed: `0.20 m/s`;
- enter foot angular speed: `0.75 rad/s`;
- leave foot angular speed: `1.25 rad/s`; and
- minimum support-normal Z: `0.50`.

A foot strike is the rising edge of either heel or toe contact after a stable
swing interval. Phase anchors are:

```text
left strike  = 0
right strike = pi
next left    = 2*pi
```

Phase is interpolated monotonically between alternating strikes. A moving
interval is rejected when strikes do not alternate, a half-cycle is outside
`[0.20, 1.00] s`, contact geometry is invalid, or phase would move backward.

For genuine starts and stops, phase is continued from the nearest valid
walking cycle while phase advance is allowed to approach zero. Stationary
poses are sampled across phase values so the idle regression cannot depend on
one arbitrary foot phase.

Every imported clip receives a phase/contact audit report. Rejected frames do
not silently become no-contact training samples.

## Feature and Target Construction

For each eligible center frame:

1. construct the gravity-aligned root frame from ground support and root yaw;
2. sample past/future root positions and facing at the 12 trajectory times;
3. raycast paired terrain at left, center, and right trajectory points;
4. transform current and previous body state into the local root frame;
5. compute local planar and yaw root velocities;
6. encode root roll/pitch without yaw;
7. attach phase, phase advance, semantic intent, and four contacts; and
8. write provenance and split identity with the sample.

Windows do not cross source-clip boundaries. Missing terrain rays, non-finite
values, invalid quaternions, model-limit violations, and unsupported slope
grades reject the window.

Training-set statistics normalize continuous inputs and outputs. Following the
paper, normalized body-position and body-velocity input channels receive a
`0.1` importance scale so trajectory and terrain remain influential. Binary
intent is normalized like other input features.

## Optimization

Training uses PyTorch and Adam. The baseline keeps the paper's structural
choices:

- two 512-unit ELU hidden layers;
- four cubic phase-control banks;
- dropout retention probability `0.7`;
- batch size `32` per GPU; and
- parameter regularization coefficient `0.01`.

Loss terms are:

- normalized continuous-output MSE;
- binary cross-entropy on four contact logits;
- unit-length penalty for predicted trajectory directions;
- nonnegative phase-advance penalty; and
- forward-kinematic consistency between predicted body positions and the
  predicted root/joint pose.

Joint, root, trajectory, phase, and contact loss groups are reported
separately. No metric is hidden inside only a total loss.

Training has two gates:

1. **Pipeline overfit:** a small fixed flat/slope subset must overfit and roll
   out through its known terrains before a full run is authorized.
2. **Full training:** train on the complete training split, select by validation
   closed-loop score, then evaluate the selected immutable checkpoint once on
   the sealed test split.

After one-step learning is stable, a short autoregressive fine-tuning stage
unrolls 16 frames and feeds predicted body state and trajectory back into the
network. It does not read test data or use rendered corrections.

## Runtime

The viewer maintains phase, the 12-knot trajectory, previous body state, and
the current qpos.

Each 30 Hz tick:

1. convert W/A/S/D into a camera-relative desired planar velocity;
2. smoothly align desired facing with nonzero travel direction;
3. blend desired future velocity with the prior PFNN prediction using the
   paper's velocity bias `tau_v = 0.5`;
4. blend desired facing with the prior prediction using `tau_d = 2.0`;
5. update past trajectory knots from realized root motion;
6. raycast center and `+/-0.25 m` terrain tracks;
7. normalize the 288-value input and evaluate the cubic PFNN at current phase;
8. denormalize the 268-value output;
9. integrate planar root velocity and yaw velocity for one tick;
10. set root Z to support height plus predicted root height;
11. combine integrated yaw with predicted roll/pitch and apply 29 joint angles;
12. advance phase modulo `2*pi` using the bounded positive phase output;
13. store predicted body state and trajectory for the next input; and
14. render qpos without IK or contact locking.

The viewer overlay reports desired/realized speed, phase and advance, terrain
grade, root height/tilt, four contact probabilities, feature envelope status,
and checkpoint identity.

### Runtime envelope handling

The controller does not extrapolate intentionally beyond its measured support:

- a future sample above 20 degrees or with no valid terrain ray is marked
  unsupported;
- desired future speed is reduced before the unsupported knot;
- if no supported continuation exists, desired speed becomes zero; and
- invalid/non-finite network output holds the last finite qpos and reports the
  exact rejected field.

Joint angles are checked against the native MuJoCo limits. A violating frame
is rejected rather than clipped invisibly into a different pose.

## Test Map

The initial MuJoCo scene contains generous flat run-up and three smooth,
symmetric mound profiles with maximum flank grades of 10, 15, and 18.9
degrees. Each mound has a short rounded summit and flat landing. Terrain
queries and rendered geometry use the same mesh.

The 18.9-degree mound is the primary canary. Its height and run remain inside
the measured GRAIL training envelope. Both travel directions use the same
controller; there are no directional portal assets.

## Verification and Acceptance

### Unit and contract tests

- Cubic phase interpolation matches the published equation.
- Phase interpolation is periodic and derivative-continuous.
- Input/output layouts contain exactly 288/268 values.
- Local/world transforms round-trip within tolerance.
- 25-to-30 Hz conversion preserves endpoints and quaternion normalization.
- Mirroring twice recovers the original sample.
- Terrain split keeps every `slope_NNN` identity in exactly one partition.
- Contact hysteresis and phase reconstruction reject malformed cycles.
- Runtime feedback uses the immediately preceding prediction.
- Unsupported terrain reduces desired speed instead of calling the network
  outside the envelope.

### Model gates

The pipeline-overfit checkpoint must reproduce its fixed samples and complete
a closed-loop known-slope rollout before full training starts.

The selected full checkpoint must meet all of these on held-out terrain:

- finite raw qpos and features for a 20-second rollout;
- no phase reversal or walking-phase freeze;
- maximum root translation step at most `0.060 m`;
- maximum root rotation step at most `0.35 rad`;
- maximum joint step at most `0.25 rad`;
- no native joint-limit violation;
- maximum raw sole penetration at most `0.015 m`;
- zero forbidden non-foot terrain penetration;
- median horizontal sole speed during predicted stance at most `0.10 m/s`;
- successful traversal of the 18.9-degree mound in both directions; and
- return to responsive flat walking after each landing.

Metrics are reported separately for flat, ascent, summit, descent, and landing
frames. Aggregate averages cannot hide a failed flank or transition.

### Interactive acceptance

The milestone is accepted when the user can launch one local viewer, steer G1
with W/A/S/D across all three hills, change direction while on a flank, and see
terrain-appropriate pelvis and leg motion without portal takeover, a pinned
foot, or runtime IK.

If the raw PFNN passes mechanically but has small residual stance skating, a
separate reviewed milestone may add the paper's contact-driven two-joint IK.
IK is not allowed to compensate for a failed PFNN checkpoint.

## Expected Risks

- GRAIL slope paths are less directionally diverse than the original PFNN
  human dataset; the LAFAN flat split supplies turning but not many turning
  slopes.
- Contact-derived phase can be noisy at starts, stops, and crest turns; those
  regions need explicit rejection and diagnostics.
- A phase-conditioned autoregressor can still drift when runtime commands are
  far from supervised trajectories; closed-loop evaluation is mandatory.
- Raw output may retain centimetre-scale foot skating even when full-body
  slope mechanics are good.
- The paper warns that unsupported steep trajectories extrapolate badly; the
  20-degree runtime envelope is therefore a hard first-milestone boundary.

These risks are measured directly. They are not addressed by reintroducing the
online hill IK or portal system into the PFNN runtime.
