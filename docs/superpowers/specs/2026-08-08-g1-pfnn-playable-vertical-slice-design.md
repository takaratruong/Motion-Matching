# G1 PFNN Playable Vertical Slice Design

## Goal

Build the shortest honest path from the approved offline PFNN-to-G1 terrain
transfer to a joystick-controlled G1 demo. The first controller covers idle,
walking, turning, uphill walking, and downhill walking. It uses released PFNN
motions, phases, gait annotations, footsteps, and terrain-fitting semantics.

This milestone proves the PFNN data and controller loop before expanding to the
complete PFNN corpus or adding GRAIL.

## Selected Approach

Use the existing native-G1 PFNN implementation as the execution shell, but
replace its current training source with a small corpus retargeted from the
released PFNN dataset. Preserve the characteristic PFNN architecture: four
phase control points, cubic interpolation, two 512-unit ELU hidden layers, and
one-step autoregressive prediction at 30 Hz.

This is preferable to two alternatives:

- Driving the released human PFNN and retargeting every output online would add
  latency and recurrent retargeting errors, and would not produce a native G1
  controller.
- Building the full PFNN plus GRAIL corpus immediately would make failures slow
  to diagnose and repeat the earlier problem of training before the data path
  was visually proven.

## Scope

The vertical slice includes:

- idle-to-walk and walk-to-idle;
- forward walking;
- continuous left and right turning;
- ascending and descending the released PFNN terrain;
- a keyboard/joystick-controlled MuJoCo viewer.

It excludes running, jumping, crouching, crawling, beam balancing, GRAIL,
physics-policy training, IK, foot locking, and post-inference pose correction.
Those are later milestones.

## Source Selection

The source adapter scans only non-mirrored released PFNN clips and their
matching `.phase`, `.gait`, and `_footsteps.txt` files. Before retargeting, it
selects a deterministic minimal set containing:

- one predominantly straight flat locomotion clip;
- one flat clip whose root path contains both left and right turns;
- two `WalkingUpSteps` clips whose valid footstep cycles collectively contain
  both ascent and descent.

Selection is based on source-space root displacement, accumulated yaw, and the
signed grade of the fitted terrain along travel. The selected filenames,
intervals, source hashes, and coverage measurements are sealed in a receipt.
If four clips cannot provide all required coverage, selection fails rather
than silently adding unrelated locomotion modes.

Mirroring happens once after feature extraction using the existing G1 mirror
contract. Pre-mirrored PFNN BVHs are excluded to prevent duplicate examples.

## Offline Retargeting and Terrain Transfer

Each selected BVH is retargeted offline using the already approved GMR mapping:

- native solve rate: 120 Hz;
- G1 lower-body/root morphology factor: `0.875`;
- no per-frame grounding, IK, or foot locking;
- output: G1 root pose and all 29 joint positions.

The original `.phase`, `.gait`, and footstep timelines are sampled at their
corresponding source times. Training frames are then sampled from the retargeted
motion at 30 Hz; retargeting is never performed on the 30 Hz result.

For terrain cycles, the adapter uses PFNN's original patch selection,
horizontal alignment, vertical alignment, and linear RBF stance residual. The
fitted surface is transformed into the G1 world with uniform XYZ scale `0.875`
and the approved fixed world-Z placement offset
`0.05224985936713168 m`. The offset is not re-fitted per clip. Every terrain
artifact retains its source patch index, cycle, hashes, scale, and offset.

Flat clips use a sealed flat surface at the same G1 support convention.

## Feature and Target Contract

Reuse the existing native-G1 `288 -> 268` feature seam and its 12 trajectory
knots:

- input: local trajectory positions and directions, three-track terrain
  heights, idle/walk intent, previous G1 body positions, and body velocities;
- phase: the released PFNN phase, expressed in the existing runtime convention;
- output: next trajectory, G1 body positions and velocities, root height and
  tilt, 29 joint positions, root planar/yaw motion, phase advance, and four
  heel/toe contacts.

This adapts only morphology-dependent pose fields. It preserves the PFNN
trajectory, phase-conditioning, terrain-track, and one-step recurrence design.
The dataset builder must demonstrate that a serialized training row reconstructs
the approved sample motion and terrain within binary32 tolerance before any
model training begins.

## Training

Train the existing classic G1 PFNN with the original-style one-step objective:
mean-squared error for continuous outputs and binary cross-entropy for contacts.
Use the four phase slices and cubic interpolation already implemented in
`PhaseFunctionedNetwork`.

The small corpus is split by source clip, never by frame. Training mirrors each
training clip exactly once; validation remains unmirrored. Checkpoints bind the source
receipt, retarget receipt, terrain receipts, dataset digest, normalization,
joint order, and kinematic signature.

There is no physical-envelope surrogate, output clamping, runtime pose repair,
or teacher-forced correction in this milestone. A model that cannot run the
closed loop is rejected rather than cosmetically fixed.

## Runtime and Viewer

Reuse the existing closed-loop `TerrainPFNNRuntime` and MuJoCo G1 viewer. Replace
the synthetic hill scene with an exact scaled PFNN terrain surface from the
vertical-slice corpus. The runtime samples its three-track terrain observation
from that same surface.

Controls are:

- forward/back command magnitude controls idle versus walking speed;
- left/right controls desired facing and turning;
- Space pauses, Home resets, and Escape closes.

The viewer displays the terrain, G1, current phase, desired trajectory, sampled
terrain tracks, and heel/toe contact markers. It does not simulate dynamics;
this is the same kinematic-controller level as the original PFNN demo.

## Failure Handling

The pipeline fails closed before training for missing annotations, source hash
mismatch, invalid phases, missing footstep terrain cycles, retargeted joint-limit
violations, nonfinite features, unavailable terrain samples, or insufficient
mode/grade coverage.

At runtime, nonfinite output, unsupported terrain, or a motion-envelope breach
halts and reports the exact frame and field. It must not hold the last pose for
the remainder of the demo while appearing healthy.

## Acceptance Gates

The milestone is accepted only when all of the following pass:

1. The four selected clips and their receipts deterministically reproduce.
2. At least one flat, turning, ascent, and descent interval is present in both
   the raw source audit and the built training rows.
3. The approved PFNN transfer sample reconstructs from a dataset row with the
   same G1 pose and terrain query within binary32 tolerance.
4. Offline playback of each selected retarget has no twisting, nonfinite pose,
   or joint-limit violation.
5. A fresh checkpoint reloads exactly and completes a closed-loop scripted
   route containing start, straight walk, left turn, right turn, ascent,
   descent, and stop without a held or invalid frame.
6. The user can drive the same viewer interactively and visually confirms that
   G1 follows the terrain comparably to the approved offline transfer.

Before presenting the viewer, the implementation owner must personally operate
the controls, inspect the live scene, and capture the exact run receipt. A
passing one-step loss alone is not sufficient.

## Follow-on Work

After this vertical slice is approved:

1. expand to the remaining suitable PFNN walking/turning/terrain clips;
2. add verified GRAIL G1 motion/terrain pairs without morphology scaling;
3. retrain with explicit source balancing;
4. add running and other gait modes only after their source coverage is proven.
