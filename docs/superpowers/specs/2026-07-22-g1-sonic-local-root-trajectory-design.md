# G1 SONIC Local Root-Trajectory Conditioning Design

## Status

Approved conversational design, awaiting written-spec review.

## Purpose

The terrain-aware Motion Matching to SONIC baseline established two different
facts:

1. The current composition supports responsive, steerable locomotion on flat
   ground, including direction reversals and capped turning.
2. It does not reliably traverse the certified `grail-curb-low` scene. In the
   deterministic straight-forward trial, the robot fell even though Motion
   Matching generated a terrain-varying physical pelvis trajectory.

The immediate question is therefore narrower than depth perception or online
latency:

> Does conditioning SONIC on Motion Matching's local pelvis trajectory make
> the already-generated terrain-aware kinematics trackable across a low curb?

This experiment retains privileged terrain and paused-physics reference
generation. It does not yet replace terrain with depth. If local root
conditioning cannot improve this idealized case, perception and real-time
latency experiments are premature.

## Corrected SONIC contract

### What the released G1 checkpoint consumes

The released G1 motion encoder consumes ten future samples at 0.1-second
spacing of:

- 29 joint positions;
- 29 joint velocities; and
- one six-dimensional robot-relative root orientation per sample.

The actor receives the resulting 64-dimensional token together with ten-frame
histories of measured joint positions, measured joint velocities, base angular
velocity, gravity direction, and previous actions. It does not receive measured
global root position or measured base linear velocity.

Root positions are present in SONIC motion data and root-position errors are
used by training rewards. Those facts do not make root translation a command
input to the released G1 encoder. The shipped deployment observation superset
also contains root-z fields, but G1 mode excludes them, and the shipped ONNX
encoder is insensitive to them.

The following direct ONNX test is part of the baseline evidence and must be
retained as a regression test:

- across twenty random G1 inputs, perturbing all eleven root-z slots by values
  up to plus or minus 100 changes the 64-dimensional output token by exactly
  zero;
- perturbing the active joint-position slots changes the token by as much as
  0.75.

Consequently, adding `body_pos` to the stream or adding root-z names to the
G1 mode list cannot by itself change the current checkpoint's behavior.

### Canonical local root trajectory

SONIC training code contains a canonical root-transform representation. For a
future window with physical pelvis positions `p_i` and root quaternions `q_i`,
let:

- `p_0` be the first position in the current future window;
- `h_0 = heading(q_0)` be the yaw-only quaternion of the first root
  orientation; and
- `project_xy(p_0) = [p_0.x, p_0.y, 0]`.

The local root-position command for future sample `i` is:

```
r_i = inverse(h_0) * (p_i - project_xy(p_0))
```

Thus:

- `r_i.x` and `r_i.y` are translation-invariant local displacements from the
  first future frame;
- `r_i.z` is the physical pelvis height, because the reference position is
  projected to z equals zero; and
- `r_0 = [0, 0, p_0.z]`.

No measured global robot XY position is needed. Adding a constant XY
translation to every streamed frame must leave every local command unchanged.
Applying one global yaw to all positions and orientations must likewise leave
the command unchanged after canonicalization.

The first experiment adds only the three position components to the existing
G1 encoder. It retains the existing robot-relative six-dimensional orientation
input instead of duplicating the canonical rotation portion of SONIC's full
nine-dimensional root-transform representation. This is the smallest change
that supplies local delta x, local delta y, and pelvis height while preserving
the existing orientation-error signal.

### Which Motion Matching root to use

The source is the resampled physical pelvis position produced by G1 forward
kinematics after Motion Matching pose projection, not the planar virtual root.
The physical pelvis is the root pose coherent with the 29 joint targets and is
the equivalent of SONIC motion library `body_pos_w[..., 0, :]`. The virtual
root remains a navigation and diagnostic trajectory only.

The existing Holden-to-MuJoCo proper coordinate conversion is applied before
the local trajectory is formed. Canonicalization is performed on float32
target-buffer frames after resampling so the offline reference, online stream,
training data, and deployed observation use one definition.

## Selected approach

Implement a root-conditioned G1 encoder and a versioned root-capable streaming
contract. Initialize the new encoder so that it exactly reproduces the released
G1 encoder before training, then teach the new channels through motion
reconstruction and terrain evaluation.

This approach is selected over two alternatives:

1. **Protocol-only root plumbing.** This is necessary infrastructure, but the
   frozen checkpoint provably ignores the fields and therefore cannot test the
   behavioral hypothesis.
2. **An external global-position servo or reuse of the flat-ground SONIC
   kinematic planner.** A servo would require the root state the eventual real
   system does not reliably have, while the planner would replace rather than
   evaluate the terrain-aware Motion Matching reference.

## Architecture

### 1. Versioned Motion Matching stream

Add one versioned joint-reference protocol revision rather than changing
protocol v1 in place. The new message carries all existing v1 fields plus:

```
body_pos: f32 [N, 3]
```

`body_pos` is root-only physical pelvis position in MuJoCo's z-up coordinate
system. It must have the same row count as joint positions, joint velocities,
root quaternions, and frame indices. Every value must be finite.

The bridge serializes `TargetChunk.physical_pelvis_position`; it does not
reconstruct root translation from joint values or substitute
`virtual_root_position`. The reference-directory sink writes the same rows to
root-only `body_pos.csv`. Root positions remain available in the diagnostic
CSV as before so the wire/reference rows can be compared exactly.

Protocol negotiation must reject an older GEAR runtime when the root-aware
checkpoint is selected. The current v1 path remains available only for the
frozen baseline checkpoint.

### 2. GEAR stream merger

The GEAR receiver validates and owns the incoming root-position array. The
stream merger copies each row into `MotionSequence::BodyPositions(frame)[0]`
alongside the already-copied root quaternion. It may no longer leave
`BodyPositions` at zero for a root-capable stream.

Catch-up, sliding-window replacement, frame holding, and stream reset apply to
root positions atomically with the other reference fields. A malformed or
partially mismatched root array rejects the complete incoming message; it must
not publish a mixed old/new motion sequence.

### 3. Local-trajectory gatherer

Add a deployment observation named for its exact sampling contract, initially:

```
motion_root_position_refheading_10frame_step5
```

It contributes 30 values. At the current stream frame it gathers frames
`t, t+5, ..., t+45`, clamps only according to the existing future-window hold
contract, and computes the canonical `r_i` formula above using frame `t` as
`p_0, q_0`.

The gatherer must:

- use the reference root quaternion's yaw only;
- preserve physical pelvis z;
- be invariant to constant world XY translation;
- be equivariant before canonicalization and invariant afterward to a common
  world yaw;
- reject non-finite inputs rather than zero-fill an active root-aware mode;
- expose its offset and dimension in startup diagnostics; and
- have an exact Python training counterpart tested against the C++ result.

### 4. Root-conditioned G1 encoder

The new G1 encoder retains the released checkpoint's two existing logical input
terms without changing their internal or flattened ordering:

```
command_multi_future_nonflat
motion_anchor_ori_b_mf_nonflat
```

It adds one separate `[N, 10, 3]` term:

```
motion_root_position_refheading_mf_nonflat
```

Logically, each future sample is described by 29 joint positions, 29 joint
velocities, six robot-relative root-orientation values, and three canonical
local root-position values. Across ten samples this is 670 motion-command
values. The implementation must preserve the released checkpoint's exact
flattened ordering and deployment offsets for every existing input; it must not
reinterpret the old 640 values as a newly interleaved per-frame array. The new
30-value term receives its own explicit, tested range in the exported model's
observation superset.

The first layer is initialized by copying every released G1 input weight into
the matching old column and setting all new root-position columns to exactly
zero. All later G1 encoder, quantizer, and control-decoder parameters begin from
the released checkpoint. Before any optimizer step, a protected parity test
must show that the new checkpoint produces the same token and action as the old
checkpoint for the same active old inputs, within the exporter/runtime numeric
tolerance.

The checkpoint and observation configuration are a pair. Runtime startup must
hash and report both and fail if their dimensions or required observation
names disagree.

### 5. Training stages

Training is deliberately staged to find the smallest sufficient change.

#### Stage A: representation adaptation

- Freeze the control decoder/policy.
- Train the root-conditioned G1 encoder and G1 kinematic motion decoder.
- Preserve the existing joint-position, joint-velocity, and orientation
  reconstruction objectives.
- Add reconstruction for the same 10-sample canonical local root positions.
- Retain token-alignment/cycle objectives needed to keep the result in SONIC's
  existing universal token space.
- Mix ordinary SONIC motions with generated terrain-aware MM clips so the new
  channel sees both normal locomotion and non-flat pelvis trajectories.

This is the fast first attempt. It asks whether the existing control decoder
already contains sufficient motor behavior and only needs an unambiguous root
trajectory token.

#### Stage B: physics fine-tuning, only if required

If Stage A fails the low-curb behavioral gate despite passing representation
tests, unfreeze the smallest required portion of the control decoder and run
PPO fine-tuning on terrain-aware references in simulation. The actor still
receives only deployment-available proprioception plus the root-conditioned
token. Simulation root state and terrain may be used by rewards and the critic,
but may not enter actor observations.

Stage B is a separate checkpoint and result. It must not be silently folded
into Stage A, because whether encoder-only adaptation suffices is useful
research evidence.

## Dataset and reference contract

Every training/evaluation sample records:

- the 29 joint positions and velocities;
- physical pelvis position and quaternion;
- canonical local root-position window;
- source scene, route, terrain mode, and MM artifact identities;
- source and resampled frame indices;
- whether the sample came from ordinary SONIC motion or terrain-aware MM; and
- all coordinate and joint-order manifest hashes.

Canonical local root positions are derived, never independently authored. The
dataset validator recomputes them from physical pelvis rows and fails on any
mismatch. Global-translation and global-yaw augmentation is applied before
canonicalization; the post-canonical values must remain invariant.

The initial terrain set contains at least flat walking and the registered
`grail-curb-low` straight route. It may include additional current scenes for
training diversity, but they do not replace the fixed evaluation route.

## Experiment matrix

Run the same deterministic straight-forward Motion Matching command and initial
condition for:

1. **A0: released checkpoint, v1 stream.** Existing baseline.
2. **A1: released checkpoint, root-capable stream.** Root data is transported,
   but the checkpoint remains unchanged. Tokens and behavior should match A0;
   this is the negative-control proof that plumbing alone is inert.
3. **B: Stage-A root-conditioned checkpoint.** Encoder/motion-decoder
   adaptation with frozen control decoder.
4. **C: Stage-B physics-fine-tuned checkpoint.** Run only if B fails the
   behavioral gate.

The exact same resampled reference buffer is replayed across conditions. No
manual steering is accepted as the primary comparison.

## Acceptance gates

### Contract and representation

- Protocol round trips root positions bit-exactly as float32.
- File and stream sinks contain identical root rows.
- C++ and Python canonicalization agree within `1e-6` per component.
- Constant global XY translation changes canonical output by at most `1e-6`.
- Common global yaw changes canonical output by at most `1e-6`.
- Root-aware checkpoint startup proves exact model/config dimensions and hashes.
- Before training, expanded-checkpoint token/action parity passes against the
  released checkpoint.
- After training, changing physically plausible local delta x, delta y, or z
  produces a nonzero token response; a protected sensitivity test reports the
  response norm per channel.

### Flat-ground preservation

The root-conditioned checkpoint must complete the existing deterministic flat
forward, reversal, lateral, and capped-turn probes without a new fall or a
material regression in command-response timing. Exact thresholds are copied
from the already registered responsive-control probes rather than invented by
the training run.

### Primary terrain gate

On `grail-curb-low`, route `curb-forward`, the first checkpoint to claim
success must complete three consecutive deterministic straight-forward trials:

- no fall or freeze;
- positive route progress through the curb corridor;
- no forbidden robot-terrain penetration;
- physics and policy processes remain healthy through trial end; and
- the robot passes the route's registered completion boundary.

All three trials must use the exact pinned MM, GEAR, model, configuration,
scene, route, and command identities.

After passing, run the same checkpoint on `grail-curb-default` as a stress test.
That scene is reported separately and does not retroactively change the
certified low-curb verdict.

## Timing evidence

This experiment retains the 50 Hz SONIC control loop and the existing ten
future samples at 0.1-second spacing. Record:

- root-aware encoder inference p50, p95, p99, and maximum;
- complete control-loop p50, p95, p99, and maximum;
- stream arrival-to-consumption age;
- MM generation time and paused-physics duration; and
- missed or held reference frames.

Timing is diagnostic in this phase. It establishes the cost of root
conditioning but does not yet impose real-world deadlines or action chunking.

## Failure interpretation

- **A0 and A1 differ:** the protocol migration changed an existing field or
  timing behavior; fix the integration before training conclusions.
- **B has zero root sensitivity:** the new input is masked, pruned, or not
  trained; the checkpoint is invalid.
- **B is root-sensitive but fails while C passes:** SONIC required physics
  adaptation, not merely a better token encoder.
- **C fails the fixed low curb:** local reference root conditioning is
  insufficient; inspect contact feasibility, action capacity, and reference
  dynamics before introducing depth.
- **B or C passes:** proceed to replace privileged terrain features with the
  proposed proprioception-plus-depth model while retaining the same root
  trajectory and evaluation contracts.

## Non-goals

This design does not implement:

- depth, lidar, height-map reconstruction, or perception training;
- measured global robot XY as an actor input;
- online correction of Motion Matching from estimated global position;
- obstacle avoidance beyond the current operator assumption;
- real-time MM deadlines, artificial latency sweeps, or action chunking;
- PFNN, behavior cloning, DAgger, or a learned replacement for Motion Matching;
- physical-robot deployment; or
- the later pick-carry-place and shelf tasks.

Those remain subsequent hypotheses after the local-root terrain gate.

## Repository isolation and supervision

Motion Matching changes stay on the existing isolated
`research/g1-low-latency-driver` branch until this experiment receives a
dedicated implementation branch/worktree. GEAR changes start from clean pinned
commit `294110cedba01ad764f1e268d57ddf7c1bbf9523` in a new isolated worktree.
Training configurations and checkpoints use explicit immutable identities.

Reliable Claude may implement bounded tasks only after an implementation plan
and protected verifiers exist. Codex owns the frozen contract, reviews all
diffs, independently runs the gates, and performs the visible low-curb canary.
A Reliable Claude `passed` result is candidate evidence, not acceptance.
