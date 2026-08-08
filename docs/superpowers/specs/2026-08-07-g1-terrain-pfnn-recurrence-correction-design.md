# G1 Terrain PFNN Runtime-Equivalent Recurrence Correction

## Status

Approved design amendment for the native G1 terrain PFNN milestone. This
document narrows the correction required after Task 7's strict closed-loop
gate rejected three provisional checkpoints. It supplements, and does not
replace, the original terrain-PFNN design.

## Root cause

The closed-loop runtime and rollout fine-tuning do not currently construct the
same recurrent input.

At the first ordinary tick after the receipt-bound bootstrap, runtime calls
`_planned_trajectory`: it reconstructs seven past knots from realized root
history, blends predicted future interval velocities and facing with the user
command, sequentially integrates the future knots, and derives idle/walk
semantics from the planned speed. Training instead calls
`_physical_to_recurrent_input`: it directly copies all twelve predicted
trajectory knots, copies semantic intent from the next sealed row, and only
feeds back predicted body state.

The measured tick-1 difference was a normalized maximum of `4.61494303` in
trajectory direction, `0.58885330` in trajectory position, and `2.94555545`
in semantic intent. Body, terrain, and phase agreed within floating-point
tolerance. The raw model was safe on the phase-correct sealed next-row input,
but unsafe on the runtime-planned input. This isolates the failure to the
training/runtime recurrence boundary.

Stationary flat phase augmentation exposes a second coverage problem. Each
idle window has eight deterministic phase bins, but rollout discovery drops
every center with multiplicity greater than one. The bins are not mixed, but
entire stationary transitions disappear from recurrent training. The selected
runtime seed can therefore sit at a transition that the rollout sampler never
sees.

## Goals

- Use one differentiable implementation of trajectory planning in training
  and runtime.
- Preserve the existing paper-style velocity bias `tau_v = 0.5` and facing
  bias `tau_d = 2.0`.
- Preserve raw PFNN output: no IK, qpos correction, stance lock, portal,
  MotionBricks call, root projection, or safety-threshold relaxation.
- Preserve test and validation isolation.
- Train stationary augmented rows as distinct, phase-stable recurrent lanes.
- Prevent checkpoint promotion when any fitted adjacent transition violates a
  runtime physical envelope, even if its average normalized loss passes.
- Make one new controlled training attempt after the corrected dataset and
  recurrence pass all contract tests.

## Non-goals

- Adding additional semantic classes beyond idle and walk.
- Changing the `288 -> 512 -> 512 -> 268` PFNN architecture.
- Adding terrain queries to the differentiable training graph.
- Replacing the raw controller with physics, tracking, IK, or MotionBricks.
- Opening the sealed test split.
- Relaxing the exact Task 7 acceptance limits.

## Considered approaches

### Shared differentiable recurrence kernel — selected

Move trajectory/history planning into a small Torch implementation used by
both rollout training and runtime. Runtime invokes it under
`torch.inference_mode()`; training retains its graph through every unrolled
step. This gives the strongest parity guarantee and prevents future drift.

### Duplicate the runtime planner in training — rejected

A Torch copy could initially match the NumPy runtime, but the two versions
would be free to diverge. Parity tests would detect some drift but would not
provide a single semantic authority.

### Simplify runtime to direct predicted-trajectory feedback — rejected

This would match the current training helper, but it would remove the
paper-style command blend and make continuous steering ineffective. It solves
the wrong side of the mismatch.

## Shared recurrence architecture

Create a focused recurrence module whose public surface consists of immutable
tensor state plus pure functions. It owns trajectory planning and state
advancement; it does not own model evaluation, normalization, terrain queries,
collision checks, or checkpoint I/O.

The tensor state contains:

- current world root planar position and yaw;
- 31 realized history samples at 30 Hz: world position, world facing, and
  idle/walk semantic;
- the previous predicted twelve-knot world trajectory and facing;
- the previous predicted local body position and velocity; and
- current phase.

The 31-sample initial history is reconstructed deterministically from the
seven stored past knots using piecewise-linear interpolation at 30 Hz. The
same initializer is used by checkpoint seed construction, training, and
runtime.

For each tick, the planner:

1. selects past history samples at offsets `0, 5, 10, 15, 20, 25, 30`;
2. overwrites knot 6 with the realized current root;
3. computes predicted interval velocities for future knots 7 through 11;
4. blends velocity using `u**0.5` and facing using `u**2`, where future
   `u = [0.2, 0.4, 0.6, 0.8, 1.0]`;
5. sequentially integrates the blended velocities;
6. renormalizes every future facing vector; and
7. derives exact idle/walk one-hot semantics using the existing
   `0.05 m/s` stationary threshold.

The state-advance function integrates predicted planar and yaw velocity for
exactly `1/30 s`, transforms the predicted next trajectory into world space,
appends the realized root/facing/semantic history sample, retains predicted
body state, and advances bounded positive phase. Torch operations remain
differentiable during training.

Runtime converts its command and state to tensors, calls the same planner, and
converts only the planned positions/directions needed by the terrain callback
back to NumPy. No parallel NumPy implementation remains.

## Training command and terrain contract

The dataset does not contain the original keyboard command, so rollout
training derives a deterministic desired velocity from each sealed row:

- if future knot 11 is idle, desired velocity is zero;
- otherwise local desired velocity is
  `(trajectory_position[11] - trajectory_position[6]) /
  (trajectory_time[11] - trajectory_time[6])`; and
- it is rotated into world space by the recurrent state's current yaw.

The first rollout input remains the sealed row. Every later trajectory and
semantic slice is produced by the shared planner. Predicted body state,
trajectory state, root motion, yaw, and phase remain inside the differentiable
unroll. The next sealed training row supplies only exogenous terrain samples
and the command source used above; it never overwrites predicted recurrent
trajectory, semantics, body state, root state, or phase.

The planner operates on physical values. The existing shared normalization
packer remains the only normalization boundary, and previous-body position and
velocity retain their exact post-normalization `0.1` importance scale in both
training and runtime.

Training terrain stays exogenous and read-only. A future differentiable
terrain sampler is outside this milestone.

## Explicit sequence lanes

Add a required `sequence_lane` to `PFNNTrainingWindow`, dataset shards,
dataset row loading, fitted-row hashes, fitted-subset receipts, runtime-seed
provenance, and checkpoint validation.

- Ordinary phase-valid motion uses lane `motion`.
- An idle augmented row uses lane `idle_phase_0` through `idle_phase_7`.
- Mirrored data remains separated by its mirrored `clip_id` and retains the
  corresponding lane.

Rollout discovery groups by `(clip_id, sequence_lane)` and then requires exact
consecutive center frames. Gaps and transitions between `motion` and an idle
phase lane terminate a sequence. It is an error for two rows to share the same
`(clip_id, sequence_lane, center_frame)`.

Rollout sampling uses deterministic shuffled passes without replacement over
the valid sequences before beginning another pass. Resume restores the exact
sequence-sampler cursor. This prevents the runtime-seed transition or a small
terrain class from being starved by replacement sampling.

## Runtime seed rule

The checkpoint seed remains the lowest-speed valid flat training row. For this
corrected contract, `valid` means:

- finite and inside canonical joint limits;
- exact train split and sealed fitted-subset membership;
- a same-clip, same-lane predecessor at `center_frame - 1`;
- at least sixteen same-clip, same-lane consecutive rows beginning at the
  first fitted row; and
- a finite reconstructable 31-sample history.

Speed is the primary ordering key; clip, lane, center frame, and row digest are
deterministic tie breakers. The validity filter is applied before speed
ordering and is recorded in the seed receipt. The chosen transition is always
included in rollout fine-tuning.

## Physical fitted-transition gate

Average normalized error remains useful for optimization but is insufficient
for authorization. Before a provisional candidate can receive a pipeline
promotion receipt, evaluate every same-lane adjacent fitted transition at its
stored phase and validate the raw prediction against the predecessor's reached
state.

The gate requires:

- finite `268`-value raw output;
- root translation step `<= 0.060 m`;
- root rotation step `<= 0.35 rad`;
- every canonical joint step `<= 0.25 rad`;
- every predicted joint inside native limits;
- positive finite root height and normalized finite root quaternion;
- finite trajectory with direction norms in `[0.5, 1.5]` before
  renormalization; and
- bounded nonnegative phase advance under the existing q99 rule.

The report records the maximum and the first rejected clip, lane, center,
field, and value. Pipeline `best.pt` remains impossible unless fixed-sample
reproduction, this transition gate, and the checkpoint-bound 600-tick
known-terrain receipt all pass.

## Provenance and schema changes

The sealed dataset and checkpoint schemas advance once. Their digests include
the exact lane string for every row. Resume compares the newly materialized
row and lane receipts before restoring model, optimizer, or sampler state.
Older shards and checkpoints fail closed with an explicit schema error; they
are not silently inferred or migrated.

Generated replacements remain under `sonic/runs/terrain-pfnn-v1/` and stay
untracked. The prior failed candidates and receipts remain preserved for the
audit trail.

## Error handling

- Invalid lane metadata rejects dataset loading or checkpoint loading.
- Duplicate lane/frame rows reject rollout discovery.
- Missing consecutive seed coverage rejects seed construction.
- Nonfinite planner state or degenerate facing rejects the training batch and
  causes a transactional runtime hold.
- A fitted-transition failure rejects the candidate before closed-loop
  promotion.
- A closed-loop failure writes complete diagnostics but never publishes
  `best.pt`.

## Verification

TDD regressions must prove:

1. training and runtime produce the same tick-1 and multi-tick planned
   trajectory, semantic, history, root, body, and phase state from the same
   seed, prediction, terrain, and command;
2. gradients from a three-step rollout reach the first prediction through the
   shared planner;
3. idle phase bins form eight independent consecutive lanes and never cross;
4. gaps, mirror boundaries, and motion-to-idle transitions terminate lanes;
5. sequence sampling covers every valid sequence before replacement and
   resumes at the exact cursor;
6. runtime seed choice applies validity first and speed ordering second;
7. row, subset, resume, seed, checkpoint, and promotion receipts reject lane
   tampering;
8. a candidate with acceptable mean loss but one unsafe fitted joint step is
   rejected; and
9. runtime still rejects unsupported terrain and invalid raw output without
   mutating recurrent state.

After unit and integration tests pass:

1. rebuild the sealed two-family canary with explicit sequence lanes;
2. verify row counts, split isolation, hashes, and normalization from scratch;
3. run one controlled 2,048-row training attempt with 16-frame fine-tuning;
4. require exact fitted-sample reproduction and the physical transition gate;
5. run the exact 600-tick source-supported known-terrain verifier; and
6. promote immutable `best.pt` only if its bound receipt passes.

Task 7 then returns to independent review. Tasks 8 and 9 remain blocked until
that review is clean.

## Success criteria

- Shared recurrence parity tests pass at float32 tolerance with no divergent
  trajectory or semantic producer.
- The controlled candidate passes every fitted physical transition.
- The known-terrain run completes 600 finite committed ticks with zero holds,
  satisfies all applicable raw safety gates, and proves realized supported
  traversal and return.
- No test identity contributes to rebuilding, fitting, normalization,
  fine-tuning, seed selection, or promotion.
- No runtime correction, IK, lock, portal, MotionBricks call, or threshold
  relaxation is introduced.
