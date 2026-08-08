# G1 Terrain PFNN Runtime-Equivalent Recurrence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PFNN rollout fine-tuning construct exactly the same recurrent trajectory, semantic, body, root, and phase state as the controllable runtime, then authorize a checkpoint only after raw fitted-transition and 600-tick terrain gates pass.

**Architecture:** A pure differentiable Torch recurrence module becomes the single trajectory/history planner for both training and runtime. Sealed dataset rows gain explicit motion/idle phase lanes, rollout sampling follows those lanes without replacement, and checkpoint promotion gains a raw physical transition gate before the existing source-supported closed-loop verifier.

**Tech Stack:** Python 3.10+, NumPy, PyTorch 2.13, MuJoCo 3.11, `unittest`, sealed NPZ/JSON manifests, NVIDIA L40S GPU.

## Global Constraints

- Runtime and dataset frame rate is exactly `30 Hz`.
- Input contains exactly `288` float values; output contains exactly `268` float values.
- Phase is separate from the input vector and selects the existing cyclic four-bank cubic PFNN.
- Runtime planning keeps `tau_v = 0.5`, `tau_d = 2.0`, future `u = [0.2, 0.4, 0.6, 0.8, 1.0]`, and the existing `0.05 m/s` stationary threshold.
- Previous-body channels retain the exact post-normalization `0.1` scale.
- Canonical joint order remains `ISAACLAB_JOINT_NAMES`; root quaternions remain WXYZ.
- Runtime terrain remains center and `+/-0.25 m` lateral at 12 trajectory knots.
- Training may read next-row terrain and derive a command from next-row trajectory/semantic, but may not overwrite predicted recurrent trajectory, semantics, body, root, yaw, or phase.
- Test identities never contribute to data normalization, fitting, rollout fine-tuning, seed choice, transition gates, early stopping, or checkpoint promotion.
- Runtime contains no MotionBricks call, portal, IK, stance lock, qpos correction, root projection, or relaxed safety threshold.
- The author PFNN checkout remains read-only; no source is copied from it.
- Generated datasets, checkpoints, receipts, and reports remain below `sonic/runs/terrain-pfnn-v1/` and untracked.
- Older dataset/checkpoint schemas fail closed; they are not inferred or silently migrated.

---

### Task 1: Add the shared differentiable recurrence and adopt it in runtime

**Files:**
- Create: `sonic/python/mm_sonic/terrain_pfnn/recurrence.py`
- Create: `tests/python/test_terrain_pfnn_recurrence.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/__init__.py`
- Test: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Consumes: physical trajectory/body slices, current phase, world root pose, desired world velocity, and physical PFNN output.
- Produces: `RecurrentTrajectoryState`, `PlannedTrajectory`, `initialize_recurrent_state`, `derive_training_desired_velocity`, `plan_recurrent_trajectory`, `pack_recurrent_input`, and `advance_recurrent_state`.

- [ ] **Step 1: Write RED tests for deterministic planning and state advancement**

Create tensor fixtures with batch size two, zero yaw, 31 stationary history samples,
constant predicted future velocity, and commands `(0,0)` and `(0.3,0)`. Assert:

```python
planned = plan_recurrent_trajectory(state, desired_velocity_world)
self.assertEqual(planned.position_world_xy.shape, (2, 12, 2))
self.assertEqual(planned.direction_world_xy.shape, (2, 12, 2))
self.assertEqual(planned.semantic_intent.shape, (2, 12, 2))
torch.testing.assert_close(planned.position_world_xy[:, 6], state.root_world_xy)
torch.testing.assert_close(
    torch.linalg.vector_norm(planned.direction_world_xy[:, 7:], dim=-1),
    torch.ones((2, 5), dtype=torch.float64),
)
self.assertEqual(planned.semantic_intent[0, 11].tolist(), [1.0, 0.0])
self.assertEqual(planned.semantic_intent[1, 11].tolist(), [0.0, 1.0])
```

Use a physical output with root velocity `(0.3, 0)`, yaw velocity `0.6`, and
phase advance `0.1`. Assert exactly `0.01 m`, `0.02 rad`, and `0.1 rad` state
advances at 30 Hz, that history shifts by one sample, and that predicted local
trajectory is transformed by the new root/yaw frame.

- [ ] **Step 2: Run the new recurrence tests and verify RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_recurrence
```

Expected: import failure for `mm_sonic.terrain_pfnn.recurrence`.

- [ ] **Step 3: Implement the immutable state and pure Torch planner**

Create these public dataclasses and functions:

```python
@dataclass(frozen=True)
class RecurrentTrajectoryState:
    root_world_xy: torch.Tensor                 # [B,2]
    root_yaw_world: torch.Tensor                # [B]
    history_position_world_xy: torch.Tensor     # [B,31,2]
    history_direction_world_xy: torch.Tensor    # [B,31,2]
    history_semantic_intent: torch.Tensor       # [B,31,2]
    predicted_position_world_xy: torch.Tensor   # [B,12,2]
    predicted_direction_world_xy: torch.Tensor  # [B,12,2]
    previous_body_position_local: torch.Tensor  # [B,30,3]
    previous_body_velocity_local: torch.Tensor  # [B,30,3]
    phase: torch.Tensor                         # [B]

@dataclass(frozen=True)
class PlannedTrajectory:
    position_world_xy: torch.Tensor             # [B,12,2]
    direction_world_xy: torch.Tensor            # [B,12,2]
    semantic_intent: torch.Tensor               # [B,12,2]

```

Add the exact public signatures below; every tensor includes a leading batch
dimension:

- `initialize_recurrent_state(*, trajectory_position_local: Tensor,
  trajectory_direction_local: Tensor, semantic_intent: Tensor,
  previous_body_position_local: Tensor, previous_body_velocity_local: Tensor,
  phase: Tensor, root_world_xy: Tensor, root_yaw_world: Tensor)
  -> RecurrentTrajectoryState`
- `derive_training_desired_velocity(trajectory_position_local: Tensor,
  semantic_intent: Tensor, root_yaw_world: Tensor) -> Tensor`
- `plan_recurrent_trajectory(state: RecurrentTrajectoryState,
  desired_velocity_world: Tensor) -> PlannedTrajectory`
- `pack_recurrent_input(*, state: RecurrentTrajectoryState,
  planned: PlannedTrajectory, terrain_height: Tensor,
  x_mean: Tensor, x_std: Tensor, body_scale: float = 0.1) -> Tensor`
- `advance_recurrent_state(state: RecurrentTrajectoryState,
  planned: PlannedTrajectory, physical_output: Tensor,
  *, phase_advance_cap: Tensor) -> RecurrentTrajectoryState`

Validate exact shapes, a shared floating dtype/device, finiteness, normalized
input directions, and one-hot semantics. Reconstruct 31 history samples by
linear interpolation between the seven past knots; normalize interpolated
directions and choose semantic one-hot by argmax. Use differentiable tensor
operations and no `.detach()` inside planning or state advancement.

- [ ] **Step 4: Refactor runtime to call the shared planner**

Delete the independent `_history_trajectory` and `_planned_trajectory` math.
Initialize the shared state from the checkpoint seed, call
`plan_recurrent_trajectory` before terrain sampling, and call
`advance_recurrent_state` only after every raw-output and terrain validation
passes. Convert planned positions/directions to NumPy only at the terrain
callback seam. A held tick must retain the prior immutable tensor state.

- [ ] **Step 5: Add a runtime/shared-kernel parity regression**

In `test_terrain_pfnn_runtime.py`, capture the planned tick-1 trajectory from a
fake model and compare it against a direct shared-kernel call using the same
seed, prediction, and command:

```python
np.testing.assert_allclose(captured_x_tick1, direct_x_tick1, atol=3e-6, rtol=0.0)
self.assertEqual(runtime.frame.diagnostics.get("hold_reason"), None)
```

Also retain the existing test proving the second input contains the first raw
prediction and the transactional-hold tests.

- [ ] **Step 6: Run focused GREEN tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_runtime
```

Expected: all tests pass with pristine output.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/recurrence.py \
  sonic/python/mm_sonic/terrain_pfnn/runtime.py \
  sonic/python/mm_sonic/terrain_pfnn/__init__.py \
  tests/python/test_terrain_pfnn_recurrence.py \
  tests/python/test_terrain_pfnn_runtime.py
git commit -m "feat: share PFNN recurrent trajectory planning"
```

---

### Task 2: Seal explicit motion and idle phase sequence lanes

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/features.py`
- Modify: `sonic/python/mm_sonic/build_terrain_pfnn_dataset.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/dataset.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Test: `tests/python/test_terrain_pfnn_features.py`
- Test: `tests/python/test_terrain_pfnn_dataset.py`
- Test: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Consumes: accepted `PFNNTrainingWindow` values and sealed shard/manifest provenance.
- Produces: required `sequence_lane: str`, dataset schema `mm-sonic-terrain-pfnn-dataset/v2`, shard field `sequence_lane` with dtype `<U16`, and lane-bound row/subset/seed receipts.

- [ ] **Step 1: Write RED feature tests for lane assignment**

Assert ordinary motion emits only lane `motion`; a stationary flat center emits
exactly lanes `idle_phase_0` through `idle_phase_7`; mirroring preserves the
lane while changing only the clip identity; invalid lane strings are rejected.

```python
self.assertEqual({row.sequence_lane for row in moving.windows}, {"motion"})
self.assertEqual(
    {row.sequence_lane for row in idle.windows},
    {f"idle_phase_{index}" for index in range(8)},
)
```

- [ ] **Step 2: Run feature tests and verify RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_features
```

Expected: `PFNNTrainingWindow` has no `sequence_lane`.

- [ ] **Step 3: Add the required lane to windows**

Add `sequence_lane: str` without a default. Validate exactly `motion` or
`idle_phase_0` through `idle_phase_7`. `_pack_window` produces `motion`; the
idle augmentation loop replaces both `phase` and `sequence_lane`; mirroring
retains the lane. Update every test fixture constructor explicitly.

- [ ] **Step 4: Write RED dataset tests for lane sealing and tampering**

Add tests that build a tiny shard, assert `<U16`, load the exact lane, and
reject each of:

```python
tampered["sequence_lane"][0] = "idle_phase_7"
tampered_manifest["schema"] = "mm-sonic-terrain-pfnn-dataset/v1"
del tampered["sequence_lane"]
```

Also assert dataset digest changes if only a row lane changes.

- [ ] **Step 5: Implement dataset schema v2**

Add `sequence_lane` to `_SHARD_FIELDS`, shard array construction, exact
shape/dtype validation, row loading, duplicate-window identity, provenance
hashing, normalization-independent dataset digest, and resume validation.
Require manifest schema `mm-sonic-terrain-pfnn-dataset/v2`; v1 fails before a
shard is opened.

- [ ] **Step 6: Bind lane into fitted receipts and checkpoint seed provenance**

Include lane in `fitted_row_sha256`, fitted-row keys, fitted-subset metadata,
runtime-seed predecessor/first-row records, resume-subset comparison, and
checkpoint validation. Advance the checkpoint schema to
`mm-sonic-terrain-pfnn-checkpoint/v5`. Add tamper tests for row, seed, subset,
resume, and checkpoint lanes.

- [ ] **Step 7: Run focused GREEN tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_dataset \
  tests.python.test_terrain_pfnn_training
```

Expected: all tests pass; old schemas fail with explicit schema errors.

- [ ] **Step 8: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/features.py \
  sonic/python/mm_sonic/build_terrain_pfnn_dataset.py \
  sonic/python/mm_sonic/terrain_pfnn/dataset.py \
  sonic/python/mm_sonic/terrain_pfnn/training.py \
  tests/python/test_terrain_pfnn_features.py \
  tests/python/test_terrain_pfnn_dataset.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "feat: seal PFNN recurrent sequence lanes"
```

---

### Task 3: Fine-tune with the runtime-equivalent recurrence and deterministic lane sampling

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Test: `tests/python/test_terrain_pfnn_training.py`
- Test: `tests/python/test_terrain_pfnn_recurrence.py`

**Interfaces:**
- Consumes: shared recurrence functions, v2 dataset rows, `(clip_id, sequence_lane, center_frame)` identities, and sealed exogenous terrain/command rows.
- Produces: runtime-equivalent `autoregressive_unroll`, lane-safe `_consecutive_starts`, and resumable without-replacement `DeterministicSequenceSampler`.

- [ ] **Step 1: Write RED training/runtime recurrence parity tests**

Construct a three-row synthetic sequence and a deterministic fake PFNN. Run
`autoregressive_unroll` and direct shared recurrence for three ticks. Assert
every normalized trajectory, semantic, terrain, body, and phase slice matches
at each tick within `3e-6`, and assert a loss on tick 3 gives a finite nonzero
gradient on the tick-1 prediction.

```python
for trained, direct in zip(result.inputs, direct_inputs):
    torch.testing.assert_close(trained, direct, atol=3e-6, rtol=0.0)
result.losses["total"].backward()
self.assertGreater(float(first_prediction.grad.abs().sum()), 0.0)
```

- [ ] **Step 2: Run the parity tests and verify RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training
```

Expected: the existing training recurrence diverges in trajectory/semantic slices.

- [ ] **Step 3: Replace `_physical_to_recurrent_input` with shared recurrence**

Initialize recurrent state from the first physical input and phase. At each
later step:

```python
desired_velocity = derive_training_desired_velocity(
    next_physical_trajectory,
    next_physical_semantic,
    recurrent_state.root_yaw_world,
)
planned = plan_recurrent_trajectory(recurrent_state, desired_velocity)
current_input = pack_recurrent_input(
    planned=planned,
    terrain_height=next_physical_terrain,
    previous_body_position=recurrent_state.previous_body_position_local,
    previous_body_velocity=recurrent_state.previous_body_velocity_local,
    normalization=normal,
    body_scale=0.1,
)
```

After model evaluation, denormalize the prediction and call
`advance_recurrent_state`. Retain next-row terrain and command source only.
Remove `_physical_to_recurrent_input` so no alternative feedback path remains.

- [ ] **Step 4: Write RED lane-discovery tests**

Create two clips with motion and eight idle lanes. Assert exact consecutive
sequences within each lane, no motion-to-idle edge, no mirror edge, no gap
crossing, and hard failure on a duplicate `(clip,lane,center)`.

- [ ] **Step 5: Group rollout sequences by clip and lane**

Change `_consecutive_starts` to key rows by `(clip_id, sequence_lane)` and
center. Do not discard multiplicity: reject duplicate lane/frame rows. Require
train split, canonical identity, exact lane grammar, and exact consecutive
centers.

- [ ] **Step 6: Write RED deterministic sampler tests**

For seven sequences, batch size three, and a fixed seed, assert the first seven
draws contain every sequence exactly once, the next draw begins the next
deterministic permutation, and save/load resumes at the exact next index.

- [ ] **Step 7: Implement `DeterministicSequenceSampler`**

The sampler owns seed, epoch/permutation number, and cursor. It generates a
seeded permutation without replacement, consumes it fully, then advances the
permutation number. Serialize exact primitive state in checkpoints and validate
it before restore. DDP partitions each global batch deterministically without
changing global order.

- [ ] **Step 8: Tighten runtime-seed validity and inclusion**

Choose among flat rows having a same-lane predecessor and at least sixteen
same-lane consecutive rows starting at the first fitted row. Apply this filter
before sorting by `(speed, clip_id, sequence_lane, center_frame, row_sha256)`.
Assert the chosen sixteen-frame sequence exists in rollout discovery and is
visited during the first without-replacement pass.

- [ ] **Step 9: Run focused GREEN and resume tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training
```

Expected: all tests pass, including three-step gradients and exact sampler resume.

- [ ] **Step 10: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_recurrence.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "fix: train PFNN with runtime recurrence"
```

---

### Task 4: Gate candidates on every raw fitted transition

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `sonic/python/mm_sonic/evaluate_terrain_pfnn.py`
- Test: `tests/python/test_terrain_pfnn_training.py`
- Test: `tests/python/test_terrain_pfnn_runtime.py`

**Interfaces:**
- Consumes: reloaded v5 candidate, exact fitted subset receipt, lane-safe adjacent pairs, native limits, and training-only phase q99.
- Produces: `FittedTransitionReport`, `evaluate_fitted_transition_envelope`, and a promotion receipt bound to the report digest.

- [ ] **Step 1: Write RED physical-envelope tests**

Use a fake model whose average normalized MSE passes but one row produces a
`0.251 rad` joint step. Assert failure names exact clip, lane, center, joint,
value, and `0.25` limit. Add independent failures for root translation
`0.061 m`, root rotation `0.351 rad`, joint limit, direction norm `0.49`,
negative/nonfinite phase advance, and nonfinite output.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python \
  -m unittest -v tests.python.test_terrain_pfnn_training
```

Expected: missing fitted-transition report/evaluator.

- [ ] **Step 3: Implement fitted-transition evaluation**

Add:

```python
@dataclass(frozen=True)
class FittedTransitionReport:
    accepted: bool
    sample_count: int
    maxima: dict[str, float]
    first_failure: dict[str, object] | None
    rows_sha256: str
    report_sha256: str

```

Add `evaluate_fitted_transition_envelope(model: nn.Module, dataset: object,
adjacent_indices: Sequence[tuple[int, int]], *, normalization: object,
joint_limits: object, phase_advance_q99: float) -> FittedTransitionReport`.

Evaluate each current row at its stored phase. Compare the prediction to the
predecessor target reached state. Apply exact runtime limits: root translation
`<=0.060 m`, root rotation `<=0.35 rad`, joint step `<=0.25 rad`, native
limits, positive root height, finite reconstructed WXYZ quaternion, direction
norms `[0.5,1.5]`, and bounded nonnegative phase advance. Hash canonical JSON
plus the exact fitted-row receipt.

- [ ] **Step 4: Bind the report into candidate authorization**

Run the transition gate after saving and reloading the candidate and before
calling the 600-tick verifier. Add the report and digest to the verification
request, pipeline receipt, JSON report, and `promote_pipeline_best` validation.
No report, mismatch, or `accepted=False` means no `best.pt`.

- [ ] **Step 5: Add tamper and ordering tests**

Prove model/checkpoint reload happens before evaluation, transition rejection
prevents the terrain callback from opening, and tampering maxima, first
failure, sample count, row digest, or report digest rejects promotion.

- [ ] **Step 6: Run focused GREEN tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime
```

Expected: all physical, provenance, no-callback, and promotion tests pass.

- [ ] **Step 7: Commit**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/evaluate_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_runtime.py
git commit -m "feat: gate PFNN fitted transitions"
```

---

### Task 5: Rebuild, train once, pass the known-terrain gate, and re-review Task 7

**Files:**
- Modify only if a real gate exposes a representation defect covered by a new RED test: files from Tasks 1–4.
- Write report: `.superpowers/sdd/recurrence-correction-task-5-report.md`
- Generate ignored artifacts: `sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/`
- Generate ignored artifacts: `sonic/runs/terrain-pfnn-v1/pipeline-overfit-recurrence-v2/`

**Interfaces:**
- Consumes: reviewed Tasks 1–4, sealed GRAIL/LAFAN sources, canonical G1 MJCF, and exact source-supported known-terrain evaluator.
- Produces: sealed v2 canary, immutable v5 `best.pt`, fitted-transition report, pipeline promotion receipt, and accepted 600-tick JSON.

- [ ] **Step 1: Run the full pre-build PFNN suite**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_terrain_pfnn_sources \
  tests.python.test_terrain_pfnn_phase \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_dataset \
  tests.python.test_terrain_pfnn_kinematics \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime
```

Expected: every test passes. If the Torch venv lacks a source-only dependency,
run that source test under `/home/ubuntu/miniconda3/bin/python` with the same
`PYTHONPATH` and record the interpreter split explicitly; do not change source
semantics to fit the environment.

- [ ] **Step 2: Build a fresh sealed two-family v2 canary**

Use the same verified GRAIL/LAFAN roots and canonical MJCF recorded in the
Task 5 report. Build to the new output path with
`--terrain-family-limit 2`. Do not resume from a v1 manifest.

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2 \
  --terrain-family-limit 2
```

Expected: accepted v2 manifest; exact selected identities `slope_000` and
`slope_001`; nonzero GRAIL and LAFAN rows; lane field/hash verification;
disjoint train/validation/test identities; finite normalization from train only.

- [ ] **Step 3: Verify byte/mtime resume no-op**

Hash every manifest/shard byte and record every mtime, rerun the builder with
`--resume`, and assert no byte or mtime changes. Record counts by split,
terrain class, and sequence lane.

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2 \
  --terrain-family-limit 2 --resume
```

- [ ] **Step 4: Run exactly one controlled training attempt**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.train_terrain_pfnn \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/pipeline-overfit-recurrence-v2 \
  --overfit-samples 2048 --steps 4000 --seed 7 \
  --rollout-finetune-frames 16 --rollout-finetune-steps 256
```

Expected before terrain evaluation: finite losses; reloaded v5 candidate;
fixed-sample gate accepted; fitted-transition gate accepted; no `best.pt` yet.

- [ ] **Step 5: Run and promote through the exact 600-tick verifier**

Use the candidate path printed by training:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.evaluate_terrain_pfnn \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit-recurrence-v2/checkpoint-step-00004256.pt \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --split train --closed-loop-seconds 20 --promote-pipeline-best
```

Expected: exit zero; 600 finite committed ticks; zero holds; no reversal/freeze;
all exact root/joint/collision/stance gates pass; realized supported
plateau-flank-turnaround-return traversal passes; fixed-sample and fitted-row
receipts match; immutable `best.pt` and promotion receipt are written once.

- [ ] **Step 6: Handle a failed real gate without guessing**

If Step 4 or 5 fails, record the first exact clip/lane/center/tick/field/value
and stop. A source change is allowed only after a new focused RED test proves a
representation mismatch in Tasks 1–4. Do not run a second training attempt,
change thresholds, or add output correction.

- [ ] **Step 7: Run post-artifact verification**

Reload `best.pt` through the safe loader, re-evaluate all fitted rows and the
600-tick receipt, verify candidate/best SHA binding and read-only permissions,
run `git diff --check`, `py_compile`, and the full PFNN suite from Step 1.

- [ ] **Step 8: Commit any RED-backed representation fix and write the report**

If no source change was required, do not create an empty commit. Write exact
commands, outputs, hashes, counts, timings, first-failure absence, and
self-review to `.superpowers/sdd/recurrence-correction-task-5-report.md`.
Generated artifacts remain ignored.

- [ ] **Step 9: Re-review Task 7**

Generate a review package from `6af614f` through the final source commit. Give
the reviewer the original Task 7 brief, original Task 7 report, this plan,
Task 5 report, and full diff package. Fix every Critical/Important finding via
RED→GREEN and re-review until both spec compliance and code quality are
approved.

---

## Completion handoff

After Task 7 is approved, update `.superpowers/sdd/progress.md` with the exact
commit range and review verdict, then resume the original terrain-PFNN plan at
Task 8. Do not open the sealed test split until the original Task 9 validation
gate authorizes it.
