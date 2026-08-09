# G1 PFNN Transfer Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the recurrent-body feature corruption, prove that a released-PFNN-only model reproduces the offline-retargeted G1 limbs, and only then add corrected GRAIL slope rows and promote an interactively verified mixed checkpoint.

**Architecture:** One authoritative inverse input-normalization function closes the broken GRAIL merge boundary. A source-aware evaluator measures articulated joint reconstruction independently of root motion. Training proceeds through two immutable gates: released PFNN only, then corrected mixed PFNN plus GRAIL; the viewer is updated only after both pass.

**Tech Stack:** Python 3.10, NumPy, PyTorch/CUDA, MuJoCo, `unittest`, existing terrain-PFNN feature/layout/checkpoint APIs.

## Global Constraints

- Preserve the 288-input/268-output PFNN ABI and 29-joint canonical IsaacLab order.
- Preserve released retargets, GRAIL sources, and existing datasets; write corrected artifacts to new directories.
- Do not add IK, foot locking, physics control, root projection, or render-only corrections to acceptance measurements.
- Use red-green TDD for every production change.
- A model cannot be promoted solely on root, trajectory, aggregate loss, or terrain metrics.
- Released-PFNN transfer must pass before corrected GRAIL training begins.
- Final acceptance requires a personally inspected W/A/S/D MuJoCo drive.

---

## File map

- `sonic/python/mm_sonic/terrain_pfnn/dataset.py`: authoritative normalization and inverse-normalization boundary.
- `sonic/python/mm_sonic/build_g1_pfnn_mixed_dataset.py`: reconstruct physical GRAIL rows and create the corrected mixed corpus.
- `sonic/python/mm_sonic/train_classic_g1_pfnn.py`: optional receipt-checked weight initialization for the mixed stage.
- `sonic/python/mm_sonic/evaluate_classic_g1_pfnn_transfer.py`: source-specific exact-input limb metrics and promotion gates.
- `sonic/python/mm_sonic/terrain_pfnn_viewer.py`: final artifact defaults only after promotion.
- `tests/python/test_terrain_pfnn_dataset.py`: inverse-normalization unit contract.
- `tests/python/test_build_g1_pfnn_mixed_dataset.py`: full source-to-mixed parity regression.
- `tests/python/test_classic_g1_pfnn.py`: initialization provenance and dataset separation.
- `tests/python/test_evaluate_classic_g1_pfnn_transfer.py`: exact metric math, source separation, and rejection behavior.
- `tests/python/test_terrain_pfnn_viewer.py`: final default artifact binding.
- `docs/g1_pfnn_demo.md`: final command and artifact receipts.

---

### Task 1: Make PFNN input normalization exactly invertible

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/dataset.py`
- Modify: `tests/python/test_terrain_pfnn_dataset.py`

**Interfaces:**
- Consumes: `normalize_pfnn_input(x, x_mean, x_std) -> np.ndarray`.
- Produces: `denormalize_pfnn_input(x, x_mean, x_std) -> np.ndarray`, preserving leading dimensions and returning contiguous `float32` physical features.

- [ ] **Step 1: Write the failing round-trip and validation tests**

Add a test with nonzero means, nonunit standard deviations, and nonzero body slices:

```python
def test_input_normalization_round_trip_restores_recurrent_body_scale(self):
    raw = np.linspace(-2.0, 3.0, INPUT_LAYOUT.size, dtype=np.float32)
    mean = np.linspace(-0.3, 0.4, INPUT_LAYOUT.size, dtype=np.float32)
    std = np.linspace(0.2, 1.7, INPUT_LAYOUT.size, dtype=np.float32)
    normalized = normalize_pfnn_input(raw, mean, std)
    restored = denormalize_pfnn_input(normalized, mean, std)
    np.testing.assert_allclose(restored, raw, rtol=2.0e-6, atol=2.0e-6)
```

Add subtests rejecting a last dimension other than 288, nonfinite values, and nonpositive `x_std`.

- [ ] **Step 2: Run the focused test and capture RED**

Run:

```bash
cd /home/ubuntu/worktrees/motion-matching-hill-conditioning
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_dataset.TerrainPFNNDatasetTest.test_input_normalization_round_trip_restores_recurrent_body_scale
```

Expected: import failure for missing `denormalize_pfnn_input`.

- [ ] **Step 3: Implement the minimal inverse**

Implement beside `normalize_pfnn_input`:

```python
def denormalize_pfnn_input(x: object, x_mean: object, x_std: object) -> np.ndarray:
    value, mean, std = _validated_input_normalization_values(x, x_mean, x_std)
    normalized = np.ascontiguousarray(value, dtype=np.float32).copy()
    for field in ("previous_body_position", "previous_body_velocity"):
        normalized[..., INPUT_LAYOUT[field]] /= np.float32(0.1)
    return np.ascontiguousarray(normalized * std + mean, dtype=np.float32)
```

Share validation with `normalize_pfnn_input` without changing its arithmetic. Export the inverse in `__all__`.

- [ ] **Step 4: Run focused and dataset suites**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_dataset
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit Task 1**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/dataset.py \
  tests/python/test_terrain_pfnn_dataset.py
git commit -m "fix: invert PFNN recurrent input normalization"
```

---

### Task 2: Correct and prove the GRAIL physical-row merge

**Files:**
- Modify: `sonic/python/mm_sonic/build_g1_pfnn_mixed_dataset.py`
- Modify: `tests/python/test_build_g1_pfnn_mixed_dataset.py`

**Interfaces:**
- Consumes: `denormalize_pfnn_input` from Task 1 and `normalize_pfnn_output` conventions.
- Produces: `_grail_physical_rows(dataset) -> dict[str, np.ndarray]` whose `x` and `y` are exact physical source rows.

- [ ] **Step 1: Write the failing recurrent-body parity regression**

Change `_GrailRows` to use nonzero `x_mean`, nonunit `x_std`, and normalized body slices generated by `normalize_pfnn_input`. Add:

```python
def test_grail_merge_restores_every_physical_input_and_target_field(self):
    grail = _GrailRows()
    expected_x = grail.physical_x_by_key
    expected_y = grail.physical_y_by_key
    mixed = combine_vertical_and_grail(
        build_vertical_dataset((
            _source("train", "released_train"),
            _source("validation", "released_val"),
        )),
        grail,
        grail_dataset_sha256="9" * 64,
    )
    for split in ("train", "validation"):
        arrays = mixed.splits[split]
        for index, clip in enumerate(arrays.clip_id):
            if not str(clip).startswith("terrain_slopes__") or arrays.mirrored[index]:
                continue
            key = (str(clip), str(arrays.sequence_lane[index]),
                   int(arrays.center_frame_120hz[index] // 4))
            np.testing.assert_allclose(arrays.x[index], expected_x[key], rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(arrays.y[index], expected_y[key], rtol=2e-6, atol=2e-6)
```

- [ ] **Step 2: Run RED**

Run the named test. Expected: body position/velocity mismatches while other fields match.

- [ ] **Step 3: Replace the broken affine reconstruction**

In `_grail_physical_rows`, replace the direct affine reconstruction with:

```python
"x": denormalize_pfnn_input(x_normalized, x_mean, x_std),
```

Keep target reconstruction unchanged because PFNN output normalization has no additional body scaling.

- [ ] **Step 4: Run GREEN and real read-only parity audit**

Run the full mixed-builder test module. Then run a real 2,285-row source-to-mixed audit against `sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2`; require every input field and target to differ by at most `2.0e-6` after float32 reconstruction.

- [ ] **Step 5: Commit Task 2**

```bash
git add sonic/python/mm_sonic/build_g1_pfnn_mixed_dataset.py \
  tests/python/test_build_g1_pfnn_mixed_dataset.py
git commit -m "fix: reconstruct GRAIL PFNN body inputs"
```

---

### Task 3: Add a limb-specific PFNN transfer gate

**Files:**
- Create: `sonic/python/mm_sonic/evaluate_classic_g1_pfnn_transfer.py`
- Create: `tests/python/test_evaluate_classic_g1_pfnn_transfer.py`

**Interfaces:**
- Produces: `joint_reconstruction_metrics(predicted_physical, target_physical) -> dict[str, float]`.
- Produces CLI JSON with separate `released_pfnn` and `grail` records containing `sample_count`, `joint_mae_rad`, `joint_rmse_rad`, `frame_max_p95_rad`, `maximum_joint_error_rad`, and worst provenance.

- [ ] **Step 1: Write metric RED tests**

Use a four-frame, 29-joint synthetic example with one known 0.2-rad error and assert exact MAE, RMSE, p95 frame maximum, maximum, and worst joint/frame. Add a mixed-source example proving a good GRAIL subset cannot hide a failing released subset.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_evaluate_classic_g1_pfnn_transfer
```

Expected: module import failure.

- [ ] **Step 3: Implement metric and CLI**

The CLI must safely load the checkpoint and dataset, reject digest mismatch, evaluate float32 normalized rows and phases, denormalize outputs once, partition by the `terrain_slopes__` clip prefix, and emit source-specific metrics and SHA-256 receipts. Return exit 2 when any configured gate fails.

Freeze the released-only gate at:

```text
joint_mae_rad <= 0.100
joint_rmse_rad <= 0.150
frame_max_p95_rad <= 0.500
maximum_joint_error_rad <= 1.000
```

These thresholds reject the current released validation metrics (`0.1825`, `0.3128`, `1.8431`, and greater than `2.0`) while remaining far above float32/retarget noise.

- [ ] **Step 4: Run GREEN and current-checkpoint negative control**

Run the new test module, then evaluate the current final-v2 checkpoint. Expected: exit 2 with a released-PFNN limb-gate failure.

- [ ] **Step 5: Commit Task 3**

```bash
git add sonic/python/mm_sonic/evaluate_classic_g1_pfnn_transfer.py \
  tests/python/test_evaluate_classic_g1_pfnn_transfer.py
git commit -m "feat: gate G1 PFNN limb reconstruction"
```

---

### Task 4: Prove released-PFNN-only transfer

**Files:**
- Modify only if the diagnostic gate exposes a training-boundary defect: `sonic/python/mm_sonic/train_classic_g1_pfnn.py`
- Modify corresponding tests only: `tests/python/test_classic_g1_pfnn.py`
- Generate ignored artifacts under: `sonic/runs/native-g1-pfnn/expanded/released-transfer-v3/`

**Interfaces:**
- Consumes: `sonic/runs/native-g1-pfnn/expanded/vertical-corpus/dataset/manifest.json`.
- Produces: an accepted released-only `best.pt` and transfer evaluation JSON.

- [ ] **Step 1: Run a 64-row contiguous diagnostic overfit**

Materialize one non-idle `motion` lane with 64 adjacent centers from a released training clip. Train `PhaseFunctionedNetwork(hidden_size=512, dropout=0)` for at most 2,000 updates and evaluate on the same 64 rows.

Require:

```text
joint_mae_rad <= 0.010
frame_max_p95_rad <= 0.050
maximum_joint_error_rad <= 0.100
```

If this fails, stop before full training and trace phase/input/output packing; do not add loss weights or smoothing speculatively.

- [ ] **Step 2: Train released-only full corpus**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.train_classic_g1_pfnn \
  --dataset sonic/runs/native-g1-pfnn/expanded/vertical-corpus/dataset/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/native-g1-pfnn/expanded/released-transfer-v3/model \
  --device cuda --epochs 160 --batch-size 64 --evaluation-batch-size 512 \
  --learning-rate 1e-4 --hidden-size 512 --seed 23456 \
  --train-source released-pfnn --runtime-seed flat
```

- [ ] **Step 3: Run the exact-input transfer gate**

Evaluate train and validation. Both must pass Task 3 thresholds. Record checkpoint, dataset, metric JSON, and SHA-256 values.

- [ ] **Step 4: Run deterministic closed-loop and visual comparison**

Use one known `LocomotionFlat` and one `WalkingUpSteps` interval. Compare target and predicted MuJoCo poses at identical frames without display damping. Reject visible mean-pose collapse, incorrect arm swing, leg crossing, twisting, or contact timing mismatch.

- [ ] **Step 5: Commit only code/test corrections, never generated artifacts**

If Step 1 exposes no additional code defect, this task creates no source commit. Otherwise follow a separate RED/GREEN cycle and commit the minimal correction before restarting the released-only run.

---

### Task 5: Initialize and train the corrected mixed model

**Files:**
- Modify: `sonic/python/mm_sonic/train_classic_g1_pfnn.py`
- Modify: `tests/python/test_classic_g1_pfnn.py`
- Generate ignored artifacts under: `sonic/runs/native-g1-pfnn/expanded/mixed-corpus-corrected-v3/` and `sonic/runs/native-g1-pfnn/expanded/model-mixed-corrected-v3/`

**Interfaces:**
- Adds CLI option `--initialize-from PATH`.
- Weight initialization consumes only a safely loaded classic checkpoint with the same kinematic signature and model configuration; it permits a different dataset digest and writes the new digest into the output checkpoint.

- [ ] **Step 1: Write initialization RED tests**

Prove initialization copies exact model weights, rejects kinematic/config mismatches before optimization, does not copy the old dataset digest/runtime seed/normalization, and remains optional.

- [ ] **Step 2: Run RED, implement minimal initialization, run GREEN**

Load the initializer before constructing the optimizer. Validate hidden size and dropout configuration, copy only `model_state`, and build the new runtime seed and receipts from the corrected dataset.

- [ ] **Step 3: Build corrected mixed corpus**

Run `mm_sonic.build_g1_pfnn_mixed_dataset` with the immutable released corpus and recurrence-v2 GRAIL dataset, outputting only to `mixed-corpus-corrected-v3`. Run the real parity audit and record the new dataset digest.

- [ ] **Step 4: Fine-tune from the accepted released checkpoint**

Train 80 epochs at `1e-4`, batch 64, initialized from Task 4. Select best validation loss but promote only if the source-specific transfer evaluator passes.

- [ ] **Step 5: Enforce non-regression**

Require the mixed model to satisfy all Task 3 absolute gates and:

```text
released validation joint MAE increase <= 0.010 rad
released validation frame-max p95 increase <= 0.050 rad
GRAIL validation joint MAE <= 0.100 rad
GRAIL validation frame-max p95 <= 0.500 rad
```

- [ ] **Step 6: Commit Task 5 source changes**

```bash
git add sonic/python/mm_sonic/train_classic_g1_pfnn.py \
  tests/python/test_classic_g1_pfnn.py
git commit -m "feat: initialize corrected mixed G1 PFNN training"
```

---

### Task 6: Promote and personally drive the final viewer

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn_viewer.py`
- Modify: `tests/python/test_terrain_pfnn_viewer.py`
- Modify: `docs/g1_pfnn_demo.md`

**Interfaces:**
- Viewer defaults point only to the Task 5 accepted checkpoint and corrected-v3 dataset.

- [ ] **Step 1: Write default-binding RED test**

Assert the exact corrected dataset/checkpoint paths and require both files to exist before viewer startup.

- [ ] **Step 2: Update defaults and documentation; run GREEN**

Keep `_apply_frame` and the canonical joint permutation unchanged. Clearly label the demo as kinematic PFNN, not physics control.

- [ ] **Step 3: Run full relevant verification**

Run all classic training, mixed builder, transfer evaluator, recurrence, runtime, and viewer tests. Run `py_compile` and `git diff --check`.

- [ ] **Step 4: Drive the real viewer**

Launch the documented command and personally test W on flat/ascent, W+A and W+D, S on descent, key release, and restart. Inspect raw undamped predictions alongside the rendered pose. Record a clean window-only video only if requested.

- [ ] **Step 5: Commit final bindings**

```bash
git add sonic/python/mm_sonic/terrain_pfnn_viewer.py \
  tests/python/test_terrain_pfnn_viewer.py docs/g1_pfnn_demo.md
git commit -m "feat: promote corrected G1 PFNN demo"
```

---

## Final verification command set

```bash
cd /home/ubuntu/worktrees/motion-matching-hill-conditioning
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_dataset \
  tests.python.test_build_g1_pfnn_mixed_dataset \
  tests.python.test_classic_g1_pfnn \
  tests.python.test_evaluate_classic_g1_pfnn_transfer \
  tests.python.test_rollout_fine_tune_classic_g1_pfnn \
  tests.python.test_terrain_pfnn_runtime \
  tests.python.test_terrain_pfnn_viewer
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m py_compile \
  sonic/python/mm_sonic/terrain_pfnn/dataset.py \
  sonic/python/mm_sonic/build_g1_pfnn_mixed_dataset.py \
  sonic/python/mm_sonic/train_classic_g1_pfnn.py \
  sonic/python/mm_sonic/evaluate_classic_g1_pfnn_transfer.py \
  sonic/python/mm_sonic/terrain_pfnn_viewer.py
git diff --check
git status --short
```

Expected: every test passes, compilation emits no output, diff check exits 0, and tracked status is clean after commits.
