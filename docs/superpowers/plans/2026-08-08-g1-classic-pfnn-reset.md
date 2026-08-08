# Classic G1 PFNN Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the published PFNN control loop with Unitree G1 morphology, trained on G1 flat and 5--20 degree slope motion, and driven directly with W/A/S/D in MuJoCo.

**Architecture:** Retain the existing cubic four-bank `288 -> 512 -> 512 -> 268` PFNN and the existing G1 feature layout. Replace the prototype's command-driven root override with the network's predicted root delta, use the original PFNN future-trajectory blending pattern, and train the network with the original one-step supervised objective on a grade-complete G1 corpus. Terrain is an input feature and the root support-height source; it is not a portal, pose correction, MotionBricks fallback, or IK target.

**Tech Stack:** Python 3, PyTorch, NumPy, MuJoCo, existing `mm_sonic.terrain_pfnn` modules, local reference source at `/home/ubuntu/datasets/pfnn/pfnn`.

## Global Constraints

- Behavioral source of truth: `/home/ubuntu/datasets/pfnn/pfnn/demo/pfnn.cpp`, especially the control/update loop at lines 1494--2035.
- Network stays `288 -> ELU(512) -> ELU(512) -> 268` with four cyclic Catmull-Rom parameter banks.
- Output stays native 29-DoF IsaacLab-order G1 joints; no human checkpoint retargeting.
- Runtime uses predicted planar root velocity, yaw velocity, root height, G1 joints, phase advance, body state, and future trajectory.
- W/A/S/D changes the desired future trajectory; it does not overwrite realized root motion.
- Terrain contributes center/left/right height samples and current root support height only.
- Root stays gravity-aligned in the viewer: predicted yaw is rendered, roll/pitch are diagnostic-only for this milestone.
- No MotionBricks runtime, portals, command-driven root, pose blending, foot locks, or IK.
- Do not reuse `pipeline-overfit-physical-envelope-v3` as the deliverable checkpoint.
- Do not train an 18.9-degree viewer against the current two-family canary: its measured training maximum is only 13.6536 degrees.
- Training uses author-equivalent Adam `lr=1e-4`, batch size `32`, dropout probability `0.30` (70% keep), and 20 complete shuffled passes unless validation selects an earlier checkpoint.
- Acceptance is visual and mechanical on flat, ascent, crest, descent, and landing; a route-completion flag alone is insufficient.

## Approaches Considered

1. **Recommended: classic PFNN reset.** Reuse the proven phase network and control loop, replace only the skeleton-dependent input/output data with G1 values, and train on grade-complete G1 clips. This directly matches the request.
2. **Continue tuning the physical-envelope trainer.** Rejected because the loss/gating system is now driving the project and the visible motion remains poor; it is not required by the published PFNN method.
3. **Retarget the author's human PFNN checkpoint to G1.** Rejected because its joint count, rest pose, body features, output layout, and motion distribution are incompatible with G1. The architecture transfers; the human weights do not.

---

### Task 1: Restore the published PFNN runtime contract

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/runtime.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn_viewer.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`
- Modify: `tests/python/test_terrain_pfnn_viewer.py`

**Interfaces:**
- Consumes: `TerrainPFNNRuntime.step(command: object, camera_yaw: float) -> PFNNRuntimeFrame`.
- Produces: a viewer path in which keyboard intent affects `plan_recurrent_trajectory`, while `_integrate_root_motion` consumes the PFNN output unchanged.

- [ ] **Step 1: Add a viewer regression that forbids command-driven root motion**

```python
def test_viewer_loads_raw_pfnn_runtime(self):
    arguments = _parser().parse_args([])
    with mock.patch.object(
        TerrainPFNNRuntime, "from_checkpoint", return_value=object()
    ) as factory:
        _load_runtime(arguments, checkpoint, dataset, model_path, terrain)
    self.assertIs(factory.call_args.kwargs["command_driven_root"], False)
```

- [ ] **Step 2: Run the viewer regression and observe RED**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_viewer.TerrainPFNNViewerTests.test_viewer_loads_raw_pfnn_runtime
```

Expected: FAIL because `_load_runtime` currently passes `command_driven_root=True`.

- [ ] **Step 3: Add a raw-root recurrence regression**

Construct a deterministic test PFNN output with local root velocity `[0.6, 0.0]` m/s and command `[0.9, 0.0]`. Assert that one tick advances X by `0.6 / 30`, not `0.9 / 30`, and that the PFNN yaw velocity—not desired command yaw—is integrated.

```python
self.assertAlmostEqual(frame.root_position_world[0], 0.6 / 30.0, places=7)
self.assertFalse(frame.diagnostics["command_driven_root"])
```

- [ ] **Step 4: Run the runtime regression and observe RED against the viewer configuration**

Run:

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_runtime.TerrainPFNNRuntimeTests.test_raw_pfnn_root_delta_is_not_replaced_by_command
```

- [ ] **Step 5: Remove the hybrid viewer behavior**

Change the factory call in `terrain_pfnn_viewer.py` to:

```python
return TerrainPFNNRuntime.from_checkpoint(
    checkpoint,
    dataset_digest=digest,
    model_path=model_path,
    height_and_grade_at=terrain,
    device=arguments.device,
    enforce_motion_envelope=False,
    command_driven_root=False,
)
```

Delete the `--strict-envelope` viewer option and remove `preview_envelope_violations` from normal viewer output. Runtime safety still rejects malformed/non-finite model output, but joint-step heuristics do not replace, freeze, or modify finite PFNN poses in this prototype.

- [ ] **Step 6: Match the author's command smoothing and future trajectory blend**

Keep the current 12-knot representation, but make `plan_recurrent_trajectory` equivalent to the author loop:

```python
target_velocity = lerp(previous_target_velocity, requested_velocity, 0.9)
scale_position = 1.0 - torch.pow(1.0 - future_fraction, 0.5)
scale_direction = 1.0 - torch.pow(1.0 - future_fraction, 2.0)
future_delta = lerp(previous_predicted_delta, target_velocity / FPS, scale_position)
future_direction = slerp_2d(previous_predicted_direction, target_direction, scale_direction)
```

Preserve the network's predicted trajectory after inference exactly as the next frame's prior prediction.

- [ ] **Step 7: Run the focused runtime tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_runtime \
  tests.python.test_terrain_pfnn_viewer
```

Expected: all tests pass; no test observes command velocity written into PFNN output fields.

- [ ] **Step 8: Commit the runtime reset**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/runtime.py \
  sonic/python/mm_sonic/terrain_pfnn_viewer.py \
  tests/python/test_terrain_pfnn_runtime.py \
  tests/python/test_terrain_pfnn_viewer.py
git commit -m "fix: restore raw G1 PFNN runtime"
```

---

### Task 2: Build a grade-complete G1 training corpus

**Files:**
- Modify: `sonic/python/mm_sonic/build_terrain_pfnn_dataset.py`
- Modify: `sonic/python/mm_sonic/terrain_pfnn/sources.py`
- Modify: `tests/python/test_terrain_pfnn_dataset.py`
- Modify: `tests/python/test_terrain_pfnn_sources.py`
- Generate: `sonic/runs/terrain-pfnn-classic-g1/dataset/`

**Interfaces:**
- Produces: dataset schema compatible with `PFNNShardDataset`, containing G1 flat locomotion plus GRAIL train identities spanning the viewer's grade range.
- Consumes later: `classic_train_indices(dataset) -> np.ndarray` with deterministic 50/50 flat/slope batches.

- [ ] **Step 1: Add a corpus-coverage RED test**

Add a deterministic selector that receives `(terrain_identity, maximum_grade_degrees)` records and selects the closest train identity to each required grade:

```python
REQUIRED_TRAINING_GRADES_DEG = (5.0, 10.0, 15.0, 18.9)
MAXIMUM_GRADE_ERROR_DEG = 1.0
```

The test must fail when the available maximum is 13.6536 degrees and must select four distinct identities when qualifying records exist.

- [ ] **Step 2: Run the selector test and observe RED**

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_sources.TerrainPFNNSourcesTests.test_grade_selector_requires_5_to_18_9_degree_train_coverage
```

- [ ] **Step 3: Implement deterministic grade-family selection**

For each GRAIL terrain identity, compute the maximum upward native triangle grade from its verified USD mesh. Restrict candidates to the train split, then for each required grade choose the unused identity minimizing:

```python
(abs(maximum_grade_degrees - target_grade), terrain_identity)
```

Reject the build if any selected error exceeds `1.0` degree. Include every motion variant belonging to each selected terrain identity.

- [ ] **Step 4: Add a manifest coverage receipt**

Persist this exact structure in `manifest.json`:

```json
{
  "required_training_grades_degrees": [5.0, 10.0, 15.0, 18.9],
  "selected_training_terrain_grades": {
    "<terrain_identity>": 18.7
  },
  "maximum_grade_error_degrees": 1.0
}
```

The reader must reject missing identities, duplicate selected identities, non-finite grades, or a target farther than one degree from its selected identity.

- [ ] **Step 5: Build the dataset once**

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL/data/slope \
  --lafan-root /home/ubuntu/datasets/lafan1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-classic-g1/dataset \
  --terrain-grade-targets 5,10,15,18.9
```

Expected: accepted manifest; selected training grades cover all four targets within one degree; no validation/test identities are used for training normalization.

- [ ] **Step 6: Verify actual row composition**

Run a read-only script that asserts:

```python
assert set(source_kind) == {"grail", "lafan"}
assert all(abs(selected[target] - target) <= 1.0 for target in required)
assert train_grail_rows > 0
assert train_lafan_rows > 0
```

- [ ] **Step 7: Run source and dataset suites**

```bash
PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_sources \
  tests.python.test_terrain_pfnn_dataset
```

- [ ] **Step 8: Commit the corpus selection code**

```bash
git add sonic/python/mm_sonic/build_terrain_pfnn_dataset.py \
  sonic/python/mm_sonic/terrain_pfnn/sources.py \
  tests/python/test_terrain_pfnn_dataset.py \
  tests/python/test_terrain_pfnn_sources.py
git commit -m "feat: cover G1 PFNN slope grades"
```

---

### Task 3: Train the conventional G1 PFNN

**Files:**
- Create: `sonic/python/mm_sonic/train_classic_g1_pfnn.py`
- Create: `tests/python/test_classic_g1_pfnn.py`
- Generate: `sonic/runs/terrain-pfnn-classic-g1/model/`

**Interfaces:**
- Consumes: `PFNNShardDataset`, `PhaseFunctionedNetwork`, G1 normalization arrays, and the grade-complete training manifest.
- Produces: `classic-g1-pfnn/v1` checkpoint containing model tensors, normalization, runtime seed, dataset digest, epoch, and validation score.

- [ ] **Step 1: Write the classic-loss RED test**

The loss has only the supervised PFNN prediction objective:

```python
continuous = F.mse_loss(prediction[:, :contact_start], target[:, :contact_start])
contacts = F.binary_cross_entropy_with_logits(
    prediction[:, contact_start:contact_stop],
    target[:, contact_start:contact_stop],
)
loss = continuous + contacts
```

Assert that changing envelope thresholds, predecessor poses, FK helpers, or rollout state cannot change this loss.

- [ ] **Step 2: Run the classic-loss test and observe RED**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_classic_g1_pfnn.ClassicG1PFNNTests.test_loss_is_only_one_step_supervision
```

- [ ] **Step 3: Implement the thin trainer**

Use existing normalized `x`, normalized `y`, and phase. Construct `PhaseFunctionedNetwork(hidden_size=512, dropout_probability=0.30)`, `torch.optim.Adam(lr=1e-4)`, and deterministic shuffled source-balanced batches:

```python
flat_count = batch_size // 2
slope_count = batch_size - flat_count
batch = flat_permutation.take(flat_count) + slope_permutation.take(slope_count)
```

Cycle the shorter source without changing its within-source permutation. One epoch ends after every row in the larger source has appeared at least once.

- [ ] **Step 4: Add exact checkpoint round-trip tests**

Save only CPU contiguous tensors and JSON-compatible scalars. Load with `torch.load(..., weights_only=True)`. Assert identical outputs before/after reload for 32 fixed rows and phases.

- [ ] **Step 5: Add a short overfit test**

Train 32 samples for 300 updates. Require final fixed-sample loss below 10% of initial loss and bitwise-identical outputs after reload.

- [ ] **Step 6: Run the trainer tests**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_classic_g1_pfnn
```

- [ ] **Step 7: Train one full model**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.train_classic_g1_pfnn \
  --dataset sonic/runs/terrain-pfnn-classic-g1/dataset/manifest.json \
  --output sonic/runs/terrain-pfnn-classic-g1/model \
  --epochs 20 --batch-size 32 --learning-rate 1e-4 --seed 23456
```

Save the lowest validation-loss epoch as `best.pt`. Do not block saving on physical-envelope, pair-receipt, rollout, or promotion gates.

- [ ] **Step 8: Reject a mechanically unusable model before opening the viewer**

On contiguous held-out train sequences, feed the true prior state and assert:

```python
assert np.isfinite(all_predictions).all()
assert median_joint_error_rad < baseline_median_joint_error_rad
assert phase_advance_positive_fraction > 0.99
```

This is a diagnostic rejection only. Do not modify or clamp model outputs.

- [ ] **Step 9: Commit the classic trainer**

```bash
git add sonic/python/mm_sonic/train_classic_g1_pfnn.py \
  tests/python/test_classic_g1_pfnn.py
git commit -m "feat: train conventional G1 PFNN"
```

---

### Task 4: Drive and visually accept the actual PFNN

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn_viewer.py`
- Modify: `tests/python/test_terrain_pfnn_viewer.py`
- Generate: `sonic/runs/terrain-pfnn-classic-g1/visual-acceptance/`

**Interfaces:**
- Consumes: `classic-g1-pfnn/v1` `best.pt`.
- Produces: the user-facing MuJoCo viewer and visual evidence from the exact interactive code path.

- [ ] **Step 1: Add checkpoint loading to the viewer**

Load `best.pt` through the classic safe loader and construct `TerrainPFNNRuntime` with raw root integration. The displayed qpos must be exactly:

```python
qpos[:3] = predicted_root_position
qpos[3:7] = yaw_only_gravity_aligned_quaternion(predicted_root_yaw)
qpos[7:36] = isaaclab_to_mujoco_joint_vector(predicted_g1_joints)
```

- [ ] **Step 2: Add an automated raw-root route test**

Hold W for 600 ticks. Assert that network-predicted root X—not `speed / 30`—causes progress, phase changes on at least 95% of walking ticks, and the root reaches ascent, crest, descent, and landing segments without non-finite output.

- [ ] **Step 3: Run the focused viewer route**

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.terrain_pfnn_viewer \
  --checkpoint sonic/runs/terrain-pfnn-classic-g1/model/best.pt \
  --device cuda --speed 0.7 --no-viewer --smoke-steps 600 --trace-every 60
```

The trace must report desired and realized speed separately. Exact commanded displacement is forbidden as evidence of success.

- [ ] **Step 4: Perform the same interaction the user will perform**

Launch the viewer and personally execute:

1. idle for two seconds;
2. W across 10 degrees and release on the descent;
3. W across 15 degrees, turn with A on the flank, then D back toward the route;
4. W across 18.9 degrees and stop on the landing;
5. S back over the 18.9-degree hill;
6. resume flat walking.

- [ ] **Step 5: Capture visual checkpoints**

Save full-resolution frames at flat, ascent, crest, descent, and landing. Reject the checkpoint if any of these are visible:

- persistent deep crouch unrelated to the source motion;
- root being dragged while the feet remain behind;
- phase freeze while moving;
- foot penetrating through the slope;
- both feet floating through a stance interval;
- discontinuous pose snap at a terrain transition.

- [ ] **Step 6: Apply the stop rule**

If visual acceptance fails, do not add IK, root overrides, portals, joint clamps, or pose blending. Record the first bad frame and compare its 288-value runtime input against the nearest supervised GRAIL frame; fix only the first mismatching data/runtime field, retrain, and repeat Task 4.

- [ ] **Step 7: Run final verification**

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -m unittest -q \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_runtime \
  tests.python.test_terrain_pfnn_viewer \
  tests.python.test_classic_g1_pfnn
git diff --check
```

- [ ] **Step 8: Commit the playable viewer**

```bash
git add sonic/python/mm_sonic/terrain_pfnn_viewer.py \
  tests/python/test_terrain_pfnn_viewer.py
git commit -m "feat: drive classic G1 PFNN on hills"
```

## Completion Definition

The task is complete only when the viewer is using the PFNN's own root and G1 pose, its training corpus includes G1 motions near 5, 10, 15, and 18.9 degrees, and a human-driven W/A/S/D pass looks acceptable on ascent, crest, descent, landing, and return. Passing a headless route while the character is visibly dragged does not count.
