# Object/Grasp Diffusion Funnels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task-by-task. Every task
> uses a fresh implementation subagent followed by spec-compliance and code-
> quality review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three fixed Smart Pickup entry slots with a reproducible
object/grasp-conditioned proposal provider that samples 32 short local funnels,
rejects invalid paths deterministically, follows one at 25 Hz, and hands the
terminal root to the existing motion matcher and actual attachment runtime.

**Architecture:** Export 16-sample root trajectories in the target object's
planar frame, reverse them into a grasp-anchored outward training convention,
and train a compact conditional DDPM offline. Sampling reverses proposals once
into entrance-to-terminal execution order and writes a strict binary artifact.
A new `SmartPickupAssistBackend` loads, certifies, previews, ranks, and follows
one frozen route through ordinary flat-locomotion controls; the current matcher,
pickup playback, IK, contact, attachment, Hold, Carry, and Plan A placement
pipeline remain authoritative.

**Tech Stack:** Python 3.10, NumPy 2.2.6, PyTorch 2.5.1, one NVIDIA L40S,
C++17, GNU Make, existing flat locomotion/runtime/Raylib, exact 25 Hz.

## Global Constraints

- Before Task 0, and before deleting or recreating any environment, run exactly:

  ```bash
  make gate-smart-pickup-plan-a \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  ```

  Stop if it does not pass. The explicit override is mandatory because the
  Makefile default is not the reviewed full pack. Plan B never repairs,
  bypasses, or weakens Plan A.
- Task 0 creates the only allowed Python environment at
  `build/venvs/g1-funnels` from `/home/ubuntu/miniconda3/bin/python3.10`
  (`Python 3.10.13`) with `numpy==2.2.6` and `torch==2.5.1`. Every Plan B
  Python command, including Task 1 RED tests, uses that environment's Python;
  never use ambient `python`, `python3`, NumPy, or Torch.
- Work only in branch/worktree `g1-tabletop-placement`; do not merge, modify,
  signal, focus, or stop terrain work or its process.
- Simulation, controller observations, funnel targets, evidence, and video are
  exactly 25 Hz (`0.04 s`). Do not introduce 60 Hz resampling.
- Use only `build/smart-pickup/full-pack`: 2,045 clips, 511,250 frames, 633
  packed object IDs. Do not parse raw pickup PKLs.
- `GraspCondition` contains active hand; object-local grasp translation;
  object-local 6D rotation (first rotation-matrix column followed by the
  second); horizontal object-local approach; object dimensions; support height
  above flat ground; and grasp height relative to support. No object, clip,
  sequence, frame, slot, or absolute world X/Z/yaw identity enters the model.
- The model condition vector has exactly 18 float32 values: two hand one-hot,
  then `3 + 6 + 2 + 3 + 1 + 1` continuous values.
- The two height scalars are authoritative. Derived world grasp height is their
  float64 sum and is never separately serialized; mismatch above `0.00002 m`
  rejects a row/artifact.
- A raw window has exactly 16 chronological root samples ending at the first
  Reach frame. Export reverses it exactly once into terminal-to-entrance outward
  training order. Proposal export reverses model output exactly once into
  entrance-to-terminal execution order. Runtime consumes execution order only.
- Every sample is object-local `(x, z, sin(yaw), cos(yaw))`.
- A learned funnel requires arc length at least `0.15 m`, at least three greedy
  waypoint clusters under `0.05 m`/`5 degrees`, per-sample translation at most
  `0.08 m`, and per-sample yaw change at most `0.261799388 rad` (15 degrees).
- Split the 633 packed object IDs by stable SHA-256 ordering into exactly 506
  train, 63 validation, and 64 test IDs. Deduplicate only the train/validation
  rows using the frozen byte-level rule in Task 1; the complete test partition
  is immutable and never participates in a deduplication comparison.
  Normalization uses retained training rows only; early stopping and model
  selection use retained validation rows only.
- Freeze and commit the complete evaluation manifest before training,
  hyperparameter selection, blocker placement, or final proposal generation.
- Each provider receives exactly 32 proposals in one stable batch. There is no
  resampling until success.
- Obstacles are not learned inputs. Sweep the live-root connector and every
  proposal segment. A blocked proposal may be replaced only by another proposal
  from the same frozen batch before movement starts.
- Ranking is: certification; route millimetres; heading milliradians; larger
  minimum clearance; runtime-preview total cost; stable proposal index.
- Once movement begins, proposal identity never changes. Revalidate remaining
  segments against the same frozen scene inputs each tick; cancel on blockage.
- A KNN/diffusion follower never writes the character root. After connector
  arrival it publishes each of 16 targets once in order on 16 consecutive
  native ticks. Tracking error above `0.18 m` or `0.436332313 rad` (25 degrees),
  a missed tick, or an invalid remaining path cancels the attempt. Endpoint-only
  has `sample_count=1`: ordinary locomotion connects directly to that terminal,
  publishes no timed funnel samples, and proceeds to final preview.
- Runtime success means actual `Carry`, registry `Held`, `attached=true`, exact
  request owner, one attachment edge, and a subsequent ordinary Carry tick.
- Learned acceptance evidence may not consult or silently use authored slots.
  Authored fallback is an explicit manual provider mode only.
- Python/Torch never executes inside the 25 Hz controller loop. Runtime loads a
  complete versioned artifact before learned mode is enabled.
- The known authored grasp is the playable V0 condition. The public condition
  constructor is GraspMolmo-compatible, but no RGB, GraspMolmo weights, or
  multi-grasp selection is added here.
- Store datasets/checkpoints/evaluation outputs under `build/g1-funnels/` and
  temporary video outside the repository under
  `/home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-b/`.
- Do not access, stat, hash, execute, modify, stage, delete, or enumerate the
  protected repository-root untracked `interaction_query_probe`.
- Inspect status only with `git status --short --untracked-files=no`; stage
  explicit paths only; never use `git add .` or `git add -A`.
- Every task follows RED -> GREEN -> focused regression -> independent review ->
  explicit-path commit -> `git push checkpoint HEAD:g1-tabletop-placement`.

## Frozen Model and Sampler Configuration

- Input/output tensor: float32 `[batch, 4, 16]`.
- Input projection: `Conv1d(4, 64, kernel_size=3, padding=1)`.
- Timestep embedding: 128-dimensional sinusoidal embedding followed by
  `Linear(128, 128)`, SiLU, `Linear(128, 128)`.
- Condition embedding: `Linear(18, 128)`, SiLU, `Linear(128, 128)`.
- Trunk: four width-64 FiLM residual Conv1D blocks. Each block uses
  `GroupNorm(8, 64)`, SiLU, `Conv1d(64,64,3,padding=1)`, a
  `Linear(256,128)` scale/shift projection, then a second
  `GroupNorm(8,64)`, SiLU, and `Conv1d(64,64,3,padding=1)` with residual add.
- Output: `GroupNorm(8,64)`, SiLU, `Conv1d(64,4,3,padding=1)` initialized to
  zero weight/bias.
- Diffusion: 1,000 DDPM steps; linear beta from `0.0001` through `0.02`
  inclusive; epsilon-prediction MSE.
- Optimizer: AdamW, learning rate `0.0003`, betas `(0.9, 0.999)`, weight decay
  `0.0001`, gradient norm clip `1.0`.
- Training: seed `2026071901`, batch 256, maximum 20,000 optimizer steps,
  validation every 250 steps, EMA `0.999`, best validation epsilon MSE,
  early-stop patience 20 validations and minimum improvement `0.000001`.
- Sampling: deterministic DDIM, `eta=0`, exactly 32 ordered samples. The 50
  ascending indices are `floor(i*999/49)` for integer `i=0..49`; sampling uses
  this exact descending list:
  `[999, 978, 958, 937, 917, 897, 876, 856, 835, 815, 795, 774, 754, 733,
  713, 693, 672, 652, 632, 611, 591, 570, 550, 530, 509, 489, 468, 448,
  428, 407, 387, 366, 346, 326, 305, 285, 265, 244, 224, 203, 183, 163,
  142, 122, 101, 81, 61, 40, 20, 0]`. Seeds are manifest-derived per row.
- Device contract: one GPU selected with `CUDA_VISIBLE_DEVICES=0`; no
  multi-GPU or hardware-count-dependent behavior.
- Build the 1,000-step beta schedule on CPU as float64 with
  `beta[t] = 0.0001 + t * (0.02 - 0.0001) / 999` for `t=0..999`, then compute
  `alpha=1-beta` and `alpha_bar=cumprod(alpha)` in increasing index order in
  float64. Store all three arrays in the checkpoint; sampling must use those
  stored arrays rather than rebuilding them.
- Normalize the 18-value condition and the `[16,4]` outward target with the
  retained-training means/scales before model use. DDIM state remains normalized
  float32 throughout sampling. After the final recurrence, transpose
  `[32,4,16]` to C-contiguous `[32,16,4]`, denormalize once in float32, and only
  then project every `(sin(yaw),cos(yaw))` pair to unit length. For that
  projection, cast the two denormalized float32 components to float64, compute
  `sqrt(sum(component * component, dtype=float64))` in component order, reject a
  non-finite norm or norm below `1e-8`, divide in float64, and cast the two
  divided values once to little-endian float32. No norm, division, or yaw
  projection occurs before denormalization. Perform nondegeneracy validation and
  exact-once reversal only after this projection; these projected float32 bytes
  are the bytes serialized into the proposal artifact.
- For proposal seed `s`, initialize
  `torch.Generator(device="cpu").manual_seed(s)` and draw exactly one
  `torch.randn((32,4,16), dtype=torch.float32, device="cpu", generator=g)` in
  proposal-major C order. Transfer that complete tensor to CUDA once. No other
  random draw is permitted during DDIM. For each descending index `t_i`, let
  `p_i` be the next listed index, or `-1` after `t=0`; with EMA epsilon prediction
  `e`, compute in float32 tensor order using float64 schedule scalars cast once
  to float32:

  ```text
  x0 = (x_t - sqrt(1 - alpha_bar[t_i]) * e) / sqrt(alpha_bar[t_i])
  x_next = sqrt(alpha_bar_previous) * x0
           + sqrt(1 - alpha_bar_previous) * e
  alpha_bar_previous = 1.0 when p_i == -1, else alpha_bar[p_i]
  ```

  This is deterministic DDIM with `eta=0`; there is no clipping, thresholding,
  fresh noise, or yaw projection between recurrences. The final `x_next` after
  `t=0` is the normalized outward prediction.

## File Map

- `resources/g1_funnel_builder/schema.py`: grasp/path/audit dataclasses and
  canonical float conventions.
- `resources/g1_funnel_builder/dataset.py`: 16-sample extraction, reversal,
  condition construction, and rejection reasons.
- `resources/g1_funnel_builder/dataset_artifact.py`: deterministic array/JSON
  dataset artifact and train-only normalization.
- `resources/g1_funnel_builder/splits.py`: independent 506/63/64 object split.
- `resources/g1_funnel_builder/baselines.py`: complete-funnel KNN and true
  endpoint-only providers.
- `resources/g1_funnel_builder/evaluation.py`: frozen manifest and metrics.
- `resources/g1_funnel_builder/model.py`: fixed denoiser.
- `resources/g1_funnel_builder/diffusion.py`: DDPM loss and DDIM sampler.
- `resources/g1_funnel_builder/proposal_artifact.py`: strict cross-language
  proposal artifact.
- `interaction_funnel_artifact.{h,cpp}`: strict C++ loading/condition matching.
- `interaction_funnel_bundle.{h,cpp}`: strict evaluation-manifest/bundle
  mapping, path, seed, and artifact identity validation.
- `interaction_funnel_selection.{h,cpp}`: covariance, full swept certification,
  runtime preview, and deterministic ranking.
- `interaction_funnel_follower.{h,cpp}`: exact 25-Hz target schedule.
- `interaction_learned_pickup_diagnostics.h`: cycle-free follower/backend audit
  value types shared with ordinary pickup diagnostics.
- `interaction_learned_pickup_backend.{h,cpp}`: proposal-provider replacement
  behind `SmartPickupAssistBackend`.
- `interaction_funnel_catalog.{h,cpp}`: exact condition-digest lookup for the
  Task-7 winning provider at the table and every Plan A certified rack tier.
- `interaction_funnel_autodemo_config.{h,cpp}`: pure four-mode environment
  validation before Raylib initialization.
- `resources/verify_g1_funnel_playable_catalog.py`: role/identity/path/staging
  verifier for generated playable artifacts.

---

### Task 0: Create and prove the pinned Python environment

**Files:**

- Create: `requirements-funnels.txt`

**Interfaces:**

- Produces: the sole Plan B interpreter at
  `build/venvs/g1-funnels/bin/python`, exactly Python 3.10.13, NumPy 2.2.6,
  and Torch 2.5.1.

- [ ] **Step 1: Write the exact dependency lock**

  `requirements-funnels.txt` contains exactly:

  ```text
  numpy==2.2.6
  torch==2.5.1
  ```

- [ ] **Step 2: Create the environment and verify exact versions before Task 1**

  ```bash
  rm -rf build/venvs/g1-funnels
  uv venv build/venvs/g1-funnels \
    --python /home/ubuntu/miniconda3/bin/python3.10
  uv pip install --python build/venvs/g1-funnels/bin/python \
    -r requirements-funnels.txt
  build/venvs/g1-funnels/bin/python - <<'PY'
  import platform
  import numpy
  import torch
  assert platform.python_version() == "3.10.13"
  assert numpy.__version__ == "2.2.6"
  assert torch.__version__.split("+")[0] == "2.5.1"
  print(platform.python_version(), numpy.__version__, torch.__version__)
  PY
  ```

  Expected: the three assertions pass. Stop before Task 1 on any mismatch.

- [ ] **Step 3: Commit the lock before dataset work**

  ```bash
  git add requirements-funnels.txt
  git diff --cached --check
  git commit -m "build: pin funnel training environment"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 1: Export the deterministic object-anchored funnel dataset

**Files:**

- Create: `resources/g1_funnel_builder/__init__.py`
- Create: `resources/g1_funnel_builder/schema.py`
- Create: `resources/g1_funnel_builder/dataset.py`
- Create: `resources/g1_funnel_builder/dataset_artifact.py`
- Create: `resources/g1_funnel_builder/splits.py`
- Create: `resources/export_g1_funnel_dataset.py`
- Create: `tests/python/test_g1_funnel_schema.py`
- Create: `tests/python/test_g1_funnel_dataset.py`
- Create: `tests/python/test_g1_funnel_dataset_artifact.py`
- Create: `tests/python/test_g1_funnel_splits.py`

**Interfaces:**

- Consumes: `read_artifact_set(pack)`, manifest clip/object joins, Simulation
  root at bone zero, first Reach, Contact-minus-one object frame.
- Produces: `GraspCondition`, `FunnelRow`, exact reversal digests, independent
  object metadata, train/validation/test-ready deterministic artifact arrays.

- [ ] **Step 1: Write failing schema and reversal tests**

  Define the public dataclasses in tests first:

  ```python
  @dataclass(frozen=True)
  class GraspCondition:
      active_hand: int
      grasp_translation_object: np.ndarray       # (3,) float32
      grasp_rotation_6d_object: np.ndarray       # (6,) float32
      approach_xz_object: np.ndarray              # (2,) float32
      object_dimensions: np.ndarray               # (3,) float32
      support_height_m: np.float32
      grasp_height_above_support_m: np.float32

  @dataclass(frozen=True)
  class FunnelRow:
      stable_key: str
      object_id: str
      sequence_id: str
      condition: GraspCondition
      outward: np.ndarray                         # (16, 4)
      execution: np.ndarray                       # (16, 4)
      reversal_digest: str

  vector = grasp_condition_model_vector(condition)
  digest = grasp_condition_digest(condition)
  assert vector.shape == (18,) and vector.dtype == np.float32
  assert len(digest) == 64
  ```

  Require `execution == outward[::-1]` bit-for-bit, `execution[-1]` is the
  first Reach root, yaw pairs are unit length within `0.00002`, condition length
  is 18, IDs never appear in it, and changing either height scalar changes the
  condition digest exactly once.

  Freeze `reversal_digest`/`reversal_sha256` to one byte-level formula used by
  every Python writer and C++ reader. For a non-endpoint row, first require
  finite C-contiguous float32 arrays with shapes `outward=[16,4]` and
  `execution=[16,4]`, convert each value to its canonical IEEE-754 little-endian
  float32 bytes without normalizing signed zero, and require the execution bytes
  to equal the C-order bytes of `outward[::-1]` bit-for-bit. Then compute:

  ```python
  def reversal_sha256(outward: np.ndarray, execution: np.ndarray) -> bytes:
      outward_le = np.ascontiguousarray(outward, dtype="<f4")
      execution_le = np.ascontiguousarray(execution, dtype="<f4")
      if outward_le.shape != (16, 4) or execution_le.shape != (16, 4):
          raise ValueError("reversal shape")
      if not np.all(np.isfinite(outward_le)) or not np.all(np.isfinite(execution_le)):
          raise ValueError("nonfinite reversal")
      outward_bytes = outward_le.tobytes(order="C")
      execution_bytes = execution_le.tobytes(order="C")
      expected_execution_bytes = np.ascontiguousarray(
          outward_le[::-1], dtype="<f4").tobytes(order="C")
      if execution_bytes != expected_execution_bytes:
          raise ValueError("not an exact reversal")
      return hashlib.sha256(
          b"g1-funnel-reversal-v1\0"
          + struct.pack("<III", 16, 16, 4)
          + outward_bytes
          + execution_bytes
      ).digest()
  ```

  The three packed integers are respectively outward sample count, execution
  sample count, and sample width. Sample order is outward first and execution
  second; field order within every sample is `x,z,sin_yaw,cos_yaw`. C++ appends
  the same canonical little-endian raw float32 bits in that exact order. The
  endpoint-only sentinel remains exactly 32 zero bytes and never invokes this
  formula. `FunnelRow.reversal_digest` and JSON `reversal_sha256` are exactly
  `reversal_sha256(...).hex()` in lowercase; the binary stores the raw 32 bytes.
  Tests pin a shared Python/C++ golden digest and reject a one-bit
  value mutation, count/width mutation, endian swap, array-order swap, and a
  digest computed before the one exact reversal.

  The same RED wave requires the stable split:

  ```python
  split = split_object_ids(object_ids, seed=20260719)
  self.assertEqual((len(split.train), len(split.validation), len(split.test)),
                   (506, 63, 64))
  self.assertFalse(set(split.train) & set(split.validation))
  self.assertFalse(set(split.train) & set(split.test))
  self.assertFalse(set(split.validation) & set(split.test))
  self.assertEqual(split, split_object_ids(reversed(object_ids), 20260719))
  ```

  Sort by `sha256(b"g1-funnel-split-v1\0" + seed_le64 + object_id_utf8)`
  with object ID as the tie-breaker; slice exactly 506/63/64.

  Add order-invariant deduplication tests. The canonical duplicate key is:

  ```python
  payload = (
      grasp_condition_model_vector(row.condition).astype("<f4").tobytes("C")
      + row.outward.astype("<f4").tobytes("C")
  )
  duplicate_key = hashlib.sha256(
      b"g1-funnel-row-dedup-v1\0" + payload
  ).hexdigest()
  ```

  Before deduplication, derive each row's `stable_key` as lowercase hexadecimal
  SHA-256 of
  `b"g1-funnel-row-v1\0" + sequence_id_utf8 + b"\0" + object_id_utf8 +
  clip_ordinal_u32_le + first_reach_local_frame_u32_le`. The clip ordinal is its
  zero-based ordinal in the full-pack manifest's canonical clip array; it is not
  a retained-row index.

  Call `deduplicate_train_validation(train_rows, validation_rows)` without
  passing test rows. For each duplicate key, retain the lexicographically
  smallest UTF-8 `stable_key` training row when any training row exists;
  otherwise retain the smallest validation row. Thus a training row always
  wins over an identical validation row. Record every dropped row as
  `{partition, stable_key, duplicate_of_stable_key, duplicate_key}` sorted by
  `(partition, stable_key)`. Tests must prove input-order invariance, train-over-
  validation precedence, same-partition stable-key precedence, and that an
  identical test row remains present and cannot affect retained train/validation
  rows or the duplicate audit.

- [ ] **Step 2: Run RED**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_schema \
    tests.python.test_g1_funnel_dataset \
    tests.python.test_g1_funnel_dataset_artifact \
    tests.python.test_g1_funnel_splits -v
  ```

  Expected: import failure for `resources.g1_funnel_builder`.

- [ ] **Step 3: Implement the exact condition encoding**

  Use column-major rotation-6D semantics explicitly:

  ```python
  def canonicalize_quaternion(q_wxyz: np.ndarray) -> np.ndarray:
      q = np.asarray(q_wxyz, dtype=np.float64)
      q = q / np.linalg.norm(q)
      first = int(np.flatnonzero(q != 0.0)[0])
      if q[first] < 0.0:
          q = -q
      return q.astype(np.float32)

  def quaternion_to_rotation_6d(q_wxyz: np.ndarray) -> np.ndarray:
      matrix = holden_quat.to_xform(canonicalize_quaternion(q_wxyz))
      return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)

  def grasp_condition_model_vector(
      condition: GraspCondition,
  ) -> np.ndarray:
      hand = np.array(
          [condition.active_hand == 0, condition.active_hand == 1],
          dtype=np.float32,
      )
      return np.concatenate((
          hand,
          condition.grasp_translation_object,
          condition.grasp_rotation_6d_object,
          condition.approach_xz_object,
          condition.object_dimensions,
          np.array([
              condition.support_height_m,
              condition.grasp_height_above_support_m,
          ], dtype=np.float32),
      )).astype(np.float32)

  def grasp_condition_digest(condition: GraspCondition) -> str:
      hand = struct.pack("<I", condition.active_hand)
      continuous = grasp_condition_model_vector(condition)[2:].astype("<f4")
      return hashlib.sha256(hand + continuous.tobytes(order="C")).hexdigest()
  ```

  Normalize horizontal approach and reject norm below `1e-5`. Canonicalize
  quaternion sign by making the first nonzero component positive.

- [ ] **Step 4: Implement first-Reach window extraction**

  ```python
  first_reach = int(np.flatnonzero(phases == int(InteractionPhase.REACH))[0])
  first_contact = int(
      np.flatnonzero(phases == int(InteractionPhase.CONTACT))[0]
  )
  if first_reach < 15 or first_contact <= first_reach:
      raise FunnelRejection("incomplete_window")
  object_frame = clip_start + first_contact - 1
  frame_indices = clip_start + np.arange(first_reach - 15, first_reach + 1)
  execution = roots_in_object_frame(
      artifact.positions[frame_indices, 0],
      artifact.rotations[frame_indices, 0],
      artifact.object_positions[object_frame],
      artifact.object_rotations[object_frame],
  )
  outward = execution[::-1].copy()
  ```

  Derive support height from table top along its rotated normal. Derive grasp
  world height from the frozen object/grasp transform and require its difference
  from the two height scalars' float64 sum at most `0.00002`.

- [ ] **Step 5: Enforce nondegeneracy and bounded motion**

  ```python
  delta = np.diff(execution[:, :2], axis=0)
  arc_m = float(np.linalg.norm(delta, axis=1).sum(dtype=np.float64))
  yaw = np.arctan2(execution[:, 2], execution[:, 3])
  yaw_delta = np.abs(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))
  if arc_m < 0.15:
      raise FunnelRejection("short_arc")
  if np.max(np.linalg.norm(delta, axis=1)) > 0.08:
      raise FunnelRejection("translation_step")
  if np.max(yaw_delta) > np.deg2rad(15.0):
      raise FunnelRejection("yaw_step")
  if meaningful_cluster_count(execution, 0.05, np.deg2rad(5.0)) < 3:
      raise FunnelRejection("waypoint_clusters")
  ```

- [ ] **Step 6: Write a deterministic directory artifact**

  Run the exact train/validation deduplication first. Preserve every test row,
  including test rows whose condition or duplicate digest equals another row.
  Final retained artifact order is: retained training rows by UTF-8
  `stable_key`, retained validation rows by UTF-8 `stable_key`, then all test
  rows by UTF-8 `stable_key`. Assign `source_row_index_u32=0..R-1` in precisely
  that final order. This retained artifact index is the only meaning of
  `source_row_index` in every later task; original full-pack location remains
  `clip_ordinal_u32`. The retained training rows therefore occupy exactly
  `[0, train_count)` and validation/test follow contiguously.

  Write NumPy format 2.0 with `allow_pickle=False`; every array is C contiguous
  and has exactly this name, dtype, and shape:

  ```text
  conditions.npy       <f4 [R,18]
  outward.npy          <f4 [R,16,4]
  execution.npy        <f4 [R,16,4]
  row_partition.npy    |u1 [R]       (0=train, 1=validation, 2=test)
  source_row_index.npy <u4 [R]       (bit-equal to arange(R,dtype="<u4"))
  condition_mean.npy   <f4 [18]
  condition_scale.npy  <f4 [18]
  outward_mean.npy     <f4 [16,4]
  outward_scale.npy    <f4 [16,4]
  ```

  Means and population standard deviations use two scalar float64 left-fold
  passes over retained training rows in retained order for each element: first
  `sum/count`, then `sum((x-mean)^2)/count`, then `sqrt`; no vectorized or tree
  reduction is permitted. Cast once to float32, and every
  scale below `1e-5` is replaced by float32 `1e-5`. Validation/test never affect
  them. `normalization_sha256` is SHA-256 of
  `b"g1-funnel-normalization-v1\0"` followed by the raw SHA-256 bytes of
  `condition_mean.npy`, `condition_scale.npy`, `outward_mean.npy`, and
  `outward_scale.npy` in that order.

  Canonical JSON bytes everywhere in this artifact are
  `json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,
  allow_nan=False).encode("utf-8") + b"\n"`. `rows.jsonl` contains one such
  object per retained row, with no missing or extra keys and this exact schema:

  ```text
  schema_version_u32: 1
  source_row_index_u32: integer
  partition: "train" | "validation" | "test"
  stable_key, object_id, sequence_id: string
  clip_ordinal_u32: integer
  clip_range_start_u64, clip_range_stop_exclusive_u64: integer
  first_reach_local_frame_u32, first_contact_local_frame_u32: integer
  contact_minus_one_global_frame_u64: integer
  window_start_global_frame_u64, window_stop_exclusive_global_frame_u64: integer
  active_hand_u32: 0 | 1
  condition_sha256, reversal_sha256: lowercase 64-hex string
  ```

  `manifest.json` uses the same canonical encoding and has no missing or extra
  top-level keys:

  ```text
  schema_version_u32: 1
  kind: "g1_funnel_dataset"
  fps_numerator_u32: 25
  fps_denominator_u32: 1
  sample_count_u32: 16
  sample_width_u32: 4
  model_condition_width_u32: 18
  retained_row_count_u32: R
  partition_counts: {train_u32,validation_u32,test_u32}
  partition_object_ids: {train:[string],validation:[string],test:[string]}
  pack: {database_sha256,features_sha256,manifest_sha256}
  extraction:
    {reach_window_u32:16,contact_object_frame_offset_i32:-1,
     minimum_arc_m:0.15,cluster_position_m:0.05,
     cluster_yaw_radians:0.08726646259971647,
     maximum_translation_step_m:0.08,
     maximum_yaw_step_radians:0.261799388,
     height_tolerance_m:0.00002,approach_minimum_norm:0.00001}
  deduplication:
    {domain:"g1-funnel-row-dedup-v1",retained_train_u32,
     retained_validation_u32,dropped_count_u32,dropped:[object]}
  normalization:
    {source_partition:"train",minimum_scale:0.00001,
     condition_mean_file:"condition_mean.npy",
     condition_scale_file:"condition_scale.npy",
     outward_mean_file:"outward_mean.npy",
     outward_scale_file:"outward_scale.npy",normalization_sha256}
  files: {filename:{kind,dtype,shape:[integer],order,sha256}}
  rejection_counts:
    {incomplete_window_u32,invalid_hand_u32,invalid_approach_u32,
     invalid_quaternion_u32,height_mismatch_u32,nonfinite_u32,short_arc_u32,
     translation_step_u32,yaw_step_u32,waypoint_clusters_u32}
  manifest_core_sha256: lowercase 64-hex string
  content_sha256: lowercase 64-hex string
  ```

  `files` contains exactly the nine NPY files above plus `rows.jsonl`, keyed in
  filename byte order. NPY entries have `kind:"npy"`, their exact dtype above,
  exact array shape, and `order:"C"`. The rows entry has `kind:"jsonl"`,
  `dtype:"utf8"`, `shape:[R]`, and `order:"retained_row_order"`; every shape
  value is a JSON integer. Every duplicate
  audit entry has exactly `partition`, `stable_key`,
  `duplicate_of_stable_key`, and `duplicate_key`, with the sorting already
  frozen in Step 1. Zero rejection and duplicate counts remain present.

  Avoid a circular digest as follows. First write all ten payload files and
  compute their hashes. Build `manifest_core` from every manifest field above
  except `manifest_core_sha256` and `content_sha256`; compute
  `manifest_core_sha256=SHA256(canonical_json_bytes(manifest_core))`. Then compute
  `content_sha256` from `b"g1-funnel-dataset-v1\0"`, the raw core digest, and,
  for every `files` key in filename byte order, its UTF-8 byte length as u32 LE,
  its bytes, and its raw file digest. Finally add the two digest fields and write
  `manifest.json`; the manifest file is deliberately not in `files` or in its
  own content digest. Downstream `dataset_sha256` means this exact
  `content_sha256`. Tests mutate every array, row key, audit/rejection value,
  dtype/shape, and ordering, prove digest rejection without circularity, and
  require two exports to be byte-identical.

- [ ] **Step 7: Run GREEN and export the reviewed dataset**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_schema \
    tests.python.test_g1_funnel_dataset \
    tests.python.test_g1_funnel_dataset_artifact \
    tests.python.test_g1_funnel_splits -v
  rm -rf build/g1-funnels/dataset-v1
  build/venvs/g1-funnels/bin/python \
    resources/export_g1_funnel_dataset.py \
    --input build/smart-pickup/full-pack \
    --output build/g1-funnels/dataset-v1 \
    --seed 20260719
  build/venvs/g1-funnels/bin/python \
    resources/export_g1_funnel_dataset.py \
    --input build/smart-pickup/full-pack \
    --output build/g1-funnels/dataset-v1-repeat \
    --seed 20260719
  diff -r build/g1-funnels/dataset-v1 \
    build/g1-funnels/dataset-v1-repeat
  ```

  Expected: tests pass and `diff` has no output.

- [ ] **Step 8: Review, commit, and push**

  ```bash
  git add resources/g1_funnel_builder/__init__.py \
    resources/g1_funnel_builder/schema.py \
    resources/g1_funnel_builder/dataset.py \
    resources/g1_funnel_builder/dataset_artifact.py \
    resources/g1_funnel_builder/splits.py \
    resources/export_g1_funnel_dataset.py \
    tests/python/test_g1_funnel_schema.py \
    tests/python/test_g1_funnel_dataset.py \
    tests/python/test_g1_funnel_dataset_artifact.py \
    tests/python/test_g1_funnel_splits.py
  git diff --cached --check
  git commit -m "feat: export object-anchored funnel dataset"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 2: Freeze the baselines, scenes, and evaluation manifest

**Files:**

- Create: `resources/g1_funnel_builder/baselines.py`
- Create: `resources/g1_funnel_builder/evaluation.py`
- Create: `resources/g1_funnel_builder/geometry.py`
- Create: `resources/freeze_g1_funnel_evaluation.py`
- Create: `resources/evaluate_g1_funnel_providers.py`
- Create: `tests/python/test_g1_funnel_baselines.py`
- Create: `tests/python/test_g1_funnel_evaluation.py`
- Create: `tests/python/test_g1_funnel_geometry.py`
- Generate/Track: `tests/fixtures/g1_funnel_manifest_geometry_v1.json`
- Generate/Track: `resources/g1_funnels/evaluation_manifest_v1.json`

**Interfaces:**

- Produces: exact object partitions, normalized KNN and true endpoint-only
  providers, immutable all-test conditions, a separate 12-row runtime subset,
  NN-derived scenes, canonical JSON, and the exact condition/provider output
  index consumed unchanged by Tasks 4, 7, and 8.

- [ ] **Step 1: Write failing baseline budget/order tests**

  ```python
  full = knn_funnel_proposals(train, condition, k=32)
  endpoint = endpoint_only_proposals(train, condition, k=32)
  self.assertEqual(full.execution.shape, (32, 16, 4))
  self.assertEqual(endpoint.terminals.shape, (32, 4))
  self.assertFalse(hasattr(endpoint, "execution"))
  expected_source_keys = tuple(
      row.stable_key
      for row in sorted(
          train,
          key=lambda row: (
              normalized_condition_distance_squared_f64(
                  condition, row.condition, normalization),
              row.stable_key.encode("utf-8"),
          ),
      )[:32]
  )
  self.assertEqual(full.source_keys, expected_source_keys)
  ```

  For query vector `q` and retained training vector `x`, compute exactly:

  ```python
  q64 = q.astype(np.float64)
  x64 = x.astype(np.float64)
  mean64 = train_condition_mean.astype(np.float64)
  scale64 = train_condition_scale.astype(np.float64)
  distance_squared_f64 = np.sum(
      ((q64 - mean64) / scale64 -
       (x64 - mean64) / scale64) ** 2,
      dtype=np.float64,
  )
  ```

  Stable-sort the complete retained training set by
  `(distance_squared_f64, source_stable_key.encode("utf-8"))`; take exactly the
  first 32. Do not use object identity, approximate search, `argpartition`, or
  float32 accumulation. Tests include exact-distance ties and input-order
  reversal.

  Endpoint-only means connector directly to each retrieved terminal root. It
  never fabricates or repeats 16 samples and is serialized as provider kind
  `endpoint_only` with sample count 1.

- [ ] **Step 2: Run RED**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_baselines \
    tests.python.test_g1_funnel_evaluation -v
  ```

- [ ] **Step 3: Freeze the exact canonical manifest schema**

  Write UTF-8 JSON using
  `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
  allow_nan=False) + "\n"`. Digests are 64 lowercase hexadecimal characters;
  IDs/keys/provider names are strings; `_u32`/`_u64` fields are nonnegative JSON
  integers in range; `_m`, `_radians`, `_seconds`, and `_f64` fields are finite
  JSON numbers. Arrays retain the order specified below. The top-level schema is
  exactly:

  ```text
  schema_version_u32: 1
  kind: "g1_funnel_evaluation_manifest"
  seed_u64: 20260719
  dataset_sha256, pack_database_sha256, pack_features_sha256,
    pack_manifest_sha256, pack_sha256, normalization_sha256: string
  proposal_budget_u32: 32
  provider_order: ["endpoint_only", "knn_funnel", "diffusion"]
  split:
    train_object_ids, validation_object_ids, test_object_ids: [string]
  knn:
    distance_name: "train_normalized_condition_squared_l2_f64"
    accumulation_order: "condition_index_0_through_17"
    tie_break: "source_stable_key_utf8_bytes"
    k_u32: 32
  support_height_quantiles:
    method: "sorted_linear_interpolation_f64_v1"
    lower_numerator_u32/lower_denominator_u32: 1/3
    upper_numerator_u32/upper_denominator_u32: 2/3
    lower_m, upper_m: number
  metrics:
    valid_at_32: {denominator: "all_test_conditions", budget_u32: 32}
    attach_at_32: {denominator: "runtime_cases", budget_u32: 32}
    endpoint_clusters:
      position_threshold_m: 0.10
      yaw_threshold_radians: 0.17453292519943295
      algorithm: "greedy_stable_proposal_order_v1"
    path_clusters:
      position_threshold_m: 0.10
      yaw_threshold_radians: 0.17453292519943295
      algorithm: "all_corresponding_samples_greedy_stable_order_v1"
      endpoint_only_representation: "not_applicable_sample_count_1"
      knn_funnel_representation: "execution_samples_16"
      diffusion_representation: "execution_samples_16"
    blocked_alternate:
      {denominator:"runtime_cases",
       applicability:"nominal_rank_zero_rejected_by_frozen_blocker",
       non_applicable_row_success:"null",
       aggregate:"all_12_applicable_and_successful"}
    all_blocked_fail_closed: {denominator: "runtime_cases"}
    continuation_rule: "attach_ge_best_and_cluster_or_blocked_strict_v1"
  test_conditions: [object]
  runtime_cases: [object]
  runtime_selection_rejections: [object]
  proposal_outputs: [object]
  ```

  `pack_database_sha256`, `pack_features_sha256`, and
  `pack_manifest_sha256` are the three lowercase hexadecimal raw file digests
  from the strict Task-1 full-pack load. `pack_sha256` is the lowercase
  hexadecimal encoding of
  `SHA256(b"g1-funnel-pack-v1\0" + raw_database_sha256 +
  raw_features_sha256 + raw_pack_manifest_sha256)`, concatenating the three raw
  32-byte values in exactly that order, never their hexadecimal text. Manifest
  generation recomputes it and rejects any disagreement with the Task-1 dataset
  manifest; later consumers compare all four fields to the active pack rather
  than trusting this JSON alone.

  `test_conditions` contains every immutable test row, stable-sorted by UTF-8
  `stable_key`, with contiguous `condition_index_u32`. Each entry contains:

  ```text
  condition_index_u32, stable_key, object_id, sequence_id,
  condition_digest, source_row_index_u32
  condition:
    active_hand_u32
    grasp_translation_object_m[3]
    grasp_rotation_6d_object[6]
    approach_xz_object[2]
    object_dimensions_m[3]
    support_height_m
    grasp_height_above_support_m
  ```

  `proposal_outputs` contains exactly
  `len(test_conditions) * 3` entries in `(condition_index_u32, provider_order)`
  order. Each contains `condition_index_u32`, `condition_digest`,
  `provider_kind`, `seed_u64`, and `relative_file`. Derive the seed as the
  little-endian uint64 in the first eight SHA-256 bytes of
  `b"g1-funnel-proposal-seed-v1\0" + manifest_seed_le64 +
  condition_index_u32_le + raw_condition_digest + provider_kind_u32_le`, where
  provider kinds are exactly endpoint-only `1`, KNN `2`, and diffusion `3`.
  The path is exactly
  `conditions/<condition_index_u32-as-8-lowercase-hex>-<condition_digest>/`
  `<provider_kind>.g1funl`. A test partition may contain multiple rows with the
  same condition digest; those are distinct condition instances with distinct
  indices, seeds, output paths, and report rows. No later lookup may key a test
  proposal solely by condition digest.

- [ ] **Step 4: Freeze quantiles, scene eligibility, and blockers**

  Compute quantiles over the support heights of all immutable test conditions,
  before scene eligibility. For sorted float64 values `v` and rational quantile
  `p/q`, use integer `numerator=(len(v)-1)*p`, `j=numerator//q`,
  `remainder=numerator%q`, float64 `r=remainder/q`, and
  `v[j] + r*(v[min(j+1,n-1)]-v[j])`. Tests use hand-computed odd and even
  arrays. Bands are `[min,lower]`, `(lower,upper]`, `(upper,max]`.

  Within each band stable-sort by
  `sha256(b"g1-funnel-runtime-v1\0" + stable_key_utf8)`, then UTF-8 stable key.
  Because Task 5 does not exist yet, all pre-training scene decisions use the
  authoritative `certify_manifest_route_geometry()` in
  `resources/g1_funnel_builder/geometry.py`, never an informal endpoint check.
  Its configuration values are the exact IEEE-754 float32 values obtained from
  connector maximum/tolerance `1.00/0.00002 m`, table expansion `0.24 m`,
  obstacle root radius `0.60 m`, and obstacle safety margin `0.00002 m`.
  Geometry promotes those float32 values to float64; additions corresponding to
  C++ float expressions, including maximum plus tolerance and obstacle radius
  plus margin, round once in float32 before promotion. For every connector and
  proposal segment:

  - reject length above `1.00 + 0.00002 m` only for the connector;
  - transform endpoints into table-local X/Z, expand table half extents by
    `0.24 m`, and use the closed-segment slab test with inclusive boundaries;
  - for each world-axis-aligned 3D obstacle, compute the exact minimum squared
    Euclidean distance from the closed 3D segment to its unexpanded AABB by
    sorting parameters `0`, `1`, and every strict in-segment crossing of its six
    AABB planes, minimizing the piecewise quadratic on every interval exactly as
    `interaction_pick_slots.cpp::closed_segment_aabb_squared_distance`; reject
    when that distance is `<= (0.60 + 0.00002)^2`; and
  - test connector first, then funnel segments in increasing index, returning
    the first reason/segment/obstacle index.

  The Python geometry result also publishes the canonical finite clearance
  defined in Task 5. The shared fixture records that clearance's float32 bits
  and quantized millimetres for every clear case, including the obstacle-free
  cap, so Task 5 must match more than the boolean/reason/index result. Every
  fixture result has exact keys `minimum_clearance_f32_hex` and
  `minimum_clearance_mm_u64`; clear cases contain the lowercase `0x` plus eight
  float32-bit hex digits and the ranked integer, while rejected cases contain
  `"0x00000000"` and JSON null because they are never ranked.

  Task 2 commits hand-computed clear, tangent, table-blocked, connector-blocked,
  middle-segment-blocked, alternate-clear, and all-blocked cases in
  `tests/fixtures/g1_funnel_manifest_geometry_v1.json`. Task 5's C++ test must
  consume this same fixture and match every boolean/reason/index plus clear-route
  clearance float32 bits and quantized millimetres before its own ranking work
  can pass.

  A row is scene-eligible only if all of these hold without changing constants:

  1. normalized KNN returns exactly 32 distinct stable training rows;
  2. its rank-zero complete route is finite and passes nominal connector plus
     all-segment geometry certification;
  3. the route has an earliest segment whose midpoint is at least `0.20 m` from
     both entrance and terminal;
  4. a `(0.24,1.00,0.24) m` AABB centered at that midpoint's X/Z and the frozen
     root-proxy Y `0.0 m` rejects rank zero;
  5. at least one of KNN ranks 1..31 remains geometrically clear with that exact
     AABB; and
  6. a `(4.00,1.00,4.00) m` AABB centered at target X/Z and root-proxy Y rejects
     all 32 under the frozen one-metre connector envelope.

  Select the first four eligible rows per band. Record every skipped candidate
  in `runtime_selection_rejections` as
  `{band_u32, stable_key, reason, knn_rank0_source_key}`. If any band yields
  fewer than four, fail nonzero, remove the temporary manifest, and do not alter
  blocker size/location or borrow from another band.

  Each of the 12 `runtime_cases`, ordered by `(band_u32, rank_in_band_u32)`, is
  an index into `test_conditions`, never a second condition definition. It has
  exactly this schema (every transform rotation is W/X/Y/Z):

  ```text
  runtime_case_index_u32, condition_index_u32, condition_digest,
  band_u32, rank_in_band_u32, request_id_u64, root_proxy_y_m
  target:
    handle: {id_u64,generation_u32}
    object_profile_id_u64, state: "Free", owner_request_u64: 0
    object_world: {position_m[3],rotation_wxyz[4]}
    object_bounds: {center_object_m[3],half_extents_object_m[3]}
    object_dimensions_m[3]
    table_world: {position_m[3],rotation_wxyz[4]}
    table_size_m[3]
    selected_affordance_id_u32: 1
    affordances: [{
      id_u32:1, hand_u32,
      hand_in_object:{position_m[3],rotation_wxyz[4]},
      approach_direction_object[3], clearance_radius_m:0.04,
      interaction_slots:[]
    }]
  live_root_world: {position_m[3],rotation_wxyz[4]}
  knn_nominal_rank0:
    {proposal_index_u32:0,source_row_index_u32,source_stable_key}
  scenes:
    nominal: {scene_id,obstacles:[]}
    blocked_alternate:
      {scene_id,blocker_source_provider:"knn_funnel",
       blocker_source_proposal_u32:0,blocker_source_segment_u32,
       obstacles:[{center_world_m[3],size_world_m[3]}]}
    all_blocked:
      {scene_id,obstacles:[{center_world_m[3],size_world_m[3]}]}
  ```

  Construct the target deterministically from the referenced condition. Handle
  ID is `0x46554e4e00000000 + runtime_case_index_u32`, generation is `1`, and
  `object_profile_id_u64` is the little-endian uint64 in the first eight raw
  condition-digest bytes bitwise-OR `1`, guaranteeing a nonzero profile. Object
  X/Z is `(0.0,3.0)`, its rotation is identity,
  and object Y is
  `support_height_m + grasp_height_above_support_m -
  grasp_translation_object_m[1]`; this makes derived grasp world height exact.
  Bounds center is `(0,0,0)` and half extents are exactly one half the condition
  dimensions. Table size is `(2.0,0.10,2.0)`, table rotation is identity, and
  table position is `(0.0,support_height_m-0.05,3.0)`, so its top is exactly the
  support height.

  Reconstruct `hand_in_object.rotation_wxyz` from the condition's two 6D columns
  by float64 Gram-Schmidt: normalize column 0, subtract its projection from
  column 1 and normalize, take column 2 as their cross product, convert the
  resulting matrix to a unit quaternion, and choose the sign whose first nonzero
  W/X/Y/Z component is positive. Reject a source condition if either norm is
  below `1e-8` or reconstruction differs from the serialized columns by more
  than `0.00002`. Affordance hand/translation are copied exactly, approach is
  `(approach_x,0,approach_z)`, clearance is `0.04`, and authored slots are empty.

  Live-root Y is exactly the frozen `root_proxy_y_m=0.0`; its rotation equals the
  KNN rank-zero ground-truth entrance rotation, and its X/Z is constructed
  `0.25 m` outward from that entrance along terminal-to-entrance X/Z direction.
  Reject a zero terminal-to-entrance norm. `request_id_u64` is
  `0x46554e4e10000000 + runtime_case_index_u32`. Scene IDs are exactly
  `runtime-<two-decimal-case-index>-nominal`,
  `runtime-<two-decimal-case-index>-blocked-alternate`, and
  `runtime-<two-decimal-case-index>-all-blocked`. The blocked obstacle has size
  `(0.24,1.00,0.24)` and center `(midpoint_x,0.0,midpoint_z)`; the all-blocked
  obstacle has size `(4.00,1.00,4.00)` and center `(0.0,0.0,3.0)`. All target
  fields, support fields, coordinates, obstacle order, scene IDs, and condition
  indices are frozen before training.

  The blocker is intentionally derived only from the frozen KNN route; do not
  claim that it blocks another provider without measuring that provider. For a
  provider/runtime row, first run the nominal scene and define
  `nominal_rank0_proposal_u32` as the stable proposal selected by the complete
  nominal geometry/preview ranking. In the blocked scene set
  `blocked_alternate_applicable=true` only when that exact nominal proposal is
  rejected by the blocker before movement. If applicable, success is true only
  when a different proposal from the same 32-member artifact reaches actual
  attached Carry. If the nominal proposal remains viable, applicability is
  false and row success is JSON null—not success. Missing nominal selection is
  also non-applicable/null. Per provider report both counts and
  `blocked_all_applicable_and_success`, which is true only when all 12 rows are
  applicable and successful; non-applicable rows can never create a continuation
  advantage.

- [ ] **Step 5: Freeze metric definitions and manifest tests**

  The manifest must define `valid@32`, `attach@32`, all-row denominators,
  endpoint greedy clusters at `0.10 m`/`10 degrees`, path clusters requiring all
  corresponding 16 samples within those thresholds, blocked-alternate success,
  all-blocked fail-closed, proposal budget 32, and the continuation rule:

  ```python
  continue_diffusion = (
      diffusion.attach_at_32 >= best_baseline.attach_at_32
      and (
          diffusion.endpoint_clusters > best_baseline.endpoint_clusters
          or (
              diffusion.blocked_all_applicable_and_success
              and not best_baseline.blocked_all_applicable_and_success
          )
      )
  )
  ```

  Path-cluster representation is provider-specific and serialized in metrics.
  KNN/diffusion use `execution_samples_16` and compare every corresponding
  sample under `0.10 m`/`10 degrees`. Endpoint-only uses
  `not_applicable_sample_count_1`: `path_clusters` is JSON null and
  `path_cluster_proposal_count_u32` is zero. It is never expanded, repeated, or
  passed to the 16-sample comparator. Endpoint clusters still compare its true
  terminal roots under the endpoint thresholds. Tests reject a numeric endpoint
  path-cluster count or any endpoint record presented as 16 samples.

  `valid@32`, endpoint/path clusters, nearest-training distance, and duplicate
  reporting cover every `test_conditions` entry. Actual `attach@32`, blocked-
  alternate success, and all-blocked fail-closed cover exactly the 12
  `runtime_cases`; their denominator is always 12 even for missing/invalid
  output. Blocked rows additionally carry exact applicability, so provider
  aggregates expose `blocked_applicable_count_u32`,
  `blocked_success_count_u32`, and `blocked_all_applicable_and_success` without
  treating a null row as success. Tests assert all field names/types/units,
  canonical bytes, complete
  condition/provider mapping, exact quantiles, deterministic scene skips,
  atomic failure when a band cannot supply four rows, and that runtime cases
  only reference immutable condition indices.

- [ ] **Step 6: Run GREEN, generate, inspect, then commit the manifest before training**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_baselines \
    tests.python.test_g1_funnel_evaluation \
    tests.python.test_g1_funnel_geometry -v
  mkdir -p resources/g1_funnels
  build/venvs/g1-funnels/bin/python \
    resources/freeze_g1_funnel_evaluation.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --output resources/g1_funnels/evaluation_manifest_v1.json \
    --seed 20260719 \
    --proposal-budget 32
  build/venvs/g1-funnels/bin/python \
    resources/freeze_g1_funnel_evaluation.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --output build/g1-funnels/evaluation_manifest_repeat.json \
    --seed 20260719 \
    --proposal-budget 32
  cmp resources/g1_funnels/evaluation_manifest_v1.json \
    build/g1-funnels/evaluation_manifest_repeat.json
  ```

  Expected: byte-identical manifests, every test condition indexed, exactly
  three proposal-output mappings per condition, and exactly 12 runtime cases.

- [ ] **Step 7: Review, commit, and push this immutable checkpoint**

  ```bash
  git add resources/g1_funnel_builder/baselines.py \
    resources/g1_funnel_builder/evaluation.py \
    resources/g1_funnel_builder/geometry.py \
    resources/freeze_g1_funnel_evaluation.py \
    resources/evaluate_g1_funnel_providers.py \
    resources/g1_funnels/evaluation_manifest_v1.json \
    tests/python/test_g1_funnel_baselines.py \
    tests/python/test_g1_funnel_evaluation.py \
    tests/python/test_g1_funnel_geometry.py \
    tests/fixtures/g1_funnel_manifest_geometry_v1.json
  git diff --cached --check
  git commit -m "test: freeze funnel evaluation manifest"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

  Record this commit hash in all later checkpoint/proposal metadata. Do not amend
  it after any training or test result.

---

### Task 3: Implement and train the fixed compact DDPM

**Files:**

- Create: `resources/g1_funnel_builder/model.py`
- Create: `resources/g1_funnel_builder/diffusion.py`
- Create: `resources/train_g1_funnel_diffusion.py`
- Create: `tests/python/test_g1_funnel_diffusion.py`
- Create: `tests/python/test_g1_funnel_training.py`

**Interfaces:**

- Consumes: normalized train/validation arrays and the committed evaluation
  manifest identity, but no test values/outcomes.
- Produces: deterministic best-EMA checkpoint with exact architecture,
  optimizer/schedule, normalization, seeds, dataset/manifest identities, and
  validation history.

- [ ] **Step 1: Write failing architecture/schedule tests**

  ```python
  model = FunnelDenoiser()
  x = torch.zeros(2, 4, 16)
  t = torch.tensor([0, 999], dtype=torch.long)
  c = torch.zeros(2, 18)
  self.assertEqual(model(x, t, c).shape, x.shape)
  self.assertEqual(model.width, 64)
  self.assertEqual(len(model.blocks), 4)

  schedule = linear_beta_schedule(1000, 1e-4, 0.02)
  self.assertEqual(schedule.shape, (1000,))
  self.assertEqual(schedule.dtype, torch.float64)
  self.assertEqual(schedule[0].item(), 0.0001)
  self.assertEqual(schedule[-1].item(), 0.02)
  ```

  Test the zero head separately from condition dependence: before an update,
  both conditions must produce exact zeros. Then perform one deterministic
  nonzero-head update and test condition dependence:

  ```python
  torch.manual_seed(1701)
  model = FunnelDenoiser()
  x = torch.arange(128, dtype=torch.float32).reshape(2, 4, 16) / 127.0
  t = torch.tensor([37, 811], dtype=torch.long)
  c0 = torch.zeros(2, 18, dtype=torch.float32)
  c1 = c0.clone()
  c1[:, 2] = 1.0
  self.assertTrue(torch.equal(model(x, t, c0), torch.zeros_like(x)))
  self.assertTrue(torch.equal(model(x, t, c1), torch.zeros_like(x)))
  target = torch.linspace(-1.0, 1.0, 128).reshape_as(x)
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  F.mse_loss(model(x, t, c0), target).backward()
  optimizer.step()
  self.assertFalse(torch.equal(model(x, t, c0), model(x, t, c1)))
  ```

  Also assert:

  ```python
  expected_ddim = [
      999, 978, 958, 937, 917, 897, 876, 856, 835, 815,
      795, 774, 754, 733, 713, 693, 672, 652, 632, 611,
      591, 570, 550, 530, 509, 489, 468, 448, 428, 407,
      387, 366, 346, 326, 305, 285, 265, 244, 224, 203,
      183, 163, 142, 122, 101, 81, 61, 40, 20, 0,
  ]
  self.assertEqual(ddim_indices(1000, 50), expected_ddim)
  ```

  Add a hand-computed two-index DDIM recurrence including terminal
  `alpha_bar_previous=1`, deterministic CPU initial-noise ordering, condition
  and outward normalization/denormalization placement, post-denormalization yaw
  projection with an explicitly float64 norm/division followed by one float32
  cast, `eta=0`, EMA ordering, and checkpoint rejection on any frozen
  configuration mismatch. The fixture uses a yaw pair for which float32 and
  float64 norm evaluation produce different projected float32 bytes and pins
  the latter bytes.

- [ ] **Step 2: Run RED in the already-proved Task 0 environment**

  Run:

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_diffusion \
    tests.python.test_g1_funnel_training -v
  ```

  Expected: import failure for the model/diffusion modules.

- [ ] **Step 3: Implement the exact FiLM residual network**

  ```python
  class FiLMBlock(nn.Module):
      def __init__(self) -> None:
          super().__init__()
          self.norm1 = nn.GroupNorm(8, 64)
          self.conv1 = nn.Conv1d(64, 64, 3, padding=1)
          self.film = nn.Linear(256, 128)
          self.norm2 = nn.GroupNorm(8, 64)
          self.conv2 = nn.Conv1d(64, 64, 3, padding=1)

      def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
          h = self.conv1(F.silu(self.norm1(x)))
          scale, shift = self.film(embedding).chunk(2, dim=-1)
          h = self.norm2(h) * (1.0 + scale[..., None]) + shift[..., None]
          h = self.conv2(F.silu(h))
          return x + h
  ```

  Implement `FunnelDenoiser` exactly from Frozen Model Configuration; assert
  input shapes and float32 dtype.

- [ ] **Step 4: Implement deterministic DDPM training and DDIM sampling**

  Seed Python, NumPy, CPU, and CUDA; enable deterministic algorithms and disable
  cuDNN benchmarking:

  ```python
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  ```

  Implement `ddim_indices()` with integer arithmetic only:

  ```python
  def ddim_indices(training_steps: int, sampling_steps: int) -> list[int]:
      assert training_steps == 1000 and sampling_steps == 50
      ascending = [
          (index * (training_steps - 1)) // (sampling_steps - 1)
          for index in range(sampling_steps)
      ]
      return ascending[::-1]
  ```

  Construct the schedule exactly as frozen globally and serialize its float64
  beta/alpha/alpha-bar bytes in every checkpoint. Training uses one CPU
  `torch.Generator` seeded `2026071901`. At each epoch draw one complete
  `torch.randperm(train_count, generator=train_generator, device="cpu")`; take
  consecutive batches of at most 256 (do not drop the final short batch), then
  draw that batch's int64 timesteps followed by its float32 `[B,4,16]` noise
  from the same generator, both on CPU, before transferring them to CUDA. The
  optimizer-step order is: zero gradients, forward epsilon MSE, backward, clip
  global norm to `1.0`, `optimizer.step()`, then increment completed-step count.

  Initialize EMA as a detached float32 copy of the freshly initialized model
  before step one. Immediately after every optimizer step update each parameter
  in registered-parameter order under `no_grad` as
  `ema = 0.999f * ema + 0.001f * parameter`; copy model buffers exactly after
  the parameter update. Validation happens after the EMA update whenever the
  completed-step count is divisible by 250.

  Precompute validation inputs once from a separate CPU generator seeded
  `2026071902`: validation rows remain in retained `source_row_index` order with
  no shuffle; draw all int64 timesteps first, then one complete float32
  `[validation_count,4,16]` noise tensor. Every validation reuses those bytes and
  the current EMA in eval/no-grad mode. Transfer each consecutive batch to CUDA,
  transfer squared errors back to CPU float64, and sum in
  row/channel/sample order; divide by the exact element count. Restore training
  mode afterward.

  Record validation history before checkpoint choice. The first validation is
  best. Later validation improves only when
  `previous_best_mse - current_mse > 0.000001`; on improvement save that
  post-step EMA and reset the non-improvement count, otherwise increment it.
  Stop immediately after recording the twentieth consecutive non-improvement.
  Maximum 20,000 completed optimizer steps is checked after any validation.
  Checkpoints store current completed step, best step/loss, EMA state,
  normalization arrays, schedule arrays, fixed validation seed and history,
  dataset/manifest identities, and all frozen configuration fields. Never load
  test rows or outcomes.

  Finalize the last normalized recurrence exactly once in
  `resources/g1_funnel_builder/diffusion.py`:

  ```python
  def finalize_ddim_outward(
      sampled_4x16: np.ndarray,
      outward_mean: np.ndarray,
      outward_scale: np.ndarray,
  ) -> np.ndarray:
      normalized = np.ascontiguousarray(
          sampled_4x16.transpose(0, 2, 1), dtype=np.float32)
      outward = np.ascontiguousarray(
          normalized * outward_scale + outward_mean, dtype="<f4")
      yaw64 = outward[..., 2:4].astype(np.float64)
      norm64 = np.sqrt(np.sum(
          yaw64 * yaw64, axis=-1, keepdims=True, dtype=np.float64))
      if not np.all(np.isfinite(norm64)) or np.any(norm64 < 1e-8):
          raise ValueError("invalid denormalized yaw norm")
      outward[..., 2:4] = (yaw64 / norm64).astype("<f4")
      return outward
  ```

  The multiply and add above are separate float32 NumPy operations; fused or
  float64 denormalization is forbidden. DDIM sampling uses the global recurrence
  verbatim, EMA in eval/no-grad mode, stable proposal indices `0..31`, and the
  exact per-condition seed. Tests prove
  that changing batch subdivision cannot change initial noise or proposal order,
  the final `t=0` transition returns `x0`, no random draw occurs after initial
  noise, and yaw is projected only after denormalization using the frozen
  float64-norm/float64-division/float32-cast order. Task 3 returns projected
  C-contiguous `[32,16,4]` outward float32 values; Task 4 must serialize those
  exact bytes without a second projection.

- [ ] **Step 5: Run GREEN and a deterministic tiny training test**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_diffusion \
    tests.python.test_g1_funnel_training -v
  ```

  Expected: same seed produces byte-identical CPU mini-checkpoints; the untouched
  zero head produces exact zero for both conditions, and the deterministic
  one-update head produces different outputs for the two conditions.

- [ ] **Step 6: Train on one L40S**

  ```bash
  mkdir -p build/g1-funnels/training-v1
  CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
    resources/train_g1_funnel_diffusion.py \
    --dataset build/g1-funnels/dataset-v1 \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --output build/g1-funnels/training-v1/checkpoint.pt \
    --history build/g1-funnels/training-v1/history.json \
    --seed 2026071901
  ```

  Expected: finite best validation loss, serialized exact metadata, no test
  metrics, and a checkpoint SHA-256 printed to stdout.

- [ ] **Step 7: Review, commit code, and push**

  ```bash
  git add resources/g1_funnel_builder/model.py \
    resources/g1_funnel_builder/diffusion.py \
    resources/train_g1_funnel_diffusion.py \
    tests/python/test_g1_funnel_diffusion.py \
    tests/python/test_g1_funnel_training.py
  git diff --cached --check
  git commit -m "feat: train compact conditional funnel diffusion"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

  Keep the large checkpoint under `build/`; its digest and reproducible config
  enter the proposal artifact, not Git.

---

### Task 4: Add the strict cross-language proposal artifact

**Files:**

- Create: `resources/g1_funnel_builder/proposal_artifact.py`
- Create: `resources/sample_g1_funnel_proposals.py`
- Create: `tests/python/test_g1_funnel_proposal_artifact.py`
- Create: `interaction_sha256.h`
- Create: `interaction_sha256.cpp`
- Create: `interaction_funnel_artifact.h`
- Create: `interaction_funnel_artifact.cpp`
- Create: `interaction_funnel_bundle.h`
- Create: `interaction_funnel_bundle.cpp`
- Create: `interaction_funnel_condition_probe.cpp`
- Create: `tests/cpp/test_interaction_funnel_artifact.cpp`
- Create: `tests/cpp/test_interaction_funnel_bundle.cpp`
- Modify: `Makefile`

**Interfaces:**

- Produces: provider-kind-aware little-endian artifact, C++ `GraspCondition`,
  32 ordered proposals, both outward/execution representations for 16-sample
  providers, identities, diagnostics, condition validation, and SHA-256.

- [ ] **Step 1: Write failing Python round-trip/corruption tests**

  All integers are unsigned little-endian, all floats are IEEE-754 little-endian,
  digests are raw 32-byte SHA-256 values, strings never appear in the binary,
  and there is no implicit alignment or padding. The exact byte sequence is:

  ```text
  FILE HEADER (348 bytes)
    magic                         char[8] = "G1FUNL01"
    version                       u32 = 1
    endian                        u32 = 0x01020304
    header_bytes                  u32 = 348
    provider_kind                 u32 = 1 endpoint, 2 KNN, 3 diffusion
    fps_numerator                 u32 = 25
    fps_denominator               u32 = 1
    proposal_count                u32 = 32
    sample_count                  u32 = 1 endpoint, 16 KNN/diffusion
    sample_width                  u32 = 4
    condition_float_count         u32 = 16 serialized floats after active_hand
    model_condition_width         u32 = 18 hand one-hot plus 16 continuous
    ddpm_training_steps           u32
    ddim_step_count               u32
    ddim_eta                      f32
    reserved_zero                 u32 = 0
    condition_namespace           u32 = 1 evaluation, 2 external playable
    condition_instance_index      u32
    dataset_retained_row_count    u32
    dataset_training_row_count    u32
    seed                          u64
    schema_sha256                 u8[32]
    pack_sha256                   u8[32]
    dataset_sha256                u8[32]
    manifest_sha256               u8[32]
    checkpoint_sha256             u8[32]
    provider_config_sha256        u8[32]
    normalization_sha256          u8[32]
    condition_sha256              u8[32]

  CONDITION (68 bytes)
    active_hand                   u32, 0 left or 1 right
    grasp_translation_object_m    f32[3], x/y/z
    grasp_rotation_6d_object      f32[6], matrix column 0 then column 1
    approach_xz_object            f32[2], x/z
    object_dimensions_m           f32[3], x/y/z
    support_height_m              f32
    grasp_height_above_support_m  f32

  PROPOSAL RECORDS, stable_index 0..31, each with this prefix (104 bytes)
    stable_index                  u32
    source_row_index              u32
    outward_sample_count          u32
    execution_sample_count        u32
    knn_distance_squared          f64
    arc_length_m                  f32
    waypoint_cluster_count        u32
    maximum_translation_step_m    f32
    maximum_yaw_step_radians      f32
    source_stable_key_sha256      u8[32]
    reversal_sha256               u8[32]
    outward samples               f32[outward_sample_count][4]
    execution samples             f32[execution_sample_count][4]

  TRAILER
    artifact_sha256               u8[32], hash of every preceding byte
  ```

  Provider-specific values are canonical, not optional:

  | Provider | samples | DDPM/DDIM/eta | checkpoint sentinel/identity | provider config | source fields | arrays/diagnostics |
  |---|---:|---|---|---|---|---|
  | endpoint | 1 | `0/0/-1.0f` | exactly 32 zero bytes | SHA-256 of canonical endpoint config JSON | retained training row index, exact normalized KNN distance, SHA-256 of source stable key | outward count 0, execution count 1 at retrieved `execution[-1]`, arc/max steps `+0.0f`, clusters 1, reversal digest 32 zero bytes |
  | KNN | 16 | `0/0/-1.0f` | exactly 32 zero bytes | SHA-256 of canonical KNN config JSON | retained training row index, exact normalized KNN distance, SHA-256 of source stable key | outward/execution counts 16 and measured diagnostics/reversal digest |
  | diffusion | 16 | `1000/50/+0.0f` | SHA-256 of checkpoint file bytes | SHA-256 of canonical frozen model/sampler config JSON | `source_row_index=0xffffffff`, `knn_distance_squared=-1.0`, source-key digest 32 zero bytes | outward/execution counts 16 and measured diagnostics/reversal digest |

  Canonical endpoint configuration JSON has exactly
  `{schema_version_u32:1,provider_kind:"endpoint_only",k_u32:32,
  distance_name:"train_normalized_condition_squared_l2_f64",
  tie_break:"source_stable_key_utf8_bytes",sample_count_u32:1,
  execution_semantics:"connector_direct_to_terminal",dataset_sha256,
  normalization_sha256,manifest_sha256}`. Canonical KNN JSON has those exact
  fields with `provider_kind:"knn_funnel"`, `sample_count_u32:16`, and
  `execution_semantics:"retrieved_complete_execution_funnel"`. Both use the
  Task-2 canonical JSON encoding. Canonical diffusion configuration JSON has
  exactly `schema_version_u32:1`, `provider_kind:"diffusion"`,
  `model_id:"film_conv1d_width64_blocks4_condition18_v1"`,
  `ddpm_training_steps_u32:1000`, `beta_start:0.0001`, `beta_end:0.02`,
  `prediction:"epsilon"`, `ddim_indices` equal to the frozen 50-element list,
  `ddim_eta:0.0`, `proposal_count_u32:32`, `sample_count_u32:16`, and
  `training_seed_u64:2026071901`, `dataset_sha256`, `normalization_sha256`, and
  `manifest_sha256`. `pack_sha256` is exactly
  `SHA256(b"g1-funnel-pack-v1\0" + raw_database_sha256 +
  raw_features_sha256 + raw_pack_manifest_sha256)`, in that order. The three raw
  32-byte digest values are concatenated, never their hexadecimal text. They
  come from the exact full-pack files recorded by Task 1; a relabeled or
  mismatched pack manifest therefore rejects. The Task-2 manifest carries all
  three raw digests plus this combined digest. `schema_sha256` is exactly
  `SHA256(b"g1-funnel-binary-schema-v1\0")`; version/header/count/provider
  validation protects its concrete layout. Dataset/manifest/normalization fields
  hash their canonical artifacts. Tests pin all eight identity fields and reject
  a nonzero baseline checkpoint or zero diffusion checkpoint.

  Define one authoritative active-pack value in C++:

  ```cpp
  struct FunnelPackIdentity {
      std::array<uint8_t, 32> database{};
      std::array<uint8_t, 32> features{};
      std::array<uint8_t, 32> manifest{};
      std::array<uint8_t, 32> combined{};
  };

  FunnelPackIdentity load_funnel_pack_identity(
      const std::filesystem::path& pack_root);
  ```

  `load_funnel_pack_identity` invokes the existing strict Plan-A pack loader on
  `pack_root`, takes the freshly recomputed raw database/features/pack-manifest
  file digests from that validated load (never from a Plan-B JSON copy), and
  recomputes `combined` with the formula above. Empty/zero identities, a pack
  loader mismatch, or a combined-digest mismatch reject. The Python sampler,
  evaluator, and playable-catalog verifier likewise receive `--pack`, hash the
  active strict pack, and compare all four values to the Task-1 dataset and
  Task-2 manifest before reading or publishing proposal results.

  For every KNN/diffusion proposal, the writer computes `reversal_sha256` only
  with Task 1's domain/count/width/outward-bytes/execution-bytes formula after
  exact reversal. The C++ reader reconstructs precisely that byte stream from
  serialized little-endian float32 fields and compares the stored digest.
  Endpoint-only instead requires the exact 32-zero-byte sentinel. Python/C++
  golden-vector tests use the same literal bytes and independently corrupt the
  domain version, each shape integer, ordering, endianness, and one float bit.

  Tests compare every exact offset, total file size `4,288` bytes for endpoint
  (`32 * 120`-byte records) and `20,160` bytes for KNN/diffusion
  (`32 * 616`-byte records), float sentinel bit patterns, provider-specific zero
  fields, and trailer digest. They independently reject
  `condition_float_count!=16` and `model_condition_width!=18`; no other sentinel
  is accepted.

  Evaluation artifacts use namespace `1` and store the Task-2
  `condition_index_u32`; external playable artifacts use namespace `2` and store
  the Task-8 unique-condition `external_artifact_index_u32`. Every artifact
  stores the exact dataset retained/training counts. Endpoint/KNN source row is
  `<dataset_training_row_count` (training rows are the retained prefix) and also
  `<dataset_retained_row_count`; diffusion alone uses `0xffffffff`. Reject zero
  counts, training count above retained count, baseline source sentinels, or a
  source index in validation/test. Provider integer mapping is endpoint `1`, KNN
  `2`, diffusion `3`; seed validation uses that integer and the namespace/index
  formulas in Tasks 2 and 8.

  The proposal bundle is exactly:

  ```text
  <bundle>/bundle_manifest.json
  <bundle>/conditions/<condition_index_8hex>-<condition_digest>/endpoint_only.g1funl
  <bundle>/conditions/<condition_index_8hex>-<condition_digest>/knn_funnel.g1funl
  <bundle>/conditions/<condition_index_8hex>-<condition_digest>/diffusion.g1funl
  ```

  `bundle_manifest.json` uses the Task-2 canonical JSON rule and contains
  `{schema_version_u32:1, evaluation_manifest_sha256, pack_sha256,
  entries:[...]}`. Its
  entries are byte-for-byte the Task-2 `proposal_outputs` mapping plus
  `artifact_sha256`, in the same order. Generation fails atomically on a
  missing/extra/duplicate
  `(condition_index_u32,condition_digest,provider_kind)` tuple or a path that
  differs from `relative_file`. Equal condition digests at different indices
  remain separate entries and files.

- [ ] **Step 2: Write failing C++ loader tests**

  Expose:

  ```cpp
  enum class FunnelProviderKind : uint32_t {
      EndpointOnly = 1U,
      KnnFunnel = 2U,
      Diffusion = 3U,
  };

  struct GraspCondition {
      Hand active_hand = Hand::Right;
      vec3 grasp_translation_object{};
      std::array<float, 6> grasp_rotation_6d_object{};
      std::array<float, 2> approach_xz_object{};
      vec3 object_dimensions{};
      float support_height_m = 0.0F;
      float grasp_height_above_support_m = 0.0F;
  };

  struct FunnelSample {
      float x_object_m = 0.0F;
      float z_object_m = 0.0F;
      float sin_yaw_object = 0.0F;
      float cos_yaw_object = 1.0F;
  };

  struct FunnelProposal {
      uint32_t stable_index = 0U;
      uint32_t source_row_index = 0U;
      uint32_t outward_sample_count = 0U;
      uint32_t execution_sample_count = 0U;
      double knn_distance_squared = 0.0;
      std::array<FunnelSample, 16> outward{};
      std::array<FunnelSample, 16> execution{};
      float arc_length_m = 0.0F;
      uint32_t waypoint_cluster_count = 0U;
      float maximum_translation_step_m = 0.0F;
      float maximum_yaw_step_radians = 0.0F;
      std::array<uint8_t, 32> source_stable_key_digest{};
      std::array<uint8_t, 32> reversal_digest{};
  };

  std::array<uint8_t, 32> funnel_reversal_sha256(
      const std::array<FunnelSample, 16>& outward,
      const std::array<FunnelSample, 16>& execution);

  struct FunnelArtifactIdentity {
      std::array<uint8_t, 32> schema{};
      std::array<uint8_t, 32> pack{};
      std::array<uint8_t, 32> dataset{};
      std::array<uint8_t, 32> manifest{};
      std::array<uint8_t, 32> checkpoint{};
      std::array<uint8_t, 32> provider_config{};
      std::array<uint8_t, 32> normalization{};
      std::array<uint8_t, 32> condition{};
      std::array<uint8_t, 32> artifact{};
  };

  struct LearnedFunnelArtifact {
      FunnelProviderKind provider_kind{};
      uint32_t sample_count = 0U;
      uint32_t condition_float_count = 16U;
      uint32_t model_condition_width = 18U;
      uint32_t ddpm_training_steps = 0U;
      uint32_t ddim_step_count = 0U;
      float ddim_eta = 0.0F;
      uint32_t condition_namespace = 0U;
      uint32_t condition_instance_index = 0U;
      uint32_t dataset_retained_row_count = 0U;
      uint32_t dataset_training_row_count = 0U;
      GraspCondition condition{};
      uint64_t seed = 0U;
      std::array<FunnelProposal, 32> proposals{};
      FunnelArtifactIdentity identity{};
  };

  LearnedFunnelArtifact load_learned_funnel_artifact(
      const std::filesystem::path& path,
      const FunnelPackIdentity& active_pack);
  GraspCondition make_grasp_condition(
      const InteractionTarget& target,
      const GraspAffordance& affordance,
      float flat_ground_height_m = 0.0F);
  bool same_grasp_condition(
      const GraspCondition& left,
      const GraspCondition& right);

  struct FunnelBundleKey {
      uint32_t condition_index = 0U;
      std::array<uint8_t, 32> condition_digest{};
      FunnelProviderKind provider_kind{};
  };

  struct FunnelBundleEntry {
      FunnelBundleKey key{};
      uint64_t seed = 0U;
      std::filesystem::path relative_file{};
      std::array<uint8_t, 32> artifact_digest{};
      LearnedFunnelArtifact artifact{};
  };

  class FunnelArtifactBundle {
  public:
      static FunnelArtifactBundle load(
          const std::filesystem::path& bundle_root,
          const std::filesystem::path& evaluation_manifest,
          const FunnelPackIdentity& active_pack);
      const FunnelBundleEntry& require(const FunnelBundleKey& key) const;
      const std::array<uint8_t, 32>& evaluation_manifest_digest() const;
  };
  ```

  Reject truncation, trailing bytes, wrong magic/version/endian/rate/counts,
  any noncanonical provider sentinel/record field, non-finite floats, non-unit
  yaw, duplicate or noncontiguous stable indices, reversal mismatch, digest
  mismatch, and derived height mismatch. Apply arc/cluster/step nondegeneracy
  only to KNN/diffusion; endpoint-only must contain exactly one terminal and no
  outward values. Require `identity.condition` to equal both the serialized
  condition digest and the expected lookup digest; independently require
  `identity.artifact` to equal the trailer digest. Neither may substitute for
  the other.

  `funnel_reversal_sha256` rejects nonfinite samples and any execution raw bits
  unequal to reversed outward raw bits. It feeds SHA-256 exactly the 22-byte
  domain/version tag including its terminating NUL, `u32_le(16)`,
  `u32_le(16)`, `u32_le(4)`, 64 outward float32 values in C/sample/field order,
  then 64 execution float32 values in that order. Each float contributes its
  four IEEE-754 bytes least-significant byte first. This is the C++ spelling of
  Task 1's Python formula, not a hash of structs or host-endian memory.

  `FunnelArtifactBundle::load` parses the exact canonical Task-2 manifest and
  `bundle_manifest.json`, compares the bundle's evaluation-manifest digest to
  the bytes supplied, requires each of the manifest's three raw pack digests
  and `pack_sha256` to equal `active_pack`, requires the bundle manifest's
  `pack_sha256` to equal the same combined value, and requires every artifact
  header `identity.pack` to equal it. It requires one entry for every Task-2
  `proposal_outputs` row and no other entry. It rejects unknown provider
  names/integers, duplicate
  keys or files, wrong order, seed mismatch under the Task-2 formula,
  missing/extra files under `conditions/`, absolute or non-normalized relative
  paths, empty components, `.`/`..`, backslashes, symlinks at any component,
  paths outside the canonical bundle root, file-hash mismatch, and every header
  identity/count/namespace/index/provider/seed mismatch. Loading is atomic: no
  entry is queryable unless all entries validate. Tests include equal-digest
  distinct condition indices and each rejection independently. Direct artifact
  loads also require `active_pack`; there is no unbound production overload
  that merely trusts the artifact header. Tests independently mutate each raw
  pack digest, the combined digest in every schema, and an artifact header.

- [ ] **Step 3: Run RED**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_proposal_artifact -v
  make build/tests/test_interaction_funnel_artifact \
       build/tests/test_interaction_funnel_bundle
  ```

- [ ] **Step 4: Implement canonical writer and exact-once reversal**

  Call Task 3's finalizer exactly once after DDIM, validate outward order, and
  reverse exactly once:

  ```python
  outward = finalize_ddim_outward(
      sampled, dataset.outward_mean, dataset.outward_scale)
  execution = outward[:, ::-1, :].copy()
  assert np.array_equal(outward, execution[:, ::-1, :])
  reversal_digest = reversal_sha256(outward, execution)
  ```

  The artifact writer accepts only this already-projected outward array and must
  not denormalize or renormalize it. Tests compare Task 3's returned outward
  bytes with Task 4's serialized outward bytes and reject a float32 norm,
  projection before denormalization, or a second projection.

  Write from explicit `struct.pack` calls; never dump native structs or pickle.
  The sampler CLI emits KNN and endpoint-only artifacts through the same writer.
  It iterates the Task-2 `proposal_outputs` mapping, writes each exact relative
  file, then writes `bundle_manifest.json`. `--providers
  endpoint_only,knn_funnel,diffusion` must generate all three artifacts for every
  immutable test condition. `--provider-from-report <report.json>` selects one
  accepted provider without changing samples, seeds, or manifest identity.
  Baseline configuration JSON contains provider kind, normalized-distance name,
  tie rule, K=32, endpoint semantics, and dataset/normalization/manifest digests;
  its canonical SHA-256 occupies `provider_config_sha256`, while the baseline
  `checkpoint_sha256` remains the required 32-zero-byte sentinel.

- [ ] **Step 5: Implement strict C++ loading and condition construction**

  Follow `interaction_database.h`'s checked-count/read/EOF style and the
  repository SHA-256 reference, but expose SHA-256 through the new focused
  module. Reconstruct 6D rotation from the authoritative grasp quaternion and
  compute support height from `target.table_world/table_size`; reject mismatch
  above `0.00002F`.

  `./interaction_funnel_condition_probe --json` constructs the condition through
  `make_grasp_condition(make_smart_pickup_demo_target(), affordance)` and emits
  the exact canonical JSON accepted by the Python sampler. This is the sole
  playable-condition bridge; do not duplicate the authored grasp in Python.

- [ ] **Step 6: Run GREEN and generate provider artifacts twice**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_proposal_artifact -v
  make build/tests/test_interaction_funnel_artifact \
       build/tests/test_interaction_funnel_bundle
  build/tests/test_interaction_funnel_artifact
  build/tests/test_interaction_funnel_bundle
  CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
    resources/sample_g1_funnel_proposals.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --checkpoint build/g1-funnels/training-v1/checkpoint.pt \
    --providers endpoint_only,knn_funnel,diffusion \
    --output build/g1-funnels/proposals-v1
  CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
    resources/sample_g1_funnel_proposals.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --checkpoint build/g1-funnels/training-v1/checkpoint.pt \
    --providers endpoint_only,knn_funnel,diffusion \
    --output build/g1-funnels/proposals-v1-repeat
  diff -r build/g1-funnels/proposals-v1 \
    build/g1-funnels/proposals-v1-repeat
  build/venvs/g1-funnels/bin/python - <<'PY'
  import json
  from pathlib import Path
  manifest = json.loads(Path(
      "resources/g1_funnels/evaluation_manifest_v1.json").read_text())
  bundle = json.loads(Path(
      "build/g1-funnels/proposals-v1/bundle_manifest.json").read_text())
  assert len(bundle["entries"]) == len(manifest["test_conditions"]) * 3
  assert {row["provider_kind"] for row in bundle["entries"]} == {
      "endpoint_only", "knn_funnel", "diffusion"
  }
  PY
  ```

- [ ] **Step 7: Review, commit, and push**

  ```bash
  git add resources/g1_funnel_builder/proposal_artifact.py \
    resources/sample_g1_funnel_proposals.py \
    tests/python/test_g1_funnel_proposal_artifact.py \
    interaction_sha256.h interaction_sha256.cpp \
    interaction_funnel_artifact.h interaction_funnel_artifact.cpp \
    interaction_funnel_bundle.h interaction_funnel_bundle.cpp \
    interaction_funnel_condition_probe.cpp \
    tests/cpp/test_interaction_funnel_artifact.cpp \
    tests/cpp/test_interaction_funnel_bundle.cpp Makefile
  git diff --cached --check
  git commit -m "feat: add versioned funnel proposal artifacts"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 5: Certify and rank the full connector/funnel batch

**Files:**

- Create: `interaction_funnel_selection.h`
- Create: `interaction_funnel_selection.cpp`
- Create: `tests/cpp/test_interaction_funnel_selection.cpp`
- Modify: `interaction_pick_slots.h`
- Modify: `interaction_pick_slots.cpp`
- Modify: `Makefile`

**Interfaces:**

- Consumes: live root, frozen target/condition/artifact, frozen scene obstacles,
  one frozen `PickSlotConfig`, and one snapshot fingerprint. Geometry never
  invokes a preview callback.
- Produces: one geometry batch and exact `PickAssistPreviewRequest`s; on the next
  controller observation, a separate function consumes one complete atomic
  `PickAssistPreviewResult` batch and deterministically ranks it.

- [ ] **Step 1: Write failing covariance/full-sweep/ranking tests**

  Require world translation/yaw covariance, byte-identical local proposals,
  connector and every one of 15 funnel segments swept, endpoint connector-only
  semantics, malformed-step rejection, stable preview request IDs, fixed blocker
  selecting a different member of the same batch, all-blocked no requests, and
  exact integer tie ordering. Preview tests pass results one observation later
  and reject missing, extra, duplicate, reordered-with-mismatched-root, stale-
  fingerprint, or mixed-fingerprint results.

  Use a nondefault valid `PickSlotConfig`, mutate each of its five fields in
  turn, and require initial certification and preview-time recertification to
  use the same frozen bits. Include a forged expected digest paired with unequal
  raw bits to prove digest equality never substitutes for exact config-value
  equality.

  Before those tests, load
  `tests/fixtures/g1_funnel_manifest_geometry_v1.json` and require the C++
  segment sweep to match every Task-2 reason, segment index, and obstacle index.
  This parity gate prevents the pre-training scene eligibility kernel from
  drifting from runtime certification.

  ```cpp
  const FunnelGeometryBatch geometry = certify_funnel_geometry(input);
  const std::vector<PickAssistPreviewRequest> requests =
      make_funnel_preview_requests(geometry);
  const FunnelSelection selected = rank_funnel_preview_batch(
      geometry, preview_observation);
  TEST_CHECK(selected.selected_index.has_value());
  TEST_CHECK(selected.preview_snapshot_fingerprint ==
             preview_observation.snapshot_fingerprint);
  ```

- [ ] **Step 2: Run RED**

  ```bash
  make build/tests/test_interaction_funnel_selection
  ```

- [ ] **Step 3: Expose the shared segment sweep without changing authored behavior**

  Move the current private route result to the header:

  ```cpp
  struct PickRouteEvaluation {
      PickSlotReason reason = PickSlotReason::InvalidGeometry;
      int32_t obstacle_index = -1;
      float minimum_clearance_m = 0.0F;
  };

  PickRouteEvaluation evaluate_pick_route_segment(
      const Transform& start,
      const Transform& end,
      const InteractionTarget& target,
      const std::vector<PickNavigationObstacle>& obstacles,
      const PickSlotConfig& config = {});
  ```

  Keep `select_pick_slot` output and ranking bit-stable under existing tests.
  Its existing collision result remains authoritative; the new clearance is
  diagnostic/ranking data for clear learned routes and cannot change authored
  eligibility.

  Freeze `minimum_clearance_m` as follows. It measures obstacle clearance only;
  the expanded table remains a binary rejection surface and never participates
  in this metric. Let `clearance_cap_f32` be the exact float32
  `config.maximum_direct_travel_m`. Let `obstacle_envelope_f32` be the result of
  the float32 addition
  `config.obstacle_root_radius_m + config.obstacle_safety_margin_m`. For one
  closed 3D route segment and one obstacle, compute
  `raw_squared_distance_f64` with the existing exact distance kernel against the
  obstacle's **unexpanded** AABB, then compute:

  ```text
  raw_distance_f64 = sqrt(raw_squared_distance_f64)
  free_clearance_f64 =
      raw_distance_f64 - float64(obstacle_envelope_f32)
  ```

  A clear segment has strictly positive `free_clearance_f64`. Start a route's
  accumulator at `float64(clearance_cap_f32)`, visit connector first, then
  funnel segments by increasing sample index, and within each segment visit
  obstacles in vector order. For every pair replace the accumulator with
  `min(accumulator, max(0.0, free_clearance_f64))`. Thus no-obstacle routes and
  routes whose obstacles are all farther than the cap have the finite value
  `clearance_cap_f32`; there is no infinity sentinel. After the entire clear
  route, cast the accumulator exactly once to float32 and store those bits as
  `minimum_clearance_m`. A malformed, table-blocked, or obstacle-blocked route
  retains `+0.0F` and is never ranked. Endpoint uses its connector only. The
  minimum covers the connector and all 15 funnel segments for 16-sample
  providers.

- [ ] **Step 4: Implement covariance and route certification**

  ```cpp
  enum class FunnelRejectionReason : uint8_t {
      None,
      Malformed,
      ConnectorTooLong,
      TableBlocked,
      ObstacleBlocked,
      PreviewBatchMissing,
      PreviewBatchMalformed,
      PreviewFingerprintMismatch,
      RuntimePreviewRejected,
  };

  struct FrozenFunnelInputIdentity {
      TargetHandle target{};
      uint32_t affordance_id = 0U;
      std::array<uint8_t, 32> artifact_digest{};
      std::array<uint8_t, 32> condition_digest{};
      std::array<uint32_t, 5> geometry_config_f32_bits{};
      std::array<uint8_t, 32> geometry_config_digest{};
      uint64_t scene_fingerprint = 0U;
      uint64_t obstacle_fingerprint = 0U;
      uint64_t activation_snapshot_fingerprint = 0U;
  };

  struct FunnelGeometryCertification {
      uint32_t proposal_index = 0U;
      std::array<Transform, 16> world_samples{};
      uint32_t sample_count = 0U;
      bool geometry_certified = false;
      FunnelRejectionReason reason = FunnelRejectionReason::None;
      int32_t blocked_segment_index = -1;  // -1 connector, 0..14 funnel
      int32_t obstacle_index = -1;
      uint64_t connector_millimetres = 0U;
      uint64_t funnel_millimetres = 0U;
      uint64_t route_millimetres = 0U;
      uint64_t heading_milliradians = 0U;
      float minimum_clearance_m = 0.0F;
      PickEntryRoot terminal_root{};
  };

  struct FunnelGeometryBatch {
      FrozenFunnelInputIdentity identity{};
      Transform activation_root{};
      InteractionTarget frozen_target{};
      GraspAffordance frozen_affordance{};
      PickSlotConfig geometry_config{};
      std::vector<PickNavigationObstacle> frozen_obstacles{};
      std::array<FunnelGeometryCertification, 32> proposals{};
      std::vector<uint32_t> survivor_indices{};
  };

  struct FunnelGeometryInput {
      Transform live_root{};
      const InteractionTarget* target = nullptr;
      const GraspAffordance* affordance = nullptr;
      const LearnedFunnelArtifact* artifact = nullptr;
      PickSlotConfig geometry_config{};
      std::vector<PickNavigationObstacle> obstacles{};
      uint64_t snapshot_fingerprint = 0U;
  };

  struct FunnelPreviewObservation {
      Transform live_root{};
      uint64_t snapshot_fingerprint = 0U;
      uint64_t preview_snapshot_fingerprint = 0U;
      FrozenFunnelInputIdentity live_identity{};
      InteractionTarget live_target{};
      GraspAffordance live_affordance{};
      std::vector<PickNavigationObstacle> live_obstacles{};
      std::vector<PickAssistPreviewResult> results{};
  };

  struct FunnelSelection {
      FunnelGeometryBatch geometry{};
      std::array<std::optional<PickEntryPreview>, 32> previews{};
      std::vector<uint32_t> certified_indices{};
      std::optional<uint32_t> selected_index{};
      uint64_t preview_snapshot_fingerprint = 0U;
  };

  FunnelGeometryBatch certify_funnel_geometry(
      const FunnelGeometryInput& input);
  FunnelGeometryBatch recertify_funnel_geometry(
      const FunnelGeometryBatch& frozen_batch,
      Transform live_root,
      const InteractionTarget& live_target,
      const GraspAffordance& live_affordance,
      const std::vector<PickNavigationObstacle>& live_obstacles);
  std::vector<PickAssistPreviewRequest> make_funnel_preview_requests(
      const FunnelGeometryBatch& batch);
  FunnelSelection rank_funnel_preview_batch(
      const FunnelGeometryBatch& batch,
      const FunnelPreviewObservation& observation);
  std::array<uint8_t, 32> pick_slot_config_digest(
      const PickSlotConfig& config);
  std::array<uint32_t, 5> pick_slot_config_f32_bits(
      const PickSlotConfig& config);
  ```

  `pick_slot_config_digest` validates all five frozen `PickSlotConfig` fields:
  `maximum_direct_travel_m` must be finite and strictly positive; travel
  tolerance, table expansion, obstacle radius, and obstacle safety margin must
  be finite and nonnegative; the float32 additions maximum-plus-tolerance and
  radius-plus-margin must remain finite. Its result is exactly

  ```text
  SHA256(
    "g1-funnel-pick-slot-config-v1\0" ||
    little_endian_raw_f32(maximum_direct_travel_m) ||
    little_endian_raw_f32(travel_tolerance_m) ||
    little_endian_raw_f32(table_root_expansion_m) ||
    little_endian_raw_f32(obstacle_root_radius_m) ||
    little_endian_raw_f32(obstacle_safety_margin_m))
  ```

  The hash uses the exact five raw float32 values, in the declaration order
  above, with no decimal reparse or float64 conversion.
  `pick_slot_config_f32_bits` returns those same five host `uint32_t` bit
  patterns in that order. Certification stores both arrays in the frozen
  identity. Bitwise equality of all five resolved values remains authoritative;
  the digest is an identity/audit field and never substitutes for that
  comparison.

  `activation_snapshot_fingerprint` is the existing `uint64_t` returned by
  `runtime_detail::locomotion_snapshot_fingerprint`. Define both other
  fingerprints as the little-endian uint64 in the first eight bytes of SHA-256.
  `obstacle_fingerprint` hashes domain
  `"g1-funnel-obstacles-v1\0"`, obstacle-count u32, then each obstacle's center
  XYZ and size XYZ raw finite float32 bits in vector order. `scene_fingerprint`
  hashes domain `"g1-funnel-scene-v1\0"`, target ID u64, generation u32,
  affordance ID u32, object transform position XYZ/quaternion WXYZ, object-bounds
  center/half-extents, object dimensions, table transform, table size, hand u32,
  hand-in-object transform, approach XYZ, and clearance radius, all in that
  order as canonical little-endian float32. Tests mutate every field. Hashes are
  diagnostics/indexes only: authority still uses
  `same_interaction_target_snapshot` plus exact ordered obstacle comparison, so
  hash equality never substitutes for full frozen-input equality.

  Map object-local X/Z with object right/forward and add object yaw to sample
  yaw. For KNN/diffusion, connector is live root→sample 0 and sweep samples
  `i→i+1` for `i=0..14`. For endpoint, the sole execution sample is terminal and
  the complete route is live root→terminal with zero funnel segments. Include
  connector in length/clearance. Reject on the first malformed/colliding segment.
  Route construction uses the exact clearance fold above; neither segment order
  nor obstacle order may be changed by a parallel reduction.

  `certify_funnel_geometry` validates `input.geometry_config`, records its
  raw bits and digest in the identity, and owns full value copies of that resolved config,
  the target, selected affordance, and ordered obstacles before returning; their
  hashes are audit fields only. `recertify_funnel_geometry` accepts no config
  override: it recomputes the raw bits and digest of
  `frozen_batch.geometry_config`, requires both to equal the frozen identity,
  and uses that owned value for every connector and segment.
  `make_funnel_preview_requests` emits one request per geometry survivor in
  stable proposal-index order with `slot_id=proposal_index` and exact terminal
  root. `rank_funnel_preview_batch` requires
  `snapshot_fingerprint == preview_snapshot_fingerprint`, exact live frozen-
  input identity, `same_interaction_target_snapshot`,
  `same_authored_grasp_affordance`, exact ordered obstacle count/float bits,
  exactly one result for every request, no other result, and bit-equal request
  IDs/roots. It first calls `recertify_funnel_geometry` with the observation's
  full live values and the batch-owned config, replacing
  connector/route/heading/clearance and survivor
  fields in `selection.geometry`; preview gates and ranking consume only that
  returned copy, never stale `batch.proposals`. Any equality or atomic-batch
  defect fails the complete selection; it never ranks a partial batch.

- [ ] **Step 5: Implement exact ranking**

  ```cpp
  auto key = [](const FunnelGeometryCertification& row,
                const PickEntryPreview& preview) {
      const uint64_t clearance_key =
          UINT64_MAX - 1U -
          quantize_clearance_mm(row.minimum_clearance_m);
      return std::tuple{
          row.route_millimetres,
          row.heading_milliradians,
          clearance_key,
          quantize_cost(preview.total_cost),
          row.proposal_index,
      };
  };
  std::stable_sort(certified.begin(), certified.end(),
      [&](size_t a, size_t b) {
          return key(selection.geometry.proposals[a], *selection.previews[a]) <
                 key(selection.geometry.proposals[b], *selection.previews[b]);
      });
  ```

  Accumulate connector distance, funnel-segment distance, and absolute wrapped
  yaw change in float64 in connector-then-segment order. Quantize connector and
  funnel separately for their diagnostic fields; quantize route from the
  unrounded float64 sum, never by adding rounded fields. Quantize clearance from
  the stored float32 `minimum_clearance_m` promoted once to float64; do not
  recompute it from obstacle geometry during ranking. `quantize_route_mm(x)`,
  `quantize_heading_milliradians(x)`, and
  `quantize_clearance_mm(x)` require finite nonnegative input and return
  `floor(x*1000.0 + 0.5)` as uint64. `quantize_cost(x)` requires finite
  nonnegative input and returns `floor(x*1000000.0 + 0.5)`. Reject a value whose
  rounded result exceeds `UINT64_MAX-1`; this bound is checked before the
  subtraction, so the rank-key subtraction cannot underflow. `UINT64_MAX` is
  reserved as the invalid or unavailable sentinel and is never ranked.
  Certified route/heading/clearance must be finite; there is no infinity-as-
  clearance shortcut. The clearance sort uses
  `UINT64_MAX - 1 - clearance_mm`, so larger clearance wins without signed
  negation: zero clearance maps to `UINT64_MAX-1`, the maximum accepted
  clearance maps to zero, and no valid input maps to `UINT64_MAX`. Tests pin
  those three facts, strict monotonic ordering, raw-versus-inflated AABB
  distance, exclusion of the table, connector and funnel-segment aggregation,
  obstacle vector order, the exact
  obstacle-free cap, cap saturation for distant obstacles, half-unit rounding,
  zero, maximum accepted value, negative/NaN/infinity/overflow rejection, and
  release-fast-math parity.

  Quantize all keys before comparison. A
  preview is certified only when available, finite, fingerprint-equal,
  `path_feasible`, `match_ready`, and its prospective root exactly equals the
  request. Preview rejection remains attached to that proposal index.

- [ ] **Step 6: Run GREEN, current-slot regression, and fast-math parity**

  ```bash
  make build/tests/test_interaction_funnel_selection \
       build/tests/test_interaction_pick_slots \
       build/tests/test_interaction_pick_assist
  build/tests/test_interaction_funnel_selection
  build/tests/test_interaction_pick_slots
  build/tests/test_interaction_pick_assist
  make build/tests/test_interaction_funnel_selection_release_fast_math
  build/tests/test_interaction_funnel_selection_release_fast_math
  ```

- [ ] **Step 7: Review, commit, and push**

  ```bash
  git add interaction_funnel_selection.h interaction_funnel_selection.cpp \
    interaction_pick_slots.h interaction_pick_slots.cpp \
    tests/cpp/test_interaction_funnel_selection.cpp Makefile
  git diff --cached --check
  git commit -m "feat: certify complete learned pickup funnels"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 6: Follow one frozen funnel through ordinary 25-Hz controls

**Files:**

- Create: `interaction_funnel_follower.h`
- Create: `interaction_funnel_follower.cpp`
- Create: `interaction_learned_pickup_diagnostics.h`
- Create: `interaction_learned_pickup_backend.h`
- Create: `interaction_learned_pickup_backend.cpp`
- Create: `tests/cpp/test_interaction_funnel_follower.cpp`
- Create: `tests/cpp/test_interaction_learned_pickup_backend.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_controller.cpp`
- Modify: `tests/cpp/test_live_flat_pick_entry_oracle.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `interaction_pick_assist.h`
- Modify: `interaction_smart_pickup_controller.h`
- Modify: `interaction_smart_pickup_controller.cpp`
- Modify: `controller.cpp`
- Modify: `Makefile`

**Interfaces:**

- Produces: `TimedFunnelFollower` and
  `LearnedSmartPickupAssistBackend final : SmartPickupAssistBackend`.
- Preserves: external preview callback, `PickRequest`, runtime preflight,
  matcher authority, and explicit authored backend mode.

- [ ] **Step 1: Write failing exact-schedule tests**

  ```cpp
  TimedFunnelFollower follower(config);
  const PickSlotConfig geometry = frozen_test_geometry();
  constexpr uint64_t first_controller_tick = 41U;
  TEST_CHECK(follower.begin(
      route, frozen_identity, frozen_target, frozen_obstacles,
      geometry, first_controller_tick));
  for (uint32_t sample = 0U; sample < 16U; ++sample) {
      const FunnelFollowOutput out = follower.observe(
          observation_at(first_controller_tick + sample));
      TEST_CHECK(out.published_sample_index.has_value());
      TEST_CHECK(*out.published_sample_index == sample);
      TEST_CHECK(out.override_steering);
  }
  TEST_CHECK(follower.diagnostics().state == FunnelFollowState::Following);
  const FunnelFollowOutput final_check = follower.observe(
      observation_at(first_controller_tick + 16U));
  TEST_CHECK(!final_check.published_sample_index.has_value());
  TEST_CHECK(!final_check.override_steering);
  TEST_CHECK(follower.diagnostics().state == FunnelFollowState::Complete);
  TEST_CHECK(follower.diagnostics().published_count == 16U);
  TEST_CHECK(follower.diagnostics().duplicate_count == 0U);
  TEST_CHECK(follower.diagnostics().skipped_count == 0U);
  ```

  Here `frozen_identity` is the identity returned by certifying the route with
  that exact `geometry`; a zero-filled or independently fabricated config
  identity is a failing test, not an accepted default.

  Add failure tests for wrong observation tick, `>0.18 m` tracking error,
  `>25°` yaw error, newly blocked current-to-target connector, newly blocked
  remaining funnel segment, target handle/generation, affordance, condition,
  artifact, geometry config, scene, or obstacle identity mutation, cancellation, and
  attempted proposal switch. Every failure asserts its distinct reason.

- [ ] **Step 2: Write failing backend lifecycle tests**

  Require geometry requests followed one observation later by one atomic preview
  batch, frozen winner, connector arrival/deadline behavior, 16 follower ticks
  for KNN/diffusion, no follower ticks for endpoint, an asynchronous final live
  preview, one submission, no authored slot read, clean cancellation, and
  fail-closed all-blocked behavior.

- [ ] **Step 3: Run RED**

  ```bash
  make build/tests/test_interaction_funnel_follower \
       build/tests/test_interaction_learned_pickup_backend \
       build/tests/test_interaction_smart_pickup_controller \
       build/tests/test_live_flat_pick_entry_oracle \
       build/tests/test_interaction_controller_adapter
  ```

  Expected: new follower/backend imports fail and the three migration tests fail
  their tick/ordered-obstacle/source-contract assertions before implementation.

- [ ] **Step 4: Implement the follower without root writes**

  Put `FunnelFollowState`, `FunnelFollowReason`,
  `FunnelFollowerDiagnostics`, `LearnedPickupState`,
  `LearnedPickupFailureReason`, and `LearnedFunnelDiagnostics` in
  `interaction_learned_pickup_diagnostics.h`. That header may include
  `interaction_funnel_artifact.h` and standard-library value types, but never
  `interaction_pick_assist.h`, the follower, or the backend. The follower,
  backend, and `interaction_pick_assist.h` include this shared header;
  `PickAssistDiagnostics` can therefore own
  `std::optional<LearnedFunnelDiagnostics>` with a complete type and no include
  cycle. The authored backend always leaves it empty.

  Extend the ordinary observation boundary exactly:

  ```cpp
  struct PickAssistStart {
      // existing fields remain
      uint64_t controller_tick = 0U;
  };

  struct PickAssistObservation {
      // existing fields remain
      uint64_t controller_tick = 0U;
      std::vector<PickNavigationObstacle> live_obstacles{};
  };

  struct SmartPickupPostStepInput {
      // existing fields remain
      uint64_t controller_tick = 0U;
      std::vector<PickNavigationObstacle> live_obstacles{};
      // remove obstacle_centers/obstacle_sizes after all callers migrate
  };
  ```

  `controller.cpp` constructs the vector in scene order every native tick.
  `SmartPickupController::post_step` copies the activation tick into
  `PickAssistStart`, and copies every later tick and exact ordered vector into
  every active `PickAssistObservation`, including preview-pending, connector,
  follow, and final-preview observations. It never reuses activation obstacles.
  Tests mutate tick, count, order, center bits, and size bits at this controller
  boundary.
  Update every existing helper/call site in
  `tests/cpp/test_interaction_smart_pickup_controller.cpp` and
  `tests/cpp/test_live_flat_pick_entry_oracle.cpp`, plus the controller source
  contract in `tests/cpp/test_interaction_controller_adapter.cpp`, in the RED
  wave; no compatibility `obstacle_centers`/`obstacle_sizes` fields remain after
  GREEN.

  ```cpp
  struct FunnelFollowerConfig {
      float step_seconds = 0.04F;
      float maximum_tracking_error_m = 0.18F;
      float maximum_yaw_error_radians = 0.436332313F;
  };

  enum class FunnelFollowReason : uint8_t {
      None,
      ObservationTickMismatch,
      TrackingPositionExceeded,
      TrackingYawExceeded,
      TargetChanged,
      AffordanceChanged,
      ConditionChanged,
      ArtifactChanged,
      GeometryConfigChanged,
      SceneChanged,
      ObstaclesChanged,
      CurrentConnectorBlocked,
      RemainingFunnelBlocked,
      Cancelled,
  };

  enum class FunnelFollowState : uint8_t {
      Idle,
      Following,
      Complete,
      Failed,
  };

  struct FunnelFollowerDiagnostics {
      FunnelFollowState state = FunnelFollowState::Idle;
      FunnelFollowReason reason = FunnelFollowReason::None;
      uint32_t published_count = 0U;
      uint32_t duplicate_count = 0U;
      uint32_t skipped_count = 0U;
      uint32_t observation_count = 0U;
      uint32_t expected_sample_index = 0U;
      bool has_last_published_sample = false;
      uint32_t last_published_sample_index = 0U;
      uint64_t first_controller_tick = 0U;
      uint64_t last_controller_tick = 0U;
      float maximum_tracking_error_m = 0.0F;
      float maximum_yaw_error_radians = 0.0F;
      int32_t blocked_segment_index = -1;
      int32_t obstacle_index = -1;
  };

  struct FunnelFollowObservation {
      uint64_t controller_tick = 0U;
      uint64_t snapshot_fingerprint = 0U;
      Transform displayed_root{};
      float camera_azimuth = 0.0F;
      const InteractionTarget* live_target = nullptr;
      uint32_t affordance_id = 0U;
      std::array<uint8_t, 32> condition_digest{};
      std::array<uint8_t, 32> artifact_digest{};
      std::array<uint32_t, 5> geometry_config_f32_bits{};
      std::array<uint8_t, 32> geometry_config_digest{};
      uint64_t scene_fingerprint = 0U;
      uint64_t obstacle_fingerprint = 0U;
      std::vector<PickNavigationObstacle> obstacles{};
  };

  struct RemainingRouteCertification {
      bool certified = false;
      bool current_connector_blocked = false;
      int32_t blocked_segment_index = -1;
      int32_t obstacle_index = -1;
      float minimum_clearance_m = 0.0F;
  };

  RemainingRouteCertification certify_remaining_route(
      Transform live_root,
      const std::array<Transform, 16>& route,
      uint32_t next_sample_index,
      const InteractionTarget& frozen_target,
      const std::vector<PickNavigationObstacle>& live_obstacles,
      const PickSlotConfig& geometry);

  struct FunnelFollowOutput {
      bool override_steering = false;
      vec3 left_stick{};
      vec3 right_stick{};
      std::optional<uint32_t> published_sample_index{};
      FunnelFollowReason reason = FunnelFollowReason::None;
  };

  class TimedFunnelFollower {
  public:
      explicit TimedFunnelFollower(FunnelFollowerConfig config = {});
      bool begin(
          const std::array<Transform, 16>& route,
          const FrozenFunnelInputIdentity& identity,
          const InteractionTarget& frozen_target,
          const std::vector<PickNavigationObstacle>& frozen_obstacles,
          const PickSlotConfig& frozen_geometry,
          uint64_t first_controller_tick);
      FunnelFollowOutput observe(
          const FunnelFollowObservation& observation);
      void cancel();
      const FunnelFollowerDiagnostics& diagnostics() const;
  private:
      PickSlotConfig geometry_config_{};
      std::array<uint32_t, 5> geometry_config_f32_bits_{};
      std::array<uint8_t, 32> geometry_config_digest_{};
      bool geometry_config_bound_ = false;
  };

  FunnelFollowOutput TimedFunnelFollower::observe(
      const FunnelFollowObservation& observation) {
      if (observation.controller_tick != expected_controller_tick_) {
          if (observation.controller_tick < expected_controller_tick_) {
              ++diagnostics_.duplicate_count;
          } else {
              diagnostics_.skipped_count += static_cast<uint32_t>(
                  observation.controller_tick - expected_controller_tick_);
          }
          return fail(FunnelFollowReason::ObservationTickMismatch);
      }
      ++diagnostics_.observation_count;
      if (!same_frozen_authority(observation)) {
          return fail(exact_authority_reason(observation));
      }
      if (diagnostics_.has_last_published_sample) {
          const Transform previous =
              route_[diagnostics_.last_published_sample_index];
          if (!record_and_check_tracking_error(
                  observation.displayed_root, previous)) {
              return fail(exact_tracking_reason());
          }
      }
      if (expected_sample_index_ == 16U) {
          state_ = FunnelFollowState::Complete;
          ++expected_controller_tick_;
          return {};
      }
      const RemainingRouteCertification remaining =
          certify_remaining_route(
              observation.displayed_root,
              route_, expected_sample_index_, frozen_target_,
              observation.obstacles, geometry_config_);
      if (!remaining.certified) {
          return fail(remaining.current_connector_blocked
              ? FunnelFollowReason::CurrentConnectorBlocked
              : FunnelFollowReason::RemainingFunnelBlocked);
      }
      FunnelFollowOutput output{};
      output.override_steering = true;
      const Transform target = route_[expected_sample_index_];
      output.left_stick = arrival_navigation_stick(
          target.position, observation.displayed_root.position,
          observation.camera_azimuth);
      output.right_stick = arrival_facing_stick(
          quat_mul_vec3(
              target.rotation, vec3(0.0F, 0.0F, 1.0F)),
          observation.camera_azimuth);
      output.published_sample_index = expected_sample_index_++;
      diagnostics_.has_last_published_sample = true;
      diagnostics_.last_published_sample_index =
          *output.published_sample_index;
      ++diagnostics_.published_count;
      ++expected_controller_tick_;
      return output;
  }
  ```

  `same_frozen_authority` first compares target handle/generation and all target
  fields except the affordance vector (`TargetChanged`), then the selected
  affordance under `same_authored_grasp_affordance` (`AffordanceChanged`), then
  affordance-vector count/order and every nonselected affordance
  (`TargetChanged`), followed by condition/artifact bytes, exact config raw
  bits/digest, scene fingerprint, obstacle fingerprint, and exact ordered
  obstacle float bits, returning the
  corresponding first enum reason in that listed order. This preserves the
  distinct selected-affordance leaf reason even though
  `same_interaction_target_snapshot` also covers affordances. The live locomotion snapshot
  fingerprint is recorded for audit and may change as the root moves.
  `begin` validates `frozen_geometry`, requires its recomputed raw bits and
  digest to equal `identity.geometry_config_f32_bits` and
  `identity.geometry_config_digest`, and stores all three as owning values
  before it can enter `Following`; it never retains the caller's reference.
  Every `observe` recomputes the stored value's raw bits and digest before route
  certification, and
  `geometry_config_` in the call above is this bound member. A separately
  default-constructed config whose bits differ, post-begin mutation, digest-only
  equality, or config switch fails with `GeometryConfigChanged` before
  publishing a sample.
  The learned backend fills both observation config arrays from its immutable
  constructor-owned config on every follower tick; callers cannot omit them or
  substitute a digest-only value.
  Re-certification sweeps actual
  displayed root→current scheduled sample plus every remaining planned segment
  against the live obstacle values whose canonical fingerprint must still equal
  the frozen obstacle fingerprint. The output contains sticks/diagnostics only;
  no mutable root reference exists. Samples `0..15` publish once on exactly 16
  consecutive observations beginning at `first_controller_tick`. Observation
  `first_controller_tick+16` is mandatory: it checks tracking against sample 15
  and then completes without publishing. Thus final preview cannot start until
  the root has been observed after the final target. A late tick increments
  `skipped_count` by the exact unsigned delta; an early/repeated tick increments
  `duplicate_count` once; either fails immediately. Tests assert every counter,
  last-sample flag/index, maximum observed errors, and first/last tick.

- [ ] **Step 5: Implement the learned backend state machine**

  ```cpp
  enum class LearnedPickupState : uint8_t {
      Idle = 0U,
      GeometryCertification = 1U,
      SelectionPreviewPending = 2U,
      Connector = 3U,
      FunnelFollow = 4U,
      FinalPreviewPending = 5U,
      ReadyToSubmit = 6U,
      Submitted = 7U,
      Failed = 8U,
  };

  constexpr size_t kLearnedPickupStateCount = 9U;

  struct LearnedPickupStateLifecycleDiagnostics {
      bool entered = false;
      uint64_t entry_controller_tick = 0U;
      uint32_t observation_count = 0U;
      uint32_t preview_request_count = 0U;
      uint32_t preview_result_count = 0U;
  };

  enum class LearnedPickupFailureReason : uint8_t {
      None,
      ArtifactMissing,
      ArtifactMalformed,
      ConditionMismatch,
      FrozenInputChanged,
      NoGeometrySurvivor,
      PreviewBatchInvalid,
      NoPreviewSurvivor,
      ConnectorDeadlineExceeded,
      FollowerFailed,
      FinalPreviewInvalid,
      Cancelled,
  };

  class FunnelArtifactResolver {
  public:
      virtual ~FunnelArtifactResolver() = default;
      virtual const LearnedFunnelArtifact* resolve(
          const std::array<uint8_t, 32>& condition_digest) const = 0;
  };

  class LearnedSmartPickupAssistBackend final
      : public SmartPickupAssistBackend {
  public:
      LearnedSmartPickupAssistBackend(
          FunnelArtifactResolver& resolver,
          PickSlotConfig geometry,
          FunnelFollowerConfig follower);
      bool begin(
          const PickAssistStart& start,
          const InteractionTarget* post_step_target) override;
      void cancel() override;
      PickAssistOutput observe(
          const PickAssistObservation& observation) override;
      std::optional<PickRequest> take_submission(
          uint64_t request_id) override;
      bool active() const override;
      bool owns_manual_interact() const override;
      const PickAssistDiagnostics& diagnostics() const override;
  private:
      const PickSlotConfig geometry_;
      const std::array<uint32_t, 5> geometry_config_f32_bits_;
      const std::array<uint8_t, 32> geometry_config_digest_;
      // Resolver, follower, frozen attempt state, and diagnostics follow.
  };

  struct LearnedFunnelDiagnostics {
      FunnelProviderKind provider_kind = FunnelProviderKind::Diffusion;
      std::array<uint8_t, 32> artifact_digest{};
      std::array<uint8_t, 32> checkpoint_digest{};
      std::array<uint8_t, 32> provider_config_digest{};
      std::array<uint8_t, 32> dataset_digest{};
      std::array<uint8_t, 32> manifest_digest{};
      std::array<uint8_t, 32> condition_digest{};
      std::array<uint8_t, 32> geometry_config_digest{};
      TargetHandle target{};
      uint32_t affordance_id = 0U;
      uint64_t scene_fingerprint = 0U;
      uint64_t obstacle_fingerprint = 0U;
      uint64_t activation_snapshot_fingerprint = 0U;
      uint64_t selection_preview_snapshot_fingerprint = 0U;
      uint64_t final_preview_snapshot_fingerprint = 0U;
      std::optional<uint32_t> selected_proposal{};
      std::optional<uint32_t> selected_source_row_index{};
      uint32_t selected_sample_count = 0U;
      uint32_t proposals_considered = 0U;
      uint32_t geometry_survivors = 0U;
      uint32_t preview_survivors = 0U;
      uint32_t malformed_rejections = 0U;
      uint32_t connector_rejections = 0U;
      uint32_t table_rejections = 0U;
      uint32_t obstacle_rejections = 0U;
      uint32_t preview_batch_rejections = 0U;
      uint32_t preview_rejections = 0U;
      float connector_distance_m = 0.0F;
      float funnel_distance_m = 0.0F;
      float total_distance_m = 0.0F;
      float minimum_clearance_m = 0.0F;
      uint64_t route_millimetres = 0U;
      uint64_t heading_milliradians = 0U;
      float preview_total_cost = 0.0F;
      uint32_t connector_ticks = 0U;
      uint32_t connector_deadline_ticks = 250U;
      uint32_t connector_arrival_ticks = 0U;
      uint32_t final_preview_ticks = 0U;
      LearnedPickupState state = LearnedPickupState::Idle;
      std::array<LearnedPickupStateLifecycleDiagnostics,
                 kLearnedPickupStateCount> state_lifecycle{};
      LearnedPickupFailureReason backend_reason =
          LearnedPickupFailureReason::None;
      FunnelFollowReason follower_reason = FunnelFollowReason::None;
      FunnelFollowerDiagnostics follower{};
  };
  ```

  `state_lifecycle` is indexed exactly as follows: `0=Idle`,
  `1=GeometryCertification`, `2=SelectionPreviewPending`, `3=Connector`,
  `4=FunnelFollow`, `5=FinalPreviewPending`, `6=ReadyToSubmit`, `7=Submitted`,
  and `8=Failed`. Add static assertions for every enum value and the array size;
  no cast of an unknown enum is accepted.

  On each `begin`, zero the array, record `Idle` entered at
  `PickAssistStart::controller_tick`, then record `GeometryCertification`
  entered at that same tick. A transition records its destination entry exactly
  once at the controller tick that caused the transition; multiple states may
  therefore have the same entry tick. At the start of `observe`, increment only
  the state active on entry, and add the complete input
  `preview_results.size()` to that state's preview-result count before
  validation. After producing the output, add the complete
  `preview_requests.size()` to that same state's preview-request count. A state
  entered at the end of an observation has zero observations until a later call
  processes it. `take_submission` records `Submitted` at the most recent
  observation tick. `cancel()` records `Failed` at the most recent authoritative
  start/observation tick because its inherited interface carries no tick; it
  never fabricates a future tick. All count additions are checked for uint32
  overflow and overflow fails the attempt. Unvisited states retain
  `entered=false`, tick zero, and zero counts.

  Tests use a fixed resolver containing one artifact. `begin` freezes
  target/generation/condition/root/obstacles and the backend's geometry config,
  resolves the exact condition
  digest once, and validates the returned artifact. A missing key is
  `ArtifactMissing`; it never asks for a nearby condition or authored slot.
  `GeometryCertification` emits terminal roots for every survivor in
  one `PickAssistOutput::preview_requests` batch. `SelectionPreviewPending`
  consumes only the next exact atomic batch/fingerprint and freezes one winner,
  including that winner's canonical float32 `minimum_clearance_m`; later states
  publish that stored value and never recompute it for ranking or diagnostics.

  Ordinary connector arrival requires position error `<=0.10 m`, yaw error
  `<=0.174532925 rad`, and displayed speed `<=0.10 m/s` for three consecutive
  observations. It fails with `ConnectorDeadlineExceeded` after 250 connector
  observations and records the actual tick count. KNN/diffusion connect to
  execution sample 0, then start the 16-tick follower. Endpoint connects directly
  to its sole terminal, records `selected_sample_count=1` and follower
  `published_count=0`, and never fabricates a follow phase. Both paths emit one
  final preview request and consume its result on the next observation before
  returning the unchanged `PickRequest` through `take_submission`.

  The backend constructor takes `PickSlotConfig` by value, validates it once,
  and stores one immutable owning copy `const PickSlotConfig geometry_` plus its
  raw-bit array and digest for the backend lifetime; no setter or borrowed config reference
  exists. Initial `FunnelGeometryInput.geometry_config`, every batch
  recertification, and `TimedFunnelFollower::begin` receive values derived from
  that same copy. Follower startup specifically receives
  `selection.geometry.geometry_config` and requires its bitwise value and
  digest to equal `geometry_`; the backend never reconstitutes defaults. Tests use one
  nondefault field at a time and prove initial certification, selection
  recertification, connector recertification, follower startup, and remaining-
  route checks all observe the exact same five bits and reject any mismatch.

  On every Connector observation, before arrival/navigation, compare the same
  full live authority used by the follower and call
  `recertify_funnel_geometry` for the frozen winner from current displayed root
  through its connector and every still-unconsumed segment (endpoint has only
  current-root→terminal). Any changed identity is `FrozenInputChanged`; any
  blocked/malformed connector or remaining segment is `NoGeometrySurvivor`,
  with the leaf geometry reason/counters preserved. Increment `connector_ticks`
  at the start of each valid Connector observation. Observations 1 through 250
  are allowed. If the third consecutive arrival sample occurs on observation
  250, transition normally; otherwise observation 250 fails immediately and
  emits no navigation override, and observation 251 is never accepted. Reset
  `connector_arrival_ticks` to zero on any valid observation outside any one of
  the three arrival thresholds.

  After KNN/diffusion connector arrival, begin the follower at the next
  controller tick. After its 16 publications, consume the mandatory next-tick
  final-sample tracking observation; only then emit the final preview request.
  Endpoint emits final preview after its third connector-arrival observation.
  Selection preview and final preview each consume exactly the next observation
  with a complete matching batch and never accept same-observation callback
  data. Every state records entry tick, observation count, preview request count,
  and preview result count through its exact `state_lifecycle` slot. Tests pin
  all nine index mappings, same-tick transitions, zero-observation transient
  states, full batch counts before validation, submitted/cancel tick semantics,
  unvisited sentinels, overflow rejection, and all transition boundaries.

  Map loading/condition/selection/connector/final-preview/cancel failures to the
  exact `LearnedPickupFailureReason`. Any failed follower sets backend reason
  `FollowerFailed` and preserves the exact leaf cause in `follower_reason`.
  Tests exercise every backend enum value and every follower-to-backend mapping;
  HUD/evidence emit both names, never cast one enum as the other.
  Add `std::optional<LearnedFunnelDiagnostics> learned` to
  `PickAssistDiagnostics` through the shared diagnostics header; the authored
  backend leaves it empty.

- [ ] **Step 6: Run GREEN and authored-backend regression**

  ```bash
  make build/tests/test_interaction_funnel_follower \
       build/tests/test_interaction_learned_pickup_backend \
       build/tests/test_interaction_pick_assist \
       build/tests/test_interaction_smart_pickup_controller \
       build/tests/test_live_flat_pick_entry_oracle \
       build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_funnel_follower
  build/tests/test_interaction_learned_pickup_backend
  build/tests/test_interaction_pick_assist
  build/tests/test_interaction_smart_pickup_controller
  build/tests/test_live_flat_pick_entry_oracle
  build/tests/test_interaction_controller_adapter
  ```

- [ ] **Step 7: Review, commit, and push**

  ```bash
  git add interaction_funnel_follower.h interaction_funnel_follower.cpp \
    interaction_learned_pickup_diagnostics.h \
    interaction_learned_pickup_backend.h \
    interaction_learned_pickup_backend.cpp interaction_pick_assist.h \
    interaction_smart_pickup_controller.h \
    interaction_smart_pickup_controller.cpp \
    controller.cpp \
    tests/cpp/test_interaction_funnel_follower.cpp \
    tests/cpp/test_interaction_learned_pickup_backend.cpp \
    tests/cpp/test_interaction_smart_pickup_controller.cpp \
    tests/cpp/test_live_flat_pick_entry_oracle.cpp \
    tests/cpp/test_interaction_controller_adapter.cpp Makefile
  git diff --cached --check
  git commit -m "feat: follow learned pickup funnels at 25 hz"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 7: Evaluate nominal, blocked, all-blocked, and actual attachment

**Files:**

- Create: `tests/cpp/test_live_flat_learned_funnel_oracle.cpp`
- Create: `tests/python/compare_learned_funnel_oracles.py`
- Create: `tests/python/test_learned_funnel_full_pack_gate.py`
- Modify: `resources/evaluate_g1_funnel_providers.py`
- Modify: `Makefile`

**Interfaces:**

- Consumes: immutable manifest, endpoint/KNN/diffusion proposal bundles, one
  unchanged runtime/registry per declared case, actual flat controller steps.
- Produces: all-test offline metrics, 12-row runtime metrics, blocker behavior,
  exact lifecycle witnesses, normal/release parity, and one canonical
  `winning_provider` consumed by Task 8.

- [ ] **Step 1: Write failing all-row evidence assertions**

  ```python
  self.assertEqual(report["all_test_condition_count_u32"],
                   len(manifest["test_conditions"]))
  self.assertEqual(report["runtime_case_count_u32"], 12)
  self.assertEqual(report["proposal_budget_u32"], 32)
  for provider in ("endpoint_only", "knn_funnel", "diffusion"):
      row = report["providers"][provider]
      self.assertEqual(row["attach_denominator_u32"], 12)
      self.assertEqual(row["offline_denominator_u32"],
                       len(manifest["test_conditions"]))
      self.assertIn("valid_at_32_f64", row)
      self.assertIn("attach_at_32_f64", row)
      self.assertIn("endpoint_cluster_count_total_u32", row)
  ```

  Runtime-success rows require exactly 16 ordered sample observations for
  KNN/diffusion. Endpoint rows require `sample_count=1`, zero published timed
  samples, and a connector ending at the sole terminal. Both require the exact
  collapsed runtime sequence `Preflight -> Align -> PickupReplay -> Hold ->
  Carry`, one attachment edge, Carry/Held/attached, exact owner, and subsequent
  Carry displacement/grasp preservation. Invalid/duplicate/no-valid rows remain
  in denominators. The winning provider must have at least three successful
  attachment witnesses on three distinct held-out condition indices and object
  IDs; those same three rows must pass the post-Carry witness below.

- [ ] **Step 2: Run RED**

  ```bash
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_learned_funnel_full_pack_gate -v
  make build/tests/test_live_flat_learned_funnel_oracle
  ```

- [ ] **Step 3: Implement the headless runtime oracle**

  For each of the 12 `runtime_cases`, provider, and frozen scene, resolve the
  active `FunnelPackIdentity` from the required `--pack` path, load the bundle
  with that identity, and resolve the artifact only through
  `FunnelArtifactBundle::require` and the exact Task-2
  `proposal_outputs` key
  `(condition_index_u32,condition_digest,provider_kind)`. Reject a bundle path,
  seed, header namespace/index, source-row bound, condition identity, or artifact
  identity that disagrees with that mapping. Construct exact
  target/support/root/obstacles from the manifest, run one ordinary
  locomotion/controller update per 0.04 s tick, and emit a case record even when
  no proposal survives.

  ```cpp
  require(output.diagnostics.state == RuntimeState::Carry, "not Carry");
  require(output.diagnostics.attached, "not attached");
  const InteractionTarget* held = registry.find(output.diagnostics.target);
  require(held != nullptr && held->state == ObjectState::Held, "not Held");
  require(held->owner_request == request_id, "wrong owner");
  require(attachment_edges == 1U, "wrong attachment edge count");
  require(collapsed_states == std::vector<RuntimeState>{
      RuntimeState::Preflight,
      RuntimeState::Align,
      RuntimeState::PickupReplay,
      RuntimeState::Hold,
      RuntimeState::Carry,
  }, "runtime lifecycle was incomplete or reordered");
  ```

  On a nominal attachment success, save character-root transform,
  object-world transform, and `object_in_character_hand` at the first Carry
  tick. Apply ordinary forward locomotion for at most 50 more native ticks. A
  post-Carry witness succeeds only when root and object each move at least
  `0.20 m`, the registry stays `Held` with the exact owner and one edge on every
  tick, and the relative grasp translation error never exceeds `0.00002 m` nor
  relative rotation geodesic error `0.00002 rad`. Record tick count, both
  displacements, maximum errors, and final root/object transforms even on
  failure. This is actual carrying motion, not a static extra Carry tick.

  In all-blocked cases assert no request, no root motion, zero ownership, and a
  responsive subsequent manual locomotion tick. "No root motion" means
  bit-identical root position/quaternion from activation through rejection;
  recovery means at least `0.05 m` root displacement within 25 ordinary manual
  locomotion ticks after learned ownership releases.

  Every oracle row with a selected proposal emits the selected backend's stored
  float32 `minimum_clearance_m` promoted once to JSON; rows without a selection
  emit JSON null. The normal and release-fast-math oracles must agree on its raw
  float32 bits and quantized millimetres before evaluator comparison.

- [ ] **Step 4: Compute frozen metrics and continuation result**

  Run the Python evaluator over oracle JSON plus offline proposal diagnostics.
  Cluster in stable proposal order and use the manifest thresholds verbatim.
  Publish nearest-training-funnel distance and exact duplicates as informational
  fields. Do not tune or regenerate anything from the result.

  `report.json` uses the Task-2 canonical JSON encoding and has exactly these
  top-level keys:

  ```text
  schema_version_u32: 1
  kind: "g1_funnel_provider_report"
  evaluation_manifest_sha256, bundle_manifest_sha256, dataset_sha256,
    pack_sha256: string
  all_test_condition_count_u32, runtime_case_count_u32,
    proposal_budget_u32: integer
  provider_order: ["endpoint_only","knn_funnel","diffusion"]
  providers: {endpoint_only:provider,knn_funnel:provider,diffusion:provider}
  best_baseline, winning_provider: string
  comparison_operands:
    {diffusion_attach_at_32_f64,best_baseline_attach_at_32_f64,
     diffusion_endpoint_cluster_count_total_u32,
     best_baseline_endpoint_cluster_count_total_u32,
     diffusion_blocked_all_applicable_and_success,
     best_baseline_blocked_all_applicable_and_success}
  continuation_passed: boolean
  graphical_cases:
    {nominal:graphical_case,blocked_alternate:graphical_case,
     all_blocked:graphical_case}
  ```

  The evaluator loads the active identity from `--pack`, requires its three raw
  values to match the dataset and evaluation manifest, and requires its
  recomputed combined value to match the evaluation manifest, bundle manifest,
  and every referenced artifact header. It then writes that combined value as
  report `pack_sha256`.
  Report publication is atomic on any mismatch; the report never derives pack
  identity solely from the proposal bundle.

  Each `provider` object has no missing/extra keys:

  ```text
  provider_kind, sample_count_u32
  offline_denominator_u32, valid_numerator_u32, valid_at_32_f64
  attach_denominator_u32, attach_numerator_u32, attach_at_32_f64
  endpoint_cluster_count_total_u32
  path_cluster_count_total_u32: integer|null
  path_cluster_proposal_count_u32
  blocked_applicable_count_u32, blocked_success_count_u32
  blocked_all_applicable_and_success: boolean
  all_blocked_success_count_u32, all_blocked_all_success: boolean
  actual_attachment_success_count_u32
  post_carry_witness_success_count_u32
  offline_rows: [offline_row]
  runtime_rows: [runtime_row]
  ```

  `offline_rows` is in condition-index order and each row has exactly
  `condition_index_u32`, `condition_digest`, `artifact_sha256`, `seed_u64`,
  `proposal_count_u32`, `valid_proposal_count_u32`, `valid_at_32`,
  `endpoint_cluster_count_u32`, `path_cluster_count_u32` (null for endpoint),
  `nearest_training_distance_squared_f64` (null only when no finite proposal),
  and `exact_duplicate_proposal_count_u32`. Equal condition digests at different
  indices remain separate rows.

  A reusable `attachment_witness` object has exactly:

  ```text
  attachment_success: boolean
  post_carry_success: boolean
  selected_proposal_u32: integer|null
  selected_source_row_index_u32: integer|null
  sample_count_u32, connector_ticks_u32
  minimum_clearance_m: number|null
  published_sample_indices_u32: [integer]
  request_count_u32, request_id_u64, attachment_edge_count_u32
  collapsed_runtime_states: [string]
  final_runtime_state, final_object_state: string
  attached: boolean
  owner_request_u64
  post_carry_tick_count_u32
  root_displacement_m, object_displacement_m: number
  maximum_grasp_translation_error_m,
    maximum_grasp_rotation_error_radians: number
  initial_root_world, final_root_world,
    initial_object_world, final_object_world:
      {position_m[3],rotation_wxyz[4]}
  failure_reason: string
  ```

  `minimum_clearance_m` is null exactly when no proposal was selected. Otherwise
  it is the selected backend diagnostic's canonical Task-5 float32 value promoted
  to a JSON number, must lie in `[0, maximum_direct_travel_m]`, and must reproduce
  both the report's selected proposal and its integer clearance ranking key.
  Oracle, evaluator, full-pack, and release-parity tests independently mutate the
  value and reject a recomputed, uncapped, infinite, or nonselected-proposal
  clearance.

  `attachment_witness.failure_reason` is never empty or null. Its complete
  allowed string domain is exactly:

  ```text
  "none"
  "artifact_missing" | "artifact_malformed" | "condition_mismatch"
  "frozen_input_changed" | "no_geometry_survivor"
  "preview_batch_invalid" | "no_preview_survivor"
  "connector_deadline_exceeded" | "follower_failed"
  "final_preview_invalid" | "cancelled"
  "runtime_rejected" | "runtime_cancelled" | "runtime_failed"
  "oracle_tick_budget_exhausted" | "attachment_lifecycle_incomplete"
  "post_carry_witness_failed"
  ```

  The eleven backend strings are the lowercase-snake-case serialization of the
  eleven non-`None` `LearnedPickupFailureReason` values, with a static exhaustive
  switch. Determine the witness sentinel by this fixed precedence: a non-None
  backend reason; runtime `Rejected`, `Cancelled`, or `Failed`; exhausted oracle
  tick budget; incomplete exact attachment lifecycle; failed post-Carry check;
  otherwise success. Exact pairings are: when `attachment_success=false`,
  `post_carry_success` must also be false and the reason is exactly one of the
  non-`none`, non-`post_carry_witness_failed` strings; when attachment is true
  and post-Carry is false, the reason is exactly
  `"post_carry_witness_failed"`; when both are true, the reason is exactly
  `"none"`. No other boolean/string combination is canonical.

  Every `runtime_row`, ordered by runtime case index, has exactly
  `runtime_case_index_u32`, `condition_index_u32`, `condition_digest`,
  `object_id`, and these scene objects:

  ```text
  nominal:
    {scene_id,geometry_survivor_count_u32,preview_survivor_count_u32,
     rank_zero_proposal_u32:null|integer,witness:attachment_witness}
  blocked_alternate:
    {scene_id,nominal_rank_zero_proposal_u32:null|integer,
     nominal_rank_zero_rejection_reason:null|string,
     nominal_rank_zero_rejection_segment_i32:null|integer,
     nominal_rank_zero_rejection_obstacle_i32:null|integer,
     applicable:boolean,selected_alternate_proposal_u32:null|integer,
     success:null|boolean,witness:null|attachment_witness}
  all_blocked:
    {scene_id,geometry_survivor_count_u32,request_count_u32,
     root_motion_m,owner_request_u64,attachment_edge_count_u32,
     manual_recovery_displacement_m,success:boolean,failure_reason:string}
  ```

  `attachment_success` covers the exact attach lifecycle; `post_carry_success`
  covers the displacement/grasp-preservation checks. `blocked_alternate.success`
  is null exactly when `applicable=false`; otherwise it is true only when both
  witness booleans are true, and its witness is present. `blocked_success_count_u32` counts
  only applicable true rows. `blocked_all_applicable_and_success` is exactly
  `blocked_applicable_count_u32==12 && blocked_success_count_u32==12`.
  `all_blocked.success` means no proposal/request/root motion/owner/edge followed
  by positive manual recovery, and its aggregate counts true rows.

  `all_blocked.failure_reason` is also never empty or null. Its complete domain
  is exactly
  `"none"|"geometry_survivor"|"request_emitted"|"root_moved"|`
  `"owner_acquired"|"attachment_edge_emitted"|`
  `"oracle_tick_budget_exhausted"|"manual_recovery_failed"`.
  Evaluate a false row in that listed non-`none` precedence: any geometry
  survivor, any request, any root-transform bit change, any nonzero owner, any
  attachment edge, failure to terminate the learned attempt within its oracle
  budget, then recovery displacement below `0.05 m` after 25 manual ticks.
  `success=true` requires reason exactly `"none"`; `success=false` requires
  exactly the first applicable non-`none` value. The evaluator, normal/release
  parity comparator, and full-pack gate reject an unknown or case-mismatched
  string, `true` with a non-`none` reason, `false` with `"none"`, post-Carry true
  without attachment, or any noncanonical attachment-witness pairing.

  Each `graphical_case` contains exactly `runtime_case_index_u32`,
  `condition_index_u32`, `condition_digest`, and `scene_id`, with all four values
  JSON null when unavailable. For the winning provider, select the lowest
  runtime index with both nominal witness booleans true, the lowest index with applicable
  blocked success, and the lowest index with all-blocked success. The full-pack
  gate requires all three records non-null; it never substitutes another scene
  or provider. It also requires at least three distinct winning-provider nominal
  rows with successful attachment and post-Carry witnesses and three distinct
  condition indices/object IDs.

  Choose the best baseline by descending tuple
  `(attach_numerator_u32, blocked_success_count_u32,
  endpoint_cluster_count_total_u32, valid_numerator_u32)` and prefer `endpoint_only`
  over `knn_funnel` on an exact tie. Set `winning_provider="diffusion"` only when
  the exact Task-2 continuation formula passes against that baseline; its blocked
  operand is the all-applicable boolean, never the success count. Otherwise set
  it to the best baseline. Task 8 may read this decision and graphical mapping
  but may not recompute or override either.

- [ ] **Step 5: Run the full evaluation and release parity**

  Add the release target and parity gate in `Makefile`; both binaries consume the
  identical frozen manifest/artifacts, and the parity script compares stable
  result fields rather than timing text:

  ```make
  .PHONY: gate-learned-funnel-headless
  gate-learned-funnel-headless: \
    build/tests/test_live_flat_learned_funnel_oracle
	@mkdir -p build/g1-funnels/evidence-v1
	@build/tests/test_live_flat_learned_funnel_oracle \
	  --pack build/smart-pickup/full-pack \
	  --manifest resources/g1_funnels/evaluation_manifest_v1.json \
	  --proposals build/g1-funnels/proposals-v1 \
	  --json > build/g1-funnels/evidence-v1/runtime.jsonl
	@build/venvs/g1-funnels/bin/python \
	  resources/evaluate_g1_funnel_providers.py \
	  --pack build/smart-pickup/full-pack \
	  --dataset build/g1-funnels/dataset-v1 \
	  --manifest resources/g1_funnels/evaluation_manifest_v1.json \
	  --proposals build/g1-funnels/proposals-v1 \
	  --runtime build/g1-funnels/evidence-v1/runtime.jsonl \
	  --output build/g1-funnels/evidence-v1/report.json
	@LEARNED_FUNNEL_REPORT="$(CURDIR)/build/g1-funnels/evidence-v1/report.json" \
	  build/venvs/g1-funnels/bin/python -m unittest \
	    tests.python.test_learned_funnel_full_pack_gate -v

  .PHONY: gate-learned-funnel-release-parity
  gate-learned-funnel-release-parity: \
    build/tests/test_live_flat_learned_funnel_oracle \
    build/tests/test_live_flat_learned_funnel_oracle_release_fast_math
	@build/venvs/g1-funnels/bin/python \
	  tests/python/compare_learned_funnel_oracles.py \
	  --normal build/tests/test_live_flat_learned_funnel_oracle \
	  --release build/tests/test_live_flat_learned_funnel_oracle_release_fast_math \
	  --pack build/smart-pickup/full-pack \
	  --manifest resources/g1_funnels/evaluation_manifest_v1.json \
	  --proposals build/g1-funnels/proposals-v1
  ```

  ```bash
  make build/tests/test_live_flat_learned_funnel_oracle
  mkdir -p build/g1-funnels/evidence-v1
  build/tests/test_live_flat_learned_funnel_oracle \
    --pack build/smart-pickup/full-pack \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --proposals build/g1-funnels/proposals-v1 \
    --json > build/g1-funnels/evidence-v1/runtime.jsonl
  build/venvs/g1-funnels/bin/python \
    resources/evaluate_g1_funnel_providers.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --proposals build/g1-funnels/proposals-v1 \
    --runtime build/g1-funnels/evidence-v1/runtime.jsonl \
    --output build/g1-funnels/evidence-v1/report.json
  LEARNED_FUNNEL_REPORT="$PWD/build/g1-funnels/evidence-v1/report.json" \
    build/venvs/g1-funnels/bin/python -m unittest \
      tests.python.test_learned_funnel_full_pack_gate -v
  make gate-learned-funnel-release-parity
  ```

  Expected: the report truthfully selects diffusion only if the frozen
  continuation rule passes. If KNN/endpoint wins, retain the interface and make
  that simpler provider the default; do not revise the test manifest.

- [ ] **Step 6: Review, commit, and push**

  ```bash
  git add tests/cpp/test_live_flat_learned_funnel_oracle.cpp \
    tests/python/compare_learned_funnel_oracles.py \
    tests/python/test_learned_funnel_full_pack_gate.py \
    resources/evaluate_g1_funnel_providers.py Makefile
  git diff --cached --check
  git commit -m "test: evaluate learned funnels through attachment"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

---

### Task 8: Expose learned keyboard mode, HUD, table/rack use, and evidence

**Files:**

- Modify: `controller.cpp`
- Modify: `interaction_debug_draw.h`
- Create: `interaction_funnel_autodemo_config.h`
- Create: `interaction_funnel_autodemo_config.cpp`
- Modify: `interaction_funnel_condition_probe.cpp`
- Modify: `resources/sample_g1_funnel_proposals.py`
- Create: `resources/verify_g1_funnel_playable_catalog.py`
- Create: `interaction_funnel_catalog.h`
- Create: `interaction_funnel_catalog.cpp`
- Create: `tests/cpp/test_interaction_funnel_catalog.cpp`
- Create: `tests/cpp/test_interaction_funnel_autodemo_config.cpp`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Create: `tests/python/test_g1_funnel_playable_catalog.py`
- Create: `tests/python/test_playable_learned_funnel_evidence.py`
- Generate/Track: `resources/g1_funnels/playable_conditions_v1.jsonl`
- Generate/Track: `resources/g1_funnels/playable_catalog_v1.json`
- Generate/Track: `resources/g1_funnels/playable_artifact_paths_v1.txt`
- Generate/Track:
  `resources/g1_funnels/playable_v1/<external-index-8hex>-<condition-digest>.g1funl`
- Modify: `Makefile`

**Interfaces:**

- Consumes: Plan A's certified table/rack catalog and Task-7
  `winning_provider`, selected only through the condition probe and an exact
  condition-digest artifact catalog.
- Produces: WASD/F/X/R learned interaction mode, artifact/model/proposal HUD,
  nominal/blocked/all-blocked/table-rack visual evidence at 25 fps, final gate.

- [ ] **Step 1: Write failing catalog, explicit-mode, and HUD tests**

  Require artifact-backed mode only when `MM_PICKUP_PROVIDER=artifact` and
  `MM_FUNNEL_ARTIFACT_CATALOG` is present/valid. The catalog has one entry per
  unique full condition digest and role aliases covering every table row and
  every Plan A certified row/rack-tier pair. Two roles with byte-identical
  canonical conditions may legitimately share one entry; two unique entries
  may not share a digest. Missing/duplicate role, unequal condition bytes under
  one digest, wrong provider, path escape, stale artifact hash, or
  artifact-condition mismatch rejects the entire catalog.
  `test_g1_funnel_playable_catalog.py` drives the sampler with synthetic role
  rows including two identical conditions and asserts winning-provider
  enforcement, alias coalescing, exact role/unique-condition ordering,
  byte-identical repeat output, missing-role failure, digest-collision failure,
  and no partial catalog publication. C++ and Python tests independently mutate
  each raw pack digest, combined pack digest, active-pack value, catalog value,
  and artifact header; a combined digest copied from another pack never repairs
  a raw-identity mismatch. Require:

  ```cpp
  struct FunnelCatalogRole {
      std::string scene_role;  // "table" or "rack_tier"
      uint32_t plan_a_scenario_row = 0U;
      uint32_t rack_tier_id = 0xffffffffU;
  };

  struct FunnelArtifactCatalogEntry {
      uint32_t external_artifact_index = 0U;
      std::array<uint8_t, 32> condition_digest{};
      std::vector<FunnelCatalogRole> roles{};
      FunnelProviderKind provider_kind{};
      std::filesystem::path artifact_path{};
      std::array<uint8_t, 32> artifact_digest{};
  };

  class FunnelArtifactCatalog final : public FunnelArtifactResolver {
  public:
      static FunnelArtifactCatalog load(
          const std::filesystem::path& catalog_path,
          const FunnelPackIdentity& active_pack);
      const LearnedFunnelArtifact* resolve(
          const std::array<uint8_t, 32>& condition_digest) const override;
      const FunnelArtifactCatalogEntry& require_role(
          const FunnelCatalogRole& role) const;
  };
  ```

  On every F edge, recompute the selected target/affordance condition with
  `make_grasp_condition`, digest it, and call `resolve`; never cache the prior
  table artifact across a rack pickup. Missing/stale/malformed resolution shows
  a visible reason, consumes no request, and disables that F attempt. Artifact
  mode never consults authored interaction slots or falls back to authored mode.
  Require sample-count-aware HUD:

  ```text
  provider=<endpoint_only|knn_funnel|diffusion>
  artifact=<sha256> checkpoint=<sha256> config=<sha256> dataset=<sha256>
  condition=<sha256> proposal=<index>
  sample_count=<1|16> connector=<metres> funnel=<metres> clearance=<metres|null>
  follow=<published>/<0 for endpoint,16 otherwise> connector_tick=<n>/250
  rejected=malformed:<n>,connector:<n>,table:<n>,obstacle:<n>,preview_batch:<n>,preview:<n>
  match=<clip>:<frame> target=<id>:<generation> owner=<request>
  state=<state> attached=<bool> route=<row>:<leg>
  backend_reason=<reason> follower_reason=<reason>
  ```

- [ ] **Step 2: Probe every Plan A condition and generate the winning-provider catalog**

  Extend the Task-4 probe, not Python-authored scene constants. It requires
  `--pack build/smart-pickup/full-pack`, loads that exact pack, enumerates every
  certified Plan A scenario row from the same catalog used by the graphical
  runtime, and emits canonical JSONL in
  `(plan_a_scenario_row_u32, role_order table=0/rack_tier=1,
  rack_tier_id_u32)` order. Every line has no missing/extra keys:

  ```text
  schema_version_u32: 1
  external_role_index_u32: contiguous zero-based line index
  scene_role: "table" | "rack_tier"
  plan_a_scenario_row_u32
  rack_tier_id_u32: 0xffffffff for table, certified tier otherwise
  condition_digest
  condition: exact Task-2 condition object
  pack_database_sha256, pack_features_sha256, pack_manifest_sha256,
    pack_sha256: lowercase 64-hex string
  ```

  The probe verifies the pack database/features/manifest identities against the
  successful pre-Task-0 Plan A gate, computes each line's `pack_sha256` from
  those raw bytes with the frozen formula, requires all lines to carry the same
  four values, and fails before writing if a role is missing, duplicated, or
  uncertified.

  ```bash
  make interaction_funnel_condition_probe
  ./interaction_funnel_condition_probe \
    --pack build/smart-pickup/full-pack \
    --plan-a-catalog-jsonl \
    > resources/g1_funnels/playable_conditions_v1.jsonl
  CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
    resources/sample_g1_funnel_proposals.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --checkpoint build/g1-funnels/training-v1/checkpoint.pt \
    --provider-from-report build/g1-funnels/evidence-v1/report.json \
    --conditions-jsonl resources/g1_funnels/playable_conditions_v1.jsonl \
    --output-catalog resources/g1_funnels/playable_catalog_v1.json \
    --output-path-list resources/g1_funnels/playable_artifact_paths_v1.txt \
    --output resources/g1_funnels/playable_v1
  rm -rf build/g1-funnels/playable-v1-repeat
  CUDA_VISIBLE_DEVICES=0 build/venvs/g1-funnels/bin/python \
    resources/sample_g1_funnel_proposals.py \
    --pack build/smart-pickup/full-pack \
    --dataset build/g1-funnels/dataset-v1 \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --checkpoint build/g1-funnels/training-v1/checkpoint.pt \
    --provider-from-report build/g1-funnels/evidence-v1/report.json \
    --conditions-jsonl resources/g1_funnels/playable_conditions_v1.jsonl \
    --output-catalog build/g1-funnels/playable-v1-repeat/catalog.json \
    --output-path-list build/g1-funnels/playable-v1-repeat/paths.txt \
    --output build/g1-funnels/playable-v1-repeat/artifacts
  cmp resources/g1_funnels/playable_catalog_v1.json \
    build/g1-funnels/playable-v1-repeat/catalog.json
  cmp resources/g1_funnels/playable_artifact_paths_v1.txt \
    build/g1-funnels/playable-v1-repeat/paths.txt
  diff -r resources/g1_funnels/playable_v1 \
    build/g1-funnels/playable-v1-repeat/artifacts
  ```

  Hash the exact `playable_conditions_v1.jsonl` bytes. Group role rows by
  condition digest in first-role occurrence order, requiring canonical condition
  bytes equal within a group. Assign contiguous
  `external_artifact_index_u32=0..U-1` to those unique groups. For each group,
  derive `seed_u64` as the little-endian uint64 in the first eight bytes of
  `SHA256(b"g1-funnel-external-proposal-seed-v1\0" +
  raw_evaluation_manifest_sha256 + raw_conditions_jsonl_sha256 +
  external_artifact_index_u32_le + raw_condition_digest +
  provider_kind_u32_le)`. This is the only external `--conditions-jsonl` seed
  mapping; it is independent of hash-map order and role aliases.

  The catalog canonical JSON has exactly
  `schema_version_u32:1`, `kind:"g1_funnel_playable_catalog"`,
  `winning_provider`, `provider_report_sha256`,
  `evaluation_manifest_sha256`, `conditions_jsonl_sha256`,
  `pack_database_sha256`, `pack_features_sha256`,
  `pack_manifest_sha256`, `pack_sha256`,
  `graphical_table_rack`, and `entries`.
  Entries are in external-artifact-index order and each has exactly
  `external_artifact_index_u32`, `condition_digest`, `roles`, `provider_kind`,
  `seed_u64`, `artifact_file`, and `artifact_sha256`. `roles` preserves the
  probe order and each object has exactly `scene_role`,
  `plan_a_scenario_row_u32`, and `rack_tier_id_u32`. Artifact file is exactly
  `playable_v1/<external_artifact_index-as-8-lowercase-hex>-`
  `<condition_digest>.g1funl`; its header uses namespace `2`, that external
  index, the derived seed, and the winning provider. Generation fails atomically
  unless the active pack, every conditions row, the dataset, evaluation
  manifest, report, and all four catalog pack identities match; `pack_sha256`
  is recomputed from the three raw bytes with the frozen Task-2/4 formula. It
  also requires that every role appears exactly once,
  every unique condition has one byte-reproducible artifact, and catalog/header
  roles, condition, provider, seed, and hashes agree.

  `playable_artifact_paths_v1.txt` is UTF-8 with one repository-relative artifact
  path plus newline per catalog entry in that same order, and no other line.
  `--output-path-list` always emits these logical repository paths from catalog
  `artifact_file`, independent of the physical repeat-test `--output` root, so
  the two path lists above must compare equal.
  Every line begins `resources/g1_funnels/playable_v1/`, equals its catalog path
  after that prefix adjustment, is normalized ASCII without whitespace,
  backslash, empty component, `.` or `..`, and names a regular non-symlink file.
  Tests verify exact role coverage even when table and rack aliases share one
  condition digest.
  `resources/verify_g1_funnel_playable_catalog.py` revalidates the canonical
  conditions/report/manifest/catalog identities, all four identities of the
  required `--pack`, role coverage, safe paths,
  regular non-symlink files, hashes, and artifact headers. Its
  `--verify-staged-paths-stdin` mode accepts `git diff --cached --name-only` and
  requires the staged `.g1funl` set to equal the path-list set exactly; tests
  cover missing/extra/escaped/symlink/stale paths.

  Select `graphical_table_rack` as the lowest
  `(plan_a_scenario_row_u32,rack_tier_id_u32)` for which that same certified Plan
  A row has both a table role and the rack role. It has exactly
  `plan_a_scenario_row_u32`, `rack_tier_id_u32`,
  `table_external_artifact_index_u32`, `table_condition_digest`,
  `rack_external_artifact_index_u32`, and `rack_condition_digest`. This mapping
  is frozen in the catalog and Task 8 must not choose a different row/tier.

- [ ] **Step 3: Integrate explicit learned backend ownership**

  ```cpp
  // Defined once in interaction_funnel_autodemo_config.h.
  enum class PickupProviderMode : uint8_t {
      Authored = 0U,
      Artifact = 1U,
  };

  std::optional<FunnelArtifactCatalog> funnel_catalog;
  std::optional<FunnelArtifactBundle> evaluation_bundle;
  std::optional<ExactFunnelEntryResolver> evaluation_case_resolver;
  std::optional<LearnedSmartPickupAssistBackend> learned_backend;
  std::optional<SmartPickupController> smart_pickup;
  if (environment.funnel.provider_mode == PickupProviderMode::Artifact) {
      const FunnelPackIdentity active_funnel_pack =
          load_funnel_pack_identity(resolved_interaction_pack_root);
      funnel_catalog.emplace(FunnelArtifactCatalog::load(
          *environment.funnel.artifact_catalog, active_funnel_pack));
      FunnelArtifactResolver* resolver = &*funnel_catalog;
      if (environment.autodemo.has_value() &&
          environment.autodemo->mode == AutodemoMode::Funnel) {
          evaluation_bundle.emplace(FunnelArtifactBundle::load(
              *environment.funnel.evaluation_bundle,
              *environment.funnel.evaluation_manifest,
              active_funnel_pack));
          if (*environment.autodemo->funnel_case !=
              FunnelAutodemoCase::TableRackRoundTrip) {
              evaluation_case_resolver.emplace(
                  evaluation_bundle->require(exact_report_key));
              resolver = &*evaluation_case_resolver;
          }
      }
      learned_backend.emplace(*resolver, geometry, follower);
      smart_pickup.emplace(*learned_backend);
  } else {
      smart_pickup.emplace();
  }
  ```

  `ExactFunnelEntryResolver` is a controller-owned adapter that exposes only its
  one already-validated bundle entry and requires both digest and the startup
  report key; it cannot choose a same-digest entry at another evaluation index.
  All Funnel autodemos load the strict bundle for startup identity validation,
  but `table_rack_round_trip` deliberately retains the playable catalog resolver;
  only nominal, blocked-alternate, and all-blocked replace it with the exact
  evaluation-entry resolver.
  `resolved_interaction_pack_root` is the exact already-resolved path used by
  the existing `MM_INTERACTION_PACK`/interactive-pack startup, not a new Funnel
  environment variable. Its loaded `active_funnel_pack` remains in controller
  scope for as long as catalog/bundle validation needs it. Catalog loading
  compares all three raw identities and the recomputed combined digest, then
  loads every artifact with that same value; neither the zero-selector manual
  path nor an autodemo can validate against catalog JSON alone.
  Keep catalog/bundle/resolver/backend/artifacts alive for the full controller
  lifetime. Every F resolves the current condition before movement. F starts the configured
  provider only. X cancels connector/funnel/runtime. R restores Plan A
  scene/route and clears learned attempt diagnostics. WASD remains unchanged.
  A zero-selector artifact environment follows the catalog resolver branch but
  never constructs an evaluation bundle or exact evaluation-case resolver.

- [ ] **Step 4: Implement the existing environment-driven autodemo path**

  `controller.cpp` already has `int main(void)`, `SetTargetFPS(25)`, and
  `parse_autodemo_environment()` using `MM_INTERACTION_AUTODEMO`,
  `MM_INTERACTION_PLACE_AUTODEMO`, `MM_INTERACTION_LOG`, and
  `MM_INTERACTION_SCREENSHOT`. Do not invent controller CLI arguments. Extend
  that parser with mutually exclusive `MM_INTERACTION_FUNNEL_AUTODEMO=1` and
  required `MM_INTERACTION_FUNNEL_CASE`, whose only values are:

  1. `nominal`
  2. `blocked_alternate`
  3. `all_blocked`
  4. `table_rack_round_trip`

  Move the provider mode from Step 3 into
  `interaction_funnel_autodemo_config.h` and extend the existing autodemo types
  exactly:

  ```cpp
  enum class PickupProviderMode : uint8_t {
      Authored = 0U,
      Artifact = 1U,
  };
  enum class AutodemoMode : uint8_t {
      Pickup = 0U,
      Placement = 1U,
      TableRack = 2U,
      Funnel = 3U,
  };
  enum class FunnelAutodemoCase : uint8_t {
      Nominal = 0U,
      BlockedAlternate = 1U,
      AllBlocked = 2U,
      TableRackRoundTrip = 3U,
  };

  struct FunnelProviderEnvironment {
      PickupProviderMode provider_mode = PickupProviderMode::Authored;
      std::optional<std::filesystem::path> artifact_catalog{};
      std::optional<std::filesystem::path> provider_report{};
      std::optional<std::filesystem::path> evaluation_manifest{};
      std::optional<std::filesystem::path> evaluation_bundle{};
  };

  struct ParsedInteractionEnvironment {
      std::optional<AutodemoConfiguration> autodemo{};
      FunnelProviderEnvironment funnel{};
  };

  ParsedInteractionEnvironment parse_autodemo_environment_values(
      const std::map<std::string, std::string>& values);
  ```

  Preserve every existing `AutodemoConfiguration` field and add
  `std::optional<FunnelAutodemoCase> funnel_case`; it is populated exactly when
  `mode == AutodemoMode::Funnel` and is null for the other three modes. Change
  the process-environment `parse_autodemo_environment()` wrapper to return
  `ParsedInteractionEnvironment`, and have `controller.cpp` retain
  `environment.autodemo` wherever it currently retains the optional autodemo
  configuration.

  `tests/cpp/test_interaction_funnel_autodemo_config.cpp` supplies maps and
  counts the existing pickup, placement, and Plan A table/rack selectors plus
  the new funnel selector. Exactly these two zero-selector interactive forms are
  valid:

  1. authored interactive: `MM_PICKUP_PROVIDER` absent or literal `authored`,
     with every `MM_FUNNEL_*` variable absent; and
  2. learned keyboard: `MM_PICKUP_PROVIDER=artifact` plus exactly one nonempty
     normalized `MM_FUNNEL_ARTIFACT_CATALOG`, with
     `MM_FUNNEL_PROVIDER_REPORT`, `MM_FUNNEL_EVALUATION_MANIFEST`,
     `MM_FUNNEL_EVALUATION_BUNDLE`, and `MM_INTERACTION_FUNNEL_CASE` absent.

  Both return `environment.autodemo == std::nullopt`. The second returns
  `provider_mode=Artifact` and the catalog path, and is the WASD/F/X/R learned
  keyboard mode promised by this task. It loads and validates the complete
  active pack identity and playable catalog before Raylib/window initialization
  but does not load a
  Task-7 report or evaluation bundle.

  A selected Funnel autodemo is a separate evaluation mode. It requires literal
  `MM_PICKUP_PROVIDER=artifact` and all four nonempty normalized paths
  `MM_FUNNEL_ARTIFACT_CATALOG`, `MM_FUNNEL_PROVIDER_REPORT`,
  `MM_FUNNEL_EVALUATION_MANIFEST`, and `MM_FUNNEL_EVALUATION_BUNDLE`, in addition
  to its case and existing autodemo paths. This requirement applies to all four
  Funnel cases, including table/rack. Load and cross-check the report, strict
  evaluation bundle, and playable catalog before initializing the window,
  including report `pack_sha256`, manifest raw/combined pack identities, bundle
  combined identity, and catalog raw/combined identities against the same
  active `FunnelPackIdentity`.
  Nominal/blocked/all-blocked resolve the exact Task-7 winning-provider
  evaluation artifact by condition index; table/rack uses the playable digest
  catalog after the same startup identity checks.

  Reject before any Raylib/window call: multiple autodemo mode variables; any
  mode value other than literal `1`; a funnel case without Funnel mode; Funnel
  mode without or with an unknown/case-mismatched case; Funnel mode with authored
  or missing provider; any missing Funnel-evaluation path; artifact provider
  without a catalog; a catalog with authored provider; any report/manifest/bundle
  path in zero-selector manual mode; any Funnel-only variable with a pickup,
  placement, or Plan A table/rack selector; and any empty or non-normalized path.
  Tests prove both valid zero-selector forms, every valid Funnel case, every
  independent rejection, and that
  `MM_INTERACTION_TABLE_RACK_AUTODEMO=1` still parses as
  `AutodemoMode::TableRack` with Plan A's existing requirements unchanged.

  Continue to use exact existing `MM_INTERACTION_PACK`, `MM_INTERACTION_LOG`,
  `MM_INTERACTION_SCREENSHOT`, and `MM_FEATURES_OUTPUT`. Funnel autodemo keeps
  their existing nonempty requirements; manual learned keyboard retains the
  ordinary interactive pack/features defaults and requires no log or screenshot
  path. Preserve the existing
  temporary/backup/atomic evidence publication machinery.

- [ ] **Step 5: Define exact per-case graphical evidence and schema**

  Under `LEARNED_FUNNEL_EVIDENCE_DIR`, write exactly:

  ```text
  nominal.jsonl                  nominal.png
  blocked_alternate.jsonl        blocked_alternate.png
  all_blocked.jsonl              all_blocked.png
  table_rack_round_trip.jsonl    table_rack_round_trip.png
  ```

  Load `report.graphical_cases` and the immutable Task-2 manifest before scene
  construction. `nominal`, `blocked_alternate`, and `all_blocked` use exactly the
  frozen runtime-case index, condition index/digest, complete target, live root,
  scene ID, and ordered obstacle float values named by their corresponding
  report record; any discrepancy is fatal. `nominal` must reach actual attached
  Carry and pass the Task-7 `0.20 m` post-Carry witness. `blocked_alternate` must
  first prove the report's nominal rank-zero proposal is rejected by the exact
  frozen obstacle with the recorded reason/segment/obstacle index, then select
  the report's different same-batch proposal and reach actual attached Carry.
  `all_blocked` must reject all 32 with no request/root motion/ownership/edge,
  then move under one ordinary manual-recovery command. A null graphical report
  mapping is fatal; the controller never searches for another case.

  `table_rack_round_trip` uses exactly the catalog's
  `graphical_table_rack.plan_a_scenario_row_u32` and `rack_tier_id_u32`, including
  its named table/rack condition artifacts and the existing Plan A scene/support
  constructors for that row. It performs learned table pickup, attached Carry,
  Plan A placement on that rack surface, learned re-pick from the resulting
  target generation at that tier, attached Carry, and Plan A placement back on
  the table surface, all without reset. Both placements must commit to the exact
  `SurfaceHandle{id,generation}` selected for the leg and report the actual
  `PlaceMotionMode` (`recorded_place`, `precomputed_reversed_pickup`, or
  same-pickup `reversed_pickup`). The shipped certified rack tier preserves Plan
  A's `precomputed_reversed_pickup` path when its catalog names it. Each
  placement tick also records Plan A's candidate `place_source_id_u64`, source
  and destination affordance IDs, `requested_support_height_m`,
  `source_support_height_m`, `target_support_height_m`, requested and applied
  vertical correction, and full source/destination surface handles. The
  validator compares every value to the frozen selected candidate and Plan A's
  `RuntimePlaceDiagnostics`; it never infers provenance, correction, support, or
  success from object height alone. Once a placement candidate exists, the five
  immutable Plan A provenance values (`place_source_id_u64`, the three support
  heights, and `requested_vertical_correction_m`) remain unchanged through that
  placement leg; only `applied_vertical_correction_m` advances as Plan A applies
  it.

  Freeze the evidence-only domains before defining the record schema:

  ```cpp
  enum class FunnelEvidenceRouteLeg : uint8_t {
      None = 0U,
      TablePickup = 1U,
      RackPlacement = 2U,
      RackPickup = 3U,
      TablePlacement = 4U,
      Complete = 5U,
  };

  enum class FunnelEvidenceOutcome : uint8_t {
      Succeeded = 0U,
  };

  enum class FunnelEvidenceTerminationReason : uint8_t {
      NominalPostCarryWitness = 0U,
      BlockedAlternateAttachedCarry = 1U,
      AllBlockedManualRecovery = 2U,
      TableRackRoundTripComplete = 3U,
  };
  ```

  Their serialized strings are respectively
  `"none"|"table_pickup"|"rack_placement"|"rack_pickup"|"table_placement"|"complete"`,
  `"succeeded"`, and
  `"nominal_post_carry_witness"|"blocked_alternate_attached_carry"|`
  `"all_blocked_manual_recovery"|"table_rack_round_trip_complete"`.
  Non-table/rack cases use route leg `none` on every record. Table/rack begins at
  `table_pickup`, advances only after the corresponding attached Carry or exact
  support commit through the four named action legs, and uses `complete` on the
  final commit tick and repeated terminal record. No failed run publishes a
  terminal record or replaces prior evidence, so the only published outcome is
  `succeeded`; its termination reason is the one value corresponding to `case`.

  `selected_proposal_u32` is JSON null before a winner is frozen and throughout
  `all_blocked`; otherwise it is an integer in `0..31`. Once nonnull it remains
  fixed through that learned pickup and its following placement. At the second
  table/rack learned-pickup activation it resets to null, then freezes the second
  winner and remains that value through the final placement and terminal record.
  `nominal_rank_zero_proposal_u32` is nonnull only for `blocked_alternate`, and
  `nominal_rank_zero_rejected` becomes true only after its exact recorded
  rejection; the other three cases use null/false on every record.

  Every native tick emits one `tick` object with exactly these keys; a terminal
  object is a separate final line and has the same base keys plus only the
  terminal keys listed afterward:

  ```text
  affordance_id_u32
  applied_vertical_correction_m        (null when no placement candidate)
  artifact_sha256
  attached
  attachment_edge_count_u32
  backend_reason
  case
  checkpoint_sha256
  condition_index_u32                 (null only for table/rack external rows)
  condition_sha256
  connector_distance_m
  connector_tick_u32
  controller_rate_hz_u32              (=25)
  dataset_sha256
  destination_place_affordance_id_u32  (null when no placement leg)
  destination_support_committed
  destination_surface_generation_u32  (null when no placement leg)
  destination_surface_id_u64          (null when no placement leg)
  external_artifact_index_u32          (null for Task-2 evaluation rows)
  follow_total_u32                     (0 endpoint, 16 otherwise)
  follower_reason
  funnel_distance_m
  manifest_sha256
  minimum_clearance_m                   (null until a proposal is selected)
  nominal_rank_zero_proposal_u32       (nonnull only for blocked_alternate)
  nominal_rank_zero_rejected
  object_position_m                    ([3])
  object_state
  obstacle_count_u32
  obstacles_sha256
  owner_request_u64
  place_motion_mode                    ("none"|"recorded_place"|
                                        "precomputed_reversed_pickup"|
                                        "reversed_pickup")
  place_source_id_u64                  (null when no placement candidate)
  plan_a_scenario_row_u32              (null outside table/rack)
  provider_config_sha256
  provider_kind
  published_sample_count_u32
  published_sample_index_u32           (null or the one index emitted this tick)
  rack_tier_id_u32                     (null outside table/rack)
  record_type                          ("tick"|"terminal")
  rejection_counts
  render_frame_u64
  render_rate_hz_u32                   (=25)
  request_count_u32
  request_id_u64                       (0 before submission)
  requested_support_height_m           (null when no placement candidate)
  requested_vertical_correction_m      (null when no placement candidate)
  result
  root_position_m                      ([3])
  route_leg                            (exact FunnelEvidenceRouteLeg string)
  runtime_case_index_u32               (null only for table/rack)
  runtime_state
  runtime_tick_u64
  sample_count_u32
  scene_id
  schema_version_u32                   (=1)
  selected_proposal_u32                (null or integer 0..31 as frozen above)
  source_place_affordance_id_u32       (null when no placement leg)
  source_support_height_m              (null when no placement candidate)
  source_surface_generation_u32        (null when no placement leg)
  source_surface_id_u64                (null when no placement leg)
  step_seconds                         (=0.04)
  target_generation_u32
  target_id_u64
  target_support_height_m              (null when no placement candidate)
  ```

  `rejection_counts` has exactly `malformed_u32`, `connector_u32`, `table_u32`,
  `obstacle_u32`, `preview_batch_u32`, and `preview_u32`. `obstacles_sha256` is
  the Task-5 ordered-obstacle hash and must equal recomputation from the frozen
  manifest scene (or the canonical zero-obstacle hash). When nonnull,
  `minimum_clearance_m` must equal the selected proposal's Task-7 report value
  and the backend's stored Task-5 float32 value after promotion to JSON; null is
  required whenever `selected_proposal_u32` is null. `result` uses the existing
  `ResultCode` enumerator names, `route_leg` uses the frozen evidence mapping
  above, and all reason/mode/state fields use their declared enum-name strings,
  never integer casts.

  For a placement candidate, serialize
  `requested_support_height_m`, `source_support_height_m`, and
  `target_support_height_m` from the same frozen `RuntimePlaceDiagnostics` as
  source ID and requested correction. The Python validator parses each number,
  casts it once to IEEE-754 float32, and requires raw-bit equality with the
  selected Plan A catalog/diagnostic value; it also enforces Plan A's declared
  relationships among requested support, source support, target support, and
  requested correction. All three are null before/after a placement candidate
  under the same lifetime rule as `place_source_id_u64`; object Y is never a
  substitute.

  `published_sample_index_u32` plus cumulative count lets the validator
  require exactly `0..15` once each for KNN/diffusion and no publication for
  endpoint. In the blocked case, evidence pins both the rejected nominal proposal
  and the different selected proposal.

  A terminal record adds exactly these keys:

  ```text
  carry_object_displacement_m
  carry_root_displacement_m
  collapsed_runtime_states
  manual_recovery_displacement_m
  maximum_case_ticks_u32
  maximum_grasp_rotation_error_radians
  maximum_grasp_translation_error_m
  outcome                              ("succeeded")
  root_motion_before_request_m
  termination_reason                   (exact case-mapped string above)
  ```

  JSON encoding is UTF-8 canonical JSON: keys lexicographically sorted, no
  insignificant whitespace, one object followed by one `\n`, lowercase hex,
  JSON booleans/null, and arrays in declared order. C++ finite floats use
  `std::to_chars(..., std::chars_format::general,
  std::numeric_limits<double>::max_digits10)` after converting negative zero to
  positive zero; NaN/infinity are fatal. No unlisted key or missing nullable key
  is accepted. Evidence begins at frame/tick zero; every next tick line increments
  both by one, requires `render_frame_u64 == runtime_tick_u64`, and the terminal
  line repeats the last frame/tick rather than simulating another tick. PNGs must
  have valid signature/IHDR, exact 1280x720 dimensions, and size greater than
  10,000 bytes.

  Internal deterministic maximums are 750 ticks for nominal, 750 for blocked-
  alternate, 125 for all-blocked, and 3,000 for table/rack. The controller exits
  zero only after the case's exact terminal contract and atomic publication.
  Unexpected reject/cancel/failure or maximum-tick exhaustion throws, cleans
  temporaries, and exits nonzero. All-blocked expected rejection is success only
  after its no-motion and manual-recovery assertions.

- [ ] **Step 6: Implement validator corruption tests before adding the Make gate**

  First build valid synthetic four-case evidence in temporary directories, then
  independently corrupt and assert rejection for: missing JSONL/PNG, unexpected
  fifth file, truncated/malformed JSON, wrong schema/case/file name, duplicate or
  skipped tick, tick/frame mismatch, terminal tick advance, unsorted/extra/missing
  key, noncanonical number/negative zero, 60-Hz rate or non-0.04 step, NaN, wrong
  artifact/condition/provider, endpoint with nonzero follow count, 16-sample
  provider with a missing/duplicate sample, incomplete runtime lifecycle,
  attachment/owner mismatch, obstacle count/hash mismatch, blocked nominal not
  rejected, wrong rejection reason/index, blocked alternate reusing nominal proposal,
  all-blocked nonnull selected proposal, request/root motion, missing manual
  recovery, selected-proposal/clearance mismatch, uncapped or wrong finite
  clearance, wrong route-leg value or transition, non-table route leg, wrong
  outcome, wrong case-specific terminal reason,
  table/rack row/tier drift, wrong source/destination surface handle, uncommitted
  support, wrong source/destination affordance, place source ID, requested or
  applied vertical correction, requested/source/target support height or float32
  support bits, support-height/correction relationship, loss of expected
  precomputed-reversed mode, invalid place-motion mode, wrong PNG
  signature/dimensions, and
  premature/max-tick termination. Run these
  unit tests before writing the Make target:

  ```bash
  make build/tests/test_interaction_funnel_catalog \
       build/tests/test_interaction_funnel_autodemo_config \
       build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_funnel_catalog
  build/tests/test_interaction_funnel_autodemo_config
  build/tests/test_interaction_controller_adapter
  build/venvs/g1-funnels/bin/python -m unittest \
    tests.python.test_g1_funnel_playable_catalog \
    tests.python.test_playable_learned_funnel_evidence -v
  ```

- [ ] **Step 7: Run the four environment-driven cases and final umbrella gates**

  Add an evidence gate that invokes the controller only in explicit learned mode
  and validates the external evidence directory:

  ```make
  LEARNED_FUNNEL_EVIDENCE_DIR ?= /home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-b
  LEARNED_FUNNEL_PACK ?= build/smart-pickup/full-pack
  LEARNED_FUNNEL_FEATURES_DIR ?= build/g1-funnels/playable-features-v1
  LEARNED_FUNNEL_REPORT ?= $(CURDIR)/build/g1-funnels/evidence-v1/report.json
  LEARNED_FUNNEL_EVALUATION_MANIFEST ?= $(CURDIR)/resources/g1_funnels/evaluation_manifest_v1.json
  LEARNED_FUNNEL_EVALUATION_BUNDLE ?= $(CURDIR)/build/g1-funnels/proposals-v1

  .PHONY: gate-playable-learned-funnel
  gate-playable-learned-funnel: controller \
    build/tests/test_interaction_funnel_catalog \
    build/tests/test_interaction_funnel_autodemo_config \
    build/tests/test_interaction_controller_adapter
	@mkdir -p "$(LEARNED_FUNNEL_EVIDENCE_DIR)" \
	  "$(LEARNED_FUNNEL_FEATURES_DIR)"
	@build/tests/test_interaction_funnel_catalog
	@build/tests/test_interaction_funnel_autodemo_config
	@build/tests/test_interaction_controller_adapter
	@build/venvs/g1-funnels/bin/python -m unittest \
	  tests.python.test_g1_funnel_playable_catalog \
	  tests.python.test_playable_learned_funnel_evidence -v
	@build/venvs/g1-funnels/bin/python \
	  resources/verify_g1_funnel_playable_catalog.py \
	  --pack "$(LEARNED_FUNNEL_PACK)" \
	  --catalog resources/g1_funnels/playable_catalog_v1.json \
	  --conditions resources/g1_funnels/playable_conditions_v1.jsonl \
	  --path-list resources/g1_funnels/playable_artifact_paths_v1.txt \
	  --report "$(LEARNED_FUNNEL_REPORT)" \
	  --manifest "$(LEARNED_FUNNEL_EVALUATION_MANIFEST)" \
	  --validate-files
	@display="$${DISPLAY:-:1}"; \
	  timeout --signal=TERM --kill-after=1s 5s \
	    xdpyinfo -display "$$display" >/dev/null
	@set -eu; \
	  display="$${DISPLAY:-:1}"; \
	  for case_name in nominal blocked_alternate all_blocked table_rack_round_trip; do \
	    case "$$case_name" in \
	      nominal|blocked_alternate) wall_timeout=90 ;; \
	      all_blocked) wall_timeout=30 ;; \
	      table_rack_round_trip) wall_timeout=240 ;; \
	    esac; \
	    DISPLAY="$$display" \
	    MM_INTERACTION_FUNNEL_AUTODEMO=1 \
	    MM_INTERACTION_FUNNEL_CASE="$$case_name" \
	    MM_INTERACTION_PACK="$(LEARNED_FUNNEL_PACK)" \
	    MM_PICKUP_PROVIDER=artifact \
	    MM_FUNNEL_ARTIFACT_CATALOG="$(CURDIR)/resources/g1_funnels/playable_catalog_v1.json" \
	    MM_FUNNEL_PROVIDER_REPORT="$(LEARNED_FUNNEL_REPORT)" \
	    MM_FUNNEL_EVALUATION_MANIFEST="$(LEARNED_FUNNEL_EVALUATION_MANIFEST)" \
	    MM_FUNNEL_EVALUATION_BUNDLE="$(LEARNED_FUNNEL_EVALUATION_BUNDLE)" \
	    MM_FEATURES_OUTPUT="$(LEARNED_FUNNEL_FEATURES_DIR)/$$case_name.bin" \
	    MM_INTERACTION_LOG="$(LEARNED_FUNNEL_EVIDENCE_DIR)/$$case_name.jsonl" \
	    MM_INTERACTION_SCREENSHOT="$(LEARNED_FUNNEL_EVIDENCE_DIR)/$$case_name.png" \
	      timeout --signal=TERM --kill-after=5s "$$wall_timeout" ./controller; \
	  done
	@LEARNED_FUNNEL_EVIDENCE_DIR="$(LEARNED_FUNNEL_EVIDENCE_DIR)" \
	 build/venvs/g1-funnels/bin/python -m unittest \
	   tests.python.test_playable_learned_funnel_evidence -v
  ```

  ```bash
  make gate-smart-pickup-plan-a \
    INTERACTION_DEMO_PACK=build/smart-pickup/full-pack
  make gate-learned-funnel-headless
  make gate-learned-funnel-release-parity
  make controller
  mkdir -p \
    /home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-b
  make gate-playable-learned-funnel \
    LEARNED_FUNNEL_EVIDENCE_DIR=/home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/plan-b
  make gate-object-grasp-diffusion-plan-b
  ```

  Define the final target with the hard dependency:

  ```make
  .PHONY: gate-smart-pickup-plan-a-full-pack
  gate-smart-pickup-plan-a-full-pack:
	@$(MAKE) gate-smart-pickup-plan-a \
	  INTERACTION_DEMO_PACK=build/smart-pickup/full-pack

  .PHONY: gate-object-grasp-diffusion-plan-b
  gate-object-grasp-diffusion-plan-b: gate-smart-pickup-plan-a-full-pack
	@$(MAKE) gate-learned-funnel-headless
	@$(MAKE) gate-learned-funnel-release-parity
	@$(MAKE) gate-playable-learned-funnel \
	  LEARNED_FUNNEL_PACK=build/smart-pickup/full-pack
  ```

- [ ] **Step 8: Review, commit, and push**

  ```bash
  git add controller.cpp interaction_debug_draw.h \
    interaction_funnel_autodemo_config.h \
    interaction_funnel_autodemo_config.cpp \
    interaction_funnel_condition_probe.cpp \
    resources/sample_g1_funnel_proposals.py \
    resources/verify_g1_funnel_playable_catalog.py \
    interaction_funnel_catalog.h interaction_funnel_catalog.cpp \
    tests/cpp/test_interaction_funnel_catalog.cpp \
    tests/cpp/test_interaction_funnel_autodemo_config.cpp \
    tests/cpp/test_interaction_controller_adapter.cpp \
    tests/python/test_g1_funnel_playable_catalog.py \
    tests/python/test_playable_learned_funnel_evidence.py \
    resources/g1_funnels/playable_conditions_v1.jsonl \
    resources/g1_funnels/playable_catalog_v1.json \
    resources/g1_funnels/playable_artifact_paths_v1.txt Makefile
  build/venvs/g1-funnels/bin/python \
    resources/verify_g1_funnel_playable_catalog.py \
    --pack build/smart-pickup/full-pack \
    --catalog resources/g1_funnels/playable_catalog_v1.json \
    --conditions resources/g1_funnels/playable_conditions_v1.jsonl \
    --path-list resources/g1_funnels/playable_artifact_paths_v1.txt \
    --report build/g1-funnels/evidence-v1/report.json \
    --manifest resources/g1_funnels/evaluation_manifest_v1.json \
    --validate-files
  while IFS= read -r artifact_path; do
    git add -- "$artifact_path"
  done < resources/g1_funnels/playable_artifact_paths_v1.txt
  git diff --cached --name-only --diff-filter=AM | \
    build/venvs/g1-funnels/bin/python \
      resources/verify_g1_funnel_playable_catalog.py \
      --pack build/smart-pickup/full-pack \
      --catalog resources/g1_funnels/playable_catalog_v1.json \
      --conditions resources/g1_funnels/playable_conditions_v1.jsonl \
      --path-list resources/g1_funnels/playable_artifact_paths_v1.txt \
      --report build/g1-funnels/evidence-v1/report.json \
      --manifest resources/g1_funnels/evaluation_manifest_v1.json \
      --verify-staged-paths-stdin
  git diff --cached --check
  git commit -m "feat: expose learned smart pickup funnels"
  git push checkpoint HEAD:g1-tabletop-placement
  ```

## Plan B Completion Gate

Completion requires byte-reproducible artifacts, the frozen all-row provider
report, actual held attachment for at least three proposal-model-held-out
conditions, same-batch blocked alternate, all-blocked fail-closed behavior, and
native-25-Hz graphical evidence. If diffusion does not beat the frozen simpler
baseline under the continuation rule, the experiment is still complete: retain
the proposal-provider interface and ship the winning simpler provider. GraspMolmo
remains a future producer of the exact `GraspCondition` boundary, not a trajectory
generator and not part of this plan.
