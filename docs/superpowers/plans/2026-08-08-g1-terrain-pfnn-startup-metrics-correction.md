# G1 Terrain PFNN Startup-Metrics Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the base-only startup metric key selection, prove the real path before any update, and authorize one fresh physical-envelope artifact attempt.

**Architecture:** Keep every physical-envelope and artifact contract frozen. Replace the positional loss-key slice with a named tuple derived only from base keys, add one direct uneven-batch regression, then repeat the complete source and artifact gates at a new output path.

**Tech Stack:** Python 3.10, PyTorch, NumPy, `unittest`, CUDA GPU 1.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-08-08-g1-terrain-pfnn-startup-metrics-correction-design.md`.
- Modify only `sonic/python/mm_sonic/terrain_pfnn/training.py` and `tests/python/test_terrain_pfnn_training.py` in Task 1.
- Use an explicit tuple of `BASE_LOSS_WEIGHT_KEYS` excluding named key `regularization`; never use positional slicing.
- Dataset v2, checkpoint v6, one-step report v2, fitted report v1, and promotion receipt v3 remain unchanged.
- Preserve `sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v1` byte-for-byte.
- Task 3 may create only `sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v2` plus ignored evidence.
- Exactly one new `4,000 + 256` attempt is authorized after Tasks 1-2 pass and review approves.
- At first failure stop: no evaluator, retry, resume, patch, parameter change, or post-freeze source/test edit or commit.
- The sealed test split remains closed until original Task 9.

---

### Task 1: Correct and directly test startup metrics

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

**Interfaces:**
- Consumes: `BASE_LOSS_WEIGHT_KEYS`, base-only `pfnn_losses`, and immutable weights.
- Produces: base-only `one_step_metrics` with ten nonregularized loss means, `one_step_score`, and `samples`.

- [ ] **Step 1: Write the direct uneven-batch RED**

Import `BASE_LOSS_WEIGHT_KEYS`. Add
`test_one_step_metrics_uses_exact_nonregularized_base_keys_and_uneven_batch_score`.
Use three deterministic identity-normalized rows, batch size two,
`PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)`, and
`_ZeroKinematics`. Call the real `one_step_metrics`; independently call real
`pfnn_losses` once on all three rows and require:

```python
names = tuple(
    name for name in BASE_LOSS_WEIGHT_KEYS if name != "regularization"
)
self.assertEqual(
    tuple(observed), (*names, "one_step_score", "samples")
)
self.assertEqual(observed["samples"], 3)
for name in names:
    self.assertAlmostEqual(observed[name], float(expected[name]), places=6)
self.assertAlmostEqual(
    observed["one_step_score"],
    sum(float(expected[name]) * DEFAULT_LOSS_WEIGHTS[name] for name in names),
    places=6,
)
```

Spy on `_sample_batch` and require batch lengths `[2, 1]`. Targets must use
binary contact labels. This test calls `one_step_metrics` directly; do not mock
it or `pfnn_losses`.

- [ ] **Step 2: Run RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_one_step_metrics_uses_exact_nonregularized_base_keys_and_uneven_batch_score
```

Expected: `KeyError: 'joint_step_envelope'` from the real aggregator.

- [ ] **Step 3: Apply the exact source fix and tighten the mock**

Replace only the key-selection line with:

```python
names = tuple(
    name for name in BASE_LOSS_WEIGHT_KEYS if name != "regularization"
)
```

In `_run_one_step_trainer_probe`, make its mocked `evaluation_metrics` return
only those ten names plus `one_step_score` and `samples`; require that exact key
order inside the fixture. It must not return regularization or envelope keys.

- [ ] **Step 4: Run GREEN and focused suite**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_one_step_metrics_uses_exact_nonregularized_base_keys_and_uneven_batch_score

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training
```

Expected: all pass.

- [ ] **Step 5: Commit and review the two-file correction**

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "fix: restrict PFNN startup metrics to base losses"
```

Review `5b8b4a4..HEAD` for spec compliance and code quality. Any
Critical/Important finding stops the cycle before Task 2; do not broaden the
fix or create a second source commit without separate approval.

---

### Task 2: Prove the complete source and no-update real-data gate

**Files:**
- Write ignored: `.superpowers/sdd/startup-metrics-correction-source-report.md`
- Do not modify source or tests

**Interfaces:**
- Consumes: the reviewed Task 1 commit and accepted recurrence-v2 train data.
- Produces: 204-test/static evidence, a real GPU startup-score result with zero updates, and a frozen reviewed source head.

- [ ] **Step 1: Run all 204 PFNN tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_sources \
  tests.python.test_terrain_pfnn_phase

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_dataset \
  tests.python.test_terrain_pfnn_kinematics \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime
```

Expected: 23/23 and 181/181 pass.

- [ ] **Step 2: Run static checks**

```bash
git diff --check
find sonic/python/mm_sonic/terrain_pfnn -maxdepth 1 -name '*.py' -print | sort | \
  xargs sonic/.torch-mm-venv/bin/python -m py_compile
printf '%s\n' sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/evaluate_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_runtime.py | \
  xargs sonic/.torch-mm-venv/bin/python -m py_compile
```

- [ ] **Step 3: Run the prescribed real-data GPU-1 no-update smoke**

Run a read-only Python heredoc with `CUDA_VISIBLE_DEVICES=1` that reuses the
Task 6 train-only dataset view, exact runtime-seed selection, and
`materialize_subset` to obtain the same 2,048 rows. Construct the real
`PhaseFunctionedNetwork(hidden_size=512, dropout_probability=0.30)` and real
`TorchG1ForwardKinematics` on CUDA, call:

```python
metrics = one_step_metrics(
    model,
    fitted_subset,
    kinematics=kinematics,
    batch_size=256,
    device="cuda",
    loss_weights=DEFAULT_LOSS_WEIGHTS,
)
```

Require exact keys
`tuple(name for name in BASE_LOSS_WEIGHT_KEYS if name != "regularization") +
("one_step_score", "samples")`, `samples == 2048`, and every value finite.
The script must instantiate no optimizer, call no backward/update, create no
run directory, and open no validation/test example. Print one canonical JSON
record containing the score, sample count, exact keys, loaded split counts,
dataset digest, `optimizer_steps: 0`, and `output_created: false`.

- [ ] **Step 4: Obtain full-range review and freeze**

Provide the design, plan, Task 7 artifact report, Task 1 RED/GREEN evidence,
smoke JSON, narrow `5b8b4a4..HEAD` diff, and full `d94b443..HEAD` diff to an
independent reviewer. Require explicit spec-compliance and code-quality
approval with no Critical/Important findings. Record exact HEAD, UTC freeze
time, all source/test hashes, commands, counts, timings, and verdicts in the
source report. From this freeze through Task 3, source/tests admit no edit or
commit.

---

### Task 3: Execute and gate one fresh physical-envelope-v2 artifact

**Files:**
- Generate: `sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v2/`
- Write ignored: `.superpowers/sdd/startup-metrics-correction-dataset-before.json`
- Write ignored: `.superpowers/sdd/startup-metrics-correction-dataset-after.json`
- Write ignored: `.superpowers/sdd/startup-metrics-correction-artifact-report.md`

**Interfaces:**
- Consumes: frozen reviewed source, accepted recurrence-v2 data, canonical G1 MJCF, GPU 1.
- Produces on success: accepted v6 candidate, v2 one-step report, accepted v1 fitted report, 600-tick evaluation, v3 promotion receipt, immutable best.

- [ ] **Step 1: Preserve v1 and snapshot the dataset around exact resume**

Require the v1 directory still contains only its zero-byte `metrics.jsonl`
with SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
record its size, mode, and `mtime_ns` before and after Task 3.

```bash
PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -c \
  'from pathlib import Path; from mm_sonic.train_terrain_pfnn import write_dataset_file_snapshot; write_dataset_file_snapshot(Path("sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2"), Path(".superpowers/sdd/startup-metrics-correction-dataset-before.json"))'

PYTHONPATH=sonic/python /home/ubuntu/miniconda3/bin/python \
  -m mm_sonic.build_terrain_pfnn_dataset \
  --grail-root /home/ubuntu/datasets/GRAIL \
  --lafan-root /home/ubuntu/.cache/g1-lafan-flat/g1 \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2 \
  --terrain-family-limit 2 --resume

PYTHONPATH=sonic/python sonic/.torch-mm-venv/bin/python -c \
  'from pathlib import Path; from mm_sonic.train_terrain_pfnn import write_dataset_file_snapshot; write_dataset_file_snapshot(Path("sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2"), Path(".superpowers/sdd/startup-metrics-correction-dataset-after.json"))'
```

Require the same six sorted path/hash/size/mtime entries, equal canonical
snapshot digests, dataset digest
`9e304bafe780aa694467e01912b7a04a67a0526d3c81a612a9618658ba4bdd48`,
v2 lanes, disjoint splits, and train-only normalization without deserializing
test examples.

- [ ] **Step 2: Confirm fresh output and GPU 1**

```bash
test ! -e sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v2
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

Reconfirm frozen HEAD/hashes and record the final UTC boundary.

- [ ] **Step 3: Launch the sole authorized attempt**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.train_terrain_pfnn \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v2 \
  --overfit-samples 2048 --steps 4000 --seed 7 \
  --rollout-finetune-frames 16 --rollout-finetune-steps 256
```

Report steps and losses about every two minutes. Require finite losses, v6
candidate `checkpoint-step-00004256.pt`, safe reload, exact fixed-sample
reproduction, exact objective/pair/target receipts, and an accepted 2,011-pair
fitted report. Require no `best.pt` yet. At first failure, record exact
provenance/value/limit and every digest, then stop without evaluator or retry.

- [ ] **Step 4: Run the sole evaluator only after fitted acceptance**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.evaluate_terrain_pfnn \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v2/checkpoint-step-00004256.pt \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --split train --closed-loop-seconds 20 --promote-pipeline-best
```

Require 600 finite ticks, zero holds/reversals/freezes, every physical/contact
gate, supported traversal, exact bindings, immutable mode-`0444` `best.pt`, and
v3 `pipeline-promotion-receipt.json`.

- [ ] **Step 5: Post-verify and report**

Safely load candidate and best with the receipt-bound dataset and kinematic
digests; require identical candidate/best SHA-256, accepted and identical fitted
report digests in `one-step-report.json` and `train-evaluation.json`, exact
objective/pair digests, accepted 600-tick payload, atomic receipt, no test state,
and unchanged v1 evidence. Re-run the complete 204-test commands and static
checks from Task 2. Record commands, timings, loss curves, metrics, hashes,
permissions, callback count, snapshots, and first-failure absence/presence in
the artifact report. Create no source/test commit.

## Successful handoff only

If Task 3 passes completely, independently re-review original Task 7 from
`6af614f` through the frozen startup-correction head, including both correction
designs/plans and artifact reports. Any Critical/Important finding stops without
source change. A blocker-free review resumes original Tasks 8 and 9 exactly,
ending with the required interactive W/A/S/D three-hill drive and final evidence.
