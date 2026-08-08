# G1 Terrain PFNN Bounded-Envelope Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scale-dominating three-family physical-envelope training
term with a checkpoint-bound, physical-unit, smooth-L1 four-family surrogate
that can pass the unchanged complete fitted and runtime gates without regressing
the base PFNN fit.

**Architecture:** Keep dataset v2, the `288 -> 512 -> 512 -> 268` PFNN,
recurrence, raw diagnostics, and runtime authorization unchanged. Compute
joint-step, native-limit, phase, and trajectory-direction excesses in physical
units; take the worst element per pair; apply the existing deterministic global
maximum-plus-positive-tail reduction independently; and multiply each optimized
family by the preregistered candidate `1 / 256`. Bind objective v2, target and
mandatory observability audits, checkpoint v7, report v3, and promotion v4.
Measure all 2,011 fitted pairs before and after rollout, then permit exactly one
fresh `4,000 + 256` v3 run after all source, feasibility, per-update gradient-
direction, real shadow-optimizer, no-update, and review gates pass.

**Tech Stack:** Python 3.10, PyTorch, NumPy, `unittest`, CPU `gloo` distributed
tests, CUDA GPU 1 for the real-data audit and sole artifact run.

## Global Constraints

- Implement the approved
  `docs/superpowers/specs/2026-08-08-g1-terrain-pfnn-bounded-envelope-surrogate-design.md`
  without reopening unchanged contracts in the physical-envelope design.
- Dataset stays `mm-sonic-terrain-pfnn-dataset/v2`; fitted pair receipt and
  fitted-transition gate stay v1; model, normalization, sampler, recurrence,
  fixed metrics, controller, raw outputs, and evaluation semantics stay fixed.
- Runtime stays exact: root translation `<= 0.060 m`, root rotation
  `<= 0.35 rad`, joint step `<= 0.25 rad`, native limits, positive root height,
  finite normalized root quaternion, direction norm `[0.5,1.5]`, and phase
  `[0,min(pi,1.5*q99)]`. Training direction bounds are `[0.55,1.45]` only.
- Preserve normalized raw risk/max/CVaR/positive-tail/active/runtime diagnostics
  for joint step, native limit, and phase; add the same raw direction family.
  Raw risks are not optimized under objective v2.
- Optimize exact physical surrogates: joint-step smooth-L1 beta `.025 rad`,
  native-limit beta `.020 rad`, negative phase linear in radians, upper phase
  beta `.020 rad`, and direction norm excess outside `[.55,1.45]` with beta
  `.05`. Take per-pair worst joint/knot before each independent global
  max-plus-positive-tail reduction.
- Each optimized family is multiplied once by exact
  `1 / 256 == 0.00390625`. This is a preregistered candidate motivated by the
  base-mean/nondilutable-reduction mismatch and observed v2 `10^4`-scale norm
  blow-up. It is not an algebraic mean identity or a fitted coefficient; the
  real per-update and shadow-optimizer gates must admit it before training.
- Objective advances to v2, checkpoint to v7, one-step report to v3, promotion
  to v4, and strict direction-aware target audit to v2. Runtime and fitted gate
  remain unchanged.
- No input expansion, PCGrad, clamp, mask, repair, IK, projection, threshold
  change, raw loss-weight/LR/budget guess, output patch, or v1/v2 resume.
- Preserve v1 and v2 byte-for-byte. The v2 candidate SHA-256 is
  `866e4796c6d7ed8a873e47c443498e560bdb31f50a70618bcdceaae12bd73ff0`,
  metrics is `97de546d9fb48ff6540df8f4bc3005f7426dfe8727e2ef6b55b652addd47068a`,
  and report is `cbe986fb74a29714675908b10da3207225f6017178cf359e6ba1d7dd0561473b`.
- Validation remains selection-only. Do not deserialize a sealed test example
  before original Task 9.
- Source work uses TDD. After the reviewed pre-run freeze, no correction
  source/test edit or commit is authorized. The first real failure stops the
  stage; never retry/resume the sole run or tune/patch in cycle.
- The exact-input interval observability audit is mandatory, canonical, and
  digest-bound. Any empty intersection stops before v3; zero empties prove only
  that this necessary fitted-population contradiction was not found.
- Complete pre/post fitted metrics belong only to strict report v3. Checkpoint
  v7 contains neither payload nor digest; no partial checkpoint/report may
  create an intermediate ownership state.

**Loss-key contract:** retain `BASE_LOSS_WEIGHT_KEYS`; retain existing raw
diagnostic names and add `direction_envelope`; add optimized names:

```python
BOUNDED_ENVELOPE_LOSS_KEYS = (
    "joint_step_bounded_envelope",
    "joint_limit_bounded_envelope",
    "phase_bounded_envelope",
    "direction_bounded_envelope",
)
```

Each bounded value already includes `1/256` and enters `total` exactly once.
Raw family values remain reports only.

---

### Task 1: Implement objective v2, physical surrogates, and four-family reduction

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

- [ ] **Step 1: Write RED contract and boundary tests**

Extend `PhysicalEnvelopeObjective` with exact schema v2 and fields:

```text
direction lower/upper/scale/hinge = .55 / 1.45 / .05 / 2
smooth-L1 betas                   = .025 / .020 / .020 / .05
reference count/coefficient       = 256 / .00390625
```

Test exact field/type/value validation; missing/extra fields; bool-for-int;
nonfinite/nonpositive beta; reversed direction bounds; coefficient identity;
and v1 rejection. For each beta, require smooth-L1 values and derivatives at
`0`, `beta/2`, `beta`, and `2*beta`: values
`0, beta/8, beta/2, 1.5*beta`, derivatives `0, .5, 1, 1`.

Test exact zero value/gradient at joint step `.225`, native insets `.020`, phase
zero/upper inset, and direction norms `.55/1.45`; finite capped gradients just
outside; lower-phase derivative `-1` for every strict negative; worst joint/knot
before global reduction; finite zero-vector direction behavior; and unchanged
old raw risk tensors.

- [ ] **Step 2: Run RED tests**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_physical_envelope_objective_v2_requires_exact_canonical_fields \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_smooth_l1_physical_excess_has_exact_values_and_capped_gradients \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_bounded_surrogates_use_physical_boundaries_and_worst_element \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_direction_raw_and_bounded_bands_are_distinct
```

Expected: FAIL because the v2 fields, helper, and direction family are absent.

- [ ] **Step 3: Implement the strict objective and physical values**

In `training.py`, update objective serialization/validation and add
`smooth_l1_physical_excess(...)` plus
`bounded_physical_envelope_surrogates(...)`. Denormalize prediction/reached
output exactly once. Preserve the old three dimensionless raw risks bit-for-bit,
add direction raw risk using inner bounds, and keep direction runtime failures
on `[.5,1.5]`. Add no CLI overrides.

- [ ] **Step 4: Write RED safe-append and DDP tests**

For each family, append up to 1,023 safe pairs to one offender and require
unchanged unweighted max/tail, weighted value `(max+positive_tail)/256`, and
outside-beta offender gradient `2/256`. Repeat for flattened rollout pair-time.
Require raw diagnostics never change optimized total. Extend the two-rank gloo
fixture to four families and prove single-rank/DDP forward-gradient parity,
lowest-ordinal ties, direction-safe rank append, and metadata-first identical
failure for count/rank/world/dtype/nonfinite/fraction/ordinal mismatch.

- [ ] **Step 5: Implement distinct raw and bounded reductions**

Keep `global_max_plus_tail(...)` unchanged. Make `PhysicalEnvelopeLosses`
carry separate raw and bounded reduction maps. Report raw max/CVaR/positive tail
and active/runtime counts, then compute four bounded reductions and multiply
each complete bounded family once by `reference_pair_coefficient`. Define
`physical_envelope_total` as their sum.

- [ ] **Step 6: Run focused plus legacy raw/DDP tests and commit**

Run all physical-risk, global-tail, safe-append, and two-rank tests. Require old
joint/limit/phase diagnostics unchanged and all new tests green.

```bash
git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "feat: add bounded physical envelope objective"
```

---

### Task 2: Integrate training, direction feasibility, and complete fitted metrics

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`

- [ ] **Step 1: Write RED one-step and rollout tests**

Require both paths to use the same objective/coefficient, four raw and bounded
families, and exact predecessor semantics: one-step and rollout time zero use
sealed predecessor targets; later rollout steps use prior predictions without
detach. Require direction gradient through the model and late recurrence,
time-major ordinal correctness, DDP parity, unchanged base aggregate when safe,
and each bounded family entering total once.

- [ ] **Step 2: Integrate objective v2 without changing optimizer semantics**

Update `one_step_training_losses(...)`, `RolloutResult`,
`autoregressive_unroll(...)`, and metric JSON. Keep Adam, LR, clip norm 1.0,
batch/sampler, and budgets unchanged. Report all raw and bounded values/counts.
Do not divide the rollout's nondilutable pair-time reduction by frames.

- [ ] **Step 3: Add the deterministic shared-trunk optimizer test**

Use the production loss, `torch.optim.Adam`, and
`clip_grad_norm_(...,1.0)` for a fixed 64-update synthetic shared-trunk fixture
containing base targets and all four failures. Repeat exactly and require:

```text
joint-step/native-limit/phase/direction failure counts  each strictly falls
base aggregate and component direction base error       do not increase
loss/gradient/model/Adam state                          finite
repeat trace/state                                      identical
clip path                                               exercised
```

If the authorized objective cannot pass, return to design review; do not tune
the constants or weaken the test.

- [ ] **Step 4: Advance the fitted target audit to v2**

Write RED/green tests for all 12 target direction norms in `[.55,1.45]`, with
equality passing and one ULP outside failing. Record minimum/maximum, lower/
upper excess counts, objective/pair/audit digests. Preserve existing joint,
limit, phase, and negative-phase tolerance fields exactly.

- [ ] **Step 5: Add complete pre/post rollout fitted metrics**

Write RED/green order-invariance and tamper tests for a canonical all-pair
payload containing pair/row/objective digests, four raw max/CVaR/positive-tail/
active/runtime metrics, four bounded max/tail/unweighted/coefficient/weighted
metrics, unchanged fitted v1 maxima/first failure, and payload SHA-256.

Measure all 2,011 pairs after step 4,000 before rollout and after step 4,256.
The pre-rollout result is diagnostic and never early-stops the 256 rollout
updates. Retain both payloads in memory until the final report-v3 write; neither
belongs in checkpoint v7. The final fitted-transition report v1 alone authorizes
evaluation.

- [ ] **Step 6: Run focused integration tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_one_step_v2_adds_four_bounded_families_once \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_rollout_v2_uses_seed_then_prior_prediction \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_shared_trunk_adam_clip_reduces_all_failures_without_regression \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_target_audit_v2_requires_direction_feasibility \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_pre_and_post_rollout_complete_fitted_metrics

git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py
git commit -m "feat: train and measure bounded PFNN envelopes"
```

---

### Task 3: Bind checkpoint v7, report v3, promotion v4, and exact resume

**Files:**
- Modify: `sonic/python/mm_sonic/terrain_pfnn/training.py`
- Modify: `sonic/python/mm_sonic/train_terrain_pfnn.py`
- Modify: `sonic/python/mm_sonic/evaluate_terrain_pfnn.py`
- Modify: `tests/python/test_terrain_pfnn_training.py`
- Modify: `tests/python/test_terrain_pfnn_runtime.py`

- [ ] **Step 1: Write RED v7 resume/tamper tests**

Build a canonical v7 checkpoint with objective v2/digest, target audit v2,
mandatory exact-input observability receipt/digest, and four raw/bounded loss
contracts. Assert its strict key set excludes pre/post fitted metric payloads
and digests. Require safe load and exact resume. Individually tamper objective
constants/coefficient, target audit, observability receipt, pair/fitted receipts,
loss keys, sampler, dataset, normalization, kinematics, seed, model, or Adam and
rehash only the outer candidate; require rejection before model/Adam restore.
Reject v6.

- [ ] **Step 2: Implement checkpoint v7 fail-closed**

Advance `CHECKPOINT_SCHEMA`, strict key sets, save/load dataclasses, and resume
inspection. Recompute active rows, canonical pairs, objective, direction target
audit, mandatory observability receipt, and all checkpoint-owned digests before
mutation. Add no migration bypass or fitted-metric checkpoint field.

- [ ] **Step 3: Write RED report-v3/promotion-v4 tests**

Report v3 binds initial/final fixed metrics and is the exclusive owner of both
complete fitted metric payloads/digests. It also binds the final fitted v1
report/digest, pair/objective/target-audit/observability digests, candidate
hash, and provisional selection. Promotion v4 binds report v3 and thereby both
fitted metric digests, plus exact evaluator receipt, candidate/best hashes, and
permissions. Reject every absent/extra/noncanonical/mismatched/rehashed field
and any attempt to put the complete fitted metrics into checkpoint v7.

Prove the evaluator recomputes post-rollout metrics and fitted v1 exactly before
the first known-terrain callback; every tamper yields callback count zero and no
receipt/best.

- [ ] **Step 4: Implement strict report/evaluator/promotion flow**

Advance `ONE_STEP_REPORT_SCHEMA` to v3 and
`PIPELINE_PROMOTION_RECEIPT_SCHEMA` to v4. Use canonical JSON/atomic writes.
Implement exact save order: compute pre metrics in memory after step 4,000;
run rollout; compute post metrics/fitted v1 in memory; atomically write the
final step-4,256 checkpoint v7 without metrics; safely reload/reproduce/hash it;
construct, validate, and atomically write one final report v3 once. Write no
step-4,000 checkpoint with a partial report and never overwrite report-v3 stage
meaning. Only an accepted 600-tick evaluator may publish mode-`0444` `best.pt`.
Preserve fitted gate v1, evaluation v1, and runtime behavior.

- [ ] **Step 5: Run schema/runtime regressions and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_v7_checkpoint_binds_objective_and_observability_but_excludes_fitted_metrics \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_v7_resume_rejects_objective_observability_and_state_tamper \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_one_step_report_v3_rejects_all_bound_tamper \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_final_save_order_has_one_report_owner_and_no_intermediate_ambiguity \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_pipeline_promotion_v4_binds_report_and_fitted_metrics \
  tests.python.test_terrain_pfnn_runtime.TerrainPFNNRuntimeTests.test_evaluator_validates_v3_before_callback

git add sonic/python/mm_sonic/terrain_pfnn/training.py \
  sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/evaluate_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_runtime.py
git commit -m "feat: bind bounded PFNN artifact schemas"
```

---

### Task 4: Prove source, target, observability, and real gradient gates; then freeze

**Files:**
- Modify before freeze only for RED-backed review fixes: Tasks 1-3 files
- Write ignored evidence: `.superpowers/sdd/bounded-envelope-surrogate-source-report.md`

- [ ] **Step 1: Add and test the per-update production gradient-direction audit**

Use seed-7 production initialization and `torch.autograd.grad` with no optimizer
or parameter mutation. Audit exactly eight independent `[B=32]` production
one-step graphs (first balanced epoch-0 batches, 256 draws, 64/class) and eight
independent production rollout16 `[T=16,B=32]` graphs (first deterministic
sequence-sampler batches). Reset max/tail per graph. Rollout uses sealed
predecessor at time zero, undetached recurrence, and time-major ordinals.

For every graph record `g_base`; each unscaled `h_family`; each scaled
`g_family=h_family/256`; `g_envelope`; `g_total`; family/base and combined/base
dots/cosines; total pre-clip norm; production clip scale
`min(1,1/(||g_total||+1e-6))`; and clipped-total normalized base projection.
Define a cosine as zero when either norm is zero, else the usual dot divided by
the norm product. Recompute clipped projection from total/base dot, base norm,
and clip scale; do not serialize parameter vectors.
Require every graph finite, `||g_base||>0`,
`dot(g_total,g_base) >= .5*||g_base||^2`, each scaled family norm
`<= ||g_base||`, and `||g_envelope|| <= .5*||g_base||`. Bind update/stage,
batch/sequence/ordinal/objective/model digests; model before/after matches,
optimizer steps are zero, and no output/split leakage occurs. Write RED/green
tests for independent resets, both shapes/recurrence, exact metrics, threshold
equality/failure, digest tamper, and fail-closed any-update aggregation.

- [ ] **Step 2: Add and test the mandatory observability audit**

Implement strict `mm-sonic-exact-input-observability-audit/v1`. Group all
2,011 pairs by exact float32 `x` bytes plus phase bits; intersect every
predecessor `[q-.25,q+.25]` interval with native limits per group/joint. Bind
dataset/objective/pair digests, key encoding, group membership/counts, checked
joints, empty count, lowest-key first contradiction or `null`, and audit digest.
Test storage-order invariance and every pair/input/phase/interval/membership/
digest tamper. Any empty intersection stops before v3. Zero empty means only
that this necessary fitted-set contradiction was not found.

- [ ] **Step 3: Add and test the real in-memory shadow Adam gate**

Clone the production-initialized model and run exactly 64 production one-step
updates then 16 production rollout16 updates using batch size 32, production
samplers/LR/Adam/objective and `clip_grad_norm_(...,1.0)`. Use train rows only;
create no artifact/output, training checkpoint, run report, or metrics file.
Before and after, evaluate the identical canonical 2,011-pair population: the
regularization-excluded complete `one_step_score` and raw runtime-failure counts
for joint step, native limit, phase, and direction. Require finite state,
post-score `<=` pre-score, no family count increase, and strictly smaller sum of
counts. Repeat from the same clone and require identical receipt and final
model/Adam digests. Test sampler/update counts, recurrence/ordinals, no-output/
split leakage, inequalities, repeatability, and fail-closed behavior. Document
that this is a short-horizon sanity gate, not convergence or runtime proof.

- [ ] **Step 4: Run focused audit tests, the full suite, and static checks**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_per_update_gradient_audit_uses_eight_independent_one_step_and_rollout_graphs \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_per_update_gradient_audit_records_projection_and_fails_any_threshold \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_mandatory_observability_receipt_is_canonical_and_stops_empty_intersection \
  tests.python.test_terrain_pfnn_training.TerrainPFNNTrainingTests.test_real_shadow_adam_gate_improves_complete_audit_set_without_output
```

Then run the full dependency split:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  /home/ubuntu/miniconda3/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_sources tests.python.test_terrain_pfnn_phase

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m unittest -v \
  tests.python.test_terrain_pfnn_model \
  tests.python.test_terrain_pfnn_features \
  tests.python.test_terrain_pfnn_dataset \
  tests.python.test_terrain_pfnn_kinematics \
  tests.python.test_terrain_pfnn_recurrence \
  tests.python.test_terrain_pfnn_training \
  tests.python.test_terrain_pfnn_runtime

git diff --check
find sonic/python/mm_sonic/terrain_pfnn -maxdepth 1 -name '*.py' -print | sort | \
  xargs sonic/.torch-mm-venv/bin/python -m py_compile
printf '%s\n' sonic/python/mm_sonic/train_terrain_pfnn.py \
  sonic/python/mm_sonic/evaluate_terrain_pfnn.py \
  tests/python/test_terrain_pfnn_training.py \
  tests/python/test_terrain_pfnn_runtime.py | \
  xargs sonic/.torch-mm-venv/bin/python -m py_compile
```

- [ ] **Step 5: Run all real pre-run gates on GPU 1**

On the recurrence-v2 manifest and exact 2,048 rows/2,011 pairs: recompute
direction-aware target audit twice; run the mandatory canonical observability
audit twice; run all 16 independent gradient-direction graphs; run the exact
64-one-step-plus-16-rollout shadow Adam gate twice; then run a separate real-
data no-update/no-output smoke. Require exact receipt equality on repeats, every
per-update gradient inequality, shadow before/after gates, no validation/test
row, and no unauthorized artifact/output. Any failure stops without tuning.

- [ ] **Step 6: Obtain full independent review**

Review the complete correction range against both designs/plans, the v2
artifact root-cause, tests, and source report. Require separate spec-compliance
and code-quality approval. Fix every Critical/Important item with RED coverage,
repeat Steps 4-5, commit, and re-review before freeze.

- [ ] **Step 7: Seal source report and frozen boundary**

Record decisions/rejected alternatives, test commands/counts/exits, target and
mandatory observability receipt, every per-update unscaled/scaled norm/dot/
cosine/preclip/projection and threshold result, shadow update counts and
complete before/after score/failure counts, sampler/objective/model/Adam
digests, repeat receipts, reviewer verdicts, HEAD/status, every source/test and
design/plan/report SHA-256, dataset/model hashes, GPU identity, and UTC. From
this instant, no source/test edit or commit.

---

### Task 5: Execute exactly one fresh v3 run, gated evaluator, and artifact report

**Files:**
- Write: `.superpowers/sdd/bounded-envelope-surrogate-dataset-before.json`
- Write: `.superpowers/sdd/bounded-envelope-surrogate-dataset-after.json`
- Create only by authorized command:
  `sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v3/`
- Write: `.superpowers/sdd/bounded-envelope-surrogate-artifact-report.md`
- Do not modify source/tests or commit.

- [ ] **Step 1: Snapshot and resume the exact dataset once**

Write snapshot v1 path/SHA-256/size/`mtime_ns` entries for the exact six files
without test deserialization. Run exact dataset `--resume` once; write the after
snapshot and require identical entries/canonical digest and dataset digest
`9e304bafe780aa694467e01912b7a04a67a0526d3c81a612a9618658ba4bdd48`.
On failure, stop without rerunning resume.

- [ ] **Step 2: Reconfirm the frozen launch boundary**

Require frozen HEAD/hashes, both snapshots, v1/v2 metadata and hashes, absent
v3 output, physical GPU 1, no train/evaluate process, accepted target and
mandatory observability receipts, all 16 per-update gradient-direction results,
both identical shadow-gate receipts, and UTC. Never delete or overwrite an
existing v3 path.

- [ ] **Step 3: Launch the sole `4000 + 256` command once**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.train_terrain_pfnn \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --output sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v3 \
  --overfit-samples 2048 --steps 4000 --seed 7 \
  --rollout-finetune-frames 16 --rollout-finetune-steps 256
```

This launch consumes authorization. Record PID, exact argv/environment, UTC,
and step/loss/gradient progress about every two minutes. Never retry/resume.

- [ ] **Step 4: Enforce fitted acceptance before evaluation**

Require finite training, v7 candidate, safe reload, exact fixed reproduction,
exact objective/pair/target/observability bindings, report-v3-only pre/post
metrics, and accepted 2,011-pair fitted v1. Assert checkpoint v7 contains no
pre/post payload/digest and the report was written once in the exact save order.
At first failure, preserve all candidates/reports/prior outputs, record first/
worst pair/field/joint/value/limit, distributions, exposures, gradient trends,
hashes, and process health, then stop with no evaluator/edit/commit/patch.

- [ ] **Step 5: Only after acceptance, run the exact evaluator once**

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=sonic/python \
  sonic/.torch-mm-venv/bin/python -m mm_sonic.evaluate_terrain_pfnn \
  --checkpoint sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v3/checkpoint-step-00004256.pt \
  --dataset sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json \
  --model-path /home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/assets/skeletons/g1/g1_29dof.xml \
  --split train --closed-loop-seconds 20 --promote-pipeline-best
```

Require 600 finite committed ticks, every unchanged gate, supported known-
terrain traversal, exact recomputation, promotion-v4, and mode-`0444` best.
First evaluator failure stops without a second launch.

- [ ] **Step 6: Reload, posttest, and finalize evidence**

Only on promotion, safely reload candidate/best and verify hashes, permissions,
bindings, and no test-derived state. Re-run Task 4's full tests/static checks
against frozen hashes. Finalize the artifact report with command count/argv,
UTC/exits, complete metrics and pre/post deltas, receipts, candidate/metrics/
report/promotion/best hashes, evaluator outcome, reload/posttests, prior-artifact
preservation, decisions, concerns, and evidence-file hashes.

---

### Task 6: Re-review original Task 7, then resume original Tasks 8/9 and final drive

**Files:**
- Review: the complete frozen correction range and all design/plan/report files
- Then use only already-approved original Tasks 8/9 scopes; do not alter the
  accepted bounded-envelope checkpoint or frozen correction.

- [ ] **Step 1: Independently re-review original Task 7**

Review recurrence, physical-envelope, startup-metrics, and bounded-surrogate
designs/plans/source/artifacts over the full frozen range. Require separate
spec-compliance and code-quality verdicts with no Critical/Important finding.
A material finding stops without changing frozen correction code.

- [ ] **Step 2: Resume original Task 8 only after blocker-free review**

Use the accepted immutable best through the approved viewer/full-corpus path,
preserving all objective/report/promotion bindings. Never substitute v1/v2 or
reopen v3 training.

- [ ] **Step 3: Execute original Task 9 and final drive**

Use the sealed test split exactly once under its original authorization. Finish
the required interactive W/A/S/D drive over three hills with visual/runtime
evidence. Report exact accepted hashes, full report chain, tests/static status,
review verdicts, final decisions, and every residual concern.
