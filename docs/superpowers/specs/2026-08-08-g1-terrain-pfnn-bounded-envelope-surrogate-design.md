# G1 Terrain PFNN Bounded-Envelope Surrogate Correction

## Status

Approved narrow correction design. This document supplements the physical-
envelope objective and startup-metrics correction designs. It does not reopen
the accepted recurrence, dataset, runtime, fitted-gate, or controller
contracts.

## Evidence and causal question

The preserved physical-envelope-v2 attempt completed its sole authorized
4,000 one-step plus 256 rollout16 updates. Its fixed-sample score improved from
`8.309912952530794` to `2.608495159148788`, a ratio of
`0.3139016225620471`, and the v6 checkpoint was finite, safely reloadable, and
exactly reproducible. The complete fitted gate nevertheless rejected it:

- 1,588 of 2,011 pairs exceeded the unchanged `0.25 rad` joint-step limit;
- 1,048 pairs violated the unchanged trajectory-direction norm gate;
- 283 pairs violated the unchanged phase envelope;
- only 195 pairs passed every fitted gate; and
- native limits, root translation, root rotation, and root height did not fail.

Coverage was complete, not starved. The one-step sampler made 128,000 pair
draws, reached all 2,011 pairs at least 40 times, and balanced all four terrain
classes to exactly 32,000 draws each. Rollout optimization covered 131,072
pair-time elements.

The failure is instead consistent with two objective defects:

1. The three optimized envelope families used dimensionless squared excesses,
   a nondilutable `maximum + positive_tail_mean`, and outer weight `1.0`
   alongside base losses that are means over batch rows and output elements.
   Every one of the 4,256 recorded gradients was clipped at norm `1.0`; the
   pre-clip means were `88,837.02807226563` for one-step and
   `102,273.87219238281` for rollout, with maxima above one million. The
   envelope therefore controlled the clipped update direction without a
   commensurate base-loss scale.
2. Trajectory-direction norm was an unchanged fitted/runtime authorization
   gate but had no predecessor-aware envelope family. The v2 artifact left
   1,048 direction failures, including 226 pairs whose first fitted failure
   was direction after earlier gate families passed.

The causal question for this cycle is narrow: can a physically scaled,
gradient-bounded four-family surrogate make the same model and exact fitted
population pass the unchanged authorization gate without degrading the base
fit?

## Goals

- Keep the raw v1 physical-envelope risks, maxima, CVaR, counts, and runtime
  diagnostics exactly available for longitudinal comparison.
- Optimize four physical-unit surrogates: joint step, native joint limit,
  phase advance, and trajectory-direction norm.
- Bound each surrogate derivative before the existing global max-plus-positive-
  tail reduction and scale every optimized family by the preregistered
  candidate coefficient `1 / 256`.
- Measure the complete 2,011-pair fitted population immediately before and
  after rollout fine-tuning so rollout regressions cannot hide behind one final
  report.
- Bind every new constant, metric, checkpoint, report, resume, evaluator, and
  promotion decision cryptographically and validate them before restoration or
  callback.
- Authorize at most one fresh physical-envelope-v3 artifact attempt after
  tests, real-data gradient/no-update gates, and independent review pass.

## Non-goals and immutable boundaries

- Model and raw layouts remain `288 -> 512 -> 512 -> 268`.
- Dataset remains `mm-sonic-terrain-pfnn-dataset/v2`; no shard is rebuilt for
  this correction.
- Canonical pair construction, fitted subset, sampler, recurrence, rollout,
  normalization, kinematics, controller, and raw output semantics are
  unchanged.
- The fitted-transition report and gate remain
  `mm-sonic-fitted-transition-report/v1` with exactly the existing thresholds
  and precedence.
- Runtime limits remain root translation `<= 0.060 m`, root rotation
  `<= 0.35 rad`, joint step `<= 0.25 rad`, native joint limits, positive root
  height, finite normalized root quaternion, trajectory-direction norms in
  `[0.5, 1.5]`, and phase advance in
  `[0, min(pi, 1.5 * training_phase_q99)]`.
- No clamp, mask, residual reconstruction, output repair, IK, stance lock,
  root projection, threshold relaxation, or artifact patch is allowed.
- Failed v1 and rejected v2 directories remain byte-for-byte preserved. No
  checkpoint from either is resumed.
- Validation remains selection-only and the sealed test split remains closed
  until original Task 9.

## Alternatives considered

### Guessing raw loss weights, learning rate, or update budget — rejected

Reducing the existing envelope weights, lowering the learning rate, or adding
updates could make the observed gradient norm smaller, but each is an
unidentified fitted hyperparameter. The v2 evidence does not identify a magic
weight or budget, all updates were already clipped, and another guess would
spend the sole artifact comparison without establishing a stable scale. The
selected coefficient is instead fixed before v3 as a testable candidate and
must pass per-update gradient-direction and shadow-optimizer gates on real
training data.

### Expanding the PFNN input — deferred pending observability evidence

Appending predecessor joints could make predecessor-relative safety directly
observable, but it changes dataset rows, model shape, checkpoint compatibility,
runtime packing, recurrence, and every trained artifact. Complete v2 exposure
does not by itself prove that the existing input is insufficient. The mandatory
exact-input audit stops the cycle before training if it finds one exact
`(x, phase)` group whose required safe joint intervals have empty intersection.
Absent that proof, input expansion remains a later architectural branch, not
part of this correction.

### Manual PCGrad or family-specific optimizer projection — rejected

PCGrad could reduce destructive gradient interference, but a hand-written
multi-objective optimizer would alter update semantics, Adam interaction, DDP
equivalence, resume state, and testing scope. The evidence first requires
removing a known scale mismatch and missing family. PCGrad is too invasive
until a bounded surrogate with audited gradients is shown insufficient.

### Physical smooth-L1 bounded surrogate — selected

Smooth-L1 keeps a quadratic basin near each training boundary, has a derivative
whose magnitude saturates at one in the physical unit, and remains interpretable
without fitting a weight from the failed artifact. Applying the existing
nondilutable reduction after a per-pair worst-case reduction preserves rare
failure pressure. Multiplying each family by `1/256` applies one preregistered
population-scale candidate before the unchanged optimizer and clip. The
coefficient is not an algebraic consequence of the two different reductions;
the real-data gates below must falsify or admit it before training.

## Objective v2 contract

Replace the v1 objective payload with a strict immutable v2 payload. Existing
raw-risk fields retain their exact values and meanings. The added fields are:

```python
@dataclass(frozen=True)
class PhysicalEnvelopeObjective:
    schema: str = "mm-sonic-physical-envelope-objective/v2"

    # Existing raw diagnostics; unchanged.
    joint_step_onset_rad: float = 0.225
    joint_step_scale_rad: float = 0.025
    joint_limit_margin_rad: float = 0.020
    phase_upper_margin_rad: float = 0.020
    phase_scale_rad: float = 0.020
    tail_fraction: float = 0.10
    joint_step_hinge_power: int = 2
    joint_limit_hinge_power: int = 2
    phase_lower_hinge_power: int = 1
    phase_upper_hinge_power: int = 2
    maximum_coefficient: float = 1.0
    positive_tail_mean_coefficient: float = 1.0

    # New direction diagnostic and bounded training surrogate.
    direction_norm_inner_lower: float = 0.55
    direction_norm_inner_upper: float = 1.45
    direction_norm_scale: float = 0.05
    direction_hinge_power: int = 2
    joint_step_smooth_l1_beta_rad: float = 0.025
    joint_limit_smooth_l1_beta_rad: float = 0.020
    phase_upper_smooth_l1_beta_rad: float = 0.020
    direction_smooth_l1_beta: float = 0.05
    reference_pair_count: int = 256
    reference_pair_coefficient: float = 0.00390625
```

Validation requires exact field names, exact Python scalar types, finite values,
all positive scales/betas, `0 < tail_fraction <= 1`, ordered direction bounds,
`reference_pair_count == 256`, and exact equality
`reference_pair_coefficient == 1.0 / reference_pair_count`. Callers may not
override any default in the controlled pipeline run.

### Candidate-coefficient rationale and falsification

`1/256` is a preregistered candidate coefficient, not a fitted loss weight and
not an algebraic mean identity. Base losses are means over rows and output
components, whereas maximum-plus-positive-tail selects extreme pair risks and
has different selection and Jacobian semantics. No cancellation law says that
the latter must be divided by a batch or reference-population count.

The candidate is motivated only at order-of-magnitude level: the safety
reduction is deliberately nondilutable, the production reference set contains
256 one-step draws, and v2 produced pre-clip norms from `6.05e3` through
`1.28e6` while every update clipped at `1.0`. Fixing `1/256` before v3 gives a
specific hypothesis that can be rejected rather than a coefficient selected
after observing a fitted result. The per-update gradient-direction audit and
real in-memory shadow Adam gate below decide whether it is admissible. Any
failure stops this cycle; it does not authorize coefficient fitting.

## Raw diagnostics remain comparable

For joint step, native limit, and phase, retain the v1 dimensionless risks
bit-for-bit:

```text
raw_step_i  = max_j((relu(abs(q_pred_ij - q_prev_ij) - 0.225) / 0.025)^2)
raw_limit_i = max_j(max(
    relu((lower_j + 0.020) - q_pred_ij) / 0.020,
    relu(q_pred_ij - (upper_j - 0.020)) / 0.020
)^2)
raw_phase_i = max(
    relu(-phase_i) / 0.020,
    (relu(phase_i - (phase_cap - 0.020)) / 0.020)^2
)
```

Add a direction raw diagnostic without changing the runtime gate:

```text
direction_norm_ik = norm(direction_ik, 2)
direction_excess_ik = max(
    relu(0.55 - direction_norm_ik),
    relu(direction_norm_ik - 1.45)
)
raw_direction_i = max_k((direction_excess_ik / 0.05)^2)
```

For all four raw risks, continue to report deterministic global maximum,
ordinary worst-10% CVaR, selected-positive tail mean, objective-active count,
and runtime-failure count. Runtime failure counts use `[0.5, 1.5]` for
direction, not the inner training bounds. Raw diagnostics are excluded from
the optimized total in v2; their role is audit and longitudinal comparison.

## Exact physical-unit surrogate

For nonnegative excess `z` and positive `beta`, define:

```text
smooth_l1_beta(z) = 0.5 * z^2 / beta    when 0 <= z < beta
                    z - 0.5 * beta      when z >= beta
```

It is exactly zero with zero gradient at `z == 0`, is continuously
differentiable at `z == beta`, and its derivative with respect to `z` is in
`[0, 1]`. PyTorch computations remain in the prediction dtype; physical
denormalization happens exactly once.

Per pair `i`:

```text
step_excess_ij = relu(abs(q_pred_ij - q_prev_ij) - 0.225)
step_surrogate_i = max_j(smooth_l1_0.025(step_excess_ij))

limit_excess_ij = max(
    relu((lower_j + 0.020) - q_pred_ij),
    relu(q_pred_ij - (upper_j - 0.020))
)
limit_surrogate_i = max_j(smooth_l1_0.020(limit_excess_ij))

phase_low_surrogate_i = relu(-phase_i)
phase_high_excess_i = relu(phase_i - (phase_cap - 0.020))
phase_surrogate_i = max(
    phase_low_surrogate_i,
    smooth_l1_0.020(phase_high_excess_i)
)

direction_excess_ik = max(
    relu(0.55 - norm(direction_ik, 2)),
    relu(norm(direction_ik, 2) - 1.45)
)
direction_surrogate_i = max_k(smooth_l1_0.05(direction_excess_ik))
```

The lower phase branch remains linear in physical radians so every strictly
negative prediction receives a nonvanishing corrective derivative. Upper
phase, joint, native-limit, and direction derivatives saturate at magnitude
one before reduction. At an exact direction vector of zero, PyTorch's finite
zero norm subgradient is accepted; the existing component-wise direction base
loss supplies target-directed gradient, and the shared-trunk optimizer test
must prove direction failures still decrease.

## Reduction and optimized total

For each family independently, reduce its vector of one physical surrogate per
pair (or pair-time element during rollout) with the already-reviewed exact
global reduction:

```text
k = max(1, ceil(0.10 * N))
maximum = deterministic lowest-global-ordinal maximum
cvar = mean(the k largest values)
active = selected top-k values strictly greater than zero
positive_tail_mean = mean(active), or differentiable zero
surrogate_envelope = maximum + positive_tail_mean
optimized_family = (1 / 256) * surrogate_envelope
```

Safe append, global ordinal, tie, and DDP semantics remain unchanged. A single
selected scalar may receive both max and positive-tail derivatives, so its
surrogate-scalar derivative is bounded by `2/256 = 0.0078125` per family
before shared-network Jacobians. No family is combined with another before its
own worst-element and global reductions.

The optimized total becomes:

```text
total = base_total
      + joint_step_bounded_envelope
      + joint_limit_bounded_envelope
      + phase_bounded_envelope
      + direction_bounded_envelope
```

Each named bounded envelope already includes `1/256`. The existing raw
`joint_step_envelope`, `joint_limit_envelope`, and `phase_envelope` names remain
diagnostic and are not added again. Add `direction_envelope` as the raw
direction diagnostic and use distinct `*_bounded_envelope` names for optimized
families. This prevents a report reader or resume validator from confusing a
dimensionless diagnostic with a physical-unit training term.

## One-step and rollout data flow

One-step training continues to consume exact canonical
`(predecessor_target_y, current_x, current_phase, current_target_y)` pairs.
Rollout step zero consumes its sealed predecessor target; later steps consume
the prior prediction without detach. The new direction surrogate reads the 12
two-component trajectory-direction vectors already present in the unchanged
268-value output. It requires no new input or target field.

The exact same objective v2 instance, pair ordinals, tail fraction, coefficient,
and family names apply to one-step and rollout. Rollout counts remain pair-time
elements in `[T, B]`; they are never divided by `T` after the global reduction.

## Complete fitted measurements before and after rollout

After step 4,000 and before any rollout update, evaluate the safely held model
over all 2,011 fitted pairs and seal `pre_rollout_fitted_metrics`. After the
256th rollout update, repeat the same computation and seal
`post_rollout_fitted_metrics`. Each payload contains:

- pair count and pair/report digests;
- all unchanged raw maxima/CVaR/positive-tail/objective-active/runtime-failure
  values for joint step, native limit, and phase;
- the new raw direction diagnostics;
- all four physical surrogate maximum/tail/envelope/weighted values;
- unchanged fitted physical maxima and first failure; and
- a canonical payload SHA-256.

These measurements are diagnostics, not intermediate authorization. A
pre-rollout failure does not stop the prescribed 256 rollout updates. The final
post-rollout fitted-transition report v1 alone controls whether the evaluator
may run. Recording both snapshots makes a rollout regression causal and
visible. They have one owner: strict one-step report v3. Checkpoint v7 never
contains either fitted-metric payload or digest.

The write order is exact and has no intermediate representation:

1. After optimizer step 4,000, compute and retain the canonical pre-rollout
   payload in memory; write no checkpoint or report for this measurement.
2. Run exactly 256 rollout16 updates, then compute and retain the canonical
   post-rollout payload and final fitted-transition report v1 in memory.
3. Atomically write the final step-4,256 checkpoint v7 without either metric
   payload, safely reload it, reproduce its fixed score, and compute its hash.
4. Construct one final report-v3 object containing both stage-labelled metric
   payloads/digests, the fitted report, and candidate hash; validate it and
   atomically write `one-step-report.json` once.
5. If the fitted report rejects, exit after preserving checkpoint and report.
   If it accepts, the evaluator consumes that immutable checkpoint/report pair.

There is no step-4,000 checkpoint carrying a partial report, no provisional
report-v3 file, and no later overwrite that changes the meaning of a stage.

## Schema and provenance evolution

- Checkpoint advances to `mm-sonic-terrain-pfnn-checkpoint/v7`.
- Physical objective advances to
  `mm-sonic-physical-envelope-objective/v2`.
- One-step report advances to `mm-sonic-terrain-pfnn-one-step-report/v3`.
- Pipeline promotion receipt advances to
  `mm-sonic-pipeline-promotion-receipt/v4`.
- The fitted target audit advances to
  `mm-sonic-fitted-target-envelope-audit/v2` only because its strict field set
  must bind direction inner bounds, minimum/maximum target direction norms,
  and lower/upper target excess counts. Its existing joint/limit/phase values
  and tolerance remain unchanged.
- Dataset remains v2, compact pair receipt remains v1, fitted-transition gate
  remains v1, snapshot remains v1, and runtime/evaluation semantics remain
  unchanged.

Checkpoint v7 binds the exact objective v2 payload/digest, target audit v2,
mandatory observability receipt/digest, pair receipt, fitted subset,
normalization, kinematics, sampler states, seed, model, and Adam state. It
explicitly excludes pre/post complete fitted metrics. A v6 checkpoint is never
accepted as a v7 resume. Safe resume recomputes active rows, full canonical
pairs, objective, direction-aware target feasibility, mandatory observability,
and all checkpoint-owned digests before model or optimizer restoration.

Report v3 exclusively owns and binds both pre/post fitted metric payloads and
digests. It also binds the final fitted-transition report v1 and digest, exact
objective, pair, target-audit, and mandatory-observability digests,
initial/final fixed metrics, candidate hash, and provisional selection. The
evaluator recomputes and requires the post-rollout metrics and final fitted
report exactly before any known-terrain callback.

Promotion v4 additionally binds the report-v3 digest and both fitted-metric
digests. It may publish immutable mode-`0444` `best.pt` only after exact fixed
reproduction, accepted fitted report, all 600 finite ticks, every unchanged
physical/contact gate, and supported known-terrain traversal.

## Target feasibility and mandatory observability audit

The required target audit recomputes all 2,011 train-only fitted pairs. Existing
joint, limit, and phase criteria remain exact. For each of the 12 target
direction vectors, require finite norm in the inner training interval
`[0.55, 1.45]`; equality passes. Record minimum, maximum, lower-excess count,
upper-excess count, objective digest, pair digest, and audit digest. Any target
direction excess stops before training.

The exact-input observability audit is mandatory before source freeze. Group
all 2,011 fitted pairs by the exact byte identity of current normalized
float32 `x` plus the exact float32 phase bits—the complete deterministic PFNN
input. For each group and joint, intersect every predecessor-relative
runtime-safe interval `[q_prev - 0.25, q_prev + 0.25]` with that joint's native
limits. If any exact-input group has an empty intersection, the unchanged input
cannot produce one deterministic joint output safe for every member. Stop
before v3; input expansion becomes the next design branch.

The canonical receipt schema is
`mm-sonic-exact-input-observability-audit/v1`. It binds dataset digest, pair
receipt/digest, objective digest, exact input-key encoding, group count,
largest-group count, canonical group-membership digest, checked joint count,
empty-intersection count, the lowest-canonical-key first contradiction or
`null`, and its own audit SHA-256. A passing receipt requires exact int zero
empty intersections and `first_contradiction is None`; it is embedded by value
and digest in checkpoint v7 and report v3. Reordering source storage must not
change it, while any pair, predecessor, input byte, phase bit, interval,
membership, or digest tamper must fail closed.

A zero-empty result establishes only that this specific necessary interval
condition found no exact-input contradiction in the sealed fitted population.
It does not prove general observability, identifiability, learnability, or
runtime safety outside those rows.

## Required verification before a run

### Unit and distributed TDD

Tests must cover:

- exact smooth-L1 values and derivatives below, at, and above every beta;
- exact zero loss/gradient at all inner boundaries and finite capped gradients
  immediately outside them;
- lower phase's linear derivative for every strict negative;
- worst-joint/worst-knot before global reduction;
- safe append nondilution for forward value and offender gradient;
- direction lower/upper boundaries, runtime-vs-training bands, zero-vector
  finiteness, and raw-vs-bounded separation;
- four-family global ordinal ties and positive-tail semantics;
- DDP metadata-first rejection and single-rank/DDP forward-gradient parity;
- one-step and rollout use of the same objective and `1/256` coefficient;
- report-v3-only ownership of complete pre/post rollout fitted metrics, exact
  final save order, and rejection of partial/intermediate ownership;
- canonical mandatory observability receipts, storage-order invariance, empty-
  intersection stopping, and tamper rejection;
- per-update one-step/rollout gradient norms, dot products, cosines, clipping
  projection, threshold boundaries, and fail-closed aggregation;
- deterministic real train-only shadow Adam/clip state, sampler, before/after
  score/count gates, repeatability, and no-output behavior;
- v7 safe load/resume and rejection of every objective/audit/report/promotion
  tamper before model/Adam restore or callback; and
- preservation of existing base losses, fixed metric schema, recurrence,
  runtime thresholds, and fitted gate v1.

A deterministic shared-trunk synthetic Adam test must execute the real
optimizer and `clip_grad_norm_(..., 1.0)` path. Its batch contains base targets
and failures in all four surrogate families. Across its fixed update count it
must show strictly fewer joint-step, native-limit, phase, and direction
failures; finite state throughout; no increase in the base aggregate; no
increase in component-wise direction base error; and identical results on a
repeat run. This test is the local guard against a mathematically bounded
family still monopolizing a shared trunk after clipping.

### Per-update production gradient-direction gate

Use GPU 1, seed 7, the exact recurrence-v2 fitted subset, and the production
`512`-hidden initialization without constructing an optimizer or changing a
parameter. Audit exactly 16 independent production-shaped graphs:

- the first eight deterministic balanced epoch-0 one-step batches, each a
  separate `[B=32]` graph, covering 256 draws and exactly 64 draws per terrain
  class; and
- the first eight deterministic production sequence-sampler rollout batches,
  each a separate `[T=16,B=32]` graph whose time zero uses the sealed
  predecessor, later recurrent inputs remain undetached, and pair-time global
  ordinals are time-major.

Maximum/tail selection is reset inside every graph. Never concatenate the
eight one-step batches into one 256-row reduction and never carry a selected
maximum or tail between updates. For every one-step and rollout graph, use
`torch.autograd.grad` on the same forward graph and define the flattened
parameter gradients:

```text
g_base       = grad(base_total)
h_family     = grad(unscaled_family_envelope)
g_family     = (1/256) * h_family
g_envelope   = sum(g_family for four families)
g_total      = g_base + g_envelope
clip_scale   = min(1, 1 / (||g_total||_2 + 1e-6))
g_clipped    = clip_scale * g_total
```

Record per update and per family the unscaled and scaled norms,
`dot(h_family,g_base)`, `cos(h_family,g_base)`,
`dot(g_family,g_base)`, and `cos(g_family,g_base)`. Also record base norm,
combined envelope norm and base dot/cosine, total pre-clip norm and base
dot/cosine, clip scale, clipped-total norm, and the normalized clipped base
projection `dot(g_clipped,g_base) / ||g_base||^2`. Bind stage, local update
index, batch/sequence receipts, global ordinals, objective, model-before/after,
and the complete audit payload digest.

For every recorded cosine, use exact canonical zero when either vector norm is
zero; otherwise use `dot(a,b) / (||a|| * ||b||)`. The clipped projection is
recomputed from the recorded total/base dot, base norm, and clip scale as
`clip_scale * dot(g_total,g_base) / ||g_base||^2`; no full parameter vector is
serialized into the receipt.

Every one of the 16 updates must satisfy, without averaging away a failure:

```text
all losses, gradient entries, norms, dots, cosines, and projections finite
||g_base|| > 0
dot(g_total, g_base) >= 0.5 * ||g_base||^2
each ||g_family|| <= ||g_base||
||g_envelope|| <= 0.5 * ||g_base||
```

The recorded clipped projection must equal the value recomputed from those
recorded scalars. Any graph or receipt failure stops before
training; it does not authorize coefficient, beta, learning-rate, or budget
tuning. Model before/after digests must match, `optimizer_steps` is exact int
zero, no output directory is created, and no validation/test row is loaded.

### Deterministic real train-only shadow optimizer gate

After the gradient-direction gate passes, clone the same production-initialized
model entirely in memory. On that clone only, run exactly 64 production
one-step Adam updates followed by exactly 16 production rollout16 Adam updates,
using seed 7, batch size 32, the production balanced pair sampler and sequence
sampler, production learning rate, the selected four-family objective, and
`clip_grad_norm_(..., 1.0)`. Rollout time zero uses its sealed predecessor,
later recurrence remains undetached, and ordinals remain time-major. Run the
whole shadow gate twice from the same cloned state and require identical
receipts and final model/Adam digests.

The audit population is the complete canonical 2,011-pair train-only fitted
population. Immediately before the first shadow update and after the final
rollout update, evaluate exactly the same ordered population. The base score is
the existing regularization-excluded `one_step_score` over all 2,011 current
rows. The four raw failure counts are the sums of the unchanged runtime-failure
masks for joint step, native limit, phase, and direction over those pairs.
Require all model, Adam, loss, gradient, score, and count values finite;
post-score `<=` pre-score; no family failure count increases; and the sum of
the four post counts is strictly smaller than the sum of the four pre counts.

This is a deterministic sanity gate for short-horizon optimizer interaction,
not proof that the 4,256-update artifact will converge or pass closed loop. It
creates no artifact/output path, writes no training checkpoint, run report, or
metrics file, loads no validation/test row, and is discarded after the
source-gate controller records its canonical digest-bound receipt in the source
evidence report. Any failure stops before the controlled run.

### Full source gate and review

Run every dependency-split PFNN test, all static checks, the complete fitted
target audit including direction, the mandatory canonical observability audit,
the per-update one-step/rollout gradient-direction audit, the deterministic
real shadow optimizer gate, and a separate real-data no-update/no-output smoke.
Obtain independent spec-compliance and code-quality approval over both the
narrow and full correction ranges. Freeze exact HEAD, every source/test hash,
all audit receipts, dataset/model hashes, GPU assignment, and UTC boundary.

## Controlled v3 artifact and stop rules

After the freeze, preserve v1 and v2 exactly and take fresh before/after
snapshots around the exact dataset `--resume`. Require the same six files and
dataset digest. Confirm GPU 1 and absence of:

```text
sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v3
```

Authorize exactly one fresh seed-7 command with the unchanged 4,000 one-step
and 256 rollout16 budgets. Require finite losses, v7 candidate, safe reload,
exact fixed-sample reproduction, exact objective/pair/audit/pre/post receipts,
and an accepted 2,011-pair fitted report before launching the one exact
600-tick evaluator. At the first startup, training, gradient, reload,
reproduction, fitted, or closed-loop failure, record exact evidence and stop:
no retry, resume, parameter change, output patch, source/test edit, or commit.

Only a fully accepted evaluator may create promotion-v4 and immutable
`best.pt`. Then safely reload candidate/best, verify identical hashes and all
bindings, run post-artifact tests/static checks, and write the artifact report.

## Downstream handoff

Only after full v3 acceptance, independently re-review original Task 7 across
its full frozen range, including the physical-envelope, startup-metrics, and
bounded-surrogate designs/plans and all source/artifact reports. Any Critical
or Important finding stops without source change. A blocker-free review resumes
original Tasks 8 and 9 exactly and ends with the required interactive W/A/S/D
three-hill drive and final evidence. No downstream task may alter the accepted
checkpoint or reopen the sealed test before its single authorized use.
