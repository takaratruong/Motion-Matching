# G1 Terrain PFNN Physical-Envelope Objective Correction

## Status

Proposed design amendment for the native G1 terrain-PFNN milestone. It is
based on the one authorized recurrence-v2 controlled experiment recorded in
`.superpowers/sdd/recurrence-correction-task-5-report.md`. It supplements the
original terrain-PFNN design and recurrence-correction design; it does not
replace either document.

## Evidence and problem statement

The recurrence correction did what it was intended to do:

- training and runtime share the trajectory recurrence;
- stationary augmentation uses explicit phase lanes;
- deterministic rollout sampling covers sequences before replacement; and
- the physical fitted-transition gate blocks unsafe candidates before any
  terrain callback or immutable-best publication.

The single `4,000 + 256` recurrence-v2 experiment nevertheless failed that
gate. The fixed-sample score improved from `8.309912952530794` to
`0.20934492463685572`, but 130 of 2,011 adjacent fitted pairs violated at least
one physical envelope:

- 104 pairs violated the `0.25 rad` joint-step limit;
- 27 pairs violated the phase-advance envelope; and
- one pair violated both.

The first joint failure was a `0.2999461625885491 rad` left-knee step at
`terrain_slopes__slope_000__000/motion/85`. The worst was a
`0.5527644072841458 rad` left-hip-yaw step at
`terrain_slopes__slope_001__001/motion/195`.

Both pairs were fitted and repeatedly sampled. The first pair received 54/53
one-step predecessor/current draws and 34 rollout exposures; the worst
received 55/50 draws and 72 rollout exposures. Both remained unsafe when the
model input was regenerated through the intended shared recurrence. The
evidence therefore does not support another representation rewrite, a simple
coverage-starvation hypothesis, or an output correction. It supports making
the already-authoritative physical envelope visible to optimization.

## Goals

- Optimize predecessor-relative joint step, native joint limits, and bounded
  phase advance using train-only information.
- Apply the same objective to one-step and differentiable rollout training.
- Prevent rare unsafe pairs from disappearing inside a global mean over rows,
  joints, or rollout time.
- Preserve gradients through prior rollout predictions.
- Preserve deterministic sampling, exact resume, DDP equivalence, provenance,
  safe loading, and fail-closed promotion.
- Authorize exactly one new controlled recurrence-v2 training run after the
  implementation and all contract tests pass.

## Non-goals and immutable constraints

- The raw `288 -> 512 -> 512 -> 268` PFNN and its output layout remain
  unchanged.
- Runtime, recurrence, controller, command semantics, terrain queries,
  normalization, raw-output interpretation, and physical gate order remain
  unchanged.
- Runtime limits remain exactly: root translation `<= 0.060 m`, root rotation
  `<= 0.35 rad`, every joint step `<= 0.25 rad`, every joint inside native
  limits, positive root height, finite normalized quaternion, trajectory
  direction norms in `[0.5, 1.5]`, and phase advance in
  `[0, min(pi, 1.5*q99)]`.
- No output clamp, masking, residual correction, qpos correction, IK, stance
  lock, root projection, portal, MotionBricks call, or threshold relaxation is
  allowed.
- Dataset, validation, and sealed-test isolation remain unchanged. The sealed
  test split stays closed until original Task 9.
- The failed recurrence-v2 candidate remains preserved and is never resumed or
  overwritten.

## Alternatives considered

### Per-element mean hinge — rejected

A squared hinge averaged over all `batch * 29` joint values is simple, but it
recreates the failure mode seen in Task 5. One unsafe joint is divided by 29
before safe rows further dilute it. A mean is useful for accuracy, not for a
rare physical maximum that controls authorization.

### Strict batch maximum alone — rejected

Taking only the maximum joint/row/time violation exactly matches worst-case
authorization, but ordinarily only one element receives gradient. The active
element can switch between rows and joints under dropout and balanced terrain
sampling, and a max alone discards useful signal from the other 129 observed
failing pairs. The selected reduction retains a global maximum as a mandatory
nondilutable term and adds a tail term.

### Deterministic global maximum plus selected-positive tail mean — selected

First reduce each constraint family to one risk per physical pair by taking
the worst joint where applicable. Compute both the global maximum and the
standard empirical CVaR of the globally worst 10%. A literal
`maximum + CVaR` still decreases for a lone offender when enough zero-risk rows
are appended, because `ceil(0.10*N)` grows and the CVaR denominator admits more
zeros. An unnormalized top-tail sum avoids that dilution but scales with
batch/time size and can dominate fixed gradient clipping during rollout. The
reviewed scale-stable max-plus-tail form is therefore:

```text
k               = max(1, ceil(0.10 * N))
CVaR_0.10        = sum(the k greatest risks) / k
active_tail      = selected top-k risks that are strictly greater than zero
a                = max(1, count(active_tail))
positive_tail_mean = sum(active_tail) / a, or differentiable zero if empty
family_envelope  = deterministic_global_max(risk) + positive_tail_mean
```

CVaR remains computed and reported, while the optimized tail mean excludes
selected zeros. With one positive offender, appending arbitrary zero-risk
pairs, rollout times, or ranks cannot reduce either loss term or the offender's
gradient, and the tail term remains scale-stable. The observed
unsafe-pair prevalence is `130/2011 = 6.4644%`, so a 10% tail covers every
currently unsafe pair plus nearby boundary cases only when reducing the
complete 2,011-pair fitted population. A training minibatch need not contain
every offender; within each minibatch the deterministic global maximum is the
nondilutable protection, while deterministic sampler coverage supplies later
exposure.

### Input expansion or residual output — rejected

Appending predecessor joints would change the frozen 288-value input and every
dataset/checkpoint/runtime contract. Predicting a residual would change the
268-value raw-output meaning and invite a runtime reconstruction path. Task 5
does not show missing predecessor information: the shared recurrence already
carries predicted body state, and both audited pairs fail from sealed and
recurrent inputs. These architecture changes are broader than the evidence.

## Exact objective contract

Add the immutable contract:

```python
@dataclass(frozen=True)
class PhysicalEnvelopeObjective:
    schema: str = "mm-sonic-physical-envelope-objective/v1"
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
```

The positive margins are training targets, not runtime changes:

- joint-step penalty begins at `0.225 rad`, leaving `0.025 rad` before the
  unchanged `0.25 rad` runtime rejection;
- predicted joints are trained to remain `0.020 rad` inside native limits;
- upper phase penalty begins at `phase_cap - 0.020 rad`; and
- lower phase remains exactly zero because zero advance is legitimate idle
  behavior.

These margins are feasible for the exact 2,048-row fitted receipt. A read-only
target audit measured:

```text
maximum adjacent target joint step     0.19434923564685413 rad
target joint-step q99                  0.09309998405937413 rad
minimum target native-limit clearance  0.03785574245610318 rad
phase q99                              0.34906582224089533 rad
phase cap                              0.523598733361343 rad
maximum fitted target phase advance    0.34906582224089533 rad
```

No target exceeds the `0.225` joint-step onset, enters the `0.020` native-limit
margin, or enters the upper `0.020` phase margin.

The feasibility audit uses exactly
`8 * np.finfo(np.float32).eps = 9.5367431640625e-7 rad` as its sole negative
round-trip tolerance. It records the unclamped physical minimum. A minimum
strictly below `-9.5367431640625e-7 rad` rejects; otherwise, and only after that
check, audit calculations use `max(reconstructed_phase, 0.0)`. Thus `-1e-6`
rejects. This tolerance exists only for sealed-target auditing. Model
predictions are never clamped or tolerance-adjusted; every strict negative
prediction receives the phase-low penalty and remains subject to the unchanged
gate.

## Per-pair risks

`normalized_prediction` and `reached_output_normalized` both use the existing
268-value output normalization. Denormalize each exactly once. Only the reached
joint-position slice is consumed from the predecessor.

For pair `i` and joint `j`:

```text
step_excess_ij = relu(abs(q_pred_ij - q_reached_ij) - 0.225) / 0.025
step_risk_i    = max_j(step_excess_ij ** 2)

lower_excess_ij = relu((lower_j + 0.020) - q_pred_ij) / 0.020
upper_excess_ij = relu(q_pred_ij - (upper_j - 0.020)) / 0.020
limit_risk_i    = max_j(max(lower_excess_ij, upper_excess_ij) ** 2)

phase_low_i  = relu(-phase_advance_i) / 0.020
phase_high_i = relu(phase_advance_i - (phase_cap - 0.020)) / 0.020
phase_risk_i = max(phase_low_i, phase_high_i ** 2)
```

Each risk is dimensionless. The worst-joint reduction occurs before the tail
reduction, so adding safe joints cannot reduce an unsafe joint's contribution.

PyTorch `relu` supplies a zero derivative at an exact boundary. Therefore a
joint step exactly `0.225`, a joint exactly on the inset native boundary, phase
advance exactly zero, or phase exactly `cap - 0.020` has zero loss and zero
envelope gradient. Values strictly inside the objective envelope also have
zero envelope gradient. The lower phase hinge is deliberately linear: every
strict negative has derivative `-1 / 0.020`, rather than the vanishing gradient
of a squared hinge near zero. This preserves legitimate zero phase without
making small negative predictions sticky. The global maximum and tail both use
descending risk then ascending ordinal; an exact maximum tie sends the max-term
gradient only to the lowest global ordinal.

## Exact global max-plus-tail reduction

For a risk vector with `N` global pair-time elements:

```text
k = max(1, ceil(0.10 * N))
global_max(risk) = max(all N risks)
CVaR_0.10(risk)  = sum(the k greatest risks) / k
active(risk)     = selected top-k risks strictly greater than zero
a                = max(1, count(active(risk)))
positive_tail_mean(risk) = sum(active(risk)) / a, or differentiable zero
family_loss      = deterministic_global_max(risk) + positive_tail_mean(risk)
```

Sort by descending detached risk value, then ascending global pair-time
ordinal. The ordinal is defined by the unpartitioned global batch order and,
for rollout, by `(time_index, global_batch_ordinal)`. Thus ties select the same
elements on every rank and after resume.

Compute a separate deterministic global max, ordinary CVaR, selected-positive
count/mean, and family loss for joint step, native limit, and phase. CVaR is
diagnostic; optimization uses the max-plus-positive-tail-mean family loss. The
training loss adds each complete family loss with exact weight `1.0` to the
existing loss:

```text
total = existing_total
      + 1.0 * joint_step_envelope
      + 1.0 * joint_limit_envelope
      + 1.0 * phase_envelope
```

Do not combine the three families before max/tail selection; a large joint
error must not hide a phase tail or vice versa. Counts in one-step metrics are
numbers of canonical predecessor/current pairs, never joints or hinge terms.
Counts in rollout metrics are numbers of pair-time elements in `[T,B]`, never
joints, sequences, or hinge terms.

### DDP validation, forward, and gradient semantics

No rank enters a variable-size risk/ordinal gather first, and no rank throws on
malformed local arguments before the metadata collective. Every rank maps its
arguments to one fixed-size sentinel-capable `int64` metadata tensor. Stable
codes are `risk_dtype={float32:1,float64:2}`, `device_type={cpu:1,cuda:2}`,
and `backend={gloo:1,nccl:2}`, with `-1` for malformed/unsupported values. The
metadata includes protocol version, local count, tensor/shape/dtype/device
validity bits, stable risk-dtype code, stable risk-device code,
backend-compatible-device bit, finiteness, ordinal validity, supplied and
actual rank/world size, backend code, tail-fraction validity, and the exact
float64 tail-fraction bit pattern computed with
`struct.unpack(">Q", struct.pack(">d", value))[0]`. Invalid fractions carry a
zero validity bit and `-1` bit-pattern sentinel.

Metadata itself uses a backend-compatible device independent of malformed
risk arguments: CPU for gloo and the current CUDA device for NCCL. After the
fixed-size all-gather, every rank performs the same validation. It requires
the same positive local count, identical supported dtype/device codes,
device/backend compatibility, every validity/finiteness bit, supplied values
equal to `torch.distributed.get_rank()`/`get_world_size()`, valid tail fractions,
and identical fraction bit patterns. Float32/float64 disagreement, CPU/CUDA
disagreement, different valid fractions, and one invalid fraction therefore
reject identically before any risk/ordinal value gather. Only then may all
ranks gather equal-size detached risks and ordinals. The
concatenated ordinals, after sorting, must equal exactly `0..N-1`; duplicates,
gaps, negative/out-of-range values, or rank-local omissions reject collectively
before backward.

All ranks then derive the same detached ordering: descending risk, ascending
global ordinal. The first element alone owns the deterministic max gradient.
Let `S_r` be rank `r`'s differentiable sum among selected top-k elements whose
detached risk is strictly positive, let `a` be their global count (with divisor
`max(1,a)`), and let `P_r` be its differentiable risk only when it owns the
first global ordinal in the ordering. If `W` is world size, use:

```text
positive-tail local surrogate = (W / max(1,a)) * S_r
max local surrogate           = W * P_r
```

If `a == 0`, construct the positive-tail term as `local_risk.sum() * 0.0` and
return exact differentiable zero. DDP's parameter-gradient average therefore
equals the single-rank gradients of the active positive mean and
lowest-ordinal maximum respectively. Each returned tensor uses
`local_surrogate + (global_detached_value - local_surrogate.detach())`, so
forward values are identical on all ranks. Ordinary CVaR uses its own
`(W/k) * selected_sum` diagnostic surrogate. Cross-rank ties select the lowest
ordinal; all active positives on one rank, unequal counts, rank/world mismatch,
dtype/device/fraction disagreement, ordinal gaps, and one-rank nonfinite risks
are bounded collective regression cases. Every metadata-rejected case asserts
that risk/ordinal gather counts remain zero.

## Predecessor-aware one-step data flow

Derive the complete same-clip, same-lane, center-adjacent pair population from
the run's active train-only optimization rows. In the controlled pipeline
rerun those rows remain the exact 2,048-row fitted receipt and produce 2,011
pairs. In ordinary full-corpus training the active rows are the complete train
split, so the same algorithm produces and binds the complete train-only pair
population rather than silently retaining the canary's 2,011-pair population.
One-step optimization samples pair positions rather than isolated row
positions:

```python
@dataclass(frozen=True)
class FittedTransitionPair:
    predecessor_index: int
    current_index: int
    clip_id: str
    sequence_lane: str
    predecessor_center_frame: int
    center_frame: int
    predecessor_row_sha256: str
    current_row_sha256: str
    terrain_class: str
```

Each one-step batch contains current `x`, current phase, current target `y`, and
the exact predecessor target `y`. Run starts without an active same-lane
predecessor are not one-step sampler elements. In pipeline mode they remain in
the fixed 2,048-row receipt and fixed-sample evaluation; in full training they
remain dataset rows but are not legal predecessor-relative updates. Pair
sampling is balanced by the current row's terrain class with the existing
deterministic epoch algorithm.

The full canonical pair records are ordered by clip, lane, predecessor center,
current center, and both v2 row digests. Pair construction derives complete
adjacency internally from all active rows; no caller-supplied subset is trusted.
An independent recomputation must exactly match the full record sequence before
resume. Omission, addition, duplication, reordering, a non-train split, or a
changed lane/predecessor/class rejects before model or optimizer restoration.

Checkpoint v6 stores only a compact receipt: schema, active-row count/digest,
pair count, current-row terrain-class counts, canonical `pairs_sha256`, and
receipt digest. It never embeds all pair records. Safe resume recomputes the
full canonical records from active train rows, verifies completeness with the
independent adjacency implementation, hashes them, and compares the compact
receipt before restoring model or Adam.

## Rollout data flow and gradients

Rollout discovery still uses exact 16-frame same-clip/same-lane consecutive
current rows, but a valid sequence now also requires the fitted predecessor of
its first row. The batch carries that predecessor target.

At rollout time:

1. step 0 compares prediction 0 with the physical joint state in the supplied
   predecessor target;
2. step `t > 0` compares prediction `t` with prediction `t-1`;
3. the previous prediction is never detached;
4. native-limit and phase risks apply to every prediction; and
5. each family is flattened over all `16 * global_batch_size` pair-time
   elements before the global 10% tail reduction.

Thus a late joint-step loss differentiates through both the current prediction
and the reached prior prediction, while the existing shared recurrence also
continues to propagate trajectory/body/phase gradients backward.

## Public APIs and ownership

`sonic/python/mm_sonic/terrain_pfnn/training.py` owns the pure differentiable
objective:

```python
def physical_envelope_risks(
    normalized_prediction: torch.Tensor,
    reached_output_normalized: torch.Tensor,
    *,
    normalization: object,
    joint_limits: torch.Tensor,
    phase_advance_cap: float,
    contract: PhysicalEnvelopeObjective,
) -> dict[str, torch.Tensor]:
    """Return joint_step, joint_limit, and phase risks with shape [B]."""


@dataclass(frozen=True)
class GlobalEnvelopeReduction:
    maximum: torch.Tensor
    cvar: torch.Tensor
    tail_count: int
    active_count: int
    positive_tail_mean: torch.Tensor
    loss: torch.Tensor


def global_max_plus_tail(
    local_risk: torch.Tensor,
    *,
    global_ordinals: torch.Tensor,
    tail_fraction: float,
    rank: int,
    world_size: int,
) -> GlobalEnvelopeReduction:
    """Return max, CVaR, active-positive mean, and scale-stable family loss."""


def physical_envelope_loss(
    normalized_predictions: torch.Tensor,
    reached_outputs_normalized: torch.Tensor,
    *,
    normalization: object,
    joint_limits: torch.Tensor,
    phase_advance_cap: float,
    contract: PhysicalEnvelopeObjective,
    global_ordinals: torch.Tensor,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor]:
    """Return per-family max/CVaR/active-tail diagnostics, losses, and total."""
```

The first dimension may be flattened pair-time. All remaining shapes are
exact: predictions and reached outputs `[N,268]`, ordinals `[N]`, joint limits
`[29,2]`, and returned per-family losses scalar.

`sonic/python/mm_sonic/train_terrain_pfnn.py` owns pair materialization,
receipts, deterministic sampling, DDP ordinals, reporting, and checkpoint
wiring:

```text
canonical_transition_pairs(dataset: object) -> tuple[FittedTransitionPair, ...]

fitted_transition_pair_receipt(dataset: object) -> dict[str, object]

validate_fitted_transition_pair_receipt(
    receipt: Mapping[str, object],
    dataset: object,
) -> dict[str, object]

materialize_transition_pairs(dataset: object) -> object
```

`autoregressive_unroll` gains required `initial_predecessor_targets`,
`joint_limits`, `envelope_contract`, `global_batch_ordinals`, `rank`, and
`world_size` arguments. It returns the three envelope losses in its existing
loss mapping and exposes no detached reached-state path.

## Schema and provenance implications

- Dataset schema remains exactly `mm-sonic-terrain-pfnn-dataset/v2`; no shard
  or normalization field changes.
- Fitted-transition report remains
  `mm-sonic-fitted-transition-report/v1`; its physical limits and ordering do
  not change.
- Checkpoint advances from v5 to
  `mm-sonic-terrain-pfnn-checkpoint/v6` because the exact loss surface and
  sampler population change.
- Checkpoint v6 adds exact `physical_envelope_objective`, compact
  `fitted_pair_receipt`, and `target_envelope_audit` fields and adds the three
  complete max-plus-tail family-loss weights to the frozen loss-weight mapping. Here
  `fitted_pair_receipt` is the historical schema name for the active train-only
  optimization-pair receipt in both pipeline and full-corpus modes. v5 fails
  before model construction.
- The one-step report advances to
  `mm-sonic-terrain-pfnn-one-step-report/v2` and records the objective contract,
  pair receipt/digest, per-family maximum risk, ordinary CVaR, tail count,
  active-positive count/mean, combined family loss, and pair counts above both training onsets and
  unchanged authorization limits. Rollout metric streams use pair-time counts.
  Objective metrics are training diagnostics, not alternate acceptance
  thresholds: authorization still requires zero failures at the unchanged
  physical gate.
- Pipeline promotion receipt advances to
  `mm-sonic-pipeline-promotion-receipt/v3`, explicitly binding the objective
  digest and pair-receipt digest in addition to all v2 fields. The candidate
  SHA continues to bind the complete checkpoint.
- The verifier request carries both digests. Missing, older, recomputed, or
  mismatched fields reject before the known-terrain callback.

Resume recomputes the v2 fitted subset, full canonical adjacent records through
two completeness paths, compact pair receipt, target-feasibility audit,
objective contract, active pair count, one-step sampler state, and sequence
sampler state before restoring model or Adam. DDP ranks receive the same
canonical receipts and objective digest before any update.

## Validation, errors, and stop rules

Fail before training if:

- a pair is not exact same-clip/same-lane/adjacent;
- a predecessor/current row digest or class is inconsistent;
- any fitted target violates the stricter training feasibility envelope;
- `phase_cap <= 0.020`;
- the pair population lacks a required terrain class;
- an objective constant, tail fraction, ordinal, DDP shape, or risk is invalid;
  or
- resume does not match v6 objective, pair, fitted-row, sampler, normalization,
  dataset, or kinematic receipts.

Exactly one controlled recurrence-v2 rerun is permitted after every focused and
full test passes. It uses the existing accepted recurrence-v2 dataset, the same
2,048-row selection, seed 7, GPU 1, 4,000 one-step updates, and 256 rollout16
updates. Do not resume the failed candidate.

Before invoking dataset `--resume`, persist
`.superpowers/sdd/physical-envelope-objective-dataset-before.json`; after the
command, persist
`.superpowers/sdd/physical-envelope-objective-dataset-after.json`. Each snapshot
has an exact entry for all six accepted dataset files (`manifest.json`,
`normalization.npz`, two train shards, one validation shard, and one test
shard), with relative path, SHA-256, byte size, and `mtime_ns`, plus a canonical
snapshot digest. Record both JSON-file SHA-256 values in the artifact report and
require the two canonical snapshot digests and every entry to be identical.
This proves the resume operation was a byte/mtime no-op without reading sealed
test examples for evaluation.

After that run:

1. require finite losses and safe v6 reload;
2. require exact fixed-sample reproduction;
3. require the unchanged full fitted-transition gate to accept every pair;
4. only then run the exact 600-tick known-terrain evaluator; and
5. only then publish immutable `best.pt` with a v3 receipt.

At the first failure, record exact clip, lane, predecessor/current center,
tick if applicable, field, joint, value, limit, objective metrics, and receipt
digests. Stop all training/evaluator attempts. No second run, margin change,
tail-fraction change, weight change, threshold change, or output correction is
authorized inside the same correction cycle.

The correction source/test range becomes immutable immediately before the sole
corrected training command starts. No corrective source/test edit, commit,
hotfix, threshold change, or retraining is authorized after that boundary, even
if the artifact reveals a representation defect. Record the exact evidence as
a proposed input to a separately approved future correction cycle and stop. A
post-artifact Task 7 review may approve the frozen correction source or record
blockers; it may not fix them in this cycle. Original Tasks 8/9 continue only
after artifact acceptance and a blocker-free review. Their already-approved
viewer/full-corpus file scope is separate downstream work, not authority to
alter the frozen physical-envelope correction.

## Verification requirements

TDD regressions must prove:

1. exact margin equations, denormalization, shapes, finiteness, and zero
   gradient at every satisfied boundary;
2. worst-joint reduction prevents 28 safe joints from diluting one failure;
3. lowest-ordinal global max plus selected-positive tail mean prevents
   appending arbitrary safe rows/time steps/ranks around a lone offender from
   reducing either family loss or that offender's gradient;
4. deterministic cross-rank ties, all selected elements on one rank, and
   two-rank DDP produce the same max/tail/CVaR forward values and parameter
   gradients as a single-rank global reference;
5. fixed-size collective metadata rejects unequal counts, supplied/actual
   rank or world-size mismatch, risk dtype/device disagreement, exact
   tail-fraction-bit disagreement or invalidity, ordinal gaps/duplicates/range
   errors, and a nonfinite risk on only one rank without entering a
   risk/ordinal value gather;
6. every strict negative predicted phase has nonzero linear corrective
   gradient while exact zero has zero loss and gradient;
7. one-step batches always carry an exact same-lane predecessor target;
8. pair adjacency omission/addition/duplication/reordering, split/lane/row
   tampering, and compact pair-receipt tampering fail closed;
9. rollout step 0 uses the sealed predecessor target and later steps use the
   prior prediction without detaching it;
10. a late rollout envelope loss has nonzero gradient to the earlier reached
   prediction;
11. target audit records the unclamped minimum, accepts exactly the fixed
    float32 tolerance, and rejects `-1e-6 rad` before its audit-only clamp;
12. v5 and v2 promotion receipts fail closed, while v6/v3 round-trip exactly;
13. objective/pair/sampler mismatch rejects before model or optimizer restore;
14. one-step report counts are pair counts and rollout stream counts are
    pair-time counts, with no joint/hinge-term multiplicity;
15. the existing physical gate, runtime, raw outputs, controller, thresholds,
    and no-callback ordering remain byte-for-byte or behaviorally unchanged;
    and
16. the real fitted target-feasibility audit and persisted before/after
    six-file snapshots reproduce the measured margins and exact no-op before
    the one controlled run.

## Completion sequence

If the controlled candidate passes the fitted and 600-tick gates, independently
re-review original Task 7 from `6af614f` through the frozen source commit. Any
Critical/Important finding is recorded and stops this cycle; it is not fixed
after the artifact boundary. Only a blocker-free review resumes original Tasks
8 and 9:

- Task 8 builds and verifies the shared three-hill MuJoCo viewer;
- Task 9 builds the full corpus, trains/selects only on validation, opens the
  sealed test exactly once, and runs the headless three-hill canary; and
- final acceptance launches the passive viewer with the accepted immutable
  checkpoint and physically drives W/A/S/D across flat, ascent, summit,
  descent, flank direction changes, both crossing directions, and resumed flat
  walking, recording visual observations without altering the model.

The milestone is not complete until that final interactive visual drive is
performed after every preceding automated and sealed gate passes.
