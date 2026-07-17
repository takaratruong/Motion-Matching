# G1 State-Dependent Inertialized Candidate Preview Design

**Date:** 2026-07-17

**Status:** Approved direction; written specification ready for user review

**Base:** `e8e7d8d6388579439709be28d213c090732f65c0`

**Implementation branch:** `g1-sonic-scene-aware-baseline`

## Purpose

The authenticated raw-feasibility mask solved its intended defect: the matcher
no longer advances from raw-safe frame 866 into raw-unsafe frame 867, and the
existing 20-frame search neighborhood remains effective.

The corrected immutable Stage A trial exposed the next boundary. After one
safe transition through selected frame 865 into emitted frame 866, the next
search selected certified frame 927 and emitted certified raw-safe frame 928.
Frame 928 projects the left ankle roll to `-0.217500582`, but applying the
current stateful inertial offsets produces `-0.27224052`, outside the registered
lower limit `-0.261799991`.

This intervention asks one narrow research question:

> Can the existing matcher complete its reference while retaining its current
> cost ordering and neighborhood semantics, but accepting only candidates whose
> exact next state-dependent inertialized joint interval, including the 50 Hz
> bridge midpoint, satisfies the authenticated G1 joint contract?

The intervention validates a candidate before selection. It does not repair or
modify an emitted pose.

## Root-cause boundary

The invalid value does not originate in the database, raw-successor mask,
registered contract, support retargeting, or neighborhood logic. It is produced
by composing two consecutive transitions with the live bone offset state.

The runtime sequence is:

1. choose a selected database frame;
2. update inertial offsets for a transition, if any;
3. advance once to the selected frame's range-clamped successor;
4. apply one inertialization update to that raw successor;
5. continue through simulation, support, horizontal-root adjustment, and final
   observation.

Joint contract rows map only to source bones 2 through 30. After step 4, later
runtime work changes root translation/orientation bookkeeping and support
positions but does not change those local joint rotations or angular
velocities. A side-effect-free preview immediately after step 4 can therefore
use the same structured joint projection as the final server boundary and is
representative of the joint-limit result. The existing final projection remains
authoritative and must still run.

## Goals

- Preview the exact one-step inertialized joint interval for ordinary
  continuation and each prospective transition, including its emitted 50 Hz
  midpoint.
- Preserve the existing feature query, transition cost, branch-and-bound cost
  order, raw-feasibility mask, range clamp, and 20-frame neighborhood.
- Reject only candidates whose structured preview fails a registered joint
  limit, then continue searching for the lowest-cost accepted candidate.
- Treat every non-limit preview failure as a fatal integration defect.
- Share the pose-advance primitive between preview and the live selected path so
  their transition/inertialization formulas cannot drift.
- Keep candidate evaluation side-effect free and keep failed runtime steps and
  protocol generation transactional.
- Publish strict per-step preview/rejection diagnostics for scientific audit.
- Rerun the unchanged Stage A command in a fresh immutable output root after all
  repository and protected gates pass.

## Non-goals

This change does not:

- clip, clamp, sanitize, blend, or interpolate an invalid joint into range;
- widen or replace the registered joint limits;
- change the published database, sidecars, GEAR checkout, policy, commands,
  scenes, feature weights, transition cost, or Stage A thresholds;
- remove the hard final `sonic_project_pose()` boundary;
- retry a step after the live final boundary has failed;
- weaken the search neighborhood or allow a rejected nearby frame to re-enter;
- enumerate or cache all state-dependent transitions offline;
- tune inertialization halflife or any motion-matching parameter;
- guarantee a seven-gate pass or conceal the next scientific blocker.

## Considered approaches

### Selected: in-search exact candidate preview

The runtime previews the exact next inertialized local joint state from a
read-only baseline and the candidate's range-clamped successor. The database
search invokes a tri-state validator before promoting a lower-cost candidate.
A registered limit violation rejects that candidate; a valid preview accepts
it; any other projection failure aborts.

This is selected because it preserves the matcher's cost objective and chooses
the true lowest-cost member of the dynamically feasible candidate population.
It also prevents an invalid candidate from becoming live state.

### Rejected: transactional whole-step retry

The server could select a candidate, execute the full step, discover a final
projection failure, roll back, exclude the candidate, and execute the step
again. This observes the final boundary directly, but it repeats command,
terrain, support, and protocol work, complicates error ownership, and creates a
second search loop outside the matcher. The joint rotations relevant to the
limit are already final at the narrower preview seam, so full-step retry adds
cost and state risk without additional joint-limit fidelity.

### Rejected: static margin or transition cooldown

A wider offline joint margin or a rule forbidding consecutive transitions is
simple, but neither models the live offset state. A margin can reject safe
candidates and still miss a different stateful composition; a cooldown changes
search scheduling and can retain an unsafe ordinary continuation. These are
heuristics, not exact tests of the observed mechanism.

## Architecture

The change stays within four existing ownership boundaries:

- `database.h` owns generic cost search, explicit neighborhood centering, and
  tri-state candidate-validation control flow, but no G1 contract knowledge;
- `sonic/cpp/g1_runtime.h` owns scratch pose preview, ordinary-continuation
  scheduling, shared transition/update application, and step diagnostics;
- `sonic/cpp/g1_joint_projection.h` owns contract-relative endpoint projection
  and the exact binary32-`dt` Hermite midpoint predicate against binary64 JSON
  contract limits;
- `sonic/cpp/mm_chunk_server.cpp` owns the real contract-bound validator and
  conversion of runtime diagnostics into protocol diagnostics;
- `sonic/cpp/mm_chunk_protocol.h`, the C++ JSON writer, strict JSON schema, and
  Python schema/runtime-log layers own wire representation and validation.

No controller, terrain, support, or feasibility-certificate module gains a
second responsibility.

### 1. Shared one-step pose-advance primitive

The transition and one-step inertialization sequence currently embedded in
`g1_runtime_step_internal()` is factored into one internal primitive. It
operates on an explicitly supplied pose workspace containing:

- bone positions, velocities, rotations, and angular velocities;
- bone offset positions, velocities, rotations, and angular velocities;
- transition source/destination root transforms.

Its inputs identify the current live pose, current raw input pose, prospective
selected frame, range-clamped emitted successor, transition flag,
inertialization halflife, and `dt`. The primitive performs the existing
`inertialize_pose_transition()` call when the selected frame differs from the
prior frame, followed by the existing `inertialize_pose_update()` call for the
emitted successor.

The live runtime calls this primitive on the cloned `next` state. Candidate
preview calls the same primitive on scratch storage. No transition or update
formula is duplicated.

Scratch arrays are allocated once per runtime step and reset from the same
read-only baseline for each candidate. The implementation does not clone the
complete controller state or allocate per candidate. Preview never mutates the
database, live state, output result, search timers, or diagnostics until a
candidate is selected.

### 1a. Exact emitted-interval predicate

The first immutable Stage A qualification trial exposed a narrower downstream
gap after the endpoint-preview implementation. At chunk 6, the accepted 25 Hz
left/right ankle-roll endpoints were `-0.260674924` and `-0.259201497`, but the
existing 25-to-50 Hz cubic-Hermite bridge produced `-0.2662098130672067` at
`u=0.5`, outside the unchanged lower limit. That terminal run is preserved as
scientific evidence; the selector must cover the interval the G1 bridge emits,
not only its right endpoint.

For each runtime step, the real adapter projects the shared live left boundary
once before search. A malformed shared projection is fatal and a registered
left-boundary violation remains a live scientific failure. Each candidate then
projects its exact scratch right boundary and evaluates the same midpoint as
`resample.py`:

```text
q_mid = 0.5*q_left + 0.125*float32(0.04)*v_left
      + 0.5*q_right - 0.125*float32(0.04)*v_right
```

Endpoint values remain binary32 and are promoted to double for the arithmetic,
matching the Python bridge. Contract lower/upper values retain the binary64
values produced by parsing the authoritative JSON; they are never rounded
through binary32 before the comparison. A well-formed endpoint or midpoint
limit result rejects only that candidate; malformed/non-finite projection or
midpoint state fails closed. The diagnostic records the rejected source-joint
row and the exact binary64 projected position, whether the first violation was
at the right endpoint or midpoint.

### 2. Explicit ordinary-continuation preview

Before deciding whether search is required, the real G1 runtime previews the
ordinary no-transition continuation from the current frame to its
range-clamped successor.

- A raw-unsafe successor continues to force search as today.
- A raw-safe successor whose exact inertialized preview has a registered limit
  violation also forces search, even when the normal search timer has not
  expired or matching was disabled by the caller.
- An end-of-range incumbent remains unavailable and forces search.
- A valid ordinary preview remains eligible as the incumbent during a scheduled
  search.
- A non-limit preview failure aborts the runtime step transactionally.

This closes the same state-dependent boundary for decaying offsets on ordinary
progression instead of protecting transitions only.

The ordinary preview result is cached for the step. If the current frame is
later considered as the search incumbent, the search reuses that result rather
than projecting it twice.

### 3. Validated search with an explicit neighborhood center

The existing unvalidated `motion_matching_search()` and `database_search()`
entry points retain their current behavior. A validated search path adds:

- an explicit `neighborhood_center`, always the prior database frame;
- a prevalidated optional incumbent;
- a candidate validator with accept, reject-limit, and fatal verdicts;
- accumulated preview diagnostics.

Separating `neighborhood_center` from `best_index` is mandatory. An unavailable
incumbent must not erase the prior frame used by `ignore_surrounding=20`, which
was the defect found in the first raw-mask qualification.

The validated branch-and-bound leaf logic is:

1. skip candidates excluded by the immutable `search_safe` mask;
2. skip candidates inside the existing prior-frame neighborhood;
3. compute the existing feature plus transition cost;
4. if the cost cannot beat the lowest accepted cost, prune it as today;
5. otherwise preview the candidate's exact emitted successor;
6. promote it only on an accept verdict;
7. continue without lowering the best cost on a registered-limit rejection;
8. stop and return fatal status on any other preview defect.

Because rejected candidates never lower the pruning threshold, every candidate
that could beat the best accepted result remains reachable. The result is the
same minimum-cost selection over the dynamically accepted population. Ties
retain the existing strict-less-than and database-order behavior.

If validation is absent, mask filtering, incumbent cost, pruning, selection,
and tie behavior remain unchanged.

### 4. Structured G1 projection validator

Generic database search does not depend on the Sonic joint contract. The G1
runtime accepts a validator supplied by its caller. The real MM adapter binds a
validator that calls the existing `sonic_project_joint_state()` primitive with
the previewed local rotations and angular velocities.

Validator verdicts are defined exactly:

- `accept`: projection succeeds with `SonicJointProjectionValid`;
- `reject-limit`: projection fails with `SonicJointProjectionLimit`, a valid
  registered row, finite position/lower/upper values, ordered limits, and a
  position actually outside that interval;
- `fatal`: shape, contract, input, singular, residual, velocity, malformed
  limit diagnostic, successful projection with a failure diagnostic, or any
  non-finite result.

Only `reject-limit` is recoverable by trying another candidate. The validator
copies the first limit diagnostic into step-owned audit data without parsing a
human-readable error string. Fatal errors preserve the structured projection's
exact rendered detail and abort before state publication.

Legacy runtime overloads use an accept-all validator so non-Sonic callers and
existing unit fixtures remain source-compatible and behavior-compatible.

### 5. Selection and live parity

After search returns an accepted index, the runtime applies the shared
pose-advance primitive to the cloned live state exactly once. It then continues
through the unchanged simulation, support, root adjustment, diagnostics, and
transactional state swap.

The MM adapter still calls `sonic_project_pose()` on every resulting boundary.
There is no fallback if that live projection fails. A live joint-limit failure
after an accepted preview remains a scientific failure and is evidence of a
preview/live parity defect or a later state mutation; it is not silently retried.

Focused parity tests compare preview and live local joint rotation/angular-
velocity bits for ordinary and transition paths. The real failure replay also
proves that candidate 927 is rejected for the recorded left-ankle value before
selection.

## Protocol and evidence

The strict `mm-chunk/v1` source chunk gains five ten-element step arrays:

- `candidate_preview_count` — number of distinct exact joint projections
  performed for the step, including the cached ordinary preview;
- `candidate_limit_rejection_count` — number of registered-limit preview
  rejections;
- `first_rejected_database_frame` — selected database frame of the first
  rejected candidate, or `-1` when none;
- `first_rejected_joint_index` — fixed source-joint index of the first
  rejection, or `-1` when none;
- `first_rejected_joint_position` — rejected projected binary64 position, or
  exact positive `0.0` when none.

For every successful real step:

- counts are nonnegative and rejection count does not exceed preview count;
- zero rejections require both indices to be `-1` and position to be exact
  zero;
- a positive rejection count requires in-range database/joint indices, a finite
  binary64 position, and a position outside the corresponding binary64 loaded
  contract interval;
- the selected frame is not one of the rejected candidates.

The fake adapter publishes zero counts and the exact no-rejection sentinels.
The bundled C++ writer, JSON schema, Python `SourceChunk`, exact-key parser,
runtime-log validation, and protocol tests change atomically. This is a strict
schema extension within the bundled `mm-chunk/v1` endpoint pair, consistent
with the existing protocol policy; old/new endpoint mixtures fail parsing.

The static hello `joint_feasibility` identity is unchanged because the raw and
search masks are unchanged. The build commit and existing artifact hashes bind
the new dynamic algorithm. Stage A retains the source chunk and its diagnostics
through its existing evidence path, and the final result report reconciles the
preview/rejection totals with the localized replay.

## Error ownership

- Registered joint-limit preview rejection with another accepted candidate:
  normal scientific search behavior, recorded in step diagnostics.
- At least one dynamically previewed candidate rejected and no accepted
  candidate: scientific `generation_failed` with exact message
  `no inertialized-joint-safe database candidate`.
- No immutable raw/search-safe candidate: existing scientific
  `generation_failed` message `no joint-limit-safe database candidate`.
- Non-limit or malformed preview diagnostic: integration failure with the
  structured projection detail.
- Malformed validator, scratch, mask, or database shape: integration failure.
- Final live projected pose outside the registered limits: unchanged scientific
  `generation_failed`; no retry after the live boundary.
- Protocol diagnostic shape/domain inconsistency: integration failure.

The Python verdict boundary recognizes the new no-candidate message only when
the remote code is exactly `generation_failed` and the message is an exact
full-string match. Prefixes, suffixes, lookalikes, wrong codes, and local stderr
remain integration-owned.

When both populations are exhausted in one step, dynamic evidence has priority:
if any exact preview produced `reject-limit`, the new inertialized-candidate
message is emitted; the existing raw-candidate message is reserved for searches
with zero dynamic limit rejection.

## Testing

Implementation is test-first and includes:

1. Shared pose-advance RED tests proving ordinary and transition preview output
   is bit-identical to the corresponding live pre-support local pose, while the
   baseline and rejected scratch state remain unchanged.
2. Validated-search RED tests proving a cheaper rejected candidate yields to the
   next accepted candidate, a fatal validator stops search, ties remain stable,
   unvalidated callers are unchanged, and an unavailable incumbent still uses
   the prior frame as the 20-frame neighborhood center.
3. Runtime RED tests proving an unsafe ordinary inertialized preview forces
   search, a valid ordinary preview remains an eligible incumbent, all rejected
   candidates produce the exact new error, and every failure preserves the
   complete state and result transactionally.
4. Projection-validator RED tests for valid, registered-limit, singular,
   residual, non-finite, velocity, malformed-diagnostic, and wrong-shape cases.
5. Protocol/schema RED tests for all five arrays, exact sentinels, count
   reconciliation, contract-relative rejected position, fake output, strict
   key rejection, and chunk equality/transaction behavior.
6. CLI verdict RED tests for the exact new scientific message and integration-
   owned variants, retaining the existing raw-mask and live joint-limit cases.
7. A protected warning-strict real-artifact replay of steps 50 through 52 that
   proves candidate 927 previews to `-0.27224052`, is rejected, cannot become the
   selected frame, and the accepted candidate's live projection matches its
   preview.
8. Existing raw-certificate, masked-search, runtime, protocol, schema, verdict,
   inventory, warning-strict C++, ASan/UBSan, and full supported Python 3.10
   suites.
9. Fresh real-server launches proving the existing raw-feasibility certificate
   identity remains stable, followed by the unchanged GPU-0 Stage A command in
   a new output root.

Every production behavior change has a witnessed RED failure before its
minimal implementation. Tests written for prior behavior must remain green.

## Scientific acceptance

The narrow candidate-preview hypothesis succeeds only if:

- the step-52 candidate 927 is rejected before selection with the recorded
  left-ankle preview value;
- no emitted frame violates the registered joint contract during gate 4;
- gate 4 retains all 601 required reference frames; and
- the immutable result evidence and inventory verify.

That is not by itself a Stage A pass. Stage A passes only when all seven existing
gates pass with unchanged commands, assets, controller, policy, limits, and
thresholds.

If search exhausts, a different candidate fails, gate 4 fails for another
reason, or a later dynamic gate fails, the run is recorded as the next truthful
scientific result. No additional intervention is folded into the same trial.

## Rollback and provenance

Implementation stays on the existing isolated research branch. The raw mask,
motion database, GEAR checkout, source MJCF, policy, known-good reference, and
active Reliable Claude release remain immutable. Each scientific trial uses a
new output root.

Rollback is the parent of the implementation commits, with
`e8e7d8d6388579439709be28d213c090732f65c0` as the reviewed pre-intervention
baseline. The prior immutable raw-mask trials and corrected failure evidence
remain unchanged and continue to document why this state-dependent experiment
is necessary.
