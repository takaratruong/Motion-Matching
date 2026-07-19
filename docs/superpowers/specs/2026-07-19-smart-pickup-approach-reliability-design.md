# Smart Pickup Approach Reliability Design

## Problem and Evidence

Smart Pickup currently has two deterministic false-failure modes before the
runtime receives a pick request.

First, `PickAssistConfig::maximum_assisted_path_m` has two unrelated jobs. It
correctly limits activation to an authored slot within 1.00 m, but it also
caps the sum of every displayed-root movement segment at 1.00 m. Motion-
matching sway, overshoot, walking backward, and obstacle detours consume that
odometry budget even when the character remains close to a clear slot. The
attempt then fails with `OutsideTravelEnvelope` instead of finishing its
funnel.

Second, candidate evaluation records hard failures from every clip. When one
physically feasible candidate is merely over the feature-cost threshold
(`PoorMatch`) but an unrelated clip hits `CorrectionLimit`, legacy aggregate
reason priority reports `CorrectionLimit`. The final-preview coordinator
retries only exact `PoorMatch`, so this mixed result becomes an immediate
`FinalPreviewRejected` even though the stationary retry path should remain
available.

These are policy/reasoning defects, not missing motion data and not evidence
that a learned approach model is required.

## Approaches Considered

1. Separate activation eligibility from execution telemetry and separate the
   preview-oriented match reason from the legacy selector reason. This is
   selected. It fixes the observed semantics without widening the interaction
   radius or changing runtime matching authority.
2. Raise the shared path limit from 1 m to 2 m. This is a small patch, but it
   leaves two concepts coupled and will fail again on sufficiently indirect
   motion.
3. Replace the authored-slot funnel with a learned or diffusion approach
   model. Such a model may later generate more varied approach paths, but it
   should consume correct deterministic slot/reachability labels rather than
   compensate for incorrect admission and reason policies.

## Decision: Admission Radius, Not Odometry Budget

The existing 1.00 m value remains the hard admission radius. Pressing `F`
selects only an authored slot whose initial route satisfies the current table,
object, obstacle, and maximum-distance checks. A genuinely out-of-range start
fails immediately and never takes long-lived steering ownership.

After an attempt begins, `assisted_travel_m` remains cumulative telemetry, but
it is no longer a rejection condition. Revalidation continues to check finite
geometry, table/object exclusion, and frozen obstacle clearance. It does not
reapply the activation radius to the character's current distance from the
already-admitted slot.

The attempt remains bounded by the existing 250-tick arrival deadline (10
seconds at 25 Hz), explicit WASD/`X` cancellation, target mutation checks, and
finite-value checks. If a future product needs an execution leash, it will be
a separately named, larger hysteretic exit radius rather than the admission
radius reused as accumulated odometry.

## Decision: Preview Reason Versus Selector Reason

`matcher_detail::PickEvaluation::selection.reason` keeps its existing hard-
reason precedence. Normal runtime preflight and `select_whole_clip()` therefore
retain their current compatibility and failure semantics.

`PickEvaluation::match_reason` answers a narrower preview question: after at
least one candidate passed hard path/correction feasibility, why is no motion
ready? If any such evaluation reached feature scoring and was over cost,
`match_reason` is `PoorMatch` unless an `OutOfRange` failure indicates malformed
or missing pack feature storage. A genuine correction-only evaluation remains
`CorrectionLimit`.

Concretely, after the unchanged legacy selection is rejected:

```cpp
result.match_reason =
    !failures.out_of_range && failures.poor_match
        ? Reason::PoorMatch
        : result.selection.reason;
```

This intentionally permits `match_reason` and `selection.reason` to differ in
the mixed `CorrectionLimit + PoorMatch` case. The runtime preview consumes
`match_reason` and can keep the stationary retry alive; authoritative preflight
continues to consume `selection`.

## Interfaces and Boundaries

`interaction_pick_slots` exposes a collision-only revalidation operation for
an already-admitted frozen slot. It shares the existing finite/table/object/
obstacle geometry implementation but omits the initial maximum-distance gate.
Initial `select_pick_slot` behavior and its inclusive 1.00 m tolerance remain
unchanged.

`ControllerPickAssist::observe` continues accumulating each finite planar
segment into diagnostics. It fails on non-finite/overflow telemetry, but not
because the finite sum exceeds `maximum_assisted_path_m`. It uses the new
collision-only revalidation operation during SlotApproach, Settling, and
FinalPreview.

`matcher_detail::finish_evaluation` derives preview `match_reason` after it
constructs the unchanged selection result. No runtime/controller public API,
request authority, clip data, matcher cost, interaction frequency, or terrain
behavior changes.

Readiness-aware fallback across multiple authored entry transforms is the next
separate design. This slice makes each individual attempt semantically sound;
it does not switch slots mid-attempt.

## Error Handling

- Initial route beyond 1.00 m: immediate `OutsideTravelEnvelope` and no
  sustained automatic movement.
- Finite cumulative travel above 1.00 m after admission: telemetry only; the
  attempt continues while its collision route remains valid.
- Non-finite root, segment, cumulative telemetry, yaw, or speed: terminal
  `OutsideTravelEnvelope`.
- Table/object/obstacle path becomes blocked: retain the existing exact hard
  failure reason.
- Mixed feasible `PoorMatch` and unrelated `CorrectionLimit`: stationary
  retry, not immediate failure.
- Missing/out-of-range feature storage: retain `OutOfRange` priority and fail
  closed.
- Arrival deadline, target mutation, cancellation, and request extraction
  semantics remain unchanged.

## Verification

Tests are written and observed RED before production edits:

1. A two-clip matcher fixture has one candidate rejected by realized
   `CorrectionLimit` and another hard-feasible candidate rejected only for
   over-cost features. It requires `path_feasible=true`,
   `match_reason=PoorMatch`, and the unchanged
   `selection.reason=CorrectionLimit`.
2. Existing pure PoorMatch, pure CorrectionLimit, mixed OutOfRange/PoorMatch,
   selector, runtime preflight, and public preview contracts remain green.
3. A Smart Pickup approach walks `0.5 -> 0.0 -> 0.5` toward a frozen slot at
   `0.8`, accumulating exactly 1.5 m while every current route is clear and at
   most 0.8 m. It remains active and reports the full telemetry instead of
   failing.
4. An already-admitted collision-only revalidation at 1.00003 m remains clear,
   while initial slot admission still accepts 1.00002 m and rejects 1.00003 m.
5. Settling uses the same execution semantics and remains bounded by the
   existing arrival deadline.
6. Focused matcher, slot, assist, Smart Pickup controller, runtime, native
   adapter, and exact-25-Hz test suites pass before reviewed commits are
   pushed.
