# G1 Unscheduled Recovery Log Grammar Design

**Date:** 2026-07-17

**Status:** Approved under the user's standing instruction to continue and
auto-approve in-scope work.

**Base:** `2f61b2691273227bad31eec0b791ec277d6184f0` plus the design-history
commit `4e88e71b30cb8c729b0e2943ef79d68f5cc6e00f`.

## Context

The repaired fast trace and no-seam release passed all six byte-parity
comparisons for 18- and 32-frame IK-off/IK-on runs. The canonical production
log checker then stopped on the authentic frame-17 recovery:

```text
ValueError: row 17: transition without search
```

The row is correct:

- matching is enabled;
- the legacy search schedule did not run, so `searched=0`;
- incumbent frame `117711` finitely rejected in hidden IK;
- the strict provider ran once and rank zero `111739` dual-certified;
- the accepted pose advanced to `111740`, so `transitioned=1`; and
- selected cost `6.49063301` strictly beats incumbent cost `7.54006243`.

The checker still encodes the pre-recovery implication
`transitioned => searched`. The production state machine now has two legal
transition owners: scheduled legacy selection and strict recovery after a
finite candidate rejection.

The 305-column production CSV already contains the authenticated permission
needed by the frame-level grammar: `matching_enabled` is copied into
`G1FrameAcceptedDiagnostic`, validated against the execution mode by
`g1_frame_accepted_diagnostic_matches_success`, and logged from that accepted
diagnostic. Production ownership tests separately bind `searched` to the one
legacy `database_search` schedule and bind `transitioned` to the accepted
candidate outcome. The paired candidate TSV supplies provider-specific
authentication for the live gate.

## Objective

Update the checker to represent the complete mode/search/transition grammar
without changing the production controller, motion log schema, matcher,
provider, terrain, timing, thresholds, or candidate trace.

## Non-goals

- No 306th CSV column and no accepted-candidate enum in production logging.
- No generic permission for unsearched transitions.
- No relabeling of strict recovery as legacy search.
- No change to row-17 motion behavior.
- No attempt to repair the separate frame-18 rejection/latch in this task.
- No visualizer interaction, long certification, or G1 mesh work.

## Considered Approaches

### 1. Mode-aware checker grammar — selected

Use the already-authenticated `matching_enabled`, `searched`, and
`transitioned` fields together. Require a strictly improving cost for an
unscheduled transition. Continue binding provider identity and exact winner
through the paired candidate TSV in the live gate.

This is the smallest truthful repair and preserves the existing schema.

### 2. Add accepted `selection_kind` to the CSV — deferred

An explicit `legacy`, `strict-recovery`, or `incumbent` field would let an
isolated CSV state the accepted owner without code or paired-trace context. It
would also change the public schema, writer, transaction diagnostic, every
fixture, release hash, and all downstream column contracts. That broader
contract is not required for the current gate and is deferred unless a future
consumer must authenticate provider ownership from a CSV alone.

### 3. Delete the old implication — rejected

Simply removing `transitioned and not searched` would admit transitions in
matching-disabled mode and would not distinguish the strict recovery cost rule
from the legacy floating-point near-tie allowance.

## Canonical Grammar

The checker parses `matching_enabled` before validating the search/transition
relationship and enforces:

1. `matching_enabled`, `searched`, and `transitioned` are binary flags.
2. `searched=1` requires `matching_enabled=1`.
3. `transitioned=1` requires `matching_enabled=1`.
4. `matching_enabled=1, searched=1` is a scheduled legacy-search prefix. A
   transition follows the existing selected/query/current/range rules. Its
   selected and incumbent costs may retain only the existing bounded four-ULP
   legacy normalization exception.
5. `matching_enabled=1, searched=0, transitioned=1` is the observable
   unscheduled-recovery grammar. It requires `selected_cost < incumbent_cost`
   exactly; the legacy four-ULP exception does not apply.
6. `matching_enabled=0` requires `searched=0` and `transitioned=0`.
7. All existing query snapshot, selected-frame, post-advance frame, source
   range, cost, finite-value, scene, and rejection-lifecycle checks remain in
   force.

The production state machine and its structural tests make the accepted
unscheduled transition reachable only through
`G1CandidateRecoveryTransition`. The Task-2 candidate trace independently
requires incumbent failure, zero legacy traversals, one provider call, one
recovery traversal, exact accelerated/exhaustive equality, and the first
dual-certified strict winner.

## Rejected Publication Rows

A finitely rejected transaction updates the publication and rejection fields
but retains the last accepted state and accepted diagnostic. A rejected row may
therefore repeat an earlier accepted row's `searched` and `transitioned` values.
Those values describe the frozen accepted baseline; they are not a new
transition event on the rejected presentation frame.

This checker repair must continue accepting that lifecycle representation when
all existing frozen-state and accepted-digest relations hold. Aggregate
transition counts must not count a repeated transition bit on a rejected row as
a new accepted transition.

The current frame-18 evidence is nevertheless a separate runtime failure:
frame 18 and later rows publish pose-certificate rejection and a safe-stop
latch. This design does not waive or normalize that failure. The candidate
visualizer remains blocked until a separate test-first runtime repair restores
accepted forward progress.

## Files and Ownership

Modify only:

- `resources/check_g1_runtime_log.py`
- `tests/python/test_runtime_log.py`
- `tests/cpp/test_g1_controller_logging.cpp`

`motion_match_log.h`, `controller.cpp`, `g1_frame_transaction.h`, provider
files, terrain/database assets, and the candidate trace remain byte-unchanged.

The C++ logging test replaces only the obsolete checker hash lock. It retains
the exact `motion_match_log.h` schema hash lock and adds an authentic
producer-to-log unscheduled recovery fixture proving
`matching_enabled && !searched && transitioned`, the strict winner frame/cost,
and checker acceptance.

## Test-First Verification

RED Python tests cover:

1. acceptance of an exact matching-enabled unsearched transition;
2. rejection of the same row with matching disabled;
3. rejection of a legacy search in matching-disabled mode;
4. rejection of equal or one-to-four-ULP unscheduled-recovery costs;
5. preservation of the searched-transition four-ULP legacy allowance;
6. preservation of selected-frame, current-frame, range, and cost failures; and
7. a rejected rollback row that repeats the last accepted recovery baseline
   without being counted as a new transition event.

RED C++ coverage constructs the authentic unscheduled recovery through the
production transaction and log builder. It must fail only because the old
checker text/hash contract cannot represent the row.

GREEN verification includes focused Python and C++ tests, strict and fast
logging/frame-production suites, ASan/UBSan, no-seam/privacy and immutable
owner checks, the complete native matrix, all 386 Python tests, and unchanged
terrain/database assets.

## Gate Continuation

After review, rebuild and rerun the same exact 18/32 paired gate. The canonical
checker must pass without data normalization. The original exact frame-17
trace/publication assertions remain unchanged.

The expected next stop is the already-observed frame-18 rejection/latch. That
failure is diagnostic input to a separate runtime design; it cannot be turned
into PASS by this logging repair. No candidate window launches until both the
checker grammar and runtime liveness gates pass independent review.

## Constraint Amendment

The earlier recovery plan froze `resources/check_g1_runtime_log.py` because it
assumed `transitioned => searched` remained universally true. The live gate
disproved only that implication. This design supersedes the checker
immutability clause for the exact mode-aware grammar above. The CSV schema,
motion log writer, controller, physical behavior, thresholds, timing, terrain,
database, and process-safety constraints remain unchanged.
