# G1 Accepted Selection Provenance Design

**Date:** 2026-07-17

**Status:** Approved under the user's standing instruction to continue and
auto-approve in-scope work. This is a narrow evidence-contract repair required
before the candidate visual checkpoint.

**Base:** `2f61b2691273227bad31eec0b791ec277d6184f0`

## Context

The out-of-line trace repair passed its complete implementation gate and
independent review. The rebuilt fast trace and no-seam release binaries then
passed all six byte-parity comparisons for 18- and 32-frame IK-off/IK-on runs.
The release binary also retained its expected pre-repair SHA-256 and contained
no trace or oracle symbols or strings.

The canonical production-log checker stopped the live gate at row 17:

```text
ValueError: row 17: transition without search
```

The row is an authentic unscheduled recovery, not an invalid transition:

- query/incumbent frame `117711`, source range `402`;
- `searched=0`, because the legacy `database_search` schedule did not run;
- incumbent hidden IK finitely rejected with `no-swing-candidate`;
- one strict provider call returned five candidates;
- rank-zero strict recovery frame `111739` dual-certified;
- the accepted pose advanced to frame `111740`, source range `378`;
- `transitioned=1`, selected cost bits `40cfb344`;
- trace/release runtime CSVs are byte-identical in both IK modes; and
- the accelerated and exhaustive provider results are equal.

The checker still encodes the pre-recovery implication
`transitioned => searched`. That implication is no longer true: `searched`
means that the one scheduled legacy `database_search` ran, while
`transitioned` means that the accepted candidate changed source frame.

No existing production CSV field authenticates which candidate kind owns the
accepted selection. The dual-seam TSV has this information, but the no-seam
release log does not. A generic exception based only on
`matching_enabled`, frame differences, or costs would weaken the production
checker.

## Objectives

1. Preserve the exact meaning of `searched`: it is true only when the existing
   scheduled legacy `database_search` ran.
2. Preserve the exact meaning of `transitioned`: the accepted candidate changed
   source frame.
3. Carry the already-authenticated accepted `G1CandidateKind` through the
   accepted diagnostic into the production CSV.
4. Make the production checker accept an unsearched transition only when the
   row self-authenticates a strict-recovery selection.
5. Fail closed on missing, unknown, or inconsistent selection provenance.
6. Re-run the repaired 18/32 live gate before launching the separate candidate
   visualizer.

## Non-goals

- No matcher, provider, database-search, query, cost, candidate ordering, IK,
  terrain, threshold, timing, heading, route, rendering, or mesh change.
- No new recovery search and no relabeling of strict recovery as a legacy
  search.
- No change to the seam-only candidate trace schema or oracle.
- No interaction with the existing visualizer.
- No long certification or G1 mesh work in this repair.

## Considered Approaches

### 1. Accepted selection-kind provenance — selected

Persist the accepted `G1CandidateKind` as an enum in
`G1FrameAcceptedDiagnostic`, map it to one of three canonical CSV strings, and
validate the complete relationship among candidate kind, legacy search,
transition, matching enablement, query frame, and selected frame.

This is self-contained, uses an authenticated production owner, covers all
accepted candidate paths, and makes malformed release logs distinguishable
without a test seam.

### 2. Recovery-transition boolean — rejected

A boolean would repair the immediate implication but would not authenticate
legacy versus incumbent ownership on other rows. It creates a second partial
state machine beside the existing candidate enum.

### 3. Checker exception or seam-TSV bridge — rejected

Allowing any matching-enabled unsearched transition, or normalizing the CSV
only when a paired seam trace exists, would not make a no-seam production log
self-authenticating. It also weakens the canonical checker and cannot support
later long certification independently.

## Production Data Flow

The authenticated candidate already exists as
`G1FrameTransactionScratch::active_candidate` during
`G1FrameStageAcceptedFinalize`.

The accepted path will copy only its `kind` into a new
`G1FrameAcceptedDiagnostic::selection_kind` enum field. The existing accepted
diagnostic validation, equality, success binding, candidate commit, rollback,
reset, copy, and observation-relation checks will include that field.

`g1_build_accepted_log_row` will map the enum to exactly one canonical string:

| Enum | CSV `selection_kind` |
| --- | --- |
| `G1CandidateLegacy` | `legacy` |
| `G1CandidateRecoveryTransition` | `strict-recovery` |
| `G1CandidateIncumbent` | `incumbent` |

Unknown enum values are controlled log-construction errors. The mapping is a
production helper, not a test-seam helper.

The CSV column is placed immediately after `transitioned`, because it explains
the ownership of `selected_database_frame`, `searched`, and `transitioned`.
The log row struct, canonical header, formatter, row equality/hash ownership,
schema constants, and test fixtures all gain the field together.

## Accepted-Selection Grammar

For every accepted diagnostic, C++ production validation and the Python checker
enforce the same grammar:

1. `selection_kind` is exactly `legacy`, `strict-recovery`, or `incumbent`.
2. `selected_database_frame == query_database_frame` if and only if
   `transitioned == 0`.
3. `legacy` requires `matching_enabled=1` and `searched=1`. It may transition or
   retain the incumbent, according to the selected/query frame relation.
4. `strict-recovery` requires `matching_enabled=1` and `transitioned=1`.
   `searched` may be either value because strict recovery may follow either a
   scheduled legacy candidate or an unscheduled incumbent candidate.
5. `incumbent` requires `transitioned=0` and the selected/query frames to be
   equal. It may follow an unscheduled prefix, a scheduled recovery tail, or
   matching-disabled sequential mode.
6. `matching_enabled=0` requires `searched=0`, `transitioned=0`, and
   `selection_kind=incumbent`.
7. Consequently, `transitioned=1 && searched=0` is legal only when
   `matching_enabled=1 && selection_kind=strict-recovery`.

The existing cost, range, query snapshot, post-advance frame, lifecycle,
rejection, and finite-value rules remain in force.

## Rejected-Frame Semantics

The production CSV is an accepted-state log plus current publication status.
When a transaction finitely rejects, the publication frame and rejection
diagnostic advance, while accepted state and accepted diagnostic remain the
last committed snapshot.

Therefore `selection_kind` describes the accepted snapshot, not the rejected
attempt. A rejected row may repeat the last accepted selection kind,
`searched`, and `transitioned` values. This matches the existing
`row_rejected_for_lifecycle` rules and does not claim that a provider ran on the
rejected presentation frame. Rejected-attempt provenance remains in the
existing rejection columns.

## Failure Handling

- An invalid candidate enum in an accepted diagnostic is a global transaction
  integrity error before publication.
- An invalid enum reaching log construction is a controlled runtime log error.
- Missing, unknown, or inconsistent CSV provenance is a checker error.
- Any strict/fast, trace/release, IK-off/on, or frame-17 mismatch blocks the
  candidate launch.
- The old visualizer is never inspected, signaled, or replaced.

## Test-First Verification

The implementation starts with RED tests that demonstrate all current blind
spots:

1. Python rejects the authentic row-17 shape when `selection_kind` is absent,
   unknown, `legacy`, or `incumbent`, and accepts it only as
   `strict-recovery` with matching enabled.
2. Python rejects every invalid complementary combination, including legacy
   without a legacy search, strict recovery without a transition, incumbent
   with a transition, and any matching-disabled non-incumbent selection.
3. Python accepts a rejected rollback row that repeats the last accepted
   strict-recovery snapshot while preserving its rejection lifecycle.
4. C++ production tests prove candidate kind is copied from
   `scratch.active_candidate`, authenticated by accepted-diagnostic
   validation/equality/success binding, and mapped fail-closed into the log.
5. Logging tests prove exact header order, values, formatting, equality, hashes,
   and I/O failure behavior.

Then run:

- focused strict and fast frame-transaction and logging suites;
- no-seam production links and negative provenance mutations;
- ASan/UBSan and privacy checks;
- the complete native matrix;
- all 386 Python tests, adjusted only for the intentional one-column schema
  revision;
- immutable terrain/database checks; and
- exact trace/release and IK-off/on parity.

The candidate trace transcript remains byte-frozen. The release binary hash and
production CSV hashes intentionally change because the release now emits one
new authenticated field; the repaired gate records and independently reviews
their new exact values.

## Repaired Visual Gate

Rebuild trace and no-seam release from the reviewed provenance commit. Run the
same finite 18/32 IK-off/on matrix. Require:

- exact trace/release runtime CSV equality for all four pairs;
- exact IK-off/on candidate trace equality for 18 and 32 frames;
- canonical production checker success without normalization or exception;
- zero frame rejection and safe-stop latch in the intended accepted window;
- the original exact frame-17 incumbent, provider, oracle, first-winner, public
  frame, and public cost assertions; and
- `selection_kind=strict-recovery` on the accepted unscheduled transition and
  any later rejected rollback row that repeats that accepted snapshot.

Only after fresh independent evidence review may a separate no-seam candidate
window launch on `mixed-multilevel`. The directly captured candidate PID is the
only process identity that may be read; no existing process or window is
queried.

## Constraint Amendment

The earlier recovery design froze `motion_match_log.h`,
`resources/check_g1_runtime_log.py`, and the CSV schema because it assumed
`transitioned => searched` remained universally valid. The exact live gate
disproved that assumption. This design supersedes only that immutability clause
and only for accepted selection provenance. All motion, terrain, database,
threshold, renderer, process-safety, and timing constraints remain unchanged.
