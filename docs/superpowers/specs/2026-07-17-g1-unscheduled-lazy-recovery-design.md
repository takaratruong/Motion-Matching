# G1 Unscheduled Lazy Recovery Design

**Date:** 2026-07-17

**Status:** Approved for implementation planning

## Problem

The bounded-candidate runtime passes its focused native, sanitizer, privacy,
and same-build oracle gates, but the authentic low-curb certification exposes
a missing recovery route.

At presentation frame 17 of the canonical `grail-curb-low / curb-forward`
run, the unscheduled incumbent continuation `117711 -> 117712` passes common
and raw certification. Hidden IK then exhausts the fixed 41-entry swing ladder
for the releasing left foot and finitely rejects with
`G1IkStopNoSwingCandidate`. Because no legacy search was scheduled, the
current coordinator publishes that first failure without calling the bounded
recovery provider. Safe-stop latches, the accepted state rolls back to frame
`117711`, and later forced searches repeatedly exhaust their authentic
bounded sets.

The failure is reproduced by the no-seam release build. IK-off and IK-on
candidate traces are byte-identical, all accelerated sets equal the
same-process exhaustive sets, and rollback is atomic. This is therefore an
interface gap: matching-enabled unscheduled candidate failures have no route
to the already-certified recovery provider.

## Goal

Allow a matching-enabled, unscheduled incumbent that finitely fails candidate
certification to invoke the existing strict bounded provider once in the same
transaction, without performing another legacy search or weakening any
physical, IK, clearance, timing, determinism, or publication contract.

## Global constraints

- Runtime frequency remains exactly 25 Hz.
- Travel direction and heading remain independent; no rotation toward travel
  direction is introduced.
- Common, raw, and hidden-IK dual certification remains mandatory before a
  candidate can commit.
- No IK reach, swing-lift, contact, clearance, walkability, or residual
  threshold changes.
- No raw-only fallback, pose hold, or acceptance after a finite IK failure.
- No candidate-capacity increase, lookahead search, second legacy traversal,
  database rebuild, or search-period change.
- Matching-disabled sequential mode never calls the recovery provider.
- The provider remains lazy: an accepted slot zero performs no provider
  traversal or exhaustive-oracle work.
- The existing transaction attempt capacity of eight remains unchanged.
- The same in-memory database and exact 31-word query feed accelerated and
  post-transaction exhaustive recovery in dual-seam builds.
- Normal production builds expose no trace or exhaustive-oracle interface and
  retain the existing production CSV schema.
- The currently running visualizer is not inspected or touched until the
  complete live certification and packaging phase succeeds.

## Considered approaches

### 1. Lazy same-transaction unscheduled recovery — selected

Prepare an authenticated request for every matching-enabled frame, but invoke
the provider only after slot zero finitely rejects. A scheduled frame retains
its legacy slot zero and current recovery path. An unscheduled frame retains
its incumbent-owned slot zero; if that record fails, the request binds both
the incumbent and legacy-selected exclusions to the same incumbent frame.

This preserves the current trajectory on successful frames, adds no second
database search, and reuses the certified strict provider only at the failure
boundary.

### 2. Force legacy matching every frame — rejected

An 18-frame diagnostic with `MM_SEARCHT=0` diverged immediately from the
canonical trajectory and still rejected at earlier frames 4 and 9. It never
reached the canonical frame-17 state, increases ordinary traversal work, and
does not isolate the missing failure route.

### 3. Relax IK or accept raw state after IK failure — rejected

This would bypass the physical condition the system is intended to certify.
It conflicts with mandatory dual certification and would turn a visible
design gap into an unproven fallback.

Increasing the recovery capacity or adding lookahead is also out of scope.
Those options may be considered only if the exact canonical frame-17 request
proves that the existing strict bounded tail contains no dual-certified
candidate.

## Architecture

The change spans three existing ownership boundaries:

1. `controller.cpp` prepares the exact recovery request from the matcher
   prefix without performing provider work.
2. `g1_frame_transaction.h` authenticates the request and decides whether the
   provider may run after a finite slot-zero outcome.
3. `g1_candidate_certification_trace.h` recognizes and serializes the second
   legal request grammar while retaining exact accelerated/exhaustive
   comparison.

The strict provider implementation and its capacity remain unchanged unless
test-first work demonstrates that it rejects the already-supported
`legacy_selected_frame == incumbent_frame` request form.

## Matcher-prefix request preparation

`G1FrameStageMatcherSearch` continues to build the exact 31-component query,
incumbent cost, terrain error, transition cost, and scheduled-search decision
once per transaction.

`database_search` remains conditional on `state.searched`. No new search is
performed on an unscheduled frame.

For every matching-enabled frame, the prefix prepares one
`G1RecoveryRequest` with:

- `db` equal to the live `G1FrameExternalInputs::db` pointer;
- all 31 `raw_query` words copied bit-exactly from the matcher query;
- `incumbent_frame` equal to the immutable pre-candidate frame;
- `legacy_selected_frame` equal to the scheduled legacy result when a search
  ran, otherwise equal to `incumbent_frame`;
- the exact transition cost and public incumbent cost; and
- the existing range-end and surrounding-exclusion values.

The prefix flags have independent meanings:

- `matching_scheduled` and `legacy_search_performed` are true only when the
  legacy search ran;
- `recovery_request_ready` is true for every matching-enabled frame and false
  for matching-disabled sequential mode.

Preparing the request performs no provider traversal, exhaustive comparison,
allocation proportional to the database, publication, or state mutation
beyond the existing scratch object.

## Coordinator behavior

Slot-zero ownership does not change:

- scheduled search: `G1CandidateLegacy / G1CandidateScoreLegacy`;
- unscheduled matching-enabled frame: `G1CandidateIncumbent /
  G1CandidateScoreIncumbent`;
- matching-disabled frame: incumbent ownership with no recovery permission.

The coordinator evaluates slot zero exactly once.

- Global error remains a global error and never invokes recovery.
- Dual acceptance commits immediately and performs zero provider work.
- Finite rejection retains the exact earliest failure.
- Matching-disabled finite rejection publishes that failure with zero
  provider work.
- Matching-enabled finite rejection authenticates the prepared request and
  calls the provider exactly once.

Request authentication has two legal forms.

### Scheduled form

This is the existing form: a legacy search occurred, slot zero is legacy
owned, the request incumbent equals the immutable baseline frame, and the
request legacy selection equals the legacy slot-zero frame.

### Unscheduled form

No legacy search occurred, slot zero is incumbent owned and untransitioned,
and all of these frames are identical:

- immutable baseline frame;
- slot-zero selected frame;
- request incumbent frame; and
- request legacy-selected frame.

The unscheduled request must also bind the same source range, incumbent cost,
transition cost, database pointer, and 31 query words produced by the prefix.
Any mismatch is a global integrity error, not a candidate rejection.

The existing provider already deduplicates equal incumbent and
legacy-selected exclusions. Therefore an unscheduled request can return at
most the six strict recovery transitions and does not append a duplicate
incumbent tail record. Slot zero plus that tail uses at most seven of the
existing eight attempt slots. Scheduled behavior retains its existing maximum
of slot zero plus seven tail records.

Tail records are evaluated in exact provider order. The first dual-certified
record commits atomically. If every record finitely rejects, the coordinator
publishes the original earliest slot-zero failure. Provider failure or an
invalid set remains a global transaction error.

## Trace and oracle grammar

The dual-seam trace keeps one slot-zero row plus every materialized tail row,
including unattempted records after success. Formatting, fixed-buffer
transactionality, alias protection, enum validation, and single-write
behavior remain unchanged.

There are exactly two legal recovery-row grammars:

- scheduled legacy recovery: `legacy_traversals=1`, legacy slot-zero owner,
  one provider call, one recovery traversal;
- unscheduled incumbent recovery: `legacy_traversals=0`, incumbent slot-zero
  owner, one provider call, one recovery traversal, and equal request
  incumbent/legacy-selected/slot-zero frames.

A slot-zero acceptance still requires zero provider calls and zero recovery
traversals. Matching-disabled execution cannot produce a request or set.
Request-without-set, set-without-request, duplicate records, invalid attempted
prefixes, counter drift, score-owner drift, hidden-slot data, and incomplete
accelerated/exhaustive equality all remain serialization failures before any
block write.

The exact live request is passed unchanged to
`g1_recovery_candidates_exhaustive_for_test` only after the transaction has
finished. The oracle remains evidence-only and cannot influence selection or
state.

## State, failure, and performance behavior

Successful recovery commits through the existing candidate state swap and
accepted-finalize path. The rejected unscheduled incumbent leaves no partial
state behind. Safe-stop is not latched when a later record accepts.

If the bounded tail exhausts, the original incumbent rejection remains the
published first failure and rollback/latch behavior is unchanged. The change
does not turn a finite physical failure into a global error.

Ordinary successful frames pay only the fixed cost of preparing a small
request already derived from the existing query. They do not traverse the
provider. Benchmark timing remains around only
`g1_frame_transaction_run`, so the live three-trial performance gate measures
the rare lazy recovery cost honestly.

## Test strategy

Implementation is strictly test-first.

### Generic coordinator RED tests

- A matching-enabled unscheduled incumbent passes common/raw, fails hidden
  IK, and a strict tail record dual-accepts. Require incumbent slot-zero score
  ownership, zero legacy traversals, one provider call, one recovery
  traversal, exact two-attempt prefix, atomic commit, and no public failure.
- An accepted unscheduled incumbent performs zero provider work.
- A matching-disabled finite incumbent rejection performs zero provider work
  and publishes the original failure.
- An unscheduled provider global error remains global and cannot publish or
  commit candidate state.
- A fully rejected unscheduled strict tail preserves the original slot-zero
  earliest failure and respects the unchanged attempt bound.
- Hostile unscheduled requests and sets fail authentication for frame, cost,
  query, owner, ordering, duplicate, rank, and counter mismatches.

### Production and trace RED tests

- The real matcher prefix prepares the unscheduled request without invoking
  `database_search` twice or changing slot-zero ownership.
- The production coordinator accepts the equal-frame request form only for
  matching-enabled unscheduled execution.
- The trace serializes a finite incumbent slot zero followed by its exact
  strict tail and same-build exhaustive comparison.
- Mutating either equal request frame, traversal counter, score owner, cost
  word, database pointer, or query word fails before writing a transaction
  block.
- Normal no-seam preprocessing and production `nm`/`strings` checks continue
  to expose no trace or exhaustive-oracle surface.

### Regression and live gates

After focused strict, fast, sanitizer, parity, negative-context, and privacy
tests pass, build disposable live binaries and rerun the canonical 18/32-frame
pair.

The frame-17 transaction must prove:

- incumbent-owned slot zero with zero legacy traversals;
- common/raw acceptance and finite hidden-IK rejection on slot zero;
- exactly one authenticated provider and accelerated traversal;
- complete accelerated/exhaustive equality;
- a bounded attempted prefix ending in the first dual-certified winner;
- the public selected frame and exact cost owner match that winner in both IK
  modes; and
- no rejection or safe-stop latch in any of the 32 paired rows.

The first successful canonical run freezes its exact frame-17 transcript and
hash as regression evidence. If the exact request contains no dual-certified
tail record, implementation stops without changing capacity or physical
rules, and the broader design question returns for review.

Only after the 32-frame marker passes may the complete Task 6 sequence restart
from Step 1: exact low-curb 800-row logs, three 832-transaction timing trials,
the normal/stress/blocked Gate E matrix, and the complete Gate L matrix and
auxiliary gates.

## Acceptance criteria

- All new RED tests fail for the missing unscheduled recovery route before
  production code changes.
- Focused tests pass in strict, fast, and sanitizer builds with exact parity.
- Existing scheduled recovery, accepted-unscheduled, matching-disabled,
  publication, and privacy tests remain green.
- Canonical frame 17 recovers in the same transaction without a second legacy
  search or a safe-stop latch.
- The paired 32-frame prerequisite creates `ZERO_REJECTION_32.PASS` only after
  every exact assertion passes.
- The complete Task 6 performance, Gate E, and Gate L certification passes
  without threshold or checker changes.
- Repository scope contains only the design, implementation plan, and the
  explicitly planned source/test changes; live evidence remains under `/tmp`.
- The running visualizer remains untouched until the separately reviewed,
  certified package is ready to replace it.

## Non-goals

- General candidate lookahead or arbitrary retry loops.
- More than one provider call per transaction.
- Increasing the six-transition strict tail or eight-attempt transaction
  capacity.
- Altering GRAIL motion data, terrain assets, scene routes, or feature weights.
- Changing search cadence, controller input semantics, heading semantics, or
  simulation frequency.
- Relaxing IK, foot clearance, footprint, contact, slope-normal, or
  multilevel requirements.
- Packaging, visualizer restart, or G1 mesh rendering in this repair itself.
