# G1 Root-Reach Boundary-Frontier Amendment

## Status and Amendment Boundary

This document freezes the implementation contract for enlarging the existing
G1 root-reach planner's deterministic boundary frontier. It amends only the
planner-wide adjacent-candidate limit and the matching test-audit capacity in:

- `docs/superpowers/specs/2026-07-16-g1-ik-root-reach-preconditioner-design.md`; and
- `docs/superpowers/plans/2026-07-16-g1-ik-root-reach-preconditioner.md`.

The new planner-wide limit is **64 production revalidations per plan**. The
test-only audit stores **64 attempts**. Every other requirement in the prior
design and plan remains controlling, including the analytic proof domain,
nearest-first cursor ordering, one-ULP inward cursor advancement, strict
production revalidation, plan validation, transactional publication,
fail-closed errors, strict/fast parity, runtime ownership, and the 5 cm root-Y
cap.

This is a bounded false-negative correction, not a new IK strategy. It does
not change contact timing, terrain targets, reach shells, correction limits,
swing search, matcher selection, safe-stop semantics, logs, terrain packs, or
the visualizer.

## Decision

Use one global frontier of at most 64 adjacent binary32 candidates across all
common analytic intervals.

The implementation changes the private production ceiling from 32 to 64 and,
under `G1_IK_ENABLE_TEST_SEAMS`, changes
`G1RootReachPlannerAudit::attempts` from 32 entries to 64 entries. It does not
add a command-line option, tuning parameter, dynamic allocation, public
runtime state, or production-only diagnostic.

Sixty-four is sufficient for both newly authenticated live frontiers while
remaining a small, exact work bound. A candidate which would require a 65th
or later production revalidation remains unavailable and publishes the
existing canonical active/non-common/unapplied positive-zero plan.

## Live Evidence

The unchanged strict root-reach kernel was replayed against the exact
support-retargeted state at the first low-curb failure and its following
safe-stop recovery frame. The existing 32-attempt implementation exhausts
before reaching either authentic candidate.

| Runtime row | Recorded contacts | Initial admitted bits | First accepted bits | Attempt count | Accepted delta |
| --- | --- | --- | --- | ---: | ---: |
| 6 | left swing, right planted | `0xbbd51d13` | `0xbbd51d41` | 47 | `-0.006503731478005648 m` |
| 7 | left planted, right planted | `0xbbc6ab9b` | `0xbbc6abc1` | 39 | `-0.006062955129891634 m` |

The attempt count is inclusive. Row 6 tries the exact sequence
`0xbbd51d13 + i` for `i = 0..46`: attempts 1 through 46 reject and attempt 47
accepts. Row 7 tries `0xbbc6ab9b + i` for `i = 0..38`: attempts 1 through 38
reject and attempt 39 accepts. These are negative finite binary32 values, so
incrementing the bit pattern by one advances the cursor one representable
value farther from positive zero and toward the negative interval interior.

The row-6 right-foot analytic interval is approximately
`[-0.9560491357, -0.006503709817] m`, capped to
`[-0.05, -0.006503709817] m`. The initially materialized candidate is
`0xbbd51d13` (`-0.0065037100575864315 m`). Production revalidation rejects
the closer values because rounded checked FK and physical-target
reconstruction do not produce an exact reachable target until
`0xbbd51d41`.

At row 7, the left and right analytic upper bounds are approximately
`-0.005621271295 m` and `-0.006062937043 m`; their common interval therefore
starts at the right-foot constraint. The initially materialized candidate is
`0xbbc6ab9b` (`-0.006062937434762716 m`) and the first production-authenticated
candidate is `0xbbc6abc1`.

Forcing only the authentic row-7 plan in a source-free debugger replay lets
that frame accept with finite physical-sole residuals of approximately
`0.00239372184 m` left and `0.0010136147 m` right, then advances matching.
This demonstrates that the old ceiling caused the row-7 planner rejection.

The row-6 root plan is also authentic, but row 6 subsequently exhausts the
unchanged 41-entry swing solve because every swing candidate fails a
controller constraint. The 64-attempt amendment therefore does not claim to
fix row 6's separate no-swing-candidate failure or to make the complete route
green. That matcher/swing problem requires its own evidence and design.

## Amended Normative Contract

### Global Budget

The planner owns one compile-time production constant with value 64. One
candidate revalidation consumes one unit of this budget, regardless of which
analytic interval cursor supplied the candidate and regardless of whether
one or two feet are recorded contacts.

The budget is planner-wide:

- at most four sorted common analytic intervals create at most four cursors;
- all cursors share the same 64-attempt counter;
- no cursor receives a private allowance;
- an accepted candidate ends the search immediately; and
- no candidate receives a 65th production revalidation or audit record.

Preserve the current loop-body semantics after a rejected 64th revalidation:
the selected cursor may advance once with `std::nextafter`, analytically
bounds-check that value, and retain it as an unconsumed cursor head before the
loop exits. That post-rejection advancement is not a candidate attempt. It
runs no strict FK, physical-target reconstruction, reach projection, or audit
write for the new head.

Cursor construction is unchanged. For each sorted interval, positive zero is
materialized when contained; otherwise the endpoint nearest positive zero is
rounded to binary32 and, when necessary, moved one ULP toward that interval's
interior until it is analytically admitted.

### Ordering and Advancement

Before every attempt, select the active cursor whose current candidate is
first under the existing total ordering:

1. smaller absolute binary32 delta promoted to binary64;
2. smaller analytic interval lower bound;
3. smaller analytic interval upper bound; and
4. smaller original sorted interval index.

After a finite production rejection, advance only the selected cursor by one
`std::nextafter` toward its analytic interval interior: toward negative
infinity for a negative interval and toward positive infinity for a positive
interval. Deactivate that cursor if its next value is non-runtime, outside
the closed analytic interval, or outside the exact promoted binary32 5 cm
cap. A rejected positive-zero cursor is deactivated because it has no single
signed inward direction.

Changing the ceiling does not permit skipping a representable float,
restarting a cursor, widening an interval, probing outside the analytic proof
domain, or translating already-computed globals.

### Production Authentication

Every one of the at most 64 candidates passes through the unchanged strict
production authentication path:

1. copy the fixed 31-bone baseline local-position array;
2. apply the proposed delta only to local `G1_Simulation.y` through
   `g1_apply_root_reach_plan_y`;
3. rerun the same checked strict FK over all 31 bones;
4. rederive each recorded foot's physical ankle target from the adjusted
   contact origin, contact rotation, and ankle origin while retaining the
   original desired sole center and normal; and
5. run the named production reach projection and require both `reachable`
   and bit-exact equality between its clamped target and the rederived ankle
   target for every recorded foot.

Only a candidate satisfying the baseline analytic domain and all applicable
production projections is publishable. Malformed arithmetic remains a global
error. An ordinary finite rejection consumes one attempt. Exhausting the
budget remains a successful planner call which publishes the canonical
active/non-common/unapplied positive-zero unavailable plan.

### Audit and Transactionality

Under `G1_IK_ENABLE_TEST_SEAMS`, the audit layout becomes:

```cpp
struct G1RootReachPlannerAudit {
    G1RootReachAuditCursor cursors[4];
    G1RootReachAuditAttempt attempts[64];
    uint32_t cursor_count;
    uint32_t attempt_count;
};
```

The ordinary and audited wrappers continue to call one private planner. The
audit records every materialized cursor and every real production
revalidation in global order. Its attempt count is never greater than 64.
The implementation checks capacity before each write and uses a local audit
candidate. Invalid status, alias, invalid input, or late arithmetic failure
returns false without assigning either the caller's plan or the caller's
audit. The full-capacity guard also remains fail-closed: if the private audit
recorder is ever called while `attempt_count == 64`, it returns false before
writing or publishing.

The production ceiling and audit capacity must have one compile-time
consistency assertion, and the production loop condition remains strictly
less than that ceiling. Consequently, a conforming ordinary or audited plan
cannot production-revalidate or audit-record a 65th attempt; runtime tests do
not pretend to synthesize one through the public API. The permitted
post-rejection cursor advancement does not increment `attempt_count`. A
future change to the loop bound or capacity which violates their equality
must fail to compile. A no-seam object retains no audit declaration, audit
storage, or audit symbol.

## Why Adjacent Enumeration Preserves the Proof

The global frontier proves nearest selection without assuming that production
authentication is monotone. At any attempt, each active cursor points to its
nearest not-yet-rejected analytically admitted binary32 member. The total
ordering selects the nearest of those heads. Advancing only the rejected
cursor by one inward ULP means every admitted candidate which precedes a
published candidate has already been authenticated and rejected. Therefore a
candidate accepted on or before attempt 64 is the nearest dual-certified
candidate under the existing deterministic tie-breaks.

If all 64 attempts reject, the planner proves only that no candidate in the
production-revalidated prefix authenticates. It deliberately does not claim
that the rest of a nonempty analytic interval is impossible. The canonical
unavailable plan is the existing bounded-search outcome for that case.

## Rejected Alternatives

### B. Exponential Bracketing Followed by Binary Refinement

This alternative is rejected because it needs a monotone accept/reject
predicate to discard the values between probes. The baseline analytic shell
is monotone within each connected interval in ideal arithmetic, but the
production predicate is not proved monotone in binary32 execution:

- applying root Y changes the local root through rounded binary64 addition
  and binary32 commit;
- checked FK performs separately rounded parent/child operations across the
  31-bone hierarchy;
- descendant contact and ankle origins can change by different stepwise
  amounts rather than an exact common translation;
- the contact-origin-to-ankle-origin offset is reconstructed after that FK;
- the physical ankle target is then reconstructed from those rounded values;
  and
- reach projection recomputes the effective shell and requires bit-exact
  target equality.

Those operations can create isolated acceptance or rejection holes. Observed
monotonicity in two runtime rows would not establish the contract needed for
binary pruning. Bracketing could skip a closer authentic candidate and break
the nearest-plan proof.

### C. Proof-Carrying Interval Subdivision

This alternative is rejected as disproportionate to the bounded defect. A
sound subdivision would have to derive every binary32 breakpoint for checked
root addition, the complete 31-bone FK, physical-target reconstruction, and
reach projection, then prove predicate constancy inside each cell. That would
duplicate the named production solver in a second numerical system, expand
the proof and test surface substantially, and risk numerical divergence. It
does not improve the current 64-revalidation worst-case bound enough to
justify that ownership cost.

## Acceptance Tests

Implementation begins with failures for the following exact contracts.

1. **Live row 6 frontier.** Materialize the captured support-retargeted local
   pose, parents, recorded contacts, and exact targets from row 6 by binary32
   bits. Require one global cursor beginning at `0xbbd51d13`, exactly 47
   attempts with candidate bits `0xbbd51d13 + i` for `i = 0..46`, rejected
   status for `i = 0..45`, accepted status for `i = 46`, and a published
   active/common/applied plan with delta bits `0xbbd51d41`. Independently
   rerun the production authentication for the published delta.
2. **Live row 7 frontier.** Materialize the captured row-7 fixture by bits.
   Require one global cursor beginning at `0xbbc6ab9b`, exactly 39 attempts
   with candidate bits `0xbbc6ab9b + i` for `i = 0..38`, rejected status for
   `i = 0..37`, accepted status for `i = 38`, and a published
   active/common/applied plan with delta bits `0xbbc6abc1`. Independently
   authenticate both recorded-foot projections and their unchanged promoted
   `<= 0.005f` physical-sole residual rule.
3. **One-cursor exhaustion.** Extend the established high-level
   projection-refusal fixture. Require exactly 64 rejected attempts from
   `0x3c8b379d` through `0x3c8b37dc`, followed by the canonical unavailable
   plan. Direct strict-kernel probes, rather than an inference from interval
   math, establish that `0x3c8b37dd`, the out-of-budget 65th candidate,
   rejects and that `0x3c8b381d`, offset 128 from the initial candidate,
   accepts. Freeze both results as diagnostic assertions executed outside the
   production frontier loop. Following rejection of production attempt 64 at
   `0x3c8b37dc`, the planner may materialize and analytically bounds-check
   `0x3c8b37dd` as the next unconsumed cursor head. It must not
   production-revalidate or audit-record either out-of-budget value.
4. **Representational no-op exhaustion.** Extend the existing single-cursor
   fixture to exactly 64 rejected candidates `0xb2a00000 + i` for
   `i = 0..63`. Every attempted nonzero plan must still fail checked root
   application without changing its poisoned output, and planning must
   publish the canonical unavailable plan.
5. **Two-cursor exhaustion.** Extend the symmetric high-level fixture to
   exactly 64 globally alternating rejections. For attempt `i`, require
   cursor `i & 1`, magnitude bits `0x3c230dce + floor(i / 2)`, the sign bit
   set only for cursor 0, and rejected status. This proves one shared budget
   rather than 64 attempts per cursor.
6. **Existing short traces.** Retain the four-attempt authentic low-curb
   fixture and the reject-then-other-cursor two-attempt fixture byte for byte.
   Enlarging the ceiling must not change their selected plans or traces.
7. **Capacity and rollback.** Require compile-time equality of production
   ceiling and audit capacity and require the loop condition to be strictly
   below that common bound. Exact exhaustion tests must return success with
   `attempt_count == 64`, never 65. Retain poisoned-output tests for invalid
   shape, topology, aliasing, error-buffer overlap, nonfinite arithmetic,
   and late failure. Retain the private recorder's invalid-status and
   pre-write `attempt_count >= capacity` rejections as reviewed fail-closed
   invariants; do not add a public fault-injection seam solely to invoke an
   unreachable recorder state or a 65th write.
8. **Frame ownership.** Prove the accepted row-7 plan retains root X/Z and all
   non-root local positions bit for bit, changes only candidate root Y, and
   leaves support state, command intent, travel direction, heading, matcher
   inputs, and persistent locks unchanged. Separately retain row 6's
   no-swing-candidate result so the amendment cannot hide that independent
   blocker.
9. **Build matrix.** Run the focused strict caller, optimized
   `-ffast-math` caller linked to the strict kernel, strict/fast parity record,
   ordinary no-seam build, negative seam compilation, ASan, UBSan,
   float-divide-by-zero, and float-cast-overflow surfaces. Require identical
   strict/fast plans and audit records and no sanitizer report.
10. **Live bounded regression.** Rebuild a disposable controller and replay
    the exact low-curb command first for 8 frames and then for the inherited
    800-frame diagnostic horizon. Require no row-7 root-planner
    `target-unreachable` caused by frontier exhaustion. Continue to report the
    row-6 swing rejection independently; this amendment does not waive it or
    convert the route into a false pass.

## Performance Contract

Worst-case planner work increases from 32 to 64 strict candidate
revalidations. Each revalidation uses fixed stack storage for 31 local
positions, 31 global positions, and 31 global rotations; copies 31 local
positions; runs one checked 31-bone FK; and runs physical-target derivation
plus reach projection for at most two recorded feet. The implementation adds
no heap allocation, I/O, database search, terrain query, swing iteration, or
persistent state.

Performance acceptance is structural and runtime-based:

- audit evidence must prove no plan executes more than 64 revalidations;
- production and no-seam objects must contain no dynamic-allocation call
  introduced by this change;
- the exact 64-rejection fixtures must complete under the sanitizer matrix;
  and
- the disposable controller must complete the inherited 800-frame exact
  25 Hz diagnostic run without a watchdog timeout or a new controller error.

No absolute microsecond threshold is introduced because the repository has no
authoritative per-frame CPU timing oracle and CI host timing is not stable.
The exact operation bound above is the normative performance limit. The new
worst case is at most twice the candidate-revalidation work of the superseded
32-attempt design; all work before and after the frontier is unchanged.

## Supersession Matrix

Only the following old statements are replaced:

| Prior document location | Superseded contract | Replacement |
| --- | --- | --- |
| Design, **Interfaces and Provenance** | At most 32 adjacent boundary floats globally | At most 64 adjacent boundary floats globally |
| Design, **Tests**, audit paragraph | Audit holds 32 attempts; exhaustion and overflow are defined around attempt 33 | Audit holds 64 attempts; exact exhaustion is 64, compile-time equality prevents a 65th production revalidation or audit record, and the full-capacity recorder remains fail-closed |
| Plan, Task 2 public test-seam sketch | `G1RootReachAuditAttempt attempts[32]` | `G1RootReachAuditAttempt attempts[64]` |
| Plan, Task 2 Step 2 | One- and two-cursor traces exhaust at 32; no 33rd write | One- and two-cursor traces exhaust at 64; loop/capacity equality prevents a 65th production revalidation or audit record |
| Plan, Task 2 Step 4 | Test at most 32 adjacent boundary floats globally | Test at most 64 adjacent boundary floats globally |

References in those same audit passages to catching a `33/64/128` loop-bound
regression are correspondingly replaced by exact 64-attempt exhaustion,
compile-time ceiling/capacity equality, the strictly bounded production loop,
and the unchanged fail-closed full-capacity recorder. No other prose,
interface, test oracle, implementation order, or acceptance gate in either
prior document is superseded.

## Non-Goals and Completion Boundary

This amendment is complete when the 64-attempt contract passes the acceptance
surface above and the exact row-6 and row-7 planner frontiers authenticate.
It does not certify the complete low-curb route, solve the row-6 swing failure,
change lateral or diagonal matching, add half-support edge handling, align
feet to slopes, increase terrain sample density, or render the G1 mesh. Those
remain separate tasks whose fixes must preserve this planner contract.
