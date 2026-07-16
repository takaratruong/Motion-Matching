# G1 Bounded Candidate Certification Design

## Objective

Recover from a geometrically good motion match that is not physically valid
after transition, contact scheduling, terrain footprint evaluation, IK, and
whole-pose clearance. Search a small, deterministic set of near-best motion
candidates and accept the first candidate that passes both the existing raw
no-IK path and the existing IK path. The ordinary incumbent continuation
participates in the same ranked set rather than acting as an out-of-band
heuristic.

The selector must preserve:

- exact binary32 25 Hz simulation time (`0x3d23d70a`);
- independent travel direction and heading, with no heading derived from the
  selected motion;
- identical matcher choice and all existing Gate E pair-invariant fields
  between IK-on and IK-off runs;
- every existing footprint, reach, residual, correction, swing-clearance,
  and pose-clearance threshold;
- atomic frame publication and exactly one presentation-frame increment; and
- bounded work and bounded failure evidence.

This is matcher recovery, not a contact-data rebuild, an IK tolerance change,
or a terrain-pack change.

## Observed Low-Curb Failure

The first currently reproducible blocker is presentation row 6 of
`grail-curb-low:curb-forward` in the 32-frame paired run.

The accepted baseline after row 5 is database frame `272446`, range `1021`,
source `terrain_curbs__curb_114__006`, with both recorded contacts true. In
`/tmp/g1-root-reach-cert/low-curb/32-0.csv`, the IK-off row 6 records the real
matcher decision:

- query frame/range: `272446 / 1021`;
- selected frame/range: `428956 / 1647`;
- selected source: `terrain_curbs__curb_186__001`;
- executed frame after the ordinary unconditional continuation step:
  `428957`;
- transition: true;
- executed contacts: left false, right true;
- incumbent cost: `3.00036263`; and
- selected production cost: `2.57850266`.

The paired IK-on row 6 in
`/tmp/g1-root-reach-cert/low-curb/32-1.csv` finite-rejects at
`ik-candidate` with `no-swing-candidate` and atomically retains frame
`272446`. Its attempted footprint predicts a left landing at schedule sample
3 with center approximately
`(-0.0658538, 0.00422484, -0.155635)`. The releasing left target remains
near `(0.0612094, 0.00629049, -0.603687)`. None of the unchanged 41 lift
candidates passes the existing constraints and clearance certificate.

The paired IK-off log is the authoritative evidence for the attempted
selected frame and public production cost. Its exact row-6 query is the
following 31-word concatenation:

```text
3de11a0a3d574158bd3b1ecabe0478ec3d573390bc5b68fabb8c82fcb9a49830b86f6000bb04f968ba8ea77cba905e00be23d1cebd34d0483d851794bd34d7ba3dc62330bde623773e683e17be3fe67f3ebd5aa9be3635e23f7bea21be64911f3f798a8ebe7176033f78c7f600000000000000003db7f28400000000
```

A disposable cross-build diagnostic rebuilt all 31 matching features from
the verified motion pack and terrain sidecar at terrain-weight word
`0x40800000`, then applied the existing 20-frame range-end and surrounding
exclusions around incumbent `272446`. Strict, fast, and native builds retained
the same first five frames. The strict build found 424,248 eligible rows; its
first six transition scores are:

- rank 0: frame `428956`, range `1647`, score `0x40250a4d`;
- rank 1: frame `117196`, range `400`, score `0x4029ef7c`;
- rank 2: frame `428955`, range `1647`, score `0x402e9e8b`;
- rank 3: frame `117197`, range `400`, score `0x4030f4a7`;
- rank 4: frame `117195`, range `400`, score `0x403398a0`; and
- rank 5: frame `21702`, range `18`, score `0x4041d433`.

Exactly the first five are below both the public incumbent word
`0x404005f1` (`3.00036263`) and the oracle's independently accumulated
incumbent word `0x4040097d`, which is 908 ULP above the public owner. The
sixth is above both. The five better transitions come from only two ranges:
`1647`, source `terrain_curbs__curb_186__001`, and `400`, source
`terrain_curbs__curb_043__007`. For diagnosis only, the same exhaustive run
counted 5 rows within zero-seeded best plus `0.25`, 16 within best plus `1`,
and 300 within best plus `4`; none of those deltas is a selection boundary in
this design.

These rebuilt feature matrices were owned by the diagnostic translation
units, not captured from the running controller. Their score words and counts
are therefore cross-build density evidence, not exact runtime-matrix oracles.
The strict diagnostic rank-zero word is not the public production-cost word.
The existing runtime's independently owned public result is `0x40250630`
(`2.57850266`). The owners agree on selected frame `428956`; their distinct
cost bits must remain distinctly labelled and must never be substituted for
one another.

The labels are physically plausible. At selected frame `428956`, the
left/right contact labels are true/true, toe-origin height above source
terrain is approximately `0.03519 / 0.03501 m`, and centered toe speed is
approximately `0.0404 / 0.0196 m/s`. These satisfy the unchanged contact
derivation thresholds of `0.06 m` and `0.15 m/s`. At executed frame `428957`,
left speed is approximately `0.15892 m/s`, so the left label becomes false;
the right label remains true. This is a legitimate phase boundary, not a
corrupt label.

The current matcher has no discrete contact feature. Its 31 dimensions are
foot positions, foot velocities, hip velocity, future trajectory positions,
future trajectory directions, and four terrain features. It chooses one
nearest feature row, transitions to that row, and then advances one database
frame before reading contacts. Consequently, a selected double-support row
can execute its first single-support row without downstream feasibility
having participated in selection.

Data density is not the limiting factor. Of the five exact-query transitions
that beat the incumbent, only rank 0 executes an immediate left release.
Ranks 1 through 4 execute double support.

The incumbent is proven feasible at the exact row-6 state by a source-free
counterfactual replay. An unmodified debug-symbol build first reproduced the
authoritative row-6 query, selected frame, and public cost bits. A conditional
GDB breakpoint after the unchanged legacy search then changed only
`selected = prior_index`, forcing `272446 -> 272447`. The raw log
`/tmp/g1-exact-row6-forced-incumbent-raw.csv` and IK log
`/tmp/g1-exact-row6-forced-incumbent.csv` both accept at presentation row 6
with public incumbent and selected cost `3.00036263`, support source `both`,
support height zero, and no rejection or stop on that row. The raw whole-pose
minimum clearance is
`-0.000789687969 m`. The IK replay records contacts and locks true/true,
contact residuals `0.0018916626 / 0.00117640046 m`, lock drift
`0.0018802149 / 0.00117427122 m`, sole alignment `1 / 1`, and whole-pose
minimum clearance `0.00507074688 m`.

The same source-free method gives physical-feasibility evidence for the four
diagnostic alternatives. In particular, forced `117196 -> 117197` accepts
raw with minimum clearance `-0.00079631526 m` in
`/tmp/g1-exact-row6-force-117196-raw.csv` and accepts IK with minimum
clearance `0.00507357065 m` and contact residuals
`0.00244133524 / 0.0012092652 m` in
`/tmp/g1-exact-row6-force-117196-ik.csv`. Individually forced frames
`428955`, `117197`, and `117195` also IK-accept with minimum clearance near
`0.00507 m` and maximum contact residual below `0.00252 m`. These debugger
runs intentionally leave the legacy rank-zero selected-cost field unchanged;
they prove candidate physics only, not recovery admission, score, or rank.

## Decision

Use one fixed-capacity attempt sequence with these constants:

- total candidate capacity `K = 8`;
- slot zero is the unchanged legacy `database_search` result;
- at most six distinct recovery transitions; and
- one final slot reserved for the incumbent continuation, deduplicated when
  the legacy result is already the incumbent.

The exact row-6 blocker has the legacy rank zero plus four other transitions
that beat the incumbent, then the proven incumbent. It therefore occupies
six of eight slots. The remaining two recovery slots retain a small fixed
margin for other queries without making capacity data-dependent. Capacity is
the only recovery bound: there is no base-cost window, score delta, or
runtime option. Increasing `K` requires a measured failure showing that every
dual-certifiable better-than-incumbent transition and the incumbent lie
beyond the current capacity, plus a new performance certification.

Contact state is not an eligibility gate and does not alter rank. A
transition that legitimately releases a foot remains selectable when it
passes the full downstream certificates. The exact row-6 candidate is
rejected because its realized candidate fails certification, not because its
left contact changes.

## Ranked Candidate Contract

### Authoritative legacy attempt

The existing `database_search` call and its normal matcher path remain the
authoritative first attempt. They use the current normalized query, range
partition, end-of-range and surrounding-frame exclusions, feature arithmetic,
idle transition cost, selected frame, and public cost ownership without
refactoring through the recovery provider. The returned transition or
incumbent is attempt slot zero.

Slot zero is dual-certified first. If it accepts, the outer frame commits
without constructing a recovery set or performing another database
traversal. Its selected frame, transition flag, source/range provenance,
`incumbent_cost`, and `selected_cost` remain byte-identical to the current
normal result. A global error also aborts immediately without recovery. Only
a finite slot-zero rejection on a scheduled-search frame may trigger the
lazy recovery traversal. When matching is disabled or search is not
scheduled, the incumbent is the only attempt.

For low-curb row 6, slot zero therefore remains transition frame `428956`
with authoritative public selected-cost word `0x40250630`. The strict
diagnostic word `0x40250a4d` is never substituted for that legacy result.

### Lazy recovery eligibility

The recovery provider receives the exact immutable query, ranges, incumbent
frame, transition-cost word, public incumbent-cost word, and exclusion values
used by slot zero. It performs at most one additional traversal through the
existing small/large bounding hierarchy and retains a fixed-capacity ordered
set. It must not perform a brute-force exhaustive scan, call
`database_search` repeatedly, or run at all after a successful slot zero.

`recovery_production_score` is the only recovery-transition ranking cost. It
owns query normalization as well as distance accumulation. For each feature
`0..30` in ascending order, the same strict no-fast-math owner authenticates
the raw query, offset, and scale inputs. If the scale bits equal the exact
`FLT_MAX` disabled sentinel, it materializes positive zero without subtraction
or division. Otherwise the scale must be positive and finite, and it performs
a separate binary32 `query - offset` and a separate division by scale. It
rejects any nonfinite input or intermediate and materializes all 31 normalized
words before traversal. A fast or strict caller must produce the same words.

The recovery transition score then starts at the exact transition-cost word
and processes those normalized words in ascending order with separate
binary32 subtract, multiply, and add operations:

```text
difference = query_normalized[feature] - database_feature[feature]
square = difference * difference
recovery_production_score = recovery_production_score + square
```

`recovery_incumbent_score` is independently accumulated by the same owner
from positive zero. A recovery score must not be formed by adding the
transition cost to a zero-seeded sum. There is no zero-seeded transition base
cost in the selection record or algorithm. Every small/large AABB lower bound
and pruning comparison is evaluated by the same strict no-fast-math recovery
owner, in ascending feature order and from the applicable exact seed; caller
floating-point contraction cannot decide pruning. Bounding-box early exits may
prune losing rows but must not change the fully accumulated score of any
retained row.

The recovery provider admits a transition only when all of these are true:

- it satisfies the unchanged range and surrounding-frame exclusions;
- its selected frame is distinct from both slot zero and the incumbent;
- its finite nonnegative `recovery_production_score` is strictly less than
  the finite `recovery_incumbent_score`; and
- when the exact bits of public `state.incumbent_cost` differ from the
  `FLT_MAX` end-of-range sentinel, the exact score that would be published for
  this recovery is also numerically less than that public incumbent cost.

Equality fails admission. Of all admitted rows, retain at most six by
numeric recovery production score and then database frame. A later lower row
evicts the current worst retained row; neither a zero-seeded ordering nor a
score window may affect retention. Emit those recovery transitions in stable
order, then append the incumbent unless slot zero already attempted it.

A recovery-transition record contains candidate kind, selected database
frame, executed clamped frame, source range, exact
`recovery_production_score` bits, and recovery rank. The test-only trace tags
the score owner as `legacy`, `strict-recovery`, or `incumbent`; no tag or new
field enters the production log schema. Malformed query, range, arithmetic,
or ownership data is a global error and publishes no recovery list.

When a recovery transition is accepted, its exact strict recovery score
becomes the existing public `selected_cost` for that recovered row. This is a
named alternate owner used only after a legacy finite rejection. Admission
guarantees it is numerically below a non-sentinel public `incumbent_cost`, so
the unchanged runtime checker accepts the row without using its four-ULP
exception. Normal slot-zero rows never use this owner and remain baseline
identical.

### Incumbent semantics

The incumbent record means the current no-transition behavior: retain the
current selected frame for transition provenance and execute its ordinary
clamped `+1` continuation. It is not a frozen pose, a zero-speed pose, or a
special IK target. It runs the same downstream stages as every transition
and is attempted at most once.

If slot zero was a transition and every admitted recovery transition
finite-rejects, the incumbent is the final bounded fallback. If slot zero was
already the incumbent, it is not appended or retried. A transition that ties
or loses to either required incumbent comparison is never retained or tried.

The private `recovery_incumbent_score` and public `state.incumbent_cost` are
different owners. At end of animation the public value remains `FLT_MAX`,
while recovery still uses the real finite private comparison score. If the
incumbent wins, both public `incumbent_cost` and `selected_cost` retain their
existing `FLT_MAX` sentinels. If a recovery transition wins, public
`incumbent_cost` remains `FLT_MAX` while `selected_cost` is its strict
recovery score. The private score must never leak into either public sentinel.

## Dual-Certified Selection

`ik_enabled` becomes a presentation choice, not a matcher-eligibility choice.
Both process modes execute the same hidden certification protocol and select
the same candidate.

For each ranked record, in order:

1. Start from a fresh deep candidate copy of the same immutable accepted
   controller baseline and the same latched external-input snapshot. No
   storage from an earlier attempt may alias or seed the next attempt.
2. Apply the record's transition or incumbent continuation exactly once.
3. Run the common inertialization, simulation, support observation,
   support retargeting, contact update, and footprint observation stages.
4. If the common footprint stage finite-rejects, the record fails. If it
   reports a global error, abort the outer frame immediately.
5. From the same common candidate, run the raw/no-IK branch first. This is
   the current disabled-IK no-op pose followed by its unchanged final FK and
   whole-pose clearance certificate. A finite rejection fails the record and
   permits the next rank; a global error aborts the frame.
6. Only when the raw branch accepts, run the IK branch from another fresh
   copy of that common candidate. It uses the current strict root-reach,
   left-then-right foot IK, 41-entry swing ladder, final FK, defensive
   clearance, and whole-pose certificate without altered limits.
7. The record is selectable only if both branches accept. Stop at the first
   selectable record; later ranks are not materialized.

This ordering is fixed in IK-on and IK-off processes. A raw failure must
short-circuit the record before hidden IK work, and no record can be selected
without both successful certificates.

### Persistent hidden IK ownership

Mode-independent choice must remain true after the first frame. It is not
enough to run a hidden IK check and then discard its lock/history state in an
IK-off process, because the next frame's IK certificate could diverge.

Both pose products and both clearance results remain candidate-local until a
record is dual-accepted. Accepted-state ownership is then exact and
mode-specific; no new persistent dual-certificate or hidden-pose metadata is
added:

- Existing `state.ik` is the sole persistent hidden-certification owner. A
  dual-accepted frame assigns it the successful IK branch's next
  `G1IkState` in both modes, including its foot locks and swing histories.
- With IK off, `state.ik_frame` remains the canonical disabled/no-op
  `G1IkFrameResult`. The persistent `state.ik_bone_positions`,
  `state.ik_bone_rotations`, `state.ik_global_bone_positions`, and
  `state.ik_global_bone_rotations` remain exact copies of the certified
  support-retargeted/raw pose. The four `state.ik_candidate_*` pose arrays
  retain the same raw local/global pose under the existing disabled
  convention. Both `state.ik_clearance` and
  `state.ik_candidate_clearance` describe that accepted raw pose;
  `state.ik_candidate_clearance_status` remains `G1ClearanceOk` and
  `state.ik_candidate_rejected` remains false. The successful hidden IK
  pose, frame result, and IK clearance are selector-local scratch and are
  not copied into those accepted public pose/result owners.
- With IK on, the existing `state.ik_frame`, IK pose arrays, candidate-pose
  arrays, and IK clearance owners receive the successful IK branch products
  under their current semantics.

Scene reset and scene switch initialize `state.ik` identically in both modes.
A finite rejection, a global error, or exhaustion of the ranked set cannot
advance it. The visible raw pose must never be used to reconstruct hidden IK
history.

After the common candidate and both certificates are complete,
`ik_enabled` selects only the published/rendered pose view. IK off publishes
the certified raw pose and retains `ik_applied == 0`; IK on publishes the
certified IK pose and its existing IK diagnostics.

The IK-off public log projection is a separate canonical disabled path. It
must emit the canonical Gate-E all-zero/default IK suffix, including
invalid-input clearance and swing statuses, `UINT32_MAX` swing selections,
and stop reason `none`. It does not materialize physical IK diagnostics from
even the visible raw pose. That path must not inspect `state.ik.feet[*].lock`,
`state.ik.feet[*].swing`, the hidden IK frame result or pose, root reach,
swing lift selection, lock drift, residuals, or hidden IK clearance. Thus
the nonzero hidden `state.ik` needed for the next frame cannot leak into a
disabled public column.

The visible branch cannot feed a later matcher query, route command,
simulation XZ, support state, contact schedule, or hidden certification
state. The common matcher/controller state and hidden IK state are identical
between paired modes. This isolates the intended rendered-pose difference
while preserving every field in `GATE_E_PAIR_INVARIANT_GROUPS`, including
query bits and provenance, selected/database/source frames, costs, terrain,
support, simulation, route, requested/applied travel, and desired/predicted
heading bits.

## Atomicity and Publication

Candidate attempts are private nested transactions inside one outer frame
transaction. They do not increment presentation or scene counters, advance a
route cursor, consume a prior safe-stop latch, write a runtime log, update a
camera, or publish a diagnostic.

On the first dual-certified candidate, the outer transaction performs one
commit containing:

- the selected common matcher/controller state;
- the certified raw/common product, the mode-selected visible pose owners,
  and the successful IK branch's next `state.ik`;
- one accepted diagnostic using the selected record's query, cost, frame,
  range, and source provenance;
- the visible pose view selected by `ik_enabled`; and
- the existing single presentation/scene/route lifecycle update.

Every rejected attempt is discarded in full, including transition offsets,
support state, contact locks, root-reach plan, swing history, clearance work,
timers, route progress, and diagnostics. A successful later attempt must be
bit-identical whether or not an earlier attempt was executed before it.

An incoming safe-stop latch is read identically by all attempts. Only the
selected outer commit consumes it, once, under the existing one-frame frozen
route-cursor rule.

## Bounded Failure Publication

Finite candidate failures are ordinary search evidence, not public safe
stops. Preserve the first finite rejection encountered in exact attempt-rank
and branch order: common stage, raw branch, then IK branch. Preservation means
a complete value snapshot of that first failing `G1FrameTransactionScratch`,
including its footprint, IK transaction, pose certificate/work, requested
intent, and rejection diagnostic, or an equivalently complete
producer-authenticated proof. Saving only `G1FrameRejectionDiagnostic` is
insufficient because the existing publisher derives and authenticates it
against the same scratch footprint and IK transaction. Later attempts cannot
overwrite, alias, merge with, or replace this snapshot.

If every ranked record finite-rejects:

- return one outer `FiniteRejected` status;
- preserve the complete accepted state and accepted-state digest;
- leave persistent hidden IK state unchanged;
- publish exactly the saved first rejection from its saved producer scratch,
  using the existing scratch/transaction authenticator, rejection stage, and
  stop reason;
- latch safe stop once;
- increment presentation exactly once; and
- publish no attempted candidate as selected or accepted provenance.

If any attempt reports a global error, abort immediately with the existing
global-error behavior. A global error is never downgraded to try-next
evidence, even when a later candidate might have passed.

No rejection or publication validator is weakened for lazy recovery. A
mismatch between the saved rejection and its saved footprint, IK transaction,
or checkpoint is a global error, not finite evidence.

Do not change the production log schema. Add a fixed-capacity test-only trace
which records candidate kind/rank/frame/cost and the common/raw/IK
disposition for each attempted record. It must be absent from non-seam
builds, fail rather than truncate on a hypothetical ninth record, and never
become a selection input.

## Threshold and Directional Invariants

Candidate recovery does not change or bypass:

- footprint walkability, multilevel, landing-patch, or work budgets;
- the 5 cm root-reach adjustment cap;
- leg reach shells, 5 mm contact residual convergence, or angular correction
  limits;
- the 41 exact swing-lift words or 8 cm maximum lift;
- planted, swing, defensive, or whole-pose clearance thresholds;
- terrain normals, sole orientation targets, or exact lock points; or
- any existing finite/global classification.

The ranked provider and certification coordinator receive the already-built
command and desired-heading snapshots as immutable inputs. They cannot write
travel velocity or heading. No candidate rotation is used to turn the robot
toward travel, so independent joystick travel and heading remain available.
Every attempt uses exact 25 Hz; no substep, resampling, variable `dt`, or
60 Hz assumption is introduced.

## Performance Budget

The deterministic work budget is part of correctness:

- exactly the existing legacy search on every scheduled-search frame;
- zero recovery traversals when slot zero accepts or globally errors;
- at most one additional accelerated recovery traversal after a slot-zero
  finite rejection;
- at most 8 total records, including the incumbent;
- at most one common evaluation, one raw certificate, and one IK certificate
  per attempted record;
- no IK evaluation after a raw finite rejection;
- no evaluation after the first dual-certified record; and
- all existing per-foot, footprint, and clearance work limits remain nested
  inside the candidate cap.

The lazy recovery search must use the existing small/large bounding hierarchy
and maintain its six-entry set during one accelerated traversal. A brute-force
scan is not an implementation. Tests authenticate work counters and reject a
ninth attempt, recovery on a normal accepted frame, a second recovery
traversal, repeated single-best searches, or evaluation after success.

The release performance gate measures only frame transactions, excluding
asset loading, window creation, rendering, logging flush, and process
startup. On the certification host, after a 32-frame warm-up, each 800-frame
headless low-curb IK-on run must satisfy both:

- mean transaction time no greater than 40 ms, preserving real-time 25 Hz;
  and
- 99th-percentile transaction time no greater than 80 ms, permitting one
  bounded recovery frame without sustained missed ticks.

Use three runs and require all three to pass. Timing is external benchmark
evidence and must not enter deterministic logs or candidate choice. If this
budget fails, optimize ranked enumeration or reuse common-stage work; do not
increase `dt`, reduce certification coverage, relax thresholds, or silently
lower `K`.

## Test-Driven Development

Write failing tests before production changes.

### Ranked provider

1. Preserve exactly one existing production `database_search` call as the
   authoritative slot-zero owner. Lexical guards and authentic/synthetic
   runtime goldens must prove that a slot-zero acceptance has the same query,
   selected and executed frames, transition flag, range/source provenance,
   public cost words, and every legacy matcher/common public field as the
   current implementation. Hidden IK ownership and each mode's public IK
   projection are tested separately; they are not part of a pre-feature
   IK-off state-digest identity claim. Recovery-enumeration and
   recovery-evaluation counters must both remain zero under strict and fast
   callers.
2. A finite slot-zero rejection on a scheduled-search frame starts exactly
   one accelerated recovery traversal. A slot-zero acceptance or global
   error, a matching-disabled frame, and a frame on which search is not
   scheduled start none. These tests separately authenticate that no lazy
   recovery work is performed merely to reproduce the legacy result.
3. For small synthetic databases, compare every emitted recovery frame and
   exact score word with an exhaustive oracle implemented with the same
   independent strict owner. Include equal-score frame ties, fewer and more
   than six admitted transitions, exact top-six eviction, stable order, and
   fixed capacity. Slot zero and the incumbent are excluded from recovery;
   append the incumbent once at the end unless slot zero already attempted
   it. The complete attempt count can never exceed eight.
4. Authenticate recovery-owned normalization before ranking. Test the
   disabled-scale bit pattern and ordinary scale values under strict and fast
   callers and require all 31 materialized normalized query words to be
   identical. Reject nonfinite query/offset values, nonpositive or nonfinite
   enabled scales, and nonfinite subtraction/division results. Prove that
   neither the caller's pre-normalized buffer nor a fast-math expression can
   become the strict recovery input.
5. Exercise both strict admission bounds independently. With the other bound
   loose, place a recovery score one ULP below, equal to, and one ULP above
   the private comparison score and then a public incumbent cost whose bits
   differ from `FLT_MAX`; equality and above both fail. Add the diagnostic
   908-ULP gap fixture with
   private word `0x4040097d`, public word `0x404005f1`, and recovery word
   `0x40400700`: it passes the private comparison but must still be excluded
   by the public comparison. With public `FLT_MAX` at end of animation, the
   same private comparison remains authoritative and no public finite bound
   is synthesized.
6. Independently authenticate transition accumulation from the exact
   transition-cost seed and incumbent accumulation from positive zero. A
   rounding-sensitive fixture must fail an implementation that computes a
   zero-seeded transition sum and adds the penalty later. No base cost,
   best-score delta, or score window may be constructed or consulted.
7. Run the fixed-bit inversion regression below at transition seed
   `0x3f800000`. Place A and B at the sixth recovery-slot boundary behind five
   unambiguously lower production rows. Require B to be retained and A to be
   evicted; any zero-seeded ordering would make the opposite choice.
8. Cover surrounding/range-end exclusion, range boundaries, legacy-frame and
   incumbent deduplication, sequential/no-search incumbent-only behavior,
   and idle transition-cost ordering.
9. Cover both end-of-animation outcomes. An incumbent win keeps public
   `incumbent_cost == selected_cost == FLT_MAX` while its finite private
   comparison score remains test-authenticated. A transition win uses that
   private score for admission, keeps public incumbent `FLT_MAX`, and
   publishes the strict recovery score as selected cost. The private value
   must not leak into either public sentinel.
10. Run the unchanged runtime checker over legacy wins, recovery-transition
    wins, incumbent wins, and independent public/private rounding. No test
    may add a schema field or relax the existing beat-incumbent/four-ULP
    rule. A recovery row's existing `selected_cost` field must contain the
    exact admitted strict score.
11. Reject nonfinite inputs, arithmetic overflow, malformed ranges, aliasing,
    and a hypothetical ninth attempt transactionally.
12. Authenticate work counters and storage: every scheduled search performs
    the one existing legacy traversal; a finite slot-zero rejection performs
    at most one additional traversal through the existing small/large
    hierarchy; accepted/global-error slot zero performs none. Reject a
    brute-force scan, a repeated single-best search, a second recovery
    traversal, or allocation proportional to database size.

Run the authentic low-curb row-6 query through the production recovery owner
and a same-build exhaustive test oracle. Authenticate the exact recovery
frames and score words actually emitted after deduplicating legacy frame
`428956` and incumbent `272446`, and require the live trace to agree. The
cross-build diagnostic list above predicts useful alternatives but its score
words are deliberately not hard-coded as runtime expectations.

#### Production-order inversion fixture

The query is 31 positive-zero words. The following are binary32 difference
words; construct each database feature as the sign-negated difference so the
query subtraction recreates the listed value exactly.

```text
A: 3e76b806 3ea08e4a bed358ea bef09450 bf6f7616 bffbad3a 3ca7c2d5 be90bb1d bf462b72 bebc2ed2 3eb6d0d9 bf52f8b9 bebcdda3 3e44dcbc 3f489f8e 3f3e1335 bed05f70 3d8da082 be870002 3e12312d bf26d190 bf309386 3c7106e2 bcb0dfde 3e95b2a6 bf8594b6 3f1a3b4e be757c29 3e9389b0 3f54c596 3eb57e61
B: 3f21629e 3f585617 3faea056 bf1fba50 3f038e09 3d84b2a9 bcb7213c bf1c6c11 3fb3b6b2 3da1eb7d bf0437fd 3cdbcd5c be41121a 3e581e78 3edcba12 3f656afc bdb8ec54 3ec1cd11 bf68ed43 3e4af82f bdf8d97a bf777d29 3ed3b084 3f8d4823 be2999bd bf05d2f2 3e338467 3eb032f5 3f4fedb4 3d78fac2 beae08de
```

The test-only zero-seeded sums are A `0x413e77bb` and B `0x413e77bc`,
so A appears lower. The actual seed-one production scores are A
`0x414e77bc` and B `0x414e77bb`, so B is lower. The provider never computes
or retains the zero-seeded sums.

### Dual selector and atomicity

Use a deterministic synthetic A/B/C fixture:

- A is the authoritative legacy slot-zero transition, executes a plausible
  support change, its raw branch passes, and its existing swing certificate
  finite-rejects.
- B is the first strict-recovery transition, satisfies both incumbent score
  bounds, and both certification branches pass.
- C must not execute after B succeeds.

Run the fixture with IK off and IK on and require the same B query, selected
frame, executed frame, range, source, cost, transition flag, support,
simulation XZ, travel intent, and heading bits. Require raw visible output
for IK off, IK visible output for IK on, canonical disabled-IK public fields,
and identical committed `state.ik`.

Also cover:

1. A raw failure short-circuits A's IK branch and proceeds to B.
2. A raw pass/IK fail cannot be selected. Independently forge the aggregate
   validator to claim raw fail/IK pass and require rejection; the runtime
   path itself must prove that raw failure never calls the IK branch.
3. Slot zero executes first, strict-recovery transitions execute in their
   exact emitted order, and the deduplicated incumbent executes last. If all
   preceding attempts finite-reject, the incumbent executes once with
   ordinary clamped `+1` continuation semantics. A legacy incumbent is not
   retried.
4. Poison every mutable owner in a rejected A attempt and prove B is
   identical to a direct B-only run.
5. Mutate each branch certificate, admission result, source/range/frame
   provenance, cost, rank, and hidden-state owner independently.
6. An immediate contact release that passes both branches is accepted,
   proving there is no contact hard gate; a double-support candidate that
   fails a certificate remains rejected.
7. All records finite-reject: one saved first failing transaction-scratch
   snapshot, one latch, one presentation increment, unchanged accepted
   digest, route cursor, common state, pose products, and hidden IK state.
   Independently mutate the saved `ik_transaction` and saved footprint before
   publication and require the existing producer authenticator to report a
   global error. A later failure must not overwrite either saved owner, and a
   diagnostic-only copy must be insufficient to publish finite evidence.
8. A candidate global error aborts without attempting the next rank or
   publishing finite evidence.
9. Scene reset/switch initializes paired hidden state identically; a
   multi-frame paired run retains identical choices after alternating raw and
   IK visible poses.
10. Poison every lock, swing-history, root-reach, lift, residual, clearance,
    and hidden IK pose/result field after a successful hidden IK-off
    certificate. Require the public disabled suffix to remain byte-identical
    to a zero-state run, and add an ownership/lexical guard that the IK-off
    projection cannot call the accepted-IK logger or read `state.ik`.
11. Prove exact commit ownership in both modes: `state.ik` equals the
    successful IK candidate in both; IK-off `state.ik_frame` is canonical,
    visible IK pose arrays equal raw, and raw clearance remains attached to
    that pose; IK-on public owners equal the successful IK products. Reject
    any new persistent dual-certificate owner.
12. Exact `dt` bits are required at ranking, common-stage, raw, and IK
    boundaries; travel and heading inputs are immutable throughout.
13. Strict/fast caller parity, ASan/UBSan/float sanitizers, no-seam builds,
    negative compile/link ownership tests, and complete logical hashes cover
    the new fixed-capacity and hidden-state owners.
14. A normal slot-zero-accepted frame must retain byte-identical legacy
    matcher/common public fields and cost/provenance in both modes and must
    never materialize recovery records or strict-recovery costs. Separately,
    the successful IK candidate must commit to `state.ik` identically across
    paired modes while each mode retains its specified visible/public IK
    projection; hidden IK state is not compared with the old IK-off digest.

Existing lexical/production guards around the single `database_search` call,
trusted no-main call closure, frame-stage ownership, candidate-audit
post-publication behavior, safe-stop lifecycle, and dirty-copy rollback must
be deliberately updated rather than bypassed.

## Live Acceptance

Build a disposable controller against the unchanged separate terrain pack.
Do not replace the running visualizer until all acceptance gates pass.

### Exact low-curb regression

Run paired IK-off and IK-on logs for 32 frames, then 800 frames, at exact
25 Hz.

The 32-frame test-only trace must show at presentation row 6:

- baseline query frame `272446`;
- legacy slot-zero selected frame `428956`, range `1647`, and authoritative
  public score word `0x40250630`;
- slot-zero raw acceptance and IK finite rejection with
  `no-swing-candidate`;
- exactly one subsequent accelerated recovery traversal, whose distinct
  frames and strict score words equal the same-build exhaustive oracle after
  excluding `428956` and incumbent `272446`;
- the first later dual-certified record, whether transition or incumbent,
  accepted only after both branches pass; and
- identical final matcher choice in the IK-off and IK-on logs.

This is a property requirement, not a hard-coded replacement rank: the
selector must choose the first actually dual-certified record in stable
order. The accepted record must belong to the bounded list and its test seam
must prove every earlier record failed. The public row must contain no
rejection or safe-stop evidence from those private attempts.
If the accepted record is a recovery transition, its public `selected_cost`
must equal its strict recovery score and be numerically less than public
`incumbent_cost`; if every recovery transition fails and the incumbent wins,
the existing incumbent public-cost semantics remain unchanged. The exact
forced replays above certify the incumbent fallback and the physical
feasibility of several likely recovery alternatives, but they do not replace
the live rank-and-admission proof.

Both 32-frame logs must pass normal runtime validation. Both 800-frame logs
must contain exactly 800 rows, make sustained route progress, complete the
low-curb route, contain no public footprint/IK/pose rejection or latched safe
stop, retain exact heading bits, and meet all existing contact drift, sole
alignment, residual, clearance, rendered-Hips step, and work thresholds.

### Gate E and paired invariants

Run the existing Gate E normal, stress, blocked, and Gate-D-IK pairs with the
current checker. For every pre-stop paired row, all fields in
`GATE_E_PAIR_INVARIANT_GROUPS` must remain byte-for-byte equal. In particular,
hidden dual certification must make query/selected/database/source frames,
costs, terrain, support, simulation XZ, route progress, travel intent, and
heading identical between IK-off and IK-on.

IK-off rows must still pass the existing canonical disabled-IK checks:
`ik_enabled == 0`, `ik_applied == 0`, and no hidden swing, root-reach, or IK
certificate may appear in public disabled fields. IK-on rows retain all
existing Gate E physical checks. Stress and blocked routes must retain their
exact allowed rejection signature and atomic tail; candidate retry cannot
turn an intended blocked course into route completion or alter the paired
control event that proves the obstruction.

Finally run inherited stair, multilevel, ramp, slope, lateral, diagonal, and
safe-stop cells, plus the complete Gate L paired matrix. No acceptance gate
may be weakened to admit this feature.

## Alternatives Rejected

### Hard contact-phase gating

Requiring selected or executed contacts to equal current contacts would
delay or prohibit legitimate liftoff and risks recreating the static holding
behavior. Requiring only one retained support foot does not solve row 6,
which already retains the right foot. Contacts may be logged as diagnostic
evidence but are not a selection oracle.

### Incumbent-only retry

The incumbent is proven to recover exact row 6, so it belongs in the ranked
set. Making it the only retry would ignore several nearly equal motion
candidates and could merely defer failure until the incumbent reaches the
same phase boundary. Ranked dual certification is the general boundary;
incumbent inclusion provides the small reliable fallback.

### IK-on-only retry

Allowing only IK-on to skip a failed candidate would immediately change
selected frames, sources, support, simulation, and route state relative to
IK-off, violating Gate E and making the toggle alter motion matching. Hidden
dual certification is required in both modes.

### Contact costs or a new learned feasibility feature

The matcher already has abundant close candidates, and the failure is known
only after current runtime locks, terrain queries, and exact certificates are
applied. A contact cost can be studied later for quality, but it cannot
replace authoritative downstream validation and is unnecessary for this
recovery.

### More motion samples or denser terrain points

The audit found hundreds of thousands of eligible frames and multiple
near-best sources. Increasing motion or terrain density would increase build
and search cost without addressing the one-best/no-downstream-certificate
architecture.

### Threshold, lift, or reach relaxation

Increasing swing lift, root reach, residual tolerance, or penetration
allowance would make an invalid candidate appear valid and weaken every
terrain case. The selector must choose a candidate that passes the existing
physical contract.

### Repeated single-best searches with exclusions

Running the full database search once per rejected rank is simple but gives
up to six extra database traversals and makes exclusion state part of the
result. One fixed-capacity ranked traversal is easier to authenticate,
strictly bounded, and fast enough to preserve 25 Hz.

## Scope

This design changes candidate enumeration, private certification
coordination, dual-pose ownership, and mode-independent publication. It does
not change database contents, GRAIL contact derivation, feature weights,
terrain artifacts, terrain scenes, foot geometry, IK thresholds, support
smoothing, route definitions, heading policy, sample rate, public log schema,
or visualizer/mesh rendering. Terrain packs remain separate. G1 mesh work
remains last.
