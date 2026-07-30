# G1 Terrain Transition Planning — Independent Code Review & Next-Experiment Plan

**Date:** 2026-07-30
**Reviewer:** Independent Claude code review
**Subject commit:** `6671c46` (`feat: align terrain clips and rescue safe transitions`)
**Benchmark headline:** 15/21 routes pass on the frozen 21-route terrain matrix.
**Scope:** Report-only. No source, test, config, existing doc, threshold, or
artifact was modified.

> **Evidence legend.** `[M]` = measured / directly read from a committed
> artifact or command output in this review. `[I]` = inference or argument;
> every `[I]` names the falsifier that would overturn it. A report-only review
> **cannot** prove any recommended architecture improves the GPU matrix; that is
> left for Codex to implement test-first and run on the frozen 21-route matrix.

---

## 1. Retained-code review of `6671c46`

**[M]** `git show --stat 6671c46`: 9 files, +717/−41. Source touched:
`torch_motion_matcher.py` (+153/−41), `torch_terrain_features.py` (+73),
`registry.py` (+28), `terrain.py` (±1). Tests added in four test files
(+192, +106, +39, +17). One new report doc. The change has two logical parts:
(a) a **motion-to-terrain registration transform** for `down-continuous-33` and
`staircase-final-v3`; (b) a **rewrite of loop-revisit costing** plus a
**second-chance ranked rescue** in the transition-reject branch.

Findings are severity-ranked.

### Critical
None found. The two behavior-changing seams are guarded by new focused tests
(§1 Minor) and the early-exit optimization is order-safe (see Important-1).

### Important

**I-1 — Early `break` on `maximum_total_cost` is correct only under ascending
rank order.** `[M]` `_ranked_terrain_rescue` (`torch_motion_matcher.py:1098`)
breaks the ranked loop when
`effective_total_cost = selected_total_cost + additional_transition_penalty
>= maximum_total_cost`. `[M]` `rank_exact_transition_candidates`
(`:529`) sorts by `argsort(eligible_total_costs, stable=True)` — ascending
`selected_total_cost`. Since `additional_transition_penalty` is a constant
offset, `effective_total_cost` is monotonically non-decreasing across ranks, so
the early break provably drops only candidates that also exceed the bound.
**Contract note:** this monotonicity is load-bearing and undocumented; a future
change adding a per-candidate penalty term would silently break the early exit
and cause premature rejection. Falsifier `[I→M]`: if any ranked penalty becomes
per-candidate, the break must become a `continue`.

**I-2 — New rescue path uses the incumbent cost as its acceptance ceiling.**
`[M]` In the transition-reject branch (`:1756`+) the second-chance rescue is
called with `maximum_total_cost=decision.incumbent_cost` and
`require_safe_candidate=False`. This is a genuine behavior change: previously the
branch fell straight through to `transitioned=False` (hold incumbent); now it
first tries to find a safe candidate strictly cheaper than the incumbent.
**Conclusion risk:** the report attributes the 14→15 gain primarily to
normalization, not to this rescue path; the rescue's isolated contribution to
the matrix is not separately measured. `[I]` Falsifier: an ablation toggling
only `require_safe_candidate` on the 21-route matrix would isolate its effect.

**I-3 — Exclusion-window semantics changed for same-clip vs. cross-clip.**
`[M]` Old `_loop_revisit_transition_costs` skipped any visit with
`age <= exclusion_frames`. New code (`:784`) only skips when the visit is on the
**same** `clip_index`; recent cross-clip visits now receive the smaller
`RECENT_CROSS_CLIP_REVISIT_PENALTY = 25.0` (`:35`) instead of being ignored, and
older visits still get `LOOP_REVISIT_PENALTY = 1000.0`. This intentionally
discourages rapid cross-clip thrashing. **Contract note:** 25.0 is a new magic
constant sitting near terrain-feature costs (~50–74 per §2), so it is large
enough to reorder candidates; it is covered by
`test_recent_low_progress_cross_clip_revisit_is_penalized` but its matrix-level
effect (esp. on rescue count and 15.695 m sliding) is not isolated.

### Minor

**M-1 — Registration transform validated at two layers.** `[M]` `registry.py`
`__post_init__` rejects non-finite/non-numeric `motion_to_terrain_xy_yaw`;
`torch_terrain_features.py` re-validates the manifest field for
`EXPANDED_TERRAIN_DATASET_SCHEMA` and raises `ContractError` on malformed
registration. Good defense-in-depth; covered by
`test_expanded_dataset_requires_finite_motion_registration`. Minor: legacy
(non-expanded) schema silently defaults to `(0,0,0)` rather than asserting
absence — acceptable but undocumented.

**M-2 — Hard-coded per-clip transforms.** `[M]` `_MOTION_TO_TERRAIN_XY_YAW`
embeds literal offsets for two clips (`-0.29, 3.99, -π/2` and `-0.24, -0.10,
π/2`). These are data-registration constants, defensible in a registry, but
carry no provenance comment linking them to the alignment measurement.

**M-3 — `search_time` accumulation reused for rescue and rerank.** `[M]` Both
`_ranked_terrain_rescue` and `_rerank_transition_window` add their own
`perf_counter_ns` deltas into the same `search_time`. Correct for a latency
budget, but double-invoking `rank_exact_transition_candidates` in the
reject-then-rescue path doubles matching cost on rejected frames (perf note,
not a bug).

---

## 2. Failure cluster: the 15/21 result

**[M]** From the committed report
`docs/superpowers/results/2026-07-30-g1-terrain-normalization-phase-selection.md`:
restoring small-corpus normalization on the combined expanded corpus moved the
matrix from **14/21 to 15/21**, restored `cross-tread-left-to-right`, kept both
180° turns and `riser-reversal`, dropped mean stalled-moving fraction 0.0261→
0.0171, and held rescue cycles at 0. The **6 remaining failures** are:
`side-mount-left`, all four `side-exit-*`, and `mixed-adversarial` final-surface
completion — now a single **lateral terrain-exit** cluster.

**[M]** Aggregate table (report §"Aggregate comparison"): retained `v10` 13/21;
expanded-norm `v23` 14/21 (161 rescues, 15.757 m slide); small-norm combined
`v41` **15/21** (168 rescues, 0.0169 stall, 15.695 m slide). The increased
rescue count and sliding vs. `v10` remain unresolved.

*This review reads 15/21 from committed artifacts only; it does not claim a GPU
re-run was performed here.*

---

## 3. Why upper-side-exit and later-reversal are causally indistinguishable

**[M]** Report §"Causal limitation" and §"v3 phase-selection tradeoff": for the
upper-exit and reversal routes, **all commands before the late command change
are identical forward ascent**. `staircase-final-v3` only reaches its useful
descent phase after an early multi-second commitment (useful ascent ~frame 60,
still ascending ~260, turnaround ~280–300); the side-step path is required for
the best lateral exit. Admitting v3 made `side-exit-upper-right` enter v3 at
route frame 80 (still flat) and fail final-flat; removing v3 fixed that exit but
regressed `riser-reversal` (0.260→0.015) and `diagonal-down-left`
(0.378→0.003). Searchable v3 start frames {100,160,220,240,280} all restored the
exit but lost the reversal/diagonal benefit.

**[I]** At the early branch point, paired runs that start from the same matcher
state have the same observations and commands available to a causal policy;
the command that distinguishes exit from reversal has not yet been issued. A
deterministic matcher must therefore make the same early choice in both runs.
The later internal phase, contact schedule, and root trajectory need not remain
identical after different earlier choices, so those are not evidence for the
claim. Subject to the falsifier below, **no static clip weight, registration, or
unconditional phase cut can choose a different early path for each future**.
This does not prove that a reactive transition after the command change is
impossible; it says only that the unissued command cannot justify a different
prefix choice.

**Falsifier `[I→M]`:** If a committed diagnostic shows the two future classes
carry a **separable feature before the commitment frame** (e.g. an early lateral
velocity, heading bias, or phase-residual that predicts the branch), then they
are *not* indistinguishable and the fix is feature enrichment, not planning /
commitment / bridge data. No such separating feature is present in the reviewed
artifacts.

---

## 4. Three causal alternatives (latency & compute tradeoffs)

The retained report (§"Next experiment") already names these three; quantified
below.

| Option | Mechanism | Command latency | Compute cost | Data cost |
|---|---|---|---|---|
| **A. Short-horizon transition planning** | After the command changes, search a bounded reachable sequence rather than only re-score one candidate's already-emitted window | No intentional command hold; adds measurable planner compute time and may take motion time to reach the selected phase | bounded beam/candidate expansion per replan; can reuse `rank_exact_transition_candidates` | none |
| **B. Command commitment / action chunks** | Expose an intended future command and commit a k-frame chunk | **+k-frame input latency** — hard non-reactive window; matcher cannot respond to a new command mid-chunk | ~flat (one search per chunk) | none, but requires an operator-intent channel that does not exist today |
| **C. Bridge-motion collection** | Capture explicit clips bridging the shared ascent state to both reversal and lateral-exit phases | ~0 added latency | +DB rows in every matching frame | **new motion capture required** (largest, slowest) |

**[I]** Latency framing (mandated by the frozen contract): any option that
assumes a command before the operator supplies it imposes input latency equal to
the horizon/chunk it commits to. Option **B** has a hard non-reactive window
(k frames = k×20 ms at 50 Hz). Option **A** does not intentionally withhold a
new command, but it still adds wall-clock planning latency; that latency must be
instrumented rather than described as zero. Reaching a selected future phase
can also take motion time, which is response dynamics rather than input
blackout. Option **C** adds database-search cost but no intentional command
hold, and is gated on data that may not exist.

**[M]** At 50 Hz, `exclusion_frames = 20` (`torch_motion_matcher.py:47`) excludes
a ±20-frame neighborhood around the current source frame during search; it is
not a temporal commitment window. `LOOP_HISTORY_FRAMES = 128` (2.56 s) is
revisit memory. These constants bound existing search behavior but do not by
themselves justify a planning horizon.

---

## 5. Recommended experiment (exactly one, smallest causal)

**Recommendation: Option A — first implement a diagnostic-only, bounded
transition-reachability planner at the command-change boundary.** This is the
smallest causal experiment: it needs no new data and does not commit runtime
behavior before feasibility is established. It must model a reachable sequence
of matcher states; simply scoring farther into one composed candidate is not
planning because the current terrain validator already inspects the full
46-frame emitted window. The oracle may reuse
`rank_exact_transition_candidates`, but it must report whether a safe
descending/lateral phase is reachable from the actual command-change state
without using an unissued command. Only if that oracle succeeds should its first
action be allowed to affect selection in a subsequent experiment.

### Test-first seams
- **First focused test:** a synthetic deterministic transition graph contains
  a two-hop safe lateral/descending phase that the one-step greedy choice
  misses. Assert the bounded oracle finds the reachable path, returns its first
  intermediate state, and never reads a future command.
- **Adversarial branch:** when no safe phase is reachable within the exact
  expansion budget, assert `reachable=False`, the incumbent decision is
  unchanged, and the diagnostic records budget exhaustion without an exception.
- **Integration test:** on a captured failing side-exit command-change state,
  run the oracle read-only beside the retained matcher and assert its trace is
  reproducible. This is the evidence that decides whether behavioral planning
  is warranted.

### Bounded candidate expansion
- Use depth **≤2 transitions**, beam width **K≤8**, and a total hard cap of
  **64 composed/validated states** per command-change event. Candidate motion
  may advance at most **15 source frames (0.30 s)** between expansions.
  Evaluate only when the shaped command changes, and do not widen the database
  candidate pool.

### Observability / public diagnostics
- Emit a public per-decision diagnostic:
  `{command_change_frame, planner_compute_ns, expanded_state_count,
  depth_used, reachable, first_clip_index, first_frame_index,
  terminal_clip_index, terminal_frame_index, budget_exhausted}` so the route
  runner can distinguish infeasible data coverage from a matcher-selection
  failure.

### Accept / reject gates (frozen matrix, no threshold weakening)
- **Accept the feasibility experiment, iff all hold:** the oracle finds a
  terrain-safe lateral/descending terminal for at least one captured
  `side-exit-*` command-change state; the path is replayable under the current
  command; ≤64 states are expanded; planner compute latency is reported; and
  the read-only oracle leaves all retained decisions and 15/21 outcomes
  bit-identical.
- **Reject behavioral planning if any:** no failing side-exit state has a
  reachable terminal; the result requires a not-yet-issued command; the path
  cannot be replayed; the state/latency bound is exceeded; or the diagnostic
  changes matcher behavior. A rejected feasibility experiment directs the next
  investigation toward bridge-motion coverage, not another scalar weight.

**Residual risk `[I]`:** This report cannot prove Option A wins the GPU matrix;
the causal comparison cannot prove which alternative wins. The experiment stays
provisional until Codex implements it test-first and runs the frozen 21-route
GPU matrix.
