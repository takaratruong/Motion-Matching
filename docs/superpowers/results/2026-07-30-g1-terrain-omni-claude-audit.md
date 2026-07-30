# G1 Terrain Omnidirectional Motion-Matching Candidate — Independent Audit

**Date:** 2026-07-30
**Scope:** Independent read-only audit of the retained G1 terrain motion-matching
candidate. Facts are separated from inference. No implementation, test, config,
or existing documentation was modified — this report is the sole artifact.
**Read for this audit:** `sonic/python/mm_sonic/torch_motion_matcher.py`,
`torch_terrain_omni_rollout.py`, `torch_terrain_omni_metrics.py`,
`torch_terrain_omni_routes.py`, and the three experiment configs under
`sonic/configs/experiments/`.

## 1. Frozen comparison (controller-observed facts)

All numbers below are controller observations over the same 21 omnidirectional
stair routes; they are not re-derived here.

| Arm | Config | Passes | Safety rescues | Rescue cycles | Mean stalled frac | Worst stall (fr) | Stance slide (m) | Worst p95 step (ms) |
|-----|--------|:------:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Retained** | dense wt 3 + source-history | **13/21** | 12 | 0 | 0.0283 | 5 | 10.763 | — |
| Matched control | dense wt 4 + source-history | 10/21 | 17 | 0 | 0.0610 | 26 | 10.554 | — |
| Data control | small corpus wt 3 + source-history | 12/21 | 165 | 4 | 0.0584 | 38 | 14.980 | 131.9 |

**Facts read from the comparison:**

- The retained arm (expanded corpus, dense weight 3, source-history on) wins on
  passes (13 vs 10 vs 12), on stall (lowest mean 0.0283 and lowest worst stall,
  5 frames), and has **zero rescue cycles** with only 12 rescues.
- Dense weight 3→4 (matched control) *lost* three routes (13→10), doubled mean
  stalled fraction (0.0283→0.0610) and worst stall (5→26). Weight 4 is strictly
  worse on every stability metric here. (Config evidence: `torch_terrain_expanded`
  sets `dense.weight 3.0`; `legacy` and `torch_stair_small` use `4.0`.)
- Shrinking the corpus (data control) collapses stability: 165 rescues, four
  rescue cycles, worst stall 38, stance slide inflated to 14.980 m, worst-route
  p95 step 131.9 ms. The penalty alone does not compensate for a thin corpus.

**Inference:** Weight 3 sits at a local stability optimum — both neighbors
(higher weight, smaller corpus) regress. Corpus size is the dominant lever for
rescue-cycle elimination; dense weight is a secondary, already-tuned lever.

## 2. Remaining failure clusters (8 of 21) — mapped to code

Each cluster maps to a specific gate in `evaluate_route_outcome`
(`torch_terrain_omni_rollout.py:114-160`) and a route in
`torch_terrain_omni_routes.py`.

1. **Mixed-adversarial turn/reverse/final-exit** — `mixed-adversarial`
   (routes:231) requires four segments incl. `reverse-down`, `final_surface=flat`,
   `final_heading_error_max_rad=0.35`. Compound chain.
2. **Riser-reversal backward progress** — `riser-reversal` (routes:222) requires
   `reverse-down` segment progress (velocity −0.38 x). Gate:
   `segment:reverse-down:progress`.
3. **Three left/lower side exits remain elevated** — `side-exit-*` (routes:189,200)
   require `final_surface="flat"`; failing → `final-surface:not-flat`, i.e. the
   foot is still horizontally over the stair footprint (`surface[tail] > 0.05`,
   rollout:139-144) — a lateral-progress miss, not an inertialization residual.
4. **Side-mount-left continue-up / final elevation** — `side-mount-left`
   (routes:116) requires `continue-up` progress and `final_surface="elevated"`.
5. **Both 180° turns' last-ten-frame settling error** — `turn-180-upper-left/right`
   (routes:159) fail `final_heading:error` over the last-10-frame tail (rollout:139).

**Pattern (inference):** clusters 1–2 are *commanded reversals*
(`reverse-down`); clusters 3–4 are *lateral / mount progress*; cluster 5 is
*terminal heading*. These are three distinct mechanisms, not one settling bug.
The reversal cluster is the one that a code path plausibly *causes* rather than
merely fails to cover (see §4), which is why it is ranked first.

## 3. Assessment of retained choices

### 3.1 Source-history penalty — scoping and transactionality (facts)
`_loop_revisit_transition_costs` (`torch_motion_matcher.py:745-800`) is:
- **Per-rollout scoped:** reads `self._selection_history`, re-initialized in
  `reset()` (line 1522) to only the reset visit; no cross-episode carryover.
- **Transactional:** history is appended **only in `commit()`** (line 1812) and
  trimmed to `LOOP_HISTORY_FRAMES=128` (line 1820). `prepare_step` computes the
  penalty from history but never mutates it (line 1597-1608), so a prepared-but-
  uncommitted step cannot pollute the penalty state. The matrix reuses one
  matcher across routes (`matcher_factory=lambda _route: matcher`, rollout:632)
  but calls `matcher.reset()` per route (rollout:350), so isolation holds.
- **Trigger predicate (fact):** penalizes candidate rows in a ±8-frame source
  neighborhood of a prior visit only when visit age ∈ (`exclusion_frames`, 128]
  **and** current root is within `LOOP_ROOT_PROGRESS_M=0.05 m` of that visit's
  world root (lines 775-786), applying `LOOP_REVISIT_PENALTY=1000.0`.

**Verdict:** correctly scoped and transactional for its stated anti-stall intent
(commit 8286044). One structural concern is in §4.

### 3.2 Outcome gates — meaningful? (facts)
`matrix_pass` requires `completed_without_exception AND outcome.completed`
(rollout:516-520); process completion alone never passes a route. The four gates
are behavioral: projected segment progress vs expected, elevated-foot engagement
(≥20 samples), final surface over the last 10 frames, and final heading over the
last 10 frames. **Verdict: meaningful, not process-completion.** Caveat:
`min_segment_progress_ratio` is fixed at **0.1** for every route (routes:90) — a
route can clear the progress gate at 10% of expected displacement, so progress is
weakly gated and the terminal surface/heading gates carry most of the teeth.

### 3.3 Should dense weight 3 remain retained? (inference from facts)
**Yes.** Weight 4 regresses on passes and every stability metric; weight 3 +
expanded corpus is the current matrix optimum with zero rescue cycles. Retention
is justified by observation, not by default.

## 4. Ranked next hypotheses (three)

1. **(Recommended) The world-root-keyed loop-revisit penalty misfires on
   commanded reversals.** During a `reverse-down` segment the robot retraces its
   own ascent path, so its current root can repeatedly fall within
   `LOOP_ROOT_PROGRESS_M=0.05 m` of ascent visits whose age lands in
   `(exclusion_frames, 128]`. The penalty then suppresses the previously visited
   source neighborhoods by `+1000` (lines 775-799). History does not record the
   old command direction, so the code alone does **not** prove that these are the
   correct backward candidates. *Inference (geometry):* for a 160-frame
   `reverse-down` retracing a ~230-frame
   ascent, a contiguous band of reverse frames re-enters penalized neighborhoods,
   suppressing the natural reverse source and degrading `reverse-down` progress —
   matching clusters 2 and the `reverse-down` leg of cluster 1. This is a path the
   code *causes*, and the anti-loop benefit (§3.1) can be preserved by exempting
   reversals only.
2. **Terminal heading convergence on 180° turns (cluster 5).** Candidate scalar
   levers are `inertialization_halflife_s` (0.1 s ≈ two half-lives over the 0.2 s
   tail, leaving ~25% offset) and `yaw_rate_rad_s`. Fewer routes; the yaw cap is
   likely non-binding (π rad over 2.0 s needs ~1.57 < 2.094 rad/s), so this is
   less certain than #1.
3. **Corpus coverage for lateral side-exit / side-mount (clusters 3–4).** Highest
   cost, inherently multi-variable (corpus rebuild), deferred.

## 5. Recommended next hypothesis and single-variable experiment

**Hypothesis:** The loop-revisit penalty degrades commanded reversals. Add a
single experimental reversal exemption that suppresses the penalty when the
current command opposes the command direction stored with the matched historical
visit. A one-frame command-transition predicate would be insufficient because
the path retrace lasts much longer than the transition. This treatment should
recover reverse-down progress **without** reintroducing rescue cycles, because
the penalty still applies to same-direction revisits.

**Single variable:** the reversal exemption (off = retained control, on =
treatment). Everything else frozen at the retained config (expanded corpus, dense
weight 3, source-history on, `LOOP_*` constants unchanged).

**Acceptance:** On the same 21-route GPU matrix, passes increase to **≥15/21**,
`riser-reversal` clears `segment:reverse-down:progress`, **rescue cycles stay at
0**, and mean stalled fraction stays **≤0.035** (no material regression from
0.0283).

**Rejection:** If passes are **≤13/21**, OR `riser-reversal` still fails its
`reverse-down` progress gate, OR rescue cycles become **>0**, OR mean stalled
fraction rises **>0.045**, reject and promote hypothesis 2 (terminal heading
convergence via a single `inertialization_halflife_s` reduction).

## 6. Residual risk

This recommendation is unqualified until Codex runs the real GPU matrix; all
numbers in §1 are the frozen controller observations and the §4/§5 geometry is
inference from the committed code, not a measured result. The reversal-misfire
hypothesis must be confirmed on that matrix, and the exemption must be shown not
to reintroduce the data-control rescue-cycle behavior before any retention.
