# Smart Pickup Approach Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop retryable full-pack previews and normal motion-matched approach paths from being rejected as hard Smart Pickup failures.

**Architecture:** Preserve legacy authoritative selector reasons while deriving a preview-specific `match_reason` that recognizes a feasible over-cost candidate. Keep the existing 1 m authored-slot admission gate, but make post-admission travel cumulative telemetry rather than an execution cap; frozen-route revalidation becomes finite/collision-only. Existing deadlines and cancellation bound control ownership.

**Tech Stack:** C++17 matcher, authored-slot geometry, Smart Pickup assist policy, Makefile C++ tests.

## Global Constraints

- Controller and runtime remain exactly 25 Hz.
- Flat locomotion only; do not modify or touch terrain behavior/processes.
- The initial authored-slot admission radius remains exactly 1.00 m with the existing 0.00002 m tolerance.
- Finite `assisted_travel_m` remains cumulative consecutive planar-distance telemetry.
- Existing 250-tick arrival deadline, WASD/`X` cancellation, target integrity, collision checks, and request authority remain unchanged.
- Legacy `MatchResult selection` hard-reason ordering remains unchanged; only preview-oriented `PickEvaluation::match_reason` may differ in the mixed CorrectionLimit/PoorMatch case.
- Missing/out-of-range feature storage remains fail-closed and outranks PoorMatch.
- Do not add multi-slot fallback, learned models, async matching, duration ranking, or playback changes in this plan.
- Do not access, stat, hash, execute, modify, stage, or delete the repo-root untracked `interaction_query_probe`.
- Use `git status --untracked-files=no`; stage only explicit paths and never `git add -A` or `git add .`.
- Preserve unrelated dirty `Makefile`, `tests/python/test_smart_pickup_full_pack_gate.py`, and exact untracked `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`.
- Commit and push every reviewed task before proceeding.

---

### Task 1: Preserve retryable PoorMatch in mixed candidate sets

**Files:**
- Modify: `interaction_matcher.cpp`
- Test: `tests/cpp/test_interaction_matcher.cpp`

**Interfaces:**
- Consumes: `FailureSet`, `matcher_detail::PickEvaluation::path_feasible`, `selection`, `match_reason`, and the existing `high_cost_fixture()` two-clip fixture.
- Produces: `match_reason=PoorMatch` for a hard-feasible over-cost candidate even when another clip reports CorrectionLimit, while `selection.reason` remains CorrectionLimit.

- [ ] **Step 1: Add a failing mixed-reason matcher test**

In `tests/cpp/test_interaction_matcher.cpp`, add and register this focused test:

```cpp
void test_preview_match_reason_preserves_retryable_poor_match() {
    using namespace interaction;
    RuntimeFixture fixture = high_cost_fixture();
    fixture.database.active_hands.at(0) = 1U;
    const MatchInput input = match_input_for(fixture);

    const matcher_detail::PickEvaluation result =
        matcher_detail::evaluate_pick_entries(
            evaluation_input(input),
            MatchConfig{},
            [](const MatchCandidate& candidate) {
                return candidate.clip == 0
                    ? Reason::CorrectionLimit
                    : Reason::None;
            });

    assert(result.path_feasible);
    assert(result.path_reason == Reason::None);
    assert(!result.match_ready);
    assert(result.feasible_entry_frame == 85);
    assert(result.contact_frame == 100);
    assert(result.total_cost_available);
    assert(near(result.total_cost, 106.40F, 1.0e-4F));
    assert(result.match_reason == Reason::PoorMatch);
    assert(result.selection.reason == Reason::CorrectionLimit);
}
```

The distinct assertions are intentional: preview retry semantics and legacy
authoritative selection semantics are different in this mixed case.

- [ ] **Step 2: Run the focused matcher suite and verify RED**

Run:

```bash
make build/tests/test_interaction_matcher
build/tests/test_interaction_matcher
```

Expected: compilation succeeds and the new `match_reason == PoorMatch`
assertion aborts because the current value is `CorrectionLimit`.

- [ ] **Step 3: Derive preview match reason after unchanged selection**

In `finish_evaluation`, keep:

```cpp
result.selection = reject(aggregate_reason(failures));
```

Then replace the direct assignment from `selection.reason` with:

```cpp
result.match_reason =
    !failures.out_of_range && failures.poor_match
        ? Reason::PoorMatch
        : result.selection.reason;
```

Do not reorder `aggregate_reason`, change `aggregate_hard_reason`, or change
`select_whole_clip`. The `out_of_range` guard preserves malformed-pack
priority in the existing mixed OutOfRange/PoorMatch tests.

- [ ] **Step 4: Verify GREEN and matcher/runtime regressions**

Run:

```bash
make build/tests/test_interaction_matcher
build/tests/test_interaction_matcher
make build/tests/test_interaction_runtime
build/tests/test_interaction_runtime
make build/tests/test_interaction_pick_assist
build/tests/test_interaction_pick_assist
```

Expected: all commands exit 0; pure reasons and legacy selector contracts
remain unchanged.

- [ ] **Step 5: Commit and push Task 1**

Stage only:

```bash
git add interaction_matcher.cpp tests/cpp/test_interaction_matcher.cpp
git commit -m "fix: preserve retryable smart pickup match reason"
git push checkpoint HEAD:g1-tabletop-placement
```

---

### Task 2: Separate slot admission distance from approach telemetry

**Files:**
- Modify: `interaction_pick_slots.cpp`
- Modify: `interaction_pick_assist.cpp`
- Test: `tests/cpp/test_interaction_pick_slots.cpp`
- Test: `tests/cpp/test_interaction_pick_assist.cpp`

**Interfaces:**
- Consumes: unchanged `select_pick_slot` admission checks,
  `revalidate_frozen_pick_slot`, `PickAssistDiagnostics::assisted_travel_m`,
  frozen-slot provenance, and the existing arrival deadline.
- Produces: collision-only post-admission frozen-route revalidation and finite
  cumulative travel telemetry without a 1 m terminal cap.

- [ ] **Step 1: Write failing frozen-route and approach-telemetry tests**

In `tests/cpp/test_interaction_pick_slots.cpp`, add a case beside the exact
1.00002/1.00003 admission-boundary test:

```cpp
const InteractionTarget admitted_target =
    make_route_target(vec3(1.00003F, 0.0F, 0.0F));
const PickSlotReason admitted_revalidation = revalidate_frozen_pick_slot(
    Transform{vec3(0.0F, 0.0F, 0.0F), quat()},
    Transform{vec3(1.00003F, 0.0F, 0.0F), quat()},
    admitted_target,
    {});
require(
    admitted_revalidation == PickSlotReason::None,
    "post-admission frozen route reapplied the activation radius");
```

Keep the existing `select_pick_slot` test that accepts 1.00002 m and rejects
1.00003 m.

Replace
`test_slot_approach_accumulates_inclusive_travel_and_rejects_overshoot()`
with a bounded scenario that begins at `(0,0,0.5)`, freezes slot 9 at
`(0.8,0,0.5)`, and observes displayed X positions `0.5`, `0.0`, then `0.5`.
Require after the third observation:

```cpp
require(same_float_bits_exact(
            assist.diagnostics().assisted_travel_m, 1.50F),
        "approach did not retain cumulative planar telemetry");
require(
    assist.diagnostics().state == PickAssistState::SlotApproach &&
        assist.diagnostics().reason == PickAssistReason::None &&
        assist.active() && output.override_steering &&
        !output.submit_interact,
    "finite admitted detour was treated as a terminal travel envelope");
```

Rewrite `test_frozen_slot_settling_enforces_consecutive_travel()` as
`test_frozen_slot_settling_records_consecutive_travel_without_failure()`.
Keep its exact accumulated value above 1.00002 m, but require Settling/None,
an active braking output, no submission, reset consecutive settle count if the
position is unstable, and unchanged frozen-slot provenance.

- [ ] **Step 2: Run slot and assist suites and verify RED**

Run:

```bash
make build/tests/test_interaction_pick_slots
build/tests/test_interaction_pick_slots
make build/tests/test_interaction_pick_assist
build/tests/test_interaction_pick_assist
```

Expected: the slot test reports `OutsideTravelEnvelope` at 1.00003 m during
revalidation, or the assist test fails because cumulative travel above the
shared cap becomes terminal.

- [ ] **Step 3: Make frozen-route revalidation collision-only**

In `revalidate_frozen_pick_slot`, retain config/transform validation and the
finite planar-distance calculation, but remove the post-admission envelope
branch:

```cpp
if (!planar_distance(
        live_root.position,
        frozen_root.position,
        remaining_route_length_m)) {
    return PickSlotReason::InvalidGeometry;
}
```

Remove the now-unused millimetre key and
`within_travel_envelope(remaining_route_length_m, config)` check. Continue to
flatten the clearance endpoint to the live root's Y and return the exact
table/object/obstacle route reason. Do not change `select_pick_slot`, which
retains initial admission distance enforcement.

- [ ] **Step 4: Keep travel telemetry but remove its finite-value cap**

In `ControllerPickAssist::observe`, retain segment calculation, accumulation,
and finite validation:

```cpp
diagnostics_.assisted_travel_m += travel_segment_m;
previous_observed_root_ = observation.displayed_root;
if (!is_finite(travel_segment_m) ||
    !is_finite(diagnostics_.assisted_travel_m)) {
    previous_observed_root_ = {};
    return fail_output(
        diagnostics_, PickAssistReason::OutsideTravelEnvelope);
}
```

Remove only the comparison against
`config_.maximum_assisted_path_m + kTravelEnvelopeToleranceM`. Continue using
the existing PickSlotConfig for collision geometry; do not change its 1 m
value at begin.

- [ ] **Step 5: Verify GREEN and approach/controller regressions**

Run:

```bash
make build/tests/test_interaction_pick_slots
build/tests/test_interaction_pick_slots
make build/tests/test_interaction_pick_assist
build/tests/test_interaction_pick_assist
make build/tests/test_interaction_smart_pickup_controller
build/tests/test_interaction_smart_pickup_controller
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
```

Expected: every command exits 0. Initial admission boundary, collision
failures, non-finite failures, deadline, cancellation, provenance, and exact
25 Hz contracts remain green.

- [ ] **Step 6: Commit and push Task 2**

Stage only:

```bash
git add interaction_pick_slots.cpp interaction_pick_assist.cpp \
  tests/cpp/test_interaction_pick_slots.cpp \
  tests/cpp/test_interaction_pick_assist.cpp
git commit -m "fix: decouple smart pickup travel telemetry"
git push checkpoint HEAD:g1-tabletop-placement
```
