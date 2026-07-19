# Smart Pickup Preview-Certified Entries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` task-by-task, with `test-driven-development`
> for every production change and an independent review before each push.

**Goal:** Make Smart Pickup choose a collision-clear authored entry whose
recorded pickup motion is compatible with the character's live pose, instead
of committing the geometry-nearest entry before matching it.

**Architecture:** Preserve authored slot order and the legacy geometry-only
winner for diagnostics, while adding deterministic ranked eligible indices.
Migrate the assist/controller preview handshake atomically from one optional
preview to an ordered batch. At activation, preview every geometry-eligible
entry on one exact flat-locomotion snapshot, filter by readiness, freeze the
first certified entry in integer geometry order, and retain the existing
one-way approach, final preview, authoritative runtime preflight, attachment,
and carry path.

**Tech Stack:** C++17, deterministic authored-slot geometry, Smart Pickup
assist/controller policy, existing interaction matcher/runtime, Makefile C++
tests, Python full-pack gate, raylib live controller.

## Global constraints

- Controller, matcher observations, and runtime remain exactly 25 Hz.
- Flat locomotion only. Do not modify, signal, focus, or otherwise touch the
  terrain process/window.
- Do not change playback, attachment, Carry, placement, runtime Preflight,
  `PickRequest`, the preview callback signature, or matcher authority.
- `PickSlotSelection::ordered` stays in authored order and legacy
  `selected_index` stays the geometry-only winner.
- Preview readiness is an eligibility filter; certified entries are ranked by
  `(route_millimetres, heading_milliradians, slot_id)`, never floating cost.
- Freeze exactly once. Never switch slots after approach begins.
- A preview batch is atomic and tied to one exact live snapshot/fingerprint.
- Partial/misordered/duplicated/malformed batches fail closed or retry exactly
  as specified; they never yield a partial winner.
- Keep the existing arrival deadline, WASD/`X` pre-attachment cancellation,
  target/affordance/slot mutation checks, and collision revalidation.
- Do not access, stat, hash, execute, modify, stage, or delete the repo-root
  untracked `interaction_query_probe`.
- Use `git status --untracked-files=no`; stage explicit paths only; never use
  `git add -A` or `git add .`.
- Preserve unrelated dirty `Makefile`,
  `tests/python/test_smart_pickup_full_pack_gate.py`, and exact untracked
  `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`.
- Do not run aggregate `make test-cpp`, `make test-python`,
  `make test-interaction`, `make test-interaction-safe`,
  `make gate-playable-interaction`, or `make gate-place-headless` targets in
  this worktree. They currently include the untracked scenario source or build
  the forbidden repo-root probe. Use only the explicit commands below.
- Commit and push every reviewed task before proceeding.

---

### Task 1: Expose deterministic geometry-eligible slot ranking

**Files:**
- Modify: `interaction_pick_slots.h`
- Modify: `interaction_pick_slots.cpp`
- Test: `tests/cpp/test_interaction_pick_slots.cpp`

**Interface:** Add
`std::vector<size_t> PickSlotSelection::ranked_eligible_indices`. It contains
only entries whose `reason == PickSlotReason::None`, sorted by the existing
integer tuple `(route_millimetres, heading_milliradians, id)`. Do not reorder
`ordered`; keep `selected_index == ranked_eligible_indices.front()` when the
vector is non-empty.

- [ ] Add RED tests proving authored order is retained, blocked/outside entries
  are absent, route then heading then slot-ID breaks ties, the legacy selected
  index remains the first ranked entry, and repeated calls are bit-stable.
- [ ] Run and confirm RED:

  ```bash
  make build/tests/test_interaction_pick_slots
  build/tests/test_interaction_pick_slots
  ```

- [ ] Populate and sort the index vector only after all authored entries have
  been mapped. Reuse existing integer keys; do not compare floats in the sort.
- [ ] Compile and run the now-GREEN source directly under release/fast-math
  without editing the dirty Makefile:

  ```bash
  g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. \
    -Wno-array-bounds -O3 -DNDEBUG -ffast-math \
    tests/cpp/test_interaction_pick_slots.cpp \
    interaction_pick_slots.cpp interaction_target.cpp interaction_pose.cpp \
    -o /tmp/test_interaction_pick_slots_release_fast_math
  /tmp/test_interaction_pick_slots_release_fast_math
  ```
- [ ] Run GREEN plus direct dependents:

  ```bash
  make build/tests/test_interaction_pick_slots
  build/tests/test_interaction_pick_slots
  make build/tests/test_interaction_pick_assist
  build/tests/test_interaction_pick_assist
  make build/tests/test_interaction_smart_pickup_controller
  build/tests/test_interaction_smart_pickup_controller
  ```

- [ ] Independently review, then explicitly stage the three task files, commit
  as `feat: rank eligible smart pickup entries`, and push
  `checkpoint HEAD:g1-tabletop-placement`.

---

### Task 2: Atomically migrate the preview handshake to ordered batches

**Files:**
- Modify: `interaction_pick_assist.h`
- Modify: `interaction_pick_assist.cpp`
- Modify: `interaction_smart_pickup_controller.cpp`
- Test: `tests/cpp/test_interaction_pick_assist.cpp`
- Test: `tests/cpp/test_interaction_smart_pickup_controller.cpp`
- Test: `tests/cpp/test_live_flat_pick_entry_oracle.cpp`
- Test source contracts: `tests/python/test_playable_interaction_evidence.py`
- Test source contracts: `tests/python/test_playable_placement_evidence.py`

**Interfaces:** Add:

```cpp
struct PickAssistPreviewRequest {
    uint32_t slot_id = 0U;
    PickEntryRoot root{};
};

struct PickAssistPreviewResult {
    PickAssistPreviewRequest request{};
    std::optional<PickEntryPreview> preview{};
};
```

Replace `PickAssistOutput::{needs_preview,preview_root}` with
`std::vector<PickAssistPreviewRequest> preview_requests`. Replace
`PickAssistObservation::preview` with
`std::vector<PickAssistPreviewResult> preview_results`; retain one
`preview_snapshot_fingerprint` for the entire batch. FinalPreview uses a
one-element batch. The external `SmartPickupPreviewCallback` signature and
runtime API stay unchanged. Add `<vector>` directly to
`interaction_pick_assist.h`; do not rely on `interaction_pick_slots.h` to
provide it transitively.

- [ ] First update tests to express the vector protocol while preserving the
  current single frozen-slot behavior. Require FinalPreview to emit one request
  with the frozen slot ID/root and to consume only an exact one-result response.
- [ ] Add controller RED tests proving every request in the prior output is
  evaluated in order against the same `live_flat_snapshot`, echoed into the
  corresponding result, and associated with the one computed fingerprint.
- [ ] Migrate the live-flat oracle and its two focused Python source-contract
  tests in the same atomic change. Retain the 25 Hz/post-step/one-observation
  guarantees while replacing stale singular-preview assumptions.
- [ ] Run and confirm RED/compile failure is caused only by missing batch API:

  ```bash
  make build/tests/test_interaction_pick_assist
  make build/tests/test_interaction_smart_pickup_controller
  ```

- [ ] Migrate the assist and controller in one atomic change. Empty callback
  results still occupy their ordered result element as `preview=nullopt`; they
  must not shorten/reorder the batch.
- [ ] Preserve one `observe` call per fixed tick. The activation tick still
  makes zero matcher callbacks; legacy FinalPreview makes exactly one.
- [ ] Run GREEN and regressions:

  ```bash
  make build/tests/test_interaction_pick_assist
  build/tests/test_interaction_pick_assist
  make test-interaction-pick-assist-release-fast-math
  make build/tests/test_interaction_smart_pickup_controller
  build/tests/test_interaction_smart_pickup_controller
  make build/tests/test_interaction_runtime
  build/tests/test_interaction_runtime
  make gate-live-flat-pick-entry-oracle gate-live-flat-pick-position-matrix
  python -m unittest \
    tests.python.test_playable_interaction_evidence.Task12PolicyTests.test_manual_smart_pickup_activation_uses_only_the_25_hz_clock \
    tests.python.test_playable_interaction_evidence.Task12PolicyTests.test_manual_pick_activation_observes_post_step_and_caches_next_tick_output \
    tests.python.test_playable_interaction_evidence.Task12PolicyTests.test_real_pack_oracle_covers_manual_assist_route_and_one_shot_handoff \
    tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_shared_smart_pickup_coordinator_invokes_frozen_preview_once \
    tests.python.test_playable_placement_evidence.Task8PlacementPolicyTests.test_native_smart_pickup_preview_callback_is_one_direct_delegation \
    -v
  ```

- [ ] Independently review, explicitly stage only the eight task files, commit
  as `refactor: batch smart pickup previews`, and push the checkpoint branch.

---

### Task 3: Certify all eligible entries before freezing one

**Files:**
- Modify: `interaction_pick_assist.h`
- Modify: `interaction_pick_assist.cpp`
- Modify: `interaction_smart_pickup_controller.cpp` only if required by the
  already-migrated batch protocol
- Test: `tests/cpp/test_interaction_pick_assist.cpp`
- Test: `tests/cpp/test_interaction_smart_pickup_controller.cpp`

**State/API additions:** Append, without renumbering existing values,
`PickAssistState::SlotSelectionPreview = 7` and
`PickAssistReason::SelectionPreviewRejected = 15`. Add these explicit
diagnostic shapes (field names may only change during review if all evidence is
retained):

```cpp
struct PickAssistSelectionPreviewDiagnostics {
    PickAssistPreviewRequest request{};
    std::optional<PickEntryRoot> prospective_root{};
    uint64_t observation_snapshot_fingerprint = 0U;
    uint64_t preview_snapshot_fingerprint = 0U;
    bool available = false;
    bool all_preview_roots_finite = false;
    bool fingerprint_equal = false;
    bool path_feasible = false;
    Reason path_reason = Reason::None;
    bool match_ready = false;
    Reason match_reason = Reason::None;
    bool prospective_root_equal = false;
    int32_t feasible_entry_frame = -1;
    int32_t contact_frame = -1;
    float total_cost = 0.0F;
};

// PickAssistDiagnostics additions
std::optional<size_t> frozen_slot_index{};
std::vector<PickAssistSelectionPreviewDiagnostics> selection_previews{};
uint32_t selection_preview_epochs = 0U;
uint32_t selection_preview_calls = 0U;
bool selection_poor_match_observed = false;
```

- [ ] Add assist RED tests for all required behavior:
  - `begin` maps K geometry-eligible entries but freezes none;
  - activation observation emits K requests in ranked order;
  - nearest incompatible plus farther ready freezes the farther entry;
  - when two entries are ready, integer geometry order wins even if costs are
    reversed;
  - an incomplete batch selects nothing and re-requests all K;
  - repeated incomplete/empty-callback epochs remain stationary and terminate
    at the existing bounded deadline without ever selecting a partial result;
  - misordered, duplicate, wrong-slot/root/fingerprint, non-finite, and mutated
    target/affordance/slot evidence fails closed;
  - all exact retryable PoorMatch results retry K until the existing deadline,
    then fail `PoorMatch`;
  - a mixed complete epoch with at least one exact retryable PoorMatch and one
    hard rejection retries the entire K-entry batch rather than failing early;
  - a complete hard-rejected batch fails `SelectionPreviewRejected`;
  - every candidate route is revalidated from the current displayed root after
    `begin`; a route that becomes table/obstacle-blocked cannot certify;
  - after freeze, later evidence never switches the slot;
  - the ordinary arrival counter resets when the slot freezes.
- [ ] Add controller RED call-count tests: activation `0`; each selection epoch
  exactly `K` in request order on one bit-identical snapshot/fingerprint;
  Approach/Settling `0`; each FinalPreview `1`; Submitted/Failed `0`. Total is
  exactly `E*K + F`.
- [ ] Run RED:

  ```bash
  make build/tests/test_interaction_pick_assist
  build/tests/test_interaction_pick_assist
  make build/tests/test_interaction_smart_pickup_controller
  build/tests/test_interaction_smart_pickup_controller
  ```

- [ ] Change `begin` to snapshot and geometry-filter only, enter
  SlotSelectionPreview, and defer all frozen-slot/selected-slot assignment.
- [ ] Validate a complete batch atomically against frozen provenance and the
  observation fingerprint. A result certifies only if path and match are both
  ready/None and its returned prospective root is finite and bit-equal.
- [ ] Choose the first certified `ranked_eligible_indices` entry, set the
  separate actual frozen index/slot ID/root once, reset the approach counter,
  and enter SlotApproach.
- [ ] Implement the exact retry/hard-failure semantics and keep singular
  frozen-slot FinalPreview terminal on hard rejection.
- [ ] Update state/reason name functions and static numeric-contract tests.
- [ ] Run GREEN and broader regressions:

  ```bash
  make build/tests/test_interaction_pick_slots
  build/tests/test_interaction_pick_slots
  make build/tests/test_interaction_pick_assist
  build/tests/test_interaction_pick_assist
  make test-interaction-pick-assist-release-fast-math
  make build/tests/test_interaction_smart_pickup_controller
  build/tests/test_interaction_smart_pickup_controller
  make build/tests/test_interaction_runtime
  build/tests/test_interaction_runtime
  make build/tests/test_interaction_controller_adapter
  build/tests/test_interaction_controller_adapter
  ```

- [ ] Independently review, explicitly stage only task files, commit as
  `feat: certify smart pickup entries before approach`, and push.

---

### Task 4: Prove full-pack attachment and refresh the live verifier

**Files:**
- Modify: `tests/cpp/test_live_flat_pick_entry_oracle.cpp`
- Modify: `controller.cpp` for truthful automatic-phase diagnostics; do not
  change controls or control frequency.
- Modify only the focused existing source-contract assertions in:
  `tests/python/test_playable_interaction_evidence.py` and
  `tests/python/test_playable_placement_evidence.py` if controller text/source
  checks require it.
- Test existing tracked sources; do not edit the dirty Python full-pack gate.
- Verification artifacts outside git:
  `/home/ubuntu/projects/motion-matching-verification/g1-tabletop-placement/`

- [ ] Mandatorily extend `run_manual_pick_assist_oracle` in
  `tests/cpp/test_live_flat_pick_entry_oracle.cpp`: use real full-pack
  `runtime.preview_pick` results for all selection requests; make the
  geometry-nearest entry non-ready while a farther authored entry is ready;
  require the farther entry to freeze; drive ordinary 25 Hz flat locomotion to
  submission; pass the unchanged `PickRequest` through ordinary runtime
  Preflight; tick recorded replay through contact; and require
  `object_state == ObjectState::Held`, `attached == true`,
  `state == RuntimeState::Carry`, plus a subsequent Carry steering tick. The
  test must prove no preview candidate/clip was copied into request authority.
- [ ] Update `controller.cpp` to display the actual `frozen_slot_index`/slot ID,
  not legacy geometry-only `selected_index`, and include
  `SlotSelectionPreview` in `SMART PICKUP AUTO - WASD or X cancels`.
- [ ] Run the complete focused verification matrix from Tasks 1-3, then:

  ```bash
  make \
    build/tests/test_live_flat_pick_entry_oracle \
    build/tests/test_live_flat_pick_entry_oracle_release_fast_math \
    build/tests/test_interaction_smart_pickup_scene \
    interaction_smart_pickup_preview_probe \
    interaction_smart_pickup_scene_probe
  build/tests/test_interaction_smart_pickup_scene
  normal_output="$(timeout --signal=TERM --kill-after=2s 30s \
    build/tests/test_live_flat_pick_entry_oracle \
    resources/database.bin build/smart-pickup/full-pack)"
  release_output="$(timeout --signal=TERM --kill-after=2s 30s \
    build/tests/test_live_flat_pick_entry_oracle_release_fast_math \
    resources/database.bin build/smart-pickup/full-pack)"
  test "$normal_output" = "$release_output"
  SMART_PICKUP_FULL_PACK="$PWD/build/smart-pickup/full-pack" \
  SMART_PICKUP_PREVIEW_PROBE="$PWD/interaction_smart_pickup_preview_probe" \
  SMART_PICKUP_SCENE_PROBE="$PWD/interaction_smart_pickup_scene_probe" \
  python -m unittest \
    tests.python.test_smart_pickup_full_pack_gate.SmartPickupFullPackGateTests.test_full_pack_preview_and_compiled_scene_are_identical \
    -v
  make controller
  ```

  A skipped external gate is not evidence of attachment; use the configured
  full-pack assets under `build/smart-pickup/full-pack` for the integration run.
- [ ] Record the exact current Smart Pickup unified-exec session, PID, and
  window ID. Send Ctrl-C only through that exact session; verify only that PID
  and window disappeared. Do not use broad `pkill`, `killall`, title-based
  window close, or global keyboard input. Confirm the terrain PID/window is
  unchanged before relaunching only Smart Pickup with:

  ```bash
  DISPLAY=:1 \
  MM_INTERACTION_PACK=build/smart-pickup/full-pack \
  MM_FEATURES_OUTPUT=build/smart-pickup/manual-flat-features.bin \
  ./controller
  ```

  Never inject global/synthetic keyboard events. Leave the terrain process and
  window untouched.
- [ ] Verify at 25 Hz that the UI distinguishes selection preview, approach,
  reach, attachment, carry, and failure. A successful pickup requires runtime
  `object=Held` and `attached=1`; submitted/animated alone is a failure.
- [ ] Record a short external verification artifact showing walk -> preview-
  certified approach -> contact -> attached carry. Record the exact commit,
  pack identity, PID/window, and outcome alongside it.
- [ ] If controller diagnostics changed, independently review, explicitly stage
  only `tests/cpp/test_live_flat_pick_entry_oracle.cpp`, `controller.cpp`, and
  any two focused Python source-contract files actually changed, commit as
  `test: verify preview-certified smart pickup`, and push. Otherwise record the
  verified commit in the ignored progress ledger and push no empty commit.

## Completion gate

Do not call Smart Pickup fixed merely because the animation plays. Completion
requires fresh automated suites, independent reviews, a pushed checkpoint,
and a full-pack run whose runtime truth is `object=Held`, `attached=1`, followed
by controllable Carry. Report separately any remaining sensitivity to object
position or authored entry coverage.
