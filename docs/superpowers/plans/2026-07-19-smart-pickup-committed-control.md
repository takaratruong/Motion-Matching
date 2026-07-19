# Smart Pickup Committed-Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a fresh movement command or `X` cancel a committed Smart Pickup before actual attachment, while showing whether the interaction is approaching, reaching, attached, carrying, or failed.

**Architecture:** Move the irreversible boundary from the assist's transport-only `Submitted` state to the runtime's authoritative attachment event. The runtime owns rollback before Contact; the native controller converts a fresh WASD edge into a runtime cancel only while cached Align/PickupReplay diagnostics are unattached, preserves raw steering on that tick, and renders phase-specific copy. Recorded post-attachment lift, Carry, placement, flat locomotion, and the 25 Hz clock remain unchanged.

**Tech Stack:** C++17 interaction runtime, raylib keyboard/controller loop, Makefile C++ tests, Python source-contract tests.

## Global Constraints

- Runtime and controller updates remain exactly 25 Hz (`dt == 1/25`).
- Flat locomotion only; do not read, modify, stop, or relaunch terrain-aware code or processes.
- `F` starts Smart Pickup/place, fresh `WASD` or `X` cancels only before attachment, and `R` resets.
- Cancellation before Contact releases the reservation, leaves the object free at its original transform, clears runtime pickup players, and returns ordinary locomotion ownership.
- Movement after attachment never silently detaches or teleports the object; the recorded lift continues to Hold and then Carry.
- `Submitted` alone is never displayed as pickup success.
- Do not change matcher costs, interaction clips/phases, playback speed, carry behavior, place behavior, or interaction pack data in this plan.
- Do not access, stat, hash, execute, modify, stage, or delete the repo-root untracked `interaction_query_probe`.
- Use `git status --untracked-files=no`; stage only exact files. Never use `git add -A` or `git add .`.
- Preserve the unrelated dirty `Makefile`, `tests/python/test_smart_pickup_full_pack_gate.py`, and exact untracked `tests/cpp/test_interaction_smart_pickup_scenarios.cpp`.
- Commit and push each reviewed task before proceeding.

---

### Task 1: Make the runtime attachment event the cancellation boundary

**Files:**
- Modify: `interaction_runtime.cpp`
- Test: `tests/cpp/test_interaction_runtime.cpp`

**Interfaces:**
- Consumes: `RuntimeInput::cancel_pressed`, `InteractionRuntime::ever_attached_`, the existing Align rollback behavior, and `TargetRegistry::release`.
- Produces: pre-contact PickupReplay cancellation with the same externally visible result as Align cancellation; attached PickupReplay remains committed.

- [ ] **Step 1: Replace the old post-commit cancellation test with two attachment-boundary tests**

In `tests/cpp/test_interaction_runtime.cpp`, replace
`test_cancel_is_ignored_after_commit_and_source_frame_is_monotonic()` with:

```cpp
void test_cancel_after_commit_before_contact_releases_pickup() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const Transform original =
        fixture.registry.find(fixture.request.target)->object_world;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    output = advance(runtime, fixture.locomotion);
    while (output.diagnostics.state == RuntimeState::Align) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::PickupReplay);
    assert(!output.diagnostics.attached);

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Cancelled);
    assert(output.diagnostics.reason == Reason::Cancelled);
    assert(output.diagnostics.object_state == ObjectState::Free);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
    assert(exact(
        fixture.registry.find(fixture.request.target)->object_world,
        original));
}

void test_cancel_after_contact_preserves_attached_pickup() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    output = advance(runtime, fixture.locomotion);
    while (!output.diagnostics.attached) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::PickupReplay ||
           output.diagnostics.state == RuntimeState::Hold);
    const int32_t attached_frame = output.diagnostics.frame;

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::PickupReplay ||
           output.diagnostics.state == RuntimeState::Hold ||
           output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.attached);
    assert(output.diagnostics.frame >= attached_frame);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}
```

Register both functions in `main()` in place of the removed test call.

- [ ] **Step 2: Run the runtime test and verify RED**

Run:

```bash
make build/tests/test_interaction_runtime
build/tests/test_interaction_runtime
```

Expected: compilation succeeds and the first new test aborts because an
unattached PickupReplay currently ignores cancel and remains pose-owned.

- [ ] **Step 3: Add the minimal pre-contact runtime rollback branch**

In `InteractionRuntime::update`, move the existing Align cancellation body
into a branch that precedes ordinary Align/PickupReplay processing:

```cpp
} else if (input.cancel_pressed && !ever_attached_ &&
           (state_ == RuntimeState::Align ||
            state_ == RuntimeState::PickupReplay)) {
    if (owns_reservation_) {
        (void)registry_->release(
            request_->target, request_->request_id);
    }
    owns_reservation_ = false;
    state_ = RuntimeState::Locomotion;
    diagnostics_.state = state_;
    diagnostics_.result = ResultCode::Cancelled;
    diagnostics_.reason = Reason::Cancelled;
    diagnostics_.object_state = ObjectState::Free;
    diagnostics_.attached = false;
    player_.reset();
    event_player_.reset();
    clearance_player_.reset();
    attachment_.reset();
    candidate_.reset();
    target_.reset();
    affordance_.reset();
} else if (state_ == RuntimeState::Align) {
```

Remove only the now-duplicated inner Align `if (input.cancel_pressed)` block;
retain its target-change validation and playback logic under the ordinary
Align branch. The authoritative `!ever_attached_` guard must remain in the
runtime even though the native controller will also gate movement-derived
cancel edges.

- [ ] **Step 4: Verify GREEN and focused regressions**

Run:

```bash
make build/tests/test_interaction_runtime
build/tests/test_interaction_runtime
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
```

Expected: every command exits 0; no test asserts that pre-contact committed
PickupReplay ignores cancellation.

- [ ] **Step 5: Commit and push Task 1**

Stage only:

```bash
git add interaction_runtime.cpp tests/cpp/test_interaction_runtime.cpp
git commit -m "fix: cancel committed pickup before attachment"
git push checkpoint HEAD:g1-tabletop-placement
```

---

### Task 2: Route fresh movement to runtime cancellation and show truthful state

**Files:**
- Modify: `controller.cpp`
- Modify: `interaction_debug_draw.h`
- Test: `tests/cpp/test_interaction_controller_adapter.cpp`
- Test: `tests/python/test_playable_interaction_evidence.py`

**Interfaces:**
- Consumes: Task 1's runtime cancellation semantics, the existing
  `manual_smart_pickup_override_pressed` edge, cached `RuntimeOutput`, raw
  keyboard/gamepad stick input, and existing assist/runtime diagnostics.
- Produces: a movement-derived scheduler cancel edge only for unattached
  Align/PickupReplay; raw steering on that tick; phase-specific banners and a
  truthful permanent control legend.

- [ ] **Step 1: Add failing native source contracts**

Extend the existing Smart Pickup controller-source assertions in
`tests/cpp/test_interaction_controller_adapter.cpp` and
`tests/python/test_playable_interaction_evidence.py` to require:

```cpp
const vec3 raw_gamepadstick_left = gamepadstick_left;
const bool committed_pickup_manual_override =
    manual_smart_pickup_override_pressed &&
    !cached_interaction_output.diagnostics.attached &&
    (cached_interaction_state == interaction::RuntimeState::Align ||
     cached_interaction_state == interaction::RuntimeState::PickupReplay);
```

Require the steering suppression guard to exclude that condition, require the
condition to set `interaction_edges.cancel_pressed = true`, and require the
raw left stick to be restored. Require these exact phase strings:

```text
SMART PICKUP REACH - WASD or X cancels
SMART PICKUP ATTACHED - finishing recorded lift
CARRY - WASD moves, F places, R resets
PICKUP FAILED: %s - reposition and press F
Interaction: F smart pickup/place  WASD/X cancel before attach  R reset
```

Also require banner selection to inspect runtime `attached` and result/reason,
not only `PickAssistState::Submitted`.

- [ ] **Step 2: Run native/source tests and verify RED**

Run:

```bash
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
python3 -m unittest tests.python.test_playable_interaction_evidence
```

Expected: the C++ source contract or Python contract fails because committed
movement cancellation, raw-left-stick restoration, and phase copy are absent.

- [ ] **Step 3: Wire state-aware movement cancellation before locomotion**

In `controller.cpp`, preserve the left stick immediately after input polling:

```cpp
const vec3 raw_gamepadstick_left = gamepadstick_left;
```

Bind the cached output once and derive the committed pre-contact condition:

```cpp
const interaction::RuntimeOutput& cached_interaction_output =
    interaction_scheduler.cached_output();
const interaction::RuntimeState cached_interaction_state =
    cached_interaction_output.diagnostics.state;
const bool committed_pickup_manual_override =
    manual_smart_pickup_override_pressed &&
    !cached_interaction_output.diagnostics.attached &&
    (cached_interaction_state == interaction::RuntimeState::Align ||
     cached_interaction_state == interaction::RuntimeState::PickupReplay);
if (committed_pickup_manual_override) {
    interaction_edges.cancel_pressed = true;
    gamepadstick_left = raw_gamepadstick_left;
}
```

Change the existing suppression condition to:

```cpp
if (cached_interaction_output.suppress_steering &&
    !committed_pickup_manual_override) {
    gamepadstick_left = vec3();
}
```

Order the two branches so movement-derived cancellation cannot be zeroed by
the cached PickupReplay output. Do not make held Carry or attached replay
cancellable by movement. Leave explicit `X` as the existing scheduler cancel
edge; Task 1's runtime is the authoritative pre/post-contact gate.

- [ ] **Step 4: Draw phase-specific state without equating submission to pickup**

Keep the existing automatic-approach banner for active assist states. Add
exclusive runtime banners for unattached Align/PickupReplay, attached
PickupReplay/Hold, and Carry. Add failure copy for assist `Failed`, and for a
runtime Locomotion result of Rejected or Failed when the assist itself is not
providing a more specific failure. Use the exact strings from Step 1 and the
existing reason-name functions.

Update the permanent legend in `interaction_debug_draw.h` to:

```cpp
"Interaction: F smart pickup/place  WASD/X cancel before attach  R reset"
```

Do not render a success banner from `PickAssistState::Submitted` alone.

- [ ] **Step 5: Verify GREEN, controller build, and 25 Hz regressions**

Run:

```bash
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
python3 -m unittest tests.python.test_playable_interaction_evidence
make build/tests/test_interaction_runtime
build/tests/test_interaction_runtime
make controller
```

Expected: all commands exit 0, the native controller builds, and existing
fixed-rate assertions still require exactly 25 Hz.

- [ ] **Step 6: Commit and push Task 2**

Stage only:

```bash
git add controller.cpp interaction_debug_draw.h \
  tests/cpp/test_interaction_controller_adapter.cpp \
  tests/python/test_playable_interaction_evidence.py
git commit -m "fix: retain control until smart pickup attaches"
git push checkpoint HEAD:g1-tabletop-placement
```

- [ ] **Step 7: Relaunch only the Smart Pickup controller for manual acceptance**

Stop only the controller launched from this worktree, rebuild it from reviewed
HEAD, and launch:

```bash
DISPLAY=:1 \
MM_INTERACTION_PACK=build/smart-pickup/full-pack \
MM_FEATURES_OUTPUT=build/smart-pickup/manual-flat-features.bin \
./controller
```

Without injecting global keyboard input, verify the process remains at 25 Hz
and the window displays the new copy. Ask the user to perform the acceptance
edges: `F`, then fresh WASD before Contact; `F`, wait for visible attachment;
then move in Carry. Do not stop or signal the terrain-aware process.
