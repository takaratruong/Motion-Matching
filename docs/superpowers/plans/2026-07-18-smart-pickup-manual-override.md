# Smart Pickup Manual Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make renewed keyboard movement cancel an in-progress pre-submission Smart Pickup attempt and return manual control on the same 25 Hz tick.

**Architecture:** Add one explicit movement-edge input and one consumed-result bit to the raylib-free `SmartPickupController` policy seam. Wire the four `WASD` press edges in the native controller and display the automatic-control/cancel contract without changing matching, animation, scheduler ownership, or timing.

**Tech Stack:** C++17, raylib keyboard polling, existing Makefile C++ tests, Python source-contract tests.

## Global Constraints

- The native controller and interaction runtime remain exactly 25 Hz.
- Flat locomotion only; do not add terrain behavior.
- `F` starts Smart Pickup from Locomotion; `X` cancels; `R` resets.
- A movement press cancels only a pending or active pre-submission Smart Pickup attempt.
- Simultaneous idle `F` plus movement starts Smart Pickup; the movement override does not win that activation tick.
- Cancellation passes the raw left stick, right stick, and strafe intent through on the same tick.
- Submitted/runtime-owned pickup, attachment, carry, and place behavior are unchanged.
- Do not touch the repo-root untracked `interaction_query_probe`; use `git status --untracked-files=no` and explicit `git add` paths only.
- Commit and push each reviewed task before proceeding.

---

### Task 1: Raylib-free manual override policy

**Files:**
- Modify: `interaction_smart_pickup_controller.h`
- Modify: `interaction_smart_pickup_controller.cpp`
- Test: `tests/cpp/test_interaction_smart_pickup_controller.cpp`

**Interfaces:**
- Consumes: existing `SmartPickupController::pre_step(const SmartPickupPreStepInput&)` and `SmartPickupAssistBackend::cancel()`.
- Produces: `SmartPickupPreStepInput::manual_override_pressed` and `SmartPickupPreStepResult::manual_override_consumed`.

- [ ] **Step 1: Write failing controller-policy tests**

Add focused tests that construct real `SmartPickupController` instances with the existing counting backend and assert these contracts:

```cpp
// Pending: F was consumed on the prior tick, but begin has not run.
interaction::SmartPickupPreStepInput override = make_pre_input(&target, false);
override.manual_override_pressed = true;
const interaction::SmartPickupPreStepResult cancelled =
    controller.pre_step(override);
require(cancelled.manual_override_consumed &&
        !cancelled.cancel_consumed &&
        same_vec3_bits(cancelled.left_stick, override.left_stick) &&
        same_vec3_bits(cancelled.right_stick, override.right_stick) &&
        backend.cancel_calls == 1U && backend.begin_calls == 0U,
        "pending Smart Pickup did not return manual control");
```

Cover active cancellation after one post-step `begin`, simultaneous idle `F` plus override (activation wins), and inactive `Idle`/`Failed`/`Submitted` backends (override is not consumed). Assert no cancelled path calls preview, observe, or submission after cancellation.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
make build/tests/test_interaction_smart_pickup_controller
```

Expected: compilation fails because the two manual-override fields do not exist.

- [ ] **Step 3: Add the minimal policy fields and precedence**

Add default-false fields:

```cpp
struct SmartPickupPreStepInput {
    RuntimeState runtime_state = RuntimeState::Locomotion;
    bool interact_pressed = false;
    bool cancel_pressed = false;
    bool manual_override_pressed = false;
    // existing fields unchanged
};

struct SmartPickupPreStepResult {
    bool interact_consumed = false;
    bool cancel_consumed = false;
    bool manual_override_consumed = false;
    // existing fields unchanged
};
```

In `pre_step`, preserve the existing explicit-cancel branch first. Immediately after it, handle renewed movement only when ownership existed before this call:

```cpp
const bool owns_existing_attempt =
    pending_activation_.has_value() ||
    backend_->active() ||
    backend_->owns_manual_interact();

if (input.manual_override_pressed && owns_existing_attempt) {
    pending_activation_.reset();
    previous_assist_output_ = {};
    backend_->cancel();
    result.manual_override_consumed = true;
    result.interact_consumed = input.interact_pressed;
    return result;
}
```

The result is initialized with the raw steering values before this branch, so the branch must not zero or replace them.

- [ ] **Step 4: Verify GREEN and focused regressions**

Run:

```bash
make build/tests/test_interaction_smart_pickup_controller
build/tests/test_interaction_smart_pickup_controller
make build/tests/test_interaction_pick_assist
build/tests/test_interaction_pick_assist
```

Expected: all commands exit 0 with no warnings or output from the test binaries.

- [ ] **Step 5: Review, commit, and push Task 1**

Stage only:

```bash
git add interaction_smart_pickup_controller.h \
  interaction_smart_pickup_controller.cpp \
  tests/cpp/test_interaction_smart_pickup_controller.cpp
git commit -m "fix: let movement cancel smart pickup assist"
git push
```

### Task 2: Native keyboard wiring and visible ownership state

**Files:**
- Modify: `controller.cpp`
- Modify: `interaction_debug_draw.h`
- Modify: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `tests/python/test_playable_interaction_evidence.py`

**Interfaces:**
- Consumes: Task 1's `manual_override_pressed` and `manual_override_consumed` fields.
- Produces: native `WASD` press-edge wiring and the visible `SMART PICKUP AUTO - WASD or X cancels` message.

- [ ] **Step 1: Write failing native integration contracts**

Extend `test_controller_smart_pickup_two_phase_production_seam()` to require one bounded edge expression and its assignment before `pre_step`:

```cpp
const bool manual_smart_pickup_override_pressed =
    IsKeyPressed(KEY_W) || IsKeyPressed(KEY_A) ||
    IsKeyPressed(KEY_S) || IsKeyPressed(KEY_D);
```

Require `manual_smart_pickup_pre_input.manual_override_pressed` to receive that value, require the diagnostics reset condition to include `manual_override_consumed`, and require the active-state message literal. Update the existing C++ and Python debug-draw assertions to expect:

```text
Interaction: F smart pickup/place  WASD/X cancel auto  R reset
```

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
make build/tests/test_interaction_controller_adapter
```

Expected: the test binary builds, then exits nonzero because the controller has no movement-edge wiring or updated copy.

Run the focused Python contract and expect the old-copy assertion to fail:

```bash
python3 -m unittest tests.python.test_playable_interaction_evidence
```

- [ ] **Step 3: Wire keyboard edges and active-state copy**

Compute the `WASD` edge once beside the existing `F`/`X`/`R` edges, assign it to the pre-step input before the sole `pre_step` call, and include `manual_override_consumed` in the bounded diagnostics-reset condition.

After reading `manual_smart_pickup_controller.diagnostics()` for drawing, show the active copy only for `SlotApproach`, `Settling`, `FinalPreview`, or `ReadyToSubmit`:

```cpp
DrawText(
    "SMART PICKUP AUTO - WASD or X cancels",
    340,
    242,
    18,
    ORANGE);
```

Update the permanent legend in `interaction_debug_draw.h`. Do not change scheduler cancel edges, post-step ordering, request IDs, or the preview callback.

- [ ] **Step 4: Verify native wiring, controller build, and regressions**

Run:

```bash
make build/tests/test_interaction_controller_adapter
build/tests/test_interaction_controller_adapter
python3 -m unittest tests.python.test_playable_interaction_evidence
make controller
make build/tests/test_interaction_smart_pickup_controller
build/tests/test_interaction_smart_pickup_controller
```

Expected: every command exits 0; controller compiles at the unchanged 25 Hz configuration.

- [ ] **Step 5: Review, commit, and push Task 2**

Stage only:

```bash
git add controller.cpp interaction_debug_draw.h \
  tests/cpp/test_interaction_controller_adapter.cpp \
  tests/python/test_playable_interaction_evidence.py
git commit -m "fix: return control when smart pickup is overridden"
git push
```

- [ ] **Step 6: Relaunch and manually verify**

Launch with the reviewed full pack and redirected flat feature output:

```bash
DISPLAY=:1 \
MM_INTERACTION_PACK=build/smart-pickup/full-pack \
MM_FEATURES_OUTPUT=build/smart-pickup/manual-flat-features.bin \
./controller
```

Verify at 25 Hz: `F` begins automatic approach, the orange ownership message appears, a newly pressed movement key immediately cancels and moves the character, `X` still cancels, `R` still resets, and the terrain-aware window/process remains untouched.
