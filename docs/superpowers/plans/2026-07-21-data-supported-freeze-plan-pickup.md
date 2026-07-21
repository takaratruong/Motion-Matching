# Data-Supported Freeze, Plan, and Pickup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze on `F`, build a collision-checked current-root-to-pickup chain around the measured near-object data, then resume once and follow it without an entry or terminal planning stop.

**Architecture:** The production proposal `wait()` is the first planning barrier: it blocks the controller update, so simulation and animation cannot advance while Python samples. A data-supported entry at 0.45-to-0.75 m is joined to the exact frozen root by a collision-certified route, then concatenated with the learned funnel into one variable-length spatial follower route. Matcher preview is consumed in the same controller update before the barrier releases, and the terminal submits pickup immediately rather than entering settle/final-preview states.

**Tech Stack:** C++17, existing 25 Hz motion-matching controller, existing Python DDIM worker at 50 steps, Raylib visualizer.

## Global Constraints

- Do not modify or stage `resources/features.bin` or generated `build/` artifacts.
- Existing authored Smart Pickup behavior remains unchanged.
- Learned generation is right-hand only and uses entry radius 0.45-to-0.75 m.
- Planned entry speed is finite, nonzero, and clamped to 0.05-to-0.30 m/s.
- No simulation update occurs between launching generation, consuming proposal previews, and arming the complete route.
- The unified route starts at the exact frozen root and has no zero-speed entry waypoint.
- Production remains at 50 DDIM sampling steps.

---

### Task 1: Variable-Length Unified Route Follower

**Files:**
- Modify: `interaction_funnel_follower.h`
- Modify: `interaction_funnel_follower.cpp`
- Modify: `tests/cpp/test_interaction_funnel_follower.cpp`

**Interfaces:**
- Consumes: existing `FunnelSample` and fixed learned-knot expansion.
- Produces: `using FunnelRoute = std::vector<FunnelSample>` and a follower constructor that accepts a complete route plus configurable terminal settle count.

- [ ] **Step 1: Write the failing variable-route test**

Add a test that constructs a 3 m route with more than 75 points, verifies forward lookahead from the exact first point, and verifies completion with `required_terminal_ticks = 1U`:

```cpp
std::vector<FunnelSample> route;
for (int i = 0; i <= 150; ++i) {
    route.push_back(sample_at(0.02F * i));
}
InteractionFunnelFollower follower(11U, route, 40U, 1U);
assert(follower.tick({40U, route.front()}).published);
const auto done = follower.tick({41U, route.back()});
assert(done.state == FunnelFollowerState::Completed);
```

- [ ] **Step 2: Run the test and verify RED**

Run: `make build/tests/test_interaction_funnel_follower && build/tests/test_interaction_funnel_follower`

Expected: compilation fails because the vector-route constructor is absent.

- [ ] **Step 3: Generalize the follower**

Add:

```cpp
using FunnelRoute = std::vector<FunnelSample>;

InteractionFunnelFollower(
    uint64_t proposal_seed,
    FunnelRoute route,
    uint64_t first_tick_index,
    uint32_t required_terminal_ticks = kFunnelRequiredTerminalTicks);
```

Store `FunnelRoute targets_` and `required_terminal_ticks_`. Replace fixed terminal indices with `targets_.size() - 1`; validate at least two finite route samples and a positive settle count. Keep the fixed-knot constructor by delegating through `expand_funnel_execution` so all existing callers and tests retain their behavior.

- [ ] **Step 4: Run follower tests and verify GREEN**

Run: `make build/tests/test_interaction_funnel_follower && build/tests/test_interaction_funnel_follower`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add interaction_funnel_follower.h interaction_funnel_follower.cpp tests/cpp/test_interaction_funnel_follower.cpp
git commit -m "feat: follow variable-length pickup routes"
```

---

### Task 2: Data-Supported Complete Route Construction

**Files:**
- Create: `interaction_funnel_plan.h`
- Create: `interaction_funnel_plan.cpp`
- Create: `tests/cpp/test_interaction_funnel_plan.cpp`
- Modify: `interaction_learned_pickup_backend.h`
- Modify: `interaction_learned_pickup_backend.cpp`
- Modify: `interaction_learned_pickup_diagnostics.h`
- Modify: `interaction_smart_pickup_controller.h`
- Modify: `tests/cpp/test_interaction_learned_pickup_backend.cpp`

**Interfaces:**
- Consumes: exact activation root, collision-safe capture candidate, planned entry velocity, certified proposal.
- Produces: `plan_funnel_approach(...)`, one `FunnelRoute selected_route_object_`, one `std::vector<FunnelSample> world_route`, and `PickAssistOutput::planning_barrier` while frozen planning is incomplete.

- [ ] **Step 1: Replace the moving-prefetch test with a failing frozen-plan test**

From a moving root at `(0, 0, 3)`, require the first observation to:

```cpp
assert(provider.begin_calls == 1);
assert(provider.wait_calls == 1);
assert(std::hypot(condition[18], condition[19]) >= 0.45F);
assert(std::hypot(condition[18], condition[19]) <= 0.75F);
assert(std::hypot(condition[22], condition[23]) >= 0.05F);
assert(std::hypot(condition[22], condition[23]) <= 0.30F);
assert(output.planning_barrier);
assert(!output.stationary_constraint);
```

After valid previews, require `FunnelFollow` immediately, with debug route first point equal to the frozen 3 m root and a later point equal to the diffusion entry.

- [ ] **Step 2: Run the backend test and verify RED**

Run: `make build/tests/test_interaction_learned_pickup_backend && build/tests/test_interaction_learned_pickup_backend`

Expected: failure because current code polls asynchronously, waits at `AwaitEntry`, and exposes only the 75-tick learned route.

- [ ] **Step 3: Test and implement deterministic collision planning**

Add `tests/cpp/test_interaction_funnel_plan.cpp` with direct-path and blocked-path cases. The blocked case places an inflated axis-aligned obstacle across the direct segment and requires a deterministic detour around visible inflated corners, with every returned segment passing `revalidate_frozen_pick_slot`.

Define:

```cpp
struct FunnelApproachPlan {
    bool feasible = false;
    Transform entry_world{};
    std::vector<Transform> waypoints_world{};
};

FunnelApproachPlan plan_funnel_approach(
    Transform frozen_root,
    const InteractionTarget& target,
    const std::vector<PickNavigationObstacle>& obstacles,
    const FunnelCaptureConfig& capture);
```

Try collision-safe entry bearings in existing deterministic capture order. For each candidate, accept a direct segment when valid; otherwise build a visibility graph from the root, entry, and inflated obstacle corners and run stable Dijkstra with node index as the tie-break. Reject any edge that fails existing route revalidation. Add the new test target and source dependencies to `Makefile`.

Run: `make build/tests/test_interaction_funnel_plan && build/tests/test_interaction_funnel_plan`

Expected: exit 0 after implementation.

- [ ] **Step 4: Launch blocking generation from the activation frame**

Add config fields:

```cpp
float supported_minimum_entry_radius_m = 0.45F;
float supported_maximum_entry_radius_m = 0.75F;
float supported_minimum_entry_speed_mps = 0.05F;
float supported_maximum_entry_speed_mps = 0.30F;
```

At the first `CoarseCapture` observation, select a collision-safe capture using a copy of `FunnelCaptureConfig` capped at `0.75F`. Freeze `observation.displayed_root` separately from the planned entry. Build the condition from the planned entry and a velocity directed along the final approach segment whose magnitude is `clamp(live_speed, 0.05F, 0.30F)`. Launch once and call `provider_->wait()`.

- [ ] **Step 5: Construct one complete object-local route**

Add a helper with this contract:

```cpp
FunnelRoute build_complete_route(
    Transform object_world,
    Transform frozen_root_world,
    Transform entry_world,
    const FunnelProposal& proposal);
```

Interpolate every collision-certified approach-polyline segment at no more than `0.08F` translation per sample, using shortest-arc yaw interpolation. Append `expand_funnel_execution(proposal.samples)` while skipping a duplicate entry point. The route must start bit-for-bit at the frozen root transformed into object-local coordinates.

- [ ] **Step 6: Arm immediately after selection**

Remove learned `AwaitEntry` behavior. After preview selection, build the complete route and construct:

```cpp
follower_.emplace(
    proposal.seed,
    selected_route_object_,
    observation.controller_tick + 1U,
    1U);
```

Set `FunnelFollow` immediately. Change learned debug route storage from fixed `FunnelExecutionTargets` to `std::vector<FunnelSample>` and render the complete world route.

- [ ] **Step 7: Verify backend GREEN**

Run: `make build/tests/test_interaction_learned_pickup_backend && build/tests/test_interaction_learned_pickup_backend`

Expected: exit 0.

- [ ] **Step 8: Commit**

```bash
git add interaction_funnel_plan.h interaction_funnel_plan.cpp tests/cpp/test_interaction_funnel_plan.cpp Makefile interaction_learned_pickup_backend.cpp interaction_learned_pickup_backend.h interaction_learned_pickup_diagnostics.h interaction_smart_pickup_controller.h tests/cpp/test_interaction_learned_pickup_backend.cpp
git commit -m "feat: build complete frozen pickup routes"
```

---

### Task 3: Consume Matcher Preview Before Releasing the Barrier

**Files:**
- Modify: `interaction_pick_assist.h`
- Modify: `interaction_smart_pickup_controller.cpp`
- Modify: `tests/cpp/test_interaction_smart_pickup_controller.cpp`

**Interfaces:**
- Consumes: `PickAssistOutput::planning_barrier` and preview requests returned after blocking generation.
- Produces: a single `post_step()` result whose backend has already consumed previews and armed the complete route.

- [ ] **Step 1: Write a failing same-step preview test**

Use a fake backend whose first `observe()` returns one preview request with `planning_barrier = true` and whose second consumes the result and returns steering. Require one `post_step()` call to invoke the backend twice, call preview once with the same snapshot fingerprint, and return the second steering output.

- [ ] **Step 2: Run the controller test and verify RED**

Run: `make build/tests/test_interaction_smart_pickup_controller && build/tests/test_interaction_smart_pickup_controller`

Expected: assertion failure because previews are currently delayed until the next controller tick.

- [ ] **Step 3: Add bounded same-step preview consumption**

After the first backend observation, if `planning_barrier` is true and preview requests are present, synchronously invoke the callback with the unchanged `live_flat_snapshot`, populate preview results and fingerprint, and call `observe()` once more. Reject a second barrier output to prevent an unbounded loop. Authored outputs leave `planning_barrier` false and preserve their existing one-tick behavior.

- [ ] **Step 4: Run Smart Pickup controller tests and verify GREEN**

Run: `make build/tests/test_interaction_smart_pickup_controller && build/tests/test_interaction_smart_pickup_controller`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add interaction_pick_assist.h interaction_smart_pickup_controller.cpp tests/cpp/test_interaction_smart_pickup_controller.cpp
git commit -m "feat: complete learned planning in one controller step"
```

---

### Task 4: Rolling Pickup Submission Without Terminal Stop

**Files:**
- Modify: `interaction_learned_pickup_backend.h`
- Modify: `interaction_learned_pickup_backend.cpp`
- Modify: `interaction_smart_pickup_controller.h`
- Modify: `interaction_smart_pickup_controller.cpp`
- Modify: `interaction_runtime.h`
- Modify: `interaction_runtime.cpp`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_interaction_learned_pickup_backend.cpp`
- Modify: `tests/cpp/test_interaction_learned_pickup_end_to_end.cpp`

**Interfaces:**
- Consumes: the frozen selection preview and one-tick terminal completion from Task 2.
- Produces: a certified learned submission carrying the frozen `PickEntryPreview`, plus direct `submit_interact` from route completion; no learned `FinalPreview` state or three-tick settling.

- [ ] **Step 1: Write a failing direct-terminal test**

Drive the complete follower to its terminal and require that the first in-tolerance terminal observation returns `submit_interact`, moves diagnostics to `ReadyToSubmit`, and emits no preview request or stationary constraint.

- [ ] **Step 2: Run backend and end-to-end tests and verify RED**

Run:

```bash
make build/tests/test_interaction_learned_pickup_backend build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_learned_pickup_end_to_end
```

Expected: failure because current completion enters `FinalPreview`.

- [ ] **Step 3: Carry the certified preview into runtime**

Add `SmartPickupAssistBackend::take_certified_preview()` with a default `nullopt` authored implementation. The learned backend returns the selected frozen `PickEntryPreview` exactly once alongside submission. Add `SmartPickupPostStepResult::certified_preview`.

Add a learned-only runtime cache API:

```cpp
bool InteractionRuntime::cache_certified_pick(
    const PickRequest& request,
    const PickEntryPreview& preview);
```

The controller calls it before pulsing `interact_pressed`. The cache validates request identity, match readiness, pickup-source identity, and finite terminal root. During Preflight, a matching cached candidate bypasses matcher reselection but still performs target generation, object transform, affordance, reservation, terminal root/yaw, correction-limit, and pickup-source validation. Authored requests without a cache follow the existing matcher path unchanged.

- [ ] **Step 4: Submit from completion**

Retain the selected frozen `PickEntryPreview` diagnostics. When the one-tick terminal follower completes, set `ReadyToSubmit`, return `submit_interact = true`, and do not set `stationary_constraint`. Keep runtime authority checks; if the tracked root is outside existing terminal position/yaw tolerances, the follower fails instead of submitting.

- [ ] **Step 5: Verify backend, worker, and Carry integration GREEN**

Run:

```bash
make build/tests/test_interaction_funnel_worker build/tests/test_interaction_learned_pickup_backend build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_funnel_worker
build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_learned_pickup_end_to_end
```

Expected: all exit 0.

- [ ] **Step 6: Commit**

```bash
git add interaction_learned_pickup_backend.cpp interaction_learned_pickup_backend.h interaction_smart_pickup_controller.h interaction_smart_pickup_controller.cpp interaction_runtime.h interaction_runtime.cpp controller.cpp tests/cpp/test_interaction_learned_pickup_backend.cpp tests/cpp/test_interaction_learned_pickup_end_to_end.cpp
git commit -m "fix: transition directly from route into pickup"
```

---

### Task 5: Production Build and Live Verification

**Files:**
- Modify only if compilation requires it: `controller.cpp`
- Verify: existing generated pack, checkpoint, and runtime work directory.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: focused learned-mode visualizer showing the complete blue route from frozen root through pickup seam.

- [ ] **Step 1: Run fresh focused verification**

```bash
git diff --check
make build/tests/test_interaction_funnel_follower \
     build/tests/test_interaction_funnel_worker \
     build/tests/test_interaction_learned_pickup_backend \
     build/tests/test_interaction_smart_pickup_controller \
     build/tests/test_interaction_learned_pickup_end_to_end
build/tests/test_interaction_funnel_follower
build/tests/test_interaction_funnel_worker
build/tests/test_interaction_learned_pickup_backend
build/tests/test_interaction_smart_pickup_controller
build/tests/test_interaction_learned_pickup_end_to_end
```

Expected: all commands exit 0.

- [ ] **Step 2: Rebuild production controller**

Run: `make controller`

Expected: exit 0; third-party Raygui warnings are allowed.

- [ ] **Step 3: Replace the live visualizer**

Stop only the currently managed controller session. Relaunch with the existing learned provider, pack, checkpoint, worker, and runtime work directory environment. Focus the new Raylib window without triggering `F` automatically.

- [ ] **Step 4: Live acceptance**

From 2-to-3 m while walking, press `F`. Confirm the displayed frame remains frozen during worker inference, then the blue route begins at that exact root and the character resumes through locomotion and pickup with no cyan-entry stop or terminal preview wait.
