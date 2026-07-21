# Moving-Entry Blocking Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture moving funnel entries immediately and hold the controller update while the existing proposal subprocess completes.

**Architecture:** Add a blocking `wait()` seam to the proposal provider while preserving non-blocking `poll()`. Learned pickup freezes the first valid moving annulus condition, calls `wait()` in the launch tick, consumes the artifact through one shared handler, and continues through the existing preview/follower path.

**Tech Stack:** C++17, POSIX `waitpid`, existing Python/PyTorch proposal worker, GNU Make.

## Global Constraints

- Preserve the 0.45-to-1.00-metre handoff annulus and 3.0-metre activation radius.
- Do not require an entry speed below 0.1 m/s or multiple settling ticks.
- Preserve exact live entry position, yaw, and simulation velocity.
- Keep production inference at 50 DDIM steps.
- Keep `poll()` non-blocking for diagnostics/tests and make `wait()` blocking only for the production provider.
- Preserve existing matcher, contact, attachment, and placement authority.

---

### Task 1: Add blocking provider completion

**Files:**

- Modify: `interaction_funnel_worker.h`
- Modify: `interaction_funnel_worker.cpp`
- Modify: `tests/cpp/test_interaction_funnel_worker.cpp`

**Interfaces:**

- Produces: `virtual FunnelProposalPoll FunnelProposalProvider::wait()` and `AsyncPythonFunnelProvider::wait()`.

- [ ] Write a test that starts `/bin/false`, calls `wait()` once, and requires a failed result:

```cpp
AsyncPythonFunnelProvider provider(
    "/bin/false", "ignored.py", "ignored.pt", directory);
assert(provider.begin(request()));
const FunnelProposalPoll result = provider.wait();
assert(result.state == FunnelProposalPollState::Failed);
assert(!result.artifact.has_value());
assert(!result.error.empty());
```
- [ ] Run `make build/tests/test_interaction_funnel_worker`; expect compilation to fail because `wait()` is absent.
- [ ] Add a default base wait and production override:

```cpp
virtual FunnelProposalPoll wait() { return poll(); }

FunnelProposalPoll AsyncPythonFunnelProvider::wait() {
    if (child_pid_ < 0 || !request_.has_value()) return no_active_request();
    int status = 0;
    pid_t result = -1;
    do {
        result = waitpid(static_cast<pid_t>(child_pid_), &status, 0);
    } while (result < 0 && errno == EINTR);
    if (result < 0) return fail("failed to wait for funnel proposal worker");
    return finish(status);
}
```

Implement `no_active_request()` and `finish(int)` as private helpers used by
both `poll()` and `wait()`.
- [ ] Share child-exit/artifact validation between `poll()` and `wait()`; preserve `poll()` with `WNOHANG`.
- [ ] Run the worker test and require exit zero.
- [ ] Commit as `feat: add blocking funnel proposal wait`.

### Task 2: Capture moving entry and wait in launch tick

**Files:**

- Modify: `interaction_learned_pickup_backend.h`
- Modify: `interaction_learned_pickup_backend.cpp`
- Modify: `interaction_learned_pickup_diagnostics.h`
- Modify: `tests/cpp/test_interaction_learned_pickup_backend.cpp`

**Interfaces:**

- Consumes: `FunnelProposalProvider::wait()`.
- Produces: first-valid-annulus moving condition and immediate transition to proposal preview after blocking completion.

- [ ] Change the handoff regression to provide a moving observation and require immediate launch:

```cpp
PickAssistObservation moving = lifecycle_observation(10U, target, crossed);
moving.simulation_velocity = vec3(0.2F, 0.0F, -0.3F);
moving.displayed_planar_speed_mps = 0.4F;
const PickAssistOutput selection = backend.observe(moving);
assert(provider.begin_calls == 1);
assert(selection.preview_requests.size() == 32U);
assert(provider.request.condition[22] == 0.2F);
assert(provider.request.condition[23] == -0.3F);
```
- [ ] Add a provider whose `poll()` remains pending but whose `wait()` returns ready; require backend launch to return 32 preview requests in the same observation.
- [ ] Run the backend test; expect launch/preview assertions to fail under settle/poll behavior.
- [ ] Remove capture speed and tick requirements. Define:

```cpp
PickAssistOutput consume_proposal_poll(
    const PickAssistObservation& observation,
    FunnelProposalPoll poll);
```

After `provider_->begin(proposal_request_)`, set `ProposalPending` and call:

```cpp
return consume_proposal_poll(observation, provider_->wait());
```

The `ProposalPending` switch case uses the same helper for providers whose
default `wait()` returns `Pending`.
- [ ] Run backend, worker, and end-to-end Carry tests; require exit zero.
- [ ] Commit as `fix: hold physics during funnel generation`.

### Task 3: Rebuild and restore live learned mode

**Files:**

- Generate only under: `build/g1-funnels/evidence/` and `build/g1-funnels/runtime-object-anchor-live/`.

- [ ] Rebuild `controller` and run focused C++ tests plus the trained-object Python sweep tests.
- [ ] Restart only the learned controller with the existing pack/checkpoint environment in a persistent PTY.
- [ ] Confirm the process and window remain alive; hand off a live test from 2-to-3 metres.
