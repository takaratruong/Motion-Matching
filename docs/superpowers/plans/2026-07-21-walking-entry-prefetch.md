# Walking-Entry Funnel Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and TDD.

**Goal:** Generate while coarse locomotion is moving and arm the learned route at a 0.12-metre rolling handoff.

**Architecture:** Launch against the frozen capture root, poll asynchronously while returning coarse steering, preview without braking, then hold the selected proposal in a new `AwaitEntry` state until the live root reaches the route.

**Tech Stack:** C++17, existing proposal provider and spatial follower.

### Task 1: Prove asynchronous moving prefetch

- Modify `tests/cpp/test_interaction_learned_pickup_backend.cpp` with a 3-metre moving start. Require request entry radius 1 metre, pending/ready/preview outputs to keep steering without `stationary_constraint`, and `AwaitEntry` before the live root reaches 0.12 metres.
- Run the backend test and observe failure because current launch waits for the annulus.

### Task 2: Implement prefetch and rolling handoff

- Add `AwaitEntry` to `LearnedPickupState`.
- Add `coarse_capture_output(observation)`, `launch_prefetch(observation)`, and `arm_if_at_entry(observation)` helpers.
- In `CoarseCapture`, launch once approach speed is nonzero or entry is already reached.
- In `ProposalPending`, call non-blocking `poll()` and return coarse steering.
- Make proposal consumption and selection preview return coarse steering plus their preview requests.
- Arm only within `0.12F` metres and the existing yaw tolerance.
- Run backend, worker, and end-to-end Carry tests and commit.

### Task 3: Rebuild and test live

- Rebuild `controller`, restart learned mode in a persistent PTY, and test from 2-to-3 metres.
