# Sonic Acknowledged LowState Startup Design

## Goal

Make the visible terrain-aware Sonic demo start and restart reliably after an
arbitrarily long operator-focus wait. Physics must remain at the registered
terrain start while the window is being focused, and GEAR must receive and
acknowledge a fresh reset-state LowState before CONTROL can begin.

This change removes the LowState startup race. It does not tune locomotion,
terrain interaction, Motion Matching, depth input, or fall recovery.

## Failure evidence

The current startup primes one reset LowState and then stops the complete GEAR
process group while the operator focuses the X11 viewer. That wait can exceed
GEAR's 500 ms LowState watchdog. When the group resumes, CONTROL activation can
race the next LowState delivery and fail with:

```
LowState is not available, waiting for robot to be ready
[ERROR] Lost LowState data connection from robot!
[ERROR] Safety check failed, cannot start control.
```

Two narrower fixes were tested and rejected:

1. Publishing one no-step LowState before resuming the stopped process did not
   help because GEAR's DDS subscriber was still stopped.
2. Resuming GEAR, publishing one no-step LowState, and immediately activating
   CONTROL still raced asynchronous DDS delivery.

Both variants passed their focused automated tests. Both failed the live
canary. Therefore an unacknowledged publish, sleep, or retry is not acceptable
evidence that GEAR has observed the reset state.

## Decision

Extend the simulation-only GEAR control protocol with an acknowledged
`RESUME` transition. Startup will use the existing `PAUSE` and `SYNC`
operations to fence policy workers and prove that GEAR's DDS callback observed
a post-focus no-step LowState. It will then explicitly resume the control gate
before entering CONTROL.

The simulator remains paused throughout this exchange. The only simulator
operation is `refresh_low_state`, which republishes the current observation
without calling `mj_step`, changing MuJoCo time, incrementing the runner step
counter, or changing scored state.

The protocol capability changes from `READY 4` to `READY 5`. Motion Matching
will reject an older GEAR binary rather than silently running without the new
guarantee.

## GEAR protocol contract

### Request and response

The new request is exactly:

```
RESUME <epoch>\n
```

It takes no stream-frame argument. A successful response is exactly:

```
RESUMED <epoch> <current-lowstate-tick>\n
```

`RESUME` is valid only for the current epoch after `SYNCED`: the gate must be
paused, `epoch_synchronized_` must be true, and a LowState tick must exist.

On a valid request, the gate enters a dedicated resuming state. That state
blocks new input, control, and planner worker leases, drains any input lease
that entered after `SYNCED`, clears the synchronized flag, changes to running,
and only then emits `RESUMED`. Thus the acknowledgement establishes one exact
boundary: all pre-resume policy/reference work is finished, the fresh LowState
has already been observed, and later policy work is allowed to start.

Malformed `RESUME` packets with a third argument fail closed with
`malformed-RESUME`. A wrong epoch, wrong state, missing LowState, or request
before `SYNCED` fails closed with `expected-SYNC`. Duplicate `RESUME` is also an
`expected-SYNC` failure because the first request has already consumed the
synchronized paused state. Existing malformed-packet, socket-failure, and
fatal-gate behavior remains unchanged.

The existing `ARM` transition is not reused. `ARM` waits for a committed stream
frame and a changed LowState tick before returning to running, which requires a
physics advance. Startup instead needs to return to running while physics and
the reset tick remain unchanged.

## Motion Matching integration

`GearProcess` will require `READY 5` and report `READY/v5` on a capability
mismatch. Its pause API will support both existing active-CONTROL pauses and the
prepared WAIT_FOR_CONTROL startup state, while continuing to require an alive,
resumed process group and an enabled simulation-control channel.

A new `resume_simulation_control()` method will:

1. require the local state to be paused and synchronized;
2. send `RESUME` for the current epoch;
3. wait for the exact `RESUMED` epoch and a valid uint32 tick;
4. clear the synchronized flag and record the local state as running; and
5. set the local gate state to unknown and propagate the error after any send,
   protocol, timeout, cancellation, or child-process failure.

For the X11 startup path, `_activate_scored_control` will perform this exact
order:

1. wait for the viewer focus barrier and print `LIVE X11`;
2. resume the OS process group;
3. request and await `PAUSED`, draining GEAR workers;
4. request and await `SYNCING`;
5. call the simulator's no-step `refresh_low_state`;
6. await `SYNCED`, proving GEAR observed that refresh;
7. request and await `RESUMED`;
8. activate CONTROL;
9. await and report the first policy action;
10. await and report the matching LowCmd received by the simulator; and
11. construct the existing physics-only `SimulationPolicyGate`.

The focus wait remains before OS-process resumption, so no LowState maintenance
is needed while the operator finds the window. Once `RESUMED` is acknowledged,
CONTROL activation follows immediately with no polling, sleep, retry, or other
operator barrier between them.

The script-mode path, which has no external focus delay, retains its current
startup order. Later per-chunk `PAUSE`/`SYNC`/`ARM` behavior also remains
unchanged.

## Failure and cleanup behavior

Any failure in focus acquisition, OS-process resumption, `PAUSE`, `SYNC`,
LowState refresh, `RESUME`, or CONTROL activation aborts the episode. Physics
must not advance and startup must not print either readiness marker unless its
underlying evidence exists. Startup failures are fatal, not operator restarts,
and are never converted into automatic retries.

The existing `run_demo` teardown remains responsible for closing the GEAR
process group, simulator/viewer, Motion Matching client, publisher, and X11
input loop. The episode supervisor must not launch a replacement episode after
a teardown failure, so two simulator or GEAR instances cannot overlap.

## Version and repository isolation

The GEAR change will be developed in a new isolated branch/worktree based on
the clean pinned commit
`4a63412b034f1fef3e8e087adbb9be6cbeaaebdc`. The existing
`research/simulation-control-gate` worktree will not be modified.

After GEAR's tests and build pass, Motion Matching will pin the resulting exact
GEAR commit in all three identity locations:

- `sonic/configs/gear_sonic.lock.json`;
- `sonic/python/mm_sonic/gated_sim.py`; and
- `sonic/python/mm_sonic/metrics.py`.

The live demo will use an explicitly selected binary built from that exact
GEAR worktree and the exact Motion Matching branch binary. No stale default
build symlink is acceptable qualification evidence.

Reliable Claude may implement bounded GEAR and Motion Matching tasks in
separate supervised jobs after the implementation plan exists. Codex owns the
contract, protected tests, diff review, exact-identity checks, and live canary.
An automated Reliable Claude `passed` status is candidate evidence only and
cannot replace the live acceptance gate.

## Test design

Tests are written before production changes.

GEAR C++ tests will prove:

- the capability marker is exactly `READY 5`;
- `PAUSE -> SYNCING -> ObserveLowState -> SYNCED -> RESUME -> RESUMED`
  preserves the epoch and current tick;
- no policy worker enters before `RESUMED`, and workers can enter afterward;
- a post-`SYNCED` input lease is drained before acknowledgement;
- a third argument produces `malformed-RESUME`; and
- a stale epoch, pre-sync request, wrong state, or duplicate request produces
  `expected-SYNC` and leaves the gate fatal.

Motion Matching tests will prove:

- `GearProcess` accepts only `READY 5`;
- startup pause is legal only in an authenticated, input-prepared
  WAIT_FOR_CONTROL state with the process group resumed;
- `resume_simulation_control()` enforces paused-and-synchronized local state,
  validates the exact response, and fails closed to unknown on errors;
- the X11 path performs focus, continue, pause, begin sync, no-step refresh,
  finish sync, resume, CONTROL, action evidence, and received-command evidence
  in exact order;
- no CONTROL activation occurs after any injected boundary failure;
- the script path and later `ARM` path retain their existing behavior;
- the no-step refresh changes neither qpos, MuJoCo time, step count, nor scored
  evidence; and
- pinned GEAR identities agree across the lock, runtime, metrics, and manifest
  tests.

The complete affected Python and C++ suites and protected validators must pass
before a live run. Review must also confirm that no sleep, retry, watchdog
weakening, elastic-band change, or physics advance entered the startup path.

## Live acceptance gate

Automated tests are necessary but not sufficient. The exact pinned binaries
must pass one visible terrain canary containing three consecutive episodes:

1. Episode 1 reaches both `SONIC first action ready` and
   `SONIC policy command received`, accepts locomotion input, and remains a
   healthy process until Backspace.
2. Backspace reports `RESTART BACKSPACE: episode 1 -> 2`; episode 2 opens at the
   registered terrain start and reaches both readiness markers before input.
3. A second Backspace reports `RESTART BACKSPACE: episode 2 -> 3`; episode 3
   opens at the same registered start and reaches both readiness markers.
4. `X` exits episode 3 cleanly.

Across all three episodes there must be no `Lost LowState`, safety-check
failure, `ChildProcessDied`, unintended MuJoCo step before the first received
policy command, overlapping episode processes, or leaked Motion Matching,
GEAR, simulator, and viewer processes after exit. Each episode must have a
distinct run/session directory, and the two interrupted episodes must preserve
their restart outcome artifacts.

## Non-goals

- Changing WASD, turn-rate, transition, or movement-model behavior.
- Improving Sonic tracking quality or terrain step clearance.
- Adding depth, lidar, learned kinematics, action chunking, or latency studies.
- Automatically restarting after a crash or fall.
- Extending the production robot protocol; `RESUME` remains simulation-only.
- Refactoring unrelated GEAR or Motion Matching lifecycle code.
