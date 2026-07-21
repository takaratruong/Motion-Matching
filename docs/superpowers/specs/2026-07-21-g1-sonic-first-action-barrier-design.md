# G1 SONIC First-Action Barrier Design

## Problem

The live terrain-aware SONIC demo can fall while the operator command remains
zero. The failed run froze at pelvis height 0.197 m after 6.26 s. Its motion
matching commands and selected database frames through the fall were identical
to a prior run that remained upright for 48 s.

The scored epoch currently queues one fresh LowState while the GEAR process
group is stopped, resumes GEAR, activates control, and immediately stops GEAR
again. The first physics release therefore races the first policy inference.
Recorded runs received their first nonzero policy action between 2.7 ms and
19.7 ms after release. This changes both the initial closed-loop response and
the reference row presented to the controller.

## Decision

Add a fail-closed first-action barrier between GEAR control activation and the
first scored physics release.

After `activate_control()` reports the authenticated CONTROL transition, GEAR
will remain running while MuJoCo remains paused. The barrier will wait for the
first policy-produced action after the logger's initial row. GEAR can consume
the priming LowState and publish its action without advancing physics. Once the
action is observed, the existing `SimulationPolicyGate.pause()` stops the full
GEAR process group. The first call to `release_steps()` therefore starts with a
policy action already available to the simulator.

The barrier will not enable or retain the elastic band, modify the target
motion, retry a failed start, or advance scored physics. Terrain contacts and
the subsequent controller experiment remain unassisted.

## Interface and ownership

`GearProcess` will own a method that waits for the first scored policy action.
It already owns the GEAR process, logs directory, launch lifecycle, and timeout
handling, so the caller will not parse child artifacts itself.

The method will:

1. Require an alive, resumed process in active CONTROL state.
2. Observe the registered `action.csv` only inside the owned logs directory.
3. Wait for a complete policy-action row after the initialization row.
4. Validate the row shape and finite action values before returning evidence.
5. Time out with `ProcessError` and leave physics unreleased.

The returned immutable evidence will identify the observed action index and
timing. It is diagnostic only; the action remains delivered through the
existing GEAR-to-simulator transport.

`manual_demo.run_demo()` will call the barrier immediately after
`gear.activate_control()` and before constructing and pausing the
`SimulationPolicyGate`. Other runners will not change in this patch.

## Failure behavior

Missing, partial, malformed, non-finite, or late action data is a startup
failure. The demo must not release physics in those cases. Existing cleanup
will stop GEAR, the simulator, and the MM server while preserving their run
artifacts for diagnosis.

The wait has a finite timeout and verifies process liveness while polling. A
dead GEAR process is reported as child-process failure rather than as a generic
timeout.

## Testing

Tests will be written before production code and will prove:

- a complete post-initialization action row satisfies the barrier;
- the initialization row alone does not satisfy it;
- partial, malformed, and non-finite rows fail closed;
- timeout and child death fail without reporting readiness;
- the manual demo orders reset/prime, resume, activate, action barrier, gate
  pause, and only then the first physics release;
- no elastic-band or target-motion behavior changes.

After unit and integration suites pass, qualification will include repeated
short no-input live trials. Each trial must report a prepared action before
scored time advances, must show no reset/freeze marker during the observation
window, and must keep pelvis height and uprightness above the existing fall
thresholds. An interactive terrain run follows only after those repeated trials
pass.

## Alternatives rejected

An elastic-band warm-up could hide the fall but would weaken the physical
experiment. Retrying launches would conceal nondeterminism rather than remove
it. Polling or sleeping for a fixed duration would still race policy inference
under variable host load. The action barrier observes the actual condition
required for safe release.
