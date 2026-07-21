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

## Live qualification correction

The inference-log barrier was implemented and passed the complete SONIC Python
catalog, but live qualification and independent review disproved its causal
claim. A visible neutral-command run observed action row 1 and still fell at
9.40 simulated seconds. A headless run stayed upright, but its scored initial
boundary hash differed even though the reset qpos and command artifact were
identical.

There are two gaps:

1. `action.csv` row 1 is written by the 50 Hz control thread. The 500 Hz
   `LowCommandWriter` publishes separately, so the row does not prove the
   simulator's DDS subscriber received that command.
2. The current scored prime calls `simulator.advance(1)`. It therefore performs
   one real `mj_step` with whichever LowCmd happened to remain from bootstrap.

The first implementation remains useful inference evidence, but it is not a
physics-release fence.

## Revised decision

Use a two-part, fail-closed scored-start barrier between reset and the first
scored physics release.

First, replace the one-step scored prime with a dedicated simulator operation
that prepares and publishes the reset LowState without applying control,
calling `mj_step`, incrementing the runner step counter, or changing MuJoCo
time. The operation also clears the simulator bridge's previous command receipt
state so bootstrap commands cannot authenticate the scored epoch.

After `activate_control()` reports the authenticated CONTROL transition, GEAR
will remain running while MuJoCo remains paused. The barrier will wait for the
first policy-produced action after the logger's initial row, then obtain a
receiver-side snapshot from the simulator's `rt/lowcmd` subscriber. A snapshot
satisfies the barrier only when its 29 target joint positions fall within the
closed float32 target bounds implied by an authenticated fixed-nine-decimal
policy-action CSV row. The bounds include every source float32 that GEAR could
have rounded to that row, then apply the pinned permutation, action scales,
default angles, and final float32 conversion.
The snapshot is held fixed while later CSV rows arrive, so a fast 500 Hz writer
cannot make the match unobservable. Once a received command is authenticated,
the live `SimulationPolicyGate` enters physics-only pause mode: MuJoCo remains
gated, but GEAR's DDS and watchdog threads stay running. The first
`release_steps()` therefore begins from reset state with a proven policy-derived
LowCmd already present at the physics consumer, without a stale-watchdog race
at `SIGCONT`. Evidence/scoring gates retain full process suspension by default.

The barrier will not enable or retain the elastic band, modify the target
motion, retry a failed start, or advance scored physics. Terrain contacts and
the subsequent controller experiment remain unassisted.

## Interface and ownership

`GearProcess` will own methods that wait for and snapshot complete scored policy
actions.
It already owns the GEAR process, logs directory, launch lifecycle, and timeout
handling, so the caller will not parse child artifacts itself.

The method will:

1. Require an alive, resumed process in active CONTROL state.
2. Observe the registered `action.csv` only inside the owned logs directory.
3. Wait for a complete policy-action row after the initialization row.
4. Validate the row shape and finite action values before returning evidence.
5. Time out with `ProcessError` and leave physics unreleased.

The gated simulator will add three exact protocol operations:

- `prime_low_state`: reset bridge receipt flags and publish the current reset
  observation without changing physics or time;
- `low_command`: return whether a body command was received and, if so, an
  immutable copy of its 29 finite target joint positions;
- `refresh_low_state`: republish the current observation without stepping or
  clearing the receiver, immediately before a paused GEAR group resumes.

The manual driver will reconstruct pinned LowCmd targets from authenticated
action rows and wait until a held receiver snapshot matches one. It will report
both the inference index and the received-command index as startup evidence.

`manual_demo.run_demo()` will call the barrier immediately after
`gear.activate_control()` and before constructing and pausing the
`SimulationPolicyGate`. Each later policy-gate release refreshes LowState before
physics advances. In the live driver GEAR remains running while only physics is
paused; the measured MM/publication interval must therefore remain below GEAR's
pinned 500 ms LowState watchdog threshold. Longer external inference requires
action chunking or an explicit no-step LowState maintenance service.
After the final physics release, the driver explicitly quiesces GEAR before
building evidence so post-run serialization cannot trip that watchdog.

## Failure behavior

Missing, partial, malformed, non-finite, late, or unmatched action/LowCmd data
is a startup failure. The demo must not release physics in those cases.
Existing cleanup will stop GEAR, the simulator, and the MM server while
preserving their run artifacts for diagnosis.

Every wait has a finite timeout, honors cooperative operator cancellation, and
verifies both child processes while polling. A dead child is reported as
child-process failure rather than as a generic timeout.

## Testing

Tests will be written before production code and will prove:

- a complete post-initialization action row is necessary but not sufficient;
- the initialization row alone does not satisfy it;
- partial, malformed, and non-finite rows fail closed;
- timeout and child death fail without reporting readiness;
- scored LowState priming changes neither qpos, MuJoCo time, nor step count;
- a stale, absent, malformed, or policy-unmatched LowCmd cannot satisfy startup;
- a receiver snapshot matching an authenticated policy row satisfies startup;
- fixed-nine CSV rounding cannot reject a genuine received float32 target;
- every release refreshes LowState before physics, and the live gate emits no
  process stop/continue signals;
- the production driver wires one cooperative-cancellation callback into both
  child-process lifecycles;
- the manual demo orders reset, no-step prime, resume, activate, inference
  evidence, receiver match, gate pause, and only then physics release;
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
under variable host load. Waiting for one or two anonymous subscriber callbacks
would not exclude a delayed pre-policy write. Matching a held received target
against authenticated policy output observes the actual condition required for
safe release without modifying official GEAR.
