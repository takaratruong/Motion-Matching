# G1 SONIC Responsive Motion-Matching Control Design

**Status:** Approved

## Purpose

Make the interactive terrain-aware Motion Matching to SONIC driver visibly
respond to forward, backward, lateral, heading, and stop input in less than
200 milliseconds when Motion Matching generation completes within that budget.
Physics remains paused whenever the next safe target prefix is unavailable.
SONIC, its observations, and its policy weights remain unmodified.

This work qualifies the privileged terrain-aware baseline. Depth-conditioned
Motion Matching remains a later experiment and must not be mixed into this
latency diagnosis.

## Current Evidence and Root Cause

The control sampler publishes key state every 20 milliseconds, but the rolling
driver consumes it only at a 0.4-second Motion Matching boundary. The driver
also keeps two complete 0.4-second chunks ahead of physics. A new command is
therefore presented approximately 0.8 to 1.2 seconds after the key transition.
That delay is intentional queueing, not X11 input latency.

The pure Holden mapper has unit coverage for MuJoCo forward and lateral axes,
but those tests do not prove that the Motion Matching result moves backward or
laterally, nor that SONIC follows it. Direction must be measured independently
at each boundary before changing buffering.

The previously tested one-chunk configuration reduced lookahead to 0.4 seconds
but failed the existing stopping-displacement threshold. It is a diagnostic
comparison, not the final responsive-control design.

## Scope

### Goals

- Measure key-transition-to-sampled-command, command-to-MM-result,
  MM-result-to-published-target, and published-target-to-visible-physics times.
- Prove that backward and lateral operator commands produce matching root
  displacement in the generated Motion Matching kinematics.
- Prove separately whether unmodified SONIC follows those generated targets.
- Present a changed command to physics within 200 milliseconds when generation
  finishes within the budget.
- Pause physics, instead of replaying stale motion, whenever a safe replacement
  target is not ready.
- Preserve authenticated terrain scene identity, reset state, evidence hashes,
  and deterministic replay artifacts.

### Non-goals

- Training or changing SONIC.
- Replacing privileged terrain input with depth.
- Feeding the MuJoCo root back into Motion Matching.
- Obstacle avoidance or manipulation.
- Claiming real-time performance while physics is paused.

## Approaches Considered

### Fixed one-chunk lookahead

Keep the existing transport and reduce preload from two chunks to one. This is
the smallest experiment and lowers presentation delay to roughly 0.4 to 0.8
seconds, but it cannot meet the 200-millisecond target and has already failed a
stop-quality gate. It remains a useful diagnostic baseline.

### Flush and restart the complete stream on every command edge

Pause physics, discard queued future targets, and restart the SONIC reference
stream from the current boundary. This can react quickly, but repeatedly
resetting the policy/reference process risks discontinuities and destroys
useful recurrent context. It is retained only as a fallback experiment if the
stream cannot safely accept short prefixes.

### Receding-horizon generation with a short committed prefix

Generate a normal Motion Matching horizon, but expose only the shortest prefix
that the existing target transport and SONIC consumer can safely accept. Sample
operator intent and replan at each prefix boundary. This preserves a useful MM
horizon while reducing irrevocably queued motion. Physics pauses during a miss
instead of consuming the superseded plan.

This is the recommended design. The initial prefix target is 100 milliseconds,
subject to an executable transport-compatibility probe. If the existing SONIC
stream requires a longer atomic unit, use the shortest proven unit and report
the resulting lower bound rather than silently claiming 200 milliseconds.

## Architecture

### Boundary trace

Add one controller-owned trace record for every input transition and committed
prefix. Each record contains monotonic timestamps and stable identifiers for:

1. X11 key transition observed;
2. mapped command snapshot sampled;
3. MM request started and completed;
4. candidate committed;
5. target prefix published and acknowledged;
6. physics first released for that prefix; and
7. first simulated frame belonging to that prefix.

The trace also records requested velocity, requested heading, generated virtual
root displacement, published physical-root displacement, and observed MuJoCo
root displacement. This makes command, generation, transport, and tracking
failures distinguishable.

### Direction probe

Before changing runtime buffering, run deterministic stand-to-command probes
for forward, backward, left, and right on the same reset state. A probe passes
the MM layer only when the generated virtual root displacement has the expected
signed projection and exceeds a registered minimum magnitude. SONIC tracking
is evaluated separately against the published physical-root trajectory.

A failed MM direction probe blocks buffer tuning for that direction. A passed
MM probe followed by a failed physics probe identifies SONIC tracking or target
conversion as the failing boundary.

### Responsive prefix scheduler

The scheduler owns exactly one committed, not-yet-consumed prefix. It samples
the latest mailbox state at a prefix boundary, pauses physics, generates and
validates the next horizon, publishes only the safe prefix, acknowledges it,
and then releases exactly that prefix's physics steps.

Input changes during generation are latched. If the generated horizon no longer
matches the latest command when validation finishes, it is aborted before
publication and regenerated from the same accepted predecessor. Published
frames are never overwritten. This keeps the existing transactional
MM candidate protocol and append-only evidence truthful.

The initial implementation supports one in-flight MM request. Speculative
parallel generation is excluded until the serial scheduler is correct.

### Physics gating

The physics gate stays closed throughout MM generation, validation, and target
publication. It opens only after the consumer acknowledges the complete safe
prefix. A timeout or worker failure leaves physics paused and produces a
terminal diagnostic; it never falls back to stale commands.

## Latency Contract

The primary metric is simulated presentation latency:

`first released frame for new command - key transition timestamp`

For an on-time MM result, the p95 presentation latency across registered
direction transitions must be at most 200 milliseconds. Paused wall-clock time
is reported separately as generation stall and does not count as simulated
real-time success.

The run must publish all of the following:

- input-to-sample latency;
- MM generation duration;
- validation and publication duration;
- queued irrevocable lookahead at the transition;
- presentation latency;
- physics pause duration; and
- direction and tracking errors.

If transport constraints make a sub-200-millisecond prefix impossible, the
probe must fail closed with the measured minimum atomic duration.

## Acceptance Tests

### Pure and protocol tests

- Backward, left, and right commands retain the expected signs through the
  MuJoCo-to-Holden request conversion.
- A stale MM result caused by a newer latched command is aborted and never
  published.
- The scheduler never has more than one unconsumed committed prefix.
- Physics cannot advance before target acknowledgement.
- Generation failure and timeout leave physics paused.
- Trace timestamps are monotonic and bind one input transition to one presented
  prefix.

### MM-only direction gate

From an identical authenticated reset, forward, backward, left, and right each
produce an expected signed generated-root displacement. Neutral input after
each direction reduces generated travel rather than continuing the prior
command indefinitely.

### Flat dynamic gate

Unmodified SONIC remains upright during direction and stop transitions. The run
reports command-to-target and target-to-root tracking separately. The p95
presentation latency is at most 200 milliseconds for on-time generations.

### Terrain dynamic gate

Repeat the responsive forward and stop sequence on
`grail-curb-default` / `curb-forward` with terrain weight `4.0`. SONIC must stay
upright and advance along the registered route. This is the immediate terrain
integration proof; the broader terrain-aware versus terrain-blind Stage C
matrix follows after it passes.

## Evidence and Rollback

Every run stores the command artifact, boundary trace, MM candidate identities,
scene and reset hashes, generated and published root trajectories, simulator
snapshot, and exact configuration. The existing two-chunk launcher remains a
rollback and comparison path. The responsive scheduler is selected explicitly
until all flat and terrain gates pass; it does not silently change formal
Stage B evidence.
