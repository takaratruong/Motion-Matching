# G1 Low-Latency Operator Preview Design

**Date:** 2026-07-19

**Status:** Recommended bounded experiment

**Base:** `bfe3a674bc8d64be67c5626552fbc144be63d1e7`

## Purpose

The real Motion Matching -> GEAR -> SONIC operator path is visually upright,
grounded, and commandable, but it keeps four 0.4-second chunks ahead of
physics. The resulting approximately 1.6-second command latency makes direct
driving unnecessarily awkward.

This experiment asks one narrow question:

> What is the smallest initial/rolling reference depth, from one through four
> chunks, that preserves an uninterrupted and visually correct 12-second
> SONIC physics run?

The already-qualified Stage A, Stage B, and four-chunk manual evidence paths
remain unchanged.

## Selected approach

Add an explicit `--preload-chunks` option to the non-scored manual demo. Its
default remains four. Use the selected value consistently for initial stand
preload, command recording, the human latency notice, and the run summary.

Run fresh scripted physics trials at depths two and one. Inspect a replay after
each trial before running broader formal checks. Select the smallest depth that
completes without a controller exit, reference underrun, fall, visibly bad
pose, or hand regression. The easy launcher may then request that experimental
depth explicitly; repository defaults and formal evidence still request four.

## Rejected alternatives

- Shortening the 0.4-second MM chunk changes the protocol, reference cadence,
  and Stage B contract. It is too broad for this usability experiment.
- Generating chunks asynchronously could reduce latency further, but introduces
  concurrency and cancellation races before the minimum safe queue is known.
- Changing the formal manual-evidence auditor to accept the experiment would
  turn a usability trial into an unsupported qualification claim.

## Contract

- `--preload-chunks` is an integer in `[1, 4]`.
- Omission is byte-for-byte equivalent in behavior and evidence semantics to
  the existing four-chunk path.
- The selected value controls every preload/latency field; no hard-coded
  `1.6`-second user message remains in the demo.
- The MM chunk duration stays `0.4` seconds.
- The four-chunk manual evidence evaluator remains strict and unchanged.
- Stage A, Stage B, MM search, inertialization, SONIC, joint limits, closed-hand
  targets, and safety gates are not modified.
- Each real trial uses a fresh run root. A failed depth is preserved as evidence
  and is not repaired in place.

## Visual-first acceptance

For depth two, then depth one:

1. Run the real scripted MM -> SONIC path for 30 chunks in physics.
2. Confirm 2,400 CONTROL steps and the expected state/contact coverage.
3. Render or capture the actual MuJoCo trajectory.
4. Visually inspect upright locomotion, ground contact, stopping behavior, and
   closed hands.
5. Compute the existing pelvis-height, pelvis-up, path, stop-drift, contact, and
   hand-target diagnostics.

The selected interactive depth is the smallest trial that passes those quick
checks. If one chunk underruns or looks unstable, two chunks is the result; no
threshold or controller behavior is weakened to force one chunk through.

## Rollback

Rollback is the parent of the implementation commit. The launcher can return
to four chunks immediately without changing repository code or prior evidence.

