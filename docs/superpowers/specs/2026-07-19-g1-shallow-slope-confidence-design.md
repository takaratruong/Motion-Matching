# G1 Shallow-Slope Confidence Design

## Problem

The terrain classifier labels the certified 5-degree ramp as `slope`, but its
confidence is capped at approximately `0.50` because slope confidence is
normalized against 10 degrees.  The motion-bank state requires confidence of
at least `0.60`, so it retains the flat bank for the entire ramp.  A bounded
runtime trace confirmed 425 requested-slope frames, a maximum confidence of
`0.50000300514888363`, and zero active or selected slope-source frames.

The quality fixture must not fabricate a higher confidence to hide this runtime
behavior.

## Decision

Normalize slope confidence against 5 degrees in both the Python reference
classifier and the C++ runtime classifier.  Keep the shared `0.60` transition
threshold, two-frame hysteresis, flat-family limits, and curb/stair confidence
rules unchanged.

This makes the shallowest certified longitudinal and cross-slope scenes strong
slope observations while preserving the existing noise rejection for all
families.  Since the flat classifier still owns profiles at or below 2 degrees,
the effective slope activation boundary becomes approximately 3 degrees.

Rejected alternatives:

- Lowering the global transition threshold would also admit weaker curb and
  stair observations.
- A slope-specific transition threshold would add state-machine branching and
  retain a fragile boundary around the observed 5-degree confidence.

## Implementation

- Change the slope-confidence reference constant from 10 to 5 degrees in
  `resources/g1_terrain_builder/terrain.py` and `motion_bank_runtime.h`.
- Update independent Python and C++ classifier parity expectations.
- Add boundary coverage proving 2 degrees remains flat, a 3-degree slope meets
  the existing transition threshold, and a 5-degree slope becomes strongly
  confident.
- Remove the synthetic ramp-05 confidence override from the motion-quality
  fixture.  Its expected state must come from the real classifier result.

No database schema, artifact layout, scene geometry, heading behavior, IK, or
motion-bank transition state machine changes.

## Verification

1. Run focused Python terrain-classifier and C++ motion-bank tests.
2. Run the motion-quality route and full evaluation suites without a fixture
   confidence override.
3. Build the release controller and run a bounded 800-frame `ramp-05-up-down`
   trace with IK disabled and terrain weight 4.
4. Require nonzero active-slope and selected-slope-source frames, exact log
   validation, and the physical Gate C route checks.
5. Run the adjacent controller/runtime suites before checkpointing.

## Rollback

The change is isolated to two mirrored confidence constants and their tests.
Reverting those constants restores the previous calibration without changing
artifacts or persisted data.
