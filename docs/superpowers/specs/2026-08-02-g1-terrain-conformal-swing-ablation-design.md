# G1 Terrain-Conformal Swing Ablation

Date: 2026-08-02

## Goal

Test whether a whole-swing, terrain-conformal mid-foot trajectory fixes the
visible stair floating and edge collisions that survived source-path ankle
lifting. This is an offline kinematic ablation. It does not change the live
matcher, Sonic, tracking, contact timing, or clip selection.

## Evidence and Scope

The current emitted-contact matcher can complete a diagonal ascent, but the
rendered 90-degree route stalls before its pivot. The contact-space oracle can
produce a turning sequence, but rigid source playback visibly snaps and places
feet poorly. Prior hard sole sampling, touchdown lifting, symmetric swing
smoothing, local root reconstruction, and relaxed contact projection were
already measured and rejected.

The selected experiment follows the terrain-conformal reference synthesis
structure in Perceptive BFM, restricted to one independently testable unit:
optimize a virtual mid-foot path over each existing source swing while keeping
liftoff, touchdown, and every support bit unchanged.

## Interface

`optimize_terrain_conformal_swing` consumes:

- a finite float32 `(frames, 3)` raw mid-foot path;
- a finite float32 `(frames,)` swing mask containing one contiguous interval;
- a batched terrain-height callback;
- world-frame toe and heel planar offsets;
- an explicit immutable optimization configuration; and
- an explicit random seed.

It returns the optimized path and a cost/quality breakdown. Inputs are never
mutated. The first and last swing samples remain bitwise equal to the raw path.
Samples outside the swing remain bitwise equal. The optimizer is deterministic
for the same inputs and seed.

## Objective

The cost is the sum of independently reported terms:

1. raw-path deviation at interior swing samples;
2. second-difference smoothness;
3. squared mid-foot, toe, and heel clearance deficits;
4. a height-discontinuity edge penalty under toe and heel samples; and
5. an endpoint hard constraint enforced by construction.

The first implementation uses temporally smoothed Gaussian perturbations and a
softmin MPPI update. It may optimize XYZ, but perturbation bounds are smaller in
XY than Z. All samples and iterations are batched on the input tensor's device.

## Failure Handling

Invalid shapes, non-finite values, missing or disjoint swing intervals, bad
terrain callback results, and out-of-domain queries fail closed with
`ContractError`. The optimizer returns the raw path when it cannot strictly
improve total cost; it never returns a worse path.

## Evaluation

Unit fixtures cover flat terrain identity, a stair-riser collision, fixed
endpoints, deterministic replay, callback batching, and invalid terrain data.
The real-data ablation extracts one failing GRAIL swing from the current turn
route, evaluates raw versus optimized toe/heel clearance and smoothness, and
renders both paths over the authenticated stair. No matcher integration is
allowed until the optimizer reduces clearance violations without increasing
endpoint error or foot acceleration beyond the configured acceptance bound.

## Deferred Work

Support-aware root reconstruction, collision repair, twelve-joint ankle/toe/
heel IK, multi-seed continuity guards, and online integration are separate
follow-up units. They depend on this swing-path ablation qualifying first.
