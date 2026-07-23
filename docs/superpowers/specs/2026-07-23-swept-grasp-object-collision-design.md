# Swept Grasp/Object Collision Filter Design

## Goal

Reject motion-matched reaches whose active grasping hand passes through the
queried object, while retaining motions that approach from a clear direction
and touch the object only at the final grasp sample.

## Scope

This changes reach-candidate collision validation only. It does not alter
motion placement, IK, grasp targets, object dimensions, furniture collision,
ranking, rendering, or the underlying motion data.

## Collision Contract

- The oriented object box is the collision shape and follows the object's full
  position and rotation.
- Before the final sample, no active wrist-chain joint or segment may intersect
  the object.
- Active wrist-chain positions are swept continuously between consecutive
  motion samples so a hand cannot tunnel through the object between frames.
- On the terminal sweep, contact is permitted only at the final endpoint.
  Entering the object earlier in that sweep, crossing through it from the
  opposite side, or exiting it at the target is rejected.
- At the final sample, only the existing active wrist contact chain is exempt.
  The active elbow, forearm, inactive hand, torso, and remaining body retain
  their current object-collision checks.
- Object-contact tolerances use a small explicit numerical epsilon. They do not
  create another multi-sample contact window.

## Architecture

`interaction_hand_trajectories.cpp` remains the owner of shaped-trajectory
collision validation. A focused oriented-box segment-interval helper reports
where a swept wrist sphere first enters and leaves the expanded object box.
`evaluate_shaped_trajectory_feasibility` applies that helper to each active
wrist-chain joint trajectory before accepting a candidate.

The current five-sample active-wrist exemption in `reach_coverage.cpp` is
removed. The collision configuration expresses final-contact behavior as one
sample, while the collision evaluator enforces continuous clearance between
samples.

## Rejection and Diagnostics

Any forbidden active-hand intersection returns the existing
`TrajectoryFeasibilityReason::ObjectCollision` and records the first offending
sample. Reach coverage therefore uses the existing `Rejection::ObjectCollision`
path and HUD diagnostics without a new rejection category.

## Tests

Tests will prove:

1. A same-side approach that reaches the final contact without entering the
   object is accepted.
2. A wrist that penetrates before the final sample is rejected.
3. Two clear sampled endpoints whose connecting sweep crosses the object are
   rejected.
4. An opposite-side terminal sweep through the object is rejected even though
   its endpoint is the requested grasp.
5. Rotating the object rotates the swept collision test.
6. Final active-wrist contact remains allowed while elbow, forearm, inactive
   hand, and body penetration remain rejected.

The focused hand-trajectory tests, reach-coverage tests, full viewer contracts,
and real 4,608-candidate probe form the regression gate.
