# Bilateral Posture-Aware Reach IK Design

## Goal

Replace the current seven-joint arm-only IK used by the exhaustive reach
baseline with the posture-aware damped least-squares solver proven in the G1
IK comparison spike, generalized consistently to left and right reaches.

## Scope

The first integration target is the current contact-anchored exhaustive
motion-matching path in `reach_coverage.cpp`, which drives the coverage probe
and mesh viewer. This does not replace unrelated runtime, placement, or
locomotion IK paths.

## Solver Contract

- Optimize ten G1 hinge joints together: three waist joints and the seven
  joints of the requested arm.
- Support `interaction::Hand::Left` and `interaction::Hand::Right` directly
  from their corresponding metadata. Do not mirror poses through left-hand
  space.
- Minimize wrist position and orientation errors while penalizing transported
  source elbow-pole error, deviation from the recorded source pose, and
  deviation from the previous solved frame.
- Preserve adaptive Levenberg-Marquardt-style damping, a numerical Jacobian,
  `0.10 rad` per-joint step limits, hard G1 joint limits, and at most 30
  iterations.
- Preserve all local pose channels not owned by the three waist and selected
  seven arm joints. In particular, root translation, legs, inactive arm, and
  non-hinge channels remain byte-identical to the placed source pose.
- A zero correction returns the placed source pose byte-exactly.
- Return the best finite bounded pose seen even when the requested target is
  not accepted, together with truthful position, orientation, elbow-pole,
  joint-limit, iteration, and objective diagnostics.

## Trajectory Integration

For each placed reach clip, derive FPS from the reach pack. The posture
correction begins exactly `0.6 s` before the final sample and uses smoothstep
weighting to reach one at the final sample. Earlier samples remain equal to
their placed recorded poses.

At each corrected sample:

1. Build the same position/orientation target currently used to converge on
   the requested grasp.
2. Use the placed pose at that sample as the source/reference pose.
3. Seed the first corrected sample from its source joint angles.
4. For later samples, transport the previous solution by adding its
   source-relative angle delta to the current source angles.
5. Solve the selected hand with the bilateral posture solver.

There is no fallback to the old arm-only solver for accepted options. Existing
final `1 mm` position, approach, orientation, joint-limit, and finite-value
diagnostics remain authoritative.

## Collision and Search Interaction

The posture-aware poses are the sole pose sequence passed to the object and
environment collision evaluator. The swept active-hand/object filter runs
after IK and can reject a numerically successful grasp whose hand trajectory
passes through the object. Candidate identity, exhaustive yaw placement,
ranking inputs, and explicit Enter-triggered search remain unchanged.

## Performance

Only the final `0.6 s` correction window invokes posture IK; earlier samples
are copied from the placed motion. Exhaustive searches must still process all
4,608 candidate identities within the existing 30-second fixture deadline.
No acceptance or collision thresholds may be weakened to meet that deadline.

## Tests and Evidence

Tests will prove:

1. Left and right ten-joint metadata, angle decomposition, application, and
   hard limits are side-correct.
2. Zero-offset targets preserve source poses byte-exactly for both hands.
3. Reachable targets converge for both hands without modifying unowned pose
   channels.
4. Small target sweeps preserve each side's source elbow bend rather than
   flipping it.
5. Temporal seeding reduces discontinuity and the correction starts no earlier
   than the final `0.6 s`.
6. Final reach error and rejection diagnostics remain truthful.
7. The posture-shaped trajectory is the trajectory evaluated by the swept
   object and furniture collision filters.
8. Real-pack coverage processes all 4,608 candidates per fixture under the
   deadline and reports any reduction in accepted coverage without fallback.

The final viewer uses the same selected posture-aware `WorldPose` for the G1
mesh and optional skeleton.
