# Posture-IK Coverage Regression Fix

## Goal

Recover the open-space grasp coverage that existed immediately before the
bilateral posture-IK integration while retaining the visibly better
posture-aware motions. A valid motion must still satisfy the existing
one-millimetre wrist-position gate, grasp-orientation gate, approach-axis
gate, hard G1 joint limits, fixed root height, and swept object/environment
collision checks.

## Proven regression

Commit `9917c08`, evaluated with the current reach pack and current strict
swept-collision implementation, accepts 94 of 4,608 open-space instances:
47 per hand across five root-azimuth sectors. The current posture-IK
integration accepts six: three per hand across four sectors.

Horizontal placement is not the failure. At a 0.70 m grasp height, targets
at the origin, +1.25 m X, and -1.25 m Z each accept eight current instances.
The current open-space result rejects 4,532 instances for wrist-position
error before collision can be the deciding rejection.

Two changes in `439ac9b` caused the regression:

1. The new weighted posture objective allows source-pose, temporal, and
   elbow-pole residuals to trade against the wrist task, so reachable hands
   can stop outside the one-millimetre gate.
2. The approach warp formerly became complete five frames before contact.
   The integration changed it to reach full weight only at contact, so the
   measured final five-frame approach is not the requested approach.

## Design

### Contact-path timing

Retain the 0.6-second posture-correction window, but divide its final portion
into a blend region and a locked approach region. All translation and
approach alignment must reach full weight five frames before contact and
remain fully applied through contact. This recreates a rigid final
0.2-second approach at 25 Hz and avoids last-frame steering.

The root transform and root height remain those produced by contact
placement. Path shaping may modify only the active hand's ten upper-body
joints: three waist joints and seven arm joints.

### Task-priority posture IK

Continue using the bilateral ten-joint posture solver. Its first solve keeps
the existing source-pose, temporal, and elbow-pole terms because those terms
produce the good-looking motion.

If that solve does not meet the wrist task, run a task-priority polishing
stage from its result. During polishing:

- wrist position and orientation are feasibility constraints;
- hard joint limits remain constraints;
- source pose, temporal continuity, and elbow pole are tie-breakers and
  cannot justify violating the wrist constraints;
- accepted task-feasible iterates outrank lower weighted-objective iterates;
- no root translation, root-height correction, legacy seven-joint fallback,
  or acceptance-threshold relaxation is allowed.

The polishing correction is applied throughout the blend region and the
locked final approach, using the previous solved frame as the temporal seed.
This keeps corrections continuous instead of changing only the contact
frame.

### Search and collision behavior

Candidate enumeration, grasp-relative horizontal placement, bilateral
mirroring, result ranking, and manual Enter-to-search behavior do not
change. Every shaped full clip still passes through the existing swept
object and environment collision checks. Final contact remains the sole
object-contact exemption.

## Failure handling

Numerically invalid, joint-limit-infeasible, position-infeasible,
orientation-infeasible, approach-infeasible, and colliding candidates retain
their existing exclusive rejection categories. A polishing failure returns
the best finite shaped trajectory for diagnosis but cannot be accepted.
Search deadlines and incomplete-result handling remain unchanged.

## Verification

Tests must be written before the implementation and cover:

1. correction weights reach one five frames before contact and stay there;
2. a warped synthetic reach has a final measured approach within the
   existing 15-degree gate;
3. posture regularization cannot make a geometrically reachable synthetic
   target fail the one-millimetre endpoint gate;
4. root position and root height remain unchanged by IK;
5. left and right hands use the same task-priority behavior;
6. swept object and environment collision regressions remain passing;
7. horizontal target translation preserves accepted counts.

The real-pack acceptance gate is the measured pre-regression baseline:

- at least 94 accepted open-space motions;
- at least 47 accepted motions per hand;
- at least five occupied root-azimuth sectors;
- no accepted position error above one millimetre;
- complete processing of all 4,608 instances within the existing
  30-second per-fixture deadline.

The final handoff requires a rebuilt viewer showing the recovered options
at several open-space positions and heights. The viewer must remain the
flat-terrain, bounded-memory viewer already in use.
