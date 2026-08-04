# G1 MotionBricks Terrain Task Actor Design

## Goal

Produce one fixed, same-heading horizontal traversal of the existing staircase:
approach from the side, step onto it, walk across unequal support heights, step
down, and depart. The result is a kinematic MotionBricks capability test, not a
Sonic tracking experiment.

## Design

Reproduce the task-actor architecture described in the MotionBricks paper.
Sparse proxy poses define the intended scene interaction, while the released
MotionBricks backbone generates the transitions.

The route has five ordered phases:

1. flat approach;
2. supported step-up;
3. alternating unequal-height walking contacts;
4. supported step-down;
5. flat departure.

Authenticated GRAIL stair, curb, or narrow-support motions may supply sparse
four-frame proxy poses. They are not replayed as route segments. Each proxy is
rigidly placed at a selected staircase contact while preserving the original
same-heading travel direction. The initial implementation is offline and
scripted for this single staircase.

For every adjacent pair of proxy poses, the current four MotionBricks frames
and the next four proxy frames are converted to MotionBricks' flexible
in-betweening constraints. MotionBricks predicts the timing, root trajectory,
pose tokens, and decoded motion. Generated segments overlap only through their
four context frames; accepted output is concatenated without linear pose
blending.

## Candidate Search

Generate multiple candidates by varying MotionBricks sampling seeds and
allowed transition lengths. A candidate is rejected when any of these is true:

- a stance sole does not have valid support on its planned surface;
- a swing sole penetrates the staircase or lacks clearance;
- the pelvis, knees, torso, hands, or non-support foot penetrate the scene;
- the generated motion misses its proxy endpoint beyond the configured
  positional or rotational tolerance;
- heading reverses or deviates from the fixed horizontal travel direction;
- joint or root discontinuity exceeds the existing G1 traversal limits.

Rank surviving candidates by endpoint error, stance sliding, minimum
clearance, acceleration, and transition discontinuity. Search speed is not a
success criterion; validity and visual quality are.

## Boundaries

- MotionBricks generates every transition.
- Do not use ARDY, motion matching, route warping, or interpolated pose
  stitching.
- Do not add Sonic or dynamic tracking.
- Do not alter the pinned MotionBricks checkout.
- Reuse existing authoritative staircase sampling, G1 FK, sole geometry,
  contact detection, and traversal validation.
- Keep the native interactive MotionBricks probe available as the flat-motion
  baseline.

## Outputs

Write a self-contained traversal archive, per-segment candidate metrics, final
route metrics, proxy/contact metadata, and the exact generation configuration.
The existing MuJoCo kinematic viewer must play the accepted route without any
hidden correction.

## Acceptance

The first version is ready for visual feedback only when:

1. it completes step-up, horizontal unequal-height walking, step-down, and
   departure without freezing;
2. the heading remains aligned with horizontal travel;
3. every planned stance has valid sole support;
4. no certified scene penetration occurs;
5. all segment boundaries satisfy the existing discontinuity limits;
6. a fresh automatic validation run passes before the viewer is launched.

