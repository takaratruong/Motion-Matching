# G1 Local Contact-Segment Retargeting Design

**Date:** 2026-07-31
**Branch:** `research/g1-torch-terrain-kinematics`
**Scope:** 50 Hz kinematic terrain motion matching only. Sonic, tracking,
physics, and learned depth models remain out of scope.

## Failure being fixed

The representative 176-clip viewer is not usable. A deterministic constant
forward trace reproduced repeated transition rejection, backward drift, and
single-step search times of 26--28 seconds. At the reproduced stair state the
contact inventory contains 1,281 segments, but the current 5 cm absolute-scene
registration gate admits only five forward entries and no reverse or lateral
entries before contact validation.

That gate makes recordings reusable only near their original XY coordinate in
their original scene. It contradicts the local terrain-feature representation
and the original support-foot placement design. The command filter also uses
facing as travel direction, so a command that travels across a tread while
facing along the stair cannot retrieve a lateral segment.

## Design

Terrain clips are local motion examples, not absolute scene trajectories.
For every candidate entry:

1. align the recorded root yaw to the commanded facing yaw;
2. place the recorded root at the next desired root XY;
3. translate the whole segment vertically by the difference between target
   and recorded terrain under the entering support ankle;
4. validate the complete emitted contact segment against the query heightfield;
5. commit only an accepted segment sequentially through its contact phase.

The absolute source-scene XY mismatch and phase-direction gates are disabled
for local retargeting. They are not replaced by a weaker global threshold.
The existing 91-point body-relative patch plus four command-path heights ranks
local geometry, and the authoritative MuJoCo-FK segment validator remains the
hard query-terrain safety gate.

Command compatibility uses travel and facing separately. Travel direction is
the normalized final command-trajectory displacement. Candidate source root
velocity is rotated by the candidate yaw and compared with that travel
direction. Candidate yaw is derived from commanded facing, not travel. This
allows forward, backward, lateral, and diagonal motion while preserving an
independent body heading.

Ranked terrain rescue and terrain-entry override inspect at most the configured
transition candidate count (32 in the current experiment). If none pass, the
matcher retains the incumbent and publishes a rejection; it may not scan the
entire corpus in one control step.

## Evaluation contract

The contact-segment policy must be available to the existing deterministic
21-route same-stair evaluator. Evaluation records full kinematics, terrain
contact, selection identity, rejection reason, and timing. A route fails fast
after any step exceeding 1 second; timing is diagnostic and excluded from the
deterministic hash.

Before another viewer launch, the current representative corpus must run:

- constant forward without backward progress or a step above 1 second;
- riser reversal with measurable downhill progress;
- both cross-tread routes;
- both diagonal-down routes;
- both upper side exits; and
- mixed-adversarial.

The first iteration need not pass all 21 behavioral outcomes, but it must have
nonzero safe candidate reachability for reverse and lateral commands, must not
freeze, and must improve over the measured zero-candidate baseline. A viewer is
shown only after those automated gates pass.

## Test strategy

Implementation is test-first:

1. a terrain transition test proves commanded facing controls candidate yaw;
2. a command-compatibility test proves travel is independent from facing;
3. a local placement test proves different absolute source XY can retarget to
   the same query contact while retaining vertical support placement;
4. a ranked-rescue test proves no more than 32 candidates are composed;
5. evaluator tests prove contact-policy injection and fail-fast timing; and
6. guarded real-data routes provide the final behavioral evidence.
