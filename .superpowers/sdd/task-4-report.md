# Task 4: Shape and Preflight the Warped Recorded Return

## Implementation summary

- Added `shape_recorded_return(...)` and the requested return rejection/result
  types.
- Published the selected outbound contact pose unchanged as return sample zero.
  Aligned each later recorded frame with the existing placement transform,
  applied the inverse-time smoothstep SE(3) contact correction, and solved the
  existing posture-aware task-priority IK with transported temporal correction.
- Derived the held object from every solved wrist and frozen
  `hand_in_object`. Reused the existing single-pose body-vs-object and
  body-vs-environment feasibility path with the active-hand object exemption,
  and added 15-axis oriented-box SAT for held-object-vs-environment overlap.
  A 1 micrometre axis-scaled overlap tolerance preserves zero-depth support
  contact while rejecting positive overlap.
- Changed playable plan selection to exclude empty returns before regeneration,
  rank all remaining playable candidates, preflight returns in cost order, and
  continue after failed regeneration/return preflight. The accepted
  `ReachPlan` stores `return_poses`.
- Did not change episode playback or outbound reach shaping, ranking, or
  collision behavior.

## Files changed

- Created: `reach_return.h`
- Created: `reach_return.cpp`
- Modified: `reach_coverage.h`
- Modified: `reach_search.cpp`
- Modified: `episode_reach_planner.h`
- Modified: `episode_reach_planner.cpp`
- Modified: `Makefile`
- Created: `tests/cpp/test_reach_return.cpp`
- Modified: `tests/cpp/test_episode_reach_planner.cpp`
- Created: `.superpowers/sdd/task-4-report.md`

The pre-existing `.superpowers/sdd/progress.md` modification and generated
packs/viewer binaries were not changed for Task 4 and are excluded from the
commit.

## TDD evidence

1. Return shaper RED:

   ```text
   make build/tests/test_reach_return
   tests/cpp/test_reach_return.cpp:3:10: fatal error:
   reach_return.h: No such file or directory
   make: *** [Makefile:621: build/tests/test_reach_return] Error 1
   ```

2. Return shaper GREEN after the minimal implementation:

   ```text
   reach return PASS
   ```

3. Planner integration RED, after correcting an unrelated missing test include:

   ```text
   tests/cpp/test_episode_reach_planner.cpp:501:15: error:
   'const struct episode::ReachPlan' has no member named 'return_poses'
   tests/cpp/test_episode_reach_planner.cpp:505:19: error:
   'const struct episode::ReachPlan' has no member named 'return_poses'
   ```

4. Planner integration GREEN:

   ```text
   reach return PASS
   episode reach planner PASS
   ```

5. Self-review found that inclusive SAT would classify the real viewer's exact
   object/support tangency as overlap. The focused regression failed before the
   fix:

   ```text
   reach return FAILED: zero-depth support contact was treated as object
   overlap: rejection 4 at sample 0, contact gap 0.000000
   ```

   After applying strict positive-overlap semantics with the axis-scaled float
   tolerance, the tangent test and the existing rotated deep-overlap test both
   passed.

## Tests and exact results

Baseline before Task 4:

```bash
make build/tests/test_reach_coverage \
  build/tests/test_reach_search \
  build/tests/test_episode_reach_planner
./build/tests/test_reach_coverage
./build/tests/test_reach_search
./build/tests/test_episode_reach_planner
```

Result: exit 0; coverage/search were silent and the planner printed:

```text
episode reach planner PASS
```

Focused and required reach regressions:

```bash
make build/tests/test_reach_return \
  build/tests/test_reach_coverage \
  build/tests/test_reach_search \
  build/tests/test_episode_reach_planner
./build/tests/test_reach_return
./build/tests/test_reach_coverage
./build/tests/test_reach_search
./build/tests/test_episode_reach_planner
```

Result: exit 0; coverage/search were silent and the focused binaries printed:

```text
reach return PASS
episode reach planner PASS
```

Header-consumer regression:

```bash
make build/tests/test_interaction_episode
./build/tests/test_interaction_episode
```

Result:

```text
interaction episode PASS
```

Static/scope check:

```bash
git diff --check
```

Result: exit 0 with no output.

## Self-review

- Confirmed sample zero is a bitwise copy of `solved_contact`; the first
  correction is therefore exactly one and no playback seam is introduced.
- Confirmed the final sample evaluates `sample / return_count == 1`, so its
  correction weight is zero and its wrist target is the aligned recorded
  endpoint.
- Confirmed every later frame uses the aligned recorded pose as posture source
  and transports the previous solved-minus-source joint correction in the
  temporal seed.
- Confirmed non-finite or unaccepted IK reports `InvalidSolver` at the first
  rejected sample.
- Confirmed body/object, body/environment, and held-object/environment checks
  use the dynamic object reconstructed from each solved wrist.
- Confirmed empty-return clips remain accepted outbound coverage identities but
  are skipped only by playable plan selection.
- Confirmed failed return preflight falls through to the next ranked accepted
  candidate and the selected plan owns the shaped return poses.
- Confirmed requested reach coverage/search/planner regressions pass and no
  generated pack or binary is staged.

## Concerns

None.
