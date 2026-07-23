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

## Review-fix implementation

- Changed the final recorded-return sample to start from the aligned recorded
  source angles without the transported previous-frame joint correction. This
  guarantees that zero target weight also means zero transported seed
  correction.
- Added an explicit final-frame wrist check against the aligned recorded source
  endpoint, with position and orientation tolerances of `1e-6`.
- Added a one-frame return regression with a valid 25 mm / 0.18 rad contact
  correction. It checks final wrist position and orientation directly against
  the aligned recorded endpoint.
- Added direct behavior tests for body/held-object rejection,
  body/environment rejection, and acceptance when the active grasp wrist is
  inside the held object.
- Changed the dynamic attached-object assertions to compare the actual frozen
  `hand_in_object` trajectory with an independently calculated expected target
  trajectory.
- Replaced the ambiguous tangency claim with a boundary test that documents the
  nominal, axis-scaled approximately 1 micrometre SAT tolerance: 0.5
  micrometre penetration is tolerated and 2 micrometres is rejected.
- Changed collision sample assertions to name the intended sample explicitly,
  including the calculated minimum-bottom sample for the SAT boundary.

Files changed by the review fix:

- `reach_return.cpp`
- `tests/cpp/test_reach_return.cpp`
- `.superpowers/sdd/task-4-report.md`

## Review-fix TDD evidence

RED — the short-return endpoint regression was added before the production
change:

```bash
make build/tests/test_reach_return
./build/tests/test_reach_return
```

`make` rebuilt the target successfully. The test command exited 1 with:

```text
reach return FAILED: final wrist retained transported position correction: 0.000926
```

The failure reproduced the review finding: the target correction weight was
zero, but the transported temporal seed still biased the final IK solution.

GREEN — after dropping the transported seed on the final sample and adding the
explicit endpoint check:

```bash
make build/tests/test_reach_return
./build/tests/test_reach_return
```

Result: exit 0.

```text
reach return PASS
```

The three direct collision-path tests, the strengthened frozen-object
trajectory assertions, and the SAT boundary/sample assertions are all part of
this focused binary and passed in the same GREEN run.

## Review-fix required regression results

Exact commands:

```bash
make build/tests/test_reach_return build/tests/test_reach_coverage build/tests/test_reach_search build/tests/test_episode_reach_planner build/tests/test_interaction_episode
./build/tests/test_reach_return
./build/tests/test_reach_coverage
./build/tests/test_reach_search
./build/tests/test_episode_reach_planner
./build/tests/test_interaction_episode
```

Result: exit 0 for the complete command sequence. Build output:

```text
make: 'build/tests/test_reach_return' is up to date.
make: 'build/tests/test_reach_coverage' is up to date.
make: 'build/tests/test_reach_search' is up to date.
make: 'build/tests/test_episode_reach_planner' is up to date.
make: 'build/tests/test_interaction_episode' is up to date.
```

Test output (`test_reach_coverage` and `test_reach_search` are silent on
success):

```text
reach return PASS
episode reach planner PASS
interaction episode PASS
```

## Review-fix self-review

- Confirmed the final source target has exactly zero warped correction and the
  final temporal seed is exactly the aligned recorded source angles.
- Confirmed the final achieved wrist position and orientation are explicitly
  checked against that source endpoint before the frame is accepted.
- Confirmed the short-return regression uses a solved, accepted large contact
  correction and checks both endpoint position and orientation.
- Confirmed direct tests bind each requested collision outcome and assert the
  intended rejected sample rather than deriving it from the truncated result.
- Confirmed the active wrist is the only body point exempted by the held-object
  collision path in the acceptance fixture.
- Confirmed attached-object expectations are calculated from the source path,
  contact delta, and inverse-time-warp weight independently of the solved wrist
  used to produce the actual object transform.
- Confirmed the SAT test accurately brackets the nominal axis-scaled
  approximately 1 micrometre tolerance.
- `git diff --check` exits 0 with no output.
- Generated reach packs and binaries, along with the pre-existing
  `.superpowers/sdd/progress.md` modification, remain outside this change.

## Review-fix concerns

None.
