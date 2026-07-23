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
  An approximately 1 micrometre axis-scaled numerical tolerance preserves
  zero-depth support contact; overlaps beyond that tolerance are rejected.
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

   After applying the axis-scaled float tolerance, the tangent test and the
   existing rotated deep-overlap test both passed.

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

## Initial review-fix implementation

- The initial endpoint correction reset the final temporal seed to the aligned
  recorded source. The follow-up contract fix documented below supersedes that
  behavior: every later frame now retains the transported previous accepted
  correction, including the final frame.
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

GREEN for the initial endpoint regression — after adding the explicit endpoint
check. The later contract review found that resetting the final transported
seed was not permitted; the superseding RED/GREEN evidence is documented
below.

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

- Confirmed the final source target has exactly zero warped target correction.
  The follow-up contract fix below supersedes the initial final-seed reset and
  preserves transported temporal correction on that frame.
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

## Final temporal-transport contract fix

### Implementation

- Restored transported previous accepted solved-minus-source joint correction
  as the temporal seed on every recorded-return frame, including the final
  frame.
- Kept the final target at the aligned recorded nominal endpoint and introduced
  a final-only `PostureIKConfig` whose accepted position and orientation errors
  are both `1e-6`. This causes task-priority polish to satisfy the endpoint
  while retaining the transported input seed.
- Kept the independent achieved-wrist position/orientation endpoint check at
  the same `1e-6` tolerances.
- Strengthened the short-return / 25 mm / 0.18 rad contact-correction test. It
  still checks the exact final wrist endpoint, proves that the fixture produces
  a materially distinct transported final seed, and compares the returned
  posture against an independently solved transported-seed endpoint reference.
  In this fixture the polished nominal endpoint posture coincides with the
  source posture, so seed use cannot be distinguished from final pose alone.
- Corrected SAT wording throughout this report: the SAT permits approximately
  1 micrometre of axis-scaled numerical tolerance and rejects overlap beyond
  that tolerance; it does not claim rejection of every positive overlap.

Files changed:

- `reach_return.cpp`
- `tests/cpp/test_reach_return.cpp`
- `.superpowers/sdd/task-4-report.md`

### RED/GREEN evidence

RED — after restoring required temporal transport on the final frame but before
tightening its IK acceptance config:

```bash
make build/tests/test_reach_return && ./build/tests/test_reach_return
```

Result: exit 1.

```text
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_reach_return.cpp \
  reach_return.cpp reach_placement.cpp reach_database.cpp reach_motion.cpp \
  interaction_hand_trajectories.cpp interaction_posture_ik.cpp \
  interaction_ik.cpp interaction_pose.cpp interaction_target.cpp -o build/tests/test_reach_return
reach return FAILED: reachable return was rejected
```

GREEN — after selecting the final-only `1e-6` position/orientation acceptance
config:

```bash
make build/tests/test_reach_return && ./build/tests/test_reach_return
```

Result: exit 0.

```text
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_reach_return.cpp \
  reach_return.cpp reach_placement.cpp reach_database.cpp reach_motion.cpp \
  interaction_hand_trajectories.cpp interaction_posture_ik.cpp \
  interaction_ik.cpp interaction_pose.cpp interaction_target.cpp -o build/tests/test_reach_return
reach return PASS
```

### Exact required regression run

Commands:

```bash
make build/tests/test_reach_return build/tests/test_reach_coverage build/tests/test_reach_search build/tests/test_episode_reach_planner build/tests/test_interaction_episode
./build/tests/test_reach_return
./build/tests/test_reach_coverage
./build/tests/test_reach_search
./build/tests/test_episode_reach_planner
./build/tests/test_interaction_episode
```

Result: exit 0. Exact output:

```text
make: 'build/tests/test_reach_return' is up to date.
make: 'build/tests/test_reach_coverage' is up to date.
make: 'build/tests/test_reach_search' is up to date.
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -I. -pthread \
  tests/cpp/test_episode_reach_planner.cpp \
  episode_grasp_provider.cpp episode_reach_planner.cpp \
  reach_return.cpp \
  reach_search.cpp reach_coverage.cpp reach_placement.cpp \
  reach_database.cpp reach_motion.cpp interaction_hand_trajectories.cpp \
  interaction_posture_ik.cpp interaction_ik.cpp interaction_pose.cpp \
  interaction_target.cpp -o build/tests/test_episode_reach_planner
make: 'build/tests/test_interaction_episode' is up to date.
reach return PASS
episode reach planner PASS
interaction episode PASS
```

`test_reach_coverage` and `test_reach_search` are silent on success.

### Self-review

- Confirmed there is no final-frame exception in transported temporal-seed
  construction.
- Confirmed only the final sample receives the tightened IK acceptance values;
  earlier return frames retain the established coverage acceptance values.
- Confirmed the endpoint tolerances are shared by the final IK acceptance
  config and the explicit achieved-wrist endpoint check.
- Confirmed target correction weight remains zero on the final sample while
  temporal correction remains transported.
- Confirmed the focused regression exercises a nontrivial transported seed and
  checks both endpoint position and orientation.
- Confirmed the exact required coverage, search, planner, and interaction
  regressions pass.
- Confirmed `git diff --check` exits 0 and no generated pack or binary is part
  of this fix.

### Concerns

None.
