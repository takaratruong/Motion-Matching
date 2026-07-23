# Direct Reach Ranking

## Goal

Prefer reaches whose wrist travels directly toward the requested grasp over
reaches that first move away and then hook back. Every candidate that already
passes IK, grasp, joint-limit, object-collision, and environment-collision
checks must remain available as a fallback.

This is a ranking-only change. It must not alter candidate enumeration,
posture IK, grasp placement, collision checks, acceptance thresholds, or the
number of accepted motions.

## Proven cause

The exhaustive search currently sorts accepted motions by average upper-body
IK deformation, final orientation error, final approach error, clip, and yaw.
It stores the full active-wrist path but does not use that path in ranking.
Consequently, a low-deformation motion can rank highly even when its wrist
makes substantial progress away from the grasp before returning.

## Path-quality metrics

For the shaped active-wrist samples `p[0] ... p[n]`, use the final wrist
sample `p[n]` as the path endpoint. This makes the score independent of world
translation and avoids mixing the wrist target with object geometry.

Compute:

- `backtrack_ratio`: the sum of all frame-to-frame increases in distance to
  `p[n]`, divided by the straight-line distance from `p[0]` to `p[n]`.
- `excess_path_ratio`: total wrist-path length divided by that straight-line
  distance, minus one, clamped to zero.
- `directness_cost`: `backtrack_ratio + 0.25 * excess_path_ratio`.

Use a 0.05 m minimum denominator so nearly stationary paths remain finite and
cannot amplify numerical noise. A one-sample path has zero cost.

Backtracking receives the dominant weight because moving away and cutting
back is the observed defect. Excess path length remains a smaller preference
so natural curved reaches are not treated like reversals.

## Ranking and diagnostics

Sort accepted motions lexicographically by:

1. `directness_cost`;
2. the existing average upper-body deformation;
3. final orientation error;
4. final approach error;
5. clip and yaw identifiers for deterministic ties.

No directness threshold or rejection category is added. Hooked reaches remain
cycleable after the direct options, preserving coverage and diversity.

Store all three path metrics on each evaluation. Compact search and regenerated
full evaluations must agree on them. The viewer HUD must display directness,
backtracking, and excess-path values for the selected option.

## Failure handling

Non-finite path samples or metrics make the candidate an invalid solver result,
consistent with existing non-finite evaluation handling. Empty paths cannot be
accepted by the existing shape pipeline. A one-sample or nearly stationary
finite path receives a finite zero or denominator-clamped score.

## Verification

Tests must be written before implementation and demonstrate:

1. a direct wrist path ranks ahead of a hooked path with the same endpoint,
   while both remain accepted;
2. a path with no backward progress has zero backtracking;
3. translation and rigid yaw rotation leave all path-quality metrics unchanged;
4. compact and regenerated evaluations publish identical metrics;
5. repeated searches return deterministic ordering;
6. all existing reach, IK, and collision tests continue to pass;
7. the real pack still accepts the same 477 open-space motions, including 231
   left-hand and 246 right-hand motions across ten root-azimuth sectors;
8. the exhaustive real-pack search remains complete within 30 seconds.

The final handoff requires a rebuilt and relaunched flat-terrain viewer so the
new ordering can be checked by cycling the first several accepted options.

## Result

The directness metric and direct-first comparator were implemented with
test-first coverage. Twelve Python viewer contract tests pass, as do the
focused reach-coverage, reach-search, posture-IK, and hand-trajectory C++
test binaries.

The rebuilt real-pack probe processed every one of the 4,608 instances in
each environment. Open-space search completed in 13.1497150 seconds and
retained exactly 477 accepted motions: 231 left-hand, 246 right-hand, across
ten root-azimuth sectors. The complete acceptance and rejection report is
identical to the pre-ranking report after excluding elapsed-time fields.
Therefore the change affects accepted-motion order, not the accepted set.

The viewer now displays directness, backtracking, and excess-path values for
the selected option so the new ordering can be assessed while cycling the
same complete set of valid motions.
