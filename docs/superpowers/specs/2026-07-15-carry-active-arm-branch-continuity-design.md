# Carry active-arm branch continuity design

## Goal

Prevent Carry from switching between redundant active-arm inverse-kinematics branches in one published tick, including the transition-completion and direct-publication boundary, while preserving the current live-root, grasp, object-continuity, recorded-selection, and rollback contracts.

## Root cause

Carry evaluates recorded and layered poses into a candidate, then publishes that candidate through the 0.50 second seam. The transition loop currently tries a blend containing the candidate's raw active-arm rotations before it tries the last published active-arm rotations. A raw recorded pose can be a different in-limit seven-joint solution with the same hand/object transform. Early in the seam the raw blend may fail IK and the preserved branch is published; near progress 1 the raw branch becomes acceptable and replaces the preserved branch immediately. Neither candidate production nor transition acceptance checks active-joint distance from `last_safe_pose_`.

A deterministic certified recorded range exposes both discontinuities under default Carry/IK configuration. Its alternate right arm has the same grasp transform as the final Hold arm but a maximum joint separation of 1.57344258 radians. With 0.04 second updates, current Carry publishes a 0.303116 radian initial correction, stays on the preserved branch, then flips 1.30238 radians near completion while reporting recorded mode and zero hand error.

## Continuity contract

Every published Carry pose must keep each active-arm local joint rotation within an inclusive per-update budget of the previous published pose. The budget is derived from the existing IK policy:

```text
maximum active-arm publication step =
    min(pi, 2 * IKConfig.maximum_step_radians)
```

The default is therefore 0.20 radians. Quaternion distance uses normalized, sign-invariant shortest-arc distance, matching existing rotation comparisons. The inclusive comparison retains the repository's floating representational slack policy. This is an internal publication invariant, not a new public configuration field, and does not change any IK request, accepted-error, Carry drift, or root threshold.

## Branch-canonical candidate

After recorded or layered evaluation produces a directly solved candidate, Carry will try to canonicalize only its active arm onto the currently published branch:

1. Copy the candidate pose so all recorded/layered non-arm channels remain unchanged.
2. Replace its active-arm local rotations with those from `last_safe_pose_`.
3. Solve the same candidate hand/object target with the existing IK configuration tightened to the existing Carry position and orientation drift limits.
4. Recompute the published object from the solved hand.
5. Accept this canonical candidate only when IK succeeds, the object remains valid and continuous under the unchanged limits, and every active joint is inside the inclusive publication-step budget.

On success, the canonical pose/object replaces the raw candidate before seam-key handling. Because every later update repeats this process from the last published branch, recorded playback can retain its non-arm motion and grasp target without importing an arbitrary redundant arm solution. At progress 1, the ordinary raw transition and the following direct candidate are already on the same branch, so direct publication can resume without a special completion jump.

If canonicalization fails, Carry marks the candidate as not directly solved rather than publishing its raw arm. The existing transition/fallback machinery remains responsible for safe progress.

## Publication gate

Candidate canonicalization alone is not sufficient because either transition retry can run IK and move the arm. Every solved transition pose—raw blend or preserved-arm retry—must pass the same active-arm distance check against `last_safe_pose_` before it can update transition progress, object anchors, recorded mode, or last-safe state. A rejected arm step proceeds through the existing retry/backoff order; if no bounded solution is valid, the existing live-root-remapped last-safe fallback is published.

The direct path receives only a successfully canonicalized candidate. Candidate keys, recorded selection epochs, cumulative seam progress, transaction snapshots, and completion/deactivation rules remain unchanged. No new mutable controller state is required, so exception rollback coverage remains intact.

## Production-path regression

The Carry test will construct a real certified recorded range `[115,150)` from the runtime fixture:

- preserve the original per-frame roots and positions;
- set all non-root rotations to the final Hold pose;
- set right-arm bones 24 through 30 to a frozen alternate in-limit seven-DOF solution;
- derive each recorded object transform from that frame's right-hand world transform and the unchanged affordance, making the grasp exact and canonical;
- force every pose/trajectory row in the range to tie at the query optimum;
- set one constant recorded hand-DOF marker so progress 1, its completion tick, and later direct updates are externally observable.

The test starts from the original final Hold pose/object and runs 20 ordinary 0.04 second updates with default Carry and IK configuration. On every tick it requires recorded mode, exact live-root publication, and the existing hand/object tolerance. It records the maximum shortest-arc delta over all seven active joints and requires every published tick to remain within the inclusive `2 * maximum_step_radians` budget. It separately checks the hand-DOF marker at progress 1, the completion tick, and direct publication, proving the continuity contract covers the whole seam rather than merely a later steady-state frame.

The current implementation must fail this regression with the independently reproduced approximately 1.30238-radian maximum step. The corrected implementation must report the exact measured maximum from the same test run.

## Verification and scope

Focused verification covers Carry, runtime, controller adapter, and the dedicated Carry release-fast-math target. The full safe suite is required before the implementation commit. Static diff checks and tracked-clean status remain mandatory.

This correction is intentionally limited to active-arm Carry branch continuity. The separately identified unmapped Head/Neck adapter fallback and 25 Hz-to-60 Hz presentation stutter remain separate follow-up work so their authority, tests, and implementation are not conflated with the Carry solver fix. The protected untracked repo-root `interaction_query_probe` is never accessed; only safe targets may be used.
