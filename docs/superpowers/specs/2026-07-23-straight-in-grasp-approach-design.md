# Straight-In Grasp Approach Preference and Retargeting

## Context

The reach-coverage viewer can place a recorded G1 reach at a requested grasp,
solve the final hand pose with posture-aware IK, reject collisions, and cycle
accepted candidates with `[` and `]`. It also measures whole-path backtracking
and excess path length.

The current approach correction is insufficient for grasping. It aligns the
net wrist displacement over the final five frames with the requested approach
axis, but it does not constrain each intermediate wrist sample. A candidate can
therefore slide laterally across the object and correct only at contact. This
is especially unsafe for an open gripper because its fingers may strike the
object before the palm reaches the grasp.

## Goal

Given a grasp pose and a unit approach direction, prefer recorded motions whose
wrist already approaches the grasp along that axis, then retarget the selected
motion through a straight collision-free pre-grasp corridor. The gripper stays
open through the corridor and closes only near contact.

This design preserves candidate diversity: approach quality changes ordering,
not the raw accepted set. A candidate that cannot be safely corridor-retargeted
remains inspectable in the coverage viewer but is not executable.

## Non-Goals

- Locomotion, carrying, placement, or motion-matched episode playback
- Training or integrating the future grasp policy
- Independently synthesizing a new reach when no recorded motion is reusable
- Independently searching outbound return motions
- Treating a poor approach score as a collision; collision remains a separate
  physical-validity result

## Grasp Contract

The straight-in system consumes a `GraspApproachTarget` with:

- `hand_world`: final wrist position and orientation
- `approach_world`: unit vector pointing from pre-grasp toward contact
- `corridor_length_m`: default `0.20`
- `corridor_radius_m`: default `0.03`
- `close_distance_m`: default `0.03`
- optional open and closed hand-DOF targets

The pre-grasp wrist position is:

```text
pregrasp = hand_world.position - corridor_length_m * approach_world
```

The grasp policy will eventually provide this contract directly. Until then,
the coverage viewer derives `approach_world` from its editable object/grasp
frame and uses canonical G1 open-hand DOFs plus the recorded contact DOFs.

## Approach-Quality Measurement

For each accepted candidate, inspect the contiguous wrist-path suffix within
`corridor_length_m` of contact. For each segment, decompose wrist displacement
into:

```text
axial   = dot(delta, approach_world)
lateral = delta - axial * approach_world
```

Measure:

- maximum lateral distance from the pre-grasp-to-contact line
- RMS lateral distance
- cumulative backward axial motion
- maximum and RMS segment-angle error from `approach_world`
- whether the source path reaches the pre-grasp plane

All measurements are expressed in the grasp frame and are therefore invariant
to any rigid world transform.

The accepted list remains complete. Its default ordering becomes
lexicographic:

1. reaches the pre-grasp plane
2. lower maximum lateral drift
3. lower backward-motion ratio
4. lower RMS approach-angle error
5. existing active-arm deformation
6. existing full-orientation error
7. clip and yaw identity for deterministic ties

The viewer retains a raw-order mode for comparison. Ranking alone never changes
an evaluation from accepted to rejected.

## Corridor Retargeting

Retargeting is performed lazily for the selected candidate, so cycling does not
rerun the full exhaustive search.

1. Find the corridor-entry sample where the final contiguous suffix first comes
   within 20 cm of contact.
2. Use at most 0.20 seconds immediately before corridor entry to blend the wrist
   from the recorded path to the exact pre-grasp point and final grasp
   orientation.
3. From corridor entry onward, generate every wrist target on the line from
   pre-grasp to contact. Axial progress follows the source motion's monotonic
   progress after clamping backward steps; smoothstep regularization prevents
   acceleration discontinuities.
4. Hold the requested grasp orientation through the corridor, allowing at most
   10 degrees during the blend into pre-grasp and the existing strict contact
   tolerance at the endpoint.
5. Solve the three waist and seven active-arm joints with the existing
   posture-aware damped least-squares solver. Preserve the source elbow plane,
   remain near the recorded pose, remain near the previous solved frame, and
   enforce G1 joint limits.
6. Preserve all motion before the correction blend. The root is never translated
   vertically or moved to satisfy the wrist target.

The retarget is accepted only if:

- contact pose and orientation satisfy the existing strict limits
- wrist lateral distance stays within the 3 cm corridor
- axial motion never moves backward by more than 2 mm in one sample
- every solved pose and hand DOF is finite
- no G1 body or gripper geometry intersects the environment
- object collision is absent before the allowed final-contact interval
- temporal joint changes remain within existing solver limits

If any condition fails, the viewer shows the recorded candidate with a
`CORRIDOR RETARGET FAILED` diagnostic. Future execution code must skip it and
try the next ranked candidate.

## Gripper Schedule

The active gripper remains at the canonical open-hand target from pre-grasp
until it is `close_distance_m` from contact. It then smoothsteps from open DOFs
to the requested contact DOFs over the final 3 cm.

If a future grasp policy supplies explicit open and closed DOFs, those values
replace the defaults without changing the trajectory interface. Finger and
object collisions are evaluated with the per-frame scheduled DOFs, not merely
the wrist transform.

## Coverage Viewer

The restored `g1_reach_coverage_viewer` remains the only UI in scope.

- `Enter`: rerun exhaustive search after the object or grasp changes
- `[` / `]`: cycle cached accepted candidates without recomputing
- `A`: cycle `RAW`, `PREFERRED`, and `RETARGETED` display modes
- Object translation, rotation, environment toggle, mesh, bones, and rejected
  candidate controls remain unchanged

The HUD adds:

- maximum and RMS lateral drift
- maximum and RMS approach-angle error
- backward-motion ratio
- pre-grasp-plane coverage
- corridor-retarget status and IK deformation
- current raw/preferred rank and total accepted count

The selected path draws the requested corridor and pre-grasp point. In
`RETARGETED` mode, the raw wrist path is a faint reference and the solved path
is emphasized.

## Data Flow

```text
editable object/grasp
  -> grasp pose + approach axis
  -> exhaustive collision/IK search
  -> compact wrist paths
  -> straight-approach metrics
  -> stable preferred ordering
  -> [ / ] selected candidate
  -> lazy corridor retarget
  -> full collision revalidation
  -> animated raw or retargeted preview
```

## Failure Handling

- Invalid or zero approach vectors fail the search request.
- A path too short to establish a pre-grasp suffix receives the worst
  preference rank and reports `NO PREGRASP COVERAGE`.
- IK, joint-limit, non-finite, or collision failure rejects only the selected
  retarget, not the cached exhaustive results.
- Moving or rotating the object marks all cached results and retargets stale;
  `Enter` is required to recompute.
- Candidate ordering and diagnostics are deterministic for identical inputs.

## Testing and Acceptance

Automated tests must prove:

- straight paths outrank lateral-slide and hooked paths
- approach metrics are invariant under rigid world transforms
- preferred ordering retains every raw accepted candidate
- `[` and `]` cycle cached candidates without rerunning search
- a retargeted synthetic path stays inside the corridor and progresses
  monotonically
- the final wrist pose and orientation remain exact
- hand DOFs stay open and close only inside the final 3 cm
- short, non-finite, joint-limited, and colliding retargets fail closed
- root height and pre-correction motion remain unchanged

Live acceptance uses the v3 reach pack in open and coverage environments. The
viewer must expose multiple raw candidates, rank visibly straighter approaches
first, replay selected retargets without lateral contact sliding, and retain
responsive bracket cycling.
