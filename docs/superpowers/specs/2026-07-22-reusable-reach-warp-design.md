# Reusable Reach Warp Design

**Date:** 2026-07-22

## Purpose

Treat each captured motion as a reusable spatial reach rather than a recorded
grasp. The same clip must be retrievable for multiple requested approach
directions and multiple wrist orientations. Requested grasp pose is imposed
during trajectory shaping, and collision checks decide whether the resulting
reuse is valid in the current scene.

This corrects two measured failures in the current baseline:

- the default query exactly reproduces a stored endpoint position and wrist
  rotation, but an unsupported assumption that wrist `+X` equals travel
  direction creates a `114.3` degree approach mismatch;
- the global-neutral extractor merged repeated reach-return cycles, producing
  98 proposals where a prominent-endpoint audit finds approximately 164.

## Invariants

1. Reach identity is independent of recorded wrist orientation.
2. Reach eligibility is independent of recorded approach direction.
3. Endpoint position and active hand are the only hard retrieval filters.
4. Recorded approach direction affects ranking because smaller warps generally
   preserve motion quality, but a large angle cannot remove a candidate.
5. Query-time shaping targets the requested endpoint, approach direction, and
   wrist orientation together.
6. Root, waist, legs, and inactive arm remain unchanged in this baseline.
7. A motion is accepted only after final kinematic gates and collision checks;
   rejected motions remain visible for diagnosis.

## Corpus Recovery

Replace one-outside-run/one-maximum segmentation with prominent endpoint peak
detection. Operate on the root-relative active wrist trace and find every local
maximum that has:

- at least `0.18 m` excursion from the estimated neutral cluster;
- at least `0.08 m` local prominence;
- at least `0.60 s` separation from another selected peak.

Prominence, rather than return inside a single global neutral radius, separates
repeated reaches even when the arm's neutral pose drifts. For each endpoint,
search backward for the nearest departure/neutral minimum, bounded to `3.6 s`.
Move a peak backward only as needed to the nearest frame with at least `1 cm`
of fixed-root approach displacement during the preceding `0.2 s`. Preserve
source frame provenance and stable proposal identities.

The rebuilt review must include all eight reach recordings. The explicit
`walking` and `carry_walking` recordings remain excluded. A before/after report
must list detected and published reaches per source so missed or duplicated
cycles are visible.

## Retrieval

Retrieve every same-hand endpoint within the existing `0.45 m` spatial
envelope. Do not compare the stored wrist quaternion with the query quaternion.
Do not reject on source/query approach angle.

Rank candidates deterministically by:

1. endpoint distance;
2. magnitude of the required approach-direction warp;
3. clip index.

This ordering favors natural reuse while retaining candidates from other
directions when closer matches do not solve.

## Trajectory Warp

Let `e` be the recorded endpoint, `g` the requested grasp position, `a_s` the
recorded final approach direction, and `a_t` the requested approach direction.
Compute the shortest rotation `R` from `a_s` to `a_t`.

For every wrist sample `p_i`, apply two smooth corrections:

```text
p'_i = p_i
     + translation_weight_i * (g - e)
     + approach_weight_i * (R * (p_i - e) - (p_i - e))
```

The translation weight uses smoothstep over the complete outbound clip. The
approach weight ramps before the terminal approach interval, reaches one by
five frames before contact, and remains one through contact. Consequently the
measured final `0.2 s` travel direction converges to `a_t` instead of merely
rotating the final wrist frame.

Independently compute the shortest wrist-orientation correction from the
recorded endpoint quaternion to the requested quaternion. Ramp it smoothly
over the outbound clip and solve the active seven-joint arm against each warped
position/orientation target. Retrieval remains unchanged for all requested
orientations; only shaping success and measured final error can differ.

The final gates remain explicit and ordered:

1. finite solver result;
2. position error at most `4 cm`;
3. approach-axis error at most `15 degrees`;
4. full wrist-orientation error at most `60 degrees`;
5. object collision;
6. environment collision.

Joint-limit saturation remains a non-exclusive diagnostic.

## Contact and Collision Semantics

The generic object grasp lies on a surface, while the current character model
uses conservative spherical/capsule wrist proxies and no hand mesh. During the
last five frames only, exempt the active wrist contact chain from collision
with the target object:

- active wrist-roll, wrist-pitch, and wrist joint spheres;
- capsules connecting those three wrist joints.

Do not exempt the elbow, the elbow-to-wrist-roll segment, torso, inactive arm,
legs, or any body part from furniture/environment collision. Before the final
five frames, the complete body remains subject to object collision.

Run collision diagnostics for every finite shaped trajectory even when an
earlier kinematic gate fails. Preserve the exclusive first rejection used for
acceptance, but also report non-exclusive observed object and environment
collisions so the viewer never implies that collision was tested when it was
not.

## Viewer Behavior

Retain the older viewer's light background, blue skeleton, lime accepted path,
camera, and furniture style. Always animate the selected complete trajectory.
Use lime only for accepted motions; use orange/red for rejected motions and
show the first rejection plus observed collision flags in the HUD.

Movement and rotation mark results stale. `Enter` reruns retrieval, warp, IK,
and collision checks. Cycling includes complete rejected options after accepted
options so failures remain inspectable.

## Verification

Automated tests must establish:

- changing requested wrist orientation does not change the retrieved clip set;
- changing requested approach direction does not change the retrieved clip
  set inside the spatial envelope;
- a source reach can be warped through large approach-angle changes and the
  final measured travel direction follows the request;
- zero-retarget remains within `1 mm` and `0.5 degrees`;
- terminal active-wrist contact is allowed while elbow/body/object penetration
  still rejects;
- open-space evaluation cannot report an environment collision;
- a synthetic long outside-neutral run with several prominent endpoints emits
  each reach rather than one merged proposal;
- real-corpus counts are reported for every included source and bilateral
  mirroring remains exact;
- accepted and rejected viewer colors/labels cannot be confused.

Rebuild the bilateral pack, rerun the deterministic perturbation report, and
compare source, direction, height, and augmentation coverage before replacing
the current baseline artifact.

## Non-Goals

This correction does not add locomotion stitching, diffusion, waist IK,
full-body IK, object-specific motion identity, or mesh rendering. Those remain
separate experiments after the reusable-reach baseline is measured correctly.
