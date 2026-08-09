# PFNN Morphology-Scaled Terrain Design

## Goal

Test the approved G1 retarget against the exact PFNN fitted terrain uniformly
scaled to the G1 morphology.

## Rationale

The pinned GMR configuration scales the PFNN lower body and root by
`0.9 * (1.75 / 1.8) = 0.875`.  Scaling the terrain by the same factor preserves
slope angles, tread-to-riser ratios, and foot-to-terrain geometry.  Scaling
only height or only horizontal coordinates is forbidden because it changes
terrain grade.

## Transform

The native PFNN terrain remains the source authority.  For scale `s = 0.875`,
the derived G1 terrain query is:

```
h_g1(x, y) = s * h_native(x / s, y / s)
```

where `h_native` already includes PFNN's axis conversion and centimeters-to-
meters conversion.  Scaling occurs about the same world origin used by GMR.
The G1 motion, terrain fit parameters, contacts, timeline, and thresholds are
unchanged.

## Vertical Placement

The first scaled comparison measures a median authored-stance gap of
`0.05224985936713168` m, meaning the uniformly scaled terrain is that far below
the G1 soles.  World height origin is arbitrary, so the approval placement adds
this one constant to every terrain height after uniform scaling:

```
h_placed(x, y) = h_g1(x, y) + 0.05224985936713168
```

The value is derived once from the median of all 80 PFNN-authored stance probes
and is recorded in the comparison and viewer.  It is not recomputed per frame.

## Provenance and Scope

Comparison reports and the viewer record the exact scale.  The original PFNN
fit artifact is never overwritten.  A scale other than finite positive 0.875
is rejected by the approval CLI.  GRAIL terrain is never scaled because GRAIL
motion and USD terrain are already paired for G1 in meters.

This design supersedes the unexecuted root-trajectory correction experiment in
`2026-08-08-pfnn-root-trajectory-correction-design.md`.  No root correction,
IK, foot locking, or terrain deformation is performed.

## Acceptance

Tests prove uniform XYZ scaling and unchanged native queries.  The existing
stance contact gate is rerun once at scale 0.875, followed by the exact terrain
viewer.  The result is preserved whether accepted or rejected; no parameter
tuning follows this experiment.
