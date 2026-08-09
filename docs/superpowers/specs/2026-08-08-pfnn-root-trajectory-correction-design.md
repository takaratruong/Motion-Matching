# PFNN Root-Trajectory Correction Design

## Goal

Test the smallest correction for the exact PFNN terrain-transfer mismatch while
preserving the G1 motion the user approved.

## Evidence

For source frames 8160 through 8279, PFNN's transformed human root travels
1.209 m horizontally and rises 0.509 m.  The pinned GMR output travels 1.058 m
and rises 0.445 m.  The ratio is 0.8758, matching GMR's configured root scale
`0.9 * (1.75 / 1.8)`.  On PFNN fitted patch 8967, the unchanged G1 has stance
gaps from -2.89 cm to +7.03 cm.

## Approaches

1. Restore only the source root trajectory and one constant vertical anchor.
   This is the recommended minimal hypothesis test because it removes the
   measured global-scale error without changing the approved joint motion.
2. Scale the PFNN terrain down to 0.8758.  This is rejected because it changes
   the physical environment instead of testing whether the G1 transfers.
3. Run per-frame leg IK with stance-foot locks.  This is deferred because it
   changes joint motion and may introduce the dragging seen in earlier IK
   attempts.  It is considered only if the root-only correction is visibly
   good but still misses the numeric contact gate.

## Correction

The prepared PFNN frames already provide source root positions in GMR's Z-up,
meter coordinate system before GMR morphology scaling.  For the displayed
interval, the corrected root is:

```
root_xy[t] = source_root_xy[t]
root_z[t]  = gmr_root_z[0] + source_root_z[t] - source_root_z[0] + z_anchor
```

All 29 G1 joint positions and root orientations remain bit-identical to the
approved GMR result.  The terrain artifact and timeline remain unchanged.

`z_anchor` is one constant computed from the exact PFNN-authored stance probes.
First evaluate the root-restored motion with `z_anchor = 0`.  Then set
`z_anchor` to the negative median signed stance gap and evaluate once more.
There is no per-frame correction, contact lock, terrain deformation, temporal
filter, or threshold change.

## Acceptance and Stop Rule

The experiment emits the raw and anchored gap distributions and preserves both
motion artifacts.  It passes only if the existing terrain-transfer contact
gate accepts and the user approves the interactive motion.  If it rejects,
stop after reporting the first failure and distributions; do not add IK or
tune the anchor.

Tests prove exact source-root restoration, bit-identical joints/orientations,
one scalar vertical shift, median-gap computation from stance probes only,
artifact provenance, and deterministic repeated output.
