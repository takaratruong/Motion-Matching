# Yaw-Invariant Diverse Approach Search Design

**Date:** 2026-07-22

## Goal

Make coverage-viewer retrieval insensitive to object yaw and retain the widest
available diversity of approach sides.

## Chosen Behavior

The coverage viewer will no longer reject a candidate because its mapped root
starts outside a fixed world-front half-plane. Every approach side remains
eligible for the exact tier and the axis-fallback tier.

The existing 12-option display cap remains. Existing exact-first retrieval,
fallback orientation refinement, grasp-position and approach-axis gates, and
orientation-quality ranking remain unchanged. Removing the directional gate
therefore broadens the candidate pool without weakening grasp quality.

## Hard Rejection Rules

Yaw invariance applies to retrieval, not physics. Candidates continue to be
rejected by:

- hand IK and the 4 cm position, 15 degree approach-axis, and 60 degree full
  orientation limits;
- object collision, with only the final active wrist contact exempt;
- full-body furniture collision with the viewer's 2 cm clearance margin.

Consequently, rotating an object in open space does not remove approach sides.
Near a table or shelf, a rotated candidate may still disappear when its body
actually intersects geometry.

## Scope

Remove `kSceneFront`, the `starts_on_allowed_side` call, and the viewer's
`wrong_side` counter and HUD field. Keep the trajectory library's
`starts_on_allowed_side` API and its unit tests because it remains a useful
optional policy for other callers.

Do not alter trajectory alignment, IK, collision geometry, the interaction
pack, controls, Enter-only search, option cycling, or the live controller.

## Verification

Update viewer source-contract tests to prove that the viewer no longer calls
the side filter or reports wrong-side rejection. Run the established Python
gate, both C++ trajectory binaries, the release viewer build, and the same
single-process live hash and bounded-memory checks. Replace only the running
coverage viewer and do not launch mesh or terrain renderers.
