# Orientation-Refined Grasp Fallback Design

**Date:** 2026-07-22

## Goal

Improve the visible wrist orientation of axis-fallback pickup motions without
losing the positional coverage that currently supplies up to 12 options. Also
reject motions whose forearm or body penetrates the object or furniture.

## Chosen Approach

Retain exact full-pose retrieval as the first tier. When exact valid coverage
is below 12, retain approach-axis retrieval but change shaping from
position-only IK to soft full-orientation refinement:

1. Position and approach-axis acceptance remain hard constraints.
2. The IK target uses the complete requested world grasp rotation.
3. Fallback IK may attempt orientation corrections up to 180 degrees and uses
   a reduced orientation weight of 0.10 metres per radian so position remains
   dominant.
4. Run 16 fallback IK iterations instead of the default 8.
5. Keep the best solved pose even when it does not satisfy exact quaternion
   acceptance, provided its final position is within 4 cm and its approach
   axis is within 15 degrees.
6. Add a soft full-quaternion term, weighted `0.25`, to fallback retrieval cost
   after the hard position and axis gates. This sends candidates with useful
   source wrist twist to IK first without excluding the rest.
7. Record each valid option's achieved full quaternion error and sort the
   accepted fallback options by achieved error, then retrieval cost and clip
   ID. Exact options always remain first.
8. Reject fallback results with achieved full orientation error above 60
   degrees. Continue processing candidates until 12 valid results survive or
   the corpus is exhausted.

This makes every displayed fallback actively pursue the requested grasp while
retaining a bounded relaxation for sparse data.

## Collision Semantics

At and after Contact, exempt only the final active wrist joint sphere and the
final wrist-pitch-to-wrist segment from object collision. Do not exempt the
elbow-to-wrist-roll or wrist-roll-to-wrist-pitch segments. The inactive arm,
forearms, upper arms, torso, hips, head, and legs remain collision-tested at
every frame.

Furniture collision has no grasp exemption. Add a 2 cm clearance margin to
torso and limb collision radii for the coverage viewer search, while retaining
the existing default radii for other callers. The HUD continues to classify
these failures as object or environment collisions.

## Alternatives Rejected

- **Exact-only orientation:** geometrically clean but reproduces the observed
  shortage of options.
- **Whole-body/root optimization:** could solve harder orientations but changes
  the recorded approach and is too large for this diagnostic iteration.
- **Orientation-disabled fallback:** preserves coverage but caused the poor
  wrist twists now visible.

## Verification

Add C++ regression tests proving that fallback orientation error decreases,
position and axis gates remain hard, errors above 60 degrees are rejected, and
only the final wrist contact geometry is exempt after Contact. Add viewer tests
for orientation-error ranking and HUD display. Run the established 104-test
Python gate, both trajectory C++ binaries, the release viewer build, and the
same single-process bounded-memory live check. Do not launch mesh or terrain
renderers.

## Out of Scope

- Training or rebuilding the interaction corpus.
- Moving the character root or torso to satisfy a grasp.
- Diffusion stitching or runtime locomotion integration.
- Changing the exact-pose tier thresholds.
