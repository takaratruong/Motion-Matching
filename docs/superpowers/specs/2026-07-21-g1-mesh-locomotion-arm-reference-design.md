# G1 Mesh Locomotion Arm Reference Fix

## Problem

The 23-bone locomotion controller publishes hands below the torso while walking,
but the 31-bone G1 mesh keeps an interaction-like raised-arm posture. The mesh
expansion currently uses frame zero of the pickup database as its calibrated G1
reference. That frame has the right wrist 0.148 m above and 0.397 m forward of
the torso, whereas the flat locomotion reference has both hands 0.36--0.38 m
below the upper torso. Unmapped G1 shoulder-roll and wrist joints therefore
inherit a pickup posture during locomotion.

## Design

Keep the interaction handoff reference unchanged because it owns runtime pose
continuity. Create a separate mesh-only G1 reference after the flat locomotion
reference is known. Align the mapped G1 joint world rotations to the flat
locomotion reference, preserve the original G1 local translations and hand
geometry, and zero dynamic channels. Each rendered frame then applies live flat
world-rotation deltas to this mesh-only reference before final mesh update.

This is preferred over selecting another arbitrary pickup frame, which remains
data-order dependent, and over feeding human joint positions directly to the G1
mesh, which would distort robot link lengths.

## Verification

- A focused C++ regression proves calibration preserves all non-root G1 local
  translations and aligns mapped arm world rotations to the flat reference.
- The controller integration test proves mesh expansion uses the mesh-only
  reference, while interaction expansion continues using the interaction
  reference.
- Existing adapter and mesh tests pass, the controller builds, and the live
  window is relaunched without external capture tools.

## Non-goals

The fix does not change diffusion, motion matching, interaction ownership,
pickup timing, DCV settings, or object-height behavior.
