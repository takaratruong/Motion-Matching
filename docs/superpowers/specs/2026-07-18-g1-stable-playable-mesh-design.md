# Stable G1 Playable Mesh Visualizer Design

## Goal

Provide the user-facing terrain motion-matching visualizer they already found
stable and familiar, with the corrected 35-part Unitree G1 mesh added to the
final displayed pose. The result must retain the old Raygui interface, live
keyboard/gamepad controls, terrain scenes, and the certified independent
travel-versus-heading behavior, including forward, backward, sideways, and
diagonal locomotion.

This is a bounded visualization port. It does not merge the newer
transactional runtime, unfinished best-effort IK work, or bounded-candidate
experiments into the playable build.

## Considered Approaches

### Selected: stable controller plus corrected mesh renderer

Use the controller behavior from commit `8f7ead2`, the last stable old-UI
checkpoint before transactional-runtime integration, and add the current
corrected G1 renderer and packaged mesh assets. This preserves known-good live
control behavior while limiting new runtime code to mesh lifecycle, pose
transfer, drawing, and display toggles.

### Rejected: rebuild the old UI around the transactional controller

This would retain newer internals but would require diagnosing the live-input
transaction failure and re-porting the large Raygui surface. It mixes an
unrelated runtime repair into a visualization checkpoint and has a larger
regression surface.

### Rejected: maintain both controllers in one source tree

Copying the old controller to a second large source file would be quick, but it
would duplicate roughly 150 KB of control, terrain, and matching code. The two
playable runtimes would drift and make later fixes ambiguous.

## Version-Control Isolation

Create a dedicated playable branch and worktree. Preserve the current
`g1-footprint-task6` branch unchanged as the transactional/IK research line.
The playable branch uses the stable `8f7ead2` controller behavior and imports
only the certified mesh renderer, corrected asset, asset manifest, and tests
needed by this visualizer.

Every coherent, verified checkpoint is committed and pushed. A cluster restart
must not lose either the stable playable result or the existing research line.

## Preserved Runtime Behavior

The stable controller remains authoritative for:

- the classic Raygui layout and diagnostics;
- keyboard and gamepad input;
- orbit camera behavior;
- terrain scene selection and rendering;
- motion matching and database transitions;
- exact 25 Hz simulation cadence;
- multi-level terrain, stairs, blocks, ramps, slopes, and wide playable areas;
- independent desired travel and desired heading commands;
- forward, backward, lateral, and diagonal command prediction;
- its existing terrain retargeting, contact, and foot-processing behavior.

No cost weights, search cadence, thresholds, pose-selection policy, terrain
sampling, foot-contact logic, or IK math changes in this checkpoint.

## Mesh Integration

Reuse `g1_mesh_renderer.h` and the corrected
`resources/g1_mesh/g1_raylib.glb` package. The corrected asset must be validated
as 35 mesh parts before the interactive runtime is launched.

The controller loads the mesh once after the Raylib window is initialized and
unloads it exactly once during normal or controlled-error cleanup. Asset or
mapping failure produces a visible diagnostic and exits cleanly; it must not
leave a partially initialized live runtime.

On every rendered frame, the renderer receives the controller's final
`global_bone_positions` and `global_bone_rotations` after terrain adjustment
and the stable runtime's existing foot processing. This makes the mesh show
the same final pose as the skeleton, without frame lag or a parallel pose path.
The renderer applies no empirical root, axis, scale, ankle, or sole correction.

The G1 mesh is enabled by default. `M` toggles it. The old skeleton and debug
layers remain available so mesh-to-skeleton, foot, sliding, and penetration
failures remain inspectable. Existing control bindings and UI interaction take
priority; any new diagnostic toggle must not replace them.

## Failure Handling

- Invalid or non-finite pose input fails closed before model animation update.
- Mesh load or bone-map failure reports the exact cause and follows normal
  Raylib cleanup.
- Mesh rendering never mutates matching, terrain, controller, or accepted-pose
  state.
- Missing mesh resources do not silently fall back to a misleading partial
  robot.
- The interactive build must remain alive when movement, strafe, diagonal, and
  camera inputs begin.

## Verification and Acceptance

Implementation is test-driven. Existing mesh asset and renderer tests are
ported with the renderer. Additional focused checks verify integration without
altering stable controller behavior.

The automated and finite-runtime gates must prove:

1. The old Raygui labels and control help remain present in the built binary.
2. The certified independent travel/heading and lateral/diagonal command paths
   remain present.
3. The corrected GLB loads all 35 mesh parts and its manifest hashes match.
4. The renderer consumes the final global pose and rejects non-finite input.
5. Mesh toggling changes rendering only.
6. A deterministic 32-frame mixed-multilevel route completes with finite
   poses and valid transitions.
7. Scripted forward, backward, lateral, diagonal, strafe, and camera input does
   not crash the live runtime.
8. Cleanup unloads mesh resources exactly once.

After those gates pass, launch one new interactive visualizer for user review.
It must visibly use the classic UI and corrected G1 mesh. The user will inspect
sideways and diagonal travel on level terrain, stairs, ramps, obstacle edges,
and partial-support approaches. This checkpoint demonstrates the already
implemented locomotion behavior; it does not claim IK or sliding quality has
improved merely because a mesh is present.

## Out of Scope

- Completing or merging best-effort IK.
- Fixing residual foot sliding or terrain-contact quality.
- Changing motion data or synthesizing terrain-specific clips.
- Merging terrain geometry with motion data.
- Repairing the transactional live-input path.
- Replacing the classic UI.
