# Diffusion Pickup G1 Mesh Design

## Goal

Render the existing articulated G1 robot mesh in the diffusion pickup visualizer so generated approach and grasp motions can be judged on the robot rather than only on the debug skeleton.

## Chosen approach

Port the proven terrain-aware renderer without changing its pose mathematics. Reuse `g1_mesh_renderer.h` and `resources/g1_mesh/g1_raylib.glb`, bind GLB joint names to the existing 31-bone G1 hierarchy, update the mesh from the controller's accepted global bone positions and rotations each frame, and draw it in the existing 3D pass. Keep the renderer source-compatible with both the terrain visualizer's Raylib 5 model layout and this branch's pinned Raylib 6 skeleton/keyframe layout.

This is preferred over the legacy `character.bin` skin because that asset uses a different 23-bone hierarchy, and over loading individual STL links because the terrain-aware GLB already has validated joint bindings and runtime tests.

## Runtime behavior

- Load the GLB once after the window and rendering context exist.
- Treat a missing or invalid mesh as a startup error with a useful diagnostic.
- Update the mesh from the final accepted pose each frame, after diffusion/follower pose changes and before drawing.
- Show the robot mesh by default.
- Keep `M` as the mesh visibility toggle.
- Hide the debug G1 bone overlay by default while preserving its existing toggle/control for diagnostics.
- Unload the model before closing the rendering context.

## Assets and source ownership

Copy the renderer header, its certified `g1_kinematic_contract.h` bone-enum dependency, the GLB, manifest, and exporter/validator support files from the terrain-aware branch. This branch's existing `g1_skeleton.h` has the same 31 indices, parents, and skeleton signature; the compatibility contract lets the renderer remain byte-for-byte certified. The diffusion branch owns a local copy so the visualizer remains runnable from its own worktree and does not depend on paths in another checkout.

## Verification

- Port the terrain-aware renderer unit test and run it first against the integration.
- Build the controller from a clean target.
- Launch the diffusion pickup visualizer from the worktree root and confirm the GLB loads.
- Confirm walking, generated approach, pickup, carry, and placement poses all drive the same mesh continuously.
- Confirm `M` hides and restores the mesh and shutdown unloads it cleanly.

## Non-goals

This change does not retrain diffusion, alter approach selection, change physics, optimize generation latency, or modify grasp attachment behavior.
