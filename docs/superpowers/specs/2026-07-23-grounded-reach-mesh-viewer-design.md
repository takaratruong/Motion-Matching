# Grounded Reach Mesh Viewer Design

## Goal

Render the certified articulated Unitree G1 mesh in the grounded exhaustive
reach-coverage viewer, lower the main tabletop to `0.65 m`, and make the camera
freely inspectable without reintroducing the terrain, controller, physics, or
an alternate pose path.

## Chosen Architecture

Reuse `g1_mesh_renderer.h` and `resources/g1_mesh/g1_raylib.glb`. The selected
motion sample already produces a canonical 31-bone `interaction::WorldPose`.
Its positions and rotations are the sole input to both renderers:

```text
selected grounded Pose
        |
        v
interaction::world_pose
        |
        +--> g1_mesh_renderer_update --> articulated G1 mesh
        |
        +--> draw_pose              --> diagnostic blue skeleton
```

There is no mesh-specific retarget, root offset, scale calibration, locomotion
reference, or pose expansion. This shared-world-pose rule prevents the earlier
mesh/skeleton size and arm-pose divergence. The grounded placement and active
arm IK code remain unchanged.

The rejected alternatives are:

- a root-attached static mesh, because it cannot display articulated reaches;
- a second mesh adaptation layer, because it can diverge from the skeleton and
  previously produced incorrect arms, scale, and apparent swimming; and
- the controller/terrain viewer, because it adds unrelated state and resource
  load to a coverage visualization.

## Runtime Behavior

- Initialize the Raylib window before loading the mesh.
- Load `resources/g1_mesh/g1_raylib.glb` once.
- Show the mesh by default.
- Hide the blue skeleton by default.
- `M` toggles the mesh and `B` toggles the skeleton overlay independently.
- For each animated selected sample, compute one `WorldPose`, update the mesh
  from its exact 31 positions and rotations, then optionally draw the skeleton
  from that same pose.
- Keep accepted paths, rejected-path inspection, grasp axes, furniture,
  collision labels, and explicit Enter search unchanged.
- If mesh load or update validation fails, disable the mesh, enable the blue
  skeleton, and show the renderer error in the HUD. Do not terminate the DCV
  session or fall back to another viewer.
- Unload the mesh before closing the graphics context on normal or exceptional
  shutdown.

## Scene Scale

The main table dimensions remain `1.20 x 0.06 x 0.75 m`, but its center moves
to Y `0.62 m`, giving an exact top surface of `0.65 m`. Object reset height is
derived from this tabletop rather than retaining the old `0.77 m` literal.
The shelf and lower side table continue to derive from
`make_coverage_environment`, so their collision geometry and rendering remain
identical.

The real coverage probe uses the same lowered table transform and derives
table/shelf/lower-table object centers from the generated geometry. This keeps
reported collision coverage consistent with what the viewer displays.

## Camera Controls

Camera movement is visual-only and never changes the object, query, search, or
animation state.

- Hold left mouse and drag to orbit around the current camera target.
- Hold middle mouse and drag to pan in the camera view plane.
- Use the mouse wheel to zoom with bounded distance so the camera cannot cross
  through its target or escape to an unusable range.
- Preserve all existing keyboard object controls.
- Initialize the camera to frame the G1, lowered table, shelf, and lower side
  table.

## Verification

- Extend the viewer source-contract test first to require mesh load, same-pose
  update-before-draw ordering, `M`/`B` toggles, fallback diagnostics, unload,
  camera orbit/pan/zoom inputs, and the `0.65 m` tabletop.
- Remove the obsolete assertion forbidding `g1_mesh_renderer` and `LoadModel`.
- Run the existing `test_g1_mesh_renderer` unit test against the certified GLB.
- Build the viewer warning-clean and run all exhaustive-search, coverage,
  collision, and unchanged-IK gates.
- Run the real lowered-table 4,608-candidate probe and retain truthful zero-
  coverage fixtures rather than weakening collision or IK gates.
- Launch exactly one viewer and verify bounded startup RSS. Do not launch mesh,
  terrain, controller, or screenshot side processes.

## Non-Goals

This change does not modify motion placement, root grounding, IK, collision
geometry semantics, motion ranking, the reach corpus, diffusion, physics, or
controller behavior.
