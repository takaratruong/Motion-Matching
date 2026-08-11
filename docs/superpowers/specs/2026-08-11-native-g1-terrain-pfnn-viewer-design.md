# Native-G1 Terrain PFNN Viewer Design

Date: 2026-08-11

Status: approved in conversation

## Objective

Show the trained native-G1 terrain PFNN on the authenticated ramp and stair
scenes without changing PFNN inference.  Terrain-conditioned motion, phase,
root motion, root height/orientation, contacts, and all 29 G1 joints remain
outputs of `TerrainPFNNRuntime`.  MuJoCo is only a renderer.

## Non-negotiable boundary

- Do not use `HybridMatcher` in this viewer path.
- Do not use `ExistingUtilityPosePostprocessor`, pose repair, foot locking,
  command-owned root placement, support-plus-source-clearance placement, or a
  second inertializer.
- Do not flatten the PFNN root quaternion or blend PFNN joints in the viewer.
- Do not use the viewer's `hold_idle_pose` bypass; neutral input is evaluated by
  PFNN so the trained recurrent model owns deceleration and the neutral pose.
- Do not modify `terrain_pfnn/runtime.py`, the checkpoint, its normalization,
  recurrent state, phase update, trajectory planner, or output decoding.
- Terrain samples supplied to PFNN and the mesh rendered by MuJoCo must come
  from the same authenticated G1HF scene.
- The only permitted transformation is one rigid coordinate-frame mapping
  between PFNN's local course frame and the scene's native MuJoCo frame.

## Existing authorities

- Runtime: `mm_sonic.terrain_pfnn.runtime.TerrainPFNNRuntime` with
  `command_driven_root=False` and `hold_idle_pose=False`.
- Model: the accepted native-G1 rollout checkpoint
  `model-mixed-filtered-rollout16-final-v2/best.pt`.
- Dataset: `mixed-corpus-filtered/manifest.json`, which contains flat,
  `WalkingUpSteps*`, and `terrain_slopes__*` training sources.
- Scene loading: existing `load_scene_terrain` and `SceneTerrainAdapter`.
- Skeleton conversion: existing `isaaclab_to_mujoco_joint_vector`.

## Viewer adapter

Extend `terrain_pfnn_viewer.py` with an optional `--scene` and
`--terrain-root`.  When `--scene` is present, load the existing G1HF scene and
construct a callback that:

1. maps PFNN course XY into scene-native XY using the scene spawn and the
   established native forward vector `(sin(heading), -cos(heading))`, so PFNN
   local +X is character-forward;
2. samples height and finite, piecewise height gradient from that exact scene;
3. maps the gradient back into PFNN course axes; and
4. exposes the unchanged scene mesh for rendering.

The corresponding output placement applies the inverse rigid-frame convention
to PFNN root XY and premultiplies its complete root quaternion by the scene yaw.
Root Z and all tilt from PFNN are preserved exactly.  Joint output is reordered
once into MuJoCo order and written without interpolation.

The existing `--terrain-fit` course remains available when `--scene` is absent.
Supplying both is rejected so the terrain authority cannot be ambiguous.

## Acceptance

- Unit tests prove the terrain callback and root placement use one rigid frame.
- Unit tests prove root roll/pitch/yaw and every emitted joint reach MuJoCo
  exactly, with no display blend.
- Source inspection/tests prove the scene path constructs no hybrid matcher or
  display postprocessor.
- A bounded headless ramp and stair run uses the frozen checkpoint and reports
  finite model frames, model-owned root motion, phase advancement, and no hold.
- The visible viewer is launched on the ramp for operator inspection.
