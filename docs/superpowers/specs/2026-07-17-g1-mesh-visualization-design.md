# G1 Mesh Visualization Design

## Goal

Render the real Unitree G1 link geometry on the accepted 31-bone
motion-matching pose so ankle, foot, penetration, sliding, and partial-support
failures are visible. This is a visualization-only checkpoint. It makes no IK,
motion-matching, terrain, contact, or threshold changes.

## Source Asset

Use the already validated rigid-skinned G1 asset at
`/home/ubuntu/projects/g1_mm/g1.fbx` as the conversion source. The source is
11,348,604 bytes and has:

- one armature;
- one rigid-skinned mesh;
- 39 bones;
- 35 vertex groups;
- 196,692 vertices;
- a 0.446 m by 0.363 m by 1.323 m standing bounding box.

The runtime asset will be a Raylib-compatible binary glTF at
`resources/g1_mesh/g1_raylib.glb`. A Blender export script in the same
directory will split rigid link geometry into submeshes below Raylib's 16-bit
index limit while preserving one skin and the exact bone hierarchy. The GLB
and a manifest containing source and output SHA-256 hashes are committed, so a
cluster restart does not depend on the external FBX.

## Bone Mapping

The 30 articulated FBX bones map one-to-one onto the production `G1Bone`
contract:

| Motion bone | GLB bone |
|---|---|
| `G1_Hips` | `pelvis` |
| `G1_LeftHipPitch` | `left_hip_pitch_link` |
| `G1_LeftHipRoll` | `left_hip_roll_link` |
| `G1_LeftHipYaw` | `left_hip_yaw_link` |
| `G1_LeftKnee` | `left_knee_link` |
| `G1_LeftAnkle` | `left_ankle_pitch_link` |
| `G1_LeftToe` | `left_ankle_roll_link` |
| `G1_RightHipPitch` | `right_hip_pitch_link` |
| `G1_RightHipRoll` | `right_hip_roll_link` |
| `G1_RightHipYaw` | `right_hip_yaw_link` |
| `G1_RightKnee` | `right_knee_link` |
| `G1_RightAnkle` | `right_ankle_pitch_link` |
| `G1_RightToe` | `right_ankle_roll_link` |
| `G1_Spine` | `waist_yaw_link` |
| `G1_Spine1` | `waist_roll_link` |
| `G1_Spine2` | `torso_link` |
| `G1_LeftShoulderPitch` | `left_shoulder_pitch_link` |
| `G1_LeftShoulderRoll` | `left_shoulder_roll_link` |
| `G1_LeftShoulderYaw` | `left_shoulder_yaw_link` |
| `G1_LeftElbow` | `left_elbow_link` |
| `G1_LeftWristRoll` | `left_wrist_roll_link` |
| `G1_LeftWristPitch` | `left_wrist_pitch_link` |
| `G1_LeftWrist` | `left_wrist_yaw_link` |
| `G1_RightShoulderPitch` | `right_shoulder_pitch_link` |
| `G1_RightShoulderRoll` | `right_shoulder_roll_link` |
| `G1_RightShoulderYaw` | `right_shoulder_yaw_link` |
| `G1_RightElbow` | `right_elbow_link` |
| `G1_RightWristRoll` | `right_wrist_roll_link` |
| `G1_RightWristPitch` | `right_wrist_pitch_link` |
| `G1_RightWrist` | `right_wrist_yaw_link` |

`G1_Simulation` has no mesh bone. The remaining nine GLB bones are fixed
visual attachments: pelvis contour, pelvis IMU, torso IMU, logo, head, two
sensors, and two rubber hands. Each inherits its authored bind-local transform
from its current animated parent. No hand-authored world offset is allowed.

Loading fails before the frame loop if a mapped bone is missing, duplicated,
has the wrong mapped parent, or if the asset contains an unsupported mesh or
skin layout.

## Pose Transfer

Raylib's skin updater consumes global bone transforms. Once per accepted
frame:

1. Mapped articulated bones receive the exact position and quaternion from
   `accepted_state.ik_global_bone_positions` and
   `accepted_state.ik_global_bone_rotations`.
2. Fixed attachment bones receive `animated_parent_global * bind_local`.
3. Every scale remains `(1, 1, 1)`.
4. The completed one-frame pose is passed to `UpdateModelAnimation` and the
   model is drawn with an identity model transform.

The converted GLB and the motion database are both Y-up, metre-scale G1 link
frames. No empirical scale, axis, ankle, or root offset is permitted. Any
alignment discrepancy must remain visible and be fixed at its source rather
than hidden with a rendering correction.

## Runtime Component

`g1_mesh_renderer.h` owns the model, one-frame animation buffer, exact mapping,
bind-local fixed transforms, load validation, pose update, draw, and unload.
The controller only supplies the accepted global pose and chooses which
diagnostic layers to draw.

The mesh is loaded once after the Raylib window opens and unloaded once before
the window closes. Its lifecycle is separate from terrain scene-model
load/unload counters, so scene-switch and CSV generation semantics do not
change. A mesh-load failure exits through normal cleanup with a controlled
error.

## Diagnostic Rendering and Controls

The mesh is enabled by default. The existing stick skeleton remains available
as an overlay rather than being removed.

- `M`: toggle the G1 mesh.
- `B`: toggle the bone/skeleton overlay.
- `P`: toggle physical sole proxies.

The sole-proxy layer draws the four production `sole_points_local` values from
each `G1LegConfig` in the accepted `G1_LeftToe` or `G1_RightToe` frame. Proxy
points and their perimeter are green for recorded contact and orange for
swing. These are the same physical points used by clearance logic, so a gap
between them and the rendered sole exposes a mesh/bone/proxy alignment error.

The 2D overlay states `G1 MESH`, `BONES`, and `SOLE PROXIES` with an on/off
value and lists the three keys. Defaults are mesh on, bones on, and sole
proxies on for the first diagnostic launch.

## Tests and Acceptance

Implementation follows test-first development.

Automated tests must prove:

- all 30 exact names map once and `G1_Simulation` does not map;
- mapped parents match the production skeleton and GLB hierarchy;
- fixed attachments preserve their bind-local transform under translated and
  rotated parent poses;
- identity/rest pose transfer reproduces the GLB bind pose within 0.5 mm and
  quaternion sign-equivalent tolerance;
- malformed bone counts, names, parents, and non-finite poses fail closed;
- the exported asset has one skin, no submesh over 65,535 vertices, rigid
  weights, a 0.8 m to 1.6 m bounding extent, and the manifest hashes match;
- toggles affect only rendering and never accepted simulation state;
- mesh resources unload exactly once on normal and controlled-error exits.

The finite runtime gate uses the existing exact-25-Hz mixed-multilevel terrain
route and then a live interactive launch. It passes when the GLB loads without
warnings, all accepted poses remain finite, terrain behavior and logs are
unchanged, the mesh follows the skeleton without frame lag, and the sole
proxies visually align with the bottoms of both feet. The user then inspects
stairs, ramps, tangential approaches, sideways travel, and half-support edges
in the new visualizer.

## Global Constraints

- Exact float32 `0.04` second simulation cadence remains unchanged.
- Terrain assets and motion data remain separate.
- Heading and travel direction remain independent.
- No IK or motion-matching behavior is changed in this checkpoint.
- The old running visualizer is not inspected, signaled, or terminated; the
  mesh build launches as a separate new window after its finite gate.
- Every coherent test-backed checkpoint is committed and pushed immediately
  to `checkpoint/g1-footprint-task6`.
