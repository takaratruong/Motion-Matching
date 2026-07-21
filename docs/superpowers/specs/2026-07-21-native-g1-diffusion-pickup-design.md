# Native G1 Diffusion Pickup Design

## Goal

Run locomotion, diffusion-guided approach, pickup, carry, placement, debug
drawing, and mesh rendering on one native 31-bone G1 pose. The 23-bone
LAFAN/Holden rig must not be an intermediate runtime representation.

## Why the current architecture is rejected

The current integration starts with a 23-bone LAFAN locomotion pose, collapses
31-bone interaction poses into that rig, and expands them back into G1 for the
mesh. The bridge cannot preserve G1's intermediate hip, shoulder, and wrist
joints or its segment lengths. Consequently the authoritative LAFAN skeleton
can reach the object while the rendered robot does not.

Further visual retarget patches are out of scope. They cannot establish that
the robot itself reaches the grasp.

## Chosen architecture

Use the terrain-aware native G1 runtime as the base and port the diffusion
pickup system into it:

```text
native G1 motion matching
        |
        v
31-bone G1 locomotion snapshot
        |
        +--> diffusion object-relative approach planner
        |
        v
31-bone G1 interaction runtime and hand constraint
        |
        v
one final 31-bone G1 pose
        +--> G1 mesh
        +--> G1 debug skeleton
        +--> grasp/object attachment
```

The implementation will use an isolated worktree based on the native G1
terrain branch. The existing dirty terrain workspace is read-only source
material and must not be reset, cleaned, or overwritten. Any required
uncommitted terrain deltas will be identified explicitly before being copied.

## Runtime behavior

- Ordinary locomotion comes directly from the G1 terrain database and its
  31-bone controller state.
- Pressing `F` immediately freezes the current G1 locomotion snapshot and starts
  the existing object-relative diffusion approach computation.
- The approach follower controls the G1 root while the native G1 pose continues
  to animate; it does not stop at a separate three-metre waypoint.
- At interaction ownership, the G1 interaction runtime publishes its native
  pose directly. No `FlatControllerPose`, collapse, or expansion is allowed in
  the native controller path.
- Handoff blends are 31-bone G1-to-G1 transitions. The active hand constraint is
  evaluated against the same final pose sent to the renderer.
- Carry, placement, and re-pick use the existing object state machine and
  provenance rules.

## Rendering and diagnostics

- The G1 mesh consumes final G1 global positions and rotations exactly as the
  terrain-aware renderer already does.
- The optional blue skeleton uses the same final 31-bone global pose and G1
  parent table, so it must coincide with the mesh joints.
- LAFAN rendering is removed from the playable native-G1 build. It may remain in
  offline comparison tools only.
- Mesh rendering remains visual-only: it cannot mutate motion, interaction, or
  object state.

## Porting boundary

Port only the interaction components needed by the playable path:

- interaction database/features and target registries;
- diffusion funnel artifact, worker, planner, and follower;
- smart-pickup request/state logic;
- G1 interaction playback, IK, attachment, carry, and placement;
- object/table/rack scene rendering and `F` input wiring.

Do not port the flat controller adapter or its 23-to-31 rendering bridge.

## Failure handling

- Missing or invalid diffusion artifacts disable learned pickup with an explicit
  diagnostic; native G1 locomotion remains usable.
- A failed or stale plan returns control to native G1 locomotion without moving
  the object.
- Invalid G1 poses, target generations, or hand constraints fail before object
  attachment.
- Only one visualizer process is launched, and no screen-capture or X11 image
  tool is used.

## Verification

Automated gates must prove:

1. The playable controller uses a 31-bone G1 database and parent hierarchy.
2. The native controller source contains no flat pose collapse/expansion path.
3. Locomotion-to-interaction and interaction-to-locomotion handoffs remain
   finite and continuous.
4. At full grasp weight, the final rendered G1 wrist is within the certified
   tolerance of the object grasp transform.
5. Mesh and debug-skeleton joint positions are identical inputs.
6. Pickup, carry, placement, and pickup-after-placement remain valid.
7. The focused diffusion and native terrain regression suites pass.

Manual acceptance is performed in the live G1 visualizer: walk, press `F` from
several entry positions, observe continuous approach and pickup, place the
object, and pick it up again. The robot mesh—not a separate LAFAN skeleton—is
the acceptance visual.

## Deferred work

Training-speed improvements, broader object conditioning, new locomotion data,
and leg-retarget tuning are not part of this pivot. The first objective is a
truthful native-G1 end-to-end pickup path using the artifacts already available.

