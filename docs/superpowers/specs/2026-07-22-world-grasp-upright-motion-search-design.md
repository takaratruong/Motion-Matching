# World-Grasp Upright Motion Search Design

## Objective

Correct the trajectory baseline so the requested grasp is the search query.
Rotating or moving the object must re-run candidate search; it must never apply
object pitch or roll to a person's full-body kinematics.

## Confirmed Root Cause

The current selector compares source and target grasps in object-local space.
When the grasp is attached to the object, object rotation cancels from that
comparison. The viewer then applies
`contact_residual * target_object * inverse(source_object)` to every body joint,
which incorrectly rolls or pitches the person.

## Candidate Alignment and Search

Use the established `interaction_matcher.cpp::scene_alignment` convention for
every source clip:

- extract only the yaw difference between source and target objects;
- translate the source scene only in world X/Z so the source object and target
  object share a planar location;
- preserve world Y and world-up, so the candidate's feet and body remain
  upright;
- never apply target object pitch or roll to the root, torso, or legs.

After this upright alignment, reconstruct the recorded Contact wrist pose and
compare it directly with the requested world-space grasp. Full-pose queries use
both position and orientation. Position-only queries omit orientation. Rank by
normalized Contact position/orientation error plus object-dimension error, with
stable clip-index tie breaking.

Use the runtime matcher correction limits as validity gates:

- maximum Contact position correction: 0.12 m;
- maximum Contact orientation correction: 0.436332313 radians (25 degrees).

Object movement or rotation rebuilds this search every rendered frame. A clip
may enter, leave, or move within the sorted valid set. If no recorded motion is
within the gates, report zero valid motions instead of rotating a body.

## Arm IK Shaping

For each candidate, map the source pose through upright yaw/XZ alignment. Build
an arm target that smoothly applies the Contact residual from zero at Reach
entry to one at Contact and remains at one through Lift. Feed this target to the
existing seven-joint `solve_hand_ik` solver. IK may change only the selected arm
chain; lower body, root, and non-active arm retain recorded motion.

At Contact, the requested target passed to IK is exactly the requested grasp.
A candidate is valid only if Contact IK is accepted under the same 0.12 m and
25-degree request limits and the shaped wrist/forearm path passes object/table
collision checks. Collision and IK failures are excluded from selectable motion
options and counted separately in the viewer.

## Viewer Behavior

The viewer re-runs selection, IK shaping, and collision classification while an
object-control key is held. Bracket keys cycle the current sorted valid set.
The highlighted hand path and animated G1 skeleton come from the IK-shaped pose,
so the wrist follows the selected path while the body stays upright. Object
pitch/roll affects the requested grasp, candidate ranking, and IK feasibility,
not the body's world orientation.

The recorded table, W/S height controls, full-pose/position-only toggle, and
standalone no-controller/no-diffusion/no-mesh architecture remain unchanged.

## Verification

- A regression test rolls the target object and proves the mapped skeleton root
  up direction remains world-up.
- A selection test supplies clips with distinct Contact wrist orientations and
  proves object roll changes the selected/ranked clip set.
- Tests prove candidates beyond 0.12 m or 25 degrees are excluded.
- IK tests prove Contact converges within solver acceptance while non-arm root
  and leg transforms remain unchanged.
- Live audit verifies one corrected viewer, no diffusion variables, and no
  runtime errors; no screenshot or DCV capture is used.
