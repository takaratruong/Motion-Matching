# Object-Relative Hand-Trajectory Baseline Design

## Objective

Build a motion-matching baseline that exposes the pickup motions already in the
interaction database. The live flat-terrain G1 skeleton viewer will draw every
recorded hand trajectory that is compatible with a generic object and requested
grasp position or pose. Every displayed Contact pose converges exactly on the
request. Diffusion
will not select or execute a pickup in this baseline; it may later stitch the
locomotion and selected pickup windows after deterministic behavior is measured.

## Considered Visualizations

### Live tolerance-selected, exactly converged trajectories — selected

Gate and rank recorded clips against the current grasp and object geometry, map
their hand paths through the current object's world transform, then apply a
small residual rigid correction so Contact equals the requested grasp. Draw all
clips passing the gates in the live 3D scene. This directly answers which
approaches exist in the data and remains correct when the object moves.

### Offline contact sheet

Render fixed front/side images of every path. This is inexpensive and useful for
reports, but it hides the paths' relationship to the live object and camera.

### Full-body candidate ghosts

Render several complete skeletons along candidate motions. This provides more
context but would clutter the scene, cost substantially more, and make hand-path
coverage harder to read. It can be added after hand-path selection is useful.

## Candidate Ranking

The query contains an object world transform, positive object dimensions,
requested hand, requested grasp world position, and optional grasp world
orientation. Omitting orientation creates a position-only query: orientation is
neither gated nor scored, and convergence preserves the source Contact
orientation after object-frame mapping.

Only clips with the requested active hand and valid Reach, Contact, and Lift
phases are eligible. Each clip must pass explicit configurable tolerances for
grasp position correction, optional grasp orientation correction, and relative
object dimensions. Every passing clip receives a deterministic cost from:

- object-local grasp position error;
- object-local grasp orientation error;
- object dimension error;
- object-local approach-direction error.

The candidates are sorted by total cost and clip index. Every passing candidate
is visualized; there is no top-K truncation. A hard safety limit of 4096 clips
rejects malformed or unexpectedly huge packs rather than silently hiding paths.
This is not locomotion-entry scoring; it intentionally exposes the complete
compatible pickup set before a walking entry pose is chosen.

## Trajectory Reconstruction

For every selected clip, use the source object pose immediately before Contact
as the canonical object frame. Reconstruct the active wrist world pose from
Reach through Lift, transform it into that source object frame, and map it
through the current scene object's world transform. Compute the mapped Contact
hand pose. For a pose query, apply
`requested_grasp_world * inverse(mapped_contact_hand)` to every mapped hand
pose. For a position-only query, apply only the translation from mapped Contact
to the requested position. Contact therefore converges exactly while the
pre-contact path preserves the selected clip's shape.

The best path is green and thicker through repeated line/sphere marks. Remaining
paths use a blue-to-purple rank gradient. Contact points are marked separately.

## Trajectory-Lab Viewer

Use a separate lightweight raylib executable rather than the playable
controller. It loads only the interaction pack, selector, shelf scene, and
trajectory renderer. Locomotion and diffusion therefore cannot influence this
diagnostic.

The viewer exposes live object controls:

- `Q`/`E`: object yaw;
- `R`/`F`: object pitch;
- `Z`/`C`: object roll;
- arrow keys: object planar translation;
- Page Up/Page Down: object height;
- `V`: toggle rejected trajectories;
- Backspace: reset the query.

The grasp is stored in object coordinates and follows every object transform.
The internal query interface also supports an independently supplied grasp world
pose or position for later tests.

## Shelf and Collision Feasibility

Construct a shelf from five oriented boxes: floor, back, left wall, right wall,
and top. The shelf remains fixed while the object moves and rotates inside it.

Each selected source clip stores wrist and elbow samples from Reach through
Lift. Map both through the object frame and the exact Contact residual. Before
Contact, reject a path if its wrist sphere or elbow-to-wrist capsule intersects
the target object's oriented box. At all frames, reject a path if those proxies
intersect any shelf box. Contact and post-Contact object overlap are exempt only
from the object test, never from shelf tests.

Accepted paths are the choices available to later playback. Rejected paths are
not selectable, but can be drawn in faint red for diagnosis. The UI reports
close, collision-safe, object-rejected, and shelf-rejected counts.

## Failure Behavior

Malformed configuration, missing interaction artifacts, zero eligible clips, or
more than 4096 compatible clips must produce a clear startup diagnostic. Invalid
shelf dimensions are rejected. A single malformed source clip is excluded
without changing the deterministic order of valid candidates.

## Verification

- Unit tests cover ranking, deterministic tie-breaking, hand filtering, phase
  bounds, tolerance gates, exact pose convergence, exact position-only
  convergence, moved/rotated target invariance, object collision, shelf
  collision, and Contact exemption boundaries.
- Viewer source tests prove all controls, rejected-path toggle, and absence of
  diffusion/controller authority.
- The viewer is built and launched as exactly one process with no controller,
  robot mesh, terrain mesh, or diffusion artifact.
- Visual success means all compatible paths converge exactly on the requested
  grasp, collision-unsafe paths are excluded, and choices update immediately
  when object position or orientation changes.

## Deterministic Playback Follow-On

After visualization, select one compatible clip using locomotion-entry cost.
Use constrained right-arm/torso IK to absorb the same bounded Contact residual
used by the visualizer, preserve lower-body foot locks, and attach the object at
Contact. IK shapes a selected recorded path; it does not generate or replace the
path.

## Follow-On Experiment

Once a deterministic walking window and pickup candidate are selected, compare
the ordinary inertialized overlap against a diffusion-generated stitch using the
same two endpoint windows. Diffusion must not alter the object-relative contact
segment in that comparison.
