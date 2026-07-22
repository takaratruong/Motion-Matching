# Grasp-Anchored Live Motion Search Design

## Goal

Make the trajectory viewer search recorded pickup motions from the requested
world-space grasp, update that search while the grasp is manipulated, and show
only complete, collision-safe pickup options whose bodies remain upright.

## Search Query and Alignment

The requested world-space grasp pose is the retrieval anchor. The target
object's identity, dimensions, pitch, and roll do not affect retrieval cost or
whole-body alignment. Object geometry remains available only for collision
testing and for deriving the viewer's requested grasp pose.

For each recorded clip, reconstruct the actual Contact wrist pose. Compute a
yaw-only rotation that aligns the recorded Contact wrist heading with the
requested grasp heading, then compute X/Z translation that aligns the rotated
Contact wrist with the requested grasp position. Do not translate the body in
Y and do not apply target grasp pitch or roll to the body. This preserves world
up and the recorded body-to-hand relationship.

After this grasp alignment, rank candidates by the remaining Contact height
and orientation residual. In position-only mode, requested grasp orientation
is still retained for scene heading, but orientation is omitted from ranking
and arm correction. Object dimensions are not a search gate or cost term.

## Live Search and Controls

Every object translation or rotation changes the requested grasp and performs
the complete pipeline in the same update:

1. Search every compatible recorded hand clip.
2. Rank clips from the current world-space grasp.
3. Shape each retained clip with active-arm IK.
4. Reject object or table collisions.
5. Select rank zero from the new valid set.

The viewer must not preserve a manually selected clip after the grasp changes,
because doing so hides reranking. `Enter` forces the same complete refresh for
diagnostics. `/` and `]` cycle forward through the current valid result set;
`[` cycles backward. Cycling never selects rejected clips. Empty result sets
remain interactive and display `0 valid motions`.

## Motion Extent and IK

Each option begins at the first Reach frame and continues through the final
contiguous Lift frame. Contact remains the first Contact frame. This displays
the complete Reach, Contact, and Lift sequence, whose median duration in the
current 2,045-clip pack is approximately 3.6 seconds.

The grasp-aligned body pose is the IK base. A smooth correction weight grows
from zero at Reach entry to one at Contact and stays at one through Lift. Only
the active seven-joint arm may change. At Contact, the requested IK target is
the exact world-space grasp. The root, torso, opposite arm, and legs retain
their recorded local transforms after yaw/XZ alignment.

## Collision Filtering

Collision filtering consumes the IK-shaped poses, not only the wrist path.
Represent the skeleton using conservative spheres at joints and capsules along
parent-child bones. Test the complete body against every table box at every
sample. Test it against the target object before Contact and continue testing
all non-grasping body segments afterward. The active grasping hand/forearm may
overlap the target object at and after Contact, but this exemption never
applies to the table.

Rejected clips remain optionally visible in red for diagnosis, but they are
not selectable. Rejection counters distinguish IK, object, and table failures.

## Verification

Automated regressions must prove:

- object identity and dimensions do not change grasp-based ranking;
- rotating the requested grasp reranks recorded Contact wrist orientations;
- grasp alignment preserves world up and Contact X/Z/heading;
- extraction reaches the final Lift frame rather than its first frame;
- full-skeleton upper-arm or torso table penetration is rejected;
- intended hand/object overlap is allowed only at and after Contact;
- `/`, `]`, `[`, and `Enter` are bound;
- every transform change calls the complete search pipeline and rank resets;
- zero valid results do not index the result vectors;
- the standalone viewer remains independent of mesh, terrain, diffusion, and
  controller code.

The replacement viewer is launched only after focused C++ tests, viewer source
contract tests, and a release build pass.
