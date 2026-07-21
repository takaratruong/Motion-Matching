# Object-Relative Hand-Trajectory Baseline Design

## Objective

Build a motion-matching baseline that exposes the pickup motions already in the
interaction database. The live flat-terrain G1 skeleton viewer will draw the
closest recorded right-hand trajectories around the current object. Diffusion
will not select or execute a pickup in this baseline; it may later stitch the
locomotion and selected pickup windows after deterministic behavior is measured.

## Considered Visualizations

### Live top-K object-relative trajectories — selected

Rank compatible recorded clips against the current grasp and object geometry,
map their hand paths through the current object's world transform, and draw them
in the live 3D scene. This directly answers which approaches exist in the data
and remains correct when the object moves.

### Offline contact sheet

Render fixed front/side images of every path. This is inexpensive and useful for
reports, but it hides the paths' relationship to the live object and camera.

### Full-body candidate ghosts

Render several complete skeletons along candidate motions. This provides more
context but would clutter the scene, cost substantially more, and make hand-path
coverage harder to read. It can be added after hand-path selection is useful.

## Candidate Ranking

Only clips with the requested active hand and valid Reach, Contact, and Lift
phases are eligible. Each clip receives a deterministic object/grasp cost from:

- object-local grasp position error;
- object-local grasp orientation error;
- object dimension error;
- object-local approach-direction error.

The candidates are sorted by total cost and clip index, then truncated to a
configurable top-K value. The default live visualization count is 24. This is
not locomotion-entry scoring; it intentionally shows the pickup motions closest
to the object/grasp before a walking entry pose is chosen.

## Trajectory Reconstruction

For every selected clip, use the source object pose immediately before Contact
as the canonical object frame. Reconstruct the active wrist world pose from
Reach through Lift, transform it into that source object frame, and map it
through the current scene object's world transform. This preserves height and
allows the visualization to follow a moved or rotated object.

The best path is green and thicker through repeated line/sphere marks. Remaining
paths use a blue-to-purple rank gradient. Contact points are marked separately.

## Runtime Integration

Add a focused, renderer-independent trajectory-ranking module. The controller
loads and ranks paths when `MM_INTERACTION_HAND_TRAJECTORIES` is a positive
integer and rebuilds world-space paths whenever the target generation or pose
changes. Rendering consumes only the resulting point arrays.

The baseline launch must omit `MM_G1_OFFLINE_OVERLAP` and any diffusion worker
variables. Existing motion matching and interaction playback remain the sole
motion authority. Flat terrain and the blue G1 skeleton remain enabled.

## Failure Behavior

Malformed configuration, missing interaction artifacts, or zero eligible clips
must produce a clear startup diagnostic. A single malformed source clip is
excluded without changing the deterministic order of valid candidates.

## Verification

- Unit tests cover ranking, deterministic tie-breaking, hand filtering, phase
  bounds, object-frame mapping, and moved/rotated target invariance.
- Controller source tests prove the environment gate, default top-K, and absence
  of diffusion authority in the baseline launch.
- The controller is built and launched with exactly one process, flat terrain,
  skeleton rendering, and no offline overlap artifact.
- Visual success means several distinct paths converge on the live object and
  remain attached to it when its transform changes.

## Follow-On Experiment

Once a deterministic walking window and pickup candidate are selected, compare
the ordinary inertialized overlap against a diffusion-generated stitch using the
same two endpoint windows. Diffusion must not alter the object-relative contact
segment in that comparison.
