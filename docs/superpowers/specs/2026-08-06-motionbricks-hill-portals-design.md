# MotionBricks GRAIL Hill Portals Design

- **Status:** Architecture approved; written design pending final review
- **Date:** 2026-08-06
- **Target:** Kinematic Unitree G1 viewer with W/A/S/D control

## Goal

Replace the hill viewer's runtime root conditioning and foot IK with the terrain
portal architecture already used by the
`research/g1-global-terrain-portals-abrupt` branch.

MotionBricks remains responsible for responsive flat-ground locomotion. Near
the hill, the runtime selects and commits to an offline-audited, full-body
GRAIL traversal of the exact hill mesh. On the landing, the final four authored
poses seed a phase-matched MotionBricks continuation and W/A/S/D control
resumes.

The first milestone is deliberately narrow:

- one small hill built from the local GRAIL slope corpus;
- approximately 18 degrees;
- purely kinematic playback;
- no runtime leg IK or world-space foot locks;
- a complete route in both travel directions; and
- the existing local MuJoCo viewer and keyboard controls.

## Why the Current Prototype Fails

The current prototype starts with a flat-ground MotionBricks pose and tries to
repair it after generation. Root-height projection can prevent gross
penetration, but leg IK must then invent the missing uphill biomechanics:
pelvis motion, knee flexion, trunk compensation, stride length, contact timing,
and swing clearance.

The authored-contact revision fixed stale contact detection, but it did not fix
the larger mismatch. A world-space stance lock applied to a flat walking clip
can preserve the wrong foothold for the wrong duration. The visible result is
the foot dragging and the leg stretching against a root trajectory that was
never authored for the slope.

The reference branch does not solve that mismatch with better online IK. It
avoids it by playing an already valid full-body terrain motion across the
non-flat interval.

## Selected Source Terrain

The local GRAIL corpus contains 1,880 slope motions paired one-to-one with exact
USD terrain assets. The first hill uses:

```text
/home/ubuntu/datasets/GRAIL/data/slope/object_usd/
terrain_slopes__slope_113__NNN.usd
```

Its single ramp grade is `18.898 degrees`, with no steeper hidden section. This
is the closest uniform local ramp to the requested 18-degree hill. Its ten
paired robot recordings are 250 frames at 25 Hz.

The exact source ramp rises about `0.232 m` over `0.678 m`. The canary map
places two copies back-to-back with a short flat summit: one in its authored
orientation and one rotated 180 degrees. The result is a small symmetric mound
with the exact GRAIL grade on both flanks and low flat ground at both ends.

The source recordings contain both straight ascents and turn-around descents.
For example, variant `004` has a clean forward ascent and variant `007` has a
clean authored descent after its crest turn. The builder searches compatible
windows from all ten recordings, registers an ascent and a rotated descent to
the two flanks, phase-stitches them across the summit, and audits the complete
result. It does not hard-code a motion only because its root trace looks
plausible.

External GRAIL files remain outside Git. The committed code records source
stems, paths, transforms, hashes, and audit measurements.

## Runtime State Machine

The local viewer has three motion states:

```text
FLAT_MOTIONBRICKS -> TERRAIN_PORTAL -> FLAT_MOTIONBRICKS
```

### Flat MotionBricks

The official MotionBricks G1 controller receives the ordinary robot-local
movement command from W/A/S/D. The robot root stays aligned with gravity.

Before applying a flat frame, the runtime predicts whether it would cross a
terrain guard plane. If no compatible portal is available, forward motion is
clamped at the guard. Flat MotionBricks is never allowed to walk onto the hill
and rely on display correction.

### Portal capture

A portal is eligible when:

- the root is within `0.20 m` of its capture pose;
- desired travel alignment with the portal direction is at least `0.45`;
- yaw mismatch is no more than `45 degrees`; and
- lower-body phase error is no more than `0.38 rad` RMS.

Selection uses the existing `select_terrain_portal()` scoring and deterministic
tie-breaking. The portal begins approximately `0.8 s` before the geometric
hill seam so the entry residual can settle on flat ground.

### Committed terrain playback

Once captured, `TerrainCoursePlayback` owns root position, root orientation,
pelvis, all 29 joints, contact timing, and traversal speed. The entry uses the
existing short residual blend (`18` frames) rather than a pose cross-fade.

Movement keys remain observable for diagnostics, but do not redirect the robot
mid-course. Escape and window-close remain authoritative. This committed
interval is intentional: the small first prototype proves motion quality
before adding a bank large enough for mid-slope replanning.

No terrain IK, display root conditioning, stance lock, or swing-foot lift runs
on portal frames.

### MotionBricks continuation

At the final authored landing:

1. take the last four portal qposes;
2. register them into MotionBricks' flat support frame;
3. try the deterministic phase seeds already used by the reference viewer;
4. choose the continuation with the smallest boundary pose/velocity error;
5. discard the four decoded context frames; and
6. resume W/A/S/D control from the first new MotionBricks frame.

The terrain course's final flat support height becomes the MotionBricks support
height. Root yaw and global placement remain continuous.

If no finite continuation passes the boundary checks, the viewer holds the
audited final pose and reports a visible failure instead of jumping to a
default MotionBricks phase.

## Course Construction

The offline builder follows the same separation as the reference branch.

### 1. Direct local GRAIL import

A small adapter loads the ten `slope_113` robot PKLs directly from the released
corpus. It validates:

- `root_trans_offset` shape `(T, 3)`;
- XYZW root quaternion shape `(T, 4)`;
- MuJoCo-order DOF shape `(T, 29)`;
- finite values and unit quaternions;
- 25 Hz source rate; and
- a matching USD stem.

It resamples translation and joints to 50 Hz, uses quaternion SLERP for root
rotation, and recomputes derivatives after resampling. The runtime never
depends on unavailable `/move/...` C490 archives.

### 2. Exact mesh and route

The builder loads the paired USD mesh with its documented GRAIL transform,
extracts its low flat, ramp, and high edge, then constructs the symmetric mound
from two rigidly transformed copies and a flat summit. It converts the complete
triangle vertices/faces into the course artifact. The local MuJoCo scene
renders and queries that mesh rather than the prototype's analytic mound.

Candidate ascent windows include low-flat lead-in, the complete ramp, and
stable high support. Candidate descent windows begin on stable high support,
contain the complete authored descent after the source turn, and end on low
flat. Planar/yaw registration maps both windows into one forward hill crossing.
The summit stitch searches compatible contact phase and is included in every
collision and mechanics audit.

The opposite portal is built from independently selected and registered
windows facing the other direction. Time reversal is not used: each direction
still plays a forward ascent followed by a forward-facing authored descent.

### 3. MotionBricks entry

The existing course composer phase-searches a flat MotionBricks approach and
aligns it to the assembled hill primitive's first flat frames. It searches
lower-body phase, global yaw, and planar registration while preserving the
terrain motion after the entry seam.

The first course has no prerecorded exit. Live exit phase matching is performed
from the exact final four terrain poses, matching the current reference
runtime.

### 4. Independent audit

A course is publishable only if all of these pass:

- maximum sole penetration: `0.005 m`;
- forbidden non-foot body penetration: exactly `0`;
- maximum root translation step: `0.060 m`;
- maximum root rotation step: `0.35 rad`;
- maximum joint step: `0.25 rad`;
- maximum root acceleration: `40 m/s^2`;
- finite root, quaternion, joint, velocity, mesh, and contact arrays; and
- accepted entry seam and live-exit continuation.

The existing clearance repair may lift root Z only upward, targets `0.003 m`
clearance, is bounded to `0.060 m`, and is allowed only during audited course
construction or the flat/entry seam. It does not feed back into MotionBricks
and it never becomes a runtime foot-lock system.

Both directional courses must pass before the builder reports success. A
failed direction is not silently exposed as a portal.

## Code Structure

The implementation reuses the branch's established components:

- `motionbricks_terrain_portal.py` remains the course schema, selector, and
  playback engine;
- `motionbricks_global_terrain_viewer.py` supplies the guard, registration,
  support-height, and live-exit behavior;
- a new direct-GRAIL hill builder handles local source discovery, conversion,
  candidate scoring, composition, and auditing; and
- `motionbricks_hill_viewer.py` becomes a thin keyboard/MuJoCo frontend over
  the shared portal runtime.

Exit phase matching and flat-to-course guards are extracted into runtime-neutral
helpers if necessary. They are not copied into a second divergent state
machine.

The previous online hill IK modules remain available for isolated tests and
comparison, but they are removed from the default interactive path. A raw-flat
diagnostic mode may bypass portals; it is clearly labeled unsupported on the
hill.

Generated course NPZs, reports, and renders live under a local run directory
and remain untracked. The repository stores only builders, schemas, tests, and
small synthetic fixtures.

## Diagnostics and Failure Handling

The viewer overlay and console report:

- runtime state;
- selected portal and source stem;
- distance, direction alignment, yaw error, and phase RMS;
- guard-clamped status;
- course frame and total frames;
- entry residual magnitude;
- final continuation seed and boundary score; and
- any hold/failure reason.

Missing GRAIL, MotionBricks, model, or built-course paths fail at startup with
the exact missing path and the command needed to build the artifact.

Malformed portal data never falls back to online IK. During flat motion it
clamps at the guard; during committed playback it holds the last finite audited
pose and reports the rejected field.

## Testing

Pure unit tests cover:

- direct PKL validation and 25-to-50 Hz resampling;
- quaternion convention and normalization;
- deterministic ascent/descent pair ranking;
- summit stitch rejection and selection;
- bidirectional portal eligibility;
- flat-frame guard rejection;
- portal state transitions;
- committed-course input behavior;
- final-four-pose extraction; and
- continuation context-frame discard.

Native model tests cover:

- exact mesh registration in the MuJoCo scene;
- entry root/joint continuity;
- portal frames equal the audited course output;
- no online IK or root conditioning on portal frames;
- no sole or forbidden-body penetration above audit limits;
- support-height registration at exit; and
- finite forward and return courses.

The CUDA runtime canary drives MotionBricks to each portal, plays the complete
course, resumes MotionBricks, and records all transition diagnostics.

## Interactive Acceptance

The prototype is accepted when the user can:

1. launch one local viewer;
2. steer the ordinary MotionBricks character with W/A/S/D on both sides of the
   hill;
3. approach either portal and see a full-body authored 18.90-degree hill
   traversal;
4. observe no planted-foot dragging caused by a runtime world lock;
5. see no visible pose pop at entry or exit; and
6. regain responsive W/A/S/D control after the landing.

This milestone proves a high-quality slope interval using the same architecture
as the reference branch. Freely steering or stopping halfway up the slope is a
later motion-bank problem, not part of this portal prototype.
