# MotionBricks Authored-Contact Terrain IK Design

## Goal

Replace the geometric contact heuristic in the controllable MotionBricks hill
viewer with an authored-contact terrain post-process that does not drag a foot
after toe-off. Add enough display-only root-height freedom to traverse the
18-degree mound without saturating the leg solver.

The generated MotionBricks trajectory remains the source motion. Terrain
conditioning still runs before inference. All root-height and leg corrections
described here apply only to the detached qpos rendered by MuJoCo.

## Failure Being Replaced

The first prototype inferred stance from sole clearance and foot speed. Its
release threshold allowed a slowly moving toe-off foot to remain classified as
stance. The stale world-space target then pulled that foot backward as the root
advanced.

The same prototype kept the display root fixed. On the uphill canary, raw sole
penetration reached `0.151 m`; six-joint leg IK saturated at its `0.35 rad`
correction limit and recovered only about `0.030 m`. A leg-only solve cannot
repair that root-height error cleanly.

## Authored Contact Source

MotionBricks retains the model features for every generated display frame.
Those features contain four authored contact channels in this order:

1. left heel;
2. left toe;
3. right heel; and
4. right toe.

Before `get_next_frame()` advances the agent cursor, the viewer records the
current frame index. It samples contact values from the same
`full_agent.frames["model_features"]` frame as the returned qpos and uses the
MotionBricks motion representation's `extract_foot_contacts()` contract.

A foot is in authored stance when either of its two channels exceeds `0.5`.
No sole-clearance or speed threshold may extend authored stance. A contact
falling edge releases its world-space target in that same display frame.

If contact features, the motion representation, or the current frame index are
unavailable or malformed, both feet are treated as swing. The system clears
world-space locks instead of guessing from geometry.

## Display Root Height

The post-process may translate only display qpos root Z:

- root X/Y exactly equal raw MotionBricks;
- root quaternion exactly equal raw MotionBricks and therefore remains aligned
  with gravity;
- root-height correction bounded to `[-0.20 m, +0.20 m]`;
- accepted correction change bounded to `0.025 m` per display frame.

For each authored stance foot, forward kinematics measures all four sole
spheres. The required support shift for one sphere is:

```text
terrain_height(sphere_xy) + sphere_radius - sphere_center_z
```

The desired root-height correction is the maximum required shift across active
stance spheres, added to the existing display correction. This first removes
deep terrain penetration with a rigid body translation. The leg solve then
handles the residual slope orientation and planted position.

When neither foot is in contact, root-height correction moves toward zero by
at most `0.010 m` per frame. This avoids using a moving swing foot to bob the
pelvis.

## Contact Targets

On an authored contact rising edge:

- preserve the sole centroid's world X/Y;
- set every sole sphere target Z to analytic terrain height at that sphere X/Y
  plus its radius;
- retain the complete target while authored contact remains true.

During continuous stance, the target does not move. This provides a genuine
plant without relying on geometric contact inference.

On the authored falling edge:

- clear the world-space target before solving the frame;
- do not run any horizontal target constraint for that foot;
- allow its prior joint correction to decay toward raw motion by at most
  `0.12 rad` per frame.

The falling-edge state change is unconditional. A later solve rejection cannot
roll back or resurrect the released target.

## Leg Solve

After applying the bounded root-height correction, solve each active stance
foot with its six native G1 leg joints.

The damped least-squares constraint contains:

- one vertical Jacobian row for each sole sphere; and
- the X/Y rows of the mean sole Jacobian for the planted centroid.

This permits the rigid foot to rotate onto the hill while keeping its overall
horizontal plant. It avoids the infeasible requirement that every probe retain
individual X/Y while the foot rotates.

Per stance leg:

- maximum DLS iteration step: `0.04 rad`;
- maximum correction from raw motion: `0.30 rad`;
- maximum correction change during stance: `0.08 rad` per frame;
- native MuJoCo joint limits always apply.

For a swing foot, no world X/Y target exists. If any sphere has less than
`0.015 m` terrain clearance, solve only vertical sphere constraints toward a
common upward lift. Never pull a valid swing foot downward.

## Data Flow and State Ownership

Each display frame executes in this order:

1. record the MotionBricks frame index;
2. read authored contacts for that index;
3. read raw qpos with `get_next_frame()`;
4. generate the next MotionBricks buffer from raw context;
5. update contact edges, clearing falling-edge locks immediately;
6. compute a bounded display root-Z correction;
7. construct any rising-edge stance target from the root-adjusted pose, then
   run stance/swing leg IK on the display copy;
8. put only the corrected copy into viewer `MjData`;
9. render and report diagnostics.

Neither corrected root Z nor corrected joints are written into
`full_agent.frames`, controller context, spring context, or any inference
tensor. The next controller update overwrites `MjData` with the next raw qpos
before generating control signals.

## Failure Handling

The adapter returns a finite display qpos on every valid raw frame.

- Missing or invalid authored contacts clear all locks and disable horizontal
  stance constraints for that frame.
- A contact falling edge always clears its target, even if root or leg solving
  later fails.
- Invalid terrain samples return raw qpos with locks already released according
  to authored contact.
- A failed leg solve falls back to the root-adjusted raw pose for that frame.
- Root X/Y, root quaternion, and all non-leg joints remain exact raw values.
- No failed solve may stop MotionBricks or the viewer.

Diagnostics report:

- four authored contact channels;
- collapsed left/right stance;
- left/right lock state;
- display root-height correction;
- raw and corrected sole penetration;
- maximum stance target residual;
- maximum leg correction;
- acceptance and fallback reason.

## Testing

Pure tests cover:

- collapsing heel/toe channels into left/right stance;
- rejecting malformed/non-finite contact features;
- same-frame falling-edge target removal;
- no geometric re-latching when authored contact is false;
- bounded root-height correction and airborne decay.

Native G1 model tests cover:

- root X/Y and quaternion invariance;
- non-leg joint invariance;
- representative 18-degree flank penetration reduction;
- lower maximum leg correction after root-height adjustment;
- stance target preservation while contact remains true;
- immediate loss of horizontal lock at authored toe-off;
- swing correction never moves a valid swing foot downward; and
- invalid terrain fallback cannot restore a released lock.

Runtime CUDA canaries run raw and corrected arms over the complete mound. The
corrected trace must show:

- finite qpos for every frame;
- no frame with `authored_stance=0` and `locked=1`;
- display root correction within `0.20 m`;
- leg correction within `0.30 rad`; and
- lower crest/flank penetration than the first prototype.

The interactive acceptance criterion is visual: pressing forward over the
mound must not leave either released foot pinned behind the character.
