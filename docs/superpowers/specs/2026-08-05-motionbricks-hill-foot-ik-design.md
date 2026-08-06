# MotionBricks 18-Degree Hill Foot IK Design

Date: 2026-08-05

## Summary

Extend the interactive G1 MotionBricks hill prototype with an 18-degree radial
mound and a bounded stance-aware foot IK pass. MotionBricks remains responsible
for root motion, pose generation, gait timing, and command response. IK runs
after generation, changes only the twelve G1 leg joints, and is never fed back
into MotionBricks context.

The purpose of this increment is visual and kinematic contact quality. It does
not add dynamics, balance control, collision response, retraining, or
terrain-normal root alignment.

## Terrain

Keep the current radial cosine mound and its surrounding flat ground. Preserve
the existing seven-metre diameter and move only its crest height.

For mound height `h`, diameter `L = 7 m`, and radial cosine profile, maximum
grade is:

`maximum_grade = atan(h * pi / L)`

Set:

`h = tan(18 degrees) * L / pi = 0.7239760607 m`

The analytic height query remains the sole terrain contract for target
conditioning, IK targets, tests, and mesh vertices. The surface is smooth at
the crest and at its transition to surrounding flat ground.

## Selected IK Architecture

Add a focused MotionBricks-qpos adapter rather than coupling the viewer to the
portal-course `KinematicPose` pipeline.

`MotionBricksHillFootIK` owns:

- one read-only MuJoCo model reference and private `MjData`;
- the four sole collision spheres beneath each ankle;
- the six hip/knee/ankle joints and limits for each leg;
- raw previous-frame sole positions for velocity measurement;
- left and right stance latch state;
- previous bounded IK corrections for temporal rate limiting; and
- the latest immutable diagnostics.

It accepts one raw native G1 qpos, the frame timestep, and the analytic terrain
height query. It returns a detached corrected qpos and diagnostics. It never
mutates the MotionBricks frame buffer.

## Contact State

Contact inference uses the raw MotionBricks pose, before IK:

1. Forward-kinematics computes every sole-sphere centre and bottom.
2. Minimum terrain clearance and rigid-foot translational speed are measured
   for each foot.
3. An unlatched foot enters stance when minimum clearance is at most `0.035 m`
   and foot speed is at most `0.35 m/s`.
4. A latched foot remains in stance under hysteresis until clearance exceeds
   `0.075 m` or speed exceeds `0.75 m/s`.
5. Both feet may be latched during a genuine slow double-support phase.
6. State changes are committed only when the complete IK result is accepted.

On a stance rising edge, preserve the sole centroid's world XY and set each
sphere's target Z to terrain height at that sphere's XY plus its radius. The
distinct target heights across heel/toe and left/right probes make the ankle
follow the local slope. While stance remains active, the projected sphere
heights and centroid target stay fixed in world space so the foot does not
skate. A rigid foot cannot preserve every probe's individual XY while rotating
onto an 18-degree slope, so the centroid is the physically feasible horizontal
plant constraint.

After release, no horizontal swing-foot target is retained.

## IK Solve

For each active stance foot:

1. Stack the vertical Jacobian row of every sole sphere plus the X/Y rows of
   their mean Jacobian for a planted-foot centroid constraint.
2. Restrict the Jacobian to that leg's six degrees of freedom.
3. Solve damped least squares with a small posture regularizer toward the raw
   MotionBricks joints.
4. Clamp each iteration step to `0.04 rad`.
5. Clamp absolute correction from the raw pose to `0.35 rad`.
6. Clamp change in IK correction from the prior accepted frame to `0.06 rad`.
7. Respect native MuJoCo joint limits.

The correction-rate bound yields to a native joint limit when the next raw
MotionBricks pose makes the prior correction infeasible. In that case the
correction moves directly toward the joint-safe interval and the forced limit
move is committed, preventing stale correction history from causing repeated
raw-pose fallbacks.

For an unlatched swing foot, bias toward its authored joints. If any
sole sphere would penetrate the terrain or violate `0.015 m` minimum swing
clearance, solve only the four vertical Jacobian rows toward a common upward
target sufficient to clear the worst sphere. Do not pull a hovering swing foot
downward. The first observed frame is a measurement-only warmup so an unknown
initial foot velocity cannot create a false swing correction.

The root seven qpos values and every non-leg joint must be exactly copied from
the raw MotionBricks output.

## Viewer Data Flow

Each display frame follows this order:

1. Read raw qpos from MotionBricks.
2. Use raw qpos for MotionBricks context, spring control, and regeneration.
3. Run the optional foot IK adapter on a detached copy.
4. Put only the corrected copy into the viewer's `MjData`.
5. Render and report diagnostics.

At the beginning of the next frame, the viewer overwrites `MjData` with the
next raw qpos before controller signal generation. IK joints therefore cannot
become MotionBricks context or influence its root-height response.

Add `--no-ik` for direct raw/IK A/B comparison. IK is enabled by default.

The live diagnostic line adds:

- left/right state: `swing`, `stance`, or `release`;
- maximum raw and corrected sole penetration;
- stance target residual;
- maximum current joint correction; and
- whether the frame was accepted or fell back to raw qpos.

## Error Handling

The IK update is transactional. A newly planted foot may require several
accepted frames to reach its target because correction changes are capped at
`0.06 rad` per frame. During that acquisition ramp, a bounded correction may
be committed above the settled `0.015 m` vertical residual only while it
strictly reduces terrain penetration. Reject the complete corrected frame and
render raw qpos when:

- input qpos, terrain samples, Jacobians, or solved values are non-finite;
- the G1 model lacks the expected sole spheres or leg joints;
- a joint limit or correction bound would be exceeded;
- settled stance vertical residual exceeds `0.015 m`, or an acquisition step
  does not reduce terrain penetration.

The first visual prototype does not enable MuJoCo terrain contact dynamics, so
non-foot collision rejection remains outside this pass.

On a rejected update, do not advance latch or correction history. Emit one
rate-limited warning containing the reason. Repeated rejection must not stop
MotionBricks or the viewer.

## Tests

Focused CPU tests cover:

- exact 18-degree maximum grade and the `0.7239760607 m` crest;
- unchanged radial smoothness and mesh/height agreement;
- stance enter/release hysteresis;
- latch target construction from per-sphere terrain heights;
- no downward correction for a valid swing foot;
- root and non-leg qpos invariance;
- absolute and per-frame joint-correction bounds;
- joint-limit enforcement;
- transactional rollback on a failed solve; and
- `--no-ik` CLI availability.

MuJoCo integration tests use the exact G1 model to require:

- lower stance-foot penetration after IK on uphill and downhill poses;
- bounded stance target residual;
- exact raw root preservation;
- exact non-leg preservation; and
- finite corrected qpos.

A deterministic CUDA canary runs the same straight 360-frame command with IK
off and on over the 18-degree mound. It records raw/corrected sole clearance,
stance state, target residual, joint correction, root height, and terrain
height. The canary must prove identical raw MotionBricks qpos between arms
before the corrected display copy is applied.

## Scope Boundaries

This increment does not:

- change MotionBricks root constraints beyond the existing terrain-height
  conditioner;
- tilt or replace the generated root;
- feed IK joints back into generation;
- solve the arms, waist, or root;
- add physical simulation or balance recovery;
- guarantee dynamically feasible 18-degree locomotion; or
- claim that improved visual contact is sim-to-real evidence.
