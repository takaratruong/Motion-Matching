# G1 Horizontal Contact-Overlay Traversal Design

## Goal

Produce one offline 50 Hz G1 kinematic trajectory that travels parallel to the
stair risers: approach from flat ground, step onto the side of the staircase,
walk across with the two feet allowed to occupy different tread heights, step
down from the opposite side, and continue on flat ground. Heading and travel
direction remain aligned for the complete route.

## Scope

This design produces one fixed horizontal route on the existing representative
GRAIL staircase. It excludes diagonal and head-on routes, interactive motion
matching, Sonic, physics tracking, depth input, and realtime constraints.

The word `horizontal` means travel parallel to the stair risers. On the current
scene grid, height varies primarily along scene Y, so the horizontal route
travels primarily along scene X. Existing angle labels are not accepted as
evidence of direction; the generated root path and height gradient determine
the route classification.

## Contact Overlay

Extract a clean alternating gait from the accepted flat Takara/GRAIL corpus.
Represent the gait as sole-sized support intervals rather than isolated contact
frames. For each interval record:

- support foot identity;
- touchdown and liftoff frames;
- sole center position and yaw;
- nominal stride length and width; and
- pelvis position, yaw, and velocity at both boundaries.

Overlay a family of this normal gait along the fixed horizontal route. Enumerate
left- and right-leading variants, gait-phase offsets, and bounded changes to
stride length, width, and duration. Project each stance sole to one target
tread while preserving the route heading. Reject a proposal when any sole
straddles a riser, lies outside the terrain, exceeds leg reach, reverses
progress, or lacks continuous support outside an explicitly planned dynamic
step-down.

The output is an authored contact plan containing the complete approach,
side-entry, unequal-height middle walk, side-exit, and departure. The contact
plan is the source of terrain truth; retrieved or generated motion never
overrides it.

## Motion Retrieval

Search authenticated GRAIL windows against the complete contact plan. Candidate
cost contains:

- support-mask and contact-timing mismatch;
- left/right supported-height mismatch;
- relative sole transform mismatch;
- root progress and fixed-heading mismatch;
- pelvis-height and pelvis-roll mismatch; and
- boundary joint-pose and joint-velocity mismatch.

Use dynamic programming over complete windows so an individually good step
cannot create an incompatible next transition. Prefer longer windows and fewer
splices. Rigid XY/yaw placement, bounded time warping, and small terrain IK are
allowed. A candidate requiring more than 0.35 rad joint correction or 0.06 m
root correction at any frame is missing coverage and is rejected rather than
heavily morphed.

## Generative Fallback

When no GRAIL path satisfies a contact-plan interval, supply that interval's
sole targets, support schedule, pelvis constraints, and exact boundary poses to
the installed G1 ARDY model. ARDY is a motion proposal only. Project the result
onto the authored contacts and reject it under the same deformation and
validation gates. MotionBricks may be evaluated as an additional proposal
source, but the route does not depend on an unreleased interface.

## Composition and Validation

Compose the accepted entry, middle, and exit intervals without unvalidated
linear pose blends. Preserve planted soles through every splice and apply
inertialized joint residuals only inside the bounded correction allowance.

The final artifact contains 50 Hz joint positions, root pose, authored support
mask, sole targets, semantic phase boundaries, source provenance, and
per-frame validation diagnostics.

The route is accepted only when all of these hold:

- it contains approach, side-entry, at least two alternating unequal-height
  walking steps, side-exit, and flat departure;
- horizontal root progress is at least 0.50 m and monotonically forward apart
  from 0.02 m of local gait oscillation;
- heading error is at most 5 degrees;
- every supported sole has at least three valid sole samples;
- maximum stance contact error is at most 0.020 m;
- maximum planted-foot horizontal movement is at most 0.010 m per frame;
- minimum sole clearance is at least -0.025 m;
- no unplanned unsupported frame exists;
- maximum joint step is at most 0.30 rad at 50 Hz;
- maximum root step is at most 0.04 m at 50 Hz; and
- the rendered contact sheet and kinematic playback show no foot floating,
  crouch-walk collapse, leg splay, freeze, heading change, or skipped phase.

Failed validation remains failure; thresholds are not relaxed to publish a
route.

## Operator Output

One documented command rebuilds the route and one documented command launches
looped passive MuJoCo playback. The viewer uses the exact accepted artifact and
performs no hidden runtime correction.
