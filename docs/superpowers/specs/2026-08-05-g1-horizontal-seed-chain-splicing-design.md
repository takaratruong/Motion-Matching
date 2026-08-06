# G1 Horizontal Seed-Chain Splicing Design

## Goal

Convert each visually approved horizontal staircase seed chain into one
continuous kinematic traversal without regenerating or deforming the accepted
mount, uneven-travel, or dismount motion interiors.

The first qualification target is `lane-pos-0p8`. It contains two strict
source boundaries and is therefore the cleanest test of the method.

## Evidence and rejected alternatives

The seed path intervals alone are insufficient for concatenation. For example,
the raw `lane-pos-0p8` mount and interior endpoints differ by approximately
0.19 m in root position and 0.405 rad RMS in joint pose even though their path
intervals overlap by 0.0047 m.

Regenerating the complete route with MotionBricks is also rejected. The prior
task-actor experiments produced visually plausible unconstrained motion but
52 terrain-unsupported frames. MotionBricks conditions on endpoint pose and
root state; it does not enforce the staircase contact schedule.

A generic linear blend is rejected because it can slide a planted foot,
interpolate through the stair geometry, and corrupt motion that has already
passed visual review.

## Architecture

For each adjacent seed pair:

1. Search supported cut-frame pairs near the outgoing and incoming boundaries.
   Rank pairs by world-space root distance, root heading, joint pose, joint
   velocity, and support compatibility.
2. If direct concatenation passes the boundary continuity and terrain-contact
   contract, use it without generated frames.
3. Otherwise, give four exact outgoing frames and four exact incoming frames
   to the released MotionBricks G1 inbetweening model. Sweep every supported
   token duration.
4. Resample each generated transition from 30 Hz to 50 Hz. Apply endpoint
   correction only to generated frames so the two source motions remain
   unchanged.
5. Infer support from sole clearance and foot velocity, then reject candidates
   that penetrate terrain, slide a stance foot, introduce an unplanned flight,
   change heading, or create a discontinuous root/joint step.
6. Rank only accepted candidates by boundary acceleration, contact margin, and
   duration. Concatenate the unchanged source prefixes/interiors/suffixes with
   the selected transitions.

MotionBricks is a proposal generator. MuJoCo sole geometry and the authored
terrain are the source of truth.

## Data products

The runner consumes `seed-chains.json`, the source connector archives, the
stair dataset/configuration, the G1 MuJoCo XML, and the local released
MotionBricks checkout.

For each lane it writes:

- `traversal.npz`: one continuous 50 Hz route;
- `boundary-candidates.json`: every cut pair and generated duration with
  deterministic acceptance or rejection reasons;
- `metrics.json`: route-level continuity and terrain metrics;
- `contact-sheet.png`: fixed-frame visual audit.

The artifact records the exact source frame ranges retained from every seed
and the exact generated transition ranges.

## Acceptance contract

The `lane-pos-0p8` route is accepted only if:

- the source arrays outside selected boundary windows are bit-identical to the
  approved connectors;
- both source boundaries are continuous with maximum root step at most
  0.05 m and maximum joint step at most 0.35 rad at 50 Hz;
- minimum sole clearance is at least -0.025 m;
- a supported sole has at least three terrain-contact samples;
- maximum planted-foot horizontal motion is at most 0.010 m per frame;
- there is no unsupported frame outside an explicitly certified dismount
  flight already present in the source dismount;
- root-heading error from the lane heading is at most 5 degrees; and
- automated validation, contact-sheet inspection, and live playback all pass.

A missing passing candidate is reported as a boundary-coverage failure. The
runner must not silently fall back to a teleport, source deformation, or an
uncertified blend.

## Generalization

After `lane-pos-0p8` passes, the same runner is applied unchanged to
`lane-pos-1p0` and `lane-pos-0p6`. The latter remains visually useful but its
approved dismount is an explicit non-strict exception, so the generated route
must keep that distinction in its metadata rather than presenting all three
lanes as equally certified.

Sonic tracking, online control, physics simulation, depth input, new terrain
angles, and MotionBricks retraining are out of scope.
