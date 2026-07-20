# Object-Anchored Entry-Conditioned Funnel Correction

Date: 2026-07-20

Status: approved for implementation planning

## Problem

The playable learned pickup currently converts every training trajectory into
the frozen entry root's frame and follows its 75 reconstructed samples as
time-indexed targets. This creates two runtime defects:

1. the interaction endpoint is only implicit in a sampled entry-relative
   displacement rather than remaining anchored in the object's frame; and
2. ordinary locomotion chases tiny, rapidly changing targets, so tracking lag
   can make the character circle an old waypoint instead of progressing toward
   the object.

The correction must preserve the intended hierarchy: ordinary locomotion gets
the character close, diffusion generates several local approaches conditioned
on the actual arrival, and the existing matcher/runtime still owns contact and
attachment.

## Selected Approach

Use object-local diffusion output with an exact live-entry boundary and follow
the selected trajectory as geometric route rather than a playback clock.

Two alternatives were rejected:

- Warping only the sampled endpoint back onto the object is small to implement
  but can distort intermediate samples and invalidate collision clearance.
- Keeping entry-relative samples and tuning the current time-locked controller
  leaves the object endpoint implicit and does not remove target-chasing
  oscillation.

## Runtime Handoff

Pressing Smart Pickup while far away retains automatic coarse travel. Coarse
travel steers toward the object until the live root enters a safe capture
annulus and has a collision-free route. It does not require the character to
reach a preselected radial waypoint.

Once the live root is inside the annulus, facing acceptably, and below the
capture speed threshold for three native ticks, the backend freezes that exact
displayed root and simulation velocity. Those values form the entry condition.
There is no connector back to a projected capture point and no second handoff
before learned following.

## Model Representation

The 24-value condition remains:

- the existing 18 object/grasp values; and
- object-local entry `(x, z, sin(yaw), cos(yaw), velocity_x, velocity_z)`.

Training targets remain in outward, interaction-to-entry order, but every
sample is expressed directly in the object's planar frame. The model therefore
generates a distribution of routes extending outward from the selected
object/grasp while the entry condition selects routes relevant to the current
arrival.

The outward entry sample is excluded from learned loss and restored exactly
from condition values 18 through 21 after sampling. Reversing once produces an
execution trajectory whose first sample is exactly the frozen live entry in
the object frame. The interaction-side sample remains object-local and is
accepted only when route certification and the existing final matcher preview
both succeed.

No generated sample is transformed through the frozen entry root. Runtime
maps every object-local sample directly through the frozen target object's
world transform.

## Proposal Certification and Selection

The worker still generates one deterministic batch of 32 proposals. A proposal
is rejected when it has invalid values, discontinuous object-local geometry,
an entry sample unequal to the frozen entry condition, collision along the
route, excessive path curvature or length, or an incompatible final matcher
preview.

Selection remains deterministic and favors short routes, small heading change,
clearance, matcher cost, and proposal index. The chosen proposal and object
snapshot are immutable for the attempt. Target movement or generation change
cancels the attempt instead of silently moving the anchor.

## Geometric Lookahead Follower

The existing 16 knots may still be expanded to a smooth 75-point polyline, but
the points are no longer deadlines tied one-to-one to controller ticks.

At each 25 Hz update the follower:

1. projects the tracked root onto the remaining polyline;
2. advances a monotonic progress cursor, never selecting a point behind the
   furthest confirmed progress;
3. chooses a target a fixed distance ahead along the route, clamped to the
   terminal sample; and
4. supplies that target's position and tangent/facing to ordinary locomotion.

Near or passed points are skipped. The follower cannot repeatedly command an
old waypoint. Completion is spatial: the root must reach the terminal position
and yaw tolerances and settle, rather than merely consuming 75 ticks.

The follower fails closed on excessive cross-track error, blocked remaining
route, target authority change, timeout, explicit cancellation, or invalid
progress. A generous total timeout prevents an attempt from running forever
without imposing the original sample timing.

## Existing Authority Boundaries

Diffusion continues to generate planar root routes only. It does not write the
simulation root, pose, object state, or attachment. After spatial route
completion, the unchanged final preview submits the unchanged `PickRequest`.
The existing matcher and interaction runtime own Reach, Contact, attachment,
Hold, Carry, placement, and release.

Authored Smart Pickup remains an explicit provider and is not consulted by
learned-mode acceptance tests.

## Verification

Regression tests must prove:

- generated samples remain object-local under common world translation and
  rotation;
- the execution start exactly equals condition values 18 through 21;
- two different entry bearings produce different object-local proposal sets
  while terminating in matcher-compatible regions around the same object;
- runtime world targets remain anchored when the entry root changes;
- lookahead progress is monotonic and skips passed points;
- a lagging character is not commanded back to an earlier waypoint;
- completion depends on terminal spatial tolerance, not elapsed sample count;
- cross-track, obstacle, timeout, and target-authority failures cancel cleanly;
- one live graphical attempt reaches Contact and Carry without orbiting; and
- existing authored pickup, placement, and attachment regressions remain green.

Playable evidence will overlay the object, frozen entry, full selected route,
monotonic progress point, current lookahead target, terminal root, and tracked
root. This makes a return toward the entry or waypoint orbit visible directly.

## Non-Goals

- Long-range diffusion navigation.
- Full-body pose generation.
- Obstacle-conditioned model training.
- Replacing final motion matching or interaction authority.
- Resampling while an attempt is already following a route.
