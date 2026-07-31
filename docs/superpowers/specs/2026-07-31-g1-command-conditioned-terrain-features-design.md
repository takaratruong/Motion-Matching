# G1 Command-Conditioned Terrain Features Design

## Goal

Replace the failed root-yaw-only dense terrain feature with a terrain query that
describes the surface along the operator's commanded path, while preserving an
exactly symmetric database/query implementation and interactive GPU search.

This remains a privileged-height-map kinematics experiment. It does not add
SONIC tracking, depth perception, IK, or a learned encoder.

## Evidence and scope

The retained dense feature samples a rigid 13 by 7 patch in current root yaw,
ignores `CommandTrajectory`, and subtracts the surface height under the root.
Visual testing still stalls, slides, and chooses incorrect stair phases. Prior
weight and corpus experiments improved aggregate metrics but did not make the
interactive stair behavior acceptable. Therefore this experiment changes the
terrain representation, not another search weight.

## Feature contract

The dense feature has 92 float32 values:

- 91 relative surface heights at 13 path stations by 7 lateral offsets;
- one pelvis clearance value: root Z minus terrain height under the root.

The lateral offsets remain `[-0.45, -0.30, ..., 0.45]` metres. The path stations
remain `[-0.15, 0.00, ..., 1.65]` metres. Positive stations follow the polyline
from the current root through the three 0.3, 0.6, and 0.9 second trajectory
positions. Negative distance extends behind the root. Stations beyond a short
or stopped trajectory extend using the last finite commanded facing. Each row's
lateral axis is perpendicular to its local path tangent, so forward, diagonal,
lateral, backward, and turning commands use one continuous construction.

Database rows use the identical sampler with each recorded frame's three future
root positions and facings. Query rows use the commanded future trajectory.
Both subtract terrain height under the current root from the 91 patch heights.

Flat clips use zero relative heights and recorded pelvis Z as clearance, treating
their implicit ground as Z=0. This prevents their clearance channel from being
an artificial zero that cannot match a normal standing query.

## Visualization and compatibility

The viewer renders the same 91 path-conditioned locations used by the query.
The clearance scalar is diagnostic only and has no separate marker. `legacy`
four-point terrain features remain unchanged. The dense database normalization
is rebuilt automatically because its dimension changes from 91 to 92.

## Verification

Automated tests must prove:

1. changing only the commanded direction changes the sampled dense terrain row;
2. the query path bends with a turning command instead of remaining root-yaw rigid;
3. database rows equal the shared sampler applied to recorded future trajectories;
4. the final channel equals pelvis-to-surface clearance, including flat clips;
5. invalid/out-of-domain samples continue to fail closed;
6. all motion-matcher, terrain, contact-segment, rollout, and viewer tests pass.

After automated qualification, the same staircase viewer must be tested with
side approaches, diagonals, reversals, turns on a tread, and side exits. A clean
test suite is not sufficient to claim the representation works visually.
