# G1 Command-Conditioned Terrain Features Design

## Goal

Augment the body-relative dense terrain feature with a command-path profile,
while preserving exactly symmetric database/query sampling and interactive GPU
search.

This remains a privileged-height-map kinematics experiment. It does not add
SONIC tracking, depth perception, IK, or a learned encoder.

## Evidence and scope

The retained dense feature samples a rigid 13 by 7 patch in current root yaw,
ignores `CommandTrajectory`, and subtracts the surface height under the root.
Visual testing still stalls, slides, and chooses incorrect stair phases. Prior
weight and corpus experiments improved aggregate metrics but did not make the
interactive stair behavior acceptable. Therefore this experiment changes the
terrain representation, not another search weight.

An initial 92-value treatment rotated the entire 13 by 7 patch into the travel
path and appended pelvis clearance. It failed the real two-clip gate: longest
unsupported run increased from 10 to 11 frames and stance slide increased from
0.339 m to 0.440 m. Removing clearance caused prolonged rescue search. Adding
the old body patch beside the full path patch also caused prolonged rescue
search.

Controlled isolation established the retained architecture:

- body patch plus four path heights, without clearance, reproduced all passing
  baseline metrics exactly;
- body patch plus clearance, without the path heights, increased stance slide
  to 0.401 m and failed qualification.

The root cause is representational: rotating the whole patch into travel
direction removes which side of the body is uphill, but that relationship is
needed to select a sideways gait and validate registered contacts. Pelvis
clearance changes gait-phase selection and is already covered by downstream
physical clearance validation, so it is not a search feature.

## Feature contract

The dense feature has 95 float32 values:

- 91 body-relative surface heights in the existing 13 by 7 root-yaw grid;
- four surface heights along the commanded/recorded trajectory at arc distances
  0.25, 0.50, 0.75, and 1.00 metres.

The body grid retains forward offsets `[-0.15, 0.00, ..., 1.65]` metres and
lateral offsets `[-0.45, -0.30, ..., 0.45]` metres at 0.15 m spacing. The four
path points follow the polyline from the current root through the three 0.3,
0.6, and 0.9 second trajectory positions. A short or stopped trajectory extends
using the last finite commanded facing.

Database rows use recorded root yaw for the 91-point body patch and recorded
future positions/facings for the four path points. Query rows use emitted root
yaw and the shaped command trajectory. All 95 values subtract terrain height
under the current root.

Flat clips use 95 zero relative heights. Root and foot clearance remain explicit
downstream safety metrics and contact gates, not nearest-neighbor features.

## Visualization and compatibility

The viewer renders the same 95 locations used by the query: 91 body-grid points
and four command-path points. `legacy` four-point terrain features remain
unchanged. Dense database normalization is rebuilt automatically because its
dimension changes from 91 to 95.

## Verification

Automated tests must prove:

1. changing only command direction leaves the 91 body values unchanged and
   changes the four path values;
2. the four-point profile bends with a turning command;
3. database rows use the same body and path definitions as runtime queries;
4. invalid/out-of-domain samples continue to fail closed;
5. all motion-matcher, terrain, contact-segment, rollout, and viewer tests pass;
6. the real two-clip safety rollout remains accepted without weakening gates.

After automated qualification, the same staircase viewer must be tested with
side approaches, diagonals, reversals, turns on a tread, and side exits. A clean
test suite is not sufficient to claim the representation works visually.
