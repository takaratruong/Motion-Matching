# G1 Terrain Landing Bridge Design

**Date:** 2026-07-31
**Branch:** `research/g1-torch-terrain-kinematics`
**Scope:** deterministic 50 Hz kinematic terrain locomotion. Sonic, learned
tracking, physics, and depth estimation remain out of scope.

## Measured failure

The retained contact-segment matcher passes 14 of the 21 same-stair routes.
All six turn routes, both diagonal-down routes, riser reversal, stop/restart,
both cross-tread routes, and one side mount complete. The remaining failures
share a missing terrain-transition action: four side exits do not land flat,
`mixed-adversarial` finishes elevated, and two asymmetric routes exhaust safe
entries.

At the captured upper-left exit state, exhaustive evaluation found 76
direction-compatible contact entries. Sixty-seven failed exact contact safety
and the nine safe entries changed root height by at most 2.1 cm. None was a
safe stair-down action. A 336-trial one-dimensional placement sweep and a
441-trial two-dimensional sweep found no rigid placement for the 16
substantial-down candidates. Expanding the index from 1,281 contact onsets to
40,701 alternate-phase suffixes also produced no flat side-exit landing.

The 176-clip representative corpus contains one authenticated lateral
drop-and-land sequence: curb clip 14, approximately frames 361--421, with a
15.5 cm drop. All 29 strict lateral landing windows in the corpus come from
that clip. The sequence crosses several support phases, so the current
onset-to-opposite-onset matcher cannot select it as one action. Rigid root
placement and per-frame vertical root warping both fail against the query
stair because the source and target contact geometry differ.

## Chosen architecture

Ordinary terrain locomotion remains motion matched. A separate, optional
landing-bridge generator handles the narrow case where a lateral/down command
has no safe substantial-down contact segment.

The generator consumes a real multi-support curb reference, a starting emitted
G1 state, the commanded planar direction, and the authoritative query height
grid. It produces one immutable 50 Hz action candidate:

1. rotate and translate the reference root path into the command frame;
2. derive stance and swing intervals from the authenticated source support
   mask;
3. lock stance feet to their current/query-terrain contacts;
4. map each landing foot to a lower query-terrain foothold along the command
   direction, with a clearance arc during swing;
5. solve only the G1 leg joints with deterministic damped-least-squares MuJoCo
   Jacobian IK while preserving the reference upper body and root orientation;
6. derive joint velocities by finite difference; and
7. reject the whole action unless the existing exact contact-segment validator
   and emitted terrain-clearance validator both accept it.

The first implementation generates and evaluates bridges offline in the
renderer-independent route harness. It does not enter the interactive matcher
until a lower side exit passes. This keeps the experiment causal and prevents
an unqualified procedural path from perturbing the 14 working routes.

## Components and interfaces

`MujocoG1FootKinematics.solve_leg_positions(...)` extends the existing
authoritative G1 FK adapter. It accepts one target-ordered joint vector, root
pose, a boolean two-foot solve mask, and target ankle positions. It returns a
new target-ordered joint vector or raises `ContractError`. The solver has fixed
iteration, damping, step, residual, and joint-limit contracts; it never changes
the floating root or non-leg joints.

`torch_terrain_landing_bridge.py` owns reference-window discovery, contact
target construction, per-frame IK, finite-difference velocities, and immutable
`TerrainLandingBridge` output. It depends on the terrain dataset, query feature
extension, support masks, and the FK/IK adapter; it does not depend on
`TorchMotionMatcher` internals.

The route evaluator receives an optional bridge provider. Initially the
provider is diagnostic: at a named route frame it evaluates the generated
action and records whether a valid bridge exists without changing matcher
output. Behavior-changing admission is a separate TDD task after the real
lower-exit oracle proves the bridge valid.

## Failure behavior

Every boundary fails closed. Malformed states, out-of-domain height queries,
unreachable IK targets, joint-limit violations, residuals above tolerance,
terrain penetration, lost support, or excessive unsupported runs reject the
entire bridge. Rejection leaves the ordinary matcher state and output bitwise
unchanged. No relaxed clearance threshold or hidden vertical teleport is
allowed.

## Acceptance

Phase A, bridge feasibility:

- the solver reaches synthetic reachable foot targets within 5 mm while
  preserving non-leg joints;
- unreachable targets fail deterministically;
- a generated bridge is finite, exactly 50 Hz, sequential, and immutable;
- the complete bridge passes the existing exact contact and terrain validators
  at one lower side-exit command boundary.

Phase B, matcher admission:

- at least one lower side-exit route finishes on flat ground;
- all 14 currently clean routes remain clean;
- rescue cycles remain zero;
- no emitted foot penetrates more than the existing 3 cm hard bound; and
- any bridge generation failure retains the ordinary safe matcher result.

Upper-tread side drops are reported separately. The corpus contains no
authenticated lateral landing near the approximately 53 cm top-tread drop. An
upper bridge may be admitted only if IK and the existing validators accept it;
otherwise the result is an explicit coverage/trackability rejection, not a
fabricated pass.

## Alternatives rejected

- **More weights, phase entries, or beam search:** prior experiments and the
  new exhaustive captured-state audit show no safe terminal candidate.
- **Rigid or root-only retargeting:** the one- and two-dimensional placement
  sweeps and vertical contact warp found zero valid placements.
- **Immediate C++ IK reuse:** it would reintroduce the legacy runtime coupling
  that this Torch function-call framework was created to remove.
- **New data collection first:** best for eventual tracking quality, but it is
  an external dependency and cannot answer kinematic feasibility now.
