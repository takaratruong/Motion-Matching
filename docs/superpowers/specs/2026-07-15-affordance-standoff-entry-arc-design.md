# Affordance-relative standoff entry arc

Date: 2026-07-15
Status: approved design checkpoint

## Context and observed failure

The playable placement flow must walk from the live default spawn, settle near the pickup object, and invoke the ordinary online pickup resolver from the live flat-controller pose. The clip-0 Reach transform is data-derived and remains the stable navigation and evidence reference. The flow must not initialize or relocate either root from a canonical pose.

The first live attempt reached the stable Reach neighborhood but the runtime correctly rejected pickup clearance. Diagnostic instrumentation located the collision in the early live-pose-to-Reach entry blend, source frames 118 to 119, while the selected contact frame was 139. It was not the terminal contact segment, so changing the collision exemption, matcher, IK, or contact thresholds would be incorrect.

A first affordance-relative correction translated the Reach point by a derived lateral clearance of `0.138841 m`. That removed the original direct approach to the collision line, but pure translation also moved the pickup distance to approximately `0.515 m`. The unchanged playable contract requires a `0.35 m` to `0.45 m` standoff, so the controller latched braking, settled with zero input, and could never satisfy the standoff condition. The failure establishes that the clearance displacement must preserve the authored Reach standoff rather than translate away from it.

## Goals

The placement auto-demo will derive a root-only interaction waypoint `P` that:

- remains exactly one data-derived clearance chord from the stable Reach point `R`;
- preserves the planar distance from the live pickup object center `O`;
- chooses the side of the arc from the authored grasp approach and active hand;
- remains invariant under a common world-yaw rotation;
- is symmetric between left- and right-hand affordances;
- stays within the existing `0.15 m` stable-Reach evidence neighborhood;
- retains the existing `0.35 m` to `0.45 m` pickup standoff;
- is reached only through ordinary left-stick navigation and one-way braking.

The stable Reach position, stable Reach rotation, and all stable-Reach position/yaw evidence remain unchanged and continue to be measured against `R`, not `P`.

## Non-goals and fixed boundaries

This design does not change:

- collision geometry, collision exemptions, or support-path checks;
- online match selection, match costs, candidate thresholds, or clip ownership;
- IK configuration, reach weighting, attachment behavior, or playback;
- the fixed 25 Hz controller/runtime schedule;
- the `0.15 m` Reach-position bound, `20 degree` Reach-yaw bound, or `0.35 m` to `0.45 m` standoff band;
- placement preview, staging, replay, release, or support validation;
- the canonical interaction pack or its source frames.

Selecting an earlier Approach-frame entry is deferred to the planned discrete multi-entry selector. That work changes the selectable entry set and match architecture; it must not be smuggled into this deterministic clip-0 playable-evidence correction.

## Inputs and coordinate conventions

The derivation consumes only retained, validated interaction data:

- `R`: the stable clip-0 Reach root transform mapped into the live target scene;
- `O`: the retained live pickup object center;
- `Q`: the retained pickup object world rotation;
- `B = (Bx, By, Bz)`: the positive object-local dimensions;
- `A_object`: the finite, unit, object-horizontal grasp approach direction;
- `clearance`: the nonnegative authored `GraspAffordance::clearance_radius`;
- `hand`: the authored left or right grasp hand.

World up is `U = (0, 1, 0)`. All waypoint and standoff geometry is planar in world XZ. Quaternion/vector operations still use the full retained object rotation when transforming between object and world space.

## Clearance chord derivation

First rotate the authored approach direction into world space, remove its world-Y component, and normalize:

```text
A_world_raw = Q * A_object
A_world = normalize((A_world_raw.x, 0, A_world_raw.z))
```

The unsigned world lateral is:

```text
L_world = normalize(cross(U, A_world))
```

The hand-sided desired lateral is:

```text
hand_sign = +1 for Right, -1 for Left
H_world = hand_sign * L_world
```

Convert the unsigned lateral into object space and compute the oriented-box support radius along it:

```text
L_object = inverse(Q) * L_world
support = 0.5 * (
    abs(L_object.x) * Bx +
    abs(L_object.y) * By +
    abs(L_object.z) * Bz)
```

The clearance chord length is derived without a tuned offset:

```text
C = support + clearance + 0.001 m
```

The `0.001 m` term is the only numeric clearance epsilon in this derivation.

## Standoff-preserving arc construction

Let the planar Reach radius be:

```text
r = (R.x - O.x, 0, R.z - O.z)
D = length(r)
```

The chord angle is:

```text
theta = 2 * asin(C / (2 * D))
```

Rotate `r` around world up by both `+theta` and `-theta`, producing `r_plus` and `r_minus`. Construct two root-position candidates while preserving the Reach height:

```text
P_plus  = (O.x + r_plus.x,  R.y, O.z + r_plus.z)
P_minus = (O.x + r_minus.x, R.y, O.z + r_minus.z)
```

Score each chord against the hand-sided affordance lateral:

```text
score_plus  = dot(P_plus  - R, H_world)
score_minus = dot(P_minus - R, H_world)
```

Choose the candidate with the greater score. For an exact numeric tie, choose `P_plus` for the right hand and `P_minus` for the left hand. This hand-aware tie break preserves left/right symmetry without depending on a world axis.

The resulting interaction transform preserves `R.rotation`; only its root position becomes `P`.

## Required invariants

The derivation must verify, within the existing JSON precision/numeric tolerances:

```text
planar_length(P - R) == C
planar_length(P - O) == planar_length(R - O) == D
C <= 0.15 m
0.35 m <= D <= 0.45 m
P.y == R.y
P.rotation == R.rotation
```

Applying the same world-yaw rotation to `R`, `O`, and `Q` must rotate `P` by that yaw without changing `C`, `D`, or the selected hand-relative side. Switching only the hand from right to left must select the opposite arc candidate.

## Validation and failure behavior

Waypoint construction fails closed before evidence begins if any of the following is true:

- the target does not contain exactly one retained grasp affordance;
- the hand is neither left nor right;
- any position, dimension, rotation, approach, clearance, radius, score, angle, or candidate is non-finite;
- an object dimension is not positive;
- the object-local approach is not unit and horizontal within the pack tolerance;
- the planar world approach cannot be normalized;
- `D` is zero, non-finite, or outside the unchanged standoff band;
- `C` is non-finite, nonpositive, greater than `0.15 m`, or greater than `2D`;
- `asin(C / (2D))` or either candidate is non-finite;
- the derived position fails either chord or preserved-standoff invariant.

Failure is a visible controller error. The controller must not clamp the chord, widen a threshold, publish partial evidence, retry the other hand, choose a world-axis fallback, or relocate the live root.

## Controller data flow

During placement auto-demo setup:

1. Build the stable root-only Reach transform `R` from clip 0 and the retained live target.
2. Derive `C` and `P` once from `R`, `O`, the retained object geometry, and the retained grasp affordance.
3. Retain `R`, `P`, `C`, and all immutable derivation inputs in placement auto-demo state.
4. Build the existing pre-entry point from `R` and its authored facing.

During live input:

1. Navigate to the pre-entry point with the ordinary left stick.
2. Navigate toward `P` after the pre-entry funnel is reached.
3. Measure the one-way braking distance against `P`.
4. Once braking latches, keep left-stick magnitude at zero; never reverse or resume approach input.
5. Continue measuring Reach position/yaw against `R` and pickup standoff against the live object.
6. Pulse ordinary Interact only after the unchanged walk, standoff, speed, five-tick settle, Reach-position, and Reach-yaw contracts all pass.

No pose, root, canonical snapshot, match candidate, or interaction state is written directly by this derivation.

## Evidence and independent recomputation

Every 25 Hz placement JSONL row records the immutable derivation inputs and observed result:

- stable Reach position and rotation;
- interaction waypoint position;
- clearance chord length;
- retained interaction object position, dimensions, and rotation;
- retained object-local approach direction;
- authored clearance radius;
- hand sign;
- observed left-stick command/magnitude and brake latch;
- live root, live object, stable-Reach errors, standoff, speed, and walk displacement.

The Python validator independently recomputes the oriented support, `C`, `D`, both arc candidates, the hand-sided score choice, and `P`. It rejects changes to any immutable input, a mismatched chord, the wrong hand side, a translated rather than arc-derived point, a changed standoff, a world-axis shortcut, or any result outside the unchanged Reach/standoff bounds.

## Test plan

TDD coverage must include:

1. Identity-rotation right-hand geometry: recomputed chord and both distance invariants match.
2. Left/right symmetry: identical inputs with opposite hands select opposite arc candidates.
3. World-yaw invariance: rotating `R`, `O`, and `Q` together rotates `P` while preserving chord and standoff.
4. Oriented support: nonuniform dimensions and a rotated object change `C` according to the object-local support formula.
5. Failure cases: zero planar approach, invalid hand, nonpositive dimensions, non-finite input, zero/out-of-band `D`, `C > 0.15 m`, and `C > 2D` all fail closed.
6. Evidence corruption: waypoint position, chord, object geometry/pose, approach, clearance, or hand-sign mutations are independently rejected.
7. Static policy: placement setup and input use the derived root-only arc waypoint and do not use a canonical pose, direct root write, world-axis offset, collision exemption, or changed threshold.
8. Existing runtime regressions: table crossing, wrong-direction object crossing, lateral object crossing, matcher bounds, IK bounds, and fixed-rate scheduling remain unchanged and green.
9. Real graphical acceptance: the native 25 Hz placement gate walks from the live default spawn, brakes once, settles within all unchanged bounds, picks, carries, stages, places, releases, and publishes validated JSONL plus a nonblank PNG.

## Acceptance criteria

The design is complete when the implementation and independent evidence validator prove all stated invariants, existing collision/runtime tests remain unchanged and passing, no temporary diagnostics remain, and the real graphical placement gate succeeds without canonical relocation or threshold changes.
