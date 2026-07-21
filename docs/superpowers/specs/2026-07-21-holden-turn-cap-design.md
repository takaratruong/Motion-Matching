# Holden Turn-Cap Design

**Date:** 2026-07-21

## Goal

Test whether bounding desired-heading speed fixes the visibly rough direction
reversals that remain after translational velocity shaping. Add an opt-in
`holden-turn-v1` profile that preserves `holden-v1` velocity behavior and caps
yaw at exactly 120 degrees/second before both motion-matching prediction and
runtime application.

This is a deliberately narrow experiment. It does not implement Holden's full
double-spring rotation model, change SONIC weights, couple turn rate to speed,
or tune the existing root-rotation spring.

## Evidence and hypothesis

The isolated live `holden-v1` run proved that translational shaping was active:
a requested `+0.9` to `-0.9 m/s` reversal braked toward zero and accelerated in
the new direction. The same trace showed desired heading changing by roughly
180 degrees while applied velocity was still positive. This allowed the heading
trajectory seen by motion matching to lead the physically achievable body turn.

Daniel Holden's [New Movement Model](https://theorangeduck.com/page/new-movement-model)
separates character rotation from translational velocity and models it explicitly.
For this experiment, an angular-rate cap is preferable to a new spring because
it gives one exact, interpretable bound and directly addresses the observed
target discontinuity.

The hypothesis is:

> If the desired heading presented to motion matching and runtime application
> follows a shared 120-degree/second path, abrupt reversals will remain
> controllable because the selected kinematics will no longer assume an
> instantaneous 180-degree turn.

## Alternatives considered

### Increase the existing rotation half-life

The runtime already applies a 0.27-second rotation spring. Increasing its
half-life would slow the simulated root but leave the motion-matching query's
desired heading discontinuous. It also provides no hard angular-rate bound.

### Couple heading change to translational speed

A brake-first/turn-second rule could be physically plausible, but it introduces
a speed threshold or coupling curve and makes two shapers interact. It is a
larger tuning surface than this first heading experiment needs.

### Cap desired yaw rate — selected

A cap is deterministic, acts at the shared prediction/application seam, and
gives a known minimum reversal time. At 120 degrees/second, a 180-degree turn
takes at least 1.5 seconds. The existing downstream 0.27-second spring remains
unchanged.

## Public contract

The server supports three exact movement profiles:

- `raw`: existing unshaped velocity and heading behavior, bit-for-bit.
- `holden-v1`: existing acceleration/deceleration shaping with uncapped heading,
  bit-for-bit.
- `holden-turn-v1`: the same velocity shaping as `holden-v1`, plus a fixed
  maximum yaw rate of 120 degrees/second (`2.094395102... rad/s`).

The profile remains selected by the existing reset field and manual-demo
`--movement-model` option. Unknown values fail before reset mutation. Hello and
reset evidence report the third supported profile and its server-authored turn
contract. No numeric tuning flag is exposed.

For `raw` and `holden-v1`, turn capping is disabled exactly. Their generated
arrays, diagnostics, reset semantics, and evidence remain compatible.

## Pure yaw limiter

A pure C++ unit accepts:

- previous committed heading;
- requested heading;
- step duration;
- selected profile; and
- the fixed turn configuration.

It validates finite unit quaternions and a positive finite step. For `raw` and
`holden-v1`, it preserves the existing unit-quaternion contract and returns the
requested heading exactly. `holden-turn-v1` additionally requires a planar-yaw
heading, computes the signed shortest yaw delta in `(-pi, pi]`, maps an exact
180-degree tie to positive pi, and advances by at most `max_yaw_rate * dt`. The
output is normalized. Validation failure leaves caller output untouched.

A companion predictor advances a local heading copy once per trajectory sample
using the same limiter. Prediction never mutates persistent state.

## Runtime state and transaction semantics

No new heading state is required. `g1_controller_state::desired_rotation`
already stores the last committed desired heading and participates in reset,
clone, swap, abort, and commit semantics.

Inside the runtime step:

1. Validate the requested heading.
2. Clone the current controller state.
3. Limit the request against the clone's committed `desired_rotation`.
4. Use that capped current heading for change diagnostics and current
   application.
5. Predict future capped headings from a local copy toward the original
   requested heading.
6. Feed those headings to the existing motion-matching trajectory-direction
   query and command evidence.
7. Commit only through the existing final state swap.

Any limiter, prediction, search, IK, or downstream failure leaves the original
heading state unchanged.

Reset seeds the anchor from the runtime's existing reset heading. A flat-scene
hold must not leave stale heading state for the next moving command.

## Prediction and application agreement

At the 25 Hz source step, the maximum heading change is exactly 4.8 degrees.
At the existing one-third-second trajectory sample spacing, a local prediction
may advance by at most 40 degrees per sample. Both computations call the same
pure limiter and fixed configuration.

The existing 0.27-second simulation rotation spring remains downstream of the
capped desired path. The cap bounds the input target; the spring continues to
smooth root application. The experiment does not change pose inertialization,
source rates, 0.2-second responsive prefixes, traversal limiting, or SONIC
tracking.

## Evidence

Reset evidence for `holden-turn-v1` records:

- profile name;
- acceleration `1.5 m/s^2`;
- deceleration `2.0 m/s^2`;
- directional acceleration disabled;
- velocity turn strength disabled; and
- maximum yaw rate `120 deg/s`.

Each generated command preserves the operator-requested heading and reports the
first and last capped heading applied over the source prefix. Responsive traces
carry the same first/last capped headings in MuJoCo coordinates. This permits an
honest comparison of requested, MM-applied, and physical root rotation without
inferring heading from displacement.

Evidence schemas are versioned and exact-key parsers remain fail closed.

## Failure handling

- Unknown profiles fail before server session mutation.
- Non-finite, non-unit, or non-planar headings fail before candidate mutation.
- Invalid fixed configuration fails without output mutation.
- Prediction/application disagreement is a test failure; there is no runtime
  fallback to an uncapped heading.
- A blocked traversal still owns velocity-stop behavior; it does not silently
  snap heading.
- Live validation runs the paired simulator and GEAR process in one isolated
  network namespace. This prevents unrelated domain-0 Unitree DDS controllers
  from contaminating the experiment.

## Verification

Implementation is test-first and must prove:

1. A 180-degree request advances by exactly 4.8 degrees per 0.04-second step.
2. A 180-degree reversal takes at least 1.5 seconds at the fixed cap.
3. Small turns below the per-step bound pass through exactly.
4. Shortest-path wraparound and the plus/minus-pi tie are deterministic.
5. Output stays finite and unit; invalid input does not mutate output.
6. Future headings advance monotonically at the same cap from a local copy.
7. Runtime prediction, application, and first/last evidence agree.
8. Reset, clone, swap, abort, commit, regeneration, and forced downstream
   failure preserve transaction semantics.
9. The force-search behavior remains effective under sustained capped rotation.
10. `raw` and `holden-v1` retain bit-identical protected outputs.
11. Protocol, client, CLI, and versioned evidence accept exactly the three
    profiles and report the fixed contract.
12. A live isolated A/B/C compares raw, velocity-only `holden-v1`, and
    `holden-turn-v1` using abrupt W/S and A/D reversals.

## Success criterion

The experiment succeeds only if protected parity and contract gates pass and a
live isolated run proves capped requested/applied headings remain within
120 degrees/second. The operator trial must complete three abrupt W/S cycles and
three abrupt A/D cycles, holding each direction long enough to reach its target,
without a fall reset, GEAR safety stop, simulator time reset, or process crash.
The user must judge the reversal behavior clearly better than velocity-only
`holden-v1`; an assessment of unchanged or worse rejects the hypothesis without
retuning in the same experiment.
