# G1 Full-Pose Transition Continuity Design

Date: 2026-07-30

Status: Approved

Branch: `research/g1-torch-terrain-kinematics`

## Purpose

Improve stair descent quality in the 50 Hz Torch terrain motion matcher without
regressing the retained ascent, terrain clearance, or transactional safety
behavior.

This remains a privileged-height, kinematic experiment. It does not add SONIC
tracking, physics, depth inference, terrain IK, or new motion clips. Curb and
omnidirectional motion expansion follow after this matcher defect is isolated.

## Reproduction and Root Cause

The deterministic diagnostic sequence runs:

1. 280 frames commanded up the authenticated stair;
2. 100 frames reversing and turning on the upper landing; and
3. 260 frames commanded down to the lower landing.

The current matcher completes all 640 frames without a safety rescue or crash.
Minimum matcher-foot clearance is `+0.01238 m` during the reversal and
`+0.02525 m` during descent. Descent is therefore geometrically accepted but
visibly and numerically rougher than ascent.

| Diagnostic | Ascent | Descent |
|---|---:|---:|
| Joint acceleration p95 (`rad/s^2`) | 235.23 | 284.91 |
| Joint jerk p95 (`rad/s^3`) | 13,116.19 | 20,767.56 |
| Root jerk p95 (`m/s^3`) | 610.94 | 1,572.00 |
| Transition-neighborhood joint jerk p95 (`rad/s^3`) | 23,151.8 | 27,280.3 |
| Transition-neighborhood maximum joint jerk (`rad/s^3`) | 39,185.5 | 96,726.0 |
| Accepted transitions | 11 | 11 |
| Cross-clip transitions | 7 | 7 |

The worst descent event occurs on the lower landing when the matcher jumps
between flat source frames. The selected raw source has a `3.128 rad` joint
position gap and a `16.867 rad/s` joint velocity gap from the currently emitted
state. Across descent transitions, raw joint velocity gap and nearby maximum
joint jerk have correlation `0.937`; joint position gap and jerk have
correlation `0.714`.

The existing 27 motion features contain both feet relative to the pelvis, both
foot velocities, pelvis velocity, future trajectory positions, and future
facing. They do not contain the remaining joint pose or joint velocities.
Terrain features can therefore make a candidate geometrically relevant while
the matcher remains blind to a large full-body transition discontinuity.
Holden inertialization preserves positional continuity, but a large offset and
velocity mismatch still creates the observed acceleration and jerk while the
offset decays.

## Chosen Approach

Add one state-dependent, full-pose transition-continuity cost to exact search.
For every eligible database row, compare the raw source joint state at that row
with the matcher's current emitted state:

`C_pose = w_q * mean_active(((q_source - q_current) / scale_q)^2)`

`C_velocity = w_v * mean_active(((dq_source - dq_current) / scale_dq)^2)`

`C_continuity = C_pose + C_velocity`

The scales are computed once from the complete row-aligned motion database.
Each component uses its database standard deviation. A component whose
standard deviation is zero or non-finite is excluded from that group rather
than assigned an arbitrary small denominator. At least one active component is
required in each enabled group. Taking the mean over active components makes
each configured group weight independent of the 29-joint dimension.

The cost applies only when selecting a non-incumbent transition. The incumbent
continuation receives zero continuity, base-transition, and settle cost exactly
as before. This preserves the semantic meaning of those costs: they price
changing source motion, not continuing it.

The exact searched total for a non-incumbent row becomes:

`feature_cost + base_transition_penalty + settle_penalty + C_continuity`

The selected raw motion-feature and terrain-feature diagnostics remain
unchanged. New diagnostics report the selected normalized pose, velocity, and
total continuity components separately.

## Search and Safety Integration

The matcher constructs immutable, row-aligned joint-position and
joint-velocity tensors beside its existing row provenance. Continuity costs are
computed on the database device once per prepared step and passed into exact
selection as one row vector.

The search helper validates that any supplied row cost is:

- a float32 tensor on the database device;
- exactly one-dimensional with one value per database row; and
- finite and non-negative.

Zero or omitted row costs preserve the current selector exactly.

Ordinary exact selection adds the row cost only to non-incumbent eligible
rows. Ranked terrain rescue uses the same pose-continuity vector and base
transition penalty, but still omits the decaying settle penalty so smoothing
cannot prevent a safety rescue. Rescue continues validating candidates in cost
order until it finds a safe emitted window. No continuity setting can bypass,
relax, or replace the terrain validator.

The two-phase prepare/commit boundary remains unchanged. Failed search,
invalid costs, or an all-unsafe rescue leaves matcher state unmodified.

## Configuration

Add two non-negative finite matcher parameters:

- `transition_joint_position_weight`;
- `transition_joint_velocity_weight`.

Both default to zero. The zero/default configuration must preserve current
flat and terrain selection exactly. The retained experiment config records the
qualified nonzero values explicitly.

No descent detector, slope mode, clip label, or direction-specific multiplier
is introduced. The missing continuity signal is general and must work for
ascent, descent, flat locomotion, and later curb/lateral motion.

## Benchmark and Weight Selection

Keep the authenticated dataset, terrain feature weight `4.0`, inertialization
half-life `0.10 s`, settle duration `0.20 s`, and settle magnitude `18.75`
fixed. Sweep only the two continuity weights.

The initial bounded sweep is:

- position weight: `0.0`, `0.05`, `0.10`, `0.25`, `0.50`;
- velocity weight: `0.0`, `0.05`, `0.10`, `0.25`, `0.50`.

The zero/zero cell is the reproduced baseline. Reject any cell that fails
terrain or completion gates. Among remaining cells, minimize descent
transition-neighborhood maximum jerk, then descent p95 joint jerk, then ascent
p95 joint jerk. Prefer the smaller total weight on ties.

The diagnostic reports ascent, reversal, and descent independently. Aggregate
values may be shown but cannot determine acceptance.

## Acceptance Gates

A retained setting must satisfy all of the following:

1. complete all 640 scripted frames without an exception;
2. descend back to a root height no greater than `0.82 m`;
3. retain minimum emitted matcher-foot clearance of at least `-0.03 m`;
4. pass the existing five-gate deterministic dense stair acceptance suite;
5. pass the full MuJoCo-FK ankle-origin clearance sweep at `-0.03 m` or better;
6. keep descent accepted-transition count at or below the baseline 11;
7. reduce descent p95 joint jerk by at least 20%, to at most
   `16,614.05 rad/s^3`;
8. reduce descent transition-neighborhood maximum joint jerk by at least 40%,
   to at most `58,035.6 rad/s^3`;
9. keep ascent p95 joint jerk no more than 10% above baseline, at most
   `14,427.81 rad/s^3`; and
10. pass the complete Torch regression suite, including exact zero-weight
    equivalence.

Root jerk, acceleration, contact-foot speed, transition intervals, rescue
ranks, and visual quality remain reported diagnostics. They break ties and
identify regressions, but do not replace the frozen gates.

After deterministic qualification, the interactive viewer must survive an
up/turn/down user sequence and the user must confirm that descent is visibly
smoother. The viewer is not called stable before that check.

## Tests

Focused tests prove:

1. row-aligned source pose tensors match clip provenance exactly;
2. normalized pose and velocity continuity costs match an independent oracle;
3. zero-variance joint components are excluded;
4. invalid row-cost shape, dtype, device, sign, or finiteness fails closed;
5. zero or omitted continuity is selection-identical to the current selector;
6. continuity cost never applies to the incumbent;
7. a smoother non-incumbent candidate can beat a feature-closer discontinuous
   candidate;
8. selected component and total continuity diagnostics are exact;
9. ranked safety rescue uses continuity order but still skips every unsafe
   candidate;
10. failed preparation remains transactional; and
11. rollout and historical-viewer artifacts carry backward-compatible
    continuity diagnostics.

After focused tests, run the complete Torch suite, the 25-cell deterministic
up/turn/down sweep, flat/legacy/dense CUDA qualification, authoritative MuJoCo
FK, and the live viewer check.

## Alternatives Rejected

### Descent-specific hysteresis

A larger transition penalty while descending could reduce switching, but it
does not distinguish a smooth candidate from the observed `16.867 rad/s`
velocity mismatch. It may also delay a necessary transition onto or off the
stairs. The root cause is candidate ranking, not missing knowledge that descent
is occurring.

### Expand the descent inventory first

More descent motion will eventually improve coverage, and curb plus
omnidirectional clips are already planned. The present database nevertheless
contains multiple safe descent and flat candidates. Adding clips before
scoring full-pose compatibility would enlarge the set in which a
feature-compatible but discontinuous candidate can win. Inventory expansion
therefore follows this matcher correction.

### Modify the frozen 27-feature schema

Appending all joint states to the normalized database feature would also expose
continuity, but it would make pose similarity part of incumbent raw feature
cost and require recertifying the shared feature contract. A transition-only
cost expresses the intended behavior directly, defaults off cleanly, and
leaves query/database terrain features unchanged.
