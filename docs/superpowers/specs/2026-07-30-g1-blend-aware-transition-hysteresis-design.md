# G1 Blend-Aware Transition Hysteresis Design

Date: 2026-07-30

Status: Approved

Branch: `research/g1-torch-terrain-kinematics`

## Purpose

Remove visible stuttering from the 50 Hz Torch terrain motion matcher without
giving up the stair-clearance, progress, height, and upper-landing results of
the retained terrain-preview candidate.

This is the next isolated hypothesis in the approved terrain motion-quality
loop. It remains privileged-height and kinematic. It does not add SONIC
tracking, physics, depth inference, or learned control.

## Evidence

The retained nine-second dense rollout accepts 36 transitions. Of the 35
transition intervals after initialization:

- 24 are at most `0.20 s`;
- the median is `0.10 s`;
- several are only one `0.02 s` frame apart; and
- a burst near the middle of the stair repeatedly alternates between source
  clips and source frames.

Each accepted transition resets the inertial offset age to one frame. Joint
acceleration at transition frames has a median of about `100 rad/s^2`, compared
with about `14 rad/s^2` over the full rollout. Root and joint jerk are also
concentrated around transition bursts. The observed stutter therefore matches
restarting an unfinished blend, not slow search or a rendering artifact.

## Design

The matcher continues to search every 50 Hz frame. Candidate eligibility,
feature normalization, exact squared-L2 costs, terrain preview, source
exclusion, and emitted output remain unchanged.

While an accepted transition's inertialization is still settling, a new
non-incumbent candidate receives an additional transition cost. The additional
cost:

- is configured separately from the existing base transition penalty;
- is maximal immediately after a transition;
- decays monotonically to zero over a configured settle duration;
- never applies to the incumbent continuation;
- is disabled when its configured magnitude or duration is zero; and
- changes selection only, never the reported raw motion or terrain feature
  cost.

The first implementation uses linear decay:

`active_penalty = magnitude * max(0, 1 - blend_age / settle_duration)`

and adds `active_penalty` to the existing transition penalty for all
non-incumbent rows. The matcher's already-authoritative inertial offset age is
the blend age, so no independent cooldown clock is introduced.

This is adaptive hysteresis rather than a hard lockout: a sufficiently better
candidate can still win during the settle interval. Command-transition and
clip-end searches remain forced exactly as before; forced search means search
now, not bypass all matching costs.

## Terrain Safety

The existing emitted-window terrain validator remains authoritative and runs
after selection. A selected transition that is unsafe still falls back
transactionally to its safe incumbent. An unsafe incumbent still fails closed.

The hysteresis parameters must be tuned only among candidates that retain:

- at least 90% of reference horizontal progress;
- at least 80% of reference root-height gain;
- the upper landing;
- at least `-0.03 m` minimum MuJoCo-FK ankle-origin clearance; and
- zero regression in the existing deterministic and flat-control tests.

## Stutter Evidence

Each deterministic candidate reports:

- accepted and rejected transition counts;
- transition-interval distribution and number of intervals at most `0.20 s`;
- same-clip source jumps and cross-clip jumps;
- joint and root acceleration and jerk at transition versus ordinary frames;
- contact-foot planar velocity as a skating diagnostic; and
- the existing terrain-quality metrics.

The initial target is to eliminate consecutive-frame transition bursts and
materially reduce transition-associated acceleration and jerk. Transition count
alone is not sufficient: the viewer remains the qualitative acceptance check.

## Tests

Focused unit tests prove:

1. zero hysteresis exactly preserves existing selection;
2. an unfinished blend adds the configured decayed penalty only to
   non-incumbent candidates;
3. the penalty is zero at and after the settle duration;
4. a substantially better candidate can still transition during settling;
5. continuation remains unpenalized; and
6. parameter validation rejects negative or non-finite values.

After focused tests, run the full Torch matcher suite and frozen stair rollout.
Retain a parameter candidate only if all safety gates pass and its stutter
diagnostics improve.

## Iteration

Start with a settle duration of `0.20 s`, twice the configured `0.10 s`
inertialization half-life. Sweep only the hysteresis magnitude while holding all
other settings fixed. If no magnitude removes transition bursts without losing
terrain quality, reject this hypothesis and proceed to contact/gait-phase
continuity rather than accumulating more transition heuristics.

