# G1 Terrain Motion-Quality Loop Design

Date: 2026-07-30

Status: Approved iterative direction

Branch: `research/g1-torch-terrain-kinematics`

## Purpose

Improve the native 50 Hz Torch motion matcher's stair kinematics until the
frozen flat-control and stair-quality gates pass. Motion quality is the
objective. Wall-clock search latency remains measured for information but is
not an acceptance criterion: the kinematic viewer may pause while an exhaustive
match is computed.

This loop remains entirely privileged-height and kinematic. It does not add
SONIC tracking, physics integration, depth inference, or a learned model.

## Diagnosed Baseline

The five-clip stair slice and its query scene remain frozen. The current dense
91-value terrain group reaches the upper landing, but its minimum foot clearance
is `-0.48198 m`, so it is not a usable kinematic reference.

The database currently implements each feature-group weight as a per-component
weight and then sums squared residuals. A 91-component dense terrain group
therefore contributes roughly in proportion to 91 while a 27-component motion
query contributes roughly in proportion to 27. A diagnostic-only
dimension-invariant normalization, applied to the terrain extension without
changing the seven established flat-motion groups, produced:

- minimum foot clearance `-0.13665 m`;
- integrated penetration `0.07061 m*s`, 78% below the dense baseline;
- maximum progress `1.9360 m`, 96% of the recorded reference;
- root-height gain `0.60834 m`; and
- a successful upper-landing result.

This is a strong improvement but not a pass. Raising the transition penalty
from `0.1` to `10.0` reduced transitions from 27 to 13 while worsening both
minimum clearance and penetration, so transition count is not itself the
objective.

At the remaining worst frame, the selected source pose clears its own recorded
terrain and the newly selected raw target also clears the query stair. The
emitted pose still penetrates because its inertial offsets preserve the
preceding foot pose. MuJoCo forward kinematics agrees with the matcher's body
diagnostic within millimetres, so the penetration is real rather than a metric
artifact. The preceding continued frames had already moved the left foot under
the next tread before the next scheduled search.

## Frozen Experimental Contract

- Dataset: `build/torch-stair-small`.
- Query scene: `stair/updown-0000/motion.npz`.
- Reset clip: `flat/motion.npz`.
- Representation: native right-handed Z-up, wxyz, 29-joint G1.
- Matcher time step: exactly `0.02 s` because all source clips are 50 Hz.
- Command and duration: the existing deterministic nine-second stair rollout.
- Viewer: native MuJoCo forward kinematics with no `mj_step`.
- Search: exact Torch squared-L2 unless a later documented hypothesis explicitly
  introduces a preview or lookahead cost.
- Flat behavior: existing 27-dimensional flat feature normalization and search
  semantics remain numerically unchanged.

## Quality Gates

Every retained candidate must be evaluated on the same deterministic rollout.
The dense condition must:

- select a stair clip by the existing pre-riser deadline;
- reach at least 90% of reference horizontal progress;
- reach at least 80% of reference root-height gain;
- reach the upper landing;
- keep minimum MuJoCo-FK ankle-origin clearance at or above `-0.03 m`; and
- improve integrated penetration relative to the last retained candidate.

The existing flat-control identity tests must pass. Search and step times are
reported but cannot reject a motion-quality improvement.

Transition count, source-clip diversity, and source-frame monotonicity are
diagnostics. They become hard constraints only if an observed discontinuity is
shown to cause a failed quality gate.

## Iteration Method

The loop changes one causal mechanism at a time:

1. capture the current deterministic baseline;
2. write a focused failing test for the hypothesized defect;
3. make the smallest implementation change;
4. run focused unit and deterministic stair tests;
5. measure progress, height, clearance, penetration, transitions, source-frame
   jumps, and timing;
6. retain the change only if it passes regressions and improves the
   quality-gate Pareto frontier; otherwise revert that isolated change; and
7. inspect the new worst frame before selecting the next hypothesis.

The viewer is relaunched only from a retained, tested candidate. Visual
inspection supplements but does not replace the deterministic gates.

## Hypothesis Order

### 1. Dimension-invariant extension weighting

Interpret an extension weight as the total importance of that feature group,
not as a multiplier repeated once per component. Scale only extension groups by
the square root of their dimension. This retains the established 27-feature
flat matcher exactly while making a four-height and a 91-height extension
comparable at the same declared group weight.

### 2. Search cadence versus terrain discontinuities

With the weighting defect removed, compare exact search every five, two, and one
50 Hz frames. The observed failure develops between scheduled searches at a
stair edge, and compute speed is explicitly not a gate. Retain a shorter cadence
only if it materially improves clearance without losing progress or height.

### 3. Terrain-aware transition preview

If exhaustive per-frame search still selects a pose whose inertialized emitted
state penetrates, preview the actual emitted candidate rather than evaluating
only its raw database feature. Candidate evaluation must remain side-effect
free. A clearance term or eligibility mask must use the same query height grid
and the emitted root/joint/body state used by the viewer.

### 4. Contact phase and source continuity

If preview alone allows visibly poor jumps, add the smallest demonstrated
continuity signal: contact phase first, then a penalty for backward jumps within
the same clip. Do not add both together without separate ablations. Eligibility
must preserve a valid incumbent and must fail explicitly when no candidate is
available.

### 5. Multi-frame terrain lookahead

If single-frame preview passes locally but later continuation fails, score a
short emitted window against the query terrain. Window length and aggregation
must be explicit and tested. This is preferred over undocumented action
chunking because it preserves 50 Hz output while allowing slower deliberation.

## Test and Evidence Requirements

- A unit test proves equal declared extension weights have
  dimension-invariant expected group cost.
- Existing flat normalized features remain bitwise identical.
- Search-cadence and candidate-preview tests use small synthetic databases
  before running the protected stair corpus.
- Deterministic stair evidence includes selected clip/frame traces and the
  exact worst-clearance frame.
- Foot clearance is cross-checked against MuJoCo forward kinematics for the
  displayed qpos; the three-body matcher diagnostic cannot be accepted on faith.
- Results for rejected hypotheses remain documented so the loop does not repeat
  failed tuning.

## Non-Goals

- Achieving a real-time search budget.
- Training from depth or proprioception.
- Root-state estimation.
- SONIC policy inference or physical balance.
- General obstacle avoidance.
- Expanding beyond the frozen stair slice before this slice passes.

## Exit

The loop exits successfully only when all quality gates and regressions pass and
the retained matcher is launched in the native MuJoCo viewer for user
inspection. If the frozen corpus cannot satisfy the gates after candidate
preview, contact continuity, and short-window lookahead, the result is a
documented data-coverage failure rather than a relaxed threshold.
