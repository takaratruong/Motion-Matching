# G1 Foothold-Conditioned Terrain Motion Matching Design

**Date:** 2026-07-31  
**Branch:** `research/g1-torch-terrain-kinematics`  
**Scope:** privileged-heightmap, 50 Hz, kinematic G1 locomotion only. Sonic,
tracking, physics, depth inference, and real-time latency are excluded.

## Goal

Determine within one bounded research sprint which terrain-selection architecture
best supports interactive omnidirectional motion on the same staircase. The
required behaviors are approach without premature stair gait, ascent, descent,
lateral and diagonal traversal, W-to-S reversal on a tread, 90/180-degree
direction changes, and exits from either side.

The experiment starts with the literature-backed hypothesis that terrain
contacts are discrete constraints and should not be traded against ordinary
pose or trajectory residuals in one continuous nearest-neighbor cost.

## Diagnosed baseline

The current dense terrain query contains 91 root-yaw-relative height samples and
four commanded-path height samples. It recognizes nearby geometry but does not
state which foot lands next, where or when it lands, or which tread supports
each foot. Exact FK/contact validation occurs only after continuous ranking.

Prior work has already tested or falsified dimension weighting, denser search,
command-path features, transition previews, phase penalties, larger stair and
curb corpora, rigid support placement, locally retargeted contact segments,
bounded transition search, and an IK landing bridge. These mechanisms improved
subsets of routes but did not provide reliable omnidirectional interaction.
They will not be repeated as unmeasured tuning.

## Shared contact-action representation

Each authenticated source action is represented from one replanning contact
boundary through at least the next two landing events, where available. Its
immutable descriptor contains:

- starting left/right support topology and per-foot source surface heights;
- entering/next-foot identity;
- first and second landing frame offsets;
- each landing ankle XY relative to the starting support frame;
- each landing surface-height change relative to the starting support surface;
- root displacement and facing change at each landing;
- minimum swing-foot clearance and maximum unsupported run; and
- the exact source clip/frame interval used for sequential playback.

Descriptors come only from authenticated source FK and source height grids.
Malformed, penetrating, unsupported, or out-of-domain actions are excluded.

## Query foothold plan

At a replanning boundary, construct a short geometric plan from the emitted FK
state, command, and authoritative query heightmap:

1. Record the current support-foot XY, surface height, and left/right topology.
2. Extend the commanded travel direction over a bounded reachable step region.
3. Enumerate stable landing samples on the heightmap for the next moving foot.
4. Reject samples beyond source-observed step length, lateral reach, vertical
   change, edge margin, or swing-clearance limits.
5. Score feasible samples by command progress, direction error, step regularity,
   and distance from terrain edges.
6. Retain a small deterministic beam for the next two alternating contacts.

The plan describes desired contacts; it does not procedurally modify joints or
invent motion. If no reachable foothold exists, the result is an explicit
coverage failure.

The first changed-height contact is constrained by actual distance to the
terrain discontinuity. This prevents choosing an elevated stair action while
both planned contacts remain on the approach floor.

## Architectures compared

### A. Discrete first-contact filter

Hard-filter source actions by current support topology, next-foot identity,
first landing direction, first landing height delta, and first landing timing.
Run the retained continuous motion-matching cost inside the feasible set.

This is the smallest test of the diagnosis and should eliminate premature
stepping. It may remain ambiguous during turns because it reasons about only one
future landing.

### B. Two-contact action-graph search

Match the query's two-contact beam against source action descriptors. An edge is
valid only when its ending support topology can start the next action and the
complete transformed source profile passes exact query-height and FK validation.
Rank complete two-contact paths by foothold error, command progress, pose
continuity, and terminal velocity/facing error.

This tests whether explicit short-horizon planning solves reversals and turning
on a tread. Search time is diagnostic, not an acceptance gate.

### C. Hybrid foothold-conditioned motion matching (recommended)

Use B to produce the discrete feasible action set, then use ordinary normalized
pose, velocity, and command-trajectory cost to choose among feasible actions.
Commit only to the next contact boundary, re-observe the heightmap and command,
then replan. This preserves motion-matching continuity without allowing it to
trade away contact feasibility.

### D. Continuous feature-only control

Append landing descriptors to the existing continuous feature vector without a
hard filter. This is included as a negative-control ablation. It is expected to
repeat the failure mode where sufficiently good pose/trajectory similarity
outweighs a wrong contact sequence.

PFNN or Learned Motion Matching is not a four-hour implementation candidate.
After an oracle works, its qualified action traces can become targets for a
learned stepper or DAgger-style distillation. Training a model before the
contact oracle is correct would hide rather than diagnose coverage holes.

## Command semantics

The retained live control convention is used for this sprint: WASD specifies
world travel direction and body heading follows that direction. Therefore W-to-S
is a requested 180-degree turnaround, while W-to-A/D requests a 90-degree
direction change. A later design may decouple travel and facing.

## Evaluation protocol

All arms use the same corpus, normalization, query staircase, start states,
commands, exact FK, and route evaluator. No viewer is launched until an arm
passes automated qualification.

The adversarial suite must include:

- flat approach with a measured first-riser boundary;
- straight ascent and descent;
- lateral traversal in both directions;
- diagonal ascent and descent in all applicable directions;
- W-to-S reversal on lower, middle, and upper treads;
- 90-degree direction changes on a tread;
- 180-degree turnaround on a tread;
- side exits from lower and upper treads; and
- rapid mixed commands with stops and restarts.

Each route records progress, final surface, first elevated-contact timing,
support loss, penetration, stance slide, command-response delay, action
coverage, rejection reason, selected contact sequence, and deterministic output
identity.

## Acceptance and selection

An arm is not retained merely because aggregate route count improves. It must:

- never begin an elevated stair contact before the first reachable riser;
- preserve at least one supported foot except for the existing bounded source
  flight allowance;
- keep ankle penetration at or above the existing `-0.03 m` bound;
- make commanded progress without a motion-selection freeze;
- retain ordinary ascent and flat locomotion; and
- improve at least one reversal, cross-tread, or exit class without regressing
  another already-qualified class.

The sprint winner is selected lexicographically by safety violations, route
classes passed, freeze rate, contact timing error, stance slide, and pose
continuity. Runtime is reported last.

If A fixes approach timing but not turning and B fixes turning with worse pose
continuity, C is the planned merge. If no arm has descriptor coverage for a
required query plan, the output must identify the missing support/landing
signature and source motions needed; thresholds are not weakened to fabricate a
pass.

## Test strategy

Implementation is test-first:

1. Synthetic support traces freeze two-contact descriptor extraction.
2. Synthetic heightmaps freeze edge-aware foothold enumeration and prove the
   planner cannot step up before the riser is reachable.
3. Candidate-filter tests prove a wrong next foot, surface delta, timing, or
   topology is ineligible regardless of continuous feature cost.
4. Action-graph tests prove alternating-contact connectivity and deterministic
   two-contact selection.
5. Transactional matcher tests prove rejected plans leave state unchanged and
   accepted actions play sequentially only to the next contact boundary.
6. Real-data guarded tests report descriptor coverage before behavioral runs.
7. The full adversarial matrix selects the retained arm before live feedback.

## Failure handling

Every query fails closed. No foothold, no matching descriptor, invalid
transformation, terrain penetration, support loss, or out-of-domain sample
rejects that action. The matcher may retain a safe standing/contact state while
reporting the exact failure, but it may not silently choose an ordinary stair
gait, warp feet, or slide the root through the terrain.
