# G1 Terrain Motion Quality Oracle Design

## Status

Approved diagnostic and correction strategy for the second kinematic terrain
motion-matching pass. This design supersedes choosing contact warping or IK as
the next assumed solution. The multi-horizon terrain-skill matcher remains the
frozen v0 baseline.

Sonic, tracking, physics stepping, depth inference, and real-time latency are
out of scope. All experiments use privileged query height maps and 50 Hz G1
kinematics in MuJoCo.

## Current-System Assessment

The v0 system is a useful root-motion and terrain-compatibility baseline:

- all six frozen routes execute without exception;
- five satisfy their coarse behavioral outcome contract;
- moving-command stalls are at most three frames;
- turning and command reversals are responsive; and
- the matcher searches 379,404 rows from 760 clips through 48,327 indexed
  terrain-horizon records.

It is not yet a successful footstep matcher:

- only 24 of 1,000 foot samples on the mixed route are classified as stance;
- the other frozen routes classify only 14.6--28.7 percent of foot samples as
  stance;
- 10 of 67 selected chunks contain no unload-and-land step anywhere in the
  committed chunk;
- 19 of 67 chunks do not unload a foot in their first 15 frames;
- 73--93 percent of measured stance slide occurs within the first 15 frames
  after transitions; and
- the selected native source windows contain substantial support-labelled
  horizontal foot travel before composition.

The current stance metric has a critical blind spot. It infers stance from
emitted clearance and vertical speed, then measures errors only on inferred
stance frames. A floating foot becomes `not stance`, removing the failure from
support-height and slide metrics. The apparently good support-height p95 is
therefore not sufficient evidence of contact quality.

The current height target is also root-centric. It samples the query surface
along the predicted root path and broadcasts the same vertical change to both
feet. It cannot describe a split-height stance or identify a specific landing
foothold.

## Question

Before changing the matcher, determine which boundary limits quality:

1. the source corpus or contact labels;
2. candidate inventory and ranking;
3. rigid placement and transition composition; or
4. the finite clip/action representation itself.

The next implementation must answer this question with frozen evidence. It
must not select IK, warping, additional data, or learned synthesis in advance.

## Approaches Under Test

### Contact-conditioned frame matching

Retain ordinary motion matching but require compatible support state, next
swing foot, and a natural unload-and-land event. This is the smallest search
correction. It may still splice incompatible steps or miss longer terrain
maneuvers.

### Landing-aware contact-phase matching

Search complete support-to-support phases or two-step coherent windows. The
height map and command define feasible landing regions; candidates contribute
their natural landing foot, time, position, and full-body motion. This is the
recommended retrieval representation to test first, but it is not assumed to
win.

### Longer-horizon synthesis

Jointly select several contact phases, or later train a PFNN-style generator.
This has the highest potential ceiling but should be considered only if no
good natural candidate or transition exists under exhaustive retrieval.

Bounded IK or contact warping is orthogonal to these approaches. It may correct
small residual errors only after an unwarped candidate already passes the
motion-quality oracle.

## Phase 1: Truthful Contact Evaluation

Extend the evaluator without changing matcher output. Retain existing progress,
heading, penetration, stall, and outcome metrics, then add:

- source-support versus emitted-contact agreement;
- expected-stance floating fraction per foot;
- maximum consecutive no-contact duration;
- unload, touchdown, and complete-step counts;
- complete steps per metre under moving commands;
- delay from a moving command to the next unload and touchdown;
- horizontal drift during source-labelled stance even when emitted geometry
  loses contact;
- transition-neighborhood and steady-source contact errors;
- native source-window stance slip and support-height error; and
- per-chunk contact sequences and landing events.

Calibrate contact thresholds from native GRAIL distributions rather than
choosing arbitrary acceptance values. A route cannot pass visual-quality gates
merely by producing few inferred stance samples.

## Phase 2: Frozen State Corpus

Capture deterministic matcher states immediately before the difficult actions:

- flat approach to the first riser;
- first ascent step;
- split-height double support;
- lateral cross-tread travel in both directions;
- diagonal ascent and descent;
- 45-, 90-, and 180-degree turns on a tread;
- reversal on a riser;
- lower and upper side exits; and
- stop/restart while elevated.

Each state stores current qpos, root and foot velocities, emitted and source
contact state, command trajectory, local height-map crop, current source
identity, and the following route outcome. State identities and arrays are
hashed; timing is excluded.

## Phase 3: Exhaustive Motion Quality Oracle

For each frozen state, enumerate the existing corpus without online shortlist
limits. The search units are complete steps, not arbitrary frame chunks:

- every support-to-support contact phase;
- alternate contact-consistent entry phases within existing clips;
- two consecutive alternating steps when available;
- all stair and curb motions; and
- exact sagittal mirrors when their FK identity is verified.

Every action descriptor contains:

- entry and terminal support masks;
- unload foot and frame;
- landing foot and frame;
- natural landing XY and height relative to entry;
- swing clearance and penetration profile;
- stance-foot native slip;
- root displacement and yaw;
- full-body entry pose and velocity; and
- source clip/frame identity.

The height map produces a landing *region*, not one mandatory point. Regions
are query-surface areas that satisfy reach, edge-margin, height, command-path,
and foot-separation constraints. Candidate motion supplies the precise natural
landing. The oracle rejects actions whose natural landing is outside every
feasible region.

For each state, preserve these ranked sets separately:

1. best natural contact/landing match regardless of transition continuity;
2. best transition-continuous match regardless of longer outcome;
3. best joint contact, continuity, and command match; and
4. best two-step coherent match.

Render the top five and the best feasible candidate from each set as native,
rigidly placed, and fully composed MuJoCo kinematics. Do not use IK or warping.

## Phase 4: Boundary Decomposition

Compare the same candidate at three boundaries:

1. **Native source:** source qpos on its recorded terrain.
2. **Rigid placement:** source motion transformed into the query scene without
   inertialization.
3. **Composed output:** exact state produced by the current transition composer.

Measure contact agreement, landing-region error, stance drift, swing clearance,
penetration, joint position/velocity boundary jumps, root jerk, and command
progress at every boundary. This isolates whether quality is already absent in
the data, lost during placement, or lost during blending.

## Decision Rules

The oracle chooses the correction category; it does not choose production
motion directly.

### Search correction

If a good native and composed candidate exists but current search misses it,
expand the index to complete contact phases and add next-foot, natural landing,
contact timing, native-slip, and full-body continuity terms. Re-run the frozen
states before closed-loop routes.

### Composition correction

If the selected candidate is good natively and after rigid placement but fails
after composition, add emitted-window contact validation and replace independent
root/joint blending with contact-preserving composition. Measure a residual
correction envelope; only then compare small IK cleanup against no IK.

### Corpus correction

If no acceptable native action exists for a state, report the missing bin by
travel direction, turn, leading foot, height change, and landing geometry. Add
only motions targeting those bins. Do not enlarge the corpus indiscriminately.

### Representation correction

If individual native actions exist but no rigidly connected sequence survives,
the finite action graph is insufficient. Compare longer coherent source windows
and multi-step search. A PFNN-style or other learned generator becomes justified
only after this evidence, using the oracle as a teacher and evaluation target.

## Acceptance for Choosing a Correction

The diagnostic passes when:

- every frozen state receives a deterministic exhaustive-search result;
- native, placed, and composed measurements are stored separately;
- top candidate videos/contact sheets are generated automatically;
- each failure is classified as source, search, placement, composition, corpus,
  or representation with supporting metrics;
- no IK, warping, or learned synthesis affects the oracle; and
- repeating the run reproduces all non-timing hashes.

Only after classification may one correction arm be implemented. That arm must
then improve contact quality without reducing the v0 baseline's route execution,
command responsiveness, or coarse behavioral coverage.

## Scope Boundaries

This design does not connect Sonic, train a policy, introduce dynamics, modify
the corpus, or admit procedural IK. It builds the evidence required to choose
among those later actions. Existing unrelated landing-bridge and contact-oracle
working changes remain untouched.
