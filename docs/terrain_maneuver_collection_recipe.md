# Terrain maneuver collection recipe

## Goal

Build an approximately 200-clip, 11-second clean kinematic bank that contains
the awkward joystick transitions a distilled policy must survive on terrain,
then fine-tune SONIC to track that bank and collect the usual noised physical
rollouts.  Global root position is allowed while creating and auditing the
offline ceiling.  It is not a policy input at deployment: the eventual runtime
interface remains two local joysticks plus robot-centric perceived terrain.

## Current maneuver operator

The operator starts from a motion that is already grounded on its exact terrain.
It finds a low-speed, supported phase within the terrain traversal and changes
only time along that motion:

1. follow the native motion;
2. decelerate with a quintic timing curve into the supported pose;
3. hold that exact pose;
4. either restart forward or retrace the source motion in reverse.

Root translation and joints use fractional linear interpolation; root rotation
uses quaternion SLERP.  This is deliberately more conservative than spatially
warping a flat walk over a surface.  It preserves the original feet/terrain
relationship and produces clean labels for stop, restart, and reversal.  More
expressive maneuvers should be added as additional grounded source snippets or
with a contact-constrained whole-body solver, not by lifting the root alone.

## Admission gates

A generated clip is eligible for visual review only when all of these pass:

- complete-G1 terrain audit: at most 5 mm foot penetration and no forbidden
  body penetration;
- planted-foot temporal motion: at most 3 mm per frame and 12 mm over a
  contiguous stance run;
- both feet at the inserted hold are within 8 mm of support;
- stance drift is no more than the source motion plus the small interpolation
  allowance;
- mechanics remain within the existing G1 step/acceleration gate.

The whole video and dense 25 fps windows around both hold boundaries are then
reviewed for sliding, cadence hitches, hovering, pose snaps, and unnatural
support.  Numeric acceptance is not visual acceptance.

The root-only procedural rolling-bump, smooth-hill, and cross-slope pilots are
explicitly rejected.  They avoided penetration but close support measurement
found 14--30 mm p95 and 34--48 mm maximum two-foot hover.  They are useful test
terrains, not usable motion data.

## Current evidence

Exact-grounded stop/restart and reversal pilots now exist on two stair courses
and twelve distinct real slope/hill traversals.  Twenty-eight pilots pass the automatic
kinematic gate.  The four newly added C490 slopes use the archive-recorded
minus-90-degree terrain rotation; dense 50 Hz boundary review shows no mesh
teleport, foot-through-surface event, or pose snap.  The two uphill
stop/restart clips pause in a conspicuously wide split stance and are tagged as
robustness examples rather than showcase motion.  The downhill pairs and the
uphill reversal boundaries are visually cleaner.

Two earlier varied-slope pilots were audited with an identity terrain rotation
even though their motions used the archive-rotated world frame.  Their renders
and the corresponding SONIC slope probe are invalidated and must not be counted
as evidence.  Direct-source terrain transforms now default to an exact archive
lookup rather than identity.  The independent stair probe remains valid and
shows that the unmodified `terrain_release` checkpoint diverges before the new
reversal.  Therefore the current checkpoint is a baseline, not a collection
engine for this bank.  Fine-tuning on the clean bank is required before noisy
rollout collection.

The current fine-tune is followed automatically by a deterministic physical
rollout of every one of the 28 admitted clips.  Diagnostic failures are saved
for review but cannot enter the training corpus.  A stratified visual set spans
both stair courses, the two clean registered downhill sources, and four of the
rough/hill/slope profiles.  No noisy collection is chained after this gate.

The first deterministic physical gate completed 19/28 clips.  In particular,
12/14 newly scaled rough/hill/slope clips survived, while all four older stair
pilots and both older composed-slope pilots failed.  Three reversal clips also
failed.  This is useful trackability evidence but not a collection pass.  The
next tracker branch upweights those nine exact failures, then must re-run all
28 clips so success cannot be purchased by forgetting the current passes.

The first varied-terrain montage also exposed a presentation and data-quality
distinction.  Hard cuts between its five review windows are not motion
teleports, but freezing a mid-stride support pose for 30 frames is visibly
unnatural.  Reversal candidates now use a shorter 16/4/16-frame timing and the
seven varied-terrain sources still pass the exact gates.  Natural sustained
stops should come from real start/stop motion, not a long frozen walking pose.
The full v2 bundle preserves all 28 v1 clip IDs, replacing only the seven
scaled source pairs.  Full-bank evaluation can therefore compare each clip
directly and detect both improvement and forgetting.

A CPU-only scale-up script tested twelve additional C490 clips stratified across
grade and round-trip structure.  Seven source clips produced fourteen admitted
pilots; three had no trustworthy supported pivot and two failed the source
mechanics gate.  The admitted profiles range from shallow roughness through
double hills, crest/valley and plateau routes to a sustained 46 cm climb.  All
fourteen pass dense boundary review without a terrain-relative teleport,
surface penetration, or pose discontinuity.  Eight are cleaner visual examples;
six use a support pose wider than 40 cm and are tagged as robustness data.  The
scale-up used no GPU.

A follow-up sweep targets terrain diversity explicitly.  It selects twelve
previously unused clean C490 sources from their registered terrain-following
profiles: large rounded hills, multi-crest and valley paths, repeated shallow
bumps, rough crests, and sustained uphill grades.  Eight sources produce
fifteen accepted short-transition pilots.  The accepted maximums are 2.637 mm
foot penetration, 11.003 mm stance-run drift, and 0.127 rad joint step at
50 Hz.  Dense renders of seven representative reversals show no long frozen
pose or cut at the pivot.  Keep this bank separate until the current tracker
gate establishes that the smaller v2 bank remains physically trackable.

Binary USD crates must be inspected as USD stages; searching their compressed
bytes for physics schema names produces false negatives.  Bundle preparation
now verifies rigid-body and collision APIs when USD bindings are available.
The fifteen accepted varied-terrain motions are kept in a separate bundle and
receive their own clean physical gate before any training or noised collection.
Bundle preparation also writes the static object-motion file from the exact
per-clip terrain position and quaternion; do not synthesize identity object
motions for archive-rotated C490 terrain.

Hard-only continuation is not a valid tracker recipe here: it can learn the
failures while catastrophically forgetting the previously tracked motions.
For a bounded repair, keep every passing clip in replay and upweight failures
with stable aliases, then re-run the original full bank.  The first balanced
bank uses one copy of every 28-clip v2 motion plus two extra copies of each of
the eight parent failures (44 clips total).  This improves the deterministic
gate from 20/28 to 24/28 while retaining every parent pass.  A subsequent
four-failure-focused replay regresses to 22/28, so repeated narrowing is not a
safe recipe.

The 24/28 core checkpoint fails all fifteen newly added hill/bump motions when
evaluated without adaptation.  The next balanced bank therefore contains all
28 core motions once and all 15 varied motions twice: 58 replay entries and 43
unique evaluation motions.  At 50 iterations it passes 25/28 core plus 10/15
varied; at 100 iterations it passes 24/28 core plus 12/15 varied.  Prefer the
50-step checkpoint when core fidelity is primary and the 100-step checkpoint
when varied-terrain coverage is primary, but do not collect noisy rollouts from
either until visual review clears the high-MPJPE survivors and the remaining
terrain failures.

Full-length review now separates three different effects that a short montage
can conflate.  The previous hard cuts were presentation cuts; the old long
mid-stride hold was a real kinematic cadence artifact; and the remaining SONIC
failures are gradual physical drift/stumble.  The 16/4/16 reversal timing removes
the long hold while retaining exact terrain registration and the collision gate.
Six full-length clean examples span hills, sustained grades, crest/valley paths,
and repeated bumps in
`artifacts/terrain_maneuver_bank/c490_slope_varied12_v2/varied_reverse6_full_grid.mp4`.
The corresponding labeled physical audit is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/tracker_review_core43_replay58_step100_varied6_cpu_v1/varied_physical6_audit_grid.mp4`.
Reviewed passes contain no teleport or long freeze, but small stance-foot
shuffles and large path errors remain, so the noised-collection gate stays
closed.

One low-rate repair improves coverage without clearing that quality gate.  Its
64-entry replay bank keeps all 43 unique motions, all varied motions twice, and
the three remaining varied failures four times.  At 25 iterations it passes
25/28 core and 13/15 varied (38/43 total), but several reversal survivors still
have 0.6--1.18 m peak MPJPE.  At 50 iterations varied coverage reaches 14/15
while core coverage collapses to 21/28, so that checkpoint is rejected.  Keep
the step-25 model only as the broadest diagnostic tracker; do not use it for
noised collection until the high-error reversals are fixed.

The required full-length visual gate for that checkpoint is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_clean_bank28_v1/tracker_review_core43_replay64_step25_varied6_cpu_v1/step25_physical6_audit_grid.mp4`.
It confirms that the normal stop/reversal survivors have no teleport or long
hold, although a small supported-foot shuffle remains.  It also shows why a
binary pass is insufficient: the orange clip-94 rollout stays upright and is
locally continuous despite reaching 1.181 m peak MPJPE.  Treat high-error
survivors as failed tracking for collection purposes.

## Scale-up strata

Target 200 clips only after a small clean fine-tune proves trackability.  A
reasonable balanced allocation is:

- 40 stair clips: ascent/descent, forward/backward-facing, stop/restart, and
  reversal at several supported phases;
- 35 genuine slopes: uphill, downhill, cross-slope, varied grade and speed;
- 30 curb/platform clips: up, down, pause, retreat, and landing transitions;
- 35 mixed transitions: flat-to-terrain, terrain-to-flat, and consecutive
  obstacle changes;
- 40 abrupt joystick clips: stop/start, forward-to-backward, travel-direction
  changes, facing changes, and combined two-stick changes while supported;
- up to 20 rolling/rough/bump clips, but only after genuine supporting motion
  or a contact-constrained solver passes the same gates.  Until then, reassign
  these slots to stairs/slopes rather than admitting hovering data.

Mirror only geometrically mirrorable clips, including the full motion history,
commands, contacts, terrain transform, and terrain observation.  Keep the
original/mirror pair identifier so a later audit can compare them exactly.

## Stairs500 execution bank

The stairs-specific scale-up is now materialized rather than hypothetical.  A
240-source CPU sweep (120 ascent and 120 descent) yields 332 automatically
admitted stop/restart or reversal motions from 174 clean source traversals.
Archive sources that fail the exact complete-G1 mesh audit are discarded.

Training uses a balanced 200-motion subset:

- 50 ascent stop/restart;
- 50 ascent reversal;
- 50 descent stop/restart;
- 50 descent reversal.

The selection spans 115 distinct registered source traversals, 3--8 steps,
0.100--0.238 m risers, 0.230--1.007 m treads, and -40.15 to +34.72 degree
approach offsets.  Its paths are:

- selection: `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_v1.json`;
- SONIC bundle: `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_bundle_v1`;
- clean review: `/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_kinematic_review_v1/selected200_kinematic_review8_grid.mp4`.

Use exactly one training environment per bundle motion for the first equal-share
tracker pass.  A 256-environment launch over this 200-motion bank duplicated 56
clips and was quarantined at iteration 75.  The corrected run uses 200
environments and separately evaluates iterations 50, 100, 200, 300, and 400.
Select the checkpoint by deterministic all-motion survival plus peak MPJPE and
visual review, not by training reward alone.

The corrected run selects iteration 400: 157/200 motions survive and 120/200
also stay at or below 300 mm peak MPJPE.  Stop/restart is already strong
(94/100 strict passes), while reversal is the remaining specialist problem
(26/100).  Do not use the original `selected200_step400_physical_review8_v1`
render: it omitted `meta/default_joint_pos` when reconstructing SONIC's
relative joint positions.  The corrected labeled physical review is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/selected200_step400_physical_review8_fixed_v2/step400_physical_review8_fixed_grid.mp4`.

For collection, do not replay the entire 200-motion bundle through iteration
400.  Use `selected120_step400_qualified_bundle_v1`, which contains only that
checkpoint's strict passes.  Train and gate a reversal-only tracker separately,
then pool its admitted noisy rollouts with the universal tracker's admitted
rollouts.  SONIC is an offline data generator here, so family routing is
preferable to admitting falls from a nominally universal checkpoint.

The reversal-only tracker qualifies 28, 36, 48, and 49 motions at iterations
50, 100, 200, and 300.  Keep iteration 300 as the main reversal route and
iteration 200 for its extra uphill cases.  Across all deterministic universal
and specialist gates, 168/200 motions have at least one strict route.  The first
collection tranche avoids duplicates by adding 28 iteration-300 reversals, 11
iteration-200 reversals, and 3 iteration-100 reversals that iteration 400 did
not already qualify.

The first iteration-400 collection retains 236/240 requested rollouts from
118/120 motions, totaling 38.96 minutes at 50 Hz.  Every stored rollout passes
the same failure and 300 mm filter (median 195.5 mm, p90 267.7 mm).  Two
descent-reversal motions produced no acceptable noisy rollout after more than
30 attempts each.  Treat the resulting nonzero collector exit as a partial
quota, not a corrupt dataset: preserve the 236 admitted episodes and exclude
the two torque-fragile motions from this checkpoint's route.

Visual evidence for the retained noisy data is
`/move/data/terrain-aware/sonic-rollouts/terrain_maneuver_stairs500_v1/noised_selected120_step400_review4_fixed_v2/noised_selected120_step400_review4_fixed_grid.mp4`.
It contains one median-error rollout from every ascent/descent x stop/reversal
cell and should accompany the numeric receipt when this tranche is handed off.
The exact-mesh post-audit uses a separate physical-contact gate: at most 10 mm
foot penetration and zero forbidden-body penetration.  All 236 episodes pass;
the median, p90, and maximum foot contacts are 5.356, 6.733, and 9.132 mm.
Keep the clean kinematic source limit at 5 mm; deterministic physical tracking
itself reaches 8.154 mm because rigid-contact rollouts have solver compliance.

## Per-clip record

Store the clean kinematics and the causal information needed downstream:

- 50 Hz root pose, 29 joint positions, body/sole positions, velocities, and
  contact labels;
- raw and filtered two-stick commands, plus event masks for stop/restart,
  travel reversal, facing jump, and simultaneous stick changes;
- exact terrain ID, terrain transform, source clip/frame coordinates, and all
  composition parameters;
- robot-centric height/depth observation and camera calibration;
- clean-reference, tracker-checkpoint, rollout-seed, and mirror provenance.

The command labels come from the clean kinematics/offline operator.  They are
not recovered from the noisy rollout, so tracking errors cannot corrupt the
joystick target.

## Collection sequence

1. Materialize and mechanically audit the clean kinematic candidates.
2. Dense-review a stratified sample and every unusual boundary.
3. Fine-tune SONIC on the admitted clean bank (mixed with the stable existing
   terrain references so flat/stair capability is not forgotten).
4. Require clean full-clip survival and acceptable tracking error per stratum.
5. Collect noised SONIC rollouts, retaining failures only as diagnostics.
6. Render depth/height observations and package the diffusion-policy dataset.

Never start noised collection from training completion alone.  First create a
checkpoint-specific qualified bundle from the deterministic full-survival and
300 mm gate, visually inspect representative passes and failures, and collect
only from that bundle.  Keep failed noisy attempts as diagnostics rather than
training examples.
