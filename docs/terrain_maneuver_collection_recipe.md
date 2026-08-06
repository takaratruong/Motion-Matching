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

Do not start the 200-clip noisy collection from the current tracker: the two-clip
probe demonstrates that it would merely collect failures.
