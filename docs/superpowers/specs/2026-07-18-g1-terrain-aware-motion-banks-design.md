# G1 Terrain-Aware Motion Banks Design

**Date:** 2026-07-18

**Status:** Approved

## Goal

Make the stable classic-UI G1 visualizer select motion that is appropriate for
the commanded travel direction and the terrain actually approaching the robot.
The first user-facing result must improve lateral/diagonal locomotion, keep
flat acceleration from selecting high-step motion, distinguish stairs from
slopes, and expose remaining penetration with the corrected G1 mesh.

IK is deliberately last. The user must receive and inspect a playable mesh
visualizer containing the data and matcher changes before any IK correction is
enabled or merged into this branch.

## Evidence and Root Cause

The current authenticated pack contains 459,682 frames:

- 17,432 flat Takara frames;
- 442,250 GRAIL curb frames;
- no GRAIL slope frames; and
- no GRAIL stair frames.

The matcher searches every range as if all sources were interchangeable. On a
flat route it can therefore select a zero-height portion of a curb clip and
later play the clip's high step. A deterministic flat-forward reproduction
switches from Takara to a curb source at the first search. Seventy of its first
100 frames come from curb sources despite a completely flat runtime query.

The same database is used for procedural stairs and ramps. It cannot select a
slope or stair source because none exists. Current route-completion gates only
prove that the controller reached the endpoint; they do not prove category-
correct motion, low transition churn, planted-foot stability, or absence of
visible sole penetration.

Lateral commands are geometrically independent from heading, as required, but
their selected costs and transition rates are much worse than forward travel.
The current corpus has usable lateral frames, especially in Takara, but they
are rare relative to the curb corpus and are not indexed or balanced by travel
angle. Terrain-lateral runs consequently alternate between weak candidates and
show sliding or penetration.

Finally, the current clearance log samples hips, knees, ankles, and toes. It
does not measure the physical sole corners shown by the corrected mesh. IK is
off in the stable visualizer. This explains why an automated route can pass
while the user still sees the foot inside the terrain, but it does not justify
using IK to hide a wrong motion source.

## Considered Approaches

### Selected: category- and direction-aware motion banks

Acquire the real GRAIL slope and stair trajectories, retain source provenance,
compute a richer terrain descriptor, and search only compatible terrain and
travel-angle banks. The runtime keeps travel velocity and heading independent.
This directly addresses wrong-source selection and makes data coverage
measurable.

### Rejected: append every clip to the existing global search

Raw concatenation would make the database even more imbalanced, allow flat
queries to select terrain clips, increase the cost of every search, and retain
the current slope/stair aliasing. It would add data without making that data
usable.

### Rejected: tune weights or add source penalties

A larger terrain weight cannot distinguish source frames whose four current
height samples are all zero. Source-name penalties and special-case startup
rules would be fragile substitutes for an explicit compatibility contract.

### Rejected: finish IK first

IK can reduce stance-foot error after a suitable pose is selected. It cannot
turn a curb step into flat walking, create a slope gait, or create sustained
lateral motion. Applying it first would make the visible symptom smaller while
leaving the matcher semantically wrong.

## Data Acquisition and Provenance

Use NVIDIA's public GRAIL release as the source of truth:

<https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL>

Download only required modalities. Both `stair_p1/robot` and
`stair_p2/robot` are required; together they contain 12,188 trajectories and
approximately 2.26 GiB of robot files. The existing 1,880 slope robot clips
remain local. Matching `objects` and `object_usd` inputs are fetched when they
are required to derive source-terrain descriptors; videos, human
reconstructions, and unrelated HOI categories are excluded. Static terrain is
placed from the post-RL object trajectory because it shares the released
simulation frame with the robot trajectory. Curb fixtures must prove parity
with the already validated reconstruction-based surface before that path is
used for slope or stair data. Missing object pose or geometry fails closed; the
builder may not infer a staircase from root height alone.

All acquired files remain external dataset inputs. They are never committed to
the motion-matching repository. A checked-in acquisition manifest records the
GRAIL repository revision, allowed path patterns, file count, byte count, and a
canonical inventory hash. The generated external inventory retains individual
file identities used by a build. A partial or changed download fails before
artifact publication.

Every source record in the generated pack includes:

- immutable source name and source range;
- terrain family: `flat`, `curb`, `slope`, or `stair`;
- ascent/descent/level direction where applicable;
- travel-angle coverage bins relative to heading;
- source artifact identity; and
- derivation/validation report.

The full stair corpus is audited and indexed. Runtime search remains bounded by
bank and direction indexes; it does not linearly compare every query against
all four million frames.

## Artifact Architecture

Keep motion, terrain scenes, and mesh assets as separate packages. This work
does not bake a precomputed animation into each scene and does not merge scene
terrain geometry into motion data.

The motion package advances to a new schema with:

1. the combined Holden pose database;
2. per-frame terrain matching features;
3. per-frame support data;
4. authenticated source records;
5. terrain-bank range indexes; and
6. heading-relative travel-angle range indexes.

Frames may share one physical pose database, but search ownership is explicit.
A bank index is a list of source ranges, not a duplicated pose buffer. Bounds
remain global-frame indexed so selected frames continue to use the existing
Holden pose and inertialization path.

The pack builder publishes transactionally. A failed source conversion,
descriptor build, index audit, or candidate validation preserves the prior
playable pack byte-for-byte.

## Terrain Observation and Classification

The existing four centerline heights are too sparse to reliably distinguish a
steady ramp from several stair treads. Replace the four terrain dimensions with
a `G1TF/v2` twelve-dimensional matching descriptor, increasing the full
motion-matching query from 31 to 39 dimensions:

- eight root-relative centerline heights at 0.125 m intervals through 1.0 m;
- four left-minus-right height differences at 0.25, 0.50, 0.75, and 1.00 m,
  sampled at fixed sole-corridor offsets about the predicted travel path.

The descriptor follows predicted travel, while the sole offsets retain the
independently commanded heading. Terrain code may not rotate heading toward
travel.

A separate deterministic classifier uses a denser longitudinal profile and
surface normals to choose a terrain bank:

- `flat`: negligible grade and no level discontinuity;
- `slope`: sustained continuous grade without repeated tread/riser changes;
- `curb`: an isolated up/down discontinuity or isolated raised obstacle; and
- `stair`: repeated discontinuities separated by tread-width plateaus.

Classification includes signed ascent/descent and confidence. The same
float32 heightfield semantics are used by interactive and deterministic runs.
Ambiguous boundary observations retain the current bank until a fixed
hysteresis threshold is crossed. Invalid or out-of-domain observations fail
closed through the existing safe-stop path; they never silently fall back to a
random terrain bank.

## Direction-Aware Search

For each database frame, derive the one-second root displacement in the
simulation frame and assign overlapping heading-relative travel bins:

- forward;
- forward diagonals;
- left and right lateral;
- backward diagonals; and
- backward.

Bins overlap at boundaries so small command changes do not remove every nearby
candidate. Stationary/start/stop frames have an explicit low-speed index.

At search time:

1. classify the runtime terrain profile;
2. select compatible terrain banks;
3. select compatible travel-angle and speed bins;
4. evaluate the current continuation if it remains compatible;
5. search only those indexed ranges; and
6. apply a fixed bank-transition hysteresis before publishing a transition.

The current frame is not allowed to remain the incumbent merely because it
belongs to an incompatible bank. Conversely, a flat acceleration query cannot
enter curb, slope, or stair banks while the full lookahead remains flat.

The command contract remains unchanged: desired velocity controls travel and
desired rotation controls heading. Matching, terrain classification, fallback,
and safe-stop logic cannot write a new heading.

## Sideways and Diagonal Quality

Direction indexes expose whether genuine candidates exist instead of letting
global nearest-neighbor search hide scarcity. The pack audit reports, for every
terrain family and angle bin:

- source clips and usable frames;
- left/right symmetry;
- speed distribution;
- ascent/descent coverage;
- contact-phase coverage; and
- nearest-query cost on fixed probes.

Search must not relabel turning clips as strafes. A lateral candidate must meet
a bounded heading-change contract over the same horizon used for its travel
classification. If a bank lacks a suitable lateral terrain candidate, the
runtime reports that exact coverage gap and uses a documented compatibility
fallback or safe-stop; it does not rotate the robot toward travel.

Transition churn and contact sliding are first-class quality metrics. A route
that completes through frequent unrelated source jumps does not pass.

## Pre-IK Foot and Penetration Evidence

Before IK, add read-only physical diagnostics for the corrected G1 mesh:

- heel, toe, inner-edge, and outer-edge sole probes for each foot;
- minimum sole-to-heightfield clearance;
- stance-foot horizontal slip while contact is active;
- swing-foot clearance; and
- source bank, direction bin, classifier result, and transition reason.

These probes do not modify the pose. Their purpose is to expose whether the
data/matcher checkpoint improved motion selection and to establish the exact
residual error that IK may later address.

The red contact/target markers must distinguish measured sole points from
future IK targets. In the pre-IK checkpoint no marker may imply that a target
was applied.

## Delivery Phases

### Phase A: data and matcher, IK off

1. Download and authenticate GRAIL stair robot data.
2. Generalize conversion for slope and stair source families.
3. Build and validate bank/direction indexes and the richer descriptor.
4. Add category-aware search and bank hysteresis.
5. Add sole/sliding diagnostics without pose correction.
6. Run automated and recorded visual gates.
7. Push the checkpoint and launch the classic-UI corrected-mesh visualizer.

The user inspects flat acceleration, forward stairs, ramps/slopes, lateral and
diagonal traversal, edges, and partial support. Phase A remains explicitly
`ik_enabled=0`.

### Phase B: review-driven matcher refinement, still IK off

Use the user's visual feedback and the new logs to adjust data coverage,
descriptor/index boundaries, and transition policy. Do not use IK to close a
motion-selection or source-coverage defect.

### Phase C: IK last

Only after the user accepts the pre-IK direction and terrain behavior may the
branch add restrained contact IK. IK consumes the measured residual sole error,
acts only on valid stance/contact phases, preserves swing motion, and remains a
toggle so pre/post behavior is directly comparable.

## Verification Gates

All gates run at exact 25 Hz and preserve the classic UI, current controls,
corrected mesh, and independent travel/heading behavior.

### Bank and data gates

- Every database frame belongs to exactly one terrain family and at least one
  valid speed/direction index.
- Source ranges, bank indexes, and direction indexes cover only authenticated
  source ranges and round-trip byte-exactly.
- Slope and stair packs contain real matching GRAIL sources; procedural scene
  labels alone are insufficient.
- Fixed flat queries cannot select non-flat banks.
- Fixed slope queries select slope sources, and fixed repeated-step queries
  select stair sources.

### Runtime quality gates

- Flat acceleration remains in the flat bank until non-flat terrain enters the
  lookahead.
- Mirrored left/right lateral and both diagonal flat routes complete without
  terrain-induced heading changes.
- Forward, lateral, and diagonal slope routes do not select stair sources.
- Forward and mirrored lateral stair routes do not select slope or curb
  sources once repeated risers enter the lookahead.
- Transition count, selected cost, stance slip, and sole penetration must beat
  the recorded current baseline; endpoint completion alone is not acceptance.
- No physical sole probe may pass through an obstacle unnoticed by the log.

Exact numeric quality thresholds are frozen from deterministic baseline and
candidate distributions before implementation tuning. They are not selected
after viewing a favored run.

### Visual gate

Create short, disposable verification clips under a dedicated project-level
verification directory. Include synchronized old/new views for flat startup,
sideways flat travel, stairs, slopes, and a tangential/partial-support edge.
Delete clips after the user accepts or rejects them so the directory remains
bounded.

After automated gates pass, close the prior visualizer and launch exactly one
new classic-UI mesh visualizer. Keep IK visibly off. The running visualizer is
updated only at pushed, reproducible checkpoints.

## Failure Handling

- Interrupted downloads resume without accepting partial files.
- Missing or changed external files fail acquisition validation before build.
- Invalid source terrain geometry, motion conversion, descriptors, or indexes
  fail artifact publication transactionally.
- An empty compatible search set produces an explicit logged coverage failure
  and safe-stop; it does not search every bank.
- Non-finite query, terrain, cost, pose, or sole data preserves the last
  accepted state and enters controlled runtime error handling.
- A failed new pack or binary never replaces the currently running user
  visualizer.

## Version Control and Isolation

Continue on the isolated `g1-playable-mesh` branch and worktree. Keep the
unfinished transactional/IK research branch separate. Commit and push every
coherent verified checkpoint to `checkpoint/g1-playable-mesh`; external GRAIL
data and generated multi-gigabyte motion packs remain untracked.

The starting branch has one known unrelated baseline failure:
`test_contract_definitions_have_one_production_owner` finds a duplicated IK
contract literal in `g1_ik_runtime.h`. This design does not modify that IK
header during Phase A. All new focused tests and the remainder of the 306-test
baseline must pass, with this exact pre-existing failure reported separately
until the IK phase or an explicitly scoped cleanup fixes it.

## Out of Scope Before the First Visual Review

- enabling or tuning IK;
- rotating heading toward travel direction;
- synthesizing lateral clips by relabeling turns;
- merging scene terrain and motion into per-scene precomputed playback;
- merging the unfinished transactional runtime;
- changing the classic UI or corrected mesh mapping; and
- replacing the currently running visualizer before the new checkpoint passes.
