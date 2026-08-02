# Privileged Global Terrain Motion Oracle

- **Status:** Architecture approved; document pending final review
- **Date:** 2026-08-01
- **Repository:** `Motion-Matching-takara-corpus`
- **Initial target:** Purely kinematic Unitree G1 motion synthesis

## 1. Summary

Build a new, fully privileged kinematic oracle that synthesizes smooth Unitree G1
locomotion over every terrain/action class supported by the available motion corpus.
The oracle may use the exact global robot pose, complete terrain geometry, and the
complete desired route. It is not constrained to be causal, real-time, or directly
deployable on the robot.

The oracle will:

1. convert all usable motion sources to one audited G1 representation;
2. segment motions into reusable contact-to-contact fragments rather than treating
   source clips as indivisible skills;
3. extract feasible support surfaces and candidate footholds from the exact terrain;
4. search globally for a complete sequence of compatible fragments and footholds;
5. fit the selected fragments with bounded spatial and temporal warping;
6. use inertialization and multi-frame contact-constrained whole-body IK to produce a
   continuous, grounded trajectory; and
7. reject or backtrack from any result that freezes, hovers, skates, penetrates,
   collides, or exceeds the audited motion envelope.

This is a replacement approach for the current local, greedy terrain viewer. The
existing viewer remains intact only as a comparison baseline.

Once the privileged oracle works, privileged information will be removed
incrementally. The oracle will also generate motion/command/terrain data for the
SONIC tracker and the downstream diffusion policy.

## 2. Motivation

The existing terrain controller is not adequate:

- flat locomotion is less smooth and responsive than the flat-only controller;
- terrain matching can freeze even away from terrain;
- some plans hover or hold nearly static poses;
- transitions are clunky and contain visible pose or velocity discontinuities;
- the controller frequently requires a narrow approach angle;
- feet can penetrate stair treads or risers;
- stair skills behave as if they are tied to their authored global scene; and
- local greedy choices can leave no valid continuation.

Repeatedly patching those symptoms does not address the architectural issue. The
current controller asks a sparse, locally queried motion bank to solve global route
planning, terrain feasibility, contact scheduling, animation selection, and terrain
adaptation simultaneously.

The new approach makes the problem deliberately easier. It first solves a
privileged, globally planned, purely kinematic upper bound. Only after that upper
bound is demonstrably good do we remove information or add tracking dynamics.

## 3. Scope

### 3.1 In scope

- Pure G1 kinematic reference motion.
- Complete global terrain mesh/heightfield/SDF.
- Exact global root pose.
- Complete desired route and facing trajectory for scripted evaluation.
- Receding global replanning for the interactive viewer.
- Independent movement and body-facing intent.
- All motion and terrain classes with measured support in the audited corpus.
- New terrain geometries inside the measured motion/warp envelope.
- Flat-to-terrain, terrain-to-flat, and mixed-terrain transitions.
- Existing retargeted motion datasets, including G1-retargeted LAFAN1.
- Offline computation that is slower than real time.

### 3.2 Not in the first milestone

- SONIC or any physics-based tracking.
- No-odometry or onboard-only inference.
- Partial visibility, occlusion, noisy perception, or state-estimation drift.
- Extrapolation to terrain/action classes absent from the motion bank.
- A learned generative controller.
- Modifying Takara's or Justin's repositories.
- Claiming arbitrary-terrain support without a measured data/warp envelope.

### 3.3 Meaning of “all terrain”

“All terrain” means all terrain/action classes for which the audited fragment graph
contains a complete feasible route. Coverage is not inferred only from a terrain
label. It is determined from contact transitions, step displacement, height change,
slope, yaw, speed, swing clearance, joint feasibility, and bounded warp limits.

An out-of-envelope request must be reported as unsupported. It must not be turned
into an arbitrary held pose, hovering pose, or penetrating motion.

## 4. Design principles

1. **Global correctness before causal deployment.** Use every available privilege to
   establish a strong kinematic upper bound.
2. **Contact events, not whole clips, are the reusable unit.** Whole-skill playback
   recreates source-scene overfitting and prevents flexible transitions.
3. **Plan the complete supported route.** Do not publish a prefix unless a valid
   suffix or a safe stop also exists.
4. **Terrain feasibility is a hard constraint.** It is not merely another soft
   nearest-neighbor feature.
5. **Warp only inside measured limits.** Large unvalidated warps are data
   extrapolation and must be rejected.
6. **Preserve contacts explicitly.** Generic pose blending is not allowed to move a
   planted foot.
7. **Keep the matcher interface robot-centred.** Global information belongs in
   replaceable providers so later ablations do not require rewriting the motion
   engine.
8. **Publish only audited trajectories.** A final independent checker either accepts
   the complete trajectory or returns it to global search.
9. **Keep provenance.** Every output frame must identify its source fragment, source
   frame, transformation, time warp, IK correction, and terrain/contact assignment.
10. **Never use tracker rollouts as clean kinematics.** Perturbed SONIC rollouts can
    later supply trackability or recovery evidence, but clean source motions define
    the oracle bank.

## 5. Relevant established methods

The design combines established components rather than relying on the previous
controller:

- Motion matching supplies future trajectory, root velocity, local foot state, and
  nearest-neighbor retrieval features.
- The Step Space supplies the footprint-driven formulation, nearest-neighbor motion
  selection, spatial/time warping, and IK.
- Carpet Unrolling demonstrates adapting canonical motion along a curved route and
  arbitrary terrain geometry.
- PFNN demonstrates conditioning character motion on future route samples and
  terrain heights and describes fitting motion/contact data to terrain databases.
- Inertialization supplies continuous pose and velocity transitions without a long
  cross-fade.
- Perceptive Humanoid Parkour demonstrates that retargeted G1 motion matching can
  densify sparse approach distances and gait phases. Its sequential playback of
  paired terrain skills is intentionally not adopted here because that retains the
  source-terrain restriction.

Primary references:

- Holden, Kanoun, Büttner et al., *Learned Motion Matching*:
  <https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/2EVXGPN6ynTrrJaaAUHmZS/060e223cce6a84e47e925105cac5e17f/Learned_Motion_Matching.pdf>
- Holden, Komura, and Saito, *Phase-Functioned Neural Networks for Character
  Control*: <https://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf>
- van Basten and Egges, *The Step Space: Example-Based Footprint-Driven Motion
  Synthesis*: <https://doi.org/10.1002/cav.342>
- Miller, Holden et al., *Carpet Unrolling for Character Control on Uneven Terrain*:
  <https://www.pure.ed.ac.uk/ws/files/21937045/MIG_2015_paper_22.pdf>
- Wu, Huang et al., *Perceptive Humanoid Parkour*:
  <https://php-parkour.github.io/static/images/paper.pdf>
- Bollo, *Inertialization: High-Performance Animation Transitions in Gears of War*:
  <https://www.gdcvault.com/play/1025331/Inertialization-High-Performance-Animation-Transitions>

## 6. Data sources

### 6.1 Flat connector bank

- TakaraWalk clean motion and the existing flat motion-matching bank.
- BONES flat locomotion already used by the flat controller.
- Existing start/stop, turning, backward, lateral, and omnidirectional variants.
- LAFAN1 walking, running, starting, stopping, turning, jumping, and transition
  intervals after G1 contract validation.

These motions form the dense connector manifold between terrain events.
When a source stores the human/controller command that produced a motion, preserve
that command and its frame convention in the canonical provenance. Inferred
movement and facing intent are stored separately and must never overwrite an
observed command.

### 6.2 Terrain bank

The currently inventoried clean GRAIL source corpus contains:

- 340 stair clips;
- 77 slope clips; and
- 72 curb clips.

Each clean source is 250 frames at 25 Hz. The 50 Hz SONIC rollout representation is
not the clean reference. Complete source motion and paired terrain assets must be
loaded from the clean `assets/robot` and `assets/object_usd` data, or from the
byte-identical repaired C490 shards.

Additional terrain sources:

- Justin's G1-retargeted ascending and descending clips at straight, ±30°, ±60°,
  and ±90° approaches;
- Justin's continuous left/right ascent and descent clips;
- Karen stair motions where the motion/terrain pairing passes audit; and
- recoverable LAFAN1 obstacle motions.

### 6.3 Existing G1-retargeted LAFAN1

Retargeting LAFAN1 again is not the default. Existing sources include:

- LocoMuJoCo's LAFAN1 datasets for supported humanoids, including Unitree G1:
  <https://github.com/robfiras/loco-mujoco>
- the full LAFAN1 retargeting release for Unitree G1:
  <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>
- an IsaacLab/ProtoMotions-formatted full G1 mirror:
  <https://huggingface.co/datasets/ember-lab-berkeley/LAFAN-G1>

The selected release will be imported read-only and validated against this
repository's exact 29-DoF G1 joint and quaternion contract. Conversion is allowed;
new retargeting occurs only for missing or failed clips.

The official LAFAN1 dataset contains 77 sequences and approximately 4.6 hours of
motion, including 17 obstacle/uneven-terrain sequences:
<https://github.com/ubisoft/ubisoft-laforge-animation-dataset>

LAFAN obstacle motion is accepted into the terrain bank only when a reliable support
surface can be recovered or paired. A human obstacle-motion label by itself is
insufficient.

### 6.4 Corpus density and balancing

The fragment graph is built from clean source intervals, not from uniformly sampled
SONIC rollout windows. Byte-identical clips and overlapping windows are deduplicated
for coverage estimation. Fragment count may affect style preference only through an
explicit, reported prior; it cannot make a transition feasible or infeasible.

Coverage reports and evaluation are stratified by action, direction, support leg,
height change, slope, and source identity rather than weighted in proportion to raw
clip or window count. In particular, the larger stair-ascent population cannot hide
weak stair-descent coverage.

### 6.5 Data ownership and repository hygiene

- External motion and terrain assets remain outside Git.
- Git stores manifests, hashes, adapters, audit reports, and small test fixtures.
- Justin's and Takara's repositories and data are read-only inputs.
- LAFAN licensing and the license of each retargeted release must be preserved.
- Mirrored or warped derived artifacts are local research outputs and are not
  redistributed unless their source license permits it.

## 7. Canonical G1 motion contract

Every source is converted to a canonical clip with:

- `fps`;
- root position in a declared Z-up world frame;
- root quaternion in declared `wxyz` order;
- 29 joint positions in the repository's G1 order;
- root and joint linear/angular velocities;
- forward-kinematic body poses;
- sole, heel, and toe poses;
- left/right contact labels and confidence;
- source terrain asset and world transform when available;
- surface queries under each contact;
- source dataset, source clip, source frame, license, and content hash; and
- audit status and rejection reasons.

Resampling uses quaternion SLERP and continuous interpolation of translations and
joint angles. Velocities and accelerations are recomputed after resampling. No
finite-difference derivative is copied across a clip boundary.

### 7.1 Required source audit

Each clip is checked for:

- coordinate-frame and quaternion consistency;
- joint-name/order consistency;
- joint-position and joint-velocity limits;
- discontinuities and duplicated/corrupt frames;
- root-height and orientation plausibility;
- contact consistency from sole geometry and velocity;
- stance-foot sliding;
- terrain/foot registration;
- sole and non-foot terrain penetration;
- mirrored-pair consistency; and
- source timing/fps consistency.

The audit produces `accepted`, `accepted_with_intervals_removed`, or `rejected`.
Repairs that change motion content must be explicit derived stages with before/after
metrics; they cannot silently alter the imported clip.

Every accepted source interval also receives an automated robot-mesh render against
its paired terrain, with contact, sole-height, and collision overlays. Batch contact
sheets make the complete accepted corpus reviewable; representative full videos
from every source/action stratum are manually inspected. A stick-figure preview is
not sufficient evidence that a source is usable.

## 8. Symmetric augmentation

Mirroring is performed only after canonicalization. The transformation jointly
mirrors:

- root translation and orientation;
- every joint angle using the validated G1 symmetry map;
- joint and body velocities;
- full history and future trajectory;
- left/right contact and support identity;
- sole/heel/toe positions and orientations;
- movement and facing commands;
- foothold transforms;
- paired terrain geometry or terrain-query coordinates; and
- provenance identifying the original and mirrored clip.

An involution test requires `mirror(mirror(x)) == x` within numerical tolerance for
every canonical field. Original and mirrored versions of a source remain in the same
train/evaluation split.

## 9. Contact-to-contact fragment bank

### 9.1 Segmentation

The reusable unit is a motion fragment between stable support events. Typical
fragments include:

- double support to left support;
- left support to right support;
- right support to left support;
- support to flight and flight to support;
- stable support to stable idle;
- idle to gait;
- step-up, step-down, curb, and slope transitions; and
- short multi-contact fragments when splitting would destroy a meaningful contact
  event.

A small context pad is retained before and after each event for entry matching and
inertialization. Source clips are never concatenated simply because their labels
match.

### 9.2 Fragment features

For each fragment, store:

- entry/exit local pose and velocity;
- entry/exit support state and contact phase;
- local root displacement, yaw change, and duration;
- movement direction and body-facing evolution;
- left/right contact transforms relative to entry root;
- swing-foot trajectory and swept clearance volume;
- pelvis and torso trajectory;
- root and joint velocity/acceleration/jerk statistics;
- nominal terrain samples under the root and both feet;
- support-height delta, surface normals, tread/slope characteristics;
- body and limb swept volumes;
- source-quality and optional tracker-quality metadata; and
- allowed spatial/time-warp intervals.

### 9.3 Warp envelopes

Warp limits are conservative and data-derived. They are estimated by:

1. grouping fragments with matching support/action semantics;
2. measuring the distribution of displacement, yaw, duration, height change,
   surface normal, and clearance;
3. validating warped fragments through the same trajectory/contact/collision audit;
4. retaining only connected ranges that pass; and
5. shrinking the boundary by a safety margin.

The planner cannot use an independent per-dimension min/max box that combines
unobserved extremes. Feasibility must respect the joint distribution or an explicit
validated neighborhood of fragment parameters.

## 10. Replaceable request providers

The motion engine consumes a robot-centred request, regardless of how it is
constructed:

- current local pose, velocities, and contact state;
- desired local root/movement trajectory;
- desired local body-facing trajectory;
- terrain/support representation along the horizon;
- optional planned foothold candidates; and
- planning horizon and behavior constraints.

### 10.1 Privileged provider

The first oracle provider uses:

- true global root transform;
- complete terrain mesh and exact transforms;
- exact global route and facing curve; and
- unlimited past/future route context.

It converts these into the same robot-centred request later providers will use.

### 10.2 Later providers

Future providers can substitute:

- a finite receding global route;
- an exact robot-centred height map;
- a partially observed depth/voxel representation;
- estimated local motion rather than global root pose; and
- a critically damped local two-stick trajectory.

No later provider changes the fragment schema or output trajectory contract.

## 11. Privileged global terrain planner

### 11.1 Terrain representation

The exact global triangle mesh is the source of truth. Derived accelerators may
include:

- heightfields for coarse search;
- signed-distance or collision queries for clearance;
- planar support polygons with normals and boundaries;
- stair tread/riser decomposition;
- connected traversable regions; and
- footprint-eroded support regions.

The exact mesh remains the final contact and collision authority. A zero value in a
heightfield must not be confused with an invalid/no-hit sample.

### 11.2 Route input

Scripted evaluation supplies a complete world-space route with:

- desired path position over arc length/time;
- desired movement speed;
- desired body facing independent of travel direction; and
- optional stop or behavior events.

The interactive oracle integrates local two-stick input into a world-space intent
using exact global state. Replanning remains privileged; the user-facing commands
are still local to the robot.

### 11.3 Foothold candidates

For each region along the route, the planner generates multiple feasible left/right
support transforms. A candidate must:

- place the complete sole inside an eroded support polygon;
- have an allowed surface normal;
- satisfy source/target step length, lateral offset, yaw, and height ranges;
- leave a collision-free swing corridor for at least one compatible fragment;
- preserve alternating/support semantics where required; and
- admit a compatible root/pelvis corridor.

The foothold generator does not commit to one footstep sequence. It provides a graph
of alternatives to the coupled motion search.

## 12. Full-horizon coupled search

### 12.1 Search state

A search state includes:

- route progress;
- current global root transform;
- current pose/velocity summary;
- support/contact state;
- planted-foot transforms;
- current fragment/source context;
- chosen foothold node;
- elapsed time; and
- accumulated cost and audit margins.

### 12.2 Expansion

An expansion selects:

1. a compatible next foothold/support event;
2. a compatible motion fragment;
3. a global placement transform;
4. bounded spatial-warp parameters; and
5. bounded monotone time-warp parameters.

Cheap analytic checks run before expensive trajectory reconstruction.

### 12.3 Costs

The ranking objective includes:

- entry pose and velocity discontinuity;
- support/contact mismatch;
- path-position and movement-direction error;
- independent facing error;
- foothold and terrain-signature error;
- root/pelvis corridor error;
- spatial/time-warp magnitude;
- joint-limit and derivative margin;
- swing/body clearance margin;
- source-quality penalty;
- repeated-fragment/style penalty where useful; and
- terminal route/stop quality.

### 12.4 Hard constraints

Reject an expansion for:

- incompatible support state;
- stance foot outside its support polygon;
- unannotated flight in a ground-bound fragment;
- foot, limb, torso, or root penetration;
- inadequate swing clearance;
- joint-position or derivative-limit violation;
- spatial/time warp outside the fragment's validated envelope;
- loss of a safe continuation or terminal stop; or
- invalid terrain data.

### 12.5 Global solution

Use deterministic beam search, dynamic programming, A*, or a hybrid selected during
implementation benchmarking. The essential contract is:

- search far enough to cover the full scripted route;
- preserve multiple alternatives at ambiguous steps;
- include a valid terminal stop/idle;
- return a complete plan before publication; and
- backtrack when reconstruction or final audit invalidates a selected edge.

The interactive mode retains and plays the previous valid plan while a replacement
is solved. It switches plans at a compatible contact state. It never consumes a
partial plan and then holds its last arbitrary pose.

## 13. Motion reconstruction

### 13.1 Placement and spatial warping

Selected source fragments are placed in the world using their entry root and support
anchors. The desired change in displacement, yaw, and height is distributed smoothly
over the fragment, with zero motion of already planted contact anchors.

Spatial adjustment operates on task-space targets and a root correction curve; it
does not directly scale arbitrary joint angles.

### 13.2 Time warping

Time maps are monotone, preserve ordered contact events, and remain inside validated
duration/phase ranges. Resampling uses continuous root/quaternion/joint
interpolation. Touchdown timing and velocity must remain continuous at fragment
boundaries.

### 13.3 Inertialization

At a fragment transition:

- compute pose and velocity offsets between the old reconstructed state and the new
  source fragment;
- immediately apply the offsets so output pose/velocity remains continuous; and
- decay the offsets with a critically damped model.

Offsets for planted end effectors are resolved through contact-constrained IK rather
than allowed to move the contact point.

### 13.4 Multi-frame contact-constrained IK

A sliding-window or whole-route kinematic optimization solves jointly for root and
joint corrections. Its priorities are:

1. exact stance-foot position and allowed orientation;
2. no terrain/body collision;
3. swing-foot clearance and target touchdown;
4. joint positions and derivative limits;
5. smooth root/pelvis/torso motion;
6. minimal deviation from the selected source motion; and
7. smooth correction fields across seams.

Contact constraints are active over complete stance intervals, not only at
touchdown. Foot locking is therefore a consequence of the solve, not a visual
post-process.

### 13.5 Final independent audit

The final checker recomputes forward kinematics, contacts, support, collision,
clearance, joint derivatives, seam discontinuities, and route completion from the
published trajectory. It does not trust planner-internal cached values.

Failure returns the violating interval and constraint to global search. A trajectory
is not emitted as successful until the independent checker passes.

## 14. Output contract

Each synthesized route exports:

- canonical G1 root and joint trajectory;
- velocities, accelerations, and contact labels;
- desired movement and body-facing commands;
- global and robot-local desired route;
- selected footholds and support polygons;
- source fragment/frame identity for every output frame;
- placement, spatial-warp, time-warp, inertialization, and IK corrections;
- exact paired terrain and transforms;
- audit metrics and margins;
- supported/unsupported status with structured reason; and
- deterministic seed/config/content hashes.

These outputs are sufficient for:

- kinematic rendering;
- interactive replay;
- SONIC motion tracking and noisy data collection;
- downstream diffusion-policy training; and
- privilege-ablation comparisons.

## 15. Interactive privileged viewer

The viewer reuses the existing browser, SSH-tunnel, and Switch-controller
infrastructure, but uses the new oracle backend.

Controls:

- left stick: local movement direction and speed;
- right stick: independent local body facing/yaw intent;
- centre stick: controlled stop;
- reset: return to a known safe standing state.

The viewer displays:

- exact terrain;
- desired route;
- candidate and selected footholds;
- current support/contact state;
- source fragment identity;
- planning status and horizon;
- audit margins; and
- explicit unsupported reasons.

Global state may be used internally during this milestone. The movement/facing
interface remains robot-centred so the interaction is representative of the desired
final controller.

## 16. Coverage atlas

The corpus build produces a machine-readable and visual coverage atlas over:

- movement direction relative to body facing;
- root speed and yaw rate;
- start/stop/reverse transitions;
- support leg and phase;
- horizontal step displacement;
- lateral step displacement;
- step height up/down;
- surface normal/slope;
- contact yaw and foot orientation;
- swing clearance;
- duration/time scale;
- approach and exit geometry; and
- action class.

Coverage is reported both marginally and jointly. Sparse or disconnected regions are
visible. Test generation samples only validated connected coverage, while dedicated
negative tests sample just outside it and require an explicit unsupported result.

The initial coverage manifest is derived from accepted clean sources and validated
warp neighborhoods, then frozen before system-level benchmark runs. A planner
failure cannot be reclassified after the fact by shrinking that manifest. Any later
coverage revision requires a versioned manifest, a stated data/audit reason, and a
fresh run of the complete benchmark.

## 17. Evaluation design

### 17.1 Split policy

- All frames/fragments from a source clip stay in one split.
- Original and mirrored clips stay in one split.
- Near-duplicate/byte-identical motions stay in one split.
- Entire authored terrain assets and procedural parameter groups are held out.
- No train/evaluation split is made by nearby windows from the same clip.

### 17.2 Supported benchmark

The benchmark includes all supported classes found by the coverage atlas:

- idle, start, stop, and stop-to-start;
- forward, backward, lateral, diagonal, and omnidirectional locomotion;
- gradual and sharp turns;
- in-place turns when supported;
- movement/facing decoupling;
- forward-to-back and other direction changes;
- stair ascent and descent;
- varied stair riser, tread, width, yaw, landing, and approach combinations;
- curb ascent/descent;
- slope ascent/descent/cross-slope where supported;
- recoverable obstacle/jump classes;
- flat-to-terrain and terrain-to-flat transitions; and
- mixed routes containing multiple supported terrain classes.

Procedural scenes are new geometric instances inside the validated fragment/warp
envelope. They include repeated/cube-style stair arrangements and platforms so
ascent, traversal, turning, and descent can be exercised continuously.

The deterministic route, terrain, seed, and coverage manifests are frozen before
tuning against the benchmark. A separate development suite may change during
implementation, but acceptance is measured on the frozen supported and negative
suites.

### 17.3 Negative benchmark

Negative cases deliberately exceed one coverage dimension or disconnect support.
They must:

- be classified unsupported;
- name the missing transition or violated envelope;
- end in a valid planned stop on the last safe support; and
- never publish a colliding, penetrating, frozen, or hovering “solution.”

### 17.4 Mechanical acceptance

The first oracle milestone requires:

1. **Route completion:** 100% on the declared supported deterministic suite and a
   seeded procedural stress suite.
2. **No freeze:** no repeated output pose/source frame beyond the configured
   intentional idle/contact dwell; non-idle plans continue making route or gait
   progress.
3. **Contact:** stance-foot translation drift no greater than 3 mm over a stance
   interval and stance orientation error no greater than 1 degree unless the source
   action explicitly rolls through the foot.
4. **Penetration:** sole contact tolerance no worse than 2 mm; no non-contact foot,
   limb, pelvis, or torso penetration.
5. **Support:** at least one valid support contact during ground-bound motion, except
   annotated flight intervals.
6. **Clearance:** swing and body clearance remain nonnegative with the configured
   mesh safety margin.
7. **Smoothness:** one-frame root/joint changes, velocities, accelerations, jerk, and
   seam discontinuities remain within canonical-bank limits and explicit robot joint
   limits.
8. **Coverage:** no false unsupported result inside the declared connected coverage
   benchmark.
9. **Determinism:** identical inputs/config/content hashes reproduce the same plan
   and trajectory.
10. **Independent audit:** every successful artifact passes the final checker from
    disk.

If source-derived limits are stricter than the absolute contact tolerances, the
stricter limit applies. If an absolute threshold proves incompatible with a valid
source action, the exception must be action-specific, justified, and visible in the
specification rather than silently widened.

### 17.5 Visual acceptance

For every class, generate:

- a robot-mesh video;
- top-down desired route, root trajectory, and footholds;
- side-view terrain, sole contacts, and pelvis trajectory;
- fragment/timing strip;
- contact/clearance trace; and
- source-versus-corrected motion comparison.

The videos are inspected for gait quality, foot crossing, knee collapse, abrupt
joint changes, skating, hovering, penetration, implausible lean, and unnatural
pauses. Mechanical success alone is insufficient when the animation is visibly
poor.

## 18. Delivery stages

### Stage 1: Corpus import and audit

- Add read-only source adapters and manifests.
- Import existing retargeted LAFAN1.
- Produce accepted/rejected interval reports and a coverage atlas.
- Validate symmetric augmentation.

Exit gate: all accepted clips satisfy the canonical contract, have robot-mesh audit
renders, and pass stratified visual review.

### Stage 2: Contact-fragment bank

- Segment accepted clips.
- Extract fragment features and warp envelopes.
- Validate smooth flat composition first.

Exit gate: scripted flat start/stop/turn/omnidirectional routes pass without seams,
freeze, or foot skate.

### Stage 3: Privileged terrain/foothold planner

- Build exact mesh support/collision queries.
- Generate support and foothold graphs.
- Validate foothold-only routes across every covered terrain class.

Exit gate: the planner completes all supported routes and rejects negative routes
with structured reasons.

### Stage 4: Global motion synthesis

- Couple footholds and fragment search.
- Add bounded warping, inertialization, and contact-constrained IK.
- Add final independent audit and backtracking.

Exit gate: the complete supported scripted suite passes.

### Stage 5: Procedural benchmark

- Generate held-out procedural scenes and mixed routes.
- Produce metrics, plots, and videos.
- Iterate only on measured failures.

Exit gate: the supported procedural stress suite passes the mechanical and visual
acceptance contract.

### Stage 6: Interactive privileged viewer

- Connect two-stick input.
- Add exact-state replanning and plan handoff.
- Expose planner/coverage diagnostics.

Exit gate: extended interactive sessions remain smooth, grounded, responsive, and
free of unexplained freezes.

### Stage 7: Export and ablations

- Export clean oracle trajectories and commands for SONIC/diffusion data.
- Run the privilege-removal ladder.

Exit gate: each ablation has the same benchmark report, making the quality loss from
each removed input measurable.

## 19. Privilege-removal ladder

The intended progression is:

1. full route + complete mesh + true global state;
2. finite receding route + complete mesh + true global state;
3. finite route + exact robot-centred terrain + true global state;
4. finite route + partial robot-centred terrain + true global state;
5. finite route + partial terrain + estimated relative state;
6. local critically damped two-stick intent + partial terrain + no global odometry;
7. oracle-generated data tracked by SONIC and learned by the diffusion policy; and
8. optional removal of the runtime kinematic planner if the learned policy absorbs
   the oracle behavior sufficiently.

Every stage uses the same robot-centred motion request and evaluation suite. Global
state is never allowed to leak into a stage that claims not to use it.

## 20. Testing strategy

### 20.1 Unit and property tests

- coordinate/quaternion/joint-order conversions;
- resampling and derivative recomputation;
- contact detection on analytic planes, steps, curbs, and slopes;
- mirror involution and mirrored terrain/contact equivalence;
- fragment segmentation around synthetic contact schedules;
- support-polygon erosion and complete-sole containment;
- warp monotonicity and envelope rejection;
- source-frame provenance through composition;
- inertialization pose/velocity continuity;
- stance-foot invariance under IK;
- collision and clearance queries;
- final audit independence; and
- deterministic planning.

### 20.2 Integration tests

- tiny synthetic motion bank with a known optimal route;
- flat start/walk/stop;
- one step up and one step down;
- multi-step ascent/landing/descent;
- angled approach;
- mixed flat/curb/slope/stair route;
- unsupported height or missing foothold;
- interactive replan while an old plan remains valid; and
- planner backtracking after reconstruction failure.

### 20.3 Regression artifacts

Keep small golden fixtures and metric JSON in Git. Large videos, banks, and
trajectories remain in artifact storage and are identified by hashes/manifests.

## 21. Failure handling

No silent fail-open behavior is permitted.

Structured failure categories include:

- no compatible support surface;
- missing support transition;
- insufficient swing clearance;
- fragment coverage gap;
- excessive spatial warp;
- excessive time warp;
- joint-limit or derivative violation;
- collision/penetration;
- reconstruction non-convergence;
- no safe terminal stop;
- invalid terrain data; and
- invalid source motion.

For scripted synthesis, any failure makes the route unsuccessful and produces a
diagnostic artifact. For interactive use, the current complete valid plan continues
until a replacement plan is ready. If the request moves outside coverage, the
controller follows a preplanned safe stop and marks the request unsupported.

An intentional safe stop is not reported as successful route following.

## 22. Risks and mitigations

### Sparse transition graph

**Risk:** Even many source clips may leave contact/phase gaps.

**Mitigation:** LAFAN/BONES/Takara connector data, correct symmetric augmentation,
fragment-level rather than clip-level reuse, global search, and an explicit coverage
atlas. Missing regions remain visible.

### Over-warping

**Risk:** IK can make a formally valid but unnatural motion.

**Mitigation:** validated joint warp envelopes, source-deviation costs, joint and
jerk gates, and visual comparison.

### Terrain/contact registration errors

**Risk:** A good motion appears to penetrate because source terrain transforms are
wrong.

**Mitigation:** exact paired mesh transforms, independent contact reconstruction,
visual overlays, and no reliance on stale prose metadata.

### Search cost

**Risk:** Whole-route search with IK is expensive.

**Mitigation:** hierarchical coarse-to-fine filtering, cached fragment descriptors,
cheap hard gates before reconstruction, parallel candidate evaluation, and offline
computation for the first milestone.

### Neural-controller temptation

**Risk:** Replacing visible failures with a learned generator before the oracle is
understood obscures data/constraint gaps.

**Mitigation:** keep the first system deterministic and kinematic. Train or distill
only after the oracle passes.

### Tracker confusion

**Risk:** SONIC failure can be mistaken for bad kinematics.

**Mitigation:** SONIC is explicitly outside the first milestone. Kinematic quality
and trackability are evaluated separately.

## 23. Success criterion

The project has reached its first target when a user can:

1. select or procedurally generate any terrain course inside the audited coverage
   envelope;
2. provide a complete route or use the privileged interactive two-stick viewer;
3. obtain a complete G1 kinematic motion with smooth flat locomotion and natural
   terrain transitions;
4. traverse the route without freezing, hovering, skating, penetrating, colliding,
   or suffering visible pose seams;
5. inspect exactly which motion fragments, contacts, terrain queries, and
   corrections produced the result; and
6. reproduce the same result and independent audit from saved artifacts.

Only then does work proceed to SONIC tracking and removal of privileged global
information.
