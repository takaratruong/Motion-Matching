# Simultaneous Walk-to-Pickup Overlap Diffusion

Date: 2026-07-21

Status: approved for implementation planning

## Goal

Generate one continuous full-body G1 motion from the character's live walking
state through object pickup. A walking diffusion expert and an
object/grasp-conditioned pickup diffusion expert denoise two overlapping motion
windows at the same time. Their shared frames are one latent region, reconciled
after every denoising step. The runtime therefore never concatenates two
independently completed clips and never inserts a stop at the transition.

The first implementation deliberately uses one walking window and one pickup
window. Multi-window long-range generation is deferred until this smallest
coupled system proves that simultaneous diffusion improves the walking-to-pickup
seam.

## User-Visible Contract

On the rising edge of `F`:

1. Freeze the exact displayed and simulated G1 state. Do not transition to idle
   or advance physics, animation, interaction, or simulation time.
2. Build the object/grasp condition and a collision-free planar route corridor
   from the frozen root to the target.
3. Ask one resident, prewarmed worker for a fixed batch of eight jointly denoised
   walk/pickup timelines.
4. Certify and select one complete timeline while the frozen state remains
   immutable.
5. Resume once and execute the selected generated timeline continuously through
   walking, Reach, Contact, and attachment, then continue directly into the
   recorded Lift continuation selected during the same frozen planning pass.

No state at the shared-window boundary may request settling, idle, zero desired
velocity, a late matcher search, or a second planning pause. Natural slowing as
the hand reaches the object is allowed; an artificial stationary dwell at the
walking/pickup seam is not.

If planning times out or every candidate fails certification, unfreeze the exact
captured state and return to manual locomotion. Learned mode must not silently
fall back to the previous Smart Pickup path.

## Scope

The first checkpoint includes:

- flat terrain;
- one 50-frame walking window;
- one 50-frame pickup window;
- one 20-frame shared overlap;
- full G1 pose generation rather than root-only funnel generation;
- right-handed interactions represented by the current processed pack;
- object geometry, support height, and grasp-transform conditioning;
- eight candidates sampled together;
- persistent, warmed Python/Torch/CUDA inference;
- deterministic route, kinematic, collision, contact, and grasp certification;
- a sequential uncoupled ablation; and
- a monolithic 80-frame diffusion baseline trained after the coupled prototype.

The first checkpoint does not claim:

- left-hand pickup generation;
- unsupported approach bearings or grasp families;
- arbitrary obstacle-rich navigation;
- more than one walking window;
- terrain-aware generation;
- online replanning after execution begins;
- physical robot control; or
- novel-object generalization beyond the frozen object-disjoint evaluation
  split.

## Timeline and Window Layout

All motion, conditions, sampling, execution, and evidence use 25 Hz.

```text
global frame       0                         30        49        50             79
                   |--------------------------|=========|---------|--------------|
walking expert     [--------------- 50 frames ----------]
pickup expert                                 [---------- 50 frames --------------]
shared overlap                                [20 frames]
```

- Walking slice: global frames `[0, 50)`; 2.00 seconds.
- Pickup slice: global frames `[30, 80)`; 2.00 seconds.
- Shared slice: global frames `[30, 50)`; 0.80 seconds.
- Complete unique timeline: 80 frames; 3.20 seconds.

The pickup window's first 20 frames are still pre-Reach approach motion. Its
remaining 30 frames cover Reach, Contact, and early post-Contact motion. The
pickup expert may continue root movement throughout its window. Therefore the
character is not required to cover the entire object distance in the walking
window alone. A compatible recorded continuation is selected before resume and
starts immediately after generated frame 79; this is an execution seam, not a
late matcher search or a second planning barrier.

## Shared Motion Representation

Both experts consume and predict exactly the same per-frame tensor schema. The
canonical joint order is the existing `FLAT_JOINT_NAMES` G1 order used by the
interaction artifacts and native bridge.

Each clean frame contains:

- object-frame root translation `(x, y, z)`;
- continuous root rotation represented by the first two columns of its rotation
  matrix;
- continuous local rotation for every canonical G1 joint in the same 6D form;
- left-foot, right-foot, and active-hand contact values.

The diffusion output does not independently predict velocities. Root linear and
angular velocity, joint angular velocity, acceleration, and foot sliding are
derived by finite differences at 25 Hz. This prevents pose and velocity channels
from disagreeing. Contact channels are trained as probabilities and use `0.5`
as the certification threshold; C++ attachment authority still requires the
geometric Contact gate and never trusts a learned contact bit by itself.

For interaction clips, the frame origin and yaw are the observed target-object
frame used by the existing interaction pack. For walking-only clips, training
constructs an equivalent goal frame from the terminal planar root transform and
marks object/grasp fields absent. At runtime the real target-object frame is
always used.

Quaternion signs are canonicalized before conversion to 6D rotations. Forward
kinematics after decoding must reproduce finite, normalized G1 transforms.

## Conditions

The common condition record contains:

- frozen initial full-body pose;
- frozen root linear and angular velocity;
- active hand;
- object dimensions and support height;
- object-local grasp translation and 6D rotation;
- object-local horizontal approach direction;
- an 80-frame planar route corridor with desired root position and facing;
- an 80-frame phase schedule covering Walk, Approach, Reach, Contact, and any
  early Lift frames present in the source window;
- a condition-presence mask distinguishing generic walking rows from object
  interaction rows; and
- the deterministic candidate seed.

No object ID, clip ID, sequence index, authored slot ID, absolute world X/Z, or
absolute world yaw enters either learned model.

The route corridor is guidance, not ground-truth pose. The planner resamples its
collision-free polyline into 80 root targets using a smooth speed profile that
starts at the frozen planar velocity, remains below the training partition's
95th-percentile walking speed, and reaches the training partition's supported
pre-Contact speed range without a zero-speed waypoint. It anchors global travel
and collision avoidance while leaving gait, body preparation, and exact arrival
dynamics to the coupled models.

## Training Data

### Walking expert

Train on contiguous 50-frame windows from the native 29-DoF G1 walking source
`lafan_walk_short.npz`, converted through the existing G1 kinematics and
canonical 31-bone interaction mapping at 25 Hz. Sample turning, acceleration,
deceleration, and steady locomotion with balanced speed and heading bins. Also
include the 50 pre-Reach frames from every valid interaction example, because
those rows provide the real distribution on which the walking and pickup
experts overlap. Do not retarget the legacy 23-bone Holden database into G1.

### Pickup expert

For each structurally valid interaction clip, locate canonical first Reach frame
`R`. Retain the row only when:

- frames `R - 50` through `R + 29` exist in the clip;
- the first Contact occurs in `R` through `R + 29`;
- the active hand is right;
- object, grasp, support, contact, and phase records are finite and valid; and
- the existing kinematic and interaction provenance checks pass.

The pickup expert target is source frames `[R - 20, R + 30)`. The walking expert
target from the same clip is `[R - 50, R)`. Their real common target is exactly
`[R - 20, R)`, giving 20 contiguous, genuine overlap frames without temporal
resampling or fabricated cross-domain labels.

Rejected-row counts and reasons are serialized in the dataset manifest. Object
IDs remain disjoint across train, validation, and test partitions. Normalization
statistics are computed only from the training partition.

Each retained interaction row also records the source clip and frame needed to
continue from generated frame 79 into that clip's post-Contact Lift. Frozen
planning must certify this continuation and its seam before execution begins.

### Augmentation

Permitted augmentation is limited to transformations that preserve the declared
condition:

- planar rotation and translation through object-frame canonicalization;
- small training-only perturbations of initial root velocity and route corridor;
- mirrored walking-only rows; and
- random middle-frame masks for ordinary inpainting reconstruction.

Interaction rows are not mirrored into unsupported left-hand claims. Object
height, grasp height, geometry, and contact labels are not randomly altered.

## Models and Losses

The walking and pickup experts use the same denoiser architecture, timestep
schedule, frame channels, and normalization contract but have independent
weights. The walking expert is trained from the larger walking corpus. The
pickup expert is trained from object-conditioned interaction windows.

Training combines:

- the standard diffusion denoising objective;
- clean-pose reconstruction in 6D rotation space;
- forward-kinematic hand and foot position losses;
- root and joint velocity and acceleration losses;
- foot-contact and foot-sliding losses;
- active-hand grasp-position and grasp-orientation losses;
- pre-contact object-separation and post-contact attachment losses; and
- overlap agreement between the experts on the genuine shared frames from
  interaction clips.

Loss weights are selected using only the validation partition and serialized in
the checkpoint. Test rows cannot change architecture, weights, sampler settings,
or candidate count.

The first coupled checkpoint is trained before the monolithic baseline. The
monolithic baseline uses the same 80-frame representation, conditions, split,
candidate budget, and certification. Its batches are balanced so walking-only
rows cannot overwhelm interaction rows.

## Simultaneous Shared-Overlap Sampling

One request allocates a global latent tensor shaped `[8, 80, D]`, where `D` is
the shared per-frame channel count. It is initialized once from the request seed.
The two experts therefore receive identical noisy values for global frames
`[30, 50)` by construction.

At every reverse-diffusion timestep:

1. Pass latent slice `[0, 50)` through the walking expert.
2. Pass latent slice `[30, 80)` through the pickup expert at the same timestep.
3. Copy non-overlap predictions directly into one global prediction tensor.
4. For overlap index `j` in `[0, 20)`, combine predictions with pickup weight
   `w(j) = j / 19` and walking weight `1 - w(j)`.
5. Apply initial-state, route, floor/contact, grasp, and attachment guidance to
   the global clean estimate.
6. Perform one global DDIM update and reslice that single updated latent for the
   next timestep.

Because the next step always starts from one global latent, the overlap cannot
diverge between experts. The walking prediction is authoritative at the first
overlap frame, the pickup prediction is authoritative at the final overlap
frame, and both contribute inside the overlap. This is bidirectional coupling;
placing two independent samples in one batch without the global consensus step
does not satisfy the design.

The interactive prototype uses 20 DDIM steps and eight candidates. A frozen
50-step sampler is evaluated as a quality oracle. Production remains at 20 steps
only if, on the frozen validation manifest, its `attach@8` and no-stop completion
are each within two percentage points of the 50-step result, its median grasp
position error is no more than 2 mm worse, and its median grasp orientation error
is no more than 2 degrees worse. Otherwise production uses 50 while performance
work continues.

## Persistent Inference Worker

Replace process-per-request inference with one resident worker launched during
controller startup. The worker:

- imports Torch once;
- initializes CUDA once;
- loads and verifies both checkpoints once;
- performs one warmup request before reporting Ready;
- accepts strict, length-delimited requests over a local pipe;
- returns strict, length-delimited candidate tensors and diagnostics; and
- remains alive for repeated pickup attempts.

Checkpoint hashes are computed at startup, not on every `F` press. A request
carries checkpoint identities, schema version, condition bytes, request ID, and
seed. C++ rejects stale, malformed, identity-mismatched, or non-finite replies.

The simulation remains frozen while sampling, but rendering, cancellation,
worker polling, and planning diagnostics remain responsive. The initial latency
target is less than 2.0 seconds at the 95th percentile after warmup on the
available L40S.

## Runtime Authority and Execution

The learned worker proposes motion only. C++ remains authoritative for target
generation, ownership, collision, reservation, Contact, attachment, Lift
validation, Carry entry, placement, release, cancellation, and reset.

Certification ranks complete 80-frame candidates, never isolated windows. A
candidate must pass:

- exact frozen-start agreement;
- finite transforms and normalized rotations;
- G1 joint limits and per-frame joint-speed limits;
- root, body, table, and obstacle clearance;
- route-corridor deviation limits;
- support-foot floor penetration and sliding limits;
- overlap velocity and acceleration limits;
- active-hand approach and grasp-transform limits;
- valid pre-contact separation;
- valid Contact timing and attachment geometry; and
- a compatible preselected recorded continuation whose certified source segment
  reaches the existing Lift witness.

Score order after hard feasibility is:

1. grasp position and orientation error;
2. collision and clearance margin;
3. overlap root/joint acceleration;
4. foot sliding and contact consistency;
5. route-corridor error;
6. total motion jerk; and
7. candidate index as the stable final tie-break.

After selection, the controller resumes once and executes the generated root and
full-body pose timeline as one authority interval. There is no controller handoff
at global frames 30 or 50 because those are internal model-window boundaries,
not runtime states. At frame 79 it immediately enters the already selected and
certified recorded post-Contact continuation, without stopping or matching. That
continuation supplies the Lift witness, after which the existing inertialized
Hold/Carry transition resumes normal interaction authority.

Material target, support, obstacle, or ownership changes before resume invalidate
the transaction. Tracking escape, collision, or target invalidation during
execution cancels through the existing fail-closed path; the character and object
are never teleported to salvage a candidate.

## Failure Handling

- Worker not Ready: retain manual control and report initialization status.
- Worker crash or malformed reply: terminate the request, unfreeze, and restart
  the resident worker outside the controller update.
- Planning timeout: cancel the request, unfreeze, and report timeout.
- No route corridor: unfreeze and report no safe approach.
- No certified candidate: unfreeze and report rejection counts by reason.
- Target or authority mutation while frozen: discard the reply and unfreeze.
- Runtime tracking or collision failure: cancel the interaction and return to
  the existing safe locomotion state.

Every terminal path proves that physics and animation are unfrozen exactly once.
No failure path invokes authored Smart Pickup from learned mode.

## Evaluation

Freeze one deterministic evaluation manifest before final checkpoint selection.
It contains object-disjoint rows, seeds, starting poses, velocities, object
heights, approach headings, route corridors, obstacles, and metric definitions.

The primary playable matrix covers:

- start distances of 1 m, 2 m, and 3 m;
- at least three starting approach headings;
- stationary, slow-walk, and ordinary-walk initial velocities; and
- every currently certified pickup height represented by the pack.

Every declared case is evaluated for:

- `contact@8`: at least one candidate reaches valid Contact;
- `attach@8`: at least one candidate attaches through runtime authority;
- `lift@8`: the selected generated candidate plus its preselected continuation
  reaches the Lift witness;
- collision-free completion;
- grasp position and orientation error;
- root and joint velocity/acceleration across global frames 29 through 51;
- foot penetration and sliding;
- route deviation;
- generation latency after worker warmup; and
- deterministic reproduction from checkpoint, condition, and seed.

The no-stop requirement is violated when an attempt entering the overlap above
`0.10 m/s` contains three or more consecutive pre-Contact frames below
`0.03 m/s` inside global frames `[30, 55)`. Natural deceleration that does not
create this dwell remains valid.

The coupled model is reported beside:

1. the current root-funnel plus recorded-pickup system;
2. sequential walking generation followed by pickup inpainting with no reverse
   influence from the pickup model; and
3. the monolithic 80-frame diffusion baseline.

The coupled prototype is accepted only when it improves no-stop pickup
completion over the sequential ablation without reducing collision-free
`attach@8`, and when graphical playback confirms that no state-machine pause is
hidden by metric aggregation. Exact rates are reported rather than replacing
failed rows or tuning on test outcomes.

## Verification

Automated verification includes:

- dataset extraction tests for exact source ranges and genuine 20-frame overlap;
- object-disjoint split and training-only normalization tests;
- representation encode/decode and forward-kinematics round trips;
- a test proving both experts receive byte-identical overlap latents;
- a test proving overlap consensus is written back before the next DDIM step;
- endpoint-weight tests for `w(0) = 0` and `w(19) = 1`;
- deterministic candidate-batch reproduction;
- resident-worker startup, warmup, repeated-request, crash, and stale-reply tests;
- frozen-state tests proving simulation time and pose do not advance while
  planning;
- certification tests for collision, joint, foot, grasp, Contact, and Lift
  rejection;
- no-stop metric tests around the overlap;
- complete learned-mode failure-path tests proving no authored fallback; and
- live flat-scene evidence from 1 m, 2 m, and 3 m.

## Deferred Work

- More than one walking window and long-horizon temporal tiling.
- Learned rather than fixed overlap mixing weights.
- Left-hand interaction data and training.
- Broader object, grasp, and approach-direction coverage.
- Terrain-conditioned walking and pickup generation.
- Receding-horizon replanning.
- Physical G1 dynamics and torque control.
