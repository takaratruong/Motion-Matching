# Object/Grasp-Conditioned Diffusion Funnels

Date: 2026-07-19

Status: approved for implementation planning

## Purpose

Replace the three fixed Smart Pickup approach slots with a learned proposal
distribution while retaining the verified parts of the interaction pipeline.
Given a grasp affordance in an object's local frame, a small conditional
diffusion model proposes several short object-local root trajectories. Ordinary
flat locomotion connects the character to a proposal, deterministic gates reject
colliding or incompatible proposals, and the existing interaction motion matcher
still owns Reach, Contact, attachment, Lift, Hold, Carry, placement, and release.

The first checkpoint answers one narrow question:

> Can a learned object/grasp-conditioned proposal distribution provide useful,
> diverse Smart Pickup funnels without weakening the existing deterministic
> interaction contracts?

The model is not an end-to-end manipulation policy. It proposes local paths;
the runtime remains authoritative.

## Decisions

- Simulation, controller updates, learned trajectories, evidence, and video use
  exactly 25 Hz (`0.04` seconds per sample).
- The first learned model uses the already-built full GRAIL interaction pack.
  It does not parse the 50 GB raw `pickup_table` PKLs.
- The model operates in the target object's planar frame. Absolute object world
  X/Z translation and yaw only transform generated trajectories into the scene
  and are not learned inputs. Gravity-relative support and grasp heights remain
  explicit conditions because low, table, and shelf interactions are not
  vertically covariant.
- V0 generates a fixed 16-sample local root trajectory, not a single endpoint.
- Generation is object/grasp anchored. Dataset export reverses each raw
  entrance-to-terminal trajectory into terminal-to-entrance outward order. The
  model samples that outward representation and is not conditioned on every
  possible live character start. Proposal export reverses it exactly once into
  the forward execution order stored in the runtime artifact.
- One frozen batch contains 32 proposals. Rejection examines that batch once;
  it may not resample until a success appears.
- Obstacles are not diffusion conditions in V0. Existing deterministic swept
  collision checks reject bad proposals and allow another proposal from the
  same batch.
- The current flat locomotion system connects the live character to the first
  funnel sample. The learned funnel then supplies the remaining local route.
- The existing global interaction matcher chooses the recorded pickup motion at
  the terminal root. Existing IK and contact/attachment checks remain unchanged.
- Python performs V0 training and sampling offline. The controller loads a
  versioned proposal artifact; it never invokes Python or Torch inside the 25 Hz
  update.
- The known authored grasp is used for the first playable proof. A
  GraspMolmo-compatible input boundary is added, but GraspMolmo inference is a
  later checkpoint because a single-grasp scene gives it no meaningful choice.
- Authored Smart Pickup remains an explicitly labelled product fallback, but
  learned-mode acceptance tests cannot consult or silently fall back to authored
  slot geometry.
- Flat ground only. Terrain integration remains outside this branch.

## Scope

V0 includes:

- a trustworthy repeated table/rack evaluation scene with one object, explicit
  current support, and an explicitly highlighted destination;
- several certified pickup/place heights and surface-local offsets where the
  existing full chain genuinely succeeds;
- a small height-indexed precomputed reversed-pickup library spanning those
  heights, with truthful source provenance and the existing same-pickup reversal
  retained only as a near-height fallback;
- a processed-pack funnel dataset with an object-disjoint split;
- nearest-neighbour and endpoint-only baselines;
- a small conditional DDPM trained and sampled with deterministic seeds;
- a versioned learned-funnel proposal artifact;
- deterministic connector, kinematic, swept-collision, and runtime-preview
  certification;
- waypoint-following through ordinary flat locomotion controls;
- actual contact-gated attachment and Carry; and
- native-25-Hz headless and graphical evidence.

V0 does not include:

- RGB-D perception or object-pose estimation;
- downloading, fine-tuning, or synchronously invoking GraspMolmo;
- a stable-grasp proposal network such as M2T2;
- obstacle-conditioned diffusion or classifier guidance;
- arbitrary global navigation or narrow-passage planning;
- generated Contact, attachment, Carry, placement, or release motion;
- several simultaneously selectable objects;
- ground pickup ingestion;
- doors, drawers, or shelf-cavity collision reasoning;
- two-handed interaction; or
- claims of novel-object manipulation beyond the declared proposal-model split.

## Terminology and authority

### Grasp condition

`GraspCondition` is the learned system's complete semantic/geometry input:

- active hand;
- object-local hand/grasp translation;
- object-local continuous 6D rotation representation;
- horizontal object-local approach direction;
- object dimensions;
- support-plane height above flat ground;
- grasp height relative to the support plane.

It contains no object ID, sequence ID, clip/frame ID, authored slot ID, or
absolute world X/Z/yaw. A later grasp selector may populate this same record.
The two serialized height scalars are authoritative. Grasp height above flat
ground is derived exactly as their sum and is not serialized independently.
Dataset export and artifact loading recompute the world grasp height from the
frozen object/grasp/support transforms and reject disagreement greater than
`0.00002 m`. The condition digest covers both authoritative height scalars and
all other fields exactly once.

### Learned funnel

A learned funnel is the runtime artifact's entrance-to-terminal ordered array of
16 object-local root samples. Each sample contains
`(x, z, sin(yaw), cos(yaw))`. It ends at a prospective interaction-entry root.
For a non-degenerate funnel, total planar arc length must be at least `0.15 m`
and a stable-order greedy scan must produce at least three meaningful waypoint
clusters. The first sample starts the first cluster; a later sample starts a new
cluster only when it differs from the current cluster representative by at least
`0.05 m` planar distance or `5 degrees` yaw. Epsilon-only changes cannot satisfy
the funnel claim.

The raw chronological training window is entrance-to-terminal and ends at first
Reach. Dataset export reverses it exactly into terminal-to-entrance,
object-anchored outward order for diffusion training. The model samples outward
order. Proposal export reverses every sampled array exactly once into
entrance-to-terminal execution order. `LearnedFunnel`, connector selection,
collision checks, ranking, artifact serialization, and online following consume
only execution order. Both representations and a canonical reversal digest are
retained for audit; no component infers order from geometry.

The reverse stochastic denoising process is not treated as physical motion time.
"Backward" here describes the task factorization and canonical boundary, not the
DDPM timestep convention.

### Deterministic connector

The connector is the ordinary locomotion segment from the post-step live root to
the first learned sample. It is not model output. It is included in distance,
clearance, and timeout accounting and may reject a proposal.

### Runtime authority

The learned system never chooses a pickup clip, event frame, attachment state,
or object owner. The frozen target generation, affordance, learned proposal,
terminal root, and live locomotion snapshot are passed through the existing
preview/preflight boundary. `InteractionRuntime` remains the only authority that
can reserve, attach, hold, place, or release the target.

## Evaluation scene and controls

The visible scene keeps the existing controls:

- WASD: manual locomotion;
- F: pick while in Locomotion, place while in Carry;
- X: cancel the current interaction attempt;
- R: reset the scenario; and
- arrow keys: camera.

The scene contains the existing table plus an open rack. A semantic slot catalog
identifies every physical support/goal using:

```text
semantic slot ID
surface handle
placement affordance ID
human-readable label
```

The initial directed cycle alternates table and rack goals and covers left,
centre, right, shallow, and deep offsets. Low, middle, and high labels are backed
by actual full-chain witnesses rather than nominal IK limits. The required
positive set contains three data-backed heights with a lowest-to-highest span
strictly greater than `0.24002 m` (`2 * 0.12 m` plus the serialized `0.00002 m`
comparison epsilon), so one source cannot cover the entire set through the
allowed vertical correction alone. Every height completes actual pickup,
attachment, placement, release, and re-pick; reports its selected motion source
and applied vertical correction; stays at or below the `0.12 m` correction
limit; and serves at least once as a pickup source and once as a placement
destination. If the data cannot certify that set, the checkpoint is reported
incomplete with the measured supported range; the scene does not display
cosmetic unreachable tiers as successes.

Only the highlighted destination can be latched for placement. Stacked surfaces
are never inferred with `resolve_single_surface`, whose planar ambiguity is
intentional. The route advances only after the runtime has completed a successful
release, the target is `Free` on the exact selected support, and Locomotion has
resumed. A failed or cancelled attempt retains the current route. R restores the
initial object/support/route state.

The HUD shows the current source, highlighted destination, learned proposal ID,
model/artifact identity, connector distance, rejection counts by reason, frozen
runtime match, target generation/owner/state, attachment, and cycle/leg count.

### Placement motion support

The live controller currently constructs an empty genuine recorded-place
library, so it always reverses the pickup clip selected at the source. Because
that fallback uses planar alignment, destination-height changes are limited by
the existing `0.12 m` hand-correction gate. V0 therefore builds a small
height-indexed precomputed reversed-pickup library before claiming varied shelf
heights.

Only genuine authored put-down clips may instantiate `RecordedPlace` or receive
recorded-place priority. Reversed pickup clips or pickup continuations retain an
explicit `PrecomputedReversedPickup` source kind and complete pickup provenance;
distinct source IDs never make them recorded placement. Every precomputed row
preserves contiguous pose/object samples, contact until release, release and
retraction events, grasp compatibility, source support, object bounds, and the
existing collision/IK gates.

Selection priority is:

1. genuine compatible `RecordedPlace`;
2. height-compatible `PrecomputedReversedPickup`; and
3. `ReversedPickup` of the immediately preceding pickup as the labelled
   near-height fallback.

If no genuine put-down exists, V0 reports exactly that it uses a height-indexed
precomputed reversed-pickup library. Low, middle, and high positive witnesses
must select distinct source IDs and publish their truthful source kind.

## Dataset

### Source

Use the certified 25 Hz full interaction pack already consumed by the runtime.
It contains 2,045 clips, 511,250 frames, and 633 packed object IDs, including
per-frame root/body pose, object pose, phase/contact state, support geometry,
active hand, object dimensions, grasp transform, and approach direction.

### Example extraction

For every structurally valid pickup clip:

1. locate the canonical first Reach frame and later Contact/Lift/Hold events;
2. use the same Contact-minus-one object frame as the existing matcher;
3. require a complete 16-sample window ending at the first Reach frame;
4. map every Simulation-root X/Z/yaw sample into that object frame;
5. derive the active-hand grasp transform and the complete `GraspCondition`;
6. apply the existing finite, table/object-clearance, hand-compatibility, and
   event-order validation; and
7. reject incomplete windows instead of padding across phase boundaries.

Output trajectories and conditions are normalized from training-partition
statistics only. Quaternion signs and yaw wrapping are canonicalized before
conversion to the continuous representations. Export retains both the
terminal-to-entrance outward training array and the exactly-once-reversed
entrance-to-terminal execution array plus their reversal digest.

### Split

Partition packed object IDs deterministically into 80% training, 10%
validation, and 10% test. Every clip for one object belongs to exactly one
partition. Dataset normalization, deduplication, model selection, and early
stopping use only training/validation data.

The existing raw-data held-out set is not silently promoted into this claim;
results are labelled "proposal-model held-out." A later raw-data artifact can
evaluate the separately held-out objects.

The export records pack hashes, extraction configuration, ordered object-ID
partitions, normalization values, rejected-row counts/reasons, and a canonical
content digest.

### Frozen evaluation manifest

Before model training, hyperparameter selection, or final proposal generation,
write and commit one canonical evaluation manifest. It contains every test
object/condition, the fixed runtime subset, live initial roots, target/support
geometry, obstacle arrays, training and inference seeds, the exact `K=32`
proposal budget, and complete metric definitions. Runtime rows are selected by a
stable-key rule from the object-disjoint test partition, never by visual
inspection.

The nominal, blocked-alternate, and all-blocked scenes are frozen in that
manifest before final model evaluation. The blocked obstacle is derived from the
predeclared nearest-neighbour baseline path and may not be repositioned after
inspecting diffusion output. Every declared row is reported; invalid, failed,
or duplicate rows may not be omitted. Test outcomes may not change the model,
sampler, seed, blocker, condition, or artifact. The accepted proposal artifact
must regenerate byte-for-byte from the frozen checkpoint, sampler configuration,
condition, and seed.

## Model and baselines

### Diffusion model

V0 uses a small conditional 1D diffusion network over the flattened 16-by-4
trajectory. A timestep embedding and `GraspCondition` embedding modulate a
compact residual denoiser. Training uses deterministic seeds and ordinary noise
prediction. Sampling uses a fixed-step DDIM schedule and produces exactly 32
ordered proposals for one condition/seed set.

Exact layer sizes, optimizer settings, noise schedule, and training budget are
frozen in the implementation plan and serialized into the checkpoint metadata.
The network is intentionally small enough to train on one available L40S; more
GPUs are not part of the reproducibility contract.

### Required baselines

Nearest-neighbour complete-funnel retrieval, endpoint-only
retrieval/regression, and diffusion are evaluated on the identical
object-disjoint test manifest with the identical 32-proposal budget, connector,
collision gates, and runtime previews. The current authored-slot controller is a
separate production-compatibility reference on its known demo target. It is not
included when determining the best held-out baseline unless equivalent authored
slots exist for every test row.

`attach@32` uses every declared runtime case as its denominator, including cases
with no valid proposal. Valid endpoint clusters use the existing `0.10 m` and
`10 degree` greedy deduplication rule in stable proposal order. Diffusion
justifies continuation only when its `attach@32` is at least the best
same-manifest baseline and it either strictly exceeds that baseline's valid
endpoint-cluster count or succeeds in the frozen blocked-alternate case where
that baseline fails. If a simpler baseline wins, the proposal-provider interface
is retained and the simpler provider becomes the default.

Path clusters are reported separately in stable proposal order. Two valid paths
belong to one cluster only when every corresponding one of their 16 samples is
within `0.10 m` planar distance and `10 degrees` yaw. These endpoint/path rules
are serialized in the pre-training evaluation manifest and cannot change after
test output is observed.

## Proposal artifact

Offline sampling writes one versioned artifact containing:

- format version and schema digest;
- full-pack identity;
- dataset/export identity;
- model checkpoint and normalization identities;
- complete `GraspCondition`;
- fixed inference seed and sampler configuration;
- exactly 32 ordered funnels;
- per-funnel finite/kinematic diagnostics; and
- a canonical artifact digest.

The controller validates the complete artifact before enabling learned mode.
World-space object X/Z translation and yaw changes do not require resampling:
the same local funnels transform covariantly with the target. A changed local
grasp, object dimensions, active hand, support height, or support-relative grasp
height invalidates the artifact and requires a newly sampled bundle. Loading
also rejects any mismatch in the derived world grasp height.

## Online selection and data flow

```text
F in Locomotion
    -> freeze target ID/generation and GraspCondition
    -> load/validate matching proposal artifact
    -> read each object-anchored sample in entrance-to-terminal execution order
    -> transform all 32 funnels into world space
    -> validate connector and every learned segment
    -> reject table/object/scene collisions and malformed kinematics
    -> preview every surviving terminal root on one live snapshot
    -> rank certified proposals deterministically
    -> freeze one proposal
    -> ordinary flat locomotion follows connector then all 16 timed samples
    -> final live preview/preflight
    -> existing global interaction match and contiguous pickup playback
    -> contact-gated attach -> Hold -> Carry
```

Ranking uses, in order:

1. complete certification;
2. connector plus funnel route length quantized to millimetres;
3. initial heading change quantized to milliradians;
4. minimum swept clearance, preferring larger clearance;
5. runtime-preview total cost; and
6. stable proposal index.

The selected proposal never changes during an attempt. After connector arrival,
the follower publishes each learned sample exactly once in order at 25 Hz; it
does not skip directly to the endpoint, duplicate frames, or write the character
root. Tracking error is bounded and a missed deadline cancels the attempt.
Remaining segments are revalidated every tick against the same frozen scene
inputs. A newly blocked route cancels cleanly; it does not switch proposals after
movement has begun.

## Collision and failure handling

Collision checks sweep the existing root proxy over the connector and every
funnel segment. They also reuse the current table/object terminal rules and the
runtime's realized hand-path preview. Endpoint-only clearance is insufficient.

Failure behavior is fail-closed:

- a missing, stale, or malformed artifact disables learned mode with a visible
  reason;
- an incompatible grasp/condition does not load a nearby artifact;
- if all 32 proposals fail, no request is submitted and the character does not
  move;
- an all-routes-blocked scene may not fall back silently in learned-mode gates;
- preview/preflight failure releases no ownership and returns responsive
  Locomotion;
- X cancels an active connector/funnel or runtime interaction; and
- R restores the scenario without advancing the table/rack route.

The existing authored provider may be invoked only through an explicitly
labelled manual fallback path outside learned-mode acceptance evidence.

## GraspMolmo boundary

GraspMolmo is a later source of `GraspCondition`, not a trajectory generator.
Its released interface uses an RGB observation and task instruction to identify
a semantically appropriate image-space grasp location, then chooses among
externally supplied stable 6-DoF grasp proposals. A future adapter therefore
has this boundary:

```text
RGB/task + stable object-local grasp candidates
    -> selected candidate ID
    -> authoritative object-local hand/grasp and approach
    -> combine with known object/support vertical context
    -> GraspCondition
    -> learned funnel proposal provider
```

GraspMolmo inference runs asynchronously on interaction intent, is cached by
scene/task/candidate identity, and never blocks the 25 Hz controller. Timeouts or
unavailable weights retain an explicitly authored default grasp. Actual image
capture, depth/calibration, external grasp proposal generation, and multi-grasp
objects receive their own later design.

## Verification

### Dataset and model gates

- exact object-disjoint partitions and no object-ID leakage;
- no object/slot/clip/frame/sequence identity in model inputs;
- training-only normalization;
- deterministic export and fixed-seed inference summaries;
- finite 16-sample trajectories with normalized yaw representation;
- at least `0.15 m` total planar arc length and three meaningful waypoint
  clusters under the frozen `0.05 m` / `5 degree` rule;
- bounded per-sample translation/yaw changes; and
- nearest-training-funnel distance plus exact-duplicate reporting.

### Geometry and covariance gates

- applying a world X/Z translation and yaw to the same condition produces
  byte-identical local proposals and correctly transformed world proposals;
- every connector/funnel segment is swept for collision;
- one fixed nominal scene accepts the highest-ranked clear proposal;
- one predeclared blocker collides with that nominal proposal and selects a
  different collision-free proposal from the same frozen batch; and
- an all-routes-blocked scene submits no request and causes no root motion.

### Actual interaction gates

For at least three predeclared proposal-model-held-out conditions:

- the controller traverses every connector/funnel sample without teleporting;
- all 16 timed sample targets are published once in order at 25 Hz and observed
  tracking error remains inside the frozen implementation bound;
- runtime progresses through Preflight, Align, PickupReplay, Hold, and Carry;
- exactly one unattached-to-attached edge occurs;
- the registry reports the target `Held` with the exact request owner;
- runtime diagnostics report `Carry`, `Held`, and `attached=true`; and
- an ordinary subsequent Carry tick moves the root/object while preserving the
  grasp transform.

Reach animation, request submission, or Align alone is not success. The existing
optional attachment witness becomes a required gate.

### Repetition and placement gates

Every declared positive height/offset matrix row executes two complete
table-to-rack-to-table round trips in one unchanged runtime/controller/registry.
Each row proves four actual attachments, four releases, four successful
re-picks, stable object ID, and final generation `g+4`, without Reset, respawn,
registry reconstruction, or target re-authoring.

Every declared negative height row is mandatory. Pickup rejection leaves the
target `Free`, unattached, owner zero, and generation unchanged. Placement
rejection retains `Held/attached` without release or generation change and must
subsequently recover through a valid placement in the same runtime.

### Metrics

Report at minimum:

- `valid@32` after connector/collision/preview certification;
- `attach@32` after actual Carry/Held/attached, with all declared runtime rows in
  the denominator;
- blocked-scene alternate success;
- all-blocked fail-closed success;
- endpoint and path cluster counts;
- distinct matched runtime clips as informational visual diversity;
- connector/funnel distance and time;
- rejection counts by reason;
- table/rack attach/release/re-pick totals; and
- normal/release stable-result parity.

Graphical evidence includes a native-25-fps representative nominal pickup, the
blocked-nominal alternate, an all-blocked rejection, and a table/rack round trip.
The HUD exposes enough identity/state to distinguish genuine learned selection
and attachment from animation-only playback.

## Operational constraints

- Work only on `g1-tabletop-placement` and its isolated worktree.
- Do not merge or modify terrain work.
- Preserve the terrain controller process/window.
- Do not access, stat, hash, execute, modify, stage, delete, or enumerate the
  protected repository-root untracked `interaction_query_probe` artifact.
- Inspect status only with `git status --short --untracked-files=no`.
- Stage only explicit paths; never use `git add .` or `git add -A`.
- Commit and push every reviewed implementation checkpoint.
- Store temporary videos and evidence outside the repository under the existing
  verification-project directory.

## Implementation decomposition

This umbrella design requires two sequential implementation plans.

### Plan A: truthful repeated table/rack baseline

1. Make actual Smart Pickup attachment a required gate.
2. Add the explicit table/rack catalog and frozen height/offset matrix.
3. Add truthful `RecordedPlace`, `PrecomputedReversedPickup`, and
   same-pickup `ReversedPickup` source modes and priority.
4. Build/certify the height-indexed precomputed reversed-pickup library.
5. Prove every positive and negative matrix row, repeated release/re-pick, and
   native-25-Hz graphical evidence.

### Plan B: learned funnel provider

1. Export the object-disjoint 16-sample funnel dataset, split, frozen evaluation
   manifest, and same-manifest baselines.
2. Train/evaluate the tiny diffusion model and write a reproducible frozen
   proposal artifact.
3. Add learned proposal validation, ranking, timed funnel following, and
   explicitly labelled fallback behind the existing Smart Pickup boundary.
4. Prove nominal, blocked-alternate, all-blocked, actual attachment, and learned
   table/rack behavior at native 25 Hz.

Plan B may begin only after Plan A's real full-chain matrix passes. The
asynchronous multi-grasp/GraspMolmo adapter receives its own later design and
implementation plan.
