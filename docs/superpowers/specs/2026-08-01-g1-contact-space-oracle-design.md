# G1 Contact-Space Terrain Oracle

Date: 2026-08-01

## Goal

Determine whether the GRAIL motion corpus can produce good omnidirectional
stair kinematics when candidate selection is allowed to plan several contacts
ahead without a real-time limit.

This experiment replaces the current greedy terrain-action selection with an
offline contact-space oracle. It must distinguish two failure classes:

1. a valid sequence exists, so the current greedy search and long commitments
   are the problem; or
2. no valid sequence exists, so the discrete action representation or corpus
   needs continuous warping, IK, or additional motion coverage.

The experiment is privileged-heightmap, kinematic, and deterministic. Sonic,
tracking, depth inference, physics, and real-time latency are out of scope.

## Current Failure Being Replaced

The retained layered graph hybrid constructs desired two-foot foothold plans,
but ordinary non-turn action selection uses contact identity and landing-height
compatibility as hard gates and discards the plan's XY and timing residual from
the ranking. It then greedily selects one motion-matching row and may commit to
as much as 60 frames of source playback. The selection does not prove that the
resulting support state has a feasible continuation.

The 95-value dense terrain feature remains useful as a diagnostic, but it is
not a sufficient planner. Its 91-point rectangular body-frame patch encodes
terrain appearance rather than the future contact task and over-specializes
retrieval to the recorded root-to-stair orientation.

## Chosen Approach

Build a receding-horizon beam-search oracle over contact-phase actions. Each
search edge represents one swing-foot landing and ends at a stable support
event. The oracle searches three to four future landings, executes only the
first edge, and replans from the new stable support state.

This is preferred over two alternatives:

- Reweighting or resampling the dense terrain grid cannot repair greedy dead
  ends or ignored landing XY/timing.
- Training a PFNN-style generator now would learn from a teacher whose contact
  decisions are visibly wrong, making failures harder to diagnose.

Continuous foot/root warping is deliberately deferred until the discrete
oracle reveals its residual placement error.

## Components

### 1. Contact-phase action index

Build an immutable index from the existing authoritative MuJoCo FK and contact
segmentation. An action begins at a stable support state and ends at the next
stable landing of the opposite or previously swinging foot. It stores:

- clip and inclusive frame bounds;
- entry and exit support masks;
- entry/exit joint position and velocity;
- source root SE(2) trajectory;
- left and right foot trajectories in the action-entry frame;
- swing-foot identity, landing XY/Z, and landing frame;
- minimum source swing clearance;
- terminal root displacement, velocity, and facing change; and
- exact sequential successor, when one exists.

The index must exclude ambiguous actions with no stable landing, invalid FK,
or inconsistent contact ordering. It must report the retained/rejected action
counts and reasons rather than silently dropping malformed data.

### 2. Search state and placement

A search state contains the world root SE(2), both world foot positions, support
mask, joint boundary state, accumulated cost, parent edge, and depth. A source
action is placed into the world with one planar rigid transform anchored to its
entry root. No per-frame deformation is allowed in this first experiment.

Applying an edge produces the action's transformed terminal root, feet, support
mask, and joint boundary state. Search nodes are immutable so alternative beams
cannot contaminate one another.

### 3. Terrain and contact feasibility

Terrain feasibility is evaluated directly on transformed foot trajectories,
not through nearest-neighbor terrain-feature distance.

Hard constraints are:

- every stance-foot sample remains within the configured terrain-height
  tolerance;
- the landing sole lies on a locally stable patch with edge margin;
- the swing foot does not penetrate the terrain before landing;
- landing height change is within the observed source action's admissible
  deformation tolerance;
- support order is compatible with the parent state; and
- the action's entry joint state is within a broad transition bound.

The first version permits the same small rigid placement tolerance already used
by the terrain matcher but does not alter source foot height, stride, timing, or
joint motion.

### 4. Search objective

The oracle performs deterministic beam search with a default width of 256 and
a default horizon of four landings. There is no wall-clock cutoff. Candidate
ordering uses a stable source-row tie break.

Hard feasibility is applied before scoring. The soft cost contains independently
reported terms for:

- command-path progress and lateral error;
- facing error, independent of travel direction;
- exact landing XY error to the planned foothold;
- landing timing error;
- entry joint-position and joint-velocity discontinuity;
- stance-foot motion;
- swing clearance margin;
- terminal reachability, estimated by the number and quality of feasible next
  actions; and
- repeated source/action use.

Terminal reachability prevents the present failure in which a locally cheap
action enters a dead end. Costs are summed across the horizon with no learned
weights. The implementation exposes every component and supports deterministic
single-term ablations.

### 5. Receding-horizon execution

At each stable support event the oracle searches a new horizon and commits only
the first contact-phase edge. During that edge, source frames play sequentially.
The planner may re-evaluate after command changes, but the initial scripted
evaluation changes commands only at stable support events so search behavior is
unambiguous.

If no complete horizon exists, the oracle first shortens the horizon down to one
landing and reports `short-horizon-only`. If no single feasible action exists,
it stops with a structured failure containing the rejected constraint counts.
It must never substitute an unchecked fallback action.

### 6. Evaluation and artifacts

The oracle runs on the existing centered GRAIL staircase and the following
scripted classes in both left/right variants where applicable:

- straight ascent and descent;
- diagonal ascent and descent;
- cross-tread lateral walking;
- lower and upper side mount/exit;
- 45, 90, and 180 degree turns while elevated;
- reversal while elevated;
- stop and restart on a riser; and
- the mixed adversarial route.

Each route emits:

- selected action sequence and placements;
- per-edge and cumulative cost decomposition;
- beam survival and rejection counts by constraint;
- planned versus emitted foot landings;
- progress, final surface, and final heading;
- stance slide, penetration, joint discontinuity, and jerk; and
- an inspectable MuJoCo kinematic rollout.

A comparison report places the oracle beside the retained layered graph hybrid
on the identical route definitions.

## Acceptance and Interpretation

The oracle is considered evidence that the search formulation was the primary
problem when all of the following hold:

- it completes at least 18 of the 21 existing route variants;
- both cross-tread routes, all four side exits, both diagonal descents, and the
  elevated reversal complete;
- no selected edge violates contact-height, edge-margin, or swing-clearance
  constraints;
- longest moving-command stall is at most five frames; and
- aggregate stance slide is at least 50 percent below the retained comparison.

Visible quality remains a required review gate; numerical completion alone does
not qualify the oracle.

Interpretation is explicit:

- Good sequence and good contacts: replace the online greedy selector with a
  bounded version of this planner.
- Good sequence but systematic landing offsets or slide: retain the planner and
  add contact-aware root/foot trajectory warping plus stance-foot IK.
- No feasible sequence despite broad transition bounds: audit action extraction
  and corpus coverage before changing search weights.
- Feasible scripted routes but poor live command changes: add interruptible
  within-swing replanning only after the offline representation qualifies.

## Testing

Unit tests cover action boundary extraction, rigid placement, terrain sampling,
contact feasibility, cost decomposition, deterministic beam pruning, horizon
fallback, and immutable parent states. Synthetic fixtures must include a case
where greedy selection chooses a dead end while lookahead chooses the only
completable sequence.

Integration tests build a small pinned action index and run at least one ascent,
cross-tread, turn, and side-exit route on CPU. The full 21-route comparison runs
on an available GPU and records deterministic artifact hashes.

Existing matcher behavior remains unchanged. The oracle is added as an isolated
experimental path and does not modify the retained live viewer until it passes
the evaluation gate.
