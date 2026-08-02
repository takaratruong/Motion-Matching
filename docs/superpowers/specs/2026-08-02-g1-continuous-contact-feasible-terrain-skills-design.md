# G1 Continuous Contact-Feasible Terrain Skills

Date: 2026-08-02

## Goal

Make held directional commands produce continuous stair locomotion while
rejecting source motions whose placed feet or body collide with the query
terrain. Preserve the improved side mounts and side exits from the current
multi-horizon terrain-skill viewer.

This remains privileged-heightmap, 50 Hz, kinematic motion matching. Physics,
Sonic tracking, depth inference, and real-time latency qualification are out of
scope. Search may pause the viewer while evaluating candidates.

## Measured Failure

The current horizon inventory admits endpoints only at frames where both feet
are supported. At each endpoint the matcher performs a global search and starts
a new placed skill. Therefore an unchanged held command repeatedly enters
double support, changes clips, and can stop at the edge of a tread.

Terrain compatibility currently samples supported ankle centers at a stride of
five frames. It checks relative surface height but does not check a landing
footprint, distance from a height discontinuity, the complete swing path, or
other robot links. A candidate can therefore pass compatibility while its sole,
shin, or knee intersects a riser.

The earlier rigid contact-space oracle already implemented landing-edge and
swing-clearance constraints. It reduced penetration but passed only 5 of 21
behavioral routes because rigid one-contact transitions were poorly connected.
This design reuses its feasibility evidence without replacing coherent skill
playback with that failed action representation.

## Considered Approaches

### Selected: coherent continuation plus layered feasibility

Continue the current source skill across a double-support boundary when the
operator command is unchanged and its next contact phase is feasible. Fall back
to global horizon search only when continuation is unavailable or invalid.
Apply the same feasibility pipeline to globally searched candidates.

This preserves recorded cadence and contact order while preventing unchecked
continuations.

### Rejected: scalar double-support and edge costs only

Increasing the existing stall weight and adding an edge-proximity cost would be
cheap, but a finite cost can still select a colliding candidate when all other
terms favor it. Physical invalidity must be a feasibility decision before
ranking.

### Deferred: full contact-oracle planning and contact warping

Contact-aware synthesis or warping may ultimately be required for sparse
motions. The rigid contact oracle's 5/21 result does not justify replacing the
current 16/21 horizon baseline before the smaller continuation and validation
experiment is measured.

## Architecture

### 1. Command and playback state

The viewer distinguishes three cases:

- An initial zero command does not advance the matcher.
- Releasing a moving command finishes the already selected chunk and freezes at
  its authenticated double-support endpoint, retaining the current action-chunk
  fix.
- Holding the identical nonzero command requests continuous continuation. The
  command identity is the exact normalized velocity and heading tuple already
  passed to the matcher; no angular threshold is introduced.

A nonzero command change permits the existing safe-boundary global replan. A
reset clears all continuation and action-chunk state.

### 2. Same-skill continuation

At the end of a selected horizon, the matcher first constructs the next
sequential interval in the current `TerrainSkill`. The interval begins at the
current exclusive endpoint and ends at the next indexed horizon endpoint inside
the same skill. It preserves the existing placed root transform, inertialization
state, foot-correction state, and sequential source-frame order; it does not
restart the clip or apply a new entry blend.

Continuation is accepted only when:

- the command is the same nonzero command that selected the active chunk;
- the source interval has another endpoint and makes at least 5 cm planar root
  progress;
- its maximum moving-command stall is at most five frames;
- the full continuation passes the contact and collision feasibility pipeline;
  and
- the interval remains inside the authenticated terrain domain.

If any condition fails, the matcher performs the ordinary global horizon search
with the same feasibility pipeline. If neither continuation nor a searched
candidate is feasible, it stops at the current safe endpoint and reports
structured rejection counts. It never emits an unchecked fallback.

### 3. Fast contact feasibility

Every candidate is rigidly placed at the current root exactly as during
playback. Validation samples every source frame through the proposed endpoint.
It classifies each foot from the authenticated support mask and checks:

- supported sole-center height error at most 5 cm;
- landing sole-center height error at most 3 cm;
- a landing cross footprint consisting of the center and offsets of 4 cm in
  both planar axes;
- no more than 2.5 cm surface-height range across that footprint;
- unsupported sole clearance of at least -5 mm; and
- source-versus-query landing-height deformation at most 6 cm.

Sampling or domain failures reject the candidate. These bounds are at least as
strict as the earlier contact oracle where visible collision matters, while the
stance and landing tolerances retain enough corpus coverage for the experiment.

The validator returns a typed result containing the first rejection reason,
minimum swing clearance, landing error, stance error, and footprint height
range. Global search retains counts for each rejection layer.

### 4. Exact MuJoCo collision preview

Candidates that pass the vectorized heightmap layer are checked sequentially in
ranked order through native MuJoCo FK and collision generation after the same
foot cleanup used by emitted playback, for every proposed frame. Intended
sole/terrain contact is allowed only for a foot marked supported on that frame.
The candidate is rejected if:

- an unsupported foot geometry penetrates terrain by more than 5 mm;
- a supported sole geometry penetrates terrain by more than 5 mm after cleanup;
- a shin, knee, pelvis, hand, or other forbidden robot geometry penetrates the
  terrain by more than 5 mm; or
- a supported foot contact occurs outside its configured sole geometries.

The first candidate passing this preview wins. There is no wall-clock cutoff.
Preview time, checked candidate count, checked frame count, rejecting link, and
maximum forbidden penetration are diagnostic outputs.

### 5. Ranking valid candidates

The existing motion-feature, displacement, yaw, height, duration, and stall
costs remain. Feasibility is applied before acceptance, not encoded as an
arbitrarily large penalty. Among feasible candidates, add independently
reported soft terms for:

- landing footprint height range;
- negative swing-clearance margin;
- moving-command double-support duration; and
- clip-transition cost, with zero transition cost for accepted sequential
  continuation.

Same-skill continuation is evaluated first because it is the only option that
preserves the recorded gait without a boundary blend. A global candidate may
replace it only when continuation fails a hard constraint, ends, or the command
changes.

## Data Flow

1. Map held WASD state to the existing normalized command.
2. At a chunk endpoint, classify the command as stop, unchanged motion, or
   changed motion.
3. For unchanged motion, build and validate the same-skill continuation.
4. If continuation fails, rank global horizon candidates.
5. For each ranked candidate, run fast contact feasibility followed by exact
   MuJoCo collision preview.
6. Commit the first passing candidate transactionally and record its validation
   and cost decomposition.
7. If no candidate passes, remain at the safe endpoint with a visible failure
   reason.

## Isolation

The experiment extends the multi-horizon terrain-skill path only. It does not
modify the ordinary flat matcher, the retained contact oracle, the uncommitted
landing-bridge work, Sonic integration, or tracking. New thresholds are explicit
multi-horizon configuration fields and preserve the current viewer defaults
until the combined experiment qualifies.

## Evaluation

### Unit and integration tests

Tests must prove:

- unchanged nonzero commands continue the same skill without restarting it;
- zero commands settle and freeze rather than continuing;
- command changes use the existing safe-boundary replan;
- continuation never skips or repeats a source frame;
- landing edge, unsupported swing, forbidden-link, and terrain-domain failures
  reject before commit;
- supported sole contacts remain allowed;
- no feasible continuation falls back to global search;
- failed continuation can select a valid global candidate; and
- no feasible candidate leaves the previous endpoint unchanged and reports the
  exact rejection reason.

### Required ablations

Run four variants on identical commands and artifacts:

1. retained multi-horizon baseline;
2. continuation only;
3. feasibility only; and
4. continuation plus feasibility.

This distinguishes cadence improvement from collision improvement and prevents
crediting one subsystem for the other.

### Behavioral qualification

The first feedback build must include:

- a held-W traversal from reset through the upper landing with no key release;
- the same ascent with release and restart at a tread;
- diagonal ascent/descent, cross-tread, both upper side exits, an elevated
  reversal, and a 90-degree stair turn;
- adversarial side approaches and edge-parallel walking; and
- a saved MuJoCo contact sheet for every tested route.

Compared with the retained source-quality baseline, the combined variant must:

- complete the held-W ascent without a visible double-support pause at every
  chunk boundary;
- have no forbidden MuJoCo terrain penetration deeper than 5 mm;
- have no accepted landing footprint exceeding the 2.5 cm edge-height range;
- keep moving-command stalls at five frames or fewer;
- retain all six frozen route outcomes and at least 16 of 21 broad outcomes;
- not regress either upper side exit; and
- reduce the number of clip transitions during held-W ascent.

Search latency is reported but is not an acceptance gate. Visual review remains
mandatory: passing route counters alone do not qualify the viewer.

## Deliverable

The deliverable is an isolated interactive viewer option, deterministic
ablation artifacts, collision/contact diagnostics, a route comparison report,
and a recommendation stating whether coherent continuation and feasibility
should replace the current multi-horizon defaults.
