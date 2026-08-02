# G1 Multi-Horizon Terrain Skill Matching Design

## Goal

Replace whole-remaining-clip intent scoring with fixed-horizon outcome matching
and stable-boundary action chunks. The experiment succeeds only if at least two
of the frozen six hard routes pass every existing behavior gate while all six
still execute without exception.

This remains a 50 Hz kinematic experiment. It does not add Sonic, tracking,
physics, depth, a viewer, or a learned model.

## Evidence motivating the change

The PHP-style slice established that terrain placement is not the dominant
failure: support-height error was 1.6--1.9 cm p95 on all six routes. Whole-clip
playback nevertheless caused long stalls, missed reversals, final-surface
errors, and lateral drift. Stable internal entries made the 90-degree pivot
reachable, proving that the corpus contains useful local behavior that is lost
when a ten-second clip is treated as one action.

## Approaches considered

### A. Fixed multi-horizon outcome descriptors — selected

Precompute the local outcome of every stable entry at 0.5, 1.0, and 2.0
seconds, ending at a nearby stable double-support state. Match those outcomes
to the requested command and commit only through the chosen endpoint.

This directly addresses time-local progress and remains deterministic,
interpretable, GPU-vectorizable, and cheap enough to evaluate over the entire
GRAIL corpus.

### B. Learned skill-outcome predictor

Train a network from proprioception, command, and candidate identity to future
root/contact outcomes. This may ultimately generalize better, but it adds a
training and validation problem before the deterministic target has been shown
to work. It is deferred until the fixed descriptor establishes a useful target.

### C. Beam search over stable contact nodes

Search multiple consecutive chunks over the whole command route. This can
optimize long-horizon progress, but it reintroduces the contact-graph complexity
that previously became slow and brittle. It is deferred unless greedy
multi-horizon chunks pass terrain safety but fail only from myopic choices.

## Descriptor inventory

Each searchable entry row owns up to three immutable
`TerrainSkillHorizon` records. Target horizons are 25, 50, and 100 source
frames. For a target, the endpoint is the first stable double-support frame at
or after the target and no more than 25 frames later. A horizon is rejected if
no such endpoint exists or fewer than 25 frames remain.

Each record contains:

- entry row, clip, entry frame, and exclusive endpoint frame;
- root displacement XY in the entry-heading frame;
- accumulated unwrapped yaw change;
- root-height change;
- endpoint support state and foot surface heights;
- per-foot supported surface-height change;
- maximum consecutive source frames below 0.03 m/s;
- source-frame count and duration.

All row-aligned scalar/vector arrays are stored as owned tensors on the matcher
device. Source identities and endpoint frames remain integer tensors. Inventory
construction must not synchronize once per frame.

## Query and layered selection

For each issued command, simulate the current bounded acceleration and yaw-rate
model for 100 frames while holding the requested velocity and heading constant.
Sample its desired local displacement and yaw change at 25, 50, and 100 frames.

Selection runs in four layers:

1. **Entry eligibility:** retain the existing pre-skill rows and internal stable
   double-support rows with a valid horizon.
2. **Coarse hard gates:** reject wrong turn direction, zero progress under a
   moving command, incompatible ascent/descent surface change, and invalid
   endpoints. Do not use the rejected ten-frame stall hard gate.
3. **GPU ranking:** compute one total cost for every eligible `(row, horizon)`:
   existing normalized 27-value entry-pose cost plus normalized displacement,
   yaw, height, duration, and stall-risk outcome costs. Stable row/horizon order
   breaks exact ties.
4. **Terrain validation:** validate ranked candidates against the complete
   placed supported-foot trace through that candidate's endpoint and select the
   first valid result.

The first implementation uses explicit frozen weights. An ablation must record
entry-only, outcome-only, and combined costs for every selected chunk so a
failure cannot be attributed to an opaque aggregate.

## Execution contract

The selected source is placed at the current root using the existing yaw and
translation convention. Existing exact spring inertialization makes the first
emitted pose equal the current pose. Playback then advances every source frame
exactly once through the selected exclusive endpoint.

Replanning rules:

- normal replanning occurs only at the selected stable endpoint;
- a changed operator command latches a replan request and releases the current
  chunk at the next stable double-support frame;
- no chunk ends in flight or single support;
- no search occurs inside a committed chunk before a release condition;
- a moving command with no valid candidate produces a structured failure rather
  than silently holding a static pose.

The executor records the selected horizon, endpoint, component costs, rejection
counts, and release reason on every chunk boundary.

## Failure handling

Malformed descriptor inventories fail before rollout. A route-level search
failure records its frame, command segment, number of candidates surviving each
layer, and exact rejection counts. One route failure cannot abort the remaining
qualification matrix. Timing remains diagnostic and excluded from deterministic
hashes.

## Testing

Unit tests cover:

- stable endpoint selection and the 25-frame lateness cap;
- exact displacement/yaw/height/stall descriptors on synthetic clips;
- row/horizon tensor alignment and deterministic tie-breaking;
- each coarse gate independently;
- combined cost parity against a small independent calculation;
- terrain validation after ranking;
- exact sequential playback with no skipped or repeated source frame;
- latched command changes releasing only at stable double support;
- structured no-candidate failure and pickle-free saved arrays;
- timing-independent deterministic hashes.

Neighboring matcher, contact, FK, and current terrain-skill tests must remain
green.

## Qualification

Run these routes concurrently on separate GPUs:

1. `cross-tread-left-to-right`
2. `turn-90-middle-left`
3. `diagonal-down-left`
4. `side-exit-upper-left`
5. `riser-reversal`
6. `mixed-adversarial`

The experiment continues to viewer/full-matrix work only when:

- all six routes execute without exception;
- at least two routes pass every frozen behavior gate;
- support-height error p95 does not regress above 0.03 m on any route;
- no route introduces a mid-flight boundary or source-sequence violation; and
- the 90-degree route retains its pivot-progress and final-heading passes.

Otherwise the result is recorded as a falsifier, including per-layer candidate
counts and the best component-cost ablation.

## Isolation

Implementation stays in new multi-horizon modules or the already isolated
terrain-skill modules. It does not modify the pre-existing dirty landing bridge,
FK, or contact-segment files. Generated qualification artifacts stay under
`build/` and are never committed.
