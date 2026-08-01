# G1 PHP-Style Terrain Skill Composer

Date: 2026-08-01

## Goal

Produce coherent privileged-heightmap G1 stair kinematics by applying the
first, kinematic stage of Perceptive Humanoid Parkour (PHP) to the existing
PyTorch motion matcher and GRAIL corpus. Motion matching selects a compatible
entry into a complete terrain interaction; the selected contact-rich skill is
then played sequentially with inertialized entry and exit.

The first milestone remains kinematic. Sonic tracking, physics, depth,
proprioceptive distillation, and real-time latency are out of scope until the
same 21-route stair benchmark qualifies.

## Evidence and Scope

The rigid one-contact oracle passed 5 of 21 behavioral routes and exhausted all
feasible one-contact successors on two routes. Its failures and visible source
boundary jumps show that independently selecting every landing destroys the
longer motion structure contained in GRAIL clips.

PHP uses standard motion matching to connect locomotion to a manually annotated
skill entry window. It then disables matching during the contact-rich skill,
plays the skill sequentially, and uses inertialization at transitions. This
design adopts that composition rule without assuming PHP's unreleased code.

The authenticated corpus contains 698 stair clips. A whole-clip audit found
351 forward-dominant, 43 lateral-dominant, 264 diagonal, 332 with at least 45
degrees of accumulated yaw, and 227 with at least 90 degrees of accumulated
yaw. This is sufficient to test coherent skill selection before adding a
learned generator. These bins are descriptive rather than acceptance evidence;
the frozen route matrix decides whether usable skills exist.

## Considered Approaches

### Selected: PHP-style complete terrain skills

Extract terrain episodes with entry windows, use the existing 27-value matcher
feature to select an entry, gate on target/source terrain compatibility, and
play the complete skill sequentially. This preserves source timing, contact
order, pelvis motion, and coordinated turns.

### Rejected: continue one-contact oracle tuning

The exhaustive audit already showed that wider search cannot create missing
rigid transitions. Continuing to change weights or grid density would repeat a
falsified approach.

### Deferred: PFNN-style terrain-conditioned generation

A learned generator may ultimately be needed for interpolation between skills,
but it should not be the first replacement. The coherent GRAIL episodes and the
PHP composition baseline provide a simpler, inspectable teacher and reveal
whether the corpus already contains the required behaviors.

## Architecture

### 1. Immutable terrain skill inventory

Each non-flat GRAIL clip is segmented into one or more `TerrainSkill` records.
A skill owns:

- source clip index and stable identity;
- entry-window start and stop frames;
- committed playback start and inclusive terminal frame;
- stable entry and exit support masks;
- source root, joint, foot, and contact trajectories;
- the paired source height grid and motion-to-terrain transform;
- the source terrain samples under both feet;
- root-local displacement and facing trajectory; and
- deterministic provenance and rejection reason.

Terrain activity is detected from authenticated foot support and source terrain,
not from clip names. A contact-rich episode begins at the last stable
double-support state before either supported foot changes surface height by at
least 8 cm from the approach baseline. It ends at the first stable
double-support state after the final changed-height support phase. The entry
window is the preceding 50 searchable frames, clipped at the source range
boundary. Multiple separated episodes in one clip become separate skills.

An episode is rejected when it lacks stable entry/exit support, an entry window,
finite authoritative FK, or a complete paired height grid. Rejections are
counted explicitly. Exact sagittal mirrors reuse the already verified G1 joint
and quaternion reflection contract.

### 2. PHP-style entry search

The existing `TorchMotionDatabase` remains the source of normalized motion
features. Candidate eligibility is a boolean row mask containing only entry
windows for skills compatible with the requested terrain interaction.

The query remains the standard 27-value motion-matching feature:

- future root positions and facing at the existing horizons;
- root planar velocity;
- left/right foot positions; and
- left/right foot velocities.

No dense terrain vector is appended to the nearest-neighbor feature. Terrain is
a hard skill-compatibility gate and a separately reported ranking term, matching
PHP's separation between locomotion matching and terrain-paired skills.

For offline scripted routes, the query includes the complete known command
trajectory across the candidate skill duration. This allows a coherent turning,
lateral, descending, or reversal skill to be selected before entering the
stairs. Live use will later replace that oracle command horizon with the normal
operator trajectory predictor.

### 3. Terrain compatibility and placement

For each candidate entry, a planar rigid transform aligns its entry root to the
current root. The source height grid is sampled at the transformed source-foot
and root-path locations, and the target heightmap is sampled at their world
locations.

A candidate is feasible only when:

- source and target support-height changes agree within 3 cm at every planned
  supported landing;
- no target landing patch exceeds 2.5 cm height range inside the existing sole
  edge footprint;
- every unsupported foot sample has at least -3 cm sole clearance from target
  terrain after accounting for the ankle-origin sole offset;
- entry foot positions differ by at most 8 cm;
- entry joint position and velocity stay inside the existing broad oracle
  bounds; and
- the whole transformed root/foot trajectory stays inside the target terrain
  domain.

Among feasible candidates, standard motion-feature cost is followed by full
route trajectory/facing error, landing error, and entry continuity. Every cost
component and rejected constraint is retained in the artifact.

### 4. Sequential skill execution

After selection, playback advances source frames sequentially through the
committed terminal frame. Motion matching is disabled inside the committed
skill. The source root trajectory receives only its selected planar placement;
there is no per-contact teleport or repeated root alignment.

Entry pose, joint velocity, root translation, and root yaw offsets are
inertialized with the existing critically damped spring implementation over a
bounded ten-frame window. Stance feet are not independently blended. The full
post-inertialization qpos is checked through MuJoCo FK, and a transition is
rejected before commit if it violates the terrain constraints.

Commands may change during playback, but the skill is interruptible only at an
authenticated stable double-support frame. An interrupt performs the same
entry search and whole-skill feasibility gate. If no replacement skill is
feasible, the current skill continues; the system never freezes a mid-swing
pose or inserts an unchecked fallback.

At a stable flat exit, ordinary flat motion matching resumes through the same
inertialization path.

### 5. Isolation

The composer is an experimental runtime beside the retained matcher and contact
oracle. It does not change existing matcher defaults or touch the current
uncommitted landing-bridge files. New skill-index, composer, rollout, CLI,
configuration, and tests use separate files and artifact roots.

## Evaluation

### Unit and integration tests

Tests must prove:

- deterministic episode and entry-window extraction;
- no skill boundary occurs during flight;
- mirrored skill FK consistency;
- entry-window-only eligibility;
- terrain mismatch rejection before feature ranking;
- standard matcher cost selects among compatible entries;
- sequential playback never changes clip or skips a source frame;
- ten-frame inertialization is continuous and decays to zero;
- command interruption occurs only at stable double support;
- no feasible replacement preserves the current skill transactionally; and
- saved artifacts authenticate arrays, selection diagnostics, and failures.

A real-data inventory test reports the number of usable stair skills and their
lateral, diagonal, ascent/descent, and turn coverage.

### Route matrix

Run the unchanged 21 same-stair routes and retain exact qpos for MuJoCo visual
review. The composer qualifies only when:

- at least 18 of 21 behavioral contracts pass;
- both cross-tread routes, all four side exits, both diagonal descents, the
  elevated reversal, and both 90-degree turn variants pass;
- every emitted foot is FK-consistent with saved qpos;
- no selected skill violates landing, edge, stance, or swing-clearance gates;
- the longest moving-command stall is at most five frames;
- aggregate stance slide is at least 50 percent below the retained layered
  baseline; and
- visual review shows no source-boundary pop, stair miss, or mid-command freeze.

Failure classification remains explicit:

- If coherent skills pass, this composer becomes the privileged kinematic
  teacher for later Sonic tracking and depth distillation.
- If source-paired skills pass but target-stair compatibility fails, add a
  bounded whole-skill contact warp rather than returning to per-contact search.
- If no complete skill covers a required route command, train a
  footstep/terrain-conditioned generator on these coherent skills and preserve
  this composer as its teacher and baseline.

## Deliverable

The milestone delivers an interactive MuJoCo kinematic viewer, deterministic
21-route artifacts, a comparison against the rigid oracle and retained layered
baseline, and a result report stating whether PHP-style skill composition
qualifies on the GRAIL stair corpus.
