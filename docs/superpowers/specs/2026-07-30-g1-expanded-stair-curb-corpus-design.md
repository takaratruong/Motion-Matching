# G1 Expanded Stair and Curb Corpus Design

Date: 2026-07-30

Status: Approved

Branch: `research/g1-torch-terrain-kinematics`

## Purpose

Build a second, comprehensive native 50 Hz terrain-motion corpus from every
locally available G1 stair or curb recording whose motion layout, terrain
geometry, and coordinate alignment can be verified. The existing
`build/torch-stair-small` five-clip dataset remains unchanged as the qualified
control.

This is a data-coverage experiment. It must not add rescue hysteresis,
anti-cycle state, a new matcher score, or another algorithmic fix in the same
change. The first question is whether relevant lateral, descent, landing-exit,
and curb kinematics eliminate the observed terrain-rescue limit cycle.

## Diagnosed Coverage Failure

The small corpus contains one 34,863-frame flat Takara clip and only four
499-frame recordings of one stair prop. During interactive lateral control, the
matcher entered a repeatable local limit cycle while the `D` command continued
to reach the matcher:

- rescue to `stair/0002`, frame 321;
- rescue to `stair/0000`, frame 336;
- return to the same two source neighborhoods every six to seven frames; and
- little net root progress because the alternating aligned transforms largely
  cancelled.

Changing the command perturbed the query and escaped the cycle. This proves the
failure is not an input freeze, rendering freeze, or GPU-latency stall.

The host already contains additional terrain motions, including an 816-frame
50 Hz `staircase_side_stepto` recording aligned to documented three-step
geometry. It traverses the stair laterally, which directly covers the command
that exposed the cycle. Separate continuous ascent, continuous descent, walk-up,
walk-down, staircase, and chair-step recordings are also present.

## Source Registry

Expanded publication uses an explicit checked source registry. Runtime
publication never discovers arbitrary files by glob. The initial candidate
registry contains these semantic families:

1. the four already-qualified GRAIL stair recordings;
2. local `staircase`, `staircase_final`, and `staircase_side_stepto`
   recordings;
3. `up_continuous`, `down_continuous`, `walk_up`, and `walk_down` recordings,
   including the Karen-stair variants when their matching metadata is present;
4. chair-step and chair-step-climbing recordings treated as curb candidates;
   and
5. later curb recordings added to the same explicit registry.

Crane and unidentified generic motions are outside the registry even if their
files share the same NPZ shape.

Every registry entry records:

- a stable logical clip name and semantic family;
- the exact motion path and SHA-256;
- source FPS and frame count;
- declared G1 joint/body layout;
- the terrain adapter and its geometry inputs;
- all geometry/metadata hashes;
- the coordinate transform from terrain space to motion world; and
- whether the entry was admitted or rejected, with a structured reason.

Rejected entries remain in the manifest evidence. They are not silently
dropped, and a rejected clip is never published as flat terrain.

## Admission Contract

A candidate enters the expanded corpus only after all of the following pass:

1. **Motion contract:** finite G1 arrays with 29 joints, 30 bodies, unit wxyz
   body quaternions, and a declared positive sampling rate.
2. **Canonical rate:** the clip is already 50 Hz or is resampled once to 50 Hz
   using the existing timestamp and shortest-arc quaternion routines.
3. **Joint and body identity:** source joint names, provenance, or a pinned
   adapter prove the existing `g1-29dof-isaaclab-v1` order. Array shape alone
   is insufficient.
4. **Terrain authority:** a USD mesh plus object pose, documented fixed
   MuJoCo/OBJ staircase, or recorded object geometry provides the actual source
   surface. No height grid is inferred only from the motion.
5. **Alignment oracle:** recorded ankle-roll positions agree with the source
   surface on contact-like frames. The check uses the same 3.5 cm ankle-origin
   allowance as the qualified small corpus and must include at least 50
   contact-like samples across both feet.
6. **Native FK consistency:** the published joint/root arrays reproduce the
   published body positions through the pinned G1 MuJoCo model within the
   existing conversion tolerance.
7. **Usable duration:** at least 46 output frames remain, because the matcher
   emits a 46-frame dense window.
8. **Duplicate control:** exact duplicate motion hashes are published once.
   Near-duplicate takes remain separate only when their terrain, direction, or
   contact trajectory is materially different.

The alignment oracle is a fail-closed admission gate. A promising motion with
missing or ambiguous terrain metadata is reported as rejected until its
alignment can be authenticated.

## Terrain Adapters

Terrain construction is explicit by source family:

- **GRAIL:** retain the qualified USD mesh, object pose, scale, and Z-up
  rasterizer.
- **Local staircase:** construct the documented fixed three-step surface from
  its checked MuJoCo/OBJ geometry. The source coordinates are X depth, Y
  lateral, and Z up; the step dimensions and transforms come from the checked
  environment files.
- **Karen stairs:** use the checked staircase metadata and matching XML/OBJ
  assets. A motion is rejected if the metadata does not identify the exact
  geometry variant.
- **Chair step / curb:** use recorded object pose and checked object geometry
  when present. A motion-only archive without authoritative step geometry is
  rejected rather than assigned an estimated curb.

Each admitted clip gets its own `terrain.npz`. Database features are generated
from that source terrain; live query features continue to come from the active
scene terrain. This preserves the existing terrain-feature abstraction and
allows clips recorded on different but geometrically compatible obstacles to
compete in one exact search.

## Generated Dataset

The publisher writes an ignored transactional tree:

```text
build/torch-terrain-expanded/
  manifest.json
  flat/takara/motion.npz
  terrain/<family>/<clip>/motion.npz
  terrain/<family>/<clip>/terrain.npz
```

The manifest uses a new schema version and contains both `accepted_clips` and
`rejected_candidates`. It pins output hashes, input hashes, conversion
parameters, terrain-grid shape and bounds, adapter identity, and admission
metrics.

Publication stages the full tree, validates it through `MotionFolder.load`,
recomputes every output hash, runs all admission checks, fsyncs, and atomically
replaces the prior expanded dataset. Failure preserves the last valid output.
The small dataset is never replaced or modified.

## Same-Stair Omnidirectional Stress Suite

The primary stress test uses one fixed authenticated staircase throughout. It
does not swap obstacles between scenarios. Deterministic command paths place
the robot at known approach locations and exercise terrain interactions that a
normal flat locomotion benchmark does not cover:

1. **Side mount, both directions:** approach the staircase perpendicular to its
   main ascent axis from the left and right. The kinematics must step up onto
   the obstacle rather than translate through the side or remain at floor
   height.
2. **Cross-tread traversal:** walk horizontally across the staircase while the
   feet occupy different tread heights. Each support foot must agree with its
   own local terrain height; the metric must not assume both feet share one
   ground plane.
3. **Turns on the obstacle:** execute 45, 90, and 180 degree turns on the lower,
   middle, and upper treads and on the landing. Include both turn directions.
4. **Diagonal ascent and descent:** traverse both diagonals in both directions,
   including paths that enter on one side and leave through the opposite side.
5. **Side exits and drop-offs:** ascend normally, then command a left or right
   exit from multiple tread heights. The kinematics must lower onto the adjacent
   floor without penetrating, hovering, or continuing on an imaginary tread.
6. **On-obstacle reversal and stop/restart:** reverse, stop, and restart while
   straddling a riser or occupying a tread, not only on the approach floor.
7. **Mixed adversarial route:** combine side entry, diagonal motion, a turn,
   cross-tread traversal, and a side exit in one uninterrupted episode so local
   success cannot hide a later state-dependent rescue cycle.

Every route is specified in staircase-local coordinates and transformed once
into matcher world. This makes left/right and rotated variants exact and keeps
the commands independent of the source motion clips selected by the matcher.

### Terrain-Interaction Metrics

The suite records the existing selection, continuity, jerk, latency, and
clearance diagnostics plus:

- command-aligned and cross-command root progress;
- root-height response relative to the terrain under the support polygon;
- per-foot terrain height, clearance, and contact state;
- stance-foot tangential speed and accumulated tangential displacement;
- left/right support-height difference while straddling different treads;
- heading error and time to settle after each turn;
- side-entry mount success and side-exit landing success;
- below-surface penetration duration and integral;
- repeated rescue-source neighborhoods and net progress across them; and
- completion of every route without an exception.

A foot is considered stance-like only when its ankle-origin clearance is within
the checked sole allowance and its vertical speed is below a frozen threshold.
Foot sliding is then its horizontal displacement relative to the fixed
staircase during each contiguous stance interval. Swing-foot motion is not
counted as sliding.

Cross-tread traversal is evaluated per foot against the local surface. A valid
unequal-height gait therefore receives credit when one support foot is on a
higher tread and both feet remain near their respective surfaces; it is not
penalized for violating a fictitious common ground height.

## Evaluation

The checked-in matcher configuration remains unchanged for the first
data-coverage comparison. The small and expanded datasets run with identical
commands, query terrain, feature weights, search cadence, continuity weights,
terrain validator, and joint-reference smoothing.

Evaluation has five layers:

1. the existing flat/legacy/dense nine-second acceptance rollout;
2. the directional up-turn-down benchmark;
3. the six-scenario hard-command matrix; and
4. the same-stair omnidirectional stress suite; and
5. a deterministic live-style lateral rescue-cycle replay recorded from the
   interactive failure.

The expansion is retained only if:

- every existing dense terrain acceptance gate remains true;
- every existing hard-command non-regression and clearance gate remains true;
- actual MuJoCo-FK ankle-origin clearance remains at least `-0.03 m`;
- the expanded matcher completes without an out-of-domain or no-safe-candidate
  exception; and
- every same-stair route completes its required mount, traversal, turn, or exit
  displacement without losing terrain-relative support;
- stance-foot sliding does not regress relative to the small-corpus baseline
  and no individual stance interval exhibits an unbounded slide; and
- the lateral replay contains no repeated `A -> B -> A -> B` terrain-rescue
  pattern in the same eight-frame source neighborhoods with less than 5 cm net
  root displacement across the four rescues.

The evaluation reports selected family/clip/frame, terrain-rescue events,
coverage by command direction, root progress, joint jerk, clearance, foot
sliding, support-height error, turn response, and latency. Latency remains
informational in this kinematic phase, but database row count and p50/p95 search
time are recorded.

After automated qualification, the expanded dataset is launched in the
kinematic MuJoCo viewer for adversarial WASD testing. Visual judgment supplements
the frozen gates and does not replace them.

## Implementation Boundaries

The existing strict small-corpus builder remains intact. Expanded behavior is
added through:

- a checked terrain-motion source registry;
- family-specific motion and terrain adapters;
- a common admission validator;
- an expanded transactional publisher; and
- an expanded-corpus evaluation command; and
- a staircase-local route generator plus a renderer-independent
  terrain-interaction evaluator.

Shared resampling, quaternion, height-grid, deterministic-NPZ, hashing, and
atomic-publication code should be reused rather than copied. Refactoring is
limited to extracting those already-qualified shared operations when needed by
both publishers.

## Testing

Test-driven implementation covers:

- exact registry identity and rejection of unregistered sources;
- 33/25/50 Hz endpoint-preserving conversion to 50 Hz;
- joint/body layout validation;
- documented staircase geometry and coordinate orientation;
- chair-step object-pose transforms;
- contact-height alignment admission and rejection;
- exact and near-duplicate handling;
- accepted/rejected manifest evidence;
- transactional replacement and failure preservation;
- strict `MotionFolder` loading of the expanded output;
- deterministic dataset identity; and
- detection of the observed rescue-cycle signature;
- exact rotated/mirrored route generation on one fixed staircase;
- per-foot support against different simultaneous tread heights; and
- stance-only foot-sliding measurement that excludes swing intervals.

Protected local qualification builds the real expanded corpus and saves its
complete admission report. Large generated NPZ files remain ignored.

## Failure Handling

Missing files, changed hashes, unsupported rates, ambiguous joint order,
missing terrain, failed contact alignment, invalid FK, non-finite arrays, and
duplicate logical names are explicit candidate rejections. Publisher defects,
manifest inconsistencies, or failure of any accepted output abort the complete
transaction.

The system does not weaken terrain validation to admit more data.

## Non-Goals

- Changing matcher selection, rescue, or hysteresis logic before the expanded
  data-only baseline and same-stair suite are frozen.
- Inferring terrain from depth or proprioception.
- SONIC tracking or physics integration.
- Training a learned model.
- Treating unidentified objects as curbs based only on filenames.
- Claiming that additional data eliminates the need for later safety
  guardrails.

## Follow-On Decision

If validated coverage removes the lateral rescue cycle and improves the hard
matrix, continue adding authenticated curb geometries and retain the expanded
corpus. The frozen same-stair suite then becomes the optimization objective for
matcher-quality work. Each later algorithm hypothesis changes one causal
mechanism at a time, adds a focused failing test, and is retained only when it
improves the aggregate terrain-interaction metrics without regressing any
qualified route or flat locomotion.

Likely follow-on mechanisms include contact-aware transition scoring,
progress-aware terrain rescue, and short emitted-window lookahead. They are not
mixed into the data expansion, because the data-only result must establish
whether the remaining defect is algorithmic.

If the cycle persists despite relevant lateral clips being selected, the
evidence justifies a separate, independently tested rescue anti-cycle design.
If useful clips are rejected only because terrain metadata is absent, recover
or recreate that metadata as a separate curation task rather than weakening
admission.
