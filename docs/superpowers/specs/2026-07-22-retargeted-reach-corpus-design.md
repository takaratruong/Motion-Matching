# Retargeted Reach Corpus and Coverage Lab Design

**Date:** 2026-07-22

## Goal

Convert the user-authored `g1_retargeted_motions.zip` recordings into a
reach-only G1 motion corpus, then measure how much of an arbitrary grasp pose
can be covered by reusing those motions with the existing arm IK. The first
pass prioritizes trustworthy data and visible diagnostics over replacing the
IK solver.

The recordings contain repeated neutral-to-grab-to-neutral actions but no
object or grasp labels. Every authored reach uses the left hand. The corpus
will retain each captured left-hand outbound reach and generate a validated
right-hand mirror so both sides can be evaluated without pretending the
mirrors are additional captures.

## Scope and Source Data

Use the recommended GMR pickle files from the trusted, local archive. Process
only the reach-oriented sources:

- `pickup_*`;
- `drawers`;
- `left_to_right_*`;
- `right_to_left_*`.

Keep `walking` and `carry_walking` in the source manifest but do not segment or
publish reaches from them. Preserve the original archive unchanged. Record its
path and checksum in generated metadata.

Each GMR source must contain finite `root_pos (N, 3)`, `root_rot (N, 4)` in
xyzw order, and `dof_pos (N, 29)` arrays. Use an embedded source frame rate
when available. If it is absent, require an explicit importer argument; use
100 Hz for this archive and record that override in provenance. Never infer a
frame rate silently. Convert through the existing G1 kinematics and resample
to the existing 25 Hz canonical skeleton representation. Reject malformed
inputs with a source-specific error rather than publishing a partial source
silently.

This archive is trusted user-authored input; loading arbitrary third-party
pickle files remains unsupported.

## Reach Definition and Annotation

A reach is the outbound portion of one action:

1. a neutral or settled hand pose;
2. departure from neutral;
3. continuous outward motion;
4. the grab frame at maximum useful extension or hand-velocity reversal.

The return to neutral is not included. A pause is not required because the
grab frame is defined by the outbound reversal, not by a fixed low-velocity
duration. Retain at most 3.6 seconds of context before the grab frame; shorter
authored approaches remain complete.

Use a hybrid extraction workflow. A detector proposes candidate grab frames
from root-relative left-wrist excursion and velocity reversal, groups nearby
peaks, and proposes the preceding neutral departure. It may use displacement,
return proximity, and left-hand dominance to suppress locomotion arm swings.
No proposal is published automatically.

A flat-skeleton annotation tool lets the operator scrub and play each source,
move between proposals, adjust the departure and grab frames, accept or reject
the candidate, and save progress. The active hand is left for this corpus.
Annotations live in a separate JSON file and include:

- archive and source provenance;
- source frame rate and source start/departure/grab indices;
- active hand;
- accepted or rejected status and optional confidence/note;
- root-relative grab-wrist position and rotation;
- final outbound approach direction.

The approach direction is the normalized wrist displacement over the 0.20
seconds ending at the grab frame, shortened only when the confirmed outbound
segment itself is shorter. Displacement below 1 cm is degenerate and is
reported for manual correction or rejection rather than replaced with an
arbitrary axis. Writes are atomic so an interrupted annotation session cannot
corrupt prior work.

## Reach-Only Artifact

Build a separate reach pack instead of forcing these recordings into the
existing object-interaction schema. It follows the repository's established
pack layout: `reach_database.bin` stores accepted outbound canonical poses and
per-reach metadata, `reach_features.bin` stores indexed endpoint retrieval
features, and `manifest.json` stores versioned schema, checksums, provenance,
and aggregate counts. Each reach contains timing, active hand, endpoint wrist
frame, approach direction, source provenance, and augmentation provenance.

It does not invent:

- object identity or object motion;
- contact, lift, or hold phases;
- a grasp-in-object transform;
- a lift phase from the authored return-to-neutral motion.

At query time, the movable scene object supplies the desired world grasp pose.
The source endpoint is a virtual wrist grasp frame used for retrieval and IK;
object and furniture collision are evaluated only after the motion is mapped
and shaped.

## Bilateral Mirroring

Annotate only the captured left-hand reaches. For every accepted source reach,
the builder emits:

1. the original captured left-hand reach; and
2. one synthetic right-hand mirror.

Mirror the entire pose, not just the arm. Reflection uses the G1 sagittal plane
(`y = 0` in the canonical G1 frame): root and joint positions negate `y`, left
and right bones are exchanged, and each world rotation is transformed as
`R_mirror = M R M`, where `M = diag(1, -1, 1)`. Reflect the root-relative wrist
endpoint and approach direction with the same `M`. Quaternions are normalized
after conversion from the reflected rotation matrix.

Mirroring must use a single explicit G1 left/right bone map and fail if a
required counterpart is absent. Center bones map to themselves. The generated
record stores `augmentation: mirrored`, the original reach ID, and right as
the active hand. Captured records store `augmentation: captured`. Do not mirror
a mirror or count a mirror as independently recorded evidence.

Validation checks every generated frame for finite transforms and proper
rotations, verifies left/right endpoint symmetry within `1e-5` m and `1e-4`
radians, and checks that mirroring twice reconstructs the original canonical
pose to those tolerances. A visual overlay of original and reflected samples
provides a corpus-level sanity check before coverage measurements.

## Coverage and IK Evaluation

Extend the flat G1 skeleton coverage viewer; do not use the mesh or terrain
renderer. Show all root-relative wrist trajectories and endpoint coverage for
front, back, lateral, ground, overhead, and intermediate reaches. Distinguish
captured left-hand paths from mirrored right-hand paths visually and in counts.

The user can position and rotate a target grasp, explicitly rerun the search,
and cycle the best reusable reaches. The viewer shows the desired and achieved
wrist frames, approach axes, complete outbound clip, and exact rejection
exclusive first-rejection counts:

1. outside the requested correction envelope;
2. invalid or non-finite solver result;
3. final position error;
4. final approach-axis error;
5. final wrist-twist or full-orientation error;
6. target-object collision by a disallowed body part;
7. body collision with furniture or the environment.

Also report joint-limit saturation as a non-exclusive diagnostic attached to
the final pose-error categories. The current solver clamps joints, so limit
saturation alone does not prove that a limit caused rejection. The active
hand's designated grasp volume is exempt from target-object collision at the
grab frame; the rest of the body and all environment geometry remain checked.

The viewer must not collapse those stages into one generic IK rejection. It
also reports captured-left and mirrored-right coverage separately, followed by
their union, so augmentation improves runtime coverage without overstating
capture diversity.

Use the current seven-joint, one-arm damped-least-squares IK for the initial
baseline. The root, legs, and waist remain fixed. Preserve the current hard
acceptance gates and expose their individual outcomes. Do not reject source
reaches because their recorded wrist rotation does not already match the
target: these recordings were authored for spatial reach coverage, and target
grasp orientation is imposed during shaping.

## Solver Sanity Experiments

Before interpreting coverage, run a zero-retarget test: set each requested
grasp to its original endpoint wrist pose. Every valid captured and mirrored
clip must finish within 1 mm position and 0.5 degrees of its stored approach
axis without collision requirements. Failures indicate conversion, mirroring,
or IK integration bugs, not missing data.

Then perturb target position by 5, 10, 20, 30, and 45 cm. Sweep rotations of
15, 30, 60, and 90 degrees in both directions about each local wrist axis,
plus a 180-degree approach-axis twist. Report acceptance and every rejection
stage by hand, source, height band, direction band, and augmentation type.
This separates corpus coverage from the limits of the current solver.

Only if those measurements show solver-limited coverage should a second pass
add hierarchical IK with waist plus seven arm joints, regularization toward the
recorded pose, position/approach-axis priority, and soft wrist twist. Full-body
IK, locomotion stitching, and diffusion are outside this first pass.

## Components and Data Flow

The implementation is divided into bounded units:

1. **Trusted GMR loader:** validates source arrays and metadata.
2. **Candidate detector:** proposes left-hand reach and neutral frames without
   making publication decisions.
3. **Annotation tool:** supports manual confirmation and writes annotation
   JSON atomically.
4. **Reach builder:** resamples accepted outbound windows and emits captured
   and mirrored reach records.
5. **Reach database:** loads and indexes the independent reach schema.
6. **Coverage evaluator:** retrieves, shapes, collision-checks, and records
   stage-specific outcomes.
7. **Flat coverage viewer:** visualizes corpus geometry, target/achieved frames,
   provenance, and diagnostics.

The data flow is:

`trusted GMR -> candidate proposals -> confirmed annotations -> 25 Hz outbound reaches -> bilateral augmentation -> reach database -> IK/collision coverage report`.

Each generated artifact records enough provenance to trace a displayed result
back to one source recording and one confirmed annotation.

## Verification and Operational Safety

Add focused tests for:

- GMR schema validation, 100-to-25 Hz resampling, and canonical FK conversion;
- synthetic neutral-reach-return candidate proposals;
- annotation adjustment, rejection, atomic persistence, and deterministic IDs;
- exclusion of `walking` and `carry_walking`;
- complete outbound clipping ending exactly at the confirmed grab frame;
- reflection geometry, left/right bone exchange, double-mirror identity, and
  mirrored endpoint/approach symmetry;
- reach artifact round trips and provenance;
- zero-retarget endpoint reproduction for captured and mirrored reaches;
- separate IK and collision rejection counters;
- deterministic perturbation coverage reports.

Run the established Python and C++ gates plus a bounded-memory viewer smoke
test. Launch at most one flat skeleton viewer. Do not launch the G1 mesh or
terrain renderer, take screenshots, or signal the controller process.

## Success Criteria

The pass is successful when all in-scope recordings can be reviewed without
editing the original archive, accepted outbound reaches produce traceable
left-hand and mirrored right-hand artifacts, zero-retarget tests pass, and the
viewer reports position/orientation/collision coverage without ambiguous
rejection categories. The resulting measurements must make it clear whether
the next limitation is data coverage or IK capacity.

## Out of Scope

- Automatic object labeling from the recordings.
- Treating the authored return as a lift or carry phase.
- Segmenting walking or walking-while-carrying sources.
- Training or running diffusion models.
- Locomotion-to-reach stitching.
- Replacing the current IK before baseline measurements justify it.
- Mesh rendering, terrain rendering, or physical simulation.
