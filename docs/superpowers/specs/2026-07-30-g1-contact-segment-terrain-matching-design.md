# G1 Contact-Segment Terrain Motion Matching Design

**Date:** 2026-07-30
**Branch:** `research/g1-torch-terrain-kinematics`
**Scope:** Kinematic motion matching on one authenticated GRAIL stair scene.
Tracking, physics, depth estimation, omnidirectional coverage, curb coverage,
and bulk-corpus publication are outside this project.

## Problem

The current Torch terrain matcher can select safe-looking root trajectories
while destroying the source motion's foot-to-terrain relationship.

Measured on
`terrain/grail/grail-stair_p1-d01be55953dba78b90e9/motion.npz`:

| Metric | Untouched source clip | Current matched rollout |
|---|---:|---:|
| Frames | 499 | 450 |
| Frames with no supported foot | 7.2% | 74.2% |
| Longest unsupported interval | 14 | 81 |
| Total stance slide | 0.313 m | 0.621 m |
| Source-sequence changes | 0 | 62 |

The selected raw source frames themselves retain plausible contact: at least one
foot is supported in 422/450 selected frames. In 312 frames, the selected source
has support while the composed output has none. More terrain data therefore
cannot solve the observed failure by itself.

The prior C++/Holden terrain program already explored height features,
candidate-conditioned foot scoring, same-foot gates, topology and sequence
search, landing reranking, support retargeting, planted-foot locking, terrain
normal IK, and selected-leg recovery. Most of that stack was never qualified
end-to-end visually. This experiment must not recreate that stack or use IK to
hide bad matched kinematics.

## Hypothesis

A terrain transition must preserve a coherent source contact segment rather
than select an independently placed frame every 20 ms.

Selecting one source support-to-support segment, aligning it through its support
foot, and retaining it through the contact phase should preserve the source
motion's valid stair interaction. Search may continue for diagnostics, but a
terrain transition may occur only at a permitted contact boundary.

## Experiment boundary

The experiment uses:

- the existing flat Takara locomotion clip;
- the one authenticated GRAIL stair clip named above;
- its exact published height grid and registration;
- the existing native-Z-up Torch matcher and MuJoCo kinematic viewer; and
- 50 Hz output.

It does not use:

- the 176-clip representative corpus or the full GRAIL corpus;
- inverse kinematics, foot locking, or physics;
- a learned model;
- future operator commands;
- a multi-step beam planner; or
- threshold weakening in the current penetration validator.

## Contact segmentation

For each terrain clip, derive left and right support from the same authenticated
conditions used by the existing metrics:

- ankle-origin clearance within `0.035 ± 0.020 m`; and
- absolute vertical ankle velocity at most `0.12 m/s`.

A contact onset is a false-to-true support transition for either foot. A
searchable terrain segment begins at an onset and ends immediately before the
next onset of the opposite foot. Double-support frames belong to the segment
that is already active. Segments shorter than 5 frames or longer than 30 frames
are invalid for this first experiment. Flat matching retains the existing
frame-level behavior until a terrain segment is entered.

Contact metadata is immutable, derived during database construction, and
indexed by clip and source frame. No contact inference is performed from the
already composed output when deciding a candidate's source phase.

## Transition and commitment

When the matcher proposes a terrain candidate:

1. Resolve the containing source contact segment.
2. Reject candidates that are not at the segment's entry boundary.
3. Require the candidate's entering support foot to be compatible with the
   current support phase. An airborne current state may enter either valid
   support side; a supported current state may not instantaneously switch to
   the opposite support foot.
4. Compose and validate the complete remaining segment against the query
   terrain.
5. If accepted, commit to sequential source frames through the segment. Search
   may still run and publish costs, but cannot replace the committed segment.
6. Release the commitment at the next contact onset. The next frame may retain
   the incumbent or choose another valid segment.

The commitment is bounded by the source segment and therefore lasts 0.10–0.60
seconds. It is not an arbitrary fixed action chunk.

## Support-foot placement

The candidate is placed in yaw and planar translation using the existing
commanded heading and root-trajectory logic. Its vertical translation is then
derived from the entering support foot:

1. Transform the source support ankle XY using the candidate yaw and planar
   placement.
2. Query the target terrain beneath that transformed ankle.
3. Query the source terrain beneath the recorded ankle.
4. Set one constant segment vertical offset equal to target support height minus
   source support height.
5. Apply that offset to both root and diagnostic body trajectories for the
   entire segment.

The segment may not independently translate individual feet or joints. This
preserves the recorded body geometry and makes incompatible two-foot/tread
geometry fail validation rather than deforming the motion.

At segment entry, joint and root-rotation inertialization remain enabled.
Independent inertialization of diagnostic foot positions is disabled for the
committed terrain segment: displayed foot positions must come from MuJoCo
forward kinematics of the emitted root and joint state. The validator and
metrics must consume those same FK positions.

## Validation

The existing minimum-clearance validator is necessary but insufficient. The
contact-segment validator evaluates every emitted frame in the proposed
segment and requires:

- no ankle clearance below `-0.03 m`;
- the entering source support foot reaches target support within
  `0.035 ± 0.020 m`;
- at least one foot is supported for all but at most 10 consecutive frames;
- source support does not disappear solely because of composition; and
- all height-grid samples stay inside the authenticated query domain.

Failures reject the transition transactionally and leave matcher state,
commitment state, and selection history unchanged.

## Diagnostics

Every output frame records:

- committed/uncommitted state;
- segment clip, start, end, and current source frame;
- entering support foot;
- vertical placement offset;
- source and emitted support masks;
- MuJoCo-FK foot clearance;
- transition accepted/rejected reason; and
- search time separately from committed playback time.

The saved rollout and live viewer use the same FK foot positions and terrain
alignment. Auxiliary independently inertialized body positions cannot satisfy
an acceptance gate.

## Acceptance

The first deterministic nine-second forward stair run passes only if all of
these hold:

| Gate | Threshold |
|---|---:|
| Frames with no supported foot | ≤ 15% |
| Longest unsupported interval | ≤ 10 frames |
| Total stance slide | ≤ 0.35 m |
| Minimum FK ankle clearance | ≥ -0.03 m |
| Source-supported frames losing all emitted support | ≤ 5% |
| Non-sequential changes inside a commitment | 0 |
| Reaches the authenticated upper landing | Yes |

The final gate is manual: the robot must visibly land on the stair treads
without missing steps, hovering through the ascent, or sliding continuously.
Automated gates cannot override a failed visual inspection.

If the experiment fails, diagnostics must classify the first cause as one of:

- no compatible segment retrieved;
- support-foot placement incompatible with the query tread;
- inertialized root/joint state broke source contact;
- commitment prevented necessary command response; or
- source clip itself failed its authenticated replay control.

No corpus expansion, weight sweep, IK, or planner work is authorized until the
failure is classified.

## Test strategy

Implementation is test-first:

1. Synthetic contact sequences freeze onset and segment-boundary semantics.
2. Transactional matcher tests freeze entry gating, bounded commitment,
   sequential playback, release, and rejected-transition state preservation.
3. Synthetic source/query height grids freeze support-foot vertical placement.
4. A real-data test verifies the authenticated stair clip's source contact
   statistics.
5. A guarded GPU/MuJoCo rollout computes the acceptance table from authoritative
   FK positions.
6. Only after automated acceptance passes is the live MuJoCo viewer launched.

## Follow-on work

After the single forward stair run passes visually:

1. qualify descent on the same clip and scene;
2. test diagonal approach and turning on the same stair;
3. add lateral and backward segments only where source data supports them;
4. add curb clips;
5. evaluate whether shorter safe commitments are needed for responsiveness.
