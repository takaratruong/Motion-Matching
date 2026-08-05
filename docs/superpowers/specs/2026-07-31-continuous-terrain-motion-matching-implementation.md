# Continuous Terrain Motion Matching: Implementation Record

## Goal

Generate clean, trackable, command-labelled terrain kinematics without treating
stairs as an uninterruptible video clip. The initial target is Justin's narrow
staircase; the same contracts later admit Karen and GRAIL geometry.

The runtime command remains two-stick: local travel velocity and independently
controlled body facing. Terrain matching uses only a causal, robot-centred
height observation plus the current robot state. Global root position, a global
map, and future user input are not inputs.

## Implemented vertical slice

`terrain_motion.py` now defines:

- the 48x32 robot-centred height/mask/confidence/normal/traversability contract;
- a deterministic stair fitter for the exact-Justin bootstrap;
- FLAT, STAIR_APPROACH, STAIR_COMMITTED, STAIR_EXIT, and SAFE_STOP modes;
- a clean per-frame terrain database contract;
- exact matching with separate kinematic, commanded-trajectory, facing, and
  environment costs;
- hard fixed-world stair-pose, support topology, tread identity, and planted
  foot gates;
- sequential continuation in flight or when no safe splice exists;
- transactional prepare/commit/rollback with 5 mm penetration and 10 mm
  planted-foot-drift execution gates.

`terrain_catalog.py` imports the 18 clean Justin MoBu clips (6,588 frames),
honours their declared `xyzw` quaternion convention, computes canonical stair
coordinates, derives foot contacts/tread identities, and emits a compressed
database plus provenance report. It stores future root/facing and foothold/
contact knots at +6/+12/+18/+24 frames. The two-stick channels are explicitly
labelled as inverse-fitted from the clean +24-frame motion rather than recorded
human input. Tracked/noised SONIC rollouts are explicitly
excluded from clean-pose ingestion.

## Artifacts and commands

Build:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.build_terrain_catalog \
  --output artifacts/terrain_catalog/justin_stairs_v1.npz
```

Audit sources:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.audit_terrain_sources \
  --output artifacts/terrain_catalog/source_audit.json
```

Plot paths and support phases:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.plot_terrain_catalog \
  artifacts/terrain_catalog/justin_stairs_v1.npz \
  artifacts/terrain_catalog/justin_stairs_diagnostics.png
```

Evaluate exact coverage:

```bash
PYTHONPATH=sonic/python python -m mm_sonic.evaluate_terrain_catalog \
  artifacts/terrain_catalog/justin_stairs_v1.npz \
  --output artifacts/terrain_catalog/justin_stairs_evaluation.json
```

## Why the terrain pose is separate from feature normalization

Ordinary motion features answer “which pose and velocity look continuous?”
The stair gate answers “would this pose put the feet on the same fixed physical
stair?” Normalizing those together permits a good pose match to compensate for
a bad environment match. This implementation instead rejects a candidate whose
recorded root-to-stair transform disagrees with the currently observed
robot-to-stair transform.

## Data expansion stages

1. Exact Justin: use direct clean clips and allow contact-safe switches only.
2. Karen/GRAIL import: resolve clean source identities and terrain pairings;
   never substitute the 489-clip tracked/noised collection for clean poses.
3. Geometry adaptation: contact-preserving optimization may adjust swing limbs,
   pelvis, and timing, while planted contacts are equality constraints. Reject
   penetration, sliding, unreachable IK, or large root/terrain displacement.
4. Tracker gate: replay every accepted kinematic clip with Justin's tuned SONIC
   checkpoint and retain both clean and noised recovery rollouts with the exact
   two-stick command stream.
5. Learned proposal: train a top-K candidate retriever only after the exact
   matcher is mechanically correct. Exact reranking and all safety gates remain;
   require at least 99.9% recall of the exact winner before timing at 50 Hz.

Sequential stair playback remains a safe fallback, not the desired final
controller. If the exact catalog reports poor cross-clip candidate coverage,
the next action is contact-preserving data expansion—not relaxing contact or
fixed-world safety thresholds.

## Robot-relative hybrid interactive extension

The browser hybrid now places the canonical Justin staircase on all four sides
of a finite raised platform.  The analytic scene is only a simulator sensor
backend: every control tick is rasterized to the same 48x32 robot-centred
height map consumed by matching.  Archive/world root coordinates are not used
to choose a stair motion.

The flat-to-terrain handoff is hierarchical:

1. The unmodified two-stick command preview establishes sustained intent and
   selects a stair corridor/direction family.
2. A data-supported approach shaper caps commands aimed into the stair corridor
   at 0.34 m/s while leaving unrelated flat commands unchanged.
3. Downhill approaches brake to a root-to-bottom-origin standoff equal to the
   observed stair length plus 8 cm, and use robot-relative stair centreline
   feedback.
4. Canonical root/stair transform, yaw, intended trajectory, current pose, and
   contacted feet rerank the remaining Justin approach frames.
5. If a candidate is airborne, it is legal only when the flat matcher's dense
   selected motion predicts the same future landing foot positions at the
   +6/+12/+18/+24 contact knots.
6. For staged downhill double support, rejected candidate footholds provide a
   bounded robot-frame planar correction.  This closes the last few centimetres
   without global localization or relaxing the final contact gate.
7. An upward or downward support discontinuity holds the last accepted flat
   pose unless a compatible stair splice commits.  The flat matcher can
   therefore neither walk through a riser nor continue above an unsupported
   landing.

Raw operator intent and the safely shaped trajectory are deliberately separate.
The intent latch therefore remains armed while a downhill stage brakes to a
standstill, while candidate trajectory scoring still sees only the motion that
the robot is actually being asked to execute.

### End-to-end clean-kinematic canary

A deterministic full-stick-forward canary traversed west flat ground, ascended
the west staircase, crossed the finite platform, descended the east staircase,
and returned to flat ground:

- uphill entry: `up_neg_30:156`, 18.34 mm support-foot mismatch;
- uphill landing: `up_neg_30:360`, 0.49 mm;
- downhill entry: `down_zero:30`, 29.90 mm;
- downhill landing: `down_zero:337`, 0.34 mm;
- unsafe flat support crossings: zero.

This validates the robot-centred clean-kinematic architecture, not physical
stability.  The two entry mismatches remain above the original 10 mm
tracker-quality target.  They must pass the tuned SONIC tracker/physics gate or
be reduced through contact-preserving data expansion before collection is
called robot-trackable.

LAFAN may later fill missing flat starts, stops, turns, and transitional gait
phases, but only after G1 retargeting, contact cleanup, and the same SONIC
trackability gate.  It is not a substitute for terrain-contact examples.

## 2026-08-01 c490/GRAIL verified expansion

The vertical slice now publishes:

```text
artifacts/terrain_catalog/justin_grail_c490_repaired_v9.zarr
artifacts/terrain_catalog/justin_grail_c490_repaired_v9_entry45.npz
```

The 29-clip baseline is followed by 28 complete-frame-audited c490 clips. The
new data includes five genuine descending GRAIL sources, mirrors, and separate
top-entry and bottom-landing crops. Descents are never created by reversing an
ascent. Ten proposed clips failed the 5 mm sole/zero forbidden-collision gate
and are absent from the published archive.

A zero-trim audit of the final merged archive then accepted all 28 appended
clips without further repair. Its maximum complete-frame sole penetration is
4.993 mm and its maximum forbidden penetration is zero.

The baseline's first 9,631 feature, source, pose, contact, tread, trajectory,
and inferred-command rows and fixed 60-dimensional feature scale are
bit-identical to v5. The only baseline metadata difference is 720 `entry`
booleans introduced by the deliberate 45-frame reviewed-entry rebuild;
v8-entry45 and v9 agree exactly. Intended contacts are recomputed from the
executable MuJoCo sole geometry with 12 mm clearance, 5.1 mm penetration, and
0.15 m/s source-foot speed thresholds. Contact relabelling does not change any
pose, velocity, trajectory, or feature row.

Two lifecycle corrections make the added data executable:

1. A complete c490 clip whose reviewed entry window starts at frame zero now
   contributes those entry rows; the matcher no longer subtracts an additional
   unavailable lookback.
2. During the stair-to-flat landing bridge, the dense root/support-edge sweep
   remains mandatory, but the nominal ankle-to-sole proxy is deferred to the
   immediately following exact articulated G1 repair. This removes a false
   permanent landing rejection without weakening collision or unsupported-edge
   checks. The browser and scripted evaluator call one shared mode predicate
   for this exception so their landing behavior cannot silently diverge.

The symmetric straight/±30/±60/±90 clean-kinematic matrix improves from 10/14
entries and 8/14 exits to 13/14 entries and 13/14 exits. Forbidden body
collisions remain zero; vertical riser-contact frames fall from 864 to 3. The
one miss is the +90-degree uphill approach, where no catalog entry matches the
stopped flat foot phase. That is a data/phase-warping target, not justification
for relaxing the handoff gate.

The browser viewer and headless evaluator now default to the v9 pair. Runtime
matching still receives only the current G1 state, ordinary two-stick command,
and 48x32 robot-centred causal height map. The scripted test operator's global
waypoint is evaluator-only and is converted to the same command before the
matcher.

Canonical evidence:

```text
artifacts/terrain_autoresearch/iteration97_v9_full14_canonical/aggregate.json
artifacts/terrain_autoresearch/iteration97_v9_full14_canonical/routes_intent_ball_actual.png
artifacts/terrain_autoresearch/iteration95_v9_rendered/down_neg30/waypoint_down_neg_30.mp4
artifacts/terrain_autoresearch/iteration95_v9_rendered/down_pos30/waypoint_down_pos_30.mp4
```

These are clean kinematics, not SONIC physics qualification. Dense foot review
finds mechanically coherent but conservative, high-kneed, pause-heavy
descents. The next accepted stage is phase-compatible entry expansion and
bounded contact-preserving timing/pose warping, followed by the tuned SONIC
tracker gate.
