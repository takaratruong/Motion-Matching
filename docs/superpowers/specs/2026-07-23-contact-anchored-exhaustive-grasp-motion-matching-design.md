# Contact-Anchored Exhaustive Grasp Motion Matching Design

**Date:** 2026-07-23

## Decision and Scope

Build the deterministic motion-matching coverage baseline around one rule:
recorded reaches are motion templates relative to their terminal hand contact,
not motions tied to their recorded root location, table, object, or world-space
endpoint.

For any physically reachable grasp pose, the search must instantiate every
captured and mirrored outbound reach at that grasp from multiple upright body
directions. Existing arm IK may shape the instantiated active arm, and complete
scene collision checks decide which instances are valid. Recorded endpoint
distance must never remove a template before placement.

This design supersedes the retrieval and whole-body alignment rules in:

- `2026-07-22-reusable-reach-warp-design.md`;
- `2026-07-22-world-grasp-upright-motion-search-design.md`;
- `2026-07-22-grasp-anchored-live-motion-search-design.md`; and
- `2026-07-22-exhaustive-motion-reuse-audit-design.md`.

Their corpus recovery, complete-clip, collision-diagnostic, explicit-search,
and standalone-viewer requirements remain applicable where they do not
conflict with this design.

The existing `interaction_ik` solver is a frozen dependency. This work does
not replace, redesign, or tune its algorithm. A separate experiment may compare
other IK methods later.

## Goal

The input is:

- a world-space grasp position;
- a world-space grasp orientation and approach axis;
- target-object geometry;
- furniture geometry; and
- the complete bilateral reach corpus.

The output is a diverse, deterministic set of complete recorded motions that:

1. place their terminal active wrist exactly at the requested grasp after
   active-arm shaping without moving the recorded root height;
2. keep the body upright;
3. approach from multiple body placements around the grasp;
4. satisfy the existing active-arm IK gates;
5. avoid target-object penetration except for terminal active-wrist contact;
6. avoid full-body furniture penetration; and
7. preserve enough provenance and diagnostics to explain every rejection.

This coverage search answers: "What recorded motions can perform this grasp if
the character is placed appropriately?" It deliberately does not answer the
later runtime question: "Which valid grasp motion best matches the character's
current walking pose, root, and velocity?" Runtime entry-state matching or a
diffusion stitch consumes this coverage set afterward.

## Coordinate Contract

Each canonical clip begins upright with its source root normalized. Let:

- `p_i` be any source world-space point at sample `i` in that canonical frame;
- `q_i` be its world-space orientation;
- `e` be the source terminal active-wrist position;
- `g` be the requested world-space grasp position; and
- `R_y(theta)` be an upright rotation around world up.

Define the horizontal contact translation:

```text
a = (g.x - (R_y(theta)e).x, 0, g.z - (R_y(theta)e).z)
p'_i = a + R_y(theta) * p_i
q'_i = R_y(theta) * q_i
```

The transform applies to the complete skeleton trajectory. It can be
implemented by composing the equivalent yaw and translation into the
Simulation root; local body motion remains recorded motion.

This contract guarantees before IK that the terminal active wrist is aligned
in the ground plane while recorded root height remains unchanged:

```text
terminal_active_wrist_position.xz = g.xz
placed_root_position.y = source_root_position.y
```

up to floating-point forward-kinematics precision. The remaining vertical
contact offset is introduced gradually through active-arm IK. This preserves
grounding and the recorded body-to-hand relationship while keeping every root,
torso, and limb upright. No object rotation is ever multiplied into the whole
skeleton.

Object translation changes `g`. Object rotation changes the requested grasp
orientation and approach axis, and may change `g` when the grasp is stored on
an object surface. It does not rotate the candidate population or its body-yaw
samples.

## Exhaustive Candidate Population

The v2 pack contains 192 captured left-hand reaches and 192 mirrored right-hand
reaches. One search includes both hands at the same requested grasp.

For every one of the 384 clips, instantiate 12 independent yaw placements:

```text
theta in {0, 30, 60, ..., 330} degrees
```

The raw population is therefore exactly 4,608 instances. The yaw index is a
search augmentation independent of object yaw. It creates body placements from
many directions while retaining every source trajectory and bilateral mirror.

There is no source-endpoint-distance eligibility envelope. There is no
recorded wrist-orientation eligibility gate. There is no source-approach
eligibility gate. Query translation and rotation cannot change the raw
population count.

Cheap broad-phase collision or reachability checks may reorder work or avoid
expensive processing only when they are logically exact and still record the
instance's truthful rejection. They may not silently reduce the population.

## Active-Arm Shaping

Grounded horizontal contact placement happens before active-arm shaping. IK is
responsible for moving the active wrist through the remaining vertical contact
offset and adapting it toward the requested wrist orientation and terminal
approach while retaining the trajectory's character.

Use the existing seven-joint `interaction::solve_hand_ik` implementation
unchanged. At the call site:

- preserve the placed pose as the initial state;
- smoothly introduce vertical contact translation before the terminal
  approach window, then approach and wrist-orientation correction over the
  existing outbound shaping interval;
- enforce the final position tolerance at every nonzero vertical-ramp sample,
  and reject a nonzero height retarget when the clip has no safe pre-contact
  ramp window;
- keep root, torso, legs, and inactive arm unchanged after rigid placement;
- require final position error at most `1 mm`;
- retain the existing `15 degree` approach-axis gate;
- retain the existing `60 degree` full-orientation gate; and
- retain existing joint-limit and finite-solver diagnostics.

The endpoint-relative approach warp must operate around `g` after placement.
At the terminal sample the endpoint-relative vector is zero, so rotating the
approach path cannot move the requested contact point. Any residual introduced
by orientation IK is measured by the final `1 mm` gate.

Keeping the current solver means no edits to `interaction_ik.h`,
`interaction_ik.cpp`, or the G1 arm metadata as part of this project. Solver
alternatives belong on another branch/session and must be compared against the
same contact-anchored candidate corpus.

## Collision Semantics

Collision checks consume each complete, placed, IK-shaped skeleton motion.

- Test all body joint spheres and parent-child capsules against every furniture
  box at every sample.
- Test the full body against the target object before terminal contact.
- During the existing terminal contact window, exempt only the active
  wrist-roll/wrist-pitch/wrist contact chain from the target object.
- Never exempt the elbow, elbow-to-wrist segment, torso, inactive arm, legs, or
  any body part from furniture.
- Record object and furniture collision observations even when an earlier
  kinematic gate is the exclusive rejection reason.

Open space should retain candidates from all body-yaw sectors. A table, shelf,
or lower table may remove sectors only through actual placed-body collision.
This is the intended measurement of data reuse around geometry.

## Ranking and Diversity

Acceptance is determined before ranking. An accepted instance must pass final
position, approach, orientation, object-collision, and environment-collision
gates.

Sort accepted instances deterministically by:

1. active-arm deformation from the placed recorded pose;
2. final orientation error;
3. final approach-axis error;
4. source clip index; and
5. yaw index.

Active-arm deformation is the mean squared wrapped joint-angle difference over
the seven active-arm joints and all shaped samples, measured against the
rigidly placed source pose. This makes the primary ranking quantity explicit,
independent of clip duration, and zero for an unchanged active arm.

Do not use original endpoint distance as cost because rigid placement removes
that quantity. Preserve hand, source recording, captured/mirrored augmentation,
and yaw index in every displayed option.

The search retains every accepted option for cycling. The viewer draws all
accepted hand paths in a lightweight representation when rendering performance
allows. If a measured rendering limit requires subsampling background paths,
the HUD must state `displayed / accepted`; the search result and cycle list must
remain exhaustive.

Rejected complete trajectories remain available for explicit diagnostic
display, with the first rejection and non-exclusive collision observations.

## Viewer Behavior

The standalone flat-skeleton viewer remains independent of controller,
diffusion, mesh, and terrain systems.

- Start in the table/shelf/lower-table coverage environment, not empty space.
- `G` toggles between coverage geometry and open space.
- Moving or rotating the object marks results stale but does not search while a
  key is held.
- `Enter` runs one exhaustive search for the current grasp.
- Bracket or slash controls cycle the complete accepted option list.
- A diagnostic toggle exposes rejected instances without confusing them with
  accepted choices.
- The selected complete outbound clip, capped at 3.6 seconds and ending at its
  recorded contact, animates as the blue G1 skeleton.
- The HUD reports raw instances, processed instances, accepted instances,
  rejection counts, both hand counts, source augmentation, yaw index, final
  errors, and observed collisions.

Exactly one viewer process may run. No screenshot, mesh renderer, terrain
renderer, or controller is launched by this work.

## Performance and Memory

The explicit search must finish within 30 seconds on the real 4,608-instance
population. Evaluation may use bounded worker threads over immutable pack and
scene data. Merge results by `(clip index, yaw index)` for deterministic output.

Do not retain full-skeleton poses for all 4,608 instances. During search,
retain compact hand paths, metrics, rejection data, and provenance. Rebuild the
full deterministic shaped pose sequence only for the selected option. The
selected regeneration must reproduce its stored metrics and status.

If the 30-second deadline is exceeded, report `INCOMPLETE`, processed versus
total instances, and retain the previous complete result set. Never publish a
partial result as exhaustive.

## Verification

Tests must fail under the old root-fixed implementation and establish all of
the following under the corrected implementation:

1. Translating a shared grasp in X/Z leaves the raw population at 4,608 and
   moves every placed terminal wrist by the same horizontal translation;
   changing grasp height never changes any placed root Y sample.
2. Rotating the object leaves the raw population at 4,608 and never tilts a
   skeleton root away from world up.
3. Each source clip produces exactly 12 yaw instances with stable identities.
4. Captured-left and mirrored-right options target the same grasp.
5. Grounded placement alone reproduces requested terminal X/Z within
   floating-point tolerance, and active-arm IK produces accepted full 3D
   contacts without changing root height.
6. Final IK-shaped accepted contacts remain within `1 mm`, while the current
   IK implementation files remain unchanged.
7. Open-space accepted options occupy multiple root-azimuth sectors around the
   same grasp.
8. Furniture removes candidates through full-body collision rather than a
   recorded endpoint or side filter.
9. Complete selected outbound clips, capped at 3.6 seconds and ending at
   contact, animate and agree with their highlighted hand paths
   sample-for-sample.
10. The real-pack search completes all 4,608 instances in less than 30 seconds
    with deterministic counts and bounded memory.
11. The viewer starts with coverage furniture visible and remains exactly one
    flat viewer process.

The real coverage report must use shared arbitrary grasp poses in open space,
on the table, on the shelf, below the table, and on the lower table. Testing
each clip only near its own stored endpoint is insufficient evidence and must
not be reported as generic grasp coverage.

## Non-Goals

- Replacing or redesigning IK.
- Matching the current locomotion entry state.
- Stitching walking to a selected reach.
- Diffusion generation.
- Object-specific motion labels.
- Grasp synthesis or finger articulation.
- Mesh or terrain rendering.
