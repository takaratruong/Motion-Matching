# Tiered Grasp Coverage Lab Design

**Date:** 2026-07-22

## Goal

Turn the existing hand-trajectory viewer into a reliable data-coverage lab for
generic world-space grasps. The lab must expose whether the accepted GRAIL
table and ground pickup corpus can supply usable complete pickup motions around
a center table, a compact shelf on its right side, and a lower table on its
left side.

This pass does not rebuild or relax acceptance of the 2,510-clip interaction
pack. It improves how the existing clips are queried, filtered, classified,
and visualized.

## Retrieval Semantics

Search runs only when the user presses Enter. Object translation or rotation
marks the displayed result stale but does not recompute it.

Retrieval has two ordered tiers:

1. **Exact pose:** use the existing position threshold and full quaternion
   orientation threshold. These matches always sort before fallback matches.
2. **Axis-constrained fallback:** if fewer than 12 exact trajectories survive
   IK and collision filtering, search additional unique clips that match grasp
   position and the grasp palm/approach axis while ignoring rotation about that
   axis. Add fallback trajectories until 12 valid trajectories are available
   or the corpus is exhausted.

The fallback is not position-only matching. It preserves the direction from
which the hand must meet the object and relaxes only wrist twist around that
direction. The desired world axis is the canonical clip's stored
`approach_direction_object` rotated by the movable object. Each candidate axis
is its stored object-space approach direction rotated by the mapped object.
Fallback orientation error is the angle between these two unit vectors, with
the same 25-degree threshold used by exact orientation search. For shaping,
construct the closest target wrist rotation that aligns the candidate axis to
the desired axis while preserving the candidate's twist about that axis. Final
IK acceptance measures axis error instead of full quaternion error. A
candidate therefore cannot be retrieved under one orientation definition and
rejected under an incompatible definition.

The HUD reports exact-compatible, fallback-compatible, exact-valid,
fallback-valid, IK-rejected, wrong-side, object-collision, and
environment-collision counts. Each cycled option is labeled `EXACT` or
`AXIS-FALLBACK`.

## Same-Side Approach Rule

The scene front is the negative-Z side, where the initial viewer camera and
canonical robot approach are located. For every mapped trajectory, normalize
the horizontal vector from the target grasp to the root position at clip start
and dot it with the scene-front vector `(0, 0, -1)`. Reject the trajectory as
`WrongSide` when the dot product is negative. A zero dot product is accepted,
so side approaches remain available while starts in the rear half-plane do not.

Use a configurable front-direction dot-product threshold of zero for the first
implementation. The test is applied after world mapping and before expensive
full-clip IK shaping. It therefore removes motions that originate behind the
table without discarding motions merely because their source recording used a
different world orientation.

## Coverage Environment

Replace the fixed five-box `ShelfGeometry` collision container with a general
`EnvironmentGeometry` collection of oriented boxes. The same collection is
used for drawing and collision testing so visible furniture and tested
furniture cannot diverge.

The environment contains:

- The existing recorded center table, including its top and four legs.
- A compact open shelf over the right third of the center table: one shelf
  board 0.32 m above the center tabletop, 0.45 m wide, 0.32 m deep, with two
  slim vertical supports. Its dimensions and placement are relative to the
  center tabletop and are clamped to fit it.
- A lower table to the left of the center table: 0.65 m wide, 0.50 m deep,
  with its top 0.24 m below the center tabletop and four legs extending to the
  ground.

The left table is separated from the center table by 0.12 m. The right shelf
does not extend beyond the center tabletop. Furniture is axis-aligned in the
scene for this diagnostic viewer. The movable object and grasp can be placed
on, under, or around any of these surfaces before Enter reruns the search.

Ground-only packs continue to work: they render the virtual floor and omit the
table-derived furniture. Mixed and table packs render the full coverage
environment.

## Candidate Processing

One Enter press performs this pipeline:

1. Build the world grasp from object transform and grasp-in-object transform.
2. Retrieve exact candidates.
3. Map candidates to the world grasp and reject wrong-side starts.
4. Shape with tier-appropriate IK and reject failed contacts.
5. Reject object and environment collisions.
6. If fewer than 12 exact candidates remain, retrieve and process unique
   axis-fallback candidates.
7. Sort exact candidates by existing cost, then fallback candidates by
   position and axis error; preserve clip-ID tie breaking.
8. Display all valid paths and animate the selected complete clip from its
   recorded start through lift.

No raw or rejected corpus recordings are added. The existing 4,096 compatible
clip safety limit remains above the 2,510-clip corpus size.

## Failure Behavior

Zero valid results is a valid diagnostic outcome. The HUD must distinguish
whether coverage was lost at retrieval, same-side filtering, IK, object
collision, or furniture collision. Invalid furniture dimensions and invalid
axis definitions fail fast with a descriptive exception rather than silently
disabling collision checks.

## Verification

Add focused C++ tests for:

- exact matches sorting before fallback matches;
- twist-invariant axis error and rejection of a wrong grasp axis;
- fallback activating below 12 exact valid motions and deduplicating clips;
- mapped front-side acceptance and back-side rejection;
- variable-size environment collision over all furniture boxes;
- deterministic construction and dimensions of the center table, right shelf,
  and lower left table;
- continued ground-only operation without table furniture.

Update viewer source-contract tests for the new HUD and geometry API. Run the
full Python suite, compact database loader tests, hand-trajectory C++ tests,
and a bounded-memory viewer smoke test. Start only one skeleton viewer process;
do not launch the mesh renderer or terrain viewer.

## Out of Scope

- Recovering any of the 2,014 rejected source recordings.
- Training or running a diffusion model.
- Replacing the skeleton viewer with the G1 mesh renderer.
- Continuous search while the object is moving.
- Runtime locomotion-to-pickup stitching.
