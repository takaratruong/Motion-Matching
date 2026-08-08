# G1 Object-Local Rasterized Motion Field Design

## Goal

Precompute a reusable, terrain-aware motion lookup for one object. The field
contains constant-heading traversal lines at several angles and sparse,
validated transitions between lines near their geometric intersections.

## Representation

The field is expressed in an object-local scene frame and is independent of
the object's global translation or rotation. A heading family contains
parallel lines at a runtime spacing, initially 0.20 m. Each line records its
terrain profile, mount/traversal/dismount contract, motion artifact when one is
available, contact events, cadence, source provenance, and admission status.

The first heading set is `0, +45, -45, and 90` degrees together with their
reverse directions. All line geometry comes from the elevated terrain
footprint. Angle-specific source frames, offsets, or terrain heights are not
allowed.

## Offline Optimization

Each line is solved independently using hierarchical retrieval:

1. prefer a single rigidly placed GRAIL window;
2. otherwise compose mount, terrain traversal, and dismount source windows;
3. repeat a compatible level gait cycle for wide surfaces;
4. use an inbetweener only for a short, contact-compatible boundary.

A line with no natural solution remains an explicit missing cell. The field
never fills it with flat-only motion, unsupported pose interpolation, or an
uncertified generated segment.

## Intersections and Transitions

Different heading lines form transition opportunities where their scene-space
segments intersect within the traversable terrain footprint. An opportunity
is retained only when both routes have admitted motion within a configurable
distance of the intersection.

For every opportunity, search nearby route frames and GRAIL turn windows using
root position, incoming/outgoing heading, support foot, contact phase, terrain
height, pose, and velocity. Prefer a continuous source window. Otherwise use a
short connector between phase-compatible boundaries. A transition edge stores
its entry and exit frames, transition motion, costs, and independent validation
report. Geometric intersection alone never creates a playable edge.

## Runtime Lookup

The runtime index is a directed graph. Continuation edges advance along one
line; transition edges change heading near an intersection. A query uses object-
local position, desired heading, current support phase, and route progress to
return the nearest admitted continuation or transition. Missing cells and
failed edges remain visible to the caller rather than silently degrading.

## Acceptance

- Every playable line crosses the object and contains mount, traversal, and
  dismount evidence.
- Every playable transition changes between its declared headings near the
  declared intersection.
- Contacts alternate naturally; planned flight is explicit and separately
  certified.
- Same-height cadence, stance locking, sole clearance, terminal support,
  joint/root continuity, and source provenance pass independent checks.
- Every admitted line and edge receives a dense visual audit before entering
  the runtime index.
- Object translation and yaw rotation preserve the same local field topology.

## Delivery Order

First publish the multi-heading geometric field and ingest the retained
horizontal and source-preserving diagonal baselines. Next solve missing lines
without weakening admission. Finally generate and validate transition edges at
intersections, beginning with horizontal-to-diagonal changes on the lower
reachable treads.
