# G1 Segmented Horizontal Grid Coverage

## Problem

The horizontal grid evaluator currently asks one raw motion window to cover an
entire lane: ground approach, mount, elevated interior, dismount, and ground
exit. A rejection in either boundary phase therefore labels the whole lane
infeasible, even when an ordinary walking motion covers the interior.

That is not the intended question. The desired traversal library is explicitly
compositional: place an existing mount, reuse or repeat an existing walking
segment along the object, then place an existing dismount.

## Evidence

The fixed staircase contains long, nearly level cross-stair interiors. For
example, the `y=+0.6 m` lane has an approximately `1.17 m` interior near
`0.35 m`, while `y=+0.8 m` has the same basic interior near `0.187 m`.
Interior walking should therefore be compared after removing the absolute
surface-height offset. The existing `3/11` result instead rejects candidates
against the complete height history and predominantly reports
`stance-height`.

## Approaches Considered

1. Keep whole-window retrieval and widen the shortlist. This cannot fix the
   abstraction error because a valid interior segment remains rejected when
   its unrelated boundary phases fail.
2. Certify the three phases independently. This isolates corpus coverage from
   transition synthesis and directly tests whether a gait can be reused on
   equivalent plank interiors.
3. Immediately synthesize full transitions. This is ultimately useful, but it
   mixes retrieval failure with splice quality and makes the coverage result
   hard to interpret.

The implementation will use approach 2 first. Accepted phase segments will be
the inputs to later compositing.

## Design

### Terrain phase segmentation

For each `+X` lane, sample the terrain at the nominal left-foot, root, and
right-foot lateral tracks. Partition the path into:

- `mount`: the boundary interval from level ground through the first stable
  elevated support region;
- `interior`: the maximal stable elevated interval;
- `dismount`: the boundary interval from that stable region back to ground.

A stable interior requires the per-foot surface-height pattern to remain
constant within the height-grid interpolation tolerance. The phase boundaries
must be derived from terrain samples, not fixed `x` coordinates.

### Height-normalized interior signature

Interior retrieval compares:

- alternating foot order;
- forward and lateral footprint placement;
- contact timing;
- left-right support-height difference.

It deliberately removes a common vertical offset. Thus the same walking
window may cover otherwise equivalent planks at different absolute heights.
Rigid placement may translate the complete motion vertically, but may not edit
joint positions or independently move either foot.

### Boundary signatures

Mount and dismount keep their absolute relative height changes. A boundary is
allowed to be infeasible when its vertical face is not represented by a
certified raw motion. This must not erase independently certified interior
coverage.

### Reporting

Each lane reports independent `mount`, `interior`, and `dismount` states with:

- phase interval and terrain signature;
- selected source clip and frame range, if any;
- certification metrics and rejection counts;
- whether the selected interior source is reused by other lanes.

The lane-level classification is:

- `full`: all required phases are certified;
- `partial`: at least one phase is certified;
- `infeasible`: no phase is certified.

The report also includes phase-level aggregate coverage so that a reusable
interior gait is visible even when a tall side mount is physically
infeasible.

## Validation

Tests will prove that:

1. a synthetic ground/elevated/ground profile produces the three expected
   phases;
2. two interiors with identical left-right splits but different common
   heights have identical normalized signatures;
3. a boundary failure does not discard an accepted interior;
4. complete phase coverage still yields `full`;
5. the fixed 20 cm grid produces deterministic phase reports.

The existing rigid-placement contact, heading, path-tube, and sole-clearance
gates remain unchanged.
