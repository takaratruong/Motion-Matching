# G1 Quick Horizontal Phase Composite

## Goal

Build and visualize one continuous `y=+0.8 m` horizontal traversal from the
already-certified mount, interior, and dismount artifacts. This is a splice
test, not another corpus search.

## Inputs

- Full mount: `lane-pos-0p8/mount/traversal.npz`
- Partial interior: `lane-pos-0p8/interior/traversal.npz`
- Full dismount: `lane-pos-0p8/dismount/traversal.npz`
- Existing phase summary, target terrain, G1 model, and source support masks

The phase intervals overlap by approximately `0.4 m`. The interior finishes
approximately `0.106 m` before the dismount begins.

## Approaches

1. **Spatial trim plus smooth boundary blend (selected).** Translate the
   complete interior rigidly along the path until its terminal root meets the
   dismount entrance, select the lowest pose/velocity mismatch in the
   mount/interior overlap, and apply short smooth blends at both joins. This is
   the quickest useful baseline and does not edit the source motions outside
   the blend horizons.
2. **Stance-foot-locked IK connectors.** Better explicit contact control, but
   slower and likely to reintroduce the ankle artifacts excluded from the
   current baseline.
3. **Boundary-compatible motion re-search.** Most principled if the quick
   blend fails, but it is a new retrieval experiment rather than a quick
   composition test.

## Composition

1. Rigidly shift the interior root trajectory so its final root position
   coincides with the first dismount root position. Joint positions and root
   orientations remain unchanged.
2. Search the physical mount/interior overlap for a frame pair with:
   - root distance at most `0.12 m`;
   - minimum joint-position and finite-difference velocity mismatch;
   - forward temporal ordering.
3. Trim the mount and interior at that pair.
4. Blend the first `10` frames of each outgoing segment from the exact
   preceding pose with cubic smoothstep interpolation and shortest-arc
   quaternion interpolation.
5. Concatenate mount, shifted interior, and dismount into one connector.

## Acceptance

The composite is accepted only if:

- every quaternion remains normalized;
- maximum per-frame joint and root steps are reported;
- minimum sole clearance is at least `-0.03 m`;
- stance-contact error is at most `0.03 m`;
- there are no unsupported frames;
- the final artifact opens in the existing 50 Hz MuJoCo viewer.

If the blend fails these gates, the script emits metrics but does not label
the composite accepted; the next attempt must use a supported-foot connector
rather than silently relaxing thresholds.
