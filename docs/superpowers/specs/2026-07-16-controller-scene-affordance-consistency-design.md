# Controller Scene Affordance Consistency

Date: 2026-07-16
Status: approved design checkpoint

## Evidence and problem

The bounded evidence and causal trace are recorded in
`.superpowers/sdd/preview-certification-root-cause-report.md`. The graphical
controller reached attached Carry, but its first placement preview was rejected:

```text
accepted=0 reason=CorrectionLimit source=0 selection=0 ik_fingerprint=0
ik_exact=1 candidate_ik_exact=1
```

The controller uses `make_controller_demo_target` and
`make_controller_demo_destination_surface` unchanged. The passing headless
probe instead rewrites both the grasp and destination from the contact-minus-one
rest object. Controlled variants proved that either difference is causal:

- exact target plus exact destination: `CorrectionLimit`;
- exact target plus adjusted destination: `PlacementOutOfBounds`;
- adjusted target plus exact destination: `CorrectionLimit`; and
- adjusted target plus adjusted destination: accepted `ReversedPickup`.

The false certification condition is `!preview.accepted`; zero candidate IDs
and fingerprint are consequences of that rejection. IK identity is intact.

## Decision

Keep the existing public constructors, but make them share one internal,
deterministic contact/rest derivation:

1. Find the certified first Contact in clip 0 and require a valid preceding
   rest frame in the same clip.
2. Read the active hand, contact hand transform, contact-minus-one object
   transform, source table transform/size, and object bounds once under the
   same validation rules.
3. Derive the target grasp as
   `inverse(rest_object) * contact_hand`; do not use inconsistent clip-level
   grasp metadata for this controller demo scene.
4. Derive the destination `object_in_surface` and `support_point_object` from
   that same rest object and source-table top plane.
5. Run `evaluate_placement_fit` inside the shared destination constructor. If
   the only failure is the already-reviewed finite vertical bounds clearance,
   apply that exact normal-axis correction and re-evaluate. Any footprint,
   overhead, malformed-geometry, or remaining fit failure throws `FormatError`.

The helper carrying the certified source data remains private to
`interaction_controller_adapter.cpp`; no new public selector or caller-supplied
geometry is introduced.

## Consumers and diagnostics

The graphical controller continues calling the two public constructors. The
headless probe removes its private target/destination rewrites and consumes the
same constructors unchanged. Its real runtime path must then certify a nonzero
`ReversedPickup` preview on the current pack.

The controller's generic certification error remains fail-closed but adds the
preview reason, accepted flag, source ID, selection ID, IK fingerprint, and both
IK-equality results. This preserves the gate while making a future rejection
actionable from one log.

## Alternatives rejected

1. **Gate exception:** accepting a rejected/zero-ID preview would invalidate
   runtime ownership and evidence provenance.
2. **Global tolerance relaxation:** changing placement, collision, correction,
   or IK limits would broaden behavior unrelated to the inconsistent scene.
3. **Pack rebuild:** the pack already validates; rebuilding without one shared
   authoring contract would preserve or merely relocate the mismatch.

The selected shared-scene correction fixes the source of divergence and gives
the graphical and headless paths one contract.

## Fixed scope

Do not change selector ordering or reasons, IK configuration, correction or
collision tolerances, placement timing, native 25 Hz scheduling, terrain,
locomotion, pack schema, evidence schema, or runtime ownership. Do not add a
graphical retry or run a new graphical gate until the implementation and safe
headless evidence receive review.
