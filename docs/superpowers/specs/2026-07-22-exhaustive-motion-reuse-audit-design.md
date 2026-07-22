# Exhaustive Motion Reuse Audit Design

**Date:** 2026-07-22

## Goal

Measure how many complete table and ground pickup clips can be reused for an
arbitrary world grasp after arm retargeting and full-motion collision checks,
without exceeding 30 seconds per explicit audit.

## User Interaction

`A` runs an exhaustive reuse audit for the current object and grasp. Ordinary
Enter search remains unchanged and responsive. Object movement marks both the
ordinary search and the last audit stale.

The audit HUD reports:

1. same-hand complete clips attempted;
2. clips whose Contact pose passes position and orientation IK;
3. fully shaped clips;
4. object-collision rejections;
5. furniture-collision rejections;
6. reusable full motions.

The viewer displays the best 12 reusable motions using achieved orientation,
retrieval cost, and clip ID ordering. Counts describe the entire audited set,
not only the displayed options.

## Candidate Scope

Audit all complete clips for the query's active hand from the mixed table and
ground pack. Do not apply the ordinary 12 cm position or 25 degree retrieval
gates. Use a 45 cm maximum Contact-position correction and allow up to 180
degrees at retrieval so IK and final acceptance, rather than the prefilter,
determine reuse.

In full-grasp mode, every candidate uses the existing approach-axis shaping:
full-quaternion IK with position-dominant weight `0.10`, at least 16 iterations,
and final hard limits of 4 cm position, 15 degree approach axis, and 60 degree
full orientation. In position-only mode, orientation remains unconstrained and
the 4 cm final position limit remains hard.

Left- and right-hand clips are not mirrored. Clips recorded for the inactive
hand are outside the audit population and are reported neither as attempted nor
as rejected.

## Exact Early Rejection

Before allocating and shaping a complete trajectory, run the same mapped
Contact target through the same arm IK and final acceptance rules. A clip that
fails Contact cannot be a reusable full motion, so it is counted as a Contact
IK rejection and requires no collision test. This is a logically exact early
exit, not sampling or estimation.

Every Contact survivor is then shaped over its complete Approach-through-Lift
window and evaluated against the object and all furniture using the existing
viewer collision configuration, including the 2 cm body clearance margin.
Only clips passing this full evaluation count as reusable.

The Contact-only and full shaping paths must share target construction and
acceptance code so their decisions cannot drift. Add C++ parity tests proving
that Contact audit acceptance and achieved orientation equal the Contact sample
from full shaping.

## Performance and Determinism

Use four worker threads. The database is immutable during audit; workers claim
candidate indices atomically and retain only per-candidate temporary pose
buffers. Accepted display results are merged by candidate index and sorted
after all workers join, making counts and displayed ordering deterministic.

The audit has a 30-second wall-clock deadline. Workers stop claiming new clips
at the deadline and finish only their current clip. If any candidate remains
unprocessed, the audit is marked `INCOMPLETE`, reports processed versus total,
does not publish reusable totals as exhaustive, and leaves the previously
displayed trajectories intact. There is no partial-success claim.

No audit worker may call raylib or mutate viewer state. The UI may block while
the explicit audit runs; ordinary Enter search remains the default iteration
path.

## Failure and Memory Rules

- Reject non-finite or invalid audit configurations through existing library
  validation.
- Do not retain shaped poses for rejected candidates.
- Retain full poses only for the selected animation; background paths retain
  only hand/elbow trajectories as today.
- Keep RSS below 1.1 GiB in the live 10-second stability check.
- Do not launch mesh or terrain renderers and do not signal the controller.

## Verification

Add C++ tests for the widened 45 cm request envelope, Contact/full-shape parity,
and unchanged 4 cm/15 degree/60 degree final gates. Add viewer tests for the
`A` control, four-worker bound, stage counters, deterministic merge, explicit
`INCOMPLETE` state, and unchanged Enter behavior.

Benchmark shelf-height and under-table query fixtures derived from
`make_coverage_environment` against `build/smart-pickup/table-ground-pack`;
each must either complete within 30 seconds or truthfully return `INCOMPLETE`
without replacing prior results. The feature is successful only when both real
audits complete within the deadline.
Then run the established Python gate, both C++ trajectory binaries, release
viewer build, and single-process live hash and bounded-memory checks.
