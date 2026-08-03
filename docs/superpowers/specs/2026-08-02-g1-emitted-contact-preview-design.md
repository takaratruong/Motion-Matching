# G1 Emitted-Contact Preview Design

## Problem

The terrain matcher currently validates a rigidly root-placed source-foot trace,
but the viewer renders a different trajectory after pose inertialization and
terrain foot-lock filtering.  The qualified four-route build therefore passed
root-progress gates while visibly floating, penetrating terrain, and landing on
stair edges.

An audit of the emitted kinematics found stance errors of 5--9 cm, landing
errors up to 10.8 cm, footprint discontinuities up to 35 cm, and a side-exit
swing penetration of 30 cm.  Replacing the coarse check with the existing
source-contact validator removed source edge contacts, but emitted stance error
still reached 38.7 cm on diagonal descent and 60.3 cm on side exit.  This proves
that validation must cover the emitted transition, not only the source clip.

## Decision

Add a read-only emitted-contact preview as the final candidate validator.
For each candidate that passes inexpensive ranking and source-terrain checks:

1. Start the skill from the current committed pose with the same placement,
   inertialization half-life, endpoint, and zero warp used by playback.
2. Advance the candidate transactionally through its proposed horizon without
   mutating matcher or foot-lock state.
3. Batch the emitted joint/root states through the existing G1 foot FK.
4. Validate every emitted support, landing footprint, and swing sample against
   the query terrain.
5. Reject unsupported entry frames and candidates that fail emitted contact.

The preview is deliberately layered after cheap checks because it is more
expensive.  Search latency is not a qualification constraint for this
kinematics stage; rendered quality is.

## Alternatives

- **Source-contact validation only:** cheaper, but the measured output proves
  it cannot see inertialization-induced failures.
- **Immediate footstep/root warping:** likely useful later, but it adds a new
  placement optimization before establishing whether the existing database has
  sufficient emitted-valid coverage.
- **Post-playback IK repair:** cannot reliably repair 10--60 cm errors and can
  create the distorted poses already visible in the viewer.

The emitted preview is the smallest change that evaluates exactly what the
user sees.  If it exhausts coverage or prevents commanded progress, that is
direct evidence that a stance/landing placement optimizer is required next.

## Interfaces

Add a pure preview helper in the horizon rollout module.  It consumes the
candidate skill, entry and endpoint frames, current pose, terrain sampler,
support trace, FK provider, and contact-feasibility configuration.  It returns
the existing `TerrainContactFeasibilityResult` so rejection reasons remain
structured.

The global horizon validator calls it only when emitted-contact preview is
enabled.  Existing behavior remains available for controlled A/B comparison.

## Qualification

Automated tests must prove that a source-valid candidate whose inertialized
output penetrates terrain is rejected, while a genuinely emitted-valid
candidate is accepted without mutating committed state.

The hard-route qualification must then satisfy all of the following on emitted
FK, not source data:

- no landing footprint crosses a height discontinuity greater than 2.5 cm;
- minimum swing clearance is at least -5 mm;
- maximum landing error is at most 3 cm;
- stance error p95 is at most 5 cm;
- no route loses its required traversal outcome.

Finally, render side-view contact sheets and transition-dense frame strips for
all four routes.  Do not launch the interactive viewer until those renders have
been inspected directly and found free of obvious floating, clipping,
edge-landings, and frozen commanded motion.
