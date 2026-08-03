# G1 Interactive Terrain Coverage Design

## Status

Approved for autonomous execution under the operator's standing instruction to
continue quality-first terrain improvement without waiting for feedback.

## Goal

Prevent valid interactive commands from appearing frozen when an immediate
mid-step replan has no feasible exact-contact replacement. Preserve the strict
terrain/contact validator and the qualified v58 route behavior.

## Evidence

The live viewer remained mapped, focused, and sampled keyboard input. At the
reported frozen pose, the matcher rejected all 64 validated candidates:

- 45 for landing-height mismatch;
- 19 for stance-height mismatch; and
- 9,637 lower-ranked candidates were never exactly validated.

The viewer then correctly latched that failed command to avoid repeating an
expensive identical search. Offline stress routes reproduced the same failure
at command boundaries. Both diagonal descents exhausted their complete
prefiltered inventories, proving that a larger shortlist alone cannot solve the
general failure.

At each reproduced boundary the phase-gated horizon adapter forcibly truncated
the current chunk even when the current source support was single stance. This
bypassed the base matcher's safe-interrupt rule. A failed replacement therefore
aborted instead of preserving the already-valid step until double support.

## Considered Approaches

### Repeat the failed 64-candidate search

Rejected. A deterministic search with unchanged state and command produces the
same failure while consuming compute and blocking the viewer.

### Relax exact contact constraints

Rejected. Responsiveness obtained by accepting invalid landing or stance
geometry would undo the sole/contact qualification and reintroduce the stair
collisions that earlier variants exhibited.

### Adaptive exact-search expansion

Retained as an ablation for failures with an unvalidated shortlist. It cannot
solve failures whose complete prefiltered inventory is exact-invalid.

### Transactional immediate replan

Selected. Keep the current immediate command-change search for responsiveness.
If that search has no exact-valid replacement while the current skill is not at
a safe interrupt frame, discard the attempted interruption, advance the
already-valid chunk, retain the new command as pending, and retry through the
base matcher's existing double-support gate. Do not catch failures at a safe
interrupt point or endpoint; those remain true coverage failures.

## Architecture

Make forced mid-step interruption transactional in the horizon matcher. The
temporary truncated state is used only for the immediate attempt. On
`HorizonSearchFailure`, restore the original state and invoke the existing base
prepare path, which advances the skill and marks the command for replanning.
The next source-labelled double-support frame performs the normal exact search.
No fallback pose is synthesized and no terrain failure is accepted.

The offline route runner and live viewer use the same matcher implementation.
No viewer-only retry or fallback motion is permitted.

## Evaluation

Use the existing deterministic same-stair route corpus, prioritizing:

- `riser-stop-restart`;
- `riser-reversal`;
- both 180-degree turns;
- upper and lower side exits;
- side mounts; and
- `mixed-adversarial`.

For every retained version:

- count completed routes and exact coverage failures;
- measure command-change-to-root-motion delay and longest moving stall;
- retain exact emitted-contact validation and sole clearance;
- compare stance slide and root jerk with v58; and
- render failed and improved routes for visual inspection.

## Acceptance

A change is retained only if it reduces deterministic coverage failures or
moving-command stalls on the adversarial routes, keeps immediate preemption
when an exact-valid replacement exists, keeps the v58
`turn-90-middle-right` route complete, never accepts an exact terrain-invalid
candidate, and does not increase that route's `0.411985839 m` stance-slide
baseline.

## Scope

This work remains privileged-height, kinematic, and independent of Sonic,
physics stepping, depth learning, and real-time latency optimization. Existing
dirty contact-oracle and landing-bridge work is not modified.
