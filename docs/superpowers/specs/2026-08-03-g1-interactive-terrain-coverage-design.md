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

## Execution Findings

The transactional single-support correction is retained. It preserved the
qualified v58 route bit-for-bit and moved `cross-tread-left-to-right` past its
frame-60 command boundary.

That longer route exposed a missing qualification check: the matrix's ankle
metric passed while exact sole geometry reached `-0.051865 m` clearance on 19
frames. The unsafe samples came from sequential source-skill continuations,
which used the coarse surface-profile validator instead of the exact emitted
sole preview. Sequential continuations must therefore pass the same filtered
pose and sole validation as newly selected chunks. A route that stops earlier
because this check rejects an extension is safer, but is not counted as a
locomotion-quality improvement.

The following recovery ideas were tested and rejected as defaults:

- ten stationary command-shaping retries only delayed diagonal descent from
  frame 282 to frame 292 and preserved the same failure;
- validating 512 instead of 64 candidates took more than 30 minutes per
  two-route slice, still failed every crossing/mount route, and often selected
  flat clips that never engaged the stair;
- disabling the terrain prefilter failed earlier because the first 64 ranked
  candidates were all exact-invalid; and
- increasing the future-surface sign gate allowed the left diagonal route to
  reach frame 360, but it moved opposite the requested descent and therefore
  regressed command outcome despite greater frame coverage.

These results rule out treating frame count alone as progress. Retained search
changes must improve commanded segment progress and exact sole safety together.

Exact continuation preview changes the old v58 turn route after output frame
328. This is intentional: the old route stops at source frame 445, one frame
before its selected continuation begins a landing that would reach
`-0.174130 m` sole clearance. The replacement route completes safely (minimum
sole clearance `-0.024206 m`) but has worse final heading and slide, so the old
frame-count contract is not sufficient evidence of a safe indefinite hold.

A latched stationary command now preserves a completed exact-safe pose when no
stationary replacement exists. Repeating the same stop does not repeat the
expensive deterministic search. This moved `riser-stop-restart` from a failure
at frame 179 to its restart boundary at frame 283; it did not by itself solve
the subsequent moving transition.

A two-tier monotonic exact search is retained. The normal 64-candidate search
and runway preference remain unchanged. Only after that search fails, a larger
shortlist is evaluated with runway preference disabled, so increasing rescue
coverage cannot replace an earlier successful choice. With a 256-candidate
rescue, `riser-stop-restart` completed all 290 frames, achieved a `0.544725`
restart progress ratio and `0.254820 m` total stance slide, and had zero sole
samples below `-0.025 m` (minimum `-0.024206 m`). The same rescue exhausted the
complete 167-candidate prefiltered crossing inventory without finding a safe
cross-tread transition, proving that rescue depth is useful but not the general
on-stair turning solution.

The accepted corpus does contain turn outcomes at split-height support: 7,870
horizons start with more than `0.10 m` foot-height split, and 278 horizons pair
that split with more than 20 degrees yaw and less than `0.20 m` root travel.
The remaining failure is therefore transition reachability/ranking rather than
simple absence of turning data. A hard support-foot entry-continuity prefilter
was also rejected: although it reduced exact stance-height rejections from 64
to 21 at one boundary, it removed useful candidates and made the staged turn
fail at frame 202 instead of frame 300. Transition reachability should be used
as a soft ranking feature or a rescue-only ordering signal, not a hard gate.
