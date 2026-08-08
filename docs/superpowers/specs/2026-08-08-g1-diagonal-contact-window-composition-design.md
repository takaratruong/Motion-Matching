# G1 Diagonal Contact-Window Composition Design

## Goal

Produce a natural same-heading diagonal staircase traversal without replacing
valid GRAIL gait cycles with independently generated MotionBricks segments.

## Diagnosis

The current terrain task actor classifies a complete proxy interval as flight
when any source frame is unsupported. In the diagonal route, a two-frame
contact gap therefore becomes a 32-frame generated flight and a 19-frame gap
becomes a 36-frame generated flight. Candidate ranking then prioritizes the
requested duration without enforcing alternating touchdowns, cadence, or
whole-route continuity. The accepted route consequently has repeated
same-foot touchdowns, five cadence violations, 68 unsupported frames, and a
0.58-radian one-frame joint jump.

## Architecture

The GRAIL proxy remains the authoritative motion outside localized invalid
contact runs. A pure detector converts each unsupported run into a bracketed
repair window with four exact context frames before and after it. MotionBricks
generates candidates only inside those windows, resamples them from 30 Hz to
the route's 50 Hz, and splices them between byte-identical source spans.

Candidate support comes from generated contact channels and terrain-relative
sole evidence; a single unsupported source frame never changes the support
schedule for an entire interval. Candidate selection evaluates both the local
window and the assembled route. A candidate is rejected if it introduces
repeated-foot touchdowns, same-height cadence violations, unsupported stance,
invalid complete-sole placement, or a joint/root discontinuity larger than
the retained source baseline.

If no generated candidate passes, the source route remains unchanged and the
failed window is reported. No aggressive pose correction or whole-route
MotionBricks replacement is allowed.

## Acceptance

- Every frame outside selected repair windows is byte-identical to the GRAIL
  proxy.
- Touchdowns alternate feet.
- Same-height cadence has zero violations; terrain-changing intervals remain
  exempt.
- Maximum joint step is at most 0.25 radians and maximum root step is at most
  0.04 metres.
- No unsupported stance frame is admitted; deliberate flight is accepted only
  inside its explicit bracketed window.
- The independent traversal validator and a dense visual audit both pass
  before the route is launched for feedback.

## Scope

This iteration repairs the current negative-45-degree center traversal. The
detector and splice contract are angle-independent so the same path can later
be applied to every diagonal grid lane without angle-specific constants.
