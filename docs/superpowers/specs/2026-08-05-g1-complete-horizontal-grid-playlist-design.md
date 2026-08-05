# G1 Complete Horizontal Grid Playlist

## Goal

Attempt a complete mount, horizontal crossing, and dismount at every 20 cm
lane center from `y=-1.0 m` through `y=+1.0 m`, and show every independently
certified feasible traversal.

## Design

Treat each of the 11 lines as a distinct height-conditioned traversal problem.
Reuse an existing complete lane only when rigid placement passes the target
lane's terrain gates. Otherwise compose that lane's own retrieved mount,
interior, and dismount. Do not substitute a motion from a different height
when it fails certification.

Certify each translated traversal against the actual target height field using
the seven oriented sole samples per foot. A lane is accepted only when minimum
sole clearance is at least `-0.03 m`, maximum stance error is at most `0.03 m`,
and every frame has at least one supported foot.

Concatenate all accepted complete traversals with frozen separator frames and
write playlist metadata that identifies the active lane. Emit an explicit
per-lane result for all 11 lines: `complete` when all three phases compose and
pass, otherwise `infeasible` with the failed phase or terrain gate. A high
mount that cannot be produced by the library is a valid infeasible result.

## Verification

- Exactly 11 lane segments at 20 cm spacing.
- Every displayed segment contains mount, crossing, and dismount phases.
- No lane is accepted using another height's motion unless the translated
  placement independently passes.
- Every lane independently passes the frozen terrain gates.
- The existing MuJoCo viewer displays all feasible complete traversals and the
  active lane.
