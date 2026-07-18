# Smart Pickup Manual Override Design

## Problem

The keyboard controller remains alive and renders at 25 Hz after `F`, but
Smart Pickup intentionally owns the locomotion sticks during its approach,
settling, and final-preview states. A player who tries to move sees their
input discarded and reasonably interprets that as a frozen interface. The
live failure reproduced on 2026-07-18 ended in
`FinalPreviewRejected / CorrectionLimit`; it was not a process deadlock.

## Decision

Keep `F` as the explicit transition from ordinary locomotion into automatic
Smart Pickup. While Smart Pickup owns a pre-submission attempt, a newly
pressed `W`, `A`, `S`, or `D` key cancels that attempt before the next
locomotion step and the same tick passes the player's raw steering through.
`X` remains the explicit cancel key and `R` remains reset.

This follows the game-interaction convention that an interaction can align
the character automatically, but renewed movement intent immediately gives
control back to the player. It also preserves the useful behavior of pressing
`F` while already walking: a movement key pressed on the same tick as `F`
does not cancel the new attempt. Cancellation requires a later movement-key
press while Smart Pickup already owns the attempt.

The controller will display an unambiguous active-state message:
`SMART PICKUP AUTO - WASD or X cancels`. The permanent control legend will
also describe the override.

## Alternatives Considered

1. Add only a status banner. This explains the behavior but still leaves the
   player unable to take control without remembering `X`.
2. Cancel on renewed movement input and pass it through immediately. This is
   the selected bounded fix because it restores control without changing
   matching or animation semantics.
3. Move all full-pack preview and preflight matching to a background worker.
   This may still be valuable later, especially for repeated `PoorMatch`
   previews, but it is a larger concurrency change and does not address the
   basic ownership surprise by itself.

## Interfaces and Data Flow

`SmartPickupPreStepInput` gains a `manual_override_pressed` boolean. The
native controller sets it from `IsKeyPressed(KEY_W)`, `KEY_A`, `KEY_S`, or
`KEY_D` on the fixed 25 Hz input tick.

`SmartPickupPreStepResult` gains `manual_override_consumed`. When an attempt
was already pending or active and this edge is true, `SmartPickupController`
clears the pending activation and prior assist output, cancels the assist
backend, reports `manual_override_consumed`, and returns the raw left stick,
right stick, and strafe intent unchanged. It must not emit a pick request or
forward a scheduler cancel edge.

The override does not cancel an interaction after the assist has submitted a
request to the runtime. At that committed stage, `X` or `R` remains required;
this avoids silently interrupting an attached or held object.

## Error and State Handling

- Pending activation: renewed movement cancels before backend `begin`.
- Active approach/settling/final preview: renewed movement cancels before the
  next observation or preview.
- Idle, failed, or submitted assist: movement passes through normally and is
  not reported as an assist cancellation.
- Simultaneous idle `F` plus movement edge: `F` starts Smart Pickup and wins
  that tick.
- Explicit `X`/`R`: existing cancellation precedence and reset behavior stay
  unchanged.

## Verification

Test first at the raylib-free `SmartPickupController` seam:

- an active attempt is cancelled exactly once and raw steering passes through;
- a pending activation is cancelled before backend begin/observe/preview;
- simultaneous idle `F` plus movement still starts an attempt;
- idle, failed, and submitted states do not consume movement override;
- existing `X`, `R`, request-latch, and exact 25 Hz contracts remain green.

Add a controller-source integration contract for the four keyboard edges and
the active-state copy, compile the native controller, run the focused C++ and
integration suites, then launch a fresh full-pack controller for manual
verification. This slice does not change the matcher, interaction data,
terrain behavior, or the 25 Hz control frequency.
