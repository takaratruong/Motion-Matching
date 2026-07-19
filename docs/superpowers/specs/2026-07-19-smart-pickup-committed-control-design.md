# Smart Pickup Committed-Control Design

## Problem and Evidence

The playable controller is not deadlocked after a successful Smart Pickup
submission. The live process continues to render and sleep at 25 Hz. The
selected full-pack clip `pickup_table__milk_7__000` enters Reach at frame
360085, Contact at 360110, Lift at 360122, and Hold at 360212. At playback
speed 1.0, the runtime therefore owns the character for 5.08 seconds before
Carry begins.

The existing movement override ends when the assist changes to `Submitted`.
The runtime then owns the pose in Align and PickupReplay, suppresses steering
in PickupReplay and Hold, and only honors `X` while it is still in Align. A
player pressing movement during the committed pre-contact replay sees no
response and reasonably interprets the application as frozen.

Submission is also not proof of pickup. A successful attempt attaches only at
the recorded Contact event and becomes Held later. Other attempts can be
rejected before submission by the approach or final-preview gates. The UI
must not present `Submitted` as equivalent to a successful pickup.

## Approaches Considered

1. Extend state-aware cancellation through the committed pre-contact replay
   and display truthful phase-specific feedback. This is selected. It follows
   the game interaction convention that renewed movement cancels an automatic
   alignment/reach until the interaction's irreversible attachment event.
2. End the recorded clip at Contact or Lift and immediately synthesize Carry.
   This would reduce the lockout, but it discards the authored lift/turn and
   risks the arm, root, and carry-transition artifacts already observed.
3. Move full-pack preview matching to a background worker. This can reduce a
   synchronous preview hitch, but the measured live failure is a healthy
   5.08-second runtime replay, so concurrency does not solve this ownership
   problem.

## Decision

Smart Pickup has three player-facing ownership phases:

- **Automatic approach:** the assist owns steering. A newly pressed WASD key
  or `X` cancels and returns raw locomotion control.
- **Committed but unattached reach:** the runtime owns the pose, but a newly
  pressed WASD key or `X` cancels before Contact. The reservation and targeted
  attachment state are released, the object remains at its original world
  transform, and the same tick must not retain the stationary constraint.
- **Attached pickup:** once Contact succeeds, ordinary movement no longer
  silently cancels the grasp. The authored lift continues to its transition-
  safe Hold phase, then Carry permits locomotion while maintaining the grasp.
  `R` remains the explicit recovery control.

The permanent controls and temporary banners must describe the active phase
without claiming success early:

- `SMART PICKUP AUTO - WASD or X cancels`
- `SMART PICKUP REACH - WASD or X cancels`
- `SMART PICKUP ATTACHED - finishing recorded lift`
- `CARRY - WASD moves, F places, R resets`
- `PICKUP FAILED: <reason> - reposition and press F`

`Submitted` remains a diagnostic transport state, not a user-visible success
claim. Pickup success requires the runtime diagnostics to report an attached
object and, ultimately, `ObjectState::Held` / `RuntimeState::Carry`.

## Interfaces and Data Flow

The native controller derives one fresh WASD edge and reuses it for both
pre-submission assist cancellation and committed pre-contact runtime
cancellation. It forwards a scheduler cancel edge only when the cached runtime
state is Align or PickupReplay and the cached diagnostics say the object is
not attached. Raw stick input remains available to ordinary flat locomotion
on that tick.

`InteractionRuntime::update` accepts `cancel_pressed` during PickupReplay only
while no contact has attached the object. It follows the same rollback path as
Align cancellation: release the reservation, preserve the original object
transform, clear players/candidate/target/affordance state, and publish
`Locomotion / Cancelled / ObjectState::Free / attached=false`.

After `ever_attached_` becomes true, movement does not emit a runtime cancel
edge. Runtime cancellation also refuses to roll back an attached pickup. This
keeps the irreversible boundary in one authoritative layer instead of relying
solely on potentially stale controller diagnostics.

No matcher cost, database phase, playback rate, carry controller, placement
behavior, terrain code, or control frequency changes in this slice.

## Error Handling

- Cancellation before Contact must release any owned reservation exactly once
  and leave the object free at its original transform.
- A stale or changed target continues through the existing rejection path.
- Cancellation at or after successful Contact does not detach or teleport the
  object; `R` remains the recovery path.
- A failed assist or runtime preflight displays its actual reason and does not
  display a pickup-success message.
- All controller and runtime updates remain exactly 1/25 second.

## Verification

Tests are written and observed failing before production edits:

1. A runtime fixture reaches PickupReplay before Contact, receives cancel, and
   returns Locomotion/Cancelled with the original free object and no owned pose.
2. A runtime fixture reaches successful Contact, receives cancel afterward,
   and proves the attached interaction is not silently rolled back.
3. A controller integration contract proves a fresh WASD edge forwards cancel
   only for cached Align/PickupReplay with `attached=false`, preserves raw
   steering, and does not affect Carry or an attached replay.
4. Source/UI contracts require the phase-specific copy and forbid treating
   `assist=Submitted` alone as pickup success.
5. Focused C++ tests, Python playable contracts, the native controller build,
   and the existing 25 Hz verification suite pass before commit and push.
6. A fresh controller is launched without touching the terrain-aware process;
   manual acceptance confirms pre-contact takeover, visible attachment, and
   normal Carry steering.
