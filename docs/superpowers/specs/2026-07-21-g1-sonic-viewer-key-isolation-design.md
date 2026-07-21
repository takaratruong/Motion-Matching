# G1 SONIC Viewer Key Isolation Design

**Date:** 2026-07-21

**Status:** Approved

**Base:** `e767b3f6a31762e4c734debf34efdb10ed2a8ffa`

## Problem

The interactive Sonic operator and MuJoCo passive viewer currently receive the
same X11 keystrokes. MuJoCo interprets several locomotion controls as viewer
shortcuts. In particular, pressing `D` toggles `mjVIS_STATIC` off, so the
world-body terrain disappears even though the authenticated heightfield remains
active for collision. `A`, `Q`, and `E` also toggle unrelated visualization
flags.

The failed visible run retained real `mm_terrain` contacts throughout. This is
a presentation/input-routing defect, not removal of terrain physics.

## Selected Approach

Keep the existing `WASD` control layout and isolate Sonic controller keys from
MuJoCo at the X11 boundary.

`X11KeyStateProvider` will install passive grabs on the focused MuJoCo operator
window for the non-modifier Sonic keys: `W`, `A`, `S`, `D`, `Q`, `E`, `X`,
`Space`, and the four arrow keys. Each grab covers every modifier combination.
The provider continues to read physical key levels through `XQueryKeymap`, so
the existing continuous control path, focus gate, latency, and command mapping
remain unchanged. Shift and Control do not need independent grabs; their
combinations are covered by the grabbed command key.

The provider retains ownership of the grab connection for its lifetime,
installs each window/key grab once, releases grabs during close, and reacquires
them if a replacement MuJoCo window becomes the operator target. A failed grab
is fatal and publishes neutral control rather than silently allowing dual key
handling.

When the onscreen backend creates a viewer, it will also establish the expected
initial presentation state explicitly: static geometry and the terrain render
group are enabled. This is startup normalization, not a per-frame override;
mouse camera controls and intentional non-controller viewer interaction remain
available.

## Alternatives Rejected

- Reapplying viewer flags every frame could mask the terrain symptom, but other
  MuJoCo shortcuts would still mutate viewer or simulation state and may flicker
  between updates.
- Remapping locomotion to the arrow keys would collide with MuJoCo step/camera
  shortcuts and with Sonic's existing orbit and Ctrl-strafe controls.
- Changing the terrain collision or mesh assets would not address the shared
  keyboard event path that caused the disappearance.

## Contracts and Failure Handling

- Headless and non-X11 modes remain unchanged.
- The movement model, MM search, control-gate protocol, physics stepping, scene
  registration, and collision geometry remain unchanged.
- Focus loss still publishes neutral control.
- A destroyed target window releases or invalidates its recorded grabs and the
  provider may bind only a newly authenticated operator-title window.
- Unexpected X11 errors, including a denied grab, fail closed through the
  existing `ContractError` path.
- Closing the provider releases all grabs and closes its X11 display exactly
  once.

## Verification

1. A regression test first demonstrates that the current provider does not
   suppress a controller key from the MuJoCo target.
2. Unit tests prove every non-modifier controller key is grabbed with
   `AnyModifier`, repeat samples do not duplicate grabs, replacement windows are
   handled, and close releases ownership.
3. Existing X11, manual-demo, gated-simulator, and responsive-control tests
   remain green.
4. A visible real canary reaches live control with the authenticated curb scene.
5. Holding `D` produces Sonic rightward intent while `mjVIS_STATIC` and terrain
   rendering remain enabled.
6. A captured MuJoCo frame visibly contains the terrain, and scored contact
   logs contain `mm_terrain` contacts, demonstrating render/collision agreement.
7. Exiting through `X` leaves no Sonic, simulator, MM server, or GEAR process.

## Rollback

Rollback is the parent of the implementation commit. No runtime artifact,
terrain database, or GEAR policy change is required.
