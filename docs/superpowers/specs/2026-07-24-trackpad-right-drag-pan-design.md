# Trackpad Right-Drag Camera Pan

## Goal

Make camera panning usable from a trackpad without a middle mouse button.

## Interaction Contract

- Left-button drag continues to orbit.
- Right-button drag pans the camera target in the current view plane.
- Middle-button drag remains a compatible alternative for panning.
- The mouse wheel continues to zoom.
- If right and left are held together, panning takes precedence so one gesture
  cannot modify both orbit angles and the camera target.

## Approaches Considered

1. Add right-drag as a second pan gesture while retaining middle-drag.
   This is selected because it serves trackpads without removing existing
   controls.
2. Replace middle-drag with right-drag. This is simpler but needlessly breaks
   the existing mouse workflow.
3. Use a modifier plus left-drag. This avoids right-click handling but is less
   discoverable and awkward on a trackpad.

## Implementation

The existing camera update function will derive a `pan_down` state from either
the right or middle mouse button. Orbiting will run only when the left button is
down and `pan_down` is false. Panning will continue to use the existing
distance-scaled view-plane translation and target clamps.

The HUD control hint will say `left orbit | right/middle pan | wheel zoom`.

## Verification

The viewer source-contract test will require right-button support, the
right-over-left precedence condition, and the updated HUD hint. The test must
fail before production code changes and pass afterward. The release viewer will
then be rebuilt and right-dragged through X11 to confirm that its rendered view
changes while the process remains responsive.
