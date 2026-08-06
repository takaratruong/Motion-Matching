# G1 Contact-Compatible Exit Retrieval Design

## Goal

Produce one natural, same-heading horizontal traversal of the target staircase:
mount, walk across, and step down. Preserve the accepted raw GRAIL motion
interiors and synthesize no missing locomotion phase with IK.

## Observed Failure

The accepted route prefix ends in asymmetric single support:

- left foot planted on the low tread;
- right foot airborne roughly 30 cm higher;
- support mask `[true, false]`.

The isolated step-down seed begins in equal-height double support. The previous
connector therefore had to invent a long re-leveling maneuver. It met numeric
clearance thresholds but looked mechanical and floated.

The raw continuation of the across-walk source was also tested. It is natural
on its original terrain, but after the selected cut it walks on an elevated
surface that does not exist in the target scene. It is not a valid solution.

## Architecture

Keep the accepted mount and across-walk unchanged. Retrieve the exit from the
existing target-terrain-certified candidate pool, conditioned on the actual
terminal contact state of the walk.

For every candidate boundary pair:

1. require compatible support semantics;
2. compare the world-space positions of the common support soles;
3. compare the swing-foot state when the incoming boundary adds a contact;
4. compare root pose, heading, joint pose, and joint velocity;
5. reject candidates whose contact geometry exceeds a hard tolerance before
   ranking the remaining candidates.

The selected exit remains an unedited raw GRAIL segment. Only a short boundary
blend is permitted after both source endpoints pass the contact gate.

## Data Flow

1. Load the accepted route prefix and compute its sole centers.
2. Load already-scanned dismount candidates and rigidly place each one on the
   target path.
3. Compute candidate sole centers after placement.
4. Rank only boundary cuts whose support and sole geometry are compatible.
5. Assemble the prefix and selected raw exit.
6. Validate terrain contact and render an offscreen visual audit.

## Acceptance

- Same heading throughout the route.
- No visible floating, pawing, crouch shuffle, or invented high-knee connector.
- The support foot does not jump at the walk-to-exit boundary.
- Stance soles remain within 3 cm of target terrain and do not penetrate more
  than 3 cm.
- The route ends on the far flat ground in supported stance.
- The full rendered traversal is inspected before launching the interactive
  viewer or claiming success.

## Non-Goals

- Sonic tracking or physics.
- The 20 cm grid or additional headings.
- IK retargeting, MotionBricks generation, or ARDY generation.
- Improving the already accepted mount and across-walk interiors.
