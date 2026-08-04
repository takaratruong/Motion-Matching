# Complete Supported G1 Step-Up

## Goal

Produce one kinematic G1 entry onto the target staircase that begins with a
lower-floor stance, lands the leading foot on an elevated tread, transfers
support, and finishes with both feet supported on the unequal stair levels
beneath them.

## Source motion

Search authenticated GRAIL stair and curb motions for a complete two-contact
sequence:

- the leading foot rises from the lower approach surface to its target tread;
- the trailing foot subsequently lands at the different height required by its
  target stair surface.

The source height difference between those contacts must match the target
height difference after rigid placement. A level-platform curb continuation is
rejected when the target requires unequal support. A cross-clip splice or IK
retarget is only a fallback if no native GRAIL sequence passes.

## Contact and placement contract

Foot contact is detected using both geometry and motion state: ankle-to-surface
distance must be within the frozen stance tolerance and vertical foot speed must
be within the frozen stance-speed threshold. The target-scene placement must
satisfy all of the following:

1. At least one source-authenticated stance foot remains in geometric contact
   with the target terrain on every frame.
2. At the first elevated contact, the leading foot is supported on the target
   tread while the trailing foot remains supported on the lower floor.
3. At completion, both feet are supported on their respective target stair
   surfaces; those surfaces may and generally will have different heights.
4. Every sampled point of each completed sole lies on its respective support
   surface within the contact tolerance.
5. No sole sample penetrates the target terrain by more than 25 mm.
6. The character faces within 10 degrees of its direction of travel.

Candidates are ranked only after these gates pass. Ranking prefers smaller
two-contact height-pattern error, smaller stance-contact error, smaller
penetration, and smaller facing error.

## Output and visualization

Write a self-contained NPZ containing joint positions, root position and
orientation, contact masks, phase boundaries, and per-frame minimum sole
clearance. The metrics file reports both contact frames, source and target
contact-height differences, unsupported frame count, final split-height
split-height contact-transfer status, contact error, sole clearance, and the selected source
identity.

Render a contact sheet for inspection before launching the existing passive
MuJoCo viewer. The viewer remains kinematic and loops the complete step-up.

## Verification

Automated tests cover contact-event extraction and reject:

- clips cropped after only the leading-foot landing;
- any unsupported interval;
- a trailing foot that never reaches its target stair surface;
- a level-platform continuation when the target contacts are split-height;
- incomplete or edge-supported final soles on either stair surface;
- excessive penetration.

The final artifact is accepted only when its reported unsupported-frame count is
zero and its final split-height contact-transfer flag is true. A dynamic
transfer is valid when the trailing contact is stable and support is
continuous, even if the landing foot has begun its next swing.
