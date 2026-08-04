# Complete Supported G1 Step-Up

## Goal

Produce one kinematic G1 step-up across the target staircase that begins with a
lower-floor stance, lands the leading foot on the elevated tread, transfers
support, and finishes with both feet supported on that tread.

## Source motion

Use a single authenticated GRAIL motion whenever possible. The selected source,
`grail-curb-02b3cb0048c3c28045fc`, already contains both required contacts:

- leading-foot elevated contact at source frame 249;
- trailing-foot elevated contact at source frame 289.

The emitted segment retains the approach before frame 249 and continues beyond
frame 289 until stable elevated double support. A cross-clip splice is only a
fallback if this native continuation fails target-scene validation.

## Contact and placement contract

Foot contact is detected using both geometry and motion state: ankle-to-surface
distance must be within the frozen stance tolerance and vertical foot speed must
be within the frozen stance-speed threshold. The target-scene placement must
satisfy all of the following:

1. At least one source-authenticated stance foot remains in geometric contact
   with the target terrain on every frame.
2. At the first elevated contact, the leading foot is supported on the target
   tread while the trailing foot remains supported on the lower floor.
3. At completion, both feet are supported on the same elevated tread.
4. Every sampled point of each completed sole lies on that tread within the
   contact tolerance.
5. No sole sample penetrates the target terrain by more than 25 mm.
6. The character faces within 10 degrees of its direction of travel.

Candidates are ranked only after these gates pass. Ranking prefers smaller
stance-contact error, smaller penetration, and smaller facing error.

## Output and visualization

Write a self-contained NPZ containing joint positions, root position and
orientation, contact masks, phase boundaries, and per-frame minimum sole
clearance. The metrics file reports both elevated contact frames, unsupported
frame count, final double-support status, contact error, sole clearance, and the
selected source identity.

Render a contact sheet for inspection before launching the existing passive
MuJoCo viewer. The viewer remains kinematic and loops the complete step-up.

## Verification

Automated tests cover contact-event extraction and reject:

- clips cropped after only the leading-foot landing;
- any unsupported interval;
- a trailing foot that never reaches the elevated tread;
- incomplete or edge-supported final soles;
- excessive penetration.

The final artifact is accepted only when its reported unsupported-frame count is
zero and its final elevated double-support flag is true.
