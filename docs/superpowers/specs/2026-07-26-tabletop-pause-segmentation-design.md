# Tabletop Pause-Bounded Reach Segmentation Design

## Goal

Extract only the inbound portion of each repeated tabletop reach: the neutral
pause through the contact pause, excluding every withdrawal frame.

## Root Cause

The existing detector searches for radial-distance peaks. In the tabletop
corpus, the hand often continues farther from neutral while withdrawing or
lifting after contact. The detector therefore starts late and can end after
withdrawal has begun.

## Design

Add a dedicated pause-bounded segmentation path for repeated reach-and-return
data. Compute the root-relative wrist trace and smoothed wrist speed. Contact
candidates are well-separated speed minima at least 0.18 m from neutral.
For each contact, the approach starts at the preceding neutral or local
distance minimum. Reject segments with less than 0.18 m wrist travel or fewer
than 10 frames.

Use this path only to regenerate the corrected tabletop corpus. Preserve the
original reach corpus and its segmentation behavior. Build a new annotation
document and tabletop-only pack from the detected boundaries.

## Verification

- A synthetic reach/hold/withdraw trace ends at the contact pause.
- No output segment contains post-contact withdrawal.
- The corrected corpus contains 203 captured reaches and 203 mirrors.
- Median inbound duration is approximately 1.60 seconds.
- The viewer shows only corrected tabletop sources.
