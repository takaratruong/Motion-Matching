# Diffusion-Assisted Pickup Evidence

## Status

The diffusion-assisted branch appears to work as a research prototype. In the
interactive flat-terrain skeleton visualizer, the object-relative pickup motion
can approach the demo object from different live entry positions and angles.
The user reported that the corrected version looked better than the earlier
fixed-origin and unrelated-slot previews.

This is evidence for continuing to evaluate diffusion as a transition or stitch
generator. It is not evidence that diffusion is required for the pickup itself.

## What Worked

- The Stage-B pickup expert was conditioned on the exact `beer_10` object/grasp
  row rather than an unrelated validation row.
- The pickup clip was mapped through its own object-relative root frame instead
  of interaction slots authored from other objects.
- A live locomotion pose can be bridged into the object-relative pickup from
  multiple entry positions and headings without returning to the origin.
- The flat-terrain blue G1 skeleton visualizer runs without loading the G1 mesh
  or terrain mesh.
- Contact-time object following and a bridge that settles before the reach are
  implemented and covered by focused tests.

## Remaining Limitations

- The demonstrated diffusion motion is an offline generated clip, not resident
  live diffusion inference for each request.
- The user still observed some sliding near the grasp before the latest
  contact/bridge update; that update is test-covered but has not yet been given
  a separate user visual verdict.
- Carry and placement ownership are not fully integrated with the offline
  diffusion preview.
- The raw Stage-B full-body samples required temporal stabilization.

## Baseline Decision

Further work moves to a separate baseline branch. The baseline uses motion
matching for locomotion and approach, object-relative recorded pickup clips,
foot-preserving trajectory warping, constrained contact IK, and explicit object
attachment. The immediate diagnostic is a live visualization of the closest
recorded hand trajectories mapped around the current object.

Diffusion remains a later candidate for stitching the selected walking and
pickup windows after the deterministic baseline is measured.
