# Walking-Entry Funnel Prefetch

Date: 2026-07-21

Status: approved correction

## Goal

Overlap proposal generation with coarse locomotion so the character transitions
from walking directly into the learned pickup route instead of stopping or
freezing for inference.

## Design

On the first moving observation after Smart Pickup activation, freeze the
collision-checked annulus capture root as the predicted entry. Build the
condition from that predicted entry, the selected grasp, and current approach
velocity, then launch the existing provider asynchronously.

While the provider is pending, keep publishing ordinary coarse steering toward
the frozen entry. When proposals arrive, validate and preview them without
braking. Store the selected route but do not start its follower while the
character is far away. At 0.12 metres from the predicted entry and within the
existing yaw tolerance, arm the spatial follower and let it project the moving
root onto the route. The 0.12-metre lead stays below the follower's 0.18-metre
cross-track limit and avoids requiring an exact stop at the first sample.

If inference is still pending when the character reaches the entry, the
ordinary arrival controller may still stop. Prefetching from the 3-metre
activation range is expected to hide the current CUDA startup/inference time;
a persistent model worker is the later fix for close-range activation.

Production remains at 50 DDIM inference steps.

## Verification

- A moving observation outside the annulus launches a request conditioned on
  the predicted annulus entry, not the current 3-metre root.
- Pending generation continues non-stationary coarse steering.
- Ready proposals and matcher preview do not brake locomotion.
- A selected route waits until the live root is within 0.12 metres of entry.
- The first follower tick starts from the moving root and existing Carry
  integration remains green.
