# Trained-Object Entry Generalization and Extended Pickup Range

Date: 2026-07-21

Status: approved for implementation

## Goal

Let learned Smart Pickup start from as far as 3.0 metres while preserving the
local role of diffusion, then measure how well the current checkpoint handles
different entry poses around the single object and grasp used for training.

## Model Scope

The 24-value diffusion condition includes hand choice, object-relative grasp,
approach direction, object dimensions, object/table height, and the live
object-local entry position, yaw, and velocity. The current training pack uses
one object and affordance, so the object/grasp channels are effectively
constant. This checkpoint must therefore be described as entry-conditioned for
the trained object, not as object-general.

## Range Separation

Learned pickup has two deliberately different spatial ranges:

- The activation and collision-checked coarse route may be up to 3.0 metres.
- The live diffusion handoff remains inside the 0.45-to-1.00-metre capture
  annulus.

Starting outside the annulus selects a collision-free capture root on the
annulus and uses ordinary locomotion to reach it. Diffusion is requested only
after the live root is in the annulus, within the existing yaw tolerance, and
settled below the existing speed threshold. The exact live root and velocity
remain the entry condition; the character must never return to a training
start or a stale entry waypoint.

An explicit 3.0-metre object-radius gate belongs to `FunnelCaptureConfig`.
The existing frozen-route revalidation continues to check table and obstacle
clearance without reapplying an activation-distance test. Authored Smart Pickup
and the final interaction matcher's 1.0-metre approach limit do not change.

## Generalization Sweep

The evaluation uses the object/grasp prefix from the current training pack and
varies only live-entry values 18 through 23. It covers:

- eight object-relative bearings at 45-degree intervals;
- radii 0.55, 0.75, and 0.95 metres, all inside the handoff annulus;
- yaw offsets of -15, 0, and +15 degrees from facing the object; and
- zero settled entry velocity.

This produces 72 deterministic conditions. Each condition samples the normal
batch of 32 proposals from the pinned checkpoint and runs the same geometric
projection and continuity certification as the production worker.

The machine-readable report records each condition, accepted proposal count,
entry-boundary error, route endpoint statistics, route diversity, and failure
reason. The summary reports pass count and worst cases. A condition passes the
proposal stage only when at least one proposal is accepted and every proposal
starts bit-exactly at its conditioned entry.

This sweep measures proposal-stage entry generalization. It does not claim
object generalization or full playable pickup success. Representative live
attempts remain necessary to verify coarse locomotion, matcher preview,
contact, and attachment.

## Visualization

The existing learned overlay remains authoritative for live testing: frozen
entry, selected route, monotonic progress, lookahead, terminal root, tracked
root, learned state, and attachment state. After the range change, restart the
learned controller on the same trained object so a user can test pickup from
several bearings and distances up to 3.0 metres.

## Verification

Automated tests must prove:

- the default learned capture accepts a root exactly 3.0 metres from the object;
- a root beyond 3.0 metres from the object is rejected even if its connector
  to the annulus would fit inside the route budget;
- the annulus still clamps the capture target to at most 1.0 metre;
- authored pickup and matcher distance defaults remain 1.0 metre;
- the sweep enumerates exactly 72 unique conditions;
- entry samples remain bit-exact after sampling/projection;
- deterministic input and seed produce a deterministic report; and
- focused C++ and Python regressions pass before restarting the visualizer.

## Non-Goals

- Generalization to unseen object geometry or grasps.
- Long-range diffusion navigation.
- Changing the final motion matcher, contact, or attachment authority.
- Training a new checkpoint as part of this change.
- Claiming success from synthetic proposals alone.
