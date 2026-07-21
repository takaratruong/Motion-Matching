# Moving-Entry Blocking Funnel Generation

Date: 2026-07-21

Status: approved for implementation

## Goal

Replace the visible walk-to-idle-to-funnel sequence with a moving entry whose
pose is held unchanged while the current proposal worker runs. Proposal speed
optimization is deferred.

## Grasp-Centric Scope

The learned behavior is organized around the selected grasp pose, approach
direction, and live entry pose/velocity. The object transform remains the
coordinate frame and object dimensions remain safety/context inputs, but this
checkpoint does not claim object-identity generalization.

## Handoff

On the first controller tick where the live root is inside the inclusive
0.45-to-1.00-metre annulus and within the existing facing tolerance, freeze
that exact moving root and simulation velocity as the 24-value condition.
Do not require low speed and do not accumulate settling ticks.

Launch the proposal worker and wait for it in that same controller update.
The controller update blocks, so locomotion physics and animation do not
advance while generation is pending. The OS window may appear held for the
worker duration, but the character does not transition into an idle clip.
After generation, the existing preview and spatial follower resume from the
same displayed root. Cancellation can be processed after the blocking worker
returns.

## Sampling

Production already uses 50 DDIM inference steps over a checkpoint trained with
1,000 diffusion timesteps. Do not change inference to 100 steps: that would
double denoiser evaluations. Persistent model loading and lower-step quality
experiments are separate future performance work.

## Verification

- A moving annulus entry above 0.1 m/s launches on its first valid tick.
- Condition values 22 and 23 preserve the moving simulation velocity.
- The production Python provider exposes a blocking wait while its existing
  poll remains non-blocking.
- Learned backend uses blocking wait immediately after launch and returns
  proposal previews without a capture-settle state.
- Worker failure remains fail-closed and attempt files are cleaned.
- Existing funnel backend, end-to-end Carry, and worker tests remain green.

## Non-Goals

- Rendering or accepting input during proposal generation.
- Changing checkpoint training timesteps or production DDIM step count.
- Persistent Python/CUDA workers.
- Prefetching or predicting an entry pose.
