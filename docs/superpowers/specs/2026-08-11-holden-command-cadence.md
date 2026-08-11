# Holden Command Cadence Design

## Problem

The full-walking diagnostic maps arrow/WASD signs correctly and places root
height from the authenticated ramp authority correctly. Its control loop does
not, however, preserve the normal Holden matcher behavior:

- diagnostic mode searches once on a command edge and then follows authored
  successors indefinitely;
- slope ranges are about ten seconds long and commonly turn through roughly
  154 degrees, so held Up can eventually follow the source back downhill;
- key release becomes an immediate zero command and the diagnostic freezes the
  current state instead of decelerating into an idle pose.

## Chosen design

Reuse the default Holden/Torch control primitive rather than introduce another
controller. `HybridMatcher` will persist a robot-local planar shaped velocity
and advance it with `torch_motion_matcher.bounded_velocity_step` using the
existing `MatcherConfig` defaults:

- acceleration: 1.5 m/s^2;
- deceleration: 2.0 m/s^2;
- stop threshold: 0.05 m/s;
- active search cadence: 0.10 seconds.

The requested target remains `CommandState.speed * walking_speed_p95_mps` in
the existing local-forward direction. Steering remains the existing yaw-rate
command; this change does not alter arrow signs or coordinate conventions.

The shaped velocity feeds both the trajectory query and terrain preview. After
key release, the shaped speed therefore decays over successive fixed-rate
ticks, continues periodic matching while it is above the stop threshold, then
becomes exact zero and enters the existing neutral hold. This avoids using the
display inertializer as a substitute for command dynamics.

## Scope and isolation

The behavior is enabled only for the interactive diagnostic single-GPU path.
The CPU diagnostic retains event-only search to avoid multi-second input stalls,
and formal/default evaluation remains byte-for-byte on its existing CPU search
semantics. The full viewer exposes Space separately as a hard-stop level; it
zeros the shaped velocity and holds immediately rather than decelerating.

The shaped velocity participates in the matcher's transactional snapshot. A
failed candidate transaction restores it alongside root placement, command,
terrain class, and source authority. Reset zeros it.

Runtime identity and the visible diagnostic receipt report the controller name,
acceleration, deceleration, stop threshold, and 0.10-second cadence. GPU search
continues to be diagnostic/non-acceptance evidence.

## Tests

Tests must first fail on the current implementation and then prove:

1. a held active GPU diagnostic command searches again after 0.10 seconds;
2. CPU diagnostic and formal/default cadence are unchanged;
3. release from motion decelerates by the exact default 2.0 m/s^2 bound, keeps
   advancing/searching during the residual-speed interval, and then holds;
4. query and terrain preview receive the same shaped speed;
5. rollback and reset restore/zero shaped velocity;
6. Space is an immediate stop while ordinary release is smoothed;
7. runtime identity reports the exact reused-control parameters.

No corpus, model, scene, terrain encoding, pose postprocessor, or arrow-sign
change is part of this fix.
