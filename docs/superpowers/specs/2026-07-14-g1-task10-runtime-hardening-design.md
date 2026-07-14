# G1 Task 10 Runtime Hardening Design

**Date:** 2026-07-14
**Status:** Approved for test-first implementation under the user's unattended
auto-approval instruction

## Goal

Finish the remaining runtime-side Task 10 failures without changing the 25 Hz
runtime, the 31-dimensional matcher, terrain weights, IK/LMM feature gates, or
the protected motion artifacts. The runtime must:

- stop with measurable clearance before a blocked footprint;
- avoid advertising inaccessible obstacle tops to the matcher;
- prevent a toe over an unwalkable surface from lifting the character;
- retain periodic matching while suppressing settled-idle clip cycling; and
- leave traversable stairs, ramps, mixed-level motion, and active commands on
  the existing path.

## Evidence and root causes

### Settled-idle cycling

`database_search` already supports a transition cost, but the controller calls
it with the default zero. During a settled route tail, searches occur about
every three frames and accept any strictly lower alternative, producing exact
6- or 9-frame clip cycles. Active terrain transitions can improve the raw cost
by as little as approximately `0.009`, so a global margin would harm active
responsiveness.

An isolated probe applied a cost of `1.0` only when raw commanded planar speed
was at most `1e-4 m/s` and planar simulation speed was at most `0.05 m/s`.
All four observed stopped-tail loops disappeared, all six routes completed,
and weight-four terrain error remained lower than weight zero on every route.

### Safe-stop clearance

The command limiter reserves `0.02 m`, but its zero desired velocity feeds a
second-order simulation spring with a `0.27 s` halflife. The simulation coasts
after the applied command reaches zero. A static `0.04 m` reserve alone still
allowed residual dynamics; a hard stop with the old reserve stopped one sample
too late (`0.009615 m` clearance). Combining a `0.04 m` reserve with a planar
velocity/acceleration stop when the blocked applied speed reaches zero produced
`0.0384615 m` clearance on both blocked routes for all 600 test frames.

### Inaccessible terrain query and support

The four future terrain samples read heightfield geometry without considering
the walkability grid. At the wall, they therefore describe a `0.45 m` step even
though the route is required to stop. Merely using the current `blocked` flag
is insufficient because it clears after the scripted command completes while
the future trajectory still points through the obstacle.

A path-based isolated probe swept the footprint through the four ordered
centerline points. At the first unwalkable segment it froze the current and
remaining query heights at the last safe surface. Both blocked queries became
flat while traversable data remained eligible.

That probe exposed a second independent path: an inertialized toe can overhang
the wall and sample its top during support retargeting. Filtering the two
support-contact flags through the same walkability map eliminated the wall
lift. The complete probe passed both Gate D routes with `0.0384615 m` minimum
clearance and zero meaningful post-block support rise.

## Design

### 1. Stateless settled-idle transition cost

Add a pure helper in `g1_controller_state.h`:

```cpp
float g1_idle_match_transition_cost(
    float command_speed,
    float planar_simulation_speed);
```

It returns exactly `1.0f` when both inputs are nonnegative and at or below
`1e-4f` and `0.05f`, respectively. It returns exactly `0.0f` otherwise,
including negative, NaN, or infinite inputs.

The controller computes it immediately before the ordinary
`database_search`, using:

- `traversal.commanded_speed`, not applied or limited speed, so a character
  actively pushing against a blocker still uses active matching; and
- `walkability_xz_length(state.simulation_velocity)`, so vertical velocity is
  irrelevant.

The value is passed through the existing `database_search` transition-cost
argument. Search cadence, end-of-animation recovery, queries, feature weights,
and controller state remain unchanged. The incumbent is unpenalized and only
alternative candidates receive the cost, which is the intended hysteresis.

### 2. Safe-stop reserve and planar dynamics stop

Name the command-side blocked reserve `0.04f` in `terrain_runtime.h`. This is a
product safety margin above the unchanged Gate D minimum of `0.02 m`, not a
checker threshold change.

Add a pure helper that receives `traversability_diagnostics`, simulation
velocity, and simulation acceleration. When diagnostics are blocked and
`applied_speed <= 1e-4f`, it writes positive zero only to X and Z velocity and
acceleration. Otherwise it is a bit-preserving no-op. It never reads or writes
position, rotation, support, or any Y component.

Call it after command limiting and before trajectory prediction. The existing
post-integration preflight/clip remains the final collision guard.

### 3. Walkability-aware centerline snapshot

Keep `terrain_centerline_snapshot_compute_v2` unchanged. Add a new v2
post-processing helper that also receives a matching `walkability_grid`, the
animation query root, the authoritative simulation-footprint origin, and a
footprint radius. The controller computes the ordinary snapshot first and then
applies this helper.

The controller first computes the ordinary four-point snapshot. A transactional
`terrain_centerline_snapshot_apply_walkability_v2` helper then sweeps in
polyline order:

1. simulation-footprint origin to sample 0;
2. sample 0 to sample 1;
3. sample 1 to sample 2;
4. sample 2 to sample 3.

The animation query root remains the base-height reference for all four 31D
features; the separate simulation origin is used only for collision
reachability. This preserves the existing matching stage while making the
sweep agree with the runtime collision guard. The radius is the controller's
existing `0.20 m` footprint. Clear segments retain the ordinary snapshot
bit-for-bit. On the
first blocked or out-of-bounds segment, the helper samples the height at the
sweep's last-safe point, latches that safe point and height, and repeats them
for that and all later snapshot entries. Repeating the point prevents a later
sample from appearing reachable through a narrow obstacle and makes the visual
marker describe the effective query. Values remain relative to the animation
query root's base height. If the starting footprint is invalid, all four
samples hold the query-root surface level.

This ordered latch represents the reachable terrain along the predicted path:
it does not let later samples “see through” a wall, but it preserves all class
1 and class 2 stairs, ramps, landings, and stress terrain.

Invalid/mismatched inputs return `false` without partially modifying the
snapshot; the controller reports a controlled runtime error. The controller
applies this helper before finite validation and before copying the
authoritative final four query features and visualization markers. No database
or log schema changes.

### 4. Walkability-filtered support contacts

Keep the generic `support_observation_build` behavior available. Add a wrapper
that builds the same source/runtime heights and deltas, then clears each toe's
contact flag when `walkability_class_at` for that toe is zero. Class 1 and class
2 contacts remain valid. Invalid/mismatched grids fail closed for toe contact.

The controller uses the wrapper. The support state then naturally selects the
other valid toe, holds briefly, or falls back to the safe root surface. Raw
terrain samples remain in the observation for diagnostics, while the filtered
contact bits are the bits published in the runtime snapshot. No support-height
clamp or special blocked-scene branch is added.

## Alternatives rejected

- Skipping all idle searches could freeze a poor incumbent and changes search
  cadence and end-of-range behavior.
- A global transition margin would suppress legitimate active improvements.
- Masking only while `state.blocked` misses stopped route tails after the flag
  clears.
- Replacing all future samples with the current root height would erase
  reachable ramps before a later blocker.
- Holding support whenever the root is blocked does not reject an individual
  toe over an inaccessible surface and hides the actual contact-level cause.
- Raising the Gate D clearance/support thresholds would preserve the user-
  visible defect rather than fixing it.

## Verification contract

Focused tests must prove:

- idle helper exact boundaries, invalid-input behavior, active-command
  fail-open behavior, controller wiring, and existing database hysteresis;
- safe-stop helper activation/no-op behavior and exact Y-bit preservation;
- clear class-1 and class-2 walkability produces a bit-identical ordinary
  centerline snapshot;
- a blocked elevated region preserves samples before the blocker and latches
  the last-safe point/height for every later sample;
- invalid filter inputs fail transactionally and controller ordering remains
  raw snapshot, walkability filter, finite validation, then 31D query copy;
- class 1/2 support contacts are unchanged, class 0 contacts are cleared, the
  other toe remains usable, and invalid grids cannot create support;
- all existing terrain, support, controller-state, and matcher tests pass in
  debug, optimized, and sanitizer configurations;
- both 600-frame Gate D routes pass unchanged acceptance thresholds;
- all six Gate C routes retain traversable terrain behavior and weight-four
  improvement; and
- protected artifact hashes and the old visualizer PID remain unchanged until
  the final certified replacement.
