# Onscreen Fall Freeze Design

**Goal:** Keep the terrain and final robot pose visible after a fall in manual
onscreen SONIC runs, instead of allowing the official simulator's automatic
reset to rewind MuJoCo time and tear down the viewer.

## Scope

The behavior is diagnostic only. `mm_sonic.manual_demo` enables it when
`--onscreen` is selected. Headless, scored, and other simulator clients retain
the current reset behavior. The pinned GEAR checkout remains byte-for-byte
unmodified.

## Design

Add an explicit `freeze_on_fall` option from `GatedSimulatorClient` through the
`mm_sonic.gated_sim` command line to `ExternalGearBackend`. The option is a
strict boolean and is rejected unless `onscreen` is also true.

When enabled, the adapter replaces only the live simulator instance's fall
callback. A pelvis height below the official 0.2 m threshold latches the first
fallen pose, logs one diagnostic to stderr, and does not call the upstream
`reset()`. The adapter zeros generalized velocity, acceleration, and controls
without changing the fallen `qpos`.

After the latch, each requested step:

1. verifies that the passive viewer is still running;
2. republishes a LowState sampled from the frozen pose so GEAR's watchdog stays
   connected;
3. advances `data.time` by exactly one `sim_dt` without calling `mj_step`;
4. synchronizes the viewer exactly once.

This keeps the existing JSONL step-count contract valid while making the
diagnostic nature explicit: simulated clock time advances, but physics is
frozen. Camera controls continue to work, and X performs the existing orderly
shutdown.

## Failure handling

- A closed viewer remains a protocol error.
- Missing official simulator fields needed to publish the frozen LowState fail
  closed with `ProtocolError`.
- The opt-in flag never changes headless behavior.
- The adapter logs the fall once; it does not repeatedly print or reset.

## Verification

- Unit-test strict flag validation and command propagation.
- Unit-test that default/headless stepping retains the existing behavior.
- Unit-test that an onscreen fall preserves qpos, prevents reset, zeros dynamic
  state, republishes LowState, advances the clock exactly, and syncs once per
  frozen step.
- Run the focused gated-simulator, process, and manual-demo suites warning
  strictly, followed by the full Python suite.
- Relaunch the terrain-aware `holden-turn-v1` demo and confirm a fall leaves the
  terrain and final pose visible until X.

