# G1 SONIC Onscreen Viewer Design

## Goal

Let an operator see and drive the already-qualified G1 SONIC MuJoCo run on the
active desktop. The onscreen path must use the same scene, physics state,
controller, rolling motion-matching reference, and terminal controls as the
qualified headless path.

## Scope

- Add an explicit, opt-in onscreen flag from the manual demo through the gated
  simulator client to the simulator child.
- Use the official GEAR `BaseSimulator` passive MuJoCo viewer.
- Synchronize that viewer while the existing gated simulator advances physics.
- Update `/home/ubuntu/drive-g1-sonic.sh` to request the viewer and use desktop
  OpenGL instead of EGL.
- Preserve all current headless defaults and evidence behavior unchanged.

Browser streaming, a graphical command panel, controller remapping, and changes
to motion generation or tracking are outside this change.

## Design

`manual_demo` accepts `--onscreen` and passes it to `GatedSimulatorClient`.
The client adds `--onscreen` only when requested. The gated simulator passes the
boolean to `ExternalGearBackend`, which constructs the pinned official
`BaseSimulator` with `onscreen=True` and `offscreen=False`.

During `ExternalGearBackend.step`, physics remains authoritative and advances
exactly once through the existing `sim_step()` call. When a viewer exists, the
backend calls the official environment's `update_viewer()` after the step. This
only synchronizes presentation; it does not advance physics or change controls.

The convenience launcher selects GLFW, preserves `DISPLAY=:1` and the current
X authority, and passes `--onscreen`. Direct library and evidence invocations
remain headless unless they explicitly opt in.

## Operator Flow

1. Run `/home/ubuntu/drive-g1-sonic.sh`.
2. Wait for the MuJoCo window and the terminal `LIVE` message.
3. Keep the launcher terminal focused for W/S/A/D, Q/E, space, and X commands.
4. Observe the same controlled MuJoCo state in the passive viewer.
5. Press X or close the viewer; cleanup must terminate GEAR and simulator
   children without leaving processes behind.

## Failure Handling

- If the desktop/OpenGL viewer cannot start, the simulator exits visibly and
  the launcher reports the failure; it must not silently fall back to headless.
- Closing the viewer must not create a second physics loop or leave child
  processes alive.
- Protocol stdout remains JSONL-only. Viewer diagnostics continue on stderr.
- The qualified headless mode remains the fallback command for automated runs,
  not an implicit fallback inside onscreen mode.

## Verification

- Unit tests prove the onscreen flag is absent by default and forwarded only
  when requested.
- Backend tests prove `onscreen` reaches `BaseSimulator` and viewer sync occurs
  after, not instead of, the single physics step.
- Existing gated-simulator and manual-demo tests remain green.
- A desktop smoke run must visibly open the MuJoCo window, reach `LIVE`, accept
  at least one movement command, exit through X, and leave no related process.
