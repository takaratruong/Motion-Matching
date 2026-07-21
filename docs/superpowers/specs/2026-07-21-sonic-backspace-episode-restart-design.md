# Sonic Backspace Episode Restart Design

## Goal

Add an operator-controlled recovery path to the visible terrain-aware Sonic
demo. A rising Backspace key edge ends the current episode and automatically
starts a completely fresh episode at the registered terrain start. The Motion
Matching session, streamed reference, GEAR controller process, simulator state,
and input state must restart together.

This is a diagnostic and iteration feature. It does not attempt to make the
current policy recover from a fall.

## Operator Contract

- Backspace requests a full episode restart only after `LIVE X11` readiness.
- The current MuJoCo window closes during teardown. A new window opens on the
  same registered terrain after normal startup, expected to take roughly
  20–30 seconds on this cluster.
- The new episode begins from the authenticated MM initial boundary and its
  corresponding simulator `qpos`; it does not reuse the fallen pose.
- The operator must focus the newly opened MuJoCo window before control becomes
  live, using the existing focus-readiness barrier.
- `X` continues to exit the demonstration completely. Backspace never aliases
  exit.
- Holding Backspace generates exactly one restart request. A new request
  requires release followed by another press.
- Backspace is passively grabbed with the other Sonic command keys so MuJoCo
  cannot consume it as a viewer shortcut.

## Architecture

### Input edge

`operator_x11.py` adds the exact Backspace keysym to the frozen key registry,
action registry, and passive-grab set. `ContinuousControlLoop` accepts a
dedicated `restart_event`. On the rising Backspace edge it sets that event and
emits `KEY BACKSPACE DOWN -> restart`; release emits the matching UP event.

Backspace is excluded from `NormalizedControlState`. It must not change
velocity, heading, camera, stand, or terminate values. The restart event is
orthogonal to the Motion Matching command mailbox.

### Episode boundary

The responsive X11 loop checks `restart_event` at every synchronized control
boundary, including while `freeze_on_fall` is servicing frozen simulator
steps. Once observed, it stops producing additional MM candidates and returns
an explicit restart outcome to `run_demo`.

`run_demo` records the interrupted outcome and unwinds through its existing
`finally` cleanup. Cleanup must reap the active GEAR process group, close the
gated simulator and its viewer, close the pose publisher and MM server client,
release all X11 grabs, and join the control thread before another episode is
created.

The CLI `main` function owns the episode supervisor loop. A restart outcome
causes `main` to call `run_demo` again with the same validated CLI namespace.
Every call creates a new unique `RunBundle`, MM session identifier, publisher,
GEAR child, and simulator child. A normal completion or `X` exit leaves the
supervisor loop. Startup, protocol, validation, and cleanup failures remain
fatal and must not be converted into automatic retries.

### Evidence

Each episode remains isolated in its own existing timestamp-and-PID run
directory. Before teardown, a restarted episode writes an
`episode-outcome.json` containing:

- schema identity and episode ordinal;
- outcome `operator_restart`;
- committed/generated chunk counts;
- current simulator time from the existing snapshot contract; and
- the triggering key `BACKSPACE`.

The replacement episode writes to a new directory and never appends to the
interrupted episode's command, target, state, contact, or responsive-trace
files. Existing successful summary schemas remain unchanged.

## Lifecycle and Failure Handling

The restart event is accepted only from a healthy, focused X11 control loop.
Provider failures and cancellation races retain the existing fail-closed
behavior. If teardown fails, the supervisor does not start another episode;
the cleanup error is surfaced so two GEAR or simulator processes cannot overlap.

The restart event is cleared by construction because every episode creates a
new event and input loop. No policy, mailbox revision, camera sequence, MM
candidate ID, frame index, or GEAR state crosses the episode boundary.

Repeated Backspace presses during teardown are ignored because the X11 provider
and its grabs are already closing. Backspace during pre-live startup is outside
this contract.

## Testing

Unit tests will establish:

- Backspace has the exact X11 keysym, action text, and passive-grab coverage.
- A rising edge sets only `restart_event`; holding does not create repeated
  edges, and release permits a later restart.
- Backspace leaves the normalized locomotion/camera command neutral and does
  not set the exit cancellation event.
- The responsive loop returns restart at a boundary without generating another
  MM candidate.
- `run_demo` records the interrupted outcome and executes all cleanup paths.
- `main` relaunches after restart, uses a fresh run/session identity, and exits
  normally after the replacement episode.
- A teardown or startup failure is surfaced without relaunch.

The protected X11, manual-demo, responsive scheduler/wiring, gated-simulator,
and viewer-isolation suites must remain green. A visible canary will deliberately
fall or use the already frozen pose, press Backspace, observe clean process
replacement and a newly focused terrain scene at the start pose, then verify
WASD and `X` in the replacement episode.

## Non-goals

- Hot-resetting GEAR while preserving the same MuJoCo window.
- Resetting only simulator `qpos` while retaining policy or MM history.
- Automatically restarting on every fall.
- Improving obstacle clearance or Sonic tracking quality in this change.
- Treating an unexpected child-process failure as an operator restart.
