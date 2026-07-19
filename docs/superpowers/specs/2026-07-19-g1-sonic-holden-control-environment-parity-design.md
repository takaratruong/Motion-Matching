# G1 SONIC Holden Control and Environment Parity Design

**Date:** 2026-07-19

**Status:** Approved conversational design, awaiting written-spec review

**Implementation branch:** `research/g1-low-latency-driver`

## Purpose

Make the interactive Motion Matching to SONIC demonstration use the same
operator command space and the same registered terrain scene as the playable
Holden runtime. The result must remain an unmodified SONIC policy driving the
G1 model through ordinary MuJoCo physics.

The first proof is deliberately visual and short: show that the robot and
terrain agree kinematically, then show the robot standing and moving on that
terrain in physics, and finally hand the running controls to the operator.
Formal regression and evidence gates follow those visual checks.

## Selected approach

Reuse the existing authenticated SONIC scene adapter and add a continuous
desktop control frontend behind a device-independent control-state contract.

The scene adapter already authenticates the published Holden heightfield,
walkability mask, and OBJ; applies the committed Holden-to-MuJoCo transform;
and installs the transformed OBJ as MuJoCo's visual and collision mesh. The
manual driver currently bypasses this path by copying a previously generated
flat scene. Connecting the existing adapter is both faster and safer than
creating a second terrain conversion.

The desktop frontend samples actual key-down and key-up state while the MuJoCo
viewer or the explicitly titled G1 control terminal has focus. It does not
infer holds from terminal autorepeat. A normalized two-stick control state
keeps keyboard handling separate from the Holden command equations and leaves
one stable seam for a later physical gamepad adapter.

Two alternatives are rejected:

- Injecting an OBJ into the copied flat XML would be visually quick, but could
  diverge from the heightfield and walkability artifacts used by Motion
  Matching.
- Piping poses from the rendered Holden application into MuJoCo would duplicate
  runtime ownership and would not test the intended Motion Matching to SONIC
  process boundary.

## Scope

### Goals

- Select any published terrain scene and route at process startup.
- Default the interactive launcher to `grail-curb-default`, route
  `curb-forward`, and terrain feature weight `4.0`.
- Reset Motion Matching and authenticate the selected scene before MuJoCo can
  advance.
- Give Motion Matching and MuJoCo the same scene identity and geometry.
- Initialize the simulated G1 from the reset boundary's named joint pose and
  physical pelvis pose rather than from an unrelated flat-model `qpos0`.
- Match the playable Holden keyboard control semantics, including continuous
  holds, releases, modifiers, camera-relative travel, independent strafe
  heading, walk/run speeds, and camera zoom.
- Print input transitions and the exact boundary command they produce.
- Preserve neutral closed hand targets.
- Preserve the existing explicit flat scene as a selectable diagnostic mode.
- Produce a short terrain-alignment check, kinematic preview, physics video,
  and interactive run before spending time on the full formal gate.

### Non-goals

- Changing the SONIC checkpoint, observation space, policy actions, or control
  frequency.
- Training or fine-tuning SONIC.
- Adding depth sensing, noisy terrain estimation, or sim-to-real transfer.
- Switching terrain scenes during one running simulator process.
- Feeding the Motion Matching virtual root back from MuJoCo.
- Qualifying a physical gamepad device in this increment. A later adapter may
  populate the same normalized control state without changing command logic.
- Removing the rolling reference buffer. Operator commands still latch at
  chunk boundaries and the UI must report the configured lookahead.

## Architecture

### 1. Startup and scene ownership

The manual driver changes from flat-scene-first startup to the following
transaction:

1. Create an immutable run bundle and start the MM chunk server.
2. Send `hello`, then reset it with the selected scene ID, route ID, and
   terrain weight.
3. Pass the returned artifact and scene identities to `register_scene`.
4. Have `register_scene` authenticate the scene catalog, heightfield,
   walkability mask, and OBJ, then create the run-local MuJoCo overlay.
5. Build the initial MuJoCo `qpos` from the validated MM reset boundary.
6. Start the unmodified GEAR-SONIC process and gated MuJoCo simulator with that
   overlay and initial state.
7. Permit physics to advance only after the existing reference-stream
   readiness sequence succeeds.

The official GEAR scene and robot hashes use the same pinned constants as the
existing Stage A scene gate. Those constants move to one shared configuration
module rather than being copied into the manual driver.

`register_scene` remains the only terrain conversion authority. For terrain
scenes, the generated MuJoCo model contains exactly one `mm_terrain` mesh geom
whose bytes derive from the authenticated published OBJ. The authenticated
G1HF heightfield and walkability hashes remain part of the identity supplied by
the MM reset response. For the explicit flat diagnostic scene,
`mm_terrain` remains the registered plane.

### 2. Initial physical state

The reset boundary is validated before it can seed physics. Its 29 named joint
positions are mapped through the committed SONIC joint contract. Its physical
pelvis position and orientation are transformed from Holden coordinates into
MuJoCo coordinates using the same proper rotation as the scene mesh. The free
joint and named joint addresses are resolved from the loaded model; positional
index assumptions are forbidden. Neutral closed hand values fill the hand
joints not present in the 29-DoF reference.

The constructed state must pass MuJoCo forward kinematics and these checks
before simulator launch:

- all values are finite and the pelvis quaternion is unit length;
- every named target joint resolves exactly once and remains within its joint
  limits;
- the root lies inside the registered scene domain;
- the pelvis and feet have finite local-terrain clearance;
- no forbidden body geom penetrates terrain beyond 0.005 metres.

The run records both the reset-boundary hash and initial-`qpos` hash so a video
cannot be mistaken for a run initialized from another pose.

### 3. Normalized control state

Device adapters publish immutable snapshots containing:

- left-stick X/Z axes for travel;
- right-stick X/Z axes for camera or strafe heading;
- `strafe`, `walk`, `stand`, and `terminate` booleans;
- a signed zoom axis;
- monotonic sample time, focus state, and a sequence number.

The desktop adapter samples at 50 Hz. Direction, modifier, and zoom values are
level state. Space and X rising edges are retained until the coordinator
acknowledges them, so a tap between chunk boundaries cannot disappear. X also
sets the shared cancellation event immediately; Space is consumed by the next
command boundary.

The keyboard adapter maps keys exactly as the Holden runtime:

| Input | Meaning |
| --- | --- |
| W / S | left-stick forward / backward |
| A / D | left-stick left / right |
| Arrow keys | right stick: orbit camera, or heading while strafing |
| Left Ctrl | strafe mode |
| Left Shift | walk gait; released selects run speeds |
| Q / E | zoom out / in |
| Space | stand and clear travel for the next command boundary |
| X | terminate the interactive run cleanly |

Opposed digital directions cancel. Diagonal input is normalized to unit
magnitude. The normalized schema uses the same axis convention and 0.2
deadzone/squared-magnitude rule as the playable runtime, even though digital
keys normally yield magnitude one.

The default Linux desktop adapter reads key state through X11 without adding a
third-party Python package. It accepts input only while the MuJoCo viewer or
the explicitly titled G1 control terminal is the active application. Losing
focus immediately publishes a neutral snapshot. A non-X11 session fails with
an actionable message and may opt into the existing terminal-character
compatibility mode; that compatibility mode is never described as continuous
Holden parity.

Every key transition prints once, for example
`KEY LEFT_CTRL DOWN -> strafe`. Every latched command also prints its chunk,
velocity, heading, strafe state, gait state, and estimated presentation time.
Autorepeat cannot create duplicate transitions.

### 4. Holden command mapper and camera

A pure command mapper ports the equations and constants from the playable
runtime. It owns camera azimuth, altitude, distance, retained desired heading,
and the smoothed walk/run blend. It is independent of X11, MuJoCo, ZMQ, and
terminal I/O.

At the desktop sampling rate:

- when not strafing, the right stick changes orbit-camera azimuth and altitude;
- Q/E changes camera distance;
- left-stick travel is rotated by camera azimuth into world space;
- forward, side, and backward speeds use the Holden run values
  `0.9/0.6/0.6 m/s` and walk values `0.5/0.4/0.4 m/s`;
- Left Shift drives the same 0.1-second-halflife walk/run blend;
- without strafe, nonzero travel makes desired heading follow travel;
- with strafe, desired heading follows the camera-forward direction when the
  right stick is neutral, otherwise the camera-relative right-stick direction;
- zero travel without a new strafe heading retains the preceding heading.

The mapper produces independent world velocity and desired heading in the
existing `CommandSample` contract. No terrain code owns or rewrites heading.
The MM runtime may reduce unsafe translation through its existing
walkability/traversability logic, but the requested heading remains the
operator's heading.

Camera state is sent through a small gated-simulator protocol operation and
applied to the existing passive MuJoCo viewer handle. Viewer-camera updates do
not advance physics and do not modify robot state. The simulator acknowledges
the applied camera sequence so the UI can distinguish command input from a
stale display.

### 5. Manual-driver and artifact changes

The interactive CLI gains explicit `--scene-id`, `--route-id`,
`--terrain-weight`, and `--input-source` options. `--input-source x11` is the
interactive desktop default and `--input-source terminal` is the explicit
compatibility fallback. The desktop launcher names the active scene and
control focus target before startup, and defaults to two preloaded 0.4-second
chunks so its existing qualified 0.8-second lookahead is unchanged. Input and
camera state can update at 50 Hz, while reference commands still take effect
only at reported chunk presentation boundaries.

The manual summary schema advances by one version and records:

- selected scene, route, and terrain weight;
- MM scene and artifact identities;
- source heightfield, OBJ, walkability, and scene-index hashes;
- transformed OBJ, generated scene, generated robot, and registration hashes;
- initial boundary and initial `qpos` hashes;
- normalized input source and control-mapper version;
- command lookahead and all latched commands;
- neutral closed-hand targets;
- terminal simulator snapshot and video/evidence paths when produced.

The existing flat manual command artifact remains readable. New artifacts are
strictly versioned and fail closed on missing, duplicate, or unknown identity
fields.

## Failure and safety behavior

- Unknown scenes or routes, hash disagreement, malformed terrain, invalid MM
  reset state, model-load failure, or initial collision failure terminates
  before physics starts.
- Focus loss, input-source disconnect, or a stale snapshot produces neutral
  input. Already published lookahead cannot be retracted, so the UI prints the
  remaining queued duration rather than claiming an immediate stop.
- Space explicitly queues stand while retaining heading. X closes the rolling
  loop, simulator, GEAR process group, publisher, and MM server.
- Closing the MuJoCo viewer terminates the run instead of continuing unseen.
- A camera acknowledgement mismatch stops camera updates and reports the
  mismatch; it cannot corrupt the physics command stream.
- The flat diagnostic launcher remains available as a rollback path.

## Visual-first qualification

Qualification proceeds in this order, stopping at the first bad visual result:

1. Load `grail-curb-default` and capture a screenshot showing the transformed
   terrain, G1 spawn, and camera framing before commanded travel.
2. Replay the first short MM reference kinematically in the same registered
   MuJoCo model and inspect feet, pelvis, direction, and terrain contacts.
3. Run a short physics stand and forward-walk clip with neutral closed hands.
4. Run a short physics clip containing forward, lateral strafe, heading change,
   and stand; render command text and core stability statistics into the
   evidence.
5. Start the persistent interactive launcher and verify, on screen, key-down,
   key-up, latched command, camera, strafe, and stand behavior while the MuJoCo
   window has focus.

A clip is rejected immediately for obvious coordinate mismatch, wrong spawn,
terrain/robot scale mismatch, floating or buried feet, unsupported flailing,
or travel opposite the displayed command. It does not proceed merely because a
numeric test passes.

## Formal verification

After the visual checks pass, focused automated coverage includes:

- golden Holden control-mapper cases for forward, backward, lateral, diagonal,
  camera-relative travel, strafe heading, retained heading, walk/run speeds,
  and zoom limits;
- key-state transition, focus-loss, opposed-key, and stale-input tests using an
  injected desktop snapshot source;
- exact scene/route/terrain-weight propagation from CLI to MM reset and scene
  registration;
- reset-boundary-to-MuJoCo initial-state basis, quaternion, named-joint, hand,
  limit, and collision tests;
- generated terrain geom and scene hash parity;
- gated-simulator camera request/acknowledgement protocol tests;
- manual-summary schema and tamper-rejection tests;
- flat-scene compatibility tests;
- the existing focused operator, manual evidence, scene, simulator protocol,
  and frozen evidence suites.

The final physics evidence reports at minimum pelvis-height range, pelvis-up
minimum, path length, yaw change, final stop speed, forbidden contacts,
selected scene hashes, and exact video SHA-256. The user receives the video in
an easy-to-access home-directory path and a running launcher for direct control.

## Delivery order

1. Add the pure normalized control state and Holden command mapper.
2. Add focus-gated continuous desktop input and observable transition logging.
3. Add camera state to the gated simulator protocol.
4. Replace manual flat-scene copying with authenticated scene registration and
   reset-boundary initialization.
5. Update artifacts and the desktop launcher.
6. Perform the visual-first terrain checks and correct visible integration
   errors immediately.
7. Run focused and frozen formal verification.
8. Produce the physics video and launch the interactive terrain driver.
