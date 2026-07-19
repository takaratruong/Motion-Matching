# G1 SONIC Holden Control and Environment Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the interactive MM-to-SONIC G1 use Holden-equivalent continuous controls and the exact authenticated Holden terrain in unmodified MuJoCo physics.

**Architecture:** A pure Holden control mapper consumes device-independent snapshots; an X11 adapter supplies real key levels and latched edges; and the existing gated simulator receives authenticated scene and camera commands. The manual driver resets MM before registering the same terrain for MuJoCo, constructs initial physics state from the validated MM boundary, and preserves the flat driver as an explicit fallback.

**Tech Stack:** Python 3.11, NumPy, `ctypes`/libX11, MuJoCo Python bindings, JSONL process protocols, existing MM chunk server and GEAR-SONIC runtime, `unittest`.

## Global Constraints

- Work only in `/home/ubuntu/projects/motion-matching/.worktrees/g1-low-latency-driver` on `research/g1-low-latency-driver`.
- Preserve the unmodified SONIC checkpoint, observations, policy actions, and 50 Hz low-level controller.
- Do not modify the pinned GEAR checkout or any shared external runtime.
- Use `apply_patch` for all source and documentation edits.
- Use `python -B -m unittest`; neither qualified venv contains pytest.
- Run visual checks before the broad formal suite and reject obvious bad motion immediately.
- Keep neutral closed Dex3 hand targets on every published frame and in initial `qpos`.
- Default terrain is `grail-curb-default`, route `curb-forward`, terrain weight `4.0`.
- Keep the explicit `sonic-flat-baseline` / `flat-12s` / `0.0` path working.
- Desktop input samples at 50 Hz; reference commands still latch at reported 0.4-second chunk boundaries with two chunks of lookahead.
- Use the qualified Reliable Claude release at `/home/ubuntu/reliable-claude-release-canaries/releases/de2ec6d3c1d46d772369a3b447b287b1d1e72e2d/venv/bin/reliable-claude` for substantial delegated implementation, then independently inspect diffs and run every stated gate in Codex.

## File structure

- Create `sonic/python/mm_sonic/holden_control.py`: immutable normalized input, camera/output records, and pure Holden command equations.
- Create `sonic/python/mm_sonic/operator_x11.py`: libX11 key-level provider, focus gate, edge latch, 50 Hz reader, and boundary mailbox.
- Create `sonic/python/mm_sonic/scene_runtime.py`: shared pinned GEAR scene constants and reset-boundary-to-MuJoCo initial-state construction.
- Modify `sonic/python/mm_sonic/gated_sim.py`: camera application in the backend/runner and one JSONL operation.
- Modify `sonic/python/mm_sonic/process.py`: strict client-side camera request and acknowledgement.
- Modify `sonic/python/mm_sonic/scene.py`: retain transformed terrain bounds on `RegisteredScene`.
- Modify `sonic/python/mm_sonic/cli.py`: import shared scene constants/helpers without changing Stage A behavior.
- Modify `sonic/python/mm_sonic/manual_demo.py`: scene-first MM transaction, input selection, mapper, camera, and terrain-aware summaries.
- Modify `sonic/python/mm_sonic/manual_evidence.py`: versioned terrain summary parsing while retaining v3 flat artifacts.
- Create `tests/python/test_sonic_holden_control.py`: golden control semantics.
- Create `tests/python/test_sonic_operator_x11.py`: provider-independent continuous input tests and libX11 capability canary.
- Create `tests/python/test_sonic_scene_runtime.py`: initial state, bounds, hands, collision, and identity tests.
- Modify `tests/python/test_sonic_gated_sim.py` and `tests/python/test_sonic_process.py`: camera protocol tests.
- Modify `tests/python/test_sonic_scene.py`, `tests/python/test_sonic_manual_demo.py`, and `tests/python/test_sonic_manual_evidence.py`: terrain startup/artifact regressions.
- Modify `/home/ubuntu/drive-g1-sonic.sh`: easy terrain-aware interactive launcher.

---

### Task 1: Pure Holden control state and command mapper

**Files:**
- Create: `sonic/python/mm_sonic/holden_control.py`
- Create: `tests/python/test_sonic_holden_control.py`

**Interfaces:**
- Consumes: no runtime dependencies beyond `CommandSample` and `ContractError`.
- Produces: `NormalizedControlState`, `CameraState`, `MappedControlState`, and `HoldenControlMapper.update(state, dt_s) -> MappedControlState`.

- [ ] **Step 1: Write failing value-contract and golden-direction tests**

Create tests that assert frozen records, finite validation, W/A diagonal normalization, asymmetric forward/side/back speeds, camera-relative travel, strafe heading, retained heading, walk smoothing, and zoom clamps. The core cases are:

```python
class HoldenControlMapperTests(unittest.TestCase):
    def test_forward_and_left_match_mujoco_axes(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        forward = mapper.update(
            NormalizedControlState(left_z=-1.0), 0.02
        )
        self.assertAlmostEqual(forward.velocity_mujoco[0], 0.9, places=6)
        self.assertAlmostEqual(forward.velocity_mujoco[1], 0.0, places=6)
        left = mapper.update(
            NormalizedControlState(left_x=-1.0), 0.02
        )
        self.assertAlmostEqual(left.velocity_mujoco[0], 0.0, places=6)
        self.assertAlmostEqual(left.velocity_mujoco[1], 0.6, places=6)

    def test_camera_relative_forward_and_strafe_heading_are_independent(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=math.pi / 2.0)
        mapped = mapper.update(
            NormalizedControlState(
                left_z=-1.0,
                right_x=-1.0,
                strafe=True,
            ),
            0.02,
        )
        self.assertAlmostEqual(mapped.velocity_mujoco[0], 0.0, places=6)
        self.assertAlmostEqual(mapped.velocity_mujoco[1], 0.9, places=6)
        self.assertNotEqual(mapped.desired_heading_mujoco_wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_shift_converges_to_registered_walk_speeds(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        mapped = None
        for _ in range(100):
            mapped = mapper.update(
                NormalizedControlState(left_z=-1.0, walk=True), 0.02
            )
        assert mapped is not None
        self.assertAlmostEqual(mapped.velocity_mujoco[0], 0.5, places=4)

    def test_stand_zeroes_velocity_but_retains_heading(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.3)
        moving = mapper.update(NormalizedControlState(left_z=-1.0), 0.02)
        standing = mapper.update(
            NormalizedControlState(left_z=-1.0, stand=True), 0.02
        )
        self.assertEqual(standing.velocity_mujoco, (0.0, 0.0, 0.0))
        self.assertEqual(
            standing.desired_heading_mujoco_wxyz,
            moving.desired_heading_mujoco_wxyz,
        )
```

- [ ] **Step 2: Run the new tests and verify the module is absent**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_holden_control
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mm_sonic.holden_control'`.

- [ ] **Step 3: Implement the pure mapper**

Implement exact validated records and these public methods:

```python
@dataclass(frozen=True)
class NormalizedControlState:
    left_x: float = 0.0
    left_z: float = 0.0
    right_x: float = 0.0
    right_z: float = 0.0
    strafe: bool = False
    walk: bool = False
    zoom: float = 0.0
    stand: bool = False
    terminate: bool = False

@dataclass(frozen=True)
class CameraState:
    sequence: int
    azimuth_rad: float
    altitude_rad: float
    distance_m: float

@dataclass(frozen=True)
class MappedControlState:
    velocity_mujoco: tuple[float, float, float]
    desired_heading_mujoco_wxyz: tuple[float, float, float, float]
    camera: CameraState
    strafe: bool
    walk_blend: float
    stand: bool
    terminate: bool

class HoldenControlMapper:
    RUN_SPEEDS_MPS = (0.9, 0.6, 0.6)
    WALK_SPEEDS_MPS = (0.5, 0.4, 0.4)
    GAIT_HALFLIFE_S = 0.1
    CAMERA_RATE_RAD_S = 2.0
    ZOOM_RATE_M_S = 10.0
    MIN_DISTANCE_M = 0.1
    MAX_DISTANCE_M = 100.0

    def __init__(
        self,
        *,
        initial_heading_yaw_rad: float,
        initial_altitude_rad: float = 0.4,
        initial_distance_m: float = 3.0,
    ) -> None:
        values = (
            initial_heading_yaw_rad,
            initial_altitude_rad,
            initial_distance_m,
        )
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ContractError("Holden control initial state must be finite")
        self._desired_yaw = float(initial_heading_yaw_rad)
        self._camera_azimuth = float(initial_heading_yaw_rad)
        self._camera_altitude = min(max(float(initial_altitude_rad), 0.0), 0.4 * math.pi)
        self._camera_distance = min(max(float(initial_distance_m), 0.1), 100.0)
        self._camera_sequence = 0
        self._gait = 0.0
        self._gait_velocity = 0.0

    @staticmethod
    def _stick(x: float, z: float) -> tuple[float, float]:
        magnitude = math.hypot(x, z)
        if magnitude <= 0.2:
            return 0.0, 0.0
        clipped = min(1.0, magnitude * magnitude)
        return x * clipped / magnitude, z * clipped / magnitude

    def update(
        self,
        state: NormalizedControlState,
        dt_s: float,
    ) -> MappedControlState:
        if type(state) is not NormalizedControlState:
            raise ContractError("Holden mapper requires NormalizedControlState")
        if type(dt_s) not in (int, float) or not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ContractError("Holden mapper dt_s must be positive and finite")
        dt = float(dt_s)
        left_x, left_z = self._stick(state.left_x, state.left_z)
        right_x, right_z = self._stick(state.right_x, state.right_z)

        goal = 1.0 if state.walk else 0.0
        y = (4.0 * math.log(2.0) / (self.GAIT_HALFLIFE_S + 1.0e-5)) / 2.0
        j0 = self._gait - goal
        j1 = self._gait_velocity + j0 * y
        x = y * dt
        eydt = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        self._gait = eydt * (j0 + j1 * dt) + goal
        self._gait_velocity = eydt * (self._gait_velocity - j1 * y * dt)

        old_camera = (
            self._camera_azimuth,
            self._camera_altitude,
            self._camera_distance,
        )
        if not state.strafe:
            self._camera_azimuth = math.remainder(
                self._camera_azimuth - self.CAMERA_RATE_RAD_S * dt * right_x,
                2.0 * math.pi,
            )
            self._camera_altitude = min(
                max(
                    self._camera_altitude
                    + self.CAMERA_RATE_RAD_S * dt * right_z,
                    0.0,
                ),
                0.4 * math.pi,
            )
        self._camera_distance = min(
            max(
                self._camera_distance + self.ZOOM_RATE_M_S * dt * state.zoom,
                self.MIN_DISTANCE_M,
            ),
            self.MAX_DISTANCE_M,
        )
        if old_camera != (
            self._camera_azimuth,
            self._camera_altitude,
            self._camera_distance,
        ):
            self._camera_sequence += 1

        forward = -left_z
        left = -left_x
        ca = math.cos(self._camera_azimuth)
        sa = math.sin(self._camera_azimuth)
        world_forward = ca * forward - sa * left
        world_left = sa * forward + ca * left

        ch = math.cos(self._desired_yaw)
        sh = math.sin(self._desired_yaw)
        local_forward = ch * world_forward + sh * world_left
        local_left = -sh * world_forward + ch * world_left
        run_forward, run_side, run_back = self.RUN_SPEEDS_MPS
        walk_forward, walk_side, walk_back = self.WALK_SPEEDS_MPS
        forward_speed = run_forward + (walk_forward - run_forward) * self._gait
        side_speed = run_side + (walk_side - run_side) * self._gait
        back_speed = run_back + (walk_back - run_back) * self._gait
        scaled_forward = local_forward * (
            forward_speed if local_forward >= 0.0 else back_speed
        )
        scaled_left = local_left * side_speed
        velocity_x = ch * scaled_forward - sh * scaled_left
        velocity_y = sh * scaled_forward + ch * scaled_left

        if state.strafe:
            heading_forward = 1.0
            heading_left = 0.0
            if math.hypot(right_x, right_z) > 0.01:
                heading_forward = -right_z
                heading_left = -right_x
            heading_x = ca * heading_forward - sa * heading_left
            heading_y = sa * heading_forward + ca * heading_left
            self._desired_yaw = math.atan2(heading_y, heading_x)
        elif math.hypot(left_x, left_z) > 0.01:
            self._desired_yaw = math.atan2(world_left, world_forward)

        if state.stand:
            velocity_x = 0.0
            velocity_y = 0.0
        half = 0.5 * self._desired_yaw
        return MappedControlState(
            velocity_mujoco=(velocity_x, velocity_y, 0.0),
            desired_heading_mujoco_wxyz=(
                math.cos(half), 0.0, 0.0, math.sin(half)
            ),
            camera=CameraState(
                self._camera_sequence,
                self._camera_azimuth,
                self._camera_altitude,
                self._camera_distance,
            ),
            strafe=state.strafe,
            walk_blend=self._gait,
            stand=state.stand,
            terminate=state.terminate,
        )

    def command(
        self,
        chunk_index: int,
        mapped: MappedControlState,
    ) -> CommandSample | None:
        if type(chunk_index) is not int or chunk_index < 0:
            raise ContractError("Holden command chunk index must be nonnegative")
        if type(mapped) is not MappedControlState:
            raise ContractError("Holden command requires MappedControlState")
        if mapped.terminate:
            return None
        return CommandSample(
            chunk_index=chunk_index,
            requested_velocity_mujoco=(
                (0.0, 0.0, 0.0)
                if mapped.stand
                else mapped.velocity_mujoco
            ),
            desired_heading_mujoco_wxyz=mapped.desired_heading_mujoco_wxyz,
        )
```

Use the code above and ensure its surrounding record validation also:

1. validates exact booleans, finite axes in `[-1, 1]`, finite positive `dt_s`, and nonnegative integer chunk indices;
2. applies the Holden 0.2 deadzone and squared magnitude to both sticks;
3. uses `fast_negexp = 1 / (1 + x + 0.48*x*x + 0.235*x*x*x)` and the exact critically damped 0.1-second-halflife gait spring from `spring.h`;
4. updates camera only from the right stick when `strafe` is false and clamps altitude to `[0, 0.4*pi]` and distance to `[0.1, 100]`;
5. rotates left-stick travel by camera azimuth, applies interpolated forward/side/back speed components, and emits MuJoCo `(x, y, 0)`;
6. follows travel heading outside strafe and uses camera/right-stick heading in strafe;
7. stores heading through stand and idle frames;
8. emits a z-axis wxyz yaw quaternion and returns `None` after termination.

- [ ] **Step 4: Run the focused mapper tests**

Run the Step 2 command again.

Expected: all `test_sonic_holden_control` tests PASS.

- [ ] **Step 5: Commit the pure control layer**

```bash
git add sonic/python/mm_sonic/holden_control.py tests/python/test_sonic_holden_control.py
git commit -m "feat: port Holden operator command mapping"
```

### Task 2: Continuous X11 key levels, focus gate, and edge mailbox

**Files:**
- Create: `sonic/python/mm_sonic/operator_x11.py`
- Create: `tests/python/test_sonic_operator_x11.py`
- Modify: `sonic/python/mm_sonic/operator_terminal.py`
- Modify: `tests/python/test_sonic_operator_terminal.py`

**Interfaces:**
- Consumes: `NormalizedControlState` and `HoldenControlMapper` from Task 1.
- Produces: `KeyLevels`, `X11KeyStateProvider`, `ContinuousControlLoop`, and `BoundaryControlMailbox.sample(chunk_index) -> tuple[CommandSample | None, MappedControlState]`.

- [ ] **Step 1: Write failing provider-independent input tests**

Use an injected provider so ordinary unit tests do not require a display:

```python
class FakeProvider:
    def __init__(self, frames: list[KeyLevels]) -> None:
        self.frames = frames

    def sample(self) -> KeyLevels:
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

class ContinuousControlLoopTests(unittest.TestCase):
    def test_hold_release_and_focus_loss_publish_exact_transitions(self) -> None:
        events: list[str] = []
        provider = FakeProvider([
            KeyLevels(focused=True, pressed=frozenset({"W"})),
            KeyLevels(focused=True, pressed=frozenset({"W", "LEFT_CTRL"})),
            KeyLevels(focused=False, pressed=frozenset()),
        ])
        loop = ContinuousControlLoop(
            provider,
            HoldenControlMapper(initial_heading_yaw_rad=0.0),
            event_sink=events.append,
            period_s=0.001,
        )
        with loop:
            self.assertTrue(loop.wait_for_sequence(3, timeout_s=1.0))
        self.assertIn("KEY W DOWN -> forward", events)
        self.assertIn("KEY LEFT_CTRL DOWN -> strafe", events)
        self.assertIn("FOCUS LOST -> neutral", events)

    def test_space_and_x_edges_survive_release_until_consumed(self) -> None:
        def mapped_state(*, stand: bool, terminate: bool) -> MappedControlState:
            return MappedControlState(
                velocity_mujoco=(0.5, 0.0, 0.0),
                desired_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
                camera=CameraState(0, 0.0, 0.4, 3.0),
                strafe=False,
                walk_blend=0.0,
                stand=stand,
                terminate=terminate,
            )
        mailbox = BoundaryControlMailbox()
        mailbox.publish(mapped_state(stand=True, terminate=False))
        mailbox.publish(mapped_state(stand=False, terminate=False))
        first, _ = mailbox.sample(0)
        self.assertEqual(first.requested_velocity_mujoco, (0.0, 0.0, 0.0))
        mailbox.publish(mapped_state(stand=False, terminate=True))
        mailbox.publish(mapped_state(stand=False, terminate=False))
        terminated, _ = mailbox.sample(1)
        self.assertIsNone(terminated)
```

Also retain the existing terminal reader as a compatibility path and add Ctrl+A/arrow escape tests proving it reports them as unsupported rather than claiming parity.

- [ ] **Step 2: Verify the X11 tests fail before implementation**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_operator_x11 tests.python.test_sonic_operator_terminal
```

Expected: `test_sonic_operator_x11` fails to import; existing terminal tests remain green.

- [ ] **Step 3: Implement the libX11 provider and 50 Hz loop**

Implement a small `_X11Bindings` wrapper with declared `ctypes` signatures for `XOpenDisplay`, `XCloseDisplay`, `XQueryKeymap`, `XKeysymToKeycode`, `XGetInputFocus`, `XFetchName`, `XQueryTree`, and `XFree`. Resolve these keysyms exactly:

```python
KEYSYMS = {
    "W": 0x0077,
    "A": 0x0061,
    "S": 0x0073,
    "D": 0x0064,
    "Q": 0x0071,
    "E": 0x0065,
    "X": 0x0078,
    "SPACE": 0x0020,
    "LEFT": 0xFF51,
    "UP": 0xFF52,
    "RIGHT": 0xFF53,
    "DOWN": 0xFF54,
    "LEFT_SHIFT": 0xFFE1,
    "LEFT_CTRL": 0xFFE3,
}
```

`X11KeyStateProvider.sample()` must return pressed levels only when the focused window or one of its ancestors has a title containing `MuJoCo` or `G1 CONTROLS`. It closes its display exactly once and hard-fails if `DISPLAY` or libX11 is unavailable.

`ContinuousControlLoop` runs one non-daemon thread, samples every 0.02 seconds by default, emits only state transitions, advances the pure mapper, edge-latches Space/X, neutralizes on focus loss, and joins within two seconds. It accepts the manual driver's shared `threading.Event`; an X rising edge sets that event immediately. The mailbox validates monotonic sequences and provides an atomic latest-state/edge-consumption boundary.

Convert pressed levels into the normalized state exactly as follows:

```python
NormalizedControlState(
    left_x=float("D" in pressed) - float("A" in pressed),
    left_z=float("S" in pressed) - float("W" in pressed),
    right_x=float("RIGHT" in pressed) - float("LEFT" in pressed),
    right_z=float("DOWN" in pressed) - float("UP" in pressed),
    strafe="LEFT_CTRL" in pressed,
    walk="LEFT_SHIFT" in pressed,
    zoom=float("Q" in pressed) - float("E" in pressed),
    stand="SPACE" in pressed,
    terminate="X" in pressed,
)
```

If sampling raises or no successful sample arrives for 0.1 seconds, publish one neutral focus-lost state, retain the exception for the context manager to raise, and stop the reader. This makes provider disconnect and stale input fail safe.

- [ ] **Step 4: Run unit tests and a real-display read-only canary**

Run the Step 2 command, then:

```bash
env DISPLAY=:1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python sonic/.venv/bin/python -B -c 'from mm_sonic.operator_x11 import X11KeyStateProvider; p=X11KeyStateProvider(); print(p.sample()); p.close()'
```

Expected: all unit tests PASS; the canary prints one finite `KeyLevels` record and exits zero.

- [ ] **Step 5: Commit continuous input**

```bash
git add sonic/python/mm_sonic/operator_x11.py sonic/python/mm_sonic/operator_terminal.py tests/python/test_sonic_operator_x11.py tests/python/test_sonic_operator_terminal.py
git commit -m "feat: capture continuous Holden desktop controls"
```

### Task 3: Strict MuJoCo viewer camera protocol

**Files:**
- Modify: `sonic/python/mm_sonic/gated_sim.py`
- Modify: `sonic/python/mm_sonic/process.py`
- Modify: `tests/python/test_sonic_gated_sim.py`
- Modify: `tests/python/test_sonic_process.py`

**Interfaces:**
- Consumes: Task 1 `CameraState` values converted to degrees by the caller.
- Produces: `GatedSimulatorClient.set_camera(sequence, azimuth_deg, elevation_deg, distance_m) -> Mapping[str, object]`.

- [ ] **Step 1: Write failing runner, server, and client camera tests**

Extend the fake backend with `set_camera()` and assert exact validation and acknowledgement:

```python
def test_camera_request_applies_without_advancing_physics(self) -> None:
    runner, backend = reset_runner_with_fake_backend(self.root)
    before = runner.snapshot()
    applied = runner.set_camera(
        sequence=7,
        azimuth_deg=45.0,
        elevation_deg=-20.0,
        distance_m=4.0,
    )
    self.assertEqual(applied, {
        "sequence": 7,
        "azimuth_deg": 45.0,
        "elevation_deg": -20.0,
        "distance_m": 4.0,
    })
    self.assertEqual(runner.snapshot(), before)
    self.assertEqual(backend.camera_calls[-1], (45.0, -20.0, 4.0))
```

Client tests must reject booleans, nonfinite values, distance outside `[0.1, 100.0]`, wrong response keys, and a sequence mismatch.

- [ ] **Step 2: Run the focused protocol tests and observe missing APIs**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_gated_sim tests.python.test_sonic_process
```

Expected: new camera tests FAIL because `set_camera` is missing; pre-existing tests PASS.

- [ ] **Step 3: Implement camera application and exact JSONL operation**

Add `set_camera(azimuth_deg, elevation_deg, distance_m)` to the backend protocol. In `ExternalGearBackend`, require an existing running passive viewer, then assign:

```python
viewer.cam.azimuth = azimuth_deg
viewer.cam.elevation = elevation_deg
viewer.cam.distance = distance_m
viewer.sync()
```

Add `GatedSimulatorRunner.set_camera()` that validates a nonnegative integer sequence, finite azimuth/elevation, and distance in `[0.1, 100.0]`, invokes the backend, and returns the exact four-field acknowledgement without stepping.

Add a `camera` request with exact keys `{v, op, request_id, sequence, azimuth_deg, elevation_deg, distance_m}` to `_handle_request`. Add the strict `GatedSimulatorClient.set_camera()` peer and verify response values bit-for-bit after finite float conversion.

- [ ] **Step 4: Re-run the protocol suites**

Run the Step 2 command.

Expected: both modules PASS with no warnings.

- [ ] **Step 5: Commit the camera seam**

```bash
git add sonic/python/mm_sonic/gated_sim.py sonic/python/mm_sonic/process.py tests/python/test_sonic_gated_sim.py tests/python/test_sonic_process.py
git commit -m "feat: control the gated MuJoCo viewer camera"
```

### Task 4: Authenticated terrain registration and reset-boundary physics state

**Files:**
- Create: `sonic/python/mm_sonic/scene_runtime.py`
- Create: `tests/python/test_sonic_scene_runtime.py`
- Modify: `sonic/python/mm_sonic/scene.py`
- Modify: `sonic/python/mm_sonic/cli.py`
- Modify: `tests/python/test_sonic_scene.py`

**Interfaces:**
- Consumes: `RegisteredScene`, validated `InitialBoundary`, `JointContract`, and `Dex3HandTargets`.
- Produces: shared GEAR scene constants and `initial_physics_state(scene, initial, contract, hand_targets) -> InitialPhysicsState`.

- [ ] **Step 1: Write failing initial-state and bounds tests**

Create synthetic registered flat and terrain scenes and assert named mapping, coordinate conversion, closed hands, bounds, hashing, and forbidden collision rejection:

```python
def test_initial_state_uses_mm_root_named_joints_and_closed_hands(self) -> None:
    state = initial_physics_state(
        self.registered,
        self.initial_boundary,
        self.contract,
        NEUTRAL_HAND_TARGETS,
    )
    model = mujoco.MjModel.from_xml_path(str(self.registered.gear_scene_xml))
    root = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
    np.testing.assert_allclose(state.qpos[root:root + 3], (1.0, 3.0, 2.0))
    for name, value in zip(LEFT_HAND_JOINT_ORDER, NEUTRAL_HAND_TARGETS.left):
        address = int(model.jnt_qposadr[model.joint(name).id])
        self.assertAlmostEqual(state.qpos[address], value)
    self.assertRegex(state.qpos_sha256, r"^[0-9a-f]{64}$")
    self.assertTrue(math.isfinite(state.pelvis_clearance_m))
    self.assertTrue(math.isfinite(state.left_foot_clearance_m))
    self.assertTrue(math.isfinite(state.right_foot_clearance_m))

def test_initial_state_rejects_root_outside_terrain_bounds(self) -> None:
    outside = replace(
        self.initial_boundary,
        physical_pelvis_position_holden=np.array((1000.0, 1.0, 1000.0)),
    )
    with self.assertRaisesRegex(SceneError, "scene domain"):
        initial_physics_state(
            self.registered, outside, self.contract, NEUTRAL_HAND_TARGETS
        )
```

- [ ] **Step 2: Run scene-focused tests and observe the missing module**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_scene_runtime tests.python.test_sonic_scene
```

Expected: new module import FAIL; existing scene tests PASS.

- [ ] **Step 3: Retain scene bounds and implement initial-state construction**

Add immutable optional `source_bounds_holden` and `transformed_bounds_mujoco` arrays to `RegisteredScene`, populate them in `register_scene`, include them in registration equality/tests, and use flat registry bounds for the flat scene.

Create shared constants in `scene_runtime.py`:

```python
SCENE_REGISTRY_PATH = SONIC_ROOT / "configs/scene_registry.json"
GEAR_SCENE_RELATIVE = Path("gear_sonic_deploy/g1/scene_29dof_with_hand.xml")
GEAR_ROBOT_RELATIVE = Path("gear_sonic_deploy/g1/g1_29dof_with_hand.xml")
GEAR_SCENE_SHA256 = "f8538904eb47cada1bfb2dcdc157099092aa63df4307d7e077b651b16bfb6c74"
GEAR_ROBOT_SHA256 = "8b68d8f06674c5c10cd2cd89764b3cfba9fabba5080b55ea67ee1dd12cf630cd"

@dataclass(frozen=True)
class InitialPhysicsState:
    qpos: np.ndarray
    qpos_sha256: str
    initial_boundary_sha256: str
    maximum_forbidden_penetration_m: float
    pelvis_clearance_m: float
    left_foot_clearance_m: float
    right_foot_clearance_m: float
```

`initial_physics_state` must map source joints through names, convert the physical pelvis vector/quaternion with existing transform functions, resolve all qpos addresses by MuJoCo joint name, write both seven-joint hand vectors, run `mj_forward`, reject a root outside transformed horizontal bounds, reject joint-limit violations, and reject terrain contact for any registered forbidden geom deeper than 0.005 metres. For terrain scenes, parse the already authenticated G1HF/v2 bytes (`<4sIIIffff` header plus z-major float32 samples), reproduce the fixed-diagonal `tx >= tz` interpolation in binary64, and sample the pelvis plus both sole geom positions after converting them back to Holden coordinates. Require finite surface heights and finite clearances; retain the three clearances in `InitialPhysicsState`. Flat scenes use registered plane height zero. Hash canonical little-endian float64 `qpos` and a deterministic encoding of the validated initial boundary.

Modify `cli.py` to import the shared constants under its existing private names, leaving Stage A output identities unchanged.

- [ ] **Step 4: Run scene tests and the real default-scene registration canary**

Run the Step 2 command. Add `RealDefaultSceneCanaryTests.test_register_default_scene_and_initial_state` to `test_sonic_scene_runtime.py`; skip it unless `SONIC_REAL_SCENE_CANARY=1`, and have it start the real MM server, reset `grail-curb-default` / `curb-forward` / `4.0`, register the pinned GEAR scene into a temporary run root, build `InitialPhysicsState`, load the XML with MuJoCo, and assert one `mm_terrain` geom and finite in-bounds qpos. Run it exactly with:

```bash
env SONIC_REAL_SCENE_CANARY=1 SONIC_TERRAIN_DIR=/home/ubuntu/projects/motion-matching/resources/g1_terrain PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_scene_runtime.RealDefaultSceneCanaryTests.test_register_default_scene_and_initial_state
```

Expected: unit tests PASS; the canary prints one authenticated terrain mesh and exits zero without advancing physics.

- [ ] **Step 5: Commit authenticated initial-state support**

```bash
git add sonic/python/mm_sonic/scene_runtime.py sonic/python/mm_sonic/scene.py sonic/python/mm_sonic/cli.py tests/python/test_sonic_scene_runtime.py tests/python/test_sonic_scene.py
git commit -m "feat: initialize SONIC from authenticated terrain scenes"
```

### Task 5: Integrate terrain, continuous controls, camera, and versioned artifacts

**Files:**
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_evidence.py`
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `tests/python/test_sonic_manual_evidence.py`
- Modify: `/home/ubuntu/drive-g1-sonic.sh`

**Interfaces:**
- Consumes: Tasks 1-4 public interfaces.
- Produces: one terrain-aware `manual_demo` startup path and backward-compatible v3/v4 artifact readers.

- [ ] **Step 1: Write failing CLI, startup-order, control, and artifact tests**

Add parser assertions:

```python
def test_terrain_and_x11_are_interactive_defaults(self) -> None:
    args = _parser().parse_args(["--mode", "interactive", "--onscreen"])
    self.assertEqual(args.scene_id, "grail-curb-default")
    self.assertEqual(args.route_id, "curb-forward")
    self.assertEqual(args.terrain_weight, 4.0)
    self.assertEqual(args.input_source, "x11")
    self.assertEqual(args.preload_chunks, 2)

def test_flat_mode_requires_registered_flat_identity(self) -> None:
    args = _parser().parse_args([
        "--scene-id", "sonic-flat-baseline",
        "--route-id", "flat-12s",
        "--terrain-weight", "0.0",
    ])
    self.assertEqual(args.scene_id, "sonic-flat-baseline")
```

Use injected fake MM, scene registration, simulator, and control loop factories to prove the order is `hello -> reset -> register_scene -> initial_physics_state -> simulator.reset`, and that one boundary forwards independent lateral velocity/heading plus the matching camera sequence.

Add a v4 summary fixture containing exact scene source/output hashes, initial hashes, input source, mapper version, lookahead, hands, commands, and snapshot. Assert one-byte mutation of each digest is rejected. Keep the existing v3 fixture and all existing v3 tests unchanged.

- [ ] **Step 2: Run manual modules and verify the new assertions fail**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_manual_demo tests.python.test_sonic_manual_evidence
```

Expected: new terrain/input/artifact tests FAIL; existing tests PASS.

- [ ] **Step 3: Refactor manual startup and interactive loop**

Remove `_copy_scene` from the terrain path. Start MM, validate `hello/reset`, call `register_scene` with the returned identities, build `InitialPhysicsState`, then construct GEAR/simulator with the registered XML and qpos. Reset MM with CLI-selected `SessionConfig` instead of the hard-coded flat identity.

For interactive `--input-source x11`, enter `ContinuousControlLoop`; at each chunk boundary consume its command/output, send any new camera sequence through `simulator.set_camera`, publish the command, and print:

```text
CONTROL chunk=000012 vx=+0.500 vy=+0.000 heading=+0.000 strafe=0 walk=1 presents_in=0.800s
```

If a fully consumed camera response has invalid acknowledgement data, print one
`CAMERA DISABLED` diagnostic and stop sending camera requests while leaving the
robot command stream intact. Child death, truncated JSONL, or an unconsumed
response remains fatal because protocol synchronization is then unknown.

Keep `TerminalInputReader` only behind `--input-source terminal`, with the old coarse W/S/A/D/Q/E behavior and an explicit non-parity warning. Script mode does not open either input source.

Write `mm-sonic-manual-demo/v4` summaries for registered terrain runs. Add exact `environment_control` fields for scene/route/weight, MM identities, source/output hashes, transform/bounds, initial hashes, input source, mapper version, and camera sequence. Parse v3 and v4 strictly in `manual_evidence.py`; never reinterpret a v3 flat run as v4 terrain evidence.

- [ ] **Step 4: Update the easy launcher**

Patch `/home/ubuntu/drive-g1-sonic.sh` so it launches the current worktree with `--scene-id grail-curb-default --route-id curb-forward --terrain-weight 4.0 --input-source x11 --preload-chunks 2`. Print the complete mapping and the line `Focus the MuJoCo window; key DOWN/UP transitions print here.` before `exec`.

- [ ] **Step 5: Run focused integration regressions**

Run the Step 2 command plus:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_operator tests.python.test_sonic_operator_terminal tests.python.test_sonic_manual_replay
```

Expected: all focused modules PASS; no warning is emitted.

- [ ] **Step 6: Commit the integrated driver**

```bash
git add sonic/python/mm_sonic/manual_demo.py sonic/python/mm_sonic/manual_evidence.py tests/python/test_sonic_manual_demo.py tests/python/test_sonic_manual_evidence.py
git commit -m "feat: drive SONIC in the registered Holden environment"
```

The home launcher is intentionally outside Git; record its SHA-256 in the task report.

### Task 6: Visual-first terrain qualification

**Files:**
- Generated only beneath `/home/ubuntu/mm-sonic-manual-runs` and `/home/ubuntu`.
- Modify source only if a visible defect has a diagnosed code cause; use a focused failing regression before each correction and commit that correction separately.

**Interfaces:**
- Consumes: integrated driver from Task 5.
- Produces: screenshots, a short kinematic inspection, a short physics video, and an interactive launcher canary.

- [ ] **Step 1: Capture the stationary terrain alignment**

Launch two stand chunks onscreen in `grail-curb-default`. Capture the MuJoCo window to `/home/ubuntu/g1-sonic-terrain-alignment.png`. Inspect robot scale, scene scale, spawn location, foot height, pelvis orientation, and camera framing.

Expected: the G1 stands at the registered MM reset position on the same visible curb scene; no floating, burial, axis swap, or 90-degree yaw error.

- [ ] **Step 2: Inspect the first reference kinematically**

Generate 2.0 seconds of Shift+W walk intent without starting policy physics, replay the resulting canonical reference against the registered scene, and render `/home/ubuntu/g1-sonic-terrain-kinematic.mp4` with command and contact overlays.

Expected: feet and pelvis move in the displayed command direction, no forbidden penetration exceeds 0.005 metres, and the reference does not visibly flail.

- [ ] **Step 3: Run a short physics stand/walk clip**

Run two stand chunks, five walk chunks, and two stand chunks in ordinary paced MuJoCo physics with closed hands. Capture `/home/ubuntu/g1-sonic-terrain-physics-smoke.mp4` and its SHA-256.

Expected: robot remains supported, travels forward on the terrain, and settles after stand. Reject the clip immediately for obvious instability or wrong-direction motion.

- [ ] **Step 4: Run the strafe/heading visual check**

Run a short input sequence containing Shift+W, Ctrl+A, Ctrl+Right, Shift+D, and Space. Overlay key transitions, velocity, heading, scene ID, pelvis-height minimum, pelvis-up minimum, path length, yaw change, and final stop speed.

Expected: Ctrl+A produces lateral travel without travel-facing rotation, Ctrl+Right changes heading independently, D travels laterally in the opposite direction, and Space queues stand.

- [ ] **Step 5: Launch the persistent driver for direct use**

Run `/home/ubuntu/drive-g1-sonic.sh 100000`, focus the MuJoCo window, and verify terminal DOWN/UP lines for W, A, Left Ctrl, Left Shift, arrows, Space, and X. Leave the qualified process running for the user only after the viewer is stable.

### Task 7: Formal regression, evidence, and result record

**Files:**
- Create: `docs/superpowers/results/2026-07-19-g1-sonic-holden-control-environment-parity.md`

**Interfaces:**
- Consumes: Tasks 1-6 commits and visual artifacts.
- Produces: repeatable gate commands, exact artifact identities, verdict, and easy paths.

- [ ] **Step 1: Run the complete focused parity set**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v \
  tests.python.test_sonic_holden_control \
  tests.python.test_sonic_operator_x11 \
  tests.python.test_sonic_operator \
  tests.python.test_sonic_operator_terminal \
  tests.python.test_sonic_gated_sim \
  tests.python.test_sonic_process \
  tests.python.test_sonic_scene_runtime \
  tests.python.test_sonic_scene \
  tests.python.test_sonic_manual_demo \
  tests.python.test_sonic_manual_replay \
  tests.python.test_sonic_manual_evidence
```

Expected: all tests PASS with `PYTHONWARNINGS=error`.

- [ ] **Step 2: Run the previously frozen manual/evidence gate**

Run the frozen manual/evidence modules exactly:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sonic/python PYTHONWARNINGS=error sonic/.venv/bin/python -B -m unittest -v tests.python.test_sonic_manual_demo tests.python.test_sonic_manual_evidence
```

Expected: all 53 pre-existing tests PASS, plus the new parity tests from Step 1.

- [ ] **Step 3: Hash and probe visual evidence**

For each PNG/MP4, record `sha256sum`, `ffprobe` codec, dimensions, frame rate, frame count, and duration. Independently inspect opening, middle, strafe, and stopping frames.

Expected: H.264 1280x720 evidence decodes successfully and matches the reported run directory/scene identities.

- [ ] **Step 4: Write the result record**

Record commit SHAs, exact test commands/counts, selected scene/route/weight, source and generated scene hashes, initial hashes, input source, launcher hash, run root, screenshots, video hashes, metrics, visible verdict, formal verdict, known limitations, and the live process/window IDs.

- [ ] **Step 5: Commit the result record and verify a clean tree**

```bash
git add docs/superpowers/results/2026-07-19-g1-sonic-holden-control-environment-parity.md
git commit -m "docs: qualify G1 Holden terrain controls"
git status --short
```

Expected: result commit succeeds and `git status --short` prints nothing.
