# Holden Movement Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, transaction-safe `holden-v1` velocity shaper that brakes through zero on abrupt reversals, predicts the same feasible velocity ramp for motion matching, and produces comparable raw-versus-shaped SONIC evidence.

**Architecture:** A focused, header-only C++ movement-model unit owns the pure bounded-velocity math. `g1_controller_state` owns the selected profile and persistent intermediate velocity so the existing clone/swap candidate transaction automatically makes abort and regeneration safe. The MM reset protocol selects the profile; Python only validates, forwards, and records it, while the existing 0.2-second responsive pipeline remains responsible for target publication and physics release.

**Tech Stack:** C++17 header-only runtime and warning-strict executable tests; JSONL MM protocol v1; Python 3 dataclasses, NumPy, and `unittest`; existing GEAR SONIC, MuJoCo, X11 responsive demo, and evidence bundle.

## Global Constraints

- Support exactly `raw` and `holden-v1`; `raw` remains the default.
- `holden-v1` acceleration is exactly `1.5 m/s^2`; deceleration is exactly `2.0 m/s^2`.
- Directional acceleration and turn strength remain disabled.
- Keep the existing MM root spring at `0.27 s` and pose inertialization at `0.10 s`.
- Advance persistent movement state exactly once per 25 Hz MM step (`dt = 0.04 s`).
- Future trajectory prediction uses a local state copy at each `1/3 s` sample and never mutates persistent state.
- Equal-speed 180-degree reversals brake through zero before accelerating in the opposite direction.
- The movement model runs after traversability limiting, and a blocked stop clears its planar state.
- Explicitly unsupported profiles fail before reset publication; never silently fall back.
- Preserve the existing generate, validate, prepare, supersession, publish, dual-commit, and physics-release ordering.
- Keep `--responsive-source-intervals 5` end-to-end matched at 0.2 seconds and leave the 10-interval default unchanged.
- Automated tests must be observed RED before production edits and GREEN afterward.
- Do not change SONIC weights, add post-MM warping, expose tuning flags, or add learned models in this experiment.

---

## File Map

- Create `sonic/cpp/g1_movement_model.h`: profile enum, fixed parameter record, parsing/name helpers, validation, one-step velocity update, and non-mutating future-sample prediction.
- Create `tests/cpp/test_g1_movement_model.cpp`: pure numeric tests for acceleration, reversal, lateral change, finite rejection, and prediction immutability.
- Modify `g1_controller_state.h`: selected profile and persistent intermediate velocity, including reset/swap/clone behavior.
- Modify `sonic/cpp/g1_runtime.h`: apply the model after traversability limiting, clear it on blocking, and feed shaped future samples to the direct predictor.
- Modify `tests/cpp/test_g1_controller_state.cpp` and `tests/cpp/test_g1_runtime.cpp`: state transaction and runtime integration coverage.
- Modify `sonic/cpp/mm_chunk_protocol.h`, `sonic/cpp/mm_chunk_json.h`, and `sonic/cpp/mm_chunk_server.cpp`: optional reset field, fail-closed validation, session selection, hello capability, and reset evidence.
- Modify `tests/cpp/test_mm_chunk_protocol.cpp` and `tests/python/test_mm_chunk_server.py`: protocol/default/rejection/transaction tests.
- Modify `sonic/python/mm_sonic/coordinator.py`, `sonic/python/mm_sonic/process.py`, `sonic/python/mm_sonic/manual_demo.py`, and `sonic/python/mm_sonic/manual_evidence.py`: immutable session configuration, client request/response validation, CLI selection, and versioned run-summary recording.
- Modify `sonic/python/mm_sonic/boundary_trace.py` and `sonic/python/mm_sonic/responsive_wiring.py`: first/last applied-velocity evidence in MuJoCo coordinates.
- Modify `tests/python/test_sonic_process.py`, `tests/python/test_sonic_manual_demo.py`, `tests/python/test_sonic_boundary_trace.py`, and `tests/python/test_sonic_responsive_wiring.py`: Python contract and evidence coverage.
- Create `docs/superpowers/results/2026-07-21-holden-movement-model-ab.md`: protected automated proof and live raw/`holden-v1` A/B findings.

---

### Task 1: Pure acceleration/deceleration movement model

**Files:**
- Create: `sonic/cpp/g1_movement_model.h`
- Create: `tests/cpp/test_g1_movement_model.cpp`
- Modify: `sonic/cpp/Makefile`

**Interfaces:**
- Produces: `enum g1_movement_model_profile { G1MovementRaw, G1MovementHoldenV1 }`.
- Produces: `struct g1_movement_model_config` and `g1_movement_model_fixed_config()` returning acceleration `1.5f`, deceleration `2.0f`, and disabled directional/turn features.
- Produces: `g1_movement_model_step(vec3& output, vec3 current, vec3 target, float dt, g1_movement_model_profile profile, const g1_movement_model_config& config, char* error, int capacity) -> bool`.
- Produces: `g1_movement_model_predict(slice1d<vec3> output, vec3 current, vec3 target, float sample_dt, g1_movement_model_profile profile, const g1_movement_model_config& config, char* error, int capacity) -> bool`.

- [ ] **Step 1: Write the pure RED tests**

Create one standalone warning-strict test executable. The test sequence must assert the exact public behavior below:

```cpp
vec3 velocity;
char error[256] = {};
const g1_movement_model_config config = g1_movement_model_fixed_config();
CHECK(g1_movement_model_step(
    velocity, vec3(), vec3(0.9f, 0.0f, 0.0f), 0.04f,
    G1MovementHoldenV1, config, error, sizeof(error)));
CHECK(nearly(velocity.x, 0.06f));

velocity = vec3(0.9f, 0.0f, 0.0f);
vec3 first_brake;
CHECK(g1_movement_model_step(
    first_brake, velocity, vec3(-0.9f, 0.0f, 0.0f), 0.04f,
    G1MovementHoldenV1, config, error, sizeof(error)));
CHECK(nearly(first_brake.x, 0.82f));
velocity = first_brake;
bool crossed_zero = false;
for (int i = 0; i < 32; ++i) {
    vec3 next;
    CHECK(g1_movement_model_step(
        next, velocity, vec3(-0.9f, 0.0f, 0.0f), 0.04f,
        G1MovementHoldenV1, config, error, sizeof(error)));
    CHECK(next.x <= velocity.x + 1.0e-6f);
    CHECK(length(next - velocity) <= 0.060001f);
    crossed_zero = crossed_zero || next.x <= 0.0f;
    velocity = next;
}
CHECK(crossed_zero);
CHECK(nearly(velocity.x, -0.9f));

vec3 lateral;
CHECK(g1_movement_model_step(
    lateral, vec3(0.6f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.6f), 0.04f,
    G1MovementHoldenV1, config, error, sizeof(error)));
CHECK(finite(lateral));
CHECK(length(lateral - vec3(0.6f, 0.0f, 0.0f)) <= 0.080001f);

array1d<vec3> predicted;
predicted.resize(4);
const vec3 persistent(0.3f, 0.0f, 0.0f);
CHECK(g1_movement_model_predict(
    predicted, persistent, vec3(-0.9f, 0.0f, 0.0f), 1.0f / 3.0f,
    G1MovementHoldenV1, config, error, sizeof(error)));
CHECK(same_bits(persistent, vec3(0.3f, 0.0f, 0.0f)));
CHECK(predicted(0).x < 0.3f);
CHECK(predicted(3).x <= predicted(2).x);

vec3 raw;
CHECK(g1_movement_model_step(
    raw, vec3(0.4f, 0.0f, 0.0f), vec3(-0.9f, 0.0f, 0.0f), 0.04f,
    G1MovementRaw, config, error, sizeof(error)));
CHECK(same_bits(raw, vec3(-0.9f, 0.0f, 0.0f)));
```

Define local `CHECK`, `nearly`, `finite`, and `same_bits` helpers at the top of the test file. Also reject zero/negative/non-finite `dt`, non-finite state/target, an invalid enum, and config copies with zero/negative/non-finite acceleration or deceleration without modifying the caller's output.

- [ ] **Step 2: Compile to verify RED**

Run:

```bash
mkdir -p /tmp/holden-movement-red
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_movement_model.cpp \
  -o /tmp/holden-movement-red/test_g1_movement_model
```

Expected: compilation fails because `sonic/cpp/g1_movement_model.h` and its symbols do not exist.

- [ ] **Step 3: Implement the minimal pure module**

Use vector magnitudes for the limit selection and delta direction. The core must be equivalent to:

```cpp
const float current_speed = length(current);
const float target_speed = length(target);
const float limit = target_speed + 1.0e-6f > current_speed
    ? config.acceleration
    : config.deceleration;
const vec3 difference = target - current;
const float distance = length(difference);
const float maximum_change = limit * dt;
output = distance <= maximum_change
    ? target
    : current + difference * (maximum_change / distance);
```

`raw` returns `target` exactly. `predict` keeps a local `predicted` variable, calls the same step once per output sample with `sample_dt`, and writes only the output slice. Validate all inputs and the fixed configuration before assigning `output`.

- [ ] **Step 4: Compile and run GREEN**

Run the command from Step 2, then:

```bash
/tmp/holden-movement-red/test_g1_movement_model
```

Expected: `G1 movement model tests passed`.

- [ ] **Step 5: Register the header dependency and commit**

Add `g1_movement_model.h` to the `mm_chunk_server` dependency list in `sonic/cpp/Makefile`, then commit:

```bash
git add sonic/cpp/g1_movement_model.h sonic/cpp/Makefile \
  tests/cpp/test_g1_movement_model.cpp
git commit -m "feat: add pure Holden movement model"
```

---

### Task 2: Transactional controller state and runtime prediction

**Files:**
- Modify: `g1_controller_state.h`
- Modify: `sonic/cpp/g1_runtime.h`
- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `tests/cpp/test_g1_runtime.cpp`

**Interfaces:**
- Consumes: `G1MovementRaw`, `G1MovementHoldenV1`, `g1_movement_model_step`, and `g1_movement_model_predict` from Task 1.
- Produces: `g1_controller_state::movement_model_profile` and `g1_controller_state::movement_velocity`.
- Produces: shaped `state.command.applied_velocity` and `state.trajectory_desired_velocities` for `holden-v1`.

- [ ] **Step 1: Add RED state-ownership tests**

Extend the existing poison/reset, swap, clone, and bit-identity helpers to include:

```cpp
state.movement_model_profile = G1MovementHoldenV1;
state.movement_velocity = vec3(0.25f, 0.0f, -0.5f);
```

Require reset to restore `G1MovementRaw` and zero velocity, swap to exchange both fields, clone to preserve both fields bit-for-bit, and the full state equality helper to compare them.

- [ ] **Step 2: Add RED runtime behavior tests**

In the existing deterministic runtime fixture, add four checks:

```cpp
state.movement_model_profile = G1MovementHoldenV1;
request.requested_velocity_holden = vec3(0.9f, 0.0f, 0.0f);
CHECK(step_once(state, request));
CHECK(nearly(state.command.applied_velocity.x, 0.06f));
CHECK(state.trajectory_desired_velocities(0).x > 0.06f);
CHECK(state.trajectory_desired_velocities(0).x <= 0.560001f);
```

Then establish `+0.9f`, request `-0.9f`, and require the first applied value to remain positive and decrease by at most `0.08f`. Verify a prediction callback failure leaves the original state's profile and movement velocity bit-identical. Verify a blocked traversal yields zero planar `movement_velocity`. Finally run the same fixture with `G1MovementRaw` and compare every pre-existing command/trajectory diagnostic against the stored raw expectation.

- [ ] **Step 3: Compile both tests to verify RED**

Run:

```bash
mkdir -p /tmp/holden-runtime-red
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/holden-runtime-red/test_g1_controller_state
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp \
  -o /tmp/holden-runtime-red/test_g1_runtime
```

Expected: both fail because movement-model state and integration are absent.

- [ ] **Step 4: Add state and current-step integration**

Include `sonic/cpp/g1_movement_model.h` from `g1_controller_state.h`. Add profile and velocity next to `desired_velocity`; swap them in `g1_controller_state_swap`. Their in-class defaults provide raw/zero reset semantics.

In `g1_runtime_step_internal`, retain both values after traversability limiting:

```cpp
const vec3 limited_velocity = traversability_limit_command(
    next.traversal_speed_scale,
    next.traversal_speed_scale_velocity,
    traversal,
    active_scene.walkability,
    active_scene.terrain,
    next.simulation_position,
    commanded_velocity,
    dt);
vec3 desired_velocity_curr;
if (!g1_movement_model_step(
        desired_velocity_curr,
        next.movement_velocity,
        limited_velocity,
        dt,
        next.movement_model_profile,
        g1_movement_model_fixed_config(),
        error,
        capacity)) {
    return false;
}
next.movement_velocity = desired_velocity_curr;
if (traversal.blocked) {
    next.movement_velocity.x = 0.0f;
    next.movement_velocity.z = 0.0f;
    desired_velocity_curr.x = 0.0f;
    desired_velocity_curr.z = 0.0f;
}
```

Keep `G1CommandIntent.requested_velocity = commanded_velocity` as raw intent evidence. Preserve raw direct-predictor behavior exactly; only the `holden-v1` branch calls `g1_movement_model_predict` from the committed step's `next.movement_velocity` toward `limited_velocity` using `runtime_config.trajectory_sample_time`.

- [ ] **Step 5: Make direct prediction explicit and non-mutating**

Add `limited_velocity` to the captured direct-prediction context. The velocity-fill callback must be structurally equivalent to:

```cpp
if (next.movement_model_profile == G1MovementRaw) {
    desired_velocities.set(direct_request.intent.requested_velocity);
    return true;
}
return g1_movement_model_predict(
    desired_velocities,
    next.movement_velocity,
    limited_velocity,
    runtime_config.trajectory_sample_time,
    next.movement_model_profile,
    g1_movement_model_fixed_config(),
    prediction_error,
    prediction_capacity);
```

This preserves the current raw arrays while giving `holden-v1` the traversability-limited braking ramp. In the flat hold-frame fast path in `mm_real_adapter::advance`, clear `state.movement_velocity` alongside `state.desired_velocity` so a later command cannot resurrect stale motion.

- [ ] **Step 6: Run focused GREEN tests and commit**

Run both executables from Step 3 and the Task 1 executable. Expected: all print their pass banner and exit zero.

```bash
git add g1_controller_state.h sonic/cpp/g1_runtime.h \
  tests/cpp/test_g1_controller_state.cpp tests/cpp/test_g1_runtime.cpp
git commit -m "feat: apply movement model transactionally"
```

---

### Task 3: Session-scoped MM protocol and reset evidence

**Files:**
- Modify: `sonic/cpp/mm_chunk_protocol.h`
- Modify: `sonic/cpp/mm_chunk_json.h`
- Modify: `sonic/cpp/mm_chunk_server.cpp`
- Modify: `tests/cpp/test_mm_chunk_protocol.cpp`
- Modify: `tests/python/test_mm_chunk_server.py`

**Interfaces:**
- Produces: optional JSON reset request key `movement_model`, defaulting to `raw` when omitted.
- Produces: hello field `supported_movement_models: ["raw", "holden-v1"]`.
- Produces: reset data field `movement_model` with exact selected profile and fixed parameters.

- [ ] **Step 1: Write RED parser and protocol tests**

Test all three reset shapes: the old seven-key request parses as `raw`; an eight-key request with `"movement_model":"holden-v1"` succeeds; and `"movement_model":"other"` returns `invalid_movement_model`. For rejection, poison the active fake state and scene identity first, then assert they remain unchanged.

Require reset evidence exactly equal to:

```json
{
  "profile": "holden-v1",
  "acceleration_mps2": 1.5,
  "deceleration_mps2": 2.0,
  "directional_acceleration": false,
  "turn_strength": false
}
```

Require hello to advertise exactly `['raw', 'holden-v1']` in that order. Generate, abort, regenerate from the same predecessor and command under `holden-v1`, then compare both candidate JSON payloads after replacing candidate/request IDs; the applied-velocity arrays must be identical.

- [ ] **Step 2: Run RED tests**

```bash
mkdir -p /tmp/holden-protocol-red
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_mm_chunk_protocol.cpp \
  -o /tmp/holden-protocol-red/test_mm_chunk_protocol
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_mm_chunk_server -v
```

Expected: the new capability, reset field, rejection code, and shaped arrays are absent.

- [ ] **Step 3: Implement additive reset parsing and pre-mutation validation**

Add `std::string movement_model = "raw"` to `mm_chunk_reset_request`. In `mm_chunk_json_parse_request`, accept only the existing exact key set or that set plus `movement_model`; parse the optional value as a nonempty string. In `mm_chunk_protocol::prepare_reset`, validate with:

```cpp
if (request.movement_model != "raw" &&
    request.movement_model != "holden-v1") {
    return mm_chunk_fail(
        error, "invalid_movement_model",
        "movement_model must be raw or holden-v1");
}
```

This check must precede `adapter_.prepare_reset`.

- [ ] **Step 4: Bind the selected profile to prepared controller state**

Translate the validated string once in the real reset adapter. Before loading a scene or rebuilding features, validate `g1_movement_model_fixed_config()` so invalid fixed parameters fail without touching prepared or active state. Set `next_state.movement_model_profile` before it is swapped into the prepared output. The fake adapter needs no duplicate profile field because protocol tests observe the request and serialized reset evidence directly. Do not store a second mutable profile in the real server; the cloned controller state is the session source of truth.

- [ ] **Step 5: Serialize capability and reset evidence**

Add the hello list next to `supported_source_intervals`. Extend `mm_json_reset_data` to accept the reset request and emit the exact five-field object from Step 1. Pass the request only after successful preparation; publication ordering remains unchanged.

- [ ] **Step 6: Run GREEN tests, build an isolated real server, and commit**

```bash
/tmp/holden-protocol-red/test_mm_chunk_protocol
BUILD_DIR=/tmp/holden-protocol-red/build make -C sonic/cpp mm_chunk_server
SONIC_MM_SERVER=/tmp/holden-protocol-red/build/mm_chunk_server \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_mm_chunk_server -v
git add sonic/cpp/mm_chunk_protocol.h sonic/cpp/mm_chunk_json.h \
  sonic/cpp/mm_chunk_server.cpp tests/cpp/test_mm_chunk_protocol.cpp \
  tests/python/test_mm_chunk_server.py
git commit -m "feat: select movement model per MM session"
```

Expected: C++ protocol pass banner, isolated server build success, and Python suite `OK` with only its registered skips.

---

### Task 4: Python client, CLI, and run identity

**Files:**
- Modify: `sonic/python/mm_sonic/coordinator.py`
- Modify: `sonic/python/mm_sonic/process.py`
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_evidence.py`
- Modify: `tests/python/test_sonic_process.py`
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `tests/python/test_sonic_manual_evidence.py`

**Interfaces:**
- Consumes: the Task 3 hello/reset fields.
- Produces: `SessionConfig(..., movement_model: str = "raw")`.
- Produces: CLI flag `--movement-model {raw,holden-v1}` with default `raw`.
- Produces: selected reset evidence copied into the manual summary and responsive evidence summary.

- [ ] **Step 1: Add RED immutable-config and process-client tests**

Require:

```python
self.assertEqual(SessionConfig("scene", "route", 4.0).movement_model, "raw")
self.assertEqual(
    SessionConfig("scene", "route", 4.0, "holden-v1").movement_model,
    "holden-v1",
)
with self.assertRaisesRegex(ContractError, "movement_model"):
    SessionConfig("scene", "route", 4.0, "other")
```

Capture one `MMChunkClient.reset` request and assert it includes `movement_model`. Return reset data with the exact Task 3 object and assert the client preserves it. Reject missing keys, wrong booleans, non-finite/non-positive rates, wrong profile, and a selected profile different from the request.

- [ ] **Step 2: Add RED CLI/startup/evidence tests**

Assert `_parser().parse_args([]).movement_model == "raw"`, explicit `holden-v1` parses, and argparse rejects other values. In both `_run_startup_transaction` and `run_demo`, capture `mm.reset` and assert the selected value is present in `SessionConfig`. Require a hello that omits the selected profile to fail before reset. Require generated `responsive-evidence.json` and the printed run summary to contain the exact reset `movement_model` object.

- [ ] **Step 3: Run Python tests to verify RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_process \
  tests.python.test_sonic_manual_demo -v
```

Expected: failures identify the missing dataclass field, request key, CLI option, and evidence record.

- [ ] **Step 4: Implement strict Python validation and forwarding**

Add the field after `terrain_weight` so old positional construction remains valid:

```python
@dataclass(frozen=True)
class SessionConfig:
    scene_id: str
    route_id: str
    terrain_weight: float
    movement_model: str = "raw"

    def __post_init__(self) -> None:
        if type(self.scene_id) is not str or not self.scene_id:
            raise ContractError("session scene_id must be a nonempty string")
        if type(self.route_id) is not str or not self.route_id:
            raise ContractError("session route_id must be a nonempty string")
        if type(self.terrain_weight) not in (int, float) or not math.isfinite(
            float(self.terrain_weight)
        ):
            raise ContractError("session terrain_weight must be finite")
        converted = np.float32(self.terrain_weight)
        if not np.isfinite(converted) or float(converted) != float(
            self.terrain_weight
        ):
            raise ContractError("session terrain_weight must be exact binary32")
        object.__setattr__(self, "terrain_weight", float(converted))
        if self.movement_model not in ("raw", "holden-v1"):
            raise ContractError("session movement_model must be raw or holden-v1")
```

`MMChunkClient.reset` must validate the same two values, send the key, accept the reset response's new exact key set, and validate the returned five-field object exactly. Add a small pure helper in `manual_demo.py` that checks `hello['supported_movement_models']` is exactly a list of unique strings and contains the selected profile before calling reset.

- [ ] **Step 5: Wire the CLI and record the server-authored profile**

Add:

```python
parser.add_argument(
    "--movement-model",
    choices=("raw", "holden-v1"),
    default="raw",
)
```

Pass `namespace.movement_model` through every `SessionConfig` constructed by the manual demo. Treat `reset['movement_model']` as authoritative evidence after the client checks it matches the request. Add that object to `responsive-evidence.json`; for the ordinary manual summary, add a versioned additive serializer/parser rather than weakening an existing exact-key parser, and keep old summary versions readable.

- [ ] **Step 6: Run GREEN tests and commit**

Run the Step 3 command. Also run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_coordinator \
  tests.python.test_sonic_manual_evidence -v
```

Expected: all suites report `OK`.

```bash
git add sonic/python/mm_sonic/coordinator.py sonic/python/mm_sonic/process.py \
  sonic/python/mm_sonic/manual_demo.py sonic/python/mm_sonic/manual_evidence.py \
  tests/python/test_sonic_process.py tests/python/test_sonic_manual_demo.py \
  tests/python/test_sonic_manual_evidence.py
git commit -m "feat: expose movement model in SONIC demo"
```

---

### Task 5: First/last applied-velocity boundary evidence

**Files:**
- Modify: `sonic/python/mm_sonic/boundary_trace.py`
- Modify: `sonic/python/mm_sonic/responsive_wiring.py`
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `tests/python/test_sonic_boundary_trace.py`
- Modify: `tests/python/test_sonic_responsive_wiring.py`
- Modify: `tests/python/test_sonic_manual_demo.py`

**Interfaces:**
- Produces: `AcceptedChunk.applied_velocity_mujoco_first` and `.applied_velocity_mujoco_last`.
- Produces: matching immutable `BoundaryTrace` fields and JSONL fields.

- [ ] **Step 1: Add RED trace and committer tests**

Construct a validated source chunk whose `command.applied_velocity_holden` begins `[0.1, 0.0, 0.2]` and ends `[-0.3, 0.0, 0.4]`. Require `ManualChunkCommitter` to transform both rows with `holden_to_mujoco_vectors`, store finite width-three tuples, and preserve them in the delayed one-prefix trace binding. Extend the exact dataclass field test and JSON serialization test with:

```python
"applied_velocity_mujoco_first": [0.1, -0.2, 0.0],
"applied_velocity_mujoco_last": [-0.3, -0.4, 0.0],
```

Use the repository's transform helper to obtain expected values rather than duplicating axis logic in production.

- [ ] **Step 2: Run focused tests to verify RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_boundary_trace \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_manual_demo -v
```

Expected: constructor/field failures show applied-velocity evidence is absent.

- [ ] **Step 3: Implement evidence extraction at the source boundary**

Immediately after `validator.validate_source(raw)` succeeds and before target resampling, read the first and last source-step applied velocities, transform them into MuJoCo coordinates, validate finiteness, and carry them in `AcceptedChunk`. Do not infer them from root displacement. Add both immutable vectors to `BoundaryTrace`, `build_boundary_trace`, and `trace_record`; bump only the responsive trace schema identifier because its exact public fields change.

- [ ] **Step 4: Run GREEN tests and commit**

Run the Step 2 command. Expected: all suites report `OK`.

```bash
git add sonic/python/mm_sonic/boundary_trace.py \
  sonic/python/mm_sonic/responsive_wiring.py \
  sonic/python/mm_sonic/manual_demo.py \
  tests/python/test_sonic_boundary_trace.py \
  tests/python/test_sonic_responsive_wiring.py \
  tests/python/test_sonic_manual_demo.py
git commit -m "feat: trace shaped movement velocities"
```

---

### Task 6: Protected regression and transactional proof

**Files:**
- Modify only if a genuine uncovered defect is found in the files already authorized above.

**Interfaces:**
- Consumes: all production and test interfaces from Tasks 1-5.
- Produces: a fresh isolated MM server and a complete automated evidence record.

- [ ] **Step 1: Compile the complete affected C++ catalog warning-strict**

```bash
set -euo pipefail
BUILD=/tmp/holden-movement-final-cpp
rm -rf "$BUILD"
mkdir -p "$BUILD"
for name in test_g1_movement_model test_g1_controller_state test_g1_runtime \
  test_mm_chunk_protocol; do
  c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/${name}.cpp" -o "$BUILD/$name"
  "$BUILD/$name"
done
BUILD_DIR="$BUILD/server" make -C sonic/cpp mm_chunk_server
```

Expected: four pass banners and a successful isolated `mm_chunk_server` build.

- [ ] **Step 2: Run the affected Python catalog against that server**

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export SONIC_MM_SERVER=/tmp/holden-movement-final-cpp/server/mm_chunk_server
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_mm_chunk_server \
  tests.python.test_sonic_schema \
  tests.python.test_sonic_process \
  tests.python.test_sonic_coordinator \
  tests.python.test_sonic_boundary_trace \
  tests.python.test_sonic_responsive_scheduler \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_manual_evidence \
  tests.python.test_sonic_manual_demo -v
```

Expected: all tests `OK`; only pre-registered environment skips are permitted.

- [ ] **Step 3: Inspect exact compatibility and transaction assertions**

Confirm from test output and source that omitted reset profile is raw, explicit invalid profile does not invoke adapter reset, raw generated arrays retain their old fixture values, abort/regenerate is bit-identical, and `holden-v1` first-step deltas are bounded at `0.08 m/s` while braking and `0.06 m/s` while accelerating.

- [ ] **Step 4: Commit any test-only hardening**

If Step 3 required no edits, record that fact in the result document rather than creating an empty commit. If it exposed a defect, add one failing regression first, apply the smallest fix inside the authorized files, rerun Steps 1-2, and commit the focused correction.

---

### Task 7: Live 0.2-second raw versus `holden-v1` A/B

**Files:**
- Create: `docs/superpowers/results/2026-07-21-holden-movement-model-ab.md`

**Interfaces:**
- Consumes: the isolated server from Task 6 and existing responsive evidence bundles.
- Produces: honest, path-addressed raw and shaped run evidence and a supported/rejected/inconclusive hypothesis verdict.

- [ ] **Step 1: Start the raw control run**

Use the existing verified GEAR checkout, runtime, terrain directory, and `DISPLAY=:1`. Launch the manual demo with the isolated server and these experiment flags:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m mm_sonic.manual_demo \
  --mode interactive --input-source x11 --onscreen --responsive \
  --responsive-source-intervals 5 --movement-model raw \
  --mm-server /tmp/holden-movement-final-cpp/server/mm_chunk_server \
  --chunks 10000 --output-root /home/ubuntu/mm-sonic-holden-ab/raw
```

After LIVE, apply long forward motion, an abrupt W-to-S reversal, long backward motion, an abrupt A-to-D reversal, and long right motion. End with X. Do not smooth the keys manually.

- [ ] **Step 2: Start the shaped run under the same conditions**

Run the identical command with only:

```text
--movement-model holden-v1 --output-root /home/ubuntu/mm-sonic-holden-ab/holden-v1
```

Repeat the same abrupt input sequence and approximate hold durations. Record shared-display contamination explicitly if it occurs.

- [ ] **Step 3: Compute the comparison from durable evidence**

For each W-to-S and A-to-D boundary, use `requested_velocity_mujoco`, `applied_velocity_mujoco_first/last`, `generated_virtual_root_displacement_mujoco`, and `observed_mujoco_root_displacement` to report:

- signed wrong-direction physical displacement accumulated after reversal;
- number of 0.2-second prefixes and simulated seconds until three consecutive positive projections;
- generated-versus-observed displacement error per prefix;
- completed-prefix latency, MM generation time, and real-time factor;
- process completion, fall/upright status, and evidence availability.

Do not replace missing physical measurements with zero. If input contamination prevents matched segments, label that direction inconclusive and retain the run.

- [ ] **Step 4: Write the result and verdict**

The result document must name both run roots, exact branch commit, MM binary path/hash, selected model records, parameters, automated test counts, per-direction tables, and one verdict:

- `supported` if `holden-v1` reduces wrong-direction displacement or tracking error without instability;
- `rejected` if tracking or stability worsens;
- `inconclusive` if comparable physical evidence is unavailable.

Longer intentional time-to-target alone is not rejection. Separate MM generated-direction response from SONIC physical tracking response.

- [ ] **Step 5: Commit the evidence record**

```bash
git add docs/superpowers/results/2026-07-21-holden-movement-model-ab.md
git commit -m "docs: record Holden movement model A/B"
```
