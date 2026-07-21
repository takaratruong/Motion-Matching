# Holden Turn Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `holden-turn-v1` movement profile that preserves `holden-v1` translational shaping while limiting the desired yaw seen by motion matching and runtime application to exactly 120 degrees/second.

**Architecture:** A new pure header-only yaw limiter owns planar-heading validation, shortest-path stepping, and local-copy trajectory prediction. The existing `g1_controller_state::desired_rotation` remains the sole persistent heading anchor, so the current clone/swap transaction automatically protects aborts and failures. The server and Python pipeline add profile selection plus versioned requested-versus-applied heading evidence without changing the historical `raw` and `holden-v1` movement math.

**Tech Stack:** C++17 header-only runtime and warning-strict executable tests; exact-key JSONL MM protocol; Python 3 dataclasses, NumPy, and `unittest`; existing GEAR SONIC, MuJoCo, X11 responsive demo, and isolated Linux network namespaces.

## Global Constraints

- Support exactly `raw`, `holden-v1`, and `holden-turn-v1`; `raw` remains the default.
- `holden-turn-v1` uses the exact `holden-v1` acceleration `1.5 m/s^2` and deceleration `2.0 m/s^2` behavior.
- `holden-turn-v1` maximum yaw rate is exactly `120 deg/s` (`2.094395102... rad/s`).
- At `dt = 0.04 s`, the maximum current-step yaw change is exactly `4.8 degrees`.
- At the existing `1/3 s` trajectory sample interval, the maximum predicted yaw change is exactly `40 degrees` per sample.
- A 180-degree reversal requires at least `1.5 s`; the exact plus/minus-pi tie advances toward positive pi.
- The shortest signed yaw delta lies in `(-pi, pi]`.
- `raw` and `holden-v1` return the requested unit quaternion exactly and retain protected bit-identical generated kinematics.
- Only `holden-turn-v1` requires planar-yaw quaternions; non-finite, non-unit, or non-planar input fails without mutation.
- Use `g1_controller_state::desired_rotation` as the only persistent heading anchor; do not add duplicate heading state.
- Limit the current heading once after state clone and request validation, before change diagnostics, prediction, motion-matching query construction, and runtime application.
- Future headings advance a local copy with the same pure limiter and never mutate persistent state.
- Keep the existing downstream `simulation_rotation_halflife = 0.27 s` unchanged.
- Keep source rate `25 Hz`, responsive prefix duration `0.2 s`, traversal behavior, pose inertialization, SONIC weights, and publication/commit/physics-release ordering unchanged.
- Preserve the operator-requested heading and record the first and last capped headings for every generated responsive prefix.
- Version changed exact-key evidence schemas; retain historical parsers unchanged and fail closed on mixed field sets.
- Automated tests must be observed RED before production edits and GREEN afterward.
- The live trial must run the simulator and GEAR controller together inside one temporary isolated network namespace.
- Do not add numeric tuning flags, speed-dependent turn coupling, a second rotation spring, post-MM warping, or a learned model in this experiment.

---

## File Map

- Create `sonic/cpp/g1_turn_model.h`: fixed yaw-rate contract, planar yaw conversion, shortest signed delta, transactional one-step limiter, and non-mutating predictor.
- Create `tests/cpp/test_g1_turn_model.cpp`: exact 4.8-degree stepping, 1.5-second reversal, wrap/tie behavior, pass-through parity, invalid-input immutability, and local-copy prediction.
- Modify `sonic/cpp/g1_movement_model.h` and `tests/cpp/test_g1_movement_model.cpp`: third profile enumeration and proof that its velocity output is bit-identical to `holden-v1`.
- Modify `sonic/cpp/g1_runtime.h`, `sonic/cpp/mm_chunk_server.cpp`, `tests/cpp/test_g1_runtime.cpp`, and `tests/cpp/test_g1_controller_state.cpp`: apply one capped current heading, predict the same capped path, expose per-step applied headings, and protect state transaction semantics.
- Modify `sonic/cpp/mm_chunk_protocol.h`, `sonic/cpp/mm_chunk_json.h`, `tests/cpp/test_mm_chunk_protocol.cpp`, and `tests/python/test_mm_chunk_server.py`: exact third-profile reset validation, capability advertisement, fixed contract evidence, and generated applied-heading arrays.
- Create `sonic/schemas/mm_chunk_v2.schema.json`; modify `sonic/python/mm_sonic/schema.py` and `tests/python/test_sonic_schema.py`: retain strict v1 parsing and add strict v2 parsing for applied headings.
- Modify `sonic/python/mm_sonic/coordinator.py`, `sonic/python/mm_sonic/process.py`, `sonic/python/mm_sonic/manual_demo.py`, `sonic/python/mm_sonic/manual_evidence.py`, and their tests: accept the third profile, validate profile-specific reset identity, and select new summary schemas only for the turn profile.
- Modify `sonic/python/mm_sonic/boundary_trace.py`, `sonic/python/mm_sonic/responsive_wiring.py`, `tests/python/test_sonic_boundary_trace.py`, and `tests/python/test_sonic_responsive_wiring.py`: carry first/last capped headings in MuJoCo coordinates and publish responsive trace v3.
- Create `docs/superpowers/results/2026-07-21-holden-turn-cap-abc.md`: protected verification and isolated raw/velocity-only/turn-capped live evidence.

---

### Task 1: Pure deterministic planar yaw limiter

**Files:**
- Create: `sonic/cpp/g1_turn_model.h`
- Create: `tests/cpp/test_g1_turn_model.cpp`
- Modify: `sonic/cpp/g1_movement_model.h`
- Modify: `tests/cpp/test_g1_movement_model.cpp`
- Modify: `sonic/cpp/Makefile`

**Interfaces:**
- Produces: `G1MovementHoldenTurnV1 = 2` in `g1_movement_model_profile`.
- Produces: `struct g1_turn_model_config { float max_yaw_rate_radians; }`.
- Produces: `g1_turn_model_fixed_config() -> g1_turn_model_config` with `max_yaw_rate_radians = 120.0f * PIf / 180.0f`.
- Produces: `g1_turn_model_step(quat& output, quat current, quat target, float dt, g1_movement_model_profile profile, const g1_turn_model_config& config, char* error, int capacity) -> bool`.
- Produces: `g1_turn_model_predict(slice1d<quat> output, quat current, quat target, float sample_dt, g1_movement_model_profile profile, const g1_turn_model_config& config, char* error, int capacity) -> bool`.
- Preserves: `g1_movement_model_step` and `g1_movement_model_predict` use identical non-raw code paths for `G1MovementHoldenV1` and `G1MovementHoldenTurnV1`.

- [ ] **Step 1: Write the pure RED test executable**

Create `tests/cpp/test_g1_turn_model.cpp` with local `CHECK`, float/quaternion bit comparison, `yaw_quat`, and `yaw_degrees` helpers. Its core public assertions are:

```cpp
char error[256] = {};
const g1_turn_model_config config = g1_turn_model_fixed_config();
quat current = yaw_quat(0.0f);
quat next = quat(9.0f, 8.0f, 7.0f, 6.0f);
CHECK(g1_turn_model_step(
    next, current, yaw_quat(PIf), 0.04f,
    G1MovementHoldenTurnV1, config, error, sizeof(error)));
CHECK(nearly(yaw_degrees(next), 4.8f, 1.0e-4f));
CHECK(nearly(quat_length(next), 1.0f, 1.0e-6f));

int steps = 1;
current = next;
while (yaw_degrees(current) < 179.999f && steps < 100) {
    CHECK(g1_turn_model_step(
        next, current, yaw_quat(PIf), 0.04f,
        G1MovementHoldenTurnV1, config, error, sizeof(error)));
    CHECK(yaw_degrees(next) + 1.0e-4f >= yaw_degrees(current));
    current = next;
    ++steps;
}
CHECK(steps >= 38);
CHECK(nearly(steps * 0.04f, 1.52f, 1.0e-5f));

const quat small = yaw_quat(2.0f * PIf / 180.0f);
CHECK(g1_turn_model_step(
    next, yaw_quat(0.0f), small, 0.04f,
    G1MovementHoldenTurnV1, config, error, sizeof(error)));
CHECK(same_quat_bits(next, small));
```

Also assert:

```cpp
// +179 to -179 takes the +2-degree shortest path.
CHECK(g1_turn_model_step(
    next, yaw_quat(179.0f * PIf / 180.0f),
    yaw_quat(-179.0f * PIf / 180.0f), 0.04f,
    G1MovementHoldenTurnV1, config, error, sizeof(error)));
CHECK(nearly(wrapped_yaw_degrees(next), -179.0f, 1.0e-4f));

// The exact 180-degree tie is positive.
CHECK(g1_turn_model_step(
    next, yaw_quat(0.0f), yaw_quat(-PIf), 0.04f,
    G1MovementHoldenTurnV1, config, error, sizeof(error)));
CHECK(nearly(yaw_degrees(next), 4.8f, 1.0e-4f));

quat predictions_storage[4];
const quat persistent = yaw_quat(0.0f);
CHECK(g1_turn_model_predict(
    slice1d<quat>(4, predictions_storage), persistent, yaw_quat(PIf),
    1.0f / 3.0f, G1MovementHoldenTurnV1, config,
    error, sizeof(error)));
CHECK(same_quat_bits(persistent, yaw_quat(0.0f)));
CHECK(nearly(yaw_degrees(predictions_storage[0]), 40.0f, 1.0e-4f));
CHECK(nearly(yaw_degrees(predictions_storage[1]), 80.0f, 1.0e-4f));
CHECK(nearly(yaw_degrees(predictions_storage[2]), 120.0f, 1.0e-4f));
CHECK(nearly(yaw_degrees(predictions_storage[3]), 160.0f, 1.0e-4f));
```

For both `G1MovementRaw` and `G1MovementHoldenV1`, pass a valid non-planar unit quaternion and require the exact requested bits to be returned. For `G1MovementHoldenTurnV1`, reject a non-planar unit quaternion, non-unit quaternion, NaN component, zero/negative/non-finite `dt`, invalid enum, and zero/negative/non-finite rate while leaving a sentinel output bit-identical.

- [ ] **Step 2: Extend the velocity-model RED parity test**

In `tests/cpp/test_g1_movement_model.cpp`, run every existing `holden-v1` one-step and prediction case again with `G1MovementHoldenTurnV1`, then compare `vec3` output bits element-by-element:

```cpp
vec3 velocity_only;
vec3 turn_profile;
CHECK(g1_movement_model_step(
    velocity_only, current, target, 0.04f,
    G1MovementHoldenV1, config, error, sizeof(error)));
CHECK(g1_movement_model_step(
    turn_profile, current, target, 0.04f,
    G1MovementHoldenTurnV1, config, error, sizeof(error)));
CHECK(same_bits(velocity_only, turn_profile));
```

- [ ] **Step 3: Compile to verify RED**

Run:

```bash
mkdir -p /tmp/holden-turn-red
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_turn_model.cpp \
  -o /tmp/holden-turn-red/test_g1_turn_model
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_movement_model.cpp \
  -o /tmp/holden-turn-red/test_g1_movement_model
```

Expected: compilation fails because `g1_turn_model.h` and `G1MovementHoldenTurnV1` do not exist.

- [ ] **Step 4: Implement the minimal pure limiter**

Use a profile gate before planar validation so historical profiles return `target` exactly. For the turn profile, extract planar yaw with `2 * atan2(q.y, q.w)`, reject `abs(q.x)` or `abs(q.z)` above `1.0e-5f`, wrap with this deterministic rule, and reconstruct around +Y:

```cpp
static inline float g1_turn_wrap_signed(float angle)
{
    float wrapped = std::fmod(angle + PIf, 2.0f * PIf);
    if (wrapped <= 0.0f) wrapped += 2.0f * PIf;
    return wrapped - PIf; // (-pi, pi]
}

const float current_yaw = 2.0f * std::atan2(current.y, current.w);
const float target_yaw = 2.0f * std::atan2(target.y, target.w);
const float delta = g1_turn_wrap_signed(target_yaw - current_yaw);
const float maximum = config.max_yaw_rate_radians * dt;
if (std::fabs(delta) <= maximum) {
    output = target;
    return true;
}
const float applied_yaw = current_yaw + std::copysign(maximum, delta);
output = quat_from_angle_axis(applied_yaw, vec3(0.0f, 1.0f, 0.0f));
return true;
```

Validate every input and the entire output slice before prediction writes. `g1_turn_model_predict` advances a local `predicted` quaternion through the same step function and copies a completed local array to the caller only after all samples succeed.

- [ ] **Step 5: Run GREEN and commit**

Run both commands from Step 3 and both executables. Expected: both print their pass banner and exit zero.

```bash
git add sonic/cpp/g1_turn_model.h sonic/cpp/g1_movement_model.h \
  sonic/cpp/Makefile tests/cpp/test_g1_turn_model.cpp \
  tests/cpp/test_g1_movement_model.cpp
git commit -m "feat: add fixed-rate Holden yaw limiter"
```

---

### Task 2: Transactional current and predicted heading integration

**Files:**
- Modify: `sonic/cpp/g1_runtime.h`
- Modify: `sonic/cpp/mm_chunk_protocol.h`
- Modify: `sonic/cpp/mm_chunk_server.cpp`
- Modify: `tests/cpp/test_g1_runtime.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`

**Interfaces:**
- Consumes: `g1_turn_model_step`, `g1_turn_model_predict`, and `G1MovementHoldenTurnV1` from Task 1.
- Produces: current `next.desired_rotation` capped from the previously committed `state.desired_rotation`.
- Produces: `G1CommandSnapshot::intent.desired_heading` as the capped applied heading while the generate request remains the original operator heading.
- Produces: `mm_chunk_step_diagnostic::applied_heading_holden_wxyz[4]` populated from the committed runtime command.

- [ ] **Step 1: Add RED runtime agreement tests**

Add a direct-runtime fixture that seeds `state.desired_rotation = yaw_quat(0.0f)`, selects `G1MovementHoldenTurnV1`, and requests 180 degrees. Require:

```cpp
check(step_once(state, request), "turn-profile step succeeds");
check(nearly(yaw_degrees(state.desired_rotation), 4.8f),
      "current desired heading is capped to 4.8 degrees");
check(same_quat_bits(
          state.command.intent.desired_heading, state.desired_rotation),
      "command evidence uses capped current heading");
check(nearly(yaw_degrees(
          state.command.predicted_desired_headings[0]), 44.8f),
      "first future heading advances another 40 degrees");
check(nearly(yaw_degrees(
          state.command.predicted_desired_headings[3]), 164.8f),
      "future rollout advances one shared capped path");
```

Step the fixture repeatedly and require each committed delta to be at most `4.8001` degrees, monotone toward the target, and no earlier than the 38th 25-Hz step. Assert `raw` and `holden-v1` still set the exact requested heading and exact prior trajectory-heading bits.

- [ ] **Step 2: Add RED transaction, blocked, hold, and force-search tests**

Clone the turn-profile state before each injected downstream failure already supported by `test_g1_runtime.cpp`; require `desired_rotation`, command snapshot, trajectory desired rotations, and movement velocity to remain bit-identical after failure. Add an explicit blocked traversal case proving velocity clears while the heading advances only by the cap. Run a flat hold followed by a moving 180-degree command and require the first moving step to advance from the hold's committed anchor, not a stale pre-hold heading. Under sustained capped rotation, require the existing rotation-change threshold logic to eventually set `searched` without introducing a snap.

Extend the state source-order/static checks in `test_g1_controller_state.cpp` to prove `desired_rotation` remains swapped, reset, and cloned exactly once and no new persistent turn-heading field appears.

- [ ] **Step 3: Compile to verify RED**

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/holden-turn-red/test_g1_controller_state
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_runtime.cpp \
  -o /tmp/holden-turn-red/test_g1_runtime
```

Expected: runtime cap, capped future headings, and applied-heading diagnostics are absent.

- [ ] **Step 4: Apply the current cap at the shared runtime seam**

In `g1_runtime_step_internal`, retain the original requested quaternion but make the applied value mutable and cap it immediately after cloning:

```cpp
const quat requested_rotation = request.desired_heading_holden;
quat desired_rotation_curr;
if (!g1_turn_model_step(
        desired_rotation_curr,
        next.desired_rotation,
        requested_rotation,
        config.dt,
        next.movement_model_profile,
        g1_turn_model_fixed_config(),
        error,
        capacity)) {
    return false;
}
```

Use `desired_rotation_curr` for rotation-change diagnostics, `next.desired_rotation`, `command_intent.desired_heading`, the frame request, query construction, and downstream simulation target. Do not overwrite `request.desired_heading_holden`; it remains requested-intent evidence at the server boundary.

- [ ] **Step 5: Roll out future capped headings from a local copy**

In the direct prediction builder, replace the constant heading override for `holden-turn-v1` with the same pure predictor:

```cpp
if (next.movement_model_profile == G1MovementHoldenTurnV1) {
    direct_request.heading_override.active = false;
} else {
    direct_request.heading_override.active = true;
    direct_request.heading_override.heading =
        direct_request.intent.desired_heading;
}
```

The heading predictor callback must be:

```cpp
return g1_turn_model_predict(
    desired_rotations,
    next.desired_rotation,
    requested_rotation,
    runtime_config.trajectory_sample_time,
    next.movement_model_profile,
    g1_turn_model_fixed_config(),
    fill_error,
    fill_capacity);
```

For raw and velocity-only profiles, the existing heading-override branch stays byte-for-byte behaviorally identical. Keep `simulation_rotation_halflife` unchanged.

- [ ] **Step 6: Capture per-step applied heading and repair the hold anchor**

Add `float applied_heading_holden_wxyz[4]` to `mm_chunk_step_diagnostic`, include it in equality, and populate it from `state.command.intent.desired_heading` beside `applied_velocity_holden`. In the flat hold path, set `state.desired_rotation` and command intent/predicted desired headings to the hold request's applied heading so the next moving request has one coherent committed anchor.

- [ ] **Step 7: Run GREEN and commit**

Run the two executables from Step 3 plus both Task 1 executables. Expected: all pass.

```bash
git add sonic/cpp/g1_runtime.h sonic/cpp/mm_chunk_protocol.h \
  sonic/cpp/mm_chunk_server.cpp tests/cpp/test_g1_runtime.cpp \
  tests/cpp/test_g1_controller_state.cpp
git commit -m "feat: cap MM heading prediction and application"
```

---

### Task 3: Exact server protocol and versioned source-chunk evidence

**Files:**
- Modify: `sonic/cpp/mm_chunk_protocol.h`
- Modify: `sonic/cpp/mm_chunk_json.h`
- Modify: `sonic/cpp/mm_chunk_server.cpp`
- Modify: `tests/cpp/test_mm_chunk_protocol.cpp`
- Modify: `tests/python/test_mm_chunk_server.py`
- Create: `sonic/schemas/mm_chunk_v2.schema.json`
- Modify: `sonic/python/mm_sonic/schema.py`
- Modify: `tests/python/test_sonic_schema.py`

**Interfaces:**
- Produces: hello `supported_movement_models = ["raw", "holden-v1", "holden-turn-v1"]`.
- Produces: profile-specific reset movement record; historical profiles keep the exact five-key record, while `holden-turn-v1` adds `"max_yaw_rate_deg_s": 120.0`.
- Preserves: `mm-chunk/v1` and `_parse_command_v1` for historical payloads.
- Produces: every newly generated chunk as `mm-chunk/v2` with exact additional `command.applied_heading_holden_wxyz` shaped `(source_intervals, 4)`; raw and `holden-v1` rows equal the requested heading bit-for-bit.

- [ ] **Step 1: Add RED server selection and immutability tests**

Require reset acceptance for all three exact strings and rejection of every other string before adapter/prepared/active mutation. Require hello order exactly as declared above. For `holden-turn-v1`, require reset evidence exactly:

```json
{
  "profile": "holden-turn-v1",
  "acceleration_mps2": 1.5,
  "deceleration_mps2": 2.0,
  "directional_acceleration": false,
  "turn_strength": false,
  "max_yaw_rate_deg_s": 120.0
}
```

Keep existing raw and `holden-v1` reset-object equality assertions unchanged. Add invalid non-planar generate-heading coverage under the turn profile and prove candidate/active session state is unchanged.

- [ ] **Step 2: Add RED v2 generated-heading tests**

Generate a 5-step responsive prefix from zero toward 180 degrees. Require schema `mm-chunk/v2`, preserve the original `desired_heading_holden_wxyz`, and require five `applied_heading_holden_wxyz` rows whose adjacent yaw deltas are at most `4.8001` degrees. Require the first and last rows to match the runtime's first and fifth committed desired headings. Abort and regenerate the same request and compare the entire applied-heading array bit-for-bit. Generate raw and `holden-v1` prefixes and require every applied-heading row to equal the requested heading bits exactly.

Add Python schema tests that parse a retained v1 fixture unchanged and parse this v2 command:

```python
self.assertEqual(chunk.command["applied_heading_holden_wxyz"].shape, (5, 4))
self.assertFalse(chunk.command["applied_heading_holden_wxyz"].flags.writeable)
np.testing.assert_array_equal(
    chunk.command["desired_heading_holden_wxyz"], requested_heading
)
```

Reject v1-with-v2-field, v2-without-field, wrong row count, non-unit rows, NaN rows, and duplicate keys.

- [ ] **Step 3: Run RED protocol and schema tests**

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_mm_chunk_protocol.cpp \
  -o /tmp/holden-turn-red/test_mm_chunk_protocol
PYTHONPATH="$PWD/sonic/python" \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_schema tests.python.test_mm_chunk_server -v
```

Expected: the third profile, v2 schema, and applied-heading evidence are absent.

- [ ] **Step 4: Implement fail-closed third-profile translation**

Update both protocol and real adapter checks to the exact three-value set. Translate `holden-turn-v1` to `G1MovementHoldenTurnV1` before any scene or state mutation. Probe both fixed velocity and turn configs during preparation. Serialize the third hello capability in stable order.

`mm_json_write_movement_model` must use the historical exact five fields for raw and `holden-v1`; append `max_yaw_rate_deg_s` only when `profile == "holden-turn-v1"`.

- [ ] **Step 5: Serialize v2 applied-heading arrays**

Add a `mm_json_write_applied_headings` function symmetric with applied velocities. Change new generate responses to `MM_CHUNK_SCHEMA_V2 = "mm-chunk/v2"` and always emit the new command field. The runtime already makes raw and `holden-v1` applied headings exact requested-heading copies, so no serializer profile branch or duplicate mutable server flag is needed. Keep the v1 constant and parser solely for retained historical artifacts.

- [ ] **Step 6: Add strict dual-version Python decoding**

Copy `sonic/schemas/mm_chunk_v1.schema.json` to v2, change its identifier/title, allow `source_intervals` exactly `5` or `10`, and add the required unit-quaternion applied-heading array. Use an `allOf` conditional so every boundary array has exactly `source_intervals + 1` rows and every step array has exactly `source_intervals` rows; this prevents accepting lengths 6-9. In `schema.py`, preserve the existing v1 parser as `_parse_command_v1`; add:

```python
def _parse_command_v2(value: object, step_count: int) -> Mapping[str, object]:
    fields = {
        "requested_velocity_holden",
        "desired_heading_holden_wxyz",
        "applied_velocity_holden",
        "applied_heading_holden_wxyz",
    }
    source = _exact_object(value, fields, "command")
    output = dict(_parse_shared_command_fields(source, step_count))
    output["applied_heading_holden_wxyz"] = _unit_quaternions(
        _float_array(
            source["applied_heading_holden_wxyz"],
            (step_count, 4),
            "command.applied_heading_holden_wxyz",
        ),
        "command.applied_heading_holden_wxyz",
    )
    return MappingProxyType(output)
```

Dispatch only on exact schema strings before parsing the command. Keep all top-level exact-key checks and detached immutable arrays.

- [ ] **Step 7: Run GREEN, build the real server, and commit**

```bash
/tmp/holden-turn-red/test_mm_chunk_protocol
BUILD_DIR=/tmp/holden-turn-red/server make -C sonic/cpp mm_chunk_server
SONIC_MM_SERVER=/tmp/holden-turn-red/server/mm_chunk_server \
PYTHONPATH="$PWD/sonic/python" \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_schema tests.python.test_mm_chunk_server -v
git add sonic/cpp/mm_chunk_protocol.h sonic/cpp/mm_chunk_json.h \
  sonic/cpp/mm_chunk_server.cpp sonic/schemas/mm_chunk_v2.schema.json \
  sonic/python/mm_sonic/schema.py tests/cpp/test_mm_chunk_protocol.cpp \
  tests/python/test_sonic_schema.py tests/python/test_mm_chunk_server.py
git commit -m "feat: publish capped heading evidence"
```

Expected: C++ pass banner, server build success, and Python `OK` with only registered skips.

---

### Task 4: Python profile selection and versioned run identity

**Files:**
- Modify: `sonic/python/mm_sonic/coordinator.py`
- Modify: `sonic/python/mm_sonic/process.py`
- Modify: `sonic/python/mm_sonic/manual_demo.py`
- Modify: `sonic/python/mm_sonic/manual_evidence.py`
- Modify: `tests/python/test_sonic_coordinator.py`
- Modify: `tests/python/test_sonic_process.py`
- Modify: `tests/python/test_sonic_manual_demo.py`
- Modify: `tests/python/test_sonic_manual_evidence.py`

**Interfaces:**
- Consumes: Task 3 hello and reset records.
- Produces: `SessionConfig(..., movement_model="holden-turn-v1")` and CLI `--movement-model holden-turn-v1`.
- Preserves: manual summary v5 and flat summary v6 parsers/bytes for historical profiles.
- Produces: terrain manual summary v7 and flat manual summary v8 for the six-key turn-profile movement record.

- [ ] **Step 1: Add RED config, process, and CLI tests**

Extend each exact allowed-value assertion to the ordered tuple:

```python
("raw", "holden-v1", "holden-turn-v1")
```

Capture reset requests and require the third value to round-trip. The process validator must accept exactly the five-key record for raw/`holden-v1` and exactly the six-key record for `holden-turn-v1`; reject a max-rate key on an old profile, a missing/wrong/bool/non-finite max rate on the turn profile, and any request/response profile mismatch. Require startup to reject hello capability omission before reset.

- [ ] **Step 2: Add RED versioned summary tests**

Keep the existing byte fixtures for summary v5 and flat v6 unchanged. Add turn-profile round trips requiring schema `mm-sonic-manual-demo/v7` for terrain and `mm-sonic-manual-demo/v8` for flat, with `max_yaw_rate_deg_s == 120.0`. Reject cross-version field sets and prove old parsers still reject the new schemas rather than weakening their exact contracts.

- [ ] **Step 3: Run RED Python suites**

```bash
PYTHONPATH="$PWD/sonic/python" \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_coordinator \
  tests.python.test_sonic_process \
  tests.python.test_sonic_manual_evidence \
  tests.python.test_sonic_manual_demo -v
```

Expected: failures identify the missing third profile and new summary serializers.

- [ ] **Step 4: Implement profile-specific movement identity validation**

Centralize the three allowed strings in a tuple local to each existing validation layer; do not add a cross-package dependency. Preserve five-key historical validation. For the turn record, require Python numeric type but reject `bool`, convert to float, require finite, and compare exactly to `120.0`.

- [ ] **Step 5: Add new summary serializers without changing old ones**

Add `manual_summary_v7_bytes`/`parse_manual_summary_v7` by reusing the validated v5 body and replacing only schema plus the turn movement validator. Add `manual_flat_summary_v8_bytes`/`parse_manual_flat_summary_v8` by reusing the validated v6 body likewise. In `manual_demo`, select v5/v6 for historical profiles and v7/v8 only for `holden-turn-v1`; update its internal post-write schema assertion to match the selected profile and scene kind.

- [ ] **Step 6: Run GREEN and commit**

Run the Step 3 suites. Expected: all report `OK`.

```bash
git add sonic/python/mm_sonic/coordinator.py sonic/python/mm_sonic/process.py \
  sonic/python/mm_sonic/manual_demo.py sonic/python/mm_sonic/manual_evidence.py \
  tests/python/test_sonic_coordinator.py tests/python/test_sonic_process.py \
  tests/python/test_sonic_manual_demo.py \
  tests/python/test_sonic_manual_evidence.py
git commit -m "feat: expose turn-capped movement profile"
```

---

### Task 5: Responsive first/last applied-heading evidence

**Files:**
- Modify: `sonic/python/mm_sonic/boundary_trace.py`
- Modify: `sonic/python/mm_sonic/responsive_wiring.py`
- Modify: `tests/python/test_sonic_boundary_trace.py`
- Modify: `tests/python/test_sonic_responsive_wiring.py`
- Modify: `tests/python/test_sonic_manual_demo.py`

**Interfaces:**
- Produces: `AcceptedChunk.applied_heading_mujoco_wxyz_first` and `.applied_heading_mujoco_wxyz_last`.
- Produces: identical immutable fields on `BoundaryTrace`.
- Produces: `mm-sonic-responsive-boundary-trace/v3` JSON records with both quaternion fields.

- [ ] **Step 1: Add RED endpoint and trace tests**

Create a fake validated v2 chunk whose first/last applied Holden headings are known planar unit quaternions. Require `_applied_heading_endpoints_mujoco` to call `holden_to_mujoco_quaternions`, return two finite unit width-four tuples, and reject missing, empty, wrong-width, non-finite, or non-unit input. Extend `AcceptedChunk`, `BoundaryTrace`, delayed-release binding, and exact JSON assertions with:

```python
"applied_heading_mujoco_wxyz_first": [1.0, 0.0, 0.0, 0.0],
"applied_heading_mujoco_wxyz_last": [0.99912283, 0.0, 0.0, -0.04187565],
```

- [ ] **Step 2: Run focused tests to verify RED**

```bash
PYTHONPATH="$PWD/sonic/python" \
  /home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_sonic_boundary_trace \
  tests.python.test_sonic_responsive_wiring \
  tests.python.test_sonic_manual_demo -v
```

Expected: the constructors and trace schema lack the heading endpoints.

- [ ] **Step 3: Extract and validate endpoints before publication**

Import `holden_to_mujoco_quaternions` beside the vector transform. Add:

```python
def _applied_heading_endpoints_mujoco(
    checked: object,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    command = getattr(checked, "command", None)
    if command is None or "applied_heading_holden_wxyz" not in command:
        raise ContractError(
            "validated turn-profile chunk must expose applied headings"
        )
    rows = np.asarray(command["applied_heading_holden_wxyz"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 4:
        raise ContractError("applied headings must be a nonempty Nx4 array")
    converted = holden_to_mujoco_quaternions(rows[[0, -1]])
    return tuple(map(float, converted[0])), tuple(map(float, converted[1]))
```

Call this after source validation and before target preparation/publication. Carry the fields through `AcceptedChunk`, `build_boundary_trace`, and `BoundaryTrace`. Its existing quaternion validation must require finite width-four unit values within the repository's `1.0e-5` tolerance.

- [ ] **Step 4: Publish trace v3 and keep v2 historical evidence intact**

Add both lists to `trace_record` and change only the new serializer's schema string to `mm-sonic-responsive-boundary-trace/v3`. Tests that construct retained v2 JSON remain unchanged; production new turn-profile traces use v3.

- [ ] **Step 5: Run GREEN and commit**

Run the Step 2 suites. Expected: all report `OK`.

```bash
git add sonic/python/mm_sonic/boundary_trace.py \
  sonic/python/mm_sonic/responsive_wiring.py \
  tests/python/test_sonic_boundary_trace.py \
  tests/python/test_sonic_responsive_wiring.py \
  tests/python/test_sonic_manual_demo.py
git commit -m "feat: trace capped heading endpoints"
```

---

### Task 6: Protected automated qualification

**Files:**
- Modify only the Task 1-5 files when a new failing regression demonstrates a defect.

**Interfaces:**
- Consumes: all production and test interfaces from Tasks 1-5.
- Produces: one isolated server binary, warning-strict test logs, Python suite logs, protected parity hashes, and a clean candidate commit.

- [ ] **Step 1: Compile and run the complete affected C++ catalog**

```bash
set -euo pipefail
BUILD=/tmp/holden-turn-final-cpp
rm -rf "$BUILD"
mkdir -p "$BUILD"
for name in test_g1_turn_model test_g1_movement_model \
  test_g1_controller_state test_g1_runtime test_mm_chunk_protocol; do
  c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/${name}.cpp" -o "$BUILD/$name"
  "$BUILD/$name"
done
BUILD_DIR="$BUILD/server" make -C sonic/cpp mm_chunk_server
```

Expected: five pass banners and a successful server build.

- [ ] **Step 2: Run the complete affected Python catalog against that server**

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/sonic/python${PYTHONPATH:+:$PYTHONPATH}"
export SONIC_MM_SERVER=/tmp/holden-turn-final-cpp/server/mm_chunk_server
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

Expected: all tests `OK`; only registered environment skips are permitted.

- [ ] **Step 3: Run protected parity and numeric gates**

Use the existing deterministic raw and `holden-v1` fixture payloads from `test_mm_chunk_server.py`. Generate them with the new server, remove only build-commit identity, and compare every numeric array as binary32/boolean/integer exact. Then generate a 180-degree turn profile prefix and assert from JSON:

```python
assert max(abs(delta_yaw_deg)) <= 4.8001
assert requested_heading_bits == original_request_bits
assert first_applied_heading_bits == first_runtime_heading_bits
assert last_applied_heading_bits == last_runtime_heading_bits
```

Expected: historical arrays are bit-identical and capped evidence agrees end-to-end.

- [ ] **Step 4: Record the server identity**

```bash
sha256sum /tmp/holden-turn-final-cpp/server/mm_chunk_server
git rev-parse HEAD
git status --short
```

Expected: a durable binary hash, the candidate commit, and a clean worktree.

---

### Task 7: Isolated raw/velocity-only/turn-capped live A/B/C

**Files:**
- Create: `docs/superpowers/results/2026-07-21-holden-turn-cap-abc.md`

**Interfaces:**
- Consumes: `/tmp/holden-turn-final-cpp/server/mm_chunk_server` and the existing paired GEAR/simulator launch configuration.
- Produces: three comparable run roots and a supported/rejected/inconclusive verdict.

- [ ] **Step 1: Launch each condition inside one isolated network namespace**

For each profile `raw`, `holden-v1`, and `holden-turn-v1`, launch both simulator and manual demo through the existing paired launcher inside:

```bash
sudo -n unshare --net --fork --kill-child /bin/bash -lc '
  ip link set lo up
  exec runuser -u ubuntu -- env DISPLAY=:1 PYTHONPATH=sonic/python \
    /home/ubuntu/miniconda3/envs/diffsim/bin/python -B \
    -m mm_sonic.manual_demo \
    --mode interactive --input-source x11 --onscreen --responsive \
    --responsive-source-intervals 5 --movement-model PROFILE \
    --mm-server /tmp/holden-turn-final-cpp/server/mm_chunk_server \
    --chunks 10000 --output-root RUN_ROOT
'
```

Replace `PROFILE` and `RUN_ROOT` with the exact condition. Retain simulator/GEAR logs in each run root. Do not launch either half outside the namespace.

- [ ] **Step 2: Execute the same abrupt reversal protocol**

After LIVE, complete three W-to-S-to-W cycles and three A-to-D-to-A cycles. Hold every direction long enough to reach its requested target; release no intermediate smoothing keys. End cleanly with X. Record fall reset, GEAR safety stop, simulator time reset, process crash, or frozen window as a failed condition rather than discarding it.

- [ ] **Step 3: Verify the cap and compare behavior from durable evidence**

For each transition report requested heading, first/last MM-applied heading, maximum per-source-step yaw delta, generated virtual-root displacement, observed pelvis displacement when available, MM generation latency, prefix completion latency, and real-time factor. Require the turn profile's applied headings to remain within `120.01 deg/s`; missing observed state remains unavailable, never zero.

- [ ] **Step 4: Record the user judgment and verdict**

Ask the user only after all three runs are live-qualified whether `holden-turn-v1` is clearly better than velocity-only `holden-v1`. Mark:

- `supported` only when all six abrupt-cycle groups complete without safety/process failure, the cap is proven, and the user judges it clearly better;
- `rejected` when the cap is proven but behavior is unchanged or worse;
- `inconclusive` when contamination, missing physical evidence, or a runtime failure prevents comparison.

Do not retune the yaw rate within this experiment after an unchanged/worse result.

- [ ] **Step 5: Write and commit the result**

The result document must include run roots, exact commit, binary path/hash, reset movement records, automated test counts, reversal tables, safety outcomes, user judgment, and verdict.

```bash
git add docs/superpowers/results/2026-07-21-holden-turn-cap-abc.md
git commit -m "docs: record Holden turn-cap A/B/C"
```
