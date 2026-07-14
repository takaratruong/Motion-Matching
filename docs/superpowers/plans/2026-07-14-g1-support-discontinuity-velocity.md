# G1 Support Discontinuity Velocity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `support_frame_update` from differentiating discontinuous support targets into artificial vertical velocity while preserving continuous target sampling and spring continuity.

**Architecture:** Classify a support target discontinuity before estimating its nominal velocity. Discontinuous and held targets rebase with zero nominal velocity; continuous same-source targets retain the existing finite-difference velocity and spring behavior.

**Tech Stack:** C++17 header-only runtime code, Daniel Holden's exact spring helpers, standalone C++ regression executable, GCC strict/optimized/sanitized builds, Raylib controller, Python CSV evidence checks.

## Global Constraints

- Fixed runtime update remains binary32 `0.04 s` at 25 Hz.
- Preserve the exact 31-dimensional matcher, database search, selected frames, trajectory, simulation XZ, terrain weights, and acceptance thresholds.
- Support remains a single downstream world-Y transform; IK remains disabled.
- Do not add velocity clamps, height snaps, halflife tuning, runtime options, or matcher changes.
- Modify only `support_runtime.h` and `tests/cpp/test_support_runtime.cpp` for the implementation commit.
- Preserve the uncommitted Task 10 changes in `resources/check_g1_runtime_log.py` and `tests/python/test_runtime_log.py` without modification or staging.
- Never stage or modify `resources/database.bin` or `resources/features.bin`; their SHA-256 values must remain `28207d915847baac7a93e63545577b277f7c4300d839782eff48260b358b17d3` and `e6564503e214be36d3b7912efc19997a9222d85ca83f7227c0b1e7cedce8c73e`.
- Keep the existing visualizer PID `1697070` alive. Build and execute candidates only under `/tmp`; do not replace the visualizer during this correction.
- Treat the stopped-tail matcher loop, remaining Hips transition jumps, support/source alignment, mixed endpoint, and safe-stop clearance as separate failures. Do not hide them or claim Task 10 complete.

---

### Task 1: Discontinuity-Aware Support Velocity

**Files:**
- Modify: `tests/cpp/test_support_runtime.cpp`
- Modify: `support_runtime.h:173-184`
- Verify only: `resources/check_g1_runtime_log.py`
- Generated, never committed: `/tmp/test_support_runtime_*`, `/tmp/controller_g1_support_discontinuity`, `/tmp/g1-multiscene-runtime/gate-c-stairs-shallow__ascent-landing-descent-support-fix-w4.csv`

**Interfaces:**
- Consumes: `support_frame_update(support_frame_state&, const support_observation&, bool, float, char*, int) -> bool` and `decay_spring_damper_exact(float&, float&, float, float) -> void`.
- Produces: unchanged public signatures; above-threshold or source-changing support targets use `nominal_velocity == 0.0f`, while same-source target deltas with magnitude at most `0.02f` retain `target_delta / dt`.

- [ ] **Step 1: Replace the above-boundary expectation with the approved zero-derivative contract**

In `test_update_rebases_only_above_nominal_discontinuity_boundary`, retain the exact-boundary half unchanged and replace the above-boundary fixture and expectations with:

```cpp
    const float above = std::nextafter(
        boundary, std::numeric_limits<float>::infinity());
    check(above > boundary, "above-boundary fixture");
    support_frame_state discontinuous;
    support_frame_reset(discontinuous, 0.0f);
    discontinuous.source = support_left;
    float expected_offset_height = -above;
    float expected_offset_velocity = 0.0f;
    decay_spring_damper_exact(
        expected_offset_height, expected_offset_velocity, 0.10f, dt);

    check(support_frame_update(discontinuous,
          observation(0.0f, above, 0.0f, true, false), false,
          dt, error, sizeof(error)), error);
    close(discontinuous.nominal_height, above, 0.0f,
          "above-boundary discontinuity nominal target");
    check(float_bits(discontinuous.nominal_velocity) == float_bits(0.0f),
          "above-boundary discontinuity zeroes nominal velocity");
    close(discontinuous.offset_height, expected_offset_height, 1e-7f,
          "above-boundary discontinuity rebases height before decay");
    close(discontinuous.offset_velocity, expected_offset_velocity, 1e-7f,
          "above-boundary discontinuity rebases velocity before decay");
    close(discontinuous.height, above + expected_offset_height, 1e-7f,
          "above-boundary discontinuity decays from prior height");
    close(discontinuous.velocity, expected_offset_velocity, 1e-7f,
          "above-boundary discontinuity decays from prior velocity");
    check(float_bits(discontinuous.height) != float_bits(above),
          "above-boundary discontinuity avoids a target snap");
```

This must preserve the preceding exact `0.02f` assertions, including `nominal_velocity == boundary / dt` and zero offset bits.

- [ ] **Step 2: Add the observed same-source airborne regression**

Add this test immediately after the boundary test:

```cpp
static void test_same_source_airborne_discontinuities_do_not_create_velocity()
{
    const float dt = 1.0f / 25.0f;
    const float targets[] = {-0.0885f, 0.1600f};
    char error[256] = {};
    support_frame_state state;
    support_frame_reset(state, 0.0f);
    state.source = support_airborne_root;
    state.airborne_frames = 2;

    float previous_height = state.height;
    for (const float target : targets) {
        check(support_frame_update(state,
              observation(target, 0.0f, 0.0f, false, false), false,
              dt, error, sizeof(error)), error);
        check(state.source == support_airborne_root,
              "observed discontinuity remains same-source airborne root");
        check(float_bits(state.nominal_velocity) == float_bits(0.0f),
              "same-source discontinuity zeroes nominal velocity");
        check(terrain_float_is_finite(state.height) &&
              terrain_float_is_finite(state.velocity),
              "same-source discontinuity output remains finite");
        const float lower = std::fmin(previous_height, target) - 1e-6f;
        const float upper = std::fmax(previous_height, target) + 1e-6f;
        check(state.height >= lower && state.height <= upper,
              "same-source discontinuity first frame remains bounded");
        check(std::fabs(state.velocity) < 1.0f,
              "same-source discontinuity avoids artificial velocity spike");
        previous_height = state.height;
    }
}
```

Register it in `main()` immediately after `test_update_rebases_only_above_nominal_discontinuity_boundary();`:

```cpp
    test_same_source_airborne_discontinuities_do_not_create_velocity();
```

- [ ] **Step 3: Run the focused test and prove RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_support_runtime.cpp -o /tmp/test_support_runtime_red
/tmp/test_support_runtime_red
```

Expected: compilation succeeds and execution exits nonzero at `above-boundary discontinuity zeroes nominal velocity` or `same-source discontinuity zeroes nominal velocity`. If it passes before production code changes, stop because the regression does not exercise the observed bug.

- [ ] **Step 4: Classify discontinuity before estimating target velocity**

In `support_frame_update`, replace the current `source_changed`, `target_velocity`, and `discontinuity` declarations with exactly:

```cpp
    const bool source_changed = source_frame_changed || source != next.source;
    const float target_delta = target - next.nominal_height;
    const bool discontinuity = source_changed ||
        std::fabs(target_delta) > 0.02f;
    const float target_velocity = source == support_held || discontinuity
        ? 0.0f
        : target_delta / dt;
```

Leave the following rebase/continuous branches, source assignment, spring decay, finite checks, and transactional state assignment unchanged.

- [ ] **Step 5: Run focused GREEN in debug, strict, production, and sanitizer configurations**

Run:

```bash
g++ -std=c++17 -O0 -g -I. tests/cpp/test_support_runtime.cpp \
  -o /tmp/test_support_runtime_debug
/tmp/test_support_runtime_debug
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_support_runtime.cpp -o /tmp/test_support_runtime_strict
/tmp/test_support_runtime_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_support_runtime.cpp -o /tmp/test_support_runtime_release
/tmp/test_support_runtime_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_support_runtime.cpp \
  -o /tmp/test_support_runtime_san
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1 /tmp/test_support_runtime_san
```

Expected: all four executions exit `0`; strict compilation is warning-free; ASan/UBSan emits no diagnostic. Existing contact/source transition tests, exact-boundary behavior, smooth-ramp monotonicity, transactional error handling, and horizontal-only helpers all remain green.

- [ ] **Step 6: Verify scope, protected inputs, and the uncommitted checker suite**

Run:

```bash
git diff --check -- support_runtime.h tests/cpp/test_support_runtime.cpp
git diff --name-only -- support_runtime.h tests/cpp/test_support_runtime.cpp
git status --short resources/check_g1_runtime_log.py \
  tests/python/test_runtime_log.py resources/database.bin \
  resources/features.bin
sha256sum resources/database.bin resources/features.bin
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
```

Expected: the owned implementation diff contains only `support_runtime.h` and `tests/cpp/test_support_runtime.cpp`; the already-dirty Task 10 Python files remain present but unchanged by this task; hashes match the Global Constraints; all 59 current Python checker tests pass.

- [ ] **Step 7: Build a fresh controller candidate without replacing the visualizer**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -Wno-unused-result -Wno-class-memaccess -Wno-sign-compare \
  -Wno-pedantic -Wno-unused-parameter -Wno-missing-field-initializers \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_g1_support_discontinuity \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
kill -0 1697070
```

Expected: build exits `0` with no output and PID `1697070` is still alive.

- [ ] **Step 8: Rerun the exact shallow-stairs treatment and compare the impulse**

Run:

```bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=stairs-shallow MM_TEST_MODE=route \
  MM_TEST_ROUTE=ascent-landing-descent MM_TEST_FRAMES=800 \
  MM_TERRAIN_WEIGHT=4 \
  MM_LOG=/tmp/g1-multiscene-runtime/gate-c-stairs-shallow__ascent-landing-descent-support-fix-w4.csv \
  /tmp/controller_g1_support_discontinuity
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import csv

old_path = "/tmp/g1-multiscene-runtime/gate-c-stairs-shallow__ascent-landing-descent-w4.csv"
new_path = "/tmp/g1-multiscene-runtime/gate-c-stairs-shallow__ascent-landing-descent-support-fix-w4.csv"
with open(old_path, newline="") as stream:
    old = list(csv.DictReader(stream))
with open(new_path, newline="") as stream:
    new = list(csv.DictReader(stream))
assert len(old) == len(new) == 800
for key in ("query_bits_hex", "query_database_frame", "selected_database_frame",
            "database_frame", "simulation_x", "simulation_z"):
    assert [row[key] for row in old] == [row[key] for row in new], key
row = 349
before, after = new[row - 1], new[row]
support_step = abs(
    (float(after["support_retargeted_hips_y"]) -
     float(before["support_retargeted_hips_y"])) -
    (float(after["raw_selected_hips_y"]) -
     float(before["raw_selected_hips_y"])))
print("old_row349_support_velocity", old[row]["support_velocity"])
print("new_row349_support_velocity", new[row]["support_velocity"])
print("new_row349_non_source_hips_step", support_step)
assert abs(float(old[row]["support_velocity"])) > 5.0
assert abs(float(new[row]["support_velocity"])) < 2.0
assert abs(float(new[row]["support_velocity"])) < \
       abs(float(old[row]["support_velocity"]))
assert support_step <= 0.05
PY
```

Expected: controller exits `0`; the matcher/query and horizontal simulation fields remain row-for-row identical; old row 349 support velocity exceeds `5 m/s`; new row 349 is both lower than the old value and below `2 m/s`; its non-source Hips step is at most `0.05 m`.

- [ ] **Step 9: Re-run the frozen Gate C checker without weakening it**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-multiscene-runtime/gate-c-stairs-shallow__ascent-landing-descent-support-fix-w4.csv \
  --gate-c \
  --compare-control \
  /tmp/g1-multiscene-runtime/gate-c-stairs-shallow__ascent-landing-descent-w0.csv
```

Expected at this narrow stage: the artificial row-349 impulse is absent, but Gate C still exits `1` on a separately documented failure such as the control stopped-tail sub-stride loop, an earlier Hips continuity event, or support alignment. Record the exact remaining diagnostic; do not edit the checker, controller thresholds, matcher, or safe-stop code in this task.

- [ ] **Step 10: Commit only the approved implementation files**

Run:

```bash
git add support_runtime.h tests/cpp/test_support_runtime.cpp
git diff --cached --name-only
git diff --cached --check
git commit -m "fix: ignore discontinuous support target velocity"
```

Expected: the staged-name list contains exactly the two owned C++ files. The commit succeeds; Task 10 Python checker/test edits and protected generated resources remain unstaged.

- [ ] **Step 11: Preserve evidence for independent review**

Report the RED diagnostic, all four GREEN configurations, exact hashes, strict controller build, row-349 before/after values, frozen Gate C residual diagnostic, commit hash, and `kill -0 1697070` result. Do not claim Task 10 complete or replace the running visualizer.
