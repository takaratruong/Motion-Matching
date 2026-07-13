# G1 Multiscene Support Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the published G1 motion/support and multiscene terrain packs in the Holden runtime, add continuous contact-aware world support and certified safe stopping, and pass deterministic multilevel, traversability, switching, and terrain-weight acceptance gates with IK disabled.

**Architecture:** The motion pack remains immutable and is loaded once. Renderer-independent C++ modules strictly validate support, walkability, manifest, index, and scene metadata; the Raylib controller owns only a transactional active-scene/model handle and a resettable dynamic controller state. Support retargeting is a scalar world-Y transform downstream of unchanged 31-dimensional matching, while adjustment, clamping, and traversal remain planar XZ operations.

**Tech Stack:** C++17; existing `array.h`, `vec.h`, `quat.h`, `spring.h`, `database.h`, `terrain_runtime.h`, and G1 skeleton/FK functions; Raylib/Raygui from `/home/ubuntu/apps`; Python 3 standard-library CSV acceptance checker; `DISPLAY=:1` for deterministic visual runs.

## Global Constraints

- This plan starts only after every task in `docs/superpowers/plans/2026-07-13-g1-scene-artifacts.md` passes, including its Gate A baseline logger/checker, G1HF/v2 fixed-diagonal loader and sampling parity, full 1,770-clip motion-pack rebuild, support sidecar, required scene publication, hashes, routes, and surface-parity gates.
- Consume, do not regenerate, `resources/g1_terrain/{database.bin,terrain_features.bin,terrain_support.bin,manifest.json,validation.json}` and `resources/g1_terrain/scenes/{index.json,<scene-id>/{scene.json,terrain.bin,terrain.obj,walkability.bin}}`.
- Preserve the original `resources/database.bin`, `resources/features.bin`, every generated/user-owned resource, and the existing dirty worktree. Never stage or commit generated artifacts, logs, videos, controller binaries, or unrelated user changes.
- Keep Daniel Holden's database matcher, full-pose inertialization, and fixed `25 Hz` runtime authoritative. Keep `dt = 1.0f / 25.0f`, database horizons `8`, `17`, and `25`, and live prediction intervals `1.0f / 3.0f`.
- Preserve the exact `27 + 4 = 31` feature order and normalization. `terrain_support.bin` is placement metadata only and must never enter features, bounds, costs, or search selection.
- Load `database.bin`, `terrain_features.bin`, and `terrain_support.bin` exactly once. Scene changes may replace only scene metadata, G1HF/v2 heightfield, G1WM/v1 walkability, and the Raylib terrain model.
- Keep learned motion matching disabled. Keep inverse kinematics compile-time disabled throughout this plan; no task may add `MM_IK`, contact locking, foot orientation, swing clearance, or any other downstream IK correction.
- Consequently, contact-lock/target-normal/IK-correction and capsule-clearance
  CSV columns belong to the later Gate E plan. That plan may append after this
  plan's locked runtime suffix; this plan must not fabricate nonzero IK state.
- Support retargeting is the accepted IK-off baseline. It may assign world Y to `G1_Simulation`, but it may not change selected frames, ranges, costs, query values/points, or local recorded joint geometry.
- Root adjustment and clamping may change XZ and yaw only. Their logged Y displacement must remain bit-exact `0.0f` on every frame.
- A blocked or non-finite runtime state must stop safely or exit through the normal diagnostic/cleanup path. It must never substitute zero terrain, lift the body through an obstacle, partially commit a scene, call `_Exit`, or bypass `UnloadModel`, log close, and `CloseWindow`.
- Startup scene selection is `MM_TERRAIN_SCENE=<id>` with default `grail-curb-default`. Unknown IDs are errors; scene IDs must never be interpreted as arbitrary paths.
- The optional G1 visual mesh is outside this plan and remains deferred until terrain IK and all final gates pass. Continue drawing the existing diagnostic skeleton.

Before editing, record the user-owned baseline once:

```bash
git status --short > /tmp/g1_runtime_status_before.txt
sha256sum resources/database.bin resources/features.bin \
  > /tmp/g1_runtime_legacy_before.sha256
```

Expected: both legacy resources hash successfully. Keep these files until the
final preservation audit; never clean the worktree to shorten the status.

---

## File Map and Ownership

- `terrain_runtime.h`: consume the scene-artifact plan's strict G1HF/v1-v2 fixed-diagonal surface implementation; add strict transactional G1SP/v1 and G1WM/v1 loaders plus walkability lookup/sweep primitives.
- `json_runtime.h` and `sha256.h`: strict renderer-independent JSON parsing
  and exact-byte digest verification for manifest/index/scene contracts.
- `scene_runtime.h`: strict JSON DOM, SHA-256 verification, motion-manifest contract, ordered scene-index/catalog contract, scene metadata/bounds/routes, safe path construction, and renderer-independent scene-pack loading.
- `support_runtime.h`: contact-aware support observations, two-frame airborne hold, critically damped root fallback, transition rebase, world-Y application, and horizontal-only position adjustment/clamping helpers.
- `route_runtime.h`: fixed-frame open-loop metadata route commands shared by
  treatment/control evidence.
- `g1_runtime_diagnostics.h`: one finite checked snapshot consumed by both CSV
  and UI.
- `scene_switch.h` and `cleanup_runtime.h`: generic candidate/model transaction
  plus atomic evidence for normal log/model/window cleanup.
- `g1_controller_state.h`: every dynamic matcher, inertializer, trajectory, simulation, contact-diagnostic, support, traversal, and deterministic-route field that must reset together at a scene spawn.
- `controller.cpp`: one-time motion-pack load, startup selection, fixed-update orchestration, Raylib model transaction, UI, rendering, log population, controlled exit, and normal cleanup.
- `motion_match_log.h`: extend the exact logger produced by the scene-artifact plan with scene, support, planar adjustment/clamp, traversal, reset, and cleanup columns.
- `resources/check_g1_runtime_log.py`: extend the exact Gate A checker produced by the scene-artifact plan with Gate C/D/F and terrain-weight A/B assertions.
- `tests/cpp/test_terrain_runtime.cpp`: G1SP/G1WM corruption, parity, lookup, and sweep tests alongside the inherited G1HF/v2 tests.
- `tests/cpp/test_scene_runtime.cpp`: manifest/index/scene JSON, SHA, bounds, route, path, catalog, and transactional candidate tests.
- `tests/cpp/test_support_runtime.cpp`: support selection, airborne fallback, transition continuity, elevated landing/ramp/descent, and horizontal-only adjustment/clamp tests.
- `tests/cpp/test_g1_controller_state.cpp`: full reset and scene-spawn state tests independent of Raylib.
- `tests/cpp/test_scene_switch.cpp`, `test_support_matching.cpp`,
  `test_route_runtime.cpp`, and `test_cleanup_runtime.cpp`: transaction,
  matcher-isolation, deterministic-input, and cleanup ownership regressions.
- `tests/python/test_runtime_log.py`: synthetic Gate C/D/F and terrain A/B checker regressions.
- Generated evidence under `/tmp/g1-multiscene-runtime/`: never committed; deterministic CSVs and captured stderr used by acceptance commands.

## Prerequisite Interface Audit

Before Task 1, compare the completed scene-artifact plan and implementation against these consumed interfaces. This is a read-only gate, not permission to edit artifact-producing Python or generated resources:

```cpp
// terrain_runtime.h, produced by the scene-artifact plan
struct heightfield {
    uint32_t version;            // 1 for migration, 2 for published scenes
    int nx, nz;
    float origin_x, origin_z, cell_size, exterior_height;
    array1d<float> heights;
};
bool heightfield_load(
    heightfield& out, const char* path, char* error, int error_capacity);
float heightfield_sample(const heightfield& field, float x, float z);
vec3 heightfield_normal(const heightfield& field, float x, float z);
```

The published-scene loader must expose G1HF version so `scene_runtime.h` can require `version == 2`; legacy v1 remains readable only by direct migration tests. The inherited `motion_match_log.h` and checker must already reproduce Gate A without changing support behavior. If names differ in the completed prerequisite plan, update this plan document first so all later interfaces use the producer's exact names; do not create adapters with duplicate surface semantics.

---

### Task 1: Strict G1SP/v1 and G1WM/v1 Runtime Loaders

**Files:**
- Modify: `terrain_runtime.h` after the scene-artifact plan's G1HF loader and before centerline helpers
- Modify: `tests/cpp/test_terrain_runtime.cpp` beside the existing binary-format tests

**Interfaces:**
- Consumes: `terrain_file_size`, `terrain_read_exact`, `terrain_decode_u32_le`, `terrain_decode_float_array_le`, `terrain_float_is_finite`, `terrain_size_multiply`, `terrain_size_add`, `terrain_error`, and `heightfield` from `terrain_runtime.h`.
- Produces: `terrain_support_set { array2d<float> values; }`, where columns are root, left toe, right toe.
- Produces: `bool terrain_support_load(terrain_support_set&, const char*, int expected_frames, char*, int)` for exact `G1SP`, version `1`, dimension `3`, and database-frame parity.
- Produces: `walkability_grid { int nx, nz; array1d<uint8_t> cells; }`.
- Produces: `bool walkability_load(walkability_grid&, const char*, const heightfield&, char*, int)` for exact `G1WM`, version `1`, grid parity, and values `0`, `1`, or `2` only.
- Failure is transactional: a rejected file leaves every destination field and payload byte unchanged.

- [ ] **Step 1: Append failing G1SP/G1WM tests**

Add these fixtures and tests before `main` in `tests/cpp/test_terrain_runtime.cpp`, then call both test functions from `main` immediately after the inherited G1HF loader tests:

```cpp
static byte_buffer make_support(
    uint32_t version,
    uint32_t frames,
    uint32_t dimensions,
    const std::vector<float>& values)
{
    byte_buffer out;
    append_bytes(out, "G1SP", 4);
    append_u32_le(out, version);
    append_u32_le(out, frames);
    append_u32_le(out, dimensions);
    for (size_t i = 0; i < values.size(); ++i) append_float_le(out, values[i]);
    return out;
}

static byte_buffer make_walkability(
    uint32_t version,
    uint32_t nx,
    uint32_t nz,
    const std::vector<uint8_t>& cells)
{
    byte_buffer out;
    append_bytes(out, "G1WM", 4);
    append_u32_le(out, version);
    append_u32_le(out, nx);
    append_u32_le(out, nz);
    if (!cells.empty()) append_bytes(out, cells.data(), cells.size());
    return out;
}

static void test_support_loader_is_strict_transactional_and_frame_exact()
{
    const char* path = "/tmp/test_g1sp.bin";
    const std::vector<float> values = {
        0.0f, 0.1f, 0.2f,
        1.0f, 1.1f, 1.2f,
    };
    write_payload(path, make_support(1, 2, 3, values));
    terrain_support_set support;
    char error[256] = {};
    check(terrain_support_load(support, path, 2, error, sizeof(error)), error);
    check(support.values.rows == 2 && support.values.cols == 3,
          "G1SP shape");
    check_close(support.values(1, 2), 1.2f, "G1SP payload");

    support.values.set(9.0f);
    write_payload(path, make_support(2, 2, 3, values));
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP version rejection");
    check(strstr(error, path) && strstr(error, "version"),
          "G1SP version diagnostic");
    check(support.values(0, 0) == 9.0f, "G1SP transaction");

    write_payload(path, make_support(1, 2, 4, values));
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP dimensions rejection");
    write_payload(path, make_support(1, 2, 3, values));
    check(!terrain_support_load(support, path, 3, error, sizeof(error)),
          "G1SP frame parity rejection");

    std::vector<float> nonfinite = values;
    nonfinite[4] = std::numeric_limits<float>::quiet_NaN();
    write_payload(path, make_support(1, 2, 3, nonfinite));
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP finite rejection");

    const byte_buffer valid = make_support(1, 2, 3, values);
    for (size_t size = 0; size < valid.size(); ++size) {
        write_prefix(path, valid, size);
        check(!terrain_support_load(support, path, 2, error, sizeof(error)),
              "G1SP truncation rejection");
    }
    byte_buffer trailing = valid;
    trailing.push_back(0x7f);
    write_payload(path, trailing);
    check(!terrain_support_load(support, path, 2, error, sizeof(error)),
          "G1SP trailing rejection");
}

static void test_walkability_loader_is_strict_transactional_and_grid_exact()
{
    const char* path = "/tmp/test_g1wm.bin";
    heightfield field;
    field.version = 2;
    initialize_heightfield(field, 3, 2, -1.0f, 2.0f, 0.02f, -3.0f);
    field.heights.zero();
    const std::vector<uint8_t> cells = {0, 1, 2, 2, 1, 0};
    write_payload(path, make_walkability(1, 3, 2, cells));

    walkability_grid grid;
    char error[256] = {};
    check(walkability_load(grid, path, field, error, sizeof(error)), error);
    check(grid.nx == 3 && grid.nz == 2 && grid.cells.size == 6,
          "G1WM shape");
    for (int i = 0; i < grid.cells.size; ++i)
        check(grid.cells(i) == cells[static_cast<size_t>(i)], "G1WM payload");

    grid.cells.set(2);
    write_payload(path, make_walkability(2, 3, 2, cells));
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM version rejection");
    check(grid.nx == 3 && grid.nz == 2 && grid.cells(0) == 2,
          "G1WM transaction");

    write_payload(path, make_walkability(1, 2, 2, {0, 1, 2, 0}));
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM grid mismatch rejection");
    write_payload(path, make_walkability(1, 3, 2, {0, 1, 3, 2, 1, 0}));
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM class rejection");

    const byte_buffer valid = make_walkability(1, 3, 2, cells);
    for (size_t size = 0; size < valid.size(); ++size) {
        write_prefix(path, valid, size);
        check(!walkability_load(grid, path, field, error, sizeof(error)),
              "G1WM truncation rejection");
    }
    byte_buffer trailing = valid;
    trailing.push_back(1);
    write_payload(path, trailing);
    check(!walkability_load(grid, path, field, error, sizeof(error)),
          "G1WM trailing rejection");
}
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
```

Expected: compilation fails with undeclared `terrain_support_set`, `terrain_support_load`, `walkability_grid`, and `walkability_load`.

- [ ] **Step 3: Implement the exact transactional loaders**

Add to `terrain_runtime.h` after `heightfield_load`:

```cpp
struct terrain_support_set
{
    array2d<float> values;
};

struct walkability_grid
{
    int nx = 0;
    int nz = 0;
    array1d<uint8_t> cells;
};

static inline bool terrain_support_load(
    terrain_support_set& out,
    const char* path,
    const int expected_frames,
    char* error,
    const int error_capacity)
{
    const char* shown = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0' || expected_frames <= 0) {
        return terrain_error(error, error_capacity,
            "%s: invalid G1SP path or expected frame count %d",
            shown, expected_frames);
    }
    FILE* file = fopen(path, "rb");
    if (file == NULL)
        return terrain_error(error, error_capacity,
            "%s: cannot open (%s)", path, strerror(errno));

    size_t actual_size = 0;
    unsigned char header[16] = {};
    if (!terrain_file_size(file, actual_size) || actual_size < sizeof(header) ||
        !terrain_read_exact(file, header, sizeof(header))) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: truncated G1SP header", path);
    }
    if (memcmp(header, "G1SP", 4) != 0) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: invalid G1SP magic", path);
    }
    const uint32_t version = terrain_decode_u32_le(header + 4);
    const uint32_t frames = terrain_decode_u32_le(header + 8);
    const uint32_t dimensions = terrain_decode_u32_le(header + 12);
    if (version != 1) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: unsupported G1SP version %u (expected 1)", path,
            static_cast<unsigned>(version));
    }
    if (dimensions != 3) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: invalid G1SP dimension %u (expected 3)", path,
            static_cast<unsigned>(dimensions));
    }
    if (frames != static_cast<uint32_t>(expected_frames)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: G1SP frame mismatch: database=%d support=%u", path,
            expected_frames, static_cast<unsigned>(frames));
    }

    size_t count = 0, bytes = 0, expected_size = 0;
    if (!terrain_size_multiply(static_cast<size_t>(frames), 3u, count) ||
        !terrain_size_multiply(count, sizeof(float), bytes) ||
        !terrain_size_add(sizeof(header), bytes, expected_size)) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: G1SP size overflow", path);
    }
    if (actual_size != expected_size) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: %s G1SP payload (expected %zu bytes, got %zu)", path,
            actual_size < expected_size ? "truncated" : "trailing",
            expected_size, actual_size);
    }

    terrain_support_set loaded;
    loaded.values.resize(expected_frames, 3);
    if (!terrain_read_exact(file, loaded.values.data, bytes)) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: truncated G1SP values", path);
    }
    terrain_decode_float_array_le(loaded.values.data, count);
    for (size_t i = 0; i < count; ++i) {
        if (!terrain_float_is_finite(loaded.values.data[i])) {
            fclose(file);
            return terrain_error(error, error_capacity,
                "%s: G1SP values must be finite (index %zu)", path, i);
        }
    }
    if (!terrain_finish_read(file, path, error, error_capacity)) return false;
    std::swap(out.values.rows, loaded.values.rows);
    std::swap(out.values.cols, loaded.values.cols);
    std::swap(out.values.data, loaded.values.data);
    return true;
}

static inline bool walkability_load(
    walkability_grid& out,
    const char* path,
    const heightfield& field,
    char* error,
    const int error_capacity)
{
    const char* shown = path != NULL ? path : "<null>";
    if (path == NULL || path[0] == '\0' || field.nx < 2 || field.nz < 2) {
        return terrain_error(error, error_capacity,
            "%s: invalid G1WM path or reference heightfield", shown);
    }
    FILE* file = fopen(path, "rb");
    if (file == NULL)
        return terrain_error(error, error_capacity,
            "%s: cannot open (%s)", path, strerror(errno));

    size_t actual_size = 0;
    unsigned char header[16] = {};
    if (!terrain_file_size(file, actual_size) || actual_size < sizeof(header) ||
        !terrain_read_exact(file, header, sizeof(header))) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: truncated G1WM header", path);
    }
    if (memcmp(header, "G1WM", 4) != 0) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: invalid G1WM magic", path);
    }
    const uint32_t version = terrain_decode_u32_le(header + 4);
    const uint32_t nx = terrain_decode_u32_le(header + 8);
    const uint32_t nz = terrain_decode_u32_le(header + 12);
    if (version != 1) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: unsupported G1WM version %u (expected 1)", path,
            static_cast<unsigned>(version));
    }
    if (nx != static_cast<uint32_t>(field.nx) ||
        nz != static_cast<uint32_t>(field.nz)) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: G1WM grid %ux%u does not match G1HF %dx%d", path,
            static_cast<unsigned>(nx), static_cast<unsigned>(nz),
            field.nx, field.nz);
    }
    size_t count = 0, expected_size = 0;
    if (!terrain_size_multiply(static_cast<size_t>(nx),
                               static_cast<size_t>(nz), count) ||
        count > static_cast<size_t>(INT_MAX) ||
        !terrain_size_add(sizeof(header), count, expected_size)) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: G1WM size overflow", path);
    }
    if (actual_size != expected_size) {
        fclose(file);
        return terrain_error(error, error_capacity,
            "%s: %s G1WM payload (expected %zu bytes, got %zu)", path,
            actual_size < expected_size ? "truncated" : "trailing",
            expected_size, actual_size);
    }

    walkability_grid loaded;
    loaded.nx = static_cast<int>(nx);
    loaded.nz = static_cast<int>(nz);
    loaded.cells.resize(static_cast<int>(count));
    if (!terrain_read_exact(file, loaded.cells.data, count)) {
        fclose(file);
        return terrain_error(error, error_capacity, "%s: truncated G1WM cells", path);
    }
    for (size_t i = 0; i < count; ++i) {
        if (loaded.cells.data[i] > 2) {
            fclose(file);
            return terrain_error(error, error_capacity,
                "%s: invalid G1WM class %u at cell %zu", path,
                static_cast<unsigned>(loaded.cells.data[i]), i);
        }
    }
    if (!terrain_finish_read(file, path, error, error_capacity)) return false;
    std::swap(out.nx, loaded.nx);
    std::swap(out.nz, loaded.nz);
    std::swap(out.cells.size, loaded.cells.size);
    std::swap(out.cells.data, loaded.cells.data);
    return true;
}
```

- [ ] **Step 4: Run debug, strict, release/fast-math, and sanitizer GREEN**

Run:

```bash
g++ -std=c++17 -O0 -g -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_debug
/tmp/test_terrain_runtime_debug
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime_strict
/tmp/test_terrain_runtime_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime_release
/tmp/test_terrain_runtime_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_terrain_runtime_san
```

Expected: every compile exits `0`; every binary exits `0` with no stdout/stderr and no sanitizer report.

- [ ] **Step 5: Probe the published support and every walkability file**

Extend the inherited optional real-artifact probe in
`test_terrain_runtime.cpp` from `argc == 1 || argc == 3` to the exact
`argc == 1 || argc == 5` contract
`TERRAIN_FEATURES TERRAIN_SUPPORT HEIGHTFIELD WALKABILITY`. Keep its existing
G1TF/G1HF checks, and append the G1SP/G1WM checks:

```cpp
static void probe_support_and_scene(
    const char* support_path,
    const char* terrain_path,
    const char* walkability_path)
{
    char error[512] = {};
    terrain_support_set support;
    check(terrain_support_load(support, support_path, 459682,
          error, sizeof(error)), error);
    check(support.values.rows == 459682 && support.values.cols == 3,
          "published G1SP dimensions");
    heightfield field;
    check(heightfield_load(field, terrain_path, error, sizeof(error)), error);
    check(field.version == 2, "published scenes require G1HF/v2");
    walkability_grid grid;
    check(walkability_load(grid, walkability_path, field,
          error, sizeof(error)), error);
}
```

Use the catalog rather than shell glob order:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('resources/g1_terrain')
ids = json.loads((root/'scenes/index.json').read_text())['scene_ids']
Path('/tmp/g1_scene_pairs.txt').write_text('\n'.join(
    f"{root/'scenes'/scene/'terrain.bin'} {root/'scenes'/scene/'walkability.bin'}"
    for scene in ids) + '\n')
PY
while read -r terrain walkability; do
  /tmp/test_terrain_runtime_strict \
    resources/g1_terrain/terrain_features.bin \
    resources/g1_terrain/terrain_support.bin \
    "$terrain" \
    "$walkability"
done < /tmp/g1_scene_pairs.txt
```

Expected: one exit-`0` probe per ordered scene, `459682x4` G1TF rows and
`459682x3` G1SP rows in every invocation, G1HF/v2 for every published terrain,
and exact G1WM/G1HF grid parity. The implementation calls the inherited
`probe_generated_artifacts(argv[1], argv[3])` and then
`probe_support_and_scene(argv[2], argv[3], argv[4])`; do not retain ambiguous
positional variants.

- [ ] **Step 6: Commit the support and walkability loaders**

```bash
git add terrain_runtime.h tests/cpp/test_terrain_runtime.cpp
git commit -m "feat: load G1 support and walkability artifacts"
```

### Task 2: Strict JSON and SHA-256 Primitives

**Files:**
- Create: `json_runtime.h`
- Create: `sha256.h`
- Create: `tests/cpp/test_scene_runtime.cpp`

**Interfaces:**
- Produces: `json_value` with exact null/bool/number/string/array/object kinds, unique object keys, finite JSON numbers, and source offsets.
- Produces: `bool json_document_load(json_value&, const char*, char*, int)` with an exact 16 MiB input ceiling, full-document consumption, UTF-8/escape validation, duplicate-key rejection, depth limit `64`, and transactional output.
- Produces: `const json_value* json_member(const json_value&, const char*)`; callers must still enforce required fields and exact types.
- Produces: `bool sha256_file_hex(std::string&, const char*, char*, int)` returning exactly 64 lowercase hex characters.
- These primitives parse and hash only; Task 4 owns the G1 semantic contracts.

- [ ] **Step 1: Write failing parser and hash tests**

Create `tests/cpp/test_scene_runtime.cpp`:

```cpp
#include "json_runtime.h"
#include "sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "scene runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static void write_text(const char* path, const std::string& text)
{
    FILE* file = std::fopen(path, "wb");
    check(file != NULL, "open fixture");
    check(std::fwrite(text.data(), 1, text.size(), file) == text.size(),
          "write fixture");
    check(std::fclose(file) == 0, "close fixture");
}

static bool load_text(
    json_value& out, const std::string& text, char error[512])
{
    const char* path = "/tmp/test_scene_json.json";
    write_text(path, text);
    error[0] = '\0';
    return json_document_load(out, path, error, 512);
}

static void test_json_document_is_strict_and_transactional()
{
    char error[512] = {};
    json_value value;
    check(load_text(value,
        "{\"schema\":\"v1\",\"number\":-1.25e2,\"flag\":true,"
        "\"none\":null,\"array\":[0,\"\\u2603\"]}", error), error);
    check(value.kind == json_object, "root object");
    check(json_member(value, "schema") != NULL &&
          json_member(value, "schema")->string_value == "v1", "member lookup");
    check(json_member(value, "number")->number_value == -125.0,
          "number decode");
    check(json_member(value, "array")->array_value[1].string_value ==
          "\xe2\x98\x83", "unicode decode");

    const std::string failures[] = {
        "", "[] trailing", "{\"x\":1,\"x\":2}", "{\"x\":01}",
        "{\"x\":NaN}", "{\"x\":1e999}", "{\"x\":\"\\q\"}",
        "{\"x\":\"\\uD800\"}", "{\"x\":\"\x01\"}",
    };
    for (const std::string& text : failures) {
        json_value prior;
        prior.kind = json_string;
        prior.string_value = "sentinel";
        check(!load_text(prior, text, error), "malformed JSON rejection");
        check(std::strstr(error, "/tmp/test_scene_json.json") != NULL,
              "JSON path diagnostic");
        check(prior.kind == json_string && prior.string_value == "sentinel",
              "JSON transactional failure");
    }

    std::string deep;
    for (int i = 0; i < 65; ++i) deep += '[';
    for (int i = 0; i < 65; ++i) deep += ']';
    check(!load_text(value, deep, error), "JSON depth rejection");
}

static void test_sha256_known_vectors_and_file_errors()
{
    char error[512] = {};
    std::string digest;
    write_text("/tmp/test_sha_empty", "");
    check(sha256_file_hex(digest, "/tmp/test_sha_empty", error, 512), error);
    check(digest ==
          "e3b0c44298fc1c149afbf4c8996fb924"
          "27ae41e4649b934ca495991b7852b855", "empty SHA-256");
    write_text("/tmp/test_sha_abc", "abc");
    check(sha256_file_hex(digest, "/tmp/test_sha_abc", error, 512), error);
    check(digest ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad", "abc SHA-256");
    check(!sha256_file_hex(
          digest, "/tmp/test_sha_missing", error, 512), "missing SHA rejection");
    check(std::strstr(error, "/tmp/test_sha_missing") != NULL,
          "SHA path diagnostic");
}

int main()
{
    test_json_document_is_strict_and_transactional();
    test_sha256_known_vectors_and_file_errors();
    return 0;
}
```

- [ ] **Step 2: Run the new test to verify RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime
```

Expected: compilation fails because `json_runtime.h` does not exist.

- [ ] **Step 3: Implement the strict JSON DOM**

Create `json_runtime.h` with this complete implementation:

```cpp
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

enum json_kind { json_null, json_boolean, json_number, json_string,
                 json_array, json_object };

struct json_value
{
    json_kind kind = json_null;
    bool boolean_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<json_value> array_value;
    std::vector<std::pair<std::string, json_value> > object_value;
    size_t source_offset = 0;
};

static inline const json_value* json_member(
    const json_value& object, const char* key)
{
    if (object.kind != json_object || key == NULL) return NULL;
    for (size_t i = 0; i < object.object_value.size(); ++i)
        if (object.object_value[i].first == key)
            return &object.object_value[i].second;
    return NULL;
}

struct json_parser
{
    const char* path;
    const std::string& text;
    size_t cursor = 0;
    int depth = 0;
    std::string reason;

    json_parser(const char* input_path, const std::string& input)
        : path(input_path), text(input) {}

    bool fail(const char* message)
    {
        if (reason.empty()) reason = message;
        return false;
    }

    void whitespace()
    {
        while (cursor < text.size() &&
               (text[cursor] == ' ' || text[cursor] == '\t' ||
                text[cursor] == '\n' || text[cursor] == '\r')) ++cursor;
    }

    static int hex(const char value)
    {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    }

    bool code_unit(uint32_t& value)
    {
        if (text.size() - cursor < 4) return fail("truncated unicode escape");
        value = 0;
        for (int i = 0; i < 4; ++i) {
            const int digit = hex(text[cursor++]);
            if (digit < 0) return fail("invalid unicode escape");
            value = value * 16u + static_cast<uint32_t>(digit);
        }
        return true;
    }

    static void append_utf8(std::string& out, const uint32_t value)
    {
        if (value <= 0x7fu) out.push_back(static_cast<char>(value));
        else if (value <= 0x7ffu) {
            out.push_back(static_cast<char>(0xc0u | (value >> 6)));
            out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
        } else if (value <= 0xffffu) {
            out.push_back(static_cast<char>(0xe0u | (value >> 12)));
            out.push_back(static_cast<char>(0x80u | ((value >> 6) & 0x3fu)));
            out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
        } else {
            out.push_back(static_cast<char>(0xf0u | (value >> 18)));
            out.push_back(static_cast<char>(0x80u | ((value >> 12) & 0x3fu)));
            out.push_back(static_cast<char>(0x80u | ((value >> 6) & 0x3fu)));
            out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
        }
    }

    bool raw_utf8(std::string& out)
    {
        const unsigned char first = static_cast<unsigned char>(text[cursor++]);
        int tails = 0;
        uint32_t value = 0, minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            tails = 1; value = first & 0x1fu; minimum = 0x80u;
        } else if (first >= 0xe0 && first <= 0xef) {
            tails = 2; value = first & 0x0fu; minimum = 0x800u;
        } else if (first >= 0xf0 && first <= 0xf4) {
            tails = 3; value = first & 0x07u; minimum = 0x10000u;
        } else return fail("invalid UTF-8 lead byte");
        if (text.size() - cursor < static_cast<size_t>(tails))
            return fail("truncated UTF-8");
        for (int i = 0; i < tails; ++i) {
            const unsigned char next = static_cast<unsigned char>(text[cursor++]);
            if ((next & 0xc0u) != 0x80u) return fail("invalid UTF-8 tail byte");
            value = (value << 6) | (next & 0x3fu);
        }
        if (value < minimum || value > 0x10ffffu ||
            (value >= 0xd800u && value <= 0xdfffu))
            return fail("invalid UTF-8 codepoint");
        append_utf8(out, value);
        return true;
    }

    bool string(std::string& out)
    {
        whitespace();
        if (cursor >= text.size() || text[cursor++] != '"')
            return fail("expected string");
        out.clear();
        while (cursor < text.size()) {
            const unsigned char value = static_cast<unsigned char>(text[cursor++]);
            if (value == '"') return true;
            if (value < 0x20u) return fail("control byte in string");
            if (value >= 0x80u) { --cursor; if (!raw_utf8(out)) return false; continue; }
            if (value != '\\') { out.push_back(static_cast<char>(value)); continue; }
            if (cursor >= text.size()) return fail("truncated escape");
            const char escape = text[cursor++];
            if (escape == '"' || escape == '\\' || escape == '/')
                out.push_back(escape);
            else if (escape == 'b') out.push_back('\b');
            else if (escape == 'f') out.push_back('\f');
            else if (escape == 'n') out.push_back('\n');
            else if (escape == 'r') out.push_back('\r');
            else if (escape == 't') out.push_back('\t');
            else if (escape == 'u') {
                uint32_t first = 0;
                if (!code_unit(first)) return false;
                uint32_t codepoint = first;
                if (first >= 0xd800u && first <= 0xdbffu) {
                    if (text.size() - cursor < 6 || text[cursor] != '\\' ||
                        text[cursor + 1] != 'u') return fail("missing low surrogate");
                    cursor += 2;
                    uint32_t second = 0;
                    if (!code_unit(second) || second < 0xdc00u || second > 0xdfffu)
                        return fail("invalid low surrogate");
                    codepoint = 0x10000u + ((first - 0xd800u) << 10) +
                                (second - 0xdc00u);
                } else if (first >= 0xdc00u && first <= 0xdfffu) {
                    return fail("unpaired low surrogate");
                }
                append_utf8(out, codepoint);
            } else return fail("invalid string escape");
        }
        return fail("unterminated string");
    }

    bool number(json_value& out)
    {
        const size_t begin = cursor;
        if (cursor < text.size() && text[cursor] == '-') ++cursor;
        if (cursor >= text.size()) return fail("truncated number");
        if (text[cursor] == '0') ++cursor;
        else if (text[cursor] >= '1' && text[cursor] <= '9')
            while (cursor < text.size() && text[cursor] >= '0' &&
                   text[cursor] <= '9') ++cursor;
        else return fail("invalid number integer");
        if (cursor < text.size() && text[cursor] == '.') {
            ++cursor;
            const size_t digits = cursor;
            while (cursor < text.size() && text[cursor] >= '0' &&
                   text[cursor] <= '9') ++cursor;
            if (cursor == digits) return fail("invalid number fraction");
        }
        if (cursor < text.size() && (text[cursor] == 'e' || text[cursor] == 'E')) {
            ++cursor;
            if (cursor < text.size() && (text[cursor] == '+' || text[cursor] == '-'))
                ++cursor;
            const size_t digits = cursor;
            while (cursor < text.size() && text[cursor] >= '0' &&
                   text[cursor] <= '9') ++cursor;
            if (cursor == digits) return fail("invalid number exponent");
        }
        const std::string token = text.substr(begin, cursor - begin);
        errno = 0;
        char* end = NULL;
        const double parsed = std::strtod(token.c_str(), &end);
        if (errno == ERANGE || end == NULL || *end != '\0' || !std::isfinite(parsed))
            return fail("non-finite or out-of-range number");
        out.kind = json_number;
        out.number_value = parsed;
        return true;
    }

    bool literal(const char* word)
    {
        const size_t size = std::strlen(word);
        if (text.size() - cursor < size || text.compare(cursor, size, word) != 0)
            return fail("invalid literal");
        cursor += size;
        return true;
    }

    bool value(json_value& out)
    {
        whitespace();
        if (cursor >= text.size()) return fail("expected value");
        out.source_offset = cursor;
        if (text[cursor] == '"') {
            out.kind = json_string;
            return string(out.string_value);
        }
        if (text[cursor] == '-' || (text[cursor] >= '0' && text[cursor] <= '9'))
            return number(out);
        if (text[cursor] == 'n') { out.kind = json_null; return literal("null"); }
        if (text[cursor] == 't') {
            out.kind = json_boolean; out.boolean_value = true; return literal("true");
        }
        if (text[cursor] == 'f') {
            out.kind = json_boolean; out.boolean_value = false; return literal("false");
        }
        if (depth >= 64) return fail("JSON nesting exceeds 64");
        if (text[cursor] == '[') {
            out.kind = json_array; ++cursor; ++depth; whitespace();
            if (cursor < text.size() && text[cursor] == ']') {
                ++cursor; --depth; return true;
            }
            while (true) {
                out.array_value.push_back(json_value());
                if (!value(out.array_value.back())) return false;
                whitespace();
                if (cursor < text.size() && text[cursor] == ']') {
                    ++cursor; --depth; return true;
                }
                if (cursor >= text.size() || text[cursor++] != ',')
                    return fail("expected array comma or close");
            }
        }
        if (text[cursor] == '{') {
            out.kind = json_object; ++cursor; ++depth; whitespace();
            if (cursor < text.size() && text[cursor] == '}') {
                ++cursor; --depth; return true;
            }
            while (true) {
                std::string key;
                if (!string(key)) return false;
                for (size_t i = 0; i < out.object_value.size(); ++i)
                    if (out.object_value[i].first == key)
                        return fail("duplicate object key");
                whitespace();
                if (cursor >= text.size() || text[cursor++] != ':')
                    return fail("expected object colon");
                out.object_value.push_back(std::make_pair(key, json_value()));
                if (!value(out.object_value.back().second)) return false;
                whitespace();
                if (cursor < text.size() && text[cursor] == '}') {
                    ++cursor; --depth; return true;
                }
                if (cursor >= text.size() || text[cursor++] != ',')
                    return fail("expected object comma or close");
            }
        }
        return fail("invalid value token");
    }
};

static inline bool json_runtime_error(
    char* error, const int capacity, const char* path,
    const size_t offset, const char* reason)
{
    if (error != NULL && capacity > 0)
        std::snprintf(error, static_cast<size_t>(capacity),
            "%s: JSON error at byte %zu: %s", path, offset, reason);
    return false;
}

static inline bool json_document_load(
    json_value& out, const char* path, char* error, const int error_capacity)
{
    if (path == NULL || path[0] == '\0')
        return json_runtime_error(error, error_capacity,
            path != NULL ? path : "<null>", 0, "invalid path");
    FILE* file = std::fopen(path, "rb");
    if (file == NULL)
        return json_runtime_error(error, error_capacity, path, 0, "cannot open");
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return json_runtime_error(error, error_capacity, path, 0, "cannot size");
    }
    const long end = std::ftell(file);
    static const size_t maximum = 16u * 1024u * 1024u;
    if (end < 0 || static_cast<unsigned long>(end) > maximum ||
        std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        return json_runtime_error(error, error_capacity, path, 0,
            end < 0 ? "cannot size" : "document exceeds 16 MiB");
    }
    std::string text(static_cast<size_t>(end), '\0');
    bool read_failed = !text.empty() &&
        std::fread(&text[0], 1, text.size(), file) != text.size();
    if (std::fclose(file) != 0) read_failed = true;
    if (read_failed)
        return json_runtime_error(error, error_capacity, path, 0, "cannot read");

    json_value loaded;
    json_parser parser(path, text);
    if (!parser.value(loaded))
        return json_runtime_error(error, error_capacity, path,
            parser.cursor, parser.reason.c_str());
    parser.whitespace();
    if (parser.cursor != text.size())
        return json_runtime_error(error, error_capacity, path,
            parser.cursor, "trailing data");
    out = std::move(loaded);
    return true;
}
```

- [ ] **Step 4: Implement streaming SHA-256**

Create `sha256.h`:

```cpp
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

struct sha256_state
{
    uint8_t block[64] = {};
    uint32_t length = 0;
    uint64_t bit_length = 0;
    uint32_t hash[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
};

static inline uint32_t sha256_rotr(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static inline void sha256_transform(sha256_state& state)
{
    static const uint32_t constants[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t words[64] = {};
    for (int i = 0; i < 16; ++i)
        words[i] = (static_cast<uint32_t>(state.block[i * 4]) << 24) |
                   (static_cast<uint32_t>(state.block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(state.block[i * 4 + 2]) << 8) |
                    static_cast<uint32_t>(state.block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = sha256_rotr(words[i - 15], 7) ^
                            sha256_rotr(words[i - 15], 18) ^
                            (words[i - 15] >> 3);
        const uint32_t s1 = sha256_rotr(words[i - 2], 17) ^
                            sha256_rotr(words[i - 2], 19) ^
                            (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a=state.hash[0], b=state.hash[1], c=state.hash[2], d=state.hash[3];
    uint32_t e=state.hash[4], f=state.hash[5], g=state.hash[6], h=state.hash[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t first = h + s1 + choose + constants[i] + words[i];
        const uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t second = s0 + majority;
        h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
    }
    state.hash[0]+=a; state.hash[1]+=b; state.hash[2]+=c; state.hash[3]+=d;
    state.hash[4]+=e; state.hash[5]+=f; state.hash[6]+=g; state.hash[7]+=h;
}

static inline void sha256_update(
    sha256_state& state, const uint8_t* data, const size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        state.block[state.length++] = data[i];
        if (state.length == 64) {
            sha256_transform(state);
            state.bit_length += 512;
            state.length = 0;
        }
    }
}

static inline std::string sha256_finish(sha256_state& state)
{
    uint32_t i = state.length;
    state.block[i++] = 0x80u;
    if (i > 56) {
        while (i < 64) state.block[i++] = 0;
        sha256_transform(state);
        std::memset(state.block, 0, 56);
    } else while (i < 56) state.block[i++] = 0;
    state.bit_length += static_cast<uint64_t>(state.length) * 8u;
    for (int byte = 0; byte < 8; ++byte)
        state.block[63 - byte] = static_cast<uint8_t>(state.bit_length >> (byte * 8));
    sha256_transform(state);
    static const char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int word = 0; word < 8; ++word)
        for (int byte = 0; byte < 4; ++byte) {
            const uint8_t value = static_cast<uint8_t>(
                state.hash[word] >> (24 - byte * 8));
            out[(word * 4 + byte) * 2] = hex[value >> 4];
            out[(word * 4 + byte) * 2 + 1] = hex[value & 15];
        }
    return out;
}

static inline bool sha256_file_hex(
    std::string& out, const char* path, char* error, const int error_capacity)
{
    if (path == NULL || path[0] == '\0') {
        if (error && error_capacity > 0)
            std::snprintf(error, static_cast<size_t>(error_capacity),
                "%s: invalid SHA-256 path", path ? path : "<null>");
        return false;
    }
    FILE* file = std::fopen(path, "rb");
    if (file == NULL) {
        if (error && error_capacity > 0)
            std::snprintf(error, static_cast<size_t>(error_capacity),
                "%s: cannot open for SHA-256 (%s)", path, std::strerror(errno));
        return false;
    }
    sha256_state state;
    uint8_t buffer[64 * 1024];
    while (true) {
        const size_t count = std::fread(buffer, 1, sizeof(buffer), file);
        sha256_update(state, buffer, count);
        if (count != sizeof(buffer)) break;
    }
    bool failed = std::ferror(file) != 0;
    if (std::fclose(file) != 0) failed = true;
    if (failed) {
        if (error && error_capacity > 0)
            std::snprintf(error, static_cast<size_t>(error_capacity),
                "%s: failed while hashing", path);
        return false;
    }
    std::string digest = sha256_finish(state);
    out.swap(digest);
    return true;
}
```

- [ ] **Step 5: Run JSON/SHA tests in all native configurations**

Run:

```bash
g++ -std=c++17 -O0 -g -I. tests/cpp/test_scene_runtime.cpp \
  -o /tmp/test_scene_runtime_debug
/tmp/test_scene_runtime_debug
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_strict
/tmp/test_scene_runtime_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_release
/tmp/test_scene_runtime_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_scene_runtime.cpp \
  -o /tmp/test_scene_runtime_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_scene_runtime_san
```

Expected: all four binaries exit `0`; the strict build emits no warnings and the sanitizer emits no report.

- [ ] **Step 6: Commit strict JSON and hashing**

```bash
git add json_runtime.h sha256.h tests/cpp/test_scene_runtime.cpp
git commit -m "feat: validate runtime JSON and artifact hashes"
```

### Task 3: Contact-Aware Support Frame and Horizontal-Only Root Helpers

**Files:**
- Create: `support_runtime.h`
- Create: `tests/cpp/test_support_runtime.cpp`

**Interfaces:**
- Consumes: `terrain_support_set`, `heightfield_sample`, `G1SP` columns `0=root`, `1=left toe`, `2=right toe`, recorded database contacts, and pre-support FK root/toe XZ.
- Produces: `bool support_observation_build(support_observation&, const terrain_support_set&, int, const heightfield&, vec3, vec3, vec3, bool, bool, char*, int)`.
- Produces: `support_frame_reset`, `support_frame_rebase`, and `support_frame_update` with exact two-frame no-contact hold and `0.10 s` critically damped root fallback/rebase at `25 Hz`.
- Produces: `support_pose_apply`, which copies the inertialized local pose and changes only `G1_Simulation.y`.
- Produces: `horizontal_adjust_character_position`, `horizontal_adjust_character_position_by_velocity`, and `horizontal_clamp_character_position`; all preserve the input Y bit pattern.
- `support_frame_update` is transactional on non-finite input. A source-frame/range transition, contact-source change, or nominal target discontinuity greater than `0.02 m` rebases the scalar inertializer before decay.

- [ ] **Step 1: Write support selection, continuity, and planar-root tests**

Create `tests/cpp/test_support_runtime.cpp`:

```cpp
#include "support_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool value, const char* message)
{
    if (!value) {
        std::fprintf(stderr, "support runtime test failed: %s\n", message);
        std::exit(1);
    }
}

static void close(float actual, float expected, float tolerance, const char* message)
{
    check(terrain_float_is_finite(actual) &&
          std::fabs(actual - expected) <= tolerance, message);
}

static uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static support_observation observation(
    float root, float left, float right, bool left_contact, bool right_contact)
{
    support_observation out = {};
    out.source_height[0] = out.source_height[1] = out.source_height[2] = 0.0f;
    out.runtime_height[0] = root;
    out.runtime_height[1] = left;
    out.runtime_height[2] = right;
    out.delta[0] = root;
    out.delta[1] = left;
    out.delta[2] = right;
    out.contact[0] = left_contact;
    out.contact[1] = right_contact;
    return out;
}

static void test_contact_choice_airborne_hold_and_root_fallback()
{
    const float dt = 1.0f / 25.0f;
    char error[256] = {};
    support_frame_state state;
    support_frame_reset(state, 0.30f);

    check(support_frame_update(state,
          observation(0.10f, 0.24f, 0.38f, true, false), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_left, "left support source");
    close(state.nominal_height, 0.24f, 1e-6f, "left support target");

    check(support_frame_update(state,
          observation(0.10f, 0.24f, 0.38f, true, true), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_both, "double support source");
    close(state.nominal_height, 0.31f, 1e-6f, "double support mean");

    const float held = state.height;
    for (int frame = 0; frame < 2; ++frame) {
        check(support_frame_update(state,
              observation(0.0f, 0.0f, 0.0f, false, false), false,
              dt, error, sizeof(error)), error);
        check(state.source == support_held, "two-frame airborne hold source");
        close(state.nominal_height, 0.31f, 1e-6f, "two-frame airborne hold target");
    }
    check(state.height > 0.0f && state.height <= held + 1e-6f,
          "hold remains bounded");
    check(support_frame_update(state,
          observation(0.0f, 0.0f, 0.0f, false, false), false,
          dt, error, sizeof(error)), error);
    check(state.source == support_airborne_root, "third-frame root fallback");
    check(state.height > 0.0f && state.height < held,
          "root fallback starts continuously");
}

static void test_rebase_preserves_height_and_vertical_velocity()
{
    support_frame_state state;
    support_frame_reset(state, 0.36f);
    state.height = 0.41f;
    state.velocity = 0.17f;
    support_frame_rebase(state, -0.12f, -0.03f);
    close(state.nominal_height + state.offset_height,
          0.41f, 1e-7f, "rebase position continuity");
    close(state.nominal_velocity + state.offset_velocity,
          0.17f, 1e-7f, "rebase velocity continuity");
}

static void test_elevated_landing_ramp_and_descent_do_not_decay_to_zero()
{
    const float dt = 1.0f / 25.0f;
    char error[256] = {};
    support_frame_state state;
    support_frame_reset(state, 0.36f);
    for (int frame = 0; frame < 75; ++frame) {
        check(support_frame_update(state,
              observation(0.36f, 0.36f, 0.36f, true, true), false,
              dt, error, sizeof(error)), error);
        close(state.height, 0.36f, 1e-6f, "three-second elevated landing");
    }
    float previous = state.height;
    for (int frame = 1; frame <= 36; ++frame) {
        const float height = 0.36f + 0.01f * static_cast<float>(frame);
        check(support_frame_update(state,
              observation(height, height, height, true, true), false,
              dt, error, sizeof(error)), error);
        check(state.height >= previous, "ramp support is monotonic");
        previous = state.height;
    }
    for (int frame = 35; frame >= 0; --frame) {
        const float height = 0.01f * static_cast<float>(frame);
        check(support_frame_update(state,
              observation(height, height, height, true, true), false,
              dt, error, sizeof(error)), error);
    }
    for (int frame = 0; frame < 50; ++frame)
        check(support_frame_update(state,
              observation(0.0f, 0.0f, 0.0f, true, true), false,
              dt, error, sizeof(error)), error);
    close(state.height, 0.0f, 0.02f, "descent returns to base level");
}

static void test_nonfinite_update_is_transactional()
{
    support_frame_state state;
    support_frame_reset(state, 0.25f);
    const support_frame_state before = state;
    support_observation bad = observation(0, 0, 0, true, false);
    bad.delta[1] = std::numeric_limits<float>::quiet_NaN();
    char error[256] = {};
    check(!support_frame_update(
          state, bad, false, 1.0f / 25.0f, error, sizeof(error)),
          "nonfinite support rejection");
    check(state.height == before.height && state.velocity == before.velocity &&
          state.nominal_height == before.nominal_height &&
          state.nominal_velocity == before.nominal_velocity &&
          state.offset_height == before.offset_height &&
          state.offset_velocity == before.offset_velocity &&
          state.airborne_frames == before.airborne_frames &&
          state.source == before.source &&
          state.initialized == before.initialized,
          "nonfinite support transaction");
    check(std::strstr(error, "finite") != NULL, "nonfinite diagnostic");
}

static void test_pose_and_root_helpers_are_horizontal_only()
{
    array1d<vec3> input(3), output(3);
    input(0) = vec3(1.0f, -7.0f, 2.0f);
    input(1) = vec3(0.0f, 0.8f, 0.0f);
    input(2) = vec3(0.1f, -0.2f, 0.3f);
    support_pose_apply(output, input, 0.42f);
    close(output(0).y, 0.42f, 0.0f, "support on Simulation Y");
    check(output(0).x == input(0).x && output(0).z == input(0).z,
          "support preserves root XZ");
    check(output(1).x == input(1).x && output(1).y == input(1).y &&
          output(1).z == input(1).z, "support preserves local child pose");

    const vec3 character(1.0f, -0.0f, 1.0f);
    const vec3 simulation(2.0f, -99.0f, 2.0f);
    vec3 result = horizontal_adjust_character_position(
        character, simulation, 0.1f, 1.0f / 25.0f);
    check(float_bits(result.y) == float_bits(character.y),
          "plain adjustment preserves Y bits");
    result = horizontal_adjust_character_position_by_velocity(
        character, vec3(0.5f, 1000.0f, 0.0f), simulation,
        0.5f, 0.1f, 1.0f / 25.0f);
    check(float_bits(result.y) == float_bits(character.y),
          "velocity adjustment preserves Y bits");
    result = horizontal_clamp_character_position(character, simulation, 0.15f);
    check(float_bits(result.y) == float_bits(character.y),
          "clamp preserves Y bits");
    close(std::sqrt((result.x - simulation.x) * (result.x - simulation.x) +
                    (result.z - simulation.z) * (result.z - simulation.z)),
          0.15f, 1e-6f, "horizontal clamp radius");
}

int main()
{
    test_contact_choice_airborne_hold_and_root_fallback();
    test_rebase_preserves_height_and_vertical_velocity();
    test_elevated_landing_ramp_and_descent_do_not_decay_to_zero();
    test_nonfinite_update_is_transactional();
    test_pose_and_root_helpers_are_horizontal_only();
    return 0;
}
```

- [ ] **Step 2: Compile to verify RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_support_runtime.cpp -o /tmp/test_support_runtime
```

Expected: compilation fails because `support_runtime.h` does not exist.

- [ ] **Step 3: Implement support observations, selection, rebase, and application**

Create `support_runtime.h`:

```cpp
#pragma once

#include "array.h"
#include "spring.h"
#include "terrain_runtime.h"
#include "vec.h"

#include <cmath>
#include <cstdio>

enum support_source
{
    support_root,
    support_left,
    support_right,
    support_both,
    support_held,
    support_airborne_root,
};

static inline const char* support_source_name(const support_source source)
{
    switch (source) {
    case support_root: return "root";
    case support_left: return "left";
    case support_right: return "right";
    case support_both: return "both";
    case support_held: return "held";
    default: return "airborne-root";
    }
}

struct support_observation
{
    float source_height[3] = {};
    float runtime_height[3] = {};
    float delta[3] = {};
    bool contact[2] = {};
};

struct support_frame_state
{
    float height = 0.0f;
    float velocity = 0.0f;
    float nominal_height = 0.0f;
    float nominal_velocity = 0.0f;
    float offset_height = 0.0f;
    float offset_velocity = 0.0f;
    int airborne_frames = 0;
    support_source source = support_root;
    bool initialized = false;
};

static inline bool support_error(
    char* error, const int capacity, const char* message)
{
    if (error != NULL && capacity > 0)
        std::snprintf(error, static_cast<size_t>(capacity), "%s", message);
    return false;
}

static inline bool support_observation_is_finite(
    const support_observation& observation)
{
    for (int i = 0; i < 3; ++i)
        if (!terrain_float_is_finite(observation.source_height[i]) ||
            !terrain_float_is_finite(observation.runtime_height[i]) ||
            !terrain_float_is_finite(observation.delta[i])) return false;
    return true;
}

static inline bool support_observation_build(
    support_observation& out,
    const terrain_support_set& support,
    const int frame,
    const heightfield& terrain,
    const vec3 root,
    const vec3 left_toe,
    const vec3 right_toe,
    const bool left_contact,
    const bool right_contact,
    char* error,
    const int error_capacity)
{
    if (frame < 0 || frame >= support.values.rows || support.values.cols != 3)
        return support_error(error, error_capacity, "support frame is out of range");
    support_observation candidate = {};
    const vec3 points[3] = {root, left_toe, right_toe};
    for (int i = 0; i < 3; ++i) {
        if (!terrain_float_is_finite(points[i].x) ||
            !terrain_float_is_finite(points[i].y) ||
            !terrain_float_is_finite(points[i].z))
            return support_error(error, error_capacity,
                "support FK points must contain only finite values");
        candidate.source_height[i] = support.values(frame, i);
        candidate.runtime_height[i] =
            heightfield_sample(terrain, points[i].x, points[i].z);
        candidate.delta[i] =
            candidate.runtime_height[i] - candidate.source_height[i];
    }
    candidate.contact[0] = left_contact;
    candidate.contact[1] = right_contact;
    if (!support_observation_is_finite(candidate))
        return support_error(error, error_capacity,
            "support observation must contain only finite values");
    out = candidate;
    return true;
}

static inline void support_frame_reset(
    support_frame_state& state, const float initial_height)
{
    state = support_frame_state();
    state.height = initial_height;
    state.nominal_height = initial_height;
    state.initialized = true;
}

static inline void support_frame_rebase(
    support_frame_state& state,
    const float new_nominal_height,
    const float new_nominal_velocity)
{
    state.offset_height = state.height - new_nominal_height;
    state.offset_velocity = state.velocity - new_nominal_velocity;
    state.nominal_height = new_nominal_height;
    state.nominal_velocity = new_nominal_velocity;
}

static inline bool support_frame_update(
    support_frame_state& state,
    const support_observation& observation,
    const bool source_frame_changed,
    const float dt,
    char* error,
    const int error_capacity)
{
    if (!support_observation_is_finite(observation) ||
        !terrain_float_is_finite(dt) || dt <= 0.0f)
        return support_error(error, error_capacity,
            "support update inputs must be finite with positive dt");
    support_frame_state next = state;
    if (!next.initialized) support_frame_reset(next, observation.delta[0]);

    float target = next.nominal_height;
    support_source source = support_held;
    if (observation.contact[0] && observation.contact[1]) {
        next.airborne_frames = 0;
        target = 0.5f * (observation.delta[1] + observation.delta[2]);
        source = support_both;
    } else if (observation.contact[0]) {
        next.airborne_frames = 0;
        target = observation.delta[1];
        source = support_left;
    } else if (observation.contact[1]) {
        next.airborne_frames = 0;
        target = observation.delta[2];
        source = support_right;
    } else {
        ++next.airborne_frames;
        if (next.airborne_frames <= 2) {
            target = next.nominal_height;
            source = support_held;
        } else {
            target = observation.delta[0];
            source = support_airborne_root;
        }
    }
    if (!terrain_float_is_finite(target))
        return support_error(error, error_capacity,
            "support target must be finite");
    const float target_velocity =
        source == support_held ? 0.0f : (target - next.nominal_height) / dt;
    const bool discontinuity = source_frame_changed || source != next.source ||
        std::fabs(target - next.nominal_height) > 0.02f;
    if (discontinuity)
        support_frame_rebase(next, target, target_velocity);
    else {
        next.nominal_height = target;
        next.nominal_velocity = target_velocity;
    }
    next.source = source;
    decay_spring_damper_exact(
        next.offset_height, next.offset_velocity, 0.10f, dt);
    next.height = next.nominal_height + next.offset_height;
    next.velocity = next.nominal_velocity + next.offset_velocity;
    if (!terrain_float_is_finite(next.height) ||
        !terrain_float_is_finite(next.velocity))
        return support_error(error, error_capacity,
            "support output must remain finite");
    state = next;
    return true;
}

static inline void support_pose_apply(
    const slice1d<vec3> output,
    const slice1d<vec3> inertialized,
    const float support_height)
{
    if (output.size != inertialized.size || output.size <= 0) return;
    for (int i = 0; i < output.size; ++i) output(i) = inertialized(i);
    output(0).y = support_height;
}

static inline float horizontal_length(const vec3 value)
{
    return std::sqrt(value.x * value.x + value.z * value.z);
}

static inline vec3 horizontal_adjust_character_position(
    const vec3 character,
    const vec3 simulation,
    const float halflife,
    const float dt)
{
    const vec3 difference(simulation.x - character.x, 0.0f,
                          simulation.z - character.z);
    const vec3 adjustment = damp_adjustment_exact(difference, halflife, dt);
    return vec3(character.x + adjustment.x, character.y,
                character.z + adjustment.z);
}

static inline vec3 horizontal_adjust_character_position_by_velocity(
    const vec3 character,
    const vec3 character_velocity,
    const vec3 simulation,
    const float max_adjustment_ratio,
    const float halflife,
    const float dt)
{
    vec3 adjustment = damp_adjustment_exact(
        vec3(simulation.x - character.x, 0.0f,
             simulation.z - character.z), halflife, dt);
    const float length_now = horizontal_length(adjustment);
    const float maximum = max_adjustment_ratio *
        horizontal_length(character_velocity) * dt;
    if (length_now > maximum && length_now > 1e-8f)
        adjustment = adjustment * (maximum / length_now);
    return vec3(character.x + adjustment.x, character.y,
                character.z + adjustment.z);
}

static inline vec3 horizontal_clamp_character_position(
    const vec3 character, const vec3 simulation, const float maximum)
{
    vec3 difference(character.x - simulation.x, 0.0f,
                    character.z - simulation.z);
    const float distance = horizontal_length(difference);
    if (distance <= maximum || distance <= 1e-8f) return character;
    difference = difference * (maximum / distance);
    return vec3(simulation.x + difference.x, character.y,
                simulation.z + difference.z);
}
```

- [ ] **Step 4: Run focused GREEN in debug, strict, fast-math, and sanitizer builds**

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
ASAN_OPTIONS=detect_leaks=1 /tmp/test_support_runtime_san
```

Expected: all four binaries exit `0`, strict compilation has no warnings, and sanitizer output is empty.

- [ ] **Step 5: Commit the renderer-independent support baseline**

```bash
git add support_runtime.h tests/cpp/test_support_runtime.cpp
git commit -m "feat: retarget G1 motion through a support frame"
```


### Task 4: Motion Manifest, Ordered Scene Catalog, and Scene-Pack Contract

**Files:**
- Create: `scene_runtime.h`
- Modify: `tests/cpp/test_scene_runtime.cpp`

**Interfaces:**
- Consumes the exact producer contracts from the prerequisite artifact plan.
  The manifest top-level key set is exactly:

  ```text
  schema,output_fps,feature_dimensions,terrain_dimensions,support_dimensions,
  terrain_feature_distances_m,total_clips,grail_clips,skipped_clips,
  database_frames,diagnostic_mode,sources,skeleton,contact,surface,database,
  sidecars,scene_index,validation_file,validation
  ```

- `surface` is exactly `{semantics,signature}`. `semantics` has keys
  `schema`, `coordinate_signature`, `source_query`, `polygon_triangulation`,
  `heightfield_schema`, `heightfield_interpolation`, `heightfield_diagonal`,
  `cell_size_m`, and `exterior_height_m`. The signature is the lowercase
  SHA-256 of canonical sorted compact JSON for `semantics` alone.
- `database={path:"database.bin",schema:"holden-database/v1",sha256}`.
  `sidecars.terrain_features={path:"terrain_features.bin",schema:"G1TF/v1",
  version:1,dimensions:4,sha256}` and
  `sidecars.terrain_support={path:"terrain_support.bin",schema:"G1SP/v1",
  version:1,dimensions:3,columns:["source_root_height_m",
  "source_left_toe_height_m","source_right_toe_height_m"],sha256}`.
  `scene_index={path:"scenes/index.json",schema:
  "g1-terrain-scene-index/v1",sha256}` and
  `validation_file={path:"validation.json",schema:
  "g1-terrain-validation/v1",sha256}`. Embedded `validation` must exactly equal
  the parsed validation file and has keys `schema`, `duration_error_s`,
  `fk_max_error_m`, and `quaternion_norm_max_error`.
- Existing v1 nested shapes remain exact: each source has
  `name,output_frames,range_start,range_stop,source_fps,source_frame_map,
  source_frames,terrain_id`; skeleton has `names,parents,signature`; contact
  has `height_threshold,median_filter_frames,speed_threshold`. Source records
  are contiguous, cover `[0,database_frames)`, and retain `name`, `terrain_id`,
  `range_start`, and `range_stop` for runtime diagnostics.
- The index is exactly `{schema,default_scene_id,scene_ids,
  coordinate_signature,surface_signature}` and uses the locked 14 IDs in order.
- Each scene is exactly `{schema,id,label,provenance,coordinate_signature,
  surface_signature,terrain_feature_distances_m,heightfield,mesh,walkability,
  bounds,spawn,regions,routes}`. The nested shapes and route outcome/class
  mapping are the exact ones in the prerequisite plan; no legacy `artifacts`,
  nested-bounds, `classification`, `outcome`, `speed_mps`, or
  `landing_height_m` aliases are accepted.
- Produces transactional `motion_manifest_load_and_verify`,
  `motion_manifest_validate_database`, `motion_source_for_frame`,
  `scene_catalog_load`, `scene_catalog_find`, `scene_route_find`,
  `scene_inside`, `scene_pack_load`, and `scene_pack_swap`. `scene_pack_load`
  receives the already-loaded manifest and never reopens motion-pack files.

- [ ] **Step 1: Add published-contract and transactional failure tests**

Append this semantic test boundary to `tests/cpp/test_scene_runtime.cpp` and
change `main` to accept exactly `--real ROOT` in addition to the Task 2 no-arg
primitive tests:

```cpp
#include "scene_runtime.h"

static const char* expected_scene_ids[] = {
    "grail-curb-default", "grail-curb-low", "grail-curb-medium",
    "grail-curb-high", "stairs-shallow", "stairs-standard",
    "stairs-unseen-variable", "ramp-05-up-down", "ramp-10-up-down",
    "ramp-15-stress", "cross-slope-05", "cross-slope-10",
    "mixed-multilevel", "blocked-course",
};
static const int expected_route_counts[] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,2,
};
static const char* expected_route_ids[][2] = {
    {"curb-forward",NULL},{"curb-forward",NULL},
    {"curb-forward",NULL},{"curb-forward",NULL},
    {"ascent-landing-descent",NULL},
    {"ascent-landing-descent",NULL},
    {"ascent-landing-descent",NULL},
    {"up-landing-down",NULL},{"up-landing-down",NULL},
    {"up-landing-down",NULL},
    {"forward-cross-slope",NULL},{"forward-cross-slope",NULL},
    {"full-course",NULL},{"wall-safe-stop","ramp-safe-stop"},
};

static void test_published_scene_contract(const char* root)
{
    char error[1024] = {};
    motion_pack_manifest manifest;
    check(motion_manifest_load_and_verify(
        manifest, root, error, sizeof(error)), error);
    check(manifest.output_fps == 25.0f &&
          manifest.feature_dimensions == 31 &&
          manifest.terrain_dimensions == 4 &&
          manifest.support_dimensions == 3, "motion dimensions");
    check(manifest.total_clips == 1770 && manifest.grail_clips == 1769 &&
          manifest.skipped_clips == 0 && !manifest.diagnostic_mode,
          "complete motion pack");
    check(manifest.sources.size() == 1770 &&
          manifest.sources.front().range_start == 0 &&
          manifest.sources.back().range_stop == manifest.database_frames,
          "source coverage");

    scene_catalog catalog;
    check(scene_catalog_load(
        catalog, root, manifest, error, sizeof(error)), error);
    check(catalog.default_scene_id == "grail-curb-default" &&
          catalog.ids.size() == 14, "catalog size/default");
    for (int i = 0; i < 14; ++i)
        check(catalog.ids[static_cast<size_t>(i)] == expected_scene_ids[i],
              "catalog order");

    scene_pack active;
    for (int i = 0; i < 14; ++i) {
        check(scene_pack_load(active, root, manifest, catalog, i,
              error, sizeof(error)), error);
        check(active.metadata.id == expected_scene_ids[i] &&
              active.terrain.version == 2 &&
              active.walkability.nx == active.terrain.nx &&
              active.walkability.nz == active.terrain.nz,
              "loaded scene identity/grid");
        check(active.metadata.routes.size() ==
              static_cast<size_t>(expected_route_counts[i]),
              "scene route count");
        for (int route = 0; route < expected_route_counts[i]; ++route)
            check(active.metadata.routes[static_cast<size_t>(route)].id ==
                  expected_route_ids[i][route], "scene route ID/order");
    }

    const std::string prior_id = active.metadata.id;
    const float prior_origin = active.terrain.origin_x;
    check(!scene_pack_load(active, root, manifest, catalog, -1,
          error, sizeof(error)), "bad scene index rejected");
    check(active.metadata.id == prior_id &&
          active.terrain.origin_x == prior_origin,
          "failed load preserves active pack");

    std::string bad_digest(64, '0');
    write_text("/tmp/test_scene_hash", "changed");
    check(!scene_verify_sha("/tmp/test_scene_hash", bad_digest,
          error, sizeof(error)), "hash mismatch rejected");
    check(std::strstr(error, "/tmp/test_scene_hash") != NULL &&
          std::strstr(error, "SHA-256") != NULL, "hash diagnostic");
    check(scene_catalog_find(catalog, "../escape") == -1,
          "path-like ID is not selected");
    check(motion_source_for_frame(manifest, 0) == 0 &&
          motion_source_for_frame(manifest, manifest.database_frames - 1) ==
              static_cast<int>(manifest.sources.size()) - 1 &&
          motion_source_for_frame(manifest, manifest.database_frames) == -1,
          "source lookup boundaries");
}
```

In `main`, require `argc == 1 || (argc == 3 &&
std::strcmp(argv[1],"--real") == 0)`; run Task 2 tests in both modes and call
`test_published_scene_contract(argv[2])` only in real mode. The test consumes
the prerequisite pack but never copies or mutates a published byte.

- [ ] **Step 2: Compile to verify semantic-loader RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime
```

Expected: compilation fails because `scene_runtime.h` and its semantic types do
not exist.

- [ ] **Step 3: Define ownership types, exact constants, and safe paths**

Create `scene_runtime.h` with these public values. Use `std::vector` only for
JSON-owned metadata; binary grids retain the Holden arrays:

```cpp
#pragma once
#include "g1_skeleton.h"
#include "json_runtime.h"
#include "sha256.h"
#include "terrain_runtime.h"
#include <cfloat>
#include <cstdarg>
#include <string>
#include <utility>
#include <vector>

struct bounds2 { float min_x=0,min_z=0,max_x=0,max_z=0; };
struct bounds3 { vec3 minimum,maximum; };
struct motion_source_record {
    std::string name,terrain_id;
    int range_start=0,range_stop=0;
};
struct artifact_reference {
    std::string path,schema,sha256;
    int version=0,dimensions=0;
    std::vector<std::string> columns;
};
struct surface_contract {
    std::string signature,coordinate_signature;
    std::string heightfield_interpolation,heightfield_diagonal;
    float cell_size=0,exterior_height=0;
};
struct motion_pack_manifest {
    float output_fps=0;
    int feature_dimensions=0,terrain_dimensions=0,support_dimensions=0;
    int total_clips=0,grail_clips=0,skipped_clips=0,database_frames=0;
    bool diagnostic_mode=false;
    std::vector<motion_source_record> sources;
    surface_contract surface;
    artifact_reference database,terrain_features,terrain_support;
    artifact_reference scene_index,validation_file;
};
struct scene_region { std::string id; bounds2 bounds; };
struct scene_route {
    std::string id,expected_outcome;
    int walkability_class=0;
    float landing_hold_seconds=0;
    std::vector<std::pair<float,float> > waypoints_xz;
};
struct scene_metadata {
    std::string id,label,provenance_kind;
    std::vector<std::string> provenance_source_ids;
    std::string coordinate_signature,surface_signature;
    artifact_reference heightfield,mesh,walkability;
    int heightfield_nx=0,heightfield_nz=0,walkability_nx=0,walkability_nz=0;
    float heightfield_origin_x=0,heightfield_origin_z=0;
    float heightfield_cell_size=0,heightfield_exterior_height=0;
    bounds3 mesh_bounds,heightfield_bounds;
    bounds2 playable_bounds,lookahead_bounds;
    vec3 spawn_position; float spawn_yaw=0;
    std::vector<scene_region> certified_regions,stress_regions,blocked_regions;
    std::vector<scene_route> routes;
};
struct scene_catalog {
    std::string default_scene_id,coordinate_signature,surface_signature;
    std::vector<std::string> ids;
};
struct scene_pack {
    scene_metadata metadata; heightfield terrain; walkability_grid walkability;
    std::string scene_path,terrain_path,mesh_path,walkability_path;
};

static const char* const G1_RuntimeSceneIds[14] = {
    "grail-curb-default","grail-curb-low","grail-curb-medium",
    "grail-curb-high","stairs-shallow","stairs-standard",
    "stairs-unseen-variable","ramp-05-up-down","ramp-10-up-down",
    "ramp-15-stress","cross-slope-05","cross-slope-10",
    "mixed-multilevel","blocked-course"
};
```

Add `scene_error`, `scene_required`, `scene_exact_keys`, `json_equal`,
`scene_number_float`, `scene_number_int`, `scene_float_array`,
`scene_sha_is_valid`, and `scene_verify_sha`. `scene_exact_keys` sorts the
actual and expected strings before comparison. Add these path rules verbatim:

```cpp
static inline bool scene_id_is_safe(const std::string& id)
{if(id.empty()||id.size()>64)return false;for(char c:id)if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='-'))return false;return true;}

static inline bool scene_relative_is_safe(const std::string& path)
{if(path.empty()||path.size()>4095||path[0]=='/'||path.find('\\')!=std::string::npos)return false;size_t begin=0;while(begin<=path.size()){const size_t end=path.find('/',begin);const std::string part=path.substr(begin,end==std::string::npos?std::string::npos:end-begin);if(part.empty()||part=="."||part=="..")return false;if(end==std::string::npos)break;begin=end+1;}return true;}

static inline bool scene_join(std::string& out,const char* root,const std::string& relative,char* error,int capacity)
{if(root==NULL||root[0]=='\0'||!scene_relative_is_safe(relative))return scene_error(error,capacity,"unsafe artifact path '%s'",relative.c_str());const std::string joined=std::string(root)+"/"+relative;if(joined.size()>4095)return scene_error(error,capacity,"artifact path exceeds 4095 bytes: %s",relative.c_str());out=joined;return true;}

static inline bool scene_inside(const bounds2& bounds,float x,float z)
{return terrain_float_is_finite(x)&&terrain_float_is_finite(z)&&x>=bounds.min_x&&x<=bounds.max_x&&z>=bounds.min_z&&z<=bounds.max_z;}
```

- [ ] **Step 4: Implement the exact manifest and index validators**

Implement `motion_manifest_load_and_verify` as a local-candidate transaction.
It enforces this complete matrix before assigning `out`:

| JSON location | Exact requirement |
|---|---|
| root | exact 20-key set listed above; schema `g1-terrain-artifacts/v2` |
| scalar dimensions | fps `25`, features `31`, terrain `4`, support `3`, distances `[0.25,0.5,0.75,1.0]` |
| complete pack | `total_clips=1770`, `grail_clips=1769`, `skipped_clips=0`, `diagnostic_mode=false`, positive integral `database_frames` |
| skeleton | exact `names`/`parents` lengths `31`, names equal the G1 bone names, parents equal the G1 parent table, signature `G1_SkeletonSignature` |
| contact | exact `speed_threshold,height_threshold,median_filter_frames`; finite nonnegative thresholds and positive odd integral filter width |
| sources | exact v1 source keys/types, unique nonempty names, first start `0`, adjacent stop/start equality, final stop `database_frames`, and `source_frame_map` length equals `output_frames` |
| surface.semantics | exact nine-key set and exact producer values, including `0.02` cell and `0.0` exterior |
| references | exact paths/schemas/versions/dimensions/columns shown above; lowercase 64-hex digests |
| validation | exact four-key embedded object; exact DOM equality with separately parsed, hashed `validation.json` |

Hash `database.bin`, both sidecars, `scenes/index.json`, and
`validation.json` over exact bytes. Recompute the surface signature by hashing
this exact canonical compact byte string after validating each semantic value:

```text
{"cell_size_m":0.02,"coordinate_signature":"holden-y-up-right-handed-forward-plus-z","exterior_height_m":0.0,"heightfield_diagonal":"min-x-min-z_to_max-x-max-z","heightfield_interpolation":"fixed-diagonal-triangles","heightfield_schema":"G1HF/v2","polygon_triangulation":"fan-from-first-index","schema":"g1-terrain-surface/v1","source_query":"vertical-triangle-top"}
```

Do not serialize parsed doubles to recompute it. Implement source lookup by
range containment:

```cpp
static inline int motion_source_for_frame(const motion_pack_manifest& manifest,const int frame)
{int low=0,high=static_cast<int>(manifest.sources.size());while(low<high){const int middle=low+(high-low)/2;const motion_source_record& source=manifest.sources[static_cast<size_t>(middle)];if(frame<source.range_start)high=middle;else if(frame>=source.range_stop)low=middle+1;else return middle;}return -1;}
```

`motion_manifest_validate_database` requires `db.nframes() ==
manifest.database_frames`, `db.nbones() == 31`, exact G1 parents, `31` feature
columns, `4` terrain columns, and database ranges equal every source range in
order. `scene_catalog_load` hashes/opens the existing index reference, requires
the exact five-key shape, exact coordinate/surface signatures, default
`grail-curb-default`, and exact ordered 14 IDs. Commit only after all checks.

- [ ] **Step 5: Parse exact scene metadata and validate the binary candidate**

Implement nested validators with these exact key sets and constraints:

| Object | Exact keys / checks |
|---|---|
| provenance | `kind,source_ids,parameters`; kind `grail` or `procedural`, string source IDs, parameters object |
| heightfield | `path,schema,version,nx,nz,origin_x,origin_z,cell_size_m,exterior_height_m,interpolation,diagonal,sha256`; exact basename/schema/version/surface values |
| mesh | `path,schema,sha256`; `terrain.obj`, `obj/v1` |
| walkability | `path,schema,version,nx,nz,classes,sha256`; exact G1WM values and classes `{blocked:0,certified:1,stress:2}` |
| bounds | eight flat arrays `mesh_min_xyz,mesh_max_xyz,heightfield_min_xyz,heightfield_max_xyz,playable_min_xz,playable_max_xz,lookahead_min_xz,lookahead_max_xz` |
| spawn | exact keys `position,yaw_radians`; three finite coordinates and finite yaw |
| regions | exact keys `certified,stress,blocked`; each value is an array of region objects |
| region | `id,bounds_xz`; unique nonempty ID and producer order `[min_x,max_x,min_z,max_z]`, converted to strict `bounds2` |
| route | `id,waypoints_xz,expected_outcome,walkability_class,landing_hold_seconds`; exact pairs `traverse/1`, `safe-stop/0`, `traverse-or-safe-stop/2` |

Require finite bounds with strict X/Z extent and `min_y <= max_y` (flat scenes
legitimately have equal Y bounds), playable/lookahead within heightfield bounds,
spawn inside playable, every region/waypoint inside lookahead, at least two
route points, nonnegative finite landing hold, unique region/route IDs, and at
least one route. A positive hold additionally requires at least four points
and uses waypoint index `2` as the landing hold with a later exit waypoint.
Require the exact route count/ID order matrix from Step 1 for each exact scene
ID; aliases, missing/extra routes, and the two blocked routes in reversed order
are invalid even if their generic outcome/class pair is otherwise valid. Add
candidate fixtures that rename one route, reverse the blocked pair, and append
an extra route; each must fail transactionally with the active scene unchanged.
Hash all three scene artifacts, load G1HF/G1WM, and require
every declared grid/origin/cell/exterior field to equal the decoded binary.
Sample every route segment at spacing no larger than half a cell using the
same strict-domain nearest-node convention as `walkability_class_at`:
`traverse` stays class `1`; `traverse-or-safe-stop` stays class `2` in its
published stress course; and `safe-stop` begins class `1`, crosses exactly once
to class `0`, and never re-enters class `1`. No route may leave the grid.

Use this exact transaction boundary and the already-loaded manifest:

```cpp
static inline bool scene_pack_load(scene_pack& out,const char* root,const motion_pack_manifest& manifest,const scene_catalog& catalog,int index,char* error,int capacity)
{
 if(index<0||index>=static_cast<int>(catalog.ids.size()))return scene_error(error,capacity,"scene index %d is out of range",index);
 const std::string& id=catalog.ids[static_cast<size_t>(index)];const std::string prefix="scenes/"+id+"/";scene_pack candidate;
 if(!scene_join(candidate.scene_path,root,prefix+"scene.json",error,capacity))return false;json_value document;
 if(!json_document_load(document,candidate.scene_path.c_str(),error,capacity)||!scene_metadata_parse(candidate.metadata,document,id.c_str(),manifest,candidate.scene_path.c_str(),error,capacity))return false;
 if(!scene_join(candidate.terrain_path,root,prefix+candidate.metadata.heightfield.path,error,capacity)||!scene_join(candidate.mesh_path,root,prefix+candidate.metadata.mesh.path,error,capacity)||!scene_join(candidate.walkability_path,root,prefix+candidate.metadata.walkability.path,error,capacity)||!scene_verify_sha(candidate.terrain_path,candidate.metadata.heightfield.sha256,error,capacity)||!scene_verify_sha(candidate.mesh_path,candidate.metadata.mesh.sha256,error,capacity)||!scene_verify_sha(candidate.walkability_path,candidate.metadata.walkability.sha256,error,capacity)||!heightfield_load(candidate.terrain,candidate.terrain_path.c_str(),error,capacity))return false;
 if(candidate.terrain.version!=2)return scene_error(error,capacity,"%s: published scene '%s' requires G1HF version 2, got %u",candidate.terrain_path.c_str(),id.c_str(),static_cast<unsigned>(candidate.terrain.version));
 if(!walkability_load(candidate.walkability,candidate.walkability_path.c_str(),candidate.terrain,error,capacity)||!scene_candidate_validate(candidate,error,capacity))return false;
 scene_pack_swap(out,candidate);return true;
}
```

`scene_pack_swap` swaps every G1HF scalar and `heights.{size,data}`, every
G1WM scalar and `cells.{size,data}`, then metadata/path strings. Add exact
linear lookup helpers:

```cpp
static inline int scene_catalog_find(const scene_catalog& catalog,const char* id)
{if(id==NULL)return -1;for(size_t i=0;i<catalog.ids.size();++i)if(catalog.ids[i]==id)return static_cast<int>(i);return -1;}
static inline const scene_route* scene_route_find(const scene_metadata& scene,const char* id)
{if(id==NULL)return NULL;for(size_t i=0;i<scene.routes.size();++i)if(scene.routes[i].id==id)return &scene.routes[i];return NULL;}
```

- [ ] **Step 6: Run semantic GREEN in strict, fast-math, and sanitizer builds**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_strict
/tmp/test_scene_runtime_strict --real resources/g1_terrain
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_scene_runtime.cpp -o /tmp/test_scene_runtime_release
/tmp/test_scene_runtime_release --real resources/g1_terrain
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_scene_runtime.cpp \
  -o /tmp/test_scene_runtime_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_scene_runtime_san --real resources/g1_terrain
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_artifacts tests.python.test_scenes tests.python.test_build_cli -v
```

Expected: every C++ configuration validates all 14 scenes with no warnings or
sanitizer report; all prerequisite artifact tests pass.

- [ ] **Step 7: Commit the strict scene-pack boundary**

```bash
git add scene_runtime.h tests/cpp/test_scene_runtime.cpp
git commit -m "feat: validate G1 multiscene runtime packs"
```

### Task 5: One Complete, Testable Controller-State Reset

**Files:**
- Create: `g1_controller_state.h`
- Create: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `controller.cpp` to replace the corresponding mutable locals with one `g1_controller_state state`

**Interfaces:**
- Consumes: immutable `database`, one-time `terrain_support_set`, and a fully validated `scene_pack`.
- Produces: `bool g1_controller_state_reset(g1_controller_state&, const database&, const terrain_support_set&, const scene_pack&, char*, int)`.
- Produces: `void g1_controller_state_swap(g1_controller_state&, g1_controller_state&)` without deep copies or ownership aliasing.
- Reset covers selected/current/transition pose arrays, all inertial offsets, trajectory/simulation/input state, recorded contacts and dormant contact-fixup buffers, support state, traversal state, search timers, deterministic route cursor, camera orbit, per-scene diagnostics, and adjusted/global FK buffers.
- Configuration that intentionally persists across a scene reset stays outside the struct: feature weights, inertialization/adjustment/clamp tuning, window/model, catalog, log file, total log row number, and exit status.
- Reset maps the first database range to scene spawn XZ/yaw, initializes support from `runtime_height(spawn_xz) - source_root_height(frame)`, leaves the planar simulation Y at zero, and applies support only to the render-pose `G1_Simulation.y`.

- [ ] **Step 1: Write a poison-and-reset test**

Create `tests/cpp/test_g1_controller_state.cpp`:

```cpp
#include "g1_controller_state.h"
#include <cstdio>
#include <cstdlib>

static void check(bool value, const char* message)
{
    if (!value) { std::fprintf(stderr, "controller reset test failed: %s\n", message); std::exit(1); }
}

static void make_database(database& db)
{
    db.bone_positions.resize(2, G1_BoneCount);
    db.bone_velocities.resize(2, G1_BoneCount);
    db.bone_rotations.resize(2, G1_BoneCount);
    db.bone_angular_velocities.resize(2, G1_BoneCount);
    db.contact_states.resize(2, 2);
    db.range_starts.resize(1); db.range_stops.resize(1);
    db.range_starts(0)=0; db.range_stops(0)=2;
    db.bone_positions.zero(); db.bone_velocities.zero();
    db.bone_rotations.set(quat()); db.bone_angular_velocities.zero();
    db.contact_states.zero();
    db.bone_positions(0,G1_Hips)=vec3(0,0.82f,0);
}

static scene_pack make_scene()
{
    scene_pack scene;
    scene.metadata.id="fixture";
    scene.metadata.spawn_position=vec3(1.25f,0.30f,-2.0f);
    scene.metadata.spawn_yaw=0.5f;
    scene.metadata.playable_bounds={1.0f,-2.25f,1.5f,-1.75f};
    scene.terrain.version=2; scene.terrain.nx=2; scene.terrain.nz=2;
    scene.terrain.origin_x=1.0f; scene.terrain.origin_z=-2.25f;
    scene.terrain.cell_size=0.5f; scene.terrain.exterior_height=-10.0f;
    scene.terrain.heights.resize(4); scene.terrain.heights.set(0.30f);
    scene.walkability.nx=2; scene.walkability.nz=2;
    scene.walkability.cells.resize(4); scene.walkability.cells.set(1);
    return scene;
}

static void test_reset_clears_every_dynamic_subsystem()
{
    database db; make_database(db);
    terrain_support_set support; support.values.resize(2,3); support.values.zero();
    scene_pack scene=make_scene();
    g1_controller_state state;
    state.frame_index=999; state.scene_frame=999; state.search_timer=-9;
    state.simulation_position=vec3(9,9,9); state.simulation_velocity=vec3(9,9,9);
    state.desired_velocity=vec3(9,9,9); state.route_waypoint=99;
    state.support.height=99; state.support.velocity=99;
    state.traversal_speed_scale=0; state.blocked=true;
    state.contact_locks.resize(2); state.contact_locks.set(true);
    state.bone_offset_positions.resize(G1_BoneCount);
    state.bone_offset_positions.set(vec3(9,9,9));
    char error[512]={};
    check(g1_controller_state_reset(state,db,support,scene,error,sizeof(error)),error);
    check(state.frame_index==0 && state.scene_frame==0,"frame reset");
    check(state.search_timer==state.search_time && state.force_search_timer==state.search_time,"search reset");
    check(state.simulation_position.x==1.25f && state.simulation_position.y==0.0f && state.simulation_position.z==-2.0f,"planar spawn");
    check(state.simulation_velocity.x==0 && state.simulation_velocity.y==0 && state.simulation_velocity.z==0,"simulation derivatives reset");
    check(state.desired_velocity.x==0 && state.route_waypoint==1,"input/route reset");
    check(state.support.height==0.30f && state.support.velocity==0,"support reset");
    check(state.adjusted_bone_positions(G1_Simulation).y==0.30f,"render support applied");
    check(state.bone_positions(G1_Simulation).y==0.0f,"inertial pose remains support-local");
    check(state.traversal_speed_scale==1.0f && !state.blocked,"traversal reset");
    for(int i=0;i<state.contact_locks.size;++i)check(!state.contact_locks(i),"contact lock reset");
    for(int i=0;i<state.bone_offset_positions.size;++i)
        check(state.bone_offset_positions(i).x==0&&state.bone_offset_positions(i).y==0&&state.bone_offset_positions(i).z==0,"pose offset reset");
}

static void test_candidate_reset_failure_preserves_prior_state()
{
    database db; make_database(db);
    terrain_support_set support; support.values.resize(2,3); support.values.zero();
    scene_pack valid=make_scene(), bad=make_scene(); bad.metadata.spawn_position.x=99;
    g1_controller_state active;
    char error[512]={};
    check(g1_controller_state_reset(active,db,support,valid,error,sizeof(error)),error);
    const float prior_x=active.simulation_position.x;
    g1_controller_state candidate;
    check(!g1_controller_state_reset(candidate,db,support,bad,error,sizeof(error)),"bad spawn rejected");
    check(active.simulation_position.x==prior_x&&active.support.height==0.30f,"active state preserved");
}

int main(){test_reset_clears_every_dynamic_subsystem();test_candidate_reset_failure_preserves_prior_state();return 0;}
```

- [ ] **Step 2: Compile to verify reset RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_controller_state
```

Expected: compilation fails because `g1_controller_state.h` does not exist.

- [ ] **Step 3: Define all resettable state and ownership-safe swaps**

Create `g1_controller_state.h`:

```cpp
#pragma once
#include "database.h"
#include "g1_skeleton.h"
#include "scene_runtime.h"
#include "support_runtime.h"
#include <cfloat>
#include <utility>

struct g1_controller_state
{
    int frame_index=0, scene_frame=0;
    float search_time=0.10f, search_timer=0.10f, force_search_timer=0.10f;
    array1d<vec3> curr_bone_positions,curr_bone_velocities,trns_bone_positions,trns_bone_velocities;
    array1d<quat> curr_bone_rotations,trns_bone_rotations;
    array1d<vec3> curr_bone_angular_velocities,trns_bone_angular_velocities;
    array1d<bool> curr_bone_contacts,trns_bone_contacts;
    array1d<vec3> bone_positions,bone_velocities,bone_angular_velocities;
    array1d<quat> bone_rotations;
    array1d<vec3> bone_offset_positions,bone_offset_velocities,bone_offset_angular_velocities;
    array1d<quat> bone_offset_rotations;
    array1d<vec3> adjusted_bone_positions,global_bone_positions,global_bone_velocities;
    array1d<quat> adjusted_bone_rotations,global_bone_rotations;
    array1d<vec3> global_bone_angular_velocities;
    array1d<bool> global_bone_computed;
    vec3 transition_src_position,transition_dst_position;
    quat transition_src_rotation,transition_dst_rotation;
    vec3 desired_velocity,desired_velocity_change_curr,desired_velocity_change_prev;
    quat desired_rotation;
    vec3 desired_rotation_change_curr,desired_rotation_change_prev;
    float desired_gait=0,desired_gait_velocity=0;
    vec3 simulation_position,simulation_velocity,simulation_acceleration;
    quat simulation_rotation;
    vec3 simulation_angular_velocity;
    array1d<vec3> trajectory_desired_velocities,trajectory_positions,trajectory_velocities,trajectory_accelerations,trajectory_angular_velocities;
    array1d<quat> trajectory_desired_rotations,trajectory_rotations;
    array1d<int> contact_bones;
    array1d<bool> contact_states,contact_locks;
    array1d<vec3> contact_positions,contact_velocities,contact_points,contact_targets,contact_offset_positions,contact_offset_velocities;
    support_frame_state support;
    support_observation support_observation_now;
    float traversal_speed_scale=1,traversal_speed_scale_velocity=0;
    bool blocked=false; int walkability_class=1; float blocked_distance=FLT_MAX;
    vec3 blocked_point;
    int route_index=0,route_waypoint=1,route_frames=0;
    float camera_azimuth=0,camera_altitude=0.4f,camera_distance=4;
    bool searched=false,transitioned=false;
    float incumbent_cost=0,selected_cost=0,selected_terrain_error=0;
    float adjustment_xz=0,adjustment_y=0,clamp_xz=0,clamp_y=0;
};

template<typename T> static inline void g1_swap(array1d<T>& a,array1d<T>& b){std::swap(a.size,b.size);std::swap(a.data,b.data);}

static inline void g1_controller_state_swap(g1_controller_state& a,g1_controller_state& b)
{
    using std::swap;
    swap(a.frame_index,b.frame_index);swap(a.scene_frame,b.scene_frame);swap(a.search_time,b.search_time);swap(a.search_timer,b.search_timer);swap(a.force_search_timer,b.force_search_timer);
#define G1_SWAP_ARRAY(name) g1_swap(a.name,b.name)
    G1_SWAP_ARRAY(curr_bone_positions);G1_SWAP_ARRAY(curr_bone_velocities);G1_SWAP_ARRAY(trns_bone_positions);G1_SWAP_ARRAY(trns_bone_velocities);
    G1_SWAP_ARRAY(curr_bone_rotations);G1_SWAP_ARRAY(trns_bone_rotations);G1_SWAP_ARRAY(curr_bone_angular_velocities);G1_SWAP_ARRAY(trns_bone_angular_velocities);
    G1_SWAP_ARRAY(curr_bone_contacts);G1_SWAP_ARRAY(trns_bone_contacts);G1_SWAP_ARRAY(bone_positions);G1_SWAP_ARRAY(bone_velocities);G1_SWAP_ARRAY(bone_angular_velocities);G1_SWAP_ARRAY(bone_rotations);
    G1_SWAP_ARRAY(bone_offset_positions);G1_SWAP_ARRAY(bone_offset_velocities);G1_SWAP_ARRAY(bone_offset_angular_velocities);G1_SWAP_ARRAY(bone_offset_rotations);
    G1_SWAP_ARRAY(adjusted_bone_positions);G1_SWAP_ARRAY(global_bone_positions);G1_SWAP_ARRAY(global_bone_velocities);G1_SWAP_ARRAY(adjusted_bone_rotations);G1_SWAP_ARRAY(global_bone_rotations);G1_SWAP_ARRAY(global_bone_angular_velocities);G1_SWAP_ARRAY(global_bone_computed);
    G1_SWAP_ARRAY(trajectory_desired_velocities);G1_SWAP_ARRAY(trajectory_positions);G1_SWAP_ARRAY(trajectory_velocities);G1_SWAP_ARRAY(trajectory_accelerations);G1_SWAP_ARRAY(trajectory_angular_velocities);G1_SWAP_ARRAY(trajectory_desired_rotations);G1_SWAP_ARRAY(trajectory_rotations);
    G1_SWAP_ARRAY(contact_bones);G1_SWAP_ARRAY(contact_states);G1_SWAP_ARRAY(contact_locks);G1_SWAP_ARRAY(contact_positions);G1_SWAP_ARRAY(contact_velocities);G1_SWAP_ARRAY(contact_points);G1_SWAP_ARRAY(contact_targets);G1_SWAP_ARRAY(contact_offset_positions);G1_SWAP_ARRAY(contact_offset_velocities);
#undef G1_SWAP_ARRAY
    swap(a.transition_src_position,b.transition_src_position);swap(a.transition_dst_position,b.transition_dst_position);swap(a.transition_src_rotation,b.transition_src_rotation);swap(a.transition_dst_rotation,b.transition_dst_rotation);
    swap(a.desired_velocity,b.desired_velocity);swap(a.desired_velocity_change_curr,b.desired_velocity_change_curr);swap(a.desired_velocity_change_prev,b.desired_velocity_change_prev);swap(a.desired_rotation,b.desired_rotation);swap(a.desired_rotation_change_curr,b.desired_rotation_change_curr);swap(a.desired_rotation_change_prev,b.desired_rotation_change_prev);swap(a.desired_gait,b.desired_gait);swap(a.desired_gait_velocity,b.desired_gait_velocity);
    swap(a.simulation_position,b.simulation_position);swap(a.simulation_velocity,b.simulation_velocity);swap(a.simulation_acceleration,b.simulation_acceleration);swap(a.simulation_rotation,b.simulation_rotation);swap(a.simulation_angular_velocity,b.simulation_angular_velocity);
    swap(a.support,b.support);swap(a.support_observation_now,b.support_observation_now);swap(a.traversal_speed_scale,b.traversal_speed_scale);swap(a.traversal_speed_scale_velocity,b.traversal_speed_scale_velocity);swap(a.blocked,b.blocked);swap(a.walkability_class,b.walkability_class);swap(a.blocked_distance,b.blocked_distance);swap(a.blocked_point,b.blocked_point);
    swap(a.route_index,b.route_index);swap(a.route_waypoint,b.route_waypoint);swap(a.route_frames,b.route_frames);swap(a.camera_azimuth,b.camera_azimuth);swap(a.camera_altitude,b.camera_altitude);swap(a.camera_distance,b.camera_distance);swap(a.searched,b.searched);swap(a.transitioned,b.transitioned);swap(a.incumbent_cost,b.incumbent_cost);swap(a.selected_cost,b.selected_cost);swap(a.selected_terrain_error,b.selected_terrain_error);swap(a.adjustment_xz,b.adjustment_xz);swap(a.adjustment_y,b.adjustment_y);swap(a.clamp_xz,b.clamp_xz);swap(a.clamp_y,b.clamp_y);
}
```

- [ ] **Step 4: Implement a transactional reset to scene spawn**

Append to `g1_controller_state.h`:

```cpp
static inline bool g1_controller_state_reset(g1_controller_state& out,const database& db,const terrain_support_set& support,const scene_pack& scene,char* error,int capacity)
{
    if(db.nframes()<=0||db.nbones()!=G1_BoneCount||db.nranges()<=0||support.values.rows!=db.nframes()||support.values.cols!=3)
        return scene_error(error,capacity,"controller reset: invalid database/support shapes");
    const vec3 spawn=scene.metadata.spawn_position;
    if(!terrain_float_is_finite(spawn.x)||!terrain_float_is_finite(spawn.y)||!terrain_float_is_finite(spawn.z)||!terrain_float_is_finite(scene.metadata.spawn_yaw)||
       !scene_inside(scene.metadata.playable_bounds,spawn.x,spawn.z))
        return scene_error(error,capacity,"controller reset: scene '%s' spawn is invalid",scene.metadata.id.c_str());
    g1_controller_state s; s.frame_index=db.range_starts(0);
    s.curr_bone_positions=db.bone_positions(s.frame_index);s.curr_bone_velocities=db.bone_velocities(s.frame_index);s.curr_bone_rotations=db.bone_rotations(s.frame_index);s.curr_bone_angular_velocities=db.bone_angular_velocities(s.frame_index);s.curr_bone_contacts=db.contact_states(s.frame_index);
    s.trns_bone_positions=s.curr_bone_positions;s.trns_bone_velocities=s.curr_bone_velocities;s.trns_bone_rotations=s.curr_bone_rotations;s.trns_bone_angular_velocities=s.curr_bone_angular_velocities;s.trns_bone_contacts=s.curr_bone_contacts;
    s.bone_positions=s.curr_bone_positions;s.bone_velocities=s.curr_bone_velocities;s.bone_rotations=s.curr_bone_rotations;s.bone_angular_velocities=s.curr_bone_angular_velocities;
    const int bones=db.nbones();s.bone_offset_positions.resize(bones);s.bone_offset_positions.zero();s.bone_offset_velocities.resize(bones);s.bone_offset_velocities.zero();s.bone_offset_rotations.resize(bones);s.bone_offset_rotations.set(quat());s.bone_offset_angular_velocities.resize(bones);s.bone_offset_angular_velocities.zero();
    s.transition_src_position=s.curr_bone_positions(0);s.transition_src_rotation=s.curr_bone_rotations(0);s.transition_dst_position=vec3(spawn.x,0,spawn.z);s.transition_dst_rotation=quat_from_angle_axis(scene.metadata.spawn_yaw,vec3(0,1,0));
    s.bone_positions(0)=s.transition_dst_position;s.bone_rotations(0)=s.transition_dst_rotation;s.bone_velocities(0)=vec3();s.bone_angular_velocities(0)=vec3();
    s.adjusted_bone_positions=s.bone_positions;s.adjusted_bone_rotations=s.bone_rotations;s.global_bone_positions.resize(bones);s.global_bone_positions.zero();s.global_bone_velocities.resize(bones);s.global_bone_velocities.zero();s.global_bone_rotations.resize(bones);s.global_bone_rotations.set(quat());s.global_bone_angular_velocities.resize(bones);s.global_bone_angular_velocities.zero();s.global_bone_computed.resize(bones);s.global_bone_computed.zero();
    s.simulation_position=vec3(spawn.x,0,spawn.z);s.simulation_rotation=s.transition_dst_rotation;s.desired_rotation=s.simulation_rotation;
    s.trajectory_desired_velocities.resize(4);s.trajectory_desired_velocities.zero();s.trajectory_positions.resize(4);s.trajectory_positions.set(s.simulation_position);s.trajectory_velocities.resize(4);s.trajectory_velocities.zero();s.trajectory_accelerations.resize(4);s.trajectory_accelerations.zero();s.trajectory_angular_velocities.resize(4);s.trajectory_angular_velocities.zero();s.trajectory_desired_rotations.resize(4);s.trajectory_desired_rotations.set(s.simulation_rotation);s.trajectory_rotations.resize(4);s.trajectory_rotations.set(s.simulation_rotation);
    s.contact_bones.resize(2);s.contact_bones(0)=G1_LeftToe;s.contact_bones(1)=G1_RightToe;s.contact_states.resize(2);s.contact_states.zero();s.contact_locks.resize(2);s.contact_locks.zero();s.contact_positions.resize(2);s.contact_positions.zero();s.contact_velocities.resize(2);s.contact_velocities.zero();s.contact_points.resize(2);s.contact_points.zero();s.contact_targets.resize(2);s.contact_targets.zero();s.contact_offset_positions.resize(2);s.contact_offset_positions.zero();s.contact_offset_velocities.resize(2);s.contact_offset_velocities.zero();
    const float runtime=heightfield_sample(scene.terrain,spawn.x,spawn.z);const float initial=runtime-support.values(s.frame_index,0);
    if(!terrain_float_is_finite(initial))return scene_error(error,capacity,"controller reset: non-finite initial support for scene '%s'",scene.metadata.id.c_str());
    support_frame_reset(s.support,initial);support_pose_apply(s.adjusted_bone_positions,s.bone_positions,s.support.height);s.search_timer=s.search_time;s.force_search_timer=s.search_time;s.camera_azimuth=scene.metadata.spawn_yaw;s.blocked_distance=FLT_MAX;
    g1_controller_state_swap(out,s);return true;
}
```

- [ ] **Step 5: Load the default scene, then replace controller locals mechanically**

Include `scene_runtime.h` and `g1_controller_state.h`. Replace the legacy root
`terrain.bin`/`terrain.obj` startup with the Task 4 contract before moving any
locals. Remove the root heightfield/mesh paths and run this sequence before
`InitWindow`:

```cpp
motion_pack_manifest motion_manifest;
if(!motion_manifest_load_and_verify(motion_manifest,terrain_directory,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 motion manifest error: %s\n",artifact_error);return 2;}
std::string database_path,feature_path,support_path;
if(!scene_join(database_path,terrain_directory,motion_manifest.database.path,artifact_error,sizeof(artifact_error))||
   !scene_join(feature_path,terrain_directory,motion_manifest.terrain_features.path,artifact_error,sizeof(artifact_error))||
   !scene_join(support_path,terrain_directory,motion_manifest.terrain_support.path,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 motion path error: %s\n",artifact_error);return 2;}
terrain_feature_set terrain_rows;
if(!terrain_features_load(terrain_rows,feature_path.c_str(),artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 terrain feature error: %s\n",artifact_error);return 2;}
database db;database_load(db,database_path.c_str());
if(!g1_database_validate(db,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 database error: %s\n",artifact_error);return 2;}
if(terrain_rows.values.rows!=db.nframes()||terrain_rows.values.cols!=4){std::fprintf(stderr,"G1 database/G1TF shape error: database=%d terrain=%dx%d expected_columns=4\n",db.nframes(),terrain_rows.values.rows,terrain_rows.values.cols);return 2;}
db.terrain_features=terrain_rows.values;
if(!g1_skeleton_validate(db,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 skeleton error: %s\n",artifact_error);return 2;}
database_build_matching_features(db,feature_weight_foot_position,feature_weight_foot_velocity,feature_weight_hip_velocity,feature_weight_trajectory_positions,feature_weight_trajectory_directions,G1_LeftAnkle,G1_RightAnkle,G1_Hips,feature_weight_terrain);
if(!g1_matching_features_validate(db,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 feature error: %s\n",artifact_error);return 2;}
if(!motion_manifest_validate_database(motion_manifest,db,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 motion/database contract error: %s\n",artifact_error);return 2;}
scene_catalog catalog;
if(!scene_catalog_load(catalog,terrain_directory,motion_manifest,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 scene index error: %s\n",artifact_error);return 2;}
const int active_scene_index=scene_catalog_find(catalog,catalog.default_scene_id.c_str());
if(active_scene_index<0){std::fprintf(stderr,"G1 default scene is absent: %s\n",catalog.default_scene_id.c_str());return 2;}
scene_pack active_scene;
if(!scene_pack_load(active_scene,terrain_directory,motion_manifest,catalog,active_scene_index,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 scene error [%s]: %s\n",catalog.default_scene_id.c_str(),artifact_error);return 2;}
terrain_support_set support_rows;
if(!terrain_support_load(support_rows,support_path.c_str(),db.nframes(),artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 support error: %s\n",artifact_error);return 2;}
g1_controller_state state;
if(!g1_controller_state_reset(state,db,support_rows,active_scene,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 reset error: %s\n",artifact_error);return 2;}
```

`motion_manifest_load_and_verify` runs before `database_load`, so a missing,
unreadable, or hash-mismatched motion artifact produces a controlled error
instead of reaching the inherited assertion-based reader. After `InitWindow`, load
`active_scene.mesh_path.c_str()` instead of a root OBJ, and use
`active_scene.terrain` for the existing Gate A terrain query and diagnostics.
No selection UI or switching is added in this task.

Then replace each moved local with the matching `state.` member. For example,
the existing search/query/inertialization sequence becomes:

```cpp
slice1d<float> query_features=db.features(state.frame_index);
query_compute_trajectory_position_feature(query,offset,state.bone_positions(0),state.bone_rotations(0),state.trajectory_positions);
query_compute_trajectory_direction_feature(query,offset,state.bone_rotations(0),state.trajectory_rotations);
// existing search and inertialize_pose_* calls use state fields one-for-one
```

Keep `feature_weight_*`, tuning values, `terrain_model`, log, catalog,
`rendered_frames`, and exit flags as existing locals. At this step, call reset
only at startup and keep support-retargeting disabled; the rendered result must
still reproduce the prerequisite Gate A log.

- [ ] **Step 6: Run reset tests and the unchanged Gate A reproduction**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_controller_state
/tmp/test_g1_controller_state
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_multiscene -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
mkdir -p /tmp/g1-multiscene-runtime
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain MM_TERRAIN_SCENE=grail-curb-default \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=375 \
  MM_LOG=/tmp/g1-multiscene-runtime/gate-a-reset-refactor.csv \
  /tmp/controller_g1_multiscene
/home/ubuntu/miniconda3/envs/diffsim/bin/python resources/check_g1_runtime_log.py \
  /tmp/g1-multiscene-runtime/gate-a-reset-refactor.csv --gate-a
```

Expected: reset test exits `0`; controller exits normally after 375 frames; checker prints `VALID gate-a`; support remains disabled and all pre-support diagnosis columns match a fresh prerequisite Gate A run except the output filename.

- [ ] **Step 7: Commit the reset boundary**

```bash
git add g1_controller_state.h tests/cpp/test_g1_controller_state.cpp controller.cpp
git commit -m "refactor: reset G1 controller state as one unit"
```

### Task 6: Startup Selection and Transactional Raylib Scene Switching

**Files:**
- Create: `scene_switch.h`
- Create: `tests/cpp/test_scene_switch.cpp`
- Modify: `controller.cpp` startup, between-update request handling, UI, and cleanup

**Interfaces:**
- Produces `scene_model_load_result { bool allocated; bool ready; }` and the
  `scene_switch_transaction` template defined in Step 3, testable with a fake
  model.
- Produces `scene_reset_current` with the exact Step 3 signature; it resets
  dynamic state without reloading scene bytes, motion bytes, or model.
- Controller loads/validates motion manifest, database, G1TF, G1SP, catalog, startup scene pack, and startup controller state before `InitWindow`; only Raylib model readiness necessarily occurs after the window opens.
- `MM_TERRAIN_SCENE` is read once. Missing means catalog `default_scene_id` (`grail-curb-default`); present empty/unknown/path-like IDs exit `2` before Raylib.
- Previous/Next wrap in catalog order. Reset preserves the active model and scene pack. Switch requests are recorded by UI during drawing and executed only at the next fixed-update boundary.
- A successful switch commits candidate scene, candidate reset state, candidate model, and active index together, then unloads the old model exactly once. Any failure unloads only an allocated candidate model, preserves all active objects, and reports candidate ID/path/reason.

- [ ] **Step 1: Write fake-model transactional tests**

Create `tests/cpp/test_scene_switch.cpp`:

```cpp
#include "scene_switch.h"
#include <cstdio>
#include <cstdlib>

static void check(bool v,const char* m){if(!v){std::fprintf(stderr,"scene switch test failed: %s\n",m);std::exit(1);}}
struct fake_model{int id=0;};
struct fake_models{int loads=0,unloads=0,live=0;bool fail=false;};

static database fixture_db(){database db;db.bone_positions.resize(1,G1_BoneCount);db.bone_velocities.resize(1,G1_BoneCount);db.bone_rotations.resize(1,G1_BoneCount);db.bone_angular_velocities.resize(1,G1_BoneCount);db.contact_states.resize(1,2);db.range_starts.resize(1);db.range_stops.resize(1);db.bone_positions.zero();db.bone_velocities.zero();db.bone_rotations.set(quat());db.bone_angular_velocities.zero();db.contact_states.zero();db.range_starts(0)=0;db.range_stops(0)=1;return db;}
static scene_pack fixture_scene(const char* id,float x){scene_pack s;s.metadata.id=id;s.metadata.spawn_position=vec3(x,0,0);s.metadata.spawn_yaw=0;s.metadata.playable_bounds={x-1,-1,x+1,1};s.terrain.version=2;s.terrain.nx=2;s.terrain.nz=2;s.terrain.origin_x=x-1;s.terrain.origin_z=-1;s.terrain.cell_size=2;s.terrain.exterior_height=0;s.terrain.heights.resize(4);s.terrain.heights.zero();s.walkability.nx=2;s.walkability.nz=2;s.walkability.cells.resize(4);s.walkability.cells.set(1);s.mesh_path=std::string(id)+".obj";return s;}

static void test_success_failure_reset_and_repeated_cycles()
{
 database db=fixture_db();terrain_support_set support;support.values.resize(1,3);support.values.zero();
 scene_catalog catalog;catalog.ids={"one","two"};motion_pack_manifest manifest;
 scene_pack active_scene=fixture_scene("one",0);g1_controller_state state;char error[512]={};check(g1_controller_state_reset(state,db,support,active_scene,error,sizeof(error)),error);
 int active_index=0;fake_model active_model{1};fake_models models;models.live=1;
 auto load_scene=[&](scene_pack& out,int index,char*,int){out=fixture_scene(catalog.ids[static_cast<size_t>(index)].c_str(),index==0?0.0f:2.0f);return true;};
 auto load_model=[&](fake_model& out,const char*,char*,int){++models.loads;out.id=100+models.loads;++models.live;return scene_model_load_result{true,!models.fail};};
 auto unload=[&](fake_model& model){if(model.id){++models.unloads;--models.live;model.id=0;}};
 check(scene_switch_transaction(active_scene,state,active_model,active_index,1,db,support,load_scene,load_model,unload,error,sizeof(error)),error);
 check(active_index==1&&active_scene.metadata.id=="two"&&state.simulation_position.x==2&&models.live==1&&models.unloads==1,"successful transaction");
 const int prior_model=active_model.id;models.fail=true;
 check(!scene_switch_transaction(active_scene,state,active_model,active_index,0,db,support,load_scene,load_model,unload,error,sizeof(error)),"model failure");
 check(active_index==1&&active_scene.metadata.id=="two"&&active_model.id==prior_model&&state.simulation_position.x==2&&models.live==1,"failure preserves active objects");
 const int loads=models.loads,unloads=models.unloads;
 check(scene_reset_current(state,db,support,active_scene,error,sizeof(error)),error);
 check(models.loads==loads&&models.unloads==unloads&&state.scene_frame==0,"reset does not touch model");
 models.fail=false;
 for(int i=0;i<20;++i)check(scene_switch_transaction(active_scene,state,active_model,active_index,1-active_index,db,support,load_scene,load_model,unload,error,sizeof(error)),error);
 check(models.live==1,"only one fake model remains live");
 unload(active_model);check(models.live==0,"normal final unload");
}
int main(){test_success_failure_reset_and_repeated_cycles();return 0;}
```

- [ ] **Step 2: Compile to verify transaction RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_scene_switch.cpp -o /tmp/test_scene_switch
```

Expected: compilation fails because `scene_switch.h` does not exist.

- [ ] **Step 3: Implement generic transaction ordering**

Create `scene_switch.h`:

```cpp
#pragma once
#include "g1_controller_state.h"
struct scene_model_load_result{bool allocated=false;bool ready=false;};

template<class Model,class SceneLoader,class ModelLoader,class Unloader>
static inline bool scene_switch_transaction(scene_pack& active_scene,g1_controller_state& active_state,Model& active_model,int& active_index,int target_index,const database& db,const terrain_support_set& support,SceneLoader load_scene,ModelLoader load_model,Unloader unload,char* error,int capacity)
{
 scene_pack candidate_scene;if(!load_scene(candidate_scene,target_index,error,capacity))return false;
 g1_controller_state candidate_state;if(!g1_controller_state_reset(candidate_state,db,support,candidate_scene,error,capacity))return false;
 Model candidate_model={};const scene_model_load_result loaded=load_model(candidate_model,candidate_scene.mesh_path.c_str(),error,capacity);
 if(!loaded.ready){if(loaded.allocated)unload(candidate_model);return false;}
 Model old_model=active_model;active_model=candidate_model;scene_pack_swap(active_scene,candidate_scene);g1_controller_state_swap(active_state,candidate_state);active_index=target_index;unload(old_model);return true;
}

static inline bool scene_reset_current(g1_controller_state& state,const database& db,const terrain_support_set& support,const scene_pack& scene,char* error,int capacity)
{g1_controller_state candidate;if(!g1_controller_state_reset(candidate,db,support,scene,error,capacity))return false;g1_controller_state_swap(state,candidate);return true;}
```

- [ ] **Step 4: Add strict startup selection to the default-scene load**

Task 5 already loads the motion pack, catalog, default scene, support rows, and
reset state once. Replace only its fixed-default lookup with this selection
before `scene_pack_load`; make `active_scene_index` mutable for later switches:

```cpp
const char* requested_scene=std::getenv("MM_TERRAIN_SCENE");
if(requested_scene==NULL)requested_scene=catalog.default_scene_id.c_str();
int active_scene_index=scene_catalog_find(catalog,requested_scene);
if(active_scene_index<0){std::fprintf(stderr,"G1 scene selection error: unknown MM_TERRAIN_SCENE '%s'\n",requested_scene);return 2;}
scene_pack active_scene;
if(!scene_pack_load(active_scene,terrain_directory,motion_manifest,catalog,active_scene_index,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 scene error [%s]: %s\n",requested_scene,artifact_error);return 2;}
```

Keep Task 5's one-time G1SP load and state reset immediately after this block.
Do not reload or rehash the motion manifest, database, G1TF, or G1SP while
selecting or switching. `default_scene_id` is already populated by Task 4; its
strict index parser requires `grail-curb-default` exactly once in `scene_ids`.

- [ ] **Step 5: Load the initial model and execute pending requests between updates**

After successful `InitWindow`, load the initial model and define the Raylib callbacks:

```cpp
Model terrain_model=LoadModel(active_scene.mesh_path.c_str());
auto model_has_allocation=[](const Model& model){return model.meshes!=NULL||model.materials!=NULL||model.meshMaterial!=NULL||model.bones!=NULL||model.bindPose!=NULL;};
if(!IsModelReady(terrain_model)||terrain_model.meshCount<=0){std::fprintf(stderr,"G1 terrain mesh failed to load: %s\n",active_scene.mesh_path.c_str());if(model_has_allocation(terrain_model))UnloadModel(terrain_model);CloseWindow();return 2;}
auto scene_loader=[&](scene_pack& candidate,int index,char* error,int cap){return scene_pack_load(candidate,terrain_directory,motion_manifest,catalog,index,error,cap);};
auto model_loader=[&](Model& model,const char* path,char* error,int cap){model=LoadModel(path);const bool allocated=model_has_allocation(model);const bool ready=IsModelReady(model)&&model.meshCount>0;if(!ready)scene_error(error,cap,"%s: Raylib model is not ready",path);return scene_model_load_result{allocated,ready};};
auto model_unloader=[&](Model& model){if(model_has_allocation(model))UnloadModel(model);model=Model{};};
int pending_scene_index=-1;bool pending_reset=false;
```

At the first line of the fixed-update lambda, before reading input:

```cpp
if(pending_reset){if(!scene_reset_current(state,db,support_rows,active_scene,artifact_error,sizeof(artifact_error))){controller_exit_code=2;controller_exit_requested=true;std::fprintf(stderr,"G1 scene reset error [%s]: %s\n",active_scene.metadata.id.c_str(),artifact_error);}pending_reset=false;}
if(pending_scene_index>=0&&!controller_exit_requested){const int target=pending_scene_index;pending_scene_index=-1;if(!scene_switch_transaction(active_scene,state,terrain_model,active_scene_index,target,db,support_rows,scene_loader,model_loader,model_unloader,artifact_error,sizeof(artifact_error)))std::fprintf(stderr,"G1 scene switch preserved '%s'; candidate '%s' failed: %s\n",active_scene.metadata.id.c_str(),catalog.ids[static_cast<size_t>(target)].c_str(),artifact_error);}
```

UI buttons only assign `pending_scene_index=(active_scene_index+catalog.ids.size()-1)%catalog.ids.size()`, the analogous next index, or `pending_reset=true`. They never call loaders during `BeginDrawing`/`EndDrawing`.

- [ ] **Step 6: Verify startup errors and fake transaction GREEN**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_scene_switch.cpp -o /tmp/test_scene_switch
/tmp/test_scene_switch
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src controller.cpp -o /tmp/controller_g1_multiscene -L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
MM_TERRAIN_SCENE=../escape /tmp/controller_g1_multiscene >/tmp/bad-scene.out 2>/tmp/bad-scene.err; test "$?" -eq 2
grep -F "unknown MM_TERRAIN_SCENE '../escape'" /tmp/bad-scene.err
```

Expected: fake transaction test exits `0`; controller builds; unsafe startup selection exits `2` before any display error and stderr contains the exact rejected value.

- [ ] **Step 7: Commit transactional selection**

```bash
git add scene_switch.h tests/cpp/test_scene_switch.cpp controller.cpp
git commit -m "feat: switch G1 terrain scenes transactionally"
```

### Task 7: Footprint Walkability Sweep and Safe Stop

**Files:**
- Modify: `terrain_runtime.h`
- Modify: `tests/cpp/test_terrain_runtime.cpp`
- Modify: `controller.cpp` fixed-update input and simulation stages

**Interfaces:**
- Produces: `int walkability_class_at(const walkability_grid&, const heightfield&, float, float)`; out-of-grid/non-finite samples are blocked (`0`).
- Produces `walkability_sweep_result` and `walkability_sweep` as defined in
  Step 3 for a circular `0.20 m` footprint, with centerline spacing at most
  half a cell and conservative node contact.
- Produces `traversability_limit_command` from Step 4; it looks `0.50 s`
  ahead, damps speed scale with `0.08 s` half-life, and reports class, reason,
  boundary point/distance, and commanded/applied speed.
- Produces `traversability_clip_step` from Step 4; it clips an integrated XZ
  step to the last safe fraction, zeros XZ velocity/acceleration, and never
  changes Y or support.
- Produces `walkability_reason_name` with exact CSV values `clear`,
  `blocked-cell`, `out-of-bounds`, and `nonfinite`.
- G1WM class `1` is certified, `2` is stress and remains traversable/diagnostic, `0` is blocked. A deterministic `traverse-or-safe-stop` route therefore attempts stress geometry; if a later acceptance requirement cannot be met, the same checker requires the safe-stop branch.

- [ ] **Step 1: Write lookup, sweep, stress, and stopping tests**

Append to `tests/cpp/test_terrain_runtime.cpp`:

```cpp
static void test_walkability_sweep_and_safe_stop()
{
 heightfield field;field.version=2;initialize_heightfield(field,11,5,0,0,0.10f,0);field.heights.zero();
 walkability_grid grid;grid.nx=11;grid.nz=5;grid.cells.resize(55);grid.cells.set(1);
 for(int z=0;z<5;++z)grid.cells(z*11+7)=0;
 grid.cells(2*11+4)=2;
 check(walkability_class_at(grid,field,0.4f,0.2f)==2,"stress lookup");
 check(walkability_class_at(grid,field,-0.01f,0.2f)==0,"exterior blocked");
 walkability_sweep_result sweep=walkability_sweep(grid,field,vec3(0.1f,8,0.2f),vec3(0.9f,-8,0.2f),0.20f);
 check(sweep.blocked&&sweep.reason==walkability_blocked_cell,"blocked sweep");
 check(sweep.safe_fraction>=0&&sweep.safe_fraction<1&&sweep.point.y==0,"safe fraction and planar point");
 check(sweep.encountered_class==2,"stress recorded before block");
 walkability_sweep_result nonfinite=walkability_sweep(grid,field,vec3(0.1f,0,0.2f),
     vec3(std::numeric_limits<float>::quiet_NaN(),0,0.2f),0.20f);
 check(nonfinite.blocked&&nonfinite.reason==walkability_nonfinite&&
       nonfinite.safe_fraction==0,"non-finite sweep stops safely");
 walkability_grid malformed;malformed.nx=11;malformed.nz=5;
 malformed.cells.resize(1);malformed.cells(0)=1;
 walkability_reason malformed_reason=walkability_clear;
 check(walkability_footprint_class(malformed,field,0.1f,0.2f,0.20f,
       malformed_reason)==0&&malformed_reason==walkability_nonfinite,
       "malformed grid stops safely");

 float invalid_scale=1,invalid_velocity=0;
 traversability_diagnostics invalid_diagnostics={};
 const vec3 invalid_command=traversability_limit_command(
     invalid_scale,invalid_velocity,invalid_diagnostics,grid,field,
     vec3(0.1f,0,0.2f),
     vec3(std::numeric_limits<float>::quiet_NaN(),0,0),1.0f/25.0f);
 check(invalid_diagnostics.blocked&&
       invalid_diagnostics.reason==walkability_nonfinite&&
       invalid_command.x==0&&invalid_command.z==0&&invalid_scale==0,
       "non-finite command stops safely");

 float scale=1,scale_velocity=0;vec3 position(0.1f,0,0.2f),velocity(0.5f,0,0),acceleration;
 float previous_speed=0.5f;bool saw_block=false;
 for(int frame=0;frame<100;++frame){traversability_diagnostics d={};vec3 applied=traversability_limit_command(scale,scale_velocity,d,grid,field,position,vec3(0.5f,0,0),1.0f/25.0f);check(walkability_xz_length(applied)<=previous_speed+1e-6f||!d.blocked,"speed reduces near block");previous_speed=walkability_xz_length(applied);vec3 candidate=position+applied*(1.0f/25.0f);if(!traversability_clip_step(position,candidate,velocity,acceleration,d,grid,field,0.20f))saw_block=true;position=candidate;check(walkability_class_at(grid,field,position.x,position.z)!=0,"center never blocked");}
 check(saw_block||previous_speed<0.01f,"guard reaches safe stop");
 check(position.x<0.50f,"footprint stops before blocked x=0.7");
}
```

Call it from `main` after G1WM loader tests.

- [ ] **Step 2: Compile to verify walkability RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
```

Expected: compilation fails for undefined `walkability_sweep_result` and traversal functions.

- [ ] **Step 3: Implement conservative lookup and sweep**

Add to `terrain_runtime.h` after `walkability_load`:

```cpp
enum walkability_reason{walkability_clear,walkability_blocked_cell,walkability_out_of_bounds,walkability_nonfinite};
struct walkability_sweep_result{bool blocked=false;int encountered_class=1;walkability_reason reason=walkability_clear;float safe_fraction=1,distance=FLT_MAX;vec3 point;};
struct traversability_diagnostics{bool blocked=false;int walkability_class=1;walkability_reason reason=walkability_clear;float distance=FLT_MAX,commanded_speed=0,applied_speed=0;vec3 point;};
static inline const char* walkability_reason_name(walkability_reason reason)
{switch(reason){case walkability_clear:return "clear";case walkability_blocked_cell:return "blocked-cell";case walkability_out_of_bounds:return "out-of-bounds";default:return "nonfinite";}}

static inline float walkability_xz_length(const vec3 value)
{return sqrtf(value.x*value.x+value.z*value.z);}

static inline bool walkability_grid_matches_heightfield(const walkability_grid& grid,const heightfield& field)
{size_t count=0;return grid.nx==field.nx&&grid.nz==field.nz&&grid.nx>=2&&grid.nz>=2&&terrain_float_is_finite(field.origin_x)&&terrain_float_is_finite(field.origin_z)&&terrain_float_is_finite(field.cell_size)&&field.cell_size>0&&terrain_size_multiply(static_cast<size_t>(grid.nx),static_cast<size_t>(grid.nz),count)&&count<=static_cast<size_t>(INT_MAX)&&grid.cells.size==static_cast<int>(count);}

static inline int walkability_class_at(const walkability_grid& grid,const heightfield& field,float x,float z)
{
 if(!terrain_float_is_finite(x)||!terrain_float_is_finite(z)||!walkability_grid_matches_heightfield(grid,field))return 0;
 const float gx=(x-field.origin_x)/field.cell_size,gz=(z-field.origin_z)/field.cell_size;
 if(!terrain_float_is_finite(gx)||!terrain_float_is_finite(gz)||gx<0||gz<0||gx>grid.nx-1||gz>grid.nz-1)return 0;
 const int ix=static_cast<int>(floorf(gx+0.5f)),iz=static_cast<int>(floorf(gz+0.5f));
 if(ix<0||ix>=grid.nx||iz<0||iz>=grid.nz)return 0;const int value=grid.cells(iz*grid.nx+ix);return value<=2?value:0;
}

static inline int walkability_footprint_class(const walkability_grid& grid,const heightfield& field,float x,float z,float radius,walkability_reason& reason)
{
 if(!terrain_float_is_finite(x)||!terrain_float_is_finite(z)||!terrain_float_is_finite(radius)||radius<0||!walkability_grid_matches_heightfield(grid,field)){reason=walkability_nonfinite;return 0;}
 const int x0=static_cast<int>(floorf((x-radius-field.origin_x)/field.cell_size));const int x1=static_cast<int>(ceilf((x+radius-field.origin_x)/field.cell_size));
 const int z0=static_cast<int>(floorf((z-radius-field.origin_z)/field.cell_size));const int z1=static_cast<int>(ceilf((z+radius-field.origin_z)/field.cell_size));int encountered=1;
 for(int iz=z0;iz<=z1;++iz)for(int ix=x0;ix<=x1;++ix){if(ix<0||ix>=grid.nx||iz<0||iz>=grid.nz){reason=walkability_out_of_bounds;return 0;}const float px=field.origin_x+ix*field.cell_size,pz=field.origin_z+iz*field.cell_size;const float dx=px-x,dz=pz-z;if(dx*dx+dz*dz<=radius*radius+1e-8f){const int value=grid.cells(iz*grid.nx+ix);if(value>2){reason=walkability_nonfinite;return 0;}if(value==0){reason=walkability_blocked_cell;return 0;}if(value==2)encountered=2;}}
 reason=walkability_clear;return encountered;
}

static inline walkability_sweep_result walkability_sweep(const walkability_grid& grid,const heightfield& field,vec3 start,vec3 stop,float radius)
{
 walkability_sweep_result out;out.point=vec3(terrain_float_is_finite(start.x)?start.x:0,0,terrain_float_is_finite(start.z)?start.z:0);
 if(!terrain_float_is_finite(start.x)||!terrain_float_is_finite(start.z)||!terrain_float_is_finite(stop.x)||!terrain_float_is_finite(stop.z)||!terrain_float_is_finite(radius)||radius<0||!walkability_grid_matches_heightfield(grid,field)){out.blocked=true;out.reason=walkability_nonfinite;out.safe_fraction=0;out.distance=0;return out;}
 const float dx=stop.x-start.x,dz=stop.z-start.z,length_xz=sqrtf(dx*dx+dz*dz);
 if(!terrain_float_is_finite(length_xz)){out.blocked=true;out.reason=walkability_nonfinite;out.safe_fraction=0;out.distance=0;return out;}
 int steps=static_cast<int>(ceilf(length_xz/(0.5f*field.cell_size)));if(steps<1)steps=1;float previous=0;
 for(int step=0;step<=steps;++step){const float t=static_cast<float>(step)/steps;walkability_reason reason=walkability_clear;const float x=start.x+dx*t,z=start.z+dz*t;const int cls=walkability_footprint_class(grid,field,x,z,radius,reason);if(cls==2)out.encountered_class=2;if(cls==0){out.blocked=true;out.reason=reason;out.safe_fraction=previous;out.distance=length_xz*previous;out.point=vec3(start.x+dx*previous,0,start.z+dz*previous);return out;}previous=t;}
 out.distance=FLT_MAX;out.point=vec3(stop.x,0,stop.z);return out;
}
```

- [ ] **Step 4: Implement smooth command limiting and hard post-integration clipping**

Append:

```cpp
static inline vec3 traversability_limit_command(float& scale,float& scale_velocity,traversability_diagnostics& d,const walkability_grid& grid,const heightfield& field,vec3 position,vec3 command,float dt)
{
 d=traversability_diagnostics();if(!terrain_float_is_finite(scale)||!terrain_float_is_finite(scale_velocity)||!terrain_float_is_finite(position.x)||!terrain_float_is_finite(position.z)||!terrain_float_is_finite(command.x)||!terrain_float_is_finite(command.z)||!terrain_float_is_finite(dt)||dt<=0){scale=0;scale_velocity=0;d.blocked=true;d.walkability_class=0;d.reason=walkability_nonfinite;d.distance=0;d.point=vec3(terrain_float_is_finite(position.x)?position.x:0,0,terrain_float_is_finite(position.z)?position.z:0);return vec3();}
 d.commanded_speed=walkability_xz_length(command);if(d.commanded_speed<=1e-8f){simple_spring_damper_exact(scale,scale_velocity,1,0.08f,dt);return vec3();}
 const vec3 direction(command.x/d.commanded_speed,0,command.z/d.commanded_speed);const float lookahead=maxf(d.commanded_speed*0.50f,d.commanded_speed*dt);const walkability_sweep_result sweep=walkability_sweep(grid,field,position,position+direction*lookahead,0.20f);
 float goal=1;if(sweep.blocked){d.blocked=true;d.reason=sweep.reason;d.distance=sweep.distance;d.point=sweep.point;goal=clampf((sweep.distance-0.02f)/maxf(d.commanded_speed*0.50f,1e-6f),0,1);}d.walkability_class=sweep.encountered_class;simple_spring_damper_exact(scale,scale_velocity,goal,0.08f,dt);scale=clampf(minf(scale,goal),0,1);vec3 out(command.x*scale,0,command.z*scale);d.applied_speed=walkability_xz_length(out);return out;
}

static inline bool traversability_clip_step(vec3 start,vec3& candidate,vec3& velocity,vec3& acceleration,traversability_diagnostics& d,const walkability_grid& grid,const heightfield& field,float radius)
{
 const float preserved_y=candidate.y;const walkability_sweep_result sweep=walkability_sweep(grid,field,start,candidate,radius);if(!sweep.blocked)return true;candidate.x=start.x+(candidate.x-start.x)*sweep.safe_fraction;candidate.z=start.z+(candidate.z-start.z)*sweep.safe_fraction;candidate.y=preserved_y;velocity.x=velocity.z=0;acceleration.x=acceleration.z=0;d.blocked=true;d.reason=sweep.reason;d.distance=sweep.distance;d.point=sweep.point;d.applied_speed=0;return false;
}
```

- [ ] **Step 5: Integrate the guard at the exact data-flow points**

After computing the unguarded desired velocity, preserve it for logging and limit the value used by trajectory prediction/matching:

```cpp
const vec3 commanded_velocity=desired_velocity_curr;traversability_diagnostics traversal={};desired_velocity_curr=traversability_limit_command(state.traversal_speed_scale,state.traversal_speed_scale_velocity,traversal,active_scene.walkability,active_scene.terrain,state.simulation_position,commanded_velocity,dt);state.blocked=traversal.blocked;state.walkability_class=traversal.walkability_class;state.blocked_distance=traversal.distance;state.blocked_point=traversal.point;
```

Immediately around the existing simulation integration:

```cpp
const vec3 simulation_before=state.simulation_position;simulation_positions_update(state.simulation_position,state.simulation_velocity,state.simulation_acceleration,state.desired_velocity,simulation_velocity_halflife,dt);traversability_clip_step(simulation_before,state.simulation_position,state.simulation_velocity,state.simulation_acceleration,traversal,active_scene.walkability,active_scene.terrain,0.20f);
```

Then call `walkability_footprint_class` with `active_scene.walkability`,
`active_scene.terrain`, `state.simulation_position.x`,
`state.simulation_position.z`, `0.20f`, and `current_reason`, assigning its
return value to `state.walkability_class`. This logged field is the
current accepted footprint class, never the prospective class returned by the
lookahead sweep. Keep the prospective reason/distance/point in `traversal`.

Do not call either guard function on support height; support state is not an argument.

- [ ] **Step 6: Run walkability GREEN in strict/release/sanitizer builds**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime_strict && /tmp/test_terrain_runtime_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime_release && /tmp/test_terrain_runtime_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime_san && ASAN_OPTIONS=detect_leaks=1 /tmp/test_terrain_runtime_san
```

Expected: all tests exit `0`, no warnings, no sanitizer report; stress cells are diagnostic/traversable and the blocked footprint stops at least `0.20 m` before the blocked node center.

- [ ] **Step 7: Commit safe stopping**

```bash
git add terrain_runtime.h tests/cpp/test_terrain_runtime.cpp controller.cpp
git commit -m "feat: stop G1 motion at blocked terrain"
```

### Task 8: Integrate Support Retargeting Without Changing Matching

**Files:**
- Create: `tests/cpp/test_support_matching.cpp`
- Modify: `controller.cpp` query/search/inertialization/root-adjust/FK stages and terrain-weight UI state

**Interfaces:**
- Consumes `support_observation_build` after pose inertialization and a preliminary support-local FK, using current selected database frame and recorded contacts.
- Calls `support_frame_update` with `state.support_observation_now`,
  `state.transitioned`, and `dt` in the same frame as
  `inertialize_pose_transition`; the two inertializers rebase together.
- Applies `horizontal_adjust_character_position*` and `horizontal_clamp_character_position` to the support-local root, records exact XZ/Y displacements, then calls `support_pose_apply` before final FK.
- Query construction/search remains before support observation/update and uses active-scene G1HF only. Support state is never an argument to `database_search`, query feature builders, feature normalization, or bounds.
- Produces a persistent `effective_terrain_weight`; UI edits a separate `requested_terrain_weight` and visibly says `unapplied` until the existing Apply/Rebuild action succeeds.
- IK stays `static constexpr bool ik_enabled = false`; the inherited LAFAN flat-floor contact/IK block is not invoked or adapted.

- [ ] **Step 1: Write a search-invariance regression**

Create `tests/cpp/test_support_matching.cpp`:

```cpp
#include "database.h"
#include "g1_skeleton.h"
#include "support_runtime.h"
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(bool v,const char* m){if(!v){std::fprintf(stderr,"support matching test failed: %s\n",m);std::exit(1);}}
static uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
int main(){database db;db.bone_positions.resize(3,G1_BoneCount);db.features.resize(3,31);db.features_offset.resize(31);db.features_scale.resize(31);db.range_starts.resize(1);db.range_stops.resize(1);db.terrain_features.resize(3,4);db.features.zero();db.features_offset.zero();db.features_scale.set(1);db.features(0,0)=2;db.features(1,0)=0;db.features(2,0)=3;db.range_starts(0)=0;db.range_stops(0)=3;database_build_bounds(db);array1d<float> query(31);query.zero();int before=0;float before_cost=FLT_MAX;database_search(before,before_cost,db,query,0,0,1);array1d<vec3> pose(G1_BoneCount),retargeted(G1_BoneCount);pose.zero();pose(G1_Hips)=vec3(0,0.8f,0);support_pose_apply(retargeted,pose,0.36f);int after=0;float after_cost=FLT_MAX;database_search(after,after_cost,db,query,0,0,1);check(before==1&&after==before,"selected frame invariant");check(bits(before_cost)==bits(after_cost),"selected cost bit invariant");check(retargeted(G1_Simulation).y==0.36f&&retargeted(G1_Hips).y==pose(G1_Hips).y,"only support transform changed");return 0;}
```

- [ ] **Step 2: Compile/run RED before controller integration guard exists**

Run the test now; it already passes the pure helper, then use a source assertion as the RED integration gate:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_support_matching.cpp -o /tmp/test_support_matching
/tmp/test_support_matching
rg -n 'support_observation_build\(' controller.cpp
```

Expected: C++ test exits `0`; the final `rg` exits `1`, providing the RED
integration result because controller support integration is still absent.

- [ ] **Step 3: Build support observation immediately after inertialization**

After `inertialize_pose_update` and before adjustment/clamping, run support-local FK and update the scalar transform:

```cpp
forward_kinematics_full(state.global_bone_positions,state.global_bone_rotations,state.bone_positions,state.bone_rotations,db.bone_parents);
if(!support_observation_build(state.support_observation_now,support_rows,state.frame_index,active_scene.terrain,state.global_bone_positions(G1_Simulation),state.global_bone_positions(G1_LeftToe),state.global_bone_positions(G1_RightToe),state.curr_bone_contacts(0),state.curr_bone_contacts(1),artifact_error,sizeof(artifact_error))||!support_frame_update(state.support,state.support_observation_now,state.transitioned,dt,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 support runtime error scene=%s frame=%d: %s\n",active_scene.metadata.id.c_str(),state.frame_index,artifact_error);controller_exit_code=2;controller_exit_requested=true;return;}
```

Set `state.transitioned=false` at the beginning of each update and true only where `best_index != prior_index`. Sequential range advance is not a source-frame change; a search transition is.

- [ ] **Step 4: Make adjustment/clamp provably horizontal and apply world support once**

Replace only the three positional calls; keep rotation/yaw logic unchanged:

```cpp
const vec3 before_adjustment=state.bone_positions(G1_Simulation);
vec3 adjusted_position=adjustment_by_velocity_enabled?horizontal_adjust_character_position_by_velocity(before_adjustment,state.bone_velocities(G1_Simulation),state.simulation_position,adjustment_position_max_ratio,adjustment_position_halflife,dt):horizontal_adjust_character_position(before_adjustment,state.simulation_position,adjustment_position_halflife,dt);
state.adjustment_xz=horizontal_length(adjusted_position-before_adjustment);state.adjustment_y=adjusted_position.y-before_adjustment.y;
inertialize_root_adjust(state.bone_offset_positions(0),state.transition_src_position,state.transition_src_rotation,state.transition_dst_position,state.transition_dst_rotation,state.bone_positions(0),state.bone_rotations(0),adjusted_position,adjusted_rotation);
const vec3 before_clamp=state.bone_positions(0);adjusted_position=horizontal_clamp_character_position(before_clamp,state.simulation_position,clamping_max_distance);state.clamp_xz=horizontal_length(adjusted_position-before_clamp);state.clamp_y=adjusted_position.y-before_clamp.y;
inertialize_root_adjust(state.bone_offset_positions(0),state.transition_src_position,state.transition_src_rotation,state.transition_dst_position,state.transition_dst_rotation,state.bone_positions(0),state.bone_rotations(0),adjusted_position,adjusted_rotation);
if(state.adjustment_y!=0.0f||state.clamp_y!=0.0f){std::fprintf(stderr,"G1 horizontal-root invariant failed\n");controller_exit_code=2;controller_exit_requested=true;return;}
support_pose_apply(state.adjusted_bone_positions,state.bone_positions,state.support.height);state.adjusted_bone_rotations=state.bone_rotations;
forward_kinematics_full(state.global_bone_positions,state.global_bone_rotations,state.adjusted_bone_positions,state.adjusted_bone_rotations,db.bone_parents);
```

Delete the later assignment that overwrites `adjusted_bone_positions` from `bone_positions`. Leave the legacy IK branch behind `if constexpr (ik_enabled)` or remove only its invocation; no IK output may overwrite the support-retargeted pose.

After the three call sites use the tested horizontal helpers, delete the now
unreferenced controller-local definitions of `adjust_character_position`,
`adjust_character_position_by_velocity`, and `clamp_character_position`.
Keeping dead 3D alternatives would make the planar invariant ambiguous.

- [ ] **Step 5: Query the active scene and display honest terrain-weight state**

Replace every `runtime_terrain` use with `active_scene.terrain`. At startup:

```cpp
float requested_terrain_weight=parsed_terrain_weight;float effective_terrain_weight=requested_terrain_weight;
```

The slider edits requested weight. The label is:

```cpp
GuiLabel((Rectangle){40,205,250,20},requested_terrain_weight==effective_terrain_weight?TextFormat("effective terrain %.3f",effective_terrain_weight):TextFormat("requested %.3f (unapplied %.3f)",requested_terrain_weight,effective_terrain_weight));
```

The Apply button passes `requested_terrain_weight`, validates the resulting 31D database, then assigns `effective_terrain_weight=requested_terrain_weight`. Logs and all cost/UI claims use only `effective_terrain_weight`.

- [ ] **Step 6: Run integration GREEN and assert no forbidden coupling**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. tests/cpp/test_support_matching.cpp -o /tmp/test_support_matching && /tmp/test_support_matching
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src controller.cpp -o /tmp/controller_g1_multiscene -L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
rg -n 'support_observation_build\(' controller.cpp
! rg -n 'database_(search|build_matching_features)\([^;]*(support|terrain_support)' controller.cpp
! rg -n 'const bool ik_enabled = true|MM_IK' controller.cpp
```

Expected: unit test/controller build exit `0`; one support observation call exists after inertialization; support metadata never reaches matcher APIs; IK cannot be enabled.

- [ ] **Step 7: Commit support integration**

```bash
git add controller.cpp tests/cpp/test_support_matching.cpp
git commit -m "feat: preserve G1 motion matching across world levels"
```

### Task 9: Deterministic Routes and One Shared Runtime Diagnostic Snapshot

**Files:**
- Create: `route_runtime.h`
- Create: `g1_runtime_diagnostics.h`
- Create: `tests/cpp/test_route_runtime.cpp`
- Modify: `g1_controller_state.h`
- Modify: `motion_match_log.h`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`
- Modify: `controller.cpp` deterministic configuration, row population, UI,
  blocked marker, and controlled-error path

**Interfaces:**
- Preserves prerequisite modes `sequential`, `flat`, and `terrain`. Adds
  `MM_TEST_MODE=route`, which requires `MM_TEST_ROUTE=<exact route id>`, and
  `MM_TEST_MODE=scene-cycle`, which rejects `MM_TEST_ROUTE` and uses
  `MM_SCENE_DWELL_FRAMES` (default `25`). Both still use `MM_TEST_FRAMES`,
  `MM_LOG`, `MM_TERRAIN_WEIGHT`, and `MM_TERRAIN_SCENE`.
- Route input is an open-loop fixed-frame traversal of metadata waypoints at
  exactly `0.50 m/s` and `dt=0.04`. It never depends on selected motion,
  simulation position, wall contact, wall-clock time, or terrain weight; A/B
  runs therefore receive bit-identical commands.
- For every route with `landing_hold_seconds > 0`, published metadata must have
  at least four waypoints and waypoint index `2` is the locked landing-hold
  waypoint. The driver inserts exactly
  `ceil(landing_hold_seconds / dt)` zero-command frames immediately after
  reaching it. Routes with zero hold have no special waypoint.
- Appends runtime columns after, and never renames/reorders, immutable Gate A
  columns. UI and CSV consume one `g1_runtime_diagnostic_snapshot`.
- Any non-finite query, pose diagnostic, support observation/state, traversal
  value, or snapshot requests exit code `2`; cleanup remains outside the loop.

- [ ] **Step 1: Write route and exact logger-suffix tests**

Create `tests/cpp/test_route_runtime.cpp`:

```cpp
#include "route_runtime.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(bool v,const char* m){if(!v){std::fprintf(stderr,"route runtime test failed: %s\n",m);std::exit(1);}}
static uint32_t bits(float v){uint32_t out;std::memcpy(&out,&v,4);return out;}
int main()
{
 scene_route route;route.id="corner";route.expected_outcome="traverse";route.walkability_class=1;route.landing_hold_seconds=2;route.waypoints_xz={{0,0},{1,0},{1,1},{2,1}};
 deterministic_route_sample a,b;char error[256]={};
 check(deterministic_route_command(a,route,0,0.04f,0.50f,error,sizeof(error)),error);check(a.waypoint==1&&bits(a.command.x)==bits(0.50f)&&a.command.z==0&&!a.complete,"first segment");
 check(deterministic_route_command(a,route,49,0.04f,0.50f,error,sizeof(error)),error);check(a.waypoint==1&&a.command.x==0.50f,"last frame first segment");
 check(deterministic_route_command(a,route,50,0.04f,0.50f,error,sizeof(error)),error);check(a.waypoint==2&&a.command.x==0&&a.command.z==0.50f,"second segment");
 check(deterministic_route_command(a,route,100,0.04f,0.50f,error,sizeof(error)),error);check(!a.complete&&a.waypoint==2&&a.command.x==0&&a.command.z==0,"landing hold begins");
 check(deterministic_route_command(a,route,149,0.04f,0.50f,error,sizeof(error)),error);check(!a.complete&&a.waypoint==2&&a.command.x==0&&a.command.z==0,"landing hold lasts two seconds");
 check(deterministic_route_command(a,route,150,0.04f,0.50f,error,sizeof(error)),error);check(!a.complete&&a.waypoint==3&&a.command.x==0.50f&&a.command.z==0,"motion resumes after hold");
 check(deterministic_route_command(a,route,200,0.04f,0.50f,error,sizeof(error)),error);check(a.complete&&a.command.x==0&&a.command.z==0,"route complete");
 check(deterministic_route_motion_frames(route)==200,"motion count includes hold");
 for(int frame=0;frame<225;++frame){check(deterministic_route_command(a,route,frame,0.04f,0.50f,error,sizeof(error)),error);check(deterministic_route_command(b,route,frame,0.04f,0.50f,error,sizeof(error)),error);check(bits(a.command.x)==bits(b.command.x)&&bits(a.command.z)==bits(b.command.z)&&a.waypoint==b.waypoint&&a.complete==b.complete,"repeatable route input");}
 route.waypoints_xz[1]=route.waypoints_xz[0];check(!deterministic_route_command(a,route,0,0.04f,0.50f,error,sizeof(error)),"zero segment rejected");return 0;
}
```

In `tests/python/test_runtime_log.py`, add this literal tuple assertion. This is
the cross-plan compatibility lock:

```python
RUNTIME_SUFFIX = (
    "source_name", "source_terrain", "source_index", "continuation_cost",
    "source_root_height", "source_left_toe_height", "source_right_toe_height",
    "runtime_support_root_height", "runtime_support_left_toe_height",
    "runtime_support_right_toe_height", "support_root_delta",
    "support_left_toe_delta", "support_right_toe_delta", "support_height",
    "support_velocity", "support_source", "airborne_frames", "left_contact",
    "right_contact", "support_retargeted_hips_y", "ik_adjusted_hips_y",
    "simulation_x", "simulation_z", "walkability_class", "blocked",
    "blocked_reason", "blocked_distance", "blocked_point_x",
    "blocked_point_z", "commanded_speed", "applied_speed", "route_waypoint",
    "route_complete", "route_target_height", "scene_generation",
    "scene_frame", "scene_reset_count", "scene_switch_failed",
    "motion_pack_load_count", "model_load_count", "model_unload_count",
    "live_model_count",
)

def test_runtime_columns_append_after_gate_a(self):
    self.assertEqual(tuple(RUNTIME_COLUMNS[-len(RUNTIME_SUFFIX):]), RUNTIME_SUFFIX)
    self.assertEqual(tuple(RUNTIME_COLUMNS[:-len(RUNTIME_SUFFIX)]), GATE_A_COLUMNS)
```

- [ ] **Step 2: Run RED for the new public contracts**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_route_runtime.cpp -o /tmp/test_route_runtime
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log.RuntimeLogTests.test_runtime_columns_append_after_gate_a -v
```

Expected: C++ compilation fails because `route_runtime.h` is missing; Python
fails because the checker does not yet export appended columns.

- [ ] **Step 3: Implement the fixed-frame route driver**

Create `route_runtime.h`:

```cpp
#pragma once
#include "scene_runtime.h"
#include <cfloat>
#include <cmath>
struct deterministic_route_sample{vec3 command;int waypoint=0;bool complete=false;};
static inline bool deterministic_route_command(deterministic_route_sample& out,const scene_route& route,int frame,float dt,float speed,char* error,int capacity)
{
 if(frame<0||!terrain_float_is_finite(dt)||dt<=0||!terrain_float_is_finite(speed)||speed<=0||route.waypoints_xz.size()<2||!terrain_float_is_finite(route.landing_hold_seconds)||route.landing_hold_seconds<0|| (route.landing_hold_seconds>0&&route.waypoints_xz.size()<4))return scene_error(error,capacity,"route '%s': invalid frame, dt, speed, hold, or waypoint count",route.id.c_str());
 int cursor=0;for(size_t i=0;i+1<route.waypoints_xz.size();++i){const float dx=route.waypoints_xz[i+1].first-route.waypoints_xz[i].first;const float dz=route.waypoints_xz[i+1].second-route.waypoints_xz[i].second;const float length=std::sqrt(dx*dx+dz*dz);if(!terrain_float_is_finite(length)||length<=1e-6f)return scene_error(error,capacity,"route '%s': segment %zu has zero or non-finite length",route.id.c_str(),i);const int frames=static_cast<int>(std::ceil(length/(speed*dt)));if(frame<cursor+frames){deterministic_route_sample sample;sample.command=vec3(speed*dx/length,0,speed*dz/length);sample.waypoint=static_cast<int>(i)+1;out=sample;return true;}cursor+=frames;if(i+1==2&&route.landing_hold_seconds>0){const int hold=static_cast<int>(std::ceil(route.landing_hold_seconds/dt));if(frame<cursor+hold){deterministic_route_sample sample;sample.waypoint=2;out=sample;return true;}cursor+=hold;}}
 deterministic_route_sample sample;sample.waypoint=static_cast<int>(route.waypoints_xz.size())-1;sample.complete=true;out=sample;return true;
}
static inline int deterministic_route_motion_frames(const scene_route& route,float dt=0.04f,float speed=0.50f)
{int total=0;for(size_t i=0;i+1<route.waypoints_xz.size();++i){const float dx=route.waypoints_xz[i+1].first-route.waypoints_xz[i].first;const float dz=route.waypoints_xz[i+1].second-route.waypoints_xz[i].second;total+=static_cast<int>(std::ceil(std::sqrt(dx*dx+dz*dz)/(speed*dt)));if(i+1==2&&route.landing_hold_seconds>0)total+=static_cast<int>(std::ceil(route.landing_hold_seconds/dt));}return total;}
static inline float deterministic_route_target_height(const scene_route& route,const heightfield& terrain)
{struct bin{int key=0,count=0;float sum=0;};std::vector<bin> bins;const float base=heightfield_sample(terrain,route.waypoints_xz.front().first,route.waypoints_xz.front().second);for(size_t i=0;i+1<route.waypoints_xz.size();++i){const float dx=route.waypoints_xz[i+1].first-route.waypoints_xz[i].first;const float dz=route.waypoints_xz[i+1].second-route.waypoints_xz[i].second;const float length=std::sqrt(dx*dx+dz*dz);int steps=static_cast<int>(std::ceil(length/(0.5f*terrain.cell_size)));if(steps<1)steps=1;for(int step=0;step<=steps;++step){const float t=static_cast<float>(step)/steps;const float height=heightfield_sample(terrain,route.waypoints_xz[i].first+dx*t,route.waypoints_xz[i].second+dz*t);if(std::fabs(height-base)<=0.02f)continue;const int key=static_cast<int>(std::lround(height*100.0f));size_t found=0;while(found<bins.size()&&bins[found].key!=key)++found;if(found==bins.size()){bin value;value.key=key;bins.push_back(value);}++bins[found].count;bins[found].sum+=height;}}if(bins.empty())return base;size_t best=0;for(size_t i=1;i<bins.size();++i)if(bins[i].count>bins[best].count||(bins[i].count==bins[best].count&&bins[i].key>bins[best].key))best=i;return bins[best].sum/bins[best].count;}
```

The target is diagnostic only; it never feeds trajectory, matcher, support, or
walkability logic.

- [ ] **Step 4: Extend deterministic parsing without changing Gate A**

Extend prerequisite `g1_test_mode` with `G1_TestRoute` and
`G1_TestSceneCycle`, and `g1_test_config` with
`int scene_dwell_frames=25`. Parsing rules are exact:

- unset mode: live; reject stray `MM_TEST_ROUTE` or `MM_SCENE_DWELL_FRAMES`;
- `sequential|flat|terrain`: preserve behavior; reject both new variables;
- `route`: require nonempty `MM_TEST_ROUTE`, positive `MM_TEST_FRAMES`, and
  reject `MM_SCENE_DWELL_FRAMES`;
- `scene-cycle`: reject `MM_TEST_ROUTE`, parse dwell in `[1,10000]` with
  default `25`, and require frame limit at least `14*dwell`.

After startup scene load, resolve route mode once with `scene_route_find`; an
unknown route exits `2` before `InitWindow`. At input use only:

```cpp
deterministic_route_sample route_sample;
if(test_config.mode==G1_TestRoute){const scene_route& route=active_scene.metadata.routes[static_cast<size_t>(state.route_index)];if(!deterministic_route_command(route_sample,route,state.route_frames,dt,0.50f,artifact_error,sizeof(artifact_error))){controller_exit_code=2;controller_exit_requested=true;}else{desired_velocity_curr=route_sample.command;desired_strafe=false;}}
```

Increment `state.route_frames` after writing the row. Scene-cycle queues next
scene when just-logged `scene_frame+1` is divisible by dwell; the transaction
still runs only at the next update boundary.

- [ ] **Step 5: Append exact logger suffix and build one shared snapshot**

In `resources/check_g1_runtime_log.py`, retain `GATE_A_COLUMNS` exactly and
define `RUNTIME_COLUMNS = GATE_A_COLUMNS + list(RUNTIME_SUFFIX)`. In
`motion_match_log.h`, append identically named fields to
`motion_match_log_row`; string fields are `const char*` defaulting to `""`.
Remove only the final newline from existing Gate A header/row formats and emit:

```text
,source_name,source_terrain,source_index,continuation_cost,source_root_height,source_left_toe_height,source_right_toe_height,runtime_support_root_height,runtime_support_left_toe_height,runtime_support_right_toe_height,support_root_delta,support_left_toe_delta,support_right_toe_delta,support_height,support_velocity,support_source,airborne_frames,left_contact,right_contact,support_retargeted_hips_y,ik_adjusted_hips_y,simulation_x,simulation_z,walkability_class,blocked,blocked_reason,blocked_distance,blocked_point_x,blocked_point_z,commanded_speed,applied_speed,route_waypoint,route_complete,route_target_height,scene_generation,scene_frame,scene_reset_count,scene_switch_failed,motion_pack_load_count,model_load_count,model_unload_count,live_model_count\n
```

Create `g1_runtime_diagnostics.h` with `g1_runtime_diagnostic_snapshot` using
those exact names/types and:

```cpp
static inline bool g1_runtime_diagnostics_build(g1_runtime_diagnostic_snapshot& out,const motion_pack_manifest& manifest,const g1_controller_state& state,const traversability_diagnostics& traversal,const deterministic_route_sample& route,float route_target_height,int scene_generation,int scene_reset_count,bool scene_switch_failed,int motion_pack_load_count,int model_load_count,int model_unload_count,char* error,int capacity);
```

The builder resolves source with `motion_source_for_frame`, copies support
arrays from `state.support_observation_now`, uses
`support_source_name(state.support.source)` and final support-retargeted FK Hips
Y, sets `ik_adjusted_hips_y` equal to it because IK is disabled, uses
`continuation_cost=state.incumbent_cost`, maps `blocked_reason` with
`walkability_reason_name(traversal.reason)`, takes current `walkability_class`
from state, and computes live models as
loads-minus-unloads. Reject out-of-range sources, invalid walkability classes,
negative counters, live model count other than `1` while looping, and every
non-finite float. All finite checks use the inherited bit-based
`terrain_float_is_finite`, including release/`-ffast-math`; do not replace it
with `std::isfinite` in this path.

Keep `scene_generation`, `scene_reset_count`, `motion_pack_load_count`,
`model_load_count`, and `model_unload_count` as persistent controller locals.
Initial values after startup are `0,1,1,1,0`. Increment generation/reset count
on successful switch or Reset, load/unload only in model callbacks, and set
`scene_switch_failed` for exactly the row after a rejected transaction.
Populate the row from this snapshot and clear the pulse only after `write`.

- [ ] **Step 6: Make UI/rendering consume the snapshot and errors exit normally**

Keep the last logged snapshot. Render from it: active ID; Previous/Next/Reset;
requested/effective weight; support height/velocity/source; source
index/name/terrain; contacts; blocked class/reason/distance; generation/frame;
and CSV enabled state. Draw its `blocked_point` red only when blocked. UI must
not resample terrain or recompute strings.

Keep `SetTargetFPS(25)`. Validate all 31 query floats, raw/inertialized/final
pose diagnostics, support observation/state, traversal diagnostics, and the
snapshot each update. On failure print one line beginning
`G1 controlled runtime error scene=<id> frame=<n>:`; set exit code `2` and
leave the loop normally.

- [ ] **Step 7: Run route/logger GREEN and strict controller build**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_route_runtime.cpp -o /tmp/test_route_runtime
/tmp/test_route_runtime
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -I/home/ubuntu/apps/raylib/src \
  -I/home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_g1_multiscene -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
MM_TEST_MODE=route MM_TEST_FRAMES=100 \
  /tmp/controller_g1_multiscene >/tmp/missing-route.out 2>/tmp/missing-route.err
test "$?" -eq 2
grep -F 'MM_TEST_ROUTE is required for route mode' /tmp/missing-route.err
```

Expected: native/Python tests pass; strict controller build is warning-free;
missing route fails before Raylib; no Gate A column or mode changes.

- [ ] **Step 8: Commit deterministic diagnostics**

```bash
git add route_runtime.h g1_runtime_diagnostics.h \
  tests/cpp/test_route_runtime.cpp g1_controller_state.h motion_match_log.h \
  resources/check_g1_runtime_log.py tests/python/test_runtime_log.py \
  controller.cpp
git commit -m "test: log deterministic G1 multiscene routes"
```

### Task 10: Gate C, Gate D, and Terrain-Weight A/B Acceptance

**Files:**
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`
- Modify: `controller.cpp` only if a diagnostic is wired to the wrong
  already-computed stage; do not tune thresholds in controller code
- Generated, never committed: `/tmp/g1-multiscene-runtime/gate-c-*.csv`
- Generated, never committed: `/tmp/g1-multiscene-runtime/gate-d-*.csv`

**Interfaces:**
- Produces `check_gate_c(rows) -> dict`,
  `check_mixed_multilevel(rows) -> dict`, `check_gate_d(rows) -> dict`, and an
  extended `compare_control(treatment,control) -> tuple[float,float]`.
- Adds CLI flags `--gate-c` and `--gate-d`; positional `LOG`, `--gate-a`, and
  `--compare-control CONTROL.csv` remain compatible.
- Gate C owns the three traversable stair routes, two certified ramps, and
  `mixed-multilevel:full-course`, including its continued elevated matching.
  Gate D owns the two genuinely blocked `blocked-course` routes. The
  class-`2` `ramp-15-stress` route remains traversable at this layer; the later
  Gate E plan must accept either a clearance-valid traversal or a safe stop.

- [ ] **Step 1: Add synthetic passing and threshold-failure tests**

Extend the test `row` helper so every suffix has a valid default, then add:

```python
def runtime_row(frame, **changes):
    values = row(frame, 100 + frame)
    values.update({
        "source_name": "terrain_curbs__fixture", "source_terrain": "fixture",
        "source_index": "1", "continuation_cost": "2.0",
        "source_root_height": "0", "source_left_toe_height": "0",
        "source_right_toe_height": "0", "runtime_support_root_height": "0",
        "runtime_support_left_toe_height": "0",
        "runtime_support_right_toe_height": "0", "support_root_delta": "0",
        "support_left_toe_delta": "0", "support_right_toe_delta": "0",
        "support_height": "0", "support_velocity": "0",
        "support_source": "both", "airborne_frames": "0",
        "left_contact": "1", "right_contact": "1",
        "support_retargeted_hips_y": "0.8", "ik_adjusted_hips_y": "0.8",
        "simulation_x": "0", "simulation_z": str(frame * .02),
        "walkability_class": "1", "blocked": "0",
        "blocked_reason": "clear", "blocked_distance": "3.4e38",
        "blocked_point_x": "0", "blocked_point_z": "0",
        "commanded_speed": ".5", "applied_speed": ".5",
        "route_waypoint": "1", "route_complete": "0",
        "route_target_height": ".36", "scene_generation": "0",
        "scene_frame": str(frame), "scene_reset_count": "1",
        "scene_switch_failed": "0", "motion_pack_load_count": "1",
        "model_load_count": "1", "model_unload_count": "0",
        "live_model_count": "1", "matching_enabled": "1",
        "support_retargeting_enabled": "1", "ik_enabled": "0",
        "adjustment_y": "0", "clamp_y": "0", "fixed_dt": ".04",
        "mode": "route", "route": "fixture-route", "scene_id": "fixture",
    })
    values.update({key: str(value) for key, value in changes.items()})
    return values

def gate_c_rows():
    rows = []
    for frame in range(170):
        if frame < 20: height = 0.0
        elif frame < 56: height = min(.36, (frame - 20) * .01)
        elif frame < 110: height = .36
        elif frame < 146: height = max(0.0, .36 - (frame - 110) * .01)
        else: height = 0.0
        rows.append(runtime_row(
            frame, terrain0=(.12 if frame >= 10 else 0),
            runtime_support_root_height=height,
            runtime_support_left_toe_height=height,
            runtime_support_right_toe_height=height,
            support_root_delta=height, support_left_toe_delta=height,
            support_right_toe_delta=height, support_height=height,
            support_retargeted_hips_y=.8 + height,
            ik_adjusted_hips_y=.8 + height, rendered_hips_y=.8 + height,
            route_complete=int(frame >= 146)))
    return rows

def gate_d_rows():
    rows = []
    for frame in range(100):
        blocked = frame >= 20
        speed = max(0.0, .5 - max(0, frame - 20) * .05)
        rows.append(runtime_row(
            frame, scene_id="blocked-course", route="wall-safe-stop",
            blocked=int(blocked),
            blocked_reason=("blocked-cell" if blocked else "clear"),
            blocked_distance=(.03 if blocked else 3.4e38),
            applied_speed=speed, simulation_z=min(frame * .02, .58),
            route_complete=int(frame >= 60)))
    return rows

def test_gate_c_accepts_persistent_landing_and_descent(self):
    self.assertGreaterEqual(check_gate_c(gate_c_rows())["landing_frames"], 50)

def test_gate_c_rejects_non_source_hips_jump(self):
    rows = gate_c_rows(); rows[80]["support_retargeted_hips_y"] = "1.3"
    with self.assertRaisesRegex(ValueError, "non-source Hips"):
        check_gate_c(rows)

def test_mixed_checker_requires_elevated_blocks_and_return_ramp(self):
    report = check_mixed_multilevel(mixed_multilevel_rows())
    self.assertGreaterEqual(report["elevated_matching_frames"], 50)
    for mutation, diagnostic in (
        (("elevated", 25, "matching_enabled", "0"), "elevated matching"),
        (("block-2", 4, "runtime_support_root_height", ".40"),
         "block plateaus"),
        (("ramp", 8, "runtime_support_root_height", ".40"),
         "return ramp"),
        (("base", 2, "runtime_support_root_height", ".08"),
         "returned to base"),
    ):
        rows = mixed_multilevel_rows()
        mutate_mixed_region(rows, *mutation)
        with self.subTest(diagnostic=diagnostic):
            with self.assertRaisesRegex(ValueError, diagnostic):
                check_mixed_multilevel(rows)

def test_gate_d_accepts_safe_stop_and_rejects_blocked_footprint(self):
    self.assertGreaterEqual(check_gate_d(gate_d_rows())["stopped_frames"], 25)
    rows = gate_d_rows(); rows[50]["walkability_class"] = "0"
    with self.assertRaisesRegex(ValueError, "entered blocked"):
        check_gate_d(rows)

def test_ab_requires_identical_script(self):
    treatment = gate_c_rows(); control = gate_c_rows()
    for a, b in zip(treatment, control):
        a["selected_terrain_error"] = ".1"; a["effective_terrain_weight"] = "4"
        b["selected_terrain_error"] = ".4"; b["effective_terrain_weight"] = "0"
    treatment[20]["route_waypoint"] = "9"
    with self.assertRaisesRegex(ValueError, "scripted input"):
        compare_control(treatment, control)
```

- [ ] **Step 2: Run checker RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log.RuntimeLogTests.test_gate_c_accepts_persistent_landing_and_descent \
  tests.python.test_runtime_log.RuntimeLogTests.test_mixed_checker_requires_elevated_blocks_and_return_ramp \
  tests.python.test_runtime_log.RuntimeLogTests.test_gate_d_accepts_safe_stop_and_rejects_blocked_footprint \
  tests.python.test_runtime_log.RuntimeLogTests.test_ab_requires_identical_script -v
```

Expected: missing Gate C/D/mixed functions and old A/B script comparison fail.

- [ ] **Step 3: Make base validation reset-aware and validate the suffix**

Extend `check_rows` without weakening a single-scene Gate A log:

- require the exact `RUNTIME_COLUMNS` header for runtime gates;
- require `fixed_dt == 0.04`, numeric suffix fields finite, booleans in
  `{0,1}`, nonempty safe scene/source/support strings (`route` may be empty
  only outside route mode), `ik_enabled == 0`, and both logged Y
  adjustment/clamp values exactly zero;
- within one generation require `scene_frame` to increment and preserve the
  existing sequential-frame/transition rules;
- across a generation change require `previous+1`, `scene_frame==0`, and
  `scene_reset_count==previous+1`; skip only the cross-reset frame comparison;
- require `motion_pack_load_count==1` and `live_model_count==1` on every row;
- run `check_substride` separately per generation.

Add `_longest_run(indices)` and `_support_alignment_error(row)`. For an active
toe it compares `source_toe_height + support_height` to runtime toe height;
with both contacts it takes their maximum; without contacts it uses the root.

- [ ] **Step 4: Implement Gate C and exact A/B checks**

`check_gate_c(rows)` calls `check_rows` and requires:

1. one route-mode scene/route, matching/support enabled, IK disabled;
2. first `max(abs(terrain0..3)) > 0.02` precedes the first root-support rise
   more than `0.04 m` above the median of the first 20 rows;
3. after activation, a source has `source_index>0` and non-flat terrain;
4. every transition beats the incumbent and sub-stride checks pass;
5. the longest block with runtime root within `0.02 m` of
   `route_target_height` is at least 50 rows; every row in it has support
   alignment error at most `0.02 m` and matching remains enabled;
6. after that block, runtime root returns within `0.02 m` of initial baseline;
7. each adjacent-pair value
   `abs(delta(support_retargeted_hips_y)-delta(raw_selected_hips_y)) <= 0.05`;
8. support-retargeted, rendered, and IK-adjusted Hips Y agree within `1e-6`.

When the exact scene/route is `mixed-multilevel/full-course`, it additionally
calls `check_mixed_multilevel(rows)`. That checker uses the published fixed
geometry (`ascent_end=3.20`, `elevated_end=6.20`, block intervals
`[6.20,6.80]`, `[6.80,7.40]`, `[7.40,8.00]`, and a 10-degree return ramp from
`8.00` to `9.8148...`) and requires, in order:

1. runtime XZ motion enters within `0.05 m` of both ends of the complete
   3.00 m elevated segment while `runtime_support_root_height` is within
   `0.02 m` of `0.32 m`;
2. at least 50 consecutive 25 Hz rows on that elevated segment retain
   `matching_enabled=1`, valid source progress, and support-alignment error at
   most `0.02 m`;
3. nonempty interior samples from all three 0.60 m block tops, ordered by
   `simulation_z`, have median `runtime_support_root_height` values within
   `0.02 m` of `0.40`, `0.28`, and `0.32 m`, thereby proving the exact `+0.08`,
   `-0.12`, `+0.04` changes rather than a single elevated hold;
4. return-ramp samples span from runtime support height at least `0.27 m` to at
   most `0.05 m`, have no upward step over `0.03 m`, retain matching, and are
   followed by at least 25 base rows whose runtime support height is within
   `0.02 m` of zero.

Reject reordered/missing plateaus, a shortened elevated span, fewer than 50
elevated matching rows, a flat or rising return ramp, failure to reach base, or
non-finite values. Return `elevated_span_m`, `elevated_matching_frames`, the
three measured plateau medians, `return_ramp_drop_m`, and `base_frames`.

Return `frames`, `transitions`, `landing_frames`, `maximum_support_error`, and
`maximum_non_source_hips_step`, plus the mixed report fields for the exact mixed
route.

Extend `compare_control` to require equal length and per-row exact equality of
`scene_id`, `mode`, `route`, `scene_generation`, `scene_frame`,
`route_waypoint`, `route_complete`, and represented `commanded_speed`. Require
treatment weight `4`, control weight `0`, then retain strictly lower treatment
mean terrain error on active frames and run `check_substride` on both.

- [ ] **Step 5: Implement Gate D and CLI dispatch**

`check_gate_d(rows)` calls `check_rows` and requires:

- one route-mode scene/route, matching/support enabled, IK disabled;
- current accepted `walkability_class` is never `0`;
- a blocked row and a blocked row with `applied_speed <= 1e-4`;
- first stopped blocked row and later blocked rows have
  `blocked_distance >= 0.02 - 1e-4`;
- at least 25 consecutive stopped rows while commanded or at route completion;
- `source_root_height+support_height` never rises more than `0.02 m` above its
  maximum in the 20 pre-block rows;
- stopped tail passes finite, transition-cost, and sub-stride checks.

Return `frames`, `blocked_frames`, `stopped_frames`, `minimum_clearance`, and
`maximum_blocked_support_rise`. Add mutually exclusive `--gate-c`/`--gate-d`;
either may combine with `--compare-control`. Print `VALID gate-c` or
`VALID gate-d` followed by sorted report fields.

- [ ] **Step 6: Run checker GREEN**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
```

Expected: all inherited Gate A and new checker tests pass.

- [ ] **Step 7: Run Gate C treatment/control on six certified routes**

Build once, then run these exact pairs for `800` fixed updates each:

```bash
mkdir -p /tmp/g1-multiscene-runtime
for item in \
  stairs-shallow:ascent-landing-descent \
  stairs-standard:ascent-landing-descent \
  stairs-unseen-variable:ascent-landing-descent \
  ramp-05-up-down:up-landing-down \
  ramp-10-up-down:up-landing-down \
  mixed-multilevel:full-course
do
  scene=${item%%:*}; route=${item#*:}; stem=${scene}__${route}
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=0 \
    MM_LOG="/tmp/g1-multiscene-runtime/gate-c-${stem}-w0.csv" \
    /tmp/controller_g1_multiscene
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=4 \
    MM_LOG="/tmp/g1-multiscene-runtime/gate-c-${stem}-w4.csv" \
    /tmp/controller_g1_multiscene
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "/tmp/g1-multiscene-runtime/gate-c-${stem}-w4.csv" --gate-c \
    --compare-control "/tmp/g1-multiscene-runtime/gate-c-${stem}-w0.csv"
done
```

Expected: twelve runs exit `0`; six checker calls print `VALID gate-c`; each has
at least 50 landing frames, support error `<=0.02`, non-source Hips step
`<=0.05`, no sub-stride loop, and lower weight-four terrain error. The mixed
summary additionally reports the full elevated span, all three ordered
plateaus, a descending return ramp, and at least 25 base rows.

- [ ] **Step 8: Run Gate D on both blocked lanes**

```bash
for item in \
  blocked-course:wall-safe-stop \
  blocked-course:ramp-safe-stop
do
  scene=${item%%:*}; route=${item#*:}; stem=${scene}__${route}
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=600 MM_TERRAIN_WEIGHT=4 \
    MM_LOG="/tmp/g1-multiscene-runtime/gate-d-${stem}.csv" \
    /tmp/controller_g1_multiscene
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "/tmp/g1-multiscene-runtime/gate-d-${stem}.csv" --gate-d
done
```

Expected: two runs/checkers exit `0` and print `VALID gate-d`; no footprint
class is `0`, each stops with at least `0.02 m` clearance for 25 frames, and
blocked-world support rises no more than `0.02 m`.

Do not run `ramp-15-stress` through `check_gate_d`: its G1WM cells are class
`2`, so a blocked-cell pulse is neither required nor correct. Gate E owns the
branch-aware clearance run for `ramp-15-stress:up-landing-down`.

- [ ] **Step 9: Commit deterministic Gate C/D acceptance**

```bash
git add resources/check_g1_runtime_log.py tests/python/test_runtime_log.py \
  controller.cpp
git commit -m "test: enforce G1 multilevel and safe-stop gates"
```

### Task 11: Gate F Switching, Malformed-Candidate Preservation, and Cleanup

**Files:**
- Create: `cleanup_runtime.h`
- Create: `tests/cpp/test_cleanup_runtime.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`
- Modify: `controller.cpp` final cleanup/report and scene-cycle counters
- Generated, never committed: `/tmp/g1-multiscene-runtime/gate-f-*.{csv,json}`

**Interfaces:**
- Produces `check_gate_f(rows,expected_scene_ids) -> dict` and
  `check_failed_switch(rows) -> dict`.
- Adds checker CLI `--gate-f --expected-scenes
  grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,
  stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,
  ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,
  mixed-multilevel,blocked-course` and independent
  `--expect-switch-failure`.
- `MM_CLEANUP_LOG=<path>` is optional. If set, normal finalization atomically
  writes exact JSON fields `exit_code`, `live_model_count`, `log_closed`,
  `model_load_count`, `model_unload_count`, `motion_pack_load_count`, and
  `window_closed` after log close, final model unload, and `CloseWindow`.
- Scene-cycle mode runs two complete ordered catalog passes. A malformed
  candidate is tested from a `/tmp` overlay and never mutates published files.

- [ ] **Step 1: Write scene-cycle/failure checker and cleanup-writer tests**

Add to `tests/python/test_runtime_log.py`:

```python
def scene_cycle_rows(ids, dwell=2, cycles=2):
    rows = []
    frame = 0
    for generation, scene_id in enumerate(ids * cycles):
        for scene_frame in range(dwell):
            rows.append(runtime_row(
                frame, scene_id=scene_id, mode="scene-cycle", route="",
                scene_generation=generation, scene_frame=scene_frame,
                scene_reset_count=generation + 1,
                model_load_count=generation + 1,
                model_unload_count=generation, live_model_count=1,
                route_waypoint=0, commanded_speed=0, applied_speed=0))
            frame += 1
    return rows

def test_gate_f_accepts_two_ordered_cycles_and_one_motion_load(self):
    ids = ["one", "two", "three"]
    report = check_gate_f(scene_cycle_rows(ids), ids)
    self.assertEqual(report["complete_cycles"], 2)
    self.assertEqual(report["motion_pack_loads"], 1)

def test_failed_switch_requires_preserved_identity_generation_and_live_model(self):
    rows = scene_cycle_rows(["one", "two"], dwell=3, cycles=1)
    rows[4]["scene_switch_failed"] = "1"
    self.assertEqual(check_failed_switch(rows)["switch_failures"], 1)
    rows[4]["scene_generation"] = "9"
    with self.assertRaisesRegex(ValueError, "preserve generation"):
        check_failed_switch(rows)
```

Create `tests/cpp/test_cleanup_runtime.cpp`:

```cpp
#include "cleanup_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(bool v,const char* m){if(!v){std::fprintf(stderr,"cleanup runtime test failed: %s\n",m);std::exit(1);}}
int main(){char error[256]={};cleanup_report report;report.exit_code=0;report.motion_pack_load_count=1;report.model_load_count=28;report.model_unload_count=28;report.log_closed=true;report.window_closed=true;check(cleanup_report_write(NULL,report,error,sizeof(error)),error);check(cleanup_report_write("/tmp/test_g1_cleanup.json",report,error,sizeof(error)),error);FILE* f=std::fopen("/tmp/test_g1_cleanup.json","rb");check(f!=NULL,"open report");char text[512]={};const size_t n=std::fread(text,1,sizeof(text)-1,f);check(std::fclose(f)==0&&n>0,"read report");check(std::strcmp(text,"{\"exit_code\":0,\"live_model_count\":0,\"log_closed\":true,\"model_load_count\":28,\"model_unload_count\":28,\"motion_pack_load_count\":1,\"window_closed\":true}\n")==0,"exact report");report.model_unload_count=27;check(!cleanup_report_write("/tmp/test_g1_cleanup_bad.json",report,error,sizeof(error)),"live model rejected");return 0;}
```

- [ ] **Step 2: Run Gate F/cleanup RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log.RuntimeLogTests.test_gate_f_accepts_two_ordered_cycles_and_one_motion_load \
  tests.python.test_runtime_log.RuntimeLogTests.test_failed_switch_requires_preserved_identity_generation_and_live_model -v
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_cleanup_runtime.cpp -o /tmp/test_cleanup_runtime
```

Expected: Python reports missing Gate F helpers; C++ reports missing header.

- [ ] **Step 3: Implement Gate F and malformed-switch checking**

`check_gate_f` first calls reset-aware `check_rows`, then requires:

- nonempty unique expected ID list and mode `scene-cycle` throughout;
- contiguous generation segments start at `0`, each starts at `scene_frame=0`,
  has the same positive dwell length, and contains one constant scene ID;
- segment IDs equal `expected_scene_ids` repeated at least twice with no skip,
  duplicate, or reorder; report exactly two complete cycles for the canonical
  700-row run;
- first row of generation `g` has `scene_reset_count=g+1`,
  `model_load_count=g+1`, `model_unload_count=g`, `live_model_count=1`,
  `route_waypoint=0`, `blocked=0`, and `airborne_frames<=1`;
- every row has motion-pack load count `1`, fixed `0.04`, planar Y deltas,
  finite support/pose/cost values, and no switch-failure pulse.

Return `frames`, `generations`, `complete_cycles`, `motion_pack_loads`,
`model_loads`, and `model_unloads_before_final_cleanup`.

`check_failed_switch` requires at least one failure pulse. At each pulse, active
scene ID and generation equal the preceding row, scene frame continues, reset
count does not change, motion load remains one, and live models remain one.
The next ten available rows must remain finite and continue the same active
scene unless a later successful generation begins. Return `switch_failures`,
`preserved_scene`, and `preserved_generation`.

Add CLI parsing. `--gate-f` requires a comma-separated `--expected-scenes`
whose 14 IDs exactly match the locked catalog; `--expect-switch-failure` may not
combine with Gate A/C/D/F flags. Print `VALID gate-f` or
`VALID switch-failure` with sorted report fields.

- [ ] **Step 4: Implement atomic cleanup reporting and one normal exit path**

Create `cleanup_runtime.h`:

```cpp
#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
struct cleanup_report{int exit_code=0,motion_pack_load_count=0,model_load_count=0,model_unload_count=0;bool log_closed=false,window_closed=false;};
static inline bool cleanup_error(char* error,int capacity,const char* message)
{
 if(error!=NULL&&capacity>0){
  std::snprintf(error,static_cast<std::size_t>(capacity),"%s",message);
 }
 return false;
}
static inline bool cleanup_report_write(const char* path,const cleanup_report& report,char* error,int capacity)
{
 if(path==NULL)return true;
 if(path[0]=='\0'||report.motion_pack_load_count!=1||report.model_load_count<0||
    report.model_unload_count<0||
    report.model_load_count-report.model_unload_count!=0||
    !report.log_closed||!report.window_closed){
  return cleanup_error(error,capacity,
      "cleanup report has invalid path or incomplete cleanup");
 }
 const std::string temporary=std::string(path)+".tmp";
 FILE* file=std::fopen(temporary.c_str(),"wb");
 if(file==NULL){
  const std::string message=temporary+": cannot open cleanup report ("+
      std::strerror(errno)+")";
  return cleanup_error(error,capacity,message.c_str());
 }
 const int written=std::fprintf(file,
     "{\"exit_code\":%d,\"live_model_count\":0,\"log_closed\":true,"
     "\"model_load_count\":%d,\"model_unload_count\":%d,"
     "\"motion_pack_load_count\":1,\"window_closed\":true}\n",
     report.exit_code,report.model_load_count,report.model_unload_count);
 bool failed=written<0;
 if(std::fflush(file)!=0)failed=true;
 if(std::fclose(file)!=0)failed=true;
 if(!failed&&std::rename(temporary.c_str(),path)!=0)failed=true;
 if(failed){
  const int saved_errno=errno;
  std::remove(temporary.c_str());
  const std::string message=std::string(path)+
      ": cannot finish cleanup report ("+std::strerror(saved_errno)+")";
  return cleanup_error(error,capacity,message.c_str());
 }
 return true;
}
```

Keep all early pre-window validation returns unchanged. Once `InitWindow`
succeeds, route every error through the loop's `controller_exit_requested` and
then this one tail, in this exact order:

```cpp
deterministic_log.close();const bool log_closed=true;
model_unloader(terrain_model);
CloseWindow();const bool window_closed=true;
cleanup_report cleanup;cleanup.exit_code=controller_exit_code;cleanup.motion_pack_load_count=motion_pack_load_count;cleanup.model_load_count=model_load_count;cleanup.model_unload_count=model_unload_count;cleanup.log_closed=log_closed;cleanup.window_closed=window_closed;
if(!cleanup_report_write(std::getenv("MM_CLEANUP_LOG"),cleanup,artifact_error,sizeof(artifact_error))){std::fprintf(stderr,"G1 cleanup report error: %s\n",artifact_error);if(controller_exit_code==0)controller_exit_code=2;}
return controller_exit_code;
```

Never call `_Exit`, `exit`, or return from inside the post-window update loop.

- [ ] **Step 5: Run cleanup/checker GREEN and the full native test set**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_cleanup_runtime.cpp -o /tmp/test_cleanup_runtime
/tmp/test_cleanup_runtime
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_scene_switch.cpp -o /tmp/test_scene_switch
/tmp/test_scene_switch
```

Expected: exact cleanup JSON test, all logger tests, and repeated fake-model
transaction tests pass with no warning.

- [ ] **Step 6: Run two complete ordered scene cycles**

```bash
SCENES=grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,mixed-multilevel,blocked-course
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=700 \
  MM_TERRAIN_WEIGHT=4 \
  MM_LOG=/tmp/g1-multiscene-runtime/gate-f-scene-cycle.csv \
  MM_CLEANUP_LOG=/tmp/g1-multiscene-runtime/gate-f-scene-cycle-cleanup.json \
  /tmp/controller_g1_multiscene
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-multiscene-runtime/gate-f-scene-cycle.csv \
  --gate-f --expected-scenes "$SCENES"
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
p='/tmp/g1-multiscene-runtime/gate-f-scene-cycle-cleanup.json'
v=json.load(open(p,encoding='utf-8'))
assert v == {'exit_code':0,'live_model_count':0,'log_closed':True,
 'model_load_count':28,'model_unload_count':28,
 'motion_pack_load_count':1,'window_closed':True}, v
print('VALID cleanup', p)
PY
```

Expected: controller exits `0`; checker reports `28` generations, two complete
cycles, one motion load, and one live model per row; cleanup prints `VALID` and
proves all 28 loaded models were unloaded.

- [ ] **Step 7: Exercise a malformed candidate through a read-only overlay**

```bash
SCENES=grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,mixed-multilevel,blocked-course
BAD_ROOT=$(mktemp -d /tmp/g1-malformed-pack.XXXXXX)
for file in database.bin terrain_features.bin terrain_support.bin manifest.json validation.json
do
  ln -s "$(realpath resources/g1_terrain/$file)" "$BAD_ROOT/$file"
done
mkdir "$BAD_ROOT/scenes"
ln -s "$(realpath resources/g1_terrain/scenes/index.json)" "$BAD_ROOT/scenes/index.json"
for scene in ${SCENES//,/ }
do
  if test "$scene" = stairs-shallow; then
    cp -a "resources/g1_terrain/scenes/$scene" "$BAD_ROOT/scenes/$scene"
    truncate -s 1 "$BAD_ROOT/scenes/$scene/terrain.obj"
  else
    ln -s "$(realpath resources/g1_terrain/scenes/$scene)" "$BAD_ROOT/scenes/$scene"
  fi
done
DISPLAY=:1 G1_TERRAIN_DIR="$BAD_ROOT" \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=350 \
  MM_TERRAIN_WEIGHT=4 \
  MM_LOG=/tmp/g1-multiscene-runtime/gate-f-malformed.csv \
  MM_CLEANUP_LOG=/tmp/g1-multiscene-runtime/gate-f-malformed-cleanup.json \
  /tmp/controller_g1_multiscene
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-multiscene-runtime/gate-f-malformed.csv \
  --expect-switch-failure
```

Expected: controller still exits `0`; at least three attempted switches to the
corrupt `stairs-shallow` fail its OBJ SHA check; CSV remains on
`grail-curb-high` with unchanged generation/reset count, one live model, finite
updates, and `VALID switch-failure`. The published pack remains untouched.

- [ ] **Step 8: Commit Gate F and normal cleanup**

```bash
git add cleanup_runtime.h tests/cpp/test_cleanup_runtime.cpp \
  resources/check_g1_runtime_log.py tests/python/test_runtime_log.py \
  controller.cpp
git commit -m "test: cycle G1 scenes through normal cleanup"
```

### Task 12: Whole-Program Regression, Preservation Audit, and Staged Handoff

**Files:**
- Verify only: all source/test files named by this plan
- Read only: `resources/g1_terrain/`
- Preserve: legacy/user resources and all unrelated dirty/untracked files
- Generated, never committed: verification evidence under
  `/tmp/g1-multiscene-runtime/`

**Interfaces:**
- Changes no code and creates no commit.
- Revalidates artifact Gate B plus runtime Gates C, D, and F with IK disabled.
- Leaves no persistent controller process. Gate F already exercises Raylib on
  `DISPLAY=:1` and normal cleanup; the downstream IK/clearance plan owns the
  one final persistent launch after its last-only optional mesh decision.

- [ ] **Step 1: Run every Python test and independent pack validation**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
```

Expected: every Python test reports `OK`; validator prints exactly
`VALID g1-terrain-artifacts/v2 frames=459682 clips=1770 bones=31 terrain_dims=4 support_dims=3 scenes=14 source_rows=0`.

- [ ] **Step 2: Run every owned C++ test in strict and fast-math builds**

```bash
for test in \
  test_terrain_runtime test_scene_runtime test_support_runtime \
  test_g1_controller_state test_scene_switch test_support_matching \
  test_route_runtime test_cleanup_runtime
do
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "tests/cpp/${test}.cpp" -o "/tmp/${test}_strict"
  "/tmp/${test}_strict"
  g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
    "tests/cpp/${test}.cpp" -o "/tmp/${test}_release"
  "/tmp/${test}_release"
done
/tmp/test_scene_runtime_strict --real resources/g1_terrain
```

Expected: 16 ordinary test executions plus the complete 14-scene semantic
probe exit `0`; strict compiles produce no warning.

- [ ] **Step 3: Run sanitizer configurations for every ownership boundary**

```bash
for test in \
  test_terrain_runtime test_scene_runtime test_support_runtime \
  test_g1_controller_state test_scene_switch test_route_runtime \
  test_cleanup_runtime
do
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer -I. "tests/cpp/${test}.cpp" \
    -o "/tmp/${test}_san"
  ASAN_OPTIONS=detect_leaks=1 "/tmp/${test}_san"
done
ASAN_OPTIONS=detect_leaks=1 /tmp/test_scene_runtime_san \
  --real resources/g1_terrain
```

Expected: all sanitizer runs exit `0` with no ASan/UBSan/leak report.

- [ ] **Step 4: Build the controller in strict and approved release modes**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_multiscene_strict \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_multiscene \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected: both builds exit `0`; strict emits no warning. Do not add or build an
optional G1 mesh, IK solver, or learned matcher.

- [ ] **Step 5: Recheck all retained acceptance evidence**

```bash
for item in \
  stairs-shallow:ascent-landing-descent \
  stairs-standard:ascent-landing-descent \
  stairs-unseen-variable:ascent-landing-descent \
  ramp-05-up-down:up-landing-down \
  ramp-10-up-down:up-landing-down \
  mixed-multilevel:full-course
do
  scene=${item%%:*}; route=${item#*:}; stem=${scene}__${route}
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "/tmp/g1-multiscene-runtime/gate-c-${stem}-w4.csv" --gate-c \
    --compare-control "/tmp/g1-multiscene-runtime/gate-c-${stem}-w0.csv"
done
for item in \
  blocked-course:wall-safe-stop \
  blocked-course:ramp-safe-stop
do
  scene=${item%%:*}; route=${item#*:}; stem=${scene}__${route}
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py \
    "/tmp/g1-multiscene-runtime/gate-d-${stem}.csv" --gate-d
done
SCENES=grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,mixed-multilevel,blocked-course
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-multiscene-runtime/gate-f-scene-cycle.csv \
  --gate-f --expected-scenes "$SCENES"
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-multiscene-runtime/gate-f-malformed.csv \
  --expect-switch-failure
```

Expected: six `VALID gate-c`, two `VALID gate-d`, one `VALID gate-f`, and
one `VALID switch-failure` reports. Re-run the producing task, not the checker
alone, if any expected CSV is absent.

- [ ] **Step 6: Audit forbidden coupling and fixed-rate invariants**

```bash
rg -n 'SetTargetFPS\(25\)|1\.0f / 25\.0f|feature_weight_terrain' controller.cpp
! rg -n 'MM_IK|const bool ik_enabled = true|static constexpr bool ik_enabled = true' \
  controller.cpp support_runtime.h g1_controller_state.h
! rg -n 'database_(search|build_matching_features)\([^;]*(support|terrain_support)' \
  controller.cpp
! rg -n '(^|[^A-Za-z0-9_])(adjust_character_position(_by_velocity)?|clamp_character_position)\(' \
  controller.cpp
rg -n 'horizontal_adjust_character_position|horizontal_clamp_character_position' \
  controller.cpp
```

Expected: fixed 25 Hz and effective terrain weight sites are present; IK and
support-to-matcher coupling searches return no match; only horizontal root
adjust/clamp helpers are invoked.

- [ ] **Step 7: Prove user resources and worktree state are preserved**

```bash
sha256sum -c /tmp/g1_runtime_legacy_before.sha256
git diff --check
git status --short > /tmp/g1_runtime_status_after.txt
diff -u /tmp/g1_runtime_status_before.txt /tmp/g1_runtime_status_after.txt
find resources -maxdepth 1 \
  \( -name '.g1_terrain.staging-*' -o -name '.g1_terrain.previous-*' \) \
  -print
```

Expected: both legacy hashes print `OK`, diff check is empty, status matches the
initial user-owned dirty state, and `find` prints nothing. Generated pack/CSV/
JSON/controller bytes remain ignored or under `/tmp`, never staged.

- [ ] **Step 8: Confirm the display smoke closed normally and reserve the final launch**

Only after Steps 1--7 pass, re-read the Gate F cleanup evidence and confirm no
runtime-plan controller remains:

```bash
test -s /tmp/g1-multiscene-runtime/gate-f-scene-cycle-cleanup.json
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
path = "/tmp/g1-multiscene-runtime/gate-f-scene-cycle-cleanup.json"
with open(path, encoding="utf-8") as stream:
    cleanup = json.load(stream)
assert cleanup["exit_code"] == 0, cleanup
assert cleanup["live_model_count"] == 0, cleanup
assert cleanup["log_closed"] is True, cleanup
assert cleanup["window_closed"] is True, cleanup
assert cleanup["model_load_count"] == cleanup["model_unload_count"], cleanup
print("VALID staged runtime handoff")
PY
! pgrep -f '/tmp/controller_g1_multiscene([[:space:]]|$)'
```

Expected: the cleanup assertion prints `VALID staged runtime handoff` and no
runtime-plan controller remains. Do not start another live window here; the
final IK/mesh task launches the mixed-multilevel controller exactly once.

- [ ] **Step 9: Record the implementation handoff**

Report: Python test count; strict/release/sanitizer C++ matrix; validator line;
six Gate C and two Gate D summaries; Gate F generation/model/cleanup counts;
malformed candidate preservation; legacy hashes; final status diff; and live
handoff reservation. Explicitly state `25 Hz`, `31D`, `IK disabled`, optional
G1 mesh deferred to the downstream last-only task, motion pack loaded once,
and no generated/user resource committed.
