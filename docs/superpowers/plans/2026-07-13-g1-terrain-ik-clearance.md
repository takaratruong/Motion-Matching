# G1 Terrain IK and Clearance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reversible, bounded G1 terrain IK that locks planted feet to the authoritative scene surface, aligns feet to its normals, clears swing feet over certified terrain, hands unreachable targets to the existing safe-stop path, and passes Gate E without changing matching or the accepted support-retargeted root.

**Architecture:** The completed scene-artifact and multiscene-support-runtime plans remain authoritative for G1HF/v2 sampling, G1SP/G1WM data, scene routes, support retargeting, traversability, logging, reset, and controlled cleanup. This plan adds renderer-independent G1 leg, IK, and clearance modules downstream of the support-retargeted IK-off pose; it copies that pose into dedicated IK output buffers, changes only named leg rotations, and exposes diagnostics and safe-stop requests without writing matcher, trajectory, planar-root, or support-frame state. Paired deterministic route logs prove byte-identical matching/support columns and bounded physical corrections.

**Tech Stack:** C++17 header-only math and existing `array.h`, `vec.h`, `quat.h`, `spring.h`, `database.h`, `g1_skeleton.h`, `terrain_runtime.h`, `scene_runtime.h`, `support_runtime.h`, and `g1_controller_state.h`; Raylib/Raygui from `/home/ubuntu/apps`; Python 3 standard-library CSV checks; `DISPLAY=:1` deterministic route runs.

## Global Constraints

- Do not start Task 1 until every task and final gate in `docs/superpowers/plans/2026-07-13-g1-scene-artifacts.md` and `docs/superpowers/plans/2026-07-13-g1-multiscene-support-runtime.md` passes with IK disabled. Preserve their accepted IK-off CSVs under `/tmp/g1-multiscene-runtime/` as the Gate C/D/F baselines.
- Consume, do not duplicate or replace, the sibling plans' checked
  `heightfield_sample_v2`, `heightfield_normal`, `terrain_support_set`,
  `walkability_grid`, `scene_pack`, `scene_route`, `support_frame_state`,
  `support_pose_apply`, controller reset, deterministic route, logger/checker,
  traversability, scene-switch, and controlled-cleanup implementations. The
  unqualified `heightfield_sample` is frozen G1HF/v1 migration code and is
  forbidden on active G1HF/v2 scenes in this plan.
- Consume only published G1HF/v2 scene surfaces, G1SP/v1 support rows, and G1WM/v1 walkability. Do not rebuild or modify generated motion/scene artifacts in this plan.
- Keep Daniel Holden's matcher, 31-dimensional `27 + 4` feature contract, feature normalization, search, selected frame/range, transition decision, continuation/incumbent/selected/terrain costs, full-pose inertialization, and fixed `25 Hz` timing authoritative.
- Treat `state.adjusted_bone_positions` and `state.adjusted_bone_rotations` from the completed support runtime as immutable per-frame IK inputs. Preserve every position byte, especially support-retargeted `G1_Simulation.y`; IK writes only dedicated `state.ik_bone_rotations` and derived IK FK buffers.
- IK is downstream and reversible. `MM_IK=0` is the default; it must reproduce the accepted support-retargeted baseline exactly. `MM_IK=1` may not change selected frames, ranges, transitions, query values/points, costs, planar simulation state, support height/velocity/source, adjustment, or clamping on certified routes.
- Configure each G1 leg by explicit named bones and measured local geometry. Never infer a leg by walking parents from a toe, never use a LAFAN bone name, and never apply Holden's inherited hard-coded `+X` toe-end solve.
- Sample planted targets, surface normals, swept swing clearance, and all physical diagnostics through the same active `scene_pack.terrain` G1HF/v2 surface used for matching and rendering.
- Use exactly `dt = 1.0f / 25.0f` for contact target inertialization, lock state, swing history, deterministic tests, and Gate E runs. Reject any non-finite IK input/result through the sibling runtime's controlled diagnostic/normal-cleanup path.
- Clamp every raw target to the analytic leg's inner/outer reachable shell with a `0.015 m` buffer. Clamp the delta of every IK-modified local joint quaternion to at most `0.35 radians` from the support-retargeted IK-off input for that frame.
- A target that needs more than the reachable shell, more than `0.08 m` of swing-only lift, or a correction above the bounded solve requests the existing traversability safe stop. It may not stretch a limb, lift `G1_Simulation`/Hips, advance through a blocked surface, or silently substitute a different target.
- Gate E thresholds are exact: planted toe/foot penetration at most `0.005 m`; every logged Hips/knee/ankle/toe point and thigh/shin capsule penetration at most `0.01 m`; correction at most `0.35 radians`; planted horizontal drift strictly lower with IK on.
- Preserve the dirty worktree. Never stage or commit generated packs, CSVs, binaries, videos, converted mesh outputs, or unrelated user files. Every commit command below names only files owned by its task.
- Rigid-link mesh rendering is optional and appears only in the final task. It cannot begin until every correctness, IK, invariance, sanitizer, release, and Gate E check passes; failure of its ease gate records a small deferral note and does not change correctness status.

---

## File Map and Ownership

- `ik.h`: renderer-independent finite/reach-shell checks, robust two-bone solve, and per-joint quaternion-delta clamping extracted from the dormant controller IK block.
- `g1_ik.h`: explicit left/right G1 configuration, planted-foot state/targets, surface-normal orientation, named leg solve, per-frame IK state/result, and safe-stop/error outcomes.
- `g1_clearance.h`: exact-surface point, sole, swept-foot, thigh-capsule, and shin-capsule clearance sampling plus frame diagnostics.
- `g1_ik_runtime.h`: one downstream observe/apply transaction that composes locks, swing lift, position solve, orientation, invariance, and safe-stop outcomes for both legs.
- `g1_controller_state.h`: replace dormant generic contact arrays with resettable `G1IkState`; add distinct IK local/global pose buffers and diagnostics.
- `controller.cpp`: parse/toggle IK, run observation in both modes, apply IK only after support FK, forward safe-stop/error outcomes, fill diagnostics, and render either the IK or exact IK-off pose.
- `motion_match_log.h`: append an IK-only suffix to the completed sibling logger; do not rename or reorder any existing column.
- `resources/check_g1_runtime_log.py`: compose existing Gate C/D/F checks with exact-string IK invariance and Gate E bounds.
- `tests/cpp/test_g1_ik.cpp`: explicit geometry, v2 local-basis provenance, locks, normals, named solve, orientation, correction, reset, reversibility, and error-path tests.
- `tests/cpp/test_g1_clearance.cpp`: point/sole/capsule/sweep diagnostics and safe-stop threshold tests on synthetic G1HF/v2 surfaces.
- `tests/cpp/test_g1_controller_state.cpp`: scene reset/swap coverage for all IK state and pose buffers.
- `tests/python/test_runtime_log.py`: synthetic IK schema, invariance, drift, penetration, safe-stop, route-matrix, and controlled-error checker regressions.
- Optional only: `resources/convert_g1_visual_meshes.py`, `tests/python/test_g1_visual_meshes.py`, `g1_visual_mesh.h`, `.gitignore`, and either `docs/superpowers/evidence/2026-07-13-g1-rigid-mesh-deferral.md` or passing render integration in `controller.cpp`.

## Consumed Cross-Plan Contracts

Before implementation, compare the completed sibling code with this block. The producer names and types win; if a sibling plan changed one while being implemented, update this document consistently before Task 1 rather than adding an adapter or duplicate implementation.

```cpp
// terrain_runtime.h and scene_runtime.h, produced by sibling plans
struct heightfield {
    int nx, nz;
    float origin_x, origin_z, cell_size, exterior_height;
    array1d<float> heights;
    uint32_t version; // appended to preserve legacy v1 member offsets
};
float heightfield_sample(const heightfield&, float x, float z); // v1 only
float heightfield_sample_v2(const heightfield&, float x, float z);
float heightfield_sample_versioned(const heightfield&, float x, float z);
vec3 heightfield_normal(const heightfield&, float x, float z);

struct scene_pack {
    scene_metadata metadata;
    heightfield terrain;
    walkability_grid walkability;
    std::string scene_path, terrain_path, mesh_path, walkability_path;
};

// support_runtime.h and g1_controller_state.h, produced by sibling plans
void support_pose_apply(
    slice1d<vec3> output,
    const slice1d<vec3> inertialized,
    float support_height);
bool g1_controller_state_reset(
    g1_controller_state&, const database&, const terrain_support_set&,
    const scene_pack&, char*, int);
```

The sibling logger's entire pre-IK schema and column order are immutable. Its final locked suffix, before this plan appends IK fields, is:

```text
source_name,source_terrain,source_index,continuation_cost,
source_root_height,source_left_toe_height,source_right_toe_height,
runtime_support_root_height,runtime_support_left_toe_height,
runtime_support_right_toe_height,support_root_delta,support_left_toe_delta,
support_right_toe_delta,support_height,support_velocity,support_source,
airborne_frames,left_contact,right_contact,support_retargeted_hips_y,
ik_adjusted_hips_y,simulation_x,simulation_z,walkability_class,blocked,
blocked_reason,blocked_distance,blocked_point_x,blocked_point_z,
commanded_speed,applied_speed,route_waypoint,route_complete,
route_target_height,scene_generation,scene_frame,scene_reset_count,
scene_switch_failed,motion_pack_load_count,model_load_count,
model_unload_count,live_model_count
```

The sibling checker entry points are `check_rows`, `check_gate_c`, `check_gate_d`, `check_gate_f`, `check_failed_switch`, `check_substride`, and `compare_control`; its CLI flags are `--gate-c`, `--gate-d`, `--gate-f`, `--expect-switch-failure`, and `--expected-scenes`. Route mode is `MM_TEST_MODE=route` with exact `MM_TEST_ROUTE`; shared options remain `MM_TEST_FRAMES`, `MM_LOG`, `MM_TERRAIN_WEIGHT`, and `MM_TERRAIN_SCENE`. The IK checker compares its named matching, simulation, and support-root invariants as raw CSV strings rather than parsing and reformatting floats. The scene-artifact schemas remain `g1-terrain-artifacts/v2`, `g1-terrain-scene-index/v1`, `g1-terrain-scene/v1`, and `g1-terrain-surface/v1`; the motion and scene-pack schemas remain unchanged. The optional final visual branch defines its independent render-only `G1VM/v1` manifest and does not alter any matcher, terrain, or runtime-log schema.

---

### Task 1: Lock Explicit G1 Leg and Local Geometry Contracts

**Files:**
- Create: `g1_ik.h`
- Create: `tests/cpp/test_g1_ik.cpp`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`

**Interfaces:**
- Consumes: `G1Bone`, the exact 31-bone parent array, `database`, `vec3`, and finite helpers from `terrain_runtime.h`.
- Produces: `G1LegConfig g1_left_leg_config()` and `G1LegConfig g1_right_leg_config()`.
- Produces: `bool g1_leg_configs_validate(const database&, char*, int)`; a bad skeleton/config is a startup error before Raylib.
- Locks the authoritative `g1-terrain-artifacts/v2` MuJoCo-to-Holden local basis `(x,y,z) -> (x,z,-y)`, including four sole probes and thigh/shin capsules. The v2 builder rotates world positions, conjugates world rotations, and then re-expresses child-local offsets; the legacy/root-only `resources/database.bin` layout is incompatible. Runtime validation does not load or parse XML.
- Scans all database rows for finite, fixed mapped `Left/RightKnee`, `Left/RightAnkle`, and `Left/RightToe` child offsets within `1e-6 m`. This provenance gate prevents pairing the mapped IK geometry with a legacy local-basis database that has the same 31-bone shape.
- Wires the validator immediately after `g1_skeleton_validate` and before feature construction or `InitWindow`; a source-order regression locks that startup gate.

- [ ] **Step 1: Write the failing named-geometry test**

Create `tests/cpp/test_g1_ik.cpp`:

```cpp
#include "g1_ik.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 IK test failed: %s\n", message);
        std::exit(1);
    }
}

static void make_g1_database(database& db)
{
    static const int parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,
        15,16,17,18,19,20,21,22,16,24,25,26,27,28,29
    };
    db.bone_positions.resize(1, G1_BoneCount);
    db.bone_rotations.resize(1, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.bone_positions.set(vec3());
    db.bone_rotations.set(quat());
    for (int i = 0; i < G1_BoneCount; ++i) db.bone_parents(i) = parents[i];
    db.bone_positions(0, G1_LeftKnee) =
        vec3(-0.078273f, -0.17734f, -0.0021489f);
    db.bone_positions(0, G1_LeftAnkle) =
        vec3(0.0f, -0.30001f, +0.000094445f);
    db.bone_positions(0, G1_LeftToe) =
        vec3(0.0f, -0.017558f, 0.0f);
    db.bone_positions(0, G1_RightKnee) =
        vec3(-0.078273f, -0.17734f, +0.0021489f);
    db.bone_positions(0, G1_RightAnkle) =
        vec3(0.0f, -0.30001f, -0.000094445f);
    db.bone_positions(0, G1_RightToe) =
        vec3(0.0f, -0.017558f, 0.0f);
}

static void test_explicit_leg_geometry()
{
    const G1LegConfig left = g1_left_leg_config();
    const G1LegConfig right = g1_right_leg_config();
    check(left.hip == G1_LeftHipYaw && left.knee == G1_LeftKnee,
          "left hip/knee names");
    check(left.ankle == G1_LeftAnkle && left.contact == G1_LeftToe,
          "left ankle/contact names");
    check(right.hip == G1_RightHipYaw && right.knee == G1_RightKnee,
          "right hip/knee names");
    check(right.ankle == G1_RightAnkle && right.contact == G1_RightToe,
          "right ankle/contact names");
    check(left.knee_hinge_axis_local.z == -1.0f &&
          right.knee_hinge_axis_local.z == -1.0f,
          "MuJoCo +Y knee hinge maps to Holden -Z");
    check(left.foot_forward_local.x == 1.0f && left.sole_normal_local.y == 1.0f,
          "foot axes");
    check(left.sole_points_local[0].x == -0.05f &&
          left.sole_points_local[0].y ==
              left.foot_sphere_centers_local[0].y -
              left.foot_sphere_radius_m &&
          left.sole_points_local[3].x == 0.12f,
          "sphere-bottom sole probes");
    check(left.foot_sphere_centers_local[0].y == -0.03f,
          "mapped XML foot-sphere center");
    check(left.foot_sphere_radius_m == 0.02f,
          "XML foot-sphere radius");
    check(left.shin_radius_m == 0.04f && left.thigh_radius_m == 0.05f,
          "XML capsule radii");
    check(left.reach_buffer_m == 0.015f &&
          left.max_swing_lift_m == 0.08f &&
          left.max_correction_radians == 0.35f,
          "IK bounds");

    database db;
    make_g1_database(db);
    char error[256] = {};
    check(g1_leg_configs_validate(db, error, sizeof(error)), error);
    db.bone_parents(G1_LeftKnee) = G1_LeftHipRoll;
    check(!g1_leg_configs_validate(db, error, sizeof(error)),
          "wrong named chain rejected");
    check(std::strstr(error, "LeftKnee") != NULL,
          "wrong named chain diagnostic");
}

int main()
{
    test_explicit_leg_geometry();
    return 0;
}
```

Before compiling RED, extend this focused test with hostile cases for null and
short error buffers; null position/rotation/parent data pointers (restored
before destruction); mismatched nonempty pose rows/columns; out-of-range and
wrong-side named indices; wrong named hip/knee/ankle/contact chains and
unrelated parent corruption; null/malformed names; non-finite/non-positive
geometry; non-unit,
non-orthogonal, or incorrectly mapped local axes; degenerate capsules; and
inconsistent, duplicate, or non-planar sole probes. Exercise both left and
right configurations. The validator checks named leg chains before the full
`g1_skeleton_validate` call so corrupt `LeftHipYaw` and `LeftKnee` links still
receive their named diagnostics. Also create a database-shaped legacy fixture
with the unmapped XML-local knee/ankle/toe offsets and require a `local basis`
diagnostic containing the failing frame and named bone. Put a legacy offset in
row 1 of a two-row otherwise-valid fixture to prove every row is scanned, and
cover a non-finite selected offset so NaN cannot bypass the tolerance checks.

- [ ] **Step 2: Compile to verify RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
```

Expected: compilation fails with `fatal error: g1_ik.h: No such file or directory`.

- [ ] **Step 3: Define the explicit configuration and validation**

Create `g1_ik.h`:

```cpp
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "database.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "g1_skeleton.h"
#include "terrain_runtime.h"

#include <cstdarg>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

struct G1LegConfig
{
    const char* name;
    int hip;
    int knee;
    int ankle;
    int contact;
    vec3 knee_hinge_axis_local;
    vec3 foot_forward_local;
    vec3 sole_normal_local;
    vec3 foot_sphere_centers_local[4];
    vec3 sole_points_local[4];
    float foot_sphere_radius_m;
    vec3 thigh_start_local;
    vec3 thigh_end_local;
    float thigh_radius_m;
    vec3 shin_start_local;
    vec3 shin_end_local;
    float shin_radius_m;
    float reach_buffer_m;
    float planted_clearance_m;
    float swing_clearance_m;
    float max_swing_lift_m;
    float max_correction_radians;
};

static inline bool g1_ik_error(
    char* output, int capacity, const char* format, ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return false;
}

static inline G1LegConfig g1_leg_config(
    const char* name, int hip, int knee, int ankle, int contact)
{
    G1LegConfig config = {};
    config.name = name;
    config.hip = hip;
    config.knee = knee;
    config.ankle = ankle;
    config.contact = contact;
    config.knee_hinge_axis_local = vec3(0.0f, 0.0f, -1.0f);
    config.foot_forward_local = vec3(1.0f, 0.0f, 0.0f);
    config.sole_normal_local = vec3(0.0f, 1.0f, 0.0f);
    config.foot_sphere_radius_m = 0.02f;
    config.foot_sphere_centers_local[0] =
        vec3(-0.05f, -0.03f, -0.025f);
    config.foot_sphere_centers_local[1] =
        vec3(-0.05f, -0.03f, +0.025f);
    config.foot_sphere_centers_local[2] =
        vec3(+0.12f, -0.03f, -0.030f);
    config.foot_sphere_centers_local[3] =
        vec3(+0.12f, -0.03f, +0.030f);
    for (int i = 0; i < 4; ++i) {
        config.sole_points_local[i] =
            config.foot_sphere_centers_local[i] -
            config.sole_normal_local * config.foot_sphere_radius_m;
    }
    config.thigh_start_local = vec3(0.0f, -0.02f, 0.0f);
    config.thigh_end_local = vec3(-0.078f, -0.17f, 0.0f);
    config.thigh_radius_m = 0.05f;
    config.shin_start_local = vec3(0.0f, -0.05f, 0.0f);
    config.shin_end_local = vec3(0.0f, -0.28f, 0.0f);
    config.shin_radius_m = 0.04f;
    config.reach_buffer_m = 0.015f;
    config.planted_clearance_m = 0.005f;
    config.swing_clearance_m = 0.015f;
    config.max_swing_lift_m = 0.08f;
    config.max_correction_radians = 0.35f;
    return config;
}

static inline G1LegConfig g1_left_leg_config()
{
    return g1_leg_config(
        "left", G1_LeftHipYaw, G1_LeftKnee, G1_LeftAnkle, G1_LeftToe);
}

static inline G1LegConfig g1_right_leg_config()
{
    return g1_leg_config(
        "right", G1_RightHipYaw, G1_RightKnee, G1_RightAnkle, G1_RightToe);
}

static inline bool g1_leg_vec3_is_finite(const vec3 value)
{
    return terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline bool g1_leg_vec3_is_exact(
    const vec3 value, const vec3 expected)
{
    return value.x == expected.x &&
           value.y == expected.y &&
           value.z == expected.z;
}

static inline float g1_leg_length_squared(const vec3 value)
{
    return dot(value, value);
}

static inline bool g1_leg_axis_is_unit(const vec3 value)
{
    const float squared = g1_leg_length_squared(value);
    return terrain_float_is_finite(squared) &&
           std::fabs(squared - 1.0f) <= 1.0e-5f;
}

static inline bool g1_leg_database_shape_validate(
    const database& db, char* error, int error_capacity)
{
    if (db.bone_positions.rows <= 0 ||
        db.bone_positions.cols != G1_BoneCount ||
        db.bone_positions.data == NULL) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 IK position pose shape mismatch: expected rows>0 cols=%d",
            G1_BoneCount);
    }
    if (db.bone_rotations.rows != db.bone_positions.rows ||
        db.bone_rotations.rows <= 0 ||
        db.bone_rotations.cols != G1_BoneCount ||
        db.bone_rotations.data == NULL) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 IK rotation pose shape mismatch: expected %dx%d",
            db.bone_positions.rows,
            G1_BoneCount);
    }
    if (db.bone_parents.size != G1_BoneCount ||
        db.bone_parents.data == NULL) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 IK parent shape mismatch: expected %d entries",
            G1_BoneCount);
    }
    return true;
}

static inline bool g1_leg_database_local_basis_validate(
    const database& db, char* error, int error_capacity)
{
    const int bones[] = {
        G1_LeftKnee,
        G1_LeftAnkle,
        G1_LeftToe,
        G1_RightKnee,
        G1_RightAnkle,
        G1_RightToe
    };
    const char* const names[] = {
        "LeftKnee",
        "LeftAnkle",
        "LeftToe",
        "RightKnee",
        "RightAnkle",
        "RightToe"
    };
    const vec3 expected_offsets[] = {
        vec3(-0.078273f, -0.17734f, -0.0021489f),
        vec3(0.0f, -0.30001f, +0.000094445f),
        vec3(0.0f, -0.017558f, 0.0f),
        vec3(-0.078273f, -0.17734f, +0.0021489f),
        vec3(0.0f, -0.30001f, -0.000094445f),
        vec3(0.0f, -0.017558f, 0.0f)
    };
    const float tolerance_m = 1.0e-6f;
    const int count = static_cast<int>(sizeof(bones) / sizeof(bones[0]));
    for (int frame = 0; frame < db.bone_positions.rows; ++frame) {
        for (int index = 0; index < count; ++index) {
            const vec3 actual = db.bone_positions(frame, bones[index]);
            const vec3 expected = expected_offsets[index];
            if (!g1_leg_vec3_is_finite(actual) ||
                std::fabs(actual.x - expected.x) > tolerance_m ||
                std::fabs(actual.y - expected.y) > tolerance_m ||
                std::fabs(actual.z - expected.z) > tolerance_m) {
                return g1_ik_error(
                    error,
                    error_capacity,
                    "G1 IK local basis mismatch at frame %d bone %s",
                    frame,
                    names[index]);
            }
        }
    }
    return true;
}

static inline bool g1_leg_measured_geometry_matches(
    const G1LegConfig& value, const G1LegConfig& expected)
{
    if (!g1_leg_vec3_is_exact(
            value.knee_hinge_axis_local,
            expected.knee_hinge_axis_local) ||
        !g1_leg_vec3_is_exact(
            value.foot_forward_local,
            expected.foot_forward_local) ||
        !g1_leg_vec3_is_exact(
            value.sole_normal_local,
            expected.sole_normal_local) ||
        value.foot_sphere_radius_m != expected.foot_sphere_radius_m ||
        !g1_leg_vec3_is_exact(
            value.thigh_start_local, expected.thigh_start_local) ||
        !g1_leg_vec3_is_exact(
            value.thigh_end_local, expected.thigh_end_local) ||
        value.thigh_radius_m != expected.thigh_radius_m ||
        !g1_leg_vec3_is_exact(
            value.shin_start_local, expected.shin_start_local) ||
        !g1_leg_vec3_is_exact(
            value.shin_end_local, expected.shin_end_local) ||
        value.shin_radius_m != expected.shin_radius_m ||
        value.reach_buffer_m != expected.reach_buffer_m ||
        value.planted_clearance_m != expected.planted_clearance_m ||
        value.swing_clearance_m != expected.swing_clearance_m ||
        value.max_swing_lift_m != expected.max_swing_lift_m ||
        value.max_correction_radians != expected.max_correction_radians) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (!g1_leg_vec3_is_exact(
                value.foot_sphere_centers_local[i],
                expected.foot_sphere_centers_local[i]) ||
            !g1_leg_vec3_is_exact(
                value.sole_points_local[i],
                expected.sole_points_local[i])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_leg_geometry_validate(
    const G1LegConfig& config,
    const G1LegConfig& expected,
    char* error,
    int error_capacity)
{
    const vec3 vectors[] = {
        config.knee_hinge_axis_local,
        config.foot_forward_local,
        config.sole_normal_local,
        config.foot_sphere_centers_local[0],
        config.foot_sphere_centers_local[1],
        config.foot_sphere_centers_local[2],
        config.foot_sphere_centers_local[3],
        config.sole_points_local[0],
        config.sole_points_local[1],
        config.sole_points_local[2],
        config.sole_points_local[3],
        config.thigh_start_local,
        config.thigh_end_local,
        config.shin_start_local,
        config.shin_end_local
    };
    for (const vec3 value : vectors) {
        if (!g1_leg_vec3_is_finite(value)) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: non-finite geometry",
                config.name);
        }
    }
    const float scalars[] = {
        config.foot_sphere_radius_m,
        config.thigh_radius_m,
        config.shin_radius_m,
        config.reach_buffer_m,
        config.planted_clearance_m,
        config.swing_clearance_m,
        config.max_swing_lift_m,
        config.max_correction_radians
    };
    for (const float value : scalars) {
        if (!terrain_float_is_finite(value)) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: non-finite geometry bound",
                config.name);
        }
        if (value <= 0.0f) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: geometry bounds must be positive",
                config.name);
        }
    }
    if (!g1_leg_axis_is_unit(config.knee_hinge_axis_local) ||
        !g1_leg_axis_is_unit(config.foot_forward_local) ||
        !g1_leg_axis_is_unit(config.sole_normal_local)) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: local axes must be unit length",
            config.name);
    }
    const float orthogonal_tolerance = 1.0e-5f;
    if (std::fabs(dot(
            config.knee_hinge_axis_local,
            config.foot_forward_local)) > orthogonal_tolerance ||
        std::fabs(dot(
            config.knee_hinge_axis_local,
            config.sole_normal_local)) > orthogonal_tolerance ||
        std::fabs(dot(
            config.foot_forward_local,
            config.sole_normal_local)) > orthogonal_tolerance) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: local axes must be orthogonal",
            config.name);
    }
    if (!g1_leg_vec3_is_exact(
            config.knee_hinge_axis_local,
            vec3(0.0f, 0.0f, -1.0f)) ||
        !g1_leg_vec3_is_exact(
            config.foot_forward_local,
            vec3(1.0f, 0.0f, 0.0f)) ||
        !g1_leg_vec3_is_exact(
            config.sole_normal_local,
            vec3(0.0f, 1.0f, 0.0f))) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: local basis does not match the mapped XML basis",
            config.name);
    }

    const float thigh_length_squared = g1_leg_length_squared(
        config.thigh_end_local - config.thigh_start_local);
    const float shin_length_squared = g1_leg_length_squared(
        config.shin_end_local - config.shin_start_local);
    if (thigh_length_squared <= 1.0e-12f ||
        shin_length_squared <= 1.0e-12f ||
        config.thigh_radius_m * config.thigh_radius_m >=
            thigh_length_squared ||
        config.shin_radius_m * config.shin_radius_m >=
            shin_length_squared) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: capsule geometry is degenerate or out of bounds",
            config.name);
    }

    const float sole_tolerance_squared = 1.0e-12f;
    for (int i = 0; i < 4; ++i) {
        const vec3 expected_point =
            config.foot_sphere_centers_local[i] -
            config.sole_normal_local * config.foot_sphere_radius_m;
        if (g1_leg_length_squared(
                config.sole_points_local[i] - expected_point) >
            sole_tolerance_squared ||
            std::fabs(dot(
                config.sole_points_local[i] - config.sole_points_local[0],
                config.sole_normal_local)) > 1.0e-6f) {
            return g1_ik_error(
                error,
                error_capacity,
                "%s leg: invalid sole probe relationship",
                config.name);
        }
        for (int j = 0; j < i; ++j) {
            if (g1_leg_length_squared(
                    config.foot_sphere_centers_local[i] -
                    config.foot_sphere_centers_local[j]) <=
                    sole_tolerance_squared ||
                g1_leg_length_squared(
                    config.sole_points_local[i] -
                    config.sole_points_local[j]) <=
                    sole_tolerance_squared) {
                return g1_ik_error(
                    error,
                    error_capacity,
                    "%s leg: duplicate sole probe relationship",
                    config.name);
            }
        }
    }

    if (!g1_leg_measured_geometry_matches(config, expected)) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: measured geometry contract mismatch",
            config.name);
    }
    return true;
}

static inline bool g1_leg_config_validate(
    const database& db,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    if (!g1_leg_database_shape_validate(db, error, error_capacity)) {
        return false;
    }
    if (config.name == NULL || config.name[0] == '\0') {
        return g1_ik_error(
            error, error_capacity, "G1 leg config has an invalid name");
    }
    const bool is_left = std::strcmp(config.name, "left") == 0;
    const bool is_right = std::strcmp(config.name, "right") == 0;
    if (!is_left && !is_right) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 leg config has malformed name '%s'",
            config.name);
    }
    const G1LegConfig expected =
        is_left ? g1_left_leg_config() : g1_right_leg_config();
    if (config.hip != expected.hip ||
        config.knee != expected.knee ||
        config.ankle != expected.ankle ||
        config.contact != expected.contact) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s leg: invalid named bone contract",
            config.name);
    }
    if (!g1_leg_geometry_validate(
            config, expected, error, error_capacity)) {
        return false;
    }
    const int expected_hip_parent =
        is_left ? G1_LeftHipRoll : G1_RightHipRoll;
    if (db.bone_parents(config.hip) != expected_hip_parent) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s %s parent mismatch",
            config.name,
            is_left ? "LeftHipYaw" : "RightHipYaw");
    }
    if (db.bone_parents(config.knee) != config.hip) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s %s parent mismatch",
            config.name,
            is_left ? "LeftKnee" : "RightKnee");
    }
    if (db.bone_parents(config.ankle) != config.knee) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s ankle parent mismatch",
            config.name);
    }
    if (db.bone_parents(config.contact) != config.ankle) {
        return g1_ik_error(
            error,
            error_capacity,
            "%s contact parent mismatch",
            config.name);
    }
    return true;
}

static inline bool g1_leg_configs_validate(
    const database& db, char* error, int error_capacity)
{
    if (!g1_leg_database_shape_validate(db, error, error_capacity)) {
        return false;
    }
    if (!g1_leg_database_local_basis_validate(
            db, error, error_capacity)) {
        return false;
    }
    if (!g1_leg_config_validate(
            db, g1_left_leg_config(), error, error_capacity) ||
        !g1_leg_config_validate(
            db, g1_right_leg_config(), error, error_capacity)) {
        return false;
    }
    return g1_skeleton_validate(db, error, error_capacity);
}
```

The XML positions are centers of four `0.02 m` collision spheres. Each stored
sole probe is the corresponding mapped center minus
`foot_sphere_radius_m * sole_normal_local`, so it represents the physical
bottom/support point rather than falsely treating the sphere center as the
sole. The two capsule definitions use their XML centerlines and retain their
radii. All values come from
`/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml`, mapped from
MuJoCo Z-up to Holden Y-up with `(x,y,z) -> (x,z,-y)`. The authoritative v2
builder applies that basis to every world position, conjugates every world
rotation, and then runs `world_to_local`, so its stored child offsets and this
geometry are both mapped. The legacy `resources/database.bin` used a
root-only/unmapped child-local convention and must never be paired with this
IK contract even though it has the same bone count and parent topology. The
six-offset all-row startup scan is the explicit provenance guard; `1e-6 m`
accepts float32 and observed roundoff while rejecting the legacy offsets by
centimeters.
`LeftAnkle/RightAnkle` map to the ankle-pitch bodies; `LeftToe/RightToe` map to
the ankle-roll bodies. `knee_hinge_axis_local` is the exact mapped MuJoCo
hinge axis, not a two-bone pole or bend vector. Task 3 derives its bend
direction from the current pose and uses this axis only for a deterministic
straight-leg fallback. Do not add an XML parser to the runtime.

- [ ] **Step 4: Wire validation into startup and lock source ordering**

Include `g1_ik.h` from `controller.cpp`. Immediately after the existing
successful `g1_skeleton_validate` block and before
`database_build_matching_features` or `InitWindow`, add:

```cpp
if (!g1_leg_configs_validate(
        db, artifact_error, static_cast<int>(sizeof(artifact_error))))
{
    fprintf(stderr, "G1 IK geometry error: %s\n", artifact_error);
    return 2;
}
```

Add a source-order regression to `tests/cpp/test_g1_controller_state.cpp`
that requires the `g1_ik.h` include and proves
`g1_skeleton_validate < g1_leg_configs_validate <
database_build_matching_features < InitWindow`.

- [ ] **Step 5: Run the named-geometry GREEN tests**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
/tmp/test_g1_ik
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_controller_state
/tmp/test_g1_controller_state
```

Expected: both compilations succeed and both tests exit `0` with no output.

- [ ] **Step 6: Commit the explicit geometry contract**

```bash
git add g1_ik.h controller.cpp tests/cpp/test_g1_ik.cpp \
  tests/cpp/test_g1_controller_state.cpp \
  docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md
git commit -m "test: lock explicit G1 terrain IK geometry"
```

### Task 2: Sample Exact Surface Normals and Lock Planted Sole Targets at 25 Hz

**Files:**
- Modify: `g1_ik.h`
- Modify: `tests/cpp/test_g1_ik.cpp`
- Modify: `docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md`

**Implemented interfaces and contract:**
- `G1SurfaceQueryStatus { Valid, Outside, Invalid }` and
  `g1_surface_query_v2` are the reusable fail-closed primitive for this task
  and Task 5. They validate a complete G1HF/v2 field, canonicalize query
  coordinates, call `terrain_v2_locate_cell` and
  `terrain_v2_cell_heights`, and reproduce the authoritative fixed-diagonal
  height and normal arithmetic exactly. Neither error status returns or
  accepts `exterior_height`, and the caller's output remains unchanged.
- `g1_surface_target_sample` is transactional. It reports exterior
  coordinates as `Outside`, malformed fields/data/math as `Invalid`, and
  uses checked binary64-to-binary32 addition for height plus clearance.
- `g1_ik_dt_is_exact_25_hz` accepts only binary32 bits `0x3d23d70a`.
  Task 5 swing planning and Task 6 frame evaluation must reuse this helper;
  tolerant comparisons and neighboring `nextafter` values are invalid.
- `g1_foot_lock_reset` is checked and transactional:
  `bool g1_foot_lock_reset(state, center, error, error_capacity)`.
- `G1FootLockState` owns explicit `position_active`, `releasing`, and
  `release_frames` state. Its invariant is
  `position_active == (locked || releasing)`; locked implies contact,
  releasing implies no contact, and inactive states have zero spring offsets.
- `G1FootTarget` mirrors `locked`, `position_active`, and `releasing`.
  Task 6 must consume `target.sole_center` whenever `position_active` is
  true. Surface alignment remains locked-only.
- Rising, falling, and release-recontact event frames preserve the previous
  output position bit-for-bit. Stable lock/release frames consume the checked
  `0.10 s` critical spring. Release deactivates when every position offset is
  at most `1e-4 m` and every velocity offset is at most `1e-3 m/s` after at
  least two stable release updates, or after the deterministic 25-frame cap,
  then snaps exactly to animation.
- Update validates the complete state, prior output, exact named-leg config,
  input, and timestep before derived arithmetic. Input velocity, transitions,
  spring stages, height plus clearance, and horizontal drift use checked
  rounding. Every failure rolls back both state and output.
- Every active lock point must bit-match a fresh checked surface target at its
  frozen XZ and configured clearance before the stable lock spring advances;
  finite state corruption or changed scene data therefore fails closed.
- Horizontal drift is measured from the frozen lock point in binary64:
  exactly `0.20f` is allowed, its next larger binary32 value is exceeded, and
  active recorded contact never auto-unlocks.

- [x] **Step 1: Add RED coverage for the checked query and lock lifecycle**

The tests cover flat and ramp surfaces, both non-planar fixed-diagonal
triangles and the tie rule, all inclusive corners, one-ULP exterior points,
malformed version/storage, selected NaN/subnormal heights, public-target
rollback, and checked height-plus-clearance overflow. Lock tests cover
convergence, exact event-frame continuity, multi-frame and bounded release,
release recontact, exact drift boundaries, poisoned state/config/output,
non-finite and finite-extreme inputs, exact timestep neighbors, reset
rollback, and hostile diagnostic buffers.

- [x] **Step 2: Confirm RED before implementing**

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task2_red
```

The compile failed on the intentionally undefined checked-query and lock
interfaces before production code was added.

- [x] **Step 3: Implement the fail-closed surface primitive and transactional lock**

Implementation lives in `g1_ik.h`. It does not touch `controller.cpp` or
any resource artifact, and does not require or shallow-copy a `database`.

- [x] **Step 4: Verify strict, fast-math release, and ASAN/UBSAN builds**

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task2_strict
/tmp/test_g1_ik_task2_strict

g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task2_release
/tmp/test_g1_ik_task2_release

g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_g1_ik.cpp \
  -o /tmp/test_g1_ik_task2_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_g1_ik_task2_san
```

All three executables must exit zero; sanitizers must emit no finding.

- [x] **Step 5: Commit exact checked-surface planted locking**

```bash
git add g1_ik.h tests/cpp/test_g1_ik.cpp \
  docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md
git commit -m "feat: lock planted G1 feet to checked terrain"
```

### Task 3: Add a Checked, Dynamic-Reach-Shell Named Two-Bone Solve

**Files:**
- Create: `ik.h`
- Modify: `g1_ik.h`
- Modify: `tests/cpp/test_g1_ik.cpp`
- Modify: `docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md`

**Final interfaces and invariants:**
- `ik.h` owns renderer-independent checked arithmetic plus
  `IKTargetProjection`, `IKClampResult`, `IKBendSelection`,
  `IKTwoBoneResult`, `ik_project_target`, `ik_clamp_local_delta`,
  `ik_select_bend_direction`, and `ik_two_bone_bounded`.
- `g1_ik.h` owns `G1LegSolveResult`,
  `g1_ik_checked_forward_kinematics`,
  `g1_apply_named_position_ik`, and
  `g1_apply_named_contact_position_ik`.
- Every public path validates complete non-null 31-bone slices, the exact
  parent topology, the exact named G1 leg configuration and chain, normal-or-
  zero local positions, and finite unit local quaternions before indexing.
  Working and immutable baseline rotation ranges may not overlap, and the two
  checked-FK output ranges may not overlap exactly or partially. Each checked-
  FK output is also disjoint from every const local-position, local-rotation,
  and parent input range, so the semantic const-input contract is explicit.
- Baseline and post-solve FK use staged checked outputs. Every produced global
  position is normal-or-zero and every global quaternion is finite/unit before
  any caller output is committed. These checks remain active under
  `NDEBUG`.
- Generic results, the caller pose, and `G1LegSolveResult` are transactional.
  A named solve stages the caller's current working pose, replaces only the
  target hip/knee from the immutable baseline solve, and preserves every
  non-target rotation byte. A second-leg solve therefore preserves the first
  staged leg.
- Norm, dot, cross, quaternion, shell, law-of-cosines, and residual arithmetic
  is promoted and checked before binary32 commit. Subnormal, non-finite,
  overflowed, and unrepresentable intermediates fail without output mutation;
  a nonzero promoted value may not silently round or flush to binary32 zero.
- Quaternion inputs admitted by the unit tolerance are normalized before
  shortest-arc measurement. `limited` is true iff the precise requested
  angle is strictly greater than the configured maximum, with no epsilon.
  A bounded slerp refinement guarantees both the retained promoted correction
  and its binary32 report are never greater than the maximum.
- Bend selection first uses the current hip-to-knee projection when its
  pre-materialization promoted length is at least `1e-6`. Otherwise it uses
  `cross(hinge_axis_world, target_direction)` when that cross's promoted
  length is at least `1e-6`, flips only when the raw promoted projection is
  strictly greater than `1e-8` and disagrees, and reserves a checked safe
  perpendicular for a degenerate/parallel hinge fallback.
  The named adapter maps the exact configured local `-Z` through the
  immutable baseline knee global rotation.
- The contact adapter recomputes checked FK from a staged working pose for at
  most four iterations against the same immutable baseline. Convergence owns
  the exact promoted predicate `residual <= double(0.005f)`; the binary32 value
  is reporting only and cannot turn an over-limit residual into convergence.
  A finite residual above that threshold commits the bounded safe-stop pose;
  invalid later math rolls back the entire pose and result. The reported
  residual equals a fresh checked FK of the committed pose.

**Dynamic reach-shell correction:**

A fixed buffered shell is unsafe for natural G1 walk frames near extension.
For example, the nominal maximum is `upper + lower - 0.015 m`; asking for
the already-current endpoint can therefore project a valid current leg inward
and immediately apply the full `0.35 rad` cap.

The checked projection now defines:

```text
nominal_min = abs(upper - lower) + reach_buffer
nominal_max = upper + lower - reach_buffer
current     = length(end - root)
effective_min = min(nominal_min, current)
effective_max = max(nominal_max, current)
```

The effective shell always contains the current endpoint. If the current pose
is beyond the nominal outer side, a target farther outward clamps at
`current`; if it is inside the nominal inner side, a target farther inward
clamps at `current`. Movement back toward the nonsingular nominal interior
remains allowed. Thus the exception never permits moving farther toward the
singularity.

Reachability and shell-side decisions use the promoted distances. A reachable
target is preserved exactly. A clamped boundary target is materialized and
independently remeasured; binary32 rounding is iteratively biased toward the
shell interior until its actual radius is within the precise effective shell,
or the operation fails transactionally. The remeasured promoted radius, not
the rounded report, drives the law of cosines.

When the promoted projected motion is within `double(1e-6f)` of the current
endpoint, the solver preserves the target hip/knee baseline quaternions
exactly and reports zero correction. A requested target within the same
tolerance is treated as a reachable no-op; a farther target projected back to
the current singular side remains unreachable and requests safe stop. This
makes the lock rising edge a fixed point while one-millimeter inward/outward
updates remain smooth and directionally bounded. A legitimate exact-folded
chain has a zero endpoint radius and is also a valid named ankle/contact fixed
point.

- [x] **Step 1: Expand RED coverage before implementation**

The expanded test surface covers:

- exact clamp boundary/neighbor classification, antipodal equivalence,
  admitted scaled-unit normalization, a 4,608-case bounded clamp sweep,
  invalid quaternions, and complete generic-result rollback;
- ordinary, folded, 128 current-end shell invariants, rotated expanded-boundary
  behavior, a fixed-bit materialized outer-shell counterexample, a promoted
  just-over-no-op folded target, zero/subnormal links, opposite `FLT_MAX`
  inputs, and finite `1e20` rejection;
- exact bend thresholds and neighbors, fixed-bit pre-materialization midpoint
  cases for primary projection, hinge cross, and sign continuity,
  `cross(-Z,-Y) == -X`, parallel fallback, and invalid hinges;
- named success, current-endpoint fixed points, one-millimeter updates,
  exact-folded named ankle/contact fixed points, mirrored legs, immutable
  baseline hinge mapping, non-target byte preservation, and staged second-leg
  preservation;
- null and short working/local/baseline/parent slices, partial aliasing, every
  parent corruption, exact/partial checked-FK output overlap, all checked-FK
  output/input type combinations including parents, malformed named
  configurations, invalid baseline and working quaternions, source and derived
  subnormals, baseline-FK overflow, candidate-only post-solve FK overflow, and
  target failure rollback;
- exact promoted `0.005 m` residual threshold ownership including a value that
  reports as `0.005f` but is precisely greater, fresh-FK result equality,
  four-iteration safe stop, and late residual-overflow rollback.

Targeted REDs were observed for the initially undefined interfaces, the
candidate-only post-FK overflow, the padding-sensitive release rollback
oracle, scaled-equivalent clamp normalization, clamp-bound overshoot, the
fixed nominal shell, materialized shell overshoot, premature rounded no-op,
pre-materialization bend predicate changes, const-input alias mutation,
derived-subnormal flushing, exact-folded result rejection, rounded residual
misclassification, and current ankle/contact fixed-point behavior.

- [x] **Step 2: Implement checked generic math and named adapters**

Implementation is in `ik.h` and `g1_ik.h`. It does not modify
`controller.cpp` or any protected resource binary. The old dormant
controller math remains untouched until the later atomic integration task.

- [x] **Step 3: Verify strict, release, sanitizer, and controller probes**

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task3_debug
/tmp/test_g1_ik_task3_debug

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task3_strict
/tmp/test_g1_ik_task3_strict

g++ -std=c++17 -O3 -ffast-math -DNDEBUG -fno-elide-constructors \
  -Wall -Wextra -Werror -Wno-unused-variable -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task3_release
/tmp/test_g1_ik_task3_release

g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_task3_san
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/test_g1_ik_task3_san

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state_task3
/tmp/test_g1_controller_state_task3

g++ -std=c++17 -O3 -ffast-math -DNDEBUG -fno-elide-constructors \
  -Wall -Wextra -Werror -Wno-unused-variable -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/test_g1_controller_state_task3_fast
/tmp/test_g1_controller_state_task3_fast

g++ -O3 -ffast-math -march=native -DNDEBUG -D_DEFAULT_SOURCE \
  -DPLATFORM_DESKTOP -I. -I /home/ubuntu/apps/raylib/src \
  -I /home/ubuntu/apps/raygui/src controller.cpp \
  -o /tmp/controller_g1_task3 -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

All executables exit zero. ASAN/UBSAN/leak detection emits no finding.
`-Wno-unused-variable` suppresses only existing `array.h` template
variables whose uses are assertions removed by `NDEBUG`. Clang is not
installed in this environment (no `clang++` or versioned alternative), so
the requested Clang probe is recorded as unavailable and nonblocking; no
system package was installed.

- [x] **Step 4: Probe certified v2 runtime frames without a test dependency**

A temporary, uncommitted probe loaded the absolute certified pack:

```text
/home/ubuntu/projects/motion-matching/resources/g1_terrain/database.bin
SHA256 1849ecbc3775fda0a7cb2fb1bfe9ed15d0f457bd86d977d60bbe6de8b0a0fed6
```

It sampled frames `0`, `114920`, `229841`, `344761`, and `459681`
for both named legs. Strict and
`-O3 -ffast-math -DNDEBUG -fno-elide-constructors` builds both passed:

- checked-FK versus controller-FK maximum position delta was
  `2.95e-7 m` strict and `4.92e-7 m` fast;
- every baseline ankle target and baseline contact target preserved all 31
  local quaternion bit patterns, reported exactly `0` correction and
  `0` checked residual, and remained reachable/unlimited/no-safe-stop;
- maximum controller-FK residuals of those committed no-op poses were only
  `1.19e-7 m` strict and `2.68e-7 m` fast.

The temporary probe source was removed before staging, so committed tests do
not depend on the 708 MB external artifact.

- [x] **Step 5: Commit the bounded named solver**

```bash
git add ik.h g1_ik.h tests/cpp/test_g1_ik.cpp \
  docs/superpowers/plans/2026-07-13-g1-terrain-ik-clearance.md
git commit -m "feat: solve named G1 legs within bounded reach"
```

### Task 4: Align the Named Foot to the Exact Surface Normal

**Files:**
- Modify: `g1_ik.h`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Produces: `G1FootOrientationResult`, `g1_surface_aligned_foot_rotation`, and `g1_apply_named_foot_orientation`.
- The target maps configured local sole normal `+Y` to the exact upward G1HF/v2 triangle normal while preserving the current configured `+X` foot heading projected into the tangent plane.
- Only `config.contact` (`LeftToe`/`RightToe`, the ankle-roll body) receives the orientation delta. The delta is measured from the support-retargeted IK-off local quaternion and clamped to `0.35 radians`.

- [ ] **Step 1: Add failing flat, ramp, cross-slope, and steep-limit tests**

Append this test before `main` in `tests/cpp/test_g1_ik.cpp`, then call it from `main`:

```cpp
static void test_surface_aligned_named_foot_orientation()
{
    database db;
    make_g1_database(db);
    const G1LegConfig leg = g1_left_leg_config();
    char error[256] = {};
    G1FootOrientationResult result = {};

    array1d<quat> output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.0f, 1.0f, 0.0f),
              result, error, sizeof(error)), error);
    check(result.applied && !result.correction_limited,
          "flat orientation applies without limit");
    check(result.correction_radians < 1e-6f,
          "flat identity orientation is unchanged");

    const float angle = 10.0f * PIf / 180.0f;
    const vec3 ramp_normal(-std::sin(angle), std::cos(angle), 0.0f);
    output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, ramp_normal,
              result, error, sizeof(error)), error);
    array1d<vec3> global_positions(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    forward_kinematics_full(
        global_positions, global_rotations,
        db.bone_positions(0), output, db.bone_parents);
    const vec3 ramp_up = quat_mul_vec3(
        global_rotations(leg.contact), leg.sole_normal_local);
    check(dot(ramp_up, ramp_normal) > 0.9999f,
          "foot aligns to longitudinal ramp normal");

    const vec3 cross_normal(0.0f, std::cos(angle), -std::sin(angle));
    output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, cross_normal,
              result, error, sizeof(error)), error);
    forward_kinematics_full(
        global_positions, global_rotations,
        db.bone_positions(0), output, db.bone_parents);
    const vec3 cross_up = quat_mul_vec3(
        global_rotations(leg.contact), leg.sole_normal_local);
    const vec3 cross_forward = quat_mul_vec3(
        global_rotations(leg.contact), leg.foot_forward_local);
    check(dot(cross_up, cross_normal) > 0.9999f,
          "foot aligns to cross-slope normal");
    check(cross_forward.x > 0.999f &&
          std::fabs(dot(cross_forward, cross_normal)) < 1e-5f,
          "foot tangent heading is preserved");

    const float steep = 45.0f * PIf / 180.0f;
    output = db.bone_rotations(0);
    check(g1_apply_named_foot_orientation(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg,
              vec3(-std::sin(steep), std::cos(steep), 0.0f),
              result, error, sizeof(error)), error);
    check(result.correction_limited && result.safe_stop_requested,
          "steep orientation requests safe stop");
    check(result.correction_radians <= 0.350001f,
          "steep orientation remains bounded");
    for (int bone = 0; bone < G1_BoneCount; ++bone)
        if (bone != leg.contact)
            check(same_quat(output(bone), db.bone_rotations(0, bone)),
                  "orientation changes only named contact bone");
}
```

- [ ] **Step 2: Compile to verify foot-orientation RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
```

Expected: compilation fails because `G1FootOrientationResult` and the named orientation functions are undefined.

- [ ] **Step 3: Implement heading-preserving surface alignment**

Append to `g1_ik.h`:

```cpp
struct G1FootOrientationResult
{
    bool applied = false;
    bool correction_limited = false;
    bool safe_stop_requested = false;
    quat target_global_rotation;
    float requested_correction_radians = 0.0f;
    float correction_radians = 0.0f;
};

static inline bool g1_surface_aligned_foot_rotation(
    quat& output,
    quat current_global_rotation,
    const G1LegConfig& config,
    vec3 surface_normal,
    char* error,
    int error_capacity)
{
    if (!ik_quat_is_unit(current_global_rotation) ||
        !g1_ik_vec3_is_runtime_value(surface_normal) ||
        surface_normal.y <= 0.0f ||
        std::fabs(length(surface_normal) - 1.0f) > 1e-4f) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation received non-finite surface normal",
            config.name);
    }
    vec3 forward = quat_mul_vec3(
        current_global_rotation, config.foot_forward_local);
    forward = forward - surface_normal * dot(forward, surface_normal);
    if (length(forward) < 1e-6f) {
        const vec3 fallback = std::fabs(surface_normal.x) < 0.75f
            ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 0.0f, 1.0f);
        forward = fallback - surface_normal * dot(fallback, surface_normal);
    }
    forward = normalize(forward);
    const quat target = quat_from_xform_xy(forward, surface_normal);
    if (!ik_quat_is_unit(target)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation produced non-finite target", config.name);
    }
    output = target;
    return true;
}

static inline bool g1_apply_named_foot_orientation(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 surface_normal,
    G1FootOrientationResult& output,
    char* error,
    int error_capacity)
{
    if (output_rotations.size != G1_BoneCount ||
        local_positions.size != G1_BoneCount ||
        baseline_rotations.size != G1_BoneCount ||
        parents.size != G1_BoneCount) {
        return g1_ik_error(
            error, error_capacity, "%s foot orientation pose shape mismatch",
            config.name);
    }
    array1d<vec3> global_positions(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    forward_kinematics_full(
        global_positions, global_rotations,
        local_positions, output_rotations, parents);
    quat target_global = {};
    if (!g1_surface_aligned_foot_rotation(
            target_global, global_rotations(config.contact),
            config, surface_normal, error, error_capacity)) return false;
    const int parent = parents(config.contact);
    if (parent < 0) {
        return g1_ik_error(
            error, error_capacity, "%s contact bone has no parent", config.name);
    }
    const quat desired_local = quat_inv_mul(
        global_rotations(parent), target_global);
    IKClampResult bounded = {};
    if (!ik_clamp_local_delta(
            bounded, baseline_rotations(config.contact), desired_local,
            config.max_correction_radians)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation produced non-finite correction", config.name);
    }
    output_rotations(config.contact) = bounded.value;
    G1FootOrientationResult candidate = {};
    candidate.applied = true;
    candidate.correction_limited = bounded.limited;
    candidate.safe_stop_requested = candidate.correction_limited;
    candidate.target_global_rotation = target_global;
    candidate.requested_correction_radians = bounded.requested_radians;
    candidate.correction_radians = bounded.actual_radians;
    output = candidate;
    return true;
}
```

Do not rotate `config.ankle` a second time for orientation: it is the two-bone end, while `config.contact` is the measured ankle-roll/sole body. This avoids the inherited heel/toe parent walk and hard-coded toe-end vector.

- [ ] **Step 4: Run orientation GREEN and all G1 IK tests**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_strict
/tmp/test_g1_ik_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_release
/tmp/test_g1_ik_release
```

Expected: both executables exit `0`; the 10-degree longitudinal and lateral normals align within the asserted tolerance, and the steep case is bounded with a safe-stop request.

- [ ] **Step 5: Commit bounded foot orientation**

```bash
git add g1_ik.h tests/cpp/test_g1_ik.cpp
git commit -m "feat: align G1 feet to terrain normals"
```

### Task 5: Add Swept Swing Clearance and Toe/Foot/Shin/Leg Diagnostics

**Files:**
- Create: `g1_surface_query.h`
- Modify: `g1_ik.h`
- Create: `g1_clearance.h`
- Create: `g1_clearance.cpp`
- Create: `tests/cpp/test_g1_clearance.cpp`
- Later Task 6 owns the specified integration in: `g1_ik_runtime.h`
- Later Task 6 owns the specified integration tests in: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Moves exactly the six existing surface-query definitions named below from
  `g1_ik.h` into one lightweight `g1_surface_query.h` definition source,
  without changing code, math, status, layout, signature, or API.
- `g1_ik.h` includes `g1_surface_query.h` for existing callers; the strict
  kernel includes it directly. `g1_clearance.h` remains unchanged by this move.
- Produces: non-inline strict-FP `g1_point_clearance`, `g1_foot_clearance`, and conservative deterministic `g1_capsule_clearance` against the continuous fixed-diagonal G1HF/v2 surface.
- Produces: `G1LegClearance` and `G1PoseClearance` with signed Hips, knee, ankle, toe, four-point sole, thigh-capsule, and shin-capsule clearances against the exact G1HF/v2 surface.
- Produces: `G1SwingHistory`, checked reset/commit, strict `g1_apply_swing_lift_y`, actual-center `g1_swing_clearance_validate`, and the immutable staged-candidate selection contract consumed by Task 6.
- Swing selection evaluates only real post-IK/FK sphere endpoints at exact 25 Hz. It never translates predicted baseline spheres, estimates a continuous required lift, or samples a guessed lifted segment.

> **Authoritative supersession (reviewed commit `66836d8`, independently CLEAN):** Sections 3, 6, 9, 10, 11, 12E/H/I/J, and 14 of `docs/superpowers/plans/2026-07-14-g1-certified-clearance-design.md` replace every sampled/predicted clearance, scalar-float safety result, boolean status use, unchecked history operation, and one-command fast-math build in Tasks 5--10. All active downstream snippets must pass immutable budgets, branch on explicit `G1ClearanceStatus`, read `G1ClearanceResult::lower_bound_m` as binary64, log safety values without a float round trip, and compile/link `g1_clearance.cpp` as a separate strict-FP object. Step 4 below and the migrated downstream tasks are the only active contracts; no old and new path may coexist.

- [ ] **Step 1: Write failing certified-geometry and real-staging tests**

Create `tests/cpp/test_g1_clearance.cpp` from the exact RED fixtures in
Sections 12A--K of the reviewed certified-clearance design. Tests use `check`,
not `assert`, seed every failure-path output/history owner with distinct bits,
and verify unchanged outputs for every non-`Ok` result. In particular:

- lock exact continuous fixed-diagonal point, sphere, capsule, foot, swept-foot,
  and pose certificates; no test may accept a spatial sample count as proof;
- lock valid, outside-domain, and invalid-field surface-query fixtures across
  strict and fast callers, comparing exact status plus raw height/normal bits
  or unchanged seeded output bits byte-for-byte;
- call `g1_swing_clearance_validate` with named prior and **actual current**
  four-sphere center bits and compare the target-subtracted capsule result to the
  high-precision endpoint-bit oracle;
- independently derive every ladder entry as `RN32(i/500 m)` for `i=0..40`,
  require exact positive-zero and `0.08f` endpoints, and compare all 41 bits;
- lock round-to-nearest/gradual-underflow rejection, nonfinite and nonzero
  subnormal input behavior, the subnormal-command failure, exact 25 Hz, and
  transactional history reset/commit.

When Task 6 creates `g1_ik_runtime.h`, add the Section 12E integration fixture
to `tests/cpp/test_g1_ik.cpp` under `G1_IK_ENABLE_TEST_SEAMS`. It must call the
one real production staging function for indices `0..40` from the same immutable
input, derive the first finite passing stage, and compare it with the full
selector. Lock all of these outcomes:

- at least one real finite rejection precedes the selected stage;
- calls are contiguous from zero, `candidates_evaluated == selected_index + 1`,
  and the selected lift, materialized command, twelve actual sphere-center bits,
  binary64 margin, and work match the probed stage;
- final rotations reproduce those endpoint bits and the selected staged result
  is reused with no additional position-IK/orientation solve;
- a real wall fixture makes exactly 41 real calls, returns
  `G1SwingNoCandidate`, requests safe stop, and leaves accepted pose, state,
  histories, support, matcher, and simulation bit-identical;
- independently OR the three candidate-local rejection bits from the probed
  real stages and require the selector aggregate to match through first-pass
  selection and all-41 failure. Mix all three statuses, repeat statuses, and
  prove negative-`Ok`/controller-only rejection adds no bit;
- strict and fast-math callers certify identical supplied endpoint bits
  byte-for-byte. Per-build IK endpoints may differ, but each build must expose
  and deterministically certify its own first actual pass.

A negative compile names `g1_ik_stage_swing_candidate_for_test` without
`G1_IK_ENABLE_TEST_SEAMS` and must fail. A production object must contain no
declaration or symbol for the seam.

- [ ] **Step 2: Run the certified-clearance RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance
```

Expected before implementation: compilation fails because the reviewed public
status/result API or `g1_clearance.cpp` implementation is absent. Do not make
this RED pass with inline sampling helpers.

- [ ] **Step 3: Implement the strict certified geometry and diagnostics**

Create `g1_clearance.h` and non-inline `g1_clearance.cpp` exactly from Sections
3--8 and reviewed design Tasks 1--6. This is the only active geometry path:

Before any geometry implementation, complete the reviewed ownership-only
prerequisite:

- create `g1_surface_query.h` and **move, never copy**, exactly these six
  existing definitions out of `g1_ik.h`: `G1SurfaceQueryStatus`,
  `G1SurfaceSample`, `g1_ik_float_is_runtime_value`,
  `g1_ik_vec3_is_runtime_value`, `g1_ik_surface_normal_is_valid`, and
  `g1_surface_query_v2`;
- preserve every name, enum value, struct layout, signature, function body,
  status mapping, and arithmetic expression exactly;
- make `g1_surface_query.h` their sole definition source. It includes only
  `terrain_runtime.h` plus required standard math declarations and does not
  include `g1_ik.h`, `g1_skeleton.h`, or `database.h`;
- have `g1_ik.h` include the lightweight header so existing callers continue
  through the same public include. Have strict `g1_clearance.cpp` include it
  directly after its implementation-TU include, without including `g1_ik.h`;
  `g1_clearance.h` remains unchanged;
- do not move `g1_ik_dt_is_exact_25_hz` or
  `g1_foot_runtime_config_validate`. Their exact-dt/config ownership is
  explicitly deferred to later certified Task 5/Task 6 integration work.

This prerequisite changes no code, math, status, layout, or API. Run the
existing IK and Task 1 clearance tests and the exact one-owner,
include/dependency, `nm`, and strict-versus-fast query-parity guards in Step 6,
then commit it separately before adding geometry:

```bash
git add g1_surface_query.h g1_ik.h g1_clearance.cpp \
  tests/cpp/test_g1_clearance.cpp
git commit -m "refactor: isolate G1 surface query"
```

Then implement the certified geometry:

- public point, sphere, capsule, foot, swept-foot, leg, and pose functions
  return `G1ClearanceStatus` and assign result/history outputs only on their
  documented success path;
- safety bounds, witnesses, pose minima, and swing margins remain binary64;
- sphere/capsule clearance minimizes continuously against the authoritative
  fixed-diagonal G1HF/v2 triangles with outward intervals, exact footprint
  containment, fixed absolute budgets, and the mandatory producer-output guard;
- point lattices, radial lattices, `segment_steps`, `ceil`-derived work, rounded
  terrain-sample minima, and header-only proof arithmetic are forbidden;
- `OutsideDomain`, `BudgetExceeded`, and `Uncertified` are distinct fail-closed
  outcomes. `InvalidInput`, `InvalidField`, and `ArithmeticFailure` retain their
  reviewed transactional meanings;
- `g1_clearance.cpp` rejects `__FAST_MATH__`, validates round-to-nearest plus
  gradual binary32/binary64 underflow before input-dependent arithmetic, and is
  compiled with `-fno-fast-math -ffp-contract=off -frounding-math`.

Implement checked `G1SwingHistory` reset/commit,
`g1_apply_swing_lift_y`, and actual-center `g1_swing_clearance_validate` in
this strict boundary. The validator receives no lift value: its current sphere
centers already contain the actual controller/IK/FK result.

- [ ] **Step 4: Implement certified staged swing lift selection**

This step supersedes every former sampled/predicted lift search. Task 5 supplies
the strict materializer and actual-center certificate; Task 6 owns the runtime
staging transaction described here.

The immutable production ladder in `g1_ik_runtime.h` is:

```cpp
constexpr uint32_t G1SwingLiftCandidateCount = 41;
constexpr uint32_t G1SwingNoCandidate = UINT32_MAX;
constexpr uint32_t G1SwingLiftCandidateBits[41] = {
    0x00000000u, 0x3b03126fu, 0x3b83126fu, 0x3bc49ba6u,
    0x3c03126fu, 0x3c23d70au, 0x3c449ba6u, 0x3c656042u,
    0x3c83126fu, 0x3c9374bcu, 0x3ca3d70au, 0x3cb43958u,
    0x3cc49ba6u, 0x3cd4fdf4u, 0x3ce56042u, 0x3cf5c28fu,
    0x3d03126fu, 0x3d0b4396u, 0x3d1374bcu, 0x3d1ba5e3u,
    0x3d23d70au, 0x3d2c0831u, 0x3d343958u, 0x3d3c6a7fu,
    0x3d449ba6u, 0x3d4ccccdu, 0x3d54fdf4u, 0x3d5d2f1bu,
    0x3d656042u, 0x3d6d9168u, 0x3d75c28fu, 0x3d7df3b6u,
    0x3d83126fu, 0x3d872b02u, 0x3d8b4396u, 0x3d8f5c29u,
    0x3d9374bcu, 0x3d978d50u, 0x3d9ba5e3u, 0x3d9fbe77u,
    0x3da3d70au,
};

constexpr uint32_t G1SwingRejectOutsideDomainBit = 1u << 0;
constexpr uint32_t G1SwingRejectBudgetExceededBit = 1u << 1;
constexpr uint32_t G1SwingRejectUncertifiedBit = 1u << 2;
constexpr uint32_t G1SwingRejectKnownMask =
    G1SwingRejectOutsideDomainBit |
    G1SwingRejectBudgetExceededBit |
    G1SwingRejectUncertifiedBit;
```

Entry `i` is canonical binary32 `RN32(i/500 m)`: immutable 2 mm
increments from `+0.00 m` through `0.08 m` inclusive. Load checked-in bits with
the existing `memcpy` helper. Never generate entries by float arithmetic, use a
binary search/non-ladder float iteration, infer a pass between entries, or
invent candidate 41.

Add `uint32_t clearance_rejection_status_mask = 0;` to
`G1SwingSelectionDiagnostic`. This is the only aggregate evidence retained from
rejected candidates. It exposes no rejected pose, endpoint, margin, witness, or
work record. Bit 0 means at least one real stage returned `OutsideDomain`, bit
1 means `BudgetExceeded`, and bit 2 means `Uncertified`; no other bit is valid.

For each frame:

1. Before caller-owned mutation, validate exact 25 Hz, shapes, state, G1HF/v2,
   leg configuration, `max_swing_lift_m` bits equal entry 40, and the strict
   arithmetic environment. Snapshot one immutable per-foot baseline. Recorded
   contact bypasses the ladder with `candidates_evaluated=0` and
   `selected_index=G1SwingNoCandidate` and mask zero; the contact flag
   distinguishes this success from all-fail safe stop.
2. For each non-contact foot, visit indices `0..40` in exact order. Every
   candidate starts from a fresh copy of the same baseline. Call non-inline
   strict `g1_apply_swing_lift_y` on the **actual**
   `desired_sole_center.y`, then run the real named bounded position IK, foot
   orientation, controller checks, and checked FK in their production order.
3. Compute the four configured world-space foot-sphere centers from that staged
   FK, record all twelve raw binary32 bits at the certificate boundary, and call
   `g1_swing_clearance_validate` on those exact values with a fresh per-call
   swing budget. For each sphere, certify the target-subtracted prior/current
   centerline as a continuous capsule against exact G1HF/v2. Form adjusted Y
   with binary64 `TwoDiff`; never round it through binary32 or predict
   `baseline_sphere_y + lift`.
4. A candidate passes only when every controller constraint passes, clearance
   status is `G1ClearanceOk`, and binary64 `lower_margin_m >= 0.0`. A finite
   controller rejection, negative `Ok` margin, `OutsideDomain`,
   `BudgetExceeded`, or `Uncertified` rejects only that candidate and advances
   to the next index. `InvalidInput`, `InvalidField`, or `ArithmeticFailure`
   aborts the frame transaction unchanged. Before continuing, OR exactly the
   corresponding bit for `OutsideDomain`/`BudgetExceeded`/`Uncertified` into a
   local aggregate; `Ok` with a negative margin and controller-only rejection
   add no bit.
5. Select the first passing candidate and move its already-staged pose/result
   into the outer scratch pose. Reuse its certificate and endpoint bits; do not
   rerun IK to apply it. Copy the aggregate mask accumulated only from earlier
   rejected real stages. After both feet compose, run checked FK once, require
   selected endpoint-bit equality, and perform the mandatory fresh actual-center
   certificate. Any mismatch or failed defensive certificate rolls back the
   entire frame.
6. If all 41 candidates finitely reject, report exactly
   `candidates_evaluated=41`, `selected_index=G1SwingNoCandidate`, default
   ignored selected fields, the exact aggregate mask, and safe stop. A zero
   all-fail mask is valid when all stages failed only controller constraints or
   negative `Ok` margins; any claim that unresolved certification caused the
   exhaustion requires the relevant nonzero bit. Roll back accepted pose, clearance,
   histories, state, support, matcher, and simulation. Never report a fabricated
   continuous `required_lift_m`.

Expose candidate probing only inside
`#if defined(G1_IK_ENABLE_TEST_SEAMS)`. The wrapper calls the same private
production stage and returns diagnostics only; it cannot duplicate geometry or
expose a mutable staged pose. Builds without the macro contain neither its
declaration nor its symbol.

Finite/subnormal semantics are exact: every public status entry checks
round-to-nearest and gradual underflow before input-dependent arithmetic; FTZ,
DAZ, or failed volatile binary32/binary64 denormal probes return transactional
`ArithmeticFailure`. `g1_apply_swing_lift_y` requires runtime `input_y`
(finite, signed zero canonicalized, nonzero subnormal rejected as
`InvalidInput`) and finite nonnegative `lift_m <= 0.08f`; either lift-zero sign
becomes `+0.0f`. It performs exactly
`RN32(double(input_y)+double(lift_m))`, canonicalizes a zero result, and rejects
a nonfinite or nonzero-subnormal materialized command as `ArithmeticFailure`
with output unchanged. Positive subnormal test lifts retain their input bits but
are not guaranteed to materialize: `+0.0f + denorm_min` must fail unchanged.
Production ladder entries above zero are normal. Geometry canonicalizes signed
zero and rejects every nonzero binary32 subnormal coordinate as `InvalidInput`.

- [ ] **Step 5: Add exact post-IK leg/pose aggregation**

Implement `G1LegClearance` and `G1PoseClearance` from reviewed design Section 3
and Task 5. Preserve the named Hips, knee, ankle, toe, four-sphere foot, thigh
capsule, and shin capsule diagnostics, but store `G1ClearanceResult` values and
use binary64 lower bounds for every safety decision. Share one validated
absolute pose budget through the complete aggregation; a late failure assigns
no public result. No float minimum or sampled `G1MinimumClearance` API remains.

- [ ] **Step 6: Run strict, release-caller, sanitizer, and staging verification**

Build the kernel separately and never pass `-ffast-math` to the final link:

```bash
mkdir -p /tmp/g1-ik-clearance/native

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-strict.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/test-clearance-strict.o
g++ /tmp/g1-ik-clearance/native/test-clearance-strict.o \
  /tmp/g1-ik-clearance/native/g1-clearance-strict.o \
  -o /tmp/g1-ik-clearance/native/test-clearance-strict
/tmp/g1-ik-clearance/native/test-clearance-strict

g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  -c tests/cpp/test_g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/test-clearance-release-caller.o
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-release-kernel.o
g++ /tmp/g1-ik-clearance/native/test-clearance-release-caller.o \
  /tmp/g1-ik-clearance/native/g1-clearance-release-kernel.o \
  -o /tmp/g1-ik-clearance/native/test-clearance-release
/tmp/g1-ik-clearance/native/test-clearance-release

g++ -std=c++17 -O1 -g -fno-fast-math -ffp-contract=off \
  -frounding-math \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-san.o
g++ -std=c++17 -O1 -g -fno-fast-math \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c tests/cpp/test_g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/test-clearance-san.o
g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all \
  /tmp/g1-ik-clearance/native/test-clearance-san.o \
  /tmp/g1-ik-clearance/native/g1-clearance-san.o \
  -o /tmp/g1-ik-clearance/native/test-clearance-san
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/g1-ik-clearance/native/test-clearance-san

if g++ -std=c++17 -O3 -ffast-math -I. -c g1_clearance.cpp \
    -o /tmp/g1-ik-clearance/native/forbidden-fast-kernel.o \
    2>/tmp/g1-ik-clearance/native/forbidden-fast-kernel.err; then
  echo "certified kernel unexpectedly accepted fast math" >&2
  exit 1
fi
rg -n "fast math" \
  /tmp/g1-ik-clearance/native/forbidden-fast-kernel.err

python - <<'PY'
import re
from pathlib import Path

source_paths = sorted(
    path
    for suffix in ("*.h", "*.hpp", "*.c", "*.cc", "*.cpp", "*.cxx")
    for path in Path(".").rglob(suffix)
    if ".git" not in path.parts
)
sources = {
    path.as_posix(): path.read_text(encoding="utf-8")
    for path in source_paths
}
definitions = {
    "G1SurfaceQueryStatus": r"\benum\s+G1SurfaceQueryStatus\s*\{",
    "G1SurfaceSample": r"\bstruct\s+G1SurfaceSample\s*\{",
    "g1_ik_float_is_runtime_value":
        r"\bstatic\s+inline\s+bool\s+g1_ik_float_is_runtime_value\s*\(",
    "g1_ik_vec3_is_runtime_value":
        r"\bstatic\s+inline\s+bool\s+g1_ik_vec3_is_runtime_value\s*\(",
    "g1_ik_surface_normal_is_valid":
        r"\bstatic\s+inline\s+bool\s+g1_ik_surface_normal_is_valid\s*\(",
    "g1_surface_query_v2":
        r"\bstatic\s+inline\s+G1SurfaceQueryStatus\s+g1_surface_query_v2\s*\(",
}
for symbol, pattern in definitions.items():
    owners = [
        (path, len(re.findall(pattern, text)))
        for path, text in sources.items()
        if re.search(pattern, text)
    ]
    assert owners == [("g1_surface_query.h", 1)], (symbol, owners)
print("VALID one G1 surface-query definition source")
PY
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -H -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-include-guard.o \
  2>/tmp/g1-ik-clearance/native/g1-clearance-include-guard.txt
rg 'g1_surface_query\.h' \
  /tmp/g1-ik-clearance/native/g1-clearance-include-guard.txt
! rg '(g1_ik|g1_skeleton|database)\.h' \
  /tmp/g1-ik-clearance/native/g1-clearance-include-guard.txt
nm -C -g --defined-only \
  /tmp/g1-ik-clearance/native/g1-clearance-include-guard.o \
  > /tmp/g1-ik-clearance/native/g1-clearance-symbols.txt
! rg 'database_|forward_kinematics|motion_matching_search|compute_(bone|trajectory)|normalize_feature|denormalize_features' \
  /tmp/g1-ik-clearance/native/g1-clearance-symbols.txt
/tmp/g1-ik-clearance/native/test-clearance-strict --query-parity \
  > /tmp/g1-ik-clearance/native/g1-surface-query-strict.txt
/tmp/g1-ik-clearance/native/test-clearance-release --query-parity \
  > /tmp/g1-ik-clearance/native/g1-surface-query-fast.txt
cmp /tmp/g1-ik-clearance/native/g1-surface-query-strict.txt \
  /tmp/g1-ik-clearance/native/g1-surface-query-fast.txt

! rg -n 'ceil\(|radial_steps|segment_steps|half.*cell.*sample' \
  g1_clearance.cpp g1_clearance.h
! rg -n 'ordered.*lift|sphere.*\+.*lift|required_lift_m' \
  g1_clearance.cpp g1_clearance.h
git diff --check
```

The query-parity mode emits, for each locked valid/outside/invalid fixture, the
exact status plus raw height/normal bits or unchanged seeded output bits.
Expected: all six query definitions have exactly one source owner; the strict
kernel includes `g1_surface_query.h` but not `g1_ik.h`, `g1_skeleton.h`, or
`database.h`, and exports no database/FK/search/feature symbols; strict and fast
query records match byte-for-byte; strict, release-caller, and sanitizer
executables exit zero; the fast kernel compile fails at its guard; identical
endpoint-bit parity records match byte-for-byte; all environment mutations are
restored; and no sampled or predicted-lift production path remains.

After Task 6 creates the runtime, compile `tests/cpp/test_g1_ik.cpp` with
`-DG1_IK_ENABLE_TEST_SEAMS` once as a strict caller and once as a fast-math
caller, link both to a separately compiled strict `g1_clearance.cpp` object, and
run the staged fixture in both builds. Require real contiguous calls, first-pass
selection, selected-pose reuse with no extra solve, exactly 41 calls on all-fail,
rollback/safe-stop, and deterministic repeated-run diagnostics. A compile
without the macro that names the seam must fail, and production `nm -C` output
must not contain the seam symbol.

- [ ] **Step 7: Commit certified clearance and physical diagnostics**

Task 5 owns two ordered commits. The move-only query-ownership commit from
Step 3 already owns `g1_surface_query.h`, `g1_ik.h`, and its initial parity-test
changes and must not be amended, folded, or repeated here. Stage only the
subsequent certified geometry and diagnostic changes for the second commit:

```bash
git add g1_clearance.h g1_clearance.cpp tests/cpp/test_g1_clearance.cpp
git commit -m "feat: certify G1 leg clearance over terrain"
```

### Task 6: Compose a Reversible Per-Frame IK Observe/Apply Transaction

**Files:**
- Create: `g1_ik_runtime.h`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Produces `G1IkState`, `G1FootFrameResult`, `G1IkFrameResult`,
  `g1_ik_state_reset`, and `g1_ik_frame_evaluate`.
- `G1FootFrameResult` owns `G1SwingSelectionDiagnostic swing_selection` plus
  the real position/orientation results; it has no legacy planner struct,
  continuous required-lift field, or predicted margin. Its aggregate rejection
  mask is the sole retained evidence about candidate-local certification
  statuses from discarded stages.
- `g1_ik_frame_evaluate` stages from an immutable baseline, evaluates the Task 5
  ladder through one private real stage function, and mutates only its
  caller-provided scratch `G1IkState` after every selected endpoint bit,
  defensive certificate, controller bound, and checked history operation passes.
  The outer controller still decides whether to commit that scratch state.
- Every clearance call passes an immutable factory-or-tighter
  `G1ClearanceBudget`, stores the returned `G1ClearanceStatus`, and accepts
  evidence only when `status == G1ClearanceOk`. Status values are never used as
  booleans or inferred from error text.
- With `apply_enabled=false`, output quaternions remain a byte copy of the input;
  observation diagnostics may advance only in scratch state. Matching, support,
  traversal, simulation, local positions, and scene state remain outside this
  API.

- [ ] **Step 1: Write transactional staging and IK-off REDs**

In `tests/cpp/test_g1_ik.cpp`, retain the named-bone isolation, exact 25 Hz,
reach, correction, orientation, and IK-off byte-equality fixtures from Tasks
1--4. Add the guarded real-stage fixtures required by Task 5 Step 1:

- derive/probe all 41 candidates from one immutable input through
  `g1_ik_stage_swing_candidate_for_test`;
- require the full selector to stop at the first real pass, copy the exact
  selected lift/command/endpoint/margin/work bits, and perform no second IK solve;
- require a wall to execute exactly 41 stages, safe-stop with
  `G1SwingNoCandidate`, preserve the exact aggregate status mask, and preserve
  the input pose/state/histories;
- table-drive `Ok`, `OutsideDomain`, `BudgetExceeded`, `Uncertified`,
  `InvalidInput`, `InvalidField`, and `ArithmeticFailure`. Only the three finite
  candidate-local statuses continue; the last three abort transactionally;
- seed reset/commit outputs and prove checked `g1_swing_history_reset` and
  `g1_swing_history_commit` failures leave them unchanged.

- [ ] **Step 2: Compile the runtime RED without violating the strict boundary**

Task 5's kernel already exists. Compile it strictly first, then compile the
missing runtime caller. The caller compile is the expected RED; do not include
`g1_clearance.cpp` into the caller translation unit:

```bash
mkdir -p /tmp/g1-ik-clearance/task6-red
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task6-red/g1-clearance.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -DG1_IK_ENABLE_TEST_SEAMS -I. \
  -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-clearance/task6-red/test-g1-ik.o
```

Expected: the strict kernel object builds; the caller fails because
`g1_ik_runtime.h` or its staging declarations do not exist.

- [ ] **Step 3: Define fixed-size state, diagnostics, and checked reset**

Create `g1_ik_runtime.h` including `g1_clearance.h`. Define the existing stop
reason enum and these ownership changes:

```cpp
struct G1FootIkState
{
    G1FootLockState lock;
    G1SwingHistory swing;
};

struct G1IkState
{
    bool initialized = false;
    G1FootIkState feet[2];
};

struct G1FootFrameResult
{
    bool recorded_contact = false;
    G1FootTarget target;
    G1SwingSelectionDiagnostic swing_selection;
    G1SwingClearanceValidation defensive_swing;
    G1LegSolveResult position;
    G1FootOrientationResult orientation;
};

struct G1IkFrameResult
{
    bool applied = false;
    bool safe_stop_requested = false;
    G1IkStopReason stop_reason = G1IkStopNone;
    float max_correction_radians = 0.0f;
    G1FootFrameResult feet[2];
};
```

`g1_ik_state_reset` validates shapes and configurations into a local candidate,
computes actual FK sphere centers, and calls checked history reset explicitly:

```cpp
if (!g1_swing_history_reset(
        candidate.feet[foot].swing, sphere_centers,
        error, error_capacity)) {
    return false;
}
```

Assign `output = candidate` only after both feet pass. No unchecked history
operation or partially initialized state is observable.

- [ ] **Step 4: Implement the one real staged transaction**

Implement one private `g1_ik_stage_swing_candidate` and the guarded diagnostic
wrapper exactly as Task 5 Step 4 and reviewed design Section 9 specify. The full
selector and wrapper both call that private function; neither duplicates a
solve or clearance path.

`g1_ik_frame_evaluate` starts with local pose/state/frame candidates. For each
foot it updates the lock observer, bypasses the ladder only for recorded
contact, and otherwise evaluates indices `0..40`. Each stage:

1. materializes the actual desired sole command through strict
   `g1_apply_swing_lift_y`;
2. runs named bounded position IK, foot orientation, controller constraints, and
   checked FK in production order;
3. records actual sphere bits and calls `g1_swing_clearance_validate` with
   `const G1ClearanceBudget swing_limits = g1_swing_foot_clearance_budget();`;
4. branches on the explicit status enum:

```cpp
switch (status) {
case G1ClearanceOk:
    candidate_passes =
        controller_constraints_passed &&
        validation.lower_margin_m >= 0.0;
    break;
case G1ClearanceOutsideDomain:
    clearance_rejection_status_mask |= G1SwingRejectOutsideDomainBit;
    candidate_passes = false;
    break;
case G1ClearanceBudgetExceeded:
    clearance_rejection_status_mask |= G1SwingRejectBudgetExceededBit;
    candidate_passes = false;
    break;
case G1ClearanceUncertified:
    clearance_rejection_status_mask |= G1SwingRejectUncertifiedBit;
    candidate_passes = false;
    break;
case G1ClearanceInvalidInput:
case G1ClearanceInvalidField:
case G1ClearanceArithmeticFailure:
    return g1_ik_error(
        error, error_capacity,
        "G1 staged swing clearance failed with status %u",
        static_cast<unsigned>(status));
}
```

Validate after every OR that no bit outside `G1SwingRejectKnownMask` is set.
Assign `selection.clearance_rejection_status_mask` only with the otherwise
transactional public selection diagnostic: selected and all-41-fail results
retain the aggregate, recorded-contact remains zero, and a global abort leaves
the caller's prior diagnostic unchanged. The guarded per-candidate seam exposes
only that candidate's existing status; tests derive the aggregate externally
and compare it with the full selector, so no rejected pose or margin is added
to the seam.

Move the first passing staged pose into outer scratch without rerunning IK.
Run both foot selectors so their real diagnostics are complete. If either foot
has all-41 finite rejection, return success with the immutable baseline pose,
unchanged scratch state/histories, and the completed frame safe-stop diagnostic
including both masks; this is a diagnostic transaction, not a controlled
error. Only when both feet select or bypass for recorded contact does the
runtime compose their staged poses, run checked FK once, compare selected
endpoint bits, and call the strict validator again with a fresh immutable swing
budget. Require explicit `status == G1ClearanceOk` and
`lower_margin_m >= 0.0`.

Only after all final checks pass, call checked history commit into the local
next state:

```cpp
if (!g1_swing_history_commit(
        next.feet[foot].swing, final_centers,
        error, error_capacity)) {
    return false;
}
```

Then assign output pose, scratch state, and frame result together. Any returned
error leaves every caller-owned output unchanged. A successful finite safe-stop
assigns only its completed frame diagnostic while preserving baseline pose and
accepted state/history as described above. The outer controller may discard a
successful staged pose transaction if its complete pose-capsule thresholds
reject it.

After creating the runtime, run the source contract scan here—not in Task 5:

```bash
test -f g1_ik_runtime.h
rg -n 'G1SwingLiftCandidateBits|G1SwingSelectionDiagnostic|clearance_rejection_status_mask|actual_sphere_center_bits|g1_ik_stage_swing_candidate_for_test' \
  g1_ik_runtime.h tests/cpp/test_g1_ik.cpp
! rg -n 'g1_swing_clearance_plan|required_lift_m|corrected_margin_m|actual_corrected_margin_m|sphere.*\+.*lift' \
  g1_ik_runtime.h tests/cpp/test_g1_ik.cpp
```

- [ ] **Step 5: Run strict, release-caller, and sanitizer GREEN**

Every mode compiles `g1_clearance.cpp` separately with strict FP and links with a
driver command that does not contain `-ffast-math`:

```bash
mkdir -p /tmp/g1-ik-clearance/task6

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task6/g1-clearance-strict.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -DG1_IK_ENABLE_TEST_SEAMS -I. \
  -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-clearance/task6/test-g1-ik-strict.o
g++ /tmp/g1-ik-clearance/task6/test-g1-ik-strict.o \
  /tmp/g1-ik-clearance/task6/g1-clearance-strict.o \
  -o /tmp/g1-ik-clearance/task6/test-g1-ik-strict
/tmp/g1-ik-clearance/task6/test-g1-ik-strict

g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task6/g1-clearance-release.o
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -DG1_IK_ENABLE_TEST_SEAMS -I. -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-clearance/task6/test-g1-ik-release.o
g++ /tmp/g1-ik-clearance/task6/test-g1-ik-release.o \
  /tmp/g1-ik-clearance/task6/g1-clearance-release.o \
  -o /tmp/g1-ik-clearance/task6/test-g1-ik-release
/tmp/g1-ik-clearance/task6/test-g1-ik-release

g++ -std=c++17 -O1 -g -fno-fast-math -ffp-contract=off \
  -frounding-math \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task6/g1-clearance-san.o
g++ -std=c++17 -O1 -g -fno-fast-math \
  -DG1_IK_ENABLE_TEST_SEAMS \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c tests/cpp/test_g1_ik.cpp \
  -o /tmp/g1-ik-clearance/task6/test-g1-ik-san.o
g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all \
  /tmp/g1-ik-clearance/task6/test-g1-ik-san.o \
  /tmp/g1-ik-clearance/task6/g1-clearance-san.o \
  -o /tmp/g1-ik-clearance/task6/test-g1-ik-san
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  /tmp/g1-ik-clearance/task6/test-g1-ik-san
```

Expected: every executable exits zero; strict/release supplied-endpoint parity
records are byte-identical; each build deterministically selects its own first
actual pass; the all-fail fixture makes exactly 41 calls and rolls back; no
sanitizer finding appears. The macro-free negative compile and production
symbol scan also pass.

- [ ] **Step 6: Commit the reversible frame transaction**

```bash
git add g1_ik_runtime.h tests/cpp/test_g1_ik.cpp
git commit -m "feat: compose certified G1 terrain IK frames"
```

### Task 7: Integrate IK State, Safe-Stop Handoff, and Controlled Cleanup

**Files:**
- Modify: `g1_controller_state.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Extends `g1_controller_state` with resettable accepted `G1IkState`,
  last-safe local/global pose arrays, `G1IkFrameResult`, accepted and rejected
  `G1PoseClearance`, exact `G1ClearanceStatus ik_candidate_clearance_status`, a
  rejection flag, and a latched safe-stop request.
- A frame evaluates through a local `G1IkState ik_candidate_state`. The
  controller commits that state only with the accepted pose and binary64 pose
  certificate; discarding a candidate therefore also discards its locks and
  checked swing-history commits.
- Every pose measurement passes
  `const G1ClearanceBudget pose_limits = g1_pose_clearance_budget();` and stores
  the exact `G1ClearanceStatus`. `Ok` may be threshold-tested;
  `OutsideDomain`/`BudgetExceeded`/`Uncertified` are finite candidate
  rejections; `InvalidInput`/`InvalidField`/`ArithmeticFailure` are controlled
  errors followed by normal cleanup.
- Any finite unsafe outcome retains the last-safe rendered pose and accepted
  clearance, latches safe stop, and cancels XZ command/inertia at the next
  existing traversability boundary. It never mutates matching, support, terrain,
  route, or Y simulation state.
- `MM_IK` accepts exactly `0` or `1` and defaults to `0`.

- [ ] **Step 1: Extend reset, status, rollback, and option tests first**

Poison every accepted IK field, pose array, clearance result, rejection flag,
and latch before the controller-state reset fixture. After reset require:

- accepted/scratch pose arrays equal the support-retargeted baseline;
- both checked histories are initialized from actual FK centers;
- `ik_clearance.minimum.lower_bound_m` is finite and at least `-0.01`;
- accepted state/pose/clearance remain bit-identical when any reset clearance
  status is non-`Ok`;
- `MM_IK` exact parsing and the latched XZ handoff behavior remain as previously
  specified.

Add controller transaction fixtures for all seven clearance statuses. Require
the three finite candidate-local failures to reject and latch without changing
accepted state, while the three global failures request controlled exit without
changing it. Seed a successful `G1ClearanceResult` whose lower bound cannot be
represented upward safely as float and prove the controller compares the
original binary64 bits.

- [ ] **Step 2: Compile controller-state RED with a separate strict kernel**

```bash
mkdir -p /tmp/g1-ik-clearance/task7-red
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task7-red/g1-clearance.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  -c tests/cpp/test_g1_controller_state.cpp \
  -o /tmp/g1-ik-clearance/task7-red/test-controller-state.o
```

Expected: the strict kernel builds and the caller fails because the new
controller-state ownership/status fields are absent. Do not use a one-command
compile that could absorb the strict implementation into caller flags.

- [ ] **Step 3: Add exact option parsing and next-update handoff**

Keep `g1_parse_ik_enabled` and `g1_ik_safe_stop_handoff` from the prior plan.
The former accepts only null/`0`/`1`. The latter returns an exact zero desired
command and `cancel_planar_inertia=true` only for a latched stop; it never
changes Y state.

- [ ] **Step 4: Replace dormant contact state and reset with explicit status**

Include `g1_ik_runtime.h` from `g1_controller_state.h`. Replace the dormant
generic contact arrays with accepted IK state, accepted/scratch local/global
pose arrays, `G1IkFrameResult`, two `G1PoseClearance` values, rejection, and
the candidate status/latch fields. Swap each owner transactionally.

Inside the local reset candidate, build baseline FK with
`g1_ik_checked_forward_kinematics`, call checked
`g1_ik_state_reset`, then measure the pose with an immutable budget and explicit
status:

```cpp
const G1ClearanceBudget pose_limits = g1_pose_clearance_budget();
G1PoseClearance reset_clearance = {};
const G1ClearanceStatus reset_status = g1_measure_pose_clearance(
    reset_clearance,
    pose_limits,
    scene.terrain,
    s.ik_global_bone_positions,
    s.ik_global_bone_rotations,
    error,
    capacity);
if (reset_status != G1ClearanceOk) {
    return scene_error(
        error, capacity,
        "controller reset: scene '%s' initial G1 clearance status=%u",
        scene.metadata.id.c_str(),
        static_cast<unsigned>(reset_status));
}
if (reset_clearance.minimum.lower_bound_m < -0.01) {
    return scene_error(
        error, capacity,
        "controller reset: scene '%s' initial G1 pose is unsafe",
        scene.metadata.id.c_str());
}
s.ik_clearance = reset_clearance;
s.ik_candidate_clearance = reset_clearance;
s.ik_candidate_clearance_status = G1ClearanceOk;
s.ik_candidate_rejected = false;
s.ik_safe_stop_latched = false;
```

The reset candidate is assigned only after every step succeeds. No
`G1ClearanceStatus` is converted to bool, no budget is mutated, and no safety
bound is rounded to float.

- [ ] **Step 5: Parse IK before Raylib and remove inherited solver code**

Parse `MM_IK` beside the other validated options before opening a window or log.
Delete file-scope generic `contact_reset`/`contact_update`/`ik_look_at`/
`ik_two_bone` and the hard-coded toe-end branch. `g1_ik.h`,
`g1_ik_runtime.h`, and strict `g1_clearance.cpp` are the only active IK and
clearance implementation.

- [ ] **Step 6: Stage after support FK and commit pose, state, and clearance together**

After support retargeting and baseline FK, create scratch state and call the
runtime:

```cpp
G1IkState ik_candidate_state = state.ik;
state.ik_candidate_bone_positions = state.adjusted_bone_positions;
if (!g1_ik_frame_evaluate(
        state.ik_candidate_bone_rotations,
        ik_candidate_state,
        state.adjusted_bone_positions,
        state.adjusted_bone_rotations,
        db.bone_parents,
        state.curr_bone_contacts,
        active_scene.terrain,
        ik_enabled,
        dt,
        state.ik_frame,
        artifact_error,
        sizeof(artifact_error))) {
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}
if (!g1_ik_checked_forward_kinematics(
    state.ik_candidate_global_bone_positions,
    state.ik_candidate_global_bone_rotations,
    state.ik_candidate_bone_positions,
    state.ik_candidate_bone_rotations,
    db.bone_parents,
    artifact_error,
    sizeof(artifact_error))) {
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}

const G1ClearanceBudget pose_limits = g1_pose_clearance_budget();
G1PoseClearance measured_candidate = {};
const G1ClearanceStatus pose_status = g1_measure_pose_clearance(
    measured_candidate,
    pose_limits,
    active_scene.terrain,
    state.ik_candidate_global_bone_positions,
    state.ik_candidate_global_bone_rotations,
    artifact_error,
    sizeof(artifact_error));
state.ik_candidate_clearance_status = pose_status;

bool finite_clearance_rejection = false;
switch (pose_status) {
case G1ClearanceOk:
    state.ik_candidate_clearance = measured_candidate;
    break;
case G1ClearanceOutsideDomain:
case G1ClearanceBudgetExceeded:
case G1ClearanceUncertified:
    finite_clearance_rejection = true;
    break;
case G1ClearanceInvalidInput:
case G1ClearanceInvalidField:
case G1ClearanceArithmeticFailure:
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}

const bool planted_penetration =
    pose_status == G1ClearanceOk &&
    ((state.ik_frame.feet[0].target.locked &&
      (measured_candidate.left.toe.lower_bound_m < -0.005 ||
       measured_candidate.left.foot.lower_bound_m < -0.005)) ||
     (state.ik_frame.feet[1].target.locked &&
      (measured_candidate.right.toe.lower_bound_m < -0.005 ||
       measured_candidate.right.foot.lower_bound_m < -0.005)));
const bool physical_penetration =
    pose_status == G1ClearanceOk &&
    measured_candidate.minimum.lower_bound_m < -0.01;

if (ik_enabled &&
    (finite_clearance_rejection ||
     planted_penetration ||
     physical_penetration)) {
    g1_ik_request_stop(state.ik_frame, G1IkStopPostSolveClearance);
}
state.ik_candidate_rejected =
    ik_enabled && state.ik_frame.safe_stop_requested;

if (!state.ik_candidate_rejected && pose_status == G1ClearanceOk) {
    state.ik = ik_candidate_state;
    state.ik_bone_positions = state.ik_candidate_bone_positions;
    state.ik_bone_rotations = state.ik_candidate_bone_rotations;
    state.ik_global_bone_positions = state.ik_candidate_global_bone_positions;
    state.ik_global_bone_rotations = state.ik_candidate_global_bone_rotations;
    state.ik_clearance = measured_candidate;
}
if (state.ik_candidate_rejected) {
    state.ik_safe_stop_latched = true;
}
```

The runtime's checked history commits exist only inside
`ik_candidate_state`; there is no second unchecked history loop in the
controller. Candidate diagnostics remain separate rejection evidence.
IK-off commits the byte-identical support-retargeted pose and its explicit
`Ok` certificate.

- [ ] **Step 7: Hand the latch to the existing traversability boundary**

At the next update, call `g1_ik_safe_stop_handoff` before
`traversability_limit_command`. If `cancel_planar_inertia` is true, set only X/Z
simulation velocity and acceleration to exact zero before prediction. Keep the
user/route command for logging, preserve both Y components, and retain the
sibling hard XZ clip. The latch clears only on scene reset or explicit live IK
reset.

- [ ] **Step 8: Keep the live toggle reversible**

The existing checkbox edge calls checked `g1_ik_state_reset` from the currently
accepted `state.ik_global_bone_positions/rotations`, handles failure as a
controlled exit, and clears the latch only on success. False restores exact
IK-off output on the next update; true begins with no stale lock/history.

- [ ] **Step 9: Run focused tests and deterministic controller smoke**

Compile every clearance-consuming executable as separate objects. Caller flags
may differ; kernel and final link remain strict:

```bash
mkdir -p /tmp/g1-ik-clearance/task7

g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task7/g1-clearance-test.o
for name in test_g1_controller_state test_g1_ik test_g1_clearance; do
  extra=()
  if test "$name" = test_g1_ik; then
    extra=(-DG1_IK_ENABLE_TEST_SEAMS)
  fi
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "${extra[@]}" -c "tests/cpp/${name}.cpp" \
    -o "/tmp/g1-ik-clearance/task7/${name}.o"
  g++ "/tmp/g1-ik-clearance/task7/${name}.o" \
    /tmp/g1-ik-clearance/task7/g1-clearance-test.o \
    -o "/tmp/g1-ik-clearance/task7/${name}"
  "/tmp/g1-ik-clearance/task7/${name}"
done

g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/task7/g1-clearance-controller.o
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  -c controller.cpp \
  -o /tmp/g1-ik-clearance/task7/controller.o
g++ /tmp/g1-ik-clearance/task7/controller.o \
  /tmp/g1-ik-clearance/task7/g1-clearance-controller.o \
  -o /tmp/controller_g1_ik \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

mkdir -p /tmp/g1-ik-clearance
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=grail-curb-low MM_TEST_ROUTE=curb-forward \
  MM_TEST_MODE=route MM_TEST_FRAMES=375 MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_LOG=/tmp/g1-ik-clearance/smoke-off.csv \
  /tmp/controller_g1_ik
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=grail-curb-low MM_TEST_ROUTE=curb-forward \
  MM_TEST_MODE=route MM_TEST_FRAMES=375 MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  MM_LOG=/tmp/g1-ik-clearance/smoke-on.csv \
  /tmp/controller_g1_ik
```

Expected: tests and both 375-frame runs exit zero; finite clearance rejection
safe-stops transactionally; global failures take controlled cleanup; no
correction exceeds `0.35`; accepted safety decisions use binary64 lower bounds.
The final link command contains no `-ffast-math`.

- [ ] **Step 10: Commit controller integration only**

```bash
git add g1_controller_state.h controller.cpp \
  tests/cpp/test_g1_controller_state.cpp tests/cpp/test_g1_ik.cpp
git commit -m "feat: integrate certified G1 terrain IK"
```

### Task 8: Append Gate E Diagnostics and Prove IK Invariance

**Files:**
- Modify: `motion_match_log.h`
- Modify: `controller.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`

**Interfaces:**
- Appends the exact IK suffix below without renaming/reordering sibling columns:

```text
ik_applied,ik_safe_stop_requested,ik_stop_reason,max_ik_correction,
actual_simulation_speed,ik_candidate_rejected,ik_candidate_clearance_status,
left_candidate_toe_clearance,left_candidate_foot_clearance,
right_candidate_toe_clearance,right_candidate_foot_clearance,
ik_candidate_minimum_clearance,
left_recorded_contact,right_recorded_contact,left_locked,right_locked,
left_observed_lock_drift,right_observed_lock_drift,
left_lock_drift,right_lock_drift,
left_sole_normal_alignment,right_sole_normal_alignment,
left_contact_residual,right_contact_residual,
left_target_height,right_target_height,
left_target_normal_x,left_target_normal_y,left_target_normal_z,
right_target_normal_x,right_target_normal_y,right_target_normal_z,
left_swing_candidates_evaluated,left_swing_selected_index,
left_swing_selected_lift_bits,left_swing_materialized_command_y_bits,
left_swing_clearance_rejection_status_mask,
left_swing_lower_margin,left_swing_witness_upper_margin,
right_swing_candidates_evaluated,right_swing_selected_index,
right_swing_selected_lift_bits,right_swing_materialized_command_y_bits,
right_swing_clearance_rejection_status_mask,
right_swing_lower_margin,right_swing_witness_upper_margin,
left_reachable,right_reachable,left_knee_clearance,left_ankle_clearance,
left_toe_clearance,left_foot_clearance,left_shin_clearance,
left_thigh_clearance,right_knee_clearance,right_ankle_clearance,
right_toe_clearance,right_foot_clearance,right_shin_clearance,
right_thigh_clearance,ik_hips_clearance,ik_minimum_clearance
```

- Candidate and accepted physical clearances plus swing lower/witness margins are
  binary64 from `G1ClearanceResult`/`G1SwingClearanceValidation` and are written
  with `%.17g`. No float member, cast, `%.9g` formatting, or upward float
  conversion may feed a safety decision or log value.
- `ik_candidate_clearance_status` is the exact unsigned enum value for the candidate pose
  measurement. A certified accepted row uses `Ok`. Finite rejected rows retain
  the exact `OutsideDomain`/`BudgetExceeded`/`Uncertified` evidence; global
  statuses take controlled cleanup and do not produce a successful row.
- Swing diagnostics describe the immutable ladder, not a continuous lift:
  evaluated count, selected index, raw lift bits, strict materialized command-Y
  bits, aggregate candidate-local rejection-status mask, and binary64
  actual-center certificate bounds. No
  `required_lift_m`/planned translated-sphere margin is logged.
- Ordinary physical fields always describe the last accepted/rendered pose.
  Candidate fields preserve the scratch `Ok` lower bounds only when such a pose
  exists; all-fail ladder diagnostics use the no-candidate sentinel instead of
  fabricating candidate clearance.
- Existing matching, support, route, simulation, lock-drift, target-normal,
  residual, reach, and Gate E invariance meanings remain unchanged.

- [ ] **Step 1: Write failing schema, binary64, ladder, and invariance tests**

Extend the valid-row helper with the exact suffix. Retain the paired matching,
support-root, contact/lock/target, sole-alignment, physical-clearance, residual,
correction, and drift fixtures. Add these migrations:

- reject every removed `*_swing_required_lift`, `*_swing_applied_lift`,
  `*_swing_planned_margin`, and `*_swing_actual_margin` column;
- certified non-contact rows require
  `1 <= candidates_evaluated <= 41`,
  `selected_index != G1SwingNoCandidate`,
  `candidates_evaluated == selected_index + 1`, exact selected ladder bits, and
  `lower_margin >= 0.0`; their mask may contain any subset of the three known
  bits from earlier real rejects;
- recorded-contact rows require zero evaluated candidates and the no-candidate
  sentinel with mask zero; all-41-fail rows require exactly 41, the sentinel,
  and safe stop. Unresolved-certification evidence requires a nonzero mask,
  while negative-`Ok`/controller-only exhaustion may have mask zero;
- reject unknown mask bits and mutate each known bit plus every mixed subset;
- mutate every pose status and require only exact `Ok` on accepted certified
  rows;
- use values adjacent to `-0.005`, `-0.01`, and zero whose binary64 decisions
  would change after binary32 rounding; verify CSV `%.17g` round-trips the
  original binary64 value and the checker makes the binary64 decision;
- require candidate and accepted `G1ClearanceResult::lower_bound_m` values to
  match on a non-rejected certified row, while a rejected row preserves the
  last-safe accepted values.

Add explicit REDs for duplicate/missing columns, row-count/frame alignment,
raw-string invariance, unit/upward normals, reachability, and
continuation-cost. Restore the route-level acceptance REDs as well:

- mutate either member of a pair so any row has a scene or route other than the
  exact requested value;
- remove every certified `route_complete=1` row, introduce a blocked or
  safe-stop terminal tail, provide only nine planted samples, and make
  corrected planted drift equal to or greater than baseline drift;
- try an unlisted stress scene/route and mutate either member's scene/route;
- label a stress traversal despite one stop request or without a
  `route_complete=1` row;
- require a class-2 traversal with equal/worse planted drift to remain accepted,
  proving that branch's intentional drift exemption; and
- set the first stress safe-stop request row's `ik_candidate_rejected` to `0`.

Run the full- and exact-legacy-baseline variants wherever the baseline schema
changes which columns supply a predicate.

- [ ] **Step 2: Run the focused checker RED**

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
```

Expected: existing sibling tests remain green; migrated suffix/status/ladder
tests fail until the logger and checker are updated.

- [ ] **Step 3: Append typed row fields without narrowing safety values**

In `motion_match_log.h` add `<cstdint>` and use `uint32_t` for candidate counts/indices/raw bits,
`double` for every clearance or swing margin, and retain `float` only for
non-safety observations such as drift, normal alignment, target height,
residual, and correction. The relevant shape is:

```cpp
struct motion_match_ik_leg_diagnostic
{
    bool recorded_contact = false;
    bool locked = false;
    bool reachable = true;
    float observed_lock_drift = 0.0f;
    float lock_drift = 0.0f;
    float sole_normal_alignment = 1.0f;
    float contact_residual = 0.0f;
    float target_height = 0.0f;
    vec3 target_normal = vec3(0.0f, 1.0f, 0.0f);
    uint32_t swing_candidates_evaluated = 0;
    uint32_t swing_selected_index = UINT32_MAX;
    uint32_t swing_selected_lift_bits = 0;
    uint32_t swing_materialized_command_y_bits = 0;
    uint32_t swing_clearance_rejection_status_mask = 0;
    double swing_lower_margin = 0.0;
    double swing_witness_upper_margin = 0.0;
    double knee_clearance = 0.0;
    double ankle_clearance = 0.0;
    double toe_clearance = 0.0;
    double foot_clearance = 0.0;
    double shin_clearance = 0.0;
    double thigh_clearance = 0.0;
};

struct motion_match_ik_diagnostic
{
    bool applied = false;
    bool safe_stop_requested = false;
    const char* stop_reason = "none";
    float max_correction = 0.0f;
    float actual_simulation_speed = 0.0f;
    bool candidate_rejected = false;
    uint32_t candidate_clearance_status = UINT32_MAX;
    double left_candidate_toe_clearance = 0.0;
    double left_candidate_foot_clearance = 0.0;
    double right_candidate_toe_clearance = 0.0;
    double right_candidate_foot_clearance = 0.0;
    double candidate_minimum_clearance = 0.0;
    motion_match_ik_leg_diagnostic left;
    motion_match_ik_leg_diagnostic right;
    double hips_clearance = 0.0;
    double minimum_clearance = 0.0;
};
```

Append `motion_match_ik_diagnostic ik` to the row. Write enum/raw integer fields
with exact integer formats, binary64 safety fields with `%.17g`, and other
floats with the sibling format. Preserve the single final newline. Add a
writer-level exact-column regression.

- [ ] **Step 4: Map only immutable observations and certified result members**

Before `deterministic_log.write`, map left/right explicitly. Candidate and
accepted clearance use lower bounds directly:

```cpp
const G1LegConfig ik_configs[2] = {
    g1_left_leg_config(), g1_right_leg_config()
};
motion_match_ik_leg_diagnostic* ik_log_feet[2] = {
    &log_row.ik.left, &log_row.ik.right
};
const G1LegClearance* ik_clearance_feet[2] = {
    &state.ik_clearance.left, &state.ik_clearance.right
};
log_row.ik.applied = state.ik_frame.applied;
log_row.ik.safe_stop_requested = state.ik_frame.safe_stop_requested;
log_row.ik.stop_reason = g1_ik_stop_reason_name(state.ik_frame.stop_reason);
log_row.ik.max_correction = state.ik_frame.max_correction_radians;
log_row.ik.actual_simulation_speed = g1_xz_length(
    state.simulation_velocity);
log_row.ik.candidate_rejected = state.ik_candidate_rejected;
log_row.ik.candidate_clearance_status = static_cast<uint32_t>(
    state.ik_candidate_clearance_status);
if (state.ik_candidate_clearance_status == G1ClearanceOk) {
    log_row.ik.left_candidate_toe_clearance =
        state.ik_candidate_clearance.left.toe.lower_bound_m;
    log_row.ik.left_candidate_foot_clearance =
        state.ik_candidate_clearance.left.foot.lower_bound_m;
    log_row.ik.right_candidate_toe_clearance =
        state.ik_candidate_clearance.right.toe.lower_bound_m;
    log_row.ik.right_candidate_foot_clearance =
        state.ik_candidate_clearance.right.foot.lower_bound_m;
    log_row.ik.candidate_minimum_clearance =
        state.ik_candidate_clearance.minimum.lower_bound_m;
}

for (int foot = 0; foot < 2; ++foot) {
    motion_match_ik_leg_diagnostic& dst = *ik_log_feet[foot];
    const G1FootFrameResult& src = state.ik_frame.feet[foot];
    const G1LegClearance& clearance = *ik_clearance_feet[foot];
    const G1SwingSelectionDiagnostic& selection = src.swing_selection;

    dst.recorded_contact = src.recorded_contact;
    dst.locked = src.target.locked;
    dst.reachable = !src.position.applied || src.position.reachable;
    dst.observed_lock_drift = src.target.horizontal_drift_m;
    dst.lock_drift = g1_horizontal_lock_drift(
        ik_configs[foot], state.ik.feet[foot].lock,
        state.ik_global_bone_positions, state.ik_global_bone_rotations);
    dst.sole_normal_alignment = g1_sole_normal_alignment(
        ik_configs[foot], src.target.surface.normal,
        state.ik_global_bone_rotations);
    dst.contact_residual = src.position.applied
        ? src.position.contact_residual_m : 0.0f;
    dst.target_height = src.target.surface.point.y;
    dst.target_normal = src.target.surface.normal;
    dst.swing_candidates_evaluated = selection.candidates_evaluated;
    dst.swing_selected_index = selection.selected_index;
    dst.swing_clearance_rejection_status_mask =
        selection.clearance_rejection_status_mask;
    if (selection.selected_index != G1SwingNoCandidate) {
        dst.swing_selected_lift_bits = selection.selected.lift_bits;
        dst.swing_materialized_command_y_bits =
            selection.selected.materialized_command_y_bits;
        dst.swing_lower_margin = selection.selected.lower_margin_m;
        dst.swing_witness_upper_margin =
            selection.selected.witness_upper_margin_m;
    }

    dst.knee_clearance = clearance.knee.lower_bound_m;
    dst.ankle_clearance = clearance.ankle.lower_bound_m;
    dst.toe_clearance = clearance.toe.lower_bound_m;
    dst.foot_clearance = clearance.foot.lower_bound_m;
    dst.shin_clearance = clearance.shin.lower_bound_m;
    dst.thigh_clearance = clearance.thigh.lower_bound_m;
}
log_row.ik.hips_clearance = state.ik_clearance.hips.lower_bound_m;
log_row.ik.minimum_clearance = state.ik_clearance.minimum.lower_bound_m;
```

Keep `g1_horizontal_lock_drift` and `g1_sole_normal_alignment` as the two pure
helpers defined by the locked pre-IK logging contract: the former measures the
accepted post-FK sole center against `lock.lock_point` in XZ and returns zero
when unlocked; the latter dots the accepted post-FK configured sole normal
with `src.target.surface.normal`. The sibling row is filled first from its
pre-IK sources. Do not reconstruct a float lift from `selected.lift_bits` for
safety, add a second correction, or rebind a sibling field to post-IK data.
Store the exact candidate pose status in controller state in Task 7; do not
infer it from `candidate_rejected` or error text.

- [ ] **Step 5: Migrate Gate E and stress evidence to statuses and ladder facts**

Implement these exact raw-string groups; none exists before this task:

```python
IK_MATCHING_INVARIANTS = (
    "frame", "fixed_dt", "scene_id", "mode", "route",
    "query_bits_hex", "query_database_frame", "query_range",
    "selected_database_frame", "database_frame", "range",
    "source_range", "searched", "transitioned", "incumbent_cost",
    "selected_cost", "selected_terrain_error", "effective_terrain_weight",
    "terrain0", "terrain1", "terrain2", "terrain3",
    "terrain_point0_x", "terrain_point0_y", "terrain_point0_z",
    "terrain_point1_x", "terrain_point1_y", "terrain_point1_z",
    "terrain_point2_x", "terrain_point2_y", "terrain_point2_z",
    "terrain_point3_x", "terrain_point3_y", "terrain_point3_z",
    "source_name", "source_terrain", "source_index", "continuation_cost",
)

IK_SUPPORT_SIMULATION_INVARIANTS = (
    "raw_selected_hips_y", "inertialized_hips_y", "rendered_hips_y",
    "hips_inertial_offset_y", "runtime_root_surface_height",
    "runtime_left_toe_surface_height", "runtime_right_toe_surface_height",
    "adjustment_xz", "adjustment_y", "clamp_xz", "clamp_y",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled", "source_root_height",
    "runtime_support_root_height", "source_left_toe_height",
    "source_right_toe_height", "runtime_support_left_toe_height",
    "runtime_support_right_toe_height", "support_root_delta",
    "support_left_toe_delta", "support_right_toe_delta", "support_height",
    "support_velocity", "support_source", "airborne_frames",
    "left_contact", "right_contact", "support_retargeted_hips_y",
    "ik_adjusted_hips_y", "simulation_x", "simulation_z",
    "walkability_class", "blocked", "blocked_reason", "blocked_distance",
    "blocked_point_x", "blocked_point_z", "commanded_speed", "applied_speed",
    "route_waypoint", "route_complete", "route_target_height",
    "scene_generation", "scene_frame", "scene_reset_count",
    "scene_switch_failed", "motion_pack_load_count", "model_load_count",
    "model_unload_count", "live_model_count",
)

IK_BASE_INVARIANTS = (
    IK_MATCHING_INVARIANTS + IK_SUPPORT_SIMULATION_INVARIANTS
)

IK_PAIR_OBSERVATION_INVARIANTS = (
    "actual_simulation_speed",
    "left_recorded_contact", "right_recorded_contact",
    "left_locked", "right_locked",
    "left_observed_lock_drift", "right_observed_lock_drift",
    "left_target_height", "right_target_height",
    "left_target_normal_x", "left_target_normal_y", "left_target_normal_z",
    "right_target_normal_x", "right_target_normal_y", "right_target_normal_z",
)
```

Define the exact mask values used by the checker, then freeze the sibling
header before appending this task's suffix:

```python
G1_SWING_REJECT_OUTSIDE_DOMAIN_BIT = 1 << 0
G1_SWING_REJECT_BUDGET_EXCEEDED_BIT = 1 << 1
G1_SWING_REJECT_UNCERTIFIED_BIT = 1 << 2
G1_SWING_REJECT_KNOWN_MASK = (
    G1_SWING_REJECT_OUTSIDE_DOMAIN_BIT |
    G1_SWING_REJECT_BUDGET_EXCEEDED_BIT |
    G1_SWING_REJECT_UNCERTIFIED_BIT
)
G1_SWING_REJECT_NAMES = (
    (G1_SWING_REJECT_OUTSIDE_DOMAIN_BIT, "OutsideDomain"),
    (G1_SWING_REJECT_BUDGET_EXCEEDED_BIT, "BudgetExceeded"),
    (G1_SWING_REJECT_UNCERTIFIED_BIT, "Uncertified"),
)

LEGACY_RUNTIME_COLUMNS = tuple(RUNTIME_COLUMNS)
FULL_RUNTIME_COLUMNS = LEGACY_RUNTIME_COLUMNS + IK_SUFFIX
```

Refactor the existing duplicate-header reader into
`_read_rows_exact(path, expected_columns)`. It requires a nonempty CSV whose
header tuple is **exactly** `expected_columns`: same order, no duplicate,
missing, extra, or partially appended suffix column. Define it and its wrappers
with raw `csv.reader` strings:

```python
def _peek_header(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return tuple(next(csv.reader(stream), ()))

def _read_rows_exact(path, expected_columns):
    if expected_columns == FULL_RUNTIME_COLUMNS:
        schema_name = "FULL_RUNTIME_COLUMNS"
    elif expected_columns == LEGACY_RUNTIME_COLUMNS:
        schema_name = "LEGACY_RUNTIME_COLUMNS"
    else:
        raise AssertionError("unknown exact runtime schema")
    header = _peek_header(path)
    if not header:
        raise ValueError(f"{path}: CSV header is empty")
    if header != expected_columns:
        raise ValueError(
            f"{path}: actual header {header!r} is not exact "
            f"{schema_name} {expected_columns!r}")
    rows = []
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        for csv_row, values in enumerate(reader, 2):
            if len(values) != len(expected_columns):
                raise ValueError(
                    f"{path}: CSV row {csv_row} has {len(values)} values; "
                    f"exact {schema_name} requires {len(expected_columns)}")
            rows.append(dict(zip(expected_columns, values)))
    if not rows:
        raise ValueError(f"{path}: exact {schema_name} CSV has no data rows")
    return rows

def read_rows(path):
    return _read_rows_exact(path, FULL_RUNTIME_COLUMNS)

def read_legacy_runtime_rows(path):
    return _read_rows_exact(path, LEGACY_RUNTIME_COLUMNS)

def read_ik_baseline(path):
    header = _peek_header(path)
    if header == FULL_RUNTIME_COLUMNS:
        return "full", read_rows(path)
    if header == LEGACY_RUNTIME_COLUMNS:
        return "legacy", read_legacy_runtime_rows(path)
    raise ValueError(
        f"{path}: IK baseline header is neither exact full schema nor "
        "exact locked pre-IK prefix")
```

`read_rows` is the only full new-schema reader. `read_legacy_runtime_rows` is
the only legacy-prefix reader and accepts no suffix column. The existing
`--compare-control` and `--ik-off-reference` inputs use the legacy reader.
`--ik-baseline` uses `read_ik_baseline`: a newly generated IK-off baseline is
full schema; an accepted pre-Task8 baseline may be exactly the locked legacy
prefix. After either dispatch, require every baseline row to have raw
`ik_enabled == "0"`; otherwise fail with
`--ik-baseline must contain only ik_enabled=0`. Never accept an arbitrary
intersection or silently drop columns.

`compare_ik_invariance` requires equal nonzero row counts and compares every
`IK_BASE_INVARIANTS` value with ordinary string equality. Its error names the
zero-based row, logged frame, column, baseline raw text, and IK raw text.
`compare_ik_observations` has the same contract for
`IK_PAIR_OBSERVATION_INVARIANTS`, but is called only when both sides use the
full schema. A legacy baseline compares shared base strings only; it can never
stand in for missing post-support observations.

Change shared full-row validation only enough to accept `ik_enabled` exactly
`0` or `1`. Keep explicit `ik_enabled == 0` requirements in
`diagnose_gate_a`, `check_gate_c`, `check_gate_d`, `check_gate_f`,
`check_failed_switch`, `compare_control`, the `--ik-off-reference` path, and
every legacy-prefix validation path. `compare_control` requires both treatment
and control to be IK-off. Gate E and Gate E stress alone require IK-on rows.
Thus adding full-schema parsing cannot weaken any sibling gate or comparison.

Add exact CLI composition:

- `--gate-e` and `--gate-e-stress` are mutually exclusive and each requires
  `--ik-baseline PATH`, `--expected-scene ID`, and `--expected-route ID`;
- missing inputs fail with respectively
  `--gate-e requires --ik-baseline/--expected-scene/--expected-route` or the
  same string beginning `--gate-e-stress`;
- `--ik-off-reference PATH` is independent of Gate E, requires the primary
  full log to contain only `ik_enabled=0`, reads the reference with the exact
  legacy reader, compares `IK_BASE_INVARIANTS`, and prints
  `VALID ik-off-reference frames=<n>`; an IK-on primary fails with
  `--ik-off-reference requires an IK-off primary log`;
- `--compare-control PATH` reads the accepted control with the exact legacy
  reader, requires both logs to be IK-off, and otherwise preserves its existing
  comparison errors and `VALID terrain-comparison` summary; an IK-on primary
  fails with `--compare-control requires an IK-off primary log`;
- neither Gate E flag may combine with `--gate-a`, `--gate-c`, `--gate-d`,
  `--gate-f`, or `--expect-switch-failure`; fail with
  `--gate-e/--gate-e-stress may not combine with sibling gate flags`.
  Conversely, any of `--ik-baseline`, `--expected-scene`, or
  `--expected-route` without one Gate E flag fails with
  `--ik-baseline/--expected-scene/--expected-route requires --gate-e or
  --gate-e-stress`;
- preserve the sibling positional `log`, optional `--compare-control PATH`,
  `--gate-a`, mutually exclusive `--gate-c`/`--gate-d`/`--gate-f`, optional
  `--expected-scenes CSV`, and `--expect-switch-failure` flags. Preserve the
  exact custom errors `--expect-switch-failure may not combine with Gate
  A/C/D/F flags`, `--gate-f requires --expected-scenes`, `--expected-scenes
  must match the locked 14-scene catalog`, and `--expected-scenes requires
  --gate-f`, plus their existing `VALID` summaries. Gate E flags do not change
  those combinations or diagnostics.

For a full `--ik-baseline`, Gate E compares base and observation groups over
the complete pair. For a legacy baseline it compares only the shared base
group. The Gate E stress traverse branch does the same over the complete pair.
The safe-stop branch finds the first request at index `stop`, then compares
only `off_rows[:stop+1]` and `on_rows[:stop+1]`; it must not require any
post-stop row, simulation, support, or observation equality because command
handoff intentionally diverges on the next update.

Add REDs for exact full/legacy headers, every hybrid/partial/duplicate header,
full and legacy `--ik-baseline`, legacy `--ik-off-reference`, all exact CLI
missing-argument diagnostics, sibling-gate IK-on rejection, accepted post-stop
divergence, and rejection of a mutation at the inclusive request row.

The shared validator finite-parses all binary64 text without reformatting it.
Its uint32 parser accepts only canonical raw decimal text matching
`0|[1-9][0-9]*` whose value is at most `4294967295`; it rejects signs,
whitespace, leading zeroes, overflow, and non-digits. Use it for every count,
index, raw-bit word, pose status, and rejection mask. On every full-schema row,
reject `mask & ~G1_SWING_REJECT_KNOWN_MASK != 0` for either foot before any
Gate E branch logic; any recorded-contact foot additionally requires mask zero.

For every certified Gate E row require:

- exact 25 Hz, IK applied, no stop/rejection, candidate status `Ok`, bounded
  correction/residual, reachable legs, exact contact/lock/target observations,
  and locked sole alignment at least `0.999`;
- for a recorded-contact foot: evaluated count zero and no-candidate sentinel;
  its rejection mask must be zero;
  for a swing foot: evaluated count in `[1,41]`, selected index exactly one less
  than the count, selected lift bits equal canonical table entry at that index,
  binary64 lower margin nonnegative with witness upper margin no smaller, and
  rejection mask containing only `G1SwingRejectKnownMask` bits. A selected
  candidate may retain bits from earlier rejected stages;
- candidate `Ok` lower bounds equal the accepted lower bounds on a committed
  row;
- locked toe/foot lower bounds at least `-0.005` and every accepted
  Hips/knee/ankle/toe/foot/thigh/shin/minimum lower bound at least `-0.01`.

`check_gate_e_pair` applies these pair-level predicates after baseline-schema
dispatch:

1. Require every baseline row to be IK-off and every primary row to be IK-on.
   Require every row in **both** inputs to have
   `scene_id == expected_scene` and `route == expected_route`; checking only a
   first row or the set of values is insufficient.
2. Run `check_gate_e_rows(on_rows)` and compare all base invariants for either
   schema. For a full baseline, also compare all observation invariants over the
   full pair. For an exact legacy baseline, omit only comparisons for columns
   that do not exist in the locked prefix.
3. Require at least one `route_complete == "1"` row and forbid a terminal
   blocked or safe-stop tail. `check_gate_e_rows` additionally keeps the
   pre-migration certified rule that no primary row may have
   `ik_safe_stop_requested == "1"`, a stop reason other than `none`, or
   `ik_candidate_rejected == "1"`.
4. Collect planted samples for each side only where that primary row's exact
   `*_locked` value is `"1"`. A full baseline uses the paired baseline and
   primary `*_lock_drift` values, exactly as the pre-migration Gate E contract.
   An exact legacy baseline has no IK suffix, so its semantically identical
   baseline value is the primary row's immutable pre-solve
   `*_observed_lock_drift`, while the corrected value remains that row's
   `*_lock_drift`. Require at least ten planted samples total before computing
   either mean, then require `mean(on_drift) < mean(off_drift)` with no rounding
   tolerance. This legacy adapter preserves the same support-retargeted
   baseline-versus-corrected measurement; it does not waive drift reduction.

On success print `VALID gate-e` followed by the exact scene ID, route ID,
planted count, off mean, and on mean as named `key=value` fields.

Define the stress allowlist exactly:

```python
GATE_E_STRESS_ROUTES = {
    ("grail-curb-default", "curb-forward"),
    ("grail-curb-medium", "curb-forward"),
    ("grail-curb-high", "curb-forward"),
    ("ramp-15-stress", "up-landing-down"),
}
```

`check_gate_e_stress_pair` rejects unless
`(expected_scene, expected_route)` is in that set and every row in both inputs
has exactly those requested values. Both branches require finite validated
rows, an IK-off baseline, an IK-on primary, and the schema-dependent raw
comparison already defined above: full baselines compare base plus observation
groups, while exact legacy baselines compare the shared base group only.

It accepts exactly one of these branches:

1. **Traverse:** no primary row requests an IK stop and at least one primary
   row has `route_complete == "1"`. Run `check_gate_e_rows(on_rows)` and compare
   the complete pair. Do **not** impose certified-route aggregate planted-drift
   reduction on a class-2 traversal; bounded binary64 clearance, residual,
   orientation, correction, status, and ladder/mask behavior remain mandatory.
2. **Safe stop:** locate the first primary row with
   `ik_safe_stop_requested == "1"`. It must precede the first route-complete
   row if the open-loop driver later emits one, and that exact request row must
   have `ik_candidate_rejected == "1"`. Compare only the inclusive prefix
   through that row, then apply the reason/status/mask evidence and stopped-tail
   predicates below. Divergence after the request remains intentional.

The stress safe-stop evidence for `swing-lift` is no longer “required lift above
0.08” or a predicted margin. It is exact all-ladder exhaustion on at least one
non-contact foot:
`candidates_evaluated == 41` and
`selected_index == G1SwingNoCandidate`. If unresolved certification is named as
the cause, its mask must be nonzero and the checker prints the exact decoded
set (`OutsideDomain`, `BudgetExceeded`, `Uncertified`, including mixtures).
Mask zero remains valid only when every real rejection was controller-only or a
negative `Ok` margin; print that zero-mask cause as
`controller-or-negative-ok`, not as a certification status. The other stop
reasons retain these exact evidence
predicates on the request row: `reach-shell` requires either reachable flag to
be false; `correction-bound` requires `max_ik_correction >= 0.35 - 1e-5`;
`lock-drift` requires either observed lock drift to exceed `0.20`;
`end-effector-residual` requires either contact residual to exceed `0.005`;
and `post-solve-clearance` requires either a non-`Ok` candidate-local pose
status or an `Ok` candidate lower bound below `-0.005` for a locked toe/foot or
below `-0.01` for the candidate minimum. Reject a reason whose predicate is
false, `none`, and unknown reasons. Candidate lower bounds are examined only
for `Ok`; a non-`Ok` pose is proved by its exact status, never fabricated
numbers.

For every safe-stop branch, require the first stop request before the first
`route_complete=1` row if one exists. The immediately following row and every
tail row require `actual_simulation_speed <= 1e-4`; from that first stopped row
through the end, total XZ displacement and support-height rise are each at most
`0.02 m`. Every accepted/rendered row, including the complete tail, retains
the ordinary `-0.01` physical lower bounds and locked `-0.005` toe/foot lower
bounds. Candidate-local `OutsideDomain`/`BudgetExceeded`/`Uncertified` remain
named evidence; global statuses are controlled errors, not a Gate E safe-stop
success. A later open-loop `route_complete=1` is allowed.

On success print `VALID gate-e-stress` with `branch=traverse` or
`branch=safe-stop`, the exact scene/route, and the stop frame/reason plus named
clearance, speed, displacement, and support-rise measurements when applicable.

Print/check binary64 values from their raw CSV text; never round them to a
binary32 surrogate before threshold comparison.

- [ ] **Step 6: Run checker and paired smoke GREEN**

Use the strictly linked `/tmp/controller_g1_ik` built in Task 7:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=grail-curb-low MM_TEST_ROUTE=curb-forward \
  MM_TEST_MODE=route MM_TEST_FRAMES=375 MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_LOG=/tmp/g1-ik-clearance/smoke-off.csv /tmp/controller_g1_ik
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=grail-curb-low MM_TEST_ROUTE=curb-forward \
  MM_TEST_MODE=route MM_TEST_FRAMES=375 MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  MM_LOG=/tmp/g1-ik-clearance/smoke-on.csv /tmp/controller_g1_ik
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-ik-clearance/smoke-on.csv --gate-e \
  --ik-baseline /tmp/g1-ik-clearance/smoke-off.csv \
  --expected-scene grail-curb-low --expected-route curb-forward
```

Expected: tests pass, both runs close normally, raw-string invariants hold, and
Gate E prints one valid summary using explicit status, ladder, and binary64
clearance evidence.

- [ ] **Step 7: Commit diagnostics and acceptance contract**

```bash
git add motion_match_log.h controller.cpp \
  resources/check_g1_runtime_log.py tests/python/test_runtime_log.py
git commit -m "test: enforce certified G1 terrain IK Gate E"
```

### Task 9: Run Full Verification and Gate E Across Every Certified Terrain Family

**Files:**
- Verify only; do not modify or commit source in this task.
- Generate only ignored evidence under `/tmp/g1-ik-clearance/`.

**Interfaces and acceptance matrix:**

| Scene | Exact route | Frames | Required Gate E coverage |
|---|---|---:|---|
| `grail-curb-low` | `curb-forward` | 800 | certified exact-surface curb regression |
| `stairs-shallow` | `ascent-landing-descent` | 800 | ascent, 2 s landing, descent |
| `stairs-standard` | `ascent-landing-descent` | 800 | ascent, 2 s landing, descent |
| `stairs-unseen-variable` | `ascent-landing-descent` | 800 | unseen variable treads, landing, descent |
| `ramp-05-up-down` | `up-landing-down` | 800 | 5-degree ascent, 2 s landing, descent |
| `ramp-10-up-down` | `up-landing-down` | 800 | 10-degree ascent, 2 s landing, descent |
| `cross-slope-05` | `forward-cross-slope` | 800 | 5-degree lateral normal/orientation |
| `cross-slope-10` | `forward-cross-slope` | 800 | 10-degree lateral normal/orientation |
| `mixed-multilevel` | `full-course` | 800 | stairs, elevated walk, blocks, return ramp |

The three class-2 GRAIL curb routes (`grail-curb-default`,
`grail-curb-medium`, and `grail-curb-high`, each `curb-forward`) plus
`ramp-15-stress/up-landing-down` are separate 800-frame
`traverse-or-safe-stop` runs. Each must take one of the two exact branches
checked by `--gate-e-stress`; none is relabeled certified. The blocked-course
safe-stop routes remain sibling Gate D, never Gate E.

Schema routing is part of this verification, not an implicit compatibility
mode:

- every CSV produced by `/tmp/controller_g1_ik` in this task has exactly
  `FULL_RUNTIME_COLUMNS` and is always the positional primary log;
- every preserved weight-zero control and accepted Gate C/D/F reference under
  `/tmp/g1-multiscene-runtime/` has exactly `LEGACY_RUNTIME_COLUMNS`.
  `--compare-control` and `--ik-off-reference` load those paths only with
  `read_legacy_runtime_rows`; no legacy file is passed to `read_rows` or used as
  the positional primary;
- every newly generated matched Gate E/stress `off` log is full schema.
  `--ik-baseline` must dispatch it as `"full"`, compare all base and observation
  invariants, and reject a partially appended or hybrid header. The exact
  legacy baseline branch is exercised in Task 8 unit tests only; it compares
  base invariants because the observation suffix does not exist there;
- Gate C, Gate D, Gate F, failed-switch, `--compare-control`, and
  `--ik-off-reference` still require every primary row to have
  `ik_enabled == "0"`. Only Gate E and Gate E stress accept IK-on primaries.

- [ ] **Step 1: Run the complete Python suite and independent published-pack validator**

Run without rebuilding or republishing any artifact:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/validate_g1_terrain_database.py resources/g1_terrain
```

Expected: every Python test passes. The validator exits `0` with `VALID g1-terrain-artifacts/v2`, `frames=459682`, `clips=1770`, `bones=31`, `terrain_dims=4`, `support_dims=3`, and `scenes=14`. It does not write the pack.

- [ ] **Step 2: Compile and run every native test in strict, release, and sanitizer modes**

Run:

```bash
mkdir -p /tmp/g1-ik-clearance/native
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-strict.o
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-release.o
g++ -std=c++17 -O1 -g -fno-fast-math -ffp-contract=off \
  -frounding-math \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-san.o
cpp_tests=(
  test_g1_skeleton test_terrain_database test_terrain_runtime
  test_scene_runtime test_support_runtime test_g1_controller_state
  test_scene_switch test_route_runtime test_support_matching
  test_cleanup_runtime test_g1_ik test_g1_clearance
)
for name in "${cpp_tests[@]}"; do
  source="tests/cpp/${name}.cpp"
  test -f "$source"
  extra=()
  if test "$name" = test_g1_ik; then
    extra=(-DG1_IK_ENABLE_TEST_SEAMS)
  fi
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "${extra[@]}" -c "$source" \
    -o "/tmp/g1-ik-clearance/native/${name}-strict.o"
  g++ "/tmp/g1-ik-clearance/native/${name}-strict.o" \
    /tmp/g1-ik-clearance/native/g1-clearance-strict.o \
    -o "/tmp/g1-ik-clearance/native/${name}-strict"
  "/tmp/g1-ik-clearance/native/${name}-strict"
  g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
    "${extra[@]}" -c "$source" \
    -o "/tmp/g1-ik-clearance/native/${name}-release.o"
  g++ "/tmp/g1-ik-clearance/native/${name}-release.o" \
    /tmp/g1-ik-clearance/native/g1-clearance-release.o \
    -o "/tmp/g1-ik-clearance/native/${name}-release"
  "/tmp/g1-ik-clearance/native/${name}-release"
  g++ -std=c++17 -O1 -g -fno-fast-math \
    -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
    -fno-sanitize-recover=all -fno-omit-frame-pointer -I. \
    "${extra[@]}" -c "$source" \
    -o "/tmp/g1-ik-clearance/native/${name}-san.o"
  g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
    -fno-sanitize-recover=all \
    "/tmp/g1-ik-clearance/native/${name}-san.o" \
    /tmp/g1-ik-clearance/native/g1-clearance-san.o \
    -o "/tmp/g1-ik-clearance/native/${name}-san"
  ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
    "/tmp/g1-ik-clearance/native/${name}-san"
done
```

Expected: all 36 invocations exit `0`; strict builds emit no warning,
sanitizers emit no report, every kernel object was compiled with strict FP, and
no final link command contains `-ffast-math`.

- [ ] **Step 3: Build the final non-sanitized controller and run source-order guards**

Run:

```bash
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/native/g1-clearance-controller.o
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  -c controller.cpp -o /tmp/g1-ik-clearance/native/controller.o
g++ /tmp/g1-ik-clearance/native/controller.o \
  /tmp/g1-ik-clearance/native/g1-clearance-controller.o \
  -o /tmp/controller_g1_ik \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
! rg -n 'contact_(reset|update)|ik_look_at|static void ik_two_bone' \
  controller.cpp
! rg -n 'database_(search|build_matching_features)\([^;]*(support|ik_)' \
  controller.cpp
! rg -n 'g1_29dof\.xml|mjx-diffphysics|meshes/.*\.STL' \
  controller.cpp g1_ik.h g1_clearance.h g1_ik_runtime.h
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
from pathlib import Path
text = Path("controller.cpp").read_text(encoding="utf-8")
support = text.index("support_pose_apply(")
ik = text.index("g1_ik_frame_evaluate(")
clearance = text.index("g1_measure_pose_clearance(")
log = text.index("deterministic_log.write(")
assert support < ik < clearance < log
print("VALID support->IK->clearance->log order")
PY
```

Expected: the controller builds; inherited generic contact/IK code and runtime XML/STL dependencies are absent; matcher calls have no support/IK argument; and the order assertion prints its `VALID` line.

- [ ] **Step 4: Re-run the sibling Gate C/D/F suite with IK explicitly off**

Do not overwrite the preserved prerequisite CSVs. First reproduce all six Gate C routes into this task's directory and compare the exact invariance columns to their accepted weight-4 baselines:

```bash
mkdir -p /tmp/g1-ik-clearance/gates
gate_c=(
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  stairs-unseen-variable:ascent-landing-descent
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
  mixed-multilevel:full-course
)
for specification in "${gate_c[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  current="/tmp/g1-ik-clearance/gates/gate-c-${scene}__${route}-ik-off.csv"
  accepted="/tmp/g1-multiscene-runtime/gate-c-${scene}__${route}-w4.csv"
  control="/tmp/g1-multiscene-runtime/gate-c-${scene}__${route}-w0.csv"
  test -s "$accepted"
  test -s "$control"
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=4 MM_IK=0 MM_LOG="$current" \
    /tmp/controller_g1_ik
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$current" --gate-c \
    --compare-control "$control" --ik-off-reference "$accepted"
done
```

Expected for every route: `VALID gate-c`, `VALID terrain-comparison`, and `VALID ik-off-reference frames=800`. The accepted baseline files remain untouched.

Re-run the two sibling Gate D routes with `MM_IK=0` into new paths. All four
class-2 routes are deliberately absent here; their only acceptance contract is
the branch-aware Gate E stress matrix below:

```bash
gate_d=(
  blocked-course:wall-safe-stop
  blocked-course:ramp-safe-stop
)
for specification in "${gate_d[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  log="/tmp/g1-ik-clearance/gates/gate-d-${scene}__${route}-ik-off.csv"
  accepted="/tmp/g1-multiscene-runtime/gate-d-${scene}__${route}.csv"
  test -s "$accepted"
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=600 MM_TERRAIN_WEIGHT=4 MM_IK=0 MM_LOG="$log" \
    /tmp/controller_g1_ik
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$log" --gate-d \
    --ik-off-reference "$accepted"
done
```

Expected: both print `VALID gate-d` and `VALID ik-off-reference frames=600`;
each blocked route stops before class 0 without support lift or a pre-IK
runtime regression.

Run the exact sibling Gate F scene cycle with IK off, using new paths:

```bash
SCENES=grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,mixed-multilevel,blocked-course
test -s /tmp/g1-multiscene-runtime/gate-f-scene-cycle.csv
rm -f /tmp/g1-ik-clearance/gates/gate-f-ik-off.csv \
  /tmp/g1-ik-clearance/gates/gate-f-cleanup-ik-off.json
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=700 \
  MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_LOG=/tmp/g1-ik-clearance/gates/gate-f-ik-off.csv \
  MM_CLEANUP_LOG=/tmp/g1-ik-clearance/gates/gate-f-cleanup-ik-off.json \
  /tmp/controller_g1_ik
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-ik-clearance/gates/gate-f-ik-off.csv --gate-f \
  --expected-scenes "$SCENES" \
  --ik-off-reference /tmp/g1-multiscene-runtime/gate-f-scene-cycle.csv
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
path = "/tmp/g1-ik-clearance/gates/gate-f-cleanup-ik-off.json"
with open(path, encoding="utf-8") as stream:
    cleanup = json.load(stream)
assert cleanup == {
    "exit_code": 0,
    "live_model_count": 0,
    "log_closed": True,
    "model_load_count": 28,
    "model_unload_count": 28,
    "motion_pack_load_count": 1,
    "window_closed": True,
}, cleanup
print("VALID gate-f cleanup")
PY
```

Re-run the malformed-candidate half of sibling Gate F with the final controller,
IK explicitly off, and compare it to the accepted pre-IK failure log:

```bash
SCENES=grail-curb-default,grail-curb-low,grail-curb-medium,grail-curb-high,stairs-shallow,stairs-standard,stairs-unseen-variable,ramp-05-up-down,ramp-10-up-down,ramp-15-stress,cross-slope-05,cross-slope-10,mixed-multilevel,blocked-course
test -s /tmp/g1-multiscene-runtime/gate-f-malformed.csv
BAD_ROOT=$(mktemp -d /tmp/g1-ik-malformed-pack.XXXXXX)
for file in database.bin terrain_features.bin terrain_support.bin manifest.json validation.json
do
  ln -s "$(realpath resources/g1_terrain/$file)" "$BAD_ROOT/$file"
done
mkdir "$BAD_ROOT/scenes"
ln -s "$(realpath resources/g1_terrain/scenes/index.json)" \
  "$BAD_ROOT/scenes/index.json"
for scene in ${SCENES//,/ }
do
  if test "$scene" = stairs-shallow; then
    cp -a "resources/g1_terrain/scenes/$scene" "$BAD_ROOT/scenes/$scene"
    truncate -s 1 "$BAD_ROOT/scenes/$scene/terrain.obj"
  else
    ln -s "$(realpath resources/g1_terrain/scenes/$scene)" \
      "$BAD_ROOT/scenes/$scene"
  fi
done
DISPLAY=:1 G1_TERRAIN_DIR="$BAD_ROOT" \
  MM_TEST_MODE=scene-cycle MM_SCENE_DWELL_FRAMES=25 MM_TEST_FRAMES=350 \
  MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_LOG=/tmp/g1-ik-clearance/gates/gate-f-malformed-ik-off.csv \
  MM_CLEANUP_LOG=/tmp/g1-ik-clearance/gates/gate-f-malformed-cleanup-ik-off.json \
  /tmp/controller_g1_ik
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  /tmp/g1-ik-clearance/gates/gate-f-malformed-ik-off.csv \
  --expect-switch-failure \
  --ik-off-reference /tmp/g1-multiscene-runtime/gate-f-malformed.csv
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
path = "/tmp/g1-ik-clearance/gates/gate-f-malformed-cleanup-ik-off.json"
with open(path, encoding="utf-8") as stream:
    cleanup = json.load(stream)
assert cleanup["exit_code"] == 0, cleanup
assert cleanup["live_model_count"] == 0, cleanup
assert cleanup["log_closed"] is True and cleanup["window_closed"] is True, cleanup
assert cleanup["model_load_count"] == cleanup["model_unload_count"], cleanup
assert cleanup["motion_pack_load_count"] == 1, cleanup
print("VALID gate-f malformed cleanup")
PY
```

Expected: the normal log covers two complete ordered passes through all 14
scenes; the checker reports 28 generations, `VALID ik-off-reference
frames=700`, one motion-pack load, and one live terrain model per row; final
cleanup balances all 28 terrain model loads and closes log/window normally.
The malformed overlay prints `VALID
switch-failure`, `VALID ik-off-reference frames=350`, and `VALID gate-f
malformed cleanup`; the active valid scene survives every failed candidate and
all candidate model ownership remains balanced.

- [ ] **Step 5: Run matched IK-off/IK-on Gate E pairs for the certified matrix**

Run:

```bash
mkdir -p /tmp/g1-ik-clearance/gate-e
gate_e=(
  grail-curb-low:curb-forward
  stairs-shallow:ascent-landing-descent
  stairs-standard:ascent-landing-descent
  stairs-unseen-variable:ascent-landing-descent
  ramp-05-up-down:up-landing-down
  ramp-10-up-down:up-landing-down
  cross-slope-05:forward-cross-slope
  cross-slope-10:forward-cross-slope
  mixed-multilevel:full-course
)
for specification in "${gate_e[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  off="/tmp/g1-ik-clearance/gate-e/${scene}__${route}-off.csv"
  on="/tmp/g1-ik-clearance/gate-e/${scene}__${route}-on.csv"
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=4 MM_IK=0 MM_LOG="$off" \
    /tmp/controller_g1_ik
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=4 MM_IK=1 MM_LOG="$on" \
    /tmp/controller_g1_ik
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$on" --gate-e \
    --ik-baseline "$off" --expected-scene "$scene" \
    --expected-route "$route"
  if test "$scene" = mixed-multilevel; then
    /home/ubuntu/miniconda3/envs/diffsim/bin/python \
      resources/check_g1_runtime_log.py "$off" --gate-c \
      --compare-control \
        /tmp/g1-multiscene-runtime/gate-c-mixed-multilevel__full-course-w0.csv \
      --ik-off-reference \
        /tmp/g1-multiscene-runtime/gate-c-mixed-multilevel__full-course-w4.csv
  fi
done
```

Expected: nine `VALID gate-e` summaries. Every pair is exactly invariant in matching/query/cost/support-root/simulation columns; each certified IK-on route completes with no safe-stop; aggregate planted drift is strictly lower; all clearance and correction bounds pass at 25 Hz.
The exact mixed Gate E off log also prints `VALID gate-c`, the mixed-specific
elevated/plateau/return measurements, `VALID terrain-comparison`, and
`VALID ik-off-reference frames=800`; this prevents a separately accepted Gate C
file from standing in for the motion actually paired with IK.

- [ ] **Step 6: Exercise every stress route, safe-stop handoff, and controlled errors**

Run all four class-2 pairs:

```bash
stress_routes=(
  grail-curb-default:curb-forward
  grail-curb-medium:curb-forward
  grail-curb-high:curb-forward
  ramp-15-stress:up-landing-down
)
for specification in "${stress_routes[@]}"; do
  IFS=: read -r scene route <<<"$specification"
  off="/tmp/g1-ik-clearance/gate-e/${scene}__${route}-off.csv"
  on="/tmp/g1-ik-clearance/gate-e/${scene}__${route}-on.csv"
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=4 MM_IK=0 MM_LOG="$off" \
    /tmp/controller_g1_ik
  DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
    MM_TERRAIN_SCENE="$scene" MM_TEST_MODE=route MM_TEST_ROUTE="$route" \
    MM_TEST_FRAMES=800 MM_TERRAIN_WEIGHT=4 MM_IK=1 MM_LOG="$on" \
    /tmp/controller_g1_ik
  /home/ubuntu/miniconda3/envs/diffsim/bin/python \
    resources/check_g1_runtime_log.py "$on" --gate-e-stress \
    --ik-baseline "$off" --expected-scene "$scene" \
    --expected-route "$route"
done
```

Expected: four `VALID gate-e-stress` summaries, each with
`branch=traverse` or `branch=safe-stop`. A safe-stop branch proves the request
reaches zero actual planar motion on the immediately following update without
stretching, body lift, accepted-pose penetration, or matching changes through
the request.

The non-finite target/surface/clearance fixtures already ran in all three native modes. Confirm invalid `MM_IK` exits before Raylib and creates no log/model state:

```bash
rm -f /tmp/g1-ik-clearance/invalid-ik.csv
set +e
MM_IK=true MM_LOG=/tmp/g1-ik-clearance/invalid-ik.csv \
  /tmp/controller_g1_ik \
  >/tmp/g1-ik-clearance/invalid-ik.out \
  2>/tmp/g1-ik-clearance/invalid-ik.err
invalid_status=$?
set -e
test "$invalid_status" -eq 2
grep -F "MM_IK must be exactly 0 or 1" \
  /tmp/g1-ik-clearance/invalid-ik.err
test ! -e /tmp/g1-ik-clearance/invalid-ik.csv
```

Expected: exit `2`, the exact diagnostic is present, and no display/log/model was opened. Runtime non-finite fixtures leave output/state transactional and report the named controlled error; the post-window controller path always reaches log/model/window cleanup.

- [ ] **Step 7: Audit the dirty tree and freeze the verified controller**

Run:

```bash
git diff --check
git status --short
sha256sum /tmp/controller_g1_ik \
  > /tmp/g1-ik-clearance/controller-g1-ik.sha256
```

Expected: `git diff --check` is silent; status contains no generated CSV, binary, converted mesh, or unrelated staged file; the checksum names `/tmp/controller_g1_ik`. Do not launch a live controller in this task: the final optional rendering decision is the only remaining task and owns the one final `DISPLAY=:1` launch.

### Task 10 (Optional, Last): Try Rigid G1 STL Link Rendering Behind an Ease Gate

**Files:**
- Create on pass: `resources/g1_visual_mesh_source.py`
- Create on pass: `resources/convert_g1_visual_meshes.py`
- Create on pass: `tests/python/test_g1_visual_meshes.py`
- Create on pass: `g1_visual_mesh.h`
- Create on pass: `tests/cpp/test_g1_visual_mesh.cpp`
- Modify on pass: `.gitignore`
- Modify on pass: `controller.cpp`
- Create on failure: `docs/superpowers/evidence/2026-07-13-g1-rigid-mesh-deferral.md`
- Generate, never commit: `resources/g1_visual_mesh/`, converted OBJ files, manifest binaries/JSON, timing JSON, logs, and binaries.

**Source and non-blocking boundary:**
- This task starts only after every Task 9 check is green. Run it in an isolated worktree using the `using-git-worktrees` skill so a failed visual experiment cannot contaminate the verified correctness branch.
- The only source is `/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml` and its `/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/meshes/` directory. Despite the filename referring to 29 actuated degrees of freedom, the XML currently names exactly 30 non-world link bodies and 36 visual mesh geoms/assets; the converter must discover and verify those facts rather than silently assuming a partial list.
- Rendering is rigid-link visualization only. It consumes the final `state.ik_global_bone_positions/rotations` as const input and may not write controller, matching, support, IK, route, log, or scene state. Skeleton rendering remains the default, live toggle, and fallback.
- A pass is optional. A mapping, conversion, load, memory, timing, lifecycle, fallback, or state-invariance failure writes the evidence note and leaves the verified skeleton controller as the final result. It never blocks Gate E or causes a correctness change.

**Ease gate (all rows required for a pass):**

| Gate | Exact pass requirement |
|---|---|
| Mapping | 30 unique XML bodies map bijectively to G1 bones `1..30`; all 36 mesh geoms map to their owning bone; all 36 unique mesh assets are referenced once; no missing/extra body, asset, geom, unsafe path, or non-finite local pose/color |
| Basis/conversion | every STL vertex/normal and geom local pose uses `(x,y,z)->(x,z,-y)`; quaternions use the equivalent Z-up-to-Y-up conjugation; binary STL length/count is exact; two conversions yield byte-identical OBJ, `manifest.bin`, `manifest.json`, and tree checksums |
| Load/lifecycle | all 36 OBJ models report ready; any partial failure unloads every allocated model and selects skeleton; successful cleanup has visual loads = unloads = 36 |
| Memory | maximum resident-set increase of mesh over matched skeleton 300-frame smoke is no more than `512 MiB` (`524288 KiB`) |
| Frame cost | over exactly 300 mesh frames, mean time inside rigid-link draw calls is no more than `5.0 ms` and p95 no more than `10.0 ms` |
| Reversibility | default is skeleton; live skeleton/mesh toggle never reloads models; missing/corrupt output warns once and falls back; skeleton and mesh 300-frame CSVs are byte-identical; all Task 9 tests/gates still pass after integration |

- [ ] **Step 1: Write failing complete-mapping and deterministic-conversion tests**

Create `resources/g1_visual_mesh_source.py` with the one explicit source-to-runtime contract used by both converter and tests:

```python
XML_PATH = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
MESH_ROOT = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/meshes"

BODY_TO_G1_BONE = {
    "pelvis": 1,
    "left_hip_pitch_link": 2,
    "left_hip_roll_link": 3,
    "left_hip_yaw_link": 4,
    "left_knee_link": 5,
    "left_ankle_pitch_link": 6,
    "left_ankle_roll_link": 7,
    "right_hip_pitch_link": 8,
    "right_hip_roll_link": 9,
    "right_hip_yaw_link": 10,
    "right_knee_link": 11,
    "right_ankle_pitch_link": 12,
    "right_ankle_roll_link": 13,
    "waist_yaw_link": 14,
    "waist_roll_link": 15,
    "torso_link": 16,
    "left_shoulder_pitch_link": 17,
    "left_shoulder_roll_link": 18,
    "left_shoulder_yaw_link": 19,
    "left_elbow_link": 20,
    "left_wrist_roll_link": 21,
    "left_wrist_pitch_link": 22,
    "left_wrist_yaw_link": 23,
    "right_shoulder_pitch_link": 24,
    "right_shoulder_roll_link": 25,
    "right_shoulder_yaw_link": 26,
    "right_elbow_link": 27,
    "right_wrist_roll_link": 28,
    "right_wrist_pitch_link": 29,
    "right_wrist_yaw_link": 30,
}
```

Create `tests/python/test_g1_visual_meshes.py`. Use `xml.etree.ElementTree`, a synthetic two-triangle binary STL, and public converter functions. The core tests are:

```python
class G1VisualMeshTests(unittest.TestCase):
    def test_xml_mapping_is_complete_and_bijective(self):
        mapping = inspect_xml(XML_PATH, MESH_ROOT)
        self.assertEqual(mapping.body_count, 30)
        self.assertEqual(mapping.geom_count, 36)
        self.assertEqual(mapping.asset_count, 36)
        self.assertEqual(set(mapping.body_names), set(BODY_TO_G1_BONE))
        self.assertEqual(set(BODY_TO_G1_BONE.values()), set(range(1, 31)))
        self.assertEqual(
            sorted(geom.mesh_name for geom in mapping.geoms),
            sorted(mapping.asset_files))
        self.assertTrue(all(geom.bone == BODY_TO_G1_BONE[geom.body_name]
                            for geom in mapping.geoms))

    def test_basis_maps_vertices_normals_and_local_pose(self):
        self.assertEqual(zup_to_yup((1.0, 2.0, 3.0)), (1.0, 3.0, -2.0))
        pose = convert_local_pose((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(pose.position, (0.1, 0.3, -0.2))
        self.assertQuaternionAlmostEqual(pose.rotation, (1.0, 0.0, 0.0, 0.0))

    def test_binary_stl_rejects_truncation_trailing_and_nonfinite(self):
        payload = tiny_binary_stl()
        self.assertEqual(len(read_binary_stl(payload)), 2)
        for bad in (payload[:-1], payload + b"x", stl_with_nan(payload)):
            with self.assertRaises(ValueError):
                read_binary_stl(bad)

    def test_conversion_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = pathlib.Path(temporary, "first")
            second = pathlib.Path(temporary, "second")
            convert_visual_meshes(self.fixture_xml, self.fixture_meshes, first)
            convert_visual_meshes(self.fixture_xml, self.fixture_meshes, second)
            self.assertEqual(tree_hashes(first), tree_hashes(second))
            self.assertEqual((first / "manifest.bin").read_bytes(),
                             (second / "manifest.bin").read_bytes())
```

Also test duplicate body names, an unmapped body, duplicate/missing asset reference, path escape/symlink, ASCII STL, zero triangles, invalid RGBA, invalid/non-unit quaternion, and that OBJ indices/normals are valid after deterministic first-occurrence vertex deduplication.

- [ ] **Step 2: Run converter tests to verify RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_g1_visual_meshes -v
```

Expected: import failure because `resources/convert_g1_visual_meshes.py` does not exist.

- [ ] **Step 3: Implement strict XML inspection and binary-STL-to-OBJ conversion**

Create `resources/convert_g1_visual_meshes.py` using only the Python standard library. Define immutable `Triangle`, `LocalPose`, `VisualGeom`, and `VisualMapping` dataclasses. The exact public interfaces and their return contracts are:

| Function | Exact return/effect |
|---|---|
| `inspect_xml(xml_path: str, mesh_root: str) -> VisualMapping` | validated ordered body/asset/geom records with integer `body_count`, `asset_count`, and `geom_count` |
| `zup_to_yup(value: tuple[float,float,float]) -> tuple[float,float,float]` | finite `(x,z,-y)` triple with signed zero canonicalized |
| `convert_local_pose(position, quaternion_wxyz) -> LocalPose` | converted position and normalized conjugated wxyz quaternion |
| `read_binary_stl(payload: bytes) -> tuple` | nonempty immutable tuple of transformed finite `Triangle` values |
| `write_obj(path: pathlib.Path, triangles: tuple) -> None` | one canonical OBJ written atomically at `path`; every tuple item must be `Triangle` |
| `convert_visual_meshes(xml_path: str, mesh_root: str, output: pathlib.Path) -> dict` | finalized canonical manifest dictionary after transactional output publication |

`inspect_xml` enforces `<compiler meshdir="meshes">`, exactly 30 mapped descendant bodies, exactly 36 `<asset><mesh>` entries and 36 `type="mesh"` geoms, one reference to every asset, no reference to an undeclared asset, and no extra/missing mapped body. Resolve every asset with `realpath/commonpath`; require a nonempty regular nonsymlink `.STL` file beneath the exact mesh root. Parse geom `pos` default `0 0 0`, `quat` default `1 0 0 0`, and four-component `rgba`; reject other orientation attributes, scale, non-finite values, and quaternion norm error above `1e-6`.

The basis quaternion is the same wxyz value used by the motion converter:

```python
Q_ZUP_TO_YUP = (2.0 ** -0.5, -2.0 ** -0.5, 0.0, 0.0)

def zup_to_yup(value):
    x, y, z = value
    return canonical_float(x), canonical_float(z), canonical_float(-y)

def convert_local_pose(position, quaternion_wxyz):
    rotation = quat_mul(
        quat_mul(Q_ZUP_TO_YUP, quaternion_wxyz),
        quat_inverse(Q_ZUP_TO_YUP),
    )
    return LocalPose(zup_to_yup(position), canonical_quaternion(rotation))
```

`read_binary_stl` requires exactly `84 + 50 * triangle_count` bytes, positive count, finite float32 normal/vertices, zero attribute byte count, and a nondegenerate triangle. Rotate vertices/normals with `zup_to_yup`; normalize a valid supplied normal or deterministically recompute it from the transformed winding when its length is below `1e-12`.

`write_obj` walks triangles in source order, assigns a vertex index on first encounter of its exact transformed float32 triple, emits one normalized `vn` per triangle, and emits `f v//n v//n v//n`. Canonical number formatting is `format(value, ".9g")` after mapping both signed zeros to `0.0`; output is UTF-8 with `\n` endings and one terminal newline.

Emit 36 files in XML geom order, using a zero-padded three-digit ordinal, one hyphen, the declared mesh name, and `.obj`; the first ordinal is `000` and the last is `035`. `manifest.bin` has exact little-endian layout:

```text
header <4sIIII>: magic="G1VM", version=1, body_count=30,
                  geom_count=36, record_size=144
36 records <I3f4f4f96s>: G1 bone index, local position xyz,
                         local quaternion wxyz, RGBA,
                         NUL-padded relative OBJ path
```

Reject a path of 96 bytes or more; require zero padding. Emit canonical sorted/indented `manifest.json` containing schema `G1VM/v1`, source XML path/SHA-256, 30-body map, the 36 records in binary order, source STL SHA-256, output OBJ SHA-256, triangle/vertex counts, and a SHA-256 of `manifest.bin`. Write into a unique sibling staging directory, fsync files/directories, and rename only after rereading both manifests and every output hash; never partially replace an existing output directory.

Add `resources/g1_visual_mesh/` as one exact line in `.gitignore`.

- [ ] **Step 4: Run mapping/conversion GREEN and the deterministic full-asset gate**

Run focused tests, then convert the complete source twice into fresh temporary directories and compare every byte:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_g1_visual_meshes -v
first=$(mktemp -d /tmp/g1-visual-first.XXXXXX)
second=$(mktemp -d /tmp/g1-visual-second.XXXXXX)
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/convert_g1_visual_meshes.py \
  --xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --mesh-root /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/meshes \
  --output "$first/output"
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/convert_g1_visual_meshes.py \
  --xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --mesh-root /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/meshes \
  --output "$second/output"
(cd "$first/output" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$first/tree.sha256"
(cd "$second/output" && find . -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$second/tree.sha256"
cmp "$first/tree.sha256" "$second/tree.sha256"
test "$(find "$first/output" -maxdepth 1 -name '*.obj' | wc -l)" -eq 36
test ! -e resources/g1_visual_mesh
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/convert_g1_visual_meshes.py \
  --xml /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  --mesh-root /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/meshes \
  --output resources/g1_visual_mesh
```

Expected: tests pass; each converter prints `CONVERTED G1VM/v1 bodies=30 geoms=36 assets=36`; checksum files are identical; exactly 36 OBJ files plus the two manifests exist. If `resources/g1_visual_mesh` already exists, preserve it and treat the ease gate as blocked unless its complete tree hash equals the just-generated output; do not delete or overwrite user output.

- [ ] **Step 5: Commit only the passing deterministic converter boundary on the optional branch**

```bash
git add .gitignore resources/g1_visual_mesh_source.py \
  resources/convert_g1_visual_meshes.py tests/python/test_g1_visual_meshes.py
git diff --cached --check
git commit -m "feat: convert complete G1 rigid visual links"
```

The ignored `resources/g1_visual_mesh/` directory must not appear in `git diff --cached --name-only`.

- [ ] **Step 6: Write failing manifest, lifecycle, transform, toggle, and fallback tests**

Create `tests/cpp/test_g1_visual_mesh.cpp`. It writes a valid 36-record `/tmp/test-g1vm-manifest.bin` using the exact layout above and then tests these public declarations from the missing header:

```cpp
enum G1VisualDrawMode { G1VisualSkeleton = 0, G1VisualRigidMesh = 1 };

struct G1VisualLinkSpec
{
    int bone = -1;
    vec3 local_position;
    quat local_rotation;
    float rgba[4] = {};
    std::string relative_obj;
};

struct G1VisualManifest
{
    std::vector<G1VisualLinkSpec> links;
};

struct G1VisualLinkTransform
{
    vec3 position;
    quat rotation;
};

bool g1_visual_draw_mode_parse(
    G1VisualDrawMode& output, const char* text, char* error, int capacity);
bool g1_visual_manifest_load(
    G1VisualManifest& output, const char* root, char* error, int capacity);
G1VisualLinkTransform g1_visual_link_transform(
    const G1VisualLinkSpec& link, vec3 bone_position, quat bone_rotation);
```

The test requires default/unset and `skeleton` to select skeleton, only exact `mesh` to select mesh, and all other strings to fail. It requires 36 parsed links, bone indices `1..30`, finite normalized local quaternions, safe existing OBJ paths, and exact local transforms. Corrupt magic/version/count/record size/bone/path padding/path traversal/quaternion/truncation/trailing bytes must fail transactionally.

Add a templated transactional ownership test using fake integer models and callbacks. A failure on model 7 must unload exactly the seven allocated candidates and preserve the prior 36-model active set; a complete load commits exactly 36 candidates and later unload visits each once. Toggle calls change only an enum/bool and leave load/unload counts unchanged. Hash the input 31-position/rotation arrays before and after all pure link-transform calls and require exact equality.

- [ ] **Step 7: Compile to verify renderer RED**

Run:

```bash
mkdir -p /tmp/g1-ik-clearance/visual-red
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/visual-red/g1-clearance.o
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -I. -I/home/ubuntu/apps/raylib/src \
  -c tests/cpp/test_g1_visual_mesh.cpp \
  -o /tmp/g1-ik-clearance/visual-red/test-g1-visual-mesh.o
g++ /tmp/g1-ik-clearance/visual-red/test-g1-visual-mesh.o \
  /tmp/g1-ik-clearance/visual-red/g1-clearance.o \
  -o /tmp/test_g1_visual_mesh \
  -L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected: compilation fails with `fatal error: g1_visual_mesh.h: No such file or directory`.

- [ ] **Step 8: Implement transactional Raylib link ownership and const-pose drawing**

Create `g1_visual_mesh.h`. Implement the declarations above plus:

```cpp
struct G1VisualRenderStats
{
    std::vector<double> draw_milliseconds;
    int visual_model_loads = 0;
    int visual_model_unloads = 0;
    bool fallback = false;
};

struct G1VisualMesh
{
    G1VisualManifest manifest;
    std::vector<Model> models;
    bool ready = false;
    int visual_model_loads = 0;
    int visual_model_unloads = 0;
};

bool g1_visual_mesh_load(
    G1VisualMesh& output, const char* root, char* error, int capacity);
void g1_visual_mesh_unload(G1VisualMesh& mesh);
void g1_visual_mesh_draw(
    const G1VisualMesh& mesh,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations);
bool g1_visual_render_stats_write(
    const char* path, const G1VisualRenderStats& stats,
    char* error, int capacity);
```

`g1_visual_manifest_load` validates exact file length/header/records, safe paths beneath `root`, all 36 existing nonempty regular OBJ files, finite RGBA in `[0,1]`, normalized local quaternions, and bone range. It builds a local candidate and assigns only after all records pass.

`g1_visual_mesh_load` calls `LoadModel` in record order after Raylib initialization. Treat `IsModelReady == false` as failure; unload every ready candidate and leave output unchanged. `g1_visual_mesh_unload` unloads every owned model exactly once, clears the vector, and sets `ready=false`. Scene switch/reset never calls either function.

`g1_visual_link_transform` is exactly:

```cpp
G1VisualLinkTransform result;
result.position = bone_position +
    quat_mul_vec3(bone_rotation, link.local_position);
result.rotation = quat_normalize(
    quat_mul(bone_rotation, link.local_rotation));
return result;
```

`g1_visual_mesh_draw` requires 31-element const slices, computes each link transform, converts its quaternion to a finite normalized axis/angle, and calls `DrawModelEx` with unit scale and the record RGBA. It has no pointer/reference to `g1_controller_state` and no mutable global. Draw timing is measured around this one call in `controller.cpp`, outside fixed-update logic.

`g1_visual_render_stats_write` atomically writes canonical compact JSON with exact keys `average_ms`, `fallback`, `frames`, `p95_ms`, `visual_model_loads`, and `visual_model_unloads`. For a ready 300-frame mesh run it requires 300 finite nonnegative samples and balanced `36/36` loads/unloads; fallback permits `0/0` and zero samples.

- [ ] **Step 9: Integrate lazy mesh loading, skeleton fallback, toggle, and cleanup**

In `controller.cpp`, parse `MM_G1_RENDER` before Raylib with `g1_visual_draw_mode_parse`; unset defaults to skeleton. `MM_G1_VISUAL_DIR` defaults to `resources/g1_visual_mesh`, and `MM_G1_RENDER_STATS` is an optional cleanup report path. Invalid render-mode text exits `2` before opening a log/window.

After `InitWindow`, load the visual set only when startup mode is mesh. A failed parse/model load prints one line beginning `G1 visual mesh fallback:`, keeps the error diagnostic, selects skeleton, and continues. Add a `rigid G1 link mesh` checkbox beside the IK checkbox. The first false-to-true edge lazily loads the set if needed; failure restores false. Later mesh/skeleton edges only change draw mode and never reload/unload.

Replace only the character draw selection:

```cpp
if (visual_draw_mode == G1VisualRigidMesh && visual_mesh.ready) {
    const double draw_begin = GetTime();
    g1_visual_mesh_draw(
        visual_mesh,
        state.ik_global_bone_positions,
        state.ik_global_bone_rotations);
    visual_stats.draw_milliseconds.push_back(
        1000.0 * (GetTime() - draw_begin));
} else {
    draw_g1_skeleton(
        state.ik_global_bone_positions,
        db.bone_parents);
}
```

Extract the current inline bone/cylinder drawing loop verbatim into `static void draw_g1_skeleton(const slice1d<vec3> global_positions, const slice1d<int> parents)`, then call that helper in the `else` branch above. Never pass `state` itself into the mesh API.

At the one normal post-window cleanup tail: close the deterministic log, call `g1_visual_mesh_unload`, copy its load/unload counters into `visual_stats`, unload the sibling terrain model, close the window, write optional visual stats, then write the sibling cleanup report. Visual-link counters are separate and must not change Gate F's terrain `model_load_count/model_unload_count` semantics.

- [ ] **Step 10: Run renderer GREEN, lifecycle, memory, frame-cost, fallback, and state-invariance gates**

Run:

```bash
mkdir -p /tmp/g1-ik-clearance/visual
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I. \
  -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/visual/g1-clearance-test.o
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I. -I/home/ubuntu/apps/raylib/src \
  -c tests/cpp/test_g1_visual_mesh.cpp \
  -o /tmp/g1-ik-clearance/visual/test-g1-visual-mesh.o
g++ /tmp/g1-ik-clearance/visual/test-g1-visual-mesh.o \
  /tmp/g1-ik-clearance/visual/g1-clearance-test.o \
  -o /tmp/test_g1_visual_mesh \
  -L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
/tmp/test_g1_visual_mesh
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -DNDEBUG -I. -c g1_clearance.cpp \
  -o /tmp/g1-ik-clearance/visual/g1-clearance-controller.o
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  -c controller.cpp \
  -o /tmp/g1-ik-clearance/visual/controller-mesh.o
g++ /tmp/g1-ik-clearance/visual/controller-mesh.o \
  /tmp/g1-ik-clearance/visual/g1-clearance-controller.o \
  -o /tmp/controller_g1_mesh \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
/usr/bin/time -v -o /tmp/g1-ik-clearance/visual/skeleton.time \
  env DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TEST_MODE=route \
  MM_TEST_ROUTE=full-course MM_TEST_FRAMES=300 MM_TERRAIN_WEIGHT=4 \
  MM_IK=1 MM_G1_RENDER=skeleton \
  MM_LOG=/tmp/g1-ik-clearance/visual/skeleton.csv \
  /tmp/controller_g1_mesh
/usr/bin/time -v -o /tmp/g1-ik-clearance/visual/mesh.time \
  env DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TEST_MODE=route \
  MM_TEST_ROUTE=full-course MM_TEST_FRAMES=300 MM_TERRAIN_WEIGHT=4 \
  MM_IK=1 MM_G1_RENDER=mesh MM_G1_VISUAL_DIR=resources/g1_visual_mesh \
  MM_G1_RENDER_STATS=/tmp/g1-ik-clearance/visual/mesh-stats.json \
  MM_LOG=/tmp/g1-ik-clearance/visual/mesh.csv \
  /tmp/controller_g1_mesh
cmp /tmp/g1-ik-clearance/visual/skeleton.csv \
  /tmp/g1-ik-clearance/visual/mesh.csv
```

Check exact thresholds and lifecycle:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json, re
root = "/tmp/g1-ik-clearance/visual"
def rss(name):
    text = open(f"{root}/{name}.time", encoding="utf-8").read()
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    assert match, text
    return int(match.group(1))
stats = json.load(open(f"{root}/mesh-stats.json", encoding="utf-8"))
assert stats["frames"] == 300, stats
assert stats["visual_model_loads"] == 36, stats
assert stats["visual_model_unloads"] == 36, stats
assert stats["fallback"] is False, stats
assert stats["average_ms"] <= 5.0, stats
assert stats["p95_ms"] <= 10.0, stats
delta = rss("mesh") - rss("skeleton")
assert delta <= 524288, delta
print(f"VALID rigid-mesh ease rss_delta_kib={delta} "
      f"average_ms={stats['average_ms']} p95_ms={stats['p95_ms']}")
PY
```

Finally point an explicitly requested mesh run at a missing directory. It must exit `0`, emit exactly one fallback diagnostic, produce a byte-identical skeleton log, and report `fallback=true`, zero visual loads/unloads, and zero timed frames:

```bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TEST_MODE=route \
  MM_TEST_ROUTE=full-course MM_TEST_FRAMES=300 MM_TERRAIN_WEIGHT=4 \
  MM_IK=1 MM_G1_RENDER=mesh MM_G1_VISUAL_DIR=/tmp/g1-visual-does-not-exist \
  MM_G1_RENDER_STATS=/tmp/g1-ik-clearance/visual/fallback-stats.json \
  MM_LOG=/tmp/g1-ik-clearance/visual/fallback.csv \
  /tmp/controller_g1_mesh \
  >/tmp/g1-ik-clearance/visual/fallback.out \
  2>/tmp/g1-ik-clearance/visual/fallback.err
test "$(grep -c '^G1 visual mesh fallback:' \
  /tmp/g1-ik-clearance/visual/fallback.err)" -eq 1
cmp /tmp/g1-ik-clearance/visual/skeleton.csv \
  /tmp/g1-ik-clearance/visual/fallback.csv
/home/ubuntu/miniconda3/envs/diffsim/bin/python - <<'PY'
import json
path = "/tmp/g1-ik-clearance/visual/fallback-stats.json"
stats = json.load(open(path, encoding="utf-8"))
assert stats == {
    "average_ms": 0.0, "fallback": True, "frames": 0,
    "p95_ms": 0.0, "visual_model_loads": 0,
    "visual_model_unloads": 0,
}, stats
print("VALID rigid-mesh skeleton fallback")
PY
```

- [ ] **Step 11: Resolve the optional branch, reverify on pass, and record evidence on failure**

If and only if every ease-gate row passes, commit and fast-forward the isolated optional branch into the verified correctness branch:

```bash
git add g1_visual_mesh.h tests/cpp/test_g1_visual_mesh.cpp controller.cpp
git diff --cached --check
git commit -m "feat: render optional rigid G1 link meshes"
```

Before integration, require the optional branch diff to contain only `resources/g1_visual_mesh_source.py`, `resources/convert_g1_visual_meshes.py`, `tests/python/test_g1_visual_meshes.py`, `g1_visual_mesh.h`, `tests/cpp/test_g1_visual_mesh.cpp`, `controller.cpp`, and `.gitignore`; require generated OBJ/manifests/logs/binaries to remain untracked/ignored. Integrate using the `finishing-a-development-branch` skill and an `--ff-only` merge. Because ignored output is worktree-local, rerun the deterministic converter once in the integrated worktree to create `resources/g1_visual_mesh/`, rebuild the integrated controller at the Task 9 path `/tmp/controller_g1_ik`, and rerun every Task 9 command, including all nine certified pairs and all four stress pairs. Task 9 must remain fully green after that merge.

If any ease-gate command fails, do not integrate either optional commit or any partial renderer file. On the verified correctness branch create `docs/superpowers/evidence/2026-07-13-g1-rigid-mesh-deferral.md` with these sections and actual observed values: source XML/mesh root and attempted commit; prerequisite Task 9 checksum; mapping/conversion counts; the first failed ease-gate name and exact command/exit status; RSS/mean/p95/load/unload measurements that were available; confirmation that skeleton fallback and all correctness/Gate E results remain green; and the decision to defer rigid meshes. Do not include unresolved tokens or hypothetical numbers. Commit only that note:

```bash
git add docs/superpowers/evidence/2026-07-13-g1-rigid-mesh-deferral.md
git diff --cached --check
git commit -m "docs: record G1 rigid mesh ease-gate result"
```

The deferral commit is a successful completion of this optional task, not a Gate E failure.

- [ ] **Step 12: Perform the one final verified `DISPLAY=:1` launch**

Run exactly one branch after all verification/optional resolution is complete.

If rigid meshes passed and were integrated:

```bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  MM_G1_RENDER=mesh MM_G1_VISUAL_DIR=resources/g1_visual_mesh \
  /tmp/controller_g1_ik \
  >/tmp/g1-ik-clearance/final-live.out \
  2>/tmp/g1-ik-clearance/final-live.err &
final_pid=$!
echo "$final_pid" > /tmp/g1-ik-clearance/final-live.pid
sleep 2
kill -0 "$final_pid"
test ! -s /tmp/g1-ik-clearance/final-live.err
```

If the optional task was skipped or deferred, launch the unchanged verified skeleton build without any mesh-only environment variable:

```bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TERRAIN_SCENE=mixed-multilevel MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  /tmp/controller_g1_ik \
  >/tmp/g1-ik-clearance/final-live.out \
  2>/tmp/g1-ik-clearance/final-live.err &
final_pid=$!
echo "$final_pid" > /tmp/g1-ik-clearance/final-live.pid
sleep 2
kill -0 "$final_pid"
test ! -s /tmp/g1-ik-clearance/final-live.err
```

Expected: one approved mixed-multilevel IK-on controller remains open on `DISPLAY=:1` at the fixed 25 Hz target, with either the proven rigid links or the accepted skeleton fallback. Record its PID and do not start a second live instance.
