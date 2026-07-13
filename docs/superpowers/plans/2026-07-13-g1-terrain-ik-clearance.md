# G1 Terrain IK and Clearance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reversible, bounded G1 terrain IK that locks planted feet to the authoritative scene surface, aligns feet to its normals, clears swing feet over certified terrain, hands unreachable targets to the existing safe-stop path, and passes Gate E without changing matching or the accepted support-retargeted root.

**Architecture:** The completed scene-artifact and multiscene-support-runtime plans remain authoritative for G1HF/v2 sampling, G1SP/G1WM data, scene routes, support retargeting, traversability, logging, reset, and controlled cleanup. This plan adds renderer-independent G1 leg, IK, and clearance modules downstream of the support-retargeted IK-off pose; it copies that pose into dedicated IK output buffers, changes only named leg rotations, and exposes diagnostics and safe-stop requests without writing matcher, trajectory, planar-root, or support-frame state. Paired deterministic route logs prove byte-identical matching/support columns and bounded physical corrections.

**Tech Stack:** C++17 header-only math and existing `array.h`, `vec.h`, `quat.h`, `spring.h`, `database.h`, `g1_skeleton.h`, `terrain_runtime.h`, `scene_runtime.h`, `support_runtime.h`, and `g1_controller_state.h`; Raylib/Raygui from `/home/ubuntu/apps`; Python 3 standard-library CSV checks; `DISPLAY=:1` deterministic route runs.

## Global Constraints

- Do not start Task 1 until every task and final gate in `docs/superpowers/plans/2026-07-13-g1-scene-artifacts.md` and `docs/superpowers/plans/2026-07-13-g1-multiscene-support-runtime.md` passes with IK disabled. Preserve their accepted IK-off CSVs under `/tmp/g1-multiscene-runtime/` as the Gate C/D/F baselines.
- Consume, do not duplicate or replace, the sibling plans' `heightfield_sample`, `heightfield_normal`, `terrain_support_set`, `walkability_grid`, `scene_pack`, `scene_route`, `support_frame_state`, `support_pose_apply`, controller reset, deterministic route, logger/checker, traversability, scene-switch, and controlled-cleanup implementations.
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
- `tests/cpp/test_g1_ik.cpp`: explicit geometry, locks, normals, named solve, orientation, correction, reset, reversibility, and error-path tests.
- `tests/cpp/test_g1_clearance.cpp`: point/sole/capsule/sweep diagnostics and safe-stop threshold tests on synthetic G1HF/v2 surfaces.
- `tests/cpp/test_g1_controller_state.cpp`: scene reset/swap coverage for all IK state and pose buffers.
- `tests/python/test_runtime_log.py`: synthetic IK schema, invariance, drift, penetration, safe-stop, route-matrix, and controlled-error checker regressions.
- Optional only: `resources/convert_g1_visual_meshes.py`, `tests/python/test_g1_visual_meshes.py`, `g1_visual_mesh.h`, `.gitignore`, and either `docs/superpowers/evidence/2026-07-13-g1-rigid-mesh-deferral.md` or passing render integration in `controller.cpp`.

## Consumed Cross-Plan Contracts

Before implementation, compare the completed sibling code with this block. The producer names and types win; if a sibling plan changed one while being implemented, update this document consistently before Task 1 rather than adding an adapter or duplicate implementation.

```cpp
// terrain_runtime.h and scene_runtime.h, produced by sibling plans
struct heightfield {
    uint32_t version;
    int nx, nz;
    float origin_x, origin_z, cell_size, exterior_height;
    array1d<float> heights;
};
float heightfield_sample(const heightfield&, float x, float z);
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

**Interfaces:**
- Consumes: `G1Bone`, the exact 31-bone parent array, `database`, `vec3`, and finite helpers from `terrain_runtime.h`.
- Produces: `G1LegConfig g1_left_leg_config()` and `G1LegConfig g1_right_leg_config()`.
- Produces: `bool g1_leg_configs_validate(const database&, char*, int)`; a bad skeleton/config is a startup error before Raylib.
- Locks the MuJoCo-to-Holden local basis `(x,y,z) -> (x,z,-y)`, including four sole probes and thigh/shin capsules. It does not load or parse XML at runtime.

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

static database make_g1_database()
{
    static const int parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,
        15,16,17,18,19,20,21,22,16,24,25,26,27,28,29
    };
    database db;
    db.bone_positions.resize(1, G1_BoneCount);
    db.bone_rotations.resize(1, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    db.bone_positions.zero();
    db.bone_rotations.set(quat());
    for (int i = 0; i < G1_BoneCount; ++i) db.bone_parents(i) = parents[i];
    return db;
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
    check(left.knee_pole_local.z == -1.0f && right.knee_pole_local.z == -1.0f,
          "MuJoCo +Y knee pole maps to Holden -Z");
    check(left.foot_forward_local.x == 1.0f && left.sole_normal_local.y == 1.0f,
          "foot axes");
    check(left.sole_points_local[0].x == -0.05f &&
          left.sole_points_local[0].y == -0.05f &&
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

    database db = make_g1_database();
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

#include "database.h"
#include "g1_skeleton.h"
#include "terrain_runtime.h"

#include <cstdarg>
#include <cfloat>
#include <cstdio>

struct G1LegConfig
{
    const char* name;
    int hip;
    int knee;
    int ankle;
    int contact;
    vec3 knee_pole_local;
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

static inline bool g1_ik_error(char* output, int capacity, const char* format, ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(output, static_cast<size_t>(capacity), format, arguments);
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
    config.knee_pole_local = vec3(0.0f, 0.0f, -1.0f);
    config.foot_forward_local = vec3(1.0f, 0.0f, 0.0f);
    config.sole_normal_local = vec3(0.0f, 1.0f, 0.0f);
    config.foot_sphere_radius_m = 0.02f;
    config.foot_sphere_centers_local[0] = vec3(-0.05f, -0.03f, -0.025f);
    config.foot_sphere_centers_local[1] = vec3(-0.05f, -0.03f, +0.025f);
    config.foot_sphere_centers_local[2] = vec3(+0.12f, -0.03f, -0.030f);
    config.foot_sphere_centers_local[3] = vec3(+0.12f, -0.03f, +0.030f);
    for (int i = 0; i < 4; ++i)
        config.sole_points_local[i] =
            config.foot_sphere_centers_local[i] -
            config.sole_normal_local * config.foot_sphere_radius_m;
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

static inline bool g1_leg_config_validate(
    const database& db, const G1LegConfig& config,
    char* error, int error_capacity)
{
    if (db.bone_parents.size != G1_BoneCount ||
        config.hip < 0 || config.hip >= G1_BoneCount ||
        config.knee < 0 || config.knee >= G1_BoneCount ||
        config.ankle < 0 || config.ankle >= G1_BoneCount ||
        config.contact < 0 || config.contact >= G1_BoneCount) {
        return g1_ik_error(error, error_capacity, "%s leg: invalid bone contract", config.name);
    }
    if (db.bone_parents(config.knee) != config.hip)
        return g1_ik_error(
            error, error_capacity, "%s %s parent mismatch", config.name,
            config.knee == G1_LeftKnee ? "LeftKnee" : "RightKnee");
    if (db.bone_parents(config.ankle) != config.knee)
        return g1_ik_error(error, error_capacity, "%s ankle parent mismatch", config.name);
    if (db.bone_parents(config.contact) != config.ankle)
        return g1_ik_error(error, error_capacity, "%s contact parent mismatch", config.name);
    return true;
}

static inline bool g1_leg_configs_validate(
    const database& db, char* error, int error_capacity)
{
    return g1_leg_config_validate(db, g1_left_leg_config(), error, error_capacity) &&
           g1_leg_config_validate(db, g1_right_leg_config(), error, error_capacity);
}
```

The XML positions are centers of four `0.02 m` collision spheres. Each stored
sole probe is the corresponding mapped center minus
`foot_sphere_radius_m * sole_normal_local`, so it represents the physical
bottom/support point rather than falsely treating the sphere center as the
sole. The two capsule definitions use their XML centerlines and retain their
radii. All values come from
`/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml`, mapped from
MuJoCo Z-up to Holden Y-up with `(x,y,z) -> (x,z,-y)`.
`LeftAnkle/RightAnkle` map to the ankle-pitch bodies; `LeftToe/RightToe` map to
the ankle-roll bodies. Do not add an XML parser to the runtime.

- [ ] **Step 4: Run the named-geometry GREEN test**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
/tmp/test_g1_ik
```

Expected: compilation succeeds and the test exits `0` with no output.

- [ ] **Step 5: Commit the explicit geometry contract**

```bash
git add g1_ik.h tests/cpp/test_g1_ik.cpp
git commit -m "test: lock explicit G1 terrain IK geometry"
```

### Task 2: Sample Exact Surface Normals and Lock Planted Sole Targets at 25 Hz

**Files:**
- Modify: `g1_ik.h`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Consumes: sibling-owned `heightfield_sample(const heightfield&,float,float)` and `heightfield_normal(const heightfield&,float,float)` on a validated G1HF/v2 scene.
- Produces: `G1SurfaceTarget`, `G1FootLockState`, `g1_foot_lock_reset`, and transactional `g1_foot_lock_update`.
- A recorded contact rising edge freezes the support-retargeted sole-center XZ, samples height and normal at that exact point, and drives a `0.10 s` critically damped target at `1/25 s`; release returns to the animation target without changing the input pose.
- Observation runs in IK-off and IK-on modes so the paired logs use identical contact edges and lock anchors. Only a subsequent apply call may alter rotations.

- [ ] **Step 1: Add failing flat, ramp-normal, lock, release, and timestep tests**

Add `#include <limits>` and append these helpers/tests before `main` in `tests/cpp/test_g1_ik.cpp`; call `test_surface_target_and_planted_lock()` from `main`:

```cpp
static heightfield make_ramp_surface()
{
    heightfield field;
    field.version = 2;
    field.nx = 2;
    field.nz = 2;
    field.origin_x = 0.0f;
    field.origin_z = 0.0f;
    field.cell_size = 1.0f;
    field.exterior_height = -10.0f;
    field.heights.resize(4);
    field.heights(0) = 0.0f;
    field.heights(1) = 0.1f;
    field.heights(2) = 0.0f;
    field.heights(3) = 0.1f;
    return field;
}

static void test_surface_target_and_planted_lock()
{
    const heightfield field = make_ramp_surface();
    const G1LegConfig leg = g1_left_leg_config();
    G1SurfaceTarget surface = {};
    char error[256] = {};
    check(g1_surface_target_sample(
              surface, field, 0.25f, 0.50f,
              leg.planted_clearance_m, error, sizeof(error)), error);
    check(std::fabs(surface.point.y - 0.030f) < 1e-6f,
          "surface height plus planted clearance");
    check(surface.normal.y > 0.99f && surface.normal.x < 0.0f,
          "upward exact ramp normal");

    G1FootLockState state = {};
    const vec3 initial(0.25f, 0.030f, 0.50f);
    g1_foot_lock_reset(state, initial);
    G1FootTarget target = {};
    check(g1_foot_lock_update(
              state, target, field, leg, initial, true,
              1.0f / 25.0f, error, sizeof(error)), error);
    check(state.locked && target.locked, "contact rising edge locks");
    check(std::fabs(state.lock_point.x - 0.25f) < 1e-7f,
          "lock X is frozen");
    check(std::fabs(state.lock_point.z - 0.50f) < 1e-7f,
          "lock Z is frozen");

    for (int frame = 0; frame < 12; ++frame) {
        const vec3 drifting(0.30f + frame * 0.002f, 0.04f, 0.50f);
        check(g1_foot_lock_update(
                  state, target, field, leg, drifting, true,
                  1.0f / 25.0f, error, sizeof(error)), error);
    }
    check(target.locked && std::fabs(target.surface.point.x - 0.25f) < 1e-7f,
          "planted target remains at lock point");
    check(target.surface.normal.x < 0.0f && target.surface.normal.y > 0.99f,
          "lock retains exact normal");

    check(g1_foot_lock_update(
              state, target, field, leg, vec3(0.46f, 0.04f, 0.50f), true,
              1.0f / 25.0f, error, sizeof(error)), error);
    check(state.locked && target.locked && target.drift_limit_exceeded,
          "active contact never silently unlocks beyond drift bound");

    check(g1_foot_lock_update(
              state, target, field, leg, vec3(0.32f, 0.04f, 0.50f), false,
              1.0f / 25.0f, error, sizeof(error)), error);
    check(!state.locked && !target.locked, "contact falling edge releases");

    const G1FootLockState before = state;
    check(!g1_foot_lock_update(
              state, target, field, leg, initial, true,
              1.0f / 60.0f, error, sizeof(error)),
          "non-25 Hz contact update rejected");
    check(std::strstr(error, "25 Hz") != NULL, "timestep diagnostic");
    check(state.locked == before.locked && state.contact == before.contact,
          "failed update is transactional");

    const float nan = std::numeric_limits<float>::quiet_NaN();
    check(!g1_surface_target_sample(
              surface, field, nan, 0.0f, 0.0f, error, sizeof(error)),
          "non-finite query rejected");
    check(std::strstr(error, "non-finite") != NULL,
          "non-finite query diagnostic");
}
```

- [ ] **Step 2: Compile to verify lock/normal RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
```

Expected: compilation fails because `G1SurfaceTarget`, `G1FootLockState`, `G1FootTarget`, and the lock functions are undefined.

- [ ] **Step 3: Add transactional exact-surface lock state**

Add `#include "spring.h"` and `<cmath>` to `g1_ik.h`, then append:

```cpp
struct G1SurfaceTarget
{
    vec3 point;
    vec3 normal;
};

struct G1FootLockState
{
    bool initialized = false;
    bool contact = false;
    bool locked = false;
    vec3 previous_input;
    vec3 lock_point;
    vec3 output_position;
    vec3 output_velocity;
    vec3 offset_position;
    vec3 offset_velocity;
};

struct G1FootTarget
{
    bool locked = false;
    bool drift_limit_exceeded = false;
    G1SurfaceTarget surface;
    vec3 sole_center;
    float horizontal_drift_m = 0.0f;
};

static inline bool g1_vec3_is_finite(vec3 value)
{
    return terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline bool g1_surface_target_sample(
    G1SurfaceTarget& output,
    const heightfield& field,
    float x,
    float z,
    float clearance,
    char* error,
    int error_capacity)
{
    if (field.version != 2 || !terrain_float_is_finite(x) ||
        !terrain_float_is_finite(z) ||
        !terrain_float_is_finite(clearance) || clearance < 0.0f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query is non-finite or not G1HF/v2");
    }
    const float height = heightfield_sample(field, x, z);
    const vec3 normal = heightfield_normal(field, x, z);
    if (!terrain_float_is_finite(height) || !g1_vec3_is_finite(normal) ||
        normal.y <= 0.0f || std::fabs(length(normal) - 1.0f) > 1e-4f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK surface query produced a non-finite or non-upward normal");
    }
    G1SurfaceTarget candidate = {};
    candidate.point = vec3(x, height + clearance, z);
    candidate.normal = normal;
    output = candidate;
    return true;
}

static inline void g1_foot_lock_reset(
    G1FootLockState& state, vec3 initial_sole_center)
{
    state = G1FootLockState();
    state.initialized = true;
    state.previous_input = initial_sole_center;
    state.lock_point = initial_sole_center;
    state.output_position = initial_sole_center;
}

static inline bool g1_foot_lock_update(
    G1FootLockState& state,
    G1FootTarget& output,
    const heightfield& field,
    const G1LegConfig& config,
    vec3 input_sole_center,
    bool input_contact,
    float dt,
    char* error,
    int error_capacity)
{
    if (!state.initialized || !g1_vec3_is_finite(input_sole_center) ||
        !terrain_float_is_finite(dt) ||
        std::fabs(dt - 1.0f / 25.0f) > 1e-7f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update requires finite state at exactly 25 Hz");
    }

    G1FootLockState next = state;
    const vec3 input_velocity =
        (input_sole_center - next.previous_input) / dt;
    next.previous_input = input_sole_center;
    G1SurfaceTarget sampled = {};

    if (!next.contact && input_contact) {
        if (!g1_surface_target_sample(
                sampled, field, input_sole_center.x, input_sole_center.z,
                config.planted_clearance_m, error, error_capacity)) return false;
        next.locked = true;
        next.lock_point = sampled.point;
        inertialize_transition(
            next.offset_position, next.offset_velocity,
            next.output_position, next.output_velocity,
            next.lock_point, vec3());
    } else if (next.locked && !input_contact) {
        next.locked = false;
        inertialize_transition(
            next.offset_position, next.offset_velocity,
            next.output_position, next.output_velocity,
            input_sole_center, input_velocity);
    }

    const vec3 goal = next.locked ? next.lock_point : input_sole_center;
    const vec3 goal_velocity = next.locked ? vec3() : input_velocity;
    inertialize_update(
        next.output_position, next.output_velocity,
        next.offset_position, next.offset_velocity,
        goal, goal_velocity, 0.10f, dt);
    next.contact = input_contact;

    if (!g1_surface_target_sample(
            sampled, field,
            next.locked ? next.lock_point.x : input_sole_center.x,
            next.locked ? next.lock_point.z : input_sole_center.z,
            config.planted_clearance_m, error, error_capacity)) return false;
    const vec3 drift(
        input_sole_center.x - next.lock_point.x,
        0.0f,
        input_sole_center.z - next.lock_point.z);
    G1FootTarget candidate = {};
    candidate.locked = next.locked;
    candidate.drift_limit_exceeded =
        next.locked && length(drift) > 0.20f;
    candidate.surface = sampled;
    candidate.sole_center = next.output_position;
    candidate.horizontal_drift_m = next.locked ? length(drift) : 0.0f;
    if (!g1_vec3_is_finite(candidate.sole_center) ||
        !terrain_float_is_finite(candidate.horizontal_drift_m)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 planted-foot update produced non-finite state");
    }
    state = next;
    output = candidate;
    return true;
}

```

Do not call these functions from `controller.cpp` in this task. They are pure downstream observation state until controller integration, and their target is a sole-center target rather than a world-Y root correction.

- [ ] **Step 4: Run lock/normal GREEN under strict, release, and sanitizers**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_strict
/tmp/test_g1_ik_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_release
/tmp/test_g1_ik_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_g1_ik.cpp \
  -o /tmp/test_g1_ik_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_g1_ik_san
```

Expected: all three executables exit `0`; the sanitizer emits no finding.

- [ ] **Step 5: Commit exact-surface planted locking**

```bash
git add g1_ik.h tests/cpp/test_g1_ik.cpp
git commit -m "feat: lock planted G1 feet to exact terrain"
```

### Task 3: Add a Finite, Reach-Shell-Bounded Named Two-Bone Solve

**Files:**
- Create: `ik.h`
- Modify: `g1_ik.h`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Produces: `IKTargetProjection`, `IKTwoBoneResult`, `ik_project_target`, `ik_two_bone_bounded`, and `ik_clamp_local_delta` in renderer-independent `ik.h`.
- Produces: `G1LegSolveResult`, `g1_apply_named_position_ik`, and the
  offset-aware `g1_apply_named_contact_position_ik` in `g1_ik.h`.
- The G1 adapter uses only `config.hip`, `config.knee`, `config.ankle`, and the
  direct contact child offset. It changes only hip/knee local rotations in a
  caller-provided copy and performs at most four bounded FK-residual iterations
  until contact-position error is at most `0.005 m`.
- An outside-shell or correction-limited target returns a finite closest bounded pose with `safe_stop_requested=true`. Invalid math returns `false` and no output mutation for controlled diagnostic exit.

- [ ] **Step 1: Add failing reachable, outside-shell, untouched-bone, and non-finite tests**

Add these functions before `main` in `tests/cpp/test_g1_ik.cpp`, then call `test_named_bounded_two_bone_solve()` from `main`:

```cpp
static bool same_quat(quat left, quat right)
{
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

static void test_named_bounded_two_bone_solve()
{
    database db = make_g1_database();
    db.bone_positions(0, G1_LeftKnee) = vec3(0.0f, -0.40f, 0.0f);
    db.bone_positions(0, G1_LeftAnkle) = vec3(0.0f, -0.40f, 0.0f);
    const G1LegConfig leg = g1_left_leg_config();
    array1d<quat> output = db.bone_rotations(0);
    char error[256] = {};
    G1LegSolveResult result = {};
    check(g1_apply_named_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(0.20f, -0.70f, 0.0f),
              result, error, sizeof(error)), error);
    check(result.applied && result.reachable,
          "reachable target applies");
    check(result.safe_stop_requested == result.correction_limited,
          "reachable target stops only when correction is limited");
    check(result.max_correction_radians <= 0.350001f,
          "reachable solve correction bound");
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (bone != leg.hip && bone != leg.knee)
            check(same_quat(output(bone), db.bone_rotations(0, bone)),
                  "non-solver local rotation is byte-identical");
    }

    db.bone_positions(0, G1_LeftToe) = vec3(0.0f, -0.017558f, 0.0f);
    output = db.bone_rotations(0);
    const vec3 desired_contact(0.16f, -0.71f, 0.02f);
    check(g1_apply_named_contact_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, desired_contact,
              result, error, sizeof(error)), error);
    array1d<vec3> contact_global_positions(G1_BoneCount);
    array1d<quat> contact_global_rotations(G1_BoneCount);
    forward_kinematics_full(
        contact_global_positions, contact_global_rotations,
        db.bone_positions(0), output, db.bone_parents);
    check(length(contact_global_positions(leg.contact) - desired_contact) <=
          0.005f + 1e-6f,
          "nonzero ankle-to-contact offset is solved by FK residual");
    check(result.contact_residual_m <= 0.005f + 1e-6f,
          "contact residual is explicit");

    output = db.bone_rotations(0);
    check(g1_apply_named_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(3.0f, 0.0f, 0.0f),
              result, error, sizeof(error)), error);
    check(result.applied && !result.reachable && result.safe_stop_requested,
          "outside-shell target uses bounded safe-stop result");
    check(std::fabs(result.clamped_distance_m - 0.785f) < 1e-5f,
          "outer shell is two links minus 15 mm");
    check(result.max_correction_radians <= 0.350001f,
          "unreachable solve correction bound");

    output = db.bone_rotations(0);
    const array1d<quat> before = output;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    check(!g1_apply_named_position_ik(
              output, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, leg, vec3(nan, 0.0f, 0.0f),
              result, error, sizeof(error)),
          "non-finite solve rejected");
    for (int bone = 0; bone < G1_BoneCount; ++bone)
        check(same_quat(output(bone), before(bone)),
              "failed solve leaves output unchanged");
    check(std::strstr(error, "non-finite") != NULL,
          "non-finite solve diagnostic");

    IKTargetProjection folded = {};
    check(ik_project_target(
              folded, vec3(), vec3(0.0f, -0.40f, 0.0f), vec3(),
              vec3(), 0.015f),
          "folded zero-direction target uses deterministic fallback");
    check(!folded.reachable &&
          g1_vec3_is_finite(folded.clamped_target) &&
          std::fabs(folded.clamped_distance_m - 0.015f) < 1e-6f,
          "folded target projection remains finite at inner shell");
}
```

- [ ] **Step 2: Compile to verify bounded-solver RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
```

Expected: compilation fails because `G1LegSolveResult` and `g1_apply_named_position_ik` are undefined.

- [ ] **Step 3: Implement robust generic two-bone math**

Create `ik.h`:

```cpp
#pragma once

#include "quat.h"
#include "terrain_runtime.h"

#include <cmath>

struct IKTargetProjection
{
    bool reachable = false;
    vec3 clamped_target;
    float raw_distance_m = 0.0f;
    float clamped_distance_m = 0.0f;
    float minimum_distance_m = 0.0f;
    float maximum_distance_m = 0.0f;
};

struct IKTwoBoneResult
{
    bool applied = false;
    bool reachable = false;
    bool correction_limited = false;
    quat root_local;
    quat middle_local;
    float root_correction_radians = 0.0f;
    float middle_correction_radians = 0.0f;
    IKTargetProjection target;
};

static inline bool ik_quat_is_finite(quat value)
{
    return terrain_float_is_finite(value.w) &&
           terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline bool ik_vec_is_finite(vec3 value)
{
    return terrain_float_is_finite(value.x) &&
           terrain_float_is_finite(value.y) &&
           terrain_float_is_finite(value.z);
}

static inline quat ik_clamp_local_delta(
    quat baseline, quat desired, float maximum_radians,
    float& requested_radians)
{
    requested_radians = quat_angle_between(baseline, desired);
    if (requested_radians <= maximum_radians) return quat_normalize(desired);
    return quat_slerp_shortest(
        baseline, desired, maximum_radians / requested_radians);
}

static inline bool ik_project_target(
    IKTargetProjection& output,
    vec3 root,
    vec3 middle,
    vec3 end,
    vec3 requested,
    float reach_buffer_m)
{
    if (!ik_vec_is_finite(root) || !ik_vec_is_finite(middle) ||
        !ik_vec_is_finite(end) || !ik_vec_is_finite(requested) ||
        !terrain_float_is_finite(reach_buffer_m) || reach_buffer_m < 0.0f)
        return false;
    const float upper = length(middle - root);
    const float lower = length(end - middle);
    if (upper <= reach_buffer_m || lower <= reach_buffer_m) return false;
    const float minimum = std::fabs(upper - lower) + reach_buffer_m;
    const float maximum = upper + lower - reach_buffer_m;
    if (!(minimum <= maximum)) return false;
    const vec3 delta = requested - root;
    const float distance = length(delta);
    vec3 direction = vec3(0.0f, -1.0f, 0.0f);
    if (distance > 1e-7f) {
        direction = delta / distance;
    } else {
        const vec3 current = end - root;
        const float current_distance = length(current);
        if (current_distance > 1e-7f) direction = current / current_distance;
    }
    if (!terrain_float_is_finite(distance) ||
        !ik_vec_is_finite(direction)) return false;
    IKTargetProjection candidate = {};
    candidate.raw_distance_m = distance;
    candidate.minimum_distance_m = minimum;
    candidate.maximum_distance_m = maximum;
    candidate.clamped_distance_m = clampf(distance, minimum, maximum);
    candidate.clamped_target = root + direction * candidate.clamped_distance_m;
    candidate.reachable = distance >= minimum && distance <= maximum;
    output = candidate;
    return true;
}

static inline vec3 ik_safe_perpendicular(vec3 direction)
{
    const vec3 axis = std::fabs(direction.x) < 0.75f
        ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 0.0f, 1.0f);
    return normalize(cross(direction, axis));
}

static inline quat ik_between_unit(vec3 from, vec3 to)
{
    const float cosine = clampf(dot(from, to), -1.0f, 1.0f);
    if (cosine < -0.99999f)
        return quat_from_angle_axis(PIf, ik_safe_perpendicular(from));
    if (cosine > 0.99999f) return quat();
    return quat_between(from, to);
}

static inline bool ik_two_bone_bounded(
    IKTwoBoneResult& output,
    quat root_local_before,
    quat middle_local_before,
    vec3 root,
    vec3 middle,
    vec3 end,
    vec3 requested_target,
    vec3 pole_world,
    quat root_global,
    quat middle_global,
    quat root_parent_global,
    float reach_buffer_m,
    float maximum_correction_radians)
{
    IKTargetProjection projection = {};
    if (!ik_project_target(
            projection, root, middle, end, requested_target, reach_buffer_m) ||
        !ik_vec_is_finite(pole_world) || !ik_quat_is_finite(root_global) ||
        !ik_quat_is_finite(middle_global) ||
        !ik_quat_is_finite(root_parent_global) ||
        !terrain_float_is_finite(maximum_correction_radians) ||
        maximum_correction_radians <= 0.0f) return false;

    const float upper = length(middle - root);
    const float lower = length(end - middle);
    const float target_distance = projection.clamped_distance_m;
    const vec3 target_direction = normalize(projection.clamped_target - root);
    vec3 bend_direction = pole_world -
        target_direction * dot(pole_world, target_direction);
    if (length(bend_direction) < 1e-6f) {
        bend_direction = (middle - root) -
            target_direction * dot(middle - root, target_direction);
    }
    if (length(bend_direction) < 1e-6f)
        bend_direction = ik_safe_perpendicular(target_direction);
    bend_direction = normalize(bend_direction);

    const float knee_along =
        (upper * upper + target_distance * target_distance - lower * lower) /
        (2.0f * target_distance);
    const float knee_height = std::sqrt(maxf(
        upper * upper - knee_along * knee_along, 0.0f));
    const vec3 desired_knee = root +
        target_direction * knee_along + bend_direction * knee_height;
    const vec3 desired_lower = projection.clamped_target - desired_knee;
    if (length(desired_knee - root) < 1e-7f ||
        length(desired_lower) < 1e-7f) return false;

    const quat root_delta = ik_between_unit(
        normalize(middle - root), normalize(desired_knee - root));
    const quat desired_root_global = quat_mul(root_delta, root_global);
    const vec3 root_rotated_lower =
        quat_mul_vec3(root_delta, end - middle);
    const quat middle_delta = ik_between_unit(
        normalize(root_rotated_lower), normalize(desired_lower));
    const quat desired_middle_global = quat_mul(
        middle_delta, quat_mul(root_delta, middle_global));
    const quat desired_root = quat_inv_mul(
        root_parent_global, desired_root_global);
    const quat desired_middle = quat_inv_mul(
        desired_root_global, desired_middle_global);

    float root_requested = 0.0f;
    float middle_requested = 0.0f;
    const quat bounded_root = ik_clamp_local_delta(
        root_local_before, desired_root,
        maximum_correction_radians, root_requested);
    const quat bounded_middle = ik_clamp_local_delta(
        middle_local_before, desired_middle,
        maximum_correction_radians, middle_requested);
    if (!ik_quat_is_finite(bounded_root) || !ik_quat_is_finite(bounded_middle))
        return false;

    IKTwoBoneResult candidate = {};
    candidate.applied = true;
    candidate.reachable = projection.reachable;
    candidate.correction_limited =
        root_requested > maximum_correction_radians + 1e-6f ||
        middle_requested > maximum_correction_radians + 1e-6f;
    candidate.root_local = bounded_root;
    candidate.middle_local = bounded_middle;
    candidate.root_correction_radians =
        quat_angle_between(root_local_before, bounded_root);
    candidate.middle_correction_radians =
        quat_angle_between(middle_local_before, bounded_middle);
    candidate.target = projection;
    output = candidate;
    return true;
}

```

This replaces the dormant controller math rather than adding another active solver. Keep the controller definitions untouched until Task 7 switches the call site atomically.

- [ ] **Step 4: Add the named G1 adapter without pose side effects**

Include `ik.h` from `g1_ik.h`, then append:

```cpp
struct G1LegSolveResult
{
    bool applied = false;
    bool reachable = false;
    bool correction_limited = false;
    bool safe_stop_requested = false;
    vec3 requested_ankle_target;
    vec3 clamped_ankle_target;
    float raw_distance_m = 0.0f;
    float clamped_distance_m = 0.0f;
    float max_correction_radians = 0.0f;
    float contact_residual_m = FLT_MAX;
};

static inline bool g1_apply_named_position_ik(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 requested_ankle_target,
    G1LegSolveResult& output,
    char* error,
    int error_capacity)
{
    if (output_rotations.size != G1_BoneCount ||
        local_positions.size != G1_BoneCount ||
        baseline_rotations.size != G1_BoneCount ||
        parents.size != G1_BoneCount ||
        !g1_vec3_is_finite(requested_ankle_target)) {
        return g1_ik_error(
            error, error_capacity,
            "%s leg IK received non-finite target or wrong pose shape",
            config.name);
    }
    array1d<vec3> global_positions(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    forward_kinematics_full(
        global_positions, global_rotations,
        local_positions, baseline_rotations, parents);
    const int hip_parent = parents(config.hip);
    if (hip_parent < 0) {
        return g1_ik_error(
            error, error_capacity, "%s leg IK hip has no parent", config.name);
    }
    IKTwoBoneResult solve = {};
    if (!ik_two_bone_bounded(
            solve,
            baseline_rotations(config.hip),
            baseline_rotations(config.knee),
            global_positions(config.hip),
            global_positions(config.knee),
            global_positions(config.ankle),
            requested_ankle_target,
            quat_mul_vec3(
                global_rotations(config.knee), config.knee_pole_local),
            global_rotations(config.hip),
            global_rotations(config.knee),
            global_rotations(hip_parent),
            config.reach_buffer_m,
            config.max_correction_radians)) {
        return g1_ik_error(
            error, error_capacity,
            "%s leg IK produced non-finite two-bone math", config.name);
    }

    output_rotations(config.hip) = solve.root_local;
    output_rotations(config.knee) = solve.middle_local;
    G1LegSolveResult candidate = {};
    candidate.applied = true;
    candidate.reachable = solve.reachable;
    candidate.correction_limited = solve.correction_limited;
    candidate.safe_stop_requested =
        !solve.reachable || solve.correction_limited;
    candidate.requested_ankle_target = requested_ankle_target;
    candidate.clamped_ankle_target = solve.target.clamped_target;
    candidate.raw_distance_m = solve.target.raw_distance_m;
    candidate.clamped_distance_m = solve.target.clamped_distance_m;
    candidate.max_correction_radians = maxf(
        solve.root_correction_radians,
        solve.middle_correction_radians);
    output = candidate;
    return true;
}

static inline bool g1_apply_named_contact_position_ik(
    slice1d<quat> output_rotations,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const G1LegConfig& config,
    vec3 desired_contact,
    G1LegSolveResult& output,
    char* error,
    int error_capacity)
{
    if (parents(config.contact) != config.ankle ||
        !g1_vec3_is_finite(desired_contact))
        return g1_ik_error(
            error, error_capacity,
            "%s contact residual solver received invalid chain/target",
            config.name);
    array1d<quat> candidate = output_rotations;
    array1d<vec3> global_positions(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    forward_kinematics_full(
        global_positions, global_rotations,
        local_positions, baseline_rotations, parents);
    vec3 ankle_target = desired_contact -
        (global_positions(config.contact) - global_positions(config.ankle));
    G1LegSolveResult aggregate = {};
    bool all_reachable = true;
    bool any_limited = false;
    float maximum_correction = 0.0f;
    for (int iteration = 0; iteration < 4; ++iteration) {
        G1LegSolveResult solve = {};
        if (!g1_apply_named_position_ik(
                candidate, local_positions, baseline_rotations, parents,
                config, ankle_target, solve, error, error_capacity))
            return false;
        aggregate = solve;
        all_reachable = all_reachable && solve.reachable;
        any_limited = any_limited || solve.correction_limited;
        maximum_correction = maxf(
            maximum_correction, solve.max_correction_radians);
        forward_kinematics_full(
            global_positions, global_rotations,
            local_positions, candidate, parents);
        const vec3 residual =
            desired_contact - global_positions(config.contact);
        aggregate.contact_residual_m = length(residual);
        if (!terrain_float_is_finite(aggregate.contact_residual_m))
            return g1_ik_error(
                error, error_capacity,
                "%s contact residual became non-finite", config.name);
        if (aggregate.contact_residual_m <= 0.001f) break;
        ankle_target += residual;
    }
    aggregate.reachable = all_reachable;
    aggregate.correction_limited = any_limited;
    aggregate.max_correction_radians = maximum_correction;
    aggregate.safe_stop_requested = aggregate.safe_stop_requested ||
        !all_reachable || any_limited ||
        aggregate.contact_residual_m > 0.005f;
    for (int bone = 0; bone < G1_BoneCount; ++bone)
        output_rotations(bone) = candidate(bone);
    output = aggregate;
    return true;
}
```

- [ ] **Step 5: Run bounded-solver GREEN in strict, release, and sanitizer modes**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_strict
/tmp/test_g1_ik_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_release
/tmp/test_g1_ik_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_g1_ik.cpp \
  -o /tmp/test_g1_ik_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_g1_ik_san
```

Expected: all executables exit `0`; no sanitizer finding; reachable and unreachable cases remain finite and within `0.35` radians.

- [ ] **Step 6: Commit the bounded named solver**

```bash
git add ik.h g1_ik.h tests/cpp/test_g1_ik.cpp
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
    database db = make_g1_database();
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
    if (!ik_quat_is_finite(current_global_rotation) ||
        !g1_vec3_is_finite(surface_normal) || surface_normal.y <= 0.0f ||
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
    if (!ik_quat_is_finite(target)) {
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
    float requested = 0.0f;
    const quat bounded = ik_clamp_local_delta(
        baseline_rotations(config.contact), desired_local,
        config.max_correction_radians, requested);
    if (!ik_quat_is_finite(bounded)) {
        return g1_ik_error(
            error, error_capacity,
            "%s foot orientation produced non-finite correction", config.name);
    }
    output_rotations(config.contact) = bounded;
    G1FootOrientationResult candidate = {};
    candidate.applied = true;
    candidate.correction_limited =
        requested > config.max_correction_radians + 1e-6f;
    candidate.safe_stop_requested = candidate.correction_limited;
    candidate.target_global_rotation = target_global;
    candidate.requested_correction_radians = requested;
    candidate.correction_radians = quat_angle_between(
        baseline_rotations(config.contact), bounded);
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
- Create: `g1_clearance.h`
- Create: `tests/cpp/test_g1_clearance.cpp`

**Interfaces:**
- Produces: `g1_point_clearance`, `g1_sole_points_world`, `g1_foot_clearance`, and conservative deterministic `g1_capsule_clearance`.
- Produces: `G1LegClearance` and `G1PoseClearance` with signed Hips, knee, ankle, toe, four-point sole, thigh-capsule, and shin-capsule clearances against the exact G1HF/v2 surface.
- Produces: `G1SwingHistory`, `G1SwingClearancePlan`, `g1_swing_history_reset`, and `g1_swing_clearance_plan`.
- Swing observation sweeps each IK-off sole probe from the prior 25 Hz support-retargeted pose to the current one at no more than half a terrain cell per sample. It requests only the missing vertical clearance, caps swing-only lift at `0.08 m`, and requests safe stop if that cap is insufficient.

- [ ] **Step 1: Write failing point, sole, capsule, sweep, and wall tests**

Create `tests/cpp/test_g1_clearance.cpp`:

```cpp
#include "g1_clearance.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 clearance test failed: %s\n", message);
        std::exit(1);
    }
}

static heightfield make_step(float height)
{
    heightfield field;
    field.version = 2;
    field.nx = 4;
    field.nz = 2;
    field.origin_x = 0.0f;
    field.origin_z = 0.0f;
    field.cell_size = 0.10f;
    field.exterior_height = -10.0f;
    field.heights.resize(8);
    for (int z = 0; z < 2; ++z) {
        field.heights(z * 4 + 0) = 0.0f;
        field.heights(z * 4 + 1) = 0.0f;
        field.heights(z * 4 + 2) = height;
        field.heights(z * 4 + 3) = height;
    }
    return field;
}

static void set_foot_centers(vec3 points[4], float x, float center_y)
{
    points[0] = vec3(x - 0.05f, center_y, 0.035f);
    points[1] = vec3(x - 0.05f, center_y, 0.065f);
    points[2] = vec3(x + 0.12f, center_y, 0.030f);
    points[3] = vec3(x + 0.12f, center_y, 0.070f);
}

static void test_point_sole_and_capsule_clearance()
{
    const heightfield flat = make_step(0.0f);
    char error[256] = {};
    float clearance = 0.0f;
    check(g1_point_clearance(
              clearance, flat, vec3(0.05f, 0.03f, 0.05f),
              error, sizeof(error)), error);
    check(std::fabs(clearance - 0.03f) < 1e-6f, "point clearance");

    vec3 sphere_centers[4];
    for (int i = 0; i < 4; ++i)
        sphere_centers[i] = vec3(0.05f, 0.025f, 0.05f);
    G1MinimumClearance foot = {};
    check(g1_foot_clearance(
              foot, flat, sphere_centers, 0.02f,
              error, sizeof(error)), error);
    check(std::fabs(foot.minimum_m - 0.005f) < 1e-6f,
          "four-sphere foot envelope clearance");

    const heightfield edge = make_step(0.20f);
    for (int i = 0; i < 4; ++i)
        sphere_centers[i] = vec3(0.095f, 0.025f, 0.05f);
    check(g1_point_clearance(
              clearance, edge, vec3(0.095f, 0.005f, 0.05f),
              error, sizeof(error)), error);
    check(clearance > 0.0049f, "sphere bottom-center probe is clear");
    check(g1_foot_clearance(
              foot, edge, sphere_centers, 0.02f,
              error, sizeof(error)), error);
    check(foot.minimum_m < 0.0f,
          "sphere lower envelope catches adjacent stair edge");

    G1MinimumClearance capsule = {};
    check(g1_capsule_clearance(
              capsule, flat,
              vec3(0.05f, 0.20f, 0.05f),
              vec3(0.05f, 0.40f, 0.05f), 0.04f,
              error, sizeof(error)), error);
    check(capsule.minimum_m > 0.159f && capsule.minimum_m < 0.161f,
          "capsule lower envelope clearance");
    check(capsule.samples > 8, "capsule uses bounded spatial samples");
}

static void test_swept_clearance_and_safe_stop()
{
    const G1LegConfig leg = g1_left_leg_config();
    char error[256] = {};
    vec3 previous[4];
    vec3 current[4];
    set_foot_centers(previous, 0.02f, 0.04f);
    set_foot_centers(current, 0.25f, 0.04f);

    G1SwingHistory history = {};
    g1_swing_history_reset(history, previous);
    G1SwingClearancePlan plan = {};
    const heightfield shallow = make_step(0.04f);
    check(g1_swing_clearance_plan(
              history, plan, shallow, leg, current, false,
              1.0f / 25.0f, error, sizeof(error)), error);
    check(plan.samples >= 5, "sweep samples at half-cell spacing");
    check(plan.required_lift_m > 0.0f &&
          plan.required_lift_m <= leg.max_swing_lift_m,
          "shallow step requests bounded lift");
    check(!plan.safe_stop_requested, "bounded shallow lift remains traversable");
    check(plan.corrected_margin_m >= -1e-5f,
          "corrected sphere sweep is actually clear");
    vec3 lifted[4];
    for (int i = 0; i < 4; ++i) {
        lifted[i] = current[i];
        lifted[i].y += plan.applied_lift_m;
    }
    check(g1_swing_clearance_validate(
              plan.actual_corrected_margin_m, history, shallow, leg,
              lifted, false, error, sizeof(error)), error);
    check(plan.actual_corrected_margin_m >= -1e-5f,
          "actual corrected endpoint sweep is clear");

    g1_swing_history_reset(history, previous);
    const heightfield wall = make_step(0.45f);
    check(g1_swing_clearance_plan(
              history, plan, wall, leg, current, false,
              1.0f / 25.0f, error, sizeof(error)), error);
    check(plan.required_lift_m > leg.max_swing_lift_m,
          "wall exceeds swing-only lift");
    check(std::fabs(plan.applied_lift_m - leg.max_swing_lift_m) < 1e-7f,
          "wall lift remains capped");
    check(plan.safe_stop_requested, "wall requests traversability safe stop");

    g1_swing_history_reset(history, current);
    check(g1_swing_clearance_plan(
              history, plan, shallow, leg, current, true,
              1.0f / 25.0f, error, sizeof(error)), error);
    check(plan.applied_lift_m == 0.0f && plan.samples == 0,
          "planted foot does not receive swing lift");
    g1_swing_history_commit(history, current);
    check(history.previous_sphere_centers[0].x == current[0].x,
          "only accepted rendered output commits sweep history");
}

int main()
{
    test_point_sole_and_capsule_clearance();
    test_swept_clearance_and_safe_stop();
    return 0;
}
```

- [ ] **Step 2: Compile to verify clearance RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance
```

Expected: compilation fails with `fatal error: g1_clearance.h: No such file or directory`.

- [ ] **Step 3: Implement exact-surface point, foot, and capsule diagnostics**

Create `g1_clearance.h`:

```cpp
#pragma once

#include "g1_ik.h"

#include <cfloat>
#include <cmath>

struct G1MinimumClearance
{
    float minimum_m = FLT_MAX;
    vec3 body_point;
    vec3 surface_point;
    int samples = 0;
};

static inline bool g1_clearance_consider(
    G1MinimumClearance& result,
    const heightfield& field,
    vec3 body_point,
    char* error,
    int error_capacity)
{
    if (!g1_vec3_is_finite(body_point))
        return g1_ik_error(error, error_capacity, "G1 clearance body point is non-finite");
    const float surface_y = heightfield_sample(field, body_point.x, body_point.z);
    const float clearance = body_point.y - surface_y;
    if (!terrain_float_is_finite(surface_y) || !terrain_float_is_finite(clearance))
        return g1_ik_error(error, error_capacity, "G1 clearance surface sample is non-finite");
    ++result.samples;
    if (clearance < result.minimum_m) {
        result.minimum_m = clearance;
        result.body_point = body_point;
        result.surface_point = vec3(body_point.x, surface_y, body_point.z);
    }
    return true;
}

static inline bool g1_point_clearance(
    float& output,
    const heightfield& field,
    vec3 point,
    char* error,
    int error_capacity)
{
    G1MinimumClearance result = {};
    if (!g1_clearance_consider(result, field, point, error, error_capacity))
        return false;
    output = result.minimum_m;
    return true;
}

static inline void g1_sole_points_world(
    vec3 output[4],
    vec3 contact_position,
    quat contact_rotation,
    const G1LegConfig& config)
{
    for (int i = 0; i < 4; ++i)
        output[i] = contact_position + quat_mul_vec3(
            contact_rotation, config.sole_points_local[i]);
}

static inline void g1_foot_sphere_centers_world(
    vec3 output[4],
    vec3 contact_position,
    quat contact_rotation,
    const G1LegConfig& config)
{
    for (int i = 0; i < 4; ++i)
        output[i] = contact_position + quat_mul_vec3(
            contact_rotation, config.foot_sphere_centers_local[i]);
}

static inline vec3 g1_sole_center_world(const vec3 points[4])
{
    return 0.25f * (points[0] + points[1] + points[2] + points[3]);
}

static inline bool g1_capsule_clearance(
    G1MinimumClearance& output,
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius_m,
    char* error,
    int error_capacity)
{
    if (field.version != 2 || !g1_vec3_is_finite(endpoint_a) ||
        !g1_vec3_is_finite(endpoint_b) ||
        !terrain_float_is_finite(radius_m) || radius_m <= 0.0f ||
        !terrain_float_is_finite(field.cell_size) || field.cell_size <= 0.0f) {
        return g1_ik_error(error, error_capacity, "G1 capsule clearance input is invalid");
    }
    const float spacing = minf(0.01f, 0.5f * field.cell_size);
    const int segment_steps = static_cast<int>(maxf(
        1.0f, std::ceil(length(endpoint_b - endpoint_a) / spacing)));
    const int radial_steps = static_cast<int>(std::ceil(radius_m / spacing));
    G1MinimumClearance result = {};
    for (int along = 0; along <= segment_steps; ++along) {
        const float alpha = static_cast<float>(along) / segment_steps;
        const vec3 center = lerp(endpoint_a, endpoint_b, alpha);
        for (int ix = -radial_steps; ix <= radial_steps; ++ix) {
            for (int iz = -radial_steps; iz <= radial_steps; ++iz) {
                const float dx = ix * spacing;
                const float dz = iz * spacing;
                const float horizontal_sq = dx * dx + dz * dz;
                if (horizontal_sq > radius_m * radius_m + 1e-8f) continue;
                const float lower = center.y - std::sqrt(maxf(
                    radius_m * radius_m - horizontal_sq, 0.0f));
                if (!g1_clearance_consider(
                        result, field, vec3(center.x + dx, lower, center.z + dz),
                        error, error_capacity)) return false;
            }
        }
    }
    output = result;
    return true;
}

static inline bool g1_foot_clearance(
    G1MinimumClearance& output,
    const heightfield& field,
    const vec3 sphere_centers[4],
    float sphere_radius_m,
    char* error,
    int error_capacity)
{
    G1MinimumClearance result = {};
    for (int sphere = 0; sphere < 4; ++sphere) {
        G1MinimumClearance current = {};
        if (!g1_capsule_clearance(
                current, field, sphere_centers[sphere],
                sphere_centers[sphere], sphere_radius_m,
                error, error_capacity)) return false;
        result.samples += current.samples;
        if (current.minimum_m < result.minimum_m) {
            result.minimum_m = current.minimum_m;
            result.body_point = current.body_point;
            result.surface_point = current.surface_point;
        }
    }
    output = result;
    return true;
}
```

- [ ] **Step 4: Implement swept swing history and bounded lift planning**

Append to `g1_clearance.h`:

```cpp
struct G1SwingHistory
{
    bool initialized = false;
    vec3 previous_sphere_centers[4];
};

struct G1SwingClearancePlan
{
    float baseline_minimum_m = FLT_MAX;
    float corrected_minimum_m = FLT_MAX;
    float corrected_margin_m = FLT_MAX;
    float actual_corrected_margin_m = FLT_MAX;
    float required_lift_m = 0.0f;
    float applied_lift_m = 0.0f;
    bool safe_stop_requested = false;
    vec3 worst_body_point;
    vec3 worst_surface_point;
    int samples = 0;
};

static inline void g1_swing_history_reset(
    G1SwingHistory& history, const vec3 sphere_centers[4])
{
    history = G1SwingHistory();
    history.initialized = true;
    for (int i = 0; i < 4; ++i)
        history.previous_sphere_centers[i] = sphere_centers[i];
}

static inline void g1_swing_history_commit(
    G1SwingHistory& history, const vec3 accepted_sphere_centers[4])
{
    for (int i = 0; i < 4; ++i)
        history.previous_sphere_centers[i] = accepted_sphere_centers[i];
}

static inline bool g1_swing_clearance_plan(
    const G1SwingHistory& history,
    G1SwingClearancePlan& output,
    const heightfield& field,
    const G1LegConfig& config,
    const vec3 current_sphere_centers[4],
    bool recorded_contact,
    float dt,
    char* error,
    int error_capacity)
{
    if (!history.initialized || field.version != 2 ||
        !terrain_float_is_finite(dt) ||
        std::fabs(dt - 1.0f / 25.0f) > 1e-7f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 swing sweep requires initialized G1HF/v2 state at 25 Hz");
    }
    for (int i = 0; i < 4; ++i)
        if (!g1_vec3_is_finite(current_sphere_centers[i]))
            return g1_ik_error(error, error_capacity, "G1 swing sphere center is non-finite");

    G1SwingClearancePlan result = {};
    if (!recorded_contact) {
        const float spacing = minf(0.01f, 0.5f * field.cell_size);
        float maximum_travel = 0.0f;
        for (int i = 0; i < 4; ++i)
            maximum_travel = maxf(
                maximum_travel,
                length(current_sphere_centers[i] -
                       history.previous_sphere_centers[i]));
        const int steps = static_cast<int>(maxf(
            1.0f, std::ceil(maximum_travel / spacing)));
        for (int step = 1; step <= steps; ++step) {
            const float alpha = static_cast<float>(step) / steps;
            vec3 centers[4];
            for (int sphere = 0; sphere < 4; ++sphere) {
                centers[sphere] = lerp(
                    history.previous_sphere_centers[sphere],
                    current_sphere_centers[sphere], alpha);
            }
            G1MinimumClearance sample = {};
            if (!g1_foot_clearance(
                    sample, field, centers, config.foot_sphere_radius_m,
                    error, error_capacity)) return false;
            const float target_clearance = lerpf(
                config.planted_clearance_m,
                config.swing_clearance_m, alpha);
            result.required_lift_m = maxf(
                result.required_lift_m,
                maxf(target_clearance - sample.minimum_m, 0.0f) / alpha);
            if (sample.minimum_m < result.baseline_minimum_m) {
                result.baseline_minimum_m = sample.minimum_m;
                result.worst_body_point = sample.body_point;
                result.worst_surface_point = sample.surface_point;
            }
            result.samples += sample.samples;
        }
        result.applied_lift_m = minf(
            result.required_lift_m, config.max_swing_lift_m);
        for (int step = 1; step <= steps; ++step) {
            const float alpha = static_cast<float>(step) / steps;
            vec3 centers[4];
            for (int sphere = 0; sphere < 4; ++sphere) {
                centers[sphere] = lerp(
                    history.previous_sphere_centers[sphere],
                    current_sphere_centers[sphere], alpha);
                centers[sphere].y += alpha * result.applied_lift_m;
            }
            G1MinimumClearance corrected = {};
            if (!g1_foot_clearance(
                    corrected, field, centers, config.foot_sphere_radius_m,
                    error, error_capacity)) return false;
            const float target_clearance = lerpf(
                config.planted_clearance_m,
                config.swing_clearance_m, alpha);
            result.corrected_minimum_m = minf(
                result.corrected_minimum_m, corrected.minimum_m);
            result.corrected_margin_m = minf(
                result.corrected_margin_m,
                corrected.minimum_m - target_clearance);
            result.samples += corrected.samples;
        }
        result.safe_stop_requested =
            result.required_lift_m > config.max_swing_lift_m + 1e-6f ||
            result.corrected_margin_m < -1e-5f;
    }
    output = result;
    return true;
}

static inline bool g1_swing_clearance_validate(
    float& output_margin,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const vec3 final_sphere_centers[4],
    bool recorded_contact,
    char* error,
    int error_capacity)
{
    if (recorded_contact) {
        output_margin = FLT_MAX;
        return true;
    }
    const float spacing = minf(0.01f, 0.5f * field.cell_size);
    float maximum_travel = 0.0f;
    for (int sphere = 0; sphere < 4; ++sphere)
        maximum_travel = maxf(
            maximum_travel,
            length(final_sphere_centers[sphere] -
                   history.previous_sphere_centers[sphere]));
    const int steps = static_cast<int>(maxf(
        1.0f, std::ceil(maximum_travel / spacing)));
    float margin = FLT_MAX;
    for (int step = 1; step <= steps; ++step) {
        const float alpha = static_cast<float>(step) / steps;
        vec3 centers[4];
        for (int sphere = 0; sphere < 4; ++sphere)
            centers[sphere] = lerp(
                history.previous_sphere_centers[sphere],
                final_sphere_centers[sphere], alpha);
        G1MinimumClearance sample = {};
        if (!g1_foot_clearance(
                sample, field, centers, config.foot_sphere_radius_m,
                error, error_capacity)) return false;
        const float target_clearance = lerpf(
            config.planted_clearance_m,
            config.swing_clearance_m, alpha);
        margin = minf(margin, sample.minimum_m - target_clearance);
    }
    output_margin = margin;
    return true;
}
```

Planning derives endpoint lift as `deficit / alpha` for every full-sphere sweep
sample, because only `alpha * endpoint_lift` exists at an intermediate point.
It then re-sweeps the corrected linear candidate and exposes the actual margin.
The planning function never advances history. Task 7 commits only the final
accepted rendered sphere centers; a rejected/rolled-back candidate leaves the
previous safe history exact.

- [ ] **Step 5: Add complete post-IK leg/pose diagnostic aggregation**

Append to `g1_clearance.h`:

```cpp
struct G1LegClearance
{
    float knee_m = FLT_MAX;
    float ankle_m = FLT_MAX;
    float toe_m = FLT_MAX;
    float foot_m = FLT_MAX;
    float thigh_m = FLT_MAX;
    float shin_m = FLT_MAX;
    float minimum_m = FLT_MAX;
};

struct G1PoseClearance
{
    float hips_m = FLT_MAX;
    G1LegClearance left;
    G1LegClearance right;
    float minimum_m = FLT_MAX;
};

static inline vec3 g1_local_point_world(
    vec3 bone_position, quat bone_rotation, vec3 local_point)
{
    return bone_position + quat_mul_vec3(bone_rotation, local_point);
}

static inline bool g1_measure_leg_clearance(
    G1LegClearance& output,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    G1LegClearance result = {};
    if (!g1_point_clearance(result.knee_m, field, global_positions(config.knee), error, error_capacity) ||
        !g1_point_clearance(result.ankle_m, field, global_positions(config.ankle), error, error_capacity) ||
        !g1_point_clearance(result.toe_m, field, global_positions(config.contact), error, error_capacity))
        return false;
    vec3 sphere_centers[4];
    g1_foot_sphere_centers_world(
        sphere_centers, global_positions(config.contact),
        global_rotations(config.contact), config);
    G1MinimumClearance foot = {};
    G1MinimumClearance thigh = {};
    G1MinimumClearance shin = {};
    if (!g1_foot_clearance(
            foot, field, sphere_centers, config.foot_sphere_radius_m,
            error, error_capacity) ||
        !g1_capsule_clearance(
            thigh, field,
            g1_local_point_world(global_positions(config.hip), global_rotations(config.hip), config.thigh_start_local),
            g1_local_point_world(global_positions(config.hip), global_rotations(config.hip), config.thigh_end_local),
            config.thigh_radius_m, error, error_capacity) ||
        !g1_capsule_clearance(
            shin, field,
            g1_local_point_world(global_positions(config.knee), global_rotations(config.knee), config.shin_start_local),
            g1_local_point_world(global_positions(config.knee), global_rotations(config.knee), config.shin_end_local),
            config.shin_radius_m, error, error_capacity)) return false;
    result.foot_m = foot.minimum_m;
    result.thigh_m = thigh.minimum_m;
    result.shin_m = shin.minimum_m;
    result.minimum_m = minf(
        minf(minf(result.knee_m, result.ankle_m), minf(result.toe_m, result.foot_m)),
        minf(result.thigh_m, result.shin_m));
    output = result;
    return true;
}

static inline bool g1_measure_pose_clearance(
    G1PoseClearance& output,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity)
{
    if (global_positions.size != G1_BoneCount ||
        global_rotations.size != G1_BoneCount)
        return g1_ik_error(error, error_capacity, "G1 clearance pose shape mismatch");
    G1PoseClearance result = {};
    if (!g1_point_clearance(
            result.hips_m, field, global_positions(G1_Hips),
            error, error_capacity) ||
        !g1_measure_leg_clearance(
            result.left, field, global_positions, global_rotations,
            g1_left_leg_config(), error, error_capacity) ||
        !g1_measure_leg_clearance(
            result.right, field, global_positions, global_rotations,
            g1_right_leg_config(), error, error_capacity)) return false;
    result.minimum_m = minf(
        result.hips_m, minf(result.left.minimum_m, result.right.minimum_m));
    output = result;
    return true;
}
```

- [ ] **Step 6: Run clearance GREEN in strict, release, and sanitizer modes**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance_strict
/tmp/test_g1_clearance_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance_release
/tmp/test_g1_clearance_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_g1_clearance.cpp \
  -o /tmp/test_g1_clearance_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_g1_clearance_san
```

Expected: all three executables exit `0`; shallow sweep produces a bounded lift, the `0.45 m` wall produces a safe-stop request, and no sanitizer finding appears.

- [ ] **Step 7: Commit swept clearance and physical diagnostics**

```bash
git add g1_clearance.h tests/cpp/test_g1_clearance.cpp
git commit -m "feat: measure and clear G1 legs over terrain"
```

### Task 6: Compose a Reversible Per-Frame IK Observe/Apply Transaction

**Files:**
- Create: `g1_ik_runtime.h`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Produces: `G1IkState`, `G1FootFrameResult`, `G1IkFrameResult`, `g1_ik_state_reset`, and `g1_ik_frame_evaluate`.
- `g1_ik_frame_evaluate` always advances downstream lock/sweep observation from the support-retargeted IK-off pose. With `apply_enabled=false`, its output quaternion array is a byte copy of the input; with `true`, only the configured hip, knee, and contact bones may differ.
- Produces only `safe_stop_requested` and a reason enum for the sibling traversability layer. It cannot mutate traversal, matching, simulation, support, local positions, or scene state itself.

- [ ] **Step 1: Add failing IK-off reversibility and IK-on isolation tests**

Change the first include in `tests/cpp/test_g1_ik.cpp` to `#include "g1_ik_runtime.h"`. Add these helpers/tests before `main`, then call `test_frame_transaction_is_reversible_and_downstream()` from `main`:

```cpp
static heightfield make_ik_frame_surface()
{
    heightfield field;
    field.version = 2;
    field.nx = 2;
    field.nz = 2;
    field.origin_x = -1.0f;
    field.origin_z = -1.0f;
    field.cell_size = 2.0f;
    field.exterior_height = -10.0f;
    field.heights.resize(4);
    field.heights(0) = 0.12f;
    field.heights(1) = 0.22f;
    field.heights(2) = 0.12f;
    field.heights(3) = 0.22f;
    return field;
}

static database make_frame_database()
{
    database db = make_g1_database();
    db.bone_positions(0, G1_Simulation) = vec3(0.0f, 1.0f, 0.0f);
    db.bone_positions(0, G1_LeftKnee) = vec3(0.0f, -0.40f, 0.0f);
    db.bone_positions(0, G1_LeftAnkle) = vec3(0.0f, -0.40f, 0.0f);
    db.bone_positions(0, G1_RightKnee) = vec3(0.0f, -0.40f, 0.0f);
    db.bone_positions(0, G1_RightAnkle) = vec3(0.0f, -0.40f, 0.0f);
    db.contact_states.resize(1, 2);
    db.contact_states(0, 0) = true;
    db.contact_states(0, 1) = true;
    return db;
}

static void test_frame_transaction_is_reversible_and_downstream()
{
    database db = make_frame_database();
    const heightfield field = make_ik_frame_surface();
    array1d<vec3> baseline_global_positions(G1_BoneCount);
    array1d<quat> baseline_global_rotations(G1_BoneCount);
    forward_kinematics_full(
        baseline_global_positions, baseline_global_rotations,
        db.bone_positions(0), db.bone_rotations(0), db.bone_parents);
    G1IkState state = {};
    char error[256] = {};
    check(g1_ik_state_reset(
              state, baseline_global_positions, baseline_global_rotations,
              error, sizeof(error)), error);

    array1d<quat> output(G1_BoneCount);
    G1IkFrameResult result = {};
    const float support_root_y = db.bone_positions(0, G1_Simulation).y;
    check(g1_ik_frame_evaluate(
              output, state, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, db.contact_states(0), field, false,
              1.0f / 25.0f, result, error, sizeof(error)), error);
    check(!result.applied, "IK-off frame is observation only");
    for (int bone = 0; bone < G1_BoneCount; ++bone)
        check(same_quat(output(bone), db.bone_rotations(0, bone)),
              "IK-off local rotation is byte-identical");
    check(db.bone_positions(0, G1_Simulation).y == support_root_y,
          "support-retargeted root input is unchanged");

    check(g1_ik_state_reset(
              state, baseline_global_positions, baseline_global_rotations,
              error, sizeof(error)), error);
    check(g1_ik_frame_evaluate(
              output, state, db.bone_positions(0), db.bone_rotations(0),
              db.bone_parents, db.contact_states(0), field, true,
              1.0f / 25.0f, result, error, sizeof(error)), error);
    check(result.applied, "IK-on frame applies downstream pose");
    check(result.max_correction_radians <= 0.350001f,
          "frame correction bound");
    const int allowed[] = {
        G1_LeftHipYaw, G1_LeftKnee, G1_LeftToe,
        G1_RightHipYaw, G1_RightKnee, G1_RightToe
    };
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        bool may_change = false;
        for (int value : allowed) may_change = may_change || bone == value;
        if (!may_change)
            check(same_quat(output(bone), db.bone_rotations(0, bone)),
                  "IK-on changes named rotations only");
    }
    check(same_quat(output(G1_Simulation), db.bone_rotations(0, G1_Simulation)),
          "simulation rotation is immutable");
    check(db.bone_positions(0, G1_Simulation).y == support_root_y,
          "IK-on cannot change support root position");
}
```

- [ ] **Step 2: Compile to verify frame-transaction RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
```

Expected: compilation fails with `fatal error: g1_ik_runtime.h: No such file or directory`.

- [ ] **Step 3: Define fixed-size per-foot state and outcomes**

Create `g1_ik_runtime.h`:

```cpp
#pragma once

#include "g1_clearance.h"

enum G1IkStopReason
{
    G1IkStopNone = 0,
    G1IkStopSwingLift = 1,
    G1IkStopReachShell = 2,
    G1IkStopCorrectionBound = 3,
    G1IkStopLockDrift = 4,
    G1IkStopEndEffectorResidual = 5,
    G1IkStopPostSolveClearance = 6
};

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
    G1SwingClearancePlan swing;
    G1LegSolveResult position;
    G1FootOrientationResult orientation;
    float applied_swing_lift_m = 0.0f;
};

struct G1IkFrameResult
{
    bool applied = false;
    bool safe_stop_requested = false;
    G1IkStopReason stop_reason = G1IkStopNone;
    float max_correction_radians = 0.0f;
    G1FootFrameResult feet[2];
};

static inline const char* g1_ik_stop_reason_name(G1IkStopReason reason)
{
    switch (reason) {
    case G1IkStopNone: return "none";
    case G1IkStopSwingLift: return "swing-lift";
    case G1IkStopReachShell: return "reach-shell";
    case G1IkStopCorrectionBound: return "correction-bound";
    case G1IkStopLockDrift: return "lock-drift";
    case G1IkStopEndEffectorResidual: return "end-effector-residual";
    case G1IkStopPostSolveClearance: return "post-solve-clearance";
    }
    return "invalid";
}

static inline vec3 g1_config_sole_center_local(const G1LegConfig& config)
{
    return 0.25f * (
        config.sole_points_local[0] + config.sole_points_local[1] +
        config.sole_points_local[2] + config.sole_points_local[3]);
}

static inline bool g1_ik_state_reset(
    G1IkState& output,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity)
{
    if (global_positions.size != G1_BoneCount ||
        global_rotations.size != G1_BoneCount)
        return g1_ik_error(error, error_capacity, "G1 IK reset pose shape mismatch");
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    G1IkState state = {};
    state.initialized = true;
    for (int foot = 0; foot < 2; ++foot) {
        vec3 sole[4];
        g1_sole_points_world(
            sole,
            global_positions(configs[foot].contact),
            global_rotations(configs[foot].contact),
            configs[foot]);
        const vec3 center = g1_sole_center_world(sole);
        vec3 sphere_centers[4];
        g1_foot_sphere_centers_world(
            sphere_centers,
            global_positions(configs[foot].contact),
            global_rotations(configs[foot].contact), configs[foot]);
        if (!g1_vec3_is_finite(center))
            return g1_ik_error(error, error_capacity, "G1 IK reset sole is non-finite");
        g1_foot_lock_reset(state.feet[foot].lock, center);
        g1_swing_history_reset(state.feet[foot].swing, sphere_centers);
    }
    output = state;
    return true;
}
```

- [ ] **Step 4: Implement one downstream observe/apply transaction**

Append to `g1_ik_runtime.h`:

```cpp
static inline void g1_ik_request_stop(
    G1IkFrameResult& frame, G1IkStopReason reason)
{
    frame.safe_stop_requested = true;
    if (frame.stop_reason == G1IkStopNone) frame.stop_reason = reason;
}

static inline bool g1_ik_frame_evaluate(
    slice1d<quat> output_rotations,
    G1IkState& state,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const slice1d<bool> recorded_contacts,
    const heightfield& field,
    bool apply_enabled,
    float dt,
    G1IkFrameResult& output,
    char* error,
    int error_capacity)
{
    if (!state.initialized || output_rotations.size != G1_BoneCount ||
        baseline_positions.size != G1_BoneCount ||
        baseline_rotations.size != G1_BoneCount ||
        parents.size != G1_BoneCount || recorded_contacts.size != 2 ||
        field.version != 2 || !terrain_float_is_finite(dt) ||
        std::fabs(dt - 1.0f / 25.0f) > 1e-7f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK frame requires initialized 31-bone G1HF/v2 state at 25 Hz");
    }

    array1d<vec3> baseline_global_positions(G1_BoneCount);
    array1d<quat> baseline_global_rotations(G1_BoneCount);
    forward_kinematics_full(
        baseline_global_positions, baseline_global_rotations,
        baseline_positions, baseline_rotations, parents);
    array1d<quat> candidate = baseline_rotations;
    G1IkState next = state;
    G1IkFrameResult frame = {};
    frame.applied = apply_enabled;
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };

    for (int foot = 0; foot < 2; ++foot) {
        const G1LegConfig& config = configs[foot];
        G1FootFrameResult& result = frame.feet[foot];
        result.recorded_contact = recorded_contacts(foot);
        vec3 baseline_sole[4];
        g1_sole_points_world(
            baseline_sole,
            baseline_global_positions(config.contact),
            baseline_global_rotations(config.contact), config);
        vec3 baseline_centers[4];
        g1_foot_sphere_centers_world(
            baseline_centers,
            baseline_global_positions(config.contact),
            baseline_global_rotations(config.contact), config);
        const vec3 baseline_center = g1_sole_center_world(baseline_sole);
        if (!g1_foot_lock_update(
                next.feet[foot].lock, result.target,
                field, config, baseline_center, recorded_contacts(foot),
                dt, error, error_capacity) ||
            !g1_swing_clearance_plan(
                next.feet[foot].swing, result.swing,
                field, config, baseline_centers, recorded_contacts(foot),
                dt, error, error_capacity)) return false;

        result.applied_swing_lift_m = result.swing.applied_lift_m;
        if (result.target.drift_limit_exceeded)
            g1_ik_request_stop(frame, G1IkStopLockDrift);
        if (result.swing.safe_stop_requested)
            g1_ik_request_stop(frame, G1IkStopSwingLift);
        const bool needs_position =
            result.target.locked || result.applied_swing_lift_m > 0.0f;
        if (!needs_position) continue;

        vec3 desired_sole_center = result.target.locked
            ? result.target.sole_center : baseline_center;
        desired_sole_center.y += result.applied_swing_lift_m;
        vec3 target_normal = result.target.surface.normal;
        if (!result.target.locked) {
            G1SurfaceTarget swing_surface = {};
            if (!g1_surface_target_sample(
                    swing_surface, field,
                    desired_sole_center.x, desired_sole_center.z,
                    0.0f, error, error_capacity)) return false;
            target_normal = swing_surface.normal;
        }

        quat target_foot_global = baseline_global_rotations(config.contact);
        if (result.target.locked && !g1_surface_aligned_foot_rotation(
                target_foot_global,
                baseline_global_rotations(config.contact),
                config, target_normal, error, error_capacity)) return false;
        const vec3 desired_contact = desired_sole_center - quat_mul_vec3(
            target_foot_global, g1_config_sole_center_local(config));

        if (!g1_apply_named_contact_position_ik(
                candidate, baseline_positions, baseline_rotations, parents,
                config, desired_contact, result.position,
                error, error_capacity)) return false;
        frame.max_correction_radians = maxf(
            frame.max_correction_radians,
            result.position.max_correction_radians);
        if (!result.position.reachable)
            g1_ik_request_stop(frame, G1IkStopReachShell);
        if (result.position.correction_limited)
            g1_ik_request_stop(frame, G1IkStopCorrectionBound);
        if (result.position.contact_residual_m > 0.005f)
            g1_ik_request_stop(frame, G1IkStopEndEffectorResidual);

        if (result.target.locked) {
            if (!g1_apply_named_foot_orientation(
                    candidate, baseline_positions, baseline_rotations, parents,
                    config, target_normal, result.orientation,
                    error, error_capacity)) return false;
            frame.max_correction_radians = maxf(
                frame.max_correction_radians,
                result.orientation.correction_radians);
            if (result.orientation.correction_limited)
                g1_ik_request_stop(frame, G1IkStopCorrectionBound);
        }
    }

    array1d<vec3> candidate_global_positions(G1_BoneCount);
    array1d<quat> candidate_global_rotations(G1_BoneCount);
    forward_kinematics_full(
        candidate_global_positions, candidate_global_rotations,
        baseline_positions, candidate, parents);
    for (int foot = 0; foot < 2; ++foot) {
        vec3 final_centers[4];
        g1_foot_sphere_centers_world(
            final_centers,
            candidate_global_positions(configs[foot].contact),
            candidate_global_rotations(configs[foot].contact),
            configs[foot]);
        if (!g1_swing_clearance_validate(
                frame.feet[foot].swing.actual_corrected_margin_m,
                next.feet[foot].swing, field, configs[foot], final_centers,
                frame.feet[foot].recorded_contact,
                error, error_capacity)) return false;
        if (frame.feet[foot].swing.actual_corrected_margin_m < -1e-5f)
            g1_ik_request_stop(frame, G1IkStopSwingLift);
    }

    if (!terrain_float_is_finite(frame.max_correction_radians) ||
        frame.max_correction_radians > 0.35f + 1e-6f) {
        return g1_ik_error(
            error, error_capacity,
            "G1 IK frame produced non-finite or over-bound correction");
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone)
        output_rotations(bone) = apply_enabled
            ? candidate(bone) : baseline_rotations(bone);
    state = next;
    output = frame;
    return true;
}
```

The current frame's matching has already completed before this function runs. A safe-stop request is an output only; Task 7 forwards it to the sibling traversability API for subsequent applied motion.

- [ ] **Step 5: Run reversible frame GREEN under all native modes**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_strict
/tmp/test_g1_ik_strict
g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik_release
/tmp/test_g1_ik_release
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -I. tests/cpp/test_g1_ik.cpp \
  -o /tmp/test_g1_ik_san
ASAN_OPTIONS=detect_leaks=1 /tmp/test_g1_ik_san
```

Expected: all modes exit `0`; IK-off is byte-identical, IK-on changes only the six named leg rotations, support-root inputs remain unchanged, and no correction exceeds `0.35` radians.

- [ ] **Step 6: Commit the reversible frame transaction**

```bash
git add g1_ik_runtime.h tests/cpp/test_g1_ik.cpp
git commit -m "feat: compose reversible G1 terrain IK frames"
```

### Task 7: Integrate IK State, Safe-Stop Handoff, and Controlled Cleanup

**Files:**
- Modify: `g1_controller_state.h`
- Modify: `controller.cpp`
- Modify: `tests/cpp/test_g1_controller_state.cpp`
- Modify: `tests/cpp/test_g1_ik.cpp`

**Interfaces:**
- Extends sibling-owned `g1_controller_state` with resettable `G1IkState`,
  separate last-safe output and scratch-candidate local/global pose arrays, the
  latest `G1IkFrameResult`/accepted and rejected `G1PoseClearance`, and a
  latched safe-stop request.
- Consumes sibling-owned support-retargeted `state.adjusted_bone_*`, active `scene_pack.terrain`, recorded `state.curr_bone_contacts`, and the existing traversability command/clip functions.
- `MM_IK` accepts exactly `0` or `1`, defaults to `0`, and is persistent configuration outside scene-reset state. A live checkbox edge resets only downstream IK observation from the current support-retargeted baseline.
- Any finite unsafe outcome rejects the candidate pose transactionally, retains
  the prior measured-safe rendered output, and on the next update zeros the
  desired XZ command plus planar simulation velocity/acceleration before
  prediction. A non-finite outcome sets the existing
  `controller_exit_requested/controller_exit_code` and returns through normal
  log/model/window cleanup.

- [ ] **Step 1: Extend reset tests first**

In `tests/cpp/test_g1_controller_state.cpp`, poison IK fields before the existing reset call:

```cpp
state.ik.initialized = true;
state.ik.feet[0].lock.locked = true;
state.ik_safe_stop_latched = true;
state.ik_bone_positions.resize(G1_BoneCount);
state.ik_bone_positions.set(vec3(9.0f, 9.0f, 9.0f));
state.ik_bone_rotations.resize(G1_BoneCount);
state.ik_bone_rotations.set(quat(0.0f, 1.0f, 0.0f, 0.0f));
state.ik_candidate_bone_positions.resize(G1_BoneCount);
state.ik_candidate_bone_positions.set(vec3(-9.0f, -9.0f, -9.0f));
state.ik_candidate_rejected = true;
```

After reset, add:

```cpp
check(state.ik.initialized, "IK observation initialized");
check(!state.ik.feet[0].lock.locked && !state.ik.feet[1].lock.locked,
      "IK locks reset");
check(!state.ik_safe_stop_latched, "IK safe stop reset");
check(!state.ik_candidate_rejected, "IK candidate rejection reset");
for (int bone = 0; bone < G1_BoneCount; ++bone) {
    check(state.ik_bone_positions(bone).x == state.adjusted_bone_positions(bone).x &&
          state.ik_bone_positions(bone).y == state.adjusted_bone_positions(bone).y &&
          state.ik_bone_positions(bone).z == state.adjusted_bone_positions(bone).z,
          "IK positions reset from support baseline");
    check(quat_angle_between(
              state.ik_bone_rotations(bone),
              state.adjusted_bone_rotations(bone)) < 1e-7f,
          "IK rotations reset from support baseline");
    check(state.ik_candidate_bone_positions(bone).x ==
              state.ik_bone_positions(bone).x &&
          quat_angle_between(
              state.ik_candidate_bone_rotations(bone),
              state.ik_bone_rotations(bone)) < 1e-7f,
          "IK scratch candidate reset from last-safe output");
}
check(state.ik_bone_positions(G1_Simulation).y == state.support.height,
      "IK reset preserves support-retargeted root Y");
```

In `tests/cpp/test_g1_ik.cpp`, add and call:

```cpp
static void test_ik_option_is_exact()
{
    bool enabled = true;
    char error[128] = {};
    check(g1_parse_ik_enabled(enabled, NULL, error, sizeof(error)) && !enabled,
          "missing MM_IK defaults off");
    check(g1_parse_ik_enabled(enabled, "1", error, sizeof(error)) && enabled,
          "MM_IK=1 enables");
    check(g1_parse_ik_enabled(enabled, "0", error, sizeof(error)) && !enabled,
          "MM_IK=0 disables");
    check(!g1_parse_ik_enabled(enabled, "true", error, sizeof(error)),
          "non-exact MM_IK rejected");
    check(std::strstr(error, "0 or 1") != NULL, "MM_IK diagnostic");

    const vec3 command(0.4f, 0.0f, -0.2f);
    const G1IkStopHandoff allowed =
        g1_ik_safe_stop_handoff(false, command);
    const G1IkStopHandoff stopped =
        g1_ik_safe_stop_handoff(true, command);
    check(allowed.desired_command.x == command.x &&
          allowed.desired_command.y == command.y &&
          allowed.desired_command.z == command.z &&
          !allowed.cancel_planar_inertia,
          "unlatched IK command is exact");
    check(stopped.desired_command.x == 0.0f &&
          stopped.desired_command.y == 0.0f &&
          stopped.desired_command.z == 0.0f &&
          stopped.cancel_planar_inertia,
          "latched IK handoff cancels command and planar coast");
}
```

- [ ] **Step 2: Compile to verify controller-state RED**

Run:

```bash
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_controller_state
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
```

Expected: compilation fails because the controller state has no IK fields and `g1_parse_ik_enabled` is undefined.

- [ ] **Step 3: Add exact option parsing**

Append to `g1_ik_runtime.h`:

```cpp
static inline bool g1_parse_ik_enabled(
    bool& enabled, const char* text, char* error, int error_capacity)
{
    if (text == NULL) {
        enabled = false;
        return true;
    }
    if (std::strcmp(text, "0") == 0) {
        enabled = false;
        return true;
    }
    if (std::strcmp(text, "1") == 0) {
        enabled = true;
        return true;
    }
    return g1_ik_error(
        error, error_capacity,
        "MM_IK must be exactly 0 or 1, got '%s'", text);
}
```

Add `#include <cstring>` to `g1_ik_runtime.h`.

Also append the explicit next-update handoff helper used by the test and controller:

```cpp
struct G1IkStopHandoff
{
    vec3 desired_command;
    bool cancel_planar_inertia = false;
};

static inline G1IkStopHandoff g1_ik_safe_stop_handoff(
    bool safe_stop_latched, vec3 desired_command)
{
    G1IkStopHandoff result = {};
    result.desired_command = safe_stop_latched ? vec3() : desired_command;
    result.cancel_planar_inertia = safe_stop_latched;
    return result;
}
```

- [ ] **Step 4: Replace dormant generic contact state with explicit IK state**

Include `g1_ik_runtime.h` from `g1_controller_state.h`. Replace the sibling plan's dormant `contact_bones`, `contact_states`, `contact_locks`, `contact_positions`, `contact_velocities`, `contact_points`, `contact_targets`, `contact_offset_positions`, and `contact_offset_velocities` members with:

```cpp
G1IkState ik;
array1d<vec3> ik_bone_positions;
array1d<quat> ik_bone_rotations;
array1d<vec3> ik_global_bone_positions;
array1d<quat> ik_global_bone_rotations;
array1d<vec3> ik_candidate_bone_positions;
array1d<quat> ik_candidate_bone_rotations;
array1d<vec3> ik_candidate_global_bone_positions;
array1d<quat> ik_candidate_global_bone_rotations;
G1IkFrameResult ik_frame;
G1PoseClearance ik_clearance;
G1PoseClearance ik_candidate_clearance;
bool ik_candidate_rejected = false;
bool ik_safe_stop_latched = false;
```

In `g1_controller_state_swap`, swap all eight arrays with the existing
`g1_swap`, then swap `ik`, `ik_frame`, both clearance structs,
`ik_candidate_rejected`, and `ik_safe_stop_latched` with `std::swap`. Remove
swaps for the deleted generic contact arrays.

In the local candidate inside `g1_controller_state_reset`, after `support_pose_apply`, initialize the separate output buffers and downstream state:

```cpp
s.ik_bone_positions = s.adjusted_bone_positions;
s.ik_bone_rotations = s.adjusted_bone_rotations;
s.ik_global_bone_positions.resize(bones);
s.ik_global_bone_rotations.resize(bones);
forward_kinematics_full(
    s.ik_global_bone_positions,
    s.ik_global_bone_rotations,
    s.ik_bone_positions,
    s.ik_bone_rotations,
    db.bone_parents);
s.ik_candidate_bone_positions = s.ik_bone_positions;
s.ik_candidate_bone_rotations = s.ik_bone_rotations;
s.ik_candidate_global_bone_positions = s.ik_global_bone_positions;
s.ik_candidate_global_bone_rotations = s.ik_global_bone_rotations;
if (!g1_ik_state_reset(
        s.ik,
        s.ik_global_bone_positions,
        s.ik_global_bone_rotations,
        error,
        capacity)) return false;
s.ik_frame = G1IkFrameResult();
s.ik_clearance = G1PoseClearance();
if (!g1_measure_pose_clearance(
        s.ik_clearance, scene.terrain,
        s.ik_global_bone_positions, s.ik_global_bone_rotations,
        error, capacity) || s.ik_clearance.minimum_m < -0.01f)
    return scene_error(
        error, capacity,
        "controller reset: scene '%s' has no physically safe initial G1 pose",
        scene.metadata.id.c_str());
s.ik_candidate_clearance = s.ik_clearance;
s.ik_candidate_rejected = false;
s.ik_safe_stop_latched = false;
```

This remains inside the reset candidate, so a failed IK reset cannot mutate the active scene/controller state.

- [ ] **Step 5: Parse IK before opening Raylib and remove the inherited solver block**

In `controller.cpp`, include `g1_ik_runtime.h`. Parse once beside the sibling runtime's other validated options:

```cpp
bool ik_enabled = false;
if (!g1_parse_ik_enabled(
        ik_enabled, std::getenv("MM_IK"),
        artifact_error, sizeof(artifact_error))) {
    std::fprintf(stderr, "G1 IK option error: %s\n", artifact_error);
    return 2;
}
```

Delete the old file-scope `contact_reset`, `contact_update`, `ik_look_at`, and `ik_two_bone` functions and delete the parent-walk/hard-coded toe-end `if (ik_enabled)` block. `ik.h`, `g1_ik.h`, and `g1_ik_runtime.h` become the only active IK implementation.

- [ ] **Step 6: Insert IK only after support retargeting and baseline FK**

At the sibling runtime's fixed-update order, keep matching, inertialization, support update, `support_pose_apply`, horizontal adjustment/clamp, and baseline `forward_kinematics_full` unchanged. Immediately after baseline support FK, add:

```cpp
state.ik_candidate_bone_positions = state.adjusted_bone_positions;
if (!g1_ik_frame_evaluate(
        state.ik_candidate_bone_rotations,
        state.ik,
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
    std::fprintf(
        stderr,
        "G1 IK controlled error scene=%s route=%s frame=%d: %s\n",
        active_scene.metadata.id.c_str(),
        active_scene.metadata.routes[static_cast<size_t>(state.route_index)].id.c_str(),
        state.scene_frame,
        artifact_error);
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}
forward_kinematics_full(
    state.ik_candidate_global_bone_positions,
    state.ik_candidate_global_bone_rotations,
    state.ik_candidate_bone_positions,
    state.ik_candidate_bone_rotations,
    db.bone_parents);
if (!g1_measure_pose_clearance(
        state.ik_candidate_clearance,
        active_scene.terrain,
        state.ik_candidate_global_bone_positions,
        state.ik_candidate_global_bone_rotations,
        artifact_error,
        sizeof(artifact_error))) {
    std::fprintf(
        stderr,
        "G1 IK clearance controlled error scene=%s frame=%d: %s\n",
        active_scene.metadata.id.c_str(), state.scene_frame, artifact_error);
    controller_exit_code = 2;
    controller_exit_requested = true;
    return;
}

const bool planted_penetration =
    (state.ik_frame.feet[0].target.locked &&
     (state.ik_candidate_clearance.left.toe_m < -0.005f ||
      state.ik_candidate_clearance.left.foot_m < -0.005f)) ||
    (state.ik_frame.feet[1].target.locked &&
     (state.ik_candidate_clearance.right.toe_m < -0.005f ||
      state.ik_candidate_clearance.right.foot_m < -0.005f));
const bool physical_penetration =
    state.ik_candidate_clearance.minimum_m < -0.01f;
if (ik_enabled && (planted_penetration || physical_penetration)) {
    g1_ik_request_stop(state.ik_frame, G1IkStopPostSolveClearance);
}
state.ik_candidate_rejected =
    ik_enabled && state.ik_frame.safe_stop_requested;
if (!state.ik_candidate_rejected) {
    state.ik_bone_positions = state.ik_candidate_bone_positions;
    state.ik_bone_rotations = state.ik_candidate_bone_rotations;
    state.ik_global_bone_positions = state.ik_candidate_global_bone_positions;
    state.ik_global_bone_rotations = state.ik_candidate_global_bone_rotations;
    state.ik_clearance = state.ik_candidate_clearance;
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        vec3 accepted_centers[4];
        g1_foot_sphere_centers_world(
            accepted_centers,
            state.ik_global_bone_positions(configs[foot].contact),
            state.ik_global_bone_rotations(configs[foot].contact),
            configs[foot]);
        g1_swing_history_commit(
            state.ik.feet[foot].swing, accepted_centers);
    }
}
if (state.ik_candidate_rejected) {
    state.ik_safe_stop_latched = true;
}
```

Do not write `state.adjusted_bone_*`, `state.bone_*`, `state.simulation_*`, or
`state.support` anywhere in this block. Render and accepted post-IK diagnostics
use only `state.ik_global_bone_*`. Every finite unsafe candidate—including
reach, correction, lock-drift, residual, corrected-sweep, planted toe/foot, or
physical capsule failure—leaves those last-safe arrays and their clearance
unchanged. Candidate diagnostics remain separately loggable as rejection
evidence. IK-off always commits the byte-identical support-retargeted baseline.

- [ ] **Step 7: Hand the latched request to the existing stop path**

At the next update's input-to-traversability boundary, before the sibling call to `traversability_limit_command`, add:

```cpp
const vec3 user_or_route_command = desired_velocity_curr;
const G1IkStopHandoff ik_stop_handoff = g1_ik_safe_stop_handoff(
    state.ik_safe_stop_latched, desired_velocity_curr);
if (ik_stop_handoff.cancel_planar_inertia) {
    state.simulation_velocity.x = 0.0f;
    state.simulation_velocity.z = 0.0f;
    state.simulation_acceleration.x = 0.0f;
    state.simulation_acceleration.z = 0.0f;
}
desired_velocity_curr = traversability_limit_command(
    state.traversal_speed_scale,
    state.traversal_speed_scale_velocity,
    traversal,
    active_scene.walkability,
    active_scene.terrain,
    state.simulation_position,
    ik_stop_handoff.desired_command,
    dt);
```

Retain `user_or_route_command` for the logger's commanded-speed field. The sibling `traversability_clip_step` remains the hard XZ boundary. The IK latch stays set until scene reset or the explicit live IK reset action; it does not invent a Y correction or overwrite G1WM class.
The only simulation-state write introduced by IK occurs at this existing
input-to-traversability boundary: a previously latched stop zeros X/Z velocity
and acceleration before trajectory prediction and simulation integration while
preserving both Y components exactly. Step 6 itself remains a pose-only
transaction.

- [ ] **Step 8: Add the live reversible toggle**

In the existing diagnostics UI, add:

```cpp
const bool ik_was_enabled = ik_enabled;
GuiCheckBox(
    (Rectangle){20, 575, 250, 20},
    "bounded G1 terrain IK",
    &ik_enabled);
if (ik_enabled != ik_was_enabled) {
    if (!g1_ik_state_reset(
            state.ik,
            state.global_bone_positions,
            state.global_bone_rotations,
            artifact_error,
            sizeof(artifact_error))) {
        std::fprintf(stderr, "G1 IK toggle reset error: %s\n", artifact_error);
        controller_exit_code = 2;
        controller_exit_requested = true;
    }
    state.ik_safe_stop_latched = false;
}
```

The false edge restores exact IK-off output on the next fixed update; the true edge starts with no stale lock or sweep history.

- [ ] **Step 9: Run state GREEN, all focused tests, and a deterministic default-scene smoke**

Run:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_controller_state.cpp -o /tmp/test_g1_controller_state
/tmp/test_g1_controller_state
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
/tmp/test_g1_ik
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
  tests/cpp/test_g1_clearance.cpp -o /tmp/test_g1_clearance
/tmp/test_g1_clearance
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_ik \
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

Expected: every native test exits `0`; both controller runs close normally after exactly 375 fixed updates; no `controlled error`, non-finite state, or correction above `0.35` appears. Exact log comparison is added in Task 8.

- [ ] **Step 10: Commit controller integration only**

```bash
git add g1_controller_state.h controller.cpp \
  tests/cpp/test_g1_controller_state.cpp tests/cpp/test_g1_ik.cpp
git commit -m "feat: integrate reversible G1 terrain IK"
```

### Task 8: Append Gate E Diagnostics and Prove IK Invariance

**Files:**
- Modify: `motion_match_log.h`
- Modify: `controller.cpp`
- Modify: `resources/check_g1_runtime_log.py`
- Modify: `tests/python/test_runtime_log.py`

**Interfaces:**
- Appends, without renaming or reordering any sibling column, this exact IK suffix:

```text
ik_applied,ik_safe_stop_requested,ik_stop_reason,max_ik_correction,
actual_simulation_speed,ik_candidate_rejected,
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
left_swing_required_lift,left_swing_applied_lift,
left_swing_planned_margin,left_swing_actual_margin,
right_swing_required_lift,right_swing_applied_lift,
right_swing_planned_margin,right_swing_actual_margin,
left_reachable,right_reachable,left_knee_clearance,left_ankle_clearance,
left_toe_clearance,left_foot_clearance,left_shin_clearance,
left_thigh_clearance,right_knee_clearance,right_ankle_clearance,
right_toe_clearance,right_foot_clearance,right_shin_clearance,
right_thigh_clearance,ik_hips_clearance,ik_minimum_clearance
```

- `left_observed_lock_drift` and `right_observed_lock_drift` are the
  input-only support-retargeted baseline drifts stored in `G1FootTarget`; they
  must be exact between paired modes and expose the `0.20 m` safe-stop bound.
  `left_lock_drift` and `right_lock_drift` are post-output horizontal
  distances from the current sole center to the frozen lock point. With IK off
  they therefore measure the support-retargeted baseline; with IK on they
  measure the actual corrected pose.
- `left_sole_normal_alignment` and `right_sole_normal_alignment` are the dot
  products between the post-FK configured sole normal and the exact target
  surface normal. A locked certified foot must reach at least `0.999`; checking
  only the target normal is insufficient.
- Existing source, query, matching, support, and simulation fields keep their sibling meanings. In particular, no pre-IK toe diagnostic is silently rebound to post-IK data; all post-IK physical measurements live in the suffix above.
- `ik_candidate_rejected` and the five candidate-clearance values preserve the
  evidence from the scratch pose that caused a safe-stop request. The ordinary
  clearance fields always describe the last accepted/rendered pose, including
  every safe-stop tail row.
- Produces Python entry points `read_invariance_rows(path)`,
  `check_gate_e_rows(rows)`, `compare_ik_invariance(off_rows, on_rows)`,
  `compare_ik_observations(off_rows, on_rows)`, and
  `check_gate_e_pair(off_rows, on_rows, expected_scene, expected_route)`. The
  CLI composes all existing checks and adds `--gate-e`, required
  `--ik-baseline PATH`, optional `--ik-off-reference PATH`,
  `--expected-scene ID`, and `--expected-route ID`.
- Certified-route Gate E requires `fixed_dt == 1/25`, no safe-stop request,
  exact paired contact/lock/target observations, finite/upward unit target
  normals, locked post-FK sole-normal alignment at least `0.999`, reachable
  targets, observed lock drift no greater than `0.20 m`, applied swing lift in
  `[0, 0.08]`, correction in `[0, 0.35]`, planted toe/foot clearance at least
  `-0.005 m`, every Hips/knee/ankle/toe/foot/thigh/shin clearance at least
  `-0.01 m`, and strictly lower aggregate planted horizontal drift with IK on.

- [ ] **Step 1: Write failing schema, invariance, clearance, and drift tests**

In `tests/python/test_runtime_log.py`, extend the sibling test helper so a valid row contains every exact IK suffix column. Add paired fixtures with at least twelve locked samples split across both feet, nonzero IK-off lock drift, smaller IK-on lock drift, and otherwise identical sibling fields. Add these tests:

```python
IK_SUFFIX = (
    "ik_applied", "ik_safe_stop_requested", "ik_stop_reason",
    "max_ik_correction", "actual_simulation_speed",
    "ik_candidate_rejected", "left_candidate_toe_clearance",
    "left_candidate_foot_clearance", "right_candidate_toe_clearance",
    "right_candidate_foot_clearance", "ik_candidate_minimum_clearance",
    "left_recorded_contact",
    "right_recorded_contact", "left_locked", "right_locked",
    "left_observed_lock_drift", "right_observed_lock_drift",
    "left_lock_drift", "right_lock_drift",
    "left_sole_normal_alignment", "right_sole_normal_alignment",
    "left_contact_residual", "right_contact_residual",
    "left_target_height",
    "right_target_height", "left_target_normal_x",
    "left_target_normal_y", "left_target_normal_z",
    "right_target_normal_x", "right_target_normal_y",
    "right_target_normal_z", "left_swing_required_lift",
    "left_swing_applied_lift", "left_swing_planned_margin",
    "left_swing_actual_margin", "right_swing_required_lift",
    "right_swing_applied_lift", "right_swing_planned_margin",
    "right_swing_actual_margin", "left_reachable", "right_reachable",
    "left_knee_clearance", "left_ankle_clearance",
    "left_toe_clearance", "left_foot_clearance",
    "left_shin_clearance", "left_thigh_clearance",
    "right_knee_clearance", "right_ankle_clearance",
    "right_toe_clearance", "right_foot_clearance",
    "right_shin_clearance", "right_thigh_clearance",
    "ik_hips_clearance", "ik_minimum_clearance",
)

def test_gate_e_accepts_bounded_clear_pair(self):
    off_rows, on_rows = self.gate_e_pair()
    check_gate_e_pair(
        off_rows, on_rows,
        "stairs-standard", "ascent-landing-descent")

def test_gate_e_rejects_any_matching_or_support_root_change(self):
    for column, changed in (
        ("query_bits_hex", "00000001" + "00000000" * 30),
        ("query_database_frame", "19"),
        ("terrain_point2_y", "0.12500001"),
        ("database_frame", "41"),
        ("range", "3"),
        ("transitioned", "1"),
        ("selected_cost", "9.5"),
        ("support_height", "0.30000001"),
        ("support_retargeted_hips_y", "0.81000001"),
        ("simulation_x", "0.50000001"),
    ):
        off_rows, on_rows = self.gate_e_pair()
        on_rows[5][column] = changed
        with self.subTest(column=column):
            with self.assertRaisesRegex(ValueError, column):
                compare_ik_invariance(off_rows, on_rows)

def test_gate_e_rejects_changed_contact_lock_or_target_observation(self):
    for column, changed in (
        ("left_recorded_contact", "0"),
        ("left_locked", "0"),
        ("left_observed_lock_drift", "0.12500001"),
        ("left_target_height", "0.12500001"),
        ("left_target_normal_x", "0.12500001"),
    ):
        off_rows, on_rows = self.gate_e_pair()
        on_rows[5][column] = changed
        with self.subTest(column=column):
            with self.assertRaisesRegex(ValueError, column):
                compare_ik_observations(off_rows, on_rows)

def test_gate_e_rejects_bounds_and_certified_stop(self):
    mutations = (
        ("max_ik_correction", "-0.00001", "correction"),
        ("max_ik_correction", "0.35001", "correction"),
        ("actual_simulation_speed", "-0.00001", "speed"),
        ("ik_candidate_rejected", "1", "rejected"),
        ("left_candidate_toe_clearance", "0.123", "candidate mismatch"),
        ("left_sole_normal_alignment", "0.9989", "alignment"),
        ("left_contact_residual", "0.00501", "residual"),
        ("left_swing_actual_margin", "-0.00002", "sweep"),
        ("left_foot_clearance", "-0.00501", "planted"),
        ("right_toe_clearance", "-0.01001", "physical"),
        ("left_shin_clearance", "-0.01001", "physical"),
        ("left_swing_applied_lift", "0.08001", "swing"),
        ("ik_safe_stop_requested", "1", "certified"),
        ("ik_stop_reason", "reach-shell", "certified"),
    )
    for column, value, diagnostic in mutations:
        off_rows, on_rows = self.gate_e_pair()
        on_rows[5][column] = value
        with self.subTest(column=column):
            with self.assertRaisesRegex(ValueError, diagnostic):
                check_gate_e_pair(
                    off_rows, on_rows,
                    "stairs-standard", "ascent-landing-descent")

def test_gate_e_requires_strictly_reduced_planted_drift(self):
    off_rows, on_rows = self.gate_e_pair()
    for before, after in zip(off_rows, on_rows):
        after["left_lock_drift"] = before["left_lock_drift"]
        after["right_lock_drift"] = before["right_lock_drift"]
    with self.assertRaisesRegex(ValueError, "drift did not decrease"):
        check_gate_e_pair(
            off_rows, on_rows,
            "stairs-standard", "ascent-landing-descent")
```

Also test duplicate/missing IK columns, unequal row counts, frame misalignment, a non-upward or non-unit target normal, `left_reachable=0`, required lift greater than applied lift on a certified swing, and a changed `continuation_cost`. Reuse the sibling CSV reader and valid Gate C row builder; do not add a second parser.

- [ ] **Step 2: Run the focused checker tests to verify RED**

Run:

```bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
```

Expected: existing sibling tests remain green, while the new imports or assertions fail because the Gate E functions and columns do not exist.

- [ ] **Step 3: Append the exact C++ row schema and writer fields**

In `motion_match_log.h`, append these types after the sibling row support fields:

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
    float swing_required_lift = 0.0f;
    float swing_applied_lift = 0.0f;
    float swing_planned_margin = 0.0f;
    float swing_actual_margin = 0.0f;
    float knee_clearance = 0.0f;
    float ankle_clearance = 0.0f;
    float toe_clearance = 0.0f;
    float foot_clearance = 0.0f;
    float shin_clearance = 0.0f;
    float thigh_clearance = 0.0f;
};

struct motion_match_ik_diagnostic
{
    bool applied = false;
    bool safe_stop_requested = false;
    const char* stop_reason = "none";
    float max_correction = 0.0f;
    float actual_simulation_speed = 0.0f;
    bool candidate_rejected = false;
    float left_candidate_toe_clearance = 0.0f;
    float left_candidate_foot_clearance = 0.0f;
    float right_candidate_toe_clearance = 0.0f;
    float right_candidate_foot_clearance = 0.0f;
    float candidate_minimum_clearance = 0.0f;
    motion_match_ik_leg_diagnostic left;
    motion_match_ik_leg_diagnostic right;
    float hips_clearance = 0.0f;
    float minimum_clearance = 0.0f;
};
```

Add `motion_match_ik_diagnostic ik;` as the final `motion_match_log_row` member. Append the exact suffix literal from this task to the existing header `fprintf`, then append its values to `write` with `%d` for booleans, `%s` for `stop_reason`, and `%.9g` for every float. Preserve the existing final newline by moving it to the end of the new suffix. Add a writer-level regression in `tests/python/test_runtime_log.py` that opens a controller-produced one-row CSV in Task 7's smoke and asserts `list(row) == sibling_columns + list(IK_SUFFIX)`.

- [ ] **Step 4: Fill the suffix only from immutable observations and post-IK FK**

In `controller.cpp`, immediately before the sibling `deterministic_log.write(log_row)`, fill the final diagnostic. Compute lock drift from the pose actually selected for rendering:

```cpp
static float g1_horizontal_lock_drift(
    const G1LegConfig& config,
    const G1FootLockState& lock,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations)
{
    if (!lock.locked) return 0.0f;
    vec3 sole[4];
    g1_sole_points_world(
        sole, global_positions(config.contact),
        global_rotations(config.contact), config);
    const vec3 center = g1_sole_center_world(sole);
    return length(vec3(
        center.x - lock.lock_point.x, 0.0f,
        center.z - lock.lock_point.z));
}

static float g1_sole_normal_alignment(
    const G1LegConfig& config,
    vec3 target_normal,
    const slice1d<quat> global_rotations)
{
    const vec3 actual_normal = quat_mul_vec3(
        global_rotations(config.contact), config.sole_normal_local);
    return dot(actual_normal, target_normal);
}
```

Use one explicit mapper so left/right cannot be crossed:

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
log_row.ik.left_candidate_toe_clearance =
    state.ik_candidate_clearance.left.toe_m;
log_row.ik.left_candidate_foot_clearance =
    state.ik_candidate_clearance.left.foot_m;
log_row.ik.right_candidate_toe_clearance =
    state.ik_candidate_clearance.right.toe_m;
log_row.ik.right_candidate_foot_clearance =
    state.ik_candidate_clearance.right.foot_m;
log_row.ik.candidate_minimum_clearance =
    state.ik_candidate_clearance.minimum_m;
for (int foot = 0; foot < 2; ++foot) {
    motion_match_ik_leg_diagnostic& dst = *ik_log_feet[foot];
    const G1FootFrameResult& src = state.ik_frame.feet[foot];
    const G1LegClearance& clearance = *ik_clearance_feet[foot];
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
    dst.swing_required_lift = src.swing.required_lift_m;
    dst.swing_applied_lift = src.swing.applied_lift_m;
    dst.swing_planned_margin = src.recorded_contact
        ? 0.0f : src.swing.corrected_margin_m;
    dst.swing_actual_margin = src.recorded_contact
        ? 0.0f : src.swing.actual_corrected_margin_m;
    dst.knee_clearance = clearance.knee_m;
    dst.ankle_clearance = clearance.ankle_m;
    dst.toe_clearance = clearance.toe_m;
    dst.foot_clearance = clearance.foot_m;
    dst.shin_clearance = clearance.shin_m;
    dst.thigh_clearance = clearance.thigh_m;
}
log_row.ik.hips_clearance = state.ik_clearance.hips_m;
log_row.ik.minimum_clearance = state.ik_clearance.minimum_m;
```

Fill the sibling row before this mapper from the same pre-IK values it already used. `state.ik_frame.applied` is exactly `MM_IK`, while contacts, locks, target samples, and sweep observations are updated in both modes.

- [ ] **Step 5: Implement raw-string invariance and Gate E bounds**

In `resources/check_g1_runtime_log.py`, add these exact invariance groups:

```python
IK_MATCHING_INVARIANTS = (
    "frame", "fixed_dt", "scene_id", "mode", "route",
    "query_bits_hex",
    "query_database_frame", "query_range", "database_frame", "range",
    "source_range", "searched", "transitioned", "incumbent_cost",
    "selected_cost", "selected_terrain_error", "effective_terrain_weight",
    "terrain0", "terrain1", "terrain2", "terrain3",
    "terrain_point0_x", "terrain_point0_y", "terrain_point0_z",
    "terrain_point1_x", "terrain_point1_y", "terrain_point1_z",
    "terrain_point2_x", "terrain_point2_y", "terrain_point2_z",
    "terrain_point3_x", "terrain_point3_y", "terrain_point3_z",
    "source_name", "source_terrain", "source_index",
    "continuation_cost",
)

IK_SUPPORT_ROOT_INVARIANTS = (
    "raw_selected_hips_y", "inertialized_hips_y",
    "rendered_hips_y", "hips_inertial_offset_y", "runtime_root_height",
    "adjustment_xz", "adjustment_y", "clamp_xz", "clamp_y",
    "matching_enabled", "adjustment_enabled", "clamping_enabled",
    "support_retargeting_enabled",
    "source_root_height", "runtime_support_root_height",
    "source_left_toe_height", "source_right_toe_height",
    "runtime_support_left_toe_height", "runtime_support_right_toe_height",
    "support_root_delta", "support_left_toe_delta", "support_right_toe_delta",
    "support_height", "support_velocity",
    "support_source", "airborne_frames", "left_contact", "right_contact",
    "support_retargeted_hips_y", "ik_adjusted_hips_y",
    "simulation_x", "simulation_z", "walkability_class", "blocked",
    "blocked_reason", "blocked_distance", "blocked_point_x",
    "blocked_point_z", "commanded_speed", "applied_speed",
    "route_waypoint", "route_complete", "route_target_height",
    "scene_generation", "scene_frame", "scene_reset_count",
    "scene_switch_failed", "motion_pack_load_count", "model_load_count",
    "model_unload_count", "live_model_count",
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

The sibling `check_rows` currently locks `ik_enabled == 0`. Replace only that
equality with a boolean `{0,1}` check. Keep explicit `ik_enabled == 0`
requirements in `diagnose_gate_a`, `check_gate_c`, `check_gate_d`,
`check_gate_f`, and `check_failed_switch`; Gate E alone requires `1`. This
preserves all earlier gates while allowing the shared base validator to read an
IK-on row. Append `ik_applied`, `ik_safe_stop_requested`,
`ik_candidate_rejected`, the recorded-contact/lock/reachable fields to the
shared boolean-column set, require `ik_stop_reason` nonempty, and finite-parse
every remaining numeric suffix field.

Refactor the sibling CSV reader's duplicate-column logic into one private
`_read_rows(path, required_columns)` helper. Keep `read_rows(path)` calling it
with the complete current schema, including every IK suffix field. Add
`read_runtime_rows(path)` using the locked sibling `RUNTIME_COLUMNS`, and make
the existing `--compare-control` path use it so preserved weight-zero Gate C
controls do not need new IK columns. Add `read_invariance_rows(path)` calling
the same helper with only
`IK_MATCHING_INVARIANTS + IK_SUPPORT_ROOT_INVARIANTS`; this is how the six
preserved pre-IK Gate C CSVs remain readable without weakening validation of
new logs. `compare_ik_invariance` first requires equal nonempty lengths, then
compares each pair's frame identity and every base invariant above with
ordinary string equality. `compare_ik_observations` requires full current
rows and exact-compares every `IK_PAIR_OBSERVATION_INVARIANTS` field. Its
diagnostic names the zero-based row, logged frame, column, baseline raw text,
and IK raw text. This list contains only observations derived from the common
support-retargeted input. Do not put solver results, correction, reachability,
residual, swing lift/margins, or stop output in it: swing history is committed
from the accepted pose, so those downstream results may legitimately differ
between modes even while matching and all pre-solve inputs remain invariant.
`check_gate_e_pair` and the stress traverse branch call both
comparators over the complete logs. The stress safe-stop branch calls both only
on the inclusive prefix through the first stop request; after that request the
next-frame command handoff intentionally changes simulation and downstream
observations. `--ik-off-reference` calls only the base comparator because the
accepted sibling logs predate the IK suffix. Do not include post-IK lock drift,
sole-normal alignment, physical clearance, candidate rejection/clearance, or
`ik_enabled` in either invariant list.

Implement bounds with finite parsing through the sibling helper. The core checks are:

```python
IK_PHYSICAL_CLEARANCE_COLUMNS = (
    "left_knee_clearance", "left_ankle_clearance",
    "left_toe_clearance", "left_foot_clearance",
    "left_shin_clearance", "left_thigh_clearance",
    "right_knee_clearance", "right_ankle_clearance",
    "right_toe_clearance", "right_foot_clearance",
    "right_shin_clearance", "right_thigh_clearance",
    "ik_hips_clearance",
)

def check_gate_e_rows(rows):
    check_rows(rows)
    for index, row in enumerate(rows):
        value = lambda name: _finite(row, name, index)
        def require(condition, message):
            if not condition:
                raise ValueError(f"row {index}: {message}")
        require(abs(value("fixed_dt") - 1.0 / 25.0) <= 1e-7,
                "Gate E fixed_dt is not 25 Hz")
        require(row["ik_enabled"] == "1" and row["ik_applied"] == "1",
                "Gate E IK-on row did not apply IK")
        require(row["ik_safe_stop_requested"] == "0" and
                row["ik_stop_reason"] == "none",
                "Gate E certified route requested safe stop")
        require(0.0 <= value("max_ik_correction") <= 0.35 + 1e-6,
                "Gate E correction exceeded 0.35")
        require(value("actual_simulation_speed") >= 0.0,
                "Gate E actual simulation speed is negative")
        require(row["ik_candidate_rejected"] == "0",
                "Gate E certified route rejected an IK candidate")
        for candidate, accepted in (
            ("left_candidate_toe_clearance", "left_toe_clearance"),
            ("left_candidate_foot_clearance", "left_foot_clearance"),
            ("right_candidate_toe_clearance", "right_toe_clearance"),
            ("right_candidate_foot_clearance", "right_foot_clearance"),
            ("ik_candidate_minimum_clearance", "ik_minimum_clearance"),
        ):
            require(abs(value(candidate) - value(accepted)) <= 1e-7,
                    f"Gate E accepted candidate mismatch: {candidate}")
        for side in ("left", "right"):
            require(row[f"{side}_reachable"] == "1",
                    f"Gate E {side} target left reachable shell")
            require(0.0 <= value(f"{side}_contact_residual") <= 0.005 + 1e-6,
                    f"Gate E {side} contact residual exceeded 0.005")
            normal = [value(f"{side}_target_normal_{axis}")
                      for axis in "xyz"]
            require(normal[1] > 0.0 and
                    abs(sum(value * value for value in normal) - 1.0) <= 2e-4,
                    f"Gate E {side} target normal is invalid")
            required = value(f"{side}_swing_required_lift")
            applied = value(f"{side}_swing_applied_lift")
            require(0.0 <= applied <= 0.08 + 1e-6 and
                    applied + 1e-6 >= required,
                    f"Gate E {side} swing lift is not bounded/complete")
            if row[f"{side}_recorded_contact"] == "0":
                require(value(f"{side}_swing_planned_margin") >= -1e-5 and
                        value(f"{side}_swing_actual_margin") >= -1e-5,
                        f"Gate E {side} corrected swing sweep penetrated")
            if row[f"{side}_locked"] == "1":
                require(value(f"{side}_observed_lock_drift") <= 0.20 + 1e-6,
                        f"Gate E {side} observed lock drift exceeded 0.20")
                require(value(f"{side}_sole_normal_alignment") >= 0.999,
                        f"Gate E {side} sole-normal alignment failed")
                require(value(f"{side}_toe_clearance") >= -0.005 - 1e-6 and
                        value(f"{side}_foot_clearance") >= -0.005 - 1e-6,
                        f"Gate E planted {side} toe/foot penetrated")
        for column in IK_PHYSICAL_CLEARANCE_COLUMNS:
            require(value(column) >= -0.01 - 1e-6,
                    f"Gate E physical clearance failed: {column}")
        require(value("ik_minimum_clearance") >= -0.01 - 1e-6,
                "Gate E physical minimum clearance failed")
```

`check_gate_e_pair` requires every off row to have `ik_enabled=0`, calls
`compare_ik_invariance`, `compare_ik_observations`, and `check_gate_e_rows`,
requires both logs to contain only the requested exact scene/route, requires a
`route_complete=1` row and no blocked/safe-stop tail, then collects each side's
`*_lock_drift` where the exact-compared `*_locked` field is `1`. Require at
least ten planted samples total and `mean(on_drift) < mean(off_drift)` with no
rounding tolerance. On CLI success print `VALID gate-e`, followed by the exact
scene ID, route ID, planted count, off mean, and on mean as named `key=value`
fields.

When `--ik-off-reference` is present, require the primary log to have only `ik_enabled=0`, load the older file with `read_invariance_rows`, and run `compare_ik_invariance(reference, rows)` before printing `VALID ik-off-reference frames=<n>`. This option is independent of `--gate-e`; it proves that enabling downstream observation code did not change the accepted support runtime when application remains disabled.

Also add `check_gate_e_stress_pair(off_rows, on_rows, expected_scene,
expected_route)` and `--gate-e-stress`. Reject any pair outside this exact
class-2 route set:

```python
GATE_E_STRESS_ROUTES = {
    ("grail-curb-default", "curb-forward"),
    ("grail-curb-medium", "curb-forward"),
    ("grail-curb-high", "curb-forward"),
    ("ramp-15-stress", "up-landing-down"),
}
```

It accepts exactly one of two branches:

1. **Traverse:** no IK stop occurs and the route completes. Run
   `check_gate_e_rows(on_rows)` and compare both base invariants and observation
   invariants over the complete pair. Do not impose the certified-route
   aggregate drift-reduction condition on a class-2 traversal; bounded
   clearance, residual, orientation, correction, and swing behavior are still
   mandatory.
2. **Safe stop:** locate the first on-row with
   `ik_safe_stop_requested=1`. It must occur before the first
   `route_complete=1` row, if the open-loop route driver later emits one, and
   the request row must have `ik_candidate_rejected=1`. Compare both invariant
   groups only on the inclusive prefix through that request; divergence after
   it is intentional because the command changes on the following update.
   Require the reason/evidence pair to be exact: `reach-shell` with either
   reachable flag false; `swing-lift` with required lift above `0.08` or actual
   corrected sweep margin below `-1e-5`; `correction-bound` with correction at
   least `0.35-1e-5`; `lock-drift` with observed drift above `0.20`; or
   `end-effector-residual` with residual above `0.005`. Reject `none`, unknown
   reasons, and `post-solve-clearance`. The request candidate must itself keep
   minimum clearance at least `-0.01` and, for each locked foot, candidate
   toe/foot clearance at least `-0.005`; this prevents a hidden earlier reason
   from laundering a post-solve penetration into a successful stress result.
   Every accepted/rendered on-row, including the complete tail, must retain
   the ordinary `-0.01` physical bounds and the planted `-0.005` bounds. On the
   row immediately after the request, require
   `actual_simulation_speed <= 1e-4`, then require that bound through the end;
   `applied_speed` alone is not stop evidence. From that first stopped row to
   the end, require total horizontal simulation displacement at most `0.02 m`
   and support-height rise at most `0.02 m`. Do not forbid a later
   `route_complete=1`: route completion is an open-loop script clock, not proof
   of physical traversal.

Both branches require finite rows and the exact requested scene/route. Print
`VALID gate-e-stress` with `branch=traverse` or `branch=safe-stop`, the stop
frame/reason when applicable, and named clearance, speed, displacement, and
support-rise measurements. Add focused passing fixtures for both branches and
negative tests for every reason/evidence mismatch, prefix mutation, accepted
tail penetration, candidate penetration, request after route completion,
next-frame residual speed, tail displacement/support rise, and an unlisted
scene/route.

- [ ] **Step 6: Run checker and paired smoke GREEN**

Run:

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

Expected: Python tests pass; both controller runs close normally; the checker prints exactly one `VALID gate-e` summary and exits `0`.

- [ ] **Step 7: Commit the diagnostic and acceptance contract**

```bash
git add motion_match_log.h controller.cpp \
  resources/check_g1_runtime_log.py tests/python/test_runtime_log.py
git commit -m "test: enforce G1 terrain IK Gate E"
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
cpp_tests=(
  test_g1_skeleton test_terrain_database test_terrain_runtime
  test_scene_runtime test_support_runtime test_g1_controller_state
  test_scene_switch test_route_runtime test_support_matching
  test_cleanup_runtime test_g1_ik test_g1_clearance
)
for name in "${cpp_tests[@]}"; do
  source="tests/cpp/${name}.cpp"
  test -f "$source"
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -I. \
    "$source" -o "/tmp/g1-ik-clearance/native/${name}-strict"
  "/tmp/g1-ik-clearance/native/${name}-strict"
  g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
    "$source" -o "/tmp/g1-ik-clearance/native/${name}-release"
  "/tmp/g1-ik-clearance/native/${name}-release"
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer -I. "$source" \
    -o "/tmp/g1-ik-clearance/native/${name}-san"
  ASAN_OPTIONS=detect_leaks=1 \
    "/tmp/g1-ik-clearance/native/${name}-san"
done
```

Expected: all 36 invocations exit `0`; strict builds emit no warning and sanitizers emit no report.

- [ ] **Step 3: Build the final non-sanitized controller and run source-order guards**

Run:

```bash
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_ik \
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
g++ -std=c++17 -O0 -g -Wall -Wextra -Werror -pedantic \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_visual_mesh.cpp -o /tmp/test_g1_visual_mesh \
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
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I. -I/home/ubuntu/apps/raylib/src \
  tests/cpp/test_g1_visual_mesh.cpp -o /tmp/test_g1_visual_mesh \
  -L/home/ubuntu/apps/raylib/src -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
/tmp/test_g1_visual_mesh
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I/home/ubuntu/apps/raylib/src -I/home/ubuntu/apps/raygui/src \
  controller.cpp -o /tmp/controller_g1_mesh \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
mkdir -p /tmp/g1-ik-clearance/visual
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
