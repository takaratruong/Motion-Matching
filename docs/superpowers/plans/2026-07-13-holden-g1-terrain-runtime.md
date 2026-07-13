# Holden G1 Terrain Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Extend Holden's C++ matcher from 27 to 31 features, consume validated G1 terrain artifacts, and pass sequential, flat-matching, and terrain-selection gates with IK disabled.

**Architecture:** Strict standalone loaders validate the Python artifact headers before extending Holden's database. Terrain sampling is a small renderer-independent C++ module used by both tests and controller queries. controller.cpp remains orchestration and Raylib visualization; database search and inertialization remain Holden's implementation.

**Tech Stack:** C++11/17-compatible g++; existing array.h, vec.h, quat.h, database.h, spring.h; raylib and raygui from /home/ubuntu/apps; Python standard-library unittest log checker from the artifact plan; DISPLAY=:1 for visual runs.

## Global Constraints

- Start only after the full artifact plan's diagnostic ten-clip build passes.
- Preserve the original resources/database.bin and resources/features.bin.
- Use G1 terrain assets from G1_TERRAIN_DIR, default resources/g1_terrain.
- Keep learned motion matching disabled.
- Keep inverse kinematics disabled throughout this plan.
- Keep Holden's 27 native feature values in their existing order and append four terrain values.
- Terrain weight zero must make terrain contribute zero cost without creating NaN or infinity in normalized queries.
- The current frame remains the incumbent; no K-frame restart, commitment timer, or synthetic root-height ramp is allowed.
- Runtime terrain features use distances 0.25, 0.50, 0.75, and 1.00 m along the geometric predicted-facing centerline.
- Fail with a nonzero exit and an actionable stderr message for every artifact mismatch.
- Preserve the user's existing controller.cpp, database.h, and spring.h changes unless a planned edit explicitly overlaps them.
- Run the controller and artifacts at a fixed 25 Hz. Use database trajectory offsets 8, 17, and 25 frames and exact one-third-second live prediction steps.

---

## File Map

- g1_skeleton.h: named G1 bone indices and skeleton signature checks.
- terrain_runtime.h: terrain-feature and heightfield schemas, strict loaders, interpolation, centerline sampling.
- database.h: optional terrain rows, 31-dimensional feature build, and explicit cost helper.
- motion_match_log.h: deterministic CSV schema and invariant-friendly event records.
- controller.cpp: artifact selection, query append, diagnostics, terrain rendering, and UI.
- tests/cpp/test_terrain_runtime.cpp: sidecar and heightfield loader/interpolation tests.
- tests/cpp/test_terrain_database.cpp: feature layout, zero weight, and incumbent search tests.
- tests/python/test_runtime_log.py: log invariant tests.
- resources/check_g1_runtime_log.py: acceptance checker.

### Task 1: Strict C++ sidecar and heightfield loaders

**Files:**
- Create: terrain_runtime.h
- Create: tests/cpp/test_terrain_runtime.cpp

**Interfaces:**
- Produces: bool terrain_features_load(terrain_feature_set&, const char*, char*, int).
- Produces: bool heightfield_load(heightfield&, const char*, char*, int).
- Produces: float heightfield_sample(const heightfield&, float x, float z).
- Sidecar header: G1TF, version 1, frame count, dimension 4.
- Heightfield header: G1HF, version 1, nx, nz, origin x/z, cell size, flat exterior height.

- [ ] **Step 1: Write loader and interpolation tests**

~~~cpp
// tests/cpp/test_terrain_runtime.cpp
#include "terrain_runtime.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void write_sidecar(const char* path)
{
    FILE* f = fopen(path, "wb");
    fwrite("G1TF", 1, 4, f);
    unsigned h[3] = {1, 2, 4};
    fwrite(h, sizeof(unsigned), 3, f);
    float values[8] = {0,1,2,3,4,5,6,7};
    fwrite(values, sizeof(float), 8, f);
    fclose(f);
}

static void write_heightfield(const char* path)
{
    FILE* f = fopen(path, "wb");
    fwrite("G1HF", 1, 4, f);
    unsigned h[3] = {1, 2, 2};
    fwrite(h, sizeof(unsigned), 3, f);
    float meta[4] = {0.0f, 0.0f, 1.0f, -1.0f};
    fwrite(meta, sizeof(float), 4, f);
    float values[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    fwrite(values, sizeof(float), 4, f);
    fclose(f);
}

int main()
{
    char err[256] = {};
    write_sidecar("/tmp/test_g1tf.bin");
    terrain_feature_set tf;
    assert(terrain_features_load(tf, "/tmp/test_g1tf.bin", err, sizeof(err)));
    assert(tf.values.rows == 2 && tf.values.cols == 4);
    assert(tf.values(1, 3) == 7.0f);

    write_heightfield("/tmp/test_g1hf.bin");
    heightfield hf;
    assert(heightfield_load(hf, "/tmp/test_g1hf.bin", err, sizeof(err)));
    assert(fabsf(heightfield_sample(hf, 0.5f, 0.5f) - 1.5f) < 1e-6f);
    assert(heightfield_sample(hf, -1.0f, 0.0f) == -1.0f);

    FILE* bad = fopen("/tmp/test_bad_g1tf.bin", "wb");
    fwrite("G1TF", 1, 4, bad);
    fclose(bad);
    terrain_feature_set rejected;
    assert(!terrain_features_load(
        rejected, "/tmp/test_bad_g1tf.bin", err, sizeof(err)));
    assert(strstr(err, "truncated") != NULL);
}
~~~

- [ ] **Step 2: Compile and verify the missing-header failure**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime
~~~

Expected: compilation fails because terrain_runtime.h is missing.

- [ ] **Step 3: Implement strict loaders and bilinear sampling**

Create terrain_runtime.h:

~~~cpp
#pragma once
#include "array.h"
#include "vec.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

struct terrain_feature_set { array2d<float> values; };
struct heightfield {
    int nx = 0, nz = 0;
    float origin_x = 0, origin_z = 0, cell_size = 0, exterior_height = 0;
    array1d<float> heights;
};

static inline bool terrain_error(
    char* out, int cap, const char* fmt, ...)
{
    va_list args; va_start(args, fmt);
    vsnprintf(out, cap, fmt, args);
    va_end(args);
    return false;
}

static inline bool read_exact(FILE* f, void* dst, size_t size)
{
    return fread(dst, 1, size, f) == size;
}

static inline bool terrain_features_load(
    terrain_feature_set& out, const char* path, char* err, int cap)
{
    FILE* f = fopen(path, "rb");
    if (!f) return terrain_error(err, cap, "%s: cannot open", path);
    char magic[4]; unsigned version, frames, dims;
    bool ok = read_exact(f, magic, 4) &&
              read_exact(f, &version, 4) &&
              read_exact(f, &frames, 4) &&
              read_exact(f, &dims, 4);
    if (!ok) { fclose(f); return terrain_error(err, cap, "%s: truncated header", path); }
    if (memcmp(magic, "G1TF", 4) || version != 1 || dims != 4) {
        fclose(f); return terrain_error(err, cap, "%s: unsupported G1TF schema", path);
    }
    out.values.resize((int)frames, (int)dims);
    if (!read_exact(f, out.values.data, frames * dims * sizeof(float))) {
        fclose(f); return terrain_error(err, cap, "%s: truncated values", path);
    }
    int extra = fgetc(f); fclose(f);
    if (extra != EOF) return terrain_error(err, cap, "%s: trailing bytes", path);
    return true;
}

static inline bool heightfield_load(
    heightfield& out, const char* path, char* err, int cap)
{
    FILE* f = fopen(path, "rb");
    if (!f) return terrain_error(err, cap, "%s: cannot open", path);
    char magic[4]; unsigned version, nx, nz;
    float meta[4];
    bool ok = read_exact(f, magic, 4) && read_exact(f, &version, 4) &&
              read_exact(f, &nx, 4) && read_exact(f, &nz, 4) &&
              read_exact(f, meta, sizeof(meta));
    if (!ok) { fclose(f); return terrain_error(err, cap, "%s: truncated header", path); }
    if (memcmp(magic, "G1HF", 4) || version != 1 || nx < 2 || nz < 2 || meta[2] <= 0) {
        fclose(f); return terrain_error(err, cap, "%s: invalid G1HF schema", path);
    }
    out.nx = (int)nx; out.nz = (int)nz;
    out.origin_x = meta[0]; out.origin_z = meta[1];
    out.cell_size = meta[2]; out.exterior_height = meta[3];
    out.heights.resize(out.nx * out.nz);
    if (!read_exact(f, out.heights.data, out.heights.size * sizeof(float))) {
        fclose(f); return terrain_error(err, cap, "%s: truncated heights", path);
    }
    int extra = fgetc(f); fclose(f);
    if (extra != EOF) return terrain_error(err, cap, "%s: trailing bytes", path);
    return true;
}

static inline float heightfield_sample(const heightfield& h, float x, float z)
{
    float gx = (x - h.origin_x) / h.cell_size;
    float gz = (z - h.origin_z) / h.cell_size;
    if (gx < 0 || gz < 0 || gx > h.nx - 1 || gz > h.nz - 1)
        return h.exterior_height;
    int x0 = (int)floorf(gx), z0 = (int)floorf(gz);
    int x1 = x0 < h.nx - 1 ? x0 + 1 : x0;
    int z1 = z0 < h.nz - 1 ? z0 + 1 : z0;
    float tx = gx - x0, tz = gz - z0;
    float a = lerpf(h.heights(z0*h.nx+x0), h.heights(z0*h.nx+x1), tx);
    float b = lerpf(h.heights(z1*h.nx+x0), h.heights(z1*h.nx+x1), tx);
    return lerpf(a, b, tz);
}
~~~

- [ ] **Step 4: Compile and run the loader test**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime
/tmp/test_terrain_runtime
~~~

Expected: exit 0 and no output.

- [ ] **Step 5: Commit strict runtime formats**

~~~bash
git add terrain_runtime.h tests/cpp/test_terrain_runtime.cpp
git commit -m "feat: load validated terrain runtime artifacts"
~~~

### Task 2: Named G1 skeleton contract

**Files:**
- Create: g1_skeleton.h
- Create: tests/cpp/test_g1_skeleton.cpp

**Interfaces:**
- Produces G1 bone constants for the 31-bone database including Simulation.
- Produces: bool g1_skeleton_validate(const database&, char*, int).

- [ ] **Step 1: Write the skeleton-order test**

~~~cpp
// tests/cpp/test_g1_skeleton.cpp
#include "database.h"
#include "g1_skeleton.h"
#include <assert.h>
#include <string.h>

int main()
{
    database db;
    db.bone_positions.resize(1, G1_BoneCount);
    db.bone_parents.resize(G1_BoneCount);
    const int expected[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,15,16,17,18,19,20,21,22,
        16,24,25,26,27,28,29
    };
    for (int i = 0; i < G1_BoneCount; ++i) db.bone_parents(i) = expected[i];
    char err[256] = {};
    assert(g1_skeleton_validate(db, err, sizeof(err)));
    db.bone_parents(G1_LeftToe) = G1_LeftKnee;
    assert(!g1_skeleton_validate(db, err, sizeof(err)));
    assert(strstr(err, "parent") != NULL);
}
~~~

- [ ] **Step 2: Verify compilation fails before the header exists**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_skeleton.cpp \
  -o /tmp/test_g1_skeleton
~~~

Expected: compilation fails because g1_skeleton.h is missing.

- [ ] **Step 3: Implement the named indices and parent validation**

~~~cpp
// g1_skeleton.h
#pragma once
#include "database.h"
#include <stdio.h>
#include <string.h>

enum G1Bone {
    G1_Simulation=0, G1_Hips=1,
    G1_LeftHipPitch=2, G1_LeftHipRoll=3, G1_LeftHipYaw=4,
    G1_LeftKnee=5, G1_LeftAnkle=6, G1_LeftToe=7,
    G1_RightHipPitch=8, G1_RightHipRoll=9, G1_RightHipYaw=10,
    G1_RightKnee=11, G1_RightAnkle=12, G1_RightToe=13,
    G1_Spine=14, G1_Spine1=15, G1_Spine2=16,
    G1_LeftShoulderPitch=17, G1_LeftShoulderRoll=18,
    G1_LeftShoulderYaw=19, G1_LeftElbow=20,
    G1_LeftWristRoll=21, G1_LeftWristPitch=22, G1_LeftWrist=23,
    G1_RightShoulderPitch=24, G1_RightShoulderRoll=25,
    G1_RightShoulderYaw=26, G1_RightElbow=27,
    G1_RightWristRoll=28, G1_RightWristPitch=29, G1_RightWrist=30,
    G1_BoneCount=31
};

static const char* G1_SkeletonSignature =
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7";

static inline bool g1_skeleton_validate(
    const database& db, char* err, int cap)
{
    static const int parents[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,15,16,17,18,19,20,21,22,
        16,24,25,26,27,28,29
    };
    if (db.nbones() != G1_BoneCount) {
        snprintf(err, cap, "G1 bone count mismatch: %d", db.nbones());
        return false;
    }
    for (int i = 0; i < G1_BoneCount; ++i) {
        if (db.bone_parents(i) != parents[i]) {
            snprintf(err, cap, "G1 parent mismatch at bone %d", i);
            return false;
        }
    }
    return true;
}

static inline bool g1_manifest_validate(
    const char* path, char* err, int cap)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        snprintf(err, cap, "%s: cannot open manifest", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        snprintf(err, cap, "%s: cannot size manifest", path);
        return false;
    }
    rewind(f);
    array1d<char> text((int)size + 1);
    bool ok = fread(text.data, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (!ok) {
        snprintf(err, cap, "%s: cannot read manifest", path);
        return false;
    }
    text((int)size) = '\0';
    if (!strstr(text.data, G1_SkeletonSignature)) {
        snprintf(err, cap, "%s: G1 skeleton signature mismatch", path);
        return false;
    }
    return true;
}
~~~

- [ ] **Step 4: Compile and run the skeleton test**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_skeleton.cpp \
  -o /tmp/test_g1_skeleton
/tmp/test_g1_skeleton
~~~

Expected: exit 0.

- [ ] **Step 5: Replace anonymous G1 index literals**

Include g1_skeleton.h from controller.cpp after database.h. Change
database_build_matching_features to accept three final integer parameters
left_foot_bone, right_foot_bone, and hip_bone, and use those parameters in its
five bone feature calls. controller.cpp passes G1_LeftAnkle, G1_RightAnkle, and
G1_Hips. Replace contact literals 7 and 13 with G1_LeftToe and G1_RightToe.
database.h does not include g1_skeleton.h, avoiding an include cycle.

- [ ] **Step 6: Commit the skeleton contract**

~~~bash
git add g1_skeleton.h tests/cpp/test_g1_skeleton.cpp database.h controller.cpp
git commit -m "refactor: name the G1 runtime skeleton"
~~~

### Task 3: Append four terrain dimensions to Holden's feature database

**Files:**
- Modify: database.h:100-161, 528-567
- Create: tests/cpp/test_terrain_database.cpp

**Interfaces:**
- database gains array2d<float> terrain_features.
- database_build_matching_features gains terrain weight and named bone indices.
- Produces: database_frame_cost(db, frame, query) -> float.
- Produces 31 normalized features in the existing 27+4 order.
- Retimes Holden's trajectory database horizons from 20/40/60 at 60 Hz to 8/17/25 at 25 Hz.

- [ ] **Step 1: Write the feature-layout and zero-weight tests**

~~~cpp
// tests/cpp/test_terrain_database.cpp
#include "database.h"
#include "g1_skeleton.h"
#include <assert.h>
#include <float.h>
#include <math.h>

int main()
{
    array2d<float> f(3, 4);
    array1d<float> off(4), scale(4);
    for (int i=0;i<3;i++) for (int j=0;j<4;j++) f(i,j) = (float)(i+j);
    normalize_feature(f, off, scale, 0, 4, 0.0f);
    for (int i=0;i<3;i++) for (int j=0;j<4;j++)
        assert(fabsf(f(i,j)) < 1e-20f);
    for (int j=0;j<4;j++) assert(scale(j) == FLT_MAX);

    database db;
    db.features.resize(2, 31);
    db.features_offset.resize(31); db.features_offset.zero();
    db.features_scale.resize(31); db.features_scale.set(1.0f);
    db.features.zero();
    array1d<float> query(31); query.zero();
    query(30) = 2.0f;
    assert(database_frame_cost(db, 0, query) == 4.0f);

    int horizons[3];
    database_trajectory_horizons(horizons);
    assert(horizons[0] == 8 && horizons[1] == 17 && horizons[2] == 25);
}
~~~

- [ ] **Step 2: Compile and verify failures for the new behavior**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_database.cpp \
  -o /tmp/test_terrain_database
~~~

Expected: compilation fails because database_frame_cost is undefined.

- [ ] **Step 3: Make zero weight explicitly produce zero normalized cost**

Change normalize_feature's scale assignment and normalization:

~~~cpp
float scale = weight > 0.0f ? std / weight : FLT_MAX;
for (int j = 0; j < size; j++) features_scale(offset + j) = scale;
for (int i = 0; i < features.rows; i++)
for (int j = 0; j < size; j++)
    features(i, offset + j) = weight > 0.0f
        ? (features(i, offset + j) - features_offset(offset + j)) / scale
        : 0.0f;
~~~

Because query normalization divides by FLT_MAX, terrain query values also
normalize to zero at weight zero.

- [ ] **Step 4: Append and normalize terrain rows**

Add to database:

~~~cpp
array2d<float> terrain_features;
~~~

Add:

~~~cpp
static inline void compute_terrain_feature(
    database& db, int& offset, float weight)
{
    assert(db.terrain_features.rows == db.nframes());
    assert(db.terrain_features.cols == 4);
    for (int i=0;i<db.nframes();++i)
        for (int j=0;j<4;++j)
            db.features(i, offset+j) = db.terrain_features(i,j);
    normalize_feature(db.features, db.features_offset, db.features_scale,
                      offset, 4, weight);
    offset += 4;
}

static inline float database_frame_cost(
    const database& db, int frame, const slice1d<float> query)
{
    float cost = 0.0f;
    for (int i=0;i<db.nfeatures();++i) {
        float q = (query(i)-db.features_offset(i))/db.features_scale(i);
        cost += squaref(q-db.features(frame,i));
    }
    return cost;
}

static inline float database_raw_terrain_error(
    const database& db, int frame, const slice1d<float> query)
{
    float error = 0.0f;
    for (int j=0;j<4;++j)
        error += squaref(query(27+j)-db.terrain_features(frame,j));
    return error;
}
~~~

Change nfeatures from 27 to 31, accept feature_weight_terrain, and call
compute_terrain_feature after trajectory directions. Preserve every existing
native feature offset.

Add database_trajectory_horizons and use it in both trajectory position and
direction feature builders:

~~~cpp
static inline void database_trajectory_horizons(int out[3])
{
    out[0] = 8; out[1] = 17; out[2] = 25;
}
~~~

- [ ] **Step 5: Compile and run all C++ unit tests**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
g++ -std=c++17 -I. tests/cpp/test_g1_skeleton.cpp -o /tmp/test_g1_skeleton
g++ -std=c++17 -I. tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database
/tmp/test_terrain_runtime
/tmp/test_g1_skeleton
/tmp/test_terrain_database
~~~

Expected: all exit 0.

- [ ] **Step 6: Commit the 31-dimensional matcher**

~~~bash
git add database.h tests/cpp/test_terrain_database.cpp
git commit -m "feat: append terrain to Holden matching features"
~~~

### Task 4: Runtime centerline and terrain query

**Files:**
- Modify: terrain_runtime.h
- Modify: tests/cpp/test_terrain_runtime.cpp
- Modify: controller.cpp:1726-1745

**Interfaces:**
- Produces: terrain_centerline_query(out4, heightfield, root, trajectory positions, trajectory rotations).
- Appends four denormalized terrain values to the live query.

- [ ] **Step 1: Add a stationary and straight-path query test**

Append to tests/cpp/test_terrain_runtime.cpp:

~~~cpp
    array1d<vec3> pos(4);
    array1d<quat> rot(4);
    for (int i=0;i<4;i++) { pos(i)=vec3(0,0,0); rot(i)=quat(); }
    float query[4];
    terrain_centerline_query(query, hf, vec3(0,0,0), pos, rot);
    assert(isfinite(query[0]) && isfinite(query[3]));
~~~

- [ ] **Step 2: Compile and verify terrain_centerline_query is missing**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime
~~~

Expected: compilation fails because terrain_centerline_query is undefined.

- [ ] **Step 3: Implement centerline arc sampling**

Add to terrain_runtime.h:

~~~cpp
static inline vec3 point_at_arc(
    const slice1d<vec3> points, float distance, vec3 final_dir)
{
    for (int i=1;i<points.size;i++) {
        vec3 d = points(i)-points(i-1);
        float len = length(d);
        if (len > 1e-6f && distance <= len)
            return points(i-1) + d * (distance/len);
        distance -= len;
    }
    return points(points.size-1) + final_dir * maxf(distance, 0.0f);
}

static inline void terrain_centerline_query(
    float out[4], const heightfield& h, vec3 root,
    const slice1d<vec3> trajectory_positions,
    const slice1d<quat> trajectory_rotations)
{
    const float distances[4] = {0.25f,0.50f,0.75f,1.00f};
    vec3 forward = quat_mul_vec3(
        trajectory_rotations(trajectory_rotations.size-1), vec3(0,0,1));
    forward.y = 0.0f;
    forward = length(forward) > 1e-6f ? normalize(forward) : vec3(0,0,1);
    float base = heightfield_sample(h, root.x, root.z);
    for (int i=0;i<4;i++) {
        vec3 p = point_at_arc(trajectory_positions, distances[i], forward);
        out[i] = heightfield_sample(h, p.x, p.z) - base;
    }
}
~~~

- [ ] **Step 4: Append live terrain values in controller.cpp**

After query_compute_trajectory_direction_feature:

~~~cpp
float terrain_query[4];
terrain_centerline_query(
    terrain_query, runtime_terrain, bone_positions(0),
    trajectory_positions, trajectory_rotations);
for (int i=0;i<4;i++) query(offset++) = terrain_query[i];
assert(offset == db.nfeatures());
~~~

The runtime_terrain object is loaded once during startup in Task 5. Do not derive
root height from terrain_query; it affects matching cost only.

- [ ] **Step 5: Run the terrain unit test**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp \
  -o /tmp/test_terrain_runtime
/tmp/test_terrain_runtime
~~~

Expected: exit 0.

- [ ] **Step 6: Commit the runtime query**

~~~bash
git add terrain_runtime.h tests/cpp/test_terrain_runtime.cpp controller.cpp
git commit -m "feat: query terrain along Holden trajectory"
~~~

### Task 5: Load the artifact set and render terrain diagnostics

**Files:**
- Modify: controller.cpp:1310-1345, 1484-1500, 2290-2375, 2475-2516

**Interfaces:**
- Environment G1_TERRAIN_DIR defaults to ./resources/g1_terrain.
- Loads database.bin, terrain_features.bin, terrain.bin, and terrain.obj.
- UI exposes terrain weight and rebuilds all 31 features.
- IK starts false; learned matching starts false.
- Controller target rate and fixed dt are 25 Hz; predicted trajectory samples are one-third second apart.

- [ ] **Step 1: Add fail-fast artifact startup**

Construct paths with snprintf into fixed 1024-byte buffers:

~~~cpp
const char* terrain_dir = getenv("G1_TERRAIN_DIR");
if (!terrain_dir) terrain_dir = "./resources/g1_terrain";
char database_path[1024], feature_path[1024], heightfield_path[1024];
char mesh_path[1024], manifest_path[1024];
snprintf(database_path, sizeof(database_path), "%s/database.bin", terrain_dir);
snprintf(feature_path, sizeof(feature_path), "%s/terrain_features.bin", terrain_dir);
snprintf(heightfield_path, sizeof(heightfield_path), "%s/terrain.bin", terrain_dir);
snprintf(mesh_path, sizeof(mesh_path), "%s/terrain.obj", terrain_dir);
snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", terrain_dir);

char artifact_error[512] = {};
FILE* database_probe = fopen(database_path, "rb");
if (!database_probe) {
    fprintf(stderr, "G1 terrain database cannot open: %s\n", database_path);
    return 2;
}
fclose(database_probe);
if (!g1_manifest_validate(
        manifest_path, artifact_error, sizeof(artifact_error))) {
    fprintf(stderr, "G1 manifest error: %s\n", artifact_error);
    return 2;
}
database_load(db, database_path);
terrain_feature_set terrain_rows;
heightfield runtime_terrain;
if (!terrain_features_load(terrain_rows, feature_path, artifact_error, sizeof(artifact_error)) ||
    !heightfield_load(runtime_terrain, heightfield_path, artifact_error, sizeof(artifact_error))) {
    fprintf(stderr, "G1 terrain artifact error: %s\n", artifact_error);
    return 2;
}
if (terrain_rows.values.rows != db.nframes()) {
    fprintf(stderr, "G1 terrain frame mismatch: database=%d sidecar=%d\n",
            db.nframes(), terrain_rows.values.rows);
    return 2;
}
db.terrain_features = terrain_rows.values;
if (!g1_skeleton_validate(db, artifact_error, sizeof(artifact_error))) {
    fprintf(stderr, "G1 skeleton error: %s\n", artifact_error);
    return 2;
}
~~~

Set `SetTargetFPS(25)` and `dt = 1.0f / 25.0f`. Replace every `20.0f * dt`
argument used to build the four live trajectory samples with the explicit
constant `trajectory_sample_time = 1.0f / 3.0f`; this preserves real-time
horizons rather than inheriting Holden's old 60 Hz frame count.

- [ ] **Step 2: Make safe phase-one defaults explicit**

Set:

~~~cpp
bool lmm_enabled = false;
bool ik_enabled = false;
float feature_weight_terrain = 0.0f;
~~~

Pass feature_weight_terrain to every database_build_matching_features call,
including the startup build and GUI rebuild button.

- [ ] **Step 3: Load and draw the terrain mesh**

At startup:

~~~cpp
Model terrain_model = LoadModel(mesh_path);
if (terrain_model.meshCount == 0) {
    fprintf(stderr, "G1 terrain mesh failed to load: %s\n", mesh_path);
    return 2;
}
~~~

In the 3D draw block:

~~~cpp
DrawModel(terrain_model, (Vector3){0,0,0}, 1.0f, WHITE);
for (int i=0;i<4;i++) {
    vec3 p = point_at_arc(
        trajectory_positions, 0.25f*(i+1),
        normalize(quat_mul_vec3(
            trajectory_rotations(trajectory_rotations.size-1), vec3(0,0,1))));
    p.y = heightfield_sample(runtime_terrain, p.x, p.z);
    DrawSphereWires(to_Vector3(p), 0.04f, 4, 8, PURPLE);
}
~~~

- [ ] **Step 4: Add terrain-weight UI and diagnostics**

Increase the feature group box height by 30 pixels and add:

~~~cpp
GuiSliderBar(
    (Rectangle){150, 180, 120, 20},
    "terrain", TextFormat("%5.3f", feature_weight_terrain),
    &feature_weight_terrain, 0.0f, 10.0f);
~~~

Move the rebuild button to y=210 and add:

~~~cpp
GuiLabel((Rectangle){20, 215, 290, 20},
         TextFormat("frame %d  range %d", frame_index, active_range));
GuiLabel((Rectangle){20, 235, 290, 20},
         TextFormat("terrain %.2f %.2f %.2f %.2f",
                    terrain_query[0], terrain_query[1],
                    terrain_query[2], terrain_query[3]));
~~~

Compute active_range by scanning db.range_starts/stops for frame_index once per
frame; use -1 only if no range contains it, which the runtime log checker rejects.

- [ ] **Step 5: Build the Raylib controller**

Run:

~~~bash
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I /home/ubuntu/apps/raylib/src -I /home/ubuntu/apps/raygui/src \
  controller.cpp -o controller_g1_terrain \
  -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
~~~

Expected: exit 0 and controller_g1_terrain exists.

- [ ] **Step 6: Verify fail-fast behavior**

Run:

~~~bash
DISPLAY=:1 G1_TERRAIN_DIR=/tmp/does-not-exist \
  ./controller_g1_terrain
~~~

Expected: exit 2 with G1 terrain artifact error and the missing path.

- [ ] **Step 7: Commit artifact integration**

~~~bash
git add controller.cpp
git commit -m "feat: run Holden matcher on G1 terrain artifacts"
~~~

### Task 6: Deterministic matching log and invariant checker

**Files:**
- Create: motion_match_log.h
- Create: resources/check_g1_runtime_log.py
- Create: tests/python/test_runtime_log.py
- Modify: controller.cpp:1750-1883, 1913-1924

**Interfaces:**
- MM_LOG selects CSV output.
- MM_TEST_MODE accepts sequential, flat, or terrain.
- MM_TEST_FRAMES selects deterministic length and exits after that many frames.
- CSV columns: frame, mode, database_frame, range, searched, transitioned,
  incumbent_cost, selected_cost, selected_terrain_error, terrain0..3, source_range.

- [ ] **Step 1: Write log-checker tests**

~~~python
# tests/python/test_runtime_log.py
import tempfile
import unittest
from resources.check_g1_runtime_log import check_rows, read_rows


class RuntimeLogTests(unittest.TestCase):
    def test_rejects_nonsequential_advance_without_transition(self):
        rows = [
            {"frame":"0","database_frame":"10","transitioned":"0","selected_cost":"1","incumbent_cost":"1","selected_terrain_error":"0"},
            {"frame":"1","database_frame":"4","transitioned":"0","selected_cost":"1","incumbent_cost":"1","selected_terrain_error":"0"},
        ]
        with self.assertRaisesRegex(ValueError, "nonsequential"):
            check_rows(rows)

    def test_rejects_transition_with_higher_cost(self):
        rows = [
            {"frame":"0","database_frame":"10","transitioned":"1","selected_cost":"2","incumbent_cost":"1","selected_terrain_error":"0"},
        ]
        with self.assertRaisesRegex(ValueError, "cost"):
            check_rows(rows)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run and verify the checker import failure**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
~~~

Expected: ERROR because check_g1_runtime_log.py does not exist.

- [ ] **Step 3: Implement the invariant checker**

~~~python
# resources/check_g1_runtime_log.py
import argparse
import csv


def read_rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def check_rows(rows):
    for i, row in enumerate(rows):
        current = int(row["database_frame"])
        transitioned = int(row["transitioned"])
        if transitioned and float(row["selected_cost"]) >= float(row["incumbent_cost"]):
            raise ValueError(f"row {i}: transition cost did not beat incumbent")
        if i and not transitioned:
            previous = int(rows[i-1]["database_frame"])
            if current != previous + 1:
                raise ValueError(f"row {i}: nonsequential advance {previous}->{current}")
    check_substride(rows)
    return {"frames": len(rows), "transitions": sum(int(r["transitioned"]) for r in rows)}


def check_substride(rows, minimum_period=13):
    values = [int(row["database_frame"]) for row in rows]
    transitions = [int(row["transitioned"]) for row in rows]
    for period in range(1, minimum_period):
        width = 3 * period
        for start in range(0, len(values)-width+1):
            if any(transitions[start+1:start+width]):
                continue
            a = values[start:start+period]
            if a == values[start+period:start+2*period] == \
                    values[start+2*period:start+3*period]:
                raise ValueError(
                    f"row {start}: repeated sub-stride period {period}")


def compare_control(treatment, control):
    if len(treatment) != len(control):
        raise ValueError("control and treatment lengths differ")
    active = [
        i for i, row in enumerate(treatment)
        if max(float(row[f"terrain{j}"]) for j in range(4)) > 0.05
    ]
    if not active:
        raise ValueError("terrain query never became active")
    treatment_error = sum(
        float(treatment[i]["selected_terrain_error"]) for i in active) / len(active)
    control_error = sum(
        float(control[i]["selected_terrain_error"]) for i in active) / len(active)
    if treatment_error >= control_error:
        raise ValueError(
            f"terrain treatment did not improve cost: "
            f"{treatment_error} >= {control_error}")
    return treatment_error, control_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--compare-control")
    args = parser.parse_args()
    rows = read_rows(args.log)
    result = check_rows(rows)
    if args.compare_control:
        treatment, control = compare_control(rows, read_rows(args.compare_control))
        print(f"VALID terrain-comparison treatment={treatment:.6g} control={control:.6g}")
    print(f"VALID runtime-log frames={result['frames']} transitions={result['transitions']}")
~~~

- [ ] **Step 4: Add a CSV writer with explicit fields**

~~~cpp
// motion_match_log.h
#pragma once
#include <stdio.h>

struct motion_match_log {
    FILE* file = NULL;

    bool open(const char* path) {
        if (!path) return true;
        file = fopen(path, "w");
        if (!file) return false;
        fprintf(file,
          "frame,mode,database_frame,range,searched,transitioned,"
          "incumbent_cost,selected_cost,selected_terrain_error,"
          "terrain0,terrain1,terrain2,terrain3,source_range\n");
        return true;
    }

    void write(
        int frame, const char* mode, int database_frame, int range,
        bool searched, bool transitioned, float incumbent_cost,
        float selected_cost, float selected_terrain_error,
        const float terrain[4], int source_range) {
        if (!file) return;
        fprintf(file,
          "%d,%s,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
          frame, mode, database_frame, range, (int)searched, (int)transitioned,
          incumbent_cost, selected_cost, selected_terrain_error,
          terrain[0], terrain[1], terrain[2], terrain[3], source_range);
        fflush(file);
    }

    void close() {
        if (file) { fflush(file); fclose(file); file = NULL; }
    }
};
~~~

- [ ] **Step 5: Capture incumbent and selected costs around search**

Before database_search:

~~~cpp
float incumbent_cost = end_of_anim
    ? FLT_MAX : database_frame_cost(db, frame_index, query);
int previous_index = frame_index;
~~~

After search and transition decision:

~~~cpp
float selected_cost = database_frame_cost(db, best_index, query);
float selected_terrain_error = database_raw_terrain_error(
    db, best_index, query);
bool transitioned = best_index != previous_index;
~~~

Write one row after frame advancement. Use database_trajectory_index_clamp and
range arrays to derive the active range. In sequential mode, skip searching and
advance only with database_trajectory_index_clamp rather than raw frame_index++.

- [ ] **Step 6: Add deterministic modes**

Read environment once:

~~~cpp
const char* test_mode = getenv("MM_TEST_MODE");
int test_frames = getenv("MM_TEST_FRAMES") ? atoi(getenv("MM_TEST_FRAMES")) : 0;
~~~

- sequential: no search, adjustment false, clamping false, IK false.
- flat: normal search, terrain weight forced to zero, IK false.
- terrain: normal search, terrain weight from MM_TERRAIN_WEIGHT default 4, IK false.

At test_frames, close the log, close the window, and return through the normal
main-loop termination path; do not call _Exit before rendering and cleanup.

- [ ] **Step 7: Run Python checker tests and rebuild**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I /home/ubuntu/apps/raylib/src -I /home/ubuntu/apps/raygui/src \
  controller.cpp -o controller_g1_terrain \
  -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
~~~

Expected: Python tests OK and compilation exit 0.

- [ ] **Step 8: Commit deterministic diagnostics**

~~~bash
git add motion_match_log.h resources/check_g1_runtime_log.py \
  tests/python/test_runtime_log.py controller.cpp
git commit -m "test: log deterministic Holden matching"
~~~

### Task 7: Sequential, flat, and terrain acceptance gates

**Files:**
- Generated, not committed: /tmp/g1_sequential.csv, /tmp/g1_flat.csv, /tmp/g1_terrain.csv
- Modify only if a failing gate identifies a root cause: files named in prior tasks.

**Interfaces:**
- Consumes diagnostic ten-clip artifacts first, then full artifacts.
- Produces evidence for Gates 2, 3, and 4 of the approved design.

- [ ] **Step 1: Run all nonvisual tests from a clean build**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
g++ -std=c++17 -I. tests/cpp/test_g1_skeleton.cpp -o /tmp/test_g1_skeleton
g++ -std=c++17 -I. tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database
/tmp/test_terrain_runtime
/tmp/test_g1_skeleton
/tmp/test_terrain_database
~~~

Expected: every test exits 0.

- [ ] **Step 2: Run the sequential playback gate**

Run:

~~~bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=sequential MM_TEST_FRAMES=250 MM_LOG=/tmp/g1_sequential.csv \
  ./controller_g1_terrain
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_sequential.csv
~~~

Expected: 250 frames, zero transitions, sequential database indices, and exit 0.
Visually inspect the Raylib skeleton during the run for source-faithful legs with
IK disabled.

- [ ] **Step 3: Run the flat matching gate**

Run:

~~~bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=flat MM_TEST_FRAMES=375 MM_LOG=/tmp/g1_flat.csv \
  ./controller_g1_terrain
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_flat.csv
~~~

Expected: 375 valid rows; every transition beats its incumbent; no nontransition
index reset; no repeated sub-stride cycle shorter than 13 frames (about 0.5 s).

- [ ] **Step 4: Verify sub-stride detection with synthetic tests**

Add:

~~~python
    def test_rejects_three_repeated_short_cycles(self):
        rows = []
        for frame, index in enumerate(list(range(10,18))*3):
            rows.append({
                "frame":str(frame), "database_frame":str(index),
                "transitioned":"0", "selected_cost":"1",
                "incumbent_cost":"1", "selected_terrain_error":"0",
            })
        with self.assertRaisesRegex(ValueError, "sub-stride"):
            check_substride(rows)

    def test_accepts_long_sequential_run(self):
        rows = [{
            "frame":str(i), "database_frame":str(10+i),
            "transitioned":"0", "selected_cost":"1",
            "incumbent_cost":"1", "selected_terrain_error":"0",
        } for i in range(40)]
        check_substride(rows)
~~~

Run tests/python/test_runtime_log.py and require both tests to pass.

- [ ] **Step 5: Run the terrain-on gate**

Run:

~~~bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 \
  MM_TEST_FRAMES=375 MM_LOG=/tmp/g1_terrain.csv \
  ./controller_g1_terrain
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_terrain.csv
~~~

Expected: terrain query becomes positive before the curb; at least one transition
selects a GRAIL range; the selected range advances sequentially through the step.

- [ ] **Step 6: Add and run the A/B terrain assertion**

Add a unit test with two active rows whose treatment selected_terrain_error values
are 0.5 and whose control values are 2.0; compare_control must return treatment
0.5 and control 2.0. The compare_control implementation from Task 6 aligns rows
by list position, activates rows from treatment terrain0..3, and rejects
treatment error greater than or equal to control.

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_terrain.csv \
  --compare-control /tmp/g1_flat.csv
~~~

Expected: VALID comparison and terrain-on error below control.

- [ ] **Step 7: Build and run the live debug skeleton**

Run:

~~~bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain ./controller_g1_terrain
~~~

Drive with WASD and confirm the UI shows the predicted trajectory, four terrain
sample spheres, selected source range, and terrain weight. Close the window
normally.

- [ ] **Step 8: Verify repository scope and commit acceptance tooling**

Run:

~~~bash
git diff --check
git status --short
~~~

Commit only checker or diagnostic changes made in Steps 4 and 6:

~~~bash
git add resources/check_g1_runtime_log.py tests/python/test_runtime_log.py
git commit -m "test: enforce terrain matching acceptance gates"
~~~

### Task 8: Record the deterministic terrain comparison

**Files:**
- Generated, not committed: resources/g1_terrain/evidence/terrain_off.mp4
- Generated, not committed: resources/g1_terrain/evidence/terrain_on.mp4
- Generated, not committed: resources/g1_terrain/evidence/terrain_compare.mp4
- Generated, not committed: resources/g1_terrain/evidence/terrain_off.csv
- Generated, not committed: resources/g1_terrain/evidence/terrain_on.csv

**Interfaces:**
- Consumes the passing Task 7 executable and full artifacts.
- Produces the approved comparison video and its numeric logs.

- [ ] **Step 1: Create the ignored evidence directory**

~~~bash
mkdir -p resources/g1_terrain/evidence
~~~

- [ ] **Step 2: Record the terrain-weight-zero control**

~~~bash
DISPLAY=:1 ffmpeg -y -loglevel error -f x11grab \
  -video_size 1280x720 -framerate 25 -i :1.0+0,0 -t 15 \
  resources/g1_terrain/evidence/terrain_off.mp4 &
CAPTURE_PID=$!
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=0 MM_TEST_FRAMES=375 \
  MM_LOG=resources/g1_terrain/evidence/terrain_off.csv \
  ./controller_g1_terrain
wait "$CAPTURE_PID"
~~~

Expected: controller and ffmpeg both exit 0 and the CSV has 375 rows.

- [ ] **Step 3: Record the terrain-aware treatment**

~~~bash
DISPLAY=:1 ffmpeg -y -loglevel error -f x11grab \
  -video_size 1280x720 -framerate 25 -i :1.0+0,0 -t 15 \
  resources/g1_terrain/evidence/terrain_on.mp4 &
CAPTURE_PID=$!
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_TEST_FRAMES=375 \
  MM_LOG=resources/g1_terrain/evidence/terrain_on.csv \
  ./controller_g1_terrain
wait "$CAPTURE_PID"
~~~

Expected: controller and ffmpeg both exit 0 and the CSV has 375 rows.

- [ ] **Step 4: Verify the A/B logs**

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py \
  resources/g1_terrain/evidence/terrain_on.csv \
  --compare-control resources/g1_terrain/evidence/terrain_off.csv
~~~

Expected: VALID terrain-comparison with treatment error below control.

- [ ] **Step 5: Compose and inspect the comparison**

~~~bash
ffmpeg -y -loglevel error \
  -i resources/g1_terrain/evidence/terrain_off.mp4 \
  -i resources/g1_terrain/evidence/terrain_on.mp4 \
  -filter_complex hstack=inputs=2 \
  resources/g1_terrain/evidence/terrain_compare.mp4
ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1 \
  resources/g1_terrain/evidence/terrain_compare.mp4
~~~

Expected: duration approximately 15 seconds. Inspect the video once and confirm
the left control and right treatment use the same scripted input and camera.
