# G1 Terrain IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace Holden's LAFAN/flat-ground IK assumptions with a bounded, reversible G1 foot-lock stage that samples the runtime terrain.

**Architecture:** A renderer-independent G1 leg configuration names every solver joint and axis. Terrain-aware contact targets are computed before IK, then the existing two-bone mathematics is reused behind bounded G1 adapters. Matching and root motion are immutable inputs; IK produces only adjusted local rotations and can be disabled to recover byte-equivalent Gate 4 output.

**Tech Stack:** Existing C++ vec/quaternion/FK/spring functions; terrain_runtime.h heightfield; standard assert-based C++ tests; Raylib visualization and deterministic CSV acceptance logs.

## Global Constraints

- Start only after the runtime plan passes sequential, flat, and terrain Gate 4 with IK disabled.
- Do not change database features, selected frames, matching costs, simulation root, or recorded root height.
- Configure both G1 legs explicitly; do not infer a LAFAN two-joint chain by walking parents from a toe.
- Use terrain height at each contact target, never a constant world Y.
- Use G1_LeftHipYaw/G1_RightHipYaw as the analytic solver hip, the named knee and ankle, and the named toe/contact bone.
- Do not apply Holden's hard-coded +X toe-end vector to the G1.
- Clamp targets to the reachable leg shell and clamp per-frame IK corrections.
- IK-off output must remain identical to the accepted terrain runtime.
- Keep IK false by default until every unit and deterministic acceptance test passes.
- Use the runtime's fixed 25 Hz timestep for contact springs and every IK acceptance run.

---

## File Map

- g1_ik.h: leg configuration, target bounds, terrain contact update, and G1 IK application.
- controller.cpp: IK orchestration, terrain target sampling, toggles, and diagnostics.
- motion_match_log.h: extra IK metrics that do not affect matching.
- tests/cpp/test_g1_ik.cpp: configuration, target, reachability, and reversibility tests.
- resources/check_g1_runtime_log.py: drift, penetration, bound, and matching-invariance checks.
- tests/python/test_runtime_log.py: synthetic IK metric regressions.

### Task 1: Explicit G1 leg configuration and reachable targets

**Files:**
- Create: g1_ik.h
- Create: tests/cpp/test_g1_ik.cpp

**Interfaces:**
- Produces: G1LegConfig g1_left_leg(), g1_right_leg().
- Produces: vec3 clamp_leg_target(hip, knee, ankle, target, buffer).
- Produces: float quat_correction_angle(before, after).

- [ ] **Step 1: Write configuration and reachability tests**

~~~cpp
// tests/cpp/test_g1_ik.cpp
#include "g1_ik.h"
#include <assert.h>
#include <math.h>

int main()
{
    G1LegConfig left = g1_left_leg();
    G1LegConfig right = g1_right_leg();
    assert(left.hip == G1_LeftHipYaw);
    assert(left.knee == G1_LeftKnee);
    assert(left.ankle == G1_LeftAnkle);
    assert(left.contact == G1_LeftToe);
    assert(left.knee_pole_local.z == -1.0f);
    assert(right.hip == G1_RightHipYaw);
    assert(right.contact == G1_RightToe);

    vec3 hip(0, 1, 0), knee(0, .55f, 0), ankle(0, .1f, 0);
    vec3 target = clamp_leg_target(
        hip, knee, ankle, vec3(3, 0, 0), 0.015f);
    float max_reach = length(knee-hip) + length(ankle-knee) - 0.015f;
    assert(fabsf(length(target-hip) - max_reach) < 1e-5f);

    quat a, b = quat_from_angle_axis(0.1f, vec3(0,1,0));
    assert(fabsf(quat_correction_angle(a,b)-0.1f) < 1e-4f);
}
~~~

- [ ] **Step 2: Compile and verify the missing-header failure**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
~~~

Expected: compilation fails because g1_ik.h is missing.

- [ ] **Step 3: Implement configuration and target clamping**

~~~cpp
// g1_ik.h
#pragma once
#include "g1_skeleton.h"
#include "database.h"
#include "vec.h"
#include "quat.h"

struct G1LegConfig {
    int hip, knee, ankle, contact;
    vec3 knee_pole_local;
    float foot_height;
    float max_correction_radians;
};

static inline G1LegConfig g1_left_leg()
{
    return {G1_LeftHipYaw, G1_LeftKnee, G1_LeftAnkle, G1_LeftToe,
            vec3(0,0,-1), 0.02f, 0.35f};
}

static inline G1LegConfig g1_right_leg()
{
    return {G1_RightHipYaw, G1_RightKnee, G1_RightAnkle, G1_RightToe,
            vec3(0,0,-1), 0.02f, 0.35f};
}

static inline vec3 clamp_leg_target(
    vec3 hip, vec3 knee, vec3 ankle, vec3 target, float buffer)
{
    float a = length(knee-hip), b = length(ankle-knee);
    float hi = maxf(a+b-buffer, 1e-4f);
    float lo = maxf(fabsf(a-b)+buffer, 0.0f);
    vec3 d = target-hip;
    float len = length(d);
    vec3 dir = len > 1e-6f ? d/len : normalize(ankle-hip);
    return hip + dir * clampf(len, lo, hi);
}

static inline float quat_correction_angle(quat before, quat after)
{
    return quat_angle_between(before, after);
}

static inline quat clamp_local_correction(
    quat before, quat after, float max_angle)
{
    float angle = quat_correction_angle(before, after);
    return angle <= max_angle ? after :
        quat_slerp_shortest_approx(before, after, max_angle / angle);
}
~~~

The G1 XML knee axis is local +Y in Z-up coordinates. The pipeline's basis
mapping (x,y,z) -> (x,z,-y) maps that pole to local -Z in the Holden skeleton,
which is why both configurations use vec3(0,0,-1).

- [ ] **Step 4: Compile and run the configuration test**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
/tmp/test_g1_ik
~~~

Expected: exit 0.

- [ ] **Step 5: Commit the G1 leg contract**

~~~bash
git add g1_ik.h tests/cpp/test_g1_ik.cpp
git commit -m "test: define bounded G1 leg IK"
~~~

### Task 2: Terrain-aware contact locking

**Files:**
- Create: contact.h
- Modify: controller.cpp:846-946 contact_update
- Modify: g1_ik.h
- Modify: tests/cpp/test_g1_ik.cpp

**Interfaces:**
- contact_update gains input_ground_height between input_contact_state and unlock_radius.
- Produces: contact floor = input_ground_height + foot_height.
- Existing flat callers pass 0.0f and preserve Holden behavior.

- [ ] **Step 1: Add a terrain contact test**

Append to tests/cpp/test_g1_ik.cpp:

~~~cpp
    bool state=false, lock=false;
    vec3 pos(0,0.31f,0), vel, point, target, offp, offv;
    contact_reset(state, lock, pos, vel, point, target, offp, offv,
                  pos, vec3(), false);
    contact_update(state, lock, pos, vel, point, target, offp, offv,
                   vec3(0,0.31f,0), true, 0.29f,
                   0.2f, 0.02f, 0.1f, 1.0f/25.0f);
    assert(lock);
    assert(fabsf(point.y - 0.31f) < 1e-6f);
~~~

Before the appended test code, add #include "contact.h". Move contact_reset and
contact_update verbatim from controller.cpp into contact.h, include vec.h and
spring.h there, and include contact.h from controller.cpp. Remove the two original
definitions from controller.cpp so there is one implementation.

- [ ] **Step 2: Compile and verify the old signature fails**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
~~~

Expected: compilation fails because contact_update lacks input_ground_height.

- [ ] **Step 3: Parameterize the ground height**

Change the contact_update signature:

~~~cpp
const bool input_contact_state,
const float input_ground_height,
const float unlock_radius,
const float foot_height,
~~~

Change the lock point and final safety clamp:

~~~cpp
const float contact_floor = input_ground_height + foot_height;
contact_point = contact_position;
contact_point.y = contact_floor;
~~~

In controller.cpp replace the later flat clamp with the same sampled floor:

~~~cpp
vec3 contact_position_clamp = contact_positions(i);
contact_position_clamp.y = maxf(
    contact_position_clamp.y, ground_height + legs[i].foot_height);
~~~

Every flat caller passes 0.0f. The G1 controller caller computes:

~~~cpp
float ground_height = heightfield_sample(
    runtime_terrain,
    global_bone_positions(toe_bone).x,
    global_bone_positions(toe_bone).z);
~~~

- [ ] **Step 4: Run G1 IK and existing spring tests**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
/tmp/test_g1_ik
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
/tmp/test_terrain_runtime
~~~

Expected: both exit 0.

- [ ] **Step 5: Commit terrain-aware contacts**

~~~bash
git add contact.h g1_ik.h controller.cpp tests/cpp/test_g1_ik.cpp
git commit -m "fix: lock G1 contacts to sampled terrain"
~~~

### Task 3: Bounded G1 two-bone solve without LAFAN toe assumptions

**Files:**
- Create: ik.h
- Modify: g1_ik.h
- Modify: controller.cpp:2130-2262
- Modify: tests/cpp/test_g1_ik.cpp

**Interfaces:**
- Produces: G1IkResult apply_g1_leg_ik(...).
- Consumes named G1LegConfig, local pose arrays, parent array, and a contact target.
- Produces adjusted hip and knee local rotations only.
- Does not run Holden's +X toe-end ik_look_at block.

- [ ] **Step 1: Add solver-bound and untouched-pose tests**

Append:

~~~cpp
    array1d<vec3> lp(G1_BoneCount); lp.zero();
    array1d<quat> lq(G1_BoneCount);
    for (int i=0;i<G1_BoneCount;i++) lq(i)=quat();
    array1d<int> parents(G1_BoneCount);
    const int expected[G1_BoneCount] = {
        -1,0,1,2,3,4,5,6,1,8,9,10,11,12,1,14,15,16,17,18,19,20,21,22,
        16,24,25,26,27,28,29
    };
    for (int i=0;i<G1_BoneCount;i++) parents(i)=expected[i];
    lp(G1_LeftKnee)=vec3(0,-0.4f,0);
    lp(G1_LeftAnkle)=vec3(0,-0.4f,0);
    array1d<quat> out=lq;
    G1IkResult result = apply_g1_leg_ik(
        out, lp, lq, parents, g1_left_leg(), vec3(.2f,.2f,0), 0.015f);
    assert(result.applied);
    assert(result.max_correction <= g1_left_leg().max_correction_radians+1e-5f);
    for (int i=0;i<G1_BoneCount;i++)
        if (i != G1_LeftHipYaw && i != G1_LeftKnee)
            assert(quat_angle_between(out(i), lq(i)) < 1e-6f);
~~~

- [ ] **Step 2: Compile and verify apply_g1_leg_ik is missing**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
~~~

Expected: compilation fails because G1IkResult and apply_g1_leg_ik are undefined.

- [ ] **Step 3: Implement the bounded adapter**

Add:

~~~cpp
struct G1IkResult {
    bool applied;
    float max_correction;
    vec3 clamped_target;
};

static inline G1IkResult apply_g1_leg_ik(
    slice1d<quat> adjusted,
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> parents,
    const G1LegConfig cfg,
    vec3 target,
    float reach_buffer)
{
    array1d<vec3> gp(local_positions.size);
    array1d<quat> gq(local_positions.size);
    forward_kinematics_full(gp, gq, local_positions, local_rotations, parents);
    vec3 clamped = clamp_leg_target(
        gp(cfg.hip), gp(cfg.knee), gp(cfg.ankle), target, reach_buffer);
    quat hip_before = adjusted(cfg.hip), knee_before = adjusted(cfg.knee);
    ik_two_bone(
        adjusted(cfg.hip), adjusted(cfg.knee),
        gp(cfg.hip), gp(cfg.knee), gp(cfg.ankle), clamped,
        quat_mul_vec3(gq(cfg.knee), cfg.knee_pole_local),
        gq(cfg.hip), gq(cfg.knee), gq(parents(cfg.hip)), reach_buffer);
    adjusted(cfg.hip) = clamp_local_correction(
        hip_before, adjusted(cfg.hip), cfg.max_correction_radians);
    adjusted(cfg.knee) = clamp_local_correction(
        knee_before, adjusted(cfg.knee), cfg.max_correction_radians);
    float max_corr = maxf(
        quat_correction_angle(hip_before, adjusted(cfg.hip)),
        quat_correction_angle(knee_before, adjusted(cfg.knee)));
    return {true, max_corr, clamped};
}
~~~

forward_kinematics_full already lives in database.h. Move ik_look_at and
ik_two_bone verbatim from controller.cpp lines 951-1040 into a new ik.h that
includes vec.h and quat.h. Include ik.h from controller.cpp and g1_ik.h, then
remove the two original controller.cpp definitions. This is a mechanical move;
the function names, parameter order, equations, and defaults do not change.

- [ ] **Step 4: Replace the parent-walk IK loop**

In controller.cpp iterate over:

~~~cpp
const G1LegConfig legs[2] = {g1_left_leg(), g1_right_leg()};
for (int i=0;i<2;i++) {
    const G1LegConfig cfg = legs[i];
    // Contact update uses cfg.contact and sampled terrain height.
    // The target places the ankle by preserving the current ankle-contact offset.
    vec3 ankle_target = contact_position_clamp +
        (global_bone_positions(cfg.ankle) - global_bone_positions(cfg.contact));
    G1IkResult result = apply_g1_leg_ik(
        adjusted_bone_rotations,
        adjusted_bone_positions,
        bone_rotations,
        db.bone_parents,
        cfg, ankle_target, ik_max_length_buffer);
    ik_max_correction_frame = maxf(
        ik_max_correction_frame, result.max_correction);
}
~~~

Delete or bypass both LAFAN-oriented ik_look_at blocks for heel-to-toe and +X
toe-end correction. The G1 debug skeleton has a contact point but no validated
toe-end geometry, so no replacement toe-end solve belongs in phase one.

- [ ] **Step 5: Run all C++ tests and rebuild**

Run:

~~~bash
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
g++ -std=c++17 -I. tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database
/tmp/test_g1_ik
/tmp/test_terrain_runtime
/tmp/test_terrain_database
g++ -std=c++17 -O3 -ffast-math -march=native -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP \
  -I. -I /home/ubuntu/apps/raylib/src -I /home/ubuntu/apps/raygui/src \
  controller.cpp -o controller_g1_terrain \
  -L /home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
~~~

Expected: all tests and build exit 0.

- [ ] **Step 6: Commit the bounded solver**

~~~bash
git add ik.h g1_ik.h controller.cpp tests/cpp/test_g1_ik.cpp
git commit -m "fix: adapt foot IK to the G1 leg chain"
~~~

### Task 4: IK metrics, reversibility, and terrain acceptance

**Files:**
- Modify: motion_match_log.h
- Modify: controller.cpp
- Modify: resources/check_g1_runtime_log.py
- Modify: tests/python/test_runtime_log.py

**Interfaces:**
- MM_IK enables IK only in terrain test mode.
- CSV adds left_locked, right_locked, left_foot_drift, right_foot_drift,
  max_penetration, and max_ik_correction.
- Checker compares IK-on matching columns with the accepted IK-off baseline.

- [ ] **Step 1: Write metric-checker failures**

Add tests:

~~~python
    def test_rejects_ik_penetration(self):
        rows = [{"database_frame":"1","transitioned":"0","selected_cost":"1",
                 "incumbent_cost":"1","max_penetration":"0.02",
                 "max_ik_correction":"0.1"}]
        with self.assertRaisesRegex(ValueError, "penetration"):
            check_ik_rows(rows, max_penetration=0.005, max_correction=0.35)

    def test_rejects_matching_change_between_ik_runs(self):
        off = [{"frame":"0","database_frame":"10","selected_cost":"1"}]
        on = [{"frame":"0","database_frame":"11","selected_cost":"1"}]
        with self.assertRaisesRegex(ValueError, "matching changed"):
            compare_ik_matching(off, on)

    def test_accepts_reduced_locked_foot_drift(self):
        off = [{"left_foot_drift":"0.02","right_foot_drift":"0",
                "left_locked":"1","right_locked":"0"}]
        on = [{"left_foot_drift":"0.005","right_foot_drift":"0",
               "left_locked":"1","right_locked":"0"}]
        self.assertEqual(compare_ik_drift(off, on), (0.02, 0.005))
~~~

Add check_ik_rows, compare_ik_matching, and compare_ik_drift to the existing
import from resources.check_g1_runtime_log.

- [ ] **Step 2: Run and verify missing checker functions**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest \
  tests.python.test_runtime_log -v
~~~

Expected: FAIL because check_ik_rows and compare_ik_matching are missing.

- [ ] **Step 3: Implement metric validation**

~~~python
def check_ik_rows(rows, max_penetration=0.005, max_correction=0.35):
    for i, row in enumerate(rows):
        if float(row["max_penetration"]) > max_penetration:
            raise ValueError(f"row {i}: penetration exceeds bound")
        if float(row["max_ik_correction"]) > max_correction + 1e-6:
            raise ValueError(f"row {i}: IK correction exceeds bound")


def compare_ik_matching(off_rows, on_rows):
    if len(off_rows) != len(on_rows):
        raise ValueError("IK runs have different lengths")
    for i, (off, on) in enumerate(zip(off_rows, on_rows)):
        if off["database_frame"] != on["database_frame"] or \
           off["selected_cost"] != on["selected_cost"]:
            raise ValueError(f"row {i}: matching changed with IK")


def compare_ik_drift(off_rows, on_rows):
    pairs = []
    for off, on in zip(off_rows, on_rows):
        if int(on["left_locked"]):
            pairs.append((float(off["left_foot_drift"]),
                          float(on["left_foot_drift"])))
        if int(on["right_locked"]):
            pairs.append((float(off["right_foot_drift"]),
                          float(on["right_foot_drift"])))
    if not pairs:
        raise ValueError("IK run contained no locked contacts")
    off_mean = sum(x for x, _ in pairs) / len(pairs)
    on_mean = sum(y for _, y in pairs) / len(pairs)
    if on_mean >= off_mean:
        raise ValueError(f"IK did not reduce foot drift: {on_mean} >= {off_mean}")
    return off_mean, on_mean
~~~

Extend the checker CLI parser and dispatch:

~~~python
parser.add_argument("--ik-baseline")
# after ordinary check_rows and optional terrain comparison:
if args.ik_baseline:
    baseline = read_rows(args.ik_baseline)
    check_ik_rows(rows)
    compare_ik_matching(baseline, rows)
    off_drift, on_drift = compare_ik_drift(baseline, rows)
    print(f"VALID ik off_drift={off_drift:.6g} on_drift={on_drift:.6g}")
~~~

- [ ] **Step 4: Log drift, penetration, and correction after IK**

For each contact, retain the previous adjusted contact point. Drift is horizontal
distance while the contact is locked:

~~~cpp
vec3 delta = adjusted_contact_position - previous_adjusted_contact_position;
float drift = contact_locks(i) ? sqrtf(delta.x*delta.x + delta.z*delta.z) : 0.0f;
float ground = heightfield_sample(
    runtime_terrain, adjusted_contact_position.x, adjusted_contact_position.z);
float penetration = maxf(
    ground + legs[i].foot_height - adjusted_contact_position.y, 0.0f);
~~~

Append both drifts, maximum penetration, and ik_max_correction_frame to every CSV
row, together with the two contact-lock booleans. Compute metrics after adjusted
forward kinematics and never feed them back to the matcher. Extend
motion_match_log::write with these values after source_range:

~~~cpp
fprintf(file, ",%d,%d,%.9g,%.9g,%.9g,%.9g",
        (int)left_locked, (int)right_locked,
        left_foot_drift, right_foot_drift,
        max_penetration, max_ik_correction);
~~~

Append this exact header suffix:

~~~cpp
",left_locked,right_locked,left_foot_drift,right_foot_drift,"
 "max_penetration,max_ik_correction\n"
~~~

Move the newline in the base writer to this suffix so each frame remains one CSV
record.

- [ ] **Step 5: Run matched IK-off and IK-on terrain tests**

Run:

~~~bash
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_IK=0 \
  MM_TEST_FRAMES=375 MM_LOG=/tmp/g1_terrain_ik_off.csv \
  ./controller_g1_terrain
DISPLAY=:1 G1_TERRAIN_DIR=resources/g1_terrain \
  MM_TEST_MODE=terrain MM_TERRAIN_WEIGHT=4 MM_IK=1 \
  MM_TEST_FRAMES=375 MM_LOG=/tmp/g1_terrain_ik_on.csv \
  ./controller_g1_terrain
~~~

Run the checker with:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python \
  resources/check_g1_runtime_log.py /tmp/g1_terrain_ik_on.csv \
  --ik-baseline /tmp/g1_terrain_ik_off.csv
~~~

Expected:

- matching frame and selected cost columns are identical;
- maximum penetration is no greater than 5 mm;
- maximum correction is no greater than 0.35 radians;
- mean planted horizontal foot drift is lower in IK-on;
- both runs retain the already-passing terrain selection invariants.

- [ ] **Step 6: Make IK available but disabled by default**

Keep bool ik_enabled=false at startup. MM_IK=1 may enable it in deterministic
terrain mode. The Raylib checkbox remains the only live-mode enable control and
resets contacts on its false-to-true edge.

- [ ] **Step 7: Run the complete verification suite**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/diffsim/bin/python -m unittest discover \
  -s tests/python -p 'test_*.py' -v
g++ -std=c++17 -I. tests/cpp/test_terrain_runtime.cpp -o /tmp/test_terrain_runtime
g++ -std=c++17 -I. tests/cpp/test_g1_skeleton.cpp -o /tmp/test_g1_skeleton
g++ -std=c++17 -I. tests/cpp/test_terrain_database.cpp -o /tmp/test_terrain_database
g++ -std=c++17 -I. tests/cpp/test_g1_ik.cpp -o /tmp/test_g1_ik
/tmp/test_terrain_runtime
/tmp/test_g1_skeleton
/tmp/test_terrain_database
/tmp/test_g1_ik
~~~

Expected: every test exits 0.

- [ ] **Step 8: Commit IK acceptance**

~~~bash
git add motion_match_log.h controller.cpp \
  resources/check_g1_runtime_log.py tests/python/test_runtime_log.py
git commit -m "test: validate reversible G1 terrain IK"
~~~
