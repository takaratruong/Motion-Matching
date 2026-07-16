# G1 Root-Reach Boundary Frontier Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the root-reach planner's false-negative 32-candidate frontier with one globally shared 64-candidate frontier which selects the exact live row-6 and row-7 plans while preserving every existing numerical, ownership, and rollback contract.

**Architecture:** Keep the existing analytic interval construction, nearest-first multi-cursor scheduler, one-ULP inward advancement, and strict production revalidation unchanged. Increase only the private production revalidation ceiling and the test-only audit storage, bind them with a compile-time assertion, and add bit-exact standalone fixtures captured from live rows 6 and 7 so strict and optimized callers certify the real 47- and 39-attempt traces without loading a controller, database, or terrain pack.

**Tech Stack:** C++17; IEEE-754 binary32/binary64; Daniel Holden array/vector/quaternion types; strict-FP `g1_ik_root_reach.cpp`; GDB batch capture; standalone C++ tests; Raylib disposable controller; Python CSV assertions.

## Global Constraints

- Implement the committed design at `docs/superpowers/specs/2026-07-16-g1-root-reach-boundary-frontier-amendment-design.md` from commit `d35780e`.
- The planner-wide production ceiling is exactly 64 revalidations per plan. It is global across all interval cursors, not 64 per cursor.
- Under `G1_IK_ENABLE_TEST_SEAMS`, `G1RootReachPlannerAudit::attempts` contains exactly 64 entries and records at most 64 real production revalidations.
- Preserve the existing baseline analytic proof domain, cursor ordering by absolute delta then `(lower, upper, original interval index)`, and one-ULP inward `std::nextafter` advancement of only the rejected cursor.
- Preserve current post-loop semantics: after rejection 64 the selected cursor may materialize and analytic-bounds-check one next head, but that head receives no strict FK, physical-target reconstruction, reach projection, or audit record.
- A 65th or later production revalidation remains unavailable and publishes the existing canonical active/non-common/unapplied positive-zero plan.
- Keep the exact binary32 root-Y cap `[-0.05f, +0.05f]` and mutate only reversible candidate `G1_Simulation.y` through `g1_apply_root_reach_plan_y`.
- Every attempted candidate continues to copy 31 local positions, run checked strict 31-bone FK, rederive physical targets from adjusted globals, and require reachable bit-exact projection for every recorded foot.
- Preserve all plan validation, alias rejection, malformed-arithmetic failure, local-audit publication, no-seam privacy, strict/fast parity, and outer atomic rollback behavior.
- Compile `g1_ik_root_reach.cpp` and `g1_clearance.cpp` with `-fno-fast-math -ffp-contract=off -frounding-math`. Callers may use `-ffast-math`; final links may not. Do not use LTO.
- Add no heap allocation, runtime option, CSV column, controller state, terrain query, database search, swing iteration, or visualizer change.
- Runtime remains exact 25 Hz with binary32 `0.04f`.
- Row 6 must publish root delta bits `0xbbd51d41` after 47 attempts, then retain the independent `no-swing-candidate` frame rejection.
- Row 7 must publish root delta bits `0xbbc6abc1` after 39 attempts and eliminate that frame's former root-planner `target-unreachable` rejection.
- Certification for this narrow amendment stops at the exact 8-frame boundary. Do not run or claim a 32- or 800-frame low-curb certification while row 6 still rejects; defer those horizons to the combined Gate E/Gate L work after matcher/swing recovery first produces a zero-rejection 32-frame run.
- Do not claim this amendment fixes the separate row-6 swing/matcher failure, later route failures, lateral or diagonal matching, half-support edges, slope-foot alignment, terrain density, or G1 mesh rendering.
- Preserve unrelated user and agent work. In particular, do not stage or edit `docs/superpowers/specs/2026-07-16-g1-bounded-candidate-certification-design.md` as part of this amendment.

## File Structure

- Create `tests/cpp/g1_root_reach_live_fixture_bits.h`: generated, bit-exact row-6 and row-7 local positions, local rotations, parents, contact flags, and target fields; no production types or behavior.
- Modify `tests/cpp/test_g1_ik.cpp`: reconstruct the live fixtures, freeze exact 47/39 traces, extend all exhaustion traces to 64, freeze diagnostic out-of-budget probes, and include both live audits in strict/fast parity output.
- Modify `g1_ik.h`: enlarge only the test-seam audit attempt array from 32 to 64.
- Modify `g1_ik_root_reach.cpp`: enlarge only the private production ceiling from 32 to 64 and add the seam-only ceiling/capacity compile-time assertion.
- Do not modify controller, runtime, transaction, state, terrain, matcher, log checker, visualizer, or motion-database files.

---

### Task 1: Implement the 64-Candidate Boundary Frontier

**Files:**
- Create: `tests/cpp/g1_root_reach_live_fixture_bits.h`
- Modify: `tests/cpp/test_g1_ik.cpp:1-10,1923-2116,2771-2854,2856-3220,12255-12489,12519-12536`
- Modify: `g1_ik.h:508-549`
- Modify: `g1_ik_root_reach.cpp:302-303,766-810,1040-1117`
- Reference only: `docs/superpowers/specs/2026-07-16-g1-root-reach-boundary-frontier-amendment-design.md`

**Interfaces:**
- Consumes: existing `G1RootReachPlan`, `G1FootTarget`, `g1_plan_recorded_contact_root_reach`, `g1_plan_recorded_contact_root_reach_audited`, `g1_apply_root_reach_plan_y`, checked FK, `g1_physical_sole_position_target`, `ik_project_target`, and named physical-sole IK.
- Produces: the unchanged production planner signatures plus this test-only layout:

```cpp
#if defined(G1_IK_ENABLE_TEST_SEAMS)
struct G1RootReachPlannerAudit
{
    G1RootReachAuditCursor cursors[4];
    G1RootReachAuditAttempt attempts[64];
    uint32_t cursor_count;
    uint32_t attempt_count;
};
#endif
```

- Produces: one private `G1RootReachMaximumBoundaryCandidateTests = 64U`, compile-time-equal to the audit attempt capacity when test seams are enabled.
- Produces no new production function, runtime setting, state field, log field, or test fault-injection seam.

- [ ] **Step 1: Record the execution base and build the bit-exact capture controller before any source edit**

Run from the worktree root:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment
mkdir -p "$out/capture"
git rev-parse HEAD > "$out/execution-base.txt"
git diff --exit-code d35780e -- \
  controller.cpp g1_ik.h g1_ik_root_reach.cpp g1_ik_runtime.h \
  g1_frame_transaction.h g1_controller_state.h g1_clearance.cpp \
  tests/cpp/test_g1_ik.cpp

g++ -std=c++17 -O3 -g -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src \
  -c controller.cpp -o "$out/capture/controller.o"
g++ -std=c++17 -O0 -g -fno-fast-math -ffp-contract=off \
  -frounding-math -I. -c g1_ik_root_reach.cpp \
  -o "$out/capture/root-reach.o"
g++ -std=c++17 -O3 -g -fno-fast-math -ffp-contract=off \
  -frounding-math -I. -c g1_clearance.cpp \
  -o "$out/capture/clearance.o"
g++ "$out/capture/controller.o" "$out/capture/root-reach.o" \
  "$out/capture/clearance.o" -o "$out/capture/controller" \
  -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
```

Expected: `git diff --exit-code` and all four build commands exit `0`. The caller contains symbols and optimized production caller arithmetic; the root-reach object remains unoptimized and strict so GDB can read the exact public planner arguments.

- [ ] **Step 2: Capture planner calls 7 and 8 and verify every raw owner hash**

Use `apply_patch` to create `/tmp/g1-root-frontier-amendment/capture.gdb` with exactly:

```gdb
set debuginfod enabled off
set env DISPLAY=:1
set env MM_IK=1
set pagination off
set breakpoint pending on
break g1_plan_recorded_contact_root_reach
set $hit=0
commands
silent
set $hit=$hit+1
if $hit == 7
  printf "CAPTURE row6 pos=%zu rot=%zu parents=%zu contacts=%zu target=%zu\n", sizeof(vec3), sizeof(quat), sizeof(int), sizeof(bool), sizeof(G1FootTarget)
  dump binary memory /tmp/g1-root-frontier-amendment/row6-positions.bin baseline_positions.data baseline_positions.data + 31
  dump binary memory /tmp/g1-root-frontier-amendment/row6-rotations.bin baseline_rotations.data baseline_rotations.data + 31
  dump binary memory /tmp/g1-root-frontier-amendment/row6-parents.bin parents.data parents.data + 31
  dump binary memory /tmp/g1-root-frontier-amendment/row6-contacts.bin recorded_contacts.data recorded_contacts.data + 2
  dump binary memory /tmp/g1-root-frontier-amendment/row6-left-target.bin &left_target ((char *)&left_target) + sizeof(G1FootTarget)
  dump binary memory /tmp/g1-root-frontier-amendment/row6-right-target.bin &right_target ((char *)&right_target) + sizeof(G1FootTarget)
end
if $hit == 8
  printf "CAPTURE row7 pos=%zu rot=%zu parents=%zu contacts=%zu target=%zu\n", sizeof(vec3), sizeof(quat), sizeof(int), sizeof(bool), sizeof(G1FootTarget)
  dump binary memory /tmp/g1-root-frontier-amendment/row7-positions.bin baseline_positions.data baseline_positions.data + 31
  dump binary memory /tmp/g1-root-frontier-amendment/row7-rotations.bin baseline_rotations.data baseline_rotations.data + 31
  dump binary memory /tmp/g1-root-frontier-amendment/row7-parents.bin parents.data parents.data + 31
  dump binary memory /tmp/g1-root-frontier-amendment/row7-contacts.bin recorded_contacts.data recorded_contacts.data + 2
  dump binary memory /tmp/g1-root-frontier-amendment/row7-left-target.bin &left_target ((char *)&left_target) + sizeof(G1FootTarget)
  dump binary memory /tmp/g1-root-frontier-amendment/row7-right-target.bin &right_target ((char *)&right_target) + sizeof(G1FootTarget)
end
continue
end
run --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 --terrain-weight 4 --test-mode route --test-route curb-forward --test-frames 8 --test-heading forward --terrain-scene grail-curb-low --log /tmp/g1-root-frontier-amendment/capture.csv
```

Then run:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment
rm -f "$out"/row{6,7}-*.bin
gdb -q -batch -x "$out/capture.gdb" "$out/capture/controller" \
  > "$out/capture.stdout" 2> "$out/capture.stderr"
rg 'CAPTURE row6 pos=12 rot=16 parents=4 contacts=1 target=56' \
  "$out/capture.stdout"
rg 'CAPTURE row7 pos=12 rot=16 parents=4 contacts=1 target=56' \
  "$out/capture.stdout"
sha256sum -c <<'EOF'
b413f47d13ee2fe6c845b2ee141af81de858df4ec549a58b7970bb96645bc8d2  /tmp/g1-root-frontier-amendment/row6-contacts.bin
71df8532713e2b4bcdc62b79fb21b8532f5fa94ae19276018d39f8b1ba310d77  /tmp/g1-root-frontier-amendment/row6-left-target.bin
b3aadd9023c97f73cdd1c25bc50066cb23411aaece05105f1531442f8b0a5af2  /tmp/g1-root-frontier-amendment/row6-parents.bin
1cd9817d5c2b947e56c248c6e73636c7970b2bf8f6b5c1b9901b1746f4347d50  /tmp/g1-root-frontier-amendment/row6-positions.bin
10fc43210c5e5b82d5a295c8f909669ab463a83567013225929033798e834fc9  /tmp/g1-root-frontier-amendment/row6-right-target.bin
e75a6ac1d1bfaf3517e207a3a8311d2685824f309241bb60d97af95f4c39e15f  /tmp/g1-root-frontier-amendment/row6-rotations.bin
9dcf97a184f32623d11a73124ceb99a5709b083721e878a16d78f596718ba7b2  /tmp/g1-root-frontier-amendment/row7-contacts.bin
89f6ddd84f2e0aad126e37883e2bd45a048bc2ecc16c1c6ef71c791f6a3784fc  /tmp/g1-root-frontier-amendment/row7-left-target.bin
b3aadd9023c97f73cdd1c25bc50066cb23411aaece05105f1531442f8b0a5af2  /tmp/g1-root-frontier-amendment/row7-parents.bin
16fa4cf05fbfda70c7ffe28ce033feb04cb4b823b7027a51b4103410a0216f4e  /tmp/g1-root-frontier-amendment/row7-positions.bin
debf43fd0a140df096f352f040c09d598b2046a6b436ea1a69e70147e4195a43  /tmp/g1-root-frontier-amendment/row7-right-target.bin
19dc6188b344ee0c250be7e9100394aa725356c7ddfe29993dbe2c3e1aa1d5aa  /tmp/g1-root-frontier-amendment/row7-rotations.bin
EOF
```

Expected: GDB exits `0`, prints exactly the two layout lines, and all twelve files report `OK`. Row 6 contacts decode to `{0,1}` and row 7 contacts decode to `{1,1}`. These hashes are the fixture-capture oracle; stop if any differs.

- [ ] **Step 3: Generate and add the standalone fixture header**

Use `apply_patch` to create `/tmp/g1-root-frontier-amendment/convert.py` with this complete converter:

```python
from pathlib import Path
import struct

ROOT = Path("/tmp/g1-root-frontier-amendment")


def read(name):
    return (ROOT / name).read_bytes()


def words(name, signed=False):
    data = read(name)
    code = "i" if signed else "I"
    return struct.unpack("<" + code * (len(data) // 4), data)


def emit_matrix(name, values, columns):
    print(
        f"static constexpr std::uint32_t {name}"
        f"[{len(values) // columns}][{columns}] = {{")
    for offset in range(0, len(values), columns):
        row = ", ".join(
            f"UINT32_C(0x{value:08x})"
            for value in values[offset:offset + columns])
        print(f"    {{{row}}},")
    print("};\n")


def emit_row(row):
    prefix = f"G1TestRootReachRow{row}"
    emit_matrix(
        prefix + "PositionBits",
        words(f"row{row}-positions.bin"), 3)
    emit_matrix(
        prefix + "RotationBits",
        words(f"row{row}-rotations.bin"), 4)
    parents = words(f"row{row}-parents.bin", signed=True)
    print(
        f"static constexpr std::int32_t {prefix}Parents"
        f"[{len(parents)}] = {{")
    print("    " + ", ".join(str(value) for value in parents))
    print("};\n")
    contacts = read(f"row{row}-contacts.bin")
    print(
        f"static constexpr std::uint8_t {prefix}Contacts[2] = "
        f"{{{contacts[0]}, {contacts[1]}}};\n")
    print(
        f"static constexpr G1TestRootReachTargetBits "
        f"{prefix}Targets[2] = {{")
    for side in ("left", "right"):
        data = read(f"row{row}-{side}-target.bin")
        flags = ", ".join(str(value) for value in data[:4])
        scalars = struct.unpack("<13I", data[4:])
        scalar_text = ", ".join(
            f"UINT32_C(0x{value:08x})" for value in scalars)
        print(f"    {{{{{flags}}}, {{{scalar_text}}}}},")
    print("};\n")
    print(
        f"static constexpr G1TestRootReachLiveFixtureBits "
        f"{prefix} = {{")
    print(f"    {prefix}PositionBits, {prefix}RotationBits,")
    print(f"    {prefix}Parents, {prefix}Contacts, {prefix}Targets")
    print("};\n")


print("#ifndef G1_ROOT_REACH_LIVE_FIXTURE_BITS_H")
print("#define G1_ROOT_REACH_LIVE_FIXTURE_BITS_H\n")
print("#include <cstdint>\n")
print("struct G1TestRootReachTargetBits")
print("{")
print("    std::uint8_t flags[4];")
print("    std::uint32_t scalars[13];")
print("};\n")
print("struct G1TestRootReachLiveFixtureBits")
print("{")
print("    const std::uint32_t (*position_bits)[3];")
print("    const std::uint32_t (*rotation_bits)[4];")
print("    const std::int32_t* parents;")
print("    const std::uint8_t* contacts;")
print("    const G1TestRootReachTargetBits* targets;")
print("};\n")
emit_row(6)
emit_row(7)
print("#endif")
```

Generate, compile-check, and hash the header:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment
python3 "$out/convert.py" > "$out/g1_root_reach_live_fixture_bits.h"
test "$(wc -l < "$out/g1_root_reach_live_fixture_bits.h")" -eq 189
test "$(wc -c < "$out/g1_root_reach_live_fixture_bits.h")" -eq 13436
printf '%s  %s\n' \
  dc531c87d6ce77c7cf85e425dd4a80a56d5133954d9cfda3a7a42fc5bde447d8 \
  "$out/g1_root_reach_live_fixture_bits.h" | sha256sum -c -
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -x c++ \
  -c "$out/g1_root_reach_live_fixture_bits.h" \
  -o "$out/g1_root_reach_live_fixture_bits.o"
```

Expected: the line count, byte count, checksum, and standalone header compile all pass. Use `apply_patch` to add that exact generated content as `tests/cpp/g1_root_reach_live_fixture_bits.h`, then rerun the same SHA-256 check against the repository file. Do not commit the raw ABI-dependent `.bin` captures or either `/tmp` script.

- [ ] **Step 4: Write RED fixture materialization and candidate-authentication helpers**

In `tests/cpp/test_g1_ik.cpp`, include the generated header only for seam builds:

```cpp
#include "g1_ik_runtime.h"

#if defined(G1_IK_ENABLE_TEST_SEAMS)
#include "g1_root_reach_live_fixture_bits.h"
#endif
```

After `G1RootReachTestFixture` and `g1_test_root_reach_locked_target`, add these seam-only reconstruction helpers. The 13 target scalar indices are exactly surface point XYZ, surface normal XYZ, desired sole normal XYZ, sole center XYZ, and horizontal drift:

```cpp
#if defined(G1_IK_ENABLE_TEST_SEAMS)

static G1FootTarget g1_test_root_reach_target_from_bits(
    const G1TestRootReachTargetBits& bits)
{
    G1FootTarget target = {};
    target.locked = bits.flags[0] != 0U;
    target.position_active = bits.flags[1] != 0U;
    target.releasing = bits.flags[2] != 0U;
    target.drift_limit_exceeded = bits.flags[3] != 0U;
    target.surface.point = vec3(
        g1_test_float_from_bits(bits.scalars[0]),
        g1_test_float_from_bits(bits.scalars[1]),
        g1_test_float_from_bits(bits.scalars[2]));
    target.surface.normal = vec3(
        g1_test_float_from_bits(bits.scalars[3]),
        g1_test_float_from_bits(bits.scalars[4]),
        g1_test_float_from_bits(bits.scalars[5]));
    target.desired_sole_normal = vec3(
        g1_test_float_from_bits(bits.scalars[6]),
        g1_test_float_from_bits(bits.scalars[7]),
        g1_test_float_from_bits(bits.scalars[8]));
    target.sole_center = vec3(
        g1_test_float_from_bits(bits.scalars[9]),
        g1_test_float_from_bits(bits.scalars[10]),
        g1_test_float_from_bits(bits.scalars[11]));
    target.horizontal_drift_m =
        g1_test_float_from_bits(bits.scalars[12]);
    return target;
}

static void g1_test_make_root_reach_live_fixture(
    G1RootReachTestFixture& fixture,
    const G1TestRootReachLiveFixtureBits& bits)
{
    fixture.positions.resize(G1_BoneCount);
    fixture.rotations.resize(G1_BoneCount);
    fixture.parents.resize(G1_BoneCount);
    fixture.contacts.resize(2);
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        fixture.positions(bone) = vec3(
            g1_test_float_from_bits(bits.position_bits[bone][0]),
            g1_test_float_from_bits(bits.position_bits[bone][1]),
            g1_test_float_from_bits(bits.position_bits[bone][2]));
        fixture.rotations(bone) = quat(
            g1_test_float_from_bits(bits.rotation_bits[bone][0]),
            g1_test_float_from_bits(bits.rotation_bits[bone][1]),
            g1_test_float_from_bits(bits.rotation_bits[bone][2]),
            g1_test_float_from_bits(bits.rotation_bits[bone][3]));
        fixture.parents(bone) = static_cast<int>(bits.parents[bone]);
    }
    for (int foot = 0; foot < 2; ++foot) {
        fixture.contacts(foot) = bits.contacts[foot] != 0U;
        fixture.targets[foot] =
            g1_test_root_reach_target_from_bits(bits.targets[foot]);
    }
}

#endif
```

After `g1_test_root_reach_plan`, add one public-math authentication helper used by both diagnostic refusal probes and live fixtures:

```cpp
static bool g1_test_root_reach_candidate_authenticates(
    const G1RootReachTestFixture& fixture,
    uint32_t delta_bits)
{
    const G1RootReachPlan plan = {
        true, true, true, g1_test_float_from_bits(delta_bits)
    };
    array1d<vec3> adjusted_positions = fixture.positions;
    check(g1_apply_root_reach_plan_y(
              adjusted_positions(G1_Simulation).y,
              fixture.positions(G1_Simulation).y,
              plan),
          "candidate authentication applies root Y");
    array1d<vec3> globals(G1_BoneCount);
    array1d<quat> global_rotations(G1_BoneCount);
    char error[512] = {};
    check(g1_ik_checked_forward_kinematics(
              globals, global_rotations,
              adjusted_positions, fixture.rotations, fixture.parents,
              error, static_cast<int>(sizeof(error))), error);
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        if (!fixture.contacts(foot)) continue;
        G1PhysicalSolePositionTarget physical = {};
        IKTargetProjection projection = {};
        const G1LegConfig& config = configs[foot];
        check(g1_physical_sole_position_target(
                  physical,
                  globals(config.contact),
                  global_rotations(config.contact),
                  globals(config.ankle),
                  config,
                  fixture.targets[foot].sole_center,
                  fixture.targets[foot].desired_sole_normal,
                  error, static_cast<int>(sizeof(error))) &&
                  ik_project_target(
                      projection,
                      globals(config.hip), globals(config.knee),
                      globals(config.ankle), physical.ankle_target,
                      config.reach_buffer_m),
              "candidate authentication materializes production projection");
        if (!projection.reachable ||
            !g1_ik_vec3_bits_equal(
                projection.clamped_target, physical.ankle_target)) {
            return false;
        }
    }
    return true;
}
```

Expected: these helpers compile once the live test is added. They perform no file I/O and reconstruct all values from committed bits.

- [ ] **Step 5: Write the exact row-6 and row-7 RED tests**

Inside `#if defined(G1_IK_ENABLE_TEST_SEAMS)`, add this expectation helper and test before the existing audit-trace test:

```cpp
static void g1_test_expect_root_reach_live_frontier(
    const G1TestRootReachLiveFixtureBits& bits,
    uint32_t initial_bits,
    uint32_t accepted_bits,
    uint32_t expected_attempts,
    bool require_two_contact_convergence)
{
    G1RootReachTestFixture fixture;
    g1_test_make_root_reach_live_fixture(fixture, bits);
    const array1d<vec3> positions_before = fixture.positions;
    const array1d<quat> rotations_before = fixture.rotations;
    const array1d<int> parents_before = fixture.parents;
    const array1d<bool> contacts_before = fixture.contacts;
    const G1FootTarget targets_before[2] = {
        fixture.targets[0], fixture.targets[1]
    };

    char error[512] = {};
    G1RootReachPlan plan = {};
    G1RootReachPlannerAudit audit = {};
    check(g1_plan_recorded_contact_root_reach_audited(
              plan, audit,
              fixture.positions, fixture.rotations, fixture.parents,
              fixture.contacts, fixture.targets[0], fixture.targets[1],
              error, static_cast<int>(sizeof(error))), error);
    check(plan.active && plan.common_interval_found && plan.applied &&
              terrain_float_bits(plan.root_y_delta_m) == accepted_bits,
          "live root frontier publishes the exact plan");
    check(audit.cursor_count == 1U &&
              audit.cursors[0].interval_index == 0U &&
              audit.cursors[0].initial_delta_bits == initial_bits &&
              audit.attempt_count == expected_attempts,
          "live root frontier owns the exact cursor and attempt count");
    for (uint32_t attempt = 0U;
         attempt < expected_attempts;
         ++attempt) {
        check(audit.attempts[attempt].cursor_index == 0U &&
                  audit.attempts[attempt].delta_bits ==
                      initial_bits + attempt &&
                  audit.attempts[attempt].status ==
                      (attempt + 1U == expected_attempts
                           ? G1RootReachAuditAccepted
                           : G1RootReachAuditRejected),
              "live root frontier owns every adjacent production attempt");
    }
    check(g1_test_root_reach_candidate_authenticates(
              fixture, accepted_bits),
          "live root frontier accepted bits authenticate independently");
    check(!g1_test_root_reach_candidate_authenticates(
              fixture, accepted_bits - 1U),
          "live root frontier preceding bits remain rejected");

    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        check(g1_test_vec3_same(
                  fixture.positions(bone), positions_before(bone)) &&
                  g1_test_quat_bits_same(
                      fixture.rotations(bone), rotations_before(bone)) &&
                  fixture.parents(bone) == parents_before(bone),
              "live root planning preserves pose and topology inputs");
    }
    for (int foot = 0; foot < 2; ++foot) {
        check(fixture.contacts(foot) == contacts_before(foot) &&
                  g1_test_foot_target_same(
                      fixture.targets[foot], targets_before[foot]),
              "live root planning preserves contact and target inputs");
    }

    if (!require_two_contact_convergence) return;
    array1d<vec3> adjusted_positions = fixture.positions;
    check(g1_apply_root_reach_plan_y(
              adjusted_positions(G1_Simulation).y,
              fixture.positions(G1_Simulation).y, plan),
          "row-7 root plan applies to the candidate pose");
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    const float residual_limits[2] = {0.0025f, 0.0011f};
    for (int foot = 0; foot < 2; ++foot) {
        array1d<quat> solved = fixture.rotations;
        G1LegSolveResult position = {};
        G1FootOrientationResult orientation = {};
        check(g1_apply_named_physical_sole_ik(
                  solved,
                  adjusted_positions, fixture.rotations, fixture.parents,
                  configs[foot],
                  fixture.targets[foot].sole_center,
                  fixture.targets[foot].desired_sole_normal,
                  position, orientation,
                  error, static_cast<int>(sizeof(error))), error);
        check(position.reachable && !position.correction_limited &&
                  g1_ik_contact_residual_is_converged(
                      position.contact_residual_m) &&
                  position.contact_residual_m < residual_limits[foot],
              "row-7 named contact solve retains the exact convergence rule");
    }
}

static void test_root_reach_live_boundary_frontiers()
{
    g1_test_expect_root_reach_live_frontier(
        G1TestRootReachRow6,
        UINT32_C(0xbbd51d13), UINT32_C(0xbbd51d41),
        47U, false);
    g1_test_expect_root_reach_live_frontier(
        G1TestRootReachRow7,
        UINT32_C(0xbbc6ab9b), UINT32_C(0xbbc6abc1),
        39U, true);
}
```

Call `test_root_reach_live_boundary_frontiers()` immediately before `test_root_reach_planner_audit_trace()` in `main`'s seam-only block.

Expected RED behavior: the current 32-attempt planner returns the canonical unavailable plan for row 6, so the first new check fails with `G1 IK test failed: live root frontier publishes the exact plan`.

- [ ] **Step 6: Extend the existing exhaustion and diagnostic RED tests from 32 to 64**

At the start of the seam-only audit section, add one test oracle and capacity assertion:

```cpp
static constexpr uint32_t G1RootReachTestAttemptLimit = 64U;
static_assert(
    sizeof(((G1RootReachPlannerAudit*)nullptr)->attempts) /
            sizeof(((G1RootReachPlannerAudit*)nullptr)->attempts[0]) ==
        G1RootReachTestAttemptLimit,
    "test audit capacity owns the amended 64-attempt limit");
```

Make these exact mechanical test changes. For each named existing loop, replace
only its old `32U` bound with the line shown; retain its already-complete
per-attempt assertions:

```cpp
// Non-seam projection-refusal direct prefix:
for (uint32_t candidate_index = 0U;
     candidate_index < 64U;
     ++candidate_index) {
    const uint32_t bits = UINT32_C(0x3c8b379d) + candidate_index;
    check(!g1_test_root_reach_candidate_authenticates(
              revalidation_refusal, bits),
          "each amended analytic boundary float is projection-rejected");
}
check(!g1_test_root_reach_candidate_authenticates(
          revalidation_refusal, UINT32_C(0x3c8b37dd)) &&
          g1_test_root_reach_candidate_authenticates(
              revalidation_refusal, UINT32_C(0x3c8b381d)),
      "out-of-budget refusal probes retain direct production results");

// Representational no-op audit:
check(representational_audit.attempt_count ==
          G1RootReachTestAttemptLimit,
      "representational no-op exhausts the amended global limit");
for (uint32_t attempt = 0U;
     attempt < G1RootReachTestAttemptLimit;
     ++attempt) {
    const uint32_t expected_bits = UINT32_C(0xb2a00000) + attempt;
    check(representational_audit.attempts[attempt].cursor_index == 0U &&
              representational_audit.attempts[attempt].delta_bits ==
                  expected_bits &&
              representational_audit.attempts[attempt].status ==
                  G1RootReachAuditRejected,
          "every amended representational no-op is audited rejected");
}

// Symmetric two-cursor audit:
check(exhaustion_audit.attempt_count ==
          G1RootReachTestAttemptLimit,
      "two-cursor refusal exhausts one global 64-attempt budget");
for (uint32_t attempt = 0U;
     attempt < G1RootReachTestAttemptLimit;
     ++attempt) {
    const uint32_t cursor = attempt & 1U;
    const uint32_t magnitude_bits =
        UINT32_C(0x3c230dce) + attempt / 2U;
    const uint32_t expected_bits = cursor == 0U
        ? magnitude_bits | UINT32_C(0x80000000)
        : magnitude_bits;
    check(exhaustion_audit.attempts[attempt].cursor_index == cursor &&
              exhaustion_audit.attempts[attempt].delta_bits ==
                  expected_bits &&
              exhaustion_audit.attempts[attempt].status ==
                  G1RootReachAuditRejected,
          "two-cursor refusal alternates through the amended ceiling");
}

// One-cursor projection-refusal audit:
check(refusal_audit.attempt_count == G1RootReachTestAttemptLimit,
      "refusal audit owns the amended global limit");
for (uint32_t attempt = 0U;
     attempt < G1RootReachTestAttemptLimit;
     ++attempt) {
    check(refusal_audit.attempts[attempt].cursor_index == 0U &&
              refusal_audit.attempts[attempt].delta_bits ==
                  UINT32_C(0x3c8b379d) + attempt &&
              refusal_audit.attempts[attempt].status ==
                  G1RootReachAuditRejected,
          "refusal audit owns every amended rejected attempt");
}
```

Do not change the existing four-attempt authentic low-curb trace or the two-attempt reject-then-other-cursor trace. The direct `0x3c8b37dd` and `0x3c8b381d` calls are diagnostic assertions outside the production loop; the audit must still stop at `0x3c8b37dc`.

- [ ] **Step 7: Add both live audit records to strict/fast parity output**

Under the seam macro, add:

```cpp
static bool g1_test_print_root_reach_live_parity(
    const char* name,
    const G1TestRootReachLiveFixtureBits& bits)
{
    G1RootReachTestFixture fixture;
    g1_test_make_root_reach_live_fixture(fixture, bits);
    G1RootReachPlan plan = {};
    G1RootReachPlannerAudit audit = {};
    char error[512] = {};
    if (!g1_plan_recorded_contact_root_reach_audited(
            plan, audit,
            fixture.positions, fixture.rotations, fixture.parents,
            fixture.contacts, fixture.targets[0], fixture.targets[1],
            error, static_cast<int>(sizeof(error)))) {
        return false;
    }
    std::printf(
        "root-frontier %s plan=%u,%u,%u,%08x cursors=%u attempts=%u\n",
        name,
        plan.active ? 1U : 0U,
        plan.common_interval_found ? 1U : 0U,
        plan.applied ? 1U : 0U,
        terrain_float_bits(plan.root_y_delta_m),
        audit.cursor_count, audit.attempt_count);
    for (uint32_t cursor = 0U; cursor < audit.cursor_count; ++cursor) {
        std::printf(
            "root-frontier %s cursor=%u interval=%u initial=%08x\n",
            name, cursor,
            audit.cursors[cursor].interval_index,
            audit.cursors[cursor].initial_delta_bits);
    }
    for (uint32_t attempt = 0U;
         attempt < audit.attempt_count;
         ++attempt) {
        std::printf(
            "root-frontier %s attempt=%u cursor=%u bits=%08x status=%u\n",
            name, attempt,
            audit.attempts[attempt].cursor_index,
            audit.attempts[attempt].delta_bits,
            static_cast<unsigned int>(audit.attempts[attempt].status));
    }
    return true;
}
```

At the start of `run_runtime_parity_mode`, before its existing output, require:

```cpp
if (!g1_test_print_root_reach_live_parity(
        "row6", G1TestRootReachRow6) ||
    !g1_test_print_root_reach_live_parity(
        "row7", G1TestRootReachRow7)) {
    return 1;
}
```

Expected: strict and optimized callers will later emit every row-6 and row-7 cursor and attempt identically, not merely the final plan.

- [ ] **Step 8: Run the focused RED build and confirm the failure is caused only by the old ceiling**

Run:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/red
mkdir -p "$out"
strict=(-std=c++17 -O2 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test.o"
g++ "$out/test.o" "$out/root-reach.o" "$out/clearance.o" \
  -o "$out/test"
if "$out/test" > "$out/stdout" 2> "$out/stderr"; then
  echo 'ERROR: 32-attempt production unexpectedly passed 64-attempt RED' >&2
  exit 1
fi
rg -n 'live root frontier publishes the exact plan' "$out/stderr"
```

Expected: all compilation and linking succeeds; the executable exits nonzero with exactly the named live-frontier assertion. No missing symbol, invalid fixture, FK error, sanitizer error, or unrelated inherited assertion is acceptable RED evidence.

- [ ] **Step 9: Implement the minimal GREEN production and audit changes**

In `g1_ik.h`, change only the attempt capacity:

```cpp
struct G1RootReachPlannerAudit
{
    G1RootReachAuditCursor cursors[4];
    G1RootReachAuditAttempt attempts[64];
    uint32_t cursor_count;
    uint32_t attempt_count;
};
```

In `g1_ik_root_reach.cpp`, change only the private bound and add the seam-only ownership assertion immediately after it:

```cpp
static constexpr uint32_t
    G1RootReachMaximumBoundaryCandidateTests = 64U;

#if defined(G1_IK_ENABLE_TEST_SEAMS)
static_assert(
    G1RootReachMaximumBoundaryCandidateTests ==
        static_cast<uint32_t>(
            sizeof(((G1RootReachPlannerAudit*)nullptr)->attempts) /
            sizeof(((G1RootReachPlannerAudit*)nullptr)->attempts[0])),
    "G1 root-reach production ceiling must equal audit capacity");
#endif
```

Do not edit the loop body at lines 1040-1114. Its strict `<` condition already gives at most 64 revalidations, and its existing post-rejection `nextafter` preserves the permitted unconsumed-head semantics after rejection 64.

- [ ] **Step 10: Run focused GREEN and perform the narrow refactor/ownership scan**

Run the Step-8 build again, without the expected-failure wrapper:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/green
mkdir -p "$out"
strict=(-std=c++17 -O2 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math -I.)
g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test.o"
g++ "$out/test.o" "$out/root-reach.o" "$out/clearance.o" \
  -o "$out/test"
"$out/test"
```

Expected: executable exits `0` with no output.

Then run the refactor/ownership scan:

```bash
rg -n 'attempt_count == 32U|attempt < 32U|candidate_index < 32U|32-attempt|32-rejection' \
  tests/cpp/test_g1_ik.cpp && exit 1 || true
test "$(rg -n 'G1RootReachMaximumBoundaryCandidateTests = 64U' \
  g1_ik_root_reach.cpp | wc -l)" -eq 1
test "$(rg -n 'G1RootReachAuditAttempt attempts\[64\]' g1_ik.h | wc -l)" -eq 1
git diff --check -- g1_ik.h g1_ik_root_reach.cpp \
  tests/cpp/test_g1_ik.cpp tests/cpp/g1_root_reach_live_fixture_bits.h
```

Expected: no stale attempt-limit assertion remains, exactly one production ceiling and one audit capacity exist, and the diff check is silent. Refactoring is limited to sharing `G1RootReachTestAttemptLimit` in test loops; do not move the production constant into a public header or alter planner scheduling.

- [ ] **Step 11: Run strict, optimized-caller parity, ordinary no-seam, positive, negative, guard, and sanitizer gates**

Run this complete matrix:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/matrix
mkdir -p "$out"
strict=(-std=c++17 -O2 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math \
  -frecord-gcc-switches -I.)
fast=(-std=c++17 -O3 -g -Wall -Wextra -Werror -pedantic \
  -ffast-math -DNDEBUG -I.)

g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-audit.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-no-seam.o"
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"

g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test-strict.o"
g++ "$out/test-strict.o" "$out/root-audit.o" "$out/clearance.o" \
  -o "$out/test-strict"
"$out/test-strict"
"$out/test-strict" --parity > "$out/strict.parity"

g++ "${fast[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test-fast.o"
g++ "$out/test-fast.o" "$out/root-audit.o" "$out/clearance.o" \
  -o "$out/test-fast"
"$out/test-fast"
"$out/test-fast" --parity > "$out/fast.parity"
cmp "$out/strict.parity" "$out/fast.parity"
rg 'root-frontier row6 plan=1,1,1,bbd51d41 cursors=1 attempts=47' \
  "$out/strict.parity"
rg 'root-frontier row7 plan=1,1,1,bbc6abc1 cursors=1 attempts=39' \
  "$out/strict.parity"

g++ "${strict[@]}" -c tests/cpp/test_g1_ik.cpp \
  -o "$out/test-no-seam.o"
g++ "$out/test-no-seam.o" "$out/root-no-seam.o" "$out/clearance.o" \
  -o "$out/test-no-seam"
"$out/test-no-seam"
if nm -C "$out/root-no-seam.o" | \
     rg 'g1_plan_recorded_contact_root_reach_audited|g1_root_reach_audit'; then
  echo 'ERROR: no-seam root object exposes audit code' >&2
  exit 1
fi

g++ "${strict[@]}" -c tests/cpp/compile_g1_ik_production.cpp \
  -o "$out/production.o"
g++ "$out/production.o" "$out/root-no-seam.o" "$out/clearance.o" \
  -o "$out/production"
"$out/production"
if g++ "${strict[@]}" -c tests/cpp/compile_g1_ik_seam_negative.cpp \
     -o "$out/seam-negative.o" 2> "$out/seam-negative.stderr"; then
  echo 'ERROR: private IK seam compiled without its macro' >&2
  exit 1
fi
rg 'was not declared in this scope' "$out/seam-negative.stderr"

if g++ -std=c++17 -O3 -ffast-math -DNDEBUG -I. \
     -c g1_ik_root_reach.cpp -o "$out/forbidden-fast-root.o" \
     2> "$out/forbidden-fast-root.stderr"; then
  echo 'ERROR: root-reach strict object accepted fast math' >&2
  exit 1
fi
rg 'must be compiled without fast math' "$out/forbidden-fast-root.stderr"
readelf --string-dump=.GCC.command.line "$out/root-audit.o" | \
  rg -- '-fno-fast-math'
readelf --string-dump=.GCC.command.line "$out/root-audit.o" | \
  rg -- '-ffp-contract=off'
readelf --string-dump=.GCC.command.line "$out/root-audit.o" | \
  rg -- '-frounding-math'
if readelf -SW "$out/root-audit.o" | rg '\.gnu\.lto'; then
  echo 'ERROR: root-reach object contains LTO sections' >&2
  exit 1
fi
if nm -C "$out/root-audit.o" | \
     rg ' U (operator new|operator new\[\]|malloc|calloc|realloc)'; then
  echo 'ERROR: root-reach amendment introduced dynamic allocation' >&2
  exit 1
fi

san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-fast-math -ffp-contract=off \
  -frounding-math -I.)
g++ "${san[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-san.o"
g++ "${san[@]}" -c g1_clearance.cpp -o "$out/clearance-san.o"
g++ "${san[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test-san.o"
g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all \
  "$out/test-san.o" "$out/root-san.o" "$out/clearance-san.o" \
  -o "$out/test-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$out/test-san"
```

Expected: every positive executable exits `0`; strict and fast parity files are byte-identical and contain the exact row lines; no-seam symbols are absent; both negative compiles fail for the intended reason; strict object flags are recorded; LTO and allocation searches are empty; sanitizers are silent.

- [ ] **Step 12: Build a disposable controller and certify the exact 8-frame live boundary**

Run:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/live
mkdir -p "$out"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -DNDEBUG -I. \
  -c g1_clearance.cpp -o "$out/clearance.o"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -DNDEBUG -I. \
  -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src \
  -c controller.cpp -o "$out/controller.o"
g++ "$out/controller.o" "$out/clearance.o" "$out/root-reach.o" \
  -o "$out/controller" -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

DISPLAY=:1 MM_IK=1 "$out/controller" \
  --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
  --terrain-weight 4 --test-mode route --test-route curb-forward \
  --test-frames 8 --test-heading forward \
  --terrain-scene grail-curb-low --log "$out/low-curb-8.csv"

/home/ubuntu/miniconda3/envs/diffsim/bin/python - "$out/low-curb-8.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 8
assert [int(row["frame"]) for row in rows] == list(range(8))
row6 = rows[6]
row7 = rows[7]
assert int(row6["frame_rejected"]) == 1
assert row6["frame_rejection_stage"] == "ik-candidate"
assert row6["rejected_stop_reason"] == "no-swing-candidate"
assert int(row6["ik_safe_stop_latched"]) == 1
assert int(row7["frame_rejected"]) == 0
assert row7["frame_rejection_stage"] == "none"
assert row7["rejected_stop_reason"] == "none"
assert int(row7["ik_safe_stop_latched"]) == 0
assert int(row7["database_frame"]) == 429693
assert int(row7["range"]) == 1650
assert float(row7["left_contact_residual"]) < 0.005
assert float(row7["right_contact_residual"]) < 0.005
assert all(row["rejected_stop_reason"] != "target-unreachable" for row in rows)
print("VALID root-frontier-live-8 row6=no-swing row7=accepted")
PY
```

Expected: the controller exits `0`, writes exactly eight 25 Hz rows, and the script prints `VALID root-frontier-live-8 row6=no-swing row7=accepted`. Row 6 remains the independent no-swing failure; row 7 accepts database frame `429693` in range `1650` instead of finite-rejecting `target-unreachable`.

- [ ] **Step 13: Commit the narrow implementation**

Review and commit exactly four files:

```bash
set -euo pipefail
git diff --check -- g1_ik.h g1_ik_root_reach.cpp \
  tests/cpp/test_g1_ik.cpp tests/cpp/g1_root_reach_live_fixture_bits.h
git add g1_ik.h g1_ik_root_reach.cpp \
  tests/cpp/test_g1_ik.cpp tests/cpp/g1_root_reach_live_fixture_bits.h
test "$(git diff --cached --name-only | wc -l)" -eq 4
git diff --cached --name-only | sort > \
  /tmp/g1-root-frontier-amendment/staged-files.txt
diff -u - /tmp/g1-root-frontier-amendment/staged-files.txt <<'EOF'
g1_ik.h
g1_ik_root_reach.cpp
tests/cpp/g1_root_reach_live_fixture_bits.h
tests/cpp/test_g1_ik.cpp
EOF
git diff --cached --check
git commit -m "fix: extend G1 root reach boundary frontier"
```

Expected: the staged-file comparison and diff check pass, and one commit is created. Do not stage the amendment plan, either design spec, raw capture files, logs, binaries, or unrelated work during implementation execution.

---

### Task 2: Independent SDD Review Gate and 8-Frame Certification

**Files:**
- Review only: `g1_ik.h`
- Review only: `g1_ik_root_reach.cpp`
- Review only: `tests/cpp/g1_root_reach_live_fixture_bits.h`
- Review only: `tests/cpp/test_g1_ik.cpp`
- Reference only: `docs/superpowers/specs/2026-07-16-g1-root-reach-boundary-frontier-amendment-design.md`
- Produce outside the repository: `/tmp/g1-root-frontier-amendment/review/`

**Interfaces:**
- Consumes: the Task-1 commit, committed amendment design, exact fixture hashes, strict/fast parity text, sanitizer results, and disposable live logs.
- Produces: the subagent-driven-development two-stage reviewer gate (specification compliance first, code quality second) and a fresh 8-frame certification record. Task 2 is not assigned to a second implementation worker and does not modify or package production code when the Task-1 commit is correct.

- [ ] **Step 1: Freeze the review diff and reject scope expansion**

Run:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/review
mkdir -p "$out"
base=$(cat /tmp/g1-root-frontier-amendment/execution-base.txt)
git diff --name-only "$base"..HEAD | sort > "$out/changed-files.txt"
diff -u - "$out/changed-files.txt" <<'EOF'
g1_ik.h
g1_ik_root_reach.cpp
tests/cpp/g1_root_reach_live_fixture_bits.h
tests/cpp/test_g1_ik.cpp
EOF
git show --stat --oneline --decorate HEAD > "$out/commit.txt"
git diff --check "$base"..HEAD
```

Expected: exactly the four Task-1 files differ from the recorded execution base and the diff check is silent. Any controller, runtime, matcher, terrain, logging, visualizer, or unrelated documentation change blocks review.

- [ ] **Step 2: Dispatch an independent specification-compliance review**

Give a fresh specification-compliance reviewer only the committed amendment
design, this plan, the Task-1 commit hash, and
`git diff "$(cat /tmp/g1-root-frontier-amendment/execution-base.txt)"..HEAD`.
Require a written verdict covering each of these exact questions:

```text
1. Is the production budget exactly 64 globally shared revalidations, not per cursor?
2. Are row 6 and row 7 frozen at 47/0xbbd51d41 and 39/0xbbc6abc1 attempts/bits?
3. Is nearest-first ordering and one-ULP inward advancement unchanged?
4. Can rejection 64 still materialize one unconsumed nextafter head without revalidating or auditing it?
5. Is audit capacity exactly 64 and compile-time-equal to the private loop bound?
6. Are strict authentication, validation, no-seam privacy, and fail-closed publication unchanged?
7. Do the +64 and +128 refusal checks run diagnostically outside the production loop?
8. Does the live oracle preserve row-6 no-swing while requiring row-7 acceptance?
9. Is every non-goal and unrelated file untouched?
```

Expected: reviewer returns `APPROVED` or concrete file/line findings. Generic praise without answers to all nine questions is not an approval.

- [ ] **Step 3: Dispatch an independent numerical/code-quality review**

Give a second fresh reviewer the same diff plus the twelve raw-capture hashes and the generated-header SHA-256. Require a written verdict covering:

```text
1. Fixture reconstruction preserves every float bit, target flag, parent, and contact bit without ABI-dependent reads.
2. The row fixtures do not load runtime files and cannot mutate planner inputs.
3. Every audit attempt is compared by cursor, bits, and status.
4. One-cursor, representational no-op, and two-cursor exhaustion all prove exactly 64 global rejections.
5. Existing 4-attempt and 2-attempt short traces remain unchanged.
6. Strict/fast parity serializes complete row-6 and row-7 audits.
7. The implementation diff is limited to 32->64 plus one compile-time assertion.
8. No allocation, new public constant, new seam, log field, or loop-body change exists.
```

Expected: reviewer returns `APPROVED` or concrete file/line findings. If either reviewer reports a finding, return it to the Task-1 implementation worker, add a separate fix commit, rerun every Task-1 verification step, and repeat both independent reviews. Do not waive a finding by changing the design after implementation.

- [ ] **Step 4: Rebuild and rerun the complete focused matrix from the reviewed commit**

Run a fresh matrix rather than reusing Task-1 objects:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/review/matrix
mkdir -p "$out"
strict=(-std=c++17 -O2 -g -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -frounding-math \
  -frecord-gcc-switches -I.)
fast=(-std=c++17 -O3 -g -Wall -Wextra -Werror -pedantic \
  -ffast-math -DNDEBUG -I.)

g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-audit.o"
g++ "${strict[@]}" -c g1_ik_root_reach.cpp -o "$out/root-no-seam.o"
g++ "${strict[@]}" -c g1_clearance.cpp -o "$out/clearance.o"
g++ "${strict[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test-strict.o"
g++ "$out/test-strict.o" "$out/root-audit.o" "$out/clearance.o" \
  -o "$out/test-strict"
"$out/test-strict"
"$out/test-strict" --parity > "$out/strict.parity"

g++ "${fast[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test-fast.o"
g++ "$out/test-fast.o" "$out/root-audit.o" "$out/clearance.o" \
  -o "$out/test-fast"
"$out/test-fast"
"$out/test-fast" --parity > "$out/fast.parity"
cmp "$out/strict.parity" "$out/fast.parity"
rg 'root-frontier row6 plan=1,1,1,bbd51d41 cursors=1 attempts=47' \
  "$out/strict.parity"
rg 'root-frontier row7 plan=1,1,1,bbc6abc1 cursors=1 attempts=39' \
  "$out/strict.parity"

g++ "${strict[@]}" -c tests/cpp/test_g1_ik.cpp \
  -o "$out/test-no-seam.o"
g++ "$out/test-no-seam.o" "$out/root-no-seam.o" "$out/clearance.o" \
  -o "$out/test-no-seam"
"$out/test-no-seam"
if nm -C "$out/root-no-seam.o" | \
     rg 'g1_plan_recorded_contact_root_reach_audited|g1_root_reach_audit'; then
  exit 1
fi

g++ "${strict[@]}" -c tests/cpp/compile_g1_ik_production.cpp \
  -o "$out/production.o"
g++ "$out/production.o" "$out/root-no-seam.o" "$out/clearance.o" \
  -o "$out/production"
"$out/production"
if g++ "${strict[@]}" -c tests/cpp/compile_g1_ik_seam_negative.cpp \
     -o "$out/seam-negative.o" 2> "$out/seam-negative.stderr"; then
  exit 1
fi
rg 'was not declared in this scope' "$out/seam-negative.stderr"

san=(-std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all -fno-fast-math -ffp-contract=off \
  -frounding-math -I.)
g++ "${san[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c g1_ik_root_reach.cpp -o "$out/root-san.o"
g++ "${san[@]}" -c g1_clearance.cpp -o "$out/clearance-san.o"
g++ "${san[@]}" -DG1_IK_ENABLE_TEST_SEAMS \
  -c tests/cpp/test_g1_ik.cpp -o "$out/test-san.o"
g++ -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero \
  -fno-sanitize-recover=all \
  "$out/test-san.o" "$out/root-san.o" "$out/clearance-san.o" \
  -o "$out/test-san"
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  "$out/test-san"

for source in \
  tests/cpp/test_g1_controller_state.cpp \
  tests/cpp/test_g1_frame_transaction.cpp; do
  name=$(basename "$source" .cpp)
  g++ "${strict[@]}" -c "$source" -o "$out/$name.o"
  g++ "$out/$name.o" "$out/root-no-seam.o" "$out/clearance.o" \
    -o "$out/$name"
  "$out/$name"
done
```

Expected: all positive tests exit `0`; parity files are byte-identical; no-seam symbols remain absent; negative compilation fails as intended; sanitizers are silent; inherited controller-state and frame-transaction ownership tests still pass.

- [ ] **Step 5: Rebuild the disposable controller and certify only the exact 8-frame boundary**

Run:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/review/live
mkdir -p "$out"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -DNDEBUG -I. \
  -c g1_clearance.cpp -o "$out/clearance.o"
g++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -frounding-math -frecord-gcc-switches -DNDEBUG -I. \
  -c g1_ik_root_reach.cpp -o "$out/root-reach.o"
g++ -std=c++17 -O3 -ffast-math -DNDEBUG \
  -D_DEFAULT_SOURCE -DPLATFORM_DESKTOP -I. \
  -isystem /home/ubuntu/apps/raylib/src \
  -isystem /home/ubuntu/apps/raygui/src \
  -c controller.cpp -o "$out/controller.o"
g++ "$out/controller.o" "$out/clearance.o" "$out/root-reach.o" \
  -o "$out/controller" -L/home/ubuntu/apps/raylib/src \
  -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
sha256sum "$out/controller" > "$out/controller.sha256"

timeout 30s env DISPLAY=:1 MM_IK=1 "$out/controller" \
  --terrain-dir /tmp/g1-terrain-footprint-runtime-v1 \
  --terrain-weight 4 --test-mode route --test-route curb-forward \
  --test-frames 8 --test-heading forward \
  --terrain-scene grail-curb-low --log "$out/low-curb-8.csv"

/home/ubuntu/miniconda3/envs/diffsim/bin/python - \
  "$out/low-curb-8.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 8
assert [int(row["frame"]) for row in rows] == list(range(8))
assert all(abs(float(row["fixed_dt"]) - 0.0399999991) < 1e-12
           for row in rows)
row6 = rows[6]
row7 = rows[7]
assert int(row6["frame_rejected"]) == 1
assert row6["rejected_stop_reason"] == "no-swing-candidate"
assert int(row7["frame_rejected"]) == 0
assert row7["rejected_stop_reason"] == "none"
assert int(row7["database_frame"]) == 429693
assert int(row7["range"]) == 1650
assert float(row7["left_contact_residual"]) < 0.005
assert float(row7["right_contact_residual"]) < 0.005
print("VALID root-frontier-live-8")
PY
```

Expected: the process exits `0` before the 30-second watchdog, the log
preserves eight exact 25 Hz rows, and the script prints
`VALID root-frontier-live-8`. Do not run 32 or 800 frames in this task. Those
horizons remain deferred until row-6 matcher/swing recovery first passes the
standing zero-rejection 32-frame prerequisite.

- [ ] **Step 6: Record final evidence without another repository change**

Run:

```bash
set -euo pipefail
out=/tmp/g1-root-frontier-amendment/review
git diff --exit-code
git diff --cached --exit-code
git log -1 --format='%H %s' > "$out/final-commit.txt"
sha256sum tests/cpp/g1_root_reach_live_fixture_bits.h \
  > "$out/fixture-header.sha256"
test "$(cut -d' ' -f1 "$out/fixture-header.sha256")" = \
  dc531c87d6ce77c7cf85e425dd4a80a56d5133954d9cfda3a7a42fc5bde447d8
git status --short > "$out/final-status.txt"
```

Expected: tracked and staged diffs are empty, the committed fixture hash is exact, and any lines in `final-status.txt` are pre-existing unrelated untracked files only. Task 2 creates no commit when both reviewers approve and all certification gates pass.
