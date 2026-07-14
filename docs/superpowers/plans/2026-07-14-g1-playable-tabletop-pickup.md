# G1 Playable Tabletop Pickup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a playable Raylib loop in which the existing flat-ground G1 locomotion approaches one tabletop target, explicitly selects and replays a matched pickup, attaches it only after verified contact, and returns controllable carrying motion.

**Architecture:** A Raylib-free C++ interaction runtime consumes the existing schema-v1 database and feature pack through focused target, feature, matcher, playback, IK, attachment, carry, and coordinator modules. A narrow adapter connects that runtime to the existing controller pose and input state, while Raylib-specific drawing remains in one debug header. The first baseline selects one valid pre-contact entry and advances that clip contiguously; staged frame-level interaction matching remains a follow-on.

**Tech Stack:** C++17, GNU Make, existing `vec3`/`quat`/`array` math, schema-v1 interaction artifacts at 25 Hz, Raylib 6.0, raygui 4.0, Python 3.10+ `unittest`, NumPy/SciPy/MuJoCo for offline parity fixtures.

## Global Constraints

- Work only in `/home/ubuntu/worktrees/motion-matching/g1-manipulation-motion-matching` on branch `g1-manipulation-motion-matching`.
- Do not merge from, rebase onto, or write to `/home/ubuntu/projects/motion-matching`; immutable terrain-derived foundation commit `cbe90b7` remains the branch base.
- Preserve the existing flat-ground controller output whenever interaction is disabled, idle, rejected, or missing its artifact pack.
- Do not change the schema-v1 binary layout, its exact magics `G1INTDB1` / `G1INTFT1`, or the ordinary locomotion files `resources/database.bin` / `resources/features.bin`.
- Canonical interaction time remains exactly 25 Hz. Render/update sampling interpolates in seconds and never crosses a clip range.
- Level 1 supports one rigid tabletop target, one authored affordance, and one active hand. Target selection is explicit even though the first resolver has one object.
- Runtime states are `Disabled`, `Locomotion`, `Preflight`, `Align`, `PickupReplay`, `Hold`, and `Carry`.
- Use a 0.25-second entry blend and at most 1.00 second of local alignment.
- Hard limits are: target approach `1.00 m`, residual planar root correction `0.25 m`, residual yaw `25 degrees`, hand position correction `0.12 m`, hand orientation correction `25 degrees`, and playback speed `[0.85, 1.15]`.
- Feature groups are pose `[0,33)`, trajectory `[33,45)`, grasp `[45,57)`, root target `[57,65)`, and context `[65,71)` with weights `1,1,2,2,1`. The weighted mean-squared normalized acceptance threshold is `9.0`.
- Attachment requires hand position error at most `0.04 m`, hand orientation error at most `15 degrees`, the authored hand, valid correction/joint bounds, and the same target generation.
- Pickup success requires attachment, object-origin lift of at least `0.15 m`, and one continuous second held above that height.
- A recorded carry range requires at least 25 contiguous `HOLD` frames, continuous active-hand contact, hand-in-object drift at most `0.02 m` / `10 degrees`, planar root travel at least `0.30 m`, and average planar root speed at least `0.20 m/s`.
- If no recorded carry range is valid, use the layered carry fallback: locomotion lower body, full recorded active arm, 25% recorded spine blend, 35% recorded inactive-arm blend, and bounded hand IK.
- Interact uses keyboard `F` / gamepad right-face-left; cancel uses keyboard `X` / gamepad right-face-up; Drop/Reset uses keyboard `R` / gamepad right-face-right. Do not reuse `E`, which already controls facing.
- Interaction unit tests require no display and no network. Only the explicit dependency bootstrap target may access GitHub.
- Pin Raylib commit `dbc56a87da87d973a9c5baa4e7438a9d20121d28` (tag `6.0`) and raygui commit `25c8c65a6e5f0f4d4b564a0343861898c6f2778b` (tag `4.0`).
- Generated dependencies live under ignored `.deps/`; generated interaction packs remain under ignored `resources/g1_interaction/`.
- Do not claim the visual gate passes until the Raylib controller compiles and an auto-demo produces a screenshot plus a deterministic state log showing `Locomotion -> Preflight -> Align -> PickupReplay -> Hold -> Carry`.

---

## File and Responsibility Map

| Path | Responsibility |
| --- | --- |
| `scripts/bootstrap_raylib.sh` | Fetch and build the two pinned visual dependencies without affecting system packages. |
| `interaction_pose.h/.cpp` | Fixed G1 pose/transform types, artifact frame access, FK, interpolation, and controller-independent pose blending. |
| `interaction_target.h/.cpp` | Stable target handles, affordances, registry, reservation, and one-target resolver. |
| `interaction_features.h/.cpp` | Exact runtime construction and normalization of the frozen 71-D query. |
| `interaction_query_probe.cpp` | Cross-language query-parity executable. |
| `interaction_matcher.h/.cpp` | Hard filters, scene alignment, clearance, group costs, and whole-clip selection. |
| `interaction_playback.h/.cpp` | Clip-safe 25 Hz clock, interpolation, source-to-scene transform, and decaying entry correction. |
| `g1_arm_joint_metadata.h` | Frozen canonical arm rest rotations, hinge axes, and limits derived from the G1 XML. |
| `resources/generate_g1_arm_joint_metadata.py` | Reproducibly derive the tracked arm metadata header from a G1 XML. |
| `interaction_ik.h/.cpp` | Limit-aware seven-hinge hand position/orientation cleanup. |
| `interaction_attachment.h/.cpp` | Contact gates, attachment transform, lift, hold, failure, and reset. |
| `interaction_carry.h/.cpp` | Recorded carry qualification/search and deterministic layered fallback. |
| `interaction_runtime.h/.cpp` | Game-facing coordinator and complete interaction state machine. |
| `interaction_controller_adapter.h/.cpp` | Convert controller arrays/snapshots to runtime values and preserve no-interaction behavior. |
| `interaction_runtime_probe.cpp` | Deterministic end-to-end, Raylib-free state/log probe. |
| `interaction_debug_draw.h` | Raylib-only table/object, affordance, correction, state, and text drawing. |
| `controller.cpp` | Minimal input, update, pose handoff, scene, and debug-draw wiring. |
| `tests/cpp/interaction_runtime_fixture.h` | Coherent two-clip in-memory fixture shared by runtime C++ tests. |
| `tests/cpp/test_interaction_*.cpp` | Focused headless tests for each runtime unit. |
| `tests/python/test_interaction_query_parity.py` | Python/C++ runtime-query parity against published artifacts. |
| `tests/python/test_raylib_bootstrap.py` | Offline lock/Make-contract tests for visual dependencies. |
| `tests/python/test_playable_interaction_evidence.py` | Validate auto-demo state order, attachment, carry motion, and screenshot evidence. |
| `Makefile` | Build pure tests/probes, pinned dependencies, controller, demo pack, and playable gate. |
| `.gitignore` | Ignore dependency sources/builds and generated demo evidence. |
| `README.md` | Exact build, run, controls, fallback, evidence, and current-scope instructions. |

## Frozen Runtime Interfaces

These names are shared across tasks and must not drift.

```cpp
namespace interaction {

enum class Hand : uint8_t { Left = 0, Right = 1 };
enum class Phase : uint8_t {
    Approach = 0, Reach = 1, Contact = 2, Lift = 3, Hold = 4
};
enum class RuntimeState : uint8_t {
    Disabled, Locomotion, Preflight, Align, PickupReplay, Hold, Carry
};
enum class ObjectState : uint8_t { Free, Targeted, Attached, Held };
enum class ResultCode : uint8_t {
    None, Accepted, Succeeded, Rejected, Cancelled, Failed, Reset
};
enum class Reason : uint8_t {
    None, PackUnavailable, TargetUnavailable, TargetChanged, OutOfRange,
    NoCandidate, PoorMatch, BlockedPath, CorrectionLimit, Cancelled,
    ContactPosition, ContactOrientation, JointLimit, LostContact,
    ClipEnded, Reset
};

struct Transform { vec3 position; quat rotation; };

struct Pose {
    std::array<vec3, g1_skeleton::BoneCount> positions{};
    std::array<vec3, g1_skeleton::BoneCount> velocities{};
    std::array<quat, g1_skeleton::BoneCount> rotations{};
    std::array<vec3, g1_skeleton::BoneCount> angular_velocities{};
    std::array<float, 14> hand_dof{};
    std::array<float, 14> hand_dof_velocities{};
    std::array<uint8_t, 2> foot_contacts{};
};

struct WorldPose {
    std::array<vec3, g1_skeleton::BoneCount> positions{};
    std::array<vec3, g1_skeleton::BoneCount> velocities{};
    std::array<quat, g1_skeleton::BoneCount> rotations{};
    std::array<vec3, g1_skeleton::BoneCount> angular_velocities{};
};

struct LocomotionSnapshot {
    Pose pose;
    std::array<vec3, 3> future_root_positions{};
    std::array<quat, 3> future_root_rotations{};
};

struct TargetHandle {
    uint64_t id = 0;
    uint32_t generation = 0;
    friend constexpr bool operator==(TargetHandle left, TargetHandle right) {
        return left.id == right.id && left.generation == right.generation;
    }
    friend constexpr bool operator!=(TargetHandle left, TargetHandle right) {
        return !(left == right);
    }
};

struct MatchConfig {
    float maximum_approach_m = 1.00F;
    float maximum_root_correction_m = 0.25F;
    float maximum_yaw_correction_radians = 0.436332313F;
    float maximum_hand_correction_m = 0.12F;
    float maximum_hand_orientation_radians = 0.436332313F;
    std::array<float, 5> group_weights{1.0F, 1.0F, 2.0F, 2.0F, 1.0F};
    float maximum_cost = 9.0F;
};

struct PlaybackConfig {
    float canonical_fps = 25.0F;
    float speed = 1.0F;
    float minimum_speed = 0.85F;
    float maximum_speed = 1.15F;
    float entry_blend_seconds = 0.25F;
    float maximum_alignment_seconds = 1.00F;
    float commit_horizon_seconds = 0.50F;
};

struct IKConfig {
    float maximum_request_position_m = 0.12F;
    float maximum_request_orientation_radians = 0.436332313F;
    float accepted_position_m = 0.04F;
    float accepted_orientation_radians = 0.261799388F;
    float damping = 0.05F;
    float finite_difference_radians = 0.001F;
    float orientation_scale_m_per_radian = 0.25F;
    float maximum_step_radians = 0.10F;
    int32_t maximum_iterations = 8;
};

struct AttachmentConfig {
    float maximum_position_error_m = 0.04F;
    float maximum_orientation_error_radians = 0.261799388F;
    float required_lift_m = 0.15F;
    float required_hold_seconds = 1.00F;
};

struct CarryConfig {
    int32_t minimum_hold_frames = 25;
    float maximum_grasp_drift_m = 0.02F;
    float maximum_grasp_drift_radians = 0.174532925F;
    float minimum_root_displacement_m = 0.30F;
    float minimum_average_speed_mps = 0.20F;
    float search_interval_seconds = 0.10F;
    float spine_weight = 0.25F;
    float inactive_arm_weight = 0.35F;
};

struct RuntimeConfig {
    MatchConfig matcher{};
    PlaybackConfig playback{};
    IKConfig ik{};
    AttachmentConfig attachment{};
    CarryConfig carry{};
};

struct GraspAffordance {
    uint32_t id = 0;
    Hand hand = Hand::Right;
    Transform hand_in_object{};
    vec3 approach_direction_object{};
    float clearance_radius = 0.04F;
};

struct InteractionTarget {
    TargetHandle handle{};
    Transform object_world{};
    vec3 object_dimensions{};
    Transform table_world{};
    vec3 table_size{};
    ObjectState state = ObjectState::Free;
    uint64_t owner_request = 0;
    std::vector<GraspAffordance> affordances;
};

struct PickRequest {
    TargetHandle target{};
    uint32_t affordance_id = 0;
    uint64_t request_id = 0;
};

struct RuntimeInput {
    float dt = 0.0F;
    LocomotionSnapshot locomotion{};
    bool interact_pressed = false;
    std::optional<PickRequest> pick_request{};
    bool cancel_pressed = false;
    bool reset_pressed = false;
};

struct RuntimeDiagnostics {
    RuntimeState state = RuntimeState::Disabled;
    ResultCode result = ResultCode::None;
    Reason reason = Reason::None;
    TargetHandle target{};
    ObjectState object_state = ObjectState::Free;
    uint32_t affordance_id = 0;
    int32_t clip = -1;
    int32_t frame = -1;
    Phase phase = Phase::Approach;
    Hand hand = Hand::Right;
    float total_cost = 0.0F;
    std::array<float, 5> group_costs{};
    float requested_root_correction_m = 0.0F;
    float applied_root_correction_m = 0.0F;
    float requested_yaw_correction_radians = 0.0F;
    float applied_yaw_correction_radians = 0.0F;
    float playback_speed = 1.0F;
    float hand_position_error_m = 0.0F;
    float hand_orientation_error_radians = 0.0F;
    bool attached = false;
    bool recorded_carry = false;
    bool pack_available = false;
};

struct RuntimeOutput {
    bool owns_pose = false;
    bool suppress_steering = false;
    Pose pose{};
    Transform object_world{};
    RuntimeDiagnostics diagnostics{};
};

}  // namespace interaction
```

---

### Task 1: Pin and Prove the Raylib Toolchain

**Files:**
- Create: `scripts/bootstrap_raylib.sh`
- Create: `tests/python/test_raylib_bootstrap.py`
- Modify: `.gitignore`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Git, GNU Make, the official Raylib/raygui repositories.
- Produces: `.deps/raylib/src/libraylib.a`, `.deps/raygui/src/raygui.h`, `make bootstrap-raylib`, and a Linux desktop `make controller` path.

- [ ] **Step 1: Write the failing offline lock test**

```python
class RaylibBootstrapTests(unittest.TestCase):
    def test_print_lock_is_exact_and_offline(self):
        completed = subprocess.run(
            ["bash", "scripts/bootstrap_raylib.sh", "--print-lock"],
            check=True, text=True, capture_output=True,
            env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
        )
        self.assertEqual(
            completed.stdout,
            "raylib dbc56a87da87d973a9c5baa4e7438a9d20121d28\n"
            "raygui 25c8c65a6e5f0f4d4b564a0343861898c6f2778b\n",
        )

    def test_make_dry_run_uses_local_dependencies(self):
        completed = subprocess.run(
            ["make", "-n", "controller"], check=True,
            text=True, capture_output=True,
        )
        self.assertIn(".deps/raylib/src", completed.stdout)
        self.assertIn(".deps/raygui/src", completed.stdout)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.python.test_raylib_bootstrap -v`

Expected: `test_print_lock_is_exact_and_offline` errors because `scripts/bootstrap_raylib.sh` does not exist, and the Make dry run lacks the local paths.

- [ ] **Step 3: Implement the exact bootstrap and Linux Make path**

Create the script with these complete behaviors:

```bash
#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
deps="${DEPS_DIR:-$root/.deps}"
raylib_commit=dbc56a87da87d973a9c5baa4e7438a9d20121d28
raygui_commit=25c8c65a6e5f0f4d4b564a0343861898c6f2778b

if [[ "${1:-}" == "--print-lock" ]]; then
    printf 'raylib %s\nraygui %s\n' "$raylib_commit" "$raygui_commit"
    exit 0
fi

checkout_exact() {
    local name="$1" url="$2" commit="$3" path="$deps/$1"
    if [[ ! -d "$path/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$url" "$path"
    fi
    git -C "$path" fetch --depth 1 origin "$commit"
    git -C "$path" checkout --detach "$commit"
    test "$(git -C "$path" rev-parse HEAD)" = "$commit"
    test -z "$(git -C "$path" status --porcelain)"
}

mkdir -p "$deps"
checkout_exact raylib https://github.com/raysan5/raylib.git "$raylib_commit"
checkout_exact raygui https://github.com/raysan5/raygui.git "$raygui_commit"
make -C "$deps/raylib/src" PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=STATIC
test -f "$deps/raylib/src/libraylib.a"
test -f "$deps/raygui/src/raygui.h"
```

Add `.deps/` and `playable-evidence/` to `.gitignore`. On Linux, set `RAYLIB_DIR ?= .deps/raylib`, `RAYGUI_DIR ?= .deps/raygui`, include both `src` directories, link `.deps/raylib/src/libraylib.a`, and use `-lGL -lm -lpthread -ldl -lrt -lX11`. Preserve the existing Windows and web branches.

- [ ] **Step 4: Verify GREEN without network, then build the pinned dependencies**

Run: `python -m unittest tests.python.test_raylib_bootstrap -v`

Expected: 2 tests pass.

Run: `make bootstrap-raylib && make controller`

Expected: both pinned commits are checked out, `libraylib.a` is built, and `controller` links successfully.

- [ ] **Step 5: Commit the toolchain**

```bash
git add scripts/bootstrap_raylib.sh tests/python/test_raylib_bootstrap.py \
  .gitignore Makefile
git commit -m "build: pin playable Raylib toolchain"
```

### Task 2: Add Headless Pose, Transform, FK, and Sampling Primitives

**Files:**
- Create: `interaction_pose.h`
- Create: `interaction_pose.cpp`
- Create: `tests/cpp/test_interaction_pose.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `interaction::Database`, `g1_skeleton.h`, `vec.h`, and `quat.h`.
- Produces: `Transform`, `Pose`, `WorldPose`, `LocomotionSnapshot`, transform composition/inversion, `pose_at_frame`, `world_pose`, and clip-safe interpolated sampling.

- [ ] **Step 1: Write failing transform, FK, and sampling tests**

The test must assert:

```cpp
const Transform a{vec3(1.0F, 0.0F, 2.0F), quat()};
const Transform b{vec3(3.0F, 4.0F, 5.0F), quat()};
assert(near(compose(a, b).position, vec3(4.0F, 4.0F, 7.0F)));
assert(near(compose(a, inverse(a)).position, vec3()));

Pose pose{};
pose.positions[g1_skeleton::Simulation] = vec3(2.0F, 0.0F, 3.0F);
pose.positions[g1_skeleton::Hips] = vec3(0.0F, 1.0F, 0.0F);
const WorldPose world = world_pose(pose);
assert(near(world.positions[g1_skeleton::Hips], vec3(2.0F, 1.0F, 3.0F)));

const Pose halfway = sample_pose(database, 0, 0.02F);
assert(near(halfway.positions[0],
            0.5F * (pose_at_frame(database, 0).positions[0] +
                    pose_at_frame(database, 1).positions[0])));
assert(throws_range_error([&] { sample_pose(database, 0, 99.0F); }));
```

- [ ] **Step 2: Run the test and verify RED**

Run: `make build/tests/test_interaction_pose`

Expected: compilation fails because `interaction_pose.h` does not exist.

- [ ] **Step 3: Implement the fixed-size types and exact math**

Use the frozen interfaces and these formulas:

```cpp
Transform compose(const Transform& parent, const Transform& local) {
    return {
        parent.position + quat_mul_vec3(parent.rotation, local.position),
        quat_normalize(quat_mul(parent.rotation, local.rotation)),
    };
}

Transform inverse(const Transform& value) {
    const quat rotation = quat_inv(value.rotation);
    return {quat_mul_vec3(rotation, -value.position), rotation};
}

Pose sample_pose(
    const Database& database, uint32_t clip, float seconds_from_entry) {
    const int32_t start = database.range_starts.at(clip);
    const int32_t stop = database.range_stops.at(clip);
    const float frame = seconds_from_entry * 25.0F + start;
    if (frame < start || frame > static_cast<float>(stop - 1)) {
        throw std::out_of_range("interaction sample outside clip range");
    }
    const int32_t left = static_cast<int32_t>(std::floor(frame));
    const int32_t right = std::min(left + 1, stop - 1);
    return interpolate_pose(
        pose_at_frame(database, left), pose_at_frame(database, right),
        frame - static_cast<float>(left));
}
```

`world_pose` must implement parent-first FK with the exact
`g1_skeleton::kParents` hierarchy. For a non-root bone, use these equations,
where `r = quat_mul_vec3(parent_rotation, local_position)`:

```cpp
world_position = parent_position + r;
world_rotation = quat_normalize(quat_mul(parent_rotation, local_rotation));
world_velocity = parent_velocity +
    cross(parent_angular_velocity, r) +
    quat_mul_vec3(parent_rotation, local_velocity);
world_angular_velocity = parent_angular_velocity +
    quat_mul_vec3(parent_rotation, local_angular_velocity);
```

Quaternion interpolation uses `quat_nlerp_shortest`; discrete contacts select
the left sample for alpha `< 0.5` and the right sample otherwise.

- [ ] **Step 4: Build and run GREEN**

Run: `make build/tests/test_interaction_pose && build/tests/test_interaction_pose`

Expected: exit 0 with no output.

Run: `make test-cpp`

Expected: all C++ tests exit 0.

- [ ] **Step 5: Commit pose primitives**

```bash
git add interaction_pose.h interaction_pose.cpp \
  tests/cpp/test_interaction_pose.cpp Makefile
git commit -m "feat: add interaction pose primitives"
```

### Task 3: Add Explicit Target, Affordance, and Request Contracts

**Files:**
- Create: `interaction_target.h`
- Create: `interaction_target.cpp`
- Create: `tests/cpp/test_interaction_target.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `Transform`, `Hand`, and the frozen target/request structs.
- Produces: `TargetRegistry::upsert`, `find`, `reserve`, `release`, `attach`, `hold`, `replace_pose`, `reset`, `validate`, `resolve_single_target`, and `find_affordance`.

- [ ] **Step 1: Write failing registry and resolver tests**

```cpp
TargetRegistry registry;
const TargetHandle handle = registry.upsert(make_target(7, 1));
assert(handle.id == 7 && handle.generation == 1);
assert(registry.resolve_single_target(vec3(), 1.0F) == handle);
assert(registry.reserve(handle, 42));
assert(registry.find(handle)->state == ObjectState::Targeted);
assert(!registry.reserve(handle, 43));
assert(registry.validate(handle, 42));
registry.replace_pose(handle.id, Transform{vec3(4.0F, 0.0F, 0.0F), quat()});
assert(!registry.validate(handle, 42));
assert(!registry.resolve_single_target(vec3(), 1.0F).has_value());
```

Add these exact boundary assertions to the same test:

```cpp
assert(throws_invalid_argument([&] {
    registry.upsert(make_target(0, 1));
}));
assert(throws_invalid_argument([&] {
    registry.upsert(make_target_with_dimensions(8, 1, vec3(0, 1, 1)));
}));
assert(throws_invalid_argument([&] {
    registry.upsert(make_target_with_affordance_ids(9, 1, {3, 3}));
}));

TargetRegistry reset_registry;
const TargetHandle before = reset_registry.upsert(make_target(10, 4));
const TargetHandle after = reset_registry.reset(
    before.id, Transform{vec3(1, 2, 3), quat()});
assert(after == (TargetHandle{10, 5}));
assert(reset_registry.find(after)->state == ObjectState::Free);
assert(reset_registry.find(after)->owner_request == 0);

TargetRegistry ambiguous;
ambiguous.upsert(make_target(11, 1));
ambiguous.upsert(make_target(12, 1));
assert(!ambiguous.resolve_single_target(vec3(), 1.0F).has_value());
```

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_target`

Expected: compilation fails because `interaction_target.h` does not exist.

- [ ] **Step 3: Implement the registry without game-policy leakage**

```cpp
class TargetRegistry {
public:
    TargetHandle upsert(InteractionTarget target);
    const InteractionTarget* find(TargetHandle handle) const;
    InteractionTarget* find(TargetHandle handle);
    std::optional<TargetHandle> resolve_single_target(
        vec3 character_root, float maximum_distance) const;
    bool reserve(TargetHandle handle, uint64_t request_id);
    bool validate(TargetHandle handle, uint64_t request_id) const;
    bool attach(TargetHandle handle, uint64_t request_id);
    bool hold(TargetHandle handle, uint64_t request_id);
    bool release(TargetHandle handle, uint64_t request_id);
    TargetHandle replace_pose(uint64_t id, Transform object_world);
    TargetHandle reset(uint64_t id, Transform object_world);
    const GraspAffordance* find_affordance(
        TargetHandle handle, uint32_t affordance_id) const;
private:
    std::vector<InteractionTarget> targets_;
};
```

`resolve_single_target` returns a handle only when exactly one `Free` target is within `maximum_distance`; it never chooses between multiple candidates. `upsert` rejects ID `0`, generation `0`, non-positive dimensions, empty affordances, and duplicate affordance IDs. `reserve` records the request without changing the generation. `replace_pose` and `reset` return the new handle after incrementing generation and invalidating stale requests; reset additionally restores `Free` state and clears the owner.

- [ ] **Step 4: Run GREEN and the full C++ suite**

Run: `make build/tests/test_interaction_target && build/tests/test_interaction_target && make test-cpp`

Expected: all commands exit 0.

- [ ] **Step 5: Commit target contracts**

```bash
git add interaction_target.h interaction_target.cpp \
  tests/cpp/test_interaction_target.cpp Makefile
git commit -m "feat: define explicit interaction targets"
```

### Task 4: Reproduce the Frozen 71-D Query in C++

**Files:**
- Create: `interaction_features.h`
- Create: `interaction_features.cpp`
- Create: `interaction_query_probe.cpp`
- Create: `tests/cpp/test_interaction_features.cpp`
- Create: `tests/python/test_interaction_query_parity.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `LocomotionSnapshot`, target/affordance, database feature offsets/scales, and FK.
- Produces: `RawQuery build_raw_query(const QueryInput&)`, `NormalizedQuery normalize_query(const RawQuery&, const Features&)`, and `interaction_query_probe <pack> <clip> <frame> --json`.

- [ ] **Step 1: Write failing dimensional and cross-language tests**

The C++ test asserts group boundaries and exact values for identity transforms. The Python test publishes `artifact_fixture()`, invokes the probe at the first `REACH` frame of every clip, and asserts:

```python
self.assertEqual(payload["dimension"], 71)
self.assertLessEqual(payload["max_abs_error"], 2e-4)
self.assertEqual(payload["groups"], [[0, 33], [33, 45], [45, 57], [57, 65], [65, 71]])
```

- [ ] **Step 2: Run and verify RED**

Run: `make interaction_query_probe build/tests/test_interaction_features`

Expected: Make fails because the query source files do not exist.

Run: `python -m unittest tests.python.test_interaction_query_parity -v`

Expected: the probe executable is missing.

- [ ] **Step 3: Port the exact feature construction**

```cpp
struct QueryInput {
    LocomotionSnapshot locomotion;
    Transform grasp_world;
    vec3 grasp_linear_velocity{};
    vec3 grasp_angular_velocity{};
    Transform table_world;
    vec3 table_size{};
    vec3 approach_direction_object{};
    vec3 object_dimensions{};
    Hand hand = Hand::Right;
};

using RawQuery = std::array<float, 71>;
using NormalizedQuery = std::array<float, 71>;
```

`build_raw_query` must port `_raw_clip_features` exactly: FK bones `[7,13,1,16,active_hand]`; root-relative positions and velocities; local root velocity/yaw velocity; future root offsets at 0.32, 0.68, and 1.00 seconds; grasp-relative hand position/orientation/velocities; grasp-relative root position/facing/velocity; grasp height above the table; object-local approach X/Z; and XYZ object dimensions.

Normalize with:

```cpp
for (size_t dimension = 0; dimension < 71; ++dimension) {
    normalized[dimension] =
        (raw[dimension] - features.offsets[dimension]) /
        features.scales[dimension];
}
```

The probe reconstructs a query from an artifact frame and that frame's recorded object, grasp, table, and future roots, then compares it against the serialized normalized feature row.

- [ ] **Step 4: Run focused and full GREEN**

Run: `make interaction_query_probe build/tests/test_interaction_features && build/tests/test_interaction_features`

Expected: C++ test exits 0.

Run: `python -m unittest tests.python.test_interaction_query_parity -v`

Expected: all clip parity cases pass with maximum error no greater than `2e-4`.

Run: `make test-interaction`

Expected: all Python and C++ tests pass.

- [ ] **Step 5: Commit feature parity**

```bash
git add interaction_features.h interaction_features.cpp \
  interaction_query_probe.cpp tests/cpp/test_interaction_features.cpp \
  tests/python/test_interaction_query_parity.py Makefile
git commit -m "feat: reproduce interaction queries in C++"
```

### Task 5: Select One Valid Whole-Clip Entry

**Files:**
- Create: `interaction_matcher.h`
- Create: `interaction_matcher.cpp`
- Create: `tests/cpp/interaction_runtime_fixture.h`
- Create: `tests/cpp/test_interaction_matcher.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: database/features, normalized query, locomotion root, target/affordance, and a support-table blocker.
- Produces: `MatchResult select_whole_clip(const MatchInput&, const MatchConfig&)` with one accepted `MatchCandidate` or one `Reason`.

The shared test header exposes these factories to Tasks 5-11:

```cpp
struct RuntimeFixture {
    Database database{};
    Features features{};
    TargetRegistry registry{};
    LocomotionSnapshot locomotion{};
    Transform original_object_world{};
    PickRequest request{};
};

RuntimeFixture make_runtime_fixture();
TargetRegistry fresh_registry();
PickRequest valid_request();
PickRequest valid_request_for(const TargetRegistry& registry);
MatchInput valid_input();
MatchInput wrong_hand_input();
MatchInput out_of_range_input();
MatchInput excessive_root_input();
MatchInput blocked_table_input();
MatchInput high_cost_input();
RuntimeFixture high_cost_fixture();
RuntimeFixture contact_failure_fixture();
```

- [ ] **Step 1: Write failing selection and rejection tests**

```cpp
const MatchResult accepted = select_whole_clip(valid_input(), MatchConfig{});
assert(accepted.accepted);
assert(accepted.candidate.clip == 1);
assert(accepted.candidate.entry_frame < accepted.candidate.contact_frame);
assert(accepted.candidate.contact_frame < accepted.candidate.lift_frame);
assert(accepted.candidate.lift_frame < accepted.candidate.hold_frame);
assert(accepted.candidate.total_cost <= 9.0F);

assert(select_whole_clip(wrong_hand_input(), {}).reason == Reason::NoCandidate);
assert(select_whole_clip(out_of_range_input(), {}).reason == Reason::OutOfRange);
assert(select_whole_clip(excessive_root_input(), {}).reason == Reason::CorrectionLimit);
assert(select_whole_clip(blocked_table_input(), {}).reason == Reason::BlockedPath);
assert(select_whole_clip(high_cost_input(), {}).reason == Reason::PoorMatch);
```

The shared fixture uses exactly 75 frames per clip. In each clip, local frames
`[0,10)` are `APPROACH`, `[10,25)` are `REACH`, `[25,30)` are `CONTACT`,
`[30,40)` are `LIFT`, and `[40,75)` are `HOLD`. Clip 0 is left-handed and its
root is stationary throughout HOLD, so carry certification rejects it. Clip 1
is right-handed; from HOLD frame 40 through 74 its root, active hand, and object
translate together by `0.40 m` on X while the hand-in-object transform is
constant, so Task 9 can certify it. The object origin rises from `0.75 m` to
`0.95 m` during LIFT in both clips. Table top is `0.70 m`, object dimensions are
`(0.08, 0.20, 0.08)`, and the right-hand affordance is identity at the object
origin.

Set every clip-1 pre-contact normalized row to constants `0.1`, `0.2`, `0.3`,
`0.4`, and `0.5` in the five respective feature groups while `valid_input()`
uses an all-zero normalized query. The selection test therefore also asserts:

```cpp
assert(near(accepted.candidate.group_costs,
            std::array<float, 5>{0.01F, 0.04F, 0.09F, 0.16F, 0.25F}));
assert(std::abs(accepted.candidate.total_cost - (0.80F / 7.0F)) < 1e-6F);
```

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_matcher`

Expected: compilation fails because `interaction_matcher.h` does not exist.

- [ ] **Step 3: Implement hard filters, alignment, cost, and clearance**

Use the frozen `MatchConfig` and these exact public records:

```cpp
struct MatchCandidate {
    int32_t clip = -1;
    int32_t entry_frame = -1;
    int32_t contact_frame = -1;
    int32_t lift_frame = -1;
    int32_t hold_frame = -1;
    Transform scene_from_source{};
    vec3 entry_root_offset{};
    float entry_yaw_offset = 0.0F;
    float total_cost = 0.0F;
    std::array<float, 5> group_costs{};
};

struct MatchInput {
    const Database* database = nullptr;
    const Features* features = nullptr;
    NormalizedQuery query{};
    LocomotionSnapshot locomotion{};
    InteractionTarget target{};
    GraspAffordance affordance{};
    PickRequest request{};
};

struct MatchResult {
    bool accepted = false;
    MatchCandidate candidate{};
    Reason reason = Reason::None;
};

MatchResult select_whole_clip(
    const MatchInput& input, const MatchConfig& config);
```

For each clip, locate the first contact/lift/hold and evaluate its first `REACH` row. Consider earlier `APPROACH` rows, in reverse time order toward that reach, only when no acceptable `REACH` candidate survives across the database. Align the source object at the last pre-contact frame to the target using planar translation and yaw. Reject target distance above `1.00 m`, mapped entry-root error above `0.25 m`, yaw error above `25 degrees`, wrong hand, invalid future range, hand correction above its limits, stale target, or blocked path.

Compute group cost as the mean squared difference over each serialized group and total cost as:

```cpp
const float total =
    (group[0] + group[1] + 2.0F * group[2] +
     2.0F * group[3] + group[4]) / 7.0F;
```

Root clearance samples every selected frame against the table's yaw-oriented XZ rectangle expanded by `0.25 m`. Hand clearance treats each adjacent warped hand pair as a swept sphere/capsule with the affordance `clearance_radius`: transform the segment into table-local space and reject intersection with the table's XYZ box expanded by that radius. Apply the same capsule test to the target object's oriented box for all pre-contact segments except the final segment ending at the stable-contact frame; only that final target intersection is exempt.

- [ ] **Step 4: Run focused and full GREEN**

Run: `make build/tests/test_interaction_matcher && build/tests/test_interaction_matcher && make test-cpp`

Expected: all commands exit 0.

- [ ] **Step 5: Commit whole-clip selection**

```bash
git add interaction_matcher.h interaction_matcher.cpp \
  tests/cpp/interaction_runtime_fixture.h \
  tests/cpp/test_interaction_matcher.cpp Makefile
git commit -m "feat: select bounded pickup clips"
```

### Task 6: Add Clip-Safe Playback and Decaying Entry Alignment

**Files:**
- Create: `interaction_playback.h`
- Create: `interaction_playback.cpp`
- Create: `tests/cpp/test_interaction_playback.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: accepted `MatchCandidate`, current pose, database, and playback `dt`.
- Produces: `SequentialPlayer::start`, `advance`, `sample`, `phase`, `frame`, `at_contact`, `at_hold`, and `finished`.

- [ ] **Step 1: Write failing 25 Hz, interpolation, correction, and range tests**

Advance a synthetic player at 60 Hz and assert source time equals elapsed time,
frame interpolation is continuous, contact order is unchanged, entry correction
begins at its full value and reaches zero at contact, and the player never reads
`range_stop`. `Reason::ClipEnded` remains a defensive runtime reason and is
covered here at the `SequentialPlayer` boundary/range tests; an immutable pack
that passed whole-clip preflight is not truncated later to manufacture this path
in coordinator tests.

```cpp
SequentialPlayer player(database);
player.start(candidate, current_pose, 1.0F);
for (int update = 0; update < 60; ++update) player.advance(1.0F / 60.0F);
assert(std::abs(player.elapsed_seconds() - 1.0F) < 1e-5F);
assert(player.frame() >= candidate.entry_frame);
assert(player.frame() < database.range_stops[candidate.clip]);
assert(length(player.entry_root_correction()) <= 1e-4F);
```

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_playback`

Expected: compilation fails because `interaction_playback.h` does not exist.

- [ ] **Step 3: Implement the clock and corrected pose sampling**

```cpp
class SequentialPlayer {
public:
    explicit SequentialPlayer(const Database& database);
    void start(const MatchCandidate&, const Pose& current, float speed = 1.0F);
    void advance(float dt);
    Pose sample() const;
    Phase phase() const;
    int32_t frame() const;
    bool at_contact() const;
    bool at_hold() const;
    bool finished() const;
    float elapsed_seconds() const;
    vec3 entry_root_correction() const;
private:
    const Database* database_ = nullptr;
    MatchCandidate candidate_{};
    float source_frame_ = 0.0F;
    float elapsed_ = 0.0F;
    float speed_ = 1.0F;
};
```

Use `source_frame += dt * 25.0F * speed`, clamp only to `range_stop - 1`, and interpolate adjacent frames. Apply `scene_from_source` to root position/rotation. Multiply the entry planar/yaw offsets by `1 - smoothstep(entry_frame, contact_frame, source_frame)` so they are full at entry and exactly zero at contact. Reject start speeds outside `[0.85,1.15]`.

- [ ] **Step 4: Run GREEN**

Run: `make build/tests/test_interaction_playback && build/tests/test_interaction_playback && make test-cpp`

Expected: all commands exit 0.

- [ ] **Step 5: Commit sequential playback**

```bash
git add interaction_playback.h interaction_playback.cpp \
  tests/cpp/test_interaction_playback.cpp Makefile
git commit -m "feat: replay pickup clips at 25 Hz"
```

### Task 7: Add Limit-Aware G1 Hand IK

**Files:**
- Create: `g1_arm_joint_metadata.h`
- Create: `resources/generate_g1_arm_joint_metadata.py`
- Create: `interaction_ik.h`
- Create: `interaction_ik.cpp`
- Create: `tests/cpp/test_interaction_ik.cpp`
- Create: `tests/python/test_g1_arm_joint_metadata.py`
- Modify: `vec.h`
- Modify: `quat.h`
- Modify: `Makefile`

**Interfaces:**
- Consumes: canonical `Pose`, `Hand`, target hand `Transform`, and G1 XML only during metadata generation.
- Produces: exact 14-joint metadata and `IKResult solve_hand_ik(Pose&, Hand, Transform, const IKConfig&)`.

- [ ] **Step 1: Write failing metadata and IK tests**

Python regenerates the header into a temporary file and compares it byte-for-byte with the tracked header. C++ tests assert local-rotation decomposition/recomposition, reachable convergence, unreachable rejection, and every solved scalar inside its XML range.

```cpp
IKConfig config;
Pose pose = reachable_arm_pose();
const IKResult result = solve_hand_ik(
    pose, Hand::Right, reachable_hand_target(), config);
assert(result.accepted);
assert(result.position_error_m <= 0.04F);
assert(result.orientation_error_radians <= 0.261799388F);
for (const float angle : result.joint_angles) {
    assert(std::isfinite(angle));
}
Pose unreachable_pose = reachable_arm_pose();
assert(!solve_hand_ik(
    unreachable_pose, Hand::Right, target_offset_by(0.13F), config).accepted);
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.python.test_g1_arm_joint_metadata -v`

Expected: import/file failure because the generator and header do not exist.

Run: `make build/tests/test_interaction_ik`

Expected: compilation fails because `interaction_ik.h` does not exist.

- [ ] **Step 3: Generate exact metadata and implement bounded DLS**

The Python module exposes
`generate_header(g1_xml: pathlib.Path) -> str`, and its CLI is exact:

```bash
python -m resources.generate_g1_arm_joint_metadata \
  --g1-xml "$G1_XML" --output g1_arm_joint_metadata.h
```

The parity test calls `generate_header(Path(os.environ["G1_XML"]))` and compares
the returned UTF-8 text byte-for-byte with the tracked header.

The tracked descriptors are:

```cpp
struct HingeJoint {
    int32_t bone;
    quat rest_rotation;
    vec3 axis;
    float lower;
    float upper;
};

inline constexpr std::array<HingeJoint, 7> kLeftArm = {{
    {17, {0.99026414F, 0.13920102F, -0.0000986868F, -0.0000138722F}, {0,0,-1}, -3.0892F, 2.6704F},
    {18, {0.990268219F,-0.139172031F,0,0}, {1,0,0}, -1.5882F, 2.2515F},
    {19, {1,0,0,0}, {0,1,0}, -2.618F, 2.618F},
    {20, {1,0,0,0}, {0,0,-1}, -1.0472F, 2.0944F},
    {21, {1,0,0,0}, {1,0,0}, -1.97222F, 1.97222F},
    {22, {1,0,0,0}, {0,0,-1}, -1.61443F, 1.61443F},
    {23, {1,0,0,0}, {0,1,0}, -1.61443F, 1.61443F},
}};

inline constexpr std::array<HingeJoint, 7> kRightArm = {{
    {24, {0.99026414F,-0.13920102F, 0.0000986868F,-0.0000138722F}, {0,0,-1}, -3.0892F, 2.6704F},
    {25, {0.990268219F,0.139172031F,0,0}, {1,0,0}, -2.2515F, 1.5882F},
    {26, {1,0,0,0}, {0,1,0}, -2.618F, 2.618F},
    {27, {1,0,0,0}, {0,0,-1}, -1.0472F, 2.0944F},
    {28, {1,0,0,0}, {1,0,0}, -1.97222F, 1.97222F},
    {29, {1,0,0,0}, {0,0,-1}, -1.61443F, 1.61443F},
    {30, {1,0,0,0}, {0,1,0}, -1.61443F, 1.61443F},
}};
```

Because these frozen arrays are `constexpr` and contain the existing `vec3`
and `quat` types, their default and value constructors must also be `constexpr`
under C++17. This is an ABI- and runtime-behavior-preserving qualifier change;
no math operators or functions become part of the compile-time contract.

Expose the solve result and function exactly as:

```cpp
struct IKResult {
    bool accepted = false;
    Reason reason = Reason::None;
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    std::array<float, 7> joint_angles{};
};

IKResult solve_hand_ik(
    Pose& pose,
    Hand hand,
    Transform target_hand_world,
    const IKConfig& config);
```

Use the frozen `IKConfig`: numerical six-dimensional Jacobians over the seven hinge angles, damping `0.05`, finite-difference step `1e-3`, maximum 8 iterations, and per-iteration angle step `0.10 rad`. Position residual is metres; orientation residual is scaled by `0.25 m/rad`. Clamp each trial angle to its joint range. Reject the request before solving above `0.12 m` or `25 degrees`; return the best bounded result and accept only if the final result is at most `0.04 m` / `15 degrees`.

- [ ] **Step 4: Run metadata parity and IK GREEN**

Run with the exact XML:

```bash
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
  python -m unittest tests.python.test_g1_arm_joint_metadata -v
make build/tests/test_interaction_ik
build/tests/test_interaction_ik
```

Expected: Python parity passes; C++ IK test exits 0.

- [ ] **Step 5: Commit arm metadata and IK**

```bash
git add g1_arm_joint_metadata.h resources/generate_g1_arm_joint_metadata.py \
  interaction_ik.h interaction_ik.cpp tests/cpp/test_interaction_ik.cpp \
  tests/python/test_g1_arm_joint_metadata.py Makefile vec.h quat.h
git commit -m "feat: add bounded G1 hand IK"
```

### Task 8: Gate Attachment, Lift, Hold, and Reset

**Files:**
- Create: `interaction_attachment.h`
- Create: `interaction_attachment.cpp`
- Create: `tests/cpp/test_interaction_attachment.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: registry reservation, active-hand world transform, authored `hand_in_object`, stable-contact frame/state, pre-lift object height, and `dt`.
- Produces: `AttachmentController::begin`, `try_contact`, `update`, `reset`, object transform/state, and result/reason.

- [ ] **Step 1: Write failing no-teleport and contact-gate tests**

```cpp
AttachmentController attachment(registry, AttachmentConfig{});
assert(attachment.begin(
    target, request, affordance, target.object_world.position.y));
const Transform original = attachment.object_world();
assert(!attachment.try_contact(position_error_measurement()));
assert(near(attachment.object_world(), original));
assert(attachment.reason() == Reason::ContactPosition);

assert(attachment.try_contact(valid_contact_measurement()));
attachment.update(hand_raised_by(0.16F), 0.5F);
assert(attachment.state() == ObjectState::Attached);
attachment.update(hand_raised_by(0.16F), 0.5F);
assert(attachment.state() == ObjectState::Held);
assert(near(
    attachment.object_world(),
    compose(hand_world, inverse(affordance.hand_in_object))));
```

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_attachment`

Expected: compilation fails because `interaction_attachment.h` does not exist.

- [ ] **Step 3: Implement explicit contact and hold gates**

```cpp
struct ContactMeasurement {
    TargetHandle target{};
    Hand hand = Hand::Right;
    Transform hand_world{};
    float position_error_m = 0.0F;
    float orientation_error_radians = 0.0F;
    bool stable_contact_event = false;
    bool hand_contact = false;
    bool joints_valid = false;
    bool clearance_valid = false;
};

class AttachmentController {
public:
    AttachmentController(
        TargetRegistry& registry,
        AttachmentConfig config = AttachmentConfig{});
    bool begin(
        const InteractionTarget& target,
        const PickRequest& request,
        const GraspAffordance& affordance,
        float pre_lift_object_height);
    bool try_contact(const ContactMeasurement& measurement);
    void update(const ContactMeasurement& measurement, float dt);
    TargetHandle reset(Transform restored_object_world);
    Transform object_world() const;
    ObjectState state() const;
    ResultCode result() const;
    Reason reason() const;
    float held_seconds() const;
private:
    TargetRegistry* registry_ = nullptr;
    AttachmentConfig config_{};
    InteractionTarget target_{};
    PickRequest request_{};
    GraspAffordance affordance_{};
    float pre_lift_object_height_ = 0.0F;
    float held_seconds_ = 0.0F;
    ResultCode result_ = ResultCode::None;
    Reason reason_ = Reason::None;
};
```

`try_contact` requires the stable-contact event, the still-valid reserved handle/request, authored hand, hand contact, `<=0.04 m`, `<=15 degrees`, joint validity, and clearance. Only then set `Attached`. `update` revalidates the same target generation, derives object pose as `hand_world * inverse(hand_in_object)`, requires object-origin height `>= pre_lift_object_height + 0.15`, accumulates one continuous second, resets the timer below the threshold, and enters `Held` at one second. A lost target generation or contact after attachment reports `TargetChanged` or `LostContact` without reattaching. `reset` restores the exact configured target transform and increments its registry generation.

- [ ] **Step 4: Run GREEN**

Run: `make build/tests/test_interaction_attachment && build/tests/test_interaction_attachment && make test-cpp`

Expected: all commands exit 0.

- [ ] **Step 5: Commit attachment behavior**

```bash
git add interaction_attachment.h interaction_attachment.cpp \
  tests/cpp/test_interaction_attachment.cpp Makefile
git commit -m "feat: gate pickup object attachment"
```

### Task 9: Add Recorded and Layered Carrying

**Files:**
- Create: `interaction_carry.h`
- Create: `interaction_carry.cpp`
- Create: `tests/cpp/test_interaction_carry.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: database HOLD ranges, current locomotion snapshot/query, final hold pose, hand target, and IK.
- Produces: `classify_carry_ranges`, `CarryController::start`, `update`, `recorded`, `object_world`, and a carry `Pose`.

- [ ] **Step 1: Write failing classification and fallback tests**

```cpp
const CarryRanges ranges = classify_carry_ranges(database);
assert(ranges.recorded.size() == 1);
assert(ranges.rejected.size() == 1);

CarryController fallback(database, features, no_recorded_ranges());
fallback.start(
    hold_pose, Hand::Right, affordance, object_world_from_hold_pose());
const Pose output = fallback.update(locomotion_snapshot, 1.0F / 60.0F);
assert(!fallback.recorded());
assert(near(output.rotations[g1_skeleton::RightShoulderPitch],
            hold_pose.rotations[g1_skeleton::RightShoulderPitch]));
assert(near(output.rotations[g1_skeleton::LeftHipPitch],
            locomotion_snapshot.pose.rotations[g1_skeleton::LeftHipPitch]));
assert(hand_error(output, fallback.object_world()) <= 0.04F);
```

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_carry`

Expected: compilation fails because `interaction_carry.h` does not exist.

- [ ] **Step 3: Implement certification, search, and fallback mask**

Expose these exact carry records and controller boundary:

```cpp
struct CarryRange {
    int32_t clip = -1;
    int32_t start_frame = -1;
    int32_t stop_frame = -1;
    Hand hand = Hand::Right;
};

struct CarryRanges {
    std::vector<CarryRange> recorded;
    std::vector<int32_t> rejected;
};

CarryRanges classify_carry_ranges(
    const Database& database,
    const CarryConfig& config = CarryConfig{});

class CarryController {
public:
    CarryController(
        const Database& database,
        const Features& features,
        CarryRanges ranges,
        CarryConfig config = CarryConfig{},
        IKConfig ik_config = IKConfig{});
    void start(
        const Pose& final_hold_pose,
        Hand hand,
        const GraspAffordance& affordance,
        Transform object_world);
    Pose update(const LocomotionSnapshot& locomotion, float dt);
    bool recorded() const;
    Transform object_world() const;
private:
    const Database* database_ = nullptr;
    const Features* features_ = nullptr;
    CarryRanges ranges_{};
    CarryConfig config_{};
    IKConfig ik_config_{};
    Pose final_hold_pose_{};
    GraspAffordance affordance_{};
    Transform object_world_{};
    Hand hand_ = Hand::Right;
    bool recorded_ = false;
    float search_seconds_ = 0.0F;
};
```

Certify a contiguous HOLD subrange only when it has at least 25 frames, continuous active contact, hand-in-object drift `<=0.02 m` / `10 degrees`, planar root displacement `>=0.30 m`, and average speed `>=0.20 m/s`.

When recorded ranges exist, search only those rows every `0.10 s` with pose/trajectory group weights and advance sequentially between searches. Otherwise construct the fallback by copying locomotion, replacing the seven active-arm rotations with the hold pose, nlerping Spine/Spine1/Spine2 25% toward hold, nlerping the inactive arm 35% toward hold, leaving every lower-body value unchanged, and running bounded IK against the carried grasp.

```cpp
const float spine_weight = 0.25F;
const float inactive_arm_weight = 0.35F;
for (int32_t bone : active_arm_bones(hand)) {
    output.rotations[bone] = hold.rotations[bone];
}
for (int32_t bone : {14, 15, 16}) {
    output.rotations[bone] = quat_nlerp_shortest(
        locomotion.pose.rotations[bone], hold.rotations[bone], spine_weight);
}
```

- [ ] **Step 4: Run GREEN**

Run: `make build/tests/test_interaction_carry && build/tests/test_interaction_carry && make test-cpp`

Expected: all commands exit 0.

- [ ] **Step 5: Commit carry behavior**

```bash
git add interaction_carry.h interaction_carry.cpp \
  tests/cpp/test_interaction_carry.cpp Makefile
git commit -m "feat: add controllable carrying motion"
```

### Task 10: Integrate the Coordinator and Headless End-to-End Probe

**Files:**
- Create: `interaction_runtime.h`
- Create: `interaction_runtime.cpp`
- Create: `interaction_runtime_probe.cpp`
- Create: `tests/cpp/test_interaction_runtime.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: registry, matcher, player, IK, attachment, carry, input snapshot, and interaction pack.
- Produces: `InteractionRuntime::disabled`, `state`, `update`, `diagnostics`, `interaction_runtime_probe <pack> --json`, and the full frozen state/result contract.

- [ ] **Step 1: Write failing state-machine and failure-path tests**

Test exact transitions and forbidden side effects:

```cpp
InteractionRuntime runtime(database, features, registry, RuntimeConfig{});
assert(runtime.state() == RuntimeState::Locomotion);
RuntimeOutput output = runtime.update(interact_input(valid_request()));
assert(output.diagnostics.state == RuntimeState::Preflight);
output = runtime.update(idle_input());
assert(output.diagnostics.state == RuntimeState::Align);
assert(output.diagnostics.result == ResultCode::Accepted);
advance_until(runtime, RuntimeState::PickupReplay);
advance_until(runtime, RuntimeState::Hold);
advance_until(runtime, RuntimeState::Carry);
assert(runtime.diagnostics().result == ResultCode::Succeeded);
assert(runtime.diagnostics().attached);

TargetRegistry cancelled_registry = fresh_registry();
InteractionRuntime cancelled(
    database, features, cancelled_registry, RuntimeConfig{});
cancelled.update(interact_input(valid_request_for(cancelled_registry)));
cancelled.update(idle_input());
output = cancelled.update(cancel_input());
assert(output.diagnostics.state == RuntimeState::Locomotion);
assert(output.diagnostics.result == ResultCode::Cancelled);
assert(!output.diagnostics.attached);
```

Add these exact path assertions, using copies of the shared fixture whose names
state the single mutation they apply:

```cpp
RuntimeDiagnostics run_rejected(RuntimeFixture fixture);
RuntimeDiagnostics run_post_commit_failure(RuntimeFixture fixture);

InteractionRuntime missing = InteractionRuntime::disabled(
    Reason::PackUnavailable);
output = missing.update(interact_input(valid_request()));
assert(output.diagnostics.state == RuntimeState::Disabled);
assert(output.diagnostics.reason == Reason::PackUnavailable);
assert(!output.owns_pose && !output.suppress_steering);

TargetRegistry stale_registry = fresh_registry();
const PickRequest stale_request = valid_request_for(stale_registry);
InteractionRuntime stale(database, features, stale_registry, RuntimeConfig{});
stale.update(interact_input(stale_request));
stale_registry.replace_pose(
    stale_request.target.id, Transform{vec3(0, 0.75F, 3), quat()});
output = stale.update(idle_input());
assert(output.diagnostics.state == RuntimeState::Locomotion);
assert(output.diagnostics.result == ResultCode::Rejected);
assert(output.diagnostics.reason == Reason::TargetChanged);
assert(!output.diagnostics.attached);

assert(run_rejected(high_cost_fixture()).reason == Reason::PoorMatch);
assert(run_post_commit_failure(contact_failure_fixture()).reason ==
       Reason::ContactPosition);
assert(!run_post_commit_failure(contact_failure_fixture()).attached);

const uint32_t generation_before_reset =
    runtime.diagnostics().target.generation;
output = runtime.update(reset_input());
assert(output.diagnostics.state == RuntimeState::Locomotion);
assert(output.diagnostics.result == ResultCode::Reset);
assert(registry.resolve_single_target(vec3(0, 0, 3), 1.0F)->generation ==
       generation_before_reset + 1);
assert(!output.owns_pose);
```

`high_cost_fixture()` adds `10.0F` to every serialized normalized candidate
dimension while leaving the runtime query unchanged;
`contact_failure_fixture()` offsets the corrected hand by `0.05 m` only at the
stable-contact frame. `run_post_commit_failure` must first advance beyond
commitment and then assert the source frame reaches `range_stop - 1` before
Locomotion returns.

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_runtime interaction_runtime_probe`

Expected: compilation fails because the runtime files do not exist.

- [ ] **Step 3: Implement the coordinator with one transition owner**

```cpp
class InteractionRuntime {
public:
    InteractionRuntime(
        const Database&, const Features&, TargetRegistry&, RuntimeConfig);
    static InteractionRuntime disabled(Reason reason);
    RuntimeState state() const;
    const RuntimeDiagnostics& diagnostics() const;
    RuntimeOutput update(const RuntimeInput& input);
private:
    InteractionRuntime() = default;
    const Database* database_ = nullptr;
    const Features* features_ = nullptr;
    TargetRegistry* registry_ = nullptr;
    RuntimeConfig config_{};
    std::optional<PickRequest> request_{};
    RuntimeState state_ = RuntimeState::Locomotion;
    RuntimeDiagnostics diagnostics_{};
};
```

Only the coordinator changes runtime state. An Interact edge enters `Preflight`, and `Preflight` remains observable for that update; validation/selection occurs on the next update so deterministic logs contain the state. The triggering input must carry an explicit `PickRequest`; an Interact edge without one reports `TargetUnavailable`. `Preflight` stores, validates, and reserves exactly that target generation, looks up exactly that affordance, builds the query, and selects without calling `resolve_single_target` or substituting another target. `Align` begins contiguous playback, permits cancel before commitment, applies the 0.25-second entry blend, and lasts no more than 1.00 second. Commitment occurs after `min(0.50 seconds, time_to_contact)`; steering is then suppressed and the state becomes `PickupReplay` without resetting the source clock. `PickupReplay` continues the same clip, runs IK, and calls the contact gate. `Hold` maintains the final valid pose until the one-second attachment timer succeeds. `Carry` delegates to `CarryController`; reset restores the registry and locomotion. A pre-commit failure releases the reservation and returns immediately to locomotion. A post-commit failure records one `Reason`, completes the selected clip remainder without attachment, then returns to locomotion. No failed gate may attach later in the same request.

The probe creates the demo target from clip 0, resolves its sole target/affordance outside the runtime, submits that explicit `PickRequest`, starts its synthetic character at the selected entry root, pulses Interact, advances at 60 Hz, then supplies forward locomotion in Carry. Emit compact sorted JSON with state sequence, selected clip, attach frame, held time, carry mode, carried root displacement, final result, and reason.

- [ ] **Step 4: Run focused and full GREEN**

Run: `make build/tests/test_interaction_runtime interaction_runtime_probe && build/tests/test_interaction_runtime`

Expected: C++ test exits 0.

Run: `./interaction_runtime_probe /tmp/g1_interaction_5 --json`

Expected: JSON contains `"final_result":"Succeeded"`, `"attached":true`, and `"carry_root_displacement_m"` greater than zero.

Run: `make test-interaction`

Expected: all Python and C++ tests pass.

- [ ] **Step 5: Commit the coordinator**

```bash
git add interaction_runtime.h interaction_runtime.cpp \
  interaction_runtime_probe.cpp tests/cpp/test_interaction_runtime.cpp Makefile
git commit -m "feat: coordinate playable pickup state"
```

### Task 11: Wire the Runtime into the Existing Controller

**Files:**
- Create: `interaction_controller_adapter.h`
- Create: `interaction_controller_adapter.cpp`
- Create: `interaction_debug_draw.h`
- Create: `tests/cpp/test_interaction_controller_adapter.cpp`
- Modify: `controller.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: existing controller pose/trajectory arrays, key/gamepad edges, runtime output, and Raylib drawing.
- Produces: unchanged idle locomotion, playable inputs, one target/table scene, runtime pose handoff, target/object drawing, and diagnostics.

- [ ] **Step 1: Write the failing no-interaction adapter test**

```cpp
ControllerInteractionAdapter adapter;
const Pose original = controller_pose_fixture();
const RuntimeOutput idle{};
const Pose output = adapter.apply(original, idle, 1.0F / 60.0F);
assert(equal_pose_bits(output, original));

RuntimeOutput pickup;
pickup.owns_pose = true;
pickup.pose = interaction_pose_fixture();
const Pose first = adapter.apply(original, pickup, 1.0F / 60.0F);
assert(!equal_pose_bits(first, original));
assert(!equal_pose_bits(first, pickup.pose));
const Pose settled = apply_for_seconds(adapter, original, pickup, 0.25F);
assert(near_pose(settled, pickup.pose, 1e-4F));
```

- [ ] **Step 2: Run and verify RED**

Run: `make build/tests/test_interaction_controller_adapter`

Expected: compilation fails because the adapter does not exist.

- [ ] **Step 3: Implement the pure adapter and thin controller wiring**

Expose the pure adapter as:

```cpp
class ControllerInteractionAdapter {
public:
    Pose apply(
        const Pose& locomotion_pose,
        const RuntimeOutput& runtime_output,
        float dt);
    void reset();
private:
    bool owned_last_update_ = false;
    float blend_seconds_ = 0.0F;
    Pose blend_source_{};
};
```

The adapter returns its input bit-for-bit when `owns_pose == false`. On ownership transitions it captures the current pose and blends positions with lerp and rotations with shortest nlerp over exactly `0.25 s`; contacts switch at blend midpoint.

In `controller.cpp`:

1. Load `interaction_database.bin` and `interaction_features.bin` from `MM_INTERACTION_PACK` or `./resources/g1_interaction`, catching `FormatError` and constructing Disabled runtime.
2. Construct the demo target from clip 0, translating its table XZ centre to `(0,3)` while preserving source height and yaw-relative object placement.
3. Read Interact on `IsKeyPressed(KEY_F)` / `GAMEPAD_BUTTON_RIGHT_FACE_LEFT`, cancel on `KEY_X` / right-face-up, and reset on `KEY_R` / right-face-right.
4. On Interact, call the one-object resolver outside `InteractionRuntime`; if it returns a handle with exactly one authored affordance, submit `PickRequest{handle, affordance.id, next_request_id++}`. Submit an empty optional on resolution failure so the runtime reports `TargetUnavailable`. This is the only policy replaced by a future camera/raycast/UI/VLM selector.
5. Zero movement input while the prior runtime output has `suppress_steering`.
6. Build a `LocomotionSnapshot` after the ordinary locomotion pose update using trajectory samples at controller indices 1, 2, and 3.
7. Call runtime update; when it owns the pose, apply the adapter before existing foot IK and copy its two foot contacts into the controller contact inputs.
8. Synchronize `simulation_position` and `simulation_rotation` to the owned root so the background locomotion handoff cannot diverge.
9. Draw table/object primitives, supported root range, the three-point predicted local trajectory, target highlight, affordance/approach axes, active-hand and corrected-hand targets, requested/applied correction vectors, and text diagnostics via `interaction_debug_draw.h`.

Use this desktop source list so probe mains cannot leak into the controller:

```make
INTERACTION_SOURCES := interaction_pose.cpp interaction_target.cpp \
  interaction_features.cpp interaction_matcher.cpp interaction_playback.cpp \
  interaction_ik.cpp interaction_attachment.cpp interaction_carry.cpp \
  interaction_runtime.cpp interaction_controller_adapter.cpp
SOURCE := controller.cpp $(INTERACTION_SOURCES)
```

- [ ] **Step 4: Run adapter, controller, and regression GREEN**

Run: `make build/tests/test_interaction_controller_adapter && build/tests/test_interaction_controller_adapter`

Expected: adapter test exits 0.

Run: `make controller`

Expected: controller links against the pinned local Raylib without warnings.

Run: `make test-interaction`

Expected: all headless tests pass.

- [ ] **Step 5: Commit controller integration**

```bash
git add interaction_controller_adapter.h interaction_controller_adapter.cpp \
  interaction_debug_draw.h tests/cpp/test_interaction_controller_adapter.cpp \
  controller.cpp Makefile
git commit -m "feat: play pickups in the Raylib controller"
```

### Task 12: Build a Real Demo Pack and Pass the Playable Gate

**Files:**
- Create: `tests/python/test_playable_interaction_evidence.py`
- Modify: `controller.cpp`
- Modify: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: local GRAIL pickup root, G1 XML, pinned controller, runtime probe, and `DISPLAY`.
- Produces: `make demo-interaction-pack`, `make gate-playable-interaction`, `MM_INTERACTION_AUTODEMO`, deterministic JSONL evidence, a screenshot, and operator instructions.

- [ ] **Step 1: Write the failing evidence validator**

```python
@unittest.skipUnless(
    os.environ.get("PLAYABLE_LOG") and
    os.environ.get("PLAYABLE_SCREENSHOT"),
    "playable evidence paths not set",
)
class PlayableEvidenceTests(unittest.TestCase):
    def test_state_order_attachment_and_carry(self):
        records = load_jsonl(Path(os.environ["PLAYABLE_LOG"]))
        states = collapse(record["state"] for record in records)
        self.assertEqual(states[:7], [
            "Locomotion", "Preflight", "Align", "PickupReplay",
            "Hold", "Carry", "Locomotion",
        ])
        self.assertTrue(any(record["attached"] for record in records))
        carry = [record for record in records if record["state"] == "Carry"]
        self.assertGreater(carry[-1]["root_displacement_m"], 0.20)
        self.assertIn(carry[-1]["carry_mode"], {"recorded", "layered"})
        screenshot = Path(os.environ["PLAYABLE_SCREENSHOT"])
        self.assertGreater(screenshot.stat().st_size, 10_000)
```

- [ ] **Step 2: Run and verify RED**

Run with nonexistent evidence paths:

```bash
PLAYABLE_LOG=/tmp/missing-playable.jsonl \
PLAYABLE_SCREENSHOT=/tmp/missing-playable.png \
python -m unittest tests.python.test_playable_interaction_evidence -v
```

Expected: failure because the evidence files do not exist.

- [ ] **Step 3: Add reproducible pack, auto-demo, and gate targets**

`make demo-interaction-pack` runs:

```make
GRAIL_ROOT ?= /home/ubuntu/datasets/GRAIL/data/pickup_table
G1_XML ?= /home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml
DEMO_INTERACTION_LIMIT ?= 5
```

```make
.PHONY: demo-interaction-pack
demo-interaction-pack:
	python -m resources.build_g1_interaction_database \
	  --source-root "$(GRAIL_ROOT)" --g1-xml "$(G1_XML)" \
	  --output resources/g1_interaction \
	  --limit "$(DEMO_INTERACTION_LIMIT)" --heldout-count 1
	python -m resources.validate_g1_interaction_database \
	  --input resources/g1_interaction
```

The artifact publisher already replaces an existing output transactionally; do
not add an unsupported force flag or delete the prior pack in the Make recipe.

Add `MM_INTERACTION_AUTODEMO=1`: place the character at the accepted entry, pulse Interact at frame 30, advance until Carry, command forward movement for 150 frames, pulse Reset, log one compact JSON record per frame to `MM_INTERACTION_LOG`, call `TakeScreenshot` at the last Carry frame using `MM_INTERACTION_SCREENSHOT`, and exit after reset. Normal input behavior is unchanged when the variable is absent.

`make gate-playable-interaction` uses this exact orchestration; normal Python
discovery skips the evidence class until the two `PLAYABLE_*` variables are set:

```make
.PHONY: gate-playable-interaction
gate-playable-interaction: bootstrap-raylib test-interaction \
  demo-interaction-pack interaction_probe interaction_query_probe \
  interaction_runtime_probe controller
	mkdir -p playable-evidence
	./interaction_probe resources/g1_interaction --json
	./interaction_runtime_probe resources/g1_interaction --json
	MM_INTERACTION_AUTODEMO=1 \
	MM_INTERACTION_PACK=resources/g1_interaction \
	MM_INTERACTION_LOG=playable-evidence/pickup.jsonl \
	MM_INTERACTION_SCREENSHOT=playable-evidence/pickup.png \
	./controller
	PLAYABLE_LOG=playable-evidence/pickup.jsonl \
	PLAYABLE_SCREENSHOT=playable-evidence/pickup.png \
	python -m unittest \
	  tests.python.test_playable_interaction_evidence -v
```

The query-parity test inside `test-interaction` executes
`interaction_query_probe` against every fixture clip. The gate writes only
beneath ignored `playable-evidence/` and `resources/g1_interaction/`.

- [ ] **Step 4: Run the complete gate and manually verify control**

Run:

```bash
PATH=$PWD/.venv/bin:/home/ubuntu/miniconda3/envs/diffsim/bin:$PATH \
GRAIL_ROOT=/home/ubuntu/datasets/GRAIL/data/pickup_table \
G1_XML=/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml \
DISPLAY=${DISPLAY:-:1} \
make gate-playable-interaction
```

Expected: all tests and probes pass; auto-demo exits 0; evidence validation passes; `playable-evidence/pickup.png` shows the attached object in Carry.

Run manually: `MM_INTERACTION_PACK=resources/g1_interaction ./controller`

Expected: locomotion remains controllable; `F` triggers pickup only near the highlighted target; successful hold returns movement with recorded or layered carrying; `R` restores the target; invalid attempts report a reason without moving the object.

- [ ] **Step 5: Document exact evidence and scope**

README must contain dependency bootstrap, pack build, gate, manual controls, artifact/environment overrides, the observed carry mode, screenshot/log locations, and this scope sentence:

`The playable baseline selects one contiguous pickup for one authored tabletop target; staged interaction matching, multi-object selection, placement, shelves, and articulated doors/drawers are follow-on work.`

- [ ] **Step 6: Commit the playable gate**

```bash
git add tests/python/test_playable_interaction_evidence.py \
  controller.cpp Makefile README.md
git commit -m "test: pass playable tabletop pickup gate"
```

## Final Verification and Review

After all 12 task reviews are clean:

1. Run `make gate-playable-interaction` from a clean tracked worktree.
2. Run `git diff --check cbe90b7..HEAD`.
3. Confirm `git status --short` has no tracked changes.
4. Generate a whole-branch review package from `cbe90b7..HEAD`.
5. Dispatch a fresh whole-branch reviewer with the approved data and playable specs, all task reports, and the complete diff package.
6. Fix every Critical/Important finding in one fix wave and re-run the affected tests plus the complete playable gate.
7. Use `superpowers:finishing-a-development-branch` for the final handoff; do not merge into the terrain-aware branch.
