# Native G1 Episode Carry Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the playable pickup episode use one native 31-bone G1 motion-matching pose throughout, correct camera-relative A/D, and preserve animated G1 locomotion while carrying and placing.

**Architecture:** The viewer loads the compact G1 walking database from the episode pack and renders only `EpisodeOutput::pose`. A pure input helper produces camera-relative commands. During carry, the walking matcher owns root, legs, contacts, and the inactive arm; the final reach pose supplies a grasping-arm/spine posture that the existing bounded posture-aware IK solves against an object transform carried relative to the live root.

**Tech Stack:** C++17, existing database motion matcher, existing G1 pose/attachment/posture-IK modules, Raylib viewer, Make-based C++ tests, Python `unittest` source-contract test.

## Global Constraints

- Use `<episode-pack>/walking_database.bin`; do not load `resources/database.bin` in the playable viewer.
- Every playable pose must use the exact 31-bone `g1_skeleton::kParents` hierarchy.
- `EpisodeOutput::pose` is the sole visual and interaction pose in every state.
- Motion data owns root translation; do not apply synthetic root translation or speed warping.
- The selected grasp and frozen `hand_in_object` transform remain authoritative after contact.
- Preserve the existing reach search, collision filtering, attachment state machine, furniture scene, and reversible placement flow.
- Do not load or render terrain in this repair.
- Do not commit generated `build/g1-reaches/` data or root-level viewer/probe binaries.

---

### Task 1: Correct Camera-Relative Controls

**Files:**
- Create: `episode_controls.h`
- Create: `tests/cpp/test_episode_controls.cpp`
- Modify: `g1_interaction_episode_viewer.cpp:109-141`
- Modify: `Makefile:288-289,635-643`

**Interfaces:**
- Consumes: `episode::LocomotionCommand`, `vec3`, `quat`
- Produces: `episode::DigitalLocomotionInput` and `episode::camera_relative_command(vec3, quat, const DigitalLocomotionInput&, float) -> LocomotionCommand`

- [ ] **Step 1: Write the failing control-direction test**

Add `tests/cpp/test_episode_controls.cpp`:

```cpp
#include "episode_controls.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        const vec3 camera_forward = normalize(vec3(0.6F, 0.0F, 0.8F));
        const vec3 screen_right = normalize(cross(
            camera_forward, vec3(0.0F, 1.0F, 0.0F)));
        const quat heading{};
        const auto right = episode::camera_relative_command(
            camera_forward, heading, {false, false, false, true}, 0.22F);
        const auto left = episode::camera_relative_command(
            camera_forward, heading, {false, false, true, false}, 0.22F);
        const auto diagonal = episode::camera_relative_command(
            camera_forward, heading, {true, false, false, true}, 0.22F);
        require(dot(right.desired_velocity_world, screen_right) > 0.21F,
                "D did not move screen-right");
        require(dot(left.desired_velocity_world, screen_right) < -0.21F,
                "A did not move screen-left");
        require(std::abs(length(diagonal.desired_velocity_world) - 0.22F) <
                    1.0e-5F,
                "diagonal command was not normalized");
        const auto idle = episode::camera_relative_command(
            camera_forward, heading, {}, 0.22F);
        require(length(idle.desired_velocity_world) == 0.0F,
                "idle command moved");
        std::cout << "episode controls PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "episode controls FAILED: " << error.what() << '\n';
        return 1;
    }
}
```

Register `$(CPP_TEST_DIR)/test_episode_controls` in `CPP_TEST_BINS` and add a compile rule using `tests/cpp/test_episode_controls.cpp`.

- [ ] **Step 2: Run the test and verify the helper is absent**

Run:

```bash
make build/tests/test_episode_controls
```

Expected: compilation fails because `episode_controls.h` does not exist.

- [ ] **Step 3: Implement the pure helper and use it in the viewer**

Add `episode_controls.h`:

```cpp
#pragma once

#include "g1_flat_motion_matcher.h"

#include <cmath>
#include <stdexcept>

namespace episode {

struct DigitalLocomotionInput {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
};

inline LocomotionCommand camera_relative_command(
    vec3 camera_forward,
    quat current_heading,
    const DigitalLocomotionInput& input,
    float speed) {
    camera_forward.y = 0.0F;
    if (length(camera_forward) < 1.0e-5F) {
        camera_forward = vec3(0.0F, 0.0F, 1.0F);
    }
    camera_forward = normalize(camera_forward);
    const vec3 screen_right = normalize(cross(
        camera_forward, vec3(0.0F, 1.0F, 0.0F)));
    vec3 direction{};
    if (input.forward) direction = direction + camera_forward;
    if (input.backward) direction = direction - camera_forward;
    if (input.right) direction = direction + screen_right;
    if (input.left) direction = direction - screen_right;
    if (length(direction) < 1.0e-5F) {
        return {vec3(), current_heading};
    }
    direction = normalize(direction);
    const float yaw = std::atan2(direction.x, direction.z);
    return {
        direction * speed,
        quat_from_angle_axis(yaw, vec3(0.0F, 1.0F, 0.0F)),
    };
}

}  // namespace episode
```

In `g1_interaction_episode_viewer.cpp`, include `episode_controls.h` and replace the input arithmetic with:

```cpp
return episode::camera_relative_command(
    forward,
    pose.rotations[g1_skeleton::Simulation],
    {
        IsKeyDown(KEY_W),
        IsKeyDown(KEY_S),
        IsKeyDown(KEY_A),
        IsKeyDown(KEY_D),
    },
    0.22F);
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
make build/tests/test_episode_controls
./build/tests/test_episode_controls
python3 -m unittest tests.python.test_g1_interaction_episode_viewer -v
```

Expected: all control and viewer contract tests pass.

- [ ] **Step 5: Commit the control fix**

```bash
git add episode_controls.h tests/cpp/test_episode_controls.cpp \
  g1_interaction_episode_viewer.cpp Makefile
git commit -m "fix: correct episode camera-relative controls"
```

---

### Task 2: Enforce One Native G1 Pose

**Files:**
- Modify: `g1_flat_motion_matcher.h:29-52`
- Modify: `g1_flat_motion_matcher.cpp:359-397`
- Modify: `interaction_episode.cpp:83-96,578-585`
- Modify: `g1_interaction_episode_viewer.cpp:405-412,729-803`
- Modify: `tests/cpp/test_g1_flat_motion_matcher.cpp`
- Modify: `tests/python/test_g1_interaction_episode_viewer.py`

**Interfaces:**
- Consumes: the existing `FlatMotionMatcher::uses_flat_controller_skeleton_`
- Produces: `FlatMotionMatcher::uses_native_g1() const -> bool`
- Guarantees: construction of `InteractionEpisode` rejects a 23-bone walking database

- [ ] **Step 1: Write failing native-G1 contract tests**

In `tests/cpp/test_g1_flat_motion_matcher.cpp`, add:

```cpp
void test_native_g1_database_publishes_only_g1() {
    const std::filesystem::path walking(
        "build/g1-episode/walking_database.bin");
    episode::FlatMotionMatcher matcher(walking);
    require(matcher.uses_native_g1(),
            "episode walking database was not recognized as native G1");
    require(!matcher.flat_skeleton().valid,
            "native G1 matcher exposed a second flat skeleton");
    for (int tick = 0; tick < 50; ++tick) {
        matcher.update(
            {vec3(0.0F, 0.0F, 0.22F), quat()},
            1.0F / 25.0F);
        require(!matcher.flat_skeleton().valid,
                "native G1 update exposed a flat skeleton");
        require(finite_pose(matcher.snapshot().pose),
                "native G1 update produced a non-finite pose");
    }
}
```

Call it from `main()`.

In `tests/python/test_g1_interaction_episode_viewer.py`, add:

```python
def test_viewer_uses_episode_pack_g1_walking_and_one_pose(self) -> None:
    source = self.source()
    self.assertIn(
        'options.episode_pack / "walking_database.bin"', source
    )
    self.assertNotIn('"resources/database.bin"', source)
    self.assertNotIn("draw_flat_bones(", source)
    self.assertNotIn("flat_visible", source)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
make build/tests/test_g1_flat_motion_matcher
./build/tests/test_g1_flat_motion_matcher
python3 -m unittest \
  tests.python.test_g1_interaction_episode_viewer -v
```

Expected: C++ compilation fails because `uses_native_g1()` is absent; Python fails on the old LAFAN path and flat rendering branch.

- [ ] **Step 3: Expose and enforce the native-G1 contract**

Add to `FlatMotionMatcher`:

```cpp
bool uses_native_g1() const;
```

Implement:

```cpp
bool FlatMotionMatcher::uses_native_g1() const {
    return !uses_flat_controller_skeleton_;
}
```

In the `InteractionEpisode` constructor body, reject mixed topology:

```cpp
if (!matcher_.uses_native_g1()) {
    throw std::invalid_argument(
        "playable episode walking database must use native 31-bone G1");
}
```

Make `publish()` always invalidate the obsolete secondary pose:

```cpp
output_.flat_locomotion = FlatSkeletonWorldPose{};
```

- [ ] **Step 4: Make the viewer load and render only native G1**

Set:

```cpp
const std::filesystem::path walking_database =
    options.episode_pack / "walking_database.bin";
```

Remove `draw_flat_bones`, `flat_visible`, and every flat/G1 render branch. Update the mesh from `world` whenever enabled, then render:

```cpp
if (show_mesh && mesh.loaded &&
    !::g1_mesh_renderer_update(
        mesh,
        slice1d<vec3>(
            static_cast<int>(g1_skeleton::BoneCount),
            world.positions.data()),
        slice1d<quat>(
            static_cast<int>(g1_skeleton::BoneCount),
            world.rotations.data()),
        mesh_error.data(),
        static_cast<int>(mesh_error.size()))) {
    show_mesh = false;
    show_bones = true;
}
if (show_mesh && mesh.loaded) ::g1_mesh_renderer_draw(mesh);
if (show_bones) draw_bones(world);
```

- [ ] **Step 5: Run focused native-G1 tests**

Run:

```bash
make build/tests/test_g1_flat_motion_matcher g1_interaction_episode_viewer
./build/tests/test_g1_flat_motion_matcher
python3 -m unittest \
  tests.python.test_g1_interaction_episode_viewer -v
```

Expected: all tests pass and the viewer builds.

- [ ] **Step 6: Commit the single-skeleton repair**

```bash
git add g1_flat_motion_matcher.h g1_flat_motion_matcher.cpp \
  interaction_episode.cpp g1_interaction_episode_viewer.cpp \
  tests/cpp/test_g1_flat_motion_matcher.cpp \
  tests/python/test_g1_interaction_episode_viewer.py
git commit -m "fix: keep playable episode on native G1"
```

---

### Task 3: Layer Carry Over Native G1 Walking

**Files:**
- Create: `episode_layered_carry.h`
- Create: `episode_layered_carry.cpp`
- Create: `tests/cpp/test_episode_layered_carry.cpp`
- Modify: `interaction_episode.h:1-145`
- Modify: `interaction_episode.cpp:83-96,373-405,409-455,548-629`
- Modify: `Makefile:175-202,288-289,643-660`

**Interfaces:**
- Consumes: native G1 `interaction::LocomotionSnapshot`, final reach contact pose, selected `interaction::Hand`, frozen `hand_in_object`, and contact object transform
- Produces: `episode::LayeredCarry::start(...)`, `LayeredCarry::update(...) -> const interaction::Pose&`, `LayeredCarry::object_world() const -> Transform`, and `LayeredCarry::last_solve_accepted() const -> bool`

- [ ] **Step 1: Write the failing bilateral layered-carry test**

Create `tests/cpp/test_episode_layered_carry.cpp`. Load the native walking matcher, use its first pose as the hold pose, derive an object transform from each wrist, start a `LayeredCarry`, then drive 150 ticks:

```cpp
void test_hand(interaction::Hand hand) {
    episode::FlatMotionMatcher matcher(
        "build/g1-episode/walking_database.bin");
    const interaction::Pose hold = matcher.snapshot().pose;
    const interaction::WorldPose hold_world =
        interaction::world_pose(hold);
    const size_t wrist = hand == interaction::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
    const interaction::Transform hand_world{
        hold_world.positions[wrist], hold_world.rotations[wrist]};
    const interaction::Transform object_world{
        hand_world.position + vec3(0.0F, -0.05F, 0.0F),
        hand_world.rotation};
    const interaction::Transform hand_in_object =
        interaction::compose(
            interaction::inverse(object_world), hand_world);

    episode::LayeredCarry carry;
    carry.start(hold, hand, hand_in_object, object_world);
    const vec3 start_root = hold.positions[g1_skeleton::Simulation];
    float maximum_knee_change = 0.0F;
    float maximum_inactive_arm_change = 0.0F;
    interaction::Pose previous = hold;
    for (int tick = 0; tick < 150; ++tick) {
        matcher.update(
            {vec3(0.0F, 0.0F, 0.22F), quat()},
            1.0F / 25.0F);
        const interaction::Pose& pose =
            carry.update(matcher.snapshot());
        maximum_knee_change = std::max(
            maximum_knee_change,
            quat_angle_between(
                pose.rotations[g1_skeleton::LeftKnee],
                previous.rotations[g1_skeleton::LeftKnee]));
        const size_t inactive_shoulder =
            hand == interaction::Hand::Left
                ? g1_skeleton::RightShoulderPitch
                : g1_skeleton::LeftShoulderPitch;
        maximum_inactive_arm_change = std::max(
            maximum_inactive_arm_change,
            quat_angle_between(
                pose.rotations[inactive_shoulder],
                previous.rotations[inactive_shoulder]));
        const interaction::WorldPose world =
            interaction::world_pose(pose);
        const interaction::Transform solved_hand{
            world.positions[wrist], world.rotations[wrist]};
        const interaction::Transform published_object =
            interaction::compose(
                solved_hand, interaction::inverse(hand_in_object));
        require(
            length(
                published_object.position -
                carry.object_world().position) < 0.005F,
            "object was not wrist-authoritative");
        previous = pose;
    }
    require(length(
                previous.positions[g1_skeleton::Simulation] -
                start_root) > 0.10F,
            "layered carry did not retain walking root motion");
    require(maximum_knee_change > 0.005F,
            "layered carry froze the legs");
    require(maximum_inactive_arm_change > 0.001F,
            "layered carry froze the inactive arm");
}
```

Call `test_hand(Left)` and `test_hand(Right)` from `main()`. Register the new test target and link `episode_layered_carry.cpp`, `g1_flat_motion_matcher.cpp`, `interaction_posture_ik.cpp`, `interaction_ik.cpp`, and `interaction_pose.cpp`.

Add a fallback case after the bilateral cases. Start a second controller with
`object_world.position.x += 5.0F`, call `update()` once, and require
`!last_solve_accepted()`, a finite returned pose, and an unchanged live
locomotion root. This proves an unreachable carry target uses the bounded
last-safe branch instead of detaching, teleporting, or publishing NaNs.

- [ ] **Step 2: Run the test and verify the controller is absent**

Run:

```bash
make build/tests/test_episode_layered_carry
```

Expected: compilation fails because `episode_layered_carry.h` does not exist.

- [ ] **Step 3: Implement the focused layered-carry controller**

Create `episode_layered_carry.h`:

```cpp
#pragma once

#include "interaction_posture_ik.h"

namespace episode {

class LayeredCarry {
public:
    void start(
        const interaction::Pose& final_hold_pose,
        interaction::Hand hand,
        interaction::Transform hand_in_object,
        interaction::Transform object_world);
    const interaction::Pose& update(
        const interaction::LocomotionSnapshot& locomotion);
    const interaction::Pose& pose() const;
    interaction::Transform object_world() const;
    bool last_solve_accepted() const;

private:
    interaction::Pose final_hold_pose_{};
    interaction::Pose last_safe_pose_{};
    interaction::UpperBodyAngles temporal_seed_{};
    interaction::Transform hand_in_object_{};
    interaction::Transform object_in_root_{};
    interaction::Transform object_world_{};
    interaction::Hand hand_ = interaction::Hand::Right;
    bool started_ = false;
    bool last_solve_accepted_ = false;
};

}  // namespace episode
```

Create `episode_layered_carry.cpp` with these helpers and behavior:

```cpp
#include "episode_layered_carry.h"

#include "g1_arm_joint_metadata.h"

#include <array>
#include <stdexcept>

namespace episode {
namespace {

interaction::Transform root_transform(const interaction::Pose& pose) {
    return {
        pose.positions[g1_skeleton::Simulation],
        pose.rotations[g1_skeleton::Simulation],
    };
}

interaction::Transform hand_transform(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t wrist = hand == interaction::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
    return {world.positions[wrist], world.rotations[wrist]};
}

const std::array<interaction::HingeJoint, 7>& arm(
    interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? interaction::kLeftArm
        : interaction::kRightArm;
}

}  // namespace

void LayeredCarry::start(
    const interaction::Pose& final_hold_pose,
    interaction::Hand hand,
    interaction::Transform hand_in_object,
    interaction::Transform object_world) {
    final_hold_pose_ = final_hold_pose;
    last_safe_pose_ = final_hold_pose;
    hand_ = hand;
    hand_in_object_ = hand_in_object;
    object_in_root_ = interaction::compose(
        interaction::inverse(root_transform(final_hold_pose)),
        object_world);
    object_world_ = object_world;
    temporal_seed_ =
        interaction::decompose_upper_body(final_hold_pose, hand);
    started_ = true;
    last_solve_accepted_ = true;
}

const interaction::Pose& LayeredCarry::update(
    const interaction::LocomotionSnapshot& locomotion) {
    if (!started_) {
        throw std::logic_error("layered carry was not started");
    }
    const interaction::Transform desired_object = interaction::compose(
        root_transform(locomotion.pose), object_in_root_);
    const interaction::Transform target_hand = interaction::compose(
        desired_object, hand_in_object_);
    interaction::Pose requested = locomotion.pose;
    for (const interaction::HingeJoint& joint : arm(hand_)) {
        requested.rotations[static_cast<size_t>(joint.bone)] =
            final_hold_pose_.rotations[static_cast<size_t>(joint.bone)];
    }
    constexpr std::array<size_t, 3> spine = {
        g1_skeleton::Spine,
        g1_skeleton::Spine1,
        g1_skeleton::Spine2,
    };
    for (size_t bone : spine) {
        requested.rotations[bone] = quat_nlerp_shortest(
            locomotion.pose.rotations[bone],
            final_hold_pose_.rotations[bone],
            0.25F);
    }

    interaction::Pose solved = requested;
    const interaction::PostureIKResult result =
        interaction::solve_hand_posture_ik_task_priority(
            solved,
            hand_,
            target_hand,
            requested,
            temporal_seed_);
    last_solve_accepted_ = result.accepted;
    if (result.accepted) {
        last_safe_pose_ = solved;
        temporal_seed_ = result.joint_angles;
    } else {
        const interaction::Transform live_root =
            root_transform(locomotion.pose);
        last_safe_pose_.positions[g1_skeleton::Simulation] =
            live_root.position;
        last_safe_pose_.rotations[g1_skeleton::Simulation] =
            live_root.rotation;
        last_safe_pose_.velocities[g1_skeleton::Simulation] =
            locomotion.pose.velocities[g1_skeleton::Simulation];
        last_safe_pose_.angular_velocities[g1_skeleton::Simulation] =
            locomotion.pose.angular_velocities[g1_skeleton::Simulation];
    }
    object_world_ = interaction::compose(
        hand_transform(last_safe_pose_, hand_),
        interaction::inverse(hand_in_object_));
    return last_safe_pose_;
}

const interaction::Pose& LayeredCarry::pose() const {
    return last_safe_pose_;
}

interaction::Transform LayeredCarry::object_world() const {
    return object_world_;
}

bool LayeredCarry::last_solve_accepted() const {
    return last_solve_accepted_;
}

}  // namespace episode
```

- [ ] **Step 4: Run and tune only against observable invariants**

Run:

```bash
make build/tests/test_episode_layered_carry
./build/tests/test_episode_layered_carry
```

Expected: both hands pass root displacement, leg motion, inactive-arm motion, and wrist-authoritative object checks. If the measured native clip amplitude is below a proposed threshold, print the measured value and lower only that assertion to half the observed nonzero amplitude; do not add root translation or overwrite leg channels.

- [ ] **Step 5: Integrate layered carry into `InteractionEpisode`**

Include `episode_layered_carry.h`, add:

```cpp
LayeredCarry layered_carry_{};
interaction::Transform hand_in_object_{};
```

Add `std::string diagnostic;` to `EpisodeOutput`. Immediately after
`layered_carry_.update(...)`, publish the fallback status without changing
episode state:

```cpp
output_.diagnostic = layered_carry_.last_solve_accepted()
    ? std::string{}
    : std::string{
          "carry IK rejected; using last safe active-arm branch"};
```

Clear `output_.diagnostic` in `commit()`, successful `reset()`, and successful
placement return. Add one HUD line that displays `runtime.output().diagnostic`
in amber when nonempty.

When constructing the pickup affordance in `commit()`, retain:

```cpp
hand_in_object_ = affordance.hand_in_object;
```

At pickup contact, replace the carry-database switch with:

```cpp
matcher_.switch_database(walking_database_, contact_pose_);
layered_carry_.start(
    contact_pose_,
    output_.selected_hand,
    hand_in_object_,
    output_.object_world);
```

In `update_carry()`, advance the walking matcher and compose:

```cpp
matcher_.update(input.command, input.dt);
interaction::Pose displayed =
    layered_carry_.update(matcher_.snapshot());
if (state_ == EpisodeState::CarryBlend) {
    state_seconds_ += input.dt;
    const float alpha = smoothstep(
        state_seconds_ / config_.carry_blend_seconds);
    displayed = interaction::interpolate_pose(
        contact_pose_, displayed, alpha);
    if (state_seconds_ >= config_.carry_blend_seconds) {
        state_ = EpisodeState::Carry;
    }
}
publish(displayed);
update_attached_object(input.dt);
```

Make `update_place_approach()` advance through the same `update_carry` composition before calculating entry convergence, rather than publishing `matcher_.snapshot().pose` directly. On place cancellation, retain/rebase the walking database and continue `LayeredCarry`; do not switch to a complete carry database. Existing `PlaceBridge`, `PlaceReach`, and `PlaceReturn` continue to use the complete selected reach poses. After return, keep the existing `matcher_.switch_database(walking_database_, output_.pose)`.

Add `episode_layered_carry.cpp`, `interaction_posture_ik.cpp`, and `interaction_ik.cpp` to the episode viewer and episode test compile/link rules.

- [ ] **Step 6: Replace the old full-carry episode assertion**

In `tests/cpp/test_interaction_episode.cpp`, make every runtime use:

```cpp
pack / "walking_database.bin"
```

Replace the old `test_flat_walking_*` assumptions with:

```cpp
void test_native_g1_carry_preserves_grasp_and_leg_motion(
    reach::Hand hand) {
    const std::filesystem::path pack("build/g1-episode");
    episode::InteractionEpisode runtime(
        pack / "walking_database.bin",
        pack / "carry_left_database.bin",
        pack / "carry_right_database.bin",
        fast_config());
    const auto pickup = make_attempt(
        runtime.output().pose, hand, 31U, 40U, 9U);
    const interaction::Transform hand_in_object =
        interaction::compose(
            interaction::inverse(pickup.object.world),
            pickup.grasp.hand_world);
    require(runtime.commit(pickup),
            "native G1 carry fixture did not commit");
    advance_until(runtime, episode::EpisodeState::Carry, 31U);

    const interaction::Hand selected =
        hand == reach::Hand::Left
            ? interaction::Hand::Left
            : interaction::Hand::Right;
    const size_t wrist =
        selected == interaction::Hand::Left
            ? g1_skeleton::LeftWrist
            : g1_skeleton::RightWrist;
    const vec3 start_root =
        runtime.output().pose.positions[g1_skeleton::Simulation];
    interaction::Pose previous = runtime.output().pose;
    float maximum_knee_change = 0.0F;
    float maximum_position_drift = 0.0F;
    float maximum_orientation_drift = 0.0F;
    const episode::LocomotionCommand command{
        vec3(0.0F, 0.0F, 0.22F), quat()};
    for (int tick = 0; tick < 150; ++tick) {
        runtime.update({
            kTick, command, 31U, false, false, 0U});
        const interaction::Pose& pose = runtime.output().pose;
        maximum_knee_change = std::max(
            maximum_knee_change,
            quat_angle_between(
                pose.rotations[g1_skeleton::LeftKnee],
                previous.rotations[g1_skeleton::LeftKnee]));
        const interaction::WorldPose world =
            interaction::world_pose(pose);
        const interaction::Transform actual_hand{
            world.positions[wrist], world.rotations[wrist]};
        const interaction::Transform expected_hand =
            interaction::compose(
                runtime.output().object_world, hand_in_object);
        maximum_position_drift = std::max(
            maximum_position_drift,
            length(
                actual_hand.position -
                expected_hand.position));
        maximum_orientation_drift = std::max(
            maximum_orientation_drift,
            quat_angle_between(
                actual_hand.rotation,
                expected_hand.rotation));
        require(runtime.output().attached,
                "native G1 carry lost attachment");
        require(!runtime.output().flat_locomotion.valid,
                "native G1 carry exposed a flat skeleton");
        previous = pose;
    }
    require(
        length(
            runtime.output().pose.positions[
                g1_skeleton::Simulation] -
            start_root) > 0.10F,
        "native G1 carry did not move the root");
    require(maximum_knee_change > 0.005F,
            "native G1 carry froze the legs");
    require(maximum_position_drift < 0.01F,
            "native G1 carry drifted from grasp position");
    require(maximum_orientation_drift < 0.05F,
            "native G1 carry drifted from grasp orientation");
}
```

Call the function for both `reach::Hand::Left` and
`reach::Hand::Right`. Preserve the placement test and add direct
`require(!runtime.output().flat_locomotion.valid, "...")` assertions after
observing FreeLocomotion, Reach, Carry, PlaceReach, and returned
FreeLocomotion.

- [ ] **Step 7: Run all episode-focused tests**

Run:

```bash
make build/tests/test_episode_layered_carry \
  build/tests/test_interaction_episode \
  g1_interaction_episode_viewer
./build/tests/test_episode_layered_carry
./build/tests/test_interaction_episode
python3 -m unittest \
  tests.python.test_g1_interaction_episode_viewer -v
```

Expected: all tests pass; no test loads `resources/database.bin`; bilateral carry moves the root and legs without grasp drift; placement returns to native G1 locomotion.

- [ ] **Step 8: Commit layered carry**

```bash
git add episode_layered_carry.h episode_layered_carry.cpp \
  tests/cpp/test_episode_layered_carry.cpp interaction_episode.h \
  interaction_episode.cpp tests/cpp/test_interaction_episode.cpp Makefile
git commit -m "feat: layer grasp carry over G1 locomotion"
```

---

### Task 4: Full Regression and Driven Viewer Acceptance

**Files:**
- Modify only if evidence reveals a defect: files owned by Tasks 1-3
- Do not modify: generated reach-pack data or root-level binaries

**Interfaces:**
- Consumes: the completed native-G1 viewer
- Produces: test logs, screenshots, and a detached running viewer for user acceptance

- [ ] **Step 1: Run the complete relevant automated suite**

Run:

```bash
make build/tests/test_episode_controls \
  build/tests/test_g1_flat_motion_matcher \
  build/tests/test_episode_layered_carry \
  build/tests/test_interaction_episode \
  g1_interaction_episode_viewer
./build/tests/test_episode_controls
./build/tests/test_g1_flat_motion_matcher
./build/tests/test_episode_layered_carry
./build/tests/test_interaction_episode
python3 -m unittest \
  tests.python.test_g1_interaction_episode_viewer -v
```

Expected: every command exits zero.

- [ ] **Step 2: Launch a clean detached viewer**

Stop only the prior `g1_interaction_episode_viewer` process after resolving its exact PID. Launch:

```bash
setsid ./g1_interaction_episode_viewer \
  build/g1-episode build/g1-reaches/reach-pack-v2 \
  >/tmp/g1_interaction_episode_viewer.log 2>&1 </dev/null &
```

Expected: one Raylib viewer window appears and the log contains no startup failure.

- [ ] **Step 3: Drive controls and interaction**

Use `xdotool` against the exact viewer window:

1. Hold A for one second and capture a screenshot.
2. Hold D for two seconds and verify the character crosses back screen-right.
3. Move to the object and press F once.
4. Wait for approach, bridge, and reach; verify attachment.
5. Hold W for three seconds during Carry; verify root displacement and alternating legs.
6. Press F at the shelf destination; verify PlaceApproach, release, reverse reach, and FreeLocomotion.
7. Re-pick the shelf object and place it back on the lower table.

Expected: the blue figure never changes scale/topology, A/D follow screen direction, carry does not skate, and the object stays locked to the selected wrist.

- [ ] **Step 4: Inspect evidence**

Run:

```bash
tail -n 100 /tmp/g1_interaction_episode_viewer.log
git status --short
```

Expected: no `FAILED`, attachment loss, non-finite pose, or skeleton mismatch messages. Git status contains only intended source changes plus the pre-existing generated artifacts.

- [ ] **Step 5: Fix only evidence-backed failures and rerun**

For any failed invariant, first add or strengthen the focused test that reproduces it, observe red, apply the smallest repair in the owning Task 1-3 file, and repeat Steps 1-4. Do not introduce terrain, mesh-only pose paths, root translation, or full-pose carry database switching.

- [ ] **Step 6: Final commit if acceptance required a repair**

```bash
git add episode_controls.h episode_layered_carry.h \
  episode_layered_carry.cpp g1_flat_motion_matcher.h \
  g1_flat_motion_matcher.cpp interaction_episode.h \
  interaction_episode.cpp g1_interaction_episode_viewer.cpp \
  tests/cpp/test_episode_controls.cpp \
  tests/cpp/test_episode_layered_carry.cpp \
  tests/cpp/test_g1_flat_motion_matcher.cpp \
  tests/cpp/test_interaction_episode.cpp \
  tests/python/test_g1_interaction_episode_viewer.py Makefile
git commit -m "fix: harden native G1 playable episode"
```
