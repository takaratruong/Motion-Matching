#include "interaction_place_controller.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using namespace interaction;

constexpr size_t kRoot = static_cast<size_t>(g1_skeleton::Simulation);
constexpr size_t kRightHand = static_cast<size_t>(g1_skeleton::RightWrist);

[[noreturn]] void fail(std::string_view message) {
    std::cerr << "place fast-math check failed: " << message << '\n';
    std::exit(1);
}

void require(bool condition, std::string_view message) {
    if (!condition) fail(message);
}

bool near(float left, float right, float tolerance = 2.0e-4F) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = 2.0e-4F) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = 2.0e-4F) {
    return quat_angle_between(left, right) <= tolerance;
}

bool near(Transform left, Transform right, float tolerance = 2.0e-4F) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
}

Transform hand_world(const Pose& pose) {
    const WorldPose world = world_pose(pose);
    return {world.positions[kRightHand], world.rotations[kRightHand]};
}

Pose make_pose(vec3 root, Transform hand) {
    Pose pose{};
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        pose.rotations[bone] = quat();
    }
    pose.positions[kRoot] = root;
    pose.positions[kRightHand] = hand.position - root;
    pose.rotations[kRightHand] = hand.rotation;
    return pose;
}

PlacementSurface make_surface(
    uint64_t id,
    vec3 position = vec3(1.0F, 0.70F, 2.0F)) {
    PlacementSurface surface{};
    surface.handle = {id, 1U};
    surface.surface_world = {position, quat()};
    surface.support_volume_world = {
        position - vec3(0.0F, 0.35F, 0.0F), quat()};
    surface.support_volume_size = vec3(1.0F, 0.70F, 1.0F);
    surface.half_extent_x_m = 0.40F;
    surface.half_extent_z_m = 0.40F;
    surface.overhead_clearance_m = 0.50F;
    surface.affordances = {
        PlaceAffordance{
            7U,
            Transform{vec3(0.0F, 0.10F, 0.0F), quat()},
            vec3(0.0F, -0.10F, 0.0F),
            vec3(0.0F, 1.0F, 0.0F),
            0.01F,
        },
    };
    return surface;
}

RecordedPlaceClip make_clip() {
    RecordedPlaceClip clip{};
    clip.id = 9001U;
    clip.object_profile_id = 81U;
    clip.entry_frame = 0;
    clip.commit_frame = 8;
    clip.release_frame = 12;
    clip.retract_stop_frame = 15;
    clip.hand = Hand::Right;
    clip.hand_in_object = {vec3(), quat()};
    clip.object_bounds = {vec3(), vec3(0.04F, 0.10F, 0.04F)};
    clip.source_surface = make_surface(41U, vec3(0.0F, 0.70F, 0.0F));
    clip.source_affordance_id = 7U;
    for (int32_t frame = 0; frame <= clip.retract_stop_frame; ++frame) {
        const float height = frame <= clip.release_frame
            ? 0.80F + 0.01F * static_cast<float>(clip.release_frame - frame)
            : 0.80F;
        const Transform object{vec3(0.0F, height, 0.0F), quat()};
        clip.object_poses.push_back(object);
        clip.poses.push_back(make_pose(
            vec3(0.005F * static_cast<float>(frame), 0.0F, 0.0F),
            object));
        clip.active_hand_contacts.push_back(
            frame <= clip.release_frame ? 1U : 0U);
    }
    return clip;
}

struct Fixture {
    Database database{};
    PlaceMotionLibrary library{};
    PlaceMatchInput input{};
};

void refresh(Fixture& fixture) {
    fixture.input.pickup_database = &fixture.database;
    fixture.input.library = &fixture.library;
}

Fixture make_fixture(float speed) {
    Fixture fixture{};
    fixture.library.recorded.push_back(make_clip());
    refresh(fixture);
    fixture.input.held_target = {11U, 2U};
    fixture.input.held_object_profile_id = 81U;
    fixture.input.held_object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    fixture.input.held_affordance = {
        3U,
        Hand::Right,
        Transform{vec3(), quat()},
        vec3(0.0F, 1.0F, 0.0F),
        0.01F,
    };
    fixture.input.surface = make_surface(77U);
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    fixture.input.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    fixture.input.timing.playback_speed = speed;
    const RecordedPlaceClip& clip = fixture.library.recorded.front();
    const Transform goal = placement_goal_world(
        fixture.input.surface,
        fixture.input.place_affordance.object_in_surface);
    const Transform scene = compose(
        goal,
        inverse(clip.object_poses[static_cast<size_t>(clip.release_frame)]));
    fixture.input.current_pose = clip.poses.front();
    const Transform source_root{
        fixture.input.current_pose.positions[kRoot],
        fixture.input.current_pose.rotations[kRoot]};
    const Transform mapped_root = compose(scene, source_root);
    fixture.input.current_pose.positions[kRoot] = mapped_root.position;
    fixture.input.current_pose.rotations[kRoot] = mapped_root.rotation;
    fixture.input.current_object_world = compose(scene, clip.object_poses.front());
    return fixture;
}

PlaceController make_controller(const Fixture& fixture) {
    PlaceControllerConfig config{};
    config.timing = fixture.input.timing;
    config.match = fixture.input.match;
    return PlaceController(config, fixture.input.ik);
}

PlaceBeginInput selected_begin(Fixture& fixture) {
    refresh(fixture);
    const PlaceResult selected = select_place_motion(fixture.input);
    require(selected.accepted, "recorded fixture selection");
    return {fixture.input, selected.candidate};
}

PlaceStep run_to_terminal(PlaceController& controller) {
    for (int tick = 0; tick < 64; ++tick) {
        const PlaceStep step = controller.update(0.04F);
        if (step.release_due || step.recover_to_carry) return step;
    }
    fail("controller did not reach release or recovery");
}

void test_release_is_one_shot_and_clamped(float speed) {
    Fixture fixture = make_fixture(speed);
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    require(controller.begin(begin).accepted, "begin at allowed speed");
    const PlaceStep release = run_to_terminal(controller);
    require(release.release_due, "release pulse emitted");
    require(!release.recover_to_carry, "release stays attached pending ack");
    require(
        release.source_frame == begin.candidate.release_frame,
        "fractional speed clamps exact release frame");
    require(near(
        release.object_world,
        compose(
            hand_world(release.pose),
            inverse(fixture.input.held_affordance.hand_in_object))),
        "pending object remains hand-derived");
    const PlaceStep pending = controller.update(0.04F);
    require(!pending.release_due, "pending release does not pulse twice");
    require(near(pending.object_world, release.object_world), "pending object frozen by clamp");
    controller.acknowledge_release(release.object_world);
    const PlaceStep retract = controller.update(0.04F);
    require(!retract.release_due, "acknowledged release stays one-shot");
    require(near(retract.object_world, release.object_world), "released object stays fixed");
}

void test_actual_boundary_rejection_is_mutation_free() {
    Fixture fixture = make_fixture(1.0F);
    fixture.input.surface.affordances.front().object_in_surface.position.x =
        0.35F;
    fixture.input.place_affordance = fixture.input.surface.affordances.front();
    const RecordedPlaceClip& authored = fixture.library.recorded.front();
    const Transform goal = placement_goal_world(
        fixture.input.surface,
        fixture.input.place_affordance.object_in_surface);
    const Transform scene = compose(
        goal,
        inverse(authored.object_poses[static_cast<size_t>(
            authored.release_frame)]));
    fixture.input.current_pose = authored.poses.front();
    const Transform mapped_root = compose(
        scene,
        Transform{
            authored.poses.front().positions[kRoot],
            authored.poses.front().rotations[kRoot]});
    fixture.input.current_pose.positions[kRoot] = mapped_root.position;
    fixture.input.current_pose.rotations[kRoot] = mapped_root.rotation;
    fixture.input.current_object_world = compose(scene, authored.object_poses.front());
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    require(controller.begin(begin).accepted, "boundary begin");
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    const float shift = std::nextafter(0.02F, 0.0F);
    clip.poses[static_cast<size_t>(clip.release_frame)]
        .positions[kRightHand].x += shift;
    PlaceStep previous{};
    PlaceStep terminal{};
    for (int tick = 0; tick < 64; ++tick) {
        const PlaceStep step = controller.update(0.04F);
        if (step.recover_to_carry) {
            terminal = step;
            break;
        }
        previous = step;
    }
    require(terminal.recover_to_carry, "actual footprint rejects while attached");
    require(!terminal.release_due, "failed gate never emits release");
    require(
        terminal.reason == Reason::PlacementOutOfBounds,
        "actual footprint reason is preserved");
    require(near(terminal.object_world, previous.object_world), "recovery preserves last-safe object");
    const PlaceStep repeated = controller.update(0.04F);
    require(!repeated.release_due, "failed release cannot emit later");
    require(near(repeated.object_world, terminal.object_world), "failed recovery is mutation-free");
}

}  // namespace

int main() {
    test_release_is_one_shot_and_clamped(0.85F);
    test_release_is_one_shot_and_clamped(1.15F);
    test_actual_boundary_rejection_is_mutation_free();
    return 0;
}
