#include "interaction_place_controller.h"
#include "interaction_rotation_gate.h"

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

void restage(Fixture& fixture) {
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
    fixture.input.current_object_world = compose(
        scene, clip.object_poses.front());
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
    restage(fixture);
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

void test_rotation_gate_preserves_adjacent_float_evidence() {
    constexpr std::array<float, 6> limits = {
        0.001745329F,
        0.05F,
        0.10F,
        0.174532925F,
        0.261799388F,
        0.436332313F,
    };
    constexpr std::array<vec3, 3> axes = {
        vec3(1.0F, 0.0F, 0.0F),
        vec3(0.0F, 1.0F, 0.0F),
        vec3(0.0F, 0.0F, 1.0F),
    };
    const rotation_gate::Rotation composite_base =
        rotation_gate::multiply(
            rotation_gate::from_quat(quat_from_angle_axis(
                0.60F, vec3(0.0F, 1.0F, 0.0F))),
            rotation_gate::from_quat(quat_from_angle_axis(
                -0.40F, vec3(1.0F, 0.0F, 0.0F))));

    for (float limit : limits) {
        const float above_limit = std::nextafter(
            limit, std::numeric_limits<float>::infinity());
        for (vec3 axis : axes) {
            const quat exact = quat_from_angle_axis(limit, axis);
            const quat above = quat_from_angle_axis(above_limit, axis);
            require(
                rotation_gate::within(exact, quat(), limit),
                "exact encoded gate limit accepted");
            require(
                !rotation_gate::within(above, quat(), limit),
                "next-float encoded gate limit rejected");
            require(
                rotation_gate::within(-exact, quat(), limit),
                "antipodal exact gate limit accepted");
            require(
                rotation_gate::within(
                    4.0F * exact, 2.0F * quat(), limit),
                "scaled exact gate limit accepted");

            const rotation_gate::Rotation exact_composed =
                rotation_gate::multiply(
                    rotation_gate::from_quat(exact), composite_base);
            const rotation_gate::Rotation above_composed =
                rotation_gate::multiply(
                    rotation_gate::from_quat(above), composite_base);
            require(
                rotation_gate::within(
                    exact_composed, composite_base, limit),
                "exact factored composite limit accepted");
            require(
                !rotation_gate::within(
                    above_composed, composite_base, limit),
                "next-float factored composite limit rejected");
        }
    }
}

void test_orientation_request_boundary_is_exact() {
    constexpr float limit = 0.10F;
    const auto solve_target = [](float angle) {
        Pose pose = make_pose(
            vec3(), Transform{vec3(0.0F, 1.0F, 0.0F), quat()});
        const Transform current = hand_world(pose);
        Transform target = current;
        target.rotation = quat_mul(
            quat_from_angle_axis(
                angle, vec3(0.0F, 1.0F, 0.0F)),
            current.rotation);
        IKConfig config{};
        config.maximum_request_position_m = 0.0F;
        config.maximum_request_orientation_radians = limit;
        return solve_hand_ik(pose, Hand::Right, target, config);
    };

    const auto solve_current = [](float angle) {
        Pose pose = make_pose(
            vec3(),
            Transform{
                vec3(0.0F, 1.0F, 0.0F),
                quat_from_angle_axis(
                    angle, vec3(0.0F, 1.0F, 0.0F))});
        Transform target = hand_world(pose);
        target.rotation = quat();
        IKConfig config{};
        config.maximum_request_position_m = 0.0F;
        config.maximum_request_orientation_radians = limit;
        return solve_hand_ik(pose, Hand::Right, target, config);
    };

    const auto solve_accepted_target = [](float angle) {
        Pose pose = make_pose(
            vec3(), Transform{vec3(0.0F, 1.0F, 0.0F), quat()});
        const Transform current = hand_world(pose);
        Transform target = current;
        target.rotation = quat_mul(
            quat_from_angle_axis(
                angle, vec3(0.0F, 1.0F, 0.0F)),
            current.rotation);
        IKConfig config{};
        config.maximum_request_position_m = 0.0F;
        config.maximum_request_orientation_radians = 0.20F;
        config.accepted_position_m = 0.0F;
        config.accepted_orientation_radians = limit;
        config.maximum_iterations = 0;
        return solve_hand_ik(pose, Hand::Right, target, config);
    };

    const IKResult exact = solve_target(limit);
    require(exact.accepted, "exact IK orientation request accepted");
    const float above_limit = std::nextafter(
        limit, std::numeric_limits<float>::infinity());
    const IKResult above = solve_target(above_limit);
    require(!above.accepted, "next-float IK orientation request rejected");
    require(
        above.reason == Reason::CorrectionLimit,
        "next-float IK orientation uses correction-limit reason");

    const IKResult exact_current = solve_current(limit);
    require(
        exact_current.accepted,
        "exact current-hand IK orientation accepted");
    const IKResult above_current = solve_current(above_limit);
    require(
        !above_current.accepted,
        "next-float current-hand IK orientation rejected");
    require(
        above_current.reason == Reason::CorrectionLimit,
        "next-float current-hand IK uses correction-limit reason");

    const IKResult accepted_exact = solve_accepted_target(limit);
    require(
        accepted_exact.accepted,
        "exact IK accepted-orientation boundary accepted");
    const IKResult accepted_above = solve_accepted_target(above_limit);
    require(
        !accepted_above.accepted,
        "next-float IK accepted-orientation boundary rejected");
    require(
        accepted_above.orientation_error_radians > limit,
        "next-float accepted-orientation diagnostic is outside boundary");

    const auto solve_factored = [](
        float angle,
        float maximum,
        float accepted) {
        Pose pose = make_pose(
            vec3(), Transform{vec3(0.0F, 1.0F, 0.0F), quat()});
        const quat scene_rotation = quat_from_angle_axis(
            0.60F, vec3(0.0F, 1.0F, 0.0F));
        const quat source_root_rotation = quat_from_angle_axis(
            -0.35F, vec3(1.0F, 0.0F, 0.0F));
        const rotation_gate::Rotation root_evidence =
            rotation_gate::multiply(
                rotation_gate::from_quat(scene_rotation),
                rotation_gate::from_quat(source_root_rotation));
        pose.rotations[kRoot] = quat_normalize(quat_mul(
            scene_rotation, source_root_rotation));
        pose.rotations[kRightHand] = quat_from_angle_axis(
            -0.40F, vec3(1.0F, 0.0F, 0.0F));
        const Transform current = hand_world(pose);
        const quat delta = quat_from_angle_axis(
            angle, vec3(0.0F, 0.0F, 1.0F));
        Transform target = current;
        target.rotation = quat_mul(delta, current.rotation);
        const rotation_gate::Rotation target_evidence =
            rotation_gate::multiply(
                rotation_gate::from_quat(delta),
                world_rotation_evidence(
                    pose, kRightHand, root_evidence));
        IKConfig config{};
        config.maximum_request_position_m = 0.0F;
        config.maximum_request_orientation_radians = maximum;
        config.accepted_position_m = 0.0F;
        config.accepted_orientation_radians = accepted;
        config.maximum_iterations = 0;
        return solve_hand_ik_with_rotation_evidence(
            pose,
            Hand::Right,
            target,
            target_evidence,
            root_evidence,
            config);
    };

    constexpr float request_limit = 0.436332313F;
    const float request_above = std::nextafter(
        request_limit, std::numeric_limits<float>::infinity());
    require(
        solve_factored(
            request_limit, request_limit, request_limit).accepted,
        "factored exact default IK request accepted");
    require(
        !solve_factored(
            request_above, request_limit, request_limit).accepted,
        "factored next-float default IK request rejected");

    constexpr float accepted_limit = 0.261799388F;
    const float accepted_limit_above = std::nextafter(
        accepted_limit, std::numeric_limits<float>::infinity());
    require(
        solve_factored(
            accepted_limit, request_limit, accepted_limit).accepted,
        "factored exact default IK accepted boundary accepted");
    require(
        !solve_factored(
            accepted_limit_above,
            request_limit,
            accepted_limit).accepted,
        "factored next-float default IK accepted boundary rejected");
}

PlaceStep release_with_authored_orientation(
    float angle,
    bool nonidentity_scene = false) {
    Fixture fixture = make_fixture(1.0F);
    if (nonidentity_scene) {
        const quat common_rotation = quat_from_angle_axis(
            0.60F, vec3(0.0F, 1.0F, 0.0F));
        fixture.input.surface.surface_world.rotation = common_rotation;
        fixture.input.surface.support_volume_world.rotation = common_rotation;
        RecordedPlaceClip& clip = fixture.library.recorded.front();
        clip.object_bounds.half_extents_object.x = 0.03F;
        fixture.input.held_object_bounds = clip.object_bounds;
        fixture.input.object_dimensions.x = 0.06F;
        for (Transform& object : clip.object_poses) {
            object.rotation = common_rotation;
        }
        for (size_t frame = 0; frame < clip.poses.size(); ++frame) {
            Pose& pose = clip.poses[frame];
            pose.rotations[kRoot] = common_rotation;
            pose.rotations[kRightHand] = quat();
            pose.positions[kRightHand] = quat_mul_vec3(
                quat_inv(common_rotation),
                clip.object_poses[frame].position -
                    pose.positions[kRoot]);
        }
        fixture.input.place_affordance =
            fixture.input.surface.affordances.front();
        restage(fixture);
    }
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    require(controller.begin(begin).accepted, "orientation release begin");
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    clip.poses[static_cast<size_t>(clip.release_frame)]
        .rotations[kRightHand] = quat_from_angle_axis(
            angle,
            nonidentity_scene
                ? vec3(0.0F, 0.0F, 1.0F)
                : vec3(0.0F, 1.0F, 0.0F));
    return run_to_terminal(controller);
}

PlaceStep release_with_factored_request_orientation(float angle) {
    constexpr float request_limit = 0.436332313F;
    Fixture fixture = make_fixture(1.0F);
    const quat common_rotation = quat_from_angle_axis(
        0.60F, vec3(0.0F, 1.0F, 0.0F));
    const quat local_rotation = quat_from_angle_axis(
        angle, vec3(0.0F, 0.0F, 1.0F));
    fixture.input.surface.surface_world.rotation = common_rotation;
    fixture.input.surface.support_volume_world.rotation = common_rotation;
    RecordedPlaceClip& clip = fixture.library.recorded.front();
    clip.object_bounds.half_extents_object.x = 0.03F;
    fixture.input.held_object_bounds = clip.object_bounds;
    fixture.input.object_dimensions.x = 0.06F;
    for (size_t frame = 0; frame < clip.poses.size(); ++frame) {
        Transform& object = clip.object_poses[frame];
        object.rotation = quat_normalize(quat_mul(
            common_rotation, local_rotation));
        Pose& pose = clip.poses[frame];
        pose.rotations[kRoot] = common_rotation;
        pose.rotations[kRightHand] = local_rotation;
        pose.positions[kRightHand] = quat_mul_vec3(
            quat_inv(common_rotation),
            object.position - pose.positions[kRoot]);
    }
    fixture.input.place_affordance =
        fixture.input.surface.affordances.front();
    fixture.input.ik.maximum_request_orientation_radians = request_limit;
    fixture.input.ik.accepted_orientation_radians = request_limit;
    fixture.input.ik.maximum_iterations = 0;
    restage(fixture);
    PlaceBeginInput begin = selected_begin(fixture);
    PlaceController controller = make_controller(fixture);
    require(
        controller.begin(begin).accepted,
        "factored request orientation begin");
    return run_to_terminal(controller);
}

void test_controller_request_boundary_uses_factored_evidence() {
    constexpr float request_limit = 0.436332313F;
    const PlaceStep exact =
        release_with_factored_request_orientation(request_limit);
    require(
        exact.recover_to_carry,
        "factored exact request reaches final release gate");
    require(
        exact.reason == Reason::ReleaseOrientation,
        "factored exact request is not rejected as correction limit");
}

void test_release_orientation_boundary_is_exact() {
    constexpr float limit = 0.174532925F;
    const PlaceStep exact = release_with_authored_orientation(limit);
    require(exact.release_due, "exact release orientation accepted");
    require(!exact.recover_to_carry, "exact release does not recover");
    require(
        exact.hand_orientation_error_radians <= limit,
        "exact release diagnostic is inside boundary");

    const PlaceStep above = release_with_authored_orientation(std::nextafter(
        limit, std::numeric_limits<float>::infinity()));
    require(!above.release_due, "next-float release orientation not emitted");
    require(above.recover_to_carry, "next-float release recovers attached");
    require(
        above.reason == Reason::ReleaseOrientation,
        "next-float release uses orientation reason");
    require(
        above.hand_orientation_error_radians > limit,
        "next-float release diagnostic is outside boundary");

    const PlaceStep composed_exact = release_with_authored_orientation(
        limit, true);
    require(
        composed_exact.release_due,
        "nonidentity exact release orientation accepted");
    require(
        !composed_exact.recover_to_carry,
        "nonidentity exact release does not recover");
    require(
        composed_exact.hand_orientation_error_radians <= limit,
        "nonidentity exact release diagnostic is inside boundary");

    const PlaceStep composed_above = release_with_authored_orientation(
        std::nextafter(limit, std::numeric_limits<float>::infinity()),
        true);
    require(
        !composed_above.release_due,
        "nonidentity next-float release not emitted");
    require(
        composed_above.recover_to_carry,
        "nonidentity next-float release recovers attached");
    require(
        composed_above.reason == Reason::ReleaseOrientation,
        "nonidentity next-float release uses orientation reason");
    require(
        composed_above.hand_orientation_error_radians > limit,
        "nonidentity next-float diagnostic is outside boundary");
}

}  // namespace

int main() {
    test_release_is_one_shot_and_clamped(0.85F);
    test_release_is_one_shot_and_clamped(1.15F);
    test_actual_boundary_rejection_is_mutation_free();
    test_rotation_gate_preserves_adjacent_float_evidence();
    test_orientation_request_boundary_is_exact();
    test_controller_request_boundary_uses_factored_evidence();
    test_release_orientation_boundary_is_exact();
    return 0;
}
