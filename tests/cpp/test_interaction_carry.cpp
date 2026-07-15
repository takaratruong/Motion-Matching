#include "interaction_carry.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr float kTolerance = 1.0e-5F;

bool near(float left, float right, float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = kTolerance) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near(quat left, quat right, float tolerance = kTolerance) {
    const float left_length = quat_length(left);
    const float right_length = quat_length(right);
    if (!(left_length > 0.0F) || !(right_length > 0.0F)) return false;
    left = left / left_length;
    right = right / right_length;
    const float cosine = std::clamp(
        std::abs(quat_dot(left, right)), 0.0F, 1.0F);
    return 2.0F * std::acos(cosine) <= tolerance;
}

float rotation_distance(quat left, quat right) {
    left = left / quat_length(left);
    right = right / quat_length(right);
    const float cosine = std::clamp(
        std::abs(quat_dot(left, right)), 0.0F, 1.0F);
    return 2.0F * std::acos(cosine);
}

float maximum_right_arm_step(
    const interaction::Pose& previous,
    const interaction::Pose& current) {
    float maximum = 0.0F;
    for (int32_t bone = g1_skeleton::RightShoulderPitch;
         bone <= g1_skeleton::RightWrist;
         ++bone) {
        maximum = std::max(
            maximum,
            rotation_distance(
                previous.rotations[static_cast<size_t>(bone)],
                current.rotations[static_cast<size_t>(bone)]));
    }
    return maximum;
}

bool near(
    interaction::Transform left,
    interaction::Transform right,
    float tolerance = kTolerance) {
    return near(left.position, right.position, tolerance) &&
           near(left.rotation, right.rotation, tolerance);
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(
    interaction::Transform left,
    interaction::Transform right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool exact(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!exact(left.positions[bone], right.positions[bone]) ||
            !exact(left.velocities[bone], right.velocities[bone]) ||
            !exact(left.rotations[bone], right.rotations[bone]) ||
            !exact(
                left.angular_velocities[bone],
                right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
           left.hand_dof_velocities == right.hand_dof_velocities &&
           left.foot_contacts == right.foot_contacts;
}

template<class Exception, class Function>
bool throws_as(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

size_t hand_bone(interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

interaction::Transform hand_world(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t bone = hand_bone(hand);
    return {world.positions[bone], world.rotations[bone]};
}

interaction::Transform root_world(const interaction::Pose& pose) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    return {
        world.positions[g1_skeleton::Simulation],
        world.rotations[g1_skeleton::Simulation],
    };
}

interaction::Transform object_world_from_hold_pose(
    const interaction::Pose& hold,
    interaction::Hand hand,
    const interaction::GraspAffordance& affordance) {
    return interaction::compose(
        hand_world(hold, hand), interaction::inverse(affordance.hand_in_object));
}

float hand_error(
    const interaction::Pose& pose,
    interaction::Hand hand,
    const interaction::GraspAffordance& affordance,
    interaction::Transform object_world) {
    const interaction::Transform expected = interaction::compose(
        object_world, affordance.hand_in_object);
    return length(expected.position - hand_world(pose, hand).position);
}

const interaction::GraspAffordance& fixture_affordance(
    const interaction::RuntimeFixture& fixture) {
    const interaction::GraspAffordance* affordance =
        fixture.registry.find_affordance(
            fixture.request.target, fixture.request.affordance_id);
    assert(affordance != nullptr);
    return *affordance;
}

interaction::CarryRanges no_recorded_ranges() {
    return {};
}

interaction::CarryConfig short_range_config() {
    interaction::CarryConfig config{};
    config.minimum_hold_frames = 5;
    config.minimum_root_displacement_m = 0.01F;
    config.minimum_average_speed_mps = 0.01F;
    return config;
}

interaction::Pose final_hold_pose(const interaction::RuntimeFixture& fixture) {
    return interaction::pose_at_frame(
        fixture.database,
        fixture.database.range_stops.at(1) - 1);
}

void write_bone_rotation(
    interaction::Database& database,
    int32_t frame,
    size_t bone,
    quat value) {
    const size_t offset =
        (static_cast<size_t>(frame) * g1_skeleton::BoneCount + bone) * 4U;
    database.rotations.at(offset) = value.w;
    database.rotations.at(offset + 1U) = value.x;
    database.rotations.at(offset + 2U) = value.y;
    database.rotations.at(offset + 3U) = value.z;
}

void write_object_rotation(
    interaction::Database& database,
    int32_t frame,
    quat value) {
    const size_t offset = static_cast<size_t>(frame) * 4U;
    database.object_rotations.at(offset) = value.w;
    database.object_rotations.at(offset + 1U) = value.x;
    database.object_rotations.at(offset + 2U) = value.y;
    database.object_rotations.at(offset + 3U) = value.z;
}

void write_object_transform(
    interaction::Database& database,
    int32_t frame,
    interaction::Transform value) {
    interaction::runtime_fixture_detail::write_vec3(
        database.object_positions,
        static_cast<size_t>(frame),
        value.position);
    write_object_rotation(database, frame, value.rotation);
}

interaction::Transform object_at_frame(
    const interaction::Database& database,
    int32_t frame) {
    const vec3 position = interaction::runtime_fixture_detail::read_vec3(
        database.object_positions, static_cast<size_t>(frame));
    const size_t offset = static_cast<size_t>(frame) * 4U;
    return {
        position,
        quat(
            database.object_rotations.at(offset),
            database.object_rotations.at(offset + 1U),
            database.object_rotations.at(offset + 2U),
            database.object_rotations.at(offset + 3U)),
    };
}

float yaw_radians(quat rotation) {
    const vec3 facing = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float shortest_angle(float value) {
    return std::atan2(std::sin(value), std::cos(value));
}

interaction::Transform planar_alignment(
    interaction::Transform source_root,
    interaction::Transform live_root) {
    const float yaw = shortest_angle(
        yaw_radians(live_root.rotation) -
        yaw_radians(source_root.rotation));
    const quat rotation = quat_from_angle_axis(
        yaw, vec3(0.0F, 1.0F, 0.0F));
    const vec3 rotated = quat_mul_vec3(rotation, source_root.position);
    return {
        vec3(
            live_root.position.x - rotated.x,
            0.0F,
            live_root.position.z - rotated.z),
        rotation,
    };
}

void mark_source_frames(interaction::Database& database) {
    for (uint32_t frame = 0; frame < database.frame_count; ++frame) {
        database.hand_dof.at(static_cast<size_t>(frame) * 14U) =
            static_cast<float>(frame);
    }
}

void shift_recorded_hand_and_object(
    interaction::Database& database,
    int32_t start_frame,
    int32_t stop_frame,
    vec3 offset) {
    for (int32_t frame = start_frame; frame < stop_frame; ++frame) {
        vec3 hand = interaction::runtime_fixture_detail::read_bone_position(
            database, frame, interaction::kRightHandBone);
        interaction::runtime_fixture_detail::write_bone_position(
            database,
            frame,
            interaction::kRightHandBone,
            hand + offset);
        vec3 object = interaction::runtime_fixture_detail::read_vec3(
            database.object_positions, static_cast<size_t>(frame));
        interaction::runtime_fixture_detail::write_vec3(
            database.object_positions,
            static_cast<size_t>(frame),
            object + offset);
    }
}

interaction::NormalizedQuery carry_query(
    const interaction::Features& features,
    const interaction::LocomotionSnapshot& locomotion,
    interaction::Hand hand,
    interaction::Transform object_world,
    const interaction::GraspAffordance& affordance) {
    interaction::QueryInput input{};
    input.locomotion = locomotion;
    input.grasp_world = interaction::compose(
        object_world, affordance.hand_in_object);
    input.table_world = {vec3(), quat()};
    input.table_size = vec3(1.0F, 1.0F, 1.0F);
    input.approach_direction_object = vec3(0.0F, 0.0F, -1.0F);
    input.object_dimensions = vec3(0.1F, 0.1F, 0.1F);
    input.hand = hand;
    return interaction::normalize_query(
        interaction::build_raw_query(input), features);
}

void set_pose_trajectory_row(
    interaction::Features& features,
    int32_t frame,
    const interaction::NormalizedQuery& query,
    float delta) {
    for (size_t dimension = interaction::kFeatureGroupStarts[0];
         dimension < interaction::kFeatureGroupStops[1];
         ++dimension) {
        features.values.at(
            static_cast<size_t>(frame) * interaction::kFeatureDimension +
            dimension) = query[dimension] + delta;
    }
}

void set_other_feature_groups_row(
    interaction::Features& features,
    int32_t frame,
    const interaction::NormalizedQuery& query,
    float delta) {
    for (size_t dimension = interaction::kFeatureGroupStarts[2];
         dimension < interaction::kFeatureDimension;
         ++dimension) {
        features.values.at(
            static_cast<size_t>(frame) * interaction::kFeatureDimension +
            dimension) = query[dimension] + delta;
    }
}

std::vector<interaction::CarryRange> ranges_for_clip(
    const interaction::CarryRanges& ranges,
    int32_t clip) {
    std::vector<interaction::CarryRange> selected;
    for (const interaction::CarryRange& range : ranges.recorded) {
        if (range.clip == clip) selected.push_back(range);
    }
    return selected;
}

bool contains_clip(const std::vector<int32_t>& clips, int32_t clip) {
    for (int32_t value : clips) {
        if (value == clip) return true;
    }
    return false;
}

void assert_range(
    const interaction::CarryRange& range,
    int32_t clip,
    int32_t start,
    int32_t stop,
    interaction::Hand hand) {
    assert(range.clip == clip);
    assert(range.start_frame == start);
    assert(range.stop_frame == stop);
    assert(range.hand == hand);
}

void test_frozen_public_interface_and_defaults() {
    using namespace interaction;
    static_assert(std::is_same_v<
        decltype(&classify_carry_ranges),
        CarryRanges (*)(const Database&, const CarryConfig&)>);
    static_assert(std::is_constructible_v<
        CarryController,
        const Database&,
        const Features&,
        CarryRanges,
        CarryConfig,
        IKConfig>);
    static_assert(std::is_same_v<
        decltype(&CarryController::start),
        void (CarryController::*)(
            const Pose&, Hand, const GraspAffordance&, Transform)>);
    static_assert(std::is_same_v<
        decltype(&CarryController::update),
        Pose (CarryController::*)(const LocomotionSnapshot&, float)>);
    static_assert(std::is_same_v<
        decltype(&CarryController::recorded),
        bool (CarryController::*)() const>);
    static_assert(std::is_same_v<
        decltype(&CarryController::object_world),
        Transform (CarryController::*)() const>);

    const CarryConfig config{};
    assert(config.minimum_hold_frames == 25);
    assert(near(config.maximum_grasp_drift_m, 0.02F));
    assert(near(config.maximum_grasp_drift_radians, 0.174532925F));
    assert(near(config.minimum_root_displacement_m, 0.30F));
    assert(near(config.minimum_average_speed_mps, 0.20F));
    assert(near(config.search_interval_seconds, 0.10F));
    assert(near(config.spine_weight, 0.25F));
    assert(near(config.inactive_arm_weight, 0.0F));

    const CarryRange range{};
    assert(range.clip == -1);
    assert(range.start_frame == -1);
    assert(range.stop_frame == -1);
    assert(range.hand == Hand::Right);
}

void test_fixture_classification_is_exact() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();

    const CarryRanges ranges = classify_carry_ranges(fixture.database);

    assert(ranges.recorded.size() == 1U);
    assert(ranges.rejected.size() == 1U);
    const CarryRange& range = ranges.recorded.front();
    assert(range.clip == 1);
    assert(range.start_frame == 115);
    assert(range.stop_frame == 150);
    assert(range.hand == Hand::Right);
    assert(ranges.rejected == std::vector<int32_t>{0});
}

void test_classification_thresholds_are_inclusive() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const int32_t start = fixture.database.range_starts.at(1) +
        runtime_fixture_detail::kHoldLocalFrame;
    const int32_t stop = fixture.database.range_stops.at(1);
    const float database_fps =
        static_cast<float>(fixture.database.fps_numerator) /
        static_cast<float>(fixture.database.fps_denominator);
    const Pose first = pose_at_frame(fixture.database, start);
    const Pose last = pose_at_frame(fixture.database, stop - 1);
    const vec3 first_root = first.positions[g1_skeleton::Simulation];
    const vec3 last_root = last.positions[g1_skeleton::Simulation];
    const float endpoint = std::hypot(
        last_root.x - first_root.x, last_root.z - first_root.z);
    float path = 0.0F;
    vec3 previous = first_root;
    for (int32_t frame = start + 1; frame < stop; ++frame) {
        const vec3 root = pose_at_frame(fixture.database, frame)
            .positions[g1_skeleton::Simulation];
        path += std::hypot(root.x - previous.x, root.z - previous.z);
        previous = root;
    }

    CarryConfig inclusive{};
    inclusive.minimum_hold_frames = stop - start;
    inclusive.minimum_root_displacement_m = endpoint;
    inclusive.minimum_average_speed_mps =
        path / (static_cast<float>(stop - start - 1) / database_fps);
    const CarryRanges accepted = classify_carry_ranges(
        fixture.database, inclusive);
    assert(ranges_for_clip(accepted, 1).size() == 1U);
    assert(!contains_clip(accepted.rejected, 1));

    CarryConfig too_many = inclusive;
    too_many.minimum_hold_frames += 1;
    const CarryRanges rejected = classify_carry_ranges(
        fixture.database, too_many);
    assert(ranges_for_clip(rejected, 1).empty());
    assert(contains_clip(rejected.rejected, 1));
}

void test_classification_partitions_contact_and_drift_runs() {
    using namespace interaction;
    RuntimeFixture contact_fixture = make_runtime_fixture();
    constexpr int32_t kSplit =
        runtime_fixture_detail::kFramesPerClip + 57;
    contact_fixture.database.hand_contacts.at(
        static_cast<size_t>(kSplit) * 2U + 1U) = 0U;
    CarryConfig permissive{};
    permissive.minimum_hold_frames = 5;
    permissive.minimum_root_displacement_m = 0.01F;
    permissive.minimum_average_speed_mps = 0.01F;
    const CarryRanges contact_ranges = classify_carry_ranges(
        contact_fixture.database, permissive);
    const std::vector<CarryRange> contacted = ranges_for_clip(
        contact_ranges, 1);
    assert(contacted.size() == 2U);
    assert_range(contacted[0], 1, 115, kSplit, Hand::Right);
    assert_range(contacted[1], 1, kSplit + 1, 150, Hand::Right);
    assert(!contains_clip(contact_ranges.rejected, 1));

    RuntimeFixture drift_fixture = make_runtime_fixture();
    for (int32_t frame = kSplit; frame < 150; ++frame) {
        vec3 hand = runtime_fixture_detail::read_bone_position(
            drift_fixture.database, frame, kRightHandBone);
        hand.x += permissive.maximum_grasp_drift_m + 0.001F;
        runtime_fixture_detail::write_bone_position(
            drift_fixture.database, frame, kRightHandBone, hand);
    }
    const CarryRanges drift_ranges = classify_carry_ranges(
        drift_fixture.database, permissive);
    const std::vector<CarryRange> drifted = ranges_for_clip(
        drift_ranges, 1);
    assert(drifted.size() == 2U);
    assert_range(drifted[0], 1, 115, kSplit, Hand::Right);
    assert_range(drifted[1], 1, kSplit, 150, Hand::Right);
    assert(!contains_clip(drift_ranges.rejected, 1));

    RuntimeFixture rotation_fixture = make_runtime_fixture();
    const quat excessive_rotation = quat_from_angle_axis(
        permissive.maximum_grasp_drift_radians + 0.01F,
        vec3(0.0F, 1.0F, 0.0F));
    for (int32_t frame = kSplit; frame < 150; ++frame) {
        write_bone_rotation(
            rotation_fixture.database,
            frame,
            kRightHandBone,
            excessive_rotation);
    }
    const CarryRanges rotation_ranges = classify_carry_ranges(
        rotation_fixture.database, permissive);
    const std::vector<CarryRange> rotated = ranges_for_clip(
        rotation_ranges, 1);
    assert(rotated.size() == 2U);
    assert_range(rotated[0], 1, 115, kSplit, Hand::Right);
    assert_range(rotated[1], 1, kSplit, 150, Hand::Right);
}

void test_grasp_threshold_and_quaternion_sign_do_not_split() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    constexpr int32_t kShift =
        runtime_fixture_detail::kFramesPerClip + 57;
    CarryConfig config{};
    config.minimum_hold_frames = 5;
    config.minimum_root_displacement_m = 0.01F;
    config.minimum_average_speed_mps = 0.01F;
    const quat threshold_rotation = quat_from_angle_axis(
        config.maximum_grasp_drift_radians,
        vec3(0.0F, 1.0F, 0.0F));
    for (int32_t frame = kShift; frame < 150; ++frame) {
        vec3 hand = runtime_fixture_detail::read_bone_position(
            fixture.database, frame, kRightHandBone);
        hand.x += config.maximum_grasp_drift_m;
        runtime_fixture_detail::write_bone_position(
            fixture.database, frame, kRightHandBone, hand);
        write_bone_rotation(
            fixture.database,
            frame,
            kRightHandBone,
            threshold_rotation);
    }
    write_bone_rotation(
        fixture.database, kShift + 3, kRightHandBone,
        -threshold_rotation);
    write_object_rotation(
        fixture.database, kShift + 3,
        quat(-1.0F, 0.0F, 0.0F, 0.0F));

    const CarryRanges ranges = classify_carry_ranges(
        fixture.database, config);
    const std::vector<CarryRange> clip = ranges_for_clip(ranges, 1);
    assert(clip.size() == 1U);
    assert_range(clip.front(), 1, 115, 150, Hand::Right);
}

void test_average_speed_uses_planar_path_length() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    constexpr int32_t kStart = 115;
    constexpr int32_t kTurning = 132;
    constexpr int32_t kStop = 150;
    for (int32_t frame = kStart; frame < kStop; ++frame) {
        const float x = frame <= kTurning
            ? 0.50F * static_cast<float>(frame - kStart) /
                static_cast<float>(kTurning - kStart)
            : 0.50F - 0.20F * static_cast<float>(frame - kTurning) /
                static_cast<float>(kStop - 1 - kTurning);
        vec3 root = runtime_fixture_detail::read_bone_position(
            fixture.database, frame, g1_skeleton::Simulation);
        root.x = x;
        runtime_fixture_detail::write_bone_position(
            fixture.database, frame, g1_skeleton::Simulation, root);
        vec3 object = runtime_fixture_detail::read_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame));
        object.x = x;
        runtime_fixture_detail::write_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame),
            object);
    }

    CarryConfig config{};
    config.minimum_root_displacement_m = 0.29F;
    config.minimum_average_speed_mps = 0.40F;
    const CarryRanges ranges = classify_carry_ranges(
        fixture.database, config);
    const std::vector<CarryRange> clip = ranges_for_clip(ranges, 1);
    assert(clip.size() == 1U);
    assert_range(clip.front(), 1, kStart, kStop, Hand::Right);
}

void test_fallback_preserves_locomotion_and_grasp() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);

    CarryController fallback(
        fixture.database, fixture.features, no_recorded_ranges());
    assert(throws_as<std::logic_error>([&] {
        (void)fallback.update(locomotion, 1.0F / 60.0F);
    }));

    fallback.start(hold, Hand::Right, affordance, object);
    const Pose output = fallback.update(locomotion, 1.0F / 60.0F);

    assert(!fallback.recorded());
    assert(near(
        output.rotations[g1_skeleton::RightShoulderPitch],
        hold.rotations[g1_skeleton::RightShoulderPitch]));
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        locomotion.pose.rotations[g1_skeleton::LeftHipPitch]));
    assert(hand_error(
        output, Hand::Right, affordance, fallback.object_world()) <= 0.04F);
}

void test_default_layered_carry_releases_inactive_arm_after_hold_seam() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;

    for (int32_t bone = g1_skeleton::LeftShoulderPitch;
         bone <= g1_skeleton::LeftWrist;
         ++bone) {
        const size_t index = static_cast<size_t>(bone);
        locomotion.pose.rotations[index] = quat_mul(
            hold.rotations[index],
            quat_from_angle_axis(0.75F, vec3(0.0F, 0.0F, 1.0F)));
        locomotion.pose.velocities[index] =
            vec3(0.01F * bone, -0.02F * bone, 0.03F * bone);
        locomotion.pose.angular_velocities[index] =
            vec3(-0.03F * bone, 0.02F * bone, -0.01F * bone);
    }

    CarryController controller(
        fixture.database, fixture.features, no_recorded_ranges());
    controller.start(hold, Hand::Right, affordance, initial_object);

    Pose previous = controller.update(locomotion, 0.0F);
    Transform previous_object = controller.object_world();
    for (int32_t bone = g1_skeleton::LeftShoulderPitch;
         bone <= g1_skeleton::LeftWrist;
         ++bone) {
        const size_t index = static_cast<size_t>(bone);
        assert(near(previous.rotations[index], hold.rotations[index], 2.0e-4F));
        assert(near(
            previous.velocities[index], hold.velocities[index], 2.0e-5F));
        assert(near(
            previous.angular_velocities[index],
            hold.angular_velocities[index],
            2.0e-5F));
    }
    assert(hand_error(
        previous,
        Hand::Right,
        affordance,
        previous_object) <= IKConfig{}.accepted_position_m);
    assert(rotation_distance(
        hand_world(previous, Hand::Right).rotation,
        compose(previous_object, affordance.hand_in_object).rotation) <=
        IKConfig{}.accepted_orientation_radians);

    for (int tick = 0; tick < 10; ++tick) {
        const Pose current = controller.update(locomotion, 0.05F);
        const Transform current_object = controller.object_world();
        for (int32_t bone = g1_skeleton::LeftShoulderPitch;
             bone <= g1_skeleton::LeftWrist;
             ++bone) {
            const size_t index = static_cast<size_t>(bone);
            assert(rotation_distance(
                previous.rotations[index], current.rotations[index]) <= 0.12F);
        }
        assert(hand_error(
            current,
            Hand::Right,
            affordance,
            current_object) <= IKConfig{}.accepted_position_m);
        assert(rotation_distance(
            hand_world(current, Hand::Right).rotation,
            compose(current_object, affordance.hand_in_object).rotation) <=
            IKConfig{}.accepted_orientation_radians);
        assert(length(
            current_object.position - previous_object.position) <=
            CarryConfig{}.maximum_grasp_drift_m + 1.0e-5F);
        assert(rotation_distance(
            current_object.rotation, previous_object.rotation) <=
            CarryConfig{}.maximum_grasp_drift_radians + 1.0e-5F);
        previous = current;
        previous_object = current_object;
    }

    // The exact seam endpoint already belongs to live locomotion; it must not
    // retain normalization or blending residue from the Hold source.
    for (int32_t bone = g1_skeleton::LeftShoulderPitch;
         bone <= g1_skeleton::LeftWrist;
         ++bone) {
        const size_t index = static_cast<size_t>(bone);
        assert(exact(previous.positions[index], locomotion.pose.positions[index]));
        assert(exact(previous.velocities[index], locomotion.pose.velocities[index]));
        assert(exact(previous.rotations[index], locomotion.pose.rotations[index]));
        assert(exact(
            previous.angular_velocities[index],
            locomotion.pose.angular_velocities[index]));
    }

    // One full-progress publication retires the seam. Later direct layered
    // Carry frames must continue publishing the current free arm unchanged.
    previous = controller.update(locomotion, 0.01F);
    previous_object = controller.object_world();
    for (int32_t bone = g1_skeleton::LeftShoulderPitch;
         bone <= g1_skeleton::LeftWrist;
         ++bone) {
        const size_t index = static_cast<size_t>(bone);
        const quat changed = quat_mul(
            locomotion.pose.rotations[index],
            quat_from_angle_axis(0.04F, vec3(1.0F, 0.0F, 0.0F)));
        locomotion.pose.rotations[index] = changed * 0.9995F;
        locomotion.pose.positions[index] =
            locomotion.pose.positions[index] + vec3(0.001F, 0.002F, 0.003F);
        locomotion.pose.velocities[index] =
            vec3(0.15F + bone, 0.25F + bone, 0.35F + bone);
        locomotion.pose.angular_velocities[index] =
            vec3(-0.45F - bone, -0.35F - bone, -0.25F - bone);
    }

    const Pose direct = controller.update(locomotion, 0.01F);
    const Transform direct_object = controller.object_world();
    for (int32_t bone = g1_skeleton::LeftShoulderPitch;
         bone <= g1_skeleton::LeftWrist;
         ++bone) {
        const size_t index = static_cast<size_t>(bone);
        assert(exact(direct.positions[index], locomotion.pose.positions[index]));
        assert(exact(direct.velocities[index], locomotion.pose.velocities[index]));
        assert(exact(direct.rotations[index], locomotion.pose.rotations[index]));
        assert(exact(
            direct.angular_velocities[index],
            locomotion.pose.angular_velocities[index]));
    }
    assert(hand_error(
        direct,
        Hand::Right,
        affordance,
        direct_object) <= IKConfig{}.accepted_position_m);
    assert(rotation_distance(
        hand_world(direct, Hand::Right).rotation,
        compose(direct_object, affordance.hand_in_object).rotation) <=
        IKConfig{}.accepted_orientation_radians);
    assert(length(direct_object.position - previous_object.position) <=
        CarryConfig{}.maximum_grasp_drift_m + 1.0e-5F);
    assert(rotation_distance(
        direct_object.rotation, previous_object.rotation) <=
        CarryConfig{}.maximum_grasp_drift_radians + 1.0e-5F);
}

void test_fallback_rotation_masks_are_layered() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;

    const quat locomotion_rotation = quat_from_angle_axis(
        -0.30F, vec3(0.0F, 1.0F, 0.0F));
    const quat hold_rotation = quat_from_angle_axis(
        0.45F, vec3(0.0F, 1.0F, 0.0F));
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        locomotion.pose.rotations[bone] = locomotion_rotation;
        hold.rotations[bone] = hold_rotation;
    }

    IKConfig accepting_ik{};
    accepting_ik.maximum_request_position_m = 100.0F;
    accepting_ik.maximum_request_orientation_radians = 10.0F;
    accepting_ik.accepted_position_m = 100.0F;
    accepting_ik.accepted_orientation_radians = 10.0F;
    CarryConfig accepting_carry{};
    accepting_carry.maximum_grasp_drift_m = 100.0F;
    accepting_carry.maximum_grasp_drift_radians = 10.0F;
    accepting_carry.inactive_arm_weight = 0.35F;
    CarryController fallback(
        fixture.database,
        fixture.features,
        no_recorded_ranges(),
        accepting_carry,
        accepting_ik);
    fallback.start(
        hold,
        Hand::Right,
        affordance,
        object_world_from_hold_pose(hold, Hand::Right, affordance));

    Pose output{};
    for (int tick = 0; tick < 32; ++tick) {
        output = fallback.update(locomotion, 1.0F / 60.0F);
    }
    for (int32_t bone = g1_skeleton::Simulation;
         bone <= g1_skeleton::RightToe;
         ++bone) {
        assert(near(output.rotations[bone], locomotion_rotation));
        assert(near(
            output.positions[bone], locomotion.pose.positions[bone]));
        assert(near(
            output.velocities[bone], locomotion.pose.velocities[bone]));
        assert(near(
            output.angular_velocities[bone],
            locomotion.pose.angular_velocities[bone]));
    }
    for (int32_t bone : {g1_skeleton::Spine,
                         g1_skeleton::Spine1,
                         g1_skeleton::Spine2}) {
        assert(near(
            output.rotations[bone],
            quat_nlerp_shortest(
                locomotion_rotation, hold_rotation, 0.25F)));
    }
    for (int32_t bone = g1_skeleton::LeftShoulderPitch;
         bone <= g1_skeleton::LeftWrist;
         ++bone) {
        assert(near(
            output.rotations[bone],
            quat_nlerp_shortest(
                locomotion_rotation, hold_rotation, 0.35F)));
        assert(exact(
            output.positions[bone], locomotion.pose.positions[bone]));
        assert(exact(
            output.velocities[bone], locomotion.pose.velocities[bone]));
        assert(exact(
            output.angular_velocities[bone],
            locomotion.pose.angular_velocities[bone]));
    }
    for (int32_t bone = g1_skeleton::RightShoulderPitch;
         bone <= g1_skeleton::RightWrist;
         ++bone) {
        assert(near(output.rotations[bone], hold_rotation));
    }
    assert(output.foot_contacts == locomotion.pose.foot_contacts);
}

void test_layered_anchor_and_nonidentity_grasp_move_with_root() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    GraspAffordance affordance = fixture_affordance(fixture);
    affordance.hand_in_object = {
        vec3(0.03F, -0.02F, 0.04F),
        quat_from_angle_axis(0.30F, vec3(0.0F, 1.0F, 0.0F)),
    };
    const Pose hold = final_hold_pose(fixture);
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;

    CarryController controller(
        fixture.database, fixture.features, no_recorded_ranges());
    controller.start(hold, Hand::Right, affordance, initial_object);
    const Pose initial_output = controller.update(locomotion, 0.0F);
    assert(!controller.recorded());
    assert(near(controller.object_world(), initial_object, 2.0e-5F));
    assert(hand_error(
        initial_output,
        Hand::Right,
        affordance,
        controller.object_world()) <= 0.04F);

    const Transform initial_root = root_world(hold);
    locomotion.pose.positions[g1_skeleton::Simulation].x += 0.25F;
    locomotion.pose.positions[g1_skeleton::Simulation].z -= 0.10F;
    locomotion.pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.20F, vec3(0.0F, 1.0F, 0.0F));
    const Transform desired_object = compose(
        root_world(locomotion.pose),
        compose(inverse(initial_root), initial_object));
    const Pose moved_output = controller.update(locomotion, 0.04F);
    const Transform published = compose(
        hand_world(moved_output, Hand::Right),
        inverse(affordance.hand_in_object));
    assert(near(controller.object_world(), published, 2.0e-5F));
    assert(near(controller.object_world(), desired_object, 2.0e-4F));
}

void test_layered_carry_smooths_ik_feasible_nonarm_seam() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const quat live_hip = quat_mul(
        hold.rotations[g1_skeleton::LeftHipPitch],
        quat_from_angle_axis(0.90F, vec3(1.0F, 0.0F, 0.0F)));
    locomotion.pose.rotations[g1_skeleton::LeftHipPitch] = live_hip;
    locomotion.pose.positions[g1_skeleton::Simulation].x += 0.04F;

    CarryController controller(
        fixture.database, fixture.features, no_recorded_ranges());
    controller.start(hold, Hand::Right, affordance, initial_object);

    Pose output = controller.update(locomotion, 0.04F);

    assert(!controller.recorded());
    assert(near(root_world(output), root_world(locomotion.pose), 2.0e-5F));
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        quat_nlerp_shortest(
            hold.rotations[g1_skeleton::LeftHipPitch],
            live_hip,
            0.08F),
        2.0e-4F));
    assert(!near(
        output.rotations[g1_skeleton::LeftHipPitch], live_hip, 0.10F));
    assert(hand_error(
        output,
        Hand::Right,
        affordance,
        controller.object_world()) <= IKConfig{}.accepted_position_m);

    for (int tick = 0; tick < 12; ++tick) {
        locomotion.pose.positions[g1_skeleton::Simulation].x += 0.02F;
        output = controller.update(locomotion, 0.04F);
        assert(near(
            root_world(output), root_world(locomotion.pose), 2.0e-5F));
        assert(hand_error(
            output,
            Hand::Right,
            affordance,
            controller.object_world()) <= IKConfig{}.accepted_position_m);
    }

    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch], live_hip, 2.0e-4F));
    assert(std::abs(
        controller.object_world().position.x - initial_object.position.x) >
        0.20F);
    const Pose direct = controller.update(locomotion, 0.04F);
    assert(near(
        direct.rotations[g1_skeleton::LeftHipPitch],
        output.rotations[g1_skeleton::LeftHipPitch],
        2.0e-4F));
}

void test_layered_carry_inertializes_hold_to_live_seam_without_stalling_root() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const Transform initial_root = root_world(hold);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    locomotion.pose.rotations[g1_skeleton::Hips] = quat_mul(
        locomotion.pose.rotations[g1_skeleton::Hips],
        quat_from_angle_axis(0.60F, vec3(0.0F, 1.0F, 0.0F)));
    IKConfig setup_ik{};
    setup_ik.maximum_request_position_m = 2.0F;
    setup_ik.maximum_request_orientation_radians = 4.0F;
    setup_ik.accepted_position_m = 0.01F;
    setup_ik.accepted_orientation_radians = 0.05F;
    setup_ik.maximum_iterations = 64;
    setup_ik.maximum_step_radians = 0.20F;
    const IKResult setup_result = solve_hand_ik(
        locomotion.pose,
        Hand::Right,
        hand_world(hold, Hand::Right),
        setup_ik);
    assert(setup_result.accepted);
    locomotion.pose.positions[g1_skeleton::Simulation].x += 0.04F;

    CarryController controller(
        fixture.database, fixture.features, no_recorded_ranges());
    controller.start(hold, Hand::Right, affordance, initial_object);

    const Transform first_expected_object = compose(
        root_world(locomotion.pose),
        compose(inverse(initial_root), initial_object));
    const Pose first = controller.update(locomotion, 0.04F);

    assert(!controller.recorded());
    assert(near(root_world(first), root_world(locomotion.pose), 2.0e-5F));
    assert(length(
        controller.object_world().position -
        first_expected_object.position) <=
        CarryConfig{}.maximum_grasp_drift_m);
    assert(std::abs(
        controller.object_world().position.y - initial_object.position.y) <=
        CarryConfig{}.maximum_grasp_drift_m);
    assert(hand_error(
        first,
        Hand::Right,
        affordance,
        controller.object_world()) <= IKConfig{}.accepted_position_m);

    for (int tick = 0; tick < 120; ++tick) {
        locomotion.pose.positions[g1_skeleton::Simulation].x += 0.01F;
        const Pose output = controller.update(locomotion, 0.04F);
        assert(near(
            root_world(output), root_world(locomotion.pose), 2.0e-5F));
        assert(hand_error(
            output,
            Hand::Right,
            affordance,
            controller.object_world()) <= IKConfig{}.accepted_position_m);
        assert(std::abs(
            controller.object_world().position.y -
            initial_object.position.y) <=
            CarryConfig{}.maximum_grasp_drift_m);
    }

    assert(std::abs(
        controller.object_world().position.x - initial_object.position.x) >
        0.20F);
    const Pose final = controller.update(locomotion, 0.04F);
    assert(near(
        final.rotations[g1_skeleton::Hips],
        locomotion.pose.rotations[g1_skeleton::Hips],
        0.01F));
}

void test_rejected_layered_ik_preserves_last_safe_nonroot_pose_and_live_anchor() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);

    IKConfig bounded{};
    bounded.maximum_request_position_m = 1.0e-5F;
    bounded.accepted_position_m = 1.0e-5F;
    CarryController controller(
        fixture.database,
        fixture.features,
        no_recorded_ranges(),
        CarryConfig{},
        bounded);
    controller.start(
        hold, Hand::Right, affordance, initial_object);
    const Pose safe_pose = controller.update(locomotion, 0.0F);
    const Transform safe_object = controller.object_world();
    assert(!controller.recorded());

    LocomotionSnapshot infeasible = locomotion;
    infeasible.pose = safe_pose;
    infeasible.pose.positions[g1_skeleton::Simulation].x += 0.25F;
    infeasible.pose.positions[g1_skeleton::RightShoulderPitch].x += 0.50F;
    const Transform desired_object = compose(
        root_world(infeasible.pose),
        compose(inverse(root_world(safe_pose)), safe_object));
    const Transform unsolved_publication = compose(
        hand_world(infeasible.pose, Hand::Right),
        inverse(affordance.hand_in_object));
    assert(length(
        compose(desired_object, affordance.hand_in_object).position -
        hand_world(infeasible.pose, Hand::Right).position) >
        bounded.maximum_request_position_m);
    assert(!near(unsolved_publication, safe_object, 0.10F));

    const Pose rejected = controller.update(infeasible, 1.0F / 60.0F);

    assert(!controller.recorded());
    assert(near(
        root_world(rejected), root_world(infeasible.pose), 2.0e-5F));
    assert(exact(
        rejected.positions[g1_skeleton::RightShoulderPitch],
        safe_pose.positions[g1_skeleton::RightShoulderPitch]));
    assert(exact(
        rejected.rotations[g1_skeleton::RightShoulderPitch],
        safe_pose.rotations[g1_skeleton::RightShoulderPitch]));
    assert(near(controller.object_world(), desired_object, 2.0e-5F));
    assert(!near(controller.object_world(), unsolved_publication, 0.10F));
}

void test_lifecycle_and_invalid_inputs_are_defensive() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);

    CarryConfig invalid_config{};
    invalid_config.search_interval_seconds = 0.0F;
    assert(throws_as<std::invalid_argument>([&] {
        CarryController invalid(
            fixture.database,
            fixture.features,
            no_recorded_ranges(),
            invalid_config);
        (void)invalid;
    }));

    Features invalid_features = fixture.features;
    invalid_features.values.pop_back();
    assert(throws_as<std::invalid_argument>([&] {
        CarryController invalid(
            fixture.database, invalid_features, no_recorded_ranges());
        (void)invalid;
    }));

    CarryRanges invalid_ranges{};
    invalid_ranges.recorded.push_back({1, 115, 151, Hand::Right});
    assert(throws_as<std::invalid_argument>([&] {
        CarryController invalid(
            fixture.database, fixture.features, invalid_ranges);
        (void)invalid;
    }));

    CarryRanges uncertified_range{};
    uncertified_range.recorded.push_back({1, 115, 120, Hand::Right});
    assert(throws_as<std::invalid_argument>([&] {
        CarryController invalid(
            fixture.database, fixture.features, uncertified_range);
        (void)invalid;
    }));

    Database noncanonical_fps = fixture.database;
    noncanonical_fps.fps_numerator = 50U;
    assert(throws_as<std::invalid_argument>([&] {
        CarryController invalid(
            noncanonical_fps,
            fixture.features,
            no_recorded_ranges());
        (void)invalid;
    }));

    RuntimeFixture unstable_fixture = make_runtime_fixture();
    for (int32_t frame = 130; frame < 150; ++frame) {
        vec3 hand = runtime_fixture_detail::read_bone_position(
            unstable_fixture.database, frame, kRightHandBone);
        hand.x += 0.03F;
        runtime_fixture_detail::write_bone_position(
            unstable_fixture.database, frame, kRightHandBone, hand);
    }
    CarryRanges unstable_range{};
    unstable_range.recorded.push_back({1, 115, 150, Hand::Right});
    assert(throws_as<std::invalid_argument>([&] {
        CarryController invalid(
            unstable_fixture.database,
            unstable_fixture.features,
            unstable_range);
        (void)invalid;
    }));

    CarryController controller(
        fixture.database, fixture.features, no_recorded_ranges());
    assert(throws_as<std::logic_error>([&] {
        (void)controller.update(fixture.locomotion, 0.0F);
    }));

    GraspAffordance wrong_hand = affordance;
    wrong_hand.hand = Hand::Left;
    assert(throws_as<std::invalid_argument>([&] {
        controller.start(hold, Hand::Right, wrong_hand, object);
    }));
    Transform invalid_object = object;
    invalid_object.rotation = quat(0.0F, 0.0F, 0.0F, 0.0F);
    assert(throws_as<std::invalid_argument>([&] {
        controller.start(hold, Hand::Right, affordance, invalid_object);
    }));

    Pose invalid_hold = hold;
    invalid_hold.positions[g1_skeleton::Simulation].x =
        std::numeric_limits<float>::quiet_NaN();
    assert(throws_as<std::invalid_argument>([&] {
        controller.start(invalid_hold, Hand::Right, affordance, object);
    }));

    controller.start(hold, Hand::Right, affordance, object);
    assert(throws_as<std::invalid_argument>([&] {
        (void)controller.update(fixture.locomotion, -0.001F);
    }));
    assert(throws_as<std::invalid_argument>([&] {
        (void)controller.update(
            fixture.locomotion,
            std::numeric_limits<float>::quiet_NaN());
    }));
    LocomotionSnapshot invalid_locomotion = fixture.locomotion;
    invalid_locomotion.pose.rotations[g1_skeleton::Simulation] =
        quat(0.0F, 0.0F, 0.0F, 0.0F);
    assert(throws_as<std::invalid_argument>([&] {
        (void)controller.update(invalid_locomotion, 0.0F);
    }));
    (void)controller.update(fixture.locomotion, 0.0F);

    Transform second_object = object;
    second_object.position.x += 0.10F;
    controller.start(hold, Hand::Right, affordance, second_object);
    assert(!controller.recorded());
    assert(near(controller.object_world(), second_object));
}

void test_recorded_search_uses_pose_trajectory_and_aligns_object() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    mark_source_frames(fixture.database);
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    locomotion.pose.positions[g1_skeleton::Simulation].x += 0.60F;
    locomotion.pose.positions[g1_skeleton::Simulation].z -= 0.20F;
    locomotion.pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.30F, vec3(0.0F, 1.0F, 0.0F));

    const NormalizedQuery query = carry_query(
        fixture.features,
        locomotion,
        Hand::Right,
        initial_object,
        affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
        set_other_feature_groups_row(fixture.features, frame, query, 0.0F);
    }
    set_pose_trajectory_row(fixture.features, 115, query, 0.0F);
    set_other_feature_groups_row(fixture.features, 115, query, 100.0F);
    set_pose_trajectory_row(fixture.features, 116, query, 0.50F);
    set_other_feature_groups_row(fixture.features, 116, query, 0.0F);

    CarryConfig config{};
    config.search_interval_seconds = 1.0F;
    CarryController controller(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database),
        config);
    controller.start(hold, Hand::Right, affordance, initial_object);
    Pose output = controller.update(locomotion, 0.0F);

    for (int32_t tick = 0; tick < 13; ++tick) {
        output = controller.update(locomotion, 0.04F);
    }
    output = controller.update(locomotion, 0.0F);

    assert(controller.recorded());
    assert(near(output.hand_dof[0], 128.0F));
    const Transform source_root = root_world(
        pose_at_frame(fixture.database, 128));
    const Transform alignment = planar_alignment(
        source_root, root_world(locomotion.pose));
    const Transform expected_object = compose(
        alignment, object_at_frame(fixture.database, 128));
    assert(near(
        output.positions[g1_skeleton::Simulation].x,
        locomotion.pose.positions[g1_skeleton::Simulation].x));
    assert(near(
        output.positions[g1_skeleton::Simulation].z,
        locomotion.pose.positions[g1_skeleton::Simulation].z));
    assert(near(
        shortest_angle(
            yaw_radians(output.rotations[g1_skeleton::Simulation]) -
            yaw_radians(locomotion.pose.rotations[g1_skeleton::Simulation])),
        0.0F));
    assert(near(controller.object_world(), expected_object, 2.0e-4F));
}

void test_initial_recorded_carry_smooths_nonarm_seam() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const quat recorded_hip = quat_mul(
        hold.rotations[g1_skeleton::LeftHipPitch],
        quat_from_angle_axis(0.75F, vec3(1.0F, 0.0F, 0.0F)));
    for (int32_t frame = 115; frame < 150; ++frame) {
        write_bone_rotation(
            fixture.database,
            frame,
            g1_skeleton::LeftHipPitch,
            recorded_hip);
    }
    const NormalizedQuery query = carry_query(
        fixture.features, locomotion, Hand::Right, object, affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 0.0F);
    }

    CarryController controller(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database));
    controller.start(hold, Hand::Right, affordance, object);

    Pose output = controller.update(locomotion, 0.04F);

    assert(controller.recorded());
    assert(near(root_world(output), root_world(locomotion.pose), 2.0e-5F));
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        quat_nlerp_shortest(
            hold.rotations[g1_skeleton::LeftHipPitch],
            recorded_hip,
            0.08F),
        2.0e-4F));
    assert(!near(
        output.rotations[g1_skeleton::LeftHipPitch], recorded_hip, 0.10F));
    assert(hand_error(
        output,
        Hand::Right,
        affordance,
        controller.object_world()) <= IKConfig{}.accepted_position_m);

    for (int tick = 0; tick < 12; ++tick) {
        output = controller.update(locomotion, 0.04F);
        assert(controller.recorded());
        assert(hand_error(
            output,
            Hand::Right,
            affordance,
            controller.object_world()) <= IKConfig{}.accepted_position_m);
    }
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        recorded_hip,
        2.0e-4F));

    const Transform completed_object = controller.object_world();
    const Pose direct = controller.update(locomotion, 0.04F);
    for (int32_t joint = g1_skeleton::RightShoulderPitch;
         joint <= g1_skeleton::RightWrist;
         ++joint) {
        const size_t bone = static_cast<size_t>(joint);
        assert(near(direct.rotations[bone], output.rotations[bone], 0.20F));
    }
    assert(length(
        controller.object_world().position - completed_object.position) <=
        CarryConfig{}.maximum_grasp_drift_m + 1.0e-4F);
}

void test_recorded_carry_keeps_active_arm_on_one_continuous_ik_branch() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery query = carry_query(
        fixture.features, locomotion, Hand::Right, object, affordance);
    const std::array<quat, 7> alternate_arm = {{
        {0.706170559F, -0.0991955474F, 0.0976584703F, 0.694223464F},
        {0.99653852F, 0.0831327066F, 0.0F, 0.0F},
        {0.891591847F, 0.0F, -0.452839911F, 0.0F},
        {0.792196512F, 0.0F, 0.0F, -0.610266089F},
        {0.943883598F, -0.330278367F, 0.0F, 0.0F},
        {0.997371793F, 0.0F, 0.0F, -0.0724532753F},
        {0.999991298F, 0.0F, 0.0041709533F, 0.0F},
    }};
    constexpr float kRecordedMarker = 42.0F;

    for (int32_t frame = 115; frame < 150; ++frame) {
        for (size_t bone = 1U; bone < g1_skeleton::BoneCount; ++bone) {
            write_bone_rotation(
                fixture.database, frame, bone, hold.rotations[bone]);
        }
        for (size_t joint = 0U; joint < alternate_arm.size(); ++joint) {
            write_bone_rotation(
                fixture.database,
                frame,
                static_cast<size_t>(g1_skeleton::RightShoulderPitch) + joint,
                alternate_arm[joint]);
        }
        fixture.database.hand_dof.at(
            static_cast<size_t>(frame) * 14U) = kRecordedMarker;
        const Pose recorded_pose = pose_at_frame(fixture.database, frame);
        const Transform recorded_object = compose(
            hand_world(recorded_pose, Hand::Right),
            inverse(affordance.hand_in_object));
        write_object_transform(fixture.database, frame, recorded_object);
        set_pose_trajectory_row(fixture.features, frame, query, 0.0F);
    }

    const CarryRanges ranges = classify_carry_ranges(fixture.database);
    assert(ranges.recorded.size() == 1U);
    assert_range(
        ranges.recorded.front(), 1, 115, 150, Hand::Right);
    CarryController controller(fixture.database, fixture.features, ranges);
    controller.start(hold, Hand::Right, affordance, object);

    Pose previous = hold;
    float maximum_step = 0.0F;
    int32_t maximum_tick = -1;
    for (int32_t tick = 0; tick < 20; ++tick) {
        const Pose output = controller.update(locomotion, 0.04F);
        const float step = maximum_right_arm_step(previous, output);
        if (step > maximum_step) {
            maximum_step = step;
            maximum_tick = tick;
        }
        assert(controller.recorded());
        assert(near(
            root_world(output), root_world(locomotion.pose), 2.0e-5F));
        assert(hand_error(
            output,
            Hand::Right,
            affordance,
            controller.object_world()) <= IKConfig{}.accepted_position_m);
        if (tick >= 12 && tick <= 14) {
            assert(near(
                output.hand_dof[0], kRecordedMarker, 2.0e-4F));
        }
        previous = output;
    }

    const float maximum_allowed =
        2.0F * IKConfig{}.maximum_step_radians;
    if (maximum_step > maximum_allowed + 2.0e-4F) {
        std::fprintf(
            stderr,
            "active-arm max step=%.9g tick=%d limit=%.9g\n",
            maximum_step,
            maximum_tick,
            maximum_allowed);
    }
    assert(maximum_step <= maximum_allowed + 2.0e-4F);
}

void test_recorded_range_switch_restarts_seam_but_progression_does_not() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const quat first_hip = quat_mul(
        hold.rotations[g1_skeleton::LeftHipPitch],
        quat_from_angle_axis(0.45F, vec3(1.0F, 0.0F, 0.0F)));
    const quat second_hip = quat_mul(
        hold.rotations[g1_skeleton::LeftHipPitch],
        quat_from_angle_axis(-0.75F, vec3(1.0F, 0.0F, 0.0F)));
    for (int32_t frame = 115; frame < 132; ++frame) {
        write_bone_rotation(
            fixture.database,
            frame,
            g1_skeleton::LeftHipPitch,
            first_hip);
    }
    for (int32_t frame = 132; frame < 150; ++frame) {
        write_bone_rotation(
            fixture.database,
            frame,
            g1_skeleton::LeftHipPitch,
            second_hip);
    }
    const NormalizedQuery query = carry_query(
        fixture.features, locomotion, Hand::Right, object, affordance);
    for (int32_t frame = 115; frame < 132; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 0.0F);
    }
    for (int32_t frame = 132; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    CarryRanges ranges{};
    ranges.recorded.push_back({1, 115, 132, Hand::Right});
    ranges.recorded.push_back({1, 132, 150, Hand::Right});
    const CarryConfig config = short_range_config();
    CarryController controller(
        fixture.database, fixture.features, ranges, config);
    controller.start(hold, Hand::Right, affordance, object);

    Pose output = controller.update(locomotion, 0.04F);
    for (int tick = 0; tick < 12; ++tick) {
        output = controller.update(locomotion, 0.04F);
    }
    assert(controller.recorded());
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        first_hip,
        2.0e-4F));

    for (int32_t frame = 115; frame < 132; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    for (int32_t frame = 132; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 0.0F);
    }
    const Pose before_switch = output;
    output = controller.update(locomotion, config.search_interval_seconds);

    assert(controller.recorded());
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        quat_nlerp_shortest(
            before_switch.rotations[g1_skeleton::LeftHipPitch],
            second_hip,
            0.20F),
        2.0e-4F));
    assert(!near(
        output.rotations[g1_skeleton::LeftHipPitch], second_hip, 0.10F));
    assert(hand_error(
        output,
        Hand::Right,
        affordance,
        controller.object_world()) <= IKConfig{}.accepted_position_m);

    for (int tick = 0; tick < 10; ++tick) {
        output = controller.update(locomotion, 0.04F);
    }
    assert(near(
        output.rotations[g1_skeleton::LeftHipPitch],
        second_hip,
        2.0e-4F));
    const Pose direct = controller.update(locomotion, 0.04F);
    for (int32_t joint = g1_skeleton::RightShoulderPitch;
         joint <= g1_skeleton::RightWrist;
         ++joint) {
        const size_t bone = static_cast<size_t>(joint);
        assert(near(direct.rotations[bone], output.rotations[bone], 0.20F));
    }
}

void test_recorded_cursor_cadence_remainder_and_tie_continuation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    mark_source_frames(fixture.database);
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery query = carry_query(
        fixture.features, locomotion, Hand::Right, object, affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 0.0F);
    }

    CarryRanges range{};
    range.recorded.push_back({1, 115, 150, Hand::Right});
    CarryController controller(
        fixture.database,
        fixture.features,
        range,
        short_range_config());
    controller.start(hold, Hand::Right, affordance, object);
    Pose settled = controller.update(locomotion, 0.0F);
    for (int32_t tick = 0; tick < 10; ++tick) {
        settled = controller.update(locomotion, 0.05F);
    }
    settled = controller.update(locomotion, 0.0F);
    assert(near(settled.hand_dof[0], 127.5F, 2.0e-4F));
    assert(near(
        controller.update(locomotion, 0.06F).hand_dof[0],
        129.0F,
        2.0e-4F));
    assert(near(
        controller.update(locomotion, 0.06F).hand_dof[0],
        130.5F,
        2.0e-4F));

    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    set_pose_trajectory_row(fixture.features, 128, query, 0.0F);
    const Pose before_remainder_crossing = controller.update(
        locomotion, 0.079F);
    assert(near(
        before_remainder_crossing.hand_dof[0], 132.475F, 2.0e-4F));
    const Pose remainder_crossing = controller.update(
        locomotion, 0.002F);
    assert(near(
        remainder_crossing.hand_dof[0],
        (1.0F - 0.004F) * before_remainder_crossing.hand_dof[0] +
            0.004F * 128.05F,
        2.0e-4F));
    assert(remainder_crossing.hand_dof[0] <
           before_remainder_crossing.hand_dof[0]);

    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    set_pose_trajectory_row(fixture.features, 134, query, 0.0F);
    const Pose inclusive_crossing = controller.update(
        locomotion, CarryConfig{}.search_interval_seconds);
    assert(near(
        inclusive_crossing.hand_dof[0],
        0.80F * remainder_crossing.hand_dof[0] + 0.20F * 136.5F,
        2.0e-4F));
    assert(inclusive_crossing.hand_dof[0] >
           remainder_crossing.hand_dof[0]);
}

void test_recorded_range_stop_is_half_open_and_clip_safe() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    mark_source_frames(fixture.database);
    fixture.database.hand_dof.at(120U * 14U) = 999.0F;
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    CarryRanges short_range{};
    short_range.recorded.push_back({1, 115, 120, Hand::Right});
    CarryController controller(
        fixture.database,
        fixture.features,
        short_range,
        short_range_config());
    controller.start(hold, Hand::Right, affordance, object);
    (void)controller.update(locomotion, 0.0F);
    const Pose terminal = controller.update(locomotion, 100.0F);
    assert(controller.recorded());
    assert(near(terminal.hand_dof[0], 119.0F));
    assert(!near(terminal.hand_dof[0], 999.0F));

    const Pose resumed = controller.update(
        locomotion, CarryConfig{}.search_interval_seconds);
    assert(controller.recorded());
    assert(resumed.hand_dof[0] >= 115.0F);
    assert(resumed.hand_dof[0] < 119.0F);
    assert(!near(resumed.hand_dof[0], 999.0F));
}

void test_recorded_requires_compatible_canonical_grasp() {
    using namespace interaction;
    RuntimeFixture compatible = make_runtime_fixture();
    runtime_fixture_detail::write_vec3(
        compatible.database.grasp_positions_object,
        1U,
        vec3(CarryConfig{}.maximum_grasp_drift_m, 0.0F, 0.0F));
    const size_t rotation_offset = 4U;
    compatible.database.grasp_rotations_object.at(rotation_offset) = -1.0F;
    const GraspAffordance affordance = fixture_affordance(compatible);
    const Pose hold = final_hold_pose(compatible);
    LocomotionSnapshot locomotion = compatible.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    CarryController accepted(
        compatible.database,
        compatible.features,
        classify_carry_ranges(compatible.database));
    accepted.start(hold, Hand::Right, affordance, object);
    (void)accepted.update(locomotion, 0.0F);
    assert(accepted.recorded());

    RuntimeFixture incompatible = make_runtime_fixture();
    runtime_fixture_detail::write_vec3(
        incompatible.database.grasp_positions_object,
        1U,
        vec3(
            CarryConfig{}.maximum_grasp_drift_m + 0.001F,
            0.0F,
            0.0F));
    const GraspAffordance incompatible_affordance =
        fixture_affordance(incompatible);
    const Pose incompatible_hold = final_hold_pose(incompatible);
    LocomotionSnapshot incompatible_locomotion = incompatible.locomotion;
    incompatible_locomotion.pose = incompatible_hold;
    CarryController rejected(
        incompatible.database,
        incompatible.features,
        classify_carry_ranges(incompatible.database));
    rejected.start(
        incompatible_hold,
        Hand::Right,
        incompatible_affordance,
        object_world_from_hold_pose(
            incompatible_hold,
            Hand::Right,
            incompatible_affordance));
    (void)rejected.update(incompatible_locomotion, 0.0F);
    assert(!rejected.recorded());
}

void test_rejected_recorded_ik_uses_layered_fallback() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    for (int32_t frame = 115; frame < 150; ++frame) {
        vec3 hand = runtime_fixture_detail::read_bone_position(
            fixture.database, frame, kRightHandBone);
        hand.x += 0.10F;
        runtime_fixture_detail::write_bone_position(
            fixture.database, frame, kRightHandBone, hand);
    }
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    IKConfig strict{};
    strict.maximum_request_position_m = 0.01F;
    strict.accepted_position_m = 0.005F;
    CarryController controller(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database),
        CarryConfig{},
        strict);
    controller.start(hold, Hand::Right, affordance, object);
    const Pose output = controller.update(locomotion, 0.0F);
    assert(!controller.recorded());
    assert(near(
        output.rotations[g1_skeleton::RightShoulderPitch],
        hold.rotations[g1_skeleton::RightShoulderPitch]));
    assert(hand_error(
        output,
        Hand::Right,
        affordance,
        controller.object_world()) <= 0.04F);
}

void test_recorded_object_is_published_from_solved_hand() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    for (int32_t frame = 115; frame < 150; ++frame) {
        vec3 hand = runtime_fixture_detail::read_bone_position(
            fixture.database, frame, kRightHandBone);
        hand.x += 0.01F;
        runtime_fixture_detail::write_bone_position(
            fixture.database, frame, kRightHandBone, hand);
    }
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    CarryController controller(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database));
    controller.start(hold, Hand::Right, affordance, object);
    const Pose output = controller.update(locomotion, 0.0F);
    assert(controller.recorded());
    const Transform published = compose(
        hand_world(output, Hand::Right),
        inverse(affordance.hand_in_object));
    assert(near(controller.object_world(), published, 2.0e-5F));
}

void test_recorded_solved_object_cannot_exceed_continuity_bounds() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery query = carry_query(
        fixture.features,
        locomotion,
        Hand::Right,
        initial_object,
        affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
        vec3 hand = runtime_fixture_detail::read_bone_position(
            fixture.database, frame, kRightHandBone);
        hand.x += 0.03F;
        runtime_fixture_detail::write_bone_position(
            fixture.database, frame, kRightHandBone, hand);
    }
    set_pose_trajectory_row(fixture.features, 115, query, 0.0F);

    const CarryRanges ranges = classify_carry_ranges(fixture.database);
    assert(ranges.recorded.size() == 1U);
    CarryController controller(
        fixture.database, fixture.features, ranges);
    controller.start(
        hold, Hand::Right, affordance, initial_object);

    const Pose output = controller.update(locomotion, 0.0F);

    assert(!controller.recorded());
    assert(near(controller.object_world(), initial_object, 2.0e-5F));
    assert(near(
        controller.object_world(),
        compose(
            hand_world(output, Hand::Right),
            inverse(affordance.hand_in_object)),
        2.0e-5F));
}

void test_recorded_initial_selection_cannot_teleport_object() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery query = carry_query(
        fixture.features,
        locomotion,
        Hand::Right,
        initial_object,
        affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    set_pose_trajectory_row(fixture.features, 115, query, 0.0F);
    shift_recorded_hand_and_object(
        fixture.database, 115, 116, vec3(0.0F, 0.0F, 1.0F));

    const CarryRanges ranges = classify_carry_ranges(fixture.database);
    assert(ranges.recorded.size() == 1U);
    IKConfig strict{};
    strict.maximum_request_position_m = 0.01F;
    strict.accepted_position_m = 0.005F;
    CarryController controller(
        fixture.database,
        fixture.features,
        ranges,
        CarryConfig{},
        strict);
    controller.start(
        hold, Hand::Right, affordance, initial_object);

    const Pose output = controller.update(locomotion, 0.0F);

    assert(!controller.recorded());
    assert(near(controller.object_world(), initial_object, 2.0e-5F));
    assert(near(
        controller.object_world(),
        compose(
            hand_world(output, Hand::Right),
            inverse(affordance.hand_in_object)),
        2.0e-5F));
}

void test_recorded_research_fallback_preserves_last_published_anchor() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    mark_source_frames(fixture.database);
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform initial_object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery initial_query = carry_query(
        fixture.features,
        locomotion,
        Hand::Right,
        initial_object,
        affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(
            fixture.features, frame, initial_query, 10.0F);
    }
    for (int32_t frame = 115; frame < 140; ++frame) {
        set_pose_trajectory_row(
            fixture.features, frame, initial_query, 0.0F);
    }
    shift_recorded_hand_and_object(
        fixture.database, 115, 140, vec3(0.0F, 0.0F, 0.08F));
    shift_recorded_hand_and_object(
        fixture.database, 145, 150, vec3(0.0F, 0.0F, 1.0F));

    CarryConfig continuity{};
    continuity.maximum_grasp_drift_m = 0.10F;
    IKConfig bounded{};
    bounded.maximum_request_position_m = 0.10F;
    bounded.accepted_position_m = 0.001F;
    CarryController controller(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database, continuity),
        continuity,
        bounded);
    controller.start(
        hold, Hand::Right, affordance, initial_object);

    Pose first_output = controller.update(locomotion, 0.0F);
    first_output = controller.update(locomotion, 0.50F);
    first_output = controller.update(locomotion, 0.0F);
    const Transform first_object = controller.object_world();
    assert(controller.recorded());
    assert(length(first_object.position - initial_object.position) > 0.07F);
    locomotion.pose = first_output;
    const NormalizedQuery later_query = carry_query(
        fixture.features,
        locomotion,
        Hand::Right,
        first_object,
        affordance);
    for (int32_t frame = 115; frame < 150; ++frame) {
        set_pose_trajectory_row(
            fixture.features, frame, later_query, 10.0F);
    }
    set_pose_trajectory_row(
        fixture.features, 145, later_query, 0.0F);

    const Pose fallback_output = controller.update(
        locomotion, continuity.search_interval_seconds);

    assert(!controller.recorded());
    assert(near(controller.object_world(), first_object, 2.0e-4F));
    assert(near(
        controller.object_world(),
        compose(
            hand_world(fallback_output, Hand::Right),
            inverse(affordance.hand_in_object)),
        2.0e-5F));
}

void test_update_exception_rolls_back_search_playback_and_anchor_state() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    mark_source_frames(fixture.database);
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery query = carry_query(
        fixture.features, locomotion, Hand::Right, object, affordance);
    for (int32_t frame = 115; frame < 125; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    set_pose_trajectory_row(fixture.features, 115, query, 0.0F);

    CarryRanges range{};
    range.recorded.push_back({1, 115, 125, Hand::Right});
    const CarryConfig config = short_range_config();
    CarryController recovered(
        fixture.database, fixture.features, range, config);
    CarryController fresh(
        fixture.database, fixture.features, range, config);
    recovered.start(hold, Hand::Right, affordance, object);
    fresh.start(hold, Hand::Right, affordance, object);

    LocomotionSnapshot extreme = locomotion;
    const float finite_max = std::numeric_limits<float>::max();
    extreme.pose.positions[g1_skeleton::Simulation].x = finite_max;
    extreme.future_root_positions[0].x = -finite_max;
    assert(std::isfinite(
        extreme.pose.positions[g1_skeleton::Simulation].x));
    assert(std::isfinite(extreme.future_root_positions[0].x));
    assert(throws_as<std::invalid_argument>([&] {
        (void)recovered.update(extreme, 0.06F);
    }));

    const Pose recovered_zero = recovered.update(locomotion, 0.0F);
    const Pose fresh_zero = fresh.update(locomotion, 0.0F);
    assert(recovered.recorded());
    assert(recovered.recorded() == fresh.recorded());
    assert(exact(recovered_zero, fresh_zero));
    assert(exact(recovered.object_world(), fresh.object_world()));

    for (int32_t frame = 115; frame < 125; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    set_pose_trajectory_row(fixture.features, 120, query, 0.0F);
    const Pose recovered_before = recovered.update(locomotion, 0.099F);
    const Pose fresh_before = fresh.update(locomotion, 0.099F);
    assert(exact(recovered_before, fresh_before));
    assert(exact(recovered.object_world(), fresh.object_world()));
    assert(near(
        recovered_before.hand_dof[0],
        (1.0F - 0.198F) * hold.hand_dof[0] + 0.198F * 117.475F,
        2.0e-4F));

    const Pose recovered_crossing = recovered.update(locomotion, 0.002F);
    const Pose fresh_crossing = fresh.update(locomotion, 0.002F);
    assert(exact(recovered_crossing, fresh_crossing));
    assert(exact(recovered.object_world(), fresh.object_world()));
    assert(near(
        recovered_crossing.hand_dof[0],
        (1.0F - 0.004F) * recovered_before.hand_dof[0] +
            0.004F * 120.05F,
        2.0e-4F));
}

void test_repeated_start_resets_recorded_cursor_and_cadence() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    mark_source_frames(fixture.database);
    const GraspAffordance affordance = fixture_affordance(fixture);
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;
    const Transform object = object_world_from_hold_pose(
        hold, Hand::Right, affordance);
    const NormalizedQuery query = carry_query(
        fixture.features, locomotion, Hand::Right, object, affordance);
    for (int32_t frame = 115; frame < 125; ++frame) {
        set_pose_trajectory_row(fixture.features, frame, query, 10.0F);
    }
    set_pose_trajectory_row(fixture.features, 115, query, 0.0F);
    CarryRanges range{};
    range.recorded.push_back({1, 115, 125, Hand::Right});
    CarryController controller(
        fixture.database,
        fixture.features,
        range,
        short_range_config());
    controller.start(hold, Hand::Right, affordance, object);
    const Pose first = controller.update(locomotion, 0.08F);
    const Transform first_object = controller.object_world();
    assert(near(
        first.hand_dof[0],
        0.84F * hold.hand_dof[0] + 0.16F * 117.0F,
        2.0e-4F));
    assert(controller.recorded());

    controller.start(hold, Hand::Right, affordance, object);
    assert(!controller.recorded());
    const Pose restarted = controller.update(locomotion, 0.0F);
    assert(controller.recorded());
    assert(near(restarted.hand_dof[0], hold.hand_dof[0]));
    const Pose replayed = controller.update(locomotion, 0.08F);
    assert(exact(replayed, first));
    assert(exact(controller.object_world(), first_object));
}

void test_wrong_hand_uses_fallback() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    GraspAffordance affordance = fixture_affordance(fixture);
    affordance.hand = Hand::Left;
    const Pose hold = final_hold_pose(fixture);
    LocomotionSnapshot locomotion = fixture.locomotion;
    locomotion.pose = hold;

    CarryController controller(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database));
    controller.start(
        hold,
        Hand::Left,
        affordance,
        object_world_from_hold_pose(hold, Hand::Left, affordance));
    const Pose output = controller.update(locomotion, 1.0F / 60.0F);

    assert(!controller.recorded());
    assert(near(
        output.rotations[g1_skeleton::LeftShoulderPitch],
        hold.rotations[g1_skeleton::LeftShoulderPitch]));
    assert(hand_error(
        output, Hand::Left, affordance, controller.object_world()) <= 0.04F);
}

}  // namespace

int main() {
    test_frozen_public_interface_and_defaults();
    test_fixture_classification_is_exact();
    test_classification_thresholds_are_inclusive();
    test_classification_partitions_contact_and_drift_runs();
    test_grasp_threshold_and_quaternion_sign_do_not_split();
    test_average_speed_uses_planar_path_length();
    test_fallback_preserves_locomotion_and_grasp();
    test_default_layered_carry_releases_inactive_arm_after_hold_seam();
    test_fallback_rotation_masks_are_layered();
    test_layered_anchor_and_nonidentity_grasp_move_with_root();
    test_layered_carry_smooths_ik_feasible_nonarm_seam();
    test_layered_carry_inertializes_hold_to_live_seam_without_stalling_root();
    test_rejected_layered_ik_preserves_last_safe_nonroot_pose_and_live_anchor();
    test_lifecycle_and_invalid_inputs_are_defensive();
    test_recorded_search_uses_pose_trajectory_and_aligns_object();
    test_initial_recorded_carry_smooths_nonarm_seam();
    test_recorded_carry_keeps_active_arm_on_one_continuous_ik_branch();
    test_recorded_range_switch_restarts_seam_but_progression_does_not();
    test_recorded_cursor_cadence_remainder_and_tie_continuation();
    test_recorded_range_stop_is_half_open_and_clip_safe();
    test_recorded_requires_compatible_canonical_grasp();
    test_rejected_recorded_ik_uses_layered_fallback();
    test_recorded_object_is_published_from_solved_hand();
    test_recorded_solved_object_cannot_exceed_continuity_bounds();
    test_recorded_initial_selection_cannot_teleport_object();
    test_recorded_research_fallback_preserves_last_published_anchor();
    test_update_exception_rolls_back_search_playback_and_anchor_state();
    test_repeated_start_resets_recorded_cursor_and_cadence();
    test_wrong_hand_uses_fallback();
    return 0;
}
