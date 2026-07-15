#pragma once

#include "interaction_matcher.h"
#include "interaction_place_controller.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace interaction {

struct RuntimeFixture {
    Database database{};
    Features features{};
    TargetRegistry registry{};
    PlacementSurfaceRegistry surface_registry{};
    PlaceMotionLibrary place_library{};
    LocomotionSnapshot locomotion{};
    Transform original_object_world{};
    PickRequest request{};
    SurfaceHandle surface{};
    uint32_t place_affordance_id = 0U;
};

namespace runtime_fixture_detail {

inline constexpr int32_t kFramesPerClip = 75;
inline constexpr int32_t kClipCount = 2;
inline constexpr int32_t kContactLocalFrame = 25;
inline constexpr int32_t kLiftLocalFrame = 30;
inline constexpr int32_t kHoldLocalFrame = 40;
inline constexpr TargetHandle kTargetHandle{42U, 1U};
inline constexpr uint32_t kAffordanceId = 7U;
inline constexpr uint64_t kRequestId = 99U;

inline size_t vector_offset(int32_t frame, size_t item) {
    return (static_cast<size_t>(frame) * g1_skeleton::BoneCount + item) * 3U;
}

inline size_t quaternion_offset(int32_t frame, size_t item) {
    return (static_cast<size_t>(frame) * g1_skeleton::BoneCount + item) * 4U;
}

inline void write_vec3(std::vector<float>& values, size_t index, vec3 value) {
    const size_t offset = index * 3U;
    values.at(offset) = value.x;
    values.at(offset + 1U) = value.y;
    values.at(offset + 2U) = value.z;
}

inline vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(
        values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

inline void write_bone_position(
    Database& database,
    int32_t frame,
    size_t bone,
    vec3 value) {
    const size_t offset = vector_offset(frame, bone);
    database.positions.at(offset) = value.x;
    database.positions.at(offset + 1U) = value.y;
    database.positions.at(offset + 2U) = value.z;
}

inline vec3 read_bone_position(
    const Database& database,
    int32_t frame,
    size_t bone) {
    const size_t offset = vector_offset(frame, bone);
    return vec3(
        database.positions.at(offset),
        database.positions.at(offset + 1U),
        database.positions.at(offset + 2U));
}

inline void write_identity_rotation(
    Database& database,
    int32_t frame,
    size_t bone) {
    const size_t offset = quaternion_offset(frame, bone);
    database.rotations.at(offset) = 1.0F;
    database.rotations.at(offset + 1U) = 0.0F;
    database.rotations.at(offset + 2U) = 0.0F;
    database.rotations.at(offset + 3U) = 0.0F;
}

inline float carry_x(int32_t clip, int32_t local_frame) {
    if (clip != 1 || local_frame < kHoldLocalFrame) return 0.0F;
    return 0.40F * static_cast<float>(local_frame - kHoldLocalFrame) /
           static_cast<float>(kFramesPerClip - 1 - kHoldLocalFrame);
}

inline float object_height(int32_t local_frame) {
    if (local_frame < kLiftLocalFrame) return 0.75F;
    if (local_frame >= kHoldLocalFrame) return 0.95F;
    return 0.75F + 0.20F *
        static_cast<float>(local_frame - kLiftLocalFrame) /
        static_cast<float>(kHoldLocalFrame - 1 - kLiftLocalFrame);
}

inline float approaching_hand_z(int32_t local_frame) {
    if (local_frame < 10) {
        return 2.20F + 0.02F * static_cast<float>(local_frame);
    }
    if (local_frame < kContactLocalFrame) {
        return 2.40F + 0.50F * static_cast<float>(local_frame - 10) / 14.0F;
    }
    return 3.0F;
}

inline Phase phase_at(int32_t local_frame) {
    if (local_frame < 10) return Phase::Approach;
    if (local_frame < kContactLocalFrame) return Phase::Reach;
    if (local_frame < kLiftLocalFrame) return Phase::Contact;
    if (local_frame < kHoldLocalFrame) return Phase::Lift;
    return Phase::Hold;
}

inline void recompute_velocities(Database& database) {
    constexpr float kFps = 25.0F;
    for (int32_t clip = 0; clip < kClipCount; ++clip) {
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        for (int32_t frame = start; frame < stop; ++frame) {
            const int32_t previous = std::max(start, frame - 1);
            const int32_t next = std::min(stop - 1, frame + 1);
            const float scale = next == previous
                ? 0.0F
                : kFps / static_cast<float>(next - previous);
            for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
                const vec3 velocity = scale * (
                    read_bone_position(database, next, bone) -
                    read_bone_position(database, previous, bone));
                const size_t offset = vector_offset(frame, bone);
                database.velocities.at(offset) = velocity.x;
                database.velocities.at(offset + 1U) = velocity.y;
                database.velocities.at(offset + 2U) = velocity.z;
            }
            const vec3 object_velocity = scale * (
                read_vec3(database.object_positions, next) -
                read_vec3(database.object_positions, previous));
            write_vec3(
                database.object_velocities,
                static_cast<size_t>(frame),
                object_velocity);
        }
    }
}

inline Database make_database() {
    Database database{};
    database.version = 1U;
    database.endian_marker = 0x01020304U;
    database.fps_numerator = 25U;
    database.fps_denominator = 1U;
    database.frame_count =
        static_cast<uint32_t>(kFramesPerClip * kClipCount);
    database.bone_count = static_cast<uint32_t>(g1_skeleton::BoneCount);
    database.clip_count = static_cast<uint32_t>(kClipCount);
    database.hand_dof_count = 14U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts = {0, kFramesPerClip};
    database.range_stops = {kFramesPerClip, kFramesPerClip * kClipCount};

    const size_t frames = database.frame_count;
    const size_t bones = g1_skeleton::BoneCount;
    database.positions.assign(frames * bones * 3U, 0.0F);
    database.velocities.assign(frames * bones * 3U, 0.0F);
    database.rotations.assign(frames * bones * 4U, 0.0F);
    database.angular_velocities.assign(frames * bones * 3U, 0.0F);
    database.foot_contacts.assign(frames * 2U, 0U);
    database.hand_contacts.assign(frames * 2U, 0U);
    database.hand_dof.assign(frames * 14U, 0.0F);
    database.hand_dof_velocities.assign(frames * 14U, 0.0F);
    database.phases.assign(frames, 0U);
    database.active_hands = {0U, 1U};
    database.time_to_contact.assign(frames, 0.0F);
    database.object_positions.assign(frames * 3U, 0.0F);
    database.object_rotations.assign(frames * 4U, 0.0F);
    database.object_velocities.assign(frames * 3U, 0.0F);
    database.object_angular_velocities.assign(frames * 3U, 0.0F);
    database.table_positions.assign(kClipCount * 3U, 0.0F);
    database.table_rotations.assign(kClipCount * 4U, 0.0F);
    database.table_sizes.assign(kClipCount * 3U, 0.0F);
    database.object_dimensions.assign(kClipCount * 3U, 0.0F);
    database.grasp_positions_object.assign(kClipCount * 3U, 0.0F);
    database.grasp_rotations_object.assign(kClipCount * 4U, 0.0F);
    database.approach_directions_object.assign(kClipCount * 3U, 0.0F);
    database.source_frames.assign(frames, 0);

    for (int32_t clip = 0; clip < kClipCount; ++clip) {
        write_vec3(database.table_positions, clip, vec3(0.0F, 0.35F, 3.0F));
        database.table_rotations.at(static_cast<size_t>(clip) * 4U) = 1.0F;
        write_vec3(database.table_sizes, clip, vec3(1.0F, 0.70F, 1.0F));
        write_vec3(
            database.object_dimensions,
            clip,
            vec3(0.08F, 0.20F, 0.08F));
        database.grasp_rotations_object.at(
            static_cast<size_t>(clip) * 4U) = 1.0F;
        write_vec3(
            database.approach_directions_object,
            clip,
            vec3(0.0F, 0.0F, -1.0F));

        for (int32_t local = 0; local < kFramesPerClip; ++local) {
            const int32_t frame = clip * kFramesPerClip + local;
            const float x = carry_x(clip, local);
            const vec3 root_world(x, 0.0F, 2.0F);
            const vec3 object_world(x, object_height(local), 3.0F);
            const vec3 hand_world = local < kContactLocalFrame
                ? vec3(0.0F, 0.75F, approaching_hand_z(local))
                : object_world;
            write_bone_position(
                database, frame, g1_skeleton::Simulation, root_world);
            write_bone_position(
                database, frame, kLeftHandBone, hand_world - root_world);
            write_bone_position(
                database, frame, kRightHandBone, hand_world - root_world);
            for (size_t bone = 0; bone < bones; ++bone) {
                write_identity_rotation(database, frame, bone);
            }
            database.phases.at(frame) =
                static_cast<uint8_t>(phase_at(local));
            database.time_to_contact.at(frame) = local < kContactLocalFrame
                ? static_cast<float>(kContactLocalFrame - local) / 25.0F
                : 0.0F;
            database.source_frames.at(frame) = local;
            if (local >= kContactLocalFrame) {
                const size_t contact =
                    static_cast<size_t>(frame) * 2U +
                    database.active_hands.at(clip);
                database.hand_contacts.at(contact) = 1U;
            }
            write_vec3(database.object_positions, frame, object_world);
            database.object_rotations.at(static_cast<size_t>(frame) * 4U) =
                1.0F;
        }
    }
    recompute_velocities(database);
    return database;
}

inline InteractionTarget make_target() {
    InteractionTarget target{};
    target.handle = kTargetHandle;
    target.object_world = {vec3(0.0F, 0.75F, 3.0F), quat()};
    target.object_profile_id = 3001U;
    target.object_dimensions = vec3(0.08F, 0.20F, 0.08F);
    target.object_bounds = {
        vec3(), vec3(0.04F, 0.10F, 0.04F)};
    target.table_world = {vec3(0.0F, 0.35F, 3.0F), quat()};
    target.table_size = vec3(1.0F, 0.70F, 1.0F);
    target.affordances = {{
        kAffordanceId,
        Hand::Right,
        Transform{vec3(), quat()},
        vec3(0.0F, 0.0F, -1.0F),
        0.04F,
    }};
    assert(target.object_profile_id != 0U);
    assert(target.object_bounds.center_object.x == 0.0F);
    assert(target.object_bounds.center_object.y == 0.0F);
    assert(target.object_bounds.center_object.z == 0.0F);
    assert(target.object_bounds.half_extents_object.x ==
           target.object_dimensions.x * 0.5F);
    assert(target.object_bounds.half_extents_object.y ==
           target.object_dimensions.y * 0.5F);
    assert(target.object_bounds.half_extents_object.z ==
           target.object_dimensions.z * 0.5F);
    return target;
}

inline PlacementSurface make_placement_surface(
    uint64_t id = 900U,
    vec3 top = vec3(0.0F, 0.65F, 3.0F)) {
    PlacementSurface surface{};
    surface.handle = {id, 3U};
    surface.surface_world = {top, quat()};
    surface.support_volume_world = {
        top - vec3(0.0F, 0.35F, 0.0F), quat()};
    surface.support_volume_size = vec3(1.0F, 0.70F, 1.0F);
    surface.half_extent_x_m = 0.40F;
    surface.half_extent_z_m = 0.40F;
    surface.overhead_clearance_m = 0.50F;
    surface.affordances = {{
        77U,
        Transform{vec3(0.0F, 0.10F, 0.0F), quat()},
        vec3(0.0F, -0.10F, 0.0F),
        vec3(0.0F, 1.0F, 0.0F),
        0.01F,
    }};
    return surface;
}

inline RecordedPlaceClip make_recorded_place_clip(
    const Database& database,
    uint64_t id = 101U) {
    RecordedPlaceClip clip{};
    clip.id = id;
    clip.object_profile_id = make_target().object_profile_id;
    clip.fps_numerator = 25U;
    clip.fps_denominator = 1U;
    clip.entry_frame = 0;
    clip.commit_frame = 8;
    clip.release_frame = 12;
    clip.retract_stop_frame = 15;
    clip.hand = Hand::Right;
    clip.hand_in_object = Transform{};
    clip.object_bounds = make_target().object_bounds;
    clip.source_surface = make_placement_surface(501U);
    clip.source_affordance_id = 77U;

    constexpr int32_t source_frame =
        kFramesPerClip + kHoldLocalFrame;
    const Pose source_pose = pose_at_frame(database, source_frame);
    for (int32_t frame = 0; frame <= clip.retract_stop_frame; ++frame) {
        const float alpha = std::min(frame, clip.release_frame) /
            static_cast<float>(clip.release_frame);
        const Transform object{
            vec3(0.10F * (1.0F - alpha),
                 0.95F - 0.20F * alpha,
                 3.0F),
            quat()};
        Pose pose = source_pose;
        pose.positions[g1_skeleton::Simulation].x = object.position.x;
        pose.positions[kRightHandBone] =
            object.position - pose.positions[g1_skeleton::Simulation];
        pose.rotations[kRightHandBone] = object.rotation;
        clip.poses.push_back(pose);
        clip.object_poses.push_back(object);
        clip.active_hand_contacts.push_back(
            frame <= clip.release_frame ? 1U : 0U);
    }
    return clip;
}

inline LocomotionSnapshot make_locomotion(const Database& database) {
    LocomotionSnapshot locomotion{};
    constexpr int32_t kFirstRightReach = kFramesPerClip + 10;
    locomotion.pose = pose_at_frame(database, kFirstRightReach);
    constexpr std::array<int32_t, 3> kFutureOffsets = {8, 17, 25};
    for (size_t sample = 0; sample < kFutureOffsets.size(); ++sample) {
        const int32_t frame = std::min(
            kFirstRightReach + kFutureOffsets[sample],
            kFramesPerClip * kClipCount - 1);
        const Pose future = pose_at_frame(database, frame);
        locomotion.future_root_positions[sample] =
            future.positions[g1_skeleton::Simulation];
        locomotion.future_root_rotations[sample] =
            future.rotations[g1_skeleton::Simulation];
    }
    return locomotion;
}

inline void set_group_row(
    Features& features,
    int32_t frame,
    const std::array<float, 5>& values) {
    for (size_t group = 0; group < values.size(); ++group) {
        for (size_t dimension = kFeatureGroupStarts[group];
             dimension < kFeatureGroupStops[group];
             ++dimension) {
            features.values.at(
                static_cast<size_t>(frame) * kFeatureDimension + dimension) =
                values[group];
        }
    }
}

}  // namespace runtime_fixture_detail

inline TargetRegistry fresh_registry() {
    TargetRegistry registry;
    registry.upsert(runtime_fixture_detail::make_target());
    return registry;
}

inline PickRequest valid_request_for(const TargetRegistry& registry) {
    if (registry.find(runtime_fixture_detail::kTargetHandle) == nullptr) {
        throw std::invalid_argument("runtime fixture target missing");
    }
    return {
        runtime_fixture_detail::kTargetHandle,
        runtime_fixture_detail::kAffordanceId,
        runtime_fixture_detail::kRequestId,
    };
}

inline PickRequest valid_request() {
    const TargetRegistry registry = fresh_registry();
    return valid_request_for(registry);
}

inline QueryInput query_input_for(
    const RuntimeFixture& fixture,
    const InteractionTarget& target,
    const GraspAffordance& affordance) {
    QueryInput query{};
    query.locomotion = fixture.locomotion;
    query.grasp_world = compose(target.object_world, affordance.hand_in_object);
    query.table_world = target.table_world;
    query.table_size = target.table_size;
    query.approach_direction_object =
        affordance.approach_direction_object;
    query.object_dimensions = target.object_dimensions;
    query.hand = affordance.hand;
    return query;
}

inline RuntimeFixture make_runtime_fixture() {
    using namespace runtime_fixture_detail;
    RuntimeFixture fixture{};
    fixture.database = make_database();
    fixture.registry = fresh_registry();
    fixture.locomotion = make_locomotion(fixture.database);
    fixture.original_object_world = make_target().object_world;
    fixture.request = valid_request_for(fixture.registry);

    fixture.features.version = 1U;
    fixture.features.endian_marker = 0x01020304U;
    fixture.features.frame_count = fixture.database.frame_count;
    fixture.features.feature_count = static_cast<uint32_t>(kFeatureDimension);
    fixture.features.dimension = static_cast<uint32_t>(kFeatureDimension);
    fixture.features.group_count = 5U;
    fixture.features.group_starts = {0U, 33U, 45U, 57U, 65U};
    fixture.features.group_stops = {33U, 45U, 57U, 65U, 71U};
    fixture.features.scales.assign(kFeatureDimension, 1.0F);
    fixture.features.values.assign(
        static_cast<size_t>(fixture.database.frame_count) *
            kFeatureDimension,
        0.0F);
    for (int32_t frame = 0;
         frame < static_cast<int32_t>(fixture.database.frame_count);
         ++frame) {
        set_group_row(
            fixture.features,
            frame,
            {0.10F, 0.20F, 0.30F, 0.40F, 0.50F});
    }

    const InteractionTarget* target = fixture.registry.find(fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    if (target == nullptr || affordance == nullptr) {
        throw std::logic_error("runtime fixture target contract is incomplete");
    }
    const RawQuery raw = build_raw_query(
        query_input_for(fixture, *target, *affordance));
    fixture.features.offsets.assign(raw.begin(), raw.end());
    return fixture;
}

inline RuntimeFixture make_place_runtime_fixture() {
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.surface = fixture.surface_registry.upsert(
        runtime_fixture_detail::make_placement_surface());
    fixture.place_affordance_id = 77U;
    fixture.place_library.recorded.push_back(
        runtime_fixture_detail::make_recorded_place_clip(fixture.database));
    return fixture;
}

inline MatchInput match_input_for(const RuntimeFixture& fixture) {
    const InteractionTarget* target = fixture.registry.find(fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    if (target == nullptr || affordance == nullptr) {
        throw std::invalid_argument("runtime fixture request is stale");
    }
    MatchInput input{};
    input.database = &fixture.database;
    input.features = &fixture.features;
    input.locomotion = fixture.locomotion;
    input.target = *target;
    input.affordance = *affordance;
    input.request = fixture.request;
    input.query = normalize_query(
        build_raw_query(query_input_for(fixture, *target, *affordance)),
        fixture.features);
    return input;
}

inline const RuntimeFixture& valid_fixture_storage() {
    static const RuntimeFixture fixture = make_runtime_fixture();
    return fixture;
}

inline MatchInput valid_input() {
    return match_input_for(valid_fixture_storage());
}

inline MatchInput wrong_hand_input() {
    static const RuntimeFixture fixture = [] {
        RuntimeFixture value = make_runtime_fixture();
        value.database.active_hands = {1U, 1U};
        for (int32_t frame = 0;
             frame < runtime_fixture_detail::kFramesPerClip;
             ++frame) {
            const bool contacted = value.database.phases.at(frame) >=
                static_cast<uint8_t>(Phase::Contact);
            value.database.hand_contacts.at(
                static_cast<size_t>(frame) * 2U) = 0U;
            value.database.hand_contacts.at(
                static_cast<size_t>(frame) * 2U + 1U) =
                contacted ? 1U : 0U;
        }
        InteractionTarget* target = value.registry.find(value.request.target);
        if (target == nullptr) throw std::logic_error("fixture target missing");
        target->affordances.at(0).hand = Hand::Left;
        return value;
    }();
    return match_input_for(fixture);
}

inline MatchInput out_of_range_input() {
    MatchInput input = valid_input();
    input.target.object_world.position.x += 1.01F;
    return input;
}

inline MatchInput excessive_root_input() {
    MatchInput input = valid_input();
    input.locomotion.pose.positions[g1_skeleton::Simulation].z += 0.30F;
    return input;
}

inline MatchInput blocked_table_input() {
    MatchInput input = valid_input();
    input.target.table_size.z = 1.60F;
    return input;
}

inline RuntimeFixture high_cost_fixture() {
    RuntimeFixture fixture = make_runtime_fixture();
    for (float& value : fixture.features.values) value += 10.0F;
    return fixture;
}

inline MatchInput high_cost_input() {
    static const RuntimeFixture fixture = high_cost_fixture();
    return match_input_for(fixture);
}

inline RuntimeFixture contact_failure_fixture() {
    RuntimeFixture fixture = make_runtime_fixture();
    constexpr int32_t kStableContact =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame;
    vec3 hand = runtime_fixture_detail::read_bone_position(
        fixture.database, kStableContact, kRightHandBone);
    hand.x += 0.05F;
    runtime_fixture_detail::write_bone_position(
        fixture.database, kStableContact, kRightHandBone, hand);
    runtime_fixture_detail::recompute_velocities(fixture.database);
    return fixture;
}

}  // namespace interaction
