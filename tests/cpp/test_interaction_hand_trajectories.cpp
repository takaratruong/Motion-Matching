#include "interaction_hand_trajectories.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(vec3 left, vec3 right, float tolerance = 1.0e-4F) {
    return length(left - right) <= tolerance;
}

bool near_rotation(quat left, quat right, float tolerance = 1.0e-4F) {
    const float dot = std::abs(
        left.w * right.w + left.x * right.x +
        left.y * right.y + left.z * right.z);
    return std::abs(dot - 1.0F) <= tolerance;
}

bool active_arm_bone(size_t bone, interaction::Hand hand) {
    const std::array<g1_skeleton::Bone, 7> left{{
        g1_skeleton::LeftShoulderPitch,
        g1_skeleton::LeftShoulderRoll,
        g1_skeleton::LeftShoulderYaw,
        g1_skeleton::LeftElbow,
        g1_skeleton::LeftWristRoll,
        g1_skeleton::LeftWristPitch,
        g1_skeleton::LeftWrist,
    }};
    const std::array<g1_skeleton::Bone, 7> right{{
        g1_skeleton::RightShoulderPitch,
        g1_skeleton::RightShoulderRoll,
        g1_skeleton::RightShoulderYaw,
        g1_skeleton::RightElbow,
        g1_skeleton::RightWristRoll,
        g1_skeleton::RightWristPitch,
        g1_skeleton::RightWrist,
    }};
    const auto& bones = hand == interaction::Hand::Left ? left : right;
    return std::find(bones.begin(), bones.end(), bone) != bones.end();
}

struct ClipSpec {
    interaction::Hand hand = interaction::Hand::Right;
    vec3 grasp_position{};
    quat grasp_rotation{};
    vec3 dimensions{1.0F, 1.0F, 1.0F};
    bool complete_phases = true;
    quat object_rotation{};
};

void write_vec3(std::vector<float>& values, size_t index, vec3 value) {
    const size_t offset = 3U * index;
    values[offset] = value.x;
    values[offset + 1U] = value.y;
    values[offset + 2U] = value.z;
}

void write_quat(std::vector<float>& values, size_t index, quat value) {
    const size_t offset = 4U * index;
    values[offset] = value.w;
    values[offset + 1U] = value.x;
    values[offset + 2U] = value.y;
    values[offset + 3U] = value.z;
}

interaction::Database make_database(const std::vector<ClipSpec>& specs) {
    constexpr int32_t frames_per_clip = 8;
    const size_t frames = specs.size() * frames_per_clip;
    interaction::Database database{};
    database.frame_count = static_cast<uint32_t>(frames);
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = static_cast<uint32_t>(specs.size());
    database.hand_dof_count = 14U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts.resize(specs.size());
    database.range_stops.resize(specs.size());
    database.positions.assign(frames * g1_skeleton::BoneCount * 3U, 0.0F);
    database.velocities.assign(database.positions.size(), 0.0F);
    database.angular_velocities.assign(database.positions.size(), 0.0F);
    database.rotations.assign(frames * g1_skeleton::BoneCount * 4U, 0.0F);
    database.foot_contacts.assign(frames * 2U, 0U);
    database.hand_contacts.assign(frames * 2U, 0U);
    database.hand_dof.assign(frames * 14U, 0.0F);
    database.hand_dof_velocities.assign(frames * 14U, 0.0F);
    database.phases.assign(frames, 0U);
    database.active_hands.resize(specs.size());
    database.time_to_contact.assign(frames, 0.0F);
    database.object_positions.assign(frames * 3U, 0.0F);
    database.object_rotations.assign(frames * 4U, 0.0F);
    database.object_velocities.assign(frames * 3U, 0.0F);
    database.object_angular_velocities.assign(frames * 3U, 0.0F);
    database.table_positions.assign(specs.size() * 3U, 0.0F);
    database.table_rotations.assign(specs.size() * 4U, 0.0F);
    database.table_sizes.assign(specs.size() * 3U, 1.0F);
    database.object_dimensions.assign(specs.size() * 3U, 0.0F);
    database.grasp_positions_object.assign(specs.size() * 3U, 0.0F);
    database.grasp_rotations_object.assign(specs.size() * 4U, 0.0F);
    database.approach_directions_object.assign(specs.size() * 3U, 0.0F);
    database.source_frames.resize(frames);

    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
            write_quat(
                database.rotations,
                frame * g1_skeleton::BoneCount + bone,
                quat());
        }
        write_quat(database.object_rotations, frame, quat());
        database.source_frames[frame] = static_cast<int32_t>(frame);
    }
    for (size_t clip = 0; clip < specs.size(); ++clip) {
        const ClipSpec& spec = specs[clip];
        const int32_t start = static_cast<int32_t>(clip) * frames_per_clip;
        database.range_starts[clip] = start;
        database.range_stops[clip] = start + frames_per_clip;
        database.active_hands[clip] = static_cast<uint8_t>(spec.hand);
        write_vec3(database.object_dimensions, clip, spec.dimensions);
        write_vec3(database.grasp_positions_object, clip, spec.grasp_position);
        write_quat(database.grasp_rotations_object, clip, spec.grasp_rotation);
        write_vec3(
            database.approach_directions_object, clip, vec3(1.0F, 0.0F, 0.0F));
        const std::array<uint8_t, frames_per_clip> phases =
            spec.complete_phases
                ? std::array<uint8_t, frames_per_clip>{
                      0U, 1U, 1U, 2U, 3U, 3U, 3U, 4U}
                : std::array<uint8_t, frames_per_clip>{
                      0U, 1U, 1U, 1U, 1U, 1U, 1U, 4U};
        const size_t wrist = spec.hand == interaction::Hand::Right
            ? static_cast<size_t>(g1_skeleton::RightWrist)
            : static_cast<size_t>(g1_skeleton::LeftWrist);
        const size_t elbow = spec.hand == interaction::Hand::Right
            ? static_cast<size_t>(g1_skeleton::RightElbow)
            : static_cast<size_t>(g1_skeleton::LeftElbow);
        for (int32_t local = 0; local < frames_per_clip; ++local) {
            const size_t frame = static_cast<size_t>(start + local);
            database.phases[frame] = phases[static_cast<size_t>(local)];
            write_quat(database.object_rotations, frame, spec.object_rotation);
            vec3 hand = spec.grasp_position;
            if (local < 3) hand.x -= 0.10F * static_cast<float>(3 - local);
            if (local > 3) hand.y += 0.08F * static_cast<float>(local - 3);
            write_vec3(
                database.positions,
                frame * g1_skeleton::BoneCount + wrist,
                vec3(0.25F, -0.05F, 0.0F));
            write_vec3(
                database.positions,
                frame * g1_skeleton::BoneCount + elbow,
                hand + vec3(-0.25F, 0.05F, 0.0F));
            write_quat(
                database.rotations,
                frame * g1_skeleton::BoneCount + wrist,
                spec.grasp_rotation);
        }
    }
    return database;
}

void test_search_is_anchored_to_grasp_not_object_metadata() {
    const quat unrelated_object_rotation =
        quat_from_angle_axis(0.8F, vec3(0.0F, 1.0F, 0.0F));
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat(),
         vec3(0.10F, 0.10F, 0.10F), true, quat()},
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat(),
         vec3(4.0F, 3.0F, 2.0F), true, unrelated_object_rotation},
    });
    interaction::HandTrajectoryQuery query{};
    query.object_world = {
        vec3(9.0F, 4.0F, -7.0F),
        quat_from_angle_axis(-0.6F, vec3(1.0F, 0.0F, 0.0F))};
    query.object_dimensions = vec3(0.02F, 8.0F, 0.50F);
    query.hand = interaction::Hand::Right;
    query.grasp_world_position = vec3(0.10F, 0.20F, 0.30F);
    query.grasp_world_rotation = quat();

    const auto selected = interaction::select_hand_trajectories(database, query);

    require(selected.size() == 2U,
            "object metadata changed grasp-based search membership");
    require(selected[0].cost == selected[1].cost,
            "object metadata changed grasp-based search ranking");
}

void test_object_roll_researches_recorded_world_grasp_orientation() {
    const quat rolled_wrist =
        quat_from_angle_axis(0.30F, vec3(1.0F, 0.0F, 0.0F));
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.0F, 0.0F), quat()},
        {interaction::Hand::Right, vec3(0.10F, 0.0F, 0.0F), rolled_wrist},
    });
    interaction::HandTrajectoryQuery rolled{};
    rolled.object_world = {vec3(), rolled_wrist};
    rolled.object_dimensions = vec3(1.0F, 1.0F, 1.0F);
    rolled.hand = interaction::Hand::Right;
    rolled.grasp_world_position = vec3(0.10F, 0.0F, 0.0F);
    rolled.grasp_world_rotation = rolled_wrist;

    const auto selected = interaction::select_hand_trajectories(
        database, rolled, interaction::HandTrajectoryConfig{});
    require(selected.size() == 2U,
            "rolled grasp unexpectedly removed a bounded candidate");
    require(selected[0].clip == 1,
            "object roll did not rerank by recorded world Contact grasp");
    const interaction::Transform alignment =
        interaction::hand_trajectory_scene_alignment(selected[0], rolled);
    require(near(
                quat_mul_vec3(
                    alignment.rotation, vec3(0.0F, 1.0F, 0.0F)),
                vec3(0.0F, 1.0F, 0.0F)),
            "object roll tilted the candidate body alignment");
}

void test_world_grasp_limits_and_position_only_orientation() {
    const interaction::Database position_database = make_database({
        {interaction::Hand::Right, vec3(0.0F, 0.0F, 0.0F), quat()},
        {interaction::Hand::Right, vec3(0.0F, 0.002F, 0.0F), quat()},
    });
    interaction::HandTrajectoryQuery position_query{};
    position_query.object_world = {vec3(), quat()};
    position_query.object_dimensions = vec3(1.0F, 1.0F, 1.0F);
    position_query.hand = interaction::Hand::Right;
    position_query.grasp_world_position = vec3(0.0F, 0.121F, 0.0F);
    position_query.grasp_world_rotation = quat();
    const auto position_selected = interaction::select_hand_trajectories(
        position_database, position_query);
    require(position_selected.size() == 1U &&
                position_selected[0].clip == 1,
            "0.12 m world-grasp correction gate changed");

    const quat near_orientation =
        quat_from_angle_axis(0.002F, vec3(1.0F, 0.0F, 0.0F));
    const quat target_orientation =
        quat_from_angle_axis(0.438F, vec3(1.0F, 0.0F, 0.0F));
    const interaction::Database orientation_database = make_database({
        {interaction::Hand::Right, vec3(), quat()},
        {interaction::Hand::Right, vec3(), near_orientation},
    });
    interaction::HandTrajectoryQuery orientation_query{};
    orientation_query.object_world = {vec3(), quat()};
    orientation_query.object_dimensions = vec3(1.0F, 1.0F, 1.0F);
    orientation_query.hand = interaction::Hand::Right;
    orientation_query.grasp_world_position = vec3();
    orientation_query.grasp_world_rotation = target_orientation;
    const auto orientation_selected = interaction::select_hand_trajectories(
        orientation_database, orientation_query);
    require(orientation_selected.size() == 1U &&
                orientation_selected[0].clip == 1,
            "25 degree world-grasp orientation gate changed");

    orientation_query.constrain_grasp_orientation = false;
    const auto position_only = interaction::select_hand_trajectories(
        orientation_database, orientation_query);
    require(position_only.size() == 2U && position_only[0].clip == 0,
            "position-only search ranked wrist orientation");
}

interaction::HandTrajectoryQuery identity_query() {
    interaction::HandTrajectoryQuery query{};
    query.object_world = {vec3(), quat()};
    query.object_dimensions = vec3(1.0F, 1.0F, 1.0F);
    query.hand = interaction::Hand::Right;
    query.grasp_world_position = vec3(0.10F, 0.20F, 0.30F);
    query.grasp_world_rotation = quat();
    return query;
}

interaction::ShelfGeometry distant_shelf() {
    interaction::ShelfGeometry shelf{};
    for (interaction::OrientedBox& box : shelf.boxes) {
        box.world = {vec3(100.0F, 100.0F, 100.0F), quat()};
        box.dimensions = vec3(1.0F, 1.0F, 1.0F);
    }
    return shelf;
}

interaction::MappedHandTrajectory mapped_line(
    std::initializer_list<vec3> hands,
    std::initializer_list<vec3> elbows) {
    interaction::MappedHandTrajectory mapped{};
    for (const vec3 hand : hands) {
        mapped.hands.push_back({hand, quat()});
    }
    mapped.elbows.assign(elbows.begin(), elbows.end());
    return mapped;
}

void test_selects_every_close_complete_same_hand_clip_in_stable_order() {
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat()},
        {interaction::Hand::Right, vec3(0.10F, 0.21F, 0.30F), quat()},
        {interaction::Hand::Right, vec3(0.10F, 1.00F, 0.30F), quat()},
        {interaction::Hand::Left, vec3(0.10F, 0.20F, 0.30F), quat()},
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat(),
            vec3(1.0F, 1.0F, 1.0F), false},
    });
    const auto selected = interaction::select_hand_trajectories(
        database, identity_query(), interaction::HandTrajectoryConfig{});
    require(selected.size() == 2U, "selector did not retain exactly all close clips");
    require(selected[0].clip == 0 && selected[1].clip == 1,
            "selector ordering changed");
    require(selected[0].cost <= selected[1].cost, "selector cost order changed");
    require(selected[0].contact_point == 2U, "contact point index changed");
    require(selected[0].lift_frame == 6,
            "trajectory stopped at the first Lift frame");
    require(selected[0].hands_in_source_object.size() == 6U,
            "trajectory does not span Reach through complete Lift");
    require(selected[0].elbows_in_source_object.size() == 6U,
            "elbow trajectory does not span Reach through complete Lift");

    interaction::HandTrajectoryConfig one_only{};
    one_only.maximum_compatible_clips = 1U;
    bool rejected = false;
    try {
        (void)interaction::select_hand_trajectories(
            database, identity_query(), one_only);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "selector silently truncated compatible clips");
}

void test_raw_mapping_uses_upright_scene_alignment_without_grasp_residual() {
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat()},
    });
    const auto selected = interaction::select_hand_trajectories(
        database, identity_query(), interaction::HandTrajectoryConfig{});
    interaction::HandTrajectoryQuery moved = identity_query();
    moved.object_world = {
        vec3(2.0F, 0.5F, -1.0F),
        quat_from_angle_axis(0.7F, vec3(0.0F, 1.0F, 0.0F))};
    moved.grasp_world_position = vec3(2.2F, 1.1F, -0.7F);
    moved.grasp_world_rotation = quat_normalize(quat_mul(
        quat_from_angle_axis(-0.4F, vec3(0.0F, 1.0F, 0.0F)),
        quat_from_angle_axis(0.2F, vec3(1.0F, 0.0F, 0.0F))));
    const auto mapped = interaction::map_hand_trajectory(selected[0], moved);
    require(mapped.elbows.size() == mapped.hands.size(),
            "full-pose mapping omitted elbows");
    const interaction::Transform mapping =
        interaction::hand_trajectory_scene_alignment(selected[0], moved);
    for (size_t sample = 0U; sample < mapped.hands.size(); ++sample) {
        const interaction::Transform source_hand = interaction::compose(
            selected[0].source_object,
            selected[0].hands_in_source_object[sample]);
        const interaction::Transform via_mapping = interaction::compose(
            mapping, source_hand);
        require(near(via_mapping.position, mapped.hands[sample].position),
                "world mapping disagrees with mapped wrist");
        require(near_rotation(via_mapping.rotation, mapped.hands[sample].rotation),
                "world mapping rotation disagrees with mapped wrist");
    }
    const interaction::Transform& contact =
        mapped.hands[selected[0].contact_point];
    require(!near(contact.position, moved.grasp_world_position),
            "raw scene mapping incorrectly forced Contact position");
    require(!near_rotation(contact.rotation, moved.grasp_world_rotation),
            "raw scene mapping incorrectly forced Contact orientation");
}

void test_position_only_mapping_preserves_object_mapped_contact_rotation() {
    const quat source_rotation =
        quat_from_angle_axis(0.25F, vec3(0.0F, 1.0F, 0.0F));
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), source_rotation},
    });
    interaction::HandTrajectoryQuery selection_query = identity_query();
    selection_query.grasp_world_rotation = source_rotation;
    const auto selected = interaction::select_hand_trajectories(
        database, selection_query, interaction::HandTrajectoryConfig{});
    interaction::HandTrajectoryQuery position_only = selection_query;
    position_only.object_world = {
        vec3(-1.0F, 0.2F, 3.0F),
        quat_from_angle_axis(0.5F, vec3(0.0F, 1.0F, 0.0F))};
    position_only.grasp_world_position = vec3(-0.5F, 0.9F, 2.8F);
    position_only.constrain_grasp_orientation = false;
    const auto mapped = interaction::map_hand_trajectory(selected[0], position_only);
    const interaction::Transform& contact =
        mapped.hands[selected[0].contact_point];
    const quat expected_rotation = source_rotation;
    require(!near(contact.position, position_only.grasp_world_position),
            "raw position-only mapping incorrectly forced Contact position");
    require(near_rotation(contact.rotation, expected_rotation),
            "position-only mapping changed Contact rotation");
}

void test_shape_converges_contact_without_moving_root_or_legs() {
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat()},
    });
    interaction::HandTrajectoryQuery query = identity_query();
    query.object_world = {
        vec3(0.35F, 0.60F, -0.20F),
        quat_from_angle_axis(0.25F, vec3(0.0F, 1.0F, 0.0F))};
    const interaction::Transform source_grasp{
        vec3(0.10F, 0.20F, 0.30F), quat()};
    const interaction::Transform alignment{
        vec3(0.35F, 0.0F, -0.20F), query.object_world.rotation};
    const interaction::Transform raw_contact = interaction::compose(
        alignment, source_grasp);
    query.grasp_world_position =
        raw_contact.position + vec3(0.0F, 0.05F, 0.0F);
    query.grasp_world_rotation = raw_contact.rotation;
    const auto selected = interaction::select_hand_trajectories(database, query);
    require(selected.size() == 1U, "bounded shaping candidate was not selected");

    const interaction::ShapedHandTrajectory shaped =
        interaction::shape_hand_trajectory(
            database, selected[0], query, interaction::IKConfig{});

    require(shaped.poses.size() == selected[0].hands_in_source_object.size(),
            "shaping did not return one pose per trajectory sample");
    require(shaped.path.hands.size() == shaped.poses.size() &&
                shaped.path.elbows.size() == shaped.poses.size(),
            "shaping path and pose counts disagree");
    require(shaped.contact_accepted,
            "bounded Contact correction was not accepted");
    require(shaped.reason == interaction::Reason::None,
            "accepted Contact correction reported a failure reason");
    const vec3 shaped_contact =
        shaped.path.hands[selected[0].contact_point].position;
    require(length(shaped_contact - query.grasp_world_position) <=
                interaction::IKConfig{}.accepted_position_m,
            "shaped Contact wrist did not converge to the requested grasp");
    require(length(shaped_contact - query.grasp_world_position) <
                length(raw_contact.position - query.grasp_world_position),
            "arm IK did not improve the Contact wrist error");

    const interaction::Transform scene_alignment =
        interaction::hand_trajectory_scene_alignment(selected[0], query);
    for (size_t sample = 0U; sample < shaped.poses.size(); ++sample) {
        interaction::Pose expected = interaction::pose_at_frame(
            database,
            selected[0].reach_frame + static_cast<int32_t>(sample));
        const interaction::Transform root = interaction::compose(
            scene_alignment,
            {expected.positions[g1_skeleton::Simulation],
             expected.rotations[g1_skeleton::Simulation]});
        expected.positions[g1_skeleton::Simulation] = root.position;
        expected.rotations[g1_skeleton::Simulation] = root.rotation;
        for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
            require(near(shaped.poses[sample].positions[bone],
                         expected.positions[bone]),
                    "IK shaping changed a local bone position");
            if (!active_arm_bone(bone, query.hand)) {
                require(near_rotation(shaped.poses[sample].rotations[bone],
                                      expected.rotations[bone]),
                        "IK shaping rotated the root, torso, or legs");
            }
        }
    }
}

void test_shape_rejects_contact_correction_above_solver_envelope() {
    const interaction::Database database = make_database({
        {interaction::Hand::Right, vec3(0.10F, 0.20F, 0.30F), quat()},
    });
    const auto selected = interaction::select_hand_trajectories(
        database, identity_query());
    interaction::HandTrajectoryQuery over_limit = identity_query();
    over_limit.grasp_world_position.y +=
        interaction::IKConfig{}.maximum_request_position_m + 0.001F;

    const interaction::ShapedHandTrajectory shaped =
        interaction::shape_hand_trajectory(
            database, selected[0], over_limit, interaction::IKConfig{});

    require(!shaped.contact_accepted,
            "over-limit Contact correction was accepted");
    require(shaped.reason == interaction::Reason::CorrectionLimit,
            "over-limit Contact correction reported the wrong reason");
}

void test_collision_feasibility_is_phase_aware() {
    const interaction::OrientedBox object{
        {vec3(), quat()}, vec3(1.0F, 1.0F, 1.0F)};
    const interaction::ShelfGeometry far_shelf = distant_shelf();

    const auto clear = mapped_line(
        {vec3(2.0F, 2.0F, 2.0F), vec3(1.5F, 2.0F, 2.0F)},
        {vec3(2.5F, 2.0F, 2.0F), vec3(2.0F, 2.0F, 2.0F)});
    require(interaction::evaluate_trajectory_feasibility(
                clear, 1U, object, far_shelf).reason ==
            interaction::TrajectoryFeasibilityReason::None,
            "clear trajectory was rejected");

    const auto precontact_object = mapped_line(
        {vec3(), vec3(1.5F, 2.0F, 2.0F)},
        {vec3(1.0F, 0.0F, 0.0F), vec3(2.0F, 2.0F, 2.0F)});
    const auto object_rejected = interaction::evaluate_trajectory_feasibility(
        precontact_object, 1U, object, far_shelf);
    require(object_rejected.reason ==
                interaction::TrajectoryFeasibilityReason::ObjectCollision &&
            object_rejected.sample == 0U,
            "pre-Contact object penetration was not identified");

    const auto contact_object = mapped_line(
        {vec3(2.0F, 2.0F, 2.0F), vec3()},
        {vec3(2.5F, 2.0F, 2.0F), vec3(1.0F, 0.0F, 0.0F)});
    require(interaction::evaluate_trajectory_feasibility(
                contact_object, 1U, object, far_shelf).reason ==
            interaction::TrajectoryFeasibilityReason::None,
            "Contact object overlap was not exempted");
}

void test_forearm_capsule_and_contact_still_collide_with_shelf() {
    const interaction::OrientedBox object{
        {vec3(), quat()}, vec3(0.2F, 0.2F, 0.2F)};
    interaction::ShelfGeometry shelf = distant_shelf();
    shelf.boxes[0] = {
        {vec3(3.0F, 0.0F, 0.0F), quat()}, vec3(0.2F, 1.0F, 1.0F)};
    const auto forearm_crossing = mapped_line(
        {vec3(2.0F, 0.0F, 0.0F)},
        {vec3(4.0F, 0.0F, 0.0F)});
    require(interaction::evaluate_trajectory_feasibility(
                forearm_crossing, 0U, object, shelf).reason ==
            interaction::TrajectoryFeasibilityReason::ShelfCollision,
            "forearm capsule crossing shelf was not rejected");

    shelf.boxes[0] = object;
    const auto contact_shelf = mapped_line(
        {vec3()}, {vec3(1.0F, 0.0F, 0.0F)});
    require(interaction::evaluate_trajectory_feasibility(
                contact_shelf, 0U, object, shelf).reason ==
            interaction::TrajectoryFeasibilityReason::ShelfCollision,
            "shelf collision was incorrectly exempted at Contact");
}

void test_recorded_table_geometry_preserves_top_and_builds_four_legs() {
    const interaction::Transform table{
        vec3(1.0F, 0.40F, -2.0F), quat()};
    const vec3 dimensions(1.20F, 0.08F, 0.70F);
    const interaction::ShelfGeometry geometry =
        interaction::make_recorded_table_geometry(table, dimensions);
    require(near(geometry.boxes[0].world.position, table.position),
            "tabletop position changed");
    require(near(geometry.boxes[0].dimensions, dimensions),
            "tabletop dimensions changed");
    for (size_t leg = 1U; leg < geometry.boxes.size(); ++leg) {
        require(geometry.boxes[leg].dimensions.x == 0.04F &&
                geometry.boxes[leg].dimensions.z == 0.04F &&
                geometry.boxes[leg].dimensions.y > 0.0F &&
                geometry.boxes[leg].world.position.y < table.position.y,
                "table leg is not slender and below tabletop");
    }

    bool rejected = false;
    try {
        (void)interaction::make_recorded_table_geometry(
            {vec3(0.0F, 0.03F, 0.0F), quat()}, dimensions);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "below-ground recorded table was accepted");
}

}  // namespace

int main() {
    test_search_is_anchored_to_grasp_not_object_metadata();
    test_object_roll_researches_recorded_world_grasp_orientation();
    test_world_grasp_limits_and_position_only_orientation();
    test_selects_every_close_complete_same_hand_clip_in_stable_order();
    test_raw_mapping_uses_upright_scene_alignment_without_grasp_residual();
    test_position_only_mapping_preserves_object_mapped_contact_rotation();
    test_shape_converges_contact_without_moving_root_or_legs();
    test_shape_rejects_contact_correction_above_solver_envelope();
    test_collision_feasibility_is_phase_aware();
    test_forearm_capsule_and_contact_still_collide_with_shelf();
    test_recorded_table_geometry_preserves_top_and_builds_four_legs();
    return 0;
}
