#include "interaction_hand_trajectories.h"

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

struct ClipSpec {
    interaction::Hand hand = interaction::Hand::Right;
    vec3 grasp_position{};
    quat grasp_rotation{};
    vec3 dimensions{1.0F, 1.0F, 1.0F};
    bool complete_phases = true;
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
    constexpr int32_t frames_per_clip = 6;
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
                ? std::array<uint8_t, frames_per_clip>{0U, 1U, 1U, 2U, 3U, 4U}
                : std::array<uint8_t, frames_per_clip>{0U, 1U, 1U, 1U, 1U, 4U};
        const size_t wrist = spec.hand == interaction::Hand::Right
            ? static_cast<size_t>(g1_skeleton::RightWrist)
            : static_cast<size_t>(g1_skeleton::LeftWrist);
        const size_t elbow = spec.hand == interaction::Hand::Right
            ? static_cast<size_t>(g1_skeleton::RightElbow)
            : static_cast<size_t>(g1_skeleton::LeftElbow);
        for (int32_t local = 0; local < frames_per_clip; ++local) {
            const size_t frame = static_cast<size_t>(start + local);
            database.phases[frame] = phases[static_cast<size_t>(local)];
            vec3 hand = spec.grasp_position;
            if (local < 3) hand.x -= 0.10F * static_cast<float>(3 - local);
            if (local > 3) hand.y += 0.08F * static_cast<float>(local - 3);
            write_vec3(
                database.positions,
                frame * g1_skeleton::BoneCount + wrist,
                hand);
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
        {interaction::Hand::Right, vec3(0.11F, 0.20F, 0.30F), quat()},
        {interaction::Hand::Right, vec3(1.00F, 0.20F, 0.30F), quat()},
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
    require(selected[0].hands_in_source_object.size() == 4U,
            "trajectory does not span Reach through Lift");
    require(selected[0].elbows_in_source_object.size() == 4U,
            "elbow trajectory does not span Reach through Lift");

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

void test_full_pose_mapping_converges_exactly_at_contact() {
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
    moved.grasp_world_rotation =
        quat_from_angle_axis(-0.4F, vec3(0.0F, 1.0F, 0.0F));
    const auto mapped = interaction::map_hand_trajectory(selected[0], moved);
    const interaction::Transform& contact =
        mapped.hands[selected[0].contact_point];
    require(near(contact.position, moved.grasp_world_position),
            "full-pose mapping missed requested Contact position");
    require(near_rotation(contact.rotation, *moved.grasp_world_rotation),
            "full-pose mapping missed requested Contact rotation");
    require(mapped.elbows.size() == mapped.hands.size(),
            "full-pose mapping omitted elbows");
    const interaction::Transform mapping =
        interaction::hand_trajectory_world_mapping(selected[0], moved);
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
    position_only.grasp_world_rotation.reset();
    const auto mapped = interaction::map_hand_trajectory(selected[0], position_only);
    const interaction::Transform& contact =
        mapped.hands[selected[0].contact_point];
    const quat expected_rotation = quat_normalize(quat_mul(
        position_only.object_world.rotation, source_rotation));
    require(near(contact.position, position_only.grasp_world_position),
            "position-only mapping missed requested Contact position");
    require(near_rotation(contact.rotation, expected_rotation),
            "position-only mapping changed Contact rotation");
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
    test_selects_every_close_complete_same_hand_clip_in_stable_order();
    test_full_pose_mapping_converges_exactly_at_contact();
    test_position_only_mapping_preserves_object_mapped_contact_rotation();
    test_collision_feasibility_is_phase_aware();
    test_forearm_capsule_and_contact_still_collide_with_shelf();
    test_recorded_table_geometry_preserves_top_and_builds_four_legs();
    return 0;
}
