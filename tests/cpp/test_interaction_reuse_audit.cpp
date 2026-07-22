#include "interaction_reuse_audit.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

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

struct AuditClip {
    interaction::Hand hand = interaction::Hand::Right;
    float contact_height = 0.20F;
    float root_z = 1.0F;
};

interaction::Database make_audit_database() {
    const std::array<AuditClip, 4> clips{{
        {interaction::Hand::Left, 0.20F, 1.0F},
        {interaction::Hand::Right, -0.30F, 1.5F},
        {interaction::Hand::Right, 0.20F, -1.0F},
        {interaction::Hand::Right, 0.20F, 1.0F},
    }};
    constexpr size_t frames_per_clip = 8U;
    const size_t frames = clips.size() * frames_per_clip;
    interaction::Database database{};
    database.frame_count = static_cast<uint32_t>(frames);
    database.bone_count = g1_skeleton::BoneCount;
    database.clip_count = static_cast<uint32_t>(clips.size());
    database.hand_dof_count = 14U;
    database.parents.assign(
        g1_skeleton::kParents.begin(), g1_skeleton::kParents.end());
    database.range_starts.resize(clips.size());
    database.range_stops.resize(clips.size());
    database.positions.assign(frames * g1_skeleton::BoneCount * 3U, 0.0F);
    database.rotations.assign(frames * g1_skeleton::BoneCount * 4U, 0.0F);
    database.phases.assign(frames, 0U);
    database.active_hands.resize(clips.size());
    database.object_positions.assign(frames * 3U, 0.0F);
    database.object_rotations.assign(frames * 4U, 0.0F);
    database.table_positions.assign(clips.size() * 3U, 0.0F);
    database.table_rotations.assign(clips.size() * 4U, 0.0F);
    database.table_sizes.assign(clips.size() * 3U, 0.0F);
    database.object_dimensions.assign(clips.size() * 3U, 0.0F);
    database.grasp_positions_object.assign(clips.size() * 3U, 0.0F);
    database.grasp_rotations_object.assign(clips.size() * 4U, 0.0F);
    database.approach_directions_object.assign(clips.size() * 3U, 0.0F);

    for (size_t frame = 0U; frame < frames; ++frame) {
        for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
            write_quat(
                database.rotations,
                frame * g1_skeleton::BoneCount + bone,
                quat());
        }
        write_quat(database.object_rotations, frame, quat());
    }
    constexpr std::array<uint8_t, frames_per_clip> phases{{
        0U, 1U, 1U, 2U, 3U, 3U, 3U, 4U,
    }};
    for (size_t clip = 0U; clip < clips.size(); ++clip) {
        const AuditClip& spec = clips[clip];
        const size_t start = clip * frames_per_clip;
        database.range_starts[clip] = static_cast<int32_t>(start);
        database.range_stops[clip] =
            static_cast<int32_t>(start + frames_per_clip);
        database.active_hands[clip] = static_cast<uint8_t>(spec.hand);
        write_vec3(database.table_positions, clip, vec3(0.0F, 0.40F, 0.0F));
        write_quat(database.table_rotations, clip, quat());
        write_vec3(database.table_sizes, clip, vec3(1.2F, 0.08F, 0.8F));
        write_vec3(
            database.object_dimensions, clip, vec3(0.05F, 0.05F, 0.05F));
        write_vec3(database.grasp_positions_object, clip, vec3());
        write_quat(database.grasp_rotations_object, clip, quat());
        write_vec3(
            database.approach_directions_object,
            clip,
            vec3(1.0F, 0.0F, 0.0F));
        const size_t wrist = spec.hand == interaction::Hand::Right
            ? static_cast<size_t>(g1_skeleton::RightWrist)
            : static_cast<size_t>(g1_skeleton::LeftWrist);
        const size_t elbow = spec.hand == interaction::Hand::Right
            ? static_cast<size_t>(g1_skeleton::RightElbow)
            : static_cast<size_t>(g1_skeleton::LeftElbow);
        for (size_t local = 0U; local < frames_per_clip; ++local) {
            const size_t frame = start + local;
            database.phases[frame] = phases[local];
            const float approach = local < 3U
                ? -0.10F * static_cast<float>(3U - local)
                : 0.0F;
            const float lift = local > 3U && local < 7U
                ? 0.04F * static_cast<float>(local - 3U)
                : 0.0F;
            write_vec3(
                database.positions,
                frame * g1_skeleton::BoneCount +
                    static_cast<size_t>(g1_skeleton::Simulation),
                vec3(0.0F, 0.0F, spec.root_z));
            write_vec3(
                database.positions,
                frame * g1_skeleton::BoneCount + elbow,
                vec3(-0.15F + approach,
                     spec.contact_height + lift,
                     -spec.root_z));
            write_vec3(
                database.positions,
                frame * g1_skeleton::BoneCount + wrist,
                vec3(0.25F, 0.0F, 0.0F));
        }
    }
    return database;
}

interaction::HandTrajectoryQuery audit_query() {
    interaction::HandTrajectoryQuery query{};
    query.object_world = {vec3(0.10F, 0.20F, 0.0F), quat()};
    query.object_dimensions = vec3(0.05F, 0.05F, 0.05F);
    query.hand = interaction::Hand::Right;
    query.grasp_world_position = query.object_world.position;
    query.grasp_world_rotation = quat();
    query.approach_world_direction = vec3(1.0F, 0.0F, 0.0F);
    query.orientation_mode = interaction::GraspOrientationMode::ApproachAxis;
    return query;
}

interaction::ReuseAuditResult run_audit(size_t workers) {
    const interaction::Database database = make_audit_database();
    const interaction::OrientedBox object{
        {audit_query().grasp_world_position, quat()},
        vec3(0.05F, 0.05F, 0.05F)};
    interaction::EnvironmentGeometry environment{};
    environment.boxes.push_back({
        {vec3(0.0F, 0.0F, -1.0F), quat()},
        vec3(0.10F, 0.10F, 0.10F),
    });
    interaction::TrajectoryCollisionConfig collision{};
    collision.wrist_radius_m = 0.005F;
    collision.forearm_radius_m = 0.005F;
    collision.joint_radius_m = 0.005F;
    collision.limb_radius_m = 0.005F;
    collision.torso_radius_m = 0.005F;
    interaction::ReuseAuditConfig config{};
    config.worker_count = workers;
    return interaction::audit_reusable_hand_trajectories(
        database, audit_query(), object, environment, collision, config);
}

void require_expected_counts(const interaction::ReuseAuditResult& result) {
    require(result.status == interaction::ReuseAuditStatus::Complete,
            "synthetic audit did not complete");
    require(result.counts.total == 3U && result.counts.processed == 3U,
            "audit population counts changed");
    require(result.counts.contact_accepted == 2U,
            "Contact survivor count changed");
    require(result.counts.fully_shaped == 2U,
            "full shaping count changed");
    require(result.counts.object_rejected == 0U,
            "clear synthetic motions hit the object");
    require(result.counts.environment_rejected == 1U,
            "environment rejection count changed");
    require(result.counts.reusable == 1U && result.displayed.size() == 1U,
            "reusable result count changed");
    require(result.displayed[0].source.clip == 3,
            "wrong synthetic trajectory was displayed");
}

void test_audit_is_exhaustive_and_worker_count_deterministic() {
    const interaction::ReuseAuditResult serial = run_audit(1U);
    const interaction::ReuseAuditResult parallel = run_audit(4U);
    require_expected_counts(serial);
    require_expected_counts(parallel);
    require(serial.displayed[0].source.clip ==
                parallel.displayed[0].source.clip,
            "worker count changed displayed ordering");
}

void test_zero_deadline_returns_no_partial_results() {
    const interaction::Database database = make_audit_database();
    interaction::ReuseAuditConfig config{};
    config.deadline_milliseconds = 0U;
    const interaction::ReuseAuditResult result =
        interaction::audit_reusable_hand_trajectories(
            database,
            audit_query(),
            {{audit_query().grasp_world_position, quat()},
             vec3(0.05F, 0.05F, 0.05F)},
            interaction::EnvironmentGeometry{},
            interaction::TrajectoryCollisionConfig{},
            config);
    require(result.status == interaction::ReuseAuditStatus::Incomplete,
            "zero-deadline audit claimed completion");
    require(result.counts.processed < result.counts.total,
            "zero-deadline audit processed the full population");
    require(result.displayed.empty(),
            "incomplete audit published partial display results");
}

template<class Update>
void require_invalid_config(Update update, const char* message) {
    const interaction::Database database = make_audit_database();
    interaction::ReuseAuditConfig config{};
    update(config);
    bool rejected = false;
    try {
        (void)interaction::audit_reusable_hand_trajectories(
            database,
            audit_query(),
            {{audit_query().grasp_world_position, quat()},
             vec3(0.05F, 0.05F, 0.05F)},
            interaction::EnvironmentGeometry{},
            interaction::TrajectoryCollisionConfig{},
            config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

void test_invalid_audit_config_is_rejected() {
    require_invalid_config(
        [](interaction::ReuseAuditConfig& value) { value.worker_count = 0U; },
        "zero workers were accepted");
    require_invalid_config(
        [](interaction::ReuseAuditConfig& value) { value.worker_count = 5U; },
        "more than four workers were accepted");
    require_invalid_config(
        [](interaction::ReuseAuditConfig& value) { value.display_limit = 0U; },
        "zero display limit was accepted");
    require_invalid_config(
        [](interaction::ReuseAuditConfig& value) {
            value.maximum_contact_correction_m = -0.1F;
        },
        "negative correction envelope was accepted");
    require_invalid_config(
        [](interaction::ReuseAuditConfig& value) {
            value.maximum_contact_correction_m =
                std::numeric_limits<float>::quiet_NaN();
        },
        "non-finite correction envelope was accepted");
}

}  // namespace

int main() {
    test_audit_is_exhaustive_and_worker_count_deterministic();
    test_zero_deadline_returns_no_partial_results();
    test_invalid_audit_config_is_rejected();
    return 0;
}
