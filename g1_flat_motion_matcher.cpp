#include "g1_flat_motion_matcher.h"

#include "database.h"
#include "g1_skeleton.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace episode {
namespace {

constexpr float kStep = 1.0F / 25.0F;
constexpr size_t kFlatBoneCount = kFlatSkeletonBoneCount;
struct FlatPose {
    std::array<vec3, kFlatBoneCount> positions{};
    std::array<vec3, kFlatBoneCount> velocities{};
    std::array<quat, kFlatBoneCount> rotations{};
    std::array<vec3, kFlatBoneCount> angular_velocities{};
    std::array<uint8_t, 2U> foot_contacts{};
};

struct FlatWorldPose {
    std::array<vec3, kFlatBoneCount> positions{};
    std::array<vec3, kFlatBoneCount> velocities{};
    std::array<quat, kFlatBoneCount> rotations{};
    std::array<vec3, kFlatBoneCount> angular_velocities{};
};

interaction::Transform root_transform(const interaction::Pose& pose) {
    return {pose.positions[0], pose.rotations[0]};
}

interaction::Transform source_root(
    const database& source,
    int frame) {
    return {
        source.bone_positions(frame, g1_skeleton::Simulation),
        source.bone_rotations(frame, g1_skeleton::Simulation),
    };
}

float smoothstep(float value) {
    value = std::clamp(value, 0.0F, 1.0F);
    return value * value * (3.0F - 2.0F * value);
}

interaction::Pose database_pose31(const database& source, int frame) {
    interaction::Pose pose{};
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        pose.positions[bone] =
            source.bone_positions(frame, static_cast<int>(bone));
        pose.velocities[bone] =
            source.bone_velocities(frame, static_cast<int>(bone));
        pose.rotations[bone] =
            source.bone_rotations(frame, static_cast<int>(bone));
        pose.angular_velocities[bone] =
            source.bone_angular_velocities(frame, static_cast<int>(bone));
    }
    pose.foot_contacts = {
        static_cast<uint8_t>(source.contact_states(frame, 0)),
        static_cast<uint8_t>(source.contact_states(frame, 1)),
    };
    return pose;
}

FlatPose database_pose23(const database& source, int frame) {
    FlatPose pose{};
    for (size_t bone = 0U; bone < kFlatBoneCount; ++bone) {
        pose.positions[bone] =
            source.bone_positions(frame, static_cast<int>(bone));
        pose.velocities[bone] =
            source.bone_velocities(frame, static_cast<int>(bone));
        pose.rotations[bone] =
            source.bone_rotations(frame, static_cast<int>(bone));
        pose.angular_velocities[bone] =
            source.bone_angular_velocities(frame, static_cast<int>(bone));
    }
    pose.foot_contacts = {
        static_cast<uint8_t>(source.contact_states(frame, 0)),
        static_cast<uint8_t>(source.contact_states(frame, 1)),
    };
    return pose;
}

FlatWorldPose flat_world_pose(const FlatPose& pose) {
    FlatWorldPose world{};
    for (size_t bone = 0U; bone < kFlatBoneCount; ++bone) {
        const int32_t parent = kFlatSkeletonParents[bone];
        if (parent < 0) {
            world.positions[bone] = pose.positions[bone];
            world.velocities[bone] = pose.velocities[bone];
            world.rotations[bone] = quat_normalize(pose.rotations[bone]);
            world.angular_velocities[bone] = pose.angular_velocities[bone];
            continue;
        }
        const size_t parent_bone = static_cast<size_t>(parent);
        const vec3 offset = quat_mul_vec3(
            world.rotations[parent_bone], pose.positions[bone]);
        world.positions[bone] = world.positions[parent_bone] + offset;
        world.rotations[bone] = quat_normalize(quat_mul(
            world.rotations[parent_bone], pose.rotations[bone]));
        world.velocities[bone] =
            world.velocities[parent_bone] +
            cross(world.angular_velocities[parent_bone], offset) +
            quat_mul_vec3(
                world.rotations[parent_bone], pose.velocities[bone]);
        world.angular_velocities[bone] =
            world.angular_velocities[parent_bone] +
            quat_mul_vec3(
                world.rotations[parent_bone],
                pose.angular_velocities[bone]);
    }
    return world;
}

}  // namespace

FlatMotionMatcher::FlatMotionMatcher(
    const std::filesystem::path& database_path,
    const std::filesystem::path& g1_reference_database) {
    if (!g1_reference_database.empty()) {
        database reference{};
        database_load(
            reference, g1_reference_database.string().c_str());
        if (reference.nbones() != g1_skeleton::BoneCount ||
            reference.nframes() < 1) {
            throw std::invalid_argument(
                "invalid G1 flat matcher geometry reference");
        }
        geometry_reference_ = database_pose31(
            reference, reference.range_starts(0));
        has_geometry_reference_ = true;
    }
    database source{};
    database_load(source, database_path.string().c_str());
    prepare(source);
}

FlatMotionMatcher::~FlatMotionMatcher() = default;

void FlatMotionMatcher::prepare(const database& input) {
    database source{};
    source = input;
    const bool flat_controller =
        source.nbones() == static_cast<int>(kFlatBoneCount);
    if ((!flat_controller &&
         source.nbones() != g1_skeleton::BoneCount) ||
        source.nframes() < 27 || source.nranges() < 1) {
        throw std::invalid_argument("invalid flat G1 motion database");
    }
    if (flat_controller && !has_geometry_reference_) {
        throw std::invalid_argument(
            "23-bone locomotion requires a G1 geometry reference");
    }
    for (int bone = 0; bone < source.nbones(); ++bone) {
        const int32_t expected = flat_controller
            ? kFlatSkeletonParents[static_cast<size_t>(bone)]
            : g1_skeleton::kParents[static_cast<size_t>(bone)];
        if (source.bone_parents(bone) != expected) {
            throw std::invalid_argument("flat G1 skeleton mismatch");
        }
    }
    source.terrain_features.resize(source.nframes(), 4);
    source.terrain_features.zero();
    database_build_matching_features(
        source, 0.75F, 1.0F, 1.0F, 1.0F, 1.5F,
        flat_controller ? 5 : g1_skeleton::LeftToe,
        flat_controller ? 9 : g1_skeleton::RightToe,
        flat_controller ? 1 : g1_skeleton::Hips,
        0.0F);
    if (source.nfeatures() != 31) {
        throw std::invalid_argument("flat G1 feature build failed");
    }
    const interaction::Pose live = snapshot_.pose;
    const bool replacing = database_ != nullptr;
    database_ = std::make_unique<database>();
    *database_ = source;
    uses_flat_controller_skeleton_ = flat_controller;
    source_frame_ = database_->range_starts(0);
    search_seconds_ = 0.10F;
    accumulator_seconds_ = 0.0F;
    if (replacing) {
        transition_source_ = live;
        world_from_source_ = interaction::compose(
            root_transform(live),
            interaction::inverse(source_root(*database_, source_frame_)));
        snapshot_.pose = live;
        transition_seconds_ = 0.0F;
    } else {
        world_from_source_ = interaction::compose(
            {
                vec3(0.0F, 0.0F, -1.40F),
                quat(),
            },
            interaction::inverse(source_root(
                *database_, source_frame_)));
        snapshot_.pose = mapped_pose(source_frame_);
        transition_source_ = snapshot_.pose;
        transition_seconds_ = 0.20F;
    }
}

interaction::Pose FlatMotionMatcher::mapped_pose(int frame) const {
    if (uses_flat_controller_skeleton_) {
        FlatPose current = database_pose23(*database_, frame);
        const interaction::Transform root = interaction::compose(
            world_from_source_,
            {current.positions[0], current.rotations[0]});
        current.positions[0] = root.position;
        current.rotations[0] = root.rotation;
        current.velocities[0] = quat_mul_vec3(
            world_from_source_.rotation, current.velocities[0]);
        current.angular_velocities[0] = quat_mul_vec3(
            world_from_source_.rotation,
            current.angular_velocities[0]);
        const FlatWorldPose flat_world = flat_world_pose(current);
        flat_skeleton_.valid = true;
        flat_skeleton_.positions = flat_world.positions;
        flat_skeleton_.rotations = flat_world.rotations;

        interaction::Pose proxy = geometry_reference_;
        proxy.positions[g1_skeleton::Simulation] = current.positions[0];
        proxy.velocities[g1_skeleton::Simulation] = current.velocities[0];
        proxy.rotations[g1_skeleton::Simulation] = current.rotations[0];
        proxy.angular_velocities[g1_skeleton::Simulation] =
            current.angular_velocities[0];
        proxy.foot_contacts = current.foot_contacts;
        return proxy;
    }
    flat_skeleton_.valid = false;
    interaction::Pose pose{};
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        pose.positions[bone] =
            database_->bone_positions(frame, static_cast<int>(bone));
        pose.velocities[bone] =
            database_->bone_velocities(frame, static_cast<int>(bone));
        pose.rotations[bone] =
            database_->bone_rotations(frame, static_cast<int>(bone));
        pose.angular_velocities[bone] =
            database_->bone_angular_velocities(frame, static_cast<int>(bone));
    }
    pose.foot_contacts = {
        static_cast<uint8_t>(database_->contact_states(frame, 0)),
        static_cast<uint8_t>(database_->contact_states(frame, 1)),
    };
    const interaction::Transform root = interaction::compose(
        world_from_source_, root_transform(pose));
    pose.positions[0] = root.position;
    pose.rotations[0] = root.rotation;
    pose.velocities[0] = quat_mul_vec3(
        world_from_source_.rotation, pose.velocities[0]);
    pose.angular_velocities[0] = quat_mul_vec3(
        world_from_source_.rotation, pose.angular_velocities[0]);
    return pose;
}

void FlatMotionMatcher::tick(const LocomotionCommand& command) {
    search_seconds_ += kStep;
    if (search_seconds_ >= 0.10F) {
        array1d<float> query(database_->nfeatures());
        for (int dimension = 0; dimension < 15; ++dimension) {
            query(dimension) =
                database_->features(source_frame_, dimension) *
                    database_->features_scale(dimension) +
                database_->features_offset(dimension);
        }
        const quat inverse_root =
            quat_inv(snapshot_.pose.rotations[0]);
        const float horizons[3] = {0.32F, 0.68F, 1.0F};
        for (int sample = 0; sample < 3; ++sample) {
            const vec3 local = quat_mul_vec3(
                inverse_root,
                horizons[sample] * command.desired_velocity_world);
            query(15 + sample * 2) = local.x;
            query(16 + sample * 2) = local.z;
            const vec3 direction = quat_mul_vec3(
                inverse_root,
                quat_mul_vec3(
                    command.desired_heading_world,
                    vec3(0.0F, 0.0F, 1.0F)));
            query(21 + sample * 2) = direction.x;
            query(22 + sample * 2) = direction.z;
        }
        for (int dimension = 27; dimension < 31; ++dimension) {
            query(dimension) = 0.0F;
        }
        int selected = source_frame_;
        float cost = 0.0F;
        database_search(selected, cost, *database_, query);
        if (selected != source_frame_) {
            const interaction::Pose live = snapshot_.pose;
            flat_transition_source_ = flat_skeleton_;
            source_frame_ = selected;
            world_from_source_ = interaction::compose(
                root_transform(live),
                interaction::inverse(source_root(
                    *database_, source_frame_)));
            transition_source_ = live;
            transition_seconds_ = 0.0F;
        }
        search_seconds_ = 0.0F;
    }
    int next = database_trajectory_index_clamp(
        *database_, source_frame_, 1);
    if (next == source_frame_) {
        const interaction::Pose live = snapshot_.pose;
        source_frame_ = database_->range_starts(0);
        world_from_source_ = interaction::compose(
            root_transform(live),
            interaction::inverse(source_root(
                *database_, source_frame_)));
    } else {
        source_frame_ = next;
    }
    interaction::Pose target = mapped_pose(source_frame_);
    const FlatSkeletonWorldPose flat_target = flat_skeleton_;
    transition_seconds_ += kStep;
    if (transition_seconds_ < 0.20F) {
        const float alpha = smoothstep(transition_seconds_ / 0.20F);
        const vec3 root_position = target.positions[0];
        const quat root_rotation = target.rotations[0];
        target = interaction::interpolate_pose(
            transition_source_, target, alpha);
        target.positions[0] = root_position;
        target.rotations[0] = root_rotation;
        if (uses_flat_controller_skeleton_ &&
            flat_transition_source_.valid && flat_target.valid) {
            flat_skeleton_.valid = true;
            for (size_t bone = 0U; bone < kFlatBoneCount; ++bone) {
                flat_skeleton_.positions[bone] = lerp(
                    flat_transition_source_.positions[bone],
                    flat_target.positions[bone],
                    alpha);
                flat_skeleton_.rotations[bone] = quat_nlerp_shortest(
                    flat_transition_source_.rotations[bone],
                    flat_target.rotations[bone],
                    alpha);
            }
            flat_skeleton_.positions[0] = flat_target.positions[0];
            flat_skeleton_.rotations[0] = flat_target.rotations[0];
        }
    }
    snapshot_.pose = target;
    const float future[3] = {0.32F, 0.68F, 1.0F};
    for (size_t sample = 0U; sample < 3U; ++sample) {
        snapshot_.future_root_positions[sample] =
            snapshot_.pose.positions[0] +
            future[sample] * command.desired_velocity_world;
        snapshot_.future_root_rotations[sample] =
            command.desired_heading_world;
    }
}

void FlatMotionMatcher::update(
    const LocomotionCommand& command,
    float dt) {
    if (!std::isfinite(dt) || dt < 0.0F) {
        throw std::invalid_argument("invalid flat motion matcher dt");
    }
    accumulator_seconds_ += dt;
    while (accumulator_seconds_ + 1.0e-7F >= kStep) {
        tick(command);
        accumulator_seconds_ -= kStep;
    }
}

void FlatMotionMatcher::switch_database(
    const std::filesystem::path& database_path) {
    database source{};
    database_load(source, database_path.string().c_str());
    prepare(source);
}

void FlatMotionMatcher::switch_database(
    const std::filesystem::path& database_path,
    const interaction::Pose& live_pose) {
    snapshot_.pose = live_pose;
    switch_database(database_path);
}

void FlatMotionMatcher::reset_database(
    const std::filesystem::path& database_path) {
    database source{};
    database_load(source, database_path.string().c_str());
    database_.reset();
    prepare(source);
}

const interaction::LocomotionSnapshot&
FlatMotionMatcher::snapshot() const {
    return snapshot_;
}

const FlatSkeletonWorldPose& FlatMotionMatcher::flat_skeleton() const {
    return flat_skeleton_;
}

bool FlatMotionMatcher::uses_native_g1() const {
    return !uses_flat_controller_skeleton_;
}

float FlatMotionMatcher::planar_speed() const {
    const vec3 velocity = snapshot_.pose.velocities[0];
    return std::sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
}

}  // namespace episode
