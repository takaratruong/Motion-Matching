#include "g1_flat_motion_matcher.h"

#include "database.h"
#include "g1_skeleton.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace episode {
namespace {

constexpr float kStep = 1.0F / 25.0F;

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

}  // namespace

FlatMotionMatcher::FlatMotionMatcher(
    const std::filesystem::path& database_path) {
    database source{};
    database_load(source, database_path.string().c_str());
    prepare(source);
}

FlatMotionMatcher::~FlatMotionMatcher() = default;

void FlatMotionMatcher::prepare(const database& input) {
    database source{};
    source = input;
    if (source.nbones() != g1_skeleton::BoneCount ||
        source.nframes() < 27 || source.nranges() < 1) {
        throw std::invalid_argument("invalid flat G1 motion database");
    }
    for (int bone = 0; bone < source.nbones(); ++bone) {
        if (source.bone_parents(bone) !=
            g1_skeleton::kParents[static_cast<size_t>(bone)]) {
            throw std::invalid_argument("flat G1 skeleton mismatch");
        }
    }
    source.terrain_features.resize(source.nframes(), 4);
    source.terrain_features.zero();
    database_build_matching_features(
        source, 0.75F, 1.0F, 1.0F, 1.0F, 1.5F,
        g1_skeleton::LeftToe, g1_skeleton::RightToe,
        g1_skeleton::Hips, 0.0F);
    if (source.nfeatures() != 31) {
        throw std::invalid_argument("flat G1 feature build failed");
    }
    const interaction::Pose live = snapshot_.pose;
    const bool replacing = database_ != nullptr;
    database_ = std::make_unique<database>();
    *database_ = source;
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
        world_from_source_ = {vec3(), quat()};
        snapshot_.pose = mapped_pose(source_frame_);
        transition_source_ = snapshot_.pose;
        transition_seconds_ = 0.20F;
    }
}

interaction::Pose FlatMotionMatcher::mapped_pose(int frame) const {
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
    transition_seconds_ += kStep;
    if (transition_seconds_ < 0.20F) {
        const float alpha = smoothstep(transition_seconds_ / 0.20F);
        const vec3 root_position = target.positions[0];
        const quat root_rotation = target.rotations[0];
        target = interaction::interpolate_pose(
            transition_source_, target, alpha);
        target.positions[0] = root_position;
        target.rotations[0] = root_rotation;
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

const interaction::LocomotionSnapshot&
FlatMotionMatcher::snapshot() const {
    return snapshot_;
}

float FlatMotionMatcher::planar_speed() const {
    const vec3 velocity = snapshot_.pose.velocities[0];
    return std::sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
}

}  // namespace episode
