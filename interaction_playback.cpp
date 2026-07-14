#include "interaction_playback.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace interaction {
namespace {

constexpr double kCanonicalFps = 25.0;
constexpr float kMinimumSpeed = 0.85F;
constexpr float kMaximumSpeed = 1.15F;
constexpr float kContinuityTolerance = 1.0e-4F;
constexpr double kFrameReadTolerance = 1.0e-5;

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) &&
           finite(value.y) && finite(value.z);
}

bool valid_rotation(quat value) {
    return finite(value) && quat_length(value) > 1.0e-6F;
}

float yaw_radians(quat rotation) {
    const vec3 facing = quat_mul_vec3(
        rotation, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(facing.x, facing.z);
}

float shortest_angle(float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

bool near(float left, float right) {
    return std::abs(left - right) <= kContinuityTolerance;
}

size_t checked_product(size_t left, size_t right) {
    if (right != 0U &&
        left > std::numeric_limits<size_t>::max() / right) {
        throw std::invalid_argument("interaction playback database overflow");
    }
    return left * right;
}

void validate_pose_storage(const Database& database, int32_t stop) {
    const size_t frames = static_cast<size_t>(stop);
    const size_t bones = g1_skeleton::BoneCount;
    const size_t bone_values = checked_product(frames, bones);
    if (database.positions.size() < checked_product(bone_values, 3U) ||
        database.velocities.size() < checked_product(bone_values, 3U) ||
        database.rotations.size() < checked_product(bone_values, 4U) ||
        database.angular_velocities.size() <
            checked_product(bone_values, 3U) ||
        database.hand_dof.size() < checked_product(frames, 14U) ||
        database.hand_dof_velocities.size() <
            checked_product(frames, 14U) ||
        database.foot_contacts.size() < checked_product(frames, 2U)) {
        throw std::invalid_argument(
            "interaction playback pose arrays do not cover clip range");
    }
}

void validate_candidate_frames(
    const Database& database,
    const MatchCandidate& candidate,
    int32_t start,
    int32_t stop) {
    if (!(start <= candidate.entry_frame &&
          candidate.entry_frame < candidate.contact_frame &&
          candidate.contact_frame < candidate.lift_frame &&
          candidate.lift_frame < candidate.hold_frame &&
          candidate.hold_frame < stop)) {
        throw std::invalid_argument(
            "interaction playback candidate frames are out of order");
    }
    if (database.phases.size() < static_cast<size_t>(stop)) {
        throw std::invalid_argument(
            "interaction playback phases do not cover clip range");
    }

    uint8_t previous = database.phases[static_cast<size_t>(
        candidate.entry_frame)];
    if (previous > static_cast<uint8_t>(Phase::Reach)) {
        throw std::invalid_argument(
            "interaction playback entry is not pre-contact");
    }
    for (int32_t source = candidate.entry_frame + 1;
         source < stop;
         ++source) {
        const uint8_t current = database.phases[static_cast<size_t>(source)];
        if (current > static_cast<uint8_t>(Phase::Hold) ||
            current < previous) {
            throw std::invalid_argument(
                "interaction playback phases are not ordered");
        }
        previous = current;
    }
    if (database.phases[static_cast<size_t>(candidate.contact_frame)] !=
            static_cast<uint8_t>(Phase::Contact) ||
        database.phases[static_cast<size_t>(candidate.lift_frame)] !=
            static_cast<uint8_t>(Phase::Lift) ||
        database.phases[static_cast<size_t>(candidate.hold_frame)] !=
            static_cast<uint8_t>(Phase::Hold) ||
        database.phases[static_cast<size_t>(candidate.contact_frame - 1)] >=
            static_cast<uint8_t>(Phase::Contact) ||
        database.phases[static_cast<size_t>(candidate.lift_frame - 1)] >=
            static_cast<uint8_t>(Phase::Lift) ||
        database.phases[static_cast<size_t>(candidate.hold_frame - 1)] >=
            static_cast<uint8_t>(Phase::Hold)) {
        throw std::invalid_argument(
            "interaction playback candidate events do not match phases");
    }
}

float entry_correction_weight(
    const MatchCandidate& candidate,
    double source_frame) {
    if (source_frame <= static_cast<double>(candidate.entry_frame)) {
        return 1.0F;
    }
    if (source_frame >= static_cast<double>(candidate.contact_frame)) {
        return 0.0F;
    }
    const double alpha =
        (source_frame - static_cast<double>(candidate.entry_frame)) /
        static_cast<double>(
            candidate.contact_frame - candidate.entry_frame);
    const double smoothstep = alpha * alpha * (3.0 - 2.0 * alpha);
    return static_cast<float>(1.0 - smoothstep);
}

double stable_read_frame(double source_frame) {
    const double nearest = std::round(source_frame);
    return std::abs(source_frame - nearest) <= kFrameReadTolerance
        ? nearest
        : source_frame;
}

}  // namespace

SequentialPlayer::SequentialPlayer(const Database& database)
    : database_(&database) {}

void SequentialPlayer::start(
    const MatchCandidate& candidate,
    const Pose& current,
    float speed) {
    if (!finite(speed) || speed < kMinimumSpeed || speed > kMaximumSpeed) {
        throw std::invalid_argument(
            "interaction playback speed must be in [0.85, 1.15]");
    }
    if (candidate.clip < 0 ||
        static_cast<uint32_t>(candidate.clip) >= database_->clip_count) {
        throw std::invalid_argument(
            "interaction playback candidate clip is invalid");
    }
    const size_t clip = static_cast<size_t>(candidate.clip);
    if (clip >= database_->range_starts.size() ||
        clip >= database_->range_stops.size()) {
        throw std::invalid_argument(
            "interaction playback candidate range is missing");
    }
    const int32_t start_frame = database_->range_starts[clip];
    const int32_t stop_frame = database_->range_stops[clip];
    if (database_->fps_numerator != 25U ||
        database_->fps_denominator != 1U ||
        start_frame < 0 || stop_frame <= start_frame ||
        static_cast<uint32_t>(stop_frame) > database_->frame_count) {
        throw std::invalid_argument(
            "interaction playback clip range is invalid");
    }
    validate_candidate_frames(
        *database_, candidate, start_frame, stop_frame);
    validate_pose_storage(*database_, stop_frame);

    if (!finite(candidate.scene_from_source.position) ||
        !valid_rotation(candidate.scene_from_source.rotation) ||
        !finite(candidate.entry_root_offset) ||
        !near(candidate.entry_root_offset.y, 0.0F) ||
        !finite(candidate.entry_yaw_offset)) {
        throw std::invalid_argument(
            "interaction playback candidate correction is invalid");
    }

    const size_t root = g1_skeleton::Simulation;
    if (!finite(current.positions[root]) ||
        !valid_rotation(current.rotations[root])) {
        throw std::invalid_argument(
            "interaction playback current root is invalid");
    }
    const Pose entry_pose = pose_at_frame(*database_, candidate.entry_frame);
    if (!finite(entry_pose.positions[root]) ||
        !valid_rotation(entry_pose.rotations[root])) {
        throw std::invalid_argument(
            "interaction playback source root is invalid");
    }
    const Transform mapped_root = compose(
        candidate.scene_from_source,
        Transform{
            entry_pose.positions[root],
            entry_pose.rotations[root],
        });
    const vec3 expected_offset(
        current.positions[root].x - mapped_root.position.x,
        0.0F,
        current.positions[root].z - mapped_root.position.z);
    const float expected_yaw = shortest_angle(
        yaw_radians(current.rotations[root]) -
        yaw_radians(mapped_root.rotation));
    if (!near(candidate.entry_root_offset.x, expected_offset.x) ||
        !near(candidate.entry_root_offset.z, expected_offset.z) ||
        !near(shortest_angle(
                  candidate.entry_yaw_offset - expected_yaw),
              0.0F)) {
        throw std::invalid_argument(
            "interaction playback candidate does not match current root");
    }

    candidate_ = candidate;
    source_frame_exact_ = static_cast<double>(candidate.entry_frame);
    elapsed_exact_ = 0.0;
    source_frame_ = static_cast<float>(source_frame_exact_);
    elapsed_ = 0.0F;
    speed_ = speed;
    final_frame_ = stop_frame - 1;
    started_ = true;
}

void SequentialPlayer::advance(float dt) {
    if (!finite(dt) || dt < 0.0F) {
        throw std::invalid_argument(
            "interaction playback dt must be finite and nonnegative");
    }
    if (!started_) {
        throw std::logic_error("interaction playback has not started");
    }

    const double next_elapsed = elapsed_exact_ + static_cast<double>(dt);
    if (!std::isfinite(next_elapsed)) {
        throw std::invalid_argument("interaction playback elapsed overflow");
    }
    const double advance_frames =
        static_cast<double>(dt) * kCanonicalFps *
        static_cast<double>(speed_);
    const double remaining =
        static_cast<double>(final_frame_) - source_frame_exact_;
    if (advance_frames >= remaining) {
        source_frame_exact_ = static_cast<double>(final_frame_);
    } else {
        source_frame_exact_ += advance_frames;
    }
    elapsed_exact_ = next_elapsed;
    source_frame_ = static_cast<float>(source_frame_exact_);
    elapsed_ = static_cast<float>(elapsed_exact_);
}

Pose SequentialPlayer::sample() const {
    if (!started_) {
        throw std::logic_error("interaction playback has not started");
    }
    const double source_frame = source_frame_exact_;
    const int32_t left = static_cast<int32_t>(std::floor(source_frame));
    const int32_t right = std::min(left + 1, final_frame_);
    const float alpha = static_cast<float>(
        source_frame - static_cast<double>(left));
    Pose pose = interpolate_pose(
        pose_at_frame(*database_, left),
        pose_at_frame(*database_, right),
        alpha);

    const size_t root = g1_skeleton::Simulation;
    const Transform mapped_root = compose(
        candidate_.scene_from_source,
        Transform{pose.positions[root], pose.rotations[root]});
    const float weight = entry_correction_weight(
        candidate_, source_frame);
    const quat yaw = quat_from_angle_axis(
        weight * candidate_.entry_yaw_offset,
        vec3(0.0F, 1.0F, 0.0F));
    pose.positions[root] =
        mapped_root.position + weight * candidate_.entry_root_offset;
    pose.rotations[root] = quat_normalize(
        quat_mul(yaw, mapped_root.rotation));
    return pose;
}

Phase SequentialPlayer::phase() const {
    if (!started_) {
        throw std::logic_error("interaction playback has not started");
    }
    return static_cast<Phase>(
        database_->phases[static_cast<size_t>(frame())]);
}

int32_t SequentialPlayer::frame() const {
    if (!started_) return -1;
    return static_cast<int32_t>(
        std::floor(stable_read_frame(source_frame_exact_)));
}

bool SequentialPlayer::at_contact() const {
    return started_ && frame() >= candidate_.contact_frame;
}

bool SequentialPlayer::at_hold() const {
    return started_ && frame() >= candidate_.hold_frame;
}

bool SequentialPlayer::finished() const {
    return !started_ ||
        stable_read_frame(source_frame_exact_) >= final_frame_;
}

float SequentialPlayer::elapsed_seconds() const {
    return elapsed_;
}

vec3 SequentialPlayer::entry_root_correction() const {
    if (!started_) return vec3();
    return entry_correction_weight(candidate_, source_frame_exact_) *
        candidate_.entry_root_offset;
}

}  // namespace interaction
