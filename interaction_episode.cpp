#include "interaction_episode.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace episode {
namespace {

constexpr float kFixedTick = 1.0F / 25.0F;
constexpr float kHeadingConvergenceDistance = 0.35F;
constexpr float kMaximumApproachSpeed = 1.20F;

float smoothstep(float value) {
    const float alpha = clampf(value, 0.0F, 1.0F);
    return alpha * alpha * (3.0F - 2.0F * alpha);
}

float planar_distance(vec3 left, vec3 right) {
    return std::hypot(left.x - right.x, left.z - right.z);
}

float planar_heading_error(quat left, quat right) {
    vec3 left_forward = quat_mul_vec3(left, vec3(0.0F, 0.0F, 1.0F));
    vec3 right_forward = quat_mul_vec3(right, vec3(0.0F, 0.0F, 1.0F));
    left_forward.y = 0.0F;
    right_forward.y = 0.0F;
    if (length(left_forward) < 1.0e-5F ||
        length(right_forward) < 1.0e-5F) {
        return 3.14159265F;
    }
    left_forward = normalize(left_forward);
    right_forward = normalize(right_forward);
    return std::acos(clampf(dot(left_forward, right_forward), -1.0F, 1.0F));
}

quat heading_from_velocity(vec3 velocity, quat fallback) {
    velocity.y = 0.0F;
    if (length(velocity) < 1.0e-5F) return fallback;
    const float yaw = std::atan2(velocity.x, velocity.z);
    return quat_from_angle_axis(yaw, vec3(0.0F, 1.0F, 0.0F));
}

interaction::Hand interaction_hand(reach::Hand hand) {
    return hand == reach::Hand::Left
        ? interaction::Hand::Left
        : interaction::Hand::Right;
}

size_t wrist_bone(interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
}

void validate_config(const EpisodeConfig& config) {
    if (!(config.entry_position_m >= 0.0F) ||
        !(config.entry_yaw_radians >= 0.0F) ||
        config.stable_entry_ticks <= 0 ||
        !(config.approach_timeout_seconds > 0.0F) ||
        !(config.bridge_seconds > 0.0F) ||
        !(config.carry_blend_seconds > 0.0F)) {
        throw std::invalid_argument("invalid interaction episode config");
    }
}

}  // namespace

InteractionEpisode::InteractionEpisode(
    std::filesystem::path walking_database,
    std::filesystem::path carry_left_database,
    std::filesystem::path carry_right_database,
    EpisodeConfig config)
    : walking_database_(std::move(walking_database)),
      carry_left_database_(std::move(carry_left_database)),
      carry_right_database_(std::move(carry_right_database)),
      config_(config),
      matcher_(walking_database_, carry_left_database_) {
    validate_config(config_);
    output_.pose = matcher_.snapshot().pose;
    output_.state = state_;
}

bool InteractionEpisode::commit(FrozenAttempt attempt) {
    if (state_ != EpisodeState::FreeLocomotion ||
        attempt.request_id == 0U ||
        attempt.object.generation == 0U ||
        attempt.plan.reach.poses.empty() ||
        attempt.object.dimensions.x <= 0.0F ||
        attempt.object.dimensions.y <= 0.0F ||
        attempt.object.dimensions.z <= 0.0F) {
        return false;
    }

    attempt.plan.grasp = attempt.grasp;
    attempt_ = std::move(attempt);
    output_.selected_hand = interaction_hand(attempt_->plan.hand);
    output_.object_world = attempt_->object.world;
    output_.attached = false;
    output_.failure.clear();

    interaction::GraspAffordance affordance{};
    affordance.id = 1U;
    affordance.hand = output_.selected_hand;
    affordance.hand_in_object = interaction::compose(
        interaction::inverse(attempt_->object.world),
        attempt_->grasp.hand_world);
    affordance.approach_direction_object = quat_inv_mul_vec3(
        attempt_->object.world.rotation,
        attempt_->grasp.approach_world);
    affordance.clearance_radius = 0.04F;

    interaction::InteractionTarget target{};
    target.handle = {1U, 1U};
    target.object_world = attempt_->object.world;
    target.object_profile_id = 1U;
    target.object_bounds = {
        vec3(),
        attempt_->object.dimensions * 0.5F,
    };
    target.object_dimensions = attempt_->object.dimensions;
    target.table_world = {
        vec3(0.0F, 0.65F, 0.0F),
        quat(),
    };
    target.table_size = vec3(1.20F, 0.06F, 0.75F);
    target.affordances = {affordance};

    registry_ = interaction::TargetRegistry{};
    target_handle_ = registry_.upsert(target);
    if (!registry_.reserve(target_handle_, attempt_->request_id)) {
        attempt_.reset();
        return false;
    }
    const interaction::InteractionTarget* reserved =
        registry_.find(target_handle_);
    if (reserved == nullptr) {
        attempt_.reset();
        return false;
    }
    attachment_.emplace(registry_);
    const interaction::PickRequest request{
        target_handle_, affordance.id, attempt_->request_id};
    if (!attachment_->begin(
            *reserved,
            request,
            affordance,
            attempt_->object.world.position.y)) {
        attempt_.reset();
        attachment_.reset();
        return false;
    }

    state_ = EpisodeState::Approach;
    state_seconds_ = 0.0F;
    frame_accumulator_ = 0.0F;
    reach_frame_ = 0U;
    stable_entry_ticks_ = 0;
    output_.state = state_;
    return true;
}

const EpisodeOutput& InteractionEpisode::update(const EpisodeInput& input) {
    if (!std::isfinite(input.dt) || input.dt < 0.0F) {
        throw std::invalid_argument("episode dt must be finite and nonnegative");
    }
    if (input.reset_pressed) {
        reset();
        return output_;
    }
    const bool before_contact =
        state_ == EpisodeState::Approach ||
        state_ == EpisodeState::Bridge ||
        state_ == EpisodeState::Reach;
    if (before_contact && input.cancel_pressed) {
        cancel_before_contact();
        return output_;
    }
    if (before_contact && attempt_.has_value() &&
        input.current_object_generation != attempt_->object.generation) {
        fail("object changed before contact");
        return output_;
    }

    switch (state_) {
        case EpisodeState::FreeLocomotion:
            matcher_.update(input.command, input.dt);
            publish(matcher_.snapshot().pose);
            break;
        case EpisodeState::Searching:
            break;
        case EpisodeState::Approach:
            update_approach(input);
            break;
        case EpisodeState::Bridge:
            update_bridge(input.dt);
            break;
        case EpisodeState::Reach:
            update_reach(input.dt);
            break;
        case EpisodeState::CarryBlend:
        case EpisodeState::Carry:
            update_carry(input);
            break;
        case EpisodeState::Failed:
            break;
    }
    output_.state = state_;
    return output_;
}

void InteractionEpisode::update_approach(const EpisodeInput& input) {
    const interaction::Pose& live = matcher_.snapshot().pose;
    const vec3 root = live.positions[g1_skeleton::Simulation];
    const vec3 entry = attempt_->plan.entry_root_world.position;
    vec3 delta(entry.x - root.x, 0.0F, entry.z - root.z);
    const float distance = length(delta);
    const float speed = std::min(
        kMaximumApproachSpeed,
        distance / std::max(input.dt, kFixedTick));
    const vec3 velocity = distance > 1.0e-5F
        ? delta * (speed / distance)
        : vec3();
    const quat heading = distance <= kHeadingConvergenceDistance
        ? attempt_->plan.entry_root_world.rotation
        : heading_from_velocity(
              velocity,
              live.rotations[g1_skeleton::Simulation]);
    matcher_.update({velocity, heading}, input.dt);
    publish(matcher_.snapshot().pose);

    state_seconds_ += input.dt;
    const interaction::Pose& current = output_.pose;
    output_.approach_distance_m = planar_distance(
        current.positions[g1_skeleton::Simulation], entry);
    output_.approach_yaw_error_radians = planar_heading_error(
        current.rotations[g1_skeleton::Simulation],
        attempt_->plan.entry_root_world.rotation);
    output_.locomotion_speed_mps = matcher_.planar_speed();
    const bool settled =
        output_.approach_distance_m <= config_.entry_position_m &&
        output_.approach_yaw_error_radians <=
            config_.entry_yaw_radians;
    stable_entry_ticks_ = settled ? stable_entry_ticks_ + 1 : 0;
    if (stable_entry_ticks_ >= config_.stable_entry_ticks) {
        bridge_start_ = output_.pose;
        bridge_flat_start_ = matcher_.flat_skeleton();
        state_ = EpisodeState::Bridge;
        state_seconds_ = 0.0F;
        frame_accumulator_ = 0.0F;
        return;
    }
    if (state_seconds_ >= config_.approach_timeout_seconds) {
        fail("approach timed out");
    }
}

void InteractionEpisode::update_bridge(float dt) {
    state_seconds_ += dt;
    const float alpha = smoothstep(
        state_seconds_ / config_.bridge_seconds);
    publish(interaction::interpolate_pose(
        bridge_start_,
        attempt_->plan.reach.poses.front(),
        alpha));
    output_.flat_locomotion = bridge_flat_start_;
    output_.bridge_alpha = alpha;
    if (state_seconds_ >= config_.bridge_seconds) {
        state_ = EpisodeState::Reach;
        state_seconds_ = 0.0F;
        frame_accumulator_ = 0.0F;
        reach_frame_ = 0U;
        publish(attempt_->plan.reach.poses.front());
    }
}

void InteractionEpisode::update_reach(float dt) {
    frame_accumulator_ += dt;
    while (frame_accumulator_ >= kFixedTick &&
           reach_frame_ + 1U < attempt_->plan.reach.poses.size()) {
        frame_accumulator_ -= kFixedTick;
        ++reach_frame_;
    }
    publish(attempt_->plan.reach.poses[reach_frame_]);
    if (reach_frame_ + 1U != attempt_->plan.reach.poses.size()) return;

    contact_pose_ = output_.pose;
    if (!attach_at_contact()) {
        fail("contact rejected");
        return;
    }
    matcher_.switch_database(
        output_.selected_hand == interaction::Hand::Left
            ? carry_left_database_
            : carry_right_database_,
        contact_pose_);
    state_ = EpisodeState::CarryBlend;
    state_seconds_ = 0.0F;
}

void InteractionEpisode::update_carry(const EpisodeInput& input) {
    matcher_.update(input.command, input.dt);
    interaction::Pose displayed = matcher_.snapshot().pose;
    if (state_ == EpisodeState::CarryBlend) {
        state_seconds_ += input.dt;
        const float alpha = smoothstep(
            state_seconds_ / config_.carry_blend_seconds);
        displayed = interaction::interpolate_pose(
            contact_pose_, displayed, alpha);
        if (state_seconds_ >= config_.carry_blend_seconds) {
            state_ = EpisodeState::Carry;
        }
    }
    publish(displayed);
    attachment_->update(hand_measurement(displayed, false), input.dt);
    output_.object_world = attachment_->object_world();
    output_.attached =
        attachment_->state() == interaction::ObjectState::Attached ||
        attachment_->state() == interaction::ObjectState::Held;
    if (!output_.attached) {
        fail("attachment lost during carry");
    }
}

void InteractionEpisode::publish(interaction::Pose pose) {
    output_.pose = std::move(pose);
    output_.state = state_;
    output_.flat_locomotion =
        state_ == EpisodeState::FreeLocomotion ||
                state_ == EpisodeState::Approach
            ? matcher_.flat_skeleton()
            : FlatSkeletonWorldPose{};
    if (state_ != EpisodeState::Bridge) output_.bridge_alpha = 0.0F;
}

void InteractionEpisode::fail(std::string reason) {
    state_ = EpisodeState::Failed;
    output_.state = state_;
    output_.failure = std::move(reason);
}

void InteractionEpisode::cancel_before_contact() {
    if (attempt_.has_value() && target_handle_.id != 0U) {
        registry_.release(target_handle_, attempt_->request_id);
    }
    attachment_.reset();
    attempt_.reset();
    target_handle_ = {};
    state_ = EpisodeState::FreeLocomotion;
    output_.state = state_;
    output_.attached = false;
    output_.failure.clear();
    publish(matcher_.snapshot().pose);
}

bool InteractionEpisode::attach_at_contact() {
    const interaction::ContactMeasurement measurement =
        hand_measurement(output_.pose, true);
    if (!attachment_->try_contact(measurement)) return false;
    output_.object_world = attachment_->object_world();
    output_.attached = true;
    return true;
}

interaction::ContactMeasurement InteractionEpisode::hand_measurement(
    const interaction::Pose& pose,
    bool contact_event) const {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t wrist = wrist_bone(output_.selected_hand);
    const interaction::Transform hand_world{
        world.positions[wrist],
        world.rotations[wrist],
    };
    return {
        target_handle_,
        output_.selected_hand,
        hand_world,
        length(hand_world.position - attempt_->grasp.hand_world.position),
        quat_angle_between(
            hand_world.rotation,
            attempt_->grasp.hand_world.rotation),
        contact_event,
        true,
        true,
        true,
    };
}

void InteractionEpisode::reset() {
    matcher_.reset_database(walking_database_);
    registry_ = interaction::TargetRegistry{};
    attachment_.reset();
    attempt_.reset();
    target_handle_ = {};
    state_ = EpisodeState::FreeLocomotion;
    state_seconds_ = 0.0F;
    frame_accumulator_ = 0.0F;
    reach_frame_ = 0U;
    stable_entry_ticks_ = 0;
    output_ = EpisodeOutput{};
    output_.pose = matcher_.snapshot().pose;
    output_.state = state_;
}

EpisodeState InteractionEpisode::state() const {
    return state_;
}

const std::optional<FrozenAttempt>& InteractionEpisode::attempt() const {
    return attempt_;
}

const EpisodeOutput& InteractionEpisode::output() const {
    return output_;
}

}  // namespace episode
